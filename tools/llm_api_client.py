"""OpenAI-compatible LLM/VLM client for MemPilot's CURATE delegation.

The selected backend processes the supplied instruction and evidence, with
image pixels attached only when requested and supported by the route.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import socket
import threading
from collections.abc import Mapping
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, OpenAIError


EXTERNAL_CALL_FAILED = "External tool call failed."
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
DEFAULT_IMAGE_MAX_EDGE_PIXELS = 2048
DEFAULT_IMAGE_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_IMAGE_JPEG_QUALITY = 90
logger = logging.getLogger(__name__)

_KEY_POOLS: dict[str, "RoundRobinKeyPool"] = {}
_KEY_POOLS_LOCK = threading.Lock()
_CLIENTS: dict[str, AsyncOpenAI] = {}
_CLIENTS_LOCK = threading.Lock()


class RoundRobinKeyPool:
    """Thread-safe round-robin selector shared by calls in one worker."""

    def __init__(self, keys: list[str], start_index: int | None = None):
        if not keys:
            raise ValueError("API key pool is empty")
        self.keys = keys
        self.index = _worker_start_index(len(keys)) if start_index is None else int(start_index) % len(keys)
        self.lock = threading.Lock()

    def next_key(self) -> str:
        with self.lock:
            key = self.keys[self.index % len(self.keys)]
            self.index += 1
            return key


async def call_openai_compatible_chat(
    config: dict[str, Any],
    model_spec: dict[str, Any],
    backend_input: str,
    image_inputs: list[Any] | None = None,
) -> tuple[str, dict[str, int | float]]:
    """Call the provider selected by a model route and return exact API usage."""
    try:
        return await _call_openai_compatible_chat(config, model_spec, backend_input, image_inputs)
    except FileNotFoundError as exc:
        # A configured credential path is a deployment error, not a transient API failure.
        _log_external_call_error(model_spec, exc)
        raise
    except Exception as exc:
        _log_external_call_error(model_spec, exc)
        return EXTERNAL_CALL_FAILED, dict(ZERO_USAGE)


async def _call_openai_compatible_chat(
    config: dict[str, Any],
    model_spec: dict[str, Any],
    backend_input: str,
    image_inputs: list[Any] | None,
) -> tuple[str, dict[str, int | float]]:
    model = str(model_spec.get("model") or "").strip()
    api_base = str(model_spec.get("api_base") or "").strip()
    if not api_base or not model:
        missing = "api_base" if not api_base else "model"
        _log_external_call_error(model_spec, ValueError(f"Missing required route setting: {missing}"))
        return EXTERNAL_CALL_FAILED, dict(ZERO_USAGE)

    key_pool = _get_key_pool(model_spec)
    if key_pool is None:
        _log_external_call_error(model_spec, ValueError("No API key is configured for this route"))
        return EXTERNAL_CALL_FAILED, dict(ZERO_USAGE)

    max_trials = max(int(_call_setting(config, model_spec, "max_trials", 3)), 1)
    retry_sleep = max(float(_call_setting(config, model_spec, "retry_sleep", 1.0)), 0.0)
    timeout = float(_call_setting(config, model_spec, "timeout", 60.0))
    user_content: str | list[dict[str, Any]] = backend_input
    if image_inputs:
        image_processing = _image_processing_settings(config, model_spec)
        user_content = await asyncio.to_thread(
            _build_user_content,
            backend_input,
            image_inputs,
            (config or {}).get("image_root"),
            image_processing,
        )
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a memory-processing assistant. Follow the instruction using only the question-image "
                    "descriptions, memory items, and images in the input. Return all findings supported by these "
                    "inputs. If the "
                    "instruction can only be partially completed, return the supported findings and briefly state "
                    "what remains unresolved. If nothing relevant is present, say so directly. Do not add unsupported "
                    "information."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "temperature": float(_call_setting(config, model_spec, "temperature", 0.0)),
        "max_tokens": int(_call_setting(config, model_spec, "max_tokens", 512)),
    }
    top_p = _optional_call_setting(config, model_spec, "top_p")
    seed = _optional_call_setting(config, model_spec, "seed")
    if top_p is not None:
        request["top_p"] = float(top_p)
    if seed is not None:
        request["seed"] = int(seed)
    extra_body = _optional_call_setting(config, model_spec, "extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be a mapping when configured.")
        request["extra_body"] = extra_body

    for attempt in range(max_trials):
        api_key = key_pool.next_key()
        client = _get_client(model_spec, api_base, api_key, timeout)
        try:
            response = await client.chat.completions.create(**request)
            content = _extract_chat_completion_text(response)
            usage = _normalize_api_usage(getattr(response, "usage", None))
            return (content or EXTERNAL_CALL_FAILED), usage
        except APIStatusError as exc:
            _log_external_call_error(
                model_spec,
                exc,
                attempt=attempt + 1,
                max_trials=max_trials,
                secret=api_key,
            )
            if exc.status_code in {400, 404, 405, 413, 422}:
                break
        except (APIConnectionError, APITimeoutError, OpenAIError) as exc:
            _log_external_call_error(
                model_spec,
                exc,
                attempt=attempt + 1,
                max_trials=max_trials,
                secret=api_key,
            )

        if attempt + 1 < max_trials and retry_sleep:
            await asyncio.sleep(retry_sleep)

    return EXTERNAL_CALL_FAILED, dict(ZERO_USAGE)


def _log_external_call_error(
    model_spec: dict[str, Any],
    exc: Exception,
    *,
    attempt: int | None = None,
    max_trials: int | None = None,
    secret: str = "",
) -> None:
    """Emit a conspicuous, credential-safe API failure without changing tool output."""
    provider = str(model_spec.get("provider") or "unknown")
    model = str(model_spec.get("model") or "unknown")
    status_code = getattr(exc, "status_code", None)
    attempt_text = f" attempt={attempt}/{max_trials}" if attempt is not None else ""
    status_text = f" status={status_code}" if status_code is not None else ""
    detail = _sanitize_error_detail(exc, secret=secret)
    logger.error(
        "\n%s\n[RUNTIME MEMORY API ERROR] provider=%s model=%s%s%s error=%s\n"
        "reason=%s\n%s",
        "!" * 80,
        provider,
        model,
        attempt_text,
        status_text,
        type(exc).__name__,
        detail,
        "!" * 80,
    )


def _sanitize_error_detail(exc: Exception, *, secret: str = "", max_chars: int = 1000) -> str:
    detail = " ".join(str(exc).split()) or "No error message was provided."
    if secret:
        detail = detail.replace(secret, "[REDACTED]")
    detail = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", detail)
    detail = re.sub(
        r"(?i)\b(api[_-]?key|authorization)(\s*[:=]\s*)([^\s,;}]+)",
        r"\1\2[REDACTED]",
        detail,
    )
    detail = re.sub(r"\b(?:sk|tgp)[-_][A-Za-z0-9_-]{8,}\b", "[REDACTED]", detail)
    return detail if len(detail) <= max_chars else detail[: max_chars - 3] + "..."


def _image_processing_settings(
    config: Mapping[str, Any] | None,
    model_spec: Mapping[str, Any] | None,
) -> dict[str, int]:
    shared = (config or {}).get("image_processing") or {}
    route = (model_spec or {}).get("image_processing") or {}
    if not isinstance(shared, Mapping) or not isinstance(route, Mapping):
        raise ValueError("image_processing must be a mapping.")
    return _normalize_image_processing_settings({**shared, **route})


def _normalize_image_processing_settings(
    raw: Mapping[str, Any] | None,
) -> dict[str, int]:
    raw = raw or {}
    if not isinstance(raw, Mapping):
        raise ValueError("image_processing must be a mapping.")
    settings = {
        "max_edge_pixels": int(raw.get("max_edge_pixels", DEFAULT_IMAGE_MAX_EDGE_PIXELS)),
        "max_bytes": int(raw.get("max_bytes", DEFAULT_IMAGE_MAX_BYTES)),
        "jpeg_quality": int(raw.get("jpeg_quality", DEFAULT_IMAGE_JPEG_QUALITY)),
    }
    if settings["max_edge_pixels"] <= 0 or settings["max_bytes"] <= 0:
        raise ValueError("image_processing size limits must be positive.")
    if not 1 <= settings["jpeg_quality"] <= 100:
        raise ValueError("image_processing.jpeg_quality must be between 1 and 100.")
    return settings


def _call_setting(config: dict[str, Any], model_spec: dict[str, Any], name: str, default: Any) -> Any:
    if name in model_spec:
        return model_spec[name]
    return ((config or {}).get("external_call", {}) or {}).get(name, default)


def _optional_call_setting(config: dict[str, Any], model_spec: dict[str, Any], name: str) -> Any:
    if name in model_spec:
        return model_spec[name]
    return ((config or {}).get("external_call", {}) or {}).get(name)


def _get_client(model_spec: dict[str, Any], api_base: str, api_key: str, timeout: float) -> AsyncOpenAI:
    headers = _provider_headers(model_spec)
    identity = {
        "event_loop": id(asyncio.get_running_loop()),
        "api_base": api_base.rstrip("/"),
        "api_key": _secret_fingerprint(api_key),
        "timeout": timeout,
        "headers": headers,
    }
    cache_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(cache_key)
        if client is None:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base.rstrip("/") + "/",
                timeout=timeout,
                max_retries=0,
                default_headers=headers or None,
            )
            _CLIENTS[cache_key] = client
        return client


def _build_user_content(
    backend_input: str,
    image_inputs: list[Any] | None,
    image_root: str | Path | None = None,
    image_processing: Mapping[str, Any] | None = None,
) -> str | list[dict[str, Any]]:
    if not image_inputs:
        return backend_input
    image_settings = _normalize_image_processing_settings(image_processing)
    content: list[dict[str, Any]] = [{"type": "text", "text": backend_input}]
    for image in image_inputs:
        if isinstance(image, dict):
            image_id = str(image.get("id") or image.get("image_id") or "").strip()
            if image_id:
                content.append({"type": "text", "text": f"Image input: {image_id}"})
        content.append(
            _normalize_image_content(
                image,
                image_root=image_root,
                image_processing=image_settings,
            )
        )
    return content


def _normalize_image_content(
    image: Any,
    *,
    image_root: str | Path | None = None,
    image_processing: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    detail = None
    if isinstance(image, str):
        url = image.strip()
    elif isinstance(image, dict):
        detail = image.get("detail")
        image_url = image.get("image_url")
        if isinstance(image_url, dict):
            url = str(image_url.get("url") or "").strip()
            detail = image_url.get("detail", detail)
        elif image_url:
            url = str(image_url).strip()
        else:
            url = str(image.get("url") or image.get("data_url") or "").strip()
            if not url:
                encoded = image.get("base64") or image.get("data")
                mime_type = str(image.get("mime_type") or "image/jpeg")
                if encoded:
                    url = f"data:{mime_type};base64,{encoded}"
            if not url:
                url = str(image.get("path") or image.get("image_path") or "").strip()
    else:
        url = ""

    if not url:
        raise ValueError("Each image input must contain a local path, URL, data URL, or base64 payload.")
    url = _resolve_image_url(
        url,
        image if isinstance(image, dict) else None,
        image_root=image_root,
        image_processing=image_processing,
    )
    payload: dict[str, Any] = {"url": url}
    if detail in {"auto", "low", "high"}:
        payload["detail"] = detail
    return {"type": "image_url", "image_url": payload}


def _resolve_image_url(
    value: str,
    image_reference: dict[str, Any] | None = None,
    *,
    image_root: str | Path | None = None,
    image_processing: Mapping[str, int] | None = None,
) -> str:
    """Resolve a local/portable image reference and return an API-compatible data URL."""
    value = str(value or "").strip()
    image_settings = _normalize_image_processing_settings(image_processing)
    if value.startswith("data:"):
        return _prepare_data_url_image(value, image_settings)
    if value.startswith(("https://", "http://")):
        return value

    path = _find_local_image(value, image_root=image_root)
    if path is None and image_reference:
        repo_id = str(image_reference.get("hf_repo_id") or "").strip()
        filename = str(image_reference.get("hf_filename") or "").strip()
        if repo_id and filename:
            path = Path(
                _download_hf_image(
                    repo_id,
                    filename,
                    str(image_reference.get("hf_repo_type") or "dataset").strip(),
                    str(image_reference.get("hf_revision") or "").strip(),
                )
            )
    if path is None:
        raise ValueError(
            f"Image path is unavailable: {value}. Configure image_root or provide a portable HF reference."
        )

    stat = path.stat()
    return _encode_local_image(
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        image_settings["max_edge_pixels"],
        image_settings["max_bytes"],
        image_settings["jpeg_quality"],
    )


def _find_local_image(value: str, *, image_root: str | Path | None = None) -> Path | None:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path if path.is_file() else None

    roots = [Path(image_root).expanduser()] if image_root else []
    roots.append(Path.cwd())
    for root in roots:
        for candidate in (root / path, root / "data" / path):
            if candidate.is_file():
                return Path(os.path.abspath(candidate))
    return None


@lru_cache(maxsize=512)
def _download_hf_image(repo_id: str, filename: str, repo_type: str, revision: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install `huggingface_hub` to resolve portable image references.") from exc

    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": repo_type or "dataset",
    }
    if revision:
        kwargs["revision"] = revision
    try:
        return hf_hub_download(**kwargs, local_files_only=True)
    except Exception:
        return hf_hub_download(**kwargs)


@lru_cache(maxsize=32)
def _encode_local_image(
    path_text: str,
    mtime_ns: int,
    size: int,
    max_edge_pixels: int = DEFAULT_IMAGE_MAX_EDGE_PIXELS,
    max_bytes: int = DEFAULT_IMAGE_MAX_BYTES,
    jpeg_quality: int = DEFAULT_IMAGE_JPEG_QUALITY,
) -> str:
    del mtime_ns, size
    path = Path(path_text)
    image_bytes = path.read_bytes()
    mime_type = (
        _image_mime_type_from_signature(image_bytes)
        or mimetypes.guess_type(path.name)[0]
        or "image/jpeg"
    )
    image_bytes, mime_type, changed = _prepare_image_bytes(
        image_bytes,
        mime_type,
        max_edge_pixels=max_edge_pixels,
        max_bytes=max_bytes,
        jpeg_quality=jpeg_quality,
    )
    if changed:
        logger.info(
            "[RUNTIME MEMORY IMAGE RESIZED] path=%s output_bytes=%d max_edge_pixels=%d max_bytes=%d",
            path,
            len(image_bytes),
            max_edge_pixels,
            max_bytes,
        )
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _prepare_data_url_image(value: str, image_processing: Mapping[str, int]) -> str:
    corrected = _correct_data_url_mime_type(value)
    header, separator, encoded = corrected.partition(",")
    if not separator or ";base64" not in header.lower():
        return corrected
    try:
        image_bytes = base64.b64decode(encoded, validate=False)
    except (ValueError, TypeError):
        return corrected
    mime_type = header.split(";", 1)[0].removeprefix("data:") or "image/jpeg"
    image_bytes, mime_type, _ = _prepare_image_bytes(
        image_bytes,
        mime_type,
        max_edge_pixels=image_processing["max_edge_pixels"],
        max_bytes=image_processing["max_bytes"],
        jpeg_quality=image_processing["jpeg_quality"],
    )
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _prepare_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    *,
    max_edge_pixels: int,
    max_bytes: int,
    jpeg_quality: int,
) -> tuple[bytes, str, bool]:
    """Bound image dimensions and payload size before an OpenAI-compatible request."""

    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        raise ImportError("Install Pillow to prepare runtime-memory image inputs.") from exc

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            needs_conversion = mime_type not in {
                "image/jpeg",
                "image/png",
                "image/gif",
                "image/webp",
            }
            needs_resize = max(width, height) > max_edge_pixels
            needs_reencode = len(image_bytes) > max_bytes
            if not (needs_conversion or needs_resize or needs_reencode):
                return image_bytes, mime_type, False

            image = _flatten_image_to_rgb(image)
            scale = min(1.0, max_edge_pixels / max(image.size))
            target_size = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            if target_size != image.size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)

            quality = jpeg_quality
            for _ in range(24):
                output = BytesIO()
                image.save(output, format="JPEG", quality=quality, optimize=True)
                prepared = output.getvalue()
                if len(prepared) <= max_bytes:
                    return prepared, "image/jpeg", True
                if quality > 60:
                    quality = max(60, quality - 10)
                    continue
                next_size = (
                    max(1, round(image.width * 0.8)),
                    max(1, round(image.height * 0.8)),
                )
                if next_size == image.size:
                    break
                image = image.resize(next_size, Image.Resampling.LANCZOS)
                quality = jpeg_quality
    except (UnidentifiedImageError, OSError) as exc:
        if len(image_bytes) <= max_bytes:
            return image_bytes, mime_type, False
        raise ValueError(
            f"Image payload is {len(image_bytes)} bytes and cannot be decoded for resizing."
        ) from exc

    raise ValueError(f"Unable to reduce image payload below {max_bytes} bytes.")


def _flatten_image_to_rgb(image: Any) -> Any:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        from PIL import Image

        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def _correct_data_url_mime_type(value: str) -> str:
    """Replace a mislabeled base64 image MIME type using its byte signature."""
    header, separator, encoded = value.partition(",")
    if not separator or ";base64" not in header.lower():
        return value
    try:
        image_bytes = base64.b64decode(encoded, validate=False)
    except (ValueError, TypeError):
        return value
    mime_type = _image_mime_type_from_signature(image_bytes)
    if not mime_type:
        return value
    header_parts = header.split(";")
    header_parts[0] = f"data:{mime_type}"
    return ";".join(header_parts) + separator + encoded


def _image_mime_type_from_signature(image_bytes: bytes) -> str | None:
    """Recognize image formats accepted by OpenAI-compatible multimodal APIs."""
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (
        len(image_bytes) >= 12
        and image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return None


def _get_key_pool(model_spec: dict[str, Any]) -> RoundRobinKeyPool | None:
    cache_key = _key_pool_cache_key(model_spec)
    with _KEY_POOLS_LOCK:
        pool = _KEY_POOLS.get(cache_key)
        if pool is None:
            keys = _load_api_keys(model_spec)
            if not keys:
                return None
            pool = RoundRobinKeyPool(keys)
            _KEY_POOLS[cache_key] = pool
        return pool


def _key_pool_cache_key(model_spec: dict[str, Any]) -> str:
    key_file = _resolve_key_file(model_spec)
    key_file_identity = str(key_file or "")
    if key_file is not None and key_file.exists():
        stat = key_file.stat()
        key_file_identity += f"|{stat.st_size}|{stat.st_mtime_ns}"

    identity = {
        "provider": str(model_spec.get("provider") or ""),
        "api_base": str(model_spec.get("api_base") or ""),
        "api_key_file": key_file_identity,
        "inline_key_fingerprint": _secret_fingerprint("\n".join(_inline_api_keys(model_spec))),
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_api_keys(model_spec: dict[str, Any]) -> list[str]:
    keys = _inline_api_keys(model_spec)
    key_file = validate_api_key_file(model_spec)
    if key_file is not None:
        keys.extend(_read_key_file(key_file))
    return _dedupe_preserve_order(keys)


def validate_api_key_file(model_spec: dict[str, Any]) -> Path | None:
    """Resolve and validate an explicitly configured credential file."""
    key_file = _resolve_key_file(model_spec)
    if key_file is None:
        return None
    if not key_file.exists():
        raise FileNotFoundError(f"Configured API key file does not exist: {key_file}")
    if not key_file.is_file():
        raise ValueError(f"Configured API key path is not a file: {key_file}")
    return key_file


def _inline_api_keys(model_spec: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    raw_keys = (
        os.environ.get(str(model_spec.get("api_key_env") or ""))
        or model_spec.get("api_keys")
        or model_spec.get("api_key")
    )
    if isinstance(raw_keys, str):
        keys.extend(_split_key_text(raw_keys))
    elif isinstance(raw_keys, list):
        keys.extend(str(key).strip() for key in raw_keys if str(key).strip())
    return keys


def _secret_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def _worker_start_index(num_keys: int) -> int:
    worker_identity = f"{socket.gethostname()}:{os.getpid()}"
    digest = hashlib.sha256(worker_identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_keys


def _resolve_key_file(model_spec: dict[str, Any]) -> Path | None:
    raw_path = str(model_spec.get("api_key_file") or "")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _read_key_file(path: Path) -> list[str]:
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.extend(_split_key_text(line))
    return keys


def _split_key_text(text: str) -> list[str]:
    return [part.strip() for part in text.replace(",", "\n").splitlines() if part.strip()]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _provider_headers(model_spec: dict[str, Any]) -> dict[str, str]:
    if model_spec.get("provider") != "openrouter":
        return {}
    headers = {"X-Title": str(model_spec.get("app_name") or "MemPilot")}
    site_url = str(model_spec.get("site_url") or "").strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    return headers


def _extract_chat_completion_text(response: Any) -> str:
    choices = _read_field(response, "choices") or []
    if not choices:
        return ""
    message = _read_field(choices[0], "message")
    content = _read_field(message, "content")
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _read_field(item, "text")
            if text:
                parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _normalize_api_usage(raw: Any) -> dict[str, int | float]:
    prompt_tokens = max(_safe_int(_read_field(raw, "prompt_tokens") or _read_field(raw, "input_tokens")), 0)
    completion_tokens = max(
        _safe_int(_read_field(raw, "completion_tokens") or _read_field(raw, "output_tokens")),
        0,
    )
    total_tokens = max(_safe_int(_read_field(raw, "total_tokens")), 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    usage: dict[str, int | float] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    reported_cost = _read_field(raw, "cost")
    if reported_cost is not None:
        usage["cost_usd"] = max(_safe_float(reported_cost), 0.0)
    return usage


def _read_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
