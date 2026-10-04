"""Prepare MemPilot's dual-view multimodal memory without question conditioning.

Raw history H preserves complete turns and image references. The default
query-agnostic bank M compresses dialogue text and captions independently while
retaining exact image IDs; each view is indexed separately for runtime search.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm.auto import tqdm

from memory_chunking import AtomicDialogueChunker, AtomicDialogueTurn
from memory_compression import TextCompression, uncompressed_memory_view
from memory_access import LOSSY_PREVIEW_FIELD
from retrieval import serialize_document_embeddings


DEFAULT_RETRIEVER_BATCH_SIZE = 64
TOKEN_CHUNKING_MODE = "token"
ROUND_CHUNKING_MODE = "round"
CHUNKING_MODES = (TOKEN_CHUNKING_MODE, ROUND_CHUNKING_MODE)
PREVIEW_TEXT_FIELD = "preview_text"
PROTECTED_PREVIEW_FIELD = "protected_preview_text"

@dataclass(frozen=True)
class MultimodalTextCompression:
    """Compressed dialogue plus caption results aligned with one memory chunk."""

    dialogue: TextCompression
    image_captions: tuple[TextCompression | None, ...]


def build_memory_bank(
    record: Mapping[str, Any],
    chunker: AtomicDialogueChunker,
    *,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
    dataset_label: str = "Multimodal memory",
) -> list[dict[str, Any]]:
    """Chunk complete dialogue turns and assign conversation-global memory IDs."""

    chunking_mode = normalize_chunking_mode(chunking_mode)
    memory_bank: list[dict[str, Any]] = []
    for session in record["sessions"]:
        session_id = str(session.get("session_id") or "").strip()
        turns = [
            AtomicDialogueTurn(
                text=turn["text"],
                preview_text=turn["preview_text"],
                protected_preview_text=turn["protected_preview_text"],
                turn_id=turn["turn_id"],
                images=tuple(turn["images"]),
            )
            for turn in session["turns"]
        ]
        turn_groups: Iterable[Sequence[AtomicDialogueTurn]] = (
            ([turn] for turn in turns)
            if chunking_mode == ROUND_CHUNKING_MODE
            else (turns,)
        )
        for turn_group in turn_groups:
            for chunk in chunker.chunk_session(turn_group, time=session["date"]):
                chunk["id"] = f"m{len(memory_bank):04d}"
                if session_id:
                    chunk["session_id"] = session_id
                memory_bank.append(chunk)

    image_ids = [
        str(image.get("id") or "")
        for item in memory_bank
        for image in item.get("images", [])
    ]
    duplicate_ids = sorted(
        image_id for image_id in set(image_ids) if image_ids.count(image_id) > 1
    )
    if duplicate_ids:
        raise ValueError(
            f"{dataset_label} conversation {record['conversation_id']!r} has duplicate image IDs: "
            f"{duplicate_ids}"
        )
    unsafe_ids = [image_id for image_id in image_ids if not image_id or "," in image_id]
    if unsafe_ids:
        raise ValueError(f"{dataset_label} image IDs must be non-empty and comma-safe: {unsafe_ids}")
    return memory_bank


def normalize_chunking_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode not in CHUNKING_MODES:
        raise ValueError(f"chunking_mode must be one of {CHUNKING_MODES}; got {value!r}.")
    return mode


def clean_memory_items(memory_items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove transient fields used only to construct persisted memory views."""

    return [
        {
            key: value
            for key, value in dict(item).items()
            if key not in {PREVIEW_TEXT_FIELD, PROTECTED_PREVIEW_FIELD}
        }
        for item in memory_items
    ]


def _compress_multimodal_sources(
    originals: Sequence[Mapping[str, Any]],
    compressor: Any,
    *,
    show_progress: bool = False,
    description: str = "Compressing multimodal memory chunks",
) -> list[MultimodalTextCompression]:
    sources: list[str] = []
    layouts: list[tuple[int, tuple[int | None, ...]]] = []
    for item in originals:
        dialogue_index = len(sources)
        sources.append(str(item.get(PREVIEW_TEXT_FIELD) or item.get("text") or ""))
        caption_indices = []
        for image in item.get("images", []):
            caption = str(image.get("caption") or "").strip()
            caption_indices.append(len(sources) if caption else None)
            if caption:
                sources.append(caption)
        layouts.append((dialogue_index, tuple(caption_indices)))

    compress_many = getattr(compressor, "compress_texts", None)
    if callable(compress_many):
        if show_progress:
            results = compress_many(
                sources,
                show_progress=True,
                description=description,
            )
        else:
            results = compress_many(sources)
    else:
        results = [
            compressor.compress_text(text)
            for text in tqdm(
                sources,
                desc=description,
                unit="text",
                dynamic_ncols=True,
                disable=not show_progress,
            )
        ]
    if len(results) != len(sources) or not all(
        isinstance(result, TextCompression) for result in results
    ):
        raise ValueError("memory compressor must return one TextCompression per dialogue or image caption.")

    return [
        MultimodalTextCompression(
            dialogue=results[dialogue_index],
            image_captions=tuple(
                results[index] if index is not None else None
                for index in caption_indices
            ),
        )
        for dialogue_index, caption_indices in layouts
    ]


def _build_multimodal_memory_views(
    originals: Sequence[Mapping[str, Any]],
    results: Sequence[Any],
    compressor: Any,
    *,
    dataset_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(results) != len(originals) or not all(
        isinstance(result, MultimodalTextCompression) for result in results
    ):
        raise ValueError(
            f"memory compressor must return one aligned multimodal result per {dataset_label} item."
        )

    original_items = clean_memory_items(originals)
    compressed_items = [dict(item) for item in original_items]
    original_tokens = 0
    compressed_tokens = 0
    for compressed, source, result in zip(compressed_items, originals, results):
        image_preview = _compressed_image_preview(source, result.image_captions)
        compressed["text"] = "\n\n".join(
            part for part in (result.dialogue.text.strip(), image_preview) if part
        )
        components = (
            result.dialogue,
            *(value for value in result.image_captions if value is not None),
        )
        original_tokens += sum(value.original_tokens for value in components)
        compressed_tokens += sum(value.compressed_tokens for value in components)
    compression = {
        "enabled": True,
        "model": str(getattr(compressor, "model_name", "")),
        "target_rate": float(getattr(compressor, "rate", 0.0)),
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "actual_rate": compressed_tokens / original_tokens if original_tokens else 0.0,
    }
    return original_items, compressed_items, compression


def _compressed_image_preview(
    source: Mapping[str, Any],
    caption_results: Sequence[TextCompression | None],
) -> str:
    """Replace caption text in the protected image template without touching image IDs."""

    images = list(source.get("images") or [])
    if len(caption_results) != len(images):
        raise ValueError("Compressed image captions must align one-to-one with memory images.")
    preview = str(source.get(PROTECTED_PREVIEW_FIELD) or "").strip()
    for image, result in zip(images, caption_results):
        caption = " ".join(str(image.get("caption") or "").split())
        if not caption:
            if result is not None:
                raise ValueError("An empty image caption unexpectedly received a compression result.")
            continue
        if result is None:
            raise ValueError("A non-empty image caption is missing its compression result.")
        source_line = f"  image_caption: {caption}"
        if source_line not in preview:
            raise ValueError(
                f"Image caption template is missing the caption for image {str(image.get('id') or '')!r}."
            )
        compressed_caption = " ".join(result.text.split())
        replacement = f"  image_caption: {compressed_caption}" if compressed_caption else ""
        preview = preview.replace(source_line, replacement, 1)
    return preview.strip()


def attach_lossy_memory_banks(
    examples: list[dict[str, Any]],
    compressor: Any,
    embedding_retriever: Any,
    *,
    dataset_label: str,
    show_progress: bool = False,
) -> None:
    """Attach lossy text and its dense index once per conversation and scope."""

    if not callable(getattr(embedding_retriever, "encode_documents", None)):
        raise TypeError("Lossy memory indexing requires an embedding retriever.")
    retriever_model_id = str(getattr(embedding_retriever, "model_id", "")).strip()
    if not retriever_model_id:
        raise ValueError("Lossy memory indexing requires a non-empty retriever model_id.")

    grouped: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for example in examples:
        key = (example.get("record_index"), str(example.get("conversation_id") or ""))
        compression_bank = list(example.get("compression_memory_bank") or example["memory_bank"])
        existing = grouped.setdefault(key, compression_bank)
        if _memory_item_ids(existing) != _memory_item_ids(compression_bank):
            raise ValueError(
                f"{dataset_label} examples from one conversation must share one compression memory bank."
            )

    flattened_items = [item for memory_bank in grouped.values() for item in memory_bank]
    flattened_results = _compress_multimodal_sources(
        flattened_items,
        compressor,
        show_progress=show_progress,
        description=f"Compressing {dataset_label} memory chunks",
    )

    prepared: dict[
        tuple[Any, str],
        tuple[list[dict[str, Any]], dict[str, Any], Any],
    ] = {}
    iterator = tqdm(
        grouped.items(),
        desc=f"Preparing {dataset_label} memory banks",
        unit="conversation",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    offset = 0
    for key, raw_bank in iterator:
        next_offset = offset + len(raw_bank)
        original_bank, lossy_bank, compression = _build_multimodal_memory_views(
            raw_bank,
            flattened_results[offset:next_offset],
            compressor,
            dataset_label=dataset_label,
        )
        offset = next_offset
        original_ids = [str(item.get("id") or "") for item in original_bank]
        lossy_ids = [str(item.get("id") or "") for item in lossy_bank]
        if original_ids != lossy_ids or any(not item_id for item_id in original_ids):
            raise ValueError(f"{dataset_label} lossy memory must preserve non-empty source IDs and order.")
        enriched_bank = []
        for original, lossy in zip(original_bank, lossy_bank):
            enriched = dict(original)
            enriched[LOSSY_PREVIEW_FIELD] = str(lossy.get("text") or "").strip()
            enriched_bank.append(enriched)
        lossy_embeddings = embedding_retriever.encode_documents(lossy_bank)
        if lossy_embeddings is None or len(lossy_embeddings) != len(lossy_bank):
            raise ValueError(
                f"{dataset_label} lossy index must contain one embedding per memory item."
            )
        prepared[key] = enriched_bank, compression, lossy_embeddings

    if offset != len(flattened_results):
        raise ValueError(f"Unexpected extra compression results for {dataset_label} memory banks.")

    index_cache: dict[tuple[tuple[Any, str], tuple[str, ...]], str] = {}
    for example in examples:
        key = (example.get("record_index"), str(example.get("conversation_id") or ""))
        prepared_bank, compression, lossy_embeddings = prepared[key]
        prepared_by_id = {str(item.get("id") or ""): item for item in prepared_bank}
        scoped_ids = _memory_item_ids(example["memory_bank"])
        try:
            example["memory_bank"] = [prepared_by_id[item_id] for item_id in scoped_ids]
        except KeyError as exc:
            raise ValueError(
                f"{dataset_label} scoped memory contains an item outside its compression bank: {exc.args[0]!r}."
            ) from exc
        index_key = (key, tuple(scoped_ids))
        lossy_memory_index = index_cache.get(index_key)
        if lossy_memory_index is None:
            row_by_id = {
                str(item.get("id") or ""): row
                for row, item in enumerate(prepared_bank)
            }
            scoped_rows = [row_by_id[item_id] for item_id in scoped_ids]
            lossy_memory_index = serialize_document_embeddings(
                _select_embedding_rows(lossy_embeddings, scoped_rows),
                model_id=retriever_model_id,
            )
            index_cache[index_key] = lossy_memory_index
        example["lossy_memory_index"] = lossy_memory_index
        example["memory_bank_compression"] = compression
        example.pop("compression_memory_bank", None)


def _select_embedding_rows(embeddings: Any, rows: Sequence[int]) -> Any:
    import numpy as np

    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError("Lossy document embeddings must be a two-dimensional matrix.")
    return matrix[np.asarray(rows, dtype=np.int64)]


def _memory_item_ids(memory_bank: Iterable[Mapping[str, Any]]) -> list[str]:
    item_ids = [str(item.get("id") or "") for item in memory_bank]
    if any(not item_id for item_id in item_ids) or len(item_ids) != len(set(item_ids)):
        raise ValueError("Memory-bank IDs must be non-empty and unique.")
    return item_ids


def resolve_agentic_memory_bank(
    example: Mapping[str, Any],
    *,
    memory_is_compressed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the complete bank and compression metadata; retrieval happens at runtime."""
    memory_bank = clean_memory_items(example["memory_bank"])
    if memory_is_compressed:
        return memory_bank, dict(example["memory_bank_compression"])
    return uncompressed_memory_view(memory_bank)


def format_image_caption_block(
    images: Sequence[Mapping[str, Any]],
    *,
    heading: str = "",
) -> str:
    """Format image handles and include captions only when they are available."""

    lines = [f"[{heading}]"] if heading and images else []
    for image in images:
        image_id = " ".join(str(image.get("id") or "").split())
        caption = " ".join(str(image.get("caption") or "").split())
        if not image_id:
            continue
        lines.append(f"- image_id: {image_id}")
        if caption:
            lines.append(f"  image_caption: {caption}")
    return "\n".join(lines)


def resolve_image_reference(
    raw_path: Any,
    source_path: Path | None,
    *,
    dataset_name: str,
    dataset_revision: str,
    dataset_label: str,
) -> dict[str, str]:
    """Validate an image and serialize a portable data-relative/HF reference."""

    if source_path is None:
        raise ValueError(f"Portable {dataset_label} image references require the source dialogue path.")
    cleaned = str(raw_path or "").strip().removeprefix("file://")
    if not cleaned:
        raise ValueError(f"{dataset_label} image paths must be non-empty.")

    raw = Path(cleaned).expanduser()
    source_path = Path(os.path.abspath(source_path))
    data_root = Path(os.path.abspath(source_path.parent.parent))
    normalized = cleaned.replace("\\", "/")
    for prefix in ("../image/", "./image/", "image/", "data/image/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                source_path.parent / raw,
                source_path.parent.parent / raw,
                data_root / "image" / Path(normalized),
            ]
        )
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = Path(os.path.abspath(candidate))
        candidate_key = str(candidate)
        if candidate_key not in seen:
            deduplicated.append(candidate)
            seen.add(candidate_key)
    path = next((candidate for candidate in deduplicated if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Could not resolve {dataset_label} image {cleaned!r} from {source_path}; "
            f"tried {deduplicated}."
        )
    try:
        relative_path = path.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{dataset_label} image must be inside the dataset data directory: {path}") from exc

    reference = {"path": relative_path}
    if dataset_name:
        reference.update(
            {
                "hf_repo_id": dataset_name,
                "hf_repo_type": "dataset",
                "hf_filename": f"data/{relative_path}",
            }
        )
        if dataset_revision:
            reference["hf_revision"] = dataset_revision
    return reference


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item or "").strip() for item in values]


def value_at(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""
