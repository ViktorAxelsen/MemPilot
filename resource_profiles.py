"""Load token-price and affine latency-proxy profiles for resource accounting."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_RESOURCE_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "runtime_memory_tool.yaml"
)


@dataclass(frozen=True)
class ResourceProfile:
    """One model's USD-per-million token prices and latency-proxy coefficients."""

    input_price_per_million_usd: float
    output_price_per_million_usd: float
    latency_base_seconds: float
    latency_input_seconds_per_token: float
    latency_output_seconds_per_token: float
    latency_image_seconds: float = 0.0
    model: str = ""
    route_name: str | None = None

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"Resource coefficient {name} for {self.model!r} must be finite and non-negative."
                )

    def as_dict(self) -> dict[str, float]:
        return {
            "input_price_per_million_usd": float(self.input_price_per_million_usd),
            "output_price_per_million_usd": float(self.output_price_per_million_usd),
            "latency_base_seconds": float(self.latency_base_seconds),
            "latency_input_seconds_per_token": float(
                self.latency_input_seconds_per_token
            ),
            "latency_output_seconds_per_token": float(
                self.latency_output_seconds_per_token
            ),
            "latency_image_seconds": float(self.latency_image_seconds),
        }

    def estimated_latency(
        self,
        *,
        num_calls: int,
        input_tokens: int,
        output_tokens: int,
        num_images: int = 0,
    ) -> float:
        """Return the affine proxy in seconds, not measured wall-clock latency."""

        values = {
            "num_calls": num_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_images": num_images,
        }
        parsed: dict[str, int] = {}
        for name, value in values.items():
            try:
                parsed[name] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
            if parsed[name] < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}.")
        return (
            parsed["num_calls"] * self.latency_base_seconds
            + parsed["input_tokens"] * self.latency_input_seconds_per_token
            + parsed["output_tokens"] * self.latency_output_seconds_per_token
            + parsed["num_images"] * self.latency_image_seconds
        )


@dataclass(frozen=True)
class ResourceProfileCatalog:
    """Validated profiles indexed by both route ID and backend model name."""

    config_path: Path
    config_sha256: str
    default_policy_model: str
    _profiles: Mapping[str, ResourceProfile]

    def resolve(self, identifier: str, *, role: str = "model") -> ResourceProfile:
        key = _normalize_identifier(identifier)
        profile = self._profiles.get(key)
        if profile is None:
            raise ValueError(
                f"No {role} resource profile for {identifier!r} in {self.config_path}. "
                "Add it to model_pool or resource_accounting.model_profiles."
            )
        return profile

    def resolve_external_call(self, call: Mapping[str, Any]) -> ResourceProfile:
        identifiers = [
            str(call.get(key) or "").strip()
            for key in ("model", "backend_model")
            if str(call.get(key) or "").strip()
        ]
        if not identifiers:
            raise ValueError("External-call metadata has neither model nor backend_model.")

        resolved: list[ResourceProfile] = []
        for identifier in identifiers:
            resolved.append(self.resolve(identifier, role="external-call"))
        canonical_models = {profile.model for profile in resolved}
        if len(canonical_models) != 1:
            raise ValueError(
                "External-call route/backend metadata resolves to different YAML profiles: "
                + ", ".join(repr(identifier) for identifier in identifiers)
            )
        return resolved[0]


def load_resource_profile_catalog(
    path: str | Path = DEFAULT_RESOURCE_CONFIG_PATH,
) -> ResourceProfileCatalog:
    """Load every reportable profile from ``runtime_memory_tool.yaml``."""

    config_path = Path(path).expanduser().resolve()
    raw_bytes = config_path.read_bytes()
    payload = yaml.safe_load(raw_bytes)
    tool_config = _runtime_memory_config(payload, config_path)
    accounting = tool_config.get("resource_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError(
            f"{config_path} must define RuntimeMemoryTool config.resource_accounting."
        )
    default_policy_model = str(accounting.get("default_policy_model") or "").strip()
    if not default_policy_model:
        raise ValueError(
            f"{config_path} resource_accounting.default_policy_model must be non-empty."
        )

    profiles: dict[str, ResourceProfile] = {}
    for index, raw_profile in enumerate(_mapping_sequence(accounting.get("model_profiles"))):
        model = str(raw_profile.get("model") or "").strip()
        if not model:
            raise ValueError(
                f"resource_accounting.model_profiles[{index}].model must be non-empty."
            )
        profile = _parse_profile(raw_profile, model=model)
        identifiers = [model, *_string_sequence(raw_profile.get("aliases"))]
        _register_profile(profiles, identifiers, profile, config_path)

    for index, raw_route in enumerate(_mapping_sequence(tool_config.get("model_pool"))):
        route_name = str(raw_route.get("name") or "").strip()
        model = str(raw_route.get("model") or "").strip()
        if not route_name or not model:
            raise ValueError(
                f"model_pool[{index}] must define non-empty name and model fields."
            )
        profile = _parse_profile(raw_route, model=model, route_name=route_name)
        identifiers = [
            route_name,
            model,
            *_string_sequence(raw_route.get("resource_profile_aliases")),
        ]
        _register_profile(profiles, identifiers, profile, config_path)

    if _normalize_identifier(default_policy_model) not in profiles:
        raise ValueError(
            f"default_policy_model {default_policy_model!r} has no profile in {config_path}."
        )
    return ResourceProfileCatalog(
        config_path=config_path,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        default_policy_model=default_policy_model,
        _profiles=profiles,
    )


def _runtime_memory_config(payload: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Resource config root must be a mapping: {path}")
    tools = payload.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise ValueError(f"Resource config must contain a tools list: {path}")
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        if not str(tool.get("class_name") or "").endswith(".RuntimeMemoryTool"):
            continue
        config = tool.get("config")
        if isinstance(config, Mapping):
            return config
    raise ValueError(f"No RuntimeMemoryTool config found in {path}.")


def _parse_profile(
    raw: Mapping[str, Any],
    *,
    model: str,
    route_name: str | None = None,
) -> ResourceProfile:
    pricing = raw.get("pricing")
    latency = raw.get("latency")
    if not isinstance(pricing, Mapping) or not isinstance(latency, Mapping):
        location = f"route {route_name!r}" if route_name else f"model {model!r}"
        raise ValueError(f"Resource profile for {location} requires pricing and latency mappings.")
    return ResourceProfile(
        model=model,
        route_name=route_name,
        input_price_per_million_usd=_coefficient(
            pricing, "input_per_million_usd", model
        ),
        output_price_per_million_usd=_coefficient(
            pricing, "output_per_million_usd", model
        ),
        latency_base_seconds=_coefficient(latency, "base_seconds", model),
        latency_input_seconds_per_token=_coefficient(
            latency, "input_seconds_per_token", model
        ),
        latency_output_seconds_per_token=_coefficient(
            latency, "output_seconds_per_token", model
        ),
        latency_image_seconds=_coefficient(
            latency, "image_seconds", model, default=0.0
        ),
    )


def _coefficient(
    values: Mapping[str, Any],
    key: str,
    model: str,
    *,
    default: float | None = None,
) -> float:
    value = values.get(key, default)
    if value is None:
        raise ValueError(f"Resource profile for {model!r} is missing {key}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Resource coefficient {key} for {model!r} must be numeric."
        ) from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"Resource coefficient {key} for {model!r} must be finite and non-negative."
        )
    return number


def _register_profile(
    registry: dict[str, ResourceProfile],
    identifiers: Sequence[str],
    profile: ResourceProfile,
    path: Path,
) -> None:
    for identifier in identifiers:
        key = _normalize_identifier(identifier)
        existing = registry.get(key)
        if existing is not None and existing != profile:
            raise ValueError(
                f"Resource identifier {identifier!r} is assigned to multiple profiles in {path}."
            )
        registry[key] = profile


def _normalize_identifier(value: str) -> str:
    return str(value or "").strip().casefold()


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Resource profile collection must be a list of mappings.")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError("Resource profile collection must contain only mappings.")
    return list(value)


def _string_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Resource profile aliases must be a list of strings.")
    aliases = [str(item).strip() for item in value]
    if any(not alias for alias in aliases):
        raise ValueError("Resource profile aliases must be non-empty strings.")
    return aliases
