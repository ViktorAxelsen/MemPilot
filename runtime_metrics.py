"""Orchestrator and delegated-model resource diagnostics for MemPilot.

Internal raw costs use token counts times USD-per-million prices. Reporting
converts them to USD; latency uses deterministic proxies rather than observed
wall-clock duration, with parallel calls aggregated along the critical path.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from tensordict.utils import LinkedList as _TensorDictLinkedList
except ImportError:  # Keep metric helpers importable without the training stack.
    _TensorDictLinkedList = None


RUNTIME_MEMORY_METRICS_KEY = "runtime_memory_metrics"
BASE_MODEL_USAGE_KEY = "base_model_usage"
_MODEL_CALL_LOG_FILENAME_MAX_BYTES = 200
_MODEL_CALL_LOG_SUFFIX = ".model_calls.log"

# Keep per-dataset comparisons focused on answer quality and resource use.
# Detailed memory-call diagnostics remain available in the global metrics.
_DATASET_REWARD_KEYS = frozenset(
    {
        "score",
        "f1",
        "exact_match",
        "format_valid",
        "answer_format_valid",
        "tool_format_valid",
        "total_api_cost",
        "external_api_cost",
        "base_model_api_cost",
        "total_estimated_latency",
        "external_estimated_latency",
        "base_model_estimated_latency",
        "base_model_num_calls",
    }
)
# verl already reports the validation score under val-core/<source>/reward/...
_VALIDATION_AUX_REWARD_KEYS = _DATASET_REWARD_KEYS - {"score"}


def record_base_model_call(
    extra_fields: dict[str, Any],
    *,
    input_tokens: int,
    output_tokens: int,
    input_price_per_million_usd: float,
    output_price_per_million_usd: float,
    latency_base_seconds: float,
    latency_input_seconds_per_token: float,
    latency_output_seconds_per_token: float,
) -> None:
    """Accumulate one policy generation's proxy cost and latency.

    Prices are quoted in USD per million tokens, but multiplication deliberately
    does not divide by one million. This matches routed-model ``memory_api_cost``
    and leaves the common raw objective to downstream normalization/GroupNorm.
    Divide by 1e6 when converting this raw cost to USD. Latency is an affine
    token-count proxy in seconds, not measured serving time.
    """
    if not isinstance(extra_fields, dict):
        raise TypeError("Base-model usage requires mapping-valued agent extra_fields.")
    input_count = int(input_tokens)
    output_count = int(output_tokens)
    input_price = float(input_price_per_million_usd)
    output_price = float(output_price_per_million_usd)
    call_cost = input_count * input_price + output_count * output_price
    call_latency = (
        float(latency_base_seconds)
        + input_count * float(latency_input_seconds_per_token)
        + output_count * float(latency_output_seconds_per_token)
    )

    extras = extra_fields.setdefault("extras", {})
    if not isinstance(extras, dict):
        raise TypeError("agent extra_fields.extras must be a mapping.")
    usage = extras.setdefault(
        BASE_MODEL_USAGE_KEY,
        {
            "num_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "api_cost": 0.0,
            "estimated_latency": 0.0,
            "input_price_per_million_usd": input_price,
            "output_price_per_million_usd": output_price,
        },
    )
    if not isinstance(usage, dict):
        raise TypeError("agent extra_fields.extras.base_model_usage must be a mapping.")
    for key, expected in (
        ("input_price_per_million_usd", input_price),
        ("output_price_per_million_usd", output_price),
    ):
        observed = usage.setdefault(key, expected)
        if abs(float(observed) - expected) > 1e-12:
            raise ValueError(
                f"Base-model pricing changed within one trajectory: {key}={observed!r}, "
                f"expected {expected!r}."
            )
    usage["num_calls"] = _nonnegative_int(usage.get("num_calls")) + 1
    usage["input_tokens"] = _nonnegative_int(usage.get("input_tokens")) + input_count
    usage["output_tokens"] = _nonnegative_int(usage.get("output_tokens")) + output_count
    usage["total_tokens"] = _nonnegative_int(usage.get("total_tokens")) + input_count + output_count
    usage["api_cost"] = _nonnegative_float(usage.get("api_cost")) + call_cost
    usage["estimated_latency"] = (
        _nonnegative_float(usage.get("estimated_latency")) + call_latency
    )


def record_base_model_trajectory(
    extra_fields: dict[str, Any],
    *,
    initial_prompt_tokens: int,
    response_mask: Iterable[int],
    input_price_per_million_usd: float,
    output_price_per_million_usd: float,
    latency_base_seconds: float,
    latency_input_seconds_per_token: float,
    latency_output_seconds_per_token: float,
) -> None:
    """Account for unique policy inputs/outputs under ideal KV-prefix reuse.

    The initial prompt is charged once. Zeros between assistant segments are
    environment-authored tool-response tokens and become the incremental input
    to the next generation. Prior assistant tokens are not charged again.
    """
    pending_input_tokens = int(initial_prompt_tokens)
    pending_output_tokens = 0
    recorded_call = False
    for value in response_mask:
        if int(value):
            pending_output_tokens += 1
            continue
        if pending_output_tokens:
            record_base_model_call(
                extra_fields,
                input_tokens=pending_input_tokens,
                output_tokens=pending_output_tokens,
                input_price_per_million_usd=input_price_per_million_usd,
                output_price_per_million_usd=output_price_per_million_usd,
                latency_base_seconds=latency_base_seconds,
                latency_input_seconds_per_token=latency_input_seconds_per_token,
                latency_output_seconds_per_token=latency_output_seconds_per_token,
            )
            recorded_call = True
            pending_input_tokens = 0
            pending_output_tokens = 0
        pending_input_tokens += 1

    if pending_output_tokens or not recorded_call:
        record_base_model_call(
            extra_fields,
            input_tokens=pending_input_tokens,
            output_tokens=pending_output_tokens,
            input_price_per_million_usd=input_price_per_million_usd,
            output_price_per_million_usd=output_price_per_million_usd,
            latency_base_seconds=latency_base_seconds,
            latency_input_seconds_per_token=latency_input_seconds_per_token,
            latency_output_seconds_per_token=latency_output_seconds_per_token,
        )


def extract_base_model_usage(container: Any) -> dict[str, float | int]:
    """Return fixed numeric policy-usage fields from an agent/reward mapping."""
    if not isinstance(container, dict):
        return _empty_base_model_usage()
    extras = container.get("extras")
    if not isinstance(extras, dict):
        return _empty_base_model_usage()
    usage = extras.get(BASE_MODEL_USAGE_KEY)
    if not isinstance(usage, dict):
        return _empty_base_model_usage()
    output = {
        "num_calls": _nonnegative_int(usage.get("num_calls")),
        "input_tokens": _nonnegative_int(usage.get("input_tokens")),
        "output_tokens": _nonnegative_int(usage.get("output_tokens")),
        "total_tokens": _nonnegative_int(usage.get("total_tokens")),
        "api_cost": _nonnegative_float(usage.get("api_cost")),
        "estimated_latency": _nonnegative_float(usage.get("estimated_latency")),
    }
    pricing_keys = (
        "input_price_per_million_usd",
        "output_price_per_million_usd",
    )
    if any(key in usage for key in pricing_keys):
        if not all(key in usage for key in pricing_keys):
            raise ValueError("Base-model usage contains incomplete pricing metadata.")
        output.update(
            {
                key: _nonnegative_float(usage[key])
                for key in pricing_keys
            }
        )
    return output


def materialize_transfer_queue_value(value: Any) -> Any:
    """Recursively restore Python containers returned by TransferQueue."""
    if _TensorDictLinkedList is not None and isinstance(value, _TensorDictLinkedList):
        value = list(value)
    elif not isinstance(value, (dict, list, tuple)):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()

    if isinstance(value, dict):
        return {
            key: materialize_transfer_queue_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [materialize_transfer_queue_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_transfer_queue_value(item) for item in value)
    return value


def extract_runtime_memory_metrics(container: Any) -> list[dict[str, Any]]:
    """Read per-call diagnostics from a verl agent/reward extra-fields mapping."""
    if not isinstance(container, dict):
        return []
    extras = container.get("extras")
    if not isinstance(extras, dict):
        return []
    values = extras.get(RUNTIME_MEMORY_METRICS_KEY)
    if not isinstance(values, (list, tuple)):
        return []
    return [value for value in values if isinstance(value, dict)]


def summarize_runtime_memory_metrics(calls: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """Produce fixed-shape per-sample values safe for verl reward batching."""
    calls = list(calls)
    successful_calls = sum(_runtime_memory_call_succeeded(call) for call in calls)
    direct_returns = sum(as_bool(call.get("direct_memory_return")) for call in calls)
    num_calls = len(calls)
    return {
        "num_runtime_memory_calls": num_calls,
        "num_successful_memory_calls": successful_calls,
        "num_failed_memory_calls": num_calls - successful_calls,
        "num_direct_memory_returns": direct_returns,
        "runtime_memory_success_rate": successful_calls / num_calls if num_calls else 0.0,
        "runtime_prompt_tokens": sum(_nonnegative_int(call.get("prompt_tokens")) for call in calls),
        "runtime_completion_tokens": sum(_nonnegative_int(call.get("completion_tokens")) for call in calls),
        "runtime_total_tokens": sum(_nonnegative_int(call.get("total_tokens")) for call in calls),
        "runtime_input_images": sum(_nonnegative_int(call.get("num_input_images")) for call in calls),
        "runtime_estimated_latency": runtime_memory_critical_path_latency(calls),
    }


def runtime_memory_critical_path_latency(calls: Iterable[dict[str, Any]]) -> float:
    """Sum delegated-call latency proxies along the trajectory's critical path.

    Use the maximum proxy within each parallel stage, then sum sequential
    stages. Orchestrator generation latency is accounted for separately.
    """

    stage_latencies: dict[tuple[str, int], float] = {}
    for index, call in enumerate(calls):
        raw_stage = call.get("latency_stage")
        try:
            stage = ("stage", int(raw_stage))
        except (TypeError, ValueError):
            # Older/missing metrics have no safe parallelism signal, so treat
            # each call as its own sequential stage rather than undercounting.
            stage = ("call", index)
        latency = _nonnegative_float(call.get("memory_estimated_latency"))
        stage_latencies[stage] = max(stage_latencies.get(stage, 0.0), latency)
    return sum(stage_latencies.values())


def configured_runtime_memory_models(tool_config: Any) -> dict[str, str]:
    """Map RuntimeMemoryTool route IDs to their provider model names."""
    if not isinstance(tool_config, dict):
        raise ValueError("The rollout tool config must contain a mapping at its root.")

    models: dict[str, str] = {}
    for tool in tool_config.get("tools", []):
        if not isinstance(tool, dict):
            continue
        if not str(tool.get("class_name") or "").endswith(".RuntimeMemoryTool"):
            continue
        tool_settings = tool.get("config")
        if not isinstance(tool_settings, dict):
            continue
        for route in tool_settings.get("model_pool", []):
            if not isinstance(route, dict):
                continue
            route_name = str(route.get("name") or "").strip()
            if route_name and route_name not in models:
                models[route_name] = str(route.get("model") or "").strip()
    return models


def count_runtime_memory_calls(extra_fields: Iterable[Any]) -> dict[str, int]:
    """Count attempted external-model calls by opaque route ID."""
    counts: dict[str, int] = defaultdict(int)
    for field in extra_fields:
        for call in extract_runtime_memory_metrics(field):
            if not _external_call_attempted(call):
                continue
            route = str(call.get("model") or "unknown").strip() or "unknown"
            counts[route] += 1
    return dict(counts)


def write_model_call_count_log(
    path: str | Path,
    call_counts: dict[str, int],
    configured_models: dict[str, str],
) -> Path:
    """Atomically write one tab-separated ``route, provider model, count`` row per model."""
    target = _shorten_model_call_log_path(Path(path).expanduser())
    target.parent.mkdir(parents=True, exist_ok=True)
    route_names = _unique_metric_names([*configured_models, *call_counts])
    content = "".join(
        f"{route}\t{configured_models.get(route, 'unknown') or 'unknown'}"
        f"\t{_nonnegative_int(call_counts.get(route, 0))}\n"
        for route in route_names
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            # Do not repeat the potentially long target name here. Linux applies
            # NAME_MAX independently to the temporary filename before replace.
            prefix=".model_calls.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def _shorten_model_call_log_path(path: Path) -> Path:
    """Keep one filename component safely below common filesystem limits."""

    name = path.name
    if len(name.encode("utf-8")) <= _MODEL_CALL_LOG_FILENAME_MAX_BYTES:
        return path

    suffix = _MODEL_CALL_LOG_SUFFIX if name.endswith(_MODEL_CALL_LOG_SUFFIX) else path.suffix
    stem = name[: -len(suffix)] if suffix else name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    separator = "."
    stem_budget = (
        _MODEL_CALL_LOG_FILENAME_MAX_BYTES
        - len((separator + digest + suffix).encode("utf-8"))
    )
    if stem_budget < 1:
        raise ValueError("Model-call log suffix leaves no room for a filename stem.")
    shortened_stem = stem.encode("utf-8")[:stem_budget].decode("utf-8", errors="ignore")
    return path.with_name(f"{shortened_stem}{separator}{digest}{suffix}")


def filter_validation_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Keep auxiliary validation means and preserve all non-auxiliary metrics."""
    output = {}
    for name, value in metrics.items():
        if not name.startswith("val-aux/") or name == "val-aux/num_turns/mean":
            output[name] = value
            continue
        # Split from the right so custom dataset names can contain slashes.
        parts = name.rsplit("/", 2)
        if (
            len(parts) == 3
            and parts[-2] in _VALIDATION_AUX_REWARD_KEYS
            and parts[-1].startswith("mean@")
        ):
            output[name] = value
    return output


def build_wandb_runtime_metrics(
    extra_fields: Iterable[Any],
    configured_routes: Iterable[str] = (),
) -> dict[str, float]:
    """Aggregate one training batch into scalar metrics accepted by W&B."""
    extra_fields = list(extra_fields)
    sample_count = len(extra_fields)
    if not sample_count:
        return {}

    output: dict[str, float] = {}
    calls_per_sample = [extract_runtime_memory_metrics(field) for field in extra_fields]
    calls = [call for sample_calls in calls_per_sample for call in sample_calls]
    num_calls = len(calls)
    successful_calls = sum(_runtime_memory_call_succeeded(call) for call in calls)
    direct_returns = sum(as_bool(call.get("direct_memory_return")) for call in calls)
    external_calls = [call for call in calls if _external_call_attempted(call)]
    num_external_calls = len(external_calls)
    total_tokens = sum(_nonnegative_int(call.get("total_tokens")) for call in calls)
    api_cost = sum(_nonnegative_float(call.get("memory_api_cost")) for call in calls)
    critical_path_latency = sum(
        runtime_memory_critical_path_latency(sample_calls)
        for sample_calls in calls_per_sample
    )
    input_images = sum(_nonnegative_int(call.get("num_input_images")) for call in calls)
    input_memory_items = sum(_nonnegative_int(call.get("num_input_memory_items")) for call in calls)

    output.update(
        {
            "runtime_memory/calls_per_sample": num_calls / sample_count,
            "runtime_memory/success_rate": successful_calls / num_calls if num_calls else 0.0,
            "runtime_memory/direct_return_share": direct_returns / num_calls if num_calls else 0.0,
            "runtime_memory/external_calls_per_sample": num_external_calls / sample_count,
            "runtime_memory/total_tokens_per_sample": total_tokens / sample_count,
            "runtime_memory/api_cost_per_sample": api_cost / sample_count,
            "runtime_memory/estimated_critical_path_latency_seconds_per_sample": (
                critical_path_latency / sample_count
            ),
            "runtime_memory/input_images_per_sample": input_images / sample_count,
            "runtime_memory/input_memory_items_per_call": (
                input_memory_items / num_calls if num_calls else 0.0
            ),
        }
    )

    calls_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in external_calls:
        route = str(call.get("model") or "unknown").strip() or "unknown"
        calls_by_route[route].append(call)

    route_names = _unique_metric_names([*configured_routes, *calls_by_route])
    for route in route_names:
        route_calls = calls_by_route[route]
        route_call_count = len(route_calls)
        prefix = f"runtime_memory/route/{_metric_component(route)}"
        route_successes = sum(as_bool(call.get("external_call_succeeded")) for call in route_calls)
        route_cost = sum(_nonnegative_float(call.get("memory_api_cost")) for call in route_calls)
        # Per-route call frequency is call_share * external_calls_per_sample.
        output.update(
            {
                f"{prefix}/call_share": (
                    route_call_count / num_external_calls if num_external_calls else 0.0
                ),
                f"{prefix}/success_rate": route_successes / route_call_count if route_call_count else 0.0,
                f"{prefix}/api_cost_per_sample": route_cost / sample_count,
            }
        )

    fields_by_source: dict[str, list[Any]] = defaultdict(list)
    for field in extra_fields:
        source = (
            str(field.get("dataset_source") or "").strip()
            if isinstance(field, dict)
            else ""
        )
        if source:
            fields_by_source[source].append(field)
    for source, source_fields in fields_by_source.items():
        prefix = f"dataset/{_metric_component(source)}"
        for metric_name, value in _aggregate_dataset_reward_metrics(source_fields).items():
            output[f"{prefix}/{metric_name}"] = value

    return output


def _aggregate_dataset_reward_metrics(extra_fields: list[Any]) -> dict[str, float]:
    reward_infos = [
        field.get("reward_extra_info", {}) if isinstance(field, dict) else {}
        for field in extra_fields
    ]
    output: dict[str, float] = {}
    for key in _DATASET_REWARD_KEYS:
        values = [info[key] for info in reward_infos if isinstance(info, dict) and _is_number(info.get(key))]
        if values:
            output[f"reward_debug/{key}_mean"] = sum(float(value) for value in values) / len(values)
    return output


def _metric_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _unique_metric_names(values: Iterable[Any]) -> list[str]:
    normalized = (str(value).strip() for value in values)
    return list(dict.fromkeys(value for value in normalized if value))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_bool(value: Any) -> bool:
    """Normalize bool-like runtime fields emitted through serialization layers."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_memory_call_succeeded(call: dict[str, Any]) -> bool:
    if "memory_call_succeeded" in call:
        return as_bool(call.get("memory_call_succeeded"))
    return as_bool(call.get("external_call_succeeded"))


def _external_call_attempted(call: dict[str, Any]) -> bool:
    if "external_call_attempted" in call:
        return as_bool(call.get("external_call_attempted"))
    return not as_bool(call.get("direct_memory_return"))


def _nonnegative_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _nonnegative_float(value: Any) -> float:
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _empty_base_model_usage() -> dict[str, float | int]:
    return {
        "num_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_cost": 0.0,
        "estimated_latency": 0.0,
    }
