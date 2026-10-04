"""Capture audit metadata for response-only final evaluation rollouts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runtime_metrics import extract_base_model_usage, extract_runtime_memory_metrics


_SAMPLE_METADATA_KEYS = (
    "split",
    "index",
    "conversation_id",
    "scenario_id",
    "question_id",
    "paired_question_id",
    "checkpoint_id",
    "question",
    "question_type",
    "question_type_main",
    "question_subtype",
    "question_type_name",
    "answer_variant",
    "variant",
    "memeye_point",
    "regime",
    "category",
    "scenario",
    "difficulty",
    "dialogue_name",
    "question_session",
    "validated",
    "evaluation_subset",
    "context_length",
    "context_token_count",
    "conversation_session_count",
    "visible_session_count",
    "visible_context_token_count",
    "full_context_token_count",
    "full_memory_chunk_count",
    "question_date",
    "rotation_index",
    "choices",
    "official_metric",
    "official_metrics",
    "has_query_image",
    "query_images",
    "chunking_mode",
    "chunk_size",
)


def capture_evaluation_metadata(
    *,
    data_source: Any,
    extra_info: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return string-valued metadata retained by Verl's validation JSONL dump.

    The final-evaluation launchers use a capture-only reward, so metric
    computation does not happen on rollout workers. The raw prompt, response,
    and ground truth are already emitted by Verl; this function adds only the
    dataset metadata and resource trace needed by the independent evaluator.
    """

    extra_info = extra_info or {}
    if str(extra_info.get("split") or "").strip().casefold() == "train":
        raise ValueError("The capture-only reward must not be used for training rows.")
    dataset_source = str(data_source or "").strip()
    if not dataset_source:
        raise ValueError("Evaluation capture requires a non-empty data_source.")
    metadata = {
        key: _json_safe(extra_info[key])
        for key in _SAMPLE_METADATA_KEYS
        if key in extra_info and extra_info[key] is not None
    }
    metadata["dataset_source"] = dataset_source
    runtime_trace = {
        "base_model": _json_safe(extract_base_model_usage(dict(extra_info))),
        "memory_calls": _json_safe(extract_runtime_memory_metrics(dict(extra_info))),
    }
    return {
        "dataset_source": dataset_source,
        "sample_metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "runtime_trace_json": json.dumps(runtime_trace, ensure_ascii=False, sort_keys=True),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    return str(value)
