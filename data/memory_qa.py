"""Shared row, policy-prompt, and conversation-split utilities for memory QA."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from memory_access import (
    DEFAULT_MAX_PARALLEL_CALLS,
    MEMORY_TOOL_CALL_REQUIRED_FIELD,
)


DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VALIDATION_RATIO = 0.1
RUNTIME_MEMORY_AGENT_IDENTITY = (
    "You are a runtime-memory agent that solves questions by planning and executing an adaptive memory pipeline, "
    "directly retrieving compressed evidence or delegating raw memory chunks to capability-diverse external LLMs "
    "for processing as needed."
)


def configure_runtime_tools(
    raw_extra_info: Any,
) -> dict[str, Any]:
    """Project one prepared row onto the RETRIEVE and CURATE interface."""

    if not isinstance(raw_extra_info, Mapping):
        raise ValueError("Runtime-memory rows require mapping-valued extra_info.")
    extra_info = copy.deepcopy(dict(raw_extra_info))
    raw_tools_kwargs = extra_info.get("tools_kwargs")
    if not isinstance(raw_tools_kwargs, Mapping):
        raise ValueError("Runtime-memory rows require mapping-valued tools_kwargs.")
    tools_kwargs = dict(raw_tools_kwargs)
    extra_info["tools_kwargs"] = tools_kwargs
    raw_runtime_kwargs = tools_kwargs.get("runtime_memory")
    if not isinstance(raw_runtime_kwargs, Mapping):
        raise ValueError("Runtime-memory rows require runtime_memory tool kwargs.")
    runtime_kwargs = dict(raw_runtime_kwargs)
    tools_kwargs["runtime_memory"] = runtime_kwargs

    tools_kwargs["retrieve_memory"] = copy.deepcopy(runtime_kwargs)
    extra_info["tool_selection"] = ["runtime_memory", "retrieve_memory"]
    # Require a memory tool call for a valid policy trajectory.
    extra_info[MEMORY_TOOL_CALL_REQUIRED_FIELD] = True
    return extra_info


def build_runtime_memory_system_prompt(
    *,
    final_answer_contract: str,
    include_image_argument_in_example: bool = False,
    max_parallel_calls: int = DEFAULT_MAX_PARALLEL_CALLS,
) -> str:
    """Build the dataset-neutral runtime-memory policy contract."""
    if max_parallel_calls < 1:
        raise ValueError("max_parallel_calls must be positive.")
    image_example = ',"include_images":true' if include_image_argument_in_example else ""
    retrieval_arguments = '"evidence_count":<integer>'
    runtime_arguments = retrieval_arguments + image_example

    runtime_query_argument = '"retrieval_query":"<self-contained evidence query>",'
    runtime_tool_example = (
        '<tool_call>{"name":"runtime_memory","arguments":{' + runtime_query_argument +
        '"instruction":"<evidence-processing instruction>","model":"<model ID>",'
        + runtime_arguments
        + '}}</tool_call>'
    )
    retrieve_tool_example = (
        '<tool_call>{"name":"retrieve_memory","arguments":{"retrieval_query":'
        '"<self-contained evidence query>",'
        + retrieval_arguments
        + '}}</tool_call>'
    )
    tool_examples = (
        "Available call forms (serialization only; neither is preferred):\n"
        "Direct compressed-memory retrieval:\n"
        f"{retrieve_tool_example}\n"
        "Routed original-memory processing:\n"
        f"{runtime_tool_example}"
    )

    agent_identity = RUNTIME_MEMORY_AGENT_IDENTITY
    image_help_step = ""
    memory_help_step = (
        "- Use retrieve_memory for faster access to precomputed query-agnostic compressed memory, which may "
        "omit important details. Use runtime_memory to delegate query-aware processing to a selected external "
        "LLM using the complete, uncompressed content of each retrieved raw memory chunk, while incurring some "
        "additional model-call cost and latency."
    )
    if include_image_argument_in_example:
        image_help_step = (
            "\n- For visual content, retrieve_memory exposes only textual captions and cannot inspect image pixels. "
            "To inspect pixels from retrieved or question images, use runtime_memory with an image-capable "
            "LLM and set include_images to true."
        )
    result_label = "memory-tool result"

    decomposition_step = (
        "- Before the first call, decompose the query into one or more atomic, evidence-dependent subtasks. "
        "Execute only the next unresolved subtask; do not delegate multiple dependent subtasks through one "
        "broad instruction.\n"
    )

    return f"""{agent_identity}

Work iteratively:
{memory_help_step}{image_help_step}
{decomposition_step}- Before each tool call, briefly assess what is known, what remains unresolved, and why the next call is useful.
  After each {result_label}, update this assessment before deciding on another call or the final answer.
- Execute dependent subtasks sequentially. Calls in the same assistant message run in parallel, so put only
  independent subtasks in separate tool_call blocks, at most {max_parallel_calls} per message. If a later call depends on an earlier
  {result_label}, issue it in a later turn.
- Each successful tool result contains findings supported by that call's inputs and may identify unresolved parts.
  Missing evidence is local to the selected inputs, not proof that the full query is unanswerable. Use partial
  findings to plan later calls or the final answer.

Tool-call format:
{tool_examples}
Choose every argument value for the current step.

Once you have enough evidence, stop calling tools and finish with
{final_answer_contract}
Do not write anything after the closing </answer> tag."""


def build_free_text_ground_truth(reference_answers: Iterable[Any]) -> str:
    """Serialize one or more non-empty free-text reference answers."""
    answers = [
        str(answer).strip()
        for answer in reference_answers
        if answer is not None and str(answer).strip()
    ]
    if not answers:
        raise ValueError("Free-text ground truth requires at least one answer.")
    return json.dumps({"answers": answers}, ensure_ascii=False)


def build_agentic_memory_row(
    *,
    example: Mapping[str, Any],
    data_source: str,
    row_index: int,
    ground_truth: str,
    chunk_size: int,
    compression: Mapping[str, Any],
    dataset_extra_info: Mapping[str, Any] | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
    tool_create_kwargs: Mapping[str, Any] | None = None,
    ability: str = "text_memory",
) -> dict[str, Any]:
    """Build a QA row carrying raw history and its query-agnostic memory view.

    Raw chunks and compressed-view indexes remain in tool state; reward labels
    remain in evaluation metadata. The policy starts from a query-only prompt;
    the runtime dataset applies the active tool and concurrency configuration.
    """
    memory_bank = list(example["memory_bank"])
    _validated_memory_ids(memory_bank, "memory_bank", require_unique=True)
    lossy_memory_index = str(example.get("lossy_memory_index") or "")
    if compression.get("enabled") and memory_bank and not lossy_memory_index:
        raise ValueError(
            "Compressed memory rows require a precomputed lossy_memory_index."
        )
    tool_create_kwargs = dict(tool_create_kwargs or {})
    reserved_create_keys = {
        "memory_bank",
        "memory_index",
        "lossy_memory_index",
        "metadata",
    }
    conflicting_keys = sorted(reserved_create_keys & tool_create_kwargs.keys())
    if conflicting_keys:
        raise ValueError(f"tool_create_kwargs cannot override core runtime state: {conflicting_keys}")
    metadata = {
        "data_source": data_source,
        "conversation_id": example["conversation_id"],
        "record_index": example["record_index"],
        "split": example["split"],
        "question_id": example["question_id"],
        "question_type": example.get("question_type"),
        "memory_chunk_count": int(example.get("memory_chunk_count", len(example.get("memory_bank", [])))),
        "chunk_size": chunk_size,
        "memory_compression": dict(compression),
        **dict(metadata_extra or {}),
    }
    extra_info = {
        "split": example["split"],
        "index": row_index,
        "conversation_id": example["conversation_id"],
        "question_id": example["question_id"],
        "question_type": example.get("question_type"),
        "question": example["question"],
        "answer": example["answer"],
        "chunk_size": chunk_size,
        "memory_compression": dict(compression),
        **dict(dataset_extra_info or {}),
        "tool_selection": ["runtime_memory"],
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "runtime_memory": {
                "create_kwargs": {
                    "memory_bank": json.dumps(memory_bank, ensure_ascii=False),
                    "memory_index": str(example.get("memory_index") or ""),
                    "lossy_memory_index": lossy_memory_index,
                    "metadata": metadata,
                    **tool_create_kwargs,
                }
            }
        },
    }
    row = {
        "data_source": data_source,
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
    }

    # Import lazily because the benchmark prompt builders share this module.
    from data.runtime_memory_prompts import build_runtime_memory_messages

    row["prompt"] = build_runtime_memory_messages(row)
    return row


def split_rows(
    rows: list[dict[str, Any]],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    seed: int = 13,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by conversation into train, validation, and test sets.

    Preserve explicit split labels. Shuffle unassigned conversation groups with
    a fixed seed and split them 70%/10%/20% by default, keeping all questions from
    the same conversation in one split.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1.")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between 0 and 1.")
    development_ratio = train_ratio + validation_ratio
    if development_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be strictly less than 1.")

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        extra_info = row.get("extra_info", {}) or {}
        group_id = str(extra_info.get("conversation_id") or extra_info.get("index"))
        groups.setdefault(group_id, []).append(row)

    explicit_train: list[list[dict[str, Any]]] = []
    explicit_validation: list[list[dict[str, Any]]] = []
    explicit_test: list[list[dict[str, Any]]] = []
    remaining: list[list[dict[str, Any]]] = []
    train_labels = {"train", "training"}
    validation_labels = {"val", "valid", "validation", "dev"}
    test_labels = {"test"}
    for group_id, group_rows in groups.items():
        labels = {
            str(row.get("extra_info", {}).get("split") or "").strip().lower()
            for row in group_rows
        }
        has_train = bool(labels & train_labels)
        has_validation = bool(labels & validation_labels)
        has_test = bool(labels & test_labels)
        if sum((has_train, has_validation, has_test)) > 1:
            raise ValueError(
                f"Conversation group {group_id!r} has conflicting train/validation/test split labels."
            )
        if has_train:
            explicit_train.append(group_rows)
        elif has_validation:
            explicit_validation.append(group_rows)
        elif has_test:
            explicit_test.append(group_rows)
        else:
            remaining.append(group_rows)

    rng = random.Random(seed)
    rng.shuffle(remaining)
    train_cut = round(len(remaining) * train_ratio)
    development_cut = round(len(remaining) * development_ratio)
    if remaining:
        if not explicit_train:
            train_cut = max(train_cut, 1)
        if not explicit_validation:
            development_cut = max(development_cut, train_cut + 1)
        if not explicit_test:
            development_cut = min(development_cut, len(remaining) - 1)
        train_cut = min(train_cut, development_cut)
        if not explicit_validation:
            train_cut = min(train_cut, development_cut - 1)

    train_groups = explicit_train + remaining[:train_cut]
    validation_groups = explicit_validation + remaining[train_cut:development_cut]
    test_groups = explicit_test + remaining[development_cut:]
    train = [row for group in train_groups for row in group]
    validation = [row for group in validation_groups for row in group]
    test = [row for group in test_groups for row in group]
    if not train or not validation or not test:
        raise ValueError(
            "A leakage-free train/validation/test split requires at least three conversation groups "
            "or explicit labels for all three splits."
        )
    _set_rows_split(train, "train")
    _set_rows_split(validation, "val")
    _set_rows_split(test, "test")
    return train, validation, test


def sample_train_subset(
    train_rows: list[dict[str, Any]],
    size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a reproducible random subset without changing source row order."""
    if size is None:
        return train_rows
    if size <= 0:
        raise ValueError("train_subset_size must be positive when provided.")
    if size > len(train_rows):
        raise ValueError(
            f"train_subset_size={size} exceeds the full train split size {len(train_rows)}."
        )
    selected_indices = sorted(random.Random(seed).sample(range(len(train_rows)), size))
    return [train_rows[index] for index in selected_indices]


def _set_rows_split(rows: list[dict[str, Any]], split: str) -> None:
    for row in rows:
        extra_info = row.setdefault("extra_info", {})
        if not isinstance(extra_info, dict):
            raise ValueError("Each row's extra_info must be a mapping.")
        extra_info["split"] = split

        tools_kwargs = extra_info.get("tools_kwargs")
        if not isinstance(tools_kwargs, dict):
            continue
        runtime_memory = tools_kwargs.get("runtime_memory")
        if not isinstance(runtime_memory, dict):
            continue
        create_kwargs = runtime_memory.get("create_kwargs")
        if not isinstance(create_kwargs, dict):
            continue
        metadata = create_kwargs.get("metadata")
        if isinstance(metadata, dict):
            metadata["split"] = split


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("Please install `datasets` in the verl environment to write parquet files.") from exc
    datasets.Dataset.from_list(rows).to_parquet(str(path))


def _validated_memory_ids(
    memory_items: Iterable[Mapping[str, Any]],
    field_name: str,
    *,
    require_unique: bool = False,
) -> list[str]:
    ids = [str(item.get("id") or "").strip() for item in memory_items]
    if any(not item_id for item_id in ids):
        raise ValueError(f"{field_name} items require non-empty IDs.")
    if require_unique and len(ids) != len(set(ids)):
        raise ValueError(f"{field_name} item IDs must be unique.")
    return ids
