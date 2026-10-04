"""Shared task rewards and resource diagnostics for memory QA."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from memory_access import MEMORY_TOOL_CALL_REQUIRED_FIELD
from runtime_metrics import (
    as_bool,
    extract_base_model_usage,
    extract_runtime_memory_metrics,
    summarize_runtime_memory_metrics,
)


def compose_memory_qa_reward(
    *,
    solution_str: str,
    extra_info: dict[str, Any] | None,
    answer_format_is_valid: bool,
    raw_performance_reward: float,
) -> dict[str, float | int]:
    """Apply the format gate and expose unweighted reward components."""
    extra_info = extra_info or {}
    tool_rewards = extract_tool_rewards(extra_info)
    external_api_cost = sum(max(0.0, -reward) for reward in tool_rewards)
    base_model_usage = extract_base_model_usage(extra_info)
    base_model_api_cost = float(base_model_usage["api_cost"])
    total_api_cost = external_api_cost + base_model_api_cost
    runtime_memory_calls = extract_runtime_memory_metrics(extra_info)
    runtime_summary = summarize_runtime_memory_metrics(runtime_memory_calls)
    external_estimated_latency = float(runtime_summary["runtime_estimated_latency"])
    base_model_estimated_latency = float(base_model_usage["estimated_latency"])
    total_estimated_latency = external_estimated_latency + base_model_estimated_latency
    format_status = _memory_qa_format_status(
        solution_str=solution_str,
        runtime_memory_calls=runtime_memory_calls,
        answer_format_is_valid=answer_format_is_valid,
        memory_tool_call_required=as_bool(
            extra_info.get(MEMORY_TOOL_CALL_REQUIRED_FIELD)
        ),
    )
    format_is_valid = format_status["format_is_valid"]

    format_reward = 0.0 if format_is_valid else -1.0
    performance_reward = float(raw_performance_reward) if format_is_valid else 0.0
    return {
        "score": format_reward + performance_reward,
        "format_reward": format_reward,
        # Counterfactual answer probes use task quality without the main
        # trajectory's tool/answer format gate.
        "raw_performance_reward": float(raw_performance_reward),
        "performance_reward": performance_reward,
        "total_api_cost": total_api_cost,
        "external_api_cost": external_api_cost,
        "base_model_api_cost": base_model_api_cost,
        "total_estimated_latency": total_estimated_latency,
        "external_estimated_latency": external_estimated_latency,
        "base_model_estimated_latency": base_model_estimated_latency,
        "base_model_num_calls": int(base_model_usage["num_calls"]),
        "base_model_input_tokens": int(base_model_usage["input_tokens"]),
        "base_model_output_tokens": int(base_model_usage["output_tokens"]),
        "base_model_total_tokens": int(base_model_usage["total_tokens"]),
        "num_memory_actions": len(tool_rewards),
        "format_valid": float(format_is_valid),
        "answer_format_valid": float(answer_format_is_valid),
        "tool_format_valid": float(format_status["tool_format_is_valid"]),
        "num_tool_format_errors": format_status["num_tool_format_errors"],
        **runtime_summary,
    }


def validate_memory_qa_trajectory_format(
    *,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
) -> bool:
    """Check the main trajectory's format without recomputing its task reward."""

    answer_format_is_valid = _answer_format_is_valid_for_ground_truth(
        solution_str,
        ground_truth,
    )
    status = _memory_qa_format_status(
        solution_str=solution_str,
        runtime_memory_calls=extract_runtime_memory_metrics(extra_info or {}),
        answer_format_is_valid=answer_format_is_valid,
        memory_tool_call_required=as_bool(
            (extra_info or {}).get(MEMORY_TOOL_CALL_REQUIRED_FIELD)
        ),
    )
    return bool(status["format_is_valid"])


def _memory_qa_format_status(
    *,
    solution_str: str,
    runtime_memory_calls: list[dict[str, Any]],
    answer_format_is_valid: bool,
    memory_tool_call_required: bool,
) -> dict[str, bool | int]:
    """Return the shared answer/tool format gate used by rollout and reward."""

    trajectory_tool_format_is_valid = validate_tool_call_format(
        solution_str,
        required=memory_tool_call_required,
    )
    num_runtime_tool_errors = sum(as_bool(call.get("error")) for call in runtime_memory_calls)
    num_tool_format_errors = num_runtime_tool_errors + int(not trajectory_tool_format_is_valid)
    tool_format_is_valid = num_tool_format_errors == 0
    return {
        "format_is_valid": bool(answer_format_is_valid and tool_format_is_valid),
        "tool_format_is_valid": tool_format_is_valid,
        "num_tool_format_errors": num_tool_format_errors,
    }


def _answer_format_is_valid_for_ground_truth(solution_str: str, ground_truth: Any) -> bool:
    """Mirror current free-text and MCQ answer-format contracts without scoring."""

    final_answer, answer_structure_is_valid = extract_tagged_answer(solution_str)
    if not answer_structure_is_valid or not final_answer:
        return False

    truth = _ground_truth_mapping(ground_truth)
    variant = str(truth.get("variant") or "").strip().casefold()
    if variant == "mcq":
        valid_choices = {
            str(choice).strip().upper()
            for choice in truth.get("valid_choices", [])
            if str(choice).strip()
        }
        prediction = final_answer.strip().upper()
        return bool(re.fullmatch(r"[A-Z]", prediction) and prediction in valid_choices)

    return True


def _ground_truth_mapping(ground_truth: Any) -> Mapping[str, Any]:
    if isinstance(ground_truth, Mapping):
        return ground_truth
    try:
        decoded = json.loads(str(ground_truth))
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, Mapping) else {}


def extract_tool_rewards(extra_info: dict[str, Any]) -> list[float]:
    """Read verl's flat per-call reward list from tool_extra_fields."""
    values = extra_info.get("tool_rewards", [])
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        return []

    rewards = []
    for value in values:
        try:
            rewards.append(float(value))
        except (TypeError, ValueError):
            continue
    return rewards


def extract_tagged_answer(text: str) -> tuple[str | None, bool]:
    """Return one strictly formatted final policy answer, excluding tool observations."""
    policy_text = _without_tool_responses(text)
    answer_markers = re.findall(r"</?answer\b", policy_text, flags=re.IGNORECASE)
    answer_blocks = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", policy_text, flags=re.DOTALL))
    if len(answer_markers) != 2 or len(answer_blocks) != 1:
        return None, False

    final_text = final_policy_segment(text)
    final_blocks = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", final_text, flags=re.DOTALL))
    if len(final_blocks) != 1 or final_text[final_blocks[0].end() :].strip():
        return None, False
    return " ".join(final_blocks[0].group(1).split()), True


def parse_free_text_ground_truth(ground_truth: Any) -> list[str]:
    """Parse the shared {"answers": [...]} contract, with a plain-text fallback."""
    data = ground_truth
    if not isinstance(data, (dict, list, tuple)):
        try:
            data = json.loads(str(ground_truth))
        except (json.JSONDecodeError, TypeError):
            data = ground_truth
    if isinstance(data, dict):
        data = data.get("answers", [])
    if not isinstance(data, (list, tuple)):
        data = [data]
    return [str(answer).strip() for answer in data if answer is not None and str(answer).strip()]


def final_policy_segment(text: str) -> str:
    """Exclude tool observations before parsing the policy's final answer."""
    text = str(text)
    tool_response_ends = list(re.finditer(r"</tool_response>", text, flags=re.IGNORECASE))
    if not tool_response_ends:
        return text
    return text[tool_response_ends[-1].end() :]


def validate_tool_call_format(text: str, *, required: bool = False) -> bool:
    """Validate policy-emitted runtime-memory tool-call markup and JSON arguments."""
    policy_text = _without_tool_responses(text)
    tool_markers = re.findall(r"</?tool_call\b", policy_text, flags=re.IGNORECASE)
    tool_blocks = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", policy_text, flags=re.DOTALL))
    if (required and not tool_blocks) or len(tool_markers) != 2 * len(tool_blocks):
        return False

    for block in tool_blocks:
        try:
            payload = json.loads(block.group(1))
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(payload, dict):
            return False
        tool_name = payload.get("name")
        if tool_name not in {"retrieve_memory", "runtime_memory"}:
            return False
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            return False
        # Mirror the two tool schemas. Route/ID/count/length failures are
        # execution errors and arrive through runtime metrics.
        if not str(arguments.get("retrieval_query", "")).strip():
            return False
        if tool_name == "runtime_memory" and (
            not str(arguments.get("instruction", "")).strip()
            or not str(arguments.get("model", "")).strip()
        ):
            return False
    return True


def _without_tool_responses(text: str) -> str:
    """Remove environment-authored tool observations before policy-format checks."""
    return re.sub(
        r"<tool_response>.*?</tool_response>",
        "",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
