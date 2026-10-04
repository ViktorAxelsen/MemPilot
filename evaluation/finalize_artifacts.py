"""Normalize saved Verl responses and compute local/resource metrics post hoc."""

from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.judge_protocols import (
    get_judge_protocol,
    judge_protocol_id_for_sample,
)
from evaluation.local_metrics import compute_local_answer_metrics
from rewards.memory_qa import extract_tagged_answer
from resource_profiles import (
    DEFAULT_RESOURCE_CONFIG_PATH,
    ResourceProfile,
    ResourceProfileCatalog,
    load_resource_profile_catalog,
)
from runtime_metrics import (
    as_bool,
    runtime_memory_critical_path_latency,
    summarize_runtime_memory_metrics,
)


SCHEMA_VERSION = "official.final_eval.v10"
# Internal token-price sums are USD times 1e6; convert once at reporting.
RAW_COST_TO_USD = 1_000_000.0

AGGREGATION_PROTOCOL = {
    "primary_reporting_scope": "by_dataset",
    "overall_scope": "diagnostic_only",
    "sample_identity": "unique (dataset_source, conversation_id)",
    "f1": "query_mean",
    "llm_judge": "query_mean",
    "cost_qa": "sum_over_queries_divided_by_samples",
    "cost_bank": "sample_mean_not_recorded",
    "latency_qa": "query_mean",
    "decoupled_answer_model": (
        "actual_usage_excluded; its resource profile prices the saved policy token counts and generation segments"
    ),
    "dataset_reporting_groups": {
        "h2hmem": ["h2hmem_dyadic", "h2hmem_multiparty"],
    },
}


def finalize_validation_dump(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    resource_config_path: str | Path = DEFAULT_RESOURCE_CONFIG_PATH,
) -> dict[str, Any]:
    """Normalize one saved response dump without invoking an LLM judge."""

    raw_file, candidates = select_validation_dump(input_path)
    raw_rows = select_final_session_rows(list(read_jsonl(raw_file)))
    if not raw_rows:
        raise ValueError(f"Validation dump is empty: {raw_file}")
    resource_profiles = load_resource_profile_catalog(resource_config_path)
    normalized = [
        normalize_validation_row(
            row,
            raw_file=raw_file,
            resource_profiles=resource_profiles,
        )
        for row in raw_rows
    ]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    responses_path = destination / "responses.jsonl"
    local_metrics_path = destination / "local_metrics.json"
    manifest_path = destination / "manifest.json"
    local_metrics = aggregate_predictions(normalized)
    write_jsonl_atomic(responses_path, normalized)
    write_json_atomic(local_metrics_path, local_metrics)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_raw_response_dump": str(raw_file),
        "candidate_raw_response_dumps": [str(path) for path in candidates],
        "responses_file": str(responses_path),
        "local_metrics_file": str(local_metrics_path),
        "num_responses": len(normalized),
        "num_queries": len(normalized),
        "num_samples": local_metrics["num_samples"],
        "judge_status": "pending",
        "resource_profile_config": str(resource_profiles.config_path),
        "resource_profile_config_sha256": resource_profiles.config_sha256,
        "cost_accounting": (
            "Saved policy and routed-model token counts multiplied by the current USD-per-million rates in "
            "runtime_memory_tool.yaml, then divided by "
            "1,000,000 for the reported USD value. QA cost is summed across questions and divided by the "
            "number of unique conversations; memory-bank construction cost is not yet recorded. When a common "
            "answer model is used, its current YAML price profile is applied to the saved policy input/output "
            "token counts, "
            "while the answer model's actual replacement-generation usage is excluded. Judge API usage is also "
            "excluded. Prices and provider-reported request costs cached in the raw trace are audit-only."
        ),
        "latency_accounting": (
            "Current runtime_memory_tool.yaml profiles are applied to saved token, image, and generation-stage "
            "counts. Routed calls in one parallel stage use the stage maximum. When a common answer model is "
            "used, its YAML latency profile is applied to the saved policy token counts and generation-segment "
            "count. Latency is averaged over questions; cached trace estimates, answer-model replacement "
            "latency, and judge wall time are excluded."
        ),
        "aggregation_protocol": AGGREGATION_PROTOCOL,
    }
    write_json_atomic(manifest_path, manifest)
    return {
        "responses_path": responses_path,
        "local_metrics_path": local_metrics_path,
        "manifest_path": manifest_path,
        "num_responses": len(normalized),
        "selected_raw_file": raw_file,
    }


def select_final_session_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the highest output index from each multi-turn Verl session."""

    selected: dict[str, tuple[int, int, dict[str, Any]]] = {}
    for position, raw_row in enumerate(rows):
        row = dict(raw_row)
        uid = str(row.get("uid") or "").strip()
        source = str(row.get("dataset_source") or "").strip()
        parsed = _parse_session_uid(uid)
        if parsed is None:
            session_key, output_index = f"{source}:plain:{uid or position}", 0
        else:
            parsed_key, output_index = parsed
            session_key = f"{source}:{parsed_key}"
        previous = selected.get(session_key)
        if previous is None or output_index > previous[0]:
            selected[session_key] = (output_index, position, row)
    return [item[2] for item in sorted(selected.values(), key=lambda item: item[1])]


def normalize_validation_row(
    row: Mapping[str, Any],
    *,
    raw_file: str | Path | None = None,
    resource_profiles: ResourceProfileCatalog | None = None,
) -> dict[str, Any]:
    """Preserve the full trajectory and derive all non-judge metrics from it."""

    input_text = str(row.get("input") or "")
    response = str(row.get("output") or "")
    if not input_text:
        raise ValueError("Every saved evaluation row must contain the complete input prompt.")
    dataset_source = str(row.get("dataset_source") or "").strip()
    if not dataset_source:
        raise ValueError("Capture metadata did not provide dataset_source.")
    metadata = _parse_json_object(row.get("sample_metadata_json"), "sample_metadata_json")
    runtime_trace = _parse_json_object(row.get("runtime_trace_json"), "runtime_trace_json")
    ground_truth = parse_ground_truth(row.get("gts"))
    references = ground_truth_references(ground_truth)
    if not references:
        raise ValueError(f"Evaluation row {row.get('uid')!r} has no reference answer.")
    conversation_id = str(metadata.get("conversation_id") or "").strip()
    if not conversation_id:
        raise ValueError(
            f"Evaluation row {row.get('uid')!r} has no conversation_id; sample-wise cost "
            "aggregation would be ambiguous."
        )

    prediction, answer_format_valid = extract_tagged_answer(response)
    prediction = str(prediction or "")
    local_metrics = compute_local_answer_metrics(
        dataset_source=dataset_source,
        prediction=prediction,
        references=references,
        metadata=metadata,
        ground_truth=ground_truth,
    )
    local_metrics["answer_format_valid"] = float(answer_format_valid)
    protocol_id = judge_protocol_id_for_sample(dataset_source, metadata, ground_truth)
    protocol = get_judge_protocol(protocol_id)
    if not protocol.requires_llm_judge:
        raise ValueError(f"Final protocol {protocol_id!r} unexpectedly has no LLM judge.")

    return {
        "schema_version": SCHEMA_VERSION,
        "uid": str(row.get("uid") or ""),
        "dataset_source": dataset_source,
        "split": str(metadata.get("split") or ""),
        "conversation_id": conversation_id,
        "question_id": metadata.get("question_id"),
        "question_type": (
            metadata.get("question_type")
            or metadata.get("question_type_name")
            or metadata.get("question_type_main")
        ),
        "question_subtype": metadata.get("question_subtype"),
        "question": metadata.get("question"),
        "input": input_text,
        "response": response,
        "prediction": prediction,
        "answer_format_valid": bool(answer_format_valid),
        "ground_truth": ground_truth,
        "references": references,
        "sample_metadata": metadata,
        "local_metrics": local_metrics,
        "resource_metrics": compute_resource_metrics(
            runtime_trace,
            resource_profiles=resource_profiles,
        ),
        "runtime_trace": runtime_trace,
        "judge": {
            "protocol_id": protocol_id,
            "status": "pending",
            "score": None,
            "label": None,
            "reasoning": None,
        },
        "raw_validation_step": row.get("step"),
        "raw_validation_file": str(raw_file) if raw_file is not None else None,
    }


def compute_resource_metrics(
    runtime_trace: Mapping[str, Any],
    *,
    resource_profiles: ResourceProfileCatalog | None = None,
) -> dict[str, Any]:
    """Recompute resources using the current static-profile reporting convention.

    When answer replacement is present, its model profile prices the saved
    policy token counts and generation segments. The replacement generation's
    own usage is retained for audit but excluded from reported totals. External
    curation calls are priced separately; latency is a deterministic proxy.
    """

    profiles = resource_profiles or load_resource_profile_catalog()

    base_usage = runtime_trace.get("base_model")
    if not isinstance(base_usage, Mapping):
        raise ValueError("Runtime trace is missing base_model usage.")
    base_model_num_calls = _nonnegative_int(base_usage.get("num_calls"))
    if base_model_num_calls == 0:
        raise ValueError(
            "Base-model cost/latency accounting is missing. Use rollout_ordered_tool_agent "
            "for final response capture."
        )
    raw_calls = runtime_trace.get("memory_calls")
    if not isinstance(raw_calls, list) or any(not isinstance(call, Mapping) for call in raw_calls):
        raise ValueError("Runtime trace memory_calls must be a list of objects.")
    calls = [dict(call) for call in raw_calls]
    repriced_calls = _reprice_external_calls(calls, profiles)
    summary = summarize_runtime_memory_metrics(repriced_calls)

    answer_usage = runtime_trace.get("answer_model")
    if answer_usage is not None and not isinstance(answer_usage, Mapping):
        raise ValueError("Runtime trace answer_model must be an object when present.")
    answer_usage = dict(answer_usage or {})
    answer_num_calls = _nonnegative_int(answer_usage.get("num_calls"))
    use_answer_profile = bool(answer_usage)
    policy_profile, policy_profile_model, policy_profile_source = _policy_profile(
        base_usage=base_usage,
        answer_usage=answer_usage,
        profiles=profiles,
    )
    cost_components = _trajectory_cost_components(
        base_usage=base_usage,
        calls=calls,
        profiles=profiles,
        policy_profile=policy_profile,
        policy_profile_model=policy_profile_model,
    )
    base_raw_cost = sum(
        float(component["raw_token_price_cost"])
        for component in cost_components
        if component["role"] == "policy"
    )
    external_raw_cost = sum(
        float(component["raw_token_price_cost"])
        for component in cost_components
        if component["role"] == "external"
    )
    answer_profile = (
        profiles.resolve(str(answer_usage.get("model") or ""), role="answer-model")
        if use_answer_profile
        else None
    )
    answer_cost_component = (
        _cost_component(
            role="answer",
            model=str(answer_usage.get("model") or "answer_model"),
            usage=answer_usage,
            profile=answer_profile,
            raw_cost=answer_usage.get("api_cost"),
            input_token_key="input_tokens",
            output_token_key="output_tokens",
        )
        if answer_num_calls
        else None
    )
    answer_raw_cost = (
        float(answer_cost_component["raw_token_price_cost"])
        if answer_cost_component is not None
        else 0.0
    )
    total_raw_cost = base_raw_cost + external_raw_cost
    trace_base_latency = _nonnegative_float(base_usage.get("estimated_latency"))
    base_latency = policy_profile.estimated_latency(
        num_calls=base_model_num_calls,
        input_tokens=_nonnegative_int(base_usage.get("input_tokens")),
        output_tokens=_nonnegative_int(base_usage.get("output_tokens")),
        num_images=_nonnegative_int(base_usage.get("num_input_images")),
    )
    trace_external_latency = runtime_memory_critical_path_latency(calls)
    external_latency = runtime_memory_critical_path_latency(repriced_calls)
    answer_trace_latency = _nonnegative_float(answer_usage.get("estimated_latency"))
    answer_latency = (
        answer_profile.estimated_latency(
            num_calls=answer_num_calls,
            input_tokens=_nonnegative_int(answer_usage.get("input_tokens")),
            output_tokens=_nonnegative_int(answer_usage.get("output_tokens")),
            num_images=_nonnegative_int(answer_usage.get("num_input_images")),
        )
        if answer_profile is not None
        else 0.0
    )
    return {
        "total_api_cost_raw": total_raw_cost,
        "external_api_cost_raw": external_raw_cost,
        "base_model_api_cost_raw": base_raw_cost,
        "answer_model_api_cost_raw": answer_raw_cost,
        "cost_accounting_source": "runtime_memory_yaml_static_profiles",
        "estimated_api_cost_usd": total_raw_cost / RAW_COST_TO_USD,
        "external_estimated_api_cost_usd": external_raw_cost / RAW_COST_TO_USD,
        "base_model_estimated_api_cost_usd": base_raw_cost / RAW_COST_TO_USD,
        "answer_model_estimated_api_cost_usd": answer_raw_cost / RAW_COST_TO_USD,
        "cost_components": cost_components,
        "answer_model_audit_cost_component": answer_cost_component,
        "answer_model_actual_usage_excluded_from_reported_resources": True,
        "policy_resource_profile_source": policy_profile_source,
        "policy_resource_profile_model": policy_profile_model,
        "policy_resource_profile_canonical_model": policy_profile.model,
        "resource_profile_config": str(profiles.config_path),
        "resource_profile_config_sha256": profiles.config_sha256,
        "base_model_trace_api_cost_raw": _nonnegative_float(base_usage.get("api_cost")),
        "total_estimated_latency_seconds": base_latency + external_latency,
        "external_estimated_latency_seconds": external_latency,
        "base_model_estimated_latency_seconds": base_latency,
        "base_model_trace_estimated_latency_seconds": trace_base_latency,
        "external_trace_estimated_latency_seconds": trace_external_latency,
        "answer_model_estimated_latency_seconds": answer_latency,
        "answer_model_trace_estimated_latency_seconds": answer_trace_latency,
        "base_model_input_tokens": _nonnegative_int(base_usage.get("input_tokens")),
        "base_model_output_tokens": _nonnegative_int(base_usage.get("output_tokens")),
        "base_model_total_tokens": _nonnegative_int(base_usage.get("total_tokens")),
        "answer_model_input_tokens": _nonnegative_int(answer_usage.get("input_tokens")),
        "answer_model_output_tokens": _nonnegative_int(answer_usage.get("output_tokens")),
        "answer_model_total_tokens": _nonnegative_int(answer_usage.get("total_tokens")),
        **{
            key: summary[key]
            for key in (
                "runtime_prompt_tokens",
                "runtime_completion_tokens",
                "runtime_total_tokens",
                "runtime_input_images",
                "runtime_estimated_latency",
            )
        },
    }


def _trajectory_cost_components(
    *,
    base_usage: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
    profiles: ResourceProfileCatalog,
    policy_profile: ResourceProfile,
    policy_profile_model: str,
) -> list[dict[str, Any]]:
    """Retain token/price terms needed to audit the reported USD cost."""

    policy_component = _cost_component(
        role="policy",
        model=policy_profile_model,
        usage=base_usage,
        profile=policy_profile,
        raw_cost=base_usage.get("api_cost"),
        input_token_key="input_tokens",
        output_token_key="output_tokens",
    )
    policy_component["token_source_model"] = str(
        base_usage.get("model") or "policy_model"
    )
    policy_component["requested_resource_profile_model"] = policy_profile_model
    components = [policy_component]
    for call in calls:
        if not _external_call_attempted(call):
            continue
        profile = profiles.resolve_external_call(call)
        components.append(
            _cost_component(
                role="external",
                model=profile.model,
                usage=call,
                profile=profile,
                raw_cost=call.get("memory_api_cost"),
                input_token_key="prompt_tokens",
                output_token_key="completion_tokens",
            )
        )
    return components


def _cost_component(
    *,
    role: str,
    model: str,
    usage: Mapping[str, Any],
    profile: ResourceProfile | None,
    raw_cost: Any,
    input_token_key: str,
    output_token_key: str,
) -> dict[str, Any]:
    if not _is_number(usage.get(input_token_key)) or not _is_number(
        usage.get(output_token_key)
    ):
        raise ValueError(
            f"Final static cost accounting is missing token counts for {role} model {model!r}."
        )
    input_tokens = _nonnegative_int(usage.get(input_token_key))
    output_tokens = _nonnegative_int(usage.get(output_token_key))
    trace_reported_raw_cost = _nonnegative_float(raw_cost)
    if profile is None:
        raise ValueError(f"No YAML resource profile for {role} model {model!r}.")
    input_price = profile.input_price_per_million_usd
    output_price = profile.output_price_per_million_usd
    static_raw_cost = input_tokens * input_price + output_tokens * output_price
    return {
        "role": role,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price_per_million_usd": input_price,
        "output_price_per_million_usd": output_price,
        "raw_token_price_cost": static_raw_cost,
        "estimated_cost_usd": static_raw_cost / RAW_COST_TO_USD,
        "pricing_recorded": True,
        "cost_source": "runtime_memory_yaml_static_profile",
        "resource_profile_model": profile.model,
        "resource_profile_route": profile.route_name,
        "trace_reported_raw_cost": trace_reported_raw_cost,
        "trace_input_price_per_million_usd": (
            _nonnegative_float(usage.get("input_price_per_million_usd"))
            if _is_number(usage.get("input_price_per_million_usd"))
            else None
        ),
        "trace_output_price_per_million_usd": (
            _nonnegative_float(usage.get("output_price_per_million_usd"))
            if _is_number(usage.get("output_price_per_million_usd"))
            else None
        ),
    }


def _policy_profile(
    *,
    base_usage: Mapping[str, Any],
    answer_usage: Mapping[str, Any],
    profiles: ResourceProfileCatalog,
) -> tuple[ResourceProfile, str, str]:
    if answer_usage:
        identifier = str(answer_usage.get("model") or "").strip()
        if not identifier:
            raise ValueError("Runtime trace answer_model is missing its model identifier.")
        return profiles.resolve(identifier, role="answer-model"), identifier, "answer_model"

    recorded = str(base_usage.get("model") or "").strip()
    if recorded and recorded.casefold() not in {"policy_model", "base_model"}:
        return profiles.resolve(recorded, role="policy-model"), recorded, "policy_trace_model"
    identifier = profiles.default_policy_model
    return profiles.resolve(identifier, role="default-policy"), identifier, "yaml_default_policy_model"


def _reprice_external_calls(
    calls: Sequence[Mapping[str, Any]],
    profiles: ResourceProfileCatalog,
) -> list[dict[str, Any]]:
    """Replace cached external latency estimates with current YAML estimates."""

    output: list[dict[str, Any]] = []
    for call in calls:
        updated = dict(call)
        updated["trace_memory_estimated_latency"] = _nonnegative_float(
            call.get("memory_estimated_latency")
        )
        if _external_call_attempted(call):
            profile = profiles.resolve_external_call(call)
            updated["memory_estimated_latency"] = profile.estimated_latency(
                num_calls=1,
                input_tokens=_nonnegative_int(call.get("prompt_tokens")),
                output_tokens=_nonnegative_int(call.get("completion_tokens")),
                num_images=_nonnegative_int(call.get("num_input_images")),
            )
            updated["resource_profile_model"] = profile.model
            updated["resource_profile_route"] = profile.route_name
        else:
            updated["memory_estimated_latency"] = 0.0
        output.append(updated)
    return output


def aggregate_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate by dataset/type with metric-specific denominators.

    Quality and latency are question means. QA cost is summed over questions,
    then divided by unique dataset-conversation samples.
    Offline memory-bank cost is not recorded by these inference traces.
    """

    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("dataset_source") or "unknown")
        reporting_source = _reporting_dataset_source(source)
        by_dataset[reporting_source].append(row)
        question_type = str(row.get("question_type") or "unknown")
        subtype = str(row.get("question_subtype") or "").strip()
        key = f"{reporting_source}/{question_type}" + (f"/{subtype}" if subtype else "")
        by_type[key].append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "num_responses": len(rows),
        "num_queries": len(rows),
        "num_samples": _sample_count(rows),
        "aggregation_protocol": AGGREGATION_PROTOCOL,
        "overall": _aggregate_group(rows),
        "by_dataset": {
            key: _aggregate_group(value) for key, value in sorted(by_dataset.items())
        },
        "by_dataset_and_question_type": {
            key: _aggregate_group(value) for key, value in sorted(by_type.items())
        },
    }


def _reporting_dataset_source(dataset_source: str) -> str:
    """Merge benchmark subsets that share one paper-level reporting group."""

    source = str(dataset_source or "unknown").strip()
    normalized = source.casefold().replace("-", "_")
    if normalized in {"h2hmem", "h2hmem_dyadic", "h2hmem_multiparty"}:
        return "h2hmem"
    return source


def format_evaluation_report(metrics: Mapping[str, Any]) -> str:
    """Render dataset-wise final metrics with auditable resource formulas."""

    sections: list[str] = []
    by_dataset = metrics.get("by_dataset")
    if isinstance(by_dataset, Mapping):
        for dataset, group in by_dataset.items():
            if isinstance(group, Mapping):
                sections.append(_format_metric_group(str(dataset), group))
    overall = metrics.get("overall")
    if isinstance(overall, Mapping):
        sections.append(_format_metric_group("Overall (diagnostic only)", overall))
    return "\n\n".join(sections)


def _format_metric_group(name: str, group: Mapping[str, Any]) -> str:
    num_queries = _nonnegative_int(group.get("num_queries"))
    num_samples = _nonnegative_int(group.get("num_samples"))
    lines = [f"=== {name} ===", f"Queries: {num_queries}", f"Samples: {num_samples}"]
    judge_complete = _nonnegative_int(group.get("llm_judge_complete_count"))
    lines.extend(
        (
            f"F1: {_format_metric_mean(group.get('f1'))}",
            f"Judge: {_format_metric_mean(group.get('llm_judge_score'))} "
            f"({judge_complete}/{num_queries} successful)",
        )
    )

    cost_bank = group.get("cost_bank_usd")
    lines.append("Cost-MB: N/A")
    if isinstance(cost_bank, Mapping):
        lines.append(
            "  [sum of memory-bank token-price costs / 1,000,000] / "
            f"{_nonnegative_int((cost_bank.get('formula') or {}).get('sample_divisor'))} "
            "= N/A (not recorded)"
        )

    cost_qa = group.get("cost_qa_usd")
    lines.append(f"Cost-QA: {_format_cost_value(cost_qa)}")
    lines.extend(_format_cost_formula(cost_qa))

    latency = group.get("latency_qa_seconds")
    if isinstance(latency, Mapping):
        lines.append(
            f"Latency: {_format_number(latency.get('mean'), digits=4)} s/query"
        )
    else:
        lines.append("Latency: N/A")
    return "\n".join(lines)


def _format_metric_mean(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "N/A"
    return _format_number(value.get("mean"), digits=4)


def _format_cost_value(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "N/A"
    return f"{_format_number(value.get('mean_per_sample'), digits=8)} USD/sample"


def _format_cost_formula(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["  formula unavailable"]
    formula = value.get("formula")
    if not isinstance(formula, Mapping):
        return ["  formula unavailable"]
    divisor = _nonnegative_int(formula.get("sample_divisor"))
    result = _format_number(value.get("mean_per_sample"), digits=8)
    raw_sum = formula.get("raw_token_price_sum")
    if _is_number(raw_sum):
        return [
            "  [sum_m(input_tokens_m*input_price_m + output_tokens_m*output_price_m) "
            f"= {_format_number(raw_sum, digits=6)}] / 1,000,000 / {divisor} "
            f"= {result} USD/sample"
        ]

    total_usd = _format_number(value.get("sum"), digits=8)
    return [
        f"  {total_usd} USD / {divisor} = {result} USD/sample"
    ]


def _format_number(value: Any, *, digits: int) -> str:
    if not _is_number(value) or not math.isfinite(float(value)):
        return "N/A"
    number = float(value)
    if digits == 0:
        return str(int(round(number))) if math.isclose(number, round(number), abs_tol=1e-9) else f"{number:g}"
    return f"{number:.{digits}f}"


def _aggregate_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    num_queries = len(rows)
    num_samples = _sample_count(rows)
    output: dict[str, Any] = {
        "count": num_queries,
        "num_queries": num_queries,
        "num_samples": num_samples,
    }
    for key in ("f1", "answer_format_valid"):
        values = _metric_values(rows, "local_metrics", key)
        if values:
            output[key] = _query_mean(values)

    cost_values = _required_metric_values(rows, "resource_metrics", "estimated_api_cost_usd")
    latency_values = _required_metric_values(
        rows, "resource_metrics", "total_estimated_latency_seconds"
    )
    cost_qa = _sample_mean_of_query_sum(cost_values, num_samples)
    cost_qa.update(
        {
            "unit": "USD/sample",
            "formula": _aggregate_cost_formula(rows, cost_values, num_samples),
        }
    )
    output["cost_qa_usd"] = cost_qa
    output["cost_bank_usd"] = _unavailable_bank_cost(num_samples)
    latency_qa = _query_mean(latency_values)
    latency_qa.update(
        {
            "unit": "seconds/query",
            "formula": {
                "expression": "sum_query_latency_seconds / num_queries",
                "total_latency_seconds": sum(latency_values),
                "query_divisor": len(latency_values),
            },
        }
    )
    output["latency_qa_seconds"] = latency_qa

    for key in (
        "base_model_input_tokens",
        "base_model_output_tokens",
        "answer_model_input_tokens",
        "answer_model_output_tokens",
        "runtime_prompt_tokens",
        "runtime_completion_tokens",
        "runtime_input_images",
    ):
        values = _metric_values(rows, "resource_metrics", key)
        if values:
            output[key] = _query_mean(values)
    judge_scores = [
        float((row.get("judge") or {})["score"])
        for row in rows
        if _is_number((row.get("judge") or {}).get("score"))
    ]
    output["llm_judge_complete_count"] = sum(
        (row.get("judge") or {}).get("status") == "complete" for row in rows
    )
    output["llm_judge_pending_count"] = sum(
        (row.get("judge") or {}).get("status") == "pending" for row in rows
    )
    output["llm_judge_error_count"] = sum(
        (row.get("judge") or {}).get("status") == "error" for row in rows
    )
    if judge_scores:
        output["llm_judge_score"] = _query_mean(judge_scores)
    return output


def _query_mean(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": sum(values) / len(values),
        "sum": sum(values),
        "aggregation": "query_mean",
        "applicable_query_count": len(values),
        "applicable_count": len(values),
    }


def _aggregate_cost_formula(
    rows: Sequence[Mapping[str, Any]],
    cost_values: Sequence[float],
    num_samples: int,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    all_rows_auditable = True
    for row in rows:
        resource_metrics = row.get("resource_metrics")
        components = (
            resource_metrics.get("cost_components")
            if isinstance(resource_metrics, Mapping)
            else None
        )
        if not isinstance(components, list):
            all_rows_auditable = False
            continue
        for component in components:
            if not isinstance(component, Mapping):
                all_rows_auditable = False
                continue
            pricing_recorded = bool(component.get("pricing_recorded"))
            input_price = (
                _nonnegative_float(component.get("input_price_per_million_usd"))
                if pricing_recorded
                else None
            )
            output_price = (
                _nonnegative_float(component.get("output_price_per_million_usd"))
                if pricing_recorded
                else None
            )
            key = (
                str(component.get("model") or "unknown"),
                pricing_recorded,
                input_price,
                output_price,
            )
            aggregate = grouped.setdefault(
                key,
                {
                    "model": key[0],
                    "pricing_recorded": pricing_recorded,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "raw_token_price_cost": 0.0,
                    "estimated_cost_usd": 0.0,
                    **(
                        {
                            "input_price_per_million_usd": input_price,
                            "output_price_per_million_usd": output_price,
                        }
                        if pricing_recorded
                        else {}
                    ),
                },
            )
            aggregate["input_tokens"] += _nonnegative_int(component.get("input_tokens"))
            aggregate["output_tokens"] += _nonnegative_int(component.get("output_tokens"))
            aggregate["raw_token_price_cost"] += _nonnegative_float(
                component.get("raw_token_price_cost")
            )
            aggregate["estimated_cost_usd"] += _nonnegative_float(
                component.get("estimated_cost_usd")
            )

    aggregated_components = sorted(
        grouped.values(),
        key=lambda item: str(item["model"]),
    )
    total_usd = sum(float(value) for value in cost_values)
    component_raw_total = sum(
        float(component["raw_token_price_cost"])
        for component in aggregated_components
    )
    if all_rows_auditable and not math.isclose(
        component_raw_total / RAW_COST_TO_USD,
        total_usd,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Aggregated cost components disagree with the recorded QA cost: "
            f"{component_raw_total / RAW_COST_TO_USD} != {total_usd}."
        )
    return {
        "expression": (
            "sum_m(input_tokens_m * input_price_per_million_m + output_tokens_m * "
            "output_price_per_million_m) / (1_000_000 * num_samples)"
        ),
        "price_unit": "USD per million tokens",
        "usd_conversion_divisor": int(RAW_COST_TO_USD),
        "sample_divisor": num_samples,
        "raw_token_price_sum": component_raw_total if all_rows_auditable else None,
        "total_cost_usd": total_usd,
        "result_usd_per_sample": total_usd / num_samples,
        "components": aggregated_components,
        "status": "complete" if all_rows_auditable else "partial_legacy_trace",
    }


def _sample_mean_of_query_sum(values: Sequence[float], num_samples: int) -> dict[str, Any]:
    if num_samples <= 0:
        raise ValueError("Sample-wise aggregation requires at least one conversation.")
    total = sum(values)
    return {
        "mean_per_sample": total / num_samples,
        "sum": total,
        "aggregation": "sum_over_queries_divided_by_samples",
        "sample_count": num_samples,
        "query_count": len(values),
    }


def _unavailable_bank_cost(sample_count: int) -> dict[str, Any]:
    return {
        "mean_per_sample": None,
        "sum": None,
        "aggregation": "sample_mean",
        "status": "not_recorded",
        "unit": "USD/sample",
        "formula": {
            "expression": (
                "sum_m(input_tokens_m * input_price_per_million_m + output_tokens_m * "
                "output_price_per_million_m) / (1_000_000 * num_samples)"
            ),
            "sample_divisor": sample_count,
            "status": "not_recorded",
        },
    }


def _sample_count(rows: Sequence[Mapping[str, Any]]) -> int:
    sample_keys: set[tuple[str, str]] = set()
    for row in rows:
        dataset_source = str(row.get("dataset_source") or "").strip()
        conversation_id = str(row.get("conversation_id") or "").strip()
        if not dataset_source or not conversation_id:
            raise ValueError(
                "Sample-wise aggregation requires non-empty dataset_source and conversation_id on every row."
            )
        sample_keys.add((dataset_source, conversation_id))
    return len(sample_keys)


def select_validation_dump(input_path: str | Path) -> tuple[Path, list[Path]]:
    """Select the highest-step JSONL dump, or use the given JSONL file."""

    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.casefold() != ".jsonl":
            raise ValueError(f"Saved response dump must be JSONL: {path}")
        return path, [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Saved response path does not exist: {path}")
    candidates = sorted(path.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No saved response JSONL files found in {path}")

    def sort_key(candidate: Path) -> tuple[int, float, str]:
        try:
            step = int(candidate.stem)
        except ValueError:
            step = -1
        return step, candidate.stat().st_mtime, candidate.name

    return max(candidates, key=sort_key), candidates


def _parse_session_uid(uid: str) -> tuple[str, int] | None:
    parts = uid.rsplit("_", 2)
    if len(parts) != 3:
        return None
    try:
        output_index = int(parts[2])
    except ValueError:
        return None
    return f"{parts[0]}_{parts[1]}", output_index


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            yield value


def parse_ground_truth(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return _json_safe(value)
    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return str(value or "")


def ground_truth_references(ground_truth: Any) -> list[str]:
    values: Any = ground_truth
    if isinstance(ground_truth, Mapping):
        values = ground_truth.get("answers")
        if values is None and ground_truth.get("answer") is not None:
            values = [ground_truth.get("answer")]
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [str(value).strip() for value in values if value is not None and str(value).strip()]


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(str(text).rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".eval.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    return Path(name)


def _parse_json_object(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value in (None, ""):
        raise ValueError(f"Saved response row is missing {name}.")
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Saved response row contains invalid {name}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"Saved response row {name} must decode to an object.")
    return decoded


def _metric_values(
    rows: Sequence[Mapping[str, Any]],
    container_name: str,
    key: str,
) -> list[float]:
    values = []
    for row in rows:
        container = row.get(container_name)
        value = container.get(key) if isinstance(container, Mapping) else None
        if _is_number(value) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _required_metric_values(
    rows: Sequence[Mapping[str, Any]],
    container_name: str,
    key: str,
) -> list[float]:
    values = _metric_values(rows, container_name, key)
    if len(values) != len(rows):
        raise ValueError(
            f"Every query requires numeric {container_name}.{key}; found {len(values)} values "
            f"for {len(rows)} queries."
        )
    return values


def _nonnegative_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        _warn_sanitized_metric("integer", value, 0, "not an integer")
        return 0
    if isinstance(value, bool):
        _warn_sanitized_metric("integer", value, number, "boolean value")
    else:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float(number)
        if math.isfinite(numeric_value) and numeric_value != number:
            _warn_sanitized_metric(
                "integer", value, number, "fractional value was truncated"
            )
    if number < 0:
        _warn_sanitized_metric("integer", value, 0, "negative value")
        return 0
    return number


def _nonnegative_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        _warn_sanitized_metric("numeric", value, 0.0, "not numeric")
        return 0.0
    if not math.isfinite(number):
        _warn_sanitized_metric("numeric", value, 0.0, "non-finite value")
        return 0.0
    if number < 0.0:
        _warn_sanitized_metric("numeric", value, 0.0, "negative value")
        return 0.0
    return number


def _warn_sanitized_metric(
    expected_type: str,
    observed: Any,
    replacement: int | float,
    reason: str,
) -> None:
    warnings.warn(
        "[FINAL EVALUATION DATA WARNING] "
        f"Invalid non-negative {expected_type} value {observed!r} ({reason}); "
        f"using {replacement!r}. Reported resource metrics may be underestimated.",
        RuntimeWarning,
        stacklevel=3,
    )


def _external_call_attempted(call: Mapping[str, Any]) -> bool:
    if "external_call_attempted" in call:
        return as_bool(call.get("external_call_attempted"))
    return not as_bool(call.get("direct_memory_return"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
