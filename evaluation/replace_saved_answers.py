"""Policy-answer model decoupling for MemPilot's controlled evaluation.

This is an optional post-hoc stage between response capture and
``evaluate_saved_responses``. It preserves the saved policy trajectory for
audit, removes its final ``<answer>`` block, and asks a separately specified
model to answer from the structured benchmark question plus the memory evidence
returned by policy tool calls. Policy reasoning, routing arguments, the original
final answer, and ground-truth references are not passed to the answer model.
Resource reporting is handled separately by ``finalize_artifacts``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from tqdm.auto import tqdm

from evaluation.answer_prompts import build_benchmark_answer_messages
from evaluation.finalize_artifacts import (
    read_jsonl,
    select_final_session_rows,
    select_validation_dump,
    write_json_atomic,
    write_jsonl_atomic,
)
from resource_profiles import (
    DEFAULT_RESOURCE_CONFIG_PATH,
    ResourceProfile,
    load_resource_profile_catalog,
)
from rewards.memory_qa import extract_tagged_answer


SCHEMA_VERSION = "official.answer_replacement.v1"
PROMPT_VERSION = "benchmark_direct_qa_v2"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPENAI_WORKERS = 8
# The answer model sees only the benchmark question and extracted memory
# evidence, not the original long-context prompt.  Leaving this unset makes
# vLLM reserve KV cache for the model-card maximum (262K for Qwen3), which can
# prevent an otherwise small model from starting on a 48 GiB GPU.
DEFAULT_VLLM_MAX_MODEL_LEN = 16_384

_TOOL_RESPONSE_PATTERN = re.compile(
    r"<tool_response(?:\s[^>]*)?>\s*(.*?)\s*</tool_response>",
    flags=re.IGNORECASE | re.DOTALL,
)

@dataclass(frozen=True)
class AnswerRequest:
    row_key: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    observed_wall_time_seconds: float


class AnswerGenerator(Protocol):
    backend_name: str
    cache_identity: Mapping[str, Any]

    def generate_batch(
        self, requests: Sequence[AnswerRequest]
    ) -> list[GenerationResult | Exception]: ...


class OpenAIAnswerGenerator:
    """OpenAI-compatible answer generator with bounded retry and concurrency."""

    backend_name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None,
        max_new_tokens: int,
        workers: int,
        max_retries: int,
        disable_reasoning: bool = False,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.workers = max(int(workers), 1)
        self.max_retries = max(int(max_retries), 1)
        self.cache_identity = {"base_url": base_url}
        self.request_options: dict[str, Any] = {}
        if disable_reasoning:
            self.request_options["extra_body"] = {"reasoning": {"enabled": False}}
            self.cache_identity["reasoning"] = {"enabled": False}
        self.client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))

    def generate_batch(
        self, requests: Sequence[AnswerRequest]
    ) -> list[GenerationResult | Exception]:
        results: list[GenerationResult | Exception | None] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._generate_one, request): index
                for index, request in enumerate(requests)
            }
            for future in as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                except Exception as exc:
                    results[futures[future]] = exc
        if any(result is None for result in results):
            raise RuntimeError("Answer-model generation lost one or more rows.")
        return [result for result in results if result is not None]

    def _generate_one(self, request: AnswerRequest) -> GenerationResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=request.messages,
                    temperature=0.0,
                    max_completion_tokens=self.max_new_tokens,
                    **self.request_options,
                )
                text = str(response.choices[0].message.content or "").strip()
                usage = getattr(response, "usage", None)
                if usage is None:
                    raise ValueError("Answer-model API response did not include token usage.")
                input_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
                output_tokens = _usage_int(usage, "completion_tokens", "output_tokens")
                return GenerationResult(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    observed_wall_time_seconds=time.perf_counter() - started,
                )
            except Exception as exc:  # Provider failures need bounded, auditable retry.
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"Answer-model request {request.row_key!r} failed after {self.max_retries} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


class VLLMAnswerGenerator:
    """In-process vLLM backend for a local/Hugging Face answer model."""

    backend_name = "vllm"

    def __init__(
        self,
        *,
        model: str,
        max_new_tokens: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int | None,
        trust_remote_code: bool,
        disable_multimodal_inputs: bool,
    ) -> None:
        cache_dirs = _configure_vllm_cache_dirs()
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        print(
            "Answer-model cache directories: "
            f"VLLM_CACHE_ROOT={cache_dirs['VLLM_CACHE_ROOT']}, "
            f"TORCHINDUCTOR_CACHE_DIR={cache_dirs['TORCHINDUCTOR_CACHE_DIR']}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )
        llm_kwargs: dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": max(int(tensor_parallel_size), 1),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "trust_remote_code": bool(trust_remote_code),
            "enable_prefix_caching": True,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max(int(max_model_len), 1)
        if disable_multimodal_inputs:
            # A VLM still profiles dummy images/videos during engine startup
            # unless unused modalities are explicitly disabled. Answer
            # replacement is text-only, so avoid that unnecessary path.
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max(int(max_new_tokens), 1),
            seed=0,
        )
        self.cache_identity = {
            "tensor_parallel_size": max(int(tensor_parallel_size), 1),
            "max_model_len": max_model_len,
            "trust_remote_code": bool(trust_remote_code),
            "disable_multimodal_inputs": bool(disable_multimodal_inputs),
        }

    def generate_batch(self, requests: Sequence[AnswerRequest]) -> list[GenerationResult]:
        prompts = [
            self.tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for request in requests
        ]
        started = time.perf_counter()
        outputs = self.llm.generate(prompts, self.sampling_params)
        elapsed_per_request = (time.perf_counter() - started) / max(len(outputs), 1)
        results = []
        for prompt, output in zip(prompts, outputs, strict=True):
            candidate = output.outputs[0]
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            if prompt_token_ids is None:
                prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            results.append(
                GenerationResult(
                    text=str(candidate.text or "").strip(),
                    input_tokens=len(prompt_token_ids),
                    output_tokens=len(candidate.token_ids),
                    observed_wall_time_seconds=elapsed_per_request,
                )
            )
        if len(results) != len(requests):
            raise RuntimeError("vLLM answer generation returned an unexpected number of rows.")
        return results


def replace_saved_answers(
    *,
    input_path: str | Path,
    output_path: str | Path,
    generator: AnswerGenerator,
    model: str,
    resource_profile: ResourceProfile,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> dict[str, Any]:
    """Generate replacement answers and emit evaluator-compatible raw JSONL."""

    raw_file, candidates = select_validation_dump(input_path)
    source_rows = select_final_session_rows(list(read_jsonl(raw_file)))
    if not source_rows:
        raise ValueError(f"Saved response dump is empty: {raw_file}")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(output.suffix + ".progress.jsonl")
    if force:
        progress_path.unlink(missing_ok=True)
    completed = _load_progress(progress_path)
    backend_config = dict(getattr(generator, "cache_identity", {}))
    prepared: list[dict[str, Any]] = []
    pending: list[tuple[int, AnswerRequest]] = []
    num_cache_hits = 0
    num_skipped = 0

    for index, source_row in enumerate(source_rows):
        row = dict(source_row)
        if row.get("answer_replacement_json") not in (None, ""):
            raise ValueError(
                f"Row {row.get('uid')!r} already contains answer-replacement metadata. "
                "Always start from the original policy dump."
            )
        response = str(row.get("output") or "")
        try:
            policy_prefix, original_answer = split_final_answer(response)
            original_answer_format_valid = True
        except ValueError:
            policy_prefix, original_answer = _recover_invalid_policy_answer(response)
            original_answer_format_valid = False
        sample_metadata = _json_object(
            row.get("sample_metadata_json"), "sample_metadata_json"
        )
        try:
            memory_evidence = extract_policy_memory_evidence(policy_prefix)
        except ValueError:
            memory_evidence = []
        row_key = replacement_row_key(
            row,
            index=index,
            model=model,
            backend=generator.backend_name,
            backend_config=backend_config,
            max_new_tokens=max_new_tokens,
        )
        prepared.append(
            {
                "row": row,
                "row_key": row_key,
                "policy_prefix": policy_prefix,
                "original_answer": original_answer,
                "original_answer_format_valid": original_answer_format_valid,
                "messages": None,
                "memory_evidence_count": len(memory_evidence),
                "replacement": None,
            }
        )
        cached = completed.get(row_key)
        if cached is not None:
            prepared[-1]["replacement"] = cached
            num_cache_hits += 1
        elif not memory_evidence:
            replacement = _build_skipped_replacement(
                prepared[-1],
                model=model,
                backend=generator.backend_name,
                backend_config=backend_config,
                resource_profile=resource_profile,
            )
            prepared[-1]["replacement"] = replacement
            _append_progress(progress_path, row_key, replacement)
            completed[row_key] = replacement
            num_skipped += 1
        else:
            messages = build_answer_messages(
                dataset_source=str(
                    row.get("dataset_source")
                    or sample_metadata.get("dataset_source")
                    or ""
                ),
                sample_metadata=sample_metadata,
                memory_evidence=memory_evidence,
            )
            prepared[-1]["messages"] = messages
            pending.append((index, AnswerRequest(row_key=row_key, messages=messages)))

    failures: list[str] = []
    size = max(int(batch_size), 1)
    num_generated = 0
    with tqdm(
        total=len(prepared),
        initial=num_cache_hits + num_skipped,
        desc="Replacing saved answers",
        unit="response",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix(
            cached=num_cache_hits,
            skipped=num_skipped,
            generated=num_generated,
            errors=len(failures),
            refresh=False,
        )
        for start in range(0, len(pending), size):
            batch = pending[start : start + size]
            requests = [request for _, request in batch]
            try:
                results = generator.generate_batch(requests)
            except Exception as exc:
                failures.append(f"batch {start // size}: {type(exc).__name__}: {exc}")
                progress.set_postfix(errors=len(failures), refresh=True)
                continue
            if len(results) != len(batch):
                failures.append(
                    f"batch {start // size}: generated {len(results)} results for "
                    f"{len(batch)} requests"
                )
                progress.set_postfix(errors=len(failures), refresh=True)
                continue
            for (row_index, request), result in zip(batch, results, strict=True):
                if isinstance(result, Exception):
                    failures.append(
                        f"{request.row_key}: {type(result).__name__}: {result}"
                    )
                    continue
                try:
                    replacement = _build_replacement(
                        prepared[row_index],
                        result=result,
                        model=model,
                        backend=generator.backend_name,
                        backend_config=backend_config,
                        resource_profile=resource_profile,
                    )
                except Exception as exc:
                    failures.append(
                        f"{request.row_key}: {type(exc).__name__}: {exc}"
                    )
                    continue
                prepared[row_index]["replacement"] = replacement
                _append_progress(progress_path, request.row_key, replacement)
                completed[request.row_key] = replacement
                num_generated += 1
                progress.update(1)
            progress.set_postfix(
                cached=num_cache_hits,
                skipped=num_skipped,
                generated=num_generated,
                errors=len(failures),
                refresh=True,
            )

    if failures:
        preview = "\n".join(f"- {failure}" for failure in failures[:20])
        raise RuntimeError(
            f"Answer replacement failed for {len(failures)} batch/row(s). Partial progress is saved at "
            f"{progress_path}; fix the cause and rerun.\n{preview}"
        )

    output_rows = []
    for item in prepared:
        replacement = item.get("replacement")
        if not isinstance(replacement, Mapping):
            raise RuntimeError(f"No replacement result for {item['row_key']!r}.")
        output_rows.append(_merge_replacement(item["row"], replacement))
    write_jsonl_atomic(output, output_rows)

    num_answer_model_calls = sum(
        int((item["replacement"].get("answer_model_usage") or {}).get("num_calls", 0))
        for item in prepared
    )
    num_skipped_without_memory_evidence = len(prepared) - num_answer_model_calls
    num_invalid_original_policy_answers = sum(
        not bool(item["original_answer_format_valid"]) for item in prepared
    )
    num_empty_answer_model_outputs = sum(
        int((item["replacement"].get("answer_model_usage") or {}).get("num_calls", 0)) > 0
        and item["replacement"].get("answer_nonempty") is False
        for item in prepared
    )

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    write_json_atomic(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_raw_response_dump": str(raw_file),
            "candidate_raw_response_dumps": [str(path) for path in candidates],
            "output_file": str(output),
            "progress_file": str(progress_path),
            "num_responses": len(output_rows),
            "num_answer_model_calls": num_answer_model_calls,
            "num_skipped_without_memory_evidence": num_skipped_without_memory_evidence,
            "num_invalid_original_policy_answers": num_invalid_original_policy_answers,
            "num_empty_answer_model_outputs": num_empty_answer_model_outputs,
            "answer_model": model,
            "backend": generator.backend_name,
            "backend_config": backend_config,
            "temperature": 0.0,
            "max_new_tokens": max(int(max_new_tokens), 1),
            "prompt_version": PROMPT_VERSION,
            "answer_prompt_inputs": "structured benchmark query plus extracted tool-response evidence",
            "policy_reasoning_sent_to_answer_model": False,
            "tool_schemas_or_call_arguments_sent_to_answer_model": False,
            "resource_profile": resource_profile.as_dict(),
            "ground_truth_sent_to_answer_model": False,
            "forced_full_regeneration": bool(force),
            "resource_accounting": (
                "The replacement generation's actual token usage, latency, and call are retained for audit "
                "and excluded from final metrics. During final evaluation, the answer model's configured "
                "price and latency profile is instead applied to the saved policy token counts and generation "
                "segments; routed external models retain their own profiles."
            ),
        },
    )
    return {
        "rows": len(output_rows),
        "output_path": output,
        "manifest_path": manifest_path,
        "progress_path": progress_path,
        "selected_raw_file": raw_file,
        "num_answer_model_calls": num_answer_model_calls,
        "num_skipped_without_memory_evidence": num_skipped_without_memory_evidence,
        "num_invalid_original_policy_answers": num_invalid_original_policy_answers,
        "num_empty_answer_model_outputs": num_empty_answer_model_outputs,
    }


def split_final_answer(response: str) -> tuple[str, str]:
    """Return the trajectory before the one strict final answer and its content."""

    parsed, valid = extract_tagged_answer(response)
    if not valid or parsed is None:
        raise ValueError(
            "Policy response must contain exactly one valid final <answer>...</answer> block before it "
            "can be decoupled."
        )
    tool_ends = list(re.finditer(r"</tool_response>", response, flags=re.IGNORECASE))
    segment_start = tool_ends[-1].end() if tool_ends else 0
    final_segment = response[segment_start:]
    blocks = list(
        re.finditer(r"<answer>\s*(.*?)\s*</answer>", final_segment, flags=re.DOTALL)
    )
    if len(blocks) != 1 or final_segment[blocks[0].end() :].strip():
        raise ValueError("Unable to isolate the policy's final answer block.")
    block = blocks[0]
    return response[: segment_start + block.start()], block.group(1).strip()


def _recover_invalid_policy_answer(response: str) -> tuple[str, str | None]:
    """Remove a malformed final answer while retaining the auditable trajectory."""

    response = str(response or "")
    tool_ends = list(re.finditer(r"</tool_response>", response, flags=re.IGNORECASE))
    segment_start = tool_ends[-1].end() if tool_ends else 0
    final_segment = response[segment_start:]
    complete_blocks = list(
        re.finditer(r"<answer>\s*(.*?)\s*</answer>", final_segment, flags=re.DOTALL)
    )
    original_answer = (
        " ".join(complete_blocks[-1].group(1).split()) if complete_blocks else None
    )
    first_marker = re.search(r"</?answer\b", final_segment, flags=re.IGNORECASE)
    if first_marker is None:
        return response, original_answer
    return response[: segment_start + first_marker.start()], original_answer


def extract_policy_memory_evidence(policy_prefix: str) -> list[str]:
    """Extract only environment-provided memory evidence from a policy trajectory."""

    evidence = [
        match.group(1).strip()
        for match in _TOOL_RESPONSE_PATTERN.finditer(str(policy_prefix or ""))
        if match.group(1).strip()
    ]
    if not evidence:
        raise ValueError(
            "Answer replacement requires at least one non-empty <tool_response> memory evidence block."
        )
    return evidence


def build_answer_messages(
    *,
    dataset_source: str,
    sample_metadata: Mapping[str, Any],
    memory_evidence: Sequence[str],
) -> list[dict[str, str]]:
    return build_benchmark_answer_messages(
        dataset_source=dataset_source,
        sample_metadata=sample_metadata,
        memory_evidence=memory_evidence,
    )


def replacement_row_key(
    row: Mapping[str, Any],
    *,
    index: int,
    model: str,
    backend: str,
    backend_config: Mapping[str, Any],
    max_new_tokens: int,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "uid": row.get("uid"),
        "dataset_source": row.get("dataset_source"),
        "input": row.get("input"),
        "output": row.get("output"),
        "runtime_trace_json": row.get("runtime_trace_json"),
        "model": model,
        "backend": backend,
        "backend_config": dict(backend_config),
        "temperature": 0.0,
        "max_new_tokens": max(int(max_new_tokens), 1),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    uid = str(row.get("uid") or f"row-{index}")
    return f"{uid}:{digest}"


def _build_replacement(
    item: Mapping[str, Any],
    *,
    result: GenerationResult,
    model: str,
    backend: str,
    backend_config: Mapping[str, Any],
    resource_profile: ResourceProfile,
) -> dict[str, Any]:
    answer, answer_parse_mode = _normalize_answer_model_output(result.text)
    input_tokens = _nonnegative_int(result.input_tokens, "input_tokens")
    output_tokens = _nonnegative_int(result.output_tokens, "output_tokens")
    profile = resource_profile.as_dict()
    raw_cost = (
        input_tokens * profile["input_price_per_million_usd"]
        + output_tokens * profile["output_price_per_million_usd"]
    )
    estimated_latency = (
        profile["latency_base_seconds"]
        + input_tokens * profile["latency_input_seconds_per_token"]
        + output_tokens * profile["latency_output_seconds_per_token"]
    )
    usage = {
        "model": model,
        "backend": backend,
        "backend_config": dict(backend_config),
        "num_calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "api_cost": raw_cost,
        "estimated_latency": estimated_latency,
        "observed_wall_time_seconds": max(float(result.observed_wall_time_seconds), 0.0),
        **profile,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "backend": backend,
        "temperature": 0.0,
        "ground_truth_sent_to_answer_model": False,
        "policy_reasoning_sent_to_answer_model": False,
        "tool_schemas_or_call_arguments_sent_to_answer_model": False,
        "memory_evidence_count": int(item["memory_evidence_count"]),
        "original_policy_answer": item["original_answer"],
        "original_policy_answer_format_valid": bool(
            item["original_answer_format_valid"]
        ),
        "replacement_answer": answer,
        "answer_nonempty": bool(answer.strip()),
        "raw_answer_model_response": result.text,
        "answer_parse_mode": answer_parse_mode,
        "answer_model_usage": usage,
        "answer_prompt_sha256": hashlib.sha256(
            json.dumps(item["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "resource_accounting_mode": (
            "profile_applied_to_policy_usage_actual_generation_excluded"
        ),
    }
    replacement_output = (
        f"{item['policy_prefix']}<answer>{answer}</answer>"
        if answer.strip()
        else str(item["policy_prefix"])
    )
    return {
        # Leave an empty model answer without a final answer block so the
        # downstream format and quality metrics score it as a failure.
        "output": replacement_output,
        "answer_replacement_json": json.dumps(audit, ensure_ascii=False, sort_keys=True),
        "answer_model_usage": usage,
        "answer_nonempty": bool(answer.strip()),
    }


def _normalize_answer_model_output(text: str) -> tuple[str, str]:
    """Recover an answer while preserving empty model outputs as scored failures."""

    raw = str(text or "").strip()
    if not raw:
        return "", "empty_response"

    answer, valid = extract_tagged_answer(raw)
    if valid and answer is not None and answer.strip():
        return answer.strip(), "strict_tagged"

    complete_blocks = [
        " ".join(match.group(1).split())
        for match in re.finditer(
            r"<answer>\s*(.*?)\s*</answer>",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match.group(1).strip()
    ]
    if complete_blocks:
        return complete_blocks[-1], "recovered_tagged"

    plain = re.sub(r"</?answer\b[^>]*>", "", raw, flags=re.IGNORECASE).strip()
    plain = " ".join(plain.split())
    if not plain:
        return "", "empty_tagged"
    return plain, "plain_text_wrapped"


def _build_skipped_replacement(
    item: Mapping[str, Any],
    *,
    model: str,
    backend: str,
    backend_config: Mapping[str, Any],
    resource_profile: ResourceProfile,
) -> dict[str, Any]:
    """Represent a no-memory trajectory as an explicit, zero-credit failure."""

    profile = resource_profile.as_dict()
    usage = {
        "model": model,
        "backend": backend,
        "backend_config": dict(backend_config),
        "num_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_cost": 0.0,
        "estimated_latency": 0.0,
        "observed_wall_time_seconds": 0.0,
        **profile,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "status": "skipped_no_memory_evidence",
        "model": model,
        "backend": backend,
        "temperature": 0.0,
        "ground_truth_sent_to_answer_model": False,
        "policy_reasoning_sent_to_answer_model": False,
        "tool_schemas_or_call_arguments_sent_to_answer_model": False,
        "memory_evidence_count": 0,
        "original_policy_answer": item["original_answer"],
        "original_policy_answer_format_valid": bool(
            item["original_answer_format_valid"]
        ),
        "replacement_answer": None,
        "raw_answer_model_response": None,
        "answer_model_usage": usage,
        "answer_prompt_sha256": None,
        "resource_accounting_mode": (
            "profile_applied_to_policy_usage_answer_not_called_no_memory_evidence"
        ),
    }
    return {
        # Removing any final answer ensures the downstream evaluator records an
        # invalid format and zero local answer quality for this trajectory.
        "output": str(item["policy_prefix"]),
        "answer_replacement_json": json.dumps(audit, ensure_ascii=False, sort_keys=True),
        "answer_model_usage": usage,
        "answer_nonempty": None,
    }


def _merge_replacement(row: Mapping[str, Any], replacement: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    runtime_trace = _json_object(updated.get("runtime_trace_json"), "runtime_trace_json")
    if "answer_model" in runtime_trace:
        raise ValueError(
            f"Row {updated.get('uid')!r} already has runtime_trace.answer_model; refusing to double-count."
        )
    usage = replacement.get("answer_model_usage")
    if not isinstance(usage, Mapping):
        raise ValueError("Cached answer replacement is missing answer_model_usage.")
    runtime_trace["answer_model"] = dict(usage)
    updated["output"] = str(replacement.get("output") or "")
    updated["answer_replacement_json"] = str(
        replacement.get("answer_replacement_json") or ""
    )
    updated["runtime_trace_json"] = json.dumps(
        runtime_trace, ensure_ascii=False, sort_keys=True
    )
    return updated


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed = {}
    for record in read_jsonl(path):
        key = str(record.get("row_key") or "").strip()
        replacement = record.get("replacement")
        if key and isinstance(replacement, Mapping):
            completed[key] = dict(replacement)
    return completed


def _append_progress(path: Path, row_key: str, replacement: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {"row_key": row_key, "replacement": replacement},
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Saved response contains invalid {name}.") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"Saved response {name} must decode to an object.")
    return decoded


def _usage_int(usage: Any, *names: str) -> int:
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return _nonnegative_int(value, name)
    raise ValueError(f"Answer-model usage is missing all of {names}.")


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Answer-model {name} must be an integer.") from exc
    if number < 0:
        raise ValueError(f"Answer-model {name} must be non-negative.")
    return number


def _resolve_resource_profile(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ResourceProfile:
    try:
        profiles = load_resource_profile_catalog(
            getattr(args, "resource_config", DEFAULT_RESOURCE_CONFIG_PATH)
        )
        return profiles.resolve(args.model, role="answer-model")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
        raise AssertionError("argparse.ArgumentParser.error must terminate") from exc


def _configure_vllm_cache_dirs() -> dict[str, str]:
    """Put local compilation caches on the project filesystem by default."""

    project_root = Path(__file__).resolve().parents[1]
    defaults = {
        "VLLM_CACHE_ROOT": project_root / ".runtime" / "vllm",
        "TORCHINDUCTOR_CACHE_DIR": project_root / ".runtime" / "torchinductor",
    }
    configured: dict[str, str] = {}
    for variable, default in defaults.items():
        path = Path(os.environ.get(variable) or default).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise OSError(f"{variable} is not writable: {path}")
        os.environ[variable] = str(path)
        configured[variable] = str(path)
    return configured


def _build_generator(args: argparse.Namespace, parser: argparse.ArgumentParser) -> AnswerGenerator:
    if args.backend == "vllm":
        return VLLMAnswerGenerator(
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
            disable_multimodal_inputs=args.disable_multimodal_inputs,
        )
    api_key = (
        args.api_key
        or os.environ.get("ANSWER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        parser.error(
            "Set --api_key, ANSWER_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY for --backend openai."
        )
    return OpenAIAnswerGenerator(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        max_new_tokens=args.max_new_tokens,
        workers=args.workers,
        max_retries=args.max_retries,
        disable_reasoning=args.disable_reasoning,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally decouple a trained memory policy from final answer generation by replacing each "
            "saved final answer with one generated by an independent model."
        )
    )
    parser.add_argument("--input", required=True, help="Raw Verl validation JSONL file or directory.")
    parser.add_argument("--output", required=True, help="Replacement raw JSONL file.")
    parser.add_argument("--backend", choices=("vllm", "openai"), default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=DEFAULT_VLLM_MAX_MODEL_LEN,
        help=(
            "Maximum vLLM context length for local answer generation "
            f"(default: {DEFAULT_VLLM_MAX_MODEL_LEN})."
        ),
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--disable_multimodal_inputs",
        action="store_true",
        help=(
            "Run a multimodal checkpoint as a text-only answer model by disabling image/video profiling."
        ),
    )
    parser.add_argument(
        "--base_url",
        default=(
            os.environ.get("ANSWER_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ),
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_OPENAI_WORKERS)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--disable_reasoning",
        action="store_true",
        help="Request OpenRouter non-thinking generation with reasoning.enabled=false (API backend only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate every replacement answer, ignoring and resetting any existing "
            "progress cache. The completed output is replaced only after the full run succeeds."
        ),
    )
    parser.add_argument(
        "--resource_config",
        default=str(DEFAULT_RESOURCE_CONFIG_PATH),
        help=(
            "Canonical YAML source for the answer model's cost and latency profile "
            f"(default: {DEFAULT_RESOURCE_CONFIG_PATH})."
        ),
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.batch_size <= 0:
        parser.error("--max_new_tokens and --batch_size must be positive.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu_memory_utilization must be in (0, 1].")
    if args.disable_reasoning and args.backend != "openai":
        parser.error("--disable_reasoning requires --backend openai.")
    profile = _resolve_resource_profile(args, parser)
    generator = _build_generator(args, parser)
    result = replace_saved_answers(
        input_path=args.input,
        output_path=args.output,
        generator=generator,
        model=args.model,
        resource_profile=profile,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(
        f"Processed {result['rows']} responses with {args.model}: "
        f"generated {result['num_answer_model_calls']} replacement answers, "
        f"kept {result['num_skipped_without_memory_evidence']} no-memory trajectories "
        f"as explicit failures; recorded {result['num_empty_answer_model_outputs']} empty answer-model "
        f"outputs as zero-credit failures; {result['num_invalid_original_policy_answers']} original "
        f"policy answers had invalid format -> {result['output_path']}"
    )
    print(
        "Compute final metrics from the replacement file with: "
        f"python3 -m evaluation.evaluate_saved_responses --input "
        f"'{result['output_path']}' --output_dir '<final-output-dir>'"
    )


if __name__ == "__main__":
    main()
