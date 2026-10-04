"""Apply benchmark-specific answer-only judge protocols to final predictions.

Only the prediction and reference answer are sample-dependent judge inputs.
Judge API usage is retained for audit, not charged to the evaluated system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from json_repair import repair_json
from openai import OpenAI
from tqdm.auto import tqdm

from evaluation.finalize_artifacts import (
    aggregate_predictions,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from evaluation.judge_protocols import (
    BINARY_SCORES,
    GRADED_SCORES,
    MEMEYE_REPORTED_SCORES,
    WORLD_LABELS,
    get_judge_protocol,
    render_judge_prompt,
)


DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
DEFAULT_JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_JUDGE_MAX_NEW_TOKENS = 512
DEFAULT_JUDGE_WORKERS = 16
JUDGE_CACHE_VERSION = "official.judge_cache.v2"


def judge_predictions(
    *,
    input_path: str | Path,
    output_path: str | Path,
    metrics_path: str | Path,
    api_key: str,
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str | None = DEFAULT_JUDGE_BASE_URL,
    workers: int = DEFAULT_JUDGE_WORKERS,
    max_tokens: int = DEFAULT_JUDGE_MAX_NEW_TOKENS,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Judge pending rows and write a complete, restart-safe result set."""

    input_rows = list(read_jsonl(input_path))
    if not input_rows:
        raise ValueError(f"No predictions found in {input_path}.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(output.suffix + ".progress.jsonl")
    completed = _load_progress(progress_path)
    client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    lock = threading.Lock()

    def process(
        index: int,
        row: Mapping[str, Any],
        uid: str,
    ) -> tuple[int, dict[str, Any]]:
        judged = _judge_row(
            dict(row),
            client=client,
            model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        with lock:
            with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        {"row_key": uid, "judge": judged.get("judge")},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            completed[uid] = dict(judged.get("judge") or {})
        return index, judged

    judged_rows: list[dict[str, Any] | None] = [None] * len(input_rows)
    pending: list[tuple[int, Mapping[str, Any], str]] = []
    for index, row in enumerate(input_rows):
        uid = _row_key(
            row,
            index,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
        )
        existing_judge = completed.get(uid)
        if existing_judge is not None and existing_judge.get("status") == "complete":
            current_row = dict(row)
            current_row["judge"] = dict(existing_judge)
            judged_rows[index] = current_row
        else:
            pending.append((index, row, uid))

    worker_count = max(int(workers), 1)
    restored = len(input_rows) - len(pending)
    with tqdm(
        total=len(input_rows),
        initial=restored,
        desc="Judging saved responses",
        unit="query",
        dynamic_ncols=True,
    ) as progress:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(process, index, row, uid): index
                for index, row, uid in pending
            }
            for future in as_completed(futures):
                index, judged = future.result()
                judged_rows[index] = judged
                progress.update(1)

    final_rows = [row for row in judged_rows if row is not None]
    if len(final_rows) != len(input_rows):
        raise RuntimeError("Judge output lost one or more rows.")
    write_jsonl_atomic(output, final_rows)
    metrics = aggregate_predictions(final_rows)
    metrics["judge_model"] = model
    metrics["judge_error_count"] = sum(
        (row.get("judge") or {}).get("status") == "error" for row in final_rows
    )
    metrics["judge_json_repair_count"] = sum(
        bool((row.get("judge") or {}).get("json_repaired")) for row in final_rows
    )
    write_json_atomic(metrics_path, metrics)
    return {
        "rows": len(final_rows),
        "errors": metrics["judge_error_count"],
        "output_path": output,
        "metrics_path": Path(metrics_path),
        "progress_path": progress_path,
    }


def _judge_row(
    row: dict[str, Any],
    *,
    client: OpenAI,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    judge = dict(row.get("judge") or {})
    protocol = get_judge_protocol(str(judge.get("protocol_id") or ""))
    if not protocol.requires_llm_judge:
        return row
    prompt = render_judge_prompt(
        protocol.protocol_id,
        prediction=row.get("prediction"),
        references=list(row.get("references") or []),
    )
    last_error: Exception | None = None
    for attempt in range(max(int(max_retries), 1)):
        try:
            request = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=DEFAULT_JUDGE_TEMPERATURE,
                max_completion_tokens=max(int(max_tokens), 1),
            )
            if protocol.labels == WORLD_LABELS or protocol.score_values != BINARY_SCORES:
                request["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request)
            text = str(response.choices[0].message.content or "").strip()
            parsed = parse_judge_response(text, protocol.protocol_id)
            usage = getattr(response, "usage", None)
            row["judge"] = {
                "protocol_id": protocol.protocol_id,
                "status": "complete",
                "model": model,
                "prompt": prompt,
                "score": parsed["score"],
                "label": parsed["label"],
                "reasoning": parsed["reasoning"],
                "json_repaired": parsed["json_repaired"],
                "raw_response": text,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
            }
            return row
        except Exception as exc:  # API/provider errors need bounded retry and an auditable row.
            last_error = exc
            if attempt + 1 < max(int(max_retries), 1):
                time.sleep(min(2**attempt, 8))
    row["judge"] = {
        "protocol_id": protocol.protocol_id,
        "status": "error",
        "model": model,
        "prompt": prompt,
        "score": None,
        "label": None,
        "reasoning": None,
        "error": f"{type(last_error).__name__}: {last_error}",
    }
    return row


def parse_judge_response(text: str, protocol_id: str) -> dict[str, Any]:
    protocol = get_judge_protocol(protocol_id)
    payload, json_repaired = _extract_json_object(text)
    if protocol.labels == WORLD_LABELS:
        reasoning = str(payload.get("reasoning") or "").strip()
        if not reasoning:
            raise ValueError("WorldMemArena judge response requires non-empty reasoning.")
        label = _canonical_label(
            payload.get("evaluation_result") or payload.get("label"), WORLD_LABELS
        )
        expected_score = 1.0 if label == "Correct" else 0.0
        supplied = payload.get("score")
        if supplied not in (None, "") and _score(supplied, BINARY_SCORES) != expected_score:
            raise ValueError("WorldMemArena judge score disagrees with its label.")
        return {
            "score": expected_score,
            "label": label,
            "reasoning": reasoning,
            "json_repaired": json_repaired,
        }
    if protocol.score_values == BINARY_SCORES:
        score = _score(payload.get("answer_score", payload.get("score")), BINARY_SCORES)
        label = "Correct" if score == 1.0 else "Incorrect"
        supplied_label = payload.get("label")
        if supplied_label not in (None, "") and _canonical_label(
            supplied_label, ("Correct", "Incorrect")
        ) != label:
            raise ValueError("Binary judge score disagrees with its label.")
        reasoning = str(payload.get("reasoning") or _scoring_rationale(text)).strip()
        return {
            "score": score,
            "label": label,
            "reasoning": reasoning,
            "json_repaired": json_repaired,
        }
    reasoning = str(payload.get("reasoning") or "").strip()
    if protocol.protocol_id == "memeye_open_answer_v1":
        score = _coerce_memeye_score(payload.get("score"))
    else:
        score = _score(payload.get("score"), GRADED_SCORES)
    return {
        "score": score,
        "label": str(payload.get("label") or _graded_label(score)).strip(),
        "reasoning": reasoning,
        "json_repaired": json_repaired,
    }


def _scoring_rationale(text: str) -> str:
    match = re.search(
        r"\[Scoring Rationale\]\s*:\s*(.*?)(?:\[Score\]|\[JSON\]|$)",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _extract_json_object(text: str) -> tuple[Mapping[str, Any], bool]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise ValueError("Judge response does not contain a JSON object.")
        end = stripped.rfind("}")
        candidate = stripped[start : end + 1] if end > start else stripped[start:]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as strict_error:
            try:
                value = repair_json(
                    candidate,
                    return_objects=True,
                    ensure_ascii=False,
                    skip_json_loads=True,
                )
            except (TypeError, ValueError) as repair_error:
                raise ValueError("Judge response contains irreparable JSON.") from repair_error
            if not isinstance(value, Mapping):
                raise ValueError("Repaired judge response must be a JSON object.") from strict_error
            return value, True
    if not isinstance(value, Mapping):
        raise ValueError("Judge response must be a JSON object.")
    return value, False


def _score(value: Any, allowed: tuple[float, ...]) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid judge score: {value!r}.") from exc
    for candidate in allowed:
        if abs(number - candidate) < 1e-8:
            return candidate
    raise ValueError(f"Judge score {number} is not in {allowed}.")


def _coerce_memeye_score(value: Any) -> float:
    """Match MemEye's official post-judge score quantization."""

    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MemEye judge score: {value!r}.") from exc
    if score < 0.25:
        return MEMEYE_REPORTED_SCORES[0]
    if score < 0.75:
        return MEMEYE_REPORTED_SCORES[1]
    return MEMEYE_REPORTED_SCORES[2]


def _canonical_label(value: Any, allowed: tuple[str, ...]) -> str:
    normalized = str(value or "").strip().casefold()
    for candidate in allowed:
        if normalized == candidate.casefold():
            return candidate
    raise ValueError(f"Judge label {value!r} is not in {allowed}.")


def _graded_label(score: float) -> str:
    return {
        0.0: "Incorrect",
        0.25: "Poor",
        0.5: "Partial",
        0.75: "Good",
        1.0: "Correct",
    }[score]


def _row_key(
    row: Mapping[str, Any],
    index: int,
    *,
    model: str,
    base_url: str | None = DEFAULT_JUDGE_BASE_URL,
    max_tokens: int = DEFAULT_JUDGE_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
) -> str:
    uid = str(row.get("uid") or "").strip()
    judge = row.get("judge") if isinstance(row.get("judge"), Mapping) else {}
    protocol = get_judge_protocol(str(judge.get("protocol_id") or ""))
    prompt = render_judge_prompt(
        protocol.protocol_id,
        prediction=row.get("prediction"),
        references=list(row.get("references") or []),
    )
    payload = {
        "cache_version": JUDGE_CACHE_VERSION,
        "uid": uid,
        "dataset_source": row.get("dataset_source"),
        "protocol_id": protocol.protocol_id,
        "protocol_scores": protocol.score_values,
        "protocol_labels": protocol.labels,
        "rendered_prompt": prompt,
        "judge_model": model,
        "judge_base_url": base_url,
        "temperature": float(temperature),
        "max_completion_tokens": max(int(max_tokens), 1),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"{uid or f'row-{index}'}:{digest}"


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        key = str(record.get("row_key") or "").strip()
        judge = record.get("judge")
        if key and isinstance(judge, dict):
            completed[key] = judge
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run centralized final-evaluation LLM judges.")
    parser.add_argument("--input", required=True, help="Normalized saved responses JSONL.")
    parser.add_argument("--output", required=True, help="Judged predictions JSONL.")
    parser.add_argument("--metrics", required=True, help="Aggregate judged metrics JSON.")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--base_url",
        default=(
            os.environ.get("JUDGE_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_JUDGE_BASE_URL
        ),
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_JUDGE_WORKERS)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_JUDGE_MAX_NEW_TOKENS)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()
    api_key = (
        args.api_key
        or os.environ.get("JUDGE_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        parser.error("Set --api_key, JUDGE_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    result = judge_predictions(
        input_path=args.input,
        output_path=args.output,
        metrics_path=args.metrics,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        workers=args.workers,
        max_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
    )
    print(
        f"Judged {result['rows']} rows with {result['errors']} errors -> {result['output_path']}"
    )


if __name__ == "__main__":
    main()
