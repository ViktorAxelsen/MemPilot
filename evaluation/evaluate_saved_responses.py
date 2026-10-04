"""Compute all final metrics from a saved response dump in one post-hoc run."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.finalize_artifacts import (
    finalize_validation_dump,
    format_evaluation_report,
    write_json_atomic,
    write_text_atomic,
)
from evaluation.run_llm_judge import (
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MAX_NEW_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_JUDGE_WORKERS,
    judge_predictions,
)
from resource_profiles import DEFAULT_RESOURCE_CONFIG_PATH


def evaluate_saved_responses(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    api_key: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_base_url: str = DEFAULT_JUDGE_BASE_URL,
    judge_workers: int = DEFAULT_JUDGE_WORKERS,
    judge_max_new_tokens: int = DEFAULT_JUDGE_MAX_NEW_TOKENS,
    judge_max_retries: int = 3,
    resource_config_path: str | Path = DEFAULT_RESOURCE_CONFIG_PATH,
) -> dict[str, Any]:
    """Normalize responses, compute F1/resources, and run the LLM judge."""

    destination = Path(output_dir).expanduser().resolve()
    normalized = finalize_validation_dump(
        input_path,
        destination,
        resource_config_path=resource_config_path,
    )
    judged_path = destination / "evaluated_responses.jsonl"
    metrics_path = destination / "metrics.json"
    judged = judge_predictions(
        input_path=normalized["responses_path"],
        output_path=judged_path,
        metrics_path=metrics_path,
        api_key=api_key,
        model=judge_model,
        base_url=judge_base_url,
        workers=judge_workers,
        max_tokens=judge_max_new_tokens,
        max_retries=judge_max_retries,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = format_evaluation_report(metrics)
    report_path = destination / "report.txt"
    write_text_atomic(report_path, report)

    manifest_path = Path(normalized["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluated_responses_file": str(judged_path),
            "metrics_file": str(metrics_path),
            "report_file": str(report_path),
            "judge_status": "complete" if judged["errors"] == 0 else "error",
            "judge_error_count": judged["errors"],
            "judge": {
                "provider": "openrouter",
                "base_url": judge_base_url,
                "model": judge_model,
                "temperature": DEFAULT_JUDGE_TEMPERATURE,
                "max_new_tokens": judge_max_new_tokens,
                "workers": judge_workers,
                "max_retries": judge_max_retries,
                "accounting_note": (
                    "Judge token usage is retained per row for audit but excluded from evaluated-system cost "
                    "and latency."
                ),
            },
        }
    )
    write_json_atomic(manifest_path, manifest)
    return {
        **normalized,
        "evaluated_responses_path": judged_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
        "report": report,
        "judge_errors": judged["errors"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read saved Verl responses and independently compute F1, GPT-4o-mini judge score, cost, "
            "and proxy latency."
        )
    )
    parser.add_argument("--input", required=True, help="Raw validation JSONL file or directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--judge_base_url",
        default=(
            os.environ.get("JUDGE_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_JUDGE_BASE_URL
        ),
    )
    parser.add_argument(
        "--judge_workers", type=int, default=DEFAULT_JUDGE_WORKERS
    )
    parser.add_argument(
        "--judge_max_new_tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_NEW_TOKENS,
    )
    parser.add_argument("--judge_max_retries", type=int, default=3)
    parser.add_argument(
        "--resource_config",
        default=str(DEFAULT_RESOURCE_CONFIG_PATH),
        help=(
            "Canonical YAML source for every cost and latency coefficient "
            f"(default: {DEFAULT_RESOURCE_CONFIG_PATH})."
        ),
    )
    args = parser.parse_args()
    api_key = (
        args.api_key
        or os.environ.get("JUDGE_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        parser.error("Set --api_key, JUDGE_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    result = evaluate_saved_responses(
        input_path=args.input,
        output_dir=args.output_dir,
        api_key=api_key,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_workers=args.judge_workers,
        judge_max_new_tokens=args.judge_max_new_tokens,
        judge_max_retries=args.judge_max_retries,
        resource_config_path=args.resource_config,
    )
    print(result["report"])
    print(
        f"Evaluated {result['num_responses']} saved responses -> {result['metrics_path']} "
        f"(judge errors: {result['judge_errors']})"
    )
    if result["judge_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
