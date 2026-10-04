"""Capture final-evaluation responses without computing benchmark metrics."""

from __future__ import annotations

from typing import Any

from evaluation.artifacts import capture_evaluation_metadata


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a neutral score plus metadata required for post-hoc evaluation."""

    del solution_str, ground_truth
    return {
        "score": 0.0,
        **capture_evaluation_metadata(data_source=data_source, extra_info=extra_info),
    }
