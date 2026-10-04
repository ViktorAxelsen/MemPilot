"""Deterministic answer metrics for saved final-evaluation responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rewards.mem_gallery_reward import token_f1 as shared_token_f1
from rewards.memeye_reward import token_f1 as memeye_token_f1
from rewards.memlens_reward import token_f1 as memlens_token_f1


def compute_local_answer_metrics(
    *,
    dataset_source: Any,
    prediction: Any,
    references: Sequence[Any],
    metadata: Mapping[str, Any] | None = None,
    ground_truth: Any = None,
) -> dict[str, float | int]:
    """Compute F1 using each benchmark's local protocol."""

    source = str(dataset_source or "").strip().casefold().replace("-", "_")
    clean_references = [str(value).strip() for value in references if str(value).strip()]
    if not clean_references:
        raise ValueError(f"{dataset_source!r} has no non-empty reference answers.")
    if "memeye" in source:
        _require_memeye_open(source, metadata or {}, ground_truth)
        token_f1 = memeye_token_f1
    elif "memlens" in source:
        token_f1 = memlens_token_f1
    elif any(name in source for name in ("mem_gallery", "worldmemarena", "h2hmem")):
        token_f1 = shared_token_f1
    else:
        raise ValueError(f"No local final-evaluation metric for data_source={dataset_source!r}.")

    prediction_text = str(prediction or "")
    f1 = max(token_f1(prediction_text, reference) for reference in clean_references)
    return {
        "f1": float(f1),
        "reference_count": len(clean_references),
    }


def _require_memeye_open(
    source: str,
    metadata: Mapping[str, Any],
    ground_truth: Any,
) -> None:
    truth = ground_truth if isinstance(ground_truth, Mapping) else {}
    variant = str(
        metadata.get("answer_variant")
        or metadata.get("variant")
        or truth.get("variant")
        or ("open" if source.endswith("_open") else "")
    ).strip().casefold()
    if variant != "open":
        raise ValueError(
            "Final MemEye evaluation supports only the open-answer split; "
            f"received variant={variant or 'unknown'!r}."
        )
