"""Shared contracts for MemPilot's prefix-based marginal utility.

For K completed memory stages, K+1 greedy answer-only probes give answer-quality
scores Q(y_hat_t). Signed adjacent differences provide stage-level credits;
the trainer adds eta times each difference to that stage's policy decision
tokens. Probe scores contain no cost/latency penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import torch


MARGINAL_UTILITY_KEY = "marginal_utility"
MARGINAL_UTILITY_OK = "ok"
MARGINAL_UTILITY_NO_STAGES = "no_stages"
MARGINAL_UTILITY_INVALID_FORMAT = "invalid_format"
MARGINAL_UTILITY_PROBE_FAILED = "probe_failed"


@dataclass(frozen=True)
class MarginalUtilitySettings:
    """Validated weight and answer-probe budget for training."""

    weight: float
    max_answer_tokens: int


@dataclass(frozen=True)
class MarginalUtilityCredits:
    """Validated credit metadata for one completed trajectory."""

    decision_spans: tuple[tuple[int, int], ...]
    probe_rewards: tuple[float, ...]
    deltas: tuple[float, ...]


def load_marginal_utility_settings(config: Any) -> MarginalUtilitySettings:
    """Load one shared config contract in rollout workers and the trainer."""

    from omegaconf import OmegaConf

    weight = float(
        OmegaConf.select(config, "algorithm.marginal_utility.weight", default=1.0)
    )
    max_answer_tokens = int(
        OmegaConf.select(
            config,
            "algorithm.marginal_utility.max_answer_tokens",
            default=256,
        )
    )
    if not isfinite(weight) or weight < 0.0:
        raise ValueError("algorithm.marginal_utility.weight must be finite and non-negative.")
    if max_answer_tokens <= 0:
        raise ValueError("algorithm.marginal_utility.max_answer_tokens must be positive.")
    return MarginalUtilitySettings(
        weight=weight,
        max_answer_tokens=max_answer_tokens,
    )


def compute_marginal_deltas(
    probe_rewards: Sequence[float],
    *,
    expected_stages: int | None = None,
) -> tuple[float, ...]:
    """Return signed prefix-quality differences ``Q(y_hat_t) - Q(y_hat_(t-1))``.

    The rollout caller supplies ungated answer-quality scores. Negative changes
    are retained rather than clipped, so unhelpful curation can receive negative
    credit.
    """

    rewards = tuple(_finite_number(value, "probe reward") for value in probe_rewards)
    if expected_stages is not None and len(rewards) != expected_stages + 1:
        raise ValueError(
            f"Expected {expected_stages + 1} probe rewards for {expected_stages} stages, "
            f"got {len(rewards)}."
        )
    if not rewards:
        raise ValueError("At least one probe reward is required.")
    return tuple(current - previous for previous, current in zip(rewards, rewards[1:]))


def build_marginal_utility_metadata(
    *,
    decision_spans: Sequence[Sequence[int]],
    probe_rewards: Sequence[float],
    probe_output_tokens: int,
    probe_seconds: float,
) -> dict[str, Any]:
    """Build compact successful-probe metadata stored beside the main rollout."""

    spans = tuple(_span(value) for value in decision_spans)
    rewards = tuple(_finite_number(value, "probe reward") for value in probe_rewards)
    deltas = compute_marginal_deltas(rewards, expected_stages=len(spans))
    output_tokens = int(probe_output_tokens)
    seconds = float(probe_seconds)
    if output_tokens < 0:
        raise ValueError("probe_output_tokens must be non-negative.")
    if not isfinite(seconds) or seconds < 0.0:
        raise ValueError("probe_seconds must be finite and non-negative.")
    return {
        "status": MARGINAL_UTILITY_OK,
        "stage_count": len(spans),
        "decision_spans": [list(span) for span in spans],
        "probe_rewards": list(rewards),
        "deltas": list(deltas),
        "probe_count": len(rewards),
        "probe_output_tokens": output_tokens,
        "probe_seconds": seconds,
    }


def build_marginal_utility_skip_metadata(
    status: str,
    *,
    stage_count: int,
    failure_type: str | None = None,
    probe_count: int = 0,
    probe_output_tokens: int = 0,
    probe_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build diagnostics for a trajectory that receives no marginal shaping."""

    if status not in {
        MARGINAL_UTILITY_NO_STAGES,
        MARGINAL_UTILITY_INVALID_FORMAT,
        MARGINAL_UTILITY_PROBE_FAILED,
    }:
        raise ValueError(f"Unsupported marginal-utility skip status: {status!r}.")
    if stage_count < 0:
        raise ValueError("stage_count must be non-negative.")
    if probe_count < 0:
        raise ValueError("probe_count must be non-negative.")
    metadata: dict[str, Any] = {
        "status": status,
        "stage_count": int(stage_count),
        "probe_count": int(probe_count),
        "probe_output_tokens": int(probe_output_tokens),
        "probe_seconds": float(probe_seconds),
    }
    if failure_type:
        metadata["failure_type"] = str(failure_type)
    return metadata


def parse_marginal_utility_credits(
    metadata: Mapping[str, Any],
    *,
    response_length: int,
) -> MarginalUtilityCredits:
    """Validate successful rollout metadata before touching trainer tensors."""

    if metadata.get("status") != MARGINAL_UTILITY_OK:
        raise ValueError("Only successful marginal-utility metadata contains credits.")
    raw_spans = metadata.get("decision_spans")
    raw_rewards = metadata.get("probe_rewards")
    raw_deltas = metadata.get("deltas")
    if not isinstance(raw_spans, (list, tuple)):
        raise ValueError("decision_spans must be a sequence.")
    if not isinstance(raw_rewards, (list, tuple)) or not isinstance(raw_deltas, (list, tuple)):
        raise ValueError("probe_rewards and deltas must be sequences.")

    spans = tuple(_span(value) for value in raw_spans)
    rewards = tuple(_finite_number(value, "probe reward") for value in raw_rewards)
    deltas = tuple(_finite_number(value, "marginal delta") for value in raw_deltas)
    expected_deltas = compute_marginal_deltas(rewards, expected_stages=len(spans))
    if len(deltas) != len(spans):
        raise ValueError("There must be one marginal delta per decision span.")
    if any(abs(actual - expected) > 1e-6 for actual, expected in zip(deltas, expected_deltas)):
        raise ValueError("Stored marginal deltas do not match adjacent probe rewards.")

    previous_end = 0
    for start, end in spans:
        if start < previous_end or end > response_length:
            raise ValueError("Marginal decision spans are overlapping, unordered, or out of range.")
        previous_end = end
    if int(metadata.get("stage_count", -1)) != len(spans):
        raise ValueError("stage_count does not match decision_spans.")
    if int(metadata.get("probe_count", -1)) != len(rewards):
        raise ValueError("probe_count does not match probe_rewards.")
    return MarginalUtilityCredits(
        decision_spans=spans,
        probe_rewards=rewards,
        deltas=deltas,
    )


def add_marginal_utility_credit(
    *,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    credits: MarginalUtilityCredits,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add eta times each marginal delta to its reasoning/action token span.

    Spans precede the corresponding tool observations and contain policy tokens
    only. All other tokens retain their trajectory-level advantages and returns.
    """

    if advantages.ndim != 1 or returns.shape != advantages.shape or response_mask.shape != advantages.shape:
        raise ValueError("advantages, returns, and response_mask must share one 1-D shape.")
    weight = _finite_number(weight, "marginal-utility weight")
    if weight < 0.0:
        raise ValueError("marginal-utility weight must be non-negative.")

    shaped_advantages = advantages.clone()
    shaped_returns = returns.clone()
    for (start, end), delta in zip(credits.decision_spans, credits.deltas, strict=True):
        stage_mask = response_mask[start:end].bool()
        if stage_mask.numel() == 0 or not torch.any(stage_mask).item():
            raise ValueError("A marginal decision span contains no policy tokens.")
        if not torch.all(stage_mask).item():
            raise ValueError("A marginal decision span crosses non-policy tokens.")
        credit = torch.as_tensor(
            weight * delta,
            dtype=shaped_advantages.dtype,
            device=shaped_advantages.device,
        )
        shaped_advantages[start:end] += credit
        shaped_returns[start:end] += credit.to(dtype=shaped_returns.dtype)
    return shaped_advantages, shaped_returns


def _span(value: Sequence[int]) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Each decision span must contain exactly [start, end].")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, Integral)
        or not isinstance(end, Integral)
    ):
        raise ValueError("Decision-span offsets must be integers.")
    start, end = int(start), int(end)
    if start < 0 or end <= start:
        raise ValueError("Decision spans must satisfy 0 <= start < end.")
    return start, end


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric.")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value
