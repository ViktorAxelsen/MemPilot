"""H2HMem reward with shared token F1 and exact-match diagnostics."""

from __future__ import annotations

from typing import Any


from rewards.mem_gallery_reward import (
    normalize_answer as normalize_training_answer,
    token_f1,
)
from rewards.memory_qa import (
    compose_memory_qa_reward,
    extract_tagged_answer,
    parse_free_text_ground_truth,
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Score answers using shared token F1 and exact match."""

    del data_source
    references = parse_free_text_ground_truth(ground_truth)
    final_answer, answer_structure_is_valid = extract_tagged_answer(solution_str)
    answer_format_is_valid = bool(answer_structure_is_valid and final_answer)
    prediction = str(final_answer or "")
    f1 = max((token_f1(prediction, reference) for reference in references), default=0.0)
    exact_match = float(
        bool(final_answer)
        and any(
            normalize_training_answer(prediction) == normalize_training_answer(reference)
            for reference in references
        )
    )
    reward = compose_memory_qa_reward(
        solution_str=solution_str,
        extra_info=extra_info,
        answer_format_is_valid=answer_format_is_valid,
        raw_performance_reward=f1,
    )
    return {
        **reward,
        "f1": f1,
        "exact_match": exact_match,
        "reference_count": len(references),
    }
