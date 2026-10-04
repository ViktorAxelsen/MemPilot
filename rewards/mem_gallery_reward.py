"""Answer-normalized token-F1 reward for Mem-Gallery."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from nltk.stem import PorterStemmer

from rewards.memory_qa import (
    compose_memory_qa_reward,
    extract_tagged_answer,
    parse_free_text_ground_truth,
)


_STEMMER = PorterStemmer()
_DOT_PLACEHOLDER = "DOTPLACEHOLDER"
_UNDERSCORE_PLACEHOLDER = "UNDERSCOREPLACEHOLDER"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Score free-text, date, yes/no, and image-ID answers with token F1."""
    del data_source
    references = parse_free_text_ground_truth(ground_truth)
    final_answer, answer_structure_is_valid = extract_tagged_answer(solution_str)
    answer_format_is_valid = bool(answer_structure_is_valid and final_answer)
    f1 = (
        max((token_f1(final_answer, reference) for reference in references), default=0.0)
        if final_answer is not None
        else 0.0
    )
    exact_match = float(
        final_answer is not None
        and any(normalize_answer(final_answer) == normalize_answer(reference) for reference in references)
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


def normalize_answer(value: Any) -> str:
    """Match Mem-Gallery's official universal answer normalization."""
    text = str(value or "").lower()
    text = re.sub(r"(?<=\d)\.(?=\d)", _DOT_PLACEHOLDER, text)
    text = text.replace("_", _UNDERSCORE_PLACEHOLDER)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    text = "".join(character if character not in string.punctuation else " " for character in text)
    text = text.replace(_DOT_PLACEHOLDER, ".")
    text = text.replace(_UNDERSCORE_PLACEHOLDER, "_")
    return " ".join(text.split())


def token_f1(prediction: Any, ground_truth: Any) -> float:
    prediction_tokens = [_STEMMER.stem(token) for token in normalize_answer(prediction).split()]
    ground_truth_tokens = [_STEMMER.stem(token) for token in normalize_answer(ground_truth).split()]
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)

    overlap = sum((Counter(prediction_tokens) & Counter(ground_truth_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)
