"""Variant-aware accuracy/F1 reward for MemEye."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any, Mapping

from nltk.stem import PorterStemmer

from rewards.memory_qa import compose_memory_qa_reward, extract_tagged_answer


_STEMMER = PorterStemmer()
_DOT_PLACEHOLDER = "DOTPLACEHOLDER"
_UNDERSCORE_PLACEHOLDER = "UNDERSCOREPLACEHOLDER"
_ARTICLES = re.compile(r"\b(a|an|the)\s+(?=\w)")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Use exact option accuracy for MCQ rows and official token F1 for open rows."""

    del data_source
    truth = _parse_ground_truth(ground_truth)
    variant = str(truth.get("variant") or "").strip().lower()
    final_answer, answer_structure_is_valid = extract_tagged_answer(solution_str)

    if variant == "mcq":
        expected = str(truth.get("answer") or "").strip().upper()
        valid_choices = {
            str(choice).strip().upper()
            for choice in truth.get("valid_choices", [])
            if str(choice).strip()
        }
        if not expected or expected not in valid_choices:
            raise ValueError("MemEye MCQ ground truth requires an answer in valid_choices.")
        prediction = str(final_answer or "").strip().upper()
        answer_format_is_valid = bool(
            answer_structure_is_valid
            and re.fullmatch(r"[A-Z]", prediction)
            and prediction in valid_choices
        )
        accuracy = float(prediction == expected)
        reward = compose_memory_qa_reward(
            solution_str=solution_str,
            extra_info=extra_info,
            answer_format_is_valid=answer_format_is_valid,
            raw_performance_reward=accuracy,
        )
        return {
            **reward,
            "acc": accuracy,
            "accuracy": accuracy,
            "exact_match": accuracy,
            "reference_count": 1,
        }

    if variant != "open":
        raise ValueError(f"Unsupported MemEye reward variant: {variant!r}.")
    references = [
        str(answer).strip()
        for answer in truth.get("answers", [])
        if answer is not None and str(answer).strip()
    ]
    if not references:
        raise ValueError("MemEye open ground truth requires at least one answer.")
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


def _parse_ground_truth(value: Any) -> Mapping[str, Any]:
    data = value
    if not isinstance(data, Mapping):
        try:
            data = json.loads(str(value))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("MemEye ground truth must be a JSON object.") from exc
    if not isinstance(data, Mapping):
        raise ValueError("MemEye ground truth must decode to an object.")
    return data


def normalize_answer(value: Any) -> str:
    """Match MemEye's official universal answer normalization."""

    text = str(value or "").lower()
    text = re.sub(r"(?<=\d)\.(?=\d)", _DOT_PLACEHOLDER, text)
    text = text.replace("_", _UNDERSCORE_PLACEHOLDER)
    text = _ARTICLES.sub(" ", text)
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
