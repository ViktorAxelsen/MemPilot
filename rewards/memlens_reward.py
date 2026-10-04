"""Official deterministic SubEM/F1 proxy for MEMLENS evaluation rollouts."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any, Mapping

from rewards.memory_qa import compose_memory_qa_reward, extract_tagged_answer


ANSWER_REFUSAL_TYPE = "answer_refusal"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Use MEMLENS's official deterministic token F1 as the local reward proxy."""

    del data_source
    truth = _parse_ground_truth(ground_truth)
    references = [
        str(answer).strip()
        for answer in truth.get("answers", [])
        if answer is not None and str(answer).strip()
    ]
    if not references:
        raise ValueError("MEMLENS ground truth requires at least one answer.")
    question_type = str(truth.get("question_type") or "").strip().lower()
    if not question_type:
        raise ValueError("MEMLENS ground truth requires question_type.")

    final_answer, answer_structure_is_valid = extract_tagged_answer(solution_str)
    answer_format_is_valid = bool(answer_structure_is_valid and final_answer)
    prediction = str(final_answer or "")
    is_refusal = is_insufficient_information(prediction)
    f1 = max((token_f1(prediction, reference) for reference in references), default=0.0)
    sub_em = float(
        any(
            normalized_reference
            and normalized_reference in normalize_answer(prediction)
            for normalized_reference in (normalize_answer(reference) for reference in references)
        )
    )
    is_abstention = question_type == ANSWER_REFUSAL_TYPE
    abstention_accuracy = float(is_refusal) if is_abstention else 0.0
    performance = abstention_accuracy if is_abstention else f1
    reward = compose_memory_qa_reward(
        solution_str=solution_str,
        extra_info=extra_info,
        answer_format_is_valid=answer_format_is_valid,
        raw_performance_reward=performance,
    )
    return {
        **reward,
        "f1": f1 if not is_abstention else 0.0,
        "sub_em": sub_em if not is_abstention else 0.0,
        "abstention_accuracy": abstention_accuracy,
        "is_abstention": int(is_abstention),
        "is_refusal": int(is_refusal),
        "reference_count": len(references),
    }


def _parse_ground_truth(value: Any) -> Mapping[str, Any]:
    data = value
    if not isinstance(data, Mapping):
        try:
            data = json.loads(str(value))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("MEMLENS ground truth must be a JSON object.") from exc
    if not isinstance(data, Mapping):
        raise ValueError("MEMLENS ground truth must decode to an object.")
    return data


def normalize_answer(value: Any) -> str:
    """Match MEMLENS's official deterministic SubEM/F1 normalization."""

    text = str(value or "").lower()
    normalized = []
    punctuation = set(string.punctuation)
    for index, character in enumerate(text):
        if character not in punctuation:
            normalized.append(character)
        elif (
            character in "./"
            and 0 < index < len(text) - 1
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            normalized.append(character)
        else:
            normalized.append(" ")
    text = "".join(normalized)
    text = re.sub(r"\ba(?=\s+[a-z])", " ", text)
    text = re.sub(r"\b(an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: Any, ground_truth: Any) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(ground_truth_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def is_insufficient_information(value: Any) -> bool:
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return True
    patterns = (
        r"\b(?:in)?sufficient information\b",
        r"\bnot enough (?:information|context|details|data)\b",
        r"\bcannot (?:determine|answer|provide|find|identify)\b",
        r"\bunable to (?:determine|answer|provide|find|identify)\b",
        r"\b(?:i )?(?:do not|don't) (?:know|have enough information)\b",
        r"\b(?:no|without) (?:information|context|details|data)\b",
        r"\b(?:does not|doesn't) provide (?:enough|sufficient)\b",
        r"\bcannot be determined\b",
        r"\bunanswerable\b",
        r"\b(?:lack|lacking) (?:information|context|details)\b",
        r"\bnone\b",
        r"^\s*n/a\s*$",
        r"^\s*unknown\s*$",
    )
    return any(re.search(pattern, text) for pattern in patterns)
