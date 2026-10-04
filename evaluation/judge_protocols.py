"""Single source of truth for final benchmark answer-judge prompts.

Every executable prompt has exactly two sample-dependent inputs: the candidate
prediction and one or more ground-truth reference answers. Dataset metadata is
used only to choose a rubric; it is never rendered into the judge prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


GRADED_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
MEMEYE_REPORTED_SCORES = (0.0, 0.5, 1.0)
WORLD_LABELS = ("Correct", "Hallucination", "Omission")
BINARY_SCORES = (0.0, 1.0)


@dataclass(frozen=True)
class JudgeProtocol:
    """A benchmark-compatible answer correctness protocol."""

    protocol_id: str
    benchmark: str
    official_role: str
    rubric: str
    official_source_url: str
    score_values: tuple[float, ...] = GRADED_SCORES
    labels: tuple[str, ...] = ()
    requires_llm_judge: bool = True


_MEMORY_QA_JUDGE_ROLE = (
    "You are an impartial judge evaluating the memory capabilities of an AI assistant "
    "with the question-answering task."
)
_MEMEYE_JUDGE_ROLE = (
    "You are an impartial judge evaluating the memory capabilities of an AI assistant "
    "on a question-answering task."
)
_WORLD_JUDGE_ROLE = "You are an **evaluation expert for AI memory system question answering**."
_MEMLENS_JUDGE_ROLE = (
    "Now your role is a grading teacher. Your task is to review and score student answers "
    "based on reference standard answers for a question-answering benchmark."
)


_FIVE_LEVEL_RUBRIC = """**Score 0 (Incorrect / Miss):**
- The answer contradicts the Ground Truth.
- For Yes/No questions: The answer has the wrong polarity (e.g., says "Yes" when Ground Truth is "No").
- For Open-ended questions: The answer provides factually wrong information or hallucinations.
- The assistant fails to provide the required information.

**Score 0.25 (Poor / Tangential):**
- The answer touches on the topic but misses the **core entity** or key value required.
- The answer contains a mix of minor correct details and **significant hallucinations** or wrong associations.
- The answer is excessively vague to the point of being useless (e.g., answering "a dog" instead of "a golden retriever").

**Score 0.5 (Partial / Vague):**
- The answer is technically correct, but lacks confidence or is incomplete.
- The answer captures the **main entity or concept** correctly but misses a part of the required supporting details.
- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant is uncertain (e.g., "I think it might be Yes").
- For Open-ended questions: The answer is too general or misses key adjectives/details present in the Ground Truth.

**Score 0.75 (Good / Minor Imperfection):**
- The answer is largely accurate and captures the core information confidently.
- It misses only **minor details** (e.g., specific adjectives or secondary details) that do not alter the main truth.
- The answer contains all the correct information but includes unnecessary "fluff" or slight conversational filler that reduces precision.

**Score 1 (Correct / Exact):**
- The answer is accurate, precise, and confident.
- For Yes/No questions: The polarity matches the Ground Truth perfectly.
- For Open-ended questions: The answer contains **all** the core information and necessary details required by the Ground Truth without hallucinations."""

_MEMEYE_RUBRIC = """### Important Principles

1. **Semantic equivalence over surface form.** The assistant may use different wording, synonyms, or sentence structure. As long as the meaning matches the Ground Truth, it should receive full credit. Do NOT penalize for stylistic differences.
2. **Numeric / counting questions are binary.** If the Ground Truth is a specific number (e.g., "3"), the answer is either exactly right (score 1) or wrong (score 0). There is no partial credit for being off by one.
3. **Negation + correction questions.** Some Ground Truth answers first negate a premise and then provide the correct information (e.g., "No — it was May 18-22, not May 15-19"). The assistant must get BOTH the negation AND the corrected fact right for full credit.
4. **Identity / "Who is" questions.** The Ground Truth may describe a person by role, actions, or relationships rather than by name. The assistant's answer is correct if it identifies the same unique person, even if it uses a name instead of a description or vice versa.
5. **Answer the thing that was asked.** Judge whether the assistant answered the requested attribute or type of information. If the question asks for a category, role, relationship, time, count, ordering, or specific item, an answer giving a different type of information is not fully correct even if it is related.
6. **Wrong entity beats related entity.** If the answer names the wrong person, object, brand, place, event, or option, do not give high scores just because it is on-topic. Being from the right semantic domain is not enough.
7. **Multi-item questions require set matching.** Full credit requires all required items and no incorrect extras. If only some required items are correct, partial credit may be appropriate.
8. **Ordering / chronology questions require the correct item.** If the question asks which item was first, last, middle, earlier, later, before, or after, the selected item must match the Ground Truth. A different plausible item is still wrong.
9. **Do not over-reward fluent explanations.** Extra detail is helpful only if it stays consistent with the Ground Truth. Confident wording or elaboration should not increase the score when the core fact is wrong.
10. **Contradicted relations or orderings are usually zero.** If the question asks about a relationship such as before/after, earlier/later, first/last, middle, sequence, relative timing, or other ordering, and the assistant gives the opposite relation, denies the relation, or selects the wrong ordered item, this is usually a clear contradiction and should normally receive score 0 rather than 0.25.

### Scoring Rubric

**Score 0 (Incorrect / Miss):**

- The answer contradicts the Ground Truth.
- For Yes/No questions: The answer has the wrong polarity (e.g., says "Yes" when Ground Truth is "No").
- For numeric questions: The number is wrong.
- For attribute-type mismatches: the answer gives the wrong kind of thing (for example, a brand when the question asks for a product category, or a related event when the question asks for the earliest specific item).
- For relation/order questions: if the answer contradicts the required before/after, earlier/later, first/last, middle, or similar relationship, score 0 in the usual case.
- The answer provides factually wrong information, hallucinations, or fails to provide any relevant information.

**Score 0.25 (Poor / Tangential):**

- The answer touches on the topic but misses the **core entity** or key value required.
- The answer contains a mix of minor correct details and **significant hallucinations** or wrong associations.
- The answer is excessively vague to the point of being useless (e.g., answering "a dog" instead of "a golden retriever").
- For multi-item questions: use 0.25 when the answer is clearly about the right topic but most of the required set is wrong.

**Score 0.5 (Partial / Vague):**

- The answer captures the **main entity or concept** correctly but misses important supporting details.
- For negation + correction questions: The polarity is correct but the corrected fact is missing or wrong, or vice versa.
- The answer is too general or misses key qualifiers present in the Ground Truth.
- For multi-item questions: use 0.5 when the answer gets a substantial part of the required set right (for example, one of two required items is correct and the rest is wrong or missing).

**Score 0.75 (Good / Minor Imperfection):**

- The answer is largely accurate and captures the core information confidently.
- It misses only **minor details** (e.g., specific adjectives or secondary details) that do not alter the main truth.
- The answer contains all the correct information but includes unnecessary filler that reduces precision.

**Score 1 (Correct / Exact):**

- The answer is accurate, precise, and confident.
- For Yes/No questions: The polarity matches the Ground Truth perfectly.
- For numeric questions: The number matches exactly.
- For descriptive questions: The answer contains **all** the core information and necessary details required by the Ground Truth without hallucinations.
- Semantic equivalence is sufficient — exact wording is NOT required.
- For multi-item questions: all required items are present and there are no incorrect extra items."""

_WORLD_RUBRIC = """### 1. Correct
* The response accurately answers the question and is **semantically equivalent** to the Reference Answer.
* No contradictions with the Reference Answer.
* Synonyms, paraphrasing, and reasonable summarization are acceptable.

### 2. Hallucination
* The response includes information that **contradicts** the Reference Answer.
* When the Reference Answer is *unknown/uncertain*, yet the response provides a specific fact.

### 3. Omission
* The response is **incomplete** compared to the Reference Answer.
* It states "don't know" or "no related memory" even though the Reference Answer supplies the answer.
* For multi-element answers, missing **any** element counts as Omission.

## Priority Rules
* Both missing info AND fabricated info -> **Hallucination**.
* No fabrication but missing info -> **Omission**.
* Fully equivalent -> **Correct**."""

_MEMLENS_COMMON = """First, extract the final answer from the student's solution, then judge whether the answer is correct. Score only the final answer; intermediate problem-solving steps do not need to be correct. If multiple inconsistent answers are given, or no clear final answer is stated, assign 0 points. Only the student's last clearly stated answer counts.

An item is covered if it is strictly mentioned or unambiguously implied by semantic equivalence. This includes numerical equivalence, synonyms, plural/singular forms, and equivalent date formats. Do not accept loosely related concepts. Ignore minor formatting differences, capitalization, punctuation, and equivalent wording when meaning is unchanged.

The scoring scale has two levels: 1 point and 0 points. Assign 1 point if the student's final answer matches the standard answer under the task-specific criteria below. Assign 0 points otherwise, including refusal or a claim of insufficient information except for answer-refusal tasks."""


def _graded_protocol(
    protocol_id: str,
    benchmark: str,
    source_url: str,
    *,
    official_role: str = _MEMORY_QA_JUDGE_ROLE,
    rubric: str = _FIVE_LEVEL_RUBRIC,
    score_values: tuple[float, ...] = GRADED_SCORES,
) -> JudgeProtocol:
    return JudgeProtocol(
        protocol_id=protocol_id,
        benchmark=benchmark,
        official_role=official_role,
        rubric=rubric,
        official_source_url=source_url,
        score_values=score_values,
    )


PROTOCOLS: dict[str, JudgeProtocol] = {
    "mem_gallery_answer_v1": _graded_protocol(
        "mem_gallery_answer_v1",
        "Mem-Gallery",
        "https://github.com/YuanchenBei/Mem-Gallery/blob/main/benchmark/memengine/evaluate/llm_judge.txt",
    ),
    "h2hmem_answer_v1": _graded_protocol(
        "h2hmem_answer_v1",
        "H2HMem",
        "https://github.com/varib1/H2HMEM/blob/main/evaluate_metrics/LLM-as-judge/LLM_as_judge.py",
    ),
    "memeye_open_answer_v1": _graded_protocol(
        "memeye_open_answer_v1",
        "MemEye-open",
        "https://github.com/MinghoKwok/MemEye/blob/main/benchmark/llm_judge.txt",
        official_role=_MEMEYE_JUDGE_ROLE,
        rubric=_MEMEYE_RUBRIC,
        score_values=MEMEYE_REPORTED_SCORES,
    ),
    "worldmemarena_answer_v1": JudgeProtocol(
        protocol_id="worldmemarena_answer_v1",
        benchmark="WorldMemArena",
        official_role=_WORLD_JUDGE_ROLE,
        rubric=_WORLD_RUBRIC,
        official_source_url=(
            "https://github.com/UCSB-AI/WorldMemArena/blob/main/eval_framework/judges/prompts.py"
        ),
        score_values=BINARY_SCORES,
        labels=WORLD_LABELS,
    ),
}


def _register_memlens(protocol_id: str, criterion: str) -> None:
    PROTOCOLS[protocol_id] = JudgeProtocol(
        protocol_id=protocol_id,
        benchmark="MEMLENS",
        official_role=_MEMLENS_JUDGE_ROLE,
        rubric=f"{_MEMLENS_COMMON}\n\n[Task-Specific Criteria]:\n{criterion}",
        official_source_url="https://github.com/xrenaf/MEMLENS/tree/main/judge_prompts",
        score_values=BINARY_SCORES,
        labels=("Correct", "Incorrect"),
    )


_register_memlens(
    "memlens_ie_entity_v1",
    """[IE — Entity / Attribute Extraction]
This is an information extraction question about a concrete entity, count, attribute, text, object, location, or visual detail from the provided memory context.
- Assign 1 point if the student's response contains the core information from the standard answer. Minor wording differences are acceptable, but the essential information must be present and correct.
- Assign 0 points if the core information is missing, contradicted, too vague, refused, or incorrect.
- For numeric or count answers, require the exact value unless the standard answer itself is approximate.""",
)
_register_memlens(
    "memlens_ie_previous_info_v1",
    """[IE — Previous Information Extraction]
This is an information extraction question about previously mentioned information, spatial relations, counts, or attributes from earlier conversation/image context.
- Assign 1 point if the student's response recovers the same previous information as the standard answer, including the correct relation, location, count, or attribute.
- Minor wording differences are acceptable if they preserve the same meaning.
- Assign 0 points if the answer refers to the wrong previous item, gives the wrong relation/count/location, is too vague, refuses, or is unsupported.""",
)
_register_memlens(
    "memlens_knowledge_update_v1",
    """[KU — Knowledge Update]
This is a knowledge update question testing whether the model tracks the latest state.
- Assign 1 point if the student gives the current (correct) answer.
- Assign 0 points if the student gives an outdated answer, any other incorrect answer, refuses, or does not clearly answer.""",
)
_register_memlens(
    "memlens_msr_yes_no_v1",
    """[MSR — Yes/No (Entity Resolution)]
This is a yes/no question.
- Assign 1 point if the student's final answer is semantically equivalent to the standard answer (e.g., "Yes" matches "Yes", "No" matches "No"), even if phrased differently (e.g., "That's correct" for "Yes").
- Assign 0 points if the answer means the opposite, refuses, or does not clearly answer.""",
)
_register_memlens(
    "memlens_msr_counting_v1",
    """[MSR — Counting]
This is a counting question. The standard answer is a specific number.
- Assign 1 point if the student clearly states the exact number in any equivalent form (e.g., "3", "three", "3.0").
- Assign 0 points if the student gives a different number, a range, an approximation, refuses, or does not clearly answer.""",
)
_register_memlens(
    "memlens_msr_arithmetic_v1",
    """[MSR — Arithmetic]
This is an arithmetic question. The standard answer is a specific value.
- Assign 1 point if the student gives the same value in an equivalent format (e.g., "0.5" and "1/2" and "50%").
- Assign 0 points if the student gives a different value, an unsupported approximation, refuses, or does not clearly answer.""",
)
_register_memlens(
    "memlens_temporal_duration_v1",
    """[TR — Duration Comparison (A/B)]
This is a binary choice question. The student must select one of two options.
- Assign 1 point if the student clearly selects the same option as the standard answer, regardless of phrasing.
- Assign 0 points if the student selects the other option, refuses, or does not make a clear selection.""",
)
_register_memlens(
    "memlens_temporal_order_v1",
    """[TR — Order Ranking]
This is an ordering question. The standard answer is an exact sequence of items.
- Assign 1 point ONLY if the student gives the exact same sequence in the same order.
- Assign 0 points if any item is missing, added, duplicated, misplaced, or if the order is unclear.""",
)
_register_memlens(
    "memlens_temporal_date_v1",
    """[TR — Date Extraction]
This is a date extraction question. The standard answer is a specific date.
- Assign 1 point if the student gives the same date in any unambiguous equivalent format (e.g., "Jan 15, 2024" and "2024-01-15" and "15/01/2024").
- Assign 0 points if the date is different, incomplete when a full date is required, ambiguous, or missing.""",
)
_register_memlens(
    "memlens_answer_refusal_v1",
    """[AR — Answer Refusal]
- Assign 1 point if the student explicitly refuses, says there is not enough information, or indicates the question is unanswerable from the provided context.
- Assign 0 points if the student gives a substantive answer, even if hedged.""",
)


def get_judge_protocol(protocol_id: str) -> JudgeProtocol:
    """Return one registered protocol or fail loudly on protocol drift."""

    try:
        return PROTOCOLS[str(protocol_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown judge protocol: {protocol_id!r}.") from exc


def judge_protocol_id_for_sample(
    data_source: Any,
    metadata: Mapping[str, Any] | None,
    ground_truth: Any,
) -> str:
    """Choose a protocol without exposing metadata to the judge model."""

    source = str(data_source or "").strip().casefold().replace("-", "_")
    metadata = metadata or {}
    truth = _ground_truth_mapping(ground_truth)
    if "worldmemarena" in source:
        return "worldmemarena_answer_v1"
    if "h2hmem" in source:
        return "h2hmem_answer_v1"
    if "mem_gallery" in source:
        return "mem_gallery_answer_v1"
    if "memeye" in source:
        variant = str(
            metadata.get("answer_variant") or truth.get("variant") or source.rsplit("_", 1)[-1]
        ).strip().casefold()
        if variant == "open":
            return "memeye_open_answer_v1"
        raise ValueError(
            "Final MemEye evaluation supports only the open-answer split; "
            f"received variant={variant or 'unknown'!r}."
        )
    if "memlens" in source:
        return _memlens_protocol_id(metadata, truth)
    raise ValueError(f"No final-evaluation judge protocol for data_source={data_source!r}.")


def render_judge_prompt(
    protocol_id: str,
    *,
    prediction: Any,
    references: Sequence[Any],
) -> str:
    """Render a prompt whose only sample-dependent inputs are prediction and GT."""

    protocol = get_judge_protocol(protocol_id)
    if not protocol.requires_llm_judge:
        raise ValueError(f"Protocol {protocol_id!r} is deterministic and has no LLM prompt.")
    clean_references = [str(reference).strip() for reference in references if str(reference).strip()]
    if not clean_references:
        raise ValueError("Judge prompts require at least one non-empty ground-truth reference.")
    reference_text = (
        clean_references[0]
        if len(clean_references) == 1
        else json.dumps(clean_references, ensure_ascii=False)
    )
    prediction_text = str(prediction or "").strip()
    if protocol.labels == WORLD_LABELS:
        return f"""{protocol.official_role}
Based **only** on the provided **Reference Answer**, strictly evaluate the **accuracy** of the **Memory System Response**. Classify it as one of **Correct**, **Hallucination**, or **Omission**. Do **not** use any external knowledge or subjective inference.

# Evaluation Criteria
{protocol.rubric}

# Information
* **Reference Answer:** {reference_text}
* **Memory System Response:** {prediction_text}

# Output
```json
{{
  "reasoning": "Concise evaluation rationale",
  "evaluation_result": "Correct | Hallucination | Omission"
}}
```"""
    if protocol.score_values == BINARY_SCORES:
        return f"""{protocol.official_role}
{protocol.rubric}

Your output format is:
[Scoring Rationale]:
[Score]: x points
[JSON]: {{"answer_score": <0 or 1>}}

[Current Case]
Standard Answer: {reference_text}
Student Answer: {prediction_text}
[Scoring Rationale]:"""
    reasoning_placeholder = (
        "" if protocol.protocol_id == "mem_gallery_answer_v1" else "<short explanation>"
    )
    rubric_heading = (
        "" if protocol.protocol_id == "memeye_open_answer_v1" else "### Scoring Rubric\n\n"
    )
    return f"""{protocol.official_role}
Your task is to compare the Assistant's Answer against the Ground Truth and assign a score of 0, 0.25, 0.5, 0.75, or 1.

{rubric_heading}{protocol.rubric}

### Input Data

Ground Truth: {reference_text}
Assistant Answer: {prediction_text}

### Output Format

Output strictly in the following JSON format:
{{"score": <0, 0.25, 0.5, 0.75, or 1>, "reasoning": "{reasoning_placeholder}"}}"""


def _memlens_protocol_id(
    metadata: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> str:
    question_type = str(
        metadata.get("question_type") or truth.get("question_type") or ""
    ).strip().casefold()
    subtype = str(
        metadata.get("question_subtype") or truth.get("question_subtype") or ""
    ).strip().casefold()
    references = _reference_values(truth)
    first_reference = references[0].casefold() if references else ""
    if question_type == "information_extraction":
        return "memlens_ie_previous_info_v1" if subtype == "previnfo" else "memlens_ie_entity_v1"
    if question_type == "knowledge_update":
        return "memlens_knowledge_update_v1"
    if question_type == "multi_session_reasoning":
        if subtype == "arithmetic":
            return "memlens_msr_arithmetic_v1"
        if subtype == "entity_resolution" and first_reference in {"yes", "no"}:
            return "memlens_msr_yes_no_v1"
        return "memlens_msr_counting_v1"
    if question_type == "temporal_reasoning":
        if subtype == "duration_comparison":
            return "memlens_temporal_duration_v1"
        if subtype == "order_ranking":
            return "memlens_temporal_order_v1"
        return "memlens_temporal_date_v1"
    if question_type == "answer_refusal":
        return "memlens_answer_refusal_v1"
    raise ValueError(
        f"Unsupported MEMLENS judge type/subtype: {question_type!r}/{subtype!r}."
    )


def _ground_truth_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return {"answers": [str(value)]}
    return decoded if isinstance(decoded, Mapping) else {"answers": decoded}


def _reference_values(truth: Mapping[str, Any]) -> list[str]:
    values: Any = truth.get("answers")
    if values is None and truth.get("answer") is not None:
        values = [truth.get("answer")]
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [str(value).strip() for value in values if value is not None and str(value).strip()]
