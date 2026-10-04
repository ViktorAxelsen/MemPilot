"""Benchmark-specific QA templates for policy-answer model decoupling.

The answer model receives the question and collected memory evidence, not the
orchestrator's reasoning, routing decisions, or original final answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_ANSWER_TAG_CONTRACT = (
    "Return only one non-empty <answer>...</answer> block, with the answer text inside the tags."
)

_H2HMEM_INSTRUCTIONS = {
    "unimodal precise recall": "Accurately recall the requested information from the conversation memory.",
    "cross-modal related retrieval": "Use related textual and visual information from the conversation memory.",
    "knowledge resolution": "Resolve knowledge consistently across the conversation memory.",
    "temporal reasoning": "Reason about temporal relationships in the conversation memory.",
    "multimodal causal reasoning": "Use textual and visual evidence for the requested causal reasoning.",
    "reference & evolution tracking": "Track references and how they evolve across the conversation memory.",
    "test-time learning": "Use knowledge introduced in the conversation memory to answer the question.",
    "conflict detection": "Determine whether the information in the question conflicts with the conversation memory.",
    "answer refusal": "Determine whether the question can be answered from the conversation memory.",
}


def build_benchmark_answer_messages(
    *,
    dataset_source: str,
    sample_metadata: Mapping[str, Any],
    memory_evidence: Sequence[str],
) -> list[dict[str, str]]:
    """Build a direct benchmark QA prompt without runtime-agent instructions."""

    if isinstance(memory_evidence, (str, bytes)):
        raise ValueError("memory_evidence must be a sequence of evidence strings.")
    evidence = [str(item).strip() for item in memory_evidence if str(item).strip()]
    if not evidence:
        raise ValueError("Direct answer generation requires at least one non-empty memory evidence item.")
    dataset_kind = _dataset_kind(
        str(dataset_source or sample_metadata.get("dataset_source") or "")
    )
    return [
        {
            "role": "system",
            "content": _system_prompt(dataset_kind, sample_metadata),
        },
        {
            "role": "user",
            "content": _user_prompt(dataset_kind, sample_metadata, evidence),
        },
    ]


def _system_prompt(dataset_kind: str, metadata: Mapping[str, Any]) -> str:
    if dataset_kind == "mem_gallery":
        task_rules = (
            "Answer from the provided multimodal conversation memory. Ground the answer in that memory, prefer "
            "the latest information when details change over time, and keep the response concise but complete. "
            "When the question asks for images, preserve the exact image_id values."
        )
    elif dataset_kind == "memeye":
        task_rules = (
            "Answer from the provided long-horizon multimodal conversation memory. Ground the answer in the "
            "memory and relevant visual evidence; when evidence conflicts, use the newest visual evidence unless "
            "the question itself asks about the conflict. For counts, give the number; for yes/no questions, begin "
            "with Yes or No; keep descriptive answers to one or two sentences. If the requested information is "
            "absent, state that it is unavailable."
        )
    elif dataset_kind == "memlens":
        task_rules = (
            "Answer from the provided conversation history. If the question cannot be answered from it, the "
            'answer text must be exactly "Insufficient information".'
        )
    elif dataset_kind == "worldmemarena":
        task_rules = (
            "Answer from the provided retrieved memories. Synthesize multiple entries when needed, use timestamps "
            "and the most recent evidence to resolve conflicts, and ground every claim in the memories. Convert "
            "relative dates using the memory timestamps. When images are requested, preserve exact image_id "
            "values. Keep the answer concise; if no relevant information is present, the answer text must be "
            'exactly "Not mentioned in memory."'
        )
    else:
        question_type = str(metadata.get("question_type") or "").strip().casefold()
        type_instruction = _H2HMEM_INSTRUCTIONS.get(
            question_type,
            _H2HMEM_INSTRUCTIONS["unimodal precise recall"],
        )
        task_rules = (
            f"You are a memory testing system. {type_instruction} Answer directly in English without reasoning, "
            "using no more than 100 words."
        )
        if question_type == "conflict detection":
            task_rules += " The answer text must be strictly either Yes or No."
        elif question_type == "answer refusal":
            task_rules += ' If the information is absent, the answer text must be exactly "Not mentioned."'
    return f"{task_rules} {_ANSWER_TAG_CONTRACT}"


def _user_prompt(
    dataset_kind: str,
    metadata: Mapping[str, Any],
    evidence: Sequence[str],
) -> str:
    question = str(metadata.get("question") or "").strip()
    if not question:
        raise ValueError("Direct answer generation requires sample_metadata.question.")

    memory_heading = {
        "memlens": "Conversation history",
        "worldmemarena": "Retrieved memories",
    }.get(dataset_kind, "Conversation memory")
    evidence_text = "\n\n".join(
        f"[Evidence {index}]\n{item}" for index, item in enumerate(evidence, start=1)
    )
    sections = [f"{memory_heading}:\n{evidence_text}"]

    image_context = _format_query_images(metadata.get("query_images"))
    if image_context:
        sections.append("Question Image:\n" + image_context)
    if dataset_kind == "memlens":
        question_date = str(metadata.get("question_date") or "").strip()
        if question_date:
            sections.append(f"Question Date: {question_date}")
    sections.append(f"Question: {question}")

    if dataset_kind == "mem_gallery":
        question_type = str(metadata.get("question_type") or "").strip().upper()
        if question_type == "CD":
            sections.append('The text inside <answer>...</answer> must be exactly "Yes." or "No."')
        elif question_type == "VS":
            sections.append(
                "Return the exact matching image_id value or values. If several apply, sort them in ascending "
                "order and separate them with commas."
            )
    elif dataset_kind == "memeye" and str(
        metadata.get("answer_variant") or ""
    ).strip().casefold() == "mcq":
        choices = metadata.get("choices")
        if not isinstance(choices, Mapping) or not choices:
            raise ValueError("MemEye MCQ answer generation requires mapping-valued choices.")
        sections.append(
            "Options:\n" + "\n".join(f"{key}. {value}" for key, value in choices.items())
        )
        sections.append("Return exactly one option letter.")
    return "\n\n".join(sections)


def _dataset_kind(dataset_source: str) -> str:
    source = str(dataset_source or "").strip().casefold().replace("-", "_")
    for marker, dataset_kind in (
        ("h2hmem", "h2hmem"),
        ("worldmemarena", "worldmemarena"),
        ("memlens", "memlens"),
        ("memeye", "memeye"),
        ("mem_gallery", "mem_gallery"),
    ):
        if marker in source:
            return dataset_kind
    raise ValueError(f"No final-evaluation answer prompt is registered for {dataset_source!r}.")


def _format_query_images(raw_images: Any) -> str:
    if isinstance(raw_images, Mapping):
        images = [raw_images]
    elif isinstance(raw_images, Sequence) and not isinstance(raw_images, (str, bytes)):
        images = [image for image in raw_images if isinstance(image, Mapping)]
    else:
        images = []
    lines = []
    for image in images:
        image_id = " ".join(str(image.get("id") or "").split())
        caption = " ".join(str(image.get("caption") or "").split())
        if not image_id:
            continue
        lines.append(f"- image_id: {image_id}")
        if caption:
            lines.append(f"  image_caption: {caption}")
    return "\n".join(lines)
