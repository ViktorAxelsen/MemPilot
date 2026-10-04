"""Policy-facing prompt views for MemPilot training and evaluation.

Shared orchestration instructions are combined with benchmark-specific answer
constraints; tool schemas provide the factorized action-argument contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from data.memory_qa import build_runtime_memory_system_prompt
from memory_access import DEFAULT_MAX_PARALLEL_CALLS


MEM_GALLERY_ANSWER_RULES = """Prefer the latest update when the history changes over time. When a question asks
for images, answer with the exact image_id value or values; for multiple images, use a comma-separated ascending
list. Follow any answer format explicitly requested in the question. Keep other answers concise and grounded."""

MEMEYE_ANSWER_RULES = """Prefer the latest update when the history changes over time. Keep the answer concise and
grounded in the conversation memory and relevant visual evidence."""

MEMLENS_ANSWER_RULES = """Use the question date and session dates when temporal order matters. Follow any exact
answer-format requirement in the question. Keep other answers concise and grounded in the conversation memory and
relevant visual evidence."""

WORLDMEMARENA_NO_EVIDENCE_ANSWER = "This information was not captured in the trajectory."
WORLDMEMARENA_ANSWER_RULES = (
    "Respect the visible history boundary and prefer the latest visible update when facts change over time. If "
    f'requested information is absent, answer exactly "{WORLDMEMARENA_NO_EVIDENCE_ANSWER}" instead of inventing it. '
    "Preserve exact image IDs, action names, and other identifiers when the question asks for them. Keep the answer "
    "concise and grounded in the conversation memory and relevant visual evidence."
)

H2HMEM_ANSWER_RULES = """Prefer the latest update when facts evolve across sessions, preserve speaker attribution,
and use dates or session order when temporal order matters. Follow any answer format explicitly requested in the
question. Keep the answer concise and grounded in the conversation memory and relevant visual evidence."""


def build_mem_gallery_system_prompt(
    *,
    max_parallel_calls: int = DEFAULT_MAX_PARALLEL_CALLS,
) -> str:
    """Build the multimodal controller contract for dynamic memory retrieval."""

    return build_runtime_memory_system_prompt(
        final_answer_contract=(
            "the final answer inside a single <answer>...</answer> tag. " + MEM_GALLERY_ANSWER_RULES
        ),
        include_image_argument_in_example=True,
        max_parallel_calls=max_parallel_calls,
    )


def build_memeye_system_prompt(
    *,
    answer_variant: str,
    max_parallel_calls: int = DEFAULT_MAX_PARALLEL_CALLS,
) -> str:
    """Build MemEye's variant-specific multimodal controller contract."""

    variant = str(answer_variant or "").strip().lower()
    if variant == "mcq":
        final_contract = (
            "exactly one option letter inside a single <answer>...</answer> tag. "
            + MEMEYE_ANSWER_RULES
        )
    elif variant == "open":
        final_contract = (
            "the concise free-form answer inside a single <answer>...</answer> tag. "
            + MEMEYE_ANSWER_RULES
        )
    else:
        raise ValueError(f"Unsupported MemEye answer variant: {answer_variant!r}.")

    return build_runtime_memory_system_prompt(
        final_answer_contract=final_contract,
        include_image_argument_in_example=True,
        max_parallel_calls=max_parallel_calls,
    )


def build_memlens_system_prompt(
    *,
    max_parallel_calls: int = DEFAULT_MAX_PARALLEL_CALLS,
) -> str:
    """Build MEMLENS's free-form multimodal controller contract."""

    return build_runtime_memory_system_prompt(
        final_answer_contract=(
            "the concise final answer inside a single <answer>...</answer> tag. "
            + MEMLENS_ANSWER_RULES
        ),
        include_image_argument_in_example=True,
        max_parallel_calls=max_parallel_calls,
    )


def build_worldmemarena_system_prompt(
    *,
    max_parallel_calls: int = DEFAULT_MAX_PARALLEL_CALLS,
) -> str:
    """Build WorldMemArena's free-text multimodal controller contract."""

    return build_runtime_memory_system_prompt(
        final_answer_contract=(
            "the concise final answer inside a single <answer>...</answer> tag. "
            + WORLDMEMARENA_ANSWER_RULES
        ),
        include_image_argument_in_example=True,
        max_parallel_calls=max_parallel_calls,
    )


def build_h2hmem_system_prompt(
    *,
    max_parallel_calls: int = DEFAULT_MAX_PARALLEL_CALLS,
) -> str:
    """Build H2HMem's free-form multimodal controller contract."""

    return build_runtime_memory_system_prompt(
        final_answer_contract=(
            "the concise final answer inside a single <answer>...</answer> tag. "
            + H2HMEM_ANSWER_RULES
        ),
        include_image_argument_in_example=True,
        max_parallel_calls=max_parallel_calls,
    )


def build_runtime_memory_messages(
    example: Mapping[str, Any],
    *,
    max_parallel_calls: int = DEFAULT_MAX_PARALLEL_CALLS,
) -> list[dict[str, str]]:
    """Build a query-only prompt; memory evidence is exposed only through tool calls."""

    extra_info = example.get("extra_info")
    if not isinstance(extra_info, Mapping):
        raise ValueError("Runtime-memory rows require mapping-valued extra_info.")
    dataset_kind = _dataset_kind(example)
    system_prompt = _build_system_prompt(
        dataset_kind,
        extra_info,
        max_parallel_calls=max_parallel_calls,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_query_user_prompt(dataset_kind, extra_info)},
    ]


def _build_system_prompt(
    dataset_kind: str,
    extra_info: Mapping[str, Any],
    *,
    max_parallel_calls: int,
) -> str:
    if dataset_kind == "mem_gallery":
        return build_mem_gallery_system_prompt(
            max_parallel_calls=max_parallel_calls,
        )
    if dataset_kind == "memeye":
        return build_memeye_system_prompt(
            answer_variant=str(extra_info.get("answer_variant") or ""),
            max_parallel_calls=max_parallel_calls,
        )
    if dataset_kind == "memlens":
        return build_memlens_system_prompt(
            max_parallel_calls=max_parallel_calls,
        )
    if dataset_kind == "worldmemarena":
        return build_worldmemarena_system_prompt(
            max_parallel_calls=max_parallel_calls,
        )
    if dataset_kind == "h2hmem":
        return build_h2hmem_system_prompt(
            max_parallel_calls=max_parallel_calls,
        )
    raise ValueError(f"Unsupported runtime-memory dataset kind: {dataset_kind!r}.")


def _build_query_user_prompt(
    dataset_kind: str,
    extra_info: Mapping[str, Any],
) -> str:
    question = str(extra_info.get("question") or "").strip()
    if not question:
        raise ValueError("Runtime-memory rows require a non-empty extra_info.question.")

    if dataset_kind == "mem_gallery":
        sections = []
        query_images = _image_records(extra_info.get("query_images"))
        if query_images:
            sections.append("Question Image:\n" + _format_image_caption_block(query_images))
        sections.append(f"Question: {question}")
        if str(extra_info.get("question_type") or "").strip().upper() == "CD":
            sections.append(
                "Answer-content constraint: the text inside <answer>...</answer> must be exactly Yes. or No."
            )
        sections.append("Answer:")
        return "\n\n".join(sections)

    if dataset_kind == "memeye":
        sections = []
        query_images = _image_records(extra_info.get("query_images"))
        if query_images:
            sections.append("Question Image:\n" + _format_image_caption_block(query_images))
        sections.append(f"Question: {question}")
        if str(extra_info.get("answer_variant") or "").strip().lower() == "mcq":
            choices = extra_info.get("choices")
            if not isinstance(choices, Mapping) or not choices:
                raise ValueError("MemEye MCQ rows require non-empty mapping-valued extra_info.choices.")
            choice_lines = [f"{key}. {value}" for key, value in choices.items()]
            sections.append("Options:\n" + "\n".join(choice_lines))
            sections.append("Answer with exactly one option letter inside <answer>...</answer>.")
        sections.append("Answer:")
        return "\n\n".join(sections)

    if dataset_kind == "memlens":
        question_date = str(extra_info.get("question_date") or "").strip()
        return f"Question Date: {question_date}\nQuestion: {question}\nAnswer:"

    if dataset_kind == "worldmemarena":
        return f"Question: {question}\nAnswer:"

    if dataset_kind == "h2hmem":
        sections = []
        query_images = _image_records(extra_info.get("query_images"))
        if query_images:
            sections.append("Question Image:\n" + _format_image_caption_block(query_images))
        sections.append(f"Question: {question}")
        if str(extra_info.get("question_type") or "").strip().casefold() == "conflict detection":
            sections.append(
                "Answer-content constraint: the text inside <answer>...</answer> must be exactly Yes or No."
            )
        sections.append("Answer:")
        return "\n\n".join(sections)

    raise ValueError(f"Unsupported runtime-memory dataset kind: {dataset_kind!r}.")


def _dataset_kind(example: Mapping[str, Any]) -> str:
    data_source = str(example.get("data_source") or "").strip().lower().replace("-", "_")
    if "h2hmem" in data_source:
        return "h2hmem"
    if "worldmemarena" in data_source:
        return "worldmemarena"
    if "memlens" in data_source:
        return "memlens"
    if "memeye" in data_source:
        return "memeye"
    if "mem_gallery" in data_source:
        return "mem_gallery"
    raise ValueError(f"Unsupported runtime-memory data_source: {data_source!r}.")


def _image_records(raw_images: Any) -> list[Mapping[str, Any]]:
    if not raw_images:
        return []
    if isinstance(raw_images, Mapping):
        return [raw_images]
    if isinstance(raw_images, Sequence) and not isinstance(raw_images, (str, bytes)):
        return [image for image in raw_images if isinstance(image, Mapping)]
    return []


def _format_image_caption_block(images: Sequence[Mapping[str, Any]]) -> str:
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
