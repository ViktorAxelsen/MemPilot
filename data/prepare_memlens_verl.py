"""Prepare one selected MEMLENS context-length configuration as test-only rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.memory_qa import (  # noqa: E402
    build_agentic_memory_row,
    write_parquet,
)
from data.multimodal_memory_data import (  # noqa: E402
    CHUNKING_MODES,
    DEFAULT_RETRIEVER_BATCH_SIZE,
    TOKEN_CHUNKING_MODE,
    attach_lossy_memory_banks,
    build_memory_bank,
    format_image_caption_block,
    normalize_chunking_mode,
    resolve_agentic_memory_bank,
)
from memory_chunking import DEFAULT_CHUNK_TOKENS, AtomicDialogueChunker  # noqa: E402
from memory_compression import (  # noqa: E402
    DEFAULT_LLMLINGUA2_MODEL,
    DEFAULT_MEMORY_COMPRESSION_RATE,
    LLMLingua2MemoryCompressor,
)
from retrieval import (  # noqa: E402
    EmbeddingRetriever,
    serialize_document_embeddings,
)


DEFAULT_DATASET_NAME = "xiyuRenBill/MEMLENS"
MEMLENS_CONTEXT_LENGTHS = ("32k", "64k", "128k", "256k")
MEMLENS_EVALUATION_SUBSETS = ("agent", "full")
DEFAULT_MEMLENS_EVALUATION_SUBSET = "agent"
AGENT_SUBSET_FILENAME = "agent_subset_195.json"
ANSWER_REFUSAL_TYPE = "answer_refusal"
MEMLENS_QUESTION_TYPES = frozenset(
    {
        "information_extraction",
        "knowledge_update",
        "temporal_reasoning",
        "multi_session_reasoning",
        ANSWER_REFUSAL_TYPE,
    }
)
EXPECTED_QUESTIONS = 789
EXPECTED_ANSWER_REFUSAL_QUESTIONS = 90
EXPECTED_DEFAULT_QUESTIONS = EXPECTED_QUESTIONS - EXPECTED_ANSWER_REFUSAL_QUESTIONS
EXPECTED_AGENT_QUESTIONS = 195
EXPECTED_AGENT_ANSWER_REFUSAL_QUESTIONS = 22
EXPECTED_AGENT_TYPE_COUNTS = {
    "information_extraction": 61,
    "multi_session_reasoning": 35,
    "temporal_reasoning": 48,
    "knowledge_update": 29,
    ANSWER_REFUSAL_TYPE: EXPECTED_AGENT_ANSWER_REFUSAL_QUESTIONS,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context_length",
        choices=MEMLENS_CONTEXT_LENGTHS,
        default="32k",
        help="Process exactly one official MEMLENS context-length configuration.",
    )
    parser.add_argument(
        "--evaluation_subset",
        choices=MEMLENS_EVALUATION_SUBSETS,
        default=DEFAULT_MEMLENS_EVALUATION_SUBSET,
        help=(
            "Evaluate the official 195-question memory-agent subset (default) or the full "
            "789-question benchmark. Answer-refusal filtering is controlled separately."
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Local MEMLENS repository root. The selected HF snapshot is downloaded when omitted.",
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--retriever_device", default="cuda:0")
    parser.add_argument("--retriever_batch_size", type=int, default=DEFAULT_RETRIEVER_BATCH_SIZE)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--chunking_mode", choices=CHUNKING_MODES, default=TOKEN_CHUNKING_MODE)
    parser.add_argument("--memory_compressor_model", default=DEFAULT_LLMLINGUA2_MODEL)
    parser.add_argument("--memory_compression_rate", type=float, default=DEFAULT_MEMORY_COMPRESSION_RATE)
    parser.add_argument("--compressor_device", default="cuda:0")
    parser.add_argument("--compressor_batch_size", type=int, default=32)
    parser.add_argument(
        "--disable_memory_compression",
        action="store_true",
        help="Skip the compressed-memory corpus; policy prompts remain query-only.",
    )
    parser.add_argument(
        "--include_answer_refusal",
        action="store_true",
        help="Retain the official answer_refusal questions; they are excluded by default.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_source", default="memlens")
    args = parser.parse_args()

    records = load_memlens_records(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        context_length=args.context_length,
        evaluation_subset=args.evaluation_subset,
    )
    validate_memlens_records(
        records,
        context_length=args.context_length,
        expected_count=expected_memlens_source_questions(args.evaluation_subset),
    )
    expected_questions = expected_memlens_questions(
        args.include_answer_refusal,
        evaluation_subset=args.evaluation_subset,
    )
    compressor = None
    if not args.disable_memory_compression:
        compressor = LLMLingua2MemoryCompressor(
            model_name=args.memory_compressor_model,
            rate=args.memory_compression_rate,
            device=args.compressor_device,
            batch_size=args.compressor_batch_size,
        )
    rows = build_rows(
        records=records,
        data_source=args.data_source,
        context_length=args.context_length,
        chunk_size=args.chunk_size,
        retriever_device=args.retriever_device,
        retriever_batch_size=args.retriever_batch_size,
        memory_compressor=compressor,
        show_progress=True,
        chunking_mode=args.chunking_mode,
        include_answer_refusal=args.include_answer_refusal,
    )
    if len(rows) != expected_questions:
        raise ValueError(f"Expected {expected_questions} MEMLENS rows, got {len(rows)}.")

    output_dir = Path(args.output_dir or f"data/memlens_{args.context_length}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(rows, output_dir / "test.parquet")
    context_tokens = [int(row["extra_info"]["context_token_count"]) for row in rows]
    print(
        f"Wrote {len(rows)} MEMLENS {args.context_length} {args.evaluation_subset} test rows to {output_dir}; "
        f"mean serialized context={sum(context_tokens) / len(context_tokens):.1f} tokens."
    )
    if compressor is not None:
        original_tokens = sum(row["extra_info"]["memory_compression"]["original_tokens"] for row in rows)
        compressed_tokens = sum(row["extra_info"]["memory_compression"]["compressed_tokens"] for row in rows)
        actual_rate = compressed_tokens / original_tokens if original_tokens else 0.0
        print(
            f"LLMLingua-2 memory bank: target_rate={compressor.rate:g}, "
            f"observed_rate={actual_rate:.3f}, unique_chunks={compressor.cache_size}, "
            f"model_calls={compressor.num_model_calls}"
        )


def normalize_context_length(value: Any) -> str:
    context_length = str(value or "").strip().lower()
    if context_length not in MEMLENS_CONTEXT_LENGTHS:
        raise ValueError(
            f"context_length must be one of {MEMLENS_CONTEXT_LENGTHS}; got {value!r}."
        )
    return context_length


def normalize_memlens_evaluation_subset(value: Any) -> str:
    evaluation_subset = str(value or "").strip().lower()
    if evaluation_subset not in MEMLENS_EVALUATION_SUBSETS:
        raise ValueError(
            f"evaluation_subset must be one of {MEMLENS_EVALUATION_SUBSETS}; got {value!r}."
        )
    return evaluation_subset


def load_memlens_records(
    dataset_dir: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    context_length: str = "32k",
    evaluation_subset: str = DEFAULT_MEMLENS_EVALUATION_SUBSET,
) -> list[dict[str, Any]]:
    """Load one context tier and retain only the requested official evaluation subset."""

    context_length = normalize_context_length(context_length)
    evaluation_subset = normalize_memlens_evaluation_subset(evaluation_subset)
    data_root = _resolve_data_root(
        dataset_dir,
        dataset_name,
        context_length,
        evaluation_subset=evaluation_subset,
    )
    data_path = data_root / f"dataset_{context_length}.json"
    with data_path.open("r", encoding="utf-8") as handle:
        raw_records = json.load(handle)
    if isinstance(raw_records, Mapping):
        raw_records = raw_records.get("data")
    if not isinstance(raw_records, list) or not all(isinstance(record, Mapping) for record in raw_records):
        raise ValueError(f"MEMLENS {data_path.name} must contain a list of objects.")
    if evaluation_subset == "agent":
        raw_records = select_memlens_records(
            raw_records,
            load_memlens_agent_question_ids(data_root / AGENT_SUBSET_FILENAME),
        )

    revision = _snapshot_revision(data_root)
    records = []
    for record in raw_records:
        copied = dict(record)
        copied["_memlens_data_root"] = os.path.abspath(data_root)
        copied["_memlens_dataset_name"] = str(dataset_name).strip()
        copied["_memlens_revision"] = revision
        copied["_memlens_context_length"] = context_length
        copied["_memlens_evaluation_subset"] = evaluation_subset
        records.append(copied)
    return records


def load_memlens_agent_question_ids(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing MEMLENS agent subset index: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MEMLENS agent subset JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("MEMLENS agent subset index must contain one object.")
    question_ids = payload.get("question_ids")
    if not isinstance(question_ids, list):
        raise ValueError("MEMLENS agent subset index requires question_ids as a list.")
    normalized_ids = [str(question_id or "").strip() for question_id in question_ids]
    if (
        len(normalized_ids) != EXPECTED_AGENT_QUESTIONS
        or any(not question_id for question_id in normalized_ids)
        or len(normalized_ids) != len(set(normalized_ids))
    ):
        raise ValueError(
            "MEMLENS agent subset must contain exactly "
            f"{EXPECTED_AGENT_QUESTIONS} unique non-empty question IDs."
        )
    if int(payload.get("n_questions", -1)) != EXPECTED_AGENT_QUESTIONS:
        raise ValueError("MEMLENS agent subset n_questions does not match question_ids.")
    per_type = payload.get("per_type")
    if not isinstance(per_type, Mapping) or {
        str(key): int(value) for key, value in per_type.items()
    } != EXPECTED_AGENT_TYPE_COUNTS:
        raise ValueError(
            f"Unexpected MEMLENS agent-subset type counts; expected {EXPECTED_AGENT_TYPE_COUNTS}."
        )
    return normalized_ids


def select_memlens_records(
    records: Iterable[Mapping[str, Any]],
    question_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Select records in the canonical index order and reject incomplete source data."""

    records_by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        question_id = str(record.get("question_id") or "").strip()
        if not question_id:
            raise ValueError("MEMLENS source records require non-empty question IDs.")
        if question_id in records_by_id:
            raise ValueError(f"Duplicate MEMLENS question ID in source data: {question_id!r}.")
        records_by_id[question_id] = record
    missing = [question_id for question_id in question_ids if question_id not in records_by_id]
    if missing:
        preview = ", ".join(repr(question_id) for question_id in missing[:5])
        raise ValueError(
            f"MEMLENS source data is missing {len(missing)} agent-subset questions: {preview}."
        )
    return [records_by_id[question_id] for question_id in question_ids]


def validate_memlens_records(
    records: Sequence[Mapping[str, Any]],
    *,
    context_length: str,
    expected_count: int = EXPECTED_AGENT_QUESTIONS,
) -> None:
    context_length = normalize_context_length(context_length)
    question_ids = [str(record.get("question_id") or "").strip() for record in records]
    if len(records) != expected_count:
        raise ValueError(
            f"Incomplete MEMLENS {context_length}: expected {expected_count} records, got {len(records)}."
        )
    if any(not question_id for question_id in question_ids) or len(question_ids) != len(set(question_ids)):
        raise ValueError(f"MEMLENS {context_length} question IDs must be non-empty and unique.")


def filter_memlens_records(
    records: Iterable[Mapping[str, Any]],
    *,
    include_answer_refusal: bool = False,
) -> list[Mapping[str, Any]]:
    records = list(records)
    if include_answer_refusal:
        return records
    return [
        record
        for record in records
        if str(record.get("question_type") or "").strip().lower() != ANSWER_REFUSAL_TYPE
    ]


def expected_memlens_source_questions(
    evaluation_subset: str = DEFAULT_MEMLENS_EVALUATION_SUBSET,
) -> int:
    evaluation_subset = normalize_memlens_evaluation_subset(evaluation_subset)
    return EXPECTED_AGENT_QUESTIONS if evaluation_subset == "agent" else EXPECTED_QUESTIONS


def expected_memlens_questions(
    include_answer_refusal: bool,
    evaluation_subset: str = DEFAULT_MEMLENS_EVALUATION_SUBSET,
) -> int:
    evaluation_subset = normalize_memlens_evaluation_subset(evaluation_subset)
    if evaluation_subset == "agent":
        return (
            EXPECTED_AGENT_QUESTIONS
            if include_answer_refusal
            else EXPECTED_AGENT_QUESTIONS - EXPECTED_AGENT_ANSWER_REFUSAL_QUESTIONS
        )
    return EXPECTED_QUESTIONS if include_answer_refusal else EXPECTED_DEFAULT_QUESTIONS


def _resolve_data_root(
    dataset_dir: str | Path | None,
    dataset_name: str,
    context_length: str,
    *,
    evaluation_subset: str = DEFAULT_MEMLENS_EVALUATION_SUBSET,
) -> Path:
    evaluation_subset = normalize_memlens_evaluation_subset(evaluation_subset)
    filename = f"dataset_{context_length}.json"
    if dataset_dir is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install `huggingface_hub` to download MEMLENS.") from exc
        dataset_name = str(dataset_name).strip().strip("/")
        if not dataset_name:
            raise ValueError("dataset_name must be non-empty.")
        root = Path(
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=[
                    filename,
                    "release_images/**",
                    *([AGENT_SUBSET_FILENAME] if evaluation_subset == "agent" else []),
                ],
            )
        )
    else:
        root = Path(dataset_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"MEMLENS dataset directory does not exist: {root}")

    required_files = [root / filename]
    if evaluation_subset == "agent":
        required_files.append(root / AGENT_SUBSET_FILENAME)
    if any(not path.is_file() for path in required_files) or not (root / "release_images").is_dir():
        raise FileNotFoundError(
            "MEMLENS requires "
            f"{', '.join(path.name for path in required_files)} and release_images/ under {root}."
        )
    return root.resolve()


def _snapshot_revision(data_root: Path) -> str:
    parts = data_root.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def build_rows(
    records: Iterable[Mapping[str, Any]],
    data_source: str,
    context_length: str,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    retriever_device: str = "cuda:0",
    retriever_batch_size: int = DEFAULT_RETRIEVER_BATCH_SIZE,
    embedding_retriever: EmbeddingRetriever | None = None,
    chunk_tokenizer: Any = None,
    memory_compressor: Any = None,
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
    include_answer_refusal: bool = False,
) -> list[dict[str, Any]]:
    context_length = normalize_context_length(context_length)
    records = filter_memlens_records(
        records,
        include_answer_refusal=include_answer_refusal,
    )
    retriever = embedding_retriever or EmbeddingRetriever(
        device=retriever_device,
        batch_size=retriever_batch_size,
    )
    examples = list(
        iter_memory_examples(
            records=records,
            context_length=context_length,
            chunk_size=chunk_size,
            retriever_device=retriever_device,
            retriever_batch_size=retriever_batch_size,
            embedding_retriever=retriever,
            chunk_tokenizer=chunk_tokenizer,
            show_progress=show_progress,
            chunking_mode=chunking_mode,
        )
    )
    if memory_compressor is not None:
        attach_lossy_memory_banks(
            examples,
            memory_compressor,
            retriever,
            dataset_label=f"MEMLENS {context_length}",
            show_progress=show_progress,
        )

    rows = []
    memory_is_compressed = memory_compressor is not None
    for row_index, example in enumerate(
        tqdm(
            examples,
            desc=f"Building MEMLENS {context_length} rows",
            unit="row",
            dynamic_ncols=True,
            disable=not show_progress,
        )
    ):
        runtime_memory_bank, compression = resolve_agentic_memory_bank(
            example, memory_is_compressed=memory_is_compressed
        )
        row_example = {
            **example,
            "memory_bank": runtime_memory_bank,
        }
        dataset_extra_info = _dataset_extra_info(example)
        rows.append(
            build_agentic_memory_row(
                example=row_example,
                data_source=f"{data_source}_{context_length}",
                row_index=row_index,
                ground_truth=build_memlens_ground_truth(example),
                chunk_size=chunk_size,
                compression=compression,
                dataset_extra_info=dataset_extra_info,
                metadata_extra={
                    "context_length": context_length,
                    "question_subtype": example["question_subtype"],
                    "question_date": example["question_date"],
                    "chunking_mode": example["chunking_mode"],
                },
                ability="multimodal_memory",
            )
        )
    return rows


def iter_memory_examples(
    records: Iterable[Mapping[str, Any]],
    context_length: str,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    retriever_device: str = "cuda:0",
    retriever_batch_size: int = DEFAULT_RETRIEVER_BATCH_SIZE,
    embedding_retriever: EmbeddingRetriever | None = None,
    chunk_tokenizer: Any = None,
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
) -> Iterable[dict[str, Any]]:
    context_length = normalize_context_length(context_length)
    chunking_mode = normalize_chunking_mode(chunking_mode)
    normalized_records = [
        normalize_memlens_record(record, record_index=index, context_length=context_length)
        for index, record in enumerate(records)
    ]
    retriever = embedding_retriever or EmbeddingRetriever(
        device=retriever_device,
        batch_size=retriever_batch_size,
    )
    tokenizer = chunk_tokenizer or retriever.tokenizer
    chunker = AtomicDialogueChunker(tokenizer, target_tokens=chunk_size)
    iterator = tqdm(
        normalized_records,
        desc=f"Preparing MEMLENS {context_length} memory",
        unit="question",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for record_index, record in enumerate(iterator):
        memory_bank = build_memory_bank(
            record,
            chunker,
            chunking_mode=chunking_mode,
            dataset_label="MEMLENS",
        )
        if not memory_bank:
            raise ValueError(f"MEMLENS question {record['conversation_id']!r} has no conversation memory.")
        document_embeddings = retriever.encode_documents(memory_bank)
        memory_index = serialize_document_embeddings(
            document_embeddings,
            model_id=retriever.model_id,
        )
        qa = record["qas"][0]
        context_token_count = sum(
            len(tokenizer.encode(item["text"], add_special_tokens=False)) for item in memory_bank
        )
        iterator.set_postfix(chunks=len(memory_bank), tokens=context_token_count, refresh=False)
        yield {
            "record_index": record_index,
            "conversation_id": record["conversation_id"],
            "split": "test",
            "question_id": qa["question_id"],
            "question_type": qa["question_type"],
            "question_subtype": qa["question_subtype"],
            "question": qa["question"],
            "question_date": qa["question_date"],
            "answer": qa["answer"],
            "reference_answers": [qa["answer"]],
            "context_length": context_length,
            "evaluation_subset": record["evaluation_subset"],
            "chunking_mode": chunking_mode,
            "context_token_count": context_token_count,
            "memory_chunk_count": len(memory_bank),
            "memory_bank": memory_bank,
            "memory_index": memory_index,
        }


def normalize_memlens_record(
    record: Mapping[str, Any],
    *,
    record_index: int,
    context_length: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"MEMLENS record {record_index} must be a mapping.")
    context_length = normalize_context_length(context_length)
    record_context_length = normalize_context_length(
        record.get("_memlens_context_length") or context_length
    )
    if record_context_length != context_length:
        raise ValueError(
            f"MEMLENS record context {record_context_length!r} does not match {context_length!r}."
        )
    question_id = _required_text(record.get("question_id"), "question_id")
    question_type = _required_text(record.get("question_type"), f"{question_id} question_type").lower()
    if question_type not in MEMLENS_QUESTION_TYPES:
        raise ValueError(f"Unsupported MEMLENS question_type: {question_type!r}.")
    question = _required_text(record.get("question"), f"{question_id} question")
    answer = _required_text(record.get("answer"), f"{question_id} answer")
    question_date = _required_text(record.get("question_date"), f"{question_id} question_date")
    question_subtype = str(record.get("question_subtype") or "").strip()
    evaluation_subset = normalize_memlens_evaluation_subset(
        record.get("_memlens_evaluation_subset") or "full"
    )

    raw_sessions = record.get("haystack_sessions")
    session_ids = record.get("haystack_session_ids")
    dates = record.get("haystack_dates")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(f"MEMLENS question {question_id!r} requires haystack_sessions.")
    if not isinstance(session_ids, list) or len(session_ids) != len(raw_sessions):
        raise ValueError(f"MEMLENS question {question_id!r} has misaligned haystack_session_ids.")
    if not isinstance(dates, list) or len(dates) != len(raw_sessions):
        raise ValueError(f"MEMLENS question {question_id!r} has misaligned haystack_dates.")

    data_root_text = str(record.get("_memlens_data_root") or "").strip()
    data_root = Path(data_root_text) if data_root_text else None
    dataset_name = str(record.get("_memlens_dataset_name") or DEFAULT_DATASET_NAME).strip()
    revision = str(record.get("_memlens_revision") or "").strip()
    sessions = []
    for session_index, raw_session in enumerate(raw_sessions):
        session_id = _required_text(session_ids[session_index], f"{question_id} session ID")
        date = _required_text(dates[session_index], f"{question_id}/{session_id} date")
        turns = raw_session
        if isinstance(raw_session, Mapping):
            session_id = str(raw_session.get("session_id") or session_id).strip()
            date = str(raw_session.get("date") or date).strip()
            turns = raw_session.get("session")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"MEMLENS session {session_id!r} requires a non-empty turn list.")
        normalized_turns = []
        for turn_index, turn in enumerate(turns):
            normalized_turn = _normalize_memlens_turn(
                turn,
                question_id=question_id,
                session_id=session_id,
                turn_index=turn_index,
                data_root=data_root,
                dataset_name=dataset_name,
                revision=revision,
            )
            if normalized_turn is not None:
                normalized_turns.append(normalized_turn)
        if normalized_turns:
            sessions.append(
                {
                    "session_id": session_id,
                    "date": date,
                    "turns": normalized_turns,
                }
            )

    return {
        "conversation_id": question_id,
        "evaluation_subset": evaluation_subset,
        "sessions": sessions,
        "qas": [
            {
                "question_id": question_id,
                "question_type": question_type,
                "question_subtype": question_subtype,
                "question": question,
                "question_date": question_date,
                "answer": answer,
            }
        ],
    }


def _normalize_memlens_turn(
    raw_turn: Any,
    *,
    question_id: str,
    session_id: str,
    turn_index: int,
    data_root: Path | None,
    dataset_name: str,
    revision: str,
) -> dict[str, Any] | None:
    if not isinstance(raw_turn, Mapping):
        raise ValueError(f"MEMLENS turn {turn_index} in {question_id}/{session_id} must be a mapping.")
    raw_role = str(raw_turn.get("role") or "").strip().casefold()
    role_map = {"user": "User", "assistant": "Assistant", "ai assistant": "Assistant"}
    role = role_map.get(raw_role)
    if role is None:
        raise ValueError(f"Unsupported MEMLENS role {raw_turn.get('role')!r} in {question_id}/{session_id}.")
    turn_id = f"{session_id}:T{turn_index + 1:03d}"
    content = str(raw_turn.get("content") or "").replace("<image>", "").strip()
    raw_images = raw_turn.get("images") or []
    if not isinstance(raw_images, list):
        raise ValueError(f"MEMLENS images in {turn_id!r} must be a list.")
    images = [
        _normalize_memlens_image(
            image,
            turn_id=turn_id,
            image_index=image_index,
            data_root=data_root,
            dataset_name=dataset_name,
            revision=revision,
        )
        for image_index, image in enumerate(raw_images)
    ]
    # The official files contain a handful of empty user turns. Their evaluator
    # emits no content block for these turns, so mirror that behavior here.
    if not content and not images:
        return None
    dialogue_text = f"[Turn {turn_id}]\n{role}: {content}".rstrip()
    image_text = format_image_caption_block(images, heading=f"Images attached to turn {turn_id}")
    return {
        "turn_id": turn_id,
        "text": f"{dialogue_text}\n{image_text}" if image_text else dialogue_text,
        "preview_text": dialogue_text,
        "protected_preview_text": image_text,
        "images": images,
    }


def _normalize_memlens_image(
    raw_image: Any,
    *,
    turn_id: str,
    image_index: int,
    data_root: Path | None,
    dataset_name: str,
    revision: str,
) -> dict[str, Any]:
    if not isinstance(raw_image, Mapping):
        raise ValueError(f"MEMLENS image {image_index} in {turn_id!r} must be a mapping.")
    filename = _required_text(raw_image.get("file"), f"{turn_id} image file").replace("\\", "/")
    filename = filename.removeprefix("./").removeprefix("release_images/")
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe MEMLENS image path: {filename!r}.")
    caption = _required_text(raw_image.get("blip_caption"), f"{turn_id} image caption")
    portable_path = (Path("release_images") / relative).as_posix()
    if data_root is not None and not (data_root / portable_path).is_file():
        raise FileNotFoundError(f"Missing MEMLENS image: {data_root / portable_path}")
    image = {
        "id": f"{turn_id}:IMG_{image_index + 1:03d}",
        "caption": caption,
        "turn_id": turn_id,
        "source": "memory",
        "path": portable_path,
    }
    if dataset_name:
        image.update(
            {
                "hf_repo_id": dataset_name,
                "hf_repo_type": "dataset",
                "hf_filename": portable_path,
            }
        )
        if revision:
            image["hf_revision"] = revision
    return image


def build_memlens_ground_truth(example: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "answers": [str(example["answer"]).strip()],
            "question_type": str(example["question_type"]).strip(),
            "question_subtype": str(example.get("question_subtype") or "").strip(),
        },
        ensure_ascii=False,
    )


def _dataset_extra_info(example: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reference_answers": list(example["reference_answers"]),
        "question_date": example["question_date"],
        "question_subtype": example["question_subtype"],
        "context_length": example["context_length"],
        "evaluation_subset": example["evaluation_subset"],
        "context_token_count": int(example["context_token_count"]),
        "chunking_mode": example["chunking_mode"],
        "query_images": [],
        "has_query_image": False,
        "official_metric": "llm_judge",
    }


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"MEMLENS {field_name} must be non-empty.")
    return text


if __name__ == "__main__":
    main()
