"""Prepare one H2HMem interaction variant for runtime-memory GRPO."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.memory_qa import (  # noqa: E402
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VALIDATION_RATIO,
    build_agentic_memory_row,
    build_free_text_ground_truth,
    sample_train_subset,
    split_rows,
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
    QWEN3_RETRIEVER,
    EmbeddingRetriever,
    serialize_document_embeddings,
)


DEFAULT_DATASET_NAME = "varib/H2HMEM"
H2HMEM_VARIANTS = ("dyadic", "multiparty")
H2HMEM_SOURCE_DIRS = {"dyadic": "dyadic", "multiparty": "multi-party"}
OFFICIAL_DIALOGUE_COUNTS = {"dyadic": 20, "multiparty": 5}
OFFICIAL_QA_COUNTS = {"dyadic": 2046, "multiparty": 190}
ANSWER_REFUSAL_TYPE = "Answer Refusal"
CONFLICT_DETECTION_TYPE = "Conflict Detection"
QUERY_IMAGE_ID = "query_image"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=H2HMEM_VARIANTS, default="dyadic")
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Local H2HMem root or selected variant directory. Downloads the selected HF variant when omitted.",
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
    parser.add_argument("--disable_memory_compression", action="store_true")
    parser.add_argument(
        "--include_answer_refusal",
        action="store_true",
        help="Include official Answer Refusal questions; they are excluded by default.",
    )
    parser.add_argument(
        "--include_cd_answer_format_mismatches",
        action="store_true",
        help=(
            "Retain Conflict Detection questions whose reference answer is not a standalone Yes/No label; "
            "they are excluded by default because they conflict with the answer-format prompt."
        ),
    )
    parser.add_argument(
        "--include_incomplete_dialogues",
        action="store_true",
        help=(
            "Retain dialogues with a nonzero session whose questions/images exist but session.json is missing. "
            "Disabled by default because those questions lack recoverable source context."
        ),
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_source", default=None)
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--validation_ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--train_subset_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    variant = normalize_h2hmem_variant(args.variant)
    records = load_h2hmem_records(
        args.dataset_dir,
        args.dataset_name,
        variant,
        include_incomplete_dialogues=args.include_incomplete_dialogues,
    )
    audit = validate_h2hmem_records(records, variant=variant)
    print(
        f"H2HMem {variant}: {audit['dialogues']} usable dialogues, {audit['sessions']} content sessions, "
        f"{audit['qas']} QA pairs ({audit['answer_refusal_qas']} Answer Refusal, "
        f"{audit['cd_answer_format_mismatch_qas']} incompatible Conflict Detection labels)."
    )

    compressor = None
    if not args.disable_memory_compression:
        compressor = LLMLingua2MemoryCompressor(
            model_name=args.memory_compressor_model,
            rate=args.memory_compression_rate,
            device=args.compressor_device,
            batch_size=args.compressor_batch_size,
        )

    data_source = args.data_source or f"h2hmem_{variant}"
    rows = build_rows(
        records=records,
        data_source=data_source,
        chunk_size=args.chunk_size,
        retriever_device=args.retriever_device,
        retriever_batch_size=args.retriever_batch_size,
        memory_compressor=compressor,
        include_answer_refusal=args.include_answer_refusal,
        include_cd_answer_format_mismatches=args.include_cd_answer_format_mismatches,
        show_progress=True,
        chunking_mode=args.chunking_mode,
        variant=variant,
    )
    if not rows:
        raise ValueError(f"No H2HMem {variant} QA rows were produced.")

    train_rows, validation_rows, test_rows = split_rows(
        rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    full_train_size = len(train_rows)
    train_rows = sample_train_subset(train_rows, size=args.train_subset_size, seed=args.seed)
    output_dir = Path(args.output_dir or f"data/h2hmem_{variant}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train_rows, output_dir / "train.parquet")
    write_parquet(validation_rows, output_dir / "val.parquet")
    write_parquet(test_rows, output_dir / "test.parquet")

    train_dialogues = _conversation_ids(train_rows)
    validation_dialogues = _conversation_ids(validation_rows)
    test_dialogues = _conversation_ids(test_rows)
    if (
        train_dialogues & validation_dialogues
        or train_dialogues & test_dialogues
        or validation_dialogues & test_dialogues
    ):
        raise ValueError("H2HMem dialogue leakage detected across train/validation/test splits.")
    subset_note = f" sampled from {full_train_size}" if args.train_subset_size is not None else ""
    print(
        f"Wrote {len(train_rows)} train rows{subset_note} from {len(train_dialogues)} dialogues, "
        f"{len(validation_rows)} validation rows from {len(validation_dialogues)} dialogues, and "
        f"{len(test_rows)} test rows from {len(test_dialogues)} dialogues to {output_dir}"
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


def normalize_h2hmem_variant(value: Any) -> str:
    variant = str(value or "").strip().lower().replace("_", "-")
    if variant == "multi-party":
        variant = "multiparty"
    if variant not in H2HMEM_VARIANTS:
        raise ValueError(f"H2HMem variant must be one of {H2HMEM_VARIANTS}; got {value!r}.")
    return variant


def load_h2hmem_records(
    dataset_dir: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    variant: str = "dyadic",
    *,
    include_incomplete_dialogues: bool = False,
) -> list[dict[str, Any]]:
    """Load dialogue descriptors while excluding irrecoverably incomplete conversations by default."""

    variant = normalize_h2hmem_variant(variant)
    data_root, variant_dir = _resolve_h2hmem_dirs(dataset_dir, dataset_name, variant)
    dialogue_paths = sorted(
        (path for path in variant_dir.glob("dialogue*") if path.is_dir()),
        key=_numbered_path_key,
    )
    if not dialogue_paths:
        raise FileNotFoundError(f"No H2HMem dialogues found under {variant_dir}.")
    if dataset_dir is None:
        expected_dialogues = OFFICIAL_DIALOGUE_COUNTS[variant]
        if len(dialogue_paths) != expected_dialogues:
            raise ValueError(
                f"Incomplete official H2HMem {variant} snapshot: expected {expected_dialogues} dialogues, "
                f"found {len(dialogue_paths)}."
            )
        source_qa_count = _source_qa_count(dialogue_paths)
        expected_qas = OFFICIAL_QA_COUNTS[variant]
        if source_qa_count != expected_qas:
            raise ValueError(
                f"Incomplete official H2HMem {variant} snapshot: expected {expected_qas} QA pairs, "
                f"found {source_qa_count}."
            )

    records = []
    revision = _snapshot_revision(data_root)
    skipped = []
    for dialogue_path in tqdm(
        dialogue_paths,
        desc=f"Loading H2HMem {variant} dialogues",
        unit="dialogue",
        dynamic_ncols=True,
    ):
        incomplete_sessions = _incomplete_nonzero_sessions(dialogue_path)
        if incomplete_sessions and not include_incomplete_dialogues:
            skipped.append((dialogue_path.name, incomplete_sessions))
            continue
        records.append(
            {
                "_h2hmem_dialogue_path": os.path.abspath(dialogue_path),
                "_h2hmem_data_root": os.path.abspath(data_root),
                "_h2hmem_dialogue_name": dialogue_path.name,
                "_h2hmem_dataset_name": str(dataset_name).strip(),
                "_h2hmem_revision": revision,
                "_h2hmem_variant": variant,
                "_h2hmem_incomplete_sessions": incomplete_sessions,
            }
        )
    if skipped:
        details = ", ".join(f"{name} ({'/'.join(sessions)})" for name, sessions in skipped)
        warnings.warn(
            "Excluded H2HMem dialogue(s) with missing nonzero-session content: " + details,
            stacklevel=2,
        )
    if not records:
        raise ValueError(f"No usable H2HMem {variant} dialogues remain after integrity filtering.")
    return records


def _resolve_h2hmem_dirs(
    dataset_dir: str | Path | None,
    dataset_name: str,
    variant: str,
) -> tuple[Path, Path]:
    source_dir = H2HMEM_SOURCE_DIRS[variant]
    if dataset_dir is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install `huggingface_hub` to download H2HMem.") from exc
        dataset_name = str(dataset_name).strip().strip("/")
        if not dataset_name:
            raise ValueError("dataset_name must be non-empty.")
        root = Path(
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=[f"{source_dir}/**"],
            )
        )
    else:
        root = Path(dataset_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"H2HMem dataset directory does not exist: {root}")

    if (root / source_dir).is_dir():
        data_root, variant_dir = root, root / source_dir
    elif root.name == source_dir and any(root.glob("dialogue*")):
        data_root, variant_dir = root.parent, root
    else:
        raise FileNotFoundError(
            f"H2HMem {variant} requires {source_dir}/dialogue*/scenes under {root}, "
            f"or dataset_dir may point directly to {source_dir}/."
        )
    return data_root.resolve(), variant_dir.resolve()


def _snapshot_revision(data_root: Path) -> str:
    parts = data_root.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def _incomplete_nonzero_sessions(dialogue_path: Path) -> list[str]:
    scenes_dir = dialogue_path / "scenes"
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"H2HMem dialogue is missing scenes/: {dialogue_path}")
    incomplete = []
    for session_dir in sorted(
        (path for path in scenes_dir.glob("session*") if path.is_dir()),
        key=_numbered_path_key,
    ):
        if session_dir.name.lower() == "session0":
            continue
        session_file = session_dir / "session.json"
        if not session_file.is_file():
            incomplete.append(session_dir.name)
            continue
        data = _load_json_object(session_file, "session")
        if not isinstance(data.get("dialogue"), list) or not data["dialogue"]:
            incomplete.append(session_dir.name)
    return incomplete


def _source_qa_count(dialogue_paths: Iterable[Path]) -> int:
    total = 0
    for dialogue_path in dialogue_paths:
        for question_file in (dialogue_path / "scenes").glob("session*/questions.json"):
            questions = _load_json_object(question_file, "questions").get("questions")
            if not isinstance(questions, list):
                raise ValueError(f"H2HMem questions must be a list: {question_file}")
            total += len(questions)
    return total


def validate_h2hmem_records(
    records: Iterable[Mapping[str, Any]],
    *,
    variant: str,
) -> dict[str, int]:
    """Validate all source JSON before loading retriever/compressor models."""

    variant = normalize_h2hmem_variant(variant)
    normalized = [
        normalize_h2hmem_record(record, record_index=index, variant=variant)
        for index, record in enumerate(records)
    ]
    if not normalized:
        raise ValueError(f"H2HMem {variant} requires at least one usable dialogue.")
    conversation_ids = [record["conversation_id"] for record in normalized]
    if len(conversation_ids) != len(set(conversation_ids)):
        raise ValueError(f"H2HMem {variant} conversation IDs must be unique.")
    question_ids = [qa["question_id"] for record in normalized for qa in record["qas"]]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"H2HMem {variant} question IDs must be globally unique.")
    return {
        "dialogues": len(normalized),
        "sessions": sum(len(record["sessions"]) for record in normalized),
        "qas": len(question_ids),
        "answer_refusal_qas": sum(
            qa["question_type"] == ANSWER_REFUSAL_TYPE
            for record in normalized
            for qa in record["qas"]
        ),
        "cd_answer_format_mismatch_qas": sum(
            not is_cd_answer_format_compatible(qa["question_type"], qa["answer"])
            for record in normalized
            for qa in record["qas"]
        ),
    }


def build_rows(
    records: Iterable[Mapping[str, Any]],
    data_source: str,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    retriever_device: str = "cuda:0",
    retriever_batch_size: int = DEFAULT_RETRIEVER_BATCH_SIZE,
    embedding_retriever: EmbeddingRetriever | None = None,
    chunk_tokenizer: Any = None,
    memory_compressor: Any = None,
    include_answer_refusal: bool = False,
    include_cd_answer_format_mismatches: bool = False,
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
    variant: str = "dyadic",
) -> list[dict[str, Any]]:
    retriever = embedding_retriever or EmbeddingRetriever(
        device=retriever_device,
        batch_size=retriever_batch_size,
    )
    examples = list(
        iter_memory_examples(
            records=records,
            chunk_size=chunk_size,
            retriever_device=retriever_device,
            retriever_batch_size=retriever_batch_size,
            embedding_retriever=retriever,
            chunk_tokenizer=chunk_tokenizer,
            include_answer_refusal=include_answer_refusal,
            include_cd_answer_format_mismatches=include_cd_answer_format_mismatches,
            show_progress=show_progress,
            chunking_mode=chunking_mode,
            variant=variant,
        )
    )
    if memory_compressor is not None:
        attach_lossy_memory_banks(
            examples,
            memory_compressor,
            retriever,
            dataset_label=f"H2HMem {variant}",
            show_progress=show_progress,
        )

    memory_is_compressed = memory_compressor is not None
    rows = []
    for example in tqdm(
        examples,
        desc=f"Building H2HMem {variant} rows",
        unit="row",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        runtime_memory_bank, compression = resolve_agentic_memory_bank(
            example, memory_is_compressed=memory_is_compressed
        )
        row_example = {**example, "memory_bank": runtime_memory_bank}
        query_images = [dict(image) for image in example.get("query_images", [])]
        rows.append(
            build_agentic_memory_row(
                example=row_example,
                data_source=data_source,
                row_index=len(rows),
                ground_truth=build_free_text_ground_truth(example["reference_answers"]),
                chunk_size=chunk_size,
                compression=compression,
                dataset_extra_info=_dataset_extra_info(example),
                metadata_extra={
                    "variant": example["variant"],
                    "question_type": example["question_type"],
                    "has_query_image": bool(query_images),
                    "chunking_mode": example["chunking_mode"],
                },
                tool_create_kwargs={"query_images": query_images},
                ability="multimodal_memory",
            )
        )
    if show_progress:
        _print_context_statistics(examples, variant=variant)
    return rows


def iter_memory_examples(
    records: Iterable[Mapping[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    retriever_device: str = "cuda:0",
    retriever_batch_size: int = DEFAULT_RETRIEVER_BATCH_SIZE,
    embedding_retriever: EmbeddingRetriever | None = None,
    chunk_tokenizer: Any = None,
    include_answer_refusal: bool = False,
    include_cd_answer_format_mismatches: bool = False,
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
    variant: str = "dyadic",
) -> Iterable[dict[str, Any]]:
    variant = normalize_h2hmem_variant(variant)
    chunking_mode = normalize_chunking_mode(chunking_mode)
    raw_records = list(records)
    normalized_records = [
        normalize_h2hmem_record(record, record_index=index, variant=variant)
        for index, record in enumerate(raw_records)
    ]
    retriever = embedding_retriever or EmbeddingRetriever(
        device=retriever_device,
        batch_size=retriever_batch_size,
    )
    tokenizer = chunk_tokenizer or retriever.tokenizer
    chunker = AtomicDialogueChunker(tokenizer, target_tokens=chunk_size)
    iterator = tqdm(
        enumerate(normalized_records),
        total=len(normalized_records),
        desc=f"Preparing H2HMem {variant} memory",
        unit="dialogue",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    qa_progress = tqdm(
        total=sum(
            _keep_h2hmem_qa(
                qa,
                include_answer_refusal=include_answer_refusal,
                include_cd_answer_format_mismatches=include_cd_answer_format_mismatches,
            )
            for record in normalized_records
            for qa in record["qas"]
        ),
        desc=f"Preparing H2HMem {variant} QAs",
        unit="qa",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for record_index, record in iterator:
        qas = [
            qa
            for qa in record["qas"]
            if _keep_h2hmem_qa(
                qa,
                include_answer_refusal=include_answer_refusal,
                include_cd_answer_format_mismatches=include_cd_answer_format_mismatches,
            )
        ]
        if not qas:
            continue
        memory_bank = build_memory_bank(
            record,
            chunker,
            chunking_mode=chunking_mode,
            dataset_label=f"H2HMem {variant}",
        )
        if not memory_bank:
            raise ValueError(f"H2HMem dialogue {record['conversation_id']!r} has no source memory.")
        document_embeddings = retriever.encode_documents(memory_bank)
        memory_index = serialize_document_embeddings(document_embeddings, model_id=retriever.model_id)
        context_token_count = sum(
            len(tokenizer.encode(str(item.get("text") or ""), add_special_tokens=False))
            for item in memory_bank
        )
        iterator.set_postfix(chunks=len(memory_bank), qas=len(qas), tokens=context_token_count, refresh=False)
        for qa in qas:
            qa_progress.update(1)
            yield {
                "record_index": record_index,
                "conversation_id": record["conversation_id"],
                "split": "",
                "question_id": qa["question_id"],
                "question_type": qa["question_type"],
                "question_type_main": qa["question_type_main"],
                "question": qa["question"],
                "answer": qa["answer"],
                "reference_answers": [qa["answer"]],
                "query_images": qa["query_images"],
                "variant": variant,
                "dialogue_name": record["dialogue_name"],
                "question_session": qa["question_session"],
                "difficulty": qa["difficulty"],
                "validated": qa["validated"],
                "chunking_mode": chunking_mode,
                "context_token_count": context_token_count,
                "conversation_session_count": len(record["sessions"]),
                "memory_chunk_count": len(memory_bank),
                "memory_bank": memory_bank,
                "memory_index": memory_index,
            }
    qa_progress.close()


def normalize_h2hmem_record(
    record: Mapping[str, Any],
    *,
    record_index: int,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"H2HMem record {record_index} must be a mapping.")
    variant = normalize_h2hmem_variant(variant)
    record_variant = normalize_h2hmem_variant(record.get("_h2hmem_variant") or variant)
    if record_variant != variant:
        raise ValueError(f"H2HMem record variant {record_variant!r} does not match {variant!r}.")
    dialogue_path = Path(_required_text(record.get("_h2hmem_dialogue_path"), "dialogue path"))
    data_root = Path(_required_text(record.get("_h2hmem_data_root"), "dataset root"))
    dialogue_name = _required_text(
        record.get("_h2hmem_dialogue_name") or dialogue_path.name,
        "dialogue name",
    )
    dataset_name = str(record.get("_h2hmem_dataset_name") or DEFAULT_DATASET_NAME).strip()
    revision = str(record.get("_h2hmem_revision") or "").strip()
    conversation_id = f"h2hmem:{variant}:{dialogue_name}"
    scenes_dir = dialogue_path / "scenes"
    session_dirs = sorted(
        (path for path in scenes_dir.glob("session*") if path.is_dir()),
        key=_numbered_path_key,
    )
    if not session_dirs:
        raise ValueError(f"H2HMem dialogue {conversation_id!r} has no session directories.")

    image_id_counts: dict[str, int] = {}
    sessions = []
    for session_dir in session_dirs:
        session_file = session_dir / "session.json"
        if not session_file.is_file():
            # session0 intentionally stores only cross-session questions. A nonzero
            # missing session reaches here only with --include_incomplete_dialogues.
            continue
        session_data = _load_json_object(session_file, "session")
        raw_turns = session_data.get("dialogue")
        if not isinstance(raw_turns, list) or not raw_turns:
            continue
        session_id = session_dir.name
        turns = []
        for turn_index, raw_turn in enumerate(raw_turns):
            turn = _normalize_h2hmem_turn(
                raw_turn,
                conversation_id=conversation_id,
                session_id=session_id,
                turn_index=turn_index,
                session_dir=session_dir,
                data_root=data_root,
                dataset_name=dataset_name,
                revision=revision,
                image_id_counts=image_id_counts,
            )
            if turn is not None:
                turns.append(turn)
        if not turns:
            continue
        sessions.append(
            {
                "session_id": session_id,
                "date": str(session_data.get("timeline_date") or "").strip(),
                "turns": turns,
            }
        )
    if not sessions:
        raise ValueError(f"H2HMem dialogue {conversation_id!r} contains no usable session content.")

    qas = []
    for session_dir in session_dirs:
        question_file = session_dir / "questions.json"
        if not question_file.is_file():
            continue
        question_data = _load_json_object(question_file, "questions")
        raw_qas = question_data.get("questions")
        if not isinstance(raw_qas, list):
            raise ValueError(f"H2HMem questions must be a list: {question_file}")
        qas.extend(
            _normalize_h2hmem_qa(
                raw_qa,
                conversation_id=conversation_id,
                dialogue_path=dialogue_path,
                question_session=session_dir.name,
                qa_index=qa_index,
                data_root=data_root,
                dataset_name=dataset_name,
                revision=revision,
            )
            for qa_index, raw_qa in enumerate(raw_qas)
        )
    if not qas:
        raise ValueError(f"H2HMem dialogue {conversation_id!r} contains no QA pairs.")
    return {
        "conversation_id": conversation_id,
        "dialogue_name": dialogue_name,
        "variant": variant,
        "sessions": sessions,
        "qas": qas,
    }


def _normalize_h2hmem_turn(
    raw_turn: Any,
    *,
    conversation_id: str,
    session_id: str,
    turn_index: int,
    session_dir: Path,
    data_root: Path,
    dataset_name: str,
    revision: str,
    image_id_counts: dict[str, int],
) -> dict[str, Any] | None:
    if not isinstance(raw_turn, Mapping):
        raise ValueError(f"H2HMem turn {turn_index} in {conversation_id!r} must be a mapping.")
    content = raw_turn.get("content")
    if not isinstance(content, Mapping):
        raise ValueError(f"H2HMem turn {turn_index} in {conversation_id!r} requires content.")
    role = _required_text(raw_turn.get("role"), f"{conversation_id}/{session_id} turn role")
    text = str(content.get("text") or "").strip()
    image_filename = str(content.get("image") or "").strip()
    if not text and not image_filename:
        return None
    turn_id = f"{session_id}:T{turn_index + 1:03d}"
    images = []
    if image_filename:
        base_image_id = f"{session_id}/{image_filename}"
        occurrence = image_id_counts.get(base_image_id, 0) + 1
        image_id_counts[base_image_id] = occurrence
        image_id = base_image_id if occurrence == 1 else f"{base_image_id}@turn{turn_index + 1}"
        image_path = session_dir / "image" / image_filename
        images.append(
            {
                "id": image_id,
                "caption": _load_image_caption(session_dir, image_filename),
                "turn_id": turn_id,
                "source": "memory",
                **_portable_image_reference(
                    image_path,
                    data_root=data_root,
                    dataset_name=dataset_name,
                    revision=revision,
                ),
            }
        )
    dialogue_text = f"[Session {session_id}, turn {turn_index + 1}]\n{role}: {text}".rstrip()
    image_text = format_image_caption_block(images, heading=f"Images attached to {turn_id}")
    return {
        "turn_id": turn_id,
        "text": f"{dialogue_text}\n{image_text}" if image_text else dialogue_text,
        "preview_text": dialogue_text,
        "protected_preview_text": image_text,
        "images": images,
    }


def _normalize_h2hmem_qa(
    raw_qa: Any,
    *,
    conversation_id: str,
    dialogue_path: Path,
    question_session: str,
    qa_index: int,
    data_root: Path,
    dataset_name: str,
    revision: str,
) -> dict[str, Any]:
    if not isinstance(raw_qa, Mapping):
        raise ValueError(f"H2HMem QA {qa_index} in {conversation_id!r} must be a mapping.")
    raw_question = raw_qa.get("question")
    if not isinstance(raw_question, Mapping):
        raise ValueError(f"H2HMem QA {qa_index} in {conversation_id!r} requires a question object.")
    question = _required_text(raw_question.get("text"), f"{conversation_id}/{question_session} question")
    answer = _required_text(raw_qa.get("original_answer"), f"{conversation_id}/{question_session} answer")
    raw_type = raw_qa.get("question_type")
    if not isinstance(raw_type, Mapping):
        raise ValueError(f"H2HMem QA {qa_index} in {conversation_id!r} requires question_type.")
    question_type_main = _required_text(raw_type.get("main_type"), "question_type.main_type")
    question_type = _required_text(raw_type.get("sub_type"), "question_type.sub_type")
    source_id = str(
        raw_qa.get("original_question_id")
        or raw_qa.get("question_id")
        or f"Q{qa_index + 1:03d}"
    ).strip()
    question_id = f"{conversation_id}:{question_session}:{source_id}"

    query_images = []
    image_reference = str(raw_question.get("image") or "").strip()
    if image_reference:
        if question_session.lower() == "session0":
            normalized = image_reference.replace("\\", "/").removeprefix("./")
            parts = Path(normalized).parts
            if len(parts) < 2:
                raise ValueError(
                    f"H2HMem cross-session query image must include its session folder: {image_reference!r}."
                )
            image_session = parts[0]
            image_filename = Path(*parts[1:]).as_posix()
        else:
            image_session = question_session
            image_filename = image_reference.replace("\\", "/").removeprefix("./")
        image_session_dir = dialogue_path / "scenes" / image_session
        image_path = image_session_dir / "image" / image_filename
        query_images.append(
            {
                "id": QUERY_IMAGE_ID,
                "caption": _load_image_caption(image_session_dir, image_filename),
                "source": "query",
                **_portable_image_reference(
                    image_path,
                    data_root=data_root,
                    dataset_name=dataset_name,
                    revision=revision,
                ),
            }
        )
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question_type_main": question_type_main,
        "question": question,
        "answer": answer,
        "question_session": question_session,
        "difficulty": str(raw_qa.get("difficulty") or "").strip(),
        "validated": bool(raw_qa.get("validated", False)),
        "query_images": query_images,
    }


def _load_image_caption(session_dir: Path, image_filename: str) -> str:
    """Read optional official-baseline captions without requiring generated captions."""

    caption_file = session_dir / "caption" / f"{Path(image_filename).stem}.json"
    if not caption_file.is_file():
        return ""
    data = _load_json_object(caption_file, "image caption")
    description = data.get("description")
    if isinstance(description, Mapping):
        return str(description.get("final_text") or "").strip()
    return str(data.get("caption") or data.get("final_text") or "").strip()


def _portable_image_reference(
    image_path: Path,
    *,
    data_root: Path,
    dataset_name: str,
    revision: str,
) -> dict[str, str]:
    path = Path(os.path.abspath(image_path))
    root = Path(os.path.abspath(data_root))
    if not path.is_file():
        raise FileNotFoundError(f"Missing H2HMem image: {path}")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"H2HMem image must remain under the dataset root: {path}") from exc
    reference = {"path": relative}
    if dataset_name:
        reference.update(
            {
                "hf_repo_id": dataset_name,
                "hf_repo_type": "dataset",
                "hf_filename": relative,
            }
        )
        if revision:
            reference["hf_revision"] = revision
    return reference


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to load H2HMem {label} JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"H2HMem {label} JSON must contain one object: {path}")
    return data


def _numbered_path_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"H2HMem {field_name} must be non-empty.")
    return text


def _dataset_extra_info(example: Mapping[str, Any]) -> dict[str, Any]:
    query_images = [dict(image) for image in example.get("query_images", [])]
    return {
        "reference_answers": list(example["reference_answers"]),
        "query_images": query_images,
        "has_query_image": bool(query_images),
        "variant": example["variant"],
        "dialogue_name": example["dialogue_name"],
        "question_session": example["question_session"],
        "question_type": example["question_type"],
        "question_type_main": example["question_type_main"],
        "difficulty": example["difficulty"],
        "validated": bool(example["validated"]),
        "context_token_count": int(example["context_token_count"]),
        "conversation_session_count": int(example["conversation_session_count"]),
        "chunking_mode": example["chunking_mode"],
        "training_metric": "token_f1",
        "official_metric": "llm_judge",
        "official_metrics": ["llm_judge", "precision", "recall", "token_f1", "bleu1"],
    }


def answer_format_instruction(question_type: Any) -> str:
    if str(question_type or "").strip().casefold() == CONFLICT_DETECTION_TYPE.casefold():
        return "Answer-content constraint: the text inside <answer>...</answer> must be exactly Yes or No."
    return ""


def is_cd_answer_format_compatible(question_type: Any, answer: Any) -> bool:
    """Return whether a Conflict Detection label can satisfy the prompted Yes/No contract."""

    if str(question_type or "").strip().casefold() != CONFLICT_DETECTION_TYPE.casefold():
        return True
    normalized_answer = str(answer or "").strip().casefold()
    if normalized_answer.endswith("."):
        normalized_answer = normalized_answer[:-1].rstrip()
    return normalized_answer in {"yes", "no"}


def _keep_h2hmem_qa(
    qa: Mapping[str, Any],
    *,
    include_answer_refusal: bool,
    include_cd_answer_format_mismatches: bool,
) -> bool:
    question_type = qa.get("question_type")
    is_answer_refusal = (
        str(question_type or "").strip().casefold() == ANSWER_REFUSAL_TYPE.casefold()
    )
    if not include_answer_refusal and is_answer_refusal:
        return False
    return include_cd_answer_format_mismatches or is_cd_answer_format_compatible(
        question_type,
        qa.get("answer"),
    )


def _print_context_statistics(examples: Sequence[Mapping[str, Any]], *, variant: str) -> None:
    tokens_by_dialogue: dict[str, int] = {}
    for example in examples:
        conversation_id = str(example.get("conversation_id") or "")
        token_count = int(example.get("context_token_count", 0))
        previous = tokens_by_dialogue.setdefault(conversation_id, token_count)
        if previous != token_count:
            raise ValueError(f"Inconsistent context length for H2HMem dialogue {conversation_id!r}.")
    values = list(tokens_by_dialogue.values())
    if values:
        print(
            f"H2HMem {variant} context tokens ({QWEN3_RETRIEVER} tokenizer, formatted original memory): "
            f"mean={mean(values):.1f}, min={min(values)}, max={max(values)}, dialogues={len(values)}"
        )


def _conversation_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str((row.get("extra_info") or {}).get("conversation_id") or "")
        for row in rows
        if str((row.get("extra_info") or {}).get("conversation_id") or "")
    }


if __name__ == "__main__":
    main()
