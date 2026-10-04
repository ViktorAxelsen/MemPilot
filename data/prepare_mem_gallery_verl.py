"""Prepare Mem-Gallery query-only policy rows and dual-view multimodal memory."""

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
    as_string_list,
    build_memory_bank,
    format_image_caption_block,
    normalize_chunking_mode,
    resolve_agentic_memory_bank,
    resolve_image_reference,
    value_at,
)
from data.runtime_memory_prompts import (  # noqa: E402
    MEM_GALLERY_ANSWER_RULES,
)
from memory_chunking import (  # noqa: E402
    DEFAULT_CHUNK_TOKENS,
    AtomicDialogueChunker,
)
from memory_compression import (  # noqa: E402
    DEFAULT_LLMLINGUA2_MODEL,
    DEFAULT_MEMORY_COMPRESSION_RATE,
    LLMLingua2MemoryCompressor,
)
from retrieval import (  # noqa: E402
    EmbeddingRetriever,
    serialize_document_embeddings,
)


DEFAULT_DATASET_NAME = "Ethan-Bei/Mem-Gallery"
QUERY_IMAGE_ID = "query_image"
ANSWER_REFUSAL_TYPE = "AR"


QA_GUIDANCE = (
    "Answer the question from the retrieved multimodal conversation memory. " + MEM_GALLERY_ANSWER_RULES
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Local Mem-Gallery repository root or data directory. The HF snapshot is downloaded when omitted.",
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--retriever_device", default="cuda:0")
    parser.add_argument("--retriever_batch_size", type=int, default=DEFAULT_RETRIEVER_BATCH_SIZE)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument(
        "--chunking_mode",
        choices=CHUNKING_MODES,
        default=TOKEN_CHUNKING_MODE,
        help=(
            "token packs complete dialogue rounds up to chunk_size; "
            "round emits exactly one original dialogue round per memory chunk."
        ),
    )
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
        help="Include AR (answer-refusal) questions; they are excluded by default.",
    )
    parser.add_argument("--output_dir", default="data/mem_gallery")
    parser.add_argument("--data_source", default="mem_gallery")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--validation_ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--train_subset_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    records = load_mem_gallery_records(args.dataset_dir, args.dataset_name)
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
        chunk_size=args.chunk_size,
        retriever_device=args.retriever_device,
        retriever_batch_size=args.retriever_batch_size,
        memory_compressor=compressor,
        include_answer_refusal=args.include_answer_refusal,
        show_progress=True,
        chunking_mode=args.chunking_mode,
    )
    if not rows:
        raise ValueError("No Mem-Gallery QA rows were produced.")

    train_rows, validation_rows, test_rows = split_rows(
        rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    full_train_size = len(train_rows)
    train_rows = sample_train_subset(train_rows, size=args.train_subset_size, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train_rows, output_dir / "train.parquet")
    write_parquet(validation_rows, output_dir / "val.parquet")
    write_parquet(test_rows, output_dir / "test.parquet")

    subset_note = f" sampled from {full_train_size}" if args.train_subset_size is not None else ""
    print(
        f"Wrote {len(train_rows)} train rows{subset_note}, {len(validation_rows)} validation rows, "
        f"and {len(test_rows)} test rows to {output_dir}"
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


def load_mem_gallery_records(
    dataset_dir: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> list[dict[str, Any]]:
    """Load all conversation JSON files and retain their source path for image resolution."""
    data_dir = _resolve_data_dir(dataset_dir, dataset_name)
    dialog_paths = sorted((data_dir / "dialog").glob("*.json"))
    if not dialog_paths:
        raise FileNotFoundError(f"No Mem-Gallery conversation JSON files found in {data_dir / 'dialog'}")

    records = []
    dataset_revision = _snapshot_revision(data_dir)
    for path in tqdm(dialog_paths, desc="Loading Mem-Gallery conversations", unit="file", dynamic_ncols=True):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if not isinstance(record, dict):
            raise ValueError(f"Mem-Gallery conversation file must contain one object: {path}")
        # Preserve the snapshot path instead of following Hugging Face cache symlinks
        # into extensionless blob files; relative image paths depend on this hierarchy.
        record["_mem_gallery_dialog_path"] = os.path.abspath(path)
        record["_mem_gallery_conversation_id"] = path.stem
        record["_mem_gallery_dataset_name"] = dataset_name
        record["_mem_gallery_revision"] = dataset_revision
        records.append(record)
    return records


def _resolve_data_dir(dataset_dir: str | Path | None, dataset_name: str) -> Path:
    if dataset_dir is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install `huggingface_hub` to download Mem-Gallery.") from exc
        dataset_name = str(dataset_name).strip().strip("/")
        if not dataset_name:
            raise ValueError("dataset_name must be non-empty.")
        root = Path(
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=["data/dialog/*.json", "data/image/**"],
            )
        )
    else:
        root = Path(dataset_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Mem-Gallery dataset directory does not exist: {root}")

    candidates = (root / "data", root)
    for candidate in candidates:
        if (candidate / "dialog").is_dir() and (candidate / "image").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Mem-Gallery requires sibling dialog/ and image/ directories under {root} or {root / 'data'}."
    )


def _snapshot_revision(data_dir: Path) -> str:
    parts = data_dir.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


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
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
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
            show_progress=show_progress,
            chunking_mode=chunking_mode,
        )
    )
    if memory_compressor is not None:
        attach_lossy_memory_banks(
            examples,
            memory_compressor,
            retriever,
            dataset_label="Mem-Gallery",
            show_progress=show_progress,
        )

    rows = []
    memory_is_compressed = memory_compressor is not None
    for example in tqdm(
        examples,
        desc="Building Mem-Gallery rows",
        unit="row",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        runtime_memory_bank, compression = resolve_agentic_memory_bank(
            example, memory_is_compressed=memory_is_compressed
        )
        row_example = {
            **example,
            "memory_bank": runtime_memory_bank,
        }
        query_images = [dict(image) for image in example.get("query_images", [])]
        rows.append(
            build_agentic_memory_row(
                example=row_example,
                data_source=data_source,
                row_index=len(rows),
                ground_truth=build_free_text_ground_truth(example["reference_answers"]),
                chunk_size=chunk_size,
                compression=compression,
                dataset_extra_info={
                    "reference_answers": example["reference_answers"],
                    "query_images": query_images,
                    "has_query_image": bool(query_images),
                    "chunking_mode": example["chunking_mode"],
                },
                metadata_extra={
                    "has_query_image": bool(query_images),
                    "chunking_mode": example["chunking_mode"],
                },
                tool_create_kwargs={"query_images": query_images},
                ability="multimodal_memory",
            )
        )
    return rows


def iter_memory_examples(
    records: Iterable[Mapping[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    retriever_device: str = "cuda:0",
    retriever_batch_size: int = DEFAULT_RETRIEVER_BATCH_SIZE,
    embedding_retriever: EmbeddingRetriever | None = None,
    chunk_tokenizer: Any = None,
    include_answer_refusal: bool = False,
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
) -> Iterable[dict[str, Any]]:
    """Chunk and embed each conversation once, then attach the bank to each QA."""
    chunking_mode = normalize_chunking_mode(chunking_mode)
    records = list(records)
    retriever = embedding_retriever or EmbeddingRetriever(
        device=retriever_device,
        batch_size=retriever_batch_size,
    )
    chunker = AtomicDialogueChunker(
        chunk_tokenizer or retriever.tokenizer,
        target_tokens=chunk_size,
    )
    iterator = tqdm(
        records,
        total=len(records),
        desc="Preparing Mem-Gallery memory",
        unit="conversation",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    qa_progress = tqdm(
        total=sum(
            sum(
                include_answer_refusal or not is_answer_refusal_qa(raw_qa)
                for raw_qa in (record.get("human-annotated QAs") or [])
            )
            for record in records
            if isinstance(record, Mapping)
        ),
        desc="Preparing Mem-Gallery QAs",
        unit="qa",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for record_index, raw_record in enumerate(iterator):
        record = normalize_mem_gallery_record(raw_record, record_index=record_index)
        qas = [
            qa
            for qa in record["qas"]
            if include_answer_refusal or qa["question_type"] != ANSWER_REFUSAL_TYPE
        ]
        if not qas:
            continue
        memory_bank = build_memory_bank(
            record,
            chunker,
            chunking_mode=chunking_mode,
            dataset_label="Mem-Gallery",
        )
        if not memory_bank:
            raise ValueError(f"Mem-Gallery conversation {record['conversation_id']!r} has no dialogue memory.")
        document_embeddings = retriever.encode_documents(memory_bank)
        memory_index = serialize_document_embeddings(
            document_embeddings,
            model_id=retriever.model_id,
        )
        iterator.set_postfix(chunks=len(memory_bank), qas=len(qas), refresh=False)

        for qa in qas:
            qa_progress.update(1)
            yield {
                "record_index": record_index,
                "conversation_id": record["conversation_id"],
                "split": "",
                "question_id": qa["question_id"],
                "question_type": qa["question_type"],
                "question": qa["question"],
                "answer": qa["answer"],
                "reference_answers": [qa["answer"]],
                "query_images": qa["query_images"],
                "chunking_mode": chunking_mode,
                "memory_chunk_count": len(memory_bank),
                "memory_bank": memory_bank,
                "memory_index": memory_index,
            }
    qa_progress.close()


def is_answer_refusal_qa(raw_qa: Any) -> bool:
    return isinstance(raw_qa, Mapping) and (
        str(raw_qa.get("point") or "").strip().upper() == ANSWER_REFUSAL_TYPE
    )


def normalize_mem_gallery_record(
    record: Mapping[str, Any],
    *,
    record_index: int,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Mem-Gallery record {record_index} must be a mapping.")
    source_path_text = str(record.get("_mem_gallery_dialog_path") or "").strip()
    source_path = Path(source_path_text) if source_path_text else None
    dataset_name = str(record.get("_mem_gallery_dataset_name") or DEFAULT_DATASET_NAME).strip()
    dataset_revision = str(record.get("_mem_gallery_revision") or "").strip()
    conversation_id = str(
        record.get("_mem_gallery_conversation_id")
        or record.get("conversation_id")
        or f"mem_gallery_{record_index:04d}"
    ).strip()
    profile = record.get("character_profile") if isinstance(record.get("character_profile"), Mapping) else {}
    character_name = str(profile.get("name") or "User").strip()

    raw_sessions = record.get("multi_session_dialogues")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(f"Mem-Gallery conversation {conversation_id!r} requires multi_session_dialogues.")
    sessions = []
    for session_index, raw_session in enumerate(raw_sessions):
        if not isinstance(raw_session, Mapping):
            raise ValueError(f"Mem-Gallery session {session_index} in {conversation_id!r} must be a mapping.")
        session_id = str(raw_session.get("session_id") or f"S{session_index + 1}").strip()
        date = str(raw_session.get("date") or "").strip()
        raw_turns = raw_session.get("dialogues")
        if not isinstance(raw_turns, list):
            raise ValueError(f"Mem-Gallery session {session_id!r} requires a dialogues list.")
        turns = [
            _normalize_dialogue_turn(
                raw_turn,
                conversation_id=conversation_id,
                session_id=session_id,
                turn_index=turn_index,
                character_name=character_name,
                source_path=source_path,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
            )
            for turn_index, raw_turn in enumerate(raw_turns)
        ]
        sessions.append({"session_id": session_id, "date": date, "turns": turns})

    raw_qas = record.get("human-annotated QAs")
    if not isinstance(raw_qas, list) or not raw_qas:
        raise ValueError(f"Mem-Gallery conversation {conversation_id!r} requires human-annotated QAs.")
    qas = [
        _normalize_qa(
            raw_qa,
            conversation_id=conversation_id,
            qa_index=qa_index,
            source_path=source_path,
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
        )
        for qa_index, raw_qa in enumerate(raw_qas)
    ]
    return {
        "conversation_id": conversation_id,
        "character_name": character_name,
        "sessions": sessions,
        "qas": qas,
    }


def _normalize_dialogue_turn(
    raw_turn: Any,
    *,
    conversation_id: str,
    session_id: str,
    turn_index: int,
    character_name: str,
    source_path: Path | None,
    dataset_name: str,
    dataset_revision: str,
) -> dict[str, Any]:
    if not isinstance(raw_turn, Mapping):
        raise ValueError(f"Dialogue turn {turn_index} in {conversation_id!r} must be a mapping.")
    turn_id = str(raw_turn.get("round") or f"{session_id}:{turn_index + 1}").strip()
    user_text = str(raw_turn.get("user") or "").strip()
    assistant_text = str(raw_turn.get("assistant") or "").strip()
    if not user_text and not assistant_text:
        raise ValueError(f"Dialogue turn {turn_id!r} contains no user or assistant text.")

    image_ids = as_string_list(raw_turn.get("image_id"))
    image_paths = as_string_list(raw_turn.get("input_image"))
    image_captions = as_string_list(raw_turn.get("image_caption"))
    image_count = max(len(image_ids), len(image_paths), len(image_captions), 0)
    images = []
    for image_index in range(image_count):
        image_id = value_at(image_ids, image_index)
        image_path = value_at(image_paths, image_index)
        caption = value_at(image_captions, image_index)
        if not image_id or not image_path or not caption:
            raise ValueError(
                f"Image metadata in turn {turn_id!r} must align image_id, input_image, and image_caption."
            )
        images.append(
            {
                "id": image_id,
                "caption": caption,
                "turn_id": turn_id,
                "source": "memory",
                **resolve_image_reference(
                    image_path,
                    source_path,
                    dataset_name=dataset_name,
                    dataset_revision=dataset_revision,
                    dataset_label="Mem-Gallery",
                ),
            }
        )

    dialogue_text = (
        f"[Round {turn_id}]\n"
        f"User ({character_name}): {user_text}\n"
        f"Assistant: {assistant_text}"
    )
    image_text = format_image_caption_block(images, heading=f"Images attached to round {turn_id}")
    return {
        "turn_id": turn_id,
        "text": f"{dialogue_text}\n{image_text}" if image_text else dialogue_text,
        "preview_text": dialogue_text,
        "protected_preview_text": image_text,
        "images": images,
    }


def _normalize_qa(
    raw_qa: Any,
    *,
    conversation_id: str,
    qa_index: int,
    source_path: Path | None,
    dataset_name: str,
    dataset_revision: str,
) -> dict[str, Any]:
    if not isinstance(raw_qa, Mapping):
        raise ValueError(f"Mem-Gallery QA {qa_index} in {conversation_id!r} must be a mapping.")
    question = str(raw_qa.get("question") or "").strip()
    answer = normalize_answer_value(raw_qa.get("answer"))
    if not question:
        raise ValueError(f"Mem-Gallery QA {qa_index} in {conversation_id!r} has no question.")

    question_image_path = str(raw_qa.get("question_image") or "").strip()
    question_image_caption = str(
        raw_qa.get("image_caption") or raw_qa.get("question_image_caption") or ""
    ).strip()
    query_images = []
    if question_image_path or question_image_caption:
        if not question_image_path or not question_image_caption:
            raise ValueError(
                f"Mem-Gallery QA {qa_index} in {conversation_id!r} must pair question_image with its caption."
            )
        query_images.append(
            {
                "id": QUERY_IMAGE_ID,
                "caption": question_image_caption,
                "source": "query",
                **resolve_image_reference(
                    question_image_path,
                    source_path,
                    dataset_name=dataset_name,
                    dataset_revision=dataset_revision,
                    dataset_label="Mem-Gallery",
                ),
            }
        )
    return {
        "question_id": f"{conversation_id}_q{qa_index:04d}",
        "question_type": str(raw_qa.get("point") or "").strip().upper(),
        "question": question,
        "answer": answer,
        "query_images": query_images,
    }


def answer_format_instruction(question_type: Any) -> str:
    if str(question_type or "").strip().upper() == "CD":
        return "Answer-content constraint: the text inside <answer>...</answer> must be exactly Yes. or No."
    return ""


def normalize_answer_value(value: Any) -> str:
    if isinstance(value, Mapping):
        answer = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        answer = ", ".join(str(item).strip() for item in value if str(item).strip())
    else:
        answer = "" if value is None else str(value).strip()
    if not answer:
        raise ValueError("Mem-Gallery answers must be non-empty.")
    return answer


if __name__ == "__main__":
    main()
