"""Prepare the full MemEye benchmark as test-only MCQ and open-answer rows."""

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
    as_string_list,
    build_memory_bank,
    format_image_caption_block as format_image_block,
    normalize_chunking_mode,
    resolve_agentic_memory_bank,
    resolve_image_reference,
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


DEFAULT_DATASET_NAME = "MemEyeBench/MemEye"
MEMEYE_VARIANTS = ("mcq", "open")
EXPECTED_PAIRED_QUESTIONS = 371
EXPECTED_MCQ_ROTATIONS = 4
EXPECTED_TEST_ROWS = {
    "mcq": EXPECTED_PAIRED_QUESTIONS * EXPECTED_MCQ_ROTATIONS,
    "open": EXPECTED_PAIRED_QUESTIONS,
}
MEMEYE_TASK_STEMS = (
    "Brand_Memory_Test",
    "Card_Playlog_Test",
    "Cartoon_Entertainment_Companion",
    "Home_Renovation_Interior_Design",
    "Multi-Scene_Visual_Case_Archive_Assistant",
    "Outdoor_Navigation_Route_Memory_Assistant",
    "Personal_Health_Dashboard_Assistant",
    "Social_Chat_Memory_Test",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Local MemEye repository root or data directory. The HF snapshot is downloaded when omitted.",
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
    parser.add_argument("--output_dir", default="data/memeye")
    parser.add_argument("--data_source_prefix", default="memeye")
    args = parser.parse_args()

    compressor = None
    if not args.disable_memory_compression:
        compressor = LLMLingua2MemoryCompressor(
            model_name=args.memory_compressor_model,
            rate=args.memory_compression_rate,
            device=args.compressor_device,
            batch_size=args.compressor_batch_size,
        )
    rows = build_rows(
        record_pairs=load_memeye_record_pairs(args.dataset_dir, args.dataset_name),
        data_source_prefix=args.data_source_prefix,
        chunk_size=args.chunk_size,
        retriever_device=args.retriever_device,
        retriever_batch_size=args.retriever_batch_size,
        memory_compressor=compressor,
        show_progress=True,
        chunking_mode=args.chunking_mode,
    )
    rows_by_variant = {
        variant: [row for row in rows if row["extra_info"]["answer_variant"] == variant]
        for variant in MEMEYE_VARIANTS
    }
    validate_memeye_variant_sizes(rows_by_variant, context="preprocessing")

    output_dir = Path(args.output_dir)
    for variant, variant_rows in rows_by_variant.items():
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(variant_rows, variant_dir / "test.parquet")
        print(f"Wrote {len(variant_rows)} MemEye {variant} test rows to {variant_dir}")

    if compressor is not None:
        original_tokens = sum(row["extra_info"]["memory_compression"]["original_tokens"] for row in rows)
        compressed_tokens = sum(row["extra_info"]["memory_compression"]["compressed_tokens"] for row in rows)
        actual_rate = compressed_tokens / original_tokens if original_tokens else 0.0
        print(
            f"LLMLingua-2 memory bank: target_rate={compressor.rate:g}, "
            f"observed_rate={actual_rate:.3f}, unique_chunks={compressor.cache_size}, "
            f"model_calls={compressor.num_model_calls}"
        )


def validate_memeye_variant_sizes(
    rows_by_variant: Mapping[str, Sequence[Any]],
    *,
    context: str,
) -> None:
    actual_sizes = {
        variant: len(rows_by_variant.get(variant, ()))
        for variant in MEMEYE_VARIANTS
    }
    if actual_sizes != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Incomplete or unexpected MemEye {context} size: "
            f"expected {EXPECTED_TEST_ROWS}, got {actual_sizes}."
        )


def load_memeye_record_pairs(
    dataset_dir: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> list[dict[str, Any]]:
    """Load only the eight original task pairs, excluding derived concat files."""

    data_dir = _resolve_data_dir(dataset_dir, dataset_name)
    dialog_dir = data_dir / "dialog"
    revision = _snapshot_revision(data_dir)
    pairs = []
    for task_stem in tqdm(
        MEMEYE_TASK_STEMS,
        desc="Loading MemEye task pairs",
        unit="task",
        dynamic_ncols=True,
    ):
        mcq_path = dialog_dir / f"{task_stem}.json"
        open_path = dialog_dir / f"{task_stem}_Open.json"
        missing = [str(path) for path in (mcq_path, open_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing MemEye task file(s): {missing}")
        pairs.append(
            {
                "scenario_id": task_stem,
                "mcq": _load_json_object(mcq_path),
                "open": _load_json_object(open_path),
                "dialog_path": os.path.abspath(mcq_path),
                "dataset_name": str(dataset_name).strip(),
                "dataset_revision": revision,
            }
        )
    return pairs


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"MemEye task file must contain one JSON object: {path}")
    return value


def _resolve_data_dir(dataset_dir: str | Path | None, dataset_name: str) -> Path:
    if dataset_dir is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install `huggingface_hub` to download MemEye.") from exc
        dataset_name = str(dataset_name).strip().strip("/")
        if not dataset_name:
            raise ValueError("dataset_name must be non-empty.")
        dialog_patterns = [
            pattern
            for stem in MEMEYE_TASK_STEMS
            for pattern in (f"data/dialog/{stem}.json", f"data/dialog/{stem}_Open.json")
        ]
        root = Path(
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=[*dialog_patterns, "data/image/**"],
            )
        )
    else:
        root = Path(dataset_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"MemEye dataset directory does not exist: {root}")

    for candidate in (root / "data", root):
        if (candidate / "dialog").is_dir() and (candidate / "image").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"MemEye requires sibling dialog/ and image/ directories under {root} or {root / 'data'}."
    )


def _snapshot_revision(data_dir: Path) -> str:
    parts = data_dir.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def build_rows(
    record_pairs: Iterable[Mapping[str, Any]],
    data_source_prefix: str,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    retriever_device: str = "cuda:0",
    retriever_batch_size: int = DEFAULT_RETRIEVER_BATCH_SIZE,
    embedding_retriever: EmbeddingRetriever | None = None,
    chunk_tokenizer: Any = None,
    memory_compressor: Any = None,
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
) -> list[dict[str, Any]]:
    retriever = embedding_retriever or EmbeddingRetriever(
        device=retriever_device,
        batch_size=retriever_batch_size,
    )
    examples = list(
        iter_memory_examples(
            record_pairs=record_pairs,
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
            dataset_label="MemEye",
            show_progress=show_progress,
        )

    rows = []
    variant_indices = {variant: 0 for variant in MEMEYE_VARIANTS}
    memory_is_compressed = memory_compressor is not None
    for example in tqdm(
        examples,
        desc="Building MemEye rows",
        unit="row",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        variant = example["answer_variant"]
        runtime_memory_bank, compression = resolve_agentic_memory_bank(
            example, memory_is_compressed=memory_is_compressed
        )
        row_example = {
            **example,
            "memory_bank": runtime_memory_bank,
        }
        query_images = [dict(image) for image in example.get("query_images", [])]
        choices = dict(example.get("choices") or {})
        dataset_extra_info = {
            "answer_variant": variant,
            "reference_answers": example["reference_answers"],
            "query_images": query_images,
            "has_query_image": bool(query_images),
            "chunking_mode": example["chunking_mode"],
            "scenario_id": example["scenario_id"],
            "paired_question_id": example["paired_question_id"],
            "memeye_point": example["memeye_point"],
        }
        if variant == "mcq":
            dataset_extra_info.update(
                {
                    "choices": choices,
                    "rotation_index": example["rotation_index"],
                }
            )
        rows.append(
            build_agentic_memory_row(
                example=row_example,
                data_source=f"{data_source_prefix}_{variant}",
                row_index=variant_indices[variant],
                ground_truth=build_memeye_ground_truth(example),
                chunk_size=chunk_size,
                compression=compression,
                dataset_extra_info=dataset_extra_info,
                metadata_extra={
                    "answer_variant": variant,
                    "has_query_image": bool(query_images),
                    "chunking_mode": example["chunking_mode"],
                    "scenario_id": example["scenario_id"],
                },
                tool_create_kwargs={"query_images": query_images},
                ability="multimodal_memory",
            )
        )
        variant_indices[variant] += 1
    return rows


def iter_memory_examples(
    record_pairs: Iterable[Mapping[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    retriever_device: str = "cuda:0",
    retriever_batch_size: int = DEFAULT_RETRIEVER_BATCH_SIZE,
    embedding_retriever: EmbeddingRetriever | None = None,
    chunk_tokenizer: Any = None,
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
) -> Iterable[dict[str, Any]]:
    """Embed each scenario memory once and share it across both answer variants."""

    chunking_mode = normalize_chunking_mode(chunking_mode)
    records = [
        normalize_memeye_record_pair(pair, record_index=index)
        for index, pair in enumerate(record_pairs)
    ]
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
        desc="Preparing MemEye memory",
        unit="scenario",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    qa_progress = tqdm(
        total=sum(len(record["qas"]) for record in records),
        desc="Preparing MemEye QAs",
        unit="qa",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for record_index, record in enumerate(iterator):
            memory_bank = build_memory_bank(
                record,
                chunker,
                chunking_mode=chunking_mode,
                dataset_label="MemEye",
            )
            if not memory_bank:
                raise ValueError(f"MemEye scenario {record['conversation_id']!r} has no dialogue memory.")
            document_embeddings = retriever.encode_documents(memory_bank)
            memory_index = serialize_document_embeddings(
                document_embeddings,
                model_id=retriever.model_id,
            )
            iterator.set_postfix(chunks=len(memory_bank), qas=len(record["qas"]), refresh=False)

            for qa in record["qas"]:
                qa_progress.update(1)
                yield {
                    "record_index": record_index,
                    "conversation_id": record["conversation_id"],
                    "scenario_id": record["conversation_id"],
                    "split": "test",
                    "question_id": qa["question_id"],
                    "paired_question_id": qa["paired_question_id"],
                    "question_type": qa["answer_variant"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "reference_answers": [qa["answer"]],
                    "answer_variant": qa["answer_variant"],
                    "choices": qa["choices"],
                    "rotation_index": qa.get("rotation_index"),
                    "memeye_point": qa["memeye_point"],
                    "query_images": qa["query_images"],
                    "chunking_mode": chunking_mode,
                    "memory_chunk_count": len(memory_bank),
                    "memory_bank": memory_bank,
                    "memory_index": memory_index,
                }
    finally:
        qa_progress.close()


def normalize_memeye_record_pair(
    pair: Mapping[str, Any],
    *,
    record_index: int,
) -> dict[str, Any]:
    if not isinstance(pair, Mapping):
        raise ValueError(f"MemEye record pair {record_index} must be a mapping.")
    mcq_record = pair.get("mcq")
    open_record = pair.get("open")
    if not isinstance(mcq_record, Mapping) or not isinstance(open_record, Mapping):
        raise ValueError(f"MemEye record pair {record_index} requires mcq and open objects.")

    scenario_id = str(pair.get("scenario_id") or f"memeye_{record_index:02d}").strip()
    _validate_paired_memory_structure(mcq_record, open_record, scenario_id=scenario_id)
    source_path_text = str(pair.get("dialog_path") or "").strip()
    source_path = Path(source_path_text) if source_path_text else None
    dataset_name = str(pair.get("dataset_name") or DEFAULT_DATASET_NAME).strip()
    dataset_revision = str(pair.get("dataset_revision") or "").strip()
    profile = mcq_record.get("character_profile")
    profile = profile if isinstance(profile, Mapping) else {}
    character_name = str(profile.get("name") or "User").strip()

    raw_sessions = mcq_record.get("multi_session_dialogues")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(f"MemEye scenario {scenario_id!r} requires multi_session_dialogues.")
    sessions = []
    for session_index, raw_session in enumerate(raw_sessions):
        if not isinstance(raw_session, Mapping):
            raise ValueError(f"MemEye session {session_index} in {scenario_id!r} must be a mapping.")
        session_id = str(raw_session.get("session_id") or f"S{session_index + 1}").strip()
        raw_turns = raw_session.get("dialogues")
        if not isinstance(raw_turns, list):
            raise ValueError(f"MemEye session {session_id!r} requires a dialogues list.")
        sessions.append(
            {
                "session_id": session_id,
                "date": str(raw_session.get("date") or "").strip(),
                "turns": [
                    _normalize_dialogue_turn(
                        raw_turn,
                        scenario_id=scenario_id,
                        session_id=session_id,
                        turn_index=turn_index,
                        character_name=character_name,
                        source_path=source_path,
                        dataset_name=dataset_name,
                        dataset_revision=dataset_revision,
                    )
                    for turn_index, raw_turn in enumerate(raw_turns)
                ],
            }
        )

    mcq_qas = _qa_by_id(mcq_record, scenario_id=scenario_id, variant="mcq")
    open_qas = _qa_by_id(open_record, scenario_id=scenario_id, variant="open")
    if set(mcq_qas) != set(open_qas):
        raise ValueError(
            f"MemEye scenario {scenario_id!r} has unpaired MCQ/open question IDs: "
            f"mcq_only={sorted(set(mcq_qas) - set(open_qas))}, "
            f"open_only={sorted(set(open_qas) - set(mcq_qas))}."
        )

    qas = []
    for question_id, raw_mcq in mcq_qas.items():
        raw_open = open_qas[question_id]
        qas.extend(
            _normalize_mcq_rotations(
                raw_mcq,
                scenario_id=scenario_id,
                question_id=question_id,
                source_path=source_path,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
            )
        )
        qas.append(
            _normalize_open_qa(
                raw_open,
                scenario_id=scenario_id,
                question_id=question_id,
                source_path=source_path,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
            )
        )
    return {
        "conversation_id": scenario_id,
        "character_name": character_name,
        "sessions": sessions,
        "qas": qas,
    }


def _validate_paired_memory_structure(
    mcq_record: Mapping[str, Any],
    open_record: Mapping[str, Any],
    *,
    scenario_id: str,
) -> None:
    """Allow harmless wording edits while requiring the paired memory layout to match."""

    def signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
        raw_sessions = record.get("multi_session_dialogues")
        if not isinstance(raw_sessions, list):
            return ("invalid_sessions",)
        sessions = []
        for session_index, session in enumerate(raw_sessions):
            if not isinstance(session, Mapping):
                return ("invalid_session", session_index)
            raw_turns = session.get("dialogues")
            if not isinstance(raw_turns, list):
                return ("invalid_turns", session_index)
            turns = []
            for turn_index, turn in enumerate(raw_turns):
                if not isinstance(turn, Mapping):
                    return ("invalid_turn", session_index, turn_index)
                turns.append(
                    (
                        str(turn.get("round") or "").strip(),
                        tuple(as_string_list(turn.get("input_image"))),
                        tuple(as_string_list(turn.get("image_id"))),
                        tuple(as_string_list(turn.get("image_caption"))),
                    )
                )
            sessions.append(
                (
                    str(session.get("session_id") or "").strip(),
                    str(session.get("date") or "").strip(),
                    tuple(turns),
                )
            )
        return tuple(sessions)

    if signature(mcq_record) != signature(open_record):
        raise ValueError(
            f"MemEye MCQ/open variants for {scenario_id!r} must share the same "
            "session, turn, and image structure."
        )


def _qa_by_id(
    record: Mapping[str, Any],
    *,
    scenario_id: str,
    variant: str,
) -> dict[str, Mapping[str, Any]]:
    raw_qas = record.get("human-annotated QAs")
    if not isinstance(raw_qas, list) or not raw_qas:
        raise ValueError(f"MemEye {variant} scenario {scenario_id!r} requires human-annotated QAs.")
    result = {}
    for index, qa in enumerate(raw_qas):
        if not isinstance(qa, Mapping):
            raise ValueError(f"MemEye {variant} QA {index} in {scenario_id!r} must be a mapping.")
        question_id = str(qa.get("question_id") or f"Q{index + 1}").strip()
        if question_id in result:
            raise ValueError(f"Duplicate MemEye {variant} question_id {question_id!r} in {scenario_id!r}.")
        result[question_id] = qa
    return result


def _normalize_mcq_rotations(
    raw_qa: Mapping[str, Any],
    *,
    scenario_id: str,
    question_id: str,
    source_path: Path | None,
    dataset_name: str,
    dataset_revision: str,
) -> list[dict[str, Any]]:
    question = _required_text(raw_qa.get("question"), f"MemEye MCQ {scenario_id}/{question_id} question")
    rotations = raw_qa.get("options")
    if not isinstance(rotations, list) or len(rotations) != EXPECTED_MCQ_ROTATIONS:
        raise ValueError(
            f"MemEye MCQ {scenario_id}/{question_id} requires exactly "
            f"{EXPECTED_MCQ_ROTATIONS} option rotations."
        )
    query_images = _normalize_qa_images(
        raw_qa,
        scenario_id=scenario_id,
        question_id=question_id,
        source_path=source_path,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
    )
    point = raw_qa.get("point")
    normalized = []
    for rotation_index, raw_rotation in enumerate(rotations):
        if not isinstance(raw_rotation, Mapping):
            raise ValueError(
                f"MemEye MCQ rotation {rotation_index} in {scenario_id}/{question_id} must be a mapping."
            )
        answer = str(raw_rotation.get("answer") or "").strip().upper()
        choices = {
            str(key).strip().upper(): str(value).strip()
            for key, value in raw_rotation.items()
            if str(key).strip().lower() != "answer" and str(value).strip()
        }
        choices = dict(sorted(choices.items()))
        if set(choices) != set("ABCD") or answer not in choices:
            raise ValueError(
                f"MemEye MCQ rotation {rotation_index} in {scenario_id}/{question_id} has invalid choices/answer."
            )
        normalized.append(
            {
                "question_id": f"{scenario_id}:{question_id}:mcq:r{rotation_index}",
                "paired_question_id": f"{scenario_id}:{question_id}",
                "answer_variant": "mcq",
                "question": question,
                "answer": answer,
                "choices": choices,
                "rotation_index": rotation_index,
                "memeye_point": point,
                "query_images": query_images,
            }
        )
    correct_option_texts = {
        qa["choices"][qa["answer"]].strip().casefold()
        for qa in normalized
    }
    if len(correct_option_texts) != 1:
        raise ValueError(
            f"MemEye MCQ {scenario_id}/{question_id} rotations disagree on the correct option text."
        )
    return normalized


def _normalize_open_qa(
    raw_qa: Mapping[str, Any],
    *,
    scenario_id: str,
    question_id: str,
    source_path: Path | None,
    dataset_name: str,
    dataset_revision: str,
) -> dict[str, Any]:
    return {
        "question_id": f"{scenario_id}:{question_id}:open",
        "paired_question_id": f"{scenario_id}:{question_id}",
        "answer_variant": "open",
        "question": _required_text(
            raw_qa.get("question"),
            f"MemEye open {scenario_id}/{question_id} question",
        ),
        "answer": _required_text(
            raw_qa.get("answer"),
            f"MemEye open {scenario_id}/{question_id} answer",
        ),
        "choices": {},
        "rotation_index": None,
        "memeye_point": raw_qa.get("point"),
        "query_images": _normalize_qa_images(
            raw_qa,
            scenario_id=scenario_id,
            question_id=question_id,
            source_path=source_path,
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
        ),
    }


def _normalize_dialogue_turn(
    raw_turn: Any,
    *,
    scenario_id: str,
    session_id: str,
    turn_index: int,
    character_name: str,
    source_path: Path | None,
    dataset_name: str,
    dataset_revision: str,
) -> dict[str, Any]:
    if not isinstance(raw_turn, Mapping):
        raise ValueError(f"Dialogue turn {turn_index} in MemEye {scenario_id!r} must be a mapping.")
    turn_id = str(raw_turn.get("round") or f"{session_id}:{turn_index + 1}").strip()
    user_text = str(raw_turn.get("user") or "").strip()
    assistant_text = str(raw_turn.get("assistant") or "").strip()
    if not user_text and not assistant_text:
        raise ValueError(f"MemEye dialogue turn {turn_id!r} contains no user or assistant text.")

    image_paths = as_string_list(raw_turn.get("input_image"))
    image_ids = _aligned_optional_values(raw_turn.get("image_id"), len(image_paths), "image_id", turn_id)
    captions = _aligned_optional_values(
        raw_turn.get("image_caption"),
        len(image_paths),
        "image_caption",
        turn_id,
    )
    images = []
    for image_index, image_path in enumerate(image_paths):
        image_id = image_ids[image_index] or f"{turn_id}:IMG_{image_index + 1:03d}"
        images.append(
            {
                "id": image_id,
                "caption": captions[image_index],
                "turn_id": turn_id,
                "source": "memory",
                **resolve_image_reference(
                    image_path,
                    source_path,
                    dataset_name=dataset_name,
                    dataset_revision=dataset_revision,
                    dataset_label="MemEye",
                ),
            }
        )

    dialogue_text = (
        f"[Round {turn_id}]\n"
        f"User ({character_name}): {user_text}\n"
        f"Assistant: {assistant_text}"
    )
    image_text = format_image_block(images, heading=f"Images attached to round {turn_id}")
    return {
        "turn_id": turn_id,
        "text": f"{dialogue_text}\n{image_text}" if image_text else dialogue_text,
        "preview_text": dialogue_text,
        "protected_preview_text": image_text,
        "images": images,
    }


def _normalize_qa_images(
    raw_qa: Mapping[str, Any],
    *,
    scenario_id: str,
    question_id: str,
    source_path: Path | None,
    dataset_name: str,
    dataset_revision: str,
) -> list[dict[str, Any]]:
    paths = [
        *as_string_list(raw_qa.get("question_image")),
        *as_string_list(raw_qa.get("question_images")),
    ]
    if not paths:
        return []
    captions_value = raw_qa.get("question_image_caption")
    if captions_value is None:
        captions_value = raw_qa.get("question_image_captions")
    if captions_value is None:
        captions_value = raw_qa.get("image_caption")
    captions = _aligned_optional_values(
        captions_value,
        len(paths),
        "question image caption",
        f"{scenario_id}/{question_id}",
    )
    ids_value = raw_qa.get("question_image_id")
    if ids_value is None:
        ids_value = raw_qa.get("question_image_ids")
    image_ids = _aligned_optional_values(
        ids_value,
        len(paths),
        "question image ID",
        f"{scenario_id}/{question_id}",
    )
    return [
        {
            "id": image_ids[index] or f"query_image_{index + 1:03d}",
            "caption": captions[index],
            "source": "query",
            **resolve_image_reference(
                path,
                source_path,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
                dataset_label="MemEye",
            ),
        }
        for index, path in enumerate(paths)
    ]


def build_memeye_ground_truth(example: Mapping[str, Any]) -> str:
    variant = str(example.get("answer_variant") or "").strip().lower()
    answer = _required_text(example.get("answer"), "MemEye ground-truth answer")
    if variant == "mcq":
        valid_choices = list((example.get("choices") or {}).keys())
        payload = {"variant": "mcq", "answer": answer, "valid_choices": valid_choices}
    elif variant == "open":
        payload = {"variant": "open", "answers": [answer]}
    else:
        raise ValueError(f"Unsupported MemEye answer variant: {variant!r}.")
    return json.dumps(payload, ensure_ascii=False)


def _aligned_optional_values(
    value: Any,
    expected_count: int,
    field_name: str,
    owner: str,
) -> list[str]:
    values = as_string_list(value)
    if values and len(values) != expected_count:
        raise ValueError(
            f"MemEye {field_name} count must match image count in {owner!r}: "
            f"{len(values)} != {expected_count}."
        )
    return values or [""] * expected_count


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


if __name__ == "__main__":
    main()
