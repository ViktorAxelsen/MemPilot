"""Prepare WorldMemArena with checkpoint-time cumulative conversation memory."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
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


DEFAULT_DATASET_NAME = "LCZZZZ/WorldMemArena"
WORLDMEMARENA_REGIMES = ("lifelong", "agentic")
SOURCE_REGIME_DIR = {"lifelong": "lifelong", "agentic": "agent"}
EXPECTED_SAMPLE_COUNTS = {"lifelong": 38, "agentic": 423}
EXPECTED_QA_COUNTS = {"lifelong": 2_090, "agentic": 22_168}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime",
        choices=WORLDMEMARENA_REGIMES,
        default="lifelong",
        help="Treat WorldMemArena Lifelong Evolution or Agentic Execution as an independent dataset.",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Local WorldMemArena repository or selected-regime directory; download from HF when omitted.",
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
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_source", default=None)
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--validation_ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--train_subset_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    regime = normalize_worldmemarena_regime(args.regime)
    records = load_worldmemarena_records(args.dataset_dir, args.dataset_name, regime)
    validate_worldmemarena_records(
        records,
        regime,
        expected_sample_count=EXPECTED_SAMPLE_COUNTS[regime],
        expected_qa_count=EXPECTED_QA_COUNTS[regime],
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
        data_source=args.data_source or f"worldmemarena_{regime}",
        chunk_size=args.chunk_size,
        retriever_device=args.retriever_device,
        retriever_batch_size=args.retriever_batch_size,
        memory_compressor=compressor,
        show_progress=True,
        chunking_mode=args.chunking_mode,
        regime=regime,
    )
    expected_qas = EXPECTED_QA_COUNTS[regime]
    if len(rows) != expected_qas:
        raise ValueError(
            f"Incomplete WorldMemArena {regime} data: expected {expected_qas} QA rows, got {len(rows)}."
        )

    train_rows, validation_rows, test_rows = split_worldmemarena_rows(
        rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    full_train_size = len(train_rows)
    train_rows = sample_train_subset(train_rows, size=args.train_subset_size, seed=args.seed)
    output_dir = Path(args.output_dir or f"data/worldmemarena_{regime}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train_rows, output_dir / "train.parquet")
    write_parquet(validation_rows, output_dir / "val.parquet")
    write_parquet(test_rows, output_dir / "test.parquet")

    train_worlds = _unique_conversation_count(train_rows)
    validation_worlds = _unique_conversation_count(validation_rows)
    test_worlds = _unique_conversation_count(test_rows)
    subset_note = f" sampled from {full_train_size}" if args.train_subset_size is not None else ""
    print(
        f"Wrote {len(train_rows)} train rows{subset_note} from {train_worlds} worlds, "
        f"{len(validation_rows)} validation rows from {validation_worlds} worlds, and "
        f"{len(test_rows)} test rows from {test_worlds} worlds to {output_dir}"
    )
    _print_context_statistics(rows, regime)
    if compressor is not None:
        print(
            f"LLMLingua-2 memory bank: target_rate={compressor.rate:g}, "
            f"unique_chunks={compressor.cache_size}, model_calls={compressor.num_model_calls}"
        )


def normalize_worldmemarena_regime(value: Any) -> str:
    regime = str(value or "").strip().lower()
    if regime not in WORLDMEMARENA_REGIMES:
        raise ValueError(f"regime must be one of {WORLDMEMARENA_REGIMES}; got {value!r}.")
    return regime


def load_worldmemarena_records(
    dataset_dir: str | Path | None,
    dataset_name: str,
    regime: str,
) -> list[dict[str, Any]]:
    """Load one official regime and retain portable image-resolution metadata."""

    regime = normalize_worldmemarena_regime(regime)
    data_root, regime_dir = _resolve_worldmemarena_dir(dataset_dir, dataset_name, regime)
    revision = _snapshot_revision(data_root)
    records = []
    paths = sorted(regime_dir.rglob("*.json"))
    for path in tqdm(paths, desc=f"Loading WorldMemArena {regime}", unit="file", dynamic_ncols=True):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if not isinstance(record, Mapping) or not record.get("sample_id"):
            continue
        relative_parent = path.parent.relative_to(regime_dir)
        parts = relative_parent.parts
        record = dict(record)
        record.update(
            {
                "_worldmemarena_source_path": os.path.abspath(path),
                "_worldmemarena_data_root": os.path.abspath(data_root),
                "_worldmemarena_dataset_name": str(dataset_name).strip(),
                "_worldmemarena_revision": revision,
                "_worldmemarena_regime": regime,
                "_worldmemarena_category": parts[0] if parts else "",
                "_worldmemarena_scenario": parts[1] if len(parts) > 1 else (parts[0] if parts else ""),
            }
        )
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No WorldMemArena {regime} sample JSON files found under {regime_dir}.")
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"WorldMemArena {regime} contains duplicate sample_id values.")
    return records


def _resolve_worldmemarena_dir(
    dataset_dir: str | Path | None,
    dataset_name: str,
    regime: str,
) -> tuple[Path, Path]:
    source_dir = SOURCE_REGIME_DIR[regime]
    if dataset_dir is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install `huggingface_hub` to download WorldMemArena.") from exc
        dataset_name = str(dataset_name).strip().strip("/")
        if not dataset_name:
            raise ValueError("dataset_name must be non-empty.")
        root = Path(
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=[f"{source_dir}/**/*.json", f"{source_dir}/**/images/**"],
            )
        ).resolve()
    else:
        root = Path(dataset_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"WorldMemArena dataset directory does not exist: {root}")

    nested = root / source_dir
    if nested.is_dir():
        return root, nested
    if root.name == source_dir and any(root.rglob("*.json")):
        return root.parent, root
    raise FileNotFoundError(
        f"Could not find WorldMemArena source directory {source_dir!r} under {root}."
    )


def _snapshot_revision(data_root: Path) -> str:
    parts = data_root.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def validate_worldmemarena_records(
    records: Sequence[Mapping[str, Any]],
    regime: str,
    *,
    expected_sample_count: int | None = None,
    expected_qa_count: int | None = None,
) -> int:
    """Validate all raw records before loading retriever or compression models."""

    regime = normalize_worldmemarena_regime(regime)
    if expected_sample_count is not None and len(records) != expected_sample_count:
        raise ValueError(
            f"Incomplete WorldMemArena {regime} data: expected {expected_sample_count} worlds, "
            f"got {len(records)}."
        )
    normalized = [
        normalize_worldmemarena_record(record, record_index=index, regime=regime)
        for index, record in enumerate(records)
    ]
    conversation_ids = [record["conversation_id"] for record in normalized]
    if len(conversation_ids) != len(set(conversation_ids)):
        raise ValueError(f"WorldMemArena {regime} contains duplicate sample_id values.")
    qa_count = sum(len(record["qas"]) for record in normalized)
    if expected_qa_count is not None and qa_count != expected_qa_count:
        raise ValueError(
            f"Incomplete WorldMemArena {regime} data: expected {expected_qa_count} QA rows, "
            f"got {qa_count}."
        )
    return qa_count


def split_worldmemarena_rows(
    rows: list[dict[str, Any]],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    seed: int = 13,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratify worlds into disjoint train/validation/test splits (70/10/20 by default)."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1.")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between 0 and 1.")
    development_ratio = round(train_ratio + validation_ratio, 12)
    if development_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be strictly less than 1.")
    groups: dict[str, list[dict[str, Any]]] = {}
    group_strata: dict[str, tuple[str, str]] = {}
    for row in rows:
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, Mapping):
            raise ValueError("WorldMemArena rows require mapping-valued extra_info.")
        conversation_id = str(extra_info.get("conversation_id") or "").strip()
        if not conversation_id:
            raise ValueError("WorldMemArena rows require a non-empty conversation_id.")
        stratum = (
            str(extra_info.get("category") or "").strip(),
            str(extra_info.get("scenario") or "").strip(),
        )
        if not all(stratum):
            raise ValueError(
                f"WorldMemArena conversation {conversation_id!r} requires category and scenario metadata."
            )
        previous = group_strata.setdefault(conversation_id, stratum)
        if previous != stratum:
            raise ValueError(
                f"WorldMemArena conversation {conversation_id!r} spans multiple split strata."
            )
        groups.setdefault(conversation_id, []).append(row)

    if len(groups) < 3:
        raise ValueError("A leakage-free WorldMemArena split requires at least three conversations.")
    development_group_count = round(len(groups) * development_ratio)
    development_group_count = min(max(development_group_count, 2), len(groups) - 1)
    train_group_count = round(len(groups) * train_ratio)
    train_group_count = min(max(train_group_count, 1), development_group_count - 1)
    validation_group_count = development_group_count - train_group_count
    test_group_count = len(groups) - development_group_count

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for conversation_id, stratum in group_strata.items():
        strata[stratum].append(conversation_id)
    rng = random.Random(seed)
    for stratum in sorted(strata):
        strata[stratum].sort()
        rng.shuffle(strata[stratum])

    test_counts = _allocate_stratified_holdout_counts(
        {stratum: len(group_ids) for stratum, group_ids in strata.items()},
        holdout_group_count=test_group_count,
        holdout_ratio=1.0 - development_ratio,
        rng=rng,
    )
    test_ids = {
        conversation_id
        for stratum, group_ids in strata.items()
        for conversation_id in group_ids[: test_counts[stratum]]
    }
    development_strata = {
        stratum: group_ids[test_counts[stratum] :]
        for stratum, group_ids in strata.items()
    }
    validation_counts = _allocate_stratified_holdout_counts(
        {stratum: len(group_ids) for stratum, group_ids in development_strata.items()},
        holdout_group_count=validation_group_count,
        holdout_ratio=validation_group_count / development_group_count,
        rng=rng,
    )
    validation_ids = {
        conversation_id
        for stratum, group_ids in development_strata.items()
        for conversation_id in group_ids[: validation_counts[stratum]]
    }
    train = [
        row
        for row in rows
        if str(row["extra_info"]["conversation_id"]) not in test_ids | validation_ids
    ]
    validation = [
        row
        for row in rows
        if str(row["extra_info"]["conversation_id"]) in validation_ids
    ]
    test = [row for row in rows if str(row["extra_info"]["conversation_id"]) in test_ids]
    _set_worldmemarena_split(train, "train")
    _set_worldmemarena_split(validation, "val")
    _set_worldmemarena_split(test, "test")
    return train, validation, test


def _allocate_stratified_holdout_counts(
    stratum_sizes: Mapping[tuple[str, str], int],
    *,
    holdout_group_count: int,
    holdout_ratio: float,
    rng: random.Random,
) -> dict[tuple[str, str], int]:
    """Allocate an exact holdout size, with per-scenario coverage when feasible."""

    counts = {stratum: 0 for stratum in stratum_sizes}
    splittable = [stratum for stratum, size in stratum_sizes.items() if size >= 2]
    preserve_both_sides = (
        holdout_group_count >= len(splittable)
        and sum(stratum_sizes.values()) - holdout_group_count >= len(splittable)
    )
    capacities = {
        stratum: size - 1 if preserve_both_sides and size >= 2 else size
        for stratum, size in stratum_sizes.items()
    }
    if preserve_both_sides:
        for stratum in splittable:
            counts[stratum] = 1

    tie_breakers = {stratum: rng.random() for stratum in stratum_sizes}
    while sum(counts.values()) < holdout_group_count:
        candidates = [
            stratum
            for stratum in stratum_sizes
            if counts[stratum] < capacities[stratum]
        ]
        if not candidates:
            raise ValueError("Unable to allocate the requested WorldMemArena holdout split.")
        selected = max(
            candidates,
            key=lambda stratum: (
                stratum_sizes[stratum] * holdout_ratio - counts[stratum],
                tie_breakers[stratum],
            ),
        )
        counts[selected] += 1
    return counts


def _set_worldmemarena_split(rows: Sequence[dict[str, Any]], split: str) -> None:
    for row in rows:
        extra_info = row["extra_info"]
        extra_info["split"] = split
        runtime_memory = (extra_info.get("tools_kwargs") or {}).get("runtime_memory") or {}
        metadata = (runtime_memory.get("create_kwargs") or {}).get("metadata")
        if isinstance(metadata, dict):
            metadata["split"] = split


def build_rows(
    records: Iterable[Mapping[str, Any]],
    data_source: str,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    retriever_device: str = "cuda:0",
    retriever_batch_size: int = DEFAULT_RETRIEVER_BATCH_SIZE,
    embedding_retriever: EmbeddingRetriever | None = None,
    chunk_tokenizer: Any = None,
    memory_compressor: Any = None,
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
    regime: str = "lifelong",
) -> list[dict[str, Any]]:
    regime = normalize_worldmemarena_regime(regime)
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
            show_progress=show_progress,
            chunking_mode=chunking_mode,
            regime=regime,
        )
    )
    if memory_compressor is not None:
        attach_lossy_memory_banks(
            examples,
            memory_compressor,
            retriever,
            dataset_label=f"WorldMemArena {regime}",
            show_progress=show_progress,
        )

    memory_is_compressed = memory_compressor is not None
    rows = []
    for example in tqdm(
        examples,
        desc=f"Building WorldMemArena {regime} rows",
        unit="row",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        runtime_memory_bank, compression = resolve_agentic_memory_bank(
            example, memory_is_compressed=memory_is_compressed
        )
        row_example = {**example, "memory_bank": runtime_memory_bank}
        dataset_extra_info = _dataset_extra_info(example)
        rows.append(
            build_agentic_memory_row(
                example=row_example,
                data_source=data_source,
                row_index=len(rows),
                ground_truth=build_free_text_ground_truth(example["reference_answers"]),
                chunk_size=chunk_size,
                compression=compression,
                dataset_extra_info=dataset_extra_info,
                metadata_extra={
                    key: dataset_extra_info[key]
                    for key in (
                        "regime",
                        "category",
                        "scenario",
                        "checkpoint_id",
                        "question_type_name",
                        "difficulty",
                        "visible_session_count",
                        "visible_context_token_count",
                        "full_context_token_count",
                        "chunking_mode",
                    )
                },
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
    show_progress: bool = False,
    chunking_mode: str = TOKEN_CHUNKING_MODE,
    regime: str = "lifelong",
) -> Iterable[dict[str, Any]]:
    """Encode each world once and retain only the history visible at each checkpoint."""

    regime = normalize_worldmemarena_regime(regime)
    chunking_mode = normalize_chunking_mode(chunking_mode)
    records = list(records)
    retriever = embedding_retriever or EmbeddingRetriever(
        device=retriever_device,
        batch_size=retriever_batch_size,
    )
    tokenizer = chunk_tokenizer or retriever.tokenizer
    chunker = AtomicDialogueChunker(tokenizer, target_tokens=chunk_size)
    iterator = tqdm(
        records,
        desc=f"Preparing WorldMemArena {regime} memory",
        unit="world",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for record_index, raw_record in enumerate(iterator):
        record = normalize_worldmemarena_record(raw_record, record_index=record_index, regime=regime)
        full_memory_bank = build_memory_bank(
            record,
            chunker,
            chunking_mode=chunking_mode,
            dataset_label=f"WorldMemArena {regime}",
        )
        if not full_memory_bank:
            raise ValueError(f"WorldMemArena world {record['conversation_id']!r} has no memory.")
        full_embeddings = retriever.encode_documents(full_memory_bank)
        token_counts = [
            len(tokenizer.encode(item["text"], add_special_tokens=False))
            for item in full_memory_bank
        ]
        full_context_token_count = sum(token_counts)
        qas_by_scope: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for qa in record["qas"]:
            qas_by_scope[tuple(qa["visible_session_ids"])].append(qa)

        for visible_session_ids, qas in qas_by_scope.items():
            visible = set(visible_session_ids)
            scoped_indices = [
                index
                for index, item in enumerate(full_memory_bank)
                if str(item.get("session_id") or "") in visible
            ]
            scoped_bank = [full_memory_bank[index] for index in scoped_indices]
            if not scoped_bank:
                raise ValueError(
                    f"WorldMemArena checkpoint in {record['conversation_id']!r} has no visible dialogue memory."
                )
            scoped_embeddings = _select_embedding_rows(full_embeddings, scoped_indices)
            memory_index = serialize_document_embeddings(scoped_embeddings, model_id=retriever.model_id)
            visible_context_token_count = sum(token_counts[index] for index in scoped_indices)
            for qa in qas:
                yield {
                    "record_index": record_index,
                    "conversation_id": record["conversation_id"],
                    "split": "",
                    "question_id": qa["question_id"],
                    "question_type": qa["question_type"],
                    "question_type_name": qa["question_type_name"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "reference_answers": [qa["answer"]],
                    "regime": regime,
                    "category": record["category"],
                    "scenario": record["scenario"],
                    "checkpoint_id": qa["checkpoint_id"],
                    "difficulty": qa["difficulty"],
                    "visible_session_count": len(visible_session_ids),
                    "visible_context_token_count": visible_context_token_count,
                    "full_context_token_count": full_context_token_count,
                    "chunking_mode": chunking_mode,
                    "memory_chunk_count": len(scoped_bank),
                    "full_memory_chunk_count": len(full_memory_bank),
                    "memory_bank": scoped_bank,
                    "compression_memory_bank": full_memory_bank,
                    "memory_index": memory_index,
                }
        iterator.set_postfix(
            chunks=len(full_memory_bank),
            qas=len(record["qas"]),
            tokens=full_context_token_count,
            refresh=False,
        )


def _select_embedding_rows(embeddings: Any, indices: Sequence[int]) -> Any:
    import numpy as np

    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError("WorldMemArena document embeddings must be a two-dimensional matrix.")
    return matrix[list(indices)]


def normalize_worldmemarena_record(
    record: Mapping[str, Any],
    *,
    record_index: int,
    regime: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"WorldMemArena record {record_index} must be a mapping.")
    regime = normalize_worldmemarena_regime(regime)
    record_regime = normalize_worldmemarena_regime(record.get("_worldmemarena_regime") or regime)
    if record_regime != regime:
        raise ValueError(f"WorldMemArena record regime {record_regime!r} does not match {regime!r}.")
    conversation_id = _required_text(record.get("sample_id"), "sample_id")
    source_text = str(record.get("_worldmemarena_source_path") or "").strip()
    root_text = str(record.get("_worldmemarena_data_root") or "").strip()
    source_path = Path(source_text) if source_text else None
    data_root = Path(root_text) if root_text else None
    dataset_name = str(record.get("_worldmemarena_dataset_name") or DEFAULT_DATASET_NAME).strip()
    revision = str(record.get("_worldmemarena_revision") or "").strip()

    raw_sessions = record.get("sessions")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(f"WorldMemArena world {conversation_id!r} requires sessions.")
    sessions = []
    session_ids = []
    for session_index, raw_session in enumerate(raw_sessions):
        if not isinstance(raw_session, Mapping):
            raise ValueError(f"WorldMemArena session {session_index} in {conversation_id!r} must be a mapping.")
        session_id = _required_text(
            raw_session.get("_v2_session_id"),
            f"{conversation_id} session {session_index} ID",
        )
        raw_turns = raw_session.get("dialogue")
        if not isinstance(raw_turns, list):
            raise ValueError(f"WorldMemArena session {session_id!r} requires a dialogue list.")
        turns = []
        for turn_index, raw_turn in enumerate(raw_turns):
            normalized_turn = _normalize_worldmemarena_turn(
                raw_turn,
                conversation_id=conversation_id,
                session_id=session_id,
                turn_index=turn_index,
                source_path=source_path,
                data_root=data_root,
                dataset_name=dataset_name,
                revision=revision,
            )
            if normalized_turn is not None:
                turns.append(normalized_turn)
        date = next(
            (
                str(turn.get("timestamp") or "").strip()
                for turn in raw_turns
                if isinstance(turn, Mapping) and turn.get("timestamp")
            ),
            "",
        )
        sessions.append({"session_id": session_id, "date": date, "turns": turns})
        session_ids.append(session_id)
    if len(session_ids) != len(set(session_ids)):
        raise ValueError(f"WorldMemArena world {conversation_id!r} has duplicate session IDs.")

    raw_checkpoints = record.get("qa_checkpoints")
    if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
        raise ValueError(f"WorldMemArena world {conversation_id!r} requires QA checkpoints.")
    qas = []
    checkpoint_ids = set()
    known_sessions = set(session_ids)
    for checkpoint_index, checkpoint in enumerate(raw_checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"WorldMemArena checkpoint {checkpoint_index} must be a mapping.")
        checkpoint_id = _required_text(
            checkpoint.get("checkpoint_id"),
            f"{conversation_id} checkpoint {checkpoint_index} ID",
        )
        if checkpoint_id in checkpoint_ids:
            raise ValueError(f"WorldMemArena world {conversation_id!r} has duplicate checkpoint {checkpoint_id!r}.")
        checkpoint_ids.add(checkpoint_id)
        covered_session_ids = _required_unique_string_list(
            checkpoint.get("covered_sessions"),
            f"{conversation_id}/{checkpoint_id} covered_sessions",
        )
        unknown_sessions = sorted(set(covered_session_ids) - known_sessions)
        if unknown_sessions:
            raise ValueError(
                f"WorldMemArena checkpoint {conversation_id}/{checkpoint_id} references unknown sessions: "
                f"{unknown_sessions}."
            )
        visible_session_ids = _checkpoint_visible_session_ids(
            session_ids,
            covered_session_ids,
        )
        raw_questions = checkpoint.get("questions")
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError(f"WorldMemArena checkpoint {conversation_id}/{checkpoint_id} has no questions.")
        for question_index, raw_qa in enumerate(raw_questions):
            qas.append(
                _normalize_worldmemarena_qa(
                    raw_qa,
                    conversation_id=conversation_id,
                    checkpoint_id=checkpoint_id,
                    question_index=question_index,
                    visible_session_ids=visible_session_ids,
                )
            )

    question_ids = [qa["question_id"] for qa in qas]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"WorldMemArena world {conversation_id!r} has duplicate derived question IDs.")
    return {
        "conversation_id": conversation_id,
        "regime": regime,
        "category": str(record.get("_worldmemarena_category") or "").strip(),
        "scenario": str(record.get("_worldmemarena_scenario") or "").strip(),
        "sessions": sessions,
        "qas": qas,
    }


def _normalize_worldmemarena_turn(
    raw_turn: Any,
    *,
    conversation_id: str,
    session_id: str,
    turn_index: int,
    source_path: Path | None,
    data_root: Path | None,
    dataset_name: str,
    revision: str,
) -> dict[str, Any] | None:
    if not isinstance(raw_turn, Mapping):
        raise ValueError(f"WorldMemArena turn {turn_index} in {conversation_id}/{session_id} must be a mapping.")
    raw_role = str(raw_turn.get("role") or "").strip().casefold()
    role = {"user": "User", "assistant": "Assistant", "system": "System"}.get(raw_role)
    if role is None:
        raise ValueError(f"Unsupported WorldMemArena role {raw_turn.get('role')!r}.")
    turn_id = f"{session_id}:T{turn_index + 1:03d}"
    content = str(raw_turn.get("content") or "").strip()
    timestamp = str(raw_turn.get("timestamp") or "").strip()
    raw_attachments = raw_turn.get("attachments") or []
    if not isinstance(raw_attachments, list):
        raise ValueError(f"WorldMemArena attachments in {turn_id!r} must be a list.")
    images = [
        _normalize_worldmemarena_image(
            attachment,
            turn_id=turn_id,
            attachment_index=attachment_index,
            source_path=source_path,
            data_root=data_root,
            dataset_name=dataset_name,
            revision=revision,
        )
        for attachment_index, attachment in enumerate(raw_attachments)
    ]
    if not content and not images:
        return None

    header = f"[Turn {turn_id}]"
    if timestamp:
        header += f"\nTimestamp: {timestamp}"
    dialogue_text = f"{header}\n{role}: {content}".rstrip()
    image_text = format_image_caption_block(images, heading=f"Images attached to turn {turn_id}")
    return {
        "turn_id": turn_id,
        "text": f"{dialogue_text}\n{image_text}" if image_text else dialogue_text,
        "preview_text": dialogue_text,
        "protected_preview_text": image_text,
        "images": images,
    }


def _normalize_worldmemarena_image(
    raw_attachment: Any,
    *,
    turn_id: str,
    attachment_index: int,
    source_path: Path | None,
    data_root: Path | None,
    dataset_name: str,
    revision: str,
) -> dict[str, Any]:
    if not isinstance(raw_attachment, Mapping):
        raise ValueError(f"WorldMemArena attachment {attachment_index} in {turn_id!r} must be a mapping.")
    image_id = _required_text(raw_attachment.get("image_id"), f"{turn_id} image ID")
    if "," in image_id:
        raise ValueError(f"WorldMemArena image IDs must be comma-safe: {image_id!r}.")
    caption = _required_text(raw_attachment.get("caption"), f"{turn_id}/{image_id} caption")
    raw_path = _required_text(raw_attachment.get("file_path"), f"{turn_id}/{image_id} file_path")
    return {
        "id": image_id,
        "caption": caption,
        "turn_id": turn_id,
        "source": "memory",
        **_resolve_worldmemarena_image_reference(
            raw_path,
            source_path=source_path,
            data_root=data_root,
            dataset_name=dataset_name,
            revision=revision,
        ),
    }


def _resolve_worldmemarena_image_reference(
    raw_path: str,
    *,
    source_path: Path | None,
    data_root: Path | None,
    dataset_name: str,
    revision: str,
) -> dict[str, str]:
    if source_path is None or data_root is None:
        raise ValueError("WorldMemArena portable image references require source and dataset-root paths.")
    relative = Path(str(raw_path).replace("\\", "/").removeprefix("./"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe WorldMemArena image path: {raw_path!r}.")
    # Keep the snapshot-facing path instead of resolving HF cache symlinks to
    # extensionless blob files outside the dataset hierarchy.
    path = Path(os.path.abspath(source_path.parent / relative))
    data_root = Path(os.path.abspath(data_root))
    if not path.is_file():
        raise FileNotFoundError(f"Missing WorldMemArena image: {path}")
    try:
        portable_path = path.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"WorldMemArena image must remain under the dataset root: {path}") from exc
    reference = {"path": portable_path}
    if dataset_name:
        reference.update(
            {
                "hf_repo_id": dataset_name,
                "hf_repo_type": "dataset",
                "hf_filename": portable_path,
            }
        )
        if revision:
            reference["hf_revision"] = revision
    return reference


def _normalize_worldmemarena_qa(
    raw_qa: Any,
    *,
    conversation_id: str,
    checkpoint_id: str,
    question_index: int,
    visible_session_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(raw_qa, Mapping):
        raise ValueError(f"WorldMemArena QA {conversation_id}/{checkpoint_id}/{question_index} must be a mapping.")
    question_type_name = str(raw_qa.get("question_type") or "unknown").strip() or "unknown"
    question_type = str(raw_qa.get("question_type_abbrev") or "UNKNOWN").strip().upper() or "UNKNOWN"
    return {
        "question_id": f"{conversation_id}:{checkpoint_id}:Q{question_index + 1:03d}",
        "checkpoint_id": checkpoint_id,
        "visible_session_ids": list(visible_session_ids),
        "question_type": question_type,
        "question_type_name": question_type_name,
        "difficulty": str(raw_qa.get("difficulty") or "").strip().lower(),
        "question": _required_text(raw_qa.get("question"), "question"),
        "answer": _required_text(raw_qa.get("answer"), "answer"),
        "query_images": [],
    }


def _checkpoint_visible_session_ids(
    session_ids: Sequence[str],
    covered_session_ids: Sequence[str],
) -> list[str]:
    """Return the cumulative session prefix present when an official checkpoint fires."""

    session_index = {session_id: index for index, session_id in enumerate(session_ids)}
    trigger_index = max(session_index[session_id] for session_id in covered_session_ids)
    return list(session_ids[: trigger_index + 1])


def _required_unique_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"WorldMemArena {field_name} must be a non-empty list.")
    values = [_required_text(item, field_name) for item in value]
    if len(values) != len(set(values)):
        raise ValueError(f"WorldMemArena {field_name} must not contain duplicates.")
    return values


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"WorldMemArena {field_name} must be non-empty.")
    return text


def _dataset_extra_info(example: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reference_answers": list(example["reference_answers"]),
        "regime": example["regime"],
        "category": example["category"],
        "scenario": example["scenario"],
        "checkpoint_id": example["checkpoint_id"],
        "question_type_name": example["question_type_name"],
        "difficulty": example["difficulty"],
        "visible_session_count": int(example["visible_session_count"]),
        "visible_context_token_count": int(example["visible_context_token_count"]),
        "full_context_token_count": int(example["full_context_token_count"]),
        "full_memory_chunk_count": int(example["full_memory_chunk_count"]),
        "chunking_mode": example["chunking_mode"],
        "query_images": [],
        "has_query_image": False,
        "training_metric": "token_f1",
        "official_metric": "llm_judge",
        "official_metrics": ["qa_correctness_llm_judge", "token_f1", "bleu1"],
    }


def _unique_conversation_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({str(row["extra_info"]["conversation_id"]) for row in rows})


def _print_context_statistics(rows: Sequence[Mapping[str, Any]], regime: str) -> None:
    by_world = {
        str(row["extra_info"]["conversation_id"]): int(row["extra_info"]["full_context_token_count"])
        for row in rows
    }
    visible_lengths = [int(row["extra_info"]["visible_context_token_count"]) for row in rows]
    mean_full = sum(by_world.values()) / len(by_world)
    mean_visible = sum(visible_lengths) / len(visible_lengths)
    print(
        f"WorldMemArena {regime} context tokens ({QWEN3_RETRIEVER} tokenizer): "
        f"mean full conversation={mean_full:.1f} across {len(by_world)} worlds; "
        f"mean QA-visible checkpoint={mean_visible:.1f}."
    )


if __name__ == "__main__":
    main()
