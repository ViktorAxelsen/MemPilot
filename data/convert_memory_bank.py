"""Adapt an existing memory system's output into MemPilot's compressed corpus.

Banks belong to a conversation, or to a conversation checkpoint when its history
changes over time. All questions at that scope share the same bank.

This is an offline format adapter, not a memory builder. Only the compressed
corpus and its index are replaced; raw history, images, questions, labels and
splits stay in the prepared parquet rows. Customize ``convert_memory_items``
for richer source formats instead of adding branches to runtime tools.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from shutil import copyfile
from typing import Any

from data.memory_qa import write_parquet
from retrieval import EmbeddingRetriever, QWEN3_RETRIEVER, serialize_document_embeddings


MemoryScope = tuple[str, str] | tuple[str, str, str]


def convert_memory_items(
    items: Any, *, text_field: str = "text", id_field: str = "id"
) -> list[dict[str, str]]:
    """Convert a list of strings or records into the runtime's id/text contract.

    Put source-specific flattening here, including any dates, speaker names or
    image captions needed in the observation text. Raw-image pixels and source
    chunk links are not part of RETRIEVE's text-only corpus.
    """
    if not isinstance(items, list):
        raise ValueError("Each memory bank must be a list.")
    bank = []
    seen = set()
    for index, item in enumerate(items):
        if isinstance(item, str):
            item_id, text = f"external_{index:04d}", item
        elif isinstance(item, Mapping):
            raw_id = item.get(id_field)
            item_id = f"external_{index:04d}" if raw_id is None else str(raw_id).strip()
            text = item.get(text_field)
        else:
            raise ValueError(f"Memory item {index} must be a string or a mapping.")
        if not item_id or item_id in seen or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Memory item {index} requires a unique non-empty ID and non-empty text.")
        seen.add(item_id)
        bank.append({"id": item_id, "text": text.strip()})
    return bank


def _scope(record: Mapping[str, Any]) -> MemoryScope:
    """Identify a conversation's history version independently of question_id."""
    values = tuple(str(record.get(key, "")).strip() for key in ("data_source", "conversation_id"))
    if any(not value or value == "None" for value in values):
        raise ValueError("Memory scopes require non-empty data_source and conversation_id.")
    checkpoint = record.get("checkpoint_id")
    if checkpoint is None:
        return values
    checkpoint = str(checkpoint).strip()
    if not checkpoint or checkpoint == "None":
        raise ValueError("checkpoint_id must be non-empty when provided; omit it or use null for unversioned history.")
    return (*values, checkpoint)


def _row_scope(row: Mapping[str, Any]) -> MemoryScope:
    return _scope({**row["extra_info"], "data_source": row["data_source"]})


def load_memory_banks(
    path: str | Path,
    *,
    items_field: str = "memories",
    text_field: str = "text",
    id_field: str = "id",
) -> dict[MemoryScope, list[dict[str, str]]]:
    """Read one bank per conversation or conversation checkpoint from JSON/JSONL.

    question_id is ignored. Checkpointed banks are matched exactly; they are never
    reused for another checkpoint or for a row without checkpoint metadata.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        records = ([json.loads(line) for line in handle if line.strip()]
                   if path.suffix.lower() == ".jsonl" else json.load(handle))
    if not isinstance(records, list):
        raise ValueError("The external memory file must contain a list of conversation/checkpoint records.")
    banks = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each conversation/checkpoint record must be a mapping.")
        key = _scope(record)
        if key in banks:
            raise ValueError(
                f"Duplicate external memory for conversation/checkpoint scope: {key!r}. "
                "Export one bank per scope, shared by all its questions."
            )
        banks[key] = convert_memory_items(
            record.get(items_field), text_field=text_field, id_field=id_field
        )
    return banks


def replace_compressed_memory(
    row: Mapping[str, Any],
    bank: list[dict[str, str]],
    *,
    retriever: Any,
    index_cache: dict[str, str],
) -> dict[str, Any]:
    """Attach a canonical bank and an index built from exactly its returned text."""
    prepared = copy.deepcopy(dict(row))
    extra_info = prepared["extra_info"]
    create_kwargs = extra_info["tools_kwargs"]["runtime_memory"]["create_kwargs"]
    raw_index = create_kwargs.get("memory_index")
    if isinstance(raw_index, str) and raw_index:
        raw_index = json.loads(raw_index)
    if raw_index and raw_index.get("model_id") != retriever.model_id:
        raise ValueError("Use the same retriever model as the prepared raw-memory index and tool config.")

    serialized = json.dumps(bank, ensure_ascii=False)
    key = hashlib.sha256((retriever.model_id + "\n" + serialized).encode("utf-8")).hexdigest()
    if key not in index_cache:
        # An explicitly empty bank returns no evidence and needs no embeddings.
        index_cache[key] = (
            serialize_document_embeddings(retriever.encode_documents(bank), model_id=retriever.model_id)
            if bank else ""
        )
    # RuntimeMemoryDataset shares this state with both tools at load time.
    # Update an already-materialized RETRIEVE state without overwriting its
    # raw history, question metadata or any other tool-specific settings.
    tools_kwargs = extra_info["tools_kwargs"]
    for tool_name in ("runtime_memory", "retrieve_memory"):
        tool_state = tools_kwargs.get(tool_name)
        if tool_state is not None:
            tool_state["create_kwargs"].update(
                compressed_memory_bank=serialized, lossy_memory_index=index_cache[key]
            )
    return prepared


def convert_prepared_splits(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    banks: Mapping[MemoryScope, list[dict[str, str]]],
    retriever: Any,
) -> dict[str, int]:
    """Convert existing train/val/test splits, including test-only OOD datasets."""
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    source = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    paths = [source / f"{split}.parquet" for split in ("train", "val", "test")
             if (source / f"{split}.parquet").is_file()]
    if not paths:
        raise FileNotFoundError(f"No prepared parquet splits found in {source}.")
    if destination == source or any((destination / path.name).exists() for path in paths):
        raise ValueError("Choose a separate output directory without existing parquet splits.")

    splits = {path.name: pq.read_table(path).to_pylist() for path in paths}
    # Coverage is per history version, not per question. Validate every split
    # before embedding/writing, and never fall back to another history version.
    required_scopes = {_row_scope(row) for rows in splits.values() for row in rows}
    missing = sorted(required_scopes - banks.keys())
    if missing:
        preview = missing[:5]
        suffix = " ..." if len(missing) > len(preview) else ""
        raise ValueError(
            f"Missing external memory for {len(missing)} conversation/checkpoint scope(s) "
            f"(data_source, conversation_id[, checkpoint_id]): {preview!r}{suffix}. "
            "Provide one bank per scope. Checkpointed rows require an exact checkpoint_id match."
        )

    destination.mkdir(parents=True, exist_ok=True)
    index_cache: dict[str, str] = {}
    counts = {}
    for name, rows in splits.items():
        converted = []
        for row in tqdm(rows, desc=f"Converting {name}", unit="row"):
            key = _row_scope(row)
            converted.append(replace_compressed_memory(
                row, banks[key], retriever=retriever, index_cache=index_cache
            ))
        if converted:
            write_parquet(converted, destination / name)
        else:
            # Re-inferring an empty table would discard the prepared schema.
            copyfile(source / name, destination / name)
        counts[name] = len(converted)
    return counts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Prepared LLMLingua or raw-only parquet directory.")
    parser.add_argument(
        "--memory_file", required=True,
        help="JSON/JSONL with one bank per data_source + conversation_id, plus checkpoint_id for versioned history.",
    )
    parser.add_argument("--output_dir", required=True, help="Separate directory consumed by existing launchers.")
    parser.add_argument("--items_field", default="memories")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--retriever_model", default=QWEN3_RETRIEVER)
    parser.add_argument("--retriever_device", default="cuda:0")
    parser.add_argument("--retriever_batch_size", type=int, default=32)
    args = parser.parse_args(argv)
    banks = load_memory_banks(
        args.memory_file, items_field=args.items_field,
        text_field=args.text_field, id_field=args.id_field,
    )
    counts = convert_prepared_splits(
        input_dir=args.input_dir, output_dir=args.output_dir, banks=banks,
        retriever=EmbeddingRetriever(
            model_id=args.retriever_model, device=args.retriever_device,
            batch_size=args.retriever_batch_size,
        ),
    )
    print(f"Wrote {counts} to {args.output_dir}")


if __name__ == "__main__":
    main()
