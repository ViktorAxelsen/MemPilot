"""Merge prepared multimodal-memory Verl splits without re-splitting them."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from data.memory_qa import write_parquet


DEFAULT_INPUT_DIRS = (
    "data/mem_gallery",
    "data/worldmemarena_lifelong",
    "data/h2hmem_dyadic",
)
SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine already-prepared multimodal-memory train/validation/test parquet files while preserving "
            "each row's data_source and original conversation-level split."
        )
    )
    parser.add_argument(
        "--input_dir",
        action="append",
        dest="input_dirs",
        help=(
            "Prepared data directory containing train.parquet, val.parquet, and test.parquet. Repeat for each "
            f"dataset. Defaults to: {', '.join(DEFAULT_INPUT_DIRS)}."
        ),
    )
    parser.add_argument("--output_dir", default="data/multimodal_unified")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    summary = merge_prepared_datasets(
        input_dirs=args.input_dirs or DEFAULT_INPUT_DIRS,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    for split in SPLITS:
        counts = ", ".join(
            f"{source}={count}"
            for source, count in summary["splits"][split]["source_counts"].items()
        )
        print(f"{split}: {summary['splits'][split]['rows']} rows ({counts})")
    print(f"Wrote unified data and manifest to {Path(args.output_dir).expanduser()}")


def merge_prepared_datasets(
    *,
    input_dirs: Sequence[str | Path],
    output_dir: str | Path,
    seed: int = 13,
) -> dict[str, Any]:
    """Merge source splits, validate provenance/leakage, and write all three splits."""

    source_dirs = _unique_input_dirs(input_dirs)
    merged: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    source_manifest: list[dict[str, Any]] = []

    for source_dir in source_dirs:
        loaded = {
            split: _load_parquet_rows(source_dir / f"{split}.parquet")
            for split in SPLITS
        }
        sources = {
            _required_text(row.get("data_source"), f"{source_dir} row data_source")
            for rows in loaded.values()
            for row in rows
        }
        if len(sources) != 1:
            raise ValueError(
                f"Each input directory must contain exactly one data_source; {source_dir} has {sorted(sources)}."
            )
        data_source = next(iter(sources))

        for split, rows in loaded.items():
            merged[split].extend(
                _prepare_row(row, expected_source=data_source, expected_split=split)
                for row in rows
            )
        source_manifest.append(
            {
                "data_source": data_source,
                "input_dir": str(source_dir),
                **{f"{split}_rows": len(loaded[split]) for split in SPLITS},
            }
        )

    _validate_sources_are_unique(source_manifest)
    _validate_no_split_leakage(merged)

    # Verl shuffles training data, but a deterministic physical shuffle avoids storing
    # the unified parquet in source-sized contiguous blocks.
    random.Random(seed).shuffle(merged["train"])

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_parquet(merged[split], destination / f"{split}.parquet")

    summary = {
        "seed": int(seed),
        "sources": source_manifest,
        "splits": {
            split: {
                "rows": len(merged[split]),
                "source_counts": dict(
                    sorted(Counter(row["data_source"] for row in merged[split]).items())
                ),
            }
            for split in SPLITS
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared split: {path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Install `pyarrow` in the Verl environment to merge parquet files.") from exc
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _prepare_row(
    row: Mapping[str, Any],
    *,
    expected_source: str,
    expected_split: str,
) -> dict[str, Any]:
    prepared = copy.deepcopy(dict(row))
    data_source = _required_text(prepared.get("data_source"), "row data_source")
    if data_source != expected_source:
        raise ValueError(
            f"Input source changed within one directory: expected {expected_source!r}, got {data_source!r}."
        )
    extra_info = prepared.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError(f"{data_source} rows require mapping-valued extra_info.")
    split = _required_text(extra_info.get("split"), f"{data_source} extra_info.split").casefold()
    if split != expected_split:
        raise ValueError(
            f"{data_source} row stored in {expected_split}.parquet declares split={split!r}."
        )
    extra_info["dataset_source"] = data_source
    _validate_runtime_metadata_source(extra_info, data_source)
    _required_text(extra_info.get("conversation_id"), f"{data_source} conversation_id")
    _required_text(extra_info.get("question_id"), f"{data_source} question_id")
    return prepared


def _validate_runtime_metadata_source(extra_info: Mapping[str, Any], data_source: str) -> None:
    tools_kwargs = extra_info.get("tools_kwargs")
    runtime_memory = tools_kwargs.get("runtime_memory") if isinstance(tools_kwargs, Mapping) else None
    create_kwargs = runtime_memory.get("create_kwargs") if isinstance(runtime_memory, Mapping) else None
    metadata = create_kwargs.get("metadata") if isinstance(create_kwargs, Mapping) else None
    metadata_source = metadata.get("data_source") if isinstance(metadata, Mapping) else None
    if metadata_source is not None and str(metadata_source).strip() != data_source:
        raise ValueError(
            f"Row data_source {data_source!r} disagrees with runtime metadata {metadata_source!r}."
        )


def _validate_sources_are_unique(source_manifest: Iterable[Mapping[str, Any]]) -> None:
    sources = [str(item["data_source"]) for item in source_manifest]
    duplicates = sorted(source for source, count in Counter(sources).items() if count > 1)
    if duplicates:
        raise ValueError(f"Multiple input directories provide the same data_source: {duplicates}")


def _validate_no_split_leakage(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    def conversation_keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
        return {
            (
                str(row["data_source"]),
                str(row["extra_info"]["conversation_id"]),
            )
            for row in rows
        }

    keys_by_split = {
        split: conversation_keys(rows_by_split[split])
        for split in SPLITS
    }
    for left, right in combinations(SPLITS, 2):
        leaked = sorted(keys_by_split[left] & keys_by_split[right])
        if leaked:
            preview = leaked[:5]
            suffix = " ..." if len(leaked) > len(preview) else ""
            raise ValueError(
                f"Conversation-level {left}/{right} leakage detected: {preview}{suffix}"
            )


def _unique_input_dirs(values: Sequence[str | Path]) -> list[Path]:
    if not values:
        raise ValueError("At least one input directory is required.")
    paths = [Path(value).expanduser() for value in values]
    normalized = [str(path.resolve()) for path in paths]
    duplicates = sorted(path for path, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate input directories: {duplicates}")
    return paths


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


if __name__ == "__main__":
    main()
