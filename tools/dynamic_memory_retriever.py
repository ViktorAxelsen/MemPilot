"""Retrieval-query-conditioned search over MemPilot's independent memory views.

RETRIEVE searches query-agnostic memory M, while CURATE searches raw history H.
In the standard decoupled interface, the policy's evidence-search query is
distinct from the instruction used for delegated curation.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from threading import Lock
from typing import Any

from retrieval import (
    QWEN3_RETRIEVER,
    EmbeddingRetriever,
    deserialize_document_embeddings,
    document_text,
)


DEFAULT_DOCUMENT_CACHE_SIZE = 8
DYNAMIC_RETRIEVAL_TOP_K_ENV = "DYNAMIC_RETRIEVAL_TOP_K"
_SHARED_RETRIEVERS: dict[tuple[Any, ...], "DynamicMemoryRetriever"] = {}
_SHARED_RETRIEVERS_LOCK = Lock()


class DynamicMemoryRetriever:
    """Retrieve ranked records while reusing embeddings per independent corpus.

    Exclusions are supplied by the caller for the current selection; this
    retriever does not maintain a trajectory-wide record of previously seen IDs.
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = config or {}
        if not isinstance(config, Mapping):
            raise ValueError("dynamic_retrieval must be a mapping.")

        top_k_config = config
        if DYNAMIC_RETRIEVAL_TOP_K_ENV in os.environ:
            top_k_config = {"top_k": os.environ[DYNAMIC_RETRIEVAL_TOP_K_ENV]}
        self.top_k = _integer_setting(top_k_config, "top_k", default=0, minimum=0)
        self.document_cache_size = _integer_setting(
            config,
            "document_cache_size",
            default=DEFAULT_DOCUMENT_CACHE_SIZE,
            minimum=1,
        )
        self.model_id = str(config.get("model_id") or QWEN3_RETRIEVER).strip()
        self.device = str(config.get("device") or "cpu").strip()
        self.batch_size = _integer_setting(config, "batch_size", default=32, minimum=1)
        if self.enabled and (not self.model_id or not self.device):
            raise ValueError("Enabled dynamic_retrieval requires non-empty model_id and device.")

        self._retriever = (
            EmbeddingRetriever(
                model_id=self.model_id,
                device=self.device,
                batch_size=self.batch_size,
            )
            if self.enabled
            else None
        )
        self._document_embeddings: OrderedDict[str, Any] = OrderedDict()
        self._cache_lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.top_k > 0

    def resolve_top_k(self, requested_top_k: int | None = None) -> int:
        """Clamp a per-call request to the configured retrieval upper bound."""
        if requested_top_k is None:
            return self.top_k
        requested_top_k = int(requested_top_k)
        if requested_top_k < 0:
            raise ValueError("requested_top_k must be non-negative.")
        return min(requested_top_k, self.top_k)

    def build_document_cache_key(
        self,
        memory_bank: list[dict[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Build the cache key once when a tool instance is created."""
        if not self.enabled or not memory_bank:
            return ""

        metadata = metadata if isinstance(metadata, Mapping) else {}
        conversation_id = str(metadata.get("conversation_id") or "").strip()
        if conversation_id:
            identity = "\x1f".join(
                (
                    "metadata-v2",
                    str(metadata.get("data_source") or ""),
                    str(metadata.get("split") or ""),
                    conversation_id,
                    str(metadata.get("chunk_size") or ""),
                    str(metadata.get("chunking_mode") or ""),
                    self.model_id,
                    _memory_fingerprint(memory_bank),
                )
            )
            return hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return _memory_fingerprint(memory_bank)

    async def retrieve(
        self,
        memory_bank: list[dict[str, Any]],
        retrieval_query: str,
        *,
        excluded_ids: Iterable[str] = (),
        memory_index: Any = None,
        document_cache_key: str = "",
        requested_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the requested number of unique records, capped by configured ``top_k``."""
        effective_top_k = self.resolve_top_k(requested_top_k)
        if not self.enabled or not memory_bank or effective_top_k == 0:
            return []
        return await asyncio.to_thread(
            self._retrieve_sync,
            memory_bank,
            str(retrieval_query),
            frozenset(str(item_id).strip() for item_id in excluded_ids if str(item_id).strip()),
            memory_index,
            str(document_cache_key).strip(),
            effective_top_k,
        )

    def _retrieve_sync(
        self,
        memory_bank: list[dict[str, Any]],
        retrieval_query: str,
        excluded_ids: frozenset[str],
        memory_index: Any,
        document_cache_key: str,
        effective_top_k: int,
    ) -> list[dict[str, Any]]:
        if self._retriever is None:
            return []

        seen_ids = set(excluded_ids)
        eligible_indices: list[int] = []
        eligible_documents: list[dict[str, Any]] = []
        for index, item in enumerate(memory_bank):
            item_id = str(item.get("id") or "").strip()
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            eligible_indices.append(index)
            eligible_documents.append(item)
        if not eligible_documents:
            return []

        cache_key = document_cache_key or _memory_fingerprint(memory_bank)
        document_embeddings = self._get_document_embeddings(
            cache_key,
            memory_bank,
            memory_index,
        )

        import numpy as np

        embedding_matrix = np.asarray(document_embeddings)
        if len(eligible_indices) != len(memory_bank):
            embedding_matrix = embedding_matrix[eligible_indices]
        return self._retriever.retrieve(
            documents=eligible_documents,
            query=retrieval_query,
            top_k=min(effective_top_k, len(eligible_documents)),
            document_embeddings=embedding_matrix,
        )

    def _get_document_embeddings(
        self,
        cache_key: str,
        memory_bank: list[dict[str, Any]],
        memory_index: Any,
    ) -> Any:
        # Only LRU bookkeeping is serialized; index decoding and model inference
        # can proceed concurrently for independent calls in the same worker.
        with self._cache_lock:
            cached = self._document_embeddings.get(cache_key)
            if cached is not None:
                self._document_embeddings.move_to_end(cache_key)
                return cached

        document_embeddings = deserialize_document_embeddings(
            memory_index,
            model_id=self.model_id,
            expected_rows=len(memory_bank),
        )
        if document_embeddings is None:
            if self._retriever is None:
                return None
            document_embeddings = self._retriever.encode_documents(memory_bank)

        with self._cache_lock:
            cached = self._document_embeddings.get(cache_key)
            if cached is not None:
                self._document_embeddings.move_to_end(cache_key)
                return cached
            self._document_embeddings[cache_key] = document_embeddings
            if len(self._document_embeddings) > self.document_cache_size:
                self._document_embeddings.popitem(last=False)
        return document_embeddings


def get_dynamic_memory_retriever(
    config: Mapping[str, Any] | None = None,
) -> DynamicMemoryRetriever:
    """Build a retriever, optionally shared by sibling tools in one worker process."""

    config = config or {}
    retriever = DynamicMemoryRetriever(config)
    shared_worker_key = str(config.get("shared_worker_key") or "").strip()
    if not shared_worker_key:
        return retriever

    cache_key = (
        shared_worker_key,
        retriever.top_k,
        retriever.model_id,
        retriever.device,
        retriever.batch_size,
        retriever.document_cache_size,
    )
    with _SHARED_RETRIEVERS_LOCK:
        return _SHARED_RETRIEVERS.setdefault(cache_key, retriever)


def _memory_fingerprint(memory_bank: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in memory_bank:
        digest.update(str(item.get("id") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(document_text(item).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _integer_setting(
    config: Mapping[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dynamic_retrieval.{key} must be an integer.") from exc
    if value < minimum:
        raise ValueError(f"dynamic_retrieval.{key} must be at least {minimum}.")
    return value
