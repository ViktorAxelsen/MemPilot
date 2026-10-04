"""Reusable dense text retrieval for raw-history and query-agnostic memory records."""

from __future__ import annotations

import base64
import importlib
import json
from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any, TypeVar


QWEN3_RETRIEVER = "Qwen/Qwen3-Embedding-0.6B"

DocumentT = TypeVar("DocumentT")
_SENTENCE_TRANSFORMER_LOAD_LOCK = Lock()


def _install_sentence_transformers_tokenizer_fallback() -> None:
    """Backport the text-only AutoProcessor fallback pending upstream release."""
    transformer_module = importlib.import_module("sentence_transformers.base.modules.transformer")
    auto_processor = transformer_module.AutoProcessor
    if getattr(auto_processor, "_mempilot_tokenizer_fallback", False) is True:
        return

    from transformers import AutoTokenizer

    class AutoProcessorWithTokenizerFallback:
        _mempilot_tokenizer_fallback = True

        @staticmethod
        def from_pretrained(model_name_or_path: str, **kwargs: Any) -> Any:
            try:
                return auto_processor.from_pretrained(model_name_or_path, **kwargs)
            except ValueError as exc:
                message = str(exc)
                if "Unrecognized processing class" not in message and "does not contain" not in message:
                    raise
                return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

    transformer_module.AutoProcessor = AutoProcessorWithTokenizerFallback


def serialize_document_embeddings(embeddings: Any, *, model_id: str) -> str:
    """Serialize a dense document index compactly for portable parquet rows."""
    import numpy as np

    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError("document embeddings must be a two-dimensional matrix.")
    matrix = np.ascontiguousarray(matrix, dtype="<f2")
    return json.dumps(
        {
            "model_id": str(model_id),
            "rows": int(matrix.shape[0]),
            "dimensions": int(matrix.shape[1]),
            "encoding": "base64-float16-le",
            "data": base64.b64encode(matrix.tobytes()).decode("ascii"),
        },
        separators=(",", ":"),
    )


def deserialize_document_embeddings(
    raw: Any,
    *,
    model_id: str,
    expected_rows: int,
) -> Any | None:
    """Decode a prepared dense index, returning ``None`` for legacy rows."""
    if raw in (None, ""):
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("memory_index must contain valid JSON.") from exc
    if not isinstance(raw, dict):
        raise ValueError("memory_index must be a mapping or JSON mapping.")
    if str(raw.get("model_id") or "") != str(model_id):
        raise ValueError(
            "Prepared memory index model does not match dynamic_retrieval.model_id; "
            "regenerate the data or restore the matching retriever."
        )
    if raw.get("encoding") != "base64-float16-le":
        raise ValueError("Unsupported memory_index encoding.")
    try:
        rows = int(raw["rows"])
        dimensions = int(raw["dimensions"])
        payload = base64.b64decode(str(raw["data"]), validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid memory_index metadata or payload.") from exc
    if rows != int(expected_rows) or rows < 0 or dimensions <= 0:
        raise ValueError("memory_index shape does not match the complete memory bank.")

    import numpy as np

    expected_bytes = rows * dimensions * np.dtype("<f2").itemsize
    if len(payload) != expected_bytes:
        raise ValueError("memory_index payload length does not match its declared shape.")
    return np.frombuffer(payload, dtype="<f2").reshape(rows, dimensions)


def document_text(document: Any) -> str:
    """Convert a string or structured record into retriever input text."""
    if isinstance(document, str):
        return document
    if not isinstance(document, dict):
        return str(document)

    preferred_fields = (
        "speaker",
        "time",
        "text",
        "instruction",
        "context",
        "model",
    )
    parts = [str(document[key]) for key in preferred_fields if document.get(key)]
    if parts:
        return " ".join(parts)
    return json.dumps(document, ensure_ascii=False, sort_keys=True, default=str)


class EmbeddingRetriever:
    """Dense text retriever backed by a SentenceTransformer model."""

    def __init__(
        self,
        model_id: str = QWEN3_RETRIEVER,
        device: str = "cuda:0",
        batch_size: int = 32,
        model: Any = None,
        tokenizer: Any = None,
        document_to_text: Callable[[Any], str] | None = None,
    ):
        model_id = str(model_id).strip()
        if not model_id:
            raise ValueError("Retriever model_id cannot be empty.")
        if batch_size <= 0:
            raise ValueError("retriever_batch_size must be positive.")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._model = model
        self._tokenizer = tokenizer
        self.document_to_text = document_to_text or document_text

    @property
    def model(self) -> Any:
        if self._model is None:
            # Transformers' meta-device loading context is not thread-safe. A
            # process-wide lock also prevents duplicate lazy loads when several
            # async tool calls first reach this retriever at the same time.
            with _SENTENCE_TRANSFORMER_LOAD_LOCK:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise ImportError(
                            "Embedding retrieval requires `sentence-transformers`. "
                            "Install requirements.txt."
                        ) from exc
                    _install_sentence_transformers_tokenizer_fallback()
                    self._model = SentenceTransformer(
                        self.model_id,
                        model_kwargs={"device_map": self.device},
                    )
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = getattr(self.model, "tokenizer", None)
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError("Dialogue chunking requires `transformers`.") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    def encode_documents(self, documents: Sequence[Any]) -> Any:
        texts = [self.document_to_text(document) for document in documents]
        if not texts:
            return None
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_queries(self, queries: Sequence[str]) -> Any:
        texts = [str(query) for query in queries]
        if not texts:
            return None
        return self.model.encode(
            texts,
            prompt_name="query",
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve_many(
        self,
        documents: Sequence[DocumentT],
        queries: Sequence[str],
        top_k: int,
        document_embeddings: Any = None,
    ) -> list[list[DocumentT]]:
        """Retrieve for a batch of queries while encoding the shared document bank once."""
        queries = list(queries)
        top_k = max(int(top_k or 0), 0)
        if not queries:
            return []
        if top_k == 0 or not documents:
            return [[] for _ in queries]
        if document_embeddings is None:
            document_embeddings = self.encode_documents(documents)

        import numpy as np

        embedding_matrix = np.asarray(document_embeddings)
        if len(embedding_matrix) != len(documents):
            raise ValueError("document_embeddings must contain one embedding per document.")
        query_embeddings = np.asarray(self.encode_queries(queries))
        if len(query_embeddings) != len(queries):
            raise ValueError("query embeddings must contain one embedding per query.")

        score_matrix = query_embeddings @ embedding_matrix.T
        results = []
        for scores in score_matrix:
            ranked_indices = _stable_top_k_indices(scores, top_k)
            results.append([documents[int(idx)] for idx in ranked_indices])
        return results

    def retrieve(
        self,
        documents: Sequence[DocumentT],
        query: str,
        top_k: int,
        document_embeddings: Any = None,
    ) -> list[DocumentT]:
        return self.retrieve_many(
            documents=documents,
            queries=[query],
            top_k=top_k,
            document_embeddings=document_embeddings,
        )[0]


def _stable_top_k_indices(scores: Any, top_k: int) -> Any:
    """Select only the required candidates while preserving index-based tie breaks."""
    import numpy as np

    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("retrieval scores must be one-dimensional.")
    count = min(max(int(top_k), 0), len(scores))
    if count == 0:
        return np.empty(0, dtype=np.intp)

    if count == len(scores):
        candidates = np.arange(len(scores), dtype=np.intp)
    else:
        threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
        above_threshold = np.flatnonzero(scores > threshold)
        tied_at_threshold = np.flatnonzero(scores == threshold)
        candidates = np.concatenate(
            (
                above_threshold,
                tied_at_threshold[: count - len(above_threshold)],
            )
        )

    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order]
