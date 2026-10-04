"""Offline LLMLingua-2 compression for the query-agnostic memory bank M.

This builds the default RETRIEVE view without conditioning on a question.
The raw history used by CURATE remains available separately; multimodal data
helpers also apply the compressor to image captions while preserving image IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from tqdm.auto import tqdm


DEFAULT_LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
DEFAULT_MEMORY_COMPRESSION_RATE = 0.4


@dataclass(frozen=True)
class TextCompression:
    """One compressed text and the token counts reported by LLMLingua."""

    text: str
    original_tokens: int
    compressed_tokens: int


class LLMLingua2MemoryCompressor:
    """Compress each memory chunk independently and cache repeated chunks."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLMLINGUA2_MODEL,
        rate: float = DEFAULT_MEMORY_COMPRESSION_RATE,
        device: str = "cuda:0",
        batch_size: int = 32,
        backend: Any = None,
    ):
        model_name = str(model_name).strip()
        if not model_name:
            raise ValueError("memory_compressor_model must be non-empty.")
        if not 0.0 < float(rate) <= 1.0:
            raise ValueError("memory_compression_rate must be in (0, 1].")
        if int(batch_size) <= 0:
            raise ValueError("compressor_batch_size must be positive.")
        self.model_name = model_name
        self.rate = float(rate)
        self.device = str(device).strip() or "cpu"
        self.batch_size = int(batch_size)
        self._backend = backend
        self._cache: dict[str, TextCompression] = {}
        self.num_model_calls = 0

    @property
    def backend(self) -> Any:
        if self._backend is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as exc:
                raise ImportError(
                    "LLMLingua-2 preprocessing requires `llmlingua==0.2.2`. "
                    "Install requirements.txt."
                ) from exc
            self._backend = PromptCompressor(
                model_name=self.model_name,
                use_llmlingua2=True,
                device_map=self.device,
                llmlingua2_config={"max_batch_size": self.batch_size},
            )
        return self._backend

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def compress_text(self, text: str) -> TextCompression:
        """Compress text without conditioning on the downstream question."""
        text = str(text or "").strip()
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        if not text:
            result = TextCompression(text="", original_tokens=0, compressed_tokens=0)
            self._cache[text] = result
            return result

        raw = self.backend.compress_prompt(text, **self._compression_kwargs())
        if not isinstance(raw, dict):
            raise TypeError("LLMLingua-2 compress_prompt() must return a mapping.")
        result = TextCompression(
            text=_visible_compressed_text(raw.get("compressed_prompt")),
            original_tokens=_nonnegative_int(raw.get("origin_tokens")),
            compressed_tokens=_nonnegative_int(raw.get("compressed_tokens")),
        )
        self._cache[text] = result
        self.num_model_calls += 1
        return result

    def compress_texts(
        self,
        texts: Iterable[str],
        *,
        show_progress: bool = False,
        description: str = "Compressing memory previews",
    ) -> list[TextCompression]:
        """Compress unique uncached texts in real LLMLingua-2 batches."""
        normalized = [str(text or "").strip() for text in texts]
        pending = list(dict.fromkeys(text for text in normalized if text and text not in self._cache))
        for text in normalized:
            if not text and text not in self._cache:
                self._cache[text] = TextCompression(text="", original_tokens=0, compressed_tokens=0)

        progress = tqdm(
            total=len(pending),
            desc=description,
            unit="chunk",
            dynamic_ncols=True,
            disable=not show_progress or not pending,
        )
        try:
            for start in range(0, len(pending), self.batch_size):
                batch = pending[start : start + self.batch_size]
                if len(batch) == 1:
                    self.compress_text(batch[0])
                else:
                    self._compress_batch(batch)
                progress.update(len(batch))
        finally:
            progress.close()
        return [self._cache[text] for text in normalized]

    def _compress_batch(self, texts: list[str]) -> None:
        raw = self.backend.compress_prompt(texts, **self._compression_kwargs())
        self.num_model_calls += 1
        if not isinstance(raw, dict):
            raise TypeError("LLMLingua-2 compress_prompt() must return a mapping.")
        outputs = raw.get("compressed_prompt_list")
        if not isinstance(outputs, list) or len(outputs) != len(texts):
            # Keep compatibility with custom backends that only implement the
            # single-string API.
            for text in texts:
                self.compress_text(text)
            return

        original_counts = _per_text_token_counts(raw, "origin_tokens_list", texts, self.backend)
        compressed_counts = _per_text_token_counts(raw, "compressed_tokens_list", outputs, self.backend)
        for text, output, original_tokens, compressed_tokens in zip(
            texts,
            outputs,
            original_counts,
            compressed_counts,
        ):
            self._cache[text] = TextCompression(
                text=_visible_compressed_text(output),
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
            )

    def _compression_kwargs(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "use_context_level_filter": False,
            "use_token_level_filter": True,
            "force_tokens": ["\n", ":", "?"],
            "force_reserve_digit": True,
            "chunk_end_tokens": [".", "\n"],
        }


def uncompressed_memory_view(
    memory_items: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Explicit no-compression view used by tests and ablations."""
    return [dict(item) for item in memory_items], {
        "enabled": False,
        "model": "",
        "target_rate": 1.0,
        "original_tokens": 0,
        "compressed_tokens": 0,
        "actual_rate": 1.0,
    }


def _nonnegative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _visible_compressed_text(value: Any) -> str:
    text = str(value or "").strip()
    # Do not silently reveal original text if an aggressive rate removes all content.
    return text or "[content omitted by compressor]"


def _per_text_token_counts(
    raw: dict[str, Any],
    list_key: str,
    texts: list[Any],
    backend: Any,
) -> list[int]:
    provided = raw.get(list_key)
    if isinstance(provided, list) and len(provided) == len(texts):
        return [_nonnegative_int(value) for value in provided]
    return [_backend_token_count(backend, str(text or "")) for text in texts]


def _backend_token_count(backend: Any, text: str) -> int:
    counter = getattr(backend, "get_token_length", None)
    if callable(counter):
        try:
            return _nonnegative_int(counter(text, use_oai_tokenizer=True))
        except TypeError:
            return _nonnegative_int(counter(text))
    tokenizer = getattr(backend, "oai_tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return len(encode(text))
    return len(text.split())
