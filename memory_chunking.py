"""Pack complete dialogue turns and image captions into memory chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_CHUNK_TOKENS = 256


@dataclass(frozen=True)
class AtomicDialogueTurn:
    """A turn that must remain intact, including any attached image captions."""

    text: str
    preview_text: str
    protected_preview_text: str = ""
    turn_id: str = ""
    images: tuple[dict[str, Any], ...] = ()


class AtomicDialogueChunker:
    """Pack complete turns without token-splitting an oversized individual turn."""

    def __init__(self, tokenizer: Any, target_tokens: int = DEFAULT_CHUNK_TOKENS):
        if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("AtomicDialogueChunker requires a tokenizer with encode().")
        if int(target_tokens) <= 0:
            raise ValueError("chunk_size must be positive.")
        self.tokenizer = tokenizer
        self.target_tokens = int(target_tokens)

    def chunk_session(
        self,
        turns: Iterable[AtomicDialogueTurn],
        *,
        time: str = "",
    ) -> list[dict[str, Any]]:
        """Return session-local chunks; a single large turn becomes one large chunk."""
        chunks: list[dict[str, Any]] = []
        packed: list[AtomicDialogueTurn] = []
        packed_tokens = 0
        for turn in turns:
            if not isinstance(turn, AtomicDialogueTurn):
                raise TypeError("AtomicDialogueChunker expects AtomicDialogueTurn values.")
            text = str(turn.text or "").strip()
            if not text:
                continue
            turn_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            if packed and packed_tokens + turn_tokens > self.target_tokens:
                chunks.append(self._build_chunk(packed, time=time))
                packed, packed_tokens = [], 0
            packed.append(turn)
            packed_tokens += turn_tokens
            if packed_tokens >= self.target_tokens:
                chunks.append(self._build_chunk(packed, time=time))
                packed, packed_tokens = [], 0
        if packed:
            chunks.append(self._build_chunk(packed, time=time))
        return chunks

    @staticmethod
    def _build_chunk(turns: list[AtomicDialogueTurn], *, time: str) -> dict[str, Any]:
        return {
            "speaker": "",
            "time": str(time or ""),
            "text": "\n\n".join(turn.text.strip() for turn in turns if turn.text.strip()),
            "preview_text": "\n\n".join(
                (turn.preview_text or turn.text).strip()
                for turn in turns
                if (turn.preview_text or turn.text).strip()
            ),
            "protected_preview_text": "\n\n".join(
                turn.protected_preview_text.strip()
                for turn in turns
                if turn.protected_preview_text.strip()
            ),
            "turn_ids": [turn.turn_id for turn in turns if turn.turn_id],
            "images": [dict(image) for turn in turns for image in turn.images],
        }
