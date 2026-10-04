"""Shared memory-field names and orchestrator concurrency settings.

Memory access always uses dynamic retrieval through RETRIEVE and CURATE.
The policy decides each next action without a mandatory initial plan block.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


MAX_PARALLEL_CALLS_ENV = "MAX_PARALLEL_CALLS"
DEFAULT_MAX_PARALLEL_CALLS = 2
LOSSY_PREVIEW_FIELD = "lossy_preview"
MEMORY_TOOL_CALL_REQUIRED_FIELD = "memory_tool_call_required"


def get_max_parallel_calls(environ: Mapping[str, str] | None = None) -> int:
    """Read the tool-call concurrency limit shared with the runtime prompt."""

    source = os.environ if environ is None else environ
    try:
        value = int(source.get(MAX_PARALLEL_CALLS_ENV, DEFAULT_MAX_PARALLEL_CALLS))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{MAX_PARALLEL_CALLS_ENV} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"{MAX_PARALLEL_CALLS_ENV} must be positive.")
    return value
