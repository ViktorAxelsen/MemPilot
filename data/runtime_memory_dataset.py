"""verl dataset view that materializes query-only runtime-memory prompts."""

from __future__ import annotations

from typing import Any

from data.memory_qa import configure_runtime_tools
from data.runtime_memory_prompts import build_runtime_memory_messages
from memory_access import get_max_parallel_calls
from verl.utils.dataset.rl_dataset import RLHFDataset


class RuntimeMemoryDataset(RLHFDataset):
    """Project rich prepared rows into query-only controller prompts."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.max_parallel_calls = get_max_parallel_calls()
        super().__init__(*args, **kwargs)

    def maybe_filter_out_long_prompts(self, dataframe=None):
        # Rebuild the query-only view before verl measures prompt length.
        # This also removes initial memory previews from older parquet rows.
        dataframe = dataframe.map(
            _replace_prompt,
            fn_kwargs={
                "prompt_key": self.prompt_key,
                "max_parallel_calls": self.max_parallel_calls,
            },
            desc="Building query-only runtime-memory prompts",
        )
        return super().maybe_filter_out_long_prompts(dataframe)


def _replace_prompt(
    example: dict[str, Any],
    *,
    prompt_key: str,
    max_parallel_calls: int,
) -> dict[str, Any]:
    extra_info = configure_runtime_tools(
        example.get("extra_info"),
    )
    return {
        prompt_key: build_runtime_memory_messages(
            {**example, "extra_info": extra_info},
            max_parallel_calls=max_parallel_calls,
        ),
        "extra_info": extra_info,
    }
