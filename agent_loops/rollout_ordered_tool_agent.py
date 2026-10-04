"""MemPilot rollouts with resource accounting and prefix-based marginal utility.

The loop preserves a rollout-local action ordering, accounts for orchestrator
generation under ideal KV-prefix reuse, and probes memory-stage
prefixes during training without extending the executed tool trajectory.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from time import perf_counter
from typing import Any

from omegaconf import OmegaConf
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

from agent_loops.route_schema import order_runtime_memory_enums
from marginal_utility import (
    MARGINAL_UTILITY_INVALID_FORMAT,
    MARGINAL_UTILITY_KEY,
    MARGINAL_UTILITY_NO_STAGES,
    MARGINAL_UTILITY_PROBE_FAILED,
    build_marginal_utility_metadata,
    build_marginal_utility_skip_metadata,
    load_marginal_utility_settings,
)
from rewards.memory_qa import validate_memory_qa_trajectory_format
from runtime_metrics import record_base_model_trajectory


logger = logging.getLogger(__name__)
_RUNTIME_ENUMS_ORDERED = "_runtime_memory_enums_ordered"
_TRAIN_SPLIT = "train"
_ANSWER_OPEN_TAG = "<answer>"
_ANSWER_CLOSE_TAG = "</answer>"
_REWARD_FUNCTION_CACHE: dict[str, Any] = {}
_DEFAULT_BASE_INPUT_PRICE_PER_MILLION_USD = 0.05
_DEFAULT_BASE_OUTPUT_PRICE_PER_MILLION_USD = 0.25
# Qwen3-VL-8B's fixed-provider profile is used as a configurable latency proxy
# for the locally hosted Qwen3-4B policy.
_DEFAULT_BASE_LATENCY_SECONDS = 0.332
_DEFAULT_BASE_INPUT_SECONDS_PER_TOKEN = 0.000003
_DEFAULT_BASE_OUTPUT_SECONDS_PER_TOKEN = 0.00664


class _MarginalUtilityContractError(RuntimeError):
    """Raised when configured reward/probe metadata violates the shared contract."""


@dataclass(frozen=True)
class _ProbeCheckpoint:
    prompt_ids: list[int]
    image_data: Any
    video_data: Any
    audio_data: Any
    mm_processor_kwargs: dict[str, Any]


class RolloutOrderedToolAgentLoop(ToolAgentLoop):
    """Run iterative memory orchestration and collect training-only stage credits.

    Parallel memory calls in one assistant turn form a single causal stage.
    For K completed stages, capture K+1 prefixes: before the first stage and
    after each returned observation. Greedy answer-only probes estimate the
    signed answer-quality gain from each stage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_input_price = _nonnegative_setting(
            self.config,
            "algorithm.gdpo.base_model_input_price_per_million_usd",
            _DEFAULT_BASE_INPUT_PRICE_PER_MILLION_USD,
        )
        self._base_output_price = _nonnegative_setting(
            self.config,
            "algorithm.gdpo.base_model_output_price_per_million_usd",
            _DEFAULT_BASE_OUTPUT_PRICE_PER_MILLION_USD,
        )
        self._base_latency = _nonnegative_setting(
            self.config,
            "algorithm.gdpo.base_model_latency_base_seconds",
            _DEFAULT_BASE_LATENCY_SECONDS,
        )
        self._base_input_latency = _nonnegative_setting(
            self.config,
            "algorithm.gdpo.base_model_latency_input_seconds_per_token",
            _DEFAULT_BASE_INPUT_SECONDS_PER_TOKEN,
        )
        self._base_output_latency = _nonnegative_setting(
            self.config,
            "algorithm.gdpo.base_model_latency_output_seconds_per_token",
            _DEFAULT_BASE_OUTPUT_SECONDS_PER_TOKEN,
        )
        self._marginal_utility = load_marginal_utility_settings(self.config)
        self._reset_marginal_capture(is_training=False)

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        split = _sample_split(kwargs.get("extra_info"))
        self._reset_marginal_capture(is_training=split == _TRAIN_SPLIT)
        output = await super().run(sampling_params, **kwargs)
        dataset_source = str(_unwrap_scalar(kwargs.get("data_source")) or "").strip()
        if dataset_source:
            output.extra_fields["dataset_source"] = dataset_source
        record_base_model_trajectory(
            output.extra_fields,
            initial_prompt_tokens=len(output.prompt_ids),
            response_mask=output.response_mask,
            input_price_per_million_usd=self._base_input_price,
            output_price_per_million_usd=self._base_output_price,
            latency_base_seconds=self._base_latency,
            latency_input_seconds_per_token=self._base_input_latency,
            latency_output_seconds_per_token=self._base_output_latency,
        )
        if self._is_training_rollout:
            # Auxiliary probes are training overhead, not part of the reported
            # policy/delegation token usage, cost, or latency proxy.
            await self._attach_marginal_utility_metadata(
                output,
                sampling_params=sampling_params,
                sample_kwargs=kwargs,
            )
        return output

    async def _handle_pending_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
    ) -> AgentState:
        if not getattr(agent_data, _RUNTIME_ENUMS_ORDERED, False):
            schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
            agent_data._active_tool_schemas = order_runtime_memory_enums(
                schemas,
                ordering_key=agent_data.request_id,
            )
            setattr(agent_data, _RUNTIME_ENUMS_ORDERED, True)

        state = await super()._handle_pending_state(agent_data, sampling_params)
        if self._is_training_rollout:
            if self._marginal_checkpoints:
                raise RuntimeError("Marginal-utility initial checkpoint was recorded twice.")
            self._marginal_request_id = agent_data.request_id
            self._marginal_checkpoints.append(_checkpoint(agent_data))
        return state

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        state = await super()._handle_generating_state(
            agent_data,
            sampling_params,
            ignore_termination=ignore_termination,
        )
        if self._is_training_rollout and state == AgentState.PROCESSING_TOOLS:
            if self._marginal_pending_span is not None:
                raise RuntimeError("A marginal-utility stage was left pending across generations.")
            end = len(agent_data.response_mask)
            start = self._marginal_next_decision_start
            if end <= start:
                raise RuntimeError("A tool stage contains no policy decision tokens.")
            self._marginal_pending_span = (start, end)
        return state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if self._is_training_rollout:
            if state == AgentState.GENERATING:
                if self._marginal_pending_span is None:
                    raise RuntimeError("A completed tool stage has no recorded decision span.")
                # Credit the policy reasoning/action span before this batch of
                # observations; concurrent calls share one stage and one delta.
                self._marginal_decision_spans.append(self._marginal_pending_span)
                self._marginal_checkpoints.append(_checkpoint(agent_data))
                self._marginal_next_decision_start = len(agent_data.response_mask)
            self._marginal_pending_span = None
        return state

    def _reset_marginal_capture(self, *, is_training: bool) -> None:
        self._is_training_rollout = is_training
        self._marginal_request_id: str | None = None
        self._marginal_checkpoints: list[_ProbeCheckpoint] = []
        self._marginal_decision_spans: list[tuple[int, int]] = []
        self._marginal_pending_span: tuple[int, int] | None = None
        self._marginal_next_decision_start = 0

    async def _attach_marginal_utility_metadata(
        self,
        output,
        *,
        sampling_params: dict[str, Any],
        sample_kwargs: dict[str, Any],
    ) -> None:
        stage_count = len(self._marginal_decision_spans)
        if not stage_count:
            output.extra_fields[MARGINAL_UTILITY_KEY] = build_marginal_utility_skip_metadata(
                MARGINAL_UTILITY_NO_STAGES,
                stage_count=0,
            )
            return
        if len(self._marginal_checkpoints) != stage_count + 1:
            raise RuntimeError("Marginal-utility checkpoints and tool stages are misaligned.")
        if self._marginal_pending_span is not None:
            raise RuntimeError("Marginal-utility rollout ended with an incomplete tool stage.")

        reward_fn, data_source, ground_truth, dataset_extra_info = _probe_reward_context(
            self.config,
            sample_kwargs,
        )
        main_extra_info = dict(dataset_extra_info)
        main_extra_info.update(output.extra_fields)
        main_solution = self.tokenizer.decode(
            output.response_ids,
            skip_special_tokens=True,
        )
        if not validate_memory_qa_trajectory_format(
            solution_str=main_solution,
            ground_truth=ground_truth,
            extra_info=main_extra_info,
        ):
            output.extra_fields[MARGINAL_UTILITY_KEY] = build_marginal_utility_skip_metadata(
                MARGINAL_UTILITY_INVALID_FORMAT,
                stage_count=stage_count,
            )
            return

        started = perf_counter()
        probe_outputs: list[Any] = []
        try:
            probe_outputs = await asyncio.gather(
                *(
                    self._generate_answer_probe(
                        checkpoint,
                        sampling_params=sampling_params,
                    )
                    for checkpoint in self._marginal_checkpoints
                ),
                return_exceptions=True,
            )
            failures = [value for value in probe_outputs if isinstance(value, BaseException)]
            if failures:
                raise RuntimeError(
                    f"{len(failures)} of {len(probe_outputs)} counterfactual probes failed"
                ) from failures[0]

            probe_rewards = []
            for probe_output in probe_outputs:
                probe_solution = _tagged_probe_solution(
                    self.tokenizer.decode(
                        probe_output.token_ids,
                        skip_special_tokens=True,
                    )
                )
                probe_reward = await _invoke_reward(
                    reward_fn,
                    data_source=data_source,
                    solution_str=probe_solution,
                    ground_truth=ground_truth,
                    extra_info=dict(dataset_extra_info),
                )
                # Probe only answer quality: resource penalties are already
                # represented by the trajectory-level cost/latency objectives.
                probe_rewards.append(
                    _numeric_reward_field(probe_reward, "raw_performance_reward")
                )
        except _MarginalUtilityContractError:
            raise
        except Exception as exc:
            elapsed = perf_counter() - started
            output_tokens = sum(
                len(value.token_ids)
                for value in probe_outputs
                if not isinstance(value, BaseException) and hasattr(value, "token_ids")
            )
            logger.exception("[MARGINAL UTILITY] counterfactual probe failed")
            output.extra_fields[MARGINAL_UTILITY_KEY] = build_marginal_utility_skip_metadata(
                MARGINAL_UTILITY_PROBE_FAILED,
                stage_count=stage_count,
                failure_type=type(exc).__name__,
                probe_count=sum(
                    not isinstance(value, BaseException) for value in probe_outputs
                ),
                probe_output_tokens=output_tokens,
                probe_seconds=elapsed,
            )
            return

        output.extra_fields[MARGINAL_UTILITY_KEY] = build_marginal_utility_metadata(
            decision_spans=self._marginal_decision_spans,
            probe_rewards=probe_rewards,
            probe_output_tokens=sum(len(value.token_ids) for value in probe_outputs),
            probe_seconds=perf_counter() - started,
        )

    async def _generate_answer_probe(
        self,
        checkpoint: _ProbeCheckpoint,
        *,
        sampling_params: dict[str, Any],
    ):
        """Greedily answer from one fixed prefix without executing further tools."""

        if not self._marginal_request_id:
            raise RuntimeError("Marginal-utility probe is missing the rollout request ID.")
        answer_prefix = self.tokenizer.encode(
            _ANSWER_OPEN_TAG,
            add_special_tokens=False,
        )
        probe_sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": sampling_params.get("repetition_penalty", 1.0),
            "logprobs": False,
            "max_tokens": self._marginal_utility.max_answer_tokens,
        }
        # Token-only rollout servers may not initialize a tokenizer and thus
        # reject string stops. Use the closing tag as a stop only when it is a
        # single token; otherwise EOS/max_tokens terminates generation and the
        # decoded result is still trimmed at the first closing tag.
        answer_close_ids = self.tokenizer.encode(
            _ANSWER_CLOSE_TAG,
            add_special_tokens=False,
        )
        if len(answer_close_ids) == 1:
            probe_sampling_params["stop_token_ids"] = answer_close_ids
        return await self.server_manager.generate(
            request_id=self._marginal_request_id,
            prompt_ids=[*checkpoint.prompt_ids, *answer_prefix],
            sampling_params=probe_sampling_params,
            image_data=checkpoint.image_data,
            video_data=checkpoint.video_data,
            audio_data=checkpoint.audio_data,
            mm_processor_kwargs=checkpoint.mm_processor_kwargs,
        )


def _nonnegative_setting(config: Any, path: str, default: float) -> float:
    value = float(OmegaConf.select(config, path, default=default))
    if value < 0.0 or not isfinite(value):
        raise ValueError(f"{path} must be a finite non-negative number.")
    return value


def _checkpoint(agent_data: AgentData) -> _ProbeCheckpoint:
    return _ProbeCheckpoint(
        prompt_ids=list(agent_data.prompt_ids),
        image_data=_copy_modal_data(agent_data.image_data),
        video_data=_copy_modal_data(agent_data.video_data),
        audio_data=_copy_modal_data(agent_data.audio_data),
        mm_processor_kwargs=dict(agent_data.mm_processor_kwargs or {}),
    )


def _copy_modal_data(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def _sample_split(extra_info: Any) -> str:
    extra_info = _unwrap_scalar(extra_info)
    if not isinstance(extra_info, Mapping):
        return ""
    return str(_unwrap_scalar(extra_info.get("split")) or "").strip().casefold()


def _probe_reward_context(config: Any, sample_kwargs: dict[str, Any]):
    reward_fn = _load_probe_reward_fn(config)
    data_source = _unwrap_scalar(sample_kwargs.get("data_source"))
    reward_model = _unwrap_scalar(sample_kwargs.get("reward_model"))
    if not isinstance(reward_model, Mapping) or "ground_truth" not in reward_model:
        raise _MarginalUtilityContractError(
            "Marginal-utility probes require reward_model.ground_truth."
        )
    extra_info = _unwrap_scalar(sample_kwargs.get("extra_info"))
    if extra_info is None:
        extra_info = {}
    if not isinstance(extra_info, Mapping):
        raise _MarginalUtilityContractError(
            "Marginal-utility probes require mapping-valued extra_info."
        )
    return reward_fn, data_source, reward_model["ground_truth"], dict(extra_info)


def _load_probe_reward_fn(config: Any):
    reward_config = OmegaConf.select(config, "reward.custom_reward_function", default=None)
    if reward_config is None:
        raise _MarginalUtilityContractError(
            "Marginal-utility probes require reward.custom_reward_function."
        )
    is_config = getattr(OmegaConf, "is_config", lambda _: False)
    resolved = (
        OmegaConf.to_container(reward_config, resolve=True)
        if is_config(reward_config)
        else reward_config
    )
    cache_key = json.dumps(resolved, sort_keys=True, default=str)
    if cache_key not in _REWARD_FUNCTION_CACHE:
        from verl.trainer.ppo.reward import get_custom_reward_fn

        reward_fn = get_custom_reward_fn(config)
        if reward_fn is None:
            raise _MarginalUtilityContractError(
                "Unable to load the configured custom reward function."
            )
        _REWARD_FUNCTION_CACHE[cache_key] = reward_fn
    return _REWARD_FUNCTION_CACHE[cache_key]


async def _invoke_reward(reward_fn, **kwargs) -> dict[str, Any]:
    result = reward_fn(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, Mapping):
        raise _MarginalUtilityContractError(
            "Marginal-utility probes require a mapping-valued custom reward."
        )
    return dict(result)


def _numeric_reward_field(reward: dict[str, Any], key: str) -> float:
    value = reward.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _MarginalUtilityContractError(
            f"Custom reward field {key!r} must be numeric."
        )
    value = float(value)
    if not isfinite(value):
        raise _MarginalUtilityContractError(
            f"Custom reward field {key!r} must be finite."
        )
    return value


def _tagged_probe_solution(generated_text: str) -> str:
    content = re.split(r"</answer>", str(generated_text), maxsplit=1, flags=re.IGNORECASE)[0]
    content = re.sub(r"</?answer>", "", content, flags=re.IGNORECASE)
    return f"{_ANSWER_OPEN_TAG}{content.strip()}{_ANSWER_CLOSE_TAG}"


def _unwrap_scalar(value: Any) -> Any:
    if hasattr(value, "shape") and getattr(value, "shape", None) == () and hasattr(value, "item"):
        return value.item()
    return value
