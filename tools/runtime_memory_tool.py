"""MemPilot's dual-view memory actions, implemented as stateful verl tools.

``retrieve_memory`` implements RETRIEVE over the query-agnostic memory bank M.
``runtime_memory`` implements CURATE over raw multimodal history H, delegating
query-specific processing to a policy-selected LLM/VLM. The two views have
independent retrieval corpora and return textual observations to the policy.
"""

from __future__ import annotations

import json
import logging
import os
import re
from math import isfinite
from typing import Any, Optional
from uuid import uuid4

from agent_loops.route_schema import append_model_cards
from memory_access import LOSSY_PREVIEW_FIELD
from tools.dynamic_memory_retriever import get_dynamic_memory_retriever
from tools.llm_api_client import (
    EXTERNAL_CALL_FAILED,
    call_openai_compatible_chat,
    validate_api_key_file,
)
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)


logger = logging.getLogger(__name__)
INVALID_EXTERNAL_RESULT = "External tool returned an invalid result."
DEFAULT_INSTRUCTION_MAX_CHARS = 512


def _runtime_memory_function_description() -> str:
    return (
        "Execute one atomic memory-processing instruction with a selected model from the configured routing "
        "pool. The requested number of relevant raw memory chunks is retrieved by semantic similarity to the "
        "retrieval query, up to the configured limit, and the complete uncompressed content of those chunks is "
        "supplied to the selected model. Question-image descriptions are always supplied as text. When "
        "include_images is true, applicable retrieved and question-image pixels are attached if the selected "
        "model supports image input. The selected model receives only the instruction, question-image "
        "descriptions, retrieved chunks, and applicable images; it does not receive the original user request or "
        "retrieval query. "
        "It returns the instruction result as text."
    )


def _retrieve_memory_function_description() -> str:
    selection = (
        "The requested number of items is selected by semantic similarity to the retrieval query, up to the "
        "configured limit. "
    )
    return (
        "Retrieve precomputed query-agnostic compressed memory and return only the compressed observation text, "
        "without memory IDs or metadata and without calling an external model. "
        f"{selection}Retrieval ranking and returned text both use the compressed-memory corpus. Visual content is "
        "represented only by stored textual captions; image pixels are never attached."
    )


def _retrieval_query_property(instruction_max_chars: int) -> OpenAIFunctionPropertySchema:
    return OpenAIFunctionPropertySchema(
        type="string",
        description=(
            "Concise, self-contained semantic search query describing the evidence needed for this call, "
            f"limited to {instruction_max_chars} characters. Preserve relevant entities, relations, "
            "time constraints, and question-image details from the user request, and incorporate useful "
            "clues from earlier memory-tool results when applicable. Do not describe how to process "
            "the evidence or include a reasoning result, candidate answer, choice list, or <answer> tag."
        ),
    )


def _evidence_count_property(
    *,
    evidence_kind: str,
    maximum: int,
    allow_zero: bool,
) -> OpenAIFunctionPropertySchema:
    return OpenAIFunctionPropertySchema(
        type="integer",
        description=(
            f"Number of {evidence_kind} requested for this retrieval query. "
            + (
                "Choose a non-negative integer. "
                if allow_zero
                else "Choose a positive integer. "
            )
            + f"Maximum memory items per call: {maximum}. "
            + (
                "Use 0 only when the external instruction needs no retrieved memory, such as a "
                "question-image-only instruction; set include_images to true when image pixels are needed."
                if allow_zero
                else ""
            )
        ),
    )


def _common_retrieval_properties(
    *,
    instruction_max_chars: int,
    evidence_kind: str,
    maximum: int,
    allow_zero: bool,
) -> tuple[dict[str, OpenAIFunctionPropertySchema], list[str]]:
    properties = {"retrieval_query": _retrieval_query_property(instruction_max_chars)}
    required = list(properties)
    properties["evidence_count"] = _evidence_count_property(
        evidence_kind=evidence_kind,
        maximum=maximum,
        allow_zero=allow_zero,
    )
    required.append("evidence_count")
    return properties, required


class RuntimeMemoryTool(BaseTool):
    """Implement CURATE with factorized retrieval and delegation controls.

    In the dynamic retrieval interface, ``retrieval_query`` (r_t) and
    ``evidence_count`` (k_t) select raw evidence; ``instruction`` (i_t),
    ``model`` (m_t), and ``include_images`` (v_t) control its curation.
    The retrieval query is not separately forwarded to the selected model.
    """

    returns_compressed_memory = False

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        self.config = config
        self._instances: dict[str, dict[str, Any]] = {}
        self.model_specs = _load_model_specs(config)
        self.model_by_name = {spec["name"]: spec for spec in self.model_specs}
        self.instruction_max_chars = (config or {}).get(
            "instruction_max_chars",
            DEFAULT_INSTRUCTION_MAX_CHARS,
        )
        self.dynamic_retriever = get_dynamic_memory_retriever(
            (config or {}).get("dynamic_retrieval")
        )
        if not self.dynamic_retriever.enabled:
            raise ValueError(
                "Runtime memory access requires DYNAMIC_RETRIEVAL_TOP_K to be greater than zero."
            )
        super().__init__(config=config, tool_schema=tool_schema or self.get_openai_tool_schema())

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        model_names = [spec["name"] for spec in self.model_specs]
        function_description = _runtime_memory_function_description()
        properties, required = _common_retrieval_properties(
            instruction_max_chars=self.instruction_max_chars,
            evidence_kind="original memory chunks",
            maximum=self.dynamic_retriever.top_k,
            allow_zero=True,
        )
        instruction_description = (
            "Open-ended, self-contained atomic instruction telling the external model how to process the "
            f"supplied evidence, limited to {self.instruction_max_chars} characters. "
        )
        properties.update(
            {
            "instruction": OpenAIFunctionPropertySchema(
                type="string",
                description=(
                    instruction_description + "Describe the required "
                    "extraction, comparison, normalization, or organization and desired result, not your reasoning "
                    "result, a candidate answer, the choice list, or an <answer> tag."
                ),
            ),
            "model": OpenAIFunctionPropertySchema(
                type="string",
                description=append_model_cards(
                    "Route this instruction to exactly one model ID. Model IDs are unordered action labels, so enum "
                    "position is not a preference. The cards expose only supported input modalities and parameter "
                    "count; performance must be learned from outcomes.",
                    self.model_specs,
                ),
                enum=model_names,
            ),
            }
        )
        required.extend(("instruction", "model"))
        properties["include_images"] = OpenAIFunctionPropertySchema(
            type="boolean",
            description=(
                "Whether this instruction requires original image pixels from the retrieved memory items or "
                "the question. When true, applicable images are attached only if the selected model supports "
                "image input; text-only models ignore the request. Omit or set false to provide captions "
                "without image pixels."
            ),
        )
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="runtime_memory",
                description=function_description,
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties=properties,
                    required=required,
                ),
                strict=False,
            ),
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        memory_bank = create_kwargs.get("memory_bank", [])
        query_images = create_kwargs.get("query_images", create_kwargs.get("image_inputs", []))
        metadata = create_kwargs.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        normalized_memory_bank = _normalize_retrieved_memory(memory_bank)
        unavailable_memory_ids: list[str] = []
        if self.returns_compressed_memory:
            if create_kwargs.get("compressed_memory_bank") is not None:
                # External builders supply the same independent compressed corpus.
                # Format adaptation and document embedding happen offline.
                retrieval_bank = _normalize_compressed_memory_bank(
                    create_kwargs["compressed_memory_bank"]
                )
                has_complete_memory_bank = True
            else:
                # Missing/null means the prepared LLMLingua bank. An explicit []
                # remains an empty external bank; the two corpora are never merged.
                retrieval_bank, unavailable_memory_ids = _build_compressed_memory_corpus(
                    normalized_memory_bank
                )
                has_complete_memory_bank = (
                    "memory_bank" in create_kwargs and not unavailable_memory_ids
                )
            # Raw and compressed corpora must never share document embeddings.
            retrieval_index = create_kwargs.get("lossy_memory_index")
            if has_complete_memory_bank and retrieval_bank and not retrieval_index:
                raise ValueError(
                    "Dynamic retrieve_memory requires a precomputed lossy_memory_index. "
                    "Regenerate the data or run the external-memory converter."
                )
        else:
            retrieval_bank = normalized_memory_bank
            retrieval_index = create_kwargs.get("memory_index")
            has_complete_memory_bank = "memory_bank" in create_kwargs
        instance_id = instance_id or str(uuid4())
        self._instances[instance_id] = {
            "memory_bank": retrieval_bank,
            "memory_bank_cache_key": self.dynamic_retriever.build_document_cache_key(
                retrieval_bank,
                metadata,
            ),
            "has_complete_memory_bank": has_complete_memory_bank,
            "unavailable_memory_ids": unavailable_memory_ids,
            "memory_index": retrieval_index,
            "query_images": _normalize_image_records(
                query_images,
                default_prefix="query_image",
                source_memory_id="",
            ),
            "metadata": metadata,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: Any, **kwargs) -> tuple[ToolResponse, float, dict]:
        return await self._execute(
            instance_id,
            parameters,
            direct_memory_return=False,
            **kwargs,
        )

    async def _execute(
        self,
        instance_id: str,
        parameters: Any,
        *,
        direct_memory_return: bool,
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        tool_name = "retrieve_memory" if direct_memory_return else "runtime_memory"
        external_call_attempted = not direct_memory_return
        if not isinstance(parameters, dict):
            return _tool_result(
                "Tool arguments must be a JSON object.",
                0.0,
                _error_metrics(
                    "",
                    "",
                    "",
                    tool_name=tool_name,
                ),
                agent_data,
            )

        state = self._instances.get(instance_id, {})
        retrieval_query = str(parameters.get("retrieval_query", "")).strip()
        instruction = str(parameters.get("instruction", "")).strip()
        model_name = str(parameters.get("model", "")).strip()

        def error_metrics() -> dict[str, Any]:
            return _error_metrics(
                retrieval_query,
                instruction,
                model_name,
                tool_name=tool_name,
            )

        unsupported_parameters = {"expand_sources", "memory_ids", "image_ids"}
        if direct_memory_return:
            unsupported_parameters.update(("instruction", "model", "include_images", "image_ids"))
        supplied_unsupported_parameters = sorted(unsupported_parameters & parameters.keys())
        if supplied_unsupported_parameters:
            message = (
                f"Unsupported tool parameter(s) for {tool_name}: "
                f"{', '.join(supplied_unsupported_parameters)}."
            )
            return _tool_result(
                message,
                0.0,
                error_metrics(),
                agent_data,
            )
        required_values = [("retrieval_query", retrieval_query)]
        if not direct_memory_return:
            required_values.extend((('instruction', instruction), ('model', model_name)))
        missing_parameters = [name for name, value in required_values if not value]
        if "evidence_count" not in parameters:
            missing_parameters.append("evidence_count")
        if missing_parameters:
            message = f"Missing required tool parameter(s): {', '.join(missing_parameters)}."
            return _tool_result(message, 0.0, error_metrics(), agent_data)

        try:
            evidence_count = _parse_nonnegative_integer(parameters["evidence_count"], "evidence_count")
            include_images = (
                _parse_boolean(parameters.get("include_images", False), "include_images")
                if not direct_memory_return
                else False
            )
        except ValueError as exc:
            return _tool_result(
                str(exc),
                0.0,
                error_metrics(),
                agent_data,
            )
        if direct_memory_return and evidence_count == 0:
            return _tool_result(
                "retrieve_memory requires evidence_count to be a positive integer.",
                0.0,
                error_metrics(),
                agent_data,
            )
        effective_evidence_count = self.dynamic_retriever.resolve_top_k(evidence_count)

        if len(retrieval_query) > self.instruction_max_chars:
            message = (
                f"retrieval_query exceeds the {self.instruction_max_chars}-character limit "
                f"({len(retrieval_query)} characters supplied)."
            )
            return _tool_result(message, 0.0, error_metrics(), agent_data)
        if not direct_memory_return and len(instruction) > self.instruction_max_chars:
            message = (
                f"instruction exceeds the {self.instruction_max_chars}-character limit "
                f"({len(instruction)} characters supplied)."
            )
            return _tool_result(message, 0.0, error_metrics(), agent_data)
        if re.search(r"</?answer\b", retrieval_query, flags=re.IGNORECASE):
            message = "retrieval_query must describe evidence to find and cannot contain answer tags."
            return _tool_result(message, 0.0, error_metrics(), agent_data)
        if not direct_memory_return and re.search(r"</?answer\b", instruction, flags=re.IGNORECASE):
            message = "instruction must specify memory processing and cannot contain answer tags."
            return _tool_result(message, 0.0, error_metrics(), agent_data)
        model_spec = None if direct_memory_return else self.model_by_name.get(model_name)
        if not direct_memory_return and model_spec is None:
            message = f"Unsupported model: {model_name}. Available models: {list(self.model_by_name)}"
            return _tool_result(message, 0.0, error_metrics(), agent_data)

        if effective_evidence_count > 0 and not state.get("has_complete_memory_bank"):
            unavailable_ids = state.get("unavailable_memory_ids", [])
            if direct_memory_return and unavailable_ids:
                message = _missing_compressed_memory_message(unavailable_ids)
            elif direct_memory_return:
                message = (
                    "retrieve_memory requires a complete compressed-memory corpus. "
                    "Regenerate the data or use runtime_memory for original evidence."
                )
            else:
                message = "Dynamic memory retrieval requires regenerated data with a complete memory_bank."
            logger.error("[RUNTIME MEMORY RETRIEVAL ERROR] %s", message)
            return _tool_result(message, 0.0, error_metrics(), agent_data)

        try:
            dynamic_entries = await self.dynamic_retriever.retrieve(
                state.get("memory_bank", []),
                retrieval_query,
                memory_index=state.get("memory_index"),
                document_cache_key=state.get("memory_bank_cache_key", ""),
                requested_top_k=evidence_count,
            )
        except Exception as exc:
            logger.exception(
                "[RUNTIME MEMORY RETRIEVAL ERROR] retrieval-query-based retrieval failed: %s",
                type(exc).__name__,
            )
            message = f"Dynamic memory retrieval failed: {type(exc).__name__}."
            return _tool_result(message, 0.0, error_metrics(), agent_data)
        entries = dynamic_entries
        backend_entries = entries
        compressed_memory_ids = (
            [str(entry.get("id", "")) for entry in entries]
            if direct_memory_return
            else []
        )

        query_images = state.get("query_images", [])
        dynamic_images: list[dict[str, Any]] = []
        selected_images: list[dict[str, Any]] = []
        if external_call_attempted and model_spec.get("kind") == "vlm" and include_images:
            dynamic_images = _available_images(dynamic_entries, [])
            selected_images = _deduplicate_images(
                [*dynamic_images, *(dict(image) for image in query_images)]
            )
        num_dynamic_input_images = sum(
            not _is_query_image(image)
            and any(_images_share_source(image, dynamic_image) for dynamic_image in dynamic_images)
            for image in selected_images
        )

        if direct_memory_return:
            raw_result = _build_direct_memory_result(backend_entries)
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        else:
            backend_input = _build_backend_input(
                instruction=instruction,
                query_images=query_images,
                retrieved_memory=backend_entries,
                selected_images=selected_images,
            )
            raw_result, usage = await call_openai_compatible_chat(
                config=self.config,
                model_spec=model_spec,
                backend_input=backend_input,
                image_inputs=selected_images or None,
            )

        api_cost = _api_cost(model_spec=model_spec, usage=usage) if external_call_attempted else 0.0
        estimated_latency = (
            0.0
            if not external_call_attempted
            else _estimated_latency(
                model_spec=model_spec,
                usage=usage,
                num_images=len(selected_images),
            )
        )
        latency_stage = _latency_stage(agent_data)
        pricing = (
            {"input_per_million_usd": 0.0, "output_per_million_usd": 0.0}
            if not external_call_attempted
            else model_spec["pricing"]
        )
        api_call_succeeded = external_call_attempted and raw_result != EXTERNAL_CALL_FAILED
        if not external_call_attempted:
            result, result_valid = raw_result, True
        elif api_call_succeeded:
            result, result_valid = _normalize_external_result(raw_result)
        else:
            result, result_valid = raw_result, False
        call_succeeded = result_valid and (not external_call_attempted or api_call_succeeded)
        external_call_succeeded = api_call_succeeded and result_valid
        dynamic_memory_ids = [str(entry.get("id", "")) for entry in dynamic_entries]
        selected_image_ids = [str(image.get("id", "")) for image in selected_images]
        common_call_fields = {
            "tool_name": tool_name,
            "retrieval_query": retrieval_query,
            "instruction": instruction,
            "model": model_name,
            "provider": model_spec.get("provider", "") if external_call_attempted else "",
            "backend_model": model_spec.get("model", "") if external_call_attempted else "",
            "dynamic_memory_ids": dynamic_memory_ids,
            "evidence_count": evidence_count,
            "effective_evidence_count": effective_evidence_count,
            "dynamic_retrieval_max_top_k": self.dynamic_retriever.top_k,
            "include_images": include_images,
            "memory_evidence_kind": "compressed_memory" if direct_memory_return else "original_chunk",
            "compressed_memory_ids": compressed_memory_ids,
            "num_dynamic_input_images": num_dynamic_input_images,
            "direct_memory_return": direct_memory_return,
            "memory_call_succeeded": call_succeeded,
            "external_call_attempted": external_call_attempted,
            "external_call_succeeded": external_call_succeeded,
            "external_api_succeeded": api_call_succeeded,
        }
        metrics = {
            **common_call_fields,
            "memory_api_cost": api_cost,
            "memory_estimated_latency": estimated_latency,
            "latency_stage": latency_stage,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "input_price_per_million_usd": pricing["input_per_million_usd"],
            "output_price_per_million_usd": pricing["output_per_million_usd"],
            "num_input_memory_items": len(backend_entries),
            "num_compressed_memory_items": len(compressed_memory_ids),
            "num_input_images": len(selected_images),
            "image_ids": selected_image_ids,
            "model_kind": model_spec.get("kind", "unknown") if external_call_attempted else "none",
        }

        # Raw token-price cost stays in verl's reward/diagnostic side channels;
        # it is not appended to the textual evidence visible to the policy.
        return _tool_result(result, -api_cost, metrics, agent_data)

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class RetrieveMemoryTool(RuntimeMemoryTool):
    """Implement RETRIEVE by returning query-agnostic compressed memory directly.

    Visual evidence is limited to stored captions, not image pixels. This
    action incurs no delegated-model cost or latency; orchestrator generation
    is still accounted for separately by the agent loop.
    """

    returns_compressed_memory = True

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        properties, required = _common_retrieval_properties(
            instruction_max_chars=self.instruction_max_chars,
            evidence_kind="compressed memory items",
            maximum=self.dynamic_retriever.top_k,
            allow_zero=False,
        )
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="retrieve_memory",
                description=_retrieve_memory_function_description(),
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties=properties,
                    required=required,
                ),
                strict=False,
            ),
        )

    async def execute(self, instance_id: str, parameters: Any, **kwargs) -> tuple[ToolResponse, float, dict]:
        return await self._execute(
            instance_id,
            parameters,
            direct_memory_return=True,
            **kwargs,
        )


def _load_model_specs(config: dict) -> list[dict[str, Any]]:
    raw_models = (config or {}).get("model_pool") or []
    provider_configs = (config or {}).get("providers", {}) or {}
    if not isinstance(raw_models, list):
        raise ValueError("RuntimeMemoryTool config.model_pool must be a list of route mappings.")
    if not isinstance(provider_configs, dict):
        raise ValueError("RuntimeMemoryTool config.providers must be a mapping.")

    specs: list[dict[str, Any]] = []
    route_names: set[str] = set()
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise ValueError(f"model_pool[{index}] must be a route mapping.")
        provider_name = str(raw.get("provider") or "").strip()
        provider_spec = provider_configs.get(provider_name, {}) if provider_name else {}
        if provider_name and provider_configs and provider_name not in provider_configs:
            raise ValueError(f"Unknown provider {provider_name!r} for model_pool[{index}].")
        if not isinstance(provider_spec, dict):
            raise ValueError(f"Provider {provider_name!r} must be a mapping.")
        spec = {**provider_spec, **raw}
        spec["api_base"] = (
            os.environ.get(str(spec.get("api_base_env") or "")) or spec.get("api_base")
        )
        spec["provider"] = provider_name
        spec["name"] = str(spec.get("name") or "").strip()
        if not spec["name"]:
            raise ValueError(f"model_pool[{index}] requires a non-empty name.")
        if spec["name"] in route_names:
            raise ValueError(f"Duplicate runtime-memory route name: {spec['name']!r}.")
        route_names.add(spec["name"])

        spec["api_type"] = str(spec.get("api_type") or "openai_compatible").strip().lower()
        if spec["api_type"] != "openai_compatible":
            raise ValueError(
                f"Unsupported api_type {spec['api_type']!r} for route {spec['name']!r}; "
                "only openai_compatible is implemented."
            )
        if not str(spec.get("api_base") or "").strip():
            raise ValueError(f"Route {spec['name']!r} requires api_base.")
        if not str(spec.get("model") or "").strip():
            raise ValueError(f"Route {spec['name']!r} requires model.")
        validate_api_key_file(spec)

        spec["kind"] = str(spec.get("kind") or "").strip().lower()
        if spec["kind"] not in {"llm", "vlm"}:
            raise ValueError(f"Route {spec['name']!r} requires kind=llm or kind=vlm.")
        spec["parameter_count"] = str(spec.get("parameter_count") or "").strip()
        if not spec["parameter_count"] or any(
            character in spec["parameter_count"] for character in "\r\n;"
        ):
            raise ValueError(
                f"Route {spec['name']!r} requires a one-line parameter_count such as '9B' "
                "or '109B total, 17B active'."
            )
        spec["pricing"] = _normalize_model_pricing(
            spec.get("pricing"),
            route_name=spec["name"],
        )
        spec["latency"] = _normalize_latency_profile(
            spec.get("latency"),
            route_name=spec["name"],
        )
        specs.append(spec)
    if not specs:
        raise ValueError("RuntimeMemoryTool requires config.model_pool to define the routing model pool.")
    return specs


def _build_backend_input(
    instruction: str,
    query_images: list[dict[str, Any]],
    retrieved_memory: list[dict[str, Any]],
    selected_images: list[dict[str, Any]] | None = None,
) -> str:
    """Build the delegation prompt from the instruction and selected evidence.

    Search uses a separate retrieval query. Neither that query nor the policy
    transcript is independently added here; image pixels, when enabled, are
    attached separately by the API client.
    """

    sections = [f"Instruction:\n{instruction}"]
    normalized_query_images = _deduplicate_images(query_images)
    if normalized_query_images:
        query_image_text = "\n".join(
            _format_backend_image(image) for image in normalized_query_images
        )
        sections.append(f"Question image descriptions:\n{query_image_text}")
    memory_text = "\n".join(_format_backend_memory_item(item) for item in retrieved_memory)
    sections.append(f"Memory items:\n{memory_text or 'No memory items.'}")
    backend_input = "\n\n".join(sections)
    if selected_images:
        query_image_ids = {str(image.get("id") or "") for image in normalized_query_images}
        selected_memory_images = [
            image for image in selected_images if str(image.get("id") or "") not in query_image_ids
        ]
        if selected_memory_images:
            image_text = "\n".join(
                _format_backend_image(image) for image in selected_memory_images
            )
            backend_input += f"\n\nAdditional image inputs:\n{image_text}"
    return backend_input


def _build_direct_memory_result(retrieved_memory: list[dict[str, Any]]) -> str:
    """Serialize selected compressed evidence directly into a tool response."""

    observations = [_normalize_multiline_text(item.get("text")) for item in retrieved_memory]
    return "\n\n".join(text for text in observations if text) or "No memory items."


def _normalize_external_result(raw: str) -> tuple[str, bool]:
    """Normalize a direct external-model result while blocking answer-tag injection."""
    text = str(raw or "").strip()
    if not text or re.search(r"</?answer\b", text, flags=re.IGNORECASE):
        return INVALID_EXTERNAL_RESULT, False
    return text, True


def _format_backend_memory_item(item: dict[str, Any]) -> str:
    item_id = _one_line_text(item.get("id"))
    speaker = _one_line_text(item.get("speaker"))
    time = _one_line_text(item.get("time"))
    text = _normalize_multiline_text(item.get("text"))
    attributes = [f"id={item_id}"]
    if speaker:
        attributes.append(f"speaker={speaker}")
    if time:
        attributes.append(f"time={time}")
    indented_text = text.replace("\n", "\n  ")
    return f"- {' '.join(attributes)}:\n  {indented_text}"


def _format_backend_image(image: dict[str, Any]) -> str:
    image_id = _one_line_text(image.get("id"))
    source_memory_id = _one_line_text(image.get("source_memory_id"))
    caption = _normalize_multiline_text(image.get("caption"))
    attributes = [f"image_id={image_id}"]
    if source_memory_id:
        attributes.append(f"memory_id={source_memory_id}")
    line = f"- {' '.join(attributes)}"
    return f"{line}\n  caption: {caption}" if caption else line


def _normalize_retrieved_memory(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [{"text": raw}]
    if isinstance(raw, dict):
        raw = raw.get("memory") or raw.get("turns") or raw.get("messages") or raw.get("sessions") or [raw]
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raw = [raw]

    entries: list[dict[str, Any]] = []
    for item in raw:
        _flatten_memory_item(item, entries)
    for idx, entry in enumerate(entries):
        if not entry.get("id"):
            entry["id"] = f"m{idx:04d}"
        entry["text"] = _normalize_multiline_text(entry.get("text"))
        if LOSSY_PREVIEW_FIELD in entry:
            entry[LOSSY_PREVIEW_FIELD] = _normalize_multiline_text(
                entry.get(LOSSY_PREVIEW_FIELD)
            )
        entry["images"] = _normalize_image_records(
            entry.get("images", []),
            default_prefix=f"{entry['id']}_image",
            source_memory_id=str(entry["id"]),
        )
    return entries







def _normalize_compressed_memory_bank(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("compressed_memory_bank must be a list of id/text records.")
    entries = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Compressed memory items must be id/text mappings.")
        item_id = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not item_id or item_id in seen or not text:
            raise ValueError("Compressed memory requires unique non-empty IDs and non-empty text.")
        seen.add(item_id)
        entries.append({"id": item_id, "text": text})
    return entries


def _flatten_memory_item(item: Any, entries: list[dict[str, Any]], inherited_time: str = "") -> None:
    if isinstance(item, str):
        entries.append({"text": item, "time": inherited_time})
        return
    if not isinstance(item, dict):
        entries.append({"text": str(item), "time": inherited_time})
        return

    item_time = str(item.get("time") or item.get("timestamp") or item.get("date") or inherited_time)
    nested = item.get("turns") or item.get("messages") or item.get("conversation")
    if isinstance(nested, list):
        for child in nested:
            _flatten_memory_item(child, entries, inherited_time=item_time)
        return

    text_parts = []
    for key in ("text", "content", "utterance", "message", "caption", "image_caption"):
        value = item.get(key)
        if value:
            text_parts.append(str(value))
    speaker = item.get("speaker") or item.get("role") or item.get("from")
    text = "\n".join(text_parts)

    entry = {
        "id": str(item.get("id") or item.get("turn_id") or item.get("message_id") or ""),
        "speaker": str(speaker or ""),
        "text": text,
        "time": item_time,
        "images": item.get("images") or item.get("image_inputs") or [],
    }
    if LOSSY_PREVIEW_FIELD in item:
        entry[LOSSY_PREVIEW_FIELD] = item.get(LOSSY_PREVIEW_FIELD)
    if entry["text"]:
        entries.append(entry)


def _build_compressed_memory_corpus(
    memory_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build an independently searchable compressed corpus without raw fallback."""

    corpus = []
    missing_ids = []
    for item in memory_entries:
        item_id = str(item.get("id") or "")
        lossy = _normalize_multiline_text(item.get(LOSSY_PREVIEW_FIELD))
        if not lossy:
            missing_ids.append(item_id)
            continue
        view = dict(item)
        view["text"] = lossy
        view.pop(LOSSY_PREVIEW_FIELD, None)
        corpus.append(view)
    return corpus, missing_ids


def _missing_compressed_memory_message(memory_ids: list[str]) -> str:
    return (
        "retrieve_memory requires a complete precomputed compressed-memory corpus; "
        f"missing for memory IDs: {sorted(memory_ids)}. Regenerate compressed memory data or use "
        "runtime_memory for original evidence."
    )


def _normalize_image_records(
    raw: Any,
    *,
    default_prefix: str,
    source_memory_id: str,
) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = raw
        if isinstance(decoded, (list, dict)):
            raw = decoded
    records: list[dict[str, Any]] = []
    for index, value in enumerate(_as_list(raw)):
        if isinstance(value, dict):
            record = dict(value)
        elif value is None or not str(value).strip():
            continue
        else:
            record = {"path": str(value).strip()}
        image_id = _one_line_text(record.get("id") or record.get("image_id"))
        record["id"] = image_id or f"{default_prefix}_{index:04d}"
        record["caption"] = _normalize_multiline_text(
            record.get("caption") or record.get("image_caption")
        )
        record["source_memory_id"] = str(
            record.get("source_memory_id") or source_memory_id or ""
        )
        records.append(record)
    return records


def _available_images(
    memory_entries: list[dict[str, Any]],
    query_images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    images = [dict(image) for entry in memory_entries for image in entry.get("images", [])]
    images.extend(dict(image) for image in query_images)
    return images


def _deduplicate_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate aliases of the same pixels while preferring query-image metadata."""

    selected: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}
    index_by_source: dict[tuple[str, ...], int] = {}
    for image in images:
        image_id = str(image.get("id") or "")
        source_keys = _image_source_keys(image)
        existing_index = index_by_id.get(image_id) if image_id else None
        if existing_index is None:
            existing_index = next(
                (index_by_source[key] for key in source_keys if key in index_by_source),
                None,
            )
        if existing_index is not None:
            existing = selected[existing_index]
            if _is_query_image(image) and not _is_query_image(existing):
                existing_id = str(existing.get("id") or "")
                if existing_id and index_by_id.get(existing_id) == existing_index:
                    del index_by_id[existing_id]
                for key in _image_source_keys(existing):
                    if index_by_source.get(key) == existing_index:
                        del index_by_source[key]
                selected[existing_index] = image
                if image_id:
                    index_by_id[image_id] = existing_index
                for key in source_keys:
                    index_by_source[key] = existing_index
            continue
        index = len(selected)
        selected.append(image)
        if image_id:
            index_by_id[image_id] = index
        for key in source_keys:
            index_by_source[key] = index
    return selected


def _images_share_source(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_id = str(first.get("id") or "")
    second_id = str(second.get("id") or "")
    if first_id and first_id == second_id:
        return True
    return bool(set(_image_source_keys(first)) & set(_image_source_keys(second)))


def _image_source_keys(image: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    keys: list[tuple[str, ...]] = []
    hf_filename = _normalized_image_location(image.get("hf_filename"))
    if hf_filename:
        keys.append(
            (
                "hf",
                str(image.get("hf_repo_type") or "dataset").strip(),
                str(image.get("hf_repo_id") or "").strip(),
                str(image.get("hf_revision") or "").strip(),
                hf_filename,
            )
        )
    path = _normalized_image_location(image.get("path"))
    if path:
        keys.append(("path", path))

    image_url = image.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    for value in (image_url, image.get("url"), image.get("data_url")):
        location = str(value or "").strip()
        if location:
            keys.append(("url", location))
    return tuple(dict.fromkeys(keys))


def _normalized_image_location(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").removeprefix("./")


def _is_query_image(image: dict[str, Any]) -> bool:
    image_id = str(image.get("id") or "").strip().lower()
    return (
        str(image.get("source") or "").strip().lower() == "query"
        or image_id == "query_image"
        or image_id.startswith("query_image_")
    )


def _one_line_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_multiline_text(text: Any) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _parse_boolean(raw: Any, name: str = "include_images") -> bool:
    if isinstance(raw, bool):
        return raw
    if raw in (0, 1):
        return bool(raw)
    normalized = str(raw).strip().lower()
    if normalized in {"true", "yes", "on"}:
        return True
    if normalized in {"false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean.")


def _parse_nonnegative_integer(raw: Any, name: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    if isinstance(raw, int):
        value = raw
    else:
        normalized = str(raw).strip()
        if not re.fullmatch(r"\+?\d+", normalized):
            raise ValueError(f"{name} must be a non-negative integer.")
        value = int(normalized)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _normalize_model_pricing(raw: Any, route_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Route {route_name!r} requires an inline pricing mapping.")
    try:
        input_price = float(raw["input_per_million_usd"])
        output_price = float(raw["output_per_million_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid pricing for route {route_name!r}; input_per_million_usd and "
            "output_per_million_usd must be numeric."
        ) from exc
    if input_price < 0 or output_price < 0:
        raise ValueError(f"Pricing for route {route_name!r} cannot be negative.")
    pricing = dict(raw)
    pricing["input_per_million_usd"] = input_price
    pricing["output_per_million_usd"] = output_price
    return pricing


def _api_cost(model_spec: dict[str, Any], usage: dict[str, int | float]) -> float:
    """Return delegated-call cost on the training raw-cost axis (USD times 1e6)."""

    if "cost_usd" in usage:
        # Keep provider-reported USD cost on the same unscaled
        # token * USD-per-million axis as static pricing.
        return max(float(usage["cost_usd"]), 0.0) * 1_000_000.0
    pricing = model_spec["pricing"]
    input_cost = usage["prompt_tokens"] * pricing["input_per_million_usd"]
    output_cost = usage["completion_tokens"] * pricing["output_per_million_usd"]
    return input_cost + output_cost


def _normalize_latency_profile(raw: Any, route_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Route {route_name!r} requires an inline latency mapping.")
    required = (
        "base_seconds",
        "input_seconds_per_token",
        "output_seconds_per_token",
    )
    try:
        values = {key: float(raw[key]) for key in required}
        values["image_seconds"] = float(raw.get("image_seconds", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid latency profile for route {route_name!r}; base and per-token "
            "coefficients must be numeric."
        ) from exc
    if any(not isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError(
            f"Latency coefficients for route {route_name!r} must be finite and non-negative."
        )
    return {**raw, **values}


def _estimated_latency(
    *,
    model_spec: dict[str, Any],
    usage: dict[str, int | float],
    num_images: int,
) -> float:
    """Estimate seconds with an affine token/image proxy, not wall-clock timing."""

    profile = model_spec["latency"]
    return (
        profile["base_seconds"]
        + max(float(usage.get("prompt_tokens", 0)), 0.0)
        * profile["input_seconds_per_token"]
        + max(float(usage.get("completion_tokens", 0)), 0.0)
        * profile["output_seconds_per_token"]
        + max(int(num_images), 0) * profile["image_seconds"]
    )


def _error_metrics(
    retrieval_query: str,
    instruction: str,
    model_name: str,
    tool_name: str = "runtime_memory",
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "memory_api_cost": 0.0,
        "memory_estimated_latency": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "num_input_memory_items": 0,
        "dynamic_memory_ids": [],
        "evidence_count": 0,
        "effective_evidence_count": 0,
        "dynamic_retrieval_max_top_k": 0,
        "include_images": False,
        "memory_evidence_kind": "",
        "num_compressed_memory_items": 0,
        "compressed_memory_ids": [],
        "num_input_images": 0,
        "num_dynamic_input_images": 0,
        "image_ids": [],
        "retrieval_query": retrieval_query,
        "instruction": instruction,
        "model": model_name,
        "direct_memory_return": tool_name == "retrieve_memory",
        "memory_call_succeeded": False,
        "external_call_attempted": False,
        "external_call_succeeded": False,
        "external_api_succeeded": False,
        "error": True,
    }


def _tool_result(
    text: str,
    reward: float,
    metrics: dict[str, Any],
    agent_data: Any,
) -> tuple[ToolResponse, float, dict[str, Any]]:
    """Return the verl tool tuple and preserve diagnostics in its extra-fields side channel."""
    metrics.setdefault("latency_stage", _latency_stage(agent_data))
    extra_fields = getattr(agent_data, "extra_fields", None)
    if isinstance(extra_fields, dict):
        extras = extra_fields.get("extras")
        if not isinstance(extras, dict):
            extras = {}
            extra_fields["extras"] = extras
        extras.setdefault("runtime_memory_metrics", []).append(metrics)
    return ToolResponse(text=text), reward, metrics


def _latency_stage(agent_data: Any) -> int | None:
    """Use verl's user-turn counter to group calls dispatched in parallel."""

    value = getattr(agent_data, "user_turns", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
