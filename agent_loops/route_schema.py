"""Presentation ordering for runtime-memory action enums and factual model cards."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any, Sequence


MODEL_CARDS_HEADING = "Model cards (factual metadata only; list order is not a preference):"
MEMORY_TOOL_NAMES = frozenset(("retrieve_memory", "runtime_memory"))


def append_model_cards(description: str, model_specs: Sequence[dict[str, Any]]) -> str:
    lines = []
    for spec in model_specs:
        modalities = "text, image" if spec["kind"] == "vlm" else "text"
        lines.append(
            f"- {spec['name']}: input modalities = {modalities}; "
            f"parameters = {spec['parameter_count']}"
        )
    return f"{description.rstrip()}\n{MODEL_CARDS_HEADING}\n" + "\n".join(lines)


def order_runtime_memory_enums(
    tool_schemas: Sequence[dict[str, Any]],
    *,
    ordering_key: str,
) -> list[dict[str, Any]]:
    """Copy schemas and permute memory-tool presentation for one rollout."""
    schemas = deepcopy(list(tool_schemas))

    memory_tool_indices = [
        index
        for index, schema in enumerate(schemas)
        if schema.get("function", {}).get("name") in MEMORY_TOOL_NAMES
    ]
    ordered_memory_tools = sorted(
        (schemas[index] for index in memory_tool_indices),
        key=lambda schema: sha256(
            f"{ordering_key}\0tool\0{schema['function']['name']}".encode("utf-8")
        ).digest(),
    )
    for index, schema in zip(memory_tool_indices, ordered_memory_tools, strict=True):
        schemas[index] = schema

    for schema in schemas:
        function = schema.get("function", {})
        if function.get("name") != "runtime_memory":
            continue
        properties = function.get("parameters", {}).get("properties", {})
        property_schema = properties.get("model", {})
        values = property_schema.get("enum")
        if not isinstance(values, list) or len(values) <= 1:
            continue
        ordered_values = sorted(
            values,
            key=lambda value: sha256(
                f"{ordering_key}\0model\0{value}".encode("utf-8")
            ).digest(),
        )
        property_schema["enum"] = ordered_values
        property_schema["description"] = _order_model_cards(
            property_schema.get("description"),
            ordered_values,
        )

    return schemas


def _order_model_cards(description: Any, model_names: Sequence[str]) -> Any:
    if not isinstance(description, str):
        return description
    prefix, separator, card_block = description.partition(MODEL_CARDS_HEADING)
    if not separator:
        return description

    cards = {}
    for line in card_block.strip().splitlines():
        if not line.startswith("- "):
            continue
        model_name, delimiter, _ = line[2:].partition(": ")
        if delimiter:
            cards[model_name] = line
    if any(model_name not in cards for model_name in model_names):
        return description
    return f"{prefix}{MODEL_CARDS_HEADING}\n" + "\n".join(cards[name] for name in model_names)
