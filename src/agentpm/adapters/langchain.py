from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypedDict, TypeGuard, cast

from ..types import JsonValue, ToolMeta


def _as_tool_meta(meta_obj: object) -> ToolMeta:
    meta: ToolMeta = {"name": "agentpm_tool", "version": "0.0.0"}
    if isinstance(meta_obj, dict):
        name = meta_obj.get("name")
        if isinstance(name, str) and name:
            meta["name"] = name

        version = meta_obj.get("version")
        if isinstance(version, str) and version:
            meta["version"] = version

        desc = meta_obj.get("description")
        if isinstance(desc, str):
            meta["description"] = desc

        inputs = meta_obj.get("inputs")
        if inputs is not None:
            meta["inputs"] = cast(JsonValue, inputs)

        outputs = meta_obj.get("outputs")
        if outputs is not None:
            meta["outputs"] = cast(JsonValue, outputs)
    return meta


# ---------- JSON Schema typing & guards ----------


class JsonSchemaProperty(TypedDict, total=False):
    type: str


class JsonSchemaObject(TypedDict, total=False):
    type: str
    properties: dict[str, JsonSchemaProperty]
    required: list[str]


def _is_json_schema_object(x: object) -> TypeGuard[JsonSchemaObject]:
    if not isinstance(x, dict):
        return False
    if x.get("type") != "object":
        return False
    props = x.get("properties")
    # ok if missing or a dict
    return props is None or isinstance(props, dict)


def _is_record_json(x: object) -> TypeGuard[dict[str, JsonValue]]:
    return isinstance(x, dict)


# ---------- Result mapping ----------


def _default_result_to_string(result: JsonValue, meta: ToolMeta | None) -> str:
    outputs = meta.get("outputs") if meta else None
    if _is_json_schema_object(outputs):
        o = outputs
        req = o.get("required")
        key = req[0] if isinstance(req, list) and req else None
        if isinstance(key, str):
            props = o.get("properties") or {}
            prop = props.get(key)
            if isinstance(prop, dict) and prop.get("type") == "string" and _is_record_json(result):
                val = result.get(key)
                if isinstance(val, str):
                    return val
    return result if isinstance(result, str) else json.dumps(result)


# ---------- Minimal adapter tool we return ----------


@dataclass
class _AdapterTool:
    name: str
    description: str
    _call_structured: Callable[[Mapping[str, JsonValue]], str]

    def invoke(self, args: Mapping[str, JsonValue]) -> str:
        return self._call_structured(args)

    def func(self, args: Mapping[str, JsonValue]) -> str:
        return self._call_structured(args)

    def __call__(self, args: Mapping[str, JsonValue]) -> str:
        return self._call_structured(args)


# ---------- Public API ----------


def to_langchain_tool(
    loaded: Mapping[str, object],
    *,
    name: str | None = None,
    description: str | None = None,
    result_to_string: Callable[[JsonValue], str] | None = None,
    force_simple: bool = False,
) -> _AdapterTool:
    # Enforce optional dependency presence (DX parity with Node)
    try:
        import langchain_core.tools as _  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "to_langchain_tool() requires langchain-core. Install with: pip install 'agentpm[langchain]'"
        ) from e

    func_obj = loaded.get("func")
    meta_obj = loaded.get("meta")

    if not callable(func_obj):
        raise TypeError("loaded['func'] must be callable")

    tool_func: Callable[[JsonValue], JsonValue] = func_obj
    meta: ToolMeta = _as_tool_meta(meta_obj)

    tool_name = name or meta.get("name") or "agentpm_tool"
    desc_base = description or meta.get("description") or ""
    rich_desc = desc_base
    if "inputs" in meta:
        rich_desc += f" Inputs: {json.dumps(meta['inputs'])}."
    if "outputs" in meta:
        rich_desc += f" Outputs: {json.dumps(meta['outputs'])}."

    r2s: Callable[[JsonValue], str] = result_to_string or (
        lambda r: _default_result_to_string(r, meta)
    )

    inputs_schema = meta.get("inputs")
    structured = _is_json_schema_object(inputs_schema) and not force_simple

    if structured:

        def call_structured(args: Mapping[str, JsonValue]) -> str:
            res = tool_func(dict(args))  # plain dict[str, JsonValue]
            return r2s(res)

        return _AdapterTool(name=tool_name, description=rich_desc, _call_structured=call_structured)

    # Simple path: coerce to object if schema hints at properties
    def call_simple(input_like: Mapping[str, JsonValue]) -> str:
        payload: JsonValue
        if _is_json_schema_object(inputs_schema):
            props = list((inputs_schema.get("properties") or {}).keys())
            if "text" in props and isinstance(input_like.get("text"), str):
                payload = {"text": input_like["text"]}
            elif len(props) == 1:
                key = props[0]
                val = input_like.get(key)
                # accept JSON-compatible values; else stringify
                if val is None or isinstance(val, str | int | float | bool | dict | list):
                    payload = {key: val}
                else:
                    payload = {key: str(val)}
            else:
                # Try common keys
                for k in ("input", "value", "text"):
                    v = input_like.get(k)
                    if isinstance(v, str | int | float | bool):
                        payload = {"text": str(v)}
                        break
                else:
                    payload = dict(input_like)
        else:
            if isinstance(input_like.get("text"), str):
                payload = {"text": input_like["text"]}
            else:
                payload = dict(input_like)

        res = tool_func(payload)
        return r2s(res)

    return _AdapterTool(name=tool_name, description=rich_desc, _call_structured=call_simple)
