from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired, Required

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | dict[str, "JsonValue"] | list["JsonValue"]


class ToolMeta(TypedDict, total=False):
    name: Required[str]
    version: Required[str]
    description: NotRequired[str]
    inputs: NotRequired[JsonValue]
    outputs: NotRequired[JsonValue]


class Entrypoint(TypedDict, total=False):
    command: str
    args: list[str]
    cwd: str
    timeout_ms: int
    env: dict[str, str]


class Manifest(TypedDict, total=False):
    name: str
    version: str
    description: str
    inputs: JsonValue
    outputs: JsonValue
    entrypoint: Entrypoint
