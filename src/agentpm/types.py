from __future__ import annotations

from collections.abc import Callable
from typing import NotRequired, Required, TypedDict

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | dict[str, "JsonValue"] | list["JsonValue"]


class ToolMeta(TypedDict, total=False):
    name: Required[str]
    version: Required[str]
    description: NotRequired[str]
    inputs: NotRequired[JsonValue]
    outputs: NotRequired[JsonValue]
    runtime: NotRequired[Runtime]


class Runtime(TypedDict, total=False):
    type: str
    version: str


class Entrypoint(TypedDict, total=False):
    command: str
    args: list[str]
    cwd: str
    timeout_ms: int
    env: dict[str, str]


class Manifest(ToolMeta, total=False):
    entrypoint: Entrypoint


ToolFunc = Callable[[JsonValue], JsonValue]


class LoadedWithMeta(TypedDict):
    func: ToolFunc
    meta: ToolMeta
