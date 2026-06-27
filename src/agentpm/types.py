from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NotRequired, Required, TypedDict

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | dict[str, "JsonValue"] | list["JsonValue"]


class DependencyReferenceObject(TypedDict, total=False):
    name: Required[str]
    version: NotRequired[str]


DependencyReference = str | DependencyReferenceObject


class ToolMeta(TypedDict, total=False):
    name: Required[str]
    version: Required[str]
    description: NotRequired[str]
    inputs: NotRequired[JsonValue]
    outputs: NotRequired[JsonValue]
    runtime: NotRequired[Runtime]
    environment: NotRequired[Environment]


class Runtime(TypedDict, total=False):
    type: str
    version: str


class Entrypoint(TypedDict, total=False):
    command: str
    args: list[str]
    cwd: str
    timeout_ms: int
    env: dict[str, str]


class EnvVar(TypedDict, total=False):
    required: bool
    description: str
    default: str | None


class Environment(TypedDict, total=False):
    vars: dict[str, EnvVar]


class Manifest(ToolMeta, total=False):
    entrypoint: Entrypoint


ToolFunc = Callable[[JsonValue], JsonValue]


class LoadedWithMeta(TypedDict):
    func: ToolFunc
    meta: ToolMeta


class AgentMeta(TypedDict, total=False):
    kind: Required[Literal["agent"]]
    name: Required[str]
    version: Required[str]
    description: NotRequired[str]
    tools: NotRequired[list[DependencyReference]]
    examples: NotRequired[list[JsonValue]]
    skills: NotRequired[list[DependencyReference]]
    knowledge: NotRequired[list[DependencyReference]]
    memory: NotRequired[list[DependencyReference]]
    profiles: NotRequired[list[DependencyReference]]


class SkillCompatibility(TypedDict, total=False):
    model_families: list[str]
    runtimes: list[str]
    environments: list[str]
    export_targets: list[str]


class SkillMetadata(TypedDict, total=False):
    entrypoint: Required[str]
    references: NotRequired[list[str]]
    scripts: NotRequired[list[str]]
    compatibility: NotRequired[SkillCompatibility]


class SkillMeta(TypedDict, total=False):
    kind: Required[Literal["skill"]]
    name: Required[str]
    version: Required[str]
    description: NotRequired[str]
    tools: NotRequired[list[DependencyReference]]
    skill: Required[SkillMetadata]


class ReservedReferences(TypedDict):
    skills: list[DependencyReference]
    knowledge: list[DependencyReference]
    memory: list[DependencyReference]
    profiles: list[DependencyReference]


class ResolvedAgentToolRef(TypedDict):
    packageKey: str
    kind: Literal["tool"]
    name: str
    version: str
    integrity: str
    root: str | None
    manifestPath: str | None


class ResolvedAgentSkillRef(TypedDict):
    packageKey: str
    kind: Literal["skill"]
    name: str
    version: str
    integrity: str
    root: str | None
    manifestPath: str | None


class LoadedAgent(TypedDict):
    root: str
    manifestPath: str
    manifest: AgentMeta
    resolvedTools: list[ResolvedAgentToolRef]
    resolvedSkills: list[ResolvedAgentSkillRef]
    reserved: ReservedReferences


class LoadedSkill(TypedDict):
    kind: Literal["skill"]
    name: str
    version: str
    description: str | None
    root: str
    manifestPath: str
    manifest: SkillMeta
    skill: SkillMetadata
    entrypointPath: str
    entrypointContent: str
    references: list[str]
    scripts: list[str]
    resolvedTools: list[ResolvedAgentToolRef]
