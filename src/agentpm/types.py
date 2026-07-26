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


class KnowledgeDocument(TypedDict, total=False):
    path: Required[str]
    content_type: NotRequired[str]
    role: NotRequired[str]
    description: NotRequired[str]
    bytes: NotRequired[int]
    sha256: NotRequired[str]


class KnowledgeContext(TypedDict, total=False):
    document_count: NotRequired[int]
    total_bytes: NotRequired[int]
    content_hash: NotRequired[str]


class KnowledgeCorpus(TypedDict, total=False):
    chunks_path: NotRequired[str]
    sources_path: NotRequired[str]
    chunk_count: NotRequired[int]
    source_count: NotRequired[int]
    content_hash: NotRequired[str]


class KnowledgeEmbedding(TypedDict, total=False):
    id: NotRequired[str]
    provider: NotRequired[str]
    model: NotRequired[str]
    dimensions: NotRequired[int]
    metric: NotRequired[str]
    normalized: NotRequired[bool]
    vectors_path: NotRequired[str]
    vector_count: NotRequired[int]
    vectors_hash: NotRequired[str]


class KnowledgeIndex(TypedDict, total=False):
    id: NotRequired[str]
    type: NotRequired[str]
    path: NotRequired[str]
    embedding_id: NotRequired[str]
    generated_by: NotRequired[str]


class KnowledgeRetrieval(TypedDict, total=False):
    strategy: NotRequired[str]
    default_top_k: NotRequired[int]
    default_score_threshold: NotRequired[float]
    return_citations: NotRequired[bool]


class KnowledgeBuilder(TypedDict, total=False):
    name: NotRequired[str]
    version: NotRequired[str]


class KnowledgeProvenance(TypedDict, total=False):
    sources_manifest_path: NotRequired[str]
    generated_at: NotRequired[str]
    builder: NotRequired[KnowledgeBuilder]


class KnowledgeMetadata(TypedDict, total=False):
    mode: Required[Literal["context", "vector"]]
    content_type: NotRequired[str]
    language: NotRequired[str]
    documents: NotRequired[list[KnowledgeDocument]]
    context: NotRequired[KnowledgeContext]
    corpus: NotRequired[KnowledgeCorpus]
    embedding: NotRequired[KnowledgeEmbedding]
    indexes: NotRequired[list[KnowledgeIndex]]
    retrieval: NotRequired[KnowledgeRetrieval]
    provenance: NotRequired[KnowledgeProvenance]


class KnowledgeMeta(TypedDict, total=False):
    kind: Required[Literal["knowledge"]]
    name: Required[str]
    version: Required[str]
    description: NotRequired[str]
    knowledge: Required[KnowledgeMetadata]


class MemoryScope(TypedDict, total=False):
    description: NotRequired[str]


class MemoryRecordType(TypedDict, total=False):
    description: NotRequired[str]
    schema: Required[str]
    version: Required[str]


class MemoryRetrieval(TypedDict):
    modes: list[str]


class MemoryCapacity(TypedDict, total=False):
    max_records: NotRequired[int]


class MemoryRetention(TypedDict, total=False):
    ttl: NotRequired[str]
    on_expire: NotRequired[str]


class MemoryConstraints(TypedDict, total=False):
    append_only: NotRequired[bool]


class MemorySpace(TypedDict, total=False):
    description: NotRequired[str]
    model: Required[str]
    scope: Required[list[str]]
    record_types: Required[list[str]]
    retrieval: Required[MemoryRetrieval]
    capacity: NotRequired[MemoryCapacity]
    retention: NotRequired[MemoryRetention]
    constraints: NotRequired[MemoryConstraints]


class MemoryOperationRef(TypedDict):
    space: str
    record_type: str


class MemoryOperationTarget(TypedDict):
    space: str
    record_type: str


class MemoryOperationTrigger(TypedDict, total=False):
    type: Required[str]
    space: NotRequired[str]
    threshold: NotRequired[int]
    every: NotRequired[str]


class MemoryOperation(TypedDict, total=False):
    type: Required[str]
    description: NotRequired[str]
    inputs: NotRequired[list[MemoryOperationRef]]
    output: NotRequired[MemoryOperationRef]
    targets: NotRequired[list[MemoryOperationTarget]]
    trigger: NotRequired[MemoryOperationTrigger]
    source_handling: NotRequired[str]
    preserve_provenance: NotRequired[bool]
    cascade_derived_records: NotRequired[bool]


class MemoryMetadata(TypedDict):
    scopes: dict[str, MemoryScope]
    record_types: dict[str, MemoryRecordType]
    spaces: dict[str, MemorySpace]
    operations: NotRequired[dict[str, MemoryOperation]]


class MemoryMeta(TypedDict, total=False):
    kind: Required[Literal["memory"]]
    name: Required[str]
    version: Required[str]
    description: NotRequired[str]
    memory: Required[MemoryMetadata]


class MemoryBuildSourceSchemaEntry(TypedDict):
    path: str
    sha256: str


class MemoryBuildMetadata(TypedDict, total=False):
    type: Required[str]
    format_version: Required[int]
    built_at: NotRequired[str]
    agentpm_version: NotRequired[str]
    manifest_path: Required[str]
    source_manifest_hash: Required[str]
    source_schemas: NotRequired[list[MemoryBuildSourceSchemaEntry]]
    source_schemas_hash: Required[str]
    source_contract_inputs_hash: Required[str]
    contracts_index_hash: Required[str]
    contracts_hash: Required[str]
    contract_count: Required[int]


class MemoryContractIndexEntry(TypedDict):
    space: str
    record_type: str
    schema_version: str
    model: str
    source_schema: str
    path: str
    sha256: str


class MemoryContractIndex(TypedDict):
    type: str
    format_version: int
    contracts: list[MemoryContractIndexEntry]


MemoryContractSchema = dict[str, object]


class ReservedReferences(TypedDict):
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


class ResolvedAgentKnowledgeRef(TypedDict):
    packageKey: str
    kind: Literal["knowledge"]
    name: str
    version: str
    integrity: str
    mode: Literal["context", "vector"] | None
    root: str | None
    manifestPath: str | None


class ResolvedAgentMemoryRef(TypedDict):
    packageKey: str
    kind: Literal["memory"]
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
    resolvedKnowledge: list[ResolvedAgentKnowledgeRef]
    resolvedMemory: list[ResolvedAgentMemoryRef]
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


class LoadedKnowledge(TypedDict):
    kind: Literal["knowledge"]
    name: str
    version: str
    description: str | None
    root: str
    manifestPath: str
    manifest: KnowledgeMeta
    knowledge: KnowledgeMetadata
    documentPaths: list[str]
    chunksPath: str | None
    sourcesPath: str | None
    vectorsPath: str | None
    indexPaths: list[str]
    provenancePath: str | None


class LoadedMemoryContractRef(TypedDict):
    space: str
    recordType: str
    schemaVersion: str
    model: str
    sourceSchemaPath: str
    path: str
    sha256: str


class LoadedMemory(TypedDict):
    kind: Literal["memory"]
    name: str
    version: str
    description: str | None
    root: str
    manifestPath: str
    manifest: MemoryMeta
    memory: MemoryMetadata
    buildPath: str
    build: MemoryBuildMetadata
    contractIndexPath: str
    contractIndex: MemoryContractIndex
    sourceSchemaPaths: list[str]
    contracts: list[LoadedMemoryContractRef]
