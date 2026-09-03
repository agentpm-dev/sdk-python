from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Required, TextIO, TypedDict, cast, overload

JsonValue = str | int | float | bool | None | dict[str, Any] | list[Any]
HarnessHookId = Literal[
    "before_model_request",
    "before_tool_selection",
    "before_tool_call",
    "before_knowledge_request",
    "after_knowledge_retrieval",
    "before_memory_read",
    "before_memory_write",
    "before_memory_operation",
]
HarnessServiceRole = Literal["model", "embedding", "hook", "knowledge", "memory", "approval"]
AgentpmServiceFrameKind = Literal[
    "initialize",
    "initialized",
    "request",
    "response",
    "event",
    "error",
]

HostServiceHandler = Callable[["HostServiceRequest"], JsonValue]
ApprovalHandler = Callable[[JsonValue], str | dict[str, Any]]
ModelProviderHandler = Callable[[JsonValue], JsonValue]


class EmbeddingProviderRequest(TypedDict):
    provider: str
    model: str
    dimensions: int
    normalized: bool
    text: str


class EmbeddingProviderVectorResult(TypedDict, total=False):
    vector: list[float]
    values: list[float]
    provider: str
    model: str
    dimensions: int
    normalized: bool


EmbeddingProviderResult = list[float] | EmbeddingProviderVectorResult
EmbeddingProviderHandler = Callable[[EmbeddingProviderRequest], EmbeddingProviderResult]
KnowledgeRequestMode = Literal["context_document", "vector_query"]


class KnowledgeRuntimeRequest(TypedDict, total=False):
    package: Required[str]
    version: Required[str]
    mode: Required[KnowledgeRequestMode]
    document: str
    query: str
    top_k: int
    score_threshold: float
    return_citations: bool


class KnowledgeRuntimeFailure(TypedDict, total=False):
    code: Required[str]
    message: Required[str]
    retryable: bool


class KnowledgeRetrievalResult(TypedDict, total=False):
    rank: Required[int]
    score: Required[float]
    chunk_id: Required[str]
    source_id: Required[str]
    source_title: str
    source_uri: str
    text: str
    chunk_metadata: JsonValue
    source_metadata: JsonValue


class KnowledgeCitation(TypedDict, total=False):
    chunk_id: Required[str]
    source_id: Required[str]
    title: str
    uri: str


class KnowledgeRuntimeResult(TypedDict, total=False):
    ok: Required[bool]
    package: Required[str]
    version: Required[str]
    mode: Required[KnowledgeRequestMode]
    document: str
    query: str
    content: str
    results: list[KnowledgeRetrievalResult]
    citations: list[KnowledgeCitation]
    error: KnowledgeRuntimeFailure


KnowledgeRuntimeHandler = Callable[[KnowledgeRuntimeRequest], KnowledgeRuntimeResult]


class HostServiceRegistration(TypedDict):
    role: HarnessServiceRole
    registry_id: str


class HostServiceRegistrationResult(TypedDict, total=False):
    registered: Required[bool]
    service: Required[HostServiceRegistration]
    active: Required[bool]
    reason: str | None


class ModelProviderCapabilities(TypedDict, total=False):
    provider: str
    model: str
    models: list[str]
    semantic_actions: Required[bool]
    structured_output: Required[bool]
    multimodal_input: Required[bool]
    context_window_tokens: int
    usage_reporting: Required[bool]


class EmbeddingSpaceCapability(TypedDict):
    provider: str
    model: str
    dimensions: int
    normalized: bool


class EmbeddingProviderCapabilities(TypedDict):
    embedding_spaces: list[EmbeddingSpaceCapability]


class KnowledgePackageRealization(TypedDict, total=False):
    package: Required[str]
    version: Required[str]
    corpus: str
    ready: Required[bool]


class KnowledgeProviderCapabilities(TypedDict, total=False):
    modes: Required[list[str]]
    features: Required[list[str]]
    packages: list[KnowledgePackageRealization]


class MemoryPackageRealization(TypedDict, total=False):
    package: Required[str]
    version: str
    ready: Required[bool]


class MemoryProviderCapabilities(TypedDict, total=False):
    descriptor: Required[JsonValue]
    packages: list[MemoryPackageRealization]


class ApprovalCapabilities(TypedDict, total=False):
    approval: Literal[True]
    cancellation: bool


class HookCapabilities(TypedDict):
    hooks: list[HarnessHookId]


HostProviderCapabilities = (
    ModelProviderCapabilities
    | EmbeddingProviderCapabilities
    | KnowledgeProviderCapabilities
    | MemoryProviderCapabilities
    | dict[str, Any]
)


class HookContinueDecision(TypedDict):
    decision: Literal["continue"]
    patch: NotRequired[dict[str, Any]]


class HookRejectDecision(TypedDict):
    decision: Literal["reject"]
    reason: str


HookDecision = HookContinueDecision | HookRejectDecision
HookHandler = Callable[[JsonValue], HookDecision | None]


class HookPhaseSnapshot(TypedDict):
    phase_id: str
    phase_objective: str
    completion: JsonValue


class BeforeModelRequestModel(TypedDict):
    provider: str
    model: str
    options: NotRequired[JsonValue]


class BeforeModelRequestSection(TypedDict):
    number: int
    title: str
    content: str
    mutable: bool


class BeforeModelRequestInput(TypedDict):
    run_id: str
    phase_execution_id: str
    phase: HookPhaseSnapshot
    model: NotRequired[BeforeModelRequestModel]
    sections: list[BeforeModelRequestSection]
    repair_feedback: NotRequired[str]


class BeforeModelRequestContextSection(TypedDict):
    title: str
    content: str


class BeforeModelRequestPatch(TypedDict, total=False):
    context_sections: list[BeforeModelRequestContextSection]
    provider_options: dict[str, JsonValue]


class BeforeModelRequestContinueDecision(TypedDict):
    decision: Literal["continue"]
    patch: NotRequired[BeforeModelRequestPatch]


BeforeModelRequestDecision = BeforeModelRequestContinueDecision | HookRejectDecision
BeforeModelRequestHookHandler = Callable[
    [BeforeModelRequestInput], BeforeModelRequestDecision | None
]


class BeforeToolSelectionCandidate(TypedDict):
    canonical_id: str
    description: str
    source: str


class BeforeToolSelectionInput(TypedDict):
    phase: HookPhaseSnapshot
    candidates: list[BeforeToolSelectionCandidate]


class BeforeToolSelectionPatch(TypedDict, total=False):
    candidate_ids: list[str]


class BeforeToolSelectionContinueDecision(TypedDict):
    decision: Literal["continue"]
    patch: NotRequired[BeforeToolSelectionPatch]


BeforeToolSelectionDecision = BeforeToolSelectionContinueDecision | HookRejectDecision
BeforeToolSelectionHookHandler = Callable[
    [BeforeToolSelectionInput], BeforeToolSelectionDecision | None
]


class BeforeToolCallInput(TypedDict):
    phase_id: str
    tool: str
    arguments: JsonValue


class BeforeToolCallPatch(TypedDict, total=False):
    arguments: JsonValue


class BeforeToolCallContinueDecision(TypedDict):
    decision: Literal["continue"]
    patch: NotRequired[BeforeToolCallPatch]


BeforeToolCallDecision = BeforeToolCallContinueDecision | HookRejectDecision
BeforeToolCallHookHandler = Callable[[BeforeToolCallInput], BeforeToolCallDecision | None]


class BeforeKnowledgeRequestInput(TypedDict):
    phase_id: str
    request: KnowledgeRuntimeRequest


class BeforeKnowledgeRequestPatch(TypedDict, total=False):
    document: str
    query: str
    top_k: int
    score_threshold: float
    return_citations: bool


class BeforeKnowledgeRequestContinueDecision(TypedDict):
    decision: Literal["continue"]
    patch: NotRequired[BeforeKnowledgeRequestPatch]


BeforeKnowledgeRequestDecision = BeforeKnowledgeRequestContinueDecision | HookRejectDecision
BeforeKnowledgeRequestHookHandler = Callable[
    [BeforeKnowledgeRequestInput], BeforeKnowledgeRequestDecision | None
]


class AfterKnowledgeRetrievalInput(TypedDict):
    phase_id: str
    result: KnowledgeRuntimeResult


class AfterKnowledgeRetrievalResultPatch(TypedDict, total=False):
    chunk_id: Required[str]
    source_id: Required[str]
    text: str


class AfterKnowledgeRetrievalPatch(TypedDict, total=False):
    content: str
    results: list[AfterKnowledgeRetrievalResultPatch]


class AfterKnowledgeRetrievalContinueDecision(TypedDict):
    decision: Literal["continue"]
    patch: NotRequired[AfterKnowledgeRetrievalPatch]


AfterKnowledgeRetrievalDecision = AfterKnowledgeRetrievalContinueDecision | HookRejectDecision
AfterKnowledgeRetrievalHookHandler = Callable[
    [AfterKnowledgeRetrievalInput], AfterKnowledgeRetrievalDecision | None
]


class BeforeMemoryReadInput(TypedDict, total=False):
    phase_id: Required[str]
    package: Required[str]
    space: Required[str]
    scope: Required[JsonValue]
    query: str
    filter: JsonValue
    limit: int
    mode: str


class BeforeMemoryReadPatch(TypedDict, total=False):
    query: str
    filter: JsonValue
    limit: int
    mode: str


class BeforeMemoryReadContinueDecision(TypedDict):
    decision: Literal["continue"]
    patch: NotRequired[BeforeMemoryReadPatch]


BeforeMemoryReadDecision = BeforeMemoryReadContinueDecision | HookRejectDecision
BeforeMemoryReadHookHandler = Callable[[BeforeMemoryReadInput], BeforeMemoryReadDecision | None]


class BeforeMemoryWriteInput(TypedDict):
    phase_id: str
    package: str
    space: str
    record_type: str
    scope: JsonValue
    content: JsonValue


class BeforeMemoryWritePatch(TypedDict, total=False):
    content: JsonValue


class BeforeMemoryWriteContinueDecision(TypedDict):
    decision: Literal["continue"]
    patch: NotRequired[BeforeMemoryWritePatch]


BeforeMemoryWriteDecision = BeforeMemoryWriteContinueDecision | HookRejectDecision
BeforeMemoryWriteHookHandler = Callable[[BeforeMemoryWriteInput], BeforeMemoryWriteDecision | None]


class BeforeMemoryOperationInput(TypedDict):
    phase_id: str
    package: str
    operation: str
    scope: JsonValue
    source_summary: JsonValue


class BeforeMemoryOperationPatch(TypedDict, total=False):
    model_guidance: str


class BeforeMemoryOperationContinueDecision(TypedDict):
    decision: Literal["continue"]
    patch: NotRequired[BeforeMemoryOperationPatch]


BeforeMemoryOperationDecision = BeforeMemoryOperationContinueDecision | HookRejectDecision
BeforeMemoryOperationHookHandler = Callable[
    [BeforeMemoryOperationInput], BeforeMemoryOperationDecision | None
]

PROTOCOL = "agentpm-harness-machine"
VERSION = 1
SERVICE_PROTOCOL = "agentpm-service"
SERVICE_VERSION = 1
DEFAULT_HOOK_REGISTRY_ID = "sdk-hooks"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


class HarnessProtocolError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def serve_knowledge_runtime_process(
    registry_id: str,
    handler: KnowledgeRuntimeHandler,
    capabilities: KnowledgeProviderCapabilities,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    ready: bool = True,
) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout

    def write(frame: dict[str, Any]) -> None:
        output_stream.write(
            json.dumps(
                {
                    "protocol": SERVICE_PROTOCOL,
                    "version": SERVICE_VERSION,
                    **frame,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        output_stream.flush()

    def write_error(frame: dict[str, Any], code: str, error: Exception) -> None:
        write(
            {
                "kind": "error",
                "id": frame.get("id"),
                "service": frame.get("service") or "knowledge",
                "error": {
                    "code": code,
                    "message": str(error) or error.__class__.__name__,
                    "retryable": False,
                },
            }
        )

    for line in input_stream:
        if not line.strip():
            continue
        frame: dict[str, Any] = {"service": "knowledge"}
        try:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise RuntimeError("service frame must be an object")
            frame = parsed
            if frame.get("protocol") != SERVICE_PROTOCOL:
                raise RuntimeError(f"unsupported service protocol {frame.get('protocol')}")
            if frame.get("version") != SERVICE_VERSION:
                raise RuntimeError(f"unsupported service protocol version {frame.get('version')}")
            if frame.get("service") != "knowledge":
                raise RuntimeError(f"unsupported service {frame.get('service')}")

            if frame.get("kind") == "initialize":
                write(
                    {
                        "kind": "initialized",
                        "id": frame.get("id"),
                        "service": "knowledge",
                        "result": {
                            **capabilities,
                            "registry_id": registry_id,
                            "ready": ready,
                        },
                    }
                )
                continue

            if frame.get("kind") != "request":
                raise RuntimeError(f"unsupported service frame kind {frame.get('kind')}")
            if frame.get("method") != "retrieve":
                raise RuntimeError(f"Unsupported KnowledgeRuntime method {frame.get('method')}")

            write(
                {
                    "kind": "response",
                    "id": frame.get("id"),
                    "service": "knowledge",
                    "result": handler(_extract_knowledge_runtime_request(frame.get("payload"))),
                }
            )
        except Exception as exc:
            write_error(frame, "knowledge_runtime_error", exc)


@dataclass(frozen=True)
class HostServiceRequest:
    role: HarnessServiceRole
    registry_id: str
    method: str
    payload: JsonValue


@dataclass
class _RegisteredService:
    role: HarnessServiceRole
    registry_id: str
    handler: HostServiceHandler
    hooks: list[HarnessHookId] | None = None
    capabilities: JsonValue = None
    request_timeout_ms: int | None = None
    registration: HostServiceRegistrationResult | None = None


class HarnessClient:
    def __init__(
        self,
        *,
        agentpm_path: str | None = None,
        args: list[str] | None = None,
        agent: str | None = None,
        config_path: str | None = None,
        state_dir: str | None = None,
        scopes: dict[str, str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        """Create a Harness machine client.

        `agent` is passed to `agentpm harness` as the Agent selector. It may be
        a local Agent manifest path, such as `./agent.json`, or an installed
        Agent package ref, such as `@scope/name@1.2.3`. Leave it unset to use
        Harness workspace discovery/default Agent selection.
        """
        self.agentpm_path = agentpm_path or os.environ.get("AGENTPM") or "agentpm"
        self.args = args
        self.agent = agent
        self.config_path = config_path
        self.state_dir = state_dir
        self.scopes = scopes or {}
        self.cwd = cwd
        self.env = env or {}
        self.request_timeout_seconds = request_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 0
        self._initialized = False
        self._transport_error: BaseException | None = None
        self._pending: dict[str, queue.Queue[JsonValue | BaseException]] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._events_condition = threading.Condition()
        self._services: dict[tuple[str, str], _RegisteredService] = {}
        self._registered_service_keys: set[tuple[str, str]] = set()
        self._registration_results: dict[tuple[str, str], HostServiceRegistrationResult] = {}
        self._next_hook_registration_id = 0

    def start(self) -> None:
        if self._transport_error is not None:
            raise self._transport_error
        if self._process is not None:
            return
        env = os.environ.copy()
        env.update(self.env)
        self._process = subprocess.Popen(
            [self.agentpm_path, *self._default_args()],
            cwd=self.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def initialize(self) -> dict[str, Any]:
        self.start()
        response = self._request("initialize", {})
        if not isinstance(response, dict):
            raise RuntimeError("Harness initialize returned a non-object response")
        self._initialized = True
        self._flush_registrations()
        return response

    def preflight(self) -> dict[str, Any]:
        self._ensure_initialized()
        response = self._request("preflight", {})
        if not isinstance(response, dict):
            raise RuntimeError("Harness preflight returned a non-object response")
        return response

    def run(self, input: str, **payload: JsonValue) -> dict[str, Any]:
        self._ensure_initialized()
        response = self._request("start_run", {**payload, "input": input})
        if not isinstance(response, dict):
            raise RuntimeError("Harness start_run returned a non-object response")
        return response

    def start_run(self, input: str, **payload: JsonValue) -> dict[str, Any]:
        return self.run(input, **payload)

    def cancel_run(self) -> JsonValue:
        self.start()
        return self._request("cancel_run", {})

    def invoke_memory_operation(self, payload: JsonValue) -> JsonValue:
        self._ensure_initialized()
        return self._request("memory_operation", payload)

    def shutdown(self) -> JsonValue:
        if self._process is None:
            return {"shutdown": True}
        response = self._request("shutdown", {})
        if self._process.stdin is not None:
            self._process.stdin.close()
        return response

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        self._process = None
        with self._events_condition:
            self._events_condition.notify_all()

    def events(self) -> Iterator[dict[str, Any]]:
        self.start()
        offset = 0
        while self._process is not None or offset < len(self._events):
            with self._events_condition:
                if offset >= len(self._events):
                    self._events_condition.wait(timeout=self.request_timeout_seconds)
                if offset < len(self._events):
                    yield self._events[offset]
                    offset += 1

    def wait_for_event(
        self, predicate: Callable[[dict[str, Any]], bool], timeout_seconds: float = 5.0
    ) -> dict[str, Any]:
        self.start()
        deadline = threading.Event()
        timer = threading.Timer(timeout_seconds, deadline.set)
        timer.start()
        try:
            with self._events_condition:
                while not deadline.is_set():
                    for event in self._events:
                        if predicate(event):
                            return event
                    self._events_condition.wait(timeout=0.05)
        finally:
            timer.cancel()
        raise TimeoutError("Timed out waiting for Harness event")

    def register_host_service(
        self,
        role: HarnessServiceRole,
        registry_id: str,
        handler: HostServiceHandler,
        *,
        hooks: list[HarnessHookId] | None = None,
        capabilities: JsonValue = None,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        self._services[(role, registry_id)] = _RegisteredService(
            role=role,
            registry_id=registry_id,
            handler=handler,
            hooks=hooks,
            capabilities=capabilities,
            request_timeout_ms=request_timeout_ms,
        )
        self._registered_service_keys.discard((role, registry_id))
        self._registration_results.pop((role, registry_id), None)
        if self._initialized:
            self._flush_registrations()
        return self

    def host_service_registration(
        self,
        role: HarnessServiceRole,
        registry_id: str,
    ) -> HostServiceRegistrationResult | None:
        return self._registration_results.get((role, registry_id))

    def host_service_registrations(self) -> list[HostServiceRegistrationResult]:
        return list(self._registration_results.values())

    def register_model_provider(
        self,
        registry_id: str,
        handler: ModelProviderHandler,
        capabilities: ModelProviderCapabilities | None = None,
    ) -> HarnessClient:
        def model_handler(request: HostServiceRequest) -> JsonValue:
            if request.method != "generate":
                raise RuntimeError(f"Unsupported model method {request.method}")
            return _normalize_model_provider_result(handler(request.payload))

        return self.register_host_service(
            "model",
            registry_id,
            model_handler,
            capabilities=_default_model_capabilities(registry_id, capabilities),
        )

    def register_embedding_provider(
        self,
        registry_id: str,
        handler: EmbeddingProviderHandler,
        capabilities: EmbeddingProviderCapabilities,
    ) -> HarnessClient:
        def embedding_handler(request: HostServiceRequest) -> JsonValue:
            if request.method != "embed":
                raise RuntimeError(f"Unsupported embedding method {request.method}")
            return cast(JsonValue, handler(cast(EmbeddingProviderRequest, request.payload)))

        return self.register_host_service(
            "embedding",
            registry_id,
            embedding_handler,
            capabilities=cast(JsonValue, capabilities),
        )

    def register_knowledge_runtime(
        self,
        registry_id: str,
        handler: KnowledgeRuntimeHandler,
        capabilities: KnowledgeProviderCapabilities,
    ) -> HarnessClient:
        def knowledge_handler(request: HostServiceRequest) -> JsonValue:
            if request.method != "retrieve":
                raise RuntimeError(f"Unsupported KnowledgeRuntime method {request.method}")
            return cast(JsonValue, handler(_extract_knowledge_runtime_request(request.payload)))

        return self.register_host_service(
            "knowledge",
            registry_id,
            knowledge_handler,
            capabilities=cast(JsonValue, capabilities),
        )

    @overload
    def register_host_provider(
        self,
        role: Literal["model"],
        registry_id: str,
        handler: HostServiceHandler,
        capabilities: ModelProviderCapabilities | None = None,
    ) -> HarnessClient: ...

    @overload
    def register_host_provider(
        self,
        role: Literal["embedding"],
        registry_id: str,
        handler: HostServiceHandler,
        capabilities: EmbeddingProviderCapabilities | None = None,
    ) -> HarnessClient: ...

    @overload
    def register_host_provider(
        self,
        role: Literal["knowledge"],
        registry_id: str,
        handler: HostServiceHandler,
        capabilities: KnowledgeProviderCapabilities | None = None,
    ) -> HarnessClient: ...

    @overload
    def register_host_provider(
        self,
        role: Literal["memory"],
        registry_id: str,
        handler: HostServiceHandler,
        capabilities: MemoryProviderCapabilities | None = None,
    ) -> HarnessClient: ...

    def register_host_provider(
        self,
        role: Literal["model", "embedding", "knowledge", "memory"],
        registry_id: str,
        handler: HostServiceHandler,
        capabilities: HostProviderCapabilities | None = None,
    ) -> HarnessClient:
        return self.register_host_service(
            role,
            registry_id,
            handler,
            capabilities=_normalize_host_provider_capabilities(
                role,
                registry_id,
                capabilities,
            ),
        )

    def register_hook(
        self,
        hook: HarnessHookId,
        handler: HookHandler,
        *,
        registry_id: str = DEFAULT_HOOK_REGISTRY_ID,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        service_registry_id = self._allocate_hook_registry_id(registry_id)

        def hook_handler(request: HostServiceRequest) -> JsonValue:
            if request.method != hook:
                raise RuntimeError(f"Unsupported Hook method {request.method}")
            decision = handler(_extract_hook_input(request.payload))
            return cast(JsonValue, decision or {"decision": "continue"})

        return self.register_host_service(
            "hook",
            service_registry_id,
            hook_handler,
            hooks=[hook],
            capabilities=cast(JsonValue, cast(HookCapabilities, {"hooks": [hook]})),
            request_timeout_ms=request_timeout_ms,
        )

    def _allocate_hook_registry_id(self, base: str) -> str:
        if ("hook", base) not in self._services:
            return base
        while True:
            self._next_hook_registration_id += 1
            registry_id = f"{base}-{self._next_hook_registration_id}"
            if ("hook", registry_id) not in self._services:
                return registry_id

    def on_before_model_request(
        self,
        handler: BeforeModelRequestHookHandler,
        *,
        registry_id: str = DEFAULT_HOOK_REGISTRY_ID,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        return self.register_hook(
            "before_model_request",
            cast(HookHandler, handler),
            registry_id=registry_id,
            request_timeout_ms=request_timeout_ms,
        )

    def on_before_tool_selection(
        self,
        handler: BeforeToolSelectionHookHandler,
        *,
        registry_id: str = DEFAULT_HOOK_REGISTRY_ID,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        return self.register_hook(
            "before_tool_selection",
            cast(HookHandler, handler),
            registry_id=registry_id,
            request_timeout_ms=request_timeout_ms,
        )

    def on_before_tool_call(
        self,
        handler: BeforeToolCallHookHandler,
        *,
        registry_id: str = DEFAULT_HOOK_REGISTRY_ID,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        return self.register_hook(
            "before_tool_call",
            cast(HookHandler, handler),
            registry_id=registry_id,
            request_timeout_ms=request_timeout_ms,
        )

    def on_before_knowledge_request(
        self,
        handler: BeforeKnowledgeRequestHookHandler,
        *,
        registry_id: str = DEFAULT_HOOK_REGISTRY_ID,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        return self.register_hook(
            "before_knowledge_request",
            cast(HookHandler, handler),
            registry_id=registry_id,
            request_timeout_ms=request_timeout_ms,
        )

    def on_after_knowledge_retrieval(
        self,
        handler: AfterKnowledgeRetrievalHookHandler,
        *,
        registry_id: str = DEFAULT_HOOK_REGISTRY_ID,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        return self.register_hook(
            "after_knowledge_retrieval",
            cast(HookHandler, handler),
            registry_id=registry_id,
            request_timeout_ms=request_timeout_ms,
        )

    def on_before_memory_read(
        self,
        handler: BeforeMemoryReadHookHandler,
        *,
        registry_id: str = DEFAULT_HOOK_REGISTRY_ID,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        return self.register_hook(
            "before_memory_read",
            cast(HookHandler, handler),
            registry_id=registry_id,
            request_timeout_ms=request_timeout_ms,
        )

    def on_before_memory_write(
        self,
        handler: BeforeMemoryWriteHookHandler,
        *,
        registry_id: str = DEFAULT_HOOK_REGISTRY_ID,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        return self.register_hook(
            "before_memory_write",
            cast(HookHandler, handler),
            registry_id=registry_id,
            request_timeout_ms=request_timeout_ms,
        )

    def on_before_memory_operation(
        self,
        handler: BeforeMemoryOperationHookHandler,
        *,
        registry_id: str = DEFAULT_HOOK_REGISTRY_ID,
        request_timeout_ms: int | None = None,
    ) -> HarnessClient:
        return self.register_hook(
            "before_memory_operation",
            cast(HookHandler, handler),
            registry_id=registry_id,
            request_timeout_ms=request_timeout_ms,
        )

    def on_approval(
        self,
        handler: ApprovalHandler,
        capabilities: ApprovalCapabilities | None = None,
    ) -> HarnessClient:
        def approval_handler(request: HostServiceRequest) -> JsonValue:
            if request.method != "request_approval":
                raise RuntimeError(f"Unsupported approval method {request.method}")
            decision = handler(request.payload)
            if isinstance(decision, str):
                return {"decision": decision}
            return decision

        return self.register_host_service(
            "approval",
            "controller",
            approval_handler,
            capabilities=_default_approval_capabilities(capabilities),
        )

    def _default_args(self) -> list[str]:
        if self.args is not None:
            return self.args
        args = ["harness"]
        if self.agent:
            args.append(self.agent)
        if self.config_path:
            args.extend(["--config", self.config_path])
        if self.state_dir:
            args.extend(["--state-dir", self.state_dir])
        for key, value in self.scopes.items():
            args.extend(["--scope", f"{key}={value}"])
        args.append("--machine")
        return args

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _flush_registrations(self) -> None:
        for key, service in list(self._services.items()):
            if key in self._registered_service_keys:
                continue
            response = self._request(
                "register_host_service",
                {
                    "role": service.role,
                    "registry_id": service.registry_id,
                    "capabilities": service.capabilities or {},
                    "hooks": service.hooks or [],
                    "request_timeout_ms": service.request_timeout_ms or 120_000,
                },
            )
            registration = _normalize_host_service_registration_result(response, service)
            service.registration = registration
            self._registration_results[key] = registration
            self._registered_service_keys.add(key)

    def _request(self, method: str, payload: JsonValue) -> JsonValue:
        if self._transport_error is not None:
            raise self._transport_error
        self.start()
        if self._transport_error is not None:
            raise self._transport_error
        self._next_id += 1
        request_id = f"py-sdk-{self._next_id}"
        response_queue: queue.Queue[JsonValue | BaseException] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_queue
        try:
            self._write_frame(
                {
                    "protocol": PROTOCOL,
                    "version": VERSION,
                    "kind": "request",
                    "id": request_id,
                    "method": method,
                    "payload": payload,
                }
            )
            response = response_queue.get(timeout=self.request_timeout_seconds)
            if isinstance(response, BaseException):
                raise response
            return response
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    self._handle_line(line)
        finally:
            returncode = process.wait()
            self._reject_all(RuntimeError(f"Harness process exited with code {returncode}"))
            self._process = None
            with self._events_condition:
                self._events_condition.notify_all()

    def _handle_line(self, line: str) -> None:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            self._fail_transport(exc)
            return
        if frame.get("protocol") != PROTOCOL or frame.get("version") != VERSION:
            self._fail_transport(RuntimeError("Unsupported Harness machine protocol frame"))
            return
        kind = frame.get("kind")
        if kind == "event":
            payload = frame.get("payload") or {}
            if isinstance(payload, dict):
                self._record_event(payload)
            return
        if kind == "request" and frame.get("method") == "host_service":
            threading.Thread(target=self._dispatch_host_service, args=(frame,), daemon=True).start()
            return
        request_id = frame.get("id")
        if not isinstance(request_id, str):
            return
        with self._pending_lock:
            response_queue = self._pending.get(request_id)
        if response_queue is None:
            return
        if kind == "error":
            error = frame.get("error") or {}
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            response_queue.put(
                HarnessProtocolError(str(code or "protocol_error"), str(message or "error"))
            )
        else:
            response_queue.put(frame.get("payload"))

    def _record_event(self, event: dict[str, Any]) -> None:
        with self._events_condition:
            self._events.append(event)
            self._events_condition.notify_all()

    def _dispatch_host_service(self, frame: dict[str, Any]) -> None:
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            self._write_error(frame.get("id"), "host_service_bad_request", "missing payload")
            return
        role = payload.get("role")
        registry_id = payload.get("registry_id")
        method = payload.get("method")
        if (
            not isinstance(role, str)
            or not isinstance(registry_id, str)
            or not isinstance(method, str)
        ):
            self._write_error(
                frame.get("id"),
                "host_service_bad_request",
                "Host service request is missing role, registry_id, or method",
            )
            return
        service = self._services.get((role, registry_id))
        if service is None:
            self._write_error(
                frame.get("id"),
                "host_service_not_registered",
                f"No host service registered for {role}:{registry_id}",
            )
            return
        try:
            response = _run_service_handler(
                service,
                HostServiceRequest(
                    role=role,  # type: ignore[arg-type]
                    registry_id=registry_id,
                    method=method,
                    payload=payload.get("payload") or {},
                ),
            )
            self._write_frame(
                {
                    "protocol": PROTOCOL,
                    "version": VERSION,
                    "kind": "response",
                    "id": frame.get("id"),
                    "payload": response or {},
                }
            )
        except Exception as exc:
            self._write_error(frame.get("id"), "host_service_callback_failed", str(exc))

    def _write_frame(self, frame: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Harness process is not running")
        with self._write_lock:
            self._process.stdin.write(json.dumps(frame, separators=(",", ":")) + "\n")
            self._process.stdin.flush()

    def _write_error(self, request_id: Any, code: str, message: str) -> None:
        self._write_frame(
            {
                "protocol": PROTOCOL,
                "version": VERSION,
                "kind": "error",
                "id": request_id if isinstance(request_id, str) else None,
                "error": {"code": code, "message": message},
            }
        )

    def _reject_all(self, error: BaseException) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for response_queue in pending:
            response_queue.put(error)

    def _fail_transport(self, error: BaseException) -> None:
        if self._transport_error is None:
            self._transport_error = error
        process = self._process
        self._process = None
        if process is not None:
            process.terminate()
        with self._events_condition:
            self._events_condition.notify_all()
        self._reject_all(self._transport_error)


Harness = HarnessClient


def _default_model_capabilities(
    registry_id: str,
    overrides: ModelProviderCapabilities | None = None,
) -> JsonValue:
    capabilities: dict[str, Any] = {
        "provider": registry_id,
        "semantic_actions": True,
        "structured_output": True,
        "multimodal_input": False,
        "usage_reporting": True,
    }
    if overrides:
        capabilities.update(overrides)
    return capabilities


def _normalize_model_provider_result(value: JsonValue) -> JsonValue:
    if not isinstance(value, dict):
        return value
    return {
        **value,
        "usage": _normalize_run_usage(value.get("usage")),
    }


def _normalize_run_usage(value: JsonValue) -> JsonValue:
    usage = value if isinstance(value, dict) else {}
    return {
        "model_calls": _usage_int(usage.get("model_calls")),
        "tokens": _normalize_token_usage(usage.get("tokens")),
        "accepted_semantic_actions": _usage_int(usage.get("accepted_semantic_actions")),
        "tool_calls": _usage_int(usage.get("tool_calls")),
        "tool_retries": _usage_int(usage.get("tool_retries")),
        "knowledge_requests": _usage_int(usage.get("knowledge_requests")),
        "memory_requests": _usage_int(usage.get("memory_requests")),
        "embedding_requests": _usage_int(usage.get("embedding_requests")),
        "duration_ms": _optional_usage_int(usage.get("duration_ms")),
        "cost": _normalize_cost_usage(usage.get("cost")),
    }


def _normalize_token_usage(value: JsonValue) -> JsonValue:
    tokens = value if isinstance(value, dict) else {}
    return {
        "input_tokens": _optional_usage_int(tokens.get("input_tokens")),
        "output_tokens": _optional_usage_int(tokens.get("output_tokens")),
        "total_tokens": _optional_usage_int(tokens.get("total_tokens")),
    }


def _normalize_cost_usage(value: JsonValue) -> JsonValue:
    cost = value if isinstance(value, dict) else {}
    currency = cost.get("currency")
    return {
        "amount": _optional_usage_number(cost.get("amount")),
        "currency": currency if isinstance(currency, str) else None,
    }


def _usage_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _optional_usage_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _optional_usage_number(value: Any) -> int | float | None:
    return (
        value
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        else None
    )


def _normalize_host_provider_capabilities(
    role: Literal["model", "embedding", "knowledge", "memory"],
    registry_id: str,
    capabilities: HostProviderCapabilities | None,
) -> JsonValue:
    if role == "model":
        return _default_model_capabilities(
            registry_id,
            cast(ModelProviderCapabilities | None, capabilities),
        )
    return cast(JsonValue, capabilities or {})


def _default_approval_capabilities(
    overrides: ApprovalCapabilities | None = None,
) -> JsonValue:
    capabilities: dict[str, Any] = {
        "approval": True,
        "cancellation": False,
    }
    if overrides:
        capabilities.update(overrides)
    return capabilities


def _extract_hook_input(payload: JsonValue) -> JsonValue:
    if isinstance(payload, dict) and "input" in payload:
        return payload["input"]
    return payload


def _extract_knowledge_runtime_request(payload: JsonValue) -> KnowledgeRuntimeRequest:
    if isinstance(payload, dict) and "request" in payload:
        return cast(KnowledgeRuntimeRequest, payload["request"])
    return cast(KnowledgeRuntimeRequest, payload)


def _run_service_handler(service: _RegisteredService, request: HostServiceRequest) -> JsonValue:
    response_queue: queue.Queue[JsonValue | BaseException] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            response_queue.put(service.handler(request))
        except BaseException as exc:
            response_queue.put(exc)

    threading.Thread(target=run, daemon=True).start()
    timeout_seconds = (service.request_timeout_ms or 120_000) / 1000
    try:
        response = response_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(f"Host service {service.role}:{service.registry_id} timed out") from exc
    if isinstance(response, BaseException):
        raise response
    return response


def _normalize_host_service_registration_result(
    value: JsonValue,
    service: _RegisteredService,
) -> HostServiceRegistrationResult:
    fallback: HostServiceRegistrationResult = {
        "registered": True,
        "service": {
            "role": service.role,
            "registry_id": service.registry_id,
        },
        "active": True,
    }
    if not isinstance(value, dict):
        return fallback
    service_value = value.get("service")
    if isinstance(service_value, dict):
        role = service_value.get("role", service.role)
        registry_id = service_value.get("registry_id", service.registry_id)
    else:
        role = service.role
        registry_id = service.registry_id
    registered_value = value.get("registered")
    active_value = value.get("active")
    result: HostServiceRegistrationResult = {
        "registered": registered_value if isinstance(registered_value, bool) else True,
        "service": {
            "role": (cast(HarnessServiceRole, role) if isinstance(role, str) else service.role),
            "registry_id": (registry_id if isinstance(registry_id, str) else service.registry_id),
        },
        "active": active_value if isinstance(active_value, bool) else True,
    }
    reason = value.get("reason")
    if isinstance(reason, str) or reason is None:
        result["reason"] = reason
    return result


__all__ = [
    "Harness",
    "HarnessClient",
    "HarnessProtocolError",
    "HostServiceRegistration",
    "HostServiceRegistrationResult",
    "HostServiceRequest",
    "serve_knowledge_runtime_process",
]
