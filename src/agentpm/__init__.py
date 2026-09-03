"""AgentPM Python SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "AfterKnowledgeRetrievalDecision",
    "AfterKnowledgeRetrievalHookHandler",
    "AfterKnowledgeRetrievalInput",
    "AfterKnowledgeRetrievalPatch",
    "AfterKnowledgeRetrievalResultPatch",
    "ApprovalCapabilities",
    "BeforeKnowledgeRequestDecision",
    "BeforeKnowledgeRequestHookHandler",
    "BeforeKnowledgeRequestInput",
    "BeforeKnowledgeRequestPatch",
    "BeforeMemoryOperationDecision",
    "BeforeMemoryOperationHookHandler",
    "BeforeMemoryOperationInput",
    "BeforeMemoryOperationPatch",
    "BeforeMemoryReadDecision",
    "BeforeMemoryReadHookHandler",
    "BeforeMemoryReadInput",
    "BeforeMemoryReadPatch",
    "BeforeMemoryWriteDecision",
    "BeforeMemoryWriteHookHandler",
    "BeforeMemoryWriteInput",
    "BeforeMemoryWritePatch",
    "BeforeModelRequestDecision",
    "BeforeModelRequestHookHandler",
    "BeforeModelRequestInput",
    "BeforeModelRequestPatch",
    "BeforeToolCallDecision",
    "BeforeToolCallHookHandler",
    "BeforeToolCallInput",
    "BeforeToolCallPatch",
    "BeforeToolSelectionDecision",
    "BeforeToolSelectionHookHandler",
    "BeforeToolSelectionInput",
    "BeforeToolSelectionPatch",
    "EmbeddingProviderCapabilities",
    "EmbeddingProviderHandler",
    "EmbeddingProviderRequest",
    "EmbeddingProviderResult",
    "EmbeddingSpaceCapability",
    "Harness",
    "HarnessClient",
    "HarnessProtocolError",
    "HookDecision",
    "HostProviderCapabilities",
    "HostServiceRegistration",
    "HostServiceRegistrationResult",
    "HostServiceRequest",
    "KnowledgeCitation",
    "KnowledgePackageRealization",
    "KnowledgeProviderCapabilities",
    "KnowledgeRequestMode",
    "KnowledgeRetrievalResult",
    "KnowledgeRuntimeFailure",
    "KnowledgeRuntimeHandler",
    "KnowledgeRuntimeRequest",
    "KnowledgeRuntimeResult",
    "MemoryPackageRealization",
    "MemoryProviderCapabilities",
    "ModelProviderCapabilities",
    "__version__",
    "load",
    "load_agent",
    "load_knowledge",
    "load_loop",
    "load_memory",
    "load_memory_contract",
    "load_profile",
    "load_skill",
    "serve_knowledge_runtime_process",
    "to_langchain_tool",
]

# Real exports
from importlib.metadata import PackageNotFoundError, version

from .core import (
    load,
    load_agent,
    load_knowledge,
    load_loop,
    load_memory,
    load_memory_contract,
    load_profile,
    load_skill,
)
from .harness import (
    AfterKnowledgeRetrievalDecision,
    AfterKnowledgeRetrievalHookHandler,
    AfterKnowledgeRetrievalInput,
    AfterKnowledgeRetrievalPatch,
    AfterKnowledgeRetrievalResultPatch,
    ApprovalCapabilities,
    BeforeKnowledgeRequestDecision,
    BeforeKnowledgeRequestHookHandler,
    BeforeKnowledgeRequestInput,
    BeforeKnowledgeRequestPatch,
    BeforeMemoryOperationDecision,
    BeforeMemoryOperationHookHandler,
    BeforeMemoryOperationInput,
    BeforeMemoryOperationPatch,
    BeforeMemoryReadDecision,
    BeforeMemoryReadHookHandler,
    BeforeMemoryReadInput,
    BeforeMemoryReadPatch,
    BeforeMemoryWriteDecision,
    BeforeMemoryWriteHookHandler,
    BeforeMemoryWriteInput,
    BeforeMemoryWritePatch,
    BeforeModelRequestDecision,
    BeforeModelRequestHookHandler,
    BeforeModelRequestInput,
    BeforeModelRequestPatch,
    BeforeToolCallDecision,
    BeforeToolCallHookHandler,
    BeforeToolCallInput,
    BeforeToolCallPatch,
    BeforeToolSelectionDecision,
    BeforeToolSelectionHookHandler,
    BeforeToolSelectionInput,
    BeforeToolSelectionPatch,
    EmbeddingProviderCapabilities,
    EmbeddingProviderHandler,
    EmbeddingProviderRequest,
    EmbeddingProviderResult,
    EmbeddingSpaceCapability,
    Harness,
    HarnessClient,
    HarnessProtocolError,
    HookDecision,
    HostProviderCapabilities,
    HostServiceRegistration,
    HostServiceRegistrationResult,
    HostServiceRequest,
    KnowledgeCitation,
    KnowledgePackageRealization,
    KnowledgeProviderCapabilities,
    KnowledgeRequestMode,
    KnowledgeRetrievalResult,
    KnowledgeRuntimeFailure,
    KnowledgeRuntimeHandler,
    KnowledgeRuntimeRequest,
    KnowledgeRuntimeResult,
    MemoryPackageRealization,
    MemoryProviderCapabilities,
    ModelProviderCapabilities,
    serve_knowledge_runtime_process,
)

try:
    __version__ = version("agentpm")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Tell type checkers that this symbol exists (no runtime import cost)
if TYPE_CHECKING:
    from .adapters.langchain import to_langchain_tool as to_langchain_tool  # re-exported type


# Lazy attribute for optional adapter (runtime)
def __getattr__(name: str) -> Any:
    if name == "to_langchain_tool":
        from .adapters.langchain import to_langchain_tool

        return to_langchain_tool
    raise AttributeError(name)
