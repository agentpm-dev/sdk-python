"""AgentPM Python SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["load", "to_langchain_tool", "__version__"]

# Real exports
from importlib.metadata import PackageNotFoundError, version

from .core import load

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
