"""AgentPM Python SDK."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentpm")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .core import load

__all__ = ["load", "to_langchain_tool", "__version__"]


# Lazy attribute for optional adapter
# def __getattr__(name: str):
#     if name == "to_langchain_tool":
#         from .adapters.langchain import to_langchain_tool
#
#         return to_langchain_tool
#     raise AttributeError(name)


# import your real implementations once you drop them in:
# from .core import load, to_langchain_tool
