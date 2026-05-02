# `agentpm-sdk-python` Repo Review Guide

This repo is the Python runtime integration layer for AgentPM. Reviews should prioritize `load()` contract stability, subprocess/env/runtime correctness, Python package and typing compatibility, and keeping optional adapters isolated from the core runtime.

## Review Focus
- import and runtime correctness
- subprocess behavior
- env passing behavior
- runtime and interpreter validation behavior
- stdout parsing behavior
- public SDK contract stability
- Python packaging and typing compatibility
- optional adapter isolation from core behavior
- README/example drift from actual usage

## Review Principles
- Treat `load()` as the primary public API surface.
- Review runtime changes end-to-end: resolution -> manifest/runtime checks -> env merge -> subprocess invocation -> stdout parsing -> JSON result/error handling.
- Treat package metadata, typed exports, and `py.typed` behavior as compatibility surfaces, not implementation details.
- Prefer findings about contract drift, determinism, compatibility, and runtime correctness over style.

## Blocking Issues
- broken load semantics
- import or path regressions
- env handling regressions
- subprocess contract regressions
- runtime or interpreter validation regressions
- stdout parsing regressions
- unintended public API drift
- package metadata, extras, or typed-package regressions
- adapter changes that make optional integrations effectively required
- stale or misleading README usage examples after behavior changes

## Specific Review Checks
- If `src/agentpm/core.py` changed, review resolution order, interpreter allowlists, env merging, subprocess behavior, stdout parsing, and error wording together.
- If env handling changed, verify required-env enforcement and precedence between process env, manifest env, and caller env.
- If subprocess or parsing behavior changed, review stdin/stdout JSON contract, timeout handling, exit-code behavior, and final-JSON extraction behavior together.
- If `src/agentpm/__init__.py` or `src/agentpm/types.py` changed, verify that exported imports, lazy adapter access, and typed structures remain compatible.
- If `src/agentpm/adapters/langchain.py` changed, verify that the adapter still remains optional and that no core runtime assumptions were introduced.
- If `pyproject.toml` changed, review extras, build metadata, typing signals, and developer-tooling expectations as compatibility surfaces.
- If README usage examples changed, verify that they still match the actual package contract.

## Decision Guide

| If the review touches... | Focus first on... | Also inspect |
|---|---|---|
| core loading behavior | `load()` contract stability | tests, README usage, runtime helper logic |
| env or runtime validation | precedence and determinism | required-env behavior, interpreter checks, tests |
| subprocess or stdout parsing | JSON contract and failure behavior | timeouts, exit codes, stderr/stdout handling |
| public imports or typed API | compatibility of exported names and types | `__init__.py`, `types.py`, `py.typed`, README |
| LangChain adapter | optionality and isolation | extras metadata, lazy import behavior, core runtime boundaries |
| Python packaging or metadata | compatibility and publish shape | `pyproject.toml`, `uv.lock`, README install examples |

## Do / Don’t
- Don’t review helper changes in isolation when they affect `load()`.
  Do follow them through the full runtime contract and its tests.
- Don’t treat env, subprocess, or stdout parsing behavior as incidental implementation detail.
  Do review them as caller-visible SDK contract behavior.
- Don’t accept adapter changes that quietly reshape the core SDK.
  Do enforce the boundary between optional integrations and core runtime behavior.
- Don’t ignore import or typing-surface changes as internal cleanup.
  Do treat exported imports, typed structures, and `py.typed` behavior as compatibility-sensitive public API changes.
- Don’t ignore packaging metadata changes as build noise.
  Do treat extras, build metadata, and typed-package signals as compatibility-sensitive surfaces.

## Low-Value Review Noise
- avoid style-only comments unless they hide compatibility, typing, or runtime risk
- avoid speculative refactor suggestions unless the current change clearly weakens determinism or maintainability
