# `agentpm-sdk-python` Repo Guide

This repo is the Python SDK for AgentPM. It owns Python-side tool discovery, subprocess execution, runtime environment handling, typed package ergonomics, and optional framework adapters for Python consumers.

## Local Rules
- Open source.
- PyPI distribution and package shape matter.
- The published contract includes typed package metadata and `py.typed`.
- Optional integrations must remain optional.
- This repo uses Python-native tooling defined in `pyproject.toml` and `uv.lock`.

## Builder Guidance
- Be conservative with SDK contracts.
- Prefer narrow, localized diffs.
- Avoid broad refactors unless the task clearly requires them or the user asked for them.
- Keep runtime behavior predictable and deterministic.
- Preserve subprocess, interpreter, and env semantics carefully.
- Keep error messages useful and specific when contract surfaces fail.
- Update tests and README examples when behavior changes.

## Important Files And Contract Surfaces
- `src/agentpm/core.py`: core `load()` implementation, tool resolution, runtime checks, env merging, subprocess execution, stdout parsing, and error behavior.
- `src/agentpm/__init__.py`: public import surface, package version export, and lazy optional-adapter export behavior.
- `src/agentpm/types.py`: exported typed structures and callable shapes; part of the public SDK contract.
- `src/agentpm/adapters/langchain.py`: optional LangChain adapter surface; must not become a core runtime dependency.
- `src/agentpm/py.typed`: marks the package as typed; part of the typing contract.
- `tests/test_basic.py`: contract-focused tests for loading, env requirements, interpreter allowlists, non-zero exits, and adapter behavior.
- `pyproject.toml`: package metadata, extras, tool configuration, and publish contract.
- `uv.lock`: locked development environment shape for repo contributors.
- `README.md`: user-facing SDK contract and example usage.

## Common Patterns In This Repo
- The public surface is centered on `load()`.
- Tool loading is strict about interpreter allowlists, runtime matching, env requirements, and helpful errors.
- Caller env and entrypoint env are merged intentionally; changes here are contract-sensitive.
- Tool invocation is subprocess-based and exchanges JSON over stdin/stdout.
- Optional integrations like LangChain should remain adapters, not core runtime requirements.
- The Python package is explicitly typed; `types.py`, `py.typed`, and import behavior are part of the public contract.
- `__init__.py` intentionally lazy-loads optional adapter functionality to avoid making optional dependencies feel required.
- Tool stdout may contain logs/noise, but invocation expects a final JSON object to parse as the tool result.
- Tool logs should go to stderr; stdout parsing behavior is contract-sensitive.

## Common Workflows

### When changing tool loading or invocation behavior
1. Start in `src/agentpm/core.py`.
2. Trace the change through resolution order, manifest parsing, interpreter validation, env merging, subprocess execution, and stdout parsing.
3. Preserve discovery order and deterministic behavior unless the task intentionally changes them.
4. Keep error messages explicit when contract validation fails.
5. Update `tests/test_basic.py` to match the changed behavior.

Example:
- The existing tests already cover unsupported interpreters, missing required env vars, and non-zero exit behavior; changes in `load()` should usually extend those tests.

### When changing env handling or runtime validation
1. Start in the env merge and interpreter helper logic in `src/agentpm/core.py`.
2. Treat `os.environ`, manifest entrypoint env, and caller-supplied env as layered contract behavior.
3. Be careful not to weaken required-env enforcement or runtime-type matching by accident.
4. Update tests and README examples if the expected calling pattern changes.

### When changing the public import or typing surface
1. Start in `src/agentpm/__init__.py` and `src/agentpm/types.py`.
2. Treat exported names, lazy adapter access, and typed structures as public API.
3. Preserve `py.typed` and typing expectations unless the task explicitly changes the typing contract.
4. Update README examples if imports, return shapes, or typed usage change.

### When changing the LangChain adapter
1. Start in `src/agentpm/adapters/langchain.py`.
2. Keep adapter assumptions separate from the core runtime path.
3. Preserve the optional extra and lazy import model in `pyproject.toml` and `__init__.py`.
4. Do not let adapter-specific abstractions leak back into `core.py` without a clear reason.

### When changing packaging or toolchain behavior
1. Start with `pyproject.toml`.
2. Preserve package metadata, extras, typing signals, and build behavior unless the task explicitly changes them.
3. Treat `uv.lock`, mypy, ruff, black, and pytest setup as part of the contributor contract.
4. Check whether README installation or import examples need to change.

## Decision Guide

| If the change is about... | Start here | Also inspect |
|---|---|---|
| core SDK loading behavior | `src/agentpm/core.py` | `tests/test_basic.py`, README examples |
| env merging or runtime validation | helper logic in `src/agentpm/core.py` | manifest/runtime assumptions, tests, error messages |
| public imports or typed API | `src/agentpm/__init__.py` and `src/agentpm/types.py` | `py.typed`, README examples, tests |
| LangChain integration | `src/agentpm/adapters/langchain.py` | `pyproject.toml` extras, lazy import behavior, tests, README |
| Python packaging or toolchain behavior | `pyproject.toml` | `uv.lock`, README install examples, typed package shape |
| user-facing SDK usage docs | `README.md` | corresponding code paths and tests |

## Do / Don’t
- Don’t change `load()` behavior casually.
  Do treat it as the primary public API surface and verify the end-to-end contract.
- Don’t weaken subprocess or env semantics by accident.
  Do trace changes through interpreter checks, env merging, stdin/stdout handling, stdout parsing, and exit behavior.
- Don’t let optional adapters become hidden core requirements.
  Do keep adapter-specific logic isolated and package metadata explicit.
- Don’t change exported imports or typed structures casually.
  Do treat import behavior, `types.py`, and `py.typed` as part of the public package contract.
- Don’t change Python packaging metadata casually.
  Do treat extras, build metadata, and typed-package signals as part of the public package contract.
- Don’t let README examples drift from the actual SDK usage model.
  Do update usage docs when imports, options, or calling conventions change.
- Don’t casually change stdout parsing behavior.
  Do preserve the final-JSON result contract and update tests if parsing behavior changes.

## Verification
- Use the repo-native commands defined in `pyproject.toml`, especially `pytest`, `mypy`, `ruff`, and `black`, typically via `uv run ...`.
- Verify load/runtime behavior when changing tool resolution, env handling, subprocess execution, parsing, adapters, imports, or package metadata.
- Treat README examples and typed usage as part of the contract when behavior changes.

## Never Do This
- Never change SDK contracts casually.
- Never regress tool discovery, env passing, stdout parsing, or subprocess behavior without updating tests.
- Never turn optional adapters into implicit core dependencies.
- Never remove or rename exported imports or typed structures without treating it as a compatibility event.
- Never change published package shape or typing signals without treating it as a compatibility event.
