from __future__ import annotations

import codecs
import contextlib
import json
import os
import queue
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast, overload

from semver import VersionInfo
from semver import match as semver_match

from .types import (
    AgentMeta,
    DependencyReference,
    Entrypoint,
    JsonValue,
    KnowledgeMeta,
    LoadedAgent,
    LoadedKnowledge,
    LoadedMemory,
    LoadedMemoryContractRef,
    LoadedSkill,
    LoadedWithMeta,
    Manifest,
    MemoryBuildMetadata,
    MemoryContractIndex,
    MemoryContractSchema,
    MemoryMeta,
    MemoryMetadata,
    ReservedReferences,
    ResolvedAgentKnowledgeRef,
    ResolvedAgentMemoryRef,
    ResolvedAgentSkillRef,
    ResolvedAgentToolRef,
    Runtime,
    SkillMeta,
    ToolFunc,
    ToolMeta,
)

READ_CHUNK = 65536
MAX_BYTES = 10 * 1024 * 1024  # 10 MB cap across stdout+stderr
GRACE_AFTER_JSON = 0.40  # seconds to let the child exit after JSON seen

DEFAULT_TIMEOUT = 120.0
_ALLOWED = {"node", "nodejs", "python", "python3"}


def _debug_enabled() -> bool:
    val = os.getenv("AGENTPM_DEBUG", "")
    return val not in ("", "0", "false", "False", "no")


def _dprint(msg: str) -> None:
    if _debug_enabled():
        sys.stderr.write(f"[agentpm-debug] {msg}\n")


def _abbrev(s: str, n: int = 240) -> str:
    return s if len(s) <= n else (s[:n] + "…")


def _merge_env(
    entry_env: dict[str, str] | None,
    caller_env: dict[str, str] | None,
) -> dict[str, str]:
    merged = os.environ.copy()
    if entry_env:
        merged.update(entry_env)
    if caller_env:
        merged.update(caller_env)
    return merged


def _canonical(cmd: str) -> str:
    # handle absolute paths and Windows extensions
    base = os.path.basename(cmd).lower()
    for ext in (".exe", ".cmd", ".bat"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def _interpreter_family(cmd: str) -> str | None:
    base = os.path.basename(cmd).lower()
    if base in ("node", "nodejs"):
        return "node"
    if base.startswith("python"):
        return "python"
    return None  # absolute paths still get matched by basename


def _resolve_interpreter_command(
    cmd: str,
    entry_env: dict[str, str] | None,
    caller_env: dict[str, str] | None,
    runtime_type: str | None,
) -> str:
    merged = _merge_env(entry_env, caller_env)

    # Prefer inferring from the command; fall back to runtime hint if needed
    inferred = _interpreter_family(cmd)
    hint = runtime_type if runtime_type == "node" or runtime_type == "python" else None
    family = inferred or hint or None

    if family == "node" and merged.get("AGENTPM_NODE"):
        _dprint(f'override interpreter (node): "{cmd}" -> "{merged["AGENTPM_NODE"]}"')
        return merged["AGENTPM_NODE"]
    if family == "python" and merged.get("AGENTPM_PYTHON"):
        _dprint(f'override interpreter (python): "{cmd}" -> "{merged["AGENTPM_PYTHON"]}"')
        return merged["AGENTPM_PYTHON"]
    return cmd


def _assert_allowed_interpreter(cmd: str) -> None:
    canon = _canonical(cmd)
    if canon not in _ALLOWED and not canon.startswith("python3"):
        raise ValueError(
            f'Unsupported agent.json.entrypoint.command "{cmd}". Allowed: node|nodejs|python|python3'
        )


# verify the interpreter exists on PATH
def _assert_interpreter_available(
    cmd: str, entry_env: dict[str, str] | None, caller_env: dict[str, str] | None
) -> None:
    merged = _merge_env(entry_env, caller_env)

    which = shutil.which(cmd, path=merged.get("PATH", ""))
    _dprint(f'interpreter="{cmd}" which={which or "<not found>"}')
    _dprint(f'MERGED PATH={_abbrev(merged.get("PATH",""))}')

    if which is None:
        raise FileNotFoundError(
            f'Interpreter "{cmd}" not found on PATH.\nChecked PATH={merged.get("PATH","")}'
        )


def _assert_interpreter_matches_runtime(cmd: str, runtime: Runtime) -> None:
    canon = _canonical(cmd)
    runtime_interpreter = _canonical(runtime["type"])

    if not is_interpreter_match(runtime_interpreter, canon):
        raise ValueError(
            f'Misconfigured tool - agent.json.entrypoint.command "{cmd}" does not match tool runtime {runtime_interpreter}'
        )


def is_interpreter_match(runtime: str, command: str) -> bool:
    if runtime == command:
        return True

    # runtime -> acceptable command aliases
    aliases = {"python": ["python3"], "node": ["nodejs"]}

    return command in aliases.get(runtime, [])


def _list_installed_versions(base: Path, name: str) -> list[str]:
    """Return all installed x.y.z versions for a tool name, searching all name dir variants."""
    seen: set[str] = set()

    for name_dir in candidate_name_dirs(str(base), name):
        root = Path(name_dir)
        if not root.is_dir():
            continue

        for child in root.iterdir():
            if not child.is_dir():
                continue

            v = child.name
            try:
                # validate semver
                VersionInfo.parse(v)
            except ValueError:
                continue

            if (child / "agent.json").exists():
                seen.add(v)

    # highest first
    return sorted(seen, key=VersionInfo.parse, reverse=True)


def candidate_name_dirs(base: str, name: str) -> list[str]:
    """
    Supports names like "@scope/name" or "scope/name".
    Tries:
      base/@scope/name, base/scope/name, base/scope__name, base/scope-name
    Falls back to base/name for unscoped.
    """
    parts = name.split("/")

    if len(parts) == 2:
        raw_scope, pkg = parts
        scope = raw_scope[1:] if raw_scope.startswith("@") else raw_scope
        return [
            os.path.join(base, f"@{scope}", pkg),  # with '@'
            os.path.join(base, scope, pkg),  # without '@'
            os.path.join(base, f"{scope}__{pkg}"),
            os.path.join(base, f"{scope}-{pkg}"),
        ]

    # Unscoped package
    return [os.path.join(base, name)]


def _find_installed(base: Path, name: str, version: str) -> tuple[Path, Path] | None:
    """Return (root, manifest_path) if this exact version exists, searching all name dir variants."""
    for name_dir in candidate_name_dirs(str(base), name):
        root = Path(name_dir) / version
        manifest = root / "agent.json"
        if manifest.exists():
            return root, manifest
    return None


def find_project_root(start_dir: str | Path) -> Path:
    """
    Walk up from start_dir looking for project markers.
    Priority: agent.json, pyproject.toml, package.json, pnpm-workspace.yaml, turbo.json, lerna.json, .git
    Returns the resolved start_dir if nothing is found.
    """
    dir_path = Path(start_dir).resolve()
    while True:
        if (dir_path / "agent.json").exists():
            return dir_path
        if (dir_path / "pyproject.toml").exists():
            return dir_path
        if (dir_path / "package.json").exists():
            return dir_path
        if (dir_path / "pnpm-workspace.yaml").exists():
            return dir_path
        if (dir_path / "turbo.json").exists():
            return dir_path
        if (dir_path / "lerna.json").exists():
            return dir_path
        if (dir_path / ".git").exists():
            return dir_path

        parent = dir_path.parent
        if parent == dir_path:  # reached filesystem root
            break
        dir_path = parent

    return Path(start_dir).resolve()


def _normalize_selector(selector: str) -> str:
    s = selector.strip()
    if not s or s.lower() == "latest":
        return ""

    def parts(ver: str) -> tuple[int, int, int, int]:
        xs = [p for p in ver.strip().split(".") if p != ""]
        n = len(xs)
        maj = int(xs[0]) if n >= 1 else 0
        min_ = int(xs[1]) if n >= 2 else 0
        pat = int(xs[2]) if n >= 3 else 0
        return maj, min_, pat, n

    if s[0] in ("^", "~"):
        op, base = s[0], s[1:].strip()
        maj, min_, pat, n = parts(base)
        lower = f">={maj}.{min_}.{pat}"
        if op == "^":
            if maj > 0:
                upper = f"<{maj+1}.0.0"
            elif n == 1:
                upper = "<1.0.0"  # ^0
            elif min_ > 0:
                upper = f"<0.{min_+1}.0"  # ^0.y
            else:
                upper = f"<0.0.{pat+1}"  # ^0.0.z
        else:  # '~'
            upper = f"<{maj + 1}.0.0" if n == 1 else f"<{maj}.{min_ + 1}.0"
        # return space-separated; we'll split on spaces/commas later
        return f"{lower} {upper}"

    # Comparator set like ">=0.1.1 <0.2.0" (or commas) → normalize whitespace
    tokens = [t for t in s.replace(",", " ").split() if t]
    return " ".join(tokens)


def _version_satisfies(ver: str, selector: str) -> bool:
    expr = _normalize_selector(selector)
    if not expr:  # empty / "latest"
        return True
    # Split on spaces or commas
    tokens = [t for t in re.split(r"[,\s]+", expr) if t]
    try:
        return all(semver_match(ver, tok) for tok in tokens)
    except ValueError:
        return False


def _resolve_tool_root(spec: str, tool_dir_override: str | None) -> tuple[Path, Path]:
    # spec form: @scope/name@<version or range or 'latest'>
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f'Invalid tool spec "{spec}". Expected "@scope/name@version".')

    selector = spec[at + 1 :].strip()

    raw_name = spec[:at]
    name = raw_name[1:] if raw_name.startswith("@") else raw_name  # drop leading '@' if present

    project_root = find_project_root(Path.cwd())
    _dprint(f"project_root={project_root}")

    # candidate search roots (project first)
    candidates: list[Path] = []
    if tool_dir_override:
        candidates.append(Path(tool_dir_override))

    env_dir = os.getenv("AGENTPM_TOOL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(project_root / ".agentpm" / "tools")
    candidates.append(Path.home() / ".agentpm" / "tools")

    _dprint("candidates:\n  " + "\n  ".join(str(c) for c in candidates))

    # 1) Exact version fast path
    try:
        if selector and selector.lower() != "latest":
            VersionInfo.parse(selector)  # raises if not exact x.y.z
            for base in candidates:
                hit = _find_installed(base, name, selector)
                if hit:
                    return hit
            raise FileNotFoundError(f'Tool "{spec}" not found in .agentpm/tools (or overrides).')
    except ValueError:
        # not an exact version → fall through to range/latest
        pass

    # 2) Range or "latest" (or empty after "@")
    want_latest = (not selector) or (selector.lower() == "latest")

    for base in candidates:
        installed = _list_installed_versions(base, name)
        if not installed:
            continue

        if want_latest:
            picked = installed[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit
            continue

        # Filter by range using semver.match, then pick highest
        satisfying: list[str] = []
        for v in installed:
            if _version_satisfies(v, selector):
                satisfying.append(v)

        if satisfying:
            picked = sorted(satisfying, key=VersionInfo.parse, reverse=True)[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f'No installed version of "{name}" matches "{selector or "latest"}". Searched: {searched}'
    )


def _resolve_agent_root(spec: str, agent_dir_override: str | None) -> tuple[Path, Path]:
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f'Invalid agent spec "{spec}". Expected "@scope/name@version".')

    selector = spec[at + 1 :].strip()
    name = spec[:at]

    project_root = find_project_root(Path.cwd())

    candidates: list[Path] = []
    if agent_dir_override:
        candidates.append(Path(agent_dir_override))

    env_dir = os.getenv("AGENTPM_AGENT_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(project_root / ".agentpm" / "agents")
    candidates.append(Path.home() / ".agentpm" / "agents")

    try:
        if selector and selector.lower() != "latest":
            VersionInfo.parse(selector)
            for base in candidates:
                hit = _find_installed(base, name, selector)
                if hit:
                    return hit
            raise FileNotFoundError(f'Agent "{spec}" not found in .agentpm/agents (or overrides).')
    except ValueError:
        pass

    want_latest = (not selector) or (selector.lower() == "latest")

    for base in candidates:
        installed = _list_installed_versions(base, name)
        if not installed:
            continue

        if want_latest:
            picked = installed[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit
            continue

        satisfying = [v for v in installed if _version_satisfies(v, selector)]
        if satisfying:
            picked = sorted(satisfying, key=VersionInfo.parse, reverse=True)[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f'No installed version of "{name}" matches "{selector or "latest"}". Searched: {searched}'
    )


def _resolve_skill_root(spec: str, skill_dir_override: str | None) -> tuple[Path, Path]:
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f'Invalid skill spec "{spec}". Expected "@scope/name@version".')

    selector = spec[at + 1 :].strip()
    name = spec[:at]

    project_root = find_project_root(Path.cwd())

    candidates: list[Path] = []
    if skill_dir_override:
        candidates.append(Path(skill_dir_override))

    env_dir = os.getenv("AGENTPM_SKILL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(project_root / ".agentpm" / "skills")
    candidates.append(Path.home() / ".agentpm" / "skills")

    try:
        if selector and selector.lower() != "latest":
            VersionInfo.parse(selector)
            for base in candidates:
                hit = _find_installed(base, name, selector)
                if hit:
                    return hit
            raise FileNotFoundError(f'Skill "{spec}" not found in .agentpm/skills (or overrides).')
    except ValueError:
        pass

    want_latest = (not selector) or (selector.lower() == "latest")

    for base in candidates:
        installed = _list_installed_versions(base, name)
        if not installed:
            continue

        if want_latest:
            picked = installed[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit
            continue

        satisfying = [v for v in installed if _version_satisfies(v, selector)]
        if satisfying:
            picked = sorted(satisfying, key=VersionInfo.parse, reverse=True)[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f'No installed version of "{name}" matches "{selector or "latest"}". Searched: {searched}'
    )


def _resolve_knowledge_root(spec: str, knowledge_dir_override: str | None) -> tuple[Path, Path]:
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f'Invalid knowledge spec "{spec}". Expected "@scope/name@version".')

    selector = spec[at + 1 :].strip()
    name = spec[:at]

    project_root = find_project_root(Path.cwd())

    candidates: list[Path] = []
    if knowledge_dir_override:
        candidates.append(Path(knowledge_dir_override))

    env_dir = os.getenv("AGENTPM_KNOWLEDGE_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(project_root / ".agentpm" / "knowledge")
    candidates.append(Path.home() / ".agentpm" / "knowledge")

    try:
        if selector and selector.lower() != "latest":
            VersionInfo.parse(selector)
            for base in candidates:
                hit = _find_installed(base, name, selector)
                if hit:
                    return hit
            raise FileNotFoundError(
                f'Knowledge package "{spec}" not found in .agentpm/knowledge (or overrides).'
            )
    except ValueError:
        pass

    want_latest = (not selector) or (selector.lower() == "latest")

    for base in candidates:
        installed = _list_installed_versions(base, name)
        if not installed:
            continue

        if want_latest:
            picked = installed[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit
            continue

        satisfying = [v for v in installed if _version_satisfies(v, selector)]
        if satisfying:
            picked = sorted(satisfying, key=VersionInfo.parse, reverse=True)[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f'No installed version of "{name}" matches "{selector or "latest"}". Searched: {searched}'
    )


def _resolve_memory_root(spec: str, memory_dir_override: str | None) -> tuple[Path, Path]:
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f'Invalid memory spec "{spec}". Expected "@scope/name@version".')

    selector = spec[at + 1 :].strip()
    name = spec[:at]

    project_root = find_project_root(Path.cwd())

    candidates: list[Path] = []
    if memory_dir_override:
        candidates.append(Path(memory_dir_override))

    env_dir = os.getenv("AGENTPM_MEMORY_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(project_root / ".agentpm" / "memory")
    candidates.append(Path.home() / ".agentpm" / "memory")

    try:
        if selector and selector.lower() != "latest":
            VersionInfo.parse(selector)
            for base in candidates:
                hit = _find_installed(base, name, selector)
                if hit:
                    return hit
            raise FileNotFoundError(
                f'Memory package "{spec}" not found in .agentpm/memory (or overrides).'
            )
    except ValueError:
        normalized = _normalize_selector(selector)
        try:
            _ = all(semver_match("0.0.0", token) for token in normalized.split() if token)
        except ValueError:
            raise ValueError(
                f'Invalid version/range "{selector}". Use exact (e.g. 0.1.2), '
                'a semver range (e.g. ^0.1), or "latest".'
            ) from None

    want_latest = (not selector) or (selector.lower() == "latest")

    for base in candidates:
        installed = _list_installed_versions(base, name)
        if not installed:
            continue

        if want_latest:
            picked = installed[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit
            continue

        satisfying = [v for v in installed if _version_satisfies(v, selector)]
        if satisfying:
            picked = sorted(satisfying, key=VersionInfo.parse, reverse=True)[0]
            hit = _find_installed(base, name, picked)
            if hit:
                return hit

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f'No installed version of "{name}" matches "{selector or "latest"}". Searched: {searched}'
    )


def _read_manifest(p: Path) -> Manifest:
    m = json.loads(p.read_text(encoding="utf-8"))
    ep = m.get("entrypoint", {})
    if not ep or not ep.get("command"):
        raise ValueError(f"agent.json missing entrypoint.command at: {p}")
    return m  # type: ignore[no-any-return]


def _read_agent_manifest(p: Path) -> AgentMeta:
    manifest = cast(AgentMeta, json.loads(p.read_text(encoding="utf-8")))
    if manifest.get("kind") != "agent":
        raise ValueError(f"agent.json is not an agent manifest at: {p}")
    return manifest


def _read_skill_manifest(p: Path) -> SkillMeta:
    manifest = cast(SkillMeta, json.loads(p.read_text(encoding="utf-8")))
    if manifest.get("kind") != "skill":
        raise ValueError(f"agent.json is not a skill manifest at: {p}")
    if not manifest.get("skill", {}).get("entrypoint"):
        raise ValueError(f"agent.json missing skill.entrypoint at: {p}")
    return manifest


def _read_knowledge_manifest(p: Path) -> KnowledgeMeta:
    manifest = cast(KnowledgeMeta, json.loads(p.read_text(encoding="utf-8")))
    if manifest.get("kind") != "knowledge":
        raise ValueError(f"agent.json is not a knowledge manifest at: {p}")
    if not manifest.get("knowledge", {}).get("mode"):
        raise ValueError(f"agent.json missing knowledge.mode at: {p}")
    return manifest


def _read_memory_manifest(p: Path) -> MemoryMeta:
    manifest = cast(MemoryMeta, json.loads(p.read_text(encoding="utf-8")))
    if manifest.get("kind") != "memory":
        raise ValueError(f"agent.json is not a memory manifest at: {p}")
    memory = manifest.get("memory")
    if not isinstance(memory, dict):
        raise ValueError(f"agent.json missing memory object at: {p}")
    return manifest


# Historical name: this helper now accepts modern agent.lock shapes v2 and v3.
# Keep the narrower name for now to avoid internal churn while Skills remain the
# only v3-specific addition on top of the same overall lock envelope.
def _read_lockfile_v2(lockfile_path: Path) -> dict[str, Any]:
    lock = cast(dict[str, Any], json.loads(lockfile_path.read_text(encoding="utf-8")))
    if lock.get("lockfile_version") not in (2, 3):
        raise ValueError(
            f'Unsupported lockfile version at {lockfile_path}; expected agent.lock v2 or v3. Run "agentpm install" to regenerate the lockfile.'
        )
    return lock


def _resolve_agent_lockfile_path(lockfile_override: str | None) -> Path:
    if lockfile_override:
        return Path(lockfile_override)
    return find_project_root(Path.cwd()) / "agent.lock"


def _empty_reserved_references() -> ReservedReferences:
    return {
        "knowledge": [],
        "memory": [],
        "profiles": [],
    }


def _is_safe_relative_path(value: str) -> bool:
    if not value or os.path.isabs(value):
        return False
    normalized = Path(value.replace("\\", "/"))
    return all(part not in ("..", "") for part in normalized.parts) and str(normalized) != "."


def _resolve_installed_memory_file(
    root: Path,
    relative_path: str,
    field_label: str,
    *,
    required_prefix: str | None = None,
) -> Path:
    if not _is_safe_relative_path(relative_path):
        raise ValueError(f"{field_label} must be a safe package-relative path.")

    normalized = Path(relative_path.replace("\\", "/"))
    normalized_text = normalized.as_posix()
    if required_prefix and not normalized_text.startswith(required_prefix):
        raise ValueError(f"{field_label} must remain under {required_prefix}.")

    root_real = root.resolve()
    target = (root / normalized).resolve()
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"{field_label} is missing at {normalized_text}.")

    with suppress(ValueError):
        target.relative_to(root_real)
        return target
    raise ValueError(f"{field_label} resolves outside the installed memory package root.")


def _read_json_file(path: Path, field_label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_label} must be a JSON object.")
    return cast(dict[str, Any], value)


def _read_memory_build_metadata(root: Path) -> tuple[Path, MemoryBuildMetadata]:
    build_path = _resolve_installed_memory_file(root, "memory/build.json", "memory/build.json")
    build_json = _read_json_file(build_path, "memory/build.json")

    if build_json.get("type") != "agentpm-memory-contracts":
        raise ValueError("memory/build.json has unsupported type.")
    if build_json.get("format_version") != 1:
        raise ValueError("memory/build.json has unsupported format_version.")
    if not isinstance(build_json.get("manifest_path"), str) or not build_json["manifest_path"]:
        raise ValueError("memory/build.json missing manifest_path.")
    for field in (
        "source_manifest_hash",
        "source_schemas_hash",
        "source_contract_inputs_hash",
        "contracts_index_hash",
        "contracts_hash",
    ):
        if not isinstance(build_json.get(field), str) or not build_json[field]:
            raise ValueError(f"memory/build.json missing {field}.")
    contract_count = build_json.get("contract_count")
    if not isinstance(contract_count, int):
        raise ValueError("memory/build.json missing contract_count.")

    source_schemas = build_json.get("source_schemas")
    if source_schemas is not None:
        if not isinstance(source_schemas, list):
            raise ValueError("memory/build.json has invalid source_schemas entries.")
        for entry in source_schemas:
            if not isinstance(entry, dict):
                raise ValueError("memory/build.json has invalid source_schemas entries.")
            if not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
                raise ValueError("memory/build.json has invalid source_schemas entries.")

    return build_path, cast(MemoryBuildMetadata, build_json)


def _read_memory_contract_index(
    root: Path, memory: MemoryMetadata, expected_contract_count: int
) -> tuple[Path, MemoryContractIndex, list[str], list[LoadedMemoryContractRef]]:
    index_path = _resolve_installed_memory_file(
        root,
        "memory/contracts/index.json",
        "memory/contracts/index.json",
        required_prefix="memory/contracts/",
    )
    index_json = _read_json_file(index_path, "memory/contracts/index.json")

    if index_json.get("type") != "agentpm-memory-contract-index":
        raise ValueError("memory/contracts/index.json has unsupported type.")
    if index_json.get("format_version") != 1:
        raise ValueError("memory/contracts/index.json has unsupported format_version.")

    contracts = index_json.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("memory/contracts/index.json missing contracts array.")
    if len(contracts) != expected_contract_count:
        raise ValueError(
            "memory/build.json contract_count does not match memory/contracts/index.json."
        )

    seen_identities: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    source_schema_paths: set[str] = set()
    resolved_contracts: list[LoadedMemoryContractRef] = []

    for index, entry in enumerate(contracts):
        if not isinstance(entry, dict):
            raise ValueError(
                f"memory/contracts/index.json contract entry {index} must be an object."
            )
        space = entry.get("space")
        record_type = entry.get("record_type")
        schema_version = entry.get("schema_version")
        model = entry.get("model")
        source_schema = entry.get("source_schema")
        contract_path = entry.get("path")
        sha256 = entry.get("sha256")
        if not all(
            isinstance(value, str)
            for value in (
                space,
                record_type,
                schema_version,
                model,
                source_schema,
                contract_path,
                sha256,
            )
        ):
            raise ValueError(
                f"memory/contracts/index.json contract entry {index} is missing required fields."
            )

        record_type_manifest = memory["record_types"].get(cast(str, record_type))
        if (
            not isinstance(record_type_manifest, dict)
            or record_type_manifest.get("schema") != source_schema
        ):
            raise ValueError(
                f'memory/contracts/index.json references undeclared source schema "{source_schema}".'
            )

        identity = (cast(str, space), cast(str, record_type))
        if identity in seen_identities:
            raise ValueError(
                f'memory/contracts/index.json contains duplicate contract entry "{identity[0]}:{identity[1]}".'
            )
        seen_identities.add(identity)
        if cast(str, contract_path) in seen_paths:
            raise ValueError(
                f'memory/contracts/index.json contains duplicate contract path "{contract_path}".'
            )
        seen_paths.add(cast(str, contract_path))

        resolved_contract_path = _resolve_installed_memory_file(
            root,
            cast(str, contract_path),
            f'memory/contracts/index.json contract path "{contract_path}"',
            required_prefix="memory/contracts/",
        )
        resolved_source_schema_path = _resolve_installed_memory_file(
            root,
            cast(str, source_schema),
            f'memory/contracts/index.json source schema "{source_schema}"',
        )
        source_schema_paths.add(str(resolved_source_schema_path))
        resolved_contracts.append(
            {
                "space": cast(str, space),
                "recordType": cast(str, record_type),
                "schemaVersion": cast(str, schema_version),
                "model": cast(str, model),
                "sourceSchemaPath": str(resolved_source_schema_path),
                "path": str(resolved_contract_path),
                "sha256": cast(str, sha256),
            }
        )

    return (
        index_path,
        cast(MemoryContractIndex, index_json),
        sorted(source_schema_paths),
        resolved_contracts,
    )


def _resolve_tool_installed_path(
    name: str, version: str, tool_dir_override: str | None
) -> tuple[Path, Path] | None:
    project_root = find_project_root(Path.cwd())
    candidates: list[Path] = []
    if tool_dir_override:
        candidates.append(Path(tool_dir_override))

    env_dir = os.getenv("AGENTPM_TOOL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(project_root / ".agentpm" / "tools")
    candidates.append(Path.home() / ".agentpm" / "tools")

    for base in candidates:
        hit = _find_installed(base, name, version)
        if hit:
            return hit
    return None


def _resolve_skill_installed_path(
    name: str, version: str, skill_dir_override: str | None
) -> tuple[Path, Path] | None:
    project_root = find_project_root(Path.cwd())
    candidates: list[Path] = []
    if skill_dir_override:
        candidates.append(Path(skill_dir_override))

    env_dir = os.getenv("AGENTPM_SKILL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(project_root / ".agentpm" / "skills")
    candidates.append(Path.home() / ".agentpm" / "skills")

    for base in candidates:
        hit = _find_installed(base, name, version)
        if hit:
            return hit
    return None


def _resolve_knowledge_installed_path(
    name: str, version: str, knowledge_dir_override: str | None
) -> tuple[Path, Path] | None:
    project_root = find_project_root(Path.cwd())
    candidates: list[Path] = []
    if knowledge_dir_override:
        candidates.append(Path(knowledge_dir_override))

    env_dir = os.getenv("AGENTPM_KNOWLEDGE_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(project_root / ".agentpm" / "knowledge")
    candidates.append(Path.home() / ".agentpm" / "knowledge")

    for base in candidates:
        hit = _find_installed(base, name, version)
        if hit:
            return hit
    return None


def _resolve_memory_installed_path(
    name: str, version: str, memory_dir_override: str | None
) -> tuple[Path, Path] | None:
    project_root = find_project_root(Path.cwd())
    candidates: list[Path] = []
    if memory_dir_override:
        candidates.append(Path(memory_dir_override))

    env_dir = os.getenv("AGENTPM_MEMORY_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(project_root / ".agentpm" / "memory")
    candidates.append(Path.home() / ".agentpm" / "memory")

    for base in candidates:
        hit = _find_installed(base, name, version)
        if hit:
            return hit
    return None


def _build_env(
    entry_env: dict[str, str], caller_env: dict[str, str], home: str, tmpdir: str
) -> dict[str, str]:
    base = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": home,
        "TMPDIR": tmpdir,
    }
    # Optional: preserve locale if present
    for k in ("LANG", "LC_ALL"):
        if k in os.environ:
            base[k] = os.environ[k]
    # Agent-provided env wins, then caller overrides
    return {**base, **entry_env, **caller_env}


def _preexec_rlimits_for(
    cmd: str,
    *,
    max_cpu_s: int | None = 10,
    max_files: int | None = 512,
    max_addr_mb: int | None = 512,
) -> Callable[[], None]:
    """
    Apply rlimits safely per interpreter.

    - Node (node/nodejs): SKIP RLIMIT_AS by default (V8 JIT/WASM need large VA space).
      You can force a value with env AGENTPM_RLIMIT_AS_MB.
    - Python: keep modest RLIMIT_AS if you want.
    """
    import os
    import resource  # type: ignore

    IS_DARWIN = os.uname().sysname == "Darwin"
    fam = _canonical(cmd)
    is_node = fam in ("node", "nodejs")

    # Optional global override
    env_override = os.getenv("AGENTPM_RLIMIT_AS_MB")
    addr_mb = max_addr_mb
    if env_override:
        with suppress(ValueError):
            parsed = int(env_override)
            if parsed > 0:
                addr_mb = parsed

    # Default: do NOT cap address space for Node
    if is_node and env_override is None:
        addr_mb = None

    _dprint(
        f"rlimits: cmd={cmd} RLIMIT_AS={'off' if is_node and env_override is None else addr_mb}MB"
    )

    def _fn() -> None:
        if max_cpu_s is not None and hasattr(resource, "RLIMIT_CPU"):
            resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_s, max_cpu_s))
        if max_files is not None and hasattr(resource, "RLIMIT_NOFILE"):
            resource.setrlimit(resource.RLIMIT_NOFILE, (max_files, max_files))
        if addr_mb is not None and not IS_DARWIN and hasattr(resource, "RLIMIT_AS"):
            limit = addr_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return _fn


def _spawn_once(
    root: Path, entry: Entrypoint, payload: JsonValue, timeout_s: float, env: dict[str, str]
) -> JsonValue:
    """
    Cross-platform:
      - Spawns child, writes JSON payload, closes stdin.
      - Reads stdout/stderr in background threads (binary), decodes incrementally.
      - Detects first complete JSON object in stdout.
      - After JSON is seen, gives a small grace window, then terminates/kills if needed.
      - Enforces timeout and 10MB total output cap.
    Returns: (parsed_json, full_stdout_text, full_stderr_text)
    Raises: TimeoutError or RuntimeError on failures.
    """
    # 1) Tool working dir (what the tool expects for relative paths)
    tool_cwd = (root / entry.get("cwd", ".")).resolve()

    # 2) Isolated run dirs for HOME/TMPDIR
    run_root: Path = tool_cwd / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=str(run_root)))
    home = str(work / "home")
    Path(home).mkdir(parents=True, exist_ok=True)
    tmpd = str(work / "tmp")
    Path(tmpd).mkdir(parents=True, exist_ok=True)

    # 3) Clean env
    env = _build_env(entry.get("env", {}), env, home, tmpd)

    # 4) Command + hardening flags
    cmd = [entry["command"], *entry.get("args", [])]
    if _canonical(entry["command"]).startswith("python"):
        if "-I" not in cmd:
            cmd.insert(1, "-I")
        if "-B" not in cmd:
            cmd.insert(1, "-B")
    elif _canonical(entry["command"]).startswith("node"):
        old_space = int(os.getenv("AGENTPM_NODE_OLD_SPACE_MB", "256"))

        if not any(a.startswith("--max-old-space-size") for a in cmd[1:]):
            cmd.insert(1, f"--max-old-space-size={old_space}")

        want_jitless = (
            any(a == "--jitless" for a in cmd[1:])
            or "--jitless" in (env or {}).get("NODE_OPTIONS", "")
            or os.getenv("AGENTPM_NODE_JITLESS", "").lower() in ("1", "true", "yes")
        )
        if want_jitless and "--jitless" not in cmd[1:]:
            cmd.insert(1, "--jitless")

    # Add -u for Python children (unbuffered); safe no-op for non-Python
    # insert after interpreter
    if cmd and os.path.basename(cmd[0]).lower().startswith("python") and "-u" not in cmd:
        cmd = [cmd[0], "-u", *cmd[1:]]

    _dprint(f"launch: argv={cmd}")
    _dprint(f"cwd={tool_cwd}")
    # _dprint(f"env={env}")

    # 5) Spawn (cwd = tool_cwd)
    proc = subprocess.Popen(
        cmd,
        cwd=str(tool_cwd),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # binary pipes
        bufsize=0,
        start_new_session=(os.name == "posix"),
        preexec_fn=(
            _preexec_rlimits_for(entry["command"]) if hasattr(resource, "setrlimit") else None
        ),
        close_fds=True,
    )

    assert proc.stdout and proc.stderr

    with contextlib.suppress(Exception):
        data = json.dumps(payload).encode("utf-8")
        assert proc.stdin is not None
        proc.stdin.write(data)
        proc.stdin.flush()
        proc.stdin.close()

    # Queues for bytes
    q_out: queue.Queue[bytes | None] = queue.Queue()
    q_err: queue.Queue[bytes | None] = queue.Queue()

    def reader(src, q: queue.Queue[bytes | None]):
        try:
            while True:
                b = src.read(READ_CHUNK)
                if not b:
                    break
                q.put(b)
        finally:
            q.put(None)

    t_out = threading.Thread(target=reader, args=(proc.stdout, q_out), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, q_err), daemon=True)
    t_out.start()
    t_err.start()

    # Incremental decoders
    dec_out = codecs.getincrementaldecoder("utf-8")()
    dec_err = codecs.getincrementaldecoder("utf-8")()

    out_parts: list[str] = []
    err_parts: list[str] = []
    total_bytes = 0

    parsed: dict[str, Any] | None = None
    got_json_at: float | None = None
    sent_term = sent_kill = False
    deadline = time.monotonic() + timeout_s

    # Main loop: drain queues, detect JSON, apply grace-and-kill
    while True:
        now = time.monotonic()
        if now > deadline:
            _kill_proc(proc, signal.SIGKILL)
            raise TimeoutError(f"Tool timed out after {timeout_s:.1f}s")

        drained = False

        # Drain stdout queue
        while True:
            try:
                b = q_out.get_nowait()
            except queue.Empty:
                break

            drained = True
            if b is None:
                # EOF marker for stdout
                pass
            else:
                total_bytes += len(b)
                if total_bytes > MAX_BYTES:
                    _kill_proc(proc, signal.SIGKILL)
                    raise RuntimeError("Tool produced too much output; 10MB limit")
                chunk = dec_out.decode(b)
                out_parts.append(chunk)
                if parsed is None:
                    obj, _slice, _s, _e = _try_extract_json("".join(out_parts))
                    if obj is not None:
                        parsed = obj
                        got_json_at = time.monotonic()

            q_out.task_done()

        # Drain stderr queue
        while True:
            try:
                b = q_err.get_nowait()
            except queue.Empty:
                break
            drained = True
            if b is None:
                pass
            else:
                total_bytes += len(b)
                if total_bytes > MAX_BYTES:
                    _kill_proc(proc, signal.SIGKILL)
                    raise RuntimeError("Tool produced too much output; 10MB limit")
                err_parts.append(dec_err.decode(b))
            q_err.task_done()

        # If we've seen JSON, give a grace window, then TERM→KILL if still alive
        if parsed is not None and got_json_at is not None and proc.poll() is None:
            if (now - got_json_at) > GRACE_AFTER_JSON and not sent_term:
                _kill_proc(proc, signal.SIGTERM)
                sent_term = True
                term_at = now
            elif sent_term and not sent_kill and (now - term_at) > 0.15:
                _kill_proc(proc, signal.SIGKILL)
                sent_kill = True

        # Break when child is gone and both reader threads have delivered EOF
        if proc.poll() is not None and not t_out.is_alive() and not t_err.is_alive():
            break

        if not drained:
            time.sleep(0.02)  # avoid busy loop

    # Flush decoders
    out_parts.append(dec_out.decode(b"", final=True))
    err_parts.append(dec_err.decode(b"", final=True))
    stdout_text = "".join(out_parts)
    stderr_text = "".join(err_parts)

    runner_forced_exit = sent_term or sent_kill

    # Decide outcome
    if parsed is None and proc.returncode == 0:
        # Exit 0 but no JSON → parse failure
        # keep run dir for inspection
        try:
            (work / "child.stdout").write_text(stdout_text or "", encoding="utf-8")
            (work / "child.stderr").write_text(stderr_text or "", encoding="utf-8")
        except Exception:
            pass
        _dprint(f"[agentpm] child logs saved in: {work}")

        raise RuntimeError(
            f"Failed to parse tool JSON output.\n Stderr:\n{stderr_text}\nStdout:\n{stdout_text}"
        )

    if proc.returncode != 0 and not runner_forced_exit:
        # Child failed on its own → persist logs and raise
        try:
            (work / "child.stdout").write_text(stdout_text or "", encoding="utf-8")
            (work / "child.stderr").write_text(stderr_text or "", encoding="utf-8")
        except Exception:
            pass
        _dprint(f"[agentpm] child logs saved in: {work}")

        # Child failed on its own → raise
        tail = stderr_text[-4000:] if stderr_text else ""
        raise RuntimeError(f"Tool exited with code {proc.returncode}. Stderr (tail):\n{tail}")

    if parsed is None:
        # Shouldn't happen if runner forced exit; guard anyway
        raise RuntimeError(
            f"Tool did not produce valid JSON.\n Stderr:\n{stderr_text}\nStdout:\n{stdout_text}"
        )

    # Success: cleanup and return parsed JSON
    shutil.rmtree(work, ignore_errors=True)
    return parsed


def _kill_proc(proc: subprocess.Popen, sig: int) -> None:
    try:
        if os.name == "posix":
            # send to process group (we started a new session)
            os.killpg(proc.pid, sig)
        else:
            # Windows: use terminate/kill equivalents
            proc.terminate() if sig == signal.SIGTERM else proc.kill()
    except Exception:
        pass


def _try_extract_json(buf: str) -> tuple[Any, str | None, int | None, int | None]:
    """
    Heuristic: find the first complete top-level JSON object in `buf`
    using brace depth. Returns (obj, slice_text, start_idx, end_idx) or
    (None, None, None, None) if not found / not parseable yet.
    """
    depth = 0
    start = None
    for i, ch in enumerate(buf):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    s = buf[start : i + 1]
                    try:
                        return json.loads(s), s, start, i + 1
                    except Exception:
                        # keep scanning; may be partial/invalid JSON fragment
                        pass
    return None, None, None, None


# --- Overloads (type-only) ---
@overload
def load(
    spec: str,
    with_meta: Literal[True],
    timeout: float | None = ...,
    tool_dir_override: str | None = ...,
    env: dict[str, str] | None = ...,
) -> LoadedWithMeta: ...
@overload
def load(
    spec: str,
    with_meta: Literal[False] = ...,
    timeout: float | None = ...,
    tool_dir_override: str | None = ...,
    env: dict[str, str] | None = ...,
) -> ToolFunc: ...


def load(
    spec: str,
    with_meta: bool = False,
    timeout: float | None = None,
    tool_dir_override: str | None = None,
    env: dict[str, str] | None = None,
) -> ToolFunc | LoadedWithMeta:
    _dprint(f"spec={spec}")

    try:
        root, manifest_path = _resolve_tool_root(spec, tool_dir_override)
    except Exception as err:
        try:
            _resolve_skill_root(spec, None)
            raise ValueError(
                f'Package "{spec}" is a Skill. load() is tool-only; use load_skill("{spec}") instead.'
            )
        except ValueError as skill_err:
            if "load_skill" in str(skill_err):
                raise skill_err
            try:
                _resolve_knowledge_root(spec, None)
                raise ValueError(
                    f'Package "{spec}" is Knowledge. load() is tool-only; use load_knowledge("{spec}") instead.'
                )
            except ValueError as knowledge_err:
                if "load_knowledge" in str(knowledge_err):
                    raise knowledge_err
                try:
                    _resolve_memory_root(spec, None)
                    raise ValueError(
                        f'Package "{spec}" is Memory. load() is tool-only; use load_memory("{spec}") instead.'
                    )
                except ValueError as memory_err:
                    if "load_memory" in str(memory_err):
                        raise memory_err
                raise err from skill_err
            except FileNotFoundError:
                raise err from skill_err
        except FileNotFoundError:
            try:
                _resolve_knowledge_root(spec, None)
                raise ValueError(
                    f'Package "{spec}" is Knowledge. load() is tool-only; use load_knowledge("{spec}") instead.'
                )
            except ValueError as knowledge_err:
                if "load_knowledge" in str(knowledge_err):
                    raise knowledge_err
            except FileNotFoundError:
                try:
                    _resolve_memory_root(spec, None)
                    raise ValueError(
                        f'Package "{spec}" is Memory. load() is tool-only; use load_memory("{spec}") instead.'
                    )
                except ValueError as memory_err:
                    if "load_memory" in str(memory_err):
                        raise memory_err
                except FileNotFoundError:
                    pass
            if isinstance(err, FileNotFoundError) and "not found in .agentpm/tools" in str(err):
                raise FileNotFoundError(
                    f'{err} If this package is a Skill, use load_skill("{spec}") instead. If it is Knowledge, use load_knowledge("{spec}") instead. If it is Memory, use load_memory("{spec}") instead.'
                ) from None
            raise err from None
    m = _read_manifest(manifest_path)

    env = env or {}

    # enforce interpreter whitelist and available
    ep = m["entrypoint"]

    _dprint(f"resolved root={root}")
    _dprint(f"manifest={manifest_path}")
    _dprint(f'entry.command="{ep["command"]}" args={ep.get("args", [])}')

    # enforce expected/required environment
    expected_env = m.get("environment") or {}
    vars_obj = expected_env.get("vars", {}) if expected_env else {}
    has_vars = bool(vars_obj) and isinstance(vars_obj, dict) and bool(list(vars_obj.values()))
    if has_vars:
        _dprint(f"tool-defined environment={expected_env}")

        for k, v in vars_obj.items():
            default_val: str | None = v.get("default")

            if v.get("required") and not v.get("default") and k not in env:
                raise ValueError(
                    f"Missing environment variable: {k}. {k} is required and has no default value."
                )
            elif default_val is not None and k not in env:
                # Set the default value
                env[k] = default_val

    runtime = m.get("runtime") or {}
    rt = runtime.get("type")
    runtime_type: str | None = rt if rt in ("node", "python") else None
    resolved_cmd = _resolve_interpreter_command(ep["command"], ep.get("env", {}), env, runtime_type)

    # enforce interpreter whitelist and available
    _assert_allowed_interpreter(resolved_cmd)
    _assert_interpreter_available(resolved_cmd, ep.get("env", {}), env)

    # enforce interpreter and runtime compatability
    if "runtime" in m and "type" in m["runtime"]:
        _assert_interpreter_matches_runtime(resolved_cmd, m["runtime"])

    t_s = (
        timeout
        if timeout is not None
        else float(ep.get("timeout_ms") or (DEFAULT_TIMEOUT * 1000)) / 1000.0
    )

    entry_for_spawn = ep | {"command": resolved_cmd}

    def func(input: JsonValue) -> JsonValue:
        return _spawn_once(root, entry_for_spawn, input, t_s, env)

    if with_meta:
        name = m["name"]
        version = m["version"]

        meta: ToolMeta = {
            "name": name,
            "version": version,
        }

        desc = m.get("description")
        if isinstance(desc, str):
            meta["description"] = desc
        if "inputs" in m:
            meta["inputs"] = cast(JsonValue, m["inputs"])
        if "outputs" in m:
            meta["outputs"] = cast(JsonValue, m["outputs"])
        if "environment" in m:
            meta["environment"] = m["environment"]

        return {"func": func, "meta": meta}

    return func


def load_agent(
    spec: str,
    *,
    agent_dir_override: str | None = None,
    skill_dir_override: str | None = None,
    tool_dir_override: str | None = None,
    knowledge_dir_override: str | None = None,
    memory_dir_override: str | None = None,
    lockfile_override: str | None = None,
) -> LoadedAgent:
    root, manifest_path = _resolve_agent_root(spec, agent_dir_override)
    manifest = _read_agent_manifest(manifest_path)
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f'Invalid agent spec "{spec}". Expected "@scope/name@version".')
    package_name = spec[:at]

    lockfile_path = _resolve_agent_lockfile_path(lockfile_override)
    if not lockfile_path.exists():
        raise FileNotFoundError(
            f'agent.lock not found at {lockfile_path}; installed agent metadata requires a lockfile v2. Run "agentpm install" to generate the lockfile.'
        )

    lock = _read_lockfile_v2(lockfile_path)
    package_key = f'agent:{package_name}@{manifest["version"]}'
    roots = cast(dict[str, Any], lock.get("roots") or {})
    root_entry = cast(dict[str, Any] | None, roots.get(package_key))
    if root_entry is None:
        raise ValueError(
            f'Agent root "{package_key}" not found in {lockfile_path}; install the agent with agentpm install first.'
        )

    reserved = _empty_reserved_references()
    root_reserved = cast(dict[str, list[DependencyReference]], root_entry.get("reserved") or {})
    for key in ("knowledge", "memory", "profiles"):
        reserved[key] = list(root_reserved.get(key, []))  # type: ignore[literal-required]

    packages = cast(dict[str, dict[str, Any]], lock.get("packages") or {})
    resolved_tools: list[ResolvedAgentToolRef] = []
    for tool_key in cast(list[str], root_entry.get("tools") or []):
        pkg = packages.get(tool_key)
        if not pkg or pkg.get("kind") != "tool":
            continue

        installed = _resolve_tool_installed_path(
            cast(str, pkg["name"]), cast(str, pkg["version"]), tool_dir_override
        )
        resolved_tools.append(
            {
                "packageKey": tool_key,
                "kind": "tool",
                "name": cast(str, pkg["name"]),
                "version": cast(str, pkg["version"]),
                "integrity": cast(str, pkg["integrity"]),
                "root": str(installed[0]) if installed else None,
                "manifestPath": str(installed[1]) if installed else None,
            }
        )

    resolved_skills: list[ResolvedAgentSkillRef] = []
    for skill_key in cast(list[str], root_entry.get("skills") or []):
        pkg = packages.get(skill_key)
        if not pkg or pkg.get("kind") != "skill":
            continue

        installed = _resolve_skill_installed_path(
            cast(str, pkg["name"]), cast(str, pkg["version"]), skill_dir_override
        )
        resolved_skills.append(
            {
                "packageKey": skill_key,
                "kind": "skill",
                "name": cast(str, pkg["name"]),
                "version": cast(str, pkg["version"]),
                "integrity": cast(str, pkg["integrity"]),
                "root": str(installed[0]) if installed else None,
                "manifestPath": str(installed[1]) if installed else None,
            }
        )

    resolved_knowledge: list[ResolvedAgentKnowledgeRef] = []
    for knowledge_key in cast(list[str], root_entry.get("knowledge") or []):
        pkg = packages.get(knowledge_key)
        if not pkg or pkg.get("kind") != "knowledge":
            continue

        installed = _resolve_knowledge_installed_path(
            cast(str, pkg["name"]), cast(str, pkg["version"]), knowledge_dir_override
        )
        knowledge_manifest = _read_knowledge_manifest(installed[1]) if installed else None
        resolved_knowledge.append(
            {
                "packageKey": knowledge_key,
                "kind": "knowledge",
                "name": cast(str, pkg["name"]),
                "version": cast(str, pkg["version"]),
                "integrity": cast(str, pkg["integrity"]),
                "mode": (
                    cast(Literal["context", "vector"], knowledge_manifest["knowledge"]["mode"])
                    if knowledge_manifest
                    else None
                ),
                "root": str(installed[0]) if installed else None,
                "manifestPath": str(installed[1]) if installed else None,
            }
        )

    resolved_memory: list[ResolvedAgentMemoryRef] = []
    for memory_key in cast(list[str], root_entry.get("memory") or []):
        pkg = packages.get(memory_key)
        if not pkg or pkg.get("kind") != "memory":
            continue

        installed = _resolve_memory_installed_path(
            cast(str, pkg["name"]), cast(str, pkg["version"]), memory_dir_override
        )
        resolved_memory.append(
            {
                "packageKey": memory_key,
                "kind": "memory",
                "name": cast(str, pkg["name"]),
                "version": cast(str, pkg["version"]),
                "integrity": cast(str, pkg["integrity"]),
                "root": str(installed[0]) if installed else None,
                "manifestPath": str(installed[1]) if installed else None,
            }
        )

    return {
        "root": str(root),
        "manifestPath": str(manifest_path),
        "manifest": manifest,
        "resolvedTools": resolved_tools,
        "resolvedSkills": resolved_skills,
        "resolvedKnowledge": resolved_knowledge,
        "resolvedMemory": resolved_memory,
        "reserved": reserved,
    }


def _resolved_tools_from_lock_keys(
    tool_keys: list[str],
    packages: dict[str, dict[str, Any]],
    tool_dir_override: str | None,
) -> list[ResolvedAgentToolRef]:
    resolved_tools: list[ResolvedAgentToolRef] = []
    for tool_key in tool_keys:
        pkg = packages.get(tool_key)
        if not pkg or pkg.get("kind") != "tool":
            continue

        installed = _resolve_tool_installed_path(
            cast(str, pkg["name"]), cast(str, pkg["version"]), tool_dir_override
        )
        resolved_tools.append(
            {
                "packageKey": tool_key,
                "kind": "tool",
                "name": cast(str, pkg["name"]),
                "version": cast(str, pkg["version"]),
                "integrity": cast(str, pkg["integrity"]),
                "root": str(installed[0]) if installed else None,
                "manifestPath": str(installed[1]) if installed else None,
            }
        )
    return resolved_tools


def _parse_dependency_reference(ref: DependencyReference) -> tuple[str, str | None]:
    if isinstance(ref, str):
        at = ref.rfind("@")
        if at <= 0 or at == len(ref) - 1:
            return ref, None
        return ref[:at], ref[at + 1 :]

    if isinstance(ref, dict) and isinstance(ref.get("name"), str):
        version = ref.get("version")
        return cast(str, ref["name"]), (
            cast(str | None, version) if isinstance(version, str) else None
        )

    raise ValueError(f"Invalid dependency reference in installed skill manifest: {ref!r}")


def _resolved_tools_from_skill_manifest(
    *,
    skill_spec_name: str,
    skill_version: str,
    tool_refs: list[DependencyReference],
    packages: dict[str, dict[str, Any]],
    tool_dir_override: str | None,
    lockfile_path: Path,
) -> list[ResolvedAgentToolRef]:
    resolved_tools: list[ResolvedAgentToolRef] = []
    for ref in tool_refs:
        tool_name, declared_version = _parse_dependency_reference(ref)

        if declared_version is None:
            matches = [
                (package_key, pkg)
                for package_key, pkg in packages.items()
                if pkg.get("kind") == "tool" and pkg.get("name") == tool_name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f'Skill "{skill_spec_name}@{skill_version}" declares tool dependency "{tool_name}" without an exact '
                    f"version, and it could not be resolved uniquely from {lockfile_path}."
                )
            tool_key, pkg = matches[0]
        else:
            tool_key = f"tool:{tool_name}@{declared_version}"
            pkg_value = packages.get(tool_key)
            if not pkg_value or pkg_value.get("kind") != "tool":
                raise ValueError(
                    f'Skill "{skill_spec_name}@{skill_version}" declares tool dependency "{tool_name}@{declared_version}" '
                    f'that is not present in {lockfile_path}. Run "agentpm install" to refresh the lockfile.'
                )
            pkg = pkg_value

        installed = _resolve_tool_installed_path(
            cast(str, pkg["name"]), cast(str, pkg["version"]), tool_dir_override
        )
        resolved_tools.append(
            {
                "packageKey": tool_key,
                "kind": "tool",
                "name": cast(str, pkg["name"]),
                "version": cast(str, pkg["version"]),
                "integrity": cast(str, pkg["integrity"]),
                "root": str(installed[0]) if installed else None,
                "manifestPath": str(installed[1]) if installed else None,
            }
        )

    return resolved_tools


def load_skill(
    spec: str,
    *,
    skill_dir_override: str | None = None,
    tool_dir_override: str | None = None,
    lockfile_override: str | None = None,
) -> LoadedSkill:
    root, manifest_path = _resolve_skill_root(spec, skill_dir_override)
    manifest = _read_skill_manifest(manifest_path)
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f'Invalid skill spec "{spec}". Expected "@scope/name@version".')
    package_name = spec[:at]

    lockfile_path = _resolve_agent_lockfile_path(lockfile_override)
    if not lockfile_path.exists():
        raise FileNotFoundError(
            f'agent.lock not found at {lockfile_path}; installed skill metadata requires a lockfile v3. Run "agentpm install" to generate the lockfile.'
        )

    lock = _read_lockfile_v2(lockfile_path)
    package_key = f'skill:{package_name}@{manifest["version"]}'
    packages = cast(dict[str, dict[str, Any]], lock.get("packages") or {})
    roots = cast(dict[str, Any], lock.get("roots") or {})
    root_entry = cast(dict[str, Any] | None, roots.get(package_key))
    resolved_tools = (
        _resolved_tools_from_lock_keys(
            cast(list[str], root_entry.get("tools") or []), packages, tool_dir_override
        )
        if root_entry is not None
        else _resolved_tools_from_skill_manifest(
            skill_spec_name=package_name,
            skill_version=manifest["version"],
            tool_refs=cast(list[DependencyReference], manifest.get("tools") or []),
            packages=packages,
            tool_dir_override=tool_dir_override,
            lockfile_path=lockfile_path,
        )
    )

    entrypoint_path = (root / manifest["skill"]["entrypoint"]).resolve()
    entrypoint_content = entrypoint_path.read_text(encoding="utf-8")

    return {
        "kind": "skill",
        "name": manifest["name"],
        "version": manifest["version"],
        "description": manifest.get("description"),
        "root": str(root),
        "manifestPath": str(manifest_path),
        "manifest": manifest,
        "skill": manifest["skill"],
        "entrypointPath": str(entrypoint_path),
        "entrypointContent": entrypoint_content,
        "references": list(manifest["skill"].get("references") or []),
        "scripts": list(manifest["skill"].get("scripts") or []),
        "resolvedTools": resolved_tools,
    }


def load_knowledge(
    spec: str,
    *,
    knowledge_dir_override: str | None = None,
) -> LoadedKnowledge:
    root, manifest_path = _resolve_knowledge_root(spec, knowledge_dir_override)
    manifest = _read_knowledge_manifest(manifest_path)
    knowledge = manifest["knowledge"]

    document_paths = [
        str((root / document["path"]).resolve())
        for document in knowledge.get("documents") or []
        if isinstance(document, dict) and isinstance(document.get("path"), str)
    ]
    chunks_path = (
        str((root / knowledge["corpus"]["chunks_path"]).resolve())
        if knowledge.get("corpus", {}).get("chunks_path")
        else None
    )
    sources_path = (
        str((root / knowledge["corpus"]["sources_path"]).resolve())
        if knowledge.get("corpus", {}).get("sources_path")
        else None
    )
    vectors_path = (
        str((root / knowledge["embedding"]["vectors_path"]).resolve())
        if knowledge.get("embedding", {}).get("vectors_path")
        else None
    )
    index_paths = [
        str((root / index["path"]).resolve())
        for index in knowledge.get("indexes") or []
        if isinstance(index, dict) and isinstance(index.get("path"), str)
    ]
    provenance_path = (
        str((root / knowledge["provenance"]["sources_manifest_path"]).resolve())
        if knowledge.get("provenance", {}).get("sources_manifest_path")
        else None
    )

    return {
        "kind": "knowledge",
        "name": manifest["name"],
        "version": manifest["version"],
        "description": manifest.get("description"),
        "root": str(root),
        "manifestPath": str(manifest_path),
        "manifest": manifest,
        "knowledge": knowledge,
        "documentPaths": document_paths,
        "chunksPath": chunks_path,
        "sourcesPath": sources_path,
        "vectorsPath": vectors_path,
        "indexPaths": index_paths,
        "provenancePath": provenance_path,
    }


def load_memory(
    spec: str,
    *,
    memory_dir_override: str | None = None,
) -> LoadedMemory:
    root, manifest_path = _resolve_memory_root(spec, memory_dir_override)
    manifest = _read_memory_manifest(manifest_path)
    build_path, build = _read_memory_build_metadata(root)
    contract_index_path, contract_index, source_schema_paths, contracts = (
        _read_memory_contract_index(
            root,
            manifest["memory"],
            cast(int, build["contract_count"]),
        )
    )

    return {
        "kind": "memory",
        "name": manifest["name"],
        "version": manifest["version"],
        "description": cast(str | None, manifest.get("description")),
        "root": str(root),
        "manifestPath": str(manifest_path),
        "manifest": manifest,
        "memory": manifest["memory"],
        "buildPath": str(build_path),
        "build": build,
        "contractIndexPath": str(contract_index_path),
        "contractIndex": contract_index,
        "sourceSchemaPaths": source_schema_paths,
        "contracts": contracts,
    }


def load_memory_contract(
    memory_package: LoadedMemory,
    *,
    space: str,
    record_type: str,
) -> MemoryContractSchema:
    contract_ref = next(
        (
            entry
            for entry in memory_package["contracts"]
            if entry["space"] == space and entry["recordType"] == record_type
        ),
        None,
    )
    if contract_ref is None:
        raise ValueError(
            f'Resolved memory contract "{space}:{record_type}" was not found in memory/contracts/index.json.'
        )
    return cast(
        MemoryContractSchema,
        _read_json_file(Path(contract_ref["path"]), f'memory contract "{space}:{record_type}"'),
    )
