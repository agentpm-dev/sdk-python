from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal, cast, overload

from semver import VersionInfo
from semver import match as semver_match

from .types import Entrypoint, JsonValue, LoadedWithMeta, Manifest, Runtime, ToolFunc, ToolMeta

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
    if canon not in _ALLOWED and not canon.startswith("pyhton3"):
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
    Priority: agent.json, package.json, pnpm-workspace.yaml, turbo.json, lerna.json, .git
    Returns the resolved start_dir if nothing is found.
    """
    dir_path = Path(start_dir).resolve()
    while True:
        if (dir_path / "agent.json").exists():
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


def _read_manifest(p: Path) -> Manifest:
    m = json.loads(p.read_text(encoding="utf-8"))
    ep = m.get("entrypoint", {})
    if not ep or not ep.get("command"):
        raise ValueError(f"agent.json missing entrypoint.command at: {p}")
    return m  # type: ignore[no-any-return]


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
        old_space = int(env.get("AGENTPM_NODE_OLD_SPACE_MB", "256"))

        if not any(a.startswith("--max-old-space-size") for a in cmd[1:]):
            cmd.insert(1, f"--max-old-space-size={old_space}")

        want_jitless = (
            any(a == "--jitless" for a in cmd[1:])
            or "--jitless" in (env or {}).get("NODE_OPTIONS", "")
            or env.get("AGENTPM_NODE_JITLESS", "").lower() in ("1", "true", "yes")
        )
        if want_jitless and "--jitless" not in cmd[1:]:
            cmd.insert(1, "--jitless")

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
        text=True,
        start_new_session=True,
        preexec_fn=(
            _preexec_rlimits_for(entry["command"]) if hasattr(resource, "setrlimit") else None
        ),
    )
    try:
        stdout, stderr = proc.communicate(input=json.dumps(payload), timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        raise TimeoutError(f"Tool timed out after {timeout_s:.1f}s") from e

    # Cap outputs (10MB)
    max_bytes = 10 * 1024 * 1024
    if len(stdout.encode("utf-8")) + len(stderr.encode("utf-8")) > max_bytes:
        raise RuntimeError("Tool produced too much output; limit is 10MB")

    if proc.returncode != 0:
        # Save full streams for inspection and KEEP the run dir on error
        try:
            (work / "child.stdout").write_text(stdout or "", encoding="utf-8")
            (work / "child.stderr").write_text(stderr or "", encoding="utf-8")
        except Exception:
            pass
        _dprint(f"[agentpm] child logs saved in: {work}")

        tail = stderr[-4000:] if stderr else ""
        raise RuntimeError(f"Tool exited with code {proc.returncode}. Stderr (tail):\n{tail}")

    try:
        return _extract_last_json(stdout)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse tool JSON output.\nStderr:\n{stderr}\nStdout:\n{stdout}\nReason: {e}"
        ) from e
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _extract_last_json(text: str) -> JsonValue:
    # naive but effective: scan for last '{' and try parse json from there.
    idx = text.rfind("{")
    if idx < 0:
        raise RuntimeError("No JSON object found on stdout.")
    return json.loads(text[idx:])  # type: ignore[no-any-return]


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

    root, manifest_path = _resolve_tool_root(spec, tool_dir_override)
    m = _read_manifest(manifest_path)

    env = env or {}

    # enforce interpreter whitelist and available
    ep = m["entrypoint"]

    _dprint(f"resolved root={root}")
    _dprint(f"manifest={manifest_path}")
    _dprint(f'entry.command="{ep["command"]}" args={ep.get("args", [])}')

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

        return {"func": func, "meta": meta}

    return func
