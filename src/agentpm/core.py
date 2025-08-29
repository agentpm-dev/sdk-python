from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Literal, cast, overload

from .types import Entrypoint, JsonValue, LoadedWithMeta, Manifest, Runtime, ToolFunc, ToolMeta

DEFAULT_TIMEOUT = 120.0
_ALLOWED = {"node", "nodejs", "python", "python3"}


def _canonical(cmd: str) -> str:
    # handle absolute paths and Windows extensions
    base = os.path.basename(cmd).lower()
    for ext in (".exe", ".cmd", ".bat"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def _assert_allowed_interpreter(cmd: str) -> None:
    if _canonical(cmd) not in _ALLOWED:
        raise ValueError(
            f'Unsupported agent.json.entrypoint.command "{cmd}". Allowed: node|nodejs|python|python3'
        )


# verify the interpreter exists on PATH
def _assert_interpreter_available(cmd: str) -> None:
    try:
        subprocess.run(
            [cmd, "--version"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f'Interpreter "{cmd}" not found on PATH. Install it to load tool.'
        ) from e


def _assert_interpreter_matches_runtime(cmd: str, runtime: Runtime) -> None:
    canon = _canonical(cmd)
    runtime_interpreter = _canonical(runtime["type"])

    if canon != runtime_interpreter:
        raise ValueError(
            f'Misconfigured tool - agent.json.entrypoint.command "{cmd}" does not match tool runtime {runtime_interpreter}'
        )


def _resolve_tool_root(spec: str, tool_dir_override: str | None) -> tuple[Path, Path]:
    # spec form: "@scope/name@1.2.3"
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f'Invalid tool spec "{spec}". Expected "@scope/name@version".')
    version, name = spec[at + 1 :], spec[:at]
    candidates = [
        tool_dir_override,
        os.getenv("AGENTPM_TOOL_DIR"),  # optional override
        Path.cwd() / ".agentpm" / "tools",  # project-local (preferred)
        Path.home() / ".agentpm" / "tools",  # fallback
    ]
    for c in candidates:
        if not c:
            continue
        root = Path(c) / f"{name}/{version}"
        mf = root / "agent.json"
        if mf.exists():
            return root, mf
    raise FileNotFoundError(f'Tool "{spec}" not found in .agentpm/tools (or overrides).')


def _read_manifest(p: Path) -> Manifest:
    m = json.loads(p.read_text(encoding="utf-8"))
    ep = m.get("entrypoint", {})
    if not ep or not ep.get("command"):
        raise ValueError(f"agent.json missing entrypoint.command at: {p}")
    return m  # type: ignore[no-any-return]


def _spawn_once(
    root: Path, entry: Entrypoint, payload: JsonValue, timeout_s: float, env: dict[str, str]
) -> JsonValue:
    cwd = (root / entry.get("cwd", ".")).resolve()
    args = entry.get("args", [])
    cmd = [entry["command"], *args]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env={**os.environ, **entry.get("env", {}), **env},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(input=json.dumps(payload), timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        raise TimeoutError(f"Tool timed out after {timeout_s:.1f}s") from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"Tool exited with code {proc.returncode}. Stderr:\n{stderr or '(empty)'}"
        )
    try:
        return _extract_last_json(stdout)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse tool JSON output.\nStderr:\n{stderr}\nStdout:\n{stdout}\nReason: {e}"
        ) from e


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
    root, manifest_path = _resolve_tool_root(spec, tool_dir_override)
    m = _read_manifest(manifest_path)

    # enforce interpreter whitelist and available
    ep = m["entrypoint"]
    _assert_allowed_interpreter(ep["command"])
    _assert_interpreter_available(ep["command"])

    # enforce interpreter and runtime compatability
    if "runtime" in m and "type" in m["runtime"]:
        _assert_interpreter_matches_runtime(ep["command"], m["runtime"])

    t_s = (
        timeout
        if timeout is not None
        else float(ep.get("timeout_ms") or (DEFAULT_TIMEOUT * 1000)) / 1000.0
    )
    env = env or {}

    def func(input: JsonValue) -> JsonValue:
        return _spawn_once(root, ep, input, t_s, env)

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
