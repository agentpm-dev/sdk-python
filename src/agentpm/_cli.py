"""Transport for the CLI-owned Tool runner. No Tool execution policy lives here."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import BinaryIO, NoReturn, cast

from .types import JsonValue

MIN_CLI = (0, 1, 43)
MAX_TRANSPORT_BYTES = 12 * 1024 * 1024
GUIDANCE = "Install AgentPM CLI >=0.1.43 and make agentpm available on PATH (or set AGENTPM)."


def _reject_constant(_value: str) -> NoReturn:
    raise ValueError("non-JSON number")


def _redact(text: str, environment: dict[str, str]) -> str:
    values = [
        v
        for k, v in environment.items()
        if v
        and re.search(
            r"token|secret|password|passphrase|api.?key|authorization|credential|_key(?:$|_)",
            k,
            re.I,
        )
    ]
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return text


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False
        )
    else:
        # The CLI handles TERM and cleans its Tool process group.
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def invoke(
    project: Path,
    directory: Path,
    spec: str,
    payload: JsonValue,
    timeout: float,
    environment: dict[str, str],
    redaction_environment: dict[str, str] | None = None,
) -> JsonValue:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Tool timeout must be a positive finite number")
    command = os.environ.get("AGENTPM") or "agentpm"
    process_env = os.environ.copy()
    process_env.update(environment)
    redaction_env = {**process_env, **(redaction_environment or {}), **environment}
    try:
        version = subprocess.run(
            [command, "--version"], env=process_env, capture_output=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError(GUIDANCE) from None
    match = re.fullmatch(rb"agentpm (\d+)\.(\d+)\.(\d+)\s*", version.stdout)
    if version.returncode or not match or tuple(map(int, match.groups())) < MIN_CLI:
        raise RuntimeError(GUIDANCE)
    request = json.dumps(
        {
            "schema_version": 1,
            "input": payload,
            "environment": environment,
            "tool_directory": str(directory.resolve()),
        },
        allow_nan=False,
    ).encode()
    try:
        process = subprocess.Popen(
            [
                command,
                "run",
                spec,
                "--machine",
                "--sdk-request",
                "--timeout-ms",
                str(max(1, round(timeout * 1000))),
            ],
            cwd=project,
            env=process_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        raise RuntimeError(GUIDANCE) from None
    buffers = [bytearray(), bytearray()]
    exceeded = threading.Event()
    lock = threading.Lock()

    def read(stream: object, target: bytearray) -> None:
        # Pipes are binary; keep the bound even when a broken CLI floods stderr.
        pipe = cast(BinaryIO, stream)
        while chunk := pipe.read(65536):
            with lock:
                if sum(map(len, buffers)) + len(chunk) > MAX_TRANSPORT_BYTES:
                    exceeded.set()
                    return
                target.extend(chunk)

    readers = [
        threading.Thread(target=read, args=(process.stdout, buffers[0]), daemon=True),
        threading.Thread(target=read, args=(process.stderr, buffers[1]), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        assert process.stdin is not None

        # Write concurrently so large input cannot deadlock against CLI output.
        def write() -> None:
            try:
                assert process.stdin is not None
                process.stdin.write(request)
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        writer = threading.Thread(target=write, daemon=True)
        writer.start()

        deadline = time.monotonic() + timeout + 10
        while process.poll() is None:
            if exceeded.wait(0.02):
                raise RuntimeError("AgentPM CLI output exceeded the SDK transport limit")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "AgentPM CLI did not complete the invocation timeout; check the installed CLI"
                )
        for reader in readers:
            reader.join(timeout=2)
        if exceeded.is_set() or any(reader.is_alive() for reader in readers):
            raise RuntimeError(
                "AgentPM CLI output exceeded the SDK transport limit or did not close"
            )
        writer.join(timeout=1)
    finally:
        _stop(process)
        for reader in readers:
            reader.join(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    stderr = _redact(buffers[1].decode("utf-8", errors="replace"), redaction_env)
    if stderr:
        sys.stderr.write(stderr)
    try:
        result = json.loads(buffers[0], parse_constant=_reject_constant)
    except (ValueError, UnicodeError):
        raise RuntimeError("Invalid AgentPM CLI machine response; " + GUIDANCE) from None
    if (
        not isinstance(result, dict)
        or type(result.get("schema_version")) is not int
        or result.get("schema_version") != 1
    ):
        raise RuntimeError("Unsupported AgentPM CLI machine response; " + GUIDANCE)
    if result.get("status") == "error":
        error = result.get("error")
        if (
            not isinstance(error, dict)
            or not isinstance(error.get("message"), str)
            or not isinstance(error.get("category"), str)
        ):
            raise RuntimeError("Invalid AgentPM CLI error response; " + GUIDANCE)
        message = _redact(f"AgentPM {error['category']}: {error['message']}", redaction_env)
        if error["category"] == "timeout":
            raise TimeoutError(message)
        raise RuntimeError(message)
    tool = result.get("tool")
    if (
        process.returncode
        or result.get("status") != "success"
        or "output" not in result
        or not isinstance(tool, dict)
        or f"{tool.get('package')}@{tool.get('version')}" != spec
    ):
        raise RuntimeError("Invalid or unsuccessful AgentPM CLI machine response; " + GUIDANCE)
    return cast(JsonValue, result["output"])
