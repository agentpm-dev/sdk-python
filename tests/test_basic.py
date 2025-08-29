from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Import from the installed package (editable install in your venv)
from agentpm import load


def _split_spec(spec: str) -> tuple[str, str]:
    """
    Split a spec like '@scope/name@1.2.3' -> (name='@scope/name', version='1.2.3')
    """
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f"Bad spec: {spec}")
    return spec[:at], spec[at + 1 :]


def _write_tool_package(
    base_dir: Path,
    spec: str,
    command: str = "python",
    script_file: str = "tool.py",
) -> dict[str, str]:
    """
    Create a fake tool package under base_dir in TWO layouts:
      1) <base>/<name>/<version>/
      2) <base>/<name>@<version>/

    Returns dict with paths and manifest.
    """
    name, version = _split_spec(spec)

    def write_at(root: Path) -> dict[str, str]:
        root.mkdir(parents=True, exist_ok=True)

        # Python entrypoint: reads stdin JSON, prints noise, then final JSON
        tool_py = """\
import sys, json
inp = json.loads(sys.stdin.read() or "{}")
print("stdout noise before json")  # intentional
print("stderr debug line", file=sys.stderr)
text = (inp.get("text") or "")
out = {"summary": text.upper()}
sys.stdout.write(json.dumps(out))
"""
        (root / script_file).write_text(tool_py, encoding="utf-8")

        agent_json = {
            "name": name,
            "version": version,
            "description": "Test summarizer",
            "inputs": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "outputs": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            "entrypoint": {
                "command": command,
                "args": [script_file],
                "cwd": ".",
                "timeout_ms": 30000,
            },
            "kind": "tool",
        }
        (root / "agent.json").write_text(json.dumps(agent_json, indent=2), encoding="utf-8")
        return {"script": script_file}

    # Layout 1: <name>/<version>
    nested_root = base_dir / name / version
    manifest = write_at(nested_root)

    # Layout 2: <name>@<version>
    flat_root = base_dir / f"{name}@{version}"
    # copy files to the alternate layout
    flat_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nested_root / script_file, flat_root / script_file)
    shutil.copy2(nested_root / "agent.json", flat_root / "agent.json")

    return {
        "nested_root": str(nested_root),
        "flat_root": str(flat_root),
        "script": script_file,
        "manifest": json.dumps(manifest),
    }


def _write_failing_tool_package(base_dir: Path, spec: str) -> None:
    name, version = _split_spec(spec)
    root = base_dir / name / version
    root.mkdir(parents=True, exist_ok=True)

    fail_py = """\
import sys
print("boom", file=sys.stderr)
sys.exit(2)
"""
    (root / "fail.py").write_text(fail_py, encoding="utf-8")

    agent_json = {
        "name": name,
        "version": version,
        "description": "Always fails",
        "inputs": {"type": "object", "properties": {}, "required": []},
        "outputs": {"type": "object", "properties": {}, "required": []},
        "entrypoint": {"command": "python", "args": ["fail.py"], "cwd": ".", "timeout_ms": 30000},
        "kind": "tool",
    }
    (root / "agent.json").write_text(json.dumps(agent_json, indent=2), encoding="utf-8")

    # Also mirror in the flat layout
    flat_root = base_dir / f"{name}@{version}"
    flat_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "fail.py", flat_root / "fail.py")
    shutil.copy2(root / "agent.json", flat_root / "agent.json")


@pytest.fixture(scope="module")  # type: ignore[misc]  # pytest decorator is untyped
def tmp_tools_dir() -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="agentpm-sdk-pytest-")).resolve()
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_loads_and_invokes_entrypoint(tmp_tools_dir: Path) -> None:
    ok_spec = "@zack/summarize@0.1.0"
    _write_tool_package(tmp_tools_dir, ok_spec, command="python", script_file="tool.py")

    summarize = load(ok_spec, tool_dir_override=str(tmp_tools_dir))
    result = summarize({"text": "hello world"})
    assert isinstance(result, dict)
    assert result == {"summary": "HELLO WORLD"}


def test_unsupported_entrypoint_command_rejected(tmp_tools_dir: Path) -> None:
    bash_spec = "@zack/scrape@0.1.0"
    _write_tool_package(
        tmp_tools_dir, bash_spec, command="bash", script_file="tool.sh"
    )  # command not allowed

    with pytest.raises(Exception) as ei:
        load(bash_spec, tool_dir_override=str(tmp_tools_dir))
    assert "nsupported agent.json.entrypoint.command" in str(ei.value)


def test_with_meta_returns_func_and_meta(tmp_tools_dir: Path) -> None:
    ok_spec = "@zack/summarize@0.1.0"
    _write_tool_package(tmp_tools_dir, ok_spec, command="python", script_file="tool.py")

    loaded = load(ok_spec, with_meta=True, tool_dir_override=str(tmp_tools_dir))
    assert isinstance(loaded, dict)
    assert "func" in loaded and "meta" in loaded
    func = loaded["func"]
    meta = loaded["meta"]
    assert callable(func)
    assert meta["name"] == "@zack/summarize"
    assert meta["version"] == "0.1.0"

    out = func({"text": "abc"})
    assert isinstance(out, dict)
    assert out.get("summary") == "ABC"


@pytest.mark.skipif(  # type: ignore[misc]  # pytest decorator is untyped
    pytest.importorskip("langchain_core", reason="langchain_core not installed") is None,
    reason="langchain_core missing",
)
def test_to_langchain_tool_adapter(tmp_tools_dir: Path) -> None:
    # Lazy import (agentpm.__getattr__) will load the adapter only when accessed
    from agentpm import to_langchain_tool

    ok_spec = "@zack/summarize@0.1.0"
    _write_tool_package(tmp_tools_dir, ok_spec, command="python", script_file="tool.py")

    loaded = load(ok_spec, with_meta=True, tool_dir_override=str(tmp_tools_dir))
    tool = to_langchain_tool(loaded)  # should produce a tool-like object

    # Name/description checks (best-effort, since adapter shape may vary)
    assert hasattr(tool, "name")
    assert hasattr(tool, "description")
    assert isinstance(tool.name, str)
    assert "Inputs:" in tool.description
    assert "Outputs:" in tool.description

    # Try common call patterns
    if hasattr(tool, "invoke"):
        r = tool.invoke({"text": "mixed Case"})
        assert isinstance(r, str)
        assert r == "MIXED CASE"
    elif hasattr(tool, "func"):
        r = tool.func({"text": "mixed Case"})
        assert isinstance(r, str)
        assert r == "MIXED CASE"
    elif callable(tool):
        r = tool({"text": "mixed Case"})
        assert isinstance(r, str)
        assert r == "MIXED CASE"
    else:
        pytest.skip("Adapter returned an unsupported tool shape")


def test_nonzero_exit_raises(tmp_tools_dir: Path) -> None:
    fail_spec = "@zack/fail@0.1.0"
    _write_failing_tool_package(tmp_tools_dir, fail_spec)

    failing = load(fail_spec, tool_dir_override=str(tmp_tools_dir))
    with pytest.raises(Exception) as ei:
        failing({})
    # Depending on your exact error, this may vary slightly:
    assert "exited with code 2" in str(ei.value)
