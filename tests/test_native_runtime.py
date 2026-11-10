from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentpm import load

pytestmark = pytest.mark.skipif(
    not (os.getenv("AGENTPM_TEST_CLI") and os.getenv("AGENTPM_TEST_UV")),
    reason="real native integration requires AGENTPM_TEST_CLI and verified AGENTPM_TEST_UV",
)


def test_real_cli_native_environment_and_fail_closed_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parent / "fixtures/sdk-v1/native_fixture.py"
    spec = importlib.util.spec_from_file_location("native_fixture", fixture)
    assert spec and spec.loader
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    root = tmp_path / "consumer with spaces"
    cli = os.environ["AGENTPM_TEST_CLI"]
    python = os.getenv("AGENTPM_TEST_PYTHON", sys.executable)
    helper.prepare(root, cli, os.environ["AGENTPM_TEST_UV"], python)
    monkeypatch.chdir(root)
    monkeypatch.setenv("AGENTPM", cli)
    monkeypatch.setenv("AGENTPM_PYTHON", python)
    monkeypatch.setenv("AGENTPM_UV_PATH", str(root / "must-not-resolve"))
    shadow = root / "shadow"
    shadow.mkdir()
    (shadow / "pydantic.py").write_text('raise RuntimeError("ambient shadow imported")')
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    # PYTHONHOME breaks a Python process before startup unless the managed runner strips it.
    monkeypatch.setenv("PYTHONHOME", str(shadow))
    locked = root / "agent.lock"
    before = locked.read_bytes()
    ready = next(root.glob(".agentpm/runtimes/python/**/ready.json"))
    ready_bytes = ready.read_bytes()
    ready_time = ready.stat().st_mtime_ns
    tool = root / ".agentpm/tools/fixture/tool/1.0.0"
    metadata = tool / "agentpm-python.json"
    metadata_bytes = metadata.read_bytes()
    loaded = load("@fixture/tool@1.0.0", with_meta=True, env={"LAYER": "caller"})
    result = loaded["func"]({"quantity": 2, "unit_price": 4.5})
    assert isinstance(result, dict)
    assert result["total"] == 9.0 and result["layer"] == "caller"
    assert result["pydantic_version"] == "2.11.7"
    assert result["pythonpath"] is None and result["pythonhome"] is None
    for key in ["prefix", "executable", "native_file", "pydantic_file"]:
        assert str(root / ".agentpm/runtimes/python") in str(result[key])
    assert str(result["native_file"]).endswith((".so", ".pyd"))
    assert locked.read_bytes() == before
    assert ready.read_bytes() == ready_bytes and ready.stat().st_mtime_ns == ready_time
    # No resolve/install endpoint is available once fixture setup has completed.
    timeout_tool = load("@fixture/tool@1.0.0", timeout=0.2)
    with pytest.raises(TimeoutError, match="timeout|timed out"):
        timeout_tool({"quantity": 2, "unit_price": 4.5, "pause": True})
    marker = tool.parent / ".1.0.0.selection.json"
    marker.write_text(json.dumps({"target": "incompatible-fixture"}))
    with pytest.raises(RuntimeError, match="target/state changed"):
        loaded["func"]({"quantity": 2, "unit_price": 4.5})
    marker.unlink()
    alternate = root / "alternate-python"
    subprocess.run(
        [python, "-I", "-m", "venv", "--without-pip", str(alternate)],
        check=True,
        capture_output=True,
    )
    selected = alternate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    monkeypatch.setenv("AGENTPM_PYTHON", str(selected))
    with pytest.raises(RuntimeError, match="agentpm install"):
        loaded["func"]({"quantity": 2, "unit_price": 4.5})
    monkeypatch.setenv("AGENTPM_PYTHON", python)
    metadata.write_text("{}")
    with pytest.raises(RuntimeError, match="agentpm install"):
        loaded["func"]({"quantity": 2, "unit_price": 4.5})
    assert metadata.read_text() == "{}"
    metadata.write_bytes(metadata_bytes)
    ready.unlink()
    with pytest.raises(RuntimeError, match="agentpm install"):
        loaded["func"]({"quantity": 2, "unit_price": 4.5})
    assert not ready.exists() and locked.read_bytes() == before
    ready.write_bytes(ready_bytes)
    assert loaded["func"]({"quantity": 1, "unit_price": 3})["total"] == 3
