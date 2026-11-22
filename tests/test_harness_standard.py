"""SDK support advertisement and actual CLI selected-version parity."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from agentpm import HarnessClient, HarnessProtocolError

STANDARD = {"id": "agentpm", "version": "1.0.0"}
CLI = os.getenv("AGENTPM_TEST_CLI")


def prepare(root: Path, mode: str) -> None:
    path = Path(__file__).parent / "fixtures/sdk-v1/harness_fixture.py"
    spec = importlib.util.spec_from_file_location("harness_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare(root, CLI, mode)


@pytest.mark.parametrize(
    "advertisement", ["absent", [], [STANDARD], None, [{"id": "agentpm"}], "malformed-report"]
)
def test_standard_advertisement_preserves_absence_empty_and_future_fields(
    tmp_path: Path, advertisement: Any
) -> None:
    report = {"status": "ready", "future_field": {"preserved": True}}
    if advertisement != "absent" and advertisement != "malformed-report":
        report["supported_standards"] = advertisement
    if advertisement == "malformed-report":
        report["report"] = None
    script = tmp_path / "cli.py"
    script.write_text(
        "import json,sys\nreport=json.loads("
        + repr(json.dumps(report))
        + ')\nfor line in sys.stdin:\n frame=json.loads(line)\n payload={"session":{"protocol":frame["protocol"],"version":1},"preflight":report} if frame["method"]=="initialize" else {"shutdown":True} if frame["method"]=="shutdown" else report\n print(json.dumps({"protocol":frame["protocol"],"version":1,"kind":"response","id":frame["id"],"payload":payload}),flush=True)\n if frame["method"]=="shutdown": break\n',
        encoding="utf-8",
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[str(script)])
    try:
        assert client.initialize()["preflight"] == report
        assert client.preflight() == report
        if (
            advertisement is None
            or advertisement == [{"id": "agentpm"}]
            or advertisement == "malformed-report"
        ):
            with pytest.raises(HarnessProtocolError) as caught:
                client.supported_standards()
            assert caught.value.code == "invalid_standard_advertisement"
        else:
            assert client.supported_standards() == (
                None if advertisement == "absent" else advertisement
            )
        client.shutdown()
    finally:
        client.stop()


@pytest.mark.skipif(not CLI, reason="requires actual AGENTPM_TEST_CLI, provided in hosted SDK CI")
@pytest.mark.parametrize("mode", ["ready", "no-loop", "missing-model", "legacy", "unsupported"])
def test_actual_cli_and_machine_preflight_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("AGENTPM_TELEMETRY", "0")
    prepare(tmp_path, mode)
    expected = json.loads((tmp_path / "expected-preflight.json").read_text())
    client = HarnessClient(agentpm_path=CLI, cwd=str(tmp_path), env={"AGENTPM_TELEMETRY": "0"})
    try:
        info = client.initialize()
        assert info["session"] == {"protocol": "agentpm-harness-machine", "version": 1}
        assert info["preflight"] == expected
        assert client.preflight()["report"] == expected
        assert client.supported_standards() == [STANDARD]
        if mode not in ["ready", "legacy"]:
            with pytest.raises(HarnessProtocolError) as caught:
                client.run("must not run")
            assert caught.value.code == "preflight_not_ready"
        client.shutdown()
    finally:
        client.stop()


@pytest.mark.skipif(not CLI, reason="requires actual AGENTPM_TEST_CLI, provided in hosted SDK CI")
def test_exact_direct_and_inherited_versions_execute_with_loop_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTPM_TELEMETRY", "0")
    prepare(tmp_path, "versions")
    for major in [1, 2]:
        version = f"{major}.0.0"
        calls = []
        client = HarnessClient(
            agentpm_path=CLI,
            cwd=str(tmp_path),
            agent=f"consumer-{major}",
            env={"AGENTPM_TELEMETRY": "0"},
        )

        def model(payload: Any, version: str = version, calls: list[Any] = calls) -> dict[str, Any]:
            request = payload["request"]
            assert [p["version"] for p in request["runtime"]["tools"]] == [version]
            assert [p["version"] for p in request["runtime"]["skills"]] == [version]
            for field in ["knowledge", "profiles"]:
                assert [p["version"] for p in request["runtime"][field]] == [version]
            assert request["runtime"]["loop_package"]["version"] == version
            assert [p["package_version"] for p in request["runtime"]["memory_operations"]] == [
                version
            ]
            assert request["effective_phase"]["active_memory"] == []
            assert [
                p["package_version"] for p in request["effective_phase"]["active_memory_operations"]
            ] == [version]
            assert [p["version"] for p in request["effective_phase"]["active_skills"]] == [version]
            assert [p["version"] for p in request["effective_phase"]["active_tools"]] == (
                [version] if request["phase_id"] == "work" else []
            )
            calls.append(request)
            action = (
                {"type": "agent_pm_tool", "tool": "@fixture/echo", "arguments": {}}
                if len(calls) == 1
                else {"type": "phase_completion", "output": {"version": version}}
            )
            return {
                "assistant_content": None,
                "actions": [{"id": f"step-{len(calls)}", "action": action}],
                "finish_reason": "tool_calls",
            }

        client.register_model_provider("sdk-model", model)
        try:
            initialized = client.initialize()
            assert initialized["preflight"]["local_readiness"] == "ready", initialized["preflight"]
            result = client.run("Check exact versions.")
            assert result["status"] == "ended", result
            assert len(calls) == 3
            client.shutdown()
        finally:
            client.stop()
    assert not (tmp_path / "skill-script-ran").exists()
    assert [
        json.loads(line) for line in (tmp_path / "tool-runs.jsonl").read_text().splitlines()
    ] == [{"version": "1.0.0"}, {"version": "2.0.0"}]
