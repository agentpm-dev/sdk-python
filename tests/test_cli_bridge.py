from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentpm import _cli, load, load_agent, load_skill

FIXTURES = Path(__file__).parent / "fixtures/sdk-v1"
VECTORS = json.loads((FIXTURES / "compatibility.json").read_text())


@pytest.fixture
def mock_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    original_run, original_popen = subprocess.run, subprocess.Popen

    def args(command: list[str]) -> list[str]:
        return (
            [sys.executable, str(FIXTURES / "mock_cli.py"), *command[1:]]
            if command[0] == "mock-agentpm"
            else command
        )

    monkeypatch.setattr(
        subprocess, "run", lambda command, **options: original_run(args(command), **options)
    )
    monkeypatch.setattr(
        subprocess, "Popen", lambda command, **options: original_popen(args(command), **options)
    )
    monkeypatch.setenv("AGENTPM", "mock-agentpm")
    monkeypatch.setenv("MOCK_CLI_CAPTURE", str(tmp_path / "capture.json"))
    return tmp_path / "capture.json"


def write_tool(project: Path) -> Path:
    root = project / ".agentpm/tools/fixture/tool/0.1.9"
    root.mkdir(parents=True)
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "tool",
                "name": "tool",
                "version": "0.1.9",
                "entrypoint": {
                    "command": "python",
                    "args": ["tool.py"],
                    "env": {"LAYER": "manifest"},
                },
            }
        )
    )
    return root


def test_public_callable_sends_exact_context_and_unwraps(
    mock_cli: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project with spaces"
    directory = write_tool(project)
    (project / "agent.lock").write_text(json.dumps(VECTORS["valid"][2]["lock"]))
    child = project / "nested"
    child.mkdir()
    monkeypatch.chdir(child)
    monkeypatch.setenv("LAYER", "ambient")
    secret = "sdk-super-secret-token"
    loaded = load(
        "@fixture/tool@^0.1.0", with_meta=True, env={"LAYER": "caller", "MOCK_SECRET_TOKEN": secret}
    )
    before = (project / "agent.lock").read_bytes()
    # Shared mock reads UTF-8 bytes even under a Windows text code page.
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    payload = {"message": 'quotes " unicode 😀', "nested": [1, False]}
    assert loaded["func"](payload) == payload
    request = json.loads(mock_cli.read_text())
    assert request["argv"] == [
        "run",
        "@fixture/tool@0.1.9",
        "--machine",
        "--sdk-request",
        "--timeout-ms",
        "120000",
    ]
    assert request["cwd"] == str(project)
    assert request["request"]["tool_directory"] == str(directory)
    assert request["request"]["input"] == payload
    assert request["request"]["environment"]["LAYER"] == "caller"
    assert secret not in json.dumps(request["argv"])
    assert (project / "agent.lock").read_bytes() == before
    assert not (directory / "run").exists()
    assert loaded["func"]({"case": "tool-error"}) == {"ok": False, "error": "Tool business error"}


@pytest.mark.parametrize(
    "case", ["malformed", "incomplete", "wrong-schema", "wrong-tool", "nonzero", "huge"]
)
def test_machine_transport_rejects_invalid_responses(
    mock_cli: Path, tmp_path: Path, case: str
) -> None:
    with pytest.raises(RuntimeError, match="AgentPM CLI"):
        _cli.invoke(tmp_path, tmp_path, "@fixture/tool@0.1.9", {"case": case}, 1, {})
    if case == "wrong-tool":
        with pytest.raises(RuntimeError, match="AgentPM CLI"):
            _cli.invoke(
                tmp_path, tmp_path, "@fixture/tool@0.1.9", {"case": "wrong-tool-type"}, 1, {}
            )


def test_runner_errors_and_stderr_redact_secrets(
    mock_cli: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "fixture-credential-do-not-print"
    for case, kind in [("error", RuntimeError), ("timeout", TimeoutError)]:
        with pytest.raises(kind, match="run agentpm install again") as error:
            _cli.invoke(
                tmp_path,
                tmp_path,
                "@fixture/tool@0.1.9",
                {"case": case},
                1,
                {"MOCK_SECRET_TOKEN": secret},
            )
        assert secret not in str(error.value)
    assert secret not in capsys.readouterr().err


@pytest.mark.parametrize("version", ["agentpm 0.1.42", "agentpm 0.1.43-beta.1", "garbage"])
def test_old_cli_rejected(
    mock_cli: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setenv("MOCK_CLI_VERSION", version)
    with pytest.raises(RuntimeError, match="Install AgentPM CLI >=0.1.43"):
        _cli.invoke(tmp_path, tmp_path, "@fixture/tool@0.1.9", {}, 1, {})
    assert not mock_cli.exists()


def test_missing_cli_and_invalid_timeouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTPM", str(tmp_path / "does not exist"))
    with pytest.raises(RuntimeError, match="Install AgentPM CLI"):
        _cli.invoke(tmp_path, tmp_path, "@fixture/tool@0.1.9", {}, 1, {})
    for timeout in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="timeout"):
            _cli.invoke(tmp_path, tmp_path, "@fixture/tool@0.1.9", {}, timeout, {})


def write_agent(project: Path) -> None:
    for kind, name, version in [
        ("agent", "agent", "0.1.7"),
        ("agent", "agent", "0.1.6"),
        ("skill", "skill", "1.0.0"),
        ("tool", "tool", "0.1.9"),
        ("tool", "tool", "0.1.8"),
    ]:
        directory = project / f".agentpm/{kind}s/fixture/{name}/{version}"
        directory.mkdir(parents=True)
        (directory / "agent.json").write_text(
            json.dumps(
                {
                    "kind": kind,
                    "name": name,
                    "version": version,
                    "tools": ["@fixture/tool@0.1.9"],
                    "skill": {"entrypoint": "SKILL.md"},
                    "entrypoint": {"command": "python", "args": ["tool.py"]},
                }
            )
        )
        (directory / "SKILL.md").write_text("# Shared fixture")


@pytest.mark.parametrize("vector", VECTORS["valid"], ids=lambda v: v["name"])
def test_shared_supported_locks_are_read_only_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vector: dict
) -> None:
    write_agent(tmp_path)
    lock = tmp_path / "agent.lock"
    lock.write_text(json.dumps(vector["lock"]))
    before = lock.read_bytes()
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    agent = load_agent(VECTORS["agent"])
    assert agent["manifest"]["version"] == "0.1.7"
    assert agent["resolvedTools"][0]["version"] == "0.1.9"
    assert len(agent["resolvedSkills"]) == 1
    for field in ["resolvedKnowledge", "resolvedMemory", "resolvedProfiles"]:
        assert len(agent[field]) == 1
    assert agent["resolvedLoop"]["version"] == "1.0.0"
    skill = load_skill("@fixture/skill@1.0.0")
    assert skill["resolvedTools"][0]["version"] == "0.1.9"
    assert lock.read_bytes() == before
    assert not (tmp_path / "agent.json").exists()
    with pytest.raises(FileNotFoundError):
        load_agent("@fixture/local-author@1.0.0")


@pytest.mark.parametrize("vector", VECTORS["invalid"], ids=lambda v: v["name"])
def test_shared_invalid_locks_fail_without_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vector: dict
) -> None:
    write_agent(tmp_path)
    lock = tmp_path / "agent.lock"
    lock.write_text(json.dumps(vector["lock"]))
    before = lock.read_bytes()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="lock"):
        load_agent(VECTORS["agent"])
    assert lock.read_bytes() == before


def test_interruption_terminates_cli(
    mock_cli: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_time = _cli.time.monotonic
    original_stop = _cli._stop
    calls = 0
    stopped = []

    def interrupt() -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_time()

    def stop(process: subprocess.Popen[bytes]) -> None:
        original_stop(process)
        stopped.append(process.poll())

    monkeypatch.setattr(_cli.time, "monotonic", interrupt)
    monkeypatch.setattr(_cli, "_stop", stop)
    with pytest.raises(KeyboardInterrupt):
        _cli.invoke(tmp_path, tmp_path, "@fixture/tool@0.1.9", {"case": "hang"}, 1, {})
    assert len(stopped) == 1 and stopped[0] is not None


def test_rejects_changed_selected_manifest_version(
    mock_cli: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = write_tool(tmp_path)
    manifest = directory / "agent.json"
    value = json.loads(manifest.read_text())
    value["version"] = "0.1.8"
    manifest.write_text(json.dumps(value))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="version mismatch"):
        load("@fixture/tool@0.1.9")
    assert not mock_cli.exists()
