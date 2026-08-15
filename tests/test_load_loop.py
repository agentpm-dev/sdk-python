from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentpm import load_loop


def _split_spec(spec: str) -> tuple[str, str]:
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f"Bad spec: {spec}")
    return spec[:at], spec[at + 1 :]


def _write_installed_loop(base_dir: Path, spec: str, loop: dict[str, object]) -> Path:
    package_name, version = _split_spec(spec)
    root = base_dir / package_name / version
    manifest_name = package_name.split("/", 1)[1]
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "loop",
                "name": manifest_name,
                "version": version,
                "description": "Installed loop fixture",
                "loop": loop,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")  # type: ignore[misc]
def tmp_loop_workspace() -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="agentpm-sdk-py-loop-")).resolve()
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_loop_loads_minimal_loop_metadata(tmp_loop_workspace: Path) -> None:
    loop_dir = tmp_loop_workspace / ".agentpm" / "loops"
    _write_installed_loop(
        loop_dir,
        "@zack/incident-response-loop@0.1.0",
        {
            "entry_phase": "assess",
            "phases": [{"id": "assess", "objective": "Assess the request."}],
            "transitions": [{"from": "assess", "on": "complete", "to": "$end"}],
        },
    )

    loaded = load_loop(
        "@zack/incident-response-loop@0.1.0",
        loop_dir_override=str(loop_dir),
    )

    assert loaded["kind"] == "loop"
    assert loaded["loop"]["entry_phase"] == "assess"
    assert loaded["loop"]["transitions"] == [{"from": "assess", "on": "complete", "to": "$end"}]


def test_load_loop_loads_full_loop_metadata(tmp_loop_workspace: Path) -> None:
    loop_dir = tmp_loop_workspace / ".agentpm" / "loops"
    _write_installed_loop(
        loop_dir,
        "@zack/review-loop@0.2.0",
        {
            "archetype": "investigate_review_respond",
            "entry_phase": "triage",
            "limits": {"max_steps": 16},
            "phases": [
                {
                    "id": "triage",
                    "objective": "Assess whether work should continue.",
                    "access": {
                        "tools": False,
                        "knowledge": True,
                        "memory": {"read": True, "write": False},
                    },
                    "outcomes": [
                        {"id": "proceed", "description": "Continue into execution."},
                        {"id": "handoff", "description": "Transfer ownership."},
                    ],
                },
                {"id": "execute", "objective": "Complete the work."},
            ],
            "transitions": [
                {"from": "triage", "on": "proceed", "to": "execute"},
                {"from": "triage", "on": "handoff", "to": "$handoff"},
                {"from": "execute", "on": "complete", "to": "$end"},
            ],
            "checkpoints": [
                {
                    "id": "approve-response",
                    "type": "approval",
                    "before_phase": "execute",
                    "on_reject": "$abort",
                }
            ],
            "error_policy": {
                "tool_failure": {
                    "action": "retry",
                    "max_retries": 2,
                    "on_exhausted": "fail_phase",
                },
                "phase_failure": {"action": "handoff"},
            },
        },
    )

    loaded = load_loop(
        "@zack/review-loop@0.2.0",
        loop_dir_override=str(loop_dir),
    )

    assert loaded["loop"]["archetype"] == "investigate_review_respond"
    assert loaded["loop"]["limits"]["max_steps"] == 16
    assert loaded["loop"]["phases"][0]["access"]["memory"]["read"] is True
    assert loaded["loop"]["checkpoints"][0]["before_phase"] == "execute"
    assert loaded["loop"]["error_policy"]["tool_failure"]["action"] == "retry"


def test_load_loop_fails_when_package_is_missing(tmp_loop_workspace: Path) -> None:
    loop_dir = tmp_loop_workspace / ".agentpm" / "loops"

    with pytest.raises(FileNotFoundError, match=r"not found in \.agentpm/loops"):
        load_loop(
            "@zack/missing-loop@0.1.0",
            loop_dir_override=str(loop_dir),
        )


def test_load_loop_fails_when_manifest_is_wrong_kind(tmp_loop_workspace: Path) -> None:
    loop_dir = tmp_loop_workspace / ".agentpm" / "loops"
    root = _write_installed_loop(
        loop_dir,
        "@zack/not-loop@0.1.0",
        {
            "entry_phase": "assess",
            "phases": [{"id": "assess", "objective": "Assess the request."}],
            "transitions": [{"from": "assess", "on": "complete", "to": "$end"}],
        },
    )
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "tool",
                "name": "not-loop",
                "version": "0.1.0",
                "entrypoint": {"command": "python", "args": ["tool.py"]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a loop manifest"):
        load_loop(
            "@zack/not-loop@0.1.0",
            loop_dir_override=str(loop_dir),
        )


def test_load_loop_fails_when_manifest_is_missing_loop_object(tmp_loop_workspace: Path) -> None:
    loop_dir = tmp_loop_workspace / ".agentpm" / "loops"
    root = _write_installed_loop(
        loop_dir,
        "@zack/missing-loop-object@0.1.0",
        {
            "entry_phase": "assess",
            "phases": [{"id": "assess", "objective": "Assess the request."}],
            "transitions": [{"from": "assess", "on": "complete", "to": "$end"}],
        },
    )
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "loop",
                "name": "missing-loop-object",
                "version": "0.1.0",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing loop object"):
        load_loop(
            "@zack/missing-loop-object@0.1.0",
            loop_dir_override=str(loop_dir),
        )
