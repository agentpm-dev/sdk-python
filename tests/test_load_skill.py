from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentpm import load_skill


def _split_spec(spec: str) -> tuple[str, str]:
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f"Bad spec: {spec}")
    return spec[:at], spec[at + 1 :]


def _write_installed_tool(base_dir: Path, spec: str) -> None:
    name, version = _split_spec(spec)
    root = base_dir / name / version
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "tool",
                "name": name,
                "version": version,
                "description": "Installed tool fixture",
                "entrypoint": {"command": "python", "args": ["tool.py"]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_installed_skill(base_dir: Path, spec: str, with_tools: bool = True) -> None:
    package_name, version = _split_spec(spec)
    root = base_dir / package_name / version
    manifest_name = package_name.split("/", 1)[1]
    (root / "references").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text("# Triage playbook\n\nUse the checklist.\n", encoding="utf-8")
    (root / "references" / "tool-contract.md").write_text("contract\n", encoding="utf-8")
    (root / "scripts" / "run.sh").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "skill",
                "name": manifest_name,
                "version": version,
                "description": "Installed skill fixture",
                "tools": ["@zack/capitalize@0.1.0"] if with_tools else [],
                "skill": {
                    "entrypoint": "SKILL.md",
                    "references": ["references/tool-contract.md"],
                    "scripts": ["scripts/run.sh"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def tmp_skill_workspace() -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="agentpm-sdk-py-skill-")).resolve()
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_skill_loads_procedural_skill_with_no_tools(tmp_skill_workspace: Path) -> None:
    tools_dir = tmp_skill_workspace / ".agentpm" / "tools"
    skills_dir = tmp_skill_workspace / ".agentpm" / "skills"
    lockfile_path = tmp_skill_workspace / "agent.lock"

    _write_installed_skill(skills_dir, "@zack/procedural-only@0.1.0", with_tools=False)
    lockfile_path.write_text(
        json.dumps(
            {
                "lockfile_version": 3,
                "generated": "2026-05-23T00:00:00Z",
                "packages": {
                    "skill:@zack/procedural-only@0.1.0": {
                        "kind": "skill",
                        "name": "@zack/procedural-only",
                        "version": "0.1.0",
                        "integrity": "sha256-procedural",
                    }
                },
                "roots": {
                    "skill:@zack/procedural-only@0.1.0": {
                        "tools": [],
                        "reserved": {
                            "knowledge": [],
                            "memory": [],
                            "profiles": [],
                        },
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = load_skill(
        "@zack/procedural-only@0.1.0",
        skill_dir_override=str(skills_dir),
        tool_dir_override=str(tools_dir),
        lockfile_override=str(lockfile_path),
    )

    assert loaded["kind"] == "skill"
    assert loaded["name"] == "procedural-only"
    assert "Triage playbook" in loaded["entrypointContent"]
    assert loaded["references"] == ["references/tool-contract.md"]
    assert loaded["scripts"] == ["scripts/run.sh"]
    assert loaded["resolvedTools"] == []


def test_load_skill_loads_tool_backed_skill_and_resolved_tools(tmp_skill_workspace: Path) -> None:
    tools_dir = tmp_skill_workspace / ".agentpm" / "tools"
    skills_dir = tmp_skill_workspace / ".agentpm" / "skills"
    lockfile_path = tmp_skill_workspace / "agent-with-tools.lock"

    _write_installed_tool(tools_dir, "@zack/capitalize@0.1.0")
    _write_installed_skill(skills_dir, "@zack/triage-playbook@0.1.0", with_tools=True)
    lockfile_path.write_text(
        json.dumps(
            {
                "lockfile_version": 3,
                "generated": "2026-05-23T00:00:00Z",
                "packages": {
                    "skill:@zack/triage-playbook@0.1.0": {
                        "kind": "skill",
                        "name": "@zack/triage-playbook",
                        "version": "0.1.0",
                        "integrity": "sha256-skill",
                    },
                    "tool:@zack/capitalize@0.1.0": {
                        "kind": "tool",
                        "name": "@zack/capitalize",
                        "version": "0.1.0",
                        "integrity": "sha256-tool",
                    },
                },
                "roots": {
                    "skill:@zack/triage-playbook@0.1.0": {
                        "tools": ["tool:@zack/capitalize@0.1.0"],
                        "reserved": {
                            "knowledge": [],
                            "memory": [],
                            "profiles": [],
                        },
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = load_skill(
        "@zack/triage-playbook@0.1.0",
        skill_dir_override=str(skills_dir),
        tool_dir_override=str(tools_dir),
        lockfile_override=str(lockfile_path),
    )

    assert ".agentpm/skills" in loaded["entrypointPath"]
    assert "Use the checklist." in loaded["entrypointContent"]
    assert loaded["resolvedTools"] == [
        {
            "packageKey": "tool:@zack/capitalize@0.1.0",
            "kind": "tool",
            "name": "@zack/capitalize",
            "version": "0.1.0",
            "integrity": "sha256-tool",
            "root": str(skills_dir.parent / "tools" / "@zack" / "capitalize" / "0.1.0"),
            "manifestPath": str(
                skills_dir.parent / "tools" / "@zack" / "capitalize" / "0.1.0" / "agent.json"
            ),
        }
    ]


def test_load_skill_falls_back_to_manifest_tools_when_skill_root_is_absent(
    tmp_skill_workspace: Path,
) -> None:
    tools_dir = tmp_skill_workspace / ".agentpm" / "tools"
    skills_dir = tmp_skill_workspace / ".agentpm" / "skills"
    lockfile_path = tmp_skill_workspace / "agent-owned-skill.lock"

    _write_installed_tool(tools_dir, "@zack/capitalize@0.1.0")
    _write_installed_skill(skills_dir, "@zack/triage-from-agent@0.1.0", with_tools=True)
    lockfile_path.write_text(
        json.dumps(
            {
                "lockfile_version": 3,
                "generated": "2026-06-29T00:00:00Z",
                "packages": {
                    "agent:@zack/ops-console@0.1.1": {
                        "kind": "agent",
                        "name": "@zack/ops-console",
                        "version": "0.1.1",
                        "integrity": "sha256-agent",
                    },
                    "skill:@zack/triage-from-agent@0.1.0": {
                        "kind": "skill",
                        "name": "@zack/triage-from-agent",
                        "version": "0.1.0",
                        "integrity": "sha256-skill",
                    },
                    "tool:@zack/capitalize@0.1.0": {
                        "kind": "tool",
                        "name": "@zack/capitalize",
                        "version": "0.1.0",
                        "integrity": "sha256-tool",
                    },
                },
                "roots": {
                    "agent:@zack/ops-console@0.1.1": {
                        "skills": ["skill:@zack/triage-from-agent@0.1.0"],
                        "tools": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = load_skill(
        "@zack/triage-from-agent@0.1.0",
        skill_dir_override=str(skills_dir),
        tool_dir_override=str(tools_dir),
        lockfile_override=str(lockfile_path),
    )

    assert loaded["kind"] == "skill"
    assert loaded["name"] == "triage-from-agent"
    assert "Use the checklist." in loaded["entrypointContent"]
    assert loaded["resolvedTools"] == [
        {
            "packageKey": "tool:@zack/capitalize@0.1.0",
            "kind": "tool",
            "name": "@zack/capitalize",
            "version": "0.1.0",
            "integrity": "sha256-tool",
            "root": str(skills_dir.parent / "tools" / "@zack" / "capitalize" / "0.1.0"),
            "manifestPath": str(
                skills_dir.parent / "tools" / "@zack" / "capitalize" / "0.1.0" / "agent.json"
            ),
        }
    ]
