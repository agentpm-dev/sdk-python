from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentpm import load_agent
from agentpm.core import find_project_root


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


def _write_installed_skill(base_dir: Path, spec: str) -> None:
    package_name, version = _split_spec(spec)
    root = base_dir / package_name / version
    manifest_name = package_name.split("/", 1)[1]
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text("# Triage playbook\n\nUse the checklist.\n", encoding="utf-8")
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "skill",
                "name": manifest_name,
                "version": version,
                "description": "Installed skill fixture",
                "tools": ["@zack/capitalize@0.1.0"],
                "skill": {"entrypoint": "SKILL.md"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_installed_agent(base_dir: Path, spec: str, skill_ref: str) -> None:
    package_name, version = _split_spec(spec)
    root = base_dir / package_name / version
    manifest_name = package_name.split("/", 1)[1]
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "agent",
                "name": manifest_name,
                "version": version,
                "description": "Installed agent fixture",
                "tools": ["@zack/capitalize@0.1.0"],
                "examples": [{"title": "Example", "prompt": "Help the user."}],
                "skills": [skill_ref],
                "knowledge": [],
                "memory": [],
                "profiles": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def tmp_agent_workspace() -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="agentpm-sdk-py-agent-")).resolve()
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_agent_loads_installed_agent_and_resolved_tools_and_skills(
    tmp_agent_workspace: Path,
) -> None:
    tools_dir = tmp_agent_workspace / ".agentpm" / "tools"
    agents_dir = tmp_agent_workspace / ".agentpm" / "agents"
    skills_dir = tmp_agent_workspace / ".agentpm" / "skills"
    lockfile_path = tmp_agent_workspace / "agent.lock"
    agent_spec = "@zack/support-agent@0.1.0"

    _write_installed_tool(tools_dir, "@zack/capitalize@0.1.0")
    _write_installed_skill(skills_dir, "@zack/triage-skill@0.1.0")
    _write_installed_agent(agents_dir, agent_spec, "@zack/triage-skill@0.1.0")
    lockfile_path.write_text(
        json.dumps(
            {
                "lockfile_version": 3,
                "generated": "2026-05-23T00:00:00Z",
                "packages": {
                    "agent:@zack/support-agent@0.1.0": {
                        "kind": "agent",
                        "name": "@zack/support-agent",
                        "version": "0.1.0",
                        "integrity": "sha256-agent",
                    },
                    "tool:@zack/capitalize@0.1.0": {
                        "kind": "tool",
                        "name": "@zack/capitalize",
                        "version": "0.1.0",
                        "integrity": "sha256-tool",
                    },
                    "skill:@zack/triage-skill@0.1.0": {
                        "kind": "skill",
                        "name": "@zack/triage-skill",
                        "version": "0.1.0",
                        "integrity": "sha256-skill",
                    },
                },
                "roots": {
                    "agent:@zack/support-agent@0.1.0": {
                        "tools": ["tool:@zack/capitalize@0.1.0"],
                        "skills": ["skill:@zack/triage-skill@0.1.0"],
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

    loaded = load_agent(
        agent_spec,
        agent_dir_override=str(agents_dir),
        skill_dir_override=str(skills_dir),
        tool_dir_override=str(tools_dir),
        lockfile_override=str(lockfile_path),
    )

    assert loaded["manifest"]["kind"] == "agent"
    assert loaded["manifest"]["name"] == "support-agent"
    assert ".agentpm/agents" in loaded["root"]
    assert loaded["reserved"]["skills"] == []
    assert loaded["resolvedTools"] == [
        {
            "packageKey": "tool:@zack/capitalize@0.1.0",
            "kind": "tool",
            "name": "@zack/capitalize",
            "version": "0.1.0",
            "integrity": "sha256-tool",
            "root": str(agents_dir.parent / "tools" / "@zack" / "capitalize" / "0.1.0"),
            "manifestPath": str(
                agents_dir.parent / "tools" / "@zack" / "capitalize" / "0.1.0" / "agent.json"
            ),
        }
    ]
    assert loaded["resolvedSkills"] == [
        {
            "packageKey": "skill:@zack/triage-skill@0.1.0",
            "kind": "skill",
            "name": "@zack/triage-skill",
            "version": "0.1.0",
            "integrity": "sha256-skill",
            "root": str(skills_dir / "@zack" / "triage-skill" / "0.1.0"),
            "manifestPath": str(skills_dir / "@zack" / "triage-skill" / "0.1.0" / "agent.json"),
        }
    ]


def test_load_agent_resolves_latest_and_ranges(tmp_agent_workspace: Path) -> None:
    tools_dir = tmp_agent_workspace / ".agentpm" / "tools"
    agents_dir = tmp_agent_workspace / ".agentpm" / "agents"
    skills_dir = tmp_agent_workspace / ".agentpm" / "skills"
    lockfile_path = tmp_agent_workspace / "agent-range.lock"
    exact_spec = "@zack/support-agent@0.1.0"
    newer_spec = "@zack/support-agent@0.2.0"

    _write_installed_tool(tools_dir, "@zack/capitalize@0.1.0")
    _write_installed_skill(skills_dir, "@zack/triage-skill@0.2.0")
    _write_installed_agent(agents_dir, exact_spec, "@zack/triage-skill@0.1.0")
    _write_installed_agent(agents_dir, newer_spec, "@zack/triage-skill@0.2.0")
    lockfile_path.write_text(
        json.dumps(
            {
                "lockfile_version": 3,
                "generated": "2026-05-23T00:00:00Z",
                "packages": {
                    "agent:@zack/support-agent@0.1.0": {
                        "kind": "agent",
                        "name": "@zack/support-agent",
                        "version": "0.1.0",
                        "integrity": "sha256-agent",
                    },
                    "agent:@zack/support-agent@0.2.0": {
                        "kind": "agent",
                        "name": "@zack/support-agent",
                        "version": "0.2.0",
                        "integrity": "sha256-agent-2",
                    },
                    "tool:@zack/capitalize@0.1.0": {
                        "kind": "tool",
                        "name": "@zack/capitalize",
                        "version": "0.1.0",
                        "integrity": "sha256-tool",
                    },
                    "skill:@zack/triage-skill@0.2.0": {
                        "kind": "skill",
                        "name": "@zack/triage-skill",
                        "version": "0.2.0",
                        "integrity": "sha256-skill-2",
                    },
                },
                "roots": {
                    "agent:@zack/support-agent@0.1.0": {
                        "tools": ["tool:@zack/capitalize@0.1.0"],
                        "skills": [],
                        "reserved": {
                            "knowledge": [],
                            "memory": [],
                            "profiles": [],
                        },
                    },
                    "agent:@zack/support-agent@0.2.0": {
                        "tools": ["tool:@zack/capitalize@0.1.0"],
                        "skills": ["skill:@zack/triage-skill@0.2.0"],
                        "reserved": {
                            "knowledge": [],
                            "memory": [],
                            "profiles": [],
                        },
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    latest = load_agent(
        "@zack/support-agent@latest",
        agent_dir_override=str(agents_dir),
        skill_dir_override=str(skills_dir),
        tool_dir_override=str(tools_dir),
        lockfile_override=str(lockfile_path),
    )
    ranged = load_agent(
        "@zack/support-agent@>=0.1.0 <0.3.0",
        agent_dir_override=str(agents_dir),
        skill_dir_override=str(skills_dir),
        tool_dir_override=str(tools_dir),
        lockfile_override=str(lockfile_path),
    )

    assert latest["manifest"]["version"] == "0.2.0"
    assert latest["resolvedSkills"][0]["version"] == "0.2.0"
    assert ranged["manifest"]["version"] == "0.2.0"
    assert ranged["resolvedSkills"][0]["version"] == "0.2.0"


def test_load_agent_fails_when_lockfile_is_missing(tmp_agent_workspace: Path) -> None:
    tools_dir = tmp_agent_workspace / ".agentpm" / "tools"
    agents_dir = tmp_agent_workspace / ".agentpm" / "agents"
    skills_dir = tmp_agent_workspace / ".agentpm" / "skills"
    agent_spec = "@zack/support-agent@0.1.0"
    _write_installed_agent(agents_dir, agent_spec, "@zack/triage-skill@0.1.0")

    missing_lock_path = tmp_agent_workspace / "missing-agent.lock"
    with pytest.raises(FileNotFoundError, match="agentpm install"):
        load_agent(
            agent_spec,
            agent_dir_override=str(agents_dir),
            skill_dir_override=str(skills_dir),
            tool_dir_override=str(tools_dir),
            lockfile_override=str(missing_lock_path),
        )


def test_load_agent_fails_when_lockfile_is_v1(tmp_agent_workspace: Path) -> None:
    tools_dir = tmp_agent_workspace / ".agentpm" / "tools"
    agents_dir = tmp_agent_workspace / ".agentpm" / "agents"
    skills_dir = tmp_agent_workspace / ".agentpm" / "skills"
    agent_spec = "@zack/support-agent@0.1.0"
    _write_installed_agent(agents_dir, agent_spec, "@zack/triage-skill@0.1.0")

    v1_lockfile_path = tmp_agent_workspace / "agent-v1.lock"
    v1_lockfile_path.write_text(
        json.dumps(
            {"lockfile_version": 1, "generated": "2026-05-23T00:00:00Z", "dependencies": {}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="agentpm install"):
        load_agent(
            agent_spec,
            agent_dir_override=str(agents_dir),
            skill_dir_override=str(skills_dir),
            tool_dir_override=str(tools_dir),
            lockfile_override=str(v1_lockfile_path),
        )


def test_load_agent_fails_when_agent_root_is_missing_from_lockfile(
    tmp_agent_workspace: Path,
) -> None:
    tools_dir = tmp_agent_workspace / ".agentpm" / "tools"
    agents_dir = tmp_agent_workspace / ".agentpm" / "agents"
    skills_dir = tmp_agent_workspace / ".agentpm" / "skills"
    agent_spec = "@zack/support-agent@0.1.0"
    _write_installed_agent(agents_dir, agent_spec, "@zack/triage-skill@0.1.0")

    wrong_root_lockfile_path = tmp_agent_workspace / "agent-missing-root.lock"
    wrong_root_lockfile_path.write_text(
        json.dumps(
            {
                "lockfile_version": 3,
                "generated": "2026-05-23T00:00:00Z",
                "packages": {
                    "agent:@zack/support-agent@0.1.0": {
                        "kind": "agent",
                        "name": "@zack/support-agent",
                        "version": "0.1.0",
                        "integrity": "sha256-agent",
                    }
                },
                "roots": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="install the agent with agentpm install first"):
        load_agent(
            agent_spec,
            agent_dir_override=str(agents_dir),
            skill_dir_override=str(skills_dir),
            tool_dir_override=str(tools_dir),
            lockfile_override=str(wrong_root_lockfile_path),
        )


def test_load_agent_returns_metadata_when_resolved_tool_is_missing_on_disk(
    tmp_agent_workspace: Path,
) -> None:
    tools_dir = tmp_agent_workspace / ".agentpm" / "tools"
    agents_dir = tmp_agent_workspace / ".agentpm" / "agents"
    skills_dir = tmp_agent_workspace / ".agentpm" / "skills"
    agent_spec = "@zack/support-agent@0.1.0"
    _write_installed_agent(agents_dir, agent_spec, "@zack/triage-skill@0.1.0")

    missing_tool_lockfile_path = tmp_agent_workspace / "agent-missing-tool.lock"
    missing_tool_lockfile_path.write_text(
        json.dumps(
            {
                "lockfile_version": 3,
                "generated": "2026-05-23T00:00:00Z",
                "packages": {
                    "agent:@zack/support-agent@0.1.0": {
                        "kind": "agent",
                        "name": "@zack/support-agent",
                        "version": "0.1.0",
                        "integrity": "sha256-agent",
                    },
                    "tool:@zack/missing-tool@0.9.0": {
                        "kind": "tool",
                        "name": "@zack/missing-tool",
                        "version": "0.9.0",
                        "integrity": "sha256-missing-tool",
                    },
                },
                "roots": {
                    "agent:@zack/support-agent@0.1.0": {
                        "tools": ["tool:@zack/missing-tool@0.9.0"],
                        "skills": [],
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
    loaded = load_agent(
        agent_spec,
        agent_dir_override=str(agents_dir),
        skill_dir_override=str(skills_dir),
        tool_dir_override=str(tools_dir),
        lockfile_override=str(missing_tool_lockfile_path),
    )
    assert loaded["resolvedTools"] == [
        {
            "packageKey": "tool:@zack/missing-tool@0.9.0",
            "kind": "tool",
            "name": "@zack/missing-tool",
            "version": "0.9.0",
            "integrity": "sha256-missing-tool",
            "root": None,
            "manifestPath": None,
        }
    ]


def test_find_project_root_prefers_pyproject_for_python_apps(tmp_agent_workspace: Path) -> None:
    app_root = tmp_agent_workspace / "python-app"
    nested = app_root / "app"
    nested.mkdir(parents=True, exist_ok=True)
    (app_root / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    (tmp_agent_workspace / ".git").mkdir(exist_ok=True)

    resolved = find_project_root(nested)
    assert resolved == app_root.resolve()
