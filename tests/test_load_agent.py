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


def _write_installed_knowledge(base_dir: Path, spec: str, mode: str) -> None:
    package_name, version = _split_spec(spec)
    root = base_dir / package_name / version
    manifest_name = package_name.split("/", 1)[1]
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        json.dumps(
            (
                {
                    "kind": "knowledge",
                    "name": manifest_name,
                    "version": version,
                    "description": "Installed knowledge fixture",
                    "knowledge": {
                        "mode": "context",
                        "documents": [{"path": "knowledge/docs/context.md"}],
                    },
                }
                if mode == "context"
                else {
                    "kind": "knowledge",
                    "name": manifest_name,
                    "version": version,
                    "description": "Installed knowledge fixture",
                    "knowledge": {
                        "mode": "vector",
                        "corpus": {
                            "chunks_path": "knowledge/chunks.jsonl",
                            "sources_path": "knowledge/sources.jsonl",
                        },
                        "embedding": {"vectors_path": "knowledge/embeddings/default.f32"},
                        "indexes": [{"path": "knowledge/indexes/default"}],
                    },
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_installed_memory(base_dir: Path, spec: str) -> None:
    package_name, version = _split_spec(spec)
    root = base_dir / package_name / version
    manifest_name = package_name.split("/", 1)[1]
    (root / "schemas").mkdir(parents=True, exist_ok=True)
    (root / "memory" / "contracts").mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "memory",
                "name": manifest_name,
                "version": version,
                "description": "Installed memory fixture",
                "memory": {
                    "scopes": {"user": {"description": "User scope"}},
                    "record_types": {
                        "user_preference": {
                            "schema": "schemas/user-preference.schema.json",
                            "version": "1.0.0",
                        }
                    },
                    "spaces": {
                        "profile": {
                            "model": "document",
                            "scope": ["user"],
                            "record_types": ["user_preference"],
                            "retrieval": {"modes": ["key"]},
                        }
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "schemas" / "user-preference.schema.json").write_text(
        json.dumps({"type": "object"}, indent=2),
        encoding="utf-8",
    )
    (root / "memory" / "build.json").write_text(
        json.dumps(
            {
                "type": "agentpm-memory-contracts",
                "format_version": 1,
                "manifest_path": "agent.json",
                "source_manifest_hash": "sha256:manifest",
                "source_schemas_hash": "sha256:schemas",
                "source_contract_inputs_hash": "sha256:inputs",
                "contracts_index_hash": "sha256:index",
                "contracts_hash": "sha256:contracts",
                "contract_count": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "memory" / "contracts" / "index.json").write_text(
        json.dumps(
            {
                "type": "agentpm-memory-contract-index",
                "format_version": 1,
                "contracts": [
                    {
                        "space": "profile",
                        "record_type": "user_preference",
                        "schema_version": "1.0.0",
                        "model": "document",
                        "source_schema": "schemas/user-preference.schema.json",
                        "path": "memory/contracts/profile.user_preference.schema.json",
                        "sha256": "sha256:contract",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "memory" / "contracts" / "profile.user_preference.schema.json").write_text(
        json.dumps({"type": "object"}, indent=2),
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
    knowledge_dir = tmp_agent_workspace / ".agentpm" / "knowledge"
    memory_dir = tmp_agent_workspace / ".agentpm" / "memory"
    lockfile_path = tmp_agent_workspace / "agent.lock"
    agent_spec = "@zack/support-agent@0.1.0"

    _write_installed_tool(tools_dir, "@zack/capitalize@0.1.0")
    _write_installed_skill(skills_dir, "@zack/triage-skill@0.1.0")
    _write_installed_knowledge(knowledge_dir, "@zack/python-docs@0.1.0", "vector")
    _write_installed_memory(memory_dir, "@zack/profile-memory@0.1.0")
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
                    "knowledge:@zack/python-docs@0.1.0": {
                        "kind": "knowledge",
                        "name": "@zack/python-docs",
                        "version": "0.1.0",
                        "integrity": "sha256-knowledge",
                    },
                    "memory:@zack/profile-memory@0.1.0": {
                        "kind": "memory",
                        "name": "@zack/profile-memory",
                        "version": "0.1.0",
                        "integrity": "sha256-memory",
                    },
                },
                "roots": {
                    "agent:@zack/support-agent@0.1.0": {
                        "tools": ["tool:@zack/capitalize@0.1.0"],
                        "skills": ["skill:@zack/triage-skill@0.1.0"],
                        "knowledge": ["knowledge:@zack/python-docs@0.1.0"],
                        "memory": ["memory:@zack/profile-memory@0.1.0"],
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
        knowledge_dir_override=str(knowledge_dir),
        memory_dir_override=str(memory_dir),
        lockfile_override=str(lockfile_path),
    )

    assert loaded["manifest"]["kind"] == "agent"
    assert loaded["manifest"]["name"] == "support-agent"
    assert ".agentpm/agents" in loaded["root"]
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
    assert loaded["resolvedKnowledge"] == [
        {
            "packageKey": "knowledge:@zack/python-docs@0.1.0",
            "kind": "knowledge",
            "name": "@zack/python-docs",
            "version": "0.1.0",
            "integrity": "sha256-knowledge",
            "mode": "vector",
            "root": str(knowledge_dir / "@zack" / "python-docs" / "0.1.0"),
            "manifestPath": str(knowledge_dir / "@zack" / "python-docs" / "0.1.0" / "agent.json"),
        }
    ]
    assert loaded["resolvedMemory"] == [
        {
            "packageKey": "memory:@zack/profile-memory@0.1.0",
            "kind": "memory",
            "name": "@zack/profile-memory",
            "version": "0.1.0",
            "integrity": "sha256-memory",
            "root": str(memory_dir / "@zack" / "profile-memory" / "0.1.0"),
            "manifestPath": str(memory_dir / "@zack" / "profile-memory" / "0.1.0" / "agent.json"),
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


def test_load_agent_keeps_memory_refs_when_package_is_missing_on_disk(
    tmp_agent_workspace: Path,
) -> None:
    agents_dir = tmp_agent_workspace / ".agentpm" / "agents"
    memory_dir = tmp_agent_workspace / ".agentpm" / "memory"
    lockfile_path = tmp_agent_workspace / "agent-missing-memory.lock"
    agent_spec = "@zack/support-agent@0.1.0"

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
                    "memory:@zack/missing-memory@0.9.0": {
                        "kind": "memory",
                        "name": "@zack/missing-memory",
                        "version": "0.9.0",
                        "integrity": "sha256-missing-memory",
                    },
                },
                "roots": {
                    "agent:@zack/support-agent@0.1.0": {
                        "memory": ["memory:@zack/missing-memory@0.9.0"],
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
        memory_dir_override=str(memory_dir),
        lockfile_override=str(lockfile_path),
    )

    assert loaded["resolvedMemory"] == [
        {
            "packageKey": "memory:@zack/missing-memory@0.9.0",
            "kind": "memory",
            "name": "@zack/missing-memory",
            "version": "0.9.0",
            "integrity": "sha256-missing-memory",
            "root": None,
            "manifestPath": None,
        }
    ]


def test_load_agent_ignores_legacy_reserved_skills_entries(
    tmp_agent_workspace: Path,
) -> None:
    tools_dir = tmp_agent_workspace / ".agentpm" / "tools"
    agents_dir = tmp_agent_workspace / ".agentpm" / "agents"
    skills_dir = tmp_agent_workspace / ".agentpm" / "skills"
    lockfile_path = tmp_agent_workspace / "agent-legacy.lock"
    agent_spec = "@zack/support-agent@0.1.0"

    _write_installed_tool(tools_dir, "@zack/capitalize@0.1.0")
    _write_installed_agent(agents_dir, agent_spec, "@zack/triage-skill@0.1.0")
    lockfile_path.write_text(
        json.dumps(
            {
                "lockfile_version": 2,
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
                },
                "roots": {
                    "agent:@zack/support-agent@0.1.0": {
                        "tools": ["tool:@zack/capitalize@0.1.0"],
                        "reserved": {
                            "skills": ["skill:@zack/legacy-skill@0.1.0"],
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

    assert loaded["reserved"] == {
        "knowledge": [],
        "memory": [],
        "profiles": [],
    }
    assert "skills" not in loaded["reserved"]
    assert loaded["resolvedSkills"] == []


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


def test_load_agent_returns_metadata_when_resolved_knowledge_is_missing_on_disk(
    tmp_agent_workspace: Path,
) -> None:
    tools_dir = tmp_agent_workspace / ".agentpm" / "tools"
    agents_dir = tmp_agent_workspace / ".agentpm" / "agents"
    skills_dir = tmp_agent_workspace / ".agentpm" / "skills"
    knowledge_dir = tmp_agent_workspace / ".agentpm" / "knowledge"
    agent_spec = "@zack/support-agent@0.1.0"
    _write_installed_agent(agents_dir, agent_spec, "@zack/triage-skill@0.1.0")

    missing_knowledge_lockfile_path = tmp_agent_workspace / "agent-missing-knowledge.lock"
    missing_knowledge_lockfile_path.write_text(
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
                    "knowledge:@zack/missing-docs@0.9.0": {
                        "kind": "knowledge",
                        "name": "@zack/missing-docs",
                        "version": "0.9.0",
                        "integrity": "sha256-missing-knowledge",
                    },
                },
                "roots": {
                    "agent:@zack/support-agent@0.1.0": {
                        "tools": [],
                        "skills": [],
                        "knowledge": ["knowledge:@zack/missing-docs@0.9.0"],
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
        knowledge_dir_override=str(knowledge_dir),
        lockfile_override=str(missing_knowledge_lockfile_path),
    )
    assert loaded["resolvedKnowledge"] == [
        {
            "packageKey": "knowledge:@zack/missing-docs@0.9.0",
            "kind": "knowledge",
            "name": "@zack/missing-docs",
            "version": "0.9.0",
            "integrity": "sha256-missing-knowledge",
            "mode": None,
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
