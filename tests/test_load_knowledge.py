from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentpm import load_knowledge


def _split_spec(spec: str) -> tuple[str, str]:
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f"Bad spec: {spec}")
    return spec[:at], spec[at + 1 :]


def _write_installed_knowledge(base_dir: Path, spec: str, mode: str) -> None:
    package_name, version = _split_spec(spec)
    root = base_dir / package_name / version
    manifest_name = package_name.split("/", 1)[1]
    root.mkdir(parents=True, exist_ok=True)

    if mode == "context":
        (root / "knowledge" / "docs").mkdir(parents=True, exist_ok=True)
        (root / "knowledge" / "docs" / "context.md").write_text("# Context\n", encoding="utf-8")
    else:
        (root / "knowledge" / "indexes" / "default").mkdir(parents=True, exist_ok=True)
        (root / "knowledge" / "chunks.jsonl").write_text("", encoding="utf-8")
        (root / "knowledge" / "sources.jsonl").write_text("", encoding="utf-8")
        (root / "knowledge" / "indexes" / "default" / "metadata.json").write_text(
            "{}",
            encoding="utf-8",
        )

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


@pytest.fixture(scope="module")  # type: ignore[misc]
def tmp_knowledge_workspace() -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="agentpm-sdk-py-knowledge-")).resolve()
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_knowledge_loads_context_mode_metadata(tmp_knowledge_workspace: Path) -> None:
    knowledge_dir = tmp_knowledge_workspace / ".agentpm" / "knowledge"
    _write_installed_knowledge(knowledge_dir, "@zack/support-playbook@0.1.0", "context")

    loaded = load_knowledge(
        "@zack/support-playbook@0.1.0",
        knowledge_dir_override=str(knowledge_dir),
    )

    assert loaded["kind"] == "knowledge"
    assert loaded["knowledge"]["mode"] == "context"
    assert loaded["documentPaths"] == [
        str(
            knowledge_dir
            / "@zack"
            / "support-playbook"
            / "0.1.0"
            / "knowledge"
            / "docs"
            / "context.md"
        )
    ]
    assert loaded["chunksPath"] is None
    assert loaded["indexPaths"] == []


def test_load_knowledge_loads_vector_mode_metadata(tmp_knowledge_workspace: Path) -> None:
    knowledge_dir = tmp_knowledge_workspace / ".agentpm" / "knowledge"
    _write_installed_knowledge(knowledge_dir, "@zack/python-docs@0.1.0", "vector")

    loaded = load_knowledge(
        "@zack/python-docs@0.1.0",
        knowledge_dir_override=str(knowledge_dir),
    )

    assert loaded["kind"] == "knowledge"
    assert loaded["knowledge"]["mode"] == "vector"
    assert loaded["chunksPath"] == str(
        knowledge_dir / "@zack" / "python-docs" / "0.1.0" / "knowledge" / "chunks.jsonl"
    )
    assert loaded["sourcesPath"] == str(
        knowledge_dir / "@zack" / "python-docs" / "0.1.0" / "knowledge" / "sources.jsonl"
    )
    assert loaded["vectorsPath"] == str(
        knowledge_dir
        / "@zack"
        / "python-docs"
        / "0.1.0"
        / "knowledge"
        / "embeddings"
        / "default.f32"
    )
    assert loaded["indexPaths"] == [
        str(knowledge_dir / "@zack" / "python-docs" / "0.1.0" / "knowledge" / "indexes" / "default")
    ]


def test_load_knowledge_fails_when_package_is_not_installed(
    tmp_knowledge_workspace: Path,
) -> None:
    knowledge_dir = tmp_knowledge_workspace / ".agentpm" / "knowledge"

    with pytest.raises(FileNotFoundError, match=r"not found in \.agentpm/knowledge"):
        load_knowledge(
            "@zack/missing-docs@0.1.0",
            knowledge_dir_override=str(knowledge_dir),
        )


def test_load_knowledge_fails_when_manifest_is_wrong_kind(tmp_knowledge_workspace: Path) -> None:
    knowledge_dir = tmp_knowledge_workspace / ".agentpm" / "knowledge"
    package_name, version = _split_spec("@zack/not-knowledge@0.1.0")
    root = knowledge_dir / package_name / version
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "tool",
                "name": "not-knowledge",
                "version": version,
                "entrypoint": {"command": "python", "args": ["tool.py"]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a knowledge manifest"):
        load_knowledge(
            "@zack/not-knowledge@0.1.0",
            knowledge_dir_override=str(knowledge_dir),
        )


def test_load_knowledge_fails_when_manifest_is_missing_mode(
    tmp_knowledge_workspace: Path,
) -> None:
    knowledge_dir = tmp_knowledge_workspace / ".agentpm" / "knowledge"
    package_name, version = _split_spec("@zack/missing-mode@0.1.0")
    root = knowledge_dir / package_name / version
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "knowledge",
                "name": "missing-mode",
                "version": version,
                "knowledge": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"missing knowledge\.mode"):
        load_knowledge(
            "@zack/missing-mode@0.1.0",
            knowledge_dir_override=str(knowledge_dir),
        )
