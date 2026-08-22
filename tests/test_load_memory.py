from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentpm import load_memory, load_memory_contract


def _split_spec(spec: str) -> tuple[str, str]:
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f"Bad spec: {spec}")
    return spec[:at], spec[at + 1 :]


def _write_installed_memory(base_dir: Path, spec: str) -> Path:
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
                    "operations": {
                        "refresh_profile": {
                            "type": "transform",
                            "inputs": [{"space": "profile", "record_type": "user_preference"}],
                            "output": {
                                "space": "profile",
                                "record_type": "user_preference",
                            },
                            "output_mode": "replace_input",
                            "source_handling": "retain",
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
                "built_at": "2026-07-20T00:00:00Z",
                "agentpm_version": "0.1.0",
                "manifest_path": "agent.json",
                "source_manifest_hash": "sha256:manifest",
                "source_schemas": [
                    {"path": "schemas/user-preference.schema.json", "sha256": "sha256:schema"}
                ],
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
        json.dumps({"type": "object", "properties": {"id": {"type": "string"}}}, indent=2),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")  # type: ignore[misc]
def tmp_memory_workspace() -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="agentpm-sdk-py-memory-")).resolve()
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_memory_loads_metadata_build_index_and_contract_refs(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    _write_installed_memory(memory_dir, "@zack/profile-memory@0.1.0")

    loaded = load_memory(
        "@zack/profile-memory@0.1.0",
        memory_dir_override=str(memory_dir),
    )

    assert loaded["kind"] == "memory"
    assert loaded["memory"]["spaces"]["profile"]["model"] == "document"
    assert loaded["memory"]["operations"]["refresh_profile"]["output_mode"] == "replace_input"
    assert loaded["build"]["type"] == "agentpm-memory-contracts"
    assert loaded["contractIndex"]["type"] == "agentpm-memory-contract-index"
    assert loaded["sourceSchemaPaths"] == [
        str(
            memory_dir
            / "@zack"
            / "profile-memory"
            / "0.1.0"
            / "schemas"
            / "user-preference.schema.json"
        )
    ]
    assert loaded["contracts"] == [
        {
            "space": "profile",
            "recordType": "user_preference",
            "schemaVersion": "1.0.0",
            "model": "document",
            "sourceSchemaPath": str(
                memory_dir
                / "@zack"
                / "profile-memory"
                / "0.1.0"
                / "schemas"
                / "user-preference.schema.json"
            ),
            "path": str(
                memory_dir
                / "@zack"
                / "profile-memory"
                / "0.1.0"
                / "memory"
                / "contracts"
                / "profile.user_preference.schema.json"
            ),
            "sha256": "sha256:contract",
        }
    ]


def test_load_memory_contract_loads_one_indexed_contract(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    _write_installed_memory(memory_dir, "@zack/contract-memory@0.1.0")

    loaded = load_memory(
        "@zack/contract-memory@0.1.0",
        memory_dir_override=str(memory_dir),
    )
    contract = load_memory_contract(loaded, space="profile", record_type="user_preference")
    assert contract["type"] == "object"
    assert "properties" in contract


def test_load_memory_fails_when_package_is_missing(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"

    with pytest.raises(FileNotFoundError, match=r"not found in \.agentpm/memory"):
        load_memory(
            "@zack/missing-memory@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_clearly_for_invalid_version_selector(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"

    with pytest.raises(ValueError, match=r'Invalid version/range "definitely-not-semver"'):
        load_memory(
            "@zack/profile-memory@definitely-not-semver",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_manifest_is_wrong_kind(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/not-memory@0.1.0")
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "tool",
                "name": "not-memory",
                "version": "0.1.0",
                "entrypoint": {"command": "python", "args": ["tool.py"]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a memory manifest"):
        load_memory(
            "@zack/not-memory@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_manifest_is_missing_memory_object(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/missing-memory@0.1.0")
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "memory",
                "name": "missing-memory",
                "version": "0.1.0",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing memory object"):
        load_memory(
            "@zack/missing-memory@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_for_unsafe_traversal_path(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/unsafe-memory@0.1.0")
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
                        "path": "../outside.schema.json",
                        "sha256": "sha256:contract",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe package-relative path"):
        load_memory(
            "@zack/unsafe-memory@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_build_json_is_missing(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/missing-build@0.1.0")
    (root / "memory" / "build.json").unlink()

    with pytest.raises(FileNotFoundError, match=r"memory/build\.json .*missing"):
        load_memory(
            "@zack/missing-build@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_index_json_is_missing(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/missing-index@0.1.0")
    (root / "memory" / "contracts" / "index.json").unlink()

    with pytest.raises(FileNotFoundError, match=r"memory/contracts/index\.json .*missing"):
        load_memory(
            "@zack/missing-index@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_build_json_is_malformed(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/malformed-build@0.1.0")
    (root / "memory" / "build.json").write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"memory/build\.json is not valid JSON"):
        load_memory(
            "@zack/malformed-build@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_index_json_is_malformed(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/malformed-index@0.1.0")
    (root / "memory" / "contracts" / "index.json").write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"memory/contracts/index\.json is not valid JSON"):
        load_memory(
            "@zack/malformed-index@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_build_json_is_missing_required_hashes(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/missing-hashes@0.1.0")
    (root / "memory" / "build.json").write_text(
        json.dumps(
            {
                "type": "agentpm-memory-contracts",
                "format_version": 1,
                "manifest_path": "agent.json",
                "source_manifest_hash": "sha256:manifest",
                "source_schemas_hash": "sha256:schemas",
                "contract_count": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match=r"missing source_contract_inputs_hash|missing contracts_hash"
    ):
        load_memory(
            "@zack/missing-hashes@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_build_json_has_unsupported_type(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/unsupported-build-type@0.1.0")
    (root / "memory" / "build.json").write_text(
        json.dumps(
            {
                "type": "other-memory-build",
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

    with pytest.raises(ValueError, match=r"memory/build\.json has unsupported type"):
        load_memory(
            "@zack/unsupported-build-type@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_index_json_has_unsupported_format(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/unsupported-index-format@0.1.0")
    (root / "memory" / "contracts" / "index.json").write_text(
        json.dumps(
            {
                "type": "agentpm-memory-contract-index",
                "format_version": 2,
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

    with pytest.raises(
        ValueError, match=r"memory/contracts/index\.json has unsupported format_version"
    ):
        load_memory(
            "@zack/unsupported-index-format@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_contract_count_mismatches_index(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/count-mismatch@0.1.0")
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
                "contract_count": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"contract_count does not match"):
        load_memory(
            "@zack/count-mismatch@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_for_duplicate_contract_identity(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/duplicate-identity@0.1.0")
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
                "contract_count": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "memory" / "contracts" / "profile.user_preference.copy.schema.json").write_text(
        json.dumps({"type": "object"}, indent=2),
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
                        "sha256": "sha256:contract-1",
                    },
                    {
                        "space": "profile",
                        "record_type": "user_preference",
                        "schema_version": "1.0.0",
                        "model": "document",
                        "source_schema": "schemas/user-preference.schema.json",
                        "path": "memory/contracts/profile.user_preference.copy.schema.json",
                        "sha256": "sha256:contract-2",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate contract entry"):
        load_memory(
            "@zack/duplicate-identity@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_for_duplicate_contract_path(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/duplicate-path@0.1.0")
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
                "contract_count": 2,
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
                        "sha256": "sha256:contract-1",
                    },
                    {
                        "space": "archive",
                        "record_type": "user_preference",
                        "schema_version": "1.0.0",
                        "model": "document",
                        "source_schema": "schemas/user-preference.schema.json",
                        "path": "memory/contracts/profile.user_preference.schema.json",
                        "sha256": "sha256:contract-2",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate contract path"):
        load_memory(
            "@zack/duplicate-path@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_when_indexed_contract_file_is_missing(
    tmp_memory_workspace: Path,
) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/missing-contract-file@0.1.0")
    (root / "memory" / "contracts" / "profile.user_preference.schema.json").unlink()

    with pytest.raises(
        FileNotFoundError, match=r"contract path|memory/contracts/index\.json contract path"
    ):
        load_memory(
            "@zack/missing-contract-file@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_fails_for_symlink_escape(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    root = _write_installed_memory(memory_dir, "@zack/symlink-memory@0.1.0")
    outside_dir = tmp_memory_workspace / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    (outside_dir / "outside.schema.json").write_text("{}", encoding="utf-8")
    (root / "memory" / "contracts" / "profile.user_preference.schema.json").unlink()
    (root / "memory" / "contracts" / "profile.user_preference.schema.json").symlink_to(
        outside_dir / "outside.schema.json"
    )

    with pytest.raises(ValueError, match="outside the installed memory package root"):
        load_memory(
            "@zack/symlink-memory@0.1.0",
            memory_dir_override=str(memory_dir),
        )


def test_load_memory_contract_fails_for_unknown_identity(tmp_memory_workspace: Path) -> None:
    memory_dir = tmp_memory_workspace / ".agentpm" / "memory"
    _write_installed_memory(memory_dir, "@zack/missing-contract@0.1.0")
    loaded = load_memory(
        "@zack/missing-contract@0.1.0",
        memory_dir_override=str(memory_dir),
    )

    with pytest.raises(ValueError, match="was not found"):
        load_memory_contract(loaded, space="missing", record_type="user_preference")
