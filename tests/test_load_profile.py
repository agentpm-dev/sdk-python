from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentpm import load_profile


def _split_spec(spec: str) -> tuple[str, str]:
    at = spec.rfind("@")
    if at <= 0 or at == len(spec) - 1:
        raise ValueError(f"Bad spec: {spec}")
    return spec[:at], spec[at + 1 :]


def _write_installed_profile(base_dir: Path, spec: str, profile: dict[str, object]) -> Path:
    package_name, version = _split_spec(spec)
    root = base_dir / package_name / version
    manifest_name = package_name.split("/", 1)[1]
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "profile",
                "name": manifest_name,
                "version": version,
                "description": "Installed profile fixture",
                "profile": profile,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")  # type: ignore[misc]
def tmp_profile_workspace() -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="agentpm-sdk-py-profile-")).resolve()
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_profile_loads_minimal_profile_metadata(tmp_profile_workspace: Path) -> None:
    profile_dir = tmp_profile_workspace / ".agentpm" / "profiles"
    _write_installed_profile(
        profile_dir,
        "@zack/support-style@0.1.0",
        {
            "identity": {"role": "Support agent"},
            "objectives": ["Help users move forward"],
            "communication": {
                "tone": ["calm"],
                "verbosity": "balanced",
            },
        },
    )

    loaded = load_profile(
        "@zack/support-style@0.1.0",
        profile_dir_override=str(profile_dir),
    )

    assert loaded["kind"] == "profile"
    assert loaded["profile"]["identity"]["role"] == "Support agent"
    assert loaded["profile"]["communication"]["verbosity"] == "balanced"


def test_load_profile_loads_full_profile_metadata(tmp_profile_workspace: Path) -> None:
    profile_dir = tmp_profile_workspace / ".agentpm" / "profiles"
    _write_installed_profile(
        profile_dir,
        "@zack/escalation-style@0.2.0",
        {
            "identity": {
                "role": "Escalation reviewer",
                "description": "Coordinate clear next steps for escalations.",
                "expertise": ["Tier-two support", "Customer recovery"],
            },
            "objectives": ["Clarify the escalation path", "Capture accountable next steps"],
            "principles": ["State ownership clearly"],
            "audience": {
                "description": "Customers awaiting escalation outcomes",
                "adaptation": ["Avoid jargon unless the user already used it"],
            },
            "communication": {
                "tone": ["direct", "calm"],
                "verbosity": "concise",
                "guidelines": ["Lead with the decision"],
                "formatting": ["Use short bullets for action items"],
                "vocabulary": {
                    "prefer": ["next step"],
                    "avoid": ["circle back"],
                },
            },
            "boundaries": ["Do not promise timelines you cannot confirm"],
            "constraints": [
                {
                    "id": "confirm-accountability",
                    "strength": "required",
                    "instruction": "Always identify the team or person who owns the next action.",
                }
            ],
            "compatibility": {
                "minimum_context_tokens": 4000,
                "requires": {"tool_use": True},
                "recommends": {"structured_output": True},
            },
        },
    )

    loaded = load_profile(
        "@zack/escalation-style@0.2.0",
        profile_dir_override=str(profile_dir),
    )

    assert loaded["profile"]["identity"]["expertise"] == ["Tier-two support", "Customer recovery"]
    assert loaded["profile"]["constraints"][0]["id"] == "confirm-accountability"
    assert loaded["profile"]["compatibility"]["requires"]["tool_use"] is True


def test_load_profile_fails_when_package_is_missing(tmp_profile_workspace: Path) -> None:
    profile_dir = tmp_profile_workspace / ".agentpm" / "profiles"

    with pytest.raises(FileNotFoundError, match=r"not found in \.agentpm/profiles"):
        load_profile(
            "@zack/missing-style@0.1.0",
            profile_dir_override=str(profile_dir),
        )


def test_load_profile_fails_when_manifest_is_wrong_kind(tmp_profile_workspace: Path) -> None:
    profile_dir = tmp_profile_workspace / ".agentpm" / "profiles"
    root = _write_installed_profile(
        profile_dir,
        "@zack/not-profile@0.1.0",
        {
            "identity": {"role": "Support agent"},
            "objectives": ["Help users move forward"],
            "communication": {
                "tone": ["calm"],
                "verbosity": "balanced",
            },
        },
    )
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "tool",
                "name": "not-profile",
                "version": "0.1.0",
                "entrypoint": {"command": "python", "args": ["tool.py"]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a profile manifest"):
        load_profile(
            "@zack/not-profile@0.1.0",
            profile_dir_override=str(profile_dir),
        )


def test_load_profile_fails_when_manifest_is_missing_profile_object(
    tmp_profile_workspace: Path,
) -> None:
    profile_dir = tmp_profile_workspace / ".agentpm" / "profiles"
    root = _write_installed_profile(
        profile_dir,
        "@zack/missing-profile-object@0.1.0",
        {
            "identity": {"role": "Support agent"},
            "objectives": ["Help users move forward"],
            "communication": {
                "tone": ["calm"],
                "verbosity": "balanced",
            },
        },
    )
    (root / "agent.json").write_text(
        json.dumps(
            {
                "kind": "profile",
                "name": "missing-profile-object",
                "version": "0.1.0",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing profile object"):
        load_profile(
            "@zack/missing-profile-object@0.1.0",
            profile_dir_override=str(profile_dir),
        )
