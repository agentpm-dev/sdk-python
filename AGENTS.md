# `agentpm-sdk-python` Repo Guide

This repo is the Python SDK for AgentPM.

## Purpose
- tool discovery/loading
- subprocess execution
- Python-side runtime integration
- env handling and package ergonomics

## Local Rules
- Open source
- PyPI distribution matters

## Builder Guidance
- Be conservative with SDK contracts.
- Keep import/runtime behavior stable.
- Preserve subprocess/env semantics carefully.
- Avoid changes that make load behavior less predictable.

## Verification
- Verify load/runtime behavior when changing discovery, env handling, subprocess logic, or packaging structure.

## Never Do This
- Don’t change SDK contracts casually.
- Don’t regress import, load, or env-passing behavior.
