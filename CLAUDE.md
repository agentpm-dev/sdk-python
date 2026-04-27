# `agentpm-sdk-python` Repo Review Guide

This repo is the Python runtime integration layer.

## Review Focus
- import/runtime correctness
- subprocess behavior
- env passing behavior
- packaging compatibility
- public SDK contract stability

## Blocking Issues
- broken load semantics
- import/path regressions
- env handling regressions
- subprocess contract regressions
- unintended public API drift

## Preferred Emphasis
- correctness
- compatibility
- runtime determinism
