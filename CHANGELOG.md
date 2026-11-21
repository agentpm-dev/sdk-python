# Changelog

## Unreleased

- Expose exact Runner-advertised APDS identities through the existing Harness
  bridge: `supportedStandards()` (Node) / `supported_standards()` (Python).
  Preserve initialization and preflight response envelopes, distinguish absent
  advertisements from empty support lists, and retain protocol version 1.
- Verify compatibility/readiness and actual multi-version Component execution
  against the real CLI using shared local fixtures.

## 0.1.17 — 2026-11-09

- Read supported v2/v3/v4 locks and installed compositions without rewriting them.
- Delegate every Tool call to AgentPM CLI >=0.1.43; keep public loading and
  callable/result APIs, optional adapters and package exports. A compatible CLI
  is now required, including for legacy Python and Node Tools.
- Preserve exact Tool selection, caller env, timeout and stderr diagnostics;
  report runtime repair failures without an SDK-owned fallback.
