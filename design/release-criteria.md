# Release criteria — two releases, two criteria, not conflated

**Status:** design note, 2026-07-18. Fixed before either release happens.

## (a) The technical release — v0.1.0, tagged locally, unpushed

**Criterion: the README doesn't lie.** Nothing else. It says what the
tool does and the *measured* list of what it doesn't (derive, control
flow, discrete steps, IEEE semantics, incident discharge 0/3). Per the
byproduct policy this release is a **changelog, not a proposition** — it
claims existence, not usefulness.

Why do it at all: the REUSE / DCO / Trusted-Publisher / PEP-740 machinery
has been built and never exercised end-to-end — a release is the only
test of it, and finding a broken attestation workflow is cheapest at
0.1.0. And the name is free on PyPI today.

Remaining steps are the maintainer's, unchanged: configure the PyPI
pending Trusted Publisher + `pypi` environment, push, create the GitHub
release.

## (b) The "useful" release

**Criterion: E2a lands its supported band — ≥ 4 of 13 mechanized, across
≥ 2 libraries** (`design/value-model-v2.md`), under the tightened
criteria (nonvacuity checked; relation to the registered property named).
No second criterion is invented here: *shown to be initially useful* is a
value claim, the model owns value claims, and its band was registered
before any case ran. This is the release that gets a proposition.

Anything between (a) and (b) ships as changelog entries, never as
claims.
