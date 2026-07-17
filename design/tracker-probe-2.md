# Tracker probe 2 — category-evidence reads, registered before reading

**Status:** REGISTRATION, 2026-07-17 (overnight categories pass). Committed
before any of the searches below run. Purpose: **existence/failure evidence
for `design/jax-verification-categories.md`** — not a value-model test. No
bands: the outputs are per-term counts, bucket classifications, and
category attributions, claimed as evidence only. A hit that touches a
previously killed hypothesis is recorded as evidence and is **not** a
resurrection — dead categories stay dead unless a *new* registration
re-opens them explicitly.

## Corpus

The same five trackers as `design/tracker-probe.md` (slugs verified there),
issues open+closed, discussions where enabled (blackjax only). **Also
recorded: per-repo total issue counts**, to normalize hit rates — needed to
answer whether jax-cfd's silence is a fact about stencil code or about a
quiet tracker (live finding #3).

## Terms — fixed, with operationalization and target category

| term | searched as | evidence for |
|---|---|---|
| `stuck` | `stuck` | termination/progress (banked from probe 1) |
| `hang` | `hang` | termination/progress (banked) |
| `explosion` / `exploded` | `explosion`, `exploded` | long-horizon/instability (banked) |
| `wrong results` | `"wrong results"` | silent-wrongness surface (banked) |
| `overflow` | `overflow` | integer-overflow candidate |
| `key reuse` | `"key reuse"` | PRNG-misuse candidate |
| `same key` | `"same key"` | PRNG-misuse candidate |
| `silently` | `silently` | silent-wrongness surface |
| `different results` | `"different results"` | equivalence/determinism candidates |
| `wrong gradient` | `"wrong gradient"` | gradient-correctness candidates |
| `nondeterministic` | `nondeterministic` | determinism candidate |
| `non-deterministic` | `"non-deterministic"` | determinism candidate |

Retrieval: per-(repo, term) total counts recorded pre-filter; up to 30
best-match results fetched per pair; classification into the probe-1
taxonomy (Long-horizon / Point / User / Perf / FR / Unclear) plus a
category-attribution column. Mid-read term ideas are recorded and not
used: *(none yet)*

## Anti-rationalisations

- These reads serve the categories artifact. No value claim, no bands, no
  sequencing argument may be built on them directly.
- A dead category looking alive in these results becomes a **new
  registration**, never a revival in place.
- Counts are incidents, not customers.
