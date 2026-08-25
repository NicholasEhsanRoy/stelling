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
used: `biased` (blackjax#925's failure word — silent statistical bias from
an ignored status flag; belongs to the next registration).

## Anti-rationalisations

- These reads serve the categories artifact. No value claim, no bands, no
  sequencing argument may be built on them directly.
- A dead category looking alive in these results becomes a **new
  registration**, never a revival in place.
- Counts are incidents, not customers.

---

# Reading (2026-07-17/18 — after the registration commit)

## Normalization (live finding #3)

Total issues: diffrax **489**, numpyro **872**, blackjax **372**, jax-md
**180**, jax-cfd **32**. Probe-1 hit rates per tracker: diffrax 2.2%,
jax-md 1.1%, blackjax 0.8%, numpyro 0.46%, jax-cfd 0%. At diffrax's rate,
32 issues predict ~0.7 hits; zero is unremarkable (Poisson P(0) ≈ 0.5).
**jax-cfd's tracker silence is base-rate-dominated — a fact about a quiet
tracker, not evidence about stencil code.** The census's style finding
(zero wedge primitives; pad+static-slice halos) stands separately and is
the only instrument that actually saw jax-cfd's code.

## Term counts (pre-filter, nonzero cells)

diffrax: stuck 15, silently 10, different-results 8, hang 4, explosion 4,
wrong-gradient 4, exploded 2, wrong-results 2, overflow 1, key-reuse 1,
nondet 1+1. numpyro: stuck 16, different-results 20, same-key 13, hang 4,
overflow 4, wrong-results 6, non-det 3, silently 2, wrong-gradient 1,
key-reuse 1. jax-md: **overflow 10 (5.6% of its whole tracker — highest
normalized rate of any cell)**, stuck 3, explosion 2, silently 2,
wrong-results 2, hang 1, key-reuse 1, same-key 1, different-results 1.
blackjax: same-key 5, different-results 4, stuck 3, silently 3, key-reuse
2, wrong-results 2, non-det 1. jax-cfd: stuck 2, different-results 1.

## Load-bearing classifications

- **jax-md `overflow` ×10 is not integer overflow.** It is the
  neighbor-list **capacity-flag protocol**: `did_buffer_overflow` is a
  hand-rolled detective defence whose manual check-and-reallocate protocol
  generates 5.6% of the tracker (usage burden), and **jmd#141 is a defect
  in the detector itself** (update misbehaves when there is *no*
  overflow). Zero incidents of silent int32 *arithmetic* overflow were
  found in any tracker — **and that zero is uninterpretable for this
  class, by this file's own argument three bullets up.** A tracker counts
  what somebody noticed; the probed behaviour is a silent wrap, so a
  filing requires the user to have discovered a wrong number by some
  other route. *This read "the int-overflow candidate dies on field
  evidence despite the probed silent wrap" — the concessive clause names
  the confound and then proceeds through it. Corrected 2026-08-24: the
  candidate is UNMEASURED, and `design/jax-verification-categories.md`'s
  D9 row is corrected with it.*
- **New incidents, bucketed and attributed:** dfx#707 (open c5,
  compile-time hang in VirtualBrownianTree — termination, compile-time
  variant); dfx#185 (controller thrash on a switched system — Perf bucket,
  termination-attributed); dfx#596 (open, BacksolveAdjoint gradient
  failure — adjoint instability family); **dfx#692 (vmap over diffeqsolve
  changes the number of accepted steps — a field incident of transform
  non-equivalence: batching alters adaptive control flow)**; **dfx#729
  (open c15, post-event state gradients "now always wrong" — silent wrong
  gradients at the events×adjoint composition, with cost)**; dfx#363
  (piecewise ⇒ zero gradient — the where/select surface); jmd#343
  (Brownian dt=1e-8 silently below f32 resolution — the float-boundary
  class, silent); **bjx#925 (maintainer-filed: inner L-BFGS
  non-convergence silently ignored → biased Laplace marginals — an
  *ignored status flag*)**; npy#1427 (pmap changes NUTS results) and
  npy#1120 (same PRNGKey, different results across machines) — platform/
  transform variance.
- Every other fetched item triaged to FR/usage/perf/point/duplicate with
  **zero additional long-horizon candidates**.
- **Author checks:** bjx#925/#927/#871 = `junpenglao` (core maintainer);
  jmd#141 = `aniquetahir`. All third-party; all countable. bjx#871 is the
  maintainer running an AI review over their own codebase — recorded as an
  audience observation, not as failure evidence.

## What this feeds

The defective/ignored-detector family now has **three independent wild
instances** (bjx#969 dead NaN detector, jmd#141 defective overflow
detector, bjx#925 ignored convergence flag) plus one census instance (the
guard experiment's target sites). Transform-equivalence has its first two
field incidents (dfx#692, npy#1427). Termination gains dfx#707/#185.
Float-boundary gains jmd#343. All consumed by
`design/jax-verification-categories.md`.
