# The LA contract layer — build, first-contact audit, fixes

**Status:** PASS RECORD, 2026-07-21. The roadmap's item 2
(`design/roadmap.md`), begun and landed the same day; the stop-clause
("if the seam exceeds a session, stop and report; publish proceeds
without LA") was **not triggered**. Twice-evidence-backed going in: the
magnetics conditioning scar (measured failure above coefficient
contrast ≈10²) and the D-clamp adjudication, whose cause needed LA
vocabulary to state. Scope was deliberately the measured conditioning
obligation — not general solver-behavior verification.

## What the layer is

`stelling.contracts`: an assume-guarantee contract for a linear-solve
leg with exactly two faces, and deliberately **no combined status**.

- **requires — mechanized.** A conditioning precondition posed as
  ordinary obligations through the ONE existing pipeline (a
  behavior-identical `_pipeline` extraction from
  `preconditions.check()`; `check()` is now that helper with the traced
  query dropped — dataclass-equality verified by the auditor on both
  paths). Standard Verdict, standard stamp, `vacuity_mode` required,
  VERIFIED widen re-checked at the same depth, solvers never-on by
  default.
- **ensures — DECLARED, never checked.** A statement that is a theorem
  given the requires, status structurally pinned to `DECLARED`
  (sealed-type funnels at `EnsuresFace`, `Contract`, and
  `ContractVerdict` construction — no constructible path through the
  layer's types yields anything else; interpreter-level
  `object.__setattr__` is outside the threat model, consistent with
  every frozen dataclass in the codebase). A contract with no exact
  derivation declares none and says why on the verdict.

Two templates, and only these two:

- **`conditioning_2x2`** — symmetric 2x2 over declared entry ranges;
  requires = closed PSD (principal minors ≥ 0) ∧
  `tr² ≤ det·(κ + 1/κ + 2)` (the probe's exact reduction of
  `cond₂ ≤ κ`). The posed set is proven equal to
  `{M SPD, cond₂ ≤ κ} ∪ {M = 0}` — the closure, boundary exactly the
  zero matrix, which the ensures excludes by name. `κ < 1` refused at
  authoring (`f(κ) = f(1/κ)` would silently pose the wrong bound).
  Ensures: `‖M⁻¹r‖₂ ≤ (κ/‖M‖₂)·‖r‖₂`, κ-derived (probe Part 4),
  conditional, DECLARED.
- **`coefficient_contrast`** — a caller-transform field over a declared
  parameter envelope; requires = field ≥ 0 ∧ `max ≤ C·min`
  (division-free). Extremum posed as a pairwise fold of already-audited
  primitives (slice + binary max/min — exact in ℝ in both directions)
  rather than adding unaudited `reduce_min/max` rows in the same build
  the layer's own auditor must attack; core reduction rows remain their
  own future censused build. Cost rides the one existing budget: 4n
  terms for μ = 1+χ, n ≤ 128 admits, above declines loudly.
  **Ensures: none declared** — contrast ≤ C does not bound κ of a
  discretized operator without mesh constants, and the verdict says so.

**Emptiness asymmetry (the F7 mandate, held):** requires-VERIFIED is
universally quantified and certifies nothing about nonemptiness of the
envelope or its requires-satisfying subset; the ensures reads "for
every point … satisfying the requires", discloses vacuous truth, and
routes nonemptiness to `stelling.exactness`. The audit's
existential-leak scan across every render/stamp/note produced by its
battery: zero hits beyond the disclosure sentences themselves.

## Known answers (probe = corpus/supply/la_contract_probe.py; all
reproduced through the public API, statuses re-verified after fixes)

| case | verdict |
|---|---|
| T1, a,c∈[1,2], b∈±0.5, κ=8, no solver | UNKNOWN — straddle quoted (dependency-shaped: intervals cannot close what Z3 proves; probe Part 3) |
| same, with solver budget | VERIFIED via QF_NRA (cvc5 primary + z3), envelope load-bearing under widen |
| T1, b∈±1.4, with solver | REFUTED, two witnesses, dyadic box membership + violation replayed exactly |
| T1, probe Part 1 mesh point (cond ≈ 2.5e29) | REFUTED by intervals alone, no solver |
| T2, n=64, χ∈[1e-6,1e2], C=200 | VERIFIED, interval path alone (independent boxes: min/max attain endpoints) |
| T2, χ∈[1e-6,1e5], with solver | REFUTED, witness names all 64 elements (high/low), replay-confirmed |

The pair of templates exercises both pipeline paths by construction:
T1's VERIFIED is solver-only (interval straddle), T2's VERIFIED is
interval-only (no dependency), and both REFUTE through their own path.

## First-contact audit (distinct fresh context, report-only)

Mandate led with the emptiness asymmetry; report format required the
"attacks that did not land" list — a clean audit that cannot show its
attack list is not a clean audit. Result: **ten findings, zero
UNSOUND** — no route to a false VERIFIED/REFUTED on the requires face
(survived: overflow at the comparison with both sides at [maxfloat,
inf]; κ=1 and closure boundaries; nine witnesses re-derived in exact
rationals, endpoints landing on the declared floats' dyadic values;
budget edge measured 512@n=128 admit / 516@n=129 decline;
nested-assert misattribution unconstructible; empty and
requires-empty envelopes honest both ways).

Adjudicated and fixed, verified by re-running the auditor's own
reproducers (each measured misbehavior flipped; statuses everywhere
unchanged):

1. **F1 MISLEADING** — "no code path can produce a non-DECLARED
   ensures" was false via subclass / duck-type / direct
   `ContractVerdict` (L18's class: duty-stated, not structural). Fixed:
   sealed-type funnels at all three constructions; docstring restated
   to the constructible-path claim.
2. **F2 MISLEADING, pre-existing in `check()`** — the false
   "widened/not-load-bearing" vacuity line on all-point envelopes under
   inputs-only (measured self-contradiction with mode=all on the
   identical query). Fixed: identical-widened-query detection, re-run
   skipped, honest inert line. **Banked as L20.**
3. **F3 MISLEADING, pre-existing** — widen re-check solver invocations
   relied on by the vacuity line but unstamped (10 spawned / 2
   recorded). Fixed: appended with a `vacuity widen re-check:` tag;
   spawn/stamp parity now measured 10/10 (and 4/4 on the inert path,
   which no longer re-runs at all).
4. **F4–F6 SHARP-EDGE (new code)** — empty-transform silent UNKNOWN →
   loud refusal; constant-transform crash → `asarray` coercion with
   vacuity disclosure (the `field_positive` posture); newline injection
   through hand-built `EnsuresFace` strings could forge column-0
   verdict lines → single-physical-line refusal (the model-echo
   screening posture).
5. **F7 SHARP-EDGE, pre-existing** — invalid `vacuity_mode` silent
   until the first VERIFIED (a CI check green-UNKNOWN for life, then
   exploding) → eager validation at `_pipeline` entry, single source of
   truth with `widen()`.
6. **F8–F10 NOTE** — authoring-time refusals for malformed
   shapes/ranges (the funnel principle; the guard's own "at authoring
   time" message now true); the straddle-note hint promises the offer,
   not the escalation ("offers exactly this obligation to solver
   escalation"); `solver_timeout_ms` exactly-int; whitespace-only
   ensures texts refused.

Three of ten were pre-existing `check()` defects surfaced by the new
seam's first contact — L14's tally grows: every first unguided contact
so far has found latent defects in the thing contacted, including this
one, where the thing contacted was partly stelling itself.

## Suite state at commit

venv-jax **939 passed** (898 baseline + 25 build + 16 fix regressions);
venv-nojax **746 passed, 14 skipped** (the new module import-safe
jax-free; authoring works, checking refuses loudly). Every audit
reproducer re-run by the orchestrator post-fix: forgery routes BLOCKED,
inert line present, spawn parity exact, eager refusals firing, known
answers unchanged.

## What this buys (and does not)

The consequence-establishing capability for solver-behavior findings
now exists in first-slice form: a violated conditioning requires can be
*stated against a declared guarantee* rather than as a bare hazard —
the finding-conjunction's `¬harmless` conjunct gets its vocabulary for
solve legs. It does NOT yet: attach to a real solver call site
(adoption work), verify any ensures (DECLARED is the design, not a
gap), or generalize past the two templates. Next per the roadmap:
reachability (item 1) — unchanged by this pass.
