# CI readiness — the construction-path census, then the CI-mode field test

**Status:** REGISTRATION, 2026-07-21, committed **before the census
reads or any field-test verdict exists**. The framing that governs the
pass: **CI trust is a reliability-and-evidence property, not a
capability property.** Three readiness claims have been answered by
argument and never by measurement — the project's own meta-rule, turned
on its own readiness:

1. *"Every reachable construction path produces safe IR"* — asserted,
   load-bearing for the I1 residual, never enumerated.
2. *"Stelling is already useful"* — every finding so far
   (F1, R1, magnetics SPD/mass) was **hand-guided**: stelling pointed at
   an obligation already suspected interesting. CI value is the other
   mode — catching what nobody was already worried about — and evidence
   in that mode is **zero** because the tool has never run in it.
3. *"Affine/LA are deepening, not broadly useful"* — argued, not
   measured (Part C corrects the record).

## PART A — the construction-path census (gates Part B)

**Claim under test:** every way an `ir` object comes into existence
routes through a gate guaranteeing well-formedness.

**Method, fixed now:** enumerate by AST/grep over the whole repo — every
instantiation of every `ir` dataclass (`Jaxpr`, `ClosedJaxpr`,
`JaxprEqn`, `Var`, `Literal`, `Array`, `Aval`, param types), every
factory/classmethod, every `dataclasses.replace`. Classify each site:
trace-door / from_dict-door / internal-transform (does it preserve
well-formedness given valid input?) / test-only (reachable from public
API or not?) / direct construction. **The census-of-the-census:** state
what makes the list complete — and confront the structural fact that
Python dataclasses are freely constructible, so any funnel claim that
is not enforced *in the types themselves* is convention.

**Bands (fixed):**

| finding | reading | action |
|---|---|---|
| every reachable path routes through the gate; the funnel invariant holds | I1 genuinely out-of-contract; "safe on all reachable paths" **verified** | proceed to Part B |
| an ungated path exists, reachable from the public API | a real soundness gap (the z3-defect shape: latent, on an unexercised path) | **STOP** — structuralize the gate to cover it (4-B witnessed constructions), then re-gate |
| completeness cannot be established (construction does not funnel) | "safe on all paths" is unprovable as architected — a design finding that reranks the CI path | **STOP, surface** |

Per the work order: a row-2 finding is fixed in-pass (structuralize,
witnessed constructions) and the gate re-adjudicated; row 3 surfaces.

## PART B — the CI-mode field test (only if Part A lands row 1)

First run of stelling in the CI workflow: preconditions posed over a
codebase **unguided** — no pre-selected interesting obligation. The
unguided protocol, fixed now so envelope choice cannot steer: for the
target module(s), enumerate mechanically every (i) division/reciprocal
(denominator nonzero over the envelope), (ii) `sqrt`/`log`/`pow`
argument (domain), (iii) config scalar with a default (nonzero /
admissible-range), (iv) coefficient field feeding a solve (positivity),
and pose **all of them** via the precondition templates over disclosed
envelopes (generic sign-unknown boxes plus any documented supported
range, both reported).

- **B1 — MADDENING** (useful-to-Nick): the wavelet solver core, posed by
  the main agent under the mechanical protocol.
- **B2 — an external repo** (useful-to-strangers, the stronger test):
  **lineax 0.1.1** — real solver code, not written by Nick, censused for
  equation counts once but never precondition-analyzed; its documented
  solver tags (callers *assert* positive-definiteness and similar) are
  exactly the assumed-precondition class. Blind transcriber, repo
  source only, no expectation of what is there.

**Adjudication duties, fixed:** every REFUTED gets a false-positive
adjudication (real unguarded precondition vs guaranteed-by-something-
the-tool-cannot-see), by the main agent, per finding; every UNKNOWN
joins the decline map (what a CI integration could check vs stay silent
on); every VERIFIED gets a `widen()` vacuity spot-check. **The
false-positive rate is the number that decides CI-viability and is
reported whatever it is.**

**Reading bands (fixed):** genuine finding + low false-positive rate =
the missing CI evidence and the adoption artifact; real findings + high
false-positive rate = the gap is precision and the causes are the
finding; clean pass = distinguish "preconditions genuinely sound" from
"class cannot reach them" via the decline map. **No outcome is a
failure of the pass.**

## PART C — the CI-trust gap, named (report-only)

(i) Which of the manual backstops that protected shipped soundness
(main-agent witness re-replay, both-faces adjudication, per-harness
vacuity runs) are already structural vs would need to become automatic
before CI-trusted stelling matches human-supervised stelling. (ii) The
affine/LA record correction: their non-usefulness was argued, not
measured, and the magnetics characterization's governing property
(residual-bounds-error via conditioning) is LA-shaped **with a measured
scar** — evidence *for* immediate usefulness. Correct the record; build
nothing.

## Non-goals

No new capability (structuralization of Part-A-exposed convention only).
Part A gates Part B. Unadjudicated findings are not evidence. No
affine/LA build. No publish. CALMS, qMRI, E2b out of scope.
