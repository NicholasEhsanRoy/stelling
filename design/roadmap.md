# Roadmap

**Status:** LIVING DOCUMENT, opened 2026-07-21. Append-style: rank
changes are recorded with dates and the evidence that moved them, not
silently rewritten.

**Discipline (L16 — readiness is evidence):** an item enters this list
only with a named evidence line — a measured scar, a probe-checked
fact, a field instance. Rank = tractability × evidence. Imagined demand
is not evidence; an item whose evidence line is empty stays parked no
matter how tractable. This document claims order-of-work, not
commitment: any item can be displaced by a measured result.

## Built (the record pointers are the evidence)

- Interval/AI propagation core, real + opt-in IEEE semantics, zero-dep
  (`design/ieee-semantics.md`)
- Solver escalation: SMT-LIB2 text, z3/cvc5 portfolio, replay-confirmed
  witnesses, never-on-defaults (`design/solver-integration-build.md`,
  hardened per `design/solver-hardening.md`)
- Constraining-assume with the exactness split
  (`design/constraining-assume.md`)
- Array-aware SMT emission, bounded static shapes, budget 512
  (`design/array-emission-payoff.md`)
- The precondition class: templates + `check()` front door + mandatory
  vacuity (`design/precondition-class.md`, `design/portability-pass.md`,
  user-facing `docs/preconditions.md`)
- IR construction funnel: audited invariants unconstructable at
  `__post_init__` (`design/ci-readiness.md`, Part A census)
- CI posture: flag-and-triage, gate-or-triage four questions, measured
  out-of-sample precision with its four causes (`design/ci-readiness.md`)
- The socket architecture: sockets live with their library, fidelity
  instruments ship in core (`ARCHITECTURE.md`,
  `design/socket-decoupling.md`)
- scatter-add / segment_sum / stack censused rows — sockets trace the
  real assembly instead of transcribing it; classification censuses
  probe-or-exempt (`design/scatter-rows.md`, L22)

## Next, in order

Ordering rule: tractability × evidence, both stated per item.

**1. Reachability conjunct** — def-use/dataflow on the jaxpr: does the
violated quantity flow to a caller-observable output, and does it pass
through a declared guard on the way?
*Tractability:* high — the jaxpr is a first-order dataflow graph
stelling already walks. *Evidence:* measured — the out-of-sample
verification's largest dissolution causes were consequence-path facts a
local obligation cannot see; reachability is the foundation that makes
`reaches-output` checkable and the `CAUGHT_DOWNSTREAM` discharge
re-checkable (`design/finding-conjunction.md`). *Rank rationale:* the
one item that upgrades both the finding conjunction and the future
discharge layer at once.

**2. LA contract layer** — requires (conditioning, mechanized as a
standard obligation) + ensures (κ-derived sensitivity, declared and
stamped, conditional, never solver-checked). Scoped to the measured
conditioning obligation, not general solver-behavior verification.
*Tractability:* medium — the 2x2 QF_NRA reduction is probe-PROVED
(`corpus/supply/la_contract_probe.py`); the seam is the two-faced
verdict structure, not the math. *Evidence:* twice-backed — the
magnetics conditioning scar (measured failure above coefficient
contrast ≈10²) and the D-clamp adjudication, whose cause needed LA
vocabulary to state. *Status:* begun 2026-07-21; explicit stop-clause —
cannot block publish; if the seam exceeds a session, stop and report.
*Update, same day:* first slice landed within the session
(`stelling.contracts`, two templates, requires mechanized / ensures
DECLARED; first-contact audit: ten findings, zero UNSOUND, all fixed —
`design/la-contract-build.md`). Stop-clause not triggered.
*Second update, same day:* attached to the real call site — MIME's
`grad_least_squares` normal system, via `conditioning_2x2_field`
(family transform, symmetry posed) and
`corpus/supply/mime_lsq_conditioning.py` (fidelity stack with measured
discriminating power, L21; the real boundary-starved default REFUTED,
boundary-fed VERIFIED, geometric region solver-decided both ways).
Round-2 critical audit: seven findings, zero UNSOUND, all adjudicated —
`design/la-attachment.md`. Item 2 is complete as scoped; solver help
for family obligations is budget-bounded at ~46 matrices (measured,
documented) — larger meshes pose deliberate sub-families.

**3. Enumerated discharge mechanism** — the A.2 architecture of
`design/finding-conjunction.md`.
*Tractability:* high mechanically. *Evidence:* deliberately incomplete
by design — the category enumeration waits for issue-campaign
maintainer feedback (A.3 sequencing). Not built pre-publish; its rank
here is a scheduling fact, not an evidence gap.

**4. ¬caught framework declarations** — a declarable registry of
framework postconditions ("every public solve rewrites nonfinite
`successful → singular` and raises, `_solve.py:99–113`") that
auto-downgrades matching findings to triage, and that reachability
(item 1) later verifies.
*Tractability:* medium — declaration format + matching. *Evidence:* one
measured instance (lineax). Per L12, wants a second framework instance
before the format is generalized; collecting that instance is part of
the issue campaign's job.

**5. Traced-vs-static classification** — mechanize gate-question 2:
classify each posed input as tracer-capable vs static Python; static →
no-tracer-excuse, which upgrades the finding.
*Tractability:* medium-high — a trace-time census of how the value is
consumed. *Evidence:* the single real lineax finding was static-only
(`max_steps` equality at `cg.py:215/219`), and the tracer excuse
dissolved two others; the classifier would have separated them
mechanically.

**6. Affine/relational domain** — zonotopes or affine forms to carry
correlations interval arithmetic drops.
*Tractability:* low-medium — real implementation cost. *Evidence:* thin
today — no measured case where interval imprecision (as opposed to
envelope truth) flipped or blocked a verdict that mattered. Parked
until a measured UNKNOWN-that-matters names it; recorded here so the
gap is a known unknown, not an invisible one.
*Update 2026-07-22:* unparked — the parking condition was met by the
LA work's own measured straddles (the probe's dependency case; the
socket's solver-discharged symmetry pair) and the build was ordered.
**Landed** as the opt-in `refine="affine"` (zero-UNSOUND audit, fourth
consecutive; built blind to a held-out evaluation measured first).
The reading (`design/affine-refinement.md`,
`corpus/supply/affine_holdout/`): v1 buys the commuted-product class
on real code (the MIME symmetry pair: 4 QF_NRA invocations → 0),
tightens-but-cannot-close the quadratic class exactly as the probe
predicted, and names the v2 frontier by measurement — scatter routing
in the slice layer (the HeatNode flagship, 18/20 elements), select_n
joins, strict roots. Those three are the item's remaining content,
each now carrying a measured evidence line.

## Irreducible — named so the ceiling is honest

- **`¬harmless`** — whether a violated precondition changes the result
  or only the computation (both-branches-valid) is a semantic-
  equivalence judgment; irreducibly human in the general case.
- **The flag half of `silent`** — whether a returned status flag counts
  as loud is framework-semantics knowledge (a BREAKDOWN flag is loud to
  one caller, ignored by another). The trace-crash half mechanizes; the
  flag half does not.

Consequence, stated flatly: the out-of-sample false-positive rate does
not go to zero. The roadmap converts asserted conjuncts into verified
ones and makes the residue cheap to triage (the four questions); it
does not promise gate-grade purity, and a future claim of zero false
positives should be read as a violation of this document.

## Standing out-of-scope

CALMS (never in scope); E2b; general solver-behavior verification
(item 2 is deliberately narrower); dynamic or unbounded-shape array
reasoning; MIME/MADDENING results as counted evidence (usefulness
only).
