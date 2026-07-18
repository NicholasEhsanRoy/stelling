# Soundness

stelling is a verifier: its output is trusted, so its defects are not
ordinary bugs. This file is the project's public account of that trust.

## Policy

**SemVer governs the API. It does not govern verdicts.**

Any change that flips any verdict on any query — verified ↔ falsified ↔
unknown/vacuous, for any harness — is a **soundness event**, regardless of
whether the release is a patch, minor, or major bump. Every soundness event
gets an entry in the log below stating:

- what the defect or behavior change was,
- which stelling versions are affected,
- which prior verdicts are retroactively invalid, characterized as precisely
  as we can,
- what to re-run to re-establish trust.

Silent fixes are forbidden. A soundness fix that ships without a log entry
is itself a soundness event.

## What every verdict must carry

Every verdict object stamps, at minimum:

- stelling version,
- jax version used to trace the harness,
- solver name and version, **and transport** — Python wheel vs. external
  binary path; for an external cvc5, its `--show-config` feature set,
- **the exact solver options used**,
- **the precision configuration the verdict assumes, and the device class
  of any concrete execution it relies on** (counterexample replay,
  differential runs, fuzz crosschecks). Verified on jax 0.10.2:
  `jax_default_matmul_precision` is unset by default — the platform
  chooses — and a `dot_general`'s `precision` param travels in the jaxpr
  as `None`, a *request* resolved per device. XLA offers no default
  precision contract, so a jaxpr's f32 matmul is not a determinate
  computation until the device is known: **one jaxpr, three devices, three
  numerics** — precision configuration is part of what a verdict claims,
  exactly as solver options are. The division of labor with the hash is
  exact and checkable: because `precision` travels in the jaxpr, a pinned
  query and an unpinned one already **hash differently** — the cache can
  never serve one for the other — while two *unpinned* queries on
  different devices **hash identically** with different numerics. The hash
  is half-right by construction; the device stamp closes exactly the other
  half,
- **the arithmetic semantics the verdict speaks** — ℝ vs IEEE float —
  separate from the endpoint representation. `interval/f64/outward-1ulp`
  names how brackets are *computed*, not what the verdict is *about*: the
  first verdict's interval propagation judges obligations in exact real
  arithmetic while the traced program runs in floats, and the gap between
  those is where false-VERIFIED lives — `t + dt > t` is trivially true in
  ℝ and was a 258-day float bug (diffrax#632, a hit in this project's own
  corpus). Conventions consequent on the declared semantics (the
  closed-real-interval `0·∞ = 0` endpoint rule, unsound under IEEE where
  `inf` is a value) are stamped as assumptions, which they are,
- **nonvacuity** — whether the declared input set was mechanically tied to
  known concrete data (membership conditions computed in traced code
  through the same transforms the set is stated in): `checked` /
  `UNCHECKED` / `FAILED`. An inverted or empty declared set verifies
  everything; the checker-level mutation control cannot catch a vacuous
  *harness*, and a VERIFIED with unchecked nonvacuity is a different claim
  from one with it checked — the stamp's job is saying which,
- the query's content hash (`stelling.ir.ClosedJaxpr.content_hash()`; the
  hash covers semantic content and excludes source locations, so identical
  programs traced from different files share verdicts),
- once the transfer registry exists: the assumption tier of every transfer
  function involved (design commitment 5),
- once transfer rules can come from outside the core registry: the
  **provenance of every transfer in the chain** — core vs. contrib vs.
  plugin, with the contributing registry's own version. A verdict that
  leaned on a contrib rule without saying so acquires the
  `TESTED_JAX_SERIES` problem with no fence and no test
  (`design/open-primitive-set.md`).

Solver options are not cosmetic. Observed on cvc5 1.3.4 (PyPI wheel): a goal
containing `exp` solves **unsat** under default options — cvc5 quietly routes
transcendental goals away from the coverings solver — while the same goal
with `nl-cov=true` forced hard-errors ("Term of kind exp is not compatible
with using the coverings-based solver"), and with `nl-cov=false nl-ext=full`
it solves again. **Three configs, three engines, one version string.**
"cvc5 1.3.4 said unsat" is not a reproducible claim; "cvc5 1.3.4, wheel,
options {…}, query `<hash>` said unsat" is.

Six further commitments bind every implementation of the stamp and its
verdicts:

- **The verdict is portable; its discharge is not.** An ℝ-with-margin
  proof is device-independent, but whether margin M absorbs the actual
  rounding is decided by the numerics the device selects — the same proof
  with the same margin can be discharged on one device and not on another,
  without the proof changing. "Verified in ℝ with margin" is a bill that
  comes due per-deployment, not a finished claim, and every artifact
  recording the deferred margin obligation must say so.

- **Never invoke a solver on defaults.** stelling always emits the complete
  option set explicitly — including options whose emitted value currently
  coincides with the solver's default — and the stamp records the emitted
  set. A verdict from "cvc5 1.3.4, defaults" is not reproducible across a
  solver upgrade even with a stamp: the stamp would record what was asked
  for, and a default is precisely what wasn't.
- **Cache the proof, not the report.** The content hash excludes source
  locations, so a cache hit can legitimately come from the same program
  traced in a different file. A rendered verdict must re-derive its
  file/line pointers from the *current* jaxpr's `source_info`; location
  data is never stored in or restored from the cache. The first violation
  of this reports a line number from someone else's file.

- **A conjunctive verdict gets a conjunctive validator.** When a
  verdict's meaning is a conjunction — a REFUTED-with-witness means
  "there exists w such that w is in the declared box AND the predicate is
  false at w" — its validator is the conjunction: one mechanized check in
  one place, never two checks in two functions a refactor can separate.
  The witness-membership defect (2026-07-18 audit) was exactly a
  separated conjunct: violation checked, membership assumed. The witness
  path is now a single validator behind the dispatch path's sole witness
  factory, and the same shape already governs E2a counting (criteria (i)
  and (ii) mechanized together, never satisfiable by halves).

- **Provenance is recorded as it happens, never narrated after.**
  Anything that assembles a description of an event separately from the
  event can diverge from it — the stamp that said "no solver invoked"
  after two real invocations (2026-07-18 audit) was narrated by a
  degradation path. The stamp accumulates: every solver invocation
  appends its full record (name, version, transport, exact options) at
  the moment of invocation, before any result exists; nothing mutates or
  removes an appended record; absence is the derived fact of zero
  appends, never writable text.

- **One invariant, two anti-correlated mechanisms.** The stamp's central
  claim — records equal real invocations — is guarded twice,
  independently: by construction (the append-only ledger above) and by a
  runtime cross-check (a spawn counter incremented at a mechanically
  disjoint code site, asserted equal to the invoked-record count before
  any escalated verdict emits; divergence raises `ProvenanceError` and
  the verdict does not emit). A verdict whose provenance cannot be
  trusted is worse than no verdict — the differential principle, applied
  to the tool's own provenance.

## Log

- **2026-07-18 (pre-release): the stamp contract gained the semantics
  field.** The first verdict (E2a case 1, unreleased) shipped silently ℝ:
  its stamp named the endpoint representation but not which arithmetic
  the verdict was about, and the `0·∞ = 0` interval convention was an
  undeclared commitment to ℝ semantics. No verdict flipped; the field and
  the stamped assumption close a disclosure gap, found before any
  release. Third field the contract has grown (solver options, precision
  config, semantics) — each because something was true and unsaid.

- **2026-07-18 (pre-release, same day): the stamp contract gained the
  nonvacuity field.** An empty or untethered declared set verifies
  everything vacuously, and no existing control catches it: the mutation
  control proves the *checker* isn't vacuous, not that any *harness*
  isn't. Fourth growth, same reason: true and unsaid.

- **2026-07-18 (pre-release, same day): four soundness defects fixed,
  found by an adversarial fresh-context audit** (`design/soundness-audit.md`
  — the portfolio-dispatch discipline applied to review: the transfers'
  author cannot be their auditor). (1) `convert_element_type` passed
  value-changing casts through (verified false VERIFIED on a real
  float32 round-trip trace) — now an exact-conversions whitelist,
  everything else ⊤. (2) `cond` clamped negative indices to branch 0;
  jax's verified convention is default-LAST (index −1 → last branch) —
  verified false VERIFIED at the `[-1, 0]` boundary; fixed to the
  verified convention. (3) int64 array constants above 2⁵³ decoded to
  point intervals excluding the true value — now bracketed. (4) untaken
  `cond` branch equations vanished from the coverage denominator — now
  counted unreached. Also: ⊤ selectors/indices and out-of-double-range
  int literals now degrade instead of crashing; branch and call scopes
  are isolated (the unbound-var check is now effective across scopes);
  `join`/`select_n` refuse shape mismatches. **No shipped verdict
  flipped** — every recorded harness re-run reproduces its recorded
  status exactly (the counting queries used only exact conversions,
  definite in-range indices, and f64 constants); verified empirically
  before this entry was written. Every audit construction is a permanent
  regression test (`tests/test_audit_findings.py`).

- **2026-07-18 (pre-release, same day): second audit pass — two more
  defects fixed, one introduced by the first audit's own fix.** A second
  fresh-context audit aimed at the transfers the first pass's attention
  gradient skipped (`design/soundness-audit.md`, second pass): (1) the
  float→int range guard — added by the previous entry's fix — admitted
  exactly ±2³¹ (int32 max is 2³¹−1; verified false VERIFIED at the
  boundary; for int64 only a strict check is sound because `2⁶³−1`
  rounds back to `2⁶³` in float). Fixed strict. The guard that fixed a
  soundness bug introduced a soundness bug: the guards-generate-hazards
  arc, now with an in-repo instance. (2) `select_n` out-of-range
  selectors **clamp** in measured jax (index −1 → case 0) while `cond`
  defaults to the *last* branch — the two conventions differ on the same
  build, both verified by binding the primitives; the transfer used
  cond's convention for select_n (verified false VERIFIED). Fixed to the
  measured clamp. Two further constructions — `(x+x)·0` on a finite box
  and `r ≤ +∞` over a ⊤ loop output — discharge correctly **under the
  registered ℝ semantics** and are false in IEEE (overflow→NaN); they are
  not code defects under the declared dial, are pinned as marker tests
  that must flip if the dial ever moves, and the ⊤-widening vacuity
  guard already excludes the second shape from every count. The
  "monotone arithmetic is float-conservative" scope claim is corrected:
  outward rounding guards rounding divergence, not overflow→NaN
  existence divergence. No shipped verdict flipped — re-verified by
  re-running every recorded harness after the fixes. 119 tests green.

- **2026-07-18 (pre-release, same day): degrade-don't-crash completed for
  transfer shape guards.** The guards added against silent mis-joins
  (second-audit entry above) still *crashed* the analysis on legal jax
  forms their domain doesn't implement (scalar-`which` `select_n`, rank
  broadcasts) — found by the `any_pytree` target probe's first contact
  with real diffrax/blackjax traces. Transfers now **decline** such forms:
  ⊤ outputs, reason quoted in the verdict notes, counted as unknown in
  coverage. No verdict semantics changed — a crash produced no verdict
  before; regression test added.

- **2026-07-18 (pre-release, same day): the `any_pytree` build passed its
  registered audit gate; three posture escapes fixed before landing.**
  The build (fresh-context builder, closed list, counts withheld) was
  audited by a second fresh context before any commit
  (`design/any-pytree-build.md`). **No unsound verdict path found** — all
  nine mandated constructions passed, including exhaustive
  value-preservation verification of the convert whitelist (every float16
  bit pattern; ±2³¹ edges) and independent reproduction of the
  hand-vs-sugar content-hash equalities. Three FRAGILE escapes from the
  degrade-don't-crash posture, fixed with the auditor's constructions as
  regression tests: (1) literal/const decoding ran outside the decline
  guard — a NaN-sentinel constant (`where(pred, x, nan)`, ubiquitous in
  real code), an undecodable dtype, or a complex literal killed the whole
  analysis; such constants now bind ⊤ with a note (the unbound-var raise
  is untouched — that one is a defect, not a value). (2) `_coords`
  yielded a phantom coordinate for zero-size shapes; the `IndexError`
  bypassed the decline channel. (3) `any_array((inf, inf))` slipped the
  empty-set refusal — an infinite point contains no real under the
  stamped ℝ semantics and yielded vacuous definite verdicts; refused at
  declaration now. No verdict flipped — every recorded harness re-run
  reproduces its status; 195 tests green.

- **2026-07-18 (pre-release, same day): the solver escalation layer
  landed through its gate; one unsound path and six lesser defects
  fixed before landing.** First `SolverStamp(invoked=True)` in the
  project's history: obligations interval propagation leaves unknown
  can now escalate to an SMT portfolio (cvc5 primary for QF_NRA with
  coverings, z3 cross-check; SMT-LIB2 text over stamped transports;
  disagreement raises; timeout is never a VERIFIED; `sat` becomes
  REFUTED only after independent exact-rational replay). The build was
  audited pre-commit by a distinct fresh-context auditor under a
  transport/emission/stamp mandate (`design/solver-integration-build.md`):
  the emission core survived every constructed attack (strictness,
  closed bounds, exact dyadic literals, variable sharing, negation
  polarity, inert-assume non-emission, stamp==wire by recomputed
  script hash), and seven findings were fixed at adjudication —
  (UNSOUND) the witness replay checked predicate violation but not
  **box membership**, so a model escaping the declared box could mint
  a wrong REFUTED-with-witness (found independently by the
  adjudicator's reading and the auditor's construction; replay now
  checks the whole claim and an out-of-box model raises
  emission-infidelity); (FRAGILE ×4) sat-with-no-usable-model
  misattributed emission infidelity and killed the analysis; malformed
  models (conflicting duplicates, undeclared names) were silently
  laundered; the external binary's unsat tolerance converted a crashed
  run (segfault banner, exit 134) into an undisclosed VERIFIED; a
  constants-only refutation degraded through an internal StampError,
  dropped both real invocation stamps, and stamped "no solver invoked"
  after two real invocations; (COSMETIC ×2) stamp-tuple and `only=()`
  validation gaps. Every finding is a permanent regression test
  (`tests/test_solver_audit_findings.py`). **No recorded verdict
  flipped** — every recorded harness runnable in the rebuilt
  environment reproduces its status exactly (hit386, e2a_417,
  exhibit_632, cf_run; verified before this entry); the no-solver path
  is byte-identical and its full pre-existing test baseline passes
  unmodified, including in a jax-present-no-solver environment (291
  passed) and the zero-dep environment (227 passed). 299 tests green
  with both solvers installed.

- **2026-07-18 (pre-release, same day): two audited invariants made
  structural — the hardening pass, no behavior change.** The prior
  entry's UNSOUND and its stamp-integrity FRAGILE shared a shape:
  invariants maintained by convention where they should be enforced by
  construction. Both are now unconstructable: the witness conjunction
  (box membership ∧ predicate violation) is computed by a single
  validator (`obligation.witness_is_valid`) behind the dispatch path's
  sole `Witness` factory, with the constants-only refutation routed
  through the same gate; the stamp is append-only — invocation records
  append fully populated at the moment of invocation, before any result
  exists, absence is derived from zero appends, and a runtime provenance
  gate (spawn counter at a mechanically disjoint site) refuses to emit
  any escalated verdict whose stamp count diverges from actual transport
  spawns (`ProvenanceError`, checked unconditionally). Three new
  commitments above; standing audit-process rules registered in
  `design/soundness-audit.md`. **No verdict status changed anywhere** —
  every recorded harness re-run identical, and both acceptance queries
  produce byte-identical content hashes and byte-identical emitted
  scripts (all four smt2 hashes unchanged); constructed provenance
  divergence in both directions verified to raise, independently of the
  implementer. One stamp-semantics change, disclosed: a transport that
  fails after the invocation is issued (e.g. an exec failure) now stamps
  the invocation with the failure quoted in notes, where it previously
  stamped absence — the ask was real and is fully described; statuses
  unaffected. 313 tests green with both solvers; 305 with jax and no
  solver; 241 zero-dep.

*(no releases yet)*
