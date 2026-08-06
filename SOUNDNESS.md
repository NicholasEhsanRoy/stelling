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
  corpus).

  **That gap is the FIXED-WIDTH BOUNDARY, and it has now been met under five
  names — the same wall each time.** `convert_element_type` declining a
  value-changing cast; int32 `add` wrapping; an accumulation dtype narrower
  than its operands; real-mode-versus-float generally; and — measured by a
  blinded audit — **real-mode boxes excluding the executed value at every
  non-binary64 float dtype**, shared by `add`, `mul` and every arithmetic row
  (float32 `1.0 + 2**-24` boxes to `[1.0000000596…, 1.0000000596…]` and
  executes to `1.0`). The last is not a defect in any row: `ieee` mode is
  gated to binary64, and real mode has no dtype gate at all, so ℝ-judgement of
  a narrower float is the *stated* posture. It is listed here because five
  sightings under five names is what an unnamed structural boundary looks
  like, and the campaign spent measurable effort re-discovering it.

  **What real mode actually brackets, MEASURED — it is not uniformly ℝ, and
  the boundary is DTYPE CLASS rather than semantics:**

  | case | real-mode box | executed jax | follows |
  |---|---|---|---|
  | `add` f32 `1.0 + 2**-24` | `[1.00000006, 1.00000006]` | `1.0` | **ℝ** |
  | `gt` f64 `[1e-320, 1e-300] > 0` | definite TRUE | False | **ℝ** |
  | `mul` f64 `[0,0] * [inf,inf]` | `≈[0, 0]` | `nan` | **ℝ** (stamped convention) |
  | `mul` f64 `1e308 * 10` | `[1.798e308, inf]` | `inf` | **both** — outward rounding brackets it |
  | `sqrt` f64 `[-4, -1]` | DECLINE | `nan` | either — undefined in ℝ too |
  | `add` int32 `INT_MAX + INT_MAX` | **DECLINE** | wraps to `-2` | **EXECUTION** |
  | `div` int32 `-7 / 2` | `[-3, -3]` | `-3` | **EXECUTION** (ℝ gives −3.5) |
  | `convert` f64→f32 rounding | **DECLINE** | rounds | **EXECUTION** |
  | `sign` f64 `[1e-320, 1e-300]` | `[0, 1]` | `0.0` | **EXECUTION** (FTZ) |

  **Floats are judged in ℝ; integers and converts are execution-faithful.**
  The stamp's *"exact real arithmetic"* describes the float half and **is
  silent about the integer half**, where the code has always modelled
  wraparound and truncation — behaviours ℝ does not have. That silence is what
  let both readings look defensible, and it is stated here rather than left to
  be re-derived. **The `sign`/`rem` rows depart from the float half
  deliberately** (they gate on `MIN_NORMAL` where `add`/`mul` do not), because
  declining is safe under both readings while admitting is safe under only
  one; closing that divergence is a mode-wide decision, not a two-row fix.

  **The campaign's strongest result met this question before it was named, and
  answered it by withholding.** The flagship declares `float32` and its
  recorded verdicts are REFUTED with a witness that EXECUTES — safe under
  either reading, because execution is the arbiter and it agreed. Its *holds*
  side was never rendered VERIFIED, for exactly the reason later measured:
  *"a wrong gate on the refuting side yields a witness that either executes or
  does not, and this one executes; on the 'holds' side it would mint a false
  VERIFIED with nothing downstream to catch it."* **A caveat written honestly
  at the time, before anyone knew which reading was right, and it held.**

  Conventions consequent on the declared semantics (the
  closed-real-interval `0·∞ = 0` endpoint rule, unsound under IEEE where
  `inf` is a value) are stamped as assumptions, which they are,
- **nonvacuity** — whether the declared input set was mechanically tied to
  known concrete data (membership conditions computed in traced code
  through the same transforms the set is stated in): `checked` /
  `UNCHECKED` / `FAILED`. An inverted or empty declared set verifies
  everything; the checker-level mutation control cannot catch a vacuous
  *harness*, and a VERIFIED with unchecked nonvacuity is a different claim
  from one with it checked — the stamp's job is saying which,

  **A witness-backed REFUTED is trustworthy on THREE clauses, not two.** It
  must replay, it must execute, **and every declared precondition must hold at
  the witness.** The third is not decorative: an `assume` whose predicate box
  is ⊤ is DROPPED rather than applied, and the query then answers the
  unconditional question. Measured — `assume(jnp.all(x >= 0))` over
  `x ∈ [-10, 10]^3` asserting `sum(x) >= 0` returns REFUTED with the
  replay-confirmed witness `[0, 0, -1]`, which **violates the precondition the
  author wrote**. The earlier claim that a witness-backed REFUTED "cannot
  suffer this failure mode — a witness is a model, and interval arithmetic does
  not produce models" is true of the interval-discharge artifact it was written
  about, and **not true in general**; it is narrowed here rather than left to
  certify results it does not cover.

  **The sharpest form of that, because it is the failure the stamp cannot
  see by itself: a corrupted DECLARATION does not make a verdict false — it
  makes the verdict answer a different question, while remaining internally
  consistent.** Every `VERIFIED` is relative to its declared envelope, so a
  declaration that silently widens (measured: deleting `lo`/`hi` from a
  persisted `stelling_any` turned a declared `[0, 100]` into `(-inf, inf)`)
  still yields a verdict that is *true of the set it was asked about*, and a
  stamp recording the envelope the author intended. Nothing downstream is
  wrong; the question moved. This is why `ir` refuses, at load, an equation
  missing a param jax always supplies, rather than validating the answer:
  **an answer cannot reveal that it is to the wrong question,**

  **A FOURTH way a declaration can be wrong, and the only one that mints a
  REFUTED.** The other three cost a VERIFIED that means less than it looks —
  deserialization corruption above, a dropped `assume` widening the question,
  and a box correctly declared but never occupied. This one manufactures a
  counterexample. `any_array` validated shape and bound ordering but not
  bounds *against dtype*, so a `uint8` declaration of `(-3, -1)` — impossible
  for the dtype, a set no execution can inhabit — was accepted, and `sign`
  returned `[-1, -1]` on it at **100% coverage with no note**, yielding a
  REFUTED whose witness the caller could not reproduce. **A false
  counterexample is the output shape a user trusts most.**

  **Correction, because the first version of this paragraph contradicted a
  measurement twenty lines above it:** it is NOT the only one of the four that
  mints a REFUTED. Way #2 does too, and this document already records the
  instance — `assume(jnp.all(x >= 0))` over `x ∈ [-10,10]^3` returning REFUTED
  with the replay-confirmed witness `[0,0,-1]` that violates the precondition
  the author wrote. What distinguishes #4 is narrower and still worth stating:
  the other three produce a verdict about a DIFFERENT question than the author
  asked, while #4 produces one about **no question at all** — the declared set
  is empty, so the witness cannot be constructed at any dtype. Found by a
  blinded audit reading the claim, not by anything running.

  **The surface was a majority of the transfer set, not one entry point.**
  Driving that same box through every integer-accepting transfer, **the six
  comparisons return a definite boolean straight into an assert**
  (`lt`/`le`/`ne` TRUE, `gt`/`ge`/`eq` FALSE). **The exact count depends on an
  operand convention that was never stated, and neither number is robust:**
  13 admit when the second operand is a valid full-range box, 20 of 26
  traceable rows admit when the impossible box is on every operand — and the
  first version reported "13 of 21" with 21 as the denominator when 21 is
  itself an admit count under a third convention. The qualitative claim is
  what survives; a blinded audit found the arithmetic. Routing `sign`
  through the overflow guard closed one. The hole was at the declaration, so
  the check is: **a declared box holding no value of its dtype is EMPTY and is
  refused at declaration time**, the fourth instance of a posture already held
  for a negative extent, `lo > hi`, and the infinite point. The rule is *no
  representable value inside the interval*, not *a bound outside the range* —
  a box wider than the dtype is an over-approximation and stays sound, so
  `uint8 (-3, 10)` and `float32 (0.0, 1e39)` are admitted while `uint8
  (-3, -1)` and `float32 (1e39, 1e40)` are not. Float bounds ARE tested for exact
  representability — `float32 (0.1, 0.1)` is refused, because it holds no
  float32 and is therefore an empty set like any other. That trade was taken
  after measuring the alternative: range-only admits an interval lying wholly
  inside a representation gap (`float32 (1e-50, 1e-49)`, below the smallest
  subnormal), which reached a REFUTED at 100% coverage. No rule admits one and
  rejects the other, and admitting an empty point costs nothing while
  admitting the gap mints a false counterexample. `float64` is unaffected:
  every python float IS a float64. Complex is admitted unconditionally. **Measured against
  every literal declaration in the campaign corpus — 105 of them — zero are
  refused,** and all 13 entry points close,


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

- **2026-07-18 (pre-release, same day): constraining assume landed
  through its gate — a semantics addition, with the sound direction
  audited through five rounds.** `stelling_assume` now narrows the
  propagated domain for a censused class (comparisons against
  finite-point bounds, either operand order, elementwise, conjunction
  recursion; exact closed-half-space meets; forward-only; scope-local),
  and stays inert — DROPPED, disclosed, with the reason quoted — for
  everything else (relational both-sides-vary above all). Verdicts on
  assume-carrying harnesses may move blocked→posed **by design**;
  `assume_mode="inert"` reproduces the prior behavior byte-identically
  and is the registered comparability/vacuity control
  (`design/obligation-vacuity.md`, constrained-vacuity variant). **No
  recorded verdict flipped** — every recorded harness re-run identical,
  including the MIME F-set (F1 VERIFIED unchanged; F2 still UNKNOWN,
  its DROPPED note now naming the relational block precisely). The
  audit (`design/constraining-assume.md`): round 1 found 2 UNSOUND
  (an empty strict-at-boundary precondition minting a REFUTED; the
  escalation-seam refusal being single-mechanism and bypassable by
  caller mispairing) + 1 FRAGILE + 3 COSMETIC; the standing
  UNSOUND-fixes-are-re-attacked rule then found, on its **first
  application**, an UNSOUND escape in the first fix (box-nonemptiness
  certifies region-nonemptiness only for exact boxes — over-approximated
  intermediates could still mint empty-precondition REFUTEDs), closed
  by the exactness split: definite REFUTEDs under an uncertified
  precondition are withheld to UNKNOWN with the reason disclosed,
  uncertified VERIFIEDs carry a stamped may-be-vacuous line, and scope
  invars never inherit exactness (selector correlation would reopen the
  hole; pinned). A second re-attack returned 0 UNSOUND / 0 FRAGILE.
  New refusals: `UnsatisfiableAssumptionError` (empty meet,
  definitely-false constant, strict-boundary collapse — harness-defect
  class, never VERIFIED) and `MispairedEscalationError` (a constrained
  propagation may only pair with a refusal-shaped escalation — the
  second, independent mechanism per the one-invariant-two-mechanisms
  commitment). Solver escalation declines under constrained assumes
  (emission covers the declared box; `sat` there could witness outside
  the precondition) until narrowed-bounds emission ships as its own
  audited build. 431 tests green with both solvers; 423 with jax and no
  solver; 354 zero-dep.

- **2026-07-19 (pre-release): IEEE semantics landed as a second dial
  position — and the "one jaxpr, three devices" commitment bit twice,
  as a soundness bug this time.** `semantics="ieee"` judges obligations
  about the traced program's binary64 execution (censused: rounding
  collapse via native-endpoint kernels with no outward rounding,
  overflow-as-value, NaN with its comparison algebra); `real` stays the
  default, byte-identical, and every recorded verdict remains a
  real-mode verdict. The two marker tests pinned at the second audit
  ("must consciously flip if the dial ever moves") flipped in the
  predicted direction — that was the registered acceptance criterion —
  and `t + dt > t` with a sub-ulp `dt` is now **REFUTED** under ieee
  while its `dt ≥ ulp(t)` sibling still verifies: the mode models
  float rather than refusing everything.
  **The audit (`design/ieee-semantics.md`) found the mode modelling a
  standard instead of a target.** Measured jax 0.11.0 CPU binary64 is
  **FTZ+DAZ** — subnormals flushed in arithmetic, *in comparisons*, and
  in libm, eager matching jit — while the mode modelled gradual
  underflow: seven end-to-end shapes contradicted the measured execution
  at the declared point (false VERIFIEDs including `x·x > 0` at
  `1e-160` and `x > 0` at `5e-324` with no arithmetic involved; a wrong
  REFUTED on distinct subnormals; and the underflow-boundary shape of
  this project's own diffrax hit). Fixed by a **subnormal haze** — any
  interval meeting the open subnormal band is hulled with 0, at the
  kernels, at comparison operands, and at declarations — so in-band
  outcomes are indeterminate and the mode is sound for flushing *and*
  gradual targets; a stamped assumption discloses this and its measured
  basis. The standing re-attack rule then found the haze **dtype-blind**
  (the band is per-dtype: f32 subnormals are normal f64 numbers, and
  the `_EXACT_CONVERSIONS` whitelist's value-preservation claim is
  measurably false for `f32→f64` under DAZ — two more false VERIFIEDs
  and a wrong REFUTED); fixed by completing the binary64-only guard
  (every ieee comparison and every non-f64-float convert source
  declines with the gap quoted; the assume classifier, which read
  comparison equations behind the guarded transfer, drops them inert —
  it could otherwise have raised a *false harness-defect* claim on a
  comparison that is true at runtime). Second re-attack: clean. Also
  fixed pre-landing: an unenforced maybe-NaN selector invariant
  (FRAGILE) and two disclosure defects. This is the
  precision/device commitment above — "one jaxpr, three devices, three
  numerics" — recurring as a soundness defect rather than a disclosure
  gap; recorded as ledger L10 (`design/lessons-ledger.md`): model the
  measured target, not the standard it claims.
  Two guards ship with the dial: **tightened domains (affine, when
  built) mechanically refuse to run under `real` semantics** — the
  registered IEEE-first precondition is now enforced rather than
  remembered — and **ieee-mode propagation refuses solver escalation**
  (the SMT backends emit over Reals; escalating a float obligation would
  prove the ℝ claim under an ieee stamp), double-guarded like the
  constrained-assume seam. **No `real`-mode verdict changed anywhere**:
  every recorded harness re-runs identical, the pre-existing baseline
  passes unmodified, and real-mode behaviour was byte-compared against
  `HEAD` by the auditor twice. 522 tests green with both solvers; 514
  with jax and no solver; 445 zero-dep.

- **2026-07-19 (pre-release): three censused registry rows landed, and
  the audit they triggered found five UNSOUND defects — including the
  oldest false VERIFIED in the project.** The rows (`reduce_sum`,
  `integer_pow` transfers; `slice` in the SMT emission set) were
  identified by measured attribution, not guessed, and **change verdicts
  by design** — obligations that previously fell to ⊤ can now be decided.
  Their own arithmetic survived exhaustive attack (5040 brackets + 4000
  independent `Decimal` cross-checks on `integer_pow`'s exact-rational
  endpoints; 211,396 commutativity pairs and all 14 ≥3-element routes on
  the ieee association bound; 1007 real and 770 ieee differential
  statuses, zero contradictions). **Nothing the three rows compute was
  unsound.**

  **Those four counts are RECORDED-HISTORICAL and no instrument in this
  repository produces them** — searched for, not assumed: no test and no
  committed sweep computes 5040, 4000, 211,396, or 1007/770, and the
  sessions that ran them left no script. They are reported as what a
  reader can check they are: *the size an attack was reported at*, not a
  number this tree re-derives. The load-bearing part of the sentence — that
  the three rows' own arithmetic held while the audit found defects
  elsewhere — rests on the defects enumerated below, each of which has a
  test. **The norm in `CONTRIBUTING.md` is *"A figure in a norm states
  the UNIT it counts"*, and these four are in this log rather than in a
  norm, so it does not reach them by its own terms — but the property it
  names is exactly the one they fail: no unit, no population, no
  instrument. Applying it here is this entry's extension of it, not a
  quotation of its scope.** They are named rather than quietly dropped,
  because a deleted number with no reason attached gets re-proposed.

  What the audit found was elsewhere, and older:

  1. **Integer arithmetic modelled as unbounded reals — live since the
     MVP, and through prior audits that did not catch it** (recorded at
     the time as *"six"*; that count has no surviving instrument or
     enumeration and is not re-derivable here, so the number is withdrawn
     and the fact it was quantifying — that more than one earlier audit
     passed over this defect — is what remains). `v*v > 0` discharged to VERIFIED
     where jax computes `−1794967296`. Fixed with an **overflow-
     reachability** guard rather than a blanket decline: the exact
     integer result interval is checked against the dtype range, the
     exact answer stands where it fits, ⊤ only where wraparound is
     genuinely reachable — so `i*i > 0` for an index in `{1,4}` still
     discharges.
  2. **The same gap at five emission sites.** A commissioned sweep of
     the emission set found `add`, `sub`, `mul`, `neg` carrying the
     identical defect alongside the reported `integer_pow`; patching only
     the reported instance would have left the original false VERIFIED
     intact through the `mul`.
  3. **`div` on integers, transfer side** — the sweep had covered the
     emission sites and not the transfer sites, and *interval
     propagation mints definite verdicts without ever reaching the
     emission*. Six more false definite verdicts (3 VERIFIED, 3 REFUTED,
     including the `INT_MIN/−1` wrap). Now modelled exactly via
     truncating integer corner arithmetic.
  4. **ieee mode falsified by FMA contraction.** XLA emits
     `multiply_add_fusion`; `(a·b)−1` is `0.0` eager and `−2⁻⁵⁴` under
     jit. Four definite ieee verdicts contradicted by the compiled
     program. A measured-false assumption may not be stamped as if true,
     so this was fixed by **hulling both roundings**, not disclosed away.
  5. **The contraction fix's first form matched jaxpr syntax; XLA
     contracts post-simplification** — ten more false VERIFIEDs through
     intervening equations XLA elides. Closed as a class by **taint**:
     product-derived taint flows from every `mul` through every
     primitive (a declined ⊤ keeps its taint), and every `add`/`sub`
     with a tainted operand hulls both roundings. Soundness no longer
     depends on recognising a shape; verified against chains nobody had
     enumerated. Precision is a separate mechanism, so missing a
     recovery costs tightness and never soundness.

  Three adversarial re-attack rounds ran under the standing
  UNSOUND-fixes-are-re-attacked rule, returning 2 → 1 → 0 further
  UNSOUND; the round that returned zero was the one whose fix had
  stopped enumerating cases. (Correction: this entry said "Four" for a
  sequence with three results. The same revision's ledger counts "three
  successive fix rounds" for the identical 2 → 1 → 0, and its successor
  enumerates exactly three — the count-error class recorded in
  CONTRIBUTING.md, this instance inside the soundness log itself.) Every finding is a permanent regression
  test, each verified to **fail against the unfixed code** — two claims
  in this pass reached a report because an edit was *made* rather than
  because it *landed* (`design/lessons-ledger.md` L15). **No recorded
  verdict moved**: all six recorded harnesses reproduce identical
  statuses, with two coverage lines improving and disclosed. 782 tests
  green with both solvers; 771 with jax and no solver; 674 zero-dep.

- **2026-07-21 (pre-release): bounded static-shape array emission landed
  through seven adversarial rounds — every defect found pre-landing,
  none shipped.** The SMT escalation extends from scalar-only to small
  static-shape array obligations: per-element terms with **broadcast
  sharing preserved** (a broadcast scalar is one constant everywhere —
  the decorrelation hazard), structural ops as index routing against a
  single measured-against-jax oracle, transparent-call descent, array
  asserts as negated element-conjunctions, witnesses that **name the
  violating element** (membership and violation replayed per element
  through the conjunctive validator), and **one per-obligation emission
  budget** (512 element terms *and* root conjuncts, measured against
  solver cost; declines quote both numbers) replacing seven scattered
  scalar gates. Array obligations that declined now decide **by
  design**; scalar emission is byte-identical (pinned script hashes);
  no recorded verdict moved at any round.
  **The first-contact discipline earned its budget.** The build carried
  the audit-this-surface-as-if-it-predates-us instruction, and the arc
  found, across builder, auditor, and four re-attacks: a latent z3
  model-echo screening defect (every expression-valued `define-fun`
  flagged non-rational — live since the first z3 transport); a latent
  routing-oracle hole (jax-illegal `broadcast_in_dim` dims silently
  mis-routing) with its propagation sibling crashing raw; **shape
  nonnegativity validated nowhere** — `any_array((-2,-2), …)` traced
  through the public API and the pipeline coherently minted a
  REFUTED-with-witness over an *empty declared set* (UNSOUND; fixed at
  declaration and at every consumption layer); a budget bypass via
  structural-only inflation; stand-in laundering through lying-aval
  consumers, closed **property-shaped at the env read** (the gate keys
  on the var id — the one handle a consumer must state truthfully —
  with bind-and-register as a single inseparable operation); decoder
  payload-length and dim-type gaps; and a bounded **validation pass at
  `from_dict`** (integral/nonnegative shapes everywhere, declaration
  aval-vs-params, payload lengths, literal/const aval-vs-value) so
  loaded IR meets the same loud posture traced IR always had — with
  per-primitive shape inference explicitly out of scope.
  **Documented residual, by owner-facing decision rather than
  exhaustion:** adversarially hand-constructed IR (never `from_dict`,
  never a trace) whose self-descriptions disagree beyond the bounded
  checks can still reach a definite propagation verdict (the final
  re-attack's I1: a params-vs-aval addend-count dispute in a
  reduction). Classified FRAGILE-by-convention; the invariant that
  holds, verified under attack: *no refused-class value — malformed or
  uninhabited shape, length-lying payload — reaches arithmetic,
  emission, replay, or a verdict by any route.* All 29 recorded
  harnesses round-trip through the validated door with byte-identical
  content hashes and statuses. 906 tests green with both solvers; 758
  zero-dep.

- **2026-07-21 (pre-release): the I1 residual superseded — structurally
  prevented, no longer out-of-contract-by-convention.** The
  construction-path census (`design/ci-readiness.md`) falsified the
  funnel claim ("all IR construction routes through the gates"): `ir` is
  a public module of freely-constructible dataclasses, so direct
  construction — exactly I1's route — was reachable and ungated.
  Validation now runs in the dataclasses' own `__post_init__` (the same
  shared predicates the `from_dict` door uses), so the I1 instance — a
  declaration whose params and aval disagree — **raises at construction
  on every path**, including hand-built IR. The stated residual: per-
  primitive shape inference for non-declaration equations remains out of
  scope (a shape-inference engine, not a validation pass), and deliberate
  constructor circumvention (`object.__new__`) is outside the contract as
  it is for any Python library. Threat-model note: if untrusted
  serialized IR ever becomes a feature, re-audit the door; the
  direct-construction route is closed. Same pass: `check()` gained the
  built-in vacuity control (an entry-point VERIFIED has always been
  widen-checked; mode explicit, never defaulted) — the first
  duty-enforced backstop converted to structure for CI (ledger L18).

- **2026-08-06 (pre-release): the scatter VERIFIED bar narrowed from the
  traced query to the decided obligation's slice — verdicts move, in the
  WITHHELD → AVAILABLE direction.** Not a defect fix, and it is logged
  anyway: the policy above is about verdicts moving, not about who was
  wrong. `verdict.VERIFIED_BARRED_PRIMITIVES` withholds solver-path
  `VERIFIED` on obligations whose SMT emission has not been through a
  distinct-context adversarial pass. It fired on whole-query presence of
  `scatter`, so a verdict resting entirely on obligations the scatter
  emission row was never asked about was withheld for a row that had not
  been consulted. It now fires only when a barred primitive is on the
  EMITTED SLICE of an obligation the solver actually decided, and the
  scope is derived from the traced query at the bar (re-slicing the
  decided obligations) rather than read off the escalation record.
  **Affected versions:** 0.1.0 pre-release only — every build from the
  bar's introduction to `8e42934` inclusive has the whole-query scope;
  nothing has been released.
  **Which prior verdicts are retroactively invalid: none, and the
  direction is the unusual one.** The new fired-set is a subset of the
  old at every verdict, and the guarantee comes from a GUARD rather than
  from the two walks agreeing: `_bar_scope` computes
  `whole = _barred_primitives(closed)` — the old whole-query set — first,
  and returns an empty scope immediately when it is empty, so the
  re-derivation only ever runs on a query the old bar already fired on.
  New-fires therefore implies old-fired structurally, whatever the slice
  walk does. (It is also true that a slice's equations come from the
  query and both roots use the same walk, which is what makes the new set
  the RIGHT subset rather than merely a subset; that argument is pinned
  in `tests/test_bar_walk_parity.py` and is not what the containment
  rests on.) The fallback on any underivable scope is the old
  whole-query set — so nothing that was
  VERIFIED becomes UNKNOWN, and no REFUTED, witness or replay path is
  touched at all. What changes is the opposite: a verdict is now VERIFIED
  where it was previously withheld to UNKNOWN, exactly when (i) it was
  assembled on the solver path with every obligation `discharged` and at
  least one discharged BY a solver, (ii) the traced query contains
  `scatter` at some depth, and (iii) no solver-decided obligation's
  emitted slice contains it. So the honest characterisation is about
  claims becoming available, not about issued claims becoming wrong — but
  a reader comparing a recorded `UNKNOWN` against a fresh run still needs
  this entry, because the two builds disagree and neither is a regression.
  Obligations whose emitted slice DOES carry `scatter` remain withheld;
  the bar is not lifted and its removal point is unchanged.
  **What to re-run:** any recorded solver-path `UNKNOWN` on a
  scatter-bearing query — re-`check()` it with the same
  `solver_timeout_ms` and look for a `VERIFIED withheld` note. Its absence
  where one was recorded is this change and not a new capability. The
  flagship `HeatNode` sweep is unmoved in both directions (its refuting
  side is `REFUTED` with a replayed witness; its holds side is settled by
  intervals and the bar never applied to it) — `docs/verdict-ledger.md`
  carries the scope note.
  Same pass, and the reason this entry is not only about precision: the
  narrowed scope was first implemented by RECORDING the per-obligation
  barred set on `solvers.ObligationEscalation` and trusting it at the bar,
  which lost two immunities the whole-query bar had — an empty recorded
  tuple is a positive claim ("nothing barred on my slice") that nothing
  validated, and `make_solver_verdict` is public and gates mispairing on
  semantics, ieee, constrained-assume and ledger provenance but binds the
  escalation to its `closed` nowhere, so a scatter-free escalation stamped
  against a scatter-bearing query returned VERIFIED where the whole-query
  bar returned UNKNOWN (both measured, both against `8e42934` as control).
  Neither reached a shipped verdict — both need a hand-assembled or
  mispaired call — and both are closed by deriving the scope from the
  query instead of reading it, with the deleted field's site left as a
  comment saying why. One-invariant-two-mechanisms holds: WHICH
  obligations the solver decided comes from the escalation and is already
  load-bearing for the VERIFIED being withheld; WHAT is on their slices
  comes from the query. Regressions in `tests/test_verified_bar.py`;
  emitted-vs-re-derived slice agreement pinned in
  `tests/test_bar_walk_parity.py`.
  **Repair pass, and it moves verdicts again — in the same withheld
  direction as the base, never the other way. Which prior verdicts are
  retroactively invalid: none, for the reason spelled out at the end of
  this paragraph** (the policy above requires the clause by name, and this
  paragraph had its substance without its label). The first implementation
  of "which obligations the solver decided" was
  `outcome == OB_DISCHARGED and r.invocations`, while the obligation loop
  discharged on `outcome == OB_DISCHARGED` alone: one concept, two
  predicates over one record. A record could give up `invocations`, keep
  the discharge that earns the VERIFIED, and drop out of the bar's
  domain. Measured against `8e42934` as control, on a two-obligation
  query whose scatter obligation is the stripped one: base UNKNOWN, that
  implementation VERIFIED. The predicate is now the discharging one, in
  one place. The unification also closes a hole the base itself had:
  with EVERY record's `invocations` emptied the old gate read "no solver
  decided anything" and never entered the bar branch at all, so
  `8e42934` returned VERIFIED there too — it is UNKNOWN on both counts
  now. Every affected assembly needs a hand-edited `ObligationEscalation`
  (the one site that emits `OB_DISCHARGED` reaches it only when a backend
  ANSWERED, and every answering backend was stamped into the ledger
  before its transport ran), so no verdict produced through `check()` or
  through an unmodified `escalate()` moves in either direction, and the
  suite is byte-identical across the change. **What is NOT claimed:** the
  unification is not a defence against a forged escalation, and
  `make_solver_verdict` does not attempt one. Its docstring now states
  the precondition instead — the escalation must be what `escalate()`
  returned for this query — because `stelling.verdict.Verdict` is public
  and a frozen dataclass whose `__post_init__` validates shape and not
  provenance, so a caller able to hand-build a record can hand-build the
  verdict and never call this function at all. The
  narrower true statement, and the one the mechanism supports: the bar's
  scope CONTENTS are re-derived from the query and unforgeable, and its
  DOMAIN cannot disagree with the discharge about the same record.
  **Affected versions:** 0.1.0 pre-release only, and the drifted predicate
  never existed outside a branch — it was introduced with the slice
  scoping and lived only in the intermediate branch-only states between
  the scoping commit and this repair (`caac1ee`, inclusive, up to but not
  including `45cf526`). No tagged build, and no build reachable from
  `main`, ever carried it.
  **What to re-run: nothing.** Stated rather than omitted, because the
  policy requires the clause and "nothing" is an answer to it. Every
  assembly the drift could move needs a hand-edited
  `ObligationEscalation`, no such record can come out of `escalate()`,
  and the suite is byte-identical across the change — so there is no
  recorded verdict whose value depends on which side of it the build was
  on.
  At that pass: 2008 tests passed, 2 skipped, with both solvers and jax
  installed
  (1995 passed, 2 skipped before the scoping pass, 2003 after it; the 8
  tests it added are the regressions named above, and the repair pass
  adds 5 more — the strip-invocations regression in both shapes, the
  containment guard cited above, the `cond` premise restated as a
  behaviour in `tests/test_bar_walk_parity.py`, and the decline-site
  accounting in `tests/test_scatter_gauge_jax.py`. The parity file's
  registry-facts test is rebuilt rather than added: its four registry
  memberships could not be falsified without an import-time census
  raise, so they are replaced by the reach count the same commit let go
  stale).

- **2026-08-06 (pre-release): the narrowed scatter bar was NOT immune to a
  mispaired query, and now is — verdicts move, in the AVAILABLE →
  WITHHELD direction.** The entry above says the narrowed bar keeps the
  whole-query bar's immunity to an escalation stamped against the wrong
  `closed`, and gives the mechanism as "the re-slice declines and it
  falls back to the whole query". **The mechanism is wrong and so is the
  claim.** An obligation index that names a real obligation of the wrong
  query slices out of it perfectly well. Measured on two scatter-bearing
  queries of the SAME SHAPE — one with the scatter on the solver-decided
  obligation (`ON`), one with it on an interval-decided obligation
  (`ELSEWHERE`), both two obligations, both `_barred_primitives ==
  ('scatter',)`:

      ON escalation + ON query         UNKNOWN on 8e42934, caac1ee, 45cf526
      ON escalation + ELSEWHERE query  UNKNOWN on 8e42934 (whole-query bar)
                                       VERIFIED on caac1ee and 45cf526

  `_bar_scope(ELSEWHERE, (0,))` re-sliced to `['ge','sub']`, found
  nothing, and returned `((), '')` — so the bar did not fire and a
  VERIFIED resting on a solver answer that WAS about a scatter slice was
  issued. That is the class this file logs as closed one entry above, in
  the same direction (a false VERIFIED the emission row could have
  caused), reachable by a narrower door. The prior test of it passed
  because its fixture is the one arrangement that fails safe: its
  mispaired index reaches no real obligation of the wrong query, so the
  re-slice declines and the fallback fires.
  **The repair keys on the property instead of on the index — and its
  FIRST version keyed on the wrong property. The paragraph that follows
  is what was believed; the entry below it is what was measured.** Every
  solver invocation already stamps `smt2_sha256`, the sha256 of the exact
  SMT-LIB2 text it was sent, and emission is a pure function of (slice,
  solver flavour, timeout). So `verdict._evidence_is_about` re-emits the
  slice re-derived out of the query being stamped, with the flavour and
  timeout the stamp itself records, and narrows the bar for that
  obligation only when the hash comes back equal. Measured on the pair
  above: the correct pairing reproduces it for both portfolio members;
  that mispairing reproduces neither. **What was NOT measured, and is
  false, is the general claim — "the mispairing reproduces neither" is
  true of that fixture and not of the shape it stands for.** See the
  entry below. Every other outcome — no stamps, no hash, an
  unrecognised option profile, an emission that raises, a slice that
  declines — returns the whole-query set. `invocations` therefore cannot
  CLEAR the bar, only fail to lift it, which is the opposite polarity
  from the deleted `barred_on_slice` field and from the
  `and r.invocations` drift the entry above repairs; the bar's DOMAIN is
  still `outcome == OB_DISCHARGED` alone.
  **Affected versions:** 0.1.0 pre-release only — `caac1ee` (the scoping
  commit) through `eb1ff86` inclusive, all branch-only. The end of the
  range was originally logged as `45cf526`; it is `eb1ff86`, because the
  hash-keyed repair at `45cf526` does not close the shape (below). *(A
  previous correction here read "`45cf526` and `eb1ff86` were originally
  logged here as the end of the range; they are not". Both halves were
  wrong: `eb1ff86` was never in the old text — `git show eb1ff86:SOUNDNESS.md`
  reads "through `45cf526` inclusive" — and saying `eb1ff86` is not the end
  of the range contradicts the clause two lines above, which is the range
  that is correct.)* Builds up to and including `8e42934` have the
  whole-query bar, and the mispairings in this entry and the next both
  measure UNKNOWN there; nothing has been released.
  **Which prior verdicts are retroactively invalid: none, and the
  direction is the ordinary one this time.** Reaching the hole needs a
  call to the public `make_solver_verdict` pairing an escalation with a
  query it did not come from — `check()` cannot produce it, and
  `escalate()` cannot produce a record for a query it was not run on. No
  verdict in `docs/verdict-ledger.md` is affected. What changes for an
  HONEST caller is the reverse and it is a withholding: a solver-decided
  VERIFIED on a scatter-bearing query is now narrowed only when the
  re-emitted script matches, so any obligation whose invocation was not
  stamped with a reproducible script hash gets the whole-query bar and
  reads UNKNOWN where it read VERIFIED. No such case exists in this
  suite — the scatter-off-the-decided-slice fixture still VERIFIES, and
  `tests/test_verified_bar.py` asserts the hash equality directly so that
  a future emission that stopped being a function of its inputs fails
  loudly instead of silently widening every bar.
  **What to re-run:** any recorded solver-path VERIFIED on a
  scatter-bearing query — re-`check()` it with the same
  `solver_timeout_ms`. A newly-present `VERIFIED withheld` note whose
  clause says "no recorded solver invocation … re-emits from this query's
  slice of it" is this change, and it means the narrowing was never
  established for that verdict rather than that the program changed. (The
  entry below re-words that clause; from that pass on it reads "…
  reproduces both this query's slice of it and the script that slice
  emits".)
  Same pass, and NOT a verdict-moving change: the SET row's routing was
  gauged only at the written index 0, where every mis-route defined as an
  offset from the index collapses back onto it. The line-neutral
  corruption `i == k` -> `i == (k - 1 if k > 0 else k)` in
  `obligation._scatter_set_plan` turned `s = x.at[2].set(u); assert
  s[1] - x[1] >= 1.0` from `violated-witness` into `discharged` — a
  MISSED violation on the very row this bar exists for — while the whole
  suite under CI's install set stayed at 2004 passed, 6 skipped, fully
  green (the only two tests that caught it are
  `pytest.importorskip("maddening")`-gated, and CI installs `".[solvers]"`
  and `".[solvers,jax]"`). A fourth routing fixture at index 2 and the
  matching gauge mutation close it: the same corruption now fails
  `test_gauge_catches_every_mutation` with no maddening installed
  (1 failed, 2007 passed, 6 skipped). No shipped code changed, so no
  verdict moves and there is nothing to re-run for it.
  2012 tests passed, 2 skipped with both solvers and jax installed
  (2008 before this pass; the 4 added are the mispairing regression and
  its hash direction, and the two fail-closed fallback pins, all in
  `tests/test_verified_bar.py` — the record-field channel test is
  rewritten rather than added, because pinning today's field list could
  not fail for the defect it exists to catch).

- **2026-08-06 (pre-release): the script hash does not identify the slice
  that produced it, so the mispairing above was NOT closed by keying on it
  — verdicts move again, AVAILABLE → WITHHELD.** The entry above narrows
  the bar for an obligation when re-emitting the slice re-derived out of
  the query being stamped reproduces the recorded `smt2_sha256`, and
  claims "the mispairing reproduces neither". Emission IS a pure function
  of (slice, flavour, timeout); what the guard needs is the CONVERSE,
  *equal script implies equal slice*, **and that is false for exactly the
  primitive under the bar.** The static-index `scatter` SET row appends no
  line at all (`smt.emit`, the `prim == "scatter"` branch: element k's
  term IS the update's, every other element's term IS the operand's), so
  for an element the write did not touch, `s[i]` aliases the operand's
  term. Measured, jax 0.11.0, x64, `s = x.at[0].set(0.5)`:

      slice of `s[1] - x[1] <= 0`   barred ('scatter',)   sha 2896a0f2…
      slice of `x[1] - x[1] <= 0`   barred ()             sha 2896a0f2…   collides
      slice of `s[0] - x[0] >= 0`   barred ('scatter',)   sha 2de5e041…
      slice of `x[0] - x[0] >= 0`   barred ()             sha 2f2e0ed8…   no collision

  So the same mispairing survives, one fixture over. With the ELSEWHERE
  query differing from the ON query ONLY in where the scatter sits (`x[1]`
  where the other reads `s[1]`, same inputs, same predicate, same second
  obligation):

      ON escalation + ON query         UNKNOWN on 8e42934, caac1ee,
                                       45cf526, eb1ff86
      ON escalation + ELSEWHERE query  UNKNOWN on 8e42934 (whole-query bar)
                                       VERIFIED on caac1ee, 45cf526, eb1ff86

  `_evidence_is_about` returned True and `_bar_scope` returned `((), '')`.
  **AND IT IS A FALSE VERIFIED, NOT A MERELY PREMATURE ONE.** The script hash
  does pin the TEXT, so an `unsat` about it is an `unsat` about the obligation
  this query's own slice emits; what it does not pin is the rest of the
  verdict, and in particular the obligations the mispaired PROPAGATION
  decided. With the ELSEWHERE query's SECOND obligation made false
  (`s >= 0.5`, which fails at `x = [0,0,0]`), the same assembly gives:

      ELSEWHERE checked honestly       REFUTED on all four builds
      ON escalation + ELSEWHERE query  UNKNOWN  on 8e42934
                                       VERIFIED on caac1ee, 45cf526, eb1ff86

  The whole-query bar was a backstop against a mispaired assembly on ANY
  scatter-bearing query; the byte-collision removed it for exactly the shape
  where the emitted script cannot tell the two slices apart, and what came
  through was a VERIFIED on a REFUTED query.
  **The regression test for the entry above could not see this, because
  its fixture built away its own trigger:** it said "the one difference is
  WHERE" while also introducing a fresh scalar input and a different
  predicate, and it was that — not the scatter's location — that made the
  two scripts differ. Sharpest evidence: at `eb1ff86`, deleting the
  `_evidence_is_about` call from `_bar_scope` entirely reddens exactly one
  test, the one whose fixture cannot exhibit the defect (measured: 1
  failed, 2011 passed, 2 skipped).
  **The repair takes the key from the SLICE, because no script-derived
  quantity can work — emission is lossy for precisely the barred
  primitive.** `smt.slice_fingerprint` hashes the slice's primitive names
  with their nesting depth, walked through the same canonical accessor
  (`coverage.sub_jaxprs`) the bar's own walk uses, and rides in the stamp
  as `slice_sha256` beside `smt2_sha256`. `_evidence_is_about` now
  requires BOTH. What each one proves, exactly: the script hash proves the
  TEXT the solver answered about is the text this slice emits, which is
  what makes the answer transferable; the slice fingerprint proves the
  emission ran on a slice with the same primitive topology, hence the same
  BARRED SET, which is the bar's actual question and the one the text
  cannot answer. **What neither proves:** that the record is honest —
  both are record-carried, as `smt2_sha256` already was. Adding a conjunct
  can only make narrowing RARER, so no bar fires less than it did at
  `eb1ff86`.
  **AND A FINDING THIS FILE FIRST RECORDED AS A "COST OF SCOPING", WHICH IT
  IS NOT — CORRECTED HERE, AND THE CORRECTION IS THE MEASUREMENT.** The
  original wording said narrowing the bar to the decided obligation's slice
  "gave up a backstop the whole-query bar provided by accident": when the two
  queries' decided slices are the SAME EXPRESSION — not merely byte-colliding
  — both hashes match, the bar narrows correctly (the barred row really was
  not involved), and the mispaired assembly returns VERIFIED on a query whose
  honest verdict is REFUTED (UNKNOWN on `8e42934`; VERIFIED on `caac1ee`,
  `45cf526`, `eb1ff86`, `f5280cf`). Everything in that sentence is true except
  what it attributes the loss to. The whole-query bar's immunity covered
  queries carrying a barred primitive — the only ones ANY version of the bar
  looks at — so it was a coincidence of scope and not a mechanism. Measured on
  this branch: the identical mispaired VERIFIED, on a query whose honest
  verdict is REFUTED, is reachable with **no barred primitive anywhere**, on
  every build including `8e42934`:

  | mispairing on a REFUTED query | 8e42934 | eb1ff86 | f5280cf | here |
  |---|---|---|---|---|
  | identical decided slice, scatter-bearing | UNKNOWN | VERIFIED | VERIFIED | refused |
  | **scatter-FREE** | **VERIFIED** | **VERIFIED** | **VERIFIED** | refused |

  So the correct statement is not "scoping cost a backstop" but **scoping
  revealed that `make_solver_verdict` never bound its three arguments to one
  query.** The repair is a fourth `MispairedEscalationError`, the QUERY
  PAIRING GATE: `escalate` records `ir.ClosedJaxpr.content_hash()` of the
  query it ran on, at every one of its five return sites, and assembly
  recomputes it from the `closed` it is handed and refuses the pair when they
  differ. Same trust model as the three gates beside it and as the
  record-carried `smt2_sha256`/`slice_sha256` — it defends an honest caller
  against an accidentally mispaired assembly, the realistic mechanism being a
  CACHED escalation, which is one of the two uses `stelling.ir`'s module
  docstring names `content_hash` for. Costs nothing at assembly (the stamp
  already took that hash; the gate compares the same value, asserted by
  counting the calls), one hash in `escalate`.
  **What the gate does NOT bind: `propagation`.** `Propagation` lives in
  `stelling.propagate`, which this pass leaves at zero line delta, so there is
  no field on it to record the query in. The residue — (this query, ANOTHER
  query's propagation, this query's escalation) — assembles to VERIFIED with
  the other query's obligations reported under this query's hash. It is a live
  test, not a comment:
  `tests/test_verified_bar.py::test_the_pairing_gate_binds_the_ESCALATION_and_not_the_propagation`.
  The bar and the gate are anti-correlated and both stay: the gate keys on the
  whole query's content hash, the bar on one slice's fingerprint and script,
  and the bar's own mispairing regressions now satisfy the gate by hand
  (`_past_the_pairing_gate`) so neither can hide the other's failure.
  Consumers needing more should judge through
  `stelling.preconditions.check`, which owns all three sides.
  **Affected versions:** 0.1.0 pre-release only — `caac1ee` through
  `eb1ff86` inclusive, all branch-only, nothing released. Builds up to and
  including `8e42934` have the whole-query bar and measure UNKNOWN on this
  mispairing.
  **Which prior verdicts are retroactively invalid: none, and this clause is
  doing more work than usual, so it says what it rests on.** The defect
  produces a false VERIFIED, but reaching it needs a call to the public
  `make_solver_verdict` pairing an escalation with a query it did not come
  from; `check()` cannot produce that and `escalate()` cannot produce a record
  for a query it was not run on. No verdict in `docs/verdict-ledger.md` is
  affected, and every verdict in this repo's own history was assembled through
  `check()`. What is NOT claimed: that a downstream caller of
  `make_solver_verdict` cannot have mispaired one. Anyone who has should
  re-run per the clause below.
  **What to re-run:** any recorded solver-path VERIFIED on a
  scatter-bearing query — re-`check()` with the same `solver_timeout_ms`
  and look for a `VERIFIED withheld` note whose clause says "no recorded
  solver invocation … reproduces both this query's slice of it and the
  script that slice emits". Its presence means the narrowing was never
  established for that verdict.
  Same pass, three more, none of them verdict-moving through `check()`:
  **(1)** the SET row's routing was gauged at index 0 and then at {0, 2},
  which is still a sample. Two line-neutral corruptions walk between the
  samples — `i == (0 if k == 1 else k)` and `i == (0 if k > 2 else k)` in
  `obligation._scatter_set_plan` — each turning a `violated-witness` into
  a `discharged` (a MISSED violation, the direction the bar exists for)
  with the full suite green under CI's install set at 2008 passed, 6
  skipped. The property is now pinned instead of sampled: every k of the
  axis at four axis lengths, against jax's own execution of the same
  `.set` as the oracle, plus the same sweep at the escalation surface.
  **(2)** the bar's domain is read in one place, `solvers._bar_domain`,
  handed a record with no field but `index`, `outcome` and `invocations`,
  so a conjunct on a new field of ANY type raises instead of evaluating.
  The field-probe test it supplements moves each field to two values of
  its declared type, which EXHAUSTS `bool` and merely SAMPLES `str`:
  measured on `eb1ff86`, `audit_token: str = ""` plus
  `and r.audit_token != "clean"` in the domain is UNKNOWN at both probe
  values and VERIFIED at `'clean'`, full suite green.
  **(3)** "a bar must never break a verdict" did not cover the read that
  feeds the bar: at `eb1ff86` a record whose `invocations` is a `list`
  raised `TypeError` out of `make_solver_verdict` (`tuple + list`), from
  outside `_bar_scope`'s protective `try`. `_bar_domain` tolerates it, and
  an unreadable escalation widens to the whole-query set rather than
  raising or silencing the bar. `45cf526` tolerated the same record, so
  this is a regression the branch introduced and the branch removes.
  At this pass: 2035 passed, 2 skipped with both solvers, jax and maddening
  installed; 2031 passed, 6 skipped under CI's install set (`.[solvers,jax]`,
  no maddening). Before it: 2012 / 2 and 2008 / 6. The 23 added tests are the
  collision measurement and its two mispairing parametrisations, the
  `_bar_scope`-level widening, the false-VERIFIED regression, the documented
  scoping LIMIT, the domain-channel pair, the three `invocations`-shape pins,
  the four-behaviour stray-index pin, the fingerprint-walk parity over five
  nesting shapes plus its old-accessor control, and the SET plan's per-k
  sweep at four axis lengths with its escalation-surface twin.

- **2026-08-06 (pre-release, later the same day): `make_solver_verdict`
  never bound its three arguments to one query, and the scatter bar was
  never what stood in for that.** The entry above records the mispaired
  false VERIFIED as "the cost of scoping the bar"; it is corrected in
  place, and this entry is the repair.
  **What the defect was.** `make_solver_verdict(closed, propagation,
  escalation)` had four gates — ledger provenance, the symmetric semantics
  pairing, the ieee refusal, the constrained-assume refusal — and none of
  them asked whether the escalation came from `closed` at all. An
  `OB_DISCHARGED` record discharges an obligation by INDEX, so an
  escalation produced on query A, assembled against query B, discharges
  B's obligations and returns VERIFIED on a query whose honest verdict is
  REFUTED. The whole-query bar looked like a backstop for that, but only on
  queries carrying a barred primitive — the only ones any version of the
  bar inspects. Measured on this branch, on a query with **no `scatter`
  anywhere**, where no version of the bar has ever fired:

  | mispairing on a REFUTED query | 8e42934 | eb1ff86 | f5280cf | here |
  |---|---|---|---|---|
  | identical decided slice, scatter-bearing | UNKNOWN | VERIFIED | VERIFIED | refused |
  | **scatter-FREE** | **VERIFIED** | **VERIFIED** | **VERIFIED** | refused |

  So the bar's immunity was a coincidence of scope, not a mechanism, and
  the finding is not a cost of scoping.
  **The repair** is a fifth gate, the QUERY PAIRING GATE: `escalate`
  records `ir.ClosedJaxpr.content_hash()` of the query it ran on at every
  one of its five return sites, assembly recomputes it from the `closed` it
  is handed, and a mismatch raises `MispairedEscalationError`. Same trust
  model as the gates beside it and as the record-carried
  `smt2_sha256`/`slice_sha256` — it defends an honest caller against an
  accidentally mispaired assembly, the realistic mechanism being a CACHED
  escalation, which is one of the two uses `stelling.ir`'s own module
  docstring names `content_hash` for. It costs no additional hash at
  assembly (the stamp already took that one; the gate compares the same
  value, asserted by counting the calls rather than by timing), and one
  hash per `escalate` — measured on an idle machine at 0.112 ms against an
  89.3 ms two-obligation escalation, 0.125% of it.
  **What it does NOT bind: `propagation`.** `Propagation` lives in
  `stelling.propagate`, held at zero line delta this pass, so there is no
  field on it to record the query in. The residue — this query, ANOTHER
  query's propagation, this query's escalation — assembles to VERIFIED with
  the other query's obligations reported under this query's hash, and is a
  live test rather than a comment
  (`test_the_pairing_gate_binds_the_ESCALATION_and_not_the_propagation`).
  **Four more, same pass, none of them verdict-moving through `check()`:**
  **(1)** the "a record cannot certify itself" pin saw ONE SPELLING. Six
  channels each turned the bar's UNKNOWN into VERIFIED with the full suite
  green in both columns: a key read out of `SolverStamp.options` (no new
  field anywhere), a conjunct at the CALL SITE before the bar, a
  `getattr`-with-default inside `_bar_domain`, a conjunct at the call site
  AFTER the bar, `type(r.index) is not int` exploited with an `int`
  subclass (no new field, no new key), and the token smuggled through
  `options` and read in `_bar_scope`. Removing a record's fields catches
  the first access form only — measured: plain attribute and `@property`
  yes; `getattr` with a default, a `hasattr` guard and `__dict__.get` no —
  and pins the invariant at one producer while the same conjunct one
  function over does the same job. The pin is now a READ LEDGER over the
  whole assembly (every attribute access on a record or a stamp is logged
  with the function that made it, and the allow-list is asserted in both
  directions), a WHITELIST PROJECTION of `options` down to four named keys
  with the key set asserted exactly, and a type-identity invariance
  property for the one channel that reads nothing new.
  **(2)** `_bar_domain`'s outer `except` was driven by nothing, so the
  sentinel's truthiness — its whole mechanism — was unpinned: `__bool__`
  returning `False` turned an unreadable escalation's UNKNOWN into
  VERIFIED, suite green.
  **(3)** a silencing path that never reached the sentinel: a `records`
  iterable that can be consumed once was exhausted by the obligation loop
  before the domain was read, so `_bar_domain` returned an honest-empty
  `{}` and the bar was skipped — VERIFIED with no withheld note, identical
  at `eb1ff86`. Closed by ORDER: the domain is read on the first pass, so a
  degenerate `records` costs the discharges rather than the bar.
  **(4)** the SET/ADD route sweep exhausted k and SAMPLED n. Two
  line-neutral corruptions walk through it, each a `violated-witness`
  turned `discharged`, both green in both columns: `i == (k if n != 6 else
  0)` in `_scatter_set_plan`, and `groups[k*rowsz + t] -> groups[k*rowsz]`
  in `_scatter_add_plan`, whose row arithmetic was ungauged above rank 1
  entirely. Both sweeps now range over a SPACE built by a rule, with the
  space asserted to be the rule's.
  **Affected versions:** 0.1.0 pre-release only — the pairing hole is
  present in EVERY build in this repository's history up to and including
  `f5280cf`, and on the scatter-free shape that includes `8e42934` and
  everything before it; all branch-only, nothing released. The five other
  items are branch-only over `caac1ee`…`f5280cf` except (3), which is
  present from `eb1ff86`.
  **Which prior verdicts are retroactively invalid: none.** Reaching any
  of these needs a call to the public `make_solver_verdict` pairing an
  escalation, a propagation or a hand-edited record with a query it did not
  come from. `stelling.preconditions.check` cannot mispair: its one
  pipeline binds `closed` and `prop` as locals off a single `trace`, and
  passes those same two to `escalate` and then to `make_solver_verdict`.
  `make_solver_verdict` is not in `stelling.__all__` and is not an
  attribute of the package at all; measured, its only mentions outside the
  library and its tests are three internal `design/` notes, this file, and
  the paragraph in `docs/verdict-ledger.md` that discloses this very
  defect — no README, tutorial or API page reaches it. And no verdict in
  `docs/verdict-ledger.md` was assembled any other way than through
  `check()`. What is NOT claimed: that a downstream caller cannot have
  mispaired one.
  **What to re-run:** any recorded solver-path VERIFIED assembled through
  `make_solver_verdict` directly rather than through `check()` — re-run it
  on this build; a mispaired pair now raises `MispairedEscalationError`
  instead of returning a verdict, so the re-run either reproduces the
  verdict or names the mispairing. Verdicts from `check()` need no re-run.
  At this pass: 2044 passed, 2 skipped with both solvers, jax and maddening
  installed; 2040 passed, 6 skipped under CI's install set (`.[solvers,jax]`,
  no maddening). Before it: 2035 / 2 and 2031 / 6.
  **Record-keeping corrections, made because this file's subject is claims
  that stopped being true:**
  * the previous pass's SOUNDNESS.md corrections are claimed by `ed183e8`'s
    commit message and are not in it — `git show --stat ed183e8` touches
    `docs/verdict-ledger.md`, `smt.py` and two test files and no
    `SOUNDNESS.md`. They are in `114b846`, whose own message never mentions
    them. Commit messages cannot be edited without rewriting the audited
    base, so the correction is here.
  * `docs/norms.md`'s skip disclosure claimed the two `blackjax` skips were
    "the ONLY skips the suite reports under jax and both solvers — two of
    them". Under exactly that install set there are SIX (2 blackjax + 4
    maddening), which this file recorded four files away. Corrected, and
    the replacement names its install set.
  * this file's four-element policy (what changed / which versions / which
    verdicts invalid / what to re-run) is met BY NAME by 4 of its 17 log
    entries — the four dated 2026-08-06. The other 13, all dated
    2026-07-18 to 2026-07-21, state their substance in prose without the
    labels; SEVEN of those lack "what to re-run" by substance as well,
    namely *the stamp contract gained the semantics field*, *the stamp
    contract gained the nonvacuity field*, *degrade-don't-crash completed
    for the escalation layer*, *the solver escalation layer landed*, *three
    censused registry rows landed*, *bounded static-shape array emission
    landed*, and *the I1 residual superseded*. Not fixed here — this pass
    fixed the entries it touched and states the rest rather than leaving
    the standard re-affirmed and unmet. Cited by headline rather than by
    line number, because a line number in this file is a claim that goes
    stale on the next edit, which is this file's own subject.

- **2026-08-06 (pre-release): the two scatter rows now DECLINE outside the
  space they are gauged on, and the bar gained a seventh closed channel.**
  A blinded audit of the previous entry's repairs, answered here.
  **What changed — the one item that MOVES VERDICTS.** The SET and ADD route
  sweeps are exhaustive over a space and blind one step past it, and two
  more line-neutral corruptions were measured living in that step, each a
  `violated-witness` turned `discharged` (a MISSED violation) with the full
  suite green in both columns: `i == (k if n != 9 else 0)` in
  `_scatter_set_plan`, where the sweep exhausts n ≤ 8; and
  `groups[(k if operand_shape[0] < 4 else 0) * rowsz + t]` in
  `_scatter_add_plan`, where the sweep's dims stop at 3 so no axis of length
  ≥ 4 exists in it — and where the non-degeneracy clause was on `rowsz`, the
  TRAILING product, which structurally cannot see a leading-axis-keyed
  corruption. This is the FIFTH instance of one pattern in this repository
  (a route gauge sampling k = {0} then {0,2}; field probes by name then by
  type; an arity family widened 2 → 3 → 8, where the escape sat at exactly
  the declared ceiling), and raising the bound has now failed four times.
  So the bounds are not raised: **admission is narrowed to the gauged
  space.** `stelling.obligation` declines a SET operand longer than 8 and a
  scatter-add operand outside rank ≤ 3 / every axis ≤ 3 / ≤ 12 elements, and
  `tests/test_scatter_gauge_jax.py` pins the two spaces EQUAL in both
  directions, so widening admission without widening the sweep is red. The
  corrupted branches are then unreachable rather than uncaught — measured:
  with both corruptions applied on top of this change, n = 9 and (4,2) come
  back `unknown`, and with the guard removed they come back `discharged`
  again. The interval TRANSFER is untouched; only the emission/slicing face
  declines.
  **Which prior verdicts are retroactively invalid: none, and this is not
  an unsoundness fix.** Nothing that was VERIFIED becomes REFUTED or vice
  versa. What changes is that some obligations that used to be ANSWERED are
  now UNDECIDED: past those bounds a solver-path `discharged` and a
  `violated-witness` both become `unknown`, so the verdict becomes UNKNOWN.
  A verdict that moves to UNKNOWN is still a verdict move under this file's
  policy, so it is logged. It costs REFUTATIONS as well as discharges, which
  is stated rather than buried:
  `test_a_shape_past_the_gauge_costs_the_ANSWER_and_never_the_SOUNDNESS`
  pins exactly that at n = 8 versus n = 9. Blast radius across the
  PYTEST-DRIVEN TREE is ZERO, measured rather than assumed (the scope is the
  instrument's — see the next entry's item on what a census of the suite can
  and cannot reach) — a census of every
  operand shape reaching either row across the whole suite finds SET at
  1..8 (plus one (200,) that already declined on the int8 index-dtype rule)
  and ADD entirely inside rank 3 / dim 3 / 12 elements.
  **What to re-run:** any recorded verdict over a harness that writes
  `x.at[k].set(v)` on an axis longer than 8, or `x.at[k].add(u)` /
  `jax.ops.segment_sum` on an operand outside rank ≤ 3 / dim ≤ 3 / 12
  elements, AND whose obligations were decided by the SOLVER rather than by
  intervals. Those verdicts are not wrong, but this build will return
  UNKNOWN where they returned an answer, and the difference is a disclosure
  about what was gauged rather than a correction. Verdicts whose
  obligations intervals settled are unaffected — the transfer is untouched.
  **Affected versions:** 0.1.0 pre-release only; branch-only, nothing
  released. Both corruptions are reachable on every build in this
  repository's history through `e35de13`.
  **Three further items this pass, none of them verdict-moving:**
  **(1)** A SEVENTH certify-itself channel, and the first that needs no
  forged record at all: a conjunct on the **VALUE** of a whitelisted option
  key, driven by a public keyword argument. `solver_timeout_ms` is carried
  verbatim into the stamp as `:timeout`; `:timeout` must be in
  `_EVIDENCE_OPTION_KEYS` (the budget is part of the emitted text and
  therefore part of the hash the narrowing compares); and nothing
  constrained what the decision did with a whitelisted key's value.
  Measured: `check(..., solver_timeout_ms=31337)` returned VERIFIED with no
  note where 20000 returns UNKNOWN, with the full suite byte-identical in
  both columns. The read ledger missed it because it attributed the read to
  the PROJECTION HELPER rather than to the function that asked, so calling
  the permitted projection from `_bar_scope` logged an already-permitted
  pair. Closed by two anti-correlated mechanisms: the ledger now attributes
  an `options` read to the function that ASKED (and the source scan forbids
  the projection CALL outside the one permitted reader), and the bar's
  answer is pinned INVARIANT under every caller-settable option value over
  seven orders of magnitude. Four spellings are red, and the split is the
  evidence: three are caught by both mechanisms, and one — the same
  conjunct written inside the permitted reader — by the property pin alone.
  **(2)** The previous entry's item (3) recorded that ordering the bar's
  domain first meant "a degenerate `records` costs the discharges rather
  than the bar". Measured, that is broader than it reads, and the ordering
  was not the whole closure either. On a SCATTER-FREE query — one the bar
  never touches — a one-shot `records` turned an honest VERIFIED into
  UNKNOWN, carrying the generic undecided-cause note, which attributes the
  UNKNOWN to an interval straddle: a wrong explanation rather than silence.
  And a TWO-FACED `records` (empty on the first pass, real on every later
  one) showed the bar an honest-empty domain and the obligation loop a full
  set of discharging records — VERIFIED, no withheld note, on the bar's own
  fixture, which ordering cannot see. Both close with ONE PASS over
  `records`, taken at the top of assembly: a degenerate `records` now
  behaves exactly like the tuple it yields. Ordering is kept as a
  now-redundant second mechanism and the comment says so, because a
  mutation of the ordering alone is inert.
  **(3)** `Escalation.query_sha256`'s docstring said the pairing gate
  refuses an empty hash "too". It did not, in the one case where it
  matters: both legs come from `_query_sha256`, which returns `""` when
  `ClosedJaxpr.content_hash()` raises, so an unhashable query and an
  unrecorded escalation compared EQUAL and the gate passed. The refusal
  came from `Stamp.__post_init__` one layer later. The gate now refuses an
  empty hash on either leg, so the sentence is true where it stands, and
  both docstrings say where the refusal happens and that they were wrong
  before. Also stated correctly and left alone: `carries_work == False`
  bypasses the gate entirely, and an inert escalation contributes nothing
  (measured UNKNOWN off the propagation alone).
  At this pass: 2055 passed, 2 skipped with both solvers, jax and maddening
  installed; 2051 passed, 6 skipped under CI's install set
  (`.[solvers,jax]`, no maddening). Before it: 2044 / 2 and 2040 / 6.
  **What this pass did NOT close, named rather than left implied:** the ADD
  row's INDEX COLUMN LENGTH. `_add_space` sweeps a single written index,
  while `jax.ops.segment_sum` reaches an index column of 4 on an operand
  the new bounds admit; that axis is gauged by a mutation battery rather
  than by an exhaustive sweep, and admission is not narrowed to it. A
  corruption keyed on the index column's length would be the sixth instance
  of the pattern above.

- **2026-08-06 (pre-release, later the same day): the sixth instance of the
  bounded-sweep pattern closed at the ADD row's INDEX COLUMN, and the bar's
  narrowing decision stopped holding any option value at all.**
  A blinded audit of the previous entry, answered here. The previous entry
  ended by naming the index column as what it had not closed; that is what
  the first item below closes, and it closes it because naming a residual
  turned out not to be the same as bounding it.
  **What changed — the one item that MOVES VERDICTS.** A census of `len(ks)`
  at `_scatter_add_plan` across the whole suite, taken by instrumentation and
  identical at `e35de13` and at the previous entry's head, reaches
  `{1, 2, 3, 4, 6, 254, 255}` — 5 is absent, and so is everything in 7..253.
  One line, line-neutral, on operand shape `(2,)` which the shape bounds
  explicitly admit —
  `groups[k * rowsz + t].append((j if len(ks) - 5 else 0) * rowsz + t)` —
  turns a `violated-witness` into `discharged` at |ks| = 5 and nowhere else,
  a MISSED violation with the full suite green; keyed on 7 instead of 5 it
  does the same thing inside the 7..253 hole. So the column is bounded the
  way the shape was: the ADMITTED column space is the union of three
  EXHAUSTIVELY swept families and nothing else — one index over every gauged
  shape; every column of `range(n)` to the power of the length, for lengths
  up to 6, on a RANK-1 operand; and the single-element operand at every
  length up to 255, where every index is forced to 0 and the length is the
  only free parameter. `tests/test_scatter_gauge_jax.py` pins those bounds
  EQUAL to the source's in both directions, as it already does for rank, dim
  and size. Measured with the corruptions applied on top: keyed on 5 the
  branch is still reachable and the sweep catches it (2 RED); keyed on 7 the
  branch is UNREACHABLE — the obligation comes back `unknown` — where at the
  previous head it came back `discharged`.
  **Which prior verdicts are retroactively invalid: none, and this is not an
  unsoundness fix.** Nothing that was VERIFIED becomes REFUTED or vice versa.
  As with the shape bounds, obligations past the column bound move from
  ANSWERED to UNDECIDED, which is a verdict move to UNKNOWN and is logged as
  one. It costs REFUTATIONS as well as discharges. Blast radius **across the
  PYTEST-DRIVEN TREE** is ZERO, measured rather than assumed — and the scope
  is stated because it is the instrument's, not the repository's: the census
  above is the evidence, and a RE-census after the guard finds nothing
  reaching the row outside the gauged column space — 821 distinct
  `(|ks|, distinct, shape)` keys over 2675 calls, zero of them outside. What
  the instrument could not see is anything pytest does not run. `corpus/`'s
  scripts are driven by hand; `corpus/run_census.py` in particular imports
  only `stelling.census` and `stelling._jax_compat` and walks jaxprs to
  classify primitives, so it never reaches the accumulate row at all — checked
  rather than assumed, and stated either way.
  **AND THE PATTERN MOST LIKELY TO MEET THE BOUND IS NAMED**, because a
  narrowing whose cost is given only in the abstract is given in the abstract.
  `tests/test_scatter_gauge_jax.py`'s own header lists "a small normal-matrix
  assembly in the segment_sum style" among the programs this file exists to
  gauge, and that assembly's natural spelling — `jax.ops.segment_sum` over
  per-point (2, 2) blocks, i.e. a RANK-3 operand with an index column longer
  than one — is exactly what now declines. Measured on that file's own
  `m-assembly` fixture at its own declared shapes: the slicing face refuses
  with *"'scatter-add' index column of 3 element(s) on operand (2, 2, 2) is
  outside the GAUGED accumulate column space"*, while the SAME accumulation
  flattened to a rank-1 operand is admitted. It costs no test here because the
  in-tree `m-assembly` cases are settled by the interval transfer and never
  reach the row; a downstream harness that escalates one gets UNKNOWN. The support itself is no longer
  `{1, 2, 3, 4, 6, 254, 255}` and is not claimed to be: the new sweep drives
  the row at every length in the gauged space, which is what "gauged" now
  means. Outside the gauge file the row is still reached only at
  |ks| ∈ {1, 2, 3, 4}. `|ks| = 256` reaches the plan and DECLINES, which is
  the guard firing.
  **What to re-run:** any recorded verdict over a harness whose
  `jax.ops.segment_sum` (or `x.at[column].add(...)`) accumulates MORE THAN
  ONE update index onto an operand of rank 2 or 3, or more than six onto a
  multi-element operand, AND whose obligations were decided by the SOLVER
  rather than by intervals. Those verdicts are not wrong; this build returns
  UNKNOWN where they returned an answer. The interval TRANSFER is untouched,
  so verdicts intervals settled are unaffected.
  **Affected versions:** 0.1.0 pre-release only; branch-only, nothing
  released. The corruption is reachable on every build in this repository's
  history through the previous entry's head.
  **Three further items this pass, none of them verdict-moving:**
  **(1)** The seventh channel's repair was measured INCOMPLETE. The previous
  entry closed "a conjunct on the VALUE of a whitelisted option key" with two
  site mechanisms plus a budget sweep — but the one function permitted to see
  option values WAS the narrowing decision, so both site mechanisms were
  satisfied by construction and the whole defence rested on the sweep, which
  is a sample and reaches only the two of four whitelisted keys a caller can
  set. Three line-neutral corruptions, each with the full suite BYTE-IDENTICAL
  to unmutated in both columns: a module global stashed in
  `_evidence_is_about` and read in `_bar_scope` keyed on `"30000"`
  (`solver_timeout_ms=30000` → VERIFIED with no withheld note, where 20000,
  29999, 30001 and 31337 all return UNKNOWN — and the IDENTICAL mutation
  keyed on `"31337"`, which the sweep samples, is 2 RED, so the only
  difference is the constant); and conjuncts on the VALUES of `smt2_sha256`
  and `slice_sha256`, keys the sweep never varies at all. **Not answered by
  widening the sweep.** The narrowing decision now consumes a BOOLEAN and
  binds no recorded value, so no conjunct on any value is expressible there
  at any value; the value zone is one four-line function whose two halves
  cannot AIM (`_evidence_options` is handed no query, `_reproduced_evidence`
  is handed no record, so neither can compute the mapping a false narrowing
  would have to produce); and the reproduction is built by
  `Script.stamp_options`, the same derivation the record is built from,
  pinned by substitution rather than by two readings agreeing.
  *(THE TWO CLAIMS IN THAT SENTENCE ARE BOTH CORRECTED BY THE NEXT ENTRY, and
  the wording is left standing because a log that edits itself is not one. The
  four lines call a FIFTH function, which was in no enumeration anywhere, and
  a signature says what a function is HANDED rather than what it can REACH;
  and "pinned by substitution" constrains the substituted function's behaviour
  not at all. Both were measured live.)* What is left
  is pinned TOTALLY over the source rather than sampled over values: no
  string OR numeric literal outside the read ledger's own attribute names, no
  comparison against a literal, no `global`/`globals()` smuggling. Six
  mutants are RED against it, including the derived-quantity dodge
  (`not (budget % 30000)`, which has no comparison in it at all and which the
  sweep does not reach). Its stated limit: it constrains what may be WRITTEN
  on that path, not every predicate Python can express — a discriminator
  spelled as a method call on a value is not matched — which is why the
  budget sweep is kept as corroboration. The sentence that sweep carried,
  "an equality on any round or memorable number is hit", was FALSE and is
  corrected rather than answered by adding values to the list.
  **(2)** The previous entry's item (2) said one pass over `records` closed
  the misattributing note. It closed it for the ONE-SHOT shape only. Measured
  on a SCATTER-FREE query with a `records` that is empty on its first pass and
  real afterwards: an honest VERIFIED becomes UNKNOWN carrying "…the
  propagated interval straddling the asserted bound" — verbatim the defect
  that entry recorded as closed. One pass at the top IS choosing pass 1, and
  the comment claiming it worked "rather than by choosing which pass wins"
  was wrong about its own mechanism. The shape is now REFUSED rather than
  absorbed: the ledger is a separate field and an independent witness that
  solvers ran, so an escalation whose ledger carries work and whose `records`
  came back empty raises instead of assembling. What that does NOT reach — a
  first pass yielding a non-empty STRICT SUBSET — is stated at the gate, with
  the reason the stronger check is not taken.
  **(3)** Two comments in `make_solver_verdict` contradicted each other: the
  top called the bar-domain ordering "a second, now-REDUNDANT mechanism", the
  bottom said "THE ORDER IS LOAD-BEARING". Measured: moving the domain read
  below `by_index` is 0 RED. The bottom sentence was the false one and is
  corrected — which is exactly the shape the top one names, an unpinned guard
  whose comment claims to be load-bearing. Also: `solvers.py` cited a test
  the previous pass had renamed away, the only dangling `::test_*` reference
  in `src/`; it is repointed, and
  `test_every_test_cited_in_core_prose_still_exists` makes the next one red.
  And a stale false sentence inside the test whose stated purpose is to stop
  census drift — "(five fixtures over four rules)", forty-six lines above an
  assertion requiring six over five — is gone rather than corrected: that
  sentence no longer restates the counts, and the three that remain are read
  out of the file's own source and checked against the derived counts.
  At this pass: 2064 passed, 2 skipped with both solvers, jax and maddening
  installed; 2060 passed, 6 skipped under CI's install set
  (`.[solvers,jax]`, no maddening). Before it: 2055 / 2 and 2051 / 6.
  **What this pass did NOT close, named rather than left implied:** the value
  zone's source pin reads `ast.Compare` nodes and literals, so a discriminator
  spelled as a method call on a recorded value is not matched by it; the
  coherence gate does not see a `records` whose first pass yields a non-empty
  strict subset; and the ADD row's column sweep stops at rank 1, so a
  multi-index column above rank 1 is DECLINED rather than gauged.

- **2026-08-06 (pre-release): the value zone is closed under CALL, and three
  checks that were satisfied by TEXT rather than by code.** A blinded audit of
  the entry above, answered here. **No item in this entry moves a verdict**:
  nothing that was VERIFIED, REFUTED or UNKNOWN changes status, no bound moves,
  no numeric constant changes, and the scatter bar is where it was. What
  changes is what the tree can be corrupted into saying, and what four of its
  own sentences claimed.
  **Which prior verdicts are retroactively invalid: none.** **Affected
  versions:** 0.1.0 pre-release only; branch-only, nothing released.
  **What to re-run:** nothing. The one behavioural change is the WORDING of the
  cause note on an UNKNOWN whose escalation arrived incomplete; no verdict, no
  obligation status and no bar decision moves.
  **(1) THE VALUE ZONE WAS NOT CLOSED UNDER CALL, and the repair that closed
  channel 8 is what opened this.** That repair split the narrowing into a value
  zone of four named functions, pinned their source TOTALLY, and pinned the two
  halves' signatures so that "neither can AIM". It also added a fifth function,
  `_whitelisted`, called once per SIDE of `recorded == reproduced` — and that
  one was in no enumeration anywhere: not in the four names the source pin
  parsed, not in the read ledger's transparent frames (it is handed a plain
  `dict` and reads no attribute, so the ledger never sees it at all), not among
  the five functions the one-place scan reads. Four line-neutral corruptions,
  each MEASURED LIVE at `faefc48` with the full suite byte-identical to
  unmutated (2064 passed / 2 skipped), each turning the bar's fallback into a
  narrowing on a MISPAIRED assembly at exactly one budget while every other
  budget falls back:
  a module-level `list` stashed in `_whitelisted` and returned on the second
  call — no `global` statement, no `globals()`, no literal in either pinned
  function, and NEITHER pinned signature touched; a DEFAULT ARGUMENT on
  `_evidence_reproduces`, which `_fn_body_ast` never walked (`tree.body[0].
  body` is the body and nothing else); a SEVENTH module-level helper called
  from the zone, which the six-name pin does not parse; and a conjunct inside
  `stelling.smt.Script.stamp_options`. Two further spellings the audit did not
  name are also live: a module-level `dict` written in `_whitelisted` and read
  in `_reproduced_evidence`, and a MUTABLE DEFAULT on `_whitelisted`, which
  introduces no module-level name at all.
  **The repair is the CLASS, not the six spellings.** The value zone is no
  longer a tuple of names; it is DERIVED as the transitive closure, over the
  compiled code objects, of every module-level name of `stelling.verdict` that
  `_evidence_reproduces` can reach — and the same for the DECISION from
  `_bar_scope`. Every rule runs over the derived set, so a helper on the path
  is inside the pin the moment it is written; every module-level name either
  closure reads must be enumerated AND immutable (a `frozenset` or a `tuple`
  cannot carry a value from one call to the next, which is the whole of what
  the stash needed); default arguments are forbidden in both closures and the
  signature is scanned by the literal rules; and the closures may import only a
  named list, because a function-level import binds a LOCAL and is invisible to
  the closure walk. All six mutants are RED, each to one of those three general
  rules and none to a rule written for it.
  **(2) "PINNED BY SUBSTITUTION RATHER THAN BY TWO READINGS AGREEING" DOES NOT
  CONSTRAIN THE SUBSTITUTED FUNCTION AT ALL.** `_reproduced_evidence` claimed
  that corrupting `Script.stamp_options` "corrupts EMISSION — which the
  byte-level emission tests hold". It does not: `stamp_options` appends
  `set-logic`/`smt2_sha256`/`slice_sha256` to an ALREADY EMITTED `Script` and
  contributes not one byte to `Script.text`, so the scripts real solvers answer
  about are byte-identical either way. Measured: a conjunct there leaves
  `tests/test_smt_emission.py` and `tests/test_verified_bar.py` both fully
  green, because the substitution test only checks that a DIFFERENT
  `stamp_options` MOVES the answer and never that the honest one is honest. The
  claim is corrected and the honest OUTPUT is now pinned twice — structurally
  (one `return`, no branch, no comparison, no call) and behaviourally (the
  exact tuple, against an expectation re-derived from the script's own TEXT and
  SLICE). One incidental correction with it: 30000 IS reached in this tree —
  `tests/test_solver_acceptance.py` runs at `SolverConfig(timeout_ms=30_000)`,
  spelled with an underscore — so the `stamp_options` conjunct keyed there is
  1 RED. Re-keyed one millisecond away it is 0 RED across the whole suite. The
  budget SWEEP still does not sample it, which is the claim that mattered.
  **(3) `_evidence_budget`'s BOUND HOLDS, AND THE ARGUMENT FOR IT DID NOT.**
  The docstring said: the recorded budget is itself in the compared set, so a
  wrong budget puts a wrong `:timeout` in the reproduction, the equality fails,
  the bar widens. Every step is true and it is an argument about an HONEST
  record — it says a wrong budget disagrees with the budget THIS record names.
  What actually forbids a mispaired narrowing is that the budget cannot reach
  `slice_sha256` AT ALL: measured invariant over twelve budgets spanning
  1..60000 including `True` (which `isinstance(budget, int)` admits), while
  `smt2_sha256` moves at every one of them; and the bar's own neighbour pair
  has EQUAL `smt2_sha256` and DIFFERENT `slice_sha256`. Those two together are
  stronger than any sweep: no budget, sampled or not, can turn one slice's
  reproduction into another slice's record.
  **(4) THE COHERENCE GATE'S RESIDUE VIOLATED THE GATE'S OWN JUSTIFICATION.**
  That justification is that absorbing a degenerate `records` produced "an
  UNKNOWN carrying a WRONG EXPLANATION … worse than silence, because a reader
  believes it". Measured at `faefc48` on a SCATTER-FREE query with two
  solver-decided obligations and a `records` whose first pass is a non-empty
  strict subset: honest VERIFIED, observed UNKNOWN, carrying verbatim "…the
  propagated interval straddling the asserted bound". The residue is
  soundness-harmless (a dropped record leaves its obligation `unknown`, which
  can never mint VERIFIED) and is still NOT refused — the comparison that would
  refuse it also refuses a deliberate probe of a different invariant. It is
  CLASSIFIED instead: the ledger witnesses invoked runs the records do not
  account for, and the note says the outcome did not arrive. Statuses,
  verdicts and bar decisions are unchanged; only the sentence moves.
  **(5) THE CITATION CHECK WAS FALSELY SATISFIABLE, and not only by globs.**
  `f"def {name}(" in body` is a raw substring test over file TEXT. Measured at
  `faefc48`: the cited family renamed away is 1 failed (not vacuous), but the
  family gone with one `# def test_…(` COMMENT left behind is 1 PASSED, the
  exact citation's own `def` commented out is 1 PASSED, and the `def` gone with
  only a string-literal mention left is 1 PASSED. Commenting a test out is how
  a test most often stops existing. Resolved by `ast.parse` + `FunctionDef`
  names, which is equally independent of collection — the reason the docstring
  gave for avoiding collection — and all three rows are now RED.
  **(6) THE CENSUS PROSE CARRIED MORE RESTATEMENTS THAN IT READ.** The drift
  test read three sentences and its docstring said "four places for three
  quantities". A search of the flattened source finds eight, and the five it
  did not read were each 0 RED at `faefc48`, perturbed one at a time: "the
  admission gate drives … of them", "The other … are NOT driven here", "the
  row's … decline sites", `f"says … over …"`, and "`_scatter_set_plan` has …
  `raise _Decline`" in two places. All eight are read now, one of them against
  a DERIVED difference rather than a fourth quantity; the source is flattened
  first, because four of the five were unread partly because they are split
  across two string literals or two comment lines; and the anti-vacuity control
  perturbs ONE capture group of ONE match at a time, where
  `text.replace(right, wrong)` rewrote every occurrence at once and so could
  never see the gap. The residue is named: a NINTH restatement, written later
  and not added to the list, is still not read — a general number-word scan was
  measured at 24 occurrences of which 18 are about something else.
  **(7) TWO RECORD CORRECTIONS, both re-derived.** The previous pass's
  "+9 tests, 0 removed" is false: the collected-id diff `3e107cf..faefc48` is
  **10 added and 1 removed** (`test_a_TWO_FACED_records_cannot_show_the_bar_
  one_thing_and_the_loop_ANOTHER`, whose direct successor is the parametrised
  refusal test, so no coverage is lost — the statement is what was wrong). And
  "exactly three integers moved in `tests/`" does not reproduce: per-file
  multisets of numeric literals across all four commits give **0 removed and
  112 added in `tests/`** (105 `int`, 7 `float`) and **0 removed and 5 added in
  `src/`**. Zero literals changed value anywhere, which is STRONGER than the
  claim it replaces.
  **(8) THE CAPABILITY CLAIM IS SCOPED TO WHAT WAS MEASURED**, and the pattern
  most likely to meet the bound is named — see the entry above, which now says
  "across the PYTEST-DRIVEN TREE", records that `corpus/run_census.py` never
  reaches the row, and states that the gauge file's own `m-assembly` fixture,
  at its own declared shapes, DECLINES through the slicing face while the same
  accumulation flattened to rank 1 is admitted.
  At this pass: 2068 passed, 2 skipped with both solvers, jax and
  maddening installed; 2064 passed, 6 skipped under CI's install set
  (`.[solvers,jax]`, no maddening). Before it: 2064 / 2 and 2060 / 6.
  **What this pass did NOT close, named rather than left implied:** the value
  zone's source pin still reads `ast.Compare` nodes and literals, so a
  discriminator spelled as a METHOD CALL on a recorded value
  (`.startswith(...)`, a hash, a length test) is not matched by it — the
  closure rules above do not reach that either, and the budget sweep is kept as
  corroboration for exactly that reason; the coherence gate still does not
  REFUSE a `records` whose first pass yields a non-empty strict subset, it only
  refuses to misattribute the result; the census drift test still cannot see a
  restatement nobody adds to its list; and the ADD row's column sweep still
  stops at rank 1, so a multi-index column above rank 1 is DECLINED rather than
  gauged.

*(no releases yet)*
