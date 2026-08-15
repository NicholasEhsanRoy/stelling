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
  the closure walk. All six mutants are RED. **The sentence that stood here
  said "each to one of those three general rules and none to a rule written for
  it", and that is false on two counts** — `PREREG_BAR13.md` scored the same
  clause honestly as **P1.c partial**, and the published document did not.
  Re-derived by running each mutant at `9fc44dd` and reading which assertion
  fires: the DEFAULT-ARGUMENT mutant is 1 RED, to the DEFAULTS rule alone
  ("`_evidence_reproduces` now has a DEFAULT ARGUMENT"), which is a FOURTH rule
  added for that mutant and not one of the three; the mutable-default mutant
  dies to the same rule; and the `Script.stamp_options` mutant dies to
  `test_the_stamps_own_derivation_is_the_HONEST_one`, a test that exists only
  because of it. Three of the six die to the general rules; three do not.
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
  multisets of numeric literals **over the same range, `3e107cf..faefc48`**,
  give **0 removed and 112 added in `tests/`** (105 `int`, 7 `float`) and
  **0 removed and 5 added in `src/`**. The range is named here because the
  sentence beside it names one and this one did not, and the figures are
  range-specific: over **`faefc48..9fc44dd`**, this branch's own four commits,
  the same measurement gives **34 added / 0 removed in `tests/`** and
  **4 added / 2 removed in `src/`** — both removals the literal `0`, from a
  deleted line rather than from a constant that moved. "No `src/` constant
  changed value" is the claim that survives at every range, and it is
  CONFIRMED tree-wide rather than per-file: over `faefc48..HEAD`, 192
  module-level assignments in `src/`, 31 numeric-bearing, **0 changed, 0 added,
  0 removed**.
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
  corroboration for exactly that reason; **and the axis the method call is only
  one instance of: the zone may spell no constant and is HANDED four, so any
  predicate over a recorded value whose constants come from
  `_EVIDENCE_*_KEYS` is invisible to every rule in the pass — closing the CALL
  axis is what left the PREDICATE axis open, and it was open at the meeting
  point itself**; the DECISION's loop filters `rule != "literal"`, so a string
  constant inside any call there matches nothing; the coherence gate still does
  not REFUSE a `records` whose first pass yields a non-empty strict subset, it
  only refuses to misattribute the result; the census drift test still cannot
  see a restatement nobody adds to its list; and the ADD row's column sweep
  still stops at rank 1, so a multi-index column above rank 1 is DECLINED
  rather than gauged.

- **2026-08-06 (pre-release): the value zone was closed under CALL and left
  open under PREDICATE, because it is HANDED its constants.** A blinded audit
  of the entry above, answered here. **No item in this entry moves a verdict**:
  nothing that was VERIFIED, REFUTED or UNKNOWN changes status, no bound moves,
  no numeric constant changes in `src/` (192 module-level assignments, 31
  numeric-bearing, 0 changed / 0 added / 0 removed over `faefc48..HEAD`), and
  the scatter bar is where it was. What changes is what the tree can be
  corrupted into saying, and what six of its own sentences claimed.
  **Which prior verdicts are retroactively invalid: none.** **Affected
  versions:** 0.1.0 pre-release only; branch-only, nothing released.
  **What to re-run:** nothing. There is no behavioural change at all — the one
  `src/` edit hoists a fallback message to a module constant, byte-identical in
  what it renders.
  **(1) THE PREDICATE AXIS WAS WIDE OPEN, AT THE MEETING POINT ITSELF, AND THE
  REASON IS THE SHAPE OF THE PIN.** The source pin forbids the value zone to
  SPELL a string, and then the module hands it `_EVIDENCE_OPTION_KEYS`,
  `_EVIDENCE_BUDGET_KEYS` and `_EVIDENCE_DERIVED_KEYS` as enumerated immutable
  constants — so every string a conjunct needs, for exactly the keys that
  matter, is already in scope under a permitted name. The literal rule is
  near-vacuous for those keys. Six line-neutral corruptions, each MEASURED LIVE
  at `faefc48` (2064 passed / 2 skipped and 2060 / 6, byte-identical to
  unmutated in both columns) and each 0 RED at `9fc44dd`: three lines inside
  `_evidence_reproduces` itself — `if set(_EVIDENCE_BUDGET_KEYS) <=
  set(recorded): return True` — with no literal, no comparison against one, no
  module-level mutable, no helper, no default argument, no import, no `global`,
  no smuggler, and no method call on a recorded value, so it is not the residue
  the entry above disclosed; the same predicate in a literal-free NESTED `def`;
  `_whitelisted.__kwdefaults__ = out`, a mutable carrier on a zone FUNCTION,
  which the immutability rule never reached because that branch runs only for
  names the module does not define; the same carrier USED across the two
  `_whitelisted` calls; and two in the DECISION, where the loop filters
  `rule != "literal"` so a string constant inside any call matches nothing —
  `str(closed).count("stelling_backdoor")`, and a per-stamp variant on the two
  budget spellings. Against the mispaired pair the bar exists for, with a stamp
  forging a second budget spelling, four of the six take `('scatter',)` to
  `()`: the bar lifted. Unmutated, forged or honest, all four rows stay
  `('scatter',)`.
  **The repair is five rules, and none is written for a spelling.** A CONSTANT
  READER LEDGER — each enumerated constant is read by exactly one function,
  asserted in both directions, the way `_ALLOWED_READS` ledgers record
  ATTRIBUTES — after which a predicate at the meeting point has no constant to
  key on. A SHAPE PIN on `_evidence_reproduces`, the one function where a
  recorded value and the re-derivation are both in scope: no branch, no loop,
  no nested definition, one `return`, of `bool(...) and ... == ...`. Those two
  are INDEPENDENT, measured by disabling each and re-running: the first mutant
  is RED to either alone. The decision's literal exemption is narrowed from a
  rule NAME to a POSITION — message text, in an f-string, a `+`, or a
  `"sep".join(...)` — and the decision may now spell no number at all, which it
  never did. A `call-literal`/`attr-literal` rule catches a literal handed to a
  method on a value, which is most of the method-call residue the entry above
  disclosed and left open. And a `dynamic` rule for `__import__`/`eval`/`exec`/
  `compile`, which bind no `ast.Import` node and so were invisible to the
  import allow-list. All six new mutants and all six published ones are RED.
  **(2) THE IMMUTABILITY RULE WAS FALSE AS WRITTEN, IN TWO PLACES.** *"Every
  module-level name the closure reads must be enumerated AND immutable"* never
  applied the immutability half to the zone's own function objects, which are
  mutable; and `_IMMUTABLE` includes `tuple`, checked SHALLOW, so a tuple
  containing a list would pass (no exploitable instance today — all four
  enumerated constants are frozensets or tuples of `str`, which is why the
  check is deepened rather than a constant changed). Both are closed, with a
  source rule (no assignment to an attribute anywhere in either closure) and a
  runtime one (no `__defaults__`, `__kwdefaults__` or `__dict__` on any
  function in the closure) kept together because neither reaches the other's
  case. A zone member that is not a plain FUNCTION — a class, whose methods the
  `__code__` walk silently skips — is an offence now rather than a hole, and
  where the walk stops is named in the docstring: attribute/method dispatch, a
  decorator's wrapper, and objects with no `__code__`.
  **(3) THE STRONGEST STATEMENT ABOUT `_evidence_budget` IS ONE LINE AND
  NEITHER TEST MADE IT.** `inspect.signature(slice_fingerprint)` is
  `(sl) -> 'str'`: the budget is not an argument, so there is no value of it to
  sample. Added. The pre-registered exhaustive sweep it replaced was also not
  expensive, which was the reason given for substituting twelve points:
  `1..60000` costs **9.5 s at load average 6.00**, and gives distinct
  `slice_sha256` 1, distinct `smt2_sha256` 60000, empty reproductions 0,
  reproductions equal to the neighbour's record 0. The substitution's standing
  is restated accurately — stronger in the generality of its argument, WEAKER
  in the sample supporting its premise — and its neighbour-pair half is kept
  because the sweep did not have it. The `True` row is labelled as what it is:
  `isinstance(budget, int)` admits it and `emit(sl, "z3", True)` emits
  `(set-option :timeout True)`, a script no solver accepts, but
  `_evidence_budget` CANNOT return a bool (`int(text)` never yields one —
  measured), so reaching it needs `int` itself corrupted.
  **(4) THE SAME GREP BLINDNESS, THREE LINES ABOVE THE CORRECTION.**
  `_CALLER_BUDGETS`' comment corrected "an equality on any round number is hit"
  by naming 30000 as absent from the tuple — and in the same sentence listed
  "2000, 15000, 25000, 50000 or 60000", read as values the suite does not
  drive. **Four of the six are live solver budgets**: 2000 at about twenty
  sites, 15000 at `tests/test_array_emission.py:1391`, and 30000 and 60000
  spelled `30_000`/`60_000` so that `grep -rn '60000'` finds only prose. The
  claim is derived now, off the AST, from BUDGET POSITIONS rather than from
  every number — 50000 appears as fixture arithmetic and is not one. Measured
  tree-wide: **542 numeric instances in `src/` and `tests/` are spelled in a
  form a digit-grep for their own value misses**, over 136 distinct forms — 331
  COMPUTED, 157 EXPONENT, 53 underscore, 1 other, and zero hex, octal or
  binary. Underscore is the small part.
  **(5) THE NINTH RESTATEMENT, FOUND.** The census entry above named "a NINTH
  restatement, written later and not added to the list" as a residue and left
  it there. It is `# The undriven six, each with its reason:` in
  `tests/test_scatter_gauge_jax.py`, restating `sites - rules` inside the file
  the checker parses; perturbed to "seven" at `9fc44dd` it is 0 RED with the
  whole gauge file green. It is read now. Two records with it: `_flat`'s "four
  of the five sentences … are SPLIT" is **three**, measured by running every
  pattern against the raw source as well as the flattened one (the flattener
  stays — 13 number-words with it against 10, and removing it hard-fails); and
  the block comment restated the count of places as "eight" **fourteen lines
  above** asserting that "the count of places is no longer restated anywhere",
  both shipped in one pass. The number is elided and the claim kept.
  **(6) FOUR RESIDUES, THREE CLOSED AND ONE DISCLOSED.** The citation
  resolver's semantic is "a `FunctionDef` of that name exists in the AST",
  which is strictly weaker than "the cited test runs": two of the seven shapes
  where that differs are closed — nested inside another function, and inside a
  non-`Test*` class, which was one of the branch's own ACCEPTED rows and which
  pytest does not collect because `pyproject.toml` sets no `python_classes` —
  and the other five are rows in the anti-vacuity table now, so the gap is
  measured rather than described (all 13 in-tree citations resolve at module
  level, checked before the change). `docs/gauge-coverage.md`'s `m-assembly`
  claim, which nothing reddened, gets a test that pins both halves of the
  asymmetry. The `scratchpad` `WITHHELD` entry says what it actually does: the
  sdist `include` list is what keeps it out, and the entry exempts the entire
  `scratchpad/` SUBTREE from the untracked-file check. And a number-word scan
  over `*.md` is NOT added: `SOUNDNESS.md`'s own restatements of these counts
  are outside the census checker by construction, and that is named rather than
  left to look covered.
  At this pass: 2070 passed, 2 skipped with both solvers, jax and maddening
  installed; 2066 passed, 6 skipped under CI's install set
  (`.[solvers,jax]`, no maddening). Before it: 2068 / 2 and 2064 / 6.
  **What this pass did NOT close, named rather than left implied:** a predicate
  in the value zone that needs NO constant at all is still not matched —
  `len(recorded) > len(reproduced)` has no `Constant` in either operand, and
  `sorted(_EVIDENCE_OPTION_KEYS).pop()` drops a key by position rather than by
  name; the meeting point's shape pin means neither can be written where both
  sides are in scope, and a constant-free predicate cannot AIM at a chosen
  record, but neither of those is a rule that catches it. The decision still
  permits a string literal handed to a call on a bare NAME, because the honest
  fallback builds its message that way and the callee's own body is inside the
  scanned closure. The citation resolver still answers "defined" for five
  shapes pytest never runs. The census drift test still reads a LIST of
  patterns, so a tenth restatement nobody adds to it is a tenth restatement
  nobody reads — the ninth is the demonstration that this is not hypothetical.
  And the coherence gate and the ADD row's column bound are where the entry
  above left them.
- **2026-08-07 (pre-release): the cvc5 WHEEL transport now refuses a
  crashed run — F4's rule, on the transport that never received it.**
  `_make_run_cvc5_binary` has refused a crashed run since F4;
  `_run_cvc5_wheel` did not, and the shape is transport-specific: the
  child driver prints `answer` and THEN walks the model through native
  `getValue`, so a death in there leaves `answer sat` on stdout with no
  terminator, and the parent harvested the partial model as a verdict.
  **Measured, real child, real SIGKILL, bisected:** the boundary is the
  8192-byte pipe buffer, not the model size — 493 value lines (8186
  bytes written) leave 0 bytes through and were correctly caught as a
  protocol violation; 494 lines (8203 written) leave 8202 through and
  returned **sat with 494 values from a dead process**. At 4000 terms:
  3572 lines through, 3570 values harvested. And 8192 is a default, not
  a floor — under `PYTHONUNBUFFERED=1` (standard in Docker images and
  CI) the threshold is **zero**: a 2-value model puts all 51 bytes
  through. Reachability is ordinary: `smt.py` emits one
  `(declare-const … Real)` per input ELEMENT, so a single (32,32) array
  is 1024 consts ≈ 17360 bytes of driver stdout, and an OOM kill is a
  SIGKILL that needs no cvc5 bug at all. **Direction of movement: toward
  UNKNOWN, only.** A crashed run that produced `sat` now produces
  `failed` → UNKNOWN; the same for `unsat` and `unknown` from a crashed
  child. Nothing moves toward VERIFIED or REFUTED, and no healthy run
  moves at all — five healthy shapes (empty model, values, opaque,
  mixed, and a model value whose own text is `end`) are byte-identical
  before and after, as are the timeout, `OSError` and internal-`error`
  paths, the whole binary transport, and z3. **A deliberate tightening
  rides along:** a child with a complete protocol that exits **nonzero**
  returned its answer with its model before (measured: clean `answer
  sat` + terminator + exit 1 → sat) and is `failed` → UNKNOWN now,
  matching the binary transport's F4 policy that a nonzero exit is not a
  transport this layer will discharge OR refute on. **What the harm
  actually was**, driven end-to-end rather than reasoned: a truncated
  model does NOT yield an unreproducible witness — every refutation
  routes through `_require_valid_refutation` → `witness_is_valid` (box
  membership AND exact-rational violation), so a truncated model either
  raises `EmissionInfidelityError` loudly (measured, when the completed
  point satisfied the predicate) or yields a witness that genuinely does
  reproduce (measured: REFUTED, replay passed). The harm is that **a
  crashed run silently became an accepted answer**, with the values the
  child never lived to write supplied by `_complete_values` from the
  declared box — the fill was disclosed as a don't-care note and reached
  the render, but the death was disclosed nowhere. Same pass, three
  further refusals in the same class, each measured accepted before:
  the terminator was a token-prefix test, so a child writing `end of the
  resource limit` raw to fd 1 (the shape native C++ output takes,
  bypassing Python's buffer) and exiting 0 defeated **both tells at
  once** — it is now required to be the last line of stdout and to carry
  the driver's own count of the model lines it wrote; values written
  AFTER the terminator, and a truncated trailing line silently dropped,
  fall to the same rule; and a second `answer` line, which used to be
  read as the new answer while KEEPING the value harvested under the
  first, is now a protocol violation. Cry-wolf cost measured at zero.
  Every construction is a permanent regression test
  (`tests/test_solver_audit_findings.py`, the `f4wheel` block).
- **2026-08-08 (pre-release): the cvc5 wheel driver and its parent
  disagreed about what a LINE is, and the payload could forge the
  terminator — defeating both tells at once. Reachable in PRINCIPLE, not
  through any script this tool emits.** Direction: **toward UNKNOWN,
  only.** The driver sanitised model text with `replace("\n", " ")`; the
  parent read with `str.splitlines()`, which breaks on **ten** characters
  and not one (measured off Python, not recalled: U+000A U+000B U+000C
  U+000D U+001C U+001D U+001E U+0085 U+2028 U+2029). A value carrying any
  of the other nine was ONE line to the writer and TWO to the reader, so
  the payload supplied the reader's LAST line — an `end <n>` whose count
  matched what the reader had parsed — while the child was truncated
  mid-model-walk. The terminator exists precisely to catch a death that
  exits zero, so with exit 0 both tells go blind together.
  **Reproduced end to end, real cvc5 1.3.4, real driver, real SIGKILL,
  the parent reading the child's whole flushed prefix (53311 bytes, cut
  mid-write by the pipe boundary — deterministic, 15/15 trials): last
  line `end 3800`, `_run_cvc5_wheel` reporting *terminator present*, and
  on exit 0 returning `sat` with **3800 values harvested from a
  corpse**.** The exit code is the one constructed ingredient, and it is
  constructed exactly as this log's `end of the resource limit` shape
  already was.
  **REACHABILITY — AND THE FIRST READING OF IT WAS FALSE, in the
  understating direction. Corrected 2026-08-08; the correction is recorded
  here because the "incompleteness, not incident" framing rests on this
  paragraph and on nothing else.** The sentence that stood here said cvc5
  escapes every separator inside a model VALUE, so the value channel was
  already closed by cvc5's own printer and only the NAME channel was ever
  open. Half of that is true and half is false.
  TRUE: inside a String LITERAL cvc5 escapes all ten
  (`"a\u{b}b"` — `scratchpad/probe_cvc5_separators.py`, driven for each).
  FALSE: a **quoted symbol reaches the VALUE field verbatim** whenever the
  value's SORT or CONSTRUCTOR was declared as one — no string literal
  anywhere in it. Driven through the driver's own route
  (`cvc5.InputParser`, SMT_LIB_2_6 string input, `sm.getDeclaredTerms()`,
  `solver.getValue`), real cvc5 1.3.4, `scratchpad/probe_cvc5_value_channel.py`:

      (declare-sort |S<VT>end 1| 0) (declare-const c |S<VT>end 1|)
        value = '(as |@_S\x0bend 1__0| |S\x0bend 1|)'
      (declare-datatypes ((D 0)) (((|c<VT>end 1|))))  (declare-const d D)
        value = '|c\x0bend 1|'
      (declare-const a (Array Int |S<VT>z|))
        value = '((as const (Array Int |S\x0bz|)) (as |@_S\x0bz__0| |S\x0bz|))'
      CONTROL, the same sort named in ASCII: value = '(as @S_0 S)' — none

  and the value channel alone was enough to harvest a corpse. Driven at
  `0ad22bb` with the base driver and the base parent, same probe with
  `--corpse`: the real child writes
  `version …\nanswer sat\nopaque c (as |@_S\x0bend 1__0| |S\x0bend 1|)\nend 1\n`,
  and its **57-byte prefix** — a mid-write truncation, with **no terminator
  record written at all** — handed to the real parent with exit 0 returns
  **`sat`**, because the reader's last line is the forged `end 1` sitting
  inside the value text. The 70-byte prefix does it again off the second
  quoted symbol in the same value. The branch's own probe
  (`scratchpad/probe_cvc5_sorts.py`) missed this by naming its datatype
  constructor in ASCII and poisoning a String SELECTOR value — a string
  literal, the one place cvc5 really does escape.
  **What survives from the original sentence, re-measured:** stelling names
  its own consts `x{k}`/`x{k}_{i}` (`obligation.py`) and `smt.py` puts nothing
  but `Real` and `Bool` into SMT-LIB. **THE CITATION HERE WAS WRONG AND THE
  CONCLUSION WAS NOT.** It read "`git grep -n 'declare-' src/stelling/smt.py`
  — `declare-const … Real`, `… Bool`", and that command returns exactly ONE
  line: `smt.py:490: lines.append(f"(declare-const {inp.name} Real)")`. There
  is no `declare-const … Bool` to find. `Bool` reaches the script through
  `define-fun`, at `smt.py:504-510` — `sort = "Bool" if out.aval.dtype ==
  "bool" else "Real"`, emitted as `(define-fun t{id} () {sort} …)`. *(The
  range first written here was `503-509`, which starts on the closing line of
  a docstring and stops one line SHORT of `510` — one of the two `{sort}`
  emitters its own next sentence names. Re-derived: `504` is the `sort =`
  assignment and `506`/`510` are the two emitters, so `504-510` is the block
  that carries `Bool`.)* So the
  two commands that carry the claim are `git grep -n 'declare-'
  src/stelling/smt.py` (**one** line, `Real`, and therefore no `declare-sort`
  and no `declare-datatypes` either) and `git grep -n 'lines.append(f"(define-
  fun' src/stelling/smt.py` (**three** lines — 506 and 510 emit `{sort}`, which
  line 504 sets to `Bool` or `Real` and to nothing else; 659 emits `Real`
  literally). A bare `git grep -n 'define-fun' src/stelling/smt.py` returns
  **eight**, five of them prose in docstrings, which is why the emitting form
  is the one quoted. Sorts: `Real` and `Bool` and nothing else — checked the
  other way too, `git grep -nE '"(Int|String|BitVec|Array|RoundingMode)"'
  src/stelling/smt.py` returns nothing. So no script this tool emits carries a quoted symbol
  into EITHER channel, and the defect stays unreachable through stelling's
  own emissions. What changes is the size of the guard's margin: it rested
  on ONE coincidence stelling does not own, not two, and the value channel
  it was said to be closed against was open the whole time.
  `(exit)` and `:rlimit` were driven against the real driver as candidate
  truncation routes and neither terminates it.
  **Fixed on both sides, because the measurement says neither alone is
  enough.** The WRITER now passes **printable ASCII only**, in every field
  — name, value, version, error text — escaping the rest as `\u{…}`. A
  whitelist and not a wider blacklist, because `splitlines()`'s set is not
  ours to freeze and the io layer's translations are not either. That half
  is **load-bearing and cannot be moved downstream**: the parent captures
  with `text=True`, so universal-newline decoding turns a `\r` into a real
  `\n` **before the parent sees the string** (measured — one record
  written, two records seen, under either splitter), and no reader-side
  rule can see it. The READER now splits on `"\n"`, `print`'s own
  boundary, so an unsanitised record stays ONE record and a payload
  holding an `end <n>` is REFUSED by the terminator check instead of read
  as it; a byte outside the protocol's alphabet is a protocol violation
  rather than something to interpret. **How far that reader-side backstop
  reaches, measured rather than asserted** (`scratchpad/probe_cvc5_backstop.py`,
  a real stale child writing real bytes, no terminator record of its own):
  it refuses **8 of the 10** separators. `\n` is excluded from it by
  construction — it is the protocol's own record boundary, so a writer that
  leaves one inside a field has written two records and there is nothing on
  this side to detect. `\r` is a genuine hole and is the reason the writer
  half is load-bearing: `text=True` has already turned it into a `\n` before
  the check runs, so a stale driver writing `opaque x1 j\rend 2\r` and no
  terminator still returns `sat` with a model **at the fixed tip** — and,
  measured since, `unsat` if that is what the corpse's `answer` line said,
  which is the DISCHARGE direction and the one with no downstream backstop.
  So the driver docstring's *"a mismatch degrades every run to UNKNOWN"* is
  true for this class **through the writer**, and only for 8 of 10 through the
  reader; the earlier wording credited the reader with all of it.
  **THE REPAIR WAS DECLINED ON A FALSIFIER NARROWER THAN THE CONCLUSION IT
  CARRIED, and the decline stands on a different reason now.** What was
  written here and in `solvers.py` was that `text=False` plus an explicit
  decode "also refuses every healthy run whose child applies a `\r\n` newline
  translation" — asserted over the whole repair class from ONE arm of it, the
  raw decode. The nearest rival, `bytes.decode().replace("\r\n", "\n")`, was
  not measured. It is now (`probe_cvc5_backstop.py` parts B/C/D, three readers
  over the same real children): identical to the shipped reader on a healthy
  POSIX child AND on a healthy Windows `\r\n` child — same answer, same values
  — `failed` on the stale `\r` child under an LF body and a CRLF body alike,
  **9 of 10 separators refused instead of 8**, no platform coupling, and
  `failed` rather than an UNCAUGHT `UnicodeDecodeError` when a child writes
  invalid UTF-8. It DOMINATES the shipped reader on every case measured. Its
  one measured cry-wolf case is a healthy child reconfigured to bare-CR line
  endings, which no platform's `print` default produces and `_cvc5_driver`
  never sets. **It is not landed here, and the reason is evidence cost, not
  behaviour** — saying otherwise would repeat the defect being corrected.
  Measured: applying it makes **16 TESTS** fail — 16 is the durable figure and
  its unit is tests; the rest of that run was 2451 passed and 2 skipped of
  2469 collected, a record of this commit that will move with the next added
  test. All 16 are in `tests/test_solver_audit_findings.py` and all 16 are the
  same `AttributeError: 'bytes' object has no attribute 'encode'` raised
  from `Popen._communicate`'s `self._input = self._input.encode(...)` in the
  standard library's `subprocess` (`subprocess.py:2172` on CPython 3.12.3 —
  the LINE NUMBER IS A PROPERTY OF THE INTERPRETER, not of this repository:
  measured 2026-08-09, 2172 is that statement on CPython 3.12.3 and 3.11.15;
  on CPython 3.10.20, which `requires-python = ">=3.10"` admits, the file is
  2122 lines long, so 2172 is 50 lines PAST END OF FILE and the statement is
  at `:2078`; cite the symbol, not the line), because `input=` must become
  bytes when `text=` goes
  and six files shim `subprocess.run` with a `str` `CompletedProcess` —
  including `fuzz_transport.py`, `repro_forgery.py`, `repro_real_kill.py` and
  `probe_cvc5_value_channel.py`, the artefacts behind figures quoted in this
  very entry. And `scratchpad/pin/corpus_pin.py`, the per-obligation
  instrument, says in its own docstring that it "has no solver escalation …
  so nothing here scores `solvers.py`", so the zero this repository requires
  of a behavioural change cannot be produced by the instrument that exists.
  **The hole is live, in both directions, and the repair that closes nine of
  ten is measured, dominant and unlanded.**
  **SUPERSEDED 2026-08-09 — READ THE ENTRY DATED 2026-08-09 BEFORE ANY OF
  THIS PARAGRAPH.** The hole is CLOSED and arm (c) IS LANDED. Everything
  above about the defect and the three arms still holds and was re-driven
  before the repair went in; what no longer holds is the disposition —
  "unlanded", "not taken in this tree", and the evidence cost that was the
  reason for it. The instrument this paragraph says cannot exist was built
  (`scratchpad/crlf/corpus_solver.py`), and the tests were paid — **the 16
  this paragraph names, and 15 more.** The 16 is this paragraph's own figure
  and is the cost of a TOLERANT decode; the full bill under a strict one is
  **31**, the extra 15 being the `str`-shim call sites. Both figures are the
  2026-08-09 entry's own, recorded when the change landed and not re-measured
  in the 2026-08-09 revision that added this sentence; what is fixed here is
  only that they are now BOTH visible at the point of reference, a hundred
  lines before the entry that states them. Saying "the 16" alone here would
  let a back-reference
  understate what landing cost, which is the same shape as the rotting count
  this paragraph is about. A decline that has been reversed reads as a live
  decline if nothing beside it says otherwise, which is that shape again.
  **A third refusal came from the fuzzer, on
  the fix rather than into it:** a record is `text + "\n"`, so a final
  record whose newline never got out is one the child did not finish
  writing — `…\nend 4`, the newline cut and nothing else, read as a
  present terminator with a matching count and became a definite answer on
  exit 0. 86 of 86 residual counterexamples over the same 200 000 (20 000
  per seed × seeds 1–10) had that shape and no other — re-driven at
  `bf905b9` by reinstating exactly the one line under test, `terminated`
  forced true, and running the shipped fuzzer over it: 86, all of them a
  final record with its newline cut. `splitlines()` accepted it
  identically, so it is the same disagreement at the other end of the
  record. **One delimiter down,
  same class:** `value`/`opaque` lines are read with `split(maxsplit=2)`,
  so a space inside a NAME shifted the value into the name's field; names
  are now whitespace-free tokens. **Cry-wolf cost measured at zero** —
  every healthy shape the `f4wheel` block pins is byte-identical, printable
  ASCII passes the whitelist untouched, `opaque x0 (root 2)` keeps its
  spaces, and the whole suite is green on BOTH jax series with
  `--collect-only` ids byte-identical between them, at this branch and at
  its merge-base alike. **A BARE PASS COUNT STOOD HERE AND HAS BEEN
  REMOVED, not bumped.** It read `2433 passed / 2 skipped`, which was true
  at `92e5ab4` and false three times in the two days after — the tree has
  been 2454 and is 2459 as of `bf905b9`. What the number was standing in
  for is a comparison, and the comparison is what is now written: the two
  series equal EACH OTHER on the same commit, and the branch's own
  `--collect-only` delta against its merge-base is **18 ids added and 10 ids
  removed**, the 10 being the retired params of the de-vacuified
  `test_f4wheel2_sweep_the_reproducer_scan_errs_toward_crying_wolf` and
  nothing else. **THE SENTENCE HERE SAID "AND NOTHING REMOVED" AND THAT WAS
  FALSE** — false in the same entry that says "10 parametrised ids retired,
  18 added" sixty-odd lines below, so the branch contradicted itself inside
  one entry while replacing a rotting count with a durable comparison. The
  measurement, and the one that would have refuted it: `pytest --collect-only
  -q | grep :: | sort` in a worktree at each commit, then `comm -13` and
  `comm -23` over the two files — 2461 ids at `bf905b9`, 2469 here, `comm -13`
  gives 18 and `comm -23` gives 10, and every one of the 10 is a
  `…crying_wolf[…]` param. A constant cannot say "nothing moved" on a tree
  that gains tests daily; neither can a delta that is only counted in one
  direction. Two measurements on one commit can, and they stay true next
  week — provided both directions of the delta are read, which is the defect
  this paragraph is now a record of. **Property fuzzer** (line-boundary AND mid-write truncation,
  exit code drawn independently of truncation, ground truth taken from what
  the WRITER emitted): **0 counterexamples at 20 000 examples per seed
  across seeds 1–10 — 200 000 in total, not 200 000 each; the same
  generator at `0ad22bb` finds 1428 in the same 200 000** (`scratchpad/
  fuzz_transport.py N SEED`, whose own default `N` is 20 000; re-driven
  seed by seed at `bf905b9`: base 134/150/136/146/141/136/136/159/145/145
  = 1428, tip 0 at every seed, cry-wolf 0 at every seed). The 134 at seed 1
  has its first counterexample at example #64, which is a figure only the
  20 000 reading admits. A seeded 4000-example run of it is a permanent
  test, with an anti-vacuity floor so a fuzzer that accepts nothing cannot
  pass.
  **Swept for the same class elsewhere and found none live:** z3 is
  in-process over the API with no text record protocol at all; the binary
  cvc5 transport splits and re-joins, but `_tokenize_sexpr` already treats
  every one of these as whitespace and its unsat/unknown leg fails closed
  on any noise;
  `slice_unknown_obligations` round-trips no text. Every construction is a
  permanent regression test (`tests/test_solver_audit_findings.py`, the
  `f4wheel2` block).

  **ONE LEG OF THAT SWEEP WAS FALSE AND IS WITHDRAWN (2026-08-08).** It
  read: *"`reproduce.py`'s no-`import stelling` scan reads MORE line-starts
  than Python's tokenizer does, which is the safe direction"*, i.e. that
  Python's statement-separator set is a strict SUBSET of `splitlines()`'.
  The two sets are not nested in either direction. Measured over the whole
  code-point range — `compile("x=1" + c + "y=2")` against
  `("a" + c + "b").splitlines()`, both read off this interpreter
  (`tests/test_solver_audit_findings.py::…_the_two_line_end_sets_are_not_nested`):

      splitlines() only : U+000B U+000C U+001C U+001D U+001E U+0085 U+2028 U+2029
      compile() only    : U+0023 '#'   U+003B ';'
      both              : U+000A U+000D

  `;` is the one that bites, because it can carry a real statement:
  `"x = 1; import stelling\ny = 2\n"` contains an `import stelling` that
  `ast.parse` finds and the line-start scan does not. (`#` only comments
  the rest of the line out, so it cannot introduce an import.) **NOT LIVE
  today, and the reason is a different one than the withdrawn sentence
  gave:** every piece of caller text reaching the emitted file goes through
  `reproduce.one_line`, which maps every character below U+0020 to a space
  and neutralises the triple quote, so nothing a caller writes can reach
  statement position; the sidecar and payload are `json.dumps`ed inside a
  raw triple-quoted string, which `json.dumps` cannot terminate. **Fixed
  rather than argued, because the scan's whole charter is to survive a
  future edit of `_TEMPLATE`** — it is called a *"structural refusal at the
  point of emission, not a comment asking the next author to be careful"*,
  and a refusal that a semicolon walks past is not that. The line-start
  scan is kept (it cries wolf on an import inside a string, which is
  cheap), and the emitter now ALSO walks the parse tree it was about to
  compile anyway and refuses any `Import`/`ImportFrom` of `stelling` or a
  `stelling.*` submodule, wherever on the line it sits.
  **Cost, measured rather than asserted.** The added refusal never fires on
  what the emitter emits: the reproducer for the same subject is
  byte-identical at `bf905b9` and here apart from the two lines that record
  the stelling sha, which is the provenance stamp doing its job. `reproduce`
  is not on the verdict path at all (importing `verdict`, `solvers`,
  `propagate`, `obligation`, `preconditions`, `harness` and `contracts`
  leaves `stelling.reproduce` absent from `sys.modules`), and the corpus that
  scores this per obligation agrees: `scratchpad/pin/corpus_pin.py`, 95 rows
  across the mechanism × shape × order × mode × semantics × leg grid, **2090
  per-obligation and verdict status keys compared, 0 moved, 0 non-status keys
  differing — on jax 0.11.0 AND on jax 0.10.2.** The vacuous test that pinned
  the withdrawn claim is replaced, not deleted: 10 parametrised ids retired,
  18 added, and every new one was driven red on a mutant before it landed.
- **2026-08-09 (pre-release): the cvc5 WHEEL transport's reader was blind
  to a `\r`, in BOTH directions, and the repair the entry above declined is
  now landed. Direction of movement: toward UNKNOWN, only. REACHABLE ONLY
  THROUGH A STALE INSTALL — a driver out of step with this parser — and NOT
  through any child the shipped `_cvc5_driver` can produce.** The entry
  above closes with *"`\r` from a driver out of step with this parser is a
  LIVE hole in both directions"*, and records a repair that was measured,
  dominant and unlanded. This is that repair.

  **The mechanism, in one line.** `_run_cvc5_wheel` captured with
  `capture_output=True, text=True`, and universal-newline decoding maps a
  bare `\r` to a real `\n` **before the string reaches the function**. The
  parser's alphabet check therefore never saw a `\r` to refuse: the hole was
  invisible to it rather than admitted by it. A stale child writing
  `opaque x1 j\rend 2\r` — with **no terminator record of its own anywhere**
  — supplied the reader's last line and got a definite answer.

  **BOTH DIRECTIONS, and the second is the one that matters.** Measured at
  `9564728`, real child, real bytes, no mocking: the same corpse returns
  `sat` with a model, `unsat`, or `unknown`, according to what its `answer`
  line happened to say. `sat` → REFUTED still runs through
  `_require_valid_refutation` and exact-rational replay, which is a
  downstream backstop. **`unsat` → VERIFIED has none.**

  **THE THREE ARMS, re-driven at `9564728` before anything was changed**
  (`scratchpad/probe_cvc5_backstop.py`, parts A–D; the shipped column is a
  record of an io layer this tree no longer contains and is checked against
  a real `text=True` spawn on every child in the run, so a drift in the
  model is an assertion failure and not a quiet wrong number):

  | case | (a) `text=True` | (b) `bytes.decode()` | **(c) `+ replace("\r\n","\n")`** |
  |---|---|---|---|
  | healthy POSIX `\n` | sat | sat | **sat** |
  | healthy Windows `\r\n` | sat | **failed** | **sat** |
  | stale `\r`, LF body | **sat** | failed | **failed** |
  | stale `\r`, CRLF body | **sat** | failed | **failed** |
  | stale `\x0b` | failed | failed | failed |
  | separators refused (LF stale) | 8 of 10 | 9 of 10 | **9 of 10** |
  | child writes invalid UTF-8 | **RAISES** | failed | **failed** |

  **(c) lands.** It dominates (a) on every case measured — identical answer
  AND identical values on both healthy children, strictly stronger on both
  stale ones, and `failed` where (a) raised an **uncaught
  `UnicodeDecodeError`** out of the transport. (b) buys the same ninth
  separator by refusing a healthy Windows child outright, and `README.md`
  names Windows for both solver wheels; that, and not behaviour on the stale
  children, is what rules (b) out. `_decode_child_stream` puts back by hand
  the one translation `text=True` was performing and nothing else.

  **A STRENGTHENING THE BRANCH DID NOT CLAIM, found by a blinded audit and
  re-measured here: (c) also closes a hole on a HEALTHY WINDOWS child.** A
  pipe cut is a byte prefix, so the set of a stream's prefixes is a superset
  of what any buffering regime can deliver. Driven byte by byte, real child,
  real bytes, both trees, `rc=0`: over the 664-byte CRLF transcript of a
  healthy run (`version`, `answer sat`, 40 `value` records, `end 40`, every
  record ending `\r\n`), **exactly one proper prefix returned a definite
  answer at `9564728` — byte 663 of 664, the cut landing between the final
  `\r` and its `\n`, returning `sat` with all 40 values from a child that
  never finished writing its terminator.** At this tip that prefix is
  `failed` (alphabet), and the count of definite answers off a proper prefix
  is **0 of 664**. The mechanism is the same one the table above is about,
  reached from the other side: universal-newline decoding turned the orphaned
  `\r` into a record boundary, so a torn terminator read as a whole one.
  Under (c) the `\r` survives the decode and the alphabet check refuses it.
  This is not the stale-driver direction and not a `README.md` platform
  caveat — it is a truncated run on the platform the branch went out of its
  way to keep working.

  **WHAT IS STILL NOT CLOSED, stated rather than left to be found.** The
  nine is over SINGLE characters. `\n` is excluded by construction — it is
  the protocol's own record boundary, so a writer that leaves one inside a
  field has written two records and there is nothing on the reader's side to
  detect — and the two-character `\r\n` is that same fact in a second
  spelling, a genuine record boundary under this reader exactly as under
  (a). Neither is a regression and neither is an improvement; both are the
  WRITER's, and the writer's printable-ASCII whitelist escapes them
  (`test_f4wheel3_a_crlf_inside_a_field_is_a_record_boundary_and_stays_one`).
  The writer half remains the load-bearing half and is not weakened by this.
  (c)'s one measured cry-wolf case is a healthy child reconfigured to BARE
  CR line endings, which no platform's `print` default produces and which
  `_cvc5_driver` never sets — asserted structurally, not remembered.

  **THAT RESIDUAL AND WINDOWS SUPPORT ARE ONE COIN, and the entry should say
  so rather than list them as two separate facts.** The `\r\n` survives the
  reader because `replace("\r\n", "\n")` runs BEFORE the alphabet check, so
  the `\r` is spent and never reaches it — the same order that makes a
  healthy Windows child readable at all. MEASURED here on a mutant that drops
  only that `replace` (arm (b)): the `\r\n`-in-a-field stale child goes
  `sat` → `failed` (alphabet), closing the tenth separator, **and** the
  healthy Windows child goes `sat` → `failed` in the same run, on the same
  line of code — `tests/test_solver_audit_findings.py -k f4wheel3` under that
  mutant is 3 failed / 8 passed, and two of the three are exactly those. **You
  cannot buy the tenth separator without breaking the platform**; the branch
  bought the platform, which is the right way round, because the residual is
  the WRITER's and the writer's whitelist closes it while nothing else closes
  a broken Windows install.

  **NO VERDICT MOVED, SCORED PER OBLIGATION — and the instrument had to be
  BUILT, because the one this repository reaches for says in its own
  docstring that it "has no solver escalation … so nothing here scores
  `solvers.py`" (`scratchpad/pin/corpus_pin.py`).** A zero from an
  instrument that is structurally blind to the file under change is not a
  zero. `scratchpad/crlf/corpus_solver.py`: 14 rows whose obligations the
  interval leg leaves UNKNOWN, each driven under cvc5-only, z3-only and the
  full portfolio — **48 escalated obligation records, 64 real solver spawns,
  24 discharged and 24 refuted-with-witness**, with rows in pairs differing
  in ONE CONSTANT so both directions are scored. Against a clean `9564728`
  worktree: **3009 leaf keys compared, 337 of them verdict-bearing
  (outcome / status / `answered_by` / witness values), 0 moved, and 0
  non-verdict keys differ.** Exactly two things are normalised and both are
  named in the code — the two tree roots, and the millisecond durations
  inside notes, which were 41 of the 41 remaining differences on the first
  run.

  **THE ZERO HAS A POSITIVE CONTROL, in each direction, and a determinism
  control.** Re-running the same tree moves 0 of 337, so the zero is not
  what the instrument always prints. `_decode_child_stream` ending in
  `.rstrip("\n")` — the fuzzer-found class, applied to the one function this
  branch adds — moves **155** verdict-bearing keys, VERIFIED → UNKNOWN and
  REFUTED → UNKNOWN (7 of each, complete list in the raw output). The
  transport reporting a `sat` as `unsat` moves **136**, **SEVEN** of them
  **REFUTED → VERIFIED** — every one of the corpus's seven refuting rows,
  under the cvc5-only portfolio — and under the full portfolio surfaces as
  the `SolverDisagreement` the portfolio exists to raise. Raw output:
  `scratchpad/crlf/RESULTS_crlf.txt`.

  **CORRECTED 2026-08-09, and the correction is the entry's own subject.**
  That count read **"six"**, written without re-running it, in an entry whose
  first paragraph is about a claim that rotted. A blinded audit caught it; the
  number was then re-measured here rather than taken on report — every arm
  re-run in this worktree, PC2 reproducing its recorded totals exactly
  (`{'discharged': 32, 'escalated_obligation_records': 40, 'spawns': 48,
  'violated_witness': 8}`) and its 136. The seven are `psd_false`,
  `cubic_false`, `product_bound_4`, `square_scalar_false`, `square_vec4_false`,
  `square_vec16_false` and `two_obligations_false`. **The evidence file could
  not have settled it either**, which is why the wrong number survived
  review: `diff` capped its per-key listing at `moved[:80]` and printed no
  sign of the cap, and the capture was piped on top of that, so
  `RESULTS_crlf.txt` displayed **3** of the 7. Both are fixed — the listing
  now announces its own truncation, and every capped listing is followed by a
  COMPLETE transition histogram and the COMPLETE list of `.verdict.status`
  moves, which is where a count in this prose should now be checked. **A
  silently truncated listing reads as a complete one**, and this file has been
  wrong that way before.

  **NEGATIVE CONTROLS.** Real unmocked cvc5 still returns `sat` with its
  model and `unsat`; VERIFIED and REFUTED both still land end to end (the
  corpus records 24 of each). z3 is scored on its own rows and moves
  nothing. The binary transport and z3 are not merely unaffected in
  behaviour but **textually identical**: of 34 functions in `solvers.py`,
  exactly two differ — `_run_cvc5_wheel` and the new `_decode_child_stream`
  — and `_make_run_cvc5_binary`, `_run_z3`, `_tokenize_sexpr`,
  `_model_values_from_text`, `escalate`, `make_solver_verdict` and
  `_screen_model` hash byte-identical to `9564728`.

  **BUT "TEXTUALLY IDENTICAL" IS NOT "COVERED BY THE REASONING ABOVE", and a
  future reader should not have to work that out.** `_make_run_cvc5_binary`
  and `_cvc5_binary_version` still capture with `text=True` and split with
  `splitlines()` — both halves of the pair this entry narrows for the wheel.
  They are unchanged from `9564728`, so nothing here regressed them and this
  branch does not own them; what does not carry across is the ARGUMENT. The
  binary leg has no record protocol of its own — no terminator, no count, no
  `end <n>` — so the forged-terminator shape the wheel's reader was narrowed
  against has nothing to forge, and the audit that scored this branch drove
  8 constructions through that transport and found **no exploitable
  direction: every failure was in the safe one** — *that last figure is the
  AUDIT's measurement and was **NOT re-measured** in this worktree; every
  other number in this entry was.* It is recorded as an open question about a
  DIFFERENT transport, not as a conclusion this entry's arm table supports.

  **WHAT IT COST, AND WHERE THE FIXTURES WERE MEASURING THEMSELVES.**
  Applying (c) reddens **16 tests**, all in
  `tests/test_solver_audit_findings.py`, all one cause —
  `AttributeError: 'bytes' object has no attribute 'encode'` from
  `Popen._communicate`'s `self._input.encode(...)` in the standard library's
  `subprocess` (line 2172 on CPython 3.12.3 and 3.11.15; on 3.10.20 that
  file is 2122 lines, so 2172 is past its end and the statement is at 2078
  — see the entry above on why the symbol is cited and the line is
  not), because `input=` must become bytes when `text=` goes.
  That figure was reproduced here exactly, ids and frame, and it is the cost
  of a TOLERANT decode; a strict one reddens **31**, the extra 15 being the
  `str`-shim call sites, which is the same population the 16's report named
  separately. Two fixtures were not merely stale but circular:
  `_wheel_child`'s shim named `text=True` **in its own body**, so every test
  routed through it scored the fixture's io choice rather than the
  transport's, and `_wheel_stdout` handed the parent a `str` that `text=True`
  had already decoded, so its cases were a model of the io layer instead of
  the io layer. Both now hand the child's bytes and forward the transport's
  own spawn kwargs. `…_the_alphabet_backstop_refuses_eight_of_the_ten` is
  retired for `…_the_reader_now_refuses_nine_of_the_ten_separators`, which
  spawns a real child instead of modelling the decode;
  `…_carriage_return_is_the_writers_alone_to_stop` is renamed and narrowed to
  `…_a_record_boundary_is_the_writers_alone_to_stop`, because `\r` is no
  longer the writer's alone and `\n` and `\r\n` still are.

  **THE FIVE ARTEFACTS BEHIND FIGURES ON THIS PAGE WERE RE-DRIVEN, not
  edited and assumed.** `fuzz_transport.py`: **0 unsound and 0 cry-wolf at
  every one of seeds 1–10, 20 000 examples each, 200 000 in total**, with
  its `decode()` — a restatement of what `text=True` did — replaced by the
  child's bytes; the restatement was a no-op in any case, since every record
  it builds goes through `_cvc5_driver._tail`.

  **AND THAT ZERO IS A REGRESSION CONTROL, NOT EVIDENCE FOR THIS REPAIR —
  the fuzzer is STRUCTURALLY BLIND to the character the branch is about.**
  The same fact that makes its `decode()` a no-op makes its generator unable
  to reach the defect: every record goes through `_cvc5_driver._token` /
  `_tail`, whose printable-ASCII whitelist escapes the separators the
  generator picks. MEASURED here over the generator alone, at the same seeds
  and counts the figure above quotes — 200 000 streams, 11 438 247
  characters: **0 raw `\r`, 0 `\r\n`, and 0 characters outside printable
  ASCII plus LF**, the whole emitted alphabet being 40 printable characters
  and the newline. So `0 unsound / 0 cry-wolf over 200 000` says the reader
  narrowing did not break the record protocol it was already being fuzzed
  against; it says nothing about `\r`, in either direction. The evidence for
  the CR repair is the arm table above, the separator sweep, the byte-prefix
  sweep and the `f4wheel3` block — all of which spawn children the fuzzer
  cannot generate. **A control that cannot fail on the change under review is
  not a control for it**, which is the same reading error as the blind
  per-obligation instrument two paragraphs up. `repro_forgery.py`,
  `repro_real_kill.py` and `probe_cvc5_value_channel.py` re-driven against
  real cvc5, every figure unchanged (3 of 4 value-channel cases still carry a
  raw separator, the ASCII control still carries none). `probe_cvc5_backstop.py`
  needed more than a fixture edit and now selects an arm by swapping the
  transport's own decode point, so the parser under test is the real one in
  every column.

  **AND A SKIPPED TEST WAS MEASURED RATHER THAN TRUSTED.**
  `tests/property/test_cvc5_protocol.py` needs hypothesis, which is on
  neither venv, and its `_FakeProc` handed `str` — quietly making its model a
  reader that does no decoding at all, so the `\r` row of its own separator
  table was scored against a parent that existed in neither direction. It
  hands bytes now, and because that change runs nowhere,
  `scratchpad/crlf/probe_property_grammar.py` drives that file's grammar and
  its oracle exhaustively without hypothesis: of **185** records the grammar
  can build, 15 contain a bare `\r`, **ZERO end in one** — which is the only
  way a `\r\n` could appear — and zero `value` records carry a separator at
  all; then **27 483 513 drives**, every transcript up to 3 records × every
  BYTE PREFIX × three exit codes, **0 counterexamples** with 30 accepted as
  the anti-vacuity floor. **That probe's own first run is its positive
  control and is recorded in it:** without the driver's grammar rule (`end`
  and `error` are its LAST record) it reported 6 counterexamples, every one a
  transcript with two terminators — the same mistake the property file
  already records making.

  **A PRE-EXISTING GAP IN THE NEIGHBOURING RULE, FOUND BY THE SAME AUDIT AND
  CLOSED HERE — UNCHANGED FROM `9564728`, so it is not this branch's doing.**
  The comment on the completeness check says *"the terminator must be the LAST
  line"*, and nothing tested that. The nearest test —
  `…_values_written_after_the_terminator_are_refused` — writes `end` early and
  then two MORE values, so the
  count inside the terminator stops matching what the parser tallies — a rule
  that asked only *"is a matching `end <n>` ANYWHERE in the stream?"* refuses
  that stream for the wrong reason, and is therefore not distinguished by it.
  POSITION and COUNT were never separated. MEASURED here: the mutant
  `complete = terminated and any(l == f"end {len(values) + opaques}" for l in
  lines)` **passed the entire suite as it stood at `ca5b7da` — 2494 passed,
  7 skipped, jax 0.11.0, rc 0** — and, through a real child with real bytes,
  returns `sat` with
  `(('x0', '1/2'),)` on
  `b"version 1.3.4\nanswer sat\nend 1\nvalue x0 1/2\n"`, a value harvested
  from a record written AFTER the run announced it was over; the shipped rule
  returns `failed`. Closed by
  `test_f4wheel_the_terminator_must_be_the_LAST_line_not_merely_present`,
  driven red against that mutant and green here before it landed. **No
  verdict moved and nothing shipped changed** — this adds a test to an
  existing rule, it does not alter the rule.

  Both series **2495 passed / 7 skipped**, `--collect-only` ids byte-identical
  between them at this tip AND at `9564728`; the branch's own delta read in
  both directions is **13 ids added and 2 removed**. (Collect-only reports
  2497 ids where the run reports 2502 items: the five `importorskip`
  module-level skips are reported at run time and contribute no id — the
  same offset at `9564728`, 2486 and 2491.) `reuse lint` rc=0.
  Constructions: the `f4wheel3` block of
  `tests/test_solver_audit_findings.py`, every one driven red at `9564728`
  first — and the three that must NOT move (both healthy children, the
  `\r\n`-in-a-field residual) were green there and are green here. **The
  counts and the id delta above moved by one on 2026-08-09** when the
  terminator-position test recorded in the paragraph above was added; they
  read 2494 / 7 and 12-added at `ca5b7da`, and both series were re-run here
  rather than adjusted on paper.
- **2026-08-07 (pre-release): jax 0.10 was in `TESTED_JAX_SERIES` and did
  not work — verdicts move, in the UNKNOWN → VERIFIED direction, on 0.10
  only.** `jex_core.ClosedJaxpr is jex_core.Jaxpr` is `False` on 0.10.2 and
  `True` on 0.11.0 (0.11 merged the two classes), so `isinstance(v,
  ir.ClosedJaxpr)` is a fact about the jax that produced a param rather than
  about the param. `propagate._is_add_combiner` and two `remat2` body readers
  tested the closed shape only. Measured at `8ef8f75` on 0.10.2, same source
  and same harness as 0.11.0: `x.at[0].add(5.0)` VERIFIED at `6 eqns: 6 known
  (100%)` on 0.11 and UNKNOWN at `4 known (67%); 1 ⊤ (scatter-add ×1)` on
  0.10; `jax.checkpoint` VERIFIED on 0.11 and UNKNOWN on 0.10. Ten tests
  failed on 0.10; **nine** were this (measured by applying the combiner hunk
  alone), the tenth being a `jit.inline` param with its own cause. The
  direction was always safe — VERIFIED → UNKNOWN, never the reverse, so this
  was capability loss and not unsoundness — but it was **not silent and not
  honest**: the coverage line disclosed the ⊤ while the note read
  `'scatter-add' has no sound rule for params {…}`, blaming the caller's
  program when a sound rule existed and the oracle had misread the container.
  Fixed structurally, with no version branching: `_is_add_combiner` accepts
  both containers, and one canonical `coverage.call_body` replaces two
  hand-rolled body walks — the fact was already written down in
  `test_remat_jaxpr_param_transcribes`, which was made series-tolerant during
  the 0.11 bump while **its two consumers were not**. **Nothing moves on
  0.11**: query hashes byte-identical at base and tip on both series
  (`3c15c4f5…` / `4dc49e99…`), the 0.11 arm behaviourally the old code over
  864 differential `ClosedJaxpr` shapes with 0 disagreements, and +7 tests /
  0 removed. Normalising `Inline.AUTO → False` was **refused**: it moves every
  0.11 query hash (proved — the rewrite turns the quickstart's `628a25ef…`
  into exactly the 0.10 `f32f4860…`), and `Inline` has 5 members of which 4
  are unexpressible on 0.10. A CI lane now pins the **series**
  `jax>=0.10,<0.11` and asserts the series it resolved, because
  `test_tested_jax_series_is_silent` passes on 0.11 too — the suite alone
  cannot tell which series a lane ran. Before this, the honest floor was
  `jax>=0.11`.
- **2026-08-07 (pre-release): a zero-element assumed predicate narrowed
  the declared box to a SUBSET and minted VERIFIED over it — a WRONG
  VERIFIED, reachable from the public API with two `any_array` calls.**
  Direction: **wrong VERIFIED → UNKNOWN**. This is the unsound direction,
  not capability loss; every affected verdict was false.

  `assume` reads its predicate universally, and jax broadcasts an `and`
  whose operand is size-0 to `bool[0]`: a universal over no elements,
  true at every point, admitting the ENTIRE declared box. The `and`
  recursion in `propagate._apply_assumed_pred` nonetheless classified
  each conjunct as if it stood alone, so

      k = any_array((),   "float64", (-1.0, 1.0))
      z = any_array((0,), "float64", (-1.0, 1.0))
      assume((k >= 0.5) & (z >= 2.0))
      return (assert_(k > 0.0),)

  narrowed `k` to `[0.5, 1.0]` and returned **VERIFIED** with the note
  `assume conjunct DROPPED … a superset` — while the narrowing had gone
  the other way. The one-sidedness the whole design rests on ("a proof
  over a superset is a proof over the subset") does not hold for a
  subset: independent dense sampling of the declared box, with no
  stelling in the loop, found 50 231 of 100 006 admitted points
  violating; `k = -1.0` is admitted by the declaration, admitted by the
  assume, and violates. The control with the assume removed returns
  UNKNOWN, so the assume was the whole of the false claim.

  **Which verdicts are retroactively invalid.** Any VERIFIED from a
  harness whose `assume` predicate has zero elements — i.e. any `assume`
  whose predicate array is `bool[0]`, which happens whenever ANY operand
  anywhere in its `&` tree is size-0, because a size-0 operand forces
  every enclosing `and` output to be size-0 (measured over all 256
  ordered pairs from a 16-shape set: 31 pairs lose an operand element,
  all 31 with a zero-size output, and no size-0 operand ever broadcasts
  to a nonzero-size output). It does not require a size-0 *declaration*:
  a comparison of a rank-0 against a size-0 value is itself `bool[0]`.
  Eleven constructions were measured VERIFIED-over-a-subset: size-0 in
  either operand position, nested at either depth, mixed with `|`, shape
  `(1,)` against `(0,)`, a rank-2 unit axis against a zero axis, a size-0
  comparison of nonzero operands, `eq` and `le` narrowings, and one
  inside a `cond` branch — each in both `vacuity_mode`s, at both `refine`
  depths, and under both semantics.

  **The rule now in the code.** A conjunct may be treated as separately
  assumed only if the predicate it belongs to actually constrains:
  `all(A & B)` implies `all(A)` only when every element of `A` survives
  the broadcast into the output. So a predicate node with zero elements
  licenses no narrowing, no satisfiability claim, and no
  unsatisfiable-precondition refusal, and the classification stops
  there — at the root, at every `and` node, and at every leaf.

  **The same root cause also minted false LOUD refusals**, in the
  opposite direction: `assume((k >= 2.0) & (z >= 2.0))` on `k` declared
  `[-1, 1]` raised `UnsatisfiableAssumptionError` — "harness defect;
  nothing was verified" — about a precondition that is `bool[0]` and
  therefore true at every point of the declared box. Both the empty-meet
  and the strict-boundary-collapse refusals did this. They no longer
  fire on a zero-element predicate.

  **Cost, measured, every unit of it confirmed unsound.** Across the 21
  swept constructions × 2 vacuity modes × 2 refine depths (84 rows): 44
  rows VERIFIED → UNKNOWN, every one with a violating admitted point in
  the independent oracle; 8 rows `UnsatisfiableAssumptionError` →
  UNKNOWN, both constructions satisfiable; 4 rows REFUTED → UNKNOWN,
  the only genuine capability loss (that refutation was sound — a
  witness drawn from the subset is still inside the admitted set — and
  it is withheld because the drop machinery cannot yet tell an exact
  drop from a widening one). 28 rows unchanged, **0 moves into
  VERIFIED**, and no post-fix VERIFIED anywhere in the sweep has a
  violating admitted point. The existing suite is unmoved: 2151 passed /
  2 skipped on jax 0.11.0 and on 0.10.2, the only red being the
  generated `docs/supported-primitives.md` line-number citation, which is
  regenerated.

  **Why no test caught it.** An instrumented full-suite run recorded 264
  calls into assume classification with atom shapes `()`, `(1,)`, `(2,)`,
  `(3,)`, `(5,)` — `size0` in **0 of 264**. The suite's size-0
  declarations all flow into `assert_` or a `jnp.all` reduction; none
  reaches an `assume` operand. A prior instrument at the adjacent
  `_conjunct_certainly_true` site read the same absence over 93 calls and
  concluded the size-0 branch was unreachable; it was measuring a suite
  with no size-0 assume in it.

  **What to re-run:** any recorded VERIFIED whose harness passes a
  possibly-empty shape to `any_array`, or compares values of different
  ranks inside an `assume`. Re-`check()` it: a `zero elements … it
  constrains nothing` note where a `assume CONSTRAINED … narrowed var`
  note used to be is this change, and the old VERIFIED was false. Every
  construction is a permanent regression test
  (`tests/test_size0_assume.py`, and the jax-free half in
  `tests/test_assume_constrain.py`); all 70 of the former fail on the
  parent commit's source.

- **2026-08-07 (pre-release): REFUTED over a superset of the assumed
  region — three paths, all wrong; A and B closed, D closed FOR ONE TRACE
  ORDER ONLY; verdicts move REFUTED → UNKNOWN and in no other
  direction.** F7's one-sided rule — *VERIFIED
  over a superset implies VERIFIED over the subset, so keep it; REFUTED
  over a superset does not, so withhold it* — was implemented on the
  whole-drop path and reached by none of three others.
  **(A)** `propagate._assume_constrain` routes on whether ANY conjunct
  narrowed, so a conjunction that narrows on one conjunct and drops
  another took the `if narrowed:` branch, where the drop earned a note
  calling the result "a SUPERSET" and then refuted over it — setting
  neither `uncertified` (the interval withhold) nor `assume_dropped` (the
  solver and affine legs' marking).
  **(B)** `affine.refine_propagation` declines wholly on
  `coverage.constrained`, which a DROPPED assume never raises, so the
  refinement re-minted a violation the interval leg had already judged and
  withheld — the same query returning UNKNOWN at `refine=None` and
  REFUTED at `refine="affine"`.
  **(D)** an assume whose only content is a **branch-scoped
  unsatisfiable** conjunct. Inside a possibly-untaken `lax.cond` branch
  `_unsatisfiable` must not raise (the assume is branch-scoped and the
  other branch is real, audit F2): it degrades to a branch-local vacuity
  note, appending to `vacuous` and NOT to `dropped`. The whole-drop guard
  read `if dropped or not vacuous:` — false in exactly that case — so the
  note and the two flags were gated *together* and neither flag was set.
  `x ∈ [-1,1]^3`, `cond(x[0] > 0, yes, no)` with `yes: assume(v >= 2.);
  assert_(v > 5.)` returned **REFUTED, witnesses=()** — identical to the
  same query with the assume deleted, while the same assume at top level
  RAISES `UnsatisfiableAssumptionError`. The run's own note said
  "obligations in this branch MAY BE VACUOUS under the branch's
  precondition" beside that REFUTED. All three detection sites reached it
  (empty meet, strict-boundary collapse, definitely-false constant
  comparison). **This one was live on `main` at `9efea6f` and still live
  at `3afbf01` after A and B were closed** — the earlier pass's
  instrument could not reach it, because every corpus row was a
  straight-line harness and D needs a `cond`. The fix is the rule the
  mixed path already carried (`if restricting or vacuous:`): the note gate
  is not the flag gate.
  **Which verdicts are retroactively invalid**: any REFUTED on a query
  whose assumed region is EMPTY and whose assume dropped a conjunct or
  was branch-scoped-unsatisfiable — the implication is then vacuously
  TRUE and the tool said the author's correct program was broken. All
  three carry `witnesses=()`: the refutation is set-level, so there is no
  concrete point to check the precondition against, which is why none had
  a user-visible tell. The measured
  exemplar, verbatim at `9efea6f`, x ∈ [-1,1]^3, `assert_(x > 5.)`:
  `assume(jnp.all(x >= 2.))` → UNKNOWN, but
  `assume((x >= -1.) & jnp.all(x >= 2.))` → **REFUTED** — and `x >= -1.`
  IS the declared lower bound, narrowing nothing (the propagator's own
  note reads "already within the assumed region"). **Cost, as a class
  with its sample size, not as a list**: over 2844 corpus rows (1710
  harnesses × `refine` ∈ {None, affine}) whose ground truth comes from a
  numpy oracle — 50 000 samples plus every corner plus a grid, stelling
  never consulted — **128 wrong REFUTEDs closed and 118 legitimate
  REFUTEDs lost, every move REFUTED → UNKNOWN; zero VERIFIED moved in
  either direction, and zero wrong VERIFIEDs before or after.** That
  corpus covers A and B only: every row is a straight-line harness, so it
  is blind to D by construction, and D's cost is not in those figures.
  **D's cost was first measured on an instrument that could not see it,
  and the figure this entry shipped was false.** That instrument — 216
  rows, 3 branch guards × 6 branch-scoped assumes × 6 obligations ×
  `refine` ∈ {None, affine} — reported `wrong-REFUTED 12 → 0` beside `CORRECT
  66 → 66`, and the entry read that as *"D's fix costs zero legitimate
  refutations, because the only rows it touches are the vacuous ones"*.
  **Both halves were false.** The two flags the fix makes unconditional
  are `_Propagator` state, **whole-run and trace-order-scoped, not
  path-scoped**: `uncertified` is set where the assume is walked and read
  where each obligation is walked, and `assume_dropped` has no order in it
  at all. They withhold refutations with nothing to do with the vacuous
  branch. The instrument could not have reported that, for three
  independent reasons, each sufficient alone. **(i)** every row's sibling
  branch carried `assert_(v > -1e9)` — an obligation **unfalsifiable over
  the declared box**, so the sibling could never be why a row refuted.
  **(ii)** the harness returned only the `cond`: **no obligation existed
  outside a branch**. **(iii)** the oracle was `admitted = taken & asm(P)`
  — *truth over the points that TAKE the branch and satisfy its assume* —
  so **an obligation outside the vacuous branch was never scored**. The
  stated cause was the instrument's premise, not a finding of it.

  **Re-measured on a corpus built to see it.** 1296 rows: 3 guards × 2
  branch SIDES for the assume × 6 branch-scoped assumes (incl. a
  no-assume control) × 2 claims in the assumed branch × 3 **falsifiable**
  claims in the sibling × 3 positions for a **top-level** obligation
  outside any cond {absent, traced before the cond, traced after it} ×
  `refine` ∈ {None, affine}; oracle 59 269 numpy points (50 000 uniform +
  8 corners + a 21³ grid), stelling never consulted, and **every**
  obligation scored over the points the **declaration** admits.

  | tree | REFUTED | UNKNOWN | VERIFIED | wrong-REFUTED | wrong-VERIFIED |
  |---|---|---|---|---|---|
  | `9efea6f` | 864 | 360 | 72 | 24 | 0 |
  | `3afbf01` | 864 | 360 | 72 | 24 | 0 |
  | `6237e07` | 672 | 552 | 72 | **0** | 0 |

  **On this corpus, `3afbf01 → 6237e07` closes 24 wrong REFUTEDs and
  costs 168 correct ones.** 192 rows move, every one REFUTED → UNKNOWN;
  VERIFIED unchanged at 72; wrong-VERIFIED 0 → 0; and the three ledgers
  are row-for-row identical on jax 0.11.0 and 0.10.2 (0 disagreements in
  1296). **72 of the 168 are provably not the vacuous branch's**: in
  those rows the assumed branch's own claim is `v > -5.`, true at every
  point of the declared box, so no refutation could have originated there
  — what was withheld is a top-level obligation's or a sibling branch's.
  **That figure is scoped to this corpus of 1296 rows of this shape**; it
  is not a rate over jax programs, and the shape mix (half the rows put
  the assume in the false branch, two thirds carry a top-level
  obligation) is chosen to REACH the defect, so it over-samples it by
  construction. What generalises is not 168: it is that the number is not
  zero and cannot be, because the withhold is not path-scoped.

  **The verdict depends on the textual position of unrelated code.**
  Measured at `6237e07`, both `refine` legs, both series, one program per
  row and nothing else varied:

  | probe | `3afbf01` | `6237e07` |
  |---|---|---|
  | top-level obligation traced BEFORE the cond | REFUTED | **REFUTED** |
  | the SAME obligation traced AFTER the cond | REFUTED | **UNKNOWN** |
  | unsat assume in the FALSE branch, refutable claim in the TRUE one | REFUTED | **UNKNOWN** |
  | unsat assume in the TRUE branch, refutable claim in the FALSE one | REFUTED | **REFUTED** |

  jax lowers `lax.cond(p, t, f, x)` to `branches=(false_fn, true_fn)` —
  read back off the jaxpr, the branch names print as `['no', 'yes']` — so
  the false branch is walked first, and **an unsatisfiable assume in the
  false branch poisons the true branch while the same assume in the true
  branch does not poison the false one.** In the corpus that asymmetry is
  the whole of the loss among rows with no top-level obligation: with a
  trivially-true claim in the assumed branch, 12 rows are lost with the
  assume in the false branch and **0** with it in the true branch. The
  decisive row is a `cond` whose `yes` branch has **no assume** and
  asserts `v > 5.`, and whose `no` branch carries the unsatisfiable
  assume beside a trivially-true claim: 99 827 of 200 008 sampled points
  take `yes` and every one of them violates the claim; `3afbf01` says
  REFUTED and `6237e07` says UNKNOWN, and **the run's own note names the
  branch that has no assume in it** — `violation WITHHELD from REFUTED at
  … (h.<locals>.yes)`, beside a vacuity note that names `no`.

  **The solver leg has no order in it at all, and the corpus above cannot
  see that either** — it runs no solver. `assume_dropped` is a whole-run
  boolean read once at the end of escalation, so a top-level obligation
  refuted by a **replay-confirmed concrete witness** is withheld by an
  unsatisfiable assume inside an unrelated `cond` branch, in BOTH trace
  positions. Measured with `solver_timeout_ms=4000`, z3 5.0.0, on a cond
  that carries the assume and no assert (so the obligation slices still
  map): `assert_(x.sum() > 0.5)` over `x ∈ [-1,1]^3` is REFUTED with a
  witness at `3afbf01` whether it is traced before or after the cond, and
  UNKNOWN at `6237e07` in both. This one is not reachable when the cond
  contains an assert — escalation declines those queries outright
  (*"asserts nested in sub-jaxprs cannot be mapped to slices"*) — which is
  why the branch's own corpus never met it.

  **D is closed for ONE TRACE ORDER.** With the assume traced AFTER the
  `assert_` *inside the same branch*, 36 of 36 rows (3 guards × 6
  branch-scoped assumes × 2 `refine` legs) still REFUTE at `6237e07`,
  identically to `9efea6f` and `3afbf01`, and **20 of those 36 refute
  over a branch precondition satisfied by 0 of 59 269 sampled points**.
  That is the same ordering question the "Still standing" paragraph
  below records, reaching D's own shape; the headline for D is closed
  *for the order the fix sees*, and the earlier word "closed" contradicted
  the paragraph two sentences later.

  **A ledger weaker than its own pre-registration is a defect of its own
  kind.** `PREREG_REF1` scopes every cost figure to the corpus that
  produced it ("on this corpus", with the command beside it); this entry
  dropped the scope and generalised — and then generalised a number the
  instrument was structurally unable to produce. The pre-registration was
  honest and the shipped document was not, which is why the correction
  here is not only the number. The three blind spots above are written
  down in `scratchpad/PREREG_REF1.md` beside the corpus so the next
  reader does not re-derive the zero from the same tool.

  **The 168 is not intrinsic to closing D.** A path-scoped withhold —
  `uncertified` saved and restored across each `cond` branch, as `env`,
  `exact`, `nan` and `taint` already are, and only where the selector
  admits more than one branch — was measured (uncommitted prototype, four
  lines in the cond handler): on the same 1296 rows it re-opens **0** of
  the 24 wrong REFUTEDs, recovers **144 of the 168**, and additionally
  refutes 216 rows that `3afbf01` was already losing to the same leak,
  every one of them CORRECT — 1104 CORRECT / 192 CONSERVATIVE / 0 wrong
  of either kind. The 24 it still withholds are the rows whose only definite
  violation lies INSIDE the vacuous branch, which is the case the withhold
  is for. It is **not** a drop-in and is not adopted here:
  it leaves the solver and affine legs whole-run (they read
  `assume_dropped`, which is also the disclosure surface, so path-scoping
  them means replacing a boolean with a per-obligation marking), and the
  `len(possible) > 1` guard is load-bearing — without it the prototype
  re-opens a wrong REFUTED whenever the selector is definite, because a
  branch that is always taken is not a branch. And the argument for it —
  *a sibling branch is a disjoint set, so a sibling assume cannot bear on
  it* — is **not** plain correctness: refuting inside a branch already
  assumes that branch is REACHABLE, and the analysis never certifies that.
  Measured, with no assume anywhere, at `9efea6f`, `3afbf01`, `6237e07`
  and `main` at `c20f38e` alike:
  `cond(x[0] - x[0] > 0., yes, no)` over `x ∈ [-1,1]^3` with
  `yes: assert_(v > 5.)` and `no: assert_(v > -5.)` returns **REFUTED**,
  and the guard is false at every point — interval subtraction is
  correlation-blind, so `x[0] - x[0]` is `[-2, 2]`, the selector is
  undetermined, and `yes` is walked. **The query is TRUE.** That is a
  wrong REFUTED of its own class, independent of assumes and of D, and it
  is live on `main`; it is recorded here because path-scoping inherits it
  rather than introduces it. Reported, not implemented:
  the scope of an `assume` is the principal's to rule on, and a concurrent
  `propagate.py` change makes a second one here a collision.

  `test_the_branch_withhold_is_ONE_SIDED` pins that a discharge inside a
  vacuous branch still discharges. The loss
  is real: nothing at the interval level establishes that a dropped
  conjunct is SATISFIABLE, so a genuine refutation over a non-empty
  region is withheld alongside the vacuous ones
  (`test_the_cost_of_the_withhold_is_pinned_where_it_falls`). It is not
  charged where it is not owed: `_conjunct_certainly_true` keeps the
  refutation when the dropped conjunct's own box is definitely TRUE over
  the boxes in force, because such a conjunct restricted nothing and its
  absence widened nothing — that recovered 20 of the 60 verdicts the
  plain withhold cost on the relational corpus, with 0 unsound
  restorations. **Still standing**: an `assume` traced AFTER the
  `assert_` it should constrain still refutes over an empty region at the
  interval leg (60 of the corpus's rows). That is a question about what
  an `assume` SCOPES OVER, not about superset judging. The principal has
  ruled it **query-scoped**; implementing that uniformly is a separate
  change with its own pre-registration, and none of it is done here.
  **The two legs do not yet agree on it, and this branch moved one of
  them.** The affine guard keys on `propagation.assume_dropped`, a
  whole-run flag with no order in it, while the interval withhold reads
  `uncertified` at assert time — so the affine leg is already
  query-scoped and the interval leg is still order-scoped. Measured
  (`refine=None` / `refine="affine"`) on an assert traced before a
  wholly-dropped assume over an empty region: interval-decided
  `REFUTED / REFUTED` at both `9efea6f` and the branch; affine-decided
  `UNKNOWN / REFUTED` at `9efea6f` but `UNKNOWN / UNKNOWN` on the branch,
  with the no-assume control still `UNKNOWN / REFUTED` on both. The
  direction is REFUTED → UNKNOWN and agrees with the ruling, but it
  arrived as a side effect of B's guard rather than as a decision, and it
  falsifies PREREG_REF1's scored C7 (see its re-scoring). All four cells
  are now pinned by
  `test_an_assume_after_the_assert_is_pinned_on_BOTH_legs`, and the
  disagreement itself by
  `test_the_two_legs_do_not_yet_agree_on_assume_ordering`, so the
  forthcoming query-scoping change lands loudly wherever it touches.
  Constructions: `tests/test_vacuous_refutation.py`; pre-registration and
  outcomes: `scratchpad/PREREG_REF1.md`.

- **2026-08-07 (pre-release): two sub-jaxpr defects closed — a REFUTED
  from a branch nothing certified is reachable, and an obligation the
  analysis never looked at.** Both moves are **towards UNKNOWN**: nothing
  became VERIFIED, and nothing became REFUTED that was not before.

  **(1) Refuting inside a branch presumed a reachability that nothing
  certified.** The `cond` transfer runs every branch the index interval
  ADMITS and judged each branch's obligations over the whole declared
  box. Admitting is not certifying: interval arithmetic
  over-approximates, so `cond(x[0] - x[0] > 0., yes, no)` over
  `x ∈ [-1,1]^3` admits both legs while the guard is false at every
  point, and `yes: assert_(v > 5.)` returned **REFUTED** on a query that
  is true everywhere — the shape the entry above recorded as live on
  `main`. The general rule, and not the `x - x` shape: **a definite
  violation found inside a branch the analysis only admits is withheld
  from REFUTED**, reported `unknown`, with
  `propagate.UNCERTIFIED_REACHABILITY_REFUSAL` naming why. It is *not*
  reported `discharged` — an unreachable obligation is vacuously true,
  and vacuous truth is a different claim, reserved.

  A branch is certified reachable in exactly two ways, both sound and
  both sufficient rather than necessary: the index box FORCES it (then it
  runs whenever the cond runs), or a **point witness** reaches it —
  propagation re-run with every declaration pinned to a single point of
  its own declared box, where a singleton index box there means the real
  program takes that branch at that point. The witness search is
  deterministic (a verdict must not depend on a seed), runs at most 16
  extra propagations and only when a branch-scoped violation is actually
  on the table, and certifies nothing at all when a constraining assume
  is in force, because a point of the declared box need not lie in the
  narrowed admitted set.

  **Retroactively invalid:** any REFUTED whose refuting obligation sat
  inside a `cond`/`switch` branch the index interval did not force.
  Re-run: such a query now returns UNKNOWN with the reason in its notes.
  A REFUTED from a top-level obligation, from a forced branch, or from a
  branch some point of the declared box reaches, is unaffected.

  **Measured cost**, on a 736-harness corpus x 3 legs (`refine=None`,
  `refine="affine"`, `vacuity_mode="inputs-only"`), scored **per
  obligation** against a numpy oracle that never calls stelling: **183**
  obligations left `violated-over-set` that the oracle finds are
  evaluated at **zero** of the sampled points — every unsound refutation
  in the corpus, removed — and **216** left it that the oracle confirms
  are really violated somewhere, which is honest withholding. Every one
  of those 216 has a guard the analysis cannot evaluate at a point: 144
  route through `sin`, which has no registered transfer; 24 sit under
  `jax.lax.switch`, which clamps its index through the unregistered
  `clamp` — **so no `switch` branch can currently be forced or witnessed,
  and none can refute**, pinned by
  `test_no_switch_branch_can_refute_while_clamp_is_unregistered`, which a
  `clamp` transfer would flip; the remaining 48 have a satisfiable guard
  whose true region is about 1% of the box, which the finite witness grid
  misses. Of the corpus's **750** sound branch-scoped refutations,
  **534** were KEPT, so the rule did not degenerate into "never refute
  inside a branch"; the 537 top-level ones are untouched.

  **(2) An obligation inside a `scan`/`while` body was silently dropped.**
  The body IS transcribed — the `stelling_assert` equation is in the IR —
  and propagation never descends into it, so the obligation was never
  collected, never judged and never mentioned: `obligations = 0`,
  `notes = ()`. A user's check then read as "could not decide" when
  nothing had looked. Worse, and measured: one true top-level assert
  beside one false assert inside a `scan` stamped **VERIFIED**. Every
  `stelling_assert`/`stelling_nonvacuity` inside a sub-jaxpr no transfer
  descends into (`scan`, `while`, any unregistered primitive with a body,
  and every decline path) is now recorded as an obligation with status
  `unknown` whose detail says **NOT EXAMINED**, plus a note naming its
  source location and the primitive that swallowed it. `unknown`, never
  `discharged`: an unexamined check must not be able to complete a
  VERIFIED.

  **Retroactively invalid:** any VERIFIED on a query with an `assert_`
  inside a `scan`/`while` body, or inside any sub-jaxpr the coverage line
  reported `unreached`. Re-run: such a query now returns UNKNOWN and
  lists the obligation. The corpus measured **9** such VERIFIED rows, all
  removed, and **12** further VERIFIED rows that the oracle finds sound
  also moved to UNKNOWN — the check they omitted happened to be true, and
  stelling still had not looked at it.

  A whole-query DECLINE was considered and rejected: it discards the
  judgments the analysis *did* make on the rest of the query, and
  DECLINED means "the query could not be read", which is false here — it
  was read, and one obligation was not examined. Naming the obligation
  says exactly that, and the obligation count stops lying.

  Constructions: `tests/test_branch_reachability.py`; pre-registration,
  corpus design and outcomes: `scratchpad/PREREG_REACH.md`.

- **2026-08-08 (pre-release): the reachability certificate certified with
  points that are not in the declared set — a live wrong REFUTED — and its
  probe grid collapsed on wide boxes.** Both are defects in the witness
  search added by the entry above; the direction of the first fix is
  **REFUTED → UNKNOWN**, and nothing here moves anything toward VERIFIED.

  **(1) A witness that is not a member certifies nothing, and the probe
  points were not members.** `_probe_point` rounded an integer
  declaration to an integer and then clamped the result to `[lo, hi]`,
  which put the non-integral endpoint straight back. The declared set of
  an integer or boolean declaration is the set of that dtype's VALUES in
  the interval — stelling says so itself when it refuses `int32
  (0.2, 0.8)`: "int32 represents the integers ... and the interval
  contains none of them" — so `any_array((), "int32", (0.2, 2.8))`
  declares `{1, 2}`. Probes pinned it to `0.2` and to `2.8`, and `i < 1`
  and `i > 2`, **false at every member**, were certified reached and
  stamped **REFUTED**. Reproduced on jax 0.11.0 and 0.10.2.

  The declared set is bounded twice, and only the first bound was ever
  applied: `int8 (-1e9, 1e9)` declares `[-128, 127]`, and
  `k.astype(f64) < -200.0` — false at every int8 value — REFUTED over it.
  The clamp now goes INTO the member set before rounding, using both the
  interval's endpoints rounded inward and the dtype's own range. A box
  holding no value of its dtype yields no probe at all: no member, no
  witness, and a probe that cannot be formed leaves the declaration at
  its full box, which certifies nothing. `any_array` refuses such a
  declaration at the door, so that path is reachable only through
  hand-built IR — the safe answer there is "no point", never "some
  point".

  Normalising the bounds at `any_array` instead — recording `int32
  (0.2, 2.8)` as `[1, 2]` — was considered and rejected. It is the wider
  change (`any_array` is the public surface, and the recorded box travels
  into every message, the SMT emission and the exactness claims), and it
  moves the propagated box in the NARROWING direction, which is the
  direction that can manufacture a VERIFIED: a declaration bound would
  then be load-bearing for discharge, not only for a witness. Fixing the
  probe can only ever change which branches are certified, and the
  certificate's only effect is to rewrite `violated-over-set` into
  `unknown`.

  **Retroactively invalid:** any REFUTED whose refuting obligation sat
  inside a `cond`/`switch` branch certified by a probe over an integer or
  boolean declaration — that is, any REFUTED on a query with an integer
  declaration whose branch guard was not already forced by the index
  interval. Re-run: such a query now returns UNKNOWN unless a real member
  witnesses the branch. **Float declarations keep a residual of exactly
  this defect, closed by entry (5) below.** The sentence that stood here
  said they were "unaffected by this half"; that is true of `float64`
  and false of every narrower float format, and the corpus behind the
  figures below could not tell the difference because it declares no
  narrow float at all.

  **(2) The probe grid degenerated on wide boxes.** `lo + f*(hi - lo)`
  overflows to `+inf` when the box is wider than half the float range:
  measured on `[-1e308, 1e308]`, 15 of the 16 probes collapsed onto `hi`
  and the 16th was NaN, so a guard satisfied only at the low end was
  never witnessed and its violation was withheld. The fraction is now
  computed as a convex combination, which never forms the difference and
  keeps 16 distinct points (`lo` at f=0 and `hi` at f=1, exactly).

  **Retroactively invalid:** an UNKNOWN on a query whose branch guard is
  satisfiable only on the low side of a very wide declared box may now be
  REFUTED. No VERIFIED is affected — the certificate cannot discharge.

  **Measured cost**, on a NEW 336-case corpus over ten declarations x 2
  legs, scored **per obligation** against a numpy oracle that never calls
  stelling and that samples MEMBERS (the previous oracle sampled
  `uniform(lo, hi)`, which agreed with the defect that `0.2` was an int32
  point — this defect was invisible to it): **18** obligations moved
  `violated-over-set → unknown`, every one of them a refutation the
  oracle finds is evaluated at **zero** points of the declared set, and
  **8** moved `unknown → violated-over-set`, every one confirmed really
  violated by the oracle (the wide-box witnesses). **0** obligations
  became `discharged`; **0** queries moved into VERIFIED; **0** sound
  refutations were lost. Unsound obligation rows in the corpus: **18 → 0**.

  **Scope of that "18 → 0"** (corrected after the fact; the paragraph
  above is what was measured and is left as recorded): the corpus is
  `scratchpad/probe/corpus.py`, whose only float declarations —
  `f_pos`, `f_str`, `f_wide` — are all `float64`. `float64` is the one
  float format that IS its own interval, so a `float64`-only corpus
  cannot observe the narrow-float residual at all, and the zero is a
  zero over the classes it declares (integer, boolean, `float64`), not
  over all declarations. The residual it could not see is entry (5),
  measured on a corpus that can.

  **(3) An index-EXCLUDED `cond`/`switch` branch is dropped, not
  recorded — deliberately.** When the index interval excludes a branch,
  the walk counts its equations unreached and does not collect its
  obligations. That is a silent drop, and as literally worded it
  contradicts clause **C5** of `scratchpad/PREREG_REACH.md` ("every
  `stelling_assert` inside a sub-jaxpr propagation does not descend into
  is recorded ... `unknown`"); C5's own falsifier could not see it,
  because it ranged only over obligations the oracle saw executed. It is
  kept, and this is the statement C5 lacked: an index-excluded branch is
  not unexamined, it is PROVED untaken. The index box over-approximates
  the true index set, so a branch outside it is taken at no point of the
  declared set, and its obligation is vacuously true. Recording it
  `unknown` would assert an ignorance the analysis does not have, and it
  is not free: measured on the same corpus, naming them costs **28 of 28**
  VERIFIED queries and removes **zero** unsound rows (the oracle finds no
  dropped obligation that is ever false — `SWALLOWED_FALSE` is 0 before
  and after). The coverage denominator still counts them, and
  `test_an_index_excluded_branch_is_deliberately_dropped` pins the
  behaviour so it can never become silent again.

  **(4) The swallowing primitive named in a NOT EXAMINED obligation is now
  the innermost one.** An assert inside a `while` inside a `scan` was
  reported against `'scan'` while its source location pointed into the
  `while`; the two disagreed and the message sent the reader to the wrong
  construct. Message-only: no verdict flips.

  **(5) The same defect, on FLOAT declarations: a probe point that is
  not a value of the declared float dtype.** Entries (1) and (2) shut
  the integer face of "a witness that is not a member certifies
  nothing" and left the float face open. A float declaration is bounded
  twice in exactly the way an integer one is: `float32 (-1e308, 1e308)`
  declares the `float32` values of that interval, i.e. `[-3.4e38,
  3.4e38]`, and `float32 (v, (v + nextafter(v))/2)` declares the single
  value `{v}`. `_member_bounds` returned floats' intervals untouched and
  `_probe_point` left every interior point on binary64's grid, so the
  witnesses were binary64 points that the declared dtype does not hold.
  Measured on the sweep this branch already ships
  (`tests/test_probe_witness.py::_SWEEP_BOUNDS`, 215 bound pairs): of
  the **6880** `float32` probe values formed, **6716** were not
  `float32` values and **90** were outside `float32`'s range entirely;
  for `float16`, **6736** and **6184**. `float64` was clean, which is
  why only it was right.

  Two of the three constructions are as old as the certificate; the
  third is NEW on this branch, because entry (2)'s wide-box repair is
  what first let a probe reach `±1e308` on a box the dtype cannot hold.
  Reproduced on jax 0.11.0 and 0.10.2, `JAX_ENABLE_X64=1` (without x64
  jax truncates the declaration to `float32` and the effect vanishes):

  - `float32 (-1e308, 1e308)` with `w < -3.4028234663852886e38` (the
    `float32` minimum): false at every `float32`, UNKNOWN on `688e829`,
    **REFUTED on `62e4190`**;
  - `float32 (v0, (v0 + nextafter(v0, inf))/2)` with `w > v0`: the
    declared set is `{v0}`, REFUTED on both;
  - `float32 (-1e308, 1e308)` with `w.astype(float64) > 1e39`: REFUTED
    on both — literally the second face of the defect entry (1) fixes
    for `int8 (-1e9, 1e9)` with `k.astype(float64) > 200.0`.

  The endpoints are now rounded INWARD onto the declared format's own
  grid and clamped to its finite range, and every probe value is rounded
  onto that grid too (`_round_in_format`, directed, in exact integer
  arithmetic on the significand — round-to-nearest can cross the
  endpoint it is narrowing, and `math.nextafter` steps to the next
  *binary64*, which for a narrower format is not a value of it: the same
  trap `_INT_DTYPE_BOUNDS` documents for `int64`). `float64` is the
  identity under this rounding. **Provenance, corrected 2026-08-08:** an
  earlier draft of this sentence read "measured over 80028 values", which
  describes no check this tree performs. What the SHIPPED test
  (`test_directed_format_rounding_never_crosses_the_value_it_rounds`)
  builds is **1013 values × 4 formats × 2 directions = 8104 directed
  roundings**, of which the `float64` leg is **2026**, all identities, 0
  mismatches. The substance is unaffected and was re-measured
  independently, on a value list this tree does not contain (uniform
  draws, `ldexp` draws across the whole binary64 exponent range, and
  Gaussian draws around `1e-300`): **60009 `float64` values × 2
  directions = 120018 roundings, 0 mismatches**.

  A dtype whose grid neither table names now yields **no member** rather
  than the raw interval: `any_array` accepts `int2`, `uint2`, five
  `float8`/`float4` formats and the two complex dtypes, and `int2 (-1e9,
  1e9)` was pinned to `±1e9`, which is not an `int2`. **That
  default-deny is latent on `int2`/`uint2` and LIVE on `float8` — both
  halves measured on jax 0.11.0, `JAX_ENABLE_X64=1`, and an earlier
  draft of this entry called the whole of it latent.**

  - `int2`/`uint2`: latent, as recorded. Five constructions per dtype of
    the shape `cond(w.astype(int32) > 1000, assert_(v > 5.0), assert_(v >
    -9.0))` over `(-1e9, 1e9)`, positive controls (`>= -2`, `>= 0`, true
    at every member) included, are **UNKNOWN at `688e829`, `62e4190`,
    `0222925` and today** — no verdict moves in either direction.
  - `float8_e4m3fn`: **a live wrong REFUTED, closed.**
    `any_array((), "float8_e4m3fn", (-1e9, 1e9))` with
    `cond(w.astype(int32) > 1000, …)` is **REFUTED at `688e829` and at
    `62e4190`, UNKNOWN from `0222925`**. Ground truth, from
    `ml_dtypes`/numpy over all 256 bit patterns and never from stelling:
    the format has **253** distinct finite values, its largest is
    **448**, so **0** members satisfy the guard — the branch runs at no
    point of the declared set and the refutation was over nothing.
  - **And the cost the earlier draft did not state at all.** The same
    default-deny withholds SOUND refutations over the same formats.
    `float8_e5m2` (247 distinct finite values, largest 57344) has **24**
    members with `int32(w) > 1000`, so the branch really runs and its
    obligation really fails: **REFUTED at `688e829` and `62e4190`,
    UNKNOWN today** — a real loss. `float8_e4m3fn`'s own positive
    control, `w.astype(int32) > -1000`, is true at every one of its 253
    members and is lost the same way. The float8 half of this fix is a
    trade, not a free withholding, and the trade is in the safe
    direction.
  - **`bfloat16` is a third case, and it is the load-bearing one.**
    `bfloat16` IS in `_FLOAT_FORMATS`, so it is not default-denied, and
    no `bfloat16` verdict moves across the repair: `bfloat16 (-1.0, 1.0)`
    with `cond(w.astype(int32) > 0, …)` is REFUTED at all four
    revisions, soundly (`1.0` is a member and casts to `1`). That stability
    is bought by ONE table row and nothing else, which the measurement
    shows rather than argues: with `("bfloat16", "float64")` added to
    `_EXACT_CONVERSIONS` in a scratch mutant — the only thing standing
    between these constructions and a verdict — `688e829` and `62e4190`
    mint **two false REFUTEDs** (`bfloat16 (-1e308, 1e308)` with
    `w.astype(float64) > 1e39`, false at every `bfloat16` since the
    format tops out at `3.39e38`; and the sub-ulp box `bfloat16 (v0, (v0
    + nextafter(v0))/2)` with `w.astype(float64) > v0`, whose declared
    set is `{v0}`), while `0222925` and today mint **none** and keep both
    positive controls (`> 1e30`, `>= v0`) REFUTED. The `bfloat16` half of
    `_member_bounds` is CONFIRMED correct, not merely untested.

  Why the fix is the default-deny and not an `int2`/`uint2` row in
  `_INT_DTYPE_BOUNDS` — measured, in its own worktree, not reasoned. With
  the row added: no `int2` verdict moves (still UNKNOWN on all 55
  constructions tried per dtype, positive controls included), the suite
  is green but for the two tests that assert the dtype is unnamed, and
  two things do change, both away from withholding. `_snap_integer`'s
  decline stops being unconditional — it goes from "no representable
  range is registered for it, so integer wraparound cannot be excluded"
  to a real range check, which ADMITS wherever the result fits — and
  `_conversion_exactness("int2", "float64")` flips from `"unknown"` to a
  minted `"exact"`, a claim about a dtype that nothing else in the
  module knows. Admission stays gated on `_EXACT_CONVERSIONS` membership,
  so neither turns into a verdict today, and that is precisely the
  argument for not doing it: the repair needed here is a withholding, and
  the table is where admissions come from.

  **One residual in the same function, recorded and hardened (2026-08-08,
  not reachable through the public API).** `_member_bounds` had no NaN
  guard, and NaN defeats the emptiness test it relies on: measured,
  `_member_bounds(nan, 1.0, "float32")` returned `(nan, 1.0)` rather than
  `(None, None)`, because `nan > 1.0` is False so `m_lo > m_hi` never
  fires; and on the integer path `_member_bounds(nan, 1.0, "int32")`
  **raised `ValueError: cannot convert float NaN to integer`**. Both are
  contained twice over, and both containments were measured, not assumed:
  `any_array` refuses a NaN bound at declaration for all three
  placements (`(nan, 1.0)`, `(1.0, nan)`, `(nan, nan)` — *"declare an
  empty set; refusing at declaration time"*), and `_probe_point` returns
  `None` on any non-finite value it forms, so the wrong pair could not
  have become a witness even if it were reached. The guard is added
  anyway — `return None, None` on a NaN endpoint, ahead of both branches
  — because the safe answer to "is this box inhabited" under NaN is *no
  member I can vouch for*, and because the next internal caller would
  otherwise meet the raise. It moves nothing: 0 of 648 obligation rows
  across the two assume corpora, and 0 suite outcomes on either jax series
  — the whole suite at that pass, every test and both skips. **The
  numerator is the claim; the denominator was a scale marker and has been
  dropped**, because the suite has gained tests on most days since and a
  frozen total in a dated entry reads as a statement about the tree now.

  **Retroactively invalid:** any REFUTED whose refuting obligation sat
  inside a `cond`/`switch` branch certified by a probe over a
  declaration of any float dtype other than `float64`. Re-run: such a
  query now returns UNKNOWN unless a real value of that dtype witnesses
  the branch. `float64` declarations are unaffected — under this
  rounding `float64` is the identity — and nothing moves toward
  VERIFIED, because the certificate can only rewrite
  `violated-over-set` into `unknown`.

  **Measured cost**, on a NEW 619-case / 1359-obligation corpus
  (`scratchpad/probe2/`) built for this entry because
  `scratchpad/probe/corpus.py` declares no narrow float and so is blind
  to it. Fourteen declarations, of which nine are narrow floats
  including **a `float32` box wider than `float32`** — the declaration
  the earlier corpus and the audit's own corpus both lack. Scored **per
  obligation** against a numpy/ml_dtypes oracle that never calls
  stelling and that samples MEMBERS *of the declared dtype*:

  | | `688e829` | `62e4190` | with this entry |
  |---|---|---|---|
  | unsound refutation rows | 35 | 31 | **0** |
  | sound refutation rows | 201 | 220 | **220** |
  | rows moved to `discharged` | — | — | **0** |
  | queries moved into VERIFIED | — | — | **0** |

  From `62e4190`: **31** obligations moved `violated-over-set →
  unknown`, every one of them a refutation the oracle finds is evaluated
  at **zero** members of the declared set; **no** obligation moved in
  any other direction, and **0** sound refutations were lost. Unsound
  obligation rows: **31 → 0**.

  The alternative remedy — withhold for every float format narrower than
  `float64` — is also sound and is one line, and it was rejected on
  measurement, not on preference: on the same corpus it reaches 0 unsound
  rows too, but costs **90** sound refutations (220 → 130), every one
  confirmed violated at a real member by the oracle, and the loss is not
  confined to pathological boxes — 13 of the 90 are an ordinary
  `float32 (-1.0, 1.0)` array declaration. Rounding inward costs zero of
  them because it narrows to the real member set rather than abandoning
  it.

  Constructions: `tests/test_probe_witness.py`; pre-registration, corpus
  design and outcomes: `scratchpad/PREREG_PROBE.md`, corpora
  `scratchpad/probe/` and `scratchpad/probe2/`.

- **2026-08-08 (pre-release): an `assume` is a precondition on the WHOLE
  QUERY, not only on the obligations traced after it — the interval
  leg's withholding was order-scoped and is now run-scoped. Verdicts
  move REFUTED → UNKNOWN and in no other direction; nothing moves toward
  VERIFIED and no `discharged` is touched.**

  **The defect.** An `assume` the checker cannot represent exactly is
  DROPPED, and the run records that it was; an `assume` that narrows a
  value whose computed box may exceed its true image leaves the narrowed
  region uncertified. Either way no definite violation of that run is a
  refutation — it may be a refutation of a claim that is vacuously true.
  The interval leg read that fact **at the assert**, so it saw only the
  assumes traced ABOVE the obligation. The same claim under the same
  precondition therefore returned **UNKNOWN with the `assume` written
  first and REFUTED with it written second**. The entry above recorded
  the measured shape: 36 of 36 rows in one survey, 20 of them over a
  precondition satisfied by 0 of 59 269 sampled points.

  Worse, the tree was **already** query-scoped where it could detect the
  assumed region empty — `UnsatisfiableAssumptionError` ends the run
  whole, obligations written above the assume included — and order-scoped
  where it could not. Which behaviour a caller got depended on whether
  they wrote `jnp.all(x >= 2)` or `x >= 2`.

  **The rule, and its argument.** If the assumed region is empty then
  **every** obligation is vacuously true, not only the ones below the
  line — which is exactly why the detectable case ends the run whole. The
  possibly-empty case is the same fact known less precisely, so it takes
  the same scope. The withholding is now applied once, at the end of the
  run, over every obligation and every nonvacuity condition.

  **Where the rule lives, and why not in either leg.** Two legs can mint
  a set-level refutation. The affine refinement was already query-scoped
  — but by ARCHITECTURE, being a post-pass over a finished propagation:
  there is no "during the walk" for it to read from, so it could not have
  been order-scoped if it tried. Left there, the two legs would agree for
  two different reasons, and that agreement breaks silently the first
  time the refinement is restructured to run inline or to interleave. The
  decision is therefore stated **once**, in `stelling.exactness` beside
  `certifies_nonemptiness`, as **`certifies_set_refutation`** — *given
  this run's whole assume state, is a set-level refutation certified?* —
  and both legs consult it. Its signature names no obligation, no
  equation and no position, so no caller can make the answer depend on
  where in the trace it was asked. `tests/test_exactness_lift.py` forces
  that shared answer False and requires **both** legs to withhold, with
  each leg's positive control (a query with no assume at all, which must
  refute on that leg) asserted in the same test.

  **What moving the withholding to the end of the run widened, and what
  closes it — added 2026-08-08; the change landed with this surface
  argued nowhere.** Withholding at the assert wrote `unknown` before the
  obligation reached anything else; withholding at the end writes it into
  the object two later layers read, and both key on exactly that word
  (`affine.refine_propagation` and `solvers.escalate` each take `[o for o
  in propagation.obligations if o.status == "unknown"]`). Obligations
  that used to arrive at those layers as `violated-over-set` now arrive
  as candidates. The one-sidedness argument that shipped with the change
  covers only what `_withhold_uncertified_refutations` itself writes, not
  this. The invariant that does close it: **both layers decline wholly on
  a run with `coverage.constrained`**, so every newly offered obligation
  comes from a run in which nothing was narrowed, where the declared
  boxes those layers judge over ARE the boxes the interval leg judged —
  and a sound layer cannot discharge a predicate found definitely FALSE
  at every point of its own domain, while a re-minted violation meets
  `certifies_set_refutation` again on the affine leg. Measured across the
  two assume corpora (100 harness-runs, jax 0.11.0, cvc5 1.3.4 + z3):
  **24** obligations newly offered to each layer and **34** additional
  solver invocations, for **0** new affine discharges, **0** new affine
  violations, **0** solver-decided outcomes (all 34 returned `unknown`)
  and **0** new VERIFIEDs — 200 verdicts, VERIFIED 14 → 14, with all 48
  verdict moves REFUTED → UNKNOWN.

  The two mechanisms stay **separate flags** — `assume_dropped` and
  `narrowing_uncertified` — rather than one merged "uncertified" bit,
  because the sentence whose job is to explain a withholding may not
  quote the mechanism that did not fire.

  **Which verdicts are retroactively invalid**: any REFUTED on a query
  that dropped an assume, or narrowed an over-approximated intermediate,
  **where the refuted obligation was traced above that assume**. Those
  are exactly the verdicts the old code let through, and the assumed
  region may be empty, in which case the tool told the author their
  correct program was broken. Re-`check()` them: a `violation WITHHELD
  from REFUTED` note where a REFUTED used to be is this change.

  **Cost, measured on this change's own corpus, not inherited.** 23
  harnesses × trace order {assumes first, assumes last, or a fixed
  interleaving} × `refine` ∈ {None, affine} × `assume_mode` ∈ {constrain,
  inert} = 168 runs and **184 obligation-runs**, scored **per
  obligation**, never per query — a corpus in this project once scored per
  query and turned a measured 24:168 trade into a fake 216:216. Ground
  truth is a jax/numpy oracle over the same source — 20 000 uniform
  samples plus every corner plus a 21³ grid (29 269 or 20 023 points per
  case), applying **every** assume of the harness regardless of trace
  position, stelling never consulted.

  | | `discharged` | `violated-over-set` | `unknown` |
  |---|---|---|---|
  | `e8b9377` | 28 | 94 | 62 |
  | this change | **28** | 76 | 80 |

  **18 obligation-runs move, every one `violated-over-set` → `unknown`;
  12 of the 18 are wrong REFUTEDs closed** (the oracle finds **0**
  admitted points — the assumed region is empty and the claim is
  vacuously true) **and 6 are legitimate REFUTEDs lost** (the oracle finds
  admitted points that really violate the obligation: 560, 4 995, 12 507
  and 29 269 of them on the four cases concerned). Nothing moved toward
  `discharged`, in either direction. **These are counts on this corpus of
  this shape, which is built to REACH the defect and so over-samples it;
  none of it is a rate.** The prior figures 92, 168 and 252 came from
  three different corpora and are not comparable with this one or with
  each other.

  **Scope of the 12:6, and what sets it — added 2026-08-08, because
  "not a rate" understated the problem.** 12:6 is not merely
  non-extrapolable; it is a restatement of this corpus's own composition.
  Nine of the 23 harnesses move, six of them with an EMPTY assumed region
  and three with a non-empty one, and each contributes exactly two
  obligation-runs (the two `refine` legs). The trade ratio IS the
  empty:non-empty ratio of the harnesses that move — 2:1 here — and
  nothing in the change fixes it there. Shown by measurement rather than
  by argument, on a SECOND corpus built for this purpose
  (`scratchpad/claims/corpus_b3.py`, jax 0.11.0): not another hand-picked
  list but the full cross product of the four uncertification mechanisms
  this tree has (`jnp.all` reduction, relational, disjunction,
  uncertified narrowing of an over-approximated intermediate) × region
  ∈ {empty, non-empty} × obligation shape ∈ {elementwise, `reduce_sum`,
  affine}, so its empty:non-empty ratio is **1:1 by construction** and
  not by selection; each cell carries one violated and one discharged
  obligation, in both trace orders, both `refine` legs and both assume
  modes; 29 harnesses, **464 obligation-runs**, scored per obligation
  against its own sampling oracle. Result: **32 rows move, all
  `violated-over-set` → `unknown`, 16 wrong REFUTEDs closed and 16 sound
  ones lost** — 1:1, tracking the corpus, with 0 moves toward
  `discharged` and 0 inert-mode differences. Two corpora, two ratios,
  both set by their own designs: neither is the trade's rate, and no
  corpus in this tree can supply one.

  The 6 are the same class the entry above records: nothing at the
  interval level establishes that a dropped conjunct is SATISFIABLE, so a
  genuine refutation over a non-empty region is withheld beside the
  vacuous ones. A "certificate" that recovers most of that class is
  planned and is not this change.

  **Order-dependence, measured with its positive control.** Rows where
  the same case gives different per-obligation statuses with the assumes
  written first and written last: **16 of 38 at `e8b9377`, 2 of 38 now**,
  across both `refine` legs. The controls that keep that from being a
  checker which withholds everything: a query with **no assume** (both
  legs still REFUTE), a **certified** narrowing of a declared input (still
  REFUTES), and a **definitely-true** assume, the audit-F8 channel (still
  REFUTES) — all unchanged, all in the corpus.

  **The 2 residual rows are not this withholding, and are disclosed
  rather than closed.** Narrowing is forward-only (equation order), so an
  `assume` traced after an obligation does not narrow the box that
  obligation is judged over; the corpus row is a CERTIFIED
  `assume(x >= 0.9)` on `x ∈ [0,1]` with `assert_(x <= 0.5)`, which is
  `violated-over-set` with the assume first and `unknown` with it last.
  That direction is the safe one and is sound in both cells: a definite
  violation over the WIDER box is a violation at every point of the
  narrowed region, and "certified" means that region was shown inhabited,
  so the refutation stands; the later-order cell merely judges over more
  and decides less.

  **The residual has a VERIFIED face too, and this entry described only
  its refutation face — corrected 2026-08-08.** The sentence that stood
  here, *"it can cost an UNKNOWN, never mint a verdict"*, is true of the
  mechanism and false as a reader parses it: it reads as "the residual
  cannot touch a VERIFIED", and it can. Same declaration, same certified
  assume, opposite obligation — `x ∈ [0,1]`, `assume(x >= 0.9)`,
  `assert_(x >= 0.5)` — is **VERIFIED with the assume written first and
  UNKNOWN with it written last**, measured in all four `refine` × solver
  cells of `check()` on jax 0.11.0, exactly as `assert_(x <= 0.5)` is
  REFUTED-then-UNKNOWN in the same four. Both cells are sound and neither
  is a wrong VERIFIED: the discharge is the CONDITIONAL claim and the
  stamp says so in itself — *"constrained assume at ...: the verdict
  holds where the precondition holds — narrowed var 2 to [0.9, 1.0]"* —
  and the assume-last cell simply judges over more and discharges less.
  What is true of the residual is that it can only cost precision, in
  BOTH directions; what is false is that a VERIFIED is out of its reach.
  `stelling.harness.assume`'s docstring now states both faces with the
  measurement, as the one positional thing about an assume.

  **Byte-identity where nothing should move.** Every corpus run with no
  assume, every run whose assumes are all certified, and **every** run in
  `assume_mode="inert"` — 110 runs, compared on obligation statuses AND
  details, notes (including their order), stamped assumptions and
  coverage counts: **0 differ**. The guard returns having done nothing.

  At the verdict layer the same 168 runs give **VERIFIED 12 → 12**,
  REFUTED 36 → 20, UNKNOWN 36 → 52: all 16 verdict moves are REFUTED →
  UNKNOWN and no query becomes VERIFIED.

  **Both jax series, measured at `c8d0304`** (the count is a record of
  that commit, not of the tree now, and `--collect-only` at `c8d0304`
  re-run today still gives 2359): 2357 passed / 2 skipped on jax 0.11.0
  and on jax 0.10.2, `--collect-only` ids byte-identical between them
  (2359), and the 168-run ledger is run-for-run identical on the two
  series (0 disagreements). The claim that does not rot is the second
  half — the two series agree with each other, and the ledger agrees
  run-for-run — and that is what a reader should check on any commit.

  Constructions: `tests/test_exactness_lift.py` (the routing pin),
  `tests/test_assume_constrain.py`, `tests/test_vacuous_refutation.py`;
  pre-registration, corpus, oracle and outcomes:
  `scratchpad/PREREG_MECHC.md` and `scratchpad/mechc/`.

- **2026-08-08 (pre-release): a NON-EMPTINESS CERTIFICATE lifts the
  withholding where a point of the declared set satisfies every assume.
  Direction: UNKNOWN → REFUTED, and in no other direction.** The entry
  above withholds every definite violation of a run whose assume state
  does not certify a set-level refutation. That rule is correct and it is
  blunt: it withholds because it cannot rule out that the assumed region
  is EMPTY, in which case every obligation is vacuously true. Where the
  region is demonstrably non-empty the refutation stands, and this change
  demonstrates it.

  **What the certificate is.** One point of the DECLARED SET at which
  every `stelling_assume` of the query is definitely true. It is found by
  re-propagating the query with each declaration pinned to a point of its
  own box, through the machinery the reachability probe already uses
  (`_Propagator.pin`, `_pinned`, `_probe_point`, `_member_bounds`) — so
  the point is a MEMBER of the declared set, a value of the declaration's
  own dtype and not merely a number in the interval. The predicates are
  read at that point in the SAME arithmetic that judged the query, whose
  endpoints are computed in `Fraction` and correctly directed-rounded, so
  a predicate box of `[1, 1]` means true at the point: the box encloses
  the true value.

  **THE DIRECTION, precisely.** A True answer can restore a withheld
  `violated-over-set` and can do nothing else. It reaches the same
  run-level decision the two existing flags reach —
  `exactness.certifies_set_refutation`, now
  `region_inhabited or (nonemptiness_certified and not assume_dropped)` —
  and that function gates violations only. `discharged` is not reachable
  from it, structurally: no code path writes `discharged` under its
  answer. **Retroactively invalid verdicts: none in the VERIFIED
  direction. In the REFUTED direction: any UNKNOWN recorded since the
  entry above, on a query whose assumed region a probed point of the
  declared set satisfies, was weaker than it needed to be** — re-run such
  queries; nothing that was VERIFIED or REFUTED changes.

  **Why lifting is sound.** Both withholding flags withhold for one
  reason and it is not that the judged set is wrong. A narrowing is a meet
  with a CLOSED half-space and a drop only widens, so in both cases the
  judged set is a SUPERSET of the assumed region, and a definite violation
  over a superset is a violation at every point of that region. The one
  missing premise is that the region has a point. Supply one and the
  refutation is a refutation of something.

  **ONE-SIDED, and the failure path is silent.** A False answer means no
  witness was FOUND: the grid is at most 16 points and fewer on a large
  declaration, the arithmetic can be indeterminate, the size cap can
  decline to search at all, and a probe that raises is skipped. It never
  means the region is empty. Every declining path leaves the run
  BYTE-IDENTICAL — no note, no status, no detail, not even a disclosure
  that a search happened — which is what makes the one-sidedness pinnable
  byte-for-byte (`test_a_failed_certificate_search_changes_nothing_at_all`
  compares whole `Propagation` objects) rather than argued.

  **THE PER-OBLIGATION LEDGER**, jax 0.11.0, 32-row corpus, 35
  obligations, oracle 20 000 concrete samples per row
  (`scratchpad/cert/`): **13 refutations recovered**, **0 obligations
  moved toward `discharged`**, **0 left `discharged`**, **0 other moves**,
  **0 recoveries the oracle can falsify**, **0 recoveries on a row where
  the oracle found no admissible point**. At the verdict layer: VERIFIED
  4 → 4, REFUTED 2 → 15, UNKNOWN 26 → 13 — every move is UNKNOWN →
  REFUTED.

  **The zero has a positive control, because a zero without one is
  unfalsifiable.** Two unsound mutants of this build's own machinery, run
  through the same ledger and the same oracle. `two_sided` — let the
  certificate lift a withheld obligation all the way to `discharged` —
  scores **13 toward `discharged` and 13 oracle-confirmed WRONG
  VERIFIEDs**. `certify_everything` — certify every run inhabited whatever
  the probe found — scores **17 recovered, of which 2 are on rows where
  the oracle found no admissible point**, i.e. 2 vacuous refutations
  caught. The instrument sees both failure modes; the real build scores
  zero on both.

  **What protects the other 8 empty rows from `certify_everything` is not
  this certificate**, and it is worth naming: their probes die of
  `UnsatisfiableAssumptionError` at the pinned point, because a relational
  or arithmetic assume that is undecidable over BOXES is decidable at a
  POINT and its meet comes out empty. The certificate is the second line
  there, not the first.

  **The boundary cases, measured — and SCOPED TO THEIR DIAL inside the
  sentence, because the qualifying paragraph is 20 lines below and the
  reader meets the strong sentence first.** At the point `(0.1, 0.2)` the
  predicate `x0 + x1 >= 0.30000000000000004` is TRUE in binary64 and FALSE
  in ℝ; **under `semantics="real"`, and only there,** the box is
  `[0x1.3333333333333p-2, 0x1.3333333333334p-2]`, which straddles the
  bound, so the predicate is INDETERMINATE and **no witness is claimed**.
  An exact-rational checker would answer FALSE there; this answers "not
  established", which withholds. Weaker, never unsound — and never in
  disagreement with the propagation that judged the query, which is the
  reason for running the check in stelling's own arithmetic rather than
  beside it. **Under `semantics="ieee"` the same query CERTIFIES and
  REFUTES**, and that is sound for its own dial: jax executes binary64,
  in which `0.1 + 0.2 == 0.30000000000000004` exactly, so the point
  genuinely satisfies the assume AS EXECUTED. Measured on this tree with
  `x0` and `x1` each declared as their own point — `region_inhabited`
  False and the obligation `unknown` under `real`, `region_inhabited`
  True and `violated-over-set` under `ieee`. The trade runs both ways and
  the other direction is not free either: measured over
  `scratchpad/pin/corpus_pin.py`, **11 rows certify under `real` and not
  under `ieee`** — among them every `float32` and every `int32` row whose
  assume narrows an over-approximated intermediate (4 each) — and **0 the
  other way in that corpus**, the `ieee`-only direction being exactly the
  boundary point above, which the corpus's grid does not contain.
  `sqrt`/`sin`/`exp`/`log` are a boundary and not a gap on the
  same rule: the enclosure at a pinned point has width, a bound inside
  that width certifies nothing (`sqrt` of the point 0.25 against `>= 0.5`
  — exactly true, and INDETERMINATE here), and a bound clear of it
  certifies soundly (`sqrt(x+1) >= 1.2` at `x = 0.5`).

  **AN ASSUME THE PROBE WALKED AROUND IS NEVER CERTIFIED, AND
  BRANCH-SCOPED VIOLATIONS ARE NEVER RESTORED — two different mechanisms,
  and an earlier version of this heading conflated them into the single
  false sentence "branch-scoped assumes are never certified".** The two,
  separately:

  * *the static requirement.* The requirement is the STATIC set of
    `stelling_assume` equations in the IR and the witness is what one
    pinned walk evaluated, so an assume in a branch the probe did not take
    is required and not witnessed. This does **not** decline every
    branch-scoped assume: pinning a declaration FORCES the cond, and
    forcing it can force it EITHER WAY. Counter-construction, measured and
    pinned: a query whose ONLY assume sits inside a `lax.cond` branch is
    certified on probe 1 — the declared box's high corner, which forces
    the branch — with `region_inhabited: True`, the note *"probe point 1
    of the declared set satisfies every assume"*, and the obligation back
    at `violated-over-set`. **The recovery is sound**: at that point the
    program really does take the branch, really does evaluate the assume
    and really does satisfy it, and a 20 000-sample oracle over the
    EXECUTED program (`scratchpad/pin/f2_repro.py`, seed 0) finds
    **20 000 of 20 000** points admissible AND violating — every point of
    the declared box, because `assume(v >= 0.25)` holds wherever the
    `x >= 0.5` branch is taken and is not evaluated anywhere else. What
    was wrong was the safety argument, not the behaviour
    (`test_an_assume_the_probe_walks_INTO_is_witnessed_and_certified`).
  * *the reachability search.* The certificate can only fire on a run that
    narrowed or dropped an assume, on which `_reachability_witnesses`
    returns the empty set (`any_constrained or assume_dropped`) and
    certifies nothing at all — so every branch-scoped violation stays
    withheld by the branch pass however inhabited the top-level region is.
    This one is true, it is the mechanism that actually protects
    branch-scoped violations, and it is independent of the first.

  **What the static requirement COSTS, in the shape the old heading
  claimed it prevented.** A region inhabited only via the UNTAKEN branch —
  every admissible point walks the side WITHOUT the assume, so nothing
  there is required of it — is required-and-not-witnessed on every probe
  and its refutation is withheld. Measured: on
  `assume(v >= 2)` inside the `x >= 0.5` branch of `x ∈ [0, 1]`, 8 of the
  16 probes walk around the assume with an EMPTY witness map, the
  certificate declines, the obligation stays `unknown` — and a 20 000-point
  oracle over the executed program (`scratchpad/pin/f2_repro.py`, seed 0)
  finds **9933 admissible violating points**, which is every `x < 0.5` it
  sampled. A sound refutation, lost to the static requirement. Withholding
  is the safe direction and this is a real price
  (`test_a_region_inhabited_only_via_the_UNTAKEN_branch_is_not_recovered`).

  **THE COST, with load averages** (`scratchpad/cert/RESULTS_cap.txt`,
  jax 0.11.0, load 0.06–0.44). **This search — `_region_witness`, the new
  one — is bounded by the DECLARED SIZE twice over. `propagate.py`'s
  OTHER witness search is not**, and the sentence used to read as though
  the module had one. `_reachability_witnesses`, the branch-reachability
  probe, still runs `for k in range(_PROBE_COUNT)` — 16 whole
  propagations — at any declared size. Measured on this tree
  (`scratchpad/pin/f6_repro.py time`, jax 0.11.0, load 1.18 before and
  1.16 after, a violation inside a `lax.cond` branch), `propagate` against
  a bare walk of the same query:

  | n | propagate | bare walk | ratio | reach probes |
  |---|---|---|---|---|
  | 16 | 1.6 ms | 0.1 ms | 16.1x | 16 |
  | 256 | 9.7 ms | 0.5 ms | 21.0x | 16 |
  | 4096 | 126.6 ms | 6.2 ms | 20.4x | 16 |
  | 16384 | **549.9 ms** | **25.7 ms** | **21.4x** | 16 |

  n = 16384 is four times the size cap below, the probe count does not
  move, and that is the same shape the certificate's cap was added to fix.
  The absolute milliseconds are load-sensitive and the ratio is not: the
  same table taken at load 19, with a concurrent agent on the machine,
  reads 1137.0 ms against 50.4 ms at n = 16384 — **22.6x**, against
  21.4x here. Both runs are in `scratchpad/pin/RESULTS_pin.txt`, because
  a load average printed beside a number is only useful if the number it
  did not suit is shown too.

  **The two are MUTUALLY EXCLUSIVE, so no query pays for both**:
  `_region_witness` gets past its gate only when
  `narrowing_uncertified or assume_dropped`, `narrowing_uncertified` is
  set in the same `if narrowed:` block that sets `any_constrained`, and
  `any_constrained or assume_dropped` is exactly when
  `_reachability_witnesses` returns ∅ before probing. That is a structural
  argument, so it is also measured: **508 propagations** over
  `scratchpad/pin/corpus_pin.py` and a size grid built to reach both —
  including 32 rows that put a branch-scoped violation beside a narrowing
  and a dropped assume — **0 pay for both, worst combined probe count 16**
  (worst for either search alone is also 16). Were they NOT exclusive the
  sum would peak at **32** probes, at small n where the certificate's
  budget is loosest — not at the size cap, where it is 16 + 3. The same
  exclusivity is why they cannot contradict each other: the certificate
  can only fire on runs where the reachability search certifies nothing.
  **The older search is deliberately left uncapped on this branch**,
  because capping it moves verdicts: over 21 branch-violation rows at
  n = 4 … 16384, scored on the keys the branch pass ASKS about rather than
  the ones it happens to find, a `_certificate_probe_count` cap loses
  **3 of 15** — the `x[0] > x[1]` guard at n ≥ 4096, first certified by
  probe index 3 (the plain anchors put every element at the same value and
  cannot witness a relation between two of them) against a budget floor of
  exactly 3 — and each loss is a `violated-over-set` → `unknown` move.
  Safe direction, real cost, measured rather than assumed away. The bounds
  that DO apply, to the new
  search only: a size cap (`_CERT_MAX_ELEMENTS = 4096`) stops it entirely
  above
  it; a probe budget in element-probes
  (`_CERT_PROBE_BUDGET = 4096`, floor `_CERT_MIN_PROBES = 3`) scales the
  probe count down as the declaration grows. The second bound was added
  because the first was not enough: with the size cap alone, a FAILING
  search at n=4096 cost **469 ms against a 23 ms propagation, 95% of the
  whole `check()` pipeline**; with the budget it is **95 ms (4.3x)**. A
  SUCCEEDING search stops at the first witness and costs 3.7x at every
  size. **What the bounds cost in recovered refutations, measured by
  turning each off in turn: the probe budget 0 across n = 64 … 16384; the
  size cap 1 per row above it (2 of the 7 sizes tested).** The floor of 3
  is not a fitted number — probes 0, 1, 2 are the declared box's low
  corner, high corner and midpoint, and across the 17 corpus rows that
  witness at all the first witnessing probe is one of those three in
  **17 of 17** (`RESULTS_probe_index.txt`); one probe alone recovers 18%.

  **WHAT THE CORPUS CANNOT SEE**, stated because five corpora in this
  project have been found structurally incapable of observing the residual
  they reported as zero. (1) `jnp.all(...)` lowers to `reduce_and`, which
  has no interval transfer in either registry, so its predicate is ⊤ at a
  POINT exactly as over a box — **the certificate cannot reach the single
  most common dropped-assume idiom**, and `r11_all_reduction_inhabited` is
  in the corpus to measure that rather than hide it. (2) The corpus's
  assumed regions are half-space-shaped; a region whose only members sit
  off the corner/midpoint grid would need a later probe and is lost at the
  budget floor. (3) The oracle samples and therefore never proves a region
  EMPTY — its two verdicts are existence claims, and "no admissible sample
  in 20 000" is reported as exactly that. (4) The oracle evaluates in
  binary64 while stelling judges in ℝ; the corpus's one deliberate
  boundary row is the only place that could matter and it is scored by
  hand. (5) Independent per-element sampling is useless on a wide
  elementwise assume — `2x >= 0.5` over `[0,1]^64` has admissible
  probability `0.75^64 ≈ 1e-8` — which the oracle's correlated and corner
  fills exist to fix; measured on `r31_wide_declaration`, which the first
  oracle reported as empty.

  **The routing (both legs, one decision).** The certificate is an INPUT
  to `exactness.certifies_set_refutation`, not a channel around it, and
  the per-probe decision is `exactness.certifies_point_witness`, which the
  propagator consults rather than reimplements. Two pins, both new:
  `test_the_witness_route_is_the_shared_primitive_too` forces the witness
  decision in BOTH directions (an inhabited region stops being certified,
  an empty one starts), which an inlined subset test cannot follow; and
  `test_every_reach_of_the_shared_point_names_the_certificate` wraps the
  shared decision in a RECORDER and inspects the keyword arguments, which
  is the first pin in that file that is not argument-blind. Three
  behaviour-preserving mutants, each in its own worktree with `python -B`
  and `__pycache__` cleared, confirm which pin sees what:
  `M2_inline_setref_interval` and `M3_inline_setref_affine` — each leg
  lifting the withholding locally instead of passing the certificate in —
  redden **exactly one test each, and it is that pin**, out of 2396;
  `M1_inline_witness` reddens **six**, its own pin plus the five tests
  that close the certificate's route as a CONTROL while measuring
  something else, and an inlined predicate makes that patch inert. **All
  32 corpus verdicts are identical to the real build under all three**
  (`scratchpad/cert/RESULTS_mutants.txt`) — a pin that checked only
  verdicts would see nothing.

  **Those two mutant NAMES are wrong, and the numbers beside them are
  right.** `M2_inline_setref_interval` and `M3_inline_setref_affine` do
  not inline anything: read
  `scratchpad/cert/apply_mutant.py`, each keeps
  `exactness.certifies_set_refutation(...)` and DROPS the third keyword
  argument, lifting the withholding locally beside a call that still
  happens. That is why they redden exactly one test — the recorder pin,
  which is the only one that looks at arguments — and that count, quoted
  above from an earlier branch, has now been RE-RUN here rather than
  carried: transcribed exactly from that file into
  `scratchpad/pin/mutants.py` and given a worktree each, both score
  **1 failed / 2397 passed / 2 skipped**, and the one failure is
  `test_every_reach_of_the_shared_point_names_the_certificate` on each.
  A GENUINE inlining, the
  call gone and the expression written out, reddens **two** on each leg —
  `test_both_legs_consult_the_shared_set_refutation_point` as well as the
  recorder — measured on this tree at `0ad22bb`: 2 failed / 2396 passed /
  2 skipped for each of the two legs. Read the names as
  *`M2_local_lift_interval`* / *`M3_local_lift_affine`*; the mutants
  themselves are unchanged and their published counts stand.

  **AND EVERY ONE OF THOSE PINS WAS HALF A PIN, which is the larger
  finding and the one that named its own fix.** All of them force a
  shared decision to `False` and require a leg to WITHHOLD. Every
  consumer reads that answer as one operand of an `and`, so a conjunct is
  observable only through the answers it VETOES — and a leg that kept a
  private copy of the rule and wrote
  `shared(<all the real arguments>) and _own_copy(...)` still calls the
  shared function, unconditionally, as the FIRST operand, with everything
  a recorder expects. The forced `False` still makes the conjunction
  `False`, the leg still withholds, and the pin still passes on a leg
  that has stopped obeying. Measured at `0ad22bb`, each mutant in its own
  worktree with `python -B` and `__pycache__` cleared
  (`scratchpad/pin/mutants.py`), against the **whole suite**:

  | mutant | | |
  |---|---|---|
  | `M4_affine_and_private` | the affine leg only | **2398 passed, 2 skipped, 0 failed** |
  | `M5_both_and_private` | both legs | **2398 passed, 2 skipped, 0 failed** |
  | `M6_nonemptiness_and_private` | the same trick on `certifies_nonemptiness` | **2398 passed, 2 skipped, 0 failed** |

  The contrast that names the fix was already in the file:
  `certifies_point_witness` **is** forced BOTH ways, by
  `test_the_witness_route_is_the_shared_primitive_too`, and the same trick
  there is caught. So the GRANTING direction is now pinned on the other
  two decisions as well —
  `test_the_nonemptiness_route_is_pinned_in_the_TRUE_direction` and
  `test_both_legs_follow_the_shared_point_in_the_TRUE_direction`, each
  forcing its decision `True` on a run that would otherwise withhold and
  requiring the leg (both legs, for the second) to STOP withholding. A
  private copy cannot follow a forcing it does not read. Each reddens its
  mutant, in a worktree carrying the mutated source and this file's tests:
  `M4` 1 failed / 16 passed, `M5` **2** failed / 15 passed (a private copy
  on both legs blocks the granting direction on the one-sidedness query
  too), `M6` 1 failed / 16 passed — against **17 passed** on the
  unmutated control tree, and against **14 passed** for every one of them
  under the `0ad22bb` version of the same file.

  A third, `test_the_TRUE_direction_is_ONE_SIDED_too`, closes the
  granting half of the ONE-SIDEDNESS contract — *"a True here can restore
  a withheld `violated-over-set` and can do nothing else at all"*, which
  nothing observed. It runs on a query that genuinely withholds, carrying
  a definitely-false, a definitely-true and a straddling obligation at
  once: forced `True`, the first must come back and the other two must not
  move. It is a weaker finding than the three above and the difference is
  stated rather than smoothed: its mutant (`M7`, a leg reading a granted
  answer as licence to DECIDE) is invisible to the routing file — 14 of 14
  pre-existing tests pass on it — but NOT to the suite, which reddens 2
  tests elsewhere. What it closes is a hole in the pin file, not a hole
  in the tree.

  **A pin that cannot fail in both directions is only half a pin**, and
  the inventory is now: `certifies_nonemptiness` False and True;
  `certifies_set_refutation` False and True, plus the argument-level
  recorder that forces no direction by design; `certifies_point_witness`
  False and True. The four remaining `lambda **k: False` patches of
  `certifies_point_witness` elsewhere in the tree are CONTROLS, not pins —
  each closes the certificate's independent route so a different mechanism
  is observable underneath it — and forcing them True would observe
  nothing.

  **The certificate is LIVE on the affine leg, and it took a second look
  to make it so.** The search's gate first asked only "did the interval
  leg withhold a violation?", which is never true on a query the interval
  leg cannot decide — so on the one class the refinement actually judges
  (`assume_dropped`, `coverage.constrained == 0`) the certificate was
  never computed and the third argument arrived False by construction,
  exactly the documented-dead situation `nonemptiness_certified` is in on
  that leg. Measured: `assume(x >= y)` (relational, dropped, region
  inhabited at `x = y = 0`) with `assert_(x - x >= 0.5)` —
  interval-undecided, affine-violated — returned UNKNOWN from the
  refinement. The gate now also fires on an `unknown` obligation when
  nothing was constrained, which is precisely when the refinement will
  run, and the same query returns REFUTED. Cost: a search on a run whose
  interval leg withheld nothing, 1.4 ms → 30 ms on a 256-element
  declaration, inside the bounds above. Pinned with its empty-region twin
  by `test_the_certificate_reaches_the_affine_leg_as_a_LIVE_argument`.

  **The stamp does not carry a known-false assumption.** Both uncertified
  assumptions say *"the conditional claim may be vacuous"* — true when the
  walk writes them, before any witness exists, and FALSE on a run the
  certificate then settles. On a certified run they are removed and
  replaced by one that states what the claim now rests on (the soundness
  of the transfers at a point, and the probed point's membership in the
  declared set); on a declining run they are untouched. A stamped
  assumption is what a verdict claims to rest on, so leaving a known-false
  one in would be a disclosure defect whatever the verdict said
  (`test_a_certified_run_does_not_stamp_a_known_false_assumption`).

  **The invariant the reading order rests on, measured.** Each assume's
  witness answer is read BEFORE `_assume_constrain` can meet anything into
  the env, and the argument that this suffices is that a `[1, 1]`
  predicate's meet with the closed half-space is a NO-OP — otherwise an
  earlier assume could certify itself AND cut the box a later one is read
  against. Measured across the corpus: **148 certifying probe runs
  inspected, 0 narrowed anything**
  (`scratchpad/cert/RESULTS_invariant.txt`).

  **The two semantics dials are NOT ordered, and the sentence that said
  they were is corrected here.** The check runs in the run's own
  semantics. `sqrt` of the declared point 0.25 is EXACTLY 0.5 in binary64,
  so the ieee transfer encloses it as a point and `>= 0.5` is definitely
  TRUE — while the real-mode transfer bumps outward unconditionally,
  straddles, and certifies nothing: **ieee certifies where real does
  not**. Three other corpus rows go the other way. Both are sound for
  their own dial, which is the whole content of "the same arithmetic the
  query was judged in". **Which dial certifies more is a property of the
  grid you ask, not of the dials**, and the two corpora in this branch
  give different ratios without contradicting each other: the sentence
  above is scored on `scratchpad/cert/corpus.py`, whose interesting rows
  are transcendental, and the F5 paragraph earlier on this page is scored
  on `scratchpad/pin/corpus_pin.py`, whose rows narrow over-approximated
  intermediates — 11 `real`-only and 0 `ieee`-only there, the `ieee`-only
  direction on that grid being the `0.1 + 0.2` boundary point, which the
  grid does not contain. Neither ratio is a rate in any population, and a
  reader who meets one of them first should not read it as the general
  fact.

  **Six existing expectations changed, every one because a withheld
  refutation was CORRECT.** `{x : x ≥ 0.9 ∧ x² ≤ 0.9}` is `[0.9, 0.948…]`;
  `{x ∈ [1,2] : x > 1}` is `(1, 2]`;
  `{a ∈ [0,10]^3, b ∈ [5,6]^3 : 0 ≤ a ≤ b}` contains `(a=0, b=5)`. Each
  site now observes the mechanism it is actually about — the UNCERTIFIED
  flag, the drop discriminant — and keeps its end-to-end consequence with
  the certificate's independent route closed. `MEMBERSHIP_IDIOM_HINT`'s
  sentence *"every definite violation is then withheld from REFUTED"* was
  made true again by naming the exception.

  **Both jax series, measured at `ef41164`** (a record of that commit;
  `--collect-only` at `ef41164` re-run today still gives 2400): 2398
  passed / 2 skipped on jax 0.11.0 and on jax 0.10.2 (140.30 s and
  139.97 s, load 0.59 and 2.85), `--collect-only` ids byte-identical
  between them (2400), `reuse lint` rc=0 — with 317/317 files carrying
  copyright and license information **at that commit, a figure that moves
  whenever anyone adds a file and is not the claim; `rc=0` is**. Baseline at `681c6ef`
  was 2369 / 2 on both: **31 tests added and 2 REMOVED, net +29** — which
  is the 2369 → 2398 delta, and the sentence that said "none removed" did
  not match its own arithmetic. The two removed are
  `tests/test_doc_examples.py::test_doc_example[harness-api.md:614]` and
  `[harness-api.md:660]`, and no example was deleted: `docs/harness-api.md`
  gained 11 lines and lost 2, so both blocks shifted 9 lines and re-entered
  the collection as `[harness-api.md:623]` and `[harness-api.md:669]`,
  which are among the 31 "added". Measured by `--collect-only` id diff
  between `681c6ef` (2371 ids) and `0ad22bb` (2400 ids). Same
  line-number-keyed-id artefact class as `docs/supported-primitives.md`:
  a doc-example id is a file:line pair, so editing prose above a block
  retires one id and mints another.

  **This branch, on top of that.** `fix/shared-point-pin-both-directions`
  adds 5 tests and removes none — `--collect-only` id diff, `0ad22bb`
  (2400 ids) vs this branch (2405), 5 added and 0 removed, so the
  arithmetic and the sentence agree this time. 2403 passed / 2 skipped on
  jax 0.11.0 and on jax 0.10.2 **at `d58e57d`** (a record of that commit;
  `--collect-only` there re-run today still gives 2405), `--collect-only`
  ids byte-identical between the two series, `reuse lint` rc=0. **No verdict moved, scored
  PER OBLIGATION**: `scratchpad/pin/corpus_pin.py`, 95 rows × {real,
  ieee} × {constrain, inert} × {interval leg, affine refinement} plus
  `check()` at `refine=None` and `refine="affine"`, diffed key-for-key
  against a clean `0ad22bb` worktree — **9228 leaf keys compared, 2090 of
  them per-obligation or per-verdict statuses, 0 moved, and 0 non-status
  keys differ** (worktree paths inside `source_info` strings normalised,
  and nothing else). Raw output for every figure on this page that this
  branch added: `scratchpad/pin/RESULTS_pin.txt`.

  Constructions: `tests/test_nonempty_certificate.py` (one-sidedness,
  the `discharged` ledger with its positive control, membership, the
  boundary cases, the dial, the invariant, the stamp swap, the bounds,
  the `lax.cond` counter-construction and its untaken-branch cost twin),
  `tests/test_exactness_lift.py` (both routings, each now forced in BOTH
  directions); pre-registration, corpus, oracle, ledger and outcomes:
  `scratchpad/PREREG_CERT.md`, `scratchpad/cert/` and
  `scratchpad/pin/`.

- **2026-08-09 (pre-release): the index-bounds round — verdicts move in
  the DIRECTION THAT MINTS, and the clamp is deliberately not modelled.**
  `dynamic_slice` and `dynamic_update_slice` gain interval transfers and
  `gather`'s covered row form accepts a range-valued index, so an index
  known only to an interval now produces a box where it produced ⊤.
  **This is the catastrophic direction: it can mint a VERIFIED that did
  not exist before**, and nothing about the round is more important than
  the hull being right. No prior verdict is retroactively invalid —
  every move is UNKNOWN → definite, and ⊤ decides nothing — but a
  re-run is what re-establishes trust in anything that was UNKNOWN
  *because of* an index. `design/index-bounds-round.md` is the full
  record.

  **What was wrong before was power, not soundness.** Measured on
  `9564728`: `u[i]` with a traced `i` collapsed to `[-inf, inf]` whether
  the index was in bounds, partly out, or wholly out, and so did an
  out-of-range static `u[30]`. stelling withheld rather than modelling
  jax's clamp, which was right; it simply withheld everywhere.

  **The measurement the round turns on: `u[i]` is not a gather.** `jnp`'s
  `__getitem__` emits the from-the-end normalisation (`lt`/`add`/
  `select_n`) and then a `dynamic_slice`, on both tested series — and an
  out-of-range STATIC index takes the same path (`u[3]` lowers to a
  static `slice`; `u[30]` and `u[-11]` do not). `dynamic_slice` had no
  transfer at all. Registering the dynamic-index gather alone would have
  closed nothing that scientific code actually writes.

  **THE CLAMP IS NOT MODELLED, and this is a real tension with the
  tree's stated posture, not a free choice.** Measured, primitive-level:
  jax CLAMPS an out-of-range read (`dynamic_slice(arange(10), 30, (1,))`
  reads element 9; start `-1` reads element 0) and DROPS an out-of-range
  scatter write (`x.at[30].set(v)` on a length-10 `x` is a no-op). The
  fixed-width boundary above records this project as *"integers and
  converts are execution-faithful"*, and an index clamp is integer index
  arithmetic — so execution-faithfulness would say model it. **What
  decides it the other way is that there is no single clamp to be
  faithful to, and that is measured rather than argued.** ONE gather,
  ONE out-of-range index, TWO values: index 30 into a 10-element operand
  returns element 9 under `GatherScatterMode.CLIP` and the fill value
  under `FILL_OR_DROP`, while in range all three modes agree. So "the
  clamp" is a property of a param, not of the operation, and modelling it
  means picking one of two answers the same jaxpr can carry. Read and
  write disagree on top of that: the gather clamps, the scatter DROPS.
  An `int32` `add`'s wrap has neither property — one defined,
  reproducible answer — which is why that is modelled and this is not.
  Modelling the clamp would be sound about the executed program and wrong
  about the program the user wrote: the shape of the integer-literal wrap
  defect, one layer over.

  **A THIRD REASON WAS ASSERTED HERE, AND THEN RETRACTED TOO BROADLY.
  Both corrections are recorded rather than quietly deleted, because the
  second made the same mistake as the first.** An early revision of this
  entry, and of the code comment and design page it mirrors, claimed
  *"reverse-mode AD does not preserve the clamp"* — reasoned and not run,
  the failure this project's method exists to catch, committed in the
  middle of a round whose whole subject is not trusting an argument over
  a measurement. The retraction that replaced it said flatly that AD
  *does* preserve it. **That is also wrong, and for the same reason: it
  generalised past what it measured.** It measured `u[30]` (a
  `dynamic_slice`, transposing to `dynamic_update_slice` — both clamp)
  and `.at[k].set(v)` at the default (a `FILL_OR_DROP` scatter,
  transposing to a `FILL_OR_DROP` gather — both drop). Those are the two
  SELF-CONSISTENT pairs. It never built the mixed one.

  Measured on jax 0.11.0 **and** 0.10.2, `JAX_ENABLE_X64=1`: under
  `GatherScatterMode.PROMISE_IN_BOUNDS` XLA's gather CLAMPS and its
  scatter DROPS, and the transpose of a gather is a scatter — so
  `u.at[array([30])].get()` (**the default** for `.at[...].get()`) reads
  element 9 and its cotangent is identically ZERO where the true `d/du₉`
  is `1.0`, and `x.at[30].set(v, mode="promise_in_bounds")` drops the
  write — `f` constant in `v`, true `d/dv = 0.0` — while AD answers
  `1.0`. The modes are part of the claim: the READ half mismatches at
  the default indexing mode, the WRITE half needs the mode spelled out
  (`.at[...].set()` defaults to `FILL_OR_DROP`, whose pair agrees), and
  under `CLIP` both halves agree.

  **So: jax's non-inverse property is real and reproduces at the default
  indexing mode on both series; it does not apply to the
  `dynamic_slice`/`dynamic_update_slice` pair this transfer sits on,
  which clamps symmetrically.** Both halves of that sentence are pinned
  as tests. It remains NOT a reason — the two measured reasons above
  carry the decision, and no code depended on the false version or on
  the true one.

  **The rule therefore computes a value ONLY where jax's clamp is
  provably the identity.** Three cases: an index range inside the axis'
  legal window gets the hull over every start the declared set admits; a
  range straddling the window DECLINES; a range disjoint from it is
  reported as an out-of-bounds FINDING. The control that separates this
  design from the other one is a single query — `u[i] == u[9]` for
  `i ∈ [12, 20]` is TRUE of what jax runs (the test measures that it is)
  and states nothing about the source. A clamp-faithful transfer
  discharges it; this one leaves it undecided.

  **A new note class, and NO new status.** A disjoint index raises
  `interval.IndexOutOfBoundsError`, a subclass of `IntervalError` caught
  one arm ahead of the generic decline. **The accounting is deliberately
  byte-identical to a decline** — ⊤, `record_unknown`, `mark_unreached`,
  and **the finding channel itself never manufactures a status** —
  because an out-of-bounds index does not make an asserted predicate
  false, and minting one from it would claim something the obligations
  do not say. *That is a property of the CHANNEL and not of the
  program*, and the distinction is measured rather than left to be
  misread: a program containing a definite out-of-bounds index can
  perfectly well carry a `violated-over-set`, and four do —
  `assert_(abs(u[30]) < 0)`, `assert_(max(u[30], 5.0) < 0)`,
  `assert_(min(u[30], -5.0) > 0)` and `assert_(exp(u[30]) < 0)` are
  refuted because ⊤ still refutes them, exactly as they would be
  refuted downstream of a plain straddle DECLINE. Checked against
  execution: false at every point the declaration admits, so each is
  sound. The earlier wording here — *"never a REFUTED"* — read as a
  claim about the program and was wrong in that reading.

  Only the note changes, and it is shouted. The old wording explained
  why *stelling* declined and never said the *program* indexes out of
  bounds; `_t_scatter` and
  `_t_gather` had detected this exact fact since their own rounds and
  filed it as a decline. This is the case a jax maintainer asked for in
  Feb 2026 (*"I would rather there be an error for OOB indexing if it's
  statically provable"*); `checkify` is runtime-only and
  `jax_check_static_indices` reaches static constants only.

  **SOUNDNESS EVIDENCE, measured, with a positive control — a zero with
  no positive control has been wrong three times in this project.** The
  oracle enumerates the WHOLE product of declared start ranges over
  randomised shapes and slice sizes, executes the real primitive at every
  one, and checks containment element by element: **6000 configurations
  and ~57 000 executed elements per run, three seeds, on jax 0.11.0 AND
  0.10.2 — 0 containment violations, 0 non-tight configurations.** Five
  deliberately wrong hulls driven through the same instrument produced
  908 / 573 / 439 / 608 / 1073 violations, so its zero is falsifiable.
  **Tightness is pinned separately and on CONCRETE data**, because a
  containment sweep cannot see it: hulling the whole operand would pass
  every soundness check and fail the ramp tests.

  **A blinded independent re-run agreed, on a far larger corpus, and it
  was re-run again against the tree AFTER the fixes below** — the hulls
  are untouched by them (the only `src/` change in the fix pass is a
  comment). Exhaustive over shapes to rank 3 and the whole legal
  start-range lattice, plus rank-3/4 boundary, size-0 and extent-1
  cases, both `dynamic_slice` and `dynamic_update_slice` and the gather
  row form: **454 784 configurations, 19 476 246 executed elements
  judged per position, 0 containment violations and 0 non-tight
  positions on jax 0.11.0 AND 0.10.2**, with the five wrong hulls
  scoring 1962 / 1514 / 900 / 2370 / 1695 through the same instrument
  and the three real hulls scoring 0 / 0 / 0. Independent probes for a
  false VERIFIED under narrow and unsigned index dtypes, wrapping index
  arithmetic, narrowing converts, `jit`, `cond`, `scan`, `fori_loop` and
  chained indexing found none, before or after.

  **ALL THREE HULLS ARE NOW SWEPT IN THE SUITE, not only in a run
  record.** The first version of this entry committed the
  `dynamic_slice` sweep and left the other two hulls this round added —
  `dynamic_update_slice_hull` and `take_row_ranges` — with evidence that
  existed in a log and nowhere a reader could re-run, which is the
  difference between evidence and a claim about evidence. Committed in
  `tests/test_index_bounds.py`, same instrument, judged per output
  position against the real primitive at every admitted start:

  | hull | elements | violations | positive control | violations |
  |---|---|---|---|---|
  | `dynamic_slice_hull` | 994 | 0 | lowest-start-only; exclusive upper | 246; 270 |
  | `dynamic_update_slice_hull` | 3420 | 0 | lowest-start-only; never-keeps-operand | 544; 1144 |
  | `take_row_ranges` | 2431 | 0 | first-reachable-row-only | 688 |

  Two choices in the write sweep are load-bearing and stated so they are
  not silently undone: operand values are drawn from `[-500, 0)` and
  update values from `[1, 500)`, **disjoint**, so that *kept the operand*
  and *took the update* are distinguishable at every position — on
  overlapping data a rule that confuses them passes; and the second
  control is the error specific to that row (take only the update
  wherever any admitted start could write), which the read row's
  controls cannot stand in for because the read row has no such join.
  The row sweep judges against a real `lax.gather` in the covered
  leading-axis geometry rather than against `arr[k]`, which is a
  different lowering.

  **Mutants: nine in the first pass, and three more that only a blinded
  re-run found.** One worktree each, `python -B`, `__pycache__` cleared,
  every mutation LINE-NEUTRAL — a mutant that shifts line numbers can be
  "killed" by the generated `docs/supported-primitives.md` citation
  check, which is an artefact and not coverage. The first pass killed 8;
  `M8_no_index_dtype_gate` SURVIVED because the test covering it drove
  the helper directly and never asked whether the transfer consults it.
  A gate proved correct and never proved wired in; closed with a query
  that goes through the walk. **That lesson had two more instances left
  in this same diff**, both surviving the FULL suite as the branch stood
  at `a5c9659` — 2515 passed / 7 skipped, rc=0, jax 0.11.0:

  * `_ieee_dynamic_update_slice`, `[flags[0] or flags[1]]` →
    `[flags[0]]`. **Reachable and UNSOUND**: a declared operand, a
    ⊤-maybe-NaN update, `dynamic_update_slice`, `assert out ≤ +inf` —
    correct answer `unknown` (NaN falsifies the comparison), mutant
    `discharged`. The scatter set-form's copy of the identical line had
    been pinned since its own round; this row's had not.
  * `_t_gather`, the `_index_dtype_covers_or_decline` call removed.
    Exactly the M8 defect, one function away:
    `test_a_narrow_index_dtype_declines_because_xla_wraps_the_bound`
    passes the string `"gather"` while driving the helper DIRECTLY, so
    the gather case *looked* covered and was not.

  A third, the `dynamic_update_slice` start-index maybe-NaN gate under
  `ieee`, was reported as SUSPECTED equivalent and is not.
  `_classify_index_range`'s finiteness check catches a ⊤ start first,
  which is why it looked equivalent; but `select_n`'s ieee rule ORs in
  every CASE's flag *including cases the selector definitely excludes*,
  so a definite selector picking a finite declared `int32` yields a
  start that is finite, in-window and flagged — `unknown` here,
  `discharged` with the gate gone. **Recorded as DEFENCE IN DEPTH and
  not as a soundness hole**, because on that query an `int32` cannot BE
  NaN so the removed-gate answer happens to be true, and a genuinely
  NaN-able float start declines one step later at the index-dtype gate.
  Wired in for the verdict; defence in depth for soundness. All three
  are now pinned, each shown red under its own line-neutral mutant.

  **SCORED PER OBLIGATION.** Own corpus, 304 keys × {real, ieee}, each
  key carrying an oracle that executes the program at every declared
  index: **81 obligations moved, every one UNKNOWN → definite, every one
  agreeing with the oracle, 0 wrong moves**, 25 out-of-bounds findings
  where the baseline emitted 0. *(Two "wrong moves" in the first scoring
  run were the ORACLE's defect, not the transfer's: it pooled every
  output position into one list instead of judging per position, and so
  called a correct `violated-over-set` wrong on two queries whose LAST
  slice element exceeds the bound at every admitted start. Recorded
  because the instrument being wrong first is the normal case.)** On
  `corpus/supply`: all 20 harnesses **byte-identical** after normalising
  solver timings — those harnesses contain no dynamic indexing, so the
  round buys nothing there and costs nothing.

  **Four expectations changed, every one because a decline was power
  lost**, shown red first: two gather tests whose in-range dynamic index
  now takes a hull (replaced with tests carrying the discrimination the
  old ones could not have), and two whose out-of-range index is now a
  finding with the same accounting.

  **Both jax series, re-measured at `5b6fd89`** — the last commit on this
  branch that touches a test; this entry is its only successor and
  changes no test, so the figures describe a tree that exists:
  **2526 passed / 7 skipped** on jax 0.11.0 and on jax 0.10.2 (167.15 s
  and 169.60 s, load 5.52 and 3.78 at start), `--collect-only` ids
  **byte-identical** between the series (2528 each), `reuse lint` rc=0.
  The baseline at `9564728` was **re-run rather than quoted**: 2484 / 7
  with 2486 ids (160.60 s, load 5.52). **46 tests added and 4 REMOVED,
  net +42** — and 2486 − 4 + 46 = 2528, and 2484 + 42 = 2526, so both
  arithmetics and the sentence agree.

  *Eleven of the 46 are the fix pass*, and one existing test was renamed
  inside the branch: against `a5c9659` the id diff is 12 added / 1
  removed, 2517 − 1 + 12 = 2528, and the run total moves 2515 → 2526.
  The rename is `test_reverse_mode_ad_DOES_preserve_the_clamp` →
  `test_reverse_mode_ad_preserves_the_clamp_for_the_ds_dus_pair`, the
  name being the over-broad claim corrected above.

  *The run total and the collect count differ by 5 on this branch and by
  5 on `9564728` alike* — `2484 + 7 = 2491 = 2486 + 5` there and
  `2526 + 7 = 2533 = 2528 + 5` here. The five are `tests/property/`
  modules skipped at import for a missing optional dependency, which the
  junit report records as testcases and `--collect-only` does not. An
  environment fact, unchanged by this branch, and named here so the two
  figures are not read as a discrepancy it introduced.

  The four removed are the four renamed gather tests
  (`test_gather_dynamic_index_declines_not_crashes`,
  `test_gather_out_of_range_index_declines_not_crashes`,
  `test_fvm_gather_dynamic_index_declines_traced`,
  `test_gather_out_of_range_static_index_declines_traced`); no test was
  deleted, each was renamed to the behaviour it now pins and is among the
  46 "added". *(This sentence said 33 while the paragraph above it said
  35, in the same entry — a stale figure from an earlier revision, and
  the reason both counts are now derived from one measured id diff.)*

  **Known limits, stated rather than left to be re-derived.** Under
  `semantics="ieee"` the from-the-end normalisation declines at its
  integer `add` before the row is reached, so the ieee leg buys nothing
  for jnp-spelled dynamic indexing; the row itself is sound as-is there.
  No SMT emission row, so an obligation reaching one cannot escalate.
  Gather geometries outside the covered row form — batching dims,
  multi-column indices, the `vmap` form, and `jnp.take` (shape-`(1,)`
  indices, not an `(N, 1)` column) — decline exactly as before, while
  `jnp.take_along_axis` along axis 0 DOES reach the widened row form and
  now decides: a capability the round gained without naming it, named
  and pinned in `design/index-bounds-round.md` and
  `tests/test_index_bounds.py`. The
  index-dtype gate refuses an UNCONFIRMED hazard: XLA computes the
  out-of-bounds comparison in the index's own element type, and probing
  `dynamic_slice` with an `int8` start over lengths 100/127/128/129/200
  did not exhibit a wrapped bound — refused anyway, because every dtype
  jnp's own indexing produces is `int32`/`int64` and the gate is free.

- **2026-08-09 (pre-release): the integer-literal wrap is a KNOWN,
  MEASURED, UNCLOSED source-to-trace divergence, and its direction is a
  WRONG VERIFIED.** No verdict moves with this entry and no rule changes.
  It closes a DISCLOSURE gap: the hazard was measured, priced, and left
  open deliberately, and it was described nowhere a reader of THIS PAGE
  could have found it.

  **THE SCOPE OF THAT GAP WAS OVERSTATED WHEN THIS ENTRY LANDED, AND THE
  CORRECTION IS RECORDED RATHER THAN QUIETLY SWAPPED.** The sentence
  here read: *"the shipped tree names this defect once, to explain
  something else, and never says what it is."* Measured at `650e678`,
  `git grep -nEi 'integer[- ]literal wrap|literal wrap' -- .
  ':!SOUNDNESS.md'` returns **12 lines in 9 files** — `ci.yml:936`,
  `CONTRIBUTING.md:30`, `design/index-bounds-round.md:248`,
  `src/stelling/propagate.py:1669`, and eight sites under
  `tests/property/` (`_grammar.py` ×2, `positive_controls.py`,
  `test_metamorphic.py`, `test_oracle.py` ×3, `test_suite_disclosure.py`).
  Every one of those nine paths is inside the sdist allowlist in
  `pyproject.toml`, and `propagate.py` is in the wheel as well, so all 12
  ship. **At least four of them say what the defect IS**, not merely that
  it exists. `tests/property/positive_controls.py:91-93`, verbatim: *"an
  out-of-dtype-range integer literal wraps mod 2\*\*bits before tracing, so
  stelling returns VERIFIED for a predicate that is false at every declared
  point"*. `tests/property/test_oracle.py:123-125` says the same in its own
  words (*"…so stelling verifies a predicate that is false at every
  declared point"*), `:17` gives the mechanism and why an execution oracle
  cannot see it, and `_grammar.py:32` names the mechanism again. *(Those
  first two anchors were written `:91` and `:123`, which is where each
  quoted string STARTS; the quoted words run on to `:93` and `:125`. A
  grep hit is the first line of a wrapped string literal, and citing it
  alone points a reader at a third of the sentence being quoted.)* That
  grep is a FLOOR, not a census: a wider pattern
  (`out-of-dtype-range|wraps mod 2|wrapping before tracing`) finds
  **13 lines in 5 files**, including `tests/property/README.md:165` —
  and it is NOT a superset. Measured at `b2e3a15`, the two greps share
  only four lines (`_grammar.py:32`, `positive_controls.py:91`,
  `test_oracle.py:17` and `:123`); their union is **21 lines**. So the
  wider pattern is a different net, not a bigger one, and "finds more" is
  true only of the count.

  **What IS true, measured the same way, is the narrower claim this entry
  should have made**: at `53f9f84` the same grep run against `SOUNDNESS.md`
  alone returns **one** line — `SOUNDNESS.md:3843`, inside the index-clamp
  entry above, reaching for *"the shape of the integer-literal wrap
  defect, one layer over"* as an ANALOGY for a different decision. The
  page a reader consults for soundness disclosures named it once, in
  passing, to explain something else. The test suite and CI said what it
  was; this page did not. It is being written down before 0.1.0 because a
  release is where an omission stops being recoverable.

  **The defect, in four lines of plain jax and no stelling idiom.**

  ```python
  OFFSET = jnp.full((), 256, jnp.int8)   # jax wraps 256 -> 0 HERE
  @jax.jit
  def shift(v): return (v + OFFSET).astype(jnp.float32)
  x = any_array((), "int8", (0, 10));  assert_(shift(x) <= 10.0)
  ```

  `x + 256 ∈ [256, 266]`, so the predicate AS WRITTEN is false at all 11
  declared points. **stelling returns VERIFIED.** Re-driven at `53f9f84`
  before this entry was written, in all four cells — jax 0.11.0 and jax
  0.10.2, `jax_enable_x64` on and off — and VERIFIED in every one; driven
  again at `650e678`, same four cells, `vacuity_mode="inputs-only"`, both
  interval-only and with the solver portfolio at 20 s, VERIFIED in all
  eight. The index-bounds transfer that landed at `53f9f84` does not touch
  it; the reproducer contains no indexing.

  **It is a wrong VERIFIED, not a lost one.** That is the expensive
  direction, and it is the direction this defect has.

  **The VERIFIED is not a blanket VERIFIED for this harness shape**, and
  that is the control rather than an inference. Same three lines, same
  declared box, only the literal changed, jax 0.11.0 (re-derived at
  `650e678`, interval-only and with the solver portfolio at 20 s, same
  four rows either way):

  | `OFFSET` literal | what jax traces | verdict |
  |---|---|---|
  | `256` — wraps | `0` | **VERIFIED** (source-false at all 11 points) |
  | `5` — no wrap | `5` | UNKNOWN |
  | `0` — no wrap | `0` | VERIFIED (and source-true) |
  | `-1` — no wrap | `-1` | VERIFIED (and source-true) |

  stelling reads the traced constant faithfully in every row. It returns
  VERIFIED at `256` for exactly the same reason it returns VERIFIED at
  `0`: by the time it looks, those are the same program.

  **Why no backward-cone rule closes it: the wrapped value is
  indistinguishable from an honest one by the time stelling sees it.**
  The wrap happens inside `jnp.full`, at eager time, before the harness is
  traced at all — and, per the mechanism section below, at a `.astype`
  cast that leaves no trace of the value it narrowed. Re-derived at
  `b2e3a15` in all four cells, via `stelling.harness.trace` and
  `ir.ClosedJaxpr.to_dict(include_metadata=False)`, jax's default
  configuration: **the transcribed tree for the wrapped `256` is
  byte-identical to the tree for an honestly written `0`** — and the same
  comparison against an honestly written `5` differs, so the comparison is
  live and not vacuously true. `0` is the third row of the control table
  above, where stelling returns VERIFIED and the source is TRUE. The two
  rows are one program, exactly, at the level any rule could read them.
  That is the fact the conclusion below rests on.

  **THE "NO INTEGER LITERAL IN THE CONE" READING OF THAT IS A FACT ABOUT
  ONE SPELLING OF THE REPRODUCER, NOT ABOUT THE DEFECT, AND IS NARROWED
  HERE.** For the reproducer exactly as written above, the figure holds
  and re-derives in all four cells: the ENTIRE jaxpr tree holds exactly
  **one** `ir.Literal` — `10.0:f32`, the comparison bound — the string
  `256` appears nowhere in it, and the wrapped value enters as a
  **constvar closed over by the `jit` sub-jaxpr**, printed by jax as
  `{ lambda c:i8[]; a:i8[]. … }`. But that is a fact about `OFFSET` being
  CLOSED OVER by the jitted function, and two respellings that change
  nothing about the defect put the wrapped value into the cone as an
  ordinary integer literal — both measured in all four cells at `b2e3a15`:

  * pass `OFFSET` as an ARGUMENT to the jitted function instead of closing
    over it, and the tree holds **two** `ir.Literal` operands, the second
    being the wrapped value as an `int8` `0` operand of the `jit`
    equation; there are then no constvars anywhere (top-level `consts=()`,
    sub-jaxpr `constvars=[]`), and jax prints the sub-jaxpr
    `{ lambda ; c:i8[] a:i8[]. … }` with the value at the call site as
    `] 0:i8[] a`;
  * or leave the reproducer alone and set jax's own transitional
    `jax_use_simplified_jaxpr_constants` (default `False` in **both**
    installed series, and carrying jax's warning that it "will exist only
    briefly, while we transition users. DO NOT RELY ON THIS FLAG"), and
    the closure spelling itself inlines the wrapped value as `0:i8[]` into
    the `add`.

  So a rule cannot be sold as safe on the ground that the value is not
  there to be seen: in two of the three configurations measured it IS
  there. What does not change across any of them is the equality above —
  wrapped `256` and honest `0` transcribe to the same tree in all four
  cells, in both spellings, with the flag and without it.

  Four families of jaxpr-level remedy were built and priced against a
  purpose-built corpus and the full suite on both series — a
  literal-immediate rule, a narrowing rule, a backward-cone rule and a
  declaration-scoped rule, plus a cross-scope cone and an index-only
  exemption. Every one is either blind at `jit`/`cond`/`custom_jvp`
  boundaries (which is universal in real jax code), or costs capability
  that is not the wrap's to take — including `u.at[3].set(0.5)`, the
  commonest jax write there is. *(That pricing is RESTATED here from the
  measurement round that did it, not re-run for this entry. Its third
  clause read "or structurally blind to the constvar form above"; that
  clause is WITHDRAWN, because the constvar form is one spelling's and the
  measurement above finds the value in the cone under the other two. What
  IS re-measured here is the tree equality, which is the fact the
  conclusion rests on.)* **No rule keyed on integer literals in a backward
  cone can close this** — not because there is nothing to key on, but
  because what there is to key on is the same `0:i8[]` an honest program
  writes, so a rule that fires on the wrapped row fires on the honest one
  too. That is the same SHAPE of failure as the one that got the detector
  branch audited SHOULD-NOT-LAND, below — firing on honest code — and it
  is stated as a shape, not as an identification of the two.

  **THAT REASON IS TRUE OF THE REPRODUCER'S SHAPE AND IS NOT TRUE OF THE
  CLASS, AND SINCE THE ENTRY HAD BY THEN BEEN KNOWN WRONG THREE TIMES IN
  EXACTLY THIS WAY THE NARROWING IS RECORDED RATHER THAN LEFT.** *(Three
  was the count KNOWN when this was written — the eleven-doors list read
  as a census, the "concrete" row, and the `except OverflowError` route.
  A fourth was already on the page and had not been found yet: `jnp.full`'s
  route through the guarded call, which entered at `d5cfd60` — the commit
  before this paragraph — and was corrected below only after a later pass
  measured it. So three is what the author knew, not what the page held,
  and the difference runs in the direction that flatters. No total is
  claimed here, because which sentences belong to the class is a
  judgement and a number would read as a measurement.)* At two of the
  eight doors that wrap — `jnp.where(c, 256, x)` and
  `jnp.clip(x, 256, 256)`, measured in all four cells and detailed under
  the fourth site below — the traced program is NOT what an honest
  program writes: the constant survives as a live `256:i32[]` /
  `256:i64[]` operand with the narrowing still standing as a
  `convert_element_type` equation, where an honest `0` would put `0` in
  its place. So "there is nothing to key on but sameness" is a fact about
  the `jnp.full` spelling this entry reproduces in, not about every door.
  What is withdrawn is sameness as the reason that covers all of them.

  **AND THE GROUND THIS ENTRY PUT UNDER THE CONCLUSION IN ITS PLACE IS
  ALSO FALSE, MEASURED, SO THE CONCLUSION IS NARROWED HERE RATHER THAN
  RE-FOUNDED ON A SECOND ASSERTION.** What was written was: *"The
  conclusion above is not withdrawn, because at those two doors it rests
  on the OTHER clause already stated — the difference is inside a nested
  `jit` sub-jaxpr, which is where every priced remedy was blind — and not
  on sameness."* Re-derived with `jax.make_jaxpr` in all four cells, at
  both doors, against an honest `0` written in the same place: **the
  nested `jit` sub-jaxpr — jax's own `_where` and `clip` — is
  BYTE-IDENTICAL between the wrapped `256` and the honest `0`.** The
  whole difference is at the CALL SITE, in the ENCLOSING jaxpr, and jax
  prints it there: at `jnp.where` the `jit` equation's operands read
  `] b 256:i32[] a` against `] b 0:i32[] a` (`i64[]` at
  `JAX_ENABLE_X64=1`), and at `jnp.clip` the same substitution happens
  twice. Nothing has to be descended into to see it.

  What IS inside that sub-jaxpr is the NARROWING — the
  `convert_element_type` equation and its `int8` target, one at
  `jnp.where` and two at `jnp.clip`, with **zero** such equations in the
  enclosing jaxpr, all four cells. That is what the paragraph on the
  fourth site below already says correctly, and it is the honest form of
  what "nested" is true of here: at top level the operand reads as an
  in-range `int32`/`int64` `256`, and the dtype that makes it wrong sits
  one level down.

  **That does not re-establish the conclusion, and is not offered as
  doing so.** The `jit` equation's OUTPUT aval — `int8[3]` — is in the
  enclosing jaxpr too, so whether a rule could key on the call-site
  literal together with that output is a question no measurement here
  answers; no such rule was built and none was priced. **So the
  conclusion is narrowed to the spelling it was measured on**: it holds
  through `jnp.full`, where wrapped and honest transcribe to the same
  tree, and at `jnp.where` and `jnp.clip` it is recorded as UNMEASURED
  and is not claimed. No remedy is proposed on the strength of any of
  this, none was built, and none was priced.

  **It is jax's, it is deliberate, and it is in the shipping release —
  BUT NOT BY THE MECHANISM THIS ENTRY FIRST NAMED, AND THAT CLAIM IS
  RETRACTED HERE RATHER THAN QUIETLY SWAPPED.** What this paragraph said
  was that `jax/_src/lax/lax.py` "wraps the narrowing conversion in
  `try: ... except OverflowError: pass` — jax catches the overflow NumPy
  raises and discards it", carrying the comment *"TODO(phawkins): remove
  the try-except block here, which would be a breaking change to users in
  the presence of overflows"*. **The comment is accurate and so are the
  line ranges. The causal claim built on them is false.**

  **THAT LEFT THE WORD "DELIBERATE" STANDING ON A COMMENT BESIDE A BLOCK
  THAT DOES NOT RUN FOR THIS CASE. IT NOW RESTS ON A COMMIT AND A TEST,
  WHICH ARE BETTER EVIDENCE AND ARE ABOUT A DIFFERENT PATH.** Both read
  in a clone of jax's own repository at tag `jax-v0.11.0`, commit
  `a1521744c6dc074443fe549f19f48d7197abf759`, working tree clean:

  * **The test.** `tests/lax_test.py:201-203`, entire:

    ```
    def testConvertElementTypeOOB(self):
      out = lax.convert_element_type(2 ** 32, 'int32')
      self.assertEqual(out, 0)
    ```

    It does not assert that jax raises. **It asserts the wrapped value.**
    *(That fence is deliberately unlabelled. This page holds exactly one
    ```` ```python ```` fence — the reproducer above — and
    `tests/test_soundness_wrap_reproducer.py` asserts that count before
    reading it, so a second labelled fence turns **15** cases red —
    measured by planting one. *(It was **nine** when that figure was
    written; the tripwire has since grown a second door, and six more
    cases read the fence.)* Quoted foreign code goes in a bare fence.)*

  * **The commit that put it there.** `c2fe350455` (Jake VanderPlas,
    2023-04-04), subject *"future-proof lax.convert_element_type"*, body
    *"In the future, np.array(large_value, 'int32') will error"*. Its
    whole diff is two files, five inserted lines and one deleted: it
    changed `operand = np.asarray(operand, new_dtype)` — which raises —
    to `operand = np.asarray(operand).astype(new_dtype)` — which
    truncates — inside the `type(operand) is int` fast path, **and added
    `testConvertElementTypeOOB` in the same diff**. Verified: the commit
    is an ancestor of the 0.11.0 tag, and the expression it introduced is
    the one standing today at `lax.py:1726` (**0.11.0**) and `lax.py:1724`
    (**0.10.2**) — the line the table below names as where `jnp.full`
    loses the value, and therefore the line this entry's own four-line
    reproducer dies on. *(The statement AROUND that expression has been
    reshaped since — `408bd93e3d` rewrote the `TypedNdArray` construction
    — so the commit created the EXPRESSION, not the line as it now
    reads.)*

  **Which path each applies to, said explicitly, because this entry has
  twice been wrong by taking something true of one door for something
  true of the class.** The commit and the test are about the
  `type(operand) is int` fast path at `lax.py:1726`/`1724`. The
  `TODO(phawkins)` quoted above, at `lax.py:1747-1748` on the block
  `1747-1754`, is about the guarded call BELOW that fast path — the one
  measured below to execute zero times for this case, whose live routes
  are a Python float and an `int` subclass. Two paths, one recorded
  intent each, pointing opposite ways: the `TODO` records an intent to
  REMOVE a swallow, the commit records a choice to KEEP a truncation and
  pinned it with a test that fails if it is undone.

  **What this does NOT support, in either direction.** Not "jax will
  never fix this" — the commit's own message anticipates NumPy starting
  to error, and the `TODO` says its block should go. Not "jax is about to"
  — nothing was found that says so, and nothing has been reported
  upstream from here. What it supports is narrower and is all the entry
  claims: on this one path the truncation was chosen over a raise, on
  purpose, and it is pinned. **That it is the ONLY test in the jax repo
  asserting a wrapped value is REPORTED by the measurement context whose
  receipts are cited below, and is not re-derived here** — proving a
  negative over a test suite is not something this entry did. What WAS
  re-derived at that tag: of all **17** mentions of `OverflowError`
  anywhere under `tests/`, **15 assert that it is RAISED** (14
  `assertRaises`/`assertRaisesRegex` calls plus one continuation line of
  a fifteenth) and the remaining two are comments saying an out-of-bounds
  value leads to one. **None of the 17 asserts a value.** And
  `api_test.py:8351` `test_integer_overflow` is parameterised over six
  entry paths ×
  `jtu.JIT_IMPLEMENTATION`, running as **10** cases after its own
  exclusion clause, every one of which asserts `OverflowError`.

  Measured by instrumenting the code object of
  `jax._src.lax.lax._convert_element_type` with `sys.monitoring` LINE
  events LOCAL to that one code object, then sweeping the eleven doors
  below × four constant spellings (Python `int`, `np.int64`, 0-d
  `np.ndarray`, 0-d `jnp` array) × seven written values × four narrow
  target dtypes: **1144 runs at `JAX_ENABLE_X64=0` and 1232 at `=1` per
  series, `_convert_element_type` entered on 607 and 655 of them
  respectively, and the `except OverflowError:` line executed ZERO
  times** — jax 0.11.0 and jax 0.10.2, both x64 settings, all four cells,
  NumPy 2.5.1. **The instrument carries its own positive control**: in the
  same process the same probe sees that line execute exactly once for
  `jnp.full((), 1e308, jnp.int8)` and for
  `lax.convert_element_type(1e308, jnp.int8)` — a Python FLOAT out of the
  target integer's range, whose answer is `127`, saturated, not wrapped.
  So the zero is a fact about the integer path, not about the probe.

  **Where the value is actually destroyed**, per door class, by
  line-execution count inside that same code object. Line numbers are
  installed-dependency figures and carry their version; the constant is a
  bare Python literal:

  | door class | the line that destroys the value | does anything raise there? |
  |---|---|---|
  | `jnp.array`, `jnp.asarray`, `jnp.int8` | nothing in `lax` — `_convert_element_type` is **entered 0 times**. jax runs an explicit overflow check first, at `jax/_src/numpy/array_constructors.py:249-250` (same line numbers in both series), which calls `dtypes.coerce_to_array`, whose `return np.asarray(x, dtype)` — `dtypes.py:478` at **0.11.0**, `dtypes.py:507` at **0.10.2** — is what raises | yes — that IS the raise |
  | `jnp.full`, `jnp.full_like` | `arr = np.asarray(operand).astype(new_dtype)`, on the `type(operand) is int` fast path — `lax.py:1726` at **0.11.0**, `lax.py:1724` at **0.10.2** | no: `.astype` is a cast, and truncates in silence |
  | `x + 256`, `x >= 256`, `jnp.where`, `jnp.clip`, `jnp.maximum` | the `256` has already been promoted to a weakly-typed `int32`/`int64` **Tracer** by the time it arrives (`JitTracer(~int32[])` / `~int64[]`); no Python-level fast path applies, so the entry falls through to `convert_element_type_p.bind` and the narrowing is the primitive's own | no |
  | `x.at[0].set` | same fall-through, but the operand is **concrete** — `ArrayImpl(256, dtype=int32/int64, weak_type=True)` | no |

  **That last split was one row until now, reading "already promoted to a
  CONCRETE `int32`/`int64` array" for all six; per-entry tracing at
  `b2e3a15`, all four cells, says concrete at `x.at[0].set` and a Tracer
  at the other five.** It changes no conclusion — both take `bind` — and
  it is corrected because the row is the section's own evidence and a
  reader reasoning from "concrete" would reason about a value that is not
  there to read.

  The `x.at[0].set` row is where the entry's own earlier grouping went
  wrong, and it is worth naming because it is the trap this whole section
  is about: that door DOES execute `np.asarray(operand).astype(new_dtype)`
  once, so a line-count alone puts it in the `jnp.full` row — but per-entry
  tracing shows the operand on that entry is the INDEX `0` (`int -> int32`)
  and the operand carrying `256` is a separate entry,
  `ArrayImpl(int32/int64 256) -> int8`, which takes the `bind` path. A
  count of line hits is not an attribution.

  *(A second thing a count is not: a repeatable figure. The attribution
  above is a FIRST-CALL measurement, taken with one door per fresh
  interpreter. jax caches its traces, so the same door called again with a
  constant of the same jit signature enters `_convert_element_type` ZERO
  times — measured at `b2e3a15`, and stated here because an entry count
  taken second in a process is a fact about the cache, not the door.)*

  **THERE IS A FOURTH SITE — WHERE `x + 256` ACTUALLY DIES — AND NOTHING
  ELSE ON THIS PAGE NAMES IT.** The third row of the table
  above ends at `convert_element_type_p.bind` and says the narrowing is
  "the primitive's own". That is true, and it is not an address. The
  address is `_convert_elt_type_folding_rule` in `jax/_src/lax/lax.py` —
  the primitive's CONSTANT-FOLDING rule, registered into
  `pe.const_fold_rules` and called from `try_constant_folding` in
  `jax/_src/interpreters/partial_eval.py` — and the line inside it that
  destroys the value is `out = out.astype(new_dtype)`: **`lax.py:5314` at
  jax 0.11.0** and **`lax.py:5304` at jax 0.10.2** (the function's `def`
  at `5295` and `5285` respectively). Read from the two installed trees,
  and the 0.11.0 file is byte-identical to the same path in a clone of
  jax at tag `jax-v0.11.0`.

  Measured in all four cells by substituting a recording wrapper for that
  rule: `jax.jit(lambda v: v + 256)(jnp.zeros((3,), jnp.int8))` returns
  `[0 0 0]`, and the rule is called with the constant still INTACT —
  `TypedInt(256, dtype=int32)` at `JAX_ENABLE_X64=0`, `int64` at `=1` —
  with `new_dtype=int8`, returning `TypedNdArray(0, dtype=int8)`. That is
  a per-call attribution, not a line count: the 256 goes in whole and
  comes out zero, there.

  **It is disjoint from the `.astype` site above, in both directions.**
  Measured differentially in-process at 0.11.0, `JAX_ENABLE_X64=1`, by
  installing range-checking wrappers one at a time: a check on the
  `type(operand) is int` `.astype` site makes `jnp.full((), 256,
  jnp.int8)` and `lax.convert_element_type(2 ** 32, 'int32')` raise and
  leaves `jit(x + 256)` returning `[0 0 0]`; a check on the folding rule
  makes `jit(x + 256)` raise and leaves those two returning `0`. The
  other two sites are already established above as not on this path at
  all: the `except` executes zero times for it, and the
  `array_constructors.py:249-250` gate belongs to the three construction
  doors, which `x + 256` is not one of.

  **AND THIS IS EXACTLY THE DISTINCTION THIS ENTRY KEEPS GETTING WRONG,
  SO IT IS SPELLED OUT: THIS SITE IS NOT WHERE THIS ENTRY'S OWN
  REPRODUCER DIES.** The four-line reproducer at the top writes its
  constant through `jnp.full`, so it dies at the `.astype` site, and a
  range check THERE fixes it — measured, both `x64` cells, with only that
  check installed: `jnp.full((), 256, jnp.int8)` raises while
  `jit(x + 256)` still returns `[0 0 0]`. `x + 256` is a DIFFERENT door
  of the eleven, reached by writing the constant inline instead, and it
  is the one no fix at the three sites above reaches. Same defect, same
  wrong VERIFIED, two narrowing sites — and a remedy priced against one
  of them buys nothing at the other.

  **It is not a LAST site either, and it is not offered as one.** Two
  measurements bound it, both in all four cells. First, it is reached
  only through TRACING: evaluated EAGERLY, all eight doors that wrap give
  the same answers they give under `jit`, and the folding rule is called
  **zero** times across all eight — so eager code loses the constant
  somewhere else again, unmeasured here. Second,
  two of those eight do not reach it even under `jit` —
  `jnp.where(c, 256, x)` and `jnp.clip(x, 256, 256)` cross into jax's own
  `jit`-wrapped internal `_where` / `clip`, so the constant survives into
  the jaxpr as a live `256:i32[]` / `256:i64[]` operand with the
  narrowing left standing as a `convert_element_type` EQUATION, and the
  answer is still `0`. **In that shape the source's own constant is
  present and unwrapped in the traced program, and no Python line
  performs the narrowing at trace time at all.** That is recorded as
  existing, not analysed: no verdict was driven through it, and it is not
  offered as something to detect on — the equation sits inside a nested
  `jit` sub-jaxpr, which is exactly where the priced remedies above were
  already blind. It does not disturb the tree-equality measured further
  up, which is about the reproducer as written, through `jnp.full`.

  **That check at `array_constructors.py:249-250` is the ONLY explicit
  overflow check jax runs on a constant, and its gate is a Python
  `isinstance`:** `if isinstance(object, (bool, int, float, complex)):`.
  Respell the constant as `np.int64(256)` — how a value read out of a
  NumPy table arrives — and a NumPy scalar is none of those four types, so
  **the CHECK is skipped entirely: `array_constructors.py:250`, the body,
  executes 0 times, all four cells.** *(This sentence carried the range
  `249-250` and said "that line is skipped entirely". Re-measured with
  `sys.monitoring` LINE events local to `array_constructors.array`'s own
  code object: line **249**, the `isinstance` gate, executes **once** —
  it has to, to decide — and line **250** executes **zero** times. The
  substance was right and the range was not, and on a page that cites
  jax's source by line the two are not the same claim. Control, same
  probe, same four cells: the bare Python literal `256` executes BOTH
  lines once, `OverflowError` propagates out of 250, and 314 is never
  reached.)*
  The value falls through to
  `out = np.asarray(object, dtype=dtype)` at `array_constructors.py:314` —
  same line number in both series — where NumPy CASTS it. Traced line by
  line, `out` is `array(0, dtype=int8)` on the very next line executed. By
  the time `lax._convert_element_type` is called, from
  `array_constructors.py:338`, its operand is **already an `int8` `0`**;
  the `arr = operand.astype(new_dtype, copy=False)` at `lax.py:1731` that
  0.11.0 then runs is a no-op on an already-narrowed array, and 0.10.2,
  which has no such branch, reaches the identical answer. Directly above
  that call, at `array_constructors.py:310-313` in **both** series, jax
  carries its own comment about it: *"falling back to numpy here fails to
  overflow for lists containing large integers … More correct would be to
  call coerce_to_array on each leaf, but this may have performance
  implications."* It is also where `jnp.array([256], int8)`
  raises — NumPy applies its Python-int bound check per element — which is
  why a Python list of those numbers raises and the NumPy array of the same
  numbers does not.

  **THAT COMMENT IS STALE, AND THIS ENTRY QUOTED IT AS THOUGH IT
  DESCRIBED CURRENT BEHAVIOUR.** *(It was also cited as being ON line
  314; it is on 310-313, annotating the call at 314. The entry has spent
  a commit on a citation off by one line before now.)* Measured in all
  four cells at NumPy **2.5.1**, every list spelling the comment is about
  now RAISES rather than overflowing silently: `jnp.array([256], int8)`,
  `[[256]]`, `[1, 256]`, `[np.int64(256)]` and `(256,)` all raise
  `OverflowError: Python integer 256 out of bounds for int8`, and
  `[2**100]` raises `OverflowError: Python int too large to convert to C
  long`. **What still goes through line 314 in silence is a NumPy ARRAY
  or a NumPy SCALAR, which is not what the comment is about**:
  `jnp.array(np.array([256]), int8)` returns `[0]` and
  `jnp.array(np.int64(256), int8)` returns `0` — the two spellings the
  table further down already lists as wrapping. So the comment is quoted
  here as jax's own record OF the site, and no longer as evidence about
  what the site does. *(Reported at NumPy 2.5.2 by the measurement whose
  receipts are cited below; re-derived here one patch version earlier, at
  NumPy 2.5.1, on both jax series and both `x64` settings. What it would
  do at some third NumPy is not a fact this page has.)*

  `jnp.full` / `jnp.full_like` take a third route again. With an
  `np.generic` constant they destroy it **inside the `try` itself**, at
  `np.asarray(operand, dtype=new_dtype)` (`lax.py:1750` at 0.11.0,
  `lax.py:1743` at 0.10.2): measured, that line executes twice and the
  `except` beneath it zero times, because under NumPy 2.5.1 `np.asarray`
  CASTS an `np.generic` rather than raising. At the NumPy level in the same
  interpreters: `np.asarray(256, dtype=np.int8)` raises `OverflowError`,
  while `np.asarray(np.int64(256), dtype=np.int8)` and
  `np.asarray(256).astype(np.int8)` both return `0` with no exception and
  no warning. **That is the correction at its sharpest: the one door class
  whose wrap really does happen inside the guarded call is the class for
  which NumPy declines to raise, so the `except` beneath it has nothing to
  catch.**

  **The block is not dead code, and this entry does not claim it is.**
  It executes for the Python float above, and for an `int` SUBCLASS — an
  `enum.IntEnum` member, or any `class MyInt(int)` — because the fast path
  above it tests `type(operand) is int` EXACTLY, so a subclass skips it,
  reaches the guarded call, NumPy raises, jax swallows it, and `256`
  becomes `0`. Measured in all four cells:
  `lax.convert_element_type(Colour.RED, jnp.int8)` with `Colour.RED = 256`
  executes the `except` once and returns `0`. That route is real and is
  the one the `TODO` describes.

  **AND TWO OF THE ELEVEN DOORS BELOW TAKE IT. THIS PARAGRAPH SAID THEY DO
  NOT, AND THAT IS CORRECTED HERE RATHER THAN QUIETLY SWAPPED.** It read:
  *"It is not the route any of the eleven doors below takes."* Measured at
  `b2e3a15`, `sys.monitoring` LINE events local to that same code object,
  in all four cells: with `class Colour(IntEnum): RED = 256`,
  **`jnp.full((), Colour.RED, jnp.int8)` and `jnp.full_like(x, Colour.RED)`
  each execute `except OverflowError:` — `lax.py:1753` at 0.11.0,
  `lax.py:1746` at 0.10.2 — exactly once, and return `0`**; identical for
  a bare `class MyInt(int)`. Per-entry attribution confirms the swallowed
  operand IS the constant (`Colour(<Colour.RED: 256>) -> int8`), not some
  other entry of the same call. The constant class is the one this entry
  introduces one sentence earlier, so the route and the doors were never
  disjoint populations; the sentence simply had not been driven through
  them. The other nine are unchanged: the three construction doors RAISE
  for an `int` subclass (it passes the `isinstance(object, (bool, int,
  float, complex))` gate), and the six remaining doors never reach the
  guarded call. *(INFERRED, not measured, and now narrowed by the above:
  because the `except` body runs on none of the 1144/1232 runs — whose
  four spellings do not include an `int` subclass — deleting the `except`
  clause alone cannot change any of THOSE; but it WOULD change the two
  `int`-subclass rows just measured, which would raise instead of
  returning `0`.)*

  **THE REST OF THAT INFERENCE HAS NOW BEEN DRIVEN, AND IT IS FALSE.** It
  ran on: *"and deleting the whole `try` block would additionally change
  the `np.generic` rows, whose wrap is the `try` body."* The second half
  is right; the first does not follow from it. Measured in all four cells
  by editing `_convert_element_type`'s source IN MEMORY — read from the
  installed file, re-`exec`'d against the module's own globals, and
  re-bound onto `jax._src.lax.lax`, which is where `lax.full` looks the
  name up; no installed tree is modified — **with the whole `try`
  STATEMENT deleted, body
  included, `jnp.full((), np.int64(256), jnp.int8)` still returns `0`.**
  The narrowing simply moves to the fall-through
  `convert_element_type_p.bind`, which narrows the same way. **The
  instrument carries its own control**: planting a `RuntimeError` — which
  `except OverflowError` does not catch — as the block's first statement
  makes that same call raise in all four cells, so the edit is live and
  the body really is on that path, while the bare-literal
  `jnp.full((), 256, jnp.int8)` is untouched by it and still returns `0`.

  **AND THE BLOCK IS FREE TO DELETE, WHICH IS A DIFFERENT FACT FROM
  WHETHER DELETING IT WOULD HELP.** Removing the `try/except` outright
  changes **0** of the 23,705 jax test cases in the cost measurement
  below — the joint-cheapest of the four candidate fixes priced there,
  and the only one of them that covers the `int`-subclass route measured
  just above, because it is the only one ON that route. **It also
  reaches neither narrowing this entry is about**, so neither the
  reproducer at the top nor the inline spelling changes under it.

  **THE REASON GIVEN FOR THAT WAS WRONG AT ONE OF THE TWO DOORS, AND SO
  WAS THE LINE THAT FOLLOWED IT. IT IS THE SAME MISTAKE THIS ENTRY KEEPS
  MAKING AND IS RECORDED, NOT SWAPPED.** What it read was: *"it is not on
  the path `x + 256` takes and not on the path `jnp.full` takes … Cheapest
  to land, and it shuts the door nobody came through."* `x + 256` is
  right — that dies at the folding rule. **`jnp.full` is a fact about the
  BARE PYTHON LITERAL, written as a fact about the door** — and the
  `np.generic` paragraph above (*"`jnp.full` / `jnp.full_like` take a
  third route again … they destroy it inside the `try` itself"*) says the
  opposite, on the same page, in the entry's own words. Measured with
  `sys.monitoring` LINE events local to `_convert_element_type`, all four
  cells: `jnp.full((), np.int64(256), jnp.int8)` — how a value read out of
  a NumPy table arrives — executes `try:` → `np.asarray(operand,
  dtype=new_dtype)` → `return stage(x)` (`1749`/`1750`/`1752` at
  **0.11.0**, `1742`/`1743`/`1745` at **0.10.2**) and loses the value at
  `1750`/`1743`, INSIDE the block, because `np.asarray(np.int64(256),
  dtype=np.int8)` is `0` and raises nothing. The bare literal reaches none
  of those lines: the `type(operand) is int` fast path at `1725`/`1723`
  takes it first. **So somebody does come through that door, and it costs
  a verdict**: the four-line reproducer with `jnp.full` left exactly as
  written and `OFFSET`'s constant supplied as `LUT[7]` (i.e.
  `np.int64(256)`) returns **VERIFIED — source-false at all 11 declared
  points — in all four cells**, interval-only, no solver invoked.

  **What survives is the conclusion, on the other measurement rather than
  on that reason.** Deleting the block does not change that row either —
  measured just above, the answer stays `0` — so "it reaches neither
  narrowing this entry is about" holds, but because the fall-through
  narrows identically, NOT because the block is off the path. Cheapest to
  land; and what it changes is the `int`-subclass rows and — measured in
  all four cells under the deletion the receipts actually applied, which
  drops the `try:`/`except` and KEEPS the body — the Python-float route,
  where `lax.convert_element_type(1e308, jnp.int8)` raises instead of
  saturating to `127`. Neither is the route this entry's wrong VERIFIED
  comes through. *(That is a different edit from the whole-statement
  deletion measured above, and the two do not have the same effect: with
  the body kept, the float route raises; with the statement gone
  entirely, it saturates to `127` as it does today.)*

  **The line range is an installed-dependency figure, not a repository
  figure**, so it is quoted with the version it was read from:
  `lax.py:1747-1754` at jax **0.11.0**, and the same block at
  `lax.py:1740-1747` at jax **0.10.2** — both re-read and both correct.
  Cite the comment, not the range; and do not cite either as the cause of
  the integer wrap.

  The silent narrowing is against jax's own published promotion rule. **JEP
  9407** states as a design goal *"Promotion should never lead to an
  unhandled overflow."* **SUSPECTED, and labelled so deliberately**: JEP
  9407 is not shipped inside the `jax` distribution, so the wording is
  restated from the published JEP and could not be re-verified offline.
  Grepping both installed trees for the sentence returns nothing; the
  digits `9407` do match, in three files per tree, and every match is
  noise — inside float literals in the SVD back-compat test data and
  inside the unrelated bug number `TODO(b/278940799)` in
  `jax2tf.py`. No reference to the JEP is present in either tree. All of
  that re-derives at `650e678` against both installed trees. The
  `try/except` above IS PRESENT in both — verified by reading; what it does
  and does not do is measured above.

  **And jax is inconsistent about which door raises**, which matters
  because it means no remedy can be scoped by "where jax wraps" without
  being scoped by an unstable surface. Measured at `53f9f84` and
  re-measured at `650e678` on both series, `int8`, **the constant written
  as a bare Python literal `256` in every cell of both columns** —
  identical on jax 0.11.0 and jax 0.10.2:

  | raises `OverflowError` | wraps in silence |
  |---|---|
  | `jnp.array(256, int8)` | `jnp.full((), 256, int8)` |
  | `jnp.asarray(256, int8)` | `jnp.full_like(x, 256)` |
  | `jnp.int8(256)` | `x + 256`, `x >= 256` |
  | | `x.at[0].set(256)` |
  | | `jnp.where(c, 256, x)` |
  | | `jnp.clip(x, 256, 256)`, `jnp.maximum(x, 256)` |

  The three that raise delegate to `np.asarray(..., dtype)` and inherit
  NumPy's check — re-measured, and the only part of the original sentence
  that survives. The rest of it read *"the eight that wrap go through the
  `try/except` above"*; **not one of them does**, and where each of them
  actually loses the value is the table further up.

  **ELEVEN IS A SAMPLE, NOT A CENSUS, AND THIS ENTRY NEVER SAID SO.** It
  does not claim the eleven are exhaustive, but a reader counting doors
  will read a table for a list. Measured at `b2e3a15`, `int8`, bare Python
  literal, identical in all four cells, eight more that wrap to `0` in
  silence and are named nowhere above: `jnp.arange(256, 257, int8)`,
  `jnp.pad(x, 1, constant_values=256)`, `jnp.select([c], [256], x)`,
  `x.at[0].add(256)`, `jnp.minimum(x, 256)`, `lax.full((), 256, int8)`,
  `jnp.astype(jnp.asarray(256), int8)`, `jnp.ones((), int8) * 256`. They
  are recorded as more of the same shape, not as a new one, and the count
  is still a FLOOR.

  **ONE OF THEM IS A DIFFERENT SHAPE, AND THE ENTRY'S "WRAPS MOD 2\*\*bits"
  FRAMING DOES NOT DESCRIBE IT AT ALL.** Measured at `b2e3a15`, all four
  cells: **`jnp.linspace(256, 256, 1, dtype=jnp.int8)` returns `127`. It
  SATURATES; it does not wrap.** So does `jnp.linspace(300, 300, 1, int8)`
  (`127`), and `−300` clamps to `−128`; at `int16`, `70000` clamps to
  `32767` while `256` passes through untouched. That is a SECOND
  source-to-trace divergence with a different arithmetic, and every
  sentence on this page that says "wraps mod `2**bits`" is about the first
  one only. Its direction is not stated here, because it was not measured:
  a clamp moves a constant toward zero-magnitude rather than around the
  ring, so whether it costs a VERIFIED or a REFUTED depends on the
  obligation, and no reproducer for it was driven. **What is claimed is
  only that it exists, is silent, and is not the shape this entry
  describes.** The saturating direction is not new to the page — the
  instrument's positive control above saturates `1e308` to `127` — but
  that is a Python FLOAT, and this is an out-of-range INTEGER going the
  same way, which nothing above accounts for.

  **THE "BARE PYTHON LITERAL" QUALIFIER ON THAT TABLE IS LOAD-BEARING AND
  WAS NOT THERE WHEN THE TABLE LANDED.** Per the mechanism above, jax's
  explicit check fires only for `isinstance(object, (bool, int, float,
  complex))`, and the fallback at `array_constructors.py:314` is what
  raises for a Python list or tuple of such scalars — measured, both
  `jnp.array([256], int8)` and `jnp.array([256.0], int8)` raise there. So
  the raise is a joint fact about the door AND the argument's Python type,
  and the table,
  pinning one literal in both columns, reads as a fact about the function
  alone. Measured at `650e678`, three doors × thirteen
  spellings of the value 256, target `int8`, **identical in all four cells
  (jax 0.11.0 and 0.10.2 × `JAX_ENABLE_X64` 0 and 1): 15 raise, 24 wrap
  silently to `0`, and the split is the same at all three doors** — it
  tracks the argument, not the door:

  | at `jnp.array`, `jnp.asarray` AND `jnp.int8` | argument spellings of 256 |
  |---|---|
  | raises `OverflowError` | `256`, `256.0`, `[256]`, `(256,)`, `[[256]]` |
  | wraps to `0`, in silence | `np.int64(256)`, `np.int16(256)`, `np.float64(256.)`, `np.array(256)`, `np.array([256])`, `np.array([256], np.int16)`, `jnp.array(256)`, `jnp.array([256])` |

  Neither factor alone predicts the outcome, and both directions were
  driven: `jnp.array(int(LUT[7]), int8)` RAISES where
  `jnp.array(LUT[7], int8)` wraps for the same `LUT[7]`; and the very same
  Python `int` that raises at `jnp.array` wraps at
  `jnp.full((), int(LUT[7]), int8)` and at `x + int(LUT[7])`. A whole
  table goes through quietly: `jnp.array(LUT, int8)` with
  `LUT = np.array([1,2,3,4,5,6,7,256], np.int64)` returns
  `[1 2 3 4 5 6 7 0]`, while `jnp.array(LUT.tolist(), int8)` raises. **Not
  one of the 24 wrapping combinations emits a warning of any kind** under
  `warnings.simplefilter("always")`.

  **Driven end to end, this is the same wrong VERIFIED and not a smaller
  one.** The four-line reproducer at the top of this entry, with `jnp.full`
  replaced by each of the three doors and `OFFSET`'s constant supplied as
  `LUT[7]` (i.e. `np.int64(256)`), returns **VERIFIED — source-false at all
  11 declared points — through `jnp.array`, through `jnp.asarray` and
  through `jnp.int8`**, in all four cells, at `650e678`, `vacuity_mode=
  "inputs-only"` with the solver portfolio at 20 s. With the bare literal
  the same three doors raise before a harness exists. The door did not
  change; the constant's provenance did.

  **A DECLARED BOX SOMETIMES CATCHES A WRAPPED `assume` BOUND, AND
  NARROWNESS IS NOT WHAT DECIDES WHETHER IT DOES.** What catches one is
  the wrapped bound landing OUTSIDE the declared box. Where it lands is a
  fact about the wrap's arithmetic, not about the box, and it is not
  something the user can see. Driven at `53f9f84` and re-driven at
  `650e678` on both series with `jax_enable_x64` on and off — four cells,
  identical in all four, obligation
  `assert_(x.astype(jnp.float32) <= 10.0)` throughout, **the `assume`
  bound written as a bare Python literal**:

  | declaration | `assume` written | traced as | outcome |
  |---|---|---|---|
  | `int8 (-10, 10)` | `x >= 300` | `x >= 44` | **raises** `UnsatisfiableAssumptionError`; no verdict emits |
  | `int8 (-128, 127)` | `x >= 300` | `x >= 44` | **REFUTED** |
  | `int8 (-10, 10)` | `x >= 261` | `x >= 5` | **VERIFIED** |
  | `int8 (-10, 10)` | `x >= 30` | `x >= 30` | **raises** `UnsatisfiableAssumptionError` |

  Row 3 is a CONTROL and it is the row that matters: `261` wraps to `5`,
  which is INSIDE the narrowest box in the table, so nothing refuses and
  stelling returns VERIFIED over `[5, 10]` — a region the source's own
  `assume` makes empty, since no `int8` is `>= 261`. **A narrow
  declaration does not protect against a wrap that lands inside it.** Row
  4 is the second control: the refusal in row 1 is not wrap-specific at
  all, since an unwrapped `x >= 30` outside `(-10, 10)` raises the same
  error by the same mechanism.

  Row 2 corrects what this paragraph said on the branch it was extracted
  from, which asserted that the full-range box "returns VERIFIED".
  Measured, it returns REFUTED. The verdict at `(-128, 127)` is whatever
  the obligation happens to be over `[44, 127]` — `<= 10.0` REFUTED,
  `>= 10.0` VERIFIED, `<= 300.0` VERIFIED — so no direction can be stated
  for that cell at all. That is the same reason narrowing cannot be sold
  as a guard: neither cell's outcome is a property of the declaration.

  **The bound's SPELLING moves this table too, and in the opposite
  direction from the doors table, which is why neither spelling can be
  recommended.** Re-run at `650e678` with the identical four rows and the
  bound written `np.int64(300)` / `np.int64(261)` / `np.int64(30)` instead
  of the literal, all four cells: `x >= np.int64(300)` does NOT wrap — jax
  widens `x` to `int32`/`int64` and compares against `300:i32[]`/`300:i64[]`
  — and every one of the four rows then raises
  `UnsatisfiableAssumptionError`, including the row that returned VERIFIED
  with the literal. **The two ends run opposite.** At the three
  construction doors the bare Python literal is the spelling that RAISES
  and `np.int64` is the spelling that wraps in silence; at an `assume`
  bound it is the bare Python literal that wraps and `np.int64` that
  refuses. No spelling is the safe one at both ends, so no spelling can be
  recommended, and this entry recommends neither.

  **WHAT IT WOULD COST JAX TO FIX THIS WAS MEASURED, AND THAT BELONGS ON
  THIS PAGE BECAUSE IT IS WHAT DECIDES HOW LONG A READER LIVES WITH IT.**
  It is also the honest answer to "why has nobody fixed it". Four
  candidate fixes, each applied alone to jax's source and
  run against jax's own test suite: jax at tag `jax-v0.11.0`
  (`a1521744`) as source against the installed jaxlib 0.11.0 wheel,
  CPython 3.12.3, NumPy 2.5.2, two tranches of 28 and 36 test modules,
  **23,705 test cases**, every figure from `--junitxml`. Three of the
  fixes were run over both tranches; the fourth was not, and the table
  below carries a denominator per row because of it.

  **That suite run is ANOTHER CONTEXT'S MEASUREMENT and is cited, not
  claimed** — 23,705 cases is not a figure this entry re-ran. Its
  receipts (report, reproduction facts, 946 run artefacts, scripts,
  transcript) are committed at `receipts-jax-wrap-blast-radius/` in
  `stelling-sweeps`, **a SEPARATE repository held beside this one and not
  shipped with it** — so a reader of the distribution cannot open it, and
  neither can a reader of THIS repository. That is stated rather than
  papered over: the figures below are only as good as a receipt the
  reader may not have.

  *(The contrast drawn here when this citation landed was with "every
  `scratchpad/` path cited elsewhere on this page", as though those were
  openable from the distribution. **They are not.** `/scratchpad` is not
  in the sdist allowlist in `pyproject.toml` — `tests/test_sdist_contents.py`
  withholds it deliberately, and nothing force-includes it — so not one
  of the **30 distinct `scratchpad/` paths** this page cites is openable
  from a distribution either. The true contrast is with a reader of the
  REPOSITORY: `git ls-files scratchpad` is **80** files, all 30 cited
  paths resolve inside it, and these receipts are in neither the tree nor
  the tarball.)*

  What WAS re-derived here, and can be
  re-derived from those artefacts by anyone who does have them, is every
  delta in the table below — recomputed from the archived junit XML with
  an independent analyser comparing per-testcase outcomes between the
  baseline and patched runs. All of them agree with the report.

  | candidate fix | site it changes | newly failing | of |
  |---|---|---|---|
  | widen the `isinstance` gate to admit NumPy scalars/arrays | `array_constructors.py:249` | **no-op** — see below | — |
  | leave the gate, add a real range check for them | `array_constructors.py:249-250` | **0** | 23,705 |
  | delete the `try/except OverflowError` | `lax.py:1747-1754` | **0** | 23,705 |
  | range-check the Python-`int` narrowing and the NumPy fallback | `lax.py:1726` **and** `array_constructors.py:314` | **1** | 23,705 |
  | …and additionally the constant-folding site | `lax.py:5314` | **1031** | **16,798** |

  *(The last two rows are cumulative: the fourth-site row is the fourth
  row's patch PLUS the folding rule. The source report labels the fourth
  row "the two `.astype` sites"; read from its patch script, only one of
  the two is an `.astype` — the other is
  `out = np.asarray(object, dtype=dtype)`. The sites are as given here.)*

  ***"One per site" was wrong and is dropped from the sentence above.***
  Read from the receipts' `patch.py`, which is what was actually applied:
  rows 1 and 2 patch the **same** site — both anchor on the same two
  lines at `array_constructors.py:249-250`, one replacing the
  `isinstance` tuple and the other adding an `elif` beside it — and row 4
  patches **two** sites in one patch, `lax.py:1726` together with
  `array_constructors.py:314`, as its own cell already says. So the rows
  are not in one-to-one correspondence with the sites in either
  direction, and no count of sites can be read off them. The entry
  self-corrects a sentence later to "the four candidate fixes **priced**
  there", which is the accurate word for a second reason: of the five
  rows, four carry suite figures and row 1 carries none — it was proved
  inert and given no run.

  **THE LAST ROW IS NOT OF 23,705, AND THIS TABLE'S HEADER SAID EVERY ROW
  WAS.** Recounted here from the archived XML: tranche 1 is **16,798**
  counted `testcase` elements across 28 modules, tranche 2 is **6,907**
  across 36, the two module lists are disjoint, and 23,705 is their sum.
  Both tranches are archived for the gate-plus-range-check, the
  `try/except` deletion and the two-site fix — 64 XMLs each. **The
  folding-site patch has 28 XMLs and no tranche 2: it was never run for
  it**, and the source report's own fix table records that with a dash.
  So the `1031`, and the `37` it collapses to below, are **of 16,798**,
  and neither may be read against the full denominator or set beside the
  `0`/`0`/`1` rows as if it were. The receipts' corrections block now
  records this too.

  **The first row's zero is not a suite figure and is not offered as
  one.** Widening the gate changes nothing at all, because the check it
  opens does not check the values it would newly admit: measured directly
  here, all four cells, `dtypes.coerce_to_array(np.int64(256),
  jnp.int8)` returns `array(0, dtype=int8)` **without raising**, while
  the same call on the Python `int` `256` raises — NumPy 2's NEP 50
  range-checks Python integers and casts NumPy ones. The patch was proved
  inert and given no suite run, so its "0" means "changes nothing",
  not "breaks nothing".

  **The `1` is the pinned test.** The single case that range-checking the
  narrowing breaks is `lax_test::LaxTest::testConvertElementTypeOOB`, with
  `OverflowError: Python integer 4294967296 out of bounds for int32` —
  the test commit `c2fe350455` added, failing for exactly the reason it
  was written. That fix at `lax.py:1726` is, read against that commit's
  diff, its revert for integer targets. **It is a policy reversal, not a
  bug fix**, and that is the sharpest thing on this page about why the
  defect is still here.

  **The `1031` — of 16,798, per the note above — is not 1031 opinions,
  and it is the figure that decides
  the answer.** 1030 of them carry one message — `Python integer
  4294967295 out of bounds for int32`, `0xFFFFFFFF` narrowed to `int32` —
  and it comes from jax's own PRNG: `jax/_src/random/threefry2x32.py:73`,
  `k2 = convert(jnp.bitwise_and(seed, np.uint32(0xFFFFFFFF)))`, relying
  on two's-complement mask reinterpretation. Making that one line
  explicit collapses 1031 to **37**, of the same 16,798: 36 in
  `random_test`, the identical
  idiom in the three other PRNG implementations (`philox2x32.py:149`,
  `philox4x32.py:160`, `threefry4x32.py:213` — all four lines re-read at
  the tag), and the 37th is `testConvertElementTypeOOB` again. **The true
  cost of the only fix that reaches `x + 256` is four lines of jax's own
  source plus one deliberately pinned test.**

  **Read the rows against the doors, not against "the defect", because
  they do not all buy the same thing.** The `1`-cost fix reaches the
  narrowing this entry's own reproducer dies on, `jnp.full` — measured
  above, with only that check installed it raises. The `1031`-cost fix is
  what it takes to reach `x + 256` as well. Neither covers the other, so
  there is no single row here that is "the price of closing this"; there
  are two prices for two doors, and the cheap one leaves the entry's
  eleven-door table mostly untouched.

  **What that licenses, and what it does not.** It does not license "jax
  will never fix this": four lines and a test is not a large bill. It
  does not license "jax is about to": nothing was found that says so, and
  nothing has been reported upstream from here. It licenses exactly one
  thing, which is the thing a reader needs — **the cost is known, it is
  small, and it is unpaid; and a VERIFIED over a narrow integer
  declaration is unprotected for as long as it stays unpaid.**

  **Four things those figures are not.** The denominator is emitted test
  cases, not jax's suite: 64 of the 164 top-level `tests/*_test.py`
  modules ran, and `pallas`/`mosaic` and GPU/TPU never did. *(That list
  said "and multi-device" too, and **that is false**: `multi_device_test`
  is one of the 36 tranche-2 modules — 20 cases, 19 passed, 1 skipped —
  and so are `pmap_test` (289 cases, 206 passed, 83 skipped) and
  `shard_map_test` (452 cases, 438 passed, 14 skipped), all counted here
  from the archived baseline XML. `pallas`/`mosaic` and GPU/TPU genuinely
  did not run: no module of either name appears in either tranche. A
  limits list that names something as unmeasured when it was measured
  understates the run in the direction that flatters it, which is the
  direction to correct first.)*
  It is also not the XML's own `tests` attribute, which totals 16,982
  against tranche 1's 16,798 counted `testcase` elements. The `1031`
  counts cases that PASSED at baseline and fail under the patch; the
  archived XML shows two further cases moving from skipped to failed,
  which that delta does not count. And none of it was measured on jax
  0.10.2, so no row above is a four-cell figure.

  **NOTHING IN THIS TREE CONSULTS ANY DIAGNOSTIC FOR THIS.** There is no
  detector on `main`, no stamp field, no note, no verdict gate, and no
  count that changes because of this entry. **A VERIFIED over a narrow
  integer declaration is exactly as trustworthy as it was before this
  entry was written** — which is the reason for writing it. A detector
  was built on a branch and audited SHOULD-NOT-LAND, for producing
  CONFIRMED findings on ordinary honest code (`jnp.zeros(256, jnp.int8)`
  — a byte-indexed LUT, the canonical `int8` idiom) while filing the real
  hazard as safe; it is not on `main` and this entry does not depend on
  it. A reader must not read this disclosure as the announcement of a
  guard.

  **What a user can do today. NOTHING BELOW IS A GUARD, AND THE VERSION OF
  THIS LIST THAT LANDED WITH THIS ENTRY SAID OTHERWISE.** It read: *"One
  thing here is a guard and the rest are odds. The guard: keep out-of-range
  constants out of narrow integer dtypes by construction — `jnp.array`,
  `jnp.asarray` and `jnp.int8` RAISE, measured above, and are the doors to
  prefer."* **THAT IS WITHDRAWN.** Measured above: those three doors raise
  for a Python scalar and wrap in silence for a NumPy scalar, a NumPy array
  or a jnp array, and the reproducer driven through each of them with the
  constant arriving as `LUT[7]` returns the same VERIFIED, source-false at
  all 11 declared points, in all four cells. A reader who followed that
  advice would have moved the wrap rather than removed it, and would have
  been told they were protected while doing so. That is the worst thing
  this page could do, so the retraction is written here rather than the
  sentence deleted.

  **No replacement guard is offered, because the search for one came back
  empty, and the search is stated so it can be re-run.** At `650e678`, all
  four cells: (i) not one of the 24 wrapping door × spelling combinations
  emits a warning of any kind under `warnings.simplefilter("always")`, and
  none becomes an exception under `-W error`; (ii) wrapping
  `np.errstate(all="raise")` around each of the eleven doors changes
  nothing; (iii) a `FutureWarning` out of `x.at[k].set(np.int64(...))` —
  *"scatter inputs have incompatible types: cannot safely cast value from
  …"* — and it is a DTYPE-class warning, not a value one. Its control: it
  fires identically for `np.int64(3)`, which is in range and does not
  wrap, and is silent for `np.int8(3)`. Escalating it would flag ordinary
  in-range code — the same failure that got the detector branch audited
  SHOULD-NOT-LAND — and would still see none of the other ten doors. It is
  not a wrap detector and is not offered as one.

  **CLAUSE (iii) SAID THAT WARNING WAS THE ONLY CATCHABLE SIGNAL ANYWHERE
  IN THE ELEVEN. IT IS NOT, AND SINCE THIS PARAGRAPH INVITES THE READER TO
  RE-RUN THE SEARCH, THE CORRECTION IS RECORDED HERE.** The words removed
  above were *"the only catchable signal found anywhere in the eleven is"*.
  Measured at `b2e3a15`, all four cells: under
  `jax_numpy_dtype_promotion="strict"` — a jax setting, not a stelling one,
  default `"standard"` — **six of the eleven doors raise
  `TypePromotionError` for the NumPy-scalar spelling**: `x + c`, `x >= c`,
  `x.at[0].set`, `jnp.where`, `jnp.clip`, `jnp.maximum`. That is a second
  catchable signal, and the search as stated does find it.

  **It is not a wrap detector either, and its control is the same
  control.** Measured in the same four cells: it fires identically for the
  in-range `np.int64(3)` (6 of 11, same doors), is silent for `np.int8(3)`
  (0 of 11), and — the direction that decides it — is silent for the bare
  Python literal `256` (0 of 11), which is the spelling that wraps at
  eight of the eleven doors. So it separates DTYPES, not values: it flags
  honest in-range code and misses the wrap the entry's own reproducer is
  written in. Strict promotion is worth its own reasons; it does not close
  this, and no replacement guard is offered.

  What is left buys no protection, and is written as what it is:

  * Treat a VERIFIED over a narrow integer declaration as a statement
    about the program jax TRACED — which is what `Floats are judged in ℝ;
    integers and converts are execution-faithful` above already says it
    is. This lowers nothing; it names what the verdict was ever about.
  * **NOT a guard:** a narrower declaration raises the chance that a
    wrapped `assume` bound lands outside the box and gets refused, but row
    3 of the table above is a wrap that lands inside the narrowest box
    there and returns VERIFIED. Narrowing a declaration is worth doing for
    its own reasons; it does not close this.
  * **NOT a guard, and recorded here to foreclose the inference the doors
    table invites:** forcing a constant through Python `int()` does restore
    the raise at `jnp.array` / `jnp.asarray` / `jnp.int8`, because it opens
    the `isinstance(object, (bool, int, float, complex))` gate at
    `array_constructors.py:249` — measured, `jnp.array(int(LUT[7]), int8)`
    raises where `jnp.array(LUT[7], int8)` wraps. It does nothing at the
    other eight doors, which never reach that gate: the same Python `int`
    wraps in silence through `jnp.full((), int(LUT[7]), int8)` and through
    `x + int(LUT[7])`, both `0`. It protects exactly the call sites
    somebody remembered to write it at, which is the property a guard does
    not have.

- **2026-08-14 (0.2.0 development, unreleased): FALSE VERIFIED — a
  forwarded relational `assume` was resolved by a bare integer, and a
  bare integer does not say which scope it is an id in.** An `assume`
  whose two sides both vary cannot be applied in the interval domain, so
  since 0.2.0 it is forwarded to the solver as a positive axiom. The
  propagator recorded the assume's producing comparison equation, whose
  operands are `ir.Var`s **in whatever scope the assume was traced** —
  inside a `jit` or `custom_jvp` body, that body's ids. `smt.emit` then
  resolved them with `names.get(atom.id)`, an integer lookup with no
  scope check, against a table keyed by the **slicer's renumbered** ids.
  The slicer allocates its fresh ids from `max(top-level ids) + 1`
  (`obligation.py`), a maximum taken over the top-level jaxpr only, so a
  sub-jaxpr's ids can land exactly where `_fresh()` allocates. A foreign
  id then resolved to an unrelated term and **the axiom was emitted about
  the wrong values.** A fabricated conjunct can only shrink the model set,
  so its failure direction is `unsat` → discharged → **VERIFIED**, and
  nothing downstream withholds a discharge.

  **Measured, and it is the converse of what the user wrote.** Harness:
  two `float64` declarations in `[-10, 10]`, `assert_(x - y <= 0.0)`, and
  `assume(p < q)` inside a jitted helper called as `side(y, x)` — i.e.
  the precondition `y < x`, under which `x - y > 0` at every admitted
  point. The emitted script asserted `(< x0 x1)`, which is `x < y`.
  Exhaustive 201×201 grid over the declared box: **20100 admitted points,
  20100 violations of the assert**, and stelling returned VERIFIED.
  It also ran the other way: the same mechanism minted a REFUTED whose
  witness `x=1, y=0` violates the user's own `assume(x < y)`, because
  `Script.relational_assumes_emitted` incremented for a fabricated axiom
  exactly as for a real one and the per-obligation un-withholding rule
  reads that count as "the solver ran with the full constraint set".

  **Two further defects, same site.** (1) A relational assume traced
  inside a `lax.cond` branch was forwarded as a **query-global** axiom,
  although the same file's `_unsatisfiable` already consults `branch_depth`
  and deliberately degrades inside a branch — *"the assume is
  branch-scoped and the other branch is real"*. A precondition that holds
  only when a branch is taken constrained the whole query. (2) The
  emission derived element indices from the assume equation's own shapes
  and applied them to whatever term tuple the ids resolved to; with
  colliding ids of different shapes that indexed out of range
  (`IndexError` out of the public `stelling.smt.emit`, degraded to UNKNOWN
  with the error quoted) or, in the other direction, **truncated in
  silence** — one `(assert (< t16_0 t17_0))` over element 0 of an
  unrelated 4-element array, with `relational_assumes_emitted == 1` and no
  note of any kind, and the un-withholding rule released a REFUTED on it.

  **WHICH VERSIONS ARE AFFECTED: 0.2.0 development builds only, and that
  is verified rather than assumed.** At `v0.1.0` the string
  `relational_assumes` does not occur anywhere in `src/`
  (`git grep -n relational_assumes v0.1.0 -- src/` is empty) and
  `smt.emit`'s signature is `(sl, solver, timeout_ms)` — there is no
  parameter and no field through which a foreign scope's ids can reach the
  emission. The fresh-id base `max(top_ids, default=0) + 1` **does** exist
  at `v0.1.0`; what did not exist was any consumer of a foreign id, so the
  id collision had nothing to collide with. **No released verdict is
  affected.**

  **WHICH PRIOR VERDICTS ARE RETROACTIVELY INVALID, and how to recognise
  one.** Any verdict produced by a 0.2.0 development build on a query
  that has *both*: (a) at least one `assume` comparing two varying
  quantities — its notes then carry `assume constraint DROPPED … `
  `(relational: both sides vary …)`; and (b) that assume traced inside a
  `jit` / `custom_jvp_call` / `custom_vjp_call` / `remat2` body or inside
  a `lax.cond` branch rather than at top level. On such a run, the
  escalation note `assert #N: K relational assume(s) forwarded to solver
  as axiom(s)` was the *only* record that anything was forwarded, and it
  said nothing about **what** was forwarded. Both directions are in
  scope: a VERIFIED may have been discharged by an axiom that is not the
  user's precondition, and a REFUTED may have been released by the
  un-withholding rule on the same non-evidence. A verdict on a query
  whose every `assume` is at top level, or which has no relational
  assume at all, is not affected — top-level assumes carry top-level ids
  and there is nothing to translate.

  **WHAT TO RE-RUN.** Every recorded verdict on an assume-carrying
  harness, under this build. The verdicts are now self-describing on
  exactly this point: each assume the obligation's slice could not state
  produces its own note, naming the assume's source line and the reason,
  and `Script.relational_assumes_emitted` now counts only assumes emitted
  **about the terms their operands denote** — so `emitted == requested`
  is a statement about content and not only a tally.

  **THE FIX IS A TYPE, NOT A BOUNDS CHECK, and that is deliberate.**
  Raising the fresh-id base above every id in every scope removes the
  collision and leaves the defect: the lookup would then simply MISS, the
  axiom would be dropped, and a run would lose a constraint the user
  wrote with nothing said about it. Instead the propagator now records a
  `RelationalAssume` carrying the **scope path** its operand ids belong to
  — a positional address in the IR tree, tagged by descent kind, so
  `("call", 4)` can never denote the scope `("cond", 4, 1)` denotes — and
  the slicer records the same path against the rename it applied when it
  inlined that scope. The translation happens once, in
  `_Slicer._carry_assumes`, and its output is a `SliceAssume` whose
  operands are atoms **of that slice**, alias-resolved and already known
  to have terms there. `smt.emit` no longer has a `relational_assumes`
  parameter at all: there is no channel through which a foreign name can
  be presented to it, so a wrong resolution is unconstructable rather than
  unlikely — the posture of the hardening-pass entry above. A scope the
  slicer did not inline has no entry, so a divergence costs a **disclosed
  skip**, never a forged axiom.

  **VERDICTS MOVE, AND THE MOVEMENT IS ALL IN ONE DIRECTION.** Measured on
  a **288-harness** generated sweep — an `assume` at top level or inside a
  `jit` / `cond` / `custom_jvp` body, 2–3 declarations, 0–2 trailing
  top-level statements walking the one-statement id window, satisfiable and
  unsatisfiable assume sets, two assert expressions, and the assume written
  either before or after the assert — run against `main` at `095bfd4` and
  against this build. 288 is the instrument's whole product
  (`stelling-sweeps/fix-0.2.0-scratch/sweep_assume_scope.py`: 4 carriers ×
  2 ndecls × 3 tails × 3 assume-sets × 2 exprs × 2 orders) and it is what
  the run stamps.

  | carrier | before → after | count |
  |---|---|---|
  | `jit` | UNKNOWN → VERIFIED | 48 |
  | `jit` | UNKNOWN → REFUTED | 18 |
  | `jit` | UNKNOWN → UNKNOWN | 6 |
  | `custom_jvp` | UNKNOWN → VERIFIED | 48 |
  | `custom_jvp` | UNKNOWN → REFUTED | 18 |
  | `custom_jvp` | UNKNOWN → UNKNOWN | 6 |
  | `cond` | UNKNOWN → UNKNOWN | 72 |
  | top level | VERIFIED → VERIFIED | 48 |
  | top level | REFUTED → REFUTED | 18 |
  | top level | UNKNOWN → UNKNOWN | 6 |

  Totals: 48 → 144 VERIFIED, 18 → 54 REFUTED, 222 → 90 UNKNOWN.

  **Not one of the 72 top-level-assume harnesses changed status** — the
  population the previous behaviour handled correctly; 48 + 18 = 66 of them
  were already decided and the remaining 6 were UNKNOWN and stayed UNKNOWN
  — and **no harness moved away from a decided verdict.** The **132** that
  moved split evenly across the id window, **44** at each of 0, 1 and 2
  trailing statements, which is the window closing measured directly: on
  `main` the answer depended on that count, and it no longer does.

  **AN EARLIER VERSION OF THIS ENTRY CLAIMED 702 HARNESSES, "78 top-level",
  and a 96/96/96 split, and none of those numbers is producible by the
  instrument at any setting** — 702 is not even divisible by the carrier
  count. They replaced a correct table that a later reading mistook for
  self-contradictory ("rows summing to 288 with 48 + 18 = 66 unchanged"
  read as contradicting "not one of the 72 top-level harnesses changed",
  which it does not: 66 already-decided plus 6 UNKNOWN-and-unchanged is 72).
  The table above is re-derived from a fresh run of the instrument against
  both trees, after the withholding repair below, and cross-checked against
  the archived `sweep_before.json` / `sweep_after.json` — which agree with
  it row for row.

  Every VERIFIED was attacked by sampling the admitted domain on a grid and
  evaluating the assert in exact `Fraction` arithmetic; every REFUTED by
  checking its witness lies in the declared box and satisfies every assume.
  **0 false VERIFIED and 0 false REFUTED on both trees.**

  **WHAT THIS SWEEP DOES NOT REACH.** Its `cond` carrier puts the assume
  inside a branch and nothing else, so no generated harness mixes a
  branch-scoped assume with a forwarded one — which is exactly the shape of
  the withholding defect recorded below. The sweep is therefore silent about
  it in both directions, and the evidence for that repair is its own
  reproducer and `tests/test_assume_ledger.py`, not this table.

  **THE SWEEP DID NOT ITSELF REPRODUCE THE FALSE VERIFIED**, and that is
  recorded rather than glossed. The collision needs a second condition
  beyond the id ranges meeting — the colliding ids must be ids that carry
  NAMES in the slice — and no generated shape here happens to satisfy it.
  What the sweep measures is that the fix mints no new wrong answer, and
  what it costs and buys in coverage. The false verdicts are reproduced by
  the audit's own archived harnesses, and those are what moved:
  `t11_false_verified_no_cond` VERIFIED → **REFUTED**, its axiom now reading
  `(assert (< x1 x0))` where it read `(assert (< x0 x1))` — the converse —
  on a query whose assert is false at all 20100 admitted grid points;
  `t03_false_refuted` REFUTED → UNKNOWN; `05_false_verified` and
  `06_fv_unrelated` VERIFIED → UNKNOWN; and `16_fv_controls` case C, a dead
  `lax.cond` moved from before the assert to after it, VERIFIED → UNKNOWN,
  so that moving it no longer moves the verdict.

  **ONE CONSEQUENCE THAT IS NOT AN IMPROVEMENT, DISCLOSED HERE RATHER
  THAN LEFT TO BE FOUND.** Forwarding correctly also forwards
  *contradictory* preconditions correctly. An unsatisfiable relational
  assume set makes the emitted script `unsat` for a reason that has
  nothing to do with the obligation, and the discharge that follows is
  **vacuous** — the standing finding that an unsatisfiable relational
  assume discharges everything, which this build does not fix and which
  the unsatisfiable-precondition refusal still does not see (it consults
  the interval domain, which by construction cannot decide a relational
  assume). Before this fix that failure was reachable only through a
  top-level assume; it is now reachable through a `jit`-carried one too.
  In the sweep above, VERIFIEDs with **no admitted point on the grid** went
  from **24 to 72** — the 48 new ones are exactly the `unsat` assume sets
  carried by a `jit` or `custom_jvp` body, 24 each, and every one of the 96
  new VERIFIEDs is either one of those 48 or one of the 48 with a sampled
  admitted region (the smallest sampled 820 admitted points and found no
  violation). None of the 48 is a new *kind* of error; all of them are the
  standing one, reached from a new place. The honest summary is that this
  entry closes the "the axiom is about the wrong values" hole and widens the
  surface of the separate "the axioms contradict each other" hole; the
  second is filed, unfixed, and its own batch.

  **A SECOND CONSEQUENCE THAT IS NOT AN IMPROVEMENT, ALSO DISCLOSED.** The
  VERIFIED bar narrows per obligation only when the recorded invocation
  reproduces both the slice's fingerprint and the script that slice emits
  (`verdict._evidence_is_about`), and its re-derivation calls
  `slice_obligation` with no propagation and therefore with no assumes. A
  script that carries a relational axiom is not reproduced by one that does
  not, so **an obligation discharged with a forwarded axiom cannot narrow
  the bar** and a query containing any barred primitive falls back to the
  whole-query set. Measured: a two-obligation query whose first obligation
  is scatter-free and assume-constrained and whose second contains a
  `scatter` — obligation #0 goes `withheld` → `discharged` with this fix,
  and the verdict stays UNKNOWN, now reading "no recorded solver invocation
  for the decided obligation #0 reproduces both this query's slice of it and
  the script that slice emits". The direction is a WIDER bar, so nothing is
  minted; what is lost is the narrowing, on exactly the runs where an axiom
  decided the obligation. This is the audit's own open question 4, it
  predates this fix, and this fix makes it reachable more often because more
  scripts now carry an axiom. Left alone deliberately: repairing it means
  deciding what the bar's re-derivation is allowed to see, and that function
  is emphatic in its own docstring about being handed no record-supplied
  value. Filed, unfixed, not this batch.

  **WHAT THAT NOTE MAY NOT SAY, corrected here because the first version of
  this entry shipped beside the wrong sentence.** The fallback clause used
  to end *"so the escalation is not evidence about this query"*. It is a
  claim `_bar_scope` has not measured, and on exactly this shape it is
  FALSE: the escalation IS about this query, and the re-derivation simply
  was not given the axioms. Measured on this tree — `smt.slice_fingerprint`
  walks `sl.eqns` and never `sl.assumes`, so the slice the escalation ran on
  and the slice `_bar_scope` re-derives carry the **same `slice_sha256`**
  and differ only in `smt2_sha256`, by the `(assert …)` axiom lines
  (`tests/test_assume_disclosure_claims.py::test_a_slice_with_assumes_and_
  one_without_share_a_slice_fingerprint`). The clause now says what the
  function did: *"so this re-derivation could not identify any recorded
  escalation as one about this obligation of this query"*. The MECHANISM is
  untouched — the same runs fall back to the same wider bar — and stays
  filed for its own batch.

  Also fixed at the same site: `smt.emit` cannot raise `IndexError` on a
  wrong-arity assume, and it cannot emit a partial axiom over element 0 of
  an unrelated array. Both arms are closed by the SCOPE-CORRECT IDENTITY
  above and not by the residual checks that sit where the crash used to be:
  the element indices and the term tuples now come from one pair of shapes
  because the operands are resolved through their own scope's rename, and
  the two `raise`s left in `smt.emit` plus the `_pair_elementwise(probe)`
  call in `_carry_assumes` are belt and braces behind that. **Measured, and
  the reason the crediting was corrected**: substituting `renamed` for
  `probe`, and deleting the emission's term-count raise, each redden 0 of
  the suite's 2954 collected tests on this tree. `docs/norms.md` § "Guard coverage is proven by
  mutation, not by construction" — no distinguishing case was constructed,
  so none is claimed; the guards are kept, uncredited.

  **THE MUTATION FIGURE FOR THE IDENTITY REPAIR ITSELF.** Ignoring the scope
  path in `_carry_assumes` (`remap = {}`, the pre-fix bare-integer lookup)
  reddens **18** tests, all of them in `tests/test_assume_scope_identity.py`.
  Method: full suite against the mutant, with the environmental baseline
  subtracted — the untracked-file check and the generated-page line-number
  check redden on any edited tree and are not detections. **No pre-existing
  test anywhere in the suite detects that mutant**, so all regression
  protection for this defect rests on that one file. An earlier build report
  recorded this mutant as "14 red"; that figure does not reproduce and is
  not used.

  ---

  **AMENDMENT, SAME DAY, SAME EVENT: THE BRANCH-SCOPING HALF OF THIS FIX
  OPENED A FALSE REFUTED, AND THE RULE IT TRIPPED IS NOW A LEDGER RATHER
  THAN TWO COUNTS.** A verdict flip is a soundness event and it belongs in
  this entry, not in a second one: the defect did not exist before the fix
  above and was introduced by it.

  **What happened.** A definite violation found by a solver is WITHHELD from
  REFUTED whenever an assume was dropped, because the solver then ran over a
  superset of the admitted region. It is released again when the query was
  fully constrained after all — and the release test was
  `n_emitted == len(propagation.relational_assumes)`. Removing branch-scoped
  assumes from `relational_assumes` moved the DENOMINATOR. A query with one
  ordinary relational assume (forwarded, emitted) plus one branch-scoped
  assume (now dropped, and no longer counted anywhere) satisfied `1 == 1`,
  and the violation was released although the branch-scoped precondition had
  never reached the solver.

  **Measured.** `x, y ∈ [0, 10]`, `s ∈ [1, 10]`, `assume(x <= y)` at top
  level, `lax.cond(s > 0, yes, no, …)` whose `yes` branch carries
  `assume(v[0] >= v[1])`, asserting `y <= x + 1`. Identical script on both
  trees (`emitted = 1`, `(assert (<= x0 x1))`); both solvers `sat` on both
  trees. `main` at `095bfd4` → **UNKNOWN**; this branch before the amendment
  → **REFUTED**, witness `x = 0, y = 2`. The selector is definitely true, so
  the assume-bearing branch is the one that runs and its precondition
  `x >= y` is false at that point: the witness is inside the declared box,
  satisfies the other assume, and is **not an admitted point**.

  **THIS IS AUDIT 0.2.0 S6 AGAIN, and both are fixed here rather than one.**
  S6 is the same rule failing for a different reason: `assume_dropped` — the
  flag that gates the whole withholding — is set by **any** drop reason (a
  predicate from a primitive outside `{ge,gt,le,lt,eq}`, `and` on non-bool,
  wrong operand count, a non-finite or subnormal constant bound, an
  out-of-scope producer), while the denominator counted only the relational
  subset. `assume(x < y)` beside `assume(jnp.logical_or(x > 5, y > 5))` over
  `[-10, 10]²` therefore returned REFUTED at `x = -3.5, y = -2.5`, which
  violates the `or`; the assert is true at every one of 87 790 sampled
  admitted points. Fixing one and leaving the other would have left the same
  defect standing under a different name.

  **THE FIX IS A LEDGER, AND DELIBERATELY NOT A SECOND COUNT.** Two counts
  that must agree is the shape that failed both times — once because one of
  them counted the wrong subset, once because a change on the propagator's
  side silently moved one of them. `Propagation.assume_ledger` now carries
  ONE `AssumeDisposition` per assumed conjunct the propagator classifies,
  written where the classification happens, with exactly one of four kinds:

  * `applied` — the interval domain narrowed (or confirmed already within)
    the target variable, so every judged point satisfies it;
  * `no-op` — not applied, but definitely TRUE over the boxes in force, so
    it excluded nothing and the judged set IS the assumed region for it;
  * `forwarded` — relational, handed to the solver layer as element *k* of
    `relational_assumes`, and carrying *k*;
  * `dropped` — anything else: dropped for any reason, branch-scoped,
    inert-mode, unclassified.

  The release test is `propagate.unaccounted_assumes(ledger,
  emitted_origins)`, and it counts nothing: every entry must be
  applied-or-no-op, or forwarded **with its own index** among the origins
  the obligation's script actually emitted (`smt.Script.emitted_origins`,
  written from the same loop that writes the axiom lines, off
  `obligation.SliceAssume.origin`). The membership test is a **whitelist**,
  so a disposition added later that nobody taught it is UNACCOUNTED: a new
  drop reason has to name a disposition to be recorded at all, and an
  unknown name fails closed instead of shrinking a denominator nobody
  re-derived. The withholding note now names the conjunct, its disposition
  and its source line rather than restating the rule.

  **WHICH VERDICTS MOVE, IN BOTH DIRECTIONS, MEASURED.** The 288-harness
  sweep above is verdict-identical before and after this amendment
  (`sweep_assume_scope.py` re-run against both trees after it), because none
  of its generated shapes mixes a branch-scoped assume with a forwarded one
  — which is why the sweep is not evidence about this repair either way. The
  two reproducers move in the withholding direction: the branch-scoped one
  REFUTED → UNKNOWN, and S6's `01_false_refuted_mixed_drops.py`
  REFUTED → UNKNOWN.

  **One shape moves the OTHER way, and it is a precision gain rather than a
  minting.** An assume whose whole content is a conjunct definitely TRUE
  over the boxes in force takes the propagator's whole-drop path, which sets
  `assume_dropped` unconditionally — and the old rule then had a denominator
  of ZERO and could never release, so the violation was withheld forever.
  The ledger records that conjunct as `no-op`, which is accounted, and the
  violation is released. Measured: `x ∈ [0, 10]`,
  `assume(jnp.logical_or(x >= -1., x >= -2.))`, `assert x <= 5.` —
  `main` UNKNOWN, this build **REFUTED** at `x = 6`, which is in the
  declared box, satisfies the assume, and falsifies the assert. Sound, and
  it is the rule the MIXED-conjunction path already applied to the same
  class of conjunct (`if restricting or vacuous:` filters no-ops out); this
  amendment makes the whole-drop path agree with it instead of being
  accidentally stricter. Nothing in the suite or the sweep exercises it,
  which is why it is disclosed here rather than left to be found
  (`tests/test_assume_ledger.py::
  test_an_assume_that_excludes_nothing_no_longer_withholds_forever`).

  **What this does NOT fix.** It does not make a branch-scoped assume
  usable — it is still not forwarded, and a query carrying one still cannot
  be refuted through the solver leg. It does not decide whether the assume
  set is SATISFIABLE (the standing vacuous-VERIFIED finding above). And it
  does not touch the interval or affine legs, which withhold through
  `exactness.certifies_set_refutation` on their own and were never reading
  the count.

  **WHICH PRIOR VERDICTS ARE RETROACTIVELY INVALID.** Any REFUTED produced
  by a 0.2.0 development build at or after the identity fix on a query
  carrying at least one relational assume AND at least one other assume that
  the interval domain did not apply — a differently-dropped one (S6) or a
  branch-scoped one. Both directions of the count are in scope. A VERIFIED
  is unaffected: the withholding was always one-sided.

  **MUTANTS FOR THE RULE, since a rule this load-bearing has to be shown
  tested and not merely present.** Reverting the release test to the
  two-count comparison reddens **6** tests, all in
  `tests/test_assume_ledger.py`; turning `unaccounted_assumes`'s whitelist
  into a blacklist (`kind != "dropped"` releases) reddens **2**, both in the
  same file, and they are the pair written for exactly that mutant. Same
  method as above — full suite, environmental baseline subtracted. **No
  pre-existing test detects either**, which is why that file exists.

  Full suite green: **2944 passed, 10 skipped**, against 2863 / 10 on
  `main`, with the same skip SET (hypothesis ×6, pytest-xdist ×1,
  blackjax ×2, the x64-only threefry mask ×1 — all environment-driven, none
  of them this change's). Every construction here is a permanent regression
  test: `tests/test_assume_scope_identity.py` for the identity repair —
  including the shape where a bare integer lookup CANNOT be right by
  accident, the assume's operands being the body's own intermediates rather
  than its arguments — `tests/test_assume_ledger.py` for the release
  invariant, and `tests/test_assume_disclosure_claims.py` for the sentences
  this entry corrects.

- **2026-08-14 (0.2.0 development, unreleased): VACUOUS VERIFIED — an
  unsatisfiable forwarded `assume` discharged every obligation, and the
  empty-precondition refusal could not see it.** An `assume` whose two
  sides both vary cannot be applied in the interval domain, so the
  empty-declared-set oracle never sees it either: `_unsatisfiable` meets a
  variable's propagated BOX with the assumed half-space, and `x < y` is not
  a half-space on either box. Since 0.2.0 the same assume is emitted to the
  solver as a POSITIVE AXIOM. If the emitted axiom set is unsatisfiable —
  on its own, or against the declared boxes — then
  `boxes ∧ axioms ∧ ¬P` is unsat **for every `P`**, the obligation is
  `discharged`, and the verdict is VERIFIED. Nothing checked whether the
  `unsat` came from the obligation or from the precondition. (Audit 0.2.0
  S7, S7′, and M5/M4 in the same batch.)

  **THE ASYMMETRY IS WHY THIS IS A DEFECT AND NOT A TECHNICALITY.** The
  non-relational form of the identical mistake is refused, loudly, by
  design: `dt ∈ [5, 10]` with `assume(dt < 1.0)` raises
  `UnsatisfiableAssumptionError` — *"the declared set as assumed is empty
  and every downstream obligation would be vacuous (harness defect; nothing
  was verified)"*. 0.2.0's forwarding built a route around that refusal, and
  the route is reached by an ordinary typo rather than an exotic
  construction.

  **Measured, three shapes, on this tree at `0874dd1` (the parent
  commit).**

  * A MIS-DECLARED BOUND, one assume: `dt = any_array((), "float64",
    (5.0, 10.0))`, `dt_max = any_array((), "float64", (0.0, 1.0))`,
    `assume(dt < dt_max)`, `assert_(dt + dt_max <= 1.0)`. No point of the
    declared box satisfies `dt < dt_max` (`dt ≥ 5`, `dt_max ≤ 1`), and
    `dt + dt_max ≥ 5` everywhere in the box, so the assert is false at every
    declared point. **VERIFIED.** Deleting the assume: REFUTED, before and
    after.
  * An AXIOM CYCLE, no help from the boxes: `assume(x < y)`,
    `assume(y < z)`, `assume(z < x)` over `[-10, 10]³`, asserting
    `x + y + z >= 100.0` where the box maximum is 30. **VERIFIED.**
  * The INDUCTIVE form (S7′), which is the user-facing one:
    `check_inductive_step` on the body `x, y -> ((x + y) * 10, (x + y) * 10)`
    with the invariant `[-1, 1]²` and a contradictory `assume(x < y)` /
    `assume(y < x)` in the body. **VERIFIED**, with the note *"the invariant
    is preserved by one step"* — and from `x = y = 0.5`, inside the
    invariant, one step gives `10.0`. Deleting the assumes: REFUTED, before
    and after.

  **AND THE STAMP MADE A POSITIVE CLAIM THAT WAS FALSE.** At
  `vacuity_mode="all"` the first harness stamped *"vacuity checked
  (mode=all): no obligation discharges with the declared bounds widened —
  under the mechanism(s) that ran, this VERIFIED was not re-derivable
  without the declared envelope"*, which a reader takes for SUBSTANTIVE. It
  was the exact opposite: the VERIFIED rested entirely on a precondition no
  declared point satisfies. The instrument measured a real dependence —
  widening the boxes makes `dt < dt_max` satisfiable again, so the negated
  obligation becomes `sat` — just not the one the sentence is read as
  claiming. No may-be-vacuous line was present either, although this file's
  own constraining-assume entry (2026-07-18) states the policy: *"definite
  REFUTEDs under an uncertified precondition are withheld to UNKNOWN with
  the reason disclosed, uncertified VERIFIEDs carry a stamped
  may-be-vacuous line"*.

  **WHICH VERSIONS ARE AFFECTED: 0.2.0 development builds only, verified
  rather than assumed.** At `v0.1.0`, `git grep -c relational_assumes
  v0.1.0 -- src/` is empty, `smt.emit`'s signature is
  `(sl, solver, timeout_ms)`, the string `sl.assumes` occurs 0 times in
  `smt.py` and `assumes` 0 times in `obligation.py` — no assume reaches a
  solver at all, so the route does not exist. Re-run on a `v0.1.0` tree:
  the mis-declared-bound harness and the axiom cycle both return
  **UNKNOWN**. S7′ is doubly out of scope there — `stelling.inductive` does
  not exist at `v0.1.0`. The exposure widened inside 0.2.0 rather than at
  one commit: the S5 identity repair (previous entry) made forwarding work
  from behind a transparent call, which is the documented
  `preconditions.*` idiom, and the 288-harness sweep's vacuous-VERIFIED
  count went 24 → 72 across it.

  **THE FIX: THE DISCHARGE IS AUDITED BY THE BACKEND THAT PRODUCED IT.**
  Before an `unsat` is credited, and only on an obligation whose script
  carries at least one forwarded relational axiom, the SAME backend is asked
  the same script with the one `(assert (not <root>))` line removed
  (`stelling.smt.emit(..., states_obligation=False)`): are the declared
  boxes and those axioms satisfiable at all? One semantic line differs,
  plus an inert `; admitted-region check: …` header comment so that a
  dumped script and the `smt2_sha256` a stamp carries say WHICH of the two
  questions they asked. *(This entry said "exactly one line" until audit
  B3; the emission had prepended the comment since the first commit and
  `test_the_admitted_region_script_is_the_obligation_script_minus_one_line`
  had always measured both directions of the diff, so code, test and prose
  disagreed. Semantically inert, and the comment is right to be there.)*

  * `unsat` — the admitted region is EMPTY. A backend has reported that its
    own first answer was about the precondition. This raises
    `UnsatisfiableAssumptionError`, the same class and the same closing
    sentence as the non-relational refusal, from
    `solvers._dispatch_obligation`. The scope is right: an assume is a
    precondition on the whole query, and a slice's axioms are a SUBSET of
    the query's assumes, so a slice whose region is empty proves the
    query's is.
  * `sat` **and the script accounts for every assume of the query** — a
    model, hence a point of the region. The discharge stands, clean.
  * `sat` **with any assume unaccounted for** — nothing established; falls
    to undecided. **This bullet is the audit B3 amendment below**, and the
    two are not the same argument: `unsat` on a relaxation proves the
    tighter set empty, a MODEL of a relaxation proves nothing about the
    tighter set. The condition is `propagate.unaccounted_assumes` — the
    predicate that already gates the release of a withheld violation, used
    here in its other direction.
  * undecided — the discharge stands (it is sound: every admitted point
    satisfies the obligation, and there may be none) and stops being clean.
    The obligation detail gains `[MAY BE VACUOUS: …]` and the stamp gains
    `precondition satisfiability uncertified: …` — the line the policy
    above already required on this path.
  * `sat` and `unsat` from two backends on one script raises
    `SolverDisagreement`, the posture the obligation script's own
    disagreement already takes. Picking either way would be a tiebreak: on
    `unsat` a solver bug is reported to the user as a defect in their
    harness; on `sat` a discharge is stamped clean while a backend says the
    region it rests on is empty.

  **WHY NOT THE CERTIFICATE ALONE, AND WHERE THE CERTIFICATE IS USED.** The
  audit's other direction was to refuse a discharge whenever
  `Propagation.region_inhabited` is absent. That certificate is ONE-SIDED by
  construction — its own docstring says so — and False means "no witness
  was found" on a probe grid of at most 16 points, which is the normal
  answer on a perfectly satisfiable precondition. Refusing on it would turn
  every relational-assume query the probe happened to miss into UNKNOWN:
  not a repair of a vacuity defect but a withdrawal of the feature. So it is
  used in the direction it is sound in — True settles the question and the
  extra solver call is SKIPPED entirely — and False falls through to the
  question actually being asked.

  **WHICH VERDICTS MOVE, MEASURED, ON THE 288-HARNESS SWEEP**
  (`stelling-sweeps/fix-0.2.0-scratch/sweep_assume_scope.py`, copied to
  scratch and run unmodified against three trees):

  | tree | REFUTED | UNKNOWN | VERIFIED | RAISED | vacuous VERIFIED |
  |---|---|---|---|---|---|
  | `main` `095bfd4` | 18 | 222 | 48 | 0 | **24** |
  | `0874dd1` (parent) | 54 | 90 | 144 | 0 | **72** |
  | this build | 54 | 90 | 72 | **72** | **0** |

  FALSE VERIFIED and FALSE REFUTED are 0 on all three. Row by row against
  the parent: **72 VERIFIED → RAISED, 72 VERIFIED → VERIFIED, 54 REFUTED →
  REFUTED, 90 UNKNOWN → UNKNOWN — nothing else moves.** Every one of the 72
  that moved is from the `unsat` assume set (`assume(x0 < x1)` beside
  `assume(x1 < x0)`), and every row with a SATISFIABLE assume set is
  verdict-identical. **The coverage cost is zero: no substantive discharge
  was lost.** The remaining 24 `unsat`-set rows are the `cond`-carried ones,
  which stay UNKNOWN and correctly so — a branch-scoped assume is not
  forwarded, so it never claims the query's region is empty.

  **WHAT IT COSTS.** One extra solver call per relational-assume-bearing
  DISCHARGE, and nothing anywhere else — not on a `sat`, not on a timeout,
  not on a query without a relational assume, not when the certificate
  already answered. Measured on the same sweep, where every harness carries
  a relational assume by construction (the worst case): **324
  admitted-region invocations out of 1044 total solver invocations (31%)**,
  of which 144 are on the 72 harnesses that now refuse and 180 on the 72
  that still verify; 0 on every REFUTED and every UNKNOWN row. Wall clock
  for the whole sweep: **35.1 s at `0874dd1`, 39.1 s here (+11%)**.

  **WHICH PRIOR VERDICTS ARE RETROACTIVELY INVALID, AND HOW TO RECOGNISE
  ONE.** Any VERIFIED from a 0.2.0 development build whose notes contain
  *"relational assume(s) forwarded to solver as axiom(s)"* — that phrase is
  the necessary condition, and it is in the notes of every affected verdict.
  It is not sufficient: most such VERIFIEDs are substantive. **Re-run the
  harness on this build and read the DISCLOSURE, not only the status.** An
  empty precondition that one obligation's script states whole raises
  `UnsatisfiableAssumptionError` naming the assume's source line; an
  inhabited one that a script or the probe certifies returns the same
  VERIFIED, clean; **everything else returns VERIFIED with `[MAY BE
  VACUOUS: …]` on the obligation and `precondition satisfiability
  uncertified` on the stamp — including the case where the precondition is
  EMPTY but no single obligation's cone contains the whole contradiction**
  (audit B3; the amendment below). A VERIFIED carrying that pair has not
  been shown to be about anything, and re-running does not answer the
  question for you. For `check_inductive_step`, additionally re-read the
  appended note: a body whose `assume` reaches one of the state-bound
  obligations says `inductive step CONDITIONAL — NOT the inductive step`
  where it used to claim preservation.

  **TWO DISCLOSURE DEFECTS FIXED IN THE SAME BATCH, both about a claim
  nothing established.**

  * A forwarded relational axiom is a PREMISE the answer rests on, exactly
    as an interval narrowing is, and only the narrowing half stamped it. The
    stamp now carries `forwarded relational assume(s) on obligation(s) …`
    with the same `the verdict holds where the precondition holds` phrase —
    which is now a named constant
    (`stelling.propagate.CONDITIONAL_ON_PRECONDITION`) read by
    `Verdict.render` and by `check_inductive_step` rather than a literal
    duplicated in two files.
  * The `vacuity checked …` line appends `WHAT THIS MEASUREMENT DOES NOT
    SAY: …` whenever the stamp carries any `precondition satisfiability
    uncertified` line, keyed on the shared prefix
    (`UNCERTIFIED_PRECONDITION_PREFIX`) rather than on a list of mechanisms
    a later mechanism would have to be added to. **The empty-region route to
    that sentence is closed where one obligation's script states the whole
    contradiction** — that run raises, so there is no VERIFIED left to stamp
    — and it is NOT closed for a cone-split one (audit B3, below), which
    reaches the sentence through the undecided route and is qualified there.
    So the qualification covers the undecided case, the cone-split empty
    case, and the two pre-existing interval ones.

  **AND TWO INDUCTIVE-STEP OVER-CLAIMS (audit 0.2.0 M5, M4).**

  * M5: an `assume` in the body is a precondition on the whole query, so a
    VERIFIED means "every state IN THE ASSUMED SUB-REGION stays within
    bounds after one step" — which is not the inductive step, because the
    successor state need not re-enter that sub-region and there is nothing
    to apply the second step to. The note claimed preservation with only an
    initial-state caveat. Measured: `x -> 1.5x` on `[-1, 1]` under
    `assume(x <= 0.5)` / `assume(x >= -0.5)` is VERIFIED, and iterating from
    the ADMITTED `x = 0.4` gives `0.4, 0.6, 0.9, 1.35` — outside the
    invariant at step 3. The note now begins `inductive step CONDITIONAL —
    NOT the inductive step`, names the fix (state the restriction in
    `state_bounds`, where the successor is checked against the same set the
    predecessor was drawn from), and fires on the conditionality stamp
    rather than on a count — so it catches the narrowing half and the
    forwarded half, and correctly does NOT fire on a dropped assume (a drop
    widens the judged set, which proves more than the inductive step needs)
    or on a no-op one (which excludes nothing). `docs/inductive-step.md`'s
    "What VERIFIED does NOT mean" list and the module docstring say the
    same.
  * M4: the REFUTED note mapped obligations to state variables
    POSITIONALLY, while the harness appends its `2 × len(state_bounds)`
    bound checks AFTER tracing `body` — so any `assert_` the body declares
    shifts every index. Measured, body `{"a": a + 10.0, "b": b * 0.5}` on
    `[-1, 1]²` with one assert in the body: the note said *"Escaped: b
    (below lower bound)"* while `b * 0.5 ∈ [-0.5, 0.5]` never escapes and
    `a + 10 ∈ [9, 11]` escapes above. The offset is now derived from the
    obligation count; a REFUTED whose violated obligations are ALL the
    body's own says so instead of blaming the invariant. Status and
    per-obligation statuses were correct throughout — only the note was
    wrong.

  **WHAT THIS DOES NOT FIX, disclosed rather than left to be found.**

  * **An empty precondition is not detected when NOTHING ESCALATES.** The
    check audits a SOLVER discharge, so a query whose obligations are all
    decided by the interval (or affine) leg never emits an admitted-region
    script. Measured on this build: the mis-declared-bound declarations with
    `assert_(dt + dt_max <= 20.0)` — true over the whole box, so the
    interval leg discharges it — returns **VERIFIED at both
    `solver_timeout_ms=None` and `=5000`**, with the empty precondition
    unreported. That VERIFIED is not vacuous (an interval discharge is an
    unconditional claim over the un-narrowed declared box, which is
    STRONGER than the conditional one), but the harness defect goes
    unmentioned, and a reader may take the precondition for meaningful.

    **The gap is narrower than that sentence, and the difference is
    `solver_timeout_ms`** (audit B3). A BOX-INDEPENDENT contradiction —
    `x, y ∈ [-10,10]` under `assume(x < y)` and `assume(y < x)`, with the
    interval-decided `assert_(x + y <= 100.0)` — **RAISES when
    `solver_timeout_ms` is passed and returns VERIFIED without it.** The
    vacuity widen re-check runs at the same pipeline depth as the original
    call, and widening the boxes to `(-inf, inf)` un-decides the assert, so
    the widened query escalates, emits the admitted-region script and finds
    the 2-cycle unsat. Measured on this build, all four combinations:
    box-independent → `VERIFIED / VERIFIED / RAISED / RAISED` at
    `(None, 'inputs-only')`, `(None, 'all')`, `(5000, 'inputs-only')`,
    `(5000, 'all')`; box-dependent (the mis-declared bound above, where
    widening makes `dt < dt_max` satisfiable again) → `VERIFIED` in all
    four. So the undetected case is precisely: an interval-decided
    obligation whose contradiction *needs the declared boxes*, or a run with
    no solver budget at all.
  * **A cone-split empty precondition is disclosed, not refused** (audit
    B3). The refusal is one script's `unsat`, and a script states only the
    assumes whose operands lie in that obligation's backward cone. Spread
    the contradiction so no cone holds it whole — `x, y, z ∈ [-10,10]` under
    `assume(x < y)`, `assume(y < z)`, `assume(z < x)`, asserting
    `x - y <= 0.0` — and every script sees one satisfiable link. Measured on
    this build: **VERIFIED**, with `[MAY BE VACUOUS: …]` on the obligation
    and `precondition satisfiability uncertified` on the stamp, over a
    region that admits no point (an exact-`Fraction` 21³ grid over
    `[-10,10]³` finds 0, and `x<y ∧ y<z ∧ z<x` gives `x<x`). Closing it
    needs a WHOLE-QUERY admitted-region script — one emission naming every
    assume's operands, which no obligation slice can — and that would also
    buy back the disclosure cost measured below. Not attempted here.
  * **A branch-scoped contradictory assume still returns UNKNOWN rather
    than refusing.** It is not forwarded at all (previous entry), so no
    axiom set is available to test, and refusing would be wrong anyway: the
    other branch is real.
  * **It does not make the check cheaper than one call.** No caching of the
    admitted-region answer across obligations of one query is attempted;
    slices differ, so their scripts differ.
  * `verdict._bar_scope`'s mechanism (audit M10) is untouched; so are the
    rational-`pow` rows and the IEEE legs.

  ---

  **AMENDED 2026-08-15 — AUDIT B3, TWO FINDINGS ON THE REPAIR ABOVE.** One
  is the hazard this entry exists to close, left open in a shape the repair
  actively certified as clean; the other is a disclosure regression the
  repair introduced. Amended rather than filed separately because both are
  about the mechanisms described above and a reader must not have to join
  two entries to learn what the first one does.

  **(1) `UNSOUND` — the check asked about the SLICE and the answer was read
  as being about the QUERY.** `obligation._Slicer._carry_assumes` skips
  every relational assume whose operands fall outside an obligation's
  backward cone, with a quoted reason, so the admitted-region script states
  a SUBSET of the query's axioms. The `unsat` direction is argued correctly
  above and is sound. The converse is not, and the code used it: `sat`
  became `REGION_INHABITED` ("a model, hence a point of the region") and the
  verdict was stamped clean.

  Measured at `1dc1b52`: `x, y, z ∈ [-10,10]`, `assume(x < y)`,
  `assume(y < z)`, `assume(z < x)`, `assert_(x - y <= 0.0)`. The assert's
  cone is `{x, y}`, so the slice carries `x < y` and quotes two skip
  reasons; both backends answered `sat` on `boxes ∧ (x < y)`; **STATUS:
  VERIFIED**, no `[MAY BE VACUOUS]`, no `precondition satisfiability
  uncertified`. The assumed region is EMPTY — `x<y ∧ y<z ∧ z<x` gives
  `x<x`, and an exact-`Fraction` 21³ grid over `[-10,10]³` finds 0 admitted
  points. It reaches `check_inductive_step` too: body
  `{x,y,z} -> {0.6(x-y)+0.6, 0.5y, 0.5z}` on `[-1,1]³` under the same
  3-cycle returned VERIFIED with "the invariant is preserved by one step".
  Not a regression — `0874dd1` does the same — but it is the shape this
  entry's own mechanism certified.

  **THE FIX IS ONE CONDITION, AND IT IS THE PROJECT'S EXISTING PREDICATE.**
  `REGION_INHABITED` may be concluded only when the region the solver ran
  over is inside the region EVERY assume of the query describes, which is
  exactly what `propagate.unaccounted_assumes(assume_ledger,
  emitted_origins)` decides for the withholding-release rule. One predicate,
  one argument, both directions: a witness of the query is only a witness if
  it lies in the assumed region, and a model of the region script is only a
  point of that region for the same reason. `solvers._region_answer` now
  takes a required keyword-only `accounts_for_every_assume` and returns
  `REGION_UNCERTIFIED` on `sat` without it; `REGION_EMPTY` on `unsat` is
  unchanged, because that is the direction the subset argument licenses.

  *The auditor stated the condition as `sl.assumes_skipped` being non-empty.
  That is necessary and NOT sufficient: it counts only assumes the slicer
  dropped, while an assume the PROPAGATOR dropped — a `jnp.all(...)`
  reduction, a non-finite bound, an unclassified predicate — is equally
  absent from the script and equally free to be violated by the model. The
  ledger read covers both by construction, and its whitelist refuses a
  disposition nobody has taught it. `Propagation.region_inhabited` needs no
  change and keeps its short-circuit: it is a WHOLE-QUERY point certificate,
  and it is the one mechanism that still clears a cone-split run.*

  **(2) `FRAGILE` — a per-obligation fact read as a whole-query one.** The
  entry above added `forwarded relational assume(s) on obligation(s) #k:
  <phrase>`, scoped per obligation. Both consumers tested
  `any(CONDITIONAL_ON_PRECONDITION in a for a in stamp.assumptions)`, which
  is the whole-query question, and an interval NARROWING line (whole-query,
  names no obligation) and a FORWARDED line (scoped) are not the same fact.

  * `verdict.Verdict.render`. Measured: one interval-refuted obligation plus
    one that escalates with a forwarded axiom made the REFUTED render
    *"conditional … judged over the propagated superset of the
    precondition-narrowed set, not over the full declared box"*. Both
    clauses false — a relational assume is inert in the interval domain so
    nothing narrowed, and assert #0's own detail line four rows below says
    "over the declared box". A true unconditional refutation was
    under-reported.
  * `inductive.check_inductive_step`. Measured: body
    `{x,y} -> {0.5x, 0.5y}` on `[-1,1]²` with `assume(x < y)` carried only
    into a body-assert's slice printed `inductive step CONDITIONAL — NOT the
    inductive step … the invariant does NOT follow for all iterations`,
    while the four bound obligations have single-variable cones, were judged
    over the full declared box, and close the induction outright
    (`|0.5·t| ≤ 0.5 ≤ 1`).

  Both reads are now SCOPED, through one shared
  `propagate.conditional_on_precondition(assumptions, indices)`: a line that
  names obligations bears on those, a line that names none is whole-query
  and bears on all. `render` passes the `violated-over-set` indices — a
  forwarded axiom lives in a script and a script exists only where the
  interval domain gave up, so it can never name one of them — and the
  inductive note passes the state-bound obligations (`index >= offset`,
  both quantities already computed there). An unparseable scope falls back
  to whole-query: the failure direction is over-disclosure.

  **WHAT MOVES, MEASURED ON THIS BUILD.** The 288-harness sweep re-run
  unmodified: **54 REFUTED / 90 UNKNOWN / 72 VERIFIED / 72 RAISED, 0 FALSE
  VERIFIED, 0 FALSE REFUTED, 0 vacuous VERIFIED — identical to the table
  above, and 756 solver invocations before and after.** No verdict moves and
  no solver call is added; what moves is disclosure. On a purpose-built
  48-harness cone-split family (relation `<`/`<=` × 3-cycle/3-chain ×
  carrier pair × threshold × unused tail variable; the region is empty iff
  strict-and-cycle, 12 of 48):

  | | before | after |
  |---|---|---|
  | VERIFIED / UNKNOWN | 40 / 8 | 40 / 8 |
  | empty-region VERIFIEDs stamped may-be-vacuous | **0 of 12** | **12 of 12** |
  | inhabited-region VERIFIEDs (controls) | 28 | 28 |
  | … of which newly qualified | — | **8** |
  | solver invocations | 256 | 256 |

  **THE COST IS A CAVEAT ON TRUE VERIFIEDS, AND IT IS REAL: 8 of the 28
  inhabited controls here, 18 of the 72 VERIFIEDs on the sweep (0 before,
  both cases).** They are the runs where a satisfiable precondition is split
  across cones AND the probe grid found no point — a STRICT chain, where no
  corner satisfies `x < y < z`. Nothing on such a run establishes
  non-emptiness, so the caveat is truthful; the same harness with `<=` is
  certified by the probe and stays clean, which is what keeps the
  qualification from being printed on everything. A whole-query
  admitted-region script would buy all of it back and close the cone-split
  gap at once. Not attempted here.

  **MUTANTS, since a rule that flips verdicts has to be shown tested and
  not merely present.** Disabling the admitted-region check (leaving the
  certificate path) reddens **6** tests, all in
  `tests/test_vacuous_precondition.py`. Reverting both inductive repairs —
  `offset = 0` and `conditional = False` — reddens **2**, in the same file.
  **No pre-existing test detects either**, on the full suite, which is why
  that file exists. Same method as the entries above: full suite, no
  environmental baseline to subtract.

  For the amendment, three mutants run on the full suite at this commit,
  each reverting exactly the changed expression:

  | mutant | tests reddened |
  |---|---|
  | `_region_answer`: `sat` → `REGION_INHABITED` unconditionally | **4** |
  | `render`: scoped read → `any(... in assumptions)` | **1** |
  | inductive note: scoped read → `any(... in assumptions)` | **1** |

  All six are in `tests/test_vacuous_precondition.py`; no pre-existing test
  detects any of the three.

  Full suite green in **both** precision environments: **2985 passed, 10
  skipped** with `JAX_ENABLE_X64=1`, and **2986 passed, 9 skipped** without
  it (the configuration CI runs). The skip sets differ by exactly one
  entry — `test_tripwire_arm.py`'s threefry case, whose skip condition IS
  x64-on — and by nothing else; the ninth run of the test it stops skipping
  is the extra pass. At `1dc1b52` before the amendment: 2969 / 10 with x64,
  and **75 FAILED without it**, which is the reason for the test-side half
  below. With `jax` genuinely unimportable (`sys.modules["jax"] = None` from
  a `-p` plugin, so the gates see an absent package during collection):
  **1476 passed, 107 skipped, 0 failed** — no undisclosed skip. The audit's
  own re-checker (`audit-0.2.0-lead/verify_findings.py`) prints `fixed` for
  S7 and S7′ here, with S5, S6 and S8 unchanged at `fixed`.

  **THE TEST-SIDE HALF, and it is the same lesson `main` learned twice at
  `942df81` / `b53a537`.** `tests/test_vacuous_precondition.py` (added by
  this entry) declares `float64` and asked for x64 from nothing, and gated
  jax with `importorskip("jax", reason="needs jax")` — a custom reason
  replaces pytest's standard `could not import 'jax'`, which is the message
  `test_skip_inventory.py`'s `_IMPORT_GATE` matches to disclose the gate.
  Both are invisible with `JAX_ENABLE_X64=1` and jax installed, which is
  every local run. Fixed here with the house autouse module-scoped `_x64`
  fixture that saves AND restores, and the bare `importorskip` idiom. The
  three modules the previous entry's parent added
  (`test_assume_ledger`, `test_assume_scope_identity`,
  `test_assume_disclosure_claims`) carry the identical fixture, byte-for-byte
  what `942df81` put on `main` after this branch forked — brought onto the
  branch so the two-environment claim above is a measurement rather than a
  prediction about a merge.

*(no releases yet)*
