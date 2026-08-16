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

- **2026-08-14 (pre-release): the rational-`pow` row emitted about a
  DIFFERENT REAL FUNCTION than the program computes, and discharged it —
  `x ** 0.1` was enough.** Direction: **wrong VERIFIED → UNKNOWN**, plus
  **RAISE → Verdict** on the refutation side. Every affected VERIFIED was
  a claim about an expression the harness does not contain.

  A non-integer `pow` exponent was rationalised —
  `Fraction(e).limit_denominator(128)` — and admitted whenever
  `abs(float(frac) - e) <= 1e-12`. **That test cannot see the
  substitution it exists to detect.** A traced exponent is a binary64
  literal and a binary64 IS a dyadic rational: `0.1` denotes
  `3602879701896397/36028797018963968`, not `1/10`. `float(Fraction(1,10))`
  rounds back to the same double, so the measured error was **exactly
  0.0** while the two rationals differ by `5.55e-18` — and the emission
  then constrained `aux^10 = x0`, the exact tenth root. No threshold on a
  binary64 distance can exclude this; the comparison had to be between
  rationals, which is what it is now.

      x = any_array((), "float64", (1.0, 1e300))
      assert_(x ** 0.1 <= 1e30)

  returned **VERIFIED** with both backends answering `unsat`, a
  well-formed script, and no degradation note. It is false at the
  declared upper bound: `jnp` executes `1e300 ** 0.1` to
  `1.0000000000000038e+30 > 1e30`, and the exact real value of the traced
  expression at 120 significant digits is
  `1.000000000000003839824955626497…e+30` — above the bound — while the
  exact real value of the EMITTED expression, `x^(1/10)`, is
  `1.000000000000000005250476025520…e+30`, below it. The two oracles
  agree with each other and disagree with the script: the discrepancy is
  entirely the exponent substitution, not the documented ℝ-vs-float gap.
  The error is amplified by the base — `x^(a+ε) = x^a·e^(ε·ln x)`, so
  `ε = 5.55e-18` over a box reaching `1e300` is a relative `3.8e-15`,
  about 17 ulps — and the guard never looked at the declared box, so it
  could not have bounded it.

  The second admission route was the tolerance itself: `limit_denominator`
  always succeeds, so `0.5000000000001` was admitted as `1/2`,
  `2.0000000000001` as `2`, and `1e-13` as `0` — the last emitting
  `aux = 1.0` for an expression that ranges over `[0, 1.000000000069]`.
  All three were measured VERIFIED with a violation at the box maximum.

  **Which verdicts are retroactively invalid.** Any VERIFIED, from
  0.2.0 development only, whose harness contains `x ** e` with `e` a
  non-integer float whose exact binary64 value is **not** a dyadic
  rational `p/2^k` with `max(p, 2^k) <= 128`. In practice that is *every*
  non-integer exponent except the ones a reader would call exact
  anyway — `0.5`, `0.25`, `0.125`, `0.75`, `1.5`, `2.5`, `1/64`, `1/128`
  and their kin are faithful and unaffected; `x ** (1.0/3.0)`,
  `x ** (2.0/3.0)`, `x ** 0.1`, `x ** (1.0/10.0)`, `x ** (1.0/80.0)`,
  `x ** 0.7`, `x ** (1.0/7.0)` were all substitutions. **The row does not
  exist in `v0.1.0`** — verified at the tag: `pow` is absent from
  `obligation._ARITH`, so a `pow` slice there declines as an unsupported
  primitive and no released verdict can be affected. Interval-only
  verdicts are also unaffected: the interval leg alone answered UNKNOWN
  on every construction above, and the false VERIFIED was minted by the
  solver escalation.

  **What to re-run:** any recorded verdict from a harness containing a
  non-integer `pow` exponent. Re-`check()` it. An `escalation declined —
  'pow' exponent … denotes exactly p/q` note where a solver discharge
  used to be is this change, and the old VERIFIED was about a different
  function. A verdict that stays VERIFIED had a dyadic exponent and was
  always about the traced one.

  **THREE MORE DEFECTS ON THE SAME ROW, CLOSED IN THE SAME CHANGE,
  because they are one defect seen from different sides.**

  * **`q == 1` emitted a unary `(* aux)`.** SMT-LIB2's `Reals` theory
    declares `*` `:left-assoc` with arity ≥ 2, so that is not a term of
    the logic — and the two backends disagree about it silently. **cvc5
    1.3.4 SEGFAULTS** (the child dies with SIGSEGV and the parent reports
    a protocol violation, so the transport's degrade-don't-crash design
    is what kept the process alive); z3 accepts it and reads it as
    `aux`. Reachable only through the rationalisation above
    (`2.0000000000001 → 2/1`), and it turned a stelling emission bug into
    a note blaming cvc5. Every repeated product now renders through one
    helper (`smt._repeated_product`) that is correct at n = 0, 1 and ≥ 2;
    the `n == 1` arm is unreachable from today's callers and is there
    because "the n-fold product" has exactly one right answer at n = 1
    and a helper correct only for its current callers is how `(* aux)`
    got written.

  * **The replay was float64 under a verdict sentence claiming exact
    rational arithmetic.** Every REFUTED witness carries *"confirmed by
    independent exact-rational replay (fractions.Fraction arithmetic,
    pure Python, no solver)"*, while the rational-`pow` branch computed
    `Fraction(float(base) ** exp)` — a binary64 libm `pow`, rounded, then
    wrapped in a `Fraction`. False as provenance, and near a predicate
    boundary the rounding decided the answer. It also made the **public
    `check()` RAISE**: on `x ** 0.5 <= 2.0` over a box starting just
    above `4.0`, cvc5's model is a real violation and the emission is
    exactly right, but the float replay evaluated `float(w) ** 0.5` to
    `2.0`, called the predicate TRUE, and `_require_valid_refutation`
    raised `EmissionInfidelityError` — the one alarm that means *the
    emitted problem does not mean the obligation*, fired at a correct
    emission, naming the wrong culprit. The replay now extracts exact
    integer `q`-th roots (integer Newton, confirmed by exponentiation, no
    float anywhere) and, where the true value is irrational, REFUSES
    through the channel the codebase already had for a model nothing can
    replay — `witness not independently replayable`, UNKNOWN by policy.
    `witness_is_valid` no longer flattens "the replay cannot evaluate
    this" into "the emission is unfaithful": the first propagates as
    `ReplayDeclined` and degrades, the second still raises. The
    `OverflowError` the float `pow` produced on large operands
    (`x ** 1.5` over `[1e200, 1e250]`, uncaught, losing the whole
    escalation) is gone with the float, confirmed by re-running the
    reproducer rather than by inference.

    **WHICH REFUSALS ARE ON WHICH SIDE OF THAT LINE**, because the split
    is a decision about who is accused and nothing downstream re-derives
    it. `ReplayDeclined` is for a refusal whose emission is CORRECT and
    which only this evaluator cannot finish — the rational-`pow` root
    above, and nothing else today. The replay's other refusal, a
    value-changing `convert_element_type`, is the opposite and stays
    `ReplayError`: the emission writes a non-bool `convert_element_type`
    as the IDENTITY on its operand, so a script carrying a
    `float64 → float32` narrowing has the rounding simply absent and
    states a different function from the one the harness computes — a
    witness reaching that point accuses the script. Two guards decline
    such a slice long before replay, which is exactly why the raise is a
    tripwire and why demoting it to a decline was silent: reverting that
    one word left the entire suite green. It is now pinned by CHANNEL
    (`ReplayError` and not the `ReplayDeclined` subclass), together with
    the emission fact the channel rests on.

    **A message about a solver model may not crash on the model.** A model
    value is unbounded, and CPython raises `ValueError` on `int` → `str`
    past `sys.get_int_max_str_digits()` (4300), so `Fraction(3**10000, 2)`
    turns a message into a crash. Same hazard `smt._renderable` exists
    for, same posture: detect by attempting the conversion, report the
    operand's `bit_length()` instead of 4300 digits, never raise the
    process-global limit on a caller's behalf. **Six sites, found in three
    passes — each pass found the previous one incomplete, so treat the
    count as the number found, not the number that exist:**

    * Both refusals in `_exact_rational_power` — a clean decline became a
      `ValueError` out of the public `evaluate_predicate`. Introduced by
      this change.
    * Both box-escape messages in `witness_is_valid` — **pre-existing**,
      and on the worse side of the channel line: that string is the
      diagnosis of *the emitted problem does not mean the obligation*, so
      the crash replaced the loud alarm's only explanation with a
      traceback out of `fractions.py`.
    * `solvers._require_valid_refutation` — **also pre-existing, and it is
      why a third pass was needed.** Repairing `witness_is_valid` alone did
      not achieve its own stated purpose: the alarm assembled its
      diagnosis safely and then died one statement later, stringifying the
      same values to attach them to the exception.
    * `solvers.make_validated_witness` — pre-existing, on the *success*
      path, and **it is not the same problem and does not get the same
      answer**. `Witness.values` is DATA with a parsed contract —
      `(input name, exact rational)`, which `reproduce._point` reads back
      with `Fraction()` to re-execute the harness at the point — so a
      summarised value would name a *different* point and break that
      contract on another module's public surface. It fails closed
      instead: an unrenderable model declines through `ReplayDeclined`,
      the dispatch degrades to UNKNOWN with the reason quoted, and no
      REFUTED is minted from a witness that cannot be re-executed. Applying
      a message renderer to a data field is the category error, and it was
      made once here before the audit caught it.

    The renderer is public (`obligation.fraction_text`) because what a
    verdict says about a model value is *disclosure*, `smt._renderable` is
    the same discipline at the other end, and a rendering rule two modules
    must agree on should be nameable by both. **The reason first given for
    making it public was false** — "nothing else in `src/` imports a
    private name across modules"; measured by AST there are **50**, and
    `smt.py` alone takes thirteen from `obligation.py`. Recorded because
    this log is where a wrong justification for a published-surface change
    belongs, not only the change.

    No verdict was ever affected. The decline sites replaced an UNKNOWN
    that was already UNKNOWN; the alarm sites replace one raise with
    another; the witness site now declines where it previously crashed
    inside a guard that already degraded to UNKNOWN.

    **Reachability — and the first figure recorded here was wrong, which
    is the more useful half of this entry.** It said 26 backend runs over
    boxes and exponents chosen to maximise model size produced a largest
    model value of "16 decimal digits" against the 4300-digit cap, and
    concluded the hazard was out of reach. That measured the wrong
    quantity: the harnesses never nested `pow`, and the number counted a
    rendered `n/d` string rather than the integer term the cap actually
    applies to. Re-measured through `check()` on an ordinary traced
    harness — `x ** 0.5` nested eight deep on `[1, 1e300]`, threshold
    under the box maximum — a REFUTED witness carries a model term of
    **4091 digits: 95% of the cap.** The term size roughly doubles per
    nesting level (2035 at seven, 4091 at eight) and collapses again at
    nine, where the solver picks a different model.

    The conclusion the old sentence drew survives — nothing crossed the
    cap in any run, and every gate degrades to UNKNOWN rather than
    misreporting — but its *margin* does not, and the margin is the part a
    future reader would rely on. These arms are not defence against
    something unreachable; they are defence against something reached to
    within five percent by a two-line harness.

    Two further rendering sites were swept and **assessed as safe rather
    than repaired**, and are listed so the inventory is not read as
    exhaustive-by-omission: `solvers.py`'s cvc5 external-binary parser and
    its z3 wheel transport both meet an over-cap numeral and degrade to
    UNKNOWN (`values=[]` and a quoted `ValueError` respectively). The
    first labels a genuinely *rational* over-cap model `nonrational=True`
    — sound in direction, untrue as a label, and left as a known
    inaccuracy rather than silently corrected here.

  * **The fragment stamp gated on the wrong property.** A non-integer
    `pow` was recorded nonlinear only when its base descended from a
    declaration — but the aux encoding introduces `aux^q = x^p`, a
    product of a fresh symbol with itself, which is nonlinear whatever
    the base is. A rational `pow` over a declaration-independent base
    (reachable: put the constant inside a `jit`) was stamped `QF_LRA`
    while the emission wrote `(* aux aux)`, and **both** backends refused
    the script — so no verdict was minted from the mislabel, but the
    whole obligation was lost and the notes attributed it to the solvers.
    A more lenient backend that auto-widened the logic would have turned
    it into a single-backend discharge, which is the direction with no
    backstop.

  **Cost, measured against an independent exact oracle, and it is real.**
  852 harnesses of the shape `∀x ∈ [lo, hi]: x**e <= bound` — 37
  exponents × 6 boxes × 4 bound placements, one of them exactly at the
  box maximum's value so escalation rather than the interval leg is what
  decides. Because `x**e` is monotone for `e > 0` on a positive box, each
  claim is true iff `hi^p <= bound^q` where `p/q = Fraction(e)`, which is
  an **exact comparison in ℚ** for every dyadic exponent here (120-digit
  `decimal` for the rest, with a guard band); no stelling, no jax and no
  solver in that oracle. Both trees, same battery:

  |  | VERIFIED | of which FALSE | REFUTED | of which false | UNKNOWN | RAISED |
  |---|---|---|---|---|---|---|
  | before | 358 | **26** | 263 | 0 | 155 | **76** |
  | after | 285 | **0** | 164 | 0 | 403 | **0** |

  The moves: **26 VERIFIED → UNKNOWN where the old VERIFIED was FALSE**
  against the oracle; 47 VERIFIED → UNKNOWN and 139 REFUTED → UNKNOWN
  where the old verdict was right — the genuine capability loss, all 186
  of them at a non-dyadic exponent and all declining at admission; and
  the 76 raises becoming 40 correct REFUTEDs and 36 UNKNOWNs. **0 moves
  into VERIFIED or REFUTED from a definite verdict**, and no post-fix
  verdict in the battery disagrees with the oracle.

  **On the 432 cases whose exponent survives — every dyadic — the only
  change in the battery is that 40 crashes became correct REFUTEDs.**
  180 VERIFIED before and after, 124 REFUTED before and 164 after, 88
  UNKNOWN before and after, and those 88 are the same 88: models carrying
  a non-rational value, which were already `witness not independently
  replayable` and still are. So the exact replay costs nothing measurable
  where the emission is faithful; the whole cost is the exponents that
  were being substituted.

  Admitting non-dyadic exponents soundly is a larger feature — the
  substitution stamped as an assumption, its amplified error
  `|x^a − x^(p/q)| ≤ x^a·(e^{|δ|·ln hi} − 1)` bounded against the
  obligation's slack over the declared box, and the discharge direction
  barred until that bound exists — and it was deliberately not built
  here; it is recorded in the CHANGELOG's known-limitations list instead.

  **Also in this change, not a verdict move:** the rational branch's
  NUMERATOR was unbounded, so `x ** 100.5` (emitting `aux^2 = x^201`) was
  admitted while the strictly smaller `x ** 100` declined at the integer
  cap of 64, and `x ** 1000000000000.5` built a 600 KB script from a
  one-line harness before dying of MemoryError. One cap now bounds the
  degree of the emitted equation on both sides, and both `pow` caps state
  the quantity they bound.

  **Why no test caught it.** The suite's rational-`pow` coverage was
  written in the idiom the defect hides in: `x ** (1.0/3.0)`,
  `x ** (2.0/3.0)`, `x ** (1.0/10.0)`, `x ** (1.0/80.0)` (twice) — five
  rows, all asserting VERIFIED, every one of them an exponent the
  emission was substituting. They passed because the solver agreed with
  the script, and nothing compared the script to the harness. Those five
  are rewritten at dyadic exponents of the same shape (`0.125`, `0.75`,
  `1/16`, `1/128`) so the rows keep their subjects, and each carries the
  reason its exponent changed. Every construction above is a permanent
  regression test (`tests/test_pow_audit_findings.py`, 68 cases); ten
  targeted mutations — one per finding, plus one per repair above — were
  applied to a copy of the source and each was measured to redden at
  least one of them. Two mutations were measured NOT to redden anything
  and are recorded rather than counted: re-deriving the rational inside
  `emit` is the identity on a fraction the guard already admitted, so no
  test can separate it; and `.is_integer()` versus `== int(...)` differs
  only at inf/nan, which the literal decoder refuses earlier.

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

  **(3) AUDIT B3-2 — A FILTER'S EMPTY RESULT READ AS A POSITIVE CLAIM.**
  The condition (1) added is `not propagate.unaccounted_assumes(ledger,
  emitted_origins)`. That function is a FILTER OVER THE LEDGER, so it
  returns `()` in two situations a caller reading one value cannot tell
  apart: every recorded assume is accounted for, and **nothing is recorded
  for an assume that exists**. `solvers._dispatch_obligation` read it as
  `accounts_for_every_assume` — the positive claim that the region the
  solver ran over is inside the region EVERY assume of the query describes.

  * **An assume that produces no ledger entry.** The propagator's walk does
    not descend `scan` or `while_loop` bodies, so a `stelling_assume` inside
    one is never classified: no narrowing, no drop record, no ledger entry.
    Measured on this tree, jax 0.11.0, `JAX_ENABLE_X64=1`: `x, y ∈
    [-10, 10]`, `assume(x < y)` at top level and `assume(y < x)` inside a
    `lax.scan` body. The query states TWO assumes;
    `Propagation.assume_ledger` has ONE; `unaccounted_assumes(ledger, (0,))`
    is empty; the run returned **VERIFIED with no may-be-vacuous line** over
    `x < y ∧ y < x`, which no strict order admits (an exact-`Fraction` 41²
    grid over `[-10, 10]²` admits **0** points —
    `_scan_body_cycle_admits_no_point`, computed in the test rather than by
    the tool). It reaches the public `check_inductive_step` by the same
    route.
  * **The default failed OPEN, and the docstring said the opposite.**
    `_dispatch_obligation`'s `assume_ledger` defaulted to `()` on the stated
    ground that "an empty ledger accounts for nothing that was forwarded, so
    a caller that forgets it gets `REGION_UNCERTIFIED` … never a clean
    stamp". An empty ledger filters to an empty result, which read as the
    positive claim. Measured on the cone-split cycle above, whose region is
    provably empty: with `assume_ledger=()` the run's `region_uncertified`
    was `()` and **no may-be-vacuous line was stamped**; with the real
    ledger, `(0,)` and the line present.

  **THE REPAIR, ON MACHINERY THAT IS ALREADY TOTAL.** The claim
  `_region_answer` reads is now the CONJUNCTION of the filter and a
  COMPLETENESS check: `not unaccounted_assumes(...) and
  every_assume_recorded`, where `every_assume_recorded` is
  `propagate.ledger_covers(propagation.assume_ledger, closed.jaxpr)` —
  whether the ledger has a record for every `stelling_assume` equation the
  query CONTAINS. The requirement is the STATIC set,
  `propagate._assume_equation_ids`, which collects every assume equation
  sub-jaxprs included *"whether or not any walk reaches it"*; the
  non-emptiness certificate's own requirement already rests on it, which is
  exactly why THAT path never had this hole. The join is on a new
  `AssumeDisposition.eqn_id` (`id()` of the assume equation, the identity
  `_Propagator.assume_witness` is already keyed on), stamped at the four
  ledger write sites, and held OUT of the dataclass's equality
  (`compare=False`) because a process-local key must not be what a
  cross-run comparison fails on. `assume_ledger` and `every_assume_recorded`
  are both REQUIRED keywords now — the discipline `_region_answer`'s own
  accounting keyword already carries, one level down.

  **`_assume_equation_ids` IS GENUINELY TOTAL OVER SUB-JAXPRS, CHECKED
  RATHER THAN TRUSTED.** Against an INDEPENDENT walk of the raw jax jaxpr
  (not stelling's IR, not stelling's traversal), on `scan`, `while_loop`,
  `jit`-inside-`cond`, and `cond`-inside-`jit`-inside-`scan`: **2 of 2 on
  every shape, all four agreeing**
  (`test_the_static_assume_set_is_total_over_sub_jaxprs`). It rests on
  `coverage.sub_jaxprs`, which yields every `Jaxpr`/`ClosedJaxpr` held in an
  equation's params through tuples and named-tuple params, and on
  `_jax_compat`'s param transcription, which converts every jaxpr-valued
  param and RAISES `UnsupportedParamError` rather than dropping one. The one
  boundary: `custom_jvp_call.jvp_jaxpr_fun` and `custom_vjp_call`'s thunks
  are transcribed as opaque, but those are DERIVATIVE jaxprs — the primal
  `call_jaxpr` transcribes normally, so a user `assume` on the primal path
  is not behind them.

  **SCOPE, STATED SO IT IS NOT OVERCLAIMED.** This closes the
  un-recorded-assume route **for the admitted-region rule only**. The same
  root cause — a `stelling_assume` the walk never reaches — also reaches the
  withholding rule at the end of `escalate`, where `unaccounted_assumes` is
  read the same way and can produce a false REFUTED. That face is
  PRE-EXISTING, reaches released 0.1.0, and is **not** fixed here.

  **RED/GREEN.** Deleting the `every_assume_recorded` conjunct from
  `_region_answer`'s argument reddens **3** tests, all in
  `tests/test_vacuous_precondition.py` §8
  (`…_is_not_stamped_clean`, `…_names_the_unrecorded_assume_…`,
  `…_fails_CLOSED`); no pre-existing test detects it. Full suite green in
  both precision environments after the repair (counts below).

  **WHAT MOVES, MEASURED ON THIS BUILD.** The 288-harness sweep re-run
  unmodified against `d2fcff2` and against this build (jax 0.11.0,
  `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu`, z3 + cvc5 wheels), comparing the
  two `--json=` row maps entry by entry: **54 REFUTED / 90 UNKNOWN / 72
  VERIFIED / 72 RAISED, 0 FALSE VERIFIED, 0 FALSE REFUTED, 0 vacuous
  VERIFIED — identical to the table above, 0 of 288 rows moved, and 1044
  solver invocations on each tree** (720 obligation-script + 324
  admitted-region, counted at the transport-entry boundary
  `solvers._Backend.run`, the act `_Ledger.spawns` counts — an earlier
  revision of this paragraph recorded 756 on a basis it did not state and
  that does not reproduce). The may-be-vacuous rate among the 72 VERIFIEDs
  is **18 on each tree**, so the B3-2 repair adds no caveat here either: the
  sweep carries no `scan`- or `while_loop`-bodied assume, which is the only
  shape it fires on. No verdict moves and no solver call is added; what
  moves is disclosure. On a purpose-built
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

  **AND THE COST IS NOT INTRINSIC TO THE RULE — IT IS THE PROBE GRID'S
  SHAPE, ISOLATED (audit B3-2, non-blocking; recorded here so the next
  person does not re-derive it).** Measured on this tree:
  `Propagation.region_inhabited` is **`False`** for `assume(x < y);
  assume(y < z)` over `[-10, 10]³` and **`True`** for the same chain written
  with `<=`, two harnesses one character apart. **THE GRID CANNOT CERTIFY A
  STRICT CHAIN**, and the mechanism is in `propagate._probe_fraction`: the
  first three probes are ANCHORS that put every declaration at the SAME
  fraction of its box (`0.0`, `1.0`, `0.5`), where no strict inequality
  between two declarations can hold; the other 13 offset the fraction by
  `element * 0.7548776662466927` mod 1 per declaration, which for three
  declarations steps DOWNWARD both times. Enumerated over all 16 probes:
  **0 produce `f₀ < f₁ < f₂`** — so no probe point of a 3-variable strict
  chain is ever ordered, and the certificate declines for a reason that is
  about the grid, not about the region. (It is not a blanket rule against
  strict relations: a 2-variable `assume(x < y)` IS certified, at probe 4 —
  3 of the 16 probes give `f₀ < f₁`.)

  So the 18-of-72 rate measures which points the grid happens to visit. One
  interior ORDERED probe point, or the whole-query admitted-region script
  named above, removes most of it. The probe is deliberately NOT changed
  here: it is one-sided and every `False` path leaves the run byte-identical
  (`test_a_failed_certificate_search_changes_nothing_at_all`), so widening
  it is a separate change owing its own cost argument.

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
  | **B3-2**: drop the `every_assume_recorded` conjunct from the accounting | **3** |

  All nine are in `tests/test_vacuous_precondition.py`; no pre-existing test
  detects any of the four. The B3-2 mutant was run at this commit on the
  full suite with `JAX_ENABLE_X64=1`, restoring the file byte-for-byte
  afterwards; the three it reddens are §8's
  `…_is_not_stamped_clean`, `…_names_the_unrecorded_assume_…` and
  `…_fails_CLOSED`.

  **SUITE COUNTS, EACH WITH THE ENVIRONMENT THAT PRODUCED IT.** Every
  figure below was re-run for this amendment; where a previously recorded
  one did not reproduce it is corrected and the discrepancy named, because
  a count without its environment is not a measurement. All runs: jax
  0.11.0, python 3.12, z3 + cvc5 wheels, `python -m pytest tests/ -q`.

  | tree | environment | result |
  |---|---|---|
  | this build | `JAX_ENABLE_X64=1` | **2993 passed, 10 skipped** |
  | this build | no `JAX_ENABLE_X64` (what CI runs) | **2994 passed, 9 skipped** |
  | `1dc1b52` (git checkout) | `JAX_ENABLE_X64=1` | 2969 passed, 10 skipped |
  | `1dc1b52` (git checkout) | no `JAX_ENABLE_X64` | **67 failed**, 2903 passed, 9 skipped |
  | `1dc1b52` (`git archive` export) | no `JAX_ENABLE_X64` | 68 failed, 2900 passed, 11 skipped |
  | this build | jax absent, `[solvers]`-only venv | **1475 passed, 108 skipped, 0 failed** |
  | this build | jax MASKED in the jax venv (`-p` plugin) | 1476 passed, 107 skipped, 0 failed |

  The two precision rows' skip SETS differ by exactly one entry —
  `tests/test_tripwire_arm.py:643`, the threefry case whose skip condition
  IS x64-on — and by nothing else; the ninth run of the test it stops
  skipping is the extra pass.

  **The `1dc1b52` without-x64 figure was recorded as 75 and does not
  reproduce at any measurement.** In a real git checkout of that commit it
  is **67 failed / 2903 passed / 9 skipped**; in a `git archive` export of
  the same commit, 68 / 2900 / 11 — the export forces the two git-gated
  skips (`test_reuse_pins.py`, `test_sdist_contents.py`) and one extra
  failure, `test_no_session_skip_is_undisclosed`, which is reacting to
  those skips. `pytest-randomly` is not installed in any of these venvs, so
  ordering cannot account for the difference; 75 has no environment on
  record and is withdrawn. **The checkout figure is the one a reader can
  regenerate** — `git clone`, `git checkout 1dc1b52`, run pytest — and it
  is what the test-side half below is the reason for. The 67 are 36 in
  `test_assume_scope_identity.py`, 18 in `test_vacuous_precondition.py`, 11
  in `test_assume_ledger.py` and 2 in `test_assume_disclosure_claims.py`.

  **The jax-free figure was recorded as 1476 / 107 without naming its
  environment, and BOTH numbers are right in the environment that produced
  them.** With `jax` MASKED inside the jax venv (`sys.modules["jax"] =
  None` from a `-p` plugin, so the gates see an absent package during
  collection) it is 1476 / 107, reproduced exactly. In a venv that has only
  `[solvers]` + pytest — no jax and, consequently, **no numpy** — it is
  1475 / 108, and that venv is the shape of CI's jax-free lane. The single
  differing test is `tests/test_reproduce.py:644`, which skips for want of
  numpy in the second and runs in the first. Zero failures either way, and
  no undisclosed skip in either. The `[solvers]`-only figure is quoted
  first above because it is the one CI reproduces.

  The audit's own re-checker (`audit-0.2.0-lead/verify_findings.py`), re-run
  on this build, prints `fixed` for S7 and S7′, with S5, S6 and S8 unchanged
  at `fixed`. It also prints **`STILL PRESENT` for S13** — *"an assume the
  walk never descends leaves no trace to withhold on"* — which is the SAME
  root cause as (3) above reaching the WITHHOLDING rule instead of the
  admitted-region one, and is deliberately not touched here. Reproduced
  independently while measuring (3): with `assume(x < y)` and
  `assume(y < x)` BOTH inside a `lax.scan` body, the ledger is empty,
  `assume_dropped` is False, nothing is withheld, and `assert_(x - y <=
  0.0)` comes back **REFUTED** with a witness that violates the
  precondition — over a region no strict order admits. Separate batch.

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

- **2026-08-15 (pre-release): an IEEE divisor box that reaches zero was
  divided as if the zero had a sign, and it flipped verdicts in ALL FOUR
  FORMATS.** Audit 0.2.0 S10 (found by that audit's IEEE-formats lens as
  § F2), fixed in `interval.ieee_div` and `interval.ieee_div_fmt`.

  **The defect.** Both kernels special-cased a divisor interval touching
  zero at exactly one boundary. `b = [lo, 0]` with `lo < 0` was read —
  the comment said so — as *"divisor approaches 0 from below"*, so for a
  non-positive dividend the quotient was `[ahi/blo, +inf]`, excluding
  `-inf`. **Under IEEE the divisor does not approach zero; it IS zero at
  that endpoint, and the sign of `x/0` is `sign(x)` XOR the SIGN BIT OF
  THE ZERO.** `+0.0 == 0.0`, so `+0.0` is a value of `[lo, 0]`; there the
  quotient is `-inf`, which the box did not contain. An interval endpoint
  has no sign bit, so *which boundary is zero* is not enough information
  to make the tightening at all — no test on the endpoints' positions
  could have repaired it.

  **Measured, all four formats** (`audit-0.2.0-ieee/F2b_div_all_formats.py`,
  re-run here on this branch before and after; harness `a = [-2,-2]`,
  `x = [-1, 0]`, `assert(a/x > 0)`, and its mirror `a = [2,2]`,
  `x = [0, 1]` at `-0.0`):

  ```
                      before        after      jax at the zero
    float16          VERIFIED      UNKNOWN     y = -inf,  y > 0 -> False
    bfloat16         VERIFIED      UNKNOWN     y = -inf,  y > 0 -> False
    float32          VERIFIED      UNKNOWN     y = -inf,  y > 0 -> False
    float64          VERIFIED      UNKNOWN     y = -inf,  y > 0 -> False
  ```

  Eight false VERIFIEDs (four formats × two mirror shapes), every one of
  them refuted by executing the same harness in jax at a point of its own
  declared box.

  **IT IS A 0.2.0 REGRESSION, and the lens that found it got that
  wrong.** § F2 concluded it "predates 0.2.0" from the fact that binary64
  is affected — which conflates the *format* with the *feature*. Measured
  at the kernel on both trees, `v0.1.0` extracted with `git archive` and
  run by `PYTHONPATH` (no worktree, repo untouched), this branch's base at
  `f0b34cd`:

  ```
  --- v0.1.0 ---
  ieee_div([-2,-2], [-1,0]) -> (-inf, inf)   contains -inf? True
  ieee_div([2,2],   [0,1])  -> (-inf, inf)   contains -inf? True
  hasattr(iv, "ieee_div_fmt") False   hasattr(iv, "boundary_div") False
  --- pre-fix tree (f0b34cd) ---
  ieee_div([-2,-2], [-1,0]) -> (2.0, inf)    contains -inf? False
  ieee_div([2,2],   [0,1])  -> (2.0, inf)    contains -inf? False
  hasattr(iv, "ieee_div_fmt") True    hasattr(iv, "boundary_div") True
  ```

  0.1.0 returned the sound hull for any zero-containing divisor. The
  boundary-aware branch arrived at `3328c9b` (2026-08-13, *"IEEE
  assume-bump + boundary-aware division"*), one day after the `v0.1.0`
  tag and in no tag since — `git describe --tags --contains 3328c9b`
  names none. **So the released 0.1.0 is NOT affected**, including
  through `propagate(closed, semantics="ieee")`, which was 0.1.0's only
  door to ieee mode.

  **WHICH PRIOR VERDICTS ARE RETROACTIVELY INVALID, and how to
  recognise one.** Any verdict produced by a tree containing `3328c9b`
  and not this fix, whose stamp names **ieee** semantics (arithmetic mode
  `interval/ieee` or `interval/ieee-fmt`, any of the four formats), on a
  query containing a float `div` **whose divisor's propagated box reaches
  zero at exactly one boundary** — `[0, hi]` or `[lo, 0]`. Only the
  DEFINITE directions are at risk: a VERIFIED or a set-level REFUTED that
  the interval leg decided downstream of such a division. An UNKNOWN was
  never wrong, and no released verdict is affected because 0.2.0 has not
  shipped.

  Three shapes reach it, and the third needs no signed-zero reasoning in
  the harness at all:

  * a declared box with an endpoint at zero (`any_array(..., (-1.0, 0.0))`);
  * `assume(b > 0)` in ieee mode — the bump lands on the format's smallest
    SUBNORMAL, which the DAZ haze then hulls back to `[0, hi]`;
  * **any subnormal-reaching divisor**, because `_elt_haze_fmt` hulls a
    band-touching interval with the positive literal `0.0` and thereby
    manufactures a zero endpoint whose sign the model has already
    discarded. The 0.2.0 audit's fuzzer found six further instances of
    this class through this route alone.

  **What to re-run.** Any recorded ieee-mode verdict matching the shape
  above: re-run it on a tree with this fix and take the new verdict. A
  cheap pre-filter that needs no re-run: if the query has no `div`, or if
  every divisor's propagated box is bounded away from zero *and* clear of
  the format's subnormal band, the verdict is unaffected.

  **The fix, and the shape NOT chosen.** A zero-containing divisor now
  divides to ⊤ under ieee, with no case split on where the zero sits —
  which is what `v0.1.0` did, and which is not merely the conservative
  answer but the EXACT hull: whenever the dividend can be nonzero both
  `±inf` are attainable and no narrower closed interval holds both, and
  in the only remaining case (dividend exactly `[0,0]`) every quotient is
  `0/0 = NaN` or `±0`, where the NaN flag is already set. The alternative
  was to carry the zero's sign in the IEEE domain so a box could
  distinguish "reaches `-0.0`" from "reaches `+0.0`". That is strictly
  more capable and was rejected on cost and on hazard: it is a new
  lattice element that must be threaded through every kernel that can
  produce or consume a signed zero (the haze, `neg`/`abs`, `select_n`,
  the comparisons, `reduce_sum`'s seed, every conversion), and a
  half-done version is worse than none — it would put a trustworthy sign
  bit on values that only some producers set, which is what this defect
  already was. It would also buy little today: the declaration surface
  has no way to say "this input reaches `+0.0` but not `-0.0`", and the
  haze's manufactured zeros have no determinate sign to record (the same
  audit measured `x + x` flushing to `-0.0` while `jnp.sum` seeds `+0.0`
  on the same backend). Recorded here as a decision, with its reasons, so
  that re-adding a tightening requires arguing against them.

  **THE REAL-MODE `boundary_div` KERNEL IS NOT WRONG FOR THIS REASON AND
  WAS NOT CHANGED** — the two kernels disagree deliberately, and the
  claim was verified rather than assumed. ℝ has one zero and `a/0` is
  undefined there, so the box must cover only `b ≠ 0`. Checked in exact
  rational arithmetic over 12 dividend boxes × 10 one-sided-boundary
  divisor boxes, at values crowding the zero endpoint (relative offsets
  down to `1e-300` of the span): **7,560 quotients, 0 containment
  failures.** So `[2, ∞)` is right for `[-2,-2] / [-1,0]` in ℝ and ⊤ is
  right for the same operands under ieee.
  `tests/test_ieee_zero_divisor_and_mul_exact.py` pins that DIFFERENCE
  with its reasoning, because the next reader to see the two kernels side
  by side will assume they should agree.

  **THE FIGURE ABOVE WAS 31,350 OVER TEN CASES AND NO RUN IN THE TREE
  PRODUCED IT** (audit 0.2.0 B5-4). The shipped test had five cases,
  executed 195 quotients, and asserted only `checked > 100`; the claim it
  was standing in for is amply true — an independent exact-rational sweep
  by the auditor did 3,291,024 quotients with 0 failures — but a reader
  chasing 31,350 found 195. The sweep is now the larger one described
  above and the count is `assert`ed exactly, against a module constant
  the sentence quotes: `BOUNDARY_DIV_SWEEP_QUOTIENTS = 7560`. A drift in
  either direction reddens the suite with "update both or neither"
  instead of leaving prose to be believed. **This was the third
  fabricated figure in this campaign**, and the mechanism is why the fix
  is an assertion rather than a corrected number.

  **AND A SOUND KERNEL IS NOT A SOUND ROW — see the B5-1 entry below.**
  Every sentence above is about `boundary_div`'s behaviour *given* `b ≠
  0`. Whether `b ≠ 0` holds is a separate question that a sweep of the
  kernel cannot see, and the transfer answered it wrong: it inferred
  "the zero can be dropped" from the SHAPE of the box. That is how a
  kernel this well measured came to sit underneath a false VERIFIED.

  **The coverage this costs, measured and named.** In ieee mode the
  advertised pair *"IEEE assume-bump + boundary-aware division"* no longer
  decides `assume(b > 0); a / b`: the bump lands on the format's smallest
  subnormal, the DAZ haze hulls that back to `[0, hi]` (measured on
  float32: `_elt_haze_fmt(2**-149, 10.0, 2**-126) == (0.0, 10.0)`), and a
  zero-containing divisor is now ⊤. That pin was a passing test asserting
  `discharged`; it now asserts `unknown` and says why. **The row is not
  dead** — an assume whose bound clears the format's subnormal band keeps
  its quotient, measured on the same query at `assume(b > 1e-30)` in
  float32, which still discharges — and real mode is untouched. That is
  the whole price: one shape, in one mode, and both sides of the boundary
  are pinned.

  **Also removed by the fix, and found only by driving the kernel:** the
  boundary branch could RAISE. `ieee_div([-inf,-inf], [-inf, 0])` took
  the `bhi == 0` arm and computed `ahi/blo = -inf/-inf = NaN`, which
  `IntervalArray.__post_init__` rejects — an `IntervalError` out of a
  kernel whose contract is to degrade, not to crash. 20 such box pairs in
  the binary64 sweep below, 16/20/16 in the three narrow formats.

  **The sweep, and what the previous one missed.** An exact containment
  sweep drives `mul`, `ieee_mul`, `ieee_div`, `ieee_mul_fmt` and
  `ieee_div_fmt` over every ordered endpoint pair from a 28-value
  adversarial pool (`±0.0`, `±5e-324`, `±2.225e-308`, `±1e300`, `±FMAX`,
  `±inf`, `nextafter(1.0, ±inf)`, and magnitudes subnormal in a *narrower*
  format), 406 boxes and 164,836 box pairs per row, checking that every
  returned box contains every value the arithmetic can produce at points
  of the operand boxes — with **`+0.0` and `-0.0` kept apart on the input
  side**, which is exactly what the sweep before it did not do (it drove
  real-mode `div` and `boundary_div`, never `ieee_div`, and deduplicated
  its sample points with `==`, under which the two zeros are one point).
  Boxes are built from POOL INDICES rather than value comparisons, which
  is what keeps the two zeros apart. Real-mode `mul` is judged against
  exact rational products; `ieee_mul`/`ieee_div` against the binary64
  value the program actually computes (their corner hull is the native
  product with no outward rounding, so the REAL product is the wrong
  yardstick); the narrow formats against the exact real value, which is
  sufficient once `_ieee_round_box` has rounded the box outward onto the
  target grid, and which assumes nothing about how that format rounds.

  **THE TABLE BELOW WAS AN OUT-OF-TREE MEASUREMENT AND IS NOW A RUN**
  (B5 follow-up audit). Its predecessor read `0 / 1,366,561`,
  `40,032 / 6,707,920`, `+ 20 RAISED`, `116,548` NaN samples and
  `301 boxes / 90,601 box pairs`; those counts came from a sweep that was
  never committed, so `grep` for them found nothing in `tests/` or
  `src/`. Nothing about them was wrong — the arithmetic was internally
  consistent and an independent sweep agreed with the conclusions — but
  they sat three paragraphs above the paragraph explaining why exactly
  that shape is a problem, and immediately after the sibling figure that
  had just been converted to an exact assert. **Every number below is
  from a run of `tests/ieee_containment_sweep.py` on this branch**, and
  `tests/test_ieee_zero_divisor_and_mul_exact.py` asserts all four
  columns of all nine rows exactly, against the module constants
  `POST_FIX_ROWS`, `PRE_FIX_IEEE_DIV_F64`, `POOL_SIZE`, `BOX_COUNT` and
  `BOX_PAIRS` — so drift in either direction reddens the suite with
  "update both or neither". Reproduce with:

      PYTHONPATH=src python tests/ieee_containment_sweep.py

  Measured 2026-08-15 at `0dff2a9` + this commit, CPython 3.12.3, jax
  0.11.0, numpy 2.5.1, Linux x86_64, `JAX_ENABLE_X64=1` (the sweep drives
  the kernels directly and does not read the flag).

  ```
  kernel        format      failures / samples      NaN samples   RAISED
  real mul      R                  0 /   851,929             0        0
  ieee_mul      float64            0 /   962,361        29,348        0
  ieee_div      float64            0 /   962,361        67,373        0
  ieee_mul_fmt  float32            0 /   962,361       104,632        0
  ieee_div_fmt  float32            0 /   962,361       128,657        0
  ieee_mul_fmt  float16            0 /   962,361       115,784        0
  ieee_div_fmt  float16            0 /   962,361       151,505        0
  ieee_mul_fmt  bfloat16           0 /   962,361       104,632        0
  ieee_div_fmt  bfloat16           0 /   962,361       128,657        0

  positive control — the PRE-FIX ieee_div, same grid:
  ieee_div      float64       15,048 /   962,361        67,373        0
  ```

  **A battery that has never failed is not evidence**, so the pre-fix
  kernel is kept in the sweep module as `prefix_ieee_div` and driven over
  the same grid: it fails 15,048 of 962,361 samples, and a run that ever
  reports 0 there fails the suite. `RAISED` is 0 in that column and that
  is not a hole — the raise this entry records above
  (`ieee_div([-inf,-inf], [-inf, 0])` computing `-inf/-inf`) was closed
  inside `boundary_div` itself by a different fix, so the pre-fix
  *transfer* driven through today's kernel degrades instead of crashing.
  The `20` / `16/20/16` raise counts in the paragraph above it remain an
  **out-of-tree measurement at `f0b34cd`**, named as such because no test
  in this tree can reproduce a crash in code the tree no longer contains.

  A NaN result is checked against the `made_nan` flag rather than
  containment, and every NaN sample counted above is flagged.
  **What the generator could not produce**, said plainly: it samples a
  finite subset of each box (the endpoints, and zero when the box
  straddles it) rather than every float, so it is a containment
  *battery*, not a proof; it drives the kernels and the `_ieee_round_box`
  composition directly, not the traced pipeline; and it says nothing
  about the transfers this batch did not touch.

- **2026-08-15 (pre-release, same batch): `mul` was the only arithmetic
  transfer with no exact-rational path, and the missing ulp decided
  verdicts.** Audit 0.2.0 M16. **Sound in both directions — no verdict
  was ever wrong — and it is in this log because it MOVED verdicts**, in
  the UNKNOWN → definite direction, on a shape the release advertises.

  `add` routes through `_exactable` → `Fraction` and returns the exact
  endpoint when it is representable; `div` has the same branch; `mul`
  bumped unconditionally. Measured on this branch, before and after:

  ```
                              before                              after
  mul([2,3],[2,3])   (3.9999999999999996, 9.000000000000002)   (4.0, 9.0)
  mul([0,4],[0,4])   (-5e-324, 16.000000000000004)             (0.0, 16.0)
  reduce_sum(x*x)    (-1e-323, 32.00000000000001)              (0.0, 32.0)
  ```

  The exactly-zero corner bumped **below zero**, which defeats
  `reduce_sum`'s nonnegative clamp: `Σ xᵢ²` became a true straddle, so
  `boundary_div` was never reached and the division declined — with a
  message recommending `assume(divisor > 0)`, which the caller had
  effectively already done. The same real property, three spellings, on
  this branch:

  ```
                     before      after
    via_mul         UNKNOWN    VERIFIED
    via_ipow       VERIFIED    VERIFIED
    via_square     VERIFIED    VERIFIED
  ```

  **The verdict depended on how the user spelled "squared"**, and the
  spelling that lost is the `safe_mask` sum-of-squares residual that the
  0.2.0 boundary-division row was built for. Clamping `mul`'s sign as a
  special case would have been a bandaid: it leaves `[2,3]×[2,3]` inexact,
  and the inexactness is the defect.

  **THE FINDING'S OWN HEADLINE IS NOT QUITE TRUE, and the remainder is
  left standing deliberately.** M16 says `mul` is "the only arithmetic
  transfer with no exact-rational path". It was not: `dot_general` carries
  an INLINED COPY of the same corner rule (`_prod` + `_down`/`_up`), and
  its docstring cited *"`mul`'s rule"* as its authority — so converting
  `mul` alone silently split one rule into two. Measured after this fix,
  on the same operands:

  ```
  reduce_sum(mul(x, x))  over x in [0,4]^2  ->  (0.0, 32.0)
  dot_general(x, x)      same contraction   ->  (-1e-323, 32.00000000000001)
  ```

  So the M16 shape survived one level up: a sum of squares written
  `jnp.dot(x, x)` lost its zero floor where `jnp.sum(x*x)` no longer did.
  Sound — a wider box only loses precision — and the docstring stated the
  divergence rather than claiming the rule it no longer followed.

  **CONVERTED, and the reason first given for not converting it was the
  wrong one** (audit 0.2.0 B5-2). That reason was *"changing a
  contraction's numerics needs its own measurement of the
  association-order argument"*. The accumulation in `dot_general` ALREADY
  used `_add_lo`/`_add_hi` — the exact-when-representable route — and only
  the four product corners bumped; a product's corners have nothing to do
  with association order, and the `nonneg = nonneg and plo >= 0.0` clamp
  the bump defeated is not "the M16 shape one level up", it is M16, in a
  second copy. The two are now ONE function, `interval._mul_corners`,
  called by `mul` and by `dot_general`'s per-term product, so the next
  conversion cannot convert one of them. Measured on this branch:

  ```
                                        before                    after
  reduce_sum(mul(x,x)), x in [0,4]^2    (0.0, 32.0)               (0.0, 32.0)
  dot_general(x, x), same contraction   (-1e-323, 32.00000000000001)  (0.0, 32.0)
  matmul of [2,3]-valued 2x2            (7.999999999999999,       (8.0, 18.0)
                                         18.000000000000004)
  ```

  Containment evidence, the same shape `mul`'s: over every ordered
  endpoint pair from an 8-value dyadic pool (1,296 box pairs), a two-term
  1-D contraction's box is checked against the exact rational image
  `2·[min corner, max corner]` — and, because every value in the pool is a
  small dyadic, is asserted EQUAL to it, not merely containing.
  0 failures. The infinite-endpoint confinement is shared with `mul` by
  construction and pinned by one test that compares the two on the same
  operands.

  **The interaction with B5-1, said plainly**: a `dot_general`-floored sum
  of squares now reaches the `div` transfer with a `[0, S]` box exactly as
  a `reduce_sum`-floored one does, so it meets the same certificate gate.
  `jnp.dot(x, x)` is the FOURTH spelling in the three-spellings control
  for that reason, and `dot_general` carries the strict-sign certificate
  for the same reason.

  **The ieee `mul` kernels deliberately do NOT take this route**, and the
  reason is not symmetry-of-effort: under ieee the value the program has
  IS `fl(x*y)`, and the native binary64 corner products already ARE that
  value — exactly for binary64 operands, and exactly for the narrower
  formats too (a product of two float32/bfloat16/float16 values needs at
  most 48 significand bits and stays well inside binary64's exponent
  range, after which `_ieee_round_box` rounds outward onto the target
  grid). Routing through `Fraction` there would bracket the REAL product
  and widen by up to an ulp on each side — slack where there is none.

  That is the whole reason, and the overflow argument that used to
  accompany it is WITHDRAWN (audit 0.2.0 B5-6). It ran: two binary64
  operands near `FMAX` multiply to `inf`, so the exact route's
  `[FMAX, inf]` would "name a value the program cannot compute". True of
  binary64, and it proves too much — the row's own narrow-format path
  already returns exactly that box, because the corners are computed in
  binary64 and only then rounded outward onto the narrow grid. Measured,
  float32: `ieee_mul_fmt` on `FMAX × FMAX` returns
  `(3.4028234663852886e+38, inf)` while `np.float32` computes `inf`.
  Sound in both places — the box holds the value — but an argument that
  condemns the sibling row cannot be this row's reason, and
  `test_the_overflow_argument_for_ieee_mul_proves_too_much` keeps the
  measurement beside the docstring that no longer makes the claim.

  **Measured coverage effects, counted rather than characterised.**
  Fifteen pre-existing tests changed status across the two fixes — **13
  from this one, 2 from S10** — and each is re-posed with its reason in
  place. (A sixteenth, `test_no_untracked_file_anywhere_would_ship`,
  reddened only because the new test file was untracked; not a behaviour
  change.)

  From `mul`, the interval leg now decides what a solver used to, or what
  nothing used to: the JAX-Fluids `beta_0 >= 0` acceptance control (a sum
  of squares — now discharged by intervals, which is why the escalating
  control moved to `beta_1 <= 18.0`, true over the box with a measured
  vertex maximum of 17.333… against a propagated box reaching 18.333…);
  the T2 coefficient-contrast obligation at the equality-attaining
  boundary (`C*min` starts at exactly 200 now, so the closed contrast
  decides where it used to escalate — and the escalation pin moved to the
  n = 1 field, where max and min are the same element and the dependency
  problem keeps it undecidable for good); the T3 symmetry pair at an
  all-point envelope (`0.0 * t` is the exact `[0,0]`, so the seam between
  the triple and array paths closed); the all-point contract envelope,
  which no longer invokes a solver at all; and four unit pins that
  asserted the bump directly.

  Three audit-pinned constructions had to be re-posed onto transfers that
  still bracket, because their subject was the pad and never `mul`: the
  uncertified-precondition channel B (audit F7) moved to `sqrt`, whose
  endpoints are irrational in general so that no exactness discipline can
  ever remove its bracket — it is the THIRD transfer to carry that
  channel, after `add` and `mul` were each converted out from under it,
  and the note predicting this move was already in the file; the
  undecided-detail exhibit moved to `exp`; and the emission-infidelity
  alarm's `true_over_box_query` moved from `x*x <= 1.0` to `x*x >= 0.0`,
  which straddles for a reason no exactness work can remove (the interval
  domain cannot see that the two operands are the same variable).

  **THE "OTHER DIRECTION" SENTENCE THIS ENTRY ORIGINALLY CARRIED WAS
  FALSE, and its falsity is the B5-1 entry below.** It read: *"In the
  other direction the loss is the S10 entry's, not this one's: nothing
  here made any verdict less decidable."* Nothing here made a verdict less
  decidable — but M16 opened a composition that made one WRONG, by moving
  a sum-of-squares residual off the true-straddle decline and onto the one
  zero-containing shape that did not decline. An entry that verifies its
  own ℝ-containment claim and then stops has not finished: the question is
  not only "is this transfer's box sound" but "what does this transfer now
  reach". Corrected in place rather than deleted, because the sentence's
  shape — a true local claim standing in for an unexamined global one — is
  the recurring failure this log exists to record.

- **2026-08-15 (pre-release, B5 follow-up): a real-mode divisor box that
  REACHES zero dropped the zero and minted a definite verdict.** Audit
  0.2.0 B5-1. **FALSE VERIFIED, real mode**, made reachable by the M16 fix
  above.

  **The defect.** `div`'s zero-containing divisor had four shapes and
  three of them declined, every one citing the same fact — ℝ has no value
  at `a/0`:

  | divisor box | before |
  |---|---|
  | `[0, 0]` | decline — *"division by zero is undefined"* |
  | `lo < 0 < hi` | decline — `DIV_STRADDLE_DECLINE` |
  | (`sqrt` of a negative box, the ℝ-undefined precedent) | decline |
  | `[lo, 0]` / `[0, hi]` | **`boundary_div`, which EXCLUDES the zero, and a definite verdict from the rest** |

  Before M16, `Σxᵢ²` bumped its zero corner below zero, so a residual
  `Σxᵢ² − c` was a TRUE STRADDLE and took row 2. With `mul` exact it
  floors at exactly `0.0`, lands on row 4, and the drop happens silently.
  Measured on this branch, `x` declared `float64 [0, 2]²`:

  ```
  1.0 / (jnp.sum(x*x) - 8.0)  <=  -0.125

    stelling                  VERIFIED
    divisor box               (-8.0, 0.0)
    boundary_div([1,1], .)    (-inf, -0.125]
    jax at x = [2.0, 2.0]     inf              <= -0.125 ?  False   <- a DECLARED point
    jax at x = [1.0, 1.0]     -0.1666...       <= -0.125 ?  True
  ```

  At `f0b34cd` the same tree returns UNKNOWN with *"the divisor interval
  [-8.000000000000002, 1.7763568394002505e-15] straddles zero"*.
  The falsifying direction is specific: `[lo, 0]` with a non-negative
  dividend gives a strictly negative upper bound while the runtime zero is
  `+0.0` (`Σx² − c` and `0.0 − x*x` both produce `+0.0`; `-(x*x)`
  produces `-0.0` and does not falsify).

  **THE KERNEL IS NOT THE DEFECT.** `boundary_div` is sound over `b ≠ 0`
  — 7,560 exact-rational quotients here, 0 failures, and 3,291,024 in the
  auditor's independent sweep, also 0. The defect is the PREMISE: the
  transfer inferred "the zero can be dropped" from the SHAPE of the box,
  and the shape does not carry that. This is why a well-measured kernel
  can sit under a false verdict, and why the sweep above now says so in
  its own docstring.

  **The design question, and how it was answered.** Reverting to "decline
  every zero-touching divisor" is sound and withdraws the 0.2.0 row
  entirely — including the case it was built for, `assume(x > 0); 1/Σxᵢ²`,
  where the assume genuinely does exclude the zero and the box cannot say
  so (an interval has no open bound, so the narrowing is the closed
  `[0, hi]`). So the honest condition is not *which endpoint is zero* but
  *whether zero is a value the divisor can take*.

  **What the propagator knew, measured rather than assumed.** Nothing.
  `_classify_cmp` meets the CLOSED half-space and writes it straight into
  `env`; strictness is destroyed at that line and no table, flag or note
  survives it. Transfers receive `(eqn, params, ins)` — boxes only — so
  even a `div` whose divisor IS the assumed variable could not recover it.
  It is *recoverable*, though, because the assume equation is still in the
  jaxpr and the propagator still sees it, which is what the fix does.

  **The fix: a strict-sign certificate, one source, one rule set, one
  consumer.** `_Propagator.strict_sign` maps a var id to `+1`/`-1`, read
  as "every element of this value is certainly > 0 (resp. < 0) at every
  point of the assumed region" — a claim about true reals, never about the
  box.

  * **Source** (`_classify_cmp`): a strict `gt` with every per-element
    bound `>= 0` records `+1`; a strict `lt` with every bound `<= 0`
    records `-1`. `assume(x > -5)` records nothing, which is right: it
    excludes no zero. Later assumes cannot invalidate a record, because
    narrowing is a meet and the assumed region is the conjunction — a fact
    true of the region stays true of every sub-region.
  * **Propagation** (`_strict_sign_out`, a closed by-NAME set):
    `mul`/`div` multiply the signs; `add`/`add_any` keep a shared sign;
    `neg` flips; `abs`/`square` give `+1`; `integer_pow` gives `+1` for
    even `y` (including `y = 0`) and `sign(x)` for odd; `reduce_sum` and
    `dot_general` keep the sign when the operand size is nonzero, which is
    exactly the guarantee that every output cell sums at least one term.
    **`sub` is absent on purpose** — two positives can differ by zero, and
    that is the `Σx² − c` shape above. Every other primitive drops the
    fact.
  * **Consumer**: `_t_div` takes the per-operand signs as a fourth
    argument (the real-mode counterpart of the `in_flags` every ieee
    transfer already receives, registered in
    `_REAL_TRANSFERS_READING_STRICT_SIGN`) and reaches `boundary_div` only
    when the divisor's is nonzero. Otherwise `DIV_BOUNDARY_ZERO_DECLINE`,
    which names the box, says ℝ has no quotient at that point, and lists
    the primitives that carry a certificate.
  * **Not gated on `exactness.certifies_nonemptiness`**, deliberately: the
    claim a certified sign licenses is "the quotient is bounded WHERE THE
    PRECONDITION HOLDS", which is what a VERIFIED under a constraining
    assume already says in its stamp. Whether that region is inhabited is
    the vacuity question, decided and disclosed by separate machinery.
  * **Real mode only.** Under ieee the strict narrowing bumps to the
    format's smallest subnormal and the DAZ haze hulls it straight back to
    a box containing 0 — the runtime value IS zero there — so `x > 0` does
    not imply "certainly nonzero" on a flush-to-zero target. That is S10's
    own lesson; nothing writes or reads the table under ieee.
  * **Scope-swapped like `env`.** Var ids are unique per jaxpr, not per
    transcription, so a transparent call and a `cond` branch each run with
    a fresh table, and nothing is carried in or out. An assume inside a
    possibly-untaken branch therefore licenses nothing outside it.

  **The coverage this costs, measured.** Two shapes lose a decision, and
  both were resting on the dropped point:

  ```
                                                     before      after
  declared b = [0, 1], no assume,   1/b > 0          VERIFIED    UNKNOWN
  declared x = [-1, 0], no assume,  -2/x > 0         VERIFIED    UNKNOWN
  ```

  Both are correct losses: `b = 0` and `x = 0` are DECLARED values, ℝ has
  no quotient there, and jax returns `+inf` / `-inf` — the second is not
  `> 0` at all. Two pre-existing tests asserted those discharges and are
  re-posed with an assume, each keeping its un-assumed form as the
  decline control.

  **What is kept**, which is the point of not simply declining:

  ```
  assume(b > 0); a / b                                       discharged
  assume(x > 0); 1 / jnp.sum(x * x)         > 0              VERIFIED
  assume(x > 0); 1 / jnp.sum(x ** 2)        > 0              VERIFIED
  assume(x > 0); 1 / jnp.sum(jnp.square(x)) > 0              VERIFIED
  assume(x > 0); 1 / jnp.dot(x, x)          > 0              VERIFIED
  ```

  **What is lost beyond the two rows above**, said plainly rather than
  left to be discovered: any divisor built through a primitive with no
  rule, and any built through a subtraction. Measured:
  `assume(x > 0); 1/(Σxᵢ² − 1.0)` is UNKNOWN even though the true divisor
  is nonzero over `x ∈ (0, 4]²` for a large part of the box, and
  `assume(x > 0); y = jnp.sqrt(x); 1/Σyᵢ²` is UNKNOWN because `sqrt` has
  no rule. Extending the set is a rule-per-primitive job and each rule is
  a soundness claim; the granularity is whole-array, not per-element, so
  a mixed-sign array carries nothing.

  **A note on the amendments below, before their content.** They come
  from a later, blinded pass over this branch whose verdict was MERGE,
  and it handed over findings without numeric IDs. They are recorded
  WITHOUT invented ones: the obvious next labels (`B5-2`, `B5-3`) are
  already spent in this same log on different findings — `dot_general`'s
  inlined corners and `boundary_div`'s `inf/inf` raise — so minting them
  here would have put two different things under one name in one file,
  which is the failure mode this log exists to prevent. "B5 follow-up
  audit" is a true label; a number would not have been.

  **AMENDED (2026-08-15, B5 follow-up audit): a CONSTANT operand used to
  drop the certificate too, and the prose said it could not.** Not a
  soundness defect — a COVERAGE defect with two pieces of prose denying
  it, one of them user-facing. `read_strict_sign` answered `0` for every
  literal, on a docstring rationale that reasoned only about a literal
  DIVISOR: *"the single consumer is the `div` boundary gate, which is
  only consulted when the operand's box REACHES zero."* **The single
  consumer was not the only reader.** `_strict_sign_out` calls
  `read_strict_sign` on *every operand of every rule*, so a literal
  COEFFICIENT zeroed the whole chain through it — nothing to do with any
  divisor. Measured before the fix, all four sound and all four declining:

  ```
  assume(x > 0); 1 / (0.5 * jnp.sum(x*x))  > 0               UNKNOWN
  assume(x > 0); 1 / (2.0 * x)             > 0               UNKNOWN
  assume(x > 0); 1 / (x / 2.0)             > 0               UNKNOWN
  assume(x > 0); 1 / jnp.mean(x*x)         > 0               UNKNOWN
  ```

  The last is the mean-squared-residual idiom — the `/n` inside
  `jnp.mean` is the literal — and it is the shape most likely to appear
  wherever this row's headline shape does. Worse, the decline the user
  read was `DIV_BOUNDARY_ZERO_DECLINE`, which told them the chain is
  carried *"on a value the divisor is built from by `* / neg abs square
  x**n sum dot`"*. All four queries are built by exactly `*` and `/`.
  **The message named a rule the code did not implement**, so the remedy
  it recommended did not work — the `docs/norms.md` "claim divergence"
  class, pointed at a user.

  **The fix: a constant answers from its own decoded value**
  (`_box_strict_sign`, beside the rule set it feeds). Five conditions,
  each load-bearing, and each with a test that reddens when that one
  alone is removed:

  * **decodable** — the tree carries dtypes with no zero-dep decoder and
    an undecodable-literal sentinel; both keep answering `0`, via the
    same guarded-decode idiom as `read_flag` and `_quiet_box`. Measured:
    a NaN literal does not even reach the finiteness test, because
    `_value_to_interval` raises `IntervalError("NaN endpoint")` first.
  * **a value, not a range** — the premise `_box_strict_sign` needs is
    that the box IS the value, which is true of a decoded literal and a
    decoded constvar and of NOTHING else here. A var's box is an
    over-approximation, and `assume(x > 0)` narrows to `[0, hi]`, which
    this function correctly calls unsigned; the whole strict-sign table
    exists because a box in general cannot carry this. The pre-boxed
    `IntervalArray` const branch — provenance unknown, which is why it is
    marked maybe-NaN under ieee — is skipped for the same reason, and
    that skip has its own test.
  * **finite, both endpoints** — `±inf` is nonzero but breaks the rules
    that consume the fact: `a / inf = 0`, so a certificate minted off an
    infinite operand would claim NONZERO of a value that is zero. Not
    hypothetical plumbing: `_int_bracket` saturates an int beyond the
    double range to `(maxf, inf)`, and that is a literal.
  * **non-empty** — a size-0 value certifies nothing about "every
    element": both quantifiers are vacuously true over an empty box,
    which would mint a sign for a value that has none.
  * **strictly nonzero, EVERY element** — the fact is quantified over all
    elements, so an array constant must be one-signed throughout. Reading
    only `los[0]` is not merely loose, it is unsound in the divisor
    direction, and the enumeration below is what shows that.

  **The divisor case is not weakened, and this was enumerated rather than
  asserted.** `_t_div`'s gate is consulted only when
  `iv.straddles_zero(divisor)` — some element with `lo <= 0 <= hi`. A
  constant answering `+1` has `lo > 0` for every element (`-1`: `hi < 0`,
  so `lo <= hi < 0`), so it cannot straddle, so no constant divisor whose
  sign is newly nonzero can reach the gate at all. Enumerated over 13
  scalars and all 4³ arrays drawn from `{0, 1, -1, inf}`: 76 decodable
  cases, 1 undecodable, 0 signed-and-straddling. Under the
  first-element-only mutation that enumeration REDS, which is what shows
  the all-elements condition is carrying the soundness and not just tidiness.

  **The certificate was re-checked semantically, not just re-run.** For
  every var the propagator records a sign for, the query is evaluated in
  exact `Fraction` arithmetic at points of the ASSUMED region — the
  half-open `(0, 2]`, sampled including a point `1e-12` from the excluded
  zero and the closed upper endpoint — and every element is confirmed to
  really have that sign. Five shapes, **850 points, 8,438 var-point
  checks, 0 failures**, with a live positive control that adds `sub` to
  the rules and finds **25 violations in 25 cells**. `sub` stays out.
  Those per-shape counts are printed, not just asserted:

      STELLING_FUZZ_REPORT=1 pytest -s tests/test_assume_bump_boundary_div.py \
        -k "TRUE_at_every or catches_a_certificate"

  ```
                                                    before      after
  assume(x > 0); 1 / (0.5 * Σx²)      > 0           UNKNOWN     VERIFIED
  assume(x > 0); 1 / (2.0 * x)        > 0           UNKNOWN     VERIFIED
  assume(x > 0); 1 / (x / 2.0)        > 0           UNKNOWN     VERIFIED
  assume(x > 0); 1 / jnp.mean(x*x)    > 0           UNKNOWN     VERIFIED
  assume(x > 0); 1 / (-2.0 * x)       < 0           UNKNOWN     VERIFIED
  ```

  **And the losses stay lost**, re-measured on the fixed tree — every one
  of them still UNKNOWN:

  ```
  declared b = [0, 1], NO assume,     1/b > 0                  UNKNOWN
  declared x = [-1, 0], NO assume,    1/x < 0                  UNKNOWN
  assume(x > 0); 1 / (Σx² − 8.0)  < 0      (the `sub` break)   UNKNOWN
  assume(x > 0); 1 / jnp.sqrt(x)  > 0      (rule-less)         UNKNOWN
  x / 0.0 > 0                     (a literal ZERO divisor)     UNKNOWN
  assume(x > 0); 1 / (0.0 * x)    > 0      (a zero coeff)      UNKNOWN
  ```

  The four spellings of `assume(x>0); 1/Σx²` above are unchanged and
  still VERIFIED. `DIV_BOUNDARY_ZERO_DECLINE` now says "with nonzero
  finite constants allowed anywhere in that chain", which is the rule the
  code implements.

  **The rule had to cover CONSTVARS, not just literals, or that new
  sentence would have been the next claim divergence.** A scalar constant
  traces to a `Literal`; an ARRAY constant traces to a CONSTVAR, which is
  a `Var` and so reads the assume-written table, where it is absent.
  Measured on the literal-only version of this fix:
  `assume(x > 0); 1/jnp.sum(jnp.array([1.,2.,3.,4.]) * x*x) > 0` was still
  UNKNOWN while the scalar `2.0 * x` had started verifying — a message
  true of scalars and false of arrays, which is the same defect class the
  fix is about. A constvar's decoded box IS its value, exactly as a
  literal's is, so `run()` writes the certificate for it from the same
  `_box_strict_sign`. Measured after:

  ```
  W = [ 1,  2, 3, 4]   assume(x>0); 1/Σ wᵢxᵢ² > 0            VERIFIED
  W = [-1, -2,-3,-4]   assume(x>0); 1/Σ wᵢxᵢ² < 0            VERIFIED
  W = [ 1,  0, 3, 4]   assume(x>0); 1/Σ wᵢxᵢ² > 0            UNKNOWN
  W = [ 1, -2, 3, 4]   assume(x>0); 1/Σ wᵢxᵢ² > 0            UNKNOWN
  W = [ 1, inf,3, 4]   assume(x>0); 1/Σ wᵢxᵢ² > 0            UNKNOWN
  ```

  The mixed-sign row is not conservatism: with `W = [1,-2,3,4]` the sum
  really can be zero over the assumed region, so a certificate there
  would be FALSE. Whole-array quantification is what refuses it.

  **AMENDED (2026-08-15, B5 follow-up audit): the sub-jaxpr disclosure
  named the two rare wrappers and omitted `jit`.** `CHANGELOG.md` said
  *"A transparent wrapper (`remat`, `custom_jvp`) or a `cond` branch runs
  with a fresh table."* `DEFAULT_TRANSPARENT` is
  `{'jit', 'remat2', 'custom_vjp_call', 'custom_jvp_call'}` — **`jit` is
  a member, and it is the one every jax user writes.** The behaviour is
  correct and conservative; the disclosure understated its own cost by
  naming only the members almost nobody hits. Measured, the certificate
  does not cross it in either direction:

  ```
  assume(x>0); 1 / jax.jit(lambda v: jnp.sum(v*v))(x) > 0      UNKNOWN
  assume moved INSIDE the jit, same query                      UNKNOWN
  ```

  Both decline with `div … REACHES zero at a boundary`. The list is now
  pinned to `stelling.coverage.DEFAULT_TRANSPARENT` by a test that reads
  `CHANGELOG.md` and requires every member to be named, so the next
  member added to the frozenset cannot quietly fall out of the prose.

- **2026-08-15 (pre-release, B5 follow-up): the crash class removed from
  `ieee_div` was still live in `boundary_div`, and surfaced as a decline
  reason.** Audit 0.2.0 B5-3. Not a false verdict — a false SENTENCE, out
  of a public entry point.

  The S10 entry above records: *"the boundary-aware branch also raised
  `IntervalError("NaN endpoint")` on `[-inf,-inf] / [-inf, 0]`; returning
  ⊤ before any endpoint arithmetic removes that too."* True of `ieee_div`,
  false of the real-mode sibling, which the same batch left untouched:

  ```
  pre-fix : ieee_div([-inf,-inf], [-inf,0])   RAISED IntervalError: NaN endpoint
            boundary_div([inf,inf], [0,inf])  RAISED IntervalError: NaN endpoint
  B5 HEAD : ieee_div(...)                     -> (-inf, inf)
            boundary_div([inf,inf], [0,inf])  RAISED IntervalError: NaN endpoint
  after   : both                              -> (-inf, inf)
  ```

  `_boundary_div_lo`/`_hi` fall to `_down(num/den)` when either operand is
  infinite, and `inf/inf` is NaN, which `IntervalArray.__post_init__`
  rejects. The dispatcher catches it, so nothing crashed; what a user saw
  was the domain's internal invariant string presented as the reason
  division declined:

  ```
  'div' declined this form at <string>:11 (h): NaN endpoint in interval arithmetic
  ```

  `div`'s own four-corner `inf/inf` guard now runs first in BOTH of
  `boundary_div`'s arms — verbatim, so the two kernels answer the
  indeterminate form identically, and deliberately not refined to
  which-arm-uses-which-corner, because ⊤ is always sound and a narrower
  test is a second thing to keep right. Measured over every legal
  one-sided-boundary call from a 10-value endpoint pool (2,016 box pairs):
  **8 raised before, 0 after.**

  **Suite, both environments, and the skip set.** CI runs plain
  `pytest -q -ra` with no `JAX_ENABLE_X64`, and this branch is measured
  under both:

  ```
                          passed   skipped
  JAX_ENABLE_X64=1          3141       10
  no JAX_ENABLE_X64 (CI)    3142        9
  ```

  The skip SET differs by exactly one member and by design:
  `test_tripwire_arm.py::…threefry…` skips *"the threefry mask fires only
  at x64=0"* when x64 is on. The other nine are identical in both
  (hypothesis ×6, pytest-xdist ×1, blackjax ×2). Baseline on this branch
  before these fixes: 3127/10. `tests/test_ieee_zero_divisor_and_mul_exact.py`
  now carries 204 cases, of which **130 fail against the merge base** —
  measured by copying the file into a `git archive` of `f0b34cd` and
  running it there.

  **Re-measured after the three follow-up amendments above**, same two
  environments, same machine (CPython 3.12.3, jax 0.11.0, numpy 2.5.1,
  Linux x86_64):

  ```
                          passed   skipped
  JAX_ENABLE_X64=1          3176       10
  no JAX_ENABLE_X64 (CI)    3177        9
  ```

  `+35` on both, and they reconcile exactly: `+33` in
  `tests/test_assume_bump_boundary_div.py` (18 → 51 collected) and `+2` in
  `tests/test_ieee_zero_divisor_and_mul_exact.py` (204 → 206). The skip
  SET is unchanged in both environments — nothing added here can skip —
  and the one-member difference between them is still the same `threefry`
  case.

  **Each fix was reverted ALONE and the whole suite re-run**, so the
  coverage is attributed rather than assumed — a test that reddens for two
  fixes tells you less than the count suggests. All six mutations are
  live:

  | reverted alone | tests red | where |
  |---|---|---|
  | B5-1 (the certificate gate removed from `_t_div`) | **4** | the composition test against jax, the two `boundary_div`-reachability controls, and the `assume`-removed sum-of-squares control |
  | B5-2 (`dot_general`'s inlined bumped corners restored) | **4** | three `dot_general` pins in the new file, plus the four-spellings control |
  | B5-3 (`boundary_div`'s `inf/inf` guard removed) | **2** | the kernel sweep and the public-entry-point decline-reason pin |
  | B5-6 (the ieee `div` decline replaced by a silent ⊤) | **4** | one per format |
  | M16 (`_mul_corners`' exact route removed — one function now, so it reverts for `mul` AND `dot_general`) | **19** | 12 in the new file, 3 in `test_contracts.py`, one each in `test_interval.py`, `test_ieee_semantics.py`, `test_square_acceptance_jaxfluids.py`, `test_undecided_detail.py` |
  | S10 (the ieee boundary branch restored in both kernels, with B5-6's gate removed so the branch is reachable) | **117** | 115 in the new file, plus the ieee-f32 assume-bump price pin and the subnormal-haze pin |

  The M16 revert additionally reds
  `test_every_registered_mutation_still_applies_exactly_once`, because the
  mutation deletes text that two registered positive controls target; that
  is an artifact of the mutation, not a control, and is excluded from the
  19. The B5-1 revert additionally reds
  `test_committed_page_matches_live_registries`, because the generated
  primitives page quotes source LINE NUMBERS and the revert shifts them;
  same treatment.

  **Two of those registered controls had stopped applying, and the static
  check could not see it.** `oracle-masked` and `widen` both mutate
  interval multiplication to keep only the two same-corner products. They
  named the `products = (...)` line — the BUMPED route — which after M16
  the int8 `[-1, 1]` query they run never reaches, so the mutants had
  stopped masking anything while `test_suite_disclosure`'s occurrence
  count kept passing. Both now replace the whole body of `_mul_corners`,
  masking both routes; measured under the new mutation,
  `mul([-1,1],[-1,1])` returns `[1, 1]` (so `x*x >= 1` discharges over a
  set containing 0) and `mul([-1,0.5]²)` / `mul([-4,3.5]²)` return
  `[0.25, 1.0]` / `[12.25, 16.0]` against the clean `[-0.5, 1.0]` /
  `[-14.0, 16.0]` (so widening turns UNKNOWN into VERIFIED). The controls
  themselves could not be EXECUTED here — both properties are
  hypothesis-gated and hypothesis is not installed in this environment —
  so what is measured is the mutant's effect on the domain, which is the
  mechanism each control's `why` describes.

  **Also in this batch, and it is the defect that took `main` red twice:**
  `tests/test_ieee_zero_divisor_and_mul_exact.py` called
  `jax.config.update("jax_enable_x64", True)` INLINE in two tests, with no
  restore. x64 is process-global in jax, so an unrestored set leaks into
  every test that runs after it in the session, and it is invisible to
  anyone running with `JAX_ENABLE_X64=1` in the environment — which CI
  does not set. Both now go through a save/restore `_x64` fixture
  (function-scoped and not autouse, because the rest of that module runs
  hand-built IR with no jax at all, so requesting the fixture is also its
  jax gate). The four analogous instances `main` fixed at `942df81` and
  which exist on this branch — `test_pow_audit_findings.py`'s
  module-scope set, and the `importorskip("jax", reason=…)` gates in the
  three property modules — are brought over by cherry-picking that
  commit's hunks rather than by predicting the merge. Its fifth and sixth
  files belong to branches this one does not contain.

- **2026-08-15: FALSE REFUTED — an `assume` inside a `scan` or
  `while_loop` body left no trace, so nothing withheld. PRESENT IN THE
  RELEASED 0.1.0, in real mode, through the ordinary `check()` path.**
  Audit 0.2.0 S13. This is the second finding of that audit to reach a
  shipped version, and the more reachable of the two: unlike the `exp`
  bracket (which needs `semantics="ieee"`, not exposed on `check()` in
  0.1.0), this needs nothing but a solver timeout and an `assume` written
  inside a loop body. It is also the worse direction — a **false REFUTED**
  presents a point the user's own precondition excludes as a
  counterexample to their program.

  **THE MECHANISM.** `propagate`'s walk descends the transparent wrappers
  (`jit`, `custom_jvp_call`, `custom_vjp_call`, `remat2`) and `cond`. It
  does not descend `scan` or `while_loop`. A `stelling_assume` written in
  one of those bodies was therefore never *classified*: it narrowed
  nothing, it was not forwarded to the solver, and — the part that made it
  a soundness defect rather than a precision limit — it left **no record
  that anything had been ignored**. `Propagation.assume_dropped` stayed
  `False` and `Propagation.assume_ledger` stayed empty, so the withholding
  rule that exists precisely to stop a violation over a superset being
  reported as a refutation never fired. The solver searched the
  un-narrowed box and its witness was reported as a counterexample.

  **Measured**, CPython 3.12.3 / jax 0.11.0 / numpy 2.5.1 / z3 5.0.0
  (wheel) / Linux x86_64, `JAX_ENABLE_X64=1`, on a `git archive` export of
  the **`v0.1.0` tag** (`e67688e`) run through `PYTHONPATH` with no
  worktree, and independently on this branch's merge base
  (`06d3bc6`). The fence below is PLAIN and not a python fence
  deliberately: this page carries exactly one ```python fence — the
  integer-wrap reproducer that `tests/test_soundness_wrap_reproducer.py`
  executes — and that test asserts the count, so a second one cannot
  silently become the fence it runs. This reproducer's executable form is
  `s13_scan` in `tests/test_undescended_assume.py`.

  ```
  def h():
      x = any_array((), "float64", (-10., 10.))
      y = any_array((), "float64", (-10., 10.))
      def body(c, _):
          assume(c <= y)        # the carry is x throughout: x <= y
          return c, 0.0
      jax.lax.scan(body, x, jnp.zeros((2,)))
      return assert_(x - y <= 0.0)     # exactly the assumed predicate
  ```

  ```
  v0.1.0 (tag)   scan form   REFUTED  witness x=0, y=-1  (replay-confirmed)
  v0.1.0 (tag)   while form  REFUTED  witness x=0, y=-1
  06d3bc6        scan form   REFUTED  witness x=0, y=-1
  06d3bc6        while form  REFUTED  witness x=0, y=-1
  ```

  `0 <= -1` is false, so the witness is **not an admitted point**. On an
  exact 41×41 `Fraction` grid of `[-10,10]²` the assumed region has 861
  admitted points and the assert is true at **all 861** — no admitted point
  is a counterexample at all. The control, the identical precondition
  written at the top level, VERIFIES on 0.2.0-dev and returns UNKNOWN on
  0.1.0 (which had no relational forwarding).

  **THE SAME MISSING RECORD REACHED THREE RULES.** Two were closed
  separately and one with this entry:

  * the **withholding** rule — the false REFUTED above;
  * the **admitted-region gate** — an empty assumed region stamped an
    entirely clean VERIFIED, including through `check_inductive_step`.
    Closed in audit B3 by requiring `propagate.ledger_covers` (recorded in
    the amended B3 entry above);
  * **`REGION_NOT_ASKED`** — when an obligation's slice carries no
    forwarded relational axiom the region question was skipped outright,
    justified in the source by "the empty case is already the
    propagation's `UnsatisfiableAssumptionError`". That refusal fires when
    a NARROWING empties a box; an assume that never narrowed empties
    nothing, so the ground was untrue. Measured on `06d3bc6`: two assumes
    inside a `scan` body (`x < y`, `y < x`), no relational assume anywhere,
    the obligation discharged by the solver — **VERIFIED, stamped entirely
    clean, with no mention of an assume in the verdict at all**, over a
    region the same exact 41×41 grid shows admits **0** points. Closed
    here.

  **THE FIX IS THE RECORD, NOT THE THREE RULES.**
  `propagate._record_undescended_assumes` runs immediately after the walk
  and before anything reads the run's assume state. It reconciles the
  ledger against the **static** set of `stelling_assume` equations the
  query contains — `_assume_equations`, whose totality is measured against
  an independent walk of the raw jax jaxpr — and writes, for every assume
  the walk never classified, a `dropped` disposition, a note, and a stamped
  `precondition satisfiability uncertified` assumption. `ledger_covers` is
  therefore now a **postcondition of a propagation** rather than a question
  about the walk's reach, and each of the three rules sees the assume
  through machinery it already had. The `REGION_NOT_ASKED` skip gained one
  condition: it stays an absence only when this obligation accounts for
  every assume of the query, or the propagation's own probe already
  exhibited a point of the region.

  **The disposition names the construct**, because "dropped" alone sends a
  reader looking for a classifier that gave up and there was none:
  `NEVER CLASSIFIED: this assume sits inside 'scan', which the
  propagation's walk does not enter, so no classifier ever saw it …`, with
  the enclosing chain outermost-first (`'scan' -> 'jit'`) so that a reader
  can tell which name matches their source line and which one is the cause.

  **WHAT IS NOT FIXED, deliberately.** The propagation still does not
  descend `scan` or `while_loop`. A loop body's `assume` is a
  per-iteration statement about a carry this analysis does not model, and
  reading one is a feature, not a repair. What changed is that ignoring it
  is now disclosed and paid for.

  **WHICH PRIOR VERDICTS ARE RETROACTIVELY INVALID.** Any verdict from
  **0.1.0** or from any 0.2.0 development tree before this commit, in
  **real** mode (`semantics="real"`, the default), on a harness where a
  `stelling.harness.assume(...)` call is executed inside a `jax.lax.scan`
  or `jax.lax.while_loop` body — including inside a `jit`/`custom_jvp`
  wrapper that is itself inside one. Specifically:

  * a **REFUTED** on such a harness may name a witness the precondition
    excludes. It says nothing about the program. This is the invalid
    direction and it needs a solver: `check(..., solver_timeout_ms=N)` or
    `check_inductive_step` with one;
  * a **VERIFIED** on such a harness is sound — the judged set contains the
    admitted region, so every admitted point satisfies the obligation — but
    may be a claim about an **empty** region, and nothing said so;
  * an **UNKNOWN** is unaffected in either direction.

  IEEE-mode verdicts cannot reach the false REFUTED (that mode never
  escalates, so it mints no witnesses), but the missing record was the same
  and its VERIFIEDs carry the same undisclosed vacuity.

  **HOW A READER RECOGNISES ONE.** Search the harness source for `assume(`
  inside a function passed to `lax.scan` or `lax.while_loop`. The verdict
  itself is no help on the affected versions — that is the defect: on
  `06d3bc6` and on `v0.1.0` such a run printed no note, no coverage
  `DROPPED` count and no stamped assumption naming the assume. **On this
  version and later, read the NOTES and the STAMP — not the coverage
  line.** The note begins `assume NEVER CLASSIFIED at <file>:<line>` and
  names the enclosing construct, and the stamp carries `precondition
  satisfiability uncertified`. That is where
  `docs/reading-a-verdict.md` sends a reader, and it is correct.

  **THE COVERAGE LINE IS LEGITIMATELY SILENT, and an earlier wording of
  this paragraph said a fixed run "prints all three".** It does not, and
  audit B9 measured it on the repaired tree:

  ```
    coverage: 9 eqns: 6 known (67%); 1 ⊤ across 1 primitives (scan ×1); 2 unreached
    'DROPPED' in coverage line: False
  ```

  The control — a top-level `jnp.all` assume the classifier drops — does
  print it (`1 constraint(s) DROPPED (stelling_assume ×1)`), so the
  difference is real and it is the right difference. `Coverage.total` is
  `known + transparent + unknown + unreached + inert + constrained`: the
  buckets are a PARTITION of the query's equations, `inert` is the one the
  `DROPPED` count renders, and the scan-body `stelling_assume` is already
  counted in `unreached` — the same line reports it, under the name that
  is true of it. Adding it to `inert` as well would put one equation in
  two cells of a partition and would claim the classifier declined an
  equation it never saw. The code is right; the sentence was wrong. A
  reader who greps the coverage line for `DROPPED` on a fixed tree would
  conclude an affected run is unaffected.

  **WHAT TO RE-RUN.** Re-run `check()` on this version. A previously
  REFUTED harness of this shape will return **UNKNOWN** with
  `violation WITHHELD from REFUTED` in the notes; a previously VERIFIED one
  stays VERIFIED and gains `precondition satisfiability uncertified`. To
  recover a decision, lift the `assume` out of the loop body to the top
  level of the harness, where it is classified and forwarded — the control
  above shows the same mathematics VERIFYING there.

  **THE COVERAGE COST, measured on both sides.** Two corpora, same
  environment as above, `vacuity_mode="inputs-only"`,
  `solver_timeout_ms=5000`, ground truth by exact `Fraction` grids
  computed in the instrument and not by stelling.

  *A 144-harness corpus generated over the loop carriers*
  (`scan` × 48, `while_loop` × 48, top-level control × 48; 2–3
  declarations, satisfiable / chained / unsatisfiable assume sets, two
  asserts, two statement orders, two tail lengths):

  ```
                                   06d3bc6      this commit
    REFUTED                            108               12
    UNKNOWN                              4              100
    VERIFIED                            16               16
    RAISED UnsatisfiableAssumption      16               16
    FALSE REFUTED (witness outside
      the user's precondition)          96                0
    verdicts mentioning no assume       96                0
  ```

  96 rows move, every one of them REFUTED → UNKNOWN, and on THAT corpus
  every one of the 96 was a false REFUTED. All 48 top-level control rows
  are verdict-identical. Of the 96, on a 40-point-per-axis exact grid:
  **32** had an assumed region with **0** admitted points; **32** had an
  inhabited region at **none** of whose admitted points the assert is
  false; and **32** had an inhabited region where *some* admitted point
  does violate — so a correct REFUTED existed for those, with a different
  witness, and it is now withheld.

  **THAT PARTITION IS A PROPERTY OF THAT CORPUS AND NOT OF THIS CHANGE,
  and stating it alone leaves the impression that nothing correct was
  lost, which is false.** Audit B9 reproduced the corpus and the partition
  exactly, then widened it; the widened figure is the one a reader needs.
  The 144-harness corpus draws its assumes from `lt`/`le` only and points
  every assert one way (`expr <= threshold`), and on both of those
  pairings the solver's first model always lands outside the assumed
  region — so the corpus cannot observe the worst category available: a
  refutation that was correct, carrying a witness that was also correct,
  now withheld.

  *A 240-harness corpus built to see it*
  (`scratchpad/s13/sweep_loop_assume_wide.py`, results in
  `scratchpad/s13/RESULTS_loop_wide.txt`; carriers `scan` / `while_loop` /
  `fori_loop` / nested `scan` / `scan`-in-`cond` / top-level control,
  comparison set `lt`/`le`/`gt`/`ge` plus a contradictory pair, four
  asserts in **both** directions, two statement orders; jax 0.11.0,
  z3 5.0.0.0, cvc5 1.3.4, CPU, `JAX_ENABLE_X64=1`). Every moved row is
  scored against the pre-fix run's OWN witness, pointwise, in exact
  `Fraction`:

  ```
    rows                                       240
    REFUTED (06d3bc6) -> UNKNOWN (this branch) 200

      the assumed region admits no grid point           40   20 %
      witness outside it, and no admitted grid point
        violates — nothing correct was lost             40   20 %
      witness outside it, but some admitted point does
        — a correct REFUTED existed, other witness      40   20 %
      witness IN the box, satisfies EVERY assume, and
        violates the assert — A CORRECT REFUTATION
        WITH A CORRECT WITNESS, WITHHELD                80   40 %
  ```

  **40 % of the withheld refutations were correct refutations carrying
  correct witnesses**, spread evenly over all five loop carriers (16
  each), all four satisfiable assume sets and all four asserts (20 each).
  The 40 top-level control rows move **0**. A worked row:
  `assume(x0 >= x1)` inside a `lax.fori_loop` body with
  `assert_(x0 + x1 >= -5.0)` — `06d3bc6` returned REFUTED at
  `x0 = 0, x1 = -6`, which lies in the declared box, satisfies the assume
  and violates the assert; 861 of the grid's 1681 points are admitted and
  240 of those violate; this branch returns UNKNOWN.

  Nothing in this is special to `ge`/`gt`. What decides the category is
  whether the assert's violation region MEETS the assumed region, which is
  a property of the assume/assert **pair**: 12 of the 16 satisfiable cells
  overlap, and in 8 of those 12 the solver's own witness was already a
  correct one — `lt` and `le` among them. So the honest headline is not
  "22 % of the corpus" and not "every one was false". It is that
  withholding is the correct one-sided response to a precondition nothing
  honoured, and that it costs real refutations, at a rate this corpus puts
  at **40 % of the ones withheld** — recoverable only by descending the
  loop.

  *The 288-harness assume-scope corpus* (`jit`/`cond`/`custom_jvp`/top
  carriers — no loop bodies), which is the regression control for
  everything else: **0 of 288 verdicts move**, and **0 of 288 caveat
  states move** (72 VERIFIED, 18 caveated, 54 clean, on both trees).

  **WHAT THE `REGION_NOT_ASKED` TIGHTENING COSTS OUTSIDE A LOOP, which is
  not nothing.** This entry claimed the tightening "costs nothing outside
  the shape it was written for", and the 288-row control was the evidence.
  That control is silent on the question rather than affirmative on it:
  the new `elif` is not gated on loops at all — it fires for **any**
  unaccounted ledger entry on a slice with no forwarded axiom — and the
  288 rows happen to carry none. Audit B9 measured two **non-loop** shapes
  that do move, from clean VERIFIED to caveated VERIFIED (same
  environment; the control, an identical obligation with no assume
  anywhere, stays clean on both trees):

  * an assume written inside one `lax.cond` branch. The walk DOES descend
    `cond`, but a branch-scoped assume narrows no box, so it is a dropped
    conjunct on a slice with nothing forwarded;
  * `assume(jnp.all(x >= 2.0))` on `x ∈ [-1,1]³` — a `reduce_and` the
    classifier cannot narrow, and **no control flow at all**.

  Both new caveats are CORRECT — in the second the assumed region really
  is empty — so this is an under-claim rather than a defect in the code,
  and it is `docs/reading-a-verdict.md` that already names the right
  population: *"an assume that never narrowed anything (a `jnp.all(...)`
  reduction, an unclassified predicate, one written inside a `scan`
  body)"*. The population is **an assume that narrowed nothing**, of which
  a loop-body assume is one member.

  **THE `nonvacuity` FAILED FACE MOVES TOO, and nothing above measured
  it.** `_withhold_uncertified_refutations` runs over two sinks, and the
  second is `p.nonvacuity_checks`: a membership condition judged
  definitely false over the judged set goes from `violated-over-set` to
  `unknown`, with `nonvacuity FAILED face WITHHELD` in the notes. Measured
  on both trees, `nonvacuity(x > 10.0)` on `x ∈ [1,2]` with an
  `assume` in a `scan` body — `06d3bc6`: `violated-over-set`, no note;
  this branch: `unknown`, note present; and the same condition with no
  assume anywhere keeps `violated-over-set` on both. It is correct and
  one-sided for the same reason the REFUTED withholding is (the FAILED
  sentence is reserved for judgments an uncertified constraint has not
  confounded), and it was undisclosed.

  **PROPAGATION COST on a loop-assume query: 4–21×.** Recording the assume
  sets `assume_dropped`, and `assume_dropped` is one of the three inputs
  to `exactness.certifies_set_refutation` at `_region_witness`'s early
  return — so a query that used to answer "nothing is withheld on this
  run, pay for no certificate" now runs the probe search. Same query, both
  trees, `jnp.sum(x) < jnp.sum(y)` in a `scan` body over two
  `n`-element declarations (jax 0.11.0, CPU, `JAX_ENABLE_X64=1`, mean of
  20 runs at `n ≤ 256` and 5 at `n = 1024`):

  ```
        n     06d3bc6   this branch    ratio
        1     0.14 ms      2.09 ms      15x
       32     0.72 ms     14.95 ms      21x
      256     5.05 ms     52.34 ms      10x
     1024    18.69 ms     78.14 ms     4.2x
  ```

  The ratio falls with `n` because the probe count is capped by declared
  size (`_certificate_probe_count`) while the per-propagation cost keeps
  growing. It is bounded by the two gates already documented above and is
  paid only on queries carrying a dropped assume.

  **WHAT AUDIT B9 FOUND IN THIS ENTRY'S OWN REPAIR, and what it changed.**

  * **The `REGION_NOT_ASKED` branch did not consult `ledger_covers`** — it
    tested `unaccounted` alone, and `unaccounted_assumes` is a FILTER over
    the ledger, so an emptied ledger filtered to an empty result and was
    read as "every assume accounted for". That is the exact failure
    `every_assume_recorded` was added to close, reintroduced one branch
    over — and on the one door whose population is an assume the walk
    never entered, so the "defence in depth" claim was false where it
    mattered most. Measured on the third-door harness, whose region an
    exact 41×41 `Fraction` grid shows admits 0 points: with the real
    ledger, `region_uncertified == (0,)` and two disclosure notes; with
    `assume_ledger=()`, `region_uncertified == ()` and **nothing
    disclosed** — a clean stamp on a claim about nothing. Unreachable from
    `check()` (the reconciliation makes `ledger_covers` a postcondition),
    hence FRAGILE rather than UNSOUND. The condition is now
    `unaccounted or not every_assume_recorded`, which is the exact
    negation of the sibling branch's `not unaccounted and
    every_assume_recorded`: one predicate, spelled once in each direction.
    A FIFTH mechanism sentence names the new case, and it names no
    conjunct on purpose — `ledger_covers` joins on `eqn_id` against THIS
    jaxpr, so a False answer includes a ledger belonging to a *different*
    query, whose entries are not conjuncts of this one. The existing
    empty-ledger test drives `cone_split_cycle`, whose slice DOES forward
    an axiom, so it only ever exercised the other door; it now asserts
    that scope and has a twin on a query that forwards nothing.
  * **The `[MAY BE VACUOUS: …]` clause claimed a forwarded axiom.** It
    opened *"this obligation's script carries forwarded relational
    axiom(s)"* — true of every route to `REGION_UNCERTIFIED` when it was
    written, and false for the route this entry added, which is taken
    precisely because the script carries none. It now states the rule and
    names no mechanism, for the reason `UNCERTIFIED_REGION_ASSUMPTION`
    already gives one line up.
  * **`where` named a line in jax's own source.** `source_info[-1]` is the
    house convention, and jax records the bind stack innermost-first, so
    the outermost frame is the user's line only when the user's function
    called `assume` directly. This entry is the first to make LOOP-BODY
    assumes visible, which is exactly where jax wraps the body in one of
    its own. Measured over ten carriers on jax 0.11.0: `lax.fori_loop`
    gave `jax/_src/lax/control_flow/loops.py:2528` and `lax.map` gave
    `loops.py:2784`; an `assume` inside any helper the harness calls gave
    the helper's caller. `_assume_source` reads the frame that CALLED
    `stelling.harness.assume` instead, which is the `assume(` line by
    construction at any nesting. `scan`, `while_loop`, `cond`, `jit`,
    `scan`-in-`cond` and nested `scan` agree with the old reading on all
    six, and every other `where` in the module keeps the convention.
  * **`coverage.sub_jaxprs` walked `tuple` and not `list`** (hardening; not
    reachable from `trace()` or `from_dict`, which both build tuples). A
    hand-built `ir.JaxprEqn` holding a sub-jaxpr in a Python list was
    invisible to the walk the STATIC assume set is collected with, and the
    failure direction is the wrong one: it shrinks the REQUIREMENT, so the
    subset test passes trivially. Measured before the one-word fix:
    *static ids 0 / ledger 0 / assume_dropped False / covers True* — the
    pre-fix state presented as a satisfied postcondition. Three rules now
    rest on that walk's totality, so it is pinned by a test.

  **AND ALL OF THEM TOGETHER MOVE NO VERDICT**, which is what their
  reachability arguments predict and is measured rather than argued: the
  240-harness corpus run against `7b5ceb3` (this branch before the B9
  fixes) and against the fix commit differs in **0 of 240 verdicts** and
  **0 of 240 witness sets**. The new disjunct is unreachable from
  `check()` because the reconciliation makes `ledger_covers` a
  postcondition of a propagation; the `where` frame and the
  `[MAY BE VACUOUS: …]` clause are strings; the `sub_jaxprs` walk is
  unreachable from `trace()`. What each one changes is what a run SAYS in
  the state the argument does not cover.

  **Suite**, this branch, both environments (CI runs plain `pytest` with
  no `JAX_ENABLE_X64`):

  ```
                          passed   skipped
  JAX_ENABLE_X64=1          3334       10
  no JAX_ENABLE_X64 (CI)    3335        9
  ```

  The skip SET differs by exactly one member and by design
  (`test_tripwire_arm.py`'s threefry case skips when x64 is on).
  `tests/test_undescended_assume.py` is the finding's file (27 cases,
  three of them audit B9's: the unasked door's empty-ledger twin, the
  may-be-vacuous clause's axiom claim, and the `where` frame across ten
  carriers). `tests/test_vacuous_precondition.py` gained the `list`-held
  sub-jaxpr test. Four pre-existing tests changed and each is justified in
  place: three in `tests/test_vacuous_precondition.py` — two measured the
  defect itself (the ledger having ONE row for a two-assume query, and the
  note falling back to the mechanism that cannot name a conjunct) and now
  measure the repaired behaviour, and the empty-ledger test now asserts
  the door it is scoped to. `docs/supported-primitives.md` was regenerated
  because it embeds source line numbers; this batch's edits do not move
  any it cites, so it is byte-identical.

- **2026-08-15 (pre-release, B4 part 1): the format-parametric ieee mode
  was still speaking binary64 in four places.** Audit 0.2.0 M12, M13, M14,
  M15. Two of the four move verdicts; both move them in the
  more-informative direction, and neither can mint a definite answer that
  the old code contradicted.

  **M12 — two of the four catalogued formats could not see a constant.**
  `propagate._STRUCT_FMT` had no `<f2` (float16) and no route for
  bfloat16's `<V2`, so `_decode_array` raised and `_Propagator.read` bound
  ⊤ with a note. `read_flag` correctly returned maybe-NaN, so nothing
  definite could leak — the failure was UNKNOWN, never a wrong verdict —
  but `CHANGELOG.md` advertised "all four catalogued formats" while any
  harness in two of them that mentioned a scalar, including the
  ubiquitous `assert_(y > 0.0)`, topped out. Measured on this branch,
  harness `y = x * 2.0 + 1.0; assert_(y > 0.0)`, four formats × two
  semantics:

  ```
                real          ieee
    float16     UNKNOWN  ->   VERIFIED (both)
    bfloat16    UNKNOWN  ->   VERIFIED (both)
    float32     VERIFIED      VERIFIED   (unchanged)
    float64     VERIFIED      VERIFIED   (unchanged)
  ```

  float16 decodes through `struct`'s `e` code, which IS IEEE binary16, so
  the value is exact and the interval is a point.
  **bfloat16 needed a decision, and the obvious decoder would have been
  worse than the ⊤ it replaced.** Its dtype `.str` is `<V2` — an anonymous
  2-byte VOID, which every 2-byte structured dtype also spells — so the
  byte string does not identify the format, and a decoder keyed on `<V2`
  alone would read an arbitrary record type as a float: a wrong VALUE
  where the old behaviour was only imprecise. The `ir.Aval` beside the
  value carries the dtype's NAME, which does identify it, so
  `_decode_array` takes the name as a disambiguator and decodes `<V2`
  **only** under `"bfloat16"`. Anything else — a missing name, a different
  name — stays ⊤-with-a-note, unchanged.

  **Which verdicts are retroactively invalid: none.** The change only
  moves UNKNOWN to a definite status, and an UNKNOWN claims nothing. A
  recorded UNKNOWN whose notes contain *"literal outside the domain (no
  zero-dep decoder for array dtype '<f2'…)"* is one this fix may now
  decide; re-`check()` it to find out. No released verdict is affected —
  float16/bfloat16 support is a 0.2.0 feature and 0.2.0 has not shipped.

  **M13 — the comparison band was picked alphabetically.**
  `_ieee_cmp_get_min_normal` sorted the equation's float dtypes and took
  `[0]` where `_ieee_get_format` also *checks the operands agree*, and
  `bfloat16 < float16 < float32 < float64`. So a `{bfloat16, float16}`
  comparison was hazed with bfloat16's `2**-126` where the float16 operand
  needs `2**-14` — 112 decades too narrow, and the band is exactly what
  makes a verdict sound for a flushing target. Not reachable through jax
  tracing (jax promotes before it computes); reachable through hand-built
  or deserialized IR, which this module's own comments treat as in scope.

  **The fix is NOT the agreement check the finding proposed, and the
  reason matters.** Declining a mixed comparison would have been
  consistent with the arithmetic face, and it would also have withdrawn
  the `{float32, float64}` mixture that hand-built IR routinely carries
  (a float32 value compared against a binary64 literal) and that the old
  code happened to get right. The general rule is available instead: take
  the **maximum** band over the operands' formats. That is sound for every
  operand, because the haze HULLS with 0 rather than replacing
  (`_elt_haze_fmt`), so a band wider than an operand needs costs precision
  and can never cost soundness. Measured over the whole 4×4 grid in both
  orders, 16 pairs, every one now gets a band no operand outgrows.
  Agreement stays REQUIRED on the arithmetic face, where a mixed equation
  has no result format to round onto; that asymmetry is deliberate and is
  written down at both sites.

  **Which verdicts are retroactively invalid**: any ieee-mode verdict from
  a hand-built or deserialized query containing a float comparison whose
  operand dtypes DISAGREE and whose alphabetically-first member has the
  narrower band — in practice, anything paired with `bfloat16`. A
  jax-traced harness cannot be affected. What to re-run: such a query, if
  one exists; the pre-filter is "does any comparison equation in the
  query have two different float dtypes".

  **M14 — two binary64 sentences stamped on narrow-format verdicts.**
  `IEEE_ENDPOINT_ASSUMPTION` says the endpoints are *"the same float
  results the traced program computes, with NO outward rounding"*, and
  `SUBNORMAL_INDETERMINACY_ASSUMPTION` names the band `2**-1022`. Both are
  false of a float16/bfloat16/float32 run: the endpoints **were**
  outward-rounded onto the target grid — that is the whole of
  `_ieee_round_box` — and the band applied was the format's. The
  `semantics:` line of the same stamp disclosed the parametric mode
  correctly, so the verdict contradicted itself. Both are now built from
  the float formats the query contains, at all three assembly sites
  (`propagate`, `verdict.make_verdict`, `solvers.make_solver_verdict` —
  the last two added the endpoint sentence a second time, independently
  of the propagation's own set). **A binary64-only run stamps the
  identical text it always did**: it is the case those sentences were
  written for, it is by far the common one, and rewording an unchanged
  run's stamp is its own disclosure noise. Disclosure only — no verdict
  moves.

  **M15 — the fallback that would have used the wrong band.**
  `_ieee_arith` dispatched to a format-parametric kernel through
  `_FMT_BINARY_OPS` and, with no row, fell back to the binary64 kernel —
  which hazes with `iv.MIN_NORMAL` (`2**-1022`) — and then called
  `_ieee_round_box`, which **cannot** recover the missing haze: rounding
  outward onto the format grid does not hull with 0. Re-measured here,
  float32 `x + x` at `x = 2**-140`:

  ```
    mapped   (format band 2**-126):  (0.0, 1.4349296274686127e-42)
    fallback (binary64 band):        (1.4349296274686127e-42, 1.4349296274686127e-42)
    what jax float32 computes:       0.0        -> the fallback box EXCLUDES it
  ```

  Dead today — all four registered ieee binary kernels have a row — and
  the hazard was that the fifth would be a silent regression with no
  failing test, because `_FMT_BINARY_OPS` and `IEEE_TRANSFERS` are two
  hand-written lists that must agree. **The fix is the census, not the
  arm.** `_ieee_arith` now tags each transfer with the kernel it closes
  over, and `_assert_ieee_binary_kernels_are_format_parametric()` refuses
  the IMPORT when the two lists disagree in either direction — a missing
  row, or a row for a kernel no registered transfer runs. The runtime
  decline is the second guard, and it names the format rather than
  guessing at a band. No verdict moves: the path was unreachable.

  **Suite, both environments, same machine** (CPython 3.12.3, jax 0.11.0,
  jaxlib 0.11.0, numpy 2.5.1, Linux x86_64, glibc 2.39). Baseline for this
  part is the branch point, `0fc6c13`.

  ```
                          passed   skipped
    baseline, x64=1         3176       10
    baseline, no x64 (CI)   3177        9
  ```

- **2026-08-15 (B4): under `semantics="ieee"`, `exp` and `pow` were
  bracketing the WRONG FUNCTION, and one half of it REACHES THE RELEASED
  0.1.0.** Audit 0.2.0 S9 (float32) and S11 (binary64). FALSE VERIFIED and
  FALSE REFUTED.

  **The defect.** Under `ieee` semantics a verdict is a claim about the
  float value **the program computes**. `iv.exp` brackets CPython's
  `math.exp` — the libm of the machine running the analysis, glibc here —
  with a ±1-binary64-ulp bump, under `EXP_LIBM_ASSUMPTION`; `iv.pow_` does
  the same around `math.pow`. **The program does not run that function.**
  It runs whatever XLA compiled for the device. A bracket of one function
  is not a bracket of another, and the assumption stamped under the
  verdict — *"exp endpoints assume a faithfully-rounded libm exp"* —
  asserts a property of the analysis host's libm while being read as one
  of the target's.

  **Measured here**, on jax 0.11.0 / jaxlib 0.11.0, CPU backend, x86_64
  Linux (glibc 2.39), CPython 3.12.3, numpy 2.5.1, ml_dtypes 0.5.4,
  `JAX_ENABLE_X64=1`, eager and under `jit` (identical results), as
  `|jnp.op(x) − true(x)|` in ulps of the target format under
  `_libm_ulp_at`'s binade convention — binary64 reference for the narrow
  formats, 60-digit `decimal` for binary64. **Every row below was re-run
  from scratch on 2026-08-15 for the B4 amendment.** An EXHAUSTIVE row has
  one right answer and the re-run is what stands. A SAMPLED row does not:
  it is a property of its draw, so the row keeps the LARGER of the two
  draws' maxima — the budget must clear anything either saw — and names
  both. `>` marks the four rows the amendment changed (three in the
  figure, one in the population):

  ```
    op   format     population                                       max ulps
  > exp  float16    EXHAUSTIVE, 37,479 normal-finite results           0.500028
  > exp  bfloat16   EXHAUSTIVE, 34,145 normal-finite results           0.499988
  > exp  float32    EXHAUSTIVE, 2,237,668,967 normal-finite results    5.5112
  > exp  float64    3,000,000 sampled args, seed 20260815              1.6660
    pow  float16    16,000,000 sampled pairs, seed 20260815            0.5001
    pow  bfloat16   16,000,000 sampled pairs, seed 20260815            0.5000
    pow  float32    16,000,000 sampled pairs, draw A (seed unrecorded) 0.5380
    pow  float64     1,045,976 sampled pairs, draw A (seed unrecorded) 0.5059
  ```

  **Four of those rows corrected what this entry previously claimed**, and
  three of the corrections are the B4 finding:

  * `exp float16` is **not** correctly rounded. Exhaustively, 2 of the
    63,487 distinct finite arguments exceed half an ulp —
    `x=0.0226898193359375` at 0.500028136794678751 and
    `x=0.007297515869140625` at 0.500011390558184612. The backend
    evaluates float16 `exp` in float32 and rounds twice; at both the true
    value sits a hair BELOW the float16 midpoint and the float32
    intermediate lands above it. The row said 0.5000 and the profile
    declared 0.5, which `_libm_widen_box` honours by widening **nothing**
    — a declared bound the backend it was measured on violates.
  * `exp bfloat16` is correctly rounded **only over normal finite
    results**, and the qualifier was missing. Over ALL finite arguments
    the maximum is **108.698176 ulps** and 11 exceed 0.5 — every one a
    subnormal result flushed to zero, worst `x=-87.5` (true
    9.982350930569248e-39, backend 0.0). That flush is covered by
    `subnormal_haze_fmt`, not by an accuracy budget, which is exactly
    what the qualifier says.
  * `exp float32`'s population is **2,237,668,967**, not the
    2,237,668,968 recorded before: an inclusive/exclusive fencepost at
    the low edge. The band is exactly the float32 values in
    `[-87.33654022216797, 88.72283172607422]` — the first argument whose
    `exp` is normal, and the last whose `exp` rounds finite. The
    2,239,854,020 figure for the raw interval `[-104, 88.73]` DOES
    reproduce; it is 2,185,053 arguments larger, and those are precisely
    the subnormal-result region the sweep never measured.
  * `exp float64`'s sampled maximum is **1.6660**, above the 1.6470 first
    recorded. Nothing is wrong with either: a sampled row is not a bound,
    and an independent draw of the same size beat it. That is the row's
    own lesson and it is now written into the row. The declared budget
    (2.0) covers both. The earlier row also claimed "a further 1,093,019
    arguments in the far tails reach 1.6319" — not reproduced, not
    re-derivable from the design recorded, and dropped rather than
    repeated.

  **The four `pow` rows were re-run too, and their provenance is now
  recorded rather than assumed.** `pow` never exceeds 1 ulp in any format,
  on 49,000,000 pairs across the two campaigns, so the declared 1.0 is not
  in question — but a SAMPLED maximum is a property of the sample, and
  until this amendment the rows named neither draw:

  ```
    format     draw A (c322cec)        draw B (B4, seed 20260815)   row
    float16    16,000,000    0.5001    16,000,000  ->  0.5001       0.5001
    bfloat16   16,000,000    0.5000    16,000,000  ->  0.5000       0.5000
    float32    16,000,000    0.5380    16,000,000  ->  0.5290       0.5380
    float64     1,045,976    0.5059     1,000,000  ->  0.5056       0.5059
  ```

  Draw B is `numpy.random.default_rng(20260815)` over four regions whose
  distributions are written out in full above `LIBM_MEASURED`, and it was
  **run twice: identical maxima and identical kept counts**
  (12,642,619 / 15,907,789 / 15,907,360 / 999,989 pairs with a normal
  finite result), so the seed reproduces it.

  **Draw A's seed was never recorded and draw A cannot be re-run.** On the
  two rows where the draws disagree it is the LARGER, so it is what the
  row keeps — a budget must clear anything either draw saw — and each of
  those rows now says in its own text that the figure it carries is the
  unreproducible one. That is the honest position, not a comfortable one:
  it is the same failure mode the auditor hit in round 1 when it flagged
  `exp@float64` as stale because its own draw gave a different number, and
  it is why `exp@float64` was rewritten to name its seed and both draws.
  The four `pow` rows now do the same.

  12,542 float32 arguments exceed 1 ulp, and 12,520 of them are inside
  `[88.54634857177734, 88.72283172607422]` — a band holding exactly 23,133
  float32 values, so **54.12% of the arguments there escape**, every one on
  the low side (12,542 low, 0 high). Two rates, and they must not be
  compared across populations: over the WHOLE float32 population the
  escape rate is 12,542 in 2,237,668,967, one in 178,414, and the danger
  is that it is not spread — 99.8% of it sits in a band 0.001% of the
  population wide. In binary64, 10,559 of 3,000,000 sampled arguments
  (0.35%, one in 284) exceed 1 ulp, which is exactly what a ±1-ulp
  bracket cannot hold.

  **THE AUDIT'S OWN SUGGESTED REMEDY IS REFUTED BY ITS OWN
  MEASUREMENT.** It proposed widening to ±2 ulps, "enough to cover any
  faithfully-rounded implementation". At 5.51 measured float32 ulps this
  backend's float32 `exp` is not faithfully rounded at all — it is **five
  and a half times worse than faithful** (5.5112 / 1.0; the 11.02 this
  entry used to quote is the ratio against CORRECTLY ROUNDED, 5.5112 /
  0.5, which contradicts `LibmBudget`'s own definitions) — so no fixed
  widening is sound: the quantity is a property of a compiled function
  stelling cannot see. Nor is one number right across formats. The same
  backend, on the same op, measures 0.500028 ulps in float16 and 5.5112
  in float32: a factor of **eleven between two formats**, both of them
  evaluated in float32 and rounded.

  **The fix: fail closed, open it with a DECLARATION.** Under `ieee`, a
  transfer whose backend accuracy stelling cannot establish declines —
  carrying the measurement that justifies it and a line that **runs as
  written** — and the caller re-enables it by declaring a per-`(op,
  format)` budget through EITHER entry point:

  ```
    check(harness, vacuity_mode="inputs-only", semantics="ieee",
          libm_budget="xla-cpu-2026-08")
    propagate(closed, semantics="ieee", libm_budget="xla-cpu-2026-08")
  ```

  Both of those are pulled out of a live decline and executed by
  `test_every_line_the_decline_prints_RUNS_AS_WRITTEN`. They were not
  runnable when this entry was first written: the decline printed
  `vacuity_mode=...` (`Ellipsis`, which raises) and
  `ulps={('exp','float32'): <ulps>}` (a `SyntaxError`), and it named only
  `check` — while the S11 exposure this whole gate exists to close runs
  through `propagate`. Both halves are fixed, and the test now executes
  what the message prints rather than a line that resembles it.

  `"xla-cpu-2026-08"` is a shipped **named, dated** profile whose numbers
  are the maxima above rounded up (1 / 0.5 / 6 / 2 for `exp`, 1.0
  throughout for `pow`); `stelling.propagate.LibmBudget` declares your own
  and requires a `name` and a `basis`. The budget widens the bracket by
  that many format ulps before the outward round onto the format's grid,
  and the verdict stamps it as **DECLARED, NOT VERIFIED**, saying in those
  words that a budget smaller than the backend's real error mints a
  VERIFIED nothing here can catch. A budget of `0.5` — read as **the
  declaration "correctly rounded"** — widens by **nothing at all**:
  round-to-nearest is monotone and the endpoints are rounded onto the grid
  anyway, so the mechanism cannot punish a good platform. That is
  `interval.sqrt`'s own argument generalised; `sqrt` is a correctly-rounded
  IEEE-754 basic operation, carries no libm demotion, and is untouched.

  **"0.5 ulps" and "correctly rounded" are not the same statement in this
  convention, and the branch is justified by the second.** `_libm_ulp_at`
  reports the spacing of the binade CONTAINING a value, so `ulp(2^k)` is
  the spacing *above* `2^k` while the float *below* it is only `2^(k−p)`
  away — half such an ulp. Read as a raw inequality, `u = 0.5` therefore
  also admits a backend returning `nextdown(2^k)` where the true value is
  exactly `2^k`, which correct rounding does not; `exp(0) = 1.0` reaches
  it, and a sweep at `u = 0.5` flags 26 boxes per narrow format, every one
  at equality and every one anchored on a power of two. The docstring used
  to call the no-op "a theorem rather than a kindness" on the weaker
  reading, which is where it was wrong. The residual is covered for this
  module's callers and ONLY by them: `iv.exp` and `iv.pow_` hand over a
  box already bumped a binary64 ulp outward, so the format-rounded lower
  endpoint sits at least one format step below any grid-point true value —
  exactly where `nextdown(2^k)` is. Pinned by
  `test_a_half_ulp_budget_is_read_as_CORRECT_ROUNDING_not_as_the_inequality`
  in all four formats, which asserts the residual exists AND that the
  transfer covers it. After B4 only `exp@bfloat16` still sits at 0.5;
  `exp@float16` moved to 1.0, because it is not correctly rounded.

  **ONE SPACING SERVES BOTH ENDPOINTS, AND IT IS THE LARGER — the first
  draft of the widening was unsound and this is where it was.** `ulp` is a
  STEP function of the magnitude: it doubles at every binade boundary, so
  `t − u·ulp(t)` is **not** monotone in `t` — it drops by half an ulp each
  time `t` crosses a power of two upward. Widening the lower endpoint by
  *its own* ulp is therefore not enough for a box straddling `2**k` from
  below, where the declaration admits values as low as
  `2**k − u·2**(k−p+1)` while `ulp(lo)` is only `2**(k−p)`, half of what
  is needed. The rule is `U = max(ulp(lo), ulp(hi))` applied to both
  endpoints, which restores `lo_out ≤ t − u·ulp(t)` and
  `hi_out ≥ t + u·ulp(t)` for every `t` in the box. Caught by re-deriving
  the monotonicity claim rather than by a test, and now held by one:
  `test_the_widening_covers_every_binade_boundary_a_box_straddles`
  discharges 372 endpoint obligations across the four formats, and its
  positive control shows the per-endpoint rule fails **31 of 31**
  boundaries. It costs nothing at a point argument — the containment
  sweep's widths are unchanged — and it is invisible to any sweep built
  from point declarations, which is exactly why it needed the derivation.

  **AND IT SURVIVED ON THE HALF-INFINITE ARM — the same class, on the arm
  the fix did not cover.** `U` was taken over the FINITE endpoints only,
  so when `hi = +inf` the infinite endpoint dropped out of the maximum and
  `U` fell back to `ulp(lo)` — while the box still holds every `t ≥ lo`,
  in binades whose spacing is 2×, 4×, … larger. Reachable by anything
  ordinary: `iv.exp` returns `hi = +inf` whenever `math.exp` overflows
  binary64, i.e. any envelope reaching past 709.78. Measured on the
  shipped profile at `x ∈ [0.6931471805599453, 10^6]`, float32:

  ```
    iv.exp box                 : (1.9999999999999998, inf)
    widened + hazed + rounded  : (1.9999991655349731, inf)     <- pre-B4
    the same, after B4         : (1.9999984502792358, inf)
    check(exp(x) >= 1.9999991655349731) -> VERIFIED   pre-B4
                                        -> UNKNOWN    after B4

    X'  = 0.6931472420692444        (next f32 above the low endpoint)
    true exp(X') = 2.000000123018602
    the LEAST-WRONG f32 the pre-B4 box excluded: 1.9999990463256836
       = 4.5160 ulps out — inside the DECLARED 6.0 and inside the 5.5112
         this profile measured exhaustively as XLA's own worst float32 exp
  ```

  So a backend **better than the one the profile was measured on** could
  return a value the box excluded, and stelling said VERIFIED while
  `LibmBudget.render` stamped *"the bracket is widened by exactly that
  much"*. (The finding as filed quoted 5.5160 ulps and called it "less
  wrong than the 5.5112 measured" — 5.5160 is *more* than 5.5112. The
  4.5160 above is the sharp figure and is the one that makes the point.)

  **The rule is `U = 2·max(ulp(finite endpoints))` when either endpoint is
  infinite, and it is exactly enough — under a side condition that is
  reachable.** Write `g(t) = t − u·ulp(t)`. For `t ≥ lo`, either
  `ulp(t) ≤ 2·ulp(lo)` and `g(t) ≥ lo − 2u·ulp(lo)` directly; or
  `ulp(t) > 2·ulp(lo)`, in which case `ulp(t)` — a power of two, and not
  the subnormal floor — is at least `4·ulp(lo)` and `t ≥ 2^(p−1)·ulp(t)`,
  so `g(t) ≥ ulp(t)·(2^(p−1) − u) ≥ 4·ulp(lo)·(2^(p−1) − u)`, while
  `lo < 2^p·ulp(lo)` gives `lo − 2u·ulp(lo) < 2·ulp(lo)·(2^(p−1) − u)`.
  Exactly enough because `g(2^k) = 2^k·(1 − u·2^(1−p))`: the first binade
  boundary above `lo` has `ulp = 2·ulp(lo)` and is where `g` dips lowest,
  and for `u < 2^(p−1)` every later boundary dips less. **The second case
  needs `u ≤ 2^(p−1)`, and past it there is no sound finite endpoint at
  all**: `1 − u·2^(1−p)` turns non-positive and `g(2^k) → −∞`. Past the
  threshold the widened endpoint is `-inf`, clamped by `floor`; the
  `lo = -inf` arm is the mirror image under `t ↦ −t` and saturates to
  `+inf`, latent today only because `exp`/`pow` have range in `[0, ∞)`,
  and closed anyway.

  **THE THRESHOLD IS CROSSED BY ROUNDING UP, NOT BY MEASURING — and the
  first two versions of this paragraph had that inequality backwards.**
  They read *"1024 for float16 and 128 for bfloat16 — **under** the 108.7
  ulps this backend's bfloat16 `exp` reaches on flushed subnormal
  results"*. 128 > 108.7; the threshold is above the measurement, not
  below it. The true statement is sharper, and it is a cap rather than an
  anecdote: a subnormal result flushed to zero has error `t/tiny`, since
  `_libm_ulp_at` floors a subnormal's ulp at `tiny`, and the largest
  representable subnormal is `tiny·(2^(p−1) − 1)` — so **a flush measures
  at most `2^(p−1) − 1` ulps, exactly one ulp BELOW the threshold, in
  every format.** Re-derived here in exact `Fraction` arithmetic:

  ```
    format     p    largest subnormal / tiny        threshold 2**(p-1)
    float16    11                       1023                      1024
    bfloat16    8                        127                       128
    float32    24                    8388607                   8388608
    float64    53           4503599627370495          4503599627370496
  ```

  Over the open subnormal band the bound is `< 2^(p−1)`, a supremum that
  is not attained. So the threshold sits just **above** anything that can
  be observed, and this backend's bfloat16 `exp` flush at 108.698176 is
  under 127 and therefore under 128. What crosses it is a caller ROUNDING
  a measurement up — 108.7 rounds to 128 as readily as to 109, and
  rounding up is this profile's own stated convention — a real and likely
  route, but a declaration rather than an observation. The side condition
  is necessary either way: control C below fails 1,538 obligations without
  it. Now held by
  `test_a_flush_to_zero_can_never_measure_up_to_the_threshold`, which
  computes the table above rather than quoting it, so the direction cannot
  drift back.

  Verifying the doubling on a copy is not enough to adopt it: the side
  condition was NOT part of the rule as proposed, and the sweep below
  shows the proposed rule failing 1,538 obligations without it.

  **THE INSTRUMENT WAS THE OTHER HALF OF THE FINDING.** The two tests that
  should have caught this could not:
  `test_the_widening_covers_every_binade_boundary_a_box_straddles` used
  only FINITE boxes, and `test_widening_leaves_infinite_endpoints_alone`
  used the one half-infinite box where the defect cannot appear and
  asserted only `w.his[0] == math.inf` — never the lower endpoint against
  the contract. `docs/norms.md` § *"an acceptance criterion must check that
  the scope covers the claim"*, failing on this change's own instrument.
  **Nothing existing went red when B4 was fixed, and that is the evidence
  the arm was never pinned.**

  What replaces them enumerates rather than samples. `t ↦ t ∓ u·ulp(t)` is
  affine with slope 1 inside a binade, so an extremum over a box sits at
  an endpoint or at a jump of `ulp`; the sweep's candidate set holds EVERY
  jump the binary64 grid can express (`±2^k` for every `k` the format's
  ulp formula distinguishes, up to `2^1023`, with the float on each side),
  so a box's true infimum and supremum are a min/max over its endpoints
  plus the candidates it contains — a SUFFIX min / PREFIX max for a
  half-infinite box, which is what makes the `±inf` arm checkable at all.
  `test_the_widening_covers_the_whole_extremiser_set_on_every_arm` reduces
  over **1,551,584 finite / 34,844,544 half-infinite / 525,888
  doubly-infinite** extremiser points, over all four formats and eight
  budgets from 0.75 up past `2^(p−1)`. Three positive controls, all of
  them rules that shipped or were drafted:

  ```
    rule                                    finite   half-infinite
    SHIPPED (after B4)                            0             0
    A: max over FINITE endpoints only (pre-B4)    0          1886
    B: per-endpoint (the first draft)          1552          1886
    C: doubled, no u <= 2**(p-1) condition        0          1538
  ```

  A is correct on exactly the arm it was written for and wrong on the
  other; C first fails at `nextafter(2**(p-1))`, the very first budget
  past the threshold.

  **BOTH DIRECTIONS.** A wider bracket makes VERIFIED and REFUTED harder
  alike, so this closes S9's false-REFUTED half in the same change:
  with the threshold moved between the executed value and the box, ieee
  mode used to call the obligation *definitely false* where the float32
  execution makes it true. Verified as its own row
  (`test_s9_the_false_refuted_half_closes_too`).

  **REAL MODE IS UNTOUCHED, and that is checked rather than assumed.**
  There the bracket is about the true real value, CPython's own `math`
  module does satisfy the ±1-ulp assumption it rides on, and the
  divergence from the compiled program is the ℝ-versus-float gap the stamp
  already names. `iv.exp` is byte-identical; a budget passed under
  `semantics="real"` is REFUSED rather than ignored; and the real-mode
  `exp` stamp still carries the unchanged `EXP_LIBM_ASSUMPTION`
  (`test_real_mode_still_judges_exp_with_no_budget_at_all`, four formats).

  **WHICH STELLING VERSIONS ARE AFFECTED.** S11 (binary64) is **0.1.0 code,
  unchanged** and present in the released tag. S9 (float32) is the
  format-parametric mode, 0.2.0 development only. Reproduced here on
  `v0.1.0` itself, extracted with `git archive` and run by `PYTHONPATH`
  (no worktree, repo untouched):

  ```
    v0.1.0 __version__ = 0.1.0
    v0.1.0 iv.exp([X,X]) = (4.244390682998849e-95, 4.2443906829988504e-95)
    jnp.exp(X)           =  4.244390682998851e-95     (one ulp ABOVE the box)
    inside bracket?       False
    v0.1.0 propagate(cj, semantics="ieee") -> ['discharged']
  ```

  at `X = -217.29998556254742`, an entirely ordinary argument. The
  obligation was `assert_(jnp.exp(x) <= 4.2443906829988504e-95)`, the
  verdict discharged it, and executing the same program in jax falsifies
  it.

  **WHICH PRIOR VERDICTS ARE RETROACTIVELY INVALID, AND HOW A READER
  RECOGNISES ONE.** A verdict is at risk when **all** of:

  * its stamp's `semantics:` line says **ieee** (`ieee (IEEE-754 binary64)`
    or the format-parametric wording) — a `real (ℝ)` verdict is not
    affected in any version; and
  * its `assumes:` lines contain **`exp endpoints assume a
    faithfully-rounded libm exp`** or the `pow` counterpart, which is
    stamped exactly when the query used that transfer; and
  * the status is **definite** — VERIFIED, or a set-level REFUTED that the
    interval leg decided downstream of the `exp`/`pow`. An UNKNOWN never
    claimed anything.

  In **0.1.0** that means: a verdict produced through
  `propagate(closed, semantics="ieee")` on a query containing `exp` or
  `pow`, in binary64 (the only format 0.1.0's ieee mode accepted).
  **`check()` in 0.1.0 had no `semantics` keyword** — verified at the tag,
  `def check(harness, *, vacuity_mode, solver_timeout_ms=None, refine=None,
  strict=False)` — so no verdict from the documented front door can be
  affected; the exposure is exactly the `propagate` entry point.

  **And the released CHANGELOG never named that entry point.** This entry
  used to say it "advertised" `propagate`; re-checked at the tag, the word
  `propagate` appears **zero** times in `v0.1.0:CHANGELOG.md`. Its only
  related line is

  ```
    - IEEE-semantics mode (opt-in): judges censused binary64 behaviours and
      stamps itself separately from real-mode verdicts.
  ```

  — a feature announced with **no route named**. That is a more
  uncomfortable fact than the one this entry asserted, not a lesser one:
  a reader who wanted the advertised mode had to find the route in the API
  surface, so the population of affected verdicts is bounded by who did
  that and not by who read the release notes. It is also why the B4
  decline now names `propagate` explicitly.

  In **0.2.0 development** the same applies in all four formats, and
  float32 is the dangerous one — not because its overall escape rate is
  high but because it is **concentrated**. Both rates, over their own
  populations: 12,542 of 2,237,668,967 float32 arguments exceed 1 ulp
  (one in 178,414, exhaustive), and 12,520 of those sit in a band of
  23,133 values where the rate is **54.12%**. In binary64, 10,559 of
  3,000,000 sampled arguments exceed 1 ulp (0.35%, one in 284, sampled).
  A previous version of this line compared float32's in-band 54% against
  an underived "one-in-130,000" over a different population; that figure
  reproduced from nothing in this campaign and has been replaced by the
  two rates above, each with the population it was measured over.

  **WHAT TO RE-RUN.** Any recorded ieee-mode verdict whose query contains
  `exp` or `pow`. Re-run it on a tree with this fix. With no
  `libm_budget` it will now be UNKNOWN with the decline quoted; with a
  budget declared it is judged against a bracket widened by that
  declaration. A verdict that stays VERIFIED under the shipped profile was
  never resting on the missing ulps. The cheap pre-filter that needs no
  re-run: **if the query contains no `exp` and no `pow`, or if the stamp
  says `real`, it is unaffected.**

  **The cost, measured.** Bracket width at a point argument, in
  representable steps of the target format, before and after:

  ```
    format     arg      declared ulps   before   after
    float16    3.0            0.5            1       1
    bfloat16   3.0            0.5            1       1
    float32    3.0            6              1      13
    float32   88.7            6              1      13
    float64    3.0            2              2       8
    float64 -217.3            2              2       8
  ```

  float16 and bfloat16 cost **nothing**, which is the 0.5-ulp branch doing
  its job. End to end, ordinary obligations survive — `exp(x) > 0` over
  `[-2, 2]` is VERIFIED in every format, and `exp(x) <= e·(1+1e-7)` over
  `[0, 1]` in binary64 — while obligations tight to the last ulp
  (`exp(1) <= fl(e)`) move VERIFIED → UNKNOWN, which is the class the
  finding was about.

  **The sweep behind the profile is in the tree**, not only in this file:
  `tests/test_libm_budget.py::test_the_declared_budget_brackets_what_the_backend_computes`
  drives 4,003 containment checks across the four formats — including 300
  arguments inside the measured float32 escape band and S11's own
  argument — and asserts that exact count against
  `LIBM_EXP_SWEEP_CHECKS`, the way `BOUNDARY_DIV_SWEEP_QUOTIENTS` is
  asserted. A sibling row shows the sweep BITES: with the widening removed
  the same arguments escape.

  **Suite, both environments, same machine** (CPython 3.12.3, jax 0.11.0,
  jaxlib 0.11.0, numpy 2.5.1, Linux x86_64, glibc 2.39):

  ```
                          passed   skipped
    branch point 0fc6c13    3176       10      (x64=1)
    branch point 0fc6c13    3177        9      (no x64, as CI runs)
    after M12-M15           3206       10      (x64=1)
    after the budget        3277       10      (x64=1)
    after the budget        3278        9      (no x64, as CI runs)
    after the B4 amendment  3293       10      (x64=1)
    after the B4 amendment  3294        9      (no x64, as CI runs)
    after the B4 re-audit   3295       10      (x64=1)
    after the B4 re-audit   3296        9      (no x64, as CI runs)
  ```

  They reconcile exactly: `+30` in `tests/test_ieee_narrow_formats.py` for
  M12–M15, then `+70` in `tests/test_libm_budget.py` and `+1` in
  `tests/test_doc_examples.py` (whose executed-block inventory went 29 to
  30 with the new documented example), then `+14` in
  `tests/test_libm_budget.py` and `+2` in
  `tests/test_ieee_narrow_formats.py` for B4, then `+2` in
  `tests/test_libm_budget.py` for the re-audit's two cosmetic items — the
  flush-cap table and the sampled-row provenance, one test each. The skip
  SET is unchanged in
  both environments and the one-member difference between them is still
  `test_tripwire_arm.py`'s `threefry` case, which skips *"the threefry
  mask fires only at x64=0"* when x64 is on; the other nine (hypothesis
  ×6, pytest-xdist ×1, blackjax ×2) are identical in both.

  **Each part was reverted ALONE and the whole suite re-run**, so the
  coverage is attributed rather than assumed:

  | reverted alone | tests red |
  |---|---|
  | the budget gate removed from `exp` AND `pow` (the pre-fix bracket) | **23** |
  | the gate kept but the WIDENING made a no-op | **9** |
  | the stamp's *declared, not verified* line dropped | **4** |
  | the widening using each endpoint's OWN ulp (the binade bug) | **1** |

  and for the B4 amendment, each reverted alone against
  `test_libm_budget.py` + `test_ieee_narrow_formats.py` +
  `test_ieee_semantics.py`:

  | reverted alone | tests red |
  |---|---|
  | the whole half-infinite rule (back to max over finite endpoints) | **3** |
  | ...just the `spacing *= 2` doubling | **2** |
  | ...just the `u <= 2**(p-1)` side condition | **2** |
  | `exp@float16` back to a declared 0.5 with the correctly-rounded row | **3** |
  | the bfloat16 row losing its *normal and finite* qualifier | **1** |
  | the decline back to a template (`vacuity_mode=...`, `<ulps>`, `check` only) | **2** |
  | the `ulps <= 0.5` branch narrowed to `< 0.5` (the Fix-3 restatement's subject) | **7** |
  | the module-level `_assert_ieee_binary_kernels_are_format_parametric()` call deleted | **1** |
  | the module-level `_assert_libm_transfers_take_a_budget()` call deleted | **1** |

  and, for the two cosmetic items the re-audit returned:

  | reverted alone | tests red |
  |---|---|
  | `_libm_ulp_at` stops flooring a subnormal's ulp at `tiny` (the flush cap's premise) | **4** |
  | a `pow` row drops the seed of the draw that CAN be re-run | **1** |
  | a `pow` row stops disclosing that its carried figure's seed was NEVER recorded | **1** |

  The two deleted-census-call rows were **0** before this amendment — that
  is the whole of finding 6. The half-infinite rows are the whole of
  finding 1, and the measure of how badly the instrument was aimed:
  reverting the *entire* B4 widening fix reds nothing that existed before
  it.
  `test_supported_primitives_doc.py::test_committed_page_matches_live_registries`
  reds on every mutation that shifts a source line and is excluded from
  every count above, the same treatment the B5 entry gives it.

  Every mutation additionally reds
  `test_supported_primitives_doc.py::test_committed_page_matches_live_registries`,
  because the generated primitives page quotes source LINE NUMBERS and
  every mutation shifts them; that is an artifact of mutating, not a
  control, and is excluded from each count — the same treatment the B5
  entry above gives it. The attribution runs were driven with a reduced
  environment, whose own clean baseline is 3,269 passed / 18 skipped
  rather than 3,277 / 10; the FAILED lists are what is read here, and the
  totals reconcile against that baseline in every row.

  Three pre-existing tests changed status and each was repaired rather
  than relaxed: `test_ieee_semantics.py::test_pow_and_exp_keep_libm_brackets_when_nan_free`
  and `::test_pow_declines_flagged_operands_with_the_gap`, and
  `test_three_rows.py::test_exp_stops_the_taint_and_a_later_add_stays_definite`
  all drove ieee `exp`/`pow` with no budget. Each now declares one — so the
  property each was written for (the bracket, the maybe-NaN gap, the
  contraction taint) is still exercised — and the first additionally pins
  that the undeclared call declines.

- **2026-08-15 (B6): FALSE VERIFIED — a `dot_general` whose operands'
  contracted extents disagree had its addends silently DROPPED by the SMT
  emission, on an equation the interval transfer refuses outright. PRESENT
  IN THE RELEASED 0.1.0, through `ir.ClosedJaxpr.from_dict`.** Audit 0.2.0
  S12. The third finding of that audit measured at the tag, and the
  narrowest of the three in reach: it needs a query that was
  **deserialized** (or hand-built), because jax will not trace the
  equation — `jax.lax.dot_general` on `(2,)` against `(4,)` raises
  `TypeError: dot_general requires contracting dimensions to have the same
  shape, got (2,) and (4,)`. Nothing produced by `stelling.harness.trace`
  can carry one.

  **THE MECHANISM, and it is a shape this project keeps being bitten by.**
  `stelling.interval.dot_general` checked contracted- and batch-dimension
  agreement and RAISED. `obligation._dot_general_plan` re-derived the same
  geometry independently, iterating the **LHS** contraction extents and
  never cross-checking the RHS. So the two faces held different opinions
  about whether the equation was well-formed — and the disagreement
  resolved in the **asserting** direction, because the transfer's refusal
  is precisely what routes the obligation to the solver: a refused transfer
  binds ⊤, ⊤ leaves the obligation `unknown`, and `unknown` is what
  escalation runs on. The face that refuses hands the work to the face that
  truncates.

  Measured directly on the plan, three shapes, on a `git archive` export of
  the **`v0.1.0` tag** and identically on this branch's base `dee8bc2`,
  with a four-element all-ones constant operand:

  ```
  lhs=(2,) rhs=(4,) -> (0, [[(Fraction(1,1), 0), (Fraction(1,1), 1)]])
                       two of the constant's FOUR elements dropped, no decline
  lhs=(4,) rhs=(2,) -> IndexError: tuple index out of range
  lhs=(3,) rhs=(3,) -> correct
  ```

  and end to end, CPython 3.12.3 / jax 0.11.0 / z3 + cvc5 wheels / Linux
  x86_64, `JAX_ENABLE_X64=1`, on the **`v0.1.0` tag** and on `dee8bc2`
  alike — a traced query whose serialized *declaration* is edited to shape
  `(2,)` while the constant operand stays `(4,)`, reloaded through
  `from_dict`:

  ```
  from_dict ACCEPTED the mismatched query
  propagation obligations: [(0, 'unknown', '... the operand spans [-inf, inf] ...')]
  ESCALATION assert #0 -> discharged
     detail: discharged by solver escalation (QF_LRA): the box with the
             negated predicate is unsat per z3 (wheel) and cvc5 (wheel)
  interval transfer on the same equation:
     IntervalError dot_general contracted dims disagree: lhs[0]=2 vs rhs[0]=4
  ```

  **Why that VERIFIED is false, in arithmetic that involves no verifier.**
  Four declared elements in `[1, 2]`, constant operand `[1,1,1,1]`,
  threshold `9/2`. The true four-term sum ranges over `[4, 8]`, and
  `8 <= 9/2` is false — the claim does not hold. The truncated two-term sum
  ranges over `[2, 4]`, and `4 <= 9/2` is true, which is the VERIFIED the
  solver honestly returned. The solver was not wrong; it was asked a
  smaller question than the query states.

  **WHICH PRIOR VERDICTS ARE RETROACTIVELY INVALID, and how to recognise
  one.** A verdict is in scope only if its query was **not produced by
  `trace()`** — it came through `ir.ClosedJaxpr.from_dict`, a hand-built
  `ClosedJaxpr`, or a direct `smt.emit` on hand-built IR. Beyond that,
  **do not screen this entry's condition on its own** — read the entry
  below it (**S12′**, same date) and use the screen stated there — noting
  that it is the BINDING witness alone, and reading its blindness list.

  > **THE TWO SCREENS THIS ENTRY ORIGINALLY GAVE WERE UNSOUND, AND ARE
  > STRUCK RATHER THAN EDITED** (audit 0.2.0 B6). They said, in
  > substance, (1) *"for every `dot_general`, check `lhs.aval.shape[i] ==
  > rhs.aval.shape[j]` over the paired dims; if they all agree, the
  > verdict is untouched by this defect"*, and (2) *"`unknown_primitives`
  > contains `('dot_general', 1)` with a `no sound rule` note, while a
  > downstream obligation is `discharged by solver escalation`"*. Both are
  > **false all-clears on an affected query**, and the reason is S12′: the
  > lie can sit on the two operand avals *consistently*, so they agree
  > with each other and disagree with the arrays that actually flow.
  > Measured on `4d793cf`, on a query the first screen clears and the
  > second one clears:
  >
  > ```
  > SCREEN 1  lhs.aval=(2,) rhs.aval=(2,) -> agree? True   -> "untouched"
  > SCREEN 2  unknown_primitives -> ()    (no "no sound rule" note)
  > ACTUAL    STATUS: VERIFIED   coverage: 4 eqns: 4 known (100%)
  > TRUTH     the four-term sum lies in [4, 8] and the bound is 4.5
  > ```
  >
  > Screen 2 is false for a structural reason worth stating: it was
  > derived from *this* defect's mechanism, in which the transfer REFUSES
  > and therefore leaves a ⊤ footprint in the coverage record. Under S12′
  > the transfer does not refuse — it succeeds, on the true shapes — so
  > there is no footprint to find. A recognition screen read off the
  > mechanism is only ever as wide as the mechanism.
  >
  > A verdict that this entry's original screens cleared may still be
  > affected. Screen with S12′'s instead; it subsumes this entry's
  > condition, because an equation whose operand avals disagree with each
  > other also has at least one aval disagreeing with its binding.

  **WHAT TO RE-RUN.** Any VERIFIED **or REFUTED** whose query was
  deserialized or hand-built and that contains a `dot_general` — see
  S12′ below for why REFUTED is in scope too, and for what an affected
  query returns on a fixed tree. The instruction this entry used to give
  ("an affected query now returns UNKNOWN with `escalation declined:
  'dot_general' declined: dot_general contracted dims disagree: …`") is
  true only of the forms whose avals disagree *with each other*; on an
  S12′-shaped query it returned VERIFIED on the fixed tree of the day,
  which is why it is withdrawn here.

  **THE FIX IS A SHARED ORACLE, AND THE PLACE WAS THE QUESTION.** Three
  candidates were live: the door (`ir._validate_loaded`), the truncating
  face (`_dot_general_plan`), or a helper both faces consult.

  * **Not the door.** `ir.py`'s own module docstring puts *"full
    per-primitive shape inference (validating every equation's output aval
    against its inputs)"* **explicitly out of scope** for that validation
    pass, and names *"the in-pipeline read gate and shape/decode
    predicates"* as the defence past its bounded set. A `dot_general` rule
    added there would be one primitive's shape inference wearing a
    validator's clothes, and it would leave the two faces still able to
    disagree on any query built another way — `ClosedJaxpr` is a public
    dataclass and `smt.emit` takes hand-built IR. That is making the symptom
    rarer, not making the defect unconstructable. `from_dict` still accepts
    the document, and `tests/test_dot_general_from_dict_door.py` pins that
    deliberately, so a later reader cannot "fix" this at the door and
    conclude the oracle is redundant.
  * **Not a second copy of the predicate in `_dot_general_plan`** either. A
    check in one place the other does not consult is the SHAPE of this
    defect, not its repair. Two copies that agree today are the arrangement
    that produced it.
  * **The shared oracle.** `stelling.interval.dot_general_geometry` is now
    the single definition of a well-formed `dot_general` — dim ranges,
    duplicate dims, list pairing, batch- and contracted-extent agreement —
    and of the geometry that follows from it, including the contraction
    ranges the coefficient loop walks. `interval.dot_general` obtains all of
    it from there and `_dot_general_plan` calls the same function and quotes
    its `IntervalError` as a decline. This is the discipline
    `obligation._route_structural` already followed for the structural rows
    — *the interval function IS the routing* — extended to the one computing
    row left out of it, and complementary to
    `propagate._dot_general_row_form`, which owns this row's PARAMS and
    DTYPES and never sees a shape.

  **WHAT THE EXTRACTION CHANGED ABOUT `interval.dot_general`, MEASURED.**
  This entry claimed the transfer side was *"a pure extraction: no
  predicate changed"*. That is **not true**, and the honest sentence is
  the list (audit 0.2.0 B6). Measured `dee8bc2` against `4d793cf`, calling
  the public `interval.dot_general` on `(3,) · (3,)` with each
  `dimension_numbers` below:

  ```
                            dee8bc2                                  4d793cf
  0                         TypeError: cannot unpack non-iterable     IntervalError
  (0, 0)                    TypeError: cannot unpack non-iterable     IntervalError
  (((0,),(0,)),((),()),…)   ValueError: too many values to unpack     IntervalError
  None                      TypeError: cannot unpack non-iterable     IntervalError
  "xy"                      ValueError: not enough values to unpack   IntervalError
  ```

  Five malformation shapes moved from a raw `TypeError`/`ValueError` to
  the decline channel, because the extraction wrapped the
  `dimension_numbers` unpack in a `try`.

  **AND THAT CENSUS IS INCOMPLETE, NOT INFLATED — audit 0.2.0 B6
  RE-AUDIT.** Every row above is real; the census stopped at one source
  line when the `try` it describes spans two. Re-derived over a 40-case
  malformed-`dimension_numbers` corpus, same call, four trees, attributing
  each raw raise to the `interval.py` line it came from on `dee8bc2`:

  ```
  dee8bc2 line 1454  `(lc, rc), (lb, rb) = dimension_numbers`
      11 cases, ALL moved at 4d793cf     <- the family counted above (5 of these)
  dee8bc2 line 1455  `lc, rc, lb, rb = tuple(lc), tuple(rc), ...`
       7 cases, ALL moved at 4d793cf     <- A SIXTH FAMILY, in no test and
                                            in no doc until now: the outer
                                            form unpacks 2x2 and an INNER
                                            element is not iterable
  dee8bc2 line 1462  `0 <= d < len(arr.shape)`
       4 cases, moved at 96ab47a (S12'''s operator.index guard)
  dee8bc2 line 1473  `a.shape[i]`
       2 cases, moved at 96ab47a
  dee8bc2 line 1457  `len(set(dims))`
       3 cases, moved in THIS commit (re-audit R4: the dims are now BOUND
       to what operator.index returns, not merely checked)

  27 of 40 corpus cases raised raw on dee8bc2; 0 do on this tree.
  ```

  **AND THAT ZERO IS CORPUS-RELATIVE — audit 0.2.0 B6 AUDIT 3, F2.** The
  40 cases enumerate malformed *shapes* of `dimension_numbers`. Not one of
  them carries a dimension whose `__index__` refuses with something other
  than `TypeError`, or whose `__repr__` refuses at all — and both of those
  still left `interval.dot_general` **raw** on `d6b6d0b`, because the
  guard R4 rewrote caught `TypeError` alone and then quoted the offending
  object with an unguarded `{d!r}`. A count over a corpus is a claim about
  the corpus, and this one was read as a claim about the function. The
  corpus is therefore extended by a family it did not contain, and the
  three trees re-measured through the same public entry point (`interval.
  dot_general`, which exists on all three; `dot_general_geometry` was only
  extracted at `4d793cf`), jax 0.11.0, `JAX_ENABLE_X64=1`:

  ```
  34-case corpus = the five families above (27) + family F (7):
      an extent whose __index__ raises ValueError / OverflowError /
      RuntimeError (5 placements) and whose __repr__ raises (2)

                    refused as IntervalError   accepted   RAISED RAW
  dee8bc2                    3                    0          31
  d6b6d0b                   25                    3           6   <- all F
  this tree                 31                    3           0
  ```

  The three ACCEPTED are well-formed after normalisation and are supposed
  to be accepted (two 0-d `numpy` dims, and one extent whose `__index__`
  answers 0 while its `__repr__` refuses — nothing is wrong with that
  document, and on `d6b6d0b` the message composer crashed on it anyway,
  which is the F3 finding one row over). The guard now catches whatever
  `__index__` raises and quotes the dimension through `interval.
  _safe_repr`.

  So the batch's true figure across the whole arc is **27 malformation
  shapes moved across five source lines**, of which 18 moved in the
  extraction commit the paragraph above is about, and **6 more moved in
  the audit-3 fixes**. Every one is in the
  **safe** direction and none is reachable through `_t_dot_general`, whose
  `dimension_numbers` has been through `_dot_general_row_form` and is a
  well-formed 2×2 by then — but the function is public, and "no predicate
  changed" is a claim about the function. The extraction also added
  `check_shape(lhs_shape)` / `check_shape(rhs_shape)`; those are **exactly
  neutral** for every caller of `interval.dot_general`, because
  `IntervalArray.__post_init__` has already run the identical check on any
  operand that could be passed. They are live only inside
  `dot_general_geometry`, which did not exist before and therefore changed
  nothing. Pinned in
  `test_dot_general_both_faces.py::test_the_extraction_DID_change_these_predicates_and_here_they_are`
  and `::test_the_two_new_check_shape_calls_are_unreachable_for_an_IntervalArray`.

  `tests/test_dot_general_both_faces.py` asserts AGREEMENT over the
  well-formed and malformed halves rather than asserting declines — a test
  that only demanded declines would be satisfied by the copy — and spies on
  the oracle to pin that each face actually calls it. The malformed half
  grew four non-integer-dim rows in the same batch; see the B6/S12″ entry.

  **A SECOND DEFECT, IN THE SAME FINDING: a raw `IndexError` out of
  `slice_obligation`.** With the *constant* operand the shorter one, the
  plan indexed off its end. `slice_obligation` is documented *"Never raises
  on legal queries"* and its caller catches only `_Decline`; worse,
  `solvers.escalate` iterates `slice_unknown_obligations` in the `for`
  **header**, outside its own per-obligation `except Exception` net, so the
  exception escaped the entire call and one ill-typed equation took every
  other obligation's verdict with it. The extent defect is fixed at its
  root, so nothing currently constructable reaches it; the call site is
  netted nonetheless, in the same posture and the same words `escalate`
  already uses around `_dispatch_obligation` — degrade to UNKNOWN, QUOTED,
  with the exception class and message in the obligation's `detail` and the
  words *internal error* in it, so a stelling defect reads as a stelling
  defect rather than as an undecided obligation. The range test in
  `slice_obligation` also became two-sided: an index past the START of the
  assert list used to raise `IndexError` (reaching the whole-query bar
  through `verdict._bar_scope`'s outer `except`) and now declines with the
  same sentence an out-of-range positive index gets. That changes one pinned
  behaviour, in
  `test_verified_bar.py::test_what_a_stray_index_ACTUALLY_DOES_all_four_of_them`,
  which is updated rather than relaxed: all four stray-index behaviours now
  end at the decline channel.

  **NO VERDICT FLIPS ON ANY WELL-FORMED QUERY.** The oracle refuses exactly
  the forms `interval.dot_general` already refused, so a query the transfer
  accepts plans exactly as it did.

- **2026-08-15 (B6): FALSE VERIFIED *and* FALSE REFUTED — the SMT emission
  summed a DIFFERENT ARRAY than the interval propagation did, on an
  equation NEITHER leg refused. `dot_general` and `reduce_sum` alike.
  PRESENT IN THE RELEASED 0.1.0, through `ir.ClosedJaxpr.from_dict`.**
  Audit 0.2.0 **S12′**, found by a blinded adversarial re-audit of the S12
  fix directly above. Read this entry and not that one when screening a
  verdict: its condition subsumes S12's, and S12's two recognition screens
  are struck as unsound (see the note there).

  **THE MECHANISM, and it is S12's own repair seen from one step back.**
  S12 gave the row a shared shape oracle,
  `interval.dot_general_geometry`, and concluded *"the two faces cannot
  hold different opinions about whether an equation is admissible"*.
  **The oracle is shared; its ARGUMENTS are not.**

  ```
  interval.dot_general      -> dot_general_geometry(a.shape,  b.shape)
                                                    ^^^^^^^^^^^^^^^^^  the
                                                    PROPAGATED BOXES
  obligation._dot_general_plan -> dot_general_geometry(_shape_of(eqn.invars[0]),
                                                       _shape_of(eqn.invars[1]))
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^  the
                                                    RECORDED AVALS
  ```

  One function, two inputs. Leave the declaration and the constant operand
  alone and edit only the equation's **invar avals** — which `from_dict`
  accepts, `ir.py` having put per-primitive shape inference out of
  `_validate_loaded`'s scope in writing — and the two faces are back to
  disagreeing, still in the asserting direction.

  **WORSE THAN S12'S OWN PRESENTATION, and this is the part that matters
  for recognition.** Under S12 the transfer REFUSED, which left a ⊤
  footprint in the coverage record — the signature that entry's screen was
  built on. Under S12′ the transfer does not refuse. It succeeds, on the
  true shapes, agrees the contraction has four terms, and prints the box.
  Measured end to end on `4d793cf`, CPython 3.12.3 / jax 0.11.0 / jaxlib
  0.11.0 / z3 5.0.0 (wheel) / cvc5 1.3.4 (wheel) / Linux x86_64,
  `JAX_ENABLE_X64=1`:

  ```
  propagation: [(0, 'unknown')]
     "undecided for 1/1 element(s); the operand spans [4.0, 8.0] and the
      asserted bound is operand <= 4.5; the operand's upper endpoint
      misses the bound by 3.5"
  ESCALATION 0 -> discharged
  STATUS: VERIFIED         coverage: 4 eqns: 4 known (100%)
  ```

  The interval leg said the sum is between 4 and 8, in those words, and
  the verdict said the claim `Σ <= 4.5` is VERIFIED.

  **THE TRUTH, in arithmetic that involves no verifier and in jax.** Four
  declared elements in `[1, 2]`, constant operand `[1,1,1,1]`, threshold
  `9/2`. Exact `Fraction`: the four-term sum ranges over `[4, 8]` and
  `8 <= 9/2` is **False**; the truncated two-term sum ranges over `[2, 4]`
  and `4 <= 9/2` is True. Concrete jax at the top corner:
  `jnp.dot(jnp.array([2.,2.,2.,2.]), jnp.array([1.,1.,1.,1.])) == 8.0` and
  `jnp.sum(jnp.array([2.,2.,2.,2.])) == 8.0`.

  **THE FALSE REFUTED, AND THE SENTENCE IT FALSIFIES.** The same lie also
  produces a REFUTED on a claim that is true everywhere, with a witness
  the verdict describes as *"confirmed by independent exact-rational
  replay (fractions.Fraction arithmetic, pure Python, no solver)"*. The
  harness is `s - t >= -1/2` where `s = jnp.dot(a, [1,1,1,1])` and
  `t = a[0]+a[1]+a[2]+a[3]` — the same four addends spelled two ways, so
  `s - t` is identically `0` and `0 >= -1/2`. Interval arithmetic loses
  the correlation (`[4,8] - [4,8] = [-4,4]`), so the obligation is
  undecided and escalates; truncated, `s` drops two addends and `s - t`
  becomes `-(a₂+a₃) ∈ [-4,-2]`. Measured on `4d793cf`:

  ```
  ESCALATION 0 -> violated-witness
     witness values: x0_0=1  x0_1=1  x0_2=1  x0_3=1
     replay: confirmed by independent exact-rational replay ...
  STATUS: REFUTED
  ```

  At `(1,1,1,1)` the predicate is **true**: `s - t = 0 ≥ -1/2`. The replay
  sentence is honest about its arithmetic and false about its **plan** —
  replay re-derives the same `_dot_general_plan` / `_group_reduce_sum` the
  emission used, so a witness is independent of the SOLVER and never of
  the plan. That distinction is now stated where the claim is made.

  **IT IS A CLASS, NOT A ROW.** `reduce_sum` reads `_shape_of(eqn.
  invars[0])` in `_group_reduce_sum` and truncates identically; all four
  combinations (two rows × false-VERIFIED / false-REFUTED) were measured on
  `4d793cf`, on `dee8bc2`, and on a `git archive` export of the **`v0.1.0`
  tag**, with identical outcomes at all three. Two further shapes reach it
  from INSIDE a `jax.jit` body, where no interval environment holds a box
  for the lying operand at all — `VERIFIED` on all three trees.

  **THE FIX IS ONE CROSS-CHECK FOR EVERY PRIMITIVE AT ONCE, in the
  slicer.** `_Slicer._one_shape_per_value`, run from `_validate` over every
  equation of the slice before any plan is built: **no equation may be
  modelled at a shape that disagrees with the shape the value actually
  has.** Not a third `dot_general` shape rule — that would leave
  `reduce_sum`, `scatter`, and every emission row not yet written exactly
  where they were.

  "Actually has" gets **two witnesses, complementary because each is blind
  where the other sees**, and both are needed:

  1. **The binding site.** A variable is bound once and referred to many
     times; every reference must agree with the binding. Needs no
     propagation, so it reaches EVERY scope the descent flattened —
     including the `jit`-nested pair above, whose operand ids this slicer
     minted and no environment has ever seen. Blind to a lie applied
     CONSISTENTLY at the binding and at every reference.

     **WHAT "THE BINDING" IS, and it took a second pass to get right —
     audit 0.2.0 B6 RE-AUDIT, UNSOUND-1.** It read the producing equation's
     outvar aval, or the constvar's aval. That is the record of the binding
     for every producer BUT ONE. A `stelling_any` describes itself twice,
     and `slice` mints one SMT constant per element of its `shape`
     **param** — never per element of that outvar's aval. So for exactly
     that binding class the check compared a quantity nothing emits: a
     declaration saying four elements in its param and two in its aval
     minted four symbols, summed the two the reference asked for, and came
     back `discharged` on a claim whose truth is `8 <= 4.5`. Measured on
     `96ab47a`, inside a `jit` body where witness 2 is blind by
     construction and where no reference disagreed with any other.
     The three sites in the emission path that need a declaration's
     element count — the element budget, `_binding_shape`, and the
     input-term construction — all call
     `stelling.obligation._Slicer._declared_shape`, so **none can
     implement a different rule from the others**. A declaration whose
     `shape` param cannot be read **declines**.

     **THAT IS NOT "A SINGLE READ", AND IT IS NOT SOLE READERSHIP —
     audit 0.2.0 B6 audit 3, F4.** The method's own docstring claimed both
     and neither is true, so both are struck there and neither is asserted
     here. Each call **re-reads** the param, and an object that answers
     differently between calls does make the check and the emission
     differ. A `list` SUBCLASS whose `__iter__` yields `(4,)` for three
     reads and `()` after — both faces accept a `list` — was checked at
     `(4,)` and minted ONE input for a four-element reference, `slice`
     alone reading it three times; swept over the flip point, identical
     on `d6b6d0b` and on this tree. What contains that is not
     `_declared_shape` but `ir.ClosedJaxpr.content_hash()`, which cannot
     encode a param that answers differently between iterations: it
     RAISES, `solvers._query_sha256` swallows that to `""`, and the
     pairing gate refuses an empty hash. And
     `stelling.propagate._declared_element_count` is a genuine SECOND
     reader of a declaration's element count that reads the outvar
     **aval**; it is sound because it gates only whether the
     certificate search runs, in the direction of REFUTED, and that search
     re-derives its witness by re-running the honest propagator. Naming
     the containment where it actually is matters: "cannot drift apart"
     tells the next reader to stop looking, which is the mistake this
     entry is otherwise about.
  2. **The propagated box.** The interval leg computed a shape for this
     value from the values flowing in rather than from what the IR says
     about them, so it is the one witness a consistent lie cannot forge.
     This is the inter-leg agreement S12 claimed and did not have. Blind
     OUTSIDE the top level, by construction: `interval_env` returns the
     top-level environment and the propagator runs each transparent call
     body in an isolated env it discards on the way out.

     **THIS ONE IS LOAD-BEARING, AND THE PARAGRAPH THAT SAID OTHERWISE IS
     STRUCK** — audit 0.2.0 B6 RE-AUDIT, UNSOUND-2. It used to read
     "defence in depth", supported by an enumeration of every route to a
     consistently-applied lie and by the claim that **"no IR document has
     been constructed on which it is the only thing that sees the lie"**.
     The re-audit constructed one: the declaration lie of item 1 at TOP
     level, where witness 1 read the aval and therefore agreed with every
     reference, and the propagated box — which the interval leg builds from
     the `shape` param — was the only thing that disagreed. Proved by
     deleting the box leg on `96ab47a` and watching the same query come
     back `discharged`. The enumeration was wrong in its first clause: a
     `stelling_any` whose `shape` param contradicts its outvar aval was NOT
     refused at `JaxprEqn` construction, because that check ran only
     `if isinstance(shape, tuple)`.

     That is **the same failure mode twice in one batch — asserting
     completeness for an enumeration nobody drove — and it is what S12
     itself did.** So no enumeration replaces it. What replaces it is three
     facts a reader can check: *the box leg is the only witness a
     consistently-applied lie cannot forge; it is load-bearing wherever a
     box exists; and it is blind inside transparent call bodies — which is
     why the binding witness must be total in its own right.* Both legs are
     load-bearing, in disjoint places, and neither is insurance for the
     other. It is exercised through `env`, a caller-supplied argument of
     the public `slice_obligation`.

  A `Var` the slicer cannot bind at all **declines** rather than passing. A
  check that goes quiet where it cannot see is the shape of the defect it
  is here to close, and that arm is a backstop rather than a live path —
  measured below, it fired zero times.

  **THE DOOR WAS CLOSED TOO, AND IT IS NOT THE FIX.** `ir.
  _validate_decl_eqn` compares a declaration's two self-descriptions in
  BOTH the containers a declaration is recorded in — a `tuple` and a
  `list`, which is the whole of `ir._SHAPE_PARAM_CONTAINERS` — and REFUSES
  a `shape` param of any other container type rather than skipping it;
  `ir._validate_param_value` recurses into lists as well as tuples. A
  validator that silently passes a param class it cannot read grants that
  class a pass, which is the same defect shape one layer down.
  *(This paragraph read "REFUSES a `shape` param that is not a sequence of
  extents at all" until audit 0.2.0 B6 audit 4, F1. That was true of
  `d6b6d0b`; from `30d4b04` the check is on the param's CONTAINER TYPE,
  under which `range`, `array.array`, `memoryview`, a numpy array and
  every custom iterable are sequences of extents and are all refused. The
  narrowing is confined to hand-built IR — every other route normalises to
  a `tuple` before the primitive is bound — and the partition is now
  measured against the rule over a computed population of container types
  in `tests/test_shape_param_rule.py`, so this sentence cannot go stale
  again without a test going red.)*

  **AND COMPARING IS NOT ENOUGH: THE DOOR NOW INSTALLS WHAT IT COMPARED**
  (audit 0.2.0 B6 audit 5, F1). Reading the param ONCE and binding it
  locally made the door's own comparison honest and left every reader
  after it re-reading the raw object — the interval transfer in
  `propagate`, `_Slicer._declared_shape`, `ir._encode` and
  `coverage.sub_jaxprs`. A `tuple` SUBCLASS whose `__iter__` yields `(2,)`
  for the door and `(1,)` for everyone after it was therefore ACCEPTED at
  two elements, propagated as one, and returned **`discharged`** for
  `sum(x) <= 3.9` over `x` in `[1,2]²`, whose exact maximum is 4 — with no
  `object.__setattr__` anywhere and with a stable `content_hash()`. `main`
  refuses that document only by accident: the door there read the param
  twice and the second read caught the lie. The read-once repair removed
  the accident and nothing replaced it. `JaxprEqn.__post_init__` now
  writes the validated extents back into `params`, and
  `Aval.__post_init__` / `Array.__post_init__` do the same for their own
  `shape`, so every later reader sees a plain `tuple` of plain `int`.
  A SHARED READER would not have done this — it makes every read use one
  PROTOCOL, not one VALUE, and neither `ir._encode` (generic over tuples,
  it cannot know which one is a shape) nor `coverage.sub_jaxprs` (which
  never asks what a param means) could have been routed through one. The
  same document now reaches **REFUTED** with a two-element witness the
  exact-rational replay confirms.

  **AND INSTALLING THROUGH A COMPARISON IS NOT ENOUGH EITHER: THE DOOR
  NOW STORES EVERY VALUE AS AN EXACT BUILT-IN, OR REFUSES IT** (audit
  0.2.0 B6 audit 6). The install above was written
  `(k, dims) if k == "shape" else (k, v)`, and `k` is document-supplied
  too: a `str` SUBCLASS answering that comparison True for
  `_validate_decl_eqn`'s two reads and **False** for the install's own
  third read let the door validate the param, report `dims`, and rewrite
  `params` with the lying object still in it — after which every later
  reader found the key again and read the lie. Same query, same oracle,
  same four read sites, same **`discharged`**, again with no
  `object.__setattr__` anywhere. Two more members of the same class were
  measured beside it: the duplicate-key refusal asked `hash` (through
  `set`) and `eq` (through `list.count`) of the same keys, so two `str`
  subclasses with equal text and different `__hash__` were **not** a
  duplicate and a document carrying both `("update_jaxpr", None)` and
  `("update_jaxpr", <the add jaxpr>)` was accepted with `params_dict()`
  picking one by hash placement — the exact `scatter-add`
  replace-vs-accumulate hazard that refusal exists to close; and the
  `dtype` param was compared with `==` at the door and consumed with
  `str()` by `propagate._ieee_any`, which selects the subnormal band from
  it.

  Five members in four rounds, each repair correct and none of them
  closing anything, is the evidence that the class is not a list. The
  repair is therefore **`ir`'s canonicalization door**: at construction,
  every document-supplied value in every `stelling.ir` dataclass is
  replaced by an EXACT instance of a type the module is closed over — a
  subclass read ONCE through its base type's own accessor (`str.__str__`,
  `int.__index__`, `float.__float__`, `bytes.__getitem__`,
  `tuple.__getitem__`, none of which an override can redirect), a `list`
  stored as a `tuple`, and a type with no exact form to store REFUSED
  naming its type. A second read cannot then differ, because there is no
  subclass left to answer it. That is a property of the stored object
  rather than of any reader, so it covers the params that have no rule at
  all (`axes`, `new_sizes`, `slice_sizes`, `dimension_numbers`) and
  readers nobody has written yet. **It does not make those params
  CORRECT** — per-primitive shape inference is still scoped out in
  writing — only single-valued, so the transfer and the emission read the
  same extents. The fields the module already had a stronger rule for are
  left to it: aval and array extents go through `_load_extents`, which
  reads any object with a working `__index__` once and installs a plain
  `int`, and a declaration's `shape` param is judged by the container rule
  above before the generic door sees it. One value is CARRIED rather than
  canonicalized, by explicit declaration from the module that builds it:
  the pre-boxed `interval.IntervalArray` a `ClosedJaxpr.consts` entry may
  hold in place of a value (recorded above, with its own test). No
  document route reaches it — `ir._decode` has no tag for it and
  `ir._encode` refuses to encode one — so it is a caller's object, and
  `interval.py` states in return that it is frozen and validated at
  construction.

  But the
  door is **not** where this class is contained, and two things say so.
  `ir.py` scopes per-primitive shape inference out of the load validation
  in writing, and `ir.ClosedJaxpr` is a public dataclass. And the door
  still BLESSES, deliberately, a declaration with **no `shape` param at
  all** — hand-built IR legitimately omits params — which on `96ab47a`
  sliced a four-element declaration into ONE scalar symbol and then raised
  `IndexError: tuple index out of range` out of `smt.emit`, reaching the
  verdict as *"escalation attempted; internal error"*. The slicer closes
  that form on its own, and `tests/test_aval_lie_both_faces.py::
  test_the_slicer_closes_the_declaration_lie_ON_ITS_OWN` installs the
  param/aval disagreement past `__post_init__` with `object.__setattr__`
  so that the slicer is measured with the door not standing in front of
  it.

  Driving that door found one more thing, **reported and not fixed here**:
  `propagate._t_stelling_any` calls `tuple(params["shape"])` with no guard,
  so a `shape` param that will not iterate raises `TypeError: 'object'
  object is not iterable` out of the public `propagate()` while the
  emission face declines — the S12″ two-faces shape once more, on the
  transfer side. It is left because it is `stelling.propagate` and the
  other face, and because the constructible route to it is now shut: only
  an `object.__setattr__` past the frozen dataclass reaches it. Pinned as
  a live expectation in
  `test_aval_lie_both_faces.py::test_the_transfer_face_still_raises_raw_on_
  an_uniterable_shape_param`, so the report cannot rot into folklore.

  **AND THE DISCLOSURE UNDERSTATED ITSELF — audit 0.2.0 B6 audit 3, Q6.**
  "The transfer face raises where the emission declines" reads like a
  difference in wording; it is a difference in what a caller can catch.
  Measured on this tree, same document:

  ```
  except interval.IntervalError   -> NOT caught (escapes as TypeError)
  except ir.TranscriptionError    -> NOT caught (escapes as TypeError)
  the escaping type               -> bare TypeError
                                     (MRO: TypeError, Exception, BaseException)
  ```

  Those two are the library's malformed-IR exception classes, and neither
  covers this. `ir.TranscriptionError` *subclasses* `TypeError`, so the
  relationship is the wrong way round: only a bare `except TypeError` — a
  handler no caller should be asked to write, since it also swallows their
  own bugs — catches it. So a caller who handles malformed IR exactly as
  the library documents gets a crash, not a decline. That is what makes
  this a residue worth carrying in writing rather than a wording nit.
  *(The audit reported this as "not catchable as `IntervalError` or
  `PropagationError`". There is no `PropagationError` in this tree; the
  two classes above are the ones that exist, and the finding holds against
  both.)*

  **DOES THE DECLARATION SHAPE REACH `v0.1.0`? YES.** Driven against the
  tag (`3e9bb9d`, `stelling 0.1.0`; jax 0.11.0, x64, z3 and cvc5 wheels):
  both the `jit`-nested and the top-level forms return `escalate(...) ->
  discharged` — *"the box with the negated predicate is unsat per z3
  (wheel) and cvc5 (wheel)"* — on `sum(a) <= 4.5` over `a ∈ [1,2]^4`,
  whose true supremum is 8. On `96ab47a` the `jit`-nested form still
  discharged and the top-level one was caught by the box leg alone. The
  RENDERED verdict is blocked on both trees, and **only accidentally**: a
  `list` shape param makes the query un-encodable, `ClosedJaxpr.
  content_hash()` raises, `solvers._query_sha256` swallows that to `""`,
  and the pairing gate refuses an empty hash. Nothing that saw the lie was
  involved, and the raw `TypeError` that containment rests on is itself
  disclosed below.

  **WHICH PRIOR VERDICTS ARE RETROACTIVELY INVALID, and the screen — which
  is WITNESS 1 ALONE and is not a clean bill of health.** (It was published
  as *"the ONE screen"*. It is one screen in the sense that it replaces the
  several this entry struck; it is not a screen for everything, and what it
  cannot see is enumerated under *WHAT THIS SCREEN IS BLIND TO* below —
  audit 0.2.0 B6 audit 3, F7.) A
  verdict is in scope only if its query was **not produced by `trace()`** —
  it came through `ir.ClosedJaxpr.from_dict`, a hand-built `ClosedJaxpr`,
  or a direct `smt.emit` on hand-built IR. jax will not trace any of these
  forms; nothing `stelling.harness.trace` produces can carry one.

  For such a query, screen the **IR**, not the verdict:

  > **1. Walk every scope.** The top-level jaxpr, and **every sub-jaxpr
  > carried in any equation's params** — including ones nested inside a
  > tuple or list param, such as a `cond`'s `branches`. Not only the four
  > transparent primitives.
  >
  > **2. Collect the bindings of each scope.** A variable is bound at:
  > - a `stelling_any` equation's **`shape` param** when that equation
  >   produces it — **the param, not that equation's outvar aval**, and an
  >   ABSENT `shape` param binds at `()`. This is the quantity the emission
  >   mints terms from;
  > - any other producing equation's outvar aval;
  > - a constvar's own aval;
  > - a jaxpr **invar's** own aval.
  >
  > **3. Resolve a reference in its own scope first, then outward through
  > the enclosing scopes** — innermost enclosing first. Var ids are not
  > globally unique and `from_dict` accepts a document that reuses one
  > across scopes.
  >
  > **4. Then**: for every equation, for every operand that is an `ir.Var`,
  > compare the shape on **that reference** with the shape it is bound at.
  > **If every reference agrees with its binding, the verdict is untouched
  > by this defect.**
  >
  > **A BINDING THAT CANNOT BE READ COUNTS AS A DISAGREEMENT, NOT AS A
  > PASS.** Two cases: an operand no scope binds at all, and a
  > `stelling_any` whose `shape` param is not a `tuple` or a `list` of
  > nonnegative integers. Treat both as **AFFECTED**. This is not
  > conservatism for its own sake — it is what the code does, and a screen
  > may be stricter than the code but may never be looser:
  > `_Slicer._one_shape_per_value` DECLINES an operand it cannot bind and
  > `_Slicer._declared_shape` DECLINES a param it cannot read, so a
  > document a reader clears here is one the tool itself refuses to
  > verify.

  **EVERY ONE OF THOSE FOUR CLAUSES WAS ADDED BECAUSE THE SCREEN WITHOUT
  IT CLEARED AN AFFECTED DOCUMENT**, and the first one cleared a document
  this very entry was written about. A remediation instruction that says
  ALL CLEAR on an actually-false VERIFIED is worse than no instruction, so
  each is recorded with what it lets through:

  - **The `stelling_any` param (clause 2).** The screen used to name the
    producing outvar's aval for every producer. On the declaration lie of
    witness 1 above — the document this entry's own reproducers are built
    from — every reference agrees with that aval and the screen returned
    ALL CLEAR on a query that returned `discharged` on `8 <= 4.5`.
  - **Scope (clause 3).** The screen was silent about scope. With a var id
    reused across scopes — `from_dict` accepts it — a global first-wins
    lookup lands on an unrelated TOP-LEVEL binding whose shape agrees with
    the lie, and returns UNTOUCHED on a document that was `STATUS:
    VERIFIED` on `4d793cf`. The code was right; the screen was the
    failure, which is the same failure the screens it replaced were struck
    for.
  - **Invars (clause 2, fourth bullet).** The screen named two binding
    sources and a jaxpr's invars is neither, so a reference to a function
    parameter had nothing to compare against. Both readings driven over the
    same population in one run — **3,534 queries, and both readings see
    all 3,534 with zero screen errors**, so they really are the same
    population (this bullet used to say 3,531 against the MEASURED
    paragraph's 3,533; one run gives one number for both, and audit 3
    re-derived it): the SHIPPED screen leaves **583 references
    unresolvable across 192 queries**, the repaired one leaves **2, in 2**.
    (Removing only this bullet from the repaired screen leaves **1,378
    across 532** — the 583 is the shipped screen as a whole, which is the
    comparison this bullet is making.) Not exploitable under the shipped
    reading — the slicer's `_rewrite` substitutes the caller's atom before
    anything reads an invar aval — but a screen that quietly cannot answer
    for 583 references is not a screen whose ALL CLEAR a reader can act
    on, and under the fail-closed convention clause 4 now states it is not
    quiet at all: those references become AFFECTED, and the bullet moves
    **526 verdicts**.
  - **Descent (clause 1).** It descended only `jit`, `custom_jvp_call`,
    `custom_vjp_call` and `remat2`, so a lie inside a `lax.cond` branch
    read UNTOUCHED. Not exploitable either — `cond` declines — but
    descending everything adds no false positives (measured below) and
    costs nothing, and "the primitives I happened to enumerate" is the
    same reasoning that produced the other three defects on this list.

  **MEASURED, on the tree this fix ships in** (jax 0.11.0, x64, one full
  `pytest -q -p no:randomly` with the screen driven on every query handed
  to `propagate()` and its clauses independently switchable): **3,534
  distinct queries** out of 3,650 `propagate()` calls — 4,257 scopes,
  29,338 equations, 31,003 `ir.Var` operand references, **0 screen
  errors**, 0 duplicate binders within one scope, **2 unresolvable
  references**. Driven against the affected documents directly, it flags
  all six known forms (the list-param declaration nested and at top level,
  the absent-param declaration, this entry's original
  `reduce_sum`/`dot_general` invar-aval reproducer, the scope-reuse
  document and the `cond`-branch document) and clears all six of their
  unedited controls.

  **AND CLAUSE 4'S SILENCE WAS WORTH THREE DIFFERENT ANSWERS ON THAT ONE
  POPULATION — audit 0.2.0 B6 audit 3, F6, and the reason clause 4 now
  states its convention.**

  ```
  convention                                            UNTOUCHED  AFFECTED
  STRICT LITERAL   only a READABLE binding that DISAGREES    3519        15
  AS FIRST PUBLISHED  + an unreadable declaration param      3518        16
  FAIL-CLOSED      + an operand with no binding — THE CODE   3516        18
  ```

  The published 16 was the middle reading, and it is the one that cannot
  be defended: it counts an unreadable `shape` param as AFFECTED and an
  UNBINDABLE OPERAND as untouched, which is not a rule so much as the
  absence of one. The code has a rule and it is fail-closed —
  `_one_shape_per_value` declines an operand it cannot bind,
  `_declared_shape` declines a param it cannot read — so on 2 of 3,534
  queries **the published screen cleared what the tool itself refuses**.
  The three queries the conventions part on:

  ```
  test_aval_lie_both_faces.py::..._transfer_face_still_raises_raw_...
      reduce_sum operand 0, var 1 at (4,), bound at an UNREADABLE shape
      param       STRICT UNTOUCHED | PUBLISHED AFFECTED | FAILCLOSED AFFECTED
  test_propagate.py::test_unbound_var_raises_instead_of_widening
      exp operand 0, var 7, NO BINDING
                  STRICT UNTOUCHED | PUBLISHED UNTOUCHED | FAILCLOSED AFFECTED
  test_audit_findings.py::test_cross_branch_read_raises_instead_of_...
      add operand 0, var 31, NO BINDING, in cond.branches[1]
                  STRICT UNTOUCHED | PUBLISHED UNTOUCHED | FAILCLOSED AFFECTED
  ```

  All 16 of the published-convention AFFECTED are in
  `tests/test_aval_lie_both_faces.py`, the file that builds affected
  documents on purpose — **zero false positives**. Under the fail-closed
  convention the other two are the two fixtures that deliberately hold an
  unbound variable, so the screen's answer there is "I cannot say", said
  out loud, which is what a screen is for.

  **WHICH CLAUSES ACTUALLY FIRE ON REAL WORK, and it is one of them —
  audit 0.2.0 B6 audit 3, F7.** Each clause switched off in turn, over the
  same 3,534 queries, counting queries whose verdict moves:

  ```
  clause switched off                     STRICT  PUBLISHED  FAIL-CLOSED
  1  descend every sub-jaxpr                   0          0            1
  2  bullet 1: a stelling_any binds at
     its shape PARAM                           5          6            6
  2  bullet 4: a jaxpr's invars bind           0          0          526
  3  resolve per scope, innermost first        0          0            1
  the SHIPPED (96ab47a) screen: 1, 2b1,
     2b4 and 3 all off at once                 5          6          194
   ... the same four off, but clause 1
       ablated to "any primitive, top-
       level jaxpr params only"                5          6          409
   ... 96ab47a's screen RE-IMPLEMENTED
       from its own published text             5          6          194
  clause 4 EXTENDED to equation outvars        0          0            0
  clause 4 EXTENDED to a jaxpr's outvars       0          0            0
  ```

  **THE 194 AND THE 409 ARE BOTH RIGHT, AND WHAT SEPARATES THEM IS WHAT
  "CLAUSE 1 OFF" MEANS — audit 0.2.0 B6 audit 4, F5.** An audit's re-run
  reported 409 where this table said 194, with STRICT and PUBLISHED and
  every individual clause row agreeing exactly. Re-driven with both
  readings in one run, on one population, the disagreement is not in the
  measurement: switching clause 1 off can mean *descend the four
  transparent primitives* (`jit`, `custom_jvp_call`, `custom_vjp_call`,
  `remat2` — what `96ab47a` actually did) or *descend any param whose own
  value is a jaxpr, for any primitive*. The second descends MORE scopes,
  so more sub-jaxpr invar references exist to be unresolvable, and
  fail-closed turns each into an AFFECTED: **583 unresolvable in 192
  queries** against **1,063 in 407**. The row means the first, which is
  what "the SHIPPED (96ab47a) screen" says — and the check on that is the
  last row: a re-implementation of `96ab47a`'s screen straight from its own
  published sentence, sharing no code with the ablation, lands on the same
  **194** and the same **583 in 192**.

  **METHOD, so both figures can be re-derived rather than believed.** One
  `pytest -q -p no:randomly` over the whole suite with `propagate()`
  wrapped, holding a hard reference to every query (`id()` alone drifts —
  CPython recycles the address of a collected object). Each distinct query
  is screened once per configuration in the same call, so every row above
  is the same population: **3,650 `propagate()` calls, 3,534 distinct
  queries, 0 screen errors**. The clause switches are `descend ∈ {all,
  transparent4, toplevel}`, `stelling_any binds at its shape param ∈
  {on, off}`, `a jaxpr's invars bind ∈ {on, off}`, `resolve per scope ∈
  {on, off}`; the three conventions are applied to the same walk rather
  than measured in separate runs. Re-measured on the tree this fix ships
  in, every published figure in this section reproduced to the unit —
  15/16/18, 526, 1,378 across 532, 583 across 192, and the four individual
  clause rows.

  So on the suite's real work **only clause 2's `stelling_any`-param
  bullet ever changes a verdict** under the reading as first published;
  clauses 1, 3 and the invars bullet change nothing at all there. Their
  necessity rests on CONSTRUCTED documents — which were constructed, and
  do flag: clause 1's `lax.cond`-branch lie reads UNTOUCHED with clause 1
  off, and clause 3's scope-reuse document reads UNTOUCHED with clause 3
  off, both measured in the same run against their unedited controls. That
  is a weaker warrant than a suite measurement and the entry says so
  rather than letting the four clauses read as equally load-bearing.

  **The convention and the necessity interact, and the entry may not state
  one without the other.** Under the fail-closed convention this entry now
  publishes, the invars bullet is NOT inert: it moves **526** verdicts,
  because without it 1,378 references across 532 queries become
  unanswerable and unanswerable now means AFFECTED. A reader running the
  screen as written is therefore running the reading in which that bullet
  is load-bearing on ordinary work.

  **WHAT THE SCREEN AS SHIPPED ACTUALLY SAYS ON THAT POPULATION, printed
  beside the 526 because the 526 invites the wrong reading — audit 0.2.0
  B6 audit 4, F5.** *The screen does not flag 526 of anything.* With the
  invars bullet ON, which is how it is published and how a reader will run
  it, those 1,378 sub-jaxpr-invar references RESOLVE — unresolvable drops
  to **2 across 2 queries** — and the screen's own output over the same
  3,534 is:

  ```
  the SHIPPED screen, all four clauses ON, fail-closed convention
      UNTOUCHED   3,516
      AFFECTED       18   = 16 in tests/test_aval_lie_both_faces.py, the
                            file that builds affected documents on purpose
                          +  2 fixtures that deliberately hold an unbound
                            variable (the screen saying "I cannot answer")
      false positives 0
      unresolvable references  2, in exactly those 2 fixtures
  ```

  The **526** is a CLAUSE-ABLATION COUNTERFACTUAL: it is how many verdicts
  move if you delete the bullet from the screen, not how many documents the
  screen accuses. Both numbers are about the same run, and quoting the
  second without the first reads as an alarm about the suite.

  **WHAT THIS SCREEN IS BLIND TO — audit 0.2.0 B6 audit 3, F7.** The
  screen is **witness 1 and nothing else**. It reads shapes recorded in
  the IR; it reads no propagated box and no const payload, and clause 4
  walks equation OPERANDS. So it inherits the binding witness's documented
  blindness exactly, and an UNTOUCHED from it is not a clean bill of
  health. **This matters more than its severity suggests, because the
  screen is RETROSPECTIVE**: it is pointed at verdicts produced by older
  trees with fewer checks, so "the current tree catches this by other
  means" is a statement about the current tree and not about the verdict
  being screened. Three classes, each with what does catch it here:

  - **A consistently relabelled computed value.** The lie applied at the
    binding AND at every reference leaves no disagreement in the IR to
    find. Caught in this tree by the **propagated box** — the witness a
    consistent lie cannot forge in general, because the interval leg
    computes the shape from the values flowing in — and, on a relabelled
    `mul`, ALSO by the emission's own elementwise pairing rule with the
    box withheld entirely (`env={}`): *"'mul': operand shapes (4,) and
    (4,) broadcast to (4,), not the output shape (2,)"*. Both are recorded
    because "the box is the sole detector" is a claim this entry has
    already been wrong about twice. The box is blind inside transparent
    call bodies, where there is none.
  - **A constvar whose aval disagrees with its const payload.** A
    payload is a self-description of a different kind — bytes, or a
    python scalar — and no clause reads it. Caught by
    `_Slicer.slice`'s const decode pass, which quotes both counts:
    *"constvar 0 decodes to 4 element(s) but its aval shape (2,) holds
    2 (aval/value mismatch, malformed IR)"*.
  - **A jaxpr's own `outvars`.** Clause 4 walks *"for every equation, for
    every operand"* — equation invars. A `Var` in a jaxpr's outvar list is
    not an operand of any equation, so a shape lie there is never
    compared. The emission's own cross-check walks the same set, which is
    why the screen mirrors it rather than being wider.

  All three are constructed and driven in `tests/test_ir_screen.py`, which
  implements the four clauses from this text — not by calling the library —
  and measures both that the screen says UNTOUCHED and what in the tree
  refuses each document.

  **There is no screen on the VERDICT for this one, and that is a finding
  rather than an omission.** S12's verdict-side signature worked because
  the transfer refused and left a ⊤ in the coverage record. Here the
  transfer succeeds: coverage reads `4 eqns: 4 known (100%)`,
  `unknown_primitives` is empty, and no note mentions the primitive. An
  affected run is indistinguishable from a clean one on the rendered
  verdict. Screen the IR.

  **WHAT TO RE-RUN.** Any **VERIFIED or REFUTED** whose query was
  deserialized or hand-built. REFUTED is in scope because of the witness
  half above — that is new relative to S12, whose entry named VERIFIED
  only. On a tree carrying this fix, an affected query returns **UNKNOWN**,
  with the obligation's detail reading

  ```
  escalation declined: 'dot_general' refers to variable 2 at shape (2,)
  but it is BOUND at shape (4,): a value has one shape, and an emission
  that read the reference would model a different array than the one the
  query computes (malformed IR)
  ```

  naming the variable and both shapes. The declaration form reads the same
  way and names the same two shapes, the bound one now being the `shape`
  param's:

  ```
  escalation declined: 'reduce_sum' refers to variable 5 at shape (2,)
  but it is BOUND at shape (4,): a value has one shape, and an emission
  that read the reference would model a different array than the one the
  query computes (malformed IR)
  ```

  Unaffected queries return exactly what they returned before — see the
  cost measurement below.

  **THE COST, MEASURED — AND THE METHOD, BECAUSE THE FIRST TABLE HERE
  COULD NOT BE RE-DERIVED FROM ITS OWN SENTENCE** (audit 0.2.0 B6
  re-audit). It said *"over the whole test suite (every obligation slice it
  builds)"* and printed zeros. Over the whole suite the check raises
  declines, necessarily: the suite contains this class's own reproducers,
  which are malformed IR on purpose. The zeros were the suite MINUS that
  file, unstated. **A figure that cannot be re-derived from a stated method
  is the defect this repository treats as first-order**, so the method is
  now the table's first line.

  **METHOD.** Wrap `_Slicer._one_shape_per_value` in a pytest plugin that
  mirrors its short-circuits exactly (a decline stops the equation, so
  counting past one is counting work the check never did), attribute every
  count to the test FILE that produced it, and run the whole suite once.
  Then partition on `declines > 0`. The partition is derived, not chosen:
  no file is named in advance.

  **RESULT** (this tree; jax 0.11.0, `JAX_ENABLE_X64=1`, one
  `pytest -q -p no:randomly` over `tests/`):

  ```
                                          WHOLE SUITE   WELL-FORMED REMAINDER
  equations validated                          10610                   10503
  Var invars examined (binding witness)        13419                   13286
     with a binding this slicer found          13419                   13286
                                            (100.00%)               (100.00%)
     disagreeing with their binding               11                       0
     with NO binding (the fail-closed arm)         0                       0
     with an unreadable declaration shape          0                       0
  Var atoms examined (box witness)             24006                   23789
     with a propagated box                     23289                   23112
                                              (97.01%)                (97.15%)
     disagreeing with their box                    2                       0
  declines raised by this check                   14                       0

  files that produced a decline:
      tests/test_aval_lie_both_faces.py   13
      tests/test_ir_screen.py              1
  ```

  The partition lands on **exactly two files**, and both are files whose
  whole purpose is to hand this check malformed documents. So the claim the
  numbers support is: **zero declines, and zero disagreements on either
  witness, over every obligation slice the suite builds from a well-formed
  query.** The 11 binding disagreements and the 2 box disagreements are the
  reproducers being caught, which is the check working rather than costing.
  **The well-formed-remainder column is byte-identical to the one this
  entry first published** (10,503 / 13,286 / 23,789 / 23,112 / 97.15%);
  only the whole-suite column moves, by the malformed documents audit 3's
  fixes added. *(Re-derived by the same method after audit 4's fixes:
  BOTH columns reproduce to the unit — 10,610/10,503 equations,
  13,419/13,286 references all bound, 24,006/23,789 atoms,
  23,289/23,112 boxed, 97.01%/97.15%, 14/0 declines, and the same
  two-file partition with the same 13/1 split. Audit 4 added no
  document this check declines on.)*

  **AND THE INSTRUMENT PERTURBS EXACTLY ONE SUBJECT, WHICH IS WORTH MORE
  THAN THE ROW IT COSTS.** The plugin's replica calls
  `self._binding_shape(atom)` in order to count, so on any document whose
  answer depends on HOW MANY TIMES the declaration is read, the
  measurement is a second reader. The suite contains exactly one such
  document — `test_the_declaration_reader_is_a_FUNCTION_and_not_a_single_
  READ`, whose `shape` param is a `list` subclass that answers differently
  between iterations — and it is the one and only replica/real mismatch in
  the run (`reduce_sum`: replica predicted no decline, the real check
  declined), and the one test that reds UNDER THE PLUGIN and nowhere else.
  That is the F4 finding measuring itself: a reader that re-reads is a
  reader, including when the reader is the instrument.

  The 2.85% of atoms the box witness cannot see are exactly the inner-scope
  ids — which is why the binding-site witness must be total in its own
  right, as UNSOUND-1 above measured the hard way.

  Suite, re-derived on this tree: **3615 passed / 10 skipped** with
  `JAX_ENABLE_X64=1`, **3616 / 9** without it as CI runs, skip sets
  differing by exactly `test_tripwire_arm.py:643`. *(This read 3589/3590
  at `30d4b04`, re-derived there before audit 4's +24.)* One pre-existing test
  changed in the first pass and was REPAIRED rather than relaxed:
  `test_three_rows.py::test_slice_params_contradicting_the_aval_decline_with_the_form_quoted`
  built its own fixture by handing a shape-`(1,)` slicer an operand
  `var(1, aval((3,)))` — an instance of this very class — so the new check
  answered before the slice-routing rule the test is named for. The
  operand is now given a slicer that really does bind it at `(3,)`.

  `tests/test_aval_lie_both_faces.py` carries all six original reproducers,
  the declaration-shape reproducers the re-audit added, the unedited
  controls for each, and the truth in exact `Fraction` and in concrete jax.

- **2026-08-15 (B6): a `TypeError` out of the public `propagate()`, and a
  docstring asserting it could not happen.** Audit 0.2.0 **S12″**. A
  non-integer entry in a `dot_general`'s `dimension_numbers` passes the
  range test — `0 <= 0.0 < 1` is True — and then reaches `lhs_shape[i]`,
  where python raises `TypeError: tuple indices must be integers or
  slices, not float`. `propagate._t_dot_general` and `propagate.eqn` catch
  `IntervalError` and nothing else, so this was a raw crash out of a
  public entry point on a document `from_dict` accepts; the emission face
  catches only `IntervalError` too, so the same malformation arrived there
  as an *"internal error"* decline through the blanket net. Two faces, two
  behaviours, one malformation — the S12 shape, in the oracle's own
  contract.

  The crash is **pre-existing**; what was new in `4d793cf` is the sentence
  *"Raises `IntervalError` on any malformation"* in
  `dot_general_geometry`'s docstring, which made a false promise where a
  reader would look for the true one. The dims now go through
  `operator.index` first, exactly as `check_shape` already does for
  extents, and raise `IntervalError`. Measured before and after, on a
  traced query whose serialized `dimension_numbers` is edited to
  `(((0.0,), (0,)), ((), ()))` and reloaded through `from_dict`:

  ```
  4d793cf:  propagate: RAISED TypeError: tuple indices must be integers
                       or slices, not float
  fixed:    propagate: returned; obligations [(0, 'unknown')]
            note: 'dot_general' has no sound rule for params {...}
            escalate 0 -> unknown: escalation declined: 'dot_general'
                       declined: dot_general lhs dimension 0.0 is not an
                       integer (malformed IR: from_dict does not coerce
                       dimension_numbers entries)
  ```

  **No verdict is retroactively invalid**: the old behaviour was a crash,
  and a crash produces no verdict.

  **AND THAT FIX WAS HALF A FIX — audit 0.2.0 B6 RE-AUDIT, R4.** It CALLED
  `operator.index(d)` and **discarded the result**, so the dims were
  validated and never NORMALISED. Everything below the guard still ran on
  the raw objects: `set(dims)` hashes them, `0 <= d < len(shape)` orders
  them, `shape[i]` indexes with them — three protocols, one unvalidated
  object. A 0-d `numpy` array satisfies `__index__` and is UNHASHABLE, so
  it passed the guard and raised a raw `TypeError: unhashable type:
  'numpy.ndarray'` out of the public `propagate()` while the emission face
  declined: **the same two-faces split, one type level up**, because the
  repair had tested a predicate where it should have produced a value.
  `check_shape` was the model all along and does it right — it binds
  `k = operator.index(d)` and tests `k`. Measured on `96ab47a` and on this
  tree over a five-object attack set through both faces:

  ```
                     96ab47a propagate()          this tree
  0-d np.array(0)    RAW TypeError: unhashable    returns; emission declines
  np.int64(0)        ok; emission declines        ok; emission declines
  float 0.0          ok; emission declines        ok; emission declines
  str "0"            ok; emission declines        ok; emission declines
  bool True          ok (leaked a bool into       ok (geometry holds int 1)
                     the returned geometry)
  ```

  The dims are now bound to what `operator.index` returns, so the
  `DotGeneralGeometry` the emission and the replay both read holds plain
  `int`s that no protocol can surprise — pinned as that property in
  `test_dot_general_both_faces.py::test_the_oracle_NORMALISES_its_dims_
  and_does_not_merely_check_them`, rather than as a list of refused
  inputs, because a list of inputs is what the next exotic type gets past.

- **2026-08-15 (B6): a REGRESSION introduced by `4d793cf` and fixed before
  release — `escalate()` raised where `dee8bc2` returned a verdict.** Audit
  0.2.0 **M17′**. The M17 fix added an unguarded
  `tuple(eqns[pos].source_info) != tuple(o.source_info)` to
  `slice_unknown_obligations`. Both callers — `solvers.escalate` and
  `affine.refine_propagation` — iterate that function **in the `for`
  header**, outside their own per-obligation `except Exception`, so a raise
  there costs every obligation's verdict and not only the offending one:
  precisely the whole-query-answer-to-a-per-obligation-question failure
  M17 exists to have fixed. Reachable from hand-built IR, which this page
  names as in scope (`from_dict` coerces at its own door and nowhere else).
  Measured on the same query, an `int` where the frames go:

  ```
  dee8bc2:  escalate() returned: [(0, 'violated-witness')]
            refine_propagation() returned
  4d793cf:  escalate() RAISED TypeError: 'int' object is not iterable
            refine_propagation() RAISED TypeError: 'int' object is not iterable
  fixed:    escalate() returned: [(0, 'unknown')]
              "escalation declined: the source_info of the query's assert
               at top-level position 4 is 7, which is not a list of source
               frames: the association ... cannot be CHECKED at all"
            refine_propagation() returned
  ```

  Two changes, and the order matters. The comparison is now **total** — a
  helper reads either side as a frame list or as `None`, and `None` means
  the association cannot be CHECKED, which is a decline with its own
  sentence rather than the disagreement sentence (which would have printed
  the useless *"traced at 7 but records 7"*). And the per-obligation body
  is **netted**, per obligation rather than per function, so the next line
  added to it cannot escape either while a sibling obligation still gets
  its own answer. The `affine` leg was fixed by the same change, at the one
  place both callers share.

  **FOUR TOTALITY CLAIMS IN THAT REPAIR WERE NOT TOTAL — audit 0.2.0 B6
  RE-AUDIT, R5 / R6 / R7 and the claimants read.** None moves a verdict:
  every one of them is a raise where a decline belongs, and three of the
  four are caught by the net the same repair installed. They are listed
  because a totality claim nobody drove is what this batch has already
  been wrong about twice.

  - **R5, `_frames` (fixed in the code).** `isinstance(v, list)` is a claim
    about the TYPE, not the object: a `list` SUBCLASS whose `__iter__`
    raises satisfies it and then raises inside `tuple(v)`, so the docstring
    saying not raising is "the structural guarantee" was an assertion the
    function did not keep. A value that will not iterate is not a frame
    list either, and now reads as `None` by the same reasoning an `int`
    does.
  - **R6, the preamble (fixed in the claim).** *"It reads `e.primitive` and
    `o.top_level_eqn_pos` and compares them, which cannot raise on any
    object"* is false — there are eight reads out there, and four of them
    are the caller's first raise. The TRUE argument is **shadowing**, and
    it is the one the docstring now makes: both callers derive
    `interval_env(closed)` and their own `unknown` list comprehension over
    `propagation.obligations` BEFORE calling, so an object that will not
    iterate has already raised in the caller, where it is the caller's own
    crash and not a lost batch of verdicts. The residual is named rather
    than denied: an object that survives those reads and raises only on
    `.primitive`, on `.top_level_eqn_pos`, or on `hash()` of what the
    latter returns.
  - **R7, the net's own handler (fixed in the code).** A net that re-raises
    while composing its own message is not a net — the escape costs every
    sibling's verdict exactly as the original raise would have. `str(e)`
    runs the exception's own `__str__`, and `getattr(o, name, default)`
    returns the default only for `AttributeError`. All three reads now go
    through `_safely`, which substitutes a visible placeholder.
  - **The claimants count (fixed in the code).** `claimants.get(pos, 0)`
    guarded a sentence that then printed `claimants[pos]` — two reads, and
    the second raised `KeyError` whenever the key was absent, turning *"N
    obligations claim top-level assert position P"* into *"internal error:
    KeyError: 3"*. The absent case is reachable because `pos` is itself a
    SECOND read of `o.top_level_eqn_pos` and `claimants` was built from a
    first. One read now serves both.

  All four are driven in `tests/test_obligation_slice.py`, by hostile
  objects rather than by injection where the route allows it.

  **AND ONE THAT IS DISCLOSED RATHER THAN FIXED: a raw `TypeError` out of
  the public `make_verdict`.** A query carrying a param `ir._encode` cannot
  serialize — a `list`, before the declaration door above refused the one
  that mattered — makes `ClosedJaxpr.content_hash()` raise, and
  `verdict.make_verdict` calls it unguarded while building the stamp. The
  caller gets `TypeError: stelling.ir cannot encode list` from three frames
  down instead of a refusal that names the reason, which is not the posture
  this library takes anywhere else. It is left for two reasons and both are
  about not making it worse. It is a **door that is shut**, not one that is
  open — the re-audit brute-forced every `ir._encode`-able alternative and
  only `str` qualifies, which both faces refuse — so the obvious "fix" of
  teaching `_encode` to emit lists would OPEN the serialized route to the
  declaration lie above; and choosing `make_verdict`'s public error
  contract is cross-module work. Note that it was never containment for
  UNSOUND-1 anyway, and is not now: the slicer declines the lie whether or
  not the query can be hashed.

- **2026-08-15 (B6): "strictly stronger than the count check" was an
  overclaim; narrowed, with the boundary measured.** Audit 0.2.0 **M17″**,
  and a CLAIM defect rather than a code one — no verdict is affected. M17's
  per-obligation association records a position and verifies three things
  about it (it names a `stelling_assert`, it carries the same
  `source_info`, exactly one obligation claims it), and the test asserting
  the safety property called that *"strictly stronger"* than the count it
  replaced. It is not. Two queries traced from the **same factory** carry
  byte-identical `source_info` at the same position, so all three guards
  pass and the wrong-query slice comes out — exactly as under the count:

  ```
  source_info identical across the two queries: True
  content hashes differ:                        True
  -> SLICED index=0, inputs bounded (0.0, 1.0)  <- query B's declaration,
                                                   under query A's propagation
  ```

  The true statement is that the mapping is **finer** — it answers per
  obligation what the count answered per query — and that on the
  wrong-query attack it catches strictly more than the count did (a
  differing `source_info`, a differing position, a contested position) but
  not all of it. **The containment is one layer up and is the same defence
  the count check had**: `make_solver_verdict` raises
  `MispairedEscalationError` when the escalation's recorded query hash is
  not the hash of the query being stamped — and also when that hash cannot
  be established at all.

  **AND THAT GATE BINDS ONE OF THE THREE ARGUMENTS, SO IT IS NOT
  CONTAINMENT FOR THIS — audit 0.2.0 B6 RE-AUDIT, UNSOUND-3, DISCLOSED AND
  NOT CLOSED.** The gate is real and unconditional on the leg it covers
  (13 drives refused, including one with an unhashable query, with an
  injected-defect control proving the harness reads the live gate). The
  leg it covers is the ESCALATION. It does not cover the PROPAGATION,
  because `Propagation` carries no query identity to check against — and
  the discharges do not have to come from an escalation. Measured on this
  tree, on `main` (`dee8bc2`) and on the released **`v0.1.0`**, on two
  queries traced from one factory:

  ```
  make_solver_verdict(B, propagation_of_A, escalate(B, p_A))  -> VERIFIED
  B's honest verdict                                          -> REFUTED
  ```

  with **no exception anywhere**: `escalate` hashes the `closed` IT was
  handed, so the pair the gate checks genuinely matches while B's
  obligations are reported with A's statuses. On an obligation the
  interval leg decides outright the escalation carries no records, no
  notes, no spawns and no stamps — `carries_work=False` — and the gate is
  not consulted at all, reaching the same false VERIFIED with no solver
  record in it. `stelling.affine.refine_propagation` is public, sits below
  the gate, and writes its refined statuses into the same unbound
  argument, so it reaches the identical outcome.

  So the sentence in `Escalation.query_sha256`'s docstring that an exempt
  pairing "returns UNKNOWN off the propagation alone" was **false, and is
  struck**. The repair is an identity on the `Propagation` itself, checked
  wherever a propagation is consumed against a query — cross-module work,
  scheduled as its own change and deliberately not attempted here. What
  this batch did is stop the tree claiming a containment it does not have:
  `CHANGELOG.md`, `solvers.Escalation`'s docstring and
  `tests/test_verified_bar.py` now say which leg is bound, which is not,
  and that `carries_work=False` exempts the gate entirely.
  `test_verified_bar.py::test_a_mispaired_PROPAGATION_mints_a_false_VERIFIED`
  holds the live measurement in both forms, so closing it later is a test
  going red rather than an archaeology exercise.

  **Deliberately not closed inside `slice_unknown_obligations`.** The
  hazard is a caller-PAIRING error and it reaches all three arguments
  alike: `closed`, `propagation`, and `env` — the last a plain dict with no
  identity to check. A query-hash check on `propagation` alone would close
  one of three channels while reading as though it closed the question,
  which is the "check in one place the other does not consult" shape this
  whole batch is about. The boundary and the containment are both pinned in
  `test_nested_assert_escalation.py::test_two_queries_from_ONE_FACTORY_share_source_info_and_slice_through`
  so that a later reader meets the measurement rather than the overclaim.

- **2026-08-15 (B6, same batch): solver escalation stopped throwing away
  the whole query because ONE `assert_` was written inside a `jit`. UNKNOWN
  → REFUTED on real harnesses; NO prior verdict becomes invalid.** Audit
  0.2.0 M17, filed MINOR there because the old behaviour was strictly
  conservative — it declined, it never asserted. It is logged here anyway,
  because this page's policy is about verdicts FLIPPING and says nothing
  about which direction they flip in: measured below, this change moves 109
  queries from UNKNOWN to REFUTED.

  **THE MECHANISM.** Solver escalation slices a top-level
  `stelling_assert` equation. It decided which one an obligation belonged
  to by COUNTING: if `len(top-level asserts) == len(propagation.obligations)`
  then index `k` was assert `k`, and otherwise nothing could be mapped, so
  **every** unknown obligation in the query declined with

  ```
  escalation declined: 2 obligation(s) but 1 top-level stelling_assert
  equation(s): asserts nested in sub-jaxprs cannot be mapped to slices
  ```

  An `assert_` inside any transparent call (`jax.jit`, `custom_jvp_call`,
  `custom_vjp_call`, `remat2`), inside a `cond` branch, or inside an
  undescended `scan`/`while_loop` body is counted by the propagation and
  not by the top-level scan — so one of those cost solver escalation for
  every other obligation beside it.

  **THE COUNT CHECK WAS SOUND, AND SAYING SO IS WHAT LOCATES THE DEFECT.**
  The walk records exactly one obligation per top-level assert (the
  malformed-shape screen explicitly EXEMPTS asserts from declining so that
  it still does), so `len(obligations) >= len(asserts)` always, with
  equality exactly when nothing came from a sub-jaxpr. Under equality index
  `k` really is assert `k`. **No obligation was ever mis-sliced and no
  verdict was ever wrong.** The instrument was simply the wrong shape: a
  per-obligation question answered with a whole-query number.

  **Measured**, CPython 3.12.3 / jax 0.11.0 / jaxlib 0.11.0 / numpy 2.5.1 /
  z3 5.0.0 (wheel) / cvc5 1.3.4 (wheel) / Linux x86_64, `JAX_ENABLE_X64=1`.
  The audit's own five harnesses (`tests/test_nested_assert_escalation.py`),
  before and after:

  ```
                            BEFORE                     AFTER
  a_alone                   VERIFIED #0 discharged     unchanged
  b_alone                   VERIFIED #0 discharged     unchanged
  both_top_level            VERIFIED #0,#1 discharged  unchanged
  one_nested                UNKNOWN  #0 unknown        #0 DISCHARGED
                                     #1 unknown        #1 unknown (nested)
  budget_is_per_obligation  UNKNOWN  #0 unknown        unchanged
                                     #1,#2 discharged
  ```

  `one_nested` states exactly the two obligations `a_alone` and `b_alone`
  state; only the second is written inside a `@jax.jit` helper. The last
  row is the control that rules out the element budget: it was always
  strictly per-obligation and is unchanged.

  **The coverage recovered, on a corpus built for it** — 246 harnesses /
  684 obligations: every 2- and 3-assert query over a working set of four
  predicates whose truth over the declared box is derived by hand in exact
  `Fraction` arithmetic from monotonicity, in every nesting pattern, so a
  nested assert appears first, last and in the middle; 208 of them contain
  at least one jit-nested assert and 38 are all-top-level controls. Run
  twice, once with the count check reinstated:

  ```
  obligations undecided   BEFORE 584     AFTER 340     recovered 244 (41.8%)
  newly decided           123 discharged (all TRUE) + 121 violated-witness
                          (all FALSE), 0 disagreements with the exact oracle
  regressions             0  (nothing decided became undecided)
  query verdicts          BEFORE UNKNOWN 208 / REFUTED 35 / VERIFIED 3
                          AFTER  UNKNOWN  99 / REFUTED 144 / VERIFIED 3
  all-top-level controls  38 harnesses, 0 status changes
  ```

  Every one of the 340 obligations still undecided afterwards is a NESTED
  one — the residual is exactly the documented scope boundary and nothing
  else. VERIFIED does not move on this corpus and cannot: a nested
  obligation stays undecided, so a query containing one cannot reach
  VERIFIED however many siblings are discharged. What the recovery buys on
  such a query is REFUTED, which one violated sibling is enough for; on
  queries whose asserts are all top-level it buys nothing, because those
  were never affected.

  **WHICH PRIOR VERDICTS ARE RETROACTIVELY INVALID: none.** Every verdict
  this changes was an UNKNOWN, and an UNKNOWN claims nothing. **How to
  recognise one worth re-running**: its per-obligation `detail` reads
  *"escalation declined: N obligation(s) but M top-level stelling_assert
  equation(s)"*. No code path produces that sentence any more — it survives
  in the tree only in this entry, in `CHANGELOG.md`, and as a `not in`
  assertion in `tests/test_nested_assert_escalation.py` — so a verdict
  carrying it was produced before this fix, and is one this change may now
  decide.

  **THE FIX IS A CARRIED ASSOCIATION, CHECKED RATHER THAN GUESSED.** The
  walk knows exactly which equation it is looking at when it records an
  obligation, so it records it:
  `propagate.ObligationReport.top_level_eqn_pos` is the assert's position
  in the top-level `eqns`, or `None` when the obligation came from a
  sub-jaxpr (`_scope_path == ()` is the exact test, since every descent
  extends that path and restores it).
  `obligation.slice_unknown_obligations` then VERIFIES the record against
  the query it was handed — the position must name a `stelling_assert`,
  carry the same `source_info` the obligation carries, and be claimed by
  exactly one obligation — before slicing by it. Anything failing that, or
  carrying no position, declines individually with its own reason.

  That is strictly stronger than the count it replaces, not merely finer:
  the count could not tell one query from another of the same shape, so
  handing `slice_unknown_obligations` a propagation of query A and the IR
  of query B sliced B's asserts under A's obligation numbers whenever the
  totals happened to match.
  `test_nested_assert_escalation.py::test_an_obligation_whose_association_cannot_be_trusted_still_declines`
  builds exactly that pair and measures both halves — that the wrong-query
  slice really does succeed on its own, and that every obligation now
  declines.

  **A NESTED `assert_` IS STILL NOT SLICEABLE**, and this batch did not try
  to make it one. Escalation slices top-level asserts, `obligation.py`
  scopes it that way in writing, and lifting that is a capability change
  with its own soundness question: an assert inside a `cond` branch is
  CONDITIONAL, and slicing it as an unconditional obligation would be
  unsound. What changed is only that the decline is that one obligation's.

  **COUNTS AND ATTRIBUTION FOR THE WHOLE B6 BATCH** (both entries above).
  Same environment, `pytest -q -p no:randomly`, full suite:

  ```
  base dee8bc2      3453 passed, 10 skipped   (JAX_ENABLE_X64=1)
  base dee8bc2      3454 passed,  9 skipped   (no x64, as CI runs)
  after B6/4d793cf  3487 passed, 10 skipped   (JAX_ENABLE_X64=1)
  after B6/4d793cf  3488 passed,  9 skipped   (no x64, as CI runs)
  after B6/96ab47a  3535 passed, 10 skipped   (JAX_ENABLE_X64=1)
  after B6/96ab47a  3536 passed,  9 skipped   (no x64, as CI runs)
  after the re-audit fixes
                    3557 passed, 10 skipped   (JAX_ENABLE_X64=1)
                    3558 passed,  9 skipped   (no x64, as CI runs)
  after audit 3's fixes
                    3589 passed, 10 skipped   (JAX_ENABLE_X64=1)
                    3590 passed,  9 skipped   (no x64, as CI runs)
  after audit 4's fixes
                    3615 passed, 10 skipped   (JAX_ENABLE_X64=1)
                    3616 passed,  9 skipped   (no x64, as CI runs)
  ```

  The first two steps reconcile exactly: **+34** for `4d793cf`, all new
  tests — 19 in `tests/test_dot_general_both_faces.py` (15 of those one
  parametrised agreement case per form, 6 well-formed and 9 malformed), 4
  in `tests/test_dot_general_from_dict_door.py`, 9 in
  `tests/test_nested_assert_escalation.py`, and 2 added to
  `tests/test_obligation_slice.py`; then **+48** for `96ab47a`, which is
  `tests/test_aval_lie_both_faces.py` arriving. *(The `+48` line is a
  repin: this table read `3487/3488` at `96ab47a` — 48 low, in the very
  entry that supplied the 48.)* The re-audit's own step is **+22** in both
  columns, all new tests and nothing deleted or renamed: 11 in
  `tests/test_aval_lie_both_faces.py` (21 → 32), 2 in
  `tests/test_array_emission.py` (68 → 70), 4 in
  `tests/test_dot_general_both_faces.py` (42 → 46, one test and three
  parametrised rows), 3 in `tests/test_obligation_slice.py` (27 → 30) and
  2 in `tests/test_verified_bar.py` (57 → 59).

  Audit 3's own step is **+32** in both columns, all new tests and
  nothing deleted or renamed: 11 in `tests/test_ir_screen.py` (a new
  file — the published four-clause screen made executable, its three
  conventions, its three blind classes, and the attribution table's own
  arithmetic), 9 in `tests/test_aval_lie_both_faces.py` (32 -> 41,
  including 2 added parametrised rows), 6 in
  `tests/test_array_emission.py` (70 -> 76) and 6 in
  `tests/test_dot_general_both_faces.py` (46 -> 52). They reconcile
  exactly: 11 + 9 + 6 + 6 = 32.

  Audit 4's own step is **+26** in both columns, all new tests, nothing
  deleted: 12 in `tests/test_shape_param_rule.py` (a new file — the
  `shape`-param container rule measured against a COMPUTED population of
  container types, on both faces), 11 in
  `tests/test_ir_message_totality.py` (a new file — the message-totality
  sweep over a canonical document's own leaves, its positive control, and
  9 parametrised rows for the named quote sites), and 3 in
  `tests/test_aval_lie_both_faces.py` (41 -> 44: the element-count
  protocol, the one-read count readers, and one added parametrised row for
  the right container whose iteration raises). They reconcile exactly:
  12 + 11 + 3 = 26. **One test was RENAMED** and is the
  only departure from "nothing deleted or renamed" in this arc:
  `test_array_emission.py::test_the_declaration_check_reads_the_EXTENTS_
  not_the_param_type` is now
  `..._compares_BOTH_holders_and_refuses_the_rest`, because the old name
  stated the rule the code stopped implementing at `30d4b04` — the same
  defect as its docstring, one level up. The count is unchanged by it.

  The skip SET is unchanged in both environments at every step, and the
  one-member difference between them is still `test_tripwire_arm.py:643`,
  *"the threefry mask fires only at x64=0"* — re-measured here, one skip
  with x64 and none without, on that file alone.

  **Each part was reverted ALONE and the whole suite re-run**, so the
  coverage is attributed rather than assumed:

  | reverted alone | tests red |
  |---|---|
  | `_dot_general_plan` back to deriving its geometry from the LHS (the pre-fix S12 emission) | **12** |
  | `slice_obligation`'s guard net removed and its range test one-sided again | **3** |
  | the per-obligation re-association back to the whole-query count check (M17) | **3** |

  Two tests red on every one of those mutations and are excluded from each
  count, as the B4 and B5 entries exclude the first of them:
  `test_supported_primitives_doc.py::test_committed_page_matches_live_registries`,
  because the generated primitives page quotes source LINE NUMBERS and
  every mutation shifts them; and
  `test_sdist_contents.py::test_no_untracked_file_anywhere_would_ship`,
  because the attribution runs were driven before the three new test files
  were `git add`ed and an untracked file under `tests/` would ship. Both
  are artifacts of how the measurement was taken, not controls.

  **One pre-existing test changed status and was repaired rather than
  relaxed**:
  `test_verified_bar.py::test_what_a_stray_index_ACTUALLY_DOES_all_four_of_them`
  pinned `pytest.raises(IndexError)` for an index past the START of the
  assert list. That behaviour is what the guard-net half of S12 closed, so
  the test now pins the decline and its sentence instead — the property it
  was written for (all four stray-index behaviours end at the whole-query
  bar) is still asserted, and the loop over all four indices is untouched.

**Releases reached by an entry in this log.** `v0.1.0`, the only release,
is reached by **three** entries, all of them audit 0.2.0 findings and all
reproduced at the tag: the 2026-08-15 `exp`/`pow` libm-bracket entry (S11)
through `propagate(closed, semantics="ieee")`; the 2026-08-15 undescended-
`assume` entry (S13), through the ordinary `check()` path in real mode; and
the 2026-08-15 B6 `dot_general` entry (S12), through `from_dict`. Every
other entry is 0.2.0 development only, and **no release has yet shipped any
fix in this log**. This line read *"(no releases yet)"* until 2026-08-15, a
few lines below the reproduction that contradicts it; it then named S11
alone while the S13 entry above it said *"the second finding of that audit
to reach a shipped version"* — the same failure, one count shorter.
