# The solver integration — orchestration record of the gated build

**Status:** BUILD RECORD, 2026-07-18. The first `SolverStamp(invoked=True)`
in the project's history: SMT escalation for obligations interval
propagation leaves unknown. Built by a fresh-context builder against a
settled design, verified against a known-answer acceptance case, audited
by a distinct fresh-context auditor under a transport/emission/stamp
mandate, fixed at adjudication, and landed in a single gated commit —
the `any_pytree` orchestration pattern, applied to the layer where the
soundness surface is translation rather than arithmetic.

## Trigger status — said precisely, before anything else

The registered solver trigger (`design/unknown-triage.md`: search-shaped
UNKNOWNs across ≥2 sources) **never fired and remains unfired** — zero
search-shaped UNKNOWNs in thirty-five passes of counting attempts. This
build was ordered at the fork the LA/stack probes earned
(`design/la-and-stack-probes.md` §3): the conditioning obligation is the
solver's **first demonstrated customer** — intervals straddle it, Z3
decides it both ways — and the owner chose the solver leg of the
affine/solver/publish fork. An owner decision on demonstrated-customer
evidence, not a trigger firing; recorded as such so the trigger's
history stays honest.

**No count moves.** The acceptance case is a worked example with a
known answer — never a corpus member; MIME stays held out; E2a standing
figures are untouched by this build.

## The settled design, as implemented

- **SMT-LIB2 text is the interchange artifact.** Every invocation is
  defined by one script; transports deliver it: z3 wheel bindings
  (`Solver.from_string`, fresh `Context` per run, thread guard +
  `interrupt`), cvc5 wheel via a **wall-guarded child process**
  (`stelling._cvc5_driver` — see the measured fact below), external
  cvc5 binary over stdin (`STELLING_CVC5` > wheel > PATH). All reached
  lazily through `stelling._optional`; nothing solver-shaped imports at
  module level (AST-enforced by the extended import-hygiene test).
- **Fragment routing:** linear → QF_LRA (z3 primary), polynomial →
  QF_NRA (cvc5 primary with `:nl-cov true` **and `:nl-ext none`** — the
  mutual exclusion pinned by explicit emission, per never-on-defaults).
  Anything outside the v1 scalar emission set declines with the
  primitive/form quoted — logics are never guessed.
- **Portfolio, disagreement as a bug oracle:** both installed solvers
  run; sat-vs-unsat raises `SolverDisagreement` carrying both verdicts,
  both option sets, and the scripts — checked at the raw-answer level,
  *before* any model handling, so model quality cannot route around it.
- **Never on defaults:** options ride in the script as `(set-option …)`
  lines; the stamp records the exact emitted set plus `set-logic` and
  the script's sha256; `SolverConfig` requires the timeout (its own
  no-defaults discipline).
- **Honest result mapping:** unsat → discharged; sat → REFUTED **only**
  after independent exact-rational replay of the witness (see the audit:
  now membership + violation); non-rational models (sat only at √2) →
  UNKNOWN by policy; unknown/timeout → UNKNOWN, invocation stamped —
  **a timeout is never a VERIFIED**, enforced at the emitted-option
  level and by a wall-clock guard.
- **Layering:** `stelling.propagate` gained no solver dependency (one
  pure accessor, `interval_env`); `make_verdict` is byte-identical; the
  solver-assisted path is a separate assembly (`solvers.escalate` +
  `solvers.make_solver_verdict`). Escalated: exactly the obligations
  propagation left `unknown`.

**Measured fact the design bent around:** the cvc5 1.3.4 wheel holds
the GIL through `checkSat`, and script-level `:tlimit` does not reliably
preempt the coverings solver (measured: >90 s wall on a 200 ms limit).
The spec's "thread guard for bindings" is unimplementable there; the
cvc5-wheel transport is therefore a killed-on-timeout child process,
disclosed in the transport string. z3 implements the thread guard
literally (releases the GIL, honors `Context.interrupt`).

## Orchestration — who did what

- **Main agent** (this record's author): wrote the builder spec
  (settled design + inlined acceptance case with known answers; no
  counts, no bands, no mention of what depends on the build), prepared
  the three venv surfaces, launched the builder, verified the
  acceptance bar independently, wrote the audit mandate, launched the
  **distinct** auditor, adjudicated every finding, routed fixes back to
  the builder, re-verified, made the single gated commit. Built
  nothing.
- **Builder** (fresh context): built `obligation.py` (slice extraction,
  fragment classification, exact-`Fraction` replay), `smt.py`
  (deterministic emission), `solvers.py` (transports, portfolio,
  dispatch, stamps), `_cvc5_driver.py`; extended `verdict.py`
  (tuple-of-invocations stamp, `Witness`, rendering honesty) and the
  hygiene/README-control tests. 28 judgement calls flagged, 2
  spec deviations (both forced by measured facts), and two spec errors
  found by the build itself — including the spec author's interval
  arithmetic (correlation-blind `b·b` upper endpoint 43.03125, not
  40.5).
- **Auditor** (fresh context, distinct): transport/emission/stamp
  mandate, 28 constructions (c01–c28), every mandated area reported
  attacked-clean / finding / not-reachable-with-reason.

Neither subagent saw counts, bands, or the dependency this build
unblocks; the builder did not read `design/` or `corpus/`.

## The acceptance bar, and its independent verification

The worked example (the conditioning obligation, stated as bare math in
the spec): `(a+c)² ≤ (a·c − b²)·10.125` over a,c ∈ [1,2].

- b ∈ [−0.5, 0.5]: interval straddles (UNKNOWN) → portfolio unsat →
  **VERIFIED**, two `invoked=True` stamps (cvc5 nl-cov primary, z3
  cross-check), nonvacuity 6/6.
- b ∈ [−1.4, 1.4]: sat → **REFUTED with witness** (a=1, c=1, b=−1; the
  spec's own example witness was a=3/2, b=5/4, c=3/2 — any replaying
  witness accepted).

Main-agent verification, independent of the builder's tests
(`scratchpad/acceptance_verify.py` — a historical measurement, tracked and
not in the sdist; what follows is what it checked): 22/22 checks — own
harnesses, own
raw-`Fraction` replay of the returned witness, own membership check of
the witness against the declared box, stamp-shape checks (both
invocations, `nl-cov`/`nl-ext` exclusion pinned, script hashes),
single-solver degradation runs, and the no-solver path (UNKNOWN,
absence stamped, `sys.modules` clean of z3/cvc5) in the dedicated
**venv-solverfree** surface (jax present, no solver installed — the
mechanical **demonstration** that the suite runs and the no-solver path
stamps its absence with z3/cvc5 out of `sys.modules`, alongside
venv-nojax). *Called "the mechanical proof of 'no solver is a required
dependency'" until 2026-08-24: a passing suite in an environment is a
demonstration over what the suite covers, and the import-hygiene check is
what makes the claim structural.*

## The audit — mandate, findings, adjudication

Mandate (fixed before launch): emission fidelity (strictness both
directions, closed bounds, exact dyadic literals, sharing/aliasing,
negation polarity, inert-assume non-emission, don't-care completion),
timeout-never-VERIFIED (option level + wall guards), disagreement
loudness and its seams, the dependency boundary, stamp integrity
(validation attacks, stamp==wire with recomputed sha256, README
can't-drift), witness-witnesses-the-whole-claim (violation AND box
membership), transport robustness zoo, mapping/decline guards, and
cross-run isolation.

**Outcome: 1 UNSOUND, 4 FRAGILE, 2 COSMETIC — all seven confirmed at
adjudication and fixed before landing; the emission core survived every
attack this audit constructed** (**no constructed emission attack reached
a wrong-VERIFIED-via-mistranslation** — the attack set is the bound, and
the sentence claimed reachability until 2026-08-24; witnesses landed
exactly on closed-box endpoints; the exact-dyadic discipline held down to
fl(0.1) = 3602879701896397/36028797018963968).

1. **F1 UNSOUND — witness box-membership was never checked.** A fake
   transport answering sat with an out-of-box model minted
   REFUTED-with-witness for a box on which the predicate is universally
   true. Found independently twice: by the main agent's own reading of
   `_dispatch_obligation` during acceptance, and by the auditor's c01.
   Fix: every witness value — solver-given and don't-care-completed —
   must be a member of its input's closed box (exact-rational, finite
   sides); an out-of-box value raises `EmissionInfidelityError`. The
   replay now checks **the whole claim**: membership and violation.
2. **F2 FRAGILE** — sat with a missing/empty model completed *every*
   input as a don't-care, then died on a misattributed
   `EmissionInfidelityError` fabricating model content. Fix: a model
   supplying none of the declared inputs is a transport failure →
   UNKNOWN quoted, invocation stamped; completion requires at least one
   solver-supplied input value.
3. **F3 FRAGILE** — duplicate model definitions resolved by sort order;
   undeclared names silently ignored. Fix: conflicting duplicates →
   UNKNOWN quoted; undeclared names → unused and disclosed.
4. **F4 FRAGILE** — the external binary's get-model-after-unsat
   tolerance ignored exit status and post-answer stdout: `unsat` + a
   segfault banner + exit 134 became an undisclosed VERIFIED. Fix: the
   tolerance is narrowed to the documented error shape and disclosed;
   any other nonzero exit / unexpected noise → UNKNOWN with everything
   quoted.
5. **F5 FRAGILE** — a constants-only (zero-input) refutable obligation
   degraded through an internal `StampError`, dropped both real
   invocation stamps, and stamped "no solver invoked" after two real
   invocations (reproduced with the real wheels). Fix: zero-input sat +
   replay-confirmed falsity is an honest constant refutation (REFUTED,
   no fabricated witness values); degrade paths keep their invocation
   stamps; the absence wording can no longer claim absence when
   invocations exist.
6. **F6 COSMETIC** — `Stamp.solver` tuples accepted an `invoked=False`
   element (renderable as a doubled-reason "invocation"). Fix: tuple
   elements must all be invocations; absence is only ever the bare
   stamp.
7. **F7 COSMETIC** — `SolverConfig(only=())` produced "no SMT solver is
   installed" + install hint with both wheels installed. Fix: `only=()`
   rejected at validation.

Every finding's construction is now a permanent regression test (the
standing rule: a fix touching the emission or stamp boundary carries
its own witnessed construction). Pattern note, fourth time: findings
cluster in the newest surface (model handling and disclosure around
the oracles), and the oracles themselves — disagreement checked before
model handling, replay before belief — were not broken by anything this
round built. *Read "held" until 2026-08-24; an oracle nothing broke is an
oracle nothing attacked successfully, which is a fact about the round.*

## Fix round — verification

All seven fixes landed (builder continuation, same context, adjudicated
directives verbatim), with every finding re-derived as a permanent
regression test (`tests/test_solver_audit_findings.py`, 18 tests —
including the c26 constants-only shape against the **real** z3+cvc5
portfolio, now REFUTED with both invocations stamped). Main-agent
re-verification, independent of the builder's report:

- Suites: **venv-jax 299 passed** / **venv-nojax 227 passed + 7
  skipped** / **venv-solverfree 291 passed + 8 skipped** — the last
  being the jax-present-no-solver surface, which exercises the boundary
  rather than proving it (see the note above).
- The 22-check independent acceptance run: all pass, h_A/h_B behavior
  unchanged (VERIFIED with two stamps; REFUTED with in-box,
  replay-confirmed witness).
- Audit constructions re-run: 24 of 28 exit 0 (every attacked-clean
  probe unregressed; c07/c18/c22 now clean); the 4 nonzero exits are
  fix-mandated behavior the pre-fix scripts hard-coded differently
  (c01 dies on the mandated `EmissionInfidelityError` naming the box
  escape; c27 dies on the mandated `only=()` ValueError; c26 shows the
  adjudicated constant-refutation REFUTED; c09's undeclared-name case
  shows the adjudicated disclose-and-proceed rather than the
  construction's stricter no-verdict criterion).
- **No recorded verdict flipped**: hit386 (VERIFIED / mutation
  REFUTED), e2a_417 (VERIFIED / mutation red), exhibit_632 (UNKNOWN),
  cf_run (VERIFIED / VERIFIED / UNKNOWN / UNKNOWN) — every recorded
  status reproduced in the rebuilt environment before the landing
  entry was written. The MADDENING/MIME harnesses require their pinned
  repos, absent from the rebuilt venv; the no-solver path they
  exercise is covered by the pre-existing test baseline, which passes
  unmodified.

## What this build changes about the README — and what it does not

The SMT/solver capability tokens leave the roadmap fence **in the same
commit** as the witness (`SolverStamp(invoked=True)` constructed in
src, the can't-drift test's `_witness_solver()` now true) — the
registered rule: unfence only with the witness, and then immediately,
or the README lies in the other direction. Two truth surfaces below
the capability region are updated in the same commit because this
build falsifies their old text: the `[solvers]` extra's "no verdict
uses a solver yet" note, and §License's "every verdict … involved no
solver at all" (now: absence stamped when true, every invocation
stamped when not, full suite green with no solver installed).

Known stale line deliberately NOT touched (out of this work order's
scope, recorded here): the "Doesn't" list still says `cond` falls to ⊤;
the control-flow build gave cond branch-descent. Fixing it requires
settling how the can't-drift control-flow witness tracks the walker's
special-cased cond handling (it is not a `TRANSFERS` row) — a
one-line-plus-witness job for a later pass.

## Standing after this build

- **Solver integration: BUILT and gated.** First `invoked=True`
  stamps; REFUTED-with-witness exists (the Kani-shaped concrete
  counterexample, replay-confirmed for membership and violation).
- **The LA contract layer** (`requires` = the QF_NRA conditioning
  bound, `ensures` = κ-derived norm-sensitivity) now has its solver leg
  in the tree — **still unbuilt, still gated**, per
  `design/la-and-stack-probes.md` §3.
- **Affine**: untouched, sibling escalation, its own gates
  (IEEE-first precondition; valid counting evidence 0).
- **Corpus expansion**: registered, unrun; nothing here moves it.
- **Scope held:** no LA contract, no affine, no corpus run, no count;
  MIME held out; no user contact; no E2b; CALMS out of scope.
