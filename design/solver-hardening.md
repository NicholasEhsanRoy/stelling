# The hardening pass — two audited invariants made unconstructable

**Status:** PASS RECORD, 2026-07-18, same day as the solver integration
landing. No new capability, no count, no corpus contact; a structural
refactor of *how* witnesses are validated and provenance is recorded,
with the registered criterion that **no verdict status may change
anywhere** — met, and verified beyond statuses: both acceptance queries
produce byte-identical content hashes AND byte-identical emitted SMT
scripts (all four smt2 hashes unchanged) after the refactor.

## Why this pass, and why before the LA contract

The solver build's audit was caught by a good auditor — and the project
has already watched "a good auditor" fail as a mechanism (the first
control-flow-era audit declared clean exactly where the second found
four defects; an audit inherits its auditor's attention gradient). The
UNSOUND (witness violation checked, membership assumed) and the worst
FRAGILE (stamp narrated "no solver invoked" after two invocations)
shared one shape: **invariants maintained by convention where they
should be enforced by construction** — the can't-drift principle, never
applied to these two because they didn't exist until the build. The LA
contract will compose across this seam and lean on the stamp harder
than anything yet; it should stand on a stamp that *cannot* lie, not
one that was caught lying once.

## The three structural changes, as landed

1. **The witness validator is a single conjunction.**
   `obligation.witness_is_valid(sl, values)` computes BOTH conjuncts —
   closed-box membership (exact rationals, finite sides) AND
   predicate-false-at-the-point (the exact-`Fraction` replay it alone
   calls for witness purposes) — and is the only place either conjunct
   is computed for witness acceptance (`_box_escape` deleted;
   `solvers.py` no longer imports the replay engine). The dispatch
   path's only `Witness(` construction site is the factory
   `make_validated_witness`, which routes through the single raising
   gate; the constants-only refutation routes through the SAME gate
   with an empty mapping. AST-pinned by tests: one construction site,
   no separate half-check helpers.
2. **The stamp is append-only, recorded at invocation.** An
   escalation-scoped ledger receives each fully-populated `SolverStamp`
   (pre-run cached version, transport, exact options + set-logic +
   script sha256) immediately *before* the transport runs — the record
   of the event exists before the event's outcome. Reasons carry
   invocation context only (fragment, portfolio role, obligation);
   outcome and latency land additively in notes. Nothing mutates, pops,
   filters, or rebuilds ledger entries. **Absence is derived, never
   written**: empty ledger → the absent stamp at a single AST-pinned
   `solver_absent` call site; the old degradation-branch narration is
   gone. Disclosed stamp-semantics change: a transport failing after
   the invocation is issued (exec failure) now stamps the invocation
   with the failure quoted in notes, where it previously stamped
   absence — statuses unaffected, the ask fully described.
3. **The spawn-counter provenance gate.** `_Backend.run` increments the
   ledger's spawn counter as its first act at the transport-entry
   boundary — mechanically disjoint from the stamp-append site, no
   shared updater (AST-pinned). `make_solver_verdict` unconditionally
   asserts spawns == invoked-stamp count before emitting anything;
   divergence raises `ProvenanceError` carrying both counts and the
   stamps. Two independent mechanisms for one invariant, deliberately
   anti-correlated: the ledger prevents the stamp lying in its own
   construction; the gate catches divergence for reasons the structure
   didn't foresee.

## Verification (main agent, independent of the refactorer)

- Suites: venv-jax **313 passed** / venv-nojax **241 + 7 skipped** /
  venv-solverfree **305 + 8 skipped** (baselines 299 / 227+7 / 291+8;
  +14 structural can't-drift tests).
- Recorded set identical: hit386 (VERIFIED / mutation REFUTED),
  e2a_417 (VERIFIED / mutation UNKNOWN), exhibit_632 (UNKNOWN),
  cf_run (VERIFIED / VERIFIED / UNKNOWN / UNKNOWN).
- Acceptance queries: h_A VERIFIED with two full-option stamps, h_B
  REFUTED with in-box replayed witness — same content hashes
  (c92d87e9…, 2cd88416…) and same four emitted-script hashes as the
  original build: the refactor changed how provenance is recorded, not
  one byte of what is asked.
- **Constructed divergence, both directions, my own constructions**
  (not the refactorer's tests): a spawn bypassing the counter → 
  `ProvenanceError` ("0 transport spawn(s) but 2 invoked=True
  stamp(s)… refusing to emit"); a double count → `ProvenanceError`
  ("4 … but 2 …"). The gate refuses to emit, as registered.
- Stamp reasons verified answer-free; outcome notes verified present.

## Orchestration

Fresh-context refactorer (the work order's own §4(b) logic applied to
this pass: the findings' adjudicator has a stake in them; a fresh
context doesn't), spec'd with the two findings, the three principles,
and the no-flip criterion — not the counts, not the LA dependency.
Environmental note: the session scratchpad (including all prepared
venvs) was destroyed mid-pass; the refactorer rebuilt all three
surfaces to spec and verified jax at exactly 0.11.0 (diffrax 0.7.2 +
blackjax 1.6.2 added to venv-jax to reproduce the 299 baseline — two
`test_any_pytree` cases need them; recorded-corpus statuses confirmed
unaffected).

## The principles and rules this pass registered

- Three commitments in `SOUNDNESS.md`: *a conjunctive verdict gets a
  conjunctive validator*; *provenance is recorded as it happens, never
  narrated after*; *one invariant, two anti-correlated mechanisms*.
- Three standing audit-process rules in `design/soundness-audit.md`:
  the mandatory structuralization question; UNSOUND fixes re-attacked
  by the auditor (forward-binding; the solver build's fix round
  predates it, recorded); acceptance/audit coverage deliberately
  anti-correlated.

## Non-goals held

No LA contract, no affine, no new capability, no count, no corpus run;
MIME held out; no user contact; no E2b; CALMS out of scope. The solver
leg the LA contract will stand on is now hardened; building on it
remains a separate, gated decision.
