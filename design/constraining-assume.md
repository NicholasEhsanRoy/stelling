# Constraining assume — enforce the precondition, don't drop it

**Status:** PRE-BUILD section written 2026-07-18, before the builder
launched; build record follows below it. The LA contract's second
prerequisite: an assume-guarantee contract whose `requires` is an inert
assume mints a VERIFIED that assumed its own precondition — F2's failure
wearing a contract's clothes. This build makes `assume` narrow the
propagated domain where — and only where — narrowing is provably sound.

## The F2 shape check — done first, because a claim hangs on it

**F2 is relational, from its own line.** The `h_f2` harness in
`corpus/supply/mime_fvm.py`:

```python
assume(dot >= 0.71 * sf_norm * d_norm)
```

*This cited `mime_fvm.py:202`, and that line is `PHI_PROTO = np.zeros(...)`;
the `assume` above is at 175, measured 2026-08-09. The number is dropped
rather than corrected, because nothing regenerates this page and the same
citation was already disagreeing with itself across two files —
`corpus/supply/affine_holdout/SCOUT_CASES.md` quotes the same tool note at
175. The harness name is what `grep` still finds.*

The comparison couples `dot` with `0.71·sf_norm·d_norm` — both sides are
computed, non-constant quantities derived from the declared `Sf`/`d`
component boxes. Under the census below that is the relational class,
which **stays inert in this build by design** (soundly constraining it
requires relational domains — affine territory — and interval narrowing
of a relation is exactly the aggressive-narrowing unsoundness).

**Therefore, stated before the build: this build does NOT close F1→F2.**
F2 remains blocked, and its block is now precisely characterised:
*relational domain machinery missing*, not "assume machinery missing."
The build is still the right next step — it is the LA contract's
box-aligned `requires` prerequisite, and box-aligned/intermediate-value
assumes are real, censusable, and soundly narrowable today. The outcome
table's third row is the operative one, recorded here in advance of the
post-build F2 re-run.

## The sound direction (the build's one real risk)

Inert-drop over-approximates (prove on a larger set — a-fortiori sound).
Constraining is sound **only if the narrowed domain is a superset of the
true assumed region** — narrow too aggressively and the tool proves the
property on a region smaller than the assume licenses, missing states
where it fails: a minted false VERIFIED. Binding consequences:

- Strict inequalities narrow to the **closed** half-space (`x > k` →
  `[k, hi]`): a `nextafter`-shifted or open narrowing excludes reals in
  `(k, nextafter(k))` — an under-approximation, forbidden.
- **No inversion through arithmetic**: `assume(w ≤ k)` where `w = x·x`
  narrows `w`'s own interval only — never `x`'s. Inverting needs sound
  inverse transfers that do not exist; a naive inversion (e.g. positive
  square root) under-approximates and is the core unsoundness the
  auditor attacks.
- **Relational assumes (both sides non-point) stay inert**, DROPPED
  disclosure intact, reason extended to name the relational block. The
  tool does not rewrite user forms to manufacture a constant side.
- **The safe default is inert.** Any shape outside the census drops,
  disclosed — inert is always sound; aggressive constraining is not.

## The census (what constrains in v1)

| pred shape | behavior |
|---|---|
| `cmp(v, k)` / `cmp(k, v)`, cmp ∈ {ge,gt,le,lt,eq}, k's propagated interval a finite point, v any env var (input or intermediate) | narrow `env[v]` by exact interval meet with the closed half-space (eq: with `[k,k]`); elementwise for arrays (assume's universal reading, matching `assert_`) |
| `and(p1, p2)` | recurse into both conjuncts |
| both sides non-point (relational) | inert + DROPPED + relational reason |
| `or`/`not`/`xor`/non-comparison producers/⊤ bools | inert + DROPPED + reason |

Narrowing is **forward-only** (equation order; earlier uses saw the
wider interval — conservative) and applies to the compared variable
itself, never through its defining arithmetic.

**Forward-only NARROWING, query-scoped WITHHOLDING — and they are not in
tension.** The two are separate mechanisms and only one of them is
positional. Narrowing forward-only means an obligation traced above an
assume is judged over a wider set than the assume describes; that costs
precision, and both faces of the cost are sound — a definite violation
over the wider set is a violation at every point of the narrower one, and
a proof over the wider set is a proof over the narrower one. The
**withholding** — whether a definite violation may be called REFUTED at
all — is a fact about the run's whole assume state and is read once, at
the end, over every obligation
(`stelling.exactness.certifies_set_refutation`; see the 2026-08-08
SOUNDNESS entry). Both scopes err toward withholding, which is why they
can differ without either being unsound. Measured on the change's corpus:
order-dependent rows fell 16 → 2 of 38, and the 2 survivors are the
forward-only narrowing above, not the withholding.

**The residual's cost is two-sided, and this section used to state only
one side (corrected 2026-08-08).** The sentence here read "costs
precision (an UNKNOWN where a REFUTED was available) and can never mint
a verdict", which reads as *the residual cannot touch a VERIFIED*. It
can. Measured on jax 0.11.0 in all four `refine` × solver cells of
`check()`, over `x = any_array((), "float64", (0.0, 1.0))` with the
certified `assume(x >= 0.9)`:

| obligation | assume traced FIRST | assume traced LAST |
|---|---|---|
| `assert_(x >= 0.5)` | **VERIFIED** | UNKNOWN |
| `assert_(x <= 0.5)` | **REFUTED** | UNKNOWN |

Eight cells, all sound. The VERIFIED is the conditional claim and stamps
itself as one (`constrained assume at ...: the verdict holds where the
precondition holds — narrowed x0 (IR var 0) to [0.9, 1.0]`), so nothing
here is
a wrong VERIFIED; what the position moves is only how much the checker
could decide, and it moves it in both directions.

That quotation read `narrowed var 2` until 2026-08-20 and `narrowed x0
(IR var 1)` until 2026-08-21, and the harness above printed neither. The
message names a declared input the way the witness does, so the SHAPE has
been right since B8a; the INDEX was not. Driven on jax 0.11.0 with
`JAX_ENABLE_X64=1` over exactly the one declaration above, `check()`
prints `narrowed x0 (IR var 0)`; driven at `aabb58d`, the commit before
that change, the same harness printed `narrowed var 0`. `IR var 1` and
`var 1` are what a harness with a SECOND declaration prints when the
assume is written on that second one — driven too. The digit was wrong
when it was written and wrong again when it was corrected; see the
2026-08-08 SOUNDNESS entry, corrected with it.

## The empty region — the empty-set bug, one level up

An empty meet means the precondition is **definitely false on the whole
over-approximated domain**, hence on every reachable state: the
precondition is unsatisfiable over the declared box, the harness is
malformed, and every downstream obligation would be vacuously true over
∅. This is the `any_array((5, 3))`/`(inf, inf)` refusal class discovered
mid-propagation: a dedicated loud error (never VERIFIED, never silent),
naming the assume's source address, the variable, its interval, and the
constraint. Degrade-don't-crash does not apply — this is a harness
defect (the `TranscriptionError`/empty-declaration class), not an
analysis limitation.

## The stamp and the instrument

- A verdict under a constraining assume is a **different claim** —
  "holds wherever the precondition holds" — and the stamp says so:
  each constrained assume contributes a stamped assumption line naming
  the source address and the narrowed region; coverage counts
  `constrained` separately from `inert`.
- **The constrained-vacuity instrument** (registered here, before any
  constrained verdict exists — the ⊤-widening pattern extended): every
  counted-or-relied-on verdict that used a constraining assume is
  re-run with `assume_mode="inert"` (a propagation mode that reproduces
  the pre-build behavior exactly). Still discharges → the constraint
  was not load-bearing (the claim is unconditional; say so). Only
  discharges constrained → the constraint is load-bearing and the
  conditional claim is the verdict's content — carried by the stamp
  lines above. Combined with the existing inputs-only ⊤-widening (box
  load-bearingness), the two instruments separate what did the work:
  the box, the precondition, both, or neither (tautology).

## Verdict integrity (registered before the build)

No-assume harnesses re-run byte-identical (hit386, dfx#417, the #632
exhibit, the solver acceptance queries). Inert-staying assumes (F2)
re-run identical, DROPPED note intact. Assume-carrying verdicts may move
blocked→posed only; every such movement is inspected against the sound
direction. No count either way; F2 is MIME, held out — usefulness
characterisation only.

---

# Build record (2026-07-18, post-build)

## What was built (fresh-context builder, census-blind to demand cases)

As registered above, with the builder's flagged additions adjudicated:

- The census as implemented matches the registration; either operand
  order (a left-hand point flips the comparator); point side may be any
  point-valued interval (literal, const, point input, point
  intermediate); narrowing is the new exact `interval.meet` (additive,
  no rounding — max/min of exact endpoints) with the **closed**
  half-space; elementwise universal reading; forward-only; conjunction
  recursion with composing meets.
- **Census shapes found and declined, not guessed** (builder's flags,
  accepted): `ne` cannot narrow an interval (the hull of
  `[lo,hi] \ {k}` is `[lo,hi]`) — inert; `reduce_or` (`jnp.any`) is a
  disjunction — inert; `reduce_and` (`jnp.all`) *would* be the
  universal reading but is unregistered and outside the census — inert,
  so `assume(x >= k)` constrains while `assume(jnp.all(x >= k))` does
  not (recorded asymmetry, a censusable v2 row); integer bitwise `and`
  gated out; huge-int literal bounds (>2⁵³) decode non-point → inert.
- **Scope locality** (builder's addition, soundness-required): an
  assume inside a `cond` branch narrows that branch's scope only; a
  sub-jaxpr assume does not narrow the outer scope.
- `interval_env` defaults to `assume_mode="inert"` — the solver
  escalation's division-emission guard reasons about the *declared*
  box, and the builder demonstrated a constrained env would let `div`
  be emitted over a declared box containing the divisor's zero.
- Mode switch: `propagate(..., assume_mode="constrain")` default;
  `"inert"` reproduces the pre-build behavior byte-identically (pinned
  against the old note text); no-assume harnesses equal under both.

## Two fix items at adjudication (before the audit)

1. **The escalation seam** — flagged by the builder, closed by the v1
   refusal: `escalate()` declines every unknown obligation when the
   propagation constrained any assume (`coverage.constrained > 0`),
   with the reason quoted; zero invocations, derived absence. Rationale:
   emission covers the *declared* box, so `unsat` would be sound
   a-fortiori but `sat` is not — the witness validator checks
   membership in the declared box, so a precondition-violating witness
   would pass the conjunction and mint a REFUTED that does not refute
   the conditional claim. Demonstrated end-to-end with real solvers:
   the constrained propagation now refuses; the same harness under
   inert mode escalates to an honest witness-backed REFUTED of the
   *unconditional* claim. Faithful narrowed-bounds emission is a later,
   separately-audited build (the LA-contract composition).
2. **Definitely-false constant assumes** raise
   `UnsatisfiableAssumptionError` (both-point comparisons decided by
   the existing three-valued transfers; never indeterminate) — the
   empty-meet refusal class, applied consistently; definitely-true
   stays a no-op inert drop.

*(The two "known limits" recorded here pre-audit — the strict-at-
boundary collapse and the branch-scope raise — were both superseded by
audit findings and fixed; see the audit section. The strict collapse now
refuses; branch-scope unsatisfiability degrades with a vacuity note.)*

## The F2 re-run — outcome row 3, as pre-registered

MIME reinstalled at the pin (`7ce1efb`), MADDENING at its pin
(`849c391`, an import-time dependency of MIME's fvm package), jax
verified 0.11.0 throughout. Result: **F1 VERIFIED / F2 UNKNOWN / F3
UNKNOWN — every status identical to the recorded run.** F2's single
propagation note now reads, verbatim:

*(**Series scope of this row, settled 2026-08-08 and deferred until
now.** "jax verified 0.11.0 throughout" is the whole claim: this row was
measured on **one** series and no other, and the row above states nothing
about jax 0.10.2. That is not a choice made here — it is measured.
`import mime` resolves in the 0.11.0 interpreter and raises
`ModuleNotFoundError` in the 0.10.2 one, and no install into either is
permitted, so the F2 re-run **cannot** be reproduced on 0.10.2 from this
tree. Every other claim in this document is series-independent by
construction: it is about stelling's own behaviour on hand-built or
harness-built queries, which both series drive identically. This one row
is the exception, and it is scoped rather than generalised.)*

> assume constraint DROPPED (inert in MVP propagation) at
> corpus/supply/mime_fvm.py:202 (h_f2): VERIFIED proves a superset;
> UNKNOWN may be confounded by this drop **(relational: both sides
> vary — constraining needs relational domains)**

*Quoted verbatim from the run, and the `:202` inside it is the TOOL's reading
of `source_info` at the time — the `assume` is at 175 today. A transcript is a
record of what a program printed, so it is not edited; what is corrected is
the sentence above that repeated the number as if it were a current fact.*

**F1→F2 is not closed, and the tool now says why in its own output**:
the alignment floor is a relation between computed quantities, and its
block is relational-domain machinery (affine territory), not "assume
machinery missing". Constraining-assume advanced the box-aligned class
(the LA contract's box-aligned `requires` prerequisite) exactly as the
pre-build section predicted. No count moves; MIME held out; the F1
usefulness datum stands with its caveat unchanged.

## The audit — three build rounds, two re-attacks, and the standing rule's first application

**Round 1 (full component audit): 2 UNSOUND / 1 FRAGILE / 3 COSMETIC —
all confirmed and fixed.**

- **F1 UNSOUND — the strict-at-boundary empty region minted a REFUTED.**
  `x ∈ [0,1]`, `assume(x > 1.0)` narrows to `[1,1]`; the true region is
  empty, and `assert_(x <= 0.5)` produced a definite REFUTED rendered as
  a claim about the declared box. **The adjudicator's own pre-audit
  assessment of this exact limit had covered the VERIFIED face only
  (vacuous-but-sound) and missed the REFUTED face — a half-checked
  conjunct in the project's own record, caught by the audit.** Fixed:
  a strict comparator whose meet collapses onto the degenerate boundary
  point refuses (`UnsatisfiableAssumptionError`).
- **F3 UNSOUND — the escalation-seam refusal was single-mechanism.**
  Enforced only inside `escalate()`; caller mispairing (inert-mode
  escalation + constrain-mode verdict assembly) minted a REFUTED whose
  witness violated the stamped precondition — the auditor cited
  SOUNDNESS's own one-invariant-two-mechanisms commitment against the
  v1 refusal. Fixed: `make_solver_verdict` independently raises
  (`MispairedEscalationError`) when a constrained propagation pairs
  with any escalation carrying solver work.
- **F2 FRAGILE** — branch-scoped unsatisfiable assumes raised globally
  with a false whole-domain-emptiness message; now they degrade
  in-branch (constraint not applied — sound over-approximation) with a
  branch-vacuity note; top-level and jit scopes still raise (true
  there). **F4/F5/F6 COSMETIC** — conditional-REFUTED wording (a
  constrained REFUTED no longer claims "the stated box is not invariant
  as stated"), dropped-conjunct visibility in the coverage summary, and
  the ±inf-bound misclassification reason.

**Re-attack round (the standing rule's first application — UNSOUND
fixes are re-attacked by the auditor, not just regression-tested):
the rule found an UNSOUND escape in the F1 fix on its first use.**

- **F7 UNSOUND — box-nonemptiness certifies region-nonemptiness only
  when the box is exact.** The strict-collapse refusal closed the
  exact-box channel only; empty-precondition REFUTEDs still minted
  through three channels on over-approximated intermediates:
  correlation-blind `x·x` (box `[-1,1]`, true image `[0,1]` —
  `assume(w <= -0.5)` then REFUTED of the theorem `w >= 0`), the
  outward-rounding pad defeating the degenerate-collapse test, and `eq`
  narrowing outside the true image. Fixed by the **exactness split**:
  per-var exactness (declared `stelling_any` outputs and exact-point
  constants only; every transfer output non-exact; **scope invars never
  inherit** — the builder proved certifying branch invars would reopen
  F7 through selector correlation, and pinned it); non-exact-target
  assumes still narrow (sound) and still fire the emptiness refusals
  (sound from the over-approximation), but definite REFUTEDs judged
  under an uncertified precondition are **withheld** to UNKNOWN with
  the reason disclosed, and uncertified VERIFIEDs carry a stamped
  may-be-vacuous line. ~~Withholding is forward-scoped — the builder's
  reasoned deviation from the adjudicator's literal directive (a
  pre-narrowing violation is an unconditional fact), accepted.~~
  **SUPERSEDED 2026-08-08: withholding is QUERY-scoped.** The deviation
  rested on "a pre-narrowing violation is an unconditional fact", and it
  is not one: if the assumed region is empty then *every* obligation of
  the query is vacuously true, the ones written above the assume
  included — which is exactly what the refusal for a *detectably* empty
  region already did, ending the run whole. The tree was therefore
  query-scoped where it could detect emptiness and order-scoped where it
  could not, so the behaviour turned on whether the caller wrote
  `jnp.all(x >= 2)` or `x >= 2`. The principal ruled it query-scoped;
  the flag is now read once at the end of the run, through
  `stelling.exactness.certifies_set_refutation`, which the affine leg
  consults for the same decision. See the 2026-08-08 SOUNDNESS entry for
  the measured cost.

**Second re-attack (on the exactness split): 0 UNSOUND / 0 FRAGILE —
the attacks that were built did not break it**, and they were
scope-descent bookkeeping attacks, flag persistence, ordering edges,
exactness-leak probes, and certified-behavior regression checks. *Read
"the split held" until 2026-08-24. A round that found nothing bounds the
round; the split is held by the exactness argument this note states, and
that argument is what a reader should check.* The auditor also corrected its
own round-2 pin (an in-branch-REFUTED nonemptiness argument that was
construction-specific — selector correlation defeats it in general;
the builder's in-tree test generalizes it correctly). Two COSMETICs
closed in a final round: a definitely-true no-op assume no longer sets
the uncertified flag (it certifies satisfiability — predicate true on
the whole superset), and nonvacuity's definite-FAILED sentence gets the
same withholding as the asserts under an uncertified constraint.

**Pattern notes for the record.** The re-attack rule, registered one
pass earlier from the 4-B lesson, found a real UNSOUND escape in a fix
on its first application — the rule's justifying event arrived
immediately. And two half-checked conjuncts surfaced in the *process's
own record* (the adjudicator's one-faced limit assessment; the
auditor's construction-specific pin) — the conjunctive-validator
principle applies to assessments as much as to code.

## Final verification (main agent, at the gate)

- Suites: **venv-jax 431 passed / venv-nojax 354 + 7 skipped /
  venv-solverfree 423 + 8 skipped** (baselines 313 / 241+7 / 305+8;
  the new `test_assume_constrain.py` alone carries 108 tests; every
  audit finding is a permanent regression test).
- Recorded set identical: hit386 (VERIFIED / mutation REFUTED), dfx#417
  (VERIFIED / mutation UNKNOWN), exhibit_632 (UNKNOWN), cf_run
  (VERIFIED / VERIFIED / UNKNOWN / UNKNOWN); MIME F1 VERIFIED / F2
  UNKNOWN / F3 UNKNOWN. The inert-mode control is byte-identical to the
  pre-build behavior (pinned; independently verified by the auditor's
  extracted-git-HEAD comparison).
- All auditor constructions land in their post-fix expected states
  (f1/f3 die on the intended refusals; f2/f4/f5/f6/f8/f9 exit 0; every
  r*-control clean).
- **Wording debt, recorded**: the DROPPED note's "(inert in MVP
  propagation)" prefix is stale as a description (assume is no longer
  uniformly inert) but is pinned verbatim by the inert-mode
  byte-identity control and the recorded F2 note text — re-pinning the
  wording is a small future pass that must move the byte-identity pins
  in the same change.
