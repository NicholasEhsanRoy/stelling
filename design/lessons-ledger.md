# The lessons ledger — cross-pass principles, durable and reviewed

**Status:** STANDING DOCUMENT, established 2026-07-18. Format per entry:
**(a)** the principle, stated generally; **(b)** the instance that taught
it; **(c)** status — *structural* (encoded, can't be violated),
*convention* (a rule followed), *watch* (a metric tracked); **(d)** where
it applies going forward, named.

**Two standing practices:**
- **Before each build**, review this ledger for entries whose
  "applies to" names the build; say in the build's record which entries
  lit up.
- **After each pass**, ask which cross-pass pattern the pass confirmed
  and whether it is here. A pattern with a second witness earns an entry
  the way a finding with a control earns a registration.

---

## L1 — An over-approximation certifies emptiness, never nonemptiness

**(a)** Under a sound over-approximation, *emptiness of the abstract
set proves true emptiness* (nothing in the superset ⟹ nothing in the
set), but *nonemptiness of the abstract set proves nothing* — the
abstraction may have added exactly the points you see. Any claim that a
region is inhabited must come from exact knowledge (a declared set, an
exact point, a concrete witness), never from an inflated box.
*(Correction recorded: the work order that commissioned this entry
stated the asymmetry in the reverse direction; the F7 mechanics are as
written here — box-nonemptiness treated as region-nonemptiness was the
hole.)*
**(b)** F7 (`design/constraining-assume.md`): `x·x` over `[−1,1]` has
box `[−1,1]`, true image `[0,1]`; a nonempty meet with an unsatisfiable
precondition minted REFUTEDs of theorems. Fixed by the exactness split.
**(c)** **Structural** — and lifted (this pass) out of the assume
machinery into a shared certification primitive that any
emptiness-dependent reasoning routes through.
**(d)** The LA contract's conditioning `requires` (whether an
ill-conditioned sub-region is empty — the same asymmetry, one build
away); anything that ever certifies a region inhabited or a
precondition satisfiable.

## L2 — Structural guarantees move fragility up a layer; the top layer is human adjudication

**(a)** Making code unable to half-check a conjunction moves the
half-checking risk to the reviewer. Human review can't be made
unconstructable, but it can be made conjunction-shaped: an adjudication
of a conjunctive property must record **both faces explicitly**, and an
adjudication that states only one face is the tell. The two-context
differential (adjudicator + auditor) is highest-value on adjudications,
not code — the code is increasingly structural, so the residual risk is
the human step.
**(b)** One pass after the single-conjunction witness validator landed,
the main agent's adjudication of the strict-at-boundary limit checked
its VERIFIED face and missed its REFUTED face (round-1 F1,
`design/constraining-assume.md`); the auditor's round-2 pin was likewise
construction-specific (its own admission).
**(c)** **Convention** — from this pass on, main-agent adjudications of
conjunctive claims state each face in writing ("face X: …; face Y: …").
**(d)** Every pass with a main-agent adjudication of a conjunctive
claim; audit mandates should aim constructions at the adjudication's
stated faces.

## L3 — The re-attack net: watch for moral hazard

**(a)** A safety net that catches fix escapes is either pure gain or a
subsidy for careless fixes; one data point cannot distinguish them.
**(b)** The UNSOUND-fixes-are-re-attacked rule found F7 in the first
fix on the rule's first application (`design/constraining-assume.md`).
**(c)** **Watch** — the metric: *UNSOUND-fix re-attacks that find an
escape / UNSOUND-fix re-attacks run*. Current: **2 / 5** —
constraining-assume pass: F1's fix → F7 found; F3's fix → clean; F7's
fix → clean. IEEE pass: U1's fix (the subnormal haze) → U2 found (the
haze was dtype-blind); U2's fix → clean. Both catches share a shape:
the fix was correct for the case that taught it and blind to a sibling
case (aggressive-vs-exact boxes; binary64-vs-per-dtype bands) — the
escapes are scope errors, not carelessness. Track per pass; act only on
a trend.
**(d)** Every pass that fixes an UNSOUND finding.

## L4 — Arguments from me, facts from the probe

**(a)** Every empirical claim gets measured; hand-assessments trend
optimistic in the direction the assessor wants, so registrations precede
readings and premises get verified before builds act on them.
**(b)** Recurring: the "trigger fired" recount (bjx#D416 was a genuine
violation, not dependency-shaped); hit386 counted as corpus when it was
the control; the solver spec's interval endpoint (40.5 vs measured
43.03125); the IEEE under-scoping admission in the IEEE work order
itself.
**(c)** **Convention** (standing meta-rule since the third review pass).
**(d)** Every pass; every premise a work order asserts.

## L5 — Guards generate hazards

**(a)** Every guard is a small program written under a safety
motivation, and the motivation over-fires; a fix or guard is new code
with the same defect rate as any code plus a pressure to close.
Mitigation that has worked twice: put the guard rule in the *builder's
spec before the build* instead of in a fix after it.
**(b)** #632's ULP guard caused #756's rejection loop; the first
audit's float→int fix introduced the ±2³¹ admission (4-B); shape guards
crashed legal forms; the predicted fourth instance did NOT occur twice
(any_pytree, solver build) when the rule was spec-carried.
**(c)** **Convention** (spec-carried guard rule in every builder spec) +
**watch** (instances vs. non-occurrences: 3 occurred, 2 prevented).
**(d)** Every builder spec; every fix round.

## L6 — An audit inherits its auditor's attention gradient

**(a)** A single audit pass is not uniform coverage; the places an
auditor declared clean are where a differently-aimed second pass finds
defects. Acceptance cases and audit mandates must be deliberately
anti-correlated (the dangerous bug is on neither path).
**(b)** The second soundness audit found 2 defects in transfers the
first had declared clean; the anti-correlation rule is now standing
(`design/soundness-audit.md`).
**(c)** **Convention** (standing audit-process rule 3).
**(d)** Every audit mandate.

## L7 — Make invariants unconstructable, not merely tested

**(a)** An invariant maintained by care (a check here, a convention
there) drifts; encode it so the violating state cannot be represented —
single-conjunction validators, append-only records, derived-not-written
facts, refusals at construction. The regression test catches the
instance; the structure closes the class.
**(b)** The hardening pass (`design/solver-hardening.md`): witness
conjunction, append-only stamp, provenance gate; the standing
structuralization question (audit-process rule 1). **Corollary with
three witnesses in one pass** (`design/ieee-semantics.md`): enforcement
belongs at the *choke point*, not at each consumer — the ieee build
enforced an invariant at one consumer and assumed it at the others three
separate times (the `select_n` selector flag, the convert whitelist, the
assume classifier reading comparison equations behind the guarded
transfer). A comment stating an invariant is the tell.
**(c)** **Convention** (the standing question) producing **structural**
artifacts per instance.
**(d)** Every UNSOUND/FRAGILE finding; every new capability's central
invariants.

## L8 — Semantics ≠ representation, and the dial must be stamped

**(a)** How brackets are computed is not what the verdict is about; a
verdict that does not name its arithmetic semantics hides the gap where
false-VERIFIED lives — and *protections that exist by accident of
imprecision* (a wide bracket straddling where float can't distinguish)
are deleted by any reasonable optimization unless the semantics is made
explicit first.
**(b)** The #632 exhibit (2-ulp brackets accidentally straddling); the
stamp's semantics field (third contract growth); the affine IEEE-first
precondition (`design/unknown-triage.md`).
**(c)** **Structural** (the stamp field, no defaults) + this pass makes
the ordering constraint mechanical (tightened domains refuse to run
under `real`).
**(d)** The IEEE build (this pass); affine, whenever built; any future
domain tightening.

## L9 — The instrument is often the constraint

**(a)** What a study can see is bounded by its instrument, and the
bound masquerades as a finding: a corpus too small to fire any ≥2
trigger reads as "nothing needed"; detection-worded search terms
returned a post-detection corpus; calendar time proxied for effort.
State the instrument's reach next to every null result.
**(b)** The corpus-limits meta-finding (7 cases, 4 single-sighting
failure modes, structurally unable to fire triggers); the attribution
probe's derivable corrections.
**(c)** **Convention** (registrations state instrument reach; null
results carry the bound).
**(d)** Every registration with bands; every null-result reading.

## L10 — Model the measured target, not the standard it claims

**(a)** A semantics mode is only sound for the execution it actually
models. Standards are aspirational; targets flush, fuse, reassociate,
and resolve precision per device. **Measure the target and model that**
— and where the target's behaviour is itself variable across devices,
make the affected outcomes *indeterminate* rather than picking either
behaviour, so the verdict is sound for both.
**(b)** The ieee build modelled textbook IEEE-754 gradual underflow;
measured jax 0.11.0 CPU binary64 is FTZ+DAZ (flushes subnormals in
arithmetic, **in comparisons**, and in libm) — seven false-verdict
shapes, including the project's own diffrax underflow-boundary shape.
The fix (subnormal haze → indeterminate in-band) was then found
**dtype-blind** on re-attack: the flush band is per-dtype, so f32
subnormals escaped it, and the `_EXACT_CONVERSIONS` whitelist's
"value-preserving for every representable source value" was measurably
false under DAZ. Both are the same lesson at two scales. Ancestor:
SOUNDNESS's "one jaxpr, three devices, three numerics" — registered as
a stamp-disclosure rationale, which then bit as an actual soundness bug.
**(c)** **Structural** where it can be (in-band outcomes indeterminate;
non-binary64 floats decline with the gap quoted) + **convention** (a
semantics claim needs a measurement, never a specification reading).
**(d)** Any future semantics work — a QF_FP solver fragment, GPU/TPU
targets, fused-multiply-add or reassociation modelling, affine under
ieee; and any claim of the form "this operation preserves values."
