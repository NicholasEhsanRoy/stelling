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
escape / UNSOUND-fix re-attacks run*. Current: **8 / 12** —
constraining-assume: F1'→F7 found, F3'→clean, F7'→clean. IEEE:
U1'→U2 found, U2'→clean. Three-rows: int-guard'→U3+U4 found,
div/FMA'→U5 found, taint'→clean (cosmetic only). Array-emission arc
(2026-07-21): F3'→**R1+R2 found** (R1 public-API UNSOUND),
negshape'→**N1+N2 found**, read-gate'→**P1 found**, P1'→**I1 found**
(inside the documented residual class; the cycle then ended by the
pre-fixed stop rule, not by exhaustion). Every catch is a fix correct
for the case that taught it and blind to a sibling — the answer stays
L12's axis-named sweep, not more care. **Second convergence signature,
recorded with the arc:** the array-arc catches were progressively
narrower (public API → 2 coordinated from_dict lies → 3 lies →
direct-construction-only residual), and the final round **found nothing
new inside the scope it attacked** — the rule kept catching, but each
catch was strictly smaller, and the honest terminus was a **decision**
(stop and surface the residual class as an owner posture question)
rather than a clean round. A net that only ever returns clean was
suspicious; a net whose catches shrink monotonically toward a
documented residual is what closing a class actually looks like.
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
**watch** (**3 occurrences observed; 2 builds in which the predicted
instance did not appear**). *The second figure read "2 prevented" until
2026-08-24, which counts two non-events as two successes of the
convention. A build in which a defect did not appear is not evidence the
convention prevented it — L23 in this file is the general rule, and the
convention rests on the three occurrences, which are readings.*
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
precondition (`design/unknown-triage.md`). **Second witness, opposite
sign** (`design/ieee-reexamination.md`, 2026-07-19): at #632 the tool's
imprecision **hid** a float divergence (accidental protection); on the
hit386 control under `ieee` it **manufactured** one — a discarded
component's `0·∞` NaN possibility, itself created by dependency loss on
a declared coordinate, spread through array co-location and blocked a
float-clean assertion. Both directions, same root: **precision and
semantics are not independent axes**, so a precision change is a
semantics-relevant change and vice versa.
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

## L11 — A harness choice can conceal a tool gap; disclosure is not sizing

**(a)** Permitted hand assistance in a harness (a reformulation, a
substitution, a conformant rewrite) can be **silently doing load-bearing
work that the tool cannot do** — not simplifying presentation but
supplying precision or coverage. Disclosing the choice is necessary and
**not sufficient**: the disclosure records *that* a hand step happened,
never *what capability it stood in for*. **Every disclosed hand step
should be priced: re-run without it and record what the tool alone
returns.** The gap that appears is a censusable tool finding; **no gap on
the re-run means the step bought nothing ON THAT CASE**, which is a
reading only because the re-run is a re-execution of a named case rather
than a search. *This read "the absence of a gap is evidence the step was
cosmetic", stated as a general convention — and stated that way it
licenses the whole class L23 forbids. The narrow form is what the
mechanism supports: one case, re-executed, with both answers recorded.*
**(b)** F1's `a²/a` cancellation (`design/regional-obligation.md`,
Part A) was recorded as one of "two disclosed derivations" and read as
convenience — measured, the uncancelled code-shaped form is **UNKNOWN
over F1's own safe region**, so the cancellation was the precision device
that made the obligation interval-provable, standing in for correlation
handling the interval domain lacks. Earlier witness, same shape: hit386's
harness was conformed from `jnp.stack` to `jnp.array` to match the
censused primitive set — a harness choice that stood in for a missing
`stack` registry row (recorded then as "the coverage discipline
working", which it was, but the substitution's *size* went unpriced).
**(c)** **Convention** — price each disclosed hand step by re-running
without it; report the delta beside the verdict.
**(d)** Every harness with hand assistance; the E2a permitted-assistance
list; any usefulness claim resting on a hand-derived form; and clause (i)
of the private-track criterion, which exists for exactly this failure.

## L12 — Sweep the class, don't patch the instance

**(a)** A guard added at one transfer or emission site **triggers a
mandatory sweep of every sibling site before the fix is called done.**
The invariant-at-one-consumer failure has now appeared four times, which
is past pattern; the sweep is the **default step in any guard-shaped
fix**, not something to remember. The sweep's output is recorded even
when it finds nothing: "swept N sibling sites, M needed the guard,
K cleared with reasons" — a silent sweep is indistinguishable from no
sweep.
**(b)** Four witnesses: guards-generate-hazards (L5); the narrated stamp;
FTZ modelled per-band but not per-dtype (L10); and the decisive one —
the integer emission guard, where the audit reported **one** unguarded
site (`integer_pow`) and the commissioned sweep found **four more**
(`add`, `sub`, `mul`, `neg`). Patching the reported instance alone would
have left the auditor's own false VERIFIED intact, minted through the
`mul`. The sweep was the difference between fixing a finding and closing
a class.
**(c)** **Convention → standing step in the fix protocol** (alongside the
structuralization question and the auditor re-attack).
**(d)** Every guard-shaped fix; every audit finding of the form "site X
lacks the check that site Y has."

> **Fifth witness, and the sharpest — the rule recurred inside the fix
> for the rule.** The integer-guard fix swept the *emission* sites and
> not the *transfer* sites, leaving `div` unguarded there; the re-attack
> found six more false definite verdicts through it (3 VERIFIED, 3
> REFUTED, including the INT_MIN/−1 wrap that UNSOUND 1 was about). The
> cleared-list entry read "div (already stricter)", true of the emission
> and false of the transfer. **So the sweep must name its axis:** a sweep
> over one layer's sites is not a sweep of the class when the same
> invariant is enforced at two layers. Record which layer was swept, and
> sweep each layer that can mint a verdict independently — interval
> propagation decides without ever reaching the emission, which is what
> made the original finding unsound in the first place.

> **Sixth witness — the instrument itself (2026-07-19,
> `design/portability-pass.md`).** The ⊤-widening vacuity transform was
> convention-copied into three corpus harnesses, and the copies had
> already forked once (all-declarations vs inputs-only). The instrument
> that polices every count was maintained by the convention this entry
> exists to forbid. Extracted to `stelling/vacuity.py` — one
> implementation, both **registered** procedures as required modes, the
> byte-identical gate proved by matching query content hashes at every
> call site. The unregistered finite-⊤ criterion was deliberately NOT
> encoded: a counting rule is a registration, and code would
> misrepresent its status.

> **The generalisation the three rounds earned, in the builder's own
> words: _"the escape route was an enumeration every time."_** Each of
> the three UNSOUND findings in this pass had the same shape — a fix
> correct for the case that taught it and blind to a sibling — and each
> was closed only when the fix stopped enumerating members and started
> enforcing the property: a **total census** (every registered primitive
> classified or the assert trips), then a **behavioural assert** (each
> computing transfer is *run* on an out-of-range value, because a marker
> attribute is a label and labels can be worn), then a **taint that
> recognises no shapes at all** (a pattern match on jaxpr syntax is a bet
> that the shape survives compilation; a taint is not). **A guard-shaped
> fix that names its cases is unfinished — the test is whether an
> unenumerated instance is caught.** Write that test: the taint fix was
> accepted only after chains nobody had listed (`abs→reshape→transpose→
> max→add`, `array+slice`, `where/select`) were confirmed caught.

## L13 — Stay sound without going useless

**(a)** When more than one *sound* option exists, prefer the one that
**preserves the tool's ability to decide**. Soundness-by-refusal is
always available and always cheap, and under audit pressure it is the
path of least resistance — a tool that declines everything is
unimpeachable and worthless. **This governs the choice among sound
options only; it is never a licence to trade soundness for utility, and
it may not be cited to relax a guard, weaken a criterion, or admit an
unmodelled behaviour.** The test: does the precision-preserving option
have a soundness argument of its own? If not, it is not one of the
options.
**(b)** Five instances now: ieee mode **models** float rather than
refusing it; the subnormal haze makes band outcomes indeterminate rather
than declining the whole mode; constraining `assume` **narrows** where
provably sound rather than always dropping; the solver escalates rather
than leaving every straddle UNKNOWN; and the integer **reachability**
guard — exact where the result provably fits, ⊤ only where wraparound is
genuinely reachable — so `i*i > 0` for an index in `{1,4}` still
discharges where a blanket integer decline would have blinded the tool to
every index and counter in the corpus.
**(c)** **Convention** — a named design principle, stated in the record
whenever a guard or decline is chosen, with its own soundness argument.
**(d)** Every guard, decline, or domain-scope decision; every audit fix
round, where refusal is the tempting default.

## L14 — Undiscovered defects live where attention has not been directed

**(a)** A surface that no build has exercised has not been audited, no
matter how many audits the project has run — audits inherit the
attention gradient of the builds that preceded them (L6, one level up).
So **a build entering a never-exercised surface carries an explicit
first-contact instruction: audit this surface as if it predates us,
because it probably does.** The corollary is a prediction: the next
undiscovered defect is on whatever surface the next build first
exercises.
**(b)** The oldest false VERIFIED in the project — integer arithmetic
modelled as unbounded reals, live since the MVP — survived **six** prior
audits, not because they were weak but because nothing had ever pushed
integer arithmetic to its overflow boundary. It took a build that
*needed* integer arithmetic (`integer_pow`, `reduce_sum`) to route
attention there, and it was found immediately once attention arrived.
**(c)** **Convention** — a standing clause in builder and auditor specs
whenever a build touches a new primitive class.
**(d)** Every build touching a primitive class, dtype family, or
execution surface the corpus has not previously exercised.

> **Scope note (2026-07-19, from the precondition-class pass):** the
> risk tracks new *primitive/execution* surfaces, **not new codebases
> per se**. First contact with the magnetics codebase was clean — posed
> end-to-end, no crash, no registry gap — because the precondition class
> deliberately stays on the well-worn scalar core: its boundary
> ("input-side, mask-free") is also an attention-surface boundary. A
> class designed to run on exercised machinery can meet a new codebase
> without paying first-contact tax; the tax returns the moment an
> obligation needs a new primitive.

> **The prediction confirmed three times in one pass (2026-07-21, the
> array-emission build).** The build carried the first-contact
> instruction explicitly, and the surface still yielded three latent
> defects, each predating the build: (1) the **builder** hit the z3
> model-echo screening defect — every expression-valued `define-fun`
> echo flagged "non-rational", degrading every z3-consulted sat
> witness — latent since the first z3 transport, exposed by the first
> QF_LRA-sat path; (2) the **auditor** found the routing-oracle hole
> (jax-illegal `broadcast_in_dim` dims silently mis-routing) and its
> propagation sibling crashing raw; (3) the **re-attack** found shape
> nonnegativity validated *nowhere* — `any_array((-2,-2), …)` traced
> through the public API and the pipeline coherently minted a
> REFUTED-with-witness over an *empty declared set*. None of the three
> was in the new code's own logic; all three were in what the new code
> first touched. The instruction is not a formality: budget the rounds.

## L15 — A regression test is a comment until it fails against the unfixed code

**(a)** A test written alongside a fix passes for *some* reason; only a
**counterfactual run — the fix disabled, the test failing** — shows it
passes for *the* reason. Same for a claim in a report: an edit *made* is
not an edit *landed*. **Verify by removal, never by assertion.** This is
L11's move (price a hand step by re-running without it) applied to the
project's own evidence: the counterfactual is the evidence, and without
it a regression test is a comment with a green tick.
**(b)** Two witnesses in one pass (fix rounds 3–4 of the three-rows
pass; the `design/three-rows-*` documents this entry originally cited
were never committed — the surviving witnesses are the descriptions
below and the regression tests in `tests/test_three_rows*.py`). (i) The claim "a joined `cond` output is tainted if any branch
tainted it" reached the build report because the edit implementing it was
*made*; the edit had silently no-op'd on an indentation mismatch, and
`branch_taints` sat declared-appended-never-read while every `cond` join
laundered the taint. Caught only when the auditor ran the behaviour.
(ii) The regression test written for that fix **did not discriminate** —
a straddling branch index made the query `unknown` from the branch join,
so it would have passed against the broken code; caught by running it
with the taint laundered and seeing it still pass.
**(c)** **Convention → mandatory step for any soundness fix here:** state
in the report that the test was run against the unfixed code and failed.
Cheap; it is the only thing separating a regression test from a comment.
**(d)** Every soundness fix; every regression test accompanying an
UNSOUND/FRAGILE finding; every claim in a build report about behaviour
rather than intent.

## L16 — Readiness is evidence, not argument

**(a)** The "arguments from me, facts from the probe" discipline applies
to **the project's own readiness claims**, not only to verdicts about
corpus targets. A load-bearing claim about stelling itself — safe
construction, usefulness, a capability's non-usefulness — deserves a
registered probe exactly as a claim about diffrax does.
**(b)** All three readiness claims of the CI pass were asserted and all
three were corrected by measurement (`design/ci-readiness.md`):
safe-construction was **false** (the funnel falsified by enumeration);
usefulness was unmeasured in the CI mode (now demonstrated, blind, on an
external repo); affine/LA-not-useful was **contradicted** (the D-clamp
false-alarm cause is an LA-shaped theorem; the magnetics conditioning
scar is LA-shaped with a measured failure).
**(c)** **Convention** — readiness claims get registrations and bands
like any reading.
**(d)** Every "is it ready / is it useful / do we need X" decision.

## L17 — Completeness claims are verified against the language's affordances

**(a)** "All construction routes through the gate" is a completeness
claim, and completeness claims fail where the enumeration missed a path
the *language* affords, not a path the author thought of. Verify against
what the language actually permits (public dataclasses are freely
constructible; `object.__new__` exists; dicts are just JSON) — and the
structural fix is to make the invariant hold **at construction**, not at
a chosen chokepoint the author believes everything passes through.
**(b)** The construction census: the funnel claim was false because `ir`
is a public module of frozen dataclasses — direct construction (I1's
route) was reachable and ungated; fixed by `__post_init__` validation
(every path, guaranteed by the language itself). Same family: the census
undercount, the sibling-site blindness, `_INT_GUARDED_INSIDE`'s
wearable marker.
**(c)** **Convention** (completeness claims name their verification
method against the language) producing **structural** fixes.
**(d)** Every funnel/coverage/completeness claim; every "all paths go
through X" assertion.

## L18 — Duty-enforced protections do not survive the CI transition

**(a)** A soundness protection enforced by discipline (an operator
re-running a control, a human replaying a witness) protects only while a
disciplined human is in the loop. CI removes the human. **The
CI-readiness question is: which backstops are still manual — and each
must go structural before unattended trust.** The failure mode is the
worst kind: silent wrong passes with nobody watching.
**(b)** The vacuity control: tautology-shaped VERIFIEDs in the first
field test were caught **only by registered duty**; `check()` — the
entry point CI would call — did not run `widen()`. Now wired: `check()`
cannot return an unchecked VERIFIED, with the mode explicit (no silent
mode, no silent skip). Next in the same class: the recorded-set no-flip
gate (automatable as a CI job).
**(c)** **Convention** (the standing CI question) producing
**structural** wiring per instance.
**(d)** Every manual verification duty; every future "run it in CI"
step.

## L19 — A finding is a conjunction; the pose mechanizes only half of it

**(a)** A defect claim is *violated ∧ silently-consequential*. A local
obligation mechanizes "violated" (the flagged arithmetic is real); the
consequence half is usually asserted by framing, and it is where false
positives live. Four measured failure modes of the unmechanized
conjunct: **(1)** a downstream framework postcondition catches the
consequence (the local hazard is the detection mechanism); **(2)** the
value is tracer-capable, so eager validation is impossible and its
absence excused — conversely, static-only values (Python `int`/`None`
in Python-level branching) have no tracer excuse, and that is where
real config defects live; **(3)** the violation is harmless (every path
still computes a valid result); **(4)** the failure is loud (trace-time
crash, flagged breakdown). **Gate-grade requires both conjuncts
established at the pose's own locality; anything else is born
triage-grade.** Calibration, not suppression. This is L2's
conjunctive-claim lesson recurring one level up — at the finding, not
the verdict.
**(b)** The lineax out-of-sample verification
(`design/ci-readiness.md`): 1 of 8 findings real; all seven
dissolutions land in the four causes above (each premise re-verified
against source/execution before encoding); the one real defect
(`max_steps` equality flag bug) survives all four questions — the
calibrated rules downgrade the seven and keep the one.
**(c)** **Convention** (the four questions in `docs/preconditions.md`'s
gate-or-triage section) + one candidate **structural** upgrade recorded:
the traced-vs-static sort is derivable from the trace (a poser build,
with audit, when taken).
**(d)** Every REFUTED adjudication; the CI triage step; any future
"unguided sweep" protocol; and the precision expectations of every
new-codebase field test.
**Meta:** the honest out-of-sample number existed only because a
stake-free blind verifier hunted for the caller contract that unmakes
each finding — the distinct-context audit discipline, applied to
findings. An in-sample rate reported by the rule-maker is a lower bound
on self-deception, not a measurement.

## L20 — A stamped disclosure must be conditioned on the act it reports

**(a)** A record line synthesized from control flow ("the widen re-check
ran, and everything still discharged") will eventually report an act
that never happened. An instrument that can be *inert* on a query class
must detect its own inertness and stamp it as inertness — a disclosure
derived from reaching a code path is a claim about the path, not about
the measurement. The failure is the worst shape for this project: a
**true verdict carrying a false stamped line**, i.e. the auditable
record lying while the answer stands.
**(b)** The vacuity instrument under `mode="inputs-only"` on an
all-point envelope: point declarations hold still by design, the
"widened" query is byte-identical, the re-run proves nothing — yet the
stamp claimed "discharge(s) with the declared bounds widened to
(-inf, inf) — the verdict does not depend on the declared envelope",
directly contradicted by `mode="all"` on the identical query
("load-bearing"). Found by the LA contract layer's first-contact audit
(2026-07-21) because point envelopes are first-class contract use
(the probe's real mesh matrix); pre-existing in `check()` since the
vacuity wiring. Same audit, same class: the widen re-check's solver
invocations were relied on by the vacuity line but absent from the
stamp (10 spawned, 2 recorded). Both fixed structurally: the identical
widened query is detected and the re-run skipped with an inert line
stamped; re-check invocations are stamped with a distinguishing tag.
**(c)** **Structural** (identity check + inert line; tagged
invocations). The convention behind it: before stamping "X was checked
by doing Y", verify Y was *distinguishable from doing nothing*.
**(d)** Every stamped line that reports an action (solver invocations,
widen re-checks, replay confirmations, coverage claims); every future
instrument with a degenerate query class on which it is a no-op.

## L21 — A fidelity check pins only what it can distinguish

**(a)** A check that binds a transcription to the real code ("bit-identical
output on random inputs") licenses claims only up to its own
discriminating power — and that power is capped silently by absorbed
parameters (a value the float arithmetic swallows) and by degenerate
instances (a symmetric case on which wrong code computes right values).
The check's PASSING says nothing about either cap; only deliberately
wrong variants run through the same gates measure where the caps are.
This is L15's counterfactual discipline applied one level up: not
"would this test fail on the unfixed code" but "would this instrument
notice the error class at all."
**(b)** The LA attachment's round-2 audit (2026-07-21): a 10× wrong
`reg` passed all three original fidelity gates — absorbed in binary64
in the boundary-fed configuration (`0.3125 + 1e-29 == 0.3125`),
invisible in the boundary-starved one because the rhs is exactly
nullspace-aligned (the check observes only the first column of M⁻¹) —
while the script was protected in fact only by an unrelated
signature-read assert. Separately, on the 2×1 mesh every cell stencil
is congruent, so cell-permutation/transpose/sign mutations produced
value-identical assemblies no value comparison can see. Fixed by
layering (a starved-config exact M cross-check — which now catches the
reg mutation — and a non-congruent 3×2 fidelity mesh) and by naming
the residual value-identical class algebraically, member by member.
The layered stack's power is now itself measured: the mutation battery
runs every gate against twelve wrong assemblies and asserts who
catches what.
**(c)** **Structural**: the mutation battery is part of the record;
"what would this gate miss" is answered by measurement, not assurance.
**(d)** Every fidelity check binding a harness to real code; witness
validators; any future "bit-identical"/"reproduces exactly" claim; the
adoption pattern — a stranger copying an attachment must copy the
mutation battery, not just the happy-path check.

## L22 — A classification is a soundness claim: census it, probe-or-exempt

**(a)** A census that derives its support sets from a hand-filed
classification cannot catch misclassification — the totality asserts
verify that every registered name is *somewhere*, not that it is filed
*truthfully*. Filing is a soundness claim like any other: it must be
behaviorally probed where a probe exists, or carry an explicit written
exemption reason where it does not, with the probe-or-exempt condition
itself asserted at import. Otherwise the census's strength is an
illusion that holds only until the first wrong filing.
**(b)** The scatter-add row audit (2026-07-22): a scratch two-edit
future misfiling — an arithmetic primitive given a transfer, classified
`_INT_NON_COMPUTING`, plus the ieee row the totality assert forces —
passed EVERY import-time census assert (the behavioral integer boundary
sweep probes only `_INT_COMPUTING` names) and minted a false VERIFIED
on an int32-wrapping cumsum. The builder had independently self-reported
the benign half of the same pattern (structural-set membership deriving
emission support). Fixed structurally at both layers: probe-or-exempt
registries with a written reason per exemption, a stale-exemption
refusal (the fidelity module's stale-residual discipline, reused), and
an assert that raises on any registered-but-unprobed-unexempted name.
Every current primitive earned an honest written reason; the silent
two-edit path is now a conscious three-edit act whose third edit is a
soundness claim in a censused registry.
**(c)** **Structural** (the import-time assert + the reason
registries). The convention behind it: wherever a set membership
*implies* a soundness property, the membership claim needs its own
census.
**(d)** Every classification registry that feeds a soundness decision
(integer computing/non-computing, structural/arithmetic, ieee
category); every future row addition; any "derived set" whose
derivation launders an unchecked claim into an asserted-looking one.

## L23 — An instrument's silence is a reading only if it could have spoken

**(a)** Every check has an *enabling condition* — the environment that
provisions it, the lane that runs it, the pool it draws from, the
domain it enumerates — and that condition is almost never asserted. So
"this check did not fire" is read as "there was nothing to find", when
the two are distinguishable only by measuring the condition itself. The
remedy is structural and always the same shape: **declare the enabling
condition, measure it against the machine that would have to deliver
it, and fail on the difference** — never against a second list, because
two lists agree until the day they do not and nothing notices when they
stop. Where the condition is an enumeration, prefer the complement of
an allow-list to a deny-list: a deny-list is silent about whatever
arrives next.

**(b)** B8b (2026-08-20) found the same shape at five altitudes in one
tree, and none of the five was a wrong answer — all five were
instruments that could not fire. *A configuration:* the environment
`pip install -e ".[jax]" --group dev` produces had no whole-suite lane,
so **72 tests were failing in it** with nothing red, every one a test
that needs a solver and never declared it, plus five skips whose reason
strings no rule in the skip inventory carried. *A lane:*
`EXPECTED_HASH_COVERAGE` recomputed doc-hash coverage over
`TESTED_JAX_SERIES` while its own failure text said *"compared on NO
tested jax LANE"* — driven forward to the day jax 0.12 ships, the
inventory stays green while the hash it certifies is compared on no
lane at all. *A search:* the property suite's `ci` profile claimed the
same tree gives the same examples; hypothesis mixes in constants
harvested from whatever local modules are in `sys.modules`, and **one
ordinary module co-collected and fully deselected flips a strict xfail
to XPASS**. *Process state:* a test left jax's const-fold rule replaced
and an unrelated exit-code battery went vacuous for two audit rounds.
*Order:* `pytest-randomly` was installed in no venv and every lane ran
plain `pytest`, so order-dependent pollution was invisible by
construction — which is why the previous item survived.

**(c)** **Structural**, and one mechanism carried three of the five:
`tests/_lanes.py` reads `ci.yml` and every claim that depends on a lane
is checked against the lane that delivers it (the tested-series tuple,
the doc-hash inventory, the supported install matrix, the shuffled
lane). The other two are the same principle at a scope no workflow can
reach: `tests/_state_guard.py` declares the process-global state and
fingerprints it around every test, and `_profiles.pin_local_constant_pool`
takes the session's import set out of the search. What does **not**
generalise is a single code object for all five — the altitudes differ
— and each instrument ships its own "what this does not watch".

**(d)** Every coverage claim keyed on a constant rather than on a job;
every skip, xfail or escape whose condition is wider than the thing it
excuses; every enumerated inventory that stands for an open domain;
any future statement of the form "no lane/test/seed found X".

**(e) THE SHAPES THIS RULE GOVERNS IN `design/` ITSELF, NAMED AFTER A
SWEEP** (2026-08-24). Every build record and pass record in this
directory was read for the shape, and it recurs in four spellings. They
are named here rather than rewritten one by one, because a dated pass
record is a record of what a round measured and rewriting it destroys the
thing it exists to keep — **but a record's HEADING and its SUMMARY
SENTENCE are live prose, and those were corrected where they converted a
null into a conclusion.** The four:

1. ***"N findings, zero UNSOUND"*, and worse, *"Nth consecutive
   zero-UNSOUND round"***. The count of findings is a reading; the
   absence of one class among them is a fact about the round, and a
   streak is a fact about the rounds. Corrected in `roadmap.md`,
   `affine-refinement.md` and `scatter-rows.md`, where it had reached a
   heading or a living list. **The sharpest counter-example is in the
   affine round itself: at the moment the streak reached four, that same
   round demonstrated four unsound mutation classes surviving every
   shipped gate.**
2. ***"X survived every attack" / "X held" / "no route to Y was
   reachable"***. Each is the attack set's bound wearing the subject's
   clothes. Corrected in `soundness-audit.md`, `constraining-assume.md`,
   `la-contract-build.md` and `solver-integration-build.md`.
3. ***"Suites: N passed" under a heading reading "Verification" or
   "Gates, verified"***. Left standing as pass records, because a suite
   count is a fact and the heading is the load-bearing part; the general
   correction is here rather than in each file. **A passing suite is not
   a verification of anything but the suite** — the property suite's own
   README says the per-push profile is *"a rot detector, not a defect
   finder"*, and that is the reading to carry to all of them.
4. ***An existential negative from a search*** — "no tool exists", "in
   existence", "in the wild", "nothing anywhere". Corrected in
   `d4-wrap-disclosure.md`, `eager-truncation-detector.md`,
   `jax-verification-categories.md`, `tracker-probe-2.md` and
   `registration-rules-and-capacity.md`. This is the spelling that
   travels furthest from its evidence, because the sentence that results
   is short, quotable, and carries no scope.

## L24 — A branch is measured against its parent; the tree that ships is the merge

**(a)** A branch measured green against its own parent has made a
statement about that parent. **The tree that gets tagged is the merge, and
no branch is that tree.** A defect can be *true on every branch
separately and false on the merge*, and when it is, **no branch can fix
it**: the guard arrives on one side, the sentence it refutes on the other,
and each side is green alone. Textual cleanliness is not evidence — the
merges that carried these were clean, and several involved a file that
exists on one side only, so there was no conflict to notice. The merge is
therefore its own object of study: it must be built first, measured
whole, and audited by a context that authored none of its branches.

**(b)** Four in the 0.2.0 release campaign, from four different
mechanisms — §1.4's responsibility table, where two branches each rewrote
a different pair of rows correctly and the union was a hand edit neither
could make; §2.6's caveat, which wrote a field-and-value pair *while
denying that the pair exists*, with the gate that refuses that
construction arriving on the other branch; `tests/test_skip_inventory.py`,
where two new predicates whose names differ by one word made a union look
like a replacement; and
`tests/test_changelog_names_the_version.py`, which illustrated its own
subject in the present tense and had the version bump on the other line
move it to the other of the two states that file checks.

**AND A FIFTH, IN 0.2.1, WHICH IS THE ONE WORTH READING** because it was
produced by two branches whose own subjects are L25 and L28, in the
release that lands this entry. One branch replaced the state guard's
syntactic scan with a behavioural check and **deleted**
`test_the_declared_limits_are_still_limits`, leaving the history block
that names it. The other widened `tests/test_prose_hygiene.py`'s bare-name
check **into `tests/`**. Each was green alone and neither could have seen
it: on the first branch `tests/` was still scoped out of that check, and
on the second the test still existed. The merge is red, by name and by
line. Closed by a hand edit in the merge commit — the first entry
`_NAMES_DECLARED_ABSENT` has ever earned on a merge rather than on a
branch.

**(c)** **Convention, promoted to structural where it can be.** The
integration tree is built before anything is tagged, measured across the
whole matrix *on the merge*, and handed to a blinded auditor at a stated
sha with the merge's own hand edits named as things to verify. The
structural half is that every hand edit is recorded in the merge commit
message, so a later reader sees what was decided rather than inferring it
from a diff.

**(d)** Every release. Every campaign that fans out to more than one
branch and lands them together — which is every campaign this project has
run since 0.1.0. Aim the audit at *the merge*, and give the auditor the
tree rather than a transcript.

## L25 — An exclusion argued for one construction must not be spelled as a directory

**(a)** A scope exclusion earns its place by an argument, and **the
argument licenses exactly what it covers and no more**. Spelled as a path
prefix, it silently excludes every *other* kind of claim in those files
too. The gap is invisible precisely because the exclusion is justified:
a reader checks that the reason is good, finds that it is, and does not
ask whether the reason is as wide as the rule. **The test is to state the
reason and the rule side by side and see whether they have the same
shape.**

**(b)** `tests/` was excluded from `tests/test_prose_hygiene.py`'s
citation checks, and the exclusion was argued in the file: test modules
write citation-shaped SOURCE STRINGS as plants and **are supposed to name
tests that do not exist**. That argument is sound and it is about a KIND
OF STRING. The rule it produced excluded the whole directory, so it also
excluded every *sentence* in `tests/` that makes a claim about the tree —
and that is what let L24's fourth instance ship undetected. Measured at
`9b5b496`: **181 of 539 tracked files**, and 181 of the sdist's 379
members, so the excluded region was neither small nor private.

**(c)** **STRUCTURAL, and landed in 0.2.1.** *This entry read "watch …
until that lands, the exclusion is disclosed rather than assumed narrow";
it landed in the same release, and the sentence is corrected rather than
left standing.* The rule is citation-shaped: in a `.py` file a citation
counts in a comment or a docstring and not in a value, decided off the
same grammar rather than off a list of the ways a string can be data.
**Measured before adopting rather than after** — which is the discipline
the original defect lacked — the uniform rule loses nothing: of the 311
citations the old scan checked, **zero** were in a value literal. The
widening found 8 dangling `path::name` citations and 24 dangling bare
mentions, all of them in `tests/`, five naming tests **no revision of this
tree has ever defined**.

**(d)** Every scope decision in this repository expressed as a path
prefix, and there are several. For each, ask what the *argument* excludes
and whether the *rule* excludes more.

## L26 — A claim in prose that nothing holds will go false, and the tense is part of the claim

**(a)** Prose beside code carries assertions, and **nothing reads prose**.
Three faces, one mechanism — a claim with no mechanical hold on it rots at
the speed of the tree around it:

1. **A figure.** A number a reader cannot re-derive is the same defect as
   a check that does not exist. Either derive it, or say precisely what it
   counts and how to re-take it.
2. **A coordinate.** A line number, a call-site count, a collection rank,
   a section index — these are properties **of a checkout**, not of the
   tree, and they are falsified by the next edit anywhere above them.
3. **A tense.** A present-tense sentence about another file is a standing
   claim about code nobody has undertaken to keep true. A once-taken
   measurement written in the present tense is the commonest way a true
   thing becomes a false thing without anyone editing it.

The three have three different repairs, and picking the wrong one is how
the defect returns: a figure gets **derived**, a coordinate gets
**deleted** unless a test can derive it, and a tense gets **dated** — a
record of what was measured, when, rather than a claim about now.

**(b)** All three, in one campaign. *A figure:* the count of what
`tests/test_documented_names_exist.py` finds RED on `main` at `115d771`,
in the file whose entire subject is a name asserted in prose that the code
does not have. Re-run against that commit it is **eight findings over
seven distinct file:line pairs**, and the gate's two declared exclusions —
a fenced block and an unchecked checkbox — are why a count of *sentences*
comes out higher. **This entry twice carried a larger figure with an
explanation of the gap, and the explanation was wrong both times**; the
figure now stands alone with the sha and the command that re-takes it,
which is this entry's own prescription and should have been the first
repair rather than the third. *A coordinate:* a `_run` helper cited by
line and by call-site count, both already stale when written, and a
collection rank pinned on a shipped page that was a property of the
checkout. *A tense:* `tests/test_propagation_identity.py` diagnosed a real
bug through one importer, deleted that importer, and wrote the consequence
as a standing property of a file it does not own — a second importer then
reintroduced it, in a third file.

**(c)** **Structural, per face.** The figures that survive say what they
count and hand the reader the re-derivation. The coordinates are gone,
replaced by assertions. The tenses are dated records with the original
measurement kept. **And 0.2.1 added the face this entry did not have: a
name.** Nine cross-references in `src/` named three things the tree has
never had — two of them never renames, established by `git log --all -S`
finding no commit that ever defined either.
`tests/test_referenced_names_resolve.py` resolves every Sphinx role in a
`src/` docstring against the parsed package, which is the derived form of
a claim that had been carried by habit.

**(d)** Every docstring and comment that carries a number, a location, a
name, or a present-tense verb about code in another file.

## L27 — A refusal never observed is not known to be a refusal

**(a)** A guard on a rare path is **a program that has never been run**.
Its author reasoned about it; nothing executed it. The whole point of a
release gate is that it fires on the day everything else has gone wrong,
which is the one day nobody wants to discover it was never exercised —
and the failure mode is not that it refuses wrongly but that it **passes
silently**, because the condition it checks was never reachable in the
shape the job actually runs. So: drive every refusal, in both directions,
in the *shape of the real invocation* rather than in a unit test of its
body.

**(b)** `.github/workflows/release.yml` is the only thing between a
hand-typed tag and PyPI, and an sdist on PyPI **cannot be unpublished,
only yanked — and a yanked file still resolves for anyone who pins it**.
Both its `actions/checkout@v4` steps ran at **depth 1**, so the job
skipped checks it exists to run; restoring `fetch-depth: 0` moved ten
nodes from skipped to passed and turned a planted bad commit from
*skipped* back into a *failure*. And the pre-bump refusal had never been
observed at all: building the tree one commit before the version bump and
driving the tag step is what turned *"it would refuse"* into `rc=1`.

**AND THE SHARPEST CASE IS THE ONE 0.2.1 FOUND, WHICH IS NOT A REFUSAL
THAT WENT UNOBSERVED BUT ONE THAT WAS NEVER WRITTEN.**
`tests/test_changelog_names_the_version.py` declined to gate the changelog
heading's date — correctly — and routed the check to `release.yml` by name
and with a command. That workflow did not carry it, and nothing in the
tree read the sentence. Measured: **47 commits** between the routing
paragraph's arrival and the release tag. A routed claim with no instrument
on it cannot go red, only stale — which makes routing prose a third
category beside a refusal that fires and one that does not: **a refusal
that does not exist and is believed to.**

**(c)** **Structural.** The tag forms, the sdist step, the manifest step,
the pre-bump refusal and now the changelog-heading gate are driven on real
artefacts inside the suite; the shallow-clone shape is reproduced with
`git init` and `git fetch --depth=1`; the header's refusal count is
derived from the file rather than typed beside it; and the routing
paragraph is held by a test that reads the workflow.

**(d)** Every workflow whose trigger is rare and whose consequence is
irreversible. Also every `if:` guard added "for safety" that no test has
ever entered — **and nothing in this repository reads `if:` in any
workflow**, which is the largest hole this entry knows about and does not
close: a step-level or job-level `if:` would switch off any refusal point
silently.

## L28 — A check that MODELS a behaviour is always one indirection behind it; measure the behaviour

**(a)** A check written by enumerating the *syntactic forms* a behaviour can
take will be defeated by one more indirection, and the next one, and the
one after — because the set of spellings is open and the behaviour is not.
Each round of enumeration looks like progress and is not: the check gets
longer, the class stays open, and the docstring quietly claims a
completeness the code has never had. **The tell is that every gap found is
one step further out than the last.** When that pattern shows up, the
repair is not another form. It is to stop modelling and **assert on the
outcome**: what the system actually did, which is decidable and immune to
every indirection at once.

**(b)** A guard asserting that no nested in-process session in one test
module loads `tests/_state_guard.py`. Five rounds, five fixes, five
audits, and each audit found a route live in the very file being scanned:
direct `Call` arguments missed a module-level tuple spread as `*ARGS`;
whole-string matching missed a generated conftest; four `make*` wrapper
names missed `makefile`, the primitive all four delegate to; and naming
`makefile` still missed a source constant handed in through a variable and
a generated conftest that imports and registers. Two of the five gaps were
the scanned file's own prevailing idiom.

**AND THE LAST ROUND IS THE ARGUMENT, BECAUSE THE SCAN COULD NOT TELL A
ROUTE FROM A CRASH.** *This entry recorded both remaining routes as
"Measured to load the plugin". That was false of one of them.* The
import-and-register route was pinned as source spelled
`pytest_configure(c)`; pluggy refuses that hookimpl's argument name against
the hookspec, the inner session dies during conftest loading, and nothing
registers anything. The route reaches only when spelled `config`. Nothing
noticed, because the pin only ever asked whether the SCAN COULD SEE THE
SOURCE — a guard that checks a claim is well-formed and never that it is
true. A syntactic scan cannot distinguish a spelling that loads the plugin
from one that kills the session before it could, because it runs neither.

**(c)** **STRUCTURAL, and landed in 0.2.1.** *This entry read "Convention,
with the instance resolved by declaration rather than by a sixth patch",
and recorded the behavioural replacement as work rather than done. It is
done, in the same release, and the sentence is corrected rather than left
standing.* The question is decided by running the program: a plugin named
to every session in the observed process tree reports what its plugin
manager registered, matching on the registered object's resolved
`__file__` rather than on the name it was registered under. Both routes
the scan declared unreachable are driven as firing tests — the second in
its corrected spelling — and the declared limits and their fixture are
struck, which is what that fixture's own failure message said to do when a
limit is outgrown. The by-construction argument that made the deferral
defensible was re-verified rather than re-read: `tests/_state_guard.py`
binds no mutable container at module scope at all, so a nested session
building a fresh `Config` is separated whether or not the plugin loads.

**(d)** Every check in this repository that reads SOURCE to decide a
question about BEHAVIOUR, and there are several: the workflow readers in
`tests/_lanes.py` and `tests/test_release_gates.py`, the citation and
construction scans in `tests/test_prose_hygiene.py` and
`tests/test_documented_names_exist.py`, the Sphinx-role resolver in
`tests/test_referenced_names_resolve.py`, and the tripwire record's reader
over the nightly canary — the one that already learned this expensively,
having gone past nine legal spellings of one `env:` mapping. For each: ask
whether the question is really about the text, and if it is about what the
program does, whether it can be decided by running it instead. Where it
cannot, the limits go in the docstring **and get a fixture** — and the
fixture must assert the limit's SUBJECT, not merely that the scan cannot
see it.
