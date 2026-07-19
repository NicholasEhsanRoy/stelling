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
escape / UNSOUND-fix re-attacks run*. Current: **3 / 6** —
constraining-assume pass: F1's fix → F7 found; F3's fix → clean; F7's
fix → clean. IEEE pass: U1's fix (the subnormal haze) → U2 found (the
haze was dtype-blind); U2's fix → clean. Three-rows pass: the integer
guard fix → **UNSOUND 3 and 4 found** (`div` unguarded on the transfer
side; ieee falsified by FMA contraction). **All three catches share one
shape** — the fix was correct for the case that taught it and blind to a
sibling case (aggressive-vs-exact boxes; binary64-vs-per-dtype bands;
emission-sites-vs-transfer-sites and reassociation-vs-contraction). The
escapes are **scope errors, not carelessness**, which is why the answer
is a mandatory sweep with its axis named (L12), not more care. Track per
pass; act only on a trend — but note every catch so far has been a real
false verdict. **First evidence of convergence, recorded 2026-07-19:**
within the three-rows pass the re-attack sequence ran 2 → 1 → 0 UNSOUND
across three successive fix rounds, and the round that returned 0 was
the one whose fix had stopped enumerating cases (L12's addendum). So the
metric may be measuring *fix architecture* rather than fix care — a
class-closing fix survives re-attack, an enumerating one does not.
Watch whether that holds on the next pass before believing it.
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
returns.** The gap that appears is a censusable tool finding; the absence
of a gap is evidence the step was cosmetic.
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

## L15 — A regression test is a comment until it fails against the unfixed code

**(a)** A test written alongside a fix passes for *some* reason; only a
**counterfactual run — the fix disabled, the test failing** — shows it
passes for *the* reason. Same for a claim in a report: an edit *made* is
not an edit *landed*. **Verify by removal, never by assertion.** This is
L11's move (price a hand step by re-running without it) applied to the
project's own evidence: the counterfactual is the evidence, and without
it a regression test is a comment with a green tick.
**(b)** Two witnesses in one pass (`design/three-rows-*`, fix rounds
3–4). (i) The claim "a joined `cond` output is tainted if any branch
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
