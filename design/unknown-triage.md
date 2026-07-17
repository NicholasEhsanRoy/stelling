# The UNKNOWN triage and the solver trigger — registered before any UNKNOWN exists

**Status:** REGISTRATION, 2026-07-18. Filed before any of the twelve E2a
cases has run and therefore before any UNKNOWN exists to fit buckets to.
The MVP discharged every obligation on the positive control with interval
arithmetic; nobody knows whether the twelve will. **The UNKNOWNs are the
solver census**, and they come free with running the cases.

## The solver design is done and is not the question

Recorded so it stops being re-opened: SMT-LIB2 text over a transport
(never bindings), a four-method interface, portfolio dispatch with
disagreement as a free bug oracle, cvc5 default (transcendentals, proofs,
CAD coverings) with Z3 second (speed on the integer fragment,
cross-check), never on defaults (SOUNDNESS.md), routed by fragment
because cvc5's coverings and nl-ext are mutually exclusive. None of that
needs revisiting. **The only open question was ever *when*, and the
answer is the discipline that built the MVP: not until a target demands
it.**

## The triage — every UNKNOWN from the twelve lands in exactly one bucket

| bucket | shape | fix |
|---|---|---|
| **dependency-shaped** | a variable occurs more than once in a face expression; intervals lose the correlation | **affine/zonotope forms** — in plan. The precision probe's 1.0× on hit386 is *not* evidence against them: that structure was separately monotone with single occurrences. Lucky, not typical |
| **search-shaped** | the obligation needs case analysis or a nonlinear decision procedure | **a solver** — the only bucket that justifies one |
| **harness** | the box is wrong or too loose | work item |
| **coverage** | something fell to ⊤ | registry item, **by census** — never by guessing |
| **totally-false** | the property is definitely false over the **whole** propagated set | surfaces as **REFUTED**; an UNKNOWN here would be a checker defect |
| **partially-false** | the property is false on **part** of the declared box and true elsewhere | surfaces as **UNKNOWN, correctly** — the straddle is the honest answer; the *property* is false somewhere, the checker is **not** defective. A real finding: the box is not invariant. (bjx#D416) |

An UNKNOWN does not mean a solver would help. It means intervals were too
coarse, and the bucket says why.

**Amendment (2026-07-18): the `false` row is split.** It read "the property
doesn't hold → REFUTED not UNKNOWN; an entry here is a checker defect."
That was the same double-duty error that produced REFUTED, one level down,
and worse: a **partial** violation (false on part of the box, true
elsewhere) correctly straddles to UNKNOWN — the checker is fine, the
property is false somewhere. bjx#D416 is exactly this and would have read
as a checker defect under the old row. Against the amendment rule
(`design/e2a-registration.md`): **(a) additive** — adds the total/partial
distinction, amendment 1's shape, no existing meaning lost (totally-false
→ REFUTED unchanged); **(b) count-neutral** — D416 was never bucketed
dependency-shaped, the count stays 1, no trigger moves; **(c) found by the
registration's own control** — the face-expression rule ("the multiple
occurrence must be pointable-at, not asserted") is what showed
`log_eps − 0.1·h_stat` has no repeated variable, so the UNKNOWN is a real
violation, not imprecision. First amendment since amendment 1 to satisfy
(c) cleanly, and the first time one of these procedural controls caught an
**error** rather than a result.

## The registered trigger

> **The solver gets built iff search-shaped UNKNOWNs appear across ≥ 2 of
> the twelve.** One is an anecdote about one system. Zero means the
> Z3-vs-cvc5 question is moot for this product — a finding worth more
> than the integration, and reported as one.

Ordering note, stated in advance so the outcome can't redraw it:
**affine forms are likelier to be the next build than a solver.** The
same trigger discipline applies to them: dependency-shaped UNKNOWNs
across ≥ 2 of the twelve, else they stay in plan and out of tree.

**Precondition on the affine trigger — IEEE semantics first
(2026-07-18, from the dfx#632 exhibit).** On an ℝ-vacuous property a
*looser* interval domain is *more* protective: the tool did not
false-green `t + dt > t` only because its brackets (~2 ulp wide) straddle
exactly where float (~0.5 ulp) cannot distinguish
(`corpus/supply/exhibit_632.py`). That is not float-awareness — it is the
tool's imprecision coinciding with float's, by accident, and the
protection is undeclared. **Affine forms are tighter; tightening closes
the straddle and the false green appears.** So affine forms and a working
IEEE/FP-exact semantics are **coupled**: if the affine trigger ever
fires, it fires with the precondition attached — **IEEE-exact semantics
(or a per-obligation FP-exact fragment) ships first, or not at all.**
Shipping affine over ℝ-only arithmetic converts accidental UNKNOWNs into
false VERIFIEDs.

Bucket assignments are recorded per-case in the E2a readings, with the
face expression quoted for dependency-shaped claims (the multiple
occurrence must be pointable-at, not asserted).

## Defect and forward fix: triggers need a ≥2-sources clause (2026-07-18)

Every band in this project carries a **≥ 2-sources clause** — concentration
in one project is a fact about that project, load-bearing in the tracker,
provenance, and E2a bands. **The solver trigger above omits it** ("≥ 2 of
the twelve" counts cases, not libraries), and the affine ordering note
mirrored the omission. This is a real registration defect, found by review.

**It is not repaired retroactively.** Against the amendment rule
(`design/e2a-registration.md`): (a) it is *restrictive*, not additive;
(b) it would *un-fire* a trigger, not stay count-neutral; (c) it was found
by review, not by the registration's own control. Fails all three.
Tightening a threshold after the fact is exactly the move the rule stops,
and it stays true when the author is the one who wrote the omission. **The
solver and affine triggers stand as written** (both are unfired anyway, so
nothing turns on it here).

**Forward fix, binding the next trigger, not this one:** every future
trigger carries a **≥ 2-sources clause by default** — search/dependency-
shaped UNKNOWNs across ≥ 2 *distinct libraries*, not merely ≥ 2 cases.
Same shape as the property-test corrections: forward-looking. And any
trigger result is reported with its source spread in the same sentence,
the relation-breakdown move applied to a build decision. (Recorded fact:
the one dependency-shaped counting sighting is bjx#969, blackjax; adding
the out-of-denominator dfx#632 exhibit makes the two sightings span
blackjax + diffrax — so no single-source concentration obtains.)

## Reading (2026-07-18, after the E2a run — `design/e2a-run.md`)

**The trigger had almost no data.** Zero search-shaped UNKNOWNs, so the
solver did not fire — but 6 of 7 hits were **unposeable** and never
reached an UNKNOWN, so this is not "the solver is moot," it is "the
failure was upstream of the triage." The one clean survivor (dfx#417)
verified. What stands: the dominant bottleneck was **unposeability**, not
decidability, and no solver/domain/proof-format addresses that. The
Z3-vs-cvc5 architecture is untested, not refuted. If control flow lands
(`design/control-flow-hypothesis.md`), the prediction is that accept/
reject `cond` predicates straddle and the search-shaped UNKNOWNs finally
appear — the solver's first *real* evidence, in either direction.

## The triage has no instrument — sampling is it (trigger registered, not built)

bjx#D416 was bucketed **by hand**: someone found `h_stat = +5` and showed
the violation. It worked, but a hand-bucketed triage is bucketed by
someone with a stake in the answer. Every other control here got an
instrument — the census got the coverage counter, the tracker got
registered terms, E2a got criterion (i) mechanized. The triage got buckets
and an analyst.

The instrument is on the founding feature list, unbuilt: the **fuzzer
fallback** — on UNKNOWN, sample the declared region (it knows the `assume`
constraints, which is why it beats a naïve fuzzer). A violating draw turns
"I can't tell" into **"your property is false, here"** with a concrete
point — the **witness** discipline, in check mode. The asymmetry is
already the project's: a violation is definitive, its absence is not, so
sampling can move UNKNOWN → **partially-false with a witness**, and can
**never** move UNKNOWN → VERIFIED. Cheap: the harness is a jax function;
sampling is `any_array` returning a concrete draw instead of a symbolic
declaration, then evaluating — no new transfers, you call the program
rather than propagate over it.

**Registered trigger (carries the §-above ≥ 2-sources clause — the class
fix binding its first trigger):**

> **Sampling is built iff ≥ 2 UNKNOWNs cannot be bucketed without hand
> analysis, across ≥ 2 distinct libraries.**

**Current count: 1 (bjx#D416). It does not fire.**

**Bias, declared:** I want this built — cheap, on the list, produces
witnesses, removes the analyst from the bucketing. Four good reasons, and
**none of them is a trigger.** The discipline that kept affine out of tree
keeps this out of tree; the trigger exists so it can't be argued in later.
