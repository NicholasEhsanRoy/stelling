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
| **false** | the property doesn't hold over the declared set | must surface as **REFUTED**, not UNKNOWN — an entry here is a checker defect, filed as such |

An UNKNOWN does not mean a solver would help. It means intervals were too
coarse, and the bucket says why.

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
