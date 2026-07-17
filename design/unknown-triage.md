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

Bucket assignments are recorded per-case in the E2a readings, with the
face expression quoted for dependency-shaped claims (the multiple
occurrence must be pointable-at, not asserted).
