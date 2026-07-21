# The finding conjunction and the enumerated-discharge architecture

**Status:** DESIGN RECORD, 2026-07-21. Recorded before it is built — the
discharge enumeration is a sticky commitment (like the license, like the
IR funnel) and is deliberately designed *after* the issue campaign
returns real maintainer feedback; this note exists so the design is
ready to show maintainers and so the reasoning predates the build.
Grounded in the measured out-of-sample data (`design/ci-readiness.md`:
1/8 real on lineax, four dissolution causes) and banked as ledger L19;
this note is the formalization.

## A.1 — The finding conjunction

A finding worth a human's attention is a conjunction:

> **`violated ∧ consequential`**, where
> **`consequential` = `reaches-output ∧ ¬caught ∧ ¬harmless ∧ silent`**

| conjunct | meaning | fails when (measured instance) | mechanizability |
|---|---|---|---|
| `violated` | the precondition does not hold over the declared envelope | — (the tracer class fails *this* conjunct: the check is not missing, the value is unvalidatable) | **mechanized today** — interval/AI propagation + replay-confirmed witness; sound |
| `reaches-output` | the violated quantity flows to something the caller observes | dead variable; used only in a branch the violation makes unreachable | **tractable** — reachability/dataflow on the jaxpr |
| `¬caught` | no downstream framework guard intercepts the bad value | lineax cause 1: `_solve.py:99–113` rewrites nonfinite `successful → singular` and raises | whole-program in general; **tractable as a *declared* invariant** |
| `¬harmless` | the violation changes the result, not just the computation | lineax cause 3: `stabilise_every=0` — both `lax.cond` branches valid | semantic equivalence — **hard; often irreducibly human** |
| `silent` | the bad outcome arrives without signaling failure | lineax cause 4: `restart<0` traps at trace time; `stagnation_iters=0` returns flagged | trace-crash half tractable; flag half needs framework knowledge |

**The structural account of stelling's imprecision:** stelling
mechanizes conjunct 1 and *asserts* conjuncts 2–5. The measured
out-of-sample false-positive rate **is** the base rate at which
asserting 2–5 is wrong; the four lineax dissolution causes are four of
these conjuncts failing (and the tracer class is `¬violated`). This is
L2's conjunctive-claim lesson recurring at the finding level (L19). It
is not a bug to drive to zero — part of `¬harmless` is irreducible —
but a characterized, partially-addressable property, and every conjunct
stelling learns to mechanize converts a slice of asserted findings into
verified ones.

## A.2 — The enumerated-discharge architecture

stelling proves the violation and **never silently drops it**; the
**author discharges** a proven violation with an enumerated,
tiered-traceability reason that is itself a claim stelling tracks and
increasingly *checks*. Precedents: pytest `xfail` (tracked; a passing
xfail is a signal), Kani `should_panic` (failing *is* correct), Rust
`#[allow(reason=…)]` / `// SAFETY:` (enumerated reason + justification
discipline).

**The enumeration — small, closed, each mapping to a consequence-
conjunct failure or the honest escape hatch. Resist growing it:**

| category | discharges | corroborability |
|---|---|---|
| `CAUGHT_DOWNSTREAM` | `¬caught` fails — a declared framework guard catches it first | **re-checkable once reachability lands**: verify the violated variable actually reaches the declared guard |
| `HARMLESS` | `¬harmless` fails — computation changes, result does not | high-traceability — a correctness claim stelling cannot prove |
| `LOUD_FAILURE` | `silent` fails — the violation causes an intended loud failure | partially re-checkable — stelling can sometimes confirm the trace-crash |
| `CALLER_CONTRACT` | the caller owns the precondition, documented as their responsibility (the `well_posed=True` case) | high-traceability — an API-contract claim; points at the documenting source |
| `ACCEPTED_RISK` | real, reachable, consequential — accepted for now | maximum traceability — the honest "tracked, not fixing yet" hatch; **must never be easy** |

**Tiering principle:** required traceability scales with (i) whether
stelling can *corroborate* the discharge (re-checkable → low) and (ii)
whether the claim bears on *correctness* (bears → high).
`LOUD_FAILURE` stelling-corroborated → marker suffices;
`HARMLESS`/`CALLER_CONTRACT`/`ACCEPTED_RISK` → enumerated reason +
justification + (where applicable) a pointer to the justifying context.

**Anti-rot mechanisms — non-negotiable; without them this is a mute
button and worse than no tool (the `# type: ignore` rot failure):**

1. **Every discharge is counted and stamped.** A report line —
   "N proven, M discharged, breakdown by category." The discharge rate
   is visible; a codebase discharging 90% is honestly not-verified.
2. **Discharges are contradictable, and a contradicted discharge
   surfaces loudly** — the `xpass` insight: if stelling later proves a
   `CAUGHT_DOWNSTREAM` variable does *not* reach the declared guard,
   the discharge is falsified and must signal. Every conjunct stelling
   mechanizes turns the matching category from *trusted* to
   *verified* — the mechanization roadmap and the discharge mechanism
   reinforce each other.
3. **Discharges are site-specific and scoped, never blanket** —
   attached to one obligation at one site (the
   `# type: ignore[specific-code]` lesson); a discharge matching no
   live finding is *stale* and flagged for removal.

**The synthesis:** stelling proves violations soundly and never
silently; the author discharges consequence-judgments as typed,
enumerated, tiered, auditable claims; stelling mechanizes increasingly
much of *checking those claims* rather than replacing them; the
discharge ledger is visible and decaying. This reframes stelling's
identity from "bug finder" to **"violation prover with a
partially-mechanizable, author-declarable consequence layer"** — the
honest identity, the explanation of the false-positive rate, and
genuinely novel (`xfail`/`should_panic`/`SAFETY`-discipline for
numerical precondition verification).

## A.3 — Sequencing (recorded with the design, so it cannot drift)

The discharge enumeration is **not built pre-publish**. It is designed
*after* the issue campaign returns maintainer feedback on which
categories authors actually reach for — the design note doubles the
campaign as taxonomy research and as a conversation-opener ("I'm
designing enumerated discharges — what categories would you use?").
Build order: **reachability conjunct** (foundation; makes
`CAUGHT_DOWNSTREAM` re-checkable) → **issue-campaign feedback** →
**discharge mechanism on that feedback**.
