# The regional obligation — the third framing, registered before it runs

**Status:** REGISTRATION, 2026-07-19, committed **before the harness is
written or run**. MIME pinned `7ce1efb`. A **usefulness test on held-out
MIME code: it never enters E2a**, produces no count, and moves no
standing figure.

## Why this exists

`design/private-track-criterion.md` scored F1 and F2 against the private
criterion and **both failed clause (iii)** — the scar mesh's
`cos ≈ 0.11` *violates* the alignment floor that F1 and F2 both take as a
**precondition**, so neither verdict fires on the failing mesh. Closing
F1→F2 yields a code-level result that is still not a CI check; the
assume-emission build failed on its own gate.

The same application surfaced a third framing, banked then and registered
now: **make the mesh-quality region the *subject* of the obligation
instead of its precondition.** The scar's `cos ≈ 0.11` is *inside* the
region the meshes actually span, so a verdict over that region can fire
exactly where F1/F2 are silent. This is the shape the heat job's **A2
control** already had (an obligation over the node's own validated
regime, not over its safe envelope) — used there as a control, never as a
product.

A premise correction from the same reading rides with this: **the
`cos ≥ 0.71` one-liner does not exist.** `0.71` appears nowhere in the
FVM package and there is no mesh-quality check of any kind, so **at the
pinned ref the scar is caught by nothing.** That makes the regional
obligation the *only* candidate for catching it, not one of two.

## The obligation, and the controlled comparison

Quantity: the over-relaxed orthogonal coefficient
`E_f = |Sf|² / (Sf·d)` (`operators.py:251-253`, introduced by the scar's
own fix `91e95e6`).

**Region (the change), owner-supplied and labelled as such:** the
cylinder mesh reaches `min cos(Sf, d) ≈ 0.11`; the declared cos region is
therefore **`[0.11, 1.0]`** — the range the scar mesh actually spans.
This characterisation is not on disk (it was supplied in the FVM job's
work order and recorded there as owner-supplied).

**Everything else is held identical to F1**, so the *only* difference
between F1 and R1 is the cos region and the absence of the floor as a
precondition: `|Sf| ∈ [0.5, 2.0]`, `|d| ∈ [0.5, 2.0]`, bound `B = 8.0`.
Round numbers, disclosed as choices, inherited unchanged.

**Two posings, paralleling F1/F2 exactly:**

- **R1 — the lemma form** (F1's polar substitution `Sf·d = |Sf||d|cos`
  and the exact `a²/a` cancellation, both previously disclosed):
  `∀ a ∈ [0.5,2], dm ∈ [0.5,2], c ∈ [0.11,1]: a/(dm·c) ≤ 8`.
  **No `assume`.** Parallels F1; scores clause (i) the same way F1 does.
- **R2 — the code's own raw form** (F2's transcription: `Sf`, `d` as
  independent component boxes, the dot product written out), obligation
  `area²/(Sf·d) ≤ 8`, **with the floor neither assumed nor imposed** —
  the region enters through the declarations alone. Parallels F2; scores
  clause (i) the way F2 does.

Both are run interval-only **and** with solver escalation, and both
verdicts are reported.

## Ring discipline (as with F1/F2/F3)

A **fresh-context transcriber** writes the harnesses. It sees **FVM
source only**; it does **not** see the validation results, the
characterisation study, `known_anomalies.yaml`, git history, this
registration's bands, or **any statement of which verdict is wanted**.
The region and the bound travel as **bare numbers**. The scar provenance
is the main agent's and stays there. A transcriber that knows "fires on
the scar" is hoped-for will, with no bad intent, transcribe toward it.

This is a **run, not a build**: the existing tool on a new obligation. No
new transfers, no new domain, no builder/auditor cycle.

## Predictions (pre-committed, before the harness exists)

- **R1 interval-only: UNKNOWN, partially-false shaped.** The quotient
  over the region spans roughly `[0.25, 36.4]` against the bound `8`; it
  is genuinely true at high cos and genuinely false at low cos, so no
  definite verdict is available to interval propagation and **no
  `assume` is present, hence no DROPPED note** — cleanly distinguishable
  from the dependency wall by the tool's own diagnosis.
- **R1 escalated: REFUTED with a witness at low cos.** The fragment is
  QF_NRA; the divisor `dm·c` has interval `[0.055, 2.0]`, which
  definitely excludes 0, so division is emittable and the escalation is
  not declined. The negation is satisfiable (e.g. `a=2, dm=0.5,
  c=0.11 → 36.4 > 8`), so the expected result is **sat**, replayed for
  box membership and violation, yielding a concrete
  mesh-quality counterexample.
- **R2: dependency-wall UNKNOWN.** `Sf·d` over independent component
  boxes straddles 0, so the division crosses zero and widens to ⊤;
  escalation additionally declines on the possibly-zero divisor. Same
  wall F2 hit, reached by a different route (no `assume` is involved
  here at all — this is the dependency problem, not the relational-assume
  problem).

If these hold, the fork the work order names is answered precisely:
**R1 fires without any build but fails clause (i); R2 satisfies clause
(i) but cannot fire.**

## Distinguishing the two UNKNOWNs — mechanical, fixed now

- **dependency-wall**: a `DROPPED`/inert note, or a ⊤ arising from a
  zero-crossing divisor / declined transfer, with the coverage line
  naming it. UNKNOWN is then a **precision** verdict, not a found
  instability.
- **partially-false**: no such note; the obligation's own interval
  straddles the bound with both sides genuinely attained, and the
  escalation (if it runs) returns sat with a replayed witness.

If the diagnosis is genuinely ambiguous, that is a **must-stop**.

## Bands — every row surfaces (this measurement gates the next move)

| verdict | reading | action |
|---|---|---|
| **REFUTED-with-witness, or clean partially-false** | **fires on the scar** — the regional framing scores (iii) where F1/F2 fail; a CI-shaped candidate that needs **no build** | **STOP, surface** |
| **dependency-wall UNKNOWN** | the regional obligation hits the wall — the machinery would be justified by (iii) this time, not by the failed F1→F2 argument; **resurrects assume-emission, re-scoped** | **STOP, surface** |
| **VERIFIED** | contradicts the scar → the transcription or the obligation is mis-specified | **STOP, surface** |
| **ambiguous diagnosis** | the judgement call that separates CI-check from wall | **STOP, surface** |

There is no clean-continue row: the terminal action is **register,
measure, report, stop**. No band is amended after reading.

## Four-clause scoring, to be applied to whatever comes back

Per `design/private-track-criterion.md`: **(i)** code's own form;
**(ii)** CI time; **(iii)** fires on the scar — the clause the regional
framing exists for; **(iv)** recurring value — the argument *to test, not
assume*, is that it re-derives the property against the actual code and
region each run, where a frozen threshold (which does not exist here)
would go stale on code change. Scores recorded for R1 and R2 separately,
since they differ on (i) by construction.

## Non-goals

No CI wiring (a live mesh-statistic pipeline is a separate human
decision — this measures only whether the check *fires*). No build; if
the wall is hit, that surfaces as a re-justified build for a later
human-in-loop pass. No count; MIME held out. No new transfers, domain,
or emission surface.
