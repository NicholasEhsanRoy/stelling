<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# The measured state at 0.1.0

**No summarizing characterization.** Four have been written and all four were
retracted. What follows is the measurements and their scopes; the reader draws
the summary, and if none is available that is itself the result.

## Verdicts

**7 of 14 contracts VERIFIED.** They are not equivalent, and three axes
distinguish them:

| axis | question | count |
|---|---|---|
| depth | did the arithmetic get carried, or did ⊤ do the work? | 7 |
| load-bearing envelope | does the verdict need the declared bounds? | **5** |
| nonvacuity | could the property have failed? | **2 checked non-vacuous** |

**Two are deep, load-bearing AND non-vacuous: `MADD RigidBody` and
`MADD row7.richardson`.**

`RigidBody` carries two qualifications that travel with it:

- its declared envelope `(-1.0, 1.0)` is **arbitrary** — one literal serving both
  `position` and `velocity`, originating in a reconstruction whose own commit
  message records that this contract *"does not reproduce the original's
  verdict"*;
- the obligation asserts `|position| <= 1e6` while the envelope is declared over
  position **and velocity**, so velocity is load-bearing through a single
  multiplication and is not what is asserted about.

Of the remaining five: two have a **non-load-bearing envelope** (`coil_array`,
`surface_contact` — each discharges with every bound widened to ⊤), and three
have **no incident data to tie a box to** (`aitken`, `projection`, `rk4_step` —
a free function, a projector built from synthetic linspaces, and a synthetic
oscillator).

**One REFUTED**, and it is the strongest single result: a witness that
**replays** (exact-rational, no solver), **executes** (node output max `101.0`
against a declared bound of `100.0`), and **satisfies its preconditions** (the
query contains no `assume` at all, and the witness lies in the declared box).
It reproduces from a clean checkout of `main`.

## Capabilities

**Eight consecutive capability counterfactuals at zero. One positive.**

| capability | per-entry | joint |
|---|---|---|
| emission coverage · (iii) transparent-call · element budget · transfer coverage (post-build) · constrained-assume refusal · `div` divisor-nonzero | 0 | **0** |
| **transfer coverage, pre-build** | 0 | **1** — `split` + `add_any` moved `coil_array` |

The exception was **invisible per-entry**. Joint stubbing is standing practice
because of it.

**The fixed-width boundary is a FAMILY, not a capability** — at most three
separately buildable members:

| member | clears alone (of 14) | measured |
|---|---|---|
| `convert` f64→f32 rounding | **1** (`HeatNode`) | **yes** — bounded-error sweep, both solvers, crossover reproducing the CFL limit |
| integer overflow | **0** | yes, by peeling |
| narrowing-int wrap | unmeasured — no contract reaches it | no |

`convert uint8→bool` was **pulled out of the family**: it is exactly `x ≠ 0`, a
total predicate, not a rounding. It needs a new emission rule (`(distinct in 0)`),
not a whitelist entry — the whitelist means *emit as identity* and this is a
sort change.

**Both LBM contracts need two members plus what sits above them. Requirements
are per-contract, not shared.**

## What blocks

**Both primitive frontiers read zero over fourteen** — and the corpus now covers
the symbolic hazard surface completely (4 of 4 hazards have contracts), so the
zeros are no longer scoped to an opportunistically assembled population.

**Primitive frontiers cannot see guards**, by construction: both enumerate
primitives, and a guard is a registered primitive refusing on a condition. The
guard frontier, measured by peeling every gap and the budget:
`convert_element_type` blocks two contracts, `div` blocks one.

**Four hazards, three distinct terminals, three nodes.** `div` on Aitken, `sqrt`
on both `coil_array` hazards (they share a node and reach the same MPEM
computation), `convert`→integer-overflow on LBM. **Nothing helps across
independent nodes — n = 3.**

## The `assume` axis

**Retired across every symbolic candidate that has a contract.** Three
candidates, three different things blocking before `assume` becomes decisive:
`div`, `sqrt` emission, `convert`. `assume` itself works — it constrains, F7
fires — it is simply never the last thing missing.

## Hazards

**Nine documented hazards over twelve nodes, every cell measured**: 4 symbolic,
2 per-step `dt`, 3 constructor-visible. **The API-validation scope is 3 of 9** —
the hazards that are *both* written-down *and* constructor-visible. `dt` is a
per-step argument on every node checked, so *"concrete at trace time"* is not
*"available at construction"*.

## The instruments

**They are the deliverable, and the honest argument for them is that most exist
because a number was wrong without them.**

- **the three axes** — depth, load-bearingness, nonvacuity. Seven VERIFIEDs
  became two under them.
- **the counterfactual method** — four proxies each overcounted before it.
- **joint stubbing** — the campaign's only positive capability result was
  invisible per-entry.
- **blind cross-checking** — twice caught an error the measurer could not:
  obligation coverage 87.5% → 62.5%, and a reachable span that was a free-fall
  artifact of the step count.
- **the gauges** — mutation batteries with declared scope, positive and negative
  controls.
- **the norms** — each carries the instances that earned it, with numbers.

Every number above is checkable against them, which is the property the numbers
themselves do not have.
