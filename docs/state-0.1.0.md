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

**Seven capability counterfactuals at zero, one positive** — counting
CAPABILITIES, which is the unit of the table below. (Earlier reports said
"eight consecutive", counting *runs* rather than capabilities: the transfer
frontier was measured twice, before and after the `split`/`add_any` build.
Stating the unit, because the same slip produced the `iota` misclassification.)

| capability | per-entry | joint |
|---|---|---|
| emission coverage · (iii) transparent-call · element budget · transfer coverage (post-build) · constrained-assume refusal · `div` divisor-nonzero | 0 | **0** |
| **transfer coverage, pre-build** | 0 | **1** — `split` + `add_any` moved `coil_array` |

The exception was **invisible per-entry**. Joint stubbing is standing practice
because of it.

### **STUB-DIRECTION AUDIT: one zero provisional, one restored**

Norm I says an over-permissive stub's zero is conclusive — a stub granting MORE
than a real implementation upper-bounds the benefit, so a zero means a real fix
can only do worse. **That argument requires the stub to be over-permissive, and
two of these were not.**

| stub | what it granted vs a real implementation | zero conclusive? |
|---|---|---|
| **emission membership** (`_SUPPORTED \|= {p}`) | **LESS.** Measured: the slicer accepts, then `emit()` raises *"emission has no rule for primitive 'sqrt' — slice validation should have declined this"*, and the verdict reads `escalation attempted; internal error`. **A real emission ROW emits terms and reaches the solver; the stub cannot.** | **NO** |
| **transfer stubs** | **LESS.** They are sound but deliberately coarse (`sin` → `[-1, 1]`, `split` → the input's hull). A real transfer is tighter, and a coarser box can fail to discharge where a tighter one succeeds. | **NO** |
| `div` divisor-nonzero | MORE — asserts what the analysis could not establish | yes |
| constrained-assume refusal | MORE — removes the policy without emitting the constraint | yes |
| element budget raise | MORE — 500,000 exceeds any plausible real bound | yes |
| `convert` exactness | MORE — pretends the rounding is free | yes |

#### The emission zero: RESTORED, on a different argument

The retraction above was **too strong for emission**, and the check that settles
it is direct. Running the joint emission stub over the corpus:

```
contracts whose decline is an INTERNAL ERROR : 0 of 12
solver invocations that ANSWERED             : 2, all from MADD projection
```

**The stub never raised, because eleven of twelve contracts never reach
emission at all** — they decline earlier (element budget, `convert_element_type`,
an `unstack` shape at slice validation) or discharge on intervals without
escalating. The twelfth reaches the solver and **already VERIFIES**.

**So no emission row could change any verdict here, and the stub's fidelity is
UNDEFINED rather than irrelevant — an unreachable stage has no direction** — a faithful row would sit behind the same
earlier declines. **The emission zero stands**, on the argument that nothing
reaches the stage rather than on the stub being over-permissive.

**Scope, stated because the earlier version over-reached in exactly this way:**
this licenses *"no emission row changes a verdict in THIS corpus"* — and it is
established **BECAUSE EMISSION IS UNREACHABLE, not because emission is
unhelpful.** Those license different downstream claims. In particular it says
**nothing at all** about a corpus where the earlier declines do not fire, which
is exactly what an external codebase is. **The emission zero is MORE
MADDENING-scoped than it looks, not less.**

#### The transfer zero: RE-RUN WITH TIGHT STUBS, AND IT HOLDS

Tightness has a ceiling, which narrowed the re-run from the whole frontier to
**two cells**. Measured input intervals: `sin`/`cos` in `rigid_body` see
`[-inf, inf]` — spanning a full period, so `[-1, 1]` **is** the tightest sound
box and the coarse stub was already faithful; `atanh` sees the point `[0, 0]`
and its stub is monotone; `not` is exact. Only **`rigid_body`'s `unstack`**
(hull vs exact slicing) and **`gnn`'s `sin`** over `[0, 0.785]` could differ.

Re-run with a monotone-piecewise `sin` and an exact `unstack`, per-entry and
jointly: **changed = 0, DECLINE→VERDICT = 0.** Pre-registered prediction
(both zeros hold, because `rigid_body`'s terminal is a SHAPE check and `gnn`'s
is a SIZE check, and box tightness affects neither) — **confirmed.**

**Six of the seven capability zeros are conclusive.** The one positive result is unaffected — `split` + `add_any`
produced a VERIFIED under COARSER stubs than the shipped transfers, so a real
implementation can only do better.

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

**Both primitive frontiers read zero over fourteen** — and the corpus covers the
**DOCUMENTED** symbolic hazard surface (4 of 4 hazards have contracts), so the
zeros are no longer scoped to an opportunistically assembled population.

**"Documented" is the load-bearing word.** The nine hazards were read from the
nodes' own metadata, and **nothing has measured whether that metadata is
complete.** It is not: the `GNNFluxCorrectedFVMNode` 2D-mesh defect — the node
cannot be stepped on any 2D mesh — appears in no hazard list and was found by
**trying to write a contract**. So the corpus covers what the authors wrote
down, which is a lower bound on the hazard surface, not the surface.

**Primitive frontiers cannot see guards**, by construction: both enumerate
primitives, and a guard is a registered primitive refusing on a condition. The
guard frontier, measured by peeling every gap and the budget:
`convert_element_type` blocks two contracts, `div` blocks one.

### Terminals — **RETRACTED and re-derived**

The earlier version (*"four hazards, three distinct terminals, three nodes,
n = 3"*) named terminals found by **peeling with emission stubs**. Since emission
is never reached, those peels showed **the next slice-validation decline and
nothing more** — not what emission would do. **`sqrt` as `coil_array`'s terminal
was wrong, not provisional**, and the n = 3 structural claim inherited it.

**Re-derived from BASELINE causes only — the sole fully faithful terminals,
measured with no stub of any kind:**

| contract | baseline terminal |
|---|---|
| MADD HeatNode | `convert_element_type` value-changing |
| MADD LBMNode · MIME d2q9 · MIME gnn | **element budget** |
| MIME rigid_body | `sqrt` emission gap |
| coil_array/caller · LBM/Re-range | **constrained-assume refusal** |
| 7 contracts | VERIFIED |

**Four distinct baseline terminals over seven blocked contracts**: the element
budget (3), the constrained-assume refusal (2), `convert_element_type` (1), and
`sqrt` (1). **The budget and the refusal each recur across independent nodes** —
which is the opposite of the retracted claim. `div` does not appear at baseline
anywhere; it was only ever visible under peeling.

**What sits BELOW each baseline terminal is unmeasured**, because measuring it
requires peeling, and peeling past an emission gap cannot be done faithfully
while emission is unreachable. **`coil_array`'s real terminal below the refusal
is the cell that remains open.**

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
- **contract-writing itself, as a bug-finding channel.** Two defects were found
  by *trying to write a contract* rather than by any verdict: the lazy-cache
  friction, and `GNNFluxCorrectedFVMNode`'s inability to step on a 2D mesh
  (`gnn.py:312` hard-wires `3` where `mesh.dim` belongs). Both forced a node
  into a configuration nobody exercised. **This is a value channel independent
  of any verdict a contract produces**, and it is the kind that vanishes from a
  record because it is not a verdict.

Every number above is checkable against them, which is the property the numbers
themselves do not have.
