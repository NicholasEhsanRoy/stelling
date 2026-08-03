<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# The measured state at 0.1.0

**No summarizing characterization.** Four have been written and all four were
retracted. What follows is the measurements and their scopes; the reader draws
the summary, and if none is available that is itself the result.

## SCOPE — read this before any number below

**Every capability conclusion in this document was measured on ONE framework
family: MADDENING/MIME.** The seven counterfactual zeros, both frontier zeros,
the terminal table, the guard frontier, and the nine-hazard population are all
**stelling-on-MADDENING** results. They are not statements about stelling.

**Measured, not assumed.** Two independently-blinded agents wrote contracts
against `jax_md` and `jaxfluids` — genuinely independent authors and domains —
having been told nothing about this project's conclusions. **The blockers are
IDIOMATIC**: external code terminates on `square`, `sign`, `unstack`, `copy`,
`rem`, `nan_to_num`'s `inf`, `iota`, `int64→float64`, `abs`, `pow` — **none of
which is a MADDENING terminal**, and five of which have neither an interval
transfer nor an emission row.

**One terminal recurs across all three, and its mechanism explains why:** the
element budget, because it is a function of query SIZE rather than of coding
style.

**THE LARGEST SCOPE CHANGE: the emission zero held BECAUSE emission is
unreachable in MADDENING** — eleven of twelve contracts decline before reaching
it. **External code DOES reach emission.** So emission's value outside this
corpus is **genuinely unmeasured, not zero.**

**And the strongest evidence stelling has produced came from code nobody here
wrote, on a first attempt.** An external `RungeKutta3` contract verified on all
three axes simultaneously — full coverage, load-bearing envelope, nonvacuity
checked against the library's own data — **with a passing negative control**
(a mutated coefficient REFUTES). The internal corpus has two contracts on three
axes and no negative control anywhere.

**Framework denominator: n = 1 before that session, n = 3 after, two of them
independent.** That is still small, and *"idiomatic"* rests on two samples
agreeing that MADDENING's terminals do not appear in them.

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
| **`int64→float64` widening from an integer literal** | **0 of 14** — no internal contract reaches it | **yes, externally**: it is the ⊤ at **41 jax-md `safe_mask` sites**, and `int64→float64` was already on the external-terminal list above with nobody having connected it to the mechanism |

**The fourth member is the first with external demand and zero internal
demand**, which is the denominator finding arriving at the family table: a
member measured at 0 on this corpus is not measured at 0. Proposal (not a
build — the family is parked) in
[proposed-int-literal-convert.md](proposed-int-literal-convert.md).

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

**And that blindness is not an internal curiosity.** Measured on `jax_md`'s
`space.distance`, `coverage.measure` reports **total=14, known=11,
transparent=3, unknown=0** — a census reading 100% known — while the interval
face returns **⊤** for the same jaxpr. The census counts membership; the
decline is a registered primitive refusing on a condition. **This is the
mechanism by which an external idiom looks fully covered and is not.**

### `div` is FIVE SITUATIONS and FOUR MECHANISMS

Earlier records carried two tables that disagreed — one listing four sites with
`gnn` absent and where-correlation loss counted twice, one listing the internal
per-divisor mechanisms with no external sites — and a summary that said "four
causes", which matches neither. Measured, with `Literal` divisors distinguished
from `Var` (the first pass misclassified literals as "no box", and that was the
instrument, caught before reporting):

| # | site | mechanism | disposition |
|---|---|---|---|
| 1 | MADDENING Aitken | **where-correlation loss** — divisor box `[0,12]` from a `where` whose guard interval propagation cannot see | the where-refinement |
| 2 | jax-md `safe_mask` (41 sites) | **⊤ at `convert_element_type int64→float64`**, from the literal `0` in `safe_mask`'s own body — the correlation question is never reached | the int-literal convert; **NOT** the refinement |
| 3 | MIME coil_array | **mechanism (iii)** — the divisor has NO ENTRY in the env | a different fix; unmeasurable while emission is unreachable |
| 4 | MIME gnn | **an upstream ⊤ cascade** — one `[-inf,inf]` box from `jit`; 15 literal divisors, the rest clean | unattributed |
| 5 | JAXFLUIDS | **`square` ⊤ poisoning `x*x + eps`** | the `square` **interval transfer** row — **BUILT**; its effect on the external harness is unverified. **The SMT emission row for `square` (2026-08-03) does NOT address this site**: the mechanism here is the INTERVAL transfer falling to ⊤, and an emission row does not touch it. **Site 5 remains open.** |

**NO CAPABILITY ADDRESSES MORE THAN ONE.** The where-refinement was credited
with two of four; it reaches **one of five**, because site 2 stops three steps
earlier than the mechanism it was credited for.

Site 2 is worth stating precisely, because it is one character. `safe_mask`'s
body is `jnp.where(mask, operand, 0)` — a **Python int** placeholder, which
promotes through `convert_element_type int64→float64`, which declines.
Measured, same box, same shape of `_where` jaxpr:

```
jnp.where(x > 1.0, x, 0)      ->  TOP        jnp.where(x > 1.0, x, 0.0)  ->  [0, 12]
```

MADDENING's Aitken survives the identical `_where` structure only because its
placeholder is already `float64`, so its convert is `float64 -> float64` and
passes through. And the predicate form was never the obstacle: `safe_mask`'s
FIRST `where` **is** the in-scope form (`where(p(x), x, k)`). Classified by AST
over all 41 call sites, not by eye:

| | sites | |
|---|---|---|
| direct ordering comparison of the operand against a constant | **18** | in scope as scoped |
| `&`-compound with one such conjunct | **3** | in scope — the true branch implies EVERY conjunct, so refining on the one recognised conjunct is sound and needs nothing from the others |
| opaque mask (tests a different variable) | 14 | out |
| `&`-compound whose operand-conjunct compares against a **variable** | 3 | out — relational, not a box |
| `!=` predicate | 2 | out — excluding a point does not narrow an interval |
| operand is not a plain name | 1 | out |

**21 of 41 in scope. The form matches and it does not matter**, because every
one of the 41 stops at the convert first.

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
  node coverage 87.5% → 62.5%, and a reachable span that was a free-fall
  artifact of the step count. **The unit is MADDENING numeric NODES** — the
  fraction whose chosen obligation's slice is transfer-covered — and the
  cross-check found **two** independent inflations, obligation choice (that
  figure) and a population omission (12 nodes exist, 8 were scored, the 4
  missed are the hardest). **The honest figure is a range, ~0.42–0.63**, and
  the two runs were **not like-for-like**: one agent was asked for "one
  plausible obligation", the other for what "a domain expert would most want
  checked."
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
