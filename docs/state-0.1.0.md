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

**WHERE THE INSTRUMENTS ARE, because five are named below and not one of them is
in this repository.** They live in the campaign repo `stelling-sweeps` and are
written throughout as `stelling-sweeps/<name>`, so that no reader looks for them
here. They load the MADDENING/MIME corpus, which is not a dependency of this
repository, so **nothing they measured can be re-derived in this tree** and
committing them here would add scripts that cannot run. Every figure sourced
from them is RECORDED-HISTORICAL in the sense clause 1 gives that word, and
carries the sha of the instrument that produced it instead of a gate. *As-of, in
`stelling-sweeps`:* `stelling-sweeps/nonvacuity_seven.py` at `311d0e2`,
`stelling-sweeps/row7_driven.py` at `396e8cd`,
`stelling-sweeps/BASELINE_pre_0_4_0.json` at `1608457`,
`stelling-sweeps/rederive_terminals_idiomatic.py` at `7a594aa`,
`stelling-sweeps/CLAUSE_CENSUS_state_0_1_0.md` at `fa69d34`. **The registry
census in clause 3 is the opposite case — its population is in this tree, so it
is gated and not stamped**, which is the
distinction this document's own rule draws further down: *a count over a
population that is not in the tree gets a sha; a count that is computable from
the tree gets a gate.*

**Measured, not assumed.** Two independently-blinded agents wrote contracts
against `jax_md` and `jaxfluids` — genuinely independent authors and domains —
having been told nothing about this project's conclusions. **The blockers are
IDIOMATIC.** That finding is four separate claims with four different
standings, and it is written out that way because it was previously one
sentence and the sentence did not survive re-derivation (2026-08-03,
`stelling-sweeps/rederive_terminals_idiomatic.py`).

1. **The external terminal list is RECORDED-HISTORICAL, not re-derivable.**
   External code was measured to terminate on `square`, `sign`, `unstack`,
   `copy`, `rem`, `nan_to_num`'s `inf`, `iota`, `int64→float64`, `abs`, `pow`.
   **The two blinded contracts are not in this tree**, so nothing here can
   re-derive that list, and **every claim quantifying over it inherits that
   status.** *Unit: primitives and forms hit by two external contracts,
   n = 2 codebases, as measured in the session that wrote them.*

2. **None of them is a MADDENING terminal — ON THE FORM UNIT, and the unit is
   where this turns.** *False on the primitive unit:* `int64→float64` **is**
   `convert_element_type`, which is MADD HeatNode's own baseline terminal.
   *True on the form unit:* MADDENING's terminal is `convert_element_type`
   **float64→float32** (value-changing); the external one is **int64→float64**.
   Same primitive, different forms. **1 of the 10 listed items turns on this
   distinction.** *Unit: forms, n = 10 listed items.*

3. **One of them has neither an interval transfer nor an emission row: `iota`.**
   **No figure in this clause is left to a human transcription, and the reason
   is its own history.** It is a claim about what is in two live registries, so it is
   computable from this tree — and a registry claim is invalidated by the next
   commit that touches a registry. It has been wrong twice for exactly that
   reason: it read *"five"*, which was measured against the tree as it stood
   immediately before the commit that published it and was invalidated by that
   same commit (see the note below); it then read *"seven … as of `f8f5850`"*,
   which was **wrong at the very sha it named** — the correct figure at
   `f8f5850` was six, because `square` had gained an emission row at `e3b9deb`,
   one merge earlier. Stamping a sha did not save it, because the number was
   still typed by a human. The census below is **executed, and its output is
   compared byte for byte, by `tests/test_doc_examples.py`**. That mechanises
   the measurement and nothing after it — and the step after it, measurement to
   prose, is where all three of this project's recorded unit failures actually
   happened. So the sentence you just read and the unit line below are
   **re-derived from those same two registries by
   `tests/test_release_doc_claims.py`**: the count, the named member, and both
   populations, with the census's `recorded` list required to be clause 1's ten
   items under the identification clause 2 states. That gate imports no jax, so
   unlike the executed census it also runs in the zero-dep configuration this
   count is about. Prose that stops agreeing with the block goes red; it can no
   longer go stale. *Unit: membership in `stelling.propagate.TRANSFERS` and
   `stelling.obligation._SUPPORTED`, n = 9 primitives of the 10 recorded items
   — `nan_to_num`'s `inf` is not a primitive and has no registry key.*

   ```python
   from stelling.obligation import _SUPPORTED
   from stelling.propagate import TRANSFERS

   # The nine primitives among the ten recorded items; `nan_to_num`'s `inf` is
   # not a primitive, so it has no registry key to look up.
   recorded = ["square", "sign", "unstack", "copy", "rem", "iota",
               "convert_element_type", "abs", "pow"]
   buckets = {"neither": [], "transfer only": [], "both": []}
   for p in recorded:
       t, e = p in TRANSFERS, p in _SUPPORTED
       buckets["both" if t and e else "transfer only" if t else "neither"].append(p)
   for name, members in buckets.items():
       print(f"{name:13s} {len(members)}  {members}")
   ```

   ```
   neither       1  ['iota']
   transfer only 6  ['sign', 'unstack', 'copy', 'rem', 'abs', 'pow']
   both          2  ['square', 'convert_element_type']
   ```

4. **Two terminals recur, not one — and the mechanism claim stands.** Within
   MADDENING/MIME the element budget recurs on **3** contracts *and* the
   constrained-assume refusal on **2** (`stelling-sweeps/BASELINE_pre_0_4_0.json`
   at `1608457`). Whether the budget is the sole *cross-framework* recurrence
   depends on claim 1 and is
   therefore not re-derivable here. **The mechanism is untouched by the count
   and stands: the budget recurs because it is a function of query SIZE rather
   than of coding style.** *Unit: blocked contracts, n = 7 of 14.*

> **A number here is safe to the extent its unit is written beside it.**
> Measured: of ten conclusion-bearing clauses in this document, four were
> re-derived, and the only one that re-derived exactly is the one whose unit the
> document names. All three failures were failures of unit, not of arithmetic.
> See `stelling-sweeps/CLAUSE_CENSUS_state_0_1_0.md` at `fa69d34`.

> **SIX CLAUSES BELOW WERE NOT RE-DERIVED, AND UNATTEMPTED IS NOT EVIDENCE
> EITHER WAY.** Each needs a counterfactual sweep or a corpus re-run rather than
> a registry read, and the pass that stated the units below did not run them.
> They are left exactly as they were measured, and are named here rather than
> quietly softened:
>
> - *"Seven capability counterfactuals at zero, one positive"* — needs the
>   counterfactual frontier re-run. Unit stated (capabilities, not runs).
> - *"one member clears 1 of 14"* in the fixed-width family table — needs the
>   bounded-error sweep across both solvers. Unit stated in the column header.
> - *"Both primitive frontiers read zero over fourteen"* — needs the frontier
>   counterfactual re-run. Note the population is **fourteen** here and
>   **twelve** for the emission figure further down; both are correct on their
>   own populations, and neither is the other's.
> - *"The API-validation scope is 3 of 9"* — a hazard census over the nodes'
>   metadata, not derivable from any registry in this tree.
> - four of the five `div` sites — only site 5 has been re-checked.
> - every MORE/LESS cell in the stub-direction table — each needs its stub
>   experiment re-run in-process.
>
> **The stub-direction table is the one to distrust first**, and not because a
> row is known wrong: its key term — "over-permissive" — had no stated unit,
> which is the property that failed three times out of four in the census above.
> A direction is now supplied above the table; the cells themselves are still
> the original measurements and none was re-run.

> **2026-08-03 — `square` now has an SMT emission row.** What that changes,
> stated only as far as it was measured. A NEW contract, written against
> `jaxfluids`' `WENO5Base.smoothness`, moves from `escalation declined:
> primitive 'square' is outside the supported emission set` to **REFUTED with
> a replay-confirmed witness** (`tests/test_square_acceptance_jaxfluids.py`).
> That is a new contract, not a re-run: the two blinded external contracts
> behind the measurement above are not in this tree, so **nothing is claimed
> here about what they would hit today**, and the terminal table's site 5
> ("its effect on the external harness is unverified") is left standing. Note
> also that site 5's mechanism is `square` **⊤ poisoning** — the INTERVAL
> transfer falling to ⊤ — which an emission row does not touch.
>
> Not a universal, and the counterexamples are in this repository. A slice
> traversing `square` still declines on an `int32`/`int64` declaration and
> through the bool→`int32` route (`square`'s own overflow guard), on boolean
> operands, over the element budget, and — for a complex value reached by
> `astype` — at the conversion, before `square` is reached at all. Measured;
> each has a test in `tests/test_square_row.py`.
>
> This note used to add that "the trailing count is untouched on purpose:
> `square` was never one of the five". **There is no longer a trailing count to
> touch** — clause 3's census is generated from the registries and executed by
> the suite, and it shows `square` in the `both` bucket, which is the same fact
> stated by a mechanism instead of by a promise. The rest of the recorded list
> is unretested by this change and is left as measured.

**The one-terminal count that used to stand here is RETRACTED; claim 4 of the
IDIOMATIC finding above replaces it, and this paragraph is left in place so the
retraction is not silent.** It read *"one terminal recurs across all three, and
its mechanism explains why: the element budget"*. Two recur internally, not one,
and the cross-framework half is not re-derivable in this tree. **The mechanism
is what survives** — the element budget recurs because it is a function of
query SIZE rather than of coding style — and it is untouched by the count.

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
| nonvacuity | could the property have failed? | **2 checked non-vacuous** — see below; the two are not the two named next |

**Two are deep, load-bearing AND non-vacuous: `MADD RigidBody` and
`MADD row7.richardson`** — **on the DRIVEN reading of `row7`, which is the unit
that was missing and is stated here.**

**Two counts of two, over two different populations.** Re-measured — **outside
this tree, and not re-derivable in it** — by
`stelling-sweeps/nonvacuity_seven.py` at `311d0e2`, which ties each box to the
node's own `initial_state()`: the contracts whose nonvacuity comes back
`checked` are
**`MADD RigidBody` and `MIME coil_array`**, and `MADD row7.richardson` comes
back **FAILED** there. So the axis table's *"2 checked non-vacuous"* and the
*"deep, load-bearing AND non-vacuous"* pair are **numerically compatible and
not the same set** — the exact confusion this project's own rule about naming
the population per figure was written against.

**`row7` is `checked` against a DRIVEN point and against no other.** Measured
outside this tree (`stelling-sweeps/row7_driven.py` at `396e8cd`): undriven,
the node's reachable span is `[0, 0]` against a declared
envelope of `(10.0, 100.0)` — **DISJOINT**, and nonvacuity at
`initial_state()`'s `0.0` is FAILED. Driven (left = 100.0, right = 20.0, 4000
steps at dt = 0.01) the span is `[0, 100]`, and nonvacuity at the driven
steady-state value `20.0` is `checked`. *Unit: one stated point per contract,
and which point is load-bearing.*

**`MIME surface_contact`'s nonvacuity has never been measured at all** —
`stelling-sweeps/nonvacuity_seven.py`'s `nv_contact` is an unfinished stub that
returns `None`
and is not in its case list. It is named here because nothing above rests on
it, and a reader counting the seven VERIFIEDs should know which cell is empty
rather than zero.

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

### **STUB-DIRECTION AUDIT: two zeros questioned — one restored, one re-run and held**

Norm I says an over-permissive stub's zero is conclusive — a stub granting MORE
than a real implementation upper-bounds the benefit, so a zero means a real fix
can only do worse. **That argument requires the stub to be over-permissive, and
two of these were not.**

**THE UNIT OF THE MORE/LESS COLUMN — half quoted, half supplied here, and the
halves are marked because an earlier version of this paragraph claimed the norm
was silent when it is not.** `CONTRIBUTING.md`'s *"An over-permissive stub's
ZERO is conclusive; its NONZERO is not"* **does define the direction**, in the
sentence that opens it: *"A stub that grants **more** than a real implementation
could deliver **upper-bounds** the benefit."* It also carries the rule *"state
what the stub grants, before reporting what it produced"*. So MORE/LESS below —
*what the stub grants the analysis relative to a faithful implementation* — is
that norm's, restated, not this page's invention. **What the norm does not say
is for which OUTCOME the grant is judged, and that is the part supplied here:**
judged **for a DEFINITE verdict** — VERIFIED or REFUTED rather than a DECLINE,
which is what every contract in this corpus seeks. Sought differently, a cell
can flip.

**And where the stubbed stage is never reached, the direction is UNDEFINED —
not neutral and not LESS.** An unreachable stage grants nothing in either
direction, so a zero measured over it needs the reachability argument and not
the over-permissiveness one. **That is exactly the emission row below**, whose
zero is restored on reachability two subsections down; its LESS cell describes
what the stub would do *if the stage were reached*, which on this corpus it
never is. That clause is this document's own finding, recorded below — it is
not in the norm either.

*Unit of the population: six stubs, one per row. Every MORE/LESS cell is a
recorded measurement from the session that ran the stub, and none was re-run in
the pass that added this paragraph.*

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

**Six of the seven capability zeros are conclusive** — and the seventh is named
here, because a count of conclusive zeros that never says which one is not is
the same defect as a count with no unit. *Unit: one zero per capability in the
table above, n = 7, on the reading the table's two columns force — six
capabilities are zero **jointly**, and the seventh, transfer coverage
**pre-build**, is zero **per-entry only**.* **That per-entry zero is the
non-conclusive one**, and it is non-conclusive for a measured reason and not a
cautious one: the joint run over the same capability returned **1**. It is the
same fact the note under that table already states — the exception was
**invisible per-entry** — counted here rather than only described. The one
positive result is unaffected — `split` + `add_any`
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

**THE CONSTRUCTION, because these four figures are construction-dependent and
without it they read as wrong.** The jaxpr is the **stelling harness**

<!-- doc-example: illustrative -->
```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax_md import space

from stelling import coverage
from stelling._jax_compat import trace
from stelling.harness import any_array
from stelling.propagate import TRANSFERS

displacement_fn, _ = space.free()

def query():
    Ra = any_array((3,), jnp.float64, (0.0, 1.0))
    Rb = any_array((3,), jnp.float64, (0.0, 1.0))
    return space.distance(displacement_fn(Ra, Rb))

print(coverage.measure(trace(query), known=frozenset(TRANSFERS)))
```

with equations counted at every depth. **The two `stelling_any` declaration
equations are inside the count**, and they are the whole of the difference
between this figure and the one a reader gets by reaching for
`jax.make_jaxpr` over two concrete arrays: that trace has no
declaration equations, so the same census reads **`12/9/3/0`**. Both readings
give `unknown=0`, and both give a ⊤ interval face, so the *conclusion* is the
same on either — but the digits are not, and an attempted re-derivation of this
sentence missed by two on each of the first two columns for exactly that reason.
`space.metric(displacement_fn)(Ra, Rb)` reads `14/11/3/0` too.

*As-of, because this one CANNOT be gated here:* stelling `0c4cead`, jax 0.11.0,
jax-md at `eec6d1f`, x64 enabled. **`jax_md` is not a dependency of this
repository**, so no test in this tree re-derives these four numbers — unlike the
registry census near the top of this page, which `tests/test_release_doc_claims.py`
re-derives from the registries on **every** run, the zero-dep one included.
(`tests/test_doc_examples.py` executes that census as well and compares its
fence byte for byte, but that module skips every example when jax is absent, so
on its own it would leave a zero-dep-core count unchecked in precisely the
configuration the count is about.) A count over a population that is not in the
tree gets a sha; a count that is computable from the tree gets a gate. The
`jax_md` block above is illustrative for that reason and is not run.

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
| MADD aitken/caller · MADD LBM/Re-range | **constrained-assume refusal** |
| 7 contracts | VERIFIED |

*Unit: one baseline terminal per contract, n = 14, read from
`stelling-sweeps/BASELINE_pre_0_4_0.json`'s `rows` at `1608457` — a file
outside this repository, so this table is recorded, not re-derivable here; the
contract labels are that file's, with
`MIME d2q9/LBM` shortened.* **The first label in the refusal row read
`coil_array/caller` — which is not one of the fourteen contracts in the
baseline, nor one of the twelve the frontier traces.** Re-read from the
baseline: the pair recorded against the constrained-assume refusal is
`MADD aitken/caller` and `MADD LBM/Re-range`; `MIME coil_array` is one of the
seven VERIFIEDs and hits no terminal at all. **`MADD aitken/caller` is not the
`aitken` counted among those seven** — that one is the baseline's
`MADD aitken.omega_floor`, shortened to `aitken` above. They are two separate
rows of the fourteen, and mistaking a contract for its `/caller` variant is the
identical slip that produced the label just corrected.

**Four distinct baseline terminals over seven blocked contracts**: the element
budget (3), the constrained-assume refusal (2), `convert_element_type` (1), and
`sqrt` (1). **The budget and the refusal each recur across independent nodes** —
which is the opposite of the retracted claim. `div` does not appear at baseline
anywhere; it was only ever visible under peeling.

**What sits BELOW each baseline terminal is unmeasured**, because measuring it
requires peeling, and peeling past an emission gap cannot be done faithfully
while emission is unreachable. **The real terminals below the constrained-assume
refusal — `MADD aitken/caller`'s and `MADD LBM/Re-range`'s — are the cells that
remain open.** (This sentence also said `coil_array`, inheriting the same wrong
label from the table above; `MIME coil_array` VERIFIES and has nothing below it.)

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
  (MIME's `gnn.py:312` hard-wires `3` where `mesh.dim` belongs). Both forced a node
  into a configuration nobody exercised. **This is a value channel independent
  of any verdict a contract produces**, and it is the kind that vanishes from a
  record because it is not a verdict.

Every number above is checkable against them, which is the property the numbers
themselves do not have.
