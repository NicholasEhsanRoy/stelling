<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Choosing a solver backend: z3, cvc5, or both

**Install both (`[solvers]`).** Not because one is unreliable, but because
the portfolio's whole design is that two independent backends answer the
same question and agreement decides. Installing one does not make verdicts
weaker in what they *claim* — a one-backend `VERIFIED` is still a
`VERIFIED` — it removes the only independent check a discharge has, and
stelling says so in the verdict rather than letting you find out later.

If you can only install one, this page says what that costs, measured. It
is not a recommendation derived from the backends' reputations; every figure
below was taken against this tool.

**How this page was measured, and what that does not cover.** This paragraph
is the provenance of the ORIGINAL figures — the ten-row table below and the
quoted declines. The re-measured table beside it carries its own, taken on a
different day and naming the CPU and the load average, neither of which this
paragraph ever did. All the original figures are from `stelling 0.2.0.dev0`,
`jax 0.11.0`, CPU, `jax_enable_x64=True`, z3 `5.0.0` (wheel) and cvc5
`1.3.4-modified` (wheel), Linux x86-64, Python 3.12,
`solver_timeout_ms=10000`, three repeats per cell. Single-backend
configurations were produced by restricting the portfolio and by hiding a
backend from `stelling._optional.available` — nothing was uninstalled, and
both routes produce the same degraded-portfolio disclosure (below). Wall
times are one machine's; the *decided / timed-out* column is the part that
carries the argument, and even that is a timeout budget away from moving.

**THE HARNESSES BEHIND THE TEN-ROW TABLE WERE NEVER COMMITTED, AND TEN ROW
LABELS DO NOT PIN TEN HARNESSES.** `grep` for `Motzkin`, `AM–GM`, `product
chain` or `elementwise products` across the whole repository still returns
this file and nothing else, and that half of the old admission here is not
repairable by editing: not one of these labels fixes a declared box, a
predicate or a threshold, and *"scalar, linear"* is satisfied by an infinity
of obligations that do not take the same time. Read the ten rows for their
*direction*, not their milliseconds — that instruction stands, and what
follows makes it sharper rather than retiring it.

**What HAS changed is that there is now a battery you can run.**
`tools/solver_battery.py` ships in the sdist (`/tools` is in the allowlist),
drives ten harnesses BUILT TO those labels, and prints — for every row — the
parameters the label leaves open and the ones it therefore had to choose:

- `python tools/solver_battery.py --rows` is the inventory: per row, what the
  label fixes and what this battery picked. It needs neither jax nor a solver,
  because it is data.
- `python tools/solver_battery.py --fragments` re-derives the `fragment`
  column. It needs jax and **no backend** — the fragment is decided by
  `stelling.obligation`'s slicer off the traced jaxpr, before `escalate`
  discovers anything — so it is the one column below that reproduces on any
  machine, including in the zero-solver lane.
- `python tools/solver_battery.py` drives the whole battery against whatever
  is installed, and says cell by cell what it could not measure and why. With
  no backend at all it still prints the mechanism column and exits 0.
- `python tools/solver_battery.py --variants` drives the *alternate readings*
  of the rows whose labels admit more than one. Read that before trusting the
  nonlinear direction below; it is where this page's headline nonlinear
  finding stopped being reproducible.

**It is deliberately NOT a re-derivation of the table below, and it says so on
every run.** The two tables are printed side by side and the tool never edits
its numbers into the page's, because they came from different harnesses.
Making one look like the other would be the worse defect: an unreproducible
table is at least honest about being unreproducible, while an invented harness
*looks* reproducible.

**WHAT IS GATED ON THIS PAGE, AND WHAT IS DELIBERATELY NOT.** Two `python`
blocks are executed and their output compared byte for byte by
`tests/test_doc_examples.py`: the `x**(1/80)` degree-cap decline and the
`semantics="ieee"` caller error. They are the two blocks that make claims
about mechanism rather than speed.

More of the ten-row table is now held, by `tests/test_solver_battery.py`,
and the cut it makes is the same one — mechanism yes, clock no:

| held | by | needs |
|---|---|---|
| the table's ten row labels, their order, their `fragment` cells and all three of their published cells, against `tools/solver_battery.py`'s copy of them | `test_the_battery_is_the_page_s_table_row_for_row` | nothing |
| every row of the battery still declaring which parameters the label left open — a row that stopped would be claiming a reconstruction nobody performed | `test_every_row_says_what_the_page_left_open` | nothing |
| the `fragment` column, re-derived from the traced jaxpr | `test_the_fragment_column_is_what_the_page_publishes` | jax, **no backend** |
| *"32 vars"*, *"64-element array"*, *"2 vars"* — every row's declared input count, read off the emitted slice rather than off its own label | `test_each_row_declares_the_variables_its_label_names` | jax, **no backend** |
| that every harness still ESCALATES — an obligation interval propagation decides never reaches a backend, so it cannot be a row of a solver comparison at all | `test_no_row_is_decided_before_a_backend_is_asked` | jax, **no backend** |
| the six cheap rows' `unsat`/`sat` — a fact about the obligation, not about a backend's speed | `test_the_cheap_rows_answer_what_their_obligation_forces` | jax and a backend |

**Not one millisecond on this page is gated, and that is a decision rather
than an omission.** A wall time is not a property of this tree; a gate on one
would go red when the box is busy and green when the claim is wrong. Rows 7
to 10's published cells are *"did this backend finish inside ten seconds"*,
which is a millisecond wearing a hat, so they are not gated either — they are
re-measured on demand by the battery instead, and the reason is recorded in
`tests/test_doc_examples.py`'s `BLIND_SPOT` entry for this page. The routing
figures and the quoted declines below remain hand-checked and will go stale
the way hand-checks do. The section this page most needed gating is the one
that had gone stalest: see the retraction below.

*How to restrict the portfolio yourself, since this page's own method needs
it:* `check(harness, …, solver="z3")` or `solver="cvc5"`. That is the
supported route and it is validated eagerly — anything other than `None`,
`"z3"` or `"cvc5"` raises `ValueError` at the call. The internal
`SolverConfig(only=…)` object is **not** accepted there (passing one raises
that same `ValueError`); `check()` builds it for you, which is what the
`SolverConfig.only=('z3',)` in the disclosure below is reporting.

## How an obligation reaches a backend

Only obligations that interval propagation (and, if you asked for it, the
affine refinement) left **undecided** are escalated at all, and only when
you pass `solver_timeout_ms`. Each escalated obligation is classified into
one of exactly two fragments by `stelling.obligation._Slicer._fragment`:

| fragment | when | primary | secondary |
|---|---|---|---|
| `QF_LRA` | every operation on declaration-dependent values is linear, and no `pow` has a non-integer exponent | **z3** | cvc5 |
| `QF_NRA` | some `mul` of two dependent operands, `div` by a dependent operand, `square`, `integer_pow` with exponent ∉ {0, 1}, `pow` with a *dependent* base at an exponent ∉ {0, 1}, **or any `pow` with a non-integer exponent** | **cvc5** | z3 |

The last clause does not mention dependence, and that is deliberate. A
non-integer `pow` exponent emits an **auxiliary-variable encoding**: the
script declares a fresh `aux` and asserts `aux^q = x^p`. `aux^q` is a
product of a fresh symbol with itself, so it is nonlinear whatever the
base is — a *constant* base does not make the script linear, and the
constant fold does not remove it (the value is generally irrational, so
there is nothing exact to fold to). Stamping such a slice `QF_LRA` shipped
`(* aux aux)` under a linear logic, and both backends refused it; that
read as two flaky solvers rather than as one wrong label.

There are no other fragments. "Primary" is **ordering, not selection**:
every installed backend runs on every fragment. Read off the stamp of a
linear obligation:

```text
solver: 2 invocation(s):
  [0] z3 5.0.0 (wheel-bindings (smt2 text)) options={':produce-models': 'true', ':timeout': '10000', 'set-logic': 'QF_LRA', …} — QF_LRA portfolio primary on assert #0
  [1] cvc5 1.3.4-modified (wheel-bindings (smt2 text; wall-guarded child process)) options={':produce-models': 'true', ':tlimit': '10000', 'set-logic': 'QF_LRA', …} — QF_LRA portfolio secondary on assert #0
```

and of a nonlinear one, where the order flips and cvc5 additionally gets
the coverings options:

```text
  [0] cvc5 1.3.4-modified (…) options={…, ':tlimit': '10000', ':nl-cov': 'true', ':nl-ext': 'none', 'set-logic': 'QF_NRA', …} — QF_NRA portfolio primary on assert #0
  [1] z3 5.0.0 (…) options={…, ':timeout': '10000', 'set-logic': 'QF_NRA', …} — QF_NRA portfolio secondary on assert #0
```

An `unsat` on the negated predicate discharges the obligation; a `sat`
becomes `REFUTED` only after its model survives an independent
exact-rational replay; `unknown` or a timeout is `UNKNOWN`, never
`VERIFIED`. A sat/unsat **disagreement** between the two backends raises
`SolverDisagreement` — it is a bug oracle, never a tiebreak.

## What each backend actually decided

Ten obligations, each run three times under the full portfolio and under
each backend alone. `unsat` = discharged, `sat` = refuted with a replayed
witness, `UNKNOWN` = the backend returned a timeout at 10 s.

| obligation | fragment | both | z3 alone | cvc5 alone |
|---|---|---|---|---|
| scalar, linear | `QF_LRA` | unsat, 78–112 ms | unsat, 8–9 ms | unsat, 71–84 ms |
| 64-element array, linear | `QF_LRA` | unsat, 86–91 ms | unsat, 10–12 ms | unsat, 77–87 ms |
| 8-element array, linear, false | `QF_LRA` | sat, 86–90 ms | sat, 11–13 ms | sat, 75–117 ms |
| 2 vars, degree 2 (AM–GM) | `QF_NRA` | unsat, 80–83 ms | unsat, 9 ms | unsat, 75–87 ms |
| 2 vars, degree 6 (Motzkin) | `QF_NRA` | unsat, 92–106 ms | unsat, 12–13 ms | unsat, 81–83 ms |
| 1 var, degree 3, false | `QF_NRA` | sat, 87–88 ms | sat, 11 ms | sat, 69–71 ms |
| 32 vars, 16 elementwise products | `QF_NRA` | unsat, ~10.3 s | **UNKNOWN** (timeout) | unsat, 166–175 ms |
| 64 vars, 32 elementwise products | `QF_NRA` | unsat, ~11.0 s | **UNKNOWN** (timeout) | unsat, 772–792 ms |
| 10-factor product chain | `QF_NRA` | unsat, ~8.1 s | unsat, 123–133 ms | unsat, 8.3–8.5 s |
| 12-factor product chain | `QF_NRA` | unsat, ~16.7 s | unsat, 689–702 ms | **UNKNOWN** (timeout) |

Three things this measured, and one it did not.

**1. On `QF_LRA`, both decided everything, and z3 decided it faster by an
order of magnitude.** Part of that gap is not solving at all: the cvc5
wheel is driven through a wall-guarded child process (its `tlimit` does not
reliably preempt the coverings solver), so every cvc5 invocation pays a
process spawn — visible as the ~70 ms floor on the cheapest rows, where z3
answers in 8.

**2. On `QF_NRA`, the split goes both ways, and that is the finding.**
Wide-and-shallow problems — many independent products, low degree per
variable — were decided by cvc5 in tenths of a second and *timed z3 out*.
Deep-and-narrow problems — one long product chain, high total degree in few
variables — were decided by z3 in tenths of a second and *timed cvc5 out*.
Neither backend dominates the other on the nonlinear fragment. There is no
"install this one" answer that survives both rows.

> **THIS IS THE ONE THAT DID NOT REPRODUCE, AND IT IS THE FINDING OF THE
> RE-MEASUREMENT.** The deep-and-narrow half held, harder than it says here.
> The wide-and-shallow half did not: on the most literal reading of *"32 vars,
> 16 elementwise products"* — two declared arrays of sixteen, sixteen
> elementwise products between them — the split ran the **other way**, z3
> discharging in about four seconds while cvc5 hit its wall guard. Two other
> readings of the same label showed **no split at all**. The label does not
> choose between them. See *"The same ten labels, re-measured here"* below;
> nothing there refutes this row, and that is exactly the problem — it cannot.

**3. A full portfolio is not the same as a full-portfolio answer.** On the
16-products row, both backends were installed, invoked, and stamped; only
cvc5 answered. The verdict was `VERIFIED` and said so itself:

```text
  PORTFOLIO DEGRADED — assert #0 was decided by ONE solver backend (cvc5 (wheel)), not the two the portfolio is designed around; the notes say which backend was lost and why
```

*(This one reproduced, as a mechanism. On the battery's reading of that row it
was **z3** that answered and cvc5 that did not, so the same disclosure comes
out with the two names swapped. WHICH backend gets lost is the part the row
label does not fix; that a lost backend is disclosed at all is the part that
does not depend on a harness, and that is what held.)*

**Not measured:** anything above 10 s, anything at `float32` or under
`semantics="ieee"` (which declines escalation wholly), the affine
refinement path (`refine="affine"`, which reduces what reaches a solver at
all), an external cvc5 binary via `STELLING_CVC5`, and any backend other
than the two wheels. The battery is ten small hand-written obligations
plus the declines listed below, not a corpus.

### The same ten labels, re-measured here, and what moved

`python tools/solver_battery.py --variants --repeats 3`, run in this tree on
**2026-08-22** at `stelling 0.2.0.dev0`, `jax 0.11.0`, `jax_enable_x64=True`,
z3 `5.0.0` (wheel), cvc5 `1.3.4` (wheel), CPU, Python 3.12, Linux x86-64, a
12th-gen i7-12850HX — **and a load average of 4.0**, because other work was
running on the box. These are not quiet-box seconds. The tool prints the load
average above its own table for exactly that reason, and every direction below
was reproduced by a second full run immediately before it at load 5.7 — the
seconds moved by up to 10% between the two, the answers by nothing. `solver_timeout_ms=10000`, three repeats per cell: the same
budget and the same repeat count as the table above.

**These are not the cells above, re-taken.** They are a different battery's
cells, on harnesses this page can point at. They are printed beside the table
above rather than substituted into it, because substituting them would file a
harness choice as a re-measurement of somebody else's harness — and the
`chose here` lines that `--rows` prints under every row are the reason that
would be a lie. `tools/solver_battery.py` is where these are maintained; this
is the reader's copy of them, and it will go stale the way a copy does.

| row (same label) | fragment | both | z3 alone | cvc5 alone |
|---|---|---|---|---|
| scalar, linear | `QF_LRA` | unsat, 73–80 ms | unsat, 6–9 ms | unsat, 71–78 ms |
| 64-element array, linear | `QF_LRA` | unsat, 81–91 ms | unsat, 9–13 ms | unsat, 74–88 ms |
| 8-element array, linear, false | `QF_LRA` | sat, 75–81 ms | sat, 8–10 ms | sat, 64–73 ms |
| 2 vars, degree 2 (AM–GM) | `QF_NRA` | unsat, 75–77 ms | unsat, 8–9 ms | unsat, 64–72 ms |
| 2 vars, degree 6 (Motzkin) | `QF_NRA` | unsat, 75–85 ms | unsat, 9–14 ms | unsat, 70–79 ms |
| 1 var, degree 3, false | `QF_NRA` | sat, 78–83 ms | sat, 7–8 ms | sat, 67–71 ms |
| 32 vars, 16 elementwise products ‡ | `QF_NRA` | unsat, 20.2–20.6 s | unsat, 4.4–4.6 s | **UNKNOWN** (16.0 s wall) |
| 64 vars, 32 elementwise products ‡ | `QF_NRA` | **UNKNOWN**, 26.1–26.2 s | **UNKNOWN**, 10.1–10.2 s | **UNKNOWN** (16.0 s wall) |
| 10-factor product chain | `QF_NRA` | unsat, 2.3–2.5 s | unsat, 6 ms | unsat, 2.3 s |
| 12-factor product chain | `QF_NRA` | unsat, 16.0 s | unsat, 6 ms | **UNKNOWN** (16.0 s wall) |

‡ **no cell in these two rows may be read against the row above it.** Their
labels admit readings that disagree about which backend finishes; the numbers
here are one reading's and the battery marks them rather than publishing them
as the row.

**What held.**

- **The `fragment` column: all ten, exactly.** It is the one column of that
  table that is a property of this tree rather than of a machine, and it now
  has a gate that runs with no backend installed.
- Every row's declared input count, too: *32 vars* is 32 declared inputs in
  the emitted slice, *64-element array* is 64, and so on for all ten.
- **Finding 1** — on `QF_LRA` both backends decided everything, and z3 was
  faster on every linear row, by 8.0x to 11.8x. The order-of-magnitude claim
  holds, with the same caveat the page already makes about the subprocess
  spawn in cvc5's floor.
- **Finding 3** — a full portfolio is not a full-portfolio answer. Rows 7 and 10
  were decided with both backends installed, invoked and stamped and only one
  answering, and the verdict said `PORTFOLIO DEGRADED` itself.
- **The deep-and-narrow half of finding 2**, and more sharply than the table
  above puts it: on this battery's reading of the chain rows, z3 discharged
  both in single-digit milliseconds while cvc5 took seconds at ten factors and
  did not finish at twelve.

**WHAT DID NOT HOLD: the wide-and-shallow half of finding 2 — and the reason
matters more than the result.** Three defensible readings of *"32 vars, 16
elementwise products"* were built and driven (`--variants`), three repeats
each:

| reading of the label | z3 alone | cvc5 alone |
|---|---|---|
| `sum(a*b) <= sum(a)`, `a,b ∈ [0,1]¹⁶` — the literal one: sixteen products, thirty-two variables | unsat, 4.4–4.6 s | **UNKNOWN** (16.0 s wall) |
| `sum(a² + b² − 2ab) >= 0`, `a,b ∈ [−1,1]¹⁶` | unsat, 22 ms | unsat, 169–182 ms |
| `sum(a*b) − sum(b*a) >= 0`, `a,b ∈ [−1,1]¹⁶` | unsat, 6–9 ms | unsat, 69–81 ms |
| **the row above, as published** | **UNKNOWN** (timeout) | unsat, 166–175 ms |

The first **reverses** this page's direction. The second reproduces the
page's cvc5 cell to within a few milliseconds and still does not reproduce its
z3 cell — which is the cleanest statement of the problem there is: **matching
one cell is not identifying a harness.** The third shows no split either.

So the honest reading is not *"the page's row 7 is wrong"*. It is that **row 7
is not reconstructible**, and no number produced by any of these three
harnesses may be filed against it. The battery marks rows 7 and 8 `‡`, prints
what its own reading measured, and refuses to file any of it against the
published row — that refusal is the finding, not the four seconds.

The chain rows are underdetermined in the same way, though less violently. A
*"10-factor product chain"* can be ten variables or one variable ten times,
and `x¹⁰ >= 0` shows no split at all: z3 6–7 ms against cvc5 71–73 ms. The
direction above survived one reading and vanished under another; it did not
reverse under either, which is why rows 9 and 10 carry no `‡`.

**One thing the re-measurement recovered rather than chose.** The page never
says *what* it timed, and that was the one open parameter that did not have to
be picked — it can be read back out of the page's own arithmetic. Its `both`
cells are the **sum** of its own two single-backend cells:

- row 10 — cvc5's wall guard (16.0 s, measured here) plus z3's 689–702 ms is
  **16.69–16.70 s**, against a published `~16.7 s`;
- row 7 — z3's timeout (10.0–10.1 s) plus cvc5's 166–175 ms is **10.2–10.3 s**,
  against a published `~10.3 s`;
- row 8 — the same timeout plus 772–792 ms is **10.8–10.9 s**, against a
  published `~11.0 s`;
- row 9 is the one that does not fit: 123–133 ms plus 8.3–8.5 s is
  **8.4–8.6 s**, against a published `~8.1 s`.

Three of four land inside 0.2 s and one is 0.3–0.5 s out, which is enough to
fix the definition and not enough to pretend it is certain. So the battery
sums the per-invocation milliseconds stelling publishes in the verdict's notes
— *not* the `check()` wall, which on a discharged row also pays the vacuity
widen re-check and would double every one of them.

**And one thing neither table says out loud: `solver_timeout_ms=10000` is not
a ten-second wall.** z3's `:timeout` is its own and lands near it — 10.1 s
measured. The cvc5 wheel is driven through a wall-guarded child process
because its `tlimit` does not reliably preempt the coverings solver, and that
guard is `timeout * 1.5 + 1 s`: **16.0 s for a ten-second budget**, measured
here on three rows. A two-backend obligation on which both backends time out
therefore costs about **26 s**, not 20 — which is what row 8 cost above. If
you are sizing a CI budget from the ten seconds in this page's provenance
paragraph, you are out by more than half. The battery prints every cell that
ran past its budget, with the backend named.

## What one backend alone costs you

Nothing about a verdict's *status* changes: every obligation in the table
that one backend could decide, that backend still decided alone, with the
same `unsat` / `sat` and the same replay of any witness. What changes is
disclosed in three places at once.

On the obligation's own detail line:

```text
  assert #0: discharged — discharged by solver escalation (QF_LRA): the box with the negated predicate is unsat per z3 (wheel) [PORTFOLIO DEGRADED: 1 of 2 backends answered; a discharge has no replay backstop]
```

In the notes, naming *which* backend was lost and *why* — "is not
installed" when it is absent, and a different phrase when a caller
restricted the portfolio, because rendering a configured restriction as a
missing dependency would send you to install something you already have:

```text
note: assert #0: portfolio degraded — only z3 (wheel) ran (cvc5 is not installed)
note: assert #0: and this is the direction with no backstop: a discharge is a universal claim over the whole declared box, so nothing downstream re-derives it the way exact-rational replay re-derives a witness. The second backend was the only independent check on this obligation and it did not answer
```

And in `Verdict.solver_redundancy`, which is the machine-readable form —
`((0, ('z3 (wheel)',)),)` for the run above, against
`((0, ('z3 (wheel)', 'cvc5 (wheel)')),)` for the two-backend run. A CI job
that cares reads it directly:

```text
one_backend = [i for i, who in v.solver_redundancy if len(who) < 2]
```

The asymmetry that note names is the reason this matters more than it
looks. A `sat` reaches `REFUTED` only through an exact-rational replay that
shares no code with the solver, so a lost second backend there costs a
cross-check that something else still performs. An `unsat` is a universal
claim over the whole declared box: nothing re-derives it, and **the second
backend is the only independent check there is.** That is the thing a
one-backend install gives up, and it gives it up on exactly the verdicts
you would most want to trust.

With **neither** backend installed nothing silently degrades either — every
escalated obligation stays `UNKNOWN` and the verdict carries:

```text
note: no SMT solver is installed — pip install "stelling[solvers]" (or set STELLING_CVC5 / put cvc5 on PATH) to enable escalation
```

## If you are installing exactly one

Pick by the shape of your obligations, then re-measure on your own —
`python tools/solver_battery.py` is where to start, and its `--rows` output
is a demonstration of how much of a row a shape description leaves open:

- **Mostly linear** — sums, scalings by constants, moving averages,
  concatenations, comparisons against thresholds: `[z3]`. It is the
  `QF_LRA` primary, it decided every linear obligation here, and it did so
  without paying a subprocess spawn per call.
- **Mostly polynomial** — products of two declared arrays, squares,
  `integer_pow`, division by a declared value: `[cvc5]`. It is the
  `QF_NRA` primary, and the wide nonlinear rows above are the ones z3
  could not finish. **This is the weakest bullet on the page, and the
  re-measurement is why**: those two rows are the ones a reader cannot
  reconstruct, and on the most literal reading of their labels the split ran
  the other way — z3 discharged and cvc5 did not finish. Read this as an
  argument for measuring your own obligations, not as one for uninstalling z3.
- **You do not know, or it is CI**: `[solvers]`. Both nonlinear failure
  directions are real, and CI is exactly where you cannot afford to
  discover which one you have.

**There is a fourth option this page used to omit, and it is usually the
right one: install both and RESTRICT.** `check(harness, …, solver="z3")`
gives you a single-backend run out of a two-backend install, per call — so
you can take z3's speed on a linear obligation and still have cvc5 there for
the nonlinear one, without a second environment. The verdict says which
backend was excluded and why, so a restricted run is never confused with a
missing install. Uninstalling is the irreversible version of the same
decision, taken once for every obligation you will ever write.

## What no backend can reach

Some obligations never reach a solver at all, and *which* backend you
installed makes no difference to any of them — escalation declines before
any invocation, with zero stamps and the reason quoted in the verdict. All
of these were measured; the text is what the tool printed:

| what | quoted decline |
|---|---|
| a primitive with no SMT emission row (`exp`, `log`, `sqrt`, …) | `primitive 'exp' is outside the supported emission set: no SMT emission rule has been built and audited for it — an unbuilt row, not a policy refusal of the form` |
| division whose divisor spans 0 | `'div': divisor may be zero over the declared box — SMT-LIB2 division is underspecified at 0` |
| an integer-dtype computation | `'add' on dtype 'int32': jax integer arithmetic wraps on overflow and SMT-LIB2 Reals are unbounded, so a Real emission does not model it` |
| `dot_general` with two symbolic operands | `'dot_general' has NO constant operand: a sum of products of two symbolic operands is NONLINEAR arithmetic, outside this row's linear scope` |
| an obligation over the emission budget (512 element terms) | `obligation not attempted: it needs 1024 element terms and 256 root conjuncts, and its element terms put it over the per-obligation emission budget of 512` |
| an `assert_` written inside a `jax.jit` helper, a `cond` branch, or a `scan`/`while_loop` body | `the assert this obligation was recorded from is not a top-level equation of the query (it sits inside a sub-jaxpr — a transparent call body, a cond branch, or an undescended scan/while body), and escalation slices top-level asserts only` |
| a `dot_general` whose operands' contracted (or batch) extents disagree — loadable through `from_dict`, never traceable, since jax refuses it | `'dot_general' declined: dot_general contracted dims disagree: lhs[0]=2 vs rhs[0]=4` |
| a propagation that constrained an assume | escalation is refused wholly, before backend discovery |

**And one case that is NOT in that table, because it is not a decline.**
`semantics="ieee"` together with `solver_timeout_ms` is a **caller error**: it
raises at `check()` and there is no verdict to quote a reason from.

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from stelling.harness import any_array, assert_
from stelling.preconditions import check


def square_under_bound():
    x = any_array((), jnp.float64, (1.0, 2.0))
    return assert_(x * x <= 4.0)


try:
    check(square_under_bound, vacuity_mode="inputs-only",
          semantics="ieee", solver_timeout_ms=5_000)
    print("returned a verdict")
except ValueError as e:
    print("ValueError:", e)
```

```
ValueError: solver_timeout_ms and semantics='ieee' are contradictory: the SMT backends emit over the reals (QF_LRA/QF_NRA) and cannot model format-specific rounding or overflow. Remove solver_timeout_ms for ieee mode, or use semantics='real' for solver escalation.
```

It used to share the last row above with the constrained-assume case, under a
caption promising a verdict line for every entry. The two behave categorically
differently — one is a decline inside a verdict, the other never reaches one —
so they are separated here.

Installing the other backend does not move any of these. If your
`UNKNOWN`s look like this, a solver extra is not what you are missing.

**Every row above the last is ONE obligation's decline**, and the nested-
assert row is the one where that had to be fixed: until audit 0.2.0 M17 a
single `assert_` inside a `jit` declined escalation for *every* obligation
in the query, so a set of asserts that each verified alone came back
UNKNOWN together. That was widely mistaken for the element budget, which
was always strictly per-obligation. If you are reading an UNKNOWN on a
multi-assert query, read the per-obligation `detail`: each one now names
its own cause.

## The cvc5 wheel, verified

The `cvc5` extra installs the official PyPI wheel — the non-GPL "BSD
version" build. Two properties of it were checked directly against
cvc5 1.3.4 in this environment rather than taken from its documentation:

- **libpoly is bundled and `nl-cov` works.** A `QF_NRA` unsat that needs
  cylindrical algebraic coverings was solved through the wheel's own
  SMT-LIB2 parser with `nl-cov=true, nl-ext=none` — the exact options
  stelling emits for that fragment — and answered `unsat`. That is
  functional evidence the coverings solver is present and running, not a
  reading of the build configuration, which this wheel does not expose.
- **CoCoALib is absent**, and cvc5 says so itself: requesting
  `nl-cov-lift=lazard` printed `nl-cov::LazardEvaluation is disabled
  because CoCoA is not available. Falling back to regular calculation of
  infeasible regions.` — and the query still solved, on the fallback path.

The other GPL-gated performance components (CLN, glpk-cut-log) are absent
for the same licensing reason; that half is reported from the build's
licensing, not probed here. Only a source build with `./configure.sh --gpl
--auto-download` has them, and stelling can be pointed at one with
`STELLING_CVC5` — see the cvc5 section of the
[README](https://github.com/NicholasEhsanRoy/stelling/blob/main/README.md#cvc5-wheel-vs-external-binary).

None of the nonlinear rows in the table above were re-run against a GPL
build, so nothing here says what those components would be worth.

## z3 and high-numerator rational pow (automatic workaround)

Since 0.2.0, stelling detects rational-pow auxiliary variables in the emitted
script and switches z3 to a custom tactic chain:

```
simplify -> solve-eqs -> factor(num_primes=4) -> purify-arith -> tseitin-cnf -> nlsat
```

It fires only when the script contains `(declare-const aux_...)` declarations
— the marker for rational-pow auxiliary variables. All other scripts use z3's
default `Solver()` unchanged, and the tactic needs no configuration.

**RETRACTION, 2026-08-20.** Until that date this section argued from
`x**(1/80)` and a *"degree-80 factoring pathology"*, with two figures —
*"measured: >10s at degree 80"* and *"measured: 0.35-0.6s at degree 80"*.
**The emission cannot produce degree 80**, so neither figure can have been
taken there. A binary64 exponent's exact value is dyadic, so
`stelling.obligation.pow_exponent_rational` returns `p/q` in lowest terms with
`q` a power of two; under `RATIONAL_POW_DEGREE_CAP = 128` the admitted degrees
are the odd values below 128 together with 2, 4, 8, 16, 32, 64 and 128, and 80
is in none of them. The worked case declines before any solver sees it, and — unlike everything
above it on this page — that is re-derived on every test run rather than
quoted:

```python
from fractions import Fraction

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from stelling.harness import any_array, assert_
from stelling.obligation import RATIONAL_POW_DEGREE_CAP
from stelling.preconditions import check


def eightieth_root():
    x = any_array((), jnp.float64, (1.0, 100.0))
    return assert_(x ** (1 / 80) >= 1.0)


print("exact value of the exponent :", Fraction(1 / 80))
print("emission cap                :", RATIONAL_POW_DEGREE_CAP)

verdict = check(eightieth_root, vacuity_mode="inputs-only",
                solver_timeout_ms=10_000)
reason = next(n for n in verdict.notes if "escalation declined" in n)
print("status                      :", verdict.status)
print("declined over the cap       :",
      f"over the emission cap {RATIONAL_POW_DEGREE_CAP}" in reason)
print("degree it would have needed :",
      reason.split("would be degree ")[1].split(",")[0])
print("backends invoked            :", len(verdict.solver_redundancy))
```

```
exact value of the exponent : 3602879701896397/288230376151711744
emission cap                : 128
status                      : UNKNOWN
declined over the cap       : True
degree it would have needed : 288230376151711744
backends invoked            : 0
```

`288230376151711744` is `2**58` — the exponent's exact denominator — and
**zero backends were invoked**, which is the whole of the retraction: no
figure can be measured at a degree the emission declines to produce.

The integer branch cannot reach it either: it expands inline, declares no
`aux_`, and 80 is over `INTEGER_POW_EXPANSION_CAP = 64` regardless.

**What the tactic actually buys, and what it costs.** The family it reaches is
the high-numerator `q=128` one. Re-measured 2026-08-20 on z3 5.0.0 over the
five pairs `obligation.py`'s cost table names, all five `unsat` in both modes
at a 120 s script timeout — default → tactic:

| pair | default | tactic |
|---|---|---|
| `1/128` | 0.31 s | 0.38 s |
| `127/2` | 0.04 s | 0.04 s |
| `127/128` | 50.03 s | **21.39 s** |
| `113/128` | 50.01 s | **18.23 s** |
| `105/128` (worst of the 448) | 50.01 s | **69.80 s** |
| **total** | 150.39 s | **109.84 s** |

**It is kept for the NET, not for any one row.** It roughly halves the two
expensive `q=128` rows and **loses about 20 s on `105/128`**, and a string
sniff on `aux_` cannot tell those rows apart. Seconds are machine-specific and
were taken on a loaded box, so the ORDERING is the content, not the digits.
The same figures and the same reasoning are at `src/stelling/solvers.py`'s
`_run_z3`, which is where they are maintained; this section is the reader's
copy of them.

**It does NOT restore full redundancy on the worst reachable cases.** At this
page's own 10 s budget, `x**(105/128)` and `x**(127/128)` both still time z3
out. Driven here, `x ∈ [1, 100]`, `solver_timeout_ms=10000`: z3 timed out at
10.0 s on both, and on this machine cvc5 timed out too, so both obligations
came back UNKNOWN with no backend deciding. A longer budget is what decides
them, and depending on which backend answers first you get either a full
cross-check or a `PORTFOLIO DEGRADED` single-backend discharge — not the "full
redundancy, and the verdict discloses no difference" this section used to
promise.

**The tactic is not gated on the z3 version, deliberately.** z3 5.1 fixes the
degree-80 factoring pathology upstream — the case that cannot be reached — and
does not touch the `q=128` family, so switching it off on 5.1 would be a
regression. z3 5.1 is not installed here; that column is cited from the
campaign measurement of 2026-08-18, not re-measured. Neither the version nor
the mode has moved an ANSWER on this family in any measured cell — only a
time.

## Related

- [Reading a verdict](reading-a-verdict.md) — the stamp's solver lines and
  the degraded-portfolio disclosure in context
- [Quickstart](quickstart.md) — where `solver_timeout_ms` enters
- [SOUNDNESS.md](https://github.com/NicholasEhsanRoy/stelling/blob/main/SOUNDNESS.md)
  — why a solver is never invoked on defaults, and why the option set is
  stamped
