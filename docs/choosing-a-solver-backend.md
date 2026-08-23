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
quoted declines. The 2026-08-22 driving reported further down carries its own,
taken on a different day and naming the CPU and the load average, neither of
which this paragraph ever did. All the original figures are from
`stelling 0.2.0.dev0`,
`jax 0.11.0`, CPU, `jax_enable_x64=True`, z3 `5.0.0` (wheel) and cvc5
`1.3.4-modified` (wheel), Linux x86-64, Python 3.12,
`solver_timeout_ms=10000`, three repeats per cell. Single-backend
configurations were produced by restricting the portfolio and by hiding a
backend from `stelling._optional.available` — nothing was uninstalled, and
both routes produce the same degraded-portfolio disclosure (below). Wall
times are one machine's; the *decided / timed-out* column is the part that
carries the argument, and even that is a timeout budget away from moving.

**THE HARNESSES BEHIND THE TEN-ROW TABLE WERE NEVER COMMITTED.** That is
still true and is not repairable by editing. What used to follow it here was
this: *"`grep` for `Motzkin`, `AM–GM`, `product chain` or `elementwise
products` across the whole repository still returns this file and nothing
else."* **THAT GREP HAS HITS NOW.** All four terms return
`tools/solver_battery.py` — the file this section introduces two paragraphs
later — along with this page and the committed run transcripts beside it;
`Motzkin` and `AM–GM` additionally return the test module and the sweeps,
while `product chain` and `elementwise products` do not, because what is
gated about those two rows is their labels and their refusals rather than a
named object. The sentence is kept above as the history it is
rather than repaired into the present tense: it records what was true when the
admission was made, and what stopped being true is the subject of everything
below it.

**The other half of that admission was FALSE, which is worse than stale.** It
said *"not one of these labels fixes a declared box, a predicate or a
threshold"*. **Two of them fix the predicate exactly.** Row 4 names AM–GM,
whose two-variable degree-2 form is `x² + y² ≥ 2xy`; row 5 names the Motzkin
polynomial, `x⁴y² + x²y⁴ − 3x²y² + 1`. A named mathematical object is
not a choice. On those two rows the only thing the label leaves open is the
box — and, for Motzkin, how the degree-6 monomials associate, which is the
same polynomial and a different emitted script.

**And the question that admission asked was the wrong one.** *"Does the label
pin a harness"* is answered no by every row label ever written, which is why
answering it graded all ten of these the same and made row 4 read as weakly as
row 7. What decides whether a published cell can be re-derived is
**whether the freedom the label leaves REACHES the published number** — and
that is a measurable question. Measured (`tools/solver_battery.py --rows` for
the parameters — it ships, and prints every grade below on your own machine —
and two sweeps under `scratchpad/`, which is tracked in the repository and is
not in the sdist, for the driving), the ten rows fall
into four groups, and the table below carries the answer in its own
`reconstruction` column rather than one disclaimer repeated ten times:

- **rows 1, 3, 4, 6 — reconstructed.** Every label-compatible reading gives
  the published outcome, and neither column moves: driven from a box of `±1`
  out to the widest a float64 declaration can express, cvc5 stays at
  55–71 ms and z3 at 3–7 ms, no further than three repeats of one unchanged
  harness move them.
- **rows 2, 5 — outcome only.** The outcome is forced the same way. The
  millisecond is not: these labels bound the declared box nowhere, and the box
  reaches the cell. Row 5's z3 column climbs with the box, from 5–6 ms at `±1`
  to 126–140 ms at the top of the format — a factor of twenty-two to
  twenty-eight, against the `1.57x` that three repeats of one unchanged
  harness move it. Row 2's does the same, more gently: 4–5 ms to 11–13.
- **rows 9, 10 — direction only.** *"A 10-factor product chain"* can be ten
  variables or one variable ten times. One reading reproduces the published
  split; another shows no split at all. Neither reverses it.
- **rows 7, 8 — unsupported.** No reading of either label reproduces its
  published split, and every reading that splits at all puts the backend this
  page has **losing** ahead of the one it has winning. See the retraction
  under the table.

**Those four words are defined in one place, not two.** The table below is
`tools/solver_battery.py`'s `GRADES` dictionary quoted verbatim — the same
strings the tool prints under `--rows` and carries in `--json` — because a
page and a tool that each state what a grade means in their own words are two
records that can drift, and the strongest sentence here is the one saying what
the strongest grade does NOT claim.
`test_the_page_states_every_grade_s_meaning_in_the_tool_s_own_words` holds
them equal in both directions.

| grade | rows | what it licenses a reader to do, and what it does not |
|---|---|---|
| `reconstructed` | 1, 3, 4, 6 | the freedom the label leaves does not reach the published number, measured over every box a float64 declaration can express — from +/-1 to the format's own ceiling, not the +/-100 an earlier sweep stopped at without saying so. Every label-compatible reading gave the published OUTCOME, and neither column moved further than three repeats of one unchanged harness move it. `reconstructed` does NOT mean `this battery reproduced the published milliseconds` — it means the label's remaining freedom is not what separates them: no choice a reader makes inside the label would have moved them |
| `outcome only` | 2, 5 | the published OUTCOME is forced by the obligation and every label-compatible reading gives it — but the label bounds the declared box nowhere, and the box REACHES the published millisecond. Driven out to the widest box float64 can declare, this row's z3 cell climbs monotonically with the box and past anything three repeats of one unchanged harness produce. The outcome is reconstructed; the millisecond is the reading's, and the label does not fix the reading |
| `direction only` | 9, 10 | the readings agree on WHICH BACKEND WINS wherever they split, and none reverses — but at least one shows no split at all, so the direction survives and the seconds do not |
| `unsupported` | 7, 8 | no reading reproduces the published split, and the readings do not agree with each other about which backend wins: at least one puts the backend this page has LOSING ahead of the one it has winning. Unsupported, which is not the same as wrong: nothing here refutes the published cell, and that is exactly the problem — nothing here CAN |

**Until 2026-08-23 this list said *rows 1–6 — reconstructed*, and that was a
universal quantified over a family that had been bounded without saying so.**
The sweep behind it caps rows 4 and 5 at a box of `±100` — nothing on this
page, in that sweep or in the tool stated the cap, and the labels supply no
bound at all: AM–GM holds over the whole of `R²` and the Motzkin polynomial is
nonnegative over the whole of `R²`, as row 4's own entry in `--rows` says.
Widening the sweep and re-asserting the universal would not have fixed it,
because the next box is always wider. What fixed it is that **these harnesses
declare a float64 box**, so the format supplies the one ceiling here that is
not a judgement call — and driven out to it, two of the six rows fail. They
are regraded above rather than re-argued.

**And half of that list's supporting argument was false.** It read: *their
cells sit on two floors this page itself names, the cvc5 spawn (~70 ms) and
z3's ~10 ms, and a cell that IS the floor cannot be moved by a choice made
above it.* True of cvc5 — measured, cvc5 is 55–71 ms on all six rows at every
declarable box, which is the wall-guarded subprocess spawn and nothing above
it moves it. **False of z3**, which has no floor of that kind: 1–3 ms on row 1
in one sweep of this battery, 126–140 ms on row 5 at the top of the format.
Row 5's ~10 ms was never a floor. It was the price of that predicate at that
box, and the box is exactly what the label leaves open.

Read the ten rows for their *direction*, not their milliseconds — that
instruction stands for all ten. For rows 1, 3, 4 and 6 it now understates what
is there. For rows 7 and 8 it is not enough on its own, and the retraction
under the table says why.

**What HAS changed is that there is now a battery you can run.**
`tools/solver_battery.py` ships in the sdist (`/tools` is in the allowlist),
drives ten harnesses BUILT TO those labels, and prints — for every row — the
parameters the label leaves open and the ones it therefore had to choose:

- `python tools/solver_battery.py --rows` is the inventory: per row, what the
  label fixes, what this battery picked, and which of the four reconstruction
  grades above the row is in. It needs neither jax nor a solver, because it is
  data.
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

**It is deliberately NOT a re-derivation of the table below wherever a
re-derivation would be a fiction — and it says which rows those are, on every
run.** The tool prints its own cells and the page's side by side and never
edits one into the other, because they came from different harnesses. For rows
7 and 8 that separation IS the finding. For rows 1, 3, 4 and 6 it says more
than it looks: those harnesses were built to the labels, what the labels left
open was swept out to the widest box the format can declare, and every reading
landed in the same place — which is what a reconstruction is. For rows 2 and 5
the same sweep is what stops them being graded that way. Making an
unsupported row look reproducible would be the worse defect: an unreproducible
table is at least honest about being
unreproducible, while an invented harness *looks* reproducible.

**WHAT IS GATED ON THIS PAGE, AND WHAT IS DELIBERATELY NOT.** Two `python`
blocks are executed and their output compared byte for byte by
`tests/test_doc_examples.py`: the `x**(1/80)` degree-cap decline and the
`semantics="ieee"` caller error. They are the two blocks that make claims
about mechanism rather than speed.

More of the ten-row table is now held, by `tests/test_solver_battery.py`,
and the cut it makes is the same one — mechanism yes, clock no:

| held | by | needs |
|---|---|---|
| the table's ten row labels, their order, their `fragment` cells, all three of their published cells and their `reconstruction` grades, against `tools/solver_battery.py`'s copy of them | `test_the_battery_is_the_page_s_table_row_for_row` | nothing |
| that the rows this page marks `unsupported` are exactly the rows the tool refuses to file a number against — neither side may quietly drop a mark the other keeps | `test_the_page_s_marks_are_the_tool_s_refusals` | nothing |
| every row of the battery still declaring which parameters the label left open | `test_every_row_says_what_the_page_left_open` | nothing |
| **the two named polynomials, coefficient by coefficient** — `x² + y² ≥ 2xy` and `x⁴y² + x²y⁴ − 3x²y² + 1`, against a reference written from their published definitions, on an exact grid | `test_the_named_objects_are_the_objects_their_labels_name` | nothing |
| the three-readings table below: that its last row still carries row 7's published cells verbatim, and that the readings above it still show the reversal and the two non-splits the prose claims | `test_the_three_readings_table_still_shows_what_the_prose_says` | nothing |
| **what each grade MEANS** — the four `GRADES` strings, this page's table of them against the tool's dictionary, byte for byte, including the sentence saying what `reconstructed` does *not* claim | `test_the_page_states_every_grade_s_meaning_in_the_tool_s_own_words` | nothing |
| that every measured figure this page quotes from a committed transcript is *in* that transcript — the check that stops a re-typed digit drifting away from the run it is attributed to | `test_every_quoted_sweep_figure_is_in_the_transcript_it_is_attributed_to` | nothing |
| that every `solvers.<name>` this page and the tool cite is a name `stelling.solvers` actually has | `test_every_solver_symbol_these_two_cite_exists` | nothing |
| the `fragment` column, re-derived from the traced jaxpr | `test_the_fragment_column_is_what_the_page_publishes` | jax, **no backend** |
| *"32 vars"*, *"64-element array"*, *"2 vars"* — every row's declared input count, read off the emitted slice rather than off its own label | `test_each_row_declares_the_variables_its_label_names` | jax, **no backend** |
| that every harness still ESCALATES — an obligation interval propagation decides never reaches a backend, so it cannot be a row of a solver comparison at all | `test_no_row_is_decided_before_a_backend_is_asked` | jax, **no backend** |
| the six cheap rows' `unsat`/`sat` — a fact about the obligation, not about a backend's speed | `test_the_cheap_rows_answer_what_their_obligation_forces` | jax and a backend |

**The fourth row of that table is there because the strongest claim on this
page had nothing behind it.** Rows 4 and 5 are the only two whose label pins a
predicate, which makes the polynomials a claim rather than a choice — and a
polynomial cannot be gated through the measurement. `x² + y² ≥ 1.5xy` is still
true, still degree 2, still two variables, still `QF_NRA`, still `unsat`;
Motzkin with `−2` in place of `−3` is still nonnegative, still degree 6, still
two variables, still `unsat`. Both mutations were applied and both left every
column of every table, every verdict and every other gate byte-identical. Only
the coefficients can tell them apart, so the coefficients are what is checked.

**Not one millisecond on this page is gated, and that is a decision rather
than an omission.** A wall time is not a property of this tree; a gate on one
would go red when the box is busy and green when the claim is wrong. Rows 7
to 10's published cells are *"did this backend finish inside ten seconds"*,
which is a millisecond wearing a hat, so they are not gated either — they are
re-measured on demand by the battery instead, and the reason is recorded in
`tests/test_doc_examples.py`'s `BLIND_SPOT` entry for this page.

Be exact about what "gated" covers, because the two things are easy to
conflate. The published cells below **are pinned** — to
`tools/solver_battery.py`'s copy of them, so the page and the tool cannot
become two disagreeing records of one hand-check — but **nothing asserts that
any of them is true**, and nothing re-measures them. What is held as a CLAIM
is every non-numeric thing in those tables: labels, order, fragments, grades,
marks, and the DIRECTION pattern of the three-readings table — because *"z3
finished and cvc5 did not"* is a claim this page makes in prose, and a table
that stopped showing it would make the prose false without making any number
wrong. The routing figures and the quoted declines below remain hand-checked
and will go stale the way hand-checks do.

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

The **five leftmost columns are the 2026-08 hand-check as published**, byte
for byte; nothing below has been edited into them. The `reconstruction` column
was added on **2026-08-23** and is about the ROWS, not about the backends —
it says what the driving described further down was able to settle about each
one. `tools/solver_battery.py` carries the same partition and
`tests/test_solver_battery.py` refuses to let the two spellings drift.

| obligation | fragment | both | z3 alone | cvc5 alone | reconstruction (2026-08-23) |
|---|---|---|---|---|---|
| scalar, linear | `QF_LRA` | unsat, 78–112 ms | unsat, 8–9 ms | unsat, 71–84 ms | reconstructed |
| 64-element array, linear | `QF_LRA` | unsat, 86–91 ms | unsat, 10–12 ms | unsat, 77–87 ms | outcome only |
| 8-element array, linear, false | `QF_LRA` | sat, 86–90 ms | sat, 11–13 ms | sat, 75–117 ms | reconstructed |
| 2 vars, degree 2 (AM–GM) | `QF_NRA` | unsat, 80–83 ms | unsat, 9 ms | unsat, 75–87 ms | reconstructed |
| 2 vars, degree 6 (Motzkin) | `QF_NRA` | unsat, 92–106 ms | unsat, 12–13 ms | unsat, 81–83 ms | outcome only |
| 1 var, degree 3, false | `QF_NRA` | sat, 87–88 ms | sat, 11 ms | sat, 69–71 ms | reconstructed |
| 32 vars, 16 elementwise products | `QF_NRA` | unsat, ~10.3 s | **UNKNOWN** (timeout) | unsat, 166–175 ms | unsupported ‡ |
| 64 vars, 32 elementwise products | `QF_NRA` | unsat, ~11.0 s | **UNKNOWN** (timeout) | unsat, 772–792 ms | unsupported ‡ |
| 10-factor product chain | `QF_NRA` | unsat, ~8.1 s | unsat, 123–133 ms | unsat, 8.3–8.5 s | direction only † |
| 12-factor product chain | `QF_NRA` | unsat, ~16.7 s | unsat, 689–702 ms | **UNKNOWN** (timeout) | direction only |

‡ **RETRACTION, 2026-08-23 — rows 7 and 8 are UNSUPPORTED.** That asserts less
than *wrong*, deliberately, and the difference is the whole point. Three
defensible readings of *"32 vars, 16 elementwise products"* were built and
driven; **one of them runs this row's direction backwards** and two show no
split at all. Nothing in that driving refutes the published cells — and that
is exactly the problem, because nothing in it CAN. **No number produced by any
reading may be filed against these two rows**, here or anywhere, and
`tools/solver_battery.py` refuses to file its own. The readings are in the
table further down.

**Row 8's evidence is its own, and it is NOT row 7's** — the tool asserted it
was until 2026-08-23, and this batch's own scratch README said the opposite in
the same breath. Driven at row 8's width, the three readings do this: the
literal one has **neither** backend finishing (z3 `UNKNOWN` at 10.1 s, cvc5
`UNKNOWN` at its 16.0 s wall guard), and the other two have **both** finishing,
with z3 six to twenty-two times ahead of cvc5. So **no reading of row 8 reverses
it** in row 7's sense; what they do is fail to reproduce its published split
while putting z3 — the backend this row has timing out — ahead every time one
of them splits at all. That meets this page's criterion for `unsupported`, and
the criterion had to be restated to say what it actually rules on. Row 8's
grade is now derived from those recorded directions rather than asserted beside
them.

† **Row 9 additionally carries a defect in its OWN arithmetic, and it is not a
rounding mismatch.** Its published `both` (`~8.1 s`) is **below** its published
`cvc5 alone` (`8.3–8.5 s`). `solvers._dispatch_obligation` — the function
`solvers.escalate` reaches for each obligation — runs the admitted backends in
a sequential loop with no short-circuit, and this page says the same in words a
few paragraphs up, *"every installed backend runs on every fragment"* — so a
two-backend run cannot finish faster than one of its own backends, under any
definition of what was timed. See *"What the `both` column actually is"* below.

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
> DRIVING BELOW.** The deep-and-narrow half held, harder than it says here.
> The wide-and-shallow half did not: on the most literal reading of *"32 vars,
> 16 elementwise products"* — two declared arrays of sixteen, sixteen
> elementwise products between them — the split ran the **other way**, z3
> discharging in about four seconds while cvc5 hit its wall guard. Two other
> readings of the same label showed **no split at all**. The label does not
> choose between them. See *"The same ten labels, driven here"* below;
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

### The same ten labels, driven here, and what that settled

`python tools/solver_battery.py --variants --repeats 3` drives ten harnesses
BUILT TO the ten labels above and prints its own cells beside the published
ones, never into them. **The milliseconds stay in the tool.** This section
used to carry a full ten-row hand-copy of them, and that copy was a second,
ungated, drifting record of exactly the thing this page exists to complain
about. It behaved like one: setting one of its cells to `999–999 ms` left the
whole module — thirty tests at the time — green; its only rows that were not
near-duplicates of the table above were the two rows it forbade the reader to
compare; and it marked row 7 `‡` while the table above — the one the three
findings are actually written about — left it unmarked. It is gone. Run the
tool for numbers; what is written down here is what the driving SETTLED, which
is not a number.

**Provenance for every figure quoted in this section.** Run of **2026-08-22**
in this tree at `stelling 0.2.0.dev0`, `jax 0.11.0`, `jax_enable_x64=True`, z3
`5.0.0` (wheel), cvc5 `1.3.4` (wheel), CPU, Python 3.12, Linux x86-64, a
12th-gen i7-12850HX, **load average 4.0**, `solver_timeout_ms=10000`, three
repeats per cell — the page's own budget and repeat count. A second full run
immediately before it, at **load 5.7**, gave every answer and every direction
identically.

It did not give the same seconds, and this section used to say they moved *"by
up to 10%"*. **Measured, cell by cell across those two runs, that is wrong in
both directions of the table**: the widest disagreement on a second-scale cell
is row 7's z3 column, `4.0–4.1 s` against `4.4–4.6 s` — **15%** — and the
widest on a millisecond cell is row 1's z3 column, `5–6 ms` against `6–9 ms` —
**80%**. The speed ratio in finding 1 is one of the things that moved: it is
`6.9x–12.8x` at load 5.7 and `8.0x–11.8x` at load 4.0. Neither number used to
be attributed to a run, on a page whose whole complaint is figures with no
provenance.

**This paragraph used to end *"every figure below is the load-4.0 run's and
says so"*, and that was FALSE OF ITS OWN SECTION** — which is the same defect
it was written to repair, one level up. The figures below come from ten
drivings, each at its own load, and each now carries it:

| figure below | transcript, all under `scratchpad/D7-solver-battery/` — a historical measurement, tracked and not in the sdist; `tools/solver_battery.py` ships and re-drives it | load |
|---|---|---|
| the ten-row battery table and finding 1's `8.0x–11.8x` | `battery-run-2-2026-08-22T1959Z.txt` | 4.0 |
| finding 1's `6.9x–12.8x` | `battery-run-1-2026-08-22T1949Z.txt` | 5.7 |
| finding 1's `14.0x–53.0x`, and rows 9/10's 3–7 ms and 2–7 ms | `battery-run-3-2026-08-23T2351Z.txt` | 0.5 |
| the first reconstruction sweep's `1.22x` | `sweep-does-the-freedom-reach-the-number-2026-08-23.txt` | 11.74 |
| its unchanged re-drive's `1.67x` | `sweep-2-does-the-freedom-reach-the-number-2026-08-23.txt` | 1.62 |
| the box-ceiling sweep's `17.82x`, `2.55x`, `1.62x` and the 55–71 ms cvc5 range | `where-does-the-box-stop-2026-08-23.txt` | 1.24–1.65 |
| the `check()`-wall ratios | `wall-and-invocation-order-2026-08-23.txt`, `wall-and-invocation-order-2-2026-08-23.txt` | 1.71, 4.10 |
| row 8's three readings | `row8-and-the-constraint-2026-08-23.txt` | 1.27 |
| row 7's one-parameter-at-a-time re-drive | `row7-re-driven-2026-08-23.txt` | 1.15 |

Every one of those transcripts is committed under
`scratchpad/D7-solver-battery/` — a historical measurement, tracked and not
in the sdist, with `tools/solver_battery.py` shipping in its place — with its
load average in its header. The
sweep whose cvc5 column read 54–85 ms was quoted here until 2026-08-23 with no
transcript behind it at all; it has been dropped and replaced by the re-drive
above, which has one.

**THAT DIRECTORY IS TRACKED AND IS NOT IN THE SDIST, so a reader of the
distribution cannot open one of these, and what they have instead is better
for the question they are asking.** `tools/solver_battery.py` ships — it is
the reason `/tools` is in the sdist allowlist at all — and re-drives this
battery on the reader's own machine, which is the only machine whose
milliseconds are about them; `tests/test_solver_battery.py` ships too and
holds every figure on this page to the transcript it is attributed to
wherever those transcripts are present, skipping rather than failing where
they are not. The wall times below are a historical measurement of two
particular machines at the loads stated, and this page's own instruction is
to read the direction rather than the multiple.

**What held.**

- **The `fragment` column: all ten, exactly.** It is the one column of that
  table that is a property of this tree rather than of a machine, and it now
  has a gate that runs with no backend installed.
- Every row's declared input count, too: *32 vars* is 32 declared inputs in
  the emitted slice, *64-element array* is 64, and so on for all ten.
- **Finding 1** — on `QF_LRA` both backends decided everything, and z3 was
  faster on every linear row. **The direction held in every run. The RATIO is
  not a stable quantity at all**: 8.0x–11.8x at load 4.0, 6.9x–12.8x at load
  5.7, and **14.0x–53.0x** on a third run of the same tool at load 0.5, where
  z3's cheapest rows fall to 1–3 ms and the ratio is dominated by a number
  near the timer's resolution. *"Faster by an order of magnitude"* survives all
  three; any particular multiple survives none of them. That is the clearest
  demonstration on this page of why the instruction is to read the direction,
  and the same caveat still applies about the subprocess spawn in cvc5's
  floor.
- **Finding 3** — a full portfolio is not a full-portfolio answer. Rows 7 and 10
  were decided with both backends installed, invoked and stamped and only one
  answering, and the verdict said `PORTFOLIO DEGRADED` itself.
- **The deep-and-narrow half of finding 2**, and more sharply than the table
  above puts it: on this battery's reading of the chain rows, z3 discharged
  both in single-digit milliseconds while cvc5 took seconds at ten factors and
  did not finish at twelve.

**What was newly established, and where the first version of it overreached.**
The `reconstruction` cells above are a measurement. Two sweeps produce it and
both are committed under `scratchpad/D7-solver-battery/` — which is in the
repository but not in the sdist — each with its load average in its header.
What the distribution carries is the ANSWER they decided rather than the
instruments: the per-row grade is in `tools/solver_battery.py`, printed by
`--rows`, and `tests/test_solver_battery.py` holds this page and that tool to
one another.

**The first sweep varies everything those six labels leave open except the
size of the box.** `scratchpad/D7-solver-battery/probe-does-the-freedom-reach-the-number.py`
— a historical measurement, tracked and not in the sdist; the grade it
decided ships in `tools/solver_battery.py` —
**34 label-compatible readings**: for AM–GM six boxes from `[0,1]²` to
`[−100,100]²`; for Motzkin five boxes plus both associations of the degree-6
monomials plus the factored form `x²y²(x² + y² − 3) + 1`; for the linear rows
boxes from `[0,1]` to `[−1000,1000]`, and reduced against elementwise for the
64-element row; for the false rows three degrees of falseness and three
different cubics — three repeats each, three portfolios each. **Every reading
gave the published outcome**, and that part has held in every driving.

Its statistic is `A/B`: for each of the eighteen (row, portfolio) pairs, **A**
is the spread over every reading and every repeat and **B** the widest spread
inside a single unchanged cell — one harness, one portfolio, three repeats,
nothing chosen differently. **Neither the ratio nor the count of exact ties is
a stable quantity.** Driven at load 11.74 it is `1.00x` in four of the eighteen
pairs and never exceeds `1.22x`; re-driven unchanged at load 1.62 it is
`1.00x` in five and never exceeds `1.67x`. Both transcripts are committed. And
`A ≥ B` **by construction** — B is a maximum over cells of a within-cell ratio,
A is that ratio over the pooled samples, and the pool contains every cell — so
the statistic is biased toward 1 in the direction that flatters it, on top of
A pooling three to eighteen times as many samples. What survives all of that
is qualitative: `A/B` stays near 1, and its worst pair is always a z3 column at
single-digit milliseconds where one millisecond of jitter is tens of percent.

**The second sweep is there because the first one has a ceiling and did not
say so.** Every row-4 and row-5 reading above is capped at a box of `±100`.
`scratchpad/D7-solver-battery/probe-where-does-the-box-stop.py` — the same
shape: a historical measurement, tracked and not in the sdist, whose
correction to the grade ships in `tools/solver_battery.py` — removes the
cap: these harnesses declare a
float64 box, so the widest reading of any of these labels is the widest box the
format holds, and each row is driven from `±1` out to the largest scale at
which every constant its predicate builds from the box is still finite. Five
interleaved passes, load 1.24–1.65, transcript committed:

| | across every declarable box | `A/B` |
|---|---|---|
| cvc5, all six rows | 55–71 ms | ≤ `1.08x` |
| z3, rows 1, 3, 4, 6 | 3–7 ms | `1.00x`–`1.13x` |
| z3, row 2 | 4–13 ms | `1.62x` |
| z3, row 5 | 5–6 ms at `±1` → 126–140 ms at the top of the format | `17.82x` |

Row 5 climbs rung by rung — 5–6 ms at `±1` and at `±100`, 6–8 at `±1e20`,
7–11 at `±1e60`, 14–20 at `±1e100`, 28–35 at `±1e150`, 131–140 at `±1e300` and
126–132 at the format's own ceiling, the last two being within each other's
jitter — so it is not a cliff at the edge of the format. Restricted to boxes
at `±1e100` and below, well inside float64, `A/B` is still `2.55x`. Every one
of those readings still answers `unsat`.

That is why rows 1, 3, 4 and 6 are graded `reconstructed` and rows 2 and 5 are
graded `outcome only`, and it is why the second grade exists at all.

**And the limit of the strongest grade is worth stating in the words the tool
states it in.** The absolute cells move between sweeps by as much as they
differ from the published ones: rows 1–6's cvc5 column came out at 73–148 ms on
the first sweep and 56–74 ms on the second, against a published 69–117; z3 at
7–13 ms and 1–8 ms, against a published 8–13. So:

> `reconstructed` does NOT mean `this battery reproduced the published milliseconds` — it means the label's remaining freedom is not what separates them

`A/B` says a choice inside the label moves those four rows' cells no further
than the clock does between two repeats, so whatever separates the two
batteries is the machine and the day, not the reading. On rows 2 and 5 it does
not say that, which is what the weaker grade records.

**WHAT DID NOT HOLD: the wide-and-shallow half of finding 2 — and the reason
matters more than the result.** Three defensible readings of *"32 vars, 16
elementwise products"* were built and driven (`--variants`), three repeats
each, at load 4.0:

| reading of the label | z3 alone | cvc5 alone |
|---|---|---|
| `sum(a*b) <= sum(a)`, `a,b ∈ [0,1]¹⁶` — the literal one: sixteen products, thirty-two variables | unsat, 4.4–4.6 s | **UNKNOWN** (16.0 s wall) |
| `sum(a² + b² − 2ab) >= 0`, `a,b ∈ [−1,1]¹⁶` | unsat, 22 ms | unsat, 169–182 ms |
| `sum(a*b) − sum(b*a) >= 0`, `a,b ∈ [−1,1]¹⁶` | unsat, 6–9 ms | unsat, 69–81 ms |
| **the row above, as published** | **UNKNOWN** (timeout) | unsat, 166–175 ms |

Those milliseconds are the load-4.0 run's and are a snapshot: re-running
`--variants` gives different ones and the same three directions. It is the
directions the prose below reads, and the directions
`tests/test_solver_battery.py` holds.

The first **reverses** this page's direction. The second reproduces the
page's cvc5 cell to within a few milliseconds and still does not reproduce its
z3 cell — which is the cleanest statement of the problem there is: **matching
one cell is not identifying a harness.** The third shows no split either.

So the honest reading is not *"the page's row 7 is wrong"*. It is that **row 7
is unsupported**, and no number produced by any of these three harnesses may
be filed against it. The battery marks rows 7 and 8, prints what its own
reading measured, and refuses to file any of it against the published row —
that refusal is the finding, not the four seconds.

**And the reversal is not an artefact of one parameter of that harness.**
Re-driven here, one parameter at a time:

- *the box.* Narrowing to `[0,0.5]¹⁶` keeps the reversal and sharpens it — z3
  discharges in under 50 ms, cvc5 still runs its full 16.0 s guard. (Widening
  instead, to `[0,2]¹⁶`, does not test the direction at all: the predicate is
  FALSE there, and both backends refute it in under 100 ms. A box that changes
  the answer is a different obligation, not a harder one.)
- *which sum is the bound.* `sum(a*b) <= sum(b)` is exactly as literal a
  reading as `<= sum(a)`, and it keeps the reversal: z3 `unsat` in 4.5 s,
  cvc5 `timeout` at 16.0 s.
- *the budget.* At `solver_timeout_ms=60000` — **six times this page's** — z3
  discharges in 3.9 s while cvc5 runs its whole 91.1 s guard and returns
  `timeout`. More time does not rescue the published direction; it costs 91
  seconds to fail to.

**One more constraint, which this page has never stated and which rules out
the obvious readings of several rows.** An obligation interval propagation
DECIDES never reaches a backend at all, so it cannot be a row of a solver
comparison. Two further label-compatible spellings of row 7 —
`sum(a*b) <= 16` and the elementwise `a*b >= −1`, both over `[0,1]¹⁶` — are
interval-exact and are discharged before any backend is discovered, with zero
invocations and the verdict reporting *"definitely true for all 1 element(s)"*
and *"…for all 16 element(s)"*. Every harness in `tools/solver_battery.py` is
therefore written with a dependency interval arithmetic cannot see through — a
repeated variable, a cancellation, an even power — and every row's `chose
here` list records that as something the battery had to work out rather than
read. The page's ten rows must have satisfied the same constraint. How, it
does not say.

**That last sentence was false on three of the ten rows until 2026-08-23, and
they were the reconstruction rows.** Rows 4, 5 and 6 did not record the
constraint, and it is not decorative on them: *"2 vars, degree 2 (AM–GM)"* says
nothing about the two variables sharing one box, and AM–GM over
`x ∈ [0, 0.1], y ∈ [10, 20]` is interval-decided — driven here, **zero solver
invocations**, so it could not have been a row of this table however
defensible a reading it is. So is Motzkin over `[1e-300, 1e-299]²`, a square
box in the same family as every other Motzkin reading swept. All ten rows
record the constraint now, and
`test_every_row_says_what_the_page_left_open` keeps it that way.

The chain rows are underdetermined in the same way, though less violently. A
*"10-factor product chain"* can be ten variables or one variable ten times,
and `x¹⁰ >= 0` shows no split at all: z3 6–7 ms against cvc5 71–73 ms. The
direction above survived one reading and vanished under another; it did not
reverse under either, which is why rows 9 and 10 are graded **direction
only** rather than unsupported.

**Their milliseconds are a different matter, and neither row is marked for
it.** Across three runs of this battery, row 9's `z3 alone` is 3–7 ms against
a published 123–133 ms — a factor of **18x–44x** — and row 10's is 2–7 ms
against a published 689–702 ms, a factor of **98x–351x**. Those two factors
used to be written *"eighteen to forty"* and *"a hundred to three hundred"*,
both understating their own top.
Every outcome agrees; only the clock is out, by two orders of magnitude. That
is the whole reason the `‡` mark asks *did the direction reverse* and not *do
the numbers agree*: on a page whose own instruction is to read the direction,
a mark for disagreeing milliseconds would be on almost every row, and would
say nothing a reader could act on. The grade is where the milliseconds are
disclaimed; the mark is for the rows where even the direction is gone.

### What the `both` column actually is

**It is the sum of the other two columns, and that tells you nothing.** Eight
of the ten `both` cells in the table above are their own `z3 alone` cell plus
their own `cvc5 alone` cell — within a fifth of a second on three of the four
second-scale rows, and within the millisecond jitter this page measures at 80%
on the cheap ones. Two are not. Row 9's misses by more than jitter can carry
and in a direction a portfolio cannot produce, which is the last paragraph of
this section. **Row 4's misses the other way and is not a finding**: its
published `both` of 80–83 ms lies entirely below the 84–96 ms of its own
published singles, by 1–16 ms, which is inside a jitter this page has already
measured at 80% on a millisecond cell. It is recorded because the identity was
stated as a blanket and the cheap rows do not support a blanket. That identity
looked like a recovery of the one thing the page never states — *what* it
timed — and it is not a recovery of anything.

`solvers._dispatch_obligation`, which `solvers.escalate` reaches for each
obligation, runs the admitted backends in a plain sequential loop with **no
short-circuit** — `solvers.py:1997`, whose body ends at 2046 and contains no
`break`, so a backend that has already answered `unsat` does not stop the next
one being asked. This page says the same thing in words two
sections up — *"Primary is ordering, not selection: every installed backend
runs on every fragment."* So a two-backend wall **is** the two single-backend
walls plus per-call overhead, for any correct measurement of any harness
whatsoever. **The identity cannot fail. Its holding is evidence of nothing,
and a third of that ten-row table carries no information about the harness
behind it.**

**What the arithmetic does rule out is real**: the page did not time the
`check()` wall. That wall also pays tracing, jit and interval propagation, and
on a discharged obligation it runs the vacuity widen re-check, which invokes
**every backend a second time** — four invoked solver stamps against two
published latencies, on every discharged row of this battery, against two
stamps and two latencies on a refuted one. Measured per repeat over two
sessions: the wall is **1.8x–3.1x** the published-latency sum on a discharged
cheap row and **1.05x–1.11x** on a refuted one. The page's cheap `both` cells
are 78–112 ms against 79–93 ms of their own singles; at the wall they would
have been 150 ms and up. So the battery sums the per-invocation milliseconds
stelling publishes in the verdict's notes, over the first escalation, **and
prints how many invocations that leaves out** — because it is a lower bound,
and saying so is cheaper than being asked.

The re-check does not simply double a discharged row, either, and it would be
easy to write that it does. The widened query is a different question and can
be far easier: row 7's two-backend run is 19.6–20.1 s of wall against a
19.6–20.0 s published-latency sum **with all four stamps invoked**, because
over unbounded reals its predicate is plainly false and both backends answer
at once. What the re-check doubles is the INVOCATION COUNT; what it adds to
the wall depends on the widened query.

**And row 9 does not merely miss the sum.** `123–133 ms` plus `8.3–8.5 s` is
`8.4–8.6 s` against a published `~8.1 s`, which on its own is a 0.3–0.5 s
mismatch and could be rounding. It is not rounding, because the same row's
published `both` is **below its own published `cvc5 alone`**, and a sequential
portfolio with no short-circuit cannot finish two backends faster than it
finishes one of them. Row 9's milliseconds are also a long way out while every
outcome agrees: z3 at 3–7 ms across three runs against a published 123–133 ms
is a factor of 18x–44x, cvc5 is out by about 3.6x, and every cell still says
`unsat`. **That third fact used to read *"the row this battery's reading
disagrees with most in milliseconds while agreeing on every outcome"*, and it
was false** — checkable from the same three transcripts, and contradicted two
sections above by this page's own figures: row 10 agrees on every cell's
outcome in all three runs and is out by 98x–351x against row 9's 18x–44x.
Row 9's case rests on the first two facts, which are about arithmetic its own
published cells cannot satisfy. The third is context, not a superlative.

### `solver_timeout_ms=10000` is not a ten-second wall

z3's `:timeout` is its own and lands near it — 10.1 s measured. The cvc5 wheel
is driven through a wall-guarded child process because its `tlimit` does not
reliably preempt the coverings solver, and that guard is `timeout * 1.5 + 1 s`:
**16.0 s for a ten-second budget**, measured here on three rows. A two-backend
obligation on which both backends time out therefore costs about **26 s**, not
20 — which is what row 8 cost. If you are sizing a CI budget from the ten
seconds in this page's provenance paragraph, you are out by more than half.
The battery prints every cell that ran past its budget, with the backend
named.

**And the ordering can cost you the whole wall guard.** On the literal reading
of row 7 driven with both backends, the invocation order is the one the
fragment table dictates — cvc5 first, as the `QF_NRA` primary. Measured four
times across two sessions, at load 5.6 and load 1.7: **cvc5 burns its full
16.0 s wall guard and returns `timeout`, and only then is z3 asked, which
answers `unsat` in 3.6–4.3 s.** The two-backend run costs 19.6–20.4 s for an
answer z3 alone gives in about four. The heuristic that puts cvc5 first is
backwards for that obligation, and with
no short-circuit there is nothing to cut the loss short: the price of the
wrong guess is the whole of the other backend's wall. That is a reason to
restrict a call rather than a reason to uninstall a backend — see the fourth
option below.

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
  `integer_pow`, division by a declared value: **this bullet used to say
  `[cvc5]`, and it is WITHDRAWN.** Its whole support was rows 7 and 8, which
  are the two rows on this page marked `unsupported`: one defensible reading
  of their labels runs the split the other way, with z3 discharging in four
  seconds and cvc5 hitting its wall guard. What survives is smaller and is
  still worth knowing — cvc5 is the `QF_NRA` **primary**, so on a nonlinear
  obligation it is asked first, and with no short-circuit a wrong first guess
  costs the whole of the other backend's wall (measured: 20 s for an answer z3
  alone gives in 4). **There is no measured recommendation here.** If your
  obligations are mostly polynomial, drive
  `python tools/solver_battery.py --variants` on shapes like your own and read
  the direction off that, or install both and restrict per call. Do not read
  this bullet's former advice out of the `QF_NRA` primary: primary is
  ordering, not competence.
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
