# D7 — solver battery: raw evidence, and what each row's grade rests on

Rescued from a scratch directory at a forced stop on 2026-08-22, extended
2026-08-23. Nothing here ships (`/scratchpad` is absent from the sdist
allowlist).

## What the files are

| file | what |
|---|---|
| `battery-run-1-2026-08-22T1949Z.txt` | first full `tools/solver_battery.py --variants --repeats 3`, load average 5.67 |
| `battery-run-2-2026-08-22T1959Z.txt` | second full run, load average 3.99. **Every figure quoted on the page is this run's**, and the page now says so |
| `probe-run-to-run.py` / `run-to-run-2026-08-23.txt` | those two runs compared cell by cell. Answers identical; the widest millisecond disagreement is **1.80x** and the widest second-scale one **1.15x** |
| `probe-does-the-freedom-reach-the-number.py` / `sweep-…-2026-08-23.txt` | the sweep that decides the grade of rows 1–6: 34 label-compatible readings, 3 repeats, 3 portfolios |
| `probe-wall-and-invocation-order.py` / `wall-and-invocation-order-2026-08-23.txt` | `check()`'s wall against the published-latency sum, the invoked-stamp count, and row 7's two-backend invocation ORDER |
| `lane-jax-solvers-partial.txt` | whole-suite run in `stelling-jax`, KILLED at ~56% by the 2026-08-22 shutdown. Green to that point |
| `controls-nojax.sh` / `controls-jax.sh` | the original eleven positive controls (C1–C11). All eleven still fire |
| `controls-fixup.sh` | the **fourteen** controls for the gates added 2026-08-23 (C12–C24). Every one of these mutations left the module GREEN before the fixup, including the two that break the named polynomials and the one that inverts the headline finding on the page |
| `nosolvers.py` | `pytest -p nosolvers`, hiding both wheels from `stelling._optional.available` — the page's own documented method — to simulate the `test-jax-no-solvers` CI lane. Its docstring records what that simulation can and cannot judge, measured |
| `probe-harness-spread.py`, `probe-wide-and-deep.py` | the harness-space probes that produced the reconstruction finding |

## WHICH ROWS ARE RECONSTRUCTED — and the question that decides it

**The binary question is the wrong question.** *"Does the row label pin a
harness"* is answered no by every label ever written, and answering it graded
all ten rows the same — which made row 4 read exactly as weakly as row 7. It
was also answered wrongly here: an earlier version of this file said *"none of
the ten is reconstructible"* and that not one label fixes a predicate. **Two
do.**

The question that decides re-derivability is **whether the freedom the label
leaves REACHES the published number.** That is measurable, it has been
measured, and the answer partitions the ten rows into three.

### RECONSTRUCTED — the freedom does not reach the number (rows 1–6)

Driven: `probe-does-the-freedom-reach-the-number.py`, **34 label-compatible
readings**, 3 repeats × 3 portfolios each.

- **row 4, `2 vars, degree 2 (AM–GM)`** — the label NAMES the object: AM–GM's
  two-variable degree-2 form is `x² + y² >= 2xy`. The box is the only free
  parameter, and it was swept over six boxes from `[0,1]²` to `[−100,100]²`.
- **row 5, `2 vars, degree 6 (Motzkin)`** — the label NAMES the object:
  `x⁴y² + x²y⁴ − 3x²y² + 1`. Free parameters are the box (five, `[−1,1]²` to
  `[−100,100]²`) and the association of the degree-6 monomials (both, plus the
  factored form `x²y²(x² + y² − 3) + 1`).
- **rows 1, 2, 3, 6** — shape fixed, box and predicate free. Boxes from `[0,1]`
  to `[−1000,1000]`; reduced against elementwise for the 64-element row; three
  degrees of falseness for the false array row; three different cubics.

**The result.** For each of the eighteen (row, portfolio) pairs, compare
**A** — the spread over every reading and every repeat — with **B**, the widest
spread inside a single unchanged cell (one harness, one portfolio, three
repeats). In the committed run `A/B` is exactly `1.00x` in four pairs and
never exceeds `1.22x`, whose worst case is row 2's z3 column at 8–11 ms, where
one millisecond of jitter is a 12% move. Row 4's cvc5 column is 75–84 ms across
a two-hundred-fold range of boxes: `A/B = 1.05x`. **A pools three to eighteen times as many
samples as B, so it is biased wider** — that it is barely wider at all is the
measurement. Every reading gave the published outcome. A cell that sits on
cvc5's subprocess-spawn floor and z3's few milliseconds cannot be moved by a
choice made above it.

**The limit of that claim**, recorded because it is the part that would be
easy to overstate: RECONSTRUCTED does not mean "this battery reproduced the
published milliseconds". The absolute cells move between sweeps by as much as
they differ from the published ones — rows 1–6's cvc5 column was 54–85 ms on
one sweep and 73–148 ms on another against a published 69–117, and z3 was
1–20 ms then 7–13 ms against a published 8–13. What the grade claims is that
the label's remaining freedom is not what separates them: `A/B` says a choice
inside the label moves a cell no further than the clock does between two
repeats.

### DIRECTION ONLY (rows 9, 10)

`(x0·…·x9)² >= 0` over `[−1,1]¹⁰` reproduces the page's direction (z3 fast,
cvc5 slow then timing out). **An equally defensible reading — one variable to
the tenth, `x¹⁰ >= 0` — shows no split at all** (z3 6–7 ms, cvc5 71–73 ms).
The direction survived one reading and vanished under another; it did not
reverse under either.

Neither row is marked `‡`, and neither is marked for its MILLISECONDS either,
which is deliberate. Across the three committed runs, row 9's `z3 alone` is
3–7 ms against a published 123–133 ms (18x–44x) and row 10's is 2–7 ms against
a published 689–702 ms (100x–350x), with every outcome agreeing. A mark for
disagreeing milliseconds would land on almost every row of a page whose own
instruction is to read the direction. The GRADE is where the milliseconds are
disclaimed; the `‡` mark is for rows where even the direction is gone.

**Row 9 additionally carries three signals about the PAGE'S OWN CELLS**, and
they are recorded in `Row.published_notes`:

1. its published `both` (`~8.1 s`) is **below** its published `cvc5 alone`
   (`8.3–8.5 s`) — impossible for a sequential portfolio with no
   short-circuit;
2. it is the only one of the four second-scale rows whose `both` is not the
   sum of its own singles — and (1) is why that is not rounding;
3. it is the row this battery disagrees with most while agreeing on every
   outcome: z3 at 6 ms against a published 123–133 ms is a factor of twenty.

### UNSUPPORTED (rows 7, 8)

**This is still the finding of the batch.** Three defensible readings of
`32 vars, 16 elementwise products`, three repeats each, load 3.99:

| reading | z3 alone | cvc5 alone |
|---|---|---|
| `sum(a*b) <= sum(a)`, `a,b ∈ [0,1]¹⁶` — the LITERAL one | unsat, 4.4–4.6 s | **UNKNOWN** (16.0 s wall) |
| `sum(a² + b² − 2ab) >= 0`, `a,b ∈ [−1,1]¹⁶` | unsat, 22 ms | unsat, 169–182 ms |
| `sum(a*b) − sum(b*a) >= 0`, `a,b ∈ [−1,1]¹⁶` | unsat, 6–9 ms | unsat, 69–81 ms |
| **the page's published row** | **UNKNOWN** (timeout) | unsat, 166–175 ms |

The literal reading **reverses** the page. The second reproduces the page's
cvc5 cell to within a few milliseconds and still does not reproduce its z3
cell — matching one cell is not identifying a harness. The third shows no
split. At row 8's width the literal reading has NEITHER backend finishing.

**The reversal is not an artefact of one harness parameter** —
`probe-row7-re-driven.py`, one parameter at a time:

| varied | z3 alone | cvc5 alone |
|---|---|---|
| box `[0,1]¹⁶` (as published above) | unsat 3.9 s | timeout 16.0 s |
| box `[0,0.5]¹⁶` | unsat <0.05 s | timeout 16.0 s |
| bound swapped: `sum(a*b) <= sum(b)` | unsat 4.5 s | timeout 16.0 s |
| budget 60 s (**6x** the page's) | unsat 3.9 s | timeout **91.1 s** |

Widening the box to `[0,2]¹⁶` instead does NOT test the direction: the
predicate is false there and both backends answer `sat` in under 100 ms. A box
that changes the answer is a different obligation.

**And the constraint the page never states**, confirmed independently by the
same probe: an obligation interval propagation DECIDES never reaches a backend.
`sum(a*b) <= 16` and elementwise `a*b >= −1`, both over `[0,1]¹⁶`, are
interval-exact — zero invocations, *"definitely true for all 1 element(s)"* and
*"…for all 16 element(s)"*. Every harness in the tool is written with a
dependency interval arithmetic cannot see through for exactly that reason.

**Do not "fix" rows 7 and 8 by searching harness space until something
reproduces the page.** That is fitting to a conclusion. The correct output is
the refusal already in the tool: `Row.contested`, the `‡` marks — which now
live in the page's OWN table rather than in a second copy of it — and
`direction_report`'s `FINDING 2 … NOT DECIDABLE FROM THIS BATTERY`.

## The zero-dep lanes, re-driven 2026-08-23

- **no jax, both wheels** (`/home/nick/venvs/stelling-nojax`): exit 0, whole
  inventory prints, every cell accounted for, and the mechanism column now
  says `<- NOT MEASURED` where it used to print ten `<- DISAGREES` markers
  while the section below correctly said jax was missing.
- **jax 0.5.1, genuinely wheel-free** (`/home/nick/venvs/jax051`, no z3, no
  cvc5, no pytest): exit 0, and the fragment column re-derives correctly for
  every row — on a jax series OUTSIDE `TESTED_JAX_SERIES`, with the runtime
  warning that says so.
- **jax 0.5.1, wheel-free, with a `cvc5` shim on PATH**: the probe now reports
  cvc5 REACHABLE via the external binary and drives it. Before the fixup it
  printed the binary's path on the `cvc5` line and then told the reader
  *"no SMT backend is installed — pip install"* for all thirty cells.

## What the page's own arithmetic settles, and what it cannot

**`both = z3 + cvc5` is FORCED and settles nothing.** `solvers._escalate` runs
the admitted backends in a plain sequential loop with **no short-circuit**,
and the page says the same in words. The identity cannot fail for any correct
measurement of any harness, so a third of that ten-row table carries no
information about the harness behind it. An earlier version of this file
reported the identity as having RECOVERED what the page timed; it recovers
nothing.

**What it does rule out** — measured, `wall-and-invocation-order-2026-08-23.txt`:
the page did not time the `check()` wall. On a discharged row that wall is
**1.8x–3.1x** the published-latency sum, because the vacuity widen re-check
invokes every backend a second time (**4 invoked stamps against 2 published
latencies**); on a refuted row it is **1.05x–1.11x** and there are 2 stamps.
The re-check does **not** simply double a row: row 7's two-backend run is
19.6–20.1 s of wall against a 19.6–20.0 s sum with all four stamps invoked,
because the widened query is trivially false.

**`solver_timeout_ms=10000` is not a ten-second wall.** `solvers._wall_seconds`
is `timeout*1.5 + 1`, so the wall-guarded cvc5 child is killed at **16.0 s** —
measured on rows 7, 8 and 10. A two-backend row where both time out costs
**26 s**, which is what row 8 cost.

**And the ordering can cost the whole guard.** On row 7's two-backend run,
cvc5 — the `QF_NRA` PRIMARY — burns its full 16.0 s wall before z3 is asked,
and z3 then answers in 3.6–4.3 s. A two-backend install pays 19.6–20.4 s for
an answer z3 alone gives in about four. Measured four times across two
sessions.
