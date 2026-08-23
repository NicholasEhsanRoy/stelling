# D7 — solver battery: raw evidence, and what each row's grade rests on

Rescued from a scratch directory at a forced stop on 2026-08-22, extended
2026-08-23 and again the same day. Nothing here ships (`/scratchpad` is absent
from the sdist allowlist).

**What round 2 changed, in one paragraph.** The reconstruction partition's
conclusion was an unqualified universal over rows 1–6 and it was false: the
sweep behind it caps rows 4 and 5 at a box of `±100` and says so nowhere, and
the labels bound the box nowhere at all. Driven out to the ceiling the float64
FORMAT supplies — the one ceiling here that is not a judgement call — rows 2
and 5 fail and are regraded `outcome only`, a fourth grade. Half the
supporting argument was false too: cvc5's spawn floor is real and now measured
across 10^308 rather than 10², and z3 has no floor of that kind at all. Four
more measured defects were repaired: the `1.22x` (three drivings, three
different digits), the missing load averages and the false *"every figure below
is the load-4.0 run's"* umbrella, row 9's third signal (row 10 disagrees more),
`solvers._escalate` (a symbol this repository has never had), and row 8's grade,
which did not meet its own criterion. Twelve new controls, C25–C36, all red.


## What the files are

| file | what |
|---|---|
| `battery-run-1-2026-08-22T1949Z.txt` | first full `tools/solver_battery.py --variants --repeats 3`, load average 5.67 |
| `battery-run-2-2026-08-22T1959Z.txt` | second full run, load average 3.99. **Every figure quoted on the page is this run's**, and the page now says so |
| `probe-run-to-run.py` / `run-to-run-2026-08-23.txt` | those two runs compared cell by cell. Answers identical; the widest millisecond disagreement is **1.80x** and the widest second-scale one **1.15x** |
| `probe-does-the-freedom-reach-the-number.py` / `sweep-…-2026-08-23.txt` | the sweep that varies everything rows 1–6's labels leave open EXCEPT the box's magnitude: 34 readings, 3 repeats, 3 portfolios, load 11.74 |
| `sweep-2-does-the-freedom-reach-the-number-2026-08-23.txt` | the same sweep, unchanged, re-driven at load 1.62. `A/B ≤ 1.67x` against the first run's `1.22x`, ties in 5 of 18 against 4 — **neither digit is a property of anything** |
| `probe-where-does-the-box-stop.py` / `where-does-the-box-stop-2026-08-23.txt` | **the sweep that regraded rows 2 and 5.** The first sweep caps rows 4 and 5 at `±100` and says so nowhere; this drives each row from `±1` to the widest box a float64 declaration can express, 5 interleaved passes, load 1.24–1.65 |
| `probe-row8-and-the-constraint-three-rows-omit.py` / `row8-and-the-constraint-2026-08-23.txt` | row 8's three readings driven at row 8's own width, and the two interval-DECIDED readings rows 4, 5 and 6 did not exclude |
| `wall-and-invocation-order-2-2026-08-23.txt` | the wall probe, second committed session, load 4.10 |
| `controls-fixup2.sh` | the **twelve** controls for the gates added in round 2 (C25–C36). Every one left the module green at `943b9c6` |
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

### RECONSTRUCTED (rows 1, 3, 4, 6) and OUTCOME ONLY (rows 2, 5)

**Round 1 graded all six RECONSTRUCTED on an unqualified universal, and the
universal was false.** The sentence was *"the choice of reading adds no more
spread than re-running the same harness does"*, resting on `A/B ≤ 1.22x` over
34 readings — and **every row-4 and row-5 reading in that sweep is capped at a
box of `±100`**, while row 1 reaches `±1000`, with nothing anywhere stating the
cap. The labels supply no bound: AM–GM holds over all of `R²` and Motzkin is
nonnegative over all of `R²`, which round 1's own row-4 text said. Widening the
sweep and re-asserting the universal does not repair that, because the next box
is always wider.

**The fix is a ceiling that is not a judgement call: these harnesses declare a
float64 box.** `probe-where-does-the-box-stop.py` drives each row from `±1` to
the largest scale at which every constant its predicate builds from the box is
still finite — `1.7976931348623157e308` for rows 4, 5, 6; `8.9e307`, `1e306`
and `1e307` for rows 1, 2, 3, whose thresholds scale with the box. Five
interleaved passes so a load drift cannot fake a trend across boxes, load
1.24–1.65 throughout:

| | across every declarable box | `A/B` |
|---|---|---|
| cvc5, all six rows | 55–71 ms | ≤ `1.08x` |
| z3, rows 1, 3, 4, 6 | 3–7 ms | `1.00x`–`1.13x` |
| z3, row 2 | 4–13 ms | `1.62x` |
| z3, row 5 | 5–6 ms at `±1` → 126–140 ms at the top of the format | `17.82x` |

Row 5 climbs rung by rung — 5–6, 5–6, 6–8, 7–11, 14–20, 28–35, 131–140, and
126–132 at the format's own ceiling, within the previous rung's jitter — so it is not a
cliff at the edge of the format, and restricted to `±1e100` and below `A/B` is
still `2.55x`. **Every one of those readings still answers `unsat`.** So the
OUTCOME is reconstructed on all six rows and the MILLISECOND is not on rows 2
and 5, which is what the fourth grade records.

**And half of round 1's supporting argument was false.** *"Their cells sit on
two floors this page itself names: the cvc5 spawn (~70 ms) and z3's ~10 ms …
a cell that IS the floor cannot be moved by a choice made above it."* True of
cvc5, and now measured across 10^308 rather than 10²: 55–71 ms on every row at
every declarable box. **False of z3**, which has no floor of that kind — 1–3 ms
on row 1 in the re-driven sweep, 126–140 ms on row 5 at the top of the format.
Row 5's ~10 ms was the price of that predicate at that box, movable sixteenfold
by a choice the label leaves open.

The first sweep is still evidence and is still committed: it varies the
predicate, the association, the threshold, the cubic and reduced-vs-elementwise,
which the box sweep does not. **34 label-compatible readings**, 3 repeats × 3
portfolios each.

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

**The result, and what is and is not stable in it.** For each of the eighteen
(row, portfolio) pairs, compare **A** — the spread over every reading and every
repeat — with **B**, the widest spread inside a single unchanged cell (one
harness, one portfolio, three repeats). Three drivings of that unchanged sweep:

| driving | load | `A/B` ceiling | pairs at exactly `1.00x` |
|---|---|---|---|
| committed, round 1 | 11.74 | `1.22x` | 4 of 18 |
| the audit's re-drive | 1.80 | `1.25x` | 7 of 18 |
| committed, round 2 | 1.62 | `1.67x` | 5 of 18 |

**Neither the ratio nor the tie count is a property of anything.** What
survives is qualitative: `A/B` stays near 1 and its worst pair is always a z3
column at single-digit milliseconds where one millisecond is tens of percent.
And **`A ≥ B` by construction** — B is a max over cells of a within-cell ratio,
A is that ratio over the pooled samples, and the pool contains every cell — so
the statistic is biased toward 1 in the direction that flatters it, on top of
A pooling three to eighteen times as many samples. Round 1 disclosed the
pooling bias and not the construction bias.

**The limit of the strongest grade**, recorded because it is the part that
would be easy to overstate, and now the single string both the page and the
tool quote (`RECONSTRUCTED_IS_NOT_A_REPRODUCTION`): *`reconstructed` does NOT
mean `this battery reproduced the published milliseconds` — it means the
label's remaining freedom is not what separates them.* The absolute cells move
between sweeps by as much as they differ from the published ones: rows 1–6's
cvc5 column was 73–148 ms on the first sweep and 56–74 ms on the second against
a published 69–117, and z3 was 7–13 ms then 1–8 ms against a published 8–13.
**Round 1 also quoted a sweep with cvc5 at 54–85 ms whose transcript was never
committed; that citation is gone.**

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
3. its milliseconds are a long way out while every outcome agrees — z3 at
   3–7 ms against a published 123–133 ms, **18x–44x**. This used to read *"the
   row this battery disagrees with most"*, and that was FALSE from these same
   three transcripts: row 10 also agrees on every cell's outcome in all three
   and is out by **98x–351x**. The factor is now derived from the transcripts
   by `test_the_clock_gap_the_page_states_is_the_one_its_transcripts_show`
   rather than typed — it had four hand-written renderings across the page and
   the tool, three of them understating their own top.

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
split.

**ROW 8'S EVIDENCE IS NOT ROW 7'S, AND ROUND 1 SAID IT WAS.** `Row(n=8)`'s
refusal read *"the same three readings disagree the same way"*, while this
README said the opposite two lines above — at row 8's width the literal reading
has NEITHER backend finishing. Driven at row 8's own width (load 1.27,
`row8-and-the-constraint-2026-08-23.txt`):

| reading at row 8's width | z3 alone | cvc5 alone |
|---|---|---|
| `sum(a*b) <= sum(a)`, `a,b ∈ [0,1]³²` — the LITERAL one | **UNKNOWN**, 10.1 s | **UNKNOWN**, 16.0 s |
| `sum(a² + b² − 2ab) >= 0` | unsat, 152–157 ms | unsat, 936–946 ms |
| `sum(a*b) − sum(b*a) >= 0` | unsat, 3–5 ms | unsat, 62–65 ms |
| **the page's published row** | **UNKNOWN** (timeout) | unsat, 772–792 ms |

**None of them reverses row 8** in row 7's sense: no reading has cvc5 timing
out while z3 answers. What they do is fail to reproduce its published split
while putting z3 — the backend the page has timing out — six to twenty-two times
ahead of cvc5 wherever they split at all. `GRADE_UNSUPPORTED`'s criterion was
restated to say what it actually rules on, and the grade is now DERIVED from
the recorded directions (`battery.READINGS`,
`test_a_contested_row_s_grade_is_DERIVED_from_its_readings`) rather than
asserted beside them.

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

**Round 1 asserted on the page that every row records that constraint, and
rows 4, 5 and 6 did not — the reconstruction rows.** It is not decorative on
them: nothing in *"2 vars, degree 2 (AM–GM)"* says the two variables share one
box, and AM–GM over `x ∈ [0,0.1], y ∈ [10,20]` is interval-decided at **zero
solver invocations**; so is Motzkin over `[1e-300, 1e-299]²`, a square box in
the same family as every Motzkin reading swept. Both driven,
`row8-and-the-constraint-2026-08-23.txt`. All ten rows record it now and
`test_every_row_says_what_the_page_left_open` holds them to it.

**Do not "fix" rows 7 and 8 by searching harness space until something
reproduces the page.** That is fitting to a conclusion. The correct output is
the refusal already in the tool: `Row.contested`, the `‡` marks — which now
live in the page's OWN table rather than in a second copy of it — and
`direction_report`'s `FINDING 2 … NOT DECIDABLE FROM THIS BATTERY`.

## The zero-dep lanes, re-driven 2026-08-23 (and again in round 2)

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
  *"no SMT backend is installed — pip install"* for all thirty cells. Driven
  end to end by the round-1 audit against a real external `cvc5`; round 2
  changed nothing on that path and did not re-drive it.

Round 2 re-drove the first two: `tools/solver_battery.py` exits 0 in both, the
module is `32 passed, 12 skipped` under `stelling-nojax`, and the fragment
column re-derives for all ten rows under `jax051`.

## What the page's own arithmetic settles, and what it cannot

**`both = z3 + cvc5` is FORCED and settles nothing.** The mechanism is real
and the SYMBOL round 1 cited for it was not: `solvers._escalate` has never
existed in this repository — `hasattr` is `False`, and it survived from a
comment in `src/stelling/smt.py` onto a user-facing page, which cited it twice
while the tool cited it five times. The function is
`solvers._dispatch_obligation`, reached from `solvers.escalate`, and its loop
is `for position, backend in enumerate(ordered)` at `solvers.py:1997` — body
1998–2046, **no `break`** — so it runs the admitted backends in a plain
sequential loop with **no short-circuit**, and the page says the same in
words. The identity cannot fail for any correct
measurement of any harness, so a third of that ten-row table carries no
information about the harness behind it. An earlier version of this file
reported the identity as having RECOVERED what the page timed; it recovers
nothing.

**What it does rule out**: the page did not time the `check()` wall. **The
stable half of that is the invocation count**, which is structural — 4 invoked
stamps against 2 published latencies on every discharged row, 2 against 2 on a
refuted one, in every driving. **The ratio is a wall time and moves like one**:
over two committed sessions (`wall-and-invocation-order-2026-08-23.txt`, load
1.71; `-2-…`, load 4.10) it is **2.07x–3.20x** the notes sum on a discharged
cheap row and **1.05x–1.09x** on a refuted one, and an independent re-drive at
load 7.5–11 reported **1.86x–4.58x**. Round 1 published `1.8x–3.1x` with
neither load stated.

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

## What is still not held, written down rather than implied

Round 2's bar was *every number on the page true, attributed and gated; every
claim the tool makes about a grade held against the page's statement of it;
every remaining gap written down.* These are the gaps.

- **A consistent inversion is still green.** Editing
  `RECONSTRUCTED_IS_NOT_A_REPRODUCTION` **and** the page's two copies of it to
  the same wrong thing passes
  `test_the_page_states_every_grade_s_meaning_in_the_tool_s_own_words`, because
  both records would still agree. What stands behind that is the transcripts:
  the strings say what was measured, the drivings are committed, and
  `test_every_quoted_sweep_figure_is_in_the_transcript_it_is_attributed_to`
  holds the figures to them. A gate cannot decide whether a sentence is true.
- **`solvers._escalate` is still in `src/`.** Three comments — `smt.py:191`,
  `smt.py:201`, `solvers.py:1960` — name a function that has never existed.
  They pre-date this branch and this branch touches nothing under `src/`, so
  they are left; `test_every_solver_symbol_these_two_cite_exists` only reads
  the page and the tool. Fixing them is a one-line change in whatever batch
  next has `src/stelling/solvers.py` open.
- **The wall ratio is attributed but not committed on both sides.** The
  `1.86x–4.58x` at load 7.5–11 is an independent re-drive whose transcript is
  not in this tree; the two ranges that are (`2.07x–3.20x`, `1.05x–1.09x`)
  have theirs. The stable claim — 4 invoked stamps against 2 published
  latencies — is the one the section leans on.
- **The 34-reading sweep still has no box above `±1000`.** That is deliberate
  now: `probe-where-does-the-box-stop.py` is the sweep that varies the box and
  it is the one the grades rest on, while the first sweep varies everything
  else. But the two are not crossed — nobody has driven the factored Motzkin
  form at `±1e300`, and no gate would notice if that reading behaved
  differently from the association the box sweep used.
- **`outcome only` rests on the z3 column alone.** cvc5 is flat on all six
  rows at every declarable box, so rows 2 and 5 are regraded on one backend's
  behaviour. If cvc5's spawn floor ever stops dominating its cell, the grade's
  evidence changes shape and nothing would say so.
- **Rows 7 and 8's `both` and the four expensive rows' outcomes remain
  ungated**, as they were: they are *"did this backend finish inside ten
  seconds"*, which is a millisecond wearing a hat.
- **The battery's own `--variants` numbers are still a live measurement with
  no gate.** `battery.READINGS` records their DIRECTIONS, which is what the
  grades use and what held across every driving; the milliseconds beside them
  in a transcript are one machine's and are not held anywhere.
