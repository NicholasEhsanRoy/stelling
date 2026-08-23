# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Run a solver battery shaped like the ten rows of
``docs/choosing-a-solver-backend.md``, and print what THIS machine measures.

    python tools/solver_battery.py                # the whole battery
    python tools/solver_battery.py --rows         # the inventory, no jax needed
    python tools/solver_battery.py --fragments    # the mechanism column, no solver needed
    python tools/solver_battery.py --variants     # the spread, see below
    python tools/solver_battery.py --json

**THE FIRST THING THIS FILE HAS TO SAY IS WHAT IT IS NOT — AND IT IS NOT THE
SAME THING FOR ALL TEN ROWS.** The harnesses behind that page's ten-row table
were never committed, and the page says so itself; what this ships is ten
harnesses BUILT TO THOSE LABELS, with every parameter the label does not fix
written down beside the harness that fixes it (:attr:`Row.chosen`).

An earlier version of this file stopped there and hung the same disclaimer on
every row. That is wrong twice over, and the second way is the interesting
one. **Rows 4 and 5 name a mathematical object** — AM-GM's degree-2 form and
the Motzkin polynomial — so their labels DO pin a predicate, and saying
otherwise under-sold the table. And the question the disclaimer answered was
the wrong one: *"does the label pin a harness"* is answered no by every label
ever written. **What decides whether a published cell is re-derivable is
whether the freedom the label leaves REACHES that cell**, which is measurable
and has been measured. See :data:`GRADE_RECONSTRUCTED` for the partition and
the sweep behind it; ``--rows`` prints a grade on every row.

That the distinction is not pedantry is itself measured, on the row the
page's headline nonlinear finding rests on. Three defensible readings of
``32 vars, 16 elementwise products`` were built and driven here (see
:data:`VARIANTS`), three repeats each, 2026-08-22 at load average 4.0, and
they do not agree with each other, let alone with the page:

    sum(a*b) <= sum(a)          z3 unsat 4.4-4.6 s  cvc5 UNKNOWN   <- REVERSED
    sum(a^2 + b^2 - 2ab) >= 0   z3 unsat     22 ms  cvc5 unsat 169-182 ms
    sum(a*b) - sum(b*a) >= 0    z3 unsat    6-9 ms  cvc5 unsat  69-81 ms
    the page's row              z3 TIMEOUT          cvc5 unsat 166-175 ms

The first is the most LITERAL reading — two arrays of sixteen, sixteen
elementwise products between them, thirty-two declared variables — and it runs
the page's direction backwards. The second reproduces the page's cvc5 cell to
within a few milliseconds and still does not reproduce its z3 cell, which is
the cleanest statement of the problem there is: **matching one cell is not
identifying a harness.**

So a tool that shipped ONE of those three and called the result "the row,
re-measured" would be publishing a harness choice as a finding about a solver.
An invented harness is worse than an unreproducible table, because it looks
reproducible. Hence the four rules this file is built to:

1. **Every row says what the page fixed and what this battery chose.**
   :attr:`Row.fixed` and :attr:`Row.chosen`; ``--rows`` prints both, and
   ``tests/test_solver_battery.py`` refuses a row whose ``chosen`` is empty.
2. **And every row says whether that choice MATTERED.** :attr:`Row.grade`,
   which is a measurement and not an opinion, and which the page publishes in
   its own table's ``reconstruction`` column. A row may not be promoted
   between grades without driving the readings that justify it.
3. **The page's cells are carried verbatim, beside ours, never overwritten.**
   :attr:`Row.page_both` / ``page_z3`` / ``page_cvc5`` are the 2026-08
   hand-check as published. This tool prints them next to what it measured and
   states, per row, whether the DIRECTION agreed. It never edits them into
   agreement. Where the published cells are internally inconsistent — row 9 —
   that is recorded on the row (:attr:`Row.published_notes`) rather than
   corrected into consistency.
4. **Milliseconds are labelled as one machine's, always.** The page's own
   instruction — read the direction, not the milliseconds — survives here.

WHAT IS ACTUALLY REPRODUCIBLE, IN THREE TIERS, because they are not equally
reproducible and printing them in one table without saying so is how the
original went unreproducible in the first place:

* **fragment** (``QF_LRA`` / ``QF_NRA``) and the declared input count are
  computed by :mod:`stelling.obligation` from the traced jaxpr. They need jax
  and NO solver, they are machine-independent, and they are gated
  (``tests/test_solver_battery.py::test_the_fragment_column_is_what_the_page_publishes``).
* **outcome** (``unsat`` / ``sat`` / ``UNKNOWN``) needs a backend. For the six
  cheap rows it is a claim about the OBLIGATION — row 3's predicate is false,
  so any backend that decides it must answer ``sat`` — and it is gated. For
  the four expensive rows the outcome is "did this backend finish inside ten
  seconds", which is a millisecond wearing a hat, and it is not gated.
* **wall time** is this machine's, under this load, and is gated nowhere.

WHAT IS MEASURED, EXACTLY — AND WHAT THAT DEFINITION LEAVES OUT. Not the
``check()`` call: that includes tracing, jit, interval propagation and — on a
VERIFIED — the vacuity widen re-check, which invokes every backend a second
time. The number reported per cell is the sum of the per-invocation
milliseconds stelling itself publishes in the verdict's notes
(``assert #0: z3 (wheel) answered unsat in 7ms``), over the FIRST escalation
only.

That is a lower bound, and this tool now says by how much rather than leaving
it implied. Two kinds of invocation carry no published latency: the widen
re-check's (every discharged row — 4 invoked stamps against 2 published
latencies) and the admitted-region check's (only a slice carrying a relational
assume; none of the ten rows does). Both are counted from
``Verdict.stamp.solver`` and printed under the table. See :data:`_UNPUBLISHED`.

The page never says what IT timed, and no arithmetic on the page recovers it:
its ``both`` column is the sum of its own singles, but that identity is forced
by ``solvers._escalate``'s sequential no-short-circuit loop and cannot fail
for any correct measurement. See :data:`_SUM_IS_FORCED`.

RUNNING WITH NOTHING INSTALLED. The zero-dep lane is a promise this project
keeps, so a battery most readers cannot run is not much of a remedy:

* **no solver**: every harness still traces and still reports its fragment,
  its input count and its emitted element terms; the three outcome columns say
  ``not measured`` and name the backends that are absent. Exit 0.
* **no jax**: ``--rows`` still prints the whole inventory, because it is data
  and this module imports jax nowhere at module scope. Everything else reports
  that jax is what is missing, and names the extra. Exit 0.

Neither case is an error, and neither is silent.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import platform
import re
import sys
from dataclasses import dataclass, field

# The page's own budget and repeat count, so a reader comparing cells is
# comparing like with like. Both are overridable on the command line; both are
# printed in the provenance header, because a timeout budget is part of what an
# UNKNOWN cell means and a repeat count is part of what a range means.
DEFAULT_TIMEOUT_MS = 10_000
DEFAULT_REPEATS = 1

# ---------------------------------------------------------------- the rows


#: THE PARTITION THAT DECIDES REPRODUCIBILITY — and it is not the partition an
#: earlier version of this file used.
#:
#: That version asked one BINARY question, *does the row label pin a harness*,
#: answered "no, not one of them", and hung the same disclaimer on all ten
#: rows. Two things were wrong with that, and the second is the interesting
#: one.
#:
#: **The blanket answer is false.** Rows 4 and 5 name a MATHEMATICAL OBJECT.
#: AM-GM's two-variable degree-2 form is ``x^2 + y^2 >= 2xy`` and the Motzkin
#: polynomial is ``x^4 y^2 + x^2 y^4 - 3 x^2 y^2 + 1``; a label that names
#: either has fixed the predicate exactly, and only the box is left (plus, for
#: Motzkin, the association of the degree-6 monomials — the same polynomial
#: and not the same emitted script).
#:
#: **And the binary question is the wrong question.** What decides whether a
#: published cell can be re-derived is not whether the label pins a harness —
#: no label anywhere pins one completely — but WHETHER THE FREEDOM THE LABEL
#: LEAVES REACHES THE PUBLISHED NUMBER. Swept and measured
#: (``scratchpad/D7-solver-battery/probe-does-the-freedom-reach-the-number.py``):
#:
#: * rows 1-6 — it does not. Every label-compatible reading lands on the same
#:   two floors: the cvc5 wall-guarded subprocess spawn the page itself names
#:   (~70 ms) and z3's ~10 ms. Across 34 readings the spread is at most 1.22x
#:   the spread of THREE REPEATS OF ONE UNCHANGED HARNESS, and exactly equal
#:   to it in four of the eighteen (row, portfolio) pairs. A cell that IS the
#:   floor cannot be moved by a choice made above it, so those six cells are
#:   RECONSTRUCTED.
#: * rows 9, 10 — it reaches the seconds but not the direction. One reading of
#:   *"10-factor product chain"* shows the page's split (z3 fast, cvc5 slow),
#:   another shows no split at all, and none reverses it. The DIRECTION is
#:   reproducible; the numbers are not.
#: * rows 7, 8 — it reaches the direction, violently. One reading of *"32
#:   vars, 16 elementwise products"* runs the page's direction BACKWARDS. No
#:   number produced here may be filed against those rows.
#:
#: Grading all ten the same is why rows 4 and 5 used to read as weakly as row
#: 7, which is a different kind of dishonesty from the one that disclaimer was
#: written to prevent, and not a smaller one.
GRADE_RECONSTRUCTED = "reconstructed"
GRADE_DIRECTION_ONLY = "direction only"
GRADE_UNSUPPORTED = "unsupported"

#: The three grades, and what each one licenses a reader to do. Printed under
#: the inventory and carried in ``--json``; the page publishes the same three
#: words in its table's own ``reconstruction`` column, and
#: ``tests/test_solver_battery.py`` refuses to let the two spellings drift.
GRADES: dict[str, str] = {
    GRADE_RECONSTRUCTED: (
        "the freedom the label leaves does not reach the published number: "
        "every label-compatible reading measured gave the published OUTCOME "
        "and landed on the same two floors, with a spread no wider than "
        "re-running one unchanged harness. It does NOT mean this battery "
        "reproduced the published milliseconds — this machine's cvc5 cells "
        "run below them throughout. It means no choice a reader makes inside "
        "the label would have moved them"
    ),
    GRADE_DIRECTION_ONLY: (
        "the readings agree on WHICH BACKEND WINS wherever they split, and "
        "none reverses — but at least one shows no split at all, so the "
        "direction survives and the seconds do not"
    ),
    GRADE_UNSUPPORTED: (
        "the readings disagree about which backend finishes, and at least one "
        "runs the published direction BACKWARDS. Unsupported, which is not "
        "the same as wrong: nothing here refutes the published cell, and that "
        "is exactly the problem — nothing here CAN"
    ),
}


@dataclass(frozen=True)
class Row:
    """One row of the page's ten-row table, and the harness built to its label.

    ``name`` is the page's row label BYTE FOR BYTE — including its en-dashes —
    because ``tests/test_solver_battery.py`` parses the page's table and
    asserts this tuple reproduces it in order. That is what stops the tool and
    the page drifting apart in either direction.
    """

    n: int
    name: str  # the page's row label, byte for byte
    fragment: str  # the page's fragment cell
    page_both: str
    page_z3: str
    page_cvc5: str
    #: One of :data:`GRADES`, and the page's table publishes the same word in
    #: its ``reconstruction`` column. This is what replaced the uniform
    #: disclaimer; see the comment above :data:`GRADE_RECONSTRUCTED`.
    grade: str
    #: What the page's row label DOES determine, and this harness honours.
    fixed: tuple[str, ...]
    #: What the label does NOT determine, and this battery therefore chose.
    #: Never empty — every label leaves at least the declared box open. It is
    #: no longer read as "so this row is not reconstructible": whether a
    #: choice MATTERS is :attr:`grade`, and it is measured.
    chosen: tuple[str, ...]
    build: str  # key into BUILDERS
    #: The mathematical object the label NAMES, when it names one. Rows 4 and
    #: 5 do; the other eight do not. A named object is a pinned predicate, and
    #: ``tests/test_solver_battery.py`` holds the harness to it coefficient by
    #: coefficient — because ``2xy -> 1.5xy`` and ``-3 -> -2`` leave the
    #: verdict, the fragment, the degree, the variable count and the answer
    #: all unchanged, so nothing about the MEASUREMENT can tell them apart.
    named_object: str = ""
    #: Set when this battery's reading did not reproduce the page's direction
    #: AND the label does not choose between readings that disagree. It is a
    #: refusal to publish a number, not a finding about a backend. Non-empty
    #: exactly on the :data:`GRADE_UNSUPPORTED` rows, which is gated.
    contested: str = ""
    #: Facts measured about THE PAGE'S OWN CELLS — not about the label and not
    #: about this battery's harness. Row 9 is the only row that carries any.
    published_notes: tuple[str, ...] = ()


#: A constraint the page never states and every one of these harnesses has to
#: satisfy: an obligation interval propagation DECIDES never reaches a solver
#: at all, so it cannot be a row of a solver-comparison table. Every harness
#: below is therefore written with a dependency (a repeated variable, a
#: cancellation, an even power) that interval arithmetic cannot see through.
#: Nothing on the page says this, and it rules out the obvious reading of
#: several row labels — `x0*...*x9 >= -1024` over independent variables is
#: interval-exact and is decided before any backend is asked.
INTERVAL_UNDECIDED = (
    "the obligation must be interval-UNDECIDED or it never reaches a backend; "
    "the page does not say how its rows achieved that"
)

#: THE ARITHMETIC THAT LOOKS LIKE A RECOVERY AND IS NOT ONE.
#:
#: The page never says WHAT it timed, and an earlier version of this file
#: reported that the missing definition could be READ BACK OUT of the page's
#: own cells: its ``both`` column is the SUM of its two single-backend
#: columns, and three of the four second-scale rows land inside 0.2 s of that
#: sum. The arithmetic is real. The inference is worthless, and the reason is
#: in this repository rather than in a judgement call.
#:
#: ``solvers._escalate`` runs the admitted backends in a plain sequential
#: ``for position, backend in enumerate(ordered)`` loop with **no
#: short-circuit** — a backend that has already answered ``unsat`` does not
#: stop the next one being asked — and the page states the same thing in
#: words: *"Primary is ordering, not selection: every installed backend runs
#: on every fragment."* So a two-backend wall IS the two single-backend walls,
#: plus per-call overhead, for ANY correct measurement of any harness
#: whatsoever. The identity cannot fail. Its holding is evidence of nothing,
#: and a third of that ten-row table therefore carries no information about
#: the harness that produced it.
#:
#: WHAT THE ARITHMETIC DOES RULE OUT is real and worth keeping: the page did
#: not time the ``check()`` wall. That wall also pays tracing, jit and
#: interval propagation, and on a VERIFIED it runs the vacuity widen re-check,
#: which invokes every backend a SECOND time (see :data:`_UNPUBLISHED`).
#: Measured here per repeat over two sessions, load average recorded with each
#: (``scratchpad/D7-solver-battery/wall-and-invocation-order-2026-08-23.txt``):
#: **1.8x-3.1x** the notes sum on a discharged cheap row, against
#: **1.05x-1.11x** on a REFUTED one, where no re-check runs at all — four
#: invoked stamps against two published latencies on every discharged row, two
#: against two on the refuted one. The page's cheap ``both`` cells are
#: 78-112 ms against 79-93 ms of their own singles; at the wall they would have
#: been 150 ms and up.
#:
#: AND ROW 9 IS WORSE THAN "DOES NOT FIT". Its published ``both`` (~8.1 s) is
#: BELOW its published ``cvc5 alone`` (8.3-8.5 s). A sequential portfolio with
#: no short-circuit cannot finish two backends faster than it finishes one of
#: them, under any definition of what was timed. That is not a rounding
#: mismatch; it is a sign violation, and it is recorded on the row.
_SUM_IS_FORCED = (
    "the page's `both` column is the SUM of its own two single-backend "
    "columns. That is FORCED by solvers._escalate's sequential no-short-"
    "circuit loop and cannot fail for any correct measurement, so it "
    "corroborates nothing about what was timed"
)

#: The page never says what it timed, and no arithmetic on the page recovers
#: it (see :data:`_SUM_IS_FORCED`). So it is a choice, recorded as one.
_TIMED = (
    "what was timed — CHOSEN. This battery sums the per-invocation "
    "milliseconds stelling publishes in the verdict's notes, over the FIRST "
    "escalation only; see the note above the inventory for what the page's "
    "own arithmetic can and cannot settle"
)

ROWS: tuple[Row, ...] = (
    Row(
        n=1,
        name="scalar, linear",
        fragment="QF_LRA",
        page_both="unsat, 78–112 ms",
        page_z3="unsat, 8–9 ms",
        page_cvc5="unsat, 71–84 ms",
        grade=GRADE_RECONSTRUCTED,
        fixed=("one declared scalar", "every operation linear", "QF_LRA",
               "true, so unsat on the negated predicate"),
        chosen=("the declared box [1, 2]", "the predicate 2x - x >= 1",
                INTERVAL_UNDECIDED, _TIMED),
        build="scalar_linear",
    ),
    Row(
        n=2,
        name="64-element array, linear",
        fragment="QF_LRA",
        page_both="unsat, 86–91 ms",
        page_z3="unsat, 10–12 ms",
        page_cvc5="unsat, 77–87 ms",
        grade=GRADE_RECONSTRUCTED,
        fixed=("one declared array of 64 elements", "every operation linear",
               "QF_LRA", "true, so unsat"),
        chosen=("the declared box [1, 2]", "the predicate sum(2x - x) >= 64",
                "that the 64 elements are reduced rather than asserted "
                "elementwise", INTERVAL_UNDECIDED, _TIMED),
        build="array64_linear",
    ),
    Row(
        n=3,
        name="8-element array, linear, false",
        fragment="QF_LRA",
        page_both="sat, 86–90 ms",
        page_z3="sat, 11–13 ms",
        page_cvc5="sat, 75–117 ms",
        grade=GRADE_RECONSTRUCTED,
        fixed=("one declared array of 8 elements", "every operation linear",
               "QF_LRA", "FALSE, so sat with a replayed witness"),
        chosen=("the declared box [1, 2]", "the predicate sum(2x - x) >= 9",
                "how far false it is — a predicate false only at a corner and "
                "one false almost everywhere are different search problems",
                INTERVAL_UNDECIDED, _TIMED),
        build="array8_linear_false",
    ),
    Row(
        n=4,
        name="2 vars, degree 2 (AM–GM)",
        fragment="QF_NRA",
        page_both="unsat, 80–83 ms",
        page_z3="unsat, 9 ms",
        page_cvc5="unsat, 75–87 ms",
        grade=GRADE_RECONSTRUCTED,
        fixed=("two declared scalars", "total degree 2",
               "THE PREDICATE ITSELF: the AM-GM inequality's two-variable "
               "degree-2 form is x^2 + y^2 >= 2xy, which is named and not "
               "chosen — this is one of only two labels on that page that "
               "pins a predicate", "QF_NRA", "true everywhere, so unsat"),
        chosen=("the declared box [-1, 1]^2 — AM-GM is a statement over all "
                "of R^2 and a declared harness must bound it somewhere. It "
                "is the ONLY free parameter this label leaves, and it was "
                "swept over six boxes from [0,1]^2 to [-100,100]^2: same "
                "answer every time, and a cvc5 spread of 75-84 ms — 1.05x the "
                "spread of three repeats of one unchanged harness",
                _TIMED),
        build="amgm",
        named_object="AM-GM (two-variable degree-2 form): x^2 + y^2 >= 2xy",
    ),
    Row(
        n=5,
        name="2 vars, degree 6 (Motzkin)",
        fragment="QF_NRA",
        page_both="unsat, 92–106 ms",
        page_z3="unsat, 12–13 ms",
        page_cvc5="unsat, 81–83 ms",
        grade=GRADE_RECONSTRUCTED,
        fixed=("two declared scalars", "total degree 6",
               "THE PREDICATE ITSELF: the Motzkin polynomial is "
               "x^4 y^2 + x^2 y^4 - 3 x^2 y^2 + 1, a named object and not a "
               "choice — the second and last label on that page that pins a "
               "predicate", "QF_NRA", "nonnegative everywhere, so unsat"),
        chosen=("the declared box [-2, 2]^2 — swept over five boxes from "
                "[-1,1]^2 to [-100,100]^2",
                "the association of the degree-6 monomials — (x2*x2)*y2 "
                "rather than x2*(x2*y2), which is the same polynomial and not "
                "the same emitted script; both associations and the factored "
                "form x2*y2*(x2 + y2 - 3) + 1 were driven, and the spread "
                "across all eleven readings is 1.00x-1.07x the spread of "
                "three repeats of one unchanged harness", _TIMED),
        build="motzkin",
        named_object=("Motzkin: x^4 y^2 + x^2 y^4 - 3 x^2 y^2 + 1 >= 0"),
    ),
    Row(
        n=6,
        name="1 var, degree 3, false",
        fragment="QF_NRA",
        page_both="sat, 87–88 ms",
        page_z3="sat, 11 ms",
        page_cvc5="sat, 69–71 ms",
        grade=GRADE_RECONSTRUCTED,
        fixed=("one declared scalar", "total degree 3", "QF_NRA",
               "FALSE, so sat with a replayed witness"),
        chosen=("the cubic — this battery uses x^3, the page could have used "
                "any of them", "the declared box [-2, 2]", _TIMED),
        build="cubic_false",
    ),
    Row(
        n=7,
        name="32 vars, 16 elementwise products",
        fragment="QF_NRA",
        page_both="unsat, ~10.3 s",
        page_z3="**UNKNOWN** (timeout)",
        page_cvc5="unsat, 166–175 ms",
        grade=GRADE_UNSUPPORTED,
        fixed=("32 declared elements in two arrays of 16",
               "16 elementwise products between them", "QF_NRA",
               "true, so unsat"),
        chosen=("the declared box [0, 1]^16 x [0, 1]^16",
                "the predicate sum(a*b) <= sum(a)",
                "WHICH IS THE WHOLE PROBLEM WITH THIS ROW — see `contested`",
                INTERVAL_UNDECIDED, _TIMED),
        build="wide_products",
        contested=(
            "the page's row says z3 TIMED OUT and cvc5 answered in 166-175 ms. "
            "Three readings of this label were built and driven (--variants): "
            "one REVERSES that (z3 unsat, cvc5 timeout) and two show no split "
            "at all (both backends under 200 ms). The label does not choose "
            "between them, so no number produced here can be filed against "
            "this row"
        ),
    ),
    Row(
        n=8,
        name="64 vars, 32 elementwise products",
        fragment="QF_NRA",
        page_both="unsat, ~11.0 s",
        page_z3="**UNKNOWN** (timeout)",
        page_cvc5="unsat, 772–792 ms",
        grade=GRADE_UNSUPPORTED,
        fixed=("64 declared elements in two arrays of 32",
               "32 elementwise products between them", "QF_NRA",
               "true, so unsat"),
        chosen=("the declared box [0, 1]^32 x [0, 1]^32",
                "the predicate sum(a*b) <= sum(a)",
                "the same contest as row 7", INTERVAL_UNDECIDED, _TIMED),
        build="wide_products",
        contested="row 7's contest at twice the width; the same three readings "
                  "disagree the same way",
    ),
    Row(
        n=9,
        name="10-factor product chain",
        fragment="QF_NRA",
        page_both="unsat, ~8.1 s",
        page_z3="unsat, 123–133 ms",
        page_cvc5="unsat, 8.3–8.5 s",
        grade=GRADE_DIRECTION_ONLY,
        fixed=("a chain of products, ten factors long", "QF_NRA",
               "true, so unsat",
               "'high total degree in few variables', per the page's own "
               "reading of its two nonlinear directions"),
        chosen=("WHETHER THE TEN FACTORS ARE TEN VARIABLES OR ONE VARIABLE "
                "TEN TIMES — the two are both 'a 10-factor product chain' and "
                "they are not the same problem (--variants measures both)",
                "the declared box [-1, 1]^10",
                "the predicate (x0*...*x9)^2 >= 0 — a bare product of "
                "independent variables is interval-EXACT and never escalates",
                INTERVAL_UNDECIDED, _TIMED),
        build="chain_squared",
        published_notes=(
            "THREE SIGNALS ON ONE ROW, and they are about the page's cells "
            "rather than about this harness.",
            "(1) SIGN VIOLATION. Its published `both` (~8.1 s) is BELOW its "
            "published `cvc5 alone` (8.3-8.5 s). solvers._escalate is a "
            "sequential loop with no short-circuit, so a two-backend run "
            "cannot finish faster than one of its own backends does, under "
            "any definition of what was timed.",
            "(2) It is the only one of the four second-scale rows whose "
            "`both` is not the sum of its own singles (123-133 ms + 8.3-8.5 s "
            "= 8.4-8.6 s against a published ~8.1 s) — and (1) is why that "
            "misfit is not a rounding mismatch.",
            "(3) It is the row this battery's reading disagrees with most in "
            "MILLISECONDS while agreeing with it on every OUTCOME: z3 at 6 ms "
            "against a published 123-133 ms is a factor of twenty, and cvc5 "
            "is out by 3.6x, and both cells still say unsat. Both facts are "
            "true at once, which is the whole reason the direction is the "
            "readable part of that table and the milliseconds are not.",
        ),
    ),
    Row(
        n=10,
        name="12-factor product chain",
        fragment="QF_NRA",
        page_both="unsat, ~16.7 s",
        page_z3="unsat, 689–702 ms",
        page_cvc5="**UNKNOWN** (timeout)",
        grade=GRADE_DIRECTION_ONLY,
        fixed=("a chain of products, twelve factors long", "QF_NRA",
               "true, so unsat"),
        chosen=("row 9's choices at twelve factors", "the declared box [-1, 1]^12",
                INTERVAL_UNDECIDED, _TIMED),
        build="chain_squared",
    ),
)


@dataclass(frozen=True)
class Variant:
    """A second defensible reading of a row's label.

    Variants exist because the SPREAD is the content. For rows 7-10 the page
    publishes a direction, the label admits several harnesses, and the
    harnesses disagree; printing one of them as "the row" would hide exactly
    the fact that makes the row unreproducible.
    """

    row: int
    key: str
    predicate: str
    build: str


VARIANTS: tuple[Variant, ...] = (
    Variant(7, "sum-of-squares",
            "sum(a^2 + b^2 - 2ab) >= 0 over [-1,1]^16 x [-1,1]^16",
            "wide_sum_of_squares"),
    Variant(7, "cancellation",
            "sum(a*b) - sum(b*a) >= 0 over [-1,1]^16 x [-1,1]^16",
            "wide_cancellation"),
    Variant(8, "sum-of-squares",
            "sum(a^2 + b^2 - 2ab) >= 0 over [-1,1]^32 x [-1,1]^32",
            "wide_sum_of_squares"),
    Variant(8, "cancellation",
            "sum(a*b) - sum(b*a) >= 0 over [-1,1]^32 x [-1,1]^32",
            "wide_cancellation"),
    Variant(9, "one-variable",
            "x^10 >= 0 over [-1, 2], the chain being one variable ten times",
            "chain_one_variable"),
    Variant(10, "one-variable",
            "x^12 >= 0 over [-1, 2], the chain being one variable twelve times",
            "chain_one_variable"),
)


# ------------------------------------------------------- the harnesses
#
# Every builder imports jax INSIDE the function. That is not style: `--rows`
# has to work in an environment with no jax at all, and a module-scope jax
# import would make this whole file unimportable there. It is the same rule
# `stelling._optional` imposes on the package.


def _f64():
    import jax.numpy as jnp

    return jnp.float64


def build_scalar_linear(row_n: int):
    from stelling.harness import any_array, assert_

    def scalar_linear():
        x = any_array((), _f64(), (1.0, 2.0))
        return assert_(2.0 * x - x >= 1.0)

    return scalar_linear


def build_array64_linear(row_n: int):
    import jax.numpy as jnp
    from stelling.harness import any_array, assert_

    def array64_linear():
        x = any_array((64,), _f64(), (1.0, 2.0))
        return assert_(jnp.sum(2.0 * x - x) >= 64.0)

    return array64_linear


def build_array8_linear_false(row_n: int):
    import jax.numpy as jnp
    from stelling.harness import any_array, assert_

    def array8_linear_false():
        x = any_array((8,), _f64(), (1.0, 2.0))
        # false at x = (1, ..., 1): the sum is 8, and 8 >= 9 is false.
        return assert_(jnp.sum(2.0 * x - x) >= 9.0)

    return array8_linear_false


# ---------------------------------------------------- the two named objects
#
# Rows 4 and 5 are the only two rows whose LABEL names a mathematical object,
# so they are the only two whose predicate the label pins. That makes their
# coefficients a CLAIM this tool makes about the page's rows, and a claim
# needs a gate.
#
# It cannot be gated through the measurement. `x^2 + y^2 >= 1.5xy` is still
# true, still degree 2, still two variables, still QF_NRA, still unsat.
# Motzkin with -2 in place of -3 is still nonnegative, still degree 6, still
# two variables, still unsat. Every column of this battery's table, every
# verdict, and every existing gate is byte-identical under both mutations —
# measured, both stayed green against the whole 30-test module. Only the
# POLYNOMIAL can tell them apart.
#
# So the polynomials are lifted out of the harnesses, where they can be
# evaluated on ordinary floats, and `tests/test_solver_battery.py` compares
# them against a reference written independently from the published
# definitions, on an exact dyadic grid. Nothing about the emitted script
# changes: `build_amgm` still asserts the same two sides in the same order,
# and `build_motzkin` still asserts the same expression in the same
# association.


def amgm_sides(x, y):
    """The two sides of AM–GM's two-variable degree-2 form, ``x² + y² ≥ 2xy``.

    Returned as a pair rather than as a difference so the harness below emits
    exactly the comparison it emitted before this was lifted out.
    """
    return x * x + y * y, 2.0 * x * y


def motzkin_value(x, y):
    """``x⁴y² + x²y⁴ − 3x²y² + 1`` — the Motzkin polynomial.

    In the association this battery chose: ``(x2*x2)*y2`` rather than
    ``x2*(x2*y2)``. Same polynomial, different emitted script, and the choice
    is recorded in :attr:`Row.chosen` — measured, it moves the cell by less
    than three repeats of one unchanged harness do.
    """
    x2 = x * x
    y2 = y * y
    return x2 * x2 * y2 + x2 * y2 * y2 - 3.0 * x2 * y2 + 1.0


def build_amgm(row_n: int):
    from stelling.harness import any_array, assert_

    def amgm():
        x = any_array((), _f64(), (-1.0, 1.0))
        y = any_array((), _f64(), (-1.0, 1.0))
        lhs, rhs = amgm_sides(x, y)
        return assert_(lhs >= rhs)

    return amgm


def build_motzkin(row_n: int):
    from stelling.harness import any_array, assert_

    def motzkin():
        x = any_array((), _f64(), (-2.0, 2.0))
        y = any_array((), _f64(), (-2.0, 2.0))
        return assert_(motzkin_value(x, y) >= 0.0)

    return motzkin


def build_cubic_false(row_n: int):
    from stelling.harness import any_array, assert_

    def cubic_false():
        x = any_array((), _f64(), (-2.0, 2.0))
        return assert_(x * x * x >= 0.0)

    return cubic_false


def build_wide_products(row_n: int):
    import jax.numpy as jnp
    from stelling.harness import any_array, assert_

    n = {7: 16, 8: 32}[row_n]

    def wide_products():
        a = any_array((n,), _f64(), (0.0, 1.0))
        b = any_array((n,), _f64(), (0.0, 1.0))
        # true because every b_i <= 1; interval-undecided because the two sums
        # are computed independently and their difference straddles 0.
        return assert_(jnp.sum(a * b) <= jnp.sum(a))

    return wide_products


def build_wide_sum_of_squares(row_n: int):
    import jax.numpy as jnp
    from stelling.harness import any_array, assert_

    n = {7: 16, 8: 32}[row_n]

    def wide_sum_of_squares():
        a = any_array((n,), _f64(), (-1.0, 1.0))
        b = any_array((n,), _f64(), (-1.0, 1.0))
        return assert_(jnp.sum(a * a + b * b - 2.0 * a * b) >= 0.0)

    return wide_sum_of_squares


def build_wide_cancellation(row_n: int):
    import jax.numpy as jnp
    from stelling.harness import any_array, assert_

    n = {7: 16, 8: 32}[row_n]

    def wide_cancellation():
        a = any_array((n,), _f64(), (-1.0, 1.0))
        b = any_array((n,), _f64(), (-1.0, 1.0))
        return assert_(jnp.sum(a * b) - jnp.sum(b * a) >= 0.0)

    return wide_cancellation


def build_chain_squared(row_n: int):
    from stelling.harness import any_array, assert_

    k = {9: 10, 10: 12}[row_n]

    def chain_squared():
        xs = [any_array((), _f64(), (-1.0, 1.0)) for _ in range(k)]
        p = xs[0]
        for x in xs[1:]:
            p = p * x
        return assert_(p * p >= 0.0)

    return chain_squared


def build_chain_one_variable(row_n: int):
    from stelling.harness import any_array, assert_

    k = {9: 10, 10: 12}[row_n]

    def chain_one_variable():
        x = any_array((), _f64(), (-1.0, 2.0))
        p = x
        for _ in range(k - 1):
            p = p * x
        return assert_(p >= 0.0)

    return chain_one_variable


BUILDERS = {
    "scalar_linear": build_scalar_linear,
    "array64_linear": build_array64_linear,
    "array8_linear_false": build_array8_linear_false,
    "amgm": build_amgm,
    "motzkin": build_motzkin,
    "cubic_false": build_cubic_false,
    "wide_products": build_wide_products,
    "wide_sum_of_squares": build_wide_sum_of_squares,
    "wide_cancellation": build_wide_cancellation,
    "chain_squared": build_chain_squared,
    "chain_one_variable": build_chain_one_variable,
}


# ---------------------------------------------------------------- environment


@dataclass
class Environment:
    """Everything a reader comparing their cells to ours needs to know differs.

    The page's whole problem was a measurement with no provenance, so this is
    printed above the table on every run and rides in the ``--json``."""

    when: str
    python: str
    platform: str
    machine: str
    cpu: str
    cpu_count: int | None
    loadavg: str
    stelling: str | None
    jax: str | None
    x64: str | None
    z3: str | None
    cvc5: str | None
    cvc5_binary: str | None
    timeout_ms: int
    repeats: int
    tree: str
    #: WHETHER STELLING WILL ACTUALLY RUN THE BACKEND, which is not the same
    #: question as "is the wheel installed" and must not be re-derived from
    #: the wheel here. There are THREE routes to a solver, not two: the z3
    #: wheel, the cvc5 wheel, and an EXTERNAL ``cvc5`` binary named by
    #: ``STELLING_CVC5`` or found on ``PATH`` — the route this page's own
    #: install instructions name. ``tests/_solver_gate.py`` documents that
    #: proxy at length as already-found-and-fixed, and an earlier version of
    #: this file reintroduced it: measured with a ``cvc5`` shim on PATH and
    #: both wheels hidden, ``solvers._backends_for`` returned the binary while
    #: this tool printed "no SMT backend is installed" for all thirty cells —
    #: having already probed and printed the binary's path two lines above.
    #: It told a reader who had followed the page's ``STELLING_CVC5``
    #: instructions to install what they already had.
    z3_reachable: bool = False
    cvc5_reachable: bool = False

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(n for n, v in (("z3", self.z3_reachable),
                                    ("cvc5", self.cvc5_reachable)) if v)

    @property
    def missing_backends(self) -> tuple[str, ...]:
        return tuple(n for n, v in (("z3", self.z3_reachable),
                                    ("cvc5", self.cvc5_reachable)) if not v)

    @property
    def cvc5_route(self) -> str:
        """How cvc5 is reached, which the version string alone cannot say."""
        if not self.cvc5_reachable:
            return "not reachable"
        if os.environ.get("STELLING_CVC5"):
            return f"external binary (STELLING_CVC5={self.cvc5_binary})"
        if self.cvc5:
            return "wheel"
        return f"external binary on PATH ({self.cvc5_binary})"


def _cpu_model() -> str:
    """A human-readable CPU name, or the best the platform will give us.

    Timings are the reason this file exists to be distrusted, so the model is
    reported rather than left to `platform.processor()`, which on Linux is
    usually the bare architecture."""
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _loadavg() -> str:
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        return "unavailable"
    return f"{one:.2f} {five:.2f} {fifteen:.2f}"


def _configure_jax() -> tuple[str, str]:
    """Import jax and put it in the configuration the page's table was taken
    under, which is not jax's default.

    ``jax_enable_x64`` is OFF by default, and under float32 the declared boxes
    below are different boxes and the emitted rationals are different
    rationals. The page states ``jax_enable_x64=True`` among its provenance,
    so this tool sets it rather than inheriting whatever the caller had — and
    then REPORTS that it set it, because a configuration a tool imposed is not
    a configuration a reader chose."""
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax.__version__, (
        f"jax_enable_x64={bool(jax.config.read('jax_enable_x64'))} "
        f"(set by this tool: the page's configuration)"
    )


def probe_environment(timeout_ms: int, repeats: int) -> Environment:
    stelling_version = jax_version = x64 = None
    z3v = cvc5v = cvc5_bin = None
    z3_ok = cvc5_ok = False
    tree = "not importable"
    try:
        import stelling
        from stelling import _optional

        stelling_version = stelling.__version__
        tree = str(pathlib.Path(stelling.__file__).resolve().parent)
        z3v = _optional.version("z3")
        cvc5v = _optional.version("cvc5")
        cvc5_bin = _optional.cvc5_binary()
        # The predicate `tests/_solver_gate.py` settles for the whole suite,
        # spelled the same way here: a wheel OR — for cvc5 — an external
        # binary. Asking only about the wheels is the proxy that lane's guard
        # step used to prove, and proving a proxy proves the proxy.
        z3_ok = _optional.available("z3")
        cvc5_ok = _optional.available("cvc5") or cvc5_bin is not None
        if _optional.available("jax"):
            jax_version, x64 = _configure_jax()
    except Exception as e:  # noqa: BLE001 — a probe never takes the run down
        tree = f"not importable ({type(e).__name__}: {e})"
    return Environment(
        when=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu=_cpu_model(),
        cpu_count=os.cpu_count(),
        loadavg=_loadavg(),
        stelling=stelling_version,
        jax=jax_version,
        x64=x64,
        z3=z3v,
        cvc5=cvc5v,
        cvc5_binary=cvc5_bin,
        timeout_ms=timeout_ms,
        repeats=repeats,
        tree=tree,
        z3_reachable=z3_ok,
        cvc5_reachable=cvc5_ok,
    )


# ---------------------------------------------------------------- measuring

#: stelling publishes each invocation's outcome and latency in the verdict's
#: notes, and this tool reads them there rather than wrapping a timer around
#: `check()`. Two reasons, both of which would corrupt a cell: `check()` also
#: traces, jits and propagates, and on a VERIFIED it runs the whole escalation
#: a SECOND time for the vacuity widen re-check. The notes carry the FIRST
#: escalation only, which is the one the table is about.
_ANSWERED = re.compile(
    r"^assert #(?P<ix>\d+): (?P<label>.+?) answered (?P<answer>\S+) in (?P<ms>\d+)ms$"
)

#: THE ONE INVOCATION THE SUM ABOVE CANNOT SEE, named rather than left out.
#: A discharge whose slice carries a relational assume is audited by a SECOND
#: script — ``solvers.escalate``'s admitted-region check, asking whether the
#: declared boxes and the forwarded axioms are satisfiable at all — and that
#: invocation is a real backend run on the same obligation. stelling publishes
#: it in the notes but WITHOUT a latency, so it cannot be added to the sum; it
#: can only be counted and disclosed. None of the ten rows carries a
#: relational assume, so no cell in this battery is affected and no published
#: number is wrong — but the DEFINITION was incomplete, and a reader taking it
#: outside this battery would have been told the wrong thing.
_REGION_ASKED = re.compile(
    r"^assert #(?P<ix>\d+): admitted-region check — (?P<label>.+?) answered "
)

#: WHAT THE TIMING DEFINITION LEAVES OUT, counted rather than asserted.
#:
#: The notes carry the FIRST escalation's per-invocation latencies. They are
#: not every invocation. Measured on this battery, at ``vacuity_mode=
#: "inputs-only"``, by counting invoked ``Verdict.stamp.solver`` stamps
#: against published latencies:
#:
#: * every DISCHARGED row invokes each backend TWICE and publishes one
#:   latency each — 4 invoked stamps, 2 published — because the VERIFIED
#:   vacuity widen re-check runs the whole pipeline again on the widened query
#:   (``preconditions._pipeline``'s second ``_finish``) and its verdict's notes
#:   are discarded. Row 4: wall 279 ms against a 103 ms notes sum, 4 stamps.
#:   Row 3 (REFUTED, no re-check): 77 ms against 71 ms, 2 stamps.
#: * an obligation whose slice carries a relational assume also pays an
#:   admitted-region check, which stelling DOES note and does NOT time.
#:
#: AND THE RE-CHECK DOES NOT SIMPLY DOUBLE THE ROW — an earlier version of
#: this file said it "would double every one of them", which is not what the
#: clock says. The widened query is a different question and can be far
#: easier: row 7's two-backend run is 19.97 s of wall against a 19.89 s notes
#: sum WITH all four stamps invoked, because over unbounded reals its
#: predicate is plainly false and both backends answer at once. The re-check
#: doubles the INVOCATION COUNT on every discharged row; what it adds to the
#: WALL depends on the widened query.
_UNPUBLISHED = (
    "invocations stelling made and published no latency for: the VERIFIED "
    "vacuity widen re-check (every discharged row), and the admitted-region "
    "check (only a slice carrying a relational assume). The cell's "
    "milliseconds are a LOWER BOUND on what the backends were asked to do"
)


@dataclass
class Cell:
    """One (row, portfolio) measurement, repeated ``repeats`` times."""

    outcome: str = ""  # unsat | sat | UNKNOWN | not measured | error | mixed
    reason: str = ""  # why, when there is no outcome to give
    ms: list[int] = field(default_factory=list)
    #: ONE ENTRY PER MEASURED REPEAT, not one per cell. It used to be
    #: overwritten on every repeat, so a three-repeat cell carried the LAST
    #: repeat's invocations beside all three of its milliseconds:
    #: ``render_over_budget`` saw a third of the evidence and ``--json``
    #: published a 3-entry ``ms`` paired with one repeat's invocation list, as
    #: though the two described the same thing.
    invocations: list[list[tuple[str, str, int]]] = field(default_factory=list)
    #: Admitted-region invocations seen across the repeats. Counted rather
    #: than summed because stelling publishes no latency for them — see
    #: :data:`_REGION_ASKED`.
    region_invocations: int = 0
    #: EVERY invocation this cell's milliseconds do NOT cover: invoked solver
    #: stamps, minus published per-invocation latencies, summed over the
    #: repeats. Nonzero means the cell's number is a LOWER BOUND on what the
    #: backends were asked to do, and on this battery it is nonzero on every
    #: discharged row — see :data:`_UNPUBLISHED`.
    unpublished_invocations: int = 0
    status: str = ""  # the verdict status behind the outcome
    degraded: bool = False
    #: Every repeat's outcome, in order — INCLUDING ``error`` and
    #: ``not measured``. A cell whose repeats DISAGREED is not a cell with a
    #: range in it: it is a cell whose answer depends on the clock, or on a
    #: harness that blew up on one pass, and collapsing it to any one repeat's
    #: answer is how a table comes to publish a decision that only happens
    #: sometimes.
    outcomes: list[str] = field(default_factory=list)
    #: The exceptions repeats raised, in order. Kept because the loop no
    #: longer returns on the first one.
    errors: list[str] = field(default_factory=list)

    @property
    def unstable(self) -> bool:
        return len(set(self.outcomes)) > 1

    @property
    def attempted(self) -> bool:
        """A backend was asked, and asked on EVERY repeat.

        Not the same as "this cell has a number in it". A repeat that RAISED
        produced no measurement, and counting it as an attempt is how
        ``direction_report`` came to print ``DID NOT HOLD — 2 of 3 linear rows
        decided`` about a row nothing ever measured. An unmeasured thing must
        never render as a negative result; that rule is asserted against this
        tool's own output in
        ``tests/test_solver_battery.py::test_it_runs_with_no_backend_...``.
        """
        return bool(self.ms) and not any(
            o in ("error", "not measured") for o in self.outcomes)

    @property
    def decided(self) -> bool:
        """One decided answer, and the SAME one on every repeat."""
        return self.attempted and not self.unstable and self.outcome in (
            "unsat", "sat")

    def render(self) -> str:
        if not self.ms:
            return self.outcome or "not measured"
        lo, hi = min(self.ms), max(self.ms)
        if self.unstable:
            return f"{'/'.join(sorted(set(self.outcomes)))}!, {_ms(lo, hi)}"
        return f"{self.outcome}, {_ms(lo, hi)}"


def _ms(lo: int, hi: int) -> str:
    def one(v: int) -> str:
        return f"{v / 1000:.1f} s" if v >= 1000 else f"{v} ms"

    if lo == hi:
        return one(lo)
    if lo >= 1000 or hi >= 1000:
        return f"{lo / 1000:.1f}–{hi / 1000:.1f} s"
    return f"{lo}–{hi} ms"


@dataclass
class Measurement:
    """Everything measured about one row, across the three portfolios."""

    n: int
    name: str
    key: str = ""  # "" for a Row, the variant key for a Variant
    fragment: str = ""
    fragment_reason: str = ""
    inputs: int | None = None
    element_terms: int | None = None
    both: Cell = field(default_factory=Cell)
    z3: Cell = field(default_factory=Cell)
    cvc5: Cell = field(default_factory=Cell)


def _harness(build: str, row_n: int):
    return BUILDERS[build](row_n)


def classify(build: str, row_n: int) -> tuple[str, str, int | None, int | None]:
    """Fragment, reason, declared inputs, element terms — with NO solver.

    This is the mechanism half of the page's table and it is the half a reader
    with an empty environment can still check: the fragment is decided by
    ``stelling.obligation._Slicer._fragment`` off the traced jaxpr, before any
    backend is discovered, so it is measurable in the zero-solver lane and is
    machine-independent."""
    try:
        from stelling.harness import trace
        from stelling.obligation import DeclinedObligation, slice_unknown_obligations
        from stelling.propagate import interval_env, propagate
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {e}", None, None
    try:
        closed = trace(_harness(build, row_n))
        prop = propagate(closed)
        env = interval_env(closed)
        items = list(slice_unknown_obligations(closed, prop, env))
    except Exception as e:  # noqa: BLE001 — a battery never takes the run down
        return "", f"{type(e).__name__}: {e}", None, None
    if not items:
        # An obligation interval propagation DECIDED is not a row of a
        # solver-comparison table at all: nothing escalates and no backend is
        # ever asked. Reported rather than silently blank.
        return "", ("interval propagation decided this obligation; it never "
                    "reaches a backend"), None, None
    item = items[0]
    if isinstance(item, DeclinedObligation):
        return "", f"escalation declined: {item.reason}", None, None
    return item.fragment, "", len(item.inputs), item.element_terms


def measure_cell(build: str, row_n: int, only: str | None, timeout_ms: int,
                 repeats: int) -> Cell:
    """Drive one (harness, portfolio) cell ``repeats`` times.

    **EVERY REPEAT IS KEPT, AND NO REPEAT IS OVERWRITTEN.** Two earlier
    versions of this loop threw evidence away, in opposite directions, and
    both produced a cell that read as a measurement it was not:

    * a repeat that RAISED, or that reached no backend, returned from
      this function immediately — discarding the repeats already measured while leaving
      their milliseconds in ``cell.ms``. Measured: a third repeat blowing up
      after two clean ones rendered as ``error, 56–62 ms``, an error wearing
      two successful repeats' clock. It now renders ``error/unsat!, 56–62 ms``
      and the ``!`` legend says which repeats the range covers.
    * ``cell.invocations`` was assigned, not appended, so a three-repeat cell
      carried one repeat's invocations beside three repeats' milliseconds.
    """
    from stelling.preconditions import check

    cell = Cell()
    reasons: list[str] = []
    for _ in range(repeats):
        kwargs = {} if only is None else {"solver": only}
        try:
            v = check(_harness(build, row_n), vacuity_mode="inputs-only",
                      solver_timeout_ms=timeout_ms, **kwargs)
        except Exception as e:  # noqa: BLE001 — quoted, never raised through
            cell.outcomes.append("error")
            cell.errors.append(f"{type(e).__name__}: {e}")
            reasons.append(f"{type(e).__name__}: {e}")
            continue
        invocations = []
        total = 0
        regions = 0
        for note in v.notes:
            m = _ANSWERED.match(note)
            if m is not None:
                invocations.append((m["label"], m["answer"], int(m["ms"])))
                total += int(m["ms"])
                continue
            if _REGION_ASKED.match(note):
                regions += 1
        # THE INVOCATIONS THE SUM CANNOT SEE, counted from the stamps rather
        # than assumed from the notes. See :data:`_UNPUBLISHED`.
        try:
            invoked = sum(1 for st in v.stamp.solver if st.invoked)
        except Exception:  # noqa: BLE001 — a disclosure never takes a run down
            invoked = len(invocations)
        unpublished = max(0, invoked - len(invocations))
        cell.status = v.status
        cell.degraded = cell.degraded or any(
            "portfolio degraded" in n for n in v.notes)
        if not invocations:
            # Escalation happened but nobody was invoked: a decline, or no
            # backend. The reason is stelling's, quoted.
            cell.outcomes.append("not measured")
            reasons.append(
                v.obligations[0].detail if v.obligations else "no obligation")
            continue
        cell.invocations.append(invocations)
        cell.region_invocations += regions
        cell.unpublished_invocations += unpublished
        cell.ms.append(total)
        cell.outcomes.append(
            {"VERIFIED": "unsat", "REFUTED": "sat"}.get(v.status, "UNKNOWN"))
    distinct = set(cell.outcomes)
    if not cell.outcomes:
        cell.outcome = "not measured"
    elif len(distinct) == 1:
        cell.outcome = cell.outcomes[0]
    else:
        # No single word is true of this cell. `render()` prints every answer
        # the repeats gave; naming the cell after any one of them is the
        # defect this loop was rewritten to end.
        cell.outcome = "mixed"
    # De-duplicated in order: three repeats failing the same way is one
    # reason, and printing it three times is how a reader learns to skip the
    # section that says why nothing was measured.
    cell.reason = " | ".join(dict.fromkeys(reasons))
    return cell


def measure(build: str, row_n: int, name: str, key: str, env: Environment,
            timeout_ms: int, repeats: int, *, cells: bool = True) -> Measurement:
    """Classify one row, and (unless ``cells`` is off) drive its three cells.

    ``cells=False`` is what ``--fragments`` runs: the mechanism column needs
    jax and no backend, and asking for it must not spend ten seconds per
    timeout on a battery the caller did not ask to time."""
    m = Measurement(n=row_n, name=name, key=key)
    if not env.jax:
        m.fragment_reason = ("jax is not installed — nothing can be traced; "
                             'pip install "stelling[jax]"')
        for cell in (m.both, m.z3, m.cvc5):
            cell.outcome = "not measured"
            cell.reason = m.fragment_reason
        return m
    m.fragment, m.fragment_reason, m.inputs, m.element_terms = classify(build, row_n)
    if not cells:
        for cell in (m.both, m.z3, m.cvc5):
            cell.outcome = "not measured"
            cell.reason = "--fragments: the mechanism column only, nothing timed"
        return m
    absent = env.missing_backends
    if not env.backends:
        for cell in (m.both, m.z3, m.cvc5):
            cell.outcome = "not measured"
            # Both routes, in stelling's own wording. The wheel is not the
            # only way to a backend and this tool must not say it is: see
            # `Environment.z3_reachable`.
            cell.reason = ('no SMT backend is reachable — pip install '
                           '"stelling[solvers]" (or set STELLING_CVC5 / put '
                           'cvc5 on PATH)')
        return m
    if len(env.backends) == 2:
        m.both = measure_cell(build, row_n, None, timeout_ms, repeats)
    else:
        m.both.outcome = "not measured"
        m.both.reason = (f"the two-backend column needs both backends; "
                         f"{' and '.join(absent)} not reachable")
    for name_, cell_attr in (("z3", "z3"), ("cvc5", "cvc5")):
        if name_ in env.backends:
            setattr(m, cell_attr,
                    measure_cell(build, row_n, name_, timeout_ms, repeats))
        else:
            c = Cell(outcome="not measured",
                     reason=(f"{name_} is not reachable"
                             + (" (no wheel, and no external binary via "
                                "STELLING_CVC5 or PATH)" if name_ == "cvc5"
                                else " (no wheel)")))
            setattr(m, cell_attr, c)
    return m


# ---------------------------------------------------------------- directions

#: The page states two directional findings about the ten rows and one about
#: the portfolio. A DIRECTION is the part of a measurement that is supposed to
#: survive a change of machine, so it is the only part this tool checks; the
#: milliseconds are reported and nothing is asserted about them.
def direction_report(rows: dict[int, Measurement],
                     variants: list[Measurement] | None = None) -> list[str]:
    out: list[str] = []

    def cell(n: int, which: str) -> Cell | None:
        m = rows.get(n)
        return getattr(m, which) if m else None

    # Finding 1 — on QF_LRA, both decide everything and z3 is an order of
    # magnitude faster. Both halves are checkable; the second is a ratio, and
    # a ratio of two timings on one machine is still a timing, so it is
    # REPORTED, never asserted.
    lra = [n for n in (1, 2, 3) if rows.get(n)]
    # ATTEMPTED IS NOT "HAS A WORD IN IT". A repeat that raised, or that
    # reached no backend, measured nothing — and counting it as an attempt is
    # how this report came to print `DID NOT HOLD — 2 of 3 linear rows
    # decided` about a row where the third row's harness had blown up, with
    # the next line still saying "on every linear row". Measured, by making
    # one linear harness raise. An unmeasured thing must never render as a
    # negative result; that is this tool's own rule and this is where it was
    # broken.
    attempted = [n for n in lra
                 if cell(n, "z3").attempted and cell(n, "cvc5").attempted]
    unattempted = [n for n in lra if n not in attempted]
    decided = [n for n in attempted
               if cell(n, "z3").decided and cell(n, "cvc5").decided]

    def _why(n: int) -> str:
        for which in ("z3", "cvc5"):
            c = cell(n, which)
            if not c.attempted:
                return f"row {n} [{which}]: {c.reason or c.outcome or 'nothing measured'}"
        return f"row {n}: nothing measured"

    if lra and not attempted:
        # NOT THE SAME AS "DID NOT HOLD", and writing it as one would be the
        # exact defect this tool exists to end: a missing measurement rendered
        # as a negative result.
        out.append(
            "FINDING 1 (QF_LRA: both backends decide everything): NOT "
            "MEASURED — no linear row was measured on both backends in this "
            "environment."
        )
        for n in unattempted:
            out.extend(_wrap(_why(n), 96, "  ", "    "))
    elif lra:
        held = len(decided) == len(attempted)
        out.append(
            f"FINDING 1 (QF_LRA: both backends decide everything): "
            f"{'HELD' if held else 'DID NOT HOLD'} — {len(decided)} of the "
            f"{len(attempted)} linear rows MEASURED on both backends were "
            f"decided by both."
        )
        if unattempted:
            out.append(
                f"  NOT MEASURED on row(s) "
                f"{', '.join(str(n) for n in unattempted)}, which are outside "
                f"the count above rather than counted as failures:"
            )
            for n in unattempted:
                out.extend(_wrap(_why(n), 96, "    ", "    "))
        ratios = []
        for n in decided:
            z, c = cell(n, "z3"), cell(n, "cvc5")
            if z.ms and c.ms and min(z.ms) > 0:
                ratios.append(min(c.ms) / min(z.ms))
        if ratios:
            out.append(
                f"  z3 was faster than cvc5 on each of the {len(ratios)} "
                f"linear row(s) decided here, by {min(ratios):.1f}x to "
                f"{max(ratios):.1f}x. THAT IS A TIMING, on one machine, at "
                f"the load printed above, and part of it is a process spawn "
                f"rather than solving — see the page. It moves between runs "
                f"of this same tool by more than the seconds do."
            )

    # Finding 2 — the nonlinear split goes both ways. This is the one the
    # reconstruction cannot settle, and the report says so in the same breath
    # as the numbers rather than a paragraph away from them.
    contested = [r.n for r in ROWS if r.contested and rows.get(r.n)]
    out.append(
        "FINDING 2 (QF_NRA: neither backend dominates): NOT DECIDABLE FROM "
        "THIS BATTERY."
    )
    out.append(
        "  Not because it is false — because the two row labels it rests on "
        "admit readings that disagree about WHICH backend finishes, one of "
        "which runs the page's direction backwards. That is a property of the "
        "labels, not of this environment, so it does not change with what is "
        "installed."
    )
    if contested:
        out.append(
            f"  Rows {', '.join(str(n) for n in contested)} carry a `contested` "
            f"note: their labels admit readings that disagree with each other "
            f"about the direction. Run --variants to see the spread. This "
            f"tool refuses to file its numbers against those rows."
        )
    deep = [n for n in (9, 10) if rows.get(n)]
    for n in deep:
        z, c = cell(n, "z3"), cell(n, "cvc5")
        if z and c and z.ms and c.ms and min(z.ms) > 0:
            out.append(
                f"  row {n} ({rows[n].name}), THIS battery's reading: "
                f"z3 {z.outcome} {_ms(min(z.ms), max(z.ms))}, "
                f"cvc5 {c.outcome} {_ms(min(c.ms), max(c.ms))}."
            )
    # The spread, READ OFF THE VARIANTS THAT RAN. Without them this paragraph
    # would be a claim about harnesses nobody drove, which is the shape of the
    # defect this whole tool answers.
    for v in variants or []:
        z, c = v.z3, v.cvc5
        if not (z.ms and c.ms):
            continue
        out.append(
            f"  row {v.n} read instead as [{v.key}]: z3 {z.outcome} "
            f"{_ms(min(z.ms), max(z.ms))}, cvc5 {c.outcome} "
            f"{_ms(min(c.ms), max(c.ms))}."
        )
    if variants:
        out.append(
            "  Compare each of those with the same row above. Where a row's "
            "readings disagree about WHICH backend finishes, the label does "
            "not settle it and neither does this battery."
        )

    # Finding 3 — a full portfolio is not the same as a full-portfolio answer.
    # This one IS a mechanism claim: it is about what the verdict discloses,
    # not about how long anything took.
    two_backend = [n for n, m in rows.items() if m.both.attempted]
    degraded = sorted(n for n, m in rows.items() if m.both.degraded)
    if not two_backend:
        out.append(
            "FINDING 3 (a full portfolio is not a full-portfolio answer): NOT "
            "MEASURED — the two-backend column did not run in this "
            "environment."
        )
    elif degraded:
        out.append(
            f"FINDING 3 (a full portfolio is not a full-portfolio answer): "
            f"HELD — rows {', '.join(str(n) for n in degraded)} were decided "
            f"with both backends installed and invoked, and the verdict "
            f"disclosed PORTFOLIO DEGRADED because only one answered."
        )
    else:
        out.append(
            "FINDING 3 (a full portfolio is not a full-portfolio answer): not "
            "exercised — every two-backend row here had both backends answer."
        )
    return out


# ---------------------------------------------------------------- rendering


def _wrap(text: str, width: int, indent: str, hang: str = "") -> list[str]:
    """Wrap ``text``, indenting continuation lines by ``hang`` extra spaces.

    Not cosmetic: a `chose here` entry can run to four lines, and without a
    hanging indent the continuations line up with the NEXT entry's label,
    which makes a list of six choices read as a list of twenty."""
    words, line, out = text.split(), "", []
    pad = indent
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(pad + line)
            pad = indent + hang
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(pad + line)
    return out


def render_rows(width: int) -> str:
    """The inventory: what each label fixes, what this battery chose, and —
    the part that decides reproducibility — whether the choice reaches the
    number.

    Pure data — no jax, no solver, no measurement. This is the section a
    reader should read before any table, and it is why the tool can be honest
    about a row it cannot reconstruct AND about the eight it can."""
    out = [
        "THE TEN ROWS: WHAT EACH LABEL FIXES, WHAT IT LEAVES OPEN, AND",
        "WHETHER WHAT IT LEAVES OPEN REACHES THE PUBLISHED NUMBER",
        "",
        "The harnesses behind docs/choosing-a-solver-backend.md's table were",
        "never committed. These are harnesses BUILT TO ITS ROW LABELS. For",
        "every row, `page fixes` is what the label determines and `chose here`",
        "is what it does not.",
        "",
        "THE SECOND LIST IS NOT A VERDICT ON THE ROW. Every label anywhere",
        "leaves something open; what matters is whether the freedom reaches",
        "the published cell, and that is measured rather than assumed. The",
        "three grades:",
        "",
    ]
    for g in (GRADE_RECONSTRUCTED, GRADE_DIRECTION_ONLY, GRADE_UNSUPPORTED):
        rows_with = [str(r.n) for r in ROWS if r.grade == g]
        out.extend(_wrap(f"{g.upper()} (rows {', '.join(rows_with)}): "
                         f"{GRADES[g]}", width - 2, "  ", "    "))
    out.extend([
        "",
        "The sweep behind the first grade is",
        "scratchpad/D7-solver-battery/probe-does-the-freedom-reach-the-number.py:",
        "34 label-compatible readings of rows 1-6 — boxes over a 200-fold",
        "range, both Motzkin associations and its factored form, three",
        "thresholds for the false row, reduced and elementwise for the array",
        "row — three repeats each, three portfolios each. Every reading gave",
        "the page's own outcome, and the spread ACROSS all of them is at most",
        "1.22x the spread across three repeats of ONE unchanged harness (and",
        "exactly equal to it in four of the eighteen row/portfolio pairs). A",
        "cell that sits on cvc5's subprocess-spawn floor and z3's few",
        "milliseconds is not a cell a choice above it can move.",
        "",
        "TWO LABELS PIN A PREDICATE, and it is worth saying which, because an",
        "earlier version of this file said none did:",
        "",
    ])
    for r in ROWS:
        if r.named_object:
            out.extend(_wrap(f"row {r.n}: {r.named_object}", width - 2, "  ",
                             "  "))
    out.extend([
        "",
        "WHAT THE PAGE'S OWN ARITHMETIC CAN AND CANNOT SETTLE. Its `both`",
        "cells are the SUM of its own two single-backend cells, and that",
        "looks like a recovery of the definition it never states. It is not.",
        "solvers._escalate runs the admitted backends in a plain sequential",
        "loop with no short-circuit, and the page says the same in words",
        "(\"every installed backend runs on every fragment\"), so the identity",
        "cannot fail for any correct measurement of any harness. Its holding",
        "is evidence of nothing, and a third of that table carries no",
        "information about the harness behind it.",
        "",
        "What it DOES rule out: the page did not time the check() wall.",
        "Measured here, published-latency sum against wall: 1.8x-3.1x on a",
        "discharged cheap row, 1.05x-1.11x on a refuted one. And row 9 does",
        "not merely miss the sum — its published `both` is BELOW its own",
        "published `cvc5 alone`, which a sequential portfolio cannot do.",
        "",
        "So this battery sums the per-invocation milliseconds stelling",
        "publishes in the verdict's notes, over the FIRST escalation, and",
        "prints how many invocations that leaves out.",
        "",
    ])
    for r in ROWS:
        out.append(f"[{r.n:2d}] {r.name}   ({r.fragment})   -- {r.grade.upper()}")
        out.append(f"     page's cells: both {r.page_both} | z3 {r.page_z3} "
                   f"| cvc5 {r.page_cvc5}")
        if r.named_object:
            out.extend(_wrap(f"NAMED OBJECT: {r.named_object}", width - 5,
                             "     ", "              "))
        for f in r.fixed:
            out.extend(_wrap(f"page fixes  : {f}", width - 5, "     ", "              "))
        for c in r.chosen:
            out.extend(_wrap(f"chose here  : {c}", width - 5, "     ", "              "))
        for note in r.published_notes:
            out.extend(_wrap(f"ABOUT THE PUBLISHED ROW: {note}", width - 5,
                             "     ", "              "))
        if r.contested:
            out.extend(_wrap(f"CONTESTED   : {r.contested}", width - 5, "     ", "              "))
        out.append("")
    out.append("ALTERNATE READINGS (--variants drives these too):")
    for v in VARIANTS:
        out.extend(_wrap(f"row {v.row} [{v.key}]: {v.predicate}", width - 2, "  "))
    return "\n".join(out)


def render_environment(env: Environment) -> str:
    lines = [
        "ENVIRONMENT (the page's whole problem was a measurement with no provenance)",
        f"  when            : {env.when}",
        f"  stelling        : {env.stelling or 'not importable'}",
        f"  stelling tree   : {env.tree}",
        f"  jax             : {env.jax or 'NOT INSTALLED'}"
        + (f"  ({env.x64})" if env.x64 else ""),
        f"  z3              : "
        + (f"{env.z3} (wheel)" if env.z3_reachable else "NOT REACHABLE"),
        f"  cvc5            : "
        + (f"{env.cvc5 or 'no wheel'} ({env.cvc5_route})"
           if env.cvc5_reachable else "NOT REACHABLE")
        + (f"  [wheel {env.cvc5} present but unused]"
           if env.cvc5 and env.cvc5_route.startswith("external") else ""),
        f"  python          : {env.python}",
        f"  platform        : {env.platform}  ({env.machine})",
        f"  cpu             : {env.cpu}  x{env.cpu_count}",
        f"  load average    : {env.loadavg}   <- timings below are load-sensitive",
        f"  solver_timeout  : {env.timeout_ms} ms",
        f"  repeats per cell: {env.repeats}",
    ]
    return "\n".join(lines)


_BANNER = """\
THIS IS NOT A RE-DERIVATION OF THE PAGE'S TABLE, AND CANNOT BE ONE.
The harnesses behind docs/choosing-a-solver-backend.md's ten rows were never
committed and cannot be recovered from ten labels. Below are ten harnesses
BUILT TO those labels; run --rows to see, per row, exactly which parameters
the label left open and what this battery chose instead. The page's own cells
are printed beside ours and are never edited into agreement. Wall times are
this machine's, under this load. Read the direction, not the milliseconds."""


def render_table(rows: dict[int, Measurement], width: int) -> str:
    out = [
        "MEASURED HERE (this battery's harnesses, not the page's)",
        "",
        f"{'#':>2}  {'row':<34} {'frag':<7} {'both':<22} {'z3 alone':<22} "
        f"{'cvc5 alone':<22}",
        "-" * min(width, 112),
    ]
    for r in ROWS:
        m = rows.get(r.n)
        if m is None:
            continue
        mark = " ‡" if r.contested else ""
        out.append(
            f"{r.n:>2}  {r.name[:34]:<34} {m.fragment or '-':<7} "
            f"{m.both.render()[:22]:<22} {m.z3.render()[:22]:<22} "
            f"{m.cvc5.render()[:22]:<22}{mark}"
        )
    out.append("")
    out.append("THE PAGE'S CELLS, AS PUBLISHED (2026-08 hand-check, harnesses lost)")
    out.append("")
    out.append(
        f"{'#':>2}  {'row':<34} {'frag':<7} {'both':<22} {'z3 alone':<22} "
        f"{'cvc5 alone':<22} {'reconstruction':<15}"
    )
    out.append("-" * min(width, 128))
    for r in ROWS:
        if r.n not in rows:
            continue
        out.append(
            f"{r.n:>2}  {r.name[:34]:<34} {r.fragment:<7} "
            f"{r.page_both[:22]:<22} {r.page_z3[:22]:<22} "
            f"{r.page_cvc5[:22]:<22} {r.grade:<15}"
            + ("‡" if r.contested else "")
            + ("†" if r.published_notes else "")
        )
    if any(ROWS[r.n - 1].contested for r in ROWS if r.n in rows):
        out.append("")
        out.append("‡ this row's label admits readings that disagree about the "
                   "direction; see --rows and --variants.")
    if any(ROWS[r.n - 1].published_notes for r in ROWS if r.n in rows):
        out.append("† this row's own published cells are internally "
                   "inconsistent; --rows says how.")
    unstable = sorted(
        f"{n}[{which}]"
        for n, m in rows.items()
        for which in ("both", "z3", "cvc5")
        if getattr(m, which).unstable
    )
    if unstable:
        out.append("")
        out.append("! these cells did NOT give the same answer on every "
                   "repeat: " + ", ".join(unstable))
        out.append("  An answer that depends on the clock is not a property "
                   "of the backend, so every answer the repeats gave is "
                   "shown.")
        out.append("  `error` or `not measured` among them means the range "
                   "beside it covers only the repeats that DID measure "
                   "something — see the section on what could not be "
                   "measured, below.")
    hidden = sorted(
        f"{n}[{which}]+{getattr(m, which).unpublished_invocations}"
        for n, m in rows.items()
        for which in ("both", "z3", "cvc5")
        if getattr(m, which).unpublished_invocations
    )
    if hidden:
        out.append("")
        out.extend(_wrap("+N these cells invoked a backend N MORE times than "
                         "stelling published a latency for, so their "
                         "milliseconds are a lower bound: " + ", ".join(hidden),
                         width - 2, "  ", "   "))
        out.extend(_wrap(_UNPUBLISHED, width - 4, "    ", "  "))
    regions = sorted(
        f"{n}[{which}]x{getattr(m, which).region_invocations}"
        for n, m in rows.items()
        for which in ("both", "z3", "cvc5")
        if getattr(m, which).region_invocations
    )
    if regions:
        out.append("")
        out.append("  of those, an ADMITTED-REGION check ran on: "
                   + ", ".join(regions))
    return "\n".join(out)


def render_over_budget(rows: dict[int, Measurement], timeout_ms: int,
                       width: int) -> str:
    """Cells that ran past the budget they were given, and by how much.

    ``solver_timeout_ms=10000`` is not a ten-second wall. z3's ``:timeout`` is
    its own and lands near it; the cvc5 wheel is driven through a wall-guarded
    child process because its ``tlimit`` does not reliably preempt the
    coverings solver, and that guard is ``timeout_ms * 1.5 + 1 s`` — SIXTEEN
    seconds for a ten-second budget. A two-backend row where both time out
    therefore costs twenty-six seconds, not twenty. Nothing on the page says
    this, and a reader sizing a CI budget from the page's ten seconds would be
    out by 60%. Derived here from what the run actually measured rather than
    from the formula, so it stays true if the formula moves."""
    over: list[str] = []
    for r in ROWS:
        m = rows.get(r.n)
        if m is None:
            continue
        for label, c in (("both", m.both), ("z3 alone", m.z3),
                         ("cvc5 alone", m.cvc5)):
            # EVERY repeat's invocations, not the last one's. While
            # `Cell.invocations` was overwritten per repeat this loop saw a
            # third of the evidence at the page's own repeat count, so a
            # backend that ran past its budget on the first two repeats and
            # not the third went unreported.
            for repeat in c.invocations:
                for backend, answer, ms in repeat:
                    if ms > timeout_ms * 1.1:
                        over.append(
                            f"row {r.n} [{label}]: {backend} answered {answer} "
                            f"after {ms / 1000:.1f} s on a "
                            f"{timeout_ms / 1000:.0f} s budget")
    if not over:
        return ""
    return "\n".join([
        "CELLS THAT RAN PAST THEIR BUDGET, AND BY HOW MUCH",
        *_wrap("solver_timeout_ms is the SOLVER's limit, not a wall. The cvc5 "
               "wheel runs in a wall-guarded child process at timeout*1.5 + 1s "
               "because its own tlimit does not reliably preempt the coverings "
               "solver.", width - 2, "  "),
        *[f"  {line}" for line in dict.fromkeys(over)],
    ])


def render_unmeasured(rows: dict[int, Measurement], width: int) -> str:
    """What this run could not measure, and why — GROUPED BY REASON.

    An environment with no backend produces thirty identical "not measured"
    lines, and thirty identical lines is how a reader learns to skip a
    section. Grouping keeps the reason readable and still names every cell it
    covers, which is the thing that has to survive: a row silently absent from
    the table is exactly the failure this tool exists to end."""
    groups: dict[tuple[str, str], list[str]] = {}
    for r in ROWS:
        m = rows.get(r.n)
        if m is None:
            continue
        if m.fragment_reason:
            groups.setdefault(("fragment", m.fragment_reason), []).append(
                f"{r.n}")
        for label, c in (("both", m.both), ("z3 alone", m.z3),
                         ("cvc5 alone", m.cvc5)):
            # `mixed` belongs here too: a cell where SOME repeat raised or
            # reached no backend has a reason, and the reason is the point.
            if c.reason or c.outcome in ("not measured", "error", "mixed"):
                groups.setdefault((c.outcome or "not measured", c.reason),
                                  []).append(f"{r.n}[{label}]")
    if not groups:
        return ""
    out = ["ROWS AND CELLS THIS RUN COULD NOT MEASURE, AND WHY"]
    for (kind, reason), cells in groups.items():
        out.extend(_wrap(f"{kind}: {reason}", width - 2, "  "))
        out.extend(_wrap("affects " + ", ".join(cells), width - 6, "      "))
    return "\n".join(out)


def render_mechanism(rows: dict[int, Measurement], width: int) -> str:
    """The machine-independent half, printed on its own.

    Fragment, declared inputs and emitted element terms need jax and no
    solver. They are the part of the table a reader in the zero-solver lane
    can still check, and the part this project gates."""
    out = [
        "MECHANISM COLUMN (needs jax, needs NO solver, machine-independent)",
        "",
        f"{'#':>2}  {'row':<34} {'page frag':<10} {'measured':<10} "
        f"{'inputs':>6} {'terms':>6}",
        "-" * min(width, 84),
    ]
    for r in ROWS:
        m = rows.get(r.n)
        if m is None:
            continue
        # A BLANK FRAGMENT IS NOT A DISAGREEING ONE. This column read
        # `m.fragment == r.fragment` and called everything else DISAGREES, so
        # the no-jax lane — where nothing can be traced and every measured
        # fragment is empty — printed TEN `<- DISAGREES` markers while the
        # section below correctly said jax was missing. That is precisely the
        # rule this tool's own no-backend test asserts: an unmeasured thing
        # must never render as a negative result.
        if not m.fragment:
            agree = "   <- NOT MEASURED"
        elif m.fragment != r.fragment:
            agree = "   <- DISAGREES"
        else:
            agree = ""
        out.append(
            f"{r.n:>2}  {r.name[:34]:<34} {r.fragment:<10} "
            f"{m.fragment or '-':<10} "
            f"{'-' if m.inputs is None else m.inputs:>6} "
            f"{'-' if m.element_terms is None else m.element_terms:>6}{agree}"
        )
    if any(not m.fragment for m in rows.values()):
        out.append("")
        out.append("`<- NOT MEASURED` means nothing was traced for that row. "
                   "It is not a mismatch with the page; the reason is below.")
    return "\n".join(out)


def render_variants(variants: list[Measurement], width: int) -> str:
    if not variants:
        return ""
    out = [
        "ALTERNATE READINGS OF THE CONTESTED ROWS",
        "",
        "The spread IS the content. Each of these is a defensible reading of",
        "the same row label; where they disagree, the label does not choose,",
        "and no number from any of them can be filed against the page's row.",
        "",
        f"{'row':>3}  {'reading':<16} {'frag':<7} {'both':<22} {'z3 alone':<22} "
        f"{'cvc5 alone':<22}",
        "-" * min(width, 104),
    ]
    for m in variants:
        out.append(
            f"{m.n:>3}  {m.key[:16]:<16} {m.fragment or '-':<7} "
            f"{m.both.render()[:22]:<22} {m.z3.render()[:22]:<22} "
            f"{m.cvc5.render()[:22]:<22}"
        )
    out.append("")
    for v in VARIANTS:
        out.extend(_wrap(f"row {v.row} [{v.key}]: {v.predicate}", width - 2, "  "))
    return "\n".join(out)


# ---------------------------------------------------------------- entry point


def _row_selection(spec: str | None) -> list[Row]:
    if not spec:
        return list(ROWS)
    wanted = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or not 1 <= int(part) <= len(ROWS):
            raise SystemExit(
                f"--only-rows: {part!r} is not a row number; the battery has "
                f"rows 1..{len(ROWS)}"
            )
        wanted.add(int(part))
    return [r for r in ROWS if r.n in wanted]


def _as_json(env: Environment, rows: dict[int, Measurement],
             variants: list[Measurement]) -> str:
    def cell(c: Cell) -> dict:
        # `invocations` is a LIST PER REPEAT, in step with `ms`. It used to be
        # one flat list overwritten on every repeat, which paired a 3-entry
        # `ms` with one repeat's invocations as though they described the same
        # measurement.
        return {"outcome": c.outcome, "outcomes": c.outcomes,
                "unstable": c.unstable, "attempted": c.attempted,
                "decided": c.decided, "reason": c.reason, "errors": c.errors,
                "ms": c.ms, "status": c.status, "degraded": c.degraded,
                "region_invocations": c.region_invocations,
                "unpublished_invocations": c.unpublished_invocations,
                "invocations": [[{"backend": b, "answer": a, "ms": t}
                                 for b, a, t in repeat]
                                for repeat in c.invocations]}

    def meas(m: Measurement, r: Row | None) -> dict:
        d = {"n": m.n, "name": m.name, "key": m.key,
             "fragment": m.fragment, "fragment_reason": m.fragment_reason,
             "inputs": m.inputs, "element_terms": m.element_terms,
             "both": cell(m.both), "z3": cell(m.z3), "cvc5": cell(m.cvc5)}
        if r is not None:
            d["page"] = {"fragment": r.fragment, "both": r.page_both,
                         "z3": r.page_z3, "cvc5": r.page_cvc5,
                         "reconstruction": r.grade}
            d["grade"] = r.grade
            d["grade_means"] = GRADES[r.grade]
            d["named_object"] = r.named_object
            d["fixed_by_the_page"] = list(r.fixed)
            d["chosen_by_this_battery"] = list(r.chosen)
            d["contested"] = r.contested
            d["about_the_published_row"] = list(r.published_notes)
        return d

    return json.dumps(
        {
            "disclaimer": _BANNER,
            "grades": GRADES,
            # `cvc5_route` is a property, so `asdict` cannot see it — and it
            # is the field that says WHICH of the three routes to a backend
            # this run took, which is the thing the wheel-only probe used to
            # get wrong.
            "environment": dict(dataclasses.asdict(env),
                                backends=list(env.backends),
                                missing_backends=list(env.missing_backends),
                                cvc5_route=env.cvc5_route),
            "rows": [meas(rows[r.n], r) for r in ROWS if r.n in rows],
            "variants": [meas(m, None) for m in variants],
            "directions": direction_report(rows, variants),
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="solver_battery.py",
        description=(
            "Run a solver battery shaped like the ten rows of "
            "docs/choosing-a-solver-backend.md. NOT a re-derivation of that "
            "table: the harnesses behind it were never committed. Use --rows "
            "to see what each label fixes and what this battery chose."
        ),
    )
    p.add_argument("--rows", action="store_true",
                   help="print the row inventory and exit (no jax, no solver "
                        "needed)")
    p.add_argument("--fragments", action="store_true",
                   help="print only the mechanism column: fragment, declared "
                        "inputs, element terms (needs jax, no solver)")
    p.add_argument("--variants", action="store_true",
                   help="also drive the alternate readings of the contested rows")
    p.add_argument("--only-rows", metavar="N,N",
                   help="restrict to these row numbers")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                   help=f"repeats per cell (default {DEFAULT_REPEATS}; the "
                        f"page used 3)")
    p.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS,
                   help=f"solver_timeout_ms (default {DEFAULT_TIMEOUT_MS}, "
                        f"the page's budget)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--env", action="store_true",
                   help="print the environment and exit")
    args = p.parse_args(argv)

    width = int(os.environ.get("COLUMNS") or 100)
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.timeout_ms < 1:
        raise SystemExit("--timeout-ms must be a positive integer")

    if args.rows:
        print(render_rows(width))
        return 0

    # EVERY ARGUMENT IS VALIDATED BEFORE THE ENVIRONMENT IS TOUCHED. Probing
    # imports jax and sets `jax_enable_x64`, which is process-global; a run
    # that is going to die on a bad `--only-rows` must not leave that behind
    # first. (Measured: with the selection below the probe,
    # `tests/_state_guard.py` reported
    # `jax:enable_x64: False -> True` against a test that only ever asked for
    # row zero.)
    selected = _row_selection(args.only_rows)

    env = probe_environment(args.timeout_ms, args.repeats)
    if args.env:
        print(render_environment(env))
        return 0

    if args.json is False:
        print(_BANNER)
        print()
        print(render_environment(env))
        print()

    rows: dict[int, Measurement] = {}
    for r in selected:
        rows[r.n] = measure(r.build, r.n, r.name, "", env,
                            args.timeout_ms, args.repeats,
                            cells=not args.fragments)

    variants: list[Measurement] = []
    if args.variants and not args.fragments:
        chosen = {r.n for r in selected}
        for v in VARIANTS:
            if v.row not in chosen:
                continue
            variants.append(
                measure(v.build, v.row, f"row {v.row} [{v.key}]", v.key, env,
                        args.timeout_ms, args.repeats))

    if args.json:
        print(_as_json(env, rows, variants))
        return 0

    print(render_mechanism(rows, width))
    print()
    if not args.fragments:
        print(render_table(rows, width))
        print()
        if variants:
            print(render_variants(variants, width))
            print()
        over = render_over_budget(rows, args.timeout_ms, width)
        if over:
            print(over)
            print()
        print("\n".join(direction_report(rows, variants)))
        print()
    unmeasured = "" if args.fragments else render_unmeasured(rows, width)
    if unmeasured:
        print(unmeasured)
        print()
    if args.fragments:
        print("The fragment column is machine-independent and needs no "
              "backend. Nothing here was timed.")
    else:
        print("Wall times above are THIS MACHINE'S, at this load, on this "
              "build of each backend.\nThey are not a property of the "
              "solvers. The fragment column is machine-independent;\nnothing "
              "else here is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
