# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Run a solver battery shaped like the ten rows of
``docs/choosing-a-solver-backend.md``, and print what THIS machine measures.

    python tools/solver_battery.py                # the whole battery
    python tools/solver_battery.py --rows         # the inventory, no jax needed
    python tools/solver_battery.py --fragments    # the mechanism column, no solver needed
    python tools/solver_battery.py --variants     # the spread, see below
    python tools/solver_battery.py --json

**THE FIRST THING THIS FILE HAS TO SAY IS WHAT IT IS NOT.** It is *not* a
re-derivation of that page's ten-row table. The harnesses behind that table
were never committed, and the page says so itself; they cannot be recovered
from ten row LABELS, and this file does not pretend to have recovered them.
What it ships is a battery of ten harnesses BUILT TO THOSE LABELS, with every
parameter the label does not fix written down next to the harness that fixes
it (:attr:`Row.chosen`). Its numbers are its own.

That distinction is the whole point, and it is not pedantry — it is measured.
``docs/choosing-a-solver-backend.md``'s headline nonlinear finding is that
wide-and-shallow problems time z3 out while cvc5 answers in tenths of a
second. Three defensible readings of its own row label ``32 vars, 16
elementwise products`` were built and driven here (see :data:`VARIANTS`), three
repeats each, 2026-08-22, and they do not agree with each other, let alone
with the page:

    sum(a*b) <= sum(a)          z3 unsat 4.0-4.1 s  cvc5 UNKNOWN   <- REVERSED
    sum(a^2 + b^2 - 2ab) >= 0   z3 unsat  22-27 ms  cvc5 unsat 163-176 ms
    sum(a*b) - sum(b*a) >= 0    z3 unsat     7 ms   cvc5 unsat  70-72 ms
    the page's row              z3 TIMEOUT          cvc5 unsat 166-175 ms

The first is the most LITERAL reading — two arrays of sixteen, sixteen
elementwise products between them, thirty-two declared variables — and it runs
the page's direction backwards. The second reproduces the page's cvc5 cell to
within 3 ms and still does not reproduce its z3 cell, which is the cleanest
statement of the problem there is: **matching one cell is not identifying a
harness.**

So a tool that shipped ONE of those three and called the result "the row,
re-measured" would be publishing a harness choice as a finding about a solver.
An invented harness is worse than an unreproducible table, because it looks
reproducible. Hence the three rules this file is built to:

1. **Every row says what the page fixed and what this battery chose.**
   :attr:`Row.fixed` and :attr:`Row.chosen`; ``--rows`` prints both, and
   ``tests/test_solver_battery.py`` refuses a row whose ``chosen`` is empty —
   there is no way to add a row here that claims to have been reconstructed.
2. **The page's cells are carried verbatim, beside ours, never overwritten.**
   :attr:`Row.page_both` / ``page_z3`` / ``page_cvc5`` are the 2026-08
   hand-check as published. This tool prints them next to what it measured and
   states, per row, whether the DIRECTION agreed. It never edits them into
   agreement.
3. **Milliseconds are labelled as one machine's, always.** The page's own
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

WHAT IS MEASURED, EXACTLY. Not the ``check()`` call: that includes tracing,
jit, interval propagation and — on a VERIFIED — the vacuity widen re-check,
which runs the whole escalation a SECOND time and would double every
discharged row. The number reported per cell is the sum of the per-invocation
milliseconds stelling itself publishes in the verdict's notes
(``assert #0: z3 (wheel) answered unsat in 7ms``), over the invocations of the
first escalation only. The page never says what it timed; that is one more
thing :attr:`Row.chosen` records rather than guesses at.

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
    #: What the page's row label DOES determine, and this harness honours.
    fixed: tuple[str, ...]
    #: What the label does NOT determine, and this battery therefore chose.
    #: Never empty — a row with nothing here would be claiming the label
    #: pinned the harness, and no label on this page does.
    chosen: tuple[str, ...]
    build: str  # key into BUILDERS
    #: Set when this battery's reading did not reproduce the page's direction
    #: AND the label does not choose between readings that disagree. It is a
    #: refusal to publish a number, not a finding about a backend.
    contested: str = ""


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

#: The one parameter the page leaves open that could be RECOVERED rather than
#: chosen, and it was recovered by arithmetic on the page's own cells rather
#: than by picking whichever definition made the numbers agree.
#:
#: The page never says what it timed. But its `both` column is the SUM of its
#: own two single-backend cells, and three of the four second-scale rows land
#: inside 0.2 s of it — row 10: cvc5's 16.0 s wall guard (measured here) plus
#: z3's 689-702 ms is 16.69-16.70 s against a published `~16.7 s`; row 7:
#: z3's 10.0-10.1 s timeout plus cvc5's 166-175 ms is 10.2-10.3 s against
#: `~10.3 s`; row 8: the same timeout plus 772-792 ms is 10.8-10.9 s against
#: `~11.0 s`. Row 9 is the one that does not fit, by 0.3-0.5 s. So this
#: battery sums the per-invocation milliseconds stelling publishes in the
#: verdict's notes, which is the definition that reproduces the page's own
#: arithmetic — NOT the `check()` wall, which on a discharged row also pays
#: the vacuity widen re-check and would double it.
_TIMED = (
    "what was timed — RECOVERED from the page's own arithmetic rather than "
    "picked; see the note above the inventory"
)

ROWS: tuple[Row, ...] = (
    Row(
        n=1,
        name="scalar, linear",
        fragment="QF_LRA",
        page_both="unsat, 78–112 ms",
        page_z3="unsat, 8–9 ms",
        page_cvc5="unsat, 71–84 ms",
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
        fixed=("two declared scalars", "total degree 2",
               "the AM-GM inequality, whose two-variable degree-2 form is "
               "x^2 + y^2 >= 2xy and is named, not chosen", "QF_NRA",
               "true everywhere, so unsat"),
        chosen=("the declared box [-1, 1]^2 — AM-GM is a statement over all "
                "of R^2 and a declared harness must bound it somewhere",
                _TIMED),
        build="amgm",
    ),
    Row(
        n=5,
        name="2 vars, degree 6 (Motzkin)",
        fragment="QF_NRA",
        page_both="unsat, 92–106 ms",
        page_z3="unsat, 12–13 ms",
        page_cvc5="unsat, 81–83 ms",
        fixed=("two declared scalars", "total degree 6",
               "the Motzkin polynomial x^4 y^2 + x^2 y^4 - 3 x^2 y^2 + 1, "
               "which is a named object and not a choice", "QF_NRA",
               "nonnegative everywhere, so unsat"),
        chosen=("the declared box [-2, 2]^2", "the association of the degree-6 "
                "monomials — (x2*x2)*y2 rather than x2*(x2*y2), which is the "
                "same polynomial and not the same emitted script", _TIMED),
        build="motzkin",
    ),
    Row(
        n=6,
        name="1 var, degree 3, false",
        fragment="QF_NRA",
        page_both="sat, 87–88 ms",
        page_z3="sat, 11 ms",
        page_cvc5="sat, 69–71 ms",
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
    ),
    Row(
        n=10,
        name="12-factor product chain",
        fragment="QF_NRA",
        page_both="unsat, ~16.7 s",
        page_z3="unsat, 689–702 ms",
        page_cvc5="**UNKNOWN** (timeout)",
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


def build_amgm(row_n: int):
    from stelling.harness import any_array, assert_

    def amgm():
        x = any_array((), _f64(), (-1.0, 1.0))
        y = any_array((), _f64(), (-1.0, 1.0))
        return assert_(x * x + y * y >= 2.0 * x * y)

    return amgm


def build_motzkin(row_n: int):
    from stelling.harness import any_array, assert_

    def motzkin():
        x = any_array((), _f64(), (-2.0, 2.0))
        y = any_array((), _f64(), (-2.0, 2.0))
        x2 = x * x
        y2 = y * y
        return assert_(x2 * x2 * y2 + x2 * y2 * y2 - 3.0 * x2 * y2 + 1.0 >= 0.0)

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

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(n for n, v in (("z3", self.z3), ("cvc5", self.cvc5)) if v)

    @property
    def missing_backends(self) -> tuple[str, ...]:
        return tuple(n for n, v in (("z3", self.z3), ("cvc5", self.cvc5)) if not v)


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
    tree = "not importable"
    try:
        import stelling
        from stelling import _optional

        stelling_version = stelling.__version__
        tree = str(pathlib.Path(stelling.__file__).resolve().parent)
        z3v = _optional.version("z3")
        cvc5v = _optional.version("cvc5")
        cvc5_bin = _optional.cvc5_binary()
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


@dataclass
class Cell:
    """One (row, portfolio) measurement, repeated ``repeats`` times."""

    outcome: str = ""  # unsat | sat | UNKNOWN | not measured | error
    reason: str = ""  # why, when there is no outcome to give
    ms: list[int] = field(default_factory=list)
    invocations: list[tuple[str, str, int]] = field(default_factory=list)
    status: str = ""  # the verdict status behind the outcome
    degraded: bool = False
    #: Every repeat's outcome, in order. A cell whose repeats DISAGREED is not
    #: a cell with a range in it — it is a cell whose answer depends on the
    #: clock, and collapsing it to the last repeat's answer is how a table
    #: comes to publish a decision that only happens sometimes.
    outcomes: list[str] = field(default_factory=list)

    @property
    def unstable(self) -> bool:
        return len(set(self.outcomes)) > 1

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
    """Drive one (harness, portfolio) cell ``repeats`` times."""
    from stelling.preconditions import check

    cell = Cell()
    for _ in range(repeats):
        kwargs = {} if only is None else {"solver": only}
        try:
            v = check(_harness(build, row_n), vacuity_mode="inputs-only",
                      solver_timeout_ms=timeout_ms, **kwargs)
        except Exception as e:  # noqa: BLE001 — quoted, never raised through
            cell.outcome = "error"
            cell.reason = f"{type(e).__name__}: {e}"
            return cell
        invocations = []
        total = 0
        for note in v.notes:
            m = _ANSWERED.match(note)
            if m is None:
                continue
            invocations.append((m["label"], m["answer"], int(m["ms"])))
            total += int(m["ms"])
        cell.status = v.status
        cell.degraded = any("portfolio degraded" in n for n in v.notes)
        cell.outcome = {"VERIFIED": "unsat", "REFUTED": "sat"}.get(v.status, "UNKNOWN")
        if not invocations:
            # Escalation happened but nobody was invoked: a decline, or no
            # backend. The reason is stelling's, quoted.
            cell.outcome = "not measured"
            cell.reason = v.obligations[0].detail if v.obligations else "no obligation"
            return cell
        cell.invocations = invocations
        cell.ms.append(total)
        cell.outcomes.append(cell.outcome)
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
            cell.reason = ('no SMT backend is installed — pip install '
                           '"stelling[solvers]"')
        return m
    if len(env.backends) == 2:
        m.both = measure_cell(build, row_n, None, timeout_ms, repeats)
    else:
        m.both.outcome = "not measured"
        m.both.reason = (f"the two-backend column needs both backends; "
                         f"{' and '.join(absent)} not installed")
    for name_, cell_attr in (("z3", "z3"), ("cvc5", "cvc5")):
        if name_ in env.backends:
            setattr(m, cell_attr,
                    measure_cell(build, row_n, name_, timeout_ms, repeats))
        else:
            c = Cell(outcome="not measured",
                     reason=f"{name_} is not installed")
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
    attempted = [n for n in lra
                 if cell(n, "z3").outcome not in ("not measured", "")
                 and cell(n, "cvc5").outcome not in ("not measured", "")]
    decided = [n for n in attempted
               if cell(n, "z3").outcome in ("unsat", "sat")
               and cell(n, "cvc5").outcome in ("unsat", "sat")]
    if lra and not attempted:
        # NOT THE SAME AS "DID NOT HOLD", and writing it as one would be the
        # exact defect this tool exists to end: a missing measurement rendered
        # as a negative result.
        out.append(
            "FINDING 1 (QF_LRA: both backends decide everything): NOT "
            "MEASURED — no backend ran on the linear rows in this "
            "environment."
        )
    elif lra:
        held = len(decided) == len(attempted)
        out.append(
            f"FINDING 1 (QF_LRA: both backends decide everything): "
            f"{'HELD' if held else 'DID NOT HOLD'} — {len(decided)} of "
            f"{len(attempted)} linear rows decided by both backends alone."
        )
        ratios = []
        for n in decided:
            z, c = cell(n, "z3"), cell(n, "cvc5")
            if z.ms and c.ms and min(z.ms) > 0:
                ratios.append(min(c.ms) / min(z.ms))
        if ratios:
            out.append(
                f"  z3 was faster than cvc5 on every linear row by "
                f"{min(ratios):.1f}x to {max(ratios):.1f}x. THAT IS A TIMING, "
                f"on one machine, and part of it is a process spawn rather "
                f"than solving — see the page."
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
        "do not fix a harness, and the readings that fit them disagree."
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
    two_backend = [n for n, m in rows.items()
                   if m.both.outcome not in ("not measured", "", "error")]
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
    """The inventory: what the page fixes, and what this battery chose.

    Pure data — no jax, no solver, no measurement. This is the section a
    reader should read before any table, and it is why the tool can be honest
    about a row it cannot reconstruct."""
    out = [
        "THE TEN ROWS, AND WHAT EACH LABEL DOES NOT FIX",
        "",
        "The harnesses behind docs/choosing-a-solver-backend.md's table were",
        "never committed. These are harnesses BUILT TO ITS ROW LABELS. For",
        "every row, `page fixes` is what the label determines and `chose here`",
        "is what it does not — the parameters this battery had to pick, and",
        "which a different picker would pick differently.",
        "",
        "ONE OF THOSE PARAMETERS WAS RECOVERED RATHER THAN PICKED, and it is",
        "the same one on every row, so it is argued here once. The page never",
        "says WHAT it timed. It did not have to: its `both` cells are the SUM",
        "of its own two single-backend cells, and three of the four",
        "second-scale rows land inside 0.2 s of that sum —",
        "",
        *[
            f"  row {n:>2}   {sums:<49} vs {pub}"
            for n, sums, pub in (
                (10, "16.0 s wall guard + 689-702 ms = 16.69-16.70 s", "~16.7 s"),
                (7, "10.0-10.1 s timeout + 166-175 ms = 10.2-10.3 s", "~10.3 s"),
                (8, "the same timeout + 772-792 ms = 10.8-10.9 s", "~11.0 s"),
                (9, "123-133 ms + 8.3-8.5 s = 8.4-8.6 s", "~8.1 s   <- does not fit"),
            )
        ],
        "",
        "So this battery sums the per-invocation milliseconds stelling",
        "publishes in the verdict's notes. NOT the check() wall, which on a",
        "discharged row also pays the vacuity widen re-check and would double",
        "every one of them.",
        "",
    ]
    for r in ROWS:
        out.append(f"[{r.n:2d}] {r.name}   ({r.fragment})")
        out.append(f"     page's cells: both {r.page_both} | z3 {r.page_z3} "
                   f"| cvc5 {r.page_cvc5}")
        for f in r.fixed:
            out.extend(_wrap(f"page fixes  : {f}", width - 5, "     ", "              "))
        for c in r.chosen:
            out.extend(_wrap(f"chose here  : {c}", width - 5, "     ", "              "))
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
        f"  z3              : {env.z3 or 'NOT INSTALLED'}",
        f"  cvc5            : {env.cvc5 or 'NOT INSTALLED'}"
        + (f"  (external binary: {env.cvc5_binary})" if env.cvc5_binary else ""),
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
        f"{'cvc5 alone':<22}"
    )
    out.append("-" * min(width, 112))
    for r in ROWS:
        if r.n not in rows:
            continue
        out.append(
            f"{r.n:>2}  {r.name[:34]:<34} {r.fragment:<7} "
            f"{r.page_both[:22]:<22} {r.page_z3[:22]:<22} {r.page_cvc5[:22]:<22}"
        )
    if any(ROWS[r.n - 1].contested for r in ROWS if r.n in rows):
        out.append("")
        out.append("‡ this row's label admits readings that disagree about the "
                   "direction; see --rows and --variants.")
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
                   "of the backend. Both answers are shown.")
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
            for backend, answer, ms in c.invocations:
                if ms > timeout_ms * 1.1:
                    over.append(
                        f"row {r.n} [{label}]: {backend} answered {answer} "
                        f"after {ms / 1000:.1f} s on a {timeout_ms / 1000:.0f} s "
                        f"budget")
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
            if c.outcome in ("not measured", "error"):
                groups.setdefault((c.outcome, c.reason), []).append(
                    f"{r.n}[{label}]")
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
        agree = "" if m.fragment == r.fragment else "   <- DISAGREES"
        out.append(
            f"{r.n:>2}  {r.name[:34]:<34} {r.fragment:<10} "
            f"{m.fragment or '-':<10} "
            f"{'-' if m.inputs is None else m.inputs:>6} "
            f"{'-' if m.element_terms is None else m.element_terms:>6}{agree}"
        )
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
        return {"outcome": c.outcome, "outcomes": c.outcomes,
                "unstable": c.unstable, "reason": c.reason, "ms": c.ms,
                "status": c.status, "degraded": c.degraded,
                "invocations": [{"backend": b, "answer": a, "ms": t}
                                for b, a, t in c.invocations]}

    def meas(m: Measurement, r: Row | None) -> dict:
        d = {"n": m.n, "name": m.name, "key": m.key,
             "fragment": m.fragment, "fragment_reason": m.fragment_reason,
             "inputs": m.inputs, "element_terms": m.element_terms,
             "both": cell(m.both), "z3": cell(m.z3), "cvc5": cell(m.cvc5)}
        if r is not None:
            d["page"] = {"fragment": r.fragment, "both": r.page_both,
                         "z3": r.page_z3, "cvc5": r.page_cvc5}
            d["fixed_by_the_page"] = list(r.fixed)
            d["chosen_by_this_battery"] = list(r.chosen)
            d["contested"] = r.contested
        return d

    return json.dumps(
        {
            "disclaimer": _BANNER,
            "environment": dataclasses.asdict(env),
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
