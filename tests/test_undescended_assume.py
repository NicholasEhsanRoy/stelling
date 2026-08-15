# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""An `assume` the propagator's walk never enters (audit 0.2.0 S13).

THE DEFECT, AND IT REACHED THE RELEASED 0.1.0. The propagator descends the
`DEFAULT_TRANSPARENT` wrappers and `cond`; it does not descend `scan` or
`while_loop`. A `stelling_assume` written inside one of those bodies was
therefore never classified — it did not narrow, it was not forwarded to the
solver, and it left NO ledger entry. Every rule that asks "was an assume
dropped?" reads the ledger or the flag the ledger's writers set, so every one
of them saw a query with no assume in it at all.

THREE RULES, ONE MISSING RECORD:

1. **The withholding rule — a FALSE REFUTED.** `assume_dropped` stayed
   `False`, nothing withheld, and the solver searched the un-narrowed box and
   returned a witness the user's own precondition excludes. Measured on the
   `v0.1.0` tag and on `main`: `x, y ∈ [-10,10]`, `assume(x <= y)` inside a
   `lax.scan` body, `assert_(x - y <= 0.0)` — REFUTED at `x = 0, y = -1`,
   where `x <= y` is false. `test_the_scan_body_precondition_is_honoured`
   and its `while_loop` twin pin it, against a ground truth this file
   computes in exact `Fraction` arithmetic: the assert holds at EVERY
   admitted point, so no admitted point can be a counterexample.
2. **The admitted-region gate** — closed in audit B3 by requiring
   `ledger_covers`; pinned in `test_vacuous_precondition.py` section 8.
3. **`REGION_NOT_ASKED`** — when an obligation's slice carries no forwarded
   relational axiom the region question was skipped outright, on the stated
   ground that an empty region is already the propagation's
   `UnsatisfiableAssumptionError`. That refusal fires when a NARROWING
   empties a box, and an assume that never narrowed empties nothing.
   Measured: two assumes inside a `scan` body, `x < y` and `y < x`, an
   obligation the solver discharges — VERIFIED, stamped entirely clean, with
   no mention of an assume anywhere, over a region an exact 41x41 `Fraction`
   grid shows admits 0 points.

THE FIX IS THE RECORD, NOT THE THREE RULES.
`propagate._record_undescended_assumes` reconciles the ledger against the
STATIC set of assume equations the query contains — `_assume_equations`,
whose totality is checked against an independent walk of the raw jax jaxpr in
`test_vacuous_precondition.py` — and writes a `dropped` disposition, a note
and a stamped assumption for every assume the walk never classified. It does
NOT descend the loop: a loop body's assume is a per-iteration statement about
a carry this analysis does not model, and this batch is about not silently
ignoring it.

THE COST IS REAL AND IT IS IN THIS FILE. Every such harness now has a dropped
constraint, so a definite violation is withheld even when the region is
inhabited and the refutation was sound
(`test_a_sound_refutation_inside_a_scan_is_now_WITHHELD`), and a substantive
discharge is caveated (`test_an_inhabited_region_keeps_its_VERIFIED`). Both
are the conservative direction of a precondition nothing honoured.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

# THE BARE IDIOM ON PURPOSE. A custom ``reason=`` replaces pytest's standard
# "could not import 'jax'" message, and that message is what
# ``test_skip_inventory.py``'s ``_IMPORT_GATE`` matches to disclose the gate.
pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from stelling import solvers  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import (  # noqa: E402
    ASSUME_DROPPED,
    UNCERTIFIED_PRECONDITION_PREFIX,
    UNDESCENDED_ASSUME_ASSUMPTION,
    _assume_equation_ids,
    _assume_equations,
    ledger_covers,
    propagate,
    unaccounted_assumes,
)
from stelling.solvers import (  # noqa: E402
    REGION_UNASKED_MECHANISM,
    UNCERTIFIED_REGION_ASSUMPTION,
)

try:
    from stelling import _optional
    HAVE_SOLVER = (
        _optional.available("z3")
        or _optional.available("cvc5")
        or _optional.cvc5_binary() is not None
    )
except Exception:  # pragma: no cover - environment probe only
    HAVE_SOLVER = False

need_solver = pytest.mark.skipif(not HAVE_SOLVER, reason="needs an SMT solver")

TIMEOUT = 5000


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """This module declares float64 inputs, so it must ask for x64 ITSELF.

    Every harness here is `any_array((), "float64", …)`, and in a float32
    session those declarations TRUNCATE: the obligations decline and the
    assertions fail on a `DeclinedObligation` rather than on anything this
    file is about. CI runs plain `pytest` with no `JAX_ENABLE_X64`, so a
    module that asks for nothing is a module that only passes on a
    developer's machine.

    A module-scoped fixture that SAVES AND RESTORES is the house pattern, and
    the restore is the load-bearing half: a bare module-scope
    `jax.config.update` runs at COLLECTION, before any test, and sets x64 for
    the whole session.
    """
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ---------------------------------------------------------------------------
# GROUND TRUTH, computed here, exactly, in `Fraction` — never by the tool
# under test. Each function is a fact about a HARNESS, and a fact about the
# harness established by stelling would make every test below circular.
# ---------------------------------------------------------------------------

_LO, _HI = Fraction(-10), Fraction(10)


def _grid(n):
    return [_LO + (_HI - _LO) * Fraction(i, n - 1) for i in range(n)]


def _region_census(admits, violates, n=41):
    """``(admitted points, admitted points violating the obligation)`` over an
    exact ``n x n`` `Fraction` grid of ``[-10, 10]^2``.

    ``admits(x, y)`` is the conjunction of the harness's assumes and
    ``violates(x, y)`` the negation of its assert, both written as ordinary
    Python on exact rationals. No float, no solver, no stelling.
    """
    pts = _grid(n)
    admitted = [(x, y) for x in pts for y in pts if admits(x, y)]
    return len(admitted), sum(1 for x, y in admitted if violates(x, y))


# ---------------------------------------------------------------------------
# THE HARNESSES. Each shape twice — `scan` and `while_loop` — because the
# defect is "the walk does not enter this construct" and one construct is one
# sample of that class.
# ---------------------------------------------------------------------------


def s13_scan():
    """AUDIT 0.2.0 S13's own reproducer, `scan` form.

    The carry is `x` at every iteration, so the body's `assume(c <= y)` is
    the precondition `x <= y`, and the assert is EXACTLY that predicate. On
    `v0.1.0` and on `main`: REFUTED at `x = 0, y = -1`.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))

    def body(c, _):
        assume(c <= y)
        return c, 0.0

    lax.scan(body, x, jnp.zeros((2,)))
    return assert_(x - y <= 0.0)


def s13_while():
    """The same precondition in a `while_loop` body."""
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))

    def cond(state):
        i, _ = state
        return i < 2

    def body(state):
        i, c = state
        assume(c <= y)
        return (i + 1, c)

    lax.while_loop(cond, body, (jnp.int32(0), x))
    return assert_(x - y <= 0.0)


def s13_control():
    """THE CONTROL: the identical precondition, written at the top level.

    It VERIFIES, which is what makes the `scan` form's REFUTED a statement
    about where the assume was written and not about the mathematics.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    assume(x <= y)
    return assert_(x - y <= 0.0)


def not_asked_scan():
    """S13's THIRD DOOR: an empty region and no forwarded axiom anywhere.

    Both assumes sit in the `scan` body, so `relational_assumes` is empty,
    the obligation's slice carries none, and the admitted-region check was
    never asked. The assert is true at every point of the declared box — the
    solver discharges it — so the VERIFIED is sound; what was missing is any
    word that the precondition it was asked under admits no point at all.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))

    def body(c, _):
        assume(x < y)
        assume(y < x)
        return c, 0.0

    lax.scan(body, x, jnp.zeros((2,)))
    return assert_((x - y) * (x - y) >= 0.0)


def not_asked_while():
    """The same empty region in a `while_loop` body."""
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))

    def cond(state):
        i, _ = state
        return i < 2

    def body(state):
        i, c = state
        assume(x < y)
        assume(y < x)
        return (i + 1, c)

    lax.while_loop(cond, body, (jnp.int32(0), x))
    return assert_((x - y) * (x - y) >= 0.0)


def inhabited_scan():
    """THE CONTROL THAT PROVES THE RIGHT THING WAS FIXED.

    An assume inside a `scan` body whose region is INHABITED and whose
    obligation genuinely HOLDS — at every admitted point, and in fact at
    every declared point, which is why the solver discharges it. A repair
    that turned every loop-body assume into UNKNOWN or REFUTED would pass
    every test above this one and be worthless.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))

    def body(c, _):
        assume(c <= y)
        return c, 0.0

    lax.scan(body, x, jnp.zeros((2,)))
    return assert_((x - y) * (x - y) >= 0.0)


def sound_refutation_scan():
    """THE MEASURED COST, as a harness rather than a paragraph.

    The region is inhabited and the assert is false at EVERY admitted point,
    so a REFUTED here would be sound — and it is withheld anyway, because
    nothing on this run establishes that the region is non-empty (the
    certificate's requirement is the same static assume set the walk missed).
    One-sided in the safe direction, and a real loss of coverage.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))

    def body(c, _):
        assume(c <= y)
        return c, 0.0

    lax.scan(body, x, jnp.zeros((2,)))
    return assert_(x - y >= 1.0)


def jit_inside_scan():
    """The chain, not one name: `jit` inside `scan`.

    The innermost enclosing primitive matches the source line and the
    outermost is the one that is not descended, so the disposition names
    both, in order.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))

    @jax.jit
    def inner(c):
        assume(c <= y)
        return c

    def body(c, _):
        return inner(c), 0.0

    lax.scan(body, x, jnp.zeros((2,)))
    return assert_(x - y <= 0.0)


# ---------------------------------------------------------------------------
# 0. the premises, measured
# ---------------------------------------------------------------------------


def test_the_s13_region_is_inhabited_and_the_assert_holds_on_all_of_it():
    """THE PREMISE OF THE HEADLINE. Two facts, and the second is what makes
    the pre-fix REFUTED false rather than merely unqualified: under
    `x <= y` there are admitted points (so the query is not vacuous) and
    `x - y <= 0` is true at every one of them (so no admitted point is a
    counterexample). The witness the tool returned, `x = 0, y = -1`, is not
    admitted — `0 <= -1` is false.
    """
    admitted, violating = _region_census(
        admits=lambda x, y: x <= y, violates=lambda x, y: not (x - y <= 0)
    )
    assert admitted == 861
    assert violating == 0
    assert not (Fraction(0) <= Fraction(-1))


def test_the_not_asked_region_admits_no_point():
    """THE PREMISE OF THE THIRD DOOR: `x < y ∧ y < x` gives `x < x`."""
    admitted, _ = _region_census(
        admits=lambda x, y: x < y and y < x, violates=lambda x, y: False
    )
    assert admitted == 0


def test_the_sound_refutation_region_is_inhabited_and_wholly_violating():
    """THE PREMISE OF THE COST: under `x <= y` the assert `x - y >= 1` is
    false at every admitted point, so the REFUTED this run withholds would
    have been sound."""
    admitted, violating = _region_census(
        admits=lambda x, y: x <= y, violates=lambda x, y: not (x - y >= 1)
    )
    assert admitted == 861
    assert violating == admitted


# ---------------------------------------------------------------------------
# 1. THE RECORD — the root cause, with no solver in it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", [s13_scan, s13_while, jit_inside_scan])
def test_an_assume_the_walk_never_enters_is_recorded_as_dropped(harness):
    """The propagation's ledger has a row for it, the row is `dropped`, and
    the flag every withholding rule reads is set.

    Before the repair all three were absent: one static assume equation, an
    EMPTY ledger, `assume_dropped` False.
    """
    q = trace(harness)
    p = propagate(q)
    assert len(_assume_equation_ids(q.jaxpr)) == 1
    (entry,) = p.assume_ledger
    assert entry.kind == ASSUME_DROPPED
    assert p.assume_dropped is True
    # ... and the filter every rule reads can now SEE it
    assert unaccounted_assumes(p.assume_ledger, ()) == (entry,)


@pytest.mark.parametrize("harness", [s13_scan, s13_while, jit_inside_scan])
def test_the_ledger_now_covers_every_assume_the_query_contains(harness):
    """`ledger_covers` is a POSTCONDITION of `propagate`, not a question
    about the walk's reach. It answered False on exactly these queries."""
    q = trace(harness)
    assert ledger_covers(propagate(q).assume_ledger, q.jaxpr) is True


@pytest.mark.parametrize(
    "harness,inside",
    [(s13_scan, "'scan'"), (s13_while, "'while'"),
     (jit_inside_scan, "'scan' -> 'jit'")],
)
def test_the_reason_says_never_classified_and_names_the_construct(
    harness, inside
):
    """"Dropped" alone would send a reader looking for the classifier that
    gave up, and there was none. The sentence has to say the assume was
    never seen, and name what it was written inside — outermost first, so a
    `jit` inside a `scan` names both and the reader can tell which matches
    their source line."""
    (entry,) = propagate(trace(harness)).assume_ledger
    assert "NEVER CLASSIFIED" in entry.reason
    assert inside in entry.reason
    assert "NO EFFECT on the analysis" in entry.reason
    assert entry.where and "unknown location" not in entry.where


def test_the_static_walk_reports_the_enclosing_chain():
    """The mapping the reason is built from, on its own. One traversal
    serves both readers: the ids ARE this mapping's keys, so the totality
    property two soundness rules rest on cannot drift from the naming."""
    q = trace(jit_inside_scan)
    ((eqn, path),) = _assume_equations(q.jaxpr).values()
    assert eqn.primitive == "stelling_assume"
    assert path == ("scan", "jit")
    assert frozenset(_assume_equations(q.jaxpr)) == _assume_equation_ids(q.jaxpr)
    # a top-level assume has an empty chain — the mapping is not only about
    # the un-descended ones
    ((_, top),) = _assume_equations(trace(s13_control).jaxpr).values()
    assert top == ()


@pytest.mark.parametrize("harness", [s13_scan, s13_while])
def test_the_run_says_so_in_its_notes_and_its_stamped_assumptions(harness):
    """A reader must be able to learn, from the verdict, that an `assume`
    they wrote had no effect. Before the repair the verdict mentioned no
    assume at all — not in the notes, not in the coverage line, not in the
    stamp."""
    p = propagate(trace(harness))
    assert any("assume NEVER CLASSIFIED" in n for n in p.notes), p.notes
    assert UNDESCENDED_ASSUME_ASSUMPTION in p.assumptions
    assert UNDESCENDED_ASSUME_ASSUMPTION.startswith(
        UNCERTIFIED_PRECONDITION_PREFIX
    )


# ---------------------------------------------------------------------------
# 2. RULE ONE — the withholding rule, and the FALSE REFUTED it let through
# ---------------------------------------------------------------------------


@need_solver
@pytest.mark.parametrize("harness", [s13_scan, s13_while])
def test_the_scan_body_precondition_is_honoured(harness):
    """THE HEADLINE, and the one that reaches the released 0.1.0.

    Measured before the repair on `v0.1.0` and on `main`: REFUTED, witness
    `x = 0, y = -1`, replay-confirmed — at a point the user's own
    `assume(x <= y)` excludes, while
    `test_the_s13_region_is_inhabited_and_the_assert_holds_on_all_of_it`
    shows the assert holds at every admitted point.

    Reverting `_record_undescended_assumes` restores the REFUTED, which is
    what makes this test load-bearing rather than descriptive.
    """
    v = check(harness, vacuity_mode="all", solver_timeout_ms=TIMEOUT)
    assert v.status != "REFUTED", v.render()
    assert v.witnesses == ()
    assert any("WITHHELD from REFUTED" in n for n in v.notes), v.notes


@need_solver
def test_the_same_precondition_at_the_top_level_still_verifies(harness=None):
    """THE CONTROL FOR THE HEADLINE: the repair is about where the assume
    was written, so the top-level spelling must be untouched."""
    v = check(s13_control, vacuity_mode="all", solver_timeout_ms=TIMEOUT)
    assert v.status == "VERIFIED", v.render()


@need_solver
def test_a_sound_refutation_inside_a_scan_is_now_WITHHELD():
    """THE COST, stated as a test so it cannot be forgotten.

    The region is inhabited and the assert is false at every admitted point
    (`test_the_sound_refutation_region_is_inhabited_and_wholly_violating`),
    so this REFUTED would have been sound. It is withheld anyway: nothing on
    the run establishes the region is non-empty, and the rule is one-sided
    in the direction that never mints a witness outside a precondition.
    """
    v = check(sound_refutation_scan, vacuity_mode="all",
              solver_timeout_ms=TIMEOUT)
    assert v.status == "UNKNOWN", v.render()
    assert any("WITHHELD from REFUTED" in n for n in v.notes), v.notes


# ---------------------------------------------------------------------------
# 3. RULE THREE — REGION_NOT_ASKED, the skip whose ground was untrue
# ---------------------------------------------------------------------------


@need_solver
@pytest.mark.parametrize("harness", [not_asked_scan, not_asked_while])
def test_a_discharge_with_no_forwarded_axiom_is_not_stamped_clean(harness):
    """THE THIRD DOOR. No relational assume exists, so no script carries an
    axiom, so no admitted-region check is asked — and the region is empty.

    Before the repair: VERIFIED with no mention of an assume anywhere.
    The discharge is SOUND and stays VERIFIED (the solver ran over a set that
    contains the assumed region); what it may not do is stay clean.
    """
    q = trace(harness)
    p = propagate(q)
    # the premise of the rule: nothing was forwarded, so the old skip fired
    assert p.relational_assumes == ()
    esc = solvers.escalate(q, p, solvers.SolverConfig(timeout_ms=TIMEOUT))
    assert [r.outcome for r in esc.records] == [solvers.OB_DISCHARGED]
    assert esc.region_uncertified == (0,)

    v = check(harness, vacuity_mode="all", solver_timeout_ms=TIMEOUT)
    assert v.status == "VERIFIED", v.render()
    assert UNCERTIFIED_REGION_ASSUMPTION in v.stamp.assumptions


@need_solver
def test_the_unasked_note_states_the_absence_and_not_an_answer():
    """The other three mechanism sentences all begin "the admitted-region
    check answered …". On this run it answered nothing — it was never run —
    so quoting any of them would state a mechanism that did not fire. The
    conjuncts are still named, in the same words the partial mechanism names
    them."""
    q = trace(not_asked_scan)
    esc = solvers.escalate(
        q, propagate(q), solvers.SolverConfig(timeout_ms=TIMEOUT)
    )
    notes = esc.records[0].notes
    assert any(REGION_UNASKED_MECHANISM in n for n in notes), notes
    assert not any("answered sat" in n for n in notes), notes
    assert not any("could not decide" in n for n in notes), notes
    assert any("NEVER CLASSIFIED" in n for n in notes), notes


# ---------------------------------------------------------------------------
# 4. THE CONTROL: a loop-body assume that costs nothing it should not
# ---------------------------------------------------------------------------


@need_solver
def test_an_inhabited_region_keeps_its_VERIFIED():
    """A sound, substantive discharge under a loop-body assume must not
    become a REFUTED and must not become an UNKNOWN. It is CAVEATED — the
    probe cannot certify a region it never evaluated — and that caveat is
    the whole of the change to this harness's verdict."""
    admitted, violating = _region_census(
        admits=lambda x, y: x <= y,
        violates=lambda x, y: not ((x - y) * (x - y) >= 0),
    )
    assert admitted == 861 and violating == 0
    v = check(inhabited_scan, vacuity_mode="all", solver_timeout_ms=TIMEOUT)
    assert v.status == "VERIFIED", v.render()
    assert UNCERTIFIED_REGION_ASSUMPTION in v.stamp.assumptions


def test_a_query_with_no_assume_at_all_is_untouched():
    """The reconciliation must be silent on every query it has nothing to
    say about — no note, no flag, no ledger row. A rule that fired on a
    `scan` with no assume in it would caveat most of the corpus."""
    def plain_scan():
        x = any_array((), "float64", (-10.0, 10.0))

        def body(c, _):
            return c + 1.0, 0.0

        lax.scan(body, x, jnp.zeros((2,)))
        return assert_(x <= 10.0)

    p = propagate(trace(plain_scan))
    assert p.assume_ledger == ()
    assert p.assume_dropped is False
    assert not any("NEVER CLASSIFIED" in n for n in p.notes)
    assert UNDESCENDED_ASSUME_ASSUMPTION not in p.assumptions
