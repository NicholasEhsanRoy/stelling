# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The `square` row's acceptance, on real external code.

The target is JAX-Fluids' own Jiang-Shu smoothness indicators —
``WENO5Base.smoothness``, six ``jnp.square`` calls in three lines, called
verbatim from the installed package, not reproduced here. The row's
acceptance is not a case this project constructed: it is a property over
that function which is **genuinely false inside a declared box**, which
before this row came back as a decline (``primitive 'square' is outside
the supported emission set``), and whose witness can be executed back
through the library to reproduce the violation.

**The property.** WENO5's optimal weights are ``d = (0.1, 0.6, 0.3)``:
the CENTRAL sub-stencil carries the largest one, and the adaptivity works
by shrinking a sub-stencil's weight as its smoothness indicator grows. The
reasoning that goes with that — "the central sub-stencil is the smoothest,
so its weight dominates" — is a statement about beta, and it is stateable
as a box invariant: ``beta_1 <= max(beta_0, beta_2)`` over the declared
envelope of cell values. It is FALSE, and stelling returns the five-point
stencil at which it fails, in the box, with the two neighbours' indicators
both strictly below the central one. On such a stencil ``omega_1`` is the
SMALLEST of the three weights and the reconstruction leans on the
one-sided stencils — the opposite of what the linear weights suggest.

**Negative control**, in the same file and against the same function:
``beta_1 <= 18.0`` over the same box. Interval propagation does not
discharge it either, so it too runs the whole pipeline — and comes back
VERIFIED by ``unsat`` from both backends. If the row minted witnesses for
true properties, this is where it would show.

The control was ``beta_0 >= 0`` until audit 0.2.0 M16 gave ``mul`` an
exact-rational path. The indicators are sums of squares with positive
rational coefficients, and with those coefficient multiplications exact
their box now floors at exactly 0, so interval propagation discharges that
one before any solver sees it — which makes it useless as a control on
this row and worth keeping as a control on the cheap layer instead. The
escalating control moved to a bound the interval leg cannot reach:
``beta_1``'s propagated box tops out at 18.333…, its true maximum over the
declared box is 17.333… (beta is a convex quadratic form in the five cell
values, so its maximum over a box is attained at a vertex; enumerated over
all 32 vertices), and 18.0 sits between the two.

Skipped without jax, jaxfluids, or a solver. The target's source is
checked for the ``jnp.square`` call before anything is measured: if
upstream rewrites it, this file must SAY so rather than quietly measure a
different program.
"""

from __future__ import annotations

import inspect
from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("jaxfluids")
pytest.importorskip("z3")
pytest.importorskip("cvc5")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jaxfluids.stencils.reconstruction.shock_capturing.weno5_base import (  # noqa: E402
    WENO5Base,
)

import stelling.obligation as OB  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import interval_env, propagate  # noqa: E402

TIMEOUT_MS = 60_000

# The declared envelope: a five-point stencil of cell values within one
# unit of each other. Non-degenerate, and small enough that the QF_NRA
# problem is decided in well under the budget.
BOX = (-1.0, 1.0)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def smoothness(*u):
    """JAX-Fluids' own indicator, called unbound.

    ``WENO5Base.smoothness`` never touches ``self`` (the coefficients are
    literals in its body) and ``WENO5Base`` is abstract, so ``self=None``
    calls the shipped code with nothing standing in for it. Nothing is
    reimplemented here — see :func:`test_the_target_is_the_library_function`,
    which reads the source and refuses if the call this row exists for is
    no longer in it."""
    return WENO5Base.smoothness(None, *u)


def _stencil_harness(predicate):
    def h():
        u = [any_array((), "float64", BOX) for _ in range(5)]
        return (assert_(predicate(*smoothness(*u))),)

    return h


def _central_is_smoothest(b0, b1, b2):
    return b1 <= jnp.maximum(b0, b2)


def _beta0_is_nonnegative(b0, b1, b2):
    return b0 >= 0.0


# The true maximum of beta_1 over the declared box, measured by vertex
# enumeration in :func:`test_the_escalating_controls_bound_is_between_the
# _true_maximum_and_the_box`; 18.0 sits strictly above it and strictly
# below the propagated box's upper endpoint, which is what makes the
# property true AND interval-undecidable.
_BETA1_BOUND = 18.0


def _beta1_is_bounded(b0, b1, b2):
    return b1 <= _BETA1_BOUND


def _witness_point(verdict):
    """The witness as five floats in declaration order (u_imm .. u_ipp) —
    x0..x4 are the declarations in the order the harness made them."""
    (w,) = verdict.witnesses
    values = dict(w.values)
    assert sorted(values) == [f"x{k}" for k in range(5)], values
    return [float(Fraction(values[f"x{k}"])) for k in range(5)]


# --- the target is what we say it is -----------------------------------------


def test_the_target_is_the_library_function_and_still_uses_jnp_square():
    """The fence. This acceptance is about a `square` in someone else's
    code; if that call goes away, the file must fail loudly rather than
    keep passing against something else."""
    src = inspect.getsource(WENO5Base.smoothness)
    assert src.count("jnp.square(") == 6, src
    path = inspect.getsourcefile(WENO5Base)
    assert path is not None and path.endswith(
        "stencils/reconstruction/shock_capturing/weno5_base.py"
    ), path
    assert "jaxfluids" in path


def test_the_traced_query_really_contains_the_square_primitive():
    """`jnp.square` binds jax's own `square_p`, so the obligation slice
    genuinely traverses the primitive this row is for — not `integer_pow`,
    and not a multiplication."""
    closed = trace(_stencil_harness(_central_is_smoothest))
    prims = [str(e.primitive) for e in closed.jaxpr.eqns]
    assert prims.count("square") == 6, prims
    assert "integer_pow" not in prims and "mul" in prims


def test_the_cheap_layers_do_not_decide_it_so_the_row_is_what_decides():
    """Both obligations must survive interval propagation, or the verdicts
    below are about the interval leg rather than about this row."""
    for predicate in (_central_is_smoothest, _beta1_is_bounded):
        p = propagate(trace(_stencil_harness(predicate)))
        assert p.obligations[0].status == "unknown", predicate.__name__
        # and it is NOT a coverage gap: every equation ran a real transfer
        assert p.coverage.unknown == 0


def test_the_cheap_layer_now_decides_beta0_and_that_is_why_it_is_not_the_control():
    """`beta_0 >= 0` was the escalating control until audit 0.2.0 M16.

    The indicator is `13/12*(a)^2 + 1/4*(b)^2` in jaxfluids' own code — six
    `jnp.square` calls and the coefficient `mul`s. With `mul` bumping every
    endpoint outward, the sum's lower endpoint went slightly NEGATIVE and
    `beta_0 >= 0` straddled; with `mul` exact on representable corners the
    floor is exactly 0 and the interval leg says so. Pinned, because a
    reader finding the old control in the history should find the reason it
    moved here rather than infer that the row got weaker."""
    p = propagate(trace(_stencil_harness(_beta0_is_nonnegative)))
    assert p.obligations[0].status == "discharged"
    assert p.coverage.unknown == 0


def test_the_escalating_controls_bound_is_between_the_true_maximum_and_the_box():
    """The control's bound must be TRUE over the box and UNREACHABLE by the
    interval leg, and both halves are measured here rather than asserted.

    beta_1 is a positive-semidefinite quadratic form in the five cell
    values, so it is convex and its maximum over the box `[-1, 1]^5` is
    attained at a vertex — all 32 are enumerated through jaxfluids' own
    function. The propagated box's upper endpoint is read from the live
    interval environment."""
    import itertools

    true_max = max(
        float(np.asarray(smoothness(*[jnp.asarray(v, jnp.float64) for v in pt])[1]))
        for pt in itertools.product(BOX, repeat=5)
    )
    closed = trace(_stencil_harness(_beta1_is_bounded))
    p = propagate(closed)
    detail = p.obligations[0].detail
    assert true_max < _BETA1_BOUND, true_max
    # the interval leg cannot decide it: its box for beta_1 reaches past the
    # bound, which is exactly what the undecided detail quotes
    assert p.obligations[0].status == "unknown"
    assert str(_BETA1_BOUND) in detail


# --- the acceptance ----------------------------------------------------------


def test_the_obligation_reaches_the_solver_where_it_used_to_decline():
    """The exact gap this row closes, stated as the before/after it is:
    the slice is an ObligationSlice in QF_NRA, where it was a
    DeclinedObligation quoting `square`."""
    closed = trace(_stencil_harness(_central_is_smoothest))
    p = propagate(closed)
    (item,) = OB.slice_unknown_obligations(closed, p, interval_env(closed))
    assert not isinstance(item, OB.DeclinedObligation), item
    assert item.fragment == OB.QF_NRA
    assert item.element_terms <= OB.ELEMENT_BUDGET


def test_acceptance_refuted_with_a_witness_that_replays():
    v = check(
        _stencil_harness(_central_is_smoothest),
        vacuity_mode="inputs-only",
        solver_timeout_ms=TIMEOUT_MS,
    )
    assert v.status == "REFUTED", v.render()
    (w,) = v.witnesses
    assert "exact-rational replay" in w.replay
    # every value is inside the box it was declared on
    for _, text in w.values:
        assert BOX[0] <= Fraction(text) <= BOX[1], w.values
    # and the exact-rational replay agrees the predicate is false there
    closed = trace(_stencil_harness(_central_is_smoothest))
    p = propagate(closed)
    (sl,) = OB.slice_unknown_obligations(closed, p, interval_env(closed))
    rationals = {name: Fraction(text) for name, text in w.values}
    assert OB.witness_is_valid(sl, rationals) is None
    assert OB.evaluate_predicate(sl, rationals) is False


def test_the_witness_executes_through_jaxfluids_and_reproduces_the_violation():
    """The independent leg, and the one the acceptance turns on: the
    witness is fed to JAX-Fluids' OWN function and the violation comes back
    out. This shares no code with the emission or the replay — eagerly and
    under `jit`, because the row must hold in both."""
    v = check(
        _stencil_harness(_central_is_smoothest),
        vacuity_mode="inputs-only",
        solver_timeout_ms=TIMEOUT_MS,
    )
    assert v.status == "REFUTED", v.render()
    point = _witness_point(v)
    args = [jnp.asarray(x, jnp.float64) for x in point]

    eager = [float(np.asarray(b)) for b in smoothness(*args)]
    jitted = [float(np.asarray(b)) for b in jax.jit(smoothness)(*args)]
    assert eager == jitted, (eager, jitted)

    b0, b1, b2 = eager
    assert b1 > max(b0, b2), (
        f"the witness {point} did NOT reproduce the violation through "
        f"jaxfluids: beta = {eager}"
    )
    # a genuine kink, not a degenerate all-zero stencil
    assert b1 > 0.0 and max(b0, b2) >= 0.0


def test_negative_control_the_same_function_verifies_a_true_property():
    """`beta_1 <= 18.0` HOLDS over the same box (its true maximum there is
    17.333…) and the interval leg cannot see it — the pipeline says so by
    `unsat`, not by declining and not by minting a witness."""
    v = check(
        _stencil_harness(_beta1_is_bounded),
        vacuity_mode="inputs-only",
        solver_timeout_ms=TIMEOUT_MS,
    )
    assert v.status == "VERIFIED", v.render()
    assert v.witnesses == ()
    solver = v.stamp.solver
    assert isinstance(solver, tuple) and all(s.invoked for s in solver)
    assert any("unsat" in n for n in v.notes), v.notes


def test_the_verdict_stamp_names_the_square_transfer_and_the_portfolio():
    v = check(
        _stencil_harness(_central_is_smoothest),
        vacuity_mode="inputs-only",
        solver_timeout_ms=TIMEOUT_MS,
    )
    tiers = dict(v.stamp.transfer_tiers)
    assert tiers["square"] == "sound"
    assert dict(v.stamp.transfer_provenance)["square"] == "core"
    assert all(
        dict(s.options or ())["set-logic"] == "QF_NRA" for s in v.stamp.solver
    )


# --- a second, larger target: reached rather than decided --------------------


def test_the_full_weno5_js_reconstruction_now_REACHES_the_solver():
    """Recorded because it is the honest shape of the win on a big target.

    ``WENO5JS.reconstruct_xi`` — the whole reconstruction, smoothness
    indicators, nonlinear weights, normalisation and polynomials — declined
    on `square` before this row. It now builds a QF_NRA slice; the
    per-obligation element budget is what bounds it, and this test pins the
    REACH and the budget rather than a verdict, because the portfolio does
    not decide this one inside the acceptance timeout (measured: cvc5
    nl-cov timed out at 91 s on 487 terms). An obligation the solver could
    not decide is a different and better UNKNOWN than one that never got
    there, and it is the one this row produces."""
    from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno5_js import (
        WENO5JS,
    )

    stencil = WENO5JS(nh=3, inactive_axes=[1, 2])

    def h():
        # (7, 7, 7) is the smallest buffer the stencil's own slice objects
        # accept with nh=3: they trim 3 cells from each side of the two
        # transverse axes and 5 from the reconstruction axis.
        b = any_array((7, 7, 7), "float64", (0.1, 1.0))
        return (assert_(stencil.reconstruct_xi(b, 0, 0) > 0.0),)

    closed = trace(h)
    prims = [str(e.primitive) for e in closed.jaxpr.eqns]
    assert prims.count("square") == 6, prims
    p = propagate(closed)
    assert p.obligations[0].status == "unknown"
    (item,) = OB.slice_unknown_obligations(closed, p, interval_env(closed))
    assert not isinstance(item, OB.DeclinedObligation), item.reason
    assert item.fragment == OB.QF_NRA
    assert item.element_terms <= OB.ELEMENT_BUDGET


# --- one row, every spelling -------------------------------------------------


def test_the_three_spellings_of_a_square_agree_on_the_same_program():
    """`jnp.square(x)`, `x * x` and `x ** 2` are three spellings of one
    mathematics that bind three DIFFERENT primitives (`square`, `mul`,
    `integer_pow`). The other two could already reach the solver; a row
    that closed `square` without agreeing with them would have made the
    verdict depend on how the user wrote it."""
    spellings = {
        "square": lambda x: jnp.square(x),
        "mul": lambda x: x * x,
        "integer_pow": lambda x: x**2,
    }
    verdicts = {}
    for name, fn in spellings.items():

        def h(fn=fn):
            x = any_array((), "float64", (-2.0, 3.0))
            return (assert_(fn(x) - x <= 5.0),)

        prims = {str(e.primitive) for e in trace(h).jaxpr.eqns}
        assert name in prims, (name, prims)
        v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT_MS)
        verdicts[name] = (
            v.status,
            tuple(Fraction(t) for _, t in v.witnesses[0].values)
            if v.witnesses
            else (),
        )
    assert len({s for s, _ in verdicts.values()}) == 1, verdicts
    assert verdicts["square"][0] == "REFUTED", verdicts
    for name, (_, point) in verdicts.items():
        (x,) = point
        assert x * x - x > 5, (name, x)
