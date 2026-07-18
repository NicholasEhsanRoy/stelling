# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The harness primitives land in the trace, and the hash covers them.

Skipped wholesale when jax is absent. x64 is enabled for this module and
restored afterwards (the harness contract puts precision in the stamp, and
these tests trace f64 declarations).
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling import ir  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.verdict import make_verdict  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def exp_harness(bound):
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(jnp.exp(x) < bound)

    return h


def primitives(cj: ir.ClosedJaxpr):
    out = []
    stack = [cj.jaxpr]
    while stack:
        j = stack.pop()
        for eqn in j.eqns:
            out.append(eqn.primitive)
    return out


def test_declarations_land_in_the_traced_query():
    cj = trace(exp_harness(8.0))
    prims = primitives(cj)
    assert "stelling_any" in prims and "stelling_assert" in prims
    any_eqn = next(e for e in cj.jaxpr.eqns if e.primitive == "stelling_any")
    params = any_eqn.params_dict()
    assert (params["lo"], params["hi"]) == (1.0, 2.0)


def test_content_hash_covers_the_bounds():
    def h_with(bounds):
        def h():
            x = any_array((), "float64", bounds)
            return assert_(jnp.exp(x) < 8.0)

        return h

    assert trace(h_with((1.0, 2.0))).content_hash() != trace(
        h_with((1.0, 3.0))
    ).content_hash()
    assert trace(h_with((1.0, 2.0))).content_hash() == trace(
        h_with((1.0, 2.0))
    ).content_hash()


def test_assume_lands_and_constrains_by_default():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        assume(x > 1.5)  # constrainable: x narrows to [1.5, 2.0]
        return assert_(jnp.exp(x) < 8.0)

    cj = trace(h)
    assert "stelling_assume" in primitives(cj)
    p = propagate(cj)
    assert p.all_discharged
    assert p.coverage.constrained == 1 and p.coverage.inert == 0
    assert "CONSTRAINED" in p.coverage.summary()
    v = make_verdict(
        cj,
        p,
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "VERIFIED"
    assert any(
        "the verdict holds where the precondition holds" in a
        for a in v.stamp.assumptions
    )
    assert "CONSTRAINED" in v.render()


def test_assume_inert_mode_reproduces_the_dropped_disclosure():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        assume(x > 1.5)  # inert mode: recorded, dropped, and SAID to be dropped
        return assert_(jnp.exp(x) < 8.0)

    cj = trace(h)
    p = propagate(cj, assume_mode="inert")
    assert p.all_discharged  # VERIFIED-with-drops stands: superset proved
    assert p.coverage.inert == 1 and "DROPPED" in p.coverage.summary()
    assert p.coverage.constrained == 0
    v = make_verdict(
        cj,
        p,
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "VERIFIED"
    assert any("DROPPED" in n for n in v.notes) and "DROPPED" in v.render()


def test_strict_boundary_assume_raises_instead_of_refuting():
    # audit F1 (UNSOUND), the auditor's exact construction: the true
    # assumed region of x > 1 over [0, 1] is EMPTY; before the fix the
    # closed stand-in [1, 1] judged `x <= 0.5` definitely false and
    # minted REFUTED — a false refutation of a vacuously-true conditional
    # claim. The emptiness is decidable at the assume site and refused.
    from stelling.propagate import UnsatisfiableAssumptionError

    def h():
        x = any_array((), "float64", (0.0, 1.0))
        assume(x > 1.0)
        return assert_(x <= 0.5)

    cj = trace(h)
    with pytest.raises(UnsatisfiableAssumptionError) as exc:
        propagate(cj)
    assert "strict" in str(exc.value)
    # inert control: a normal verdict exists, obligation undecided
    assert propagate(cj, assume_mode="inert").obligations[0].status == "unknown"


def test_branch_scoped_unsatisfiable_assume_gets_a_verdict_with_the_note():
    # audit F2 (FRAGILE), the auditor's exact construction: a
    # jax.lax.cond branch guards an assume that is empty over the box;
    # before the fix the whole analysis died with a whole-domain
    # misdiagnosis. Branch-scoped unsatisfiability now degrades with the
    # branch-vacuity note and the query gets its verdict.
    from jax import lax

    def h():
        x = any_array((), "float64", (0.0, 1.0))

        def ta(x):
            assume(x >= 5.0)  # empty over [0,1] — this branch's precondition
            return x + 1.0

        def fa(x):
            return x - 1.0

        r = lax.cond(x < 0.5, ta, fa, x)
        return assert_(r < 10.0)

    cj = trace(h)
    p = propagate(cj)  # must not raise
    assert p.obligations[0].status == "discharged"
    note = next(
        n for n in p.notes if "assume unsatisfiable within this cond branch" in n
    )
    assert "may be vacuous under the branch's precondition" in note
    assert p.coverage.constrained == 0  # no narrowing was applied
    # inert control agrees on the obligation status
    assert propagate(cj, assume_mode="inert").obligations[0].status == "discharged"


def test_image_gap_assume_never_refutes_the_theorem():
    # audit F7 (UNSOUND), the auditor's traced channel A: x ∈ [-1,1],
    # w = x·x has true image [0,1] but a correlation-blind box ⊇ [-1,1];
    # assume(w <= -0.5) is unsatisfiable in ℝ yet meets the box, and
    # before the fix the THEOREM w >= 0 came back REFUTED. The target is
    # a transfer output — precondition satisfiability uncertified — so
    # the definite violation is withheld: UNKNOWN, never REFUTED.
    def h():
        x = any_array((), "float64", (-1.0, 1.0))
        w = x * x
        assume(w <= -0.5)
        return assert_(w >= 0.0)

    cj = trace(h)
    p = propagate(cj)  # no raise; the narrowing is sound and applied
    assert p.obligations[0].status == "unknown"
    assert any("violation WITHHELD from REFUTED" in n for n in p.notes)
    v = make_verdict(
        cj,
        p,
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "UNKNOWN"
    assert any(
        "precondition satisfiability uncertified" in a
        for a in v.stamp.assumptions
    )


def test_assume_narrowing_flips_unknown_to_verified_and_inert_mode_measures_it():
    # the vacuity instrument's two arms on one harness: exp(x) > 4.9 is
    # undecidable over x ∈ [1, 2] but definite over the assumed [1.6, 2]
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        assume(x >= 1.6)
        return assert_(jnp.exp(x) > 4.9)  # exp(1.6) ≈ 4.953

    cj = trace(h)
    p_constrain = propagate(cj)
    assert p_constrain.obligations[0].status == "discharged"
    p_inert = propagate(cj, assume_mode="inert")
    assert p_inert.obligations[0].status == "unknown"


def test_end_to_end_verified():
    cj = trace(exp_harness(8.0))
    v = make_verdict(
        cj,
        propagate(cj),
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "VERIFIED"


def test_positive_control_mutation_comes_back_red():
    """§10.8: a harness that cannot fail proves nothing — this one must."""
    cj = trace(exp_harness(2.0))  # exp([1,2]) ⊆ [2.71, 7.39]: never < 2
    v = make_verdict(
        cj,
        propagate(cj),
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "REFUTED"  # red is a fact, with a verdict of its own
    assert v.obligations[0].status == "violated-over-set"


def test_eager_any_array_refuses():
    with pytest.raises(RuntimeError):
        any_array((), "float64", (0.0, 1.0))


def test_f32_roundtrip_does_not_false_verify():
    """Audit finding 1, the real-trace construction: float32(0.1) rounds UP,
    so `roundtrip(x) <= 0.1` is false for the only declared value — the old
    pass-through discharged it."""

    def h():
        x = any_array((), "float64", (0.1, 0.1))
        y = x.astype(jnp.float32)
        return assert_(y.astype(jnp.float64) <= 0.1)

    p = propagate(trace(h))
    assert p.obligations[0].status == "unknown"  # ⊤, never a false VERIFIED


def test_empty_declared_set_refused_at_declaration():
    import math

    for bounds in ((5.0, 3.0), (math.nan, 1.0), (0.0, math.nan)):
        with pytest.raises(ValueError):
            any_array((), "float64", bounds)


def test_infinite_point_bounds_refused_as_empty_real_set():
    # (inf, inf) slipped the lo<=hi check and yielded vacuous definite
    # verdicts over an empty ℝ-set (audit-gate finding 3)
    import math

    for bounds in ((math.inf, math.inf), (-math.inf, -math.inf)):
        with pytest.raises(ValueError):
            any_array((), "float64", bounds)

    def h():  # half-infinite and full-⊤ declarations stay legal
        a = any_array((), "float64", (-math.inf, math.inf))
        b = any_array((), "float64", (0.0, math.inf))
        return assert_(b >= 0.0), a

    assert propagate(trace(h)).obligations[0].status == "discharged"


def test_nan_sentinel_where_survives_end_to_end():
    # the real-trace flavor of audit-gate finding 1
    def h():
        x = any_array((), "float64", (0.0, 1.0))
        y = jnp.where(x > 0.5, x, jnp.nan)
        return assert_(y <= 1.0)

    p = propagate(trace(h))  # must not raise
    assert p.obligations[0].status == "unknown"


def test_nonvacuity_checked_and_failed_paths():
    from stelling.harness import nonvacuity

    def h(y0):
        def inner():
            x = any_array((), "float64", (1.0, 2.0))
            p = any_array((), "float64", (y0, y0))  # the "incident" point
            m = nonvacuity(p > 1.0)
            m2 = nonvacuity(p < 2.0)
            return assert_(jnp.exp(x) < 8.0), m, m2

        return inner

    v_in = make_verdict(
        trace(h(1.5)),
        propagate(trace(h(1.5))),
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v_in.status == "VERIFIED"
    assert v_in.stamp.nonvacuity.startswith("checked")
    assert not any("vacuous" in n for n in v_in.notes)

    v_out = make_verdict(
        trace(h(9.0)),  # the point is outside the declared box
        propagate(trace(h(9.0))),
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v_out.status == "VERIFIED"  # the box claim itself still holds
    assert v_out.stamp.nonvacuity.startswith("FAILED")
    assert any("vacuous" in n for n in v_out.notes)
