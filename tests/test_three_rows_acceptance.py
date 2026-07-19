# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The three-row round's acceptance family — needs jax to trace.

Separate from ``test_three_rows.py`` deliberately: that file is jax-free
and must run in the zero-dep environment, and a module-level
``importorskip`` would take the whole of it out of that run. One file, one
dependency level.

What is recorded here: the acceptance family at both r_lo values, the
agreement check against the algebraically equivalent hand-simplified
form, the measured-jax cross-checks (L10 — model the measured target),
and — **pinned as a finding rather than worked around** — the emission
boundary the acceptance family runs into. Skipped per availability: jax
to trace; a solver for the escalated cases.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from stelling import interval as iv
from stelling.obligation import DeclinedObligation, slice_unknown_obligations
from stelling.propagate import interval_env, propagate

INF = math.inf

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling import _optional  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.solvers import (  # noqa: E402
    SolverConfig,
    escalate,
    make_solver_verdict,
)

HAVE_Z3 = _optional.available("z3")
HAVE_CVC5 = _optional.available("cvc5") or _optional.cvc5_binary() is not None
need_solver = pytest.mark.skipif(
    not (HAVE_Z3 or HAVE_CVC5), reason="needs a solver"
)

VERSIONS = dict(
    stelling_version="test",
    jax_version=jax.__version__,
    precision_config="jax_enable_x64=True",
)
CONFIG = SolverConfig(timeout_ms=30_000)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def acceptance_harness(r_lo):
    """The acceptance family, verbatim from the round's specification."""

    def h():
        p = any_array((), "float64", (0.5, 2.0))
        q = any_array((), "float64", (0.5, 2.0))
        r = any_array((), "float64", (r_lo, 1.0))
        u = jnp.array([p, 0.0 * p, 0.0 * p])
        v = jnp.array([q * r, 0.0 * q, 0.0 * q])
        s = jnp.sum(u * v)  # reduce_sum
        return (assert_(p**2 / s <= 8.0),)  # integer_pow, div

    return h


def simplified_harness(r_lo):
    """The algebraically equivalent hand-simplified form: p/(q*r) <= 8."""

    def h():
        p = any_array((), "float64", (0.5, 2.0))
        q = any_array((), "float64", (0.5, 2.0))
        r = any_array((), "float64", (r_lo, 1.0))
        return (assert_(p / (q * r) <= 8.0),)

    return h


def test_measured_jax_integer_pow_values_lie_inside_the_transfer():
    """L10, applied: the transfer is checked against the MEASURED target,
    not against the standard it claims."""
    import jax.lax as lax

    for base in (0.0, -0.0, 2.0, -2.0, 0.5, -0.5, INF, -INF):
        for y in (0, 1, 2, 3, 4, -1, -2):
            got = float(
                jax.jit(lambda x, y=y: lax.integer_pow(x, y))(jnp.float64(base))
            )
            a = iv.IntervalArray(shape=(), los=(base,), his=(base,))
            r = iv.integer_pow(a, y)
            if math.isnan(got):
                continue
            assert r.los[0] <= got <= r.his[0], (base, y, got, r.los, r.his)


def test_measured_jax_empty_sum_is_zero_like_the_transfer():
    got = float(jax.jit(lambda x: jnp.sum(x))(jnp.zeros(0, dtype="float64")))
    empty = iv.IntervalArray(shape=(0,), los=(), his=())
    r = iv.reduce_sum(empty, (0,))
    assert got == 0.0 == r.los[0] == r.his[0]


def test_acceptance_interval_only_is_unknown_at_full_coverage():
    """Row 1 + row 2 together take this query to 100% coverage — no ⊤ —
    and it is STILL unknown: the expression's self-correlation is not
    interval-resolvable, which is expected and correct."""
    for r_lo in (0.71, 0.11):
        cj = trace(acceptance_harness(r_lo))
        p = propagate(cj)
        assert [o.status for o in p.obligations] == ["unknown"]
        assert p.coverage.unknown == 0, p.coverage.summary()
        assert p.coverage.fraction_known == 1.0, p.coverage.summary()
        assert dict(p.transfers_used)["reduce_sum"] == "sound"
        assert dict(p.transfers_used)["integer_pow"] == "sound"


def test_acceptance_family_escalation_is_blocked_by_the_emission_boundary():
    """**A pinned coverage finding, not a pass.**

    The three rows take the acceptance family to full interval coverage
    and unblock its division guard, but they do not make it escalatable:
    its obligation slice crosses genuine 3-element arrays, and the v1 SMT
    emission is scalar-only. Escalation therefore declines and both
    verdicts stay UNKNOWN — where the round's specification expected
    VERIFIED at r_lo=0.71 and REFUTED at r_lo=0.11.

    Closing it needs a FOURTH row (`concatenate`) plus array-element term
    tracking and array-shaped elementwise arithmetic. The list is closed
    at three, so this is reported rather than built, and pinned here so
    the day it changes is a conscious act.
    """
    for r_lo in (0.71, 0.11):
        cj = trace(acceptance_harness(r_lo))
        p = propagate(cj)
        items = slice_unknown_obligations(cj, p, interval_env(cj))
        assert len(items) == 1
        assert isinstance(items[0], DeclinedObligation)
        assert "scalar-only" in items[0].reason


def test_acceptance_family_division_guard_is_unblocked_by_row_one():
    """What row 1's interval transfer DID buy the acceptance family: the
    divisor `s` is no longer ⊤, so it definitely excludes zero and the
    div guard passes. Before this row the slice died on the divisor."""
    from stelling.obligation import _atom_excludes_zero

    cj = trace(acceptance_harness(0.71))
    env = interval_env(cj)
    (div_eqn,) = [e for e in cj.jaxpr.eqns if e.primitive == "div"]
    assert _atom_excludes_zero(div_eqn.invars[1], env)
    s_box = env[div_eqn.invars[1].id]
    assert s_box.los[0] > 0.0, (s_box.los, s_box.his)


@need_solver
def test_agreement_the_simplified_form_verifies_on_the_safe_region():
    """The mandated agreement check. The unsimplified form is not
    evaluable (see the pinned finding above), so this records the half
    that IS evaluable: the hand-simplified form must VERIFY at r_lo=0.71.
    Nothing disagrees — one side simply cannot be asked."""
    cj = trace(simplified_harness(0.71))
    p = propagate(cj)
    assert p.obligations[0].status == "discharged"
    v = make_solver_verdict(cj, p, escalate(cj, p, CONFIG), **VERSIONS)
    assert v.status == "VERIFIED"


@need_solver
def test_agreement_the_simplified_form_refutes_with_a_low_r_witness():
    """r_lo=0.11 must REFUTE with a witness whose r sits near the low end
    of [0.11, 1.0] — the only region where p/(q*r) can exceed 8."""
    cj = trace(simplified_harness(0.11))
    p = propagate(cj)
    assert p.obligations[0].status == "unknown"
    esc = escalate(cj, p, CONFIG)
    (record,) = esc.records
    assert record.outcome == "violated-witness"
    values = dict(record.witness.values)
    pv, qv, rv = (Fraction(values[k]) for k in ("x0", "x1", "x2"))
    assert Fraction(11, 100) <= rv <= 1  # in the declared box
    assert pv / (qv * rv) > 8  # genuinely violating, recomputed here
    assert rv < Fraction(1, 2), f"witness r={float(rv)} is not near the low end"
    v = make_solver_verdict(cj, p, esc, **VERSIONS)
    assert v.status == "REFUTED"


def test_three_rows_do_not_disturb_the_simplified_forms_verdicts():
    """The rows must reach an answer, never CHANGE one: the simplified
    family touches none of them and must behave exactly as before."""
    assert propagate(trace(simplified_harness(0.71))).obligations[0].status == (
        "discharged"
    )
    assert propagate(trace(simplified_harness(0.11))).obligations[0].status == (
        "unknown"
    )


# =============================================================================
# Audit fix round, end to end through a real jax trace: the auditor's own
# UNSOUND 1 / UNSOUND 2 constructions, which is where these bit.
# =============================================================================


def wrapping_int(x):
    """int32 value in {50000, 150000}; its square WRAPS (measured jax)."""
    return (x > 0.5).astype(jnp.int32) * 100000 + 50000


@pytest.mark.parametrize("expr,label", [
    (lambda v: v * v, "mul"),
    (lambda v: v**2, "integer_pow"),
])
def test_int32_wraparound_never_mints_a_definite_verdict(expr, label):
    """AUDIT UNSOUND 1 (mul, no solver) and UNSOUND 2 (integer_pow, through
    the solver). jax MEASURES ((x>0.5)*100000 + 50000)**2 as -1794967296 at
    x = 0.4, so `> 0` is FALSE over the declared box. Both routes used to
    return VERIFIED — one from interval propagation alone, one from an
    unguarded emission handing the wrapping power to a solver as an exact
    Real power."""
    measured = {
        xv: int(jax.jit(lambda t: expr(wrapping_int(t)))(jnp.float64(xv)))
        for xv in (0.0, 0.4, 0.6, 1.0)
    }
    assert any(m < 0 for m in measured.values()), measured  # jax really wraps

    def h():
        x = any_array((), "float64", (0.0, 1.0))
        return (assert_(expr(wrapping_int(x)) > 0),)

    cj = trace(h)
    p = propagate(cj)
    assert [o.status for o in p.obligations] == ["unknown"]
    assert any("wraparound is not excluded" in n for n in p.notes), p.notes

    # and the escalation path must not rescue it either
    items = slice_unknown_obligations(cj, p, interval_env(cj))
    assert all(isinstance(i, DeclinedObligation) for i in items)
    assert all("wraps on overflow" in i.reason for i in items)

    if HAVE_Z3 or HAVE_CVC5:
        v = make_solver_verdict(cj, p, escalate(cj, p, CONFIG), **VERSIONS)
        assert v.status == "UNKNOWN", f"{label} minted {v.status}"


def test_small_integer_arithmetic_still_reaches_a_definite_verdict():
    """The reachability guard must not have cost the common case: an index
    computation that cannot overflow keeps its exact result."""

    def h():
        x = any_array((), "float64", (0.0, 1.0))
        i = (x > 0.5).astype(jnp.int32) * 3 + 1  # in {1, 4}
        return (assert_(i * i > 0),)

    p = propagate(trace(h))
    assert [o.status for o in p.obligations] == ["discharged"]
    assert not any("wraparound" in n for n in p.notes)


def test_legal_jax_slice_spellings_are_unaffected_by_the_bounds_check():
    """AUDIT FRAGILE 1's control: jax clamps its own slice params, so no
    legal spelling may start declining."""
    import jax.lax as lax

    v = jnp.arange(4.0)
    for name, fn in (
        ("v[1]", lambda a: a[1]), ("v[0:2]", lambda a: a[0:2]),
        ("v[::2]", lambda a: a[::2]), ("v[-1]", lambda a: a[-1]),
        ("v[5:9]", lambda a: a[5:9]), ("v[2:99]", lambda a: a[2:99]),
        ("lax.slice", lambda a: lax.slice(a, (1,), (3,), (1,))),
    ):
        def h(fn=fn):
            def hh():
                x = any_array((), "float64", (0.0, 1.0))
                arr = jnp.arange(4.0) * x
                sub = fn(arr)
                return (assert_(jnp.sum(sub) >= -1e9),)
            return hh

        p = propagate(trace(h()))
        assert not any("outside the operand's extent" in n for n in p.notes), name


def test_a_legal_jax_trace_can_set_the_exponent_arbitrarily():
    """AUDIT FRAGILE 2's premise, measured rather than assumed: `x ** y`
    for a huge y is ONE legal equation, so the cap has real work to do."""
    import time

    def h():
        x = any_array((), "float64", (0.1, 0.2))
        return (assert_(x**10_000_000 <= 1.0),)

    cj = trace(h)
    (pe,) = [e for e in cj.jaxpr.eqns if e.primitive == "integer_pow"]
    assert pe.params_dict()["y"] == 10_000_000  # jax bounds it nowhere
    t0 = time.perf_counter()
    p = propagate(cj)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"propagate took {elapsed:.1f}s on one equation"
    assert p.obligations[0].status == "unknown"
    assert any("exact-rational endpoint cap" in n for n in p.notes)


# =============================================================================
# Audit re-attack round, end to end through jax: integer div and FMA
# contraction, which is where these were found.
# =============================================================================


@pytest.mark.parametrize("a,b,cmp,bound", [
    (-7, 2, "lt", -3),   # measured lax.div(-7,2) = -3, so `< -3` is FALSE
    (-7, 2, "ge", -3),   # ...and `>= -3` is TRUE
    (7, 2, "gt", 3),
    (7, 2, "le", 3),
])
def test_integer_div_verdicts_match_measured_jax(a, b, cmp, bound):
    """AUDIT UNSOUND 3, end to end. jax integer division TRUNCATES toward
    zero; `iv.div` computed real division, so every predicate separating -3
    from -3.5 was decided wrongly — six false definite verdicts, three
    VERIFIED and three REFUTED. Each verdict is now checked against
    measured jax rather than against real arithmetic."""
    import operator

    import jax.lax as lax

    op = {"lt": operator.lt, "ge": operator.ge,
          "gt": operator.gt, "le": operator.le}[cmp]
    measured = bool(
        op(int(jax.jit(lambda: lax.div(jnp.int32(a), jnp.int32(b)))()), bound)
    )

    def h():
        x = any_array((), "float64", (0.0, 1.0))
        # a genuine trace: the int32 operands ride in on the whitelisted
        # bool->int32 conversion, exactly as the auditor's route did
        one = (x >= 0.0).astype(jnp.int32)
        num, den = one * a, one * b
        return (assert_(op(lax.div(num, den), bound)),)

    p = propagate(trace(h))
    status = p.obligations[0].status
    assert status in ("discharged", "violated-over-set", "unknown")
    if status == "discharged":
        assert measured is True, f"VERIFIED but jax measures {measured}"
    if status == "violated-over-set":
        assert measured is False, f"REFUTED but jax measures {measured}"


def test_int_min_over_minus_one_wraps_and_is_never_discharged():
    """The wraparound class the guard exists for, reached through `div`."""
    measured = int(
        jax.jit(lambda: jax.lax.div(jnp.int32(-(2**31)), jnp.int32(-1)))()
    )
    assert measured == -(2**31)  # WRAPS; the real answer is +2**31

    def h():
        x = any_array((), "float64", (0.0, 1.0))
        one = (x >= 0.0).astype(jnp.int32)
        return (assert_(jax.lax.div(one * (-(2**31)), one * -1) > 0),)

    p = propagate(trace(h))
    assert p.obligations[0].status == "unknown"


def test_ieee_contraction_is_measured_on_this_target_not_assumed():
    """AUDIT UNSOUND 4's premise, measured: XLA fuses multiply+add into one
    FMA, so (a*b)-1 differs between eager and jit. Four definite ieee
    verdicts contradicted the compiled program."""
    a = 1.0 + 2.0**-27
    b = 1.0 - 2.0**-27
    eager = float((jnp.float64(a) * jnp.float64(b)) - 1.0)
    jitted = float(jax.jit(lambda u, w: (u * w) - 1.0)(
        jnp.float64(a), jnp.float64(b)))
    assert eager == 0.0
    assert jitted == -(2.0**-54), jitted  # the contracted, single-rounding answer

    def h():
        u = any_array((), "float64", (a, a))
        w = any_array((), "float64", (b, b))
        return (assert_((u * w) - 1.0 == 0.0),)

    cj = trace(h)
    # ieee must not claim what the compiled program contradicts
    assert propagate(cj, semantics="ieee").obligations[0].status == "unknown"
    # real mode was already unknown here and is untouched: the exact product
    # 1-2**-54 needs 54 bits, so the outward-rounded bracket straddles 1.0
    # and `== 0.0` was never decidable in ℝ either
    assert propagate(cj).obligations[0].status == "unknown"


# =============================================================================
# Third re-attack, end to end: the ten forms XLA still contracts after its
# own simplification passes.
# =============================================================================

_A = 1.0 + 2.0**-27
_B = 1.0 - 2.0**-27


@pytest.mark.parametrize("name,prog", [
    # each lands the result on ~0, where the two roundings differ; the
    # `neg` forms ADD one because the sign has been flipped
    ("neg", lambda u, w: (-(u * w)) + 1.0),
    ("neg neg", lambda u, w: (-(-(u * w))) - 1.0),
    ("reshape", lambda u, w: jnp.reshape(u * w, ()) - 1.0),
    ("broadcast+squeeze",
     lambda u, w: jnp.squeeze(jnp.broadcast_to(u * w, (1,))) - 1.0),
    ("broadcast+slice+squeeze",
     lambda u, w: jnp.squeeze(jnp.broadcast_to(u * w, (1,))[0:1]) - 1.0),
    ("broadcast+reduce_sum",
     lambda u, w: jnp.sum(jnp.broadcast_to(u * w, (1,))) - 1.0),
    ("max", lambda u, w: jnp.maximum(u * w, -jnp.inf) - 1.0),
    ("stop_gradient", lambda u, w: jax.lax.stop_gradient(u * w) - 1.0),
    ("nested jit", lambda u, w: jax.jit(lambda v: v)(u * w) - 1.0),
])
def test_ten_contraction_forms_are_indeterminate_under_ieee(name, prog):
    """AUDIT UNSOUND 5, end to end. Each of these keeps XLA's contraction
    alive (jit != eager) while breaking a syntactic `is my producer a mul?`
    match. Measured first, then checked."""
    eager = float(prog(jnp.float64(_A), jnp.float64(_B)))
    jitted = float(jax.jit(prog)(jnp.float64(_A), jnp.float64(_B)))
    if eager == jitted:
        pytest.skip(f"{name}: XLA did not contract this form on this build")

    def h():
        u = any_array((), "float64", (_A, _A))
        w = any_array((), "float64", (_B, _B))
        return (assert_(prog(u, w) == 0.0),)

    p = propagate(trace(h), semantics="ieee")
    assert p.obligations[0].status == "unknown", (
        f"{name}: ieee claims {p.obligations[0].status} while jit computes "
        f"{jitted} and eager computes {eager}"
    )


def test_agreeing_products_still_discharge_end_to_end():
    """The precision cost is bounded and the mode stays useful: where the
    two roundings agree, an obligation over a product still discharges."""

    def h():
        u = any_array((), "float64", (2.0, 2.0))
        w = any_array((), "float64", (3.0, 3.0))
        return (assert_((u * w) + 1.0 == 7.0),)

    assert propagate(trace(h), semantics="ieee").obligations[0].status == (
        "discharged"
    )


def test_mul_free_ieee_obligations_keep_full_decidability():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return (assert_((x + 1.0) - 0.5 <= 3.0),)

    assert propagate(trace(h), semantics="ieee").obligations[0].status == (
        "discharged"
    )
