# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The constraining assume: the census is this test list — no jax.

Soundness direction first: a narrowing is applied only where it is
provably a SUPERSET of the true assumed region, so these tests pin the
binding rules from both sides — strict inequalities close (never
``nextafter``), nothing inverts through arithmetic, relational
comparisons stay inert, the meet is exact, an empty meet is a loud
harness defect, and ``assume_mode="inert"`` reproduces the MVP's
drop-everything behavior byte-identically.
"""

from __future__ import annotations

import math
import struct

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.propagate import (
    UnsatisfiableAssumptionError,
    interval_env,
    propagate,
)

INF = math.inf


def f64(shape=()):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype="float64")


def boolav(shape=()):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype="bool")


def var(i, aval=None):
    return ir.Var(id=i, aval=aval if aval is not None else f64())


def lit(v, aval=None):
    return ir.Literal(val=v, aval=aval if aval is not None else f64())


def arr_lit(vals):
    n = len(vals)
    return ir.Literal(
        val=ir.Array(dtype="<f8", shape=(n,), data=struct.pack(f"<{n}d", *vals)),
        aval=f64((n,)),
    )


def any_eqn(out, lo, hi, shape=()):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", shape), ("dtype", "float64"), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out, params=(), source_info=()):
    return ir.JaxprEqn(
        primitive=prim,
        invars=tuple(ins),
        outvars=(out,),
        params=tuple(params),
        source_info=tuple(source_info),
    )


def close(eqns, outvars, constvars=(), consts=()):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=tuple(constvars),
            invars=(),
            outvars=tuple(outvars),
            eqns=tuple(eqns),
        ),
        consts=tuple(consts),
    )


# --- the census: cmp × variable kind × shape form -----------------------------
#
# ge/gt/le/lt/eq × (declared input, intermediate) × (scalar,
# array-elementwise, scalar-bound-broadcast), literal point bound. The
# bound-side kinds beyond literals (const, point-declared input) and the
# flipped operand order get their own tests below.

CMPS = ("ge", "gt", "le", "lt", "eq")

# x is always declared over [0, 4]; the intermediate v = neg(x) lives on
# [-4, 0]. Bound values per form, chosen strictly inside so the meet
# narrows.
_X_BOX = (0.0, 4.0)
_KIND = {
    "input": dict(scalar=1.5, broadcast=2.5, elementwise=(1.0, 3.0)),
    "intermediate": dict(scalar=-1.5, broadcast=-2.5, elementwise=(-3.0, -1.0)),
}


def _v_box(varkind):
    lo, hi = _X_BOX
    return (-hi, -lo) if varkind == "intermediate" else (lo, hi)


def _expected(cmp, lo, hi, k):
    if cmp in ("ge", "gt"):  # strict closes to the SAME closed half-space
        return max(lo, k), hi
    if cmp in ("le", "lt"):
        return lo, min(hi, k)
    return k, k  # eq


def census_query(cmp, varkind, form):
    shape = () if form == "scalar" else (2,)
    spec = _KIND[varkind]
    x = var(0, f64(shape))
    eqns = [any_eqn(x, *_X_BOX, shape=shape)]
    if varkind == "intermediate":
        v = var(1, f64(shape))
        eqns.append(eqn("neg", [x], v))  # exact transfer: [-hi, -lo]
    else:
        v = x
    if form == "elementwise":
        bound = arr_lit(list(spec["elementwise"]))
        ks = spec["elementwise"]
    else:
        k = spec[form if form == "broadcast" else "scalar"]
        bound = lit(k)
        ks = (k,) * (1 if shape == () else 2)
    pred = var(2, boolav(shape))
    aout = var(3, boolav(shape))
    eqns.append(eqn(cmp, [v, bound], pred))
    eqns.append(eqn("stelling_assume", [pred], aout))
    lo, hi = _v_box(varkind)
    expect = [_expected(cmp, lo, hi, k) for k in ks]
    return close(eqns, [aout]), v.id, expect


@pytest.mark.parametrize("cmp", CMPS)
@pytest.mark.parametrize("varkind", ["input", "intermediate"])
@pytest.mark.parametrize("form", ["scalar", "elementwise", "broadcast"])
def test_census_row_narrows_exactly(cmp, varkind, form):
    q, vid, expect = census_query(cmp, varkind, form)
    env = interval_env(q, assume_mode="constrain")
    got = env[vid]
    assert list(zip(got.los, got.his)) == expect  # exact endpoints, no bump
    p = propagate(q)
    assert p.coverage.constrained == 1
    assert p.coverage.constrained_primitives == (("stelling_assume", 1),)
    assert p.coverage.inert == 0
    assert any(n.startswith("assume CONSTRAINED") for n in p.notes)
    assert any("the verdict holds where the precondition holds" in a for a in p.assumptions)
    # the inert control leaves the variable untouched
    inert_env = interval_env(q, assume_mode="inert")
    lo, hi = _v_box(varkind)
    assert set(zip(inert_env[vid].los, inert_env[vid].his)) == {(lo, hi)}


def test_point_bound_from_const():
    kvar = var(9)
    x, pred, aout = var(0), var(2, boolav()), var(3, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, kvar], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
        constvars=(kvar,),
        consts=(2.0,),
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (2.0, 4.0)
    assert propagate(q).coverage.constrained == 1


def test_point_bound_from_point_declared_input():
    x, k, pred, aout = var(0), var(1), var(2, boolav()), var(3, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            any_eqn(k, 2.0, 2.0),  # a point-declared input is a point interval
            eqn("le", [x, k], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (0.0, 2.0)
    assert (env[1].los[0], env[1].his[0]) == (2.0, 2.0)  # the bound itself: untouched
    assert propagate(q).coverage.constrained == 1


@pytest.mark.parametrize(
    "cmp, expect",
    [
        ("le", (2.0, 4.0)),  # k <= x  ===  x >= k
        ("lt", (2.0, 4.0)),  # k < x   ===  x > k, closed
        ("ge", (0.0, 2.0)),  # k >= x  ===  x <= k
        ("gt", (0.0, 2.0)),  # k > x   ===  x < k, closed
        ("eq", (2.0, 2.0)),
    ],
)
def test_point_bound_on_the_left_flips_the_comparator(cmp, expect):
    x, pred, aout = var(0), var(2, boolav()), var(3, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn(cmp, [lit(2.0), x], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == expect


# --- binding rule 1: strict inequalities close --------------------------------


def test_strict_closure_direction_witnessed():
    # x ∈ [0, 2], assume(x > 1), re-assert x > 1: on the SOUND closed
    # narrowing [1, 2] the obligation stays unknown (x = 1 is in the
    # narrowed set and does not satisfy x > 1); the unsound open narrowing
    # [nextafter(1), 2] would flip it to discharged — while excluding the
    # reals in (1, nextafter(1)) that the assumption licenses.
    x, p1, a1, p2, out = var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 2.0),
            eqn("gt", [x, lit(1.0)], p1),
            eqn("stelling_assume", [p1], a1),
            eqn("gt", [x, lit(1.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    env = interval_env(q, assume_mode="constrain")
    assert env[0].los[0] == 1.0  # exactly k — never nextafter(k)
    assert env[0].his[0] == 2.0
    p = propagate(q)
    assert p.obligations[0].status == "unknown"  # the sound behavior
    # sanity: the flip really is one ulp away — this is what unsoundness
    # would have discharged
    assert math.nextafter(1.0, INF) > 1.0


def test_closed_narrowing_still_discharges_the_nonstrict_consequence():
    # assume(x > 1) does soundly prove x >= 1
    x, p1, a1, p2, out = var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 2.0),
            eqn("gt", [x, lit(1.0)], p1),
            eqn("stelling_assume", [p1], a1),
            eqn("ge", [x, lit(1.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    assert propagate(q).obligations[0].status == "discharged"


# --- binding rule 2: no inversion through arithmetic --------------------------


def test_no_inversion_witnessed():
    # assume(x·x <= 4) with x ∈ [-3, 3]: only the square's interval
    # narrows; x is NEVER touched, so the obligation x >= -2.5 must stay
    # UNKNOWN. Any narrowing of x here is a soundness failure (naive
    # inversion under-approximates: sqrt loses the sign structure).
    x, w, p1, a1, p2, out = var(0), var(1), var(2, boolav()), var(3, boolav()), var(4, boolav()), var(5, boolav())
    q = close(
        [
            any_eqn(x, -3.0, 3.0),
            eqn("mul", [x, x], w),
            eqn("le", [w, lit(4.0)], p1),
            eqn("stelling_assume", [p1], a1),
            eqn("ge", [x, lit(-2.5)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (-3.0, 3.0)  # x untouched
    assert env[1].his[0] == 4.0  # the square's own interval narrowed
    assert env[1].los[0] <= -9.0  # lower side untouched (outward-rounded mul)
    p = propagate(q)
    assert p.obligations[0].status == "unknown"  # x was not invented into [-2, 2]
    assert p.coverage.constrained == 1


# --- binding rule 3: relational comparisons stay inert ------------------------


def test_relational_assume_stays_inert_with_the_reason():
    u, w, pred, aout = var(0), var(1), var(2, boolav()), var(3, boolav())
    q = close(
        [
            any_eqn(u, 0.0, 1.0),
            any_eqn(w, 0.0, 2.0),
            eqn("ge", [u, w], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    p = propagate(q)
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    note = next(n for n in p.notes if "DROPPED" in n)
    assert note.startswith("assume constraint DROPPED (inert in MVP propagation)")
    assert note.endswith(
        " (relational: both sides vary — constraining needs relational domains)"
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (0.0, 1.0)
    assert (env[1].los[0], env[1].his[0]) == (0.0, 2.0)


# --- conjunctions recurse; other logic stays inert ----------------------------


def test_conjunction_recursion_narrows_both_sides():
    x, p1, p2, c, aout = var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 10.0),
            eqn("ge", [x, lit(2.0)], p1),
            eqn("le", [x, lit(8.0)], p2),
            eqn("and", [p1, p2], c),
            eqn("stelling_assume", [c], aout),
        ],
        [aout],
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (2.0, 8.0)
    p = propagate(q)
    assert p.coverage.constrained == 1  # ONE assume equation
    assert sum(1 for n in p.notes if n.startswith("assume CONSTRAINED")) == 2


def test_nested_conjunction_recursion():
    x, p1, p2, p3, c1, c2, aout = (
        var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()),
        var(4, boolav()), var(5, boolav()), var(6, boolav()),
    )
    q = close(
        [
            any_eqn(x, 0.0, 10.0),
            eqn("ge", [x, lit(1.0)], p1),
            eqn("le", [x, lit(9.0)], p2),
            eqn("and", [p1, p2], c1),
            eqn("ge", [x, lit(3.0)], p3),
            eqn("and", [c1, p3], c2),
            eqn("stelling_assume", [c2], aout),
        ],
        [aout],
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (3.0, 9.0)


def test_mixed_conjunction_constrains_one_side_and_discloses_the_other():
    x, u, w, p1, p2, c, aout = (
        var(0), var(1), var(2), var(3, boolav()), var(4, boolav()),
        var(5, boolav()), var(6, boolav()),
    )
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            any_eqn(u, 0.0, 1.0),
            any_eqn(w, 0.0, 2.0),
            eqn("ge", [x, lit(2.0)], p1),
            eqn("ge", [u, w], p2),  # relational: no narrowing
            eqn("and", [p1, p2], c),
            eqn("stelling_assume", [c], aout),
        ],
        [aout],
    )
    p = propagate(q)
    assert p.coverage.constrained == 1 and p.coverage.inert == 0
    assert any(n.startswith("assume CONSTRAINED") for n in p.notes)
    conjunct_note = next(n for n in p.notes if "assume conjunct DROPPED" in n)
    assert "relational: both sides vary" in conjunct_note
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (2.0, 4.0)
    assert (env[1].los[0], env[1].his[0]) == (0.0, 1.0)  # u untouched
    # audit F5: the partial drop is visible in the coverage SUMMARY, not
    # only in the notes — both facts on one line
    assert p.coverage.dropped_conjuncts == 1
    summary = p.coverage.summary()
    assert "CONSTRAINED" in summary and "DROPPED" in summary
    assert "1 conjunct(s) DROPPED inside constrained assume(s)" in summary


@pytest.mark.parametrize("logic", ["or", "not", "xor"])
def test_disjunctive_and_negated_predicates_stay_inert(logic):
    x, p1, p2, c, aout = var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav())
    eqns = [
        any_eqn(x, 0.0, 4.0),
        eqn("ge", [x, lit(1.0)], p1),
        eqn("le", [x, lit(3.0)], p2),
    ]
    if logic == "not":
        eqns.append(eqn("not", [p1], c))
    else:
        eqns.append(eqn(logic, [p1, p2], c))
    eqns.append(eqn("stelling_assume", [c], aout))
    q = close(eqns, [aout])
    p = propagate(q)
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    note = next(n for n in p.notes if "DROPPED" in n)
    assert f"'{logic}'" in note
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (0.0, 4.0)


def test_ne_predicate_stays_inert():
    # ne is a comparison, but excluding a point does not narrow an interval
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ne", [x, lit(2.0)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    p = propagate(q)
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    assert any("'ne'" in n for n in p.notes)
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (0.0, 4.0)


def test_bool_literal_predicate_stays_inert():
    aout = var(0, boolav())
    q = close(
        [eqn("stelling_assume", [lit(True, boolav())], aout)],
        [aout],
    )
    p = propagate(q)
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    assert any("literal" in n for n in p.notes if "DROPPED" in n)


def test_top_producer_predicate_stays_inert():
    x, y, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("mystery_pred_op", [x], y),
            eqn("stelling_assume", [y], aout),
        ],
        [aout],
    )
    p = propagate(q)
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    assert any("'mystery_pred_op'" in n for n in p.notes)


def test_both_point_sides_definitely_true_stay_inert():
    p1, aout = var(0, boolav()), var(1, boolav())
    q = close(
        [
            eqn("ge", [lit(2.0), lit(1.0)], p1),  # definitely true: no-op
            eqn("stelling_assume", [p1], aout),
        ],
        [aout],
    )
    p = propagate(q)
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    assert any("both comparison sides are point intervals" in n for n in p.notes)
    assert any("definitely true" in n for n in p.notes)


@pytest.mark.parametrize(
    "cmp, a, b",
    [
        ("ge", 1.0, 2.0),
        ("gt", 2.0, 2.0),  # strict: a point is not above itself
        ("le", 3.0, 2.0),
        ("lt", 2.0, 2.0),
        ("eq", 1.0, 2.0),
    ],
)
def test_both_point_sides_definitely_false_raise(cmp, a, b):
    # no variable to narrow, but the vacuity is the same harness defect
    # the empty meet names: the precondition is unsatisfiable over the
    # declared set. Constrain mode raises; inert mode never does.
    p1, aout = var(0, boolav()), var(1, boolav())
    q = close(
        [
            eqn(cmp, [lit(a), lit(b)], p1),
            eqn(
                "stelling_assume", [p1], aout,
                source_info=["consts.py:9 (h)"],
            ),
        ],
        [aout],
    )
    with pytest.raises(UnsatisfiableAssumptionError) as exc:
        propagate(q)
    msg = str(exc.value)
    assert "consts.py:9" in msg
    assert "definitely false" in msg
    p = propagate(q, assume_mode="inert")  # never raises: dropped, disclosed
    assert p.coverage.inert == 1


def test_both_point_arrays_with_one_false_element_raise():
    # the universal reading: one definitely-false element falsifies the
    # whole assumed predicate
    p1, aout = var(0, boolav((2,))), var(1, boolav((2,)))
    q = close(
        [
            eqn("eq", [arr_lit([1.0, 2.0]), arr_lit([1.0, 3.0])], p1),
            eqn("stelling_assume", [p1], aout),
        ],
        [aout],
    )
    with pytest.raises(UnsatisfiableAssumptionError):
        propagate(q)


def test_definitely_false_point_conjunct_raises_through_the_conjunction():
    # and(x >= 1, 1 >= 2): the false constant conjunct falsifies the whole
    # precondition regardless of the narrowable half
    x, p1, p2, c, aout = var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, lit(1.0)], p1),
            eqn("ge", [lit(1.0), lit(2.0)], p2),
            eqn("and", [p1, p2], c),
            eqn("stelling_assume", [c], aout),
        ],
        [aout],
    )
    with pytest.raises(UnsatisfiableAssumptionError):
        propagate(q)
    assert propagate(q, assume_mode="inert").coverage.inert == 1


def test_cross_scope_predicate_stays_inert():
    # the assumed predicate is computed OUTSIDE the (transparent) scope the
    # assume lives in: its producing equation is not visible there, so the
    # assume stays inert rather than guessing across scopes.
    x, pred, jout = var(0), var(1, boolav()), var(2, boolav())
    pv, av = var(10, boolav()), var(11, boolav())
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(pv,),
            outvars=(av,),
            eqns=(eqn("stelling_assume", [pv], av),),
        )
    )
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, lit(2.0)], pred),
            eqn("jit", [pred], jout, params=[("jaxpr", inner)]),
        ],
        [jout],
    )
    p = propagate(q)
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    assert any("not visible in this scope" in n for n in p.notes)
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (0.0, 4.0)


# --- empty meet: the loud harness defect --------------------------------------


def test_empty_meet_raises_scalar_with_the_address():
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("ge", [x, lit(2.0)], pred),
            eqn(
                "stelling_assume", [pred], aout,
                source_info=["harness.py:42 (h)"],
            ),
        ],
        [aout],
    )
    with pytest.raises(UnsatisfiableAssumptionError) as exc:
        propagate(q)
    msg = str(exc.value)
    assert "harness.py:42" in msg  # the assume's source address
    assert "var 0" in msg  # the variable
    assert "[0.0, 1.0]" in msg  # its interval
    assert ">= 2.0" in msg  # the constraint
    # the accessor raises identically in constrain mode
    with pytest.raises(UnsatisfiableAssumptionError):
        interval_env(q, assume_mode="constrain")
    # inert mode does not raise: the constraint is dropped, disclosed
    assert propagate(q, assume_mode="inert").coverage.inert == 1


def test_empty_meet_raises_elementwise():
    x, pred, aout = var(0, f64((2,))), var(1, boolav((2,))), var(2, boolav((2,)))
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("ge", [x, arr_lit([0.5, 5.0])], pred),
            eqn(
                "stelling_assume", [pred], aout,
                source_info=["harness.py:7 (h)"],
            ),
        ],
        [aout],
    )
    with pytest.raises(UnsatisfiableAssumptionError) as exc:
        propagate(q)
    assert "harness.py:7" in str(exc.value)


def test_empty_meet_raises_for_eq_outside_the_box():
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("eq", [x, lit(3.0)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    with pytest.raises(UnsatisfiableAssumptionError):
        propagate(q)


def test_satisfiable_point_narrowing_does_not_raise():
    # assume(x >= 1) over [0, 1] narrows to the point [1, 1]: satisfiable
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("ge", [x, lit(1.0)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (1.0, 1.0)
    assert propagate(q).coverage.constrained == 1


# --- audit F1: strict-at-boundary emptiness is decidable and refused ----------


def _strict_boundary_query(cmp, lo, hi, k):
    x, pred, aout, p2, out = var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav())
    return close(
        [
            any_eqn(x, lo, hi),
            eqn(cmp, [x, lit(k)], pred),
            eqn(
                "stelling_assume", [pred], aout,
                source_info=["strict.py:5 (h)"],
            ),
            eqn("le", [x, lit(0.5)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )


@pytest.mark.parametrize(
    "cmp, lo, hi, k, fragment",
    [
        # the auditor's f1 shape: region (1, 1] = ∅ — the collapse channel
        ("gt", 0.0, 1.0, 1.0, "boundary"),
        # mirror: region [1, 1) = ∅ — the collapse channel
        ("lt", 1.0, 2.0, 1.0, "boundary"),
        # point variable: {1} > 1 = ∅ — a point-declared input is a point
        # interval, so this raises through the point-vs-point channel
        ("gt", 1.0, 1.0, 1.0, "definitely false"),
    ],
)
def test_strict_boundary_collapse_raises_never_refutes(cmp, lo, hi, k, fragment):
    # audit F1 (UNSOUND): the closed narrowing of a strict boundary assume
    # is the nonempty point [k, k] while the true region is EMPTY — a
    # violated-over-set judged on the stand-in minted a REFUTED whose
    # conditional claim is vacuously TRUE. The emptiness is decidable at
    # the assume site (strict comparator + meet collapsed onto the bound):
    # same refusal class as the empty meet. No verdict is ever assembled.
    q = _strict_boundary_query(cmp, lo, hi, k)
    with pytest.raises(UnsatisfiableAssumptionError) as exc:
        propagate(q)
    msg = str(exc.value)
    assert "strict.py:5" in msg
    assert fragment in msg
    # inert control: no raise, obligation judged over the full box
    p = propagate(q, assume_mode="inert")
    assert p.coverage.inert == 1


def test_strict_boundary_collapse_raises_elementwise():
    # any collapsed element empties the universal predicate
    x, pred, aout = var(0, f64((2,))), var(1, boolav((2,))), var(2, boolav((2,)))
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("gt", [x, arr_lit([0.5, 1.0])], pred),  # element 1 collapses
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    with pytest.raises(UnsatisfiableAssumptionError):
        propagate(q)


def test_strict_narrowing_away_from_the_boundary_still_constrains():
    # gt with k strictly below hi does not collapse: nonempty true region,
    # closed narrowing stands (the F1 fix must not over-refuse)
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("gt", [x, lit(0.999)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (0.999, 1.0)
    assert propagate(q).coverage.constrained == 1


def test_nonstrict_boundary_point_still_narrows_without_raising():
    # ge collapsing to a point stays satisfiable ({k}): the pre-existing
    # does-not-raise behavior stands (test_satisfiable_point_narrowing_
    # does_not_raise is the ge face; this pins le)
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("le", [x, lit(1.0)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (1.0, 1.0)
    assert propagate(q).coverage.constrained == 1


# --- forward-only narrowing ---------------------------------------------------


def test_forward_only_earlier_obligation_unaffected_later_one_narrowed():
    x = var(0)
    pA, oA = var(1, boolav()), var(2, boolav())
    ap, aout = var(3, boolav()), var(4, boolav())
    pB, oB = var(5, boolav()), var(6, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("le", [x, lit(2.0)], pA),
            eqn("stelling_assert", [pA], oA),  # BEFORE the assume: un-narrowed
            eqn("le", [x, lit(2.0)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("le", [x, lit(2.0)], pB),
            eqn("stelling_assert", [pB], oB),  # AFTER: narrowed
        ],
        [oA, oB],
    )
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown", "discharged"]
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (0.0, 2.0)


def test_assume_output_value_passes_through_unnarrowed():
    # the assume's own output is the predicate value computed BEFORE the
    # narrowing (forward-only, pass-through): still three-valued unknown
    x, ap, aout, out = var(0), var(1, boolav()), var(2, boolav()), var(3, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("le", [x, lit(2.0)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("stelling_assert", [aout], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"


# --- already-within, violated-after-narrowing, disclosure text ----------------


def test_already_within_still_counts_and_says_so():
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 2.0, 4.0),
            eqn("ge", [x, lit(1.0)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    p = propagate(q)
    assert p.coverage.constrained == 1  # the claim is still conditional
    assert any("already within the assumed region" in n for n in p.notes)
    assert any("the verdict holds where the precondition holds" in a for a in p.assumptions)
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (2.0, 4.0)


def test_narrowing_can_refute_over_the_assumed_set():
    # definitely false over the narrowed superset of (box ∩ assumed
    # region): a sound set-level refutation of the conditional claim
    x, ap, aout, pred, out = var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("ge", [x, lit(0.9)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("le", [x, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "violated-over-set"
    assert p.any_violated
    # the inert control leaves it unknown
    assert propagate(q, assume_mode="inert").obligations[0].status == "unknown"


def test_disclosure_texts_are_exact():
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, lit(1.5)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    p = propagate(q)
    assert (
        "assume CONSTRAINED at unknown location: narrowed var 0 to [1.5, 4.0]"
        in p.notes
    )
    assert (
        "constrained assume at unknown location: the verdict holds where the "
        "precondition holds — narrowed var 0 to [1.5, 4.0]"
    ) in p.assumptions
    assert "1 assume(s) CONSTRAINED (stelling_assume ×1)" in p.coverage.summary()


def test_constrained_assumption_rides_into_the_verdict_stamp():
    from stelling.verdict import make_verdict

    x, pred, aout, p2, out = var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, lit(1.5)], pred),
            eqn("stelling_assume", [pred], aout),
            eqn("ge", [x, lit(1.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    p = propagate(q)
    v = make_verdict(
        q,
        p,
        stelling_version="test",
        jax_version="none (hand-built IR)",
        precision_config="n/a",
    )
    assert v.status == "VERIFIED"
    assert any(
        "the verdict holds where the precondition holds" in a
        for a in v.stamp.assumptions
    )
    assert "CONSTRAINED" in v.render()


# --- the mode switch ----------------------------------------------------------

# DERIVED from the emitter, not a fourth hand-written copy. What this test
# pins is the RELATIONSHIP -- inert emits the shared base text with NO reason
# parenthetical, constrain appends one -- not a historical string. Pinning the
# literal made a correction to the disclosure look like a regression.
from stelling.propagate import ASSUME_DROP_NOTE  # noqa: E402

_OLD_DROP_NOTE = ASSUME_DROP_NOTE.format(where="unknown location")


def assume_rich_query():
    x, c1, a1 = var(0), var(1, boolav()), var(2, boolav())
    u, w, c2, a2 = var(3), var(4), var(5, boolav()), var(6, boolav())
    pred, out = var(7, boolav()), var(8, boolav())
    return close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("gt", [x, lit(1.5)], c1),
            eqn("stelling_assume", [c1], a1),  # constrainable
            any_eqn(u, 0.0, 1.0),
            any_eqn(w, 0.0, 2.0),
            eqn("ge", [u, w], c2),
            eqn("stelling_assume", [c2], a2),  # relational
            eqn("le", [x, lit(3.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_inert_mode_reproduces_the_pre_change_behavior_byte_identically():
    q = assume_rich_query()
    p = propagate(q, assume_mode="inert")
    # notes: exactly the pre-change DROPPED text, twice, with NO appended
    # reason — the reason parenthetical belongs to constrain mode only
    assert p.notes.count(_OLD_DROP_NOTE) == 2
    assert all(n == _OLD_DROP_NOTE for n in p.notes)
    # coverage: both assumes inert, none constrained
    assert p.coverage.inert == 2 and p.coverage.constrained == 0
    assert p.coverage.inert_primitives == (("stelling_assume", 2),)
    # obligation judged on the un-narrowed interval
    assert p.obligations[0].status == "unknown"
    assert p.assumptions == ()
    # env via the public accessor: un-narrowed
    env = interval_env(q, assume_mode="inert")
    assert (env[0].los[0], env[0].his[0]) == (0.0, 4.0)
    # and the accessor's default mode IS inert
    assert interval_env(q) == env


def test_constrain_mode_on_the_same_query_narrows_and_discloses():
    q = assume_rich_query()
    p = propagate(q)
    assert p.coverage.constrained == 1 and p.coverage.inert == 1
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (1.5, 4.0)
    assert p.obligations[0].status == "unknown"  # le(x, 3) undecided on [1.5, 4]


def no_assume_query():
    x, ex, pred, out = var(0), var(1), var(2, boolav()), var(3, boolav())
    return close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("exp", [x], ex),
            eqn("lt", [ex, lit(8.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_no_assume_query_is_identical_under_both_modes():
    # the §-integrity guard: a harness with no assumes must be untouched
    q = no_assume_query()
    assert propagate(q, assume_mode="constrain") == propagate(q, assume_mode="inert")
    assert propagate(q) == propagate(q, assume_mode="inert")
    assert interval_env(q, assume_mode="constrain") == interval_env(q, assume_mode="inert")


def test_unrecognized_assume_mode_raises_value_error():
    q = no_assume_query()
    with pytest.raises(ValueError):
        propagate(q, assume_mode="both")
    with pytest.raises(ValueError):
        interval_env(q, assume_mode="aggressive")


# --- the exact meet helper (unit) ---------------------------------------------


def scalar(lo, hi):
    return iv.IntervalArray(shape=(), los=(float(lo),), his=(float(hi),))


def test_meet_is_exact_no_outward_rounding():
    r = iv.meet(scalar(1.0, 3.0), scalar(2.0, 4.0))
    assert (r.los[0], r.his[0]) == (2.0, 3.0)  # exactly the operand endpoints
    # endpoint-identity even for values where a 1-ulp bump would show
    tight_lo = math.nextafter(2.0, INF)
    tight_hi = math.nextafter(3.0, -INF)
    r2 = iv.meet(scalar(tight_lo, 3.0), scalar(1.0, tight_hi))
    assert r2.los[0] == tight_lo and r2.his[0] == tight_hi


def test_meet_elementwise_and_point_results():
    a = iv.IntervalArray(shape=(2,), los=(0.0, 0.0), his=(4.0, 4.0))
    b = iv.IntervalArray(shape=(2,), los=(1.0, 4.0), his=(2.0, 9.0))
    r = iv.meet(a, b)
    assert list(zip(r.los, r.his)) == [(1.0, 2.0), (4.0, 4.0)]  # point allowed


def test_meet_empty_detection_raises():
    with pytest.raises(iv.IntervalError):
        iv.meet(scalar(0.0, 1.0), scalar(2.0, 3.0))
    a = iv.IntervalArray(shape=(2,), los=(0.0, 0.0), his=(1.0, 1.0))
    b = iv.IntervalArray(shape=(2,), los=(0.5, 5.0), his=(2.0, 6.0))
    with pytest.raises(iv.IntervalError):  # one empty element suffices
        iv.meet(a, b)


def test_meet_with_infinite_sides():
    r = iv.meet(scalar(-INF, INF), scalar(0.0, INF))
    assert (r.los[0], r.his[0]) == (0.0, INF)
    r2 = iv.meet(scalar(-INF, 2.0), scalar(-3.0, INF))
    assert (r2.los[0], r2.his[0]) == (-3.0, 2.0)


def test_meet_shape_mismatch_raises():
    a = iv.IntervalArray(shape=(2,), los=(0.0, 0.0), his=(1.0, 1.0))
    with pytest.raises(iv.IntervalError):
        iv.meet(a, scalar(0.0, 1.0))


# --- coverage bookkeeping for the new category --------------------------------


def test_coverage_counter_constrained_category():
    from stelling.coverage import CoverageCounter

    c = CoverageCounter()
    c.record_known("add")
    c.record_constrained("stelling_assume")
    c.record_inert("stelling_assume")
    cov = c.freeze()
    assert cov.total == 3
    assert cov.constrained == 1 and cov.inert == 1
    assert cov.constrained_primitives == (("stelling_assume", 1),)
    text = cov.summary()
    assert "1 assume(s) CONSTRAINED (stelling_assume ×1)" in text
    assert "1 constraint(s) DROPPED (stelling_assume ×1)" in text


def test_constrained_assume_is_not_counted_inert_and_not_known():
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, lit(1.5)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    cov = propagate(q).coverage
    assert cov.total == 3  # any, ge, assume
    assert cov.known == 2  # any, ge
    assert cov.inert == 0
    assert cov.constrained == 1
    assert cov.fraction_known < 1.0  # conditionality does not hide inside 100%


# --- audit F2: branch-scoped unsatisfiability degrades, never raises ----------


def _cond_query(branch_pred_eqns, branch_pred_var):
    """x ∈ [0, 1]; cond(sel, [identity, branch-with-assume], x); assert
    on the joined output. The selector interval straddles, so both
    branches run."""
    bx = var(20)
    ba = var(22, boolav())
    branch_t = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(bx,),
            outvars=(bx,),
            eqns=tuple(
                branch_pred_eqns(bx)
                + [eqn("stelling_assume", [branch_pred_var], ba)]
            ),
        )
    )
    by = var(30)
    branch_f = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(by,), outvars=(by,), eqns=())
    )
    x, w = var(0), var(1)
    sel = var(2, ir.Aval(kind="ShapedArray", shape=(), dtype="int32"))
    cout, pred, out = var(3), var(4, boolav()), var(5, boolav())
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(w, 0.0, 1.0),
            eqn("convert_element_type", [w], sel, params=[("new_dtype", "int32")]),
            eqn(
                "cond", [sel, x], cout,
                params=[("branches", (branch_f, branch_t))],
            ),
            eqn("lt", [cout, lit(10.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _branch_empty_meet(bx):
    bp = var(21, boolav())
    return [eqn("ge", [bx, lit(5.0)], bp)], bp


def _branch_strict_collapse(bx):
    bp = var(21, boolav())
    return [eqn("gt", [bx, lit(1.0)], bp)], bp


def _branch_point_false(bx):
    bp = var(21, boolav())
    return [eqn("ge", [lit(1.0), lit(2.0)], bp)], bp


@pytest.mark.parametrize(
    "builder", [_branch_empty_meet, _branch_strict_collapse, _branch_point_false]
)
def test_branch_scoped_unsatisfiable_assume_degrades_with_the_note(builder):
    # audit F2 (FRAGILE): an unsatisfiable assume inside a maybe-not-taken
    # cond branch is BRANCH-scoped — the other branch is real — so it must
    # not raise with the whole-domain message: the constraint is simply
    # not applied, and a note discloses the branch-local vacuity. All
    # three unsatisfiability shapes take the same branch-local path.
    q = _cond_query(lambda bx: builder(bx)[0], builder(var(20))[1])
    p = propagate(q)  # constrain default: MUST NOT raise
    note = next(
        n for n in p.notes if "assume unsatisfiable within this cond branch" in n
    )
    assert "definitely false whenever this branch is taken" in note
    assert "may be vacuous under the branch's precondition" in note
    # no narrowing was applied; the assume counts as a drop, not constrained
    assert p.coverage.constrained == 0
    assert p.coverage.inert == 1
    assert p.assumptions == ()  # no conditional claim was stamped
    # the outer domain is untouched and the obligation is judged normally
    env = interval_env(q, assume_mode="constrain")
    assert (env[0].los[0], env[0].his[0]) == (0.0, 1.0)
    assert p.obligations[0].status == "discharged"
    # inert control agrees on the obligation (the audit's regression check)
    assert propagate(q, assume_mode="inert").obligations[0].status == "discharged"


def test_jit_internal_unsatisfiable_assume_still_raises():
    # transparent call scopes execute unconditionally: the whole-domain
    # raise stands inside a top-level jit (audit F2's counter-face)
    jx = var(40)
    jp, ja = var(41, boolav()), var(42, boolav())
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(jx,),
            outvars=(jx,),
            eqns=(
                eqn("ge", [jx, lit(5.0)], jp),
                eqn("stelling_assume", [jp], ja),
            ),
        )
    )
    x, jout = var(0), var(1)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("jit", [x], jout, params=[("jaxpr", inner)]),
        ],
        [jout],
    )
    with pytest.raises(UnsatisfiableAssumptionError) as exc:
        propagate(q)
    assert "whole over-approximated domain" in str(exc.value)


def test_jit_inside_a_cond_branch_inherits_the_branch_scope():
    # a transparent scope nested in a possibly-untaken branch is still
    # branch-scoped: degrade with the note, never the whole-domain raise
    jx = var(50)
    jp, ja = var(51, boolav()), var(52, boolav())
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(jx,),
            outvars=(jx,),
            eqns=(
                eqn("ge", [jx, lit(5.0)], jp),
                eqn("stelling_assume", [jp], ja),
            ),
        )
    )
    bx, bout = var(20), var(21)
    branch_t = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(bx,),
            outvars=(bout,),
            eqns=(eqn("jit", [bx], bout, params=[("jaxpr", inner)]),),
        )
    )
    by = var(30)
    branch_f = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(by,), outvars=(by,), eqns=())
    )
    x, w = var(0), var(1)
    sel = var(2, ir.Aval(kind="ShapedArray", shape=(), dtype="int32"))
    cout = var(3)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(w, 0.0, 1.0),
            eqn("convert_element_type", [w], sel, params=[("new_dtype", "int32")]),
            eqn(
                "cond", [sel, x], cout,
                params=[("branches", (branch_f, branch_t))],
            ),
        ],
        [cout],
    )
    p = propagate(q)  # must not raise
    assert any(
        "assume unsatisfiable within this cond branch" in n for n in p.notes
    )


# --- audit F4: conditional REFUTED wording ------------------------------------

_UNCONDITIONAL_SET_LEVEL = (
    "  (set-level: at least one obligation is definitely false over "
    "the declared set — the stated box is not invariant as stated. "
    "Not a witness; not a counterexample to the program.)"
)


def _conditional_refuted_query():
    # the auditor's f4 shape: nonempty assumed region, obligation false on it
    x, ap, aout, pred, out = (
        var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav()),
    )
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("ge", [x, lit(0.9)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("le", [x, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_conditional_refuted_names_the_narrowed_set():
    from stelling.verdict import make_verdict

    q = _conditional_refuted_query()
    p = propagate(q)
    assert p.obligations[0].status == "violated-over-set"
    # audit F4: the detail must name the set actually judged
    assert "over the precondition-narrowed set" in p.obligations[0].detail
    assert "declared box" not in p.obligations[0].detail
    v = make_verdict(
        q, p, stelling_version="test", jax_version="none (hand-built IR)",
        precision_config="n/a",
    )
    assert v.status == "REFUTED"
    render = v.render()
    assert "set-level, conditional" in render
    assert "where the assumed precondition holds" in render
    assert "not invariant as stated" not in render  # a claim about the box
    assert "Not a witness" in render  # the non-witness caveat stays


def test_unconstrained_refuted_wording_is_byte_identical():
    from stelling.verdict import make_verdict

    x, pred, out = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("ge", [x, lit(2.0)], pred),  # definitely false, no assume
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert "over the declared box" in p.obligations[0].detail  # unchanged
    v = make_verdict(
        q, p, stelling_version="test", jax_version="none (hand-built IR)",
        precision_config="n/a",
    )
    assert v.status == "REFUTED"
    assert _UNCONDITIONAL_SET_LEVEL in v.render().splitlines()
    # the inert control on the constrained query also keeps the old texts
    q2 = _conditional_refuted_query()
    p2 = propagate(q2, assume_mode="inert")
    assert p2.obligations[0].status == "unknown"  # judged over the full box


# --- audit F6: non-finite constant bounds get the true reason -----------------


def test_infinite_constant_bound_reason_names_the_shape():
    # assume(x < inf): sound to drop, but the reason must not claim both
    # sides vary — the bound is the constant literal +inf
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("lt", [x, lit(math.inf)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    p = propagate(q)
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    note = next(n for n in p.notes if "DROPPED" in n)
    assert "not a finite point" in note
    assert "relational: both sides vary" not in note


def test_nan_constant_bound_reason_names_the_shape():
    x, pred, aout = var(0), var(1, boolav()), var(2, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, lit(math.nan)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    p = propagate(q)  # must not raise: the undecodable bound drops soundly
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    note = next(n for n in p.notes if "DROPPED" in n)
    assert "not a finite point" in note
    assert "relational: both sides vary" not in note


# --- audit F7: definite-verdict licensing splits by box exactness -------------


def test_exactness_classifier_unit():
    # declared input exact; any transfer output non-exact; exact-point
    # const exact; bracketed huge int (> 2**53) non-exact
    from stelling.propagate import _Propagator

    x, y = var(0), var(1)
    c, big = var(8), var(9)
    q = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(c, big),
            invars=(),
            outvars=(y,),
            eqns=(
                any_eqn(x, 0.0, 4.0),
                eqn("neg", [x], y),
            ),
        ),
        consts=(2.0, 2**60),
    )
    p = _Propagator("constrain")
    p.run(q.jaxpr, list(q.consts), [])
    assert 0 in p.exact  # stelling_any output: the box IS the declared set
    assert 1 not in p.exact  # transfer output (neg): over-approximation
    assert 8 in p.exact  # const 2.0: exact point
    assert 9 not in p.exact  # 2**60: decoded as a bracket, not its value


def _image_gap_query(assume_cmp, assume_bound):
    # the auditor's r1b core: x ∈ [-1, 1], w = x·x (true image [0, 1],
    # box [down(-1), up(1)]), assume(w cmp bound) — unsatisfiable in ℝ —
    # then the THEOREM w >= 0 as the obligation.
    x, w, ap, aout, p2, out = (
        var(0), var(1), var(2, boolav()), var(3, boolav()), var(4, boolav()), var(5, boolav()),
    )
    return close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("mul", [x, x], w),
            eqn(assume_cmp, [w, lit(assume_bound)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("ge", [w, lit(0.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )


def _rounding_pad_query():
    # the auditor's r1b channel B: a transfer's rounding pad pushes the box
    # past the strict boundary, so the collapse detector cannot fire; the
    # true region of v > 1 is empty over x ∈ [0, 1].
    #
    # This channel USES `mul` deliberately. It was `add [x, 0.0]` until
    # `add` moved to the exact-when-representable discipline, at which
    # point `x + 0.0` became exact, the box stopped exceeding its image,
    # `v > 1` collapsed onto [1.0, 1.0], and stelling correctly raised
    # UnsatisfiableAssumptionError — a harness defect, not a guard failure.
    # `mul` still carries an unconditional one-ulp bump (see the two
    # endpoint disciplines in interval.py), so `x * 1.0` reproduces the
    # padded box this channel exists to exercise. If `mul` is ever
    # converted too, this channel needs the same treatment again.
    x, v, ap, aout, p2, out = (
        var(0), var(1), var(2, boolav()), var(3, boolav()), var(4, boolav()), var(5, boolav()),
    )
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mul", [x, lit(1.0)], v),
            eqn("gt", [v, lit(1.0)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("le", [v, lit(0.5)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )


@pytest.mark.parametrize(
    "q_builder",
    [
        lambda: _image_gap_query("le", -0.5),  # channel A: image-gap (mul)
        _rounding_pad_query,  # channel B: rounding-pad (mul)
        lambda: _image_gap_query("eq", -0.5),  # channel C: eq outside the image
    ],
)
def test_uncertified_precondition_withholds_the_refutation(q_builder):
    # audit F7 (UNSOUND): all three channels minted a definite REFUTED
    # from an EMPTY precondition region (A and C refuting the theorem
    # x² >= 0). The narrowing target is a transfer output — its box may
    # exceed its true image, so the nonempty meet certifies nothing: the
    # definite violation is withheld from REFUTED.
    from stelling.verdict import make_verdict

    q = q_builder()
    p = propagate(q)  # no raise: the narrowing is sound and stays applied
    assert p.coverage.constrained == 1
    (ob,) = p.obligations
    assert ob.status == "unknown"  # never violated-over-set
    assert "WITHHELD from REFUTED" in ob.detail
    note = next(n for n in p.notes if "violation WITHHELD from REFUTED" in n)
    assert "precondition's satisfiability is uncertified" in note
    assert "a possibly-vacuous refutation is not a refutation" in note
    assert any(
        "precondition satisfiability UNCERTIFIED" in n for n in p.notes
    )
    assert any(
        "the conditional claim may be vacuous" in a for a in p.assumptions
    )
    v = make_verdict(
        q, p, stelling_version="test", jax_version="none (hand-built IR)",
        precision_config="n/a",
    )
    assert v.status == "UNKNOWN"  # never REFUTED
    # the inert control is the visibility instrument
    assert propagate(q, assume_mode="inert").obligations[0].status == "unknown"


def test_uncertified_verified_stays_allowed_with_the_vacuity_disclosure():
    # a satisfiable intermediate assume still cannot be certified from the
    # box: VERIFIED remains allowed (true, vacuously if empty) but carries
    # the may-be-vacuous note and stamped line
    from stelling.verdict import make_verdict

    x, w, ap, aout, p2, out = (
        var(0), var(1), var(2, boolav()), var(3, boolav()), var(4, boolav()), var(5, boolav()),
    )
    q = close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("mul", [x, x], w),
            eqn("le", [w, lit(0.5)], ap),  # satisfiable: |x| <= sqrt(0.5)
            eqn("stelling_assume", [ap], aout),
            eqn("le", [w, lit(0.6)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "discharged"
    v = make_verdict(
        q, p, stelling_version="test", jax_version="none (hand-built IR)",
        precision_config="n/a",
    )
    assert v.status == "VERIFIED"
    assert any(
        "precondition satisfiability uncertified" in a
        for a in v.stamp.assumptions
    )
    assert any(
        "precondition satisfiability UNCERTIFIED" in n for n in v.notes
    )


def test_declared_input_assume_still_certifies_both_faces():
    # an EXACT-box target (declared input): the box IS the true set, so
    # nonempty meets certify satisfiability and definite verdicts stand —
    # the REFUTED face (the F4 query) and the VERIFIED face, with no
    # uncertified disclosure anywhere
    from stelling.verdict import make_verdict

    q_refuted = _conditional_refuted_query()  # assume(x >= 0.9), x an input
    p = propagate(q_refuted)
    assert p.obligations[0].status == "violated-over-set"
    assert not any("UNCERTIFIED" in n for n in p.notes)
    assert make_verdict(
        q_refuted, p, stelling_version="test",
        jax_version="none (hand-built IR)", precision_config="n/a",
    ).status == "REFUTED"

    x, ap, aout, p2, out = (
        var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav()),
    )
    q_verified = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("ge", [x, lit(0.9)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("ge", [x, lit(0.5)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    p2r = propagate(q_verified)
    assert p2r.obligations[0].status == "discharged"
    v2 = make_verdict(
        q_verified, p2r, stelling_version="test",
        jax_version="none (hand-built IR)", precision_config="n/a",
    )
    assert v2.status == "VERIFIED"
    assert not any("uncertified" in a for a in v2.stamp.assumptions)


def test_withholding_is_forward_scoped():
    # a violation judged BEFORE any uncertified constraint was applied is
    # an unconditional set-level fact over the un-narrowed domain: it
    # stands. Only violations judged while the uncertified constraint is
    # in force are withheld.
    x = var(0)
    p0, o0 = var(1, boolav()), var(2, boolav())
    w, ap, aout = var(3), var(4, boolav()), var(5, boolav())
    p1, o1 = var(6, boolav()), var(7, boolav())
    q = close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("ge", [x, lit(2.0)], p0),  # definitely false over [-1, 1]
            eqn("stelling_assert", [p0], o0),  # BEFORE: unconditional
            eqn("mul", [x, x], w),
            eqn("le", [w, lit(-0.5)], ap),  # image-gap: uncertified
            eqn("stelling_assume", [ap], aout),
            eqn("ge", [w, lit(0.0)], p1),
            eqn("stelling_assert", [p1], o1),  # AFTER: withheld
        ],
        [o0, o1],
    )
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["violated-over-set", "unknown"]
    assert "WITHHELD" in p.obligations[1].detail


def test_uncertified_flag_withholds_even_exact_target_violations_after_it():
    # conservative by design: once ANY uncertified constraint is in
    # force, every subsequent definite violation is withheld — including
    # one whose comparison involves only the exact-box input (the
    # judgment env still contains the uncertified narrowing, and
    # value-flow precision is out of scope). Sound: withholding never
    # mints a wrong definite verdict.
    x, xa, xaout = var(0), var(1, boolav()), var(2, boolav())
    w, ap, aout = var(3), var(4, boolav()), var(5, boolav())
    p1, o1 = var(6, boolav()), var(7, boolav())
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("ge", [x, lit(0.9)], xa),  # certified input assume
            eqn("stelling_assume", [xa], xaout),
            eqn("mul", [x, x], w),
            eqn("le", [w, lit(0.9)], ap),  # uncertified intermediate assume
            eqn("stelling_assume", [ap], aout),
            eqn("le", [x, lit(0.5)], p1),  # false over the narrowed [0.9, 1]
            eqn("stelling_assert", [p1], o1),
        ],
        [o1],
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"
    assert "WITHHELD" in p.obligations[0].detail


def test_branch_invars_never_inherit_exactness_selector_correlation():
    # why branch invars are non-exact even when bound from a declared
    # input: the branch invar's box is the OPERAND's box, but the values
    # actually reaching the branch are filtered by the selector. Here the
    # in-branch assume(x >= 0.9) has a nonempty meet with the operand box
    # [0, 1], yet NO execution reaching the branch (x < 0.5) satisfies it
    # — the in-branch assumed region is empty by selector correlation, an
    # F7-class channel. The in-branch definite violation must therefore
    # be withheld, exactly as for any uncertified target. (This
    # supersedes the round-2 r1c pin "in-branch REFUTED stands".)
    bx = var(20)
    bp, ba = var(21, boolav()), var(22, boolav())
    bq, bo = var(23, boolav()), var(24, boolav())
    branch_t = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(bx,),
            outvars=(bx,),
            eqns=(
                eqn("ge", [bx, lit(0.9)], bp),
                eqn("stelling_assume", [bp], ba),  # nonempty per box; empty
                eqn("le", [bx, lit(0.5)], bq),  # per the reachable set
                eqn("stelling_assert", [bq], bo),  # false over [0.9, 1]
            ),
        )
    )
    by = var(30)
    branch_f = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(by,), outvars=(by,), eqns=())
    )
    x, w = var(0), var(1)
    sel = var(2, ir.Aval(kind="ShapedArray", shape=(), dtype="int32"))
    cout = var(3)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(w, 0.0, 1.0),
            eqn("convert_element_type", [w], sel, params=[("new_dtype", "int32")]),
            eqn(
                "cond", [sel, x], cout,
                params=[("branches", (branch_f, branch_t))],
            ),
        ],
        [cout],
    )
    p = propagate(q)  # must not raise
    (ob,) = p.obligations  # the in-branch assert
    assert ob.status == "unknown"  # withheld, never violated-over-set
    assert "WITHHELD" in ob.detail
    assert any(
        "precondition satisfiability UNCERTIFIED" in n for n in p.notes
    )


# --- audit F8: definitely-true assumes certify themselves ---------------------


def test_definitely_true_noop_assume_certifies_itself():
    # audit F8: assume(w <= 10) over w's box ⊂ (-inf, 10] is definitely
    # true on the whole over-approximated domain — the assumed region
    # EQUALS the reachable set, so satisfiability is certified even though
    # w is a transfer output. The subsequent genuine, unconditionally
    # decidable refutation (w >= 5 is false for every input) must land as
    # an honest REFUTED: no withholding, no uncertified disclosure.
    from stelling.verdict import make_verdict

    x, w, ap, aout, p2, out = (
        var(0), var(1), var(2, boolav()), var(3, boolav()), var(4, boolav()), var(5, boolav()),
    )
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("add", [x, lit(0.0)], w),
            eqn("le", [w, lit(10.0)], ap),  # definitely true: no-op meet
            eqn("stelling_assume", [ap], aout),
            eqn("ge", [w, lit(5.0)], p2),  # definitely false for every input
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "violated-over-set"
    assert "WITHHELD" not in p.obligations[0].detail
    assert not any("UNCERTIFIED" in n for n in p.notes)
    assert not any("uncertified" in a for a in p.assumptions)
    assert any("already within the assumed region" in n for n in p.notes)
    assert p.coverage.constrained == 1
    v = make_verdict(
        q, p, stelling_version="test", jax_version="none (hand-built IR)",
        precision_config="n/a",
    )
    assert v.status == "REFUTED"
    # agreement with the inert control on the decidable refutation
    assert propagate(q, assume_mode="inert").obligations[0].status == "violated-over-set"


def test_strict_boundary_noop_assume_does_not_certify():
    # the F8 refinement, pinned: a STRICT no-op at the box's lower
    # boundary (box lo == k) is NOT definitely true — x == k is reachable
    # per the box and excluded by the predicate, so the region may be a
    # strict subset and satisfiability stays uncertified. The nonstrict
    # form at the same bound IS definitely true and certifies.
    # k must BE w's box lower endpoint — that is what makes the strict
    # form a boundary no-op. `add [x, 0.0]` used to pad the box down by an
    # ulp, so this was nextafter(1.0, -inf); under the
    # exact-when-representable rule `x + 0.0` is exact and the endpoint is
    # 1.0 itself. Pinning the old padded value would make `w > k`
    # definitely true and silently stop testing the boundary case.
    k = 1.0  # w's box lower endpoint (x + 0.0 is exact)

    def q_with(cmp):
        x, w, ap, aout, p2, out = (
            var(0), var(1), var(2, boolav()), var(3, boolav()), var(4, boolav()), var(5, boolav()),
        )
        return close(
            [
                any_eqn(x, 1.0, 2.0),
                eqn("add", [x, lit(0.0)], w),
                eqn(cmp, [w, lit(k)], ap),
                eqn("stelling_assume", [ap], aout),
                eqn("ge", [w, lit(5.0)], p2),  # definitely false over the box
                eqn("stelling_assert", [p2], out),
            ],
            [out],
        )

    p_strict = propagate(q_with("gt"))  # no-op meet, but NOT definitely true
    assert p_strict.obligations[0].status == "unknown"
    assert "WITHHELD" in p_strict.obligations[0].detail
    p_nonstrict = propagate(q_with("ge"))  # definitely true: certified
    assert p_nonstrict.obligations[0].status == "violated-over-set"
    assert not any("UNCERTIFIED" in n for n in p_nonstrict.notes)


# --- audit F9: nonvacuity is withheld like asserts ----------------------------


def test_uncertified_nonvacuity_failed_face_is_withheld():
    # audit F9: the auditor's construction — nonvacuity(w >= 0), true for
    # every real, judged over the uncertified-narrowed [.., -0.5] — must
    # not mint the definite FAILED stamp sentence: undecided, with the
    # note naming the uncertified narrowing.
    from stelling.verdict import make_verdict

    x, w, ap, aout = var(0), var(1), var(2, boolav()), var(3, boolav())
    m, om = var(4, boolav()), var(5, boolav())
    p2, out = var(6, boolav()), var(7, boolav())
    q = close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("mul", [x, x], w),
            eqn("le", [w, lit(-0.5)], ap),  # image-empty: uncertified cut
            eqn("stelling_assume", [ap], aout),
            eqn("ge", [w, lit(0.0)], m),  # true for every real
            eqn("stelling_nonvacuity", [m], om),
            eqn("le", [x, lit(2.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out, om],
    )
    p = propagate(q)
    assert p.nonvacuity_checks[0].status == "unknown"  # never violated-over-set
    assert "WITHHELD from FAILED" in p.nonvacuity_checks[0].detail
    assert any("nonvacuity FAILED face WITHHELD" in n for n in p.notes)
    v = make_verdict(
        q, p, stelling_version="test", jax_version="none (hand-built IR)",
        precision_config="n/a",
    )
    assert v.stamp.nonvacuity.startswith("undecided")
    assert "FAILED" not in v.stamp.nonvacuity
    assert any("nonvacuity FAILED face WITHHELD" in n for n in v.notes)


def test_certified_nonvacuity_failed_keeps_the_exact_sentence():
    # no uncertified constraint in force: the definite FAILED sentence is
    # unchanged byte-for-byte
    from stelling.verdict import make_verdict

    x, m, om, p2, out = (
        var(0), var(1, boolav()), var(2, boolav()), var(3, boolav()), var(4, boolav()),
    )
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("lt", [x, lit(0.0)], m),  # definitely false membership
            eqn("stelling_nonvacuity", [m], om),
            eqn("ge", [x, lit(1.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out, om],
    )
    p = propagate(q)
    assert p.nonvacuity_checks[0].status == "violated-over-set"
    v = make_verdict(
        q, p, stelling_version="test", jax_version="none (hand-built IR)",
        precision_config="n/a",
    )
    assert v.stamp.nonvacuity == (
        "FAILED — a membership condition is definitely false: the stated "
        "point is NOT in the declared set (harness defect, not a box fact)"
    )
