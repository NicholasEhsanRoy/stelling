# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The div-straddle decline: real-mode float division DECLINES when the
divisor interval straddles zero, instead of silently returning [-inf, inf].

The decline produces a noted UNKNOWN rather than an uninformative wide box
that reads as "100% known" while carrying no information. The message names
the divisor interval and suggests narrowing the envelope or adding an assume.

Three faces:
1. A straddling divisor gets the decline (UNKNOWN with a note).
2. A strictly-positive divisor does NOT decline (the division proceeds).
3. An assume that narrows the divisor before the division prevents the
   decline (the narrowed interval no longer straddles).

Hand-built IR throughout — no jax needed.
"""

from __future__ import annotations

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.propagate import propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, a=F64):
    return ir.Var(id=i, aval=a)


def lit(v, a=F64):
    return ir.Literal(val=v, aval=a)


def any_eqn(out, lo, hi):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(
            ("shape", out.aval.shape),
            ("dtype", out.aval.dtype),
            ("lo", lo),
            ("hi", hi),
        ),
    )


def eqn(prim, ins, out, params=()):
    return ir.JaxprEqn(
        primitive=prim, invars=tuple(ins), outvars=(out,), params=tuple(params)
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns)
        )
    )


# --- face 1: straddling divisor -> decline (UNKNOWN with note) ----------------


def test_div_straddle_declines_with_note():
    """a / b where b spans (-1, 1): the divisor straddles zero, so the
    transfer declines and the obligation goes to UNKNOWN with a note
    mentioning the straddle."""
    a, b = var(0), var(1)
    q_out, pred, out = var(2), var(3, BOOL), var(4, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 10.0),
            any_eqn(b, -1.0, 1.0),
            eqn("div", [a, b], q_out),
            eqn("gt", [q_out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert p.obligations[0].status == "unknown"
    # The note must mention the straddle and give actionable guidance
    assert any("straddles zero" in n for n in p.notes), (
        f"expected a 'straddles zero' note, got: {p.notes}"
    )
    assert any("assume" in n.lower() for n in p.notes), (
        f"expected 'assume' remedy in notes, got: {p.notes}"
    )


# --- face 2: non-straddling divisor -> no decline (VERIFIED) ------------------


def test_div_positive_divisor_verifies():
    """a / b where b spans (1, 10): the divisor is strictly positive, so the
    transfer proceeds normally and the obligation can discharge."""
    a, b = var(0), var(1)
    q_out, pred, out = var(2), var(3, BOOL), var(4, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 10.0),
            any_eqn(b, 1.0, 10.0),
            # a/b with a in [1,10], b in [1,10] => result in [0.1, 10]
            # assert result > 0 should verify
            eqn("div", [a, b], q_out),
            eqn("gt", [q_out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert p.obligations[0].status == "discharged"
    # No straddle note
    assert not any("straddles zero" in n for n in p.notes), (
        f"unexpected straddle note: {p.notes}"
    )


# --- face 3: assume narrows divisor before division -> no decline -------------


def test_div_assume_narrows_past_straddle():
    """a / b where b is declared (-1, 10) but assume(b >= 1) narrows it to
    [1, 10] before the division. The narrowed divisor no longer straddles
    zero, so the transfer proceeds and the obligation can decide.

    Note: assume(b > 0) now also works — in real mode it narrows to [0, 10]
    which boundary-aware division handles, and in ieee mode it bumps to
    [min_positive, 10]. Using >= 1 here gives the cleanest test shape."""
    a, b = var(0), var(1)
    # assume(b >= 1): we need a ge comparison and a stelling_assume
    pred_assume = var(2, BOOL)
    assume_out = var(3, BOOL)
    q_out, pred, out = var(4), var(5, BOOL), var(6, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 10.0),
            any_eqn(b, -1.0, 10.0),
            # assume(b >= 1)
            eqn("ge", [b, lit(1.0)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            # now b is narrowed to [1, 10]
            eqn("div", [a, b], q_out),
            eqn("gt", [q_out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    # The assume should narrow b past the straddle, allowing the div to proceed
    assert p.obligations[0].status == "discharged", (
        f"expected 'discharged', got {p.obligations[0].status}; notes: {p.notes}"
    )
    # No straddle note
    assert not any("straddles zero" in n for n in p.notes), (
        f"unexpected straddle note: {p.notes}"
    )


# --- edge cases ---------------------------------------------------------------


def test_div_boundary_at_zero_lower_declines_without_a_certificate():
    """b DECLARED as [0, 1]: zero at the lower boundary, and zero is a
    declared value of b.

    This used to discharge — `boundary_div` returned `[1/1, +inf]` and
    `a/b > 0` followed. It was a false VERIFIED of exactly the class the
    other three zero-containing shapes decline for (audit 0.2.0 B5-1):
    `b = 0` is a point of the declared set, real division has no value
    there, and nothing in the verdict said the point had been dropped.
    `boundary_div` is reached only under a strict-assume certificate now,
    and a DECLARATION whose endpoint is zero is not one — the endpoint is
    a value the caller asked for.
    """
    a, b = var(0), var(1)
    q_out, pred, out = var(2), var(3, BOOL), var(4, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 5.0),
            any_eqn(b, 0.0, 1.0),
            eqn("div", [a, b], q_out),
            eqn("gt", [q_out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert p.obligations[0].status == "unknown"
    assert any("REACHES zero at a boundary" in n for n in p.notes), p.notes


def test_div_boundary_at_zero_lower_decides_under_a_strict_assume():
    """The same shape with the zero EXCLUDED: `assume(b > 0)` narrows to
    the closed `[0, 1]` — an interval cannot hold an open bound — but the
    strictness is recorded, `boundary_div` is reached, and `a/b > 0`
    discharges. This is the capability the 0.2.0 row exists for, and it is
    also the remedy the decline above recommends."""
    a, b = var(0), var(1)
    pred_assume, assume_out = var(5, BOOL), var(6, BOOL)
    q_out, pred, out = var(2), var(3, BOOL), var(4, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 5.0),
            any_eqn(b, 0.0, 1.0),
            eqn("gt", [b, lit(0.0)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            eqn("div", [a, b], q_out),
            eqn("gt", [q_out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert p.obligations[0].status == "discharged", p.obligations[0].detail
    assert not any("straddles zero" in n for n in p.notes)


def test_div_negative_divisor_does_not_decline():
    """b declared as [-10, -1]: strictly negative, does NOT straddle zero.
    The division should proceed."""
    a, b = var(0), var(1)
    q_out, pred, out = var(2), var(3, BOOL), var(4, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 5.0),
            any_eqn(b, -10.0, -1.0),
            # a/b with a in [1,5], b in [-10,-1] => result in [-5, -0.1]
            # assert result < 0 should verify
            eqn("div", [a, b], q_out),
            eqn("lt", [q_out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert p.obligations[0].status == "discharged"
    assert not any("straddles zero" in n for n in p.notes)
