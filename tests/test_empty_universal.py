# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE EMPTY UNIVERSAL IS DISCLOSED WHERE A CONSUMER READS — audit 0.2.0
B8a, item 6 (M18).

A universal claim over a size-0 array is vacuously true and stelling said
so — but only as the words *"for all 0 element(s)"* inside
`ObligationReport.detail`. There was no note, no coverage entry and no
stamp field naming the empty universal, so a consumer reading `status` and
`notes` — which is what the rendered verdict leads with, and what a CI gate
reads — saw an ordinary VERIFIED over a program in which nothing was
checked.

Zero-dep on purpose: hand-built IR, no jax, so the zero-dep lane runs it.
"""

from __future__ import annotations

from stelling import ir
from stelling.propagate import EMPTY_UNIVERSAL_DETAIL, propagate
from stelling.verdict import make_verdict


def test_a_vacuous_verified_over_a_zero_element_array_reaches_the_notes():
    """REDDENS ON REVERT of the disclosure.

    A universal claim over a size-0 array is vacuously true and stelling
    said so — but only as the words "for all 0 element(s)" inside
    `ObligationReport.detail`. Measured on `aabb58d`:

        STATUS: VERIFIED
        coverage: 3 eqns: 3 known (100%)
        notes: nothing about the emptiness
        obligation #0: discharged | definitely true for all 0 element(s)

    A consumer reading `status` and `notes` — the rendered verdict's first
    lines, and what a CI gate reads — saw an ordinary VERIFIED over a query
    in which nothing was checked. The status is unchanged (the claim IS
    true over the empty set); what changes is that the reader is told.
    """
    f64 = ir.Aval(kind="ShapedArray", shape=(0,), dtype="float64")
    b = ir.Aval(kind="ShapedArray", shape=(0,), dtype="bool")
    x, pred, out = (
        ir.Var(id=0, aval=f64),
        ir.Var(id=1, aval=b),
        ir.Var(id=2, aval=b),
    )
    closed = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(),
            outvars=(out,),
            eqns=(
                ir.JaxprEqn(
                    primitive="stelling_any",
                    invars=(),
                    outvars=(x,),
                    params=(
                        ("shape", (0,)),
                        ("dtype", "float64"),
                        ("lo", 1.0),
                        ("hi", 2.0),
                    ),
                ),
                ir.JaxprEqn(
                    primitive="le",
                    invars=(
                        x,
                        ir.Literal(
                            val=0.5,
                            aval=ir.Aval(
                                kind="ShapedArray", shape=(), dtype="float64"
                            ),
                        ),
                    ),
                    outvars=(pred,),
                ),
                ir.JaxprEqn(
                    primitive="stelling_assert", invars=(pred,), outvars=(out,)
                ),
            ),
        )
    )
    p = propagate(closed)
    assert p.obligations[0].status == "discharged"
    assert p.obligations[0].detail == EMPTY_UNIVERSAL_DETAIL
    v = make_verdict(
        closed,
        p,
        stelling_version="test",
        jax_version="none: hand-built IR",
        precision_config="jax_enable_x64=True (hand-built f64 IR)",
    )
    assert v.status == "VERIFIED", "a universal over the empty set IS true"
    vacuous = [n for n in v.notes if "VACUOUSLY discharged" in n]
    assert len(vacuous) == 1, v.notes
    assert "ZERO elements" in vacuous[0]
    assert "VACUOUSLY discharged" in v.render()
