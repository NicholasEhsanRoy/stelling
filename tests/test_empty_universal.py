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


# -- the SUMMARY line, not only the note beside it ---------------------------
#
# Audit 0.2.0 B8a FIXUP, item 2. `Stamp.nonvacuity` is a PUBLIC field and
# `Verdict.render` prints it ABOVE the notes, so it is the sentence a reader
# meets first. For a zero-element membership condition it read
#
#     nonvacuity: checked — 1 membership condition(s) definitely true
#                 (the declared set contains the stated point)
#
# — measured on `8772ced`, with the item-6 note sitting underneath it. No
# point was tested and the parenthetical is false. The note was put BESIDE
# the false sentence rather than correcting it.

_F64 = lambda sh: ir.Aval(kind="ShapedArray", shape=sh, dtype="float64")
_BOOL = lambda sh: ir.Aval(kind="ShapedArray", shape=sh, dtype="bool")


def _with_membership(sizes: tuple[int, ...]) -> ir.ClosedJaxpr:
    """A real 2-element assert plus one membership condition per size."""
    ids = iter(range(100))
    x = ir.Var(id=next(ids), aval=_F64((2,)))
    apred = ir.Var(id=next(ids), aval=_BOOL((2,)))
    aout = ir.Var(id=next(ids), aval=_BOOL((2,)))
    zero = ir.Literal(val=0.0, aval=_F64(()))
    eqns = [
        ir.JaxprEqn(
            primitive="stelling_any",
            invars=(),
            outvars=(x,),
            params=(
                ("shape", (2,)),
                ("dtype", "float64"),
                ("lo", 1.0),
                ("hi", 2.0),
            ),
        ),
        ir.JaxprEqn(primitive="ge", invars=(x, zero), outvars=(apred,)),
        ir.JaxprEqn(
            primitive="stelling_assert", invars=(apred,), outvars=(aout,)
        ),
    ]
    outs = [aout]
    for size in sizes:
        pt = ir.Var(id=next(ids), aval=_F64((size,)))
        pred = ir.Var(id=next(ids), aval=_BOOL((size,)))
        out = ir.Var(id=next(ids), aval=_BOOL((size,)))
        eqns += [
            ir.JaxprEqn(
                primitive="stelling_any",
                invars=(),
                outvars=(pt,),
                params=(
                    ("shape", (size,)),
                    ("dtype", "float64"),
                    ("lo", 1.0),
                    ("hi", 2.0),
                ),
            ),
            ir.JaxprEqn(primitive="ge", invars=(pt, zero), outvars=(pred,)),
            ir.JaxprEqn(
                primitive="stelling_nonvacuity", invars=(pred,), outvars=(out,)
            ),
        ]
        outs.append(out)
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=tuple(outs), eqns=tuple(eqns)
        )
    )


def _verdict_for(sizes):
    closed = _with_membership(sizes)
    p = propagate(closed)
    assert all(c.status == "discharged" for c in p.nonvacuity_checks)
    return make_verdict(
        closed,
        p,
        stelling_version="test",
        jax_version="none: hand-built IR",
        precision_config="jax_enable_x64=True (hand-built f64 IR)",
    )


def test_an_all_vacuous_nonvacuity_does_not_say_checked():
    """REDDENS ON REVERT of the summary line.

    Every membership condition is over a zero-element array, so nothing
    tied the declared set to a point — and `startswith("checked")`, which
    is what gates the VERIFIED caveat, must therefore be false here."""
    v = _verdict_for((0,))
    assert v.status == "VERIFIED", "the conditions ARE true over the empty set"
    assert v.stamp.nonvacuity.startswith("VACUOUS — "), v.stamp.nonvacuity
    assert "ZERO elements" in v.stamp.nonvacuity
    assert "contains the stated point" not in v.stamp.nonvacuity
    # ... and the caveat the prefix gates now fires, where "checked"
    # silenced it
    assert any(
        "this VERIFIED may be vacuous" in n for n in v.notes
    ), v.notes
    assert "nonvacuity: VACUOUS — " in v.render()


def test_a_mix_counts_the_vacuous_conditions_out_of_the_total():
    """One real condition decided true DOES tie the declared set to a
    stated point, so this stays `checked` — but the total must not absorb
    the vacuous one, which is what "2 membership condition(s) definitely
    true" did."""
    v = _verdict_for((0, 2))
    assert v.stamp.nonvacuity.startswith("checked in part — 1 of 2 ")
    assert "VACUOUSLY over ZERO elements" in v.stamp.nonvacuity
    assert not any("this VERIFIED may be vacuous" in n for n in v.notes)


def test_no_zero_element_condition_leaves_the_sentence_untouched():
    """THE CONTROL. The ordinary case is byte-for-byte what it was."""
    v = _verdict_for((2, 3))
    assert v.stamp.nonvacuity == (
        "checked — 2 membership condition(s) definitely true "
        "(the declared set contains the stated point)"
    )


def test_the_two_assembly_paths_cannot_spell_this_sentence_differently():
    """The solver-assisted verdict used to carry its own copy of this
    wording, with a comment requiring the two to stay byte-identical by
    hand. A requirement is not a mechanism: there is ONE minter, and both
    paths call it."""
    from stelling import solvers, verdict
    from stelling.propagate import nonvacuity_summary

    assert verdict.nonvacuity_summary is nonvacuity_summary
    assert solvers.nonvacuity_summary is nonvacuity_summary
    assert not hasattr(solvers, "_nonvacuity_summary")
