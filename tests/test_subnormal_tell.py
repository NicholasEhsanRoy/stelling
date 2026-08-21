# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The real-mode subnormal TELL: it must fire, and it must stay quiet.

`propagate._refuse_non_f64_float` calls gating on ``MIN_NORMAL`` *"A
DELIBERATE DEPARTURE FROM A STATED POSTURE"*, and `_t_sign`'s own docstring
names the residue the departure left behind: *"real-mode ``gt([1e-320,
1e-300], 0)`` returns definite TRUE today — the same device-dependence
answered the other way."* `sign` and `rem` took the departure; the
comparisons did not, and the ruling is that they still must not. The posture
stays; what the verdict gains is a sentence saying WHICH SEMANTICS it is
speaking, so a 100%-coverage VERIFIED with no decline and no ⊤, over a box
the target reads as zero, cannot be mistaken for a hardware-compliance
claim.

**BOTH DIRECTIONS, AND THE SECOND ONE IS THE HARD ONE.** A tell that cannot
fire and a tell that fires on everything are the same defect wearing
different clothes, and this campaign has closed that shape six times or
more. So the silence cases here outnumber the firing ones, and they are not
"some other program": they are the SAME declared box with a claim the flush
does not decide differently, the same comparison one ulp above the band, and
a declaration that merely CONTAINS the band the way an ordinary ``[-10, 10]``
does.

Hand-built IR, so the whole file runs in the zero-dep lane; the jax leg at
the bottom is `importorskip`ed and is the end-to-end one.
"""

from __future__ import annotations

import pytest

from stelling import interval as iv
from stelling import ir
from stelling import propagate as P
from stelling.propagate import propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
F32 = ir.Aval(kind="ShapedArray", shape=(), dtype="float32")
I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")

TELL_HEAD = "SUBNORMAL-SENSITIVE DEFINITE ANSWER"

# float64's smallest positive normal and one subnormal box strictly inside
# the band below it. 1e-300 is a NORMAL float64 — the box REACHES into the
# band rather than sitting wholly inside it, which is the shape the sweep
# measured and the weaker precondition for the tell.
MIN_NORMAL_F64 = 2.0**-1022
SUB_LO, SUB_HI = 1e-320, 1e-300


def var(i, aval=F64):
    return ir.Var(id=i, aval=aval)


def any_eqn(out, lo, hi, dtype="float64"):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", dtype), ("lo", lo), ("hi", hi)),
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=outvars, eqns=tuple(eqns))
    )


def cmp_query(prim, lo, hi, rhs, *, dtype="float64", aval=F64):
    """``assert_(x <prim> rhs)`` over a declared ``[lo, hi]``."""
    x, pred, out = var(0, aval), var(1, BOOL), var(2, BOOL)
    return close(
        [
            any_eqn(x, lo, hi, dtype),
            ir.JaxprEqn(
                primitive=prim,
                invars=(x, ir.Literal(val=rhs, aval=aval)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )


def tells(p) -> list[str]:
    return [n for n in p.notes if n.startswith(TELL_HEAD)]


# --------------------------------------------------------------------------
# 1. IT FIRES — on the shape the sweep measured
# --------------------------------------------------------------------------


def test_the_tell_fires_on_the_subnormal_verified():
    """The subject: VERIFIED, 100% coverage, no decline, no ⊤ — and now a
    sentence saying it is an ℝ claim."""
    p = propagate(cmp_query("gt", SUB_LO, SUB_HI, 0.0))
    assert p.obligations[0].status == "discharged"
    assert p.coverage.known == p.coverage.total  # 100%: nothing declined
    assert len(tells(p)) == 1
    note = tells(p)[0]
    assert "semantics='real'" in note
    assert "NOT a claim about what the hardware computes" in note
    assert "semantics='ieee'" in note


def test_the_tell_names_the_dtypes_own_min_normal_not_a_constant():
    """float64 says 2**-1022; float32 says 2**-126. A tell that named one
    hard-coded threshold would be wrong for three of the four formats."""
    f64 = tells(propagate(cmp_query("gt", SUB_LO, SUB_HI, 0.0)))[0]
    assert "float64: 0 < |x| < 2**-1022" in f64
    assert repr(MIN_NORMAL_F64) in f64
    assert "2**-126" not in f64

    f32 = tells(
        propagate(cmp_query("gt", 1.4e-45, 1e-40, 0.0, dtype="float32", aval=F32))
    )[0]
    assert "float32: 0 < |x| < 2**-126" in f32
    assert repr(2.0**-126) in f32
    assert "2**-1022" not in f32


@pytest.mark.parametrize(
    "prim,rhs",
    [
        # each row paired with a right-hand side whose ℝ-definite answer the
        # DAZ flush actually takes away. A row absent here would be a row the
        # tell CANNOT fire on, which is the defect this file is shaped
        # against — so every registered row is listed.
        ("gt", 0.0),        # ℝ definitely true;  flushed: undecided
        ("le", 0.0),        # ℝ definitely false; flushed: undecided
        ("eq", 0.0),        # ℝ definitely false; flushed: undecided
        ("ne", 0.0),        # ℝ definitely true;  flushed: undecided
        ("lt", SUB_LO),     # ℝ definitely false; flushed: undecided
        ("ge", SUB_LO),     # ℝ definitely true;  flushed: undecided
    ],
)
def test_every_comparison_row_can_fire(prim, rhs):
    assert sorted(P._SUBNORMAL_TELL_ROWS) == ["eq", "ge", "gt", "le", "lt", "ne"]
    p = propagate(cmp_query(prim, SUB_LO, SUB_HI, rhs))
    assert len(tells(p)) == 1, p.notes
    assert f"{prim!r}" in tells(p)[0]


def test_the_tell_counts_elements_and_reports_more_than_one():
    """An array declaration answers definitely for every element, and the
    count in the sentence is the number of them the flush undecides."""
    aval = ir.Aval(kind="ShapedArray", shape=(3,), dtype="float64")
    x, pred, out = var(0, aval), var(1, ir.Aval(kind="ShapedArray", shape=(3,), dtype="bool")), var(2, BOOL)
    q = close(
        [
            ir.JaxprEqn(
                primitive="stelling_any",
                invars=(),
                outvars=(x,),
                params=(("shape", (3,)), ("dtype", "float64"),
                        ("lo", SUB_LO), ("hi", SUB_HI)),
            ),
            ir.JaxprEqn(
                primitive="gt",
                invars=(x, ir.Literal(val=0.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    p = propagate(q)
    assert p.obligations[0].status == "discharged"
    assert "definitely for 3 element(s)" in tells(p)[0]


# --------------------------------------------------------------------------
# 2. IT STAYS QUIET — the controls, which are the point
# --------------------------------------------------------------------------


def test_silent_on_the_same_box_when_the_flush_changes_nothing():
    """THE SHARPEST CONTROL. Same declared box, same dtype, same band
    contact — a claim the flush does not decide differently. A tell that
    fired here would be firing on the BOX, which is the wrong predicate."""
    p = propagate(cmp_query("lt", SUB_LO, SUB_HI, 1.0))
    assert p.obligations[0].status == "discharged"
    assert tells(p) == []


def test_silent_on_an_ordinary_normal_box():
    p = propagate(cmp_query("gt", 1.0, 2.0, 0.0))
    assert p.obligations[0].status == "discharged"
    assert tells(p) == []


def test_silent_on_a_declaration_that_merely_contains_the_band():
    """``[-10, 10]`` CONTAINS the whole subnormal band, and the ordinary
    harness in docs/harness-api.md declares exactly that. The haze is the
    identity on a box that already holds 0, so nothing here reaches the
    tell — this is the "fires on everything" leg."""
    p = propagate(cmp_query("gt", -10.0, 10.0, -20.0))
    assert p.obligations[0].status == "discharged"
    assert tells(p) == []


def test_silent_one_step_above_the_band():
    """The smallest NORMAL float64 is not in the open band, so a box that
    starts there is untouched. Off-by-one on the band edge would show here."""
    p = propagate(cmp_query("gt", MIN_NORMAL_F64, 4.0 * MIN_NORMAL_F64, 0.0))
    assert p.obligations[0].status == "discharged"
    assert tells(p) == []
    # and the control for that control: one ulp of exponent lower DOES fire
    low = propagate(cmp_query("gt", MIN_NORMAL_F64 / 2.0, 4.0 * MIN_NORMAL_F64, 0.0))
    assert len(tells(low)) == 1


def test_silent_on_integer_comparisons():
    """Integers have no subnormals; the band lookup must not invent one."""
    p = propagate(cmp_query("gt", 1.0, 5.0, 0, dtype="int32", aval=I32))
    assert p.obligations[0].status == "discharged"
    assert tells(p) == []


def test_silent_on_a_top_operand():
    """⊤ = [−∞, ∞] contains 0, so the haze is the identity. A ⊤ that
    produced a tell would attach an ℝ-vs-hardware sentence to a box that is
    not about the declaration at all."""
    x, r, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, SUB_LO, SUB_HI),
            ir.JaxprEqn(primitive="mystery_primitive", invars=(x,), outvars=(r,)),
            ir.JaxprEqn(
                primitive="ge",
                invars=(r, ir.Literal(val=float("-inf"), aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    p = propagate(q)
    assert p.obligations[0].status == "discharged"
    assert tells(p) == []


# --------------------------------------------------------------------------
# 3. IT CHANGES NOTHING — the posture is kept, not extended
# --------------------------------------------------------------------------


def test_the_verdict_boxes_and_counts_are_untouched_by_the_tell():
    """The tell writes a note and nothing else. Everything a consumer reads
    off a `Propagation` — statuses, coverage, tiers, assumptions — is what
    it was, which is what "keep the posture" means operationally."""
    q = cmp_query("gt", SUB_LO, SUB_HI, 0.0)
    p = propagate(q)
    assert p.obligations[0].status == "discharged"
    assert (p.coverage.known, p.coverage.total) == (3, 3)
    assert dict(p.transfers_used)["gt"] == P.TIER_EXACT
    # the row itself still returns the ℝ answer, unhazed: definitely true
    eqn = q.jaxpr.eqns[1]
    box = P.TRANSFERS["gt"][0](
        eqn,
        dict(eqn.params_dict()),
        [
            iv.IntervalArray(shape=(), los=(SUB_LO,), his=(SUB_HI,)),
            iv.IntervalArray(shape=(), los=(0.0,), his=(0.0,)),
        ],
    )[0]
    assert (box.los[0], box.his[0]) == iv.BOOL_TRUE


def test_ieee_mode_gains_no_tell_and_keeps_its_own_answer():
    """The ieee dial already models the flush and already returns unknown
    here. It must not ALSO grow the note — the note exists because real mode
    does not model it."""
    p = propagate(cmp_query("gt", SUB_LO, SUB_HI, 0.0), semantics="ieee")
    assert p.obligations[0].status != "discharged"
    assert tells(p) == []


def test_the_tell_is_the_only_thing_the_note_list_gained():
    """No second note, no decline, no ⊤ text: the difference between the
    subnormal query's notes and the ordinary one's is exactly the tell."""
    sub = propagate(cmp_query("gt", SUB_LO, SUB_HI, 0.0))
    ordinary = propagate(cmp_query("gt", 1.0, 2.0, 0.0))
    assert [n for n in sub.notes if not n.startswith(TELL_HEAD)] == list(
        ordinary.notes
    )


# --------------------------------------------------------------------------
# 4. THE ANTI-VACUITY CONTROL — delete the trigger and the tell must vanish
# --------------------------------------------------------------------------


def test_the_trigger_is_load_bearing(monkeypatch):
    """Norm D: with `_subnormal_flush_tell` neutered the tell disappears, so
    these tests are pinning the trigger and not some other sentence that
    happens to contain the same words."""
    assert len(tells(propagate(cmp_query("gt", SUB_LO, SUB_HI, 0.0)))) == 1
    monkeypatch.setattr(P, "_subnormal_flush_tell", lambda *a, **k: None)
    assert tells(propagate(cmp_query("gt", SUB_LO, SUB_HI, 0.0))) == []


# --------------------------------------------------------------------------
# 5. END TO END, THROUGH THE PUBLIC ENTRY POINT
# --------------------------------------------------------------------------


def test_end_to_end_the_verdict_carries_the_tell_and_the_control_does_not():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_
    from stelling.preconditions import check

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        def h_sub():
            x = any_array((), "float64", (SUB_LO, SUB_HI))
            return assert_(x > 0.0)

        def h_normal():
            x = any_array((), "float64", (1.0, 2.0))
            return assert_(x > 0.0)

        v = check(h_sub, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED"
        assert TELL_HEAD in v.render()
        # and the target really does disagree, which is why the tell exists
        assert bool(jnp.asarray(SUB_LO, "float64") > 0.0) is False

        c = check(h_normal, vacuity_mode="inputs-only")
        assert c.status == "VERIFIED"
        assert TELL_HEAD not in c.render()
        assert bool(jnp.asarray(1.5, "float64") > 0.0) is True
    finally:
        jax.config.update("jax_enable_x64", old)
