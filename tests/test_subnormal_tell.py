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

**AND IT IS PER FORMAT, WHICH IS THE LAYER THE FIRST CUT HAD NOTHING FOR.**
The band this file guards was per-dtype from the start; the CLAIM the band
supports was not. Section 2b is the repair: whether the measured target
flushes a format at all is itself a per-format measured fact — on this one
float16 keeps gradual underflow while the other three flush — and the band
applied to an operand must be that operand's OWN, not the widest in the
equation, because over-hazing a HAZE only costs precision while over-hazing
an ASSERTION makes it false. A file containing no float16, no bfloat16 and
no mixed-dtype comparison could not have caught either, whatever its band
mutants said.

Hand-built IR, so the whole file runs in the zero-dep lane; the jax leg at
the bottom is `importorskip`ed and is the end-to-end one, and
`test_the_measured_flush_table_still_matches_the_TARGET` is the one test
here that pins the TABLE against the machine rather than the code against
the table.
"""

from __future__ import annotations

import pytest

from stelling import interval as iv
from stelling import ir
from stelling import propagate as P
from stelling.propagate import propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
F32 = ir.Aval(kind="ShapedArray", shape=(), dtype="float32")
F16 = ir.Aval(kind="ShapedArray", shape=(), dtype="float16")
BF16 = ir.Aval(kind="ShapedArray", shape=(), dtype="bfloat16")
I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")

TELL_HEAD = "SUBNORMAL-SENSITIVE DEFINITE ANSWER"

# float64's smallest positive normal and one subnormal box strictly inside
# the band below it. 1e-300 is a NORMAL float64 — the box REACHES into the
# band rather than sitting wholly inside it, which is the shape the sweep
# measured and the weaker precondition for the tell.
MIN_NORMAL_F64 = 2.0**-1022
SUB_LO, SUB_HI = 1e-320, 1e-300

# The same shape in the three narrow formats. float16's band is by far the
# widest in absolute terms (2**-14 is 6.1e-05), which is exactly why a
# float16 operand hazed at SOMEONE ELSE'S band, or at its own band on a
# target that does not flush it, is the loud failure mode and not a corner.
MIN_NORMAL_F16 = 2.0**-14
MIN_NORMAL_F32 = 2.0**-126  # bfloat16 shares this emin
SUB_F16 = (1e-6, 1e-5)          # strictly inside float16's band
SUB_F32 = (1.4e-45, 1e-40)      # strictly inside float32's band
SUB_BF16 = (1e-40, 1e-39)       # strictly inside bfloat16's band


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


def cmp_query(prim, lo, hi, rhs, *, dtype="float64", aval=F64, rhs_aval=None):
    """``assert_(x <prim> rhs)`` over a declared ``[lo, hi]``.

    ``rhs_aval`` defaults to the declared operand's own aval; passing a
    different one builds the MIXED-dtype comparison that hand-built and
    deserialized IR routinely carries (a narrow value against a wider
    literal, or the reverse).
    """
    x, pred, out = var(0, aval), var(1, BOOL), var(2, BOOL)
    return close(
        [
            any_eqn(x, lo, hi, dtype),
            ir.JaxprEqn(
                primitive=prim,
                invars=(x, ir.Literal(val=rhs, aval=rhs_aval or aval)),
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
# 2b. THE FLUSH IS PER FORMAT, AND SO IS THE BAND — the layer above the band
# --------------------------------------------------------------------------
#
# THE HOLE THESE CLOSE, NAMED. The first cut of this file made the BAND
# per-dtype and guarded it with a mutant, and then had nothing at all in it
# for the layer above: whether the target flushes that format, and whether
# the band applied to an operand is its OWN. It contained zero occurrences of
# float16, of bfloat16, and of a mixed-dtype comparison, so both defects
# shipped green:
#
#   float16 [1e-6, 1e-5] > 0.0   fired, citing 5e-324 — a float64 denormal —
#                                as evidence, while the target reads x > 0 as
#                                True at EVERY subnormal float16 magnitude,
#                                eager and jit (asked of jax itself in
#                                test_the_measured_flush_table_still_matches
#                                _the_TARGET below);
#   gt(float64 [1e-10, 1e-9], float16 0.0)
#                                fired on a NORMAL float64 box 298 decades
#                                clear of its own band, because the band came
#                                from the widest operand.
#
# A mutant for the band text is not a mutant for the claim the text makes.


def test_silent_on_float16_which_this_target_keeps_gradual_underflow_on():
    """THE CASE THE FIRST CUT GOT WRONG. float16's subnormal band is the
    widest of the four in absolute terms, and this target does NOT flush it:
    `x > 0` is True at every subnormal float16 magnitude, eager and jit. A
    tell here would assert a flush that does not happen."""
    assert iv.target_flushes_subnormals("float16") is False
    p = propagate(cmp_query("gt", *SUB_F16, 0.0, dtype="float16", aval=F16))
    assert p.obligations[0].status == "discharged"
    assert tells(p) == []


def test_silent_on_float16_at_the_very_bottom_of_its_band():
    """Not an artefact of where in the band the box sits: the same silence
    one exponent below the smallest normal, which is where a size test would
    be loudest."""
    p = propagate(
        cmp_query(
            "gt", MIN_NORMAL_F16 / 2.0, MIN_NORMAL_F16, 0.0,
            dtype="float16", aval=F16,
        )
    )
    assert p.obligations[0].status == "discharged"
    assert tells(p) == []


def test_bfloat16_and_float32_DO_fire_and_name_their_own_band():
    """The other side of the same table: bfloat16 and float32 are measured to
    flush, so the tell is loud on them — and names 2**-126, never 2**-1022.
    Without this row, silencing float16 by silencing every narrow format
    would pass."""
    for dtype, aval, box in (
        ("bfloat16", BF16, SUB_BF16),
        ("float32", F32, SUB_F32),
    ):
        assert iv.target_flushes_subnormals(dtype) is True
        p = propagate(cmp_query("gt", *box, 0.0, dtype=dtype, aval=aval))
        assert p.obligations[0].status == "discharged"
        assert len(tells(p)) == 1, (dtype, p.notes)
        note = tells(p)[0]
        assert f"{dtype}: 0 < |x| < 2**-126" in note
        assert repr(MIN_NORMAL_F32) in note
        assert "2**-1022" not in note


def test_silent_on_a_MIXED_comparison_where_no_operand_reaches_its_own_band():
    """THE SECOND CASE THE FIRST CUT GOT WRONG. `_ieee_cmp_get_min_normal`
    takes the WIDEST operand band, which is right for the ieee haze (hulling
    wider only costs precision) and wrong for an assertion (hazing wider
    makes it false). A normal float64 box 298 decades clear of 2**-1022,
    compared against a float16 zero, must be silent."""
    p = propagate(
        cmp_query("gt", 1e-10, 1e-9, 0.0, dtype="float64", aval=F64,
                  rhs_aval=F16)
    )
    assert p.obligations[0].status == "discharged"
    assert tells(p) == []
    # the same shape one format down, and the control that the query really
    # is mixed: the widest band in it is float16's, which sits 15.8 decades
    # ABOVE the top of the box, while the box's own float32 band is 7.9
    # decades BELOW its bottom
    q = cmp_query("gt", 1e-30, 1e-20, 0.0, dtype="float32", aval=F32,
                  rhs_aval=F16)
    assert P._ieee_cmp_get_min_normal(q.jaxpr.eqns[1]) == MIN_NORMAL_F16
    assert tells(propagate(q)) == []


def test_a_MIXED_comparison_fires_on_the_operand_that_DOES_reach_its_band():
    """And the control for that control. Same mixed pair, but now the float64
    operand is genuinely inside float64's band: the tell fires, and it names
    float64 — the format that reached its band and that the target flushes —
    and not float16, which supplied neither."""
    p = propagate(
        cmp_query("gt", SUB_LO, SUB_HI, 0.0, dtype="float64", aval=F64,
                  rhs_aval=F16)
    )
    assert p.obligations[0].status == "discharged"
    assert len(tells(p)) == 1
    note = tells(p)[0]
    assert "float64: 0 < |x| < 2**-1022" in note
    assert "float16: 0 < |x| <" not in note


def test_the_band_helper_answers_per_dtype_and_says_nothing_off_the_table():
    """The trigger's band is read off the operand's OWN dtype, and a format
    with no measurement returns None — which is not False, and is a reason to
    stay silent rather than to assume gradual underflow."""
    assert P._subnormal_tell_band("float64") == MIN_NORMAL_F64
    assert P._subnormal_tell_band("float32") == MIN_NORMAL_F32
    assert P._subnormal_tell_band("bfloat16") == MIN_NORMAL_F32
    assert P._subnormal_tell_band("float16") is None  # measured: no flush
    assert P._subnormal_tell_band("int32") is None
    assert P._subnormal_tell_band("float8_e4m3fn") is None
    assert iv.target_flushes_subnormals("float8_e4m3fn") is None


def test_the_flush_SENTENCE_is_the_ieee_stamps_own_sentence():
    """ONE SOURCE OF TRUTH, DRIVEN. The tell's claim about which formats this
    target flushes is the very string the `ieee` stamp carries — not a second
    copy of it that can drift. The first cut wrote its own, hard-coded to
    float64 (*"reads 5e-324 > 0 as False"*), and stamped it on float16 runs
    where the `ieee` face of the SAME run said the opposite, correctly."""
    clause = iv.measured_flush_clause()
    assert "float16 keeps gradual underflow" in clause
    assert "bfloat16/float32/float64 flush" in clause
    note = tells(propagate(cmp_query("gt", SUB_LO, SUB_HI, 0.0)))[0]
    assert clause in note
    stamp = iv.subnormal_indeterminacy_assumption(("float16", "float64"))
    assert clause in stamp
    # and the ONE sentence deliberately kept verbatim — the binary64-only
    # stamp, which is the common run and predates the parametric builders —
    # must still be saying what the table says about binary64.
    assert iv.target_flushes_subnormals("float64") is True
    assert "binary64 flushes subnormals" in iv.SUBNORMAL_INDETERMINACY_ASSUMPTION


def test_the_tell_only_ever_fires_on_a_format_the_table_calls_flushing():
    """The invariant behind every row above, stated once and swept over the
    whole table: whatever the tell names as flushed, the table agrees."""
    boxes = {
        "float16": (SUB_F16, F16),
        "bfloat16": (SUB_BF16, BF16),
        "float32": (SUB_F32, F32),
        "float64": ((SUB_LO, SUB_HI), F64),
    }
    fired = set()
    for dtype, (box, aval) in boxes.items():
        p = propagate(cmp_query("gt", *box, 0.0, dtype=dtype, aval=aval))
        if tells(p):
            fired.add(dtype)
            assert f"{dtype}: 0 < |x| <" in tells(p)[0]
    assert fired == {
        d for d, v in iv._FORMAT_TARGET_FLUSHES.items() if v
    }, fired


def test_the_two_format_tables_describe_the_same_formats():
    """A format with a band and no measurement would be hazed with no
    evidence; a format measured and bandless could not be hazed at all.
    interval.py raises at import if they part company — this is that guard
    said out loud, so deleting it is a test failure and not a silence."""
    assert set(iv._FORMAT_TARGET_FLUSHES) == set(iv._FORMAT_MIN_NORMAL_TEXT)
    assert set(iv._FORMAT_TARGET_FLUSHES) == set(P._FLOAT_FORMATS)


def test_the_measured_flush_table_still_matches_the_TARGET():
    """THE ROOT GUARD, and the one that would have caught this at the source.
    Every other test here pins the code against the table; this one pins the
    TABLE against the machine. For each format, ask jax the question the
    table answers — `x > 0` at magnitudes strictly inside that format's own
    subnormal band, eager AND under jit — and require the answers to agree
    with the entry."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    probes = {
        "float16": [2.0**-24, 2.0**-20, MIN_NORMAL_F16 / 2.0, 1e-6, 1e-5],
        "bfloat16": [1e-40, 1e-39, 2.0**-130],
        "float32": [1.4e-45, 1e-40, 1e-39],
        "float64": [5e-324, 1e-320, 1e-310],
    }
    gt0 = lambda v: v > 0  # noqa: E731
    gt0_jit = jax.jit(gt0)
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        for dtype, vals in probes.items():
            flushes = iv.target_flushes_subnormals(dtype)
            for v in vals:
                a = jnp.asarray(v, dtype)
                # the value really is subnormal in that format: stored, and
                # stored as something other than zero
                assert float(a) != 0.0, (dtype, v)
                assert abs(float(a)) < float(
                    jnp.finfo(jnp.dtype(dtype)).tiny
                ), (dtype, v)
                for reads_positive in (bool(gt0(a)), bool(gt0_jit(a))):
                    assert reads_positive is not flushes, (dtype, v, flushes)
    finally:
        jax.config.update("jax_enable_x64", old)


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

# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
#
# ===========================================================================
# THREE REGRESSION PINS FOR THE B18 FIXUP — APPEND-ONLY SNIPPET
# ===========================================================================
#
# APPEND TARGET, exactly:  tests/test_subnormal_tell.py   (at 958f56e)
# Append verbatim to the END of that file. Nothing above it changes.
#
# NAMES USED: every one is already defined by that file — `iv` (line 48),
# `P` (50), `propagate` (51), `F64`/`F32`/`F16`/`BF16` (53-56),
# `SUB_LO, SUB_HI` (67, a TUPLE UNPACK — `grep '^SUB_HI'` misses it, which
# is why it can look absent), `SUB_F16`/`SUB_F32`/`SUB_BF16` (75-77),
# `cmp_query` (99, whose `rhs_aval=` kwarg the fixup itself added) and
# `tells` (122). NO new helper is needed and nothing here is contorted to
# avoid one. The only literals introduced are the three smallest-positive
# subnormals, and they are deliberately literal: deriving them from
# `P._ieee_format_min_positive(P._FLOAT_FORMATS[d])` would restate the code
# under test and would NOT redden mutant M-C below, which changes that
# call's ARGUMENT rather than the function.
#
# ---------------------------------------------------------------------------
# WHY THESE THREE — the mutants they close, each of which survives the
# shipped 30-test file, in every lane, with ZERO failures.
# ---------------------------------------------------------------------------
#
# M-A  THE FLUSH GATE MOVED FROM PER OPERAND TO PER EQUATION.
#      In `propagate._subnormal_flush_tell`, replace
#          band = _subnormal_tell_band(v.aval.dtype)
#      with a gate hoisted out of the loop:
#          _g = any(iv.target_flushes_subnormals(w.aval.dtype) is True
#                   for w in eqn.invars)
#          band = (_ieee_format_min_normal(_FLOAT_FORMATS[v.aval.dtype])
#                  if (_g and v.aval.dtype in _FLOAT_FORMATS) else None)
#      Bands stay each operand's own; the table is untouched; only the
#      PAIRING of "which operand" with "does THAT operand's format flush"
#      breaks. Measured (line-count-neutral copy, 12142 lines, so the
#      `.py:N` doc test cannot flag it): tests/test_subnormal_tell.py 30
#      passed; FULL nojax lane 2143 passed / 148 skipped / 0 failed; FULL
#      jax lane x64 unset 4366 passed / 10 skipped / 0 failed — bit for bit
#      the fixup's own figures. What it does: `gt(float16 [1e-6,1e-5],
#      float64 0.0)` FIRES, asserting "jax 0.11.0 CPU reads
#      5.960464477539063e-08 > 0 as False", which is measurably FALSE
#      (float16 reads it True, eager and jit, both x64 cells). That is
#      defect F1 restored. The row is corpus/subnormal_tell_renders.py
#      GROUP C row 4, whose prose already says "SILENT" — but `drive_ir()`
#      only prints and `testpaths = ["tests"]`, so nothing asserts it. Both
#      mixed rows in the shipped file put the NARROW format on the literal
#      side, so the float16-in-band x other-operand-flushes corner is the
#      one shape section 2b does not carry.
#
# M-B  THE WITNESS REGRESSED TO float64's 5e-324 ON EVERY FORMAT.
#      In `propagate._subnormal_tell_text`, replace
#          f"{_ieee_format_min_positive(_FLOAT_FORMATS[d])!r} > 0 as False"
#      with `_FLOAT_FORMATS["float64"]` in place of `_FLOAT_FORMATS[d]`.
#      Measured: 30 passed. The bfloat16 note then reads "bfloat16: 0 < |x|
#      < 2**-126 = 1.1754943508222875e-38, and jax 0.11.0 CPU reads 5e-324
#      > 0 as False" — the exact sentence the commit message quotes as the
#      original symptom. It walks through
#      `test_bfloat16_and_float32_DO_fire_and_name_their_own_band` because
#      that test checks the MIN NORMAL (`repr(MIN_NORMAL_F32)`, and
#      `"2**-1022" not in note`) while the witness is a MIN POSITIVE.
#
# M-C  THE FLUSH SENTENCE WRITTEN OUT A SECOND TIME, VERBATIM-CORRECT.
#      In `propagate._subnormal_tell_text`, replace
#          f"PER-FORMAT — {iv.measured_flush_clause()} — so on the formats "
#      with today's exact string spelled out inline. Measured: 30 passed.
#      `test_the_flush_SENTENCE_is_the_ieee_stamps_own_sentence` asserts the
#      two faces carry the SAME STRING, which a correct-today second copy
#      satisfies — so "there is no second spelling left to drift" is a
#      property of today's code, not an invariant the suite maintains.
#
# ---------------------------------------------------------------------------
# ACCEPTANCE, DRIVEN (stelling-jax venv, jax 0.11.0 CPU, python 3.12.3,
# COLUMNS=200, PYTHONPATH pinned absolute, __pycache__ purged per run):
#
#   clean 958f56e + these three pins .............. 33 passed
#   M-A  + these three pins ....................... 1 failed, 32 passed
#         -> test_the_flush_gate_is_PER_OPERAND_not_per_equation
#   M-B  + these three pins ....................... 1 failed, 32 passed
#         -> test_the_witness_is_each_formats_OWN_smallest_subnormal
#   M-C  + these three pins ....................... 1 failed, 32 passed
#         -> test_the_flush_SENTENCE_is_DERIVED_and_not_a_second_copy
#
#   Each mutant reddens exactly one pin, and the right one.
#
# ---------------------------------------------------------------------------
# ONE CONSTRAINT WORTH KNOWING, because it will look like a bug otherwise:
# the derivation pin MUST call `subnormal_indeterminacy_assumption` with a
# TWO-FORMAT tuple. `subnormal_indeterminacy_assumption(())` and
# `(("float64",))` short-circuit to the hard-coded
# `SUBNORMAL_INDETERMINACY_ASSUMPTION` constant, which is a genuine second
# spelling deliberately kept verbatim (the fixup's own test calls it "the
# ONE sentence deliberately kept verbatim"). Passing `("float64",)` there
# makes the pin fail on the CLEAN tree, for a true reason that is not the
# one the pin is about.
# ===========================================================================


# --------------------------------------------------------------------------
# 2c. THE PAIRING AND THE CLAIM — the layer the mutants walked through
# --------------------------------------------------------------------------
#
# Section 2b pins WHICH FORMATS the tell fires on. These three pin the two
# things that survived it: that the flush gate is asked of the OPERAND
# rather than of the equation, and that the sentence the note then makes is
# DERIVED rather than restated. A mutant for which formats fire is not a
# mutant for what the firing note asserts about them.


def test_the_flush_gate_is_PER_OPERAND_not_per_equation():
    """corpus/subnormal_tell_renders.py GROUP C row 4, ASSERTED rather than
    printed. The float16 operand is deep inside float16's own band, and the
    OTHER operand's format (float64) is one this target does flush. A gate
    asked of the EQUATION opens here and hazes the float16 operand at its
    own band, producing a note that asserts a flush this target does not
    perform — defect F1, restored, and green on all thirty tests above.

    The control is the row directly below it: the same mixed pair with the
    float64 operand inside ITS band still fires, so this is not "mixed rows
    are silent"."""
    p = propagate(
        cmp_query("gt", *SUB_F16, 0.0, dtype="float16", aval=F16, rhs_aval=F64)
    )
    assert p.obligations[0].status == "discharged"
    assert tells(p) == [], p.notes
    # the control, and it is what makes the silence above a PAIRING claim
    q = propagate(
        cmp_query("gt", SUB_LO, SUB_HI, 0.0, dtype="float64", aval=F64,
                  rhs_aval=F16)
    )
    assert len(tells(q)) == 1
    assert "float64: 0 < |x| <" in tells(q)[0]


def test_the_witness_is_each_formats_OWN_smallest_subnormal():
    """The note's *"reads X > 0 as False"* must cite the smallest subnormal
    of the format that reached its band — not float64's 5e-324, which is
    the evidence the defect stamped on every run regardless of format.

    Stated as an EXCLUSION as well as an inclusion, because "contains my own
    witness" alone is satisfied by a note that contains all four. float64 is
    in the sweep as the control: its own witness IS 5e-324, so this pin
    forbids a string only where that string is the wrong format's."""
    # each format's 2**(emin - p + 1), written out rather than derived from
    # `P._ieee_format_min_positive`, which is the function under test
    min_positive = {
        "bfloat16": 2.0**-133,   # emin -126, p 8   -> 9.183549615799121e-41
        "float32": 2.0**-149,    # emin -126, p 24  -> 1.401298464324817e-45
        "float64": 2.0**-1074,   # emin -1022, p 53 -> 5e-324
    }
    for dtype, aval, box in (
        ("bfloat16", BF16, SUB_BF16),
        ("float32", F32, SUB_F32),
        ("float64", F64, (SUB_LO, SUB_HI)),
    ):
        assert iv.target_flushes_subnormals(dtype) is True
        note = tells(
            propagate(cmp_query("gt", *box, 0.0, dtype=dtype, aval=aval))
        )[0]
        assert f"reads {min_positive[dtype]!r} > 0 as False" in note, note
        for other, mp in min_positive.items():
            if other != dtype:
                assert f"reads {mp!r} > 0 as False" not in note, (dtype, other, note)


def test_the_flush_SENTENCE_is_DERIVED_and_not_a_second_copy(monkeypatch):
    """Move the TABLE and every face must move with it. The test above this
    one in section 2b asserts the tell and the `ieee` stamp carry the SAME
    STRING — which a second copy that is correct today satisfies, and a
    second copy that is correct today is exactly how this defect shipped.
    This one asserts the string is DERIVED: change the table under it and
    both faces follow, or the pin fails.

    `assert clause != before` is the anti-vacuity guard: if the builder
    itself were hard-coded, the pin would otherwise pass by comparing a
    constant to itself.

    The stamp is asked for a TWO-FORMAT tuple deliberately — see the header:
    `("float64",)` short-circuits to the verbatim constant, which is a
    second spelling this project keeps on purpose."""
    before = iv.measured_flush_clause()
    monkeypatch.setitem(iv._FORMAT_TARGET_FLUSHES, "bfloat16", False)
    clause = iv.measured_flush_clause()
    assert clause != before, (
        "measured_flush_clause() did not move with _FORMAT_TARGET_FLUSHES, "
        "so it is not derived from it and this pin proves nothing"
    )
    note = tells(propagate(cmp_query("gt", SUB_LO, SUB_HI, 0.0)))[0]
    assert clause in note, note
    stamp = iv.subnormal_indeterminacy_assumption(("float16", "float64"))
    assert clause in stamp, stamp
