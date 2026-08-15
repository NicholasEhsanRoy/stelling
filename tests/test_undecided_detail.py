# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The undecided obligation detail quotes the straddle it was judged on.

docs/proposed-decline-messages.md #1: "unknown — undecided for N/N
element(s)" was rated 1/10 by two independent external agents ("this
message told me nothing"), and the single highest-value line is the
operand's interval, which the judgment already holds. The detail now
quotes, at the ONE place the judgment is made (the propagator's assert
site, so every front door inherits it — the one-pipeline principle):

* the operand's propagated span and the asserted bound, normalized so
  the quote always reads "operand <cmp> bound";
* the failing endpoint's exact miss distance (Fraction-exact, marked ≈
  only when its float rendering rounds) with an exact ulp-step count
  when small — the one-ulp-miss vs wide-dependency-box distinction an
  external evaluator said the old message could not give;
* for a strict bound whose endpoint sits ON the bound: that fact, in
  place of a zero miss (the exactly-stated-threshold shape);
* the artifact-⊤ origin of a quoted side, from the same top_origin
  record the decline notes use;
* NOTHING where no honest quote exists — a non-comparison operand keeps
  the old detail byte-identically.

The verdict assemblers append one coverage-cause note per verdict with
undecided obligation(s), claiming only what the coverage instrument
measured (complete coverage does NOT claim the property holds — a
straddle is compatible with both a near-miss and a genuine violation,
and the note says so).

Message content only: statuses, coverage, and accounting pinned per path.
Hand-built IR — no jax needed.
"""

from __future__ import annotations

import math

from stelling import ir
from stelling.propagate import interval_env, propagate
from stelling.verdict import make_verdict, undecided_cause_note

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
F64_8 = ir.Aval(kind="ShapedArray", shape=(8,), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")
BOOL_8 = ir.Aval(kind="ShapedArray", shape=(8,), dtype="bool")

_VERSIONS = dict(
    stelling_version="test", jax_version="none", precision_config="n/a"
)


def var(i, a=F64):
    return ir.Var(id=i, aval=a)


def lit(v, a=F64):
    return ir.Literal(val=v, aval=a)


def any_eqn(out, lo, hi, src=()):
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
        source_info=src,
    )


def eqn(prim, ins, out, params=(), src=()):
    return ir.JaxprEqn(
        primitive=prim,
        invars=tuple(ins),
        outvars=(out,),
        params=tuple(params),
        source_info=src,
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns)
        )
    )


_PAD_BOUND = 1.0


def _padded_query(cmp, point_side):
    """exp(x) over x in [0, 4] against the bound 1.0 — a bracket-padded
    endpoint one ulp on the wrong side of the bound.

    **This exhibit was `0.5 * x` over `[0, 4]` against `0.0`** (the proposal's
    own), where outward rounding turned the exact 0 endpoint into `-5e-324`.
    Audit 0.2.0 M16 gave `mul` the exact-rational route `add` and `div`
    already had, so that box is now exactly `[0.0, 2.0]`, the obligation
    `>= 0.0` DISCHARGES, and the sentence under test was no longer reachable
    through it. Nothing about the sentence changed — the subject moved to a
    transfer that still brackets, and must: `exp` is irrational at every
    nonzero rational argument, so its endpoints can only ever be bracketed
    (`exp([0, 4])` -> `[0.9999999999999999, 54.59815003314424]`, whose true
    infimum is exactly 1). The shape is identical: the failing endpoint
    misses the bound by exactly one ulp at its magnitude.
    """
    x, m, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    sides = (
        [lit(_PAD_BOUND), m] if point_side == "left" else [m, lit(_PAD_BOUND)]
    )
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("exp", [x], m),
            eqn(cmp, sides, pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    return q, propagate(q)


def test_the_old_mul_exhibit_now_decides_and_that_is_why_the_subject_moved():
    """The exhibit that moved, kept as the record of why: `0.5 * x` over
    `[0, 4]` is now the exact `[0.0, 2.0]` and `>= 0.0` is definitely true.
    Before audit 0.2.0 M16 the box was `[-5e-324, 2.0000000000000004]` and
    this obligation was undecided by one denormal ulp."""
    x, m, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("mul", [x, lit(0.5)], m),
            eqn("ge", [m, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    box = interval_env(q)[1]
    assert (box.los[0], box.his[0]) == (0.0, 2.0)
    assert p.obligations[0].status == "discharged"


# -- the point-bound scalar form ----------------------------------------------


def test_scalar_span_bound_and_one_ulp_miss_all_measured():
    q, p = _padded_query("ge", "right")
    detail = p.obligations[0].detail
    # the pre-change sentence survives as the prefix — same count, same n
    assert detail.startswith("undecided for 1/1 element(s)")
    # the quoted span IS the judged box, read back from the live env
    box = interval_env(q)[1]
    assert box.los[0] < _PAD_BOUND < box.his[0]
    assert f"the operand spans [{box.los[0]}, {box.his[0]}]" in detail
    assert "the asserted bound is operand >= 1.0" in detail
    # the miss is the exact distance from the FAILING (lower) endpoint to
    # the bound: one ulp, and both the number and the ulp count are
    # measured, not narrated
    miss = _PAD_BOUND - box.los[0]
    assert miss == 1.1102230246251565e-16
    assert math.nextafter(box.los[0], _PAD_BOUND) == _PAD_BOUND  # one step
    assert (
        f"the operand's lower endpoint misses the bound by {miss} "
        "(1 ulp step at this magnitude)" in detail
    )
    # message content only
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 0 and p.coverage.known == 4


def test_point_on_the_left_normalizes_to_operand_cmp_bound():
    # assert 1.0 < exp(x)  ===  operand > 1.0: the quote reads in operand
    # terms, with the direction preserved by the flip
    _, p = _padded_query("lt", "left")
    detail = p.obligations[0].detail
    assert p.obligations[0].status == "unknown"
    assert "the asserted bound is operand > 1.0" in detail
    assert (
        "the operand's lower endpoint misses the bound by "
        "1.1102230246251565e-16" in detail
    )


def test_strict_bound_with_endpoint_on_the_bound_says_so():
    # x in [0, 4] against x > 0: the lower endpoint EQUALS the bound —
    # the exactly-stated-threshold shape, stated instead of a zero miss
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    detail = p.obligations[0].detail
    assert p.obligations[0].status == "unknown"
    assert (
        "the operand's lower endpoint equals the bound, which strict > "
        "does not admit" in detail
    )
    assert "misses the bound" not in detail
    # the mirrored upper case: x in [0, 4] against x < 4
    x2, pred2, out2 = var(0), var(1, BOOL), var(2, BOOL)
    q2 = close(
        [
            any_eqn(x2, 0.0, 4.0),
            eqn("lt", [x2, lit(4.0)], pred2),
            eqn("stelling_assert", [pred2], out2),
        ],
        [out2],
    )
    p2 = propagate(q2)
    assert (
        "the operand's upper endpoint equals the bound, which strict < "
        "does not admit" in p2.obligations[0].detail
    )


def test_both_sides_vary_quotes_the_straddle_in_order():
    a, b, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(a, 0.0, 1.0),
            any_eqn(b, 0.5, 2.0),
            eqn("le", [a, b], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    detail = p.obligations[0].detail
    assert p.obligations[0].status == "unknown"
    # lhs and rhs are quoted on their own sides, bound to their declared
    # envelopes — a swap is a false quote and fails here
    assert (
        "the comparison straddles: lhs in [0.0, 1.0] <= rhs in [0.5, 2.0]"
        in detail
    )
    assert "misses the bound" not in detail  # no per-side miss is claimed


def test_array_operand_quotes_the_hull_and_claims_no_miss():
    x, pred, out = var(0, F64_8), var(1, BOOL_8), var(2, BOOL)
    q = close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("ge", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    detail = p.obligations[0].detail
    assert detail.startswith("undecided for 8/8 element(s)")
    assert "the operand spans hull [-1.0, 1.0] (8 elements)" in detail
    assert "the asserted bound is operand >= 0.0" in detail
    # per-element miss is a scalar claim; the hull form does not fake one
    assert "misses the bound" not in detail


def test_non_comparison_operand_keeps_the_bare_detail_byte_identically():
    # the no-guess path: a declared boolean asserted directly has no
    # comparison to quote, and the pre-change detail stands alone
    b_aval = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")
    b, out = var(0, b_aval), var(1, BOOL)
    q = close(
        [
            any_eqn(b, 0.0, 1.0),
            eqn("stelling_assert", [b], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"
    assert p.obligations[0].detail == "undecided for 1/1 element(s)"


def test_artifact_top_side_names_its_origin_in_the_quote():
    x, t, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("sin", [x], t, src=("s.py:9 (f)",)),
            eqn("ge", [t, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    detail = p.obligations[0].detail
    assert p.obligations[0].status == "unknown"
    assert "the operand spans [-inf, inf]" in detail
    # the unbounded side is stelling's own artifact and the quote says so
    # — the #2 provenance rule applied to the straddle quote
    assert (
        "lhs is stelling's own ⊤ from 'sin' at s.py:9 (f) (no interval "
        "transfer is registered for it), not a declaration-derived range"
        in detail
    )
    # and no miss-distance claim rides on an infinite endpoint
    assert "misses the bound" not in detail
    assert p.coverage.unknown == 1


# -- totality of the miss fragment (blinded-lens audit R1) --------------------


def test_out_of_range_miss_states_the_class_not_a_number():
    # x in [-1e308, 1.5e308] against x >= 1e308: the EXACT miss is
    # 2e308 — beyond the binary64 range, where float(Fraction) raises.
    # The fragment must stay total and state the true class instead.
    from fractions import Fraction

    from stelling.propagate import _MAX_BINARY64

    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, -1e308, 1.5e308),
            eqn("ge", [x, lit(1e308)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)  # must not raise
    detail = p.obligations[0].detail
    assert p.obligations[0].status == "unknown"
    # the class claim is measured: the exact miss really exceeds the range
    assert Fraction(1e308) - Fraction(-1e308) > _MAX_BINARY64
    assert float(_MAX_BINARY64) == 1.7976931348623157e308
    assert (
        "the operand's lower endpoint misses the bound by more than the "
        "largest finite binary64 value (1.7976931348623157e+308)" in detail
    )
    # no fabricated number rides on this class
    assert "misses the bound by 2" not in detail
    assert p.coverage.unknown == 0 and p.coverage.known == 3


def test_out_of_range_miss_reproducer_through_check():
    # the audited reproducer, end to end through the front door: the tip
    # under audit CRASHED here (OverflowError) where the parent returned
    # UNKNOWN — the fragment must never be the thing that kills a verdict
    import pytest

    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from stelling.harness import any_array, assert_
        from stelling.preconditions import check

        def h():
            x = any_array((), "float64", (-1e308, 1.5e308))
            return assert_(x >= 1e308)

        v = check(h, vacuity_mode="inputs-only")
    finally:
        jax.config.update("jax_enable_x64", old)
    assert v.status == "UNKNOWN"
    assert (
        "misses the bound by more than the largest finite binary64 value"
        in v.obligations[0].detail
    )


# -- a declined declaration is pointed AT, not away from (audit R4) -----------


def test_declined_declaration_side_points_at_the_declaration():
    from math import nan

    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, nan, 4.0),
            eqn("ge", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    # give the declaration a site so the pointer is usable
    decl = q.jaxpr.eqns[0]
    q = close(
        [
            ir.JaxprEqn(
                primitive=decl.primitive,
                invars=decl.invars,
                outvars=decl.outvars,
                params=decl.params,
                source_info=("decl.py:3 (h)",),
            ),
            *q.jaxpr.eqns[1:],
        ],
        [out],
    )
    p = propagate(q)
    detail = p.obligations[0].detail
    assert p.obligations[0].status == "unknown"
    # the side's ⊤ was minted at its OWN declaration: the quote must say
    # so, mirroring _operand_provenance's branch...
    assert (
        "lhs is ⊤ because its own declaration declined at decl.py:3 (h) "
        "(its interval transfer declined this form)" in detail
    )
    # ...and the away-pointing sentence must not appear for it
    assert "not a declaration-derived range" not in detail
    assert p.coverage.unknown == 1  # the declined declaration, counted


# -- the coverage-cause note --------------------------------------------------


def test_cause_note_complete_coverage_counts_are_the_instruments_own():
    q, p = _padded_query("ge", "right")
    v = make_verdict(q, p, **_VERSIONS)
    assert v.status == "UNKNOWN"
    notes = [n for n in v.notes if "transfer coverage is not the cause" in n]
    assert len(notes) == 1
    (note,) = notes
    # numbers bound to the instrument, not narrated
    assert p.coverage.unknown == 0 and p.coverage.inert == 0
    assert (
        f"{p.coverage.known}/{p.coverage.total} equations ran a "
        f"registered transfer" in note
    )
    # complete coverage does NOT claim the property holds: both readings
    # are named, because interval arithmetic cannot tell them apart
    assert "precision near-miss" in note and "genuine violation" in note
    assert "cannot tell which" in note


def test_cause_note_gap_branch_names_the_top_primitives():
    x, t, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("sin", [x], t),
            eqn("ge", [t, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    v = make_verdict(q, p, **_VERSIONS)
    notes = [n for n in v.notes if "coverage gaps in the query" in n]
    assert len(notes) == 1
    (note,) = notes
    assert p.coverage.unknown == 1
    assert "1 equation(s) fell to ⊤ (sin ×1)" in note
    assert "may be downstream" in note
    # the complete-coverage sentence must not appear on the gap branch
    assert not any("transfer coverage is not the cause" in n for n in v.notes)


def test_cause_note_absent_when_nothing_is_undecided():
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("ge", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    v = make_verdict(q, p, **_VERSIONS)
    assert v.status == "VERIFIED"
    assert undecided_cause_note(p.coverage, p.obligations) == ()
    assert not any("undecided obligation(s)" in n for n in v.notes)


def test_cause_note_rides_the_solver_assembly_path_too():
    from stelling.solvers import SolverConfig, escalate, make_solver_verdict

    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, -1.0, 4.0),
            eqn("ge", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    esc = escalate(q, p, SolverConfig(timeout_ms=100))  # ieee: refusal-shaped
    v = make_solver_verdict(q, p, esc, **_VERSIONS)
    assert v.status == "UNKNOWN"
    notes = [n for n in v.notes if "transfer coverage is not the cause" in n]
    assert len(notes) == 1
