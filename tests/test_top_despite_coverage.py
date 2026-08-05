# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The stamp field that says what the coverage line did NOT establish.

`unknown = 0` on the coverage line means every primitive in the query
has a registered transfer. It reads as "the analysis saw all of it",
and it is not that: a registered transfer can run and return ⊤ on the
values it is handed, and no count on that line moves when it does.

Both queries below are the SAME five equations — declaration, `exp`,
`sub`, `lt`, `assert` — with 100% census coverage in both. They differ
only in the declared bounds. Over [-1000, 1000] the `exp` overflows to
[0, inf] and the `sub` propagates [-inf, inf]; over [1, 2] nothing
widens. The field appears on the first and must be ABSENT on the
second: a disclosure that always fires discloses nothing.

Beyond those two, one harness per unpinned number or branch: a
three-⊤-boxes-at-two-primitives query for the published TOTAL (which is a
sum, and reads the same as every wrong derivation of it when there is
only one box), a `shape=(0,)` declaration for the `box.size > 0` guard in
`_is_top`, and one query per suppressing conjunct of the gate — a dropped
constraint (`inert`) and an untaken `cond` branch (`unreached`) — each
carrying a real ⊤ that must NOT be disclosed.

Hand-built IR, no jax — same construction as tests/test_propagate.py.
"""

from __future__ import annotations

from stelling import interval as iv
from stelling import ir
from stelling.propagate import _is_top, propagate
from stelling.solvers import Escalation, make_solver_verdict
from stelling.verdict import make_verdict

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")
I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")
EMPTY_F64 = ir.Aval(kind="ShapedArray", shape=(0,), dtype="float64")
EMPTY_BOOL = ir.Aval(kind="ShapedArray", shape=(0,), dtype="bool")

VERSIONS = dict(
    stelling_version="test",
    jax_version="none: hand-built IR",
    precision_config="jax_enable_x64=True (hand-built f64 IR)",
)


def var(i, aval=F64):
    return ir.Var(id=i, aval=aval)


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=outvars, eqns=tuple(eqns))
    )


def any_eqn(out, lo, hi, *, dtype="float64", shape=()):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", shape), ("dtype", dtype), ("lo", lo), ("hi", hi)),
    )


def lt_eqn(operand, out, bound=1.0):
    return ir.JaxprEqn(
        primitive="lt",
        invars=(operand, ir.Literal(val=bound, aval=F64)),
        outvars=(out,),
    )


def assert_eqn(pred, out):
    return ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,))


def cancelling_exp_harness(lo: float, hi: float) -> ir.ClosedJaxpr:
    """x ∈ [lo, hi] ⊢ exp(x) - exp(x) < 1.

    Every primitive is registered, so the census reads 5/5 known either
    way. Whether the propagated `sub` is ⊤ is a fact about the VALUES,
    which is the fact the census cannot hold.
    """
    x, ex, d, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    return ir.ClosedJaxpr(
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
                        ("shape", ()), ("dtype", "float64"),
                        ("lo", lo), ("hi", hi),
                    ),
                ),
                ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
                ir.JaxprEqn(primitive="sub", invars=(ex, ex), outvars=(d,)),
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(d, ir.Literal(val=1.0, aval=F64)),
                    outvars=(pred,),
                ),
                ir.JaxprEqn(
                    primitive="stelling_assert", invars=(pred,), outvars=(out,)
                ),
            ),
        )
    )


def unregistered_harness() -> ir.ClosedJaxpr:
    """A ⊤ the census DOES see: an unregistered primitive, counted."""
    x, u, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return ir.ClosedJaxpr(
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
                        ("shape", ()), ("dtype", "float64"),
                        ("lo", 1.0), ("hi", 2.0),
                    ),
                ),
                ir.JaxprEqn(
                    primitive="no_such_primitive", invars=(x,), outvars=(u,)
                ),
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(u, ir.Literal(val=1.0, aval=F64)),
                    outvars=(pred,),
                ),
                ir.JaxprEqn(
                    primitive="stelling_assert", invars=(pred,), outvars=(out,)
                ),
            ),
        )
    )


def two_cancellations_harness() -> ir.ClosedJaxpr:
    """THREE ⊤ boxes across TWO primitives: `sub ×2`, `add ×1`.

    The published total is a sum over the per-primitive tail, and on the
    single-⊤ harness above every plausible mis-derivation of it — count
    of boxes, count of distinct primitives, a hardcoded 1 — agrees at 1.
    Here they disagree (3 boxes, 2 primitives), which is the only reason
    the total is observable at all.
    """
    x, ex, d1, d2, s, pred, out = (
        var(0), var(1), var(2), var(3), var(4), var(5, BOOL), var(6, BOOL)
    )
    return close(
        [
            any_eqn(x, -1000.0, 1000.0),
            ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
            ir.JaxprEqn(primitive="sub", invars=(ex, ex), outvars=(d1,)),
            ir.JaxprEqn(primitive="sub", invars=(ex, ex), outvars=(d2,)),
            ir.JaxprEqn(primitive="add", invars=(d1, d2), outvars=(s,)),
            lt_eqn(s, pred),
            assert_eqn(pred, out),
        ],
        (out,),
    )


def empty_declaration_harness() -> ir.ClosedJaxpr:
    """A `shape=(0,)` declaration over [1, 2], and nothing widens.

    `exp` of an empty array is an empty array; so is the comparison. Every
    box in this query has size 0 and every one of them is bounded — there
    is no ⊤ here and nothing for the field to disclose.
    """
    x, ex, pred, out = (
        var(0, EMPTY_F64), var(1, EMPTY_F64),
        var(2, EMPTY_BOOL), var(3, EMPTY_BOOL),
    )
    return close(
        [
            any_eqn(x, 1.0, 2.0, shape=(0,)),
            ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
            lt_eqn(ex, pred, bound=8.0),
            assert_eqn(pred, out),
        ],
        (out,),
    )


def dropped_constraint_harness() -> ir.ClosedJaxpr:
    """A real ⊤, and a census gap in the `inert` column ONLY.

    The assume's predicate is produced by `ne`, which is not one of the
    narrowing comparisons (`propagate._ASSUME_CMPS`), so the constraint is
    dropped in the DEFAULT constrain mode — no `assume_mode` argument
    needed, and nothing about the ⊤ changes. `unknown` and `unreached`
    stay at zero, so this reaches the gate through the third conjunct and
    only the third.
    """
    x, ex, d, q, aq, pred, out = (
        var(0), var(1), var(2), var(3, BOOL), var(4, BOOL),
        var(5, BOOL), var(6, BOOL),
    )
    return close(
        [
            any_eqn(x, -1000.0, 1000.0),
            ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
            ir.JaxprEqn(primitive="sub", invars=(ex, ex), outvars=(d,)),
            ir.JaxprEqn(
                primitive="ne",
                invars=(x, ir.Literal(val=0.0, aval=F64)),
                outvars=(q,),
            ),
            ir.JaxprEqn(primitive="stelling_assume", invars=(q,), outvars=(aq,)),
            lt_eqn(d, pred),
            assert_eqn(pred, out),
        ],
        (out,),
    )


def unreached_branch_harness() -> ir.ClosedJaxpr:
    """A real ⊤, and a census gap in the `unreached` column ONLY.

    A `cond` whose index is pinned to 0: branch 1 is not selectable, so
    its equation was never analysed and counts unreached, while the
    `cond` itself counts KNOWN — that is the one path that moves
    `unreached` without also moving `unknown` (`propagate` records the
    two together on every decline). The ⊤ from the cancellation flows
    through the taken branch.
    """
    idx, x, ex, d, y, pred, out = (
        var(0, I32), var(1), var(2), var(3), var(4), var(5, BOOL), var(6, BOOL)
    )
    taken = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(var(10),),
            outvars=(var(11),),
            eqns=(
                ir.JaxprEqn(
                    primitive="add",
                    invars=(var(10), ir.Literal(val=0.0, aval=F64)),
                    outvars=(var(11),),
                ),
            ),
        )
    )
    untaken = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(var(20),),
            outvars=(var(21),),
            eqns=(
                ir.JaxprEqn(
                    primitive="mystery_op", invars=(var(20),), outvars=(var(21),)
                ),
            ),
        )
    )
    return close(
        [
            any_eqn(idx, 0.0, 0.0, dtype="int32"),
            any_eqn(x, -1000.0, 1000.0),
            ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
            ir.JaxprEqn(primitive="sub", invars=(ex, ex), outvars=(d,)),
            ir.JaxprEqn(
                primitive="cond",
                invars=(idx, d),
                outvars=(y,),
                params=(("branches", (taken, untaken)),),
            ),
            lt_eqn(y, pred),
            assert_eqn(pred, out),
        ],
        (out,),
    )


def both_stamps(query):
    """The stamp from EACH assembly site, for the same propagation.

    The two published surfaces are assembled in different modules
    (verdict.make_verdict, solvers.make_solver_verdict) and a field on
    one and not the other is a surface that disagrees with itself — so
    every assertion below is made against both.
    """
    p = propagate(query)
    interval_only = make_verdict(query, p, **VERSIONS)
    escalated = make_solver_verdict(query, p, Escalation(records=()), **VERSIONS)
    return p, interval_only.stamp, escalated.stamp


def test_top_with_a_gapless_census_is_disclosed_at_both_sites():
    p, plain, solved = both_stamps(cancelling_exp_harness(-1000.0, 1000.0))

    # the census: no gap of its own — nothing fell to ⊤, nothing unreached
    assert (p.coverage.unknown, p.coverage.unreached, p.coverage.inert) == (0, 0, 0)
    assert p.coverage.summary() == "5 eqns: 5 known (100%)"
    # ...and the walk still produced ⊤, at the `sub` that cancelled
    assert p.top_boxes == (("sub", 1),)

    for stamp in (plain, solved):
        field = stamp.top_despite_coverage
        assert field is not None
        # THE INVERSION: it states what was not established, and names
        # the ⊤ and the census's own zero in the same breath
        assert field.startswith("NOT ESTABLISHED — that the coverage line "
                                "bounded this query")
        assert "⊤" in field and "sub ×1" in field
        assert "5/5" in field
        # the LEADING TOTAL, and exactly one box produced it. Everything
        # downstream of this number was pinned before it was: the
        # per-primitive tail and the census figures both read correctly
        # while the total said anything at all.
        assert "1 propagated value(s) came out ⊤" in field
        # ...and never the reassurance. No wording here may assert that
        # coverage is complete, sufficient, or correct.
        low = field.lower()
        for banned in ("complete", "fully covered", "correct", "sound"):
            assert banned not in low, f"the field asserts {banned!r}: {field}"

    # one derivation, so the two surfaces cannot drift apart
    assert plain.top_despite_coverage == solved.top_despite_coverage
    for stamp in (plain, solved):
        assert stamp.top_despite_coverage in stamp.render()


def test_full_coverage_without_a_top_leaves_the_field_absent():
    # ANTI-VACUITY. Same five equations, same 100% census, bounded
    # declaration: nothing widened, so there is nothing to disclose and
    # the field must not appear. Without this, a field that always
    # appeared would pass the test above.
    p, plain, solved = both_stamps(cancelling_exp_harness(1.0, 2.0))

    assert p.coverage.summary() == "5 eqns: 5 known (100%)"
    assert p.top_boxes == ()
    assert plain.top_despite_coverage is None
    assert solved.top_despite_coverage is None
    for stamp in (plain, solved):
        assert "NOT ESTABLISHED" not in stamp.render()


def test_a_top_the_census_already_counted_is_not_restated():
    # the field is for the ⊤ the coverage line CANNOT report. When the
    # census counted the ⊤ itself, the coverage line already carries the
    # disclosure and this field would only repeat it under a
    # stronger-sounding name.
    p, plain, solved = both_stamps(unregistered_harness())

    assert p.coverage.unknown == 1
    assert p.top_boxes  # a ⊤ was propagated
    assert plain.top_despite_coverage is None
    assert solved.top_despite_coverage is None


def test_the_published_total_counts_BOXES_not_primitives():
    # THE COUNT, pinned where the readings disagree. `sum(k for _, k in
    # top_boxes)` and `len(top_boxes)` are both 1 on every other harness
    # in this file, so neither the tail nor the census figures constrain
    # the total there. Here the sentence is a claim about three values
    # and only one derivation produces it.
    p, plain, solved = both_stamps(two_cancellations_harness())

    assert p.coverage.summary() == "7 eqns: 7 known (100%)"
    assert (p.coverage.unknown, p.coverage.unreached, p.coverage.inert) == (0, 0, 0)
    assert p.top_boxes == (("sub", 2), ("add", 1))

    for stamp in (plain, solved):
        field = stamp.top_despite_coverage
        assert field is not None
        assert "3 propagated value(s) came out ⊤" in field
        assert "at sub ×2, add ×1" in field
        # the two wrong readings the right one is indistinguishable from
        # everywhere else: the number of DISTINCT primitives, and a total
        # that never moves off the single-box case
        assert "2 propagated value(s)" not in field
        assert "1 propagated value(s)" not in field


def test_a_zero_size_box_is_not_top_so_nothing_is_disclosed():
    # THE `box.size > 0` GUARD in propagate._is_top, pinned end to end.
    # "⊤ on every element" is an all-quantifier, and over no elements it
    # is vacuously true — so without the guard every empty array reads as
    # a total loss of information and this query, where NOTHING widened,
    # publishes a ⊤ disclosure. A disclosure that fires on a bounded
    # query is the one failure mode this field cannot have: it is the
    # false claim the whole surface exists to avoid making.
    assert not _is_top(iv.from_bounds((0,), 1.0, 2.0))
    assert _is_top(iv.from_bounds((1,), float("-inf"), float("inf")))

    p, plain, solved = both_stamps(empty_declaration_harness())

    assert p.coverage.summary() == "4 eqns: 4 known (100%)"
    assert (p.coverage.unknown, p.coverage.unreached, p.coverage.inert) == (0, 0, 0)
    assert p.top_boxes == ()  # the census is gapless AND nothing is ⊤
    for stamp in (plain, solved):
        assert stamp.top_despite_coverage is None
        assert "coverage-not-established:" not in stamp.render()


def test_a_dropped_constraint_suppresses_the_disclosure():
    # THE GATE'S `inert` CONJUNCT. The census reports a gap of its own —
    # in this column and no other — so the coverage line already carries
    # the disclosure, and this field would restate it under a
    # stronger-sounding name.
    p, plain, solved = both_stamps(dropped_constraint_harness())

    assert (p.coverage.unknown, p.coverage.unreached, p.coverage.inert) == (0, 0, 1)
    assert p.coverage.summary() == (
        "7 eqns: 6 known (86%); 1 constraint(s) DROPPED (stelling_assume ×1)"
    )
    assert p.top_boxes == (("sub", 1),)  # and the ⊤ is real, not incidental

    for stamp in (plain, solved):
        assert stamp.top_despite_coverage is None
        assert "coverage-not-established:" not in stamp.render()


def test_an_unreached_equation_suppresses_the_disclosure():
    # THE GATE'S `unreached` CONJUNCT — the same argument as above, in
    # the column that moves when equations were never analysed at all.
    p, plain, solved = both_stamps(unreached_branch_harness())

    assert (p.coverage.unknown, p.coverage.unreached, p.coverage.inert) == (0, 1, 0)
    assert p.coverage.summary() == "9 eqns: 8 known (89%); 1 unreached"
    assert p.top_boxes == (("cond", 1), ("sub", 1))

    for stamp in (plain, solved):
        assert stamp.top_despite_coverage is None
        assert "coverage-not-established:" not in stamp.render()


def test_the_disclosure_renders_directly_under_the_line_it_qualifies():
    # THE PREFIX AND THE POSITION. No consumer keys on
    # `coverage-not-established:` today, which is why renaming it to
    # `coverage:` breaks nothing and produces a render carrying TWO
    # `coverage:` lines — the exact shape a future consumer would key on
    # and get wrong. One line per prefix, pinned before that consumer
    # exists.
    _, plain, solved = both_stamps(cancelling_exp_harness(-1000.0, 1000.0))

    for stamp in (plain, solved):
        lines = stamp.render().split("\n")
        qualified = [i for i, ln in enumerate(lines) if ln.startswith("coverage: ")]
        disclosure = [
            i for i, ln in enumerate(lines)
            if ln.startswith("coverage-not-established: ")
        ]
        assert len(qualified) == 1, lines
        assert len(disclosure) == 1, lines
        # POSITION, per the render's own stated rationale: a caveat a
        # reader reaches after the number it is about has already been
        # read is a caveat that arrives too late.
        assert disclosure[0] == qualified[0] + 1, lines
        assert lines[qualified[0]] == f"coverage: {stamp.coverage}"
        assert lines[disclosure[0]] == (
            f"coverage-not-established: {stamp.top_despite_coverage}"
        )


def test_the_disclosure_leaves_the_coverage_line_byte_identical():
    # THE COVERAGE STRING IS TREND DATA: `coverage.split(" eqns")[0]` is
    # parsed by reproduce.py, by Verdict.render, and by the sweep
    # scripts. The disclosure is a separate field for that reason, and
    # the line it qualifies must not move a byte when it fires.
    p, plain, solved = both_stamps(cancelling_exp_harness(-1000.0, 1000.0))

    assert plain.top_despite_coverage is not None
    for stamp in (plain, solved):
        assert stamp.coverage == p.coverage.summary()
        head = stamp.coverage.split(" eqns")[0]
        assert head.isdigit() and head == "5"
