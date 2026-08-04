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

Hand-built IR, no jax — same construction as tests/test_propagate.py.
"""

from __future__ import annotations

from stelling import ir
from stelling.propagate import propagate
from stelling.solvers import Escalation, make_solver_verdict
from stelling.verdict import make_verdict

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")

VERSIONS = dict(
    stelling_version="test",
    jax_version="none: hand-built IR",
    precision_config="jax_enable_x64=True (hand-built f64 IR)",
)


def var(i, aval=F64):
    return ir.Var(id=i, aval=aval)


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
