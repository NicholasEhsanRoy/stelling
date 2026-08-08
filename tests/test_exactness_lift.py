# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The exactness lift: stelling.exactness is the shared certification
primitive, and the assume machinery routes through it — no jax.

Behavior is byte-identical to the pre-lift inline machinery (the whole
audit-F7/F8 test battery in test_assume_constrain.py passes unmodified);
this file pins the ROUTING — that the certification decision really is
the shared function, so a future layer importing it gets the same
principle the propagator enforces — plus the module's own contract.

**Two routings, at two levels.** :func:`certifies_nonemptiness` is
per-variable ("is this region inhabited?") and only the propagator
consults it. :func:`certifies_set_refutation` is per-RUN ("is a
set-level refutation certified on this run?") and **two** legs consult
it — the interval propagator and the affine refinement. The second pin
is the one that matters for the second leg: the two would otherwise
agree only by coincidence, because the refinement reads a whole-run
quantity by ARCHITECTURE (it is a post-pass over a finished
propagation) while the propagator reads one by DECISION. The pin forces
the shared answer to False and requires BOTH legs to withhold — it is
not a test of today's verdicts, it is a test that fails if either leg
stops consulting the shared point.
"""

from __future__ import annotations

from stelling import affine, exactness, ir
from stelling.exactness import (
    ExactSet,
    certifies_nonemptiness,
    certifies_set_refutation,
)
from stelling.propagate import propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, aval=F64):
    return ir.Var(id=i, aval=aval)


def lit(v):
    return ir.Literal(val=v, aval=F64)


def any_eqn(out, lo, hi):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", "float64"), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out):
    return ir.JaxprEqn(primitive=prim, invars=tuple(ins), outvars=(out,))


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns))
    )


# --- the module's own contract ------------------------------------------------


def test_exact_set_maintenance_rules():
    s = ExactSet()
    assert 0 not in s
    s.mark_declared(0)
    assert 0 in s
    # point consts mark; brackets do not
    assert s.mark_if_point(1, (2.0, 3.0), (2.0, 3.0)) is True
    assert 1 in s
    assert s.mark_if_point(2, (2.0,), (4.0,)) is False  # a genuine bracket
    assert 2 not in s


def test_certification_decision():
    s = ExactSet()
    s.mark_declared(7)
    assert certifies_nonemptiness(s, 7, definitely_true=False)  # exact box
    assert certifies_nonemptiness(s, 9, definitely_true=True)  # F8 channel
    # an over-approximated box with a cutting predicate certifies nothing
    assert not certifies_nonemptiness(s, 9, definitely_true=False)


def test_run_level_certification_decision():
    # the run-level layer: a refutation is certified iff every
    # nonemptiness question this run asked was answered yes AND no assume
    # was dropped. Both arguments are load-bearing and neither implies
    # the other.
    assert certifies_set_refutation(
        nonemptiness_certified=True, assume_dropped=False
    )
    assert not certifies_set_refutation(
        nonemptiness_certified=False, assume_dropped=False
    )
    assert not certifies_set_refutation(
        nonemptiness_certified=True, assume_dropped=True
    )
    assert not certifies_set_refutation(
        nonemptiness_certified=False, assume_dropped=True
    )


def test_the_run_level_decision_takes_no_position_argument():
    # the scope, structurally: an assume is a precondition on the WHOLE
    # QUERY, so the decision may not be askable "here" versus "there".
    # Nothing in this signature names an obligation, an equation or a
    # position, so no caller can make the answer depend on where in the
    # trace it was asked.
    import inspect

    sig = inspect.signature(certifies_set_refutation)
    assert set(sig.parameters) == {"nonemptiness_certified", "assume_dropped"}
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for p in sig.parameters.values()
    )


def test_module_docstring_states_the_principle():
    # the principle is the module's reason to exist; a future layer
    # importing it must find it stated
    doc = exactness.__doc__
    assert "certifies emptiness" in doc
    assert "never nonemptiness" in doc
    assert "exact knowledge" in doc


# --- the routing pin ----------------------------------------------------------


def _certified_refuted_query():
    """x ∈ [0, 1] declared (exact box), assume(x >= 0.9) — certified —
    then assert(x <= 0.5): definitely false over the narrowed set."""
    x, ap, aout, pred, out = (
        var(0), var(1, BOOL), var(2, BOOL), var(3, BOOL), var(4, BOOL),
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


def test_assume_certification_routes_through_the_shared_primitive(monkeypatch):
    # positive control: the declared-input assume certifies and the
    # definite violation stands
    q = _certified_refuted_query()
    p = propagate(q)
    assert p.obligations[0].status == "violated-over-set"
    assert not any("UNCERTIFIED" in n for n in p.notes)

    # the pin: force the shared decision to "certifies nothing" and the
    # SAME query must withhold — proof the propagator consults
    # stelling.exactness.certifies_nonemptiness and not a private copy
    monkeypatch.setattr(
        exactness, "certifies_nonemptiness", lambda *a, **k: False
    )
    p2 = propagate(q)
    assert p2.obligations[0].status == "unknown"
    assert "WITHHELD" in p2.obligations[0].detail
    assert any("UNCERTIFIED" in n for n in p2.notes)


def test_routing_pin_covers_the_f8_channel_too(monkeypatch):
    # a definitely-true no-op assume certifies via the definitely_true
    # argument of the shared primitive; forcing the primitive False must
    # withhold the downstream refutation as well
    x, w, ap, aout, p2, out = (
        var(0), var(1), var(2, BOOL), var(3, BOOL), var(4, BOOL), var(5, BOOL),
    )
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("add", [x, lit(0.0)], w),
            eqn("le", [w, lit(10.0)], ap),  # definitely true over w's box
            eqn("stelling_assume", [ap], aout),
            eqn("ge", [w, lit(5.0)], p2),  # definitely false
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    assert propagate(q).obligations[0].status == "violated-over-set"
    monkeypatch.setattr(
        exactness, "certifies_nonemptiness", lambda *a, **k: False
    )
    assert propagate(q).obligations[0].status == "unknown"


# --- the RUN-level routing pin: both legs, one shared point -------------------


def _interval_refuted_query():
    """No assume at all: `x ∈ [-1, 1]`, `assert(x >= 5)` — the INTERVAL leg
    decides it violated. Nothing here is uncertified, so the unpatched run
    must REFUTE: the pin cannot be satisfied by an inert control."""
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    return close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("ge", [x, lit(5.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _affine_refuted_query():
    """No assume at all, and the INTERVAL leg cannot decide it: `x - x` is
    the box [-2, 2] and the affine form exactly 0, so `x - x >= 0.5` is
    interval-undecided and affine-violated. The only query shape that puts
    the affine leg's own withholding on the table."""
    x, w, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("sub", [x, x], w),
            eqn("ge", [w, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_both_legs_consult_the_shared_set_refutation_point(monkeypatch):
    """The routing pin for the RUN-level decision, on both legs at once.

    After the query-scoping change the two legs withhold on the same
    queries — but they would do so for two different reasons if each held
    its own copy of the rule: this leg by decision, the refinement by the
    accident of being a post-pass. That agreement is fragile, and nothing
    else in the tree would catch it drifting. So: force the shared answer
    to "certifies nothing" and require BOTH legs to withhold.
    """
    qi, qa = _interval_refuted_query(), _affine_refuted_query()

    # positive controls, in the same test so the pin cannot pass by both
    # legs going dark: neither query carries an assume, so both must
    # refute on their own leg with the shared point answering normally.
    pi = propagate(qi)
    assert pi.obligations[0].status == "violated-over-set"
    pa = propagate(qa)
    assert pa.obligations[0].status == "unknown"  # interval cannot decide it
    ra, _ = affine.refine_propagation(qa, pa)
    assert ra.obligations[0].status == "violated-over-set"

    monkeypatch.setattr(
        exactness, "certifies_set_refutation", lambda **k: False
    )

    # leg 1, the interval propagator
    pi2 = propagate(qi)
    assert pi2.obligations[0].status == "unknown", (
        "the interval leg must reach exactness.certifies_set_refutation, "
        "not a private copy of the same expression"
    )
    assert "WITHHELD from REFUTED" in pi2.obligations[0].detail

    # leg 2, the affine refinement — the same forced answer, the same duty
    ra2, rep2 = affine.refine_propagation(qa, propagate(qa))
    assert ra2.obligations[0].status == "unknown", (
        "the affine refinement must reach the SAME shared point; it reads "
        "a whole-run quantity by architecture, which is not the same as "
        "consulting the rule"
    )
    assert rep2.violated == ()
    assert any("WITHHELD from REFUTED" in n for n in ra2.notes)


def test_the_shared_point_is_one_sided_on_both_legs(monkeypatch):
    """Forcing the shared answer False may withhold violations and nothing
    else. A discharge over a superset implies the discharge over the
    intended set; `solvers.py` tried the two-sided version on its own leg
    and reverted it."""
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("ge", [x, lit(-5.0)], pred),  # definitely TRUE over the box
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert propagate(q).obligations[0].status == "discharged"
    monkeypatch.setattr(
        exactness, "certifies_set_refutation", lambda **k: False
    )
    p = propagate(q)
    assert p.obligations[0].status == "discharged", (
        "discharged must never be withheld — the withholding is one-sided"
    )
    assert not any("WITHHELD" in n for n in p.notes)

    # and the affine leg's half of the same rule
    xa, w, preda, outa = var(0), var(1), var(2, BOOL), var(3, BOOL)
    qa = close(
        [
            any_eqn(xa, -1.0, 1.0),
            eqn("sub", [xa, xa], w),
            eqn("ge", [w, lit(-0.5)], preda),  # affine-DISCHARGED (w == 0)
            eqn("stelling_assert", [preda], outa),
        ],
        [outa],
    )
    pa = propagate(qa)
    assert pa.obligations[0].status == "unknown"
    ra, rep = affine.refine_propagation(qa, pa)
    assert ra.obligations[0].status == "discharged"
    assert rep.discharged == (0,)


def test_propagator_exact_set_is_the_shared_class():
    # the per-var exact set the propagator maintains IS the lifted class
    # (importable by future layers), not a leftover builtin set
    from stelling.propagate import _Propagator

    p = _Propagator("constrain")
    assert isinstance(p.exact, ExactSet)
