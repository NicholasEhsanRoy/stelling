# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The exactness lift: stelling.exactness is the shared certification
primitive, and the assume machinery routes through it — no jax.

Behavior is byte-identical to the pre-lift inline machinery (the whole
audit-F7/F8 test battery in test_assume_constrain.py passes unmodified);
this file pins the ROUTING — that the certification decision really is
the shared function, so a future layer importing it gets the same
principle the propagator enforces — plus the module's own contract.
"""

from __future__ import annotations

from stelling import exactness, ir
from stelling.exactness import ExactSet, certifies_nonemptiness
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


def test_propagator_exact_set_is_the_shared_class():
    # the per-var exact set the propagator maintains IS the lifted class
    # (importable by future layers), not a leftover builtin set
    from stelling.propagate import _Propagator

    p = _Propagator("constrain")
    assert isinstance(p.exact, ExactSet)
