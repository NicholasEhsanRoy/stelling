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

**THREE routings now.** The NON-EMPTINESS CERTIFICATE added a second,
POSITIVE route to the same nonemptiness fact —
:func:`certifies_point_witness`, "did one probed point of the declared
set satisfy every assume this query wrote?" — and fed it into the
run-level decision as a third argument. Both halves are pinned below:
``test_the_witness_route_is_the_shared_primitive_too`` forces that
function in BOTH directions and requires the propagator to follow, and
``test_every_reach_of_the_shared_point_names_the_certificate`` requires
both legs to carry the answer into the shared point rather than act on it
locally.

**WHAT THIS FILE DID NOT PIN — the shape of its own blind spot, and the
one part of it that is now closed.** The ``lambda **k: False`` /
``lambda *a, **k: False`` routing pins IGNORE their arguments, which is
what makes them ROUTING pins rather than behaviour pins — but it also
means **no argument-level defect could redden them**. A leg that reaches
the shared point with the wrong arguments still gets ``False`` back and
still withholds, so every one of those tests passes. Measured at
``43896fc`` on jax 0.11.0: an ``affine.py`` mutant hardcoding
``assume_dropped=False`` at the shared call passes this file **10/10**,
and is caught only elsewhere — 5 tests in
``tests/test_vacuous_refutation.py`` fail on it (whole suite: 2352
passed, 5 failed).

``test_every_reach_of_the_shared_point_names_the_certificate`` is the
first pin here that is NOT argument-blind: it wraps the real decision in
a recorder and inspects the keyword arguments, so behaviour is untouched
and the arrival of each argument is the observable. It closes the blind
spot for ``region_inhabited`` specifically — it asserts every reach names
it and that a run which certified passes a True — and leaves the other
two arguments exactly as described above.

The OTHER argument is a different case, recorded here so the next reader
does not read its silence as coverage: a mutant hardcoding
``nonemptiness_certified=True`` at the same call passes this file 10/10
**and the whole suite, 2357 passed / 2 skipped**. That is not a coverage
hole, it is an unreachable argument. ``narrowing_uncertified`` is set
only in the branch that also calls ``counter.record_constrained``, so it
implies ``coverage.constrained >= 1``, and
:func:`stelling.affine.refine_propagation` declines wholly on
``coverage.constrained`` before reaching the shared point. MEASURED, not
reasoned, and RE-MEASURED on this tree because the count moves with the
suite: instrumenting that call site records **31** reaches (it was 25 at
``43896fc``), ``narrowing_uncertified`` False at every one — 12 with
``assume_dropped`` True, 2 with a certificate — and
``coverage.constrained == 0`` at every one. On this leg the argument is
a constant today; a refinement restructured to run under a constraining
assume would make it live, and this file would still not see it. The
same fact is now stated AT the call site in ``affine.py``, which is
where a reader of that line meets it.

**AND THE PINS THEMSELVES WERE HALF PINS.** Forcing a shared decision
``False`` observes only the answers it VETOES. Every consumer reads the
answer as one operand of an ``and``, so a leg that kept a private copy of
the rule and wrote ``shared(<all the real arguments>) and _own_copy(...)``
still calls the shared function, unconditionally, first, with everything
a recorder expects — and a forced ``False`` still makes the conjunction
``False``, so the leg withholds and the pin passes on a leg that has
stopped obeying. Measured on this tree at ``0ad22bb``: that mutant on the
affine leg (``M4``), on both legs (``M5``), and the same trick on
:func:`stelling.exactness.certifies_nonemptiness` (``M6``) each pass the
WHOLE SUITE, 2398 passed / 2 skipped / 0 failed. The three
``..._in_the_TRUE_direction`` / ``..._is_ONE_SIDED_too`` tests below force
the granting direction, which a private copy cannot follow, and redden
all three.
"""

from __future__ import annotations

import dataclasses

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


def test_the_certificate_is_an_input_to_the_run_level_decision():
    # the third argument, on its own truth table. A point witness settles
    # nonemptiness outright, which is the ONE thing the other two
    # arguments withhold for, so it overrides either of them; and it
    # defaults to the pre-certificate answer, so a caller that does not
    # know about it gets the conservative behaviour rather than a
    # signature error.
    for dropped in (False, True):
        for certified in (False, True):
            assert certifies_set_refutation(
                nonemptiness_certified=certified,
                assume_dropped=dropped,
                region_inhabited=True,
            ), (certified, dropped)
            assert certifies_set_refutation(
                nonemptiness_certified=certified,
                assume_dropped=dropped,
                region_inhabited=False,
            ) is certifies_set_refutation(
                nonemptiness_certified=certified, assume_dropped=dropped
            ), "omitting the argument must equal passing it False"


def test_the_point_witness_decision_is_one_sided_and_static():
    # the positive channel's own contract. A subset test, in that
    # direction: every assume the QUERY contains must be witnessed, and a
    # witness set that misses one certifies nothing however large it is.
    ka, kb = 101, 202
    assert exactness.certifies_point_witness(
        required_assumes=frozenset({ka}), witnessed_assumes=frozenset({ka, kb})
    )
    assert not exactness.certifies_point_witness(
        required_assumes=frozenset({ka, kb}), witnessed_assumes=frozenset({ka})
    )
    # no assume, no claim: a query with nothing to certify does not get a
    # certificate it never earned
    assert not exactness.certifies_point_witness(
        required_assumes=frozenset(), witnessed_assumes=frozenset({ka})
    )
    assert not exactness.certifies_point_witness(
        required_assumes=frozenset(), witnessed_assumes=frozenset()
    )


def test_the_run_level_decision_takes_no_position_argument():
    # the scope, structurally: an assume is a precondition on the WHOLE
    # QUERY, so the decision may not be askable "here" versus "there".
    # Nothing in this signature names an obligation, an equation or a
    # position, so no caller can make the answer depend on where in the
    # trace it was asked. `region_inhabited` is a RUN quantity for the
    # same reason the other two are: a point either is or is not a member
    # of the whole query's assumed region.
    import inspect

    sig = inspect.signature(certifies_set_refutation)
    assert set(sig.parameters) == {
        "nonemptiness_certified",
        "assume_dropped",
        "region_inhabited",
    }
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for p in sig.parameters.values()
    )
    wsig = inspect.signature(exactness.certifies_point_witness)
    assert set(wsig.parameters) == {"required_assumes", "witnessed_assumes"}
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for p in wsig.parameters.values()
    )


def test_module_docstring_states_the_principle():
    # the principle is the module's reason to exist; a future layer
    # importing it must find it stated
    doc = exactness.__doc__
    assert "certifies emptiness" in doc
    assert "never nonemptiness" in doc
    assert "exact knowledge" in doc


# --- the routing pin ----------------------------------------------------------


def _close_the_certificate(monkeypatch):
    """Force the NON-EMPTINESS CERTIFICATE to find nothing.

    Every routing pin below observes a leg's duty through the SAME
    channel: it forces one shared decision to "certifies nothing" and
    requires the leg to withhold. The certificate is a **second,
    legitimate route to the same conclusion** — a point of the declared
    set that satisfies every assume settles nonemptiness whatever
    `certifies_nonemptiness` was made to answer — so on a query whose
    region really is inhabited it restores the refutation and the pin
    stops observing anything. That is not the certificate misbehaving; it
    is a pin whose channel now has two sources.

    So a pin for the FIRST route closes the second one explicitly. This is
    the shape the whole file already relies on and the reason it can: the
    routing is what is under test, and each source has its own pin
    (`test_the_witness_route_is_the_shared_primitive_too`). Every caller
    below checks the POSITIVE CONTROL with only this patch in force — the
    query must still refute — so a pin can never pass on this patch alone.
    """
    monkeypatch.setattr(
        exactness, "certifies_point_witness", lambda **k: False
    )


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

    # the OTHER route to the same conclusion, closed — and re-checked as a
    # control, so this pin can never pass on the closing patch alone
    _close_the_certificate(monkeypatch)
    assert propagate(q).obligations[0].status == "violated-over-set"

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


def test_the_nonemptiness_route_is_pinned_in_the_TRUE_direction(monkeypatch):
    """The OTHER half of the routing pin above, and the half that was
    missing: force the shared decision **True** on a run that would
    otherwise withhold, and require the propagator to STOP withholding.

    **Why a False-only pin is only half a pin.** The consumer at
    ``propagate.py`` reads the shared answer as one operand of an ``and``.
    A leg that kept a private copy of the same rule and wrote
    ``exactness.certifies_nonemptiness(...) and _own_copy(...)`` still
    calls the shared function, with the real arguments, first — so the
    False-direction pin above forces the conjunction False, the leg
    withholds exactly as required, and the pin passes on a leg that has
    stopped obeying the shared decision. A conjunct can only ever be
    observed through the answer it VETOES. Forcing True is what observes
    the answer it GRANTS, and a private copy cannot follow it.

    The certificate is closed throughout: it is the second, independent
    route to the same conclusion, and leaving it open would let the
    positive control pass without the shared decision doing anything.
    """
    q = _uncertified_but_inhabited_query()
    _close_the_certificate(monkeypatch)

    # positive control, with only the certificate closed: the run really
    # does withhold, so the lift below is observing something
    p0 = propagate(q)
    assert p0.obligations[0].status == "unknown"
    assert p0.narrowing_uncertified is True

    monkeypatch.setattr(
        exactness, "certifies_nonemptiness", lambda *a, **k: True
    )
    p1 = propagate(q)
    assert p1.narrowing_uncertified is False, (
        "the propagator must reach exactness.certifies_nonemptiness and "
        "take its answer — not AND it with a private copy of the same rule, "
        "which a False-only pin cannot tell apart"
    )
    assert p1.obligations[0].status == "violated-over-set"
    assert not any("UNCERTIFIED" in n for n in p1.notes)


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
    _close_the_certificate(monkeypatch)
    assert propagate(q).obligations[0].status == "violated-over-set"  # control
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


def test_both_legs_follow_the_shared_point_in_the_TRUE_direction(monkeypatch):
    """The missing half of the RUN-level routing pin, on both legs at once.

    ``test_both_legs_consult_the_shared_set_refutation_point`` forces the
    shared answer False and requires both legs to withhold. That is the
    RESTRICTIVE direction only, and on its own it is **half a pin**: a leg
    that kept a private copy of the run-level rule and wrote

        not (exactness.certifies_set_refutation(<all three kwargs>)
             and _own_copy(propagation))

    still reaches the shared point, unconditionally, as the FIRST operand,
    with all three real keyword arguments — so the recorder pin below sees
    exactly what it expects, the forced ``False`` still makes the whole
    conjunction False, and both legs still withhold. Measured on this
    tree: that mutant on the affine leg alone, and on both legs at once,
    each pass the whole suite **2398 passed / 2 skipped / 0 failed**. A
    conjunct is observable only through the answers it VETOES; the answers
    it GRANTS are what this test observes.

    So: force the shared answer **True** on runs that would otherwise
    withhold, and require BOTH legs to stop withholding. A private copy
    cannot follow that forcing, because its own answer is still False.

    (The interval leg's query is ``_dropped_and_uncertifiable_query``,
    defined below with the certificate's pins: its assumed region is empty
    and unwitnessable, which is precisely why the run withholds with
    nothing else lifting it.)
    """
    # --- leg 1, the interval propagator ---------------------------------
    qi = _dropped_and_uncertifiable_query()
    pi = propagate(qi)
    # positive control: unpatched, this run really does withhold
    assert pi.obligations[0].status == "unknown"
    assert pi.assume_dropped is True
    assert pi.region_inhabited is False

    # --- leg 2, the affine refinement -----------------------------------
    qa = _affine_refuted_query()
    pa = dataclasses.replace(propagate(qa), assume_dropped=True)
    ra, rep = affine.refine_propagation(qa, pa)
    # positive control: unpatched, this refinement really does withhold
    assert ra.obligations[0].status == "unknown"
    assert rep.violated == ()

    monkeypatch.setattr(
        exactness, "certifies_set_refutation", lambda **k: True
    )

    pi2 = propagate(qi)
    assert pi2.obligations[0].status == "violated-over-set", (
        "the interval leg must TAKE the shared answer, not AND it with a "
        "private copy of the same rule — a False-only pin cannot tell the "
        "two apart"
    )
    assert not any("WITHHELD from REFUTED" in n for n in pi2.notes)

    ra2, rep2 = affine.refine_propagation(qa, pa)
    assert ra2.obligations[0].status == "violated-over-set", (
        "the affine refinement must TAKE the same shared answer, for the "
        "same reason"
    )
    assert rep2.violated == (0,)
    assert not any("WITHHELD from REFUTED" in n for n in ra2.notes)


def _withheld_run_with_all_three_outcomes():
    """`x ∈ [0, 1]`, `w = x·x`, `assume(w <= 0.9)` — a narrowing on an
    OVER-APPROXIMATED intermediate, so the run withholds — carrying one
    obligation of each kind over the declared box:

        #0  `x <= -1`   definitely FALSE  -> violated, and WITHHELD
        #1  `x <= 10`   definitely TRUE   -> discharged
        #2  `x <= 0.5`  straddles         -> unknown, undecided by anyone

    One query with all three outcomes is what makes the one-sidedness
    observable in the GRANTING direction: the shared answer is False here
    for its own reasons, so forcing it True is not a no-op, and the two
    obligations that must not move are on the same run as the one that
    must.
    """
    x, w, ap, aout = var(0), var(1), var(2, BOOL), var(3, BOOL)
    p0, o0 = var(4, BOOL), var(5, BOOL)
    p1, o1 = var(6, BOOL), var(7, BOOL)
    p2, o2 = var(8, BOOL), var(9, BOOL)
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mul", [x, x], w),
            eqn("le", [w, lit(0.9)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("le", [x, lit(-1.0)], p0),
            eqn("stelling_assert", [p0], o0),
            eqn("le", [x, lit(10.0)], p1),
            eqn("stelling_assert", [p1], o1),
            eqn("le", [x, lit(0.5)], p2),
            eqn("stelling_assert", [p2], o2),
        ],
        [o0, o1, o2],
    )


def test_the_TRUE_direction_is_ONE_SIDED_too(monkeypatch):
    """The one-sidedness pin's own missing half.

    ``test_the_shared_point_is_one_sided_on_both_legs`` forces the shared
    answer **False** and requires a discharge to survive. That is the
    direction in which one-sidedness is easy: False is the WITHHOLDING
    answer, and the sentence it pins is "withholding touches violations
    only".

    The other direction is the one the contract actually spends its words
    on — *"a True here can restore a withheld ``violated-over-set`` and
    can do nothing else at all"* — and nothing observed it. Forcing True
    on a query that already answers True is a no-op and would pin
    nothing, so the run below is one that genuinely withholds, with the
    certificate closed, carrying an obligation of every kind at once:
    the violated one must come BACK, and the discharged and the merely
    undecided ones must not move a step.
    """
    q = _withheld_run_with_all_three_outcomes()
    _close_the_certificate(monkeypatch)

    p0 = propagate(q)
    assert [o.status for o in p0.obligations] == [
        "unknown", "discharged", "unknown",
    ], "the positive control must have all three outcomes on one run"
    assert p0.narrowing_uncertified is True
    assert "WITHHELD" in p0.obligations[0].detail
    assert "WITHHELD" not in p0.obligations[2].detail, (
        "obligation #2 must be undecided for its OWN reason, or its "
        "staying unknown below says nothing about one-sidedness"
    )

    monkeypatch.setattr(
        exactness, "certifies_set_refutation", lambda **k: True
    )
    p1 = propagate(q)
    assert p1.obligations[0].status == "violated-over-set", (
        "the granted answer must restore the withheld violation, or this "
        "run is not observing the grant at all"
    )
    assert p1.obligations[1].status == "discharged", (
        "a granted answer may not touch a discharge — the shared point is "
        "one-sided in BOTH directions, not only in the one that withholds"
    )
    assert p1.obligations[2].status == "unknown", (
        "a granted answer may not decide an obligation nobody decided; it "
        "lifts a withholding and does nothing else at all"
    )


# --- the CERTIFICATE's own routing pins ---------------------------------------
#
# Two of them, because the certificate touches the shared machinery in two
# places and each place has its own way of going wrong:
#
#   1. the WITNESS decision — "did this probed point satisfy every assume?"
#      — is `exactness.certifies_point_witness`, and the propagator must
#      CONSULT it. The failure mode is an inlined `required <= witnessed`
#      in propagate.py: behaviour-preserving, invisible to every verdict
#      test in the tree.
#   2. the certificate must reach the run-level decision as an ARGUMENT,
#      from BOTH legs. The failure mode is a leg writing
#      `p.region_inhabited or certifies_set_refutation(<old two kwargs>)`
#      — also behaviour-preserving, and invisible to the pins above,
#      which patch with `lambda **k: False` and so cannot see an argument
#      at all (this file says as much about itself, in its docstring).
#      The pin below therefore RECORDS the keyword arguments instead of
#      ignoring them.


def _uncertified_but_inhabited_query():
    """`x ∈ [0, 1]`, `w = x·x`, `assume(w <= 0.9)` — a narrowing on an
    OVER-APPROXIMATED intermediate, so `narrowing_uncertified` is set and
    the run would withhold — then `assert(x <= -1)`, definitely false.

    The assumed region `{x ∈ [0, 1] : x² ≤ 0.9}` is `[0, 0.948…]`:
    genuinely inhabited, so the certificate genuinely fires here and the
    REFUTED it restores is a correct one."""
    x, w, ap, aout, pred, out = (
        var(0), var(1), var(2, BOOL), var(3, BOOL), var(4, BOOL), var(5, BOOL),
    )
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mul", [x, x], w),
            eqn("le", [w, lit(0.9)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("le", [x, lit(-1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _dropped_and_uncertifiable_query():
    """`x ∈ [0, 1]`, `assume(reduce_and(x >= 2))` — `reduce_and` has no
    interval transfer in either registry, so the assumed predicate is ⊤,
    the classifier drops the assume (`assume_dropped`) and the run
    withholds `assert(x <= -1)`.

    The assumed region is EMPTY (no `x ∈ [0, 1]` is ≥ 2) and the probe
    CANNOT SEE THAT: the predicate is ⊤ at every pinned point too, so no
    witness is recorded and the certificate correctly declines. That is
    what makes it the right query for the second direction of the routing
    pin — the probe run completes, `certifies_point_witness` really is
    asked, and its answer is the only thing standing between this query
    and a wrong REFUTED.

    NOT the `x - x >= 0.5` or the relational `x >= y` shape: at a PINNED
    point both sides of a relational comparison are points, so the
    classifier narrows, the meet comes out empty and the probe dies of
    ``UnsatisfiableAssumptionError`` before the witness decision is ever
    reached. Sound, and useless for pinning what that decision is asked.
    """
    x, c, ap, aout, pred, out = (
        var(0), var(1, BOOL), var(2, BOOL), var(3, BOOL), var(4, BOOL),
        var(5, BOOL),
    )
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("ge", [x, lit(2.0)], c),
            eqn("reduce_and", [c], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("le", [x, lit(-1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_the_witness_route_is_the_shared_primitive_too(monkeypatch):
    """The propagator must CONSULT `exactness.certifies_point_witness`,
    not hold a private copy of the same subset test.

    Pinned in BOTH directions, which is what makes an inlined predicate
    impossible to hide: forced False, an inhabited region stops being
    certified; forced True, an EMPTY one starts being. A `propagate.py`
    that computed `required <= witnessed` itself would ignore both
    forcings and keep its own answer, and neither half of this test would
    pass.
    """
    live, dead = _uncertified_but_inhabited_query(), _dropped_and_uncertifiable_query()

    # positive controls, unpatched: the mechanism really does separate
    # these two queries, so neither half below can pass by going dark
    pl, pd = propagate(live), propagate(dead)
    assert pl.region_inhabited is True
    assert pl.obligations[0].status == "violated-over-set"
    assert pd.region_inhabited is False
    assert pd.obligations[0].status == "unknown"

    monkeypatch.setattr(
        exactness, "certifies_point_witness", lambda **k: False
    )
    p1 = propagate(live)
    assert p1.region_inhabited is False, (
        "the certificate must reach exactness.certifies_point_witness, not "
        "a private copy of the same subset test"
    )
    assert p1.obligations[0].status == "unknown"

    monkeypatch.setattr(
        exactness, "certifies_point_witness", lambda **k: True
    )
    p2 = propagate(dead)
    assert p2.region_inhabited is True, (
        "the same, in the direction an inlined predicate could not follow"
    )
    assert p2.obligations[0].status == "violated-over-set"


def test_every_reach_of_the_shared_point_names_the_certificate(monkeypatch):
    """The ARGUMENT-level routing pin, on both legs, and the one thing
    this file's `lambda **k: False` pins structurally cannot see.

    The recorder DELEGATES to the real decision rather than replacing it,
    so behaviour is untouched and what is under test is only which
    arguments arrive. A leg that lifted the withholding locally —
    `region_inhabited or certifies_set_refutation(<old two kwargs>)` —
    keeps every verdict in the tree exactly as it is and reddens only
    this.
    """
    seen: list[dict] = []
    real = exactness.certifies_set_refutation

    def recorder(**k):
        seen.append(dict(k))
        return real(**k)

    monkeypatch.setattr(exactness, "certifies_set_refutation", recorder)

    # leg 1, the interval propagator, on a query whose certificate FIRES:
    # the True has to be what arrives, not a hardcoded False that happens
    # to agree on the runs where nothing was certified anyway
    p = propagate(_uncertified_but_inhabited_query())
    assert p.region_inhabited is True
    assert p.obligations[0].status == "violated-over-set"
    assert seen, "the interval leg reached the shared point zero times"
    assert all("region_inhabited" in k for k in seen), (
        f"a reach of the shared point omitted the certificate: {seen}"
    )
    assert any(k["region_inhabited"] is True for k in seen), (
        "the interval leg must pass the run's OWN certificate, not a "
        "constant — every reach carried False on a run that certified"
    )

    # leg 2, the affine refinement. Its own certificate arrives on the
    # propagation it is handed, so the pin hands it one that carries a
    # True beside a dropped assume: without the third argument the leg
    # withholds, with it the refinement's violation stands.
    seen.clear()
    qa = _affine_refuted_query()
    pa = propagate(qa)
    withheld = dataclasses.replace(pa, assume_dropped=True)
    ra, rep = affine.refine_propagation(qa, withheld)
    assert ra.obligations[0].status == "unknown"  # control: it does withhold
    assert seen and all("region_inhabited" in k for k in seen), (
        f"the affine leg reached the shared point without the certificate: {seen}"
    )
    assert all(k["region_inhabited"] is False for k in seen)

    seen.clear()
    certified = dataclasses.replace(pa, assume_dropped=True, region_inhabited=True)
    ra2, rep2 = affine.refine_propagation(qa, certified)
    assert any(k["region_inhabited"] is True for k in seen), (
        "the affine leg must pass the propagation's certificate through "
        "the shared point, not decide the lift for itself"
    )
    assert ra2.obligations[0].status == "violated-over-set"
    assert rep2.violated == (0,)


def test_propagator_exact_set_is_the_shared_class():
    # the per-var exact set the propagator maintains IS the lifted class
    # (importable by future layers), not a leftover builtin set
    from stelling.propagate import _Propagator

    p = _Propagator("constrain")
    assert isinstance(p.exact, ExactSet)
