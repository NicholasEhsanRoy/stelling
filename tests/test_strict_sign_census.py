# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE STRICT-SIGN CENSUS: every registered transfer classified, and every
classification probed.

0.2.0 disclosed the certificate as "dropped by every primitive without an
explicit rule". Sound, and not a decision: measured on 2026-08-28 at
`2e4b780`, `TRANSFERS` had 50 entries and `_STRICT_SIGN_PRIMITIVES` had 10,
and the other forty dropped the fact BY ABSENCE — nothing in the tree said
whether any of them had been looked at. This module is the instrument that
makes absence impossible: the census in `propagate.py` classifies all fifty,
and every class is probed here, in BOTH directions.

**PROBE-OR-EXEMPT, and the exemption is probed too.** A rule-carrying
primitive gets a case where it MINTS and a case where it DROPS — a presence
check with no absence half can never go red. A primitive in
`_SIGN_BOOLEAN` or `_SIGN_NO_RULE` gets the opposite probe: every operand
certified, and still no certificate. `PROBES` is asserted TOTAL over the
census, so classifying a primitive without probing it is a red.

**WHAT THIS MODULE DOES NOT REACH.** These are RULE probes: they seed
`_Propagator.strict_sign` by hand and call `_strict_sign_out` directly, so
they check what the rule ANSWERS and never that the answer is TRUE of the
program. Truth is the other instrument's job —
`tests/test_assume_bump_boundary_div.py::
test_strict_sign_certificate_is_TRUE_at_every_assumed_point` evaluates
certified values in exact `Fraction` arithmetic at points of the assumed
region, and `tests/property/test_strict_sign_property.py` searches for a
counterexample over generated programs. A rule that answers `+1` for a value
that is negative passes every probe in this file and fails both of those.
Nor does anything here reach a primitive with no registered transfer: those
are ⊤ at the walk and never see a sign rule.
"""

from __future__ import annotations

import math

import pytest

from stelling import interval as iv
from stelling import ir
from stelling import propagate as P
from stelling.propagate import _Propagator


F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")


def _aval(shape):
    return ir.Aval(kind="ShapedArray", shape=tuple(shape), dtype="float64")


def _box(sign, shape=()):
    """A box consistent with `sign`. The rules read boxes only for the two
    SIZE guards, so the endpoints are illustrative and the size is not."""
    n = 1
    for d in shape:
        n *= d
    lo, hi = {1: (1.0, 2.0), -1: (-2.0, -1.0), 0: (-1.0, 1.0)}[sign]
    return iv.IntervalArray(shape=tuple(shape), los=(lo,) * n, his=(hi,) * n)


class Case:
    """One probe: operand certificates in, expected minted sign out."""

    def __init__(self, signs, want, *, params=(), shapes=None, outvars=1,
                 why=""):
        self.signs = tuple(signs)
        self.want = want
        self.params = tuple(params)
        self.shapes = shapes if shapes is not None else [()] * len(self.signs)
        self.outvars = outvars
        self.why = why


def _run_case(prim: str, case: Case) -> int:
    p = _Propagator("constrain")
    invars = []
    ins = []
    for i, (sgn, shape) in enumerate(zip(case.signs, case.shapes)):
        v = ir.Var(id=i + 1, aval=_aval(shape))
        invars.append(v)
        ins.append(_box(sgn if sgn else 0, shape))
        if sgn:
            p.strict_sign[v.id] = sgn
    outs = tuple(
        ir.Var(id=900 + k, aval=F64) for k in range(case.outvars)
    )
    eqn = ir.JaxprEqn(
        primitive=prim,
        invars=tuple(invars),
        outvars=outs,
        params=case.params,
    )
    return p._strict_sign_out(eqn, dict(case.params), ins)


# --- the probe table ---------------------------------------------------------
#
# Every rule-carrying primitive: at least one case that MINTS and one that
# DROPS. The dropping half is not decoration — it is what makes the probe
# able to go red at all, and several of the cases below are the specific
# way each rule could be written wrong:
#
#   `sub` same-sign        — the `Σx² − c` false VERIFIED, still refused
#   `max`/`min` swapped    — each is probed on the arm the OTHER one owns
#   `select_n` selector    — certified selector, uncertified case: must drop
#   `gather` indices       — certified indices, uncertified data: must drop
#   `scatter` updates      — certified operand, opposite-signed update: drop
#   `sqrt` negative        — a `-1` operand answers 0 here and does not lean
#                            on `interval.sqrt` having declined first
PROBES: dict[str, tuple[Case, ...]] = {
    # -- arithmetic ---------------------------------------------------------
    "mul": (Case((1, 1), 1), Case((1, -1), -1), Case((1, 0), 0)),
    "div": (Case((1, -1), -1), Case((-1, -1), 1), Case((0, 1), 0)),
    "add": (Case((1, 1), 1), Case((-1, -1), -1), Case((1, -1), 0)),
    "add_any": (Case((1, 1), 1), Case((1, -1), 0)),
    "sub": (
        Case((1, -1), 1, why="a>0, b<0 => a-b>0"),
        Case((-1, 1), -1, why="the mirror"),
        Case((1, 1), 0, why="THE false-VERIFIED shape; stays refused"),
        Case((-1, -1), 0),
        Case((1, 0), 0),
    ),
    "neg": (Case((1,), -1), Case((-1,), 1), Case((0,), 0)),
    "abs": (Case((-1,), 1), Case((1,), 1), Case((0,), 0)),
    "square": (Case((-1,), 1), Case((0,), 0)),
    "sqrt": (
        Case((1,), 1),
        Case((-1,), 0, why="one-sided by construction, not by decline"),
        Case((0,), 0),
    ),
    "integer_pow": (
        Case((-1,), 1, params=(("y", 2),)),
        Case((-1,), -1, params=(("y", 3),)),
        Case((-1,), 1, params=(("y", 0),)),
        Case((0,), 0, params=(("y", 2),)),
        Case((1,), 0, params=(("y", "not-an-int"),)),
    ),
    "reduce_sum": (
        Case((1,), 1, shapes=[(2,)], params=(("axes", (0,)),)),
        Case((1,), 0, shapes=[(0,)], params=(("axes", (0,)),),
             why="the empty sum is the identity 0"),
        Case((0,), 0, shapes=[(2,)], params=(("axes", (0,)),)),
    ),
    "dot_general": (
        Case((1, -1), -1, shapes=[(2,), (2,)]),
        Case((1, 0), 0, shapes=[(2,), (2,)]),
        Case((1, 1), 0, shapes=[(0,), (0,)],
             why="an empty contraction is the identity 0"),
    ),
    "scatter-add": (
        Case((1, 0, 1), 1, shapes=[(2,), (1,), ()],
             why="operand and updates certified; the index is not a value"),
        Case((1, 0, -1), 0, shapes=[(2,), (1,), ()]),
        Case((0, 1, 1), 0, shapes=[(2,), (1,), ()],
             why="a certified INDEX does not stand in for the operand"),
    ),
    # -- routing ------------------------------------------------------------
    "max": (
        Case((1, 0), 1, why="ONE certified + operand certifies the max"),
        Case((0, 1), 1, why="...on either side"),
        Case((-1, -1), -1),
        Case((-1, 0), 0, why="min's arm; max must not take it"),
    ),
    "min": (
        Case((-1, 0), -1),
        Case((0, -1), -1),
        Case((1, 1), 1),
        Case((1, 0), 0, why="max's arm; min must not take it"),
    ),
    "select_n": (
        Case((0, 1, 1), 1, why="the SELECTOR is an index, not a value"),
        Case((0, 1, -1), 0, why="the cases must agree"),
        Case((1, 1, 0), 0, why="a certified selector rescues nothing"),
    ),
    "gather": (
        Case((1, 0), 1, shapes=[(2,), (1,)], why="indices are not values"),
        Case((0, 1), 0, shapes=[(2,), (1,)]),
    ),
    "dynamic_slice": (
        Case((1, 0), 1, shapes=[(2,), ()]),
        Case((0, 1), 0, shapes=[(2,), ()]),
    ),
    "scatter": (
        Case((1, 0, 1), 1, shapes=[(2,), (1,), ()]),
        Case((1, 0, -1), 0, shapes=[(2,), (1,), ()],
             why="the UPDATE is a value operand and must agree"),
        Case((0, 1, 1), 0, shapes=[(2,), (1,), ()]),
    ),
    "dynamic_update_slice": (
        Case((1, 1, 0), 1, shapes=[(2,), (1,), ()]),
        Case((1, -1, 0), 0, shapes=[(2,), (1,), ()],
             why="the UPDATE is a value operand and must agree"),
        Case((0, 1, 0), 0, shapes=[(2,), (1,), ()]),
    ),
    "concatenate": (
        Case((1, 1), 1, shapes=[(2,), (2,)], params=(("dimension", 0),)),
        Case((1, -1), 0, shapes=[(2,), (2,)], params=(("dimension", 0),)),
        Case((1, 0), 0, shapes=[(2,), (2,)], params=(("dimension", 0),)),
    ),
    "stack": (
        Case((-1, -1), -1, shapes=[(2,), (2,)], params=(("axis", 0),)),
        Case((-1, 1), 0, shapes=[(2,), (2,)], params=(("axis", 0),)),
    ),
    "split": (
        Case((1,), 1, shapes=[(4,)], outvars=2,
             why="multi-output: routing's claim covers every outvar"),
        Case((0,), 0, shapes=[(4,)], outvars=2),
    ),
    "unstack": (
        Case((-1,), -1, shapes=[(2,)], outvars=2),
        Case((0,), 0, shapes=[(2,)], outvars=2),
    ),
    "reshape": (Case((1,), 1), Case((-1,), -1), Case((0,), 0)),
    "transpose": (Case((1,), 1), Case((0,), 0)),
    "broadcast_in_dim": (Case((-1,), -1), Case((0,), 0)),
    "slice": (Case((1,), 1), Case((0,), 0)),
    "squeeze": (Case((1,), 1), Case((0,), 0)),
    "copy": (Case((-1,), -1), Case((0,), 0)),
    "stop_gradient": (Case((1,), 1), Case((0,), 0)),
}

# The exempt half. Every operand certified `+1`, and the answer must still
# be 0 — the whole point of a classification that says "no rule here".
EXEMPT_PROBES: dict[str, Case] = {
    # boolean-valued: False IS zero
    "and": Case((1, 1), 0), "or": Case((1, 1), 0), "not": Case((1,), 0),
    "eq": Case((1, 1), 0), "ne": Case((1, 1), 0), "lt": Case((1, 1), 0),
    "le": Case((1, 1), 0), "gt": Case((1, 1), 0), "ge": Case((1, 1), 0),
    "is_finite": Case((1,), 0), "reduce_or": Case((1,), 0),
    # deliberately no rule
    "exp": Case((1,), 0), "pow": Case((1, 1), 0), "sign": Case((1,), 0),
    "rem": Case((1, 1), 0), "convert_element_type": Case((1,), 0),
    "stelling_any": Case((), 0), "stelling_assert": Case((1,), 0),
    "stelling_nonvacuity": Case((1,), 0),
}


# --- the census is total, and the totality check can go red ------------------


def test_the_census_is_total_over_TRANSFERS():
    """Every registered transfer is classified. Derived from the registry
    at test time rather than typed, so a new transfer reds here."""
    union = (
        set(P._SIGN_ROUTING) | set(P._SIGN_ARITHMETIC)
        | set(P._SIGN_BOOLEAN) | set(P._SIGN_NO_RULE)
    )
    assert union == set(P.TRANSFERS), (
        f"unclassified {sorted(set(P.TRANSFERS) - union)}, "
        f"stale {sorted(union - set(P.TRANSFERS))}"
    )


def test_the_census_classes_are_disjoint():
    classes = {
        "routing": set(P._SIGN_ROUTING),
        "arithmetic": set(P._SIGN_ARITHMETIC),
        "boolean": set(P._SIGN_BOOLEAN),
        "no_rule": set(P._SIGN_NO_RULE),
    }
    for a in classes:
        for b in classes:
            if a < b:
                assert not classes[a] & classes[b], (
                    f"{a} and {b} both claim {sorted(classes[a] & classes[b])}"
                )


def test_registering_a_transfer_without_classifying_it_RAISES(monkeypatch):
    """THE ABSENCE HALF of the totality check: it must actually fire.

    A totality assert nobody has driven is a totality assert that might be
    comparing a set with itself."""
    monkeypatch.setitem(P.TRANSFERS, "a_primitive_nobody_classified", (None, "exact"))
    with pytest.raises(RuntimeError, match="total over TRANSFERS"):
        P._assert_sign_census_is_total()


def test_a_primitive_in_two_classes_RAISES(monkeypatch):
    monkeypatch.setattr(P, "_SIGN_BOOLEAN", P._SIGN_BOOLEAN | {"mul"})
    with pytest.raises(RuntimeError, match="disjoint"):
        P._assert_sign_census_is_total()


def test_the_gated_set_is_DERIVED_from_the_census():
    """`_STRICT_SIGN_PRIMITIVES` is the union of the two rule-carrying
    classes and is not a fourth hand-written list of names."""
    assert P._STRICT_SIGN_PRIMITIVES == P._SIGN_ROUTING | P._SIGN_ARITHMETIC


def test_DELIBERATELY_NO_RULE_is_non_empty_and_every_member_carries_a_reason():
    """A census whose "we looked and said no" class is empty is a census
    that only recorded the easy half."""
    assert P._SIGN_NO_RULE, "no primitive was deliberately refused a rule"
    for name, why in P._SIGN_NO_RULE.items():
        assert len(why) > 40, f"{name}'s reason is too thin to act on: {why!r}"


# --- probe-or-exempt, total over the census ---------------------------------


def test_every_rule_carrying_primitive_is_PROBED():
    assert set(PROBES) == set(P._STRICT_SIGN_PRIMITIVES), (
        f"unprobed {sorted(set(P._STRICT_SIGN_PRIMITIVES) - set(PROBES))}, "
        f"stale {sorted(set(PROBES) - set(P._STRICT_SIGN_PRIMITIVES))}"
    )


def test_every_exempt_primitive_is_PROBED():
    exempt = set(P._SIGN_BOOLEAN) | set(P._SIGN_NO_RULE)
    assert set(EXEMPT_PROBES) == exempt, (
        f"unprobed {sorted(exempt - set(EXEMPT_PROBES))}, "
        f"stale {sorted(set(EXEMPT_PROBES) - exempt)}"
    )


def test_every_rule_carrying_primitive_has_a_case_that_MINTS_and_one_that_DROPS():
    """A probe with only minting cases could not catch a rule that always
    answers +1; a probe with only dropping cases could not catch a rule
    that was deleted."""
    for prim, cases in sorted(PROBES.items()):
        assert any(c.want for c in cases), f"{prim}: no case mints"
        assert any(not c.want for c in cases), f"{prim}: no case drops"


@pytest.mark.parametrize(
    "prim,index",
    [(p, i) for p, cs in sorted(PROBES.items()) for i in range(len(cs))],
)
def test_the_rule_answers_what_the_census_says_it_does(prim, index):
    case = PROBES[prim][index]
    got = _run_case(prim, case)
    assert got == case.want, (
        f"{prim} with operand signs {case.signs} minted {got}, expected "
        f"{case.want}{' — ' + case.why if case.why else ''}"
    )


@pytest.mark.parametrize("prim", sorted(EXEMPT_PROBES))
def test_an_exempt_primitive_mints_nothing_even_with_every_operand_certified(prim):
    case = EXEMPT_PROBES[prim]
    got = _run_case(prim, case)
    assert got == 0, (
        f"{prim} is classified as carrying NO rule but minted {got} from "
        f"operands {case.signs}"
    )


# --- the two contract changes, pinned -----------------------------------------


def test_a_multi_output_equation_mints_nothing_outside_the_routing_class():
    """The routing class's claim is quantified over every output element of
    every output, which is what licenses multi-output there. Arithmetic
    makes no such claim, so the short-circuit still holds it to one."""
    for prim in sorted(P._SIGN_ARITHMETIC):
        cases = [c for c in PROBES[prim] if c.want]
        assert cases, prim
        c = cases[0]
        multi = Case(c.signs, c.want, params=c.params, shapes=c.shapes,
                     outvars=2)
        assert _run_case(prim, multi) == 0, (
            f"{prim} minted a sign for an equation with two outvars without "
            f"saying which output it spoke about"
        )


def test_a_routing_primitive_with_no_value_operands_mints_nothing():
    """`_sign_value_operands` returns () on an arity that does not match the
    shape it knows, and () must read as "certify nothing" rather than as
    "every operand agreed vacuously"."""
    assert P._sign_value_operands("select_n", [1]) == ()
    assert P._sign_value_operands("scatter", [1, 1]) == ()
    assert P._sign_value_operands("dynamic_update_slice", [1]) == ()
    # ...and the rule reads that as 0
    assert _run_case("select_n", Case((1,), 0)) == 0


def test_the_value_operand_map_defaults_to_EVERY_operand():
    """The fallthrough is the fail-closed one: a primitive this helper has
    never heard of gets its whole operand list, so an index operand has to
    be certified too and the rule almost always drops. Over-conservative,
    never wrong."""
    assert P._sign_value_operands("a_primitive_it_never_heard_of", [1, 0]) == (1, 0)


# --- the certificate against the EXECUTED program ----------------------------
#
# The probes above check what the rules ANSWER. This section checks that the
# answer is TRUE, and it does it by running the program rather than by
# modelling it: each case is a real jax harness, traced by `stelling.harness`,
# whose asserted predicate's operand is produced by the primitive under test.
# The propagator's certificate for that var is read out of `strict_sign`, and
# the same `fn` is then EVALUATED with jax at points of the assumed region.
#
# This is the instrument for the routing class's INDEXING members —
# `gather`, `scatter`, `scatter-add`, `dynamic_slice`,
# `dynamic_update_slice` — whose admitted forms carry dimension-number
# params and row geometry. The exact-`Fraction` check in
# `tests/test_assume_bump_boundary_div.py` is rank-1 and C-order-flat and
# cannot evaluate them; re-implementing that geometry inside a test would be
# a second copy of the thing under test.
#
# **WHAT THIS DOES NOT REACH.** It SAMPLES a grid; it does not prove. It
# evaluates in binary64, so it is not the exact-rational witness the other
# check is — which is why the two coexist rather than one replacing the
# other. The declared boxes below are chosen so every sampled value and
# every operation on it is exact in binary64 (halves and quarters, sums and
# copies), so a violation it reports is a real one and not a rounding
# artifact; a rule that were wrong only in the last ulp would pass here.


_JAX_POINTS = (
    (0.25, 0.5, 1.0, 2.0),
    (2.0, 1.0, 0.5, 0.25),
    (0.5, 0.5, 0.5, 0.5),
    (1.5, 0.25, 2.0, 0.75),
)
"""Points of the assumed region `x > 0` inside the declared `[0, 2]`, all
exactly representable in binary64 so no sampled arithmetic rounds."""

_JAX_POINTS_STRADDLING = (
    (-2.0, 0.5, 1.0, -0.25),
    (1.0, -1.5, 0.25, 2.0),
)
"""Points of a declared `[-2, 2]` with NO assume — used by the mutation
controls, where the value under test is deliberately not certified."""


def _jax_case(fn, *, bounds=(0.0, 2.0), assume_positive=True):
    """Trace `assume(x > 0); assert_(fn(x) > 0)` and run the propagator.

    Returns ``(certified sign of fn(x), the primitive that produced it)``.
    The value is located as the operand of the assert's own comparison, so
    a case whose `fn` does not END with the primitive under test is caught
    by the primitive name rather than passing on a neighbour's rule.
    """
    import jax
    import jax.numpy as jnp

    from stelling import ir
    from stelling.harness import any_array, assert_, assume, trace

    def h():
        x = any_array((4,), jnp.float64, bounds)
        if assume_positive:
            assume(x > 0)
        return assert_(fn(x) > 0)

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        cj = trace(h)
    finally:
        jax.config.update("jax_enable_x64", old)
    p = _Propagator("constrain")
    p.run(cj.jaxpr, list(cj.consts), [])

    asserts = [e for e in cj.jaxpr.eqns if e.primitive == "stelling_assert"]
    assert len(asserts) == 1, [e.primitive for e in cj.jaxpr.eqns]
    pred = asserts[0].invars[0]
    cmp_eqn = next(
        e for e in cj.jaxpr.eqns
        if e.outvars and e.outvars[0].id == pred.id
    )
    value = cmp_eqn.invars[0]
    assert not isinstance(value, ir.Literal), "the compared value folded away"
    producer = next(
        (e for e in cj.jaxpr.eqns
         if any(o.id == value.id for o in e.outvars)),
        None,
    )
    assert producer is not None, "no equation produces the compared value"
    return p.strict_sign.get(value.id, 0), producer.primitive


def _eval_points(fn, points):
    import jax
    import jax.numpy as jnp

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return [
            [float(v) for v in jnp.ravel(fn(jnp.array(pt, dtype=jnp.float64)))]
            for pt in points
        ]
    finally:
        jax.config.update("jax_enable_x64", old)


# primitive -> (fn, expected certified sign)
JAX_TRUTH_CASES = {
    "gather": (lambda x: x[__import__("jax").numpy.array([1, 2])], 1),
    "scatter": (lambda x: x.at[1].set(3.0), 1),
    "scatter-add": (lambda x: x.at[1].add(1.0), 1),
    "dynamic_slice": (
        lambda x: __import__("jax").lax.dynamic_slice(
            x, (__import__("jax").numpy.int32(1),), (2,)), 1),
    "dynamic_update_slice": (
        lambda x: __import__("jax").lax.dynamic_update_slice(
            x, x[:2], (__import__("jax").numpy.int32(1),)), 1),
    "split": (lambda x: __import__("jax").numpy.split(x, 2)[1], 1),
    "unstack": (lambda x: __import__("jax").numpy.unstack(x)[2], 1),
    "stack": (lambda x: __import__("jax").numpy.stack([x, x]), 1),
    "transpose": (
        lambda x: __import__("jax").numpy.transpose(
            __import__("jax").numpy.reshape(x, (2, 2))), 1),
    "squeeze": (
        lambda x: __import__("jax").numpy.squeeze(
            __import__("jax").numpy.reshape(x, (4, 1)), axis=1), 1),
    "broadcast_in_dim": (
        lambda x: __import__("jax").numpy.broadcast_to(x, (2, 4)), 1),
    "copy": (lambda x: __import__("jax").numpy.array(x), 1),
    "stop_gradient": (lambda x: __import__("jax").lax.stop_gradient(x), 1),
    "concatenate": (
        lambda x: __import__("jax").numpy.concatenate([x, x]), 1),
    "slice": (lambda x: x[1:3], 1),
    "reshape": (lambda x: __import__("jax").numpy.reshape(x, (2, 2)), 1),
    "select_n": (
        lambda x: __import__("jax").lax.select_n(
            (x > 1.0).astype(__import__("jax").numpy.int32), x, 2.0 * x), 1),
    "max": (lambda x: __import__("jax").numpy.maximum(x, -5.0), 1),
    "min": (lambda x: __import__("jax").numpy.minimum(x, 5.0), 1),
    "sqrt": (lambda x: __import__("jax").numpy.sqrt(x * x), 1),
    "sub": (lambda x: x - (-x), 1),
}


def test_every_0_3_0_rule_has_an_EXECUTED_truth_case():
    """The 0.3.0 additions are the routing class plus `sub`, `sqrt` and
    `scatter-add`. Derived from the census at test time, so a rule added to
    a class without a truth case reds here."""
    pytest.importorskip("jax")
    want = set(P._SIGN_ROUTING) | {"sub", "sqrt", "scatter-add"}
    assert set(JAX_TRUTH_CASES) == want, (
        f"missing {sorted(want - set(JAX_TRUTH_CASES))}, "
        f"stale {sorted(set(JAX_TRUTH_CASES) - want)}"
    )


@pytest.mark.parametrize("prim", sorted(JAX_TRUTH_CASES))
def test_the_certificate_is_TRUE_of_the_EXECUTED_program(prim):
    pytest.importorskip("jax")
    fn, want = JAX_TRUTH_CASES[prim]
    sign, producer = _jax_case(fn)
    assert producer == prim, (
        f"{prim}: the compared value is produced by {producer!r}, so this "
        f"case does not exercise the {prim!r} rule"
    )
    assert sign == want, (
        f"{prim}: the certified sign is {sign}, expected {want} — this case "
        f"would pass with the {prim!r} rule deleted"
    )
    for pt, vals in zip(_JAX_POINTS, _eval_points(fn, _JAX_POINTS)):
        assert vals, f"{prim}: the executed value is empty at {pt}"
        for v in vals:
            assert (v > 0) if sign > 0 else (v < 0), (
                f"{prim}: certified sign={sign}, but jax computes {v} at "
                f"the assumed point {pt}"
            )


# primitive -> (wrong `_sign_value_operands`, fn, bounds, assume, why)
JAX_MUTATIONS = {
    # NOT "reads the indices": measured, the traced index column passes
    # through `convert_element_type`, which is in `_SIGN_NO_RULE`, so an
    # indices-reading mutant mints nothing and would be a control that
    # cannot fire. The live unsoundness shape for a routing rule is
    # DEFAULTING an uncertified operand instead of dropping it.
    "gather": (
        lambda prim, sgn: (sgn[0] or 1,),
        lambda x: x[__import__("jax").numpy.array([1, 2])],
        (-2.0, 2.0), False,
        "gather treats an UNCERTIFIED data operand as +1 instead of "
        "dropping the fact",
    ),
    "dynamic_slice": (
        lambda prim, sgn: tuple(sgn[1:]),
        lambda x: __import__("jax").lax.dynamic_slice(
            x, (__import__("jax").numpy.int32(1),), (2,)),
        (-2.0, 2.0), False,
        "dynamic_slice reads the START indices instead of the data",
    ),
    "scatter": (
        lambda prim, sgn: tuple(sgn[:1]),
        lambda x: x.at[1].set(-3.0),
        (0.0, 2.0), True,
        "scatter reads only the operand and ignores the UPDATE",
    ),
    # `scatter-add`'s rule requires TWO value operands and answers 0 on any
    # other arity, so a "drop the update" mutant cannot fire. The live
    # shape is assuming the update shares the operand's sign.
    "scatter-add": (
        lambda prim, sgn: (sgn[0], sgn[0]),
        lambda x: x.at[1].add(-5.0),
        (0.0, 2.0), True,
        "scatter-add assumes the UPDATE shares the operand's sign",
    ),
    "dynamic_update_slice": (
        lambda prim, sgn: tuple(sgn[:1]),
        lambda x: __import__("jax").lax.dynamic_update_slice(
            x, -x[:2], (__import__("jax").numpy.int32(1),)),
        (0.0, 2.0), True,
        "dynamic_update_slice reads only the operand and ignores the UPDATE",
    ),
}


@pytest.mark.parametrize("prim", sorted(JAX_MUTATIONS))
def test_a_wrong_VALUE_OPERAND_map_is_caught_by_the_executed_program(prim):
    """PER-RULE MUTATION for the indexing members.

    `_sign_value_operands` is where a routing rule can be unsound: naming
    FEWER operands than the output is built from lets the rule mint on an
    operand that does not determine the value. Each mutant below names the
    wrong set, and the executed program contradicts the certificate it then
    mints. Both halves are asserted — that the mutant fired at all, and
    that the program refutes it.
    """
    pytest.importorskip("jax")
    wrong, fn, bounds, assume_pos, why = JAX_MUTATIONS[prim]
    real = P._sign_value_operands
    points = _JAX_POINTS if assume_pos else _JAX_POINTS_STRADDLING

    def mutated(p_name, sgn):
        if p_name == prim:
            return wrong(p_name, sgn)
        return real(p_name, sgn)

    P._sign_value_operands = mutated
    try:
        sign, producer = _jax_case(
            fn, bounds=bounds, assume_positive=assume_pos
        )
    finally:
        P._sign_value_operands = real
    assert producer == prim, producer
    assert sign, (
        f"{prim}: the mutant ({why}) minted nothing, so this control "
        f"demonstrates nothing"
    )
    violations = 0
    for pt, vals in zip(points, _eval_points(fn, points)):
        for v in vals:
            if not ((v > 0) if sign > 0 else (v < 0)):
                violations += 1
    assert violations, (
        f"{prim}: the mutant ({why}) certified sign={sign} and the executed "
        f"program agreed at every sampled point — this control is not a "
        f"control"
    )
    # ...and the SHIPPED map does not mint it
    shipped, _ = _jax_case(fn, bounds=bounds, assume_positive=assume_pos)
    assert shipped != sign, (
        f"{prim}: the shipped value-operand map answers the same as the "
        f"mutant ({why}) on this query, so the control tests the query"
    )


# --- the EMPTY-VALUE rule, stated once and enforced in one place -------------
#
# "Every element of this value is > 0" is VACUOUSLY TRUE of a value with no
# elements, and the `div` boundary gate cannot tell a vacuous certificate
# from an earned one. `_Propagator._record_strict_sign` is the one write
# path and the one place the rule lives; these are its two halves plus the
# three writers that go through it.


def test_the_one_writer_refuses_a_size_0_value_and_accepts_a_sized_one():
    """Presence AND absence on the rule itself. A guard with no accepting
    case could be `if False`."""
    p = _Propagator("constrain")
    p._record_strict_sign(7, 0, 1)
    assert 7 not in p.strict_sign, p.strict_sign
    p._record_strict_sign(7, 3, 1)
    assert p.strict_sign[7] == 1
    p._record_strict_sign(8, 3, 0)
    assert 8 not in p.strict_sign, "sign 0 means unknown and writes nothing"


def _assume_query(shape, *, tail=()):
    """`assume(x > 0)` over a declared `x` of `shape`, plus `tail(x)`
    equations. Hand-built; no jax."""
    n = 1
    for d in shape:
        n *= d
    xa = _aval(shape)
    ba = ir.Aval(kind="ShapedArray", shape=tuple(shape), dtype="bool")
    x = ir.Var(id=0, aval=xa)
    pa, ao = ir.Var(id=1, aval=ba), ir.Var(id=2, aval=ba)
    eqns = [
        ir.JaxprEqn(
            primitive="stelling_any", invars=(), outvars=(x,),
            params=(("shape", tuple(shape)), ("dtype", "float64"),
                    ("lo", 0.0), ("hi", 2.0)),
        ),
        ir.JaxprEqn(primitive="gt", invars=(x, ir.Literal(val=0.0, aval=F64)),
                    outvars=(pa,), params=()),
        ir.JaxprEqn(primitive="stelling_assume", invars=(pa,), outvars=(ao,),
                    params=()),
    ]
    extra = list(tail(x)) if tail else []
    eqns.extend(extra)
    out = ir.Var(id=99, aval=ir.Aval(kind="ShapedArray", shape=(), dtype="bool"))
    eqns.append(ir.JaxprEqn(primitive="stelling_assert", invars=(ao,),
                            outvars=(out,), params=()))
    closed = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=(out,),
                       eqns=tuple(eqns))
    )
    p = _Propagator("constrain")
    p.run(closed.jaxpr, list(closed.consts), [])
    return p, x, extra


def test_a_strict_assume_on_a_SIZE_0_declaration_certifies_nothing():
    """WRITER 1's empty case — **and this docstring used to call it "a live
    hole", which the 0.3.0 audit measured false.**

    It read: "`ks` is the per-element bound list, so on an empty target
    `all(k >= 0.0 for k in ks)` is vacuously true and the arm used to write
    `+1` for a value with no elements." The first clause is true; the
    conclusion was inferred and never driven. MEASURED, by replacing
    `_record_strict_sign` with an unguarded writer and running this very
    query: the table comes back EMPTY either way, because an assume over a
    size-0 predicate is dropped inert before the writer is reached.

    **SO THIS TEST HAS NO ABSENCE HALF AND CANNOT GO RED ON THE GUARD** —
    the shape this project's rules name as a defect, disclosed rather than
    dressed up. It pins the shipped ANSWER, which is worth pinning because
    a future change to the inert-drop ordering would make the writer
    reachable. The guard's actual control is
    `test_a_ROUTING_rule_that_produces_an_EMPTY_output_certifies_nothing`
    below, which does go red without it."""
    p, x, _ = _assume_query((0,))
    assert p.strict_sign == {}, (
        f"a size-0 declaration minted {p.strict_sign}; 'every element is "
        f"> 0' is vacuous over no elements"
    )
    # ...and the same query with elements DOES certify, so the test above
    # is not passing because the assume never fired
    q, _x, _ = _assume_query((3,))
    assert q.strict_sign, "the sized control certified nothing"


def test_a_ROUTING_rule_that_produces_an_EMPTY_output_certifies_nothing():
    """WRITER 3's empty case, and THE control for the whole empty-value
    rule. `slice(x, [0], [0])` over a certified `x` has a size-0 output and
    the routing rule's operand agreement holds, so the only thing refusing
    the vacuous certificate is the output-side guard.

    DRIVEN, not asserted: with `_record_strict_sign` replaced by an
    unguarded writer this query mints `{0: 1, 10: 1}` — the size-0 slice
    certified `+1` — and shipped it mints only `{0: 1}`. The audit
    reproduced the same for `broadcast_to(x, (0,))`."""
    def tail(x):
        out = ir.Var(id=10, aval=_aval((0,)))
        return [ir.JaxprEqn(
            primitive="slice", invars=(x,), outvars=(out,),
            params=(("start_indices", (0,)), ("limit_indices", (0,)),
                    ("strides", None)),
        )]

    p, x, extra = _assume_query((3,), tail=tail)
    assert p.strict_sign.get(x.id) == 1, "the operand was not certified"
    empty = extra[0].outvars[0]
    assert not p.strict_sign.get(empty.id), (
        f"a size-0 slice minted {p.strict_sign.get(empty.id)}"
    )
    # the same slice with a NON-empty window does carry it — the absence
    # above is the size and not the rule
    def tail2(x):
        out = ir.Var(id=10, aval=_aval((2,)))
        return [ir.JaxprEqn(
            primitive="slice", invars=(x,), outvars=(out,),
            params=(("start_indices", (0,)), ("limit_indices", (2,)),
                    ("strides", None)),
        )]

    q, _x, extra2 = _assume_query((3,), tail=tail2)
    assert q.strict_sign.get(extra2[0].outvars[0].id) == 1


# --- the audit repairs: the three claims that were false, pinned ------------
#
# A blinded audit of `e698abb` built an independent oracle (`trace_with_jaxpr`
# gives the stelling IR and jax's own ClosedJaxpr from ONE trace, so the same
# program can be propagated and then EXECUTED at points filtered to the assumed
# region). It found no wrong verdict — 500 traced programs, 168 verdict-level
# checks, 0 false VERIFIED, 0 false REFUTED — and three ARGUMENTS that were
# false. On a change whose whole premise is "absence is not a decision, here is
# an argument per row", a false argument is the defect it was written to remove,
# so each is repaired in place and pinned here.


def _f64_lax():
    import jax
    import jax.numpy as jnp
    from jax import lax

    return jax, jnp, lax


def _at_x64(fn):
    import jax

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return fn()
    finally:
        jax.config.update("jax_enable_x64", old)


def test_the_target_flushes_a_subnormal_SQRT_OPERAND_to_zero():
    """F1. The `sqrt` row was admitted on the sentence "`sqrt` cannot
    underflow a nonzero to zero in binary64 … even at the smallest
    subnormal". That is true of `math.sqrt` — which is what it was measured
    against — and FALSE of `lax.sqrt`, which is the program stelling is
    pointed at, because the target flushes subnormal INPUTS.

    REDDENS ON A LOWERING CHANGE, which is the point: the row's honest
    argument now rests on this measurement rather than on its denial."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()
    assert iv.target_flushes_subnormals("float64") is True

    jit_sqrt = jax.jit(lax.sqrt)

    def go():
        out = []
        for x in (5e-324, math.nextafter(iv.MIN_NORMAL, 0.0), 1e-320, 1e-310):
            out.append((
                x,
                math.sqrt(x),
                float(lax.sqrt(jnp.float64(x))),
                float(jit_sqrt(jnp.float64(x))),
            ))
        normal = float(lax.sqrt(jnp.float64(iv.MIN_NORMAL)))
        return out, normal

    rows, normal = _at_x64(go)
    for x, real_root, eager, jitted in rows:
        assert real_root > 0.0, x
        assert eager == 0.0 and jitted == 0.0, (
            f"lax.sqrt({x!r}) = {eager!r}/{jitted!r}; the `sqrt` row's "
            f"argument is that this IS zero on the target and that the band "
            f"it happens on is smaller than `mul`'s"
        )
    assert normal == math.sqrt(iv.MIN_NORMAL) > 0.0, (
        "at MIN_NORMAL the flush stops — without this the test would pass "
        "on a target that returned 0 for everything"
    )


def test_sqrt_diverges_on_a_STRICTLY_SMALLER_set_than_the_mul_this_table_admits():
    """F1's replacement argument, measured rather than asserted.

    `sqrt` is admitted while `exp` and `pow` are refused, and the honest
    reason is not that `sqrt` has no real-vs-executable gap — it has one —
    but that its gap is contained in the gap of `mul`, which this table has
    carried since 0.2.0. If that containment ever stops holding, the row's
    argument stops holding with it."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()

    def smallest_nonzero(f, lo, hi):
        def go():
            a, b = lo, hi
            for _ in range(400):
                mid = math.sqrt(a * b)
                if not (a < mid < b):
                    break
                if f(mid) != 0.0:
                    b = mid
                else:
                    a = mid
            return b

        return _at_x64(go)

    s = smallest_nonzero(
        lambda v: float(lax.sqrt(jnp.float64(v))), 1e-324, 1e-300
    )
    m = smallest_nonzero(
        lambda v: float(lax.mul(jnp.float64(v), jnp.float64(v))),
        1e-324, 1e-100,
    )
    assert s < m, (
        f"sqrt first computes nonzero at {s:.6e} and mul(x,x) at {m:.6e}; "
        f"the `sqrt` row's argument is that sqrt's divergence set is the "
        f"SMALLER of the two"
    )
    # ...and by a wide margin, so the claim is not resting on one ulp
    assert m / s > 1e100, (m, s)


def test_sign_is_refused_because_ITS_TRANSFER_adds_a_zero_and_sqrt_does_not():
    """F1's consistency half. `sign` and `sqrt` sit on the same DAZ band and
    the census answers them differently; this is the fact that makes that
    consistent rather than arbitrary.

    A box that EXCLUDES zero goes into both transfers. `_t_sign` returns one
    that CONTAINS zero — it models the flush, in both semantics modes, on
    purpose — so a certificate there would contradict its own transfer in
    exactly the shape that unlocks `boundary_div`. `interval.sqrt` returns
    one that does not."""
    from stelling.propagate import _t_sign, _t_sqrt

    box = iv.IntervalArray(shape=(), los=(1e-320,), his=(1e-310,))
    assert not iv.straddles_zero(box), "the operand box must exclude zero"

    def one(prim, fn):
        v = ir.Var(id=1, aval=F64)
        e = ir.JaxprEqn(
            primitive=prim, invars=(v,),
            outvars=(ir.Var(id=2, aval=F64),), params=(),
        )
        return fn(e, {}, [box])[0]

    signed = one("sign", _t_sign)
    rooted = one("sqrt", _t_sqrt)
    assert (signed.los[0], signed.his[0]) == (0.0, 1.0)
    assert iv.straddles_zero(signed), (
        "`sign`'s refusal rests on its transfer ADDING a zero here"
    )
    assert not iv.straddles_zero(rooted), (
        f"`sqrt` returned {(rooted.los[0], rooted.his[0])}, which straddles "
        f"zero — the distinction the two census rows rest on is gone"
    )
    assert rooted.los[0] > 0.0
    assert "sign" in P._SIGN_NO_RULE and "sqrt" in P._SIGN_ARITHMETIC


def test_max_and_min_are_the_two_ROUTING_members_that_are_not_a_bit_copy():
    """F2. The ROUTING class comment claimed, unqualified, that every output
    element IS an element of a value operand and that the class "introduces
    none of its own". True over ℝ, which is all the rule needs; false of the
    EXECUTABLE for exactly two members, because DAZ destroys a subnormal
    operand before the comparison. The comment now says "over ℝ" and names
    them; this is the measurement behind that."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()
    tiny = 5e-324

    def go():
        return (
            float(lax.max(jnp.float64(tiny), jnp.float64(-1.0))),
            float(lax.min(jnp.float64(-tiny), jnp.float64(1.0))),
            float(jnp.reshape(jnp.float64(tiny), ())),
            float(jnp.array(jnp.float64(tiny))),
        )

    mx, mn, reshaped, copied = _at_x64(go)
    assert mx == 0.0 and mx not in (tiny, -1.0), (
        f"lax.max(5e-324, -1.0) = {mx!r}: the class comment's exception"
    )
    assert mn == 0.0 and math.copysign(1.0, mn) == -1.0, (
        f"lax.min(-5e-324, 1.0) = {mn!r} with sign bit "
        f"{math.copysign(1.0, mn)}: it must be the NEGATIVE zero, which is "
        f"what keeps it out of the wrong-signed-zero class (see _t_div)"
    )
    # the control: every other routing member really is a bit copy
    assert reshaped == tiny and copied == tiny, (
        "reshape/copy of a subnormal must be exact, or the class comment's "
        "'exactly two members' is wrong in the other direction"
    )


# --- F4: the sign bit of an executed zero ------------------------------------


def test_boundary_div_tolerates_only_a_MATCHING_signed_zero():
    """F4, the structural half, and it is the reason no false certificate the
    audit could build produced a wrong verdict.

    `boundary_div` drops only the divisor's zero ENDPOINT; at that point IEEE
    division yields ±inf, and each arm returns a box whose infinite end is
    exactly the one a MATCHING-signed zero produces. An opposite-signed zero
    falls OUTSIDE in all four arms — a false VERIFIED on a lower bound and a
    false REFUTED on an upper one. Nothing enforces the matching; this pins
    the asymmetry so that a change to `boundary_div`'s arms cannot quietly
    move which zero is tolerated."""
    def box(lo, hi):
        return iv.IntervalArray(shape=(), los=(lo,), his=(hi,))

    def ieee_div(a, zero):
        return math.copysign(math.inf, a) * math.copysign(1.0, zero)

    arms = [
        ((1.0, 2.0), (0.0, 4.0), +1),
        ((1.0, 2.0), (-4.0, 0.0), -1),
        ((-2.0, -1.0), (0.0, 4.0), +1),
        ((-2.0, -1.0), (-4.0, 0.0), -1),
    ]
    for (alo, ahi), (blo, bhi), cert in arms:
        out = iv.boundary_div(box(alo, ahi), box(blo, bhi))
        lo, hi = out.los[0], out.his[0]
        matching = ieee_div(alo, 0.0 if cert > 0 else -0.0)
        opposite = ieee_div(alo, -0.0 if cert > 0 else 0.0)
        assert lo <= matching <= hi, (
            f"divisor [{blo},{bhi}] under certificate {cert:+d}: the box "
            f"[{lo}, {hi}] excludes {matching}, which is what a "
            f"matching-signed zero divisor produces — the gap the census "
            f"tolerates is no longer free"
        )
        assert not (lo <= opposite <= hi), (
            f"divisor [{blo},{bhi}] under certificate {cert:+d}: the box "
            f"[{lo}, {hi}] CONTAINS {opposite}, which is what an "
            f"opposite-signed zero produces. That is the one route from a "
            f"false certificate to a wrong verdict, and this assertion is "
            f"the record that it was closed by asymmetry rather than by a "
            f"check"
        )


# --- F4 CONTINUED: the table that could not draw its own class --------------


class WrongSignedZeroUnderCertificate(AssertionError):
    """Raised ONLY by the sign-bit comparison in the row runner below.

    A dedicated type because the diverging rows are `xfail`ed and a blanket
    amnesty over `AssertionError` would swallow this table's two VACUITY
    guards as well — the "a certificate was minted" assert and the "an
    executed zero was seen" assert. Under a blanket `xfail` a row that
    stopped minting, or stopped producing a zero, would report XFAIL and
    read green. That is the shape
    `tests/property/test_suite_disclosure.py::
    test_every_xfail_in_the_suite_is_strict_and_narrowed_by_raises`
    forbids in the property suite, applied here by hand because that check
    walks `tests/property/` and does not reach this file.
    """


class ZeroRow:
    """One end-to-end chain: a declared box, a strict assume, a function,
    and the sign bit the certificate claims for its output.

    ``assumed`` and ``want`` are SEPARATE and that is the change that let
    this table draw its own class. They used to be one field — the row
    said ``"lt"`` and the runner used it both to pick the assume direction
    and to pick the certified sign — which made a row whose output sign
    DIFFERS from its input's inexpressible. `neg` downstream of a
    reduction is exactly that row: assumed negative, certified positive.
    """

    def __init__(self, shape, build, bounds, assumed, want, *, producer,
                 diverges=False, why=""):
        self.shape = tuple(shape)
        self.build = build
        self.bounds = bounds
        self.assumed = assumed
        self.want = want
        self.producer = producer
        self.diverges = diverges
        self.why = why


# The rows the census admits whose ℝ value is nonzero where the target's is
# zero, each with the sign bit the certificate claims.
#
# **THIS TABLE USED TO BE SCALAR-ONLY AND THAT IS WHY IT COULD NOT DRAW THE
# DEFECT IT WAS BUILT FOR.** It held exactly four rows, every one of them
# `any_array((), jnp.float64, …)`, and its comment read: *"Not
# hypothetical: these are the three DAZ-band rows named in `_t_div`'s
# standing-constraint paragraph, driven end to end."* Both sentences were
# true and the table was still blind, because the class it exists to detect
# needs a REDUCTION, and a reduction over a scalar is not a reduction:
# `reduce_sum` and `dot_general` are lowered as accumulations SEEDED with
# `+0.0`, and in round-to-nearest `(+0) + (−0) = +0`, so the divergence
# begins at a reduced extent of 2. MEASURED on jax 0.11.0 CPU binary64,
# eager and under `jit` over an array operand:
#
#     jnp.sum([-0.0])                 -> -0.0   n=1 AGREES
#     jnp.sum([-0.0, -0.0])           -> +0.0   n=2 DISAGREES
#     jnp.sum(reshape(...,(2,1)), 1)  -> -0.0   two cells, one term each
#     dot_general([-0.,-0.],[1.,1.])  -> +0.0   DISAGREES
#     dot_general([-0.],[1.])         -> -0.0   n=1 AGREES
#
# The last two lines are why the guard is the REDUCED EXTENT and not the
# operand's size: a `reduce_sum` over an operand of six elements whose
# every output cell sums ONE term keeps the sign bit.
#
# **"OVER AN ARRAY OPERAND" IS A QUALIFIER AND NOT A HEDGE, AND THE COUNT OF
# LOWERINGS IS AT LEAST THREE.** *This paragraph said "compiles two ways" and
# named only `reduce_sum`; both halves were short.* `jit(lambda v: sum(v))`
# over an array keeps the seeded reduction and returns `+0.0`;
# `jit(lambda p, q: sum(stack([p, q])))` is rewritten by XLA into a bare
# `add(p, q)` with no seed and returns `-0.0`; a constant-folded
# `jit(lambda: sum(array([-0.,-0.])))` also returns `-0.0`. **`dot_general`
# splits the same way**, one leg further out — eager and jit-from-scalars
# both `+0.0`, constant-folded `-0.0`. All eight cells are measured in
# `tests/test_executed_sign_bit_sweep.py::
# test_the_SAME_reduction_compiles_AT_LEAST_THREE_WAYS_and_they_disagree_on_the_sign_bit`.
# The rows below evaluate the traced `fn` at points, which is the first
# form; the sign bit of an executed zero is not a function of the IR alone
# and no row here should be read as if it were.
#
# Every row below therefore carries a SHAPE. The four original rows carry
# `()` and are otherwise untouched, so this widening cannot have moved what
# they measured.
#
# **WHAT THIS TABLE DOES NOT REACH.** It samples three points of a declared
# box per row and reads the sign bit of what jax computes there. It is not
# a proof and it is not a search: a row whose divergence needs a point this
# table does not sample is invisible here. The per-primitive question —
# *can this primitive produce a zero whose sign bit is not the certified
# one* — is asked of every carrying primitive, with the operand values
# chosen adversarially rather than sampled from a box, in
# `tests/test_executed_sign_bit_sweep.py`. Nor does anything here reach a
# certificate that is never CONSUMED: `boundary_div` is the one consumer,
# and whether a wrong-signed zero reaches a verdict is
# `test_boundary_div_tolerates_only_a_MATCHING_signed_zero`'s question.
_C = 1e-200
"""Half of the underflow chain: `(x * _C) * _C` is a zero carrying `x`'s own
sign bit for every declared magnitude below a MEASURED bound, so a row's box
can be ordinary and readable instead of subnormal.

The bound, bisected on jax 0.11.0 CPU binary64 against
`float(jnp.float64(x) * 1e-200 * 1e-200) == 0.0`: it flushes up to
`2.225073858507201e+92` and does not at `2.2250738585072013e+92`. Every box
below is between 0.25 and 1, so every row is inside it by ninety decades —
this is a constant chosen with a margin, not a constant that happens to
work."""

ZERO_UNDER_CERTIFICATE = {
    # -- the four original rows, now carrying `shape=()` ------------------
    "mul (the cube that underflows)": ZeroRow(
        (), lambda jnp: (lambda x: x * x * x), (-1e-100, -1e-200), -1, -1,
        producer="mul",
    ),
    "sqrt (a flushed subnormal operand)": ZeroRow(
        (), lambda jnp: (lambda x: jnp.sqrt(x)), (0.0, 1e-310), 1, 1,
        producer="sqrt",
    ),
    "max (a flushed subnormal operand)": ZeroRow(
        (), lambda jnp: (lambda x: jnp.maximum(x, -1.0)), (0.0, 1e-310), 1, 1,
        producer="max",
    ),
    "min (a flushed subnormal operand)": ZeroRow(
        (), lambda jnp: (lambda x: jnp.minimum(x, 1.0)), (-1e-310, 0.0), -1, -1,
        producer="min",
    ),
    # -- the controls that prove the probe is not simply always-red -------
    "reduce_sum n=1 (the control: one term, no seed to lose to)": ZeroRow(
        (1,), lambda jnp: (lambda x: jnp.sum(x * _C * _C)), (-1.0, -0.25),
        -1, -1, producer="reduce_sum",
        why="a one-term sum keeps the sign bit; if this row ever reds, the "
            "probe is broken and not the target",
    ),
    "reduce_sum n=6 reduced to 1 term per cell (the control)": ZeroRow(
        (6,), lambda jnp: (
            lambda x: jnp.sum(jnp.reshape(x * _C * _C, (6, 1)), axis=1)
        ), (-1.0, -0.25), -1, -1, producer="reduce_sum",
        why="six elements, six output cells, ONE term each — the guard is "
            "the reduced extent and not the operand's size",
    ),
    "dot_general n=1 (the control)": ZeroRow(
        (1,), lambda jnp: (
            lambda x: jnp.dot(x * _C * _C, jnp.ones((1,), jnp.float64))
        ), (-1.0, -0.25), -1, -1, producer="dot_general",
    ),
    "reduce_sum n=2 POSITIVE (the safe direction, pinned)": ZeroRow(
        (2,), lambda jnp: (lambda x: jnp.sum(x * _C * _C)), (0.25, 1.0),
        1, 1, producer="reduce_sum",
        why="`(+0) + (+0) = +0` and the seed is `+0.0`, so a POSITIVE "
            "certificate survives the same reduction that destroys a "
            "negative one. This asymmetry is the whole shape of the "
            "eventual repair, and an instrument that cannot show the safe "
            "direction cannot show that the repair kept it",
    ),
    # -- the class itself ---------------------------------------------------
    "reduce_sum n=2 (THE CLASS)": ZeroRow(
        (2,), lambda jnp: (lambda x: jnp.sum(x * _C * _C)), (-1.0, -0.25),
        -1, -1, producer="reduce_sum", diverges=True,
        why="the seeded accumulation: `(+0) + (−0) + (−0) = +0`",
    ),
    "reduce_sum n=5 (the same, wider)": ZeroRow(
        (5,), lambda jnp: (lambda x: jnp.sum(x * _C * _C)), (-1.0, -0.25),
        -1, -1, producer="reduce_sum", diverges=True,
        why="not an n=2 artifact",
    ),
    "dot_general n=2 (THE CLASS)": ZeroRow(
        (2,), lambda jnp: (
            lambda x: jnp.dot(x * _C * _C, jnp.ones((2,), jnp.float64))
        ), (-1.0, -0.25), -1, -1, producer="dot_general", diverges=True,
        why="the same seed, reached through a contraction; the `ones` "
            "operand is a CONSTVAR the propagator certifies `+1`, so the "
            "product's certificate is `-1`",
    ),
    "neg DOWNSTREAM of a diverged reduction": ZeroRow(
        (2,), lambda jnp: (lambda x: -jnp.sum(x * _C * _C)), (-1.0, -0.25),
        -1, 1, producer="neg", diverges=True,
        why="`neg(+0.0) = -0.0`, so a wrong `-1` becomes a wrong `+1` and "
            "the defect travels. The assumed direction is NEGATIVE and the "
            "certified sign is POSITIVE — the row shape the old one-field "
            "table could not express",
    ),
}


def _zero_row_points(row):
    """The sampled points of the declared box, on the ASSUMED side.

    Three scalars — the two endpoints and the midpoint — filtered to the
    assumed half, each broadcast to the row's shape, plus (for a row with
    at least two elements) one NON-UNIFORM point built by alternating them,
    so an all-elements-equal vector is not the only thing the row ever
    sees.
    """
    lo, hi = row.bounds
    scalars = [
        v for v in (lo, hi, (lo + hi) / 2.0)
        if (v > 0 if row.assumed > 0 else v < 0)
    ]
    assert scalars, row.bounds
    n = 1
    for d in row.shape:
        n *= d
    points = [tuple([v] * n) for v in scalars]
    if n > 1 and len(scalars) > 1:
        points.append(tuple(scalars[i % len(scalars)] for i in range(n)))
    return points


_DIVERGING_XFAIL = (
    "OPEN DEFECT, not a gap in this test: under the lowering these rows "
    "execute — an ARRAY operand, eager or jit — XLA accumulates `reduce_sum` "
    "and `dot_general` from a `+0.0` seed, so a value the strict-sign "
    "certificate calls NEGATIVE whose every element flushes to `-0.0` "
    "reduces to a `+0.0`. THE SEED IS A PROPERTY OF THE LOWERING AND NOT OF "
    "THE EQUATION: at least three lowerings of the same equation exist and "
    "they do not agree on the sign bit "
    "(`tests/test_executed_sign_bit_sweep.py::"
    "test_the_SAME_reduction_compiles_AT_LEAST_THREE_WAYS_and_they_disagree"
    "_on_the_sign_bit`). An opposite-signed zero falls outside every arm of "
    "`interval.boundary_div` "
    "(`test_boundary_div_tolerates_only_a_MATCHING_signed_zero`), which "
    "turns it into a false VERIFIED on a lower-bound obligation and a "
    "false REFUTED on an upper one. Disclosed here rather than deleted: "
    "the row is the instrument the SEEDED-REDUCTION REPAIR item will be "
    "measured by, and `strict=True` means the day that repair lands this "
    "goes XPASS and reds instead of passing quietly. Making the rule right "
    "is that item's work and is explicitly NOT this one's."
)


def _zero_row_params():
    """The parametrize list, with the diverging rows marked.

    Derived from `ZERO_UNDER_CERTIFICATE[...].diverges` rather than from a
    second list of names, so a row cannot be reclassified in one place and
    not the other.
    """
    out = []
    for name in sorted(ZERO_UNDER_CERTIFICATE):
        row = ZERO_UNDER_CERTIFICATE[name]
        marks = ()
        if row.diverges:
            marks = (pytest.mark.xfail(
                strict=True,
                raises=WrongSignedZeroUnderCertificate,
                reason=_DIVERGING_XFAIL,
            ),)
        out.append(pytest.param(name, marks=marks))
    return out


@pytest.mark.parametrize("name", _zero_row_params())
def test_an_executed_zero_under_a_certificate_carries_the_CERTIFIED_sign_bit(name):
    """F4, the driven half, and the probe the audit asked for.

    Each row here is a chain the census certifies whose executed value on
    the target is ZERO. That is disclosed and costs no verdict — but only
    while the zero's SIGN BIT agrees with the certificate (see `_t_div`).
    This is the check that would tell the author of a future rule that
    theirs does not, and for four rows it is the check that says the
    SHIPPED rules already do not.

    NOT VACUOUS BY CONSTRUCTION, and the guards are typed so the `xfail`
    cannot eat them: the sign-bit comparison raises
    :class:`WrongSignedZeroUnderCertificate` and everything else raises a
    plain `AssertionError`. So a row that stops minting a certificate, or
    stops producing a zero, or stops being produced by the primitive it
    names, FAILS — including the rows that are expected to xfail."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()
    from stelling.harness import any_array, assert_, assume, trace

    row = ZERO_UNDER_CERTIFICATE[name]
    fn = row.build(jnp)

    def h():
        x = any_array(row.shape, jnp.float64, row.bounds)
        if row.assumed > 0:
            assume(x > 0)
        else:
            assume(x < 0)
        d = fn(x)
        return assert_(d > 0 if row.want > 0 else d < 0)

    cj = _at_x64(lambda: trace(h))
    p = _Propagator("constrain")
    p.run(cj.jaxpr, list(cj.consts), [])

    asserts = [e for e in cj.jaxpr.eqns if e.primitive == "stelling_assert"]
    pred = asserts[0].invars[0]
    cmp_eqn = next(
        e for e in cj.jaxpr.eqns if e.outvars and e.outvars[0].id == pred.id
    )
    value = cmp_eqn.invars[0]
    producer = next(
        (e for e in cj.jaxpr.eqns if any(o.id == value.id for o in e.outvars)),
        None,
    )
    assert producer is not None and producer.primitive == row.producer, (
        f"{name}: the compared value is produced by "
        f"{producer.primitive if producer else None!r}, not by "
        f"{row.producer!r} — this row does not exercise what it names"
    )
    sign = p.strict_sign.get(value.id, 0)
    assert sign == row.want, (
        f"{name}: certified {sign}, expected {row.want} — this row no "
        f"longer exercises the constraint it is here to guard"
    )

    zeros = 0
    for pt in _zero_row_points(row):
        arr = _at_x64(
            lambda pt=pt: jnp.reshape(
                jnp.array(pt, dtype=jnp.float64), row.shape
            )
        )
        vals = _at_x64(lambda arr=arr: [float(v) for v in jnp.ravel(fn(arr))])
        assert vals, (name, pt)
        for v in vals:
            if v == 0.0:
                zeros += 1
                if math.copysign(1.0, v) != float(row.want):
                    raise WrongSignedZeroUnderCertificate(
                        f"{name}: the certificate says {row.want:+d} and the "
                        f"target computes {v!r} at x={pt!r}, whose sign bit "
                        f"is {math.copysign(1.0, v)}. An opposite-signed "
                        f"zero falls outside every arm of boundary_div and "
                        f"is the one route from this gap to a wrong verdict"
                        + (f" — {row.why}" if row.why else "")
                    )
            else:
                assert (v > 0) if row.want > 0 else (v < 0), (name, v, pt)
    assert zeros, (
        f"{name}: no sampled point executed as zero, so this row did not "
        f"exercise the sign-bit constraint at all"
    )


def test_the_zero_table_has_a_control_in_BOTH_directions():
    """The absence half of the table itself.

    A table of only-diverging rows would be indistinguishable from a probe
    that is broken; a table of only-agreeing rows is the table this file
    shipped with, and it could not see the defect. Both halves are required
    here, derived from the rows rather than counted by hand."""
    rows = ZERO_UNDER_CERTIFICATE.values()
    assert any(r.diverges for r in rows), (
        "no row diverges — either the seeded-reduction defect is repaired "
        "(in which case the xfails above are XPASSing and this suite is "
        "already red) or this table lost the rows that saw it"
    )
    assert any(not r.diverges for r in rows), "no row agrees"
    assert any(r.diverges and r.want > 0 for r in rows), (
        "no row shows the defect travelling INTO a positive certificate; "
        "`neg` downstream of a reduction is that row"
    )
    assert any(not r.diverges and r.want > 0 and len(r.shape) == 1
               and r.shape[0] > 1 for r in rows), (
        "no row pins the SAFE direction of a real reduction, so nothing "
        "here could show that a repair kept it"
    )
    assert any(r.shape == () for r in rows), (
        "the scalar rows this table shipped with are gone; the widening was "
        "supposed to keep them"
    )



# --- F3: the decline message may not claim a COUNT it cannot keep -----------


def _conditioned_carriers() -> set[str]:
    """Carriers that DROP with every value operand certified nonzero, derived
    from the probe table above rather than typed."""
    out = set()
    for prim, cases in PROBES.items():
        for c in cases:
            vals = P._sign_value_operands(prim, list(c.signs))
            if vals and all(v for v in vals) and c.want == 0:
                out.add(prim)
                break
    return out


_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)


def test_the_decline_message_makes_no_COUNT_claim_about_the_conditioned_rules():
    """F3. The message used to read "**Two of those** carry it only under a
    side condition on the operands' signs" — in the very sentence whose
    hand-written carrier list this change had just replaced with a derived
    one. Derived from `PROBES`, the count is thirteen.

    A number in that string would be a THIRD place the same fact lives, after
    the rules and the probe table, and the two conditions are not even of one
    kind. So the count is gone rather than corrected, and this refuses its
    return."""
    import re

    msg = P.DIV_BOUNDARY_ZERO_DECLINE
    hit = re.search(rf"\b({_NUMBER_WORDS})\b\s+of\s+those", msg, re.I)
    assert hit is None, (
        f"the decline message counts the conditioned carriers again "
        f"({hit.group(0)!r}); derived from PROBES the count is "
        f"{len(_conditioned_carriers())}, and it is not a number this "
        f"string can keep true"
    )
    assert "SEVERAL of those" in msg, (
        "the message must still say that SOME carriers are conditioned — "
        "dropping the count must not drop the warning"
    )


def test_every_rule_the_message_names_as_conditioned_really_is():
    """The absence half of the test above. Saying "several" is only honest if
    the examples given are examples of the thing."""
    conditioned = _conditioned_carriers()
    assert len(conditioned) > 2, (
        f"only {sorted(conditioned)} are conditioned; the count the message "
        f"used to give would not have been false, and this repair would be "
        f"about nothing"
    )
    msg = P.DIV_BOUNDARY_ZERO_DECLINE
    for named in ("add", "sub"):
        assert f"`{named}` needs" in msg, named
        assert named in conditioned, (
            f"the message offers `{named}` as an example of a conditioned "
            f"carrier and the probe table says it is not one"
        )


def test_the_message_names_every_transfer_in_one_of_its_three_lists():
    """F3's other half: the closing clause used to read as if the carrier and
    no-rule lists were exhaustive over `TRANSFERS`, while 9 of the 11
    boolean-valued transfers appeared nowhere in the string. The three lists
    are now jointly total, derived, and checked here."""
    msg = P.DIV_BOUNDARY_ZERO_DECLINE
    missing = [t for t in sorted(P.TRANSFERS) if t not in msg]
    assert not missing, (
        f"{missing} appear in no list the decline message prints, so a user "
        f"reading it cannot tell whether they carry the certificate"
    )


# --- F3: the narrowing that makes those four xfails non-vacuous, ENFORCED ----


_TESTS_DIR = __import__("pathlib").Path(__file__).resolve().parent


def _xfail_markers_under(root, skip_dir):
    """Every `pytest.mark.xfail(...)` under `root`, as ``(where, kwargs)``.

    AST, and by the CALL rather than by decorator position — two reasons,
    both measured.

    A regex is wrong because `tests/test_skip_inventory.py` carries the text
    `@pytest.mark.xfail(run=False, ...)` inside the SUBJECT SOURCES it writes
    to temp files for its miniature sessions, and a text scan reads those as
    live markers on this suite. Measured: a text scan over `tests/` reports
    seven files, the AST reports one.

    Decorator position alone is wrong because THIS MODULE'S OWN MARKERS ARE
    NOT IN IT. They are built inside `_zero_row_params` and handed to
    `pytest.param(..., marks=marks)` through a local variable, so a walker
    that reads `decorator_list` and `marks=` literals finds zero of them —
    measured, this test failed exactly that way when it was written. What is
    stable is the CALL: `pytest.mark.xfail(...)` is the thing the rule is
    about, wherever it is spelled.
    """
    import ast

    out = []
    for path in sorted(root.rglob("test_*.py")):
        if skip_dir in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken test file reds first
            continue
        enclosing = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    enclosing.setdefault(id(child), node.name)
        rel = path.relative_to(_TESTS_DIR.parent)
        # the `func` of a Call is itself an Attribute node that `ast.walk`
        # visits, so a called marker would be counted twice — once as the
        # call and once as the bare attribute. The bare form (`@pytest.mark
        # .xfail` with no parentheses) still has to be caught, since it
        # carries none of the three required kwargs, so the called ones are
        # subtracted rather than the bare ones ignored.
        called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and id(node) in called:
                continue
            func = node.func if isinstance(node, ast.Call) else node
            if not (isinstance(func, ast.Attribute) and func.attr == "xfail"):
                continue
            chain, cur = [], func.value
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chain.append(cur.id)
            if "mark" not in chain:
                continue
            kwargs = (
                {k.arg: k.value for k in node.keywords}
                if isinstance(node, ast.Call) else {}
            )
            out.append((
                f"{rel}::{enclosing.get(id(node), '<module>')}", kwargs,
            ))
    return out


def test_every_xfail_OUTSIDE_the_property_suite_is_strict_and_narrowed():
    """THE RULE THAT WAS APPLIED BY HAND, APPLIED BY A CHECK INSTEAD.

    `tests/property/test_suite_disclosure.py::
    test_every_xfail_in_the_suite_is_strict_and_narrowed_by_raises` states
    the rule — strict, narrowed to an exception type, carrying a reason —
    and says in its own docstring that it "applies to every xfail in the
    tree rather than to one named test". **Its implementation walks
    `tests/property/` only**, and it is gated on `importorskip("hypothesis")`,
    which is absent from all three merge lanes. So the four `xfail` markers
    this module carries — MEASURED by the AST walk above, the only live ones
    anywhere outside `tests/property/` — stood under a comment saying the
    rule had been "applied by hand", and by nothing else.

    Applied by hand means enforced by nothing. DRIVEN: with
    `raises=WrongSignedZeroUnderCertificate` deleted AND the class row broken
    so it executes no zero at all, this file came back
    `164 passed, 4 xfailed` and `test_suite_disclosure.py` plus
    `test_skip_inventory.py` came back `81 passed, 1 skipped`. A blanket
    amnesty over `AssertionError` swallows this table's own vacuity guards
    and nothing objected.

    This check is here rather than in the property suite for one reason: it
    must RUN in the lanes that gate a merge, and it needs neither jax nor
    hypothesis to do it. The two checks are asserted jointly TOTAL over
    `tests/` below, so the division of labour cannot leave a gap."""
    bad = []
    for nodeid, kw in _xfail_markers_under(_TESTS_DIR, "property"):
        import ast as _ast

        strict = kw.get("strict")
        if not (isinstance(strict, _ast.Constant) and strict.value is True):
            bad.append(
                f"{nodeid}: xfail is not `strict=True`. A non-strict xfail "
                f"passes silently the day the defect is fixed, which is the "
                f"one day it must not be the first to notice."
            )
        if "raises" not in kw:
            bad.append(
                f"{nodeid}: xfail carries no `raises=`, so it is a blanket "
                f"amnesty over every exception the test can raise — its "
                f"vacuity guards included, which then fail GREEN."
            )
        if not kw.get("reason"):
            bad.append(
                f"{nodeid}: xfail carries no `reason=`, so the run's summary "
                f"line does not say what is not being checked."
            )
    assert not bad, "\n  ".join(["xfail markers outside tests/property/:", *bad])


def test_this_module_is_where_those_markers_ARE_so_the_check_is_not_vacuous():
    """The absence half. A walker that finds nothing passes for free.

    Both directions: this module must still carry markers for the check above
    to be about anything, and the count is DERIVED from the diverging rows
    rather than typed, so adding a diverging row without a marker reds."""
    found = _xfail_markers_under(_TESTS_DIR, "property")
    mine = [n for n, _ in found if "test_strict_sign_census.py::" in n]
    assert len(mine) == 1, (
        f"this module builds its markers at ONE `pytest.mark.xfail(...)` "
        f"site inside `_zero_row_params`, and the walk finds {len(mine)}: "
        f"{mine}. The check above is measuring something other than this "
        f"table."
    )
    # ...and that one site is what marks every diverging row, derived
    want = sum(1 for r in ZERO_UNDER_CERTIFICATE.values() if r.diverges)
    marked = sum(
        1 for prm in _zero_row_params()
        if getattr(prm, "marks", ())
    )
    assert marked == want > 0, (
        f"{want} row(s) declare `diverges` and {marked} carry a mark; the "
        f"one site above is not reaching every diverging row"
    )
    assert len(found) == len(mine), (
        f"an xfail marker appeared outside `tests/property/` and outside this "
        f"module: {sorted(n for n, _ in found if n not in mine)}. That is "
        f"fine — but it is now this check's business, and this assertion is "
        f"how its author finds that out."
    )


def test_the_two_xfail_WALKERS_are_jointly_total_over_the_tests_tree():
    """Neither check may assume the other's scope.

    The property suite's walker takes `tests/property/`; the one above takes
    everything else. Total by construction only while both predicates are the
    complement of each other, so the complement is asserted here rather than
    trusted — over the file set pytest would actually collect."""
    everywhere = _xfail_markers_under(_TESTS_DIR, "\0no-such-part\0")
    outside = _xfail_markers_under(_TESTS_DIR, "property")
    inside = [n for n, _ in everywhere if "property/" in n.replace("\\", "/")]
    assert len(everywhere) == len(outside) + len(inside), (
        f"{len(everywhere)} xfail markers under `tests/`, {len(outside)} "
        f"claimed by this module's walker and {len(inside)} by the property "
        f"suite's — the two do not partition the tree and a marker is "
        f"governed by neither"
    )
    assert inside, (
        "the property suite carries no xfail marker, so the walker this one "
        "divides labour with has nothing to do and the division is a claim "
        "about nothing"
    )


def test_the_seeded_reduction_sign_bit_has_a_witness_that_is_NOT_an_amnesty():
    """THE SIGN-BIT FACT, ASSERTED POSITIVELY, in an ordinary green test.

    The four rows above disclose the defect through `xfail(strict=True)`,
    which is the right shape for "this obligation is violated and the repair
    will make it pass" — but an xfail is an ABSENCE of a verdict. Every
    statement it makes is made by not raising, and this project's own rule is
    that an instrument whose evidence is an absence is the weak form.

    So the underlying fact about the TARGET is stated here as a presence, in
    a test that is green today, has no marker on it, and cannot be silenced
    by widening an amnesty. It is deliberately a claim about jax and NOT
    about the certificate: it does not read `strict_sign`, so the day the
    SEEDED-REDUCTION REPAIR lands this keeps passing (the rule will change,
    the lowering will not) and the four xfails become XPASS and red. One
    tripwire on the repair, not five.

    Measured on jax 0.11.0 CPU binary64, and re-measured on 0.10.2 with the
    same answers."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()

    def go():
        neg2 = jnp.array([-0.0, -0.0], dtype=jnp.float64)
        neg1 = jnp.array([-0.0], dtype=jnp.float64)
        pos2 = jnp.array([0.0, 0.0], dtype=jnp.float64)
        ones = jnp.array([1.0, 1.0], dtype=jnp.float64)
        dot = lambda a, b: lax.dot_general(a, b, (((0,), (0,)), ((), ())))
        return {
            "sum n=2 eager": float(jnp.sum(neg2)),
            "sum n=2 jit": float(jax.jit(jnp.sum)(neg2)),
            "sum n=1 eager": float(jnp.sum(neg1)),
            "sum n=1 jit": float(jax.jit(jnp.sum)(neg1)),
            "sum n=2 POSITIVE": float(jnp.sum(pos2)),
            "dot n=2 eager": float(dot(neg2, ones)),
            "dot n=2 jit": float(jax.jit(dot)(neg2, ones)),
        }

    got = _at_x64(go)
    for key, value in got.items():
        assert value == 0.0, f"{key} is {value!r}, not a zero at all"

    # THE FACT: a reduction of two negative zeros carries the PLUS sign bit
    for key in ("sum n=2 eager", "sum n=2 jit",
                "dot n=2 eager", "dot n=2 jit"):
        assert math.copysign(1.0, got[key]) > 0, (
            f"{key} is {_show_zero(got[key])}. The seeded-reduction class is "
            f"this fact; if it has gone, the four xfail rows above are "
            f"XPASSing and this suite is already red — check there before "
            f"believing the target changed."
        )
    # ...and the two controls that make it a fact about the SEED and not
    # about reductions in general
    for key in ("sum n=1 eager", "sum n=1 jit"):
        assert math.copysign(1.0, got[key]) < 0, (
            f"{key} is {_show_zero(got[key])}; a one-term reduction must "
            f"keep the sign bit, or the boundary is not the reduced extent"
        )
    assert math.copysign(1.0, got["sum n=2 POSITIVE"]) > 0, (
        "a POSITIVE-certified reduction no longer keeps its sign bit either, "
        "which would make this a symmetric defect rather than the asymmetric "
        "one the repair is shaped around"
    )


def _show_zero(v: float) -> str:
    return "-0.0" if math.copysign(1.0, v) < 0 else "+0.0"
