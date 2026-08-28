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
    """WRITER 1's empty case, and it was a live hole: `ks` is the
    per-element bound list, so on an empty target `all(k >= 0.0 for k in
    ks)` is vacuously true and the arm used to write `+1` for a value with
    no elements."""
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
    """WRITER 3's empty case. `slice(x, [0], [0])` over a certified `x` has
    a size-0 output, and the routing rule's operand agreement holds — the
    only thing refusing the vacuous certificate is the output-side rule."""
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
