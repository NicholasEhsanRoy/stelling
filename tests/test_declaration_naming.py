# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""ONE NUMBERING FOR ONE INPUT — audit 0.2.0 B8a, item 5 (M3).

The emission names the k-th declaration `x{k}` (`obligation.SliceInput`,
`smt.emit`, `reproduce`, and the witness a REFUTED prints). `propagate`'s
assume messages named their subject `var {atom.id}` — the internal IR id —
and the two numberings genuinely disagree. Measured on `aabb58d`, on the
two-declaration query below:

    declaration k=0  ->  IR var id 1   witness name x0
    declaration k=1  ->  IR var id 2   witness name x1

    UnsatisfiableAssumptionError: ... var 2 has propagated interval
    [0.0, 1.0], but the assumed constraint requires var 2 >= 2.0 ...

"var 2" for the input the witness calls `x1`, beside a `var 1` that would
have meant the input the witness calls `x0`. Two 0-based namespaces, one
reader, nothing relating them.

What is pinned here:

1. the message names the declaration the way the witness will;
2. `propagate._declaration_names` and `obligation._Slicer.any_order` assign
   the SAME index to the same declaration, including one inside a `jit`
   body — the case where the two walks could most easily have diverged;
3. where they could diverge, the numbering is ABANDONED and the message
   falls back to an id spelled so it cannot be read as a declaration index.
"""

from __future__ import annotations

import pytest

from stelling import ir
from stelling.coverage import declaration_name
from stelling.propagate import _declaration_names

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")


def var(i, aval=F64):
    return ir.Var(id=i, aval=aval)


def any_eqn(out, lo=0.0, hi=1.0):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", "float64"), ("lo", lo), ("hi", hi)),
    )


def test_a_declaration_under_a_cond_abandons_the_numbering():
    """The propagator descends a `cond`; the slicer never does. A
    declaration in a branch has no `x{k}` at all, so which of the survivors
    is #0 would be a guess — and a guess printed as a witness name is worse
    than an internal id."""
    branch = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(var(20),),
            outvars=(var(21),),
            eqns=(any_eqn(var(21)),),
        )
    )
    j = ir.Jaxpr(
        constvars=(),
        invars=(),
        outvars=(var(0),),
        eqns=(
            any_eqn(var(0), 1.0, 2.0),
            ir.JaxprEqn(
                primitive="cond",
                invars=(var(0), var(0)),
                outvars=(var(3),),
                params=(("branches", (branch, branch)),),
            ),
        ),
    )
    assert _declaration_names(j) is None
    # ... and with no such declaration, the numbering stands
    plain = ir.Jaxpr(
        constvars=(), invars=(), outvars=(var(0),), eqns=(any_eqn(var(0)),)
    )
    assert _declaration_names(plain) == {((), 0): "x0"}


def test_a_wrapper_the_slicer_will_not_inline_abandons_the_numbering():
    """The propagator's descent test checks invar arity; the slicer's also
    checks outvar arity and that the body's constvars pair with its consts.
    A wrapper only one of them inlines shifts every later index."""
    body = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(var(20),),
            outvars=(var(21), var(21)),  # two outvars, the call has one
            eqns=(any_eqn(var(21)),),
        )
    )
    j = ir.Jaxpr(
        constvars=(),
        invars=(),
        outvars=(var(3),),
        eqns=(
            ir.JaxprEqn(
                primitive="jit",
                invars=(var(0),),
                outvars=(var(3),),
                params=(("jaxpr", body),),
            ),
            any_eqn(var(0)),
        ),
    )
    assert _declaration_names(j) is None


# -- the two walks agree, including through a transparent wrapper -------------
#
# jax is imported INSIDE these tests, not at module scope: the two cases
# above are zero-dep and the zero-dep lane must keep running them (a
# module-level `importorskip` skips the whole file).

def _fixtures():
    """The three harnesses, built with jax in hand."""
    import jax
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_, assume

    @jax.jit
    def declares_inside(v):
        w = any_array((), jnp.float64, (0.0, 1.0))
        return v + w

    def two_declarations():
        x = any_array((), jnp.float64, (1.0, 2.0))
        y = any_array((), jnp.float64, (0.0, 1.0))
        return assert_(x + y <= 5.0)

    def declaration_inside_a_jit():
        x = any_array((), jnp.float64, (1.0, 2.0))
        z = declares_inside(x)
        y = any_array((), jnp.float64, (0.0, 1.0))
        return assert_(z + y <= 5.0)

    def unsatisfiable_on_the_second_declaration():
        x = any_array((), jnp.float64, (1.0, 2.0))
        y = any_array((), jnp.float64, (0.0, 1.0))
        assume(y >= 2.0)  # empty over [0, 1]
        return assert_(x + y <= 0.5)

    return {
        "top-level-only": two_declarations,
        "one-declaration-inside-a-jit": declaration_inside_a_jit,
        "unsatisfiable": unsatisfiable_on_the_second_declaration,
    }


@pytest.mark.parametrize(
    "which", ["top-level-only", "one-declaration-inside-a-jit"]
)
def test_propagate_and_the_slicer_number_declarations_identically(which):
    """REDDENS ON DRIFT of either walk.

    `_Slicer._flatten` renumbers the ids it meets inside a wrapper, so the
    two maps cannot be compared key for key. What must agree is the INDEX
    each declaration gets, so the comparison is: the top-level ids map to
    the same names in both, and the multiset of names is the same.
    """
    jax = pytest.importorskip("jax")

    from stelling import obligation as OB
    from stelling.harness import trace
    from stelling.propagate import interval_env

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        cj = trace(_fixtures()[which])
    finally:
        jax.config.update("jax_enable_x64", old)
    names = _declaration_names(cj.jaxpr)
    assert names is not None, "this query's numbering must be derivable"
    sl = OB._Slicer(cj, interval_env(cj), ())
    slicer = {vid: declaration_name(k) for vid, k in sl.any_order.items()}

    top_level = {vid: n for (path, vid), n in names.items() if path == ()}
    for vid, name in top_level.items():
        assert slicer.get(vid) == name, (
            f"top-level declaration {vid} is {name} to propagate and "
            f"{slicer.get(vid)!r} to the slicer"
        )
    assert sorted(names.values()) == sorted(slicer.values()), (
        f"the two walks numbered different declaration sets: "
        f"{sorted(names.values())} vs {sorted(slicer.values())}"
    )


def test_the_message_names_the_declaration_the_witness_will_name():
    """The finding, driven. `y` is declaration #1 — witness name `x1` — and
    IR var 2, and the message used to print only the 2."""
    jax = pytest.importorskip("jax")

    from stelling.harness import trace
    from stelling.propagate import UnsatisfiableAssumptionError, propagate

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        cj = trace(_fixtures()["unsatisfiable"])
    finally:
        jax.config.update("jax_enable_x64", old)
    decls = [e for e in cj.jaxpr.eqns if e.primitive == "stelling_any"]
    assert decls[1].outvars[0].id != 1, (
        "this fixture only says something while the IR id and the "
        "declaration index disagree"
    )
    with pytest.raises(UnsatisfiableAssumptionError) as exc:
        propagate(cj)
    message = str(exc.value)
    assert f"x1 (IR var {decls[1].outvars[0].id})" in message, message
    # ... and no bare `var N` survives in it to be misread as a witness name
    assert " var 2 " not in message, message
