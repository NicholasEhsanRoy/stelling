# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The even-power non-negativity rule must not be applied to complex operands.

Found by a blinded class-level audit of the `square` row. jax declares
``square_p`` as ``standard_unop(_int | _float | _complex, 'square')`` — complex
is in its domain — and `interval.integer_pow`'s even-exponent branch returns
``[0, +inf]`` for a ⊤ operand. So a complex square produced a definite
non-negativity claim that complex squaring does not satisfy, recorded as KNOWN
coverage rather than declined.

`integer_pow` had the identical hole (it guards dtype only for a NEGATIVE
exponent), which is where `square` inherited it by delegating. Both are pinned
here.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp
import numpy as np

from stelling import interval as iv
from stelling import propagate as P
from stelling._jax_compat import transcribe
from stelling.harness import any_array


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _box_for(prim, fn):
    """The propagated box for `prim`'s output on a complex operand."""
    def build():
        x = any_array((), "float64", (1.0, 2.0))
        return (fn(x.astype(jnp.complex128) * 1j),)

    cj = transcribe(jax.make_jaxpr(build)())
    env = P.interval_env(cj)
    eqn = [e for e in cj.jaxpr.eqns if str(e.primitive) == prim]
    assert eqn, f"fixture did not produce a {prim!r} equation"
    return env.get(eqn[0].outvars[0].id)


FORMS = {"square": lambda z: jnp.square(z), "integer_pow": lambda z: z ** 2}


@pytest.mark.parametrize("prim", sorted(FORMS))
def test_complex_operand_does_not_get_the_non_negativity_rule(prim):
    box = _box_for(prim, FORMS[prim])
    if box is None:
        return  # declined outright: sound
    lo, hi = box.los[0], box.his[0]
    assert lo == float("-inf"), (
        f"{prim!r} claimed a lower bound of {lo} on a COMPLEX operand. The "
        f"even-power non-negativity rule is a real-arithmetic fact; a pure "
        f"imaginary squares to a NEGATIVE REAL, which that bound excludes."
    )


@pytest.mark.parametrize("prim", sorted(FORMS))
def test_the_excluded_value_is_real_and_reachable(prim):
    """THE CONSEQUENCE, pinned so the guard above is known load-bearing.

    Executes the program and shows the truth is a real negative — so this is
    not an interpretive question about complex-vs-interval containment.
    """
    for xv in (1.0, 1.5, 2.0):
        z = jnp.asarray(xv, jnp.float64).astype(jnp.complex128) * 1j
        got = complex(np.asarray(FORMS[prim](z)))
        assert got.imag == 0.0, "the fixture should produce a purely real value"
        assert got.real < 0.0, f"expected a negative real, got {got}"


def test_the_probe_can_see_the_defect_it_checks_for(monkeypatch):
    """Anti-vacuity. Remove the guard and the false box must come back —
    otherwise these tests pass for a reason unrelated to it."""
    monkeypatch.setattr(P, "_refuse_complex", lambda eqn, prim: None)
    box = _box_for("square", FORMS["square"])
    assert box is not None and box.los[0] == 0.0, (
        "with the guard removed the defect must reappear as a [0, inf] box; "
        "if it does not, this file is not testing what it claims to"
    )


def test_real_and_integer_operands_are_unaffected():
    """Don't fix the bad path by breaking the good one."""
    a = iv.IntervalArray(shape=(1,), los=(-2.0,), his=(3.0,))
    assert iv.integer_pow(a, 2).los[0] == 0.0   # straddling -> [0, 9]
    assert iv.integer_pow(a, 2).his[0] == 9.0

    def build():
        x = any_array((3,), "float64", (-2.0, 3.0))
        return (jnp.square(x),)

    cj = transcribe(jax.make_jaxpr(build)())
    env = P.interval_env(cj)
    eqn = [e for e in cj.jaxpr.eqns if str(e.primitive) == "square"][0]
    box = env.get(eqn.outvars[0].id)
    assert box is not None, "a real square must still be admitted"
    assert box.los[0] == 0.0 and box.his[0] == 9.0
