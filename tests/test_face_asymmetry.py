# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Face asymmetry: the signal a survivor count cannot carry.

"Caught" is DISJUNCTIVE — any gate declining a mutation is enough — so a
regression confined to one gate's surface leaves the survivor count at zero,
because the other gate still catches the mutation. Extending a gauge to a
second face therefore does NOT, on its own, close the hole that motivated the
extension. What survives that is two gates disagreeing about one subject.

The scatter case is the concrete one: `_scatter_indices_dtype` is called from
`_t_scatter` and `_t_scatter_add` and from nowhere else, so overriding it is a
precisely transfer-only regression with the emission face untouched.
"""
from __future__ import annotations

import pytest

from stelling.fidelity import gauge

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp

from stelling import propagate as P
from stelling._jax_compat import transcribe
from stelling.harness import any_array, assert_
from stelling.interval import IntervalArray

_SCOPE = "both faces of the scatter rows: the emission plans and the transfers"


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _eqn(n, idx_dtype):
    """A traced scatter-add whose index array carries `idx_dtype`."""
    def build(x, idx, v):
        return jax.lax.scatter_add(
            x, idx, v,
            jax.lax.ScatterDimensionNumbers(
                update_window_dims=(), inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,)),
            indices_are_sorted=True, unique_indices=True,
            mode=jax.lax.GatherScatterMode.FILL_OR_DROP)

    cj = transcribe(jax.make_jaxpr(build)(
        jnp.zeros(n, jnp.float64), jnp.zeros((1,), idx_dtype),
        jnp.zeros((), jnp.float64)))
    return [e for e in cj.jaxpr.eqns if str(e.primitive) == "scatter-add"][0]


def _transfer_admits(eqn):
    n = eqn.invars[0].aval.shape[0]
    out = P._t_scatter_add(eqn, dict(eqn.params_dict()), [
        IntervalArray(shape=(n,), los=(0.0,) * n, his=(1.0,) * n),
        IntervalArray(shape=(1,), los=(1.0,), his=(1.0,)),
        IntervalArray(shape=(), los=(1.0,), his=(1.0,)),
    ])
    return out is not None


def _emission_admits(eqn):
    """The emission face's own dtype question, read the way it reads it."""
    return P._scatter_index_dtype_covers(
        eqn.invars[1].aval.dtype, eqn.invars[0].aval.shape[0])


GATES = {
    "emission": lambda e: _emission_admits(e),
    "transfer": lambda e: _transfer_admits(e),
}
# n=129 with an int8 index: the bound 128 wraps, so jax DROPS the write while
# the row models it as landing. Both faces must decline it.
MUTATIONS = {"idx-int8-n129": lambda: _eqn(129, jnp.int8)}


def _run():
    base = _eqn(4, jnp.int32)
    return gauge(base, GATES, {k: v() for k, v in MUTATIONS.items()},
                 residual={}, scope=_SCOPE)


def test_both_faces_agree_and_there_is_no_asymmetry():
    report = _run()
    assert report.residual == ()
    assert report.asymmetries == (), (
        "the two faces disagree about a scatter-add: they have drifted apart"
    )


def test_a_transfer_only_regression_shows_up_as_asymmetry_not_as_a_survivor(
        monkeypatch):
    """THE ANTI-VACUITY COMPANION for the asymmetry instrument itself.

    Without this, "0 asymmetries" and "this report cannot see an asymmetry"
    are the same output — which is the exact failure the asymmetry check was
    added to catch, one level up.
    """
    # transfer-only: this function is called from the two transfers and from
    # nowhere else, so the emission face is untouched by the override
    monkeypatch.setattr(P, "_scatter_indices_dtype", lambda eqn: "int64")

    report = _run()

    # the survivor count is BLIND to it — the emission face still catches
    assert report.residual == (), "emission should still catch the mutation"
    survivors = [n for n, c in report.caught_by if not c]
    assert survivors == [], (
        "a one-face regression must NOT show up as a survivor; if it does, "
        "this test is no longer demonstrating the thing it exists to "
        "demonstrate"
    )

    # ...and the asymmetry check is what sees it
    assert len(report.asymmetries) == 1
    name, caught, admitted = report.asymmetries[0]
    assert name == "idx-int8-n129"
    assert caught == ("emission",)
    assert admitted == ("transfer",)
    assert "ADMITTED by transfer" in report.render()


def test_asymmetry_is_empty_for_a_single_gate_rather_than_a_vacuous_zero():
    base = _eqn(4, jnp.int32)
    report = gauge(base, {"emission": GATES["emission"]},
                   {k: v() for k, v in MUTATIONS.items()},
                   residual={}, scope=_SCOPE)
    assert report.asymmetries == ()
    # and it must not claim to have looked
    assert "face asymmetry" not in report.render()
