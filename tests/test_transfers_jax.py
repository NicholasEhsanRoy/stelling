# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The registration round end-to-end: real traces bind the real names.

A transfer keyed on a name jax never emits would silently never fire, so
each new row is exercised from an actual traced harness. Skipped without
jax.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.verdict import make_verdict  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def run(h):
    return propagate(trace(h))


def test_abs_traced():
    def h():
        x = any_array((), "float64", (-3.0, 2.0))
        return assert_(jnp.abs(x) <= 3.0)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert ("abs", "exact") in p.transfers_used


def test_eq_ne_traced():
    def h():
        x = any_array((), "float64", (2.0, 2.0))
        return assert_(x == 2.0), assert_(x != 3.0)

    p = run(h)
    assert [o.status for o in p.obligations] == ["discharged", "discharged"]
    used = dict(p.transfers_used)
    assert used.get("eq") == "exact" and used.get("ne") == "exact"


def test_logical_and_or_traced():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        t = x >= 1.0  # definitely true
        u = x >= 1.5  # unknown
        return assert_(jnp.logical_or(t, u)), assert_(jnp.logical_and(t, x <= 2.0))

    p = run(h)
    assert [o.status for o in p.obligations] == ["discharged", "discharged"]
    used = dict(p.transfers_used)
    assert used.get("and") == "exact" and used.get("or") == "exact"


def test_bitwise_integer_and_declines_on_real_trace():
    def h():
        x = any_array((), "int32", (0.0, 7.0))
        y = any_array((), "int32", (0.0, 7.0))
        return assert_((x & y) <= 7)

    p = run(h)  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("and" in n and "declined" in n for n in p.notes)


def test_stop_gradient_traced():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(jax.lax.stop_gradient(x) >= 1.0)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert ("stop_gradient", "exact") in p.transfers_used


def test_reshape_traced():
    def h():
        x = any_array((2, 3), "float64", (0.0, 1.0))
        return assert_(jnp.reshape(x, (3, 2)) <= 1.0)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert ("reshape", "exact") in p.transfers_used


def test_pow_traced_and_assumption_rides_into_stamp():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        y = any_array((), "float64", (1.0, 2.0))
        return assert_(x**y <= 4.001)

    cj = trace(h)
    p = propagate(cj)
    assert p.obligations[0].status == "discharged"
    assert ("pow", "sound-libm") in p.transfers_used
    v = make_verdict(
        cj,
        p,
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "VERIFIED"
    assert any("pow" in a and "libm" in a for a in v.stamp.assumptions)
    assert ("pow", "sound-libm") in v.stamp.transfer_tiers


def test_where_scalar_predicate_traced():
    # h_clean's own decline note: scalar selector, array cases — registered now
    def h():
        pr = any_array((), "bool", (0.0, 1.0))
        a = any_array((2,), "float64", (0.0, 1.0))
        b = any_array((2,), "float64", (2.0, 3.0))
        return assert_(jnp.where(pr, a, b) <= 3.0)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert not any("select_n" in n for n in p.notes)  # no decline anymore


def test_reduce_or_traced():
    def h():
        x = any_array((2, 3), "bool", (1.0, 1.0))  # definitely all true
        return assert_(jnp.any(x, axis=0))

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert ("reduce_or", "exact") in p.transfers_used


def test_vmap_batched_mul_rank_broadcasts():
    # vmap batching emits mul on (2,1) vs (1,1): the size-1 broadcast form
    # that previously declined (the h_hard notes)
    def h():
        x = any_array((2,), "float64", (0.0, 1.0))
        y = any_array((1,), "float64", (0.0, 2.0))
        z = jax.vmap(lambda a, b: a * b, in_axes=(0, None))(x, y)
        return assert_(z <= 2.001)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert p.coverage.unknown == 0
    assert not any("declined" in n for n in p.notes)


def test_at_set_scatter_traced():
    # the maddening heat-node round's census addition must bind the real
    # name jax emits for x.at[k].set(v) — scatter — or it never fires
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = x.at[1].set(5.0)
        return assert_(y <= 5.0)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert ("scatter", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_at_set_scatter_definite_false_traced():
    # the written element is definitely above the bound: sound refutation
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = x.at[1].set(5.0)
        return assert_(y <= 2.0)

    p = run(h)
    assert p.obligations[0].status == "violated-over-set"


def test_at_set_dynamic_index_declines_traced():
    # a traced (non-point) index reaches the transfer as a real interval:
    # it must decline to a noted ⊤, not guess a landing site
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        i = any_array((), "int32", (0.0, 2.0))
        y = x.at[i].set(5.0)
        return assert_(y <= 5.0)

    p = run(h)
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown >= 1


def test_fvm_gather_static_row_traced():
    # the MIME fvm round's census addition must bind the real name jax
    # emits for x[const_idx] — gather — or it never fires
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = x.at[1].set(5.0)
        # int32 index, as mesh topology arrays carry it — the int64 default
        # would route through a narrowing convert_element_type that soundly
        # ⊤s the indices before the gather ever sees them
        z = y[jnp.array([1], dtype=jnp.int32)]
        return assert_(z <= 5.0)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert ("gather", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_fvm_gather_static_row_definite_false_traced():
    # the gathered element is definitely above the bound: sound refutation
    # proving the take landed on exactly the written element
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = x.at[1].set(5.0)
        z = y[jnp.array([1], dtype=jnp.int32)]
        return assert_(z <= 2.0)

    p = run(h)
    assert p.obligations[0].status == "violated-over-set"


def test_fvm_gather_dynamic_index_declines_traced():
    # a traced (non-point) index reaches the transfer as a real interval:
    # decline to a noted ⊤ — clamp/drop/fill is mode-dependent, never guessed
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        i = any_array((1,), "int32", (0.0, 2.0))
        return assert_(x[i] <= 1.0)

    p = run(h)
    assert p.obligations[0].status == "unknown"
    assert any("gather" in n and "no sound rule" in n for n in p.notes)
    assert "gather" not in dict(p.transfers_used)


def test_fvm_transpose_traced():
    # transpose must bind the real primitive name; the point row's values
    # land in exactly the transposed positions
    def h():
        top = any_array((1, 2), "float64", (0.0, 1.0))
        bottom = any_array((1, 2), "float64", (5.0, 5.0))
        xt = jnp.transpose(jnp.concatenate([top, bottom], axis=0))
        return assert_(xt <= 5.0), assert_(xt <= 2.0)

    p = run(h)
    assert [o.status for o in p.obligations] == [
        "discharged",
        "violated-over-set",
    ]
    assert ("transpose", "exact") in p.transfers_used
    # the bottom row's two 5.0s occupy column 1 of the (2, 2) transpose
    assert "2/4" in p.obligations[1].detail


# --- split: the traced form, and why its declines are hand-IR-only -----------


def test_split_traced_discharges_and_carries_the_params_the_row_reads():
    def h():
        x = any_array((5,), "float64", (0.0, 1.0))
        a, b = jax.lax.split(x, (2, 3))
        return assert_(jnp.concatenate([a, b]) <= 1.0)

    cj = trace(h)
    # the traced form the row is written against: one operand, integer
    # sizes and axis params present, one output per piece — the facts the
    # transfer's decline messages lean on
    (sp,) = [e for e in cj.jaxpr.eqns if str(e.primitive) == "split"]
    prm = dict(sp.params)
    assert len(sp.invars) == 1
    assert prm["sizes"] == (2, 3) and prm["axis"] == 0
    assert len(sp.outvars) == len(prm["sizes"])

    p = propagate(cj)
    assert p.obligations[0].status == "discharged"
    assert ("split", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_split_malformed_params_cannot_be_traced():
    """The reachability claim behind the split row's named declines,
    measured: jax validates the split params against the operand at trace
    time, so mis-summing sizes, a negative size and an out-of-range axis
    never reach the transfer from a traced program. Their named declines
    are exercised by hand-built IR in test_transfers.py."""
    x = jnp.zeros((5,))
    for bad in (
        lambda: jax.lax.split(x, (2, 2)),        # sizes do not sum to 5
        lambda: jax.lax.split(x, (6, -1)),       # negative size
        lambda: jax.lax.split(x, (5,), axis=3),  # axis out of range
    ):
        with pytest.raises(ValueError):
            jax.make_jaxpr(bad)()
