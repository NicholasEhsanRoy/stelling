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
    # decline to a noted ⊤ — clamp/drop/fill is mode-dependent, never
    # guessed. The note used to read "no sound rule"; strengthened — it
    # must name the element and print ITS declared span.
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        i = any_array((1,), "int32", (0.0, 2.0))
        return assert_(x[i] <= 1.0)

    p = run(h)
    assert p.obligations[0].status == "unknown"
    assert any(
        "'gather' declined this form" in n
        and "index element 0 spans [0.0, 2.0]" in n
        for n in p.notes
    ), p.notes
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


# --- unstack: the traced form, and why its declines are hand-IR-only ---------


def test_unstack_traced_discharges_and_carries_the_facts_the_row_reads():
    def h():
        x = any_array((3, 2), "float64", (0.0, 1.0))
        a, b, c = jnp.unstack(x)
        return assert_(jnp.stack([a, b, c]) <= 1.0)

    cj = trace(h)
    (us,) = [e for e in cj.jaxpr.eqns if str(e.primitive) == "unstack"]
    prm = dict(us.params)
    assert len(us.invars) == 1
    assert prm["axis"] == 0
    assert len(us.outvars) == us.invars[0].aval.shape[prm["axis"]]

    p = propagate(cj)
    assert p.obligations[0].status == "discharged"
    assert ("unstack", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_unstack_malformed_forms_cannot_be_traced():
    """The reachability claim behind the unstack row's named declines,
    measured: the axis is validated (and a negative one normalized) at
    trace time, and the abstract eval binds one output per index — so no
    traced program reaches those paths. They are exercised by hand-built
    IR in test_transfers.py."""
    x = jnp.zeros((3, 2))
    for bad in (
        lambda: jnp.unstack(x, axis=5),               # axis out of range
        lambda: jnp.unstack(jnp.float64(1.0)),        # rank-0 operand
    ):
        with pytest.raises(ValueError):
            jax.make_jaxpr(bad)()
    (e,) = jax.make_jaxpr(lambda: jnp.unstack(x, axis=-1))().eqns
    assert e.params["axis"] == 1  # normalized before binding
    assert len(e.outvars) == 2    # one output per index of axis 1


# --- gather: traced declines name their reason with the numbers --------------


def _gather_traced_declined(h, *frags):
    p = run(h)  # must not raise: declines never kill the walk
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("gather", 1),)
    assert "gather" not in dict(p.transfers_used)
    note = next(n for n in p.notes if "'gather' declined this form" in n)
    for f in frags:
        assert f in note, (f, note)
    return note


def test_gather_multi_column_indices_decline_traced():
    # x[idx] with a 2-D index array traces to gather indices (2, 2, 1) —
    # not the (N, 1) column the covered row form reads
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        return assert_(x[jnp.array([[0, 1], [1, 2]])] <= 1.0)

    _gather_traced_declined(
        h,
        "indices have shape (2, 2, 1)",
        "(N, 1) column",
    )


def test_gather_non_leading_axis_declines_traced_with_both_geometries():
    # x[:, idx] collapses axis 1, not the leading axis: the note must
    # print the OBSERVED dimension numbers and the COVERED ones, each
    # bound to its side of the sentence
    def h():
        x = any_array((2, 3), "float64", (0.0, 1.0))
        return assert_(x[:, jnp.array([0])] <= 1.0)

    _gather_traced_declined(
        h,
        "got offset_dims=(0,), collapsed_slice_dims=(1,), "
        "start_index_map=(1,)",
        "the covered leading-axis row form is offset_dims=(1,), "
        "collapsed_slice_dims=(0,), start_index_map=(0,)",
    )


def test_gather_partial_row_slice_sizes_decline_traced():
    # a direct lax.gather that takes half a row: everything matches the
    # covered form except slice_sizes — the note prints got and covered
    dn = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )

    def h():
        x = any_array((3, 4), "float64", (0.0, 1.0))
        y = jax.lax.gather(
            x, jnp.array([[0], [2]]), dn, slice_sizes=(1, 2)
        )
        return assert_(jnp.sum(y) <= 99.0)

    _gather_traced_declined(
        h,
        "slice_sizes (1, 2) do not take one full row",
        "= (1, 4)",
        "(shape (3, 4))",
    )


def test_gather_out_of_range_static_index_declines_traced():
    # a static 7 into a 3-row operand reaches the transfer as the point
    # interval [7, 7]: the failing comparison is printed with the true
    # bound
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        return assert_(x[jnp.array([7])] <= 1.0)

    _gather_traced_declined(
        h,
        "index element 0 is 7",
        "0 <= 7 < 3 fails",
        "mode-dependent",
    )


def test_gather_rank0_operand_declines_traced():
    # a rank-0 gather DOES trace on jax 0.11.0 — empty offset dims, empty
    # slice_sizes, indices (1, 0) — and reaches the transfer's rank-0
    # decline through the real walk (audit repair: this path was first
    # claimed hand-IR-only, measured false)
    gdn = jax.lax.GatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(), start_index_map=()
    )

    def h():
        s = any_array((), "float64", (0.0, 1.0))
        y = jax.lax.gather(s, jnp.zeros((1, 0), jnp.int32), gdn, slice_sizes=())
        return assert_(y <= 9.0)

    _gather_traced_declined(h, "rank-0", "no leading axis to take rows from")


def test_gather_out_of_range_mode_behaviours_as_the_decline_states():
    """The decline's measured fragment, measured: "mode 'clip' takes the
    clamped row, mode 'fill' yields the fill value instead of any row".
    If either mode stops behaving as the message states, this goes red
    and the message must be rewritten, not trusted."""
    x = jnp.asarray([1.0, 5.0, 2.0])
    idx = jnp.array(7)
    clip = x.at[idx].get(mode="clip")
    fill = x.at[idx].get(mode="fill", fill_value=jnp.nan)
    assert float(clip) == float(x[2])  # the clamped (last) row
    assert bool(jnp.isnan(fill))      # the fill value, not a row
