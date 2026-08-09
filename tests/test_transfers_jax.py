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
    # the decline names the element's declared span, the right way round
    assert any(
        "'scatter' declined this form" in n
        and "index spans [0.0, 2.0]" in n
        for n in p.notes
    ), p.notes


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


def test_fvm_gather_dynamic_index_takes_the_hull_traced():
    # CHANGED EXPECTATION, index-bounds round: a traced index whose whole
    # declared range lands on the operand's axis no longer declines. Every
    # row of x is in [0, 1], so whichever row the index reaches, x[i] <= 1
    # holds — and this is the shape of query the decline used to lose.
    #
    # THE HAZARD THIS TEST GUARDS: jnp inserts the from-the-end
    # normalisation (lt/add/select_n) ahead of the take, so the interval
    # reaching the transfer is the NORMALISED one. The declared [0, 2] is
    # non-negative, the normalisation is the identity on it, and the range
    # arrives intact — if any of that broke, the range would widen and the
    # obligation would fall back to unknown rather than to a wrong answer.
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        i = any_array((1,), "int32", (0.0, 2.0))
        return assert_(x[i] <= 1.0)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert dict(p.transfers_used)["gather"] == "exact"

    # discrimination: the same query with a bound the hull cannot meet must
    # NOT discharge. x spans [0, 1] elementwise, so <= 0.5 is undecided.
    def h2():
        x = any_array((3,), "float64", (0.0, 1.0))
        i = any_array((1,), "int32", (0.0, 2.0))
        return assert_(x[i] <= 0.5)

    assert run(h2).obligations[0].status == "unknown"


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


def test_gather_out_of_range_static_index_is_a_finding_traced():
    # CHANGED EXPECTATION: a static 7 into a 3-row operand is out of bounds
    # for every input, through the REAL trace. This is the case a jax
    # maintainer asked for in Feb 2026 — "I would rather there be an error
    # for OOB indexing if it's statically provable instead of silently
    # giving the wrong answer" — and nothing in the ecosystem reports it:
    # checkify is runtime-only and jax_check_static_indices reaches only
    # static constants. jax itself does not raise; measured on 0.11.0 it
    # returns the clamped row.
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        return assert_(x[jnp.array([7])] <= 1.0)

    p = run(h)
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("gather", 1),)
    assert "gather" not in dict(p.transfers_used)
    note = next(n for n in p.notes if "OUT-OF-BOUNDS INDEX (definite)" in n)
    for f in ("'gather'", "spans [7, 7]", "the legal positions are [0, 2]"):
        assert f in note, (f, note)


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


# --- scatter: traced declines name their reason with the numbers -------------


def _scatter_traced_declined(h, *frags):
    p = run(h)  # must not raise: declines never kill the walk
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("scatter", 1),)
    assert "scatter" not in dict(p.transfers_used)
    note = next(n for n in p.notes if "'scatter' declined this form" in n)
    for f in frags:
        assert f in note, (f, note)
    return note


def test_scatter_apply_declines_traced_naming_the_combiner():
    # x.at[k].apply(f) traces to the same primitive as .set; the decline
    # must name the combiner rather than the shapes (which are identical
    # to a legitimate set — blaming them sends the reader nowhere)
    def h():
        x = any_array((3,), "float64", (2.0, 3.0))
        return assert_(x.at[0].apply(lambda t: t + 100.0)[0] >= 1.0)

    _scatter_traced_declined(
        h,
        "carries a combiner (update_jaxpr)",
        "`x.at[k].apply(f)`-shaped",
        "write the dummy where the program computes f(operand[k])",
    )


def test_scatter_apply_traces_as_the_combiner_decline_states():
    """The combiner decline's measured fragments, measured: `.apply`
    traces with the same dimension numbers, shapes, mode and static index
    as `.set`, beside a DUMMY updates operand that is 0.0 regardless of
    f, and the traced set form carries update_consts=(). If jax stops
    tracing it that way, this goes red and the sentence must be
    rewritten, not trusted."""
    from stelling._jax_compat import transcribe

    def build_apply(x):
        return x.at[1].apply(lambda t: t + 100.0)

    def build_set(x):
        return x.at[1].set(0.0)

    x0 = jnp.zeros(3, jnp.float64)
    ap = [e for e in transcribe(jax.make_jaxpr(build_apply)(x0)).jaxpr.eqns
          if str(e.primitive) == "scatter"][0]
    st = [e for e in transcribe(jax.make_jaxpr(build_set)(x0)).jaxpr.eqns
          if str(e.primitive) == "scatter"][0]
    ap_p, st_p = dict(ap.params), dict(st.params)
    # same dimension numbers, mode, shapes, static index; the combiner is
    # the distinguishing field
    assert ap_p["dimension_numbers"] == st_p["dimension_numbers"]
    assert ap_p["mode"] == st_p["mode"]
    assert [tuple(v.aval.shape) for v in ap.invars] == [
        tuple(v.aval.shape) for v in st.invars
    ]
    assert ap_p["update_jaxpr"] is not None
    assert st_p["update_jaxpr"] is None
    # the traced set form carries update_consts=() (quoted by the
    # leftover-consts decline in test_transfers.py)
    assert st_p["update_consts"] == ()
    # the updates operand of .apply is a DUMMY: 0.0, untouched by f's +100
    # (the traced literal rides as a little-endian float64 payload)
    import struct

    dummy = ap.invars[2].val
    assert struct.unpack("<d", dummy.data)[0] == 0.0


def test_scatter_rank2_write_declines_traced_with_the_configuration():
    # x.at[0, 0].set(v) is outside the covered rank-1 form: the note
    # prints the observed configuration and the covered core
    def h():
        x = any_array((2, 2), "float64", (0.0, 1.0))
        return assert_(x.at[0, 0].set(5.0) <= 9.0)

    _scatter_traced_declined(
        h,
        "operand (2, 2), indices (2,), updates ()",
        "outside the covered static-index set row form",
        "update_window_dims=(), inserted_window_dims=(0,), "
        "scatter_dims_to_operand_dims=(0,)",
    )


def test_scatter_multi_index_write_declines_traced():
    # two index rows write two elements: outside the one-row covered form
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = x.at[jnp.array([0, 1])].set(jnp.array([5.0, 6.0]))
        return assert_(y <= 9.0)

    _scatter_traced_declined(
        h,
        "indices (2, 1), updates (2,)",
        "outside the covered static-index set row form",
    )


def test_scatter_out_of_range_static_index_declines_traced():
    # a static 7 into a 3-element operand reaches the transfer as the
    # point interval [7, 7]: the failing comparison is printed with the
    # true bound
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        return assert_(x.at[jnp.array(7)].set(5.0) <= 9.0)

    _scatter_traced_declined(
        h,
        "index 7 is out of range",
        "0 <= 7 < 3 fails",
        "mode-dependent",
        "never guessed",
    )


def test_scatter_narrow_index_dtype_declines_traced_with_the_numbers():
    # the int8 wedge, traced end to end: at operand length 129 the
    # out-of-bounds bound 128 does not fit int8, and XLA's range check is
    # no longer the one the row models — the note prints the dtype's true
    # range and the measured 128/129 boundary
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def h():
        x = any_array((129,), "float64", (0.0, 1.0))
        y = jax.lax.scatter(
            x, jnp.array([5], jnp.int8), jnp.asarray(7.0), dn,
            mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
        )
        return assert_(jnp.sum(y) <= 1e9)

    _scatter_traced_declined(
        h,
        "leading-axis bound 128",
        "int8 represents [-128, 127], which does not contain 128",
        "writes at operand length 128",
        "129",
    )


def test_scatter_out_of_range_mode_behaviours_as_the_decline_states():
    """The out-of-range decline's measured fragments, measured FOR BOTH
    SIGNS: FILL_OR_DROP drops the write and the operand passes through
    unchanged (7 and -1 alike); clip clamps the index into range from
    both sides — 7 onto the last element, -2 onto the first (audit
    repair F1: the sentence used to say clip rewrites "onto the last
    element", false for negative indices). The `.at[...]` sugar
    normalizes a negative index at trace time (an lt/add/select_n
    prologue before the scatter), so the negative-sign measurements ride
    direct lax.scatter — the route on which a genuine negative reaches
    the primitive. If either mode stops behaving as the message states,
    this goes red and the message must be rewritten, not trusted."""
    x = jnp.asarray([1.0, 5.0, 2.0])
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def sc(idx, mode):
        return jax.lax.scatter(
            x, jnp.array([idx]), jnp.asarray(9.0), dn, mode=mode
        )

    drop = jax.lax.GatherScatterMode.FILL_OR_DROP
    clip = jax.lax.GatherScatterMode.CLIP
    assert sc(7, drop).tolist() == x.tolist()       # unchanged
    assert sc(-1, drop).tolist() == x.tolist()      # unchanged, both signs
    assert sc(7, clip).tolist() == [1.0, 5.0, 9.0]  # onto the last element
    assert sc(-2, clip).tolist() == [9.0, 5.0, 2.0]  # onto the first


def test_scatter_negative_index_declines_traced_at_the_lower_boundary():
    # k = -1, the lower boundary (audit repair F2a): FILL_OR_DROP drops
    # it (measured above), while python's los[-1] wraps to the LAST
    # element — so admitting it would substitute at an element jax
    # leaves untouched: a silent false model, and the refutation built
    # on it would be false. The `.at` sugar normalizes negatives at
    # trace time, so the genuine -1 rides direct lax.scatter.
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = jax.lax.scatter(
            x, jnp.array([-1]), jnp.asarray(5.0), dn,
            mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
        )
        return assert_(jnp.sum(y) <= 99.0)

    _scatter_traced_declined(
        h,
        "index -1 is out of range",
        "0 <= -1 < 3 fails",
    )


def test_scatter_index_equal_to_length_declines_traced():
    # k = n, the upper boundary one past the last element (audit repair
    # F2c): admitting it would subscript los[3] and kill the walk with a
    # raw IndexError — declines never raise anything but IntervalError
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = jax.lax.scatter(
            x, jnp.array([3]), jnp.asarray(5.0), dn,
            mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
        )
        return assert_(jnp.sum(y) <= 99.0)

    _scatter_traced_declined(
        h,
        "index 3 is out of range",
        "0 <= 3 < 3 fails",
    )


def test_scatter_int8_covered_length_128_still_discharges_traced():
    # the ADMIT side of the index-dtype boundary (audit repair F2b): at
    # operand length 128 the out-of-bounds bound 127 IS int8's maximum,
    # the covered form holds, and the row must still DISCHARGE — pinned
    # against the dtype pre-check (a +1 perturbation of its argument
    # declines this very form and goes red here)
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def h():
        x = any_array((128,), "float64", (0.0, 1.0))
        y = jax.lax.scatter(
            x, jnp.array([5], jnp.int8), jnp.asarray(7.0), dn,
            mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
        )
        return assert_(y <= 7.0)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert ("scatter", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_scatter_int8_boundary_writes_and_drops_as_the_decline_states():
    """The index-dtype decline's measured fragment, measured: an int8
    index column writes at operand length 128 and silently DROPS an
    in-range-looking write at 129 (the bound 128 wraps in int8, so XLA's
    comparison rejects index 5 that the row's own range check would
    admit)."""
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def write(n):
        return jax.lax.scatter(
            jnp.zeros(n), jnp.array([5], jnp.int8), jnp.asarray(7.0), dn,
            mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
        )

    assert float(write(128)[5]) == 7.0  # lands
    assert float(write(129)[5]) == 0.0  # silently dropped


def test_a_remat_wrapper_is_descended_on_every_tested_series():
    """`jax.checkpoint` must be inlined, not left opaque.

    The regression this pins is invisible to a single-series lane. jax 0.11
    merged `Jaxpr` and `ClosedJaxpr` into one class, so on 0.11 `remat2`'s
    body param transcribes to `ir.ClosedJaxpr`; on 0.10 it is still a bare
    `ir.Jaxpr`. Both descents — the propagation's and the obligation
    slice's — tested `isinstance(v, ir.ClosedJaxpr)`, so on 0.10 they found
    no body and refused the wrapper. Measured before the fix: this harness
    was VERIFIED on 0.11.0 and UNKNOWN on 0.10.2, with the note
    "transparent 'remat2': arity mismatch or no sub-jaxpr; ⊤".

    Asserted through COVERAGE rather than through the status, because the
    status alone would also pass if the body were never reached but the
    obligation happened to discharge from the operands.
    """
    def h():
        x = any_array((4,), "float64", (1.0, 2.0))
        return assert_(jax.checkpoint(lambda y: y * 2.0 + 1.0)(x) > 2.0)

    p = run(h)
    assert p.coverage.transparent >= 1, "the remat2 wrapper was not descended"
    assert p.coverage.unknown == 0, (
        f"nothing should fall to ⊤ here; got {p.coverage.unknown_primitives}"
    )
    assert p.coverage.unreached == 0, "the remat body must be reached"
    assert p.obligations[0].status == "discharged"
    assert not any("remat2" in n for n in p.notes), f"wrapper refused: {p.notes}"
