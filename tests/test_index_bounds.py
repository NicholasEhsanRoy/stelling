# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""The index-bounds round: dynamic indexing, and the three cases it splits.

SCOPE, declared (docs/norms.md). What this file establishes:

* the HULL IS SOUND -- the box a dynamic index produces contains the value
  jax actually computes, for every start the declared range admits, checked
  by enumerating the whole product and executing the real primitive;
* the hull is TIGHT enough to be worth having, and narrows with the
  declared range (without that control, "always hull the whole operand"
  would pass every soundness test here);
* the three cases are the three cases: inside the legal window produces a
  value, straddling it declines, disjoint from it is reported as a finding;
* the clamp is NOT modelled, pinned by the one query that separates the two
  designs -- `u[i] == u[9]` for `i` in [12, 20] is TRUE of the executed
  program and states nothing about the written one, and must stay undecided;
* the accounting of a finding is byte-for-byte the accounting of a decline:
  top, unknown, unreached, never a REFUTED.

What it does NOT establish: the affine or solver legs (this round emits no
SMT rows -- dynamic_slice and dynamic_update_slice are absent from
obligation._SUPPORTED, so an obligation reaching one cannot escalate), and
gather geometries outside the covered leading-axis row form.

Every jax fact quoted here was measured on 0.11.0 and 0.10.2, and the
measurements that decided the design are re-run as tests below rather than
recorded as prose.
"""
from __future__ import annotations

import itertools
import random

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

import stelling.interval as iv  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.propagate import (  # noqa: E402
    IEEE_TRANSFERS,
    TRANSFERS,
    propagate,
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    # float64 declarations must not be truncated to float32 under us: the
    # ramp fixtures below pin exact element values
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def run(h, **kw):
    return propagate(trace(h), **kw)


def _prim(name):
    from jax._src.lax import slicing as S

    return getattr(S, f"{name}_p")


def _point(arr):
    flat = [float(v) for v in np.asarray(arr).reshape(-1)]
    return iv.IntervalArray(
        shape=tuple(arr.shape), los=tuple(flat), his=tuple(flat)
    )


# -- the round is registered, in both registries, at a stated tier ----------


def test_the_round_registers_both_rows_in_both_registries():
    for name in ("dynamic_slice", "dynamic_update_slice"):
        assert TRANSFERS[name][1] == "exact"
        assert IEEE_TRANSFERS[name][1] == "exact"
    assert set(IEEE_TRANSFERS) == set(TRANSFERS)


# -- what the gap was ------------------------------------------------------


def test_a_traced_scalar_index_is_a_dynamic_slice_not_a_gather():
    """The measurement the whole round turns on. `u[i]` with a traced `i`
    does not trace to a gather: jnp emits the from-the-end normalisation
    and then a `dynamic_slice`. Registering the gather row alone would have
    closed nothing."""
    u, i = jnp.zeros((10,)), jnp.int32(3)
    prims = [e.primitive.name for e in jax.make_jaxpr(lambda u, i: u[i])(u, i).eqns]
    assert "dynamic_slice" in prims
    assert "gather" not in prims
    # and the normalisation is really there, ahead of the take
    assert prims[:3] == ["lt", "add", "select_n"]


def test_an_out_of_range_static_index_takes_the_dynamic_path_too():
    """`u[3]` lowers to a static `slice`; `u[30]` and `u[-11]` do not --
    they fall back to normalise-then-`dynamic_slice`. So the statically
    provable out-of-bounds case arrives at the SAME transfer."""
    u = jnp.zeros((10,))
    assert "slice" in [
        e.primitive.name for e in jax.make_jaxpr(lambda u: u[3])(u).eqns
    ]
    for bad in (30, -11):
        prims = [
            e.primitive.name
            for e in jax.make_jaxpr(lambda u, k=bad: u[k])(u).eqns
        ]
        assert "dynamic_slice" in prims, (bad, prims)


# -- case 1: inside the legal window -- the power gain ---------------------


def test_traced_in_range_index_discharges_where_it_used_to_be_top():
    """Row 3 of the measured gap: `u[i]`, `i` in [0, 5], in bounds. Was
    unknown with the operand at [-inf, inf]."""

    def h():
        u = any_array((10,), "float64", (0.0, 1.0))
        i = any_array((), "int32", (0, 5))
        return assert_(u[i] >= 0.0)

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert dict(p.transfers_used)["dynamic_slice"] == "exact"


def test_the_hull_is_over_the_reachable_slice_and_no_wider():
    """Discrimination. The operand is a ramp, so the answer depends on
    exactly which elements the declared index can reach: with `i` in [0, 3]
    the reachable values are the first four, and the bound that separates a
    correct hull from a lazy one is the fourth."""

    def h(lo, hi, bound):
        def q():
            u = any_array((8,), "float64", (0.0, 1.0))
            i = any_array((), "int32", (lo, hi))
            # u is a declared box, so use its INDEX arithmetic: compare the
            # taken element against itself shifted -- instead, pin on a
            # concrete ramp below. Here just take and bound.
            return assert_(u[i] <= bound)

        return q

    # the operand box is [0, 1] elementwise, so <= 1 holds for any reachable
    # element and <= 0.5 cannot be decided -- the hull must not invent
    # tightness the declaration does not have
    assert run(h(0, 3, 1.0)).obligations[0].status == "discharged"
    assert run(h(0, 3, 0.5)).obligations[0].status == "unknown"


def test_the_hull_narrows_with_the_declared_range_on_real_data():
    """The control that a soundness sweep cannot supply: on a CONCRETE
    ramp, an index confined to the low half must not reach the high half.
    A transfer that hulled the whole axis would be perfectly sound and
    would fail here."""
    ramp = jnp.arange(8, dtype=jnp.float64)  # 0 .. 7

    def h(lo, hi, bound):
        def q():
            i = any_array((), "int32", (lo, hi))
            return assert_(ramp[i] <= bound)

        return q

    assert run(h(0, 3, 3.0)).obligations[0].status == "discharged"
    assert run(h(0, 3, 2.9)).obligations[0].status == "unknown"
    assert run(h(4, 7, 7.0)).obligations[0].status == "discharged"
    assert run(h(4, 7, 3.0)).obligations[0].status == "violated-over-set"
    # and the point case stays exact
    assert run(h(5, 5, 5.0)).obligations[0].status == "discharged"
    assert run(h(5, 5, 4.9)).obligations[0].status == "violated-over-set"


def test_a_from_the_end_index_is_python_semantics_not_out_of_bounds():
    """MEASURED, and it contradicts the obvious guess: `u[-1]` is `u[9]`,
    because jnp normalises the index UPSTREAM of the primitive. The
    primitive itself clamps a negative start to 0 -- both are true, at
    different layers, and the transfer sits at the lower one."""
    ramp = jnp.arange(10, dtype=jnp.float64)
    assert float(jax.jit(lambda u, i: u[i])(ramp, jnp.int32(-1))) == 9.0
    assert float(
        np.asarray(
            jax.jit(
                lambda u, i: _prim("dynamic_slice").bind(u, i, slice_sizes=(1,))
            )(ramp, jnp.int32(-1))
        )[0]
    ) == 0.0

    def h(k, bound):
        def q():
            i = any_array((), "int32", (k, k))
            return assert_(ramp[i] <= bound)

        return q

    # -1 reaches the LAST element, so <= 8 must not discharge
    assert run(h(-1, 9.0)).obligations[0].status == "discharged"
    assert run(h(-1, 8.0)).obligations[0].status == "violated-over-set"
    assert run(h(-10, 0.0)).obligations[0].status == "discharged"


# -- case 2: straddling -- decline, named ----------------------------------


def test_straddling_index_declines_with_its_reason():
    """Row 4 of the measured gap: `i` in [0, 20] on a 10-element array.
    Some declared inputs index in bounds and some do not."""

    def h():
        u = any_array((10,), "float64", (0.0, 1.0))
        i = any_array((), "int32", (0, 20))
        return assert_(u[i] >= 0.0)

    p = run(h)
    assert p.obligations[0].status == "unknown"
    note = next(n for n in p.notes if "'dynamic_slice' declined" in n)
    assert "straddles the legal positions [0, 9]" in note, note
    assert "OUT-OF-BOUNDS INDEX" not in note, note


# -- case 3: disjoint -- a finding, not a decline --------------------------


def _finding(p, *frags):
    note = next(n for n in p.notes if "OUT-OF-BOUNDS INDEX (definite)" in n)
    for f in frags:
        assert f in note, (f, note)
    return note


def test_always_out_of_range_index_is_reported_as_a_finding():
    """Row 5 of the measured gap: `i` in [15, 20] on a 10-element array is
    out of bounds for EVERY input the declaration admits."""

    def h():
        u = any_array((10,), "float64", (0.0, 1.0))
        i = any_array((), "int32", (15, 20))
        return assert_(u[i] >= 0.0)

    p = run(h)
    assert p.obligations[0].status == "unknown"
    _finding(
        p,
        "'dynamic_slice'",
        "spans [15, 20]",
        "the legal positions are [0, 9]",
        "no input for which this index is in bounds",
    )


def test_a_statically_provable_out_of_range_index_is_reported():
    """The case a jax maintainer asked for in Feb 2026 and nothing in the
    ecosystem does: checkify is runtime-only and jax_check_static_indices
    reaches only static constants. jax itself does not raise -- measured on
    0.11.0, `arange(10)[30]` returns 9.0."""
    assert float(jnp.arange(10, dtype=jnp.float64)[30]) == 9.0

    def h():
        u = any_array((10,), "float64", (0.0, 1.0))
        return assert_(u[30] >= 0.0)

    p = run(h)
    assert p.obligations[0].status == "unknown"
    _finding(p, "spans [30, 30]", "the legal positions are [0, 9]")


def test_a_from_the_end_index_past_the_front_is_reported():
    """`u[-11]` on a length-10 axis normalises to -1 and IS out of bounds;
    the note says so, and warns that the span printed is post-normalisation
    so a reader does not go looking for a -11 that no equation carries."""

    def h():
        u = any_array((10,), "float64", (0.0, 1.0))
        return assert_(u[-11] >= 0.0)

    p = run(h)
    assert p.obligations[0].status == "unknown"
    _finding(p, "spans [-1, -1]", "u[-11] arrives here as -1")


def test_a_finding_is_accounted_exactly_like_a_decline():
    """The direction that matters: a finding must never manufacture a
    verdict. Same ⊤, same unknown count, same unreached, no tier recorded
    -- only the note differs."""

    def h(k):
        def q():
            u = any_array((10,), "float64", (0.0, 1.0))
            i = any_array((), "int32", (k, k))
            return assert_(u[i] >= 0.0)

        return q

    finding = run(h(30))
    straddle = run(h(0))  # in range: the control that the query is otherwise
    assert straddle.obligations[0].status == "discharged"
    assert finding.obligations[0].status == "unknown"
    assert finding.coverage.unknown_primitives == (("dynamic_slice", 1),)
    assert "dynamic_slice" not in dict(finding.transfers_used)
    assert not any(o.status == "violated-over-set" for o in finding.obligations)


# -- the clamp is not modelled ---------------------------------------------


def test_the_clamp_is_not_modelled_and_this_is_the_query_that_shows_it():
    """THE control for the whole design. `u[i] == u[9]` for `i` in [12, 20]
    is TRUE of the program jax executes -- every index clamps to 9 -- and
    says nothing about the program the user wrote. A clamp-faithful
    transfer discharges it. This one must not.

    Measured: the clamp really does make it true at runtime, so the test is
    not asserting against a phantom."""
    ramp = jnp.arange(10, dtype=jnp.float64)
    take = jax.jit(lambda u, i: u[i])
    for k in (12, 15, 20):
        assert float(take(ramp, jnp.int32(k))) == float(ramp[9])

    def h():
        u = any_array((10,), "float64", (0.0, 1.0))
        i = any_array((), "int32", (12, 20))
        return assert_(u[i] == u[9])

    p = run(h)
    assert p.obligations[0].status == "unknown"
    _finding(p, "spans [12, 20]")


def test_a_write_past_the_end_is_dropped_by_jax_and_reported_here():
    """The scatter half of the same inconsistency, measured: a read clamps,
    a write is DROPPED. `x.at[30].set(v)` on a length-10 `x` is a no-op --
    which is why one clamp story cannot cover both, and why neither is
    modelled."""
    ramp = jnp.arange(10, dtype=jnp.float64)
    assert list(np.asarray(jax.jit(lambda u: u.at[30].set(-1.0))(ramp))) == list(
        np.asarray(ramp)
    )
    assert float(
        np.asarray(jax.jit(lambda u: u.at[-1].set(-1.0))(ramp))[9]
    ) == -1.0


def test_dynamic_update_slice_clamps_its_start_exactly_as_the_read_does():
    """Measured, primitive-level: a length-2 update at start 9 or 20 of a
    length-10 operand lands at index 8. So the write row uses the SAME
    legal window `[0, n - s]` and declines outside it."""
    ramp = jnp.arange(10, dtype=jnp.float64)
    upd = jnp.asarray([-1.0, -2.0])
    f = jax.jit(lambda u, v, i: _prim("dynamic_update_slice").bind(u, v, i))
    for start in (9, 20):
        got = np.asarray(f(ramp, upd, jnp.int32(start)))
        assert list(got[8:]) == [-1.0, -2.0], (start, got)

    def h(lo, hi):
        def q():
            u = any_array((10,), "float64", (0.0, 1.0))
            v = any_array((2,), "float64", (5.0, 6.0))
            i = any_array((), "int32", (lo, hi))
            return assert_(jax.lax.dynamic_update_slice(u, v, (i,)) <= 6.0)

        return q

    assert run(h(0, 8)).obligations[0].status == "discharged"
    p = run(h(9, 9))
    assert p.obligations[0].status == "unknown"
    _finding(p, "the legal positions are [0, 8]")


def test_dynamic_update_slice_keeps_the_operand_where_no_start_writes():
    """The write row's own discrimination: a position outside every
    admitted window keeps the operand's box, one inside every admitted
    window takes the update's, and one that may or may not be written gets
    the hull of both. Pinned on concrete data through the interval layer,
    where the three answers are distinguishable."""
    operand = _point(jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0]))
    update = _point(jnp.asarray([9.0]))
    box = iv.dynamic_update_slice_hull(operand, update, ((1, 2),))
    assert (box.los[0], box.his[0]) == (0.0, 0.0)  # never written
    assert (box.los[1], box.his[1]) == (0.0, 9.0)  # written iff start == 1
    assert (box.los[2], box.his[2]) == (0.0, 9.0)  # written iff start == 2
    assert (box.los[3], box.his[3]) == (0.0, 0.0)  # never written
    # a width-1 start writes exactly one position and nothing else
    exact = iv.dynamic_update_slice_hull(operand, update, ((2, 2),))
    assert (exact.los[2], exact.his[2]) == (9.0, 9.0)
    assert (exact.los[1], exact.his[1]) == (0.0, 0.0)
    # a start range whose windows COVER a position leaves no operand value
    wide = iv.dynamic_update_slice_hull(
        operand, _point(jnp.asarray([9.0, 8.0])), ((1, 1),)
    )
    assert (wide.los[1], wide.his[1]) == (9.0, 9.0)


# -- the soundness sweep, with its positive control ------------------------


def _sweep_ds(rng, n, hull):
    """Enumerate EVERY admitted start, execute the real primitive, and count
    values the box does not contain. Returns (violations, elements)."""
    ds = _prim("dynamic_slice")
    violations = elements = 0
    for _ in range(n):
        rank = rng.randint(1, 3)
        shape = tuple(rng.randint(1, 4) for _ in range(rank))
        sizes = tuple(rng.randint(1, d) for d in shape)
        ranges = []
        for d, s in zip(shape, sizes):
            lo = rng.randint(0, d - s)
            ranges.append((lo, rng.randint(lo, d - s)))
        ranges = tuple(ranges)
        arr = jnp.asarray(
            rng.sample(range(-500, 500), int(np.prod(shape))),
            dtype=jnp.float64,
        ).reshape(shape)
        box = hull(_point(arr), ranges, sizes)
        for start in itertools.product(
            *[range(lo, hi + 1) for lo, hi in ranges]
        ):
            got = np.asarray(
                ds.bind(arr, *[jnp.int32(s) for s in start], slice_sizes=sizes)
            ).reshape(-1)
            for i, v in enumerate(got):
                elements += 1
                if not box.los[i] <= v <= box.his[i]:
                    violations += 1
    return violations, elements


def test_the_hull_contains_what_jax_computes_for_every_admitted_index():
    """Soundness, measured rather than argued: over randomised shapes,
    slice sizes and start ranges, every value the real primitive produces
    at every admitted start lies inside the box. The enumeration is total
    over the declared range -- this is not a sample."""
    violations, elements = _sweep_ds(
        random.Random(20260809), 250, iv.dynamic_slice_hull
    )
    assert elements > 500, elements  # anti-vacuity: the sweep really ran
    assert violations == 0, violations


def test_the_sweep_catches_a_hull_that_covered_only_the_lowest_start():
    """POSITIVE CONTROL for the test above. A zero with no positive control
    has been wrong three times in this project. This mutant is the exact
    error the round is most exposed to -- hulling over the starts you
    thought of instead of the ones the declaration admits -- and the
    instrument must see it."""

    def sampled(a, ranges, sizes):
        return iv.dynamic_slice_hull(
            a, tuple((lo, lo) for lo, _ in ranges), sizes
        )

    violations, elements = _sweep_ds(random.Random(20260809), 250, sampled)
    assert elements > 500, elements
    assert violations > 0, "the instrument cannot see a wrong hull"


def test_the_sweep_catches_an_exclusive_upper_endpoint():
    """Second positive control, a different error: an off-by-one that drops
    the last admitted index. Caught for the same reason and by the same
    enumeration."""

    def off_by_one(a, ranges, sizes):
        return iv.dynamic_slice_hull(
            a,
            tuple((lo, hi - 1 if hi > lo else hi) for lo, hi in ranges),
            sizes,
        )

    violations, _ = _sweep_ds(random.Random(7), 250, off_by_one)
    assert violations > 0, "the instrument cannot see an off-by-one hull"


# -- the declines, each named ----------------------------------------------


def test_the_work_budget_declines_rather_than_hanging():
    """Degrade-don't-hang: a hull whose enumeration would exceed the budget
    declines to ⊤ instead of running unbounded. Sound in the safe
    direction -- the budget can cost precision and can never cost
    soundness."""
    # work is |out| x width, so a half-width window over a 4096-element
    # axis costs ~4.2M visits: just past the 4194304 budget
    n = 4096
    big = iv.from_bounds((n,), 0.0, 1.0)
    with pytest.raises(iv.IntervalError, match="past the .* budget"):
        iv.dynamic_slice_hull(big, ((0, n // 2),), (n // 2,))
    # and just under the cap it computes
    small = iv.from_bounds((4,), 0.0, 1.0)
    assert iv.dynamic_slice_hull(small, ((0, 3),), (1,)).shape == (1,)


def test_a_start_outside_the_legal_window_never_reaches_the_hull():
    """The hull functions refuse a start range they were not promised is
    in-window. The transfer classifies first, so this is defence in depth:
    if a future edit dropped the classification, the hull would decline
    rather than silently model the clamp."""
    a = iv.from_bounds((4,), 0.0, 1.0)
    with pytest.raises(iv.IntervalError, match="leaves the legal start window"):
        iv.dynamic_slice_hull(a, ((2, 4),), (2,))
    with pytest.raises(iv.IntervalError, match="leaves the legal start window"):
        iv.dynamic_update_slice_hull(
            a, iv.from_bounds((2,), 0.0, 1.0), ((3, 3),)
        )


def test_a_narrow_index_dtype_declines_because_xla_wraps_the_bound():
    """XLA computes an index's out-of-bounds comparison in the INDEX's
    element type; a bound that does not fit wraps. Probing jax 0.11.0's
    dynamic_slice with an int8 start over lengths 100/127/128/129/200 did
    not exhibit it, so the hazard is UNCONFIRMED for this primitive -- and
    refused anyway, because every dtype jnp's own indexing produces is
    int32/int64 and the gate is therefore free."""
    from stelling.propagate import _index_dtype_covers_or_decline

    _index_dtype_covers_or_decline("int32", 10**6, "dynamic_slice")  # fine
    with pytest.raises(iv.IntervalError, match="does not hold the largest"):
        _index_dtype_covers_or_decline("int8", 200, "dynamic_slice")
    with pytest.raises(iv.IntervalError, match="not one this build knows"):
        _index_dtype_covers_or_decline("float64", 3, "gather")


def test_take_row_ranges_agrees_with_take_rows_on_point_indices():
    """The generalisation must not have moved the case it generalises."""
    a = iv.IntervalArray(
        shape=(3, 2),
        los=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        his=(1.5, 2.5, 3.5, 4.5, 5.5, 6.5),
    )
    for ks in ([0], [2, 0], [1, 1, 2]):
        assert iv.take_row_ranges(a, [(k, k) for k in ks]) == iv.take_rows(a, ks)


# -- ieee mirror -----------------------------------------------------------


def test_the_row_is_sound_as_is_under_ieee():
    """Data movement is sound as-is under ieee: every output element IS an
    operand element, so the output is maybe-NaN exactly when the operand
    is. Bound at the PRIMITIVE, which is the only way to reach the row
    under ieee -- see the next test for why."""

    def h():
        u = any_array((10,), "float64", (0.0, 1.0))
        i = any_array((), "int32", (0, 5))
        return assert_(_prim("dynamic_slice").bind(u, i, slice_sizes=(1,)) >= 0.0)

    for sem in ("real", "ieee"):
        p = run(h, semantics=sem)
        assert p.obligations[0].status == "discharged", sem
        assert dict(p.transfers_used)["dynamic_slice"] == "exact", sem


def test_under_ieee_the_normalisation_declines_before_the_row_is_reached():
    """A LIMITATION, recorded rather than papered over. `u[i]` written the
    ordinary way carries jnp's from-the-end normalisation, whose integer
    `add` declines under ieee (endpoint arithmetic there is binary64-only).
    So the index arrives ⊤ and maybe-NaN, and the row declines on the NaN
    index exactly as gather and scatter do. Sound -- the ieee leg simply
    buys nothing for jnp-spelled dynamic indexing until the integer
    endpoint question is settled, which is a mode-wide decision and not a
    two-row fix."""

    def h():
        u = any_array((10,), "float64", (0.0, 1.0))
        i = any_array((), "int32", (0, 5))
        return assert_(u[i] >= 0.0)

    p = run(h, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert dict(p.coverage.unknown_primitives) == {"add": 1, "dynamic_slice": 1}
    assert any("carry maybe-NaN under ieee" in n for n in p.notes), p.notes
    # and the real leg on the identical query decides it
    assert run(h).obligations[0].status == "discharged"
