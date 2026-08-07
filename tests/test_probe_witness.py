# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The reachability certificate's witnesses must be MEMBERS of the box.

A branch-scoped refutation is only allowed to stand when a POINT of the
declared set is found that takes the branch. Everything here is about the
word *point*: the pinned probe run is the witness search, so a probe that
pins a declaration to a value the declared set does not contain is not a
witness at all, and any REFUTED resting on it is wrong.

Measured on ``688e829``: ``any_array((), "int32", (0.2, 2.8))`` declares
the integers of that interval — stelling's own refusal for ``(0.2, 0.8)``
says "int32 represents the integers ... and the interval contains none of
them" — so the declared set is ``{1, 2}``. The probes pinned it to ``0.2``
and ``2.8``, and ``i < 1`` and ``i > 2``, **false at every member**, were
certified reached and stamped REFUTED.

The oracle for every reachability claim below is stated in the test: the
declared set and the guard, worked out by hand. No test here asks
stelling what the answer is.

Each test names the surface it pins, because these are the surfaces a
mutation survey found nothing pinning (integer pinning, the
constraining-assume gate, ``_PROBE_COUNT``, nonvacuity's presence in
``_UNEXAMINABLE_OBLIGATIONS``, the branch index in the path, and the path
in the reachability key).
"""

from __future__ import annotations

import dataclasses
import math
import random

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling import ir  # noqa: E402
from stelling.harness import (  # noqa: E402
    any_array,
    assert_,
    assume,
    nonvacuity,
    trace,
)
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import (  # noqa: E402
    _INT_DTYPE_BOUNDS,
    _is_integer_dtype,
    _PROBE_COUNT,
    _probe_point,
    propagate,
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def statuses(harness):
    return [o.status for o in propagate(trace(harness)).obligations]


def _cond(guard, decls=("x",)):
    """`cond(guard, assert_(v > 5.0), assert_(v > -9.0), x)`.

    The 'yes' obligation is definitely false over ``x``'s box and the 'no'
    obligation definitely true, so the statuses read directly as "was the
    yes branch certified reachable".
    """

    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            guard(x),
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    return h


# --- F1: the point must be a member ----------------------------------------


def test_an_integer_declaration_is_never_pinned_off_its_own_integers():
    """PINS: the integer pinning's CLAMP ORDER (mutation M6's neighbour).

    `int32 (0.2, 2.8)` declares `{1, 2}`. `i < 1` is false at 1 and at 2;
    so is `i > 2`. Both were REFUTED on `688e829` because the clamp ran
    AFTER the rounding and restored `0.2` / `2.8`.
    """

    def below():
        i = any_array((), "int32", (0.2, 2.8))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            i < 1,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    def above():
        i = any_array((), "int32", (0.2, 2.8))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            i > 2,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert sorted(statuses(below)) == ["discharged", "unknown"]
    assert sorted(statuses(above)) == ["discharged", "unknown"]
    assert check(below, vacuity_mode="all").status == "UNKNOWN"
    assert check(above, vacuity_mode="all").status == "UNKNOWN"


def test_the_same_guard_over_a_set_that_contains_a_witness_still_refutes():
    """The control for the two above: `int32 (0.0, 3.0)` declares
    `{0, 1, 2, 3}`, `i < 1` holds at `0`, the branch really runs there and
    its obligation really fails. A fix that just stopped certifying
    integer declarations would pass the test above and fail this one."""

    def h():
        i = any_array((), "int32", (0.0, 3.0))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            i < 1,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert "violated-over-set" in statuses(h)
    assert check(h, vacuity_mode="all").status == "REFUTED"


def test_a_non_integral_point_is_what_the_rounding_stops():
    """PINS: the integer ROUNDING itself (mutation M6).

    `int32 (0.0, 3.0)` declares `{0, 1, 2, 3}` and `(i-1)*(i-2) < 0` is
    false at every one of them — it is only true strictly between 1 and 2.
    The middle anchor lands on `1.5`, which without the rounding is a
    non-member witness for a branch no execution takes.
    """

    def h():
        i = any_array((), "int32", (0.0, 3.0))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            (i - 1) * (i - 2) < 0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert sorted(statuses(h)) == ["discharged", "unknown"]

    def control():
        # `(i-1)*(i-3) < 0` IS true at the member 2: unmoved.
        i = any_array((), "int32", (0.0, 3.0))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            (i - 1) * (i - 3) < 0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert "violated-over-set" in statuses(control)


def test_the_pin_stays_inside_the_dtypes_own_range():
    """The declared set is bounded TWICE: by the interval and by the dtype.

    `int8 (-1e9, 1e9)` declares `[-128, 127]`, so `k < -200.0` (read
    through a float cast, which is what keeps the literal from wrapping
    into int8) is false at every member — and REFUTED on `688e829`.
    """

    def unreachable():
        k = any_array((), "int8", (-1e9, 1e9))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            k.astype(jnp.float64) < -200.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    def reachable():
        # true at the member -128: the neighbour that must not move
        k = any_array((), "int8", (-1e9, 1e9))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            k.astype(jnp.float64) < -100.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert sorted(statuses(unreachable)) == ["discharged", "unknown"]
    assert "violated-over-set" in statuses(reachable)


_SWEEP_BOUNDS = [
    (0.2, 2.8),
    (-2.8, -0.2),
    (0.0, 3.0),
    (0.5, 1.5),
    (-0.5, 0.5),
    (2.0, 2.0),
    (-1e9, 1e9),
    (-1e308, 1e308),
    (0.0, 1e308),
    (-1e308, 0.0),
    (1e-9, 2.5),
    (-math.inf, math.inf),
    (1.5, math.inf),
    (-math.inf, -1.5),
    (0.0, math.inf),
]
_rng = random.Random(20260808)
for _ in range(200):
    _a = _rng.uniform(-1e6, 1e6)
    _b = _a + abs(_rng.gauss(0.0, 1e3))
    _SWEEP_BOUNDS.append((_a, _b))


def _declares_a_member(lo, hi, dtype):
    """Does `[lo, hi]` under `dtype` declare a non-empty set? The
    SPECIFICATION, in exact integer arithmetic: the values of an integer
    dtype are the integers of its range, and the values of a float dtype
    are dense enough that any non-empty interval holds one."""
    if not _is_integer_dtype(dtype):
        return lo <= hi
    d_lo, d_hi = _INT_DTYPE_BOUNDS[dtype]
    lo_i = d_lo if lo == -math.inf else math.ceil(lo)
    hi_i = d_hi if hi == math.inf else math.floor(hi)
    return max(lo_i, d_lo) <= min(hi_i, d_hi)


@pytest.mark.parametrize("dtype", sorted(_INT_DTYPE_BOUNDS) + ["float32", "float64"])
def test_every_probe_point_it_can_form_is_a_member(dtype):
    """Over every dtype, 215 bound pairs and every probe index: the point
    is inside the box, integral when the dtype is, and inside the dtype's
    own range. Ranges wider than the defect — it includes the float
    dtypes and the infinite-endpoint ladder, neither of which the reported
    construction touches."""
    formed = 0
    inhabited = 0
    for lo, hi in _SWEEP_BOUNDS:
        pts = [_probe_point(k, (2,), lo, hi, dtype, 0) for k in range(_PROBE_COUNT)]
        if not _declares_a_member(lo, hi, dtype):
            # no member, no witness — checked in EXACT integer arithmetic,
            # so this half of the sweep is independent of the float
            # reasoning the implementation does
            assert all(p is None for p in pts), (dtype, lo, hi)
            continue
        inhabited += 1
        for k, vals in enumerate(pts):
            if vals is None:
                continue  # conservative: a withheld probe certifies nothing
            formed += 1
            for v in vals:
                assert math.isfinite(v), (dtype, lo, hi, k, v)
                assert lo <= v <= hi, (dtype, lo, hi, k, v)
                if _is_integer_dtype(dtype):
                    assert v == math.floor(v), (dtype, lo, hi, k, v)
                    d_lo, d_hi = _INT_DTYPE_BOUNDS[dtype]
                    assert d_lo <= v <= d_hi, (dtype, lo, hi, k, v)
    # not vacuous: an implementation that formed nothing would satisfy
    # every assertion above
    assert inhabited > 0 and formed >= _PROBE_COUNT * inhabited, (
        dtype,
        inhabited,
        formed,
    )


def test_a_box_with_no_member_yields_no_probe():
    """PINS: "no member, no witness" — the `None` return.

    `int32 [0.2, 0.8]` holds no integer. `any_array` refuses such a
    declaration outright, so this is reachable only through hand-built
    IR — and there the answer must be "no point", never "some point".
    """
    assert _probe_point(0, (), 0.2, 0.8, "int32", 0) is None
    assert _probe_point(3, (2,), -0.5, -0.2, "int64", 0) is None
    assert _probe_point(1, (), 2.0, 3.0, "bool", 0) is None
    # the neighbours: a box that DOES hold one still gives a point
    assert _probe_point(0, (), 0.2, 1.8, "int32", 0) == [1.0]
    assert _probe_point(1, (), 0.2, 1.8, "bool", 0) == [1.0]


def _int_box_ir(lo, hi):
    """The `_cond` harness with its int32 declaration's box replaced —
    the only route to a memberless declaration, since `any_array` refuses
    one at the door."""

    def h():
        i = any_array((), "int32", (0.0, 3.0))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            i < 0.5,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    cj = trace(h)
    eqns = list(cj.jaxpr.eqns)
    for n, e in enumerate(eqns):
        if e.primitive == "stelling_any" and dict(e.params)["dtype"] == "int32":
            eqns[n] = dataclasses.replace(
                e,
                params=tuple(
                    (k, lo if k == "lo" else hi if k == "hi" else v)
                    for k, v in e.params
                ),
            )
    return ir.ClosedJaxpr(
        jaxpr=dataclasses.replace(cj.jaxpr, eqns=tuple(eqns)), consts=cj.consts
    )


def test_a_probe_that_cannot_be_formed_withholds_rather_than_certifies():
    """The caller's half of "no member, no witness": with no point to pin,
    the declaration keeps its full box, forces nothing, and the branch
    violation is WITHHELD. The control declares the same guard over a box
    that does hold integers, where the refutation stands."""
    empty = [o.status for o in propagate(_int_box_ir(0.2, 0.8)).obligations]
    holds = [o.status for o in propagate(_int_box_ir(0.0, 3.0)).obligations]
    assert sorted(empty) == ["discharged", "unknown"]
    assert "violated-over-set" in holds


# --- F4: the grid must not degenerate on a wide box -------------------------


@pytest.mark.parametrize(
    "lo,hi",
    [
        (-1e308, 1e308),
        (0.0, 1e308),
        (-1e308, 0.0),
        (-1e300, 1e300),
        (-1.0, 1.0),
    ],
)
def test_a_wide_box_keeps_sixteen_distinct_probes(lo, hi):
    """`lo + f*(hi - lo)` overflows on a box wider than half the float
    range: measured on `[-1e308, 1e308]`, 15 of 16 probes collapsed onto
    `hi` and the 16th was NaN. The convex combination never forms the
    difference."""
    pts = [_probe_point(k, (), lo, hi, "float64", 0) for k in range(_PROBE_COUNT)]
    assert all(p is not None for p in pts), pts
    vals = [p[0] for p in pts]
    assert len(set(vals)) == _PROBE_COUNT, sorted(vals)
    assert all(lo <= v <= hi for v in vals)
    assert lo in vals and hi in vals  # the two end anchors, exactly


def test_a_reachable_guard_on_a_wide_box_is_witnessed():
    """`w < -1e307` is true at `w = -1e308`, a point of the declared box.
    Withheld on `688e829` because no probe ever reached the low end."""

    def low():
        w = any_array((), "float64", (-1e308, 1e308))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            w < -1e307,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    def never():
        # the control: on the SAME wide box, a guard false everywhere
        # stays withheld — the fix is more probes, not more refutations
        w = any_array((), "float64", (-1e308, 1e308))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            w - w > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert "violated-over-set" in statuses(low)
    assert sorted(statuses(never)) == ["discharged", "unknown"]


# --- F7: the surfaces nothing pinned ----------------------------------------


def test_a_witness_that_needs_a_probe_beyond_the_three_anchors():
    """PINS: `_PROBE_COUNT` (mutation M9), by a REQUIREMENT rather than by
    its value.

    `x[1] - x[0] > 1.0` is false at all three anchors — every element at
    `lo`, at `hi`, at the middle all give `0` — and true at the mixed
    point probe 4 pins (`x[0] = -0.82`, `x[1] = 0.69`, difference
    `1.51`). So the grid must run past the anchors, and past probe 3
    (difference `0.49`), for this genuinely-reachable branch to refute.
    """
    h = _cond(lambda x: x[1] - x[0] > 1.0)
    assert "violated-over-set" in statuses(h)
    assert check(h, vacuity_mode="all").status == "REFUTED"


def test_the_certificate_is_withheld_while_a_precondition_narrows():
    """PINS: the constraining-assume gate (mutation M8).

    A probe is a point of the DECLARED box; once an assume narrows the
    admitted set, a point of the box is not necessarily a point of what is
    being judged, so the search certifies nothing and every branch-scoped
    violation is withheld. Without the gate the corner `x = (1, 1, 1)`
    certifies this branch and the obligation refutes.
    """

    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        assume(x[0] > -0.5)
        return jax.lax.cond(
            x[1] > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert sorted(statuses(h)) == ["discharged", "unknown"]
    # the positive control: the SAME guard with no assume refutes, so the
    # 'unknown' above is the gate and not an unwitnessable guard
    assert "violated-over-set" in statuses(_cond(lambda x: x[1] > 0.0))


def test_a_nonvacuity_condition_inside_a_scan_is_recorded_unexamined():
    """PINS: `stelling_nonvacuity` in `_UNEXAMINABLE_OBLIGATIONS`
    (mutation M12).

    A nonvacuity condition STATES something, exactly as an assert does;
    dropped inside a body nothing descends into it vanishes silently.
    """

    def h():
        x = any_array((3,), "float64", (1.0, 2.0))

        def body(c, v):
            return c, nonvacuity(v > 5.0)

        return jax.lax.scan(body, 0.0, x)

    p = propagate(trace(h))
    assert [c.status for c in p.nonvacuity_checks] == ["unknown"]
    (c,) = p.nonvacuity_checks
    assert "NOT EXAMINED" in c.detail
    assert "nonvacuity condition" in c.detail
    assert "scan" in c.detail
    assert any("NOT EXAMINED" in n for n in p.notes)


def _shared_branch_object_ir():
    """An IR whose two `cond` branches are the SAME Python object.

    `jax` caches traced branch jaxprs, so `cond(p, f, f, x)` already gives
    both branches the same assert VAR ID; it does not give them the same
    equation object, and the transcriber mints a fresh `JaxprEqn` per
    occurrence. `ir` accepts the aliased form, and only the (cond, branch)
    path can tell the two occurrences apart there.
    """

    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        # guard is 0 > 0: branch 0 (the false leg) runs at every point of
        # the box, branch 1 at none
        return jax.lax.cond(
            x[0] - x[0] > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > 5.0),
            x,
        )

    cj = trace(h)
    eqns = list(cj.jaxpr.eqns)
    (ci,) = [i for i, e in enumerate(eqns) if e.primitive == "cond"]
    branches = dict(eqns[ci].params)["branches"]
    eqns[ci] = dataclasses.replace(
        eqns[ci],
        params=tuple(
            (k, (branches[0], branches[0]) if k == "branches" else v)
            for k, v in eqns[ci].params
        ),
    )
    return ir.ClosedJaxpr(
        jaxpr=dataclasses.replace(cj.jaxpr, eqns=tuple(eqns)), consts=cj.consts
    )


def test_two_branches_that_are_one_object_are_certified_separately():
    """PINS: both halves of the reachability key (mutations N23 and M18).

    With the two branches sharing one jaxpr object, the assert EQUATION
    objects are identical too — `id(eqn)` cannot separate the occurrence
    that runs everywhere from the one that never runs. Keying on the id
    alone certifies both; dropping the branch index from the path does the
    same. Only `(path, eqn)` with the index in it says `['violated-over-
    set', 'unknown']`.
    """
    cj = _shared_branch_object_ir()
    branches = dict(
        [e for e in cj.jaxpr.eqns if e.primitive == "cond"][0].params
    )["branches"]
    assert branches[0] is branches[1]
    asserts = [
        [e for e in b.jaxpr.eqns if e.primitive == "stelling_assert"][0]
        for b in branches
    ]
    assert asserts[0] is asserts[1]  # the collision is real, not assumed
    assert [o.status for o in propagate(cj).obligations] == [
        "violated-over-set",  # branch 0: taken at every point, really false
        "unknown",  # branch 1: never taken, withheld
    ]


# --- the swallower named is the innermost one ------------------------------


def test_the_swallowing_primitive_named_is_the_innermost_one():
    """An assert inside a `while` inside a `scan` sits in the `while`
    body. Reported against `'scan'` on `688e829`, while its source
    location pointed into the `while` — the two disagreed."""

    def h():
        x = any_array((3,), "float64", (1.0, 2.0))

        def body(c, v):
            def wbody(s):
                i, a = s
                return i + 1, jnp.where(assert_(a > 5.0), a, a)

            return c, jax.lax.while_loop(lambda s: s[0] < 3, wbody, (0, v))[1]

        return jax.lax.scan(body, 0.0, x)

    p = propagate(trace(h))
    assert [o.status for o in p.obligations] == ["unknown"]
    (o,) = p.obligations
    assert "'while'" in o.detail and "'scan'" not in o.detail
    # the source location is what it always was, and the note agrees with
    # the detail
    assert o.source_info and o.source_info[-1].split(" (")[0] in p.notes[0]
    assert "'while'" in p.notes[0]


def test_a_body_directly_under_a_scan_still_names_the_scan():
    """The neighbour: one level of nesting, and `'scan'` is both the
    outermost and the innermost answer."""

    def h():
        x = any_array((3,), "float64", (1.0, 2.0))

        def body(c, v):
            return c, assert_(v > 5.0)

        return jax.lax.scan(body, 0.0, x)

    (o,) = propagate(trace(h)).obligations
    assert "'scan'" in o.detail


# --- the index-EXCLUDED branch: dropped deliberately ------------------------


def test_an_index_excluded_branch_is_deliberately_dropped():
    """An index-EXCLUDED branch is a different case from an unexamined one,
    and it is dropped on purpose. See `SOUNDNESS.md`, "An index-excluded
    cond/switch branch is dropped, not recorded".

    `c` is declared over `[1, 2]`, so `c > 0.` is TRUE at every point and
    the false leg is not merely unwitnessed — the index interval excludes
    it, which is a proof that no point of the declared set takes it. Its
    obligation is vacuously true, so recording it `unknown` would be a
    claim the analysis does not hold, and it would cost this query its
    VERIFIED. What the walk does record is the coverage: the excluded
    branch's equations count as unreached.
    """

    def h():
        c = any_array((), "float64", (1.0, 2.0))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            c > 0.0,
            lambda v: assert_(v > -9.0),  # taken everywhere, true
            lambda v: assert_(v > 5.0),  # excluded: never taken
            x,
        )

    p = propagate(trace(h))
    assert [o.status for o in p.obligations] == ["discharged"]
    assert not any("NOT EXAMINED" in o.detail for o in p.obligations)
    assert check(h, vacuity_mode="all").status == "VERIFIED"
    # the drop is not a coverage lie: the excluded branch's equations are
    # counted unreached, so the denominator still knows about them
    assert p.coverage.unreached >= 2  # its `gt` and its `stelling_assert`


# --- the direction the certificate is allowed to move ----------------------


def test_the_certificate_can_only_ever_withhold():
    """No probe change can move anything toward VERIFIED, because the
    certificate's only effect is to rewrite `violated-over-set` into
    `unknown`. Measured by forcing the search to certify EVERYTHING and to
    certify NOTHING and comparing what is discharged: the two extremes of
    the witness search bracket every probe grid there could be.
    """
    import stelling.propagate as P

    harnesses = [
        _cond(lambda x: x[0] - x[0] > 0.0),
        _cond(lambda x: x[0] > 0.0),
        _cond(lambda x: x[1] - x[0] > 1.0),
    ]

    def run(patched):
        original = P._reachability_witnesses
        P._reachability_witnesses = patched
        try:
            return [
                [o.status for o in propagate(trace(h)).obligations]
                for h in harnesses
            ]
        finally:
            P._reachability_witnesses = original

    nothing = run(lambda closed, p, **kw: frozenset())
    everything = run(
        lambda closed, p, **kw: frozenset(
            k for _, k in p.branch_violations + p.branch_nonvacuity_violations
        )
    )
    for a, b in zip(nothing, everything):
        assert a.count("discharged") == b.count("discharged"), (a, b)
        # and the only difference is which violations survive
        assert [s == "discharged" for s in a] == [s == "discharged" for s in b]
    assert nothing != everything  # the control: the patch does something
