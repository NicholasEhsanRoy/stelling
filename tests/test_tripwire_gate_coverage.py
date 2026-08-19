# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""WHICH constant-construction routes the armed trace gate actually watches.

The gate in :func:`stelling.preconditions.check` refuses to certify a trace
in which an integer constant was narrowed. What it has never said is which
ways of writing that constant it can SEE — and the answer is not "all of
them". ``jnp.full(shape, N, dt)`` narrows N in numpy before any jax primitive
runs, so the const-fold rule the tripwire hooks is handed a value that is
already in range, does not fire, and the gate certifies a program whose
written constant no longer exists.

THIS FILE IS THE INVENTORY, AND IT IS ASSERTED RATHER THAN ASSERTED-TO.
:data:`GATE_COVERAGE` declares a bucket per route; the test below MEASURES
every route through ``check()`` itself and compares. Prose alone is how this
gap survived: ``report.UNCOVERED`` has named ``jnp.full`` since the tripwire
shipped, and nothing anywhere would have gone red if a jax release had moved
a watched route into the silent set, or if a stelling change had quietly
stopped watching one. This is ``tests/test_doc_examples.py``'s
``EXPECTED_HASH_COVERAGE`` idiom, applied to the door list.

THE FOUR BUCKETS, which are four different facts and not four shades of one:

``watched``
    the constant is destroyed at TRACE time and the gate sees it: ``check()``
    returns UNKNOWN with ``trace unfaithful``. This is the covered case.

``unwatched``
    the constant is destroyed and the gate does NOT see it. ``check()``
    returns a verdict about a program the source does not describe. Every
    route in this bucket is a live hole, is named in ``report.UNCOVERED``,
    and is measured here so that it stays named.

``loud``
    jax itself raises ``OverflowError`` before anything is traced. Nothing
    silent happens, so there is nothing for a tripwire to catch.

``deferred``
    the written constant reaches the jaxpr INTACT — the narrowing is a
    ``convert_element_type`` the program performs at run time, not a
    transcription loss — so the trace gate has nothing to see and correctly
    sees nothing. These are not holes: the propagation's ``convert_element_
    type`` transfer declines them, and the test below drives that too, since
    "the gate ignores it" is only acceptable while something else does not.

WHY N=40000 AND int16. It is out of range for int16 and wraps to -25536, far
enough from the declared envelope that a verdict on the wrapped program and a
verdict on the written one cannot agree by accident.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

from stelling import _tripwire  # noqa: E402
from stelling._tripwire import report  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402

#: Out of int16 range; wraps to -25536.
OVER = 40000
DTYPE = jnp.int16
ENVELOPE = (0, 100)
#: Every route's obligation is ``value < BOUND``. True of the wrapped program
#: (-25536 + [0, 100]) and false of the written one (40000 + [0, 100]), so a
#: verdict cannot be right about both.
BOUND = 200


@pytest.fixture(autouse=True)
def _armed():
    status, _ = _tripwire.arm()
    if not status.armed:
        pytest.skip(f"tripwire could not arm: {status.code}")
    yield
    _tripwire.disarm()


#: One jitted helper per inline setting, built once and SHARED, because the
#: door being measured is a warm trace cache and a helper rebuilt per call
#: would be cold every time — the shape that made the original measurement
#: read as "no problem here".
_HELPERS: dict = {}


def _helper(inline):
    h = _HELPERS.get(inline)
    if h is None:
        h = jax.jit(lambda z: z + OVER, inline=inline)
        _HELPERS[inline] = h
    return h


#: route -> the body that writes OVER into a traced program.
ROUTES = {
    "x + N": lambda x: x + OVER,
    "N + x": lambda x: OVER + x,
    "x - N": lambda x: x - OVER,
    "x * N": lambda x: x * OVER,
    "x < N": lambda x: (x < OVER).astype(DTYPE),
    "x & N": lambda x: x & OVER,
    "jnp.maximum(x, N)": lambda x: jnp.maximum(x, OVER),
    "jnp.array(N).astype(dt)": lambda x: x + jnp.array(OVER).astype(DTYPE),
    "jnp.astype(jnp.array(N), dt)": lambda x: x + jnp.astype(jnp.array(OVER), DTYPE),
    "x.at[0].set(N)": lambda x: x.at[0].set(OVER),
    "x.at[0].add(N)": lambda x: x.at[0].add(OVER),
    "lax.cond branch": lambda x: lax.cond(
        x[0] > 0, lambda z: z + OVER, lambda z: z, x
    ),
    "lax.scan body": lambda x: lax.scan(lambda c, y: (c, y + OVER), x[0], x)[1],
    "lax.while_loop body": lambda x: lax.while_loop(
        lambda z: z[0] < 0, lambda z: z + OVER, x
    ),
    "jax.vmap(z + N)": lambda x: jax.vmap(lambda z: z + OVER)(x),
    "@jax.jit helper": lambda x: _helper(False)(x),
    "@jax.jit(inline=True) helper": lambda x: _helper(True)(x),
    "jnp.full(shape, N, dt)": lambda x: x + jnp.full(x.shape, OVER, DTYPE),
    "jnp.full_like(x, N)": lambda x: x + jnp.full_like(x, OVER),
    "lax.full(shape, N, dt)": lambda x: x + lax.full(x.shape, OVER, DTYPE),
    "lax.convert_element_type(N, dt)": lambda x: x
    + lax.convert_element_type(OVER, DTYPE),
    "np.asarray(N).astype(dt)": lambda x: x + np.array(OVER).astype(np.int16),
    "jnp.stack([x, jnp.full(N)])": lambda x: jnp.stack(
        [x, jnp.full(x.shape, OVER, DTYPE)]
    )[0],
    "jnp.array(N, dtype=dt)": lambda x: x + jnp.array(OVER, dtype=DTYPE),
    "jnp.asarray(N, dtype=dt)": lambda x: x + jnp.asarray(OVER, dtype=DTYPE),
    "jnp.int16(N)": lambda x: x + jnp.int16(OVER),
    "x // N": lambda x: x // OVER,
    "x % N": lambda x: x % OVER,
    "jnp.where(c, N, x)": lambda x: jnp.where(x > 0, OVER, x),
    "jnp.clip(x, 0, N)": lambda x: jnp.clip(x, 0, OVER),
    "jnp.pad(x, 1, constant_values=N)": lambda x: jnp.pad(
        x, 1, constant_values=OVER
    )[:2],
}

#: THE DECLARATION. Measured on jax 0.11.0 with x64 on, by the test below and
#: not typed from hope. Moving a route between buckets is a change to what
#: this tool can see and must be argued in this comment, not merely made
#: green: `unwatched` -> `watched` is a hole closing (say what closed it);
#: `watched` -> `unwatched` is a hole OPENING and needs a line in
#: `report.UNCOVERED` in the same commit, which the second test enforces.
GATE_COVERAGE = {
    "x + N": "watched",
    "N + x": "watched",
    "x - N": "watched",
    "x * N": "watched",
    "x < N": "watched",
    "x & N": "watched",
    "jnp.maximum(x, N)": "watched",
    "jnp.array(N).astype(dt)": "watched",
    "jnp.astype(jnp.array(N), dt)": "watched",
    "x.at[0].set(N)": "watched",
    "x.at[0].add(N)": "watched",
    "lax.cond branch": "watched",
    "lax.scan body": "watched",
    "lax.while_loop body": "watched",
    "jax.vmap(z + N)": "watched",
    # B15: both of these were `unwatched` on a WARM cache until the gate
    # started emptying jax's trace caches before the trace it watches. The
    # inline one is the reason that fix is an eviction and not a detector --
    # it leaves no nested jaxpr behind for a detector to see.
    "@jax.jit helper": "watched",
    "@jax.jit(inline=True) helper": "watched",
    # numpy narrows the value before any jax primitive runs, so the const-fold
    # rule is handed something already in range. All six are one mechanism.
    "jnp.full(shape, N, dt)": "unwatched",
    "jnp.full_like(x, N)": "unwatched",
    "lax.full(shape, N, dt)": "unwatched",
    "lax.convert_element_type(N, dt)": "unwatched",
    "np.asarray(N).astype(dt)": "unwatched",
    "jnp.stack([x, jnp.full(N)])": "unwatched",
    # jnp.array and friends VALIDATE the Python int against the dtype and
    # raise. The contrast with jnp.full, three lines up, is jax's and not
    # stelling's, and it is the single most useful thing on this page for a
    # reader deciding how to write a constant.
    "jnp.array(N, dtype=dt)": "loud",
    "jnp.asarray(N, dtype=dt)": "loud",
    "jnp.int16(N)": "loud",
    # the written 40000 is in the jaxpr; the narrowing is a runtime convert
    "x // N": "deferred",
    "x % N": "deferred",
    "jnp.where(c, N, x)": "deferred",
    "jnp.clip(x, 0, N)": "deferred",
    "jnp.pad(x, 1, constant_values=N)": "deferred",
}


def _harness(body):
    def h():
        x = any_array((2,), DTYPE, ENVELOPE)
        assert_(body(x) < BOUND)

    return h


def _bucket_once(body):
    try:
        verdict = check(_harness(body), vacuity_mode="inputs-only")
    except OverflowError:
        return "loud", None
    gated = (
        verdict.status == "UNKNOWN"
        and bool(verdict.notes)
        and "trace unfaithful" in verdict.notes[0]
    )
    if gated:
        return "watched", verdict
    # not gated: did the written constant survive into the traced program?
    from stelling._jax_compat import trace as _trace

    _tripwire.disarm()
    try:
        survives = str(OVER) in str(_trace(_harness(body)))
    finally:
        _tripwire.arm()
    return ("deferred" if survives else "unwatched"), verdict


def _measure(name):
    """The bucket this route is in, decided by driving it TWICE.

    Once is not a measurement here, and that is the whole of B15 in one line:
    the first ``check()`` of a route with a jitted helper traces that helper
    cold and sees the narrowing; every later one finds the cache warm and sees
    nothing. A single-call inventory would have recorded ``watched`` for a
    route that is watched exactly once per process, which is indistinguishable
    from covered right up until it matters.

    So the answer is the SECOND call's, and a route whose two calls disagree
    gets its own bucket rather than being averaged into silence.
    """
    body = ROUTES[name]
    first, _ = _bucket_once(body)
    second, verdict = _bucket_once(body)
    if first != second:
        return f"unstable:{first}->{second}", verdict
    return second, verdict


def test_the_declared_gate_coverage_is_the_measured_gate_coverage():
    """The inventory, held against what the routes actually do.

    An unwatched door that nobody enumerated is how B15 got here: the gate
    read its own silence as evidence of a clean trace, and no test anywhere
    asked which doors that silence covered.
    """
    measured = {name: _measure(name)[0] for name in ROUTES}
    assert measured == GATE_COVERAGE, (
        "the trace gate's coverage moved.\n"
        + "".join(
            f"  {k!r}: declared {GATE_COVERAGE.get(k)!r}, measured {v!r}\n"
            for k, v in sorted(measured.items())
            if GATE_COVERAGE.get(k) != v
        )
        + "Update GATE_COVERAGE *and* say in its comment why the new bucket "
        "is the right one. A route moving into `unwatched` is a hole opening "
        "and needs report.UNCOVERED updated in the same commit."
    )


def test_every_unwatched_route_really_certifies_a_destroyed_constant():
    """The `unwatched` bucket is only worth naming if it is really a hole.

    Both halves, because "the gate did not fire" is also what a route with
    nothing to catch produces: the verdict says VERIFIED, and the program jax
    actually executes violates the very obligation that VERIFIED discharged.
    """
    unwatched = [k for k, v in GATE_COVERAGE.items() if v == "unwatched"]
    assert unwatched, "the inventory declares no holes, so it proves nothing"
    x = jnp.array([0, 100], DTYPE)
    for name in unwatched:
        bucket, verdict = _measure(name)
        assert bucket == "unwatched", name
        assert verdict.status == "VERIFIED", (
            f"{name} is declared a silent hole but did not certify anything, "
            f"so the disclosure names the wrong consequence: {verdict.status}"
        )
        executed = np.asarray(jax.jit(ROUTES[name])(x))
        assert (executed < BOUND).all(), (
            f"{name}: the executed program does NOT satisfy the obligation "
            f"the wrapped one does, so this route is not the hole described"
        )
        assert int(executed.max()) < OVER, (
            f"{name}: {OVER} survived execution, so nothing was destroyed "
            "and this route does not belong in the unwatched bucket"
        )


def test_every_deferred_route_is_caught_by_the_transfer_instead():
    """`deferred` says the gate ignores it. Something else must not.

    The written constant reaches the jaxpr, so there is no transcription loss
    for the trace gate to report — but the program still wraps at run time,
    and a bucket that meant "the gate ignores it and so does everyone else"
    would be a hole wearing a reassuring name. Measured: the propagation's
    `convert_element_type` transfer declines the form, so the verdict is
    UNKNOWN and never VERIFIED.
    """
    deferred = [k for k, v in GATE_COVERAGE.items() if v == "deferred"]
    assert deferred, "no deferred routes, so this claim is vacuous"
    for name in deferred:
        bucket, verdict = _measure(name)
        assert bucket == "deferred", name
        assert verdict.status != "VERIFIED", (
            f"{name} is declared covered by the convert transfer rather than "
            f"by the gate, and NOTHING covered it: {verdict.status}"
        )


def test_every_unwatched_route_is_named_in_the_reports_coverage_claim():
    """Prose and measurement cannot drift while this holds.

    The report is what a user reads to answer "what does it not see". A door
    measured here and unnamed there is a door the reader is not told about.
    """
    text = " ".join(report.UNCOVERED)
    for name, phrase in {
        "jnp.full(shape, N, dt)": "jnp.full(shape, N, dt)",
        "jnp.full_like(x, N)": "jnp.full_like(x, N)",
        "lax.full(shape, N, dt)": "lax.full(shape, N, dt)",
        "lax.convert_element_type(N, dt)": "lax.convert_element_type(N, dt)",
        "np.asarray(N).astype(dt)": "np.asarray(N).astype(dt)",
        "jnp.stack([x, jnp.full(N)])": "jnp.stack([x, jnp.full(shape, N, dt)])",
    }.items():
        assert GATE_COVERAGE[name] == "unwatched", name
        assert phrase in text, f"report.UNCOVERED does not name {phrase}"
    assert "GATE_COVERAGE" in text, (
        "the report does not point at this inventory, so a reader who wants "
        "the enumerated version has no way to find it"
    )


def test_the_warm_trace_cache_door_is_recorded_as_closed_for_the_GATE_only():
    """B15's door, and the exact half of it that is still open.

    `check()` empties jax's trace caches, so a verdict's observation is
    complete. The SESSION report has no such moment — it watches whatever the
    suite happens to trace — so a user's jitted function first traced in an
    earlier test is still never re-traced and still never reported. Saying
    only the first half would be the same over-claim in a new place.
    """
    text = " ".join(report.UNCOVERED)
    assert "WARM TRACE CACHE" in text
    assert "inline=True" in text
    assert "preconditions.check()" in text and "empties" in text
    assert "session report" in text.lower()


def test_the_report_does_not_still_say_clear_caches_is_never_called():
    """A disclosure that B15 made false, held down where it was made false.

    The session report's arm-order line read *"`jax.clear_caches()` is NOT
    called -- that would change your suite's timing and behaviour to flatter
    a report"*. It is still true of the report and it stopped being true of
    the session the moment the gate started calling it, and a reader who
    budgets their suite's runtime off that sentence would be budgeting off a
    claim about a different program. The line now says which of the two it
    is talking about, and this asserts BOTH halves so that deleting either
    one goes red.
    """
    from stelling._tripwire import record, report

    text = " ".join(report.render(_tripwire.Status("armed"), record.Recorder()))
    assert "arm order" in text
    assert "THIS REPORT never calls `jax.clear_caches()`" in text, (
        "the report no longer says that IT does not clear caches"
    )
    assert "`preconditions.check()` DOES call it" in text, (
        "the report claims caches are never cleared, which the trace gate "
        "has made false for any session that calls check()"
    )


def test_the_inventory_discriminates():
    """Four buckets, each non-empty, and the predicate that sorts them is not
    a constant function."""
    buckets = set(GATE_COVERAGE.values())
    assert buckets == {"watched", "unwatched", "loud", "deferred"}
    for b in buckets:
        assert sum(1 for v in GATE_COVERAGE.values() if v == b) >= 3, b
    assert set(GATE_COVERAGE) == set(ROUTES), (
        "a route with no declared bucket, or a bucket with no route: the "
        "inventory and the probes must be the same set"
    )
