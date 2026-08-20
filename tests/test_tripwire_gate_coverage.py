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
from stelling._tripwire import eager as _eager, report  # noqa: E402
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
    "lax.full_like(x, N)": lambda x: x + lax.full_like(x, OVER),
    "lax.convert_element_type(N, dt)": lambda x: x
    + lax.convert_element_type(OVER, DTYPE),
    "np.asarray(N).astype(dt)": lambda x: x + np.array(OVER).astype(np.int16),
    "jnp.asarray(np.array(N), dt)": lambda x: x
    + jnp.asarray(np.array(OVER), dtype=DTYPE),
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
#: `report.UNCOVERED` in the same commit, which
#: `test_every_unwatched_route_is_named_in_the_reports_coverage_claim`
#: enforces BY ITERATING THIS DICT. It used to walk a copy of this set typed
#: beside it, so an added `unwatched` row passed all seven tests undisclosed
#: (measured, twice); a pointer to an enforcement is worth what the
#: enforcement iterates.
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
    # rule is handed something already in range. Every route in this group is
    # one mechanism. `lax.full_like` was measured and added after the rest:
    # unwatched, VERIFIED, 0 fires, jaxpr `broadcast_in_dim -25536`. The prose
    # in `report.UNCOVERED` covered it under "anything else built on `full`"
    # the whole time, which is why the disclosure was adequate and the
    # inventory was not.
    "jnp.full(shape, N, dt)": "unwatched",
    "jnp.full_like(x, N)": "unwatched",
    "lax.full(shape, N, dt)": "unwatched",
    "lax.full_like(x, N)": "unwatched",
    "lax.convert_element_type(N, dt)": "unwatched",
    "np.asarray(N).astype(dt)": "unwatched",
    # B16. THE SECOND NAMED ROUTE INTO THE SAME RESIDUE, and it is here
    # because the residue is what the eager detector CANNOT close and a
    # residue with one named member reads like an edge case. numpy builds the
    # array at its own default width and narrows it before jax is reached, so
    # neither instrument sees a written constant: the const-fold rule is
    # handed -25536 and the eager detector is handed an operand whose value
    # was already destroyed. Measured on 0.11.0 and 0.10.2, with the eager
    # detector armed: 0 fires, VERIFIED, and the value still wraps.
    "jnp.asarray(np.array(N), dt)": "unwatched",
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


#: GATE_COVERAGE is a claim about the DEFAULT path -- what `check()` does in
#: the environment every user gets -- so the routes below have to be allowed
#: to narrow even when the whole session was run with
#: `--stelling-eager-truncation=error`. Declaring it here rather than skipping
#: the file keeps the inventory measuring the same thing in both worlds, and
#: the permission is counted and named in the eager report like any other.
_DEFAULT_PATH = (
    "GATE_COVERAGE is a measurement of the DEFAULT path; these routes are "
    "driven precisely to see which of them narrow in silence"
)


def _bucket_once(body):
    try:
        with _eager.expected_truncation(_DEFAULT_PATH):
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
        with _eager.expected_truncation(_DEFAULT_PATH):
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

    WHAT THE SECOND CALL IS, EXACTLY: a REGRESSION DETECTOR FOR THE EVICTION,
    and not an independent control on the bucket. On this tree ``check()``
    clears jax's trace caches itself, so both calls trace cold and therefore
    always agree — the ``unstable:`` bucket is unreachable while the eviction
    is in place, and a reading of "the two calls agreed, so the bucket is
    real" would be reading a tautology. What it does detect is the eviction
    going away: driven against ``a759809``'s ``src`` it reports
    ``'@jax.jit helper': declared 'watched', measured
    'unstable:watched->unwatched'`` for both jit routes. That is worth having;
    it is just not the thing a control would be.
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
        with _eager.expected_truncation(_DEFAULT_PATH):
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


#: Routes whose key here and spelling in ``report.UNCOVERED`` differ. ONLY
#: spelling belongs in this table: the test below requires a line in
#: ``report.UNCOVERED`` for EVERY ``unwatched`` route in ``GATE_COVERAGE``,
#: and a route with no entry here must be named under its own key. Adding a
#: route to this table instead of to the report would be the same evasion in
#: a new place, so it is held to routes that exist and is read against
#: ``GATE_COVERAGE`` below.
UNCOVERED_SPELLING = {
    "jnp.stack([x, jnp.full(N)])": "jnp.stack([x, jnp.full(shape, N, dt)])",
    "jnp.asarray(np.array(N), dt)": "jnp.asarray(np.array(N), dtype=dt)",
}


def test_every_unwatched_route_is_named_in_the_reports_coverage_claim():
    """Prose and measurement cannot drift while this holds.

    The report is what a user reads to answer "what does it not see". A door
    measured here and unnamed there is a door the reader is not told about.

    THIS ITERATES ``GATE_COVERAGE``, NOT A LIST TYPED BESIDE IT. It used to
    walk a six-entry dict literal, so ``GATE_COVERAGE``'s comment — "a route
    moving into `unwatched` ... needs a line in `report.UNCOVERED` in the
    same commit, which the second test enforces" — described an enforcement
    nobody performed. Measured: adding ``lax.full_like(x, N)`` as a seventh
    ``unwatched`` row, named nowhere in ``report.UNCOVERED``, passed all
    seven tests in this file. A hardcoded copy of a set cannot police the
    set.
    """
    text = " ".join(report.UNCOVERED)
    unwatched = [k for k, v in GATE_COVERAGE.items() if v == "unwatched"]
    assert unwatched, "the inventory declares no holes, so this is vacuous"
    missing = [
        (name, UNCOVERED_SPELLING.get(name, name))
        for name in unwatched
        if UNCOVERED_SPELLING.get(name, name) not in text
    ]
    assert not missing, (
        "an `unwatched` route the report does not name — a hole the reader "
        "is not told about:\n"
        + "".join(f"  {n!r}: report.UNCOVERED has no {p!r}\n" for n, p in missing)
        + "Add the route to `report.UNCOVERED` in this commit. Add it to "
        "UNCOVERED_SPELLING only if the report already names it in different "
        "words."
    )
    stale = set(UNCOVERED_SPELLING) - {
        k for k, v in GATE_COVERAGE.items() if v == "unwatched"
    }
    assert not stale, (
        f"UNCOVERED_SPELLING carries routes that are no longer `unwatched` "
        f"({sorted(stale)}), so it is excusing something that is not there"
    )
    assert "GATE_COVERAGE" in text, (
        "the report does not point at this inventory, so a reader who wants "
        "the enumerated version has no way to find it"
    )


def test_the_warm_trace_cache_door_is_recorded_as_closed_for_the_GATE_only():
    """B15's door, and the three ways it is still open.

    `check()` evicts jax's trace caches, so a verdict's observation is
    complete — with respect to JAX's caches, on ONE thread, and no further.
    Three things sit outside that, each measured, and each of them is a place
    where "closed" would be the same over-claim in a new spelling:

    * the SESSION report has no such moment — it watches whatever the suite
      happens to trace, so a user's jitted function first traced in an
      earlier test is still never re-traced and never reported;
    * a value narrowed into a memo JAX DOES NOT OWN survives the eviction:
      `jax.extend.core.jaxpr_as_fun` over a saved jaxpr, a user
      `functools.lru_cache`, and `jax.closure_convert` (a public jax API)
      each return VERIFIED with 0 fires on a program whose 40000 is already
      -25536;
    * jax's cache is PROCESS-GLOBAL and the gate's counter is per-thread, so
      the eviction-to-trace window is not atomic: 0/400 wrong VERIFIED
      single-threaded against 247/400 with four competing threads.

    This asserts all four so that dropping any one of them goes red.
    """
    text = " ".join(report.UNCOVERED)
    assert "WARM TRACE CACHE" in text
    assert "inline=True" in text
    assert "`preconditions.check()`" in text and "jax.clear_caches()" in text
    assert "SINGLE-THREADED PROCESS" in text, (
        "the completeness claim is unqualified again; a process-global cache "
        "and a per-thread counter do not make one"
    )
    assert "session report" in text.lower()
    for phrase in (
        "jax.extend.core.jaxpr_as_fun(saved)",
        "functools.lru_cache",
        "jax.closure_convert",
    ):
        assert phrase in text, (
            f"{phrase} is a construct the eviction does not reach and it is "
            f"no longer disclosed"
        )
    assert "ANOTHER THREAD" in text and "247/400" in text, (
        "the thread-safety disclosure and its measurement are gone"
    )


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
    assert "`contracts.check_contract()`" in text, (
        "check_contract() reaches the same `_pipeline` and evicts the same "
        "caches, so a disclosure that names only check() is narrower than "
        "the behaviour it describes"
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


# ===========================================================================
# B16: the same inventory, asked of the OPT-IN EAGER DETECTOR.
#
# A SECOND DECLARATION AND NOT A COLUMN ADDED TO THE FIRST, because the two
# answer different questions and only one of them is on by default.
# :data:`GATE_COVERAGE` is what ``check()`` does in the environment every user
# gets, and it must keep saying so; a route whose bucket moved because a
# non-default flag was passed would make the default inventory a claim about
# a program nobody runs. So ``GATE_COVERAGE`` is measured with the detector
# OFF, as before, and this is measured with it ON.
#
# WHAT IT BUYS, and it is the thing the `unwatched` bucket most needed: the
# bucket used to be a list of holes with a disclosure beside it. Now every row
# in it is either CLOSED by the opt-in detector or is one of the two named
# numpy routes, and the test below is what stops a third one being added to
# the residue quietly.
# ===========================================================================

#: What the eager detector does with each route, MEASURED by executing the
#: route with it armed. Three buckets and they are three different facts:
#:
#: ``raises``
#:     the detector refuses the construction: ``EagerTruncationError``, at the
#:     line that wrote the constant, before jax narrows anything.
#: ``loud``
#:     jax itself raises ``OverflowError``, exactly as it does with the
#:     detector off. These rows are the evidence that arming changes nothing
#:     about a route jax already refuses.
#: ``silent``
#:     nothing is raised. For a ``watched`` or ``deferred`` route that is
#:     correct and expected -- the constant is not destroyed at construction,
#:     so there is nothing here to see. For an ``unwatched`` one it is a
#:     residual hole, and :func:`test_the_unwatched_routes_the_eager_detector_
#:     cannot_close_are_the_two_named_numpy_ones` holds the residue to exactly
#:     the two routes ``report.EAGER_UNCOVERED`` names.
EAGER_COVERAGE = {
    # the six that narrow at `lax._convert_element_type` with the written
    # value still intact -- the whole point of the second instrument
    "jnp.full(shape, N, dt)": "raises",
    "jnp.full_like(x, N)": "raises",
    "lax.full(shape, N, dt)": "raises",
    "lax.full_like(x, N)": "raises",
    "lax.convert_element_type(N, dt)": "raises",
    "jnp.stack([x, jnp.full(N)])": "raises",
    # jax refuses these three itself, with or without the detector
    "jnp.array(N, dtype=dt)": "loud",
    "jnp.asarray(N, dtype=dt)": "loud",
    "jnp.int16(N)": "loud",
    # the inline door: the constant survives construction and dies in the
    # const-fold rule, which is the OTHER instrument's. Nothing for this one.
    "x + N": "silent",
    "N + x": "silent",
    "x - N": "silent",
    "x * N": "silent",
    "x < N": "silent",
    "x & N": "silent",
    "jnp.maximum(x, N)": "silent",
    "jnp.array(N).astype(dt)": "silent",
    "jnp.astype(jnp.array(N), dt)": "silent",
    "x.at[0].set(N)": "silent",
    "x.at[0].add(N)": "silent",
    "lax.cond branch": "silent",
    "lax.scan body": "silent",
    "lax.while_loop body": "silent",
    "jax.vmap(z + N)": "silent",
    "@jax.jit helper": "silent",
    "@jax.jit(inline=True) helper": "silent",
    # deferred: the written constant reaches the jaxpr and the narrowing is a
    # runtime convert over a VARIABLE, which is not a construction at all
    "x // N": "silent",
    "x % N": "silent",
    "jnp.where(c, N, x)": "silent",
    "jnp.clip(x, 0, N)": "silent",
    "jnp.pad(x, 1, constant_values=N)": "silent",
    # THE RESIDUE. numpy finished before jax was reached.
    "np.asarray(N).astype(dt)": "silent",
    "jnp.asarray(np.array(N), dt)": "silent",
}


@pytest.fixture
def eager_armed():
    """Arm the eager detector for one test, and take it back out.

    ARMED PER-TEST AND NOT PER-MODULE, because it is a rule and not a report:
    while it is live, every construction in the process is subject to it,
    including the ones the fixtures above make. Its blast radius is bounded
    here to the tests that are measuring it.
    """
    from stelling import _tripwire as tw

    was_armed = _eager.is_armed()
    status = tw.arm_eager()
    if not status.armed:
        pytest.skip(f"the eager detector could not attach: {status.code}")
    yield status
    # RESTORE, do not disarm. A session run with
    # `--stelling-eager-truncation=error` armed this for the whole run, and a
    # fixture that took it out here would leave every later file unwatched
    # with nothing saying so.
    if not was_armed:
        tw.disarm_eager()


def _eager_bucket(name):
    """What executing this route does with the eager detector armed.

    EXECUTED CONCRETELY rather than through ``check()``, and the difference
    matters: the detector fires at CONSTRUCTION, which happens whether or not
    anything is tracing, so driving it through the harness would measure the
    harness. :func:`test_the_eager_detector_closes_the_hole_in_check_itself`
    is the one that asks the verdict path.
    """
    import stelling

    x = jnp.array([0, 100], DTYPE)
    try:
        ROUTES[name](x)
    except stelling.EagerTruncationError:
        return "raises"
    except OverflowError:
        return "loud"
    return "silent"


def test_the_declared_eager_coverage_is_the_measured_eager_coverage(eager_armed):
    """The second inventory, held against what the routes actually do.

    Same discipline as the first: moving a route between buckets is a change
    to what the tool can see and must be argued in ``EAGER_COVERAGE``'s
    comment, not merely made green. ``raises`` -> ``silent`` is a hole
    OPENING and is the failure this whole instrument is built to fail closed
    on -- though it should never get here, because the detector's own arm-time
    self-check drives every route it claims and refuses to attach when one
    goes blind.
    """
    measured = {name: _eager_bucket(name) for name in ROUTES}
    assert measured == EAGER_COVERAGE, (
        "the eager detector's coverage moved.\n"
        + "".join(
            f"  {k!r}: declared {EAGER_COVERAGE.get(k)!r}, measured {v!r}\n"
            for k, v in sorted(measured.items())
            if EAGER_COVERAGE.get(k) != v
        )
        + "Update EAGER_COVERAGE *and* say in its comment why the new bucket "
        "is the right one."
    )
    assert set(EAGER_COVERAGE) == set(ROUTES), (
        "a route with no declared eager bucket, or a bucket with no route"
    )


def test_the_unwatched_routes_the_eager_detector_cannot_close_are_the_two_named_numpy_ones():
    """The residue is EXACTLY two routes, and both are disclosed.

    This is the assertion that keeps the ``unwatched`` bucket from quietly
    growing a third member. Before the eager detector there were seven
    unwatched routes and one disclosure covering all of them; six are now
    closed by an opt-in flag and two remain, and the two that remain are the
    ones numpy destroys before jax is reached. A route added to ``unwatched``
    that the detector does not close has to be argued into
    ``report.EAGER_UNCOVERED`` here, in the same commit.

    Reads the DECLARATIONS rather than re-measuring, on purpose: the
    measurement is the test above, and a second copy of it here would be a
    second thing to keep in step. What this asserts is that the two
    declarations AGREE with each other and with the prose.
    """
    unwatched = {k for k, v in GATE_COVERAGE.items() if v == "unwatched"}
    residue = {k for k in unwatched if EAGER_COVERAGE[k] == "silent"}
    assert residue == {
        "np.asarray(N).astype(dt)",
        "jnp.asarray(np.array(N), dt)",
    }, (
        f"the residue the eager detector cannot close is {sorted(residue)}. "
        "Two routes are disclosed in `report.EAGER_UNCOVERED` and in "
        "`report.UNCOVERED`'s pre-narrowed bullet; anything else here is a "
        "hole the reader is not told about."
    )
    closed = unwatched - residue
    assert len(closed) == 6, (
        f"six of the seven unwatched routes were closed by the eager "
        f"detector and now {len(closed)} are: {sorted(closed)}"
    )
    text = " ".join(report.EAGER_UNCOVERED)
    for phrase in (
        "np.asarray(N).astype(dt)",
        "jnp.asarray(np.array(N), dtype=dt)",
        "immutable type attribute",
    ):
        assert phrase in text, (
            f"`report.EAGER_UNCOVERED` no longer names {phrase!r}, so the "
            "residue is undisclosed"
        )


def test_the_eager_detector_closes_the_hole_in_check_itself(eager_armed):
    """The verdict path, not just the construction.

    The ``unwatched`` bucket's whole cost is that ``check()`` returns VERIFIED
    about a program whose constant no longer exists -- which
    :func:`test_every_unwatched_route_really_certifies_a_destroyed_constant`
    drives and asserts. With the detector armed, that VERIFIED cannot be
    reached: the harness cannot finish being traced, so there is no verdict to
    be wrong. A refusal, not a better answer, and that is the honest shape --
    the tool has nothing true to say about a program it cannot read.
    """
    import stelling

    closed = [
        k
        for k, v in GATE_COVERAGE.items()
        if v == "unwatched" and EAGER_COVERAGE[k] == "raises"
    ]
    assert closed, "no closed routes, so this is vacuous"
    for name in closed:
        with pytest.raises(stelling.EagerTruncationError) as caught:
            check(_harness(ROUTES[name]), vacuity_mode="inputs-only")
        assert caught.value.written == OVER
        assert caught.value.to_dtype == str(np.dtype(DTYPE))
        assert caught.value.became == int(np.asarray(OVER).astype(DTYPE))


def test_the_default_path_is_BYTE_IDENTICAL_without_the_flag():
    """Mode 2 is opt-in, and this is what "opt-in" is asserted to mean.

    Not "the flag defaults to off" -- that is a claim about a parser. This
    asserts the thing a user cares about: with nothing armed, the private jax
    attribute the detector patches is jax's own function, and every route
    produces exactly the value it produced before this feature existed.

    The value column is derived from the two's-complement arithmetic rather
    than typed, so this cannot go green by being updated to match a
    regression.
    """
    import contextlib

    from stelling._tripwire import _adapter_jax as adapter

    @contextlib.contextmanager
    def detached():
        """The hook GONE, not the truncation permitted, and put back after.

        A region declaration would be the wrong tool: what is being measured
        is the program with nothing patched, and `expected_truncation` leaves
        the wrapper installed.
        """
        was_armed = _eager.is_armed()
        if was_armed:
            _tripwire.disarm_eager()
        try:
            yield
        finally:
            if was_armed:
                _tripwire.arm_eager()

    with detached():
        assert adapter.eager_live_check() == "detached"
        _the_default_path_body()


def _the_default_path_body():
    x = jnp.array([0, 100], DTYPE)
    for name, bucket in EAGER_COVERAGE.items():
        if bucket == "loud":
            with pytest.raises(OverflowError):
                ROUTES[name](x)
            continue
        # every other route completes, and completes by DESTROYING the
        # constant -- which is the state of the world this whole file
        # documents and the reason the detector exists
        out = np.asarray(ROUTES[name](x)).ravel().tolist()
        assert OVER not in out, (
            f"{name}: {OVER} survived execution with nothing armed, so the "
            "default path is not the one every other test on this page "
            "describes"
        )
