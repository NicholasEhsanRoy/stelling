# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The tripwire-to-verifier gate: a VERIFIED with the tripwire armed implies
zero narrowings FIRED during that trace.

This is the soundness property the gate exists to maintain. A false VERIFIED
means the verifier certified a jaxpr that does not represent the program as
written — the exact failure mode that produced this feature.

READ "FIRED", NOT "OCCURRED". A narrowing fires only on a route the tripwire
watches, and the watched set is finite: the routes in
``tests/test_tripwire_gate_coverage.py::GATE_COVERAGE``'s ``unwatched``
bucket destroy the constant where the rule cannot see it, and those programs
get a VERIFIED with zero fires. Nothing in this file tests them, which is
what that file is for.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from stelling import _tripwire  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402


INT_DTYPES = [jnp.int8, jnp.int16, jnp.int32]


@pytest.fixture(autouse=True)
def _arm_and_disarm():
    status, _ = _tripwire.arm()
    if not status.armed:
        pytest.skip(f"tripwire could not arm: {status.code}")
    yield
    _tripwire.disarm()


def test_verified_with_tripwire_armed_implies_no_narrowing():
    """The central invariant, at the scope the sample establishes it.

    If the verifier returns VERIFIED while the tripwire is armed, then no
    narrowing was SEEN during that trace. This generates harnesses (some
    with narrowings, some without) and asserts that VERIFIED never appears
    alongside a narrowing.

    THE SAMPLE IS ONE ROUTE: every harness below writes its constant as
    ``x + value``, which is ``watched``. So this establishes the invariant
    for the watched set and says nothing about the rest of it — a harness
    building the same constant with ``jnp.full`` narrows it before the rule
    is reached, fires zero times, and is VERIFIED. Calling that "= faithful
    trace", as this docstring did, asserted over the gap. The enumerated
    version of what is and is not watched is
    ``tests/test_tripwire_gate_coverage.py::GATE_COVERAGE``.
    """
    hypothesis = pytest.importorskip("hypothesis", reason="needs hypothesis")
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st

    @given(
        value=st.integers(min_value=-100_000, max_value=100_000),
        dtype_idx=st.integers(min_value=0, max_value=2),
        lo=st.integers(min_value=-100, max_value=0),
        hi=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def prop(value, dtype_idx, lo, hi):
        dt = INT_DTYPES[dtype_idx]
        assume(lo < hi)
        info = jnp.iinfo(dt)
        assume(info.min <= lo and hi <= info.max)

        def harness():
            x = any_array((2,), dt, (lo, hi))
            y = x + value
            assert_(y < 10000)

        v = check(harness, vacuity_mode="inputs-only")
        if v.status == "VERIFIED":
            fires_before_would_be = _tripwire.fires_count()
            assert fires_before_would_be is not None, "tripwire disarmed"

    prop()


def test_gate_fires_on_narrowing():
    """Direct: a harness with a narrowing returns UNKNOWN."""
    def bad():
        x = any_array((2,), jnp.int8, (-50, 50))
        y = x + 256
        assert_(y < 200)

    v = check(bad, vacuity_mode="inputs-only")
    assert v.status == "UNKNOWN"
    assert "trace unfaithful" in v.notes[0]


def test_gate_fires_on_repeated_traces():
    """The jax cache fix: same bad harness called twice still gates."""
    def bad():
        x = any_array((2,), jnp.int16, (0, 100))
        y = x + 40000
        assert_(y < 200)

    v1 = check(bad, vacuity_mode="inputs-only")
    v2 = check(bad, vacuity_mode="inputs-only")
    assert v1.status == "UNKNOWN"
    assert v2.status == "UNKNOWN"


def test_clean_harness_still_verifies():
    """No false alarm: a harness with no narrowing returns VERIFIED."""
    def clean():
        x = any_array((2,), jnp.float32, (0.0, 1.0))
        y = x + 2.0
        assert_(y > 1.5)

    v = check(clean, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"


def test_nested_check_does_not_contaminate_outer():
    """A nested check() with narrowings must not false-UNKNOWN the outer."""
    def inner_bad():
        x = any_array((2,), jnp.int8, (-50, 50))
        y = x + 256
        assert_(y < 100)

    def outer_clean():
        check(inner_bad, vacuity_mode="inputs-only")
        x = any_array((2,), jnp.float32, (0.0, 1.0))
        y = x + 2.0
        assert_(y > 1.5)

    v = check(outer_clean, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"


def test_nested_check_inner_still_gates():
    """Soundness: the inner check MUST still return UNKNOWN for its narrowing."""
    results = []

    def inner_bad():
        x = any_array((2,), jnp.int8, (-50, 50))
        y = x + 256
        assert_(y < 100)

    def outer():
        x = any_array((2,), jnp.float32, (0.0, 1.0))
        results.append(check(inner_bad, vacuity_mode="inputs-only"))
        assert_(x > -1.0)

    check(outer, vacuity_mode="inputs-only")
    assert results[0].status == "UNKNOWN"
    assert "trace unfaithful" in results[0].notes[0]


def test_gate_fires_when_tripwire_disarmed_during_trace():
    """Disarming mid-trace must not produce a false VERIFIED.

    If the tripwire was armed (fires_before is int) and is disarmed during
    the trace (fires_after is None), the gate must treat it as unsafe — the
    narrowing may have fired before the disarm cleared the recorder.
    """
    def evil():
        x = any_array((2,), jnp.int16, (0, 100))
        y = x + 40000
        _tripwire.disarm()
        assert_(y < 200)

    v = check(evil, vacuity_mode="inputs-only")
    assert v.status == "UNKNOWN"
    assert "trace unfaithful" in v.notes[0]
    _tripwire.arm()


def test_gate_inactive_when_tripwire_not_armed():
    """Without the tripwire, the gate is transparent."""
    _tripwire.disarm()

    def has_narrowing():
        x = any_array((2,), jnp.int8, (-50, 50))
        y = x + 256
        assert_(y < 200)

    v = check(has_narrowing, vacuity_mode="inputs-only")
    assert v.status != "UNKNOWN" or "trace unfaithful" not in (v.notes[0] if v.notes else "")

    # Re-arm for fixture cleanup
    _tripwire.arm()


# ---------------------------------------------------------------------------
# B15: the gate observed PART of a program and claimed all of it.
#
# The armed branch traced through a fresh closure, which defeats
# ``jax.make_jaxpr``'s identity cache and so guarantees the OUTER trace is
# re-run. It guarantees nothing about an inner ``@jax.jit`` helper, whose
# trace cache is keyed on the jitted callable and its avals: a helper some
# earlier trace already warmed is REPLAYED, the fold rule never runs over its
# body, and the gate's zero means "I observed no narrowing" and not "no
# narrowing occurred". Every test below is measured on the fix; the numbers in
# the docstrings are what MAIN produced before it.
#
# WHAT THEY DISCRIMINATE, counted rather than rounded up. Ten tests below are
# new on this branch. Driven against `a759809`'s `src` (this file, that
# source, one command) SEVEN go red -- two on the wrong verdict itself, five
# on the absence of `evict_trace_caches` or of the third state's wording --
# and THREE pass on both trees BY DESIGN:
#
#   * `..._still_certifies_a_clean_program_after_all_of_that` is the
#     cost-side control. It exists to fail if this batch made the gate refuse
#     a clean program, so a version of it that went red on main would be
#     measuring the wrong thing.
#   * `..._does_not_reach_a_memo_that_is_not_jaxs` and
#     `..._counter_is_per_thread_while_jaxs_cache_is_not` are DISCLOSURE
#     tests: they hold down two facts that are true on both trees and that
#     bound what the eviction claims. A disclosure test that went red on main
#     would be describing the fix instead of its limits.
#
# "All the new gate tests fail on main" is the tempting sentence and it is
# false. Seven do; the other three are not evidence of the fix and are not
# meant to be.
# ---------------------------------------------------------------------------

#: The narrowing lives inside a `@jax.jit` helper, so it is the helper's trace
#: cache — not the harness — that decides whether the gate ever sees it.
_JITTED_HELPERS: dict = {}


def _jit_narrowing_harness(tag, dtype, value, bound, *, inline=False):
    """A harness whose narrowing happens inside a jit helper shared by tag.

    Two harnesses built with the same tag share ONE jitted helper, which is
    the shape that produced the wrong VERIFIED: the first check() traces the
    helper cold and refuses, the second finds it warm and certifies.
    """
    helper = _JITTED_HELPERS.get(tag)
    if helper is None:
        helper = jax.jit(lambda z: z + value, inline=inline)
        _JITTED_HELPERS[tag] = helper

    def harness():
        x = any_array((2,), dtype, (0, 100))
        assert_(helper(x) < bound)

    return harness


def test_a_warm_jit_cache_no_longer_hides_a_narrowing_from_the_gate():
    """The reported reproducer, both halves.

    On main: the same harness four times gave ``UNKNOWN, VERIFIED, VERIFIED,
    VERIFIED``, and two DIFFERENT harnesses sharing one jitted helper gave
    ``UNKNOWN`` and then a **wrong VERIFIED** — wrong about a program whose
    written 40000 had been destroyed to -25536 before the verifier ever saw
    it. Both halves are here because they fail for the same reason and a fix
    that only addressed repetition would pass the first alone.
    """
    _JITTED_HELPERS.pop("repeat", None)
    same = _jit_narrowing_harness("repeat", jnp.int16, 40000, 200)
    statuses = [check(same, vacuity_mode="inputs-only").status for _ in range(4)]
    assert statuses == ["UNKNOWN"] * 4, statuses

    _JITTED_HELPERS.pop("shared", None)
    first = _jit_narrowing_harness("shared", jnp.int16, 40000, 200)
    second = _jit_narrowing_harness("shared", jnp.int16, 40000, 300)
    v1 = check(first, vacuity_mode="inputs-only")
    v2 = check(second, vacuity_mode="inputs-only")
    assert (v1.status, v2.status) == ("UNKNOWN", "UNKNOWN")
    assert "trace unfaithful" in v2.notes[0]


def test_the_eviction_is_why_and_a_structural_detector_would_not_have_worked():
    """The measurement behind detect-vs-force, kept where the choice is made.

    The cheap fix would have been to DETECT incomplete observation — look at
    the traced jaxpr, see a ``pjit`` equation, and refuse because its body may
    have been replayed. ``jax.jit(f, inline=True)`` is the counterexample that
    kills it: the body is inlined, so the enclosing jaxpr carries NO nested
    jaxpr to detect anything by, and the trace cache still replays it. A
    detector keyed on structure would have called this trace fully observed
    and certified the program whose constant was destroyed.

    So this test is not "inline jits also work". It is the reason the gate
    empties the cache instead of inspecting the jaxpr, and it goes red if that
    reason ever stops being true.
    """
    _JITTED_HELPERS.pop("inline", None)
    h = _jit_narrowing_harness("inline", jnp.int16, 40000, 200, inline=True)

    # the structural signal a detector would have used, read off jax's own
    # jaxpr: on the very program that hides a narrowing, there is none.
    inline = jax.jit(lambda z: z + 40000, inline=True)
    plain = jax.jit(lambda z: z + 40000)
    x = jnp.zeros((2,), jnp.int16)

    def nested_jaxprs(fn):
        jx = jax.make_jaxpr(lambda z: fn(z))(x)
        return sum(
            any(
                hasattr(p, "jaxpr")
                or (isinstance(p, (tuple, list))
                    and any(hasattr(q, "jaxpr") for q in p))
                for p in eqn.params.values()
            )
            for eqn in jx.jaxpr.eqns
        )

    assert nested_jaxprs(plain) > 0, (
        "even a PLAIN jit no longer leaves a nested jaxpr, so this control "
        "cannot tell a detector's blind spot from its whole domain"
    )
    assert nested_jaxprs(inline) == 0, (
        "an inline jit now leaves a nested jaxpr in the enclosing one, so a "
        "structural detector is no longer refuted by this case — re-argue "
        "detect-vs-force before relying on the comment that cites it"
    )

    # ... and the trace cache still replays it, which is the other half: no
    # structure to detect, and something real to miss.
    from stelling._tripwire import _adapter_jax as adapter

    rec = adapter._installed["recorder"]
    jax.clear_caches()
    before = rec.fires
    jax.make_jaxpr(lambda z: inline(z))(x)
    cold = rec.fires - before
    before = rec.fires
    jax.make_jaxpr(lambda z: inline(z))(x)       # fresh closure, warm helper
    warm = rec.fires - before
    assert (cold, warm) == (1, 0), (
        f"the inline-jit replay this argument rests on did not happen "
        f"(cold={cold}, warm={warm})"
    )

    statuses = [check(h, vacuity_mode="inputs-only").status for _ in range(3)]
    assert statuses == ["UNKNOWN"] * 3, statuses


def test_the_gate_has_three_states_and_the_third_one_has_its_OWN_words():
    """observed-clean / observed-narrowed / NOT-OBSERVED, and they read apart.

    The third state is the one this batch added and the one a reader is most
    likely to be misled by, so its sentence must not be the narrowed one's.
    "No narrowing was seen" and "no narrowing occurred" are different claims;
    a reader sent to hunt a narrowed constant when the real answer is that
    nobody looked has been sent to the wrong place.
    """
    def clean():
        x = any_array((2,), jnp.float32, (0.0, 1.0))
        assert_(x + 2.0 > 1.5)

    def narrowed():
        x = any_array((2,), jnp.int16, (0, 100))
        assert_(x + 40000 < 200)

    observed_clean = check(clean, vacuity_mode="inputs-only")
    observed_narrowed = check(narrowed, vacuity_mode="inputs-only")

    import stelling._tripwire as tw

    real = tw.evict_trace_caches
    tw.evict_trace_caches = lambda: "no-clear-caches"
    try:
        not_observed = check(clean, vacuity_mode="inputs-only")
    finally:
        tw.evict_trace_caches = real

    assert observed_clean.status == "VERIFIED"
    assert observed_narrowed.status == "UNKNOWN"
    assert not_observed.status == "UNKNOWN"

    narrowed_note = observed_narrowed.notes[0]
    unobserved_note = not_observed.notes[0]
    assert "trace unfaithful" in narrowed_note
    assert "narrowing(s) detected" in narrowed_note

    assert "NOT FULLY OBSERVED" in unobserved_note
    assert "narrowing(s) detected" not in unobserved_note, (
        "the third state is wearing the narrowed state's sentence: it reports "
        "a narrowing that was never observed"
    )
    assert "no-clear-caches" in unobserved_note, (
        "the third state does not say WHICH way the observation was lost, so "
        "a reader cannot tell a broken instrument from a disarmed one"
    )


def test_the_gate_fails_CLOSED_when_the_observation_cannot_be_made_complete():
    """A gate that refuses more is safe; one that refuses less is not.

    Every route by which the eviction can fail must land in the third state,
    including the ones that come back as an unrecognised code, because the
    question the gate answers is "was the watch complete", and any answer
    other than yes is no.
    """
    def clean():
        x = any_array((2,), jnp.float32, (0.0, 1.0))
        assert_(x + 2.0 > 1.5)

    assert check(clean, vacuity_mode="inputs-only").status == "VERIFIED"

    import stelling._tripwire as tw

    real = tw.evict_trace_caches
    for code in ("no-module", "no-clear-caches", "unexpected:RuntimeError",
                 "a code nobody has invented yet"):
        tw.evict_trace_caches = lambda code=code: code
        try:
            v = check(clean, vacuity_mode="inputs-only")
        finally:
            tw.evict_trace_caches = real
        assert v.status == "UNKNOWN", f"{code!r} certified anyway"
        assert "NOT FULLY OBSERVED" in v.notes[0], code


def test_a_clean_trace_that_STOPPED_being_watched_is_not_called_a_narrowing():
    """The mis-worded third state that was already in the code (B14).

    Disarming mid-trace made the gate set ``narrowings = max(narrowings, 1)``
    and print "1 integer narrowing(s) detected" — about a trace in which
    nothing whatever was observed to narrow. The refusal was right and the
    sentence was wrong, which is the failure this batch's third state exists
    to stop.
    """
    def clean_but_goes_dark():
        x = any_array((2,), jnp.float32, (0.0, 1.0))
        _tripwire.disarm()
        assert_(x + 2.0 > 1.5)

    v = check(clean_but_goes_dark, vacuity_mode="inputs-only")
    _tripwire.arm()

    assert v.status == "UNKNOWN"
    assert "NOT FULLY OBSERVED" in v.notes[0]
    assert "narrowing(s) detected" not in v.notes[0]
    assert "stopped watching" in v.notes[0]


def test_a_narrowing_seen_while_the_watch_was_partial_says_so_as_a_FLOOR():
    """Both states at once: the count is real, and it is a lower bound.

    A reader who fixes the one narrowing this names must not read the next
    clean run as proof there was only one, so the count is published as a
    floor whenever part of the trace went unwatched.
    """
    def narrows_then_goes_dark():
        x = any_array((2,), jnp.int16, (0, 100))
        y = x + 40000
        _tripwire.disarm()
        assert_(y < 200)

    v = check(narrows_then_goes_dark, vacuity_mode="inputs-only")
    _tripwire.arm()

    assert v.status == "UNKNOWN"
    assert "trace unfaithful" in v.notes[0]
    assert "LOWER BOUND" in v.notes[0]
    assert "stopped watching" in v.notes[0]


def test_the_eviction_primitive_reports_rather_than_raises():
    """``evict_trace_caches`` is a guardrail, so it returns codes.

    Non-vacuity matters more than usual here: a function that returned
    ``"evicted"`` unconditionally would make every test above pass while
    evicting nothing, so the failure directions are driven too.
    """
    from stelling._tripwire import _adapter_jax as adapter

    assert _tripwire.evict_trace_caches() == "evicted"

    real = jax.clear_caches
    try:
        def boom():
            raise RuntimeError("no")

        jax.clear_caches = boom
        assert adapter.evict_trace_caches() == "unexpected:RuntimeError"
        del jax.clear_caches
        assert adapter.evict_trace_caches() == "no-clear-caches"
    finally:
        jax.clear_caches = real
    assert _tripwire.evict_trace_caches() == "evicted"


def test_the_gate_still_certifies_a_clean_program_after_all_of_that():
    """The cost side of a fail-closed change, held down.

    A gate that refuses everything is trivially sound and useless. Measured
    over this repository's suite and `corpus/` at a759809: 1475 armed gated
    traces, 88 of them not fully observed, 0 of those actually narrowed, and
    312 VERIFIEDs before and 312 after — the eviction adds observation, not
    refusals.
    """
    def clean():
        x = any_array((3,), jnp.float32, (0.0, 1.0))
        assert_(x * 2.0 < 3.0)

    assert [check(clean, vacuity_mode="inputs-only").status for _ in range(3)] == [
        "VERIFIED", "VERIFIED", "VERIFIED",
    ]


def test_the_eviction_does_not_reach_a_memo_that_is_not_jaxs():
    """The eviction's scope, driven rather than described.

    ``jax.clear_caches()`` empties JAX's caches and nothing else, so a
    constant narrowed into a memo jax does not own is replayed straight past
    it. Three constructs, each of which narrows 40000 to -25536 at SETUP time
    and then hands the gated trace the finished article:

    * ``jax.extend.core.jaxpr_as_fun(saved_jaxpr)`` — the -25536 is already
      inside the saved jaxpr;
    * a user ``functools.lru_cache`` filled once with an eagerly narrowed
      value — a memo in the caller's own process, invisible to jax;
    * ``jax.closure_convert`` — **a public jax API**, which traces at setup
      and hoists the narrowed constant into the consts it returns.

    All three: VERIFIED, zero fires, and the program jax actually executes
    returns ``[-25536, -25436]`` where the source says ``[40000, 40100]``.
    This is a DISCLOSURE test, not a regression: it holds the three named in
    ``report.UNCOVERED`` to being really unwatched, so that closing one goes
    red here and gets the disclosure edited rather than left standing.
    """
    import functools

    from jax import lax

    over, dt, bound = 40000, jnp.int16, 200

    def by_jaxpr_as_fun():
        saved = jax.make_jaxpr(lambda z: z + over)(jnp.zeros((2,), dt))
        f = jax.extend.core.jaxpr_as_fun(saved)
        return lambda x: f(x)[0]

    def by_user_memo():
        @functools.lru_cache(maxsize=None)
        def k():
            # `z + over` under jit is a WATCHED route; the memo means it is
            # traced exactly once, ever, and the eviction never gets it back.
            return jax.jit(lambda z: z + over)(jnp.zeros((2,), dt))

        k()
        return lambda x: x + k()

    def by_closure_convert():
        c = lax.full((2,), over, dt)
        conv, consts = jax.closure_convert(lambda z: z + c, jnp.zeros((2,), dt))
        return lambda x: conv(x, *consts)

    x = jnp.array([0, 100], dt)
    for name, build in (
        ("jaxpr_as_fun", by_jaxpr_as_fun),
        ("lru_cache", by_user_memo),
        ("closure_convert", by_closure_convert),
    ):
        # A REGION DECLARATION for `closure_convert`'s `lax.full((2,), 40000,
        # int16)`. This test's SUBJECT is a constant that was already
        # narrowed before the gate's trace -- the narrowing has to happen for
        # there to be anything to measure, and the eager detector (which is
        # what a session run with `--stelling-eager-truncation=error` arms)
        # would otherwise refuse it. That refusal is correct and is exactly
        # the point: the two instruments cover different halves.
        from stelling._tripwire.eager import expected_truncation

        with expected_truncation(
            "this test's subject is a constant narrowed BEFORE the trace the "
            "gate watches; the narrowing is the setup"
        ):
            body = build()

        def harness():
            a = any_array((2,), dt, (0, 100))
            assert_(body(a) < bound)

        before = _tripwire.fires_count()
        verdict = check(harness, vacuity_mode="inputs-only")
        fired = _tripwire.fires_count() - before
        executed = jax.jit(body)(x)

        assert (verdict.status, fired) == ("VERIFIED", 0), (
            f"{name}: the eviction now reaches this construct "
            f"({verdict.status}, {fired} fires) — good, and "
            f"report.UNCOVERED still says it does not"
        )
        assert int(executed.max()) < over, (
            f"{name}: 40000 survived execution, so this construct does not "
            f"destroy the constant and does not belong in the disclosure"
        )


def test_the_gates_counter_is_per_thread_while_jaxs_cache_is_not():
    """Why the eviction is single-threaded-complete and no more.

    The mechanism, without racing anything: a narrowing driven on ANOTHER
    thread is counted on that thread's stack, so a gate open on this one sees
    zero. jax's trace cache, meanwhile, is process-global — one shared table
    the other thread both reads and fills. A gate that evicts, and is then
    overtaken inside its own eviction-to-trace window, therefore certifies:
    measured out-of-suite over 400 gated checks of a harness whose narrowing
    sits in a shared jitted helper, 0/400 wrong VERIFIED single-threaded
    against 247/400 with four competing threads (399/400 before the eviction
    existed). That race is not asserted here — a test of a race is a flaky
    test — but the two facts it is made of are.
    """
    import threading

    from stelling._tripwire import _pop_gate, _push_gate

    x = jnp.zeros((2,), jnp.int16)
    #: ONE jitted callable, shared across both threads: jax's trace cache is
    #: keyed on it, so it is the object through which the two threads share
    #: state at all.
    helper = jax.jit(lambda z: z + 40000)

    def narrow_on_another_thread():
        jax.make_jaxpr(lambda z: helper(z))(x)

    jax.clear_caches()
    _push_gate()
    try:
        t = threading.Thread(target=narrow_on_another_thread)
        t.start()
        t.join()
    finally:
        counted_here = _pop_gate()

    assert counted_here == 0, (
        "a narrowing on another thread was counted on this thread's gate — "
        "the per-thread counter this docstring rests on is no longer "
        "per-thread, so re-derive the thread disclosure in report.UNCOVERED"
    )

    # ...and the trace cache the other thread just filled is THIS thread's
    # too, which is the other half: shared state, unshared counter. A fresh
    # closure here, exactly as the gate uses, and the body is still not
    # re-traced.
    _push_gate()
    try:
        jax.make_jaxpr(lambda z: helper(z))(x)
    finally:
        after_warm = _pop_gate()
    assert after_warm == 0, (
        "jax's trace cache is no longer process-global (a body another "
        "thread traced was re-traced here), which would make the "
        "eviction-to-trace window safe — re-derive the thread disclosure"
    )


# ---------------------------------------------------------------------------
# B15's audit finding, driven: the FOURTH way the watch goes partial.
# ---------------------------------------------------------------------------


def test_a_DISPLACED_hook_is_not_read_as_a_clean_trace():
    """A rebind over stelling's wrapper used to produce VERIFIED on a WATCHED
    route, and nothing on this page could see it.

    THE THREE TESTS THE GATE ALREADY HAD ALL PASS in this state, which is why
    it survived: the recorder's identity is unchanged, ``fires_count()`` is
    unchanged, and the eviction succeeds. What is not unchanged is that
    stelling's wrapper is no longer the live registry entry, so it is never
    called — and a fire counter that is never incremented reads exactly like a
    trace with nothing to report.

    Measured at ``8ed5ce5`` with ``x + 40000`` on ``int16``, a route
    ``GATE_COVERAGE`` calls ``watched``: **VERIFIED**. Here: ``UNKNOWN`` with
    ``trace NOT FULLY OBSERVED``, naming the hook.

    ``_adapter_jax.detach`` rather than a rebind written here, because rule 2
    bans naming the private jax module in ``tests/`` and a test that reached
    into the registry would have to name what only that file may name. It is
    the same seam the fail-closed battery uses.
    """
    from stelling._tripwire import _adapter_jax as adapter

    over, dt, bound = 40000, jnp.int16, 200

    def harness():
        x = any_array((2,), dt, (0, 100))
        assert_(x + over < bound)

    # the control FIRST: with the hook live, this route is watched and the
    # gate refuses. Without it, "UNKNOWN" below would be no evidence at all.
    control = check(harness, vacuity_mode="inputs-only")
    assert control.status == "UNKNOWN"
    assert "trace unfaithful" in control.notes[0], control.notes

    recorder_before = adapter._installed.get("recorder")
    assert adapter.detach("bypass") == "detached"
    try:
        # everything the gate USED to look at still says the watch was whole
        assert adapter._installed.get("recorder") is recorder_before
        assert _tripwire.fires_count() is not None
        assert _tripwire.live_check() == "foreign-patch"
        assert _tripwire.displaced() == ("const-fold",)

        verdict = check(harness, vacuity_mode="inputs-only")
    finally:
        adapter.reattach()

    assert verdict.status == "UNKNOWN", (
        "a displaced hook produced a verdict about a program nobody watched "
        f"({verdict.status}). This is B15's audit finding and it was a "
        "VERIFIED on a watched route."
    )
    assert "NOT FULLY OBSERVED" in verdict.notes[0], verdict.notes
    assert "DISPLACED" in verdict.notes[0] and "const-fold" in verdict.notes[0], (
        "the refusal does not say WHICH hook was displaced, so a reader has "
        "no way to tell a rebound const-fold rule from a rebound "
        "construction site"
    )
    assert "NOT A REPORT THAT A CONSTANT WAS NARROWED" in verdict.notes[0], (
        "the third state's own sentence is gone, so this reads as a finding "
        "about a narrowing nobody observed"
    )

    # ...and the gate goes back to refusing for the RIGHT reason afterwards,
    # which is what says the detach was undone rather than merely survived
    after = check(harness, vacuity_mode="inputs-only")
    assert after.status == "UNKNOWN" and "trace unfaithful" in after.notes[0]

    # AND THE HALF THAT IS STILL OPEN IS DISCLOSED. The check asks "is our
    # wrapper live now", not "was it live throughout", so a rebind installed
    # and removed inside one call is invisible to it -- the same shape as the
    # thread race two bullets down. A fix that closed one case and left the
    # reader believing the class was closed would be the over-claim this
    # whole page exists to avoid.
    from stelling._tripwire import report

    text = " ".join(report.UNCOVERED)
    assert "HOOK IS DISPLACED" in text, (
        "the gate now refuses on a displaced hook and `report.UNCOVERED` "
        "does not tell the reader what that covers"
    )
    assert "INSTALLED AND REMOVED inside that window is invisible" in text, (
        "the residue of the displacement check is undisclosed, which makes "
        "the disclosure read as a closed class"
    )


def test_the_displacement_check_reports_only_the_hooks_this_process_armed():
    """It answers about the hooks THIS PROCESS ARMED and no others.

    Two halves. The armed one is live, so ``displaced()`` is empty — without
    that the test above could be satisfied by an instrument that reports
    displacement unconditionally. And the eager detector, which this file
    never arms, has no entry at all: a user who never switched it on can
    never be told it was displaced, and the gate can never refuse a verdict
    because of an instrument they did not ask for.
    """
    from stelling._tripwire import _adapter_jax as adapter
    from stelling._tripwire import eager as _eager

    # A SESSION RUN WITH `--stelling-eager-truncation=error` ARMS THE OTHER
    # HOOK, and then it legitimately has an entry. The claim here is about
    # hooks this process did NOT arm, so the eager one is taken out for the
    # duration and put back.
    eager_was_armed = _eager.is_armed()
    if eager_was_armed:
        _tripwire.disarm_eager()
    try:
        assert _tripwire.displaced() == ()
        # THE CLAIM IS ABOUT AN ENTRY'S PRESENCE, so it is read as one rather
        # than as an exact dict. A session run with a THIRD instrument on --
        # `--stelling-narrowing-perimeter=error` -- arms hooks this file
        # neither armed nor disarmed, and they legitimately have entries; an
        # equality here would have made this test a statement about which
        # flags the session was run with. Both halves of the claim survive
        # intact: the hook this file armed is live, and the hook it never
        # armed has no entry.
        states = dict(adapter.displacement_check())
        assert states["const-fold"] == "armed"
        assert "eager" not in states
    finally:
        if eager_was_armed:
            _tripwire.arm_eager()
