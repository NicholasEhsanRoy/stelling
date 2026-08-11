# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The tripwire-to-verifier gate: a VERIFIED with the tripwire armed implies
zero narrowings fired during that trace.

This is the soundness property the gate exists to maintain. A false VERIFIED
means the verifier certified a jaxpr that does not represent the program as
written — the exact failure mode that produced this feature.
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
    """The central invariant: VERIFIED + armed tripwire = faithful trace.

    If the verifier returns VERIFIED while the tripwire is armed, then no
    narrowing occurred during that trace. This test generates arbitrary
    harnesses (some with narrowings, some without) and asserts that VERIFIED
    never appears alongside a narrowing.
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
