# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Arming, the self-check in both directions, and the doors the hook cannot see.

**This file names no private jax module.** Rule 2 covers ``tests/`` with no
exemption, so every route to the registry here goes through the adapter's own
API — including :func:`stelling._tripwire._adapter_jax.detach`, which exists
in shipped code precisely so that the fail-closed contract can be *driven*
rather than asserted from a test that would otherwise have to name what only
one file may name. ``design/private-jax-boundary.md`` is the rule.

The uncovered doors are MEASURED here rather than quoted from the plan. A
guard written up as closing a class it does not close is the defect this
repository has already had; ``jnp.where`` and ``jnp.clip`` are named UNCOVERED
in the report, and the tests below are what makes that label a measurement.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from stelling import _tripwire  # noqa: E402
from stelling._tripwire import _adapter_jax as adapter  # noqa: E402
from stelling._tripwire import record  # noqa: E402


@pytest.fixture
def armed():
    """An armed tripwire, always disarmed afterwards even on failure."""
    status, rec = _tripwire.arm()
    try:
        yield status, rec
    finally:
        adapter.reattach()
        _tripwire.disarm()


@pytest.fixture
def disarmed():
    """A process with nothing installed, restored afterwards."""
    _tripwire.disarm()
    try:
        yield
    finally:
        adapter.reattach()
        _tripwire.disarm()


def test_it_arms_and_says_what_it_attached_to(armed):
    status, rec = armed
    assert status.armed, status
    assert status.code == "armed"
    assert status.jax_version == jax.__version__
    assert status.rule_name == adapter._KNOWN_RULE
    assert status.registry_size == 3, (
        "the registry held a different number of rules than the two tested "
        "series do. That is the fact that tells `no-registry` from "
        f"`no-entry`; found {status.registry_size}."
    )
    assert _tripwire.is_armed()


def test_the_rule_hash_is_recorded_and_not_gated_on(armed):
    """§5. A cosmetic edit upstream must not disable the tool, and a changed
    hash must still be visible — which is what makes the canary diagnosable."""
    status, _ = armed
    assert status.rule_hash and len(status.rule_hash) == 12
    assert status.known_hash == adapter._KNOWN_HASH
    # the pin, on the series that have a lane
    assert status.rule_hash == adapter._KNOWN_HASH, (
        f"the const-fold rule's source changed: {status.rule_hash} != "
        f"{adapter._KNOWN_HASH}. Nothing is gated on this — the tool armed "
        "anyway — but it is the signal the nightly canary exists to raise."
    )


def test_arming_twice_does_not_double_wrap(armed):
    """A second wrapper would double every count in the denominator."""
    _, rec = armed
    again, rec_again = _tripwire.arm(rec)
    assert again.armed
    jax.make_jaxpr(lambda a: a + 256)(jnp.zeros((), jnp.int8))
    assert rec.fires == 1, (
        f"one traced wrap produced {rec.fires} fires: the rule is wrapped "
        "more than once."
    )


def test_disarm_restores_by_identity_and_reports_a_foreign_patch(disarmed):
    """§4. If something else patched over us, say so rather than clobber it."""
    status, _ = _tripwire.arm()
    assert status.armed
    assert adapter.detach("bypass") == "detached"
    assert not _tripwire.is_armed()
    assert _tripwire.disarm() == "foreign-patch"
    assert adapter.reattach() == "reattached"
    assert _tripwire.disarm() in ("not-armed", "restored")


def test_disarming_twice_is_quiet(disarmed):
    _tripwire.arm()
    assert _tripwire.disarm() == "restored"
    assert _tripwire.disarm() == "not-armed"


# --- the self-check, both detachment modes ---------------------------------


def test_the_anchor_removed_gives_no_entry_and_does_not_crash(disarmed):
    """§9 row 1, and §10 acceptance criterion 3.

    The registry is there and the rule the tripwire keys on is not. This is
    the shape a jax release produces, and the contract is a clean disabled
    status rather than an exception.
    """
    assert adapter.detach("entry") == "detached"
    assert adapter.locate() == "no-entry"
    status, rec = _tripwire.arm()
    assert not status.armed
    assert status.code == "no-entry"
    assert "Static checking is unaffected" in status.explanation
    assert "convert_element_type" in status.explanation


def test_attached_but_never_invoked_is_caught_which_a_presence_check_misses(disarmed):
    """The detachment mode a naive ``hasattr`` cannot see, and the one a
    version bump actually produces: the wrapper is installed, something else
    is the live entry, and jax never calls us.

    The positive control for this test is the run above it — the same
    ``arm()`` on the same registry returns ``armed`` — so a failure here is
    the detachment and not the environment.
    """
    rec = record.Recorder()
    assert adapter.install(rec) == "installed"
    assert adapter.detach("bypass") == "detached"
    assert adapter.selfcheck() == "not-invoked"
    assert rec.invocations == 0


def test_a_hook_that_records_everything_is_caught_as_cries_wolf(disarmed, monkeypatch):
    """The other direction, and the reason the probe runs both.

    A positive-only self-check passes on a hook replaced by "record
    everything". Driven by moving the semantics the tool depends on — every
    value is out of range — which is exactly the shape of the failure the code
    names: not "the hook is gone" but "the hook's meaning moved".
    """
    status, rec = _tripwire.arm()
    assert status.armed, "the positive control did not arm"
    monkeypatch.setattr(record, "in_range", lambda value, dtype: False)
    assert adapter.selfcheck() == "cries-wolf"


def test_a_hook_that_records_nothing_is_caught_as_not_invoked(disarmed, monkeypatch):
    """The mirror of the test above, through the same seam. A tripwire that
    never fires and one that always fires are both broken, and the probe has
    to fail on each for its own reason."""
    status, _ = _tripwire.arm()
    assert status.armed, "the positive control did not arm"
    monkeypatch.setattr(record, "in_range", lambda value, dtype: True)
    assert adapter.selfcheck() == "not-invoked"


def test_a_hook_that_points_at_the_wrong_line_is_caught_as_mis_attributed(
    disarmed, monkeypatch
):
    """The third direction. A finding a user cannot reproduce costs more trust
    than a missing one, so an attribution that has stopped working disables
    the tool rather than shipping wrong line numbers."""
    status, _ = _tripwire.arm()
    assert status.armed, "the positive control did not arm"
    monkeypatch.setattr(
        record, "attribute", lambda frames, root, names=None: (0, record.ORIGIN_USER)
    )
    assert adapter.selfcheck() == "mis-attributed"


def test_the_self_check_does_not_appear_in_the_user_s_denominator(disarmed):
    """A probe that inflated the denominator would make every "0 findings over
    N" figure the tool prints a little bit false."""
    status, rec = _tripwire.arm()
    assert status.armed
    assert rec.invocations == 0 and rec.count == 0 and rec.fires == 0


def test_arm_never_raises_whatever_the_adapter_does(disarmed, monkeypatch):
    """Fail-closed means fail quietly and legibly: every route out of
    :func:`arm` is a Status."""
    monkeypatch.setattr(adapter, "locate", lambda: 1 / 0)
    status, _ = _tripwire.arm()
    assert status.code == "unexpected:ZeroDivisionError"
    assert "Static checking is unaffected" in status.explanation


# --- version bounding -------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.11.0", (0, 11, 0)),
        ("0.10.2", (0, 10, 2)),
        ("0.4.36.dev20240101", (0, 4, 36)),
        ("0.12.0rc1", (0, 12, 0)),
        ("", None),
        ("unknown", None),
        ("nightly", None),
    ],
)
def test_version_parsing_survives_nightlies_and_never_crashes(text, expected):
    assert adapter._parse_version(text) == expected


def test_below_the_floor_refuses_without_probing(disarmed, monkeypatch):
    monkeypatch.setattr(adapter, "jax_version", lambda: "0.4.7")
    code, disclosure = adapter.version_check()
    assert code == "below-floor"
    assert "0.4.8" in disclosure
    status, _ = _tripwire.arm()
    assert status.code == "below-floor"
    assert not _tripwire.is_armed(), "it probed a version it refused"


def test_above_the_tested_range_it_arms_anyway_and_discloses(disarmed, monkeypatch):
    """§5. Refusing on every new jax release makes the tool useless the day
    jax ships."""
    monkeypatch.setattr(adapter, "jax_version", lambda: "0.99.0")
    code, disclosure = adapter.version_check()
    assert code == "untested"
    assert "0.99.0" in disclosure and "Armed anyway" in disclosure
    status, _ = _tripwire.arm()
    assert status.armed, status
    assert "0.99.0" in status.detail


def test_an_unparseable_version_probes_anyway(disarmed, monkeypatch):
    monkeypatch.setattr(adapter, "jax_version", lambda: "who-knows")
    code, disclosure = adapter.version_check()
    assert code == "ok"
    assert "unparseable" in disclosure
    status, _ = _tripwire.arm()
    assert status.armed, status


def test_the_floor_is_below_the_floor_stelling_itself_declares():
    """The bound is inert in every environment stelling supports, and saying
    so keeps it a refusal boundary rather than a support claim."""
    from stelling import _optional

    assert adapter._FLOOR < (0, 10, 0)
    assert min(_optional.TESTED_JAX_SERIES) == "0.10"


# --- what fires, and what does not ------------------------------------------


def test_a_user_written_wrap_is_found_with_both_halves_observed(armed):
    _, rec = armed

    def widen(x):
        return x + 256

    jax.make_jaxpr(widen)(jnp.zeros(3, jnp.int8))
    assert rec.count == 1
    finding = rec.sorted_findings()[0]
    assert (finding.written, finding.became, finding.to_dtype) == (256, 0, "int8")
    assert finding.from_dtype in ("int32", "int64")  # x64 off / on
    assert finding.func == "widen"
    assert finding.line == widen.__code__.co_firstlineno + 1
    assert finding.agrees, "the independent recomputation disagreed with the hook"


def test_an_in_range_constant_is_counted_and_not_reported(armed):
    """The denominator has to move even when nothing is found, or "0 findings"
    means nothing."""
    _, rec = armed
    jax.make_jaxpr(lambda a: a + 3)(jnp.zeros(3, jnp.int8))
    assert rec.count == 0
    assert rec.int_narrowings >= 1


def test_it_fires_once_per_trace_not_once_per_call(armed):
    _, rec = armed
    fn = jax.jit(lambda a: a + 256)
    x = jnp.zeros(3, jnp.int8)
    for _ in range(20):
        fn(x)
    assert rec.fires == 1, (
        f"{rec.fires} fires for 20 calls of one jitted function; the jit cache "
        "means one trace, and a report that scaled with call count would be "
        "noise."
    )


def test_an_array_constant_does_not_crash_the_trace(armed):
    """MEASURED: the rule is invoked with non-scalar constants and declines to
    fold them. ``int()`` on those raises, so a wrapper that assumed scalars
    would take down the first user trace that met an array."""
    import numpy as np

    _, rec = armed
    big = np.arange(4, dtype=np.int32) + 300
    jax.make_jaxpr(lambda a: a + jnp.asarray(big).astype(jnp.int8))(
        jnp.zeros(4, jnp.int8)
    )
    assert rec.internal_errors == 0, (
        "the wrapper raised inside a user trace and swallowed it. That is a "
        "bug even though it was caught: the counts become a lower bound."
    )
    assert rec.invocations >= 1


@pytest.mark.parametrize(
    "label,build",
    [
        ("where", lambda: (lambda p, a: jnp.where(p, 256, a))),
        ("clip", lambda: (lambda p, a: jnp.clip(a, None, 256))),
        ("where_jit", lambda: jax.jit(lambda p, a: jnp.where(p, 256, a))),
        ("clip_jit", lambda: jax.jit(lambda p, a: jnp.clip(a, None, 256))),
    ],
)
def test_where_and_clip_are_uncovered_and_the_value_still_wraps(armed, label, build):
    """UNCOVERED, measured, and labelled as such in the report and the docs.

    The const-fold hook provably does not fire here: the literal sits at the
    enclosing call site and the ``convert_element_type`` inside the sub-jaxpr
    operates on a VARIABLE, so no constant is folded. The narrowing still
    happens — which is why this is a hole and not a non-issue — and the
    assertion below measures BOTH halves, because "no fires" alone is what a
    dead instrument also produces.
    """
    _, rec = armed
    predicate = jnp.array([True, False, True])
    x = jnp.zeros(3, jnp.int8)
    fn = build()

    before = rec.fires
    jax.make_jaxpr(fn)(predicate, x)
    assert rec.fires == before, (
        f"{label} fired: the plan's UNCOVERED label is now wrong in the "
        "safe direction, and the report and docs must be updated to say so."
    )

    # ... and the value really does wrap, so this is a hole
    assert int(jnp.asarray(fn(predicate, x)).ravel()[0]) == 0

    # ... and the instrument is live in this same process, which is what
    # stops the assertion above from being a beautiful zero
    def control(x):
        return x + 256

    jax.make_jaxpr(control)(x)
    assert rec.fires == before + 1, "the live positive control did not fire"


def test_eager_execution_is_uncovered_and_the_value_still_wraps(armed):
    """The other hole, with the same two halves and the same live control."""
    _, rec = armed
    x = jnp.zeros(3, jnp.int8)
    before = rec.invocations
    assert int((x + 256)[0]) == 0
    assert rec.invocations == before, (
        "the const-fold site was reached outside jit. Measured, it is not: "
        "the constant arrives as a pjit argument and XLA truncates it."
    )
    jax.make_jaxpr(lambda a: a + 256)(x)
    assert rec.invocations > before, "the live positive control did not fire"


def test_jax_s_own_prng_mask_is_suppressed_and_named_not_blamed_on_the_caller(armed):
    """The one honest fire across jax's whole test suite, at ``x64=0`` only.

    Skipped where x64 is on, because the fire does not happen there and a test
    that passed by measuring nothing is the shape this file exists to avoid.
    """
    if jax.config.read("jax_enable_x64"):
        pytest.skip("the threefry mask fires only at x64=0")
    _, rec = armed
    jax.random.key(0)
    assert rec.suppressed_jax == 1
    assert rec.count == 0, "jax's own constant was blamed on the caller"
    suppressed = rec.sorted_suppressed()[0]
    assert suppressed.written == 4294967295
    assert suppressed.became == -1
    assert "threefry" in suppressed.file
    assert suppressed.origin == record.ORIGIN_JAX


def test_strict_promotion_is_orthogonal_to_this_defect_not_a_weaker_form(armed):
    """The report tells a user what to do instead, so what it tells them has
    to be true. ``PLAN-tripwire.md`` §8 says strict dtype promotion makes "6
    of 11 doors" raise; re-measured here it makes **none** of the wrapping
    spellings raise and **every** non-wrapping one, which is the opposite
    relationship and the one the report now states.

    Both directions in one test, because either half alone is satisfiable by
    a promotion setting that does nothing and by one that rejects everything.
    """
    _, rec = armed
    x = jnp.zeros(3, jnp.int8)
    with jax.numpy_dtype_promotion("strict"):
        # the spelling that WRAPS is not rejected, and still wraps
        assert int((x + 256)[0]) == 0
        # every concrete-dtype operand IS rejected -- and none of them wraps
        import numpy as np

        for operand in (np.int64(256), np.int32(256), jnp.int32(256), True):
            with pytest.raises(Exception) as caught:
                x + operand
            assert "TypePromotion" in type(caught.value).__name__, caught.value

    # the control: those same operands are fine, and lossless, without strict
    import numpy as np

    for operand in (np.int64(256), np.int32(256), jnp.int32(256)):
        assert int(np.asarray(x + operand).ravel()[0]) == 256, (
            "an operand strict promotion rejects would have LOST its value "
            "anyway, which would make strict a partial mitigation after all"
        )


def test_hoisting_the_constant_really_does_raise(armed):
    """The other half of what the report suggests. ``jnp.array(N, dtype)``
    raises for a Python int, which is what turns a silent wrap into an
    immediate error at the definition site."""
    for ctor in (jnp.array, jnp.asarray):
        with pytest.raises(OverflowError):
            ctor(256, jnp.int8)
    # in range, so it is the VALUE being rejected and not the spelling
    assert int(jnp.array(127, jnp.int8)) == 127
