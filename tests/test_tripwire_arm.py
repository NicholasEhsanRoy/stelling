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

import os
import pathlib
import sys

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

np = pytest.importorskip("numpy")

from stelling import _tripwire  # noqa: E402
from stelling._tripwire.eager import expected_truncation  # noqa: E402
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
    hash must still be visible — which is what makes the canary diagnosable.

    TWO FAILURE MODES THAT USED TO BE ONE. Against a single ``_KNOWN_HASH``
    constant, "jax 0.11.1 shipped and nobody has read its rule" and "0.11.0
    is reporting a rule that is not 0.11.0's" produced the same red line and
    the same remedy-shaped-like-a-typo: paste the observed hash in. They are
    different findings with different remedies, so they are different
    assertions, keyed on the exact release — 0.11.0 and 0.11.1 are one series
    carrying two different rule sources, which is why series is the wrong key
    here even though it is the right key for ``TESTED_JAX_SERIES``.
    """
    status, _ = armed
    assert status.rule_hash and len(status.rule_hash) == 12
    assert status.known_hash == adapter._KNOWN_HASHES.get(status.jax_version)

    if not adapter.is_release(status.jax_version):
        # A NIGHTLY OR AN RC, which is what the `nightly` job of
        # `.github/workflows/nightly-jax-canary.yml` runs this file
        # against — it installs jax from the nightly index. It cannot
        # have a row — the version names a tree that will never be published
        # under that name again — so demanding one would redden that lane
        # every night for a fact nobody can act on, and an alarm that is red
        # every night is not read. The third state is asserted instead, so
        # this stays a measurement rather than an exemption.
        assert status.hash_state == "never-read", (
            f"jax {status.jax_version} is not a release, so no row can name "
            f"it, yet the map reports {status.hash_state!r}."
        )
        return

    expected = adapter._KNOWN_HASHES.get(status.jax_version)
    assert expected is not None, (
        f"jax {status.jax_version} has never been READ on this release. Read "
        "it, diff it against the nearest entry, and add a row naming what "
        "changed — do not copy the observed hash in. The tool ARMED anyway "
        "and nothing is gated on this; what is missing is the human step "
        f"that makes {status.rule_hash} mean something."
    )
    assert status.rule_hash == expected, (
        f"jax {status.jax_version} is recorded as carrying rule {expected} "
        f"and is reporting {status.rule_hash}: the same release is reporting "
        "a different rule. A released wheel does not change, so this is not "
        "upstream moving under us — either the row is wrong for this release "
        "or this environment is not running the jax it reports."
    )


def test_every_key_of_the_hash_map_is_a_release():
    """A row for a nightly would be a row nothing can ever match again.

    `_KNOWN_HASHES` is keyed on the exact release, and `is_release` is the
    definition of what a key IS — so a key that is not one silently turns
    the map into a set with extra steps: it can never be looked up (the
    version string it names is a build, not a release), and it makes the
    map's own "missing means never read" reading unreliable for the reader
    who finds it there.
    """
    strays = [k for k in adapter._KNOWN_HASHES if not adapter.is_release(k)]
    assert not strays, (
        f"_KNOWN_HASHES is keyed on releases and these are not: {strays}. A "
        "dev build or an rc names a tree that is never published under that "
        "name again, so a row for one can never be matched — read the rule "
        "on the RELEASE and key it there."
    )
    # and the definition is not vacuous here: the map is non-empty and its
    # keys really do go through the predicate the test names
    assert adapter._KNOWN_HASHES and not adapter.is_release("0.11.2.dev20260817")


#: Version strings :func:`is_release` MUST accept, each with the reason it is
#: a final release. The first entry is the one this table exists for.
_MUST_BE_RELEASES = {
    # THE MEASURED REGRESSION. jax really shipped this: 0.9.0.1, uploaded
    # 2026-02-05, wheel + sdist, not yanked, read off PyPI's JSON API. A
    # bare-`X.Y.Z` predicate called it a non-release, which silenced the row
    # check for a published wheel.
    "0.9.0.1": "a four-component release segment; jax shipped exactly this",
    "0.11.0": "the plain three-component case, still a release",
    "0.0": "two components; jax's first two uploads are 0.0 and 0.1",
    "0.12": "two components, the shape a future jax could ship",
    "0.11.1.post1": "a post-release is a final release",
    "0.11.1-1": "PEP 440's implicit post-release spelling",
    "1!0.12.0": "an epoch does not make a version mutable — see _adapter_jax",
}

#: Version strings :func:`is_release` MUST reject, each with the reason no row
#: can ever name it. Rejecting these is what keeps the nightly lane green for
#: a fact nobody can act on; accepting one would redden it every night.
_MUST_NOT_BE_RELEASES = {
    "0.11.2.dev20260817": "a dev build: the same name is rebuilt tomorrow",
    "0.12.0rc1": "a release candidate is superseded by its release",
    "0.12.0a1": "an alpha, same reason",
    "0.12.0b2": "a beta, same reason",
    "0.11.1+cuda": "a local version is never published to an index",
    "0.11.1.dev0+g1234": "dev and local at once",
    "0.11.1\n": "a trailing newline: `\\Z` and not `$` is why this fails",
    "": "no version at all",
    "abc": "not a version",
}


def test_the_release_predicate_is_pinned_in_BOTH_directions():
    """`is_release` is a MEANING, and both halves of it are load-bearing.

    THE WIDENING THIS PINS. The predicate used to be `^\\d+\\.\\d+\\.\\d+\\Z` —
    a bare `X.Y.Z`. jax shipped `0.9.0.1`, so a real published wheel was not
    a release by that definition, took the never-read carve-out above, and
    the row check that is this file's whole point never ran for it. Driven
    on real jax 0.11.1 — whose const-fold rule really has moved — with the
    version reported as `0.11.1.1`: the tree as merged at `3482822` PASSED,
    and `fb646b4`, which had no shape carve-out at all, FAILED. With the
    widened predicate it fails again. So the widening is not cosmetic and
    neither direction of it may drift.

    THE NARROWING IS EQUALLY LOAD-BEARING and is why this test has a second
    table. Accepting a dev build would demand a row for a name that is
    rebuilt nightly — the `nightly` job of `nightly-jax-canary.yml` runs
    this file against exactly that — and an alarm that is red every night is
    not read.
    """
    wrongly_rejected = {
        v: why for v, why in _MUST_BE_RELEASES.items()
        if not adapter.is_release(v)
    }
    assert not wrongly_rejected, (
        "these name immutable published versions, so a row CAN name them and "
        "`_KNOWN_HASHES` must be required to have one — `is_release` says "
        f"otherwise: {wrongly_rejected}"
    )
    wrongly_accepted = {
        v: why for v, why in _MUST_NOT_BE_RELEASES.items()
        if adapter.is_release(v)
    }
    assert not wrongly_accepted, (
        "these name a tree that is mutable or never published under that "
        "name again, so a row for one could never be matched and demanding "
        f"one reddens a lane for nothing — `is_release` accepts: "
        f"{wrongly_accepted}"
    )
    # THE ANTI-DODGE CONTROL, the same idiom as the test above: a predicate
    # that returned True for everything would pass the first assertion and a
    # predicate that returned False for everything would pass the second, so
    # the tables have to be non-empty and the predicate has to disagree
    # across them. It does, by construction of the two assertions, and this
    # line is what makes "by construction" a measurement.
    assert _MUST_BE_RELEASES and _MUST_NOT_BE_RELEASES
    assert not set(_MUST_BE_RELEASES) & set(_MUST_NOT_BE_RELEASES)


def test_arming_twice_does_not_double_wrap(armed):
    """A second wrapper sees every invocation the first does.

    THE COUNT IT DOUBLES DEPENDS ON WHOSE RECORDER, and this is the row where
    there is one: ``arm(rec)`` hands the same recorder to both wrappers, so
    ``rec.fires`` would count each fire twice. The orphaned-probe case in
    ``_adapter_jax.restore``'s docstring is the other row -- two recorders,
    the older of them dead -- where the recorder counts do NOT double and the
    module-level gate counter does. Neither is evidence for the other, and
    the first account of this branch's pollution defect stated the recorder
    half of it wrongly.
    """
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


def test_undoing_a_detach_after_disarming_leaves_JAXS_rule_live(disarmed):
    """The sequence above, and what the registry holds when it is over.

    ``detach("bypass")`` -> ``disarm()`` -> ``reattach()`` is what the test
    above drives, and it used to be irreversible: ``restore()`` cleared the
    installation record and reported ``foreign-patch``, and ``reattach()``
    then put back the entry it had saved -- which was OUR WRAPPER. Nobody
    owned it, so ``is_armed()`` said no while stelling's own probe sat in
    jax's const-fold registry for the rest of the interpreter's life. Every
    later ``arm()`` read that wrapper as jax's rule: ``rule_name`` became
    ``stelling_const_fold_probe``, ``hash_state`` became ``changed`` against
    a jax whose rule had not moved, and the next wrapper wrapped the wrapper
    -- a live registry two stelling wrappers deep. THE RECORDER FIGURES WERE
    NOT AFFECTED, which the first telling of this got wrong: the orphan holds
    the dead recorder and writes into that, so the counts a report prints are
    the same on a polluted process as on a clean one. The doubled number is
    the GATE's, shared by both wrappers and printed by
    ``preconditions.check()`` as ``trace unfaithful: N integer narrowing(s)``
    -- 2 where it should be 1. ``_adapter_jax.restore`` carries the
    measurement.

    THIS IS MEASURED THROUGH THE STATUS THE TOOL PUBLISHES, not through the
    registry -- rule 2 bans naming the private module here, and the status is
    what ``report.render_status`` and the canary read anyway. A run before and
    a run after must say the same things about jax's rule, because nothing
    that happened in between was about jax's rule.

    BOTH DETACH MODES, because the defect is not about either of them. What
    ``restore()`` fixes up is a saved entry that IS the wrapper being
    retired, and ``detach`` saves whatever the registry held whichever mode
    it was asked for: under an armed tripwire that is our wrapper in both.
    Driving only ``bypass`` leaves the fix free to be narrowed to the shape
    that route produces -- the registry still holding the original -- and the
    ``entry`` route, where the registry holds NOTHING at the moment
    ``restore`` runs, then re-orphans the probe with every test still green.
    """
    before, _ = _tripwire.arm()
    assert before.armed and _tripwire.disarm() == "restored"

    for mode in ("bypass", "entry"):
        status, _ = _tripwire.arm()
        assert status.armed
        assert adapter.detach(mode) == "detached"
        assert _tripwire.disarm() == "foreign-patch"
        assert adapter.reattach() == "reattached"
        _tripwire.disarm()

        after, _ = _tripwire.arm()
        try:
            assert (after.rule_name, after.rule_hash, after.hash_state) == (
                before.rule_name, before.rule_hash, before.hash_state
            ), (
                f"undoing a detach({mode!r}) left something other than jax's "
                f"rule in the registry: the tool now reports "
                f"{after.rule_name!r} / {after.rule_hash} / "
                f"{after.hash_state!r} where it reported {before.rule_name!r} "
                f"/ {before.rule_hash} / {before.hash_state!r} before any of "
                "this ran"
            )
            assert after.rule_name == adapter._KNOWN_RULE
        finally:
            _tripwire.disarm()


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

    THE POSITIVE CONTROL USED TO BE A DIFFERENT TEST. This docstring said it
    was *"the run above it -- the same ``arm()`` on the same registry returns
    ``armed``"*, which is a control this test cannot see and a run in another
    order does not have. Both readings here are ABSENCES -- ``not-invoked``
    and ``invocations == 0`` -- and an absence is exactly what a process in
    which the probe cannot trace AT ALL also produces. Measured, with a plant
    that makes ``selfcheck``'s ``make_jaxpr`` a no-op so that no trace in the
    process reaches the const-fold site: every other test in this file that
    drives the self-check went red and this one stayed GREEN, still claiming
    to have measured a detachment. So the control is now in the test, on the
    same recorder and the same registry, one line down.

    ``selfcheck() == "armed"`` is the strongest positive signal available
    here, because it is returned only when the probe reached the site in BOTH
    directions with the expected value, dtype, file and line. That it leaves
    ``rec.invocations`` at 0 is not an oversight: ``selfcheck`` saves and
    restores the recorder's counters so that a probe never lands in a user's
    denominator, so the verdict is the reading and the counter cannot be.
    """
    rec = record.Recorder()
    assert adapter.install(rec) == "installed"
    assert adapter.detach("bypass") == "detached"
    assert adapter.selfcheck() == "not-invoked"
    assert rec.invocations == 0

    assert adapter.reattach() == "reattached"
    assert adapter.selfcheck() == "armed", (
        "the live positive control did not fire: with the bypass undone, the "
        "same probe on the same registry and the same recorder still did not "
        "reach the const-fold site, so `not-invoked` above is a fact about "
        "this process and not about the detachment this test is named for"
    )


def test_ARM_ITSELF_refuses_on_a_blind_hook_and_does_not_leave_it_installed(disarmed):
    """The test above drives ``selfcheck`` directly, and nothing drove the
    GATE.

    Deleting the ``selfcheck()`` call from ``arm()`` — so that arming reports
    ``armed`` for an attached-but-blind hook — survived the entire suite: the
    probe was tested and the decision that consumes it was not. The whole
    fail-closed contract is that decision.

    ``restore()`` is the other half: a refused arm must not leave a wrapper in
    the registry, or the tool is disabled AND still in the way.
    """
    rec = record.Recorder()
    assert adapter.install(rec) == "installed"
    assert adapter.detach("bypass") == "detached"

    status, _ = _tripwire.arm()
    assert not status.armed, (
        "arm() reported armed for a hook that is attached and never invoked, "
        "which is what a jax version bump actually produces"
    )
    assert status.code == "not-invoked"
    assert "Static checking is unaffected" in status.explanation

    adapter.reattach()
    assert _tripwire.disarm() in ("not-armed", "restored"), (
        "a refused arm left its wrapper installed"
    )

    # the positive control: the same arm() on the same registry does arm
    again, _ = _tripwire.arm()
    assert again.armed, "the control did not arm, so the refusal above is not news"


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
    # THE TWO INSTRUMENTS SHARE THIS ARITHMETIC, deliberately and by
    # construction: `record.in_range` is the single implementation the
    # const-fold tripwire and the eager detector both range-check against, so
    # a value the one calls out of range is a value the other calls out of
    # range. That is the property this repository wants -- one arithmetic
    # cannot disagree with itself -- and the visible cost of it is right here:
    # breaking it to drive ONE instrument's self-check makes the OTHER refuse
    # every construction it sees. So this region is not permitting a
    # truncation, it is permitting an alarm this test manufactured, and it
    # says so where a reader of the eager report will meet it.
    with expected_truncation(
        "record.in_range is deliberately broken here to drive the const-fold "
        "self-check; the eager detector reads the same arithmetic, so every "
        "in-range construction inside this block reads as out of range"
    ):
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
    """The refusal, and WITHOUT PROBING measured rather than inferred.

    ``not is_armed()`` is an absence, and it is also what a process that
    cannot probe at all produces: under the plant described in
    ``test_attached_but_never_invoked_is_caught_which_a_presence_check_misses``
    -- every trace in the process turned into a no-op -- this test stayed
    green. It was a weak reading of its own name besides, since a probe that
    RAN and failed leaves ``is_armed()`` false too. So the second half of the
    name is now read off ``_adapter_jax._probe_seq``, the counter
    ``selfcheck`` bumps once per run to get itself a fresh shape: it may not
    move across a refused ``arm()``, and the control right after it is an
    ``arm()`` that is not refused, where it must.
    """
    probes = adapter._probe_seq
    monkeypatch.setattr(adapter, "jax_version", lambda: "0.4.7")
    code, disclosure = adapter.version_check()
    assert code == "below-floor"
    assert "0.4.8" in disclosure
    status, _ = _tripwire.arm()
    assert status.code == "below-floor"
    assert not _tripwire.is_armed(), "it probed a version it refused"
    assert adapter._probe_seq == probes, (
        f"arm() ran the self-check {adapter._probe_seq - probes} time(s) on a "
        "version it refused below the floor. The refusal is supposed to "
        "happen before any probing, and probing a jax this package does not "
        "support is the thing the floor exists to avoid."
    )
    # THE LIVE POSITIVE CONTROL. `monkeypatch.undo()` puts the real
    # `jax_version` back, so this is the same `arm()` on the same process with
    # the one forced input removed: it must probe, or the zero above says
    # nothing about the floor and everything about the environment.
    monkeypatch.undo()
    status, _ = _tripwire.arm()
    assert status.armed and adapter._probe_seq > probes, (
        "the live positive control did not fire: with the version restored, "
        f"arm() reported {status.code!r} and the probe counter moved "
        f"{adapter._probe_seq - probes} time(s)"
    )


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


@pytest.mark.parametrize(
    ("written", "dtype", "became"),
    [
        (-200, "int8", 56),
        (-129, "int8", 127),
        (-1, "uint8", 255),
        (-40000, "int16", 25536),
    ],
)
def test_a_NEGATIVE_out_of_range_constant_fires_too(armed, written, dtype, became):
    """The range check has two sides and only one was driven.

    ``in_range(...) or written < 0`` — every negative out-of-range constant
    silently accepted, ``int8 + (-200)`` wrapping to 56 and vanishing — survived
    the ENTIRE suite. Every wrap this file traced was positive, so half the
    predicate was never exercised on a real trace.

    Driven through the live hook rather than through :func:`record.in_range`,
    because the mutant is in the wrapper's use of it.
    """
    _, rec = armed
    x = jnp.zeros((7 + abs(written) % 5,), getattr(jnp, dtype))

    before = rec.count
    jax.make_jaxpr(lambda a: a + written)(x)
    assert rec.count == before + 1, (
        f"{written} -> {dtype} did not produce a finding; a negative "
        "out-of-range constant is being accepted silently"
    )
    finding = next(f for f in rec.findings.values() if f.written == written)
    assert finding.became == became and finding.agrees

    # ...and the value really is destroyed, which is what makes it a finding
    assert int(np.asarray(jax.jit(lambda a: a + written)(x)).ravel()[0]) == became


def test_the_LITERAL_VISIBLE_note_is_decided_at_the_site_not_asserted(armed):
    """``literal_visible = False`` — every finding carrying the "not textually
    on that line" caveat — survived the entire suite.

    Every test that exercised the note built a :class:`record.Finding` by hand
    with the flag already set, so the wrapper's own decision was never
    measured. Here the flag comes off a real trace, both ways, and the
    difference between the two lines is the only difference between them.
    """
    _, rec = armed
    x = jnp.zeros((9,), jnp.int8)

    def on_the_line(a):
        return a + 300

    limit = 301

    def behind_a_name(a):
        return a + limit

    jax.make_jaxpr(on_the_line)(x)
    jax.make_jaxpr(behind_a_name)(jnp.zeros((10,), jnp.int8))

    visible = next(f for f in rec.findings.values() if f.written == 300)
    hidden = next(f for f in rec.findings.values() if f.written == 301)
    assert visible.literal_visible, (
        "`return a + 300` has the literal on the line and the finding says it "
        "does not, so every finding carries the caveat and it means nothing"
    )
    assert not hidden.literal_visible, (
        "`return a + limit` does not have the literal on the line, so the "
        "caveat is what tells a reader to check the chain"
    )

    from stelling._tripwire import report

    assert "not textually on that line" not in "\n".join(
        report.render_finding(1, visible)
    )
    assert "not textually on that line" in "\n".join(report.render_finding(1, hidden))


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


# The doors that reach the const-fold site with a value numpy ALREADY
# narrowed, so the rule is handed something in range, does not fire -- and
# counts the visit in the printed denominator. That last part is why they had
# to be named: a large denominator is not evidence of coverage.
PRE_NARROWED_DOORS = {
    "jnp.full": lambda a: a + jnp.full(a.shape, 300, jnp.int8),
    "jnp.full_like": lambda a: a + jnp.full_like(a, 300),
    "lax.convert_element_type": lambda a: a
    + jax.lax.convert_element_type(300, jnp.int8),
    "lax.select": lambda a: jax.lax.select(
        a == 0, jnp.full(a.shape, 300, jnp.int8), a
    ),
    # measured: 4 invocations, 3 integer const-folds, all of them in range --
    # the fill value arrives already truncated to 44
    "jnp.take fill_value": lambda a: jnp.take(
        a, jnp.array([99]), mode="fill", fill_value=300
    ),
}

# The doors that never reach the site at all -- 0 invocations, not 0 fires.
UNREACHED_DOORS = {
    "jnp.pad": lambda a: jnp.pad(a, 1, constant_values=300),
    "jnp.clip LOWER bound": lambda a: jnp.clip(a, 300, None),
}


@pytest.mark.parametrize("label", sorted({**PRE_NARROWED_DOORS, **UNREACHED_DOORS}))
def test_the_silent_doors_are_NAMED_and_each_one_is_measured(armed, label):
    """Every door named UNCOVERED, driven: it wraps, and it produces no fire.

    None of these was in the doors table or in ``report.UNCOVERED``. The report
    never claimed the list was complete and always printed "never a clean bill
    of health", so that obligation was met — but a reader is invited to read
    the table as the answer to "what does it not see", and a door that is known
    and unnamed is a defect of the table.

    THE TWO CLASSES ARE DIFFERENT AND THE SECOND IS THE ONE WORTH KNOWING.
    ``UNCOVERED``'s old item 4 said "the operand was already an ARRAY"; the
    real class is "already narrowed before this site", scalars included — numpy
    truncates on the way in, the rule is handed a value that is IN RANGE, and
    the visit is COUNTED IN THE DENOMINATOR.

    Both halves, with a live control in the same process, because "0 fires" is
    also what a dead instrument produces.
    """
    door = {**PRE_NARROWED_DOORS, **UNREACHED_DOORS}[label]
    _, rec = armed
    x = jnp.zeros(5, jnp.int8)

    fires, invocations = rec.fires, rec.invocations
    # THE SUBJECT OF THIS TEST IS THE TRUNCATION, which is why the region
    # declaration and not `stelling.intentional_wrap` is the right tool here.
    # These doors are driven precisely to show that 300 becomes 44 in silence;
    # writing 44 instead would delete the demonstration and leave a test that
    # asserts a door does nothing. The region is inert unless
    # `--stelling-eager-truncation=error` is on, and when it is, this
    # truncation is COUNTED and NAMED with this reason in the session report
    # rather than disappearing.
    with expected_truncation(
        "the subject of this test IS the silent narrowing; 300 -> 44 is what "
        "is being demonstrated"
    ):
        out = jax.jit(door)(x)
    d_fires = rec.fires - fires
    d_invocations = rec.invocations - invocations

    assert d_fires == 0, (
        f"{label} fired: it is named UNCOVERED in report.UNCOVERED and in "
        "docs/overflow-tripwire.md, and both must now be corrected."
    )
    assert int(np.asarray(out).ravel()[0]) == 44, (
        f"{label} did not wrap 300 to 44, so it is not a hole at all"
    )
    if label in UNREACHED_DOORS:
        assert d_invocations == 0, (
            f"{label} is documented as never reaching the site, and it "
            f"reached it {d_invocations} time(s)"
        )
    else:
        assert d_invocations > 0, (
            f"{label} is documented as reaching the site with an already "
            "narrowed value and COUNTING it in the denominator; it did not "
            "reach the site at all, so the disclosure names the wrong cause"
        )

    # the live control, in this same process and at a fresh shape
    before = rec.fires
    jax.make_jaxpr(lambda a: a + 301)(jnp.zeros(6, jnp.int8))
    assert rec.fires == before + 1, "the live positive control did not fire"


def test_a_SCOPED_disable_jit_swallows_a_door_that_is_otherwise_COVERED(armed):
    """``with jax.disable_jit():`` and ``a + 200`` on ``int8``.

    The jaxpr is BYTE-IDENTICAL to the one that fires outside the block, the
    value still wraps to -56, and the tripwire sees nothing: the constant is
    narrowed before the site and the rule is handed -56 instead of 200. So the
    denominator counts the visit and the finding never exists.

    Process-wide ``JAX_DISABLE_JIT=1`` is a different case and IS handled --
    ``arm()`` returns ``not-invoked`` and the tool disables itself, driven in
    ``test_attached_but_never_invoked_is_caught_which_a_presence_check_misses``
    -- so only the scoped block is silently blind.
    """
    _, rec = armed

    def outside(a):
        return a + 200

    def inside(a):  # identical body, distinct code object -> distinct cache key
        return a + 200

    x = jnp.zeros(5, jnp.int8)

    before = rec.fires
    jaxpr_out = str(jax.make_jaxpr(outside)(x))
    assert rec.fires == before + 1, "the control did not fire outside the block"

    # The same argument as the pre-narrowed doors above, and this door is the
    # sharper case: the EAGER detector closes it outright (measured, 200 ->
    # -56 raised at the line that writes it), so with that detector armed this
    # block is exactly a demonstration of something now caught. It still has
    # to run, because what it demonstrates is that the CONST-FOLD tripwire
    # cannot see it.
    with expected_truncation(
        "the subject of this test IS the scoped disable_jit block swallowing "
        "the narrowing; 200 -> -56 is what is being demonstrated"
    ), jax.disable_jit():
        before = rec.fires
        jaxpr_in = str(jax.make_jaxpr(inside)(x))
        assert rec.fires == before, (
            "the scoped disable_jit block fired after all, and the UNCOVERED "
            "entry naming it must be corrected"
        )
        assert int(np.asarray(inside(x)).ravel()[0]) == -56, "it did not wrap"

    assert jaxpr_in == jaxpr_out, (
        "the two jaxprs differ, so this test is measuring two different "
        f"programs rather than one door going blind:\n{jaxpr_out}\n{jaxpr_in}"
    )


def test_every_door_this_file_measures_is_NAMED_in_the_report():
    """The tuple a reader reads, held against the doors driven above.

    A measurement that never reaches ``UNCOVERED`` is a door the reader is not
    told about, which is exactly the finding this commit closes.
    """
    from stelling._tripwire import report

    text = " ".join(report.UNCOVERED)
    for name in (
        "jnp.full(shape, N, dt)", "jnp.full_like(x, N)",
        "lax.convert_element_type(N, dt)", "lax.select(",
        "jnp.pad(", "jnp.take(", "jnp.clip(x, N, None)", "jnp.clip(x, lo, N)",
        "jnp.where(pred, N, x)", "jax.disable_jit()", "eager execution",
        "JAX_DISABLE_JIT=1",
    ):
        assert name in text, f"report.UNCOVERED does not name {name}"
    assert "already narrowed before this site" in text.lower() or (
        "ALREADY NARROWED BEFORE this site" in text
    ), "the class is still described as 'operand was already an array'"
    assert "COUNTED IN THE DENOMINATOR" in text, (
        "a pre-narrowed visit counts in the printed denominator, and that is "
        "the part that makes a large denominator not evidence of coverage"
    )


def test_eager_execution_is_uncovered_and_the_value_still_wraps(armed):
    """The other hole, with the same two halves and the same live control."""
    _, rec = armed
    x = jnp.zeros(3, jnp.int8)
    before = rec.invocations
    # DECLARED FOR THE THIRD INSTRUMENT, whose whole subject is `x + 256`.
    # This test's subject is that the const-fold site is NOT reached here, and
    # the only way to show that is to perform the narrowing; under
    # `--stelling-narrowing-perimeter=error` the dunder perimeter refuses it
    # at this line, correctly, before the value is ever computed. The region
    # permits it and prints this reason beside the site.
    with expected_truncation(
        "the wrap is this test's subject: it shows the const-fold site is "
        "NOT reached by eager `x + 256`, which requires performing it"
    ):
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

    **THE EVICTION IS THIS TEST'S PRECONDITION. 0.2.0 SHIPPED WITHOUT IT, AND
    WITH THE WRONG EXPLANATION OF WHY.** That release disclosed this test as
    order-dependent -- red under a shuffled whole-suite run at seed
    ``20260825``, green in file order, green in isolation -- and said the
    shape *"points at a warning already having been emitted by an earlier
    test"*, i.e. a primed ``__warningregistry__``. That sentence is wrong, and
    it is written out here rather than quietly deleted because a wrong
    diagnosis carried into a fix is how a fix ends up correct for a case that
    never happens: no warning is involved anywhere in this, and a fix that
    reset a warning registry would have left the test exactly as
    order-dependent as it found it.

    MEASURED 2026-08-28 at ``9b5b496`` on ``/home/nick/venvs/stelling-jax``
    (jax 0.11.0). ONE prior ``jax.random.key(0)`` ANYWHERE in the process is
    sufficient, and is the whole of it. Planted as its own test module ahead
    of this one, this test failed with ``invocations=0`` -- the recorder saw
    nothing at all, which is neither a fire attributed elsewhere nor a
    suppression it missed. ``jax.random.key`` is jitted and jax's trace cache
    is process-wide, so a warm cache replays it and the second call performs
    no conversion for the const-fold hook to observe. The same plant with
    ``jax.clear_caches()`` appended is green. Without the eviction below,
    then, what this test measured was "is this the first time in this process
    that anybody built a PRNG key" -- which coincides with its real subject in
    file order and comes apart under every other order.

    **EMPTYING THOSE CACHES IS NOT A DECLARED-STATE EVENT**, answered from
    ``tests/_state_guard.py`` rather than assumed: jax's trace and compilation
    caches are named in that module's own list of what it does NOT watch
    ("Not watchable as a fingerprint, and eviction is a real operation with
    real cost, so this stays a subprocess-isolation problem"), so no
    ``PINNED_EXEMPTIONS`` entry is owed for this line and none is added.
    ``tests/test_tripwire_eager.py`` already evicts at many sites against an
    exemption list that is empty, which is the same answer measured rather
    than read.

    ``evict_trace_caches()`` IS THE SHIPPED INSTRUMENT FOR THIS and not a
    local trick: ``stelling.preconditions.check``'s gate calls it immediately
    before the trace it watches, for this reason exactly;
    ``_adapter_jax.evict_trace_caches`` prices it and states what it does not
    cover; and
    ``test_tripwire_eager.py::test_the_origin_filter_keeps_JAX_S_OWN_constants_out_of_the_alarm``
    already evicts before driving these same ``jax.random`` entry points, with
    the comment *"B15's finding applied to a test"*. Its RETURN CODE is
    asserted rather than dropped, because a jax with no ``clear_caches``
    answers ``no-clear-caches``, and a caller that ignored that would be back
    to measuring the process's history in silence.

    NARROWER INSTRUMENTS WERE LOOKED FOR AND EACH ONE LOST. Measured the same
    day, each in a fresh interpreter with a warming ``jax.random.key(0)``
    ahead of it and the tripwire armed after it:

    * A SEED WHOSE AVAL NOTHING ELSE USES. ``np.int8(0)`` and ``np.int16(0)``
      do re-fire -- a fresh aval misses the cache -- but ``np.uint16``,
      ``np.uint32``, ``np.uint64`` and ``jnp.asarray(0, jnp.uint32)`` reach
      the site ZERO times even on a cold cache, so half that spelling space
      silently measures nothing; and the half that works rests on "no other
      test in this tree writes this spelling", which is a claim about every
      file at once and stops being true the day somebody writes one line.
    * ``with jax.default_matmul_precision("float32"):``, which re-fires
      because the trace context is part of jax's jit cache key. That is a
      check modelling jax's cache-key algebra rather than emptying the cache:
      one indirection behind the question.
    * ``jax.make_jaxpr(jax.random.key)(0)`` -- ``invocations=0``. The inner
      jaxpr is cached too, so tracing it ourselves buys nothing.
    * A FRESH INTERPRETER, which is what ``_CANARY_PROCESS_TABLE`` below does
      and is unarguably correct, at about 0.7 s a cell by that comment's own
      measurement against 2-25 ms for the eviction (jax 0.11.0 and 0.10.2,
      warm process) -- and it would have to carry the four attributed fields
      below back out through a pipe.

    WHAT THIS DOES NOT REACH. The eviction empties JAX's caches and nothing
    else, so it makes this test independent of ONE thing an earlier test can
    do to it rather than of everything: ``jax_enable_x64`` is read and skipped
    on rather than evicted, and the registry state arming depends on is
    ``tests/_state_guard.py``'s subject and not this line's. It is
    process-global, and the window between it and the call below is not
    atomic -- a competing thread could re-warm the cache inside it, the same
    qualifier ``_adapter_jax.evict_trace_caches`` carries, which is why this
    measurement means what it says under a single-threaded runner and not
    under ``-n``. And the control below says the site was REACHED, nothing
    more: that the one suppression is jax's, and the only one, is what the
    assertions after it are for.
    """
    if jax.config.read("jax_enable_x64"):
        pytest.skip("the threefry mask fires only at x64=0")
    _, rec = armed
    before = rec.invocations
    assert _tripwire.evict_trace_caches() == "evicted", (
        "jax's trace cache was not emptied, so `jax.random.key(0)` below may "
        "be replayed out of it and reach the const-fold site zero times. "
        "Every assertion under this line would then be about an event that "
        "did not happen."
    )
    jax.random.key(0)
    assert rec.invocations > before, (
        "the const-fold site was not reached at all -- the live positive "
        "control did not fire. That is what a warm trace cache produces and "
        "it is NOT what a misattributed fire produces: the assertions below "
        "would be measuring nothing rather than measuring the wrong thing."
    )
    assert rec.suppressed_jax == 1
    assert rec.count == 0, "jax's own constant was blamed on the caller"
    suppressed = rec.sorted_suppressed()[0]
    assert suppressed.written == 4294967295
    assert suppressed.became == -1
    assert "threefry" in suppressed.file
    assert suppressed.origin == record.ORIGIN_JAX


def test_the_pages_OPENING_DEMO_is_what_jax_does():
    """``docs/overflow-tripwire.md``'s *"What it finds"* block, driven.

    It is the first thing a reader of that page runs and the whole argument
    for the tool -- ``x + 256`` on ``int8`` reaching the jaxpr as ``add a
    0:i8[]``, and the jitted function returning its input unchanged. It is
    marked ``illustrative`` because its output is written INTO the block as
    comments, with an arrow under the byte that is gone, so
    ``test_doc_examples.py`` has no fence to attach to it. That is a
    legitimate reason not to run it there and it is not a reason for nobody
    to run it: measured before this existed, ``grep`` for the jaxpr text and
    for ``[100, 50, -10]`` across ``tests/`` had no hits, so the page's
    flagship demo -- and the excerpt fence six paragraphs above it, which
    quotes one equation of the same jaxpr -- were verified by nothing at all.

    The page says the output is byte-identical on 0.10.2 and 0.11.0. This
    runs on whichever is installed, so both lanes assert it.

    **AND IT RUNS THE PAGE'S PROGRAM, NOT A COPY OF IT.** The first version
    re-implemented ``add_offset`` here and compared the page's ANSWERS to the
    re-implementation, so the one direction it could not see was the page's
    CODE drifting from the answers printed under it -- which is the exact
    question this batch was commissioned to answer. Driven: changing the
    page's line to ``return x + 300`` left 325 tests green, while the real
    jaxpr becomes ``add a 44:i8[]`` and the result ``[-112, 94, 34]``. The
    block is now executed out of the page, so the program and its printed
    reading move together or not at all.
    """
    import re

    page = (
        pathlib.Path(__file__).resolve().parents[1] / "docs" / "overflow-tripwire.md"
    ).read_text(encoding="utf-8")

    # THE PAGE'S OWN BLOCK, EXECUTED. Located by the two lines that make it
    # this block and not another fence: the `def add_offset` the prose names
    # and the `jax.jit` call whose reading is quoted below it.
    blocks = [
        b for b in re.findall(r"^```python\n(.*?)^```", page, re.M | re.S)
        if "def add_offset" in b and "jax.jit(add_offset)" in b
    ]
    assert len(blocks) == 1, (
        f"the page carries {len(blocks)} opening-demo blocks; this test "
        f"executes exactly the one, located by `def add_offset` and "
        f"`jax.jit(add_offset)`"
    )
    ns: dict = {}
    exec(compile(blocks[0], "docs/overflow-tripwire.md", "exec"), ns)  # noqa: S102
    assert "jaxpr" in ns and "result" in ns, (
        "the page's demo no longer binds `jaxpr` and `result`; this test reads "
        "the values it printed rather than recomputing them"
    )

    jaxpr = ns["jaxpr"]
    result = ns["result"].tolist()

    # the full jaxpr, quoted in the block as a comment
    quoted = [
        ln.lstrip("# ").rstrip()
        for ln in page.splitlines()
        if ln.startswith("# { lambda ")
    ]
    assert len(quoted) == 1, (
        f"the page quotes {len(quoted)} jaxpr lines in its opening demo; this "
        f"test reads exactly the one"
    )
    assert quoted[0] == str(jaxpr), (
        f"the page's opening demo says the jaxpr is\n  {quoted[0]}\nand jax "
        f"{jax.__version__} produces\n  {str(jaxpr)}"
    )

    # the executed values, quoted on the next comment line
    executed = re.search(r"^# (\[[-0-9, ]+\])  — the function is", page, re.M)
    assert executed, "the page no longer quotes what the jitted function returns"
    assert executed.group(1) == str(result), (
        f"the page says the jitted function returns {executed.group(1)} and it "
        f"returns {result}"
    )

    # ...and the excerpt fence above it, which is one equation of that jaxpr
    excerpt = "add a 0:i8[]"
    assert f"\n{excerpt}\n" in page, (
        "the page's excerpt fence no longer carries the equation it excerpts"
    )
    assert excerpt in str(jaxpr), (
        f"the page excerpts {excerpt!r} from a jaxpr that now reads "
        f"{str(jaxpr)!r}"
    )


#: The eleven doors ``SOUNDNESS.md`` enumerates, as callables of
#: ``(array, operand)``. Split by SHAPE, because that split is the finding:
#: six promote an operand against an array and five construct an array from an
#: operand, and strict dtype promotion only has an opinion about the first
#: kind.
PROMOTING_DOORS = {
    "x + N": lambda x, c: x + c,
    "x >= N": lambda x, c: x >= c,
    "x.at[0].set(N)": lambda x, c: x.at[0].set(c),
    "jnp.where": lambda x, c: jnp.where(x > 0, c, x),
    "jnp.clip": lambda x, c: jnp.clip(x, c, c),
    "jnp.maximum": lambda x, c: jnp.maximum(x, c),
}
CONSTRUCTION_DOORS = {
    "jnp.array": lambda x, c: jnp.array(c, jnp.int8),
    "jnp.asarray": lambda x, c: jnp.asarray(c, jnp.int8),
    "jnp.int8": lambda x, c: jnp.int8(c),
    "jnp.full": lambda x, c: jnp.full((), c, jnp.int8),
    "jnp.full_like": lambda x, c: jnp.full_like(x, c),
}
ELEVEN_DOORS = {**CONSTRUCTION_DOORS, **PROMOTING_DOORS}


def _rejected_under_strict(operand_factory) -> set[str]:
    """Which of the eleven raise ``TypePromotionError`` under strict promotion."""
    rejected = set()
    # A REGION DECLARATION, because five of the eleven doors are construction
    # doors that narrow the operand in silence -- which is the finding this
    # helper's caller exists to measure. `stelling.intentional_wrap` is the
    # wrong tool for it: it would hand each door a value that fits, and the
    # thing being measured is what happens to one that does not. The
    # truncations this permits are counted and named with this reason in the
    # eager detector's session report.
    #
    # NOTE THE `except Exception` THREE LINES DOWN. It is why the alarm this
    # region suppresses inherits from BaseException: without that, an
    # `EagerTruncationError` raised inside a construction door would be caught
    # here, classified as "not a TypePromotionError", and dropped -- and this
    # helper would go on returning a confident answer about a door it never
    # drove. The region is the declaration; BaseException is what makes the
    # absence of one visible.
    with expected_truncation(
        "the subject of this measurement IS which doors narrow in silence, so "
        "the construction doors are driven with an operand that does not fit"
    ), jax.numpy_dtype_promotion("strict"):
        x = jnp.zeros(3, jnp.int8)
        operand = operand_factory()
        for name, door in ELEVEN_DOORS.items():
            try:
                door(x, operand)
            except Exception as exc:  # noqa: BLE001
                if "TypePromotion" in type(exc).__name__:
                    rejected.add(name)
    return rejected


@pytest.mark.filterwarnings("ignore:scatter inputs have incompatible types")
def test_strict_promotion_is_a_DTYPE_check_measured_over_the_WHOLE_door_set():
    """The report tells a user what to do instead, so what it tells them has
    to be true — and it was pinned at ONE door of eleven.

    ``SOUNDNESS.md`` says *"six of the eleven doors raise TypePromotionError
    for the NumPy-scalar spelling"*, and that is TRUE as written. It was
    replaced by three unqualified sentences, all three false, held by a test
    that ran ``x + operand`` and nothing else: a verbatim copy of that test
    passes alongside every assertion below.

    Re-measured across all eleven, and each of the three is pinned in the
    direction it failed:

    1. *"raises for every CONCRETE-dtype operand"* — false at 5 of 11. The
       construction doors take no promotion path at all and narrow in silence.
    2. *"every operand strict rejects would have kept its value"* — false at
       ``x.at[0].set(np.int64(256))``, which strict rejects and standard wraps
       to 0. Substituting that one door into the old test fails it on its
       author's own control message.
    3. *"it is the Python int that wraps"*, as an exclusive — false for a
       weakly-typed ``jax.Array``, which strict never rejects and which wraps
       at every wrapping door.

    Needs no armed tripwire: this is a claim about jax, and the report only
    repeats it. Identical on 0.11.0 and 0.10.2, x64 on and off.
    """
    import numpy as np

    assert len(ELEVEN_DOORS) == 11

    # (1) six of eleven, and WHICH six, for the NumPy-scalar spelling
    assert _rejected_under_strict(lambda: np.int64(256)) == set(PROMOTING_DOORS)
    for name, door in CONSTRUCTION_DOORS.items():
        with expected_truncation(
            "measuring that a construction door narrows 256 to 0 under strict "
            "promotion instead of raising"
        ), jax.numpy_dtype_promotion("strict"):
            got = int(np.asarray(door(jnp.zeros(3, jnp.int8), np.int64(256))).ravel()[0])
        assert got == 0, (
            f"{name} under strict promotion returned {got}. 'strict raises "
            "for every concrete-dtype operand' rests on this door being "
            "unreachable, and it is not."
        )

    # (2) it separates DTYPES and not values: the IN-RANGE control fires the
    # same way, a narrow one does not fire at all, and a rejection is no sign
    # the value would have survived
    assert _rejected_under_strict(lambda: np.int64(3)) == set(PROMOTING_DOORS)
    assert _rejected_under_strict(lambda: np.int8(3)) == set()
    with expected_truncation(
        "measuring that a door strict REJECTS would have wrapped anyway"
    ):
        wrapped = int(
            np.asarray(jnp.zeros(3, jnp.int8).at[0].set(np.int64(256))).ravel()[0]
        )
    assert wrapped == 0, (
        "x.at[0].set(np.int64(256)) kept its value, which would make 'every "
        "operand strict rejects would have kept its value' true after all"
    )

    # (3) the Python int is rejected NOWHERE -- and it is not the only one
    assert _rejected_under_strict(lambda: 256) == set()
    weak = jnp.asarray(256)
    assert weak.weak_type, "jnp.asarray(256) stopped being weakly typed"
    assert _rejected_under_strict(lambda: jnp.asarray(256)) == set()
    assert int(np.asarray(jnp.zeros(3, jnp.int8) + weak).ravel()[0]) == 0, (
        "a weakly-typed jax.Array is a second spelling strict exempts and "
        "that loses its value, so 'it is the Python int that wraps' is not "
        "the exclusive the report made of it"
    )

    # (3b) AND "NEVER RAISES" IS NOT THE SAME SENTENCE AS "STRICT NEVER
    # REJECTS IT". `docs/overflow-tripwire.md`'s row for the bare Python int
    # said the first one, flat, and the page contradicted it ten lines later
    # -- `jnp.array(256, jnp.int8)` raises `OverflowError` and the page says
    # so, twice. Three of the eleven do, on the VALUE rather than the dtype,
    # under strict AND standard alike, and that is exactly what makes the
    # weakly-typed-array row below it a DIFFERENT row rather than the same
    # sentence twice: for a weak `jax.Array`, "never raises" is true at all
    # eleven. Measured in all four cells (0.10.2/0.11.0 x64 on/off).
    loud_for_a_bare_int = None
    for mode in ("strict", "standard"):
        got = set()
        with expected_truncation(
            "the bare-int row: all eleven doors driven with a value that does "
            "not fit, to separate a dtype rejection from a value rejection"
        ), jax.numpy_dtype_promotion(mode):
            for name, door in ELEVEN_DOORS.items():
                try:
                    door(jnp.zeros(3, jnp.int8), 256)
                except OverflowError:
                    got.add(name)
                except Exception:  # noqa: BLE001 - promotion refusals are (3)
                    pass
        assert got == {"jnp.array", "jnp.asarray", "jnp.int8"}, (
            f"under {mode} promotion a bare Python int raises OverflowError at "
            f"{sorted(got)}; the page's row is written against exactly three"
        )
        loud_for_a_bare_int = got
    assert _rejected_under_strict(lambda: jnp.asarray(256)) == set()
    with expected_truncation("the weak-array control for the row below it"):
        for name, door in ELEVEN_DOORS.items():
            door(jnp.zeros(3, jnp.int8), jnp.asarray(256))  # raises nowhere

    page = (
        pathlib.Path(__file__).resolve().parents[1] / "docs" / "overflow-tripwire.md"
    ).read_text(encoding="utf-8")
    row = next(
        (ln for ln in page.splitlines()
         if ln.startswith("| a bare Python `int`, at any of the 11 |")),
        None,
    )
    assert row is not None, (
        "docs/overflow-tripwire.md no longer carries the bare-Python-int row "
        "of the strict-promotion table this measurement is for"
    )
    assert "OverflowError" in row, (
        f"the page's bare-int row reads {row!r}. It said 'never raises', and "
        f"three of the eleven raise OverflowError on the value -- which the "
        f"same page states ten lines below it. A row that is true only of "
        f"TypePromotionError has to say so, because the row under it means "
        f"'never raises' literally."
    )
    for door in sorted(loud_for_a_bare_int):
        assert f"`{door}`" in row, f"the page's bare-int row does not name {door}"

    # (3c) AND THE SURFACE A USER ACTUALLY READS. The same sentence ships in
    # `report._suggestions`, printed under every finding, and the page's
    # correction left it behind: "it rejects a Python int at none of the
    # eleven, which is the spelling in front of you" -- flat, and contradicted
    # by the bullet immediately above it in the same list, which tells the
    # reader that `jnp.array(N, dt)` raises OverflowError for a Python int.
    # Shipping the corrected claim in the docs and the wrong one in the
    # message is worse than either alone, so the SAME measurement holds both.
    from stelling._tripwire import report as _report

    advice = " ".join(_report._suggestions(record.Finding(
        file="m.py", line=1, func="f", written=256,
        from_dtype="int64", to_dtype="int8", became=0, origin=record.ORIGIN_USER,
    )))
    assert "at none of the eleven" not in advice, (
        "report._suggestions still says a Python int is rejected at none of "
        "the eleven doors. Three of them raise OverflowError on the VALUE, "
        "which the bullet above it in the same list already tells the reader."
    )
    assert "no TypePromotionError" in advice, (
        f"the strict-promotion bullet no longer says WHICH rejection a bare "
        f"Python int does not get. 'never rejects it' is the false reading "
        f"that was there. Rendered advice:\n{advice}"
    )
    # ...against the CLAUSE, not against the whole rendered advice. `door in
    # advice` passed for any door list at all: all three names already appear
    # earlier in the same bullet, as members of the FIVE construction doors
    # strict promotion is silent at. Measured: rewriting the OverflowError
    # clause to `(jnp.full, jnp.full_like, jnp.where)` left this green.
    clause = (
        f"three of the construction doors "
        f"({', '.join(sorted(loud_for_a_bare_int))}) raise OverflowError on "
        f"the VALUE instead"
    )
    assert clause in advice, (
        f"report._suggestions does not name the doors that raise "
        f"OverflowError on a bare Python int as this test just measured "
        f"them. Expected the clause:\n  {clause}\nRendered advice:\n{advice}"
    )
    assert len(loud_for_a_bare_int) == 3, sorted(loud_for_a_bare_int)

    # ...and the control for all of it: without strict, none of the eleven
    # raises, so the sets above are about the SETTING and not about the doors
    x = jnp.zeros(3, jnp.int8)
    with expected_truncation(
        "the control: all eleven doors driven with an operand that does not "
        "fit, to show the sets above are about the SETTING and not the doors"
    ):
        for name, door in ELEVEN_DOORS.items():
            door(x, np.int64(256))


def test_the_report_states_the_measured_strict_promotion_result(armed):
    """The bullet printed beside every finding, read back against the
    measurement above rather than against itself.

    It is a claim about jax that the report REPEATS, and the way it went wrong
    was that nothing tied the words to the grid.
    """
    from stelling._tripwire import report

    _, rec = armed
    rec.add(
        record.Finding(
            file=__file__, line=1, func="f", written=300, from_dtype="int32",
            to_dtype="int8", became=44, origin=record.ORIGIN_USER,
        )
    )
    bullet = next(
        line for line in report.render(_tripwire.Status(code="armed"), rec)
        if "numpy_dtype_promotion" in line
    )
    assert "SIX doors" in bullet and "FIVE construction" in bullet
    for door in ("x + N", "x >= N", "x.at[i].set(N)", "jnp.where", "jnp.clip",
                 "jnp.maximum"):
        assert door in bullet, f"the bullet names six doors and not {door}"
    for door in ("jnp.array", "jnp.asarray", "jnp.int8", "jnp.full",
                 "jnp.full_like"):
        assert door in bullet
    assert "IN-RANGE" in bullet
    # the three retracted claims must not come back
    assert "every CONCRETE-dtype operand" not in bullet
    assert "would have kept its value" not in bullet
    assert "it is the Python int that wraps" not in bullet


@pytest.mark.parametrize(
    ("written", "dtype"),
    [(300, "int8"), (256, "int8"), (-200, "int8"), (70000, "int16"), (256, "uint8")],
)
def test_the_reproducer_is_RUN_and_its_prediction_matched(armed, written, dtype):
    """§10a.4 exists so a user can confirm a finding themselves in seconds, and
    nothing ran it.

    The predicted output was wrong every single time — it said
    ``it prints 44:int8[]`` and jaxprs print ``44:i8[]``. Over the findings of
    a real session: 13 of 13 reproduced the value, 0 of 13 printed the claimed
    text. A mutant making the predicted value wrong by one survived the whole
    suite, because the prediction was compared against nothing.

    So this EXECUTES the emitted lines and matches the comment against the
    jaxpr they print, character for character.
    """
    from stelling._tripwire import report

    finding = record.Finding(
        file=__file__, line=1, func="f", written=written, from_dtype="int32",
        to_dtype=dtype, became=record.narrow(written, dtype),
        origin=record.ORIGIN_USER,
    )
    setup, call, comment = report.reproducer(finding)
    assert setup == "import jax, jax.numpy as jnp"

    namespace: dict = {}
    exec(setup, namespace)  # noqa: S102 - the point is that the emitted line runs
    printed = str(eval(call[len("print(") : -1], namespace))  # noqa: S307

    predicted = comment.split("it prints ", 1)[1]
    assert predicted in printed, (
        f"the reproducer predicts `{predicted}` and the jaxpr it prints is "
        f"`{printed.strip()}`"
    )
    # ...and the prediction is not vacuous: a value off by one is not in there
    off_by_one = predicted.replace(
        str(finding.recomputed), str(finding.recomputed + 1), 1
    )
    assert off_by_one not in printed, off_by_one


def test_hoisting_the_constant_really_does_raise(armed):
    """The other half of what the report suggests. ``jnp.array(N, dtype)``
    raises for a Python int, which is what turns a silent wrap into an
    immediate error at the definition site."""
    for ctor in (jnp.array, jnp.asarray):
        with pytest.raises(OverflowError):
            ctor(256, jnp.int8)
    # in range, so it is the VALUE being rejected and not the spelling
    assert int(jnp.array(127, jnp.int8)) == 127

# --- the canary, as a process ------------------------------------------------
#
# `.github/scripts/tripwire_canary.py` is the alarm the nightly workflow runs,
# and its EXIT STATUS is the whole of what CI reads. Everything below runs the
# real script in a FRESH INTERPRETER against the real `arm()`, the real jax and
# the real probe, and reads back the number the shell got and the reason codes
# the script printed for it.
#
# WHY A SUBPROCESS FOR ALL OF IT, and not just for the clean case.
#
#   * INDEPENDENCE. The previous version of this battery drove `main()` in
#     process and asserted `== 1`. By the time it ran, another test in this
#     file had left stelling's own wrapper installed as jax's live const-fold
#     rule, so every `arm()` in the process reported a CONTRADICTED hash row --
#     and all six of its cells were satisfied by the hash branch while the
#     control branch they named was never entered. Gating the control on
#     `--require`, which is the defect this batch closed, left the file green.
#     A fresh interpreter cannot inherit that, and the reason codes below say
#     which branch answered rather than only that something answered.
#   * THE TRACE CACHE. jax's is process-wide, so by the time an in-process test
#     runs, shape `(7,)` -- which the canary hardcodes -- may already have been
#     traced; the const-fold site is then reached zero times and a perfectly
#     live hook reports `0 finding`. `fired` only means what it says in a fresh
#     interpreter. (Found by asserting it in process first and watching a
#     healthy canary exit 1.)
#   * IT IS CHEAP. About 0.7 s a cell, measured.
#
# EACH SHIM FORCES ONE INPUT AND NOTHING ELSE, through shipped API wherever
# there is one: `detach("bypass")` for a dead hook is the failure a jax version
# bump actually produces, and a row written into `_KNOWN_HASHES` is the real
# contradiction rather than a stubbed verdict. Only the two `unrenderable`
# rows have to reach into the recorder, because "this repository's own shape
# moved" has no other spelling.
#
# AND EVERY SHIM RUNS AFTER `arm()` RETURNS. `arm()`'s own `selfcheck()` traces
# the SAME probe and reads the SAME recorder, so a shim that takes effect at
# import time breaks ARMING, the run never reaches the control, and the cell
# measures `not-armed` while claiming to measure the control. That is not a
# hypothetical -- it is what the first draft of the `int_narrowings` shim did,
# and the set-equality assertion below is what caught it.

_PIN_STELLING = """
import os
import pathlib

import stelling

_want = pathlib.Path(os.environ["CANARY_PARENT_STELLING"]).resolve()
_got = pathlib.Path(stelling.__file__).resolve()
if _want != _got:
    raise SystemExit(
        "the child imported %s and the test process is running %s"
        % (_got, _want)
    )
"""

_SHIM_DEAD_HOOK = """
import stelling._tripwire as tw
from stelling._tripwire import _adapter_jax as ad

_real = tw.arm
def _arm(*a, **k):
    s, r = _real(*a, **k)
    ad.detach("bypass")          # attached, and never invoked again
    return s, r
tw.arm = _arm
"""

_SHIM_PROBE_RAISES = """
import stelling._tripwire as tw
import stelling._tripwire._probe as p

_real_over, _live = p.over, {"yes": False}
def _gated(*a, **k):
    if _live["yes"]:
        raise RuntimeError("FORCED: the probe could not execute")
    return _real_over(*a, **k)
p.over = _gated

_real = tw.arm
def _arm(*a, **k):
    s, r = _real(*a, **k)
    _live["yes"] = True
    return s, r
tw.arm = _arm
"""

_SHIM_FINDINGS_UNREADABLE = """
import stelling._tripwire as tw
from stelling._tripwire import record

def _moved(self):
    raise AttributeError("`sorted_findings` moved off Recorder")

_real = tw.arm
def _arm(*a, **k):
    s, r = _real(*a, **k)
    record.Recorder.sorted_findings = _moved
    return s, r
tw.arm = _arm
"""

_SHIM_FINDING_SHAPE_MOVED = """
import stelling._tripwire as tw
from stelling._tripwire import record

class _Moved:
    def __getattr__(self, name):
        raise AttributeError("finding field %r moved" % name)

_real = tw.arm
def _arm(*a, **k):
    s, r = _real(*a, **k)
    record.Recorder.sorted_findings = lambda self: [_Moved()]
    return s, r
tw.arm = _arm
"""

_SHIM_DEAD_HOOK_AND_UNRENDERABLE = """
import stelling._tripwire as tw
from stelling._tripwire import _adapter_jax as ad
from stelling._tripwire import record

def _moved(self):
    raise AttributeError("`int_narrowings` moved off Recorder")

_real = tw.arm
def _arm(*a, **k):
    s, r = _real(*a, **k)
    ad.detach("bypass")
    record.Recorder.int_narrowings = property(_moved, lambda self, v: None)
    return s, r
tw.arm = _arm
"""

_SHIM_HASH_CONTRADICTED = """
import stelling._tripwire._adapter_jax as ad
import jax

ad._KNOWN_HASHES[jax.__version__] = "000000000000"
"""

_SHIM_NEVER_READ = """
import stelling._tripwire._adapter_jax as ad

ad._KNOWN_HASHES.clear()
"""

_SHIM_BELOW_FLOOR = """
import stelling._tripwire._adapter_jax as ad

ad._FLOOR = (999, 0, 0)
"""

#: ``(id, shim, argv, expected exit, expected reason codes, control state,
#: control report)``. ``summary`` says where ``$GITHUB_STEP_SUMMARY`` points.
# `--no-sweep` ON EVERY CELL BUT THE FIRST, and the first is what pays for
# the default. The canary re-derives `_adapter_jax._JAX_EAGER_CONSTANTS` by
# running ~700 jax operations under `disable_jit`, which costs about twelve
# seconds; this battery drives the script a dozen times to check WHICH reason
# produced which exit status, and twelve identical sweeps measure the same
# thing eleven extra times. The `clean` cell runs the DEFAULT command line, so
# the sweep is still driven end to end in a real process, and
# `tests/test_tripwire_record.py` asserts neither workflow leg passes the flag.
_CANARY_PROCESS_TABLE = [
    ("clean", "", ["--require"], 0, [], "fired", "rendered"),
    ("clean-no-require", "", ["--no-sweep"], 0, [], "fired", "rendered"),
    # TWO REASONS, AND BOTH ARE TRUE. `detach("bypass")` puts jax's own rule
    # back over stelling's wrapper, so the hook is attached-but-blind
    # (`control:did-not-fire`) AND stelling's wrapper is no longer the live
    # registry entry (`hooks:displaced`). B16 added the second: the canary now
    # asks `_tripwire.displaced()` once, before either disarm and while both
    # hooks are live, and a bypass is exactly what that question is for. Before
    # it, this run printed `disarm(): foreign-patch` on stdout and paged for
    # the control alone.
    ("dead-hook", _SHIM_DEAD_HOOK, ["--require", "--no-sweep"], 1,
     ["control:did-not-fire", "hooks:displaced"], "did-not-fire", "rendered"),
    # `--require` MUST NOT MATTER here. `arm()` says the hook is attached and
    # the control says nothing reached it, so the armed status is unverified
    # either way -- and gating this on `--require` is the shape of the defect
    # this batch closed.
    ("dead-hook-no-require", _SHIM_DEAD_HOOK, ["--no-sweep"], 1,
     ["control:did-not-fire", "hooks:displaced"], "did-not-fire", "rendered"),
    ("probe-raised", _SHIM_PROBE_RAISES, ["--require", "--no-sweep"], 1,
     ["control:raised"], "raised", "not-run"),
    ("probe-raised-no-require", _SHIM_PROBE_RAISES, ["--no-sweep"], 1,
     ["control:raised"], "raised", "not-run"),
    ("findings-unreadable", _SHIM_FINDINGS_UNREADABLE,
     ["--require", "--no-sweep"], 1,
     ["control:indeterminate"], "ran", "not-run"),
    ("finding-shape-moved", _SHIM_FINDING_SHAPE_MOVED,
     ["--require", "--no-sweep"], 1,
     ["control:unrenderable"], "fired", "unrenderable"),
    ("dead-hook-and-unrenderable", _SHIM_DEAD_HOOK_AND_UNRENDERABLE,
     ["--require", "--no-sweep"], 1,
     ["control:did-not-fire", "control:unrenderable", "hooks:displaced"],
     "did-not-fire", "unrenderable"),
    ("hash-contradicted", _SHIM_HASH_CONTRADICTED,
     ["--require", "--no-sweep"], 1,
     ["hash:contradicted"], "fired", "rendered"),
    ("hash-contradicted-no-require", _SHIM_HASH_CONTRADICTED,
     ["--no-sweep"], 1,
     ["hash:contradicted"], "fired", "rendered"),
    ("never-read", _SHIM_NEVER_READ, ["--require", "--no-sweep"], 0, [],
     "fired", "rendered"),
    ("not-armed", _SHIM_BELOW_FLOOR, ["--no-sweep"], 0, [], "not-run", "not-run"),
    ("not-armed-require", _SHIM_BELOW_FLOOR, ["--require", "--no-sweep"], 1,
     ["not-armed"], "not-run", "not-run"),
]


def _run_canary(tmp_path, shim, argv, summary="writable"):
    import re
    import subprocess

    import stelling

    (tmp_path / "sitecustomize.py").write_text(
        _PIN_STELLING + shim, encoding="utf-8"
    )
    root = pathlib.Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    # The child must import the SAME stelling this test process is running.
    # THAT IS ALL THIS GUARANTEES -- agreement, not that either is a
    # particular checkout; the shim asserts the agreement and fails loudly
    # rather than measuring some other tree quietly. Which tree the PARENT is
    # running is a question for whoever built the venv.
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), *sys.path])
    env["CANARY_PARENT_STELLING"] = stelling.__file__
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    page = tmp_path / ("nowhere" if summary == "unwritable" else "") / "summary.md"
    env["GITHUB_STEP_SUMMARY"] = str(page)

    result = subprocess.run(
        [sys.executable, str(root / ".github" / "scripts" / "tripwire_canary.py")]
        + argv,
        env=env, capture_output=True, text=True, timeout=300,
    )
    rows = {}
    for line in result.stdout.splitlines():
        name, sep, value = line.partition(": ")
        if sep:
            rows.setdefault(name, value)
    reasons = re.findall(r"^canary \[([a-z:-]+)\]:", result.stderr, re.M)
    return result, rows, reasons, page


@pytest.mark.parametrize(
    "shim, argv, expected, reasons, control, report",
    [row[1:] for row in _CANARY_PROCESS_TABLE],
    ids=[row[0] for row in _CANARY_PROCESS_TABLE],
)
def test_the_canary_process_exits_for_the_reason_it_measured(
    tmp_path, shim, argv, expected, reasons, control, report
):
    """The alarm, end to end, in the only place its answer means anything.

    THE ASSERTION IS THE SET OF REASON CODES, not the number alone. `main()`
    has eight ways to reach 1; an assertion that it reached 1 is satisfied by
    any of them, which is how two audits in a row found this battery vacuous.
    A set equality cannot be satisfied by the wrong branch, and an extra live
    fault makes the cell RED rather than quietly satisfying it -- which is the
    behaviour that was missing when a polluted process silently answered six
    cells with a hash contradiction nobody had asked for.
    """
    result, rows, got, page = _run_canary(tmp_path, shim, argv)
    context = (
        f"\n--- argv {argv} ---\n--- stdout ---\n{result.stdout}"
        f"\n--- stderr ---\n{result.stderr}"
    )
    assert rows.get("status") in ("armed", "below-floor"), (
        "the child did not get as far as a status row, so this cell is "
        "measuring the environment and not the canary" + context
    )
    assert (result.returncode, sorted(got)) == (expected, sorted(reasons)), context
    assert (rows.get("control state"), rows.get("control report")) == (
        control, report
    ), context
    # the page a human reads carries the same verdict as the exit status
    summary = page.read_text(encoding="utf-8")
    for code in got:
        assert f"`{code}`" in summary, f"{code} is not on the summary page"
    assert ("**exit 1**" if expected else "**exit 0**") in summary, context


def test_a_canary_that_cannot_write_its_summary_page_does_not_page(tmp_path):
    """Property 3 of the workflow: INFRASTRUCTURE MUST NOT PAGE.

    ``$GITHUB_STEP_SUMMARY`` is a convenience channel for whoever reads the
    run page. It is not the measurement, and a runner that hands the script an
    unwritable path is not a statement about the tripwire. It used to raise
    ``FileNotFoundError`` straight out of `main()`: a traceback and an exit 1,
    with no ``canary:`` sentence at all, in the script whose own `_hash_row`
    invokes "infrastructure must not page" as an argument.
    """
    result, rows, reasons, _ = _run_canary(
        tmp_path, "", ["--require"], summary="unwritable"
    )
    assert result.returncode == 0, result.stderr
    assert reasons == [], result.stderr
    assert "canary note:" in result.stderr, (
        "the summary silently went missing; a channel that vanishes without "
        "saying so is the beautiful zero this project keeps finding"
    )
    assert rows.get("control state") == "fired", result.stdout


def test_a_mistyped_flag_is_argparses_exit_and_the_list_says_so(tmp_path):
    """The exit code the exit-code list twice claimed did not exist.

    Not this script's decision -- argparse's -- and in the list anyway,
    because an exit status a reader can meet is one the list owes them. The
    list is checked against the reasons the script produces in
    ``tests/test_tripwire_record.py``; this is the one entry in it that no
    reason can produce, so it is driven here.
    """
    import subprocess

    (tmp_path / "sitecustomize.py").write_text("", encoding="utf-8")
    root = pathlib.Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), *sys.path])
    result = subprocess.run(
        [sys.executable,
         str(root / ".github" / "scripts" / "tripwire_canary.py"), "--requrie"],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert "canary [" not in result.stderr, (
        "argparse's rejection is not one of this script's reasons and must "
        "not be dressed as one"
    )
