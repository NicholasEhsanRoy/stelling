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

np = pytest.importorskip("numpy")

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

    with jax.disable_jit():
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
    with jax.numpy_dtype_promotion("strict"):
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
        with jax.numpy_dtype_promotion("strict"):
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

    # ...and the control for all of it: without strict, none of the eleven
    # raises, so the sets above are about the SETTING and not about the doors
    x = jnp.zeros(3, jnp.int8)
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
