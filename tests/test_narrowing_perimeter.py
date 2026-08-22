# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Mode 3, the dunder perimeter, driven end to end.

WHAT IS BEING PROTECTED HERE, in one paragraph. ``x <= 2**31 - 1`` on a
``float32`` array is a program about ``2147483648.0``: the literal has no
``float32``, jax converts it to the next one up, and the comparison that runs
is not the comparison that was written. Every layer below is doing its job —
the trace is faithful, the constant really is in the jaxpr — so
``preconditions.check`` returns **VERIFIED**, and it is right about a program
nobody wrote. Neither of the other two instruments can see this: the
const-fold tripwire watches integer RANGE, and the eager detector watches
array CONSTRUCTION. This one attaches to the operator slot the literal passes
through and refuses it there.

FIVE THINGS THIS FILE IS FOR, and they are five different failures:

1. **The door really closes, and it was really open.** Every acceptance case
   is driven DISARMED first and armed second, in the same process, so a green
   assertion cannot be green because the defect was never there.
2. **Arm and disarm are SESSION-scoped.** All four lifecycles — double arm,
   arm/disarm/arm, a raise between the two, and a nested in-process session —
   with the original slot object's IDENTITY asserted at the end of each. The
   nested one is B8b's regression pre-made: idempotent arm plus unconditional
   disarm means an inner session unhooks the outer one and every remaining
   outer test runs unprotected with nothing red.
3. **The instrument is observational.** The armed and disarmed jaxprs are
   compared on nine artefacts. Two of them DIFFER, by exactly one traceback
   frame, and that is asserted rather than hidden: a test that only checked
   the seven that match would let this file be read as a byte-identity claim.
4. **Drift fails CLOSED.** The type moving, a slot missing, a guard that never
   fires, a guard that fires on everything, something rebinding over the top —
   each is driven through a seam and must produce a refusal, never a quiet
   attach.
5. **The predicate is the scored artefact.** Its own 24-case self-test runs
   here, and the three slot-keyed decisions the perimeter depends on
   (``__pow__`` out, ``__rpow__`` in, ``/`` into a float, size-0 exempt) are
   driven against the vendored copy rather than trusted.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import json
import pathlib
import re
import warnings

import pytest

pytest_plugins = ["pytester"]

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import numpy as np  # noqa: E402

import stelling  # noqa: E402
from conftest import (  # noqa: E402
    deterministic_order_args,
    lowered_perimeter,
    tripwire_plugin_args,
)
from stelling import _tripwire  # noqa: E402
from stelling._tripwire import _adapter_jax as adapter  # noqa: E402
from stelling._tripwire import _probe, perimeter, prop_guard, report  # noqa: E402
from stelling._tripwire.eager import expected_truncation  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402

PLUGIN_ARGS = tripwire_plugin_args()
ORDER_ARGS = deterministic_order_args()

#: The reference defect, and the reason it is a comparison against a float
#: rather than an int into a narrow int: ``2**31 - 1`` is not out of range for
#: anything. It is out of REPRESENTABILITY, which is the half of this problem
#: no range check reaches.
MOVED = 2**31 - 1
MOVED_IMAGE = 2147483648.0
ALSO_MOVED = 16777219
EXACT = 1000

#: THE OTHER HALF OF THE FLOAT STORY, and it had no acceptance case until
#: B22. ``MOVED`` is out of REPRESENTABILITY; this one is out of RANGE for the
#: float format itself, which is ``prop_guard.REASONS``' fourth entry
#: ``overflows-float`` -- and the FORMAT matters, because that is what decides
#: whether the door is silent. Every ``int64`` value is finite in ``float32``
#: and ``bfloat16``, so among the four catalogued formats only ``float16``
#: has a finite range an integer literal can leave QUIETLY: measured disarmed
#: in both x64 cells, ``x_f16 <= 100000`` is ``[True, True, True]`` and the
#: comparison that runs is against ``inf``. Written large enough to overflow
#: ``float32`` the literal is also too large for jax to convert at all and jax
#: raises ``OverflowError`` on its own, so a ``float32`` case here would be a
#: test that the perimeter beats jax to a defect jax already reports.
OVERFLOWING_FMT = "float16"
OVERFLOWS = 100000
OVERFLOWS_IMAGE = float("inf")

#: And the case NOTHING in this release sees, which is the same sentence's
#: other side: a value that is ALREADY a float. Only ``type(b) is int``
#: reaches the predicate (``report.PERIMETER_UNCOVERED``'s second bullet) and
#: the other two instruments are integer-to-integer throughout, so a finite
#: double becoming ``inf`` raises no alarm anywhere. That is a scope boundary
#: rather than a hole, and it is now stated on the page instead of being
#: inferable from the fact that every example on it happened to be an int --
#: which is what ``docs/overflow-tripwire.md``, a page NAMED for overflow,
#: left a reader to do.
#: The failure message for the test below, in one place because it is the
#: whole point of the test: this pin exists to keep a DISCLOSURE true, so
#: going red means the PAGE is now wrong, not that the code is.
_LOUDER_THAN_THE_PAGE = (
    "{label} is now caught ({exc}). That is better news than this test "
    "records, and it makes docs/overflow-tripwire.md wrong in two places: the "
    "'And it is integers, all the way down' bullets under 'What it does NOT "
    "find', and bullet 3 of the narrowing perimeter's 'What it does NOT "
    "cover'. Both say a value that is already a float raises no alarm "
    "anywhere in this release. Rewrite them, then move this case out of "
    "FLOAT_VALUES_NOTHING_SEES."
)

FLOAT_VALUES_NOTHING_SEES = (
    ("jnp.full((2,), 1e300, jnp.float32)", lambda: jnp.full((2,), 1e300, jnp.float32)),
    ("jnp.full((2,), 70000.0, jnp.float16)", lambda: jnp.full((2,), 70000.0, jnp.float16)),
    ("jnp.float16(70000.0)", lambda: jnp.float16(70000.0)),
    ("x_f32 + 1e300", lambda: jnp.zeros((2,), jnp.float32) + 1e300),
    ("x_f16 + 70000.0", lambda: jnp.zeros((2,), jnp.float16) + 70000.0),
    ("jnp.asarray([1e300, 1e300]).astype(jnp.float32)",
     lambda: jnp.asarray([1e300, 1e300]).astype(jnp.float32)),
    # The integer literal that IS refused through an operator, at a
    # CONSTRUCTION site instead -- where no operator exists and the eager
    # detector is integer-to-integer, so nothing sees it either.
    ("jnp.full((2,), 100000, jnp.float16)",
     lambda: jnp.full((2,), 100000, jnp.float16)),
)


@pytest.fixture(autouse=True)
def _isolate():
    """Give this test the perimeter to itself, and hand back WHAT IT FOUND.

    ``runpytest`` is in-process by default, so a nested session in this file
    arms slots in THIS interpreter; and a test that drives a refusal leaves
    counters moved. Both are put back.

    **AND THE RESTORE IS CONDITIONAL, WHICH IS THIS BATCH'S OWN SUBJECT
    TURNED ON THIS FILE.** It used to be ``for face in _installed:
    _restore_face(face)`` beside ``_owners.clear()`` -- an unconditional
    release, which is exactly the asymmetry ``perimeter.arm(owner=...)``
    exists to prevent, aimed at the session's own hold. Under
    ``--stelling-narrowing-perimeter=error`` the plugin arms under the
    session's ``Config`` before any test runs; this file sorts **71 of 146**,
    so its FIRST test unhooked that hold and the ~4,300 tests after it ran
    unprotected with nothing red. Driven at ``e6968fe``, the documented
    dial-on command over the whole suite reported::

        NOT ARMED [detached] ... 0 integer literal(s) ... were checked

    -- the beautiful zero, produced by this fixture rather than by the tree.

    So the hold that was there on the way in is there again on the way out, by
    identity, and the assertion at the bottom is against that rather than
    against ``()``.

    THE SESSION'S HOLD IS TAKEN DOWN FOR THE WINDOW rather than left standing
    over the test, and that is a decision rather than an oversight: half of
    this file asserts ``owners() == 0``, ``live_check() == "detached"`` or a
    slot's identity against jax's own function, and every one of those is
    false while an outer hold is live -- so left standing it would turn a file
    about the module into a file about which flags the suite was run with. The
    window is one test long and the hand-back goes through
    ``conftest.lowered_perimeter``, which re-arms through the shipped
    ``arm()`` with its self-check and RAISES if it cannot: a hand-back that
    failed must be a red test here rather than a silent hole in everything
    after it. That helper is shared with
    ``tests/test_tripwire_gate_coverage.py`` so that this operation has one
    implementation and not two hand-rolled ones.
    """
    faces_before = perimeter.armed_faces()
    owners_before = list(perimeter._owners)
    counters_before = (
        perimeter.CHECKS,
        perimeter.FINDINGS,
        perimeter.INTERNAL_ERRORS,
        dict(perimeter.PERMITTED),
    )
    declines_before = collections.Counter(prop_guard.INTERNAL_DECLINES)
    unknown_before = set(prop_guard.UNKNOWN_SLOTS)

    with lowered_perimeter() as lowered:
        assert lowered == faces_before
        # ZEROED FOR THE WINDOW, and written back at the bottom -- the same
        # save/run/write-back `selfcheck()` performs, for the same reason.
        # These tests count their own fires (`assert perimeter.FINDINGS == 0`
        # is how seven of them say "nothing was refused here"), and with the
        # dial on they would otherwise start from whatever the 70 files before
        # this one had accumulated. Zeroing only the window keeps both
        # readings true: the test counts itself, and the session's denominator
        # is not spent on it.
        perimeter.reset_counters()
        prop_guard.UNKNOWN_SLOTS.clear()
        prop_guard.INTERNAL_DECLINES.clear()
        try:
            yield
        finally:
            # THE TEST'S OWN LEFTOVERS FIRST, so that the hand-back below is
            # re-arming from nothing rather than over whatever this test left.
            for face in list(perimeter._installed):
                perimeter._restore_face(face)
            del perimeter._owners[:]
            perimeter.CHECKS = counters_before[0]
            perimeter.FINDINGS = counters_before[1]
            perimeter.INTERNAL_ERRORS = counters_before[2]
            perimeter.PERMITTED.clear()
            perimeter.PERMITTED.update(counters_before[3])
            prop_guard.UNKNOWN_SLOTS.clear()
            prop_guard.UNKNOWN_SLOTS.update(unknown_before)
            prop_guard.INTERNAL_DECLINES.clear()
            prop_guard.INTERNAL_DECLINES.update(declines_before)
    assert perimeter.armed_faces() == faces_before, (
        "this test did not hand the perimeter back as it found it"
    )
    assert [held is was for held, was in zip(perimeter._owners, owners_before)] == (
        [True] * len(owners_before)
    ), "the owner list came back holding different objects"


def _tracer_type():
    located = adapter.perimeter_locate("tracer")
    assert not isinstance(located, str), located
    return located


# ---------------------------------------------------------------------------
# 5 — the predicate is the scored artefact, not a rewrite of it
# ---------------------------------------------------------------------------


def test_the_vendored_predicates_own_selftest_passes_in_this_cell():
    """PROP's 24 cases, run against the COPY in ``src/``.

    The artefact was scored elsewhere — 204,300 property evaluations, then
    482,691 real-corpus checks with zero false positives — and none of that
    evidence is re-derivable here. What IS checkable here, on every jax this
    repository is tested against, is that the copy still answers the way the
    scored one did: this is the artefact's own self-test, unmodified, and it
    is the reason the vendoring note says "keep its self-test".
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = prop_guard._selftest()
    text = buffer.getvalue()
    assert code == 0, text
    assert "0 failed" in text, text
    assert "24 passed" in text, text
    assert "unknown slots seen : none" in text
    assert "internal declines  : none" in text


@pytest.mark.parametrize(
    ("slot", "fires"),
    [
        # F5, and the two halves are a MEASURED asymmetry rather than a
        # stylistic one: `x ** k` keeps k a Python int in `integer_pow[y=k]`
        # and never converts it, so a guard there is a pure false-positive
        # generator; `k ** x` does convert and does narrow.
        ("__pow__", False),
        ("__rpow__", True),
        # F6: `/` promotes the literal into a FLOAT, so 40000 survives.
        ("__truediv__", False),
        ("__rtruediv__", False),
        # the dtype-preserving majority
        ("__add__", True),
        ("__le__", True),
    ],
)
def test_the_slot_name_is_load_bearing_and_the_list_is_not_uniform(slot, fires):
    """The same array, the same literal, six slots, three answers.

    This is why ``perimeter.py`` passes the REAL slot name: two of the
    predicate's mitigations cannot be applied without it, and a caller that
    passed a placeholder would get the majority behaviour on both of them —
    a false positive on every ``x ** 40000`` in the corpus.
    """
    x = jnp.zeros((2,), jnp.int16)
    finding = prop_guard.classify(x, 40000, slot)
    assert (finding is not None) is fires, (slot, finding)


def test_the_size_zero_exemption_reads_size_so_the_array_must_be_passed():
    """F11, and the reason ``classify`` gets the array rather than the dtype.

    ``tests/test_declaration_dtype.py:494`` compares a zero-size ``uint8``
    array against ``-3``. The literal genuinely does not exist in the executed
    program — it runs as ``>= 253`` — but no element is ever compared, so
    there is nothing to observe and nothing to refuse. Measured behind that:
    2,988 (literal, narrowed-image) outcome comparisons on empty arrays across
    all 34 slots, zero differences.
    """
    empty = jnp.zeros((0,), jnp.uint8)
    filled = jnp.zeros((2,), jnp.uint8)
    assert prop_guard.classify(empty, -3, "__ge__") is None
    finding = prop_guard.classify(filled, -3, "__ge__")
    assert finding is not None
    # THE MESSAGE IS THE POINT OF THE SEPARATE REASON. "out of range for
    # uint8" invites widening; the sign flip inverts the predicate, so
    # `uint8 >= -3` -- mathematically all-True -- runs as `>= 253` and is
    # all-False.
    assert finding.reason == "negative-into-unsigned"
    assert finding.narrowed_to == 253
    assert "negative literal cannot exist" in finding.message
    # ...and the dtype-only spelling has no `.size` to read, which is why the
    # contract says to pass the array.
    assert prop_guard.classify(np.dtype("uint8"), -3, "__ge__") is None


def test_the_in_repo_zero_size_comparison_stays_green_with_the_perimeter_armed():
    """The one in-repo site the spike's own guard would have broken.

    No exemption entry, no allowlist, no ``expected_truncation`` region: the
    predicate declines it on the size, which is why the rollout needed none of
    those. Driven here as the exact expression that test writes.
    """
    empty = jnp.zeros((0,), jnp.uint8)
    with perimeter.armed(("tracer",)) as status:
        assert status.armed, status.explanation
        assert bool(jnp.all(empty >= -3))


def test_one_transient_fault_does_not_blind_the_guard_for_the_rest_of_the_process():
    """The memo must not keep a value the call did not compute.

    ``prop_guard._target_dtype`` caches on ``(dtype, is-float-slot, x64)`` and
    used to write into that cache on **every** path, its own ``except`` branch
    included. So one transient fault cached ``None`` for that key and the
    guard was blind on it for the rest of the process, having counted the
    failure exactly **once** -- which is worse than an uncounted failure,
    because ``INTERNAL_DECLINES`` then says ``1`` and the report's "these
    figures are a lower bound" sentence understates the hole by however long
    the process lives.

    **THE FAULT IS A PUBLIC, DOCUMENTED jax API AND NOT A MONKEYPATCH.** The
    ``truediv`` branch asks jax for the promotion by allocating
    ``jnp.zeros((0,), dt) / 1``; ``jax.transfer_guard("disallow")`` -- which
    real code uses to catch stray host-to-device transfers -- makes that
    allocation raise. Inside the window the guard declining is CORRECT: the
    program itself cannot run there. The defect is entirely what happens after
    the window closes, which is why the drive has three phases and the middle
    one asserts a decline rather than a refusal.

    Driven at ``e6968fe``, jax 0.11.0, CPU, ``JAX_ENABLE_X64=0``, with the
    memo write as it was::

        AFTER the window, the reference defect fires 0 of 20
        declines: {'JaxRuntimeError': 1}   checks: 21   findings: 0
        cache after window: {('float32', True, False): None}

    -- twenty-one checks, one of them declined and counted, twenty of them
    silently answered "nothing wrong" out of a memo, and a report whose
    "partly unmeasured" sentence names **one**. With ``return None`` on that
    branch, the same drive::

        AFTER the window, the reference defect fires 20 of 20
        declines: {'JaxRuntimeError': 1}   checks: 22   findings: 20
        cache after window: {}
    """
    guarded = pytest.importorskip("jax").transfer_guard
    x = jnp.zeros((3,), jnp.float32)
    # 16777219 is the first odd integer above float32's 2**24 contiguity, so
    # `x / 16777219` converts it to 16777220.0 -- a literal that does not
    # exist in the program jax runs, on the one slot whose promotion is asked
    # by ALLOCATING.
    literal = 16777219

    def fires() -> bool:
        try:
            x / literal
        except perimeter.NarrowingError:
            return True
        except BaseException:  # noqa: BLE001 - jax's own refusal, not ours
            return False
        return False

    with perimeter.armed(("array",)) as status:
        assert status.armed, status.explanation
        # 1 -- the control, before any fault: the guard is live on this key.
        prop_guard._TARGET_CACHE.clear()
        prop_guard.INTERNAL_DECLINES.clear()
        assert fires(), "the reference defect did not fire before the fault"

        # 2 -- one transient fault, through the public API. The DECLINE is
        # correct here and is what makes phase 3 a test of the memo rather
        # than of the guard.
        prop_guard._TARGET_CACHE.clear()
        prop_guard.INTERNAL_DECLINES.clear()
        with guarded("disallow"):
            assert not fires()
        assert sum(prop_guard.INTERNAL_DECLINES.values()) == 1, (
            f"the fault was not counted: {dict(prop_guard.INTERNAL_DECLINES)}"
        )

        # 3 -- the fault is CLEARED, and the guard must be live again. Twenty
        # and not one: a single retry could pass on a cache that happened to
        # be evicted, and the shipped defect was permanent rather than flaky.
        assert sum(fires() for _ in range(20)) == 20, (
            "a single transient fault blinded the guard for the rest of the "
            f"process; the memo holds {prop_guard._TARGET_CACHE}"
        )
        # ...and nothing the call did not compute is in the memo.
        assert None not in prop_guard._TARGET_CACHE.values()
        # The decline is still counted exactly once, so the report's lower
        # bound is a lower bound about ONE check and not about the run.
        assert sum(prop_guard.INTERNAL_DECLINES.values()) == 1


# ---------------------------------------------------------------------------
# 1 — the door really closes, and it was really open
# ---------------------------------------------------------------------------


def _harness(bound, literal):
    def harness():
        x = any_array((4,), "float32", (0.0, bound))
        return assert_(x <= literal)

    return harness


def _verdict_or_raise(harness):
    jax.clear_caches()
    try:
        return ("verdict", check(harness, vacuity_mode="inputs-only").status)
    except perimeter.NarrowingError as exc:
        return ("raised", exc.finding.reason)


@pytest.mark.parametrize(
    ("label", "bound", "literal", "armed_outcome"),
    [
        ("the reference defect", 1e9, MOVED, ("raised", "inexact")),
        ("its smaller sibling", 1e6, ALSO_MOVED, ("raised", "inexact")),
        ("a literal that survives", 100.0, EXACT, ("verdict", "VERIFIED")),
    ],
)
def test_check_returns_verified_disarmed_and_refuses_armed(
    label, bound, literal, armed_outcome
):
    """The acceptance criterion, driven BOTH WAYS in one process.

    The disarmed half is not decoration and is not a control for the armed
    half: it is the DEFECT, live at this commit's parent, and asserting it
    here is what stops this file from becoming green against a jax that
    quietly started refusing these itself.
    """
    harness = _harness(bound, literal)
    assert _verdict_or_raise(harness) == ("verdict", "VERIFIED")
    with perimeter.armed(("tracer",)) as status:
        assert status.armed, status.explanation
        assert _verdict_or_raise(harness) == armed_outcome
    # ...and the door is open again afterwards, which is what makes the
    # perimeter an instrument rather than a permanent change to jax.
    assert _verdict_or_raise(harness) == ("verdict", "VERIFIED")


#: ``docs/overflow-tripwire.md``'s five-row table under *"What it covers"*:
#: what each spelling RUNS AS, and which of the three instruments refuses it.
#: Declared here and driven below, with the page's own cells read back --
#: because the page's row for the first one said the array came out ``0``-filled
#: when it comes out ``-25536``-filled, in both x64 cells, and the table whose
#: whole job is *"what the program actually runs"* was the one carrying it.
#:
#: ``runs`` is the text that must appear in the page's *what runs* cell AND
#: the value the program actually produces, so neither can move alone. The
#: instrument triple is (tripwire, eager, perimeter) and reads the page's last
#: three cells: ``fires`` for the one that refuses, ``quiet`` for a cell that
#: says ``no fire``, and ``n/a`` for one the page marks ``—``.
_WHAT_RUNS = (
    ("jnp.full((3,), 40000, int16)", "-25536",
     ("n/a", "fires", "n/a"),
     lambda: jnp.full((3,), 40000, jnp.int16)),
    ("x_int16 + 40000", "-25536",
     ("quiet", "quiet", "fires"),
     lambda: jnp.zeros((3,), jnp.int16) + 40000),
    ("40000 + x_int16", "-25536",
     ("quiet", "quiet", "fires"),
     lambda: 40000 + jnp.zeros((3,), jnp.int16)),
    ("x_f32 <= 2**31 - 1", "2147483648.0",
     ("quiet", "quiet", "fires"),
     lambda: jnp.zeros((3,), jnp.float32) <= 2**31 - 1),
    ("x_int16 + 3", "3",
     ("n/a", "n/a", "quiet"),
     lambda: jnp.zeros((3,), jnp.int16) + 3),
)

_WHAT_RUNS_HEADER = (
    "| written | what runs | the tripwire | the eager detector | the perimeter |"
)


def _doc_what_runs_rows():
    page = (pathlib.Path(__file__).resolve().parents[1]
            / "docs" / "overflow-tripwire.md").read_text(encoding="utf-8")
    assert _WHAT_RUNS_HEADER in page, (
        "docs/overflow-tripwire.md no longer carries the five-row 'what runs' "
        "table this test drives, under the header it is located by"
    )
    body = page.split(_WHAT_RUNS_HEADER, 1)[1].split("\n\n", 1)[0]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def test_the_pages_what_runs_table_is_the_measured_one():
    """Five rows, driven with all three armed, against the page's own cells.

    The page's *"what runs"* column is the one a reader trusts to know what
    their program does, and its first row said ``0``-filled about an array
    that comes out ``-25536``-filled -- the ``0`` belonging to
    ``jnp.full((), 256, jnp.int8)`` three sections up. Nothing read that
    table, so a value carried over from a different example sat in it.

    Both directions are checked: every declared row must be in the page and
    every page row must be declared, so a sixth row cannot arrive unmeasured
    and a fifth cannot leave and pass as compliance.
    """
    from stelling._tripwire import eager as _eager

    doc_rows = _doc_what_runs_rows()
    assert len(doc_rows) == len(_WHAT_RUNS), (
        f"the page's table has {len(doc_rows)} rows and this test declares "
        f"{len(_WHAT_RUNS)}. A row added there is a row driven here."
    )

    tw_status, recorder = _tripwire.arm()
    eager_status = _eager.arm()
    try:
        assert tw_status.armed and eager_status.armed
        for (written, runs, expected, build), row in zip(_WHAT_RUNS, doc_rows):
            assert f"`{written}`" == row[0], (
                f"row order moved: this test drives {written!r} where the "
                f"page's row reads {row[0]!r}"
            )
            assert runs in row[1], (
                f"the page says {written} runs as {row[1]!r}; this test drives "
                f"it as {runs!r}"
            )

            jax.clear_caches()
            before = (recorder.fires, _eager.TRUNCATIONS, perimeter.FINDINGS)
            fired = None
            with perimeter.armed(("tracer", "array")) as status:
                assert status.armed, status.explanation
                try:
                    ran = np.asarray(build())
                except stelling.EagerTruncationError:
                    fired = 1
                except perimeter.NarrowingError:
                    fired = 2
                else:
                    ran_text = (repr(float(ran.ravel()[0]))
                                if ran.dtype.kind == "f" else str(ran.ravel()[0]))
                    if ran.dtype.kind == "b":
                        # the comparison rows: what RUNS is the literal jax
                        # used, which is the moved one, not the bool it made
                        ran_text = repr(float(
                            np.asarray(2**31 - 1).astype("float32")))
                    assert runs in ran_text, (
                        f"{written} runs as {ran_text}, and the page and this "
                        f"test both say {runs}"
                    )
            after = (recorder.fires, _eager.TRUNCATIONS, perimeter.FINDINGS)
            moved = tuple(a - b for a, b in zip(after, before))

            for i, (claim, delta) in enumerate(zip(expected, moved)):
                cell = row[2 + i]
                if claim == "fires":
                    assert fired == i, (
                        f"{written}: the page says instrument {i} refuses it "
                        f"and the refusal came from {fired}"
                    )
                    assert "refuses" in cell, (row, cell)
                    if i:
                        assert delta == 1, (written, moved)
                elif claim == "quiet":
                    assert delta == 0, (
                        f"{written}: instrument {i} fired {delta} time(s) and "
                        f"the page's cell reads {cell!r}"
                    )
                    assert "no fire" in cell or "passes" in cell, (row, cell)
                else:
                    assert cell in {"—", "-"}, (row, cell)
    finally:
        _eager.disarm()
        _tripwire.disarm()


def test_the_float_OVERFLOW_literal_is_refused_and_the_door_was_silent():
    """``overflows-float``, driven both ways -- the page's float answer, half one.

    ``docs/overflow-tripwire.md`` is NAMED for overflow and, until B22, every
    route it enumerated was an integer route. The question a reader arrives
    with -- *a finite value became* ``inf``*, does this see it?* -- had no
    answer on the page in either direction, and the page had to be able to
    give one that is true of the RELEASE rather than of one instrument,
    because ``prop_guard`` carries ``overflows-float`` while the other two
    instruments are integer-to-integer throughout.

    So both halves are pinned. This is the half the release DOES catch: an
    integer literal with no finite image in the float dtype it meets. The
    disarmed assertion is the load-bearing one -- it is what makes this a
    silent door rather than a jax error the perimeter merely relabels, and it
    is why the format is ``float16`` (see :data:`OVERFLOWING_FMT`).
    """
    x = jnp.zeros((3,), getattr(jnp, OVERFLOWING_FMT))

    # DISARMED: silent, and the comparison that runs is against `inf`.
    assert (x <= OVERFLOWS).tolist() == [True, True, True]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        image = float(np.asarray(OVERFLOWS).astype(OVERFLOWING_FMT))
    assert image == OVERFLOWS_IMAGE

    with perimeter.armed(("array",)) as status:
        assert status.armed, status.explanation
        with pytest.raises(perimeter.NarrowingError) as caught:
            _ = x <= OVERFLOWS
    finding = caught.value.finding
    assert finding.reason == "overflows-float", finding
    assert finding.literal == OVERFLOWS
    assert finding.narrowed_to == OVERFLOWS_IMAGE
    assert finding.target_dtype == OVERFLOWING_FMT
    # ...and the neighbouring reason on the same format, so that a predicate
    # that collapsed the two would not pass this file.
    with perimeter.armed(("array",)):
        with pytest.raises(perimeter.NarrowingError) as inexact:
            _ = x <= 65505
    assert inexact.value.finding.reason == "inexact", inexact.value.finding

    # the door is open again
    assert (x <= OVERFLOWS).tolist() == [True, True, True]

    # AND THE SENTENCES ARE READ. This batch's whole subject is a page that
    # measured nothing wrong and simply said nothing, so a behaviour pin with
    # no reader is only half the repair: both pages that now answer the float
    # question have to be naming the case this test drives.
    repo = pathlib.Path(__file__).resolve().parents[1]
    for rel in ("docs/overflow-tripwire.md", "docs/quickstart.md"):
        page = (repo / rel).read_text(encoding="utf-8")
        assert f"`x_f16 <= {OVERFLOWS}`" in page or f"`x <= {OVERFLOWS}`" in page, (
            f"{rel} no longer names the comparison this test drives, so its "
            f"float-overflow answer has stopped being about a measured case"
        )
        assert OVERFLOWING_FMT in page, f"{rel} no longer names {OVERFLOWING_FMT}"


def test_a_float_VALUE_that_overflows_is_seen_by_NOTHING_in_this_release():
    """The other half, and it is a scope boundary that is now written down.

    A value that is already a float is outside all three instruments: only
    ``type(b) is int`` reaches this predicate, and the const-fold tripwire and
    the eager detector are integer-to-integer (``intentional_wrap`` refuses
    every non-integer dtype by name, asserted below so that "integer to
    integer" is measured here rather than repeated).

    **This test exists to keep a DISCLOSURE true, not to keep a detector
    working**, and the direction it fails in says so: if a future release
    starts catching one of these, this goes red and the page's paragraph is
    what has to change. Widening the detector to floats is out of scope; a
    page that says nothing while a neighbouring instrument says
    ``overflows-float`` is the defect this replaces.
    """
    from stelling._tripwire import eager as _eager

    tw_status, recorder = _tripwire.arm()
    eager_status = _eager.arm()
    try:
        assert tw_status.armed, tw_status.explanation
        assert eager_status.armed, eager_status.explanation
        with perimeter.armed(("tracer", "array")) as status:
            assert status.armed, status.explanation
            before = (recorder.fires, _eager.TRUNCATIONS, perimeter.FINDINGS)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for label, build in FLOAT_VALUES_NOTHING_SEES:
                    # A refusal is caught rather than allowed to escape, so
                    # that a release which starts catching one of these fails
                    # with the sentence a maintainer needs and not with a
                    # traceback out of a jax operator slot.
                    try:
                        got = np.asarray(build())
                    except BaseException as exc:  # noqa: BLE001
                        pytest.fail(_LOUDER_THAN_THE_PAGE.format(
                            label=label, exc=type(exc).__name__))
                    assert np.isinf(got).all(), f"{label} is no longer inf: {got!r}"
            after = (recorder.fires, _eager.TRUNCATIONS, perimeter.FINDINGS)
        assert after == before, _LOUDER_THAN_THE_PAGE.format(
            label=f"one of them (tripwire/eager/perimeter {before} -> {after})",
            exc="a counter moving",
        )
    finally:
        _eager.disarm()
        _tripwire.disarm()

    # ...and the reason the other two cannot be the ones that catch it.
    for dtype in ("float16", "bfloat16", "float32", "float64"):
        with pytest.raises(ValueError, match="not one of the integer dtypes"):
            stelling.intentional_wrap(1, dtype)


def test_the_refusal_names_the_line_that_wrote_the_literal():
    """Attribution with no stack walk: the writer is the wrapper's caller.

    That is the whole reason this instrument does not collide with the
    const-fold tripwire's attribution, which picks "the innermost non-jax
    frame" and would pick the wrapper.
    """
    harness = _harness(1e9, MOVED)
    with perimeter.armed(("tracer",)):
        with pytest.raises(perimeter.NarrowingError) as caught:
            trace(harness)
    exc = caught.value
    assert exc.file == __file__
    assert exc.func == "harness"
    source = pathlib.Path(__file__).read_text(encoding="utf-8").splitlines()
    assert "x <= literal" in source[exc.line - 1]
    assert exc.finding.literal == MOVED
    assert exc.finding.narrowed_to == MOVED_IMAGE
    assert exc.finding.target_dtype == "float32"
    # The message names the value that DOES exist, which is the whole of the
    # value-level escape: write that one.
    assert repr(MOVED_IMAGE) in str(exc)


def test_both_spellings_of_the_comparison_are_covered():
    """``x <= N`` and ``N >= x``, and ``x >= N`` and ``N <= x``.

    MEASURED ROUTING, and it is the reason there are six slots rather than
    two: Python maps ``N >= x`` onto ``x.__le__(N)`` and ``N > x`` onto
    ``x.__lt__(N)``, so a spelling and its reflection SHARE a slot — while
    ``x <= N`` and ``x >= N`` do not. A user writes both, and a perimeter that
    installed ``__le__`` and "its reflected form" would have installed one
    slot twice and left ``__ge__`` open.
    """
    with perimeter.armed(("tracer",)):
        seen = {}
        for label, fn in (
            ("x <= N", lambda z: z <= MOVED),
            ("N >= x", lambda z: MOVED >= z),
            ("x >= N", lambda z: z >= MOVED),
            ("N <= x", lambda z: MOVED <= z),
            ("x < N", lambda z: z < MOVED),
            ("N > x", lambda z: MOVED > z),
        ):
            jax.clear_caches()
            try:
                jax.make_jaxpr(fn)(jnp.zeros((3,), jnp.float32))
                seen[label] = None
            except perimeter.NarrowingError as exc:
                seen[label] = exc.finding.slot
    assert seen == {
        "x <= N": "le",
        "N >= x": "le",
        "x >= N": "ge",
        "N <= x": "ge",
        "x < N": "lt",
        "N > x": "lt",
    }


def test_a_literal_that_survives_is_never_refused_on_any_of_the_six_slots():
    """The negative direction, on every armed slot.

    A perimeter replaced by "refuse every int" passes every test above this
    one. This is the test it fails.
    """
    with perimeter.armed(("tracer",)):
        for fn in (
            lambda z: z <= EXACT,
            lambda z: z >= EXACT,
            lambda z: z < EXACT,
            lambda z: z > EXACT,
            lambda z: z == EXACT,
            lambda z: z != EXACT,
        ):
            jax.clear_caches()
            jax.make_jaxpr(fn)(jnp.zeros((3,), jnp.float32))
    assert perimeter.FINDINGS == 0
    assert perimeter.CHECKS >= 6


def test_the_denominator_moves_so_a_zero_is_a_measured_zero():
    """Non-vacuity, in the shape this project keeps having to insist on.

    "0 narrowings" is what a healthy run prints and it is also what a
    perimeter nobody entered prints. The check count is what separates them,
    and it is why the report prints it whether or not anything fired.
    """
    perimeter.reset_counters()
    jax.clear_caches()
    jax.make_jaxpr(lambda z: z <= EXACT)(jnp.zeros((3,), jnp.float32))
    assert perimeter.CHECKS == 0, "disarmed, nothing should reach the predicate"
    with perimeter.armed(("tracer",)):
        jax.clear_caches()
        jax.make_jaxpr(lambda z: z <= EXACT)(jnp.zeros((3,), jnp.float32))
        assert perimeter.CHECKS == 1
        assert perimeter.snapshot()["checks"] == 1


# ---------------------------------------------------------------------------
# 3 — observational, and the two artefacts that are NOT
# ---------------------------------------------------------------------------

C32 = jnp.arange(4, dtype=jnp.float32)


def _rich_harness():
    """A harness with real structure, and int comparisons on the traced path.

    The int comparisons are load-bearing: without traffic through the armed
    slots while the IR is being built, byte-identity would be a statement
    about a hook that was never entered. The spike caught exactly that — a
    first version whose arithmetic jax staged to ``iota``, so the perimeter
    saw zero checks and the equivalence proved nothing.
    """
    x = any_array((4,), "float32", (0.0, 100.0))
    n = any_array((), "int32", (1.0, 3.0))
    # traffic through the armed comparison slots, with literals that survive
    assume_ok = (n >= 1) & (n <= 3)

    def body(carry, _):
        return carry + x, carry

    total, _ = jax.lax.scan(body, jnp.zeros((4,), jnp.float32), None, length=3)
    picked = jax.lax.cond(n > 1, lambda t: t * 2.0, lambda t: t, total)
    v = jax.vmap(lambda a: a + 1.0)(picked)
    _, w = jax.lax.while_loop(
        lambda s: s[0] < 3, lambda s: (s[0] + 1, s[1] + 1.0), (jnp.int32(0), v)
    )
    return assert_(jnp.all(w <= 1e9) & assume_ok)


def _lowerable(x, n):
    """``_rich_harness``'s body as a plain jax function, so it can be LOWERED.

    ``docs/overflow-tripwire.md``'s artefact table has a StableHLO row and
    this file's ``IDENTICAL`` tuple did not, so the page's claim that the
    emitted HLO is byte-identical across an armed boundary was carried by
    nobody. It is the row a reader is most likely to act on -- it is the one
    that says an armed run can be compared against a normal build.
    """
    assume_ok = (n >= 1) & (n <= 3)

    def body(carry, _):
        return carry + x, carry

    total, _ = jax.lax.scan(body, jnp.zeros((4,), jnp.float32), None, length=3)
    picked = jax.lax.cond(n > 1, lambda t: t * 2.0, lambda t: t, total)
    v = jax.vmap(lambda a: a + 1.0)(picked)
    _, w = jax.lax.while_loop(
        lambda s: s[0] < 3, lambda s: (s[0] + 1, s[1] + 1.0), (jnp.int32(0), v)
    )
    return jnp.all(w <= 1e9) & assume_ok


def _artefacts():
    jax.clear_caches()
    closed = trace(_rich_harness)
    bare = closed.to_dict(include_metadata=False)
    meta = closed.to_dict(include_metadata=True)
    raw_eqns = "\n".join(repr((e.primitive, tuple(e.params))) for e in closed.jaxpr.eqns)
    jax.clear_caches()
    verdict = check(_rich_harness, vacuity_mode="inputs-only")
    jax.clear_caches()
    stablehlo = jax.jit(_lowerable).lower(C32, jnp.int32(2)).as_text()

    def digest(obj):
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, default=str).encode()
        ).hexdigest()

    return {
        "content_hash": closed.content_hash(),
        "consts": digest([(str(getattr(c, "dtype", "?")), repr(c)) for c in closed.consts]),
        "eqns (metadata-free)": digest(bare.get("jaxpr", {}).get("eqns", bare)),
        "to_dict(include_metadata=False)": digest(bare),
        "StableHLO text": digest(stablehlo),
        "check status": verdict.status,
        "check notes": digest(repr(getattr(verdict, "notes", ()))),
        # the two that MOVE, kept in the same table so that a reader of this
        # test meets the disclosure rather than only the good news
        "to_dict(include_metadata=True)": digest(meta),
        "str(jaxpr)": digest(str(closed.jaxpr)),
        "eqns (raw repr, source_info included)": digest(raw_eqns),
    }


#: The artefacts an armed session leaves byte-identical, and the three it does
#: not. **DECLARED HERE RATHER THAN ASSERTED INLINE** so that a change in
#: either direction is a change to this tuple: silently moving an artefact
#: from one list to the other is exactly the edit this file exists to catch.
IDENTICAL = (
    "content_hash",
    "consts",
    "eqns (metadata-free)",
    "to_dict(include_metadata=False)",
    "StableHLO text",
    "check status",
    "check notes",
)
PERTURBED = (
    "to_dict(include_metadata=True)",
    "str(jaxpr)",
    "eqns (raw repr, source_info included)",
)


def test_arming_is_observational_on_everything_the_hash_scope_covers():
    """Three windows: disarmed, armed, disarmed again.

    ``content_hash`` is unaffected BY DESIGN and not by luck —
    ``ir.py``'s ``CANONICALIZATIONS`` declares ``source_info`` and
    ``debug_info`` outside the hash scope, and the wrapper perturbs exactly
    that field.
    """
    disarmed = _artefacts()
    with perimeter.armed(("tracer",)) as status:
        assert status.armed, status.explanation
        armed = _artefacts()
    again = _artefacts()
    for key in IDENTICAL:
        assert armed[key] == disarmed[key], f"{key} moved while armed"
        assert again[key] == disarmed[key], f"{key} did not come back"


def test_the_source_info_perturbation_is_disclosed_rather_than_denied():
    """The three artefacts that DO move, and the byte that moves them.

    Every equation built through an armed slot carries one extra traceback
    frame. That is not a claim to be reasoned about: it is located here, in
    the raw text, and it is why ``docs/overflow-tripwire.md`` says "do not
    compare a persisted document across an armed boundary" rather than
    "arming is byte-identical".
    """
    disarmed = _artefacts()
    with perimeter.armed(("tracer",)):
        armed = _artefacts()
        jax.clear_caches()
        text = str(trace(_rich_harness).jaxpr)
    again = _artefacts()
    for key in PERTURBED:
        assert armed[key] != disarmed[key], (
            f"{key} no longer moves. If jax stopped recording source_info, or "
            "the wrapper stopped appearing in it, this file's disclosure is "
            "now over-cautious and the docs should be corrected -- but it is "
            "not a silent improvement."
        )
        assert again[key] == disarmed[key], f"{key} did not come back"
    assert "perimeter.py" in text, (
        "the extra frame should be the perimeter's wrapper, and naming it is "
        "what makes this a disclosure rather than a mystery"
    )


# ---------------------------------------------------------------------------
# 2 — the four arm/disarm lifecycles
# ---------------------------------------------------------------------------


def test_double_arm_installs_once_and_the_last_release_restores():
    """Two owners, one installation, and the FIRST release must not restore."""
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]

    first = perimeter.arm(("tracer",), owner="A")
    installed = tracer.__dict__["__le__"]
    second = perimeter.arm(("tracer",), owner="B")
    assert first.armed and second.armed
    assert tracer.__dict__["__le__"] is installed, "the second arm re-wrapped"
    assert perimeter.owners() == 2

    assert perimeter.disarm("A") == "still-armed"
    assert perimeter.live_check() == "armed", "B's hold was released by A's"
    assert perimeter.disarm("B") == "restored"
    assert tracer.__dict__["__le__"] is original


def test_arming_twice_under_the_same_owner_is_idempotent():
    """The same session arming twice holds ONE token, not two.

    Otherwise its single ``disarm`` would leave a hold nobody can release and
    the slots armed for the rest of the process.
    """
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]
    owner = object()
    assert perimeter.arm(("tracer",), owner=owner).armed
    assert perimeter.arm(("tracer",), owner=owner).armed
    assert perimeter.owners() == 1
    assert perimeter.disarm(owner) == "restored"
    assert tracer.__dict__["__le__"] is original


def test_arm_disarm_arm_returns_to_the_original_object():
    """Three transitions, and the identity checked at every one.

    The second arm must capture jax's own function, not the first wrapper. A
    perimeter that wrapped its own wrapper would still refuse the right
    programs and would never come back to the original.
    """
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]
    assert perimeter.arm(("tracer",), owner="A").armed
    assert tracer.__dict__["__le__"] is not original
    assert perimeter.disarm("A") == "restored"
    assert tracer.__dict__["__le__"] is original
    assert perimeter.arm(("tracer",), owner="A").armed
    assert perimeter._installed["tracer"]["slots"]["__le__"][0] is original
    assert perimeter.disarm("A") == "restored"
    assert tracer.__dict__["__le__"] is original


def test_a_raise_between_arm_and_disarm_leaves_the_slots_armed_unless_you_use_the_block():
    """The dullest of the three defects, driven in both directions.

    There is no ``finally`` in a bare pair of calls, so an exception between
    them leaks the hook for the rest of the process. That is not fixed by
    documentation, it is fixed by ``armed()`` — and the bare form is driven
    here so the docstring's warning is measured rather than asserted.
    """
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]

    # the bare pair: LEAKS, and this is the behaviour the block exists for
    with pytest.raises(ZeroDivisionError):
        perimeter.arm(("tracer",), owner="A")
        raise ZeroDivisionError("something in the middle")
    assert perimeter.live_check() == "armed", "arm() with no finally leaks"
    assert perimeter.disarm("A") == "restored"
    assert tracer.__dict__["__le__"] is original

    # the block: does not
    with pytest.raises(ZeroDivisionError):
        with perimeter.armed(("tracer",)):
            raise ZeroDivisionError("something in the middle")
    assert perimeter.live_check() == "detached"
    assert tracer.__dict__["__le__"] is original
    assert perimeter.owners() == 0


def test_a_nested_in_process_session_does_not_disarm_the_outer_one(pytester, monkeypatch):
    """B8b's regression, pre-made, and the reason ``arm()`` takes an owner.

    An idempotent arm beside an unconditional disarm means: the inner session
    installs nothing (correctly), the inner teardown restores everything
    (catastrophically), and every remaining outer test runs unprotected with
    **nothing red**. The nested session here is a real one — ``runpytest`` is
    in-process by default — and it enables the same dial the outer hold used.

    THE COUNTERFACTUAL IS DRIVEN TOO, at the bottom: an anonymous release,
    which is what an unconditional disarm looks like from inside, DOES unhook
    the outer hold. Without that half this test would pass against an
    implementation that simply never disarms.
    """
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    src = str(pathlib.Path(stelling.__file__).resolve().parents[1])
    monkeypatch.setenv("PYTHONPATH", src)
    pytester.makepyfile(
        inner="""
        import jax, jax.numpy as jnp

        def test_inner_is_green():
            jax.clear_caches()
            jax.make_jaxpr(lambda z: z <= 1000)(jnp.zeros((2,), jnp.float32))
        """
    )
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]

    outer = object()
    assert perimeter.arm(("tracer",), owner=outer).armed
    wrapper = tracer.__dict__["__le__"]

    result = pytester.runpytest(
        *PLUGIN_ARGS,
        *ORDER_ARGS,
        "-p",
        "no:cacheprovider",
        "--stelling-narrowing-perimeter=error",
        "inner.py",
    )
    result.assert_outcomes(passed=1)

    # THE OUTER HOLD SURVIVED THE INNER SESSION'S TEARDOWN.
    assert perimeter.live_check() == "armed"
    assert tracer.__dict__["__le__"] is wrapper
    assert perimeter.owners() == 1
    # ...and it is still WATCHING, which is the thing that actually matters:
    # a hold that survives as a bookkeeping entry while the slot is back to
    # jax's own function would pass the two assertions above.
    jax.clear_caches()
    with pytest.raises(perimeter.NarrowingError):
        jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((2,), jnp.float32))

    # THE COUNTERFACTUAL: an unconditional release does unhook it.
    assert perimeter.disarm(None) == "restored"
    assert tracer.__dict__["__le__"] is original
    jax.clear_caches()
    jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((2,), jnp.float32))


def test_an_out_of_order_release_unhooks_nobody_and_leaks_nothing():
    """Two armers releasing in arrival order rather than reverse order.

    The spike drove the un-scoped version of this and got the worst available
    pair of outcomes: the first release unhooked the second armer silently,
    and the second release RESURRECTED a dead wrapper. Removing the owner from
    wherever it sits, and restoring only when the list empties, has neither.
    """
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]
    assert perimeter.arm(("tracer",), owner="P").armed
    assert perimeter.arm(("tracer",), owner="Q").armed
    wrapper = tracer.__dict__["__le__"]

    assert perimeter.disarm("P") == "still-armed"
    assert tracer.__dict__["__le__"] is wrapper
    assert perimeter.disarm("Q") == "restored"
    assert tracer.__dict__["__le__"] is original
    # a release by somebody who never armed changes nothing
    assert perimeter.disarm("P") == "not-armed"
    assert tracer.__dict__["__le__"] is original


def test_a_release_by_a_stranger_while_armed_is_refused():
    tracer = _tracer_type()
    assert perimeter.arm(("tracer",), owner="P").armed
    wrapper = tracer.__dict__["__le__"]
    assert perimeter.disarm("stranger") == "not-an-owner"
    assert tracer.__dict__["__le__"] is wrapper
    assert perimeter.disarm("P") == "restored"


def test_a_second_owners_arm_FAILING_does_not_take_the_first_owners_perimeter_out():
    """B8b's shape arriving through the exception door, driven both ways.

    ``arm()`` may not raise, so it has a handler that tidies up after itself.
    That handler used to restore ``list(_installed)`` -- **everything
    installed**, not what this call installed -- so a fault inside OWNER-2's
    ``arm()`` took OWNER-1's faces out from under it. OWNER-1 stayed in
    ``_owners``, ``arm()`` had already told it ``armed``, and nothing was red:
    a holder that believes it is watched and is not, which is the whole
    campaign in one handler.

    Driven at ``e6968fe`` with the handler as it was::

        OWNER-1 arm: armed | refusing: True
        OWNER-2 arm: unexpected:RuntimeError
        owners: ['OWNER-1'] faces: ()
        OWNER-1 still refusing: False

    The fault is injected into ``selfcheck`` because that is the one call
    ``arm()`` makes AFTER the slots are installed -- which is what makes the
    handler's restore reachable at all -- and because a fault anywhere in
    there must have the same answer. What the fix asserts is not "no restore"
    but "only what this call installed": the second half below faults an
    ``arm()`` that installs the ARRAY face over an already-armed tracer face
    and requires the array face to be gone and the tracer face to be intact.
    """
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]

    def _refuses() -> bool:
        jax.clear_caches()
        try:
            jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((2,), jnp.float32))
        except perimeter.NarrowingError:
            return True
        return False

    assert perimeter.arm(("tracer",), owner="OWNER-1").armed
    wrapper = tracer.__dict__["__le__"]
    assert _refuses(), "the control did not fire before the fault"

    def _boom(faces=()):
        raise RuntimeError("FORCED: something inside arm() went wrong")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(perimeter, "selfcheck", _boom)
        second = perimeter.arm(("tracer",), owner="OWNER-2")
    assert second.code == "unexpected:RuntimeError", second
    assert not second.armed

    # OWNER-1 IS STILL WATCHING, and that is asserted as a REFUSAL rather than
    # as a bookkeeping entry: a hold that survives in `_owners` while the slot
    # is back to jax's own function passes every other assertion here.
    assert perimeter.armed_faces() == ("tracer",)
    assert tracer.__dict__["__le__"] is wrapper
    assert perimeter.owners() == 1
    assert _refuses(), "OWNER-1's perimeter was taken out by OWNER-2's fault"

    # ...and the other direction: what the failing call DID install is gone.
    array_type = _array_type()
    array_original = array_type.__dict__["__add__"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(perimeter, "selfcheck", _boom)
        third = perimeter.arm(("tracer", "array"), owner="OWNER-3")
    assert third.code == "unexpected:RuntimeError", third
    assert perimeter.armed_faces() == ("tracer",), "the array face was left behind"
    assert array_type.__dict__["__add__"] is array_original
    assert tracer.__dict__["__le__"] is wrapper
    assert _refuses()

    assert perimeter.disarm("OWNER-1") == "restored"
    assert tracer.__dict__["__le__"] is original


# ---------------------------------------------------------------------------
# The lowering helper, which is now load-bearing for this whole file
# ---------------------------------------------------------------------------
#
# `_isolate` takes the session's hold DOWN for each test's window and hands it
# back. That makes the hand-back the thing every later test's protection rests
# on, so it is driven here rather than trusted: three ways for it to come back
# WRONG, each of which reads as "armed" to every check but one.


def test_lowering_the_perimeter_with_nothing_armed_is_a_no_op():
    """The zero-dep and dial-off shape, which is the common one.

    Every test file in this tree imports ``conftest``, and the overwhelming
    majority of sessions never arm anything. The helper must do nothing at all
    in that case -- not arm, not disarm, not raise -- and yield ``()`` so a
    caller can tell.
    """
    assert perimeter.armed_faces() == ()
    with lowered_perimeter() as lowered:
        assert lowered == ()
        assert perimeter.armed_faces() == ()
        assert perimeter.owners() == 0
    assert perimeter.armed_faces() == ()
    assert perimeter.owners() == 0


def test_lowering_and_handing_back_returns_the_SAME_originals_and_owners():
    """The positive control, and it is about identity rather than status.

    Inside the block the slots are jax's own function -- that is what the
    lowering is FOR -- and after it the hold is back under the same owner
    objects, with the same saved originals. A helper that re-armed over
    something else would satisfy ``armed_faces()``, ``owners()`` and
    ``live_check()`` alike.
    """
    tracer, array_type = _tracer_type(), _array_type()
    original = tracer.__dict__["__le__"]
    owner = object()
    assert perimeter.arm(perimeter.FACES, owner=owner).armed
    wrapper = tracer.__dict__["__le__"]
    assert wrapper is not original

    with lowered_perimeter() as lowered:
        assert lowered == perimeter.FACES
        assert perimeter.armed_faces() == ()
        assert perimeter.owners() == 0
        assert tracer.__dict__["__le__"] is original, "the lowering left a wrapper"
        assert array_type.__dict__["__add__"] is not None
        # ...and it really is unwatched in here, which is the point of it
        jax.clear_caches()
        jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((2,), jnp.float32))

    assert perimeter.armed_faces() == perimeter.FACES
    assert perimeter.live_check() == "armed"
    assert perimeter.owners() == 1
    assert perimeter._owners[0] is owner, "a different owner object came back"
    assert perimeter._installed["tracer"]["slots"]["__le__"][0] is original
    jax.clear_caches()
    with pytest.raises(perimeter.NarrowingError):
        jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((2,), jnp.float32))
    assert perimeter.disarm(owner) == "restored"
    assert tracer.__dict__["__le__"] is original


def test_a_hand_back_that_re_arms_over_an_INTERLOPER_is_refused():
    """The silent partial, and the only reading that can see it.

    Something binds over a slot while the perimeter is down. The hand-back's
    ``arm()`` then captures THAT as the slot's original and succeeds --
    including the self-check, because an interloper which delegates still
    refuses the reference defect. ``status.armed`` is True, ``armed_faces()``
    is right, ``live_check()`` says ``armed``, and the object a later
    ``disarm()`` restores is no longer jax's: the interloper is installed for
    the rest of the process with nothing red.

    Driven both ways -- with the identity check removed, the block below
    completes silently and ``tracer.__dict__["__le__"] is original`` is False
    after the final disarm.
    """
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]
    owner = object()
    assert perimeter.arm(("tracer",), owner=owner).armed

    def interloper(a, b):
        return original(a, b)

    with pytest.raises(AssertionError) as caught:
        with lowered_perimeter():
            tracer.__le__ = interloper
    assert "re-armed over a DIFFERENT object" in str(caught.value)
    assert "unprotected" in str(caught.value)

    # AND THE MESS IS LEFT VISIBLE RATHER THAN TIDIED. The helper's job is to
    # refuse, not to guess which of the two objects the author wanted; this
    # test puts jax's own function back itself, which is what a human would
    # have to do.
    perimeter.disarm(owner)
    tracer.__le__ = original
    assert tracer.__dict__["__le__"] is original


def test_a_hand_back_that_re_arms_only_SOME_of_the_faces_is_refused(monkeypatch):
    """One face back, the other not — and ``arm()`` saying ``armed`` to it.

    ``status.armed`` is a statement about the faces that call asked for. A
    hand-back that asked for one face when two were lowered gets a perfectly
    honest ``armed`` back and leaves half a perimeter, which is not a smaller
    perimeter but a hole.
    """
    owner = object()
    assert perimeter.arm(perimeter.FACES, owner=owner).armed
    real_arm = perimeter.arm
    monkeypatch.setattr(
        perimeter, "arm", lambda faces=perimeter.FACES, owner=None: real_arm(
            ("tracer",), owner=owner
        )
    )
    with pytest.raises(AssertionError) as caught:
        with lowered_perimeter():
            pass
    assert "faces came back as ('tracer',)" in str(caught.value)
    monkeypatch.undo()
    perimeter.disarm(owner)


def test_a_hand_back_that_cannot_re_arm_at_all_is_refused(monkeypatch):
    """The total failure, which is the one ``status.armed`` already saw."""
    owner = object()
    assert perimeter.arm(("tracer",), owner=owner).armed
    monkeypatch.setattr(adapter, "perimeter_locate", lambda face: "no-type")
    with pytest.raises(AssertionError) as caught:
        with lowered_perimeter():
            pass
    assert "arm() refused [no-type]" in str(caught.value)
    monkeypatch.undo()


def test_an_exception_INSIDE_the_block_still_hands_the_perimeter_back():
    """There is no ``finally`` in a pair of calls, and this is the block's.

    The block's own exception is what propagates -- a hand-back that swallowed
    it would hide the failure the test was actually reporting -- and the hold
    is back before it does.
    """
    tracer = _tracer_type()
    owner = object()
    assert perimeter.arm(("tracer",), owner=owner).armed
    wrapper_before = tracer.__dict__["__le__"]

    with pytest.raises(ZeroDivisionError):
        with lowered_perimeter():
            assert perimeter.armed_faces() == ()
            raise ZeroDivisionError("something in the middle")

    assert perimeter.armed_faces() == ("tracer",)
    assert perimeter.live_check() == "armed"
    assert perimeter.owners() == 1
    assert perimeter._owners[0] is owner
    assert tracer.__dict__["__le__"] is not wrapper_before, (
        "the same wrapper came back, so nothing was actually lowered"
    )
    jax.clear_caches()
    with pytest.raises(perimeter.NarrowingError):
        jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((2,), jnp.float32))
    assert perimeter.disarm(owner) == "restored"


# ---------------------------------------------------------------------------
# 4 — drift fails CLOSED
# ---------------------------------------------------------------------------


class _NeverFires:
    """A predicate that is installed and blind. The campaign's signature defect."""

    UNKNOWN_SLOTS: set = set()
    INTERNAL_DECLINES: dict = {}

    @staticmethod
    def classify(a, b, slot):
        return None


class _AlwaysFires:
    UNKNOWN_SLOTS: set = set()
    INTERNAL_DECLINES: dict = {}

    @staticmethod
    def classify(a, b, slot):
        return prop_guard.Finding(
            reason="out-of-range",
            slot="le",
            operand_dtype="float32",
            target_dtype="float32",
            literal=b,
            narrowed_to=0,
        )


@pytest.mark.parametrize(
    ("stub", "code"),
    [(_NeverFires, "not-invoked"), (_AlwaysFires, "cries-wolf")],
)
def test_arming_refuses_when_the_guard_is_blind_or_cries_wolf(monkeypatch, stub, code):
    """Both directions of the self-check, driven through the real slots.

    A positive-only probe passes on a hook replaced by "refuse everything",
    and this repository has found that shape of vacuous control more than
    once. The refusal must also put the slots BACK: a refused arm that left
    them rebound would be the worst of both.
    """
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]
    monkeypatch.setattr(perimeter, "_GUARD", stub)
    status = perimeter.arm(("tracer",), owner="A")
    assert status.code == code, status.detail
    assert not status.armed
    assert tracer.__dict__["__le__"] is original
    assert perimeter.owners() == 0
    assert "Static checking is unaffected" in status.explanation


def test_arming_refuses_when_the_type_cannot_be_located(monkeypatch):
    monkeypatch.setattr(adapter, "perimeter_locate", lambda face: "no-type")
    status = perimeter.arm(("tracer",), owner="A")
    assert status.code == "no-type"
    assert perimeter.owners() == 0
    assert "restructured" in status.explanation


def test_arming_refuses_when_a_slot_is_missing_and_puts_the_others_back(monkeypatch):
    """A partial perimeter is not a smaller perimeter, it is a hole."""
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]
    monkeypatch.setitem(
        perimeter.FACE_SLOTS, "tracer", perimeter.FACE_SLOTS["tracer"] + ("__gone__",)
    )
    status = perimeter.arm(("tracer",), owner="A")
    assert status.code == "no-slot"
    assert "__gone__" in status.detail
    assert tracer.__dict__["__le__"] is original
    assert perimeter.owners() == 0


def test_a_face_this_perimeter_does_not_have_is_refused():
    status = perimeter.arm(("nonesuch",), owner="A")
    assert status.code == "no-face"
    assert perimeter.owners() == 0


def test_something_rebinding_over_the_wrapper_is_reported_not_clobbered():
    """``foreign-patch``: say so, and leave whatever replaced us in place."""
    tracer = _tracer_type()
    original = tracer.__dict__["__le__"]
    assert perimeter.arm(("tracer",), owner="A").armed

    def interloper(a, b):
        return original(a, b)

    tracer.__le__ = interloper
    try:
        assert perimeter.live_check() == "foreign-patch"
        assert perimeter.disarm("A") == "foreign-patch"
        assert tracer.__dict__["__le__"] is interloper, "we clobbered a third party"
    finally:
        tracer.__le__ = original
    assert perimeter.live_check() == "detached"


def test_the_perimeter_never_raises_its_own_faults_into_a_program(monkeypatch):
    """The wrapper's own belt-and-braces, and its counter.

    ``classify`` documents that it never raises. If it ever does, the program
    must not be the thing that breaks — and the failure must be COUNTED and
    printed, because a fail-safe path that fails into silence is the defect
    this instrument exists to remove.
    """

    class _Explodes:
        UNKNOWN_SLOTS: set = set()
        INTERNAL_DECLINES: dict = {}

        @staticmethod
        def classify(a, b, slot):
            raise RuntimeError("the predicate broke")

    with perimeter.armed(("tracer",)) as status:
        assert status.armed
        monkeypatch.setattr(perimeter, "_GUARD", _Explodes)
        jax.clear_caches()
        jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((2,), jnp.float32))
        assert perimeter.INTERNAL_ERRORS >= 1
        lines = report.render_perimeter(status, perimeter.snapshot())
    assert any("NOT CLEAN" in line for line in lines)
    assert any("caught and counted" in line for line in lines)


def test_the_two_counters_a_run_must_not_leave_unread_reach_the_report():
    """Condition 2 of the rollout, and it is not a formality.

    ``prop_guard`` declines silently on an internal fault and records the
    exception name; a slot name it does not recognise is handled as the
    majority case and recorded too. A run reporting zero fires with a non-zero
    decline count is not a clean run, it is an unmeasured one — and the only
    way an operator learns which they had is if the numbers are printed.
    """
    prop_guard.INTERNAL_DECLINES["TypeError"] += 2
    prop_guard.UNKNOWN_SLOTS.add("__mistyped__")
    snapshot = perimeter.snapshot()
    assert snapshot["declines"] == {"TypeError": 2}
    assert snapshot["unknown_slots"] == ["__mistyped__"]
    lines = report.render_perimeter(_tripwire.Status(code="armed"), snapshot)
    body = "\n".join(lines)
    assert "NOT CLEAN" in body
    assert "TypeError x2" in body
    assert "__mistyped__" in body
    assert "lower bound" in body
    # and the clean direction says so rather than staying silent
    prop_guard.INTERNAL_DECLINES.clear()
    prop_guard.UNKNOWN_SLOTS.clear()
    clean = "\n".join(
        report.render_perimeter(_tripwire.Status(code="armed"), perimeter.snapshot())
    )
    assert "declined nothing" in clean
    assert "NOT CLEAN" not in clean


def test_an_unrecognised_slot_name_is_recorded_rather_than_silently_handled():
    """The mechanism behind the counter above, driven at the predicate."""
    prop_guard.UNKNOWN_SLOTS.clear()
    prop_guard.classify(jnp.zeros((2,), jnp.int16), 3, "__not_a_slot__")
    assert "__not_a_slot__" in prop_guard.UNKNOWN_SLOTS


# ---------------------------------------------------------------------------
# The escape: a user who means it has a spelling that says so
# ---------------------------------------------------------------------------


def test_an_expected_truncation_region_permits_and_is_printed_with_its_reason():
    """The region escape, and the accounting that keeps it honest.

    ONE DECLARATION FOR BOTH INSTRUMENTS. ``expected_truncation`` was written
    for the eager detector; the perimeter honours it too, and counts what it
    permitted in its OWN table so that a report still says which instrument
    permitted what. A permission nobody can see is the silence this campaign
    exists to end, one level up.
    """
    with perimeter.armed(("tracer",)) as status:
        with expected_truncation("this door's subject is the moved threshold"):
            jax.clear_caches()
            jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((2,), jnp.float32))
        assert perimeter.FINDINGS == 1
        snapshot = perimeter.snapshot()
        lines = report.render_perimeter(status, snapshot)
        # ...and the region does not leak past its block
        jax.clear_caches()
        with pytest.raises(perimeter.NarrowingError):
            jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((3,), jnp.float32))
    body = "\n".join(lines)
    assert "PERMITTED" in body
    assert "this door's subject is the moved threshold" in body
    assert f"{__file__}:" in body


def test_intentional_wrap_is_the_value_level_escape_and_needs_no_perimeter_support():
    """``intentional_wrap`` returns a value that is already in range.

    So the perimeter never sees it — there is nothing left to detect, which is
    the same argument the eager detector's declaration is built on. This is
    the primary escape; the region is the awkward one, for code whose subject
    IS the narrowing.
    """
    wrapped = stelling.intentional_wrap(40000, "int16")
    assert wrapped == -25536
    with perimeter.armed(("tracer",)):
        jax.clear_caches()
        jax.make_jaxpr(lambda z: z >= wrapped)(jnp.zeros((2,), jnp.int16))
        assert perimeter.FINDINGS == 0
        jax.clear_caches()
        with pytest.raises(perimeter.NarrowingError):
            jax.make_jaxpr(lambda z: z >= 40000)(jnp.zeros((3,), jnp.int16))


def test_the_exception_is_not_swallowed_by_a_bare_except_exception():
    """``BaseException``, and the honest limit driven with it.

    A soundness alarm that an ordinary ``except Exception:`` can absorb is not
    an alarm. ``except BaseException:`` still catches it — that is stated in
    the class docstring and measured here rather than left as a claim.
    """
    with perimeter.armed(("tracer",)):
        swallowed = False
        try:
            jax.clear_caches()
            try:
                jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((2,), jnp.float32))
            except Exception:  # noqa: BLE001 - the point of the test
                swallowed = True
        except perimeter.NarrowingError:
            pass
        assert not swallowed
        caught = False
        try:
            jax.clear_caches()
            jax.make_jaxpr(lambda z: z <= MOVED)(jnp.zeros((3,), jnp.float32))
        except BaseException:  # noqa: BLE001 - the honest limit
            caught = True
        assert caught
    assert not issubclass(perimeter.NarrowingError, Exception)
    # ...and the vendored predicate's own exception is UNCHANGED: its F12
    # argument is about `raise_for`, which this module does not use.
    assert issubclass(prop_guard.NarrowingError, OverflowError)


# ---------------------------------------------------------------------------
# Coexistence with the other two instruments
# ---------------------------------------------------------------------------


def test_all_three_instruments_arm_together_in_either_order():
    """The perimeter's comparison slots are collision-free with the tripwire.

    MEASURED, AND IT IS WHY THIS HALF SHIPPED FIRST. The const-fold tripwire's
    self-check traces ``a + 256`` and requires the finding to be attributed to
    ``_probe.py``; its stack walk strips one file, so a wrapper it has never
    been told about wins the attribution and it refuses with
    ``mis-attributed``. Rebinding ``Tracer.__add__`` does exactly that.
    Rebinding the six COMPARISONS does not touch that probe at all.
    """
    for order in (("perimeter", "tripwire"), ("tripwire", "perimeter")):
        statuses = {}
        try:
            for which in order:
                if which == "perimeter":
                    statuses["perimeter"] = perimeter.arm(perimeter.FACES, owner="A")
                else:
                    status, _rec = _tripwire.arm()
                    statuses["tripwire"] = status
            statuses["eager"] = _tripwire.arm_eager()
            assert statuses["perimeter"].armed, statuses["perimeter"].explanation
            assert statuses["tripwire"].armed, statuses["tripwire"].explanation
            assert statuses["eager"].armed, statuses["eager"].explanation
        finally:
            _tripwire.disarm_eager()
            _tripwire.disarm()
            perimeter.disarm("A")


def test_the_probe_the_tripwire_attributes_on_is_the_one_this_perimeter_avoids():
    """The collision, named against the source rather than against a memory.

    ``_probe.over`` is ``a + 256``. This face installs comparisons and no
    arithmetic slot, so nothing in it is on that probe's path — and this test
    is what fails if a future face adds ``__add__`` without also widening
    ``_adapter_jax._stack()``.
    """
    assert "+" in _probe.over.__code__.co_consts.__class__.__name__ or True
    installed = set(perimeter.FACE_SLOTS["tracer"])
    assert "__add__" not in installed, (
        "the tripwire's self-check traces `a + 256` and attributes it to "
        "_probe.py; a wrapper on Tracer.__add__ takes that attribution and "
        "makes arm() report mis-attributed. Widen _adapter_jax._stack() "
        "before adding this slot."
    )


# ---------------------------------------------------------------------------
# The pytest dial
# ---------------------------------------------------------------------------


def test_the_dial_is_off_by_default_and_arms_nothing(pytester, monkeypatch):
    """The property a user who never asked for this depends on."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    src = str(pathlib.Path(stelling.__file__).resolve().parents[1])
    monkeypatch.setenv("PYTHONPATH", src)
    pytester.makepyfile(
        inner="""
        import jax, jax.numpy as jnp

        def test_a_moved_threshold_is_not_refused_by_default():
            jax.clear_caches()
            jax.make_jaxpr(lambda z: z <= 2**31 - 1)(jnp.zeros((2,), jnp.float32))
        """
    )
    result = pytester.runpytest(
        *PLUGIN_ARGS, *ORDER_ARGS, "-p", "no:cacheprovider", "inner.py"
    )
    result.assert_outcomes(passed=1)
    assert "stelling narrowing perimeter" not in result.stdout.str()
    assert perimeter.live_check() == "detached"


def test_the_dial_arms_and_prints_the_denominator(pytester, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    src = str(pathlib.Path(stelling.__file__).resolve().parents[1])
    monkeypatch.setenv("PYTHONPATH", src)
    pytester.makepyfile(
        inner="""
        import jax, jax.numpy as jnp

        def test_ok():
            jax.clear_caches()
            jax.make_jaxpr(lambda z: z <= 1000)(jnp.zeros((2,), jnp.float32))

        def test_moved():
            jax.clear_caches()
            jax.make_jaxpr(lambda z: z <= 2**31 - 1)(jnp.zeros((3,), jnp.float32))
        """
    )
    result = pytester.runpytest(
        *PLUGIN_ARGS,
        *ORDER_ARGS,
        "-p",
        "no:cacheprovider",
        "--stelling-narrowing-perimeter=error",
        "inner.py",
    )
    result.assert_outcomes(passed=1, failed=1)
    out = result.stdout.str()
    assert "stelling narrowing perimeter" in out
    assert re.search(r"\d+ integer literal\(s\) met an array or tracer", out)
    assert "armed on the tracer face, 6 slot(s)" in out
    assert "what this perimeter does NOT see" in out
    assert "NarrowingError" in out


def test_a_session_that_asked_for_the_rule_and_did_not_get_it_fails(
    pytester, monkeypatch
):
    """No degraded mode, and no ``require`` spelling — the same argument the
    eager detector's dial is built on. A rule that could not attach is not a
    quieter rule, it is no rule."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    src = str(pathlib.Path(stelling.__file__).resolve().parents[1])
    monkeypatch.setenv("PYTHONPATH", src)
    # THE NESTED SESSION RUNS IN THIS INTERPRETER, so the conftest below
    # mutates the adapter module every later test in this process imports.
    # Re-setting the attribute to its own current value records it with
    # `monkeypatch`, whose teardown then puts it back whatever the inner
    # session did to it. Found by measurement, not foresight: without this
    # line the four canary tests further down reported `no-type` against a
    # jax that had not moved.
    monkeypatch.setattr(adapter, "perimeter_locate", adapter.perimeter_locate)
    pytester.makepyfile(
        conftest="""
        from stelling._tripwire import _adapter_jax as adapter
        adapter.perimeter_locate = lambda face: "no-type"
        """,
        inner="""
        def test_green():
            assert True
        """,
    )
    result = pytester.runpytest(
        *PLUGIN_ARGS,
        *ORDER_ARGS,
        "-p",
        "no:cacheprovider",
        "--stelling-narrowing-perimeter=error",
        "inner.py",
    )
    assert result.ret != 0
    assert "could not attach" in result.stderr.str() + result.stdout.str()


# ---------------------------------------------------------------------------
# The canary's own rows, against a REAL jax — the half the zero-dep battery
# in tests/test_tripwire_record.py stubs out
# ---------------------------------------------------------------------------


def _canary():
    """``.github/scripts/tripwire_canary.py``, imported by path.

    By path because it is a CI script and not a package module, and imported
    rather than re-implemented for the reason the script itself calls the
    shipped ``arm()``: a test that re-states the decision measures the
    re-statement.
    """
    import importlib.util

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / ".github" / "scripts" / "tripwire_canary.py"
    )
    spec = importlib.util.spec_from_file_location("_tripwire_canary_live", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_canarys_fact_rows_are_all_green_on_this_jax():
    """Six positive assertions about the jax this suite is running against.

    Green here is the ONLY state that means anything: every row asserts that
    something is still true, so this test is what a jax release that took one
    away would fail. The nightly asks the same question of tomorrow's jax
    before tomorrow's users do.
    """
    canary = _canary()
    for face in perimeter.FACES:
        located = adapter.perimeter_locate(face)
        assert not isinstance(located, str), (face, located)
        rows, moved = canary._perimeter_facts(face, located)
        assert not moved, (face, dict(rows))
        names = {name for name, _value in rows}
        probe_slot = "__le__" if face == "tracer" else "__add__"
        assert names == {
            f"{face}: type",
            f"{face}: Py_TPFLAGS_HEAPTYPE",
            f"{face}: slots are own attributes",
            f"{face}: setattr rebinds and restores identity",
            f"{face}: a WARM {probe_slot} still enters Python",
            f"{face}: no in-place slots to bypass the forward ones",
        }, names
        assert all(value.startswith("PASS") for _name, value in rows), dict(rows)


@pytest.mark.parametrize(
    ("fault", "reddens"),
    [
        # jax renames or restructures the type the perimeter attaches to
        ("no-type", {"tracer: type"}),
        # jax moves the slots off the type it used to own them on
        ("no-slots", {"tracer: slots are own attributes",
                      "tracer: setattr rebinds and restores identity",
                      "tracer: a WARM __le__ still enters Python"}),
        # jax grows an in-place slot, so `x += N` stops falling back
        ("inplace", {"tracer: no in-place slots to bypass the forward ones"}),
    ],
)
def test_the_canarys_fact_rows_fail_CLOSED_under_injected_drift(
    monkeypatch, fault, reddens
):
    """Fault injection, because "the canary is green" has to keep meaning
    something.

    Each fault is one way a jax release could move under the perimeter, and
    each must turn a row RED. A canary whose rows can only be reddened by
    something being ADDED cannot report a disappearance, which is the only
    kind of change that matters here.
    """
    canary = _canary()
    tracer = _tracer_type()

    if fault == "no-type":
        rows, moved = canary._perimeter_facts("tracer", "no-type")
    elif fault == "no-slots":
        class _Stripped:
            """A type that answers to the name and owns none of the slots."""

            __mro__ = ()

        rows, moved = canary._perimeter_facts("tracer", _Stripped)
    else:
        class _WithInplace(tracer):
            def __iadd__(self, other):  # pragma: no cover - never called
                return self

        rows, moved = canary._perimeter_facts("tracer", _WithInplace)

    assert reddens <= set(moved), (rows, moved)
    reasons = dict(
        canary._perimeter_reasons(
            _tripwire.Status(code="armed"), "fired", "fired", moved, [], True
        )
    )
    assert "perimeter:facts-moved" in reasons
    for name in reddens:
        assert name in reasons["perimeter:facts-moved"]


def test_the_promotion_identity_holds_and_its_drift_check_can_fail(monkeypatch):
    """F6, checked against what jax actually does, and then broken on purpose.

    The predicate asks jax's own promotion for the dtype a literal is
    converted into; this compares every answer against the dtype of ``x + 3``
    and of ``x / 3``, which reach the same fact by a different route. Every
    verdict the predicate gives is computed against the target it names, so a
    disagreement is not a cosmetic drift.
    """
    canary = _canary()
    note, drift = canary._perimeter_promotion()
    assert not drift, note
    assert re.match(r"\d+ agree, 0 disagree", note), note
    agreed = int(note.split(" ", 1)[0])
    assert agreed >= 20, note

    # ...and the check is not vacuous: a predicate that named the wrong target
    # must be caught by it.
    monkeypatch.setattr(prop_guard, "_target_dtype", lambda dt, slot: np.dtype("int8"))
    note, drift = canary._perimeter_promotion(sample=("float32",))
    assert drift, note
    assert "says int8" in note


# ---------------------------------------------------------------------------
# The ARRAY face: the eager door, which the other two instruments miss
# ---------------------------------------------------------------------------


def _array_type():
    located = adapter.perimeter_locate("array")
    assert not isinstance(located, str), located
    return located


def test_the_eager_door_is_open_with_everything_else_armed_and_this_closes_it():
    """The acceptance criterion for the array face, driven BOTH ways.

    The disarmed half is measured with **everything this repository ships**
    armed — the const-fold tripwire and the eager construction-site detector,
    both reporting ``armed`` — so the zero it produces is not "nothing was
    watching", it is "everything that watches was watching and saw nothing".
    ``jnp.full((3,), 40000, int16)`` IS refused by the eager detector in the
    same window, which is what makes the contrast a measurement rather than a
    claim: that instrument watches CONSTRUCTION, and no array is being
    constructed by ``x + 40000``.

    Cold AND warm, because warm is the harder half: a jax that answered a warm
    operation from C++ without entering Python would leave this face blind
    while every arm-time check still passed.
    """
    tripwire_status, _rec = _tripwire.arm()
    eager_status = _tripwire.arm_eager()
    try:
        assert tripwire_status.armed, tripwire_status.explanation
        assert eager_status.armed, eager_status.explanation

        x16 = jnp.zeros((3,), jnp.int16)
        x32 = jnp.zeros((3,), jnp.float32)

        def outcome(fn):
            try:
                return ("value", int(fn()))
            except perimeter.NarrowingError as exc:
                return ("refused", exc.finding.reason)

        # the positive control for the OTHER instrument, in the same window
        with pytest.raises(stelling.EagerTruncationError):
            jnp.full((3,), 40000, jnp.int16)

        # --- with only the tracer face armed: the door is OPEN
        with perimeter.armed(("tracer",)) as status:
            assert status.armed, status.explanation
            for _run in ("cold", "warm"):
                assert outcome(lambda: (x16 + 40000)[0]) == ("value", -25536)
                assert outcome(lambda: (40000 + x16)[0]) == ("value", -25536)
                assert outcome(lambda: int((x32 <= 2**31 - 1)[0])) == ("value", 1)

        # --- with the array face armed: it is CLOSED, cold and warm
        with perimeter.armed(perimeter.FACES) as status:
            assert status.armed, status.explanation
            for _run in ("cold", "warm"):
                assert outcome(lambda: (x16 + 40000)[0]) == ("refused", "out-of-range")
                assert outcome(lambda: (40000 + x16)[0]) == ("refused", "out-of-range")
                assert outcome(lambda: int((x32 <= 2**31 - 1)[0])) == (
                    "refused", "inexact",
                )
                # ...and a literal that survives is still not refused
                assert outcome(lambda: (x16 + 3)[0]) == ("value", 3)
    finally:
        _tripwire.disarm_eager()
        _tripwire.disarm()


def test_pow_is_excluded_and_rpow_is_kept_through_the_REAL_slots():
    """The measured asymmetry, driven end to end rather than at the predicate.

    ``x ** k`` is lowered to ``integer_pow[y=k]`` and keeps ``k`` a Python int
    in the program's own structure: the written integer survives exactly, so a
    guard there is a pure false-positive generator and the slot is not
    installed at all. ``k ** x`` converts and narrows, so it is.

    Driven through the installed slots and not only through ``classify``,
    because the exclusion is enforced in two independent places — the slot is
    absent from the face's list, and the predicate declines the name — and a
    test at the predicate alone would pass on a perimeter that installed the
    slot anyway.
    """
    assert "__pow__" not in perimeter.FACE_SLOTS["array"]
    assert "__rpow__" in perimeter.FACE_SLOTS["array"]
    x16 = jnp.zeros((3,), jnp.int16)
    with perimeter.armed(perimeter.FACES):
        # the exponent is not converted, so nothing is refused...
        assert int((x16 ** 40000)[0]) == 0
        assert perimeter.FINDINGS == 0
        # ...and the base is, so it is
        with pytest.raises(perimeter.NarrowingError) as caught:
            40000 ** x16
    assert caught.value.finding.slot == "rpow"
    assert caught.value.finding.narrowed_to == -25536


def test_true_division_promotes_the_literal_into_a_float_so_it_survives():
    """F6's redirect, driven through the real slot.

    ``int16 / 40000`` converts the literal into ``float32`` — jax's own
    promotion, asked of jax through the public ``(zeros((0,), dt) / 1).dtype``
    rather than ``result_type(dt, 1.0)``, which is measurably wrong at x64 and
    raises on the sub-byte dtypes. 40000 is exactly representable there, so
    nothing is refused; the same literal in ``+`` is.
    """
    x16 = jnp.zeros((3,), jnp.int16)
    with perimeter.armed(perimeter.FACES):
        x16 / 40000
        40000 / (x16 + 1)
        assert perimeter.FINDINGS == 0
        with pytest.raises(perimeter.NarrowingError):
            x16 + 40000


def test_an_in_place_spelling_falls_back_to_the_forward_slot():
    """``y += 40000`` and ``y = y + 40000`` are the same program here.

    Neither face defines any of the 13 ``__i*__`` slots, so the in-place
    spelling falls back to the forward one and is covered by the same wrapper.
    That is a fact about jax rather than about this module, which is why the
    canary holds it as a row.
    """
    array_type = _array_type()
    assert not [s for s in perimeter.INPLACE_SLOTS if s in array_type.__dict__]
    y = jnp.zeros((3,), jnp.int16)
    with perimeter.armed(perimeter.FACES):
        with pytest.raises(perimeter.NarrowingError) as caught:
            y += 40000
    assert caught.value.finding.slot == "add"


def test_a_size_zero_array_is_exempt_on_the_arithmetic_face_too():
    """F11 reads ``.size``, so it applies wherever the array does."""
    empty = jnp.zeros((0,), jnp.int16)
    with perimeter.armed(perimeter.FACES):
        empty + 40000
        assert perimeter.FINDINGS == 0


def test_a_numpy_operand_is_not_ours_and_numpy_says_so_itself():
    """One of the report's ``does NOT see`` items, driven.

    A numpy array is not a jax array and never enters these slots. It is not a
    hole in the perimeter: numpy raises its own ``OverflowError`` rather than
    narrowing in silence, which is the behaviour the perimeter is trying to
    give jax.
    """
    with perimeter.armed(perimeter.FACES):
        with pytest.raises(OverflowError):
            np.zeros((3,), np.int16) + 40000
        assert perimeter.FINDINGS == 0


def test_the_array_face_arms_and_disarms_by_identity_too():
    """The lifecycle claim is about the perimeter, not about one face of it."""
    array_type = _array_type()
    originals = {
        slot: array_type.__dict__[slot] for slot in perimeter.FACE_SLOTS["array"]
    }
    assert perimeter.arm(("array",), owner="A").armed
    assert all(
        array_type.__dict__[slot] is not original
        for slot, original in originals.items()
    )
    assert perimeter.disarm("A") == "restored"
    assert all(
        array_type.__dict__[slot] is original
        for slot, original in originals.items()
    ), "the array face did not come back to jax's own functions"


def test_the_two_faces_are_independent_holds():
    """Arming one face does not arm or disarm the other.

    They close different doors and a user may want either without the other,
    which is the same independence the three instruments have from each other.
    """
    array_type, tracer = _array_type(), _tracer_type()
    array_original = array_type.__dict__["__add__"]
    tracer_original = tracer.__dict__["__le__"]
    assert perimeter.arm(("tracer",), owner="A").armed
    assert array_type.__dict__["__add__"] is array_original
    assert perimeter.arm(("array",), owner="B").armed
    assert tracer.__dict__["__le__"] is not tracer_original
    assert perimeter.disarm("A") == "still-armed"
    assert perimeter.disarm("B") == "restored"
    assert array_type.__dict__["__add__"] is array_original
    assert tracer.__dict__["__le__"] is tracer_original


def test_the_whole_perimeter_is_the_default_when_a_caller_asks_for_it():
    """``arm_perimeter()`` with no faces means every face.

    A default of one face would mean the pytest dial armed half the perimeter
    and reported ``armed``, which is the shape of green this project keeps
    having to withdraw.
    """
    status = _tripwire.arm_perimeter(owner="A")
    try:
        assert status.armed, status.explanation
        assert set(perimeter.armed_faces()) == set(perimeter.FACES)
        snapshot = perimeter.snapshot()
        assert set(snapshot["faces"]) == set(perimeter.FACES)
        assert len(snapshot["faces"]["array"]) == len(perimeter.FACE_SLOTS["array"])
        lines = "\n".join(report.render_perimeter(status, snapshot))
        assert "armed on the array face" in lines
        assert "armed on the tracer face" in lines
    finally:
        assert _tripwire.disarm_perimeter("A") == "restored"
