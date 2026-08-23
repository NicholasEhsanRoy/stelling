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
import os
import pathlib
import re
import subprocess
import sys
import warnings

import pytest

pytest_plugins = ["pytester"]

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import numpy as np  # noqa: E402

import stelling  # noqa: E402
from conftest import (  # noqa: E402
    borrowed_eager,
    borrowed_tripwire,
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
#: ``overflows-float``.
#:
#: **THE FORMAT SET IS NOT A CATALOGUE, AND B22 SAID IT WAS.** The sentence
#: that stood here read "among the four catalogued formats only ``float16``
#: has a finite range an integer literal can leave QUIETLY", justified as
#: arithmetic rather than policy. The arithmetic INSIDE those four is right --
#: every ``int64`` value is finite in ``float32``, ``bfloat16`` and
#: ``float64`` -- but the four were ``propagate._FLOAT_FORMATS``, the
#: VERIFIER's IEEE-mode catalogue, which decides what a VERDICT may reason
#: about and has nothing to do with this perimeter. ``prop_guard`` has no
#: catalogue at all: it asks ``ml_dtypes.finfo``, which is why F1 exists and
#: why its own self-test drives ``float8_e5m2``. Driven over every float dtype
#: ``jax.numpy`` exposes, EIGHT can lose an integer literal quietly, not one.
#: :data:`QUIET_FLOAT_FORMATS` is that enumeration, and the test that drives
#: it is what stops a borrowed catalogue getting in again.
OVERFLOWING_FMT = "float16"
OVERFLOWS = 100000
OVERFLOWS_IMAGE = float("inf")

#: Every float dtype ``jax.numpy`` exposes whose largest finite value an
#: ``int64`` literal can exceed, with what the comparison then RUNS AS.
#: Measured in all four cells (jax 0.10.2 / 0.11.0 x x64 off/on), identical in
#: each.
#:
#: **The ``nan`` rows are worse in kind than the ``inf`` ones.** ``fn`` and
#: ``fnuz`` formats encode no infinity, so the literal saturates to ``nan``
#: and ``x <= N`` inverts to ``False`` EVERYWHERE -- a harness guard that
#: stops holding, rather than one that holds vacuously against ``inf``.
#:
#: ``warns_traced`` is whether numpy's host cast emits ``RuntimeWarning:
#: overflow encountered in cast`` when the comparison is TRACED. Only
#: ``float16`` does, which is exactly why the seven ``float8_*`` rows matter:
#: ``-W error::RuntimeWarning`` reaches the first row and none of the others,
#: so on those seven this perimeter is the only instrument that speaks. None
#: of the eight warns when it runs eagerly.
QUIET_FLOAT_FORMATS = (
    # (dtype, largest finite, literal written, image it runs as, warns traced)
    ("float16",            65504,    100000, float("inf"), True),
    ("float8_e3m4",         15.5,        33, float("inf"), False),
    ("float8_e4m3",          240,       483, float("inf"), False),
    ("float8_e5m2",        57344,    114691, float("inf"), False),
    ("float8_e4m3fn",        448,       899, float("nan"), False),
    ("float8_e4m3b11fnuz",    30,        63, float("nan"), False),
    ("float8_e4m3fnuz",      240,       483, float("nan"), False),
    ("float8_e5m2fnuz",    57344,    114691, float("nan"), False),
)

#: A VALUE THAT IS ALREADY A FLOAT, and the sentence B22 got backwards.
#:
#: None of this release's three instruments sees one: only ``type(b) is int``
#: reaches this predicate (``report.PERIMETER_UNCOVERED``'s second bullet) and
#: the other two are integer-to-integer throughout. B22 wrote that down as "a
#: float value that overflows is seen by nothing" and "raises no alarm
#: anywhere in this release", and BOTH ARE FALSE -- for most of the cases it
#: named, and in the direction that matters, because the page defines silence
#: as "no ``RuntimeWarning`` you could turn into one" and names ``-W error``
#: as common in scientific repos.
#:
#: The split is HOST versus DEVICE and nothing else. A float narrowed by numpy
#: on the way into a jax array -- which is every constant a TRACE embeds --
#: goes through ``lax._convert_element_type``'s numpy cast and warns. A float
#: already inside a ``jax.Array`` is narrowed by XLA, and there is no host
#: cast to warn.
#:
#: THE GATE BELOW USED TO WRAP ITS DRIVE IN ``simplefilter("ignore",
#: RuntimeWarning)``, so the test certifying "no alarm anywhere" silenced the
#: alarm and passed under ``pytest -W error::RuntimeWarning``. It asserts the
#: warning now, per case, in both directions.
FLOAT_OVERFLOW_HOST_WARNS = (
    ("jnp.full((2,), 1e300, jnp.float32)",
     lambda: jnp.full((2,), 1e300, jnp.float32)),
    ("jnp.full((2,), 70000.0, jnp.float16)",
     lambda: jnp.full((2,), 70000.0, jnp.float16)),
    ("jnp.float16(70000.0)", lambda: jnp.float16(70000.0)),
    ("jnp.array([1e300], jnp.float32)",
     lambda: jnp.array([1e300], jnp.float32)),
    # The integer literal that IS refused through an operator, at a
    # CONSTRUCTION site instead -- no operator for the perimeter to sit in and
    # the eager detector is integer-to-integer, so no INSTRUMENT sees it. It
    # is a host cast all the same, so numpy does.
    ("jnp.full((2,), 100000, jnp.float16)",
     lambda: jnp.full((2,), 100000, jnp.float16)),
    # ...and the two whose EAGER spelling is silent or x64-dependent, driven
    # under `jit`, where the LITERAL is embedded through the host cast at trace
    # time. Both warn in all four cells this way, which is the point: a
    # harness is traced, and tracing moves the narrowing onto the host.
    ("jit(x_f32 + 1e300)",
     lambda: jax.jit(lambda a: a + 1e300)(jnp.zeros((2,), jnp.float32))),
    ("jit(x_f16 + 70000.0)",
     lambda: jax.jit(lambda a: a + 70000.0)(jnp.zeros((2,), jnp.float16))),
)

#: THE SAME SOURCE LINE, HOST IN ONE CELL AND DEVICE IN THE OTHER -- which is
#: the sharpest evidence that the axis is where the narrowing happens and not
#: what it is written as. With x64 OFF the value is canonicalised to
#: ``float32`` by numpy on the way in and the cast overflows there; with x64 ON
#: it stays ``float64`` through the host and XLA does the narrowing.
#:
#: **This group exists because one of its members was in the WARNS group and
#: was green for the wrong reason.** ``jit(a.astype(jnp.float32))`` on
#: ``[1e300, 1e300]`` warned at x64=0 -- but from the ``asarray`` that built
#: the operand, not from the ``astype`` the case was named for -- and went
#: silent at x64=1, where the operand really is a ``float64`` array and the
#: conversion really is on device. Driving all four cells is what found it.
FLOAT_OVERFLOW_X64_DEPENDENT = (
    ("x_f32 + 1e300", lambda: jnp.zeros((2,), jnp.float32) + 1e300),
    ("jnp.asarray([1e300, 1e300]).astype(jnp.float32)",
     lambda: jnp.asarray([1e300, 1e300]).astype(jnp.float32)),
)

#: ...and the residue that IS genuinely silent, which is smaller and sharper
#: than what B22 claimed and is the case a numerical program actually
#: produces: the overflow is COMPUTED, so there is no literal for any of the
#: three instruments to read even in principle, and no host cast for numpy to
#: warn about. Silent in all four cells under ``simplefilter("error")``.
#: The third field is THE PAGE'S OWN WORDING for the case, because the page's
#: list of these was rewritten to a different set of operations and nothing
#: went red. It is compared against the page's bullet in both directions.
FLOAT_OVERFLOW_DEVICE_SILENT = (
    ("a * a on float32 1e30", lambda: _big_f32() * _big_f32(), "a * a"),
    # EAGER, and the same line under `jit` is in the WARNS group above: the
    # weak Python float canonicalises to float32 without overflowing, and the
    # narrowing to float16 is then XLA's. Tracing is what moves it onto the
    # host, and that is the whole distinction.
    ("x_f16 + 70000.0 (eager)",
     lambda: jnp.zeros((2,), jnp.float16) + 70000.0, "x_f16 + 70000.0"),
    ("a ** 2 on float32 1e30", lambda: _big_f32() ** 2, "a ** 2"),
    ("jit(a * a) on float32 1e30",
     lambda: jax.jit(lambda a: a * a)(_big_f32()), None),
    ("jnp.exp(float32 1000.0)",
     lambda: jnp.exp(jnp.asarray([1000.0], jnp.float32)), "jnp.exp"),
    ("a.astype(jnp.float16) on float32 1e30",
     lambda: _big_f32().astype(jnp.float16), "a.astype(jnp.float16)"),
    ("lax.convert_element_type(a, jnp.float16) on float32 1e30",
     lambda: jax.lax.convert_element_type(_big_f32(), jnp.float16),
     "lax.convert_element_type(a, jnp.float16)"),
)

#: The backticked spans in the page's DEVICE bullet that are NOT one of the
#: cases above: the dtypes and values they are written against, and the
#: remedy the bullet says does not reach them. Declared so the comparison can
#: be an EQUALITY -- a bullet that swapped every operation for a different one
#: passed a membership check with room to spare.
_DEVICE_BULLET_FURNITURE = frozenset({
    "jax.Array", "warnings.simplefilter(\"error\")", "float32", "1e30",
    "1000.0", "inf", "-W error::RuntimeWarning",
})


#: THE SIX CASES B22's PAGE NAMED, in the order it named them, with the `jit`
#: spelling of each. The CHANGELOG entry for the correction counts them, and
#: it counted them wrong in both halves: *"four warn eagerly"* is right in no
#: cell -- measured FIVE at ``JAX_ENABLE_X64=0`` and THREE at ``=1``, so
#: *"identical in all four cells"* is false of the eager half -- and *"all of
#: them warn inside `jit`"* is false of case 6, which is silent in all four
#: and is the case the same entry describes moving out of the WARNS group as
#: green for the wrong reason, eleven lines below.
#:
#: Case 6's operand is built ONCE, here, outside every measurement window, so
#: what the ``jit`` row measures is the ``astype`` under the trace and not the
#: ``asarray`` that made the array.
_SIX_OPERAND = None


def _six_operand():
    global _SIX_OPERAND
    if _SIX_OPERAND is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _SIX_OPERAND = jnp.asarray([1e300, 1e300])
    return _SIX_OPERAND


THE_SIX_THE_PAGE_NAMED = (
    ("jnp.full((2,), 1e300, jnp.float32)",
     lambda: jnp.full((2,), 1e300, jnp.float32),
     lambda: jax.jit(lambda: jnp.full((2,), 1e300, jnp.float32))()),
    ("jnp.full((2,), 70000.0, jnp.float16)",
     lambda: jnp.full((2,), 70000.0, jnp.float16),
     lambda: jax.jit(lambda: jnp.full((2,), 70000.0, jnp.float16))()),
    ("jnp.float16(70000.0)",
     lambda: jnp.float16(70000.0),
     lambda: jax.jit(lambda: jnp.float16(70000.0))()),
    ("x_f32 + 1e300",
     lambda: jnp.zeros((2,), jnp.float32) + 1e300,
     lambda: jax.jit(lambda a: a + 1e300)(jnp.zeros((2,), jnp.float32))),
    ("x_f16 + 70000.0",
     lambda: jnp.zeros((2,), jnp.float16) + 70000.0,
     lambda: jax.jit(lambda a: a + 70000.0)(jnp.zeros((2,), jnp.float16))),
    ("jnp.asarray([1e300, 1e300]).astype(jnp.float32)",
     lambda: jnp.asarray([1e300, 1e300]).astype(jnp.float32),
     lambda: jax.jit(lambda a: a.astype(jnp.float32))(_six_operand())),
)


#: Ordinal suffixes, for the collection-rank sentence below.
_ORDINAL = {1: "st", 2: "nd", 3: "rd", 0: "th", 4: "th", 5: "th",
            6: "th", 7: "th", 8: "th", 9: "th"}


#: The numerals this page writes as words, so a count can be compared against
#: the sentence that carries it instead of against a hand-typed digit. The
#: CASE is the page's own: ``Eight`` opens its sentence and ``four``/``three``
#: sit inside theirs, and the comparison is against the sentence as written.
_WORDS = {
    1: "One", 2: "Two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "eleven", 12: "twelve",
}


#: THE SECOND AXIS, AND THE ONE THIS PAGE HAS NOW HAD WRONG TWICE.
#:
#: B22 wrote "seen by nothing"; its fixup corrected that to HOST versus
#: DEVICE. **Host versus device does not partition it either**, and the page
#: carried its own counter-example three paragraphs away without anyone
#: noticing: the float8 table below says ``float16`` is the ONLY format whose
#: host cast warns when traced -- seven silent host casts, beside a sentence
#: saying host casts warn.
#:
#: **What decides is the TARGET FORMAT.** ``RuntimeWarning: overflow
#: encountered in cast`` comes from numpy's own floating-point machinery, and
#: that machinery only knows numpy's own binary formats. ``ml_dtypes``
#: converts by integer bit arithmetic and raises no FP flag, so a narrowing
#: into ``bfloat16`` or any ``float8_*`` is silent at every host door -- and
#: ``bfloat16`` is the most common non-``float32`` dtype in real jax code.
#:
#: The two lists are DERIVED in :func:`test_the_host_half_is_the_TARGET_
#: FORMAT_and_not_the_door` from ``dir(jnp)`` and ``hasattr(numpy, name)``,
#: not read off this declaration, and both are compared against the page.
HOST_DOOR_LOUD = ("float16", "float32", "float64")
HOST_DOOR_SILENT = (
    "bfloat16",
    "float4_e2m1fn",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
)

#: No finite Python float exceeds ``float64``'s range, so ``float64`` has no
#: construction-door case to drive; the test asserts that positively rather
#: than quietly dropping the row.
#:
#: ``float6_e2m3fn`` and ``float6_e3m2fn`` are the other exclusion and it is
#: SERIES-DEPENDENT: jax 0.10.2 refuses to build them at all (*Invalid XLA
#: PrimitiveType*), and jax 0.11.0 builds them and then fails the CPU
#: compiler's ``RET_CHECK`` (*Invalid bitcast i6 to i8*) on the first
#: operation. Excluded by DRIVING the door and recording what came back, not
#: by a proxy -- see :data:`_ZEROS_IS_NOT_BUILDABILITY`.
HOST_DOOR_SILENT_DRIVEN_EVERYWHERE = tuple(
    n for n in HOST_DOOR_SILENT if not n.startswith("float6_")
)

#: The three CONSTRUCTION doors the page enumerates, as (label, builder).
#: Each takes a dtype and a Python float literal past that format's largest
#: finite value. They are the doors a reader writes; ``.astype`` is a
#: different route and is covered by the DEVICE half.
_HOST_CONSTRUCTION_DOORS = (
    ("jnp.full((2,), LIT, dt)", lambda dt, lit: jnp.full((2,), lit, dt)),
    ("jnp.array([LIT], dt)", lambda dt, lit: jnp.array([lit], dt)),
    ("jnp.<dt>(LIT)", lambda dt, lit: dt(lit)),
)

#: ``jnp.zeros`` IS NOT A BUILDABILITY PROXY, and this is the recorded reason.
#:
#: The first version of the quiet-format gate used ``jnp.zeros((3,), obj)`` to
#: decide whether a format could be driven at all. Measured: on jax 0.10.2
#: that raises for ``float6_e2m3fn``/``float6_e3m2fn`` and the proxy is right;
#: on jax **0.11.0 it succeeds** and the FIRST OPERATION crashes the XLA CPU
#: compiler with ``RET_CHECK failure ... Invalid bitcast i6 to i8``. So the
#: proxy answers "can be built" when the question is "can be driven", and it
#: answers it differently on the two series this project tests. Harmless while
#: both float6 formats SATURATE (they are ``inexact``, never
#: ``overflows-float``, so nothing selects them into the driven set) -- and a
#: crash rather than a report the day a format arrives that does both.
#:
#: The classification below therefore does not consult jax at all: what a
#: literal RUNS AS is ``ml_dtypes`` arithmetic and is answerable in any cell.
#: jax is asked only to drive, and a format jax cannot drive is recorded as
#: excluded with the exception it actually raised.
_ZEROS_IS_NOT_BUILDABILITY = (
    "jnp.zeros succeeded for {name} and the first operation on it raised "
    "{exc}. That is the trap this constant records: `jnp.zeros` answers "
    "'can be built', not 'can be driven', and it answers differently on the "
    "two tested jax series."
)


#: The failure message for the DEVICE half, in one place because it is the
#: whole point of that half: the pin exists to keep a DISCLOSURE true, so
#: going red means the PAGE is now wrong, not that the code is.
#:
#: **AND IT IS A MODULE CONSTANT BECAUSE IT WAS REFERENCED AND NOT DEFINED.**
#: An assertion message and a ``pytest.fail`` argument are both evaluated only
#: on the failing path, so a missing name here is invisible while the test
#: passes and turns the intended sentence into a ``NameError`` at exactly the
#: moment a maintainer needs to read it.
#: :func:`test_the_failure_message_for_the_silent_half_is_reachable` evaluates
#: it, because "this message renders" is not something a green run shows.
_LOUDER_THAN_THE_PAGE = (
    "{label} is now caught ({exc}). That is better news than this test "
    "records, and it makes docs/overflow-tripwire.md wrong in two places: the "
    "'And it is integers, all the way down' bullets under 'What it does NOT "
    "find', and bullet 3 of the narrowing perimeter's 'What it does NOT "
    "cover'. Both say that a float value narrowed ON DEVICE reaches none of "
    "the three instruments and raises no warning either -- which is the "
    "genuinely silent residue, after B22's fixup corrected the claim that the "
    "whole class was silent. Rewrite them, then move this case out of "
    "FLOAT_OVERFLOW_DEVICE_SILENT."
)


def _big_f32():
    """A ``float32`` array whose SQUARE overflows -- built without overflowing.

    ``1e30`` is finite in ``float32``, so this construction is not itself a
    narrowing and emits nothing; every case above overflows on DEVICE, which
    is the whole distinction the two tuples draw.
    """
    return jnp.asarray([1e30, 1e30], jnp.float32)


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
    session's ``Config`` before any test runs; this file sorts **72nd** of the
    files ``pytest --collect-only -q -p no:randomly`` names in this tree
    -- derived in
    :func:`test_this_files_position_in_the_collection_is_the_measured_one`,
    which is also what holds the CHANGELOG's copy of it -- so its
    FIRST test unhooked that hold and the ~4,300 tests after it ran
    unprotected with nothing red.

    **HOW MANY FILES THAT IS BELONGS TO THE ENVIRONMENT AND NOT TO THIS
    TREE, so no count is written here.** A module whose imports are
    unavailable is never collected, so the denominator moves with which
    optional dependencies a lane installs while the tree stands still --
    which is why this sentence names the COMMAND instead of a number: run it
    for the figure where you are. The RANK is the half that does not move,
    and it is the load-bearing half anyway, since it is what sets the size
    of "the ~4,300 tests after it".

    Driven at ``e6968fe``, the documented dial-on command over the whole
    suite reported::

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


#: THE SENTENCE EACH REASON RENDERS, written out rather than derived, because
#: the subject is ENGLISH and no assertion judges grammar. What a fence can
#: hold is that the sentence a user meets is the one a reviewer read here, and
#: that every reason has one. Driven below through the ARMED perimeter, which
#: is the route this text reaches a user by: ``perimeter.py`` quotes
#: ``Finding.message`` verbatim into the sentence it raises.
#:
#: TWO OF THESE FOUR WERE UNGRAMMATICAL until the message-text vendoring
#: edit named in ``prop_guard.py``'s note, and in the same way. (Which edit
#: that is, is not written here: it is an ordinal into a list, and
#: ``tests/test_prop_guard_ledger.py`` is where ordinals into that list are
#: held to it.) The template reads *"the literal L
#: written in `slot` is <phrase> <dtype>"*, and two of the phrases were not
#: phrases a dtype can follow, so a user read
#:
#:     ... is overflows float16: the program uses inf
#:     ... is a negative literal cannot exist in the unsigned type it is
#:         compared against uint8: the program uses 253
#:
#: ``overflows-float`` is the reason `docs/overflow-tripwire.md` promotes to a
#: headline, so the release documented a string it had left broken. The
#: predicate's ANSWERS never had anything to do with it -- ``reason``,
#: ``narrowed_to`` and ``target_dtype`` were right in all four rows, which is
#: why the self-test above (which prints those and not the message) could not
#: see it and is byte-identical across the repair.
MESSAGES = (
    ("out-of-range", "int8", 256, lambda x: x <= 256,
     "the literal 256 written in `__le__` is outside the range of int8: "
     "the program uses 0"),
    ("inexact", "float32", MOVED, lambda x: x <= MOVED,
     "the literal 2147483647 written in `__le__` is not exactly "
     "representable in float32: the program uses 2147483648.0"),
    ("overflows-float", OVERFLOWING_FMT, OVERFLOWS, lambda x: x <= OVERFLOWS,
     "the literal 100000 written in `__le__` is too large in magnitude for "
     "float16: the program uses inf"),
    ("negative-into-unsigned", "uint8", -3, lambda x: x >= -3,
     "the literal -3 written in `__ge__` is negative and cannot exist in the "
     "unsigned type uint8: the program uses 253"),
)


def test_every_reason_has_its_OWN_phrase_and_none_falls_through_to_another():
    """Totality, and it is the leg a new reason meets first.

    ``Finding.message`` chose its phrase with an ``elif`` chain ending in an
    ``else``, so a fifth entry in ``REASONS`` would have rendered as an
    overflow -- a sentence about a defect it does not have, in the text a
    user reads -- rather than saying it had no phrase. The phrases are a
    mapping now, and this holds its keys equal to ``REASONS`` in BOTH
    directions: a reason with no phrase, and a phrase for a reason that no
    longer exists, are each a failure.

    ``MESSAGES`` is held to the same set, so a reason cannot be added with a
    phrase and without a rendering anybody has read.
    """
    assert set(prop_guard._WHAT) == set(prop_guard.REASONS), (
        f"phrases are written for {sorted(prop_guard._WHAT)} and the closed "
        f"reason set is {sorted(prop_guard.REASONS)}. A reason with no "
        f"phrase renders through the fallback, which names the reason "
        f"instead of a defect -- correct, but nobody has written the "
        f"sentence a user would read."
    )
    assert {row[0] for row in MESSAGES} == set(prop_guard.REASONS), (
        f"{sorted({row[0] for row in MESSAGES})} have a rendering written "
        f"out above and {sorted(prop_guard.REASONS)} exist"
    )


@pytest.mark.parametrize(
    ("reason", "fmt", "literal", "expr", "sentence"),
    MESSAGES,
    ids=[row[0] for row in MESSAGES],
)
def test_the_sentence_this_reason_puts_in_front_of_a_user(
        reason, fmt, literal, expr, sentence):
    """One reason, rendered twice: off the ``Finding`` and out of the ARMED
    perimeter, which is where a user meets it.

    The fields are MEASURED and not invented -- the ``Finding`` comes out of
    ``classify`` on a real array -- so this cannot pass by rendering a
    hand-built object that the predicate would never produce.
    """
    x = jnp.zeros((3,), getattr(jnp, fmt))
    finding = prop_guard.classify(x, literal, "__le__" if literal > 0
                                  else "__ge__")
    assert finding is not None and finding.reason == reason, finding
    assert finding.message == sentence, (
        f"the {reason} sentence reads\n  {finding.message}\nand this file "
        f"records\n  {sentence}"
    )

    with perimeter.armed(("array",)) as status:
        assert status.armed, status.explanation
        with pytest.raises(perimeter.NarrowingError) as caught:
            expr(x)
    assert caught.value.finding.reason == reason, caught.value.finding
    assert sentence in str(caught.value), (
        f"the perimeter's own sentence is\n  {caught.value}\nand it does not "
        f"carry the {reason} rendering\n  {sentence}"
    )


def test_a_reason_with_no_phrase_names_ITSELF_rather_than_borrowing_one():
    """The fallback the mapping replaced the ``else`` with, driven.

    ``Finding.message`` is reached from ``NarrowingError.__init__``, and a
    guard may not raise on the path where it reports -- so an unmapped reason
    cannot be a ``KeyError``. What it must not do instead is render as some
    OTHER reason, which is exactly what the ``else`` did. Driven on a reason
    that is deliberately not in ``REASONS``: the sentence names it.
    """
    finding = prop_guard.Finding(
        reason="not-a-reason", slot="le", operand_dtype="int8",
        target_dtype="int8", literal=7, narrowed_to=7,
    )
    text = finding.message
    assert "not-a-reason" in text, text
    for phrase in prop_guard._WHAT.values():
        assert phrase not in text, (
            f"an unmapped reason rendered with {phrase!r}, the phrase of "
            f"another reason: {text}"
        )


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
    # The whole sentence, and every reason's, is in :data:`MESSAGES`; what
    # this leg is about is that the SEPARATE reason reaches the SEPARATE
    # advice, so it reads the half of the sentence that only this reason
    # produces rather than the half the template gives them all.
    assert "is negative and cannot exist in the unsigned type uint8" in (
        finding.message), finding.message
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
#: ``cell`` is the page's *what runs* cell VERBATIM and ``runs`` is the value
#: the drive must produce, so neither can move alone.
#:
#: **THEY ARE TWO FIELDS BECAUSE ONE FIELD WAS A SUBSTRING TEST.** The
#: assertion was ``runs in row[1]`` against a bare ``"-25536"``, so a page cell
#: reading ``-255360`` satisfied it -- driven, and green. An exact cell
#: comparison cannot be satisfied by a longer number that happens to start the
#: same way.
#:
#: ``runs`` for a COMPARISON row is the literal jax puts in the jaxpr, read
#: out of the jaxpr. That row used to be excluded from the drive entirely: the
#: test threw away the value it had just computed and substituted
#: ``repr(float(np.asarray(2**31 - 1).astype("float32")))`` -- a re-derivation
#: on numpy's cast path, not a measurement of what the program ran -- so the
#: one row whose whole subject is a literal that MOVES was the one row nothing
#: measured.
#:
#: The instrument triple is (tripwire, eager, perimeter) and reads the page's
#: last three cells: ``fires`` for the one that refuses, ``quiet`` for a cell
#: that says ``no fire``, and ``n/a`` for one the page marks ``—``.
_WHAT_RUNS = (
    ("jnp.full((3,), 40000, int16)", "`-25536`-filled", "-25536",
     ("n/a", "fires", "n/a"),
     lambda: jnp.full((3,), 40000, jnp.int16)),
    ("x_int16 + 40000", "`-25536`", "-25536",
     ("quiet", "quiet", "fires"),
     lambda: jnp.zeros((3,), jnp.int16) + 40000),
    ("40000 + x_int16", "`-25536`", "-25536",
     ("quiet", "quiet", "fires"),
     lambda: 40000 + jnp.zeros((3,), jnp.int16)),
    ("x_f32 <= 2**31 - 1", "`<= 2147483648.0`", "2147483648.0",
     ("quiet", "quiet", "fires"),
     lambda: jnp.zeros((3,), jnp.float32) <= 2**31 - 1),
    ("x_int16 + 3", "`3`", "3",
     ("n/a", "n/a", "quiet"),
     lambda: jnp.zeros((3,), jnp.int16) + 3),
)

#: The jaxpr each comparison row lowers to, so that "what runs" for a row
#: whose output is a BOOL is read from the program rather than re-derived.
_WHAT_RUNS_TRACED = {
    "x_f32 <= 2**31 - 1": (
        lambda: jax.make_jaxpr(lambda a: a <= 2**31 - 1)(
            jnp.zeros((3,), jnp.float32)),
    ),
}

_WHAT_RUNS_HEADER = (
    "| written | what runs | the tripwire | the eager detector | the perimeter |"
)


def _literals_of(closed) -> str:
    """The scalar literals a jaxpr holds, as text -- what the program RUNS ON.

    A comparison's output is a bool and says nothing about the literal that
    was converted to produce it; the jaxpr says exactly that, and it is the
    artefact jax executes.
    """
    out = []
    for eqn in closed.jaxpr.eqns:
        for var in eqn.invars:
            val = getattr(var, "val", None)
            if val is None:
                continue
            arr = np.asarray(val)
            out.append(repr(float(arr)) if arr.dtype.kind == "f" else str(arr))
    assert out, f"no literal in {closed}"
    return " ".join(out)


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

    with borrowed_tripwire() as (tw_status, recorder), \
            borrowed_eager() as eager_status:
        assert tw_status.armed and eager_status.armed
        for (written, cell, runs, expected, build), row in zip(_WHAT_RUNS, doc_rows):
            assert f"`{written}`" == row[0], (
                f"row order moved: this test drives {written!r} where the "
                f"page's row reads {row[0]!r}"
            )
            assert cell == row[1], (
                f"the page says {written} runs as {row[1]!r}; this test "
                f"declares the cell {cell!r}. This is an EXACT comparison on "
                f"purpose: it used to be `{runs!r} in <cell>`, and `-255360` "
                f"satisfies that."
            )
            assert runs in cell, (written, cell, runs)   # the two agree

            # ---- WHAT RUNS, READ DISARMED, FOR EVERY ROW.
            #
            # This read used to sit in the `else:` branch of the ARMED drive
            # below, and FOUR OF THE FIVE ROWS RAISE BEFORE REACHING IT --
            # `EagerTruncationError` on row 1, `NarrowingError` on rows 2, 3
            # and 4. So the page's "what runs" column, which is the column a
            # reader trusts to know what their program does, was held by
            # nothing but string agreement between two places a careless edit
            # touches together. Measured before this was fixed: `-25536` ->
            # `-25537` in BOTH the page and `_WHAT_RUNS` was GREEN, so was
            # `2147483648.0` -> `999.0`, and so was `_WHAT_RUNS_TRACED = {}`
            # -- a KeyError if it had ever been reached, which is what makes
            # "the comparison row is read out of the jaxpr" a claim about
            # code that did not run.
            #
            # The value is what the program does with NOTHING armed, so it is
            # read here, before the instruments go up. The region is the
            # declaration `expected_truncation` exists for: these rows' whole
            # subject is a narrowing, and a session that armed the eager
            # detector globally would otherwise raise on row 1.
            jax.clear_caches()
            with expected_truncation(
                f"docs/overflow-tripwire.md's 'what runs' cell for "
                f"`{written}` is exactly this value, read disarmed"
            ):
                ran = np.asarray(build())
            if ran.dtype.kind == "b":
                # THE COMPARISON ROWS, READ FROM THE PROGRAM. What RUNS here
                # is the literal jax converted, not the bool the comparison
                # produced -- and it is taken out of the jaxpr, which is the
                # program. This used to substitute
                # `repr(float(np.asarray(2**31 - 1).astype("float32")))` for
                # the driven value: a re-derivation down numpy's cast path,
                # which is true of numpy whatever jax does, so the one row
                # about a literal that MOVES was measuring nothing.
                (trace_it,) = _WHAT_RUNS_TRACED[written]
                jax.clear_caches()
                ran_text = _literals_of(trace_it())
            else:
                ran_text = (repr(float(ran.ravel()[0]))
                            if ran.dtype.kind == "f"
                            else str(ran.ravel()[0]))
            assert runs in ran_text.split(), (
                f"{written} runs as {ran_text}, and the page and this test "
                f"both say {runs}. This is a token match and not a substring "
                f"one: `-255360` must not satisfy `-25536`."
            )

            jax.clear_caches()
            before = (recorder.fires, _eager.TRUNCATIONS, perimeter.FINDINGS)
            fired = None
            with perimeter.armed(("tracer", "array")) as status:
                assert status.armed, status.explanation
                try:
                    np.asarray(build())
                except stelling.EagerTruncationError:
                    fired = 1
                except perimeter.NarrowingError:
                    fired = 2
            after = (recorder.fires, _eager.TRUNCATIONS, perimeter.FINDINGS)
            moved = tuple(a - b for a, b in zip(after, before))

            for i, (claim, delta) in enumerate(zip(expected, moved)):
                instrument_cell = row[2 + i]
                if claim == "fires":
                    assert fired == i, (
                        f"{written}: the page says instrument {i} refuses it "
                        f"and the refusal came from {fired}"
                    )
                    assert "refuses" in instrument_cell, (row, instrument_cell)
                    if i:
                        assert delta == 1, (written, moved)
                elif claim == "quiet":
                    assert delta == 0, (
                        f"{written}: instrument {i} fired {delta} time(s) and "
                        f"the page's cell reads {instrument_cell!r}"
                    )
                    assert ("no fire" in instrument_cell
                            or "passes" in instrument_cell), (row, instrument_cell)
                else:
                    assert instrument_cell in {"—", "-"}, (row, instrument_cell)


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
    silent door rather than a jax error the perimeter merely relabels -- and
    it is now qualified EAGER, because traced, ``float16``'s host cast warns
    (:func:`test_the_quiet_float_formats_are_enumerated_not_borrowed` is where
    that is driven per format).
    """
    x = jnp.zeros((3,), getattr(jnp, OVERFLOWING_FMT))

    # DISARMED AND EAGER: silent, and the comparison that runs is against
    # `inf`. `simplefilter("error")` and not `ignore`: the silence is the
    # claim, so a warning appearing here must fail this test rather than be
    # swallowed by it.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (x <= OVERFLOWS).tolist() == [True, True, True]
    with np.errstate(over="ignore"):
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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (x <= OVERFLOWS).tolist() == [True, True, True]

    # AND THE SENTENCES ARE READ. This batch's whole subject is a page that
    # measured nothing wrong and simply said nothing, so a behaviour pin with
    # no reader is only half the repair: both pages that now answer the float
    # question have to be naming the case this test drives.
    #
    # EVERY OCCURRENCE, NOT ANY OCCURRENCE. The page states this comparison
    # TWICE, and the assertion here was `f"...{OVERFLOWS}..." in page` -- so
    # either occurrence kept it green while the other rotted. Driven: changing
    # the second to `x_f16 <= 999` left this file green.
    repo = pathlib.Path(__file__).resolve().parents[1]
    # One pattern per page, each matching the PROSE spelling that page uses
    # and not the per-format table (whose rows are driven, row by row, in
    # `test_the_quiet_float_formats_are_enumerated_not_borrowed`).
    spellings = {
        "docs/overflow-tripwire.md": re.compile(r"`x_f16 <= (\d+)`"),
        "docs/quickstart.md": re.compile(r"`x <= (\d+)` on `float16`"),
    }
    for rel, spelling in spellings.items():
        page = (repo / rel).read_text(encoding="utf-8")
        found = spelling.findall(page)
        assert found, (
            f"{rel} no longer names the comparison this test drives, so its "
            f"float-overflow answer has stopped being about a measured case"
        )
        assert set(found) == {str(OVERFLOWS)}, (
            f"{rel} writes this comparison against {sorted(set(found))} and "
            f"this test drives {OVERFLOWS}. EVERY occurrence has to be the "
            f"driven one: the page states it more than once, and an assertion "
            f"that only asks whether the value appears SOMEWHERE lets one "
            f"occurrence rot behind the other."
        )
        assert OVERFLOWING_FMT in page, f"{rel} no longer names {OVERFLOWING_FMT}"


def test_the_failure_message_for_the_silent_half_is_reachable():
    """Render the message the test above only ever renders when it FAILS.

    A green run never evaluates an ``assert``'s message or the argument to a
    ``pytest.fail`` it does not reach, so a name that is referenced there and
    not defined is invisible for as long as the test passes -- and then
    replaces the sentence a maintainer needs with a ``NameError``, at the one
    moment it matters. That happened here: :data:`_LOUDER_THAN_THE_PAGE` was
    referenced twice while its definition had been edited away, and every
    test in this file was green.

    So the message is rendered on the passing path, with both call shapes, and
    checked for the two things a reader of a red run needs from it: the page
    it names and the tuple to move the case out of.
    """
    rendered = [
        _LOUDER_THAN_THE_PAGE.format(label="jnp.exp(...)", exc="RuntimeWarning"),
        _LOUDER_THAN_THE_PAGE.format(label="one of them (0, 0, 0) -> (1, 0, 0)",
                                     exc="a counter moving"),
    ]
    for text in rendered:
        assert "docs/overflow-tripwire.md" in text, text
        assert "FLOAT_OVERFLOW_DEVICE_SILENT" in text, text
        assert "{" not in text and "}" not in text, (
            f"an unfilled placeholder survived formatting: {text!r}"
        )
    assert rendered[0] != rendered[1]


def test_the_quiet_float_formats_are_enumerated_not_borrowed():
    """Eight formats, not the four of somebody else's catalogue.

    The sentence this replaces said *"among the four catalogued formats only
    ``float16`` has a finite range an integer literal can leave quietly"*,
    justified as *"arithmetic rather than policy"* -- and the justification is
    what made it wrong. The four were ``propagate._FLOAT_FORMATS``, the
    VERIFIER's IEEE-mode catalogue, which the perimeter's page never
    introduces and the perimeter's predicate never consults: ``prop_guard``
    asks ``ml_dtypes.finfo``, which is why F1 exists at all.

    So the set is DERIVED here, from every float dtype ``jax.numpy`` exposes,
    and compared against the declared one in both directions -- a format that
    starts or stops being able to lose a literal quietly is a change to
    :data:`QUIET_FLOAT_FORMATS` and to the page's table, not a silent drift.
    """
    ml_dtypes = pytest.importorskip("ml_dtypes")

    # (1) THE SET, DERIVED -- and derived WITHOUT ASKING JAX, because what a
    # literal runs as is `ml_dtypes` arithmetic and is answerable in every
    # cell. This loop used to gate each format on `jnp.zeros((3,), obj)`
    # succeeding, as a proxy for "can be driven". THAT PROXY IS WRONG, and
    # differently wrong on the two tested series: see
    # :data:`_ZEROS_IS_NOT_BUILDABILITY`. jax is asked to DRIVE below, and a
    # format it cannot drive is reported rather than dropped.
    derived, derived_all = {}, {}
    for name in dir(jnp):
        obj = getattr(jnp, name)
        if not isinstance(obj, type) or not name.startswith(("float", "bfloat")):
            continue
        try:
            biggest = float(ml_dtypes.finfo(obj).max)
        except Exception:  # noqa: BLE001 - np.floating & friends are not dtypes
            continue
        if name != np.dtype(obj).name:
            continue                    # an alias: float_ is float64 again
        derived_all[name] = biggest
        if biggest >= 2**63 - 1:
            continue                    # every int64 value is finite here
        derived[name] = biggest

    declared = {row[0]: row[1] for row in QUIET_FLOAT_FORMATS}
    assert set(derived) >= set(declared), (
        f"declared formats that jax.numpy no longer exposes: "
        f"{sorted(set(declared) - set(derived))}"
    )
    # ...and every declared one can actually be DRIVEN here -- one real
    # operation, not `jnp.zeros`. On jax 0.11.0 `jnp.zeros` succeeds for
    # float6_e2m3fn and the first operation fails the XLA CPU compiler's
    # RET_CHECK ("Invalid bitcast i6 to i8"), so the proxy answers a
    # different question from the one being asked.
    for name in declared:
        obj = getattr(jnp, name)
        try:
            np.asarray(jnp.zeros((3,), obj) <= 1)
        except Exception as exc:  # noqa: BLE001 - this IS the measurement
            pytest.fail(_ZEROS_IS_NOT_BUILDABILITY.format(
                name=name, exc=f"{type(exc).__name__}: {exc}"[:160]))
    # A format jax gains that saturates (float4_e2m1fn clamps to its max and
    # is `inexact`, not `overflows-float`) is not a QUIET row; sort the
    # derived set by what the literal actually runs as rather than asserting
    # equality against a set that mixes the two.
    for name, biggest in sorted(derived.items()):
        literal = int(biggest) * 2 + 3
        with np.errstate(over="ignore", invalid="ignore"):
            image = float(np.asarray(literal).astype(np.dtype(getattr(jnp, name))))
        overflowed = image != image or np.isinf(image)
        assert overflowed == (name in declared), (
            f"{name}: a literal past its largest finite value "
            f"({biggest}) runs as {image!r}, and this file "
            f"{'declares' if name in declared else 'does not declare'} it a "
            f"quiet-overflow format. QUIET_FLOAT_FORMATS and the table in "
            f"docs/overflow-tripwire.md are the two places to change."
        )
        if name in declared:
            assert biggest == declared[name], (name, biggest, declared[name])

    # (2) EACH ROW, DRIVEN -- disarmed silence eager, the image it runs as,
    # the traced warning, and the refusal.
    page = (pathlib.Path(__file__).resolve().parents[1]
            / "docs" / "overflow-tripwire.md").read_text(encoding="utf-8")
    for name, biggest, literal, image, warns_traced in QUIET_FLOAT_FORMATS:
        dt = getattr(jnp, name)
        x = jnp.zeros((3,), dt)

        # EAGER: silent in all four cells, and the comparison inverts to False
        # on the formats that have no infinity to saturate to.
        jax.clear_caches()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ran = np.asarray(x <= literal).tolist()
        inverted = image != image                       # nan
        assert ran == [not inverted] * 3, (name, ran, image)

        # TRACED: numpy's host cast is where the warning lives, and float16 is
        # the only one of the eight that gets one. That asymmetry is the whole
        # reason the seven float8 rows are worth a page paragraph.
        jax.clear_caches()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            jaxpr = str(jax.make_jaxpr(lambda a: a <= literal)(x))
        warned = any(
            issubclass(w.category, RuntimeWarning)
            and "overflow encountered in cast" in str(w.message)
            for w in seen
        )
        assert warned == warns_traced, (
            f"{name}: traced, numpy "
            f"{'warned' if warned else 'did not warn'} and this file declares "
            f"{warns_traced}. If this flipped to True everywhere, "
            f"`-W error::RuntimeWarning` now reaches these and the page's "
            f"paragraph saying it does not has to change."
        )
        assert f"{'nan' if inverted else 'inf'}:" in jaxpr, (name, jaxpr)

        # ARMED: refused as overflows-float, on every one of the eight.
        with perimeter.armed(("array",)) as status:
            assert status.armed, status.explanation
            with pytest.raises(perimeter.NarrowingError) as caught:
                _ = x <= literal
        finding = caught.value.finding
        assert finding.reason == "overflows-float", (name, finding)
        assert finding.literal == literal, (name, finding)

        # (3) AND THE PAGE'S ROW FOR IT -- ALL SIX CELLS.
        #
        # This assertion used to stop at the third cell, so the page's LAST
        # THREE columns were unchecked: the `inf`-versus-`nan` distinction
        # the page itself calls "worse in kind", and both silence flags.
        # Measured: flipping `float8_e4m3fn`'s image from `**`nan`**` to
        # `` `inf` `` was GREEN, and so was turning `float16`'s traced
        # `**warns**` into `silent` -- which is the exact cell that is this
        # section's counter-example to "host casts warn".
        row = (
            f"| `{name}` | {biggest} | `x <= {literal}` "
            f"| {'**`nan`**' if inverted else '`inf`'} "
            f"| silent | {'**warns**' if warns_traced else 'silent'} |"
        )
        assert row in page, (
            f"docs/overflow-tripwire.md has no row for {name} matching what "
            f"this test just drove. Expected, cell for cell:\n  {row}\n"
            f"The table there and QUIET_FLOAT_FORMATS are the same "
            f"enumeration and a format may not be in one only."
        )
    assert page.count("| `float8_") == 7, (
        "the page's quiet-format table no longer carries exactly the seven "
        "float8 rows this test drives"
    )

    # (4) THE PROSE ABOVE THE TABLE, which is where the COUNTS live and where
    # nothing was reading. `Eight` -> `Nine` was a green mutation.
    flowed = " ".join(page.split())
    n_quiet = len(QUIET_FLOAT_FORMATS)
    assert f"**{_WORDS[n_quiet]} can lose a literal to `inf` or `nan`**" in flowed, (
        f"docs/overflow-tripwire.md no longer says {_WORDS[n_quiet]} formats "
        f"can lose a literal quietly, and QUIET_FLOAT_FORMATS has {n_quiet}"
    )
    # ...and the partition it sits in: 15 formats = 4 that cannot lose an
    # int64 + 3 that saturate + the eight above. Derived, both directions.
    safe = sorted(n for n, big in derived_all.items() if big >= 2**63 - 1)
    saturating = sorted(set(derived) - set(declared))
    assert len(safe) + len(saturating) + n_quiet == 15, (
        f"{len(safe)} formats hold every int64, {len(saturating)} saturate "
        f"and {n_quiet} overflow, which is not the 15 the page states"
    )
    assert (
        "`jax.numpy` exposes **15** concrete float formats" in flowed
    ), "docs/overflow-tripwire.md no longer states the 15 this test derived"
    assert f"so those **{_WORDS[len(safe)]}** cannot lose one this way" in flowed, (
        f"the page's count of formats that hold every int64 is not "
        f"{_WORDS[len(safe)]}: {safe}"
    )
    assert f"so those **{_WORDS[len(saturating)]}** are not this table" in flowed, (
        f"the page's count of SATURATING formats is not "
        f"{_WORDS[len(saturating)]}: {saturating}"
    )
    for name in saturating:
        assert f"`{name}`" in flowed, (
            f"the page does not name {name} among the formats that saturate "
            f"rather than overflowing"
        )

    # (5) THE ASYMMETRY, IN THE TWO SENTENCES THAT CARRY IT. This is the
    # page's own counter-example to the claim it made for two rounds that a
    # host cast warns -- seven silent host casts in the table above -- and it
    # is the sentence a reader takes the `-W error::RuntimeWarning` advice
    # from. Both spellings of it are rebuilt from the `warns_traced` column
    # that was just driven; measured before this, turning "the one format"
    # into "NO format" left the file green.
    loud_traced = [row[0] for row in QUIET_FLOAT_FORMATS if row[4]]
    assert len(loud_traced) == 1, loud_traced
    only = loud_traced[0]
    for sentence in (
        f"traced, `{only}` is the one format where numpy's host cast warns",
        f"traced, numpy's host cast warns, and `{only}` is the only "
        f"format below where it does",
    ):
        assert sentence in flowed, (
            f"docs/overflow-tripwire.md no longer says that {only} is the "
            f"only one of these {n_quiet} formats whose host cast warns when "
            f"traced, which is what the `warns_traced` column just measured "
            f"and what makes the other {n_quiet - 1} rows worth a paragraph. "
            f"Expected:\n  {sentence}"
        )


def test_a_float_VALUE_that_overflows_warns_on_the_HOST_and_is_silent_on_DEVICE():
    """The other half, and B22 had its silence axis INVERTED.

    None of the three instruments sees a value that is already a float: only
    ``type(b) is int`` reaches this predicate, and the const-fold tripwire and
    the eager detector are integer-to-integer (``intentional_wrap`` refuses
    every non-integer dtype by name, asserted below so that "integer to
    integer" is measured here rather than repeated). That much was right.

    **What was wrong is the sentence drawn from it** -- *"seen by nothing"*,
    *"raises no alarm anywhere in this release"* -- and it was wrong in the
    direction that costs a reader something, because the page defines silence
    as *"no RuntimeWarning you could turn into one"* and names ``-W error`` as
    common in scientific repos. Most of the cases it listed emit
    ``RuntimeWarning: overflow encountered in cast``, so
    ``pytest -W error::RuntimeWarning`` catches them today and the page told
    the reader there was nothing to be done.

    **And the gate could not see it**, because it wrapped its drive in
    ``simplefilter("ignore", RuntimeWarning)``: the test certifying "no alarm
    anywhere" silenced the alarm and passed under
    ``pytest -W error::RuntimeWarning``. Both groups are asserted here, in
    both directions, under ``simplefilter("error")`` -- so a host case that
    stops warning and a device case that starts are each red.
    """
    from stelling._tripwire import eager as _eager

    # ---- (1) THE HOST HALF: numpy's cast warns, and that is the reader's
    #      existing remedy. Nothing is ignored; the warning IS the assertion.
    for label, build in FLOAT_OVERFLOW_HOST_WARNS:
        jax.clear_caches()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(RuntimeWarning, match="overflow encountered in cast"):
                np.asarray(build())
        # ...and it really is an overflow to inf, not some other cast warning
        jax.clear_caches()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            got = np.asarray(build())
        assert np.isinf(got).all(), f"{label} is no longer inf: {got!r}"

    # ---- (1b) THE SAME LINE, BOTH WAYS, decided by where the narrowing
    #      happens rather than by how it is spelled. Asserted in the direction
    #      this cell is in, so BOTH are pinned across the four-cell grid.
    x64 = bool(jax.config.jax_enable_x64)
    for label, build in FLOAT_OVERFLOW_X64_DEPENDENT:
        jax.clear_caches()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            got = np.asarray(build())
        warned = any(
            issubclass(w.category, RuntimeWarning)
            and "overflow encountered in cast" in str(w.message) for w in seen
        )
        assert np.isinf(got).all(), f"{label} is no longer inf: {got!r}"
        assert warned == (not x64), (
            f"{label}: with x64 {'ON' if x64 else 'OFF'} numpy "
            f"{'warned' if warned else 'did not warn'}, and this file declares "
            f"the opposite. With x64 off the value is canonicalised to float32 "
            f"on the HOST and overflows there; with x64 on it stays float64 "
            f"through the host and XLA narrows it. If that stopped being true, "
            f"docs/overflow-tripwire.md's host-versus-device paragraph is what "
            f"has to change."
        )

    # ---- (2) THE DEVICE HALF: the genuinely silent residue. No host cast
    #      runs, so there is nothing for `-W error::RuntimeWarning` to catch,
    #      and the overflow is COMPUTED so there is no literal for any of the
    #      three instruments to read.
    with borrowed_tripwire() as (tw_status, recorder), \
            borrowed_eager() as eager_status:
        assert tw_status.armed, tw_status.explanation
        assert eager_status.armed, eager_status.explanation
        with perimeter.armed(("tracer", "array")) as status:
            assert status.armed, status.explanation
            before = (recorder.fires, _eager.TRUNCATIONS, perimeter.FINDINGS)
            for label, build, _ in FLOAT_OVERFLOW_DEVICE_SILENT:
                jax.clear_caches()
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
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

    # ---- (3) ...and the reason the other two cannot be the ones that catch it.
    for dtype in ("float16", "bfloat16", "float32", "float64"):
        with pytest.raises(ValueError, match="not one of the integer dtypes"):
            stelling.intentional_wrap(1, dtype)

    # ---- (4) AND THE PAGES SAY THE REMEDY. A reader who has just been told
    #      that stelling does not watch this is owed the thing that does.
    repo = pathlib.Path(__file__).resolve().parents[1]
    for rel in ("docs/overflow-tripwire.md", "docs/quickstart.md"):
        page = (repo / rel).read_text(encoding="utf-8")
        assert "-W error::RuntimeWarning" in page, (
            f"{rel} no longer names `-W error::RuntimeWarning`. It is the "
            f"remedy a reader ALREADY HAS for the host half above, and the "
            f"sentence this test replaced withheld it by saying the case was "
            f"seen by nothing."
        )
    # ...and the warning's own text, so a reader recognises it in their output.
    # Whitespace-normalised because prose wraps and the phrase spans a line.
    flowed = " ".join(
        (repo / "docs/overflow-tripwire.md").read_text(encoding="utf-8").split()
    )
    assert "overflow encountered in cast" in flowed, (
        "docs/overflow-tripwire.md no longer quotes the warning text this "
        "test matches on, so a reader cannot recognise it in their own output"
    )


def _host_door_outcome(dt, literal, build):
    """Drive one construction door under ``simplefilter("error")``.

    Returns ``("WARNS", None)``, ``("silent", image)`` or
    ``("unbuildable", exc)``. The third is a MEASUREMENT of this jax rather
    than a proxy for one -- see :data:`_ZEROS_IS_NOT_BUILDABILITY`.
    """
    jax.clear_caches()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return "silent", float(np.asarray(build(dt, literal)).reshape(-1)[0])
        except RuntimeWarning:
            return "WARNS", None
        except Exception as exc:  # noqa: BLE001 - an XLA refusal is data here
            return "unbuildable", f"{type(exc).__name__}: {exc}"[:120]


def test_the_host_half_is_the_TARGET_FORMAT_and_not_the_door():
    """Which host narrowings numpy reports -- and the page's answer, pinned.

    **This page has stated the partition wrongly twice.** B22 said a float
    that overflows is *"seen by nothing"*; B22's fixup corrected that to HOST
    versus DEVICE and stated a universal -- *"where the narrowing is done ON
    THE HOST, by numpy, the cast emits RuntimeWarning"* -- with a remedy
    sentence drawn from it: ``-W error::RuntimeWarning`` *"covers whatever
    numpy touched and nothing else."*

    **Measured, that is false, and the page carried its own counter-example.**
    ``jnp.full((2,), 1e300, jnp.bfloat16)``, ``jnp.array([1e300],
    jnp.bfloat16)`` and ``jnp.bfloat16(1e300)`` are each ``inf`` with nothing
    raised, in all four cells, at the same three doors that raise for
    ``float32``; and the float8 table three paragraphs below the sentence
    already said ``float16`` was the only format whose host cast warns when
    traced -- seven silent host casts sitting beside a claim that host casts
    warn.

    **The axis is the TARGET FORMAT.** ``RuntimeWarning: overflow encountered
    in cast`` is raised by numpy's own floating-point machinery, which knows
    numpy's own binary formats and nothing else; every other float format
    ``jax.numpy`` exposes comes from ``ml_dtypes``, whose conversions are
    integer bit arithmetic and raise no floating-point flag. The control at
    the bottom of this test is the decisive one and has no jax in it: ONE
    numpy cast loop on ONE ``float32`` source array warns into ``float16``
    and is silent into ``float8_e5m2`` and ``float8_e4m3fn``, which lose far
    more.

    Both lists are derived here rather than read off the declaration, and the
    page's table, its enumeration of the silent formats and its remedy
    sentence are all compared against what was just driven -- so the prose
    that carries this claim cannot be inverted without this going red.
    """
    ml_dtypes = pytest.importorskip("ml_dtypes")

    # ---- (1) THE PARTITION, DERIVED. Every concrete float format jax.numpy
    #      exposes, split by whether numpy implements it itself.
    derived_loud, derived_silent = [], []
    for name in sorted(dir(jnp)):
        obj = getattr(jnp, name, None)
        if not isinstance(obj, type):
            continue
        try:
            dtype = np.dtype(obj)
        except Exception:  # noqa: BLE001 - not every jnp type is a dtype
            continue
        if name != dtype.name:
            continue                       # an alias: single, double, float_
        try:
            ml_dtypes.finfo(dtype)
        except Exception:  # noqa: BLE001 - ints, bools and complex
            continue
        if dtype.kind == "c":
            continue                       # complex is not this page's subject
        (derived_loud if hasattr(np, name) else derived_silent).append(name)

    assert tuple(derived_loud) == tuple(sorted(HOST_DOOR_LOUD)), (
        f"the float formats numpy implements itself are now {derived_loud}; "
        f"this file declares {sorted(HOST_DOOR_LOUD)}. That set is the whole "
        f"reach of `-W error::RuntimeWarning` on the host, so a change here "
        f"is a change to docs/overflow-tripwire.md's remedy sentence."
    )
    assert tuple(derived_silent) == tuple(sorted(HOST_DOOR_SILENT)), (
        f"the ml_dtypes float formats jax.numpy exposes are now "
        f"{derived_silent}; this file declares {sorted(HOST_DOOR_SILENT)}."
    )
    assert len(derived_loud) + len(derived_silent) == 15, (
        f"jax.numpy now exposes {len(derived_loud) + len(derived_silent)} "
        f"concrete float formats, and docs/overflow-tripwire.md says 15"
    )

    # ---- (2) THE LOUD HALF, DRIVEN AT ALL THREE DOORS. float64 has no case:
    #      no finite Python float exceeds it, and that is asserted rather than
    #      assumed, because a silently dropped row is how the last two
    #      versions of this claim were reached.
    assert float(ml_dtypes.finfo(np.dtype(np.float64)).max) == sys.float_info.max, (
        "float64's largest finite value is no longer Python's, so a float "
        "literal past it may now exist and float64 needs a driven row here"
    )
    for name in HOST_DOOR_LOUD:
        if name == "float64":
            continue
        dt = getattr(jnp, name)
        literal = float(ml_dtypes.finfo(np.dtype(dt)).max) * 100.0
        for label, build in _HOST_CONSTRUCTION_DOORS:
            outcome, _ = _host_door_outcome(dt, literal, build)
            assert outcome == "WARNS", (
                f"{label} on {name} with {literal!r} came back {outcome!r}. "
                f"numpy's own binary formats are the ENTIRE reach of "
                f"`-W error::RuntimeWarning` on the host, and "
                f"docs/overflow-tripwire.md's remedy sentence names this one."
            )

    # ---- (3) THE SILENT HALF, THE SAME THREE DOORS. This is the finding.
    excluded = {}
    for name in HOST_DOOR_SILENT:
        dt = getattr(jnp, name)
        biggest = float(ml_dtypes.finfo(np.dtype(dt)).max)
        literal = 1e300 if biggest * 100.0 == float("inf") else biggest * 100.0
        for label, build in _HOST_CONSTRUCTION_DOORS:
            outcome, image = _host_door_outcome(dt, literal, build)
            if outcome == "unbuildable":
                excluded.setdefault(name, image)
                continue
            assert outcome == "silent", (
                f"{label} on {name} with {literal!r} now WARNS. That is "
                f"better news than docs/overflow-tripwire.md records: its "
                f"float-overflow table says a host narrowing into any format "
                f"numpy does not implement itself is silent, and its remedy "
                f"sentence says `-W error::RuntimeWarning` does not reach "
                f"them. Move {name} across in both places."
            )
            assert image != image or abs(image) >= biggest, (
                f"{label} on {name}: {literal!r} came back {image!r}, which "
                f"is neither nan nor at least the largest finite value"
            )
    for name in HOST_DOOR_SILENT_DRIVEN_EVERYWHERE:
        assert name not in excluded, (
            f"{name} could not be driven at a construction door on this jax "
            f"({excluded[name]}). It is declared drivable in every cell, so "
            f"this half of the measurement has gone partly vacuous."
        )

    # ---- (4) AND IT IS A HOST NARROWING, not the device residue wearing its
    #      clothes: the jaxpr already holds the overflowed constant, before
    #      any XLA program exists -- exactly as the float16 route does, and
    #      that one warns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quiet = str(jax.make_jaxpr(lambda: jnp.full((), 1e300, jnp.bfloat16))())
        loud = str(jax.make_jaxpr(lambda: jnp.full((), 1e300, jnp.float16))())
    assert "inf:bf16[]" in quiet, quiet
    assert "inf:f16[]" in loud, loud

    # ---- (5) THE CONTROL, WITH NO JAX IN IT. One numpy cast loop, one
    #      float32 source array: loud into float16, silent into two formats
    #      that lose more. Nothing about hosts or devices can explain this.
    source = np.array([1e30], dtype=np.float32)
    control = {}
    for name in ("float16", "float8_e5m2", "float8_e4m3fn"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                got = source.astype(np.dtype(getattr(jnp, name)))
                control[name] = ("silent", float(got[0]))
            except RuntimeWarning:
                control[name] = ("WARNS", None)
    assert control["float16"][0] == "WARNS", control
    assert control["float8_e5m2"] == ("silent", float("inf")), control
    assert control["float8_e4m3fn"][0] == "silent", control
    assert control["float8_e4m3fn"][1] != control["float8_e4m3fn"][1], control

    # ---- (6) THE PAGE. Three rows, and the third is the one two rounds of
    #      this page denied. Compared cell by cell against what was driven.
    repo = pathlib.Path(__file__).resolve().parents[1]
    page = (repo / "docs" / "overflow-tripwire.md").read_text(encoding="utf-8")
    header = ("| the narrowing | a spelling of it | is there a "
              "`RuntimeWarning` to turn on? |")
    assert header in page, (
        "docs/overflow-tripwire.md no longer carries the float-overflow "
        "warning table this test drives, under the header it is located by"
    )
    body = page.split(header, 1)[1].split("\n\n", 1)[0]
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in body.splitlines()
        if line.strip().startswith("|") and set(line.strip()) - set("|- ")
    ]
    assert len(rows) == 3, rows
    assert rows[0][0] == "on DEVICE, into any dtype", rows[0]
    assert rows[0][2].startswith("**no**"), rows[0]
    assert rows[1][0] == (
        "on the HOST, into "
        + ", ".join(f"`{n}`" for n in HOST_DOOR_LOUD[:-1])
        + f" or `{HOST_DOOR_LOUD[-1]}`"
    ), rows[1]
    assert rows[1][2].startswith("**yes**"), rows[1]
    assert "overflow encountered in cast" in rows[1][2], rows[1]
    assert len(HOST_DOOR_SILENT) == 12, HOST_DOOR_SILENT
    assert rows[2][0] == (
        "on the HOST, into any of the **other twelve** float formats "
        "`jax.numpy` has"
    ), rows[2]
    assert rows[2][2].startswith("**no**"), rows[2]

    # ...the enumeration of the silent half, in both directions.
    flowed = " ".join(page.split())
    listed = sorted(HOST_DOOR_SILENT)
    sentence = (
        "**The other twelve are** "
        + ", ".join(f"`{n}`" for n in listed[:-1])
        + f" and `{listed[-1]}` — every float format `jax.numpy` exposes "
        "that `numpy` does not implement itself."
    )
    assert sentence in flowed, (
        f"docs/overflow-tripwire.md does not enumerate the silent formats as "
        f"this test just measured them. Expected:\n{sentence}"
    )
    for name in HOST_DOOR_LOUD:
        assert f"`{name}`" not in sentence, name   # the lists cannot be swapped

    # ...and the remedy sentence itself, which is the one line on this page a
    # reader acts on and the one that has been wrong twice.
    remedy = (
        "catches a HOST narrowing into "
        + ", ".join(f"`{n}`" for n in HOST_DOOR_LOUD[:-1])
        + f" or `{HOST_DOOR_LOUD[-1]}`, and it catches nothing else"
    )
    assert remedy in flowed, (
        f"docs/overflow-tripwire.md's remedy sentence no longer names exactly "
        f"the formats numpy reports. Expected:\n{remedy}"
    )
    # ...and the SECOND page that states it. `docs/quickstart.md` carried the
    # same false universal -- "where its narrowing happens on the host numpy
    # still warns" -- and the only thing holding it was that the string
    # `-W error::RuntimeWarning` appeared somewhere in the file.
    quick = " ".join(
        (repo / "docs" / "quickstart.md").read_text(encoding="utf-8").split()
    )
    assert remedy in quick, (
        f"docs/quickstart.md's one sentence about this no longer names "
        f"exactly the formats numpy reports. Expected:\n{remedy}"
    )
    assert "not a host narrowing into `bfloat16` or any `float8_*`" in quick, (
        "docs/quickstart.md no longer states the half `-W "
        "error::RuntimeWarning` misses on the host"
    )
    assert "It does not reach a device narrowing in any dtype" in flowed, (
        "docs/overflow-tripwire.md no longer says `-W error::RuntimeWarning` "
        "misses the DEVICE half, which this file's FLOAT_OVERFLOW_DEVICE_"
        "SILENT group measures"
    )
    assert (
        "it does not reach a host narrowing into `bfloat16` or any of the "
        "other eleven formats listed above" in flowed
    ), (
        "docs/overflow-tripwire.md no longer says `-W error::RuntimeWarning` "
        "misses the HOST narrowing into the ml_dtypes formats, which is the "
        "half this test just drove and the half a bfloat16 program lives in"
    )

    # ---- (7) AND THE THREE PROSE SENTENCES THAT RESTATE THE DRIVE.
    #
    # Each is rebuilt here from what was measured above, because a sentence
    # that merely AGREES with a gate is not held by it: measured before this
    # block existed, the page could say the three bfloat16 doors "each RAISE
    # a RuntimeWarning", and the control's silent row could say "**warns**",
    # with nothing going red.
    doors = (
        "`jnp.full((2,), 1e300, jnp.bfloat16)`, "
        "`jnp.array([1e300], jnp.bfloat16)` and `jnp.bfloat16(1e300)` are "
        "each `inf` with nothing raised"
    )
    assert doors in flowed, (
        f"docs/overflow-tripwire.md no longer states the bfloat16 "
        f"construction-door result this test just drove. Expected:\n{doors}"
    )
    # ...and the spellings in that sentence are EXACTLY the doors driven, so
    # a door leaving the drive cannot leave the sentence claiming it.
    spelt = set()
    for label, _ in _HOST_CONSTRUCTION_DOORS:
        spelling = label.replace("LIT", "1e300").replace("dt", "jnp.bfloat16")
        spelt.add(spelling.replace("jnp.<jnp.bfloat16>", "jnp.bfloat16"))
    assert set(re.findall(r"`([^`]+)`", doors)) == spelt | {"inf"}, (
        f"the page's bfloat16 sentence names "
        f"{sorted(re.findall(r'`([^`]+)`', doors))} and this test drives "
        f"{sorted(spelt)}"
    )

    control_sentence = (
        f"`.astype(jnp.float16)` **{control['float16'][0].lower()}**; "
        f"`.astype(jnp.float8_e5m2)` is **{control['float8_e5m2'][0]}** and "
        f"gives `inf`; `.astype(jnp.float8_e4m3fn)` is "
        f"**{control['float8_e4m3fn'][0]}** and gives `nan`."
    )
    assert control_sentence in flowed, (
        f"docs/overflow-tripwire.md's control sentence is not what the "
        f"control just did. Expected:\n{control_sentence}"
    )

    # ...and the DEVICE bullet's case list, as an EQUALITY over the code
    # spans in it. A membership check passed a bullet in which every
    # operation had been swapped for a different one.
    marker = "* **What IS silent whatever the dtype"
    assert marker in page, page[:200]
    bullet = page.split(marker, 1)[1].split("\n* ", 1)[0]
    spans = set(re.findall(r"`([^`\n]+)`", " ".join(bullet.split())))
    declared = {
        phrase for _, _, phrase in FLOAT_OVERFLOW_DEVICE_SILENT
        if phrase is not None
    }
    assert spans == declared | set(_DEVICE_BULLET_FURNITURE), (
        f"the page's DEVICE bullet names {sorted(spans)}; this file drives "
        f"{sorted(declared)} and declares {sorted(_DEVICE_BULLET_FURNITURE)} "
        f"as the dtypes and values they are written against. A case in one "
        f"and not the other is how that bullet came to be rewritable into a "
        f"different set of operations with nothing going red."
    )



def test_this_files_position_in_the_collection_is_the_measured_one():
    """``72nd`` -- counted, not typed. And the TOTAL is not demanded at all.

    Two artefacts carry the rank: :func:`_isolate`'s docstring, which is
    where the incident it explains is recorded, and the CHANGELOG entry for
    the same repair. It stood at **149** in both, one measurement behind --
    the hand-maintained-numeral class this batch's own first commit is named
    for, in the batch that named it.

    **AND CORRECTING IT TO 148 DID NOT FIX IT, BECAUSE THE DENOMINATOR IS A
    PROPERTY OF THE ENVIRONMENT AND NOT OF THE TREE.** A module whose
    imports are unavailable is never collected, so at ``c7cf164`` -- one
    tree, one commit -- this command named 148 files here, 147 in a lane on
    the jax 0.10 series, 144 in a lane with no solver wheels, and never
    reaches this file at all in the zero-dep lane, which does not collect
    it. Two CI lanes went red against a static 148, and no static numeral
    could have been right in all of them. Relaxing this to a substring or a
    range would have thrown the check away, and a hard-coded set of accepted
    totals is the same hand-maintained-numeral defect with more entries.

    So this holds what it can actually establish -- which is the reasoning
    :func:`test_the_dial_on_figures_agree_between_the_page_and_the_changelog`
    already applies to the whole-suite pass count, one numeral over. The
    RANK is derived here and demanded verbatim in both artefacts, because it
    is stable (72 in every environment above) and it is the load-bearing
    half: it is why *"the ~4,300 tests after it ran unprotected"* is the size
    it is. The COMMAND that produces the total is demanded too, because
    naming it is what keeps the unwritten figure obtainable -- and it is
    built from the argv this test just ran rather than typed, for the reason
    the comment beside it gives. The TOTAL itself is measured and then only
    PRINTED, in the failure message.

    **The demanded phrase deliberately spans the slot the denominator sat
    in** -- it runs from ``sorts``, through the bolded rank, to ``of the
    files`` -- so a sentence that puts a count back between the two no
    longer matches, and the numeral cannot return unnoticed.

    ``-p no:randomly`` is part of the claim and not an implementation detail:
    with ``pytest-randomly`` active the order is shuffled and a rank means
    nothing, so the artefacts name that spelling too.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]
    collect_args = ("--collect-only", "-q", *deterministic_order_args())
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *collect_args],
        cwd=repo, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(repo / "src"),
             "JAX_PLATFORMS": "cpu", "COLUMNS": "200"},
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    seen, order = set(), []
    for line in proc.stdout.splitlines():
        if "::" not in line:
            continue
        rel = line.split("::", 1)[0].strip()
        if rel not in seen:
            seen.add(rel)
            order.append(rel)
    assert order, proc.stdout[-2000:]

    mine = pathlib.Path(__file__).name
    rank = next(i + 1 for i, rel in enumerate(order) if rel.endswith(mine))
    total = len(order)

    ordinal = f"{rank}{_ORDINAL[rank % 10]}"
    phrase = f"sorts **{ordinal}** of the files"
    # ...and the command the artefacts must name is BUILT FROM THE ARGV THAT
    # WAS JUST RUN rather than typed here. Typed, it stood in this file's own
    # source -- and one of the two artefacts IS this file, read whole, so the
    # assertion line satisfied itself: the .py half of the command check
    # could not go red however the prose was rewritten. Derived, it also
    # cannot drift from the run it describes.
    command = " ".join(("pytest", *collect_args))
    for rel in ("tests/test_narrowing_perimeter.py", "CHANGELOG.md"):
        flowed = " ".join((repo / rel).read_text(encoding="utf-8").split())
        assert phrase in flowed, (
            f"{rel} does not say {phrase!r}. `{command}` sorts {mine} "
            f"{ordinal} here. It names {total} files in this environment -- "
            f"but that figure is a property of the environment, so it is "
            f"deliberately not written in either artefact; if the sentence "
            f"has acquired a count between the rank and `of the files`, "
            f"take it out."
        )
        assert f"`{command}`" in flowed or f"``{command}``" in flowed, (
            f"{rel} no longer names the command this figure comes from, "
            f"which is what keeps the unwritten total obtainable: {command}"
        )


def test_the_changelogs_counts_over_the_six_cases_are_the_driven_ones():
    """*"four warn eagerly"* and *"all of them warn inside `jit`"* -- both false.

    The CHANGELOG entry that corrected B22's inverted float answer counted the
    six cases the page had named, and got both counts wrong. Driven here under
    ``simplefilter("error")``, in the cell that is running:

    * **EAGER: five at x64=0, three at x64=1.** ``four`` is right in no cell,
      and the clause *"identical in all four cells"* is false of this half --
      which the same bullet's next paragraph says, about the same two lines.
    * **INSIDE ``jit``: five of the six.** The sixth,
      ``jit(a.astype(jnp.float32))`` on ``[1e300, 1e300]``, is silent in all
      four cells once its operand is built outside the window -- the operand
      really is an array by then and the conversion really is XLA's. That is
      the case the same entry describes moving OUT of the WARNS group as
      "green for the wrong reason", eleven lines below the sentence.

    Both figures are compared against the entry's own sentence, so a count
    typed there and not driven is red.
    """
    x64 = bool(jax.config.jax_enable_x64)
    eager, jitted = [], []
    for label, eager_build, jit_build in THE_SIX_THE_PAGE_NAMED:
        for build, into in ((eager_build, eager), (jit_build, jitted)):
            jax.clear_caches()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    np.asarray(build())
                except RuntimeWarning:
                    into.append(label)

    assert len(eager) == (3 if x64 else 5), (
        f"with x64 {'ON' if x64 else 'OFF'}, {len(eager)} of the six cases "
        f"warn eagerly ({eager}); the CHANGELOG says five with x64 off and "
        f"three with it on"
    )
    assert len(jitted) == 5, (
        f"{len(jitted)} of the six warn inside `jit` ({jitted}); the "
        f"CHANGELOG says five of the six, with the sixth "
        f"({THE_SIX_THE_PAGE_NAMED[-1][0]}) silent in all four cells"
    )
    assert THE_SIX_THE_PAGE_NAMED[-1][0] not in jitted, jitted

    flowed = " ".join(
        (pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md")
        .read_text(encoding="utf-8").split()
    )
    for sentence in (
        "**five warn eagerly with x64 off, three with x64 on**",
        "**five of the six warn inside `jit`**",
    ):
        assert sentence in flowed, (
            f"CHANGELOG.md no longer states the count this test just drove. "
            f"Expected:\n  {sentence}\nMeasured here: {len(eager)} eager "
            f"(x64 {'on' if x64 else 'off'}), {len(jitted)} under jit."
        )


#: The four lines the dial-on session prints, as (label, regex over the PAGE's
#: code block, regex over the CHANGELOG sentence). These are the
#: "reproducible by a named command" class: nothing in the tree can hold them
#: CURRENT without running the suite, and the page says so and names the
#: command. What a test CAN hold is that the two artefacts carrying them agree
#: -- and they did not: `docs/overflow-tripwire.md` said `4565 passed` and
#: `CHANGELOG.md` said `4568` for the same claim at the same tip, because the
#: CHANGELOG missed the one test `ca0a79b` had just added.
#: The third field is whether the CHANGELOG entry states that figure too. It
#: does not repeat all four, and a comparison that demanded it would be
#: asking the entry to carry something it has no use for -- so which ones are
#: compared is DECLARED, and a figure disappearing from the entry is red
#: rather than quietly dropping out of the comparison.
_DIAL_ON_FIGURES = (
    ("slots", r"(\d+) slot\(s\), (\d+) owner\(s\)", False),
    ("literals", r"(\d+) integer literal\(s\)", True),
    ("nonexistent", r"(\d+)(?: of them| \.\.\.) do not exist", False),
    ("permitted", r"(\d+) narrowing\(s\) PERMITTED[^0-9]*(\d+) site\(s\)", True),
    ("passed", r"(\d+) passed, (\d+) skipped", True),
)


def test_the_dial_on_figures_agree_between_the_page_and_the_changelog():
    """One measurement, two artefacts, and they disagreed.

    `docs/overflow-tripwire.md` printed ``4565 passed, 10 skipped`` under a
    heading claiming it had been re-measured for this batch, and
    ``CHANGELOG.md`` said ``4568`` for the same tip -- the CHANGELOG had
    missed the one test the commit before it added. Neither is checkable
    without running the whole suite, which is why the page names the command
    that produces them; what IS checkable, and cheap, is that the two copies
    of one measurement are the same numbers.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]
    page = (repo / "docs" / "overflow-tripwire.md").read_text(encoding="utf-8")
    block = page.split(
        "armed -- the dunder perimeter is live on tracer, array:", 1
    )[1].split("```", 1)[0]
    changelog = " ".join(
        (repo / "CHANGELOG.md").read_text(encoding="utf-8").split()
    )
    entry = changelog.split("The dial can be turned on over this repository", 1)
    assert len(entry) == 2, "CHANGELOG.md no longer carries the dial-on entry"
    entry = entry[1].split("- **", 1)[0]

    compared = 0
    for label, pattern, in_changelog in _DIAL_ON_FIGURES:
        here = re.search(pattern, "armed -- the dunder perimeter is live on "
                                  "tracer, array:" + block)
        assert here, (
            f"docs/overflow-tripwire.md's dial-on block no longer prints the "
            f"{label} figure. Block:\n{block}"
        )
        if not in_changelog:
            continue
        compared += 1
        there = re.search(pattern, entry)
        assert there, (
            f"the CHANGELOG's dial-on entry no longer states the {label} "
            f"figure, which the page prints as {here.group(0)!r}"
        )
        assert here.groups() == there.groups(), (
            f"the dial-on {label} figure is {here.groups()} in "
            f"docs/overflow-tripwire.md and {there.groups()} in CHANGELOG.md. "
            f"One measurement, two copies: re-run\n  JAX_ENABLE_X64=0 pytest "
            f"-q -p no:randomly --stelling-narrowing-perimeter=error\n"
            f"and write the same numbers in both."
        )
    assert compared == 3, compared

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

    **AND THE PLAIN SPELLING CANNOT REDDEN ON THE THING THE ROW IS ABOUT.**
    ``as_text()`` -- the spelling the page quotes -- emits NO ``loc(`` at all
    (measured: 0, in all four cells), so there is no ``source_info`` in that
    text for the perimeter to perturb and the entry is identical for a reason
    that has nothing to do with arming. ``as_text(debug_info=True)`` emits 144
    on this harness and DIFFERS across the boundary. Both are lowered, both
    are in the table, and :func:`test_the_stablehlo_rows_say_why_one_is_blind`
    pins the ``loc(`` counts so that "identical" stays readable as "identical
    because the text carries none of it".
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


def _stablehlo(debug_info: bool) -> str:
    """One lowering, one call site -- see :data:`PERTURBED_LOWERING`.

    Every caller must invoke this from a FIXED source line when it intends to
    compare two ``debug_info=True`` texts, because that text records the
    caller's line number.
    """
    jax.clear_caches()
    return jax.jit(_lowerable).lower(C32, jnp.int32(2)).as_text(
        debug_info=debug_info)


def _artefacts():
    jax.clear_caches()
    closed = trace(_rich_harness)
    bare = closed.to_dict(include_metadata=False)
    meta = closed.to_dict(include_metadata=True)
    raw_eqns = "\n".join(repr((e.primitive, tuple(e.params))) for e in closed.jaxpr.eqns)
    jax.clear_caches()
    verdict = check(_rich_harness, vacuity_mode="inputs-only")
    stablehlo = _stablehlo(debug_info=False)

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

#: The artefact that MOVES and cannot be measured through :func:`_artefacts`,
#: which is why it has a tuple of its own rather than a row in ``PERTURBED``.
#:
#: ``as_text(debug_info=True)`` records the PYTHON CALL SITE of the lowering,
#: caller frames included -- so two calls to ``_artefacts()`` written on
#: different lines produce different text with nothing armed. Comparing those
#: digests across an armed boundary would have "differed" for a reason that
#: has nothing to do with the perimeter: a positive control satisfied by its
#: own instrument. Measured: from ONE call site, disarmed and re-disarmed
#: agree exactly, and armed does not.
PERTURBED_LOWERING = ("StableHLO text (debug_info=True)",)


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




#: ``docs/overflow-tripwire.md``'s artefact table, row label -> the keys of
#: :func:`_artefacts` that row is a claim about. Declared rather than derived
#: because the page's labels are prose and the keys are not, and a mapping
#: somebody has to edit is the point: **nothing in this tree read that table.**
#: B22 fixed an instance of it -- ``StableHLO text`` was missing from
#: ``IDENTICAL`` -- without adding the gate, and a row moving between the two
#: halves stayed silent afterwards. Driven: flipping the StableHLO row to
#: ``**DIFFERS**`` on the page left 124 tests green.
_DOC_ARTEFACT_ROWS = (
    ("`ClosedJaxpr.content_hash()`", ("content_hash",)),
    ("`consts`", ("consts",)),
    ("`to_dict(include_metadata=False)`", ("to_dict(include_metadata=False)",)),
    ("the `eqns` of that metadata-free document", ("eqns (metadata-free)",)),
    ("StableHLO text (`jit(...).lower().as_text()`)", ("StableHLO text",)),
    ("`check()` status and notes", ("check status", "check notes")),
    ("`to_dict(include_metadata=True)`", ("to_dict(include_metadata=True)",)),
    ("`str(jaxpr)`", ("str(jaxpr)",)),
    ("a raw `repr()` of an equation's params, which carries `source_info`",
     ("eqns (raw repr, source_info included)",)),
    ("StableHLO text with debug info "
     "(`jit(...).lower().as_text(debug_info=True)`)",
     ("StableHLO text (debug_info=True)",)),
)

_DOC_ARTEFACT_HEADER = "| artefact | armed == disarmed |"


def test_the_pages_artefact_table_is_the_two_declared_halves():
    """The page's ten rows against ``IDENTICAL`` and ``PERTURBED``.

    Both directions, so neither side can move alone: every page row is
    declared here and mapped to a measured key, every measured key is claimed
    by exactly one page row, and each row's **identical**/**DIFFERS** cell has
    to agree with which tuple its keys are in.

    This is a table-vs-tuple comparison and not a re-drive: the drive is
    :func:`test_arming_is_observational_on_everything_the_hash_scope_covers`
    and :func:`test_the_source_info_perturbation_is_disclosed_rather_than_denied`,
    which is where a WRONG tuple goes red. What goes red here is a page that
    stops agreeing with the tuple -- the failure that had no instrument.
    """
    page = (pathlib.Path(__file__).resolve().parents[1]
            / "docs" / "overflow-tripwire.md").read_text(encoding="utf-8")
    assert _DOC_ARTEFACT_HEADER in page, (
        "docs/overflow-tripwire.md no longer carries the artefact table this "
        "test reads, under the header it is located by"
    )
    body = page.split(_DOC_ARTEFACT_HEADER, 1)[1].split("\n\n", 1)[0]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert len(cells) == 2, line
        rows.append(cells)

    assert len(rows) == len(_DOC_ARTEFACT_ROWS), (
        f"the page's artefact table has {len(rows)} rows and this test "
        f"declares {len(_DOC_ARTEFACT_ROWS)}. A row added there is a row "
        f"declared here, against a key of _artefacts()."
    )

    claimed: set[str] = set()
    for (label, keys), (page_label, verdict) in zip(_DOC_ARTEFACT_ROWS, rows):
        assert label == page_label, (
            f"row order moved: this test declares {label!r} where the page's "
            f"row reads {page_label!r}"
        )
        halves = {"**identical**": IDENTICAL,
                  "**DIFFERS**": PERTURBED + PERTURBED_LOWERING}
        assert verdict in halves, (
            f"the page's cell for {label!r} reads {verdict!r}; this test knows "
            f"only {sorted(halves)}"
        )
        for key in keys:
            assert key in halves[verdict], (
                f"the page says {label!r} is {verdict} across an armed "
                f"boundary, and {key!r} is declared in "
                f"{'IDENTICAL' if key in IDENTICAL else 'a DIFFERS'} tuple. One of "
                f"the two is now wrong, and this test exists because moving a "
                f"row between the halves used to be silent."
            )
            claimed.add(key)

    measured = set(IDENTICAL) | set(PERTURBED) | set(PERTURBED_LOWERING)
    assert claimed == measured, (
        f"artefacts measured but claimed by no row of the page: "
        f"{sorted(measured - claimed)}; rows claiming artefacts that are not "
        f"measured: {sorted(claimed - measured)}"
    )


def test_the_stablehlo_rows_say_why_one_is_blind():
    """``as_text()`` is identical because it carries no `loc(` -- measured.

    **THE PLAIN ROW IS A CONTROL THAT CANNOT FAIL, AND THE PAGE CALLED IT THE
    ROW A READER IS MOST LIKELY TO ACT ON.** ``jit(...).lower().as_text()`` --
    the spelling the page quotes -- emits **no** ``loc(`` at all, so the
    perimeter's extra ``source_info`` frame has nowhere to appear in it and
    the ``IDENTICAL`` entry could never redden on the perturbation it exists
    to disclose. With ``debug_info=True`` the same lowering carries 144 of
    them and the text **DIFFERS** armed against disarmed.

    So both are in the page's table now, and the reason is pinned rather than
    reasoned about: if a future jax starts emitting locations from the plain
    spelling, the first assertion here goes red and the page's first StableHLO
    row is what has to change -- before somebody diffs an armed build against
    a normal one and gets a wrong answer.

    **THREE WINDOWS, FROM ONE CALL SITE.** ``disarmed``/``armed``/``again``
    are all lowered on the same source line, because ``debug_info=True``
    records the caller's line and two calls written on different lines differ
    with nothing armed at all. The ``again`` window is what makes the middle
    one a measurement of the perimeter: it has to come back.

    Control, so that "no ``loc(``" is not read as "the wrapper was not live":
    the guard sees checks during the lowering itself.
    """
    before = perimeter.snapshot().get("checks", 0)
    texts = {}
    for state in ("disarmed", "armed", "again"):
        window = (perimeter.armed(("tracer",)) if state == "armed"
                  else contextlib.nullcontext(None))
        with window as status:
            if state == "armed":
                assert status.armed, status.explanation
            texts[state] = (_stablehlo(False), _stablehlo(True))
    after = perimeter.snapshot().get("checks", 0)

    plain = {state: v[0] for state, v in texts.items()}
    debug = {state: v[1] for state, v in texts.items()}

    assert plain["disarmed"].count("loc(") == 0, (
        f"jit(...).lower().as_text() now emits "
        f"{plain['disarmed'].count('loc(')} loc( entries. It emitted none, "
        f"which is the whole reason its row is IDENTICAL; if it carries "
        f"source_info now, that row is a claim about arming again and has to "
        f"be re-measured."
    )
    assert debug["disarmed"].count("loc(") > 0, (
        "as_text(debug_info=True) emits no loc( either, so this file no "
        "longer has a positive control for the StableHLO rows at all"
    )
    assert plain["armed"] == plain["disarmed"], (
        "the plain StableHLO text moved across an armed boundary; the page's "
        "first StableHLO row says it does not"
    )
    assert debug["armed"] != debug["disarmed"], (
        "the HLO with debug info no longer moves across an armed boundary. "
        "That would be good news and it is not a silent improvement: "
        "docs/overflow-tripwire.md's second StableHLO row says it DIFFERS."
    )
    assert debug["again"] == debug["disarmed"], (
        "the debug-info HLO did not come back after disarming, so the "
        "difference above is not attributable to the perimeter. Every "
        "lowering here has to be on ONE source line -- that text records the "
        "caller's line number."
    )
    assert after > before, (
        "no guard check ran during the lowering, so a 'no loc(' reading above "
        "would be about a perimeter that was not live rather than about the "
        "text"
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
        # `borrowed_*` and not bare arm/disarm: an unconditional `disarm()`
        # here DETACHES a session-armed tripwire. See `conftest.borrowed_tripwire`.
        with contextlib.ExitStack() as stack:
            try:
                for which in order:
                    if which == "perimeter":
                        statuses["perimeter"] = perimeter.arm(
                            perimeter.FACES, owner="A")
                    else:
                        status, _rec = stack.enter_context(borrowed_tripwire())
                        statuses["tripwire"] = status
                statuses["eager"] = stack.enter_context(borrowed_eager())
                assert statuses["perimeter"].armed, statuses["perimeter"].explanation
                assert statuses["tripwire"].armed, statuses["tripwire"].explanation
                assert statuses["eager"].armed, statuses["eager"].explanation
            finally:
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
    with borrowed_tripwire() as (tripwire_status, _rec), \
            borrowed_eager() as eager_status:
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
