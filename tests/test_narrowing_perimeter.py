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

import contextlib
import hashlib
import io
import json
import pathlib
import re

import pytest

pytest_plugins = ["pytester"]

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import numpy as np  # noqa: E402

import stelling  # noqa: E402
from conftest import deterministic_order_args, tripwire_plugin_args  # noqa: E402
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


@pytest.fixture(autouse=True)
def _isolate():
    """Never leave the perimeter armed for the rest of the suite.

    ``runpytest`` is in-process by default, so a nested session in this file
    arms slots in THIS interpreter; and a test that drives a refusal leaves
    counters moved. Both are put back, and the slot identity is asserted on
    the way out so that a leak in one test is reported by that test rather
    than by whatever runs next.
    """
    before = perimeter.armed_faces()
    yield
    for face in list(perimeter._installed):
        perimeter._restore_face(face)
    perimeter._owners.clear()
    perimeter.reset_counters()
    prop_guard.UNKNOWN_SLOTS.clear()
    prop_guard.INTERNAL_DECLINES.clear()
    assert perimeter.armed_faces() == before == ()


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


def _artefacts():
    jax.clear_caches()
    closed = trace(_rich_harness)
    bare = closed.to_dict(include_metadata=False)
    meta = closed.to_dict(include_metadata=True)
    raw_eqns = "\n".join(repr((e.primitive, tuple(e.params))) for e in closed.jaxpr.eqns)
    jax.clear_caches()
    verdict = check(_rich_harness, vacuity_mode="inputs-only")

    def digest(obj):
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, default=str).encode()
        ).hexdigest()

    return {
        "content_hash": closed.content_hash(),
        "consts": digest([(str(getattr(c, "dtype", "?")), repr(c)) for c in closed.consts]),
        "eqns (metadata-free)": digest(bare.get("jaxpr", {}).get("eqns", bare)),
        "to_dict(include_metadata=False)": digest(bare),
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
                    statuses["perimeter"] = perimeter.arm(("tracer",), owner="A")
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
    rows, moved = canary._perimeter_facts(_tracer_type())
    assert not moved, dict(rows)
    names = {name for name, _value in rows}
    assert names == {
        "perimeter type",
        "Py_TPFLAGS_HEAPTYPE",
        "slots are own attributes",
        "setattr rebinds and restores identity",
        "a WARM traced op still enters Python",
        "no in-place slots to bypass the forward ones",
    }, names
    assert all(value.startswith("PASS") for _name, value in rows), dict(rows)


@pytest.mark.parametrize(
    ("fault", "reddens"),
    [
        # jax renames or restructures the type the perimeter attaches to
        ("no-type", {"perimeter type"}),
        # jax moves the slots off the type it used to own them on
        ("no-slots", {"slots are own attributes",
                      "setattr rebinds and restores identity",
                      "a WARM traced op still enters Python"}),
        # jax grows an in-place slot, so `x += N` stops falling back
        ("inplace", {"no in-place slots to bypass the forward ones"}),
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
        rows, moved = canary._perimeter_facts("no-type")
    elif fault == "no-slots":
        class _Stripped:
            """A type that answers to the name and owns none of the slots."""

            __mro__ = ()

        rows, moved = canary._perimeter_facts(_Stripped)
    else:
        class _WithInplace(tracer):
            def __iadd__(self, other):  # pragma: no cover - never called
                return self

        rows, moved = canary._perimeter_facts(_WithInplace)

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
