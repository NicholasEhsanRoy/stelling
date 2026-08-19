# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Mode 2, the eager construction-site detector, driven end to end.

WHAT IS BEING PROTECTED HERE, in one paragraph. ``jnp.full((), 256,
jnp.int8)`` is ``0``. The 256 is destroyed before any jax primitive is bound,
so nothing downstream — no jaxpr, no transfer, no solver, and not the overflow
tripwire, which watches a const-fold rule the value never reaches — can tell
that the ``0`` it is reasoning about was written as a ``256``. The detector
raises at the line that wrote it, which needs no attribution because the
writing frame is still on the stack.

FOUR THINGS THIS FILE IS FOR, and they are four different failures:

1. **The exception really is unswallowable by the common swallow.** Driven
   through a real ``except Exception:``, and the honest limit — ``except
   BaseException:`` still catches it — is driven too, so the claim in
   ``eager.py``'s docstring is measured rather than asserted.
2. **The two declarations are exact and cannot leak.** A declaration is
   scoped to one value at one site, or lexically to one block on one thread,
   and both of those are driven by showing the NEXT truncation still raises.
3. **Drift fails CLOSED.** This patches a private jax function. Every way it
   can go wrong — the module moving, the signature moving, one construction
   route no longer reaching the site, something rebinding over the top — is
   driven through a seam in ``_adapter_jax`` and must produce a refusal, never
   a quiet attach.
4. **The default path is untouched.** Mode 2 is opt-in; a user who does not
   switch it on must get byte-identical behaviour, and that is asserted
   against the private attribute itself rather than against the flag's
   default.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import textwrap
import threading

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

import stelling  # noqa: E402
from stelling import _tripwire  # noqa: E402
from stelling._tripwire import _adapter_jax as adapter  # noqa: E402
from stelling._tripwire import eager, record, report  # noqa: E402
from stelling._tripwire.eager import expected_truncation  # noqa: E402


@pytest.fixture
def armed():
    """The detector, armed for one test and put back the way it was found.

    PER-TEST, because it is a rule rather than a report: while it is live
    every construction in the process is subject to it, including the ones
    other fixtures make.

    IT RESTORES RATHER THAN DISARMS, and that is not fussiness. This suite is
    meant to be runnable with ``--stelling-eager-truncation=error``, which
    arms the detector for the WHOLE session -- and a fixture that disarmed
    unconditionally would take the session's detector out at the first test
    here and leave every later file unwatched, silently. The state at entry
    is what is restored.
    """
    was_armed = eager.is_armed()
    status = _tripwire.arm_eager()
    if not status.armed:
        pytest.skip(f"could not attach: {status.code} -- {status.explanation}")
    eager.reset_counters()
    yield status
    if not was_armed:
        _tripwire.disarm_eager()
    eager.reset_counters()


@contextlib.contextmanager
def _detached():
    """Take the detector out for the duration, and put it back as it was.

    For the handful of measurements whose SUBJECT is the unarmed program --
    "the default path is byte-identical", "the wrapper changes no result".
    Those need the hook gone, not the truncation permitted, so
    ``expected_truncation`` is the wrong tool for them.
    """
    was_armed = eager.is_armed()
    if was_armed:
        _tripwire.disarm_eager()
    try:
        yield
    finally:
        if was_armed:
            _tripwire.arm_eager()


@pytest.fixture
def unarmed():
    """The detector OFF for this test, and back as it was afterwards.

    For the tests that arm and disarm on their own -- the fail-closed
    battery, which has to drive `arm_eager()` and read its refusal. Starting
    them from a known state is what lets them be written as if the session
    had not armed anything, which is the only way they read.
    """
    with _detached():
        yield


@pytest.fixture(autouse=True)
def _the_arm_state_is_left_as_it_was_found():
    """Every test in this file leaves the process as it found it.

    An eager detector left armed by a failing test would turn every later test
    in the session into a test of the detector, which is how one red becomes
    forty; one left DISARMED under a session that asked for it would be worse,
    because it is silent. Asserted rather than trusted, in both directions.
    """
    before = eager.is_armed()
    yield
    assert eager.is_armed() == before, (
        f"a test in this file changed the process's arm state: it was "
        f"{'armed' if before else 'not armed'} and is now the other"
    )


# ---------------------------------------------------------------------------
# 1. The exception
# ---------------------------------------------------------------------------


def test_the_alarm_inherits_from_BaseException_and_not_from_Exception():
    """The single most consequential line in the module, pinned.

    ``Exception`` would put the alarm inside the reach of every
    ``except Exception:`` in numerical Python — retry loops, fallback kernels,
    warnings shims, and this repository's own guardrails, which catch
    ``Exception`` on principle precisely so that an instrument cannot break
    what it measures. A soundness alarm swallowed by a handler written for
    something else is a silent program with extra steps.
    """
    assert issubclass(stelling.EagerTruncationError, BaseException)
    assert not issubclass(stelling.EagerTruncationError, Exception), (
        "the alarm is catchable by `except Exception:`, which is the one "
        "handler shape it exists to escape"
    )
    assert stelling.EagerTruncationError.__bases__ == (BaseException,), (
        "it must inherit DIRECTLY from BaseException; an intermediate base "
        "is a place for someone to hang an `Exception` mixin later"
    )


def test_a_real_except_Exception_does_not_swallow_it(armed):
    """Driven through the handler shape, not asserted about the class tree."""
    swallowed = []
    with pytest.raises(stelling.EagerTruncationError):
        try:
            jnp.full((), 256, jnp.int8)
        except Exception as exc:  # noqa: BLE001 - the point of the test
            swallowed.append(exc)
    assert not swallowed, "an `except Exception:` block caught the alarm"


def test_what_the_choice_does_NOT_claim_is_also_driven(armed):
    """"Uncatchable" is not achievable in Python and this does not claim it.

    A claim a reader might over-read is worth measuring in the direction it
    would be over-read. ``except BaseException:`` catches this, a bare
    ``except:`` catches this, and ``finally:`` still runs — so a caller whose
    cleanup lives in ``except Exception:`` and not in ``finally:`` WILL leak
    it. That cost is stated in ``eager.py`` and is measured here.
    """
    caught = None
    try:
        jnp.full((), 256, jnp.int8)
    except BaseException as exc:  # noqa: BLE001 - the point of the test
        caught = exc
    assert isinstance(caught, stelling.EagerTruncationError)

    ran = []
    with pytest.raises(stelling.EagerTruncationError):
        try:
            jnp.full((), 256, jnp.int8)
        finally:
            ran.append("finally")
    assert ran == ["finally"], "a `finally:` block did not run"


def test_the_alarm_carries_the_facts_as_fields_and_not_only_as_text(armed):
    """A caller that does catch this must not have to parse the message."""
    with pytest.raises(stelling.EagerTruncationError) as caught:
        jnp.full((4,), 300, jnp.int8)
    exc = caught.value
    assert (exc.written, exc.to_dtype, exc.became) == (300, "int8", 44)
    assert exc.file == __file__
    assert exc.func == "test_the_alarm_carries_the_facts_as_fields_and_not_only_as_text"
    # THE LINE IS CHECKED BY READING IT, not by arithmetic on a code object.
    # An offset from `co_firstlineno` is a number that has to be maintained
    # whenever a docstring gains a sentence, and a maintained number is one
    # that gets "fixed" to match whatever the instrument said -- which is
    # exactly the direction an attribution test must not be able to go.
    assert "300" in record.source_line(exc.file, exc.line), (
        f"attributed to {exc.file}:{exc.line}, which does not write 300: "
        f"{record.source_line(exc.file, exc.line)!r}"
    )
    assert "jnp.full" in record.source_line(exc.file, exc.line)


def test_the_message_carries_the_report_s_four_obligations(armed):
    """Both halves observed, the arithmetic, the user's own line, the remedy.

    ``report.py`` argues that a finding a reader cannot check costs more trust
    than ten they can. The same bar applies to an alarm that stops their
    program, and it is checked here against the same list.
    """
    with pytest.raises(stelling.EagerTruncationError) as caught:
        jnp.full((4,), 300, jnp.int8)   # the quoted line
    text = str(caught.value)
    assert "written   300" in text and "became    44" in text
    assert record.arithmetic_sentence(300, "int8") in text
    assert "jnp.full((4,), 300, jnp.int8)   # the quoted line" in text, (
        "the alarm does not quote the line that wrote the constant"
    )
    assert "intentional_wrap(300, 'int8')" in text, (
        "the alarm does not say what to write instead"
    )
    assert "no value-based exemption" in text


# ---------------------------------------------------------------------------
# 2. intentional_wrap
# ---------------------------------------------------------------------------

#: (value, dtype) pairs spanning both signs, both edges and both directions of
#: the wrap. Compared against NUMPY, which is the independent route: stelling's
#: own arithmetic is `record.narrow`, and checking one against the other is
#: what makes the declaration a declaration about jax's behaviour rather than
#: about stelling's opinion of it.
WRAP_GRID = [
    (255, "int8"), (256, "int8"), (300, "int8"), (-129, "int8"),
    (0xFF, "int8"), (0xF0, "int8"), (-1, "uint8"), (256, "uint8"),
    (40000, "int16"), (-32769, "int16"), (2**31, "int32"), (-1, "uint32"),
    (2**63, "int64"), (-1, "uint64"),
]


@pytest.mark.parametrize("value,dtype", WRAP_GRID)
def test_intentional_wrap_agrees_with_numpy_over_the_whole_grid(value, dtype):
    """The declaration returns exactly what jax's narrowing would have made.

    That equality is the whole design: the program a declaration produces is
    byte-identical to the program without one, so switching the detector on
    and off cannot change a result.
    """
    got = stelling.intentional_wrap(value, dtype)
    assert got == int(np.asarray(value).astype(dtype)), (
        f"intentional_wrap({value}, {dtype!r}) is {got} and numpy makes "
        f"{int(np.asarray(value).astype(dtype))} of the same narrowing"
    )
    assert record.in_range(got, dtype), (
        "the declared value is itself out of range, so it would fire"
    )


def test_intentional_wrap_answers_where_numpy_itself_cannot():
    """The declaration is Python integer arithmetic, and that is not academic.

    ``np.asarray(2**70)`` raises ``OverflowError`` before any narrowing can
    happen, so the grid above cannot reach these values through numpy at all —
    and jax's narrowing line, which is ``np.asarray(operand).astype(...)``,
    raises there too. The declaration still has an answer, and it is the
    answer two's-complement truncation has.
    """
    assert stelling.intentional_wrap(2**70, "int8") == 0
    assert stelling.intentional_wrap(2**64 + 7, "uint64") == 7
    assert stelling.intentional_wrap(-(2**70) + 5, "int16") == 5
    with pytest.raises(OverflowError):
        np.asarray(2**70).astype(np.int8)


def test_intentional_wrap_takes_every_spelling_of_a_dtype_jax_uses():
    """A user writes ``jnp.int8``; the docs write ``"int8"``. Both, and the
    numpy spellings, or the declaration is a trap for the obvious call."""
    for spelling in ("int8", np.int8, np.dtype("int8"), jnp.int8):
        assert stelling.intentional_wrap(0xFF, spelling) == -1, spelling


def test_a_declaration_is_refused_rather_than_accepted_unchecked():
    """A premise this module cannot check the arithmetic of is not a premise.

    Silently passing an unmodelled dtype through would make
    ``intentional_wrap`` a no-op that reads like a declaration — the worst of
    both — so it raises where it is written.
    """
    with pytest.raises(ValueError, match="not one of the integer dtypes"):
        stelling.intentional_wrap(300, "float32")
    with pytest.raises(ValueError):
        stelling.intentional_wrap(300, "not-a-dtype")
    with pytest.raises(TypeError, match="must be a Python int"):
        stelling.intentional_wrap(300.5, "int8")
    with pytest.raises(TypeError):
        stelling.intentional_wrap(True, "int8")


def test_a_declaration_licenses_ONE_site_and_not_the_next_one(armed):
    """Scoped, and the scope is the expression it is written in.

    This is the property that rules out every design where a declaration is a
    mode: a flag, a suppression list, a decorator on the enclosing function.
    """
    assert int(jnp.full((), stelling.intentional_wrap(300, "int8"), jnp.int8)) == 44
    with pytest.raises(stelling.EagerTruncationError):
        jnp.full((), 300, jnp.int8)


def test_a_declaration_cannot_license_a_DIFFERENT_DTYPE(armed):
    """The dtype is half the declaration, and drifting from it fires.

    ``intentional_wrap(0xFF, "int8")`` is ``-1``, which ``uint8`` cannot hold
    — so a declaration copied to a site whose dtype has changed is caught by
    the detector rather than honoured. A declaration that carried only the
    value could not do this.
    """
    declared = stelling.intentional_wrap(0xFF, "int8")
    assert declared == -1
    with pytest.raises(stelling.EagerTruncationError) as caught:
        jnp.full((), declared, jnp.uint8)
    assert caught.value.written == -1 and caught.value.to_dtype == "uint8"


def test_a_declaration_produces_the_same_program_as_no_declaration_at_all(unarmed):
    """"Byte-identical" is the claim; this is the measurement of it.

    The array a declared construction builds with the detector ARMED, and the
    array the undeclared construction builds with it OFF, are the same array.
    """
    plain = np.asarray(jnp.full((3,), 300, jnp.int8))
    status = _tripwire.arm_eager()
    if not status.armed:  # pragma: no cover - environment
        pytest.skip(status.code)
    try:
        declared = np.asarray(
            jnp.full((3,), stelling.intentional_wrap(300, "int8"), jnp.int8)
        )
    finally:
        _tripwire.disarm_eager()
    assert plain.dtype == declared.dtype
    assert plain.tolist() == declared.tolist() == [44, 44, 44]


def test_a_declaration_for_int64_does_not_license_x64_off_s_int32(armed):
    """The one place a declaration can be right and the program still wrong.

    Under ``jax_enable_x64=False`` jax canonicalises ``int64`` to ``int32`` at
    construction, so ``intentional_wrap(2**40, "int64")`` is a TRUE statement
    about a dtype the array will not have. The declaration is not honoured by
    default and it is not silently re-interpreted: the detector reads the
    dtype jax actually used and fires, naming ``int32``.

    Measured in both cells, which is the only way this claim is worth making:
    with x64 ON the same line builds an ``int64`` holding ``2**40``.
    """
    import warnings

    declared = stelling.intentional_wrap(2**40, "int64")
    assert declared == 2**40, "int64 can hold 2**40; nothing should have wrapped"

    old = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", True)
        assert int(jnp.full((), declared, jnp.int64)) == 2**40

        jax.config.update("jax_enable_x64", False)
        with warnings.catch_warnings():
            # jax warns that it is truncating the requested dtype. The
            # warning is jax's and is not the finding: it says the DTYPE was
            # narrowed and says nothing about the value.
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(stelling.EagerTruncationError) as caught:
                jnp.full((), declared, jnp.int64)
        assert caught.value.to_dtype == "int32", (
            "the alarm names the dtype that was DECLARED rather than the one "
            "jax used, which would make it unactionable"
        )
        assert caught.value.written == 2**40
    finally:
        jax.config.update("jax_enable_x64", old)


def test_a_declaration_is_RECORDED_and_reaches_the_report():
    """A premise nobody can see is the silence this module exists to end.

    The same standing an ``assume()`` has in a verdict — carried, and stamped
    where a reader meets it — rather than applied quietly.
    """
    eager.reset_counters()
    try:
        stelling.intentional_wrap(300, "int8")
        stelling.intentional_wrap(300, "int8")
        snapshot = eager.snapshot()
        assert len(snapshot["declared"]) == 2, snapshot
        assert sum(row[0] for row in snapshot["declared"].values()) == 2
        lines = " ".join(
            report.render_eager(_tripwire.Status(code="armed"), snapshot)
        )
        assert "2 wrap(s) DECLARED" in lines
        assert "300 -> 44 (int8)" in lines
    finally:
        eager.reset_counters()


def test_intentional_wrap_needs_no_jax_and_no_numpy():
    """It is the one part of this feature a user writes in their own source,
    so it has to work in the environment stelling promises: none.

    Driven in a subprocess with jax hidden, which is the same instrument
    ``tests/test_import_hygiene.py`` uses for the rest of the package.
    """
    program = textwrap.dedent(
        """
        import importlib.util as u, sys


        class _NoJax:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("jax", "jaxlib"):
                    raise ModuleNotFoundError(name)
                return None


        sys.meta_path.insert(0, _NoJax())
        import stelling
        assert stelling.intentional_wrap(0xFF, "int8") == -1
        assert stelling.intentional_wrap(-1, "uint8") == 255
        assert issubclass(stelling.EagerTruncationError, BaseException)
        assert not [m for m in sys.modules if m.split(".")[0] == "jax"]
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True, text=True, check=False,
        env={**_env(), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def _env():
    """The environment a subprocess needs to import the tree under test."""
    import os

    from stelling import __file__ as marker

    src = os.path.dirname(os.path.dirname(os.path.abspath(marker)))
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


#: Real spellings of a narrowing constant, each with the verdict a HUMAN gives
#: it. This is the corpus the "no value-based carve-out" ruling was decided on,
#: and it is here rather than only in a design note because it is the input to
#: a proof the suite can run.
#:
#: TWO SPELLINGS OF ONE VALUE APPEAR MORE THAN ONCE ON PURPOSE — ``0xFF`` and
#: ``255``, ``0xFFFF`` and ``65535``, ``~1`` and ``-2``, ``0xFFFFFFFF`` and
#: ``2**32-1``. Those pairs are the whole argument: the source text differs and
#: the observation does not.
#:
#: ``(source text, value, dtype, is it deliberate)``
NARROWING_CORPUS = [
    ("-1", -1, "uint8", True),                    # set every bit
    ("-1", -1, "uint32", True),
    ("~0", ~0, "uint16", True),
    ("0xFF", 0xFF, "int8", True),
    ("0xFFFF", 0xFFFF, "int16", True),
    ("0xFFFFFFFF", 0xFFFFFFFF, "int32", True),
    ("2**32-1", 2**32 - 1, "int32", True),
    ("0xF0", 0xF0, "int8", True),                 # nibble mask
    ("0x80", 0x80, "int8", True),                 # sign-bit / MSB mask
    ("~1", ~1, "uint8", True),                    # clear bit 0
    ("-2", -2, "uint8", True),
    ("0xFFFE", 0xFFFE, "int16", True),
    ("256", 256, "int8", False),                  # off-by-one on a byte
    ("40000", 40000, "int16", False),             # SOUNDNESS.md's reproducer
    ("300", 300, "uint8", False),                 # pixel overflow
    ("255", 255, "int8", False),                  # saturated pixel into SIGNED
    ("65535", 65535, "int16", False),             # uint16 sentinel into signed
    ("2**31", 2**31, "int32", False),             # INT_MAX + 1
    ("-129", -129, "int8", False),                # below min
]


def test_NO_rule_over_the_observation_can_separate_intent():
    """The "no value-based carve-out" ruling, proved rather than asserted.

    Everything a hook can see about a narrowing is ``(written, to_dtype)`` and
    what two's-complement truncation makes of them. THE CORPUS CONTAINS
    COLLISIONS: ``0xFF`` into ``int8`` is a mask idiom and ``255`` into
    ``int8`` is a saturated pixel written into a signed byte, and those are the
    SAME ``(value, dtype)`` pair. Any function of the observation gives them
    the same answer, and one of the two answers is wrong.

    That is a stronger statement than "these two heuristics score badly": it
    says the class of rules is EMPTY, so a future contributor with a cleverer
    discriminator has to change this corpus -- and defend the change -- rather
    than merely score better on it.
    """
    from collections import defaultdict

    by_observation = defaultdict(set)
    for text, value, dtype, deliberate in NARROWING_CORPUS:
        assert not record.in_range(value, dtype), f"{text} into {dtype} fits"
        by_observation[(value, dtype, record.narrow(value, dtype))].add(deliberate)

    collisions = sorted(k for k, v in by_observation.items() if len(v) > 1)
    assert collisions == [(255, "int8", -1), (65535, "int16", -1)], (
        f"the corpus's collisions are {collisions}. If it no longer contains "
        "two rows that a hook cannot tell apart, it no longer proves what it "
        "is here to prove: a value-based exemption needs an argument, not a "
        "corpus with the counterexample deleted."
    )


def _below_the_minimum(value, dtype):
    """Rule A, as proposed: a value below the dtype's minimum is deliberate.

    For an UNSIGNED dtype that is exactly "a negative literal into an unsigned
    type", which is how the rule was described; the implementation generalises
    it to signed types, where it reads "further negative than the type goes".
    """
    return value < record.dtype_range(dtype)[0]


def _all_ones_result(value, dtype):
    """Rule B, as proposed: an all-ones result is deliberate."""
    signed, bits = record.INT_DTYPES[dtype]
    return record.narrow(value, dtype) == (-1 if signed else 2**bits - 1)


CANDIDATE_RULES = {
    "below-the-minimum": _below_the_minimum,
    "all-ones-result": _all_ones_result,
    # the two degenerate ends, so the claim is about the CLASS and not about
    # two rules someone happened to think of
    "everything-is-deliberate": lambda value, dtype: True,
    "nothing-is-deliberate": lambda value, dtype: False,
}


@pytest.mark.parametrize("name", sorted(CANDIDATE_RULES))
def test_every_candidate_rule_is_wrong_somewhere_on_the_corpus(name):
    """Four rules scored, and none of them is right everywhere.

    What matters is not the individual scores but that the minimum is not
    zero, which the collision above makes inevitable and this measures.
    """
    rule = CANDIDATE_RULES[name]
    wrong = [
        (text, value, dtype, deliberate)
        for text, value, dtype, deliberate in NARROWING_CORPUS
        if rule(value, dtype) != deliberate
    ]
    assert wrong, (
        f"{name} classified every row correctly, which the collision test "
        "above says is impossible. Either the corpus lost its collision or "
        "this rule is reading something a hook cannot see."
    )


def test_the_two_proposed_rules_score_what_the_ruling_records():
    """The two figures `design/eager-truncation-detector.md` turns on,
    recomputed from the corpus rather than typed beside it.

    Both directions are counted separately because they cost differently: a
    hard error on correct code is an unusable tool, and a silently allowed bug
    is the defect this whole feature exists to catch.
    """
    scores = {}
    for name in ("below-the-minimum", "all-ones-result"):
        rule = CANDIDATE_RULES[name]
        scores[name] = (
            sum(1 for _, v, d, ok in NARROWING_CORPUS if ok and not rule(v, d)),
            sum(1 for _, v, d, ok in NARROWING_CORPUS if not ok and rule(v, d)),
        )
    assert scores == {
        "below-the-minimum": (7, 1),
        "all-ones-result": (5, 2),
    }, scores


# ---------------------------------------------------------------------------
# 3. expected_truncation
# ---------------------------------------------------------------------------


def test_a_region_requires_a_reason():
    """A region with nothing to say for itself is a silent opt-out."""
    for bad in ("", "   ", None, 7):
        with pytest.raises(TypeError, match="non-empty reason"):
            expected_truncation(bad)


def test_a_region_is_lexically_bounded(armed):
    """It licenses the block and the next line is on its own again."""
    with expected_truncation("driving the region's boundary"):
        assert int(jnp.full((), 300, jnp.int8)) == 44
    with pytest.raises(stelling.EagerTruncationError):
        jnp.full((), 300, jnp.int8)


def test_a_region_nests_and_the_inner_one_ends_where_it_says(armed):
    with expected_truncation("outer"):
        with expected_truncation("inner"):
            jnp.full((), 300, jnp.int8)
        # still inside the outer one
        jnp.full((), 301, jnp.int8)
    with pytest.raises(stelling.EagerTruncationError):
        jnp.full((), 302, jnp.int8)


def test_a_region_does_not_license_ANOTHER_THREAD(armed):
    """Thread-local, because a region is a statement about a block of code.

    Process-global would let one thread's demonstration license another
    thread's accident, which is the "silently license a different site"
    failure the whole declaration design is built to avoid.
    """
    seen = []

    def other():
        try:
            jnp.full((), 300, jnp.int8)
            seen.append("allowed")
        except stelling.EagerTruncationError:
            seen.append("raised")

    with expected_truncation("this thread is demonstrating the truncation"):
        jnp.full((), 300, jnp.int8)
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
    assert seen == ["raised"], (
        "a region on one thread licensed a truncation on another"
    )


def test_a_region_NAMES_what_it_permitted_rather_than_hiding_it(armed):
    """An opt-out that hid what it suppressed would be the same silence, one
    level up. Every permitted truncation is counted, sited and given the
    reason its author wrote."""
    eager.reset_counters()
    with expected_truncation("a reason a reader will meet in the report"):
        jnp.full((), 300, jnp.int8)
        jnp.full((), 301, jnp.int8)
    snapshot = eager.snapshot()
    assert snapshot["truncations"] == 2
    assert sum(row[0] for row in snapshot["permitted"].values()) == 2
    lines = " ".join(report.render_eager(_tripwire.Status(code="armed"), snapshot))
    assert "2 truncation(s) PERMITTED" in lines
    assert "a reason a reader will meet in the report" in lines


def test_a_region_unwinds_when_the_block_raises(armed):
    with pytest.raises(ValueError):
        with expected_truncation("a region that is left by an exception"):
            raise ValueError("boom")
    with pytest.raises(stelling.EagerTruncationError):
        jnp.full((), 300, jnp.int8)


# ---------------------------------------------------------------------------
# 4. The hook
# ---------------------------------------------------------------------------

#: The construction routes the detector claims, spelled the way a user writes
#: them. Held against :func:`_adapter_jax._eager_routes` below, so that the
#: self-check's list and the list a test drives cannot drift apart.
ROUTES = {
    "jnp.full": lambda v: jnp.full((3,), v, jnp.int8),
    "jnp.full_like": lambda v: jnp.full_like(jnp.zeros((3,), jnp.int8), v),
    "lax.full": lambda v: lax.full((3,), v, jnp.int8),
    "lax.full_like": lambda v: lax.full_like(jnp.zeros((3,), jnp.int8), v),
    "lax.convert_element_type": lambda v: lax.convert_element_type(v, jnp.int8),
    "jnp.stack of full": lambda v: jnp.stack(
        [jnp.zeros((3,), jnp.int8), jnp.full((3,), v, jnp.int8)]
    ),
    "lax.select of full": lambda v: lax.select(
        jnp.zeros((3,), jnp.int8) == 0, jnp.full((3,), v, jnp.int8),
        jnp.zeros((3,), jnp.int8),
    ),
    "jnp.take fill_value": lambda v: jnp.take(
        jnp.zeros((3,), jnp.int8), jnp.array([9]), mode="fill", fill_value=v
    ),
    "numpy scalar": lambda v: jnp.full((3,), np.int64(v), jnp.int8),
    "0-d numpy array": lambda v: jnp.full((3,), np.array(v), jnp.int8),
}


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_every_construction_route_raises_at_the_line_that_wrote_it(armed, name):
    with pytest.raises(stelling.EagerTruncationError) as caught:
        ROUTES[name](300)
    assert (caught.value.written, caught.value.became) == (300, 44), name
    assert caught.value.to_dtype == "int8"
    assert caught.value.file == __file__, (
        f"{name} was attributed to {caught.value.file}, not to the file that "
        "wrote the constant"
    )


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_no_construction_route_fires_on_a_value_that_FITS(armed, name):
    """The negative direction. A detector that refused everything would pass
    every test above and be useless, which is the shape of vacuous control
    this repository keeps having to withdraw."""
    out = np.asarray(ROUTES[name](44)).ravel().tolist()
    assert 44 in out, (
        f"{name} with an in-range 44 produced {out}, which does not contain it"
    )


TRANSFORMS = {
    "jit": lambda: jax.jit(lambda z: z + jnp.full(z.shape, 300, jnp.int8))(
        jnp.zeros((3,), jnp.int8)
    ),
    "vmap": lambda: jax.vmap(lambda z: z + jnp.full((), 300, jnp.int8))(
        jnp.zeros((3,), jnp.int8)
    ),
    "scan": lambda: lax.scan(
        lambda c, y: (c, y + jnp.full((), 300, jnp.int8)),
        jnp.int8(0), jnp.zeros((3,), jnp.int8),
    ),
    "cond": lambda: lax.cond(
        True, lambda z: z + jnp.full(z.shape, 300, jnp.int8), lambda z: z,
        jnp.zeros((3,), jnp.int8),
    ),
    "jacfwd": lambda: jax.jacfwd(
        lambda z: (z + jnp.full(z.shape, 300, jnp.int8).astype(jnp.float32)).sum()
    )(jnp.zeros((3,), jnp.float32)),
    "make_jaxpr": lambda: jax.make_jaxpr(
        lambda z: z + jnp.full(z.shape, 300, jnp.int8)
    )(jnp.zeros((3,), jnp.int8)),
}


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_the_alarm_escapes_every_jax_transform_cleanly(armed, name):
    """jax's transforms are full of ``try``/``except`` and of re-raising
    machinery that rewrites exceptions. The alarm has to come out as itself,
    from inside every one of them, or the ``BaseException`` argument buys
    nothing where it is most needed."""
    with pytest.raises(stelling.EagerTruncationError) as caught:
        TRANSFORMS[name]()
    assert caught.value.written == 300
    assert caught.value.file == __file__, (
        f"inside {name} the alarm was attributed to {caught.value.file}"
    )


def test_the_two_numpy_routes_stay_silent_and_are_the_disclosed_residue(armed):
    """The residue, driven. This is the honest half of the coverage claim."""
    assert int(np.asarray(np.asarray(300).astype(np.int8))) == 44
    assert int(np.asarray(jnp.asarray(np.array(300), dtype=jnp.int8))) == 44


def test_jax_s_own_loud_refusals_are_unchanged(armed):
    """Arming must not convert an ``OverflowError`` a user already handles
    into a different exception type."""
    for door in (
        lambda: jnp.array(300, dtype=jnp.int8),
        lambda: jnp.asarray(300, dtype=jnp.int8),
        lambda: jnp.int8(300),
    ):
        with pytest.raises(OverflowError):
            door()


def test_the_wrapper_forwards_verbatim_and_changes_no_result(unarmed):
    """A hook that changed a result would be worse than no hook.

    Every route driven with an IN-RANGE value, armed and unarmed, and the
    arrays compared including dtype and weak-typing — the last of which is the
    property most easily lost by re-supplying a default in a wrapper.
    """
    def snap():
        return {
            name: (
                np.asarray(route(44)).tolist(),
                str(route(44).dtype),
                getattr(route(44), "weak_type", None),
            )
            for name, route in ROUTES.items()
        }

    before = snap()
    status = _tripwire.arm_eager()
    if not status.armed:  # pragma: no cover - environment
        pytest.skip(status.code)
    try:
        after = snap()
    finally:
        _tripwire.disarm_eager()
    assert before == after


def test_the_denominator_counts_every_conversion_and_not_only_the_fires(armed):
    """"0 truncations" is what a dead hook reports too."""
    eager.reset_counters()
    for _ in range(5):
        jnp.full((), 44, jnp.int8)
    assert eager.CONVERSIONS >= 5
    assert eager.TRUNCATIONS == 0
    lines = " ".join(report.render_eager(armed, eager.snapshot()))
    assert "scalar integer conversion(s) observed" in lines


#: Third-party workloads, RUN rather than imported. The blast-radius claim
#: that matters is not "importing a library does not fire" -- almost nothing
#: happens at import -- it is that a real computation does not. Each entry is
#: a small but complete use of the library.
#:
#: NOT `importorskip`, and that is a decision rather than an oversight. A
#: library that is absent makes the workload UNMEASURED, not the test skipped:
#: `tests/test_skip_inventory.py` pins this suite's skip set by condition, and
#: eight new conditional skips would be eight undisclosed ones in every lane
#: that does not install flax and diffrax -- which is every lane in `ci.yml`
#: except the two acceptance jobs. So the runner below imports what it can,
#: measures what it imported, SAYS which ones it measured, and requires the
#: two that need nothing but jax to have run, so the test can never be
#: satisfied by having measured nothing.
def _optax_step():
    import optax
    opt = optax.adam(1e-3)
    p = {"w": jnp.ones((4,))}
    state = opt.init(p)
    grads = jax.grad(lambda q: (q["w"] ** 2).sum())(p)
    updates, state = opt.update(grads, state, p)
    return optax.apply_updates(p, updates)


def _flax_dense():
    import flax.linen as nn

    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(3)(x)

    model = M()
    variables = model.init(jax.random.key(0), jnp.ones((2, 4)))
    return model.apply(variables, jnp.ones((2, 4)))


def _diffrax_solve():
    import diffrax
    term = diffrax.ODETerm(lambda t, y, args: -y)
    return diffrax.diffeqsolve(
        term, diffrax.Tsit5(), t0=0.0, t1=1.0, dt0=0.1, y0=1.0
    ).ys


def _equinox_mlp():
    import equinox as eqx
    model = eqx.nn.MLP(3, 2, 8, 2, key=jax.random.key(0))
    return jax.vmap(model)(jnp.ones((5, 3)))


def _lineax_solve():
    import lineax as lx
    return lx.linear_solve(
        lx.MatrixLinearOperator(jnp.eye(3) * 2.0), jnp.ones((3,))
    ).value


def _jax_md_space():
    import jax_md
    displacement, _ = jax_md.space.periodic(10.0)
    positions = jax.random.uniform(jax.random.key(0), (8, 2)) * 10.0
    return jax_md.space.map_product(displacement)(positions, positions)


def _jax_random():
    key = jax.random.key(0)
    return (
        jax.random.normal(key, (5,)),
        jax.random.randint(key, (5,), 0, 255),
        jax.random.bits(key, (5,)),
        jax.random.permutation(key, 10),
    )


def _integer_bit_ops():
    x = jnp.arange(16, dtype=jnp.uint8)
    return x << 3, x >> 2, x & 0xF0, jnp.packbits(jnp.ones((16,), bool))


WORKLOADS = {
    "optax adam step": _optax_step,
    "flax nn.Dense": _flax_dense,
    "diffrax diffeqsolve": _diffrax_solve,
    "equinox MLP under vmap": _equinox_mlp,
    "lineax linear_solve": _lineax_solve,
    "jax_md periodic space": _jax_md_space,
    "jax.random incl. randint/bits": _jax_random,
    "uint8 shift/mask/packbits": _integer_bit_ops,
}


#: The two workloads that need nothing but jax, so they run in every lane
#: this file is collected in. Without them the test below could be satisfied
#: by an environment in which nothing was measured at all.
ALWAYS_AVAILABLE = ("jax.random incl. randint/bits", "uint8 shift/mask/packbits")


def test_real_work_does_not_fire(armed, capsys):
    """THE BLAST RADIUS, and it is the measurement the BaseException choice
    rests on.

    An alarm that inherits from ``BaseException`` is only affordable if it
    fires on almost nothing, and "almost nothing" has to be measured against
    real COMPUTATION rather than against imports -- almost nothing happens at
    import. Each entry runs a complete small use of a library.

    WHAT IT CLAIMS IS WHAT IT MEASURED. The libraries are optional and most
    lanes do not install them, so the ones that are absent are reported as
    unmeasured and the two that need only jax are REQUIRED to have run. A
    blast-radius test that quietly measured nothing would be the shape of
    control this repository keeps having to withdraw.
    """
    measured, unavailable, fired = [], [], []
    for name, workload in sorted(WORKLOADS.items()):
        eager.reset_counters()
        try:
            workload()
        except ImportError:
            unavailable.append(name)
            continue
        except stelling.EagerTruncationError as exc:
            fired.append((name, exc.written, exc.became, exc.to_dtype))
            continue
        measured.append((name, eager.CONVERSIONS, eager.TRUNCATIONS))

    print(f"measured: {[m[0] for m in measured]}")
    print(f"not installed, so UNMEASURED: {unavailable}")
    assert not fired, (
        f"real third-party work produced undeclared truncations: {fired}. If "
        "these are real, the library writes an out-of-range constant into a "
        "narrow dtype; if they are not, the detector has a false positive "
        "and is unshippable."
    )
    assert all(count == 0 for _, _, count in measured), measured
    ran = {name for name, _, _ in measured}
    assert set(ALWAYS_AVAILABLE) <= ran, (
        f"the workloads that need only jax did not run ({sorted(ran)}), so "
        "this measured nothing and proves nothing"
    )
    assert sum(conversions for _, conversions, _ in measured) > 0, (
        "no workload reached the hook at all"
    )


def test_the_detector_needs_no_origin_filter_and_the_tripwire_does(armed):
    """The asymmetry that makes an alarm this loud affordable at all.

    The const-fold tripwire fires on JAX'S OWN constants —
    ``jax.random.key(0)`` folds ``4294967295 -> -1`` inside ``threefry2x32``
    — and carries ``record.attribute``, an ``ORIGIN_JAX`` bucket and a
    suppressed-findings section to keep that out of a user's results. This
    detector never sees it, so it needs none of that: an instrument that
    RAISED inside jax's own PRNG would be unusable at any blast radius.

    Both halves are driven, because the interesting claim is the contrast. If
    the tripwire ever stops firing there, this test says so rather than
    quietly becoming a statement about nothing.
    """
    eager.reset_counters()
    # EVICT FIRST, both halves. `jax.random.key` is jitted, so on a warm cache
    # it is replayed and neither hook is reached -- which is B15's finding
    # applied to a test rather than to a verdict, and would make BOTH halves
    # of this test vacuous in the same direction.
    assert _tripwire.evict_trace_caches() == "evicted"
    for i in range(1, 6):
        key = jax.random.key(i)
        jax.random.split(key, 3)
        jax.random.randint(key, (4,), 0, 255)
        jax.random.bits(key, (4,))
    assert eager.TRUNCATIONS == 0, "the eager detector fired inside jax's PRNG"
    assert eager.CONVERSIONS > 0, "it saw nothing at all, so it proves nothing"

    # ...and the control, which is the other instrument on the same program
    from stelling._tripwire import record as _record

    fold_status, recorder = _tripwire.arm()
    if not fold_status.armed:  # pragma: no cover - environment
        pytest.skip(fold_status.code)
    try:
        assert _tripwire.evict_trace_caches() == "evicted"
        before = recorder.suppressed_jax
        jax.random.key(0)
        assert recorder.suppressed_jax > before, (
            "the const-fold tripwire no longer fires on jax's own PRNG mask, "
            "so the contrast this test is about has gone away and the "
            "attribution machinery it justifies should be re-examined"
        )
        suppressed = recorder.sorted_suppressed()[0]
        assert suppressed.origin == _record.ORIGIN_JAX
        assert (suppressed.written, suppressed.became) == (4294967295, -1)
    finally:
        _tripwire.disarm()


# ---------------------------------------------------------------------------
# 5. Fail closed
# ---------------------------------------------------------------------------


def test_the_route_list_the_selfcheck_drives_covers_the_routes_this_file_drives():
    """The self-check's claim and the test's claim are the same claim.

    A self-check that drove one route while the tool advertised six would be
    exactly the "attached but blind" failure it exists to catch, one level up.
    """
    driven = {name for name, _ in adapter._eager_routes()}
    assert {"jnp.full", "jnp.full_like", "lax.full", "lax.full_like",
            "lax.convert_element_type"} <= driven, driven
    assert any("numpy" in name for name in driven), (
        "the self-check no longer drives the NumPy-scalar branch, which is a "
        "different branch inside jax and not a different spelling"
    )


def test_a_signature_that_moved_refuses_to_attach(unarmed):
    """The half a presence check misses.

    ``eager_locate()`` answers "is there a function called that"; a release
    that reordered the parameters leaves it answering yes while the hook reads
    a ``sharding`` where it expects a dtype. A hook that reported the wrong
    line is worse than no hook, so it refuses.
    """
    assert adapter.eager_detach("signature") == "detached"
    try:
        assert adapter.eager_signature_check() != "ok"
        status = _tripwire.arm_eager()
        assert status.code == "signature-drift", status
        assert "operand" in status.detail or "new_dtype" in status.detail
        assert "refuses to attach" in status.meaning
        assert not eager.is_armed()
    finally:
        adapter.eager_reattach()
    # the control: the same call on the same jax attaches
    again = _tripwire.arm_eager()
    assert again.armed, "the control did not attach, so the refusal is not news"
    _tripwire.disarm_eager()


def test_a_route_going_blind_refuses_to_attach_and_names_the_route(unarmed):
    """The failure this design turns on, and the only one that is silent.

    A jax release that stopped routing one construction spelling through the
    site leaves the attribute there and the wrapper installed. Keeping five
    routes and losing one quietly is not a trade this tool makes on a user's
    behalf.
    """
    status = _tripwire.arm_eager()
    if not status.armed:  # pragma: no cover - environment
        pytest.skip(status.code)
    try:
        assert adapter.eager_detach("bypass") == "detached"
        probe = adapter.eager_selfcheck()
        assert probe.startswith("route-blind:"), probe
        assert probe.split(":", 1)[1] in {n for n, _ in adapter._eager_routes()}
        blind = _tripwire.Status(code=probe)
        assert "no longer reaches it" in blind.meaning, (
            "a `route-blind:<route>` code does not find its explanation, so "
            "the one fact a reader needs arrives with the generic fallback"
        )
    finally:
        adapter.eager_reattach()
        _tripwire.disarm_eager()


def test_a_rebind_over_the_top_is_reported_and_not_clobbered(unarmed):
    status = _tripwire.arm_eager()
    if not status.armed:  # pragma: no cover - environment
        pytest.skip(status.code)
    try:
        assert _tripwire.eager_live_check() == "armed"
        assert _tripwire.displaced() == ()
        assert adapter.eager_detach("rebind") == "detached"
        assert _tripwire.eager_live_check() == "foreign-patch"
        assert _tripwire.displaced() == ("eager",)
        assert _tripwire.disarm_eager() == "foreign-patch", (
            "disarming clobbered whatever replaced us instead of saying so"
        )
    finally:
        adapter.eager_reattach()
    assert not eager.is_armed()


def test_a_detach_whose_saved_entry_is_our_wrapper_is_fixed_up_by_the_restore(unarmed):
    """The orphan, and it is a measurement rather than a defensive branch.

    ``eager_detach`` saves whatever the attribute held; ``disarm_eager`` may
    then retire that very wrapper. Without the fix-up in ``eager_restore``,
    ``eager_reattach`` binds an orphaned stelling wrapper as the live function
    -- one no record owns, whose observer lookup finds nothing, so it watches
    nothing -- and every later ``arm_eager()`` in the process reads ITS
    ``*args, **kwargs`` signature as jax's and answers ``signature-drift``.
    Driven when this file was written: four later tests skipped with that
    code against a jax whose function had not moved.
    """
    import importlib

    module = importlib.import_module(adapter.EAGER_MODULE)
    jaxs_own = getattr(module, adapter.EAGER_ATTR)
    status = _tripwire.arm_eager()
    if not status.armed:  # pragma: no cover - environment
        pytest.skip(status.code)
    try:
        adapter.eager_detach("rebind")
        assert _tripwire.disarm_eager() == "foreign-patch"
    finally:
        adapter.eager_reattach()
    assert getattr(module, adapter.EAGER_ATTR) is jaxs_own, (
        "reattaching restored an orphaned stelling wrapper as the live "
        "function"
    )
    assert adapter.eager_signature_check() == "ok"
    again = _tripwire.arm_eager()
    assert again.armed, again.code
    _tripwire.disarm_eager()


def test_the_displacement_check_covers_BOTH_hooks_in_one_instrument(unarmed):
    """B15's finding and B16's hook, answered by one function.

    Two displacement instruments — one per hook, each consulted by a different
    caller — is two chances to teach one caller about one hook and forget the
    other. That is not hypothetical: ``live_check()`` existed, cost nothing,
    and was consulted nowhere on the trace gate's path, which is exactly how a
    displaced const-fold hook produced a VERIFIED on a watched route.
    """
    fold_status, _ = _tripwire.arm()
    if not fold_status.armed:  # pragma: no cover - environment
        pytest.skip(fold_status.code)
    eager_status = _tripwire.arm_eager()
    try:
        if not eager_status.armed:  # pragma: no cover - environment
            pytest.skip(eager_status.code)
        assert _tripwire.displaced() == ()
        assert dict(adapter.displacement_check()) == {
            "const-fold": "armed", "eager": "armed"
        }
        adapter.detach("bypass")
        try:
            assert _tripwire.displaced() == ("const-fold",)
        finally:
            adapter.reattach()
        adapter.eager_detach("bypass")
        try:
            assert _tripwire.displaced() == ("eager",)
        finally:
            adapter.eager_reattach()
        assert _tripwire.displaced() == ()
    finally:
        _tripwire.disarm_eager()
        _tripwire.disarm()


def test_the_running_release_has_a_row_in_the_eager_hash_map(unarmed):
    """Same discipline as ``_KNOWN_HASHES``, and the same deliberate cost.

    A release with no row goes RED here rather than quietly recording an
    unread hash. The remedy is manual and is meant to be: somebody reads the
    function, diffs it against the nearest row, and writes down what moved.
    A nightly is not a release and takes the never-read state instead.
    """
    status = _tripwire.arm_eager()
    try:
        if not status.armed:  # pragma: no cover - environment
            pytest.skip(status.code)
        assert status.rule_hash and len(status.rule_hash) == 12
        assert status.rule_name == "_convert_element_type"
        if not adapter.is_release(status.jax_version):
            assert status.hash_state == "never-read"
            return
        assert status.known_hash is not None, (
            f"jax {status.jax_version} has never been READ at the eager "
            "narrowing site. Read the function, diff it against the nearest "
            "row in `_adapter_jax._KNOWN_EAGER_HASHES`, and add a row naming "
            "what changed -- do not copy the observed hash in."
        )
        assert status.hash_state == "as-tested", (
            f"jax {status.jax_version} records {status.known_hash} for the "
            f"eager site and is reporting {status.rule_hash}"
        )
    finally:
        _tripwire.disarm_eager()


def test_the_two_hash_maps_are_keyed_on_releases_and_are_not_the_same_map():
    """Two sites, two maps, and the reason is measurable rather than tidy.

    The two functions move on DIFFERENT releases: 0.10.2 and 0.11.0 are
    byte-identical at the const-fold rule and differ at the eager site;
    0.11.0 and 0.11.1 are the other way round. One shared "jax internals
    moved" flag would be wrong about one of the two on every release either
    of them changed.
    """
    strays = [
        k for k in adapter._KNOWN_EAGER_HASHES if not adapter.is_release(k)
    ]
    assert not strays, f"_KNOWN_EAGER_HASHES is keyed on releases: {strays}"
    assert set(adapter._KNOWN_EAGER_HASHES) >= {"0.10.2", "0.11.0"}
    assert (
        adapter._KNOWN_EAGER_HASHES["0.10.2"]
        != adapter._KNOWN_EAGER_HASHES["0.11.0"]
    ), "the eager site is recorded as identical on 0.10.2 and 0.11.0"
    assert adapter._KNOWN_HASHES["0.10.2"] == adapter._KNOWN_HASHES["0.11.0"], (
        "the const-fold rule is recorded as differing on 0.10.2 and 0.11.0, "
        "which is the fact the sentence above is contrasted against"
    )


def test_a_jax_that_is_not_installed_is_reported_as_no_module(unarmed, monkeypatch):
    """The zero-dep lane's answer, driven through the seam that decides it.

    A subprocess cannot produce this one faithfully: hiding jax with a
    ``meta_path`` hook that RAISES makes ``importlib.util.find_spec`` raise
    rather than answer, which is a different environment from one where jax
    was never installed. The test below drives what the subprocess CAN say --
    that arming never raises and never reports armed -- and this drives the
    code itself.
    """
    from stelling import _optional

    monkeypatch.setattr(_optional, "available", lambda name: False)
    status = _tripwire.arm_eager()
    assert status.code == "no-module"
    assert not status.armed
    assert "jax is not installed" in status.explanation
    assert "Static checking is unaffected" in status.explanation


def test_arming_in_a_process_that_cannot_import_jax_never_raises():
    program = textwrap.dedent(
        """
        import sys


        class _NoJax:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("jax", "jaxlib"):
                    raise ModuleNotFoundError(name)
                return None


        sys.meta_path.insert(0, _NoJax())
        from stelling import _tripwire
        status = _tripwire.arm_eager()
        assert not status.armed, status.code
        assert _tripwire.disarm_eager() == "not-armed"
        assert _tripwire.eager_live_check() == "detached"
        assert _tripwire.displaced() == ()
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True, text=True, check=False,
        env={**_env(), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# ---------------------------------------------------------------------------
# 6. The default path
# ---------------------------------------------------------------------------


def test_importing_stelling_patches_nothing_at_all():
    """Opt-in, asserted against the private attribute rather than the flag.

    A subprocess, because the truth being measured is about a fresh process:
    in this one the fixtures above have armed and disarmed the hook a dozen
    times, and "it is not armed right now" is a weaker statement than "nothing
    ever touched it".
    """
    program = textwrap.dedent(
        """
        import importlib
        import stelling
        from stelling._tripwire import eager
        L = importlib.import_module("jax." + "_src.lax.lax")
        marker = L._convert_element_type
        assert marker.__module__.startswith("jax"), marker.__module__
        assert "stelling" not in getattr(marker, "__qualname__", "")
        assert not eager.is_armed()
        assert eager.CONVERSIONS == 0 and eager.TRUNCATIONS == 0
        import jax.numpy as jnp
        assert int(jnp.full((), 300, jnp.int8)) == 44
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True, text=True, check=False,
        env={**_env(), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_disarming_restores_jax_s_own_function_by_identity(unarmed):
    import importlib

    module = importlib.import_module(adapter.EAGER_MODULE)
    before = getattr(module, adapter.EAGER_ATTR)
    status = _tripwire.arm_eager()
    if not status.armed:  # pragma: no cover - environment
        pytest.skip(status.code)
    assert getattr(module, adapter.EAGER_ATTR) is not before
    assert _tripwire.disarm_eager() == "restored"
    assert getattr(module, adapter.EAGER_ATTR) is before, (
        "disarming left something other than jax's own function behind"
    )


def test_arming_twice_does_not_double_wrap(unarmed):
    first = _tripwire.arm_eager()
    if not first.armed:  # pragma: no cover - environment
        pytest.skip(first.code)
    try:
        assert adapter.eager_install(eager.observe) == "already-armed"
        eager.reset_counters()
        with pytest.raises(stelling.EagerTruncationError):
            jnp.full((), 300, jnp.int8)
        assert eager.TRUNCATIONS == 1, (
            "a second wrapper over the first counted the same conversion twice"
        )
    finally:
        _tripwire.disarm_eager()
        eager.reset_counters()
