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

import asyncio
import contextlib
import dataclasses
import functools
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


def test_a_declaration_at_a_DIFFERENT_DTYPE_is_CAUGHT_or_SILENT_and_both_happen(armed):
    """The claim used to be "it cannot license a different dtype". Measured.

    Sometimes the drifted declaration is caught: ``intentional_wrap(0xFF,
    "int8")`` is ``-1``, which ``uint8`` cannot hold, so the detector fires
    on the declared value. Often it is not: ``intentional_wrap(300, "int8")``
    is ``44``, which every other integer dtype holds.

    WHAT SURVIVES IS THE SAFETY PROPERTY AND NOT THE DETECTION, and the third
    assertion is the one that matters: in every silent case the value written
    at the new site is IN RANGE there, so no narrowing happens and there is no
    truncation for the declaration to have hidden. What a drifted declaration
    can still do is write the wrong constant, which this instrument does not
    claim to catch.
    """
    caught_case = stelling.intentional_wrap(0xFF, "int8")
    assert caught_case == -1
    with pytest.raises(stelling.EagerTruncationError) as caught:
        jnp.full((), caught_case, jnp.uint8)
    assert caught.value.written == -1 and caught.value.to_dtype == "uint8"

    silent_case = stelling.intentional_wrap(300, "int8")
    assert silent_case == 44
    # The 64-bit names are left out of the CONSTRUCTION half and only of that
    # half: under the default `jax_enable_x64=False` jax canonicalises them to
    # 32 bits and warns, which is jax telling the truth about a different
    # thing and not what this test is measuring. The arithmetic below covers
    # all eight.
    for other in ("int16", "int32", "uint8", "uint16", "uint32"):
        assert int(jnp.full((), silent_case, jnp.dtype(other))) == 44, other

    # ...and the ratio, over this file's own grid against the seven other
    # dtypes, so that a change to either half shows up as a number.
    silent = sum(
        1
        for value, dtype in WRAP_GRID
        for other in record.INT_DTYPES
        if other != dtype and record.in_range(record.narrow(value, dtype), other)
    )
    pairs = len(WRAP_GRID) * (len(record.INT_DTYPES) - 1)
    assert (pairs, silent) == (98, 53), (
        f"{silent} of {pairs} (declaration, misuse) pairs pass silently; "
        "eager.py's docstring quotes these two numbers"
    )
    # ...AND THE TWO HALVES ARE MEASURED AGAINST JAX, not restated. What
    # stood here was `assert all(in_range(...) for ... if in_range(...))` --
    # an assertion over the set its own condition defines, which cannot fail
    # and therefore measured nothing. The claim the ruling actually rests on
    # is that in every SILENT case the site performs no narrowing at all, and
    # the only witness for that is the array jax builds: it has to hold the
    # declared value, and the armed detector has to say nothing about it.
    # The CAUGHT half is driven in the same loop, because "both happen" is
    # half of this test's name.
    #
    # The 64-bit dtypes are out of the construction half for the reason given
    # above; the arithmetic ratio just asserted covers all eight.
    constructible = ("int8", "uint8", "int16", "uint16", "int32", "uint32")
    silent_pairs = caught_pairs = 0
    for value, dtype in WRAP_GRID:
        wrapped = record.narrow(value, dtype)
        for other in constructible:
            if other == dtype:
                continue
            fired = None
            built = None
            try:
                built = int(jnp.full((), wrapped, jnp.dtype(other)))
            except stelling.EagerTruncationError as exc:
                fired = exc
            if record.in_range(wrapped, other):
                silent_pairs += 1
                assert fired is None, (
                    f"declaring {value} for {dtype} and misusing it at {other} "
                    f"fired, and {wrapped} is in range there"
                )
                assert built == wrapped, (
                    f"jax made {built} of {wrapped} at {other}, so the site "
                    "DID narrow and the silence hid a truncation"
                )
            else:
                caught_pairs += 1
                assert fired is not None, (
                    f"declaring {value} for {dtype} and misusing it at {other} "
                    f"was silent, and {wrapped} does not fit {other}"
                )
                assert fired.written == wrapped and fired.to_dtype == other
    assert (silent_pairs, caught_pairs) == (34, 38), (
        f"{silent_pairs} silent and {caught_pairs} caught over the six "
        "constructible dtypes"
    )


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


def test_a_region_ends_where_the_block_ends_in_straight_line_code(armed):
    """It licenses the block and the next line is on its own again.

    THE STRAIGHT-LINE CASE ONLY, and it used to be called
    ``test_a_region_is_lexically_bounded`` while the module said "lexically
    bounded" in three places. It is not lexically bounded and no context
    manager can be; the test below is the one that measures what it is.
    """
    with expected_truncation("driving the region's boundary"):
        assert int(jnp.full((), 300, jnp.int8)) == 44
    with pytest.raises(stelling.EagerTruncationError):
        jnp.full((), 300, jnp.int8)


def test_a_region_is_DYNAMICALLY_scoped_and_says_so_in_all_three_directions(armed):
    """What "lexically bounded" was claiming, measured in the three ways it fails.

    A ``with`` block LOOKS lexical and a context manager is dynamic: the
    region is open from ``__enter__`` to ``__exit__`` in whatever code runs
    between them. Threads and asyncio tasks are isolated because the stack is
    a :mod:`contextvars` variable; generators are not isolated by anything
    Python offers, so that one is disclosed rather than claimed away.
    """
    seen = {}

    def wraps(label):
        try:
            jnp.full((), 300, jnp.int8)
            seen[label] = "licensed"
        except stelling.EagerTruncationError:
            seen[label] = "raised"

    # 1. A GENERATOR SUSPENDED INSIDE A REGION licenses its resumer. This is
    #    the residue: a plain generator shares its caller's context, so the
    #    region it entered is open in the code that called `next()`.
    def suspends():
        with expected_truncation("a region a generator opens and does not close"):
            yield 1
            yield 2

    generator = suspends()
    next(generator)
    wraps("caller while a generator holds the region")
    for _ in generator:
        pass
    wraps("after the generator ran to completion")
    assert seen["caller while a generator holds the region"] == "licensed", (
        "the generator residue has gone away -- if a Python release fixed "
        "this, the disclosure in eager.py and report.EAGER_UNCOVERED should "
        "go with it"
    )
    assert seen["after the generator ran to completion"] == "raised"

    # 2. ANOTHER THREAD is not licensed. A `threading.local` got this right
    #    and a `contextvars.ContextVar` still does: a new thread starts in a
    #    fresh context.
    with expected_truncation("this thread is demonstrating the truncation"):
        thread = threading.Thread(target=wraps, args=("another thread",))
        thread.start()
        thread.join()
    assert seen["another thread"] == "raised", (
        "a region on one thread licensed a truncation on another"
    )

    # 3. ANOTHER ASYNCIO TASK on the same loop is not licensed -- and this is
    #    the half a `threading.local` got WRONG: one thread runs every task,
    #    so a thread-local region held across an `await` licensed all of them.
    async def holder():
        with expected_truncation("a region held across an await"):
            await asyncio.sleep(0.02)

    async def bystander():
        await asyncio.sleep(0.005)
        wraps("another task on the same loop")

    async def both():
        await asyncio.gather(holder(), bystander())

    asyncio.run(both())
    assert seen["another task on the same loop"] == "raised", (
        "a region held by one asyncio task licensed a truncation in another"
    )

    # 4. AND ONE INSTANCE RE-ENTERED pops what it pushed. The list-and-pop
    #    this replaced handled that; a single reset token would lose the outer
    #    one on the inner `__enter__` and leave the outer entry open forever.
    reused = expected_truncation("one instance, entered twice")
    with reused:
        with reused:
            jnp.full((), 300, jnp.int8)
        jnp.full((), 301, jnp.int8)
    with pytest.raises(stelling.EagerTruncationError):
        jnp.full((), 302, jnp.int8)


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


#: Every ``jax.random`` entry point the U1 counterexample reaches, as a
#: callable of a key. Driven with ``jit`` on AND off, because the claim this
#: replaces was measured only with it on -- which is the one configuration
#: where it holds.
PRNG_ENTRY_POINTS = {
    "key": lambda: jax.random.key(3),
    "PRNGKey": lambda: jax.random.PRNGKey(3),
    "split": lambda: jax.random.split(jax.random.key(3), 3),
    "randint": lambda: jax.random.randint(jax.random.key(3), (4,), 0, 255),
    "bits": lambda: jax.random.bits(jax.random.key(3), (4,)),
    "uniform": lambda: jax.random.uniform(jax.random.key(3), (4,)),
    "normal": lambda: jax.random.normal(jax.random.key(3), (4,)),
    "permutation": lambda: jax.random.permutation(jax.random.key(3), 8),
    "choice": lambda: jax.random.choice(jax.random.key(3), 8, (2,)),
    "categorical": lambda: jax.random.categorical(jax.random.key(3), jnp.zeros(3)),
    "fold_in": lambda: jax.random.fold_in(jax.random.key(3), 7),
    "gumbel": lambda: jax.random.gumbel(jax.random.key(3), (4,)),
}


@pytest.mark.parametrize("jit_on", [True, False], ids=["jit-on", "disable_jit"])
def test_the_origin_filter_keeps_JAX_S_OWN_constants_out_of_the_alarm(armed, jit_on):
    """U1. The alarm must not raise inside jax's own PRNG, jit on OR off.

    THIS TEST USED TO DRIVE ONLY THE FIRST HALF OF THIS PARAMETRISATION, and
    it was called ``test_the_detector_needs_no_origin_filter_and_the_tripwire
    _does``. With ``jit`` on, the threefry mask reaches the const-fold site as
    a traced constant and never reaches this one, so "0 truncations" was true
    and the conclusion drawn from it -- that this hook needs no origin filter
    at all -- was not.

    Under ``jax.disable_jit()`` jax evaluates ``bitwise_and(seed,
    uint32(0xFFFFFFFF))`` eagerly and the mask arrives here as a written
    scalar. Before ``eager._origin`` existed, every entry point below raised
    ``EagerTruncationError(4294967295 -> -1, int32)`` at the user's own
    ``jax.random`` line, telling them to declare a constant they never wrote;
    for flax nnx the line was inside ``flax/nnx/rnglib.py``, which they cannot
    edit. ``jax.disable_jit()`` is jax's own documented debugging workflow and
    is what ``chex.fake_jit()`` and ``chex.fake_pmap_and_jit()`` install.

    What the filter does with them is COUNT them, which the second half
    asserts: a suppression nobody can see is the silence this module exists to
    end, one level up.
    """
    eager.reset_counters()
    # EVICT FIRST. `jax.random.key` is jitted, so on a warm cache it is
    # replayed and the hook is not reached -- B15's finding applied to a test,
    # and it would make this vacuous in the direction that passes.
    assert _tripwire.evict_trace_caches() == "evicted"
    with contextlib.nullcontext() if jit_on else jax.disable_jit():
        for name, entry in PRNG_ENTRY_POINTS.items():
            try:
                entry()
            except stelling.EagerTruncationError as exc:  # pragma: no cover
                pytest.fail(
                    f"jax.random.{name} raised inside jax's own PRNG: "
                    f"{exc.written} -> {exc.became} ({exc.to_dtype}) attributed "
                    f"to {exc.file}:{exc.line}, which is not a line that wrote it"
                )
    assert eager.CONVERSIONS > 0, "the hook saw nothing at all, so this proves nothing"
    if not jit_on:
        assert eager.SUPPRESSED_JAX > 0, (
            "with jit off jax narrows its own threefry mask eagerly and this "
            "hook must SEE it -- 0 suppressions here means the measurement "
            "stopped happening, not that the problem went away"
        )
        assert eager.snapshot()["suppressed"], "suppressed but not sited"
        lines = " ".join(
            report.render_eager(_tripwire.Status(code="armed"), eager.snapshot())
        )
        assert "written BY JAX ITSELF" in lines
        assert "4294967295 -> -1 (int32)" in lines


#: The basis the ``jit``-independence claim is driven over. It was FOUR
#: programs and four was too few: widened to these, an audit found one that
#: genuinely flipped -- ``jit(partial(jnp.full_like, fill_value=300))(x)``
#: gave no alarm with ``jit`` on and raised with it off, on the SAME observed
#: conversion, because the predicate of the day keyed on the identity of "the
#: outermost jax frame" and how many wrapper frames jax installs is exactly
#: what ``jit`` changes.
#:
#: The carriers are the point of the list. A constant reaches jax through a
#: ``functools.partial``, a ``jax.tree_util.Partial``, a bound method, a
#: closure cell and a registered-dataclass pytree carry, under every
#: higher-order entry point jax offers -- ``jit``, ``vmap``, ``tree.map``,
#: ``lax.map``, ``lax.scan`` and ``lax.fori_loop`` -- and NONE of them is a
#: constant the caller can be said not to have written.
def _jit_equality_basis():
    i8, u8 = jnp.int8, jnp.uint8
    flat = jnp.zeros((2,), i8)
    batched = jnp.zeros((3, 2), i8)
    tree = {"a": jnp.zeros((2,), i8)}
    partial300 = functools.partial(jnp.full_like, fill_value=300)

    class Maker:
        """A user callable carrying the constant on itself."""

        def __init__(self, value):
            self.value = value

        def __call__(self, leaf):
            return jnp.full_like(leaf, self.value)

    def scan_body(carry, _):
        return _ScanCarry(jnp.full_like(carry.value, carry.fill), carry.fill), None

    return (
        # --- the four the fix was originally sold on ---
        ("jax's own: random.key", "no alarm", lambda: jax.random.key(11)),
        ("yours: full 300 int8", "raised 300 -> 44",
         lambda: jnp.full((2,), 300, i8)),
        ("yours: fill_value=-1 uint8", "raised -1 -> 255",
         lambda: jnp.full((2,), -1, u8)),
        ("in range: full 44 int8", "no alarm", lambda: jnp.full((2,), 44, i8)),
        # --- the carriers, under every higher-order entry point ---
        ("jit(partial)", "raised 300 -> 44", lambda: jax.jit(partial300)(flat)),
        ("vmap(partial)", "raised 300 -> 44", lambda: jax.vmap(partial300)(batched)),
        ("tree.map(partial)", "raised 300 -> 44",
         lambda: jax.tree.map(partial300, tree)),
        ("lax.map(partial)", "raised 300 -> 44",
         lambda: lax.map(functools.partial(jnp.full_like, fill_value=300, dtype=i8),
                         batched)),
        ("tree.map(tree_util.Partial)", "raised 300 -> 44",
         lambda: jax.tree.map(
             jax.tree_util.Partial(jnp.full_like, fill_value=300), tree)),
        ("tree.map(bound method)", "raised 300 -> 44",
         lambda: jax.tree.map(Maker(300).__call__, tree)),
        ("tree.map(callable object)", "raised 300 -> 44",
         lambda: jax.tree.map(Maker(300), tree)),
        ("lax.scan(dataclass carry)", "raised 300 -> 44",
         lambda: lax.scan(scan_body, _ScanCarry(jnp.zeros((2,), i8)), None, length=2)),
        ("lax.fori_loop(closure)", "raised 300 -> 44",
         lambda: lax.fori_loop(0, 2, lambda i, c: jnp.full_like(c, 300),
                               jnp.zeros((2,), i8))),
        ("jit(lambda writing 300)", "raised 300 -> 44",
         lambda: jax.jit(lambda x: jnp.full((2,), 300, i8))(flat)),
        # --- pytrees big enough to have exhausted the old scan's budget ---
        ("tree.map(PRNGKey, five levels deep)", "no alarm",
         lambda: jax.tree.map(jax.random.PRNGKey, (((((0,),),),),))),
        ("tree.map(PRNGKey, 9000 leaves)", "no alarm",
         lambda: jax.tree.map(jax.random.PRNGKey, list(range(9000)))),
        # --- jax's mask and the caller's seed, the same integer ---
        ("random.PRNGKey(2**32 - 1)", "no alarm",
         lambda: jax.random.PRNGKey(2 ** 32 - 1)),
        # --- narrowings reached by their own routes ---
        ("arange(2**40, 2**40+3)", "raised 1099511627776 -> 0",
         lambda: jnp.arange(2 ** 40, 2 ** 40 + 3)),
        ("at[9].get(fill_value=-1) uint8", "raised -1 -> 255",
         lambda: jnp.arange(4, dtype=u8).at[9].get(mode="fill", fill_value=-1)),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _ScanCarry:
    """A registered-dataclass pytree carrying a constant as a STATIC field.

    The audit's U1 named this shape: the constant never appears in the
    arguments of the jax call, so a rule that scanned those arguments
    suppressed the narrowing it produces.
    """

    value: object
    fill: int = dataclasses.field(default=300, metadata={"static": True})


def test_the_origin_answer_does_not_depend_on_a_trace_being_in_progress(armed):
    """The property the design is built on, driven as an equality on 19 programs.

    THE CLAIM IS NARROWER THAN THE ONE IT REPLACES, and that is the point.
    What stood here was *"a call boundary exists whether or not a trace is in
    progress, which is why this answer does not depend on ``jit``"* -- and it
    was false, measured: ``jit(partial(jnp.full_like, fill_value=300))(x)``
    gave no alarm with ``jit`` on and raised with it off, on the same observed
    conversion, because which frame is "the outermost jax frame" depends on
    how many wrapper frames jax installs and that is what ``jit`` changes.

    What is true of the design that replaced it: the verdict is a function of
    the written VALUE, the target DTYPE, and WHICH jax functions are in the
    unbroken run of jax frames beneath the caller. ``jit`` changes none of the
    three -- the decision is an existence test over that run, not a question
    about which of its frames is outermost.

    WHAT ``jit`` DOES CHANGE, and it is a different sentence: which narrowings
    happen eagerly at all. With ``jit`` on, jax's threefry mask is traced and
    never arrives here, so there is nothing to attribute; with it off it
    arrives and is attributed. That is a statement about the POPULATION of
    observations, not about the verdict on a member of it, and this test
    measures the second.
    """
    basis = _jit_equality_basis()

    def verdicts():
        out = {}
        for name, _expected, body in basis:
            assert _tripwire.evict_trace_caches() == "evicted"
            try:
                body()
                out[name] = "no alarm"
            except stelling.EagerTruncationError as exc:
                out[name] = f"raised {exc.written} -> {exc.became}"
        return out

    with_jit = verdicts()
    with jax.disable_jit():
        without_jit = verdicts()
    differ = {k: (with_jit[k], without_jit[k])
              for k in with_jit if with_jit[k] != without_jit[k]}
    assert not differ, f"the origin decision changed when jit went away: {differ}"
    # ...and the verdicts are the RIGHT ones, not merely equal: a rule that
    # suppressed everything would pass the equality above with 19 "no alarm"s.
    assert with_jit == {name: expected for name, expected, _ in basis}


def test_the_carriers_that_a_DATA_SCAN_missed_all_raise(armed):
    """U1, the regression that made the enumeration necessary, driven.

    A general predicate over the boundary call's arguments -- *"is the
    narrowed integer among the arguments of the call that crossed out of
    non-jax code into jax?"* -- cannot see a constant carried in a
    ``functools.partial``, a ``jax.tree_util.Partial``, a bound method, a
    closure cell or a registered-dataclass pytree, because none of those puts
    the integer in the argument list. It therefore SUPPRESSED them: a missed
    narrowing, in the DEFAULT ``jit``-on configuration, on idiomatic jax, and
    a regression against the tree before the filter existed.

    Every one of them is a constant the caller really wrote, so every one of
    them must raise. This is the same set the equality above drives; it is
    asserted separately because "they agree with each other" and "they are
    right" are different claims and only the second one is soundness.
    """
    carried = [
        (name, expected, body)
        for name, expected, body in _jit_equality_basis()
        if expected.startswith("raised 300")
    ]
    assert len(carried) >= 9, "the carrier basis shrank"
    for name, _expected, body in carried:
        assert _tripwire.evict_trace_caches() == "evicted"
        with pytest.raises(stelling.EagerTruncationError) as caught:
            body()
        assert caught.value.written == 300, name


def test_the_user_s_own_constants_still_fire_with_jit_OFF(armed):
    """The other direction of U1: the filter must not have bought silence.

    Every route the detector claims, driven inside ``jax.disable_jit()``,
    where the version that raised on jax's PRNG at least raised on these too.
    ``a + 200`` is here because it is the door ``report.UNCOVERED`` says this
    detector closes for a scoped ``disable_jit`` block, and that claim is now
    the only thing standing between that bullet and a false one.
    """
    fired = {}
    with jax.disable_jit():
        for name, route in ROUTES.items():
            try:
                route(300)
                fired[name] = "SILENT"
            except stelling.EagerTruncationError as exc:
                fired[name] = (exc.written, exc.became)
        try:
            jnp.zeros((3,), jnp.int8) + 200
            fired["a + 200"] = "SILENT"
        except stelling.EagerTruncationError as exc:
            fired["a + 200"] = (exc.written, exc.became)
    assert fired["a + 200"] == (200, -56)
    assert all(value != "SILENT" for value in fired.values()), fired


#: Four narrowings a user really can write, spelled the way they write them,
#: and NONE of them is in ROUTES: `jnp.arange` and `.at[].get` reach the site
#: by their own paths, and the two huge-integer ones are narrowed by x64
#: canonicalisation rather than by a dtype the caller named. They are here as
#: the positive control on the origin filter -- a filter that suppressed these
#: would be a filter that had turned the detector off.
TRUE_POSITIVES = {
    "jnp.full((2,), 2**40)": (lambda: jnp.full((2,), 2**40), 2**40),
    "jnp.arange(2**40, 2**40 + 3)": (lambda: jnp.arange(2**40, 2**40 + 3), 2**40),
    "jnp.full((3,), -1, uint32)": (lambda: jnp.full((3,), -1, jnp.uint32), -1),
    "x.at[9].get(fill_value=-1) on uint8": (
        lambda: jnp.arange(10, dtype=jnp.uint8).at[9].get(mode="fill", fill_value=-1),
        -1,
    ),
}


@pytest.mark.parametrize("name", sorted(TRUE_POSITIVES))
@pytest.mark.parametrize("jit_on", [True, False], ids=["jit-on", "disable_jit"])
def test_a_constant_the_user_really_wrote_fires_with_jit_ON_or_OFF(armed, name, jit_on):
    """The origin filter must not have been bought with a missed narrowing."""
    body, written = TRUE_POSITIVES[name]
    with contextlib.nullcontext() if jit_on else jax.disable_jit():
        with pytest.raises(stelling.EagerTruncationError) as caught:
            body()
    assert caught.value.written == written
    assert caught.value.file == __file__, (
        f"attributed to {caught.value.file}:{caught.value.line} rather than "
        "to the line in this file that wrote it"
    )


# ---------------------------------------------------------------------------
# The enumeration itself: re-derived, driven at arm time, and its residue
# measured rather than asserted.
# ---------------------------------------------------------------------------


def test_the_enumeration_of_jax_s_own_eager_truncations_is_RE_DERIVED(armed):
    """THE CANARY. A jax release that adds a second one turns this red.

    ``_adapter_jax._JAX_EAGER_CONSTANTS`` is what decides a suppression, and a
    map is only as good as the reading behind it. So the reading is REDONE
    here rather than trusted: ``eager_jax_constant_sweep`` runs jax's own
    integer surface -- every key implementation and seed spelling, then
    ``jax.random``'s consumers and ``jnp``'s integer ops over six integer
    dtypes -- under ``jax.disable_jit()``, which is the mode in which jax
    evaluates its own constants eagerly and this hook can see them.

    EVERY VALUE THE SWEEP HANDS JAX IS IN RANGE, so every truncation it
    observes is jax's own. That is what makes the result comparable to the map
    by equality rather than by inspection:

    * ``unmatched`` must be empty -- jax performs no eager truncation this
      repository has not read. It is measured at 649-729 conversions (the
      figure moves with ``jax_enable_x64``, which changes how many promotions
      happen, not how many truncations do) and, at x64 off, **13 truncation
      events, all of them one row**: the threefry mask, identically on jax
      0.11.0 and 0.10.2.
    * every row in the map must have been EXERCISED -- a row that has stopped
      being real is a suppression waiting for a value that never comes, and
      the map is the thing this test exists to keep honest.

    IT IS A SAMPLE AND NOT A PROOF, and the design says so: it establishes
    that no second eager truncation exists on the surface it covers, not that
    none exists in jax. What covers the rest is the direction the design fails
    in, which the test below measures.

    THE DENOMINATOR IS ASSERTED FIRST. "0 unmatched" is also what a sweep that
    never reached the hook reports, and that is the shape of vacuous control
    this repository keeps having to withdraw.
    """
    assert _tripwire.evict_trace_caches() == "evicted"
    swept = adapter.eager_jax_constant_sweep()
    assert swept["code"] == "swept", swept
    assert swept["conversions"] > 500, (
        f"the sweep saw only {swept['conversions']} conversions, so it is "
        "measuring something other than jax's integer surface"
    )
    assert swept["unmatched"] == (), (
        "jax performs an eager truncation of its own that "
        "`_adapter_jax._JAX_EAGER_CONSTANTS` has no row for. Until somebody "
        "reads it and writes a row down, the detector attributes it to "
        "whoever called jax and RAISES there: " + repr(swept["unmatched"])
    )
    exercised = {(file, func) for file, func, _w, _d in swept["matched"]}
    if jax.config.jax_enable_x64:
        # x64 ON widens jax's mask to int64, which holds it, so there is no
        # truncation to attribute and nothing to exercise. That is not a hole
        # -- it is the row being unreachable rather than wrong -- and the
        # `unmatched` half above is the half that still means something here.
        assert not exercised
        return
    assert exercised == set(adapter._JAX_EAGER_CONSTANTS), (
        "a row in the map was not exercised by the sweep, so it is a "
        "suppression this repository can no longer show is real: "
        f"{sorted(set(adapter._JAX_EAGER_CONSTANTS) - exercised)}"
    )
    assert swept["truncations"] > 0


def test_an_UNENUMERATED_constant_of_jax_s_own_RAISES_rather_than_hiding(armed):
    """The residue of an enumeration, measured in the direction it fails.

    A map cannot know a row nobody has written. The question is what happens
    then, and the answer this design chose is the LOUD one: no row means the
    narrowing is the caller's, and the caller's narrowings raise. That costs a
    false alarm at a line inside jax on the day a release adds a constant --
    which is audit 1's finding, and it is the direction an instrument must
    fail in, because an over-report is visible to a reader holding the quoted
    line and a suppression is not.

    Driven by taking the one row away, which is exactly what a release that
    moved the site would do.
    """
    assert _tripwire.evict_trace_caches() == "evicted"
    saved = dict(eager._JAX_CONSTANTS)
    eager._JAX_CONSTANTS.clear()
    try:
        with jax.disable_jit():
            with pytest.raises(stelling.EagerTruncationError) as caught:
                jax.random.key(0)
    finally:
        eager._JAX_CONSTANTS.clear()
        eager._JAX_CONSTANTS.update(saved)
    assert (caught.value.written, caught.value.became) == (4294967295, -1)
    # ...and the message tells the reader who did NOT write it what to do,
    # which is the only thing that turns this into a row somebody adds.
    text = str(caught.value)
    assert "not one of the constants stelling records jax as writing ITSELF" in text
    assert "_threefry_seed()" in text, (
        "the alarm does not name the jax function that WROTE the constant, "
        "so a reader who did not write it has nothing to report. It is five "
        "frames below the narrowing, which is what eager.MAX_JAX_FRAMES is "
        "sized for"
    )
    # ...and with the row back, the same program is silent again.
    assert _tripwire.evict_trace_caches() == "evicted"
    with jax.disable_jit():
        jax.random.key(0)


def test_ARMING_drives_the_origin_decision_in_BOTH_directions(unarmed):
    """F3. The self-check never reaches ``observe``, so arming proves nothing
    about what decides a raise.

    ``eager_selfcheck`` swaps a collector in for the observer. That is
    deliberate -- it keeps a self-check out of the user's denominator -- and
    it means every route it drives bypasses :func:`eager.observe` and
    therefore :func:`eager._origin`. A detector whose origin rule suppressed
    everything passes every route probe there is.

    So arming also drives one narrowing of each origin through the live
    policy, and the status says which legs ran.
    """
    assert not eager._JAX_CONSTANTS, (
        "an unarmed process is carrying the map that decides suppressions"
    )
    status = _tripwire.arm_eager()
    try:
        assert status.armed, status.code
        assert "origin-checked" in (status.detail or ""), status.detail
        if not jax.config.jax_enable_x64:
            assert "both directions" in status.detail, (
                "the jax leg did not run, so arming did not establish that "
                "jax's own mask is still attributed to jax"
            )
        assert eager._JAX_CONSTANTS == adapter.jax_eager_constants()
    finally:
        _tripwire.disarm_eager()
    assert not eager._JAX_CONSTANTS, (
        "disarming left the map behind, so a process that believes nothing "
        "is attached would go on suppressing at jax's sites"
    )


def test_arming_REFUSES_when_the_jax_leg_of_that_control_stops_holding(unarmed, monkeypatch):
    """...and it refuses, rather than attaching with a broken attribution.

    The same argument ``route-blind`` refuses on. A map that no longer names
    jax's mask is a detector that is about to raise inside jax's own PRNG at a
    line the user cannot edit -- audit 1's finding, exactly -- and the remedy
    is one row. Refusing names the row; attaching would name a user.
    """
    if jax.config.jax_enable_x64:
        pytest.skip("with x64 on jax's mask does not narrow, so there is no leg")
    monkeypatch.setattr(adapter, "_JAX_EAGER_CONSTANTS", {})
    status = _tripwire.arm_eager()
    try:
        assert not status.armed, "it attached with no row for jax's own mask"
        assert status.code.startswith("origin-blind:jax-attributed-to-you"), status.code
        assert not eager.is_armed(), "it refused and left the hook attached"
        assert not eager._JAX_CONSTANTS, "a refused arm left its map loaded"
    finally:
        if eager.is_armed():  # pragma: no cover - defensive
            _tripwire.disarm_eager()


def test_arming_REFUSES_when_the_user_leg_of_that_control_stops_holding(unarmed, monkeypatch):
    """The other direction, and it is the one a suppress-everything rule takes.

    A rule that attributed the user's own constants to jax would be silent,
    which is the defect this whole instrument exists to end. Simulated by
    giving the map a row for the probe's own constant at the jax function the
    probe's construction goes through.
    """
    written = adapter.EAGER_OVER
    # The jax function `jnp.full((), 256, int8)` reaches -- read off the live
    # run rather than typed, so this cannot go stale against a jax release.
    seen = {}

    def spy(value, to_dtype):
        seen.setdefault("run", eager.jax_segment(1))

    probe_status = _tripwire.arm_eager()
    assert probe_status.armed, probe_status.code
    saved_observer = adapter._eager_installed.get("observer")
    adapter._eager_installed["observer"] = spy
    try:
        jnp.full((), written, jnp.int8)
    finally:
        adapter._eager_installed["observer"] = saved_observer
        _tripwire.disarm_eager()
    run = seen["run"]
    assert run, "no jax frame beneath a jnp.full: the walk is measuring nothing"

    monkeypatch.setattr(
        adapter,
        "_JAX_EAGER_CONSTANTS",
        {run[0]: ((written, adapter.EAGER_DTYPE, "a row that must not exist"),)},
    )
    status = _tripwire.arm_eager()
    try:
        assert not status.armed, "it attached with the user's own constant suppressed"
        assert status.code == "origin-blind:user-not-raised", status.code
        assert not eager.is_armed()
        assert not eager._JAX_CONSTANTS, (
            "a refused arm left a STAND-IN map loaded, which would decide "
            "suppressions for every later test in this session"
        )
    finally:
        if eager.is_armed():  # pragma: no cover - defensive
            _tripwire.disarm_eager()


def test_that_arm_time_control_leaves_the_counters_EXACTLY_as_it_found_them(unarmed):
    """A self-check that appeared in the denominator would be a rate about itself.

    The control runs the REAL ``observe``, so it moves every counter and
    writes rows. They are saved and written back rather than reset, because
    ``arm_eager()`` can be called part-way through a session that already has
    figures and ``reset_counters()`` would take those with it.
    """
    eager.reset_counters()
    eager.observe(300, "int16")  # in range: one conversion, no truncation
    before = eager.snapshot()
    assert before["conversions"] == 1
    status = _tripwire.arm_eager()
    try:
        assert status.armed, status.code
        after = eager.snapshot()
    finally:
        _tripwire.disarm_eager()
    assert after == before, (
        "arming moved the user's figures: " +
        repr({k: (before[k], after[k]) for k in before if before[k] != after[k]})
    )


def test_the_const_fold_tripwire_still_needs_ITS_origin_filter(armed):
    """The control, and it is the other instrument on the same program.

    The const-fold tripwire fires on jax's own PRNG mask at TRACE time and
    carries ``record.attribute`` and an ``ORIGIN_JAX`` bucket for it. That is
    the fact that makes the eager hook's own filter the same rule rather than
    a special case, so if it ever stops being true this says so.

    IT RESTORES RATHER THAN DISARMS. This file is meant to be runnable under
    ``-p stelling.overflow``, which arms the tripwire for the whole session;
    a ``finally: disarm()`` here took the session's tripwire out at this test
    and left every later file unwatched. Same argument as the ``armed``
    fixture, one instrument over.
    """
    from stelling._tripwire import record as _record

    was_armed = _tripwire.is_armed()
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
        if not was_armed:
            _tripwire.disarm()


def test_the_arming_ORDER_costs_a_denominator_and_not_a_crash(unarmed):
    """The plugin arms the report before the rule. The reason is measured here.

    The comment on that line used to say that arming in the other order would
    make the tripwire's self-check RAISE, disabling the tripwire on a healthy
    pair of hooks. Driven both ways in fresh interpreters: both report
    ``armed`` either way. What the reversed order actually costs is that the
    tripwire's self-check reaches the eager hook with two IN-RANGE
    conversions, so the session's eager denominator starts at 2 instead of 0 —
    stelling's own traffic in the user's numerator's denominator, which is
    small and is not nothing.
    """
    program = textwrap.dedent(
        """
        from stelling import _tripwire
        from stelling._tripwire import eager

        if {eager_first}:
            eager_status = _tripwire.arm_eager()
            fold_status, _ = _tripwire.arm()
        else:
            fold_status, _ = _tripwire.arm()
            eager_status = _tripwire.arm_eager()
        print(fold_status.code, eager_status.code, eager.CONVERSIONS,
              eager.TRUNCATIONS)
        """
    )
    results = {}
    for label, eager_first in (("report first", False), ("rule first", True)):
        run = subprocess.run(
            [sys.executable, "-c", program.format(eager_first=eager_first)],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr[-2000:]
        results[label] = run.stdout.split()
    if results["report first"][0] != "armed":  # pragma: no cover - environment
        pytest.skip(f"the tripwire could not arm: {results['report first'][0]}")

    assert [row[:2] for row in results.values()] == [["armed", "armed"]] * 2, (
        f"arming in one order disabled an instrument: {results}"
    )
    assert results["report first"][2:] == ["0", "0"], results
    assert results["rule first"][2:] == ["2", "0"], (
        "the reversed order no longer costs the eager denominator the "
        f"tripwire's own self-check: {results}"
    )


def test_arming_twice_hands_back_the_recorder_that_is_ACTUALLY_recording(armed):
    """A second ``arm()`` must not return a recorder connected to nothing.

    ``install()`` returns ``already-armed`` without re-wrapping, which is
    right; ``arm()`` then handed back the fresh ``Recorder()`` it had built on
    the way in, which is connected to nothing and stays at zero however much
    the caller traces. Any assertion written against it is false by
    construction rather than by measurement -- which is exactly how the test
    above failed under ``-p stelling.overflow``, the mode this project's own
    docs recommend.
    """
    was_armed = _tripwire.is_armed()
    first_status, first = _tripwire.arm()
    if not first_status.armed:  # pragma: no cover - environment
        pytest.skip(first_status.code)
    try:
        second_status, second = _tripwire.arm()
        assert second_status.armed
        assert second is first, (
            "the second arm() returned a different recorder from the one the "
            "live wrapper writes to"
        )
        assert _tripwire.evict_trace_caches() == "evicted"
        before = second.invocations
        jax.make_jaxpr(lambda x: x + 300)(jnp.zeros((2,), jnp.int8))
        assert second.invocations > before, (
            "the recorder arm() handed back saw nothing while the hook fired"
        )
    finally:
        if not was_armed:
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
