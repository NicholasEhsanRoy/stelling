# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Mode 2: the eager call-site crash, and the two ways an author declares a wrap.

**Pure Python. No jax, and no numpy.** Everything here is testable in a bare
interpreter; the one function that touches a private jax surface lives in
``_adapter_jax.py``, which is the only file in this repository allowed to name
one, and is reached lazily from :func:`arm`.

WHAT THIS CLOSES. The overflow tripwire (``_tripwire.arm``) hooks jax's
const-fold rule for ``convert_element_type``, which catches the INLINE door:
``x + 256`` on an ``int8`` array traces to ``add a 0:i8[]`` and the rule sees
the written 256 die. It does not catch the EAGER door. ``jnp.full((), 256,
jnp.int8)`` is ``0`` before any primitive is bound — the value is narrowed at
array construction, inside the private jax function ``_adapter_jax`` hooks
(``EAGER_MODULE`` / ``EAGER_ATTR`` there -- this module may not spell that
name and does not, which is ``design/private-jax-boundary.md``'s rule), in its
``if type(operand) is int`` branch, by ``np.asarray(operand).astype(new_dtype)``
— and nothing downstream can tell that the 0 in the jaxpr was written as a 256.
``tests/test_tripwire_gate_coverage.py::GATE_COVERAGE`` enumerates the routes;
six of the seven unwatched ones narrow at that one line.

THE TWO MODES, AND THIS IS THE SECOND. Mode 1 would record the eager narrowing
and refuse the verdict from inside ``preconditions.check()``'s trace gate, which
needs ATTRIBUTION: some way to decide that the constant destroyed at 11:04 is
the constant the trace at 11:05 is standing on. That is not built here. Mode 2
needs no attribution at all, because it never has to connect two events: it
raises **at the construction site**, in the frame that wrote the constant,
while that frame is still on the stack.

**IT IS OPT-IN AND OFF BY DEFAULT.** Nothing here is armed by importing
``stelling``, by importing this module, or by running pytest with the tripwire
plugin registered. :func:`arm` is called by ``--stelling-eager-truncation=error``
and by an explicit ``_tripwire.arm_eager()``, and by nothing else. A user who
does not switch it on gets a byte-identical program: the module attribute jax
looks the constructor up on is untouched, and ``intentional_wrap`` is a pure
function of its arguments whether or not anything is armed.

---------------------------------------------------------------------------
WHY THE EXCEPTION INHERITS FROM ``BaseException``
---------------------------------------------------------------------------

:class:`EagerTruncationError` derives **directly from BaseException**, not from
``Exception``. That is a deliberate choice and it is the single most
consequential line in this module, so the argument is written down rather than
left to the reader.

The alarm fires inside arbitrary user code — inside ``jnp.full``, inside a
library's array constructor, inside a ``jit``-decorated body. Numerical Python
is full of ``except Exception:`` — retry loops, fallback paths, "try the fast
kernel and fall back", ``warnings``-to-error shims, and this repository's own
guardrails, which catch ``Exception`` on principle so that an instrument can
never break the thing it measures. An alarm that says *"the constant you wrote
does not exist in the program that will run"* and is then swallowed by a
handler written for a different purpose is not an alarm; it is a silent
program with extra steps. ``BaseException`` is what ``KeyboardInterrupt`` and
``SystemExit`` inherit from for exactly this reason: they are not the running
program's errors to handle.

**"Uncatchable" is not achievable in Python and this module does not claim
it.** ``except BaseException:``, a bare ``except:``, and
``contextlib.suppress(BaseException)`` all still catch this, and
``sys.excepthook`` still runs. The claim is narrow and is exactly this: the
COMMON swallow does not catch it. A census with this armed -- 25 module
imports, then eleven real workloads across eight of those libraries -- gives
**0 fires over 94 scalar integer conversions**, so the radius in which the
choice matters at all is small. The census is in
``design/eager-truncation-detector.md`` and is driven by
``tests/test_tripwire_eager.py``.

**IT NEEDS NO ORIGIN FILTER, and that is measured rather than hoped.** The
const-fold tripwire fires on JAX'S OWN constants -- ``jax.random.key(0)``
folds ``4294967295 -> -1`` inside ``threefry2x32.py`` -- and carries a whole
attribution machinery to keep that out of a user's findings. This never sees
it: the same program gives 2 scalar conversions and 0 truncations here. That
is a property of WHERE the two hooks sit rather than of anything clever, and
it is what makes an alarm that stops the program affordable at all.

**WHAT IT COSTS, said plainly.** ``finally:`` blocks still run and context
managers' ``__exit__`` still runs, so ordinary resource cleanup is unaffected.
Cleanup written as ``except Exception: release()`` does NOT run — a caller who
releases a lock or closes a file in an ``except Exception`` handler and not in
a ``finally`` will leak it when this fires. That is a real cost and it is the
price of the alarm not being swallowable by the same handler. pytest reports a
``BaseException`` raised in a test BODY as a FAILURE (measured: ``1 failed``,
not ``1 error``), and one raised during COLLECTION or in a fixture as an
error, which is the same handling any other exception gets there. The
``BaseException`` choice changes what can SWALLOW the alarm, not how pytest
files it.

---------------------------------------------------------------------------
ERROR BY DEFAULT ON EVERY TRUNCATION. NO VALUE-BASED CARVE-OUT.
---------------------------------------------------------------------------

The obvious refinement is to guess intent from the numbers: let ``0xFF`` into
``int8`` through as a mask idiom and stop ``300`` into ``int8`` as an accident.
It cannot be done, and this was measured rather than asserted.
``jnp.full((4,), 0xFF, jnp.int8)`` and ``jnp.full((4,), 255, jnp.int8)``
produce **identical observations at the hook**: the same written value, the
same target dtype, the same result, the same frame. They differ only in the
source TEXT, which is not available as data at the point the decision has to be
made and which a variable, a computed value or a constant defined in another
module does not carry at all. Intent is therefore not a function of
``(value, dtype, result)``, and no rule over that data can be sound.

Two candidate rules were driven over a corpus of real narrowings:

* **Rule A** — "a value below the dtype's minimum is deliberate", which for
  an unsigned dtype is exactly "a negative literal into an unsigned type":
  hard-errors correct code **7** times and lets a real bug through **1** time.
* **Rule B** — "an all-ones result is deliberate": **5** and **2**.

Both are wrong in both directions. **Ship neither.** And the corpus is not
just a scoreboard: ``0xFF`` into ``int8`` is a mask idiom, ``255`` into
``int8`` is a saturated pixel written into a signed byte, and those are the
SAME ``(value, dtype)`` pair — so the class of value-based rules is EMPTY, not
merely badly-scoring. ``tests/test_tripwire_eager.py`` carries the corpus, the
collision and both scores, and recomputes them. What replaces them is a
declaration the author writes, which is exact because the author asserts it —
the same standing ``assume()`` has in a verdict: not inferred, not guessed,
carried as a premise and disclosed.

---------------------------------------------------------------------------
THE TWO DECLARATIONS
---------------------------------------------------------------------------

:func:`intentional_wrap` is the one users write, and it is the primary.
``intentional_wrap(0xFF, "int8")`` returns ``-1`` — the wrap, computed here in
Python integer arithmetic — so the value that reaches jax is the value jax
would have produced anyway. Three properties follow, and they are why this
shape was chosen over a flag, a suppression list, or a decorator:

* **It is exact**, because the author asserts it rather than the tool inferring
  it.
* **It cannot license a different site.** It is a value, not a mode: it changes
  what happens at the one expression it is written in and has no effect
  anywhere else, in any other thread, or one line later.
* **It cannot license a different dtype.** The dtype is half the declaration.
  ``intentional_wrap(0xFF, "int8")`` used where a ``uint8`` is wanted yields
  ``-1``, which is out of range for ``uint8``, so the detector fires on the
  declared value. A declaration that drifted from its use is caught, not
  honoured.

:func:`expected_truncation` is the second, and it exists for one narrow case:
code whose SUBJECT is the truncation. This repository has several — the doors
in ``tests/test_tripwire_arm.py`` are driven precisely to demonstrate that they
narrow in silence, and ``SOUNDNESS.md``'s reproducer exists to be executed and
observed wrapping. Rewriting those with :func:`intentional_wrap` would delete
the demonstration: the point of the line is that ``300`` becomes ``44``, and a
line that writes ``44`` no longer shows it. So a region declaration exists, and
it is deliberately awkward — it takes a mandatory reason, it is a context
manager and therefore lexically bounded, it is thread-local, and **every
truncation it permits is counted and named in the report**. An opt-out that
hid what it suppressed would be the same silence this module exists to end.
"""

from __future__ import annotations

import os
import sys
import threading

from stelling._tripwire import record

#: The dtypes this module range-checks, from :mod:`record` rather than a second
#: copy: a name that could be modelled in one place and not the other is a
#: constant that is checked by one half of the tool and not the other.
INT_DTYPES = record.INT_DTYPES


class EagerTruncationError(BaseException):
    """An integer constant was silently truncated at its construction site.

    Inherits **directly from BaseException** so that an ``except Exception:``
    block cannot swallow it. The module docstring carries the argument, what
    the claim is not, and what it costs.

    The fields are attributes rather than only text, so a caller that does
    catch this can act on it without parsing the message:

    ``written``   the integer the author wrote.
    ``to_dtype``  the dtype it was being narrowed into, as a string.
    ``became``    what two's-complement truncation makes of it.
    ``file``, ``line``, ``func``
                  the innermost frame outside jax and outside stelling —
                  the writer, by the same rule the tripwire's report uses.
    """

    def __init__(self, message, *, written, to_dtype, became, file, line, func):
        super().__init__(message)
        self.written = written
        self.to_dtype = to_dtype
        self.became = became
        self.file = file
        self.line = line
        self.func = func


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def dtype_name(dtype) -> str:
    """The dtype's name as a plain string, from any of the spellings jax uses.

    ``"int8"``, ``numpy.int8``, ``numpy.dtype("int8")``, ``jax.numpy.int8`` all
    answer ``"int8"``. Nothing is imported to do it: a dtype object is asked
    for its own name and never converted, so this module keeps working in an
    interpreter with neither jax nor numpy installed.

    Raises ``ValueError`` for anything not one of the eight integer dtypes
    :data:`INT_DTYPES` models. That refusal is deliberate and is not a
    limitation being hidden: :func:`intentional_wrap` is a declaration with
    formal weight, and a declaration this module cannot check the arithmetic of
    is one it must not accept.
    """
    if isinstance(dtype, str):
        name = dtype
    else:
        name = getattr(dtype, "name", None)
        if not isinstance(name, str):
            name = getattr(dtype, "__name__", None)
        if not isinstance(name, str):
            name = str(dtype)
    name = name.strip()
    if name not in INT_DTYPES:
        raise ValueError(
            f"stelling.intentional_wrap: {dtype!r} is not one of the integer "
            f"dtypes this declaration covers ({', '.join(sorted(INT_DTYPES))}). "
            "A wrap declaration is a premise the tool carries, so it is "
            "refused rather than accepted unchecked."
        )
    return name


#: ``(file, line)`` -> ``[count, written->became text]`` for every declaration
#: :func:`intentional_wrap` has honoured this process. The report prints it.
#: A declaration nobody can see is indistinguishable from the silence this
#: module exists to end, so declarations are DISCLOSED, exactly as an
#: ``assume()`` is disclosed in a verdict's stamp rather than applied quietly.
DECLARED: dict[tuple[str, int], list] = {}

#: The same, for truncations permitted by an :func:`expected_truncation`
#: region: ``(file, line)`` -> ``[count, reason]``.
PERMITTED: dict[tuple[str, int], list] = {}

#: How many scalar integer->integer conversions the hook has seen, in total.
#: THE DENOMINATOR, and it is printed whether or not anything fired: "0
#: truncations" over an unknown number of conversions is the beautiful zero a
#: dead hook also produces.
CONVERSIONS = 0

#: How many of those were out of range — the numerator, including the ones a
#: declaration permitted.
TRUNCATIONS = 0


def intentional_wrap(value, dtype):
    """Declare that a value is MEANT to wrap, and return the wrapped integer.

    ``intentional_wrap(0xFF, "int8")`` is ``-1``. ``intentional_wrap(-1,
    "uint8")`` is ``255``. The arithmetic is :func:`record.narrow`, the same
    two's-complement recomputation the tripwire's report checks every observed
    narrowing against — one implementation, so the value a declaration produces
    and the value the report predicts cannot disagree.

    WHY IT RETURNS THE WRAPPED VALUE RATHER THAN MARKING THE ORIGINAL. Because
    then there is nothing left to detect: the integer handed to jax is already
    in range, jax's narrowing branch is not reached, and the program is
    byte-identical to the one an author would have written by typing the
    wrapped value themselves. A marker object — an ``int`` subclass, a sentinel
    — would have to survive into jax, and jax's own narrowing branch is guarded
    by ``type(operand) is int``, which no subclass satisfies: the declaration
    would have changed which code path jax takes, which is exactly what a
    declaration must never do.

    WHY THE DTYPE IS REQUIRED. It is half the declaration. A wrap is only
    meaningful relative to a width and a signedness, and requiring it here is
    what stops one declaration from licensing a different one: the value
    returned is in range for *this* dtype and, in general, out of range for
    others, so using it somewhere else fires the detector rather than passing
    silently.

    **Under ``jax_enable_x64=False``, jax canonicalises ``int64`` to ``int32``
    on construction.** ``intentional_wrap(2**40, "int64")`` therefore returns
    ``2**40`` — in range for ``int64`` — and jax then narrows it to ``int32``.
    That is not a hole: the detector observes the dtype jax actually used, so
    it fires, names ``int32``, and the mismatch between what was declared and
    what happened is reported rather than absorbed.

    Zero-dependency and always available: it imports nothing, does not consult
    whether the detector is armed, and behaves identically whether it is.
    """
    name = dtype_name(dtype)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"stelling.intentional_wrap: the value must be a Python int, got "
            f"{type(value).__name__} ({value!r}). A declaration about an "
            "integer wrap is refused for anything that is not one."
        )
    wrapped = record.narrow(value, name)
    file, line, func = _writer_frame(1)
    entry = DECLARED.setdefault((file, line), [0, ""])
    entry[0] += 1
    entry[1] = f"{value} -> {wrapped} ({name}), in {func}()"
    return wrapped


class _Regions(threading.local):
    """Per-thread :func:`expected_truncation` nesting.

    THREAD-LOCAL, and that is the same discipline the trace gate's fire counter
    already uses. A region is a statement about the code inside a ``with``
    block on one thread; making it process-global would let one thread's
    demonstration license another thread's accident, which is precisely the
    "silently license a different site" failure the declaration design is
    built to avoid.
    """

    def __init__(self):
        self.stack: list[tuple[str, str, int]] = []


_REGIONS = _Regions()


class expected_truncation:  # noqa: N801 - a context manager reads as a verb
    """Declare that truncations inside this block are the SUBJECT of the code.

    ::

        with expected_truncation("this door's subject is the silent narrowing"):
            out = jax.jit(door)(x)

    The narrow case this exists for is code that must actually perform the
    truncation and observe the result — a test that demonstrates a door narrows
    in silence, a reproducer in a soundness disclosure that exists to be run.
    :func:`intentional_wrap` cannot serve those: writing the wrapped value
    deletes the very thing being demonstrated.

    IT IS DELIBERATELY THE AWKWARD ONE. The reason is mandatory; the region is
    lexically bounded and thread-local; nesting is honoured; and every
    truncation it permits is counted in :data:`PERMITTED`, printed in the
    report with its site and its reason, and included in the numerator. An
    opt-out that hid what it suppressed would reintroduce the silence this
    module exists to end, one level up.

    It is NOT the answer to "the detector is noisy in my code" — that answer is
    :func:`intentional_wrap` at the site, or not arming the detector at all.
    """

    __slots__ = ("_reason", "_frame")

    def __init__(self, reason: str):
        if not isinstance(reason, str) or not reason.strip():
            raise TypeError(
                "expected_truncation(reason) requires a non-empty reason. The "
                "reason is printed beside every truncation the region permits; "
                "a region with nothing to say for itself is a silent opt-out."
            )
        self._reason = reason.strip()
        self._frame = None

    def __enter__(self):
        file, line, func = _writer_frame(1)
        self._frame = (file, line)
        _REGIONS.stack.append((self._reason, file, line))
        return self

    def __exit__(self, exc_type, exc, tb):
        if _REGIONS.stack:
            _REGIONS.stack.pop()
        return False


# ---------------------------------------------------------------------------
# The policy the hook calls
# ---------------------------------------------------------------------------

#: The directory this package lives in. Frames under it are stelling's own and
#: are never attributed to the user, the same rule the tripwire's stack walk
#: applies to jax's frames.
_STELLING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

#: Set at :func:`arm` time from the adapter. A string and only ever a string —
#: this module must not learn what jax is.
_JAX_ROOT = ""


def _writer_frame(skip: int) -> tuple[str, int, str]:
    """The innermost frame outside jax and outside stelling: the writer.

    Same rule as ``record.attribute`` and stated in the same words in the
    report — *the innermost frame OUTSIDE JAX, your own code or a library you
    called that is not jax*. It is a filter and not a lookup, and it can be
    wrong in the one direction the report already discloses: a constant written
    inside a third-party library is attributed to that library's line, which is
    correct and is not "your own code".

    Falls back to the outermost frame rather than to nothing: an alarm that
    cannot say where is still an alarm, and ``<unknown>:0`` in a message is a
    defect a reader can report.
    """
    try:
        frame = sys._getframe(skip + 1)
    except ValueError:  # pragma: no cover - defensive
        return "<unknown>", 0, "?"
    last = ("<unknown>", 0, "?")
    while frame is not None:
        name = frame.f_code.co_filename
        last = (name, frame.f_lineno, frame.f_code.co_name)
        absolute = os.path.abspath(name) if os.path.sep in name else name
        if absolute.startswith(_STELLING_ROOT):
            frame = frame.f_back
            continue
        if _JAX_ROOT and absolute.startswith(_JAX_ROOT):
            frame = frame.f_back
            continue
        return name, frame.f_lineno, frame.f_code.co_name
    return last


def observe(written: int, to_dtype: str) -> None:
    """Every scalar integer -> integer conversion jax performs, one call each.

    This is the whole policy, and it lives here rather than in the adapter so
    that it can be driven in an interpreter with no jax at all — which is what
    ``tests/test_tripwire_eager.py`` does for the branches that do not need a
    trace.

    Raises :class:`EagerTruncationError` when the value does not fit and no
    declaration covers it. Returns ``None`` in every other case, including
    when a declaration DOES cover it: a permitted truncation is counted and
    recorded, never silently dropped.
    """
    global CONVERSIONS, TRUNCATIONS

    CONVERSIONS += 1
    if record.in_range(written, to_dtype):
        return
    TRUNCATIONS += 1

    file, line, func = _writer_frame(1)
    if _REGIONS.stack:
        reason = _REGIONS.stack[-1][0]
        entry = PERMITTED.setdefault((file, line), [0, reason])
        entry[0] += 1
        entry[1] = reason
        return

    became = record.narrow(written, to_dtype)
    raise EagerTruncationError(
        _message(written, to_dtype, became, file, line, func),
        written=written,
        to_dtype=to_dtype,
        became=became,
        file=file,
        line=line,
        func=func,
    )


def _message(written, to_dtype, became, file, line, func) -> str:
    """The alarm, in the shape ``report.py`` argues every finding must have.

    Both halves observed, the arithmetic a reader can check with a pencil, the
    user's own line quoted, and what to do about it. No inference about intent
    is offered, because there is none to be had: the module docstring records
    why ``0xFF`` and ``255`` are the same event here.
    """
    bounds = record.dtype_range(to_dtype)
    quoted = record.source_line(file, line)
    lines = [
        f"stelling: {written} was TRUNCATED to {became} at its construction "
        f"site, and {written} does not exist anywhere in the program jax "
        "will run.",
        "",
        f"    written   {written}",
        f"    dtype     {to_dtype}  (range {bounds[0]} .. {bounds[1]})"
        if bounds
        else f"    dtype     {to_dtype}",
        f"    became    {became}",
        f"    arithmetic  {record.arithmetic_sentence(written, to_dtype)}",
        f"    at        {file}:{line}, in {func}()",
    ]
    if quoted:
        lines.append(f"              {quoted}")
    lines += [
        "",
        "This is jax narrowing an out-of-range Python integer during array "
        "construction. It is not an error in jax and jax does not warn: the "
        "value is gone before any primitive is bound, so every later check — "
        "stelling's included — is a check of a program you did not write.",
        "",
        "If the wrap is what you meant, DECLARE it and the declaration will "
        "be carried and disclosed:",
        "",
        f"    from stelling import intentional_wrap",
        f"    intentional_wrap({written}, {to_dtype!r})   # == {became}",
        "",
        "There is no value-based exemption and there will not be one: "
        f"{written} written as a mask and {written} written by accident are "
        "the same event at this point in jax, differing only in source text.",
    ]
    return "\n".join(lines)


#: How many times :func:`reset_counters` has been called this process. THE
#: REPORT PRINTS IT, because a denominator that was reset is a denominator
#: about part of a session and reads exactly like a denominator about all of
#: it. Measured: running this repository's own suite with the detector armed
#: session-wide printed ``0 scalar integer conversion(s)`` -- not because the
#: hook saw nothing, but because the test files that drive it reset the
#: counters and the last reset is what the summary read.
RESETS = 0


def reset_counters() -> None:
    """Empty the counters and the declaration tables. Test support.

    Nothing in the shipped path calls this: a user's session never resets, so
    its figures cover the whole run. A SUITE THAT TESTS THIS DETECTOR does
    reset, and the count of resets is carried into the snapshot so the report
    can say its figures are partial rather than quietly presenting a fragment
    as a total.
    """
    global CONVERSIONS, TRUNCATIONS, RESETS
    CONVERSIONS = 0
    TRUNCATIONS = 0
    RESETS += 1
    DECLARED.clear()
    PERMITTED.clear()


def internal_errors() -> int:
    """How many times the hook's own bookkeeping failed and was swallowed.

    The wrapper catches everything that is not the alarm, because an
    instrument that breaks the program it measures is worse than no
    instrument. A swallowed error that nobody can see is the other half of
    that bargain unpaid, so it is counted and printed.
    """
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return int(adapter._eager_internal_errors[0])
    except Exception:  # noqa: BLE001
        return 0


def snapshot() -> dict:
    """What the report prints. Primitives only, so it crosses an xdist boundary."""
    return {
        "conversions": CONVERSIONS,
        "truncations": TRUNCATIONS,
        "resets": RESETS,
        "internal_errors": internal_errors(),
        "declared": {
            f"{file}:{line}": [count, text]
            for (file, line), (count, text) in sorted(DECLARED.items())
        },
        "permitted": {
            f"{file}:{line}": [count, reason]
            for (file, line), (count, reason) in sorted(PERMITTED.items())
        },
    }


# ---------------------------------------------------------------------------
# Arming
# ---------------------------------------------------------------------------


def arm():
    """Install the eager detector. Returns a :class:`_tripwire.Status`; never raises.

    FAIL CLOSED, and for this hook that phrase has to mean more than it does
    for the const-fold one. The tripwire attaches to a registry ENTRY, and a
    release that moves the registry makes it disappear — loudly, as
    ``no-registry``. This attaches to a module ATTRIBUTE on a private module,
    and the failure that matters is not the attribute disappearing (also loud)
    but the attribute surviving while jax stops routing a construction route
    through it. That is silent, and it is the whole risk of the design.

    So the self-check drives **every route this detector claims**, positively,
    at arm time, and refuses to arm if any one of them stops reaching the hook.
    Losing five routes quietly to keep one is not a trade this tool gets to
    make on a user's behalf; the status code names which route went blind.
    """
    from stelling import _tripwire
    from stelling._tripwire import _adapter_jax as adapter

    global _JAX_ROOT

    def status(code: str, detail: str = "") -> _tripwire.Status:
        return _tripwire.Status(
            code=code,
            detail=detail,
            jax_version=_safe(adapter.jax_version),
            rule_name=_safe(adapter.eager_site_name),
            rule_hash=_safe(adapter.eager_site_hash),
            known_hash=_safe(adapter.known_eager_hash),
        )

    try:
        located = adapter.eager_locate()
        if located != "located":
            return status(located)

        signature = adapter.eager_signature_check()
        if signature != "ok":
            return status("signature-drift", signature)

        _JAX_ROOT = adapter.jax_root()
        installed = adapter.eager_install(observe)
        if installed not in ("installed", "already-armed"):
            return status(installed)

        probe = adapter.eager_selfcheck()
        if probe != "armed":
            adapter.eager_restore()
            return status(probe)
        return status("armed", "the eager construction-site detector is live")
    except Exception as exc:  # noqa: BLE001 - a guardrail may not raise
        try:
            adapter.eager_restore()
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive
            pass
        from stelling import _tripwire as _tw

        return _tw.Status(code=f"unexpected:{type(exc).__name__}", detail=str(exc)[:200])


def disarm() -> str:
    """Put jax's own constructor back, by identity. Never raises."""
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return adapter.eager_restore()
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"


def is_armed() -> bool:
    """Whether this process's wrapper is still the live module attribute."""
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return adapter.eager_is_armed()
    except Exception:  # noqa: BLE001
        return False


def live_check() -> str:
    """``armed``, ``detached`` or ``foreign-patch``. Never raises."""
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return adapter.eager_live_check()
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None
