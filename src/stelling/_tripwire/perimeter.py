# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Mode 3: the dunder perimeter — the literal you wrote, checked where it dies.

WHAT IT IS. ``x <= 2**31 - 1`` on a ``float32`` array is not the program it
looks like. ``2147483647`` has no ``float32``, so jax converts it to
``2147483648.0`` and the comparison the machine runs is one greater than the
one written. Today ``stelling.preconditions.check`` returns **VERIFIED** about
that harness: the moved threshold is in the jaxpr as a constant, the trace is
faithful to the Python, and every layer below is doing its job. The written
number simply never existed. This module attaches to the Python slot the
literal passes through on its way into jax and refuses it there.

THE THIRD INSTRUMENT, AND THE THREE DO NOT SUBSUME ONE ANOTHER:

* the **const-fold tripwire** (``_adapter_jax.py``) watches a jaxpr being
  built and REPORTS an out-of-range integer narrowing;
* the **eager construction-site detector** (``eager.py``) RAISES when
  ``jnp.full((), 256, jnp.int8)`` is written, which no trace ever sees;
* **this** RAISES when a literal meets an array or a tracer in a binary op —
  the route the other two are blind to, and the only one of the three that
  can see **inexactness** rather than only integer range. ``x + 40000`` on
  ``int16`` is caught by the tripwire; ``x <= 2**31 - 1`` on ``float32`` is
  caught by nothing else, because nothing is out of range: it is out of
  *representability*, and the value that reaches the jaxpr is a perfectly
  ordinary ``float32``.

ONE FACE TODAY, AND IT IS THE ONE VERIFICATION RUNS THROUGH. ``"tracer"`` is
the six comparison slots on jax's ``Tracer``. Inside a harness the operand is
a ``DynamicJaxprTracer``, **not** an ``ArrayImpl`` — measured, and it is the
correction that reshaped this rollout: an array-only perimeter never fires
during ``check()`` at all, so it could not have closed the defect above. The
eager array face is a second door, over the spelling a user types at a REPL,
and it arms separately.

THE PREDICATE IS NOT WRITTEN HERE. ``prop_guard.classify(a, b, slot)`` answers
the one question — *does the integer the author wrote exist, exactly, in the
program jax will actually run?* — and it is vendored rather than written,
scored over 482,691 checks of real library code with zero false positives.
Read ``prop_guard.py``'s header for what that census does and does not
establish. **This module supplies the slot name and the array**, both of
which are load-bearing: two of the guard's mitigations key on the slot
(``__pow__`` is excluded, ``__truediv__`` redirects into a float) and the
size-0 exemption reads ``a.size``.

WHAT ARMING COSTS, AND WHAT IT DOES NOT. The wrapper's first act is
``type(b) is int``; when it is false — the overwhelming majority of real ops —
the cost is one Python frame, measured at ×0.99–1.01 of the unarmed op, i.e.
nil. A whole-suite run with a 34-slot perimeter armed took 626.97 s against a
633.82 s baseline: below run-to-run noise, with identical pass counts. This is
a VERIFICATION-TIME instrument — it runs under ``pytest`` and inside
``check()``, never inside XLA execution.

**IT PERTURBS ``source_info`` AND THAT IS DISCLOSED RATHER THAN DENIED.** Every
equation built through an armed slot carries one extra traceback frame — this
module's wrapper. Measured, on a harness with a scan, a cond, a vmap and a
while_loop: ``content_hash``, ``eqns``, ``consts``, the metadata-free document
and the StableHLO text are **byte-identical** armed and disarmed, because
``ir.py``'s ``CANONICALIZATIONS`` declares ``source_info`` outside the hash
scope; ``str(jaxpr)`` and ``to_dict(include_metadata=True)`` **differ**. Do not
claim byte-identity of a persisted document across an armed/disarmed boundary.
jax's own source-info exclusion registry does not fix this — driven, both
spellings registered, no change — and it has no un-register API, so calling it
would be an irreversible process-global write that ``disarm()`` could not
undo. It is not used. (The token for that module is not spelled here: this
repository allows a private jax module name in exactly one file, and
``tests/test_import_hygiene.py`` holds that shut whether the name is in code
or in prose.)

ARM/DISARM IS SESSION-SCOPED, AND THAT IS THE WHOLE OF :func:`arm`'s TOKEN.
Slots are process-global state, and an *idempotent arm with an unconditional
disarm* is the asymmetry that makes a nested in-process session unhook its
parent: the inner ``arm()`` installs nothing (correctly), the inner
``disarm()`` restores everything (catastrophically), and every remaining test
in the outer session runs unprotected with nothing red. That is B8b's
regression in five lines, and it is why :func:`arm` takes an owner and
:func:`disarm` refuses to restore while another owner is still holding.
"""

from __future__ import annotations

import contextlib
import sys

#: The predicate, bound on the first :func:`arm` and read by every wrapper.
#:
#: **NOT IMPORTED AT MODULE SCOPE, AND THE REASON IS THE ZERO-DEPENDENCY
#: CORE.** ``stelling/__init__.py`` exports :class:`NarrowingError` -- a user
#: has to be able to name it in an ``except`` clause -- so ``import stelling``
#: imports THIS module, and ``prop_guard`` imports numpy, which the core does
#: not depend on. Binding it once at arm time keeps the wrapper's fast path to
#: a single global lookup and keeps ``import stelling`` free of numpy. Measured
#: the other way round first: with the import at module scope, ``import
#: stelling`` in the zero-dependency environment died with
#: ``ModuleNotFoundError: No module named 'numpy'``.
_GUARD = None

#: The faces this perimeter can arm, and the slots it installs on each.
#:
#: **THE SIX COMPARISONS ARE THE REFLECTED FORMS**, which is why there are six
#: and not two. Python maps ``N >= x`` to ``x.__le__(N)`` and ``N <= x`` to
#: ``x.__ge__(N)`` — measured on both faces — so ``x <= N`` and ``N >= x`` land
#: in the *same* slot, while ``x <= N`` and ``x >= N`` land in different ones.
#: A user writes both spellings; installing all six is what covers them, and
#: installing "``__le__`` and its reflection" would be installing ``__le__``
#: twice.
#:
#: ``__eq__`` and ``__ne__`` are in the list although a moved literal makes
#: them harmlessly False rather than dangerously True. They are here because
#: the question this instrument asks is "does the number you wrote exist", and
#: the answer does not depend on which direction the resulting wrongness runs.
_COMPARISONS = ("__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__")

#: face -> the slots armed on it. Ordered so a report reads the same twice.
FACE_SLOTS: dict[str, tuple[str, ...]] = {
    "tracer": _COMPARISONS,
}

#: Every face this module knows how to arm, in arming order.
FACES = ("tracer",)


class NarrowingError(BaseException):
    """The literal written at this line does not exist in the program jax runs.

    **``BaseException``, DELIBERATELY, AND IT DIVERGES FROM THE PREDICATE'S OWN
    EXCEPTION ON PURPOSE.** ``prop_guard.NarrowingError`` subclasses
    ``OverflowError`` (its finding F12) so that a caller who already writes
    ``except OverflowError`` around jax's own out-of-int-width refusal keeps
    working. That argument is about a *library call a user makes*; this class
    is raised from inside a dunder slot that jax's own dispatch machinery
    calls, and there two things go wrong with an ``Exception``:

    * an ordinary ``except Exception:`` swallows a soundness alarm — the
      defect this whole campaign exists to close, and the same argument
      :class:`~stelling._tripwire.eager.EagerTruncationError` is built on;
    * worse, a binary-op protocol that catches an ``Exception`` from one
      operand's slot is entitled to *retry the reflected operation*, which
      would turn a refusal into a silently different code path.

    ``prop_guard.raise_for`` and its ``OverflowError`` subclass are untouched
    and still self-tested: this module simply does not use them. A caller who
    was catching ``OverflowError`` must name :class:`NarrowingError` — which is
    exported as ``stelling.NarrowingError`` for exactly that reason.

    The fields are attributes and not only text:

    ``finding``    the :class:`prop_guard.Finding`: reason, slot, operand
                   dtype, the dtype the literal was converted INTO, the
                   literal, and the value the program uses instead.
    ``file``, ``line``, ``func``
                   the frame that wrote the literal — the wrapper's immediate
                   caller, which needs no stack walk and therefore cannot
                   collide with the const-fold tripwire's attribution.
    """

    def __init__(self, finding, *, file: str, line: int, func: str, message: str):
        super().__init__(message)
        self.finding = finding
        self.file = file
        self.line = line
        self.func = func


# ---------------------------------------------------------------------------
# The counters. Module-level and process-global, like the other two
# instruments', because the state they describe (a rebound type slot) is
# process-global too. The plugin reads them through :func:`snapshot`.
# ---------------------------------------------------------------------------

#: How many ``array <op> python_int`` evaluations reached the predicate.
#: **THE DENOMINATOR, AND IT IS PRINTED WHETHER OR NOT ANYTHING FIRED.** "0
#: narrowings" is also exactly what a hook nobody called produces; the number
#: of checks is what separates those two, and a section that printed only the
#: zero would be the beautiful zero this project keeps finding.
CHECKS = 0

#: How many of those the predicate returned a Finding for, including the ones
#: a declaration permitted. The numerator.
FINDINGS = 0

#: ``(file, line) -> [count, reason text]`` for narrowings an
#: ``expected_truncation`` region permitted. A permission that nobody can see
#: is indistinguishable from the silence this instrument exists to end, so
#: they are counted, sited and printed.
PERMITTED: dict = {}

#: Internal failures of THIS module — not of the predicate, which keeps its
#: own :data:`prop_guard.INTERNAL_DECLINES`. Both reach the report.
INTERNAL_ERRORS = 0


def _guard():
    """The predicate module, or ``None`` when this interpreter cannot have it.

    ``None`` is not a defect and not an error: an environment with no numpy
    has no jax either, so there is nothing to put a perimeter around. Every
    caller turns it into a status rather than an exception.
    """
    global _GUARD
    if _GUARD is None:
        try:
            from stelling._tripwire import prop_guard
        except Exception:  # noqa: BLE001 - no numpy means nothing to arm
            return None
        _GUARD = prop_guard
    return _GUARD


def reset_counters() -> None:
    """Zero the counters and the permission table. Used by the self-check."""
    global CHECKS, FINDINGS, INTERNAL_ERRORS
    CHECKS = 0
    FINDINGS = 0
    INTERNAL_ERRORS = 0
    PERMITTED.clear()


def snapshot() -> dict:
    """The whole reportable state, as primitives. Crosses execnet unchanged.

    **THE PREDICATE'S TWO COUNTERS TRAVEL IN HERE**, and that is condition 2 of
    the rollout rather than a convenience. ``prop_guard`` declines silently on
    an internal fault and records the exception name; it records a slot name it
    does not recognise the same way. A run that reports zero fires with a
    non-zero decline count is **not a clean run, it is an unmeasured one**, and
    the only way anybody learns that is if the numbers are in front of them.
    """
    guard = _guard()
    return {
        # WHAT WAS ACTUALLY ARMED, and it is in the payload rather than in the
        # report's prose because the report must not have to know which
        # commit's slot list this process is running. A reader is told what
        # the perimeter covered on THIS run, not what the module could cover.
        "faces": {face: list(entry["slots"]) for face, entry in _installed.items()},
        "checks": CHECKS,
        "findings": FINDINGS,
        "internal_errors": INTERNAL_ERRORS,
        "permitted": {
            f"{file}:{line}": [count, why]
            for (file, line), (count, why) in PERMITTED.items()
        },
        "unknown_slots": sorted(guard.UNKNOWN_SLOTS) if guard else [],
        "declines": dict(guard.INTERNAL_DECLINES) if guard else {},
    }


def _merge(into: dict | None, payload: dict) -> dict:
    """Sum two snapshots. Counts add, tables union, sets union.

    A sum and not a max, for the reason the eager detector's merge is: two
    xdist workers that each declined once declined twice.
    """
    merged = dict(into or {})
    faces = dict(merged.get("faces") or {})
    for face, slots in (payload.get("faces") or {}).items():
        faces[face] = sorted(set(faces.get(face, ())) | set(slots))
    merged["faces"] = faces
    for key in ("checks", "findings", "internal_errors"):
        merged[key] = merged.get(key, 0) + payload.get(key, 0)
    rows = dict(merged.get("permitted") or {})
    for site, row in (payload.get("permitted") or {}).items():
        existing = rows.get(site)
        rows[site] = [existing[0] + row[0], row[1]] if existing else [row[0], row[1]]
    merged["permitted"] = rows
    merged["unknown_slots"] = sorted(
        set(merged.get("unknown_slots") or ()) | set(payload.get("unknown_slots") or ())
    )
    declines = dict(merged.get("declines") or {})
    for name, count in (payload.get("declines") or {}).items():
        declines[name] = declines.get(name, 0) + count
    merged["declines"] = declines
    return merged


# ---------------------------------------------------------------------------
# The policy the wrapper calls
# ---------------------------------------------------------------------------


def _open_region():
    """The innermost ``expected_truncation`` region, or ``None``.

    ONE DECLARATION FOR BOTH INSTRUMENTS, and the widening is stated rather
    than assumed: ``expected_truncation`` was written for the eager detector
    and its docstring now says it covers this one too. The alternative was a
    second context manager with the same shape and the same argument, which is
    a second thing for a user to find and a second thing to forget.

    The accounting stays separate — a narrowing this module permits is counted
    in :data:`PERMITTED` here and never in the eager detector's table — so a
    report still says which instrument permitted what.
    """
    from stelling._tripwire import eager

    stack = eager._REGIONS.get()
    return stack[-1] if stack else None


def _observe(finding, frame) -> None:
    """Count a Finding and either permit it or raise. Never returns a value.

    The writing site is ``frame`` — the wrapper's immediate caller — read
    directly rather than walked. That is why this instrument does not collide
    with the const-fold tripwire's attribution: the tripwire's stack walk picks
    "the innermost non-jax frame", and a wrapper it has never been told about
    wins that contest and produces ``mis-attributed``. Nothing here walks
    anything.
    """
    global FINDINGS
    FINDINGS += 1
    file = frame.f_code.co_filename
    line = frame.f_lineno
    func = frame.f_code.co_name

    region = _open_region()
    if region is not None:
        entry = PERMITTED.setdefault((file, line), [0, ""])
        entry[0] += 1
        # EVERY DISTINCT REASON IS KEPT, not the most recent one. Two regions
        # with different reasons can permit narrowings at the same line -- a
        # helper called from two places -- and a table that overwrote would
        # report one author's justification for another's narrowing, which is
        # worse than reporting neither.
        reasons = [r for r in entry[1].split(" | ") if r]
        if region[0] not in reasons:
            reasons.append(region[0])
        entry[1] = " | ".join(reasons)
        return

    raise NarrowingError(
        finding,
        file=file,
        line=line,
        func=func,
        message=(
            f"{file}:{line} in {func}(): {finding.message}. "
            "Write the value the program actually uses, declare the wrap with "
            "stelling.intentional_wrap(value, dtype), or open a "
            "stelling._tripwire.eager.expected_truncation(reason) region if "
            "the narrowing is the subject of this code."
        ),
    )


def _make_wrapper(slot: str, original):
    """One wrapper per slot. The slot NAME is closed over and it is load-bearing.

    ``prop_guard`` keys two of its mitigations on the slot and records anything
    it does not recognise in ``UNKNOWN_SLOTS``, which the report prints: a
    mistyped slot name is how a guard goes blind, and this is the only place
    that name is produced.
    """

    def wrapper(a, b):
        # THE TYPE TEST IS THE WHOLE HOT PATH. Measured at ×0.99-1.01 of the
        # unarmed op when it is false, which is the overwhelming majority of
        # real operations. `type(b) is int` and not `isinstance`: `bool` is an
        # `int` subclass and `x + True` narrows nothing.
        if type(b) is int:
            global CHECKS, INTERNAL_ERRORS
            CHECKS += 1
            try:
                finding = _GUARD.classify(a, b, slot)
            except Exception:  # noqa: BLE001 - a guardrail may not break a run
                # `classify` documents that it never raises and records its own
                # declines. This is the belt to that braces: an instrument that
                # can take a program down is worse than one that misses.
                #
                # `Exception` AND NOT `BaseException`: a `KeyboardInterrupt`
                # arriving inside the predicate must stay a KeyboardInterrupt.
                # Turning one into "no finding, carry on" would make Ctrl-C
                # unreliable in exactly the sessions this instrument is armed
                # for, and it would be counted as an internal error, which is
                # a false report about the guard as well.
                INTERNAL_ERRORS += 1
                finding = None
            if finding is not None:
                _observe(finding, sys._getframe(1))
        return original(a, b)

    wrapper.__name__ = f"stelling_perimeter_{slot.strip('_')}"
    wrapper.__qualname__ = wrapper.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Arming, disarming, and the owner discipline that makes them session-scoped
# ---------------------------------------------------------------------------

#: ``face -> {"type": T, "slots": {slot: (original, wrapper)}}`` for what this
#: process has installed.
_installed: dict = {}

#: The owners currently holding the perimeter armed, in arrival order. An
#: owner is any hashable token; the pytest plugin uses the session's
#: ``Config``, so a nested in-process session — which builds a fresh
#: ``Config`` — is a different owner **by construction**.
_owners: list = []


def _status(code: str, detail: str = ""):
    from stelling import _tripwire

    return _tripwire.Status(code=code, detail=detail, jax_version=_safe_version())


def _safe_version():
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return adapter.jax_version()
    except Exception:  # noqa: BLE001
        return None


def arm(faces=("tracer",), owner=None):
    """Install the perimeter on ``faces``. Returns a ``Status``; never raises.

    ``owner`` IS THE SESSION SCOPE AND IT IS NOT OPTIONAL IN PRACTICE. Every
    caller that will later disarm must pass one, and two callers that are not
    the same session must pass different ones. Arming twice under the same
    owner is idempotent and registers the owner once; arming under a second
    owner installs nothing (the slots are already ours) and registers the
    second owner, so that the *first* ``disarm()`` restores nothing and the
    last one restores everything.

    FAIL CLOSED, AND FOR THIS INSTRUMENT THE SELF-CHECK IS THE POINT. Locating
    a type and rebinding a slot proves nothing about whether an operation still
    reaches Python: a jax that routed a warm binary op entirely through C++
    would leave every ``setattr`` succeeding and every wrapper cold. So arming
    drives the reference defects through the live slots, in both directions,
    and refuses to arm on ``not-invoked`` (attached and blind) or
    ``cries-wolf`` (fires on a value that is fine).
    """
    from stelling._tripwire import _adapter_jax as adapter

    if _guard() is None:
        return _status(
            "no-module",
            "numpy is not importable, so neither is jax and there is nothing "
            "to put a perimeter around.",
        )
    if owner is None:
        owner = object()
    wanted = tuple(faces)
    unknown = [f for f in wanted if f not in FACE_SLOTS]
    if unknown:
        return _status("no-face", f"this perimeter has no face named {unknown}")

    try:
        # ALREADY ARMED: register the owner, install nothing, and hand back a
        # status that says the state is real rather than one that says "0
        # slots" and leaves the caller to guess whether it is protected.
        already = [f for f in wanted if f in _installed]
        missing = [f for f in wanted if f not in _installed]

        for face in missing:
            located = adapter.perimeter_locate(face)
            if isinstance(located, str):
                return _status(located, f"face {face!r}: {_LOCATE_DETAIL.get(located, '')}")
            slots = {}
            for slot in FACE_SLOTS[face]:
                original = located.__dict__.get(slot)
                if original is None:
                    _uninstall(slots, located)
                    return _status(
                        "no-slot",
                        f"face {face!r} does not carry {slot} as its own "
                        "attribute, so there is nothing to rebind",
                    )
                wrapper = _make_wrapper(slot, original)
                setattr(located, slot, wrapper)
                slots[slot] = (original, wrapper)
            _installed[face] = {"type": located, "slots": slots}

        probe = selfcheck(wanted)
        if probe != "armed":
            for face in missing:
                _restore_face(face)
            return _status(probe, _PROBE_DETAIL.get(probe, ""))

        if not any(held is owner for held in _owners):
            _owners.append(owner)
        return _status(
            "armed",
            f"the dunder perimeter is live on {', '.join(wanted)}: "
            f"{sum(len(_installed[f]['slots']) for f in wanted)} slot(s), "
            f"{len(_owners)} owner(s)"
            + (f"; {', '.join(already)} was already armed" if already else ""),
        )
    except Exception as exc:  # noqa: BLE001 - a guardrail may not raise
        try:
            for face in list(_installed):
                _restore_face(face)
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive
            pass
        return _status(f"unexpected:{type(exc).__name__}", str(exc)[:200])


_LOCATE_DETAIL = {
    "no-module": "jax is not installed, so there are no slots to rebind.",
    "no-type": (
        "the type whose slots this face rebinds could not be located through "
        "any public route. A jax release renamed or restructured it."
    ),
}

_PROBE_DETAIL = {
    "not-invoked": (
        "the slots are rebound and a program that must reach them did not. "
        "Attached-but-blind is what a jax release actually produces."
    ),
    "cries-wolf": (
        "a literal that IS exactly representable was refused, so every "
        "refusal this run would be noise."
    ),
    "mis-attributed": (
        "the guard fired at the wrong line. A refusal that points at code "
        "the author did not write costs more trust than a missing one."
    ),
}


def disarm(owner=None) -> str:
    """Release ``owner``'s hold. Restores the slots only when it is the last.

    Returns ``restored``, ``still-armed`` (someone else is holding),
    ``not-armed``, ``not-an-owner`` or ``foreign-patch``. Never raises.

    **THE REFUSAL TO RESTORE IS THE FEATURE.** A nested in-process pytest
    session arms (idempotently, installing nothing) and then disarms at its own
    teardown; with an unconditional restore that teardown unhooks the OUTER
    session, whose remaining tests then run unprotected with nothing red. The
    owner is removed from wherever it sits in the list rather than only from
    the top, so an out-of-order release leaks nothing and unhooks nobody early.
    """
    if owner is not None and any(held is owner for held in _owners):
        _owners[:] = [held for held in _owners if held is not owner]
    elif owner is not None:
        return "not-armed" if not _installed else "not-an-owner"
    elif _owners:
        # A caller with no token at all releases the most recent hold. Kept
        # for a REPL and for `python -m`, and deliberately not used by the
        # plugin: an anonymous release cannot be session-scoped.
        _owners.pop()

    if _owners:
        return "still-armed"
    if not _installed:
        return "not-armed"

    codes = {_restore_face(face) for face in list(_installed)}
    if "foreign-patch" in codes:
        return "foreign-patch"
    return "restored"


def _restore_face(face: str) -> str:
    """Put one face's slots back **by identity**. ``restored``/``foreign-patch``.

    Something else bound over our wrapper is reported, not clobbered: whatever
    replaced it is left in place, exactly as the other two instruments do.
    """
    entry = _installed.pop(face, None)
    if entry is None:
        return "not-armed"
    owner_type = entry["type"]
    foreign = False
    for slot, (original, wrapper) in entry["slots"].items():
        if owner_type.__dict__.get(slot) is not wrapper:
            foreign = True
            continue
        setattr(owner_type, slot, original)
    return "foreign-patch" if foreign else "restored"


def _uninstall(slots: dict, owner_type) -> None:
    """Undo a partial install. Only reachable when a face is missing a slot."""
    for slot, (original, _wrapper) in slots.items():
        setattr(owner_type, slot, original)


def is_armed() -> bool:
    """Whether every slot this process installed is still the live attribute."""
    return live_check() == "armed"


def live_check() -> str:
    """``armed``, ``detached`` or ``foreign-patch``. Never raises.

    Same three states, and the same reason for keeping the two "no"s apart, as
    the other two instruments: ``detached`` means we hold no installation and
    ``foreign-patch`` means we hold one and are not the live binding. What
    :func:`arm` established at the start of a session is not still true at the
    end of it, and the report is written at the end.
    """
    try:
        if not _installed:
            return "detached"
        for entry in _installed.values():
            owner_type = entry["type"]
            for slot, (_original, wrapper) in entry["slots"].items():
                if owner_type.__dict__.get(slot) is not wrapper:
                    return "foreign-patch"
        return "armed"
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"


def armed_faces() -> tuple[str, ...]:
    """The faces currently installed, in arming order."""
    return tuple(f for f in FACES if f in _installed)


def owners() -> int:
    """How many owners are holding the perimeter armed."""
    return len(_owners)


# ---------------------------------------------------------------------------
# The self-check
# ---------------------------------------------------------------------------


def selfcheck(faces=()) -> str:
    """Drive the reference defects through the LIVE slots, both directions.

    ``armed``, ``not-invoked``, ``cries-wolf``, ``mis-attributed``, or
    ``unexpected:<ExcType>``. Never raises.

    A POSITIVE-ONLY PROBE PASSES ON A HOOK REPLACED BY "REFUSE EVERYTHING", and
    this repository has repeatedly found exactly that shape of vacuous control,
    so each face is driven with a literal that does **not** survive and one
    that does. The tracer face additionally clears jax's trace caches first:
    a warm trace cache means the probe's Python never re-runs, the hook is
    never entered, and a blind hook and a healthy one produce the same
    silence. That hazard is measured — ``trace(h)`` fires on call #1 and not on
    #2 or #3 — and it is the reason this function exists rather than a
    presence check.

    **IT LEAVES THE COUNTERS EXACTLY AS IT FOUND THEM.** It runs the real
    wrapper, so it moves :data:`CHECKS` and :data:`FINDINGS`; a self-check that
    appeared in a user's denominator would make every rate this instrument
    prints a rate about itself. They are saved and written back rather than
    reset, because ``arm()`` may be called part-way through a session that
    already has figures.
    """
    global CHECKS, FINDINGS, INTERNAL_ERRORS
    from stelling._tripwire import _adapter_jax as adapter
    from stelling._tripwire import eager

    wanted = tuple(faces) or armed_faces()
    saved = (CHECKS, FINDINGS, INTERNAL_ERRORS, dict(PERMITTED))
    # A region open around `arm()` would PERMIT the probe's narrowing instead
    # of raising it, and a control a caller's context can switch off is not a
    # control.
    token = eager._REGIONS.set(())
    try:
        for face in wanted:
            try:
                probes = adapter.perimeter_probes(face)
            except Exception as exc:  # noqa: BLE001
                return f"unexpected:{type(exc).__name__}"

            for label, run, want_line in probes["bad"]:
                try:
                    run()
                except NarrowingError as exc:
                    if want_line is not None and exc.line != want_line:
                        return "mis-attributed"
                except Exception as exc:  # noqa: BLE001
                    return f"unexpected:{type(exc).__name__}"
                else:
                    return "not-invoked"

            for label, run, _line in probes["good"]:
                try:
                    run()
                except NarrowingError:
                    return "cries-wolf"
                except Exception as exc:  # noqa: BLE001
                    return f"unexpected:{type(exc).__name__}"
        return "armed"
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"
    finally:
        eager._REGIONS.reset(token)
        CHECKS, FINDINGS, INTERNAL_ERRORS, permitted = saved
        PERMITTED.clear()
        PERMITTED.update(permitted)


@contextlib.contextmanager
def armed(faces=("tracer",), owner=None):
    """Arm for the duration of a block, and restore on **every** way out.

    ::

        with perimeter.armed() as status:
            assert status.armed
            ...

    THE THIRD OF THE THREE ARM/DISARM DEFECTS IS THE ONE THIS CLOSES, and it is
    the dullest: an exception between a bare ``arm()`` and a bare ``disarm()``
    leaves the slots rebound for the rest of the process, because nothing put
    them back. There is no ``finally`` in a pair of calls. Any long-lived
    caller should use this or the pytest plugin, both of which have one.

    The status is yielded rather than returned so that a caller can see a
    refusal — ``arm()`` never raises, so ``with armed():`` on a jax that moved
    would otherwise run the block believing it was watched.
    """
    token = object() if owner is None else owner
    status = arm(faces, owner=token)
    try:
        yield status
    finally:
        if status.armed:
            disarm(token)
