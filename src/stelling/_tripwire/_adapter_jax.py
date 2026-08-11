# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The only file in stelling permitted to name a private jax module.

It locates the const-fold registry, installs a recording wrapper over the
``convert_element_type`` rule, probes it in both directions, and restores it
by identity.

WHY THE EXEMPTION EXISTS, in one paragraph, because a reader who finds this
file needs the reason before the code. The tripwire attaches to
``const_fold_rules``, keyed on ``convert_element_type_p``. The primitive is
public — ``jax.extend.core.primitives`` — and is the same object; the registry
keyed on it is not exported by any public or ``jax.extend`` module. Measured on
both tested series: seven candidate modules, including
``jax.interpreters.partial_eval``, which imports cleanly and does not carry it.
So there is no route to this hook that does not name a private module, and the
choice was an exemption or no feature. ``design/private-jax-boundary.md``
carries the table and the alternatives that were rejected.

WHAT THE EXEMPTION IS, EXACTLY. It is rule 2 only — ``jax._src`` — and it is
pinned to this one path in both controls (the ``jax-import-hygiene`` hook
filters on the anchored path; ``tests/test_import_hygiene.py`` compares the
repo-relative path). **This file is still subject to rule 1**: it may not spell
``import jax`` or ``from jax``, and it does not. The private module is reached
through :func:`importlib.import_module` with a plain literal name — which is
also what the fail-closed contract needs, since this adapter must *probe* and
report rather than import and die — and public jax, which the self-check needs
for real work, comes through ``stelling._jax_compat`` like everything else in
the package. The exemption bought one private module, not a second jax
boundary.

NOTHING JAX-SHAPED LEAVES THIS MODULE. Every public function here returns a
string, an int, a tuple of primitives, or None. That is this rule's
requirement — the exemption is void the moment the private object is reachable
from a module that does not have it — and it is also the tripwire's own, since
a finding has to survive an xdist process boundary. One discipline, two
payoffs.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import re
import sys

from stelling import _optional
from stelling._tripwire import record

# The private module and the attribute on it. Spelled as data rather than as an
# import statement on purpose: this module must import cleanly with jax absent,
# report ``no-module``, and never raise at import.
PRIVATE_MODULE = "jax._src.interpreters.partial_eval"
REGISTRY_ATTR = "const_fold_rules"

# ---------------------------------------------------------------------------
# Version bounding (PLAN-tripwire.md §5)
#
# BEHAVIOUR IS THE CONTRACT; THE VERSION IS A PRE-FILTER AND A DISCLOSURE. The
# probe in :func:`selfcheck` is the authority, because a version claim is one
# nothing verifies and a patch release can move the rule without moving the
# minor version.
# ---------------------------------------------------------------------------

#: Below this, refuse without probing. 0.4.8 is the release nearest jax commit
#: ``c2fe350455``, which created the line that wraps: below it there is no
#: reason to believe the rule this tool attaches to does what it attaches for.
#:
#: THE 0.5.1 HASH DOES NOT JUSTIFY THIS BOUND, and this comment used to say it
#: did — *"the rule's source is measurably different on 0.5.1 (sha1
#: ``f5f2d0057376``), so probing there would be probing a function this tool
#: has never read"*. 0.5.1 is ABOVE 0.4.8, so the floor does not exclude it and
#: it IS probed. The differing hash is a disclosure that the rule's source
#: moves inside the range this tool will arm on — which is exactly why §5 makes
#: the probe the authority and records the hash rather than gating on it — and
#: not an argument for where the floor sits.
#:
#: THE PRECISION IS NOT MEASURED AND SAYS SO. Nothing here has been run on
#: 0.4.8, and stelling's own floor is ``jax>=0.10`` anyway
#: (``pyproject.toml``), so in every environment stelling supports this bound
#: is inert. It is a refusal boundary, not a support claim.
_FLOOR = (0, 4, 8)

#: The series with a CI lane, from the one place that fact is kept.
_TESTED = _optional.TESTED_JAX_SERIES

#: sha1[:12] of the rule's source. Byte-identical on 0.10.2 and 0.11.0,
#: independently re-derived twice. RECORDED, NOT GATED ON: a cosmetic edit
#: upstream must not disable the tool, but a changed hash in the status line
#: is what makes the nightly canary diagnosable without re-running it.
_KNOWN_HASH = "c808b3001114"

#: The rule's name on 0.5.1, 0.10.2 and 0.11.0. Also recorded, not gated on.
_KNOWN_RULE = "_convert_elt_type_folding_rule"

#: Frames that open a new traced region. See :func:`record.attribute` for the
#: two measured stacks that make this necessary; the names live here because
#: this is the module that is allowed to know things about jax's internals.
TRACE_ENTRY_NAMES = record.DEFAULT_TRACE_ENTRY_NAMES

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")

# Module state. One process arms once; ``install`` refuses to double-wrap and
# ``restore`` refuses to clobber someone else's patch.
_installed: dict = {}

#: Bumped once per :func:`selfcheck`, and used as the probe's input length.
#: jax's trace cache is process-wide, so a probe that traced the same avals
#: twice would reach the const-fold site zero times the second time. Measured.
_probe_seq: int = 0


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """``(major, minor, micro)``, or None if unparseable.

    Survives nightlies (``0.4.36.dev20240101``) by reading the leading three
    components and ignoring the rest. **An unparseable version means probe
    anyway** — §5 — so this returning None is not a refusal.
    """
    match = _VERSION_RE.match(text or "")
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def jax_version() -> str | None:
    """The installed jax version, without importing jax."""
    return _optional.version("jax")


def jax_root() -> str:
    """The directory prefix that marks a frame as jax's own.

    A string, and only ever a string: :mod:`record` filters frames by it and
    must not learn what jax is. Empty when jax is not importable, which makes
    :func:`record.attribute` treat every frame as the user's — lenient, and
    unreachable in practice since nothing gets that far without jax.
    """
    try:
        module = importlib.import_module("jax")
    except ImportError:
        return ""
    path = getattr(module, "__file__", None)
    return os.path.dirname(path) + os.sep if path else ""


def _registry():
    """The registry dict, or None. Never raises, never leaves this module."""
    try:
        module = importlib.import_module(PRIVATE_MODULE)
    except ImportError:
        return None
    return getattr(module, REGISTRY_ATTR, None)


def _primitive():
    """``convert_element_type_p``, from the PUBLIC surface that exports it."""
    from stelling._jax_compat import jax as _jax  # public jax, via the boundary

    importlib.import_module("jax.extend.core.primitives")
    return _jax.extend.core.primitives.convert_element_type_p


def locate() -> str:
    """Report whether the const-fold registry is where the tripwire expects it.

    A code from the tripwire's fail-closed vocabulary, never an exception and
    never a jax object:

    ``no-module``
        jax is not installed. Static checking is unaffected by this.
    ``no-registry``
        jax is installed and the private module or its registry is not where
        this expects it — a series moved it, which is what a tool keyed on a
        private surface is supposed to survive by refusing rather than by
        guessing.
    ``no-entry``
        the registry is there and has no rule for ``convert_element_type``.
        Distinguishable from ``no-registry`` because the registry's size is
        itself a measured fact — 3 entries on both tested series.
    ``located``
        the registry is there and the rule is in it.

    Codes are stable and greppable.
    """
    if not _optional.available("jax"):
        return "no-module"
    registry = _registry()
    if registry is None:
        return "no-registry"
    try:
        primitive = _primitive()
    except Exception:  # noqa: BLE001 - a probe may not raise
        return "no-registry"
    return "located" if primitive in registry else "no-entry"


def registry_size() -> int | None:
    """How many rules the registry holds. 3 on both tested series — the fact
    that tells ``no-registry`` (nothing there) from ``no-entry`` (a registry
    that no longer keys our primitive)."""
    registry = _registry()
    return None if registry is None else len(registry)


def rule_hash() -> str | None:
    """sha1[:12] of the installed rule's source, or None if unreadable.

    Recorded in the status so a change upstream is *visible*. Never gated on:
    refusing to arm because someone reflowed a comment is the failure mode §5
    names explicitly.
    """
    registry = _registry()
    if registry is None:
        return None
    try:
        primitive = _primitive()
        rule = _installed.get("original") or registry.get(primitive)
        if rule is None:
            return None
        return hashlib.sha1(inspect.getsource(rule).encode()).hexdigest()[:12]
    except Exception:  # noqa: BLE001 - a disclosure may not raise either
        return None


def rule_name() -> str | None:
    """The installed rule's ``__name__``, for the same reason as :func:`rule_hash`."""
    registry = _registry()
    if registry is None:
        return None
    try:
        primitive = _primitive()
        rule = _installed.get("original") or registry.get(primitive)
        return getattr(rule, "__name__", None)
    except Exception:  # noqa: BLE001
        return None


def version_check() -> tuple[str, str]:
    """``(code, disclosure)``. ``code`` is ``"ok"``, ``"below-floor"``, or
    ``"untested"``; the disclosure is a sentence for the report.

    Above the tested range this arms anyway and says so. Refusing on every new
    jax release makes the tool useless the day jax ships — §5.
    """
    text = jax_version() or "unknown"
    parsed = _parse_version(text)
    if parsed is None:
        return "ok", (
            f"jax {text}: version unparseable, so the probe is the only "
            f"authority here. Tested against {', '.join(_TESTED)}."
        )
    if parsed < _FLOOR:
        floor = ".".join(str(n) for n in _FLOOR)
        return "below-floor", (
            f"jax {text} is below the tripwire's floor of {floor}, where the "
            "const-fold rule's source is measurably different. Refusing to "
            "probe rather than guessing."
        )
    if _optional.jax_series_tested(text):
        return "ok", f"jax {text}, a series with a CI lane."
    return "untested", (
        f"tested against {', '.join(_TESTED)}, running against {text}. Armed "
        "anyway — the probe below is the contract, the version is a "
        "disclosure."
    )


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------

# THIS FILE, and only this file. Stripped before attribution so the wrapper
# never appears in its own report.
#
# NOT the whole ``_tripwire`` directory, which is what it was first and which
# was wrong in a way that mattered: ``_probe.py`` lives next door, and
# stripping the package would have removed the self-check's own writer frame,
# sent its finding to the suppressed bucket, and made ``selfcheck`` report
# ``not-invoked`` on a perfectly live hook. Measured, not reasoned about — it
# is what the first run of the self-check did.
_OWN_FILE = os.path.abspath(__file__)


def _stack(skip: int) -> tuple[record.Frame, ...]:
    """The current stack as primitive tuples, outermost first, without our own.

    Hand-walked rather than :func:`traceback.extract_stack` because that reads
    and caches every source file in the stack; this runs only on a fire, but a
    suite with many fires should not pay for it repeatedly.
    """
    frames: list[record.Frame] = []
    frame = sys._getframe(skip)
    while frame is not None:
        code = frame.f_code
        filename = code.co_filename
        if os.path.abspath(filename) != _OWN_FILE:
            frames.append((filename, frame.f_lineno, code.co_name))
        frame = frame.f_back
    frames.reverse()
    return tuple(frames)


def _int_or_none(value) -> int | None:
    """A Python int from a folded constant, or None.

    MEASURED, and the shape check is a CHEAP EARLY RETURN rather than a crash
    guard. The rule is invoked with NON-SCALAR constants too (it returns None
    for them, so nothing folds), and ``int()`` on those raises ``TypeError:
    only 0-dimensional arrays can be converted to Python scalars`` — re-driven
    on ``np.ndarray`` and ``jax.Array`` alike.

    This used to say a wrapper without the check "would crash the user's
    trace on the first array constant it met". It would not, twice over: the
    ``except`` two lines below catches ``TypeError`` itself, and the wrapper's
    own ``except Exception`` catches anything that escaped and counts it in
    ``internal_errors``. What the check buys is not safety but silence — no
    raised-and-caught exception per non-scalar constant, and no
    ``internal_errors`` count for something entirely ordinary.
    """
    if value is None:
        return None
    if getattr(value, "shape", ()) != ():
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


# Per-THREAD stack of gate fire counters. Each _pipeline invocation pushes
# a 0; the wrapper increments stack[-1]; _pipeline pops and reads its own
# count. Empty when no gate is active — selfcheck and session-wide
# accumulation go only to recorder.fires.
#
# Thread-local because the stack is POSITIONAL: push/increment/pop must be
# on the same thread to maintain LIFO semantics. A shared list under a
# ThreadPoolExecutor would let thread A's narrowing increment thread B's
# counter (the top of a shared stack is whoever pushed last, not whoever
# is tracing). JAX tracing is single-threaded, but a test harness in a
# ThreadPoolExecutor is not JAX's to forbid.
import threading as _threading

_gate_fire_local = _threading.local()


def _gate_fire_stack() -> list[int]:
    """The current thread's gate-fire stack (created on first access)."""
    try:
        return _gate_fire_local.stack
    except AttributeError:
        _gate_fire_local.stack = []
        return _gate_fire_local.stack


def _make_wrapper(original, recorder: record.Recorder, jaxroot: str):
    """Build the recording wrapper. Nothing in here may raise into a trace.

    ``*args, **kwargs`` rather than the measured ``(consts, params,
    out_avals)``. jax calls the rule positionally today; a release that added
    a keyword would make a fixed signature raise ``TypeError`` **inside the
    user's trace**, and a measurement instrument does not get to break the
    thing it measures. ``arm()``'s self-check catches that and disables the
    tool either way — that is the fail-closed floor — but delegating verbatim
    means a purely additive change upstream costs nothing at all.
    """

    def stelling_const_fold_probe(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            recorder.invocations += 1
            if result is None:
                return result
            recorder.folded += 1

            consts = (args[0] if args else kwargs.get("consts")) or ()
            params = (args[1] if len(args) > 1 else kwargs.get("params")) or {}

            written = _int_or_none(consts[0] if consts else None)
            became = _int_or_none(result[0] if result else None)
            from_dtype = str(getattr(consts[0], "dtype", "")) if consts else ""
            to_dtype = str(params.get("new_dtype", ""))

            if (
                written is None
                or became is None
                or from_dtype not in record.INT_DTYPES
                or to_dtype not in record.INT_DTYPES
            ):
                recorder.unmodelled += 1
                return result

            recorder.int_narrowings += 1
            if record.in_range(written, to_dtype):
                return result

            recorder.fires += 1
            stack = _gate_fire_stack()
            if stack:
                stack[-1] += 1
            frames = _stack(2)
            index, origin = record.attribute(frames, jaxroot, TRACE_ENTRY_NAMES)
            if origin == record.ORIGIN_JAX:
                recorder.suppressed_jax += 1
            elif origin == record.ORIGIN_UNKNOWN:
                recorder.unattributed += 1

            if index is None:
                file, line, func = (frames[-1] if frames else ("<unknown>", 0, "?"))
            else:
                file, line, func = frames[index]
            text = record.source_line(file, line)
            # Recorded whatever the origin: §10a.9 — a silent filter is
            # indistinguishable from a blind instrument, so a suppressed fire
            # is named, counted, and kept in its own bucket rather than
            # dropped. :meth:`record.Recorder.add` routes it by origin.
            recorder.add(
                record.Finding(
                    file=file,
                    line=line,
                    func=func,
                    written=written,
                    from_dtype=from_dtype,
                    to_dtype=to_dtype,
                    became=became,
                    origin=origin,
                    chain=record.user_chain(frames, jaxroot, TRACE_ENTRY_NAMES),
                    literal_visible=str(written) in text,
                )
            )
        except Exception:  # noqa: BLE001
            # An instrument that breaks the suite it measures is worse than no
            # instrument. Counted, disclosed in the report, never raised.
            try:
                recorder.internal_errors += 1
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                pass
        return result

    return stelling_const_fold_probe


def install(recorder: record.Recorder) -> str:
    """Wrap the rule. Returns a status code; never raises, never leaks a jax object.

    ``already-armed`` if this process already installed one — arming twice must
    not double-wrap, and a double wrapper would double every count.
    """
    code = locate()
    if code != "located":
        return code
    if _installed:
        return "already-armed"
    try:
        registry = _registry()
        primitive = _primitive()
        original = registry[primitive]
        wrapper = _make_wrapper(original, recorder, jax_root())
        registry[primitive] = wrapper
        _installed.update(
            {"registry": registry, "primitive": primitive,
             "original": original, "wrapper": wrapper, "recorder": recorder}
        )
    except Exception as exc:  # noqa: BLE001
        _installed.clear()
        return f"unexpected:{type(exc).__name__}"
    return "installed"


def restore() -> str:
    """Put the original rule back, **by identity**.

    ``not-armed``
        nothing to restore.
    ``foreign-patch``
        something else replaced our wrapper after we installed it. Say so
        rather than silently clobbering it — §4.
    ``restored``
        done.
    """
    if not _installed:
        return "not-armed"
    registry = _installed["registry"]
    primitive = _installed["primitive"]
    current = registry.get(primitive)
    if current is not _installed["wrapper"]:
        _installed.clear()
        return "foreign-patch"
    registry[primitive] = _installed["original"]
    _installed.clear()
    return "restored"


#: Saved state for :func:`detach`. Separate from ``_installed`` because
#: detaching is deliberately something that happens *to* an armed tripwire.
_detached: dict = {}


def detach(mode: str) -> str:
    """Break the tripwire the way a jax release would. Returns a status code.

    THIS IS SHIPPED CODE WITH A TEST-SUPPORT PURPOSE, and it is here rather
    than in ``tests/`` for one reason: rule 2 bans naming the private jax
    module in ``tests/`` with no exemption, so a test that reached into the
    registry to break it would have to name what only this file may name. The
    alternative — asserting the fail-closed contract instead of driving it —
    is the shape of control this repository has been burned by.
    ``design/private-jax-boundary.md`` records the rule; ``PLAN-tripwire.md``
    §9 records the two rows this exists for.

    ``mode="entry"``
        remove the rule the tripwire keys on. :func:`locate` then answers
        ``no-entry``, which is the "anchor removed" row.
    ``mode="bypass"``
        put the original rule back under an armed tripwire, so the wrapper is
        attached and never invoked. That is what a version bump actually
        produces, and what a presence check misses.

    :func:`reattach` undoes either. Nothing here leaks a jax object: both
    arguments and both return values are strings.
    """
    registry = _registry()
    if registry is None:
        return "no-registry"
    try:
        primitive = _primitive()
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"
    if _detached:
        return "already-detached"
    current = registry.get(primitive)
    _detached.update({"registry": registry, "primitive": primitive, "entry": current})
    if mode == "entry":
        registry.pop(primitive, None)
        return "detached"
    if mode == "bypass":
        registry[primitive] = _installed.get("original", current)
        return "detached"
    _detached.clear()
    return f"unknown-mode:{mode}"


def reattach() -> str:
    """Undo :func:`detach`. ``not-detached`` if there is nothing to undo."""
    if not _detached:
        return "not-detached"
    registry = _detached["registry"]
    primitive = _detached["primitive"]
    entry = _detached["entry"]
    if entry is None:
        registry.pop(primitive, None)
    else:
        registry[primitive] = entry
    _detached.clear()
    return "reattached"


def is_armed() -> bool:
    """Whether the wrapper this process installed is still the live entry."""
    if not _installed:
        return False
    return _installed["registry"].get(_installed["primitive"]) is _installed["wrapper"]


def live_check() -> str:
    """Whether the tripwire is STILL live, and if not, which way it went.

    :func:`is_armed` answers only yes or no, and the two nos are different
    failures a user has to be told apart:

    ``armed``
        our wrapper is still the live registry entry.
    ``detached``
        we hold no installation any more. Something called :func:`restore` —
        a mid-session ``disarm()``, or a NESTED pytest session that also
        enabled the tripwire, whose ``pytest_unconfigure`` restores the
        original and disarms the outer one along with itself.
    ``foreign-patch``
        we still hold an installation and the live entry is not ours: someone
        rebound the registry over us. The same condition :func:`restore`
        reports, reached before the report is written instead of after.

    Returns a string, like everything else here.
    """
    if not _installed:
        return "detached"
    if _installed["registry"].get(_installed["primitive"]) is _installed["wrapper"]:
        return "armed"
    return "foreign-patch"


def selfcheck() -> str:
    """Probe the armed hook **in both directions**, and return a status code.

    A positive-only self-check passes on a hook replaced by "record
    everything", and this repository has repeatedly found exactly that shape
    of vacuous control. So:

    ``not-invoked``
        a trace that must produce exactly one finding produced none. The
        wrapper is attached and blind — which is what a version bump actually
        produces, and what a naive ``hasattr`` misses.
    ``cries-wolf``
        a trace whose value **fits** produced a finding. The semantics moved;
        every "fire" this run would be noise.
    ``mis-attributed``
        the hook fired the right number of times and got the content wrong —
        the wrong value, the wrong dtype, the wrong file or the wrong line, or
        a narrowing whose independent recomputation disagrees with what was
        observed. A finding that points at the wrong line is worse than no
        finding, so it disables the tool rather than shipping.
    ``armed``
        all three behaved.

    Both directions reach the const-fold site — measured, one invocation each
    — and they differ in whether a *finding* results, not in whether the hook
    ran. That is what makes ``not-invoked`` and ``cries-wolf`` two different
    failures rather than two names for one, and it is why the negative
    direction also asserts ``invocations > 0``: a hook that stopped running
    altogether would otherwise pass the negative direction by producing
    nothing, exactly as a correct one does.

    The probe runs ``_probe.over`` / ``_probe.under`` rather than local
    lambdas so that the attributed frame is a real module — the same path a
    user's code takes, including the stack walk and the source quote.

    EACH RUN USES A FRESH INPUT SHAPE, and that is not tidiness. Measured on
    both tested series: jax's trace cache is process-wide and outlives
    disarm/rearm, so tracing ``_probe.over`` at the *same* avals reaches the
    const-fold site ``[1, 0, 0]`` times over three runs — once, and then never
    again. (``f86bafe`` recorded that sequence as "same shape three times, 0
    invocations each", which is wrong in its first element: the first trace
    does reach the site, which is the only reason a fixed-shape probe passed
    at all. Shapes 1, 2, 3 give ``[1, 1, 1]``, as that commit says.) A probe with a fixed shape therefore passes exactly
    once per process and reports ``not-invoked`` for ever after — which would
    make a second ``arm()`` in one process, and every test that arms twice,
    look like a broken hook. The shape is the cheapest thing in the cache key
    that is ours to vary.

    The recorder's counters are saved and restored: a self-check must not
    appear in the user's denominator.

    IT TAKES NO RECORDER, and that is a correction rather than a style
    choice. It used to take one, and a caller who passed anything other than
    the recorder :func:`install` closed over got ``not-invoked`` from a
    perfectly live hook — the probe read a counter nothing was writing to.
    The wrapper's recorder is the only one that can answer, so the wrapper's
    recorder is the one this reads.
    """
    recorder = _installed.get("recorder")
    if recorder is None:
        return "not-invoked"
    from stelling._jax_compat import jax as _jax  # public jax, via the boundary
    from stelling._jax_compat import jnp as _jnp

    from stelling._tripwire import _probe

    global _probe_seq
    _probe_seq += 1
    shape = (_probe_seq,)

    saved = recorder.as_payload()
    # Isolate the probe from any active gate: push our own stack entry so
    # probe fires go HERE and not into a gate that happens to be active.
    stack = _gate_fire_stack()
    stack.append(0)
    try:
        recorder.reset()
        try:
            _jax.make_jaxpr(_probe.over)(_jnp.zeros(shape, _jnp.int8))
        except Exception as exc:  # noqa: BLE001
            return f"unexpected:{type(exc).__name__}"
        if recorder.invocations == 0:
            return "not-invoked"
        if recorder.count != 1:
            return "not-invoked"

        found = next(iter(recorder.findings.values()))
        if (
            found.written != _probe.OVER
            or found.to_dtype != _probe.DTYPE
            or not found.agrees
            or os.path.abspath(found.file) != os.path.abspath(_probe.__file__)
            or found.line != _probe.OVER_LINE
        ):
            return "mis-attributed"

        recorder.reset()
        try:
            _jax.make_jaxpr(_probe.under)(_jnp.zeros(shape, _jnp.int8))
        except Exception as exc:  # noqa: BLE001
            return f"unexpected:{type(exc).__name__}"
        if recorder.invocations == 0:
            return "not-invoked"
        if recorder.count != 0:
            return "cries-wolf"
        return "armed"
    finally:
        stack.pop()
        recorder.reset()
        recorder.absorb(saved)
