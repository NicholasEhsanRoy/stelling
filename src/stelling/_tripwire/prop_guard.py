# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

# VENDORED, NOT WRITTEN HERE, AND DELIBERATELY NOT REWRITTEN. This file is
# `/home/nick/MSF/stelling-sweeps/artifacts/prop_guard.py` -- the mitigation
# bundle of the 0.2.0 dunder-perimeter fuzz round -- copied in with FIVE edits
# and no others, so that a reviewer can `diff` it against the artefact and see
# the whole of what changed:
#
#   1. these SPDX lines and this note;
#   2. `_jnp()` takes `jnp` from `stelling._jax_compat` instead of importing
#      `jax.numpy` itself;
#   3. `_x64()` takes `jax` from the same place;
#   4. `_selftest()` takes both from the same place;
#   5. `_target_dtype()` RETURNS on its own except branch instead of caching
#      the `None` that branch produced -- one line, and the only edit that
#      touches behaviour at all. See below.
#
# Edits 2-4 exist because this repository allows the `import jax` / `from jax`
# token in exactly one file (`_jax_compat.py`) and
# `tests/test_import_hygiene.py::test_jax_imported_only_in_compat_module`
# holds that shut. They are import-route changes only: the same modules, the
# same laziness, the same objects. NOTHING in the predicate itself was
# touched, and `tests/test_narrowing_perimeter.py` re-runs the artefact's own
# 24-case self-test against this copy.
#
# EDIT 5, AND WHY IT DOES NOT SPEND THE ARTEFACT'S EVIDENCE. `_target_dtype`
# memoises on `(dtype, is-float-slot, x64)` and wrote its result into the memo
# on EVERY path, its own `except` branch included -- so a single transient
# fault cached `None` for that key and the guard was blind on it for the rest
# of the process, having counted the failure exactly ONCE. It is reachable
# through public API: the `truediv` branch allocates `jnp.zeros((0,), dt) / 1`
# and `jax.transfer_guard` -- documented, and used by real code -- makes that
# allocation raise. Driven end to end, jax 0.11.0 at x64=0: after the guard
# window CLOSES the reference defect fires 0 times in 20, over a run of 21
# checks whose report named exactly ONE decline while twenty more were
# answered out of the memo. With the return, the same drive fires 20 of 20 and
# the memo is empty.
#
# The change is ANSWER-PRESERVING ON THE SCORED CORPUS, which is what keeps
# the census below evidence about this copy: `INTERNAL_DECLINES` was empty in
# all nine scored configurations, so this branch was never taken during
# scoring and no scored answer can differ. What it changes is only what
# happens AFTER a fault -- the next call recomputes rather than reading a
# cached blindness -- which is the fail-safe the module already documents.
#
# WHY IT IS NOT REWRITTEN. It came out of a 204,300-evaluation property census
# and was then scored, by a context that did not write it, over 482,691
# guard-eligible checks of real code. That census covered jax's own shipped
# harness corpus, the shipped test suites of optax and of jax-md (census), and
# eight eval-target libraries driven both normally and under
# `JAX_DISABLE_JIT=1`, which turns a library's whole interior into concrete
# ops -- ZERO false positives, twice end to end, `UNKNOWN_SLOTS` and
# `INTERNAL_DECLINES` empty in all nine configurations. Two hand-written
# predicates that preceded it were measurably wrong. The thirteen mitigations
# F1-F13 in the docstring below each name the finding they implement; none of
# them is decoration.

# Provenance: PROP, the mitigation bundle of FUZZ-dunder-perimeter.md (stelling
# 0.2.0 fuzz round), measured at commit 4e17d986475303a4183e1cebf20eac513425d856.
"""PROP -- the dunder-perimeter narrowing guard, standalone.

WHAT THIS IS
============
One question, asked of a Python ``int`` literal met in a binary op with a jax
array or tracer:

    *Does the integer the author wrote exist, exactly, in the program jax will
    actually run?*

If it does, ``classify`` returns ``None``.  If it does not, ``classify``
returns a :class:`Finding` naming the reason, the dtype the literal was
converted into, and the value it became.

This module is the guard only.  It installs nothing, rebinds no slots, raises
nothing of its own unless you ask it to (:func:`raise_for`), and has no
dependency on the fuzz harness that produced it.  It imports only ``numpy``,
``jax``, ``ml_dtypes`` and the standard library, all lazily, and uses **no
private jax API**.

ENTRY POINT
===========
::

    classify(a, b, slot) -> None | Finding

``a``
    The **array operand** -- a ``jaxlib._jax.ArrayImpl``, a ``Tracer``, or
    anything exposing ``.dtype`` and (optionally) ``.size``.  It must be the
    array, not the dtype: the size-0 exemption needs ``.size``.  If ``.size``
    is absent the exemption is simply not applied (the guard checks anyway),
    so passing a dtype-like object is safe but slightly more trigger-happy.

``b``
    The literal.  Only ``type(b) is int`` is examined; ``bool`` is a subclass
    of ``int`` and is deliberately **excluded** (``type(True) is bool``).
    Anything else returns ``None`` immediately.

``slot``
    The dunder slot the op came through: ``"__add__"``, ``"__rpow__"``,
    ``"__le__"`` ...  Bare forms (``"add"``, ``"rpow"``) are accepted too.
    **The slot is load-bearing** and must be supplied -- see CONTRACT below.

Returns ``None`` (nothing wrong, or the guard declined) or a :class:`Finding`.

``Finding`` carries:

    ``reason``          one of :data:`REASONS`
    ``slot``            the normalised slot name
    ``operand_dtype``   ``str`` of the array's dtype
    ``target_dtype``    ``str`` of the dtype the literal is converted INTO,
                        which is often NOT the array's dtype (``int16 / lit``
                        converts into ``float32``)
    ``literal``         the written ``int``
    ``narrowed_to``     the value the program uses instead: an ``int`` for the
                        integer reasons, a ``float`` for ``inexact``, and
                        ``float('inf')`` / ``float('-inf')`` / ``float('nan')``
                        for ``overflows-float``
    ``message``         a one-line human-readable rendering

CONTRACT -- what the caller must supply, and what happens if it does not
=======================================================================
1. **The slot name is required and must be one of** :data:`KNOWN_SLOTS`.
   Two mitigations are keyed on it and cannot be applied without it:
   ``__pow__`` is excluded (jax keeps the exponent as a Python int in
   ``integer_pow[y=...]``; it is never converted, so checking it is a false
   positive) while ``__rpow__`` is kept (it *does* narrow), and
   ``__truediv__`` / ``__rtruediv__`` redirect the literal into a float.
   An unrecognised slot is handled as a dtype-preserving op -- the majority
   behaviour, and the conservative choice -- and is recorded in
   :data:`UNKNOWN_SLOTS`.  **Print that set at the end of a run**; a silently
   mistyped slot name is how a guard goes blind.

2. **The x64 cell is read at call time, not cached across a flip.** Promotion
   targets are memoised per ``(dtype, slot, jax_enable_x64)``, because a
   process can change its own x64 setting mid-run.

3. **The guard never raises.** Any internal failure returns ``None`` and is
   recorded in :data:`INTERNAL_DECLINES`.  **Print that too.**  A fail-safe
   path that fails into silence is the defect this guard exists to avoid;
   :func:`report` prints both counters.

WHAT IT DOES AND WHY -- the thirteen measured mitigations
=========================================================
Each line names the finding in ``FUZZ-dunder-perimeter.md`` it implements.

F1  Float-ness is decided by ``ml_dtypes.finfo``, never by ``dtype.kind`` and
    never by ``np.finfo``.  ``np.dtype(bfloat16).kind`` is ``'V'``, not
    ``'f'``, and so is ``float8_e4m3fn``'s -- while ``float8_e5m2``'s is
    ``'f'``.  ``np.finfo`` raises on all three.
F2  ``inf``/``nan`` are tested BEFORE ``Fraction`` is constructed.
    ``Fraction(float('inf'))`` raises ``OverflowError``; without this,
    ``x_f16 <= 100000`` throws out of the guard (65520 is the threshold).
F3  The conversion catches ``TypeError`` as well as ``OverflowError`` and
    ``ValueError``: ml_dtypes raises ``TypeError: expected number, got int``
    for ``|lit| >= 2**64``.  F1's carve-out reaches that path, so F1 without
    F3 trades a miss for a fault.
F4  Rule 0: an operand whose dtype is not an ``np.dtype`` -- a jax EXTENDED
    dtype such as ``key<fry>`` -- is declined before any attribute is touched.
    ``KeyTy`` has no ``.kind``; reading it raises ``AttributeError``, and that
    fault is reachable through a real ``Tracer.__add__``.
F5  ``__pow__`` excluded, ``__rpow__`` kept.
F6  The target dtype is derived from jax's own promotion, per slot, not from
    ``a.dtype`` and not from a hand-written table.
F7  ``bool`` arrays need no special case once F6 is in: they promote to the
    default int, and to the default float under ``/``.
F8  A complex target is checked on its REAL part's dtype -- an int into
    ``complex64`` lands in a ``float32`` real part.
F9  Integer bounds come from ``np.iinfo`` falling back to ``ml_dtypes.iinfo``,
    and the wrap modulus is derived from those BOUNDS, never from
    ``itemsize``: ``int4`` occupies a whole byte but wraps modulo 16.
F10 A negative literal against an unsigned target gets its own reason,
    ``negative-into-unsigned``.  The verdict is the same but the advice is
    not: the sign flip inverts the predicate (``uint8 >= -3`` runs as
    ``>= 253``), and "out of range for uint8" wrongly invites widening.
F11 A size-0 operand is declined.  Measured: 2,988 (literal, narrowed-image)
    outcome comparisons on empty arrays across all 34 slots, zero differences.
F12 :class:`NarrowingError` subclasses ``OverflowError``, so a caller that
    raises on a Finding does not break code that already catches jax's own
    ``OverflowError`` for an out-of-default-int-width literal.
F13 ``weak_type`` and 0-d operands need no special case; measured, not assumed.

The promotion identity behind F6 was checked against the dtype jax actually
converts to, on jax 0.10.2 / 0.11.0 / 0.11.1 in both x64 cells: 315 agree, 0
disagree, in each of the six configurations.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

__all__ = [
    "classify", "raise_for", "Finding", "NarrowingError", "REASONS",
    "KNOWN_SLOTS", "UNKNOWN_SLOTS", "INTERNAL_DECLINES", "report",
]

# --------------------------------------------------------------- lazy imports
# Imported on first use so that importing this module cannot force a jax import
# before the caller has set its config flags.

_JNP = None
_ML = None
_JAX = None


def _jnp():
    global _JNP
    if _JNP is None:
        from stelling._jax_compat import jnp      # VENDORING EDIT 2
        _JNP = jnp
    return _JNP


def _ml():
    global _ML
    if _ML is None:
        import ml_dtypes
        _ML = ml_dtypes
    return _ML


def _x64():
    """The x64 cell, read FRESH on every call.  Only the module object is
    cached: a process can flip `jax_enable_x64` mid-run (JAXFLUIDS does), and
    a stale reading would give the wrong promotion target for every dtype."""
    global _JAX
    if _JAX is None:
        from stelling._jax_compat import jax      # VENDORING EDIT 3
        _JAX = jax
    return bool(_JAX.config.jax_enable_x64)


# ---------------------------------------------------------------- the slot set

_FORWARD = (
    "add", "sub", "mul", "truediv", "floordiv", "mod", "divmod", "pow",
    "lshift", "rshift", "and", "xor", "or", "matmul",
)
_COMPARISON = ("lt", "le", "eq", "ne", "gt", "ge")
_BARE_SLOTS = _FORWARD + tuple("r" + n for n in _FORWARD) + _COMPARISON

#: every slot the perimeter can install, in dunder form
KNOWN_SLOTS = frozenset("__%s__" % n for n in _BARE_SLOTS)

#: slots on which NO literal is converted to any dtype (F5).  jax lowers
#: `x ** k` to `integer_pow[y=k]`, keeping `k` a Python int in the program's
#: structure; it does not even reach StableHLO as a constant.
_NO_CONVERSION = frozenset({"pow"})

#: slots whose literal is promoted into a float rather than the array's dtype
_FLOAT_SLOTS = frozenset({"truediv", "rtruediv"})

#: slot names seen that are not in KNOWN_SLOTS.  SURFACE THIS.
UNKNOWN_SLOTS: set = set()

#: internal failures, keyed by exception type name.  SURFACE THIS.
INTERNAL_DECLINES: Counter = Counter()


def _normalise(slot) -> str:
    s = str(slot)
    if s.startswith("__") and s.endswith("__"):
        s = s[2:-2]
    if "__%s__" % s not in KNOWN_SLOTS:
        UNKNOWN_SLOTS.add(str(slot))
    return s


# ------------------------------------------------------------------ the result

#: the closed set of reason codes `Finding.reason` can carry
REASONS = ("out-of-range", "negative-into-unsigned", "inexact",
           "overflows-float")


class NarrowingError(OverflowError):
    """Raised by :func:`raise_for`.

    Subclasses ``OverflowError`` on purpose (F12): jax itself raises
    ``OverflowError`` for a literal outside the default int width, and an
    armed guard fires first.  Subclassing keeps existing ``except
    OverflowError`` clauses working.
    """

    def __init__(self, finding: "Finding"):
        super().__init__(finding.message)
        self.finding = finding


@dataclass(frozen=True)
class Finding:
    reason: str
    slot: str
    operand_dtype: str
    target_dtype: str
    literal: int
    narrowed_to: object

    @property
    def message(self) -> str:
        if self.reason == "negative-into-unsigned":
            what = ("a negative literal cannot exist in the unsigned type it "
                    "is compared against")
        elif self.reason == "out-of-range":
            what = "outside the range of"
        elif self.reason == "inexact":
            what = "not exactly representable in"
        else:
            what = "overflows"
        return (
            "the literal %r written in `%s` is %s %s: the program uses %r"
            % (self.literal, "__%s__" % self.slot, what, self.target_dtype,
               self.narrowed_to)
        )

    def __str__(self) -> str:          # pragma: no cover - convenience only
        return self.message


# ------------------------------------------------------------- dtype questions

def _is_float_like(dt) -> bool:
    """F1.  True for every numpy float AND every ml_dtypes float, including the
    ones whose `kind` is 'V' (bfloat16, float8_e4m3fn) -- and False for the
    ones that only look float-ish (int4, uint4, float0).  `np.finfo` is NOT
    used: it raises on bfloat16 and on both float8 types."""
    try:
        _ml().finfo(dt)
        return True
    except Exception:
        return False


def _int_bounds(dt):
    """F9.  (min, max) for any integer-like dtype, including the sub-byte
    ml_dtypes ones on which `np.iinfo` raises.  None if not integer-like."""
    try:
        i = np.iinfo(dt)
        return int(i.min), int(i.max)
    except Exception:
        pass
    try:
        i = _ml().iinfo(dt)
        return int(i.min), int(i.max)
    except Exception:
        return None


def _wrap(lit: int, lo: int, hi: int) -> int:
    """F9.  The two's-complement image, with the modulus taken from the BOUNDS
    rather than from itemsize -- int4 occupies a byte but wraps modulo 16."""
    mod = hi - lo + 1
    return int((lit - lo) % mod + lo)


def _real_dtype(dt):
    """F8.  A complex dtype's real component; anything else unchanged."""
    try:
        if dt.kind == "c":
            return np.dtype(dt.type(0).real.dtype)
    except Exception:
        pass
    return dt


# ---------------------------------------------------- the promotion target (F6)

_TARGET_CACHE: dict = {}


def _target_dtype(dt, slot):
    """The dtype the literal is ACTUALLY converted into for this slot.

    Both branches are jax's own promotion, asked of jax, using only public API:

      * dtype-preserving slots -- `jnp.result_type(dt, 1)`, which is the
        weak-int promotion jax performs.  This is what makes the answer right
        for `bool` (-> the default int) and for `int64` at x32 (-> int32).
      * `truediv` / `rtruediv` -- the dtype of `zeros((0,), dt) / 1`.  A
        size-0 array makes this nearly free, and it is exactly the promotion
        `true_divide` performs; measured equal to the private
        `to_inexact_dtype(result_type(dt, 1))` on all 19 dtypes in both x64
        cells, and unlike `result_type(dt, 1.0)` -- which is WRONG at x64,
        giving float64 where jax uses float32 -- it needs no private import.

    Returns None when nothing is converted (F5) or the target cannot be
    determined.
    """
    if slot in _NO_CONVERSION:
        return None
    key = (str(dt), slot in _FLOAT_SLOTS, _x64())
    if key in _TARGET_CACHE:
        return _TARGET_CACHE[key]
    jnp = _jnp()
    try:
        if slot in _FLOAT_SLOTS:
            out = np.dtype((jnp.zeros((0,), dt) / 1).dtype)
        else:
            out = np.dtype(jnp.result_type(dt, 1))
    except Exception as exc:
        INTERNAL_DECLINES[type(exc).__name__] += 1
        # EDIT 5: `return None` rather than falling through to the memo. A
        # value this call did not compute must not be cached: one transient
        # fault -- `jax.transfer_guard`, a public and documented API, closing
        # over the `zeros((0,), dt) / 1` allocation this branch makes -- would
        # otherwise blind the guard for that key for the REST OF THE PROCESS,
        # while `INTERNAL_DECLINES` recorded exactly 1. Declining is fail-safe
        # only while it is counted, and a memoised decline is counted once and
        # taken forever after. See the note at the top of this file.
        return None
    _TARGET_CACHE[key] = out
    return out


# -------------------------------------------------------------- the entry point

def classify(a, b, slot):
    """Is the literal ``b``, met in ``slot`` against array ``a``, in the
    program jax will run?  ``None`` if yes (or if the guard declines);
    a :class:`Finding` if no.  Never raises."""
    try:
        # bool is a subclass of int and is not a narrowing candidate
        if type(b) is not int:
            return None

        dt = getattr(a, "dtype", None)

        # F4 -- RULE 0, before any attribute of the dtype is touched.  A jax
        # EXTENDED dtype (key<fry>) has no `.kind`; reading it raises.
        if not isinstance(dt, np.dtype):
            return None

        # F11 -- a size-0 operand narrows nothing observably.
        size = getattr(a, "size", None)
        if size is not None:
            try:
                if int(size) == 0:
                    return None
            except Exception:
                pass

        name = _normalise(slot)

        # F5/F6/F7 -- ask about the dtype the literal actually enters.
        tgt = _target_dtype(dt, name)
        if tgt is None or not isinstance(tgt, np.dtype):
            return None

        def found(reason, narrowed):
            return Finding(reason=reason, slot=name, operand_dtype=str(dt),
                           target_dtype=str(tgt), literal=b,
                           narrowed_to=narrowed)

        # ---- integer targets, including the sub-byte ml_dtypes ones (F9)
        if tgt.kind != "b":
            bnds = _int_bounds(tgt)
            if bnds is not None:
                lo, hi = bnds
                if lo <= b <= hi:
                    return None
                narrowed = _wrap(b, lo, hi)
                if lo == 0 and b < 0:
                    return found("negative-into-unsigned", narrowed)   # F10
                return found("out-of-range", narrowed)

        # ---- float targets, including complex real parts (F8) and the
        #      ml_dtypes extension floats (F1)
        comp = _real_dtype(tgt)
        if not _is_float_like(comp):
            return None                      # bool, int4-as-non-float, opaque

        try:
            with np.errstate(over="ignore"):
                image = float(np.asarray(b, comp))
        except (OverflowError, ValueError, TypeError):
            # F3 -- ml_dtypes raises TypeError for |b| >= 2**64; numpy raises
            # OverflowError for an int too large to convert to float at all.
            return found("overflows-float", float("inf") if b > 0
                         else float("-inf"))

        # F2 -- test inf/nan BEFORE Fraction; Fraction(inf) raises.
        if image != image or math.isinf(image):
            return found("overflows-float", image)

        if Fraction(int(b)) != Fraction(image):
            return found("inexact", image)
        return None

    except Exception as exc:                 # a guard must never break a run
        INTERNAL_DECLINES[type(exc).__name__] += 1
        return None


def raise_for(a, b, slot):
    """:func:`classify`, but raise :class:`NarrowingError` on a Finding.
    Returns ``None`` otherwise.  The exception subclasses ``OverflowError``
    (F12)."""
    f = classify(a, b, slot)
    if f is not None:
        raise NarrowingError(f)
    return None


def report(stream=None):
    """Print the two counters a caller must not leave unread.

    A guard that fails safe into silence is the defect this campaign exists to
    close; run this at the end of any census."""
    import sys
    out = stream or sys.stdout
    print("[prop_guard] unknown slots seen : %s"
          % (sorted(UNKNOWN_SLOTS) or "none"), file=out)
    print("[prop_guard] internal declines  : %s"
          % (dict(INTERNAL_DECLINES) or "none"), file=out)


# ------------------------------------------------------------------- self-test

def _selftest():                                    # pragma: no cover
    import warnings
    warnings.simplefilter("ignore")
    import ml_dtypes

    from stelling._jax_compat import jax, jnp     # VENDORING EDIT 4

    print("prop_guard self-test  jax=%s numpy=%s ml_dtypes=%s x64=%s"
          % (jax.__version__, np.__version__, ml_dtypes.__version__,
             jax.config.jax_enable_x64))
    print("-" * 78)

    bf = ml_dtypes.bfloat16
    e5 = ml_dtypes.float8_e5m2

    def arr(dtype, shape=(2,)):
        return jnp.zeros(shape, dtype)

    # (label, operand, literal, slot, expected reason or None, expected narrowed
    #  value or None -- None means "do not check the value")
    CASES = [
        # --- the reference fires
        ("int16 + 40000",            arr("int16"),  40000,      "__add__",
         "out-of-range", -25536),
        ("float32 <= 2**31-1",       arr("float32"), 2**31 - 1, "__le__",
         "inexact", 2147483648.0),
        ("float16 <= 65520 (used to RAISE)",
         arr("float16"), 65520, "__le__", "overflows-float", float("inf")),
        ("bfloat16 + 40000",         arr(bf),       40000,      "__add__",
         "inexact", 39936.0),
        ("bfloat16 <= 2049",         arr(bf),       2049,       "__le__",
         "inexact", 2048.0),
        ("uint8 >= -3",              arr("uint8"),  -3,         "__ge__",
         "negative-into-unsigned", 253),
        ("complex64 + 16777217",     arr("complex64"), 16777217, "__add__",
         "inexact", 16777216.0),
        ("int4 + 9",                 arr(ml_dtypes.int4), 9,    "__add__",
         "out-of-range", -7),
        ("float8_e5m2 + 2**64 (TypeError path)",
         arr(e5), 2**64, "__add__", "overflows-float", None),
        ("40000 ** int16  (__rpow__ IS kept)",
         arr("int16"), 40000, "__rpow__", "out-of-range", -25536),

        # --- in-range controls, one per branch
        ("int16 + 3            (integer branch, in range)",
         arr("int16"), 3, "__add__", None, None),
        ("uint8 >= 3           (unsigned branch, in range)",
         arr("uint8"), 3, "__ge__", None, None),
        ("float32 <= 1000      (float branch, exact)",
         arr("float32"), 1000, "__le__", None, None),
        ("bfloat16 + 3         (ml_dtypes float branch, exact)",
         arr(bf), 3, "__add__", None, None),
        ("complex64 + 3        (complex branch, exact)",
         arr("complex64"), 3, "__add__", None, None),
        ("int4 + 3             (sub-byte branch, in range)",
         arr(ml_dtypes.int4), 3, "__add__", None, None),
        ("bool <= 2049         (promotes to the default int)",
         arr("bool"), 2049, "__le__", None, None),

        # --- the carve-outs
        ("int16 ** 40000       (__pow__ excluded, F5)",
         arr("int16"), 40000, "__pow__", None, None),
        ("int16 / 40000        (promotes into float32, F6)",
         arr("int16"), 40000, "__truediv__", None, None),
        ("uint8(0,) >= -3      (size-0 exemption, F11)",
         arr("uint8", (0,)), -3, "__ge__", None, None),
        ("int16 + True         (bool literal is not an int literal)",
         arr("int16"), True, "__add__", None, None),
    ]

    ok = bad = 0
    for label, operand, lit, slot, want_reason, want_val in CASES:
        try:
            f = classify(operand, lit, slot)
            raised = None
        except BaseException as exc:                # the guard must never raise
            f, raised = None, "%s: %s" % (type(exc).__name__, exc)
        got_reason = None if f is None else f.reason
        good = (raised is None) and (got_reason == want_reason)
        if good and want_val is not None:
            good = (f is not None
                    and (f.narrowed_to == want_val
                         or (isinstance(f.narrowed_to, float)
                             and isinstance(want_val, float)
                             and math.isinf(f.narrowed_to)
                             and math.isinf(want_val))))
        detail = raised or (
            "None" if f is None
            else "%s narrowed_to=%r target=%s" % (f.reason, f.narrowed_to,
                                                  f.target_dtype))
        print("%-4s %-42s -> %s" % ("PASS" if good else "FAIL", label, detail))
        ok, bad = (ok + 1, bad) if good else (ok, bad + 1)

    # --- the extended-dtype operand, which has no `.kind` at all (F4).
    #     Driven on BOTH faces: eager PRNGKeyArray and a traced key tracer.
    key = jax.random.key(0)
    try:
        f = classify(key, 1, "__add__")
        good = f is None
        detail = "None (declined)" if good else repr(f)
    except BaseException as exc:
        good, detail = False, "RAISED %s: %s" % (type(exc).__name__, exc)
    print("%-4s %-42s -> %s" % ("PASS" if good else "FAIL",
                                "key<fry> eager operand (F4)", detail))
    ok, bad = (ok + 1, bad) if good else (ok, bad + 1)

    seen = {}

    def _probe(k):
        try:
            seen["r"] = classify(k, 1, "__add__")
            seen["raised"] = None
        except BaseException as exc:
            seen["r"], seen["raised"] = None, "%s" % type(exc).__name__
        return k
    jax.make_jaxpr(_probe)(key)
    good = seen.get("raised") is None and seen.get("r") is None
    print("%-4s %-42s -> %s" % ("PASS" if good else "FAIL",
                                "key<fry> TRACER operand (F4)",
                                seen.get("raised") or "None (declined)"))
    ok, bad = (ok + 1, bad) if good else (ok, bad + 1)

    # --- raise_for is an OverflowError subclass (F12)
    try:
        raise_for(arr("int16"), 40000, "__add__")
        good, detail = False, "did not raise"
    except OverflowError as exc:
        good = isinstance(exc, NarrowingError)
        detail = "%s (caught as OverflowError)" % type(exc).__name__
    except BaseException as exc:
        good, detail = False, "wrong type %s" % type(exc).__name__
    print("%-4s %-42s -> %s" % ("PASS" if good else "FAIL",
                                "raise_for subclasses OverflowError (F12)",
                                detail))
    ok, bad = (ok + 1, bad) if good else (ok, bad + 1)

    print("-" * 78)
    report()
    print("%d passed, %d failed" % (ok, bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":                          # pragma: no cover
    import sys
    sys.exit(_selftest())
