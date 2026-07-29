# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The only module in stelling that imports jax.

Everything jax-shaped is transcribed here, once, into :mod:`stelling.ir`;
every analysis, encoder, and report consumes the IR and never jax. When jax
churns — and ``jax.extend`` explicitly reserves the right to — the blast
radius is this file.

Only public and ``jax.extend`` surfaces are used. Private jax modules are
banned repo-wide (enforced by a pre-commit hook and a test, not a comment).
"""

from __future__ import annotations

import enum
import functools
import math
import warnings

import jax
import jax.extend.core as jex_core
import jax.numpy as jnp
import jax.sharding
import jax.tree_util
import numpy as np

from stelling import ir
from stelling._optional import TESTED_JAX_SERIES, jax_series_tested

__all__ = [
    "jax_version",
    "transcribe",
    "any_array",
    "any_pytree",
    "assume",
    "assert_",
    "fold_extrema",
    "nonvacuity",
    "trace",
]

# jax types that are pure zero-payload sentinels: recording the type name is
# lossless. Matched by name because isinstance would require importing the
# private jax modules they live in. Sentinels with actual content (e.g. a
# real NamedSharding) must NOT be added here — they raise until a rule exists.
_SENTINEL_PARAM_TYPES = frozenset({"UnspecifiedValue"})

# Mesh types are public API; an *empty* mesh (no mesh in scope — the invariant
# state for single-device programs) is zero-payload and safe to record as a
# sentinel. Non-empty meshes mean a sharded program, which stelling does not
# support yet, so they raise.
_MESH_TYPES = tuple(
    t
    for t in (
        getattr(jax.sharding, "Mesh", None),
        getattr(jax.sharding, "AbstractMesh", None),
    )
    if t is not None
)

# (primitive, param) slots that hold transform-time thunks: callables jax
# keeps for its own later transforms, unreachable by construction for any
# analysis stelling performs. Recorded as ir.OpaqueParam (explicitly lossy);
# callables in unlisted slots still raise. Evidence and rationale live in
# design/transparent-primitives.md. Every entry is verified against jax
# 0.10.2 — no speculative entries for other series: on an untested series,
# raising is the correct outcome, surfacing exactly what TESTED_JAX_SERIES
# exists to say.
_OPAQUE_PARAMS = frozenset(
    {
        ("custom_jvp_call", "jvp_jaxpr_fun"),
        ("custom_vjp_call", "fwd_jaxpr_thunk"),
        ("custom_vjp_call", "bwd"),
        ("custom_vjp_call", "out_trees"),
        ("remat2", "policy"),
        # PRNGImpl carries jax's own key-handling functions; the impl's
        # identity survives in its name/tag/key_shape fields, which
        # transcribe normally. Found by census contact (blackjax, 0.10.2).
        ("random_wrap", "impl.seed"),
        ("random_wrap", "impl.split"),
        ("random_wrap", "impl.random_bits"),
        ("random_wrap", "impl.fold_in"),
        # a host callback is an arbitrary Python function: ⊤ at the param
        # level by definition, for every analysis stelling will ever run.
        # Found by census contact (diffrax/equinox error paths, 0.10.2).
        ("pure_callback", "callback"),
        # lineax's `linear_solve` is an equinox-defined primitive (census
        # contact, lineax 0.1.1); `flatten` is equinox plumbing (same census).
        ("linear_solve", "flatten"),
    }
)


@functools.lru_cache(maxsize=None)  # warn once per distinct version
def _warn_untested_jax(version: str) -> None:
    warnings.warn(
        f"stelling is tested against jax {', '.join(TESTED_JAX_SERIES)}.x but is "
        f"running under jax {version}. Transcription fails loudly on anything it "
        f"does not recognize, and verdicts stamp the exact jax version.",
        RuntimeWarning,
        stacklevel=3,
    )


def jax_version() -> str:
    return jax.__version__


def x64_enabled() -> bool:
    """The live ``jax_enable_x64`` state — for stamps that must record the
    configuration the trace actually ran under, not an assumption."""
    return bool(jax.config.jax_enable_x64)


# -- harness primitives -------------------------------------------------------
#
# The harness API (any_array / assume / assert_) binds real jax primitives so
# that the declarations land in the traced jaxpr itself: the query's content
# hash then covers the bounds and the obligations, not just the computation —
# a stamp that hashed the program but not what was assumed about its inputs
# would under-hash the claim.

_any_p = jex_core.Primitive("stelling_any")
_assume_p = jex_core.Primitive("stelling_assume")
_assert_p = jex_core.Primitive("stelling_assert")
_nonvacuity_p = jex_core.Primitive("stelling_nonvacuity")


@_any_p.def_abstract_eval
def _any_abstract(*, shape, dtype, lo, hi):
    return jax.core.ShapedArray(shape, np.dtype(dtype))


@_any_p.def_impl
def _any_impl(*, shape, dtype, lo, hi):
    raise RuntimeError(
        "stelling.harness.any_array is a tracing-time declaration; it has no "
        "concrete value. Call the harness under stelling.harness.trace (or "
        "jax.make_jaxpr), not eagerly."
    )


def _identity_abstract(aval):
    return aval


_assume_p.def_abstract_eval(_identity_abstract)
_assert_p.def_abstract_eval(_identity_abstract)
_nonvacuity_p.def_abstract_eval(_identity_abstract)
_assume_p.def_impl(lambda x: x)
_assert_p.def_impl(lambda x: x)
_nonvacuity_p.def_impl(lambda x: x)


def _int_neighbours(d_lo: int, d_hi: int, lo: float, hi: float) -> str:
    """The nearest representable integers OUTSIDE an empty interval, on each
    side, clamped into the dtype. Same pattern as the float case: print the
    numbers.

    THIRD VERSION, and the first two were both wrong in the same helper. The
    first ran `math.floor` on a raw bound and raised `OverflowError` on an
    infinite endpoint — the hazard a comment twelve lines from the call site
    names. The fix clamped both ends into `[d_lo, d_hi]` BEFORE choosing the
    direction word, which collapsed them onto the same value while keeping
    their original labels: `uint8 (-3, -1)` read *"0 below and 0 above"*, and
    a re-audit measured the words false in 77 of 113 cases while the numbers
    stayed right. So the sides are computed separately now, and a side that
    has no representable value simply does not appear.
    """
    below = above = None
    if math.isfinite(lo):
        k = math.floor(lo)
        below = d_hi if k > d_hi else (k if k >= d_lo else None)
    if math.isfinite(hi):
        k = math.ceil(hi)
        above = d_lo if k < d_lo else (k if k <= d_hi else None)
    parts = []
    if below is not None:
        parts.append(f"{below} below")
    if above is not None:
        parts.append(f"{above} above")
    if not parts:
        # Unreachable for a REFUSED box: both sides are empty only when the
        # interval is unbounded in both directions or strictly contains the
        # whole dtype range, and either of those holds a value. Stated rather
        # than defended with a branch that cannot fire (the previous version's
        # fallback here was measured dead in 877 of 877 calls).
        return ""
    note = ""
    lossy = [v for v in (below, above) if v is not None and float(v) != v]
    if lossy:
        note = (
            f" ({', '.join(str(v) for v in lossy)} is not exactly representable "
            f"as the binary64 the IR stores, so it works as the UPPER bound of "
            f"a box — where rounding only widens — but not as a lower bound or "
            f"a point, where it would narrow the declared set)"
        )
    return (
        f" The nearest representable integers are {' and '.join(parts)}; "
        f"declaring one of those, or the interval spanning them, describes a "
        f"set the program can actually inhabit.{note}"
    )


@functools.lru_cache(maxsize=None)
def _finite_values(dt: str):
    """Every finite value of a narrow float dtype, sorted — computed by
    enumerating its bit patterns, so it is exact rather than derived.

    Only for dtypes of at most two bytes, where the whole domain is 256 or
    65536 patterns. `numpy.nextafter` covers the wider ones and is used
    instead; enumeration is what covers `float8_e8m0fnu`, where nextafter is
    unusable because the type has no zero to step from (measured:
    ``nextafter(0, 1)`` returns NaN there).
    """
    d = np.dtype(dt)
    bits = 8 * d.itemsize
    raw = np.arange(2**bits, dtype=np.uint8 if bits == 8 else np.uint16)
    with np.errstate(invalid="ignore"):
        # NaN bit patterns warn on the cast; they are filtered out next line
        vals = raw.view(d).astype(np.float64)
    return np.unique(vals[np.isfinite(vals)])


def _smallest_at_or_above(dt: str, x: float):
    """The least value of dtype ``dt`` that is >= ``x``, or None if there is
    none. Exact: enumeration for narrow dtypes, nearest-then-step for wide
    ones (if the nearest landed below ``x``, the next one up is the answer,
    because nothing representable lies strictly between them)."""
    d = np.dtype(dt)
    if d.itemsize <= 2:
        vals = _finite_values(dt)
        i = int(np.searchsorted(vals, x, side="left"))
        return None if i >= vals.size else float(vals[i])
    # a bound outside the dtype's range overflows the cast; the value is
    # still what we want (an infinity, filtered below) and the warning is
    # noise, so it is silenced rather than left to litter every refusal
    with np.errstate(over="ignore", invalid="ignore"):
        nearest = float(np.array(x, d))
    if nearest >= x:
        # An out-of-range x rounds to an infinity, which is NOT a value of the
        # dtype: returning it made a refusal print "inf above" as a nearest
        # representable value, advice the caller cannot follow (`(inf, inf)`
        # is itself refused). The narrow-dtype path returned None for the same
        # question, so the two mechanisms disagreed. Re-audit.
        return nearest if math.isfinite(nearest) else None
    with np.errstate(over="ignore", invalid="ignore"):
        step = float(np.nextafter(np.array(x, d), np.array(np.inf, d)))
    return step if math.isfinite(step) else None


def _largest_at_or_below(dt: str, x: float):
    """Mirror of :func:`_smallest_at_or_above`, for the message."""
    d = np.dtype(dt)
    if d.itemsize <= 2:
        vals = _finite_values(dt)
        i = int(np.searchsorted(vals, x, side="right")) - 1
        return None if i < 0 else float(vals[i])
    # a bound outside the dtype's range overflows the cast; the value is
    # still what we want (an infinity, filtered below) and the warning is
    # noise, so it is silenced rather than left to litter every refusal
    with np.errstate(over="ignore", invalid="ignore"):
        nearest = float(np.array(x, d))
    if nearest <= x:
        return nearest if math.isfinite(nearest) else None
    with np.errstate(over="ignore", invalid="ignore"):
        step = float(np.nextafter(np.array(x, d), np.array(-np.inf, d)))
    return step if math.isfinite(step) else None


def _neighbours(dt: str, lo: float, hi: float) -> str:
    """The representable values bracketing an empty interval, named so the
    user learns what to write instead — the `div`-guard pattern: name it,
    explain it, PRINT THE NUMBERS."""
    below = _largest_at_or_below(dt, lo)
    above = _smallest_at_or_above(dt, hi)
    if below is None and above is None:
        return ""
    parts = []
    if below is not None:
        parts.append(f"{below!r} below")
    if above is not None:
        parts.append(f"{above!r} above")
    return (
        f" The nearest {dt} values are {' and '.join(parts)}; declaring one of "
        f"those, or the interval spanning them, describes a set the program "
        f"can actually inhabit"
    )


def _dtype_holds_a_value_in(dt: str, lo: float, hi: float) -> tuple[bool, str]:
    """Does the closed interval ``[lo, hi]`` contain ANY value of dtype ``dt``?

    The declaration layer's fourth empty-set refusal, and the narrowest test
    that closes it. What makes an interval empty here is not "a bound outside
    the dtype's range" — it is "no representable value inside the interval",
    and the difference is the whole design:

    * ``uint8 (-3, -1)`` holds nothing. **Empty. Rejected.**
    * ``uint8 (-3, 10)`` holds 0..10. **Non-empty, admitted** — the box is an
      OVER-approximation of what the dtype can produce, which is sound; every
      transfer's answer over it still contains the executed value.
    * ``float32 (0.0, 1e39)`` holds every float32 in [0, 3.4e38].
      **Admitted**, for the same reason, even though the upper bound is not
      representable.
    * ``float32 (1e39, 1e40)`` holds nothing — float32's finite max is
      3.4e38, and `inf` is not a member of that interval as a real either.
      **Empty. Rejected.** (An earlier version of this line credited the
      check with closing an f32 case in `rem`'s dividend guard. Measured:
      `_refuse_non_f64_float` declines every f32 `rem` before that guard is
      reached, so the case was already closed and this is a redundant second
      closure, not the closure. Blinded audit.)
    * ``int8 (0.2, 0.8)`` holds no integer. **Empty. Rejected.**

    THE RANGE LOOKUP, and why it is not a table. A first version dispatched on
    membership in ``propagate._INT_DTYPE_BOUNDS`` and then on ``numpy.finfo``,
    and a blinded audit measured that combination **silently exempting 16 of
    the 30 dtypes jax builds arrays in** — `int2`/`uint2` are absent from the
    registry (it holds widths 4/8/16/32/64), and `numpy.finfo` raises for
    every ml_dtypes extended type (`bfloat16`, the eight `float8_*`, the two
    `float6_*`, `float4_e2m1fn`) — fourteen falling through an
    ``except: return True`` fallback, plus the two complex dtypes admitted by
    policy. (**The first report of this said 9**, from an enumeration of 23
    dtypes rather than all 30: the same population error as the corpus
    histogram, one session later. The auditor's 16 was right and was
    overridden with a smaller measurement.) so the box this check exists to reject
    was **accepted verbatim one dtype-width away**: `uint8 (-3, -1)` refused,
    `uint2 (-3, -1)` admitted and discharged at 100% coverage — a VERIFIED over
    an empty set.

    So the lookup asks **jax**, which knows its own dtypes: ``jnp.iinfo`` then
    ``jnp.finfo``, which between them cover 29 of 30 (`bool` is the exception
    and the registry has it). No table, no new constant, no new dependency.

    ``finfo.min`` is read rather than ``-finfo.max``, because the range is not
    always symmetric: `float8_e8m0fnu` is exponent-only, its 255 finite values
    are all strictly positive, and negating the max would have declared
    ``(-100, -50)`` inhabited.

    THE INTEGER COMPARISON IS EXACT, in python ints, never through a float.
    ``float(2**63 - 1)`` rounds UP to ``2**63``, so comparing against a
    float-converted dtype bound made `int64` and `uint64` wrong at exactly the
    boundary the registry's own comment says it exists to be exact at — a box
    wholly above `int64` was admitted while the same shape on `uint8` was
    refused.

    WHERE THIS IS UNSURE IT ADMITS, deliberately — a declaration check that
    refuses a legitimate envelope is worse than the hole it closes:

    * **float bounds ARE tested for exact representability** — the trade was
      taken, and this paragraph said the opposite for one commit. The
      range-only rule admitted an interval lying wholly inside a
      representation gap: `float32 (1e-50, 1e-49)` sits below the smallest
      subnormal and reached a REFUTED at 100% coverage. No rule admits that
      and rejects ``float32 (0.1, 0.1)`` — both hold no value of the dtype —
      so the choice was which to refuse, and it resolves asymmetrically:
      admitting ``(0.1, 0.1)`` costs nothing because it IS an empty set, while
      admitting the gap interval mints a false counterexample. Refusing a
      point declaration at a decimal literal is not over-strict; it tells the
      caller their declaration does not mean what they think, which is this
      check's purpose. **float64 is unaffected**: every python float is a
      float64, so a point declaration there is always exact.
    * **complex dtypes are admitted unconditionally.** What a real-bounded
      box means for a complex array is undefined, and that is already an
      open item; this check does not settle it by refusing.
    """
    if dt.startswith("complex"):
        return True, ""
    # `_INT_DTYPE_BOUNDS` first (it has `bool`, which jnp.iinfo does not),
    # then jax. NOT "one registry, not two" any more — that comment predates
    # the jnp fallback and a re-audit caught it. jax IS a second source, and
    # the two disagree in coverage: this layer knows int2/uint2 and the
    # transfer layer declines them. Direction is safe (declare-admit ->
    # transfer-decline) and the values agree on all 11 shared dtypes.
    from stelling.propagate import _INT_DTYPE_BOUNDS

    ints = _INT_DTYPE_BOUNDS.get(dt)   # has `bool`, which jnp.iinfo does not
    if ints is None:
        try:
            info = jnp.iinfo(np.dtype(dt))
            ints = (int(info.min), int(info.max))
        except (TypeError, ValueError):
            ints = None
    if ints is not None:
        d_lo, d_hi = ints
        # clamp BEFORE ceil/floor: math.ceil(-inf) raises, and an infinite
        # endpoint here means "unbounded", which the dtype bound absorbs
        if hi < d_lo or lo > d_hi:
            first, last = 1, 0
        else:
            first = d_lo if lo <= d_lo else math.ceil(lo)
            last = d_hi if hi >= d_hi else math.floor(hi)
        if first <= last:
            return True, ""
        return False, (
            f"{dt} represents the integers [{d_lo}, {d_hi}] and the interval "
            f"contains none of them."
            + _int_neighbours(d_lo, d_hi, lo, hi)
        )
    try:
        info = jnp.finfo(np.dtype(dt))
        f_lo, f_hi = float(info.min), float(info.max)
    except (TypeError, ValueError):
        # No dtype reaches here on jax 0.11.0. Left as an admit rather than a
        # guess, and NAMED: the previous version's silent version of this line
        # is what exempted sixteen dtypes.
        return True, ""
    if hi < f_lo or lo > f_hi:
        return False, (
            f"{dt}'s finite values span [{f_lo}, {f_hi}] and the interval "
            f"lies entirely outside them." + _neighbours(dt, lo, hi)
        )
    # THE EXACT TEST. Range alone admits an interval lying wholly inside a
    # REPRESENTATION GAP -- `float32 (1e-50, 1e-49)` sits below the smallest
    # subnormal, holds no float32, and was measured reaching a REFUTED at 100%
    # coverage. So: clamp into the dtype's range, then ask for the least
    # representable value at or above the low end, and compare it to the high
    # end. Exact for every float dtype jax has.
    first = _smallest_at_or_above(dt, max(lo, f_lo))
    if first is not None and first <= min(hi, f_hi):
        return True, ""
    return False, (
        f"no {dt} value lies in the interval — it falls inside a gap between "
        f"representable values." + _neighbours(dt, lo, hi)
    )


def any_array(shape, dtype, bounds):
    """Declare a harness input: an arbitrary array of ``shape``/``dtype``
    with every element in ``bounds = (lo, hi)``. Traces to a
    ``stelling_any`` equation carrying the bounds as params."""
    dims = tuple(int(d) for d in shape)
    if any(d < 0 for d in dims):
        # measured on jax 0.11.0: every concrete context rejects a
        # negative extent (jnp.zeros((-2,-2)) raises "shape must have
        # every element be nonnegative"), so no array of this shape
        # exists and the declared set is EMPTY — declaring it would let
        # universal claims verify (or refute!) vacuously. Same
        # refusal-at-declaration posture as the empty/NaN bounds and the
        # (inf, inf) empty real set below (audit-gate finding 3; fix
        # re-attack R1). Zero-size shapes stay legal: jax constructs
        # them, and their vacuous discharge is the measured jnp.all
        # convention.
        raise ValueError(
            f"any_array shape {dims} has a negative extent; no jax "
            f"program can construct such an array (measured: jax rejects "
            f"negative dims in every concrete context), so the declared "
            f"set is empty; refusing at declaration time"
        )
    lo, hi = float(bounds[0]), float(bounds[1])
    if not lo <= hi:  # also rejects NaN: an empty declared set verifies
        raise ValueError(  # everything vacuously, and is never what anyone meant
            f"any_array bounds ({bounds[0]!r}, {bounds[1]!r}) declare an empty "
            f"set; refusing at declaration time"
        )
    if lo == hi and (lo == float("inf") or lo == float("-inf")):
        # under the stamped ℝ semantics an infinite endpoint means
        # "unbounded", so (inf, inf) contains no real — an empty set that
        # slipped the check above and yielded vacuous definite verdicts
        # (audit-gate finding 3)
        raise ValueError(
            f"any_array bounds ({bounds[0]!r}, {bounds[1]!r}) declare an "
            f"empty real set (an infinite point has no members under ℝ "
            f"semantics); refusing at declaration time"
        )
    dt = str(np.dtype(dtype))
    for name, raw in (("lo", bounds[0]), ("hi", bounds[1])):
        # The IR stores bounds as binary64, and an integer bound past 2**53
        # does not survive that. But ONLY A NARROWING LOSS IS A PROBLEM.
        # Rounding `lo` DOWN or `hi` UP widens the stored box, which is an
        # over-approximation and sound — every transfer's answer over it still
        # contains the executed value. Rounding the other way shrinks the set
        # the tool reasons about below the one the caller declared.
        #
        # The first version refused on ANY inexact integer bound, which made
        # `int64 (-2**63, 2**63-1)` — the natural way to say "any int64" —
        # a refusal, though `float(2**63-1)` rounds UP and only widens it.
        # A re-audit caught that as a false rejection of a legitimate
        # envelope, which is the failure this layer must not have.
        # `np.integer` is NOT a python int, and `_is_bounds_pair` explicitly
        # accepts it — so a numpy-typed bound skipped this guard entirely and
        # got the EMPTY-set message, which names the wrong cause on exactly
        # the case this guard exists for. Re-audit.
        # `int(raw)` first, and the comparison in PYTHON ints. numpy compares a
        # float64 against an int64 by converting the int to float64, which
        # discards exactly the inexactness under test: `float(np.int64(2**63-1))
        # != np.int64(2**63-1)` is False while the python-int form is True.
        exact = (
            isinstance(raw, (int, np.integer))
            and not isinstance(raw, (bool, np.bool_))
        )
        if not exact:
            continue
        v = int(raw)
        stored = float(v)
        narrows = (stored > v) if name == "lo" else (stored < v)
        if stored != v and narrows:
            raise ValueError(
                f"any_array bound {name}={raw!r} is not representable as the "
                f"binary64 the IR stores; it would be recorded as {stored!r}, "
                f"which NARROWS the declared set — the tool would reason about "
                f"fewer values than you declared. (Rounding the other way is "
                f"admitted: it widens the box, and an over-approximation still "
                f"contains every executed value.) Integer bounds that must be "
                f"exact as doubles satisfy |bound| <= 2**53"
            )
    if 0 in dims:
        # A ZERO-SIZE array satisfies ANY element-wise bounds vacuously, and
        # jax constructs one, so no bounds/dtype pair is empty for it — the
        # negative-extent comment above already says zero-size stays legal.
        # The dtype check is per-element and never sees the shape, so without
        # this it refused a declaration that IS inhabited (blinded audit; the
        # one false-rejection finding, and the failure the check must not
        # have).
        return _any_p.bind(shape=dims, dtype=dt, lo=lo, hi=hi)
    holds, why = _dtype_holds_a_value_in(dt, lo, hi)
    if not holds:
        raise ValueError(
            f"any_array bounds ({bounds[0]!r}, {bounds[1]!r}) declare a set "
            f"EMPTY under dtype {dt!r} — {why}. No execution of the program "
            f"can produce a value in this box, so every claim over it is "
            f"vacuous — measured, such a declaration reaches a DEFINITE "
            f"verdict at 100% coverage, in both directions, with a witness "
            f"the caller cannot construct. Refusing at declaration time, as "
            f"for a negative extent, lo > hi, and the infinite point"
        )
    return _any_p.bind(
        shape=dims,
        dtype=dt,
        lo=lo,
        hi=hi,
    )


def _is_array_prototype(x) -> bool:
    """An array-shaped leaf: anything carrying shape+dtype (np.ndarray,
    np.generic, jax.Array, tracers). Only shape and dtype are ever read —
    prototype *values* never enter the trace."""
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _is_bounds_pair(b) -> bool:
    return (
        isinstance(b, (tuple, list))
        and len(b) == 2
        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in b)
    )


def any_pytree(tree, bounds):
    """Declare a pytree of harness inputs: tracing-time sugar over
    :func:`any_array`, one call per array leaf, tree rebuilt around the
    traced values (``jax.tree_util``). No new primitive, no new tracing
    machinery — a harness declared through this helper traces to the very
    same equations as the equivalent hand declaration, so the query
    content hash is identical. **Including the refusals**: this delegates
    every bound decision to :func:`any_array` rather than pre-processing
    the bounds, so a declaration the hand form refuses is refused here too.
    An earlier version converted bounds to float first and therefore
    ADMITTED, with a silently rounded bound, a declaration the hand form
    rejects — the claim in this sentence was false for that class, which is
    what made it worth stating precisely rather than loosely.

    ``tree`` is a *prototype* pytree:

    * **array leaves** (anything with ``shape`` and ``dtype``: numpy or
      jax arrays, scalars of either) contribute shape+dtype to one
      ``any_array`` declaration each; their values never enter the trace;
    * **non-array static leaves** (python numbers, strings, ...) and the
      tree structure itself (including ``None``) pass through verbatim;
    * **aliasing is object identity**: the same array object at two tree
      positions is declared ONCE and the traced value is shared between
      the positions; two distinct objects are never merged into one
      declaration, however equal their content;
    * a leaf whose dtype is a **jax PRNG key type is refused** — a key has
      no bounds representation here. Declare the raw key bits and build
      the key explicitly, e.g. ``jax.random.wrap_key_data(any_array((2,),
      "uint32", (0, 2**32 - 1)))``.

    ``bounds`` is either a single ``(lo, hi)`` pair of numbers, broadcast
    to every array leaf, or a pytree matching ``tree`` with a ``(lo, hi)``
    pair at each array-leaf position and ``None`` at each static-leaf
    position.

    Errors here are harness-AUTHORING errors (the caller's setup), so they
    raise; analysis-time guards, by contrast, never do.
    """
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(tree)
    broadcast = _is_bounds_pair(bounds)
    if broadcast:
        per_leaf = [bounds] * len(leaves_with_paths)
    else:
        try:
            per_leaf = treedef.flatten_up_to(bounds)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"any_pytree: bounds must be a single (lo, hi) pair or a "
                f"pytree matching the declared tree's structure; got neither "
                f"({e})"
            ) from e

    def canon(pair, where):
        """Validate, and pass the bounds through UNCONVERTED.

        This used to `return float(pair[0]), float(pair[1])`, and that one
        conversion was a soundness hole: `any_array`'s storability guard keys
        on the operand's own type, so a pre-converted bound arrived already
        rounded and the guard never saw the value it exists to judge.
        Measured — the hand form REFUSED `int64 (0, 2**53 + 1)` while the sugar
        ADMITTED it with `hi` stored as `9007199254740992.0`, so the declared
        integer was not in the box the tool then reasoned about.

        The decision about a bound the IR cannot hold exactly was already made
        one layer over, and it is reused rather than re-invented: NARROWING is
        refused (the tool would reason about fewer values than were declared),
        WIDENING is admitted (an over-approximation still contains every
        executed value). Deciding it again here is how the two layers came to
        disagree in the first place.

        The alias comparison below is unaffected: python compares `0 == 0.0`
        as equal, so `(0, 1)` and `(0.0, 1.0)` still read as the same bounds,
        while `2**53 + 1` and `2.0**53` correctly read as different.
        """
        if not _is_bounds_pair(pair):
            raise ValueError(
                f"any_pytree: bounds for array leaf at {where} must be a "
                f"(lo, hi) pair of numbers, got {pair!r}"
            )
        return pair[0], pair[1]

    declared: dict[int, object] = {}  # id(prototype) -> traced value
    declared_bounds: dict[int, tuple[object, object]] = {}
    out_leaves = []
    for (path, leaf), b in zip(leaves_with_paths, per_leaf):
        where = jax.tree_util.keystr(path) or "<root>"
        if not _is_array_prototype(leaf):
            if not broadcast and b is not None:
                raise ValueError(
                    f"any_pytree: leaf at {where} is a non-array static value "
                    f"({type(leaf).__name__}); it passes through verbatim and "
                    f"takes no bounds — put None at its position"
                )
            out_leaves.append(leaf)
            continue
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            raise TypeError(
                f"any_pytree: leaf at {where} has PRNG key dtype "
                f"{leaf.dtype}: a key has no bounds representation, and "
                f"inventing one would mis-declare the input set. Declare the "
                f"raw key bits instead and construct the key explicitly — "
                f"e.g. jax.random.wrap_key_data(any_array((2,), 'uint32', "
                f"(0, 2**32 - 1)))."
            )
        if id(leaf) in declared:  # same object, same declaration, same value
            if canon(b, where) != declared_bounds[id(leaf)]:
                raise ValueError(
                    f"any_pytree: array leaf at {where} aliases an earlier "
                    f"leaf (same object) but with different bounds "
                    f"{canon(b, where)} vs {declared_bounds[id(leaf)]}; one "
                    f"object is one declaration — give it one bounds pair"
                )
            out_leaves.append(declared[id(leaf)])
            continue
        pair = canon(b, where)
        val = any_array(tuple(leaf.shape), leaf.dtype, pair)
        declared[id(leaf)] = val
        declared_bounds[id(leaf)] = pair
        out_leaves.append(val)
    return treedef.unflatten(out_leaves)


def fold_extrema(value):
    """``(max, min)`` of a traced array's elements, as ``(1,)``-shaped
    values, via a pairwise fold of binary ``max``/``min`` over
    single-element ``slice``\\ s of the flattened array.

    This is :mod:`stelling.contracts`' extremum encoding: ``reduce_max``
    and ``reduce_min`` have no transfer or emission row, so the fold
    stays on already-supported primitives (``reshape`` — structural,
    often elided for 1-D — ``slice``, binary ``max``/``min``). In ℝ the
    fold IS the extremum, exactly, in both directions: not an
    approximation. It lives here because this is the one module allowed
    to touch jax; the encoding choice and its budget arithmetic are
    documented at the posing site
    (:func:`stelling.contracts.coefficient_contrast`).
    """
    # a transform may return a plain Python/numpy constant (the
    # field_positive convention this mirrors tolerates constant returns);
    # coerce before reading .size so a constant field poses
    # trivially-discharged obligations instead of crashing raw (audit F5)
    value = jax.numpy.asarray(value)
    n = int(value.size)
    if n == 0:
        raise ValueError(
            "fold_extrema over a zero-element value: max/min of an empty "
            "field does not exist (measured jax refuses size-0 reductions)"
        )
    flat = jax.numpy.reshape(value, (n,))
    parts = [jax.lax.slice(flat, (i,), (i + 1,)) for i in range(n)]
    mx = mn = parts[0]
    for p in parts[1:]:
        mx = jax.numpy.maximum(mx, p)
        mn = jax.numpy.minimum(mn, p)
    return mx, mn


def assume(pred):
    """Record an assumption in the trace. The MVP interval propagation does
    **not** refine by assumptions (they are inert, conservative); bounds
    belong in :func:`any_array`."""
    return _assume_p.bind(pred)


def assert_(pred):
    """State an obligation: ``pred`` must hold for every input admitted by
    the ``any_array`` declarations. Returns ``pred`` — harnesses should
    return their asserts so no obligation can be dropped as dead code."""
    return _assert_p.bind(pred)


def nonvacuity(pred):
    """Declare a membership condition tying the declared set to the
    incident's own data: ``pred`` states (a conjunct of) "the known point
    lies in the declared box", computed in traced code through the same
    transforms the box is stated in. The stamp reports whether all
    declared conditions hold — a VERIFIED with unchecked nonvacuity is a
    different claim from one with it checked. Returns ``pred``; return it
    from the harness alongside the asserts."""
    return _nonvacuity_p.bind(pred)


def trace(harness) -> ir.ClosedJaxpr:
    """Trace a nullary harness (inputs declared via :func:`any_array`) and
    transcribe it. The caller owns jax config (e.g. ``jax_enable_x64``)."""
    return transcribe(jax.make_jaxpr(harness)())


def transcribe(closed_jaxpr) -> ir.ClosedJaxpr:
    """Transcribe a ``jax.extend.core.ClosedJaxpr`` (e.g. the result of
    ``jax.make_jaxpr``) into the jax-free :class:`stelling.ir.ClosedJaxpr`.

    Purely mechanical; raises :class:`stelling.ir.UnsupportedParamError` on
    any eqn param whose type has no transcription rule, naming the primitive
    and the param. Unknown primitives never raise.
    """
    if not jax_series_tested(jax.__version__):
        _warn_untested_jax(jax.__version__)
    return _Transcriber().closed_jaxpr(closed_jaxpr)


class _Transcriber:
    def __init__(self) -> None:
        self._var_ids: dict[int, int] = {}  # id(jax Var) -> IR id, encounter order

    # -- atoms and avals ----------------------------------------------------

    def var(self, v) -> ir.Var:
        vid = self._var_ids.get(id(v))
        if vid is None:
            vid = len(self._var_ids)
            self._var_ids[id(v)] = vid
        return ir.Var(id=vid, aval=self.aval(v.aval))

    def atom(self, a) -> ir.Atom:
        if isinstance(a, jex_core.Literal):
            return ir.Literal(val=self.value(a.val), aval=self.aval(a.aval))
        return self.var(a)

    def aval(self, av) -> ir.Aval:
        dims = []
        for d in getattr(av, "shape", ()):
            if isinstance(d, (int, np.integer)):
                dims.append(int(d))
            else:
                raise ir.TranscriptionError(
                    f"non-static dimension {d!r} in aval {av}; stelling handles "
                    f"fixed shapes only (design commitment 3)"
                )
        dtype = getattr(av, "dtype", None)
        return ir.Aval(
            kind=type(av).__name__,
            shape=tuple(dims),
            dtype=str(dtype) if dtype is not None else None,
            weak_type=bool(getattr(av, "weak_type", False)),
        )

    def value(self, v):
        """A const or Literal value -> IR scalar or inert Array."""
        if v is None or isinstance(v, (bool, int, float, complex, str)):
            return v
        if isinstance(v, np.generic):
            return v.item()
        try:
            arr = np.asarray(v)  # materializes jax arrays on host
        except Exception as e:
            # A const that is not a materialisable array. The live case is a
            # nested ``ClosedJaxpr`` hidden in a primitive's params whose
            # consts are still TRACERS (third-party solver primitives that
            # stash a sub-jaxpr in a static param do this), and
            # ``np.asarray`` then raises from inside stelling.
            #
            # That leak is the defect, not the const: a caller who points
            # stelling at unsupported code and gets a jax tracer exception
            # concludes the TOOL is broken, where a named refusal would tell
            # them the tool does not cover this yet. Same information,
            # opposite conclusion. So the refusal is raised in stelling's own
            # taxonomy, with the reason attached.
            raise ir.UnsupportedParamError(
                f"const of type {type(v).__name__!r} could not be "
                f"materialised as an array ({type(e).__name__}: {e}); a "
                f"nested jaxpr whose consts are live tracers cannot be "
                f"transcribed"
            ) from e
        return ir.Array(
            dtype=arr.dtype.str,
            shape=tuple(int(d) for d in arr.shape),
            data=arr.tobytes(),
        )

    # -- params ---------------------------------------------------------------

    def param(self, prim: str, name: str, v):
        if v is None or isinstance(v, (bool, str)):
            return v
        if isinstance(v, enum.Enum):  # before int: IntEnum is an int subclass
            return ir.EnumParam(cls=type(v).__name__, member=v.name)
        if isinstance(v, (int, float, complex)):
            return v
        if isinstance(v, np.dtype):
            return str(v)
        if isinstance(v, type) and issubclass(v, np.generic):  # e.g. np.float32 the class
            return str(np.dtype(v))
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, tuple) and hasattr(v, "_fields"):  # e.g. GatherDimensionNumbers
            return ir.NamedTupleParam(
                cls=type(v).__name__,
                fields=tuple(
                    (f, self.param(prim, f"{name}.{f}", getattr(v, f))) for f in v._fields
                ),
            )
        if isinstance(v, (tuple, list)):
            return tuple(self.param(prim, f"{name}[{i}]", x) for i, x in enumerate(v))
        if isinstance(v, jex_core.ClosedJaxpr):
            return self.closed_jaxpr(v)
        if isinstance(v, jex_core.Jaxpr):
            return self.jaxpr(v)
        if isinstance(v, np.ndarray):
            return self.value(v)
        if (
            type(v).__name__ in _SENTINEL_PARAM_TYPES
            and type(v).__module__.startswith("jax.")
        ):
            return ir.SentinelParam(cls=type(v).__name__)
        if _MESH_TYPES and isinstance(v, _MESH_TYPES):
            if getattr(v, "empty", False):
                return ir.SentinelParam(cls=f"{type(v).__name__}.empty")
            raise ir.UnsupportedParamError(
                f"primitive {prim!r}: param {name!r} is a non-empty mesh "
                f"({v!r}); sharded programs are not supported yet."
            )
        if isinstance(v, jax.tree_util.PyTreeDef):
            return ir.TreeDefParam(text=str(v))
        if type(v).__name__ == "FTTuple" and "flattree" in type(v).__module__:
            # jax 0.11's flat-tree structure metadata (scan ft_in/ft_out):
            # same epistemic category as PyTreeDef — structural text only.
            # Matched by name (private module). Verified against jax 0.11.0.
            return ir.TreeDefParam(text=str(v))
        if type(v).__name__ == "ShapedArray" and type(v).__module__.startswith("jax"):
            # avals appear as params on callback primitives (result_avals);
            # mirror them exactly like any other aval (census contact, 0.10.2)
            return self.aval(v)
        if isinstance(v, jax.sharding.NamedSharding):
            # trivial = empty mesh, nothing partitioned: zero semantic payload
            # (found by census contact: numpyro on 0.10.2 attaches one to
            # convert_element_type). Anything non-trivial is a sharded program.
            if getattr(v.mesh, "empty", False) and all(p is None for p in tuple(v.spec)):
                return ir.SentinelParam(cls="NamedSharding.trivial")
            raise ir.UnsupportedParamError(
                f"primitive {prim!r}: param {name!r} is a non-trivial sharding "
                f"({v!r}); sharded programs are not supported yet."
            )
        if type(v) is object:
            # a bare object() is an identity-only placeholder: it cannot
            # carry payload by construction (census contact: lineax's
            # `linear_solve.static`, 0.10.2)
            return ir.SentinelParam(cls="object")
        if (prim, name) in _OPAQUE_PARAMS:
            return ir.OpaqueParam(cls=type(v).__qualname__)
        raise ir.UnsupportedParamError(
            f"primitive {prim!r}: param {name!r} has unsupported type "
            f"{type(v).__module__}.{type(v).__qualname__}; refusing to guess — "
            f"dropping a param changes semantics. Add a transcription rule for it."
        )

    # -- structure ------------------------------------------------------------

    def eqn(self, e) -> ir.JaxprEqn:
        prim = e.primitive.name
        return ir.JaxprEqn(
            primitive=prim,
            invars=tuple(self.atom(a) for a in e.invars),
            outvars=tuple(self.var(v) for v in e.outvars),
            params=tuple((k, self.param(prim, k, v)) for k, v in e.params.items()),
            effects=tuple(str(eff) for eff in getattr(e, "effects", ()) or ()),
            source_info=self.source(getattr(e, "source_info", None)),
        )

    def source(self, si) -> tuple[str, ...]:
        if si is None:
            return ()
        try:
            tb = si.traceback
            if tb is None:
                return ()
            return tuple(
                f"{f.file_name}:{f.line_num} ({f.function_name})" for f in tb.frames
            )
        except Exception:
            return (str(si),)

    def debug_info(self, di) -> ir.DebugInfo | None:
        if di is None:
            return None
        return ir.DebugInfo(
            func=str(getattr(di, "func_src_info", "") or ""),
            arg_names=tuple(str(a) for a in (getattr(di, "arg_names", ()) or ())),
            result_paths=tuple(str(r) for r in (getattr(di, "result_paths", ()) or ())),
        )

    def jaxpr(self, j) -> ir.Jaxpr:
        return ir.Jaxpr(
            constvars=tuple(self.var(v) for v in j.constvars),
            invars=tuple(self.var(v) for v in j.invars),
            outvars=tuple(self.atom(a) for a in j.outvars),
            eqns=tuple(self.eqn(e) for e in j.eqns),
            effects=tuple(str(eff) for eff in getattr(j, "effects", ()) or ()),
            debug_info=self.debug_info(getattr(j, "debug_info", None)),
        )

    def closed_jaxpr(self, cj) -> ir.ClosedJaxpr:
        return ir.ClosedJaxpr(
            jaxpr=self.jaxpr(cj.jaxpr),
            consts=tuple(self.value(c) for c in cj.consts),
        )
