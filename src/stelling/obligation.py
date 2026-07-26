# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Obligation slices for solver escalation: extraction, routing, replay.

For each obligation interval propagation left ``unknown``, this module
extracts the *expression slice* — the ir equations from the
``stelling_any`` declarations and constants to the ``stelling_assert``
operand (descending transparent call wrappers such as ``jit`` for the
computation; obligations themselves remain top-level-only) — validates
it against the emission set, and classifies its fragment (``QF_LRA`` for
purely linear content, ``QF_NRA`` for polynomial: a product of two
non-constant subterms, an integer power >= 2, or division by a
non-constant). It also evaluates a slice at a concrete rational point
(:func:`evaluate_predicate`) in exact :class:`fractions.Fraction`
arithmetic — the solver-free replay that witness-backed refutations
require.

Nothing here guesses. The emission is **bounded static-shape**: scalars
and small fixed-shape arrays, one SMT term per element, gated by a single
per-obligation element budget (:data:`ELEMENT_BUDGET`) — deliberately NOT
general array reasoning (no quantified array theory, no dynamic shapes).
The emission set: elementwise ``add``, ``sub``, ``mul``, ``neg``, guarded
``div``, ``integer_pow``, ``max``, ``min``, the comparisons, boolean
``and``/``or``/``not``/``xor``, boolean-selector ``select_n``,
value-preserving ``convert_element_type`` (all with jax/numpy rank
broadcasting, sharing preserved: a broadcast scalar is ONE term
everywhere); the structural index-routing ops ``broadcast_in_dim``,
``reshape``, ``squeeze``, ``transpose``, ``slice``, ``concatenate``,
``stack`` (no new terms — an output element IS its source element's
term); the exact n-ary ``reduce_sum``; and static-index ``scatter-add``
(the accumulate scatter: per output element, the exact sum of the
operand element and its statically-known contributing update elements —
duplicate indices contribute once each). Everything else — over-budget slices,
transcendentals, unknown primitives, possibly-zero divisor elements,
non-float input declarations, obligations that cannot be mapped
one-to-one onto top-level asserts — **declines**, with the primitive and
form (and, for the budget, the count and the budget) quoted, and the
obligation stays UNKNOWN. Declines never raise; :exc:`ReplayError` is
raised only by the replay evaluator, whose caller treats it as an
emission-infidelity signal.

The index bookkeeping (element pairing, structural routing, reduction
grouping) is computed by ONE set of helpers, driven through the very
:mod:`stelling.interval` functions the propagation transfers use, and is
shared by slice validation, the SMT emission, and the replay — three
consumers, one routing, so they cannot disagree with each other.

Zero-dep: this module imports only the standard library and stelling's
own jax-free modules.
"""

from __future__ import annotations

import itertools
import math
import struct
from operator import index as _op_index
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from stelling import interval as iv
from stelling import ir
from stelling.coverage import DEFAULT_TRANSPARENT
from stelling.propagate import _EXACT_CONVERSIONS, _INT_DTYPE_BOUNDS, Propagation

__all__ = [
    "DeclinedObligation",
    "ELEMENT_BUDGET",
    "ObligationSlice",
    "ReplayError",
    "SliceInput",
    "evaluate_predicate",
    "slice_obligation",
    "slice_unknown_obligations",
    "violating_elements",
    "witness_is_valid",
]

QF_LRA = "QF_LRA"
QF_NRA = "QF_NRA"

# The reason text the division guard quotes, verbatim per the design.
DIV_GUARD_REASON = (
    "divisor may be zero over the declared box — SMT-LIB2 division is "
    "underspecified at 0"
)

# The single per-obligation element budget — THE ONE choke point replacing
# the seven scalar-only gates of the v1 emission. An obligation is
# emittable iff its slice touches only static shapes and BOTH of the
# gate's quantities stay at or under this number: the emitted
# element-terms (declared input elements + output elements of every
# term-producing equation; reduce_sum also counts its operand elements —
# its n-ary bodies inline one addend per operand element; structural
# index routing adds none) AND the root conjunct count (the assert
# operand's elements — structural routing mints no terms, but the negated
# conjunction it feeds repeats one term per element, so a structural-only
# inflation of the root is script cost the term count cannot see;
# first-contact audit F1). Anything above declines with both quantities
# and the budget quoted, computed from static shape metadata BEFORE any
# per-element work (an over-budget decline is O(#equations), never
# O(#elements) — audit F2; measured post-fix: 0.01 ms at 1.6M declared
# elements where the pre-fix order cost 2.91 s). The value is measured,
# not guessed. Emitted scripts of
# correlation-dependent per-face ring obligations (the acceptance FACE
# shape; roll = slice+concatenate) were timed on this machine's portfolio
# (z3 5.0.0 wheel; cvc5 1.3.4 wheel, coverings) at growing element
# counts, in both answer directions (2026-07-21; full table in the build
# report). The binding fragment is QF_NRA unsat under cvc5 coverings —
# the portfolio's QF_NRA *primary*:
#
#   terms:            256     512     1024     2048
#   cvc5 nl-cov unsat 0.32 s  2.2 s   18.5 s   timeout (> 120 s)
#   z3 unsat          ≤ 60 ms everywhere; sat ≤ 140 ms everywhere
#   QF_LRA (both)     ≤ 0.4 s out to 8192 terms
#
# 512 keeps the worst measured fragment near 2 s — inside the 30 s
# acceptance-config solver limit with more than 10x margin — while one
# doubling already spends 18.5 s and two doublings hang the primary. One
# budget for both fragments, deliberately: a per-fragment pair would be
# two gates that can disagree, and the single-gate discipline is the
# point; the cost is LRA headroom the measurements show it could afford.
# The budget is a solver-cost gate, so an over-budget obligation is an
# UNKNOWN with the number quoted, never a hang.
ELEMENT_BUDGET = 512

# integer_pow exponents are expanded to explicit products; beyond this the
# script stops being auditable by eye and the emission declines instead.
INTEGER_POW_EXPANSION_CAP = 64

_FLOAT_INPUT_DTYPES = frozenset({"float16", "float32", "float64"})

# scalar decoders for size-1 ir.Array literals/consts (numpy dtype .str)
_SCALAR_STRUCT_FMT = {
    "<f8": "d", "<f4": "f", "<f2": "e",
    "<i8": "q", "<i4": "i", "<i2": "h", "|i1": "b",
    "<u8": "Q", "<u4": "I", "<u2": "H", "|u1": "B",
    "|b1": "?",
}

_ARITH = frozenset({"add", "sub", "mul", "neg", "div", "integer_pow", "max", "min"})
_COMPARE = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})
_BOOL_OPS = frozenset({"and", "or", "not", "xor"})
# Structural ops are index bookkeeping, not new terms: an output element
# ALIASES its source element's existing term (a concatenate's output
# element IS its source element's term — sharing by construction, never a
# fresh variable). The routing is computed through the very
# stelling.interval functions the propagation transfers use
# (:func:`_route_structural`), so the emission cannot route an index
# differently from the propagation that judged the obligation.
_STRUCTURAL = frozenset(
    {"broadcast_in_dim", "reshape", "squeeze", "transpose", "slice",
     "concatenate", "stack"}
)
# Harness primitives whose value semantics are the identity on their input.
# stelling_assume's *constraint* is inert (dropped, disclosed by the
# propagation notes) and is deliberately NOT emitted — only its data flow
# passes through, exactly as in propagation.
_IDENTITY_HARNESS = frozenset({"stelling_assume", "stelling_nonvacuity"})

_SUPPORTED = (
    _ARITH | _COMPARE | _BOOL_OPS | _STRUCTURAL | _IDENTITY_HARNESS
    | {"reduce_sum", "select_n", "convert_element_type", "scatter-add",
       "dot_general", "scatter"}
)

# Emitted primitives that COMPUTE a new numeric value, and therefore can
# overflow an integer dtype. SMT-LIB2 Reals are unbounded, so emitting jax
# integer arithmetic as Real arithmetic models a program that wraps with
# one that does not — and the solver then proves the wrong claim, minting a
# false VERIFIED with the full weight of a proof behind it (audit
# UNSOUND 2, found on `integer_pow`; the sweep it prompted found the same
# gap on `add`/`sub`/`mul`/`neg`, all reachable through the bool->int
# conversion the whitelist admits).
#
# The rest of the emission set is safe by construction and stays unguarded:
# `max`/`min`/`select_n` SELECT an operand value rather than computing one,
# the comparisons are exact over Reals for integers, the shape ops and
# harness primitives are identities, `div` already carries its own stricter
# float-only guard, and `convert_element_type` is whitelist-guarded.
_INT_OVERFLOW_EMITTED = frozenset(
    {"add", "sub", "mul", "neg", "div", "integer_pow", "reduce_sum",
     # the accumulate scatter SUMS operand and update elements — the
     # add/reduce_sum class exactly, so integer dtypes decline here too
     "scatter-add",
     # the contraction sums PRODUCTS; same class, and the generic guard
     # above (integer OUTPUT dtype declines) covers it with no extra code
     "dot_general"}
)

# Mechanised for the same reason the transfer-side census is (audit
# UNSOUND 3): the first sweep of this defect class was a review, it cleared
# `div` on the emission's own float guard, and the sibling transfer site
# went unswept. A review finds siblings once; an assert finds them every
# time. Every emittable primitive is classified, and the union must be
# total over `_SUPPORTED`. The structural ops are pure data movement
# (copies of in-range values), so they are int-safe by construction.
_INT_SAFE_EMITTED = frozenset(
    _COMPARE | _BOOL_OPS | _STRUCTURAL | _IDENTITY_HARNESS
    | {"max", "min", "select_n", "convert_element_type", "scatter"}
)

if _INT_OVERFLOW_EMITTED | _INT_SAFE_EMITTED != _SUPPORTED:
    raise RuntimeError(
    "the integer-semantics census must stay total over the emission set: "
    f"unclassified {_SUPPORTED - _INT_OVERFLOW_EMITTED - _INT_SAFE_EMITTED}"
)
if _INT_OVERFLOW_EMITTED & _INT_SAFE_EMITTED:
    raise RuntimeError(
        "a primitive cannot be both integer-overflow-guarded and "
        f"int-safe: {sorted(_INT_OVERFLOW_EMITTED & _INT_SAFE_EMITTED)}"
    )

# -- the probe-or-exempt census over the emission classification --------------
#
# Third audit, F3, emission face (the transfer-side sibling sweep — audit
# UNSOUND 3's lesson applied forward): _INT_SAFE_EMITTED derives partly
# from set unions (_STRUCTURAL above all), so a primitive could join the
# int-safe class by the same edit that adds its support, with no
# independent check on the claim. The emission layer has NO behavioural
# boundary sweep, so here EVERY int-safe name carries a written soundness
# reason: a future addition to any constituent set is a conscious edit to
# this registry too, and the reason is the claim.
_INT_SAFE_EMITTED_REASONS: dict[str, str] = {
    "scatter": (
        "the static-index SET form emits NO term: element k's term IS the "
        "update's and every other element's IS the operand's, so the "
        "equation performs pure data movement. Nothing is computed, so "
        "nothing can overflow an integer dtype — unlike scatter-add, whose "
        "accumulate is Real addition and therefore carries the guard. The "
        "operand/updates/output dtypes are required to agree, so the "
        "aliasing never equates values of different sorts"
    ),
    "lt": "emits a comparison; exact over Reals for in-range integers, result sort Bool",
    "le": "emits a comparison; exact over Reals for in-range integers, result sort Bool",
    "gt": "emits a comparison; exact over Reals for in-range integers, result sort Bool",
    "ge": "emits a comparison; exact over Reals for in-range integers, result sort Bool",
    "eq": "emits a comparison; exact over Reals for in-range integers, result sort Bool",
    "ne": "emits a comparison; exact over Reals for in-range integers, result sort Bool",
    "and": "boolean connective on Bool terms (non-bool operands decline in _validate)",
    "or": "boolean connective on Bool terms (non-bool operands decline in _validate)",
    "not": "boolean connective on Bool terms (non-bool operands decline in _validate)",
    "xor": "boolean connective on Bool terms (non-bool operands decline in _validate)",
    "broadcast_in_dim": "emits NO terms: output elements alias source terms (index routing only)",
    "reshape": "emits NO terms: output elements alias source terms (index routing only)",
    "squeeze": "emits NO terms: output elements alias source terms (index routing only)",
    "transpose": "emits NO terms: output elements alias source terms (index routing only)",
    "slice": "emits NO terms: output elements alias source terms (index routing only)",
    "concatenate": "emits NO terms: output elements alias source terms (index routing only)",
    "stack": "emits NO terms: output elements alias source terms (index routing only)",
    "stelling_assume": "identity data flow; the constraint is deliberately never emitted",
    "stelling_nonvacuity": "identity data flow",
    "max": "emits an ite SELECTING an operand term; no arithmetic term is created",
    "min": "emits an ite SELECTING an operand term; no arithmetic term is created",
    "select_n": "emits an ite selecting a case term; no arithmetic term is created",
    "convert_element_type": (
        "whitelist-guarded in _validate: only value-preserving conversions "
        "emit (identity or the bool->{0,1} ite)"
    ),
}


def _assert_emission_classification_censused(
    supported=None, guarded=None, reasons=None
) -> None:
    """Probe-or-exempt at the emission layer (third audit, F3): every
    emittable primitive is either integer-GUARDED (in
    ``_INT_OVERFLOW_EMITTED`` — integer dtypes decline in ``_validate``)
    or carries a written int-safety reason in
    :data:`_INT_SAFE_EMITTED_REASONS`; stale reasons refuse (a claim
    about nothing). Callable with explicit arguments so the regression
    test can doctor a copy in-process; the import-time call runs on the
    live sets."""
    supported = _SUPPORTED if supported is None else set(supported)
    guarded = _INT_OVERFLOW_EMITTED if guarded is None else set(guarded)
    reasons = (
        dict(_INT_SAFE_EMITTED_REASONS) if reasons is None else dict(reasons)
    )
    for prim in sorted(supported):
        if prim in guarded:
            continue
        reason = reasons.get(prim)
        if not isinstance(reason, str) or not reason.strip():
            raise AssertionError(
                f"emission-classification census: primitive {prim!r} is "
                f"emittable, carries no integer overflow guard, and has "
                f"no written int-safety reason — the classification "
                f"itself must be censused: either guard it "
                f"(_INT_OVERFLOW_EMITTED) or write its soundness claim "
                f"into _INT_SAFE_EMITTED_REASONS"
            )
    stale = sorted(set(reasons) - (supported - guarded))
    if stale:
        raise AssertionError(
            f"emission-classification census: reason entr"
            f"{'ies' if len(stale) > 1 else 'y'} {stale} name no live "
            f"unguarded emittable primitive — a stale reason is a "
            f"soundness claim about nothing; delete or correct it"
        )


_assert_emission_classification_censused()


class ReplayError(RuntimeError):
    """The replay evaluator met something it cannot evaluate exactly.

    Raised only from :func:`evaluate_predicate`. The dispatch layer treats
    it as an emission-infidelity signal, never as a verdict.
    """


@dataclass(frozen=True)
class SliceInput:
    """One ELEMENT of a ``stelling_any`` declaration reachable from the
    obligation.

    A scalar (shape ``()``) declaration contributes one input named
    ``x{k}`` (``k`` = declaration order in the query — the pre-array
    scheme, byte-identical); an array declaration of static shape
    contributes one input per element, named ``x{k}_{i}`` with ``i`` the
    flat C-order element index, each carrying the declaration's bounds
    (scalar bounds broadcast to every element). These names are the SMT
    constants, the witness value names, and the model names the dispatch
    layer screens — one naming, three consumers."""

    name: str
    var_id: int
    lo: float
    hi: float
    shape: tuple[int, ...] = ()  # the DECLARATION's shape
    element: int = 0  # this element's flat C-order index in the declaration


@dataclass(frozen=True)
class ObligationSlice:
    """A validated, emittable expression slice for one unknown obligation."""

    index: int  # the obligation's index in the propagation report
    fragment: str  # QF_LRA | QF_NRA
    inputs: tuple[SliceInput, ...]  # declaration order, then element order
    # (var id, exact value) — an array-shaped constant stores the tuple of
    # its decoded elements in flat C-order; a scalar stores the bare value
    # (the pre-array shape of this field, unchanged).
    consts: tuple[tuple[int, object], ...]
    eqns: tuple[ir.JaxprEqn, ...]  # flattened topological order, sans stelling_any
    root: ir.Atom  # the assert operand (boolean, any static shape)
    source_info: tuple[str, ...]
    # total emitted element-terms (input elements + term-producing output
    # elements), the number the single budget gate judged; <= ELEMENT_BUDGET
    element_terms: int = 1


@dataclass(frozen=True)
class DeclinedObligation:
    """An obligation escalation declined, with the reason quoted."""

    index: int
    reason: str
    source_info: tuple[str, ...] = ()


class _Decline(Exception):
    """Internal control flow only; always converted to DeclinedObligation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _shape_problem(shape) -> str | None:
    """The uninhabited/malformed-shape problem of a static shape, or None.
    Two measured predicates (fix re-attacks R1/N1/N2): every extent must
    be INTEGRAL (from_dict does not coerce shape entries — a string
    extent made `d < 0` raise raw, and `1 * "x"` is silent garbage in a
    size product) and NONNEGATIVE (jax rejects negative extents in every
    concrete context: the type is uninhabited). Zero extents are legal."""
    for d in shape:
        try:
            k = _op_index(d)
        except TypeError:
            return f"a non-integer extent {d!r} (malformed IR)"
        if k < 0:
            return (
                "a negative extent (no jax program constructs such a value)"
            )
    return None


def _size(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _decode_scalar(val):
    """A size-1 literal/const value -> exact python bool | int | float, or
    decline. (The emission's single-element decoder; array constants go
    through :func:`_decode_elements`.)"""
    if isinstance(val, ir.Array):
        if _size(val.shape) != 1:
            raise _Decline(
                f"array-shaped constant of shape {val.shape} where a single "
                f"value is required"
            )
        fmt = _SCALAR_STRUCT_FMT.get(val.dtype)
        if fmt is None:
            raise _Decline(f"constant with undecodable dtype {val.dtype!r}")
        (v,) = struct.unpack(f"<1{fmt}", val.data)
        return v
    if isinstance(val, (bool, int, float)):
        return val
    raise _Decline(
        f"constant of type {type(val).__name__!r} has no exact Real emission"
    )


def _decode_elements(val) -> tuple:
    """A literal/const value -> tuple of exact python values, one per
    element in flat C-order, or decline. A scalar decodes to a 1-tuple; a
    static-shape :class:`stelling.ir.Array` decodes every element."""
    if isinstance(val, ir.Array):
        problem = _shape_problem(val.shape)
        if problem is not None:
            # fix-re-attack R1/N2: a negative or non-integer element
            # count would reach struct.unpack as a malformed format and
            # raise struct.error raw — decline quoted instead
            raise _Decline(
                f"array-shaped constant of shape {val.shape} has {problem}"
            )
        n = _size(val.shape)
        fmt = _SCALAR_STRUCT_FMT.get(val.dtype)
        if fmt is None:
            raise _Decline(f"constant with undecodable dtype {val.dtype!r}")
        expect = struct.calcsize(f"<{n}{fmt}")
        if len(val.data) != expect:
            # the length predicate (fix-re-attack N2, the sibling of the
            # negative route one assumption away: "what else does unpack
            # assume") — truncated, oversized, and empty payloads under a
            # positive shape raised raw struct.error before this
            raise _Decline(
                f"array-shaped constant of shape {val.shape} dtype "
                f"{val.dtype!r} carries {len(val.data)} byte(s), expected "
                f"{expect} — truncated or oversized payload (malformed IR)"
            )
        return struct.unpack(f"<{n}{fmt}", val.data)
    if isinstance(val, (bool, int, float)):
        return (val,)
    raise _Decline(
        f"constant of type {type(val).__name__!r} has no exact Real emission"
    )


def _numeric_fraction(v) -> Fraction:
    """Exact Fraction of a decoded numeric literal; declines non-finite."""
    if isinstance(v, bool):
        raise _Decline("boolean constant where a number is required")
    if isinstance(v, int):
        return Fraction(v)
    if not math.isfinite(v):
        raise _Decline(
            f"non-finite constant {v!r} has no exact Real emission"
        )
    return Fraction(v)  # exact dyadic rational of the f64 value


def _is_bool_dtype(aval: ir.Aval) -> bool:
    return aval.dtype == "bool"


def _zero_element_problem(
    atom: ir.Atom, env: Mapping[int, iv.IntervalArray]
) -> str | None:
    """None when EVERY element of the atom definitely excludes 0, else a
    string naming the first element that may be zero (the per-element div
    guard: the propagation env is already per-element, and any element
    straddling 0 declines the whole division, naming the element).

    For a size-1 atom the returned string is empty — the pre-array decline
    wording stands alone; for arrays the element is named."""

    def named(i: int, extra: str, size: int) -> str:
        return "" if size == 1 else f" (element {i}{extra})"

    if isinstance(atom, ir.Literal):
        try:
            vals = _decode_elements(atom.val)
        except _Decline:
            return " (undecodable constant divisor)"
        if not vals:
            return " (zero-size constant divisor)"
        for i, v in enumerate(vals):
            if isinstance(v, bool) or v != v or v == 0:  # NaN-safe; exact
                return named(i, f" = {v!r}", len(vals))
        return None
    size = _size(atom.aval.shape)
    got = env.get(atom.id)
    if got is None:
        return (
            " (no top-level propagated interval for this divisor — a value "
            "produced inside a transparent call carries none)"
        )
    if got.size != size or size == 0:
        return (
            f" (propagated interval has {got.size} element(s) for a divisor "
            f"of shape {tuple(atom.aval.shape)})"
        )
    for i, (lo, hi) in enumerate(zip(got.los, got.his)):
        if not (lo > 0.0 or hi < 0.0):
            return named(i, f" spans [{lo}, {hi}]", size)
    return None


# -- the one index bookkeeping (validation, emission, and replay share it) ----
#
# Element pairing, structural routing, and reduction grouping are computed
# by driving the SAME stelling.interval functions the propagation transfers
# invoke, on synthetic boxes whose element VALUES are their own flat
# indices: the interval function's data movement then reads back as an
# index map. One implementation — the sibling's — so the emission cannot
# route an element differently from the propagation that judged the
# obligation; the differential tests measure the routing against jax
# itself. Index values are exact in float far beyond ELEMENT_BUDGET
# (< 2**53). Interval-layer refusals (IntervalError) and malformed-IR
# index escapes (IndexError — parameters a legal jax trace cannot produce
# but ClosedJaxpr.from_dict can) both surface as declines with the reason
# quoted, never as crashes.


def _index_box(shape: tuple[int, ...], base: int) -> iv.IntervalArray:
    vals = tuple(float(base + i) for i in range(_size(shape)))
    return iv.IntervalArray(shape=shape, los=vals, his=vals)


def _shape_of(atom: ir.Atom) -> tuple[int, ...]:
    return tuple(atom.aval.shape)


def _pair_elementwise(eqn: ir.JaxprEqn) -> list[list[int]]:
    """Per-operand source flat-index lists for an elementwise equation, one
    entry per OUTPUT element (flat C-order) — jax/numpy rank broadcasting
    via :func:`stelling.interval._pair_elements`, the propagation's own
    pairing. A broadcast scalar operand yields index 0 for every output
    element: the SAME source term everywhere (sharing, never fresh
    variables). Declines on incompatible shapes or an output aval that
    contradicts the broadcast result."""
    shapes = [_shape_of(a) for a in eqn.invars]
    out_shape = _shape_of(eqn.outvars[0])
    if len(shapes) == 1:
        if shapes[0] != out_shape:
            raise _Decline(
                f"{eqn.primitive!r}: operand shape {shapes[0]} does not "
                f"match output shape {out_shape}"
            )
        return [list(range(_size(out_shape)))]
    if len(shapes) == 2:
        try:
            got_shape, xs, ys = iv._pair_elements(
                _index_box(shapes[0], 0), _index_box(shapes[1], 0)
            )
        except (iv.IntervalError, IndexError) as e:
            raise _Decline(f"{eqn.primitive!r}: {e}") from e
        if tuple(got_shape) != out_shape:
            raise _Decline(
                f"{eqn.primitive!r}: operand shapes {shapes[0]} and "
                f"{shapes[1]} broadcast to {tuple(got_shape)}, not the "
                f"output shape {out_shape}"
            )
        return [[int(lo) for lo, _ in xs], [int(lo) for lo, _ in ys]]
    raise _Decline(
        f"{eqn.primitive!r} with {len(shapes)} operands has no elementwise "
        f"pairing rule"
    )


def _pair_select_n(eqn: ir.JaxprEqn) -> list[list[int]]:
    """select_n(which, on_false, on_true) pairing: the cases must share the
    output shape, and ``which`` is either case-shaped (element i selects
    element i) or a SCALAR broadcast across every element (the measured
    jax form — ``jnp.where(pred_scalar, arr, arr)`` traces to exactly
    this inside its transparent jit): one selector term shared by every
    output element. Exactly the two forms
    :func:`stelling.interval.select_n` accepts; anything else declines."""
    w_shape, f_shape, t_shape = (_shape_of(a) for a in eqn.invars)
    out_shape = _shape_of(eqn.outvars[0])
    if f_shape != t_shape or f_shape != out_shape:
        raise _Decline(
            f"'select_n' case shapes {f_shape} and {t_shape} must both "
            f"equal the output shape {out_shape}"
        )
    n = _size(out_shape)
    if w_shape == out_shape:
        which = list(range(n))
    elif w_shape == () and out_shape != ():
        which = [0] * n  # one scalar selector term, shared by every element
    else:
        raise _Decline(
            f"'select_n' selector shape {w_shape} with case shape "
            f"{out_shape}: equal shapes or a scalar selector are the only "
            f"supported forms"
        )
    return [which, list(range(n)), list(range(n))]


def _route_structural(eqn: ir.JaxprEqn) -> list[tuple[int, int]]:
    """(operand index, source flat element) per OUTPUT element for a
    structural op — pure index bookkeeping, no new terms. Computed by
    running the corresponding :mod:`stelling.interval` function on
    index-valued boxes (each operand's elements encoded with a disjoint
    offset), so the routing IS the propagation transfer's data movement.
    Malformed params (wrong ranks, out-of-range selections, permutations
    that aren't, concatenate pieces whose off-axis extents disagree, an
    output aval contradicting the computed shape) decline quoted."""
    prim = eqn.primitive
    params = eqn.params_dict()
    in_shapes = [_shape_of(a) for a in eqn.invars]
    out_shape = _shape_of(eqn.outvars[0])
    offsets: list[int] = []
    boxes: list[iv.IntervalArray] = []
    off = 0
    for s in in_shapes:
        offsets.append(off)
        boxes.append(_index_box(s, off))
        off += _size(s)
    try:
        if prim == "broadcast_in_dim":
            out = iv.broadcast_in_dim(
                boxes[0],
                tuple(params["shape"]),
                tuple(params["broadcast_dimensions"]),
            )
        elif prim == "reshape":
            if params.get("dimensions") is not None:
                # a dimensions= reshape permutes before reshaping — the
                # propagation transfer declines it, and so does the
                # emission (same sibling, same boundary)
                raise _Decline(
                    f"'reshape' with dimensions={params['dimensions']!r} "
                    f"permutes before reshaping; no emission rule"
                )
            out = iv.reshape(boxes[0], tuple(params["new_sizes"]))
        elif prim == "squeeze":
            out = iv.squeeze(boxes[0], tuple(params.get("dimensions", ())))
        elif prim == "transpose":
            out = iv.transpose(
                boxes[0], tuple(params.get("permutation", ()) or ())
            )
        elif prim == "slice":
            strides = params.get("strides")
            out = iv.slice_(
                boxes[0],
                tuple(params["start_indices"]),
                tuple(params["limit_indices"]),
                tuple(strides) if strides else None,
            )
        elif prim == "concatenate":
            # rank/extent congruence and dimension range are validated by
            # the ORACLE itself (interval.concatenate raises IntervalError
            # on jax-illegal forms — one choke point shared with the
            # propagation transfer and the replay), surfacing here as the
            # quoted decline below
            out = iv.concatenate(boxes, int(params["dimension"]))
        elif prim == "stack":
            # same oracle discipline: interval.stack IS the routing (and
            # the propagation transfer), so shape/axis violations surface
            # as the quoted decline below
            out = iv.stack(boxes, int(params["axis"]))
        else:  # pragma: no cover - guarded by _STRUCTURAL membership
            raise _Decline(f"no structural routing rule for {prim!r}")
    except KeyError as e:
        raise _Decline(f"{prim!r} is missing required param {e}") from e
    except (iv.IntervalError, IndexError) as e:
        raise _Decline(f"{prim!r}: {e}") from e
    if out.shape != out_shape:
        raise _Decline(
            f"{prim!r}: computed output shape {out.shape} contradicts the "
            f"recorded aval shape {out_shape}"
        )
    routes: list[tuple[int, int]] = []
    for v in out.los:
        enc = int(v)
        op_idx = 0
        for j in range(len(offsets) - 1, -1, -1):
            if enc >= offsets[j]:
                op_idx = j
                break
        routes.append((op_idx, enc - offsets[op_idx]))
    return routes


def _group_reduce_sum(eqn: ir.JaxprEqn) -> list[list[int]]:
    """Per-OUTPUT-element groups of source flat indices for reduce_sum
    over static ``axes`` — the exact n-ary sum's addend sets. Shape and
    axis validation mirror :func:`stelling.interval.reduce_sum` (whose
    identity for an empty group is exactly 0, matching measured jax:
    ``jnp.sum`` of a size-0 array is 0.0)."""
    params = eqn.params_dict()
    in_shape = _shape_of(eqn.invars[0])
    out_shape = _shape_of(eqn.outvars[0])
    rank = len(in_shape)
    try:
        axes = tuple(int(a) for a in params["axes"])
    except KeyError as e:
        raise _Decline(f"'reduce_sum' is missing required param {e}") from e
    if len(set(axes)) != len(axes) or any(
        not 0 <= ax < rank for ax in axes
    ):
        raise _Decline(
            f"'reduce_sum' axes {axes} are not distinct in-range axes of "
            f"shape {in_shape}"
        )
    ax_set = set(axes)
    expect = tuple(d for k, d in enumerate(in_shape) if k not in ax_set)
    if expect != out_shape:
        raise _Decline(
            f"'reduce_sum' over axes {axes} of {in_shape} yields shape "
            f"{expect}, not the recorded aval shape {out_shape}"
        )
    groups: list[list[int]] = [[] for _ in range(_size(out_shape))]
    for coord in iv._coords(in_shape):
        i = iv._flat_index(coord, in_shape)
        j = iv._flat_index(
            tuple(c for k, c in enumerate(coord) if k not in ax_set),
            out_shape,
        )
        groups[j].append(i)
    return groups


def _dot_general_plan(eqns, consts, eqn: ir.JaxprEqn):
    """Per-OUTPUT-element linear combinations for a constant-operand
    ``dot_general`` — the one bookkeeping the SMT emission and the exact
    replay both drive, so neither can route a coefficient the other would
    not.

    Returns ``(sym_operand, groups)`` where ``sym_operand`` is 0 or 1 (which
    invar carries the SYMBOLIC terms) and ``groups[i]`` is a list of
    ``(Fraction coefficient, flat index into that operand)`` pairs. The
    output element is ``Σ coeff * sym[idx]`` — a constant-coefficient linear
    combination, which is what QF_LRA is for.

    **Why a constant operand is required, stated as the decline says it.**
    With both operands symbolic each term is a product of two variables and
    the obligation leaves linear arithmetic entirely. That is a scope
    decision, not a capability gap, and the decline names it.

    Admissibility is decided by
    :func:`stelling.propagate._dot_general_row_form`, the SAME oracle the
    interval transfer drives — built shared from the start rather than
    extracted later, which is how the scatter row acquired a gate the
    transfer had and the emission did not.

    TWO CHECKS THIS FACE MAKES THAT THE TRANSFER DOES NOT, each with its
    reason, per the shared-oracle discipline:

    1. **operand constancy.** The interval transfer is indifferent to it —
       it propagates boxes whether or not an operand is pinned — so it
       cannot live in the shared oracle without declining forms the transfer
       handles correctly.
    2. **float operand dtypes.** The oracle already forbids integer
       ACCUMULATION, and integer operands under float64 accumulation measure
       benign (5.9e-16), so the transfer admits them. The emission is over
       SMT-LIB2 Reals, which are unbounded, and an integer-dtyped operand
       reaching a Real emission is the same class the ``reduce_sum`` and
       ``scatter-add`` guards refuse. Declined here and only here.
    """
    from stelling.propagate import _dot_general_row_form

    if len(eqn.invars) != 2 or len(eqn.outvars) != 1:
        raise _Decline(
            f"'dot_general' with {len(eqn.invars)} operand(s) and "
            f"{len(eqn.outvars)} output(s) is outside the measured form"
        )
    lhs_dt = eqn.invars[0].aval.dtype or ""
    rhs_dt = eqn.invars[1].aval.dtype or ""
    dn, reason = _dot_general_row_form(eqn.params_dict(), lhs_dt, rhs_dt)
    if dn is None:
        raise _Decline(f"'dot_general' declined: {reason}")

    for name, dt in (("lhs", lhs_dt), ("rhs", rhs_dt)):
        if not dt.startswith("float"):
            raise _Decline(
                f"'dot_general' {name} operand dtype {dt!r}: the emission is "
                f"over SMT-LIB2 Reals, which are unbounded, so a non-float "
                f"operand is not modelled here even where the interval "
                f"transfer admits it"
            )

    lhs_shape = _shape_of(eqn.invars[0])
    rhs_shape = _shape_of(eqn.invars[1])
    lhs_vals = _exact_static_elements(eqns, consts, eqn.invars[0])
    rhs_vals = _exact_static_elements(eqns, consts, eqn.invars[1])
    if lhs_vals is None and rhs_vals is None:
        raise _Decline(
            "'dot_general' has NO constant operand: a sum of products of two "
            "symbolic operands is NONLINEAR arithmetic, outside this row's "
            "linear scope. The row models a constant-coefficient linear "
            "combination; it is not that the primitive is unsupported"
        )
    # Constant in EITHER position. Contract #4's constant is operand ZERO
    # (the projection matrix) while LBM's is operand one, so a row scoped to
    # "second operand constant" would miss the contract it was built for.
    if rhs_vals is not None:
        const_side, const_vals, sym_operand = 1, rhs_vals, 0
    else:
        const_side, const_vals, sym_operand = 0, lhs_vals, 1

    (lc, rc), (lb, rb) = dn
    lfree = tuple(i for i in range(len(lhs_shape)) if i not in lb and i not in lc)
    rfree = tuple(j for j in range(len(rhs_shape)) if j not in rb and j not in rc)
    out_shape = (
        tuple(lhs_shape[i] for i in lb)
        + tuple(lhs_shape[i] for i in lfree)
        + tuple(rhs_shape[j] for j in rfree)
    )
    if out_shape != _shape_of(eqn.outvars[0]):
        raise _Decline(
            f"'dot_general' output shape {_shape_of(eqn.outvars[0])} "
            f"contradicts the contraction's {out_shape} (malformed IR)"
        )
    for v in const_vals:
        if isinstance(v, bool) or not isinstance(v, (int, float, Fraction)):
            raise _Decline(
                f"'dot_general' constant operand carries a non-numeric "
                f"element {v!r}"
            )

    nb, nl = len(lb), len(lfree)
    groups: list[list[tuple[Fraction, int]]] = []
    for out_coord in _coords_of(out_shape):
        bcoord = out_coord[:nb]
        lcf = out_coord[nb:nb + nl]
        rcf = out_coord[nb + nl:]
        terms: list[tuple[Fraction, int]] = []
        for c in itertools.product(*[range(lhs_shape[i]) for i in lc]):
            ac = [0] * len(lhs_shape)
            bc = [0] * len(rhs_shape)
            for d, v in zip(lb, bcoord):
                ac[d] = v
            for d, v in zip(rb, bcoord):
                bc[d] = v
            for d, v in zip(lfree, lcf):
                ac[d] = v
            for d, v in zip(rfree, rcf):
                bc[d] = v
            for d, v in zip(lc, c):
                ac[d] = v
            for d, v in zip(rc, c):
                bc[d] = v
            ia = _flat_of(tuple(ac), lhs_shape)
            ib = _flat_of(tuple(bc), rhs_shape)
            if const_side == 1:
                coeff, sym_idx = Fraction(const_vals[ib]), ia
            else:
                coeff, sym_idx = Fraction(const_vals[ia]), ib
            terms.append((coeff, sym_idx))
        groups.append(terms)
    return sym_operand, groups


def _coords_of(shape):
    return itertools.product(*[range(d) for d in shape])


def _flat_of(coord, shape) -> int:
    i = 0
    for c, d in zip(coord, shape):
        i = i * d + c
    return i


def _exact_static_elements(eqns, consts, atom):
    """Exact per-element values of ``atom`` where it derives from
    constants/literals through STRUCTURAL routing (and the identity
    harness primitives) alone; ``None`` where it does not — i.e. where
    the value is traced/dynamic. Drives the same :func:`_route_structural`
    oracle validation, emission, and replay drive, so the three consumers
    of a static index column cannot read it differently. ``consts``
    values may be raw (:class:`stelling.ir.Array` / python scalars) or
    already-decoded element tuples (the :class:`ObligationSlice.consts`
    shape) — both are normalized here."""
    env: dict[int, tuple] = {}
    for vid, val in consts.items():
        if isinstance(val, tuple):
            env[vid] = val
            continue
        try:
            env[vid] = _decode_elements(val)
        except _Decline:
            continue  # undecodable const: simply not statically known

    def read(a: ir.Atom):
        if isinstance(a, ir.Literal):
            try:
                return _decode_elements(a.val)
            except _Decline:
                return None
        return env.get(a.id)

    for e in eqns:
        if len(e.outvars) != 1:
            continue
        if e.primitive in _IDENTITY_HARNESS:
            v = read(e.invars[0]) if e.invars else None
            if v is not None:
                env[e.outvars[0].id] = v
            continue
        if e.primitive not in _STRUCTURAL:
            continue
        ins = [read(a) for a in e.invars]
        if any(v is None for v in ins):
            continue
        try:
            routes = _route_structural(e)
        except _Decline:
            continue
        env[e.outvars[0].id] = tuple(ins[op][src] for op, src in routes)
    return read(atom)


def _scatter_set_plan(eqns, consts, eqn) -> list[tuple[int, int]]:
    """Per-OUTPUT-element ``(operand position, source flat element)`` routes for
    a static-index ``scatter`` SET equation — deliberately the same shape
    :func:`_route_structural` produces, because the set form IS pure data
    movement: every output element is either the operand's element or the one
    scalar update. No arithmetic, so nothing to round and nothing to overflow.

    One bookkeeping, three consumers (slice validation, emission, replay), as
    for every other grouping in this module.

    Declines, each quoted, and each load-bearing:

    * a form outside :func:`stelling.propagate._scatter_set_row_form` — the
      SAME predicate the propagation transfer uses, so the two faces cannot
      drift apart on shapes or dimension numbers;
    * a scatter ``mode`` other than ``FILL_OR_DROP``. ``CLIP`` silently
      REWRITES an out-of-range index to an in-range one, so encoding the
      operation as substitution at the written index would model a different
      program; ``PROMISE_IN_BOUNDS`` is undefined out of range;
    * an index not statically derivable from constants through structural
      routing, or non-integral;
    * an OUT-OF-RANGE index. This is the soundness check, not a tidiness one:
      under ``FILL_OR_DROP`` an out-of-range update is DROPPED, so the
      program's result is ``out = operand`` — emitting substitution would
      assert the update landed, which is a false model. The transfer declines
      the same case for the same reason.

    ``unique_indices`` and ``indices_are_sorted`` are deliberately NOT relied
    on: they are caller promises jax does not verify, and the covered form
    writes exactly one element, for which both are vacuous.
    """
    from stelling.propagate import _scatter_set_row_form

    if len(eqn.invars) != 3 or len(eqn.outvars) != 1:
        raise _Decline(
            f"'scatter' with {len(eqn.invars)} operand(s) and "
            f"{len(eqn.outvars)} output(s) is outside the measured form"
        )
    operand_shape = _shape_of(eqn.invars[0])
    indices_shape = _shape_of(eqn.invars[1])
    updates_shape = _shape_of(eqn.invars[2])
    out_shape = _shape_of(eqn.outvars[0])
    if out_shape != operand_shape:
        raise _Decline(
            f"'scatter' output shape {out_shape} contradicts the operand "
            f"shape {operand_shape} (malformed IR)"
        )
    params = eqn.params_dict()
    if not _scatter_set_row_form(
        params, operand_shape, indices_shape, updates_shape
    ):
        raise _Decline(
            f"'scatter' configuration (operand {operand_shape}, indices "
            f"{indices_shape}, updates {updates_shape}) is outside the "
            f"measured static-index set row form"
        )
    mode = str(params.get("mode"))
    if "FILL_OR_DROP" not in mode:
        raise _Decline(
            f"'scatter' mode {mode!r} is not FILL_OR_DROP: CLIP rewrites an "
            f"out-of-range index to an in-range one and PROMISE_IN_BOUNDS is "
            f"undefined out of range, so neither is substitution at the "
            f"written index — the mode is never guessed"
        )
    vals = _exact_static_elements(eqns, consts, eqn.invars[1])
    if vals is None:
        raise _Decline(
            "'scatter' index data not statically derivable through the "
            "supported derivation forms (constants through structural "
            "routing) — the static-index set emission needs a statically "
            "known index"
        )
    if len(vals) != 1:
        raise _Decline(
            f"'scatter' indices decode to {len(vals)} element(s) but the "
            f"covered form writes exactly one (aval/value mismatch)"
        )
    v = vals[0]
    if isinstance(v, bool) or (
        not isinstance(v, int)
        and not (isinstance(v, float) and v == math.floor(v) and math.isfinite(v))
    ):
        raise _Decline(f"'scatter' index value {v!r} is not an integer")
    k = int(v)
    if not 0 <= k < operand_shape[0]:
        raise _Decline(
            f"'scatter' index {k} is out of range for the operand's leading "
            f"axis {operand_shape[0]} — under FILL_OR_DROP the update is "
            f"DROPPED, so the result is the operand unchanged rather than a "
            f"substitution, and that is never guessed"
        )
    n = _size(operand_shape)
    return [(2, 0) if i == k else (0, i) for i in range(n)]


def _scatter_add_plan(eqns, consts, eqn) -> list[list[int]]:
    """Per-OUTPUT-element groups of contributing UPDATES flat indices for
    a pinned-form static-index ``scatter-add`` equation — the accumulate
    semantics' addend sets, shared by slice validation, the SMT emission,
    and the replay (one bookkeeping, three consumers, as for every other
    grouping in this module). ``groups[i]`` lists the updates elements
    added onto operand element ``i``; **duplicate indices produce
    multiple entries in one group** — that is the primitive's defining
    semantic, and collapsing them would be the set-form confusion.

    Declines (reason quoted) on any configuration outside the measured
    static row forms (:func:`stelling.propagate._scatter_add_row_form` is
    the one form oracle, shared with the propagation transfer), on
    indices that are not statically derivable from constants through
    structural routing (traced/dynamic indices), on non-integral index
    values, and on out-of-range indices (mode-dependent drop/clamp,
    never guessed).
    """
    from stelling.propagate import (
        _check_unique_indices_promise,
        _scatter_add_row_form,
    )

    if len(eqn.invars) != 3 or len(eqn.outvars) != 1:
        raise _Decline(
            f"'scatter-add' with {len(eqn.invars)} operand(s) and "
            f"{len(eqn.outvars)} output(s) is outside the measured form"
        )
    operand_shape = _shape_of(eqn.invars[0])
    indices_shape = _shape_of(eqn.invars[1])
    updates_shape = _shape_of(eqn.invars[2])
    out_shape = _shape_of(eqn.outvars[0])
    if out_shape != operand_shape:
        raise _Decline(
            f"'scatter-add' output shape {out_shape} contradicts the "
            f"operand shape {operand_shape} (malformed IR)"
        )
    n = _scatter_add_row_form(
        eqn.params_dict(), operand_shape, indices_shape, updates_shape
    )
    if n is None:
        raise _Decline(
            f"'scatter-add' configuration (operand {operand_shape}, "
            f"indices {indices_shape}, updates {updates_shape}) is outside "
            f"the measured static-index accumulate row forms"
        )
    vals = _exact_static_elements(eqns, consts, eqn.invars[1])
    if vals is None:
        # wording (third audit, F5a): the derivation walks constants
        # through STRUCTURAL routing only, so a constant index column
        # reaching here through a non-structural op (the traced
        # negative-index normalization arithmetic above all) is also
        # refused — "traced/dynamic" would misdescribe it
        raise _Decline(
            "'scatter-add' index data not statically derivable through "
            "the supported derivation forms (constants through structural "
            "routing) — the static-index accumulate emission needs a "
            "statically-known index column"
        )
    ks: list[int] = []
    for v in vals:
        if isinstance(v, bool) or (
            not isinstance(v, int)
            and not (isinstance(v, float) and v == math.floor(v) and math.isfinite(v))
        ):
            raise _Decline(
                f"'scatter-add' index value {v!r} is not an integer"
            )
        k = int(v)
        if not 0 <= k < operand_shape[0]:
            raise _Decline(
                f"'scatter-add' index {k} is out of range for the "
                f"operand's leading axis {operand_shape[0]} — out-of-range "
                f"handling is mode-dependent (drop vs clamp) and is never "
                f"guessed"
            )
        ks.append(k)
    if len(ks) != n:
        raise _Decline(
            f"'scatter-add' indices decode to {len(ks)} element(s) but the "
            f"form implies {n} (aval/value mismatch, malformed IR)"
        )
    # the unique_indices promise check (third audit, F2), through the SAME
    # oracle the transfer uses — promise-violated duplicates are
    # implementation-defined and decline at both layers identically
    _check_unique_indices_promise(eqn.params_dict(), ks, _Decline)
    rowsz = _size(operand_shape[1:])
    groups: list[list[int]] = [[] for _ in range(_size(operand_shape))]
    for j, k in enumerate(ks):
        for t in range(rowsz):
            groups[k * rowsz + t].append(j * rowsz + t)
    return groups


class _Slicer:
    def __init__(
        self,
        closed: ir.ClosedJaxpr,
        env: Mapping[int, iv.IntervalArray],
    ) -> None:
        self.env = env
        jaxpr = closed.jaxpr
        self.producers: dict[int, ir.JaxprEqn] = {}
        # var id -> the atom it stands for, one step (a transparent call's
        # invar binding, or its outvar's binding to the inner result)
        self.aliases: dict[int, ir.Atom] = {}
        self.consts: dict[int, object] = {}
        self.any_order: dict[int, int] = {}  # any outvar id -> declaration index
        self.eqn_order: dict[int, int] = {}  # id(eqn) -> flattened position
        self.defined: set[int] = set()  # every id bound in the flat view
        # A reused id would silently alias two different values — the
        # decorrelation bug this build must never commit. The ids must
        # therefore be unique across the FLATTENED scopes.
        #
        # THE TRANSCRIBER DOES NOT GIVE US THAT, and the comment that used to
        # sit here said it did ("the transcriber guarantees it"). That
        # claim-as-citation is why the resulting ceiling went unexamined for
        # months: it read as established, so nobody re-derived it.
        #
        # What actually happens (measured, and NOT what a first reading
        # suggests): `_Transcriber` numbers with ONE GLOBAL counter keyed by
        # `id(jax Var)` in first-encounter order — not per scope. Ids collide
        # because JAX REUSES JAXPR OBJECTS. One callee invoked at several call
        # sites yields the same inner jaxpr object each time, so the same jax
        # Vars are encountered again and correctly receive the same IR id —
        # and flattening then binds that one id twice. Two DISTINCT callees
        # collide the same way when they share a nested library jaxpr:
        # `jnp.where` is itself jit-wrapped and cached, so two callees that
        # both use it embed one identical nested jaxpr object.
        #
        # The consequence is what matters here and it is unchanged: one id,
        # two bindings, in a single flattened namespace.
        #
        # So uniqueness is now ENFORCED HERE rather than assumed: every
        # binding introduced by a transparent-call descent is given a FRESH
        # id (:meth:`_fresh`), allocated above every id the top-level scope
        # uses. The poison remains as a backstop for IR that arrives with a
        # duplicate at the SAME level — hand-built or deserialized — which
        # renumbering does not and should not paper over.
        self.poisoned: str | None = None
        self.const_avals: dict[int, ir.Aval] = {}
        # above every top-level id, so a fresh id can never collide with one
        top_ids = [v.id for v in jaxpr.constvars]
        top_ids += [v.id for v in jaxpr.invars]
        for e in jaxpr.eqns:
            top_ids += [v.id for v in e.outvars]
            top_ids += [a.id for a in e.invars if isinstance(a, ir.Var)]
        self._next_id = max(top_ids, default=0) + 1
        for var, val in zip(jaxpr.constvars, closed.consts):
            self._define(var.id, f"constvar {var.id}")
            self.consts[var.id] = val
            self.const_avals[var.id] = var.aval
        self._flatten(jaxpr.eqns)

    def _fresh(self) -> int:
        """A variable id no scope has used. Allocated strictly above the
        top-level ids, and monotonically, so no two descents can collide
        with each other either."""
        vid = self._next_id
        self._next_id += 1
        return vid

    def _renumber(self, inner: ir.ClosedJaxpr) -> dict[int, int]:
        """A fresh id for every binding the inner scope introduces.

        Covers constvars, invars and every equation outvar. Deeper scopes
        are NOT covered here: each is renumbered by its own descent, which
        is what keeps two sibling nested calls from colliding with each
        other."""
        remap: dict[int, int] = {}
        for v in (*inner.jaxpr.constvars, *inner.jaxpr.invars):
            remap[v.id] = self._fresh()
        for e in inner.jaxpr.eqns:
            for ov in e.outvars:
                remap.setdefault(ov.id, self._fresh())
        return remap

    @staticmethod
    def _renamed(atom: ir.Atom, remap: dict[int, int]) -> ir.Atom:
        if isinstance(atom, ir.Var) and atom.id in remap:
            return ir.Var(id=remap[atom.id], aval=atom.aval)
        return atom

    def _renumber_eqn(self, eqn: ir.JaxprEqn, remap: dict[int, int]) -> ir.JaxprEqn:
        """`eqn` with its inner-scope references renamed.

        `params` is carried through UNCHANGED on purpose: a nested
        ClosedJaxpr in there belongs to a deeper scope that its own descent
        will renumber, and rewriting it here would renumber it twice."""
        return ir.JaxprEqn(
            primitive=eqn.primitive,
            invars=tuple(self._renamed(a, remap) for a in eqn.invars),
            outvars=tuple(self._renamed(v, remap) for v in eqn.outvars),
            params=eqn.params,
            effects=eqn.effects,
            source_info=eqn.source_info,
        )

    def _define(self, vid: int, what: str) -> None:
        if vid in self.defined and self.poisoned is None:
            self.poisoned = (
                f"variable id {vid} is bound more than once across the "
                f"flattened call scopes ({what}); inlining transparent "
                f"calls would alias two different values"
            )
        self.defined.add(vid)

    def _flatten(self, eqns) -> None:
        """The transparent call descent for the COMPUTATION slice: inline
        every well-formed transparent wrapper's equations (jit,
        custom_jvp_call, custom_vjp_call, remat2 — the same set the
        propagation descends), binding inner invars to the call's operand
        atoms and the call's outvars to the inner results, recursively.
        Obligations/asserts remain top-level-only exactly as before — the
        assert-count mapping in :func:`slice_unknown_obligations` is
        untouched, and an inner ``stelling_assert`` still declines the
        whole mapping there. A wrapper that resists sound inlining (no
        sub-jaxpr, arity mismatch, const mismatch) is left in place as an
        opaque equation, which the validator declines with the form
        quoted."""
        for eqn in eqns:
            if eqn.primitive in DEFAULT_TRANSPARENT:
                inner = next(
                    (
                        v
                        for _, v in eqn.params
                        if isinstance(v, ir.ClosedJaxpr)
                    ),
                    None,
                )
                if (
                    inner is not None
                    and len(inner.jaxpr.invars) == len(eqn.invars)
                    and len(inner.jaxpr.outvars) == len(eqn.outvars)
                    and len(inner.jaxpr.constvars) == len(inner.consts)
                ):
                    # EVERY inner binding gets a fresh id before anything is
                    # recorded, so two descents of the same callee — or of
                    # two callees whose scopes happen to number alike —
                    # cannot land on each other. The aliases are set on the
                    # FRESH ids, which is what keeps alias resolution (and
                    # therefore the env lookups that ride on it) landing on
                    # the same outer atoms as before.
                    remap = self._renumber(inner)
                    for var, val in zip(inner.jaxpr.constvars, inner.consts):
                        nid = remap[var.id]
                        self._define(nid, f"inner constvar of {eqn.primitive!r}")
                        self.consts[nid] = val
                        self.const_avals[nid] = var.aval
                    for ivar, atom in zip(inner.jaxpr.invars, eqn.invars):
                        nid = remap[ivar.id]
                        self._define(nid, f"inner invar of {eqn.primitive!r}")
                        self.aliases[nid] = atom
                    self._flatten(
                        [self._renumber_eqn(e, remap) for e in inner.jaxpr.eqns]
                    )
                    for out, iatom in zip(eqn.outvars, inner.jaxpr.outvars):
                        self._define(out.id, f"outvar of {eqn.primitive!r}")
                        self.aliases[out.id] = self._renamed(iatom, remap)
                    continue
                # malformed wrapper: keep it opaque; _validate quotes it
            self.eqn_order[id(eqn)] = len(self.eqn_order)
            for out in eqn.outvars:
                self._define(out.id, f"outvar of {eqn.primitive!r}")
                self.producers[out.id] = eqn
            if eqn.primitive == "stelling_any":
                self.any_order[eqn.outvars[0].id] = len(self.any_order)

    def _resolve(self, atom: ir.Atom) -> ir.Atom:
        """Follow alias bindings to the atom that actually carries the
        value: a produced var, a declaration, a const var, or a literal."""
        hops = 0
        while isinstance(atom, ir.Var) and atom.id in self.aliases:
            atom = self.aliases[atom.id]
            hops += 1
            if hops > len(self.aliases):
                raise _Decline(
                    "alias resolution does not terminate (cyclic call "
                    "binding in the flattened scopes)"
                )
        return atom

    # -- validation of one equation form -------------------------------------
    #
    # Form checks only (dtypes, params, guards). There is deliberately NO
    # per-equation size gate here: the one size gate is the per-obligation
    # element budget in :meth:`slice` — a single choke point cannot
    # disagree with itself, and the seven scattered scalar-only gates it
    # replaced could.

    def _validate(self, eqn: ir.JaxprEqn) -> None:
        prim = eqn.primitive
        for atom in (*eqn.invars, *eqn.outvars):
            problem = _shape_problem(atom.aval.shape)
            if problem is not None:
                # an uninhabited/malformed shape (fix-re-attacks R1/N1/N2):
                # measured jax rejects negative extents in every concrete
                # context, and a non-integer extent is IR no trace can
                # carry — nothing here may route, count, or emit over
                # either. The interval oracle refuses them too
                # (check_shape); this is the slicer's OWN guard, because a
                # layer that can mint emission independently needs its own
                # refusal.
                raise _Decline(
                    f"primitive {prim!r} touches a value of shape "
                    f"{tuple(atom.aval.shape)} with {problem}"
                )
        if prim in DEFAULT_TRANSPARENT:
            raise _Decline(
                f"transparent {prim!r} could not be inlined (no sub-jaxpr, "
                f"or call/sub-jaxpr arity mismatch); the wrapper form "
                f"resists sound descent"
            )
        if prim not in _SUPPORTED:
            raise _Decline(
                f"primitive {prim!r} is outside the supported emission set"
            )
        params = eqn.params_dict()
        if prim in _BOOL_OPS:
            dtypes = [a.aval.dtype for a in eqn.invars]
            if any(d != "bool" for d in dtypes):
                raise _Decline(
                    f"primitive {prim!r} on non-boolean operands (dtypes "
                    f"{dtypes}): bitwise integer form has no Real emission"
                )
        if prim in ("lt", "le", "gt", "ge"):
            if any(_is_bool_dtype(a.aval) for a in eqn.invars):
                raise _Decline(
                    f"order comparison {prim!r} on boolean operands has no "
                    f"v1 emission"
                )
        if prim in ("eq", "ne"):
            kinds = {_is_bool_dtype(a.aval) for a in eqn.invars}
            if len(kinds) > 1:
                raise _Decline(
                    f"comparison {prim!r} mixes boolean and numeric operands"
                )
        if prim in ("max", "min", "add", "sub", "mul", "neg", "div"):
            if any(_is_bool_dtype(a.aval) for a in eqn.invars):
                raise _Decline(
                    f"arithmetic {prim!r} on boolean operands has no v1 "
                    f"emission"
                )
        if prim == "div":
            dt = eqn.invars[0].aval.dtype or ""
            if not dt.startswith("float"):
                raise _Decline(
                    f"'div' on dtype {dt!r}: jax integer division truncates, "
                    f"which Real division does not model"
                )
            problem = _zero_element_problem(eqn.invars[1], self.env)
            if problem is not None:
                # the per-element div guard: every divisor element must
                # have an interval definitely excluding 0; any element
                # straddling declines, naming the element
                raise _Decline(f"'div': {DIV_GUARD_REASON}{problem}")
        if prim == "integer_pow":
            y = params.get("y")
            if not isinstance(y, int):
                raise _Decline(f"'integer_pow' with non-integer exponent {y!r}")
            if abs(y) > INTEGER_POW_EXPANSION_CAP:
                raise _Decline(
                    f"'integer_pow' exponent {y} exceeds the v1 expansion "
                    f"cap ({INTEGER_POW_EXPANSION_CAP})"
                )
            if y < 0:
                problem = _zero_element_problem(eqn.invars[0], self.env)
                if problem is not None:
                    raise _Decline(
                        f"'integer_pow' with negative exponent {y}: "
                        f"{DIV_GUARD_REASON}{problem}"
                    )
        if prim in _INT_OVERFLOW_EMITTED:
            # SMT-LIB2 Reals are unbounded; jax integers wrap. Emitting a
            # computed integer as a Real would let the solver prove a claim
            # the program falsifies, so integer dtypes decline here — the
            # emission is stricter than the transfer on purpose. The
            # transfer can settle in-range integer arithmetic itself
            # (propagate._int_overflow_guard); escalating it would need the
            # Real relaxation to be argued faithful, and this round declines
            # rather than makes that argument load-bearing.
            dt = eqn.outvars[0].aval.dtype or ""
            if dt in _INT_DTYPE_BOUNDS:
                raise _Decline(
                    f"{prim!r} on dtype {dt!r}: jax integer arithmetic wraps "
                    f"on overflow and SMT-LIB2 Reals are unbounded, so a Real "
                    f"emission does not model it"
                )
        if prim == "reduce_sum":
            dt = eqn.invars[0].aval.dtype or ""
            if not dt.startswith("float"):
                raise _Decline(
                    f"'reduce_sum' on dtype {dt!r}: jax integer addition "
                    f"wraps on overflow, which Real addition does not model"
                )
        if prim == "scatter":
            if len(eqn.invars) != 3:
                raise _Decline(
                    f"'scatter' with {len(eqn.invars)} operand(s) is "
                    f"outside the measured form"
                )
            dts = {
                (a.aval.dtype or "")
                for a in (eqn.invars[0], eqn.invars[2], eqn.outvars[0])
            }
            if len(dts) != 1:
                raise _Decline(
                    f"'scatter' operand/updates/output dtypes {sorted(dts)} "
                    f"must agree: the set form ALIASES terms rather than "
                    f"computing them, and aliasing across dtypes would equate "
                    f"values of different sorts"
                )
            # NOTE the deliberate absence of scatter-add's float-only clause.
            # The accumulate needs it because jax integer addition wraps; the
            # SET form performs no arithmetic at all, so an integer scatter is
            # exact data movement and admissible.
        if prim == "scatter-add":
            if len(eqn.invars) != 3:
                raise _Decline(
                    f"'scatter-add' with {len(eqn.invars)} operand(s) is "
                    f"outside the measured form"
                )
            dts = {
                (a.aval.dtype or "")
                for a in (eqn.invars[0], eqn.invars[2], eqn.outvars[0])
            }
            if len(dts) != 1 or not next(iter(dts)).startswith("float"):
                raise _Decline(
                    f"'scatter-add' operand/updates/output dtypes {sorted(dts)} "
                    f"must be one float dtype: the accumulate is Real "
                    f"addition, and jax integer addition wraps on overflow"
                )
        if prim == "select_n":
            if len(eqn.invars) != 3:
                raise _Decline(
                    f"'select_n' with {len(eqn.invars) - 1} cases (only the "
                    f"two-case boolean form has an ite emission)"
                )
            if not _is_bool_dtype(eqn.invars[0].aval):
                raise _Decline(
                    f"'select_n' with non-boolean selector dtype "
                    f"{eqn.invars[0].aval.dtype!r} (only the boolean-selector "
                    f"form has an ite emission)"
                )
            case_bool = {_is_bool_dtype(a.aval) for a in eqn.invars[1:]}
            if len(case_bool) > 1:
                raise _Decline("'select_n' cases mix boolean and numeric sorts")
        if prim == "convert_element_type":
            src = eqn.invars[0].aval.dtype or ""
            dst = str(params.get("new_dtype"))
            if not (src == dst or (src, dst) in _EXACT_CONVERSIONS):
                raise _Decline(
                    f"'convert_element_type' {src!r} -> {dst!r} is "
                    f"value-changing (outside the exact-conversions whitelist)"
                )
        # the index bookkeeping itself is the shape validation: malformed
        # structural params, non-broadcastable operand shapes, and an
        # output aval contradicting the computed shape all decline here —
        # through the SAME helpers the emission and the replay will drive,
        # so what validates is exactly what emits and replays
        if prim in _STRUCTURAL:
            _route_structural(eqn)
        elif prim == "reduce_sum":
            _group_reduce_sum(eqn)
        elif prim == "select_n":
            _pair_select_n(eqn)
        elif prim in ("scatter", "scatter-add", "dot_general"):
            # ALL THREE plans need the whole slice's dataflow -- the scatter
            # rows' index columns and dot_general's constant operand each
            # derive from constants through structural routing -- so they are
            # validated once over the completed topological slice in
            # :meth:`slice`, through the same plan the emission and the replay
            # drive. One branch rather than three: the reason is identical and
            # a per-primitive branch is three places for it to drift.
            pass
        else:  # elementwise ops and the identity harness primitives
            _pair_elementwise(eqn)

    # -- the backward slice ---------------------------------------------------

    def _rewrite(self, eqn: ir.JaxprEqn) -> ir.JaxprEqn:
        """The eqn with every invar resolved through the call-scope alias
        bindings — the inlining substitution. Unchanged eqns (the whole
        query, when no transparent call was descended) are returned as the
        ORIGINAL objects, so a slice through top-level-only computation is
        object-identical to the pre-descent one."""
        if not self.aliases:
            return eqn
        resolved = tuple(self._resolve(a) for a in eqn.invars)
        if all(r is a for r, a in zip(resolved, eqn.invars)):
            return eqn
        return ir.JaxprEqn(
            primitive=eqn.primitive,
            invars=resolved,
            outvars=eqn.outvars,
            params=eqn.params,
            effects=eqn.effects,
            source_info=eqn.source_info,
        )

    def slice(self, index: int, assert_eqn: ir.JaxprEqn) -> ObligationSlice:
        if self.poisoned is not None:
            raise _Decline(self.poisoned)
        root = self._resolve(assert_eqn.invars[0])
        root_size = _size(root.aval.shape)
        if not _is_bool_dtype(root.aval):
            raise _Decline(
                f"assert operand has dtype {root.aval.dtype!r}, expected bool"
            )
        root_problem = _shape_problem(root.aval.shape)
        if root_problem is not None:
            raise _Decline(
                f"assert operand has shape {tuple(root.aval.shape)} with "
                f"{root_problem}"
            )
        if root_size == 0:
            # zero elements: the universal claim is vacuously true, and
            # interval propagation already discharges it (matching measured
            # jax: jnp.all of a size-0 predicate is True), so nothing
            # reaches escalation through the normal path — a direct ask
            # declines rather than mints a vacuous proof obligation.
            raise _Decline(
                f"assert operand has shape {tuple(root.aval.shape)} with "
                f"zero elements: the empty universal claim is vacuously "
                f"true and is decided by interval propagation, not by "
                f"emission"
            )

        # -- pass 1: reachability + THE choke point, from static shape ---
        # metadata ONLY. The budget is the single gate that replaced the
        # seven scalar-only sites, and it must be CHEAP to say no: no
        # constant decoding, no routing validation, no per-element
        # allocation happens before it — an over-budget decline costs
        # O(#equations), never O(#elements) (first-contact audit F2).
        needed: list[ir.Atom] = [root]
        # id(original eqn) -> (flattened order, original eqn)
        seen_eqns: dict[int, tuple[int, ir.JaxprEqn]] = {}
        inputs: dict[int, None] = {}
        const_ids: dict[int, None] = {}
        literal_atoms: list[ir.Literal] = []
        while needed:
            atom = self._resolve(needed.pop())
            if isinstance(atom, ir.Literal):
                literal_atoms.append(atom)  # decoded in pass 2, post-gate
                continue
            if atom.id in self.consts:
                const_ids[atom.id] = None  # decoded in pass 2, post-gate
                continue
            producer = self.producers.get(atom.id)
            if producer is None:
                raise _Decline(
                    f"variable {atom.id} has no producer in the flattened "
                    f"computation (an opaque sub-jaxpr boundary the descent "
                    f"could not cross)"
                )
            if producer.primitive == "stelling_any":
                inputs[atom.id] = None
                continue
            if id(producer) in seen_eqns:
                continue
            for patom in (*producer.invars, *producer.outvars):
                problem = _shape_problem(patom.aval.shape)
                if problem is None and isinstance(patom, ir.Literal) and isinstance(patom.val, ir.Array):
                    problem = _shape_problem(patom.val.shape)
                if problem is not None:
                    # probed in pass 1, before the budget count reads any
                    # of these shapes: a malformed extent would corrupt
                    # the count silently (`1 * "x"` repeats the string)
                    # or raise raw (fix-re-attack N2's adjacent case)
                    raise _Decline(
                        f"primitive {producer.primitive!r} touches a value "
                        f"of shape {tuple(patom.aval.shape)} with {problem}"
                    )
            seen_eqns[id(producer)] = (self.eqn_order[id(producer)], producer)
            needed.extend(producer.invars)

        # The gate's two quantities (audit F1): the emitted element TERMS
        # (declared input elements + term-producing output elements;
        # reduce_sum additionally counts its operand elements — its n-ary
        # bodies inline one addend per operand element, so the operand,
        # not the output, is its script cost) and the ROOT CONJUNCT count
        # (the assert operand's elements: structural routing mints no
        # terms, but the negated conjunction it feeds repeats one term
        # per element — that repetition is script cost the term count
        # cannot see). One gate, both quantities, both quoted.
        element_terms = 0
        for vid in inputs:
            decl_shape = tuple(
                self.producers[vid].params_dict().get("shape", ())
            )
            problem = _shape_problem(decl_shape)
            if problem is not None:
                raise _Decline(
                    f"input declaration of shape {decl_shape!r} has "
                    f"{problem}"
                )
            element_terms += _size(tuple(int(d) for d in decl_shape))
        for _, eqn in seen_eqns.values():
            prim = eqn.primitive
            if prim in _STRUCTURAL or prim in _IDENTITY_HARNESS:
                continue  # index routing: no new terms
            if prim == "scatter":
                # the SET form aliases: element k IS the update's term, the
                # rest ARE the operand's. No arithmetic anywhere, so no new
                # term anywhere — the same accounting as the structural
                # routes above, and unlike scatter-add, which inlines one
                # addend per operand element and one per updates element.
                continue
            element_terms += _size(eqn.outvars[0].aval.shape)
            if prim == "reduce_sum" and eqn.invars:
                element_terms += _size(eqn.invars[0].aval.shape)
            if prim == "scatter-add" and len(eqn.invars) == 3:
                # the accumulate bodies inline one addend per operand
                # element and one per updates element (each update
                # contributes to exactly one output element) — the same
                # operand-side script cost reduce_sum counts
                element_terms += _size(eqn.invars[0].aval.shape)
                element_terms += _size(eqn.invars[2].aval.shape)
        if element_terms > ELEMENT_BUDGET or root_size > ELEMENT_BUDGET:
            raise _Decline(
                f"obligation needs {element_terms} element terms and "
                f"{root_size} root conjuncts, over the per-obligation "
                f"emission budget of {ELEMENT_BUDGET} (bounded static-shape "
                f"emission; the budget is measured solver cost, see "
                f"stelling.obligation.ELEMENT_BUDGET)"
            )

        # -- pass 2: decode, validate, build — all bounded by the gate ---
        for atom in literal_atoms:
            for v in _decode_elements(atom.val):  # declines undecodables
                if not isinstance(v, bool):
                    _numeric_fraction(v)  # declines non-finite
        used_consts: dict[int, object] = {}
        for cid in const_ids:
            c_aval = self.const_avals.get(cid)
            if c_aval is not None:
                problem = _shape_problem(c_aval.shape)
                if problem is not None:
                    # P1(b): the slicer rebuilds values independently of
                    # the propagation, so it must refuse the refused
                    # class itself — the constvar's DECLARED aval, which
                    # a lying consumer reference never shows it
                    raise _Decline(
                        f"constvar {cid} has aval shape "
                        f"{tuple(c_aval.shape)} with {problem}"
                    )
            vals = _decode_elements(self.consts[cid])
            if c_aval is not None and len(vals) != _size(c_aval.shape):
                raise _Decline(
                    f"constvar {cid} decodes to {len(vals)} element(s) "
                    f"but its aval shape {tuple(c_aval.shape)} holds "
                    f"{_size(c_aval.shape)} (aval/value mismatch, "
                    f"malformed IR)"
                )
            for v in vals:
                if not isinstance(v, bool):
                    _numeric_fraction(v)
            # a single-element const stores its bare value (the pre-array
            # shape of the field, byte-identical); an array const stores
            # the flat tuple of its elements
            used_consts[cid] = vals[0] if len(vals) == 1 else vals
        ordered = tuple(
            self._validated_rewrite(e)
            for _, e in sorted(seen_eqns.values(), key=lambda t: t[0])
        )
        for e in ordered:
            if e.primitive == "scatter":
                # same posture as the accumulate below: form oracle, static
                # index, mode, and the in-range check all run over the
                # completed slice through the SAME _scatter_set_plan the
                # emission and the replay drive
                _scatter_set_plan(ordered, used_consts, e)
            if e.primitive == "scatter-add":
                # whole-slice validation of the accumulate plan: the form
                # oracle, the static index column (derivable from the
                # slice's own constants through structural routing), and
                # the in-range check — through the SAME _scatter_add_plan
                # the emission and the replay drive, so what validates is
                # exactly what emits and replays
                _scatter_add_plan(ordered, used_consts, e)
            elif e.primitive == "dot_general":
                # same posture, same reason: the constant operand derives
                # from the slice's own constants through structural
                # routing, so the plan needs the completed slice and is
                # driven here through the one function the emission and
                # the replay also drive
                _dot_general_plan(ordered, used_consts, e)
        slice_inputs: list[SliceInput] = []
        for vid in sorted(inputs, key=lambda v: self.any_order[v]):
            params = self.producers[vid].params_dict()
            dtype = str(params.get("dtype"))
            if dtype not in _FLOAT_INPUT_DTYPES:
                raise _Decline(
                    f"input declaration of dtype {dtype!r}: only float "
                    f"declarations are supported (an int/bool input's "
                    f"real relaxation would admit non-member witnesses)"
                )
            lo = float(params.get("lo", math.nan))
            hi = float(params.get("hi", math.nan))
            if math.isnan(lo) or math.isnan(hi) or lo > hi:
                raise _Decline(
                    f"input declaration with bounds ({lo!r}, {hi!r}) has "
                    f"no emission (empty or NaN-bounded set)"
                )
            shape = tuple(int(d) for d in params.get("shape", ()))
            if any(d < 0 for d in shape):
                # an empty declared set (fix-re-attack R1): no array of a
                # negative-extent shape exists, so there is nothing to
                # declare and any universal claim over it is vacuous —
                # the public API refuses this at declaration; from_dict
                # queries decline here, quoted
                raise _Decline(
                    f"input declaration of shape {shape} has a negative "
                    f"extent: no jax program constructs such an array, so "
                    f"the declared set is empty"
                )
            k = self.any_order[vid]
            if shape == ():
                slice_inputs.append(
                    SliceInput(name=f"x{k}", var_id=vid, lo=lo, hi=hi)
                )
            else:
                # one SMT constant per element, deterministically named
                # x{k}_{i} (flat C-order), each carrying the declaration's
                # bounds — scalar bounds broadcast to every element
                for i in range(_size(shape)):
                    slice_inputs.append(
                        SliceInput(
                            name=f"x{k}_{i}",
                            var_id=vid,
                            lo=lo,
                            hi=hi,
                            shape=shape,
                            element=i,
                        )
                    )
        fragment = self._fragment(tuple(slice_inputs), ordered)
        return ObligationSlice(
            index=index,
            fragment=fragment,
            inputs=tuple(slice_inputs),
            consts=tuple(sorted(used_consts.items())),
            eqns=ordered,
            root=root,
            source_info=assert_eqn.source_info,
            element_terms=element_terms,
        )

    def _validated_rewrite(self, eqn: ir.JaxprEqn) -> ir.JaxprEqn:
        sub = self._rewrite(eqn)
        self._validate(sub)
        return sub

    # -- fragment classification ---------------------------------------------

    def _fragment(
        self,
        inputs: tuple[SliceInput, ...],
        eqns: tuple[ir.JaxprEqn, ...],
    ) -> str:
        dependent: set[int] = {i.var_id for i in inputs}

        def dep(atom: ir.Atom) -> bool:
            return isinstance(atom, ir.Var) and atom.id in dependent

        nonlinear = False
        for eqn in eqns:  # topological order: dependency flows forward
            prim = eqn.primitive
            ins_dep = [dep(a) for a in eqn.invars]
            if prim == "mul" and all(ins_dep):
                nonlinear = True
            if prim == "div" and ins_dep[1]:
                nonlinear = True
            if prim == "integer_pow" and ins_dep[0]:
                y = eqn.params_dict().get("y")
                if y not in (0, 1):
                    nonlinear = True
            if any(ins_dep):
                for out in eqn.outvars:
                    dependent.add(out.id)
        return QF_NRA if nonlinear else QF_LRA


def slice_obligation(
    closed: ir.ClosedJaxpr,
    index: int,
    env: Mapping[int, iv.IntervalArray],
) -> ObligationSlice | DeclinedObligation:
    """Extract the slice for obligation ``index`` (top-level assert order),
    or decline with the reason quoted. Never raises on legal queries."""
    jaxpr = closed.jaxpr
    asserts = [e for e in jaxpr.eqns if e.primitive == "stelling_assert"]
    if index >= len(asserts):
        return DeclinedObligation(
            index=index,
            reason=(
                f"obligation #{index} has no matching top-level "
                f"stelling_assert equation"
            ),
        )
    assert_eqn = asserts[index]
    try:
        return _Slicer(closed, env).slice(index, assert_eqn)
    except _Decline as d:
        return DeclinedObligation(
            index=index, reason=d.reason, source_info=assert_eqn.source_info
        )


def slice_unknown_obligations(
    closed: ir.ClosedJaxpr,
    propagation: Propagation,
    env: Mapping[int, iv.IntervalArray],
) -> tuple[ObligationSlice | DeclinedObligation, ...]:
    """Slices (or quoted declines) for exactly the obligations interval
    propagation left ``unknown``. Discharged and violated obligations are
    already decided and are not re-decided."""
    unknown = [o for o in propagation.obligations if o.status == "unknown"]
    asserts = [
        e for e in closed.jaxpr.eqns if e.primitive == "stelling_assert"
    ]
    if len(asserts) != len(propagation.obligations):
        # obligations were recorded from inside sub-jaxprs (transparent
        # wrappers / cond branches): index-based mapping onto top-level
        # asserts would be a guess, so every unknown obligation declines.
        # The call descent inlines sub-jaxpr COMPUTATION only; obligations
        # remain top-level-only, exactly as before.
        return tuple(
            DeclinedObligation(
                index=o.index,
                reason=(
                    f"{len(propagation.obligations)} obligation(s) but "
                    f"{len(asserts)} top-level stelling_assert equation(s): "
                    f"asserts nested in sub-jaxprs cannot be mapped to "
                    f"slices"
                ),
                source_info=o.source_info,
            )
            for o in unknown
        )
    return tuple(slice_obligation(closed, o.index, env) for o in unknown)


# -- exact-rational replay ----------------------------------------------------
#
# THE EMISSION == REPLAY INVARIANT, machine-checked.
#
# Replay is what makes REFUTED self-certifying: a solver model is only ever
# promoted to a Witness after this module re-derives the violation in exact
# rational arithmetic, independently of the solver. That guarantee is only as
# broad as replay's own coverage — a primitive that can be EMITTED but not
# REPLAYED yields a witness nobody can independently check, which is the
# interval-REFUTED-without-a-witness problem wearing a different costume.
#
# Measured 2026-07-26: the two sets are currently EQUAL, in both directions.
# That equality is an invariant to preserve, not a coincidence to note, and it
# is exactly the kind that decays silently — adding a primitive to _SUPPORTED
# is a one-line edit, and nothing about that edit points at this file. So it is
# asserted at import, in the same posture as the int-semantics census above.
#
# The declaration below must mirror the dispatch in _root_elements and
# _scalar_binop. It is the weaker half of the check: it pins that the SETS
# agree, not that each branch is reachable and correct. The stronger half is a
# probe test that drives every member through replay; the two together are the
# census.
_REPLAY_SUPPORTED = (
    _ARITH | _COMPARE | _BOOL_OPS | _STRUCTURAL | _IDENTITY_HARNESS
    | {"reduce_sum", "select_n", "convert_element_type", "scatter-add",
       "dot_general", "scatter"}
)

if _REPLAY_SUPPORTED != _SUPPORTED:
    raise RuntimeError(
    "the exact-rational replay must cover exactly the emission set, or a "
    "witness can be produced that replay cannot independently confirm: "
    f"emittable-but-not-replayable {sorted(_SUPPORTED - _REPLAY_SUPPORTED)}, "
    f"replayable-but-not-emittable {sorted(_REPLAY_SUPPORTED - _SUPPORTED)}"
)



def _scalar_binop(prim: str, a, b):
    if prim == "add":
        return a + b
    if prim == "sub":
        return a - b
    if prim == "mul":
        return a * b
    if prim == "div":
        return a / b
    if prim == "max":
        return max(a, b)
    if prim == "min":
        return min(a, b)
    if prim == "lt":
        return a < b
    if prim == "le":
        return a <= b
    if prim == "gt":
        return a > b
    if prim == "ge":
        return a >= b
    if prim == "eq":
        return a == b
    if prim == "ne":
        return a != b
    if prim == "and":
        return a and b
    if prim == "or":
        return a or b
    if prim == "xor":
        return bool(a) != bool(b)
    raise ReplayError(
        f"no replay rule for primitive {prim!r} (slice should have declined)"
    )


def _root_elements(
    sl: ObligationSlice, values: Mapping[str, Fraction]
) -> tuple[bool, ...]:
    """The per-element truth values of the assert operand at a concrete
    rational point, flat C-order — the replay engine. Pure
    :class:`fractions.Fraction` arithmetic, elementwise through the same
    pairing/routing/grouping helpers validation and emission drive.
    Anything inexact or impossible raises :exc:`ReplayError`."""
    env: dict[int, tuple] = {}
    per_var: dict[int, dict[int, Fraction]] = {}
    sizes: dict[int, int] = {}
    for inp in sl.inputs:
        if inp.name not in values:
            raise ReplayError(f"witness has no value for input {inp.name}")
        v = values[inp.name]
        if not isinstance(v, Fraction):
            raise ReplayError(
                f"witness value for {inp.name} is {type(v).__name__}, not an "
                f"exact Fraction"
            )
        per_var.setdefault(inp.var_id, {})[inp.element] = v
        sizes[inp.var_id] = max(sizes.get(inp.var_id, 1), _size(inp.shape))
    for vid, elems in per_var.items():
        n = sizes[vid]
        if set(elems) != set(range(n)):
            raise ReplayError(
                f"witness elements for input variable {vid} do not cover "
                f"its {n} element(s)"
            )
        env[vid] = tuple(elems[i] for i in range(n))

    for var_id, val in sl.consts:
        vals = val if isinstance(val, tuple) else (val,)
        env[var_id] = tuple(
            v if isinstance(v, bool) else _numeric_fraction(v) for v in vals
        )

    def read(atom: ir.Atom) -> tuple:
        if isinstance(atom, ir.Literal):
            return tuple(
                v if isinstance(v, bool) else _numeric_fraction(v)
                for v in _decode_elements(atom.val)
            )
        if atom.id in env:
            return env[atom.id]
        if _size(atom.aval.shape) == 0:
            return ()  # a zero-size value has no elements to bind
        raise ReplayError(f"replay reads unbound variable {atom.id}")

    for eqn in sl.eqns:
        prim = eqn.primitive
        params = eqn.params_dict()
        try:
            ins = [read(a) for a in eqn.invars]
            n_out = _size(_shape_of(eqn.outvars[0]))
            if prim in _STRUCTURAL:
                routes = _route_structural(eqn)
                out = tuple(ins[op][src] for op, src in routes)
            elif prim in _IDENTITY_HARNESS:
                out = ins[0]
            elif prim == "reduce_sum":
                groups = _group_reduce_sum(eqn)
                out = tuple(
                    sum((ins[0][i] for i in group), Fraction(0))
                    for group in groups
                )
            elif prim == "scatter":
                # pure data movement, so replay is the same aliasing the
                # emission performs and the transfer performs: element k IS
                # the update, every other element IS the operand's. Driven by
                # the SAME _scatter_set_plan, so replay cannot route an
                # element differently from the emission it is checking.
                routes = _scatter_set_plan(sl.eqns, dict(sl.consts), eqn)
                out = tuple(ins[op][src] for op, src in routes)
            elif prim == "scatter-add":
                # the same accumulate the transfer and the emission
                # perform: operand element + Σ contributing updates, in
                # exact rational arithmetic (ℝ addition is associative, so
                # the fold order here denotes THE value); duplicates
                # contribute once EACH — the defining semantic
                groups = _scatter_add_plan(sl.eqns, dict(sl.consts), eqn)
                out = tuple(
                    sum((ins[2][u] for u in groups[i]), ins[0][i])
                    for i in range(n_out)
                )
            elif prim == "dot_general":
                # the same linear combination the emission builds, in exact
                # rational arithmetic: Σ c_k · sym_k. ℝ addition is
                # associative so the fold order denotes THE value, and the
                # coefficients come from the one plan both faces drive — a
                # replay that recomputed them independently could confirm a
                # witness the emission never described.
                sym_operand, groups = _dot_general_plan(
                    sl.eqns, dict(sl.consts), eqn
                )
                out = tuple(
                    sum(
                        (coeff * ins[sym_operand][idx] for coeff, idx in terms),
                        Fraction(0),
                    )
                    for terms in groups
                )
            elif prim == "select_n":
                which, on_false, on_true = _pair_select_n(eqn)
                out = tuple(
                    ins[2][on_true[i]] if ins[0][which[i]] else ins[1][on_false[i]]
                    for i in range(n_out)
                )
            elif prim == "neg":
                (idx,) = _pair_elementwise(eqn)
                out = tuple(-ins[0][i] for i in idx)
            elif prim == "not":
                (idx,) = _pair_elementwise(eqn)
                out = tuple(not ins[0][i] for i in idx)
            elif prim == "integer_pow":
                (idx,) = _pair_elementwise(eqn)
                y = int(params["y"])
                out = tuple(ins[0][i] ** y for i in idx)
            elif prim == "convert_element_type":
                (idx,) = _pair_elementwise(eqn)
                dst = str(params.get("new_dtype"))
                src = str(eqn.invars[0].aval.dtype or "")
                # RE-DERIVE the exactness judgement; do not inherit it.
                # Replay's whole job is to confirm a violation INDEPENDENTLY
                # of the machinery that produced it, and every other branch
                # here re-derives its routing through the shared helpers. This
                # one used to read only the destination dtype and treat any
                # other conversion as the identity on the rational — so a
                # value-changing narrowing (float64 -> float32) would have been
                # carried through unrounded, and replay would have been
                # agreeing with the emission because it had asked the same
                # question, not because it had answered it. Unreachable today
                # (the emission declines such a slice before replay sees it),
                # which is precisely why it could sit here unnoticed.
                if not (src == dst or (src, dst) in _EXACT_CONVERSIONS):
                    raise ReplayError(
                        f"replay cannot re-derive a value-changing conversion "
                        f"{src!r} -> {dst!r}: the exact rational would have to "
                        f"be rounded to the destination format, and replay "
                        f"models exact arithmetic. The emission declines this "
                        f"form too — replay refuses independently rather than "
                        f"relying on that."
                    )
                out = tuple(
                    Fraction(1 if ins[0][i] else 0)
                    if isinstance(ins[0][i], bool) and dst != "bool"
                    else ins[0][i]
                    for i in idx
                )
            else:
                ia, ib = _pair_elementwise(eqn)
                out = tuple(
                    _scalar_binop(prim, ins[0][ia[i]], ins[1][ib[i]])
                    for i in range(n_out)
                )
        except ZeroDivisionError as e:
            raise ReplayError(
                f"division by zero at {prim!r} during replay: {e}"
            ) from e
        except IndexError as e:
            raise ReplayError(
                f"element routing escaped its operand at {prim!r} during "
                f"replay: {e}"
            ) from e
        except _Decline as d:
            raise ReplayError(f"replay cannot evaluate: {d.reason}") from d
        for outvar in eqn.outvars:
            env[outvar.id] = out

    result: tuple | None = None
    if isinstance(sl.root, ir.Literal):
        try:
            result = tuple(_decode_elements(sl.root.val))
        except _Decline as d:
            raise ReplayError(
                f"undecodable predicate literal: {d.reason}"
            ) from d
    elif sl.root.id in env:
        result = env[sl.root.id]
    if result is None or not result or not all(
        isinstance(r, bool) for r in result
    ):
        got = (
            "nothing"
            if result is None
            else "zero elements"
            if not result
            else type(next(r for r in result if not isinstance(r, bool))).__name__
        )
        raise ReplayError(
            f"replay produced {got} for the predicate, expected bool"
        )
    return tuple(result)


def evaluate_predicate(
    sl: ObligationSlice, values: Mapping[str, Fraction]
) -> bool:
    """Evaluate the obligation predicate at a concrete rational point.

    ``values`` maps input names (``x0``, ``x1_2``, …) to exact rationals,
    one per declared element. Pure Python :class:`fractions.Fraction`
    arithmetic — no solver, no floats. The obligation is the UNIVERSAL
    elementwise claim, so this returns the conjunction of the per-element
    truth values (``False`` means at least one element is violated at the
    point — for a scalar operand, exactly the pre-array meaning).
    Anything inexact or impossible raises :exc:`ReplayError` — the caller
    treats that as emission infidelity.
    """
    return all(_root_elements(sl, values))


def violating_elements(
    sl: ObligationSlice, values: Mapping[str, Fraction]
) -> tuple[int, ...]:
    """Flat C-order indices of the assert-operand elements that are FALSE
    at the point — the "which element(s) violate" fact of an array-scale
    witness, computed by the same replay the validator's violation
    conjunct runs (:func:`_root_elements`; one computation, so the naming
    can never disagree with the decision). For a SCALAR assert operand
    this returns ``()`` — the violated predicate is the scalar itself,
    and the pre-array witness rendering carries no element line."""
    elements = _root_elements(sl, values)
    if len(elements) == 1 and _size(sl.root.aval.shape) == 1:
        return ()
    return tuple(i for i, ok in enumerate(elements) if not ok)


# -- the witness validator -----------------------------------------------------


def witness_is_valid(
    sl: ObligationSlice, values: Mapping[str, Fraction]
) -> str | None:
    """The single validator of the whole refutation claim, both conjuncts.

    A REFUTED-with-witness means "∃w: in_box(w) ∧ violates(w)". This
    function is the ONLY place either conjunct is computed for witness
    purposes — membership and violation live as one conjunction here, so a
    refactor cannot separate them again:

    * **membership**: every value is a member of its input's declared
      closed box, compared as exact rationals on finite sides only (a
      half-infinite bound checks its finite side; a ``(-inf, inf)`` input
      is unconstrained). The box constraints are part of the emitted
      problem, so an escaping model means the emitted problem does not
      mean the obligation.
    * **violation**: the predicate is false at the point, by the exact
      :class:`fractions.Fraction` replay (:func:`evaluate_predicate`, the
      shared replay engine this validator calls — dispatch code never
      calls it directly for witness acceptance).

    A constants-only refutation routes through here too, with an empty
    ``values`` mapping: membership is vacuously true and the violation is
    the empty-environment replay of the closed formula.

    Returns None when the refutation is real, else a human-readable
    string naming the failing conjunct (for the loud
    emission-infidelity error).
    """
    for inp in sl.inputs:
        v = values.get(inp.name)
        if not isinstance(v, Fraction):
            # a missing or inexact value: the replay conjunct below names
            # it precisely (ReplayError), so membership defers
            continue
        if inp.lo != float("-inf") and v < Fraction(inp.lo):
            return (
                f"the model escapes the declared box ({inp.name} = {v} is "
                f"below its declared lower bound {Fraction(inp.lo)}); the "
                f"box constraints were part of the emitted problem"
            )
        if inp.hi != float("inf") and v > Fraction(inp.hi):
            return (
                f"the model escapes the declared box ({inp.name} = {v} is "
                f"above its declared upper bound {Fraction(inp.hi)}); the "
                f"box constraints were part of the emitted problem"
            )
    try:
        holds = evaluate_predicate(sl, values)
    except ReplayError as e:
        return f"the replay could not evaluate it ({e})"
    if holds:
        return (
            "the exact-rational replay found the predicate TRUE at that point"
        )
    return None
