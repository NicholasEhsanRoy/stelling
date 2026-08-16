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
``div``, ``integer_pow``, ``square`` (the self-product of ONE term — jax's
own primitive on this series, not sugar for ``integer_pow``), ``max``,
``min``, the comparisons, boolean
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
non-float input declarations, and obligations whose ``stelling_assert`` is
not a top-level equation of the query (one written inside a ``jax.jit``
helper, a ``cond`` branch, or an undescended ``scan``/``while_loop`` body)
— **declines**, with the primitive and
form (and, for the budget, the count and the budget) quoted, and the
obligation stays UNKNOWN. **Every one of those declines is ONE
obligation's**, including the last: an unmappable assert costs its own
escalation and never its siblings' (audit 0.2.0 M17, where a single nested
``assert_`` declined escalation for every obligation in the query).
Declines never raise; :exc:`ReplayError` is
raised only by the replay evaluator, whose caller treats it as an
emission-infidelity signal — except for its :exc:`ReplayDeclined`
subclass, which says the replay itself cannot evaluate the point exactly
and degrades the witness to UNKNOWN instead of accusing the emission.

The index bookkeeping (element pairing, structural routing, reduction
grouping, and the ``dot_general`` contraction geometry) is computed by ONE
set of helpers, driven through the very :mod:`stelling.interval` functions
the propagation transfers use, and is shared by slice validation, the SMT
emission, and the replay — three consumers, one routing, so they cannot
disagree with each other. The ``dot_general`` row joined that discipline
late and the gap was a soundness defect while it lasted: it re-derived its
geometry from the LHS alone, so an equation whose contracted extents
disagreed made :func:`stelling.interval.dot_general` RAISE while this
module emitted a silently TRUNCATED sum (audit 0.2.0 S12). It now asks
:func:`stelling.interval.dot_general_geometry`, which is that transfer's
own well-formedness.

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
from stelling.coverage import DEFAULT_TRANSPARENT, call_body
from stelling.propagate import (
    _EXACT_CONVERSIONS,
    _INT_DTYPE_BOUNDS,
    TRANSFERS,
    Propagation,
    RelationalAssume,
)

__all__ = [
    "DeclinedObligation",
    "ELEMENT_BUDGET",
    "ObligationSlice",
    "ReplayDeclined",
    "ReplayError",
    "SliceAssume",
    "SliceInput",
    "evaluate_predicate",
    "fraction_text",
    "slice_obligation",
    "slice_unknown_obligations",
    "violating_elements",
    "witness_is_valid",
]

QF_LRA = "QF_LRA"
QF_NRA = "QF_NRA"

# The comparison primitives a forwarded relational assume can be emitted as,
# with their SMT-LIB2 spellings. ONE definition, read by the slicer (which
# decides what may be carried into a slice) and by the emission (which writes
# it): a `SliceAssume` therefore cannot exist for a comparison the emission
# has no rule for, and the emission's "unsupported comparison" skip — one of
# the five silent `continue`s this fix removed — has no reachable case left.
ASSUME_CMP_SYM = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "="}

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

# THE TWO POW CAPS BOUND ONE QUANTITY: the DEGREE of a polynomial the
# `pow`/`integer_pow` rows write into the emitted script. Nothing else is
# being gated here — not the exponent's magnitude, not the operand count —
# and stating the quantity once is the point: the predecessor capped
# `integer_pow`'s |y| and the rational branch's DENOMINATOR while leaving
# its NUMERATOR unbounded, so `x ** 100.5` (an emitted degree-201 product)
# was admitted while the strictly smaller `x ** 100` was declined at 64,
# and `x ** 1000000000000.5` built a 600 KB script from a one-line harness
# before dying of MemoryError (audit 0.2.0 M7). The two caps still differ,
# and the reason is the construct each bounds, written out below.

# Degree of a product expanded INLINE into a `define-fun` body:
# `integer_pow(x, y)` and `pow(x, y)` at integer y both write
# `(* x x … x)` with |y| factors. Beyond this the script stops being
# auditable by eye and the emission declines instead. This form declares
# no auxiliary variable, so the z3 tactic workaround in
# :mod:`stelling.solvers` — which keys on `aux_` declarations — does not
# fire on it, and the default solver path is what has to carry it.
INTEGER_POW_EXPANSION_CAP = 64

# Degree of the auxiliary-variable EQUATION a non-integer exponent emits:
# `pow` at exponent p/q declares a fresh `aux` and asserts
# `aux^q = x^p`, whose degree is max(p, q) — BOTH sides, which is why one
# cap governs numerator and denominator alike. Beyond this the polynomial
# risks solver timeout, so the obligation declines. 128 rather than the
# expansion cap's 64: measurements showed both solvers handle every
# denominator up to 100 AT NUMERATOR 1 in <1 s on this construct (cvc5
# natively, z3 via the custom tactic chain in solvers.py that fires on
# scripts containing `aux_` declarations, which every emission of this
# shape has). The numerator qualifier is load-bearing and is the subject
# of the next paragraph. Whether the inline expansion could afford the
# same 128 is unmeasured, and raising it would ADMIT harnesses that
# decline today — the direction that needs the measurement, not the
# direction that needs a comment.
#
# THE p=1 FAMILY IS THE EASY ONE, so the sweep above says nothing about
# the worst admitted case. The admitted set is small enough to measure
# whole, and it was: every admissible exponent has q a power of two,
# since the exact value of a binary64 is dyadic
# (:func:`pow_exponent_rational`), so the reachable denominators are
# exactly 2, 4, 8, 16, 32, 64, 128, the numerators are the odd values
# below 128, and the admitted set is 7 x 64 = 448 pairs. All 448 were
# run (2026-08-14) as
#
#     x = any_array((), "float64", (1.0, 100.0))
#     assert_(x ** (p / q) >= 1.0)
#
# at `solver_timeout_ms=60000` on the default portfolio. Every one of the
# 448 was decided BY A SOLVER on the obligation itself — `unsat` in
# QF_NRA from both wheels, never discharged by interval propagation and
# never timed out (worst single backend 53.6 s) — so these are solve
# costs and not the cost of some easier query. Portfolio wall, the two
# backends in sequence:
#
#     x**(1/128)     0.41 s  (cvc5  0.06 + z3  0.30)  the cheap corner
#     x**(127/128)  49.25 s  (cvc5 30.13 + z3 19.11)  both sides at cap
#     x**(105/128)  73.47 s  (cvc5 19.82 + z3 53.64)  THE WORST OF 448
#     x**(127/2)     0.11 s  (cvc5  0.07 + z3  0.03)  degree 127, one
#                                                     below 127/128's
#
# Read off that: cost tracks the DENOMINATOR, not the degree max(p, q)
# the cap bounds — `x**63.5` is degree 127 and costs 0.11 s — and within
# the q=128 row it is not monotone in the numerator either (p=105 is
# 73 s, p=113 is 41 s, p=127 is 49 s). So "both sides near the cap" was
# the wrong guess at the worst case, by 49%, and no closed form is
# offered here in place of the sweep. The per-row worst rises 0.9, 6.5,
# 6.5, 7.6, 9.0, 19.1, 73.5 s for q = 2 … 128. Seconds are
# machine-specific (24-core x86-64, cvc5 1.3.4 wheel, z3 5.0.0 wheel);
# the ordering is the content.
#
# Every one of the 448 is a solve that FINISHES, bounded by the caller's
# own `solver_timeout_ms`, and a timeout is never a VERIFIED; none is a
# pathological script of the kind this cap exists to refuse.
RATIONAL_POW_DEGREE_CAP = 128

_FLOAT_INPUT_DTYPES = frozenset({"float16", "float32", "float64"})

# scalar decoders for size-1 ir.Array literals/consts (numpy dtype .str)
_SCALAR_STRUCT_FMT = {
    "<f8": "d", "<f4": "f", "<f2": "e",
    "<i8": "q", "<i4": "i", "<i2": "h", "|i1": "b",
    "<u8": "Q", "<u4": "I", "<u2": "H", "|u1": "B",
    "|b1": "?",
}

_ARITH = frozenset(
    {"add", "sub", "mul", "neg", "div", "integer_pow", "pow", "max", "min",
     # `square` is jax's own primitive on this series, NOT sugar for
     # integer_pow(y=2) — `jnp.square(x)` binds `square_p` and a slice that
     # traverses one had no emission at all, so a property genuinely false
     # over the declared box declined instead of producing a witness. It
     # emits as the SELF-PRODUCT of one term (:func:`stelling.smt._square_body`),
     # which is what the primitive is: one operand, so the two factors are
     # the SAME SMT constant and the solver sees the correlation an
     # interval cannot. Same class as `mul` for every guard that follows.
     "square"}
)
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
       "dot_general", "scatter", "is_finite"}
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
    {"add", "sub", "mul", "neg", "div", "integer_pow", "pow", "reduce_sum",
     # `square` multiplies its operand by itself, so it computes a value the
     # operand did not contain and can leave an integer dtype's range —
     # `mul`'s class exactly, and the class `integer_pow` was found in
     # (audit UNSOUND 2). The transfer already classifies it this way
     # (propagate._INT_COMPUTING); the emission must not be laxer than the
     # transfer about a defect the transfer catches.
     "square",
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
    | {"max", "min", "select_n", "convert_element_type", "scatter",
       "is_finite"}
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
    "is_finite": (
        "emits constant true: under real semantics all declared variables "
        "are finite rationals by construction, so is_finite is a tautology "
        "over the emitted sort. Result sort Bool — no numeric value created"
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


class ReplayDeclined(ReplayError):
    """The replay REFUSES this point: the exact value is not a rational,
    so no amount of care in this evaluator can decide the predicate there.

    A subclass, so every ``except ReplayError`` handler still catches it —
    but the dispatch layer separates the two, and the separation is the
    whole point. ``ReplayError`` means *the emitted problem does not mean
    the obligation*: an alarm about the EMISSION, raised loudly, because
    a solver model that escapes the box or satisfies the predicate is
    evidence the script was wrong. ``ReplayDeclined`` means *the replay
    cannot keep up with a correct emission*, which is a fact about the
    replay, not about the script, and it costs a REFUTED rather than
    earning an exception.

    Before the separation, ``x ** 0.5`` over a box starting just above
    ``4.0`` made the public ``check()`` RAISE: cvc5's model
    ``4 + 2^-108`` is a real violation (``sqrt`` of it exceeds 2 in the
    reals) and the emission ``aux^2 = x0, aux >= 0`` is exactly right,
    but the replay evaluated ``float(w) ** 0.5 == 2.0`` and reported the
    predicate TRUE — so the one alarm that means "the emission is wrong"
    fired at a correct emission and named the wrong culprit
    (audit 0.2.0 S3). The dispatch layer now routes this to UNKNOWN with
    ``witness not independently replayable``, the posture it already had
    for a model containing a non-rational value.
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
class SliceAssume:
    """One relational ``assume`` CARRIED INTO THIS SLICE'S ID NAMESPACE.

    THE TYPE IS THE FIX (audit 0.2.0 S5-B2). A
    :class:`stelling.propagate.RelationalAssume` names its operands with the
    ids of the scope it was traced in — inside a ``jit`` body those are that
    body's ids, and the slicer renumbers every inlined body. Resolving them
    with a bare integer lookup against the slice's table is what emitted an
    axiom about unrelated values, measured as the CONVERSE of the user's own
    precondition.

    A ``SliceAssume`` exists only where that translation has already happened
    and succeeded: ``invars`` are atoms of THIS slice, alias-resolved exactly
    as :meth:`_Slicer._rewrite` resolves an equation's operands, and each one
    is known to have a term in the emitted script. An assume that could not be
    translated is not a ``SliceAssume`` at all — it is a quoted reason in
    :attr:`ObligationSlice.assumes_skipped`. There is therefore no value of
    this type that names a term its operands do not denote, and no parameter
    anywhere through which an untranslated assume can reach emission.

    ``pairs`` is the elementwise pairing — ``(lhs element, rhs element)`` per
    element of the comparison's broadcast output — computed HERE, from the
    resolved operands' own avals, by the same
    :func:`_pair_elementwise` the slice's equations go through. The emission
    indexes term tuples with it and never re-derives it: the arity crash and
    the silent truncation (audit M6, id-collision findings 3 and 4) were both
    the emission deriving indices from one pair of shapes and applying them to
    terms of another.
    """

    primitive: str  # lt | le | gt | ge | eq
    invars: tuple[ir.Atom, ...]  # SLICE-scope atoms, alias-resolved
    pairs: tuple[tuple[int, int], ...]  # (lhs element, rhs element) per output
    source_info: tuple[str, ...] = ()
    # WHICH forwarded assume this is: the index of the
    # :class:`stelling.propagate.RelationalAssume` it was translated from, in
    # the tuple the slicer was handed. It is the IDENTITY a downstream join is
    # keyed on — `stelling.propagate.unaccounted_assumes` asks whether the
    # assume that produced ledger entry *k* is among the ones a given script
    # emitted, and no count can answer that. `-1` is the "did not come from a
    # propagation" value a hand-built SliceAssume carries, and it matches no
    # ledger entry, so a hand-built slice can never release a withheld
    # violation.
    origin: int = -1


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
    # The relational assumes this slice CAN state, already in this slice's own
    # id namespace, and the quoted reason for every one it cannot. The two are
    # a partition of the assumes the slicer was handed:
    #
    #     len(assumes) + len(assumes_skipped) == len(relational_assumes given)
    #
    # so a caller holding the slice can derive "emitted versus requested"
    # without consulting the propagation, and every shortfall has a sentence
    # attached to it. Before this pair existed the emission had five silent
    # `continue`s and a run could lose a user-stated constraint with nothing
    # said about it.
    #
    # A COUNT OF THIS PARTITION IS NOT ENOUGH FOR THE WITHHOLDING RULE, which
    # is why every carried assume also names WHICH forwarded assume it is
    # (:attr:`SliceAssume.origin`). "How many of them arrived" cannot answer
    # "did the one that constrains this witness arrive"; two rounds of false
    # REFUTED came out of asking the first question and reading the answer as
    # the second (audit 0.2.0 S6, and the branch-scoping regression that
    # followed it).
    assumes: tuple[SliceAssume, ...] = ()
    assumes_skipped: tuple[str, ...] = ()


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


def _extents(shape) -> tuple[str | None, tuple[int, ...]]:
    """A static shape's extents NORMALISED to plain ``int``, paired with
    the uninhabited/malformed-shape problem that stopped the
    normalisation — ``(None, extents)`` when there is none.

    Two measured predicates (fix re-attacks R1/N1/N2): every extent must
    be INTEGRAL (from_dict does not coerce shape entries — a string
    extent made `d < 0` raise raw, and `1 * "x"` is silent garbage in a
    size product) and NONNEGATIVE (jax rejects negative extents in every
    concrete context: the type is uninhabited). Zero extents are legal.

    **IT RETURNS A VALUE AND NOT ONLY A VERDICT, AND THAT IS WHY IT
    EXISTS** — audit 0.2.0 B6 audit 3, F1. Its first spelling was
    :func:`_shape_problem` alone: it bound ``k = _op_index(d)``, tested
    ``k``, and DISCARDED it, so :meth:`_Slicer._declared_shape` re-read
    every extent with a second ``_op_index(d)`` and returned THAT — two
    reads per extent, and the returned one was the unvalidated one. An
    object whose ``__index__`` answers ``4`` and then ``-1`` was validated
    at ``4`` and emitted as ``(-1,)``, where ``_size`` takes a NEGATIVE
    contribution to the element budget and ``range(-1)`` mints no symbols
    at all. A guard that tests a value nobody keeps has not guarded the
    value the emission uses. This is the identical defect this same batch
    fixed one module over in
    :func:`stelling.interval.dot_general_geometry`'s ``_indices``, and the
    repair is the identical one: read once, and hand the caller what was
    read.

    **NOT ONLY** ``TypeError`` — audit 0.2.0 B6 audit 3, F2.
    ``operator.index`` raises whatever ``__index__`` raises, so a
    ``ValueError`` or an ``OverflowError`` from a hostile extent escaped
    this predicate raw and reached the caller as *"internal error"*
    through :func:`slice_obligation`'s net. An extent that will not answer
    ``__index__`` is a non-integer extent whatever it raises saying so,
    and the extent is quoted through :func:`_safely` because an object
    that refuses ``__index__`` may refuse ``__repr__`` too.
    """
    out: list[int] = []
    for d in shape:
        try:
            k = _op_index(d)
        except Exception:  # noqa: BLE001 — unreadable IS the finding
            return (
                f"a non-integer extent "
                f"{_safely(lambda: repr(d), '<unreadable>')} (malformed IR)",
                (),
            )
        if k < 0:
            return (
                "a negative extent (no jax program constructs such a value)",
                (),
            )
        out.append(k)
    return None, tuple(out)


def _shape_problem(shape) -> str | None:
    """:func:`_extents`' problem alone, for the readers that only need the
    VERDICT — an aval's shape they are about to compare, not a param they
    are about to mint terms from.

    WHAT THIS DOES NOT SAY, because the sentence that used to stand here
    said it and it was not true of the code below (audit 0.2.0 B6 audit 4,
    F3): *"every reader that needs the COUNT calls `_extents` and binds
    what it returns"*. The four readers this docstring was written for did
    not, and neither did the others: a census of :func:`_size`'s call
    sites at ``30d4b04`` finds **14 whose argument is a shape read
    straight off an `ir.Aval` or an `ir.Array` at the call site** — the
    divisor probe, the term-count pass, four in the element budget, two
    on the replay path, and more at one remove through a local. Listing
    them is the same defect one level up, so the repair is not a list of
    call sites. It is :func:`_size`: an element count is now the product
    of extents read through the SAME ``__index__`` a guard validates with,
    so no caller anywhere can obtain a count from a third protocol.

    What remains true only of the named readers, and is therefore said
    here rather than globally: :func:`_decode_scalar`,
    :func:`_decode_elements`, :meth:`_Slicer.slice`'s root and its
    constvar pass BIND what :func:`_extents` returned and count that, so
    they read each shape ONCE. Every other :func:`_size` caller reads a
    second time. That second read is safe against a third protocol and is
    NOT safe against an object that answers ``__index__`` differently
    between reads.

    **WHAT CONTAINS THAT IS THE CONSTRUCTOR, AND THIS USED TO SAY THE
    HASH** — audit 0.2.0 B6 audit 5, F1. The sentence here was *"what
    contains that is ``ClosedJaxpr.content_hash()``, which cannot encode
    such a param"*, and it was true of ``__index__`` for a reason it did
    not give and false one protocol over.

    For ``__index__``: an ``int`` SUBCLASS cannot answer differently
    between reads at all, because ``operator.index`` short-circuits on a
    real ``int`` and never calls ``__index__`` (measured: zero calls, the
    stored value returned) — so the guard, ``ir._encode`` and
    ``json.dumps`` all read the same number. A two-faced ``__index__``
    therefore needs a NON-``int`` class, and ``ir._encode`` refuses to
    encode one at all. The containment was `_encode`'s TYPE closure, not
    an inability to encode a drifting answer.

    One protocol over, the claim is simply false. ``ir._encode`` iterates
    a ``shape`` param ONCE and encodes what that read returned, so a
    ``tuple`` SUBCLASS whose ``__iter__`` answered ``(4,)`` once and
    ``()`` afterwards hashed cleanly at ``321209d`` — measured at every
    flip point — and the same trick one element narrower minted a VERIFIED
    on a claim exact arithmetic falsifies.

    The containment is that ``ir.Aval``, ``ir.Array`` and a declaration's
    ``shape`` param now CARRY the extents their own ``__post_init__``
    validated, as plain ``int`` in a plain ``tuple``. A shape this
    function or :func:`_size` reads off one of those objects is
    single-valued before it arrives, whatever protocol is asked of it, and
    a shape read off anything else is not covered by that and is listed in
    :func:`_size`. See :meth:`_Slicer._declared_shape`, where the same
    correction is recorded."""
    return _extents(shape)[0]


def _size(shape) -> int:
    """The element count of ``shape``, read through ``__index__``.

    **NOT ``for d in shape: n *= d`` OVER THE RAW OBJECTS — audit 0.2.0 B6
    audit 4, F3.** That spelling reached ``__mul__``/``__rmul__``, a THIRD
    protocol beside the ``__index__`` every guard in this module validates
    with and the ``__eq__`` the shape comparisons use, so an extent could
    be validated at 2, compared equal to 2, and counted as 1 — measured:
    ``operator.index(d) == 2`` and ``_shape_problem((d,)) is None`` while
    ``_size((d,)) == 1``. The audit named FOUR readers that took the
    predicate face of :func:`_extents` and then counted with a raw second
    read, and could not drive any of them to a false verdict: the constvar
    route is closed earlier by the ``ir`` door and the
    :func:`_decode_elements` route by the byte-length check. **That
    containment was accidental**, and the four were not the count either —
    at ``30d4b04`` fourteen call sites handed this function a shape read
    straight off an ``ir.Aval`` or an ``ir.Array``, some with no
    validation in front of them at all. Fixing the sites someone
    enumerated is the defect one level up; fixing the function makes a
    count and the guard that validated it come from one protocol whatever
    the caller did first.

    A malformed extent is a DECLINE and not a number: an element count
    that cannot be read is not zero, not one, and not the caller's
    problem to notice. Netted to a `DeclinedObligation` everywhere the
    slicer drives it.

    **WHERE IT IS NOT NETTED — THREE SITES, and this paragraph named two**
    (audit 0.2.0 B6 audit 5, F5). Making a total function partial gives
    every caller a channel to answer for, so the census is here rather
    than in whichever caller someone remembered:

    * :func:`_root_elements` — CORRECT and unchanged. Its per-equation
      ``_size`` and routing calls are inside a ``try`` that converts
      :exc:`_Decline` to :exc:`ReplayError`, which is the replay path's
      own channel. Its input-size loop runs BEFORE that ``try`` and is
      the one place the old argument was right about: ``inp.shape`` is a
      :attr:`SliceInput.shape`, which IS an :func:`_extents` result and
      not a re-read of anything.
    * :func:`violating_elements` — its ``_size(sl.root.aval.shape)`` runs
      after :func:`_root_elements` returns, outside any net, and would
      reach :mod:`stelling.solvers`' generic handler as *"escalation
      attempted; internal error"*.
    * :func:`_index_box` — ``range(_size(shape))``, reached from
      :func:`_pair_elementwise` and :func:`_route_structural`. The
      DECLINE is new at ``321209d``: at ``30d4b04`` ``_size`` was a raw
      product, so the same shape came out of this helper as a bare
      ``TypeError: unsupported operand type(s) for *=`` — measured, and
      the reason the change is a narrowing of the escape and not the
      creation of one. Both routing helpers are driven AGAIN by
      :func:`stelling.smt.emit`, after :func:`slice_obligation` has
      returned, and `smt.py` nets no :exc:`_Decline` at all — so a
      decline there does not become a :class:`DeclinedObligation`; it
      lands in :mod:`stelling.solvers`' generic handler as *"escalation
      attempted; internal error"*. :mod:`stelling.affine` drives the same
      two helpers and DOES net them (three sites), which is how the
      asymmetry between the two consumers is visible at all.

      DRIVEN, not deduced: sweeping the read at which a hostile extent
      starts refusing, over one declaration query, ``321209d`` produced
      *"escalation attempted; internal error: `_Decline`"* from this
      helper at two points in the sweep and a `ReplayError` at a third.
      After audit 5's F1 the SAME sweep produces no decline at any point,
      because :meth:`ir.Aval.__post_init__` freezes the extent at the read
      it validated and this helper's argument is a :func:`_shape_of` of
      that. The site is recorded rather than deleted because that
      containment is about where the shape COMES FROM, not about this
      helper.

    **AND THE ARGUMENT FOR WHY IT CANNOT FIRE WAS TWO ARGUMENTS.** This
    paragraph said the replay shapes *"come from an `ObligationSlice`
    whose extents `_Slicer._validate` and `_Slicer._declared_shape`
    already normalised"*. That is true of :attr:`SliceInput.shape`, which
    IS an :func:`_extents` result. It was never true of
    ``sl.root.aval.shape`` or of ``_shape_of(eqn.outvars[0])``: those were
    fresh reads of a raw ``ir.Aval`` field, and normalising a DIFFERENT
    object earlier does not make a later read of this one safe. Since
    audit 5's F1 the containment there is real rather than incidental, and
    it is a different mechanism with a different name:
    :meth:`ir.Aval.__post_init__` INSTALLS the extents it validated, so
    ``aval.shape`` is a plain ``tuple`` of plain ``int`` and every re-read
    of it — here, in :func:`_shape_of`, in :func:`_index_box` — returns
    the same ints. What is NOT covered by that is a shape reaching these
    callers from somewhere other than an ``ir.Aval``, an ``ir.Array`` or
    an :func:`_extents` result; the three sites above are recorded so that
    such a caller has somewhere to be checked against rather than being
    discovered by an *"internal error"* in a verdict."""
    problem, extents = _extents(shape)
    if problem is not None:
        raise _Decline(
            f"a shape with {problem} has no element count (malformed IR)"
        )
    n = 1
    for d in extents:
        n *= d
    return n


def _decode_scalar(val):
    """A size-1 literal/const value -> exact python bool | int | float, or
    decline. (The emission's single-element decoder; array constants go
    through :func:`_decode_elements`.)"""
    if isinstance(val, ir.Array):
        # ONE READ: `_extents` validates and hands back what it validated,
        # and the count below is that (audit 0.2.0 B6 audit 4, F3 — this
        # reader had no validation at all and counted the raw objects)
        problem, extents = _extents(val.shape)
        if problem is not None:
            raise _Decline(
                f"array-shaped constant of shape "
                f"{_safely(lambda: repr(tuple(val.shape)), '<unreadable>')} "
                f"has {problem}"
            )
        if _size(extents) != 1:
            raise _Decline(
                f"array-shaped constant of shape {extents} where a single "
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
        # ONE READ, and the count below is bound from it — audit 0.2.0 B6
        # audit 4, F3. This called `_shape_problem(val.shape)` and then
        # `_size(val.shape)`: the extents it validated and the extents it
        # counted were two different reads of the same self-describing
        # objects.
        problem, extents = _extents(val.shape)
        if problem is not None:
            # fix-re-attack R1/N2: a negative or non-integer element
            # count would reach struct.unpack as a malformed format and
            # raise struct.error raw — decline quoted instead
            raise _Decline(
                f"array-shaped constant of shape "
                f"{_safely(lambda: repr(tuple(val.shape)), '<unreadable>')} "
                f"has {problem}"
            )
        n = _size(extents)
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
                f"array-shaped constant of shape {extents} dtype "
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


def pow_exponent_rational(exp_float: float) -> Fraction:
    """The EXACT real the traced ``pow`` exponent literal denotes.

    ONE DERIVATION, TWO READERS: the admission guard
    (:meth:`_Slicer._validate`) and the emission
    (:func:`stelling.smt.emit`) both call this, so the rational the guard
    admitted and the rational the script is written about cannot drift
    apart. Both take :class:`fractions.Fraction` of the *binary64 value*,
    which is exact and finite for every finite double — a binary64 IS a
    dyadic rational, and ``Fraction(0.1)`` is
    ``3602879701896397/36028797018963968``, not ``1/10``.

    **The predecessor rationalised with ``limit_denominator(128)`` and
    admitted the result whenever ``abs(float(frac) - exp_float) <= 1e-12``
    (audit 0.2.0 S1).** That test cannot see the substitution it exists to
    detect: ``float(Fraction(1, 10))`` rounds back to the same binary64,
    so the measured error for ``0.1`` was exactly ``0.0`` while the two
    rationals differ by ``5.55e-18`` — and the emitted problem was about
    ``x^(1/10)``, a DIFFERENT function from the ``x^0.1`` the program
    computes. Since an ``unsat`` is a universal claim with nothing
    downstream to re-derive it, that minted false VERIFIEDs. No threshold
    on a binary64 distance can fix it; the comparison has to be between
    rationals, and taking the exact one removes the comparison entirely.
    """
    return Fraction(exp_float)


def rational_pow_problem(exp_float: float) -> str | None:
    """``None`` when a NON-INTEGER ``pow`` exponent is emittable through
    the auxiliary-variable encoding, else the reason it is not — written
    so a reader can act on it.

    Emittable means: the exact rational
    (:func:`pow_exponent_rational`) is non-negative and the degree of the
    equation ``aux^q = x^p`` stays within
    :data:`RATIONAL_POW_DEGREE_CAP`. Nothing here approximates: an
    exponent that is not a dyadic rational of small degree DECLINES to
    UNKNOWN rather than being analysed as a nearby rational, because a
    nearby rational is a different function.
    """
    if not math.isfinite(exp_float):
        return (
            f"'pow' exponent {exp_float!r} is not finite, so it denotes no "
            f"real and has no exact rational encoding"
        )
    frac = pow_exponent_rational(exp_float)
    p, q = frac.numerator, frac.denominator
    if p < 0:
        return (
            f"'pow' with negative rational exponent {frac}: negative "
            f"rational exponents are not supported"
        )
    degree = max(p, q)
    if degree <= RATIONAL_POW_DEGREE_CAP:
        return None
    if q <= RATIONAL_POW_DEGREE_CAP:
        # The exponent IS a small dyadic; only the emitted polynomial's
        # size is the problem, and nothing is being approximated. Say so,
        # rather than borrowing the wording of the other branch.
        return (
            f"'pow' exponent {exp_float!r} is exactly {p}/{q}, and the "
            f"auxiliary encoding aux^{q} = x^{p} would be a degree-"
            f"{degree} polynomial — over the emission cap "
            f"{RATIONAL_POW_DEGREE_CAP} "
            f"(stelling.obligation.RATIONAL_POW_DEGREE_CAP), which bounds "
            f"BOTH sides of that equation, so a large numerator declines "
            f"exactly as a large denominator does. Escalation declines "
            f"rather than emit it; nothing about the exponent is "
            f"approximated"
        )
    # The literal is a binary64 and therefore IS an exact dyadic rational —
    # say which one, and say that the low-denominator rational the reader
    # has in mind is a different number. A message reading "cannot be
    # represented as p/q with q <= 128" would be FALSE: it can, exactly,
    # and the denominator is a power of two.
    near = Fraction(p, q).limit_denominator(RATIONAL_POW_DEGREE_CAP)
    return (
        f"'pow' exponent {exp_float!r} denotes exactly {p}/{q} — a binary64 "
        f"literal IS a dyadic rational, so that is the exponent's value and "
        f"not an approximation of it — and its auxiliary encoding "
        f"aux^q = x^p would be degree {degree}, over the emission cap "
        f"{RATIONAL_POW_DEGREE_CAP} "
        f"(stelling.obligation.RATIONAL_POW_DEGREE_CAP). Emitting about the "
        f"nearby rational {near} instead would be analysing a DIFFERENT "
        f"function from the one the program computes — x^({near}) and "
        f"x^{exp_float!r} do not agree, and a discharge has nothing "
        f"downstream to catch the difference — so this declines to UNKNOWN. "
        f"Exponents whose exact binary64 value is a small dyadic rational "
        f"(0.5, 0.25, 0.125, 1.5, 1/128) do escalate"
    )


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


def _render_scope(path: tuple) -> str:
    """A scope path in words, for a decline the reader can act on.

    The path is positional (:data:`stelling.propagate.ScopePath`), so this is
    the only place it becomes prose; nothing reads it back.

    THE ``cond`` ARM IS UNREACHABLE FROM THE ONLY CALLER, and says so here
    rather than reading as a live case. The one caller is
    :meth:`_Slicer._carry_assumes`, which renders the scope of a
    :class:`stelling.propagate.RelationalAssume`; and the propagator increments
    ``branch_depth`` BEFORE it pushes any ``("cond", pos, i)`` step, while its
    forwarding guard refuses to forward at all while ``branch_depth`` is
    nonzero. So no ``RelationalAssume`` can carry a cond step, and this arm
    cannot fire through that path. It is kept because the function is a
    renderer of a general type: a path is data, a caller that hands it one
    with a cond step in it should get prose rather than a ``repr``, and the
    ``else`` beneath it exists for exactly the same reason.
    """
    if not path:
        return "the top level"
    parts = []
    for step in path:
        if len(step) == 2 and step[0] == "call":
            parts.append(f"the body of equation {step[1]}")
        elif len(step) == 3 and step[0] == "cond":  # unreachable; see docstring
            parts.append(f"branch {step[2]} of the cond at equation {step[1]}")
        else:  # an unrecognised step is quoted, never guessed at
            parts.append(repr(step))
    return " → ".join(parts)


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
    ``jnp.sum`` of a size-0 array is 0.0).

    ``in_shape`` COMES FROM THE RECORDED AVAL, and that is load-bearing:
    audit 0.2.0 S12′ is this line and its ``dot_general`` twin. The
    interval transfer sums the operand's propagated BOX, whose element
    count is the real one; this function sums ``_size(in_shape)`` addends,
    whose count is whatever the IR claims. Shrink only the aval and the
    two legs sum different arrays — the transfer reporting ``[4, 8]`` while
    the emission proves a claim about a two-element sum. The agreement is
    enforced once for every primitive at
    :meth:`_Slicer._one_shape_per_value`, which runs before this function
    is reached; nothing about it is specific to ``reduce_sum``, which is
    exactly why it is not written here."""
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

    Admissibility is decided by TWO shared oracles, complementary and
    disjoint — and by ONE cross-check that is neither, named after them:

    * :func:`stelling.propagate._dot_general_row_form` — the PARAMS and
      DTYPES. Built shared from the start rather than extracted later,
      which is how the scatter row acquired a gate the transfer had and the
      emission did not. It never sees a shape.
    * :func:`stelling.interval.dot_general_geometry` — the SHAPES: dim
      ranges, duplicate dims, list pairing, batch- and contracted-EXTENT
      agreement, and the derived ``out_shape`` and contraction ranges. It
      never sees a param.

    * :meth:`_Slicer._one_shape_per_value` — WHICH SHAPES the oracle above
      is handed. Not an oracle: it is a cross-check, it runs over every
      equation of the slice rather than this row, and it runs before this
      function is called at all. It is listed here because without it the
      two bullets above do not compose into what they look like they say.

    THE SECOND ONE IS AUDIT 0.2.0 S12 AND IT WAS LEARNED THE HARD WAY. The
    extent-agreement check used to live inline in
    :func:`stelling.interval.dot_general` while this function re-derived the
    geometry from the LHS alone, so on ``lhs=(2,) @ rhs=(4,)`` the transfer
    RAISED and this face returned a two-term combination — two of the
    constant operand's four addends DROPPED, no decline — and on
    ``lhs=(4,) @ rhs=(2,)`` it indexed off the end of the constant operand
    and raised a raw ``IndexError`` out of a slicer that catches only
    ``_Decline``. The two faces disagreed about whether the equation was
    well-formed and the disagreement resolved in the ASSERTING direction.
    Neither face owns a shape predicate now; both ask the oracle.

    THE THIRD ONE IS AUDIT 0.2.0 S12′, AND IT IS WHY "both ask the oracle"
    was not enough. The oracle is shared and its ARGUMENTS are not: the
    transfer asks about ``a.shape``/``b.shape``, the shapes of the boxes
    that actually flowed in, and this function asks about
    ``_shape_of(eqn.invars[i])``, the shapes the IR claims. Edit only those
    avals — ``from_dict`` accepts it — and the transfer computes the true
    four-term box while this function plans two, with no refusal on either
    side to notice. Measured on ``4d793cf``: VERIFIED at 100% coverage on a
    claim whose truth is ``8 <= 4.5``, plus a REFUTED whose witness the
    verdict called "confirmed by independent exact-rational replay" —
    honest about the arithmetic, false about the plan, since replay
    re-derives this same plan.

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

    try:
        geom = iv.dot_general_geometry(lhs_shape, rhs_shape, dn)
    except iv.IntervalError as e:
        # The SHAPE oracle's refusal, quoted — the propagation face raises
        # the identical sentence from the identical function.
        #
        # WHAT THAT ESTABLISHES, CORRECTED (audit 0.2.0 S12′). This comment
        # used to end "...so an equation one face refuses cannot be planned
        # by the other (S12)", and that was false as written: the oracle is
        # shared and its ARGUMENTS are not. The transfer asks it about the
        # propagated BOXES; the line above asks it about the recorded AVALS.
        # A query whose avals lie about their operands' extents still splits
        # the two faces, in the asserting direction, and `reduce_sum`
        # carries the identical defect through `_group_reduce_sum` — a
        # class, not this row.
        #
        # The true statement is: GIVEN THE SAME SHAPES, an equation one face
        # refuses cannot be planned by the other. That the two faces ARE
        # given the same shapes is enforced separately, by
        # `_Slicer._one_shape_per_value`, which runs over every equation of
        # the slice before any plan here is built.
        raise _Decline(f"'dot_general' declined: {e}") from e
    lc, rc, lb, rb = geom.lc, geom.rc, geom.lb, geom.rb
    lfree, rfree, out_shape = geom.lfree, geom.rfree, geom.out_shape
    contracted_ranges = [range(n) for n in geom.contracted_extents]
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
        # THE ORACLE'S extents, not the LHS's. Reading `lhs_shape[i]` here is
        # exactly S12: on a shorter LHS it truncates the sum and on a shorter
        # RHS it indexes off the constant operand.
        for c in itertools.product(*contracted_ranges):
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


# ── THE GAUGED SPACE OF THE TWO SCATTER ROWS, AND WHY ADMISSION STOPS THERE ──
#
# A BOUNDED SWEEP IS BLIND JUST PAST ITS BOUND, AND THIS PROJECT HAS NOW
# WATCHED THAT COST IT FIVE TIMES. A route gauge sampling k = {0} then {0, 2};
# field probes by name then by type; an arity family widened 2 -> 3 and then
# 3 -> 8, where the escape sat at EXACTLY the declared ceiling; and the two
# route corruptions this pair of constants answers. Each time the reflex was
# to raise the bound, and each time the next corruption was written one step
# past the new one. Measured on `e35de13`, both line-neutral, both a
# `violated-witness` turned `discharged` — a MISSED violation, the direction
# the VERIFIED bar exists for — and both with the FULL SUITE GREEN in both
# columns (2044 / 2 and 2040 / 6):
#
#   * `i == (k if n != 9 else 0)` in the SET row. `_set_space` exhausts
#     1 <= n <= 8 and the surface sweeps n in 3..6, so n = 9 is gauged by
#     nothing: `x=(9,); s=x.at[2].set(u); assert s[0]-x[0] >= 1.0` went
#     `violated-witness` -> `discharged`, while n = 8 and n = 10 stayed honest.
#   * `groups[(k if operand_shape[0] < 4 else 0) * rowsz + t]` in the ADD row.
#     `_add_space`'s dims stop at 3, so it contains NO axis of length >= 4,
#     and the sweep's non-degeneracy clause was on `rowsz` — the TRAILING
#     product — which structurally cannot see a LEADING-axis-keyed corruption.
#     `(2,2)` and `(3,2)` stayed `violated-witness`; `(4,2)`, `(5,2)` and
#     `(6,2)` went `discharged`.
#
# So the answer is not a larger number. It is that THE ADMITTED SPACE MUST NOT
# EXCEED THE GAUGED SPACE: past these bounds the two rows DECLINE, so the
# routing code cannot execute on a shape nothing measured. A declined
# obligation does not slice, does not emit and is never solver-decided — it
# stays undischarged and the verdict is UNKNOWN — so this fails toward the
# same side every other decline in this module does. It is not free: it costs
# REFUTATIONS as well as discharges past the bound, which SOUNDNESS.md records
# rather than leaving as a silent narrowing.
#
# The sweeps that define these numbers are
# `tests/test_scatter_gauge_jax.py::_set_space` and `::_add_space`, and that
# file asserts its bounds EQUAL to the four below in both directions: widening
# admission without widening the sweep is red, and shrinking the sweep below
# admission is red. That is the whole mechanism — one identifiable place, so
# widening the rows is a deliberate act with a gauge attached.
#
# THE SIXTH INSTANCE WAS THE ADD ROW'S INDEX COLUMN LENGTH, and the paragraph
# that used to stand here named it and left it open: "`_add_space` sweeps a
# single written index, while `jax.ops.segment_sum` reaches an index column of
# 4 on an operand these bounds admit; that axis is gauged by the segment-sum
# mutation battery, which is a battery and not an exhaustive sweep, and nothing
# here narrows admission to it." Naming a residual is not closing it, and this
# one was then demonstrated. A census of `len(ks)` at this row across the whole
# suite, by instrumentation, at `e35de13` and at `3e107cf` alike:
#
#     |ks| reaching the row:  {1, 2, 3, 4, 6, 254, 255}
#
# — 5 is absent, and so is everything in 7..253. One line, line-neutral:
#
#     groups[k * rowsz + t].append((j if len(ks) - 5 else 0) * rowsz + t)
#
# On operand shape `(2,)`, which these bounds explicitly admit, driven through
# `escalate`: |ks| in {2,3,4,6,7,8} stayed `violated-witness` (honest) and
# |ks| = 5 went `discharged` — a MISSED violation, with the full suite green.
#
# So the index column is bounded the way the shape is: THE ADMITTED COLUMN
# SPACE IS THE GAUGED COLUMN SPACE, and it is the union of two families that
# are each swept EXHAUSTIVELY rather than sampled —
#
#   * a single written index, over every gauged SHAPE — the sweep that already
#     existed (`tests/test_scatter_gauge_jax.py::_add_space`);
#   * every column of length 1..`_ADD_ROW_GAUGED_MAX_COLUMN` on a RANK-1
#     operand (`::_add_column_space`, exhaustive over `range(n) ** length`,
#     checked against jax's own accumulation); and
#   * the SINGLE-ELEMENT operand at every column length up to
#     `_ADD_ROW_GAUGED_MAX_SINGLE_SEGMENT`. On a one-element operand every
#     index is forced to 0 and `rowsz` is 1, so there is exactly ONE column per
#     length and the length is the only free parameter — which is why it can be
#     exhausted to 255 at all. This is the family the per-obligation element
#     budget's own boundary gate rides on (`gate_budget_boundary`: 2n + 3 terms,
#     n = 254 slices at 511 <= 512 and n = 255 declines at 513), and the census
#     above says it is the ONLY thing in this repo that reaches the row with
#     |ks| > 6. It is accounted for rather than declined by accident.
#
# WHY THE COLUMN SWEEP STOPS AT RANK 1, which is an admission bound and not
# just a sweep bound: the column space is `n ** length`, so exhausting it over
# every gauged shape costs 12510 traces and 80 seconds — measured, against 3
# for the rank-1 family. The census is what makes the trade defensible rather
# than convenient: every |ks| > 1 that reaches this row anywhere in the suite
# is on a RANK-1 operand. So the row declines a multi-index column above rank
# 1, which nothing here reaches, instead of running the arithmetic on a space
# a 3-second sweep could not cover.
#
# WHAT THAT COSTS, stated rather than left to be discovered: a `segment_sum`
# accumulating several update ROWS onto a rank-2 operand — normal-matrix
# assembly, say — now declines, and so does any column longer than
# `_ADD_ROW_GAUGED_MAX_COLUMN` on a multi-element operand. The direction is
# the same one every other decline here fails in: the obligation is not
# sliced, not emitted and never solver-decided, so it comes back `unknown`
# rather than wrong. It costs REFUTATIONS as well as discharges, which
# SOUNDNESS.md records.
#
# The interval TRANSFER is deliberately untouched: it has its own row
# arithmetic and its own gauge, and bounding it here would turn a bounded
# emission into a ⊤ box for no measured reason.
_SET_ROW_GAUGED_MAX_LEN = 8
_ADD_ROW_GAUGED_MAX_RANK = 3
_ADD_ROW_GAUGED_MAX_DIM = 3
_ADD_ROW_GAUGED_MAX_SIZE = 12
_ADD_ROW_GAUGED_MAX_COLUMN = 6
_ADD_ROW_GAUGED_MAX_SINGLE_SEGMENT = 255


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
      the same case for the same reason;
    * an operand LONGER THAN THE GAUGE (:data:`_SET_ROW_GAUGED_MAX_LEN`). The
      routes below are exhaustively gauged against jax's own execution for
      every ``(n, k)`` up to that length and by nothing past it, and a
      line-neutral mis-route wrong ONLY at n = 9 was measured green through
      the whole suite. See the block comment above the constants.

    ``unique_indices`` and ``indices_are_sorted`` are deliberately NOT relied
    on: they are caller promises jax does not verify, and the covered form
    writes exactly one element, for which both are vacuous.
    """
    from stelling.propagate import (
        _scatter_index_dtype_covers,
        _scatter_set_row_form,
    )

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
    # Named before the general form failure so the REASON is legible. The
    # shared oracle rejects this too — the check belongs there, so both faces
    # get it — but a caller told only "outside the measured form" would look
    # at the shapes, which are identical to a plain `.set`, and learn nothing.
    if params.get("update_jaxpr") is not None or params.get("update_consts"):
        raise _Decline(
            "'scatter' carries a combiner (update_jaxpr): this is an "
            "`x.at[k].apply(f)`-shaped equation, not `x.at[k].set(v)`. It has "
            "the same dimension numbers, shapes, mode and static index as a "
            "set, and a DUMMY updates operand — the combiner is the only "
            "thing distinguishing them, so treating it as a set would model "
            "out[k] = <dummy> where the program computes out[k] = f(operand[k])"
        )
    indices_dtype = eqn.invars[1].aval.dtype
    # Named before the general form failure, for the same reason the combiner
    # is: the shapes and dimension numbers here are those of a legitimate
    # `.set`, so "outside the measured form" would send the reader to fields
    # that are exactly right. The shared oracle rejects this too and is the
    # authority; this only supplies the reason.
    if not _scatter_index_dtype_covers(indices_dtype, operand_shape[0]):
        raise _Decline(
            f"'scatter' index dtype {indices_dtype!r} cannot exactly represent "
            f"the operand's leading-axis bound {operand_shape[0] - 1}: XLA "
            f"computes the out-of-bounds bound in the INDEX element type, so "
            f"the range check it performs is not the one this row models and "
            f"in-range-looking updates are silently DROPPED (measured on jax "
            f"0.11.0: an int8 index column writes at operand length 128 and "
            f"drops at 129)"
        )
    if not _scatter_set_row_form(
        params, operand_shape, indices_shape, updates_shape, indices_dtype
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
    if operand_shape[0] > _SET_ROW_GAUGED_MAX_LEN:
        raise _Decline(
            f"'scatter' operand axis {operand_shape[0]} is outside the GAUGED "
            f"static-index set row space (1..{_SET_ROW_GAUGED_MAX_LEN}): the "
            f"route sweep exhausts every (length, written index) up to that "
            f"bound and measures nothing past it, and a route nothing gauged "
            f"is never guessed — a bounded sweep is blind just past its "
            f"bound, so admission stops where the gauge does"
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
    values, on out-of-range indices (mode-dependent drop/clamp, never
    guessed), on an operand SHAPE OUTSIDE THE GAUGE
    (:data:`_ADD_ROW_GAUGED_MAX_RANK` / ``_DIM`` / ``_SIZE``) — the group
    arithmetic below is gauged against jax's own accumulation over exactly
    that shape space, and a line-neutral corruption keyed on a LEADING axis
    of 4 or more was measured green through the whole suite — and on an INDEX
    COLUMN outside the gauge (:data:`_ADD_ROW_GAUGED_MAX_COLUMN` /
    ``_SINGLE_SEGMENT``), which is the same argument on the axis the shape
    bounds do not touch: a census of ``len(ks)`` at this row across the whole
    suite reaches ``{1, 2, 3, 4, 6, 254, 255}``, and a mis-route wrong ONLY at
    a column of 5 was likewise green. See the block comment above the
    constants.
    """
    from stelling.propagate import (
        _check_unique_indices_promise,
        _scatter_add_row_form,
        _scatter_index_dtype_covers,
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
    indices_dtype = eqn.invars[1].aval.dtype
    if not _scatter_index_dtype_covers(indices_dtype, operand_shape[0]):
        raise _Decline(
            f"'scatter-add' index dtype {indices_dtype!r} cannot exactly "
            f"represent the operand's leading-axis bound "
            f"{operand_shape[0] - 1}: XLA computes the out-of-bounds bound in "
            f"the INDEX element type, so the range check it performs is not "
            f"the one this row models and in-range-looking updates are "
            f"silently DROPPED rather than accumulated (measured on jax "
            f"0.11.0: an int8 index column accumulates at operand length 128 "
            f"and drops at 129)"
        )
    n = _scatter_add_row_form(
        eqn.params_dict(), operand_shape, indices_shape, updates_shape,
        indices_dtype,
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
    if (
        len(operand_shape) > _ADD_ROW_GAUGED_MAX_RANK
        or any(d > _ADD_ROW_GAUGED_MAX_DIM for d in operand_shape)
        or _size(operand_shape) > _ADD_ROW_GAUGED_MAX_SIZE
    ):
        raise _Decline(
            f"'scatter-add' operand shape {operand_shape} is outside the "
            f"GAUGED accumulate row space (rank at most "
            f"{_ADD_ROW_GAUGED_MAX_RANK}, every axis at most "
            f"{_ADD_ROW_GAUGED_MAX_DIM}, at most "
            f"{_ADD_ROW_GAUGED_MAX_SIZE} elements): the shape sweep exhausts "
            f"that space against jax's own accumulation and measures nothing "
            f"past it, and a row arithmetic nothing gauged is never guessed — "
            f"a bounded sweep is blind just past its bound, so admission "
            f"stops where the gauge does"
        )
    if not (
        len(ks) == 1
        or (len(operand_shape) == 1
            and len(ks) <= _ADD_ROW_GAUGED_MAX_COLUMN)
        or (_size(operand_shape) == 1
            and len(ks) <= _ADD_ROW_GAUGED_MAX_SINGLE_SEGMENT)
    ):
        raise _Decline(
            f"'scatter-add' index column of {len(ks)} element(s) on operand "
            f"{operand_shape} is outside the GAUGED accumulate column space "
            f"(one index on any gauged shape; at most "
            f"{_ADD_ROW_GAUGED_MAX_COLUMN} on a rank-1 operand; at most "
            f"{_ADD_ROW_GAUGED_MAX_SINGLE_SEGMENT} on a single-element "
            f"operand, where every index is forced to 0 and the length is the "
            f"only free parameter): the column sweep exhausts that space "
            f"against jax's own accumulation and measures nothing past it, "
            f"and a row arithmetic nothing gauged is never guessed — a "
            f"line-neutral mis-route wrong ONLY at a column of 5 was measured "
            f"green through the whole suite"
        )
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
        top_primitives: frozenset[str] | None = None,
        relational_assumes: tuple[RelationalAssume, ...] = (),
    ) -> None:
        # The relational assumes the propagation forwarded, each carrying the
        # SCOPE its operand ids belong to. They are translated into this
        # slice's namespace in :meth:`_carry_assumes`, per obligation, and the
        # untranslatable ones come out as quoted reasons — never as silence,
        # and never as an id resolved in a namespace it does not belong to.
        self.relational_assumes = tuple(relational_assumes)
        # scope path -> the rename this slicer applied when it inlined that
        # scope. `()` is the top level, whose ids the slicer never renames, so
        # its rename is the empty (identity) map. Written by `_flatten`, read
        # only by `_carry_assumes`.
        self._scope_remaps: dict[tuple, dict[int, int]] = {(): {}}
        # the primitives the interval RUN recorded as fallen-to-⊤ (from
        # coverage.unknown_primitives), or None when the caller has no run
        # record. MESSAGE WORDING ONLY (blinded-lens audit R2/R3): the
        # unsupported-primitive decline may describe the interval leg's
        # behaviour on this run only from this record — never from a
        # guess — and with None it claims neither direction. Admission is
        # never consulted from it.
        self.top_primitives = top_primitives
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

    def _resolve_for_guard(self, atom: ir.Atom) -> ir.Atom:
        """Find an atom whose id IS in self.env, for guard lookups.

        When the slicer inlines a transparent call, it rewrites equations
        to use renumbered inner var IDs. The guard (div-straddle, is_finite)
        needs the PROPAGATED interval, which was computed for the ORIGINAL
        outer var IDs. The alias system maps outer→inner (for emission),
        so we search REVERSE: find which outer var (in env) aliases TO
        an atom chain ending at this one.

        Falls back to the atom itself if no resolution is found (the env
        lookup will then succeed or fail on its own).
        """
        if not isinstance(atom, ir.Var):
            return atom
        if atom.id in self.env:
            return atom
        # Reverse search: find an outer var whose alias chain ends at atom
        for outer_id, target in self.aliases.items():
            if isinstance(target, ir.Var) and target.id == atom.id:
                if outer_id in self.env:
                    return ir.Var(id=outer_id, aval=atom.aval)
                # The outer might itself alias somewhere — recurse
                return self._resolve_for_guard(
                    ir.Var(id=outer_id, aval=atom.aval)
                )
        return atom

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

    def _flatten(self, eqns, path: tuple = ()) -> None:
        """The transparent call descent for the COMPUTATION slice: inline
        every well-formed transparent wrapper's equations (jit,
        custom_jvp_call, custom_vjp_call, remat2 — the same set the
        propagation descends), binding inner invars to the call's operand
        atoms and the call's outvars to the inner results, recursively.
        Obligations/asserts remain top-level-only: an inner
        ``stelling_assert`` is inlined here as an equation like any other,
        but nothing SLICES it — :func:`slice_unknown_obligations` declines
        the obligation it produced. What audit 0.2.0 M17 changed is that
        the decline is now that ONE obligation's, quoted, instead of every
        unknown obligation in the query; an inner assert did not become
        sliceable. A wrapper that resists sound inlining (no
        sub-jaxpr, arity mismatch, const mismatch) is left in place as an
        opaque equation, which the validator declines with the form
        quoted."""
        for pos, eqn in enumerate(eqns):
            if eqn.primitive in DEFAULT_TRANSPARENT:
                # coverage.call_body, the canonical accessor — see the note
                # there: remat2's body is a bare Jaxpr on jax 0.10 and a
                # ClosedJaxpr on 0.11, so a ClosedJaxpr-only test left every
                # remat'd wrapper opaque on 0.10 and declined the slice.
                inner = call_body(eqn)
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
                    # THE RENAME IS RECORDED, not only applied. It is the only
                    # thing that can translate a name from that scope into
                    # this slice, and the propagation hands us assumes stated
                    # in exactly those names. Keyed by the scope's positional
                    # address, which the propagator computes the same way from
                    # the same tree — so a key either denotes this scope or
                    # denotes nothing.
                    inner_path = path + (("call", pos),)
                    self._scope_remaps[inner_path] = remap
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
                        [self._renumber_eqn(e, remap) for e in inner.jaxpr.eqns],
                        inner_path,
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

    def _declared_shape(self, decl: ir.JaxprEqn, vid: int) -> tuple[int, ...]:
        """THE SHAPE A ``stelling_any`` BINDS ITS VALUE AT: its ``shape``
        PARAM, normalised, or a decline when that param cannot be read.

        ONE RULE FOR THE EMISSION PATH, and that is the point of it (audit
        0.2.0 B6 re-audit, UNSOUND-1). A declaration describes itself TWICE
        — a ``shape`` param and an outvar aval — and :meth:`slice` mints
        one SMT constant per element of the **param** (``x{k}_{i}`` over
        ``_size(shape)``), never per element of the aval. So the param is
        the quantity every reader on that path must also read; a reader
        that reaches for the aval instead is comparing a quantity nothing
        emits. The three sites in the emission path that need a
        declaration's element count — the budget, :meth:`_binding_shape`
        and the input-term construction — all call this method, so none
        can implement a different rule from the others.

        **THIS IS NOT A SINGLE READ, and the docstring said it was —
        audit 0.2.0 B6 audit 3, F4.** Each call re-reads the param, and an
        object that answers differently between calls does make the check
        and the emission differ. A ``list`` SUBCLASS — which
        ``isinstance(raw, (tuple, list))`` admits — whose ``__iter__``
        yields ``(4,)`` for three reads and ``()`` after was checked at
        ``(4,)`` and minted ONE input for a four-element reference;
        :meth:`slice` alone reads it three times. That is still true of
        this METHOD, and it is why the containment is not here.

        **AND THE CONTAINMENT NAMED HERE WAS THE WRONG ONE — audit 0.2.0
        B6 audit 5, F1.** This paragraph said *"what catches that is
        :meth:`ir.ClosedJaxpr.content_hash`, which cannot encode a param
        that answers differently between iterations: it RAISES"*. It does
        not. ``ir._encode`` iterates a ``shape`` param ONCE and encodes
        whatever that read returned, so a drifting param hashes cleanly
        and stably. The ``list`` the sentence was measured on raised for
        an unrelated reason — ``_encode`` has no ``list`` arm at all, so
        an honest, undrifting ``shape=[4]`` raises the identical
        ``TypeError`` — and the ``tuple`` half of
        :data:`ir._SHAPE_PARAM_CONTAINERS`, which the sentence also
        covered, was never contained by anything. Driven with the aval at
        ``(2,)`` and the param answering ``(2,)`` once and ``(1,)``
        afterwards, that was a VERIFIED on a false claim.

        What catches it is :meth:`ir.JaxprEqn.__post_init__`: the door
        validates one read of the param against the outvar aval and
        INSTALLS it, so the equation this method is handed carries a plain
        ``tuple`` of plain ``int`` and there is no second answer to get. A
        param this method can still be lied to by is one that never went
        through that constructor: the constructible route is shut, and
        only an `object.__setattr__` past the frozen dataclass reaches it
        — the same boundary `SOUNDNESS.md` records for this method's
        sibling disclosures, and the technique this batch's own tests use
        to measure the emission face with the door out of the way.
        Naming the containment where it actually is matters, because
        "cannot drift apart" invites the next reader to stop looking, and
        naming it in the wrong place invites the same thing with a
        citation attached.

        **AND IT IS NOT THE LIBRARY'S ONLY READER of a declaration's
        element count.** :func:`stelling.propagate._declared_element_count`
        reads the outvar AVAL — the other quantity — for the
        certificate-search cap. That is sound, and it is not an exception
        to the rule above: the cap only gates WHETHER the region search
        runs, its direction is toward REFUTED, and the search re-derives
        its witness by re-running the honest propagator, so no verdict is
        derived from the miscount. It is named here because a global claim
        of sole readership that a `grep` refutes is worse than no claim.
        The narrower statement in ``SOUNDNESS.md`` — that the budget, the
        input-term construction and :meth:`_binding_shape` all read this —
        is the true one.

        FAILING CLOSED IS THE POINT OF THE VALIDATION HERE, AND IT DOES NOT
        REST ON ``ir.py``. That door was closed in the same commit, but
        this method may not depend on it: ``ir.ClosedJaxpr`` is a public
        dataclass, ``SOUNDNESS.md`` names hand-built IR as in scope, and
        the door still blesses — deliberately — a declaration with NO
        ``shape`` param at all, which reads as ``()`` here and is
        precisely the form that minted one scalar symbol for a
        four-element declaration on ``96ab47a``. A param this method
        cannot read is neither an internal error nor a pass: it is a value
        whose element count nothing can agree on, and the slice declines.

        WHAT IS ACCEPTED IS STATED POSITIVELY, AND IT IS STATED IN ONE
        PLACE: :data:`ir._SHAPE_PARAM_CONTAINERS`, which is the object the
        branch below asks and the object :func:`ir._validate_decl_eqn`
        asks. This docstring deliberately does not restate the list —
        audit 0.2.0 B6 audit 4, F1, where the door's docstring described a
        rule the door had stopped implementing two commits earlier. The
        reason for a positive rule is the one :func:`_frames` gives about
        ``str`` — ``tuple(b"34")`` is ``(51, 52)``, a pair of perfectly
        plausible extents the declaration never said — but the rule may
        not be spelled as the list of containers that hazard was first
        noticed in. It was, and ``memoryview`` and ``array.array`` walked
        past it with the identical reading (audit 0.2.0 B6 audit 3).
        Stated the other way round, the character sequences fall out of
        the rule instead of being named by it, and so does whichever
        sequence type is noticed next.
        """
        raw = decl.params_dict().get("shape", ())
        # A POSITIVE TEST, not a list of refused containers — audit 0.2.0
        # B6 audit 3, the optional item. This branch used to read
        # `isinstance(raw, (str, bytes, bytearray))`, and `memoryview` and
        # `array.array` walked past it: `tuple(memoryview(b"34"))` is
        # `(51, 52)`, the same pair of perfectly plausible extents the
        # `bytes` arm exists to refuse, and the door accepted it. Adding
        # two more names to that tuple is "the container type I happened
        # to enumerate", which is the reasoning `ir._validate_param_value`
        # is annotated as condemning.
        #
        # THE RULE LIVES IN ONE PLACE AND THIS ASKS IT — audit 0.2.0 B6
        # audit 4, F1. `ir._SHAPE_PARAM_CONTAINERS` is the same object
        # `ir._validate_decl_eqn` refuses on, so the load door and this
        # emission face cannot come to hold different rules by one of them
        # being edited, and `tests/test_shape_param_rule.py` measures both
        # partitions against that object over a population it computes.
        if not isinstance(raw, ir._SHAPE_PARAM_CONTAINERS):
            # quoted through `_safely` for the reason the branch below
            # already was: a `str` SUBCLASS can carry a `__repr__` that
            # raises, and an unguarded `{raw!r}` here turned this clean
            # decline into "internal error: RuntimeError: repr refuses"
            # (audit 0.2.0 B6 audit 3, F3). A refusal message about an
            # object already known to be malformed may not itself raise.
            raise _Decline(
                f"input declaration of variable {vid} has a shape param "
                f"{_safely(lambda: repr(raw), '<unreadable>')} of type "
                f"{_safely(lambda: type(raw).__name__, '<unreadable>')}: a "
                f"declaration records its extents in {ir._SHAPE_PARAM_RULE}, "
                f"and reading any other container as one would model an "
                f"array the declaration never described "
                f"(`tuple(b\"34\")` is `(51, 52)`) (malformed IR)"
            )
        try:
            shape = tuple(raw)
        except Exception:  # noqa: BLE001 — unreadable IS the finding
            # `isinstance` is a claim about the TYPE and not the object:
            # a `list` SUBCLASS whose `__iter__` raises satisfies the test
            # above, which is the same R5 finding `_frames` carries.
            raise _Decline(
                f"input declaration of variable {vid} has a shape param "
                f"{_safely(lambda: repr(raw), '<unreadable>')} that is "
                f"{ir._SHAPE_PARAM_RULE} whose iteration RAISES, so it is "
                f"not a readable sequence of extents and the number of "
                f"elements the emission would mint for it cannot be read "
                f"(malformed IR)"
            ) from None
        # BOUND to what the guard read, not re-read after it — audit 0.2.0
        # B6 audit 3, F1. The first spelling called `_shape_problem(shape)`,
        # which validated each extent and threw the answer away, and then
        # returned `tuple(_op_index(d) for d in shape)`: a SECOND read, and
        # the one the emission got. `_extents` reads once and hands back
        # what it tested.
        problem, extents = _extents(shape)
        if problem is not None:
            raise _Decline(
                f"input declaration of shape "
                f"{_safely(lambda: repr(shape), '<unreadable>')} has "
                f"{problem}"
            )
        return extents

    def _binding_shape(self, atom: ir.Var) -> tuple[int, ...] | None:
        """The shape the value ACTUALLY HAS, read at the site that binds
        ``atom``, or ``None`` when this slicer has no binding for it.

        The binding sites of the flattened namespace, and there are only
        these two: a constvar (top-level or carried in by a transparent
        descent — :attr:`const_avals`) and the equation that produces the
        value (:attr:`producers`, which includes ``stelling_any``
        declarations). Alias bindings are not a third: :meth:`_rewrite`
        has already resolved them away by the time this is asked, and
        :meth:`_resolve` ends only on a produced var, a constvar or a
        literal.

        **A DECLARATION IS BOUND BY ITS PARAM, NOT BY ITS OUTVAR AVAL, and
        that distinction is audit 0.2.0 B6's re-audit finding UNSOUND-1.**
        For every other producer the outvar aval IS the record of the
        binding — nothing downstream reads a competing one. A
        ``stelling_any`` is the single exception, because :meth:`slice`
        builds its ``SliceInput`` terms from the declaration's ``shape``
        PARAM and never from that outvar's aval. Reading the aval here
        made this method answer with a quantity the emission does not
        use, and for exactly that one binding class the check then
        compared the wrong pair: a declaration whose param says four
        elements and whose aval says two minted four symbols, summed the
        two the lying reference asked for, and returned ``discharged`` on
        a claim whose truth is ``8 <= 4.5`` — measured on ``96ab47a``,
        inside a ``jit`` body where the box witness is blind by
        construction and no reference disagreed with any other. The phrase
        this method's docstring always used was "the shape the value
        actually has"; it now returns it.
        """
        val = self.const_avals.get(atom.id)
        if val is not None:
            return tuple(val.shape)
        producer = self.producers.get(atom.id)
        if producer is None:
            return None
        if producer.primitive == "stelling_any":
            return self._declared_shape(producer, atom.id)
        for ov in producer.outvars:
            if isinstance(ov, ir.Var) and ov.id == atom.id:
                return tuple(ov.aval.shape)
        return None

    def _one_shape_per_value(self, eqn: ir.JaxprEqn) -> None:
        """A value has ONE shape, and both legs must be modelling it.

        AUDIT 0.2.0 S12′ — THE CLASS, NOT THE ROW. S12's repair gave
        ``dot_general`` a shared shape oracle
        (:func:`stelling.interval.dot_general_geometry`) that both the
        interval transfer and the SMT emission drive, and stated that "the
        two faces cannot hold different opinions about whether an equation
        is admissible". That was false as written, because **the oracle is
        shared and its ARGUMENTS are not**:
        :func:`stelling.interval.dot_general` asks it about the shapes of
        the PROPAGATED BOXES (``a.shape``, ``b.shape``) and
        :func:`_dot_general_plan` asks the same function about the shapes
        recorded on the equation's INVAR AVALS. Move the lie off the
        declaration and onto those avals — which ``ir.ClosedJaxpr.
        from_dict`` accepts, per ``ir.py``'s own scoping of per-primitive
        shape inference out of ``_validate_loaded`` — and the two faces
        disagree again, in the asserting direction. Measured on ``4d793cf``:
        the interval leg AGREED the contraction has four terms and printed
        the box ``[4, 8]``, the emission planned two, and ``Σ <= 4.5``
        came back VERIFIED on a claim whose truth is ``8 <= 4.5``. The
        same lie on a ``reduce_sum`` invar does the same thing, and both
        rows also mint a false REFUTED whose witness the verdict calls
        "confirmed by independent exact-rational replay" — true about the
        arithmetic, false about the plan, because replay re-derives the
        SAME truncated plan.

        So the repair is not a third shape rule for ``dot_general``. It is
        this: **no equation may be modelled at a shape that disagrees with
        the shape the value actually has**, checked once here for every
        primitive at once — ``dot_general``, ``reduce_sum``, ``scatter``,
        and every emission row not yet written.

        TWO WITNESSES TO "actually has", and they are complementary because
        each is blind exactly where the other sees:

        1. **The binding site** (:meth:`_binding_shape`). A variable is
           bound once and referred to many times; every reference must
           agree with the binding. Needs no propagation at all, so it
           reaches EVERY scope the descent flattened — including an
           equation inside a ``jit`` body, whose operands carry ids this
           slicer minted and no interval environment has ever seen.
           Measured: the ``jit``-nested form of both reproducers is a live
           false VERIFIED on ``4d793cf`` and is closed by this leg alone.
           Blind to a lie applied CONSISTENTLY at the binding and at every
           reference.

        2. **The propagated box** (:attr:`env`). The interval leg computed
           a shape for this value independently, from the values flowing
           in rather than from what the IR says about them, so it is the
           one witness a consistent lie cannot forge. This is the
           inter-leg agreement the S12 commit claimed and did not have.
           Blind OUTSIDE the top level: :func:`stelling.propagate.
           interval_env` returns the top-level environment, and the
           propagator runs every transparent call body in an ISOLATED env
           that it discards on the way out — so an inner id has no box,
           by construction and not by oversight. That blindness is why
           this leg is the SECOND witness here and not the only one.

           **AND IT IS DEFENCE IN DEPTH — stated plainly, rather than left
           to read as load-bearing.** Witness 1 answered first in all six
           measured reproducers, and over the whole test suite this leg
           examined 23,072 boxed atoms and disagreed with none. **No IR
           document has been constructed on which it is the only thing
           that sees the lie.** The consistently-applied lie it exists for
           has, so far, no live route: a ``stelling_any`` whose ``shape``
           param contradicts its outvar aval is refused by
           :class:`stelling.ir.JaxprEqn`'s own construction check, a
           constvar whose aval contradicts its value by :meth:`slice`'s
           pass 2, and a computed outvar whose aval contradicts its
           operands by that row's own shape rule
           (:func:`_route_structural`, :func:`_pair_elementwise`,
           :func:`_group_reduce_sum`, :func:`_dot_general_plan`). It is
           kept because it IS the inter-leg property, stated where it can
           be checked rather than asserted in a docstring — which is the
           mistake S12 made — and because ``env`` is a caller-supplied
           argument of :func:`slice_obligation`, which is how
           ``tests/test_aval_lie_both_faces.py`` drives it.

        A ``Var`` this slicer cannot bind at all declines rather than
        passing: an unbindable operand is precisely the case where neither
        witness can see, and a check that goes quiet where it cannot see
        is the shape of the defect it is here to close. Pass 1 of
        :meth:`slice` has already resolved every operand to a producer, a
        constvar or a literal, so this arm is unreachable from a slice
        that got this far — it is a backstop, and it fails closed.

        Literals are not checked here and need no check: a literal carries
        its value, and :meth:`slice`'s pass 2 decodes every one of them
        (``_decode_elements`` / ``_numeric_fraction``) while
        :func:`_shape_problem` above has already screened both its aval
        shape and its array shape.
        """
        for atom in eqn.invars:
            if not isinstance(atom, ir.Var):
                continue
            here = tuple(atom.aval.shape)
            bound = self._binding_shape(atom)
            if bound is None:
                raise _Decline(
                    f"{eqn.primitive!r} operand {atom.id} has no binding in "
                    f"the flattened computation, so the shape "
                    f"{here} it is referred to at cannot be checked "
                    f"against the shape it was bound at"
                )
            if bound != here:
                raise _Decline(
                    f"{eqn.primitive!r} refers to variable {atom.id} at "
                    f"shape {here} but it is BOUND at shape {bound}: a "
                    f"value has one shape, and an emission that read the "
                    f"reference would model a different array than the one "
                    f"the query computes (malformed IR)"
                )
        for atom in (*eqn.invars, *eqn.outvars):
            if not isinstance(atom, ir.Var):
                continue
            box = self.env.get(atom.id)
            if box is None:
                continue  # no top-level box: witness 1 above is what stands
            here = tuple(atom.aval.shape)
            if tuple(box.shape) != here:
                raise _Decline(
                    f"{eqn.primitive!r} refers to variable {atom.id} at "
                    f"shape {here} but interval propagation computed a box "
                    f"of shape {tuple(box.shape)} for it: the two legs are "
                    f"modelling different arrays, and the emission may not "
                    f"answer an obligation the propagation left open by "
                    f"reading a program the propagation never saw"
                )

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
            # docs/proposed-decline-messages.md #3: the decline names not
            # only the primitive but WHAT KIND of gap this is (an unbuilt
            # row — nothing here judges the form unemittable) and whether
            # the OTHER leg has a row, derived from the live registry so
            # the sentence cannot drift from it. An external evaluator
            # read the bare sentence next to "square [sound]" in the same
            # stamp's transfer list and could not tell whether the two
            # facts were in contradiction; now the decline states both.
            #
            # What the interval leg DID on this run comes only from the
            # run's own record (self.top_primitives — blinded-lens audit
            # R2/R3): a registered row may still have DECLINED the
            # triggering form (then BOTH legs have a gap here, and 'the
            # emission row alone' would be false), and a row-less
            # primitive may have been handled STRUCTURALLY (cond's branch
            # join — then 'propagated ⊤' would be false). With no run
            # record, neither direction is claimed.
            head = (
                f"primitive {prim!r} is outside the supported emission "
                f"set: no SMT emission rule has been built and audited "
                f"for it — an unbuilt row, not a policy refusal of the "
                f"form. "
            )
            tops = self.top_primitives
            row = TRANSFERS.get(prim)
            if row is not None:
                fact = (
                    f"An interval transfer row for {prim!r} IS registered "
                    f"(tier {row[1]!r})"
                )
                if tops is not None and prim in tops:
                    raise _Decline(
                        head + fact + f", but on this run the interval "
                        f"leg ALSO fell to ⊤ at {prim!r} (its transfer "
                        f"declined a form — see the coverage line and the "
                        f"decline notes), so both legs have a gap here"
                    )
                if tops is not None:
                    raise _Decline(
                        head + fact + ", so the gap is the solver-emission "
                        "row alone"
                    )
                raise _Decline(
                    head + fact + "; the coverage line records whether it "
                    "covered this run's forms"
                )
            fact = "It has no interval transfer row either"
            if tops is not None and prim in tops:
                raise _Decline(
                    head + fact + ", so the interval leg propagated ⊤ for "
                    "it (see the coverage line)"
                )
            if tops is not None:
                raise _Decline(
                    head + fact + ", yet it did not fall to ⊤ on this run: "
                    "the interval leg handled it structurally (see the "
                    "coverage line)"
                )
            raise _Decline(
                head + fact + "; the coverage line records how the "
                "interval leg treated it on this run"
            )
        # THE ONE-SHAPE-PER-VALUE CROSS-CHECK, before any per-primitive form
        # rule and before any plan is built (audit 0.2.0 S12′). It sits after
        # the two declines above only so that an unsupported or un-inlinable
        # primitive keeps naming THAT as its reason — both of those decline
        # the equation either way, so nothing reaches a plan through the
        # gap. See :meth:`_one_shape_per_value` for what each of its two
        # witnesses covers and where each is blind.
        self._one_shape_per_value(eqn)
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
        if prim in ("max", "min", "add", "sub", "mul", "neg", "div", "square"):
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
            divisor = self._resolve_for_guard(eqn.invars[1])
            problem = _zero_element_problem(divisor, self.env)
            if problem is not None:
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
                base = self._resolve_for_guard(eqn.invars[0])
                problem = _zero_element_problem(base, self.env)
                if problem is not None:
                    raise _Decline(
                        f"'integer_pow' with negative exponent {y}: "
                        f"{DIV_GUARD_REASON}{problem}"
                    )
        if prim == "pow":
            # pow has TWO invars: [base, exponent]. The exponent must be a
            # Literal (constant) — variable-exponent pow is not emittable
            # because SMT-LIB2 ^ requires a concrete exponent in QF_NRA.
            if not isinstance(eqn.invars[1], ir.Literal):
                raise _Decline(
                    f"'pow' with variable exponent: only literal-exponent "
                    f"pow is emittable (the exponent must be a compile-time "
                    f"constant)"
                )
            exp_val = _decode_elements(eqn.invars[1].val)[0]
            exp_float = float(exp_val)
            # `is_integer()` rather than `== int(exp_float)`: the latter
            # RAISES on an infinite or NaN exponent instead of declining.
            # Unreachable today (literal decode declines non-finite
            # constants in an earlier pass), but this branch should not
            # depend on that ordering to avoid a crash — the non-integer
            # arm below quotes a non-finite exponent as a decline.
            if exp_float.is_integer():
                # Integer exponent: expand to products (same as integer_pow).
                y = int(exp_float)
                if abs(y) > INTEGER_POW_EXPANSION_CAP:
                    raise _Decline(
                        f"'pow' exponent {y} exceeds the v1 expansion "
                        f"cap ({INTEGER_POW_EXPANSION_CAP})"
                    )
                if y < 0:
                    base = self._resolve_for_guard(eqn.invars[0])
                    problem = _zero_element_problem(base, self.env)
                    if problem is not None:
                        raise _Decline(
                            f"'pow' with negative exponent {y}: "
                            f"{DIV_GUARD_REASON}{problem}"
                        )
            else:
                # Non-integer exponent: the aux encoding is emitted about
                # the EXACT rational the literal denotes, or not at all.
                # No rationalisation, no tolerance — see
                # :func:`pow_exponent_rational` for why a binary64
                # distance cannot police a substitution of one real for
                # another (audit 0.2.0 S1).
                problem = rational_pow_problem(exp_float)
                if problem is not None:
                    raise _Decline(problem)
                frac = pow_exponent_rational(exp_float)
                # Base must be non-negative: JAX returns NaN for
                # pow(negative, fractional), so the Real encoding (which
                # always has a solution for odd q, or no solution for even q)
                # does not model JAX's execution. Same guard pattern as the
                # div-straddle and is_finite guards.
                base = self._resolve_for_guard(eqn.invars[0])
                base_iv = self.env.get(
                    base.id if hasattr(base, "id") else None
                )
                if base_iv is None:
                    raise _Decline(
                        f"'pow' with rational exponent {frac}: base has no "
                        f"propagated interval — cannot verify non-negativity"
                    )
                for i, lo in enumerate(base_iv.los):
                    if lo < 0.0:
                        raise _Decline(
                            f"'pow' with rational exponent {frac}: base "
                            f"interval includes negative values (lo={lo}); "
                            f"JAX returns NaN for pow(negative, fractional) "
                            f"— the Real encoding does not model this"
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
        if prim == "is_finite":
            # is_finite emits constant `true` — sound ONLY when the operand's
            # interval has finite endpoints (the real-number tautology holds).
            # When the interval reaches ±inf (overflow, unbounded declaration),
            # emitting `true` would let the solver certify a property that is
            # provably false in IEEE execution. Decline in that case.
            resolved = self._resolve_for_guard(eqn.invars[0])
            iv_box = self.env.get(resolved.id if hasattr(resolved, "id") else None)
            if iv_box is None:
                raise _Decline(
                    f"'is_finite' operand has no propagated interval: "
                    f"cannot verify finiteness without bounds"
                )
            import math
            if any(not math.isfinite(v) for v in (*iv_box.los, *iv_box.his)):
                raise _Decline(
                    f"'is_finite' operand interval has non-finite endpoints "
                    f"(lo={iv_box.los}, hi={iv_box.his}): cannot emit as "
                    f"constant true — the value may not be finite"
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

    def _carry_assumes(
        self,
        ordered: tuple[ir.JaxprEqn, ...],
        slice_inputs: list[SliceInput],
        used_consts: dict[int, object],
    ) -> tuple[tuple[SliceAssume, ...], tuple[str, ...]]:
        """Translate every forwarded relational assume into THIS slice's id
        namespace, or say why it cannot be. Returns ``(carried, skipped)``,
        a partition of ``self.relational_assumes``.

        THE ONE PLACE A FOREIGN-SCOPE ID IS EVER RESOLVED. The translation is
        the slicer's own rename — the same ``remap`` the inlining applied to
        the scope's equations — followed by the same alias resolution
        :meth:`_rewrite` applies to an equation's operands. So an operand that
        is a jit body's own intermediate lands on the term that intermediate
        got, and an operand that is the jit's *invar* lands on the caller's
        atom, which is what an inlined invar means. There is no arithmetic on
        ids anywhere in here and no lookup keyed on a bare integer: a scope
        the slicer did not inline has no entry, and its assume is SKIPPED WITH
        A REASON rather than resolved against a namespace it does not belong
        to.

        Every rejection produces a sentence. That is not politeness: the
        emission's five silent ``continue``s each cost a constraint the user
        wrote, and a run that quietly drops one answers a question nobody
        asked. It also makes the count derivable — see
        :attr:`ObligationSlice.assumes`.
        """
        if not self.relational_assumes:
            return (), ()
        # Every id the emission will hold a term for, derived IN THE SAME
        # ORDER AND BY THE SAME RULE the emission derives `names` by: the
        # decoded constants first, then one term appended per declared input
        # ELEMENT, then one tuple per equation. Deriving it here is what lets
        # the arity check happen before any term tuple is indexed (audit M6),
        # and mirroring the order rather than restating it is what keeps the
        # two from disagreeing about an id that is both.
        #
        # THE THIRD LOOP TAKES `outvars[0]` AND NOTHING ELSE, because that is
        # what the emission takes: every row of its equation loop binds
        # `out = eqn.outvars[0]` and writes exactly one `names[out.id]`. A
        # second outvar has no term there either, so claiming one here would
        # send a carried assume to the emission's unbound-variable raise —
        # an internal error quoted into an UNKNOWN — where the honest answer
        # is the disclosed skip this returns.
        term_count: dict[int, int] = {}
        for cid, cval in used_consts.items():
            term_count[cid] = len(cval) if isinstance(cval, tuple) else 1
        for inp in slice_inputs:
            term_count[inp.var_id] = term_count.get(inp.var_id, 0) + 1
        for e in ordered:
            out = e.outvars[0]
            term_count[out.id] = _size(_shape_of(out))

        def n_terms(atom: ir.Atom) -> int | None:
            """How many element terms the emission will have for this atom,
            or None when it will have none at all."""
            if isinstance(atom, ir.Literal):
                vals = _decode_elements(atom.val)  # declines undecodables
                for v in vals:
                    if not isinstance(v, bool):
                        _numeric_fraction(v)  # declines non-finite
                return len(vals)
            return term_count.get(atom.id)

        carried: list[SliceAssume] = []
        skipped: list[str] = []
        for origin, ra in enumerate(self.relational_assumes):
            eqn = ra.eqn
            at = f" at {ra.where}" if ra.where else ""
            head = f"relational assume{at} not forwarded to the solver: "
            # THE NEXT THREE SKIPS ARE SCREENED BY THE PROPAGATOR FIRST, and
            # are DEFENCE IN DEPTH rather than live paths. Written down here
            # because a reader who takes them for live paths will look for
            # (and "fix") a cause that does not exist:
            #
            #   * the two `eqn.*` shape screens — `_Walker._classify_
            #     assumed_pred` forwards only a producer whose primitive is in
            #     `_ASSUME_CMPS` and whose `invars` are exactly two `ir.Var`s
            #     (it drops on `prim not in _ASSUME_CMPS` and on
            #     `len(producer.invars) != 2` before ever reaching the forward
            #     site), and `_ASSUME_CMPS` is the same set as `ASSUME_CMP_SYM`
            #     here. Their reachable trigger is a `RelationalAssume`
            #     assembled by hand;
            #   * `remap is None` — this was written for a scope the slicer
            #     does not descend, which means a `cond` branch, and a
            #     branch-scoped relational assume is no longer forwarded at
            #     all (`branch_depth` is incremented before the branch's scope
            #     step is pushed, and the forwarding guard refuses while it is
            #     nonzero), so no forwarded assume can name a scope for that
            #     reason. What is left reachable is narrow and is not a cond:
            #     the propagator descends a transparent wrapper on
            #     `len(inner.jaxpr.invars) == len(ins)` while `_flatten`
            #     additionally requires the outvar and constvar counts to
            #     match, so a wrapper malformed in exactly that way is
            #     descended by one and left opaque by the other. jax tracing
            #     does not produce one.
            #
            # All three are kept: each names an assume and a reason, which is
            # the posture this whole method exists to enforce, and none of
            # them can mint anything — a skip is a disclosed loss of
            # precision, and the withholding rule reads the skip.
            if eqn.primitive not in ASSUME_CMP_SYM:
                skipped.append(
                    head + f"the assumed comparison {eqn.primitive!r} has no "
                    f"SMT emission rule"
                )
                continue
            if len(eqn.invars) != 2 or not eqn.outvars:
                skipped.append(
                    head + f"the assumed comparison {eqn.primitive!r} has "
                    f"{len(eqn.invars)} operand(s) and {len(eqn.outvars)} "
                    f"result(s), not the binary form the emission writes"
                )
                continue
            remap = self._scope_remaps.get(ra.scope)
            if remap is None:
                skipped.append(
                    head + f"it was stated in {_render_scope(ra.scope)}, "
                    f"which this obligation's slice does not inline, so its "
                    f"operands name nothing here"
                )
                continue
            try:
                renamed = self._renumber_eqn(eqn, remap)
                probe = ir.JaxprEqn(
                    primitive=renamed.primitive,
                    invars=tuple(self._resolve(a) for a in renamed.invars),
                    outvars=renamed.outvars,
                    params=renamed.params,
                    effects=renamed.effects,
                    source_info=renamed.source_info,
                )
                # The SAME pairing the slice's own equations go through, run on
                # the RESOLVED operands: whatever indices come out of here
                # index the very term tuples the emission will build.
                #
                # BELT AND BRACES BEHIND THE IDENTITY REPAIR, and measured to
                # be exactly that — not the M6 repair, which an earlier
                # version of this comment credited it with. Substituting
                # `renamed` for `probe` here reddens 0 of the suite's 2954
                # collected tests on this tree, because alias resolution cannot
                # change an aval: `_flatten` binds an inner invar to the
                # call's operand atom and an outer outvar to the inner result
                # atom, both of which jax has already type-checked equal, and
                # `_renumber_eqn` renames without touching avals. So the two
                # calls see the same shapes on every reachable input and
                # return the same indices.
                #
                # What actually closed M6's crash and its silent truncation is
                # the SCOPE-CORRECT IDENTITY: `remap` above resolves the
                # assume's operands through the rename of the scope they were
                # traced in, so they can no longer land on an unrelated term
                # of a different shape — which is where the mismatched indices
                # came from. Reading `probe` is kept because it is the
                # honest reading (the emission indexes the resolved operands'
                # terms, so the indices should come from the resolved
                # operands), not because a case is known that distinguishes
                # it. `docs/norms.md` § "Guard coverage is proven by mutation,
                # not by construction": no such case has been constructed, so
                # none is claimed.
                idx = _pair_elementwise(probe)
                counts = [n_terms(a) for a in probe.invars]
            except _Decline as d:
                skipped.append(head + d.reason)
                continue
            if len(idx) != 2:
                skipped.append(
                    head + f"the assumed comparison paired into {len(idx)} "
                    f"operand index list(s), not two"
                )
                continue
            missing = [
                side for side, c in zip(("left", "right"), counts) if c is None
            ]
            if missing:
                # "its left and right operandS ARE", not "operand is": the
                # sentence is read by whoever lost the constraint, and a
                # reason that does not parse reads as a template rather than
                # as a finding
                noun, verb, pron = (
                    ("operands", "are", "them") if len(missing) == 2
                    else ("operand", "is", "it")
                )
                skipped.append(
                    head + f"its {' and '.join(missing)} {noun} {verb} not "
                    f"in this obligation's backward cone, so no term for "
                    f"{pron} exists in this slice"
                )
                continue
            if any(c == 0 for c in counts):
                skipped.append(
                    head + "an operand has zero elements, so the comparison "
                    "constrains nothing"
                )
                continue
            # ARITY IS CHECKED, AND `_pair_elementwise` ABOVE IS THE CHECK.
            # The M6 crash and the M6 silent truncation were one missing check
            # with two directions, and their shared cause was that the element
            # indices came from the ASSUME EQUATION's shapes while the terms
            # came from whatever the ids HAPPENED TO RESOLVE TO — a foreign
            # scope's id landing on an unrelated term of a different shape.
            # THE SCOPE-CORRECT RESOLUTION ABOVE IS WHAT REMOVES THAT: an
            # operand is now resolved through the rename of the scope it was
            # traced in, or the assume is skipped with a reason, so no id lands
            # on an unrelated term at all. `_pair_elementwise` running on the
            # resolved operands makes the indices and the terms come from one
            # pair of shapes rather than two, and it declines — quoted, into
            # `skipped` — when they do not broadcast. It is the check for the
            # shapes; it is not what closed M6, and the mutation measurement
            # at the call site says so.
            #
            # There is deliberately NO second arity check here comparing
            # `counts` against the operands' avals. The one place that holds
            # both the indices and the term tuples is the emission, and that is
            # where the residual check lives — as a raise with a message
            # rather than a bare tuple subscript. A copy here would be
            # unreachable (the terms a slice holds for an atom are sized by
            # that atom's own aval), and an unreachable guard is a second
            # implementation of an invariant that no mutation can prove.
            ia, ib = idx
            carried.append(
                SliceAssume(
                    primitive=probe.primitive,
                    invars=probe.invars,
                    pairs=tuple(zip(ia, ib)),
                    source_info=eqn.source_info,
                    origin=origin,
                )
            )
        return tuple(carried), tuple(skipped)

    def slice(self, index: int, assert_eqn: ir.JaxprEqn) -> ObligationSlice:
        if self.poisoned is not None:
            raise _Decline(self.poisoned)
        root = self._resolve(assert_eqn.invars[0])
        # ONE READ, AND THE COUNT IS TAKEN AFTER THE VERDICT — audit 0.2.0
        # B6 audit 4, F3. `root_size` was computed here from the raw
        # objects, BEFORE `_shape_problem` had said anything about them.
        root_problem, root_extents = _extents(root.aval.shape)
        if not _is_bool_dtype(root.aval):
            raise _Decline(
                f"assert operand has dtype {root.aval.dtype!r}, expected bool"
            )
        if root_problem is not None:
            raise _Decline(
                f"assert operand has shape "
                f"{_safely(lambda: repr(tuple(root.aval.shape)), '<unreadable>')} "
                f"with {root_problem}"
            )
        root_size = _size(root_extents)
        if root_size == 0:
            # zero elements: the universal claim is vacuously true, and
            # interval propagation already discharges it (matching measured
            # jax: jnp.all of a size-0 predicate is True), so nothing
            # reaches escalation through the normal path — a direct ask
            # declines rather than mints a vacuous proof obligation.
            raise _Decline(
                f"assert operand has shape {root_extents} with "
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
            # through _declared_shape, which is THE reader of a
            # declaration's element count — see its docstring: the budget,
            # the input-term construction below and _binding_shape must
            # all count the same elements the emission mints, and one
            # function is how that stops being a coincidence
            element_terms += _size(
                self._declared_shape(self.producers[vid], vid)
            )
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
            # docs/proposed-decline-messages.md #4: the old sentence was
            # accurate and quantitative, which made the WRONG lever more
            # likely to be pulled — it read as a diagnosis, and a reader
            # went to tune the budget. The decline now names the exceeding
            # quantity, says what the budget gates (the ATTEMPT — the
            # interval result is budget-independent by construction), and
            # names the lever that actually lowers the counts. The
            # proposal's 'obligations past the budget declined again on a
            # different cause once it was raised' is a claim about
            # unrecorded measurements and is deliberately NOT stated.
            over = " and ".join(
                name
                for name, v in (
                    ("element terms", element_terms),
                    ("root conjuncts", root_size),
                )
                if v > ELEMENT_BUDGET
            )
            raise _Decline(
                f"obligation not attempted: it needs {element_terms} "
                f"element terms and {root_size} root conjuncts, and its "
                f"{over} put it over the per-obligation emission budget of "
                f"{ELEMENT_BUDGET} (bounded static-shape emission; the "
                f"budget is measured solver cost, see "
                f"stelling.obligation.ELEMENT_BUDGET). The budget bounds "
                f"what escalation will ATTEMPT; it is not a diagnosis of "
                f"the obligation, and raising it does not change the "
                f"interval result that left this obligation undecided. "
                f"What lowers the counts is a smaller obligation — a "
                f"smaller declared array, or a per-element property "
                f"instead of a whole-array one"
            )

        # -- pass 2: decode, validate, build — all bounded by the gate ---
        for atom in literal_atoms:
            for v in _decode_elements(atom.val):  # declines undecodables
                if not isinstance(v, bool):
                    _numeric_fraction(v)  # declines non-finite
        used_consts: dict[int, object] = {}
        for cid in const_ids:
            c_aval = self.const_avals.get(cid)
            c_extents: tuple[int, ...] | None = None
            if c_aval is not None:
                # ONE READ — audit 0.2.0 B6 audit 4, F3. This screened the
                # shape with `_shape_problem` and then counted it TWICE
                # more with `_size(c_aval.shape)`, three reads of the same
                # self-describing objects where the guard had validated
                # only the first.
                problem, c_extents = _extents(c_aval.shape)
                if problem is not None:
                    # P1(b): the slicer rebuilds values independently of
                    # the propagation, so it must refuse the refused
                    # class itself — the constvar's DECLARED aval, which
                    # a lying consumer reference never shows it
                    raise _Decline(
                        f"constvar {cid} has aval shape "
                        f"{_safely(lambda: repr(tuple(c_aval.shape)), '<unreadable>')} "
                        f"with {problem}"
                    )
            vals = _decode_elements(self.consts[cid])
            if c_extents is not None and len(vals) != _size(c_extents):
                raise _Decline(
                    f"constvar {cid} decodes to {len(vals)} element(s) "
                    f"but its aval shape {c_extents} holds "
                    f"{_size(c_extents)} (aval/value mismatch, "
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
            # THE SAME READER the budget above and _binding_shape use, so
            # the shape this mints terms from is by construction the shape
            # the cross-check compared every reference against. An empty
            # declared set (a negative extent — fix-re-attack R1) declines
            # inside it, through `_extents`: no array of such a shape
            # exists, so there is nothing to declare and any universal
            # claim over it is vacuous.
            #
            # AND THIS CALL IS UNREACHABLE AS A GUARD, RECORDED RATHER THAN
            # CLAIMED — audit 0.2.0 B6 audit 3, F5. Reverting it alone to
            # its own independent read reds NOTHING in the suite, because
            # the budget loop above has already called `_declared_shape`
            # over the same vids and declined for the same reasons. The
            # batch's "each change has a test that reds when reverted
            # alone" was therefore false here, and `docs/norms.md` forbids
            # exactly the move of asserting coverage by construction.
            #
            # It is KEPT, and not deleted, because it is not a guard: it is
            # a VALUE read, and the value it must produce is the one the
            # budget counted and the one `_binding_shape` compared every
            # reference against. An independent read here — which is what
            # the code did before — is UNSOUND-1 itself, a second reader of
            # a declaration's element count implementing its own rule. That
            # no test can tell the two apart today is a fact about today's
            # readers agreeing, not a reason to let them diverge again.
            shape = self._declared_shape(self.producers[vid], vid)
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
        assumes, assumes_skipped = self._carry_assumes(
            ordered, slice_inputs, used_consts
        )
        return ObligationSlice(
            index=index,
            fragment=fragment,
            inputs=tuple(slice_inputs),
            consts=tuple(sorted(used_consts.items())),
            eqns=ordered,
            root=root,
            source_info=assert_eqn.source_info,
            element_terms=element_terms,
            assumes=assumes,
            assumes_skipped=assumes_skipped,
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
            if prim == "pow":
                # TWO different reasons, and only one of them is about the
                # base's dependence.
                #
                # (a) A NON-INTEGER exponent emits the auxiliary-variable
                # encoding: a fresh `aux` constrained by `aux^q = x^p`.
                # `aux^q` is a product of a fresh symbol with ITSELF, so
                # the emitted script is nonlinear whatever the base is —
                # a constant base does not make it linear, and the
                # constant fold never removes it (smt._fold_values
                # declines rational exponents: the value is generally
                # irrational). Stamping such a slice QF_LRA shipped
                # `(* aux aux)` under a linear logic and BOTH backends
                # refused the script, losing the whole obligation and
                # blaming the solvers for stelling's own label
                # (audit 0.2.0 M9).
                #
                # (b) An INTEGER exponent outside {0, 1} expands to a
                # product of the base with itself — nonlinear exactly when
                # the base descends from a declaration, same rule as
                # integer_pow; a constant base folds to a numeral and the
                # decision problem really is linear.
                exp_atom = eqn.invars[1]
                if not isinstance(exp_atom, ir.Literal):
                    nonlinear = True  # variable exponent is always nonlinear
                else:
                    exp_f = float(_decode_elements(exp_atom.val)[0])
                    # `is_integer()` rather than `== int(...)`: it is False
                    # for inf/nan instead of raising on them, and a
                    # non-finite exponent has already declined by here
                    # anyway (literal decode, pass 2).
                    if not exp_f.is_integer():
                        nonlinear = True                       # (a)
                    elif ins_dep[0] and exp_f not in (0.0, 1.0):
                        nonlinear = True                       # (b)
            if prim == "square" and ins_dep[0]:
                # unconditionally nonlinear when the operand depends on a
                # declaration: `square` has no exponent param to fall back
                # to a linear case with, unlike integer_pow's y in (0, 1).
                nonlinear = True
            if any(ins_dep):
                for out in eqn.outvars:
                    dependent.add(out.id)
        return QF_NRA if nonlinear else QF_LRA


def slice_obligation(
    closed: ir.ClosedJaxpr,
    index: int,
    env: Mapping[int, iv.IntervalArray],
    *,
    assert_position: int | None = None,
    top_primitives: frozenset[str] | None = None,
    relational_assumes: tuple[RelationalAssume, ...] = (),
) -> ObligationSlice | DeclinedObligation:
    """Extract the slice for obligation ``index`` (top-level assert order),
    or decline with the reason quoted. Never raises on legal queries.

    ``assert_position`` separates the two things ``index`` used to be at
    once: WHICH top-level ``stelling_assert`` to slice, and WHICH
    obligation number to report. They coincide exactly when every
    obligation is a top-level assert, and they stop coinciding the moment
    one ``assert_`` is written inside a ``jax.jit`` helper — audit 0.2.0
    M17. Callers that know the association (:func:`slice_unknown_
    obligations`, which reads it off
    :attr:`stelling.propagate.ObligationReport.top_level_eqn_pos`) pass it;
    a caller that does not leaves it ``None`` and gets the old
    ``index``-is-the-position reading, which is right for a query whose
    asserts are all top-level and is the only thing an uninformed caller
    can honestly assume.

    ``top_primitives`` (message wording only, never admission): the
    primitives the interval RUN recorded as fallen-to-⊤, from
    ``propagation.coverage.unknown_primitives`` — the unsupported-
    primitive decline describes the interval leg's run behaviour from
    this record and claims neither direction when it is ``None``.

    ``relational_assumes`` are the propagation's forwarded relational
    assumes. A slice built WITHOUT them states none: the axioms are part of
    what the slice claims, so a re-derivation that was not given them is not
    the same slice and its emitted script is not the same script. That is why
    :func:`stelling.verdict._bar_scope`'s re-derivation, which has no
    propagation to draw them from, still fails to reproduce an
    assume-carrying script — unchanged by this fix, and its own finding."""
    jaxpr = closed.jaxpr
    asserts = [e for e in jaxpr.eqns if e.primitive == "stelling_assert"]
    position = index if assert_position is None else assert_position
    # THE RANGE TEST IS TWO-SIDED, and the lower side is new. A negative
    # index WITHIN range has always been Python indexing from the end and
    # stays that way (it is measured in
    # `test_what_a_stray_index_ACTUALLY_DOES_all_four_of_them`); one PAST the
    # start used to raise a raw `IndexError` out of a function whose contract
    # is the second line of this docstring, and reached the whole-query bar
    # through `stelling.verdict._bar_scope`'s outer `except` rather than
    # through the decline channel. An out-of-range index is not an internal
    # error and should not be reported as one, so it is named here rather
    # than left to the net below.
    if not -len(asserts) <= position < len(asserts):
        return DeclinedObligation(
            index=index,
            reason=(
                f"obligation #{index} has no matching top-level "
                f"stelling_assert equation"
            ),
        )
    assert_eqn = asserts[position]
    try:
        return _Slicer(
            closed, env, top_primitives, relational_assumes
        ).slice(index, assert_eqn)
    except _Decline as d:
        return DeclinedObligation(
            index=index, reason=d.reason, source_info=assert_eqn.source_info
        )
    except ir.TranscriptionError as t:
        # A MALFORMED DOCUMENT IS NOT A STELLING DEFECT, and the sentence
        # must not say it is. The slicer REBUILDS equations as it inlines a
        # transparent call body (:meth:`_Slicer._renumber_eqn`), so
        # `ir.JaxprEqn.__post_init__`'s own construction checks run again
        # over the descended scope — and a document that only reaches the
        # library through a public dataclass can fail one there. Routed to
        # the decline channel with the transcription refusal quoted: the
        # obligation stays UNKNOWN either way, but a reader is told the IR
        # is malformed rather than that the tool broke (which is what the
        # generic net below would have said, in the M17′ wording that audit
        # already named as misleading).
        return DeclinedObligation(
            index=index,
            reason=(
                f"the obligation's computation could not be re-transcribed "
                f"for slicing: {t}"
            ),
            source_info=assert_eqn.source_info,
        )
    except Exception as e:  # noqa: BLE001 — guard rule: degrade, quoted
        # THIS FUNCTION'S CONTRACT IS THE SECOND LINE OF ITS DOCSTRING and
        # it was breakable. Audit 0.2.0 S12's other half: a `dot_general`
        # whose operands' contracted extents disagreed indexed off the end
        # of the constant operand and raised a raw `IndexError` here — an
        # exception class NO caller handles. `escalate` iterates
        # `slice_unknown_obligations` in the FOR HEADER, outside its own
        # per-obligation `except Exception` net, so the crash escaped the
        # whole call and a batch caller lost every other obligation's
        # verdict with it.
        #
        # The extent defect itself is fixed at its root (the shared shape
        # oracle `_dot_general_plan` now drives), so this net catches
        # NOTHING that is currently constructable — which is the point. It
        # is the same posture, in the same words, that `stelling.solvers.
        # escalate` already takes around `_dispatch_obligation`: mid-analysis
        # a bug degrades to UNKNOWN, QUOTED, never silently and never as a
        # crash. Quoted is what keeps it from being a swallow: the sentence
        # names the exception class and its message, rides into the
        # obligation's `detail`, and says INTERNAL ERROR in those words, so
        # a reader sees a stelling defect rather than an undecided
        # obligation.
        return DeclinedObligation(
            index=index,
            reason=(
                f"slice attempted; internal error: {type(e).__name__}: {e}"
            ),
            source_info=assert_eqn.source_info,
        )


def _safely(read, fallback):
    """``read()``, or ``fallback`` when reading raises.

    For composing a message ABOUT AN OBJECT ALREADY KNOWN TO BE
    MISBEHAVING, and for nothing else — the inside of an exception
    handler, and the refusal a guard raises once it has decided to refuse.
    Neither may itself be able to raise: a handler that re-raises costs
    every sibling's verdict exactly as the original raise would have, and
    a decline whose message raises is not a decline at all but the raw
    escape the guard existed to prevent (audit 0.2.0 B6 audit 3, F3). The
    same swallow anywhere on a DECIDING path would hide a defect instead
    of quoting one, so the fallbacks are visible placeholders: a reader
    sees that something could not be read, rather than a plausible value.
    """
    try:
        return read()
    except Exception:  # noqa: BLE001 — the handler's own totality
        return fallback


def _frames(v) -> tuple | None:
    """``v`` read as a tuple of source frames, or ``None`` when it is not
    one — TOTAL, and that is the whole point of it.

    :attr:`stelling.ir.JaxprEqn.source_info` and
    :attr:`stelling.propagate.ObligationReport.source_info` are both
    declared ``tuple[str, ...]`` and ``from_dict`` builds them that way, so
    on every query that comes through the documented door this is
    ``tuple`` in and ``tuple`` out. ``ir.ClosedJaxpr`` is a public
    dataclass, though, and ``SOUNDNESS.md`` names hand-built IR as in
    scope: the association check in :func:`slice_unknown_obligations` used
    to write ``tuple(eqn.source_info)`` directly and raised ``TypeError``
    on an ``int`` there (audit 0.2.0 B6/M17′).

    A ``list`` is accepted because the check it serves is about the FRAMES,
    not about which sequence type holds them, and the old ``tuple(...)``
    normalization accepted one. Anything else — an ``int``, a bare ``str``
    (whose ``tuple()`` is a tuple of CHARACTERS, so coercing it would
    compare the wrong thing rather than raise) — is not a frame list, and
    the caller reads ``None`` as "this association cannot be checked",
    which is a decline. Not raising is the structural guarantee; the net
    around the caller's body is the backstop for the lines this helper does
    not cover.

    **AND THE ``list`` ARM IS NETTED, because ``isinstance`` is a claim
    about the type and not about the object** — audit 0.2.0 B6 re-audit R5.
    A ``list`` SUBCLASS whose ``__iter__`` raises satisfies the
    ``isinstance`` test and then raises inside ``tuple(v)``, so "TOTAL" was
    a docstring assertion this function did not keep. It keeps it now: a
    value that will not iterate is not a frame list either, and reads as
    ``None`` by exactly the same reasoning as an ``int`` does."""
    if isinstance(v, tuple):
        return v
    if isinstance(v, list):
        try:
            return tuple(v)
        except Exception:  # noqa: BLE001 — totality is the contract here
            return None
    return None


def slice_unknown_obligations(
    closed: ir.ClosedJaxpr,
    propagation: Propagation,
    env: Mapping[int, iv.IntervalArray],
) -> tuple[ObligationSlice | DeclinedObligation, ...]:
    """Slices (or quoted declines) for exactly the obligations interval
    propagation left ``unknown``. Discharged and violated obligations are
    already decided and are not re-decided.

    **THE ASSOCIATION IS PER-OBLIGATION, AND IT IS CHECKED RATHER THAN
    GUESSED** — audit 0.2.0 M17. Escalation slices a top-level
    ``stelling_assert``; an obligation recorded from inside a sub-jaxpr has
    no such equation and must decline. This used to be decided by COUNTING:
    when ``len(asserts) != len(propagation.obligations)`` nothing could be
    mapped and EVERY unknown obligation declined. One ``assert_`` written
    inside a ``jax.jit`` helper therefore cost solver escalation for every
    other obligation in the query, and the resulting UNKNOWN was widely
    misread as the per-obligation element budget (which is genuinely
    per-obligation and was never the cause).

    **The count check was SOUND, and stating that precisely is what says
    where the defect actually is.** Every top-level ``stelling_assert``
    records exactly one obligation (the walk visits every top-level
    equation, and the malformed-shape screen explicitly EXEMPTS asserts
    from declining so that they are still recorded), so
    ``len(obligations) >= len(asserts)`` always, with equality exactly when
    no obligation came from a sub-jaxpr — and under equality index ``k``
    really is assert ``k``. Nothing was ever mis-sliced. The defect is
    purely that the instrument answers a PER-OBLIGATION question with a
    WHOLE-QUERY number: the unmappable obligation is one of them, and its
    siblings are ordinary top-level equations with ordinary slices that
    were being thrown away with it.

    So the walk records where it saw each assert
    (:attr:`stelling.propagate.ObligationReport.top_level_eqn_pos`) and
    this function VERIFIES that record against the IR it was handed —
    the position must name a top-level ``stelling_assert``, it must carry
    the same ``source_info`` the obligation carries, and no two obligations
    may claim it. An obligation failing any of those, or carrying no
    position at all, declines individually with the reason quoted. The
    safety property is unchanged and stated the same way: **an obligation
    whose association cannot be trusted still declines** — what changed is
    that its siblings no longer decline with it.

    **AND IT MAY NOT RAISE — audit 0.2.0 B6/M17′.** Both callers
    (:func:`stelling.solvers.escalate`, :func:`stelling.affine.
    refine_propagation`) iterate this function IN THE ``for`` HEADER,
    outside their own per-obligation ``except Exception`` net, so anything
    raised here escapes the whole call and a batch caller loses every
    obligation's verdict — not only the offending one. That is precisely
    the whole-query-for-a-per-obligation-question failure M17 was about,
    and the M17 fix itself opened a fresh instance of it: the association
    check below called ``tuple(...)`` on a ``source_info`` it had not
    established was iterable, and on hand-built IR carrying a non-tuple
    there (``ir.ClosedJaxpr`` is a public dataclass; ``from_dict``
    coerces only at its own door) ``escalate`` raised ``TypeError: 'int'
    object is not iterable`` where ``dee8bc2`` returned
    ``[(0, 'violated-witness')]``.

    So the per-obligation body is netted, and netted PER OBLIGATION rather
    than around the whole function: a net around the whole function would
    answer the per-obligation question with a whole-query outcome again.

    **THE PREAMBLE IS OUTSIDE THE NET, AND THE REASON IS SHADOWING, NOT
    TOTALITY** — audit 0.2.0 B6 re-audit R6. The claim that used to stand
    here, that the preamble "cannot raise on any object", is false: it
    iterates ``closed.jaxpr.eqns`` and ``propagation.obligations``, reads
    ``e.primitive`` and ``o.top_level_eqn_pos``, and hashes the latter into
    ``claimants`` — eight sites, any of which a hostile object can make
    raise. What is true is that **none of them is the FIRST raise** on such
    an object. ``escalate`` and ``refine_propagation`` both build their
    interval environment (``propagate.interval_env(closed)``) and their own
    list comprehensions over ``propagation.obligations`` before they call
    this function at all, so an ``obligations`` that will not iterate, or a
    ``jaxpr`` that will not yield equations, has already raised in the
    caller — where it is the caller's own crash and not a lost batch of
    verdicts. The residual is named rather than denied: an object that
    survives those earlier reads and raises only on ``.primitive``, on
    ``.top_level_eqn_pos``, or on ``hash()`` of what the latter returns,
    raises here and takes the whole call with it. Nothing constructs one
    today, and if one is ever built the fix is to move the preamble into a
    per-obligation net, not to restore the sentence.
    """
    unknown = [o for o in propagation.obligations if o.status == "unknown"]
    eqns = closed.jaxpr.eqns
    # position in the top-level eqns -> ordinal among top-level asserts,
    # which is the index `slice_obligation` selects by
    ordinal_of_pos: dict[int, int] = {}
    for pos, e in enumerate(eqns):
        if e.primitive == "stelling_assert":
            ordinal_of_pos[pos] = len(ordinal_of_pos)
    claimants: dict[int, int] = {}
    for o in propagation.obligations:
        if o.top_level_eqn_pos is not None:
            claimants[o.top_level_eqn_pos] = (
                claimants.get(o.top_level_eqn_pos, 0) + 1
            )
    # the run record rides along for message wording (audit R2/R3): the
    # unsupported-primitive decline describes what the interval leg did
    # on THIS run from the coverage instrument's own record
    tops = frozenset(
        name for name, _ in propagation.coverage.unknown_primitives
    )

    def _decide(o) -> ObligationSlice | DeclinedObligation:
        pos = o.top_level_eqn_pos
        if pos is None:
            return DeclinedObligation(
                index=o.index,
                reason=(
                    "the assert this obligation was recorded from is not a "
                    "top-level equation of the query (it sits inside a "
                    "sub-jaxpr — a transparent call body, a cond branch, or "
                    "an undescended scan/while body), and escalation slices "
                    "top-level asserts only"
                ),
                source_info=o.source_info,
            )
        if pos not in ordinal_of_pos:
            return DeclinedObligation(
                index=o.index,
                reason=(
                    f"the recorded top-level position {pos} of this "
                    f"obligation's assert does not name a stelling_assert "
                    f"equation in the query handed to escalation: the "
                    f"propagation and the query disagree, so no slice can be "
                    f"attributed to it"
                ),
                source_info=o.source_info,
            )
        # READ ONCE. The count and the sentence that quotes it must be the
        # SAME read: `claimants.get(pos, 0)` followed by `claimants[pos]`
        # disagreed whenever the key was absent, and the absent case is
        # reachable — `claimants` was built from a first read of every
        # obligation's `top_level_eqn_pos` and `pos` here is a SECOND read
        # of the same attribute, so an obligation whose attribute does not
        # answer the same way twice lands on `0 != 1` and then raises
        # `KeyError`, which the net below turns the intended sentence into
        # "internal error: KeyError: 3" (audit 0.2.0 B6 re-audit).
        n_claimants = claimants.get(pos, 0)
        if n_claimants != 1:
            return DeclinedObligation(
                index=o.index,
                reason=(
                    f"{n_claimants} obligations claim top-level assert "
                    f"position {pos}: the association is not one-to-one, so "
                    f"none of them may be sliced by it"
                ),
                source_info=o.source_info,
            )
        eqn_frames = _frames(eqns[pos].source_info)
        obl_frames = _frames(o.source_info)
        if eqn_frames is None or obl_frames is None:
            # NOT the same finding as a disagreement, and saying so is the
            # difference between a reader who can act and one who reads
            # "traced at 7 but records 7" and concludes the tool is broken.
            # Nothing was compared here: one of the two sides is not a list
            # of frames at all, so the association is UNCHECKABLE rather
            # than checked-and-refuted.
            side = "the query's assert" if eqn_frames is None else (
                "this obligation"
            )
            bad = eqns[pos].source_info if eqn_frames is None else o.source_info
            return DeclinedObligation(
                index=o.index,
                reason=(
                    f"the source_info of {side} at top-level position {pos} "
                    f"is {bad!r}, which is not a list of source frames: the "
                    f"association between the propagation's record and the "
                    f"query cannot be CHECKED at all, so no slice may be "
                    f"attributed to it"
                ),
                source_info=o.source_info,
            )
        if eqn_frames != obl_frames:
            return DeclinedObligation(
                index=o.index,
                reason=(
                    f"the assert at top-level position {pos} was traced at "
                    f"{eqn_frames!r} but this obligation records "
                    f"{obl_frames!r}: the propagation and the query "
                    f"disagree, so no slice can be attributed to it"
                ),
                source_info=o.source_info,
            )
        return slice_obligation(
            closed,
            o.index,
            env,
            assert_position=ordinal_of_pos[pos],
            top_primitives=tops,
            relational_assumes=propagation.relational_assumes,
        )

    def one(o) -> ObligationSlice | DeclinedObligation:
        # THE NET, per obligation. See this function's docstring: both
        # callers iterate it in the `for` HEADER, so a raise here is not one
        # obligation's problem but every obligation's. `slice_obligation`
        # carries the same posture around `_Slicer.slice`; this one covers
        # everything OUTSIDE that call — the association checks above, which
        # is where B6/M17′ actually escaped from.
        #
        # AND THE HANDLER IS ITSELF NETTED — audit 0.2.0 B6 re-audit R7. A
        # net that re-raises while composing its own message is not a net.
        # Three of the four reads in the block below can raise on a hostile
        # object: `str(e)` runs the exception's own `__str__`,
        # `getattr(o, "index", -1)` returns the default only for
        # `AttributeError` and propagates anything else a property raises,
        # and `getattr(o, "source_info", ())` likewise. Each is therefore
        # taken through `_safely`, which substitutes a quoted placeholder
        # rather than a second traceback: the point of this net is that ONE
        # obligation's malformation costs one obligation's verdict, and an
        # escape from the handler costs every sibling's exactly as the
        # original raise would have.
        try:
            return _decide(o)
        except Exception as e:  # noqa: BLE001 — guard rule: degrade, quoted
            return DeclinedObligation(
                index=_safely(lambda: o.index, -1),
                reason=(
                    f"associating this obligation with a top-level assert "
                    f"attempted; internal error: "
                    f"{_safely(lambda: type(e).__name__, '<unreadable>')}: "
                    f"{_safely(lambda: str(e), '<unreadable message>')}"
                ),
                source_info=_safely(lambda: o.source_info, ()),
            )

    return tuple(one(o) for o in unknown)


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
       "dot_general", "scatter", "is_finite"}
)

if _REPLAY_SUPPORTED != _SUPPORTED:
    raise RuntimeError(
    "the exact-rational replay must cover exactly the emission set, or a "
    "witness can be produced that replay cannot independently confirm: "
    f"emittable-but-not-replayable {sorted(_SUPPORTED - _REPLAY_SUPPORTED)}, "
    f"replayable-but-not-emittable {sorted(_REPLAY_SUPPORTED - _SUPPORTED)}"
)



def _square_value(v):
    """One element of `square`'s replay: the exact rational self-product.

    Named and module-level for the same reason
    :func:`stelling.smt._square_body` is — the two faces of this row are
    one expression each, and a row whose faces cannot be mutated
    INDEPENDENTLY cannot be gauged (a single seam patched in both places at
    once measures agreement with itself). ``v * v`` and not ``v ** 2``: the
    replay's whole job is to re-derive the violation, and the self-product
    is the shape the emission writes, in the arithmetic — exact
    :class:`fractions.Fraction` — the emission does not have.
    """
    return v * v


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


def _exact_integer_root(n: int, q: int) -> int | None:
    """The exact integer ``q``-th root of the non-negative integer ``n``,
    or ``None`` when ``n`` is not a perfect ``q``-th power.

    Integer Newton iteration on ``int`` only — no float ever touches this,
    which is the requirement: ``round(n ** (1 / q))`` agrees with the
    truth for small ``n`` and stops agreeing exactly where the replay
    matters, at operands with more significant bits than a double holds.
    The answer is CONFIRMED by exponentiation before it is returned, so a
    wrong iteration cannot yield a wrong root — only ``None``.
    """
    if n < 0 or q < 1:
        return None
    if n in (0, 1) or q == 1:
        return n
    x = 1 << -(-n.bit_length() // q)  # ceil(bits/q): an over-estimate
    while True:
        y = ((q - 1) * x + n // x ** (q - 1)) // q
        if y >= x:
            break
        x = y
    return x if x**q == n else None


def _int_text(n: int) -> str:
    """``n`` as decimal digits, or its BIT LENGTH when CPython refuses to
    render it.

    THE HAZARD :func:`stelling.smt._renderable` EXISTS FOR, met at the other
    end. CPython caps ``int`` -> ``str`` at ``sys.get_int_max_str_digits()``
    (4300 since 3.11) and RAISES ``ValueError`` past it. There the integer
    was being written into a script; here it is being written into a REFUSAL
    MESSAGE, where the raise is worse — a clean decline becomes a crash out
    of :func:`evaluate_predicate` / :func:`witness_is_valid`, both public,
    and the escalation that was about to record "witness not independently
    replayable" records a ``ValueError`` from a formatting expression
    instead. Nothing bounds the operand: it is a solver model value, and
    ``Fraction(3**10000, 2)`` (a 15850-bit numerator) is already over the
    cap.

    Same posture as ``_renderable``: detect by ATTEMPTING the conversion,
    never by raising ``sys.set_int_max_str_digits`` — a global process
    mutation, performed by a library, on behalf of a caller who did not ask
    for it. The fallback reports magnitude instead of digits, which is what
    a reader of a decline needs; 4300 digits of numerator identify nothing.
    """
    try:
        return str(n)
    except ValueError:
        return f"{'-' if n < 0 else ''}<{n.bit_length()}-bit integer>"


def fraction_text(fr: Fraction) -> str:
    """``fr`` rendered the way ``str(Fraction)`` renders it — ``n`` when the
    denominator is 1, ``n/d`` otherwise — with every term through
    :func:`_int_text`, so an unrenderable term degrades to its bit length
    rather than raising.

    PUBLIC because a solver model value is described on BOTH sides of the
    module boundary and there must be one renderer for it:
    ``solvers._require_valid_refutation`` attaches the same values to an
    :exc:`EmissionInfidelityError`, and it used bare ``str()``, so the
    box-escape alarm here returned its diagnosis safely and then died one
    statement later rendering the same value.

    **Public rather than imported privately, and the first reason given
    for that was wrong.** It was "nothing else in ``src/`` imports a
    private name across modules" — measured by AST, there are **50**, and
    ``smt.py`` alone takes thirteen from this module. The real reason is
    narrower and survives: this renderer is part of a *disclosure*
    contract, not an internal helper. What a verdict says about a model
    value is published surface, ``smt._renderable`` is the same discipline
    at the other end, and a rendering rule that two modules must agree on
    should be nameable by both. Note the counterpart restriction: only a
    MESSAGE may be rendered this way. ``Witness.values`` is data with a
    parsed contract, and ``make_validated_witness`` deliberately does NOT
    call this — it declines instead."""
    if fr.denominator == 1:
        return _int_text(fr.numerator)
    return f"{_int_text(fr.numerator)}/{_int_text(fr.denominator)}"


def _exact_rational_power(base: Fraction, p: int, q: int) -> Fraction:
    """``base ** (p / q)`` as an exact rational, or
    :exc:`ReplayDeclined` when that real is irrational.

    ``p/q`` is in lowest terms with ``q >= 2`` and ``p > 0`` (the
    admission guard refuses a negative rational exponent, and a
    denominator of 1 is the integer branch). Under those conditions
    ``base^(p/q)`` is rational exactly when ``base``'s numerator and
    denominator are both perfect ``q``-th powers, so the test IS the
    computation: extract both roots exactly, or decline.

    **The predecessor computed ``Fraction(float(base) ** exp_float)``** —
    a binary64 libm ``pow``, rounded, wrapped in a ``Fraction`` — while
    every REFUTED witness carried the sentence "confirmed by independent
    exact-rational replay (fractions.Fraction arithmetic, pure Python, no
    solver)". That sentence was false for these slices, and near the
    predicate boundary the rounding decided the answer (audit 0.2.0 S3).
    It also raised ``OverflowError`` on large operands, uncaught, which
    took out the whole escalation (M8); exact rational arithmetic has no
    overflow to raise.
    """
    if base < 0:
        # No real q-th root for even q, and the guard already refuses a
        # base interval reaching below zero — declining is the safe read
        # of a point that should not exist.
        raise ReplayDeclined(
            f"replay cannot evaluate x^({p}/{q}) at the negative base "
            f"{fraction_text(base)}: it has no real value for even q, and "
            f"jax computes NaN here"
        )
    root_num = _exact_integer_root(base.numerator, q)
    root_den = _exact_integer_root(base.denominator, q)
    if root_num is None or root_den is None:
        raise ReplayDeclined(
            f"{fraction_text(base)}^({p}/{q}) is irrational — "
            f"{_int_text(base.numerator)} and/or "
            f"{_int_text(base.denominator)} is not a perfect {q}-th power — "
            f"and the replay is exact rational arithmetic; it will not decide "
            f"this point with a rounded float"
        )
    return Fraction(root_num, root_den) ** p


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
            elif prim == "is_finite":
                # Under real semantics all values are finite rationals.
                (idx,) = _pair_elementwise(eqn)
                out = tuple(True for _ in idx)
            elif prim == "integer_pow":
                (idx,) = _pair_elementwise(eqn)
                y = int(params["y"])
                out = tuple(ins[0][i] ** y for i in idx)
            elif prim == "pow":
                # pow [base, exponent_literal]. The exponent arrives here
                # as the EXACT Fraction of the literal (_numeric_fraction
                # on the decoded binary64) — the replay never round-trips
                # it through a float, so it is reading the same real the
                # admission guard and the emission read.
                ia, ib = _pair_elementwise(eqn)
                exp = ins[1][ib[0]]
                if exp.denominator == 1:
                    y = exp.numerator
                    out = tuple(ins[0][ia[i]] ** y for i in range(n_out))
                else:
                    # Rational exponent: EXACT when the value is rational,
                    # and a refusal otherwise. Replaying a float here is
                    # what made check() raise on a correct emission
                    # (audit 0.2.0 S3) — the replay is the one thing that
                    # makes a REFUTED trustworthy, so it either does exact
                    # arithmetic or says it cannot.
                    p, q = exp.numerator, exp.denominator
                    out = tuple(
                        _exact_rational_power(ins[0][ia[i]], p, q)
                        for i in range(n_out)
                    )
            elif prim == "square":
                (idx,) = _pair_elementwise(eqn)
                out = tuple(_square_value(ins[0][i]) for i in idx)
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
                    # ReplayError, NOT ReplayDeclined, and the channel is the
                    # whole content of this branch. The emission writes a
                    # non-bool `convert_element_type` as the IDENTITY on its
                    # operand (:func:`stelling.smt.emit`, the
                    # `names[out.id] = tuple(ins[0][i] for i in idx)` line), so
                    # a script carrying a value-changing narrowing has the
                    # rounding simply ABSENT — it states a different function
                    # from the one the harness computes. A witness that reaches
                    # here therefore accuses the SCRIPT, which is exactly what
                    # ReplayError means. It is NOT the replay falling behind a
                    # correct emission, which is the other channel
                    # (:exc:`ReplayDeclined`) and which the rational-pow
                    # refusal above genuinely is: there the emitted
                    # `aux^q = x^p` means the obligation exactly and only this
                    # evaluator cannot finish it. Being unreachable is why a
                    # demotion to a decline is SILENT — reverting this one word
                    # left the whole suite green — so the channel itself is
                    # pinned, in tests/test_pow_audit_findings.py, together
                    # with the identity emission it rests on.
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

    **A returned string means the EMISSION is wrong.** When the replay
    instead REFUSES the point — it cannot produce the exact value at all
    — :exc:`ReplayDeclined` propagates out of here rather than being
    flattened into that string. The two are different findings and the
    caller routes them differently: an unfaithful emission is loud, an
    unreplayable witness degrades to UNKNOWN. Flattening them made
    ``check()`` raise ``EmissionInfidelityError`` at emissions that were
    correct (audit 0.2.0 S3).
    """
    for inp in sl.inputs:
        v = values.get(inp.name)
        if not isinstance(v, Fraction):
            # a missing or inexact value: the replay conjunct below names
            # it precisely (ReplayError), so membership defers
            continue
        # fraction_text, not str(): `v` is a SOLVER MODEL value, so its
        # terms are unbounded and `{v}` raises ValueError past CPython's
        # int -> str cap. The raise would land HERE, inside the message of
        # the LOUD alarm — the one that means the emitted problem does not
        # mean the obligation — replacing its diagnosis with a traceback
        # out of `fractions.py`. That is worse than the same hazard in a
        # decline message, not better, because this is the channel that
        # exists to be read. The bounds are `Fraction(float)` and cannot
        # reach the cap, but they go through the same helper so nobody has
        # to re-derive that to see the line is safe.
        if inp.lo != float("-inf") and v < Fraction(inp.lo):
            return (
                f"the model escapes the declared box ({inp.name} = "
                f"{fraction_text(v)} is below its declared lower bound "
                f"{fraction_text(Fraction(inp.lo))}); the box constraints "
                f"were part of the emitted problem"
            )
        if inp.hi != float("inf") and v > Fraction(inp.hi):
            return (
                f"the model escapes the declared box ({inp.name} = "
                f"{fraction_text(v)} is above its declared upper bound "
                f"{fraction_text(Fraction(inp.hi))}); the box constraints "
                f"were part of the emitted problem"
            )
    try:
        holds = evaluate_predicate(sl, values)
    except ReplayDeclined:
        raise  # the caller degrades to UNKNOWN; see this function's docstring
    except ReplayError as e:
        return f"the replay could not evaluate it ({e})"
    if holds:
        return (
            "the exact-rational replay found the predicate TRUE at that point"
        )
    return None
