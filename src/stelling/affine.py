# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Affine-form (zonotope) refinement of interval-undecided obligations.

The first relational domain: an affine form is ``center + Σ_i coef_i·ε_i
+ err`` with one noise symbol ``ε_i ∈ [-1, 1]`` per declared scalar input
element (and one per canonicalized nonlinear product residue), and
``err ≥ 0`` an unsigned bound absorbing every float rounding step the
form arithmetic performs. Where interval arithmetic forgets that the two
occurrences of ``x`` in ``x - x`` are the same input, the shared symbols
remember it — the input correlations intervals drop are exactly what
this domain keeps.

**The containment invariant, which everything below rests on:** the
concretization ``[center - Σ|coef_i| - err, center + Σ|coef_i| + err]``
(outward-rounded) always contains the true range of the expression over
the declared boxes. Both definite directions of the decision rule are
licensed by that over-approximation: the affine range CONTAINS the true
range, so range ⊆ [0, ∞) proves the obligation universally true over
the declared box (discharged) and range ⊆ (−∞, 0) proves it universally
false there (violated-over-set — the same set-level refutation class
interval judging uses, no witness). A refinement that can influence
VERIFIED and definite REFUTED is a soundness surface; the fidelity gauge
(:mod:`stelling.fidelity`) measures this module's claimed behaviors
against a mutation battery in the test suite.

**Rounding accounting.** Centers and coefficients are doubles, but no
bare float operation ever touches them: every arithmetic step is
computed as an exact rational (:class:`fractions.Fraction` — a double is
a dyadic rational) and snapped back to a double through the interval
module's own outward kernel :func:`stelling.interval._frac_bracket`,
with the snap distance accounted into ``err`` (exactly 0 when the float
result is exact — which is what lets ``x - x`` concretize to exactly
``[0, 0]`` rather than a bumped straddle). ``err`` accumulation itself
rounds upward through the same kernel. A center or coefficient beyond
the double range SATURATES at the largest finite double with the full
distance accounted into ``err`` through the same kernel — the measured
behavior of the snap (``float(Fraction)`` never returns a non-finite
double; the kernel maps overflow to the saturated bracket), sound
because the excess rides ``err`` — and every obligation whose
evaluation saturated carries a disclosure note, so a definite verdict
never rests on saturation silently. An ``err`` bound that itself
overflows the double range declines the whole obligation honestly.

**v1 scope, deliberately small and exact-where-linear**
(:data:`AFFINE_SUPPORTED`, asserted at import to be a subset of the
censused transfer registry — the refinement never judges what the census
has not admitted): add/sub/neg and multiply-by-constant are exact on
coefficients; ``reduce_sum`` is exact; ``mul`` of two forms carries its
nonlinear residue in a canonicalized product symbol (see
:func:`mul_forms`); ``integer_pow`` with y=2 is the self-product;
the structural ops route forms exactly. Everything else — division,
min/max, comparisons inside the slice, transcendentals, integer dtypes —
declines the WHOLE obligation loudly with the primitive quoted, and the
obligation falls through unchanged. ``semantics="ieee"`` declines
wholly: this domain models exact real arithmetic (ℝ), the same
arithmetic the real-mode verdict is about and the same relaxation the
SMT escalation already speaks — a binary64 claim is outside it.

**Opt-in, never on by default.** The pipeline keyword
(``check(..., refine="affine")``) follows the solver-opt-in precedent;
an unknown value raises eagerly at entry. With ``refine`` unset, nothing
in this module runs and every existing path is byte-identical.

Zero-dep and jax-free: pure Python over the transcribed IR, walking the
same validated obligation slices — and driving the same element
pairing/routing/grouping helpers — the SMT emission and the exact
replay use (:mod:`stelling.obligation`): one bookkeeping, now four
consumers, so the refinement cannot route an element differently from
the propagation that judged the obligation.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from fractions import Fraction

from stelling import interval as iv
from stelling import ir
from stelling.obligation import (
    DeclinedObligation,
    ObligationSlice,
    _Decline as _SliceDecline,
    _group_reduce_sum,
    _pair_elementwise,
    _route_structural,
    slice_unknown_obligations,
)
from stelling.propagate import (
    ObligationReport,
    Propagation,
    TRANSFERS,
    interval_env,
)

__all__ = [
    "AFFINE_SUPPORTED",
    "AffineForm",
    "RefinementReport",
    "SliceDecision",
    "SymbolTable",
    "add_forms",
    "concretization",
    "decide_slice",
    "declared_form",
    "mul_forms",
    "neg_form",
    "point_form",
    "pow2_form",
    "refine_propagation",
    "sub_forms",
    "sum_forms",
]

# The registry: the v1 primitive set, explicit and visible. Linear ops
# are exact on coefficients (rounding accounted); mul carries its
# canonicalized residue symbol; integer_pow is admitted only as the y=2
# self-product; the structural ops are exact element routing of forms
# (an output element IS its source element's form — the same aliasing
# convention the emission uses). Everything else declines the whole
# obligation with the primitive quoted.
AFFINE_SUPPORTED = frozenset(
    {
        "add",
        "sub",
        "neg",
        "mul",
        "integer_pow",
        "reduce_sum",
        # structural routing (exact, no arithmetic)
        "reshape",
        "slice",
        "squeeze",
        "concatenate",
        "stack",
        "broadcast_in_dim",
    }
)

_AFFINE_STRUCTURAL = frozenset(
    {"reshape", "slice", "squeeze", "concatenate", "stack", "broadcast_in_dim"}
)

# The L22-style totality discipline: the refinement never judges a
# primitive the censused transfer registry has not admitted — affine is
# a refinement OF the interval judgment, not a side door for semantic
# extension. Asserted at import so the invariant cannot rot silently;
# the test suite additionally requires a dedicated behavior test per
# registry name.
if not AFFINE_SUPPORTED <= set(TRANSFERS):
    raise RuntimeError(
    "AFFINE_SUPPORTED must be a subset of the censused transfer registry: "
    f"unadmitted {sorted(AFFINE_SUPPORTED - set(TRANSFERS))}"
)

# The two decision comparisons. Closed half-spaces (ge/le) ONLY — a
# strict SUBSET of what interval judging decides (interval also settles
# definite gt/lt/eq/ne roots); every other root primitive declines with
# the primitive quoted rather than being approximated.
_DECISION_ROOTS = ("ge", "le")

DISCHARGED_BY_AFFINE = (
    "discharged by affine refinement (zonotope over the declared boxes, "
    "outward-rounded)"
)

IEEE_AFFINE_DECLINE = (
    "semantics='ieee' — the affine domain models exact real arithmetic "
    "(ℝ), the same relaxation the SMT escalation speaks; a binary64 "
    "execution claim is outside it, so every obligation declines under "
    "ieee semantics in v1"
)

CONSTRAINED_AFFINE_DECLINE = (
    "the propagation constrained {n} assume(s) — the refinement judges "
    "over the DECLARED boxes and would ignore the assumed "
    "preconditions, so it declines wholly (the same refusal shape the "
    "solver escalation applies under constraining assumes)"
)

_FR0 = Fraction(0)


class _Decline(Exception):
    """Internal control flow only; always converted into a quoted decline."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# -- the outward-rounded snap --------------------------------------------------


def _snap(fr: Fraction) -> tuple[float, Fraction, bool]:
    """An exact rational as ``(double, snap-error bound, saturated)``.

    The double is an end of the tightest sound bracket the interval
    module computes (:func:`stelling.interval._frac_bracket` — the same
    kernel ``integer_pow`` endpoints ride on), and the second element is
    an exact upper bound on ``|fr - double|`` — EXACTLY 0 when the
    rational is representable, which is what keeps exact cancellations
    exact. The double is ALWAYS finite: a value beyond the double range
    saturates at the largest finite double (``float(Fraction)`` raises
    on overflow and the kernel maps that to the saturated bracket, so a
    non-finite return is unconstructable) with the full distance in the
    error bound — sound, because the excess rides ``err`` — and the
    third element flags it so the obligation's notes can disclose that
    saturation participated.
    """
    blo, bhi = iv._frac_bracket(fr)
    saturated = not (math.isfinite(blo) and math.isfinite(bhi))
    z = blo if math.isfinite(blo) else bhi
    return z, abs(fr - Fraction(z)), saturated


def _up_float(fr: Fraction) -> float:
    """The smallest sound double upper bound of an exact rational —
    ``err`` accumulation always rounds this way (upward)."""
    return iv._frac_bracket(fr)[1]


@dataclass(frozen=True)
class AffineForm:
    """One scalar affine form: ``center + Σ coef·ε + err``, ``ε ∈ [-1, 1]``.

    ``coeffs`` maps symbol id -> coefficient, stored sorted with zero
    coefficients dropped, so value-identical forms are representation-
    identical (the canonicalization the product cache and the commuted-
    product cancellation rest on). ``err`` is the unsigned accumulated
    bound; every numeric field is finite by construction (an ``err``
    that overflows declines before an instance exists; centers and
    coefficients saturate through the snap kernel, never non-finite).
    ``saturated`` records whether any snap in this form's dataflow
    saturated at the double range — bookkeeping for the disclosure note,
    deliberately OUTSIDE the product-cache content keys (saturation
    history does not change the value a determinate form denotes).
    """

    center: float
    coeffs: tuple[tuple[int, float], ...]
    err: float
    saturated: bool = False

    def __post_init__(self) -> None:
        # construction-path validation (the same lesson ir.py's
        # dataclasses learned): no route may build a form these
        # invariants do not hold for.
        if not math.isfinite(self.center) or not math.isfinite(self.err):
            raise ValueError(
                f"AffineForm with non-finite center/err "
                f"({self.center!r}, {self.err!r}) — the arithmetic layer "
                f"must decline before construction"
            )
        if self.err < 0.0:
            raise ValueError(f"AffineForm with negative err {self.err!r}")
        last = None
        for s, k in self.coeffs:
            if not math.isfinite(k) or k == 0.0:
                raise ValueError(
                    f"AffineForm coefficient {k!r} for symbol {s} — zero "
                    f"and non-finite coefficients may not be stored"
                )
            if last is not None and s <= last:
                raise ValueError(
                    f"AffineForm coefficients not strictly sorted at "
                    f"symbol {s}"
                )
            last = s


def _snap_form(
    c_fr: Fraction,
    coefs_fr: dict[int, Fraction],
    err_fr: Fraction,
    saturated: bool = False,
) -> AffineForm:
    """Snap exact-rational form data to doubles, folding every snap
    distance into ``err`` (upward) and OR-ing the operands' saturation
    bit (``saturated``) with this form's own snap saturations. Centers
    and coefficients saturate soundly (the excess rides ``err``); an
    ``err`` bound that itself overflows declines — the one reachable
    overflow, measured."""
    extra = _FR0
    center, dc, sat = _snap(c_fr)
    extra += dc
    saturated = saturated or sat
    items: list[tuple[int, float]] = []
    for s in sorted(coefs_fr):
        k_fr = coefs_fr[s]
        if not k_fr:
            continue
        k, dk, sat = _snap(k_fr)
        extra += dk
        saturated = saturated or sat
        if k != 0.0:  # a coefficient snapped to 0 lives on in err (dk = |k_fr|)
            items.append((s, k))
    err = _up_float(err_fr + extra)
    if not math.isfinite(err):
        raise _Decline(
            "the accumulated err bound overflowed the double range (inf "
            "err) — the obligation declines as undecided rather than "
            "carry an unrepresentable bound"
        )
    return AffineForm(
        center=center, coeffs=tuple(items), err=err, saturated=saturated
    )


def point_form(value: float) -> AffineForm:
    """The exact point form of a finite double: no symbols, err 0."""
    v = float(value)
    if not math.isfinite(v):
        raise _Decline(f"non-finite constant {v!r} has no affine form")
    return AffineForm(center=v, coeffs=(), err=0.0)


def _exact_point_form(fr: Fraction) -> AffineForm:
    """The point form of an exact rational (an int constant above 2**53
    is a rational the double center only approximates — the snap
    distance is accounted, exactly as for every other center)."""
    return _snap_form(fr, {}, _FR0)


def declared_form(lo: float, hi: float, symbol: int) -> AffineForm:
    """The form of one declared input element over ``[lo, hi]``:
    ``center + r·ε_symbol`` with the radius rounded upward so the real
    segment ``center ± r`` contains the whole declared interval. A point
    declaration is a symbol-free point. Non-finite bounds decline — the
    domain models bounded declared boxes only (the widened re-check's
    ``(-inf, inf)`` declarations land here and decline honestly)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise _Decline(
            f"input declaration bounds [{lo}, {hi}] are not finite: the "
            f"affine domain models bounded declared boxes only"
        )
    if lo == hi:
        return point_form(lo)
    c_fr = (Fraction(lo) + Fraction(hi)) / 2
    # the midpoint of two finite doubles is at most the largest finite
    # double in magnitude, so this snap can neither overflow nor
    # saturate; the returned flag is False by construction
    center, _, _ = _snap(c_fr)
    # the radius is measured FROM the snapped center, so the center's
    # snap distance is inside it; rounded upward so containment is real
    r_fr = max(Fraction(hi) - Fraction(center), Fraction(center) - Fraction(lo))
    r = _up_float(r_fr)
    if not math.isfinite(r):
        raise _Decline(
            f"declaration radius of [{lo}, {hi}] left the double range"
        )
    return AffineForm(center=center, coeffs=((symbol, r),), err=0.0)


# -- form arithmetic -----------------------------------------------------------


def add_forms(f: AffineForm, g: AffineForm) -> AffineForm:
    """Exact-on-coefficients addition; shared symbols merge exactly (the
    cancellation channel); every snap distance is accounted into err."""
    coefs = {s: Fraction(k) for s, k in f.coeffs}
    for s, k in g.coeffs:
        coefs[s] = coefs.get(s, _FR0) + Fraction(k)
    return _snap_form(
        Fraction(f.center) + Fraction(g.center),
        coefs,
        Fraction(f.err) + Fraction(g.err),
        saturated=f.saturated or g.saturated,
    )


def sub_forms(f: AffineForm, g: AffineForm) -> AffineForm:
    """``f - g``: negation is exact, so this is :func:`add_forms` on the
    exact negation. Identical forms cancel to the exact zero form —
    center 0, no coefficients, err ``f.err + g.err`` (unsigned bounds
    never cancel; they are exactly 0 in the exact-arithmetic cases)."""
    return add_forms(f, neg_form(g))


def neg_form(f: AffineForm) -> AffineForm:
    """Negation: exact (IEEE negation is exact) — no snap, no err growth."""
    return AffineForm(
        center=-f.center,
        coeffs=tuple((s, -k) for s, k in f.coeffs),
        err=f.err,
        saturated=f.saturated,
    )


def _rad_fr(f: AffineForm) -> Fraction:
    """The exact deviation radius ``Σ|coef| + err`` of a form."""
    total = Fraction(f.err)
    for _, k in f.coeffs:
        total += abs(Fraction(k))
    return total


def _is_point(f: AffineForm) -> bool:
    return not f.coeffs and f.err == 0.0


def _mul_residual(
    f: AffineForm, g: AffineForm, deltas: Fraction
) -> Fraction:
    """The exact bound on a product's deviation from its snapped affine
    part: ``|f_c|·err_g + |g_c|·err_f + rad(f)·rad(g)`` plus the
    product's own snap distances — the claimed residue accounting, one
    named seam so the fidelity battery can mutate exactly this claim."""
    return (
        abs(Fraction(f.center)) * Fraction(g.err)
        + abs(Fraction(g.center)) * Fraction(f.err)
        + _rad_fr(f) * _rad_fr(g)
        + deltas
    )


def _element_true(lo: float) -> bool:
    """The discharge test for one element: ``lo ≥ 0`` — the CLOSED
    comparison, so an exactly-cancelled form (range exactly [0, 0])
    discharges ``g ≥ 0``. The claimed decision rule; a named seam for
    the fidelity battery."""
    return lo >= 0.0


def _element_false(hi: float) -> bool:
    """The violation test for one element: ``hi < 0`` — STRICT, because
    ``hi == 0`` leaves the point ``g = 0`` satisfying ``g ≥ 0`` inside
    the range, and a definite set-level refutation may not include a
    satisfying point. The claimed decision rule; a named seam for the
    fidelity battery."""
    return hi < 0.0


class SymbolTable:
    """Symbol allocation for one obligation evaluation: the declared-
    input symbols first, then one symbol per CANONICALIZED nonlinear
    product residue.

    The product cache is the load-bearing canonicalization: ``mul(a, b)``
    and ``mul(b, a)`` must yield the IDENTICAL affine form — same
    coefficients (exact rational arithmetic makes the linear part
    symmetric by construction), same residue accounting, and the SAME
    residue symbol — so that syntactically-commuted products cancel
    exactly in differences. This matters because commuted-product
    obligations (``x·y - y·x ≥ 0``) straddle under plain intervals for
    lack of exactly this: intervals treat the two products as unrelated
    ranges, and the difference inflates to ``±2·rad`` instead of
    collapsing to 0. The cache key is the UNORDERED pair of operand
    keys, where an operand's key is its full form content when the form
    is determinate (err 0 — the form then IS one real-valued function of
    the ε's, so equal keys mean equal values and sharing a residue
    symbol is sound) and its slice identity (variable id, element)
    otherwise (an err-carrying form hides an unknown residual, and only
    the SAME occurrence is guaranteed the same value).
    """

    def __init__(self, n_inputs: int) -> None:
        self._next = n_inputs
        self._products: dict[tuple, int] = {}

    def product_symbol(self, key_a: tuple, key_b: tuple) -> int:
        key = tuple(sorted((key_a, key_b)))
        got = self._products.get(key)
        if got is None:
            got = self._next
            self._next += 1
            self._products[key] = got
        return got


def _operand_key(f: AffineForm, atom: ir.Atom, element: int) -> tuple:
    """The product-cache key of one mul operand element (see
    :class:`SymbolTable`)."""
    if f.err == 0.0:
        return (0, (f.center, f.coeffs))
    if isinstance(atom, ir.Var):
        return (1, (atom.id, element))
    # unreachable: literal/const operands decode to err-0 point forms
    raise _Decline(
        "a non-determinate operand without a variable identity reached "
        "mul — internal invariant violated; declining"
    )


def mul_forms(
    table: SymbolTable,
    f: AffineForm,
    key_f: tuple,
    g: AffineForm,
    key_g: tuple,
) -> AffineForm:
    """The affine product. The affine part multiplies out exactly
    (``f_c·g + g_c·f − f_c·g_c`` on the coefficients, in exact rational
    arithmetic — symmetric by construction, so operand order cannot
    change it); the WHOLE deviation from that snapped affine part —

        ``|f_c|·err_g + |g_c|·err_f + rad(f)·rad(g)`` (the nonlinear
        residue, with ``rad = Σ|coef| + err``) plus every snap distance
        of the product's own center/coefficients —

    is carried on the canonicalized product symbol from ``table``
    (upward-rounded coefficient, err 0), NOT in unsigned ``err``. That
    placement is load-bearing, not cosmetic: the deviation is one
    determinate real-valued function of the ε's per product occurrence,
    so the identical product — including its commuted spelling — reuses
    the same symbol and cancels EXACTLY in differences, where an
    unsigned err bound would add instead of cancel and leave a residual
    straddle. A point operand short-circuits to the exact linear
    scaling: no residue, no symbol (multiply-by-constant is linear).

    ``ε² ∈ [0, 1]`` sharpening for self-products is deliberately NOT
    performed: it would be a claimed behavior needing its own gauge, and
    v1 claims only the symmetric residue.
    """
    if _is_point(g) or _is_point(f):
        if _is_point(g):
            scale_fr, other = Fraction(g.center), f
        else:
            scale_fr, other = Fraction(f.center), g
        coefs = {s: scale_fr * Fraction(k) for s, k in other.coeffs}
        return _snap_form(
            scale_fr * Fraction(other.center),
            coefs,
            abs(scale_fr) * Fraction(other.err),
            saturated=f.saturated or g.saturated,
        )
    fc, gc = Fraction(f.center), Fraction(g.center)
    center, deltas, saturated = _snap(fc * gc)
    saturated = saturated or f.saturated or g.saturated
    coefs_fr: dict[int, Fraction] = {}
    for s, k in f.coeffs:
        coefs_fr[s] = gc * Fraction(k)
    for s, k in g.coeffs:
        coefs_fr[s] = coefs_fr.get(s, _FR0) + fc * Fraction(k)
    items: list[tuple[int, float]] = []
    for s in sorted(coefs_fr):
        k, dk, sat = _snap(coefs_fr[s])
        deltas += dk
        saturated = saturated or sat
        if k != 0.0:
            items.append((s, k))
    residual = _mul_residual(f, g, deltas)
    if residual:
        sym = table.product_symbol(key_f, key_g)
        if any(s == sym for s, _ in items):  # cannot happen: symbols are
            # fresh or belong to this same product, never to an operand
            raise _Decline(
                "product symbol collided with a linear coefficient — "
                "internal invariant violated; declining"
            )
        rcoef = _up_float(residual)
        if not math.isfinite(rcoef):
            raise _Decline(
                "the product residue bound overflowed the double range "
                "(inf residue) — the obligation declines as undecided "
                "rather than carry an unrepresentable bound"
            )
        items = sorted(items + [(sym, rcoef)])
    return AffineForm(
        center=center, coeffs=tuple(items), err=0.0, saturated=saturated
    )


def pow2_form(table: SymbolTable, f: AffineForm, key_f: tuple) -> AffineForm:
    """``integer_pow`` with y=2, as the self-product ``f·f`` — the same
    canonical residue symbol an explicit ``x·x`` product gets, so the two
    spellings cancel against each other exactly."""
    return mul_forms(table, f, key_f, f, key_f)


def sum_forms(forms: list[AffineForm]) -> AffineForm:
    """Exact n-ary sum (reduce_sum's kernel): centers, coefficients, and
    errs are accumulated as exact rationals and snapped ONCE, so the
    result is order-free — in ℝ every association order denotes the same
    number, and exact rational arithmetic keeps it that way. The empty
    sum is exactly 0, matching the interval transfer's fold identity."""
    c_fr = _FR0
    err_fr = _FR0
    coefs: dict[int, Fraction] = {}
    for f in forms:
        c_fr += Fraction(f.center)
        err_fr += Fraction(f.err)
        for s, k in f.coeffs:
            coefs[s] = coefs.get(s, _FR0) + Fraction(k)
    return _snap_form(
        c_fr, coefs, err_fr, saturated=any(f.saturated for f in forms)
    )


def concretization(f: AffineForm) -> tuple[float, float]:
    """The outward-rounded concretization ``[center - Σ|coef| - err,
    center + Σ|coef| + err]``: computed in exact rationals and bracketed
    outward through :func:`stelling.interval._frac_bracket`, so an
    exactly-zero form concretizes to exactly ``[0, 0]`` (no unconditional
    bump — the kernel bumps only where the double cannot represent the
    endpoint). Always contains the true range of the value the form
    stands for (the containment invariant)."""
    total = _rad_fr(f)
    c = Fraction(f.center)
    lo = iv._frac_bracket(c - total)[0]
    hi = iv._frac_bracket(c + total)[1]
    return lo, hi


# -- the obligation-slice evaluator -------------------------------------------


@dataclass(frozen=True)
class SliceDecision:
    """The affine decision for one obligation slice.

    ``status``: ``"discharged"`` | ``"violated-over-set"`` |
    ``"unknown"`` | ``"declined"``. ``detail`` is the mechanism note
    (or, for a decline, the quoted reason WITHOUT the standard prefix).
    ``ranges`` are the per-element concretizations of the obligation
    slack ``g`` (``g ≥ 0`` being the decided claim); ``ops`` the
    primitives actually evaluated, sorted. ``saturated`` records whether
    any endpoint snap in the slack's dataflow saturated at the double
    range — the refinement's notes disclose it per obligation.
    """

    status: str
    detail: str
    ranges: tuple[tuple[float, float], ...]
    ops: tuple[str, ...]
    saturated: bool = False


def _decline_decision(reason: str, ops: tuple[str, ...] = ()) -> SliceDecision:
    return SliceDecision(status="declined", detail=reason, ranges=(), ops=ops)


def _declared_arrays(
    sl: ObligationSlice,
) -> tuple[dict[int, tuple[AffineForm, ...]], int]:
    """Input forms per declared variable: one symbol per element, in
    ``sl.inputs`` order (declaration order, then flat C-order element —
    the same naming discipline the emission's ``x{k}_{i}`` constants
    follow)."""
    env: dict[int, tuple[AffineForm, ...]] = {}
    per_var: dict[int, list[AffineForm]] = {}
    order: list[int] = []
    for symbol, inp in enumerate(sl.inputs):
        if inp.var_id not in per_var:
            per_var[inp.var_id] = []
            order.append(inp.var_id)
        per_var[inp.var_id].append(declared_form(inp.lo, inp.hi, symbol))
    for vid in order:
        env[vid] = tuple(per_var[vid])
    return env, len(sl.inputs)


def _const_forms(val) -> tuple[AffineForm, ...]:
    vals = val if isinstance(val, tuple) else (val,)
    forms = []
    for v in vals:
        if isinstance(v, bool):
            raise _Decline(
                "boolean constant in numeric dataflow has no affine form"
            )
        if isinstance(v, int):
            forms.append(_exact_point_form(Fraction(v)))
        else:
            forms.append(point_form(v))
    return tuple(forms)


def _evaluate_root(
    table: SymbolTable,
    producer: ir.JaxprEqn,
    read,
) -> tuple[AffineForm, ...]:
    """The slack forms ``g`` of the root comparison, per element:
    ``ge(a, b)`` decides ``a - b ≥ 0`` and ``le(a, b)`` decides
    ``b - a ≥ 0`` — closed half-spaces only, exactly the definite forms
    interval judging decides for closed comparisons."""
    prim = producer.primitive
    if prim in ("gt", "lt"):
        raise _Decline(
            f"root comparison {prim!r} is a strict half-space; the v1 "
            f"decision rule covers closed half-spaces (ge/le) only"
        )
    if prim not in _DECISION_ROOTS:
        raise _Decline(
            f"root primitive {prim!r} is outside the closed-half-space "
            f"decision rule (ge/le)"
        )
    try:
        ia, ib = _pair_elementwise(producer)
    except _SliceDecline as d:
        raise _Decline(d.reason) from d
    lhs = read(producer.invars[0])
    rhs = read(producer.invars[1])
    n = len(ia)
    if prim == "ge":
        return tuple(sub_forms(lhs[ia[i]], rhs[ib[i]]) for i in range(n))
    return tuple(sub_forms(rhs[ib[i]], lhs[ia[i]]) for i in range(n))


def decide_slice(sl: ObligationSlice) -> SliceDecision:
    """Evaluate one validated obligation slice in the affine domain and
    decide it — or decline it whole, loudly, with the reason quoted.
    Never raises on a validated slice; declines are values."""
    ops: set[str] = set()
    try:
        if isinstance(sl.root, ir.Literal):
            raise _Decline(
                "the assert operand is a constant, not a computed "
                "comparison; nothing for the affine domain to refine"
            )
        producer = None
        for eqn in sl.eqns:
            if any(out.id == sl.root.id for out in eqn.outvars):
                producer = eqn
        if producer is None:
            raise _Decline(
                "the assert operand has no producing equation in the "
                "slice; the decision rule needs a closed comparison root"
            )
        env, n_inputs = _declared_arrays(sl)
        for vid, val in sl.consts:
            env[vid] = _const_forms(val)
        table = SymbolTable(n_inputs)

        def read(atom: ir.Atom) -> tuple[AffineForm, ...]:
            if isinstance(atom, ir.Literal):
                from stelling.obligation import _decode_elements

                try:
                    return _const_forms(tuple(_decode_elements(atom.val)))
                except _SliceDecline as d:
                    raise _Decline(d.reason) from d
            got = env.get(atom.id)
            if got is None:
                raise _Decline(
                    f"variable {atom.id} is unbound in the affine walk "
                    f"(slice/evaluator disagreement)"
                )
            return got

        for eqn in sl.eqns:
            if eqn is producer:
                continue
            prim = eqn.primitive
            if prim not in AFFINE_SUPPORTED:
                raise _Decline(
                    f"primitive {prim!r} is outside AFFINE_SUPPORTED"
                )
            params = eqn.params_dict()
            try:
                if prim in _AFFINE_STRUCTURAL:
                    routes = _route_structural(eqn)
                    ins = [read(a) for a in eqn.invars]
                    out = tuple(ins[op][src] for op, src in routes)
                elif prim == "reduce_sum":
                    groups = _group_reduce_sum(eqn)
                    (src,) = (read(eqn.invars[0]),)
                    out = tuple(
                        sum_forms([src[i] for i in group]) for group in groups
                    )
                elif prim == "neg":
                    (idx,) = _pair_elementwise(eqn)
                    src = read(eqn.invars[0])
                    out = tuple(neg_form(src[i]) for i in idx)
                elif prim == "integer_pow":
                    y = params.get("y")
                    if y != 2:
                        raise _Decline(
                            f"'integer_pow' exponent y={y!r} is outside "
                            f"AFFINE_SUPPORTED (v1 models only y=2, the "
                            f"self-product)"
                        )
                    (idx,) = _pair_elementwise(eqn)
                    src = read(eqn.invars[0])
                    atom = eqn.invars[0]
                    out = tuple(
                        pow2_form(table, src[i], _operand_key(src[i], atom, i))
                        for i in idx
                    )
                elif prim in ("add", "sub"):
                    ia, ib = _pair_elementwise(eqn)
                    fa = read(eqn.invars[0])
                    fb = read(eqn.invars[1])
                    op = add_forms if prim == "add" else sub_forms
                    out = tuple(
                        op(fa[ia[i]], fb[ib[i]]) for i in range(len(ia))
                    )
                elif prim == "mul":
                    ia, ib = _pair_elementwise(eqn)
                    fa = read(eqn.invars[0])
                    fb = read(eqn.invars[1])
                    a_atom, b_atom = eqn.invars
                    out = tuple(
                        mul_forms(
                            table,
                            fa[ia[i]],
                            _operand_key(fa[ia[i]], a_atom, ia[i]),
                            fb[ib[i]],
                            _operand_key(fb[ib[i]], b_atom, ib[i]),
                        )
                        for i in range(len(ia))
                    )
                else:  # pragma: no cover - registry and dispatch in sync
                    raise _Decline(
                        f"primitive {prim!r} has no affine evaluation rule"
                    )
            except _SliceDecline as d:
                raise _Decline(f"{prim!r}: {d.reason}") from d
            ops.add(prim)
            for outvar in eqn.outvars:
                env[outvar.id] = out

        ops.add(producer.primitive)
        g = _evaluate_root(table, producer, read)
        ranges = tuple(concretization(f) for f in g)
        saturated = any(f.saturated for f in g)
    except _Decline as d:
        return _decline_decision(d.reason, tuple(sorted(ops)))

    n = len(ranges)
    n_true = sum(1 for lo, _ in ranges if _element_true(lo))
    n_false = sum(1 for _, hi in ranges if _element_false(hi))
    if n_true == n:
        status, detail = "discharged", DISCHARGED_BY_AFFINE
    elif n_false:
        status = "violated-over-set"
        detail = (
            f"definitely false over the declared box by affine refinement "
            f"(zonotope over the declared boxes, outward-rounded): the "
            f"concretized slack lies strictly below 0 for {n_false}/{n} "
            f"element(s) — set-level, no witness"
        )
    else:
        lo = min(lo for lo, _ in ranges)
        hi = max(hi for _, hi in ranges)
        status = "unknown"
        detail = (
            f"affine refinement ran and did not separate: concretized "
            f"slack range [{lo}, {hi}] straddles 0 "
            f"({n - n_true}/{n} element(s) undecided)"
        )
    return SliceDecision(
        status=status,
        detail=detail,
        ranges=ranges,
        ops=tuple(sorted(ops)),
        saturated=saturated,
    )


# -- the pipeline refinement step ---------------------------------------------


@dataclass(frozen=True)
class RefinementReport:
    """What the refinement did — the assemblers derive the stamped line
    and the truthful solver-absence wording from this record, never from
    narration."""

    domain: str
    attempted: tuple[int, ...]
    discharged: tuple[int, ...]
    violated: tuple[int, ...]
    declined: tuple[tuple[int, str], ...]
    undecided: tuple[int, ...]
    ops_used: tuple[str, ...]

    @property
    def decided(self) -> int:
        return len(self.discharged) + len(self.violated)

    def stamp_line(self) -> str:
        # "enabled", not "ran": the pre-slice-decline and zero-unknown
        # cases ("0 obligation(s) attempted") must read naturally — the
        # counts carry the truth about what actually happened
        ops = ", ".join(self.ops_used) if self.ops_used else "none"
        return (
            f"affine refinement enabled (domain={self.domain} over the "
            f"declared boxes; registry of {len(AFFINE_SUPPORTED)} "
            f"primitives; {len(self.attempted)} obligation(s) attempted: "
            f"{len(self.discharged)} discharged, {len(self.violated)} "
            f"violated, {len(self.declined)} declined, "
            f"{len(self.undecided)} undecided; ops used: {ops})"
        )

    def reword_arithmetic(self, mode: str) -> str:
        """The stamp's ``arithmetic:`` line with the deciding abstraction
        named when the affine domain decided at least one obligation —
        the endpoint representation is then not interval arithmetic
        alone. Same single-derivation-point discipline as
        :meth:`reword_absence`."""
        if not self.decided:
            return mode
        return mode + (
            " + affine/zonotope refinement (stelling.affine, same "
            "outward kernel)"
        )

    def reword_absence(self, reason: str) -> str:
        """The solver-absence line with the deciding layers named
        truthfully: a line claiming "interval arithmetic alone" may not
        stand when the refinement decided anything. The rewording is the
        assemblers' single derivation point for this fact — both call
        here, so they cannot drift from each other."""
        if not self.decided:
            return reason + (
                f" (affine refinement was enabled and decided nothing: "
                f"{len(self.attempted)} obligation(s) attempted)"
            )
        if "interval arithmetic alone" in reason:
            return reason.replace(
                "interval arithmetic alone",
                f"interval arithmetic with affine (zonotope) refinement "
                f"— {self.decided} obligation(s) decided by the affine "
                f"domain",
            )
        return reason


def _decline_all(
    propagation: Propagation, unknown: list[ObligationReport], reason: str
) -> tuple[Propagation, RefinementReport]:
    notes = tuple(
        f"assert #{o.index}: affine refinement declined: {reason}"
        for o in unknown
    )
    report = RefinementReport(
        domain="affine",
        attempted=tuple(o.index for o in unknown),
        discharged=(),
        violated=(),
        declined=tuple((o.index, reason) for o in unknown),
        undecided=(),
        ops_used=(),
    )
    return (
        dataclasses.replace(propagation, notes=propagation.notes + notes),
        report,
    )


def refine_propagation(
    closed: ir.ClosedJaxpr, propagation: Propagation
) -> tuple[Propagation, RefinementReport]:
    """Attempt the affine refinement on every obligation interval judging
    left ``unknown``; return the refined :class:`Propagation` (statuses
    replaced only where the refinement DECIDED — discharged or violated
    per the containment-licensed rule; everything else byte-identical,
    with declines and non-separations quoted as appended notes) plus the
    :class:`RefinementReport` the verdict assemblers stamp from.

    Refuses to judge outside its model: ``semantics="ieee"`` declines
    wholly (this domain speaks ℝ), and a propagation that CONSTRAINED
    any assume declines wholly (the refinement reads the declared boxes
    and would ignore the precondition — the solver escalation's refusal
    shape, mirrored).
    """
    unknown = [o for o in propagation.obligations if o.status == "unknown"]
    empty = RefinementReport(
        domain="affine",
        attempted=(),
        discharged=(),
        violated=(),
        declined=(),
        undecided=(),
        ops_used=(),
    )
    if not unknown:
        return propagation, empty
    if propagation.semantics != "real":
        return _decline_all(propagation, unknown, IEEE_AFFINE_DECLINE)
    if propagation.coverage.constrained:
        return _decline_all(
            propagation,
            unknown,
            CONSTRAINED_AFFINE_DECLINE.format(
                n=propagation.coverage.constrained
            ),
        )

    env = interval_env(closed)
    decisions: dict[int, tuple[str, str]] = {}
    declined: list[tuple[int, str]] = []
    undecided: list[int] = []
    discharged: list[int] = []
    violated: list[int] = []
    notes: list[str] = []
    ops: set[str] = set()
    for item in slice_unknown_obligations(closed, propagation, env):
        if isinstance(item, DeclinedObligation):
            reason = f"the obligation slice is unavailable: {item.reason}"
            declined.append((item.index, reason))
            notes.append(
                f"assert #{item.index}: affine refinement declined: {reason}"
            )
            continue
        decision = decide_slice(item)
        ops.update(decision.ops)
        if decision.saturated and decision.status != "declined":
            # the F3 disclosure: an evaluation that saturated says so per
            # obligation, so a definite verdict never rests on saturation
            # silently (the saturation itself is sound — the excess is in
            # the accounted err)
            notes.append(
                f"assert #{item.index}: endpoint computation saturated at "
                f"the double range; the accounted err covers the excess"
            )
        if decision.status == "declined":
            declined.append((item.index, decision.detail))
            notes.append(
                f"assert #{item.index}: affine refinement declined: "
                f"{decision.detail}"
            )
        elif decision.status == "unknown":
            undecided.append(item.index)
            notes.append(f"assert #{item.index}: {decision.detail}")
        else:
            decisions[item.index] = (decision.status, decision.detail)
            if decision.status == "discharged":
                discharged.append(item.index)
            else:
                violated.append(item.index)

    obligations = tuple(
        dataclasses.replace(
            o,
            status=decisions[o.index][0],
            detail=decisions[o.index][1],
        )
        if o.index in decisions
        else o
        for o in propagation.obligations
    )
    report = RefinementReport(
        domain="affine",
        attempted=tuple(o.index for o in unknown),
        discharged=tuple(discharged),
        violated=tuple(violated),
        declined=tuple(declined),
        undecided=tuple(undecided),
        ops_used=tuple(sorted(ops)),
    )
    refined = dataclasses.replace(
        propagation,
        obligations=obligations,
        notes=propagation.notes + tuple(notes),
    )
    return refined, report
