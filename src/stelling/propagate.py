# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Forward interval propagation over :mod:`stelling.ir`.

The autodidax pattern: walk the equations of a transcribed query in order,
mapping each primitive through a transfer function over
:class:`stelling.interval.IntervalArray`. Consumes the IR, never jaxprs.

Scope, held deliberately (design/e2a-registration.md): no widening, no
fixpoints, no cond/scan descent, no solver. The transfer registry contains
exactly the primitives the target census returned
(`design/primitive-census.md`, "The target census") plus the three harness
primitives, plus the closed pytree-probe registration round (abs, eq, ne,
and, or, stop_gradient, reshape, pow, reduce_or, and the scalar-selector /
rank-broadcast forms of already-registered transfers), plus one
allowed-by-census structural addition from the maddening HeatNode trace
(``scatter`` in its static-index ``x.at[k].set(v)`` form only), plus two
allowed-by-census structural additions from the MIME fvm laplacian trace
(``gather`` in its static-index leading-axis row form only, and
``transpose`` — both pure data movement with exact semantics). Everything
else falls to ⊤ — soundly, with coverage recording exactly how much fell.

Every transfer declares an assumption tier (design commitment 5):
``exact`` (no arithmetic, or arithmetic with no rounding), ``sound``
(outward-rounded interval arithmetic), or ``sound-libm`` (outward-rounded
around a faithfully-rounded libm call — carries
:data:`stelling.interval.EXP_LIBM_ASSUMPTION`). The tiers of every
transfer *used* ride into the verdict stamp.

Obligations are ``stelling_assert`` equations. Their statuses:
``discharged`` (predicate definitely true over the declared box),
``unknown`` (interval too wide to decide — *our* imprecision), or
``violated-over-set`` (definitely false over the propagated superset of
the judged set — the declared box when no assume has constrained, the
precondition-narrowed set once one has; a **sound set-level
refutation** of that set, rendered as a REFUTED verdict per
`design/e2a-registration.md` amendment 1 with the conditional wording
when an assume constrained; not a witness, not a counterexample to the
program).

``stelling_assume`` **constrains** by default (``assume_mode="constrain"``):
where the predicate's producing equation is a comparison against a
point-interval side, the compared variable's own propagated interval is
narrowed by an exact meet with the *closed* half-space. The soundness
direction is the whole design: the narrowed domain must be a SUPERSET of
the true assumed region intersected with the current domain, never a
subset — so strict inequalities close (``x > k`` narrows to ``[k, hi]``,
never ``[nextafter(k), hi]``), nothing is ever inverted through
arithmetic (``assume(x*x <= 4)`` narrows the square's interval only,
never ``x``), and both-sides-varying comparisons stay inert (intervals
cannot represent relational coupling). Conjunctions recurse into their
conjuncts; every other predicate shape stays inert exactly as before —
dropped, which is sound (propagation runs over a superset) and never
silent: counted in coverage as ``inert`` (outside the "known" fraction)
and noted with its source address plus the reason (amendment 2).
Narrowing is forward-only: later equations read the narrowed interval,
earlier results are not revisited (conservative). Every applied
narrowing is disclosed three ways: a ``constrained`` coverage category,
an ``assume CONSTRAINED`` note, and a stamped assumption (the verdict
holds where the precondition holds).

Definite-verdict licensing splits by BOX EXACTNESS (audit F7): a box is
exact iff it equals the variable's true value set — ``stelling_any``
outputs and exact-point consts only; every transfer output is an
over-approximation (rounding pads, correlation-blind arithmetic). An
assume constraining an exact-box variable certifies its region's
satisfiability by the nonempty meet, and definite verdicts stand. An
assume constraining a NON-exact variable is still applied (the meet
over-approximates true-region ∩ reachable — sound) and the emptiness
refusals still fire (they prove emptiness from the over-approximation),
but the precondition's satisfiability is UNCERTIFIED: every subsequent
definite violation is withheld from REFUTED (status ``unknown``, note
quoted — a possibly-vacuous refutation is not a refutation), while
VERIFIED remains allowed carrying a may-be-vacuous note and stamped
line; the inert-mode control is the visibility instrument. An assume whose region is provably
EMPTY — an empty meet, a definitely-false constant comparison, or a
STRICT constraint whose meet collapses onto the closed boundary point
(``x > k`` narrowing to ``[k, k]``: the true region ``(k, k]`` is empty
though the closed stand-in is not) — is a harness defect at top level
and inside transparent call scopes: the precondition is definitely false
on the whole over-approximated domain, so every downstream obligation
would be vacuous — :class:`UnsatisfiableAssumptionError`, never a
verdict, never silent (degrade-don't-crash does not apply to harness
defects, exactly as for the unbound-var ``TranscriptionError``). Inside
a possibly-untaken ``cond`` BRANCH the same unsatisfiability is
branch-scoped (the other branch is real): the constraint is not applied
and a note discloses that obligations in that branch may be vacuous
under the branch's precondition — no raise, no narrowing.
``assume_mode="inert"`` reproduces the drop-everything MVP behavior
byte-identically — the comparability control for the vacuity instrument.

``semantics`` is the dial SOUNDNESS.md's stamp contract names: ``"real"``
(the default, byte-identical to everything above) judges obligations in
exact real arithmetic; ``semantics="ieee"`` judges them about the traced
program's **IEEE binary64 round-to-nearest float execution**. The ieee
domain is the same :class:`stelling.interval.IntervalArray` plus a
per-array ``maybe_nan`` flag (a parallel ``var id -> bool`` table beside
the interval env): endpoint arithmetic for the monotone core is native
binary64 with NO outward rounding (the float value itself is computable
— :data:`stelling.interval.IEEE_ENDPOINT_ASSUMPTION`), overflow
saturates to the VALUE ±inf, NaN-producing corner classes (``inf−inf``,
``0·±inf``, ``0/0``, ``inf/inf``) set ``maybe_nan``, ⊤ is
maybe-NaN, and a predicate over a maybe-NaN operand is never definitely
true (NaN falsifies every comparison except ``ne``, which it satisfies).
Every registered transfer is censused for ieee in
:data:`IEEE_TRANSFERS` — sound as-is, given an ieee variant, or declined
to ⊤-maybe-NaN with the gap quoted; the real mode's extended-real
conventions (the ``0·∞ = 0`` endpoint rule) are never reused. The
``domain`` parameter (only registered value ``"interval"``) is the
integration point for future tightened domains, which are refused under
``semantics="real"`` outright: tightening ℝ arithmetic without float
semantics converts accidental UNKNOWNs into false VERIFIEDs.

The exactness-certification machinery (the per-var exact set and the
box-nonemptiness-certifies-region-nonemptiness decision) lives in
:mod:`stelling.exactness` and is consumed here; the principle it holds —
a sound over-approximation certifies emptiness but never nonemptiness —
is that module's docstring.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from stelling import exactness
from stelling import interval as iv
from stelling import ir
from stelling.coverage import DEFAULT_TRANSPARENT, Coverage, CoverageCounter, sub_jaxprs

__all__ = [
    "IEEE_TRANSFERS",
    "ObligationReport",
    "Propagation",
    "TIGHTENED_DOMAIN_REAL_REFUSAL",
    "TRANSFERS",
    "UnsatisfiableAssumptionError",
    "interval_env",
    "propagate",
]

TIER_EXACT = "exact"
TIER_SOUND = "sound"
TIER_SOUND_LIBM = "sound-libm"


class UnsatisfiableAssumptionError(ValueError):
    """An assume's precondition is definitely false on the whole
    over-approximated domain: the meet of the constrained variable's
    propagated interval with the assumed half-space is empty, so the
    declared set contains no point satisfying the precondition and every
    downstream obligation would be vacuously "verified".

    This is the empty-declared-set refusal class — a harness defect, like
    the unbound-var :class:`stelling.ir.TranscriptionError` — so
    degrade-don't-crash does not apply: raised loudly, never a VERIFIED,
    never silent.
    """

# 2**53: the largest magnitude below which every int is exactly a double.
_EXACT_INT = 9007199254740992


@dataclass(frozen=True)
class ObligationReport:
    """One ``stelling_assert`` equation, judged over the declared box."""

    index: int  # traversal order among obligations
    status: str  # "discharged" | "unknown" | "violated-over-set"
    detail: str
    source_info: tuple[str, ...]


@dataclass(frozen=True)
class Propagation:
    obligations: tuple[ObligationReport, ...]
    nonvacuity_checks: tuple[ObligationReport, ...]  # y0-membership conditions
    coverage: Coverage
    transfers_used: tuple[tuple[str, str], ...]  # (primitive, tier), sorted
    assumptions: tuple[str, ...]
    notes: tuple[str, ...]
    # which arithmetic the obligations were judged about ("real" | "ieee");
    # the verdict assemblers stamp from this field, never from a guess
    semantics: str = "real"

    @property
    def all_discharged(self) -> bool:
        return bool(self.obligations) and all(
            o.status == "discharged" for o in self.obligations
        )

    @property
    def any_violated(self) -> bool:
        return any(o.status == "violated-over-set" for o in self.obligations)

    @property
    def dropped_constraints(self) -> int:
        return self.coverage.inert


# -- literals and consts ------------------------------------------------------
#
# The unsigned formats joined with the pytree-probe registration round:
# registering `and`/`or`/`eq`/`ne` made the propagator *read* the uint mask
# literals of RNG plumbing (threefry constants and friends) that previously
# sat behind unknown primitives — without decoders the read raised and the
# whole analysis died on a legal trace (h_hard), the exact failure the
# guard rule forbids. Unsigned ints decode to python ints, which
# `_int_bracket` already brackets soundly above 2**53.
_STRUCT_FMT = {
    "<f8": "d", "<f4": "f", "<i8": "q", "<i4": "i", "|b1": "?",
    "|u1": "B", "<u2": "H", "<u4": "I", "<u8": "Q",
    "|i1": "b", "<i2": "h",
}


def _int_bracket(v: int) -> tuple[float, float]:
    """A sound double bracket of an arbitrary python int (audit findings 3/5:
    int64 above 2**53 must widen; ints beyond the double range must saturate
    rather than crash on float())."""
    if abs(v) <= _EXACT_INT:
        x = float(v)
        return x, x
    try:
        x = float(v)
    except OverflowError:
        maxf = math.nextafter(math.inf, 0.0)
        return (maxf, math.inf) if v > 0 else (-math.inf, -maxf)
    return iv._down(x), iv._up(x)


def _decode_array(a: ir.Array) -> iv.IntervalArray:
    fmt = _STRUCT_FMT.get(a.dtype)
    if fmt is None:
        raise ir.TranscriptionError(
            f"no zero-dep decoder for array dtype {a.dtype!r}; add one before "
            f"propagating this query"
        )
    n = 1
    for d in a.shape:
        n *= d
    los, his = [], []
    for v in struct.unpack(f"<{n}{fmt}", a.data):
        if isinstance(v, int):
            lo, hi = _int_bracket(v)
        else:
            lo = hi = float(v)
        los.append(lo)
        his.append(hi)
    return iv.IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))


def _value_to_interval(v, shape: tuple[int, ...]) -> iv.IntervalArray:
    if isinstance(v, ir.Array):
        return _decode_array(v)
    if isinstance(v, bool):
        return iv.point(1.0 if v else 0.0, shape)
    if isinstance(v, int):
        lo, hi = _int_bracket(v)
        n = 1
        for d in shape:
            n *= d
        return iv.IntervalArray(shape=shape, los=(lo,) * n, his=(hi,) * n)
    if isinstance(v, float):
        return iv.point(v, shape)
    raise ir.TranscriptionError(
        f"no interval meaning for literal/const of type {type(v).__name__}"
    )


# -- the transfer registry ----------------------------------------------------
#
# Exactly the target census list + the harness primitives, plus the closed
# pytree-probe registration round (abs, eq, ne, and, or, stop_gradient,
# reshape, pow, reduce_or — the primitives the probe's own ⊤ list named),
# plus the maddening heat-node round's one allowed structural addition
# (scatter, static-index set form only), plus the MIME fvm round's two
# allowed structural additions (gather, static-index row form only;
# transpose).
# Signature:
# transfer(eqn, params: dict, ins: list[IntervalArray]) -> list[IntervalArray]


# Conversions that are exact for EVERY representable source value — the only
# ones the pass-through is sound for (audit finding 1: f64->f32 rounds,
# int64->int32 wraps, int->bool collapses; passing those through produced a
# verified false VERIFIED on a real f32 round-trip trace).
_EXACT_CONVERSIONS = frozenset(
    {
        ("float32", "float64"),
        ("float16", "float32"),
        ("float16", "float64"),
        ("int8", "int16"), ("int8", "int32"), ("int8", "int64"),
        ("int16", "int32"), ("int16", "int64"),
        ("int32", "int64"),
        ("int8", "float32"), ("int16", "float32"),
        ("int8", "float64"), ("int16", "float64"), ("int32", "float64"),
        ("bool", "int8"), ("bool", "int16"), ("bool", "int32"),
        ("bool", "int64"), ("bool", "float32"), ("bool", "float64"),
    }
)

_INT_RANGE = {"int32": 2.0**31, "int64": 2.0**63}


def _t_convert(eqn, params, ins):
    (a,) = ins
    src = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    dst = str(params.get("new_dtype"))
    if src == dst or (src, dst) in _EXACT_CONVERSIONS:
        return [a]
    if "float" in src and dst in _INT_RANGE:
        # float -> integer truncates toward zero; trunc is monotone, so
        # [trunc(lo), trunc(hi)] brackets — but only while the values fit the
        # target's range (outside it jax clamps/wraps, which trunc does not
        # model). The upper check is STRICT: +2**n-1 itself is not
        # representable (intN max is 2**(n-1) - 1), and for int64 the float
        # `bound - 1` rounds back to `bound` — second audit, finding 4-B: the
        # inclusive check admitted exactly ±2**31 and claimed a value int32
        # cannot hold.
        bound = _INT_RANGE[dst]
        if any(not (-bound <= x < bound) for x in (*a.los, *a.his)):
            return None
        return [iv.IntervalArray(
            shape=a.shape,
            los=tuple(float(math.trunc(x)) for x in a.los),
            his=tuple(float(math.trunc(x)) for x in a.his),
        )]
    return None  # value-changing or unrecognized conversion -> ⊤, noted


def _t_reshape(eqn, params, ins):
    if params.get("dimensions") is not None:
        # a dimensions= reshape permutes before reshaping — not the C-order
        # flat identity; no rule yet, so decline (⊤ with the params noted).
        return None
    return [iv.reshape(ins[0], tuple(params["new_sizes"]))]


def _t_bool_logic(name, op):
    """jax's ``and``/``or`` are *bitwise*: on bool operands they are the
    logical connectives our three-valued encoding models; on integers they
    are bit arithmetic, which no interval rule here covers — decline."""

    def t(eqn, params, ins):
        dtypes = [v.aval.dtype for v in eqn.invars]
        if any(d != "bool" for d in dtypes):
            raise iv.IntervalError(
                f"{name!r} transfer covers bool operands only (three-valued "
                f"logic); got dtypes {dtypes} — bitwise integer {name} has "
                f"no interval rule"
            )
        return [op(*ins)]

    return t


def _t_reduce_or(eqn, params, ins):
    dtypes = [v.aval.dtype for v in eqn.invars]
    if any(d != "bool" for d in dtypes):
        raise iv.IntervalError(
            f"'reduce_or' transfer covers bool operands only; got dtypes "
            f"{dtypes}"
        )
    return [iv.reduce_or(ins[0], tuple(params["axes"]))]


# The single-element scatter dimension_numbers of ``x.at[k].set(v)`` on a
# rank-1 operand; any OTHER field a jax version adds (batching dims today)
# must be empty or the transfer declines.
_SCATTER_SET_CORE = {
    "update_window_dims": (),
    "inserted_window_dims": (0,),
    "scatter_dims_to_operand_dims": (0,),
}


def _t_scatter(eqn, params, ins):
    """``x.at[k].set(v)`` in its static-index form — the allowed-by-census
    structural addition from the maddening HeatNode trace (the Dirichlet
    boundary writes ``T_new.at[0].set(T_left)`` / ``.at[-1].set(T_right)``).

    Covered form, exactly: rank-1 operand, one index row holding one
    definite in-range integer, scalar update, no update computation
    (``update_jaxpr=None``), canonical single-element dimension numbers.
    The output is the operand's box with element ``k``'s interval replaced
    by the update's — pure data movement, no arithmetic, no rounding.

    Everything else declines to a noted ⊤: dynamic (non-point) or
    out-of-range indices (mode-dependent clamp/drop is the census's wedge
    bug class, never guessed), update windows, batching dims, higher
    ranks, computed updates.
    """
    if len(ins) != 3 or params.get("update_jaxpr") is not None:
        return None
    operand, indices, updates = ins
    dn = params.get("dimension_numbers")
    if not isinstance(dn, ir.NamedTupleParam):
        return None
    fields = dict(dn.fields)
    if any(fields.get(k) != v for k, v in _SCATTER_SET_CORE.items()):
        return None
    if any(v != () for k, v in fields.items() if k not in _SCATTER_SET_CORE):
        return None
    if len(operand.shape) != 1 or indices.shape != (1,) or updates.shape != ():
        return None
    lo, hi = indices.los[0], indices.his[0]
    if lo != hi or not math.isfinite(lo) or lo != math.floor(lo):
        return None  # dynamic or non-integral index: no exact rule
    k = int(lo)
    if not 0 <= k < operand.shape[0]:
        return None  # out of range: FILL_OR_DROP drops, CLIP clamps — decline
    los = list(operand.los)
    his = list(operand.his)
    los[k] = updates.los[0]
    his[k] = updates.his[0]
    return [iv.IntervalArray(shape=operand.shape, los=tuple(los), his=tuple(his))]


def _t_gather(eqn, params, ins):
    """``x[idx]`` in its static-index leading-axis row form — the
    allowed-by-census structural addition from the MIME fvm laplacian
    trace (the gather half of the operators' gather→compute→scatter
    pattern: ``phi[mesh.owner]`` / ``phi[mesh.neighbour]`` on rank-1
    fields and ``grad[mesh.owner]`` on rank-2, with the mesh topology
    entering as definite const indices).

    Covered form, exactly: operand of rank r >= 1; indices ``(N, 1)``
    holding definite integral in-range points; dimension numbers that
    collapse exactly the leading axis (``offset_dims = (1, …, r-1)``,
    ``collapsed_slice_dims = (0,)``, ``start_index_map = (0,)``, every
    batching field empty); ``slice_sizes = (1, *operand.shape[1:])``.
    The output stacks the selected rows: ``out[i] = operand[k_i]`` —
    pure data movement, no arithmetic, no rounding. All
    ``GatherScatterMode``\\ s agree on definitely-in-range indices, so
    the mode is not constrained here.

    Everything else declines to a noted ⊤: dynamic (non-point) or
    out-of-range indices (mode-dependent clamp/drop/fill is the census's
    wedge bug class, never guessed), batching dims, window offsets not
    covering the full trailing block, multi-column index vectors.
    """
    if len(ins) != 2:
        return None
    operand, indices = ins
    r = len(operand.shape)
    if r < 1 or len(indices.shape) != 2 or indices.shape[1] != 1:
        return None
    dn = params.get("dimension_numbers")
    if not isinstance(dn, ir.NamedTupleParam):
        return None
    fields = dict(dn.fields)
    want = {
        "offset_dims": tuple(range(1, r)),
        "collapsed_slice_dims": (0,),
        "start_index_map": (0,),
    }
    if any(fields.get(k) != v for k, v in want.items()):
        return None
    if any(v != () for k, v in fields.items() if k not in want):
        return None
    if tuple(params.get("slice_sizes", ())) != (1,) + operand.shape[1:]:
        return None
    ks = []
    for lo, hi in zip(indices.los, indices.his):
        if lo != hi or not math.isfinite(lo) or lo != math.floor(lo):
            return None  # dynamic or non-integral index: no exact rule
        k = int(lo)
        if not 0 <= k < operand.shape[0]:
            return None  # out of range: clamp/drop/fill is mode-dependent
        ks.append(k)
    return [iv.take_rows(operand, ks)]


TRANSFERS = {
    "add": (lambda eqn, p, ins: [iv.add(*ins)], TIER_SOUND),
    "sub": (lambda eqn, p, ins: [iv.sub(*ins)], TIER_SOUND),
    "mul": (lambda eqn, p, ins: [iv.mul(*ins)], TIER_SOUND),
    "div": (lambda eqn, p, ins: [iv.div(*ins)], TIER_SOUND),
    "neg": (lambda eqn, p, ins: [iv.neg(ins[0])], TIER_EXACT),
    "abs": (lambda eqn, p, ins: [iv.abs_(ins[0])], TIER_EXACT),
    "max": (lambda eqn, p, ins: [iv.maximum(*ins)], TIER_EXACT),
    "min": (lambda eqn, p, ins: [iv.minimum(*ins)], TIER_EXACT),
    # pow's corner rule holds for strictly positive bases only; anything
    # else declines inside iv.pow_ (IntervalError -> noted ⊤).
    "pow": (lambda eqn, p, ins: [iv.pow_(*ins)], TIER_SOUND_LIBM),
    "stop_gradient": (lambda eqn, p, ins: [ins[0]], TIER_EXACT),
    "reshape": (_t_reshape, TIER_EXACT),
    # select_n(which, *cases): the masked case-split. Exact where `which` is
    # definite; a sound join where it straddles (design/control-flow-*).
    "select_n": (lambda eqn, p, ins: [iv.select_n(ins[0], ins[1:])], TIER_EXACT),
    "exp": (lambda eqn, p, ins: [iv.exp(ins[0])], TIER_SOUND_LIBM),
    "lt": (lambda eqn, p, ins: [iv.lt(*ins)], TIER_EXACT),
    "gt": (lambda eqn, p, ins: [iv.gt(*ins)], TIER_EXACT),
    "le": (lambda eqn, p, ins: [iv.le(*ins)], TIER_EXACT),
    "ge": (lambda eqn, p, ins: [iv.ge(*ins)], TIER_EXACT),
    "eq": (lambda eqn, p, ins: [iv.eq(*ins)], TIER_EXACT),
    "ne": (lambda eqn, p, ins: [iv.ne(*ins)], TIER_EXACT),
    "and": (_t_bool_logic("and", iv.logical_and), TIER_EXACT),
    "or": (_t_bool_logic("or", iv.logical_or), TIER_EXACT),
    "reduce_or": (_t_reduce_or, TIER_EXACT),
    "squeeze": (
        lambda eqn, p, ins: [iv.squeeze(ins[0], tuple(p.get("dimensions", ())))],
        TIER_EXACT,
    ),
    "slice": (
        lambda eqn, p, ins: [
            iv.slice_(
                ins[0],
                tuple(p["start_indices"]),
                tuple(p["limit_indices"]),
                tuple(p["strides"]) if p.get("strides") else None,
            )
        ],
        TIER_EXACT,
    ),
    # x.at[k].set(v), static index only — census addition from the maddening
    # HeatNode trace; every other scatter configuration declines (see
    # _t_scatter).
    "scatter": (_t_scatter, TIER_EXACT),
    # x[idx], static-index leading-axis row form only — census addition from
    # the MIME fvm laplacian trace; every other gather configuration
    # declines (see _t_gather).
    "gather": (_t_gather, TIER_EXACT),
    # axis permutation: pure data movement (MIME fvm round, reached inside
    # the transparent jnp.linalg.inv jit); malformed permutations decline
    # inside iv.transpose (IntervalError -> noted ⊤).
    "transpose": (
        lambda eqn, p, ins: [
            iv.transpose(ins[0], tuple(p.get("permutation", ()) or ()))
        ],
        TIER_EXACT,
    ),
    "broadcast_in_dim": (
        lambda eqn, p, ins: [
            iv.broadcast_in_dim(
                ins[0], tuple(p["shape"]), tuple(p["broadcast_dimensions"])
            )
        ],
        TIER_EXACT,
    ),
    "concatenate": (
        lambda eqn, p, ins: [iv.concatenate(list(ins), int(p["dimension"]))],
        TIER_EXACT,
    ),
    "convert_element_type": (_t_convert, TIER_EXACT),
    "stelling_any": (
        lambda eqn, p, ins: [
            iv.from_bounds(tuple(p["shape"]), float(p["lo"]), float(p["hi"]))
        ],
        TIER_EXACT,
    ),
    "stelling_assert": (lambda eqn, p, ins: [ins[0]], TIER_EXACT),
    "stelling_nonvacuity": (lambda eqn, p, ins: [ins[0]], TIER_EXACT),
}


# Every sound-libm transfer names the exact libm-fidelity assumption it
# rides on; the tier check below stamps it into the verdict. A transfer at
# that tier without an entry still stamps a (generic) assumption — the
# demotion is never silent.
_LIBM_ASSUMPTIONS = {
    "exp": iv.EXP_LIBM_ASSUMPTION,
    "pow": iv.POW_LIBM_ASSUMPTION,
}


# -- the ieee transfer registry -----------------------------------------------
#
# The censused counterpart of TRANSFERS for semantics="ieee": every key of
# TRANSFERS appears here explicitly — (i) verified sound as-is for float
# (pure data movement, structural ops — wrapped only to route the
# maybe-NaN flag), (ii) given an ieee variant (the arithmetic core with
# native binary64 endpoints and NaN-corner routing; comparisons with the
# NaN-falsification rule; anything that touched the extended-real
# conventions — the real-mode 0·∞ = 0 inside iv.mul is exactly what is
# NOT reused), or (iii) declined to ⊤-maybe-NaN with the gap quoted.
# No silent reuse; the registry is the census, mirrored in the build
# report.
#
# Signature: t(eqn, params, ins, flags) -> (outs, out_flags) | None
# where flags/out_flags are the per-array maybe-NaN flags. None and
# IntervalError decline exactly as in real mode, except ⊤ outputs carry
# maybe_nan=True (⊤ under ieee is maybe-NaN: an unmodeled value could be
# anything a float can be, including NaN).


def _ieee_f64_only(eqn) -> None:
    """The binary64 guard for the arithmetic core: ieee endpoints are
    native binary64, which does not model float32/float16 rounding or
    integer wraparound — any other dtype declines with the gap quoted."""
    bad = sorted(
        {v.aval.dtype for v in (*eqn.invars, *eqn.outvars)} - {"float64"}
    )
    if bad:
        raise iv.IntervalError(
            f"ieee endpoint arithmetic is binary64-only; operand/result "
            f"dtypes {bad} are not modeled (float32/float16 round "
            f"differently, integer arithmetic wraps) — declined"
        )


def _ieee_arith(op):
    """The monotone core (add/sub/mul/div): native binary64 corner
    endpoints, NaN corners routed to the flag (never into an interval).
    A maybe-NaN operand poisons the result (NaN propagates through all
    four ops), so operand flags OR into the output flag."""

    def t(eqn, params, ins, flags):
        _ieee_f64_only(eqn)
        box, made_nan = op(ins[0], ins[1])
        return [box], [made_nan or any(flags)]

    return t


def _ieee_unary_exact(fn):
    """neg/abs: exact sign arithmetic — the float result IS the real
    result, no rounding, no new NaN (neg(nan)/abs(nan) stay NaN: flag
    propagates)."""

    def t(eqn, params, ins, flags):
        _ieee_f64_only(eqn)
        # subnormal haze on the result: whether a sign operation is
        # flushed is target-dependent; the hull with 0 covers both
        return [iv.subnormal_haze(fn(ins[0]))[0]], [flags[0]]

    return t


def _ieee_minmax(fn):
    """max/min: float max/min of non-NaN operands is exact (no rounding),
    so the real transfer's endpoint rule is float-exact when neither
    operand is maybe-NaN. With a maybe-NaN operand the result is the
    other operand, the extremum, or NaN depending on the backend's NaN
    ordering (measured on jax 0.11.0 cpu: lax.max/min PROPAGATE NaN in
    both operand orders) — the operand hull covers every one of those
    non-NaN outcomes, so hull + flag is sound without leaning on the
    measurement."""

    def t(eqn, params, ins, flags):
        _ieee_f64_only(eqn)
        a, b = ins
        if any(flags):
            mn, mx = iv.minimum(a, b), iv.maximum(a, b)
            hull = iv.IntervalArray(shape=mn.shape, los=mn.los, his=mx.his)
            return [iv.subnormal_haze(hull)[0]], [True]
        return [iv.subnormal_haze(fn(a, b))[0]], [False]

    return t


def _ieee_exp(eqn, params, ins, flags):
    """exp keeps its 1-ulp outward libm bracket — a faithfully-rounded
    libm result lands within 1 ulp of the true value, so the outward
    bracket contains the float the program computes (the same
    EXP_LIBM_ASSUMPTION stamps, via the sound-libm tier). exp(NaN) is
    NaN and nothing else: flag propagation is exact. The subnormal haze
    covers flushing libm targets (measured jax 0.11.0 CPU:
    exp(-720) = 0.0 while IEEE exp is subnormal — the 1-ulp bracket alone
    cannot absorb a flush to 0)."""
    _ieee_f64_only(eqn)
    return [iv.subnormal_haze(iv.exp(ins[0]))[0]], [flags[0]]


def _ieee_pow(eqn, params, ins, flags):
    """pow keeps its libm corner brackets (strictly positive base — the
    same decline otherwise), but a maybe-NaN operand DECLINES rather than
    flag-propagating: IEEE pow has non-NaN results at NaN inputs
    (pow(NaN, 0) = 1 and pow(1, NaN) = 1 — measured on jax 0.11.0), and 1
    may lie outside the corner bracket, so flag propagation alone would
    be unsound."""
    _ieee_f64_only(eqn)
    if any(flags):
        raise iv.IntervalError(
            "pow over a maybe-NaN operand: IEEE pow(NaN, 0) = 1 and "
            "pow(1, NaN) = 1 escape both the corner bracket and the NaN "
            "flag — no sound rule here, declined"
        )
    # subnormal haze on the result: a flushing libm pow may return 0
    # where the bracket is subnormal
    return [iv.subnormal_haze(iv.pow_(*ins))[0]], [False]


def _non_f64_float_dtypes(atoms) -> list[str]:
    """The non-binary64 FLOAT dtypes among these atoms' avals. The ieee
    mode models binary64 only, and the measured flush is PER-DTYPE
    (jax 0.11.0 CPU flushes float32 subnormals — |x| < 2**-126, which are
    NORMAL binary64 numbers invisible to the binary64 haze — while
    float16 is not flushed on this target): any surface that would judge
    a non-f64 float definitely must decline instead of mismodelling."""
    return sorted(
        {
            v.aval.dtype
            for v in atoms
            if "float" in (v.aval.dtype or "") and v.aval.dtype != "float64"
        }
    )


def _ieee_cmp_f64_only(eqn) -> None:
    """The comparison face of the binary64-only guard (re-attack U2): DAZ
    reaches comparisons per-dtype (measured jax 0.11.0 CPU:
    ``float32(1e-45) > 0`` is False, ``float32(1e-39) == float32(1e-40)``
    is True), and the binary64 haze cannot see the f32 band — so any
    comparison with a non-f64 float operand declines with the gap
    quoted. Integer/bool comparisons are unaffected (no flush hazard)."""
    bad = _non_f64_float_dtypes(eqn.invars)
    if bad:
        raise iv.IntervalError(
            f"ieee mode models binary64 only; {'/'.join(bad)} comparison "
            f"semantics (incl. per-dtype subnormal flush — measured on jax "
            f"0.11.0 CPU: float32 subnormals flush, float32(1e-45) > 0 is "
            f"False) are not modelled — declined"
        )


def _ieee_cmp(fn, nan_answer):
    """Comparisons under ieee: exact on non-NaN operands (float compare
    agrees with the real order on ±inf), three-valued as in real mode.
    NaN falsifies lt/gt/le/ge/eq and satisfies ne, so with a maybe-NaN
    operand the elements whose definite answer disagrees with the NaN
    answer degrade to unknown: definite-true is blocked for the
    falsified comparisons (never definitely true), definite-false is
    blocked for ne — the other face stands (NaN agrees with it).
    Comparison outputs are bools: never NaN, flag False.

    Operands are subnormal-hazed before judging: DAZ reaches
    comparisons themselves (measured jax 0.11.0 CPU: ``5e-324 > 0`` is
    False, ``5e-324 == 1e-320`` is True), so a band-touching operand
    must be judged as possibly 0."""
    blocked = iv.BOOL_FALSE if nan_answer else iv.BOOL_TRUE

    def t(eqn, params, ins, flags):
        _ieee_cmp_f64_only(eqn)
        r = fn(*(iv.subnormal_haze(x)[0] for x in ins))
        if any(flags):
            los, his = [], []
            for lo, hi in zip(r.los, r.his):
                if (lo, hi) == blocked:
                    lo, hi = iv.BOOL_UNKNOWN
                los.append(lo)
                his.append(hi)
            r = iv.IntervalArray(shape=r.shape, los=tuple(los), his=tuple(his))
        return [r], [False]

    return t


def _ieee_bool_logic(name, op):
    """Kleene and/or on bools (same bool-dtype guard as real mode —
    bitwise integer forms decline). Bools are never NaN; a maybe-NaN
    flag on a bool operand is a decline artifact (⊤-maybe-NaN), so that
    operand's elements read as unknown before the Kleene op — false ∧
    unknown is still false, nothing definite leaks from a flagged
    operand."""

    def t(eqn, params, ins, flags):
        dtypes = [v.aval.dtype for v in eqn.invars]
        if any(d != "bool" for d in dtypes):
            raise iv.IntervalError(
                f"{name!r} transfer covers bool operands only (three-valued "
                f"logic); got dtypes {dtypes} — bitwise integer {name} has "
                f"no interval rule"
            )
        ops = [
            iv.IntervalArray(
                shape=b.shape, los=(0.0,) * b.size, his=(1.0,) * b.size
            )
            if f
            else b
            for b, f in zip(ins, flags)
        ]
        return [op(*ops)], [False]

    return t


def _ieee_reduce_or(eqn, params, ins, flags):
    dtypes = [v.aval.dtype for v in eqn.invars]
    if any(d != "bool" for d in dtypes):
        raise iv.IntervalError(
            f"'reduce_or' transfer covers bool operands only; got dtypes "
            f"{dtypes}"
        )
    a = ins[0]
    if flags[0]:
        a = iv.IntervalArray(
            shape=a.shape, los=(0.0,) * a.size, his=(1.0,) * a.size
        )
    return [iv.reduce_or(a, tuple(params["axes"]))], [False]


# the selector dtypes jax's select_n accepts (measured: lax.select_n
# rejects float selectors at trace time) — the ONLY dtypes for which
# "a selector value is never NaN" is a fact rather than an assumption
_SELECTOR_DTYPES = frozenset(
    {"bool", "int8", "int16", "int32", "int64",
     "uint8", "uint16", "uint32", "uint64"}
)


def _ieee_select_n(eqn, params, ins, flags):
    """Selection/join is pure data movement — the real transfer's measured
    clamp semantics and straddle join are float-sound as-is. The output
    flag is the OR of every case's flag — conservative (a case the
    selector excludes may still set it), sound.

    The selector invariant is ENFORCED, not assumed (audit F1): only
    bool/integer selectors are accepted (jax rejects float selectors at
    trace time, but hand-built/deserialized IR arrives here unchecked —
    a float selector's value could be NaN, which no selection rule
    models), and a maybe-NaN-flagged selector declines outright (its
    provenance is untrusted; picking a case from its interval would
    silently drop the NaN possibility)."""
    sel_dtype = eqn.invars[0].aval.dtype if eqn.invars else ""
    if sel_dtype not in _SELECTOR_DTYPES:
        raise iv.IntervalError(
            f"select_n selector dtype {sel_dtype!r} is not bool/integer "
            f"(jax rejects float selectors at trace time; a float selector "
            f"may be NaN) — no selection rule; declined"
        )
    if flags[0]:
        raise iv.IntervalError(
            "select_n selector carries maybe-NaN under ieee semantics — "
            "its provenance is untrusted and selection would drop the NaN "
            "possibility; declined"
        )
    r = iv.select_n(ins[0], list(ins[1:]))
    return [r], [any(flags[1:])]


def _ieee_passthrough(real_transfer):
    """Category (i): pure data movement with exact semantics — element
    routing only, no arithmetic, no rounding, dtype-agnostic, NaN moves
    like any other value. The real transfer is reused with the operand
    flags OR-ed onto every output."""

    def t(eqn, params, ins, flags):
        outs = real_transfer(eqn, params, ins)
        if outs is None:
            return None
        return outs, [any(flags)] * len(outs)

    return t


def _ieee_scatter(eqn, params, ins, flags):
    """The static-index x.at[k].set(v) form: data movement (sound as-is);
    output flag = operand ∨ updates. Maybe-NaN INDICES decline — the
    static-index rule needs definite integer indices, and a NaN index is
    mode-dependent garbage."""
    if len(ins) == 3 and flags[1]:
        raise iv.IntervalError(
            "scatter indices carry maybe-NaN under ieee semantics — the "
            "static-index rule needs definite non-NaN indices; declined"
        )
    outs = _t_scatter(eqn, params, ins)
    if outs is None:
        return None
    return outs, [flags[0] or flags[2]]


def _ieee_gather(eqn, params, ins, flags):
    """The static-index leading-axis row form: data movement (sound
    as-is); output flag = operand's. Maybe-NaN indices decline, as for
    scatter."""
    if len(ins) == 2 and flags[1]:
        raise iv.IntervalError(
            "gather indices carry maybe-NaN under ieee semantics — the "
            "static-index rule needs definite non-NaN indices; declined"
        )
    outs = _t_gather(eqn, params, ins)
    if outs is None:
        return None
    return outs, [flags[0]]


def _ieee_convert(eqn, params, ins, flags):
    """convert_element_type under ieee. A non-f64 FLOAT source declines
    outright (re-attack U2): the whitelist's value-preservation claim is
    a gradual-semantics fact, measured FALSE under this target's
    per-dtype DAZ — jax 0.11.0 CPU converts an f32 subnormal to f64 as
    0.0 (storage keeps the bits; compute flushes), which would carry the
    flushed 0 into f64 dataflow past the binary64 haze and the f64-only
    arithmetic guard. The remaining whitelisted conversions (int/bool
    sources, f64 identity) really are value-preserving for every source
    value including ±inf and NaN (NaN converts to NaN: flag propagates).
    The float→int trunc path is float-exact (trunc is the measured
    conversion semantics, the in-range guard keeps it defined, integer
    results are exact doubles) but declines a maybe-NaN input —
    converting NaN to int is target-dependent garbage. Everything else
    declines exactly as in real mode."""
    src = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    dst = str(params.get("new_dtype"))
    if "float" in src and src != "float64":
        raise iv.IntervalError(
            f"ieee mode models binary64 only; {src} conversion semantics "
            f"(incl. per-dtype subnormal flush — measured on jax 0.11.0 "
            f"CPU: f32→f64 convert of a subnormal yields 0.0, so the "
            f"exact-conversion whitelist's value-preservation claim fails "
            f"under DAZ) are not modelled — declined"
        )
    if src == dst or (src, dst) in _EXACT_CONVERSIONS:
        return [ins[0]], [flags[0]]
    if "float" in src and dst in _INT_RANGE:
        if flags[0]:
            raise iv.IntervalError(
                "float->int conversion of a maybe-NaN value is "
                "target-dependent garbage under ieee semantics — declined"
            )
        outs = _t_convert(eqn, params, ins)
        if outs is None:
            return None
        return outs, [False]
    return None  # value-changing or unrecognized conversion -> ⊤, noted


def _ieee_any(eqn, params, ins, flags):
    """Declared inputs start maybe_nan=False: the declared closed box is
    a set of floats, and NaN bounds are refused (iv.from_bounds raises on
    NaN, which declines the declaration loudly rather than admitting an
    unbounded-NaN input). The box is subnormal-hazed on entry — DAZ
    flushes inputs, so a band-located declared value may be consumed as
    0; the dispatcher withholds exactness from declarations the haze
    changed."""
    box = iv.from_bounds(
        tuple(params["shape"]), float(params["lo"]), float(params["hi"])
    )
    return [iv.subnormal_haze(box)[0]], [False]


IEEE_TRANSFERS = {
    # (ii) ieee variants: the monotone arithmetic core — native binary64
    # endpoints, NaN corners routed to the flag; the real-mode 0·∞ = 0
    # convention (iv._prod inside iv.mul) is NOT reused.
    "add": (_ieee_arith(iv.ieee_add), TIER_EXACT),
    "sub": (_ieee_arith(iv.ieee_sub), TIER_EXACT),
    "mul": (_ieee_arith(iv.ieee_mul), TIER_EXACT),
    "div": (_ieee_arith(iv.ieee_div), TIER_EXACT),
    # (i)/(ii) exact sign arithmetic; flag propagates (NaN stays NaN)
    "neg": (_ieee_unary_exact(iv.neg), TIER_EXACT),
    "abs": (_ieee_unary_exact(iv.abs_), TIER_EXACT),
    # (ii) exact on non-NaN operands; NaN-ordering ambiguity covered by
    # the operand hull + flag
    "max": (_ieee_minmax(iv.maximum), TIER_EXACT),
    "min": (_ieee_minmax(iv.minimum), TIER_EXACT),
    # (ii) libm brackets kept (faithful rounding lands within 1 ulp);
    # pow declines maybe-NaN operands (pow(NaN,0)=1 escapes the flag)
    "pow": (_ieee_pow, TIER_SOUND_LIBM),
    "exp": (_ieee_exp, TIER_SOUND_LIBM),
    # (i) identity
    "stop_gradient": (_ieee_passthrough(lambda eqn, p, ins: [ins[0]]), TIER_EXACT),
    # (i) data movement (the dimensions= reshape declines as in real mode)
    "reshape": (_ieee_passthrough(_t_reshape), TIER_EXACT),
    # (ii) selection/join reused; flag = OR over cases
    "select_n": (_ieee_select_n, TIER_EXACT),
    # (ii) comparisons: NaN falsifies lt/gt/le/ge/eq (definite-true
    # blocked under maybe-NaN), satisfies ne (definite-false blocked)
    "lt": (_ieee_cmp(iv.lt, nan_answer=False), TIER_EXACT),
    "gt": (_ieee_cmp(iv.gt, nan_answer=False), TIER_EXACT),
    "le": (_ieee_cmp(iv.le, nan_answer=False), TIER_EXACT),
    "ge": (_ieee_cmp(iv.ge, nan_answer=False), TIER_EXACT),
    "eq": (_ieee_cmp(iv.eq, nan_answer=False), TIER_EXACT),
    "ne": (_ieee_cmp(iv.ne, nan_answer=True), TIER_EXACT),
    # (ii) Kleene logic; flagged bool operands read as unknown
    "and": (_ieee_bool_logic("and", iv.logical_and), TIER_EXACT),
    "or": (_ieee_bool_logic("or", iv.logical_or), TIER_EXACT),
    "reduce_or": (_ieee_reduce_or, TIER_EXACT),
    # (i) pure data movement, dtype-agnostic, flags ride along
    "squeeze": (
        _ieee_passthrough(
            lambda eqn, p, ins: [iv.squeeze(ins[0], tuple(p.get("dimensions", ())))]
        ),
        TIER_EXACT,
    ),
    "slice": (
        _ieee_passthrough(
            lambda eqn, p, ins: [
                iv.slice_(
                    ins[0],
                    tuple(p["start_indices"]),
                    tuple(p["limit_indices"]),
                    tuple(p["strides"]) if p.get("strides") else None,
                )
            ]
        ),
        TIER_EXACT,
    ),
    "scatter": (_ieee_scatter, TIER_EXACT),
    "gather": (_ieee_gather, TIER_EXACT),
    "transpose": (
        _ieee_passthrough(
            lambda eqn, p, ins: [
                iv.transpose(ins[0], tuple(p.get("permutation", ()) or ()))
            ]
        ),
        TIER_EXACT,
    ),
    "broadcast_in_dim": (
        _ieee_passthrough(
            lambda eqn, p, ins: [
                iv.broadcast_in_dim(
                    ins[0], tuple(p["shape"]), tuple(p["broadcast_dimensions"])
                )
            ]
        ),
        TIER_EXACT,
    ),
    "concatenate": (
        _ieee_passthrough(
            lambda eqn, p, ins: [iv.concatenate(list(ins), int(p["dimension"]))]
        ),
        TIER_EXACT,
    ),
    # (ii) exact whitelist passes with flag propagation; trunc path
    # declines maybe-NaN inputs; everything else declines as before
    "convert_element_type": (_ieee_convert, TIER_EXACT),
    # (i) declarations: flag starts False, NaN bounds refused
    "stelling_any": (_ieee_any, TIER_EXACT),
    # (i) identity pass-throughs (judging is flag-aware in the dispatcher)
    "stelling_assert": (
        _ieee_passthrough(lambda eqn, p, ins: [ins[0]]), TIER_EXACT
    ),
    "stelling_nonvacuity": (
        _ieee_passthrough(lambda eqn, p, ins: [ins[0]]), TIER_EXACT
    ),
}

# the census must stay total: a registered transfer with no ieee census
# entry would be silent reuse, the exact thing rule 6 forbids
assert set(IEEE_TRANSFERS) == set(TRANSFERS), (
    "every registered transfer must be censused for ieee semantics"
)


# -- constraining assume ------------------------------------------------------

_ASSUME_MODES = ("constrain", "inert")

# The comparisons a point-bounded assume can narrow through. `ne` is a
# comparison but NOT here: excluding a single point from an interval does
# not narrow it (the hull of [lo, hi] \ {k} is [lo, hi]) — it stays inert
# with the reason quoted.
_ASSUME_CMPS = frozenset({"ge", "gt", "le", "lt", "eq"})

# cmp(k, v) === flipped-cmp(v, k): normalization when the point side is on
# the left.
_CMP_FLIP = {"ge": "le", "le": "ge", "gt": "lt", "lt": "gt", "eq": "eq"}

_CMP_SYMBOL = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "=="}

# three-valued comparison transfers, for deciding point-vs-point assumes
_CMP_FN = {"ge": iv.ge, "gt": iv.gt, "le": iv.le, "lt": iv.lt, "eq": iv.eq}

# The spec-fixed relational disclosure, appended verbatim to the DROPPED
# note when neither comparison side's interval is a finite point AND both
# sides genuinely vary.
_RELATIONAL_REASON = (
    "relational: both sides vary — constraining needs relational domains"
)

# Audit F6: a CONSTANT bound that is not a finite point (±inf, NaN, an
# undecodable literal) must be disclosed as such — "both sides vary" is
# false of it, and drop reasons are quoted into verdict notes as fact.
_NONFINITE_BOUND_REASON = (
    "the comparison bound is a constant but not a finite point (infinite "
    "or undecodable, e.g. ±inf/NaN) — no closed half-space represents it"
)


def _finite_point(box: iv.IntervalArray | None) -> bool:
    """Every element is a degenerate finite interval (``lo == hi``,
    finite) — the shape of a usable assume bound."""
    return box is not None and all(
        lo == hi and math.isfinite(lo) for lo, hi in zip(box.los, box.his)
    )


def _subnormal_const_literal(atom: ir.Atom) -> bool:
    """A literal constant whose raw decode the subnormal haze changes —
    i.e. a subnormal-band constant. Under ieee its as-consumed value is
    flush-indeterminate, so the drop reason must name that shape rather
    than claim both sides vary (the audit-F6 discipline)."""
    if not isinstance(atom, ir.Literal):
        return False
    try:
        raw = _value_to_interval(atom.val, atom.aval.shape)
    except (iv.IntervalError, ir.TranscriptionError):
        return False
    return iv.subnormal_haze(raw)[1]


_SUBNORMAL_BOUND_REASON = (
    "a comparison side is a subnormal-band constant — its as-consumed "
    "value is flush-vs-gradual indeterminate under ieee semantics "
    "(FTZ/DAZ targets read it as 0); no certified half-space represents it"
)


def _strict_flush_witness(cmp: str, k: float, nlo: float, nhi: float) -> bool:
    """Flush-robust certification witness for a STRICT assume under ieee:
    does the closed meet stand-in contain a NON-subnormal point strictly
    satisfying the comparison? A strict region whose only content is
    subnormal may be EMPTY at runtime on a DAZ target (every member reads
    as 0, which the strict comparison excludes), so certification demands
    a witness whose runtime reading is itself (0, a normal, or ±inf)."""
    if cmp == "gt":  # region (k, nhi]
        if k < 0.0 <= nhi:
            return True  # w = 0
        if nhi > k and abs(nhi) >= iv.MIN_NORMAL:
            return True  # w = nhi (normal or ±inf)
        if k < -iv.MIN_NORMAL <= nhi:
            return True  # w = -MIN_NORMAL
        if k < iv.MIN_NORMAL <= nhi:
            return True  # w = +MIN_NORMAL
        return False
    # lt: region [nlo, k)
    if nlo <= 0.0 < k:
        return True  # w = 0
    if nlo < k and abs(nlo) >= iv.MIN_NORMAL:
        return True  # w = nlo (normal or ±inf)
    if nlo <= iv.MIN_NORMAL < k:
        return True  # w = +MIN_NORMAL
    if nlo <= -iv.MIN_NORMAL and k > -iv.MIN_NORMAL:
        return True  # w = -MIN_NORMAL
    return False


def _nonfinite_const(atom: ir.Atom, box: iv.IntervalArray | None) -> bool:
    """A constant comparison side that is not a finite point: an
    undecodable literal (NaN, complex, …) or a degenerate interval whose
    value is non-finite (±inf). Distinguished from a genuinely varying
    side so the drop reason names the actual shape (audit F6)."""
    if box is None:
        return isinstance(atom, ir.Literal)
    return all(lo == hi for lo, hi in zip(box.los, box.his)) and not all(
        math.isfinite(lo) for lo in box.los
    )


def _render_box(box: iv.IntervalArray) -> str:
    """Concise interval rendering for notes/assumptions: the scalar
    interval, a per-element list for small arrays, the hull for large."""
    if box.size == 1:
        return f"[{box.los[0]}, {box.his[0]}]"
    if box.size <= 4:
        return "[" + ", ".join(
            f"[{lo}, {hi}]" for lo, hi in zip(box.los, box.his)
        ) + "]"
    return f"hull [{min(box.los)}, {max(box.his)}] ({box.size} elements)"


def _render_bound(bound: iv.IntervalArray) -> str:
    """Render a point-interval bound as its value(s)."""
    if bound.size == 1:
        return f"{bound.los[0]}"
    if bound.size <= 4:
        return "[" + ", ".join(f"{v}" for v in bound.los) + "]"
    return f"values in [{min(bound.los)}, {max(bound.los)}] ({bound.size} elements)"


def _bool_status(b: iv.IntervalArray, *, constrained: bool = False) -> tuple[str, str]:
    n_true = sum(1 for lo, hi in zip(b.los, b.his) if (lo, hi) == iv.BOOL_TRUE)
    n_false = sum(1 for lo, hi in zip(b.los, b.his) if (lo, hi) == iv.BOOL_FALSE)
    n = b.size
    if n_true == n:
        return "discharged", f"definitely true for all {n} element(s)"
    if n_false:
        # audit F4: once an assume has constrained, the judgment set is no
        # longer the declared box — the detail must name the set actually
        # judged, or the refutation sentence misstates its own claim.
        judged = (
            "the precondition-narrowed set"
            if constrained
            else "the declared box"
        )
        return (
            "violated-over-set",
            f"definitely false for {n_false}/{n} element(s) over {judged}",
        )
    return "unknown", f"undecided for {n - n_true}/{n} element(s)"


class _Propagator:
    def __init__(self, assume_mode: str, semantics: str = "real") -> None:
        self.assume_mode = assume_mode
        self.semantics = semantics
        self.env: dict[int, iv.IntervalArray] = {}
        # ieee mode's parallel table: var id -> maybe_nan. Real mode never
        # writes it (the flag machinery is fenced behind semantics checks),
        # so the real path's behavior is untouched.
        self.nan: dict[int, bool] = {}
        # var id -> producing equation, for the jaxpr currently being run
        # (scoped in run(); assume classification only — never a transfer
        # input)
        self.producers: dict[int, ir.JaxprEqn] = {}
        self.counter = CoverageCounter()
        self.obligations: list[ObligationReport] = []
        self.nonvacuity_checks: list[ObligationReport] = []
        self.used: dict[str, str] = {}
        self.assumptions: set[str] = set()
        self.notes: list[str] = []
        # > 0 while executing a possibly-untaken cond branch: an
        # unsatisfiable assume there is a BRANCH-scoped vacuity (the other
        # branch is real), not a whole-domain harness defect — it degrades
        # with a note instead of raising (audit F2). Transparent call
        # scopes (jit) inherit the current value: they execute
        # unconditionally, so at depth 0 the raise stands.
        self.branch_depth = 0
        # set once any assume narrows: violated-over-set details are then
        # judged over the precondition-narrowed set, and must say so
        # (audit F4)
        self.any_constrained = False
        # var ids whose box is EXACT — equal to the variable's true value
        # set, not an over-approximation: stelling_any outputs (the
        # declared closed box IS the declared set) and consts decoded to
        # exact points. EVERY transfer output is non-exact (minimal,
        # conservative rule — audit F7): rounding pads and
        # correlation-blind joins can inflate a box past the true image,
        # so box-nonemptiness certifies true-region nonemptiness only for
        # exact boxes. Scope-local (swapped with env at every sub-jaxpr
        # descent): a scope certifies only its own declarations. The set,
        # its maintenance rules, and the certification decision live in
        # stelling.exactness (the shared primitive future layers import).
        self.exact = exactness.ExactSet()
        # set once a constraining assume narrows a NON-exact variable: an
        # uncertified-precondition constraint is then in force, and every
        # subsequent definite violation is withheld from REFUTED (a
        # possibly-vacuous refutation is not a refutation — audit F7).
        self.uncertified = False

    def read(self, atom: ir.Atom) -> iv.IntervalArray:
        if isinstance(atom, ir.Literal):
            # a constant the domain cannot represent (a NaN sentinel — the
            # ubiquitous `where(pred, x, nan)` pattern — an undecodable
            # dtype, a complex literal) is a LEGAL program value outside the
            # ℝ model: it degrades to ⊤ with a note, it does not kill the
            # analysis (audit-gate finding 1). The unbound-var raise below
            # is untouched — that one is a transcription defect, not a
            # value.
            try:
                box = _value_to_interval(atom.val, atom.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError) as e:
                self.notes.append(f"literal outside the domain ({e}); ⊤")
                return iv.top(atom.aval.shape)
            if self.semantics == "ieee":
                # DAZ flushes inputs: literal constants entering ieee
                # propagation are subnormal-hazed like every other value
                box = iv.subnormal_haze(box)[0]
            return box
        got = self.env.get(atom.id)
        if got is None:
            # not a coverage gap: an equation reading a never-bound var is a
            # transcription/traversal defect, and ⊤-ing it would soundly hide
            # a bug in the one module that must not have bugs. Same discipline
            # as the params whitelist: unknown primitives are fine, unknown
            # *structure* is not.
            raise ir.TranscriptionError(
                f"equation reads var {atom.id} before any binding — "
                f"transcription defect, refusing to widen it away"
            )
        return got

    def read_flag(self, atom: ir.Atom) -> bool:
        """The atom's maybe-NaN flag (ieee mode). Decodable literals are
        definite non-NaN values; undecodable ones (the NaN sentinel above
        all) may be NaN. Vars default to False — every flag-True binding
        is written explicitly."""
        if isinstance(atom, ir.Literal):
            try:
                _value_to_interval(atom.val, atom.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError):
                return True
            return False
        return self.nan.get(atom.id, False)

    def top_out(self, eqn: ir.JaxprEqn) -> None:
        for out in eqn.outvars:
            self.env[out.id] = iv.top(out.aval.shape)
            if self.semantics == "ieee":
                # ⊤ under ieee is maybe-NaN: an unknown/declined value
                # could be anything a float can be, including NaN
                self.nan[out.id] = True

    def mark_unreached(self, eqn: ir.JaxprEqn) -> None:
        stack = list(sub_jaxprs(eqn))
        while stack:
            j = stack.pop()
            for e in j.eqns:
                self.counter.record_unreached(e.primitive)
                stack.extend(sub_jaxprs(e))

    # -- constraining assume --------------------------------------------------

    def _quiet_interval(self, atom: ir.Atom) -> iv.IntervalArray | None:
        """The atom's current interval for assume classification only:
        never notes, never raises — ``None`` where no interval is
        readable (an undecodable literal, an unbound var). Classification
        failures make the assume inert, which is always sound."""
        if isinstance(atom, ir.Literal):
            try:
                box = _value_to_interval(atom.val, atom.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError):
                return None
            if self.semantics == "ieee":
                # the same haze read() applies: assume classification must
                # see the interval the judging paths see (a subnormal-band
                # bound is then not a finite point and cannot narrow)
                box = iv.subnormal_haze(box)[0]
            return box
        return self.env.get(atom.id)

    def _unsatisfiable(
        self, where: str, desc: str, vacuous: list[str], message: str
    ) -> bool:
        """One unsatisfiability posture for all three detection sites
        (empty meet, strict-boundary collapse, definitely-false constant).

        At top level and inside transparent call scopes the whole-domain
        claim is true: raise :class:`UnsatisfiableAssumptionError` with
        ``message``. Inside a possibly-untaken cond branch it is NOT — the
        assume is branch-scoped, the other branch is real (audit F2): the
        constraint degrades to a branch-local vacuity note (``desc`` rides
        into it) and the caller applies no narrowing. Returns True in the
        branch-local case."""
        if self.branch_depth:
            vacuous.append(desc)
            return True
        raise UnsatisfiableAssumptionError(message)

    def _assume_constrain(self, eqn: ir.JaxprEqn, where: str) -> None:
        """Classify the assumed predicate by its producing equation and
        narrow where — and only where — the narrowing is provably a
        superset of the true assumed region. Everything else stays inert
        with the reason disclosed."""
        narrowed: list[tuple[int, iv.IntervalArray, bool, bool]] = []
        dropped: list[str] = []
        vacuous: list[str] = []
        self._apply_assumed_pred(eqn.invars[0], where, narrowed, dropped, vacuous)
        for desc in vacuous:
            # audit F2: the branch-scoped unsatisfiability disclosure
            self.notes.append(
                f"assume unsatisfiable within this cond branch at {where}: "
                f"the precondition is definitely false whenever this branch "
                f"is taken — constraint not applied; obligations in this "
                f"branch may be vacuous under the branch's precondition "
                f"({desc})"
            )
        if narrowed:
            self.any_constrained = True
            self.counter.record_constrained(eqn.primitive)
            # audit F5: conjuncts that did NOT apply inside a constrained
            # assume must be visible in the coverage summary, not only in
            # the notes
            for _ in (*dropped, *vacuous):
                self.counter.record_dropped_conjunct()
            for var_id, box, changed, certified in narrowed:
                if changed:
                    self.notes.append(
                        f"assume CONSTRAINED at {where}: narrowed var "
                        f"{var_id} to {_render_box(box)}"
                    )
                else:
                    self.notes.append(
                        f"assume CONSTRAINED at {where}: var {var_id} "
                        f"already within the assumed region {_render_box(box)}"
                    )
                self.assumptions.add(
                    f"constrained assume at {where}: the verdict holds "
                    f"where the precondition holds — narrowed var {var_id} "
                    f"to {_render_box(box)}"
                )
                if not certified:
                    # audit F7: the target's box is an over-approximation
                    # (a transfer output / bracketed const), so a nonempty
                    # meet does NOT certify the true assumed region
                    # nonempty. The narrowing itself is sound (superset of
                    # true-region ∩ reachable) and stays applied; the
                    # conditional claim may be vacuous, and every
                    # subsequent definite violation is withheld from
                    # REFUTED.
                    self.uncertified = True
                    self.notes.append(
                        f"precondition satisfiability UNCERTIFIED at "
                        f"{where}: var {var_id} is an over-approximated "
                        f"intermediate (its box may exceed its true image) "
                        f"— the conditional claim may be vacuous"
                    )
                    self.assumptions.add(
                        "precondition satisfiability uncertified: a "
                        "constraining assume narrowed an over-approximated "
                        "intermediate whose box may exceed its true image "
                        "— the conditional claim may be vacuous; the "
                        "inert-mode control is the visibility instrument"
                    )
            for reason in dropped:
                # a conjunction can mix constrainable and inert conjuncts;
                # the un-narrowable part is still a drop and still says so
                self.notes.append(
                    f"assume conjunct DROPPED at {where}: constraining "
                    f"proceeded without this conjunct — a superset ({reason})"
                )
        else:
            self.counter.record_inert(eqn.primitive)
            if dropped or not vacuous:
                # an assume whose only content was a branch-scoped
                # unsatisfiable conjunct already carries the directed note
                # above; everything else gets the pre-existing DROPPED
                # disclosure with the reason(s) appended
                reasons = "; ".join(dropped) if dropped else "unclassified predicate"
                self.notes.append(
                    f"assume constraint DROPPED (inert in MVP propagation) at {where}: "
                    f"VERIFIED proves a superset; UNKNOWN may be confounded by this drop"
                    f" ({reasons})"
                )

    def _apply_assumed_pred(
        self,
        atom: ir.Atom,
        where: str,
        narrowed: list[tuple[int, iv.IntervalArray, bool, bool]],
        dropped: list[str],
        vacuous: list[str],
    ) -> None:
        if isinstance(atom, ir.Literal):
            dropped.append(
                f"predicate is a literal ({atom.val!r}), not a traced comparison"
            )
            return
        producer = self.producers.get(atom.id)
        if producer is None:
            dropped.append(
                "the predicate's producing equation is not visible in this "
                "scope (constvar, invar, or cross-scope value)"
            )
            return
        prim = producer.primitive
        if prim == "and":
            # a conjunction holds iff both conjuncts hold: recurse,
            # classifying each independently. jax's `and` is BITWISE — it
            # is the logical connective on bool operands only.
            if any(v.aval.dtype != "bool" for v in producer.invars):
                dropped.append(
                    "'and' on non-bool operands is bit arithmetic, not "
                    "conjunction"
                )
                return
            for conj in producer.invars:
                self._apply_assumed_pred(conj, where, narrowed, dropped, vacuous)
            return
        if prim not in _ASSUME_CMPS:
            dropped.append(
                f"predicate produced by {prim!r} admits no sound box narrowing"
            )
            return
        if len(producer.invars) != 2:
            dropped.append(
                f"comparison {prim!r} with {len(producer.invars)} operand(s)"
            )
            return
        if self.semantics == "ieee":
            # re-attack U2's swept surface: assume classification consumes
            # the comparison EQUATION directly, bypassing the guarded
            # comparison transfer — the same invariant must be enforced
            # here too. A non-f64-float comparison must neither narrow,
            # nor certify satisfiability, nor raise the unsatisfiable-
            # precondition oracle (a "definitely false" f32-band
            # comparison can be TRUE at runtime under per-dtype DAZ:
            # measured float32(1e-45) == float32(1e-40) is True) — inert
            # with the gap quoted is the only sound posture.
            bad = _non_f64_float_dtypes(producer.invars)
            if bad:
                dropped.append(
                    f"ieee mode models binary64 only; {'/'.join(bad)} "
                    f"comparison semantics (incl. per-dtype subnormal "
                    f"flush) are not modelled — no narrowing, no "
                    f"satisfiability claim"
                )
                return
        a, b = producer.invars
        box_a, box_b = self._quiet_interval(a), self._quiet_interval(b)
        point_a, point_b = _finite_point(box_a), _finite_point(box_b)
        if point_a and point_b:
            # no variable to narrow — but a point-vs-point comparison is
            # DECIDABLE (never indeterminate: order comparisons of two
            # degenerate finite intervals are definite, and two points are
            # either equal or disjoint). Definitely false means the
            # precondition is unsatisfiable over the declared set — the
            # same harness defect the empty meet names, so the same loud
            # refusal. Definitely true stays a no-op inert drop.
            try:
                result = _CMP_FN[prim](box_a, box_b)
            except iv.IntervalError as e:
                dropped.append(
                    f"point comparison sides do not broadcast ({e})"
                )
                return
            if any(
                (lo, hi) == iv.BOOL_FALSE
                for lo, hi in zip(result.los, result.his)
            ):
                # under ieee a NaN side would ALSO falsify the comparison,
                # so the definitely-false refusal stands with or without a
                # maybe-NaN flag on either side
                self._unsatisfiable(
                    where,
                    f"constant comparison {_render_bound(box_a)} "
                    f"{_CMP_SYMBOL[prim]} {_render_bound(box_b)} is "
                    f"definitely false",
                    vacuous,
                    f"unsatisfiable assume at {where}: the assumed "
                    f"comparison {_render_bound(box_a)} {_CMP_SYMBOL[prim]} "
                    f"{_render_bound(box_b)} is definitely false — the "
                    f"precondition is unsatisfiable over the declared set "
                    f"and every downstream obligation would be vacuous "
                    f"(harness defect; nothing was verified)",
                )
                return
            if self.semantics == "ieee" and (
                self.read_flag(a) or self.read_flag(b)
            ):
                # the interval parts compare definitely true, but a NaN
                # side would falsify — "definitely true" cannot be claimed
                dropped.append(
                    "a comparison side may be NaN under ieee semantics "
                    "(NaN falsifies the comparison) — the assumed "
                    "comparison is not certified true; dropped"
                )
                return
            dropped.append(
                "both comparison sides are point intervals — nothing to "
                "narrow (the assumed comparison is definitely true)"
            )
            return
        if not point_a and not point_b:
            # audit F6: a constant-but-not-finite bound (±inf, NaN, an
            # undecodable literal) is not a varying side — the disclosed
            # reason must name the actual shape. Under ieee a subnormal-
            # band literal is hazed to a non-point for the same reason,
            # and gets its own truthful reason.
            if self.semantics == "ieee" and (
                _subnormal_const_literal(a) or _subnormal_const_literal(b)
            ):
                dropped.append(_SUBNORMAL_BOUND_REASON)
            elif _nonfinite_const(a, box_a) or _nonfinite_const(b, box_b):
                dropped.append(_NONFINITE_BOUND_REASON)
            else:
                dropped.append(_RELATIONAL_REASON)
            return
        if point_a:
            target_atom, target_box, bound, bound_atom = b, box_b, box_a, a
            cmp = _CMP_FLIP[prim]  # cmp(k, v) === flipped-cmp(v, k)
        else:
            target_atom, target_box, bound, bound_atom = a, box_a, box_b, b
            cmp = prim
        if self.semantics == "ieee" and self.read_flag(bound_atom):
            # a maybe-NaN bound: if the bound IS NaN the true assumed
            # region is empty (NaN falsifies every _ASSUME_CMPS
            # comparison), so a half-space built from its interval could
            # certify a vacuous precondition — no certified half-space
            # represents a maybe-NaN bound; inert is the sound posture
            dropped.append(
                "the comparison bound may be NaN under ieee semantics "
                "(its producer carries maybe-NaN) — no certified "
                "half-space represents it"
            )
            return
        if not isinstance(target_atom, ir.Var):
            dropped.append(
                "the varying comparison side is a literal, not an "
                "environment variable"
            )
            return
        if target_box is None:
            dropped.append(
                f"var {target_atom.id} has no propagated interval to narrow"
            )
            return
        # bound shape: a scalar broadcast over the variable, or an exact
        # elementwise match — nothing else (general broadcasting between
        # the bound and the variable is not attempted; inert is sound).
        if not (bound.is_scalar() or bound.shape == target_box.shape):
            dropped.append(
                f"bound shape {bound.shape} is neither scalar nor equal to "
                f"the constrained variable's shape {target_box.shape}"
            )
            return
        n = target_box.size
        ks = bound.los * n if bound.is_scalar() else bound.los
        # the CLOSED half-space, deliberately: for strict cmps the true
        # ℝ-region (k, hi] must be contained in the narrowing, and
        # [nextafter(k), hi] excludes the reals in (k, nextafter(k)) — an
        # unsound under-approximation (binding rule 1).
        if cmp in ("ge", "gt"):
            half = iv.IntervalArray(
                shape=target_box.shape, los=ks, his=(math.inf,) * n
            )
        elif cmp in ("le", "lt"):
            half = iv.IntervalArray(
                shape=target_box.shape, los=(-math.inf,) * n, his=ks
            )
        else:  # eq
            half = iv.IntervalArray(shape=target_box.shape, los=ks, his=ks)
        try:
            new = iv.meet(target_box, half)  # exact: no outward rounding
        except iv.IntervalError:
            self._unsatisfiable(
                where,
                f"var {target_atom.id} ∈ {_render_box(target_box)} cannot "
                f"satisfy {_CMP_SYMBOL[cmp]} {_render_bound(bound)} (empty "
                f"meet)",
                vacuous,
                f"unsatisfiable assume at {where}: var {target_atom.id} has "
                f"propagated interval {_render_box(target_box)}, but the "
                f"assumed constraint requires var {target_atom.id} "
                f"{_CMP_SYMBOL[cmp]} {_render_bound(bound)}; the "
                f"precondition is definitely false on the whole "
                f"over-approximated domain, so the declared set as assumed "
                f"is empty and every downstream obligation would be vacuous "
                f"(harness defect; nothing was verified)",
            )
            return
        if cmp in ("gt", "lt") and any(
            new.los[i] == new.his[i] == ks[i] for i in range(n)
        ):
            # audit F1: a STRICT constraint whose meet collapses onto the
            # closed boundary point has an EMPTY true region ((k, k] or
            # [k, k)) — the closed narrowing cannot represent the
            # exclusion, but the emptiness is decidable right here, and a
            # definite verdict judged over the nonempty stand-in [k, k]
            # would misstate a claim about an empty admitted set (the
            # REFUTED face mints a false refutation). Same refusal class
            # as the empty meet. Elementwise: any collapsed element
            # empties the universal predicate.
            self._unsatisfiable(
                where,
                f"strict constraint var {target_atom.id} {_CMP_SYMBOL[cmp]} "
                f"{_render_bound(bound)} collapses onto the boundary point "
                f"— the true region is empty",
                vacuous,
                f"unsatisfiable assume at {where}: the strict constraint "
                f"var {target_atom.id} {_CMP_SYMBOL[cmp]} "
                f"{_render_bound(bound)} collapses the narrowed interval "
                f"onto the closed boundary point {_render_box(new)}, which "
                f"the strict comparison itself excludes — the true assumed "
                f"region is empty, so the precondition is definitely false "
                f"on the whole over-approximated domain and every "
                f"downstream obligation would be vacuous (harness defect; "
                f"nothing was verified)",
            )
            return
        # audit F8: an assume whose predicate is DEFINITELY TRUE over the
        # target's whole box certifies itself, exactness aside — pred true
        # on the box superset is true on the whole current domain, so the
        # assumed region EQUALS the (nonempty) reachable set and no
        # vacuity can arise from this assume. Checked with the
        # comparator's own strictness: a no-op meet does NOT suffice for
        # gt/lt at the boundary (box lo == k leaves x == k reachable and
        # excluded — pred not definitely true there), so the strict forms
        # demand strict inequality against the bound.
        if cmp == "ge":
            def_true = all(lo >= k for lo, k in zip(target_box.los, ks))
        elif cmp == "gt":
            def_true = all(lo > k for lo, k in zip(target_box.los, ks))
        elif cmp == "le":
            def_true = all(hi <= k for hi, k in zip(target_box.his, ks))
        elif cmp == "lt":
            def_true = all(hi < k for hi, k in zip(target_box.his, ks))
        else:  # eq: definitely true only if the box IS the bound point
            def_true = all(
                lo == hi == k
                for lo, hi, k in zip(target_box.los, target_box.his, ks)
            )
        if self.semantics == "ieee" and self.read_flag(target_atom):
            # a maybe-NaN target: NaN falsifies the assumed comparison, so
            # the predicate is NOT definitely true over the value set even
            # when the interval part is — the audit-F8 self-certification
            # channel stays closed for flagged targets
            def_true = False
        self.env[target_atom.id] = new
        if self.semantics == "ieee" and self.nan.get(target_atom.id):
            # an assumed-true comparison excludes NaN (NaN would falsify
            # it), so the narrowed target's maybe-NaN flag is soundly
            # cleared — the clearing is a judgement call the spec flags;
            # disclosed here so no verdict rests on it silently
            self.nan[target_atom.id] = False
            self.notes.append(
                f"assume cleared maybe-NaN on var {target_atom.id} at "
                f"{where}: an assumed-true comparison excludes NaN"
            )
        narrowed.append(
            (
                target_atom.id,
                new,
                new != target_box,
                # certified iff the target's box is EXACT (its true value
                # set: a stelling_any output or an exact-point const,
                # narrowed only by other assumes — meets of exact sets are
                # exact), OR the predicate is definitely true over the box
                # (audit F8). Everything else — a cutting assume on a
                # transfer output — is an over-approximation (audit F7).
                # The decision is the shared primitive in
                # stelling.exactness (module-attr call, so tests can pin
                # the routing). Under ieee, a STRICT certification
                # additionally needs a flush-robust witness: a strict
                # region whose only content is subnormal may be empty at
                # runtime on a DAZ target (its members read as 0, which
                # the strict comparison excludes) — without one the
                # precondition stays uncertified (indeterminate, never
                # definite).
                exactness.certifies_nonemptiness(
                    self.exact, target_atom.id, definitely_true=def_true
                )
                and not (
                    self.semantics == "ieee"
                    and cmp in ("gt", "lt")
                    and not def_true
                    and not all(
                        _strict_flush_witness(cmp, k, nlo, nhi)
                        for k, nlo, nhi in zip(ks, new.los, new.his)
                    )
                ),
            )
        )

    def run(
        self, jaxpr: ir.Jaxpr, consts, args, arg_flags=None
    ) -> list[iv.IntervalArray]:
        ieee = self.semantics == "ieee"
        for var, c in zip(jaxpr.constvars, consts):
            if isinstance(c, iv.IntervalArray):
                self.env[var.id] = c  # pre-boxed const: provenance unknown,
                if ieee:  # so non-exact (conservative — audit F7) and,
                    # under ieee, maybe-NaN (a box of unknown provenance
                    # carries no NaN-freedom claim)
                    self.nan[var.id] = True
                continue
            try:
                box = _value_to_interval(c, var.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError) as e:
                # same posture as literals: an unrepresentable const binds ⊤
                # (audit-gate finding 1 — a NaN closure const killed the run)
                self.notes.append(f"const outside the domain ({e}); ⊤")
                self.env[var.id] = iv.top(var.aval.shape)
                if ieee:  # ⊤ under ieee is maybe-NaN
                    self.nan[var.id] = True
                continue
            if ieee:
                # DAZ flushes inputs: constants entering ieee propagation
                # are subnormal-hazed; a band const stops being a point,
                # so mark_if_point below withholds exactness by itself
                box = iv.subnormal_haze(box)[0]
            self.env[var.id] = box
            # exact iff the decoded box is a point per element — a >2**53
            # int decodes to a genuine bracket, which is NOT its value set
            self.exact.mark_if_point(var.id, box.los, box.his)
        for i, (var, a) in enumerate(zip(jaxpr.invars, args)):
            self.env[var.id] = a
            if ieee and arg_flags is not None and arg_flags[i]:
                self.nan[var.id] = True
        # assume classification looks up the predicate's producing equation
        # at the CURRENT jaxpr level only; sub-jaxpr runs (transparent
        # wrappers, cond branches) get their own map, restored on exit —
        # same scoping discipline as the env swap the callers perform.
        prev_producers = self.producers
        self.producers = {
            out.id: e for e in jaxpr.eqns for out in e.outvars
        }
        try:
            for eqn in jaxpr.eqns:
                self.eqn(eqn)
        finally:
            self.producers = prev_producers
        return [self.read(o) for o in jaxpr.outvars]

    def eqn(self, eqn: ir.JaxprEqn) -> None:
        params = eqn.params_dict()
        if eqn.primitive in DEFAULT_TRANSPARENT:
            inner = next(
                (v for _, v in eqn.params if isinstance(v, ir.ClosedJaxpr)), None
            )
            ins = [self.read(a) for a in eqn.invars]
            if inner is not None and len(inner.jaxpr.invars) == len(ins):
                self.counter.record_transparent(eqn.primitive)
                ieee = self.semantics == "ieee"
                in_flags = (
                    [self.read_flag(a) for a in eqn.invars] if ieee else None
                )
                outer_env = self.env  # isolated scope, as for cond branches
                outer_exact = self.exact
                outer_nan = self.nan
                self.env = {}
                self.exact = exactness.ExactSet()
                self.nan = {}
                out_flags = None
                try:
                    outs = self.run(inner.jaxpr, inner.consts, ins, in_flags)
                    if ieee:
                        out_flags = [
                            self.read_flag(o) for o in inner.jaxpr.outvars
                        ]
                finally:
                    self.env = outer_env
                    self.exact = outer_exact
                    self.nan = outer_nan
                for out, val in zip(eqn.outvars, outs):
                    self.env[out.id] = val
                if ieee:
                    for out, f in zip(eqn.outvars, out_flags):
                        self.nan[out.id] = f
                return
            self.notes.append(
                f"transparent {eqn.primitive!r}: arity mismatch or no sub-jaxpr; ⊤"
            )
            self.counter.record_unknown(eqn.primitive)
            self.mark_unreached(eqn)
            self.top_out(eqn)
            return

        if eqn.primitive == "cond":
            # cond(index, *operands)[branches=(jaxpr, ...)]: run every branch
            # the index interval can select and JOIN outputs (sound
            # over-approximation of an undetermined branch). Out-of-range
            # indices select the FINAL branch — the jax convention, verified
            # by binding cond_p directly (index -1 -> last, index 5 -> last;
            # audit finding 2: clamping negative indices to branch 0 produced
            # a verified false VERIFIED at the [-1, 0] boundary).
            branches = next(
                (v for k, v in eqn.params if k == "branches"), None
            )
            ins = [self.read(a) for a in eqn.invars]
            if branches is None or not ins:
                self.notes.append("cond without branches/operands; ⊤")
                self.counter.record_unknown(eqn.primitive)
                self.mark_unreached(eqn)
                self.top_out(eqn)
                return
            ieee = self.semantics == "ieee"
            op_flags = (
                [self.read_flag(a) for a in eqn.invars[1:]] if ieee else None
            )
            # under ieee the index invariant is enforced, not assumed
            # (the F1/U2 lesson): jax's cond index is always int32, so a
            # FLOAT-dtyped index is out-of-contract hand-built IR whose
            # value could flush per-dtype — its interval is untrusted and
            # every branch is joined; same for a maybe-NaN-flagged index
            index_untrusted = ieee and (
                self.read_flag(eqn.invars[0])
                or eqn.invars[0].aval.dtype not in _SELECTOR_DTYPES
            )
            index, operands = ins[0], ins[1:]
            last = len(branches) - 1
            w_lo, w_hi = index.los[0], index.his[0]
            if w_lo == -math.inf or w_hi == math.inf or index_untrusted:
                # ⊤ index, or an untrusted (flagged / non-int) index
                # under ieee: any branch
                possible = set(range(last + 1))
            else:
                lo_i, hi_i = int(math.floor(w_lo)), int(math.floor(w_hi))
                possible = {i for i in range(last + 1) if lo_i <= i <= hi_i}
                if lo_i < 0 or hi_i > last:
                    possible.add(last)  # out-of-range mass -> the final branch
            if not possible:
                possible = {last}
            self.counter.record_known(eqn.primitive)
            self.used[eqn.primitive] = TIER_EXACT
            # untaken branches are still part of the query: their equations
            # count as unreached, or the coverage denominator lies (audit
            # finding 4)
            for i, b in enumerate(branches):
                if i not in possible:
                    stack = [b.jaxpr]
                    while stack:
                        jx = stack.pop()
                        for e in jx.eqns:
                            self.counter.record_unreached(e.primitive)
                            stack.extend(sub_jaxprs(e))
            # each branch runs in an isolated scope: branch jaxprs are closed
            # (invars + consts only), and a shared flat env would let one
            # branch read another's internals, defeating the unbound-var
            # check (audit finding 6)
            outer_env = self.env
            outer_exact = self.exact
            outer_nan = self.nan
            results = []
            branch_flags = []  # ieee: per-branch outvar flags
            # every branch here is possibly-untaken from the analysis's
            # view (the selector interval admits it, nothing more): an
            # unsatisfiable assume inside is a branch-scoped vacuity, not
            # a whole-domain harness defect (audit F2) — branch_depth
            # switches the unsatisfiability posture from raise to a
            # branch-local note. Conservative even for a definite
            # single-branch selector: not raising is always sound.
            self.branch_depth += 1
            try:
                for i in sorted(possible):
                    b = branches[i]
                    self.env = {}
                    self.exact = exactness.ExactSet()
                    self.nan = {}
                    results.append(
                        self.run(b.jaxpr, list(b.consts), operands, op_flags)
                    )
                    if ieee:
                        branch_flags.append(
                            [self.read_flag(o) for o in b.jaxpr.outvars]
                        )
            finally:
                self.branch_depth -= 1
                self.env = outer_env
                self.exact = outer_exact
                self.nan = outer_nan
            for j, out in enumerate(eqn.outvars):
                self.env[out.id] = iv.join([r[j] for r in results])
                if ieee:
                    # the output is SOME branch's output whatever the
                    # index value is (out-of-range selects the final
                    # branch), so the join's flag is the OR over possible
                    # branches — the index's own flag does not propagate
                    self.nan[out.id] = any(f[j] for f in branch_flags)
            return

        if eqn.primitive == "stelling_assume":
            # value semantics: the identity on the predicate — the assume's
            # output passes its input through unchanged in BOTH modes
            # (conservative: the output interval computed before any
            # narrowing is a superset of the predicate's value under the
            # assumption). The constraint semantics differ by mode.
            ins = [self.read(a) for a in eqn.invars]
            in_flags = (
                [self.read_flag(a) for a in eqn.invars]
                if self.semantics == "ieee"
                else None
            )
            where = eqn.source_info[-1] if eqn.source_info else "unknown location"
            if self.assume_mode == "constrain" and eqn.invars:
                self._assume_constrain(eqn, where)
            else:
                # inert (amendment 2): sound, counted, addressed — never
                # silent, never "known". Byte-identical to the MVP note.
                self.counter.record_inert(eqn.primitive)
                self.notes.append(
                    f"assume constraint DROPPED (inert in MVP propagation) at {where}: "
                    f"VERIFIED proves a superset; UNKNOWN may be confounded by this drop"
                )
            for i, (out, val) in enumerate(zip(eqn.outvars, ins)):
                self.env[out.id] = val
                if in_flags is not None:
                    self.nan[out.id] = in_flags[i]
            return

        ieee = self.semantics == "ieee"
        entry = (IEEE_TRANSFERS if ieee else TRANSFERS).get(eqn.primitive)
        if entry is None:
            self.counter.record_unknown(eqn.primitive)
            self.mark_unreached(eqn)
            self.top_out(eqn)
            return

        transfer, tier = entry
        ins = [self.read(a) for a in eqn.invars]
        in_flags = [self.read_flag(a) for a in eqn.invars] if ieee else None
        try:
            result = (
                transfer(eqn, params, ins, in_flags)
                if ieee
                else transfer(eqn, params, ins)
            )
        except iv.IntervalError as e:
            # a transfer whose domain doesn't cover this legal form (rank
            # broadcasting, batched selectors, …) DECLINES: sound ⊤
            # degradation with the reason quoted — the registered
            # degrade-don't-crash posture (second audit, FRAGILE 5; the
            # shape guards previously killed the whole analysis here).
            # Under ieee, top_out marks the outputs maybe-NaN.
            self.notes.append(f"{eqn.primitive!r} declined this form: {e}; ⊤")
            self.counter.record_unknown(eqn.primitive)
            self.top_out(eqn)
            return
        if result is None:  # a known transfer declining this configuration
            self.notes.append(
                f"{eqn.primitive!r} has no sound rule for params "
                f"{ {k: v for k, v in params.items() if not isinstance(v, ir.ClosedJaxpr)} }; ⊤"
            )
            self.counter.record_unknown(eqn.primitive)
            self.top_out(eqn)
            return
        outs, out_flags = result if ieee else (result, None)
        self.counter.record_known(eqn.primitive)
        self.used[eqn.primitive] = tier
        if tier == TIER_SOUND_LIBM:
            self.assumptions.add(
                _LIBM_ASSUMPTIONS.get(
                    eqn.primitive,
                    f"{eqn.primitive} endpoints assume a faithfully-rounded "
                    f"libm (error <= 1 ulp), bumped 1 ulp outward",
                )
            )
        if eqn.primitive == "stelling_assert":
            status, detail = _bool_status(ins[0], constrained=self.any_constrained)
            if ieee and in_flags and in_flags[0] and status != "unknown":
                # the predicate VALUE arrived flagged maybe-NaN (a decline
                # artifact ⊤ reaching the assert, or a flagged selector
                # path): a bool cannot BE NaN, so the flag marks untrusted
                # provenance — neither definite face is claimed
                where = (
                    eqn.source_info[-1] if eqn.source_info else "unknown location"
                )
                self.notes.append(
                    f"obligation withheld from a definite status at {where}: "
                    f"the predicate value carries maybe-NaN under ieee "
                    f"semantics (its producer was declined/unmodeled) — "
                    f"judged unknown"
                )
                status = "unknown"
                detail = (
                    "predicate value carries maybe-NaN under ieee semantics "
                    "(see notes); no definite status is claimed"
                )
            if status == "violated-over-set" and self.uncertified:
                # audit F7: a definite violation judged while an
                # UNCERTIFIED-precondition constraint is in force is
                # withheld from REFUTED — the narrowed superset it was
                # judged over may over-approximate an EMPTY true region,
                # and a possibly-vacuous refutation is not a refutation.
                where = (
                    eqn.source_info[-1] if eqn.source_info else "unknown location"
                )
                self.notes.append(
                    f"violation WITHHELD from REFUTED at {where}: a "
                    f"definite violation was found over the narrowed "
                    f"superset, but the precondition's satisfiability is "
                    f"uncertified (it constrains an over-approximated "
                    f"intermediate whose box may exceed its true image) — "
                    f"a possibly-vacuous refutation is not a refutation; "
                    f"REFUTED under constraining assumes requires "
                    f"certified-satisfiable preconditions"
                )
                status = "unknown"
                detail = (
                    "definite violation over the precondition-narrowed "
                    "superset WITHHELD from REFUTED (precondition "
                    "satisfiability uncertified; see notes)"
                )
            self.obligations.append(
                ObligationReport(
                    index=len(self.obligations),
                    status=status,
                    detail=detail,
                    source_info=eqn.source_info,
                )
            )
        if eqn.primitive == "stelling_nonvacuity":
            status, detail = _bool_status(ins[0], constrained=self.any_constrained)
            if ieee and in_flags and in_flags[0] and status != "unknown":
                # same posture as the assert above: a maybe-NaN membership
                # condition supports neither the checked nor the FAILED face
                where = (
                    eqn.source_info[-1] if eqn.source_info else "unknown location"
                )
                self.notes.append(
                    f"nonvacuity condition withheld from a definite status "
                    f"at {where}: the membership predicate carries "
                    f"maybe-NaN under ieee semantics — judged unknown"
                )
                status = "unknown"
                detail = (
                    "membership condition carries maybe-NaN under ieee "
                    "semantics (see notes); no definite status is claimed"
                )
            if status == "violated-over-set" and self.uncertified:
                # audit F9: the definite FAILED stamp sentence is the same
                # claim class the assert withholding guards — a membership
                # condition judged definitely false over an
                # uncertified-narrowed region may be vacuously so. Same
                # forward scoping as the assert withholding; certified and
                # exact runs keep FAILED byte-identically.
                where = (
                    eqn.source_info[-1] if eqn.source_info else "unknown location"
                )
                self.notes.append(
                    f"nonvacuity FAILED face WITHHELD at {where}: the "
                    f"membership condition was judged definitely false over "
                    f"an uncertified-narrowed region (a constraining assume "
                    f"cut an over-approximated intermediate whose box may "
                    f"exceed its true image) — the FAILED sentence is "
                    f"reserved for judgments not confounded by an "
                    f"uncertified constraint"
                )
                status = "unknown"
                detail = (
                    "membership condition definitely false over the "
                    "precondition-narrowed superset WITHHELD from FAILED "
                    "(precondition satisfiability uncertified; see notes)"
                )
            self.nonvacuity_checks.append(
                ObligationReport(
                    index=len(self.nonvacuity_checks),
                    status=status,
                    detail=detail,
                    source_info=eqn.source_info,
                )
            )
        for i, (out, val) in enumerate(zip(eqn.outvars, outs)):
            self.env[out.id] = val
            if ieee:
                self.nan[out.id] = out_flags[i]
        if eqn.primitive == "stelling_any":
            # the ONE transfer whose output box is exact: the declared
            # closed box IS the declared value set (no rounding at
            # declaration). Every other transfer output stays non-exact
            # (audit F7: rounding pads and correlation-blind arithmetic
            # can inflate a box past the true image). Under ieee, a
            # declaration the subnormal haze CHANGED is not exact: its
            # as-consumed value set is flush-indeterminate (the hazed box
            # is a sound hull of both semantics, not the declared set),
            # so it must not certify assume-satisfiability.
            if ieee:
                raw = iv.from_bounds(
                    tuple(params["shape"]),
                    float(params["lo"]),
                    float(params["hi"]),
                )
                if iv.subnormal_haze(raw)[1]:
                    return
            for out in eqn.outvars:
                self.exact.mark_declared(out.id)


def _check_assume_mode(assume_mode: str) -> None:
    if assume_mode not in _ASSUME_MODES:
        raise ValueError(
            f"assume_mode must be one of {_ASSUME_MODES}, got {assume_mode!r}"
        )


_SEMANTICS_MODES = ("real", "ieee")
_DOMAINS = ("interval",)

# The mechanical guard on the domain dial: quoted verbatim when a
# tightened (non-interval) domain is requested under semantics="real".
TIGHTENED_DOMAIN_REAL_REFUSAL = (
    "a tightened domain under semantics='real' is refused outright: "
    "tightening ℝ arithmetic without float semantics converts accidental "
    "UNKNOWNs into false VERIFIEDs (the interval slack was masking the "
    "ℝ-vs-float gap, not modeling it) — tightened domains run only under "
    "semantics='ieee'"
)


def _check_semantics(semantics: str) -> None:
    if semantics not in _SEMANTICS_MODES:
        raise ValueError(
            f"semantics must be one of {_SEMANTICS_MODES}, got {semantics!r}"
        )


def _check_domain(domain: str, semantics: str) -> None:
    """Guard 1: ``"interval"`` is the only registered domain. A
    non-interval value always raises; under ``semantics="real"`` the
    refusal carries the rationale a future tightened domain must not be
    allowed to erode (:data:`TIGHTENED_DOMAIN_REAL_REFUSAL`)."""
    if domain in _DOMAINS:
        return
    if semantics == "real":
        raise ValueError(
            f"domain must be one of {_DOMAINS} (the only registered "
            f"domain), got {domain!r}. {TIGHTENED_DOMAIN_REAL_REFUSAL}"
        )
    raise ValueError(
        f"domain must be one of {_DOMAINS} (the only registered domain), "
        f"got {domain!r}"
    )


def interval_env(
    closed: ir.ClosedJaxpr, *, assume_mode: str = "inert"
) -> dict[int, iv.IntervalArray]:
    """Read-only accessor: the top-level ``var id -> interval`` environment
    after the forward walk :func:`propagate` performs under ``assume_mode``.

    Exists for the escalation layer, whose division guard needs the
    propagated interval of a divisor ("emit ``div`` only if the divisor's
    interval definitely excludes 0"). Pure: it re-runs the identical
    traversal on a fresh propagator and returns a copy of the resulting
    environment; nothing observable by :func:`propagate` changes.

    The default here is ``"inert"``, deliberately NOT :func:`propagate`'s
    ``"constrain"`` default: this accessor's consumer reasons over the
    DECLARED box. The emitted SMT problem carries the ``stelling_any``
    bounds and never the assume constraints, so its guards' premises
    ("the divisor definitely excludes 0 *over the declared box*") must be
    judged on the un-narrowed environment — an assume-narrowed divisor
    interval would let ``div`` be emitted into a script whose domain
    still contains the zero, and a model at that zero would be
    misdiagnosed as emission infidelity. Un-narrowed intervals are wider,
    so every guard judged on them is conservative — inert is always
    sound. Callers that want the constrained view ask for it explicitly.
    """
    _check_assume_mode(assume_mode)
    p = _Propagator(assume_mode)
    if closed.jaxpr.invars:
        raise ir.TranscriptionError(
            "propagate expects a self-contained harness query (inputs declared "
            f"via any_array), got {len(closed.jaxpr.invars)} free invar(s)"
        )
    p.run(closed.jaxpr, list(closed.consts), [])
    return dict(p.env)


def propagate(
    closed: ir.ClosedJaxpr,
    *,
    semantics: str = "real",
    assume_mode: str = "constrain",
    domain: str = "interval",
) -> Propagation:
    """Forward-propagate the declared boxes through a transcribed query and
    judge every ``stelling_assert`` obligation.

    ``semantics="real"`` (the default) judges obligations in exact real
    arithmetic — byte-identical to the pre-dial behavior;
    ``semantics="ieee"`` judges them about the traced program's IEEE
    binary64 round-to-nearest execution (see the module docstring). Any
    other value raises :class:`ValueError`. The returned
    :class:`Propagation` records which semantics ran.

    ``assume_mode="constrain"`` (the default) narrows the propagated
    domain at each soundly-constrainable ``stelling_assume`` (see the
    module docstring); ``assume_mode="inert"`` reproduces the
    drop-every-assume MVP behavior byte-identically (notes, coverage,
    env) — the comparability control for the vacuity instrument. Any
    other value raises :class:`ValueError`.

    ``domain="interval"`` is the only registered abstract domain. Any
    other value raises :class:`ValueError`; under ``semantics="real"``
    the refusal quotes why a tightened domain may never run there
    (:data:`TIGHTENED_DOMAIN_REAL_REFUSAL`).
    """
    _check_semantics(semantics)
    _check_assume_mode(assume_mode)
    _check_domain(domain, semantics)
    p = _Propagator(assume_mode, semantics)
    if closed.jaxpr.invars:
        raise ir.TranscriptionError(
            "propagate expects a self-contained harness query (inputs declared "
            f"via any_array), got {len(closed.jaxpr.invars)} free invar(s)"
        )
    p.run(closed.jaxpr, list(closed.consts), [])
    assumptions = set(p.assumptions)
    if semantics == "ieee":
        # the mode-wide stamped assumptions: how ieee endpoints are
        # computed and what their soundness relies on, and the
        # subnormal-band indeterminacy (flush-vs-gradual is
        # target-dependent; band outcomes are never definite)
        assumptions.add(iv.IEEE_ENDPOINT_ASSUMPTION)
        assumptions.add(iv.SUBNORMAL_INDETERMINACY_ASSUMPTION)
    return Propagation(
        obligations=tuple(p.obligations),
        nonvacuity_checks=tuple(p.nonvacuity_checks),
        coverage=p.counter.freeze(),
        transfers_used=tuple(sorted(p.used.items())),
        assumptions=tuple(sorted(assumptions)),
        notes=tuple(p.notes),
        semantics=semantics,
    )
