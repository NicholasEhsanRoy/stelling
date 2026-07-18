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
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from stelling import interval as iv
from stelling import ir
from stelling.coverage import DEFAULT_TRANSPARENT, Coverage, CoverageCounter, sub_jaxprs

__all__ = [
    "ObligationReport",
    "Propagation",
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
    def __init__(self, assume_mode: str) -> None:
        self.assume_mode = assume_mode
        self.env: dict[int, iv.IntervalArray] = {}
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
        # descent): a scope certifies only its own declarations.
        self.exact: set[int] = set()
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
                return _value_to_interval(atom.val, atom.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError) as e:
                self.notes.append(f"literal outside the domain ({e}); ⊤")
                return iv.top(atom.aval.shape)
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

    def top_out(self, eqn: ir.JaxprEqn) -> None:
        for out in eqn.outvars:
            self.env[out.id] = iv.top(out.aval.shape)

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
                return _value_to_interval(atom.val, atom.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError):
                return None
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
            dropped.append(
                "both comparison sides are point intervals — nothing to "
                "narrow (the assumed comparison is definitely true)"
            )
            return
        if not point_a and not point_b:
            # audit F6: a constant-but-not-finite bound (±inf, NaN, an
            # undecodable literal) is not a varying side — the disclosed
            # reason must name the actual shape
            if _nonfinite_const(a, box_a) or _nonfinite_const(b, box_b):
                dropped.append(_NONFINITE_BOUND_REASON)
            else:
                dropped.append(_RELATIONAL_REASON)
            return
        if point_a:
            target_atom, target_box, bound = b, box_b, box_a
            cmp = _CMP_FLIP[prim]  # cmp(k, v) === flipped-cmp(v, k)
        else:
            target_atom, target_box, bound = a, box_a, box_b
            cmp = prim
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
        self.env[target_atom.id] = new
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
                target_atom.id in self.exact or def_true,
            )
        )

    def run(self, jaxpr: ir.Jaxpr, consts, args) -> list[iv.IntervalArray]:
        for var, c in zip(jaxpr.constvars, consts):
            if isinstance(c, iv.IntervalArray):
                self.env[var.id] = c  # pre-boxed const: provenance unknown,
                continue  # so non-exact (conservative — audit F7)
            try:
                box = _value_to_interval(c, var.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError) as e:
                # same posture as literals: an unrepresentable const binds ⊤
                # (audit-gate finding 1 — a NaN closure const killed the run)
                self.notes.append(f"const outside the domain ({e}); ⊤")
                self.env[var.id] = iv.top(var.aval.shape)
                continue
            self.env[var.id] = box
            # exact iff the decoded box is a point per element — a >2**53
            # int decodes to a genuine bracket, which is NOT its value set
            if all(lo == hi for lo, hi in zip(box.los, box.his)):
                self.exact.add(var.id)
        for var, a in zip(jaxpr.invars, args):
            self.env[var.id] = a
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
                outer_env = self.env  # isolated scope, as for cond branches
                outer_exact = self.exact
                self.env = {}
                self.exact = set()
                try:
                    outs = self.run(inner.jaxpr, inner.consts, ins)
                finally:
                    self.env = outer_env
                    self.exact = outer_exact
                for out, val in zip(eqn.outvars, outs):
                    self.env[out.id] = val
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
            index, operands = ins[0], ins[1:]
            last = len(branches) - 1
            w_lo, w_hi = index.los[0], index.his[0]
            if w_lo == -math.inf or w_hi == math.inf:
                possible = set(range(last + 1))  # ⊤ index: any branch
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
            results = []
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
                    self.exact = set()
                    results.append(self.run(b.jaxpr, list(b.consts), operands))
            finally:
                self.branch_depth -= 1
                self.env = outer_env
                self.exact = outer_exact
            for j, out in enumerate(eqn.outvars):
                self.env[out.id] = iv.join([r[j] for r in results])
            return

        if eqn.primitive == "stelling_assume":
            # value semantics: the identity on the predicate — the assume's
            # output passes its input through unchanged in BOTH modes
            # (conservative: the output interval computed before any
            # narrowing is a superset of the predicate's value under the
            # assumption). The constraint semantics differ by mode.
            ins = [self.read(a) for a in eqn.invars]
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
            for out, val in zip(eqn.outvars, ins):
                self.env[out.id] = val
            return

        entry = TRANSFERS.get(eqn.primitive)
        if entry is None:
            self.counter.record_unknown(eqn.primitive)
            self.mark_unreached(eqn)
            self.top_out(eqn)
            return

        transfer, tier = entry
        ins = [self.read(a) for a in eqn.invars]
        try:
            outs = transfer(eqn, params, ins)
        except iv.IntervalError as e:
            # a transfer whose domain doesn't cover this legal form (rank
            # broadcasting, batched selectors, …) DECLINES: sound ⊤
            # degradation with the reason quoted — the registered
            # degrade-don't-crash posture (second audit, FRAGILE 5; the
            # shape guards previously killed the whole analysis here)
            self.notes.append(f"{eqn.primitive!r} declined this form: {e}; ⊤")
            self.counter.record_unknown(eqn.primitive)
            self.top_out(eqn)
            return
        if outs is None:  # a known transfer declining this configuration
            self.notes.append(
                f"{eqn.primitive!r} has no sound rule for params "
                f"{ {k: v for k, v in params.items() if not isinstance(v, ir.ClosedJaxpr)} }; ⊤"
            )
            self.counter.record_unknown(eqn.primitive)
            self.top_out(eqn)
            return
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
        for out, val in zip(eqn.outvars, outs):
            self.env[out.id] = val
        if eqn.primitive == "stelling_any":
            # the ONE transfer whose output box is exact: the declared
            # closed box IS the declared value set (no rounding at
            # declaration). Every other transfer output stays non-exact
            # (audit F7: rounding pads and correlation-blind arithmetic
            # can inflate a box past the true image).
            for out in eqn.outvars:
                self.exact.add(out.id)


def _check_assume_mode(assume_mode: str) -> None:
    if assume_mode not in _ASSUME_MODES:
        raise ValueError(
            f"assume_mode must be one of {_ASSUME_MODES}, got {assume_mode!r}"
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
    closed: ir.ClosedJaxpr, *, assume_mode: str = "constrain"
) -> Propagation:
    """Forward-propagate the declared boxes through a transcribed query and
    judge every ``stelling_assert`` obligation.

    ``assume_mode="constrain"`` (the default) narrows the propagated
    domain at each soundly-constrainable ``stelling_assume`` (see the
    module docstring); ``assume_mode="inert"`` reproduces the
    drop-every-assume MVP behavior byte-identically (notes, coverage,
    env) — the comparability control for the vacuity instrument. Any
    other value raises :class:`ValueError`.
    """
    _check_assume_mode(assume_mode)
    p = _Propagator(assume_mode)
    if closed.jaxpr.invars:
        raise ir.TranscriptionError(
            "propagate expects a self-contained harness query (inputs declared "
            f"via any_array), got {len(closed.jaxpr.invars)} free invar(s)"
        )
    p.run(closed.jaxpr, list(closed.consts), [])
    return Propagation(
        obligations=tuple(p.obligations),
        nonvacuity_checks=tuple(p.nonvacuity_checks),
        coverage=p.counter.freeze(),
        transfers_used=tuple(sorted(p.used.items())),
        assumptions=tuple(sorted(p.assumptions)),
        notes=tuple(p.notes),
    )
