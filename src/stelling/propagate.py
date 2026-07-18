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
rank-broadcast forms of already-registered transfers). Everything else
falls to ⊤ — soundly, with coverage recording exactly how much fell.

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
the declared box — a **sound set-level refutation of the stated box**,
rendered as a REFUTED verdict per `design/e2a-registration.md` amendment
1; not a witness, not a counterexample to the program).

``stelling_assume`` is **inert** in this MVP: the constraint is dropped,
which is sound (propagation runs over a superset) and must never be
silent — each drop is counted in coverage as ``inert`` (outside the
"known" fraction) and noted with its source address (amendment 2).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from stelling import interval as iv
from stelling import ir
from stelling.coverage import DEFAULT_TRANSPARENT, Coverage, CoverageCounter, sub_jaxprs

__all__ = ["ObligationReport", "Propagation", "propagate"]

TIER_EXACT = "exact"
TIER_SOUND = "sound"
TIER_SOUND_LIBM = "sound-libm"

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
# reshape, pow, reduce_or — the primitives the probe's own ⊤ list named).
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


def _bool_status(b: iv.IntervalArray) -> tuple[str, str]:
    n_true = sum(1 for lo, hi in zip(b.los, b.his) if (lo, hi) == iv.BOOL_TRUE)
    n_false = sum(1 for lo, hi in zip(b.los, b.his) if (lo, hi) == iv.BOOL_FALSE)
    n = b.size
    if n_true == n:
        return "discharged", f"definitely true for all {n} element(s)"
    if n_false:
        return (
            "violated-over-set",
            f"definitely false for {n_false}/{n} element(s) over the declared box",
        )
    return "unknown", f"undecided for {n - n_true}/{n} element(s)"


class _Propagator:
    def __init__(self) -> None:
        self.env: dict[int, iv.IntervalArray] = {}
        self.counter = CoverageCounter()
        self.obligations: list[ObligationReport] = []
        self.nonvacuity_checks: list[ObligationReport] = []
        self.used: dict[str, str] = {}
        self.assumptions: set[str] = set()
        self.notes: list[str] = []

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

    def run(self, jaxpr: ir.Jaxpr, consts, args) -> list[iv.IntervalArray]:
        for var, c in zip(jaxpr.constvars, consts):
            if isinstance(c, iv.IntervalArray):
                self.env[var.id] = c
                continue
            try:
                self.env[var.id] = _value_to_interval(c, var.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError) as e:
                # same posture as literals: an unrepresentable const binds ⊤
                # (audit-gate finding 1 — a NaN closure const killed the run)
                self.notes.append(f"const outside the domain ({e}); ⊤")
                self.env[var.id] = iv.top(var.aval.shape)
        for var, a in zip(jaxpr.invars, args):
            self.env[var.id] = a
        for eqn in jaxpr.eqns:
            self.eqn(eqn)
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
                self.env = {}
                outs = self.run(inner.jaxpr, inner.consts, ins)
                self.env = outer_env
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
            results = []
            for i in sorted(possible):
                b = branches[i]
                self.env = {}
                results.append(self.run(b.jaxpr, list(b.consts), operands))
            self.env = outer_env
            for j, out in enumerate(eqn.outvars):
                self.env[out.id] = iv.join([r[j] for r in results])
            return

        if eqn.primitive == "stelling_assume":
            # inert by design (amendment 2): sound, counted, addressed —
            # never silent, never "known".
            ins = [self.read(a) for a in eqn.invars]
            self.counter.record_inert(eqn.primitive)
            where = eqn.source_info[-1] if eqn.source_info else "unknown location"
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
            status, detail = _bool_status(ins[0])
            self.obligations.append(
                ObligationReport(
                    index=len(self.obligations),
                    status=status,
                    detail=detail,
                    source_info=eqn.source_info,
                )
            )
        if eqn.primitive == "stelling_nonvacuity":
            status, detail = _bool_status(ins[0])
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


def propagate(closed: ir.ClosedJaxpr) -> Propagation:
    """Forward-propagate the declared boxes through a transcribed query and
    judge every ``stelling_assert`` obligation."""
    p = _Propagator()
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
