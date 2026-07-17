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
primitives. Everything else falls to ⊤ — soundly, with coverage recording
exactly how much fell.

Every transfer declares an assumption tier (design commitment 5):
``exact`` (no arithmetic, or arithmetic with no rounding), ``sound``
(outward-rounded interval arithmetic), or ``sound-libm`` (outward-rounded
around a faithfully-rounded libm call — carries
:data:`stelling.interval.EXP_LIBM_ASSUMPTION`). The tiers of every
transfer *used* ride into the verdict stamp.

Obligations are ``stelling_assert`` equations. Their statuses:
``discharged`` (predicate definitely true over the declared box),
``unknown`` (interval too wide to decide — *our* imprecision), or
``violated-over-set`` (definitely false somewhere in the box — still
reported under an UNKNOWN verdict, never as a refutation: the checker's
job is VERIFIED-or-not, and a witness discipline is the wedge's, not
ours).
"""

from __future__ import annotations

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
    coverage: Coverage
    transfers_used: tuple[tuple[str, str], ...]  # (primitive, tier), sorted
    assumptions: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def all_discharged(self) -> bool:
        return bool(self.obligations) and all(
            o.status == "discharged" for o in self.obligations
        )


# -- literals and consts ------------------------------------------------------

_STRUCT_FMT = {"<f8": "d", "<f4": "f", "<i8": "q", "<i4": "i", "|b1": "?"}


def _decode_array(a: ir.Array) -> list[float]:
    fmt = _STRUCT_FMT.get(a.dtype)
    if fmt is None:
        raise ir.TranscriptionError(
            f"no zero-dep decoder for array dtype {a.dtype!r}; add one before "
            f"propagating this query"
        )
    n = 1
    for d in a.shape:
        n *= d
    return [float(v) for v in struct.unpack(f"<{n}{fmt}", a.data)]


def _value_to_interval(v, shape: tuple[int, ...]) -> iv.IntervalArray:
    if isinstance(v, ir.Array):
        return iv.from_values(v.shape, _decode_array(v))
    if isinstance(v, bool):
        return iv.point(1.0 if v else 0.0, shape)
    if isinstance(v, int):
        if abs(v) > _EXACT_INT:
            x = float(v)
            return iv.IntervalArray(
                shape=shape, los=(iv._down(x),), his=(iv._up(x),)
            )
        return iv.point(float(v), shape)
    if isinstance(v, float):
        return iv.point(v, shape)
    raise ir.TranscriptionError(
        f"no interval meaning for literal/const of type {type(v).__name__}"
    )


# -- the transfer registry ----------------------------------------------------
#
# Exactly the target census list + the harness primitives. Signature:
# transfer(eqn, params: dict, ins: list[IntervalArray]) -> list[IntervalArray]


def _t_convert(eqn, params, ins):
    (a,) = ins
    new = str(params.get("new_dtype"))
    if new in ("float64", "bool"):
        return [a]  # every endpoint is already a double; widening is exact
    return None  # unhandled conversion target -> caller widens to ⊤, noted


TRANSFERS = {
    "add": (lambda eqn, p, ins: [iv.add(*ins)], TIER_SOUND),
    "sub": (lambda eqn, p, ins: [iv.sub(*ins)], TIER_SOUND),
    "mul": (lambda eqn, p, ins: [iv.mul(*ins)], TIER_SOUND),
    "div": (lambda eqn, p, ins: [iv.div(*ins)], TIER_SOUND),
    "neg": (lambda eqn, p, ins: [iv.neg(ins[0])], TIER_EXACT),
    "exp": (lambda eqn, p, ins: [iv.exp(ins[0])], TIER_SOUND_LIBM),
    "lt": (lambda eqn, p, ins: [iv.lt(*ins)], TIER_EXACT),
    "gt": (lambda eqn, p, ins: [iv.gt(*ins)], TIER_EXACT),
    "le": (lambda eqn, p, ins: [iv.le(*ins)], TIER_EXACT),
    "ge": (lambda eqn, p, ins: [iv.ge(*ins)], TIER_EXACT),
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
    "stelling_assume": (lambda eqn, p, ins: [ins[0]], TIER_EXACT),
    "stelling_assert": (lambda eqn, p, ins: [ins[0]], TIER_EXACT),
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
        self.used: dict[str, str] = {}
        self.assumptions: set[str] = set()
        self.notes: list[str] = []

    def read(self, atom: ir.Atom) -> iv.IntervalArray:
        if isinstance(atom, ir.Literal):
            return _value_to_interval(atom.val, atom.aval.shape)
        got = self.env.get(atom.id)
        if got is None:  # a var never bound: soundness demands ⊤, loudly noted
            self.notes.append(f"unbound var {atom.id}; treated as ⊤")
            return iv.top(atom.aval.shape)
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
            self.env[var.id] = (
                c if isinstance(c, iv.IntervalArray) else _value_to_interval(c, var.aval.shape)
            )
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
                outs = self.run(inner.jaxpr, inner.consts, ins)
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

        entry = TRANSFERS.get(eqn.primitive)
        if entry is None:
            self.counter.record_unknown(eqn.primitive)
            self.mark_unreached(eqn)
            self.top_out(eqn)
            return

        transfer, tier = entry
        ins = [self.read(a) for a in eqn.invars]
        outs = transfer(eqn, params, ins)
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
            self.assumptions.add(iv.EXP_LIBM_ASSUMPTION)
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
        coverage=p.counter.freeze(),
        transfers_used=tuple(sorted(p.used.items())),
        assumptions=tuple(sorted(p.assumptions)),
        notes=tuple(p.notes),
    )
