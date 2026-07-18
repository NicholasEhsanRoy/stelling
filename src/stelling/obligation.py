# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Obligation slices for solver escalation: extraction, routing, replay.

For each obligation interval propagation left ``unknown``, this module
extracts the *expression slice* — the top-level ir equations from the
``stelling_any`` declarations and constants to the ``stelling_assert``
operand — validates it against the v1 emission set, and classifies its
fragment (``QF_LRA`` for purely linear content, ``QF_NRA`` for polynomial:
a product of two non-constant subterms, an integer power >= 2, or division
by a non-constant). It also evaluates a slice at a concrete rational point
(:func:`evaluate_predicate`) in exact :class:`fractions.Fraction`
arithmetic — the solver-free replay that witness-backed refutations
require.

Nothing here guesses. The v1 emission set is scalar-only: ``add``,
``sub``, ``mul``, ``neg``, guarded ``div``, ``integer_pow``, ``max``,
``min``, the comparisons, boolean ``and``/``or``/``not``/``xor``,
boolean-selector ``select_n``, value-preserving
``convert_element_type``, and size-1 shape plumbing. Everything else —
array-shaped inputs, transcendentals, unknown primitives, possibly-zero
divisors, non-float input declarations, obligations that cannot be
mapped one-to-one onto top-level asserts — **declines**, with the
primitive and form quoted, and the obligation stays UNKNOWN. Declines
never raise; :exc:`ReplayError` is raised only by the replay evaluator,
whose caller treats it as an emission-infidelity signal.

Zero-dep: this module imports only the standard library and stelling's
own jax-free modules.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from stelling import interval as iv
from stelling import ir
from stelling.propagate import _EXACT_CONVERSIONS, Propagation

__all__ = [
    "DeclinedObligation",
    "ObligationSlice",
    "ReplayError",
    "SliceInput",
    "evaluate_predicate",
    "slice_obligation",
    "slice_unknown_obligations",
]

QF_LRA = "QF_LRA"
QF_NRA = "QF_NRA"

# The reason text the division guard quotes, verbatim per the design.
DIV_GUARD_REASON = (
    "divisor may be zero over the declared box — SMT-LIB2 division is "
    "underspecified at 0"
)

# integer_pow exponents are expanded to explicit products; beyond this the
# script stops being auditable by eye and v1 declines instead.
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
_IDENTITY_SHAPE = frozenset({"broadcast_in_dim", "reshape", "squeeze"})
# Harness primitives whose value semantics are the identity on their input.
# stelling_assume's *constraint* is inert (dropped, disclosed by the
# propagation notes) and is deliberately NOT emitted — only its data flow
# passes through, exactly as in propagation.
_IDENTITY_HARNESS = frozenset({"stelling_assume", "stelling_nonvacuity"})

_SUPPORTED = (
    _ARITH | _COMPARE | _BOOL_OPS | _IDENTITY_SHAPE | _IDENTITY_HARNESS
    | {"select_n", "convert_element_type"}
)


class ReplayError(RuntimeError):
    """The replay evaluator met something it cannot evaluate exactly.

    Raised only from :func:`evaluate_predicate`. The dispatch layer treats
    it as an emission-infidelity signal, never as a verdict.
    """


@dataclass(frozen=True)
class SliceInput:
    """One ``stelling_any`` declaration reachable from the obligation."""

    name: str  # deterministic: "x{k}", k = declaration order in the query
    var_id: int
    lo: float
    hi: float


@dataclass(frozen=True)
class ObligationSlice:
    """A validated, emittable expression slice for one unknown obligation."""

    index: int  # the obligation's index in the propagation report
    fragment: str  # QF_LRA | QF_NRA
    inputs: tuple[SliceInput, ...]  # declaration order
    consts: tuple[tuple[int, bool | int | float], ...]  # (var id, exact value)
    eqns: tuple[ir.JaxprEqn, ...]  # original (topological) order, sans stelling_any
    root: ir.Atom  # the assert operand (boolean)
    source_info: tuple[str, ...]


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


def _size(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _decode_scalar(val):
    """A literal/const value -> exact python bool | int | float, or decline."""
    if isinstance(val, ir.Array):
        if _size(val.shape) != 1:
            raise _Decline(
                f"array-shaped constant of shape {val.shape} (v1 emission is "
                f"scalar-only)"
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


def _atom_excludes_zero(atom: ir.Atom, env: Mapping[int, iv.IntervalArray]) -> bool:
    """Whether the atom's propagated interval definitely excludes 0."""
    if isinstance(atom, ir.Literal):
        try:
            v = _decode_scalar(atom.val)
        except _Decline:
            return False
        if isinstance(v, bool):
            return False
        return bool(v == v and v != 0)  # NaN-safe; exact
    got = env.get(atom.id)
    if got is None or got.size != 1:
        return False
    return got.los[0] > 0.0 or got.his[0] < 0.0


class _Slicer:
    def __init__(
        self,
        closed: ir.ClosedJaxpr,
        env: Mapping[int, iv.IntervalArray],
    ) -> None:
        self.env = env
        jaxpr = closed.jaxpr
        self.producers: dict[int, ir.JaxprEqn] = {}
        for eqn in jaxpr.eqns:
            for out in eqn.outvars:
                self.producers[out.id] = eqn
        self.consts: dict[int, object] = {
            var.id: val for var, val in zip(jaxpr.constvars, closed.consts)
        }
        self.any_order: dict[int, int] = {}  # any outvar id -> declaration index
        for eqn in jaxpr.eqns:
            if eqn.primitive == "stelling_any":
                self.any_order[eqn.outvars[0].id] = len(self.any_order)
        self.eqn_order = {id(eqn): i for i, eqn in enumerate(jaxpr.eqns)}

    # -- validation of one equation form -------------------------------------

    def _check_scalar(self, eqn: ir.JaxprEqn) -> None:
        for atom in (*eqn.invars, *eqn.outvars):
            if _size(atom.aval.shape) != 1:
                raise _Decline(
                    f"primitive {eqn.primitive!r} touches value of shape "
                    f"{atom.aval.shape} (v1 emission is scalar-only)"
                )

    def _validate(self, eqn: ir.JaxprEqn) -> None:
        prim = eqn.primitive
        if prim not in _SUPPORTED:
            raise _Decline(
                f"primitive {prim!r} is outside the supported emission set"
            )
        self._check_scalar(eqn)
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
            if not _atom_excludes_zero(eqn.invars[1], self.env):
                raise _Decline(f"'div': {DIV_GUARD_REASON}")
        if prim == "integer_pow":
            y = params.get("y")
            if not isinstance(y, int):
                raise _Decline(f"'integer_pow' with non-integer exponent {y!r}")
            if abs(y) > INTEGER_POW_EXPANSION_CAP:
                raise _Decline(
                    f"'integer_pow' exponent {y} exceeds the v1 expansion "
                    f"cap ({INTEGER_POW_EXPANSION_CAP})"
                )
            if y < 0 and not _atom_excludes_zero(eqn.invars[0], self.env):
                raise _Decline(
                    f"'integer_pow' with negative exponent {y}: "
                    f"{DIV_GUARD_REASON}"
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

    # -- the backward slice ---------------------------------------------------

    def slice(self, index: int, assert_eqn: ir.JaxprEqn) -> ObligationSlice:
        root = assert_eqn.invars[0]
        if _size(root.aval.shape) != 1:
            raise _Decline(
                f"assert operand has shape {root.aval.shape} (v1 emission is "
                f"scalar-only)"
            )
        if not _is_bool_dtype(root.aval):
            raise _Decline(
                f"assert operand has dtype {root.aval.dtype!r}, expected bool"
            )
        needed: list[ir.Atom] = [root]
        seen_eqns: dict[int, ir.JaxprEqn] = {}
        inputs: dict[int, None] = {}
        used_consts: dict[int, bool | int | float] = {}
        while needed:
            atom = needed.pop()
            if isinstance(atom, ir.Literal):
                v = _decode_scalar(atom.val)  # declines undecodables
                if not isinstance(v, bool):
                    _numeric_fraction(v)  # declines non-finite
                continue
            if atom.id in self.consts:
                v = _decode_scalar(self.consts[atom.id])
                if not isinstance(v, bool):
                    _numeric_fraction(v)
                used_consts[atom.id] = v
                continue
            producer = self.producers.get(atom.id)
            if producer is None:
                raise _Decline(
                    f"variable {atom.id} has no top-level producer (slices "
                    f"crossing into sub-jaxprs have no v1 emission)"
                )
            if producer.primitive == "stelling_any":
                if atom.id in inputs:
                    continue
                params = producer.params_dict()
                if tuple(params.get("shape", ())) != ():
                    raise _Decline(
                        f"input declaration of shape "
                        f"{tuple(params.get('shape', ()))} (v1 emission "
                        f"accepts scalar shape () declarations only)"
                    )
                dtype = str(params.get("dtype"))
                if dtype not in _FLOAT_INPUT_DTYPES:
                    raise _Decline(
                        f"input declaration of dtype {dtype!r}: only float "
                        f"scalar declarations are supported (an int/bool "
                        f"input's real relaxation would admit non-member "
                        f"witnesses)"
                    )
                lo, hi = float(params["lo"]), float(params["hi"])
                if math.isnan(lo) or math.isnan(hi) or lo > hi:
                    raise _Decline(
                        f"input declaration with bounds ({lo!r}, {hi!r}) has "
                        f"no emission (empty or NaN-bounded set)"
                    )
                inputs[atom.id] = None
                continue
            if id(producer) in seen_eqns:
                continue
            self._validate(producer)
            seen_eqns[id(producer)] = producer
            needed.extend(producer.invars)

        ordered = tuple(
            sorted(seen_eqns.values(), key=lambda e: self.eqn_order[id(e)])
        )
        slice_inputs = tuple(
            SliceInput(
                name=f"x{self.any_order[vid]}",
                var_id=vid,
                lo=float(self.producers[vid].params_dict()["lo"]),
                hi=float(self.producers[vid].params_dict()["hi"]),
            )
            for vid in sorted(inputs, key=lambda v: self.any_order[v])
        )
        fragment = self._fragment(slice_inputs, ordered)
        return ObligationSlice(
            index=index,
            fragment=fragment,
            inputs=slice_inputs,
            consts=tuple(sorted(used_consts.items())),
            eqns=ordered,
            root=root,
            source_info=assert_eqn.source_info,
        )

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
        return tuple(
            DeclinedObligation(
                index=o.index,
                reason=(
                    f"{len(propagation.obligations)} obligation(s) but "
                    f"{len(asserts)} top-level stelling_assert equation(s): "
                    f"asserts nested in sub-jaxprs cannot be mapped to "
                    f"slices in v1"
                ),
                source_info=o.source_info,
            )
            for o in unknown
        )
    return tuple(slice_obligation(closed, o.index, env) for o in unknown)


# -- exact-rational replay ----------------------------------------------------


def evaluate_predicate(
    sl: ObligationSlice, values: Mapping[str, Fraction]
) -> bool:
    """Evaluate the obligation predicate at a concrete rational point.

    ``values`` maps input names (``x0``, …) to exact rationals. Pure
    Python :class:`fractions.Fraction` arithmetic — no solver, no floats.
    Returns the predicate's truth value at the point (``False`` means the
    obligation is violated there). Anything inexact or impossible raises
    :exc:`ReplayError` — the caller treats that as emission infidelity.
    """
    env: dict[int, object] = {}
    for inp in sl.inputs:
        if inp.name not in values:
            raise ReplayError(f"witness has no value for input {inp.name}")
        v = values[inp.name]
        if not isinstance(v, Fraction):
            raise ReplayError(
                f"witness value for {inp.name} is {type(v).__name__}, not an "
                f"exact Fraction"
            )
        env[inp.var_id] = v

    for var_id, val in sl.consts:
        env[var_id] = val if isinstance(val, bool) else _numeric_fraction(val)

    def read(atom: ir.Atom):
        if isinstance(atom, ir.Literal):
            v = _decode_scalar(atom.val)
            return v if isinstance(v, bool) else _numeric_fraction(v)
        if atom.id in env:
            return env[atom.id]
        raise ReplayError(f"replay reads unbound variable {atom.id}")

    for eqn in sl.eqns:
        prim = eqn.primitive
        params = eqn.params_dict()
        try:
            ins = [read(a) for a in eqn.invars]
            if prim == "add":
                out = ins[0] + ins[1]
            elif prim == "sub":
                out = ins[0] - ins[1]
            elif prim == "mul":
                out = ins[0] * ins[1]
            elif prim == "neg":
                out = -ins[0]
            elif prim == "div":
                out = ins[0] / ins[1]
            elif prim == "integer_pow":
                out = ins[0] ** int(params["y"])
            elif prim == "max":
                out = max(ins[0], ins[1])
            elif prim == "min":
                out = min(ins[0], ins[1])
            elif prim == "lt":
                out = ins[0] < ins[1]
            elif prim == "le":
                out = ins[0] <= ins[1]
            elif prim == "gt":
                out = ins[0] > ins[1]
            elif prim == "ge":
                out = ins[0] >= ins[1]
            elif prim == "eq":
                out = ins[0] == ins[1]
            elif prim == "ne":
                out = ins[0] != ins[1]
            elif prim == "and":
                out = ins[0] and ins[1]
            elif prim == "or":
                out = ins[0] or ins[1]
            elif prim == "not":
                out = not ins[0]
            elif prim == "xor":
                out = bool(ins[0]) != bool(ins[1])
            elif prim == "select_n":
                out = ins[2] if ins[0] else ins[1]
            elif prim == "convert_element_type":
                v = ins[0]
                dst = str(params.get("new_dtype"))
                if isinstance(v, bool) and dst != "bool":
                    out = Fraction(1 if v else 0)
                else:
                    out = v
            elif prim in _IDENTITY_SHAPE or prim in _IDENTITY_HARNESS:
                out = ins[0]
            else:
                raise ReplayError(
                    f"no replay rule for primitive {prim!r} (slice should "
                    f"have declined)"
                )
        except ZeroDivisionError as e:
            raise ReplayError(
                f"division by zero at {prim!r} during replay: {e}"
            ) from e
        except _Decline as d:
            raise ReplayError(f"undecodable value during replay: {d.reason}") from d
        for outvar in eqn.outvars:
            env[outvar.id] = out

    result = None
    if isinstance(sl.root, ir.Literal):
        v = _decode_scalar(sl.root.val)
        result = v if isinstance(v, bool) else None
    elif sl.root.id in env:
        result = env[sl.root.id]
    if not isinstance(result, bool):
        raise ReplayError(
            f"replay produced {type(result).__name__} for the predicate, "
            f"expected bool"
        )
    return result
