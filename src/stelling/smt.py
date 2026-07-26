# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""SMT-LIB2 text emission for obligation slices. Zero-dep, deterministic.

The emitted script is the interchange artifact of every solver invocation:
transports deliver this text (wheel bindings through their own SMT-LIB2
parser, an external binary over stdin), and its sha256 rides in the stamp,
so an invocation is auditable and diffable. The query asserts the declared
box constraints (closed, exact dyadic rationals — never a decimal
approximation) **and the negation of the obligation predicate**: for an
array-shaped assert operand the predicate is the universal elementwise
claim, so the negated assert is ``(not (and e_0 … e_{n-1}))`` — ``unsat``
means every element holds everywhere in the box, and a ``sat`` model is a
point where at least one element violates.

Emission is deterministic and bounded static-shape: a scalar (shape
``()``) input is the constant ``x{k}`` (declaration order — the pre-array
scheme, byte-identical); an array input declares one constant per element,
``x{k}_{i}`` with ``i`` the flat C-order index. Intermediate terms are
``define-fun``\\ s named ``t{var id}`` when the output has one element and
``t{var id}_{i}`` per element otherwise. **Sharing is preserved by
construction**: every consumer references terms by name, a broadcast
scalar operand is the SAME single term in every element's body (never one
fresh variable per element — fresh variables would invent decorrelation),
and structural ops (``broadcast_in_dim``, ``reshape``, ``squeeze``,
``transpose``, ``slice``, ``concatenate``, ``stack``) emit NO terms at
all: their output elements alias their source elements' existing terms
through the same index bookkeeping the slice validator and the replay use
(:mod:`stelling.obligation`). ``reduce_sum`` emits the exact n-ary
``(+ …)`` of its addend terms — exact over Reals; static-index
``scatter-add`` emits, per touched output element, the exact n-ary sum
of the operand element's term and its contributing update terms (one
addend per contribution — duplicates accumulate), aliasing untouched
elements to the operand's terms. The per-solver option
block is a fixed list — a solver is never invoked on defaults, so
``:produce-models`` and the time limit (plus ``:nl-cov true`` /
``:nl-ext none`` for cvc5 on QF_NRA — the two are mutually exclusive and
both are therefore pinned) are always in the text.

This module only emits; it declines nothing (the slice was validated by
:mod:`stelling.obligation`, including the single element-budget gate) and
invokes nothing (:mod:`stelling.solvers` owns transports).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

from stelling import ir
from stelling.obligation import (
    QF_NRA,
    ObligationSlice,
    _decode_elements,
    _dot_general_plan,
    _group_reduce_sum,
    _numeric_fraction,
    _pair_elementwise,
    _pair_select_n,
    _route_structural,
    _scatter_add_plan,
    _scatter_set_plan,
    _size,
    _shape_of,
    _IDENTITY_HARNESS,
    _STRUCTURAL,
)

__all__ = ["Script", "emit", "rational"]


@dataclass(frozen=True)
class Script:
    """One solver invocation's SMT-LIB2 text and its exact option set."""

    solver: str  # "z3" | "cvc5"
    logic: str
    text: str
    options: tuple[tuple[str, str], ...]  # the emitted (set-option ...) pairs
    sha256: str

    def stamp_options(self) -> tuple[tuple[str, str], ...]:
        """The option set as the stamp records it: the exact emitted
        ``set-option`` pairs, the ``set-logic``, and the script hash."""
        return self.options + (
            ("set-logic", self.logic),
            ("smt2_sha256", self.sha256),
        )


def _scatter_add_sum_body(operand_term: str, update_terms) -> str:
    """The accumulate body of one touched scatter-add output element: the
    exact n-ary sum of the operand element's term and every contributing
    update term. A named seam (third audit, F4c) so the fidelity gauge
    can express emitted-sum mutations — behavior-identical extraction,
    pinned by the byte-level emission tests."""
    return f"(+ {operand_term} {' '.join(update_terms)})"


def rational(fr: Fraction) -> str:
    """An exact SMT-LIB2 Real literal: integers as ``N.0``, non-integers as
    ``(/ p q)``, negatives wrapped ``(- ...)``. Never a decimal
    approximation."""
    if fr < 0:
        return f"(- {rational(-fr)})"
    if fr.denominator == 1:
        return f"{fr.numerator}.0"
    return f"(/ {fr.numerator} {fr.denominator})"


def _value_text(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return rational(_numeric_fraction(v))


def _options(solver: str, logic: str, timeout_ms: int) -> tuple[tuple[str, str], ...]:
    if solver == "z3":
        return ((":produce-models", "true"), (":timeout", str(timeout_ms)))
    if solver == "cvc5":
        opts = [(":produce-models", "true"), (":tlimit", str(timeout_ms))]
        if logic == QF_NRA:
            # coverings on, and nl-ext explicitly OFF: the two are mutually
            # exclusive in cvc5, and leaving nl-ext at its default would be
            # both an invocation-on-defaults and a conflict.
            opts.append((":nl-cov", "true"))
            opts.append((":nl-ext", "none"))
        return tuple(opts)
    raise ValueError(f"no option profile for solver {solver!r}")


def emit(sl: ObligationSlice, solver: str, timeout_ms: int) -> Script:
    """Emit the escalation script for one obligation slice.

    The same slice emits a per-solver flavor differing only in the option
    block (option names are solver-specific); the logical content —
    declarations, bounds, definitions, the negated predicate — is
    identical, so both portfolio members see the same query.
    """
    if timeout_ms <= 0:
        raise ValueError(f"timeout_ms must be positive, got {timeout_ms}")
    options = _options(solver, sl.fragment, timeout_ms)

    names: dict[int, tuple[str, ...]] = {}  # var id -> per-element term texts
    for var_id, val in sl.consts:
        vals = val if isinstance(val, tuple) else (val,)
        names[var_id] = tuple(_value_text(v) for v in vals)
    for inp in sl.inputs:
        # per-element input names accumulate in element order (the slice
        # orders inputs by declaration then element)
        names[inp.var_id] = names.get(inp.var_id, ()) + (inp.name,)

    def term(atom: ir.Atom) -> tuple[str, ...]:
        if isinstance(atom, ir.Literal):
            return tuple(_value_text(v) for v in _decode_elements(atom.val))
        got = names.get(atom.id)
        if got is None:
            if _size(atom.aval.shape) == 0:
                return ()  # a zero-size value has no element terms
            raise ValueError(
                f"emission reads unbound variable {atom.id} — slice "
                f"validation should have declined this"
            )
        return got

    lines: list[str] = [
        f"; stelling escalation: obligation #{sl.index} ({sl.fragment})"
    ]
    for key, value in options:
        lines.append(f"(set-option {key} {value})")
    lines.append(f"(set-logic {sl.fragment})")
    for inp in sl.inputs:
        lines.append(f"(declare-const {inp.name} Real)")
    for inp in sl.inputs:
        # closed, exact bounds per element; a half-infinite bound emits only
        # its finite side; a (-inf, inf) declaration emits no bound at all.
        if inp.lo != -math.inf:
            lines.append(f"(assert (<= {rational(Fraction(inp.lo))} {inp.name}))")
        if inp.hi != math.inf:
            lines.append(f"(assert (<= {inp.name} {rational(Fraction(inp.hi))}))")

    def define(out: ir.Var, bodies: list[str]) -> tuple[str, ...]:
        """Emit define-funs for a term-producing output and return its
        element term names: ``t{id}`` for a one-element output (the
        pre-array naming, byte-identical), ``t{id}_{i}`` per element
        otherwise."""
        sort = "Bool" if out.aval.dtype == "bool" else "Real"
        if len(bodies) == 1:
            lines.append(f"(define-fun t{out.id} () {sort} {bodies[0]})")
            return (f"t{out.id}",)
        made = []
        for i, body in enumerate(bodies):
            lines.append(f"(define-fun t{out.id}_{i} () {sort} {body})")
            made.append(f"t{out.id}_{i}")
        return tuple(made)

    for eqn in sl.eqns:
        prim = eqn.primitive
        params = eqn.params_dict()
        ins = [term(a) for a in eqn.invars]
        out = eqn.outvars[0]
        n_out = _size(_shape_of(out))
        if prim in _STRUCTURAL:
            # index bookkeeping, not new terms: each output element IS its
            # source element's term
            routes = _route_structural(eqn)
            names[out.id] = tuple(ins[op][src] for op, src in routes)
            continue
        if prim == "scatter":
            # the static-index SET form is pure data movement, so it makes NO
            # new terms: element k's term IS the update's, every other
            # element's term IS the operand's. Same shape as _STRUCTURAL
            # above, driven by the same plan the slice validator and the
            # replay drive, so the three cannot route an element differently.
            try:
                routes = _scatter_set_plan(sl.eqns, dict(sl.consts), eqn)
            except Exception as e:  # validation admitted it; this cannot
                raise ValueError(  # decline — an inconsistency is a bug
                    f"emission cannot plan 'scatter' ({e}) — slice "
                    f"validation should have declined this"
                ) from e
            names[out.id] = tuple(ins[op][src] for op, src in routes)
            continue
        if prim in _IDENTITY_HARNESS:
            # assume/nonvacuity data flow: identity; the assume CONSTRAINT
            # is deliberately never emitted (inert, as in propagation)
            names[out.id] = ins[0]
            continue
        if prim == "reduce_sum":
            groups = _group_reduce_sum(eqn)
            if all(len(g) == 1 for g in groups) and groups:
                # a sum of one addend is that addend: alias, no new term
                names[out.id] = tuple(ins[0][g[0]] for g in groups)
                continue
            bodies = []
            for group in groups:
                if not group:
                    bodies.append("0.0")  # jax's measured empty-sum identity
                elif len(group) == 1:
                    bodies.append(ins[0][group[0]])
                else:
                    bodies.append(f"(+ {' '.join(ins[0][i] for i in group)})")
            names[out.id] = define(out, bodies)
            continue
        if prim == "dot_general":
            # A CONSTANT-COEFFICIENT LINEAR COMBINATION, which is what this
            # row exists to produce: per output element, Σ c_k · sym_k with
            # every c_k a rational literal. QF_LRA, no nonlinear cost.
            #
            # The coefficient/index pairs come from the same
            # _dot_general_plan the slice validation and the replay drive,
            # so the emission cannot route a coefficient either of them
            # would not.
            #
            # Coefficients of 0 are DROPPED and coefficients of 1 emit the
            # bare term. Both are exact identities over the reals, not
            # simplifications: a structurally-zero coefficient contributes
            # nothing to the sum, and dropping it keeps the formula the size
            # of the contraction's support rather than its index space.
            try:
                sym_operand, groups = _dot_general_plan(
                    sl.eqns, dict(sl.consts), eqn
                )
            except Exception as e:  # validation admitted it; this cannot
                raise ValueError(  # decline — an inconsistency is a bug
                    f"emission cannot plan 'dot_general' ({e}) — slice "
                    f"validation should have declined this"
                ) from e
            bodies = []
            for terms in groups:
                parts = []
                for coeff, idx in terms:
                    if coeff == 0:
                        continue
                    # NOT named `term`: that is the enclosing scope's
                    # helper, and binding it here shadowed the function for
                    # every LATER equation in the emission loop -- a failure
                    # that appears one equation after the mistake.
                    sym_term = ins[sym_operand][idx]
                    if coeff == 1:
                        parts.append(sym_term)
                    else:
                        parts.append(f"(* {rational(coeff)} {sym_term})")
                if not parts:
                    # every coefficient was zero: the element IS zero, and
                    # jax's measured empty-sum identity is 0.0
                    bodies.append("0.0")
                elif len(parts) == 1:
                    bodies.append(parts[0])
                else:
                    bodies.append(f"(+ {' '.join(parts)})")
            names[out.id] = define(out, bodies)
            continue
        if prim == "scatter-add":
            # the accumulate scatter: per output element, the exact n-ary
            # sum of the operand element's term and its contributing
            # update elements' terms — DUPLICATE indices contribute one
            # addend each (the primitive's defining semantic; collapsing
            # them would be the set-form confusion). The contribution
            # groups come from the same _scatter_add_plan the slice
            # validator and the replay drive, so the emission cannot
            # route a contribution differently from either. Untouched
            # elements ALIAS the operand element's term: no arithmetic
            # happened there, so no new term exists there.
            try:
                groups = _scatter_add_plan(sl.eqns, dict(sl.consts), eqn)
            except Exception as e:  # validation admitted it; this cannot
                raise ValueError(  # decline — an inconsistency is a bug
                    f"emission cannot plan 'scatter-add' ({e}) — slice "
                    f"validation should have declined this"
                ) from e
            made = []
            for i in range(n_out):
                g = groups[i]
                if not g:
                    made.append(ins[0][i])  # alias: the operand's term
                    continue
                body = _scatter_add_sum_body(
                    ins[0][i], [ins[2][u] for u in g]
                )
                name = f"t{out.id}_{i}" if n_out > 1 else f"t{out.id}"
                lines.append(f"(define-fun {name} () Real {body})")
                made.append(name)
            names[out.id] = tuple(made)
            continue
        if prim == "select_n":
            # select_n(which, on_false, on_true): ite takes the true case
            # first; a scalar `which` is the SAME selector term in every
            # element's ite (sharing, never a fresh per-element selector)
            which, on_false, on_true = _pair_select_n(eqn)
            bodies = [
                f"(ite {ins[0][which[i]]} {ins[2][on_true[i]]} {ins[1][on_false[i]]})"
                for i in range(n_out)
            ]
            names[out.id] = define(out, bodies)
            continue
        if prim == "integer_pow":
            y = int(params["y"])
            (idx,) = _pair_elementwise(eqn)
            if y == 0:
                names[out.id] = ("1.0",) * n_out
                continue
            if y == 1:
                names[out.id] = tuple(ins[0][i] for i in idx)
                continue
            n = abs(y)
            bodies = []
            for i in idx:
                base = ins[0][i]
                prod = base if n == 1 else f"(* {' '.join([base] * n)})"
                bodies.append(prod if y > 0 else f"(/ 1.0 {prod})")
            names[out.id] = define(out, bodies)
            continue
        if prim == "convert_element_type":
            src = eqn.invars[0].aval.dtype or ""
            dst = str(params.get("new_dtype"))
            (idx,) = _pair_elementwise(eqn)
            if src == "bool" and dst != "bool":
                bodies = [f"(ite {ins[0][i]} 1.0 0.0)" for i in idx]
                names[out.id] = define(out, bodies)
            else:
                names[out.id] = tuple(ins[0][i] for i in idx)  # identity
            continue
        if prim in ("neg", "not"):
            (idx,) = _pair_elementwise(eqn)
            op = "-" if prim == "neg" else "not"
            names[out.id] = define(
                out, [f"({op} {ins[0][i]})" for i in idx]
            )
            continue
        sym = {
            "add": "+", "sub": "-", "mul": "*", "div": "/",
            "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
            "eq": "=", "ne": "distinct",
            "and": "and", "or": "or", "xor": "xor",
        }
        if prim not in sym and prim not in ("max", "min"):
            raise ValueError(
                f"emission has no rule for primitive {prim!r} — slice "
                f"validation should have declined this"
            )
        ia, ib = _pair_elementwise(eqn)
        bodies = []
        for i in range(n_out):
            a, b = ins[0][ia[i]], ins[1][ib[i]]
            if prim == "max":
                bodies.append(f"(ite (>= {a} {b}) {a} {b})")
            elif prim == "min":
                bodies.append(f"(ite (<= {a} {b}) {a} {b})")
            else:
                bodies.append(f"({sym[prim]} {a} {b})")
        names[out.id] = define(out, bodies)

    root_terms = term(sl.root)
    if len(root_terms) == 1:
        negated = f"(not {root_terms[0]})"
    else:
        # the array assert is the universal elementwise claim: its negation
        # is satisfied where AT LEAST ONE element predicate fails
        negated = f"(not (and {' '.join(root_terms)}))"
    lines.append(f"(assert {negated})")
    lines.append("(check-sat)")
    lines.append("(get-model)")
    text = "\n".join(lines) + "\n"
    return Script(
        solver=solver,
        logic=sl.fragment,
        text=text,
        options=options,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
