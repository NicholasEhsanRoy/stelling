# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""SMT-LIB2 text emission for obligation slices. Zero-dep, deterministic.

The emitted script is the interchange artifact of every solver invocation:
transports deliver this text (wheel bindings through their own SMT-LIB2
parser, an external binary over stdin), and its sha256 rides in the stamp,
so an invocation is auditable and diffable. The query asserts the declared
box constraints (closed, exact dyadic rationals — never a decimal
approximation) **and the negation of the obligation predicate**: ``unsat``
means the predicate holds over the whole declared box.

Emission is deterministic: input constants are named ``x{k}`` in the
declaration order of the query's ``stelling_any`` equations, intermediate
terms are ``define-fun``\\ s named ``t{var id}`` in slice order, and the
per-solver option block is a fixed list — a solver is never invoked on
defaults, so ``:produce-models`` and the time limit (plus ``:nl-cov
true`` / ``:nl-ext none`` for cvc5 on QF_NRA — the two are mutually
exclusive and both are therefore pinned) are always in the text.

This module only emits; it declines nothing (the slice was validated by
:mod:`stelling.obligation`) and invokes nothing (:mod:`stelling.solvers`
owns transports).
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
    _decode_scalar,
    _numeric_fraction,
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


def rational(fr: Fraction) -> str:
    """An exact SMT-LIB2 Real literal: integers as ``N.0``, non-integers as
    ``(/ p q)``, negatives wrapped ``(- ...)``. Never a decimal
    approximation."""
    if fr < 0:
        return f"(- {rational(-fr)})"
    if fr.denominator == 1:
        return f"{fr.numerator}.0"
    return f"(/ {fr.numerator} {fr.denominator})"


def _value_text(v: bool | int | float) -> str:
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

    names: dict[int, str] = {}  # var id -> term text
    for var_id, val in sl.consts:
        names[var_id] = _value_text(val)
    for inp in sl.inputs:
        names[inp.var_id] = inp.name

    def term(atom: ir.Atom) -> str:
        if isinstance(atom, ir.Literal):
            return _value_text(_decode_scalar(atom.val))
        got = names.get(atom.id)
        if got is None:
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
        # closed, exact bounds; a half-infinite bound emits only its finite
        # side; a (-inf, inf) declaration emits no bound constraint at all.
        if inp.lo != -math.inf:
            lines.append(f"(assert (<= {rational(Fraction(inp.lo))} {inp.name}))")
        if inp.hi != math.inf:
            lines.append(f"(assert (<= {inp.name} {rational(Fraction(inp.hi))}))")

    for eqn in sl.eqns:
        prim = eqn.primitive
        params = eqn.params_dict()
        ins = [term(a) for a in eqn.invars]
        out = eqn.outvars[0]
        body: str | None = None
        alias: str | None = None
        if prim == "add":
            body = f"(+ {ins[0]} {ins[1]})"
        elif prim == "sub":
            body = f"(- {ins[0]} {ins[1]})"
        elif prim == "mul":
            body = f"(* {ins[0]} {ins[1]})"
        elif prim == "neg":
            body = f"(- {ins[0]})"
        elif prim == "div":
            body = f"(/ {ins[0]} {ins[1]})"
        elif prim == "integer_pow":
            y = int(params["y"])
            if y == 0:
                alias = "1.0"
            elif y == 1:
                alias = ins[0]
            else:
                n = abs(y)
                prod = ins[0] if n == 1 else f"(* {' '.join([ins[0]] * n)})"
                body = prod if y > 0 else f"(/ 1.0 {prod})"
        elif prim == "max":
            body = f"(ite (>= {ins[0]} {ins[1]}) {ins[0]} {ins[1]})"
        elif prim == "min":
            body = f"(ite (<= {ins[0]} {ins[1]}) {ins[0]} {ins[1]})"
        elif prim == "lt":
            body = f"(< {ins[0]} {ins[1]})"
        elif prim == "le":
            body = f"(<= {ins[0]} {ins[1]})"
        elif prim == "gt":
            body = f"(> {ins[0]} {ins[1]})"
        elif prim == "ge":
            body = f"(>= {ins[0]} {ins[1]})"
        elif prim == "eq":
            body = f"(= {ins[0]} {ins[1]})"
        elif prim == "ne":
            body = f"(distinct {ins[0]} {ins[1]})"
        elif prim == "and":
            body = f"(and {ins[0]} {ins[1]})"
        elif prim == "or":
            body = f"(or {ins[0]} {ins[1]})"
        elif prim == "not":
            body = f"(not {ins[0]})"
        elif prim == "xor":
            body = f"(xor {ins[0]} {ins[1]})"
        elif prim == "select_n":
            # select_n(which, on_false, on_true): ite takes the true case first
            body = f"(ite {ins[0]} {ins[2]} {ins[1]})"
        elif prim == "convert_element_type":
            src = eqn.invars[0].aval.dtype or ""
            dst = str(params.get("new_dtype"))
            if src == "bool" and dst != "bool":
                body = f"(ite {ins[0]} 1.0 0.0)"
            else:
                alias = ins[0]  # value-preserving: identity in emission
        else:
            # identity plumbing: shape ops, assume/nonvacuity data flow, and
            # the validated single-element forms (a one-index `slice`, a
            # one-addend `reduce_sum`) — each denotes its operand's own
            # value, so it ALIASES that operand's term. No new term, no new
            # declaration: sharing is preserved by construction, and the
            # slice validator is what guarantees the form really is
            # single-element.
            alias = ins[0]
        if alias is not None:
            names[out.id] = alias
        else:
            sort = "Bool" if out.aval.dtype == "bool" else "Real"
            names[out.id] = f"t{out.id}"
            lines.append(f"(define-fun t{out.id} () {sort} {body})")

    lines.append(f"(assert (not {term(sl.root)}))")
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
