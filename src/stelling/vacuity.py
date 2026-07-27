# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The ⊤-widening vacuity transform — one implementation, imported everywhere.

The obligation-vacuity instrument (`design/obligation-vacuity.md`): re-run
a query with its declared bounds widened to (−inf, +inf) and see whether
the obligations still discharge. An obligation that discharges with the
bounds gone was never about the bounds — a tautology — and the registered
consequence is that it cannot count.

Two registered procedures, selected by ``mode``:

* ``"all"`` — every top-level ``stelling_any`` widens. The original
  procedure (`design/obligation-vacuity.md`).
* ``"inputs-only"`` — only **non-point** declarations widen; point
  declarations (transcribed constants, nonvacuity operating points) and
  literals stay. The fixed successor procedure
  (`design/mime-fvm-job.md`): widening a threshold makes almost any
  comparison straddle and defeats the tautology detector, so thresholds
  that entered as point declarations must hold still.

This module exists because the transform was convention-copied into each
corpus harness and the procedure had already changed once — the ledger's
choke-point lesson (L12) pointing at the vacuity machinery itself. A
drifting vacuity check undermines every count that relies on it; one
implementation, imported everywhere, cannot silently diverge.

The **finite-⊤ / named-range-theorem criterion** carried by
`design/corpus-expansion.md` is deliberately NOT implemented here: it is
a registration-level counting criterion (a reading rule about range
theorems), not a mechanical IR transform, and encoding it as code would
misrepresent its status.

Zero-dep: consumes :mod:`stelling.ir` only; usable in the jax-free
environments.
"""

from __future__ import annotations

import math

from stelling import ir
from stelling.coverage import sub_jaxprs

__all__ = ["NestedDeclaration", "widen"]


class NestedDeclaration(Exception):
    """A stelling_any sits inside a transparent call, so widening cannot
    reach it.

    Raised, not asserted. The predecessor was an assert, which
    python -O strips — and stripped, the guard did not merely stop
    reporting: the widened query silently kept the nested declaration's
    ORIGINAL bounds, and the verdict then carried "envelope not
    load-bearing" about an envelope that was load-bearing. Same class as the
    seven module-level census asserts converted for the same reason; this
    one was missed by that sweep.

    Callers treat this as the vacuity instrument being INERT — no
    load-bearing claim in either direction — which is sound. A false
    qualification is not.
    """

_INF = math.inf

_MODES = ("all", "inputs-only")


def _refuse_nested_declarations(jaxpr, _depth: int = 0) -> None:
    """RECURSIVE. A declaration at ANY depth escapes the widening below,
    which rewrites top-level stelling_any bounds only.

    The predecessor walked one level — sub_jaxprs(eqn) then sub.eqns
    — so a declaration two calls down escaped both the guard and the
    widening. A guard that handles depth 2 and misses depth 3 is the same
    defect with a different constant, so this recurses to the bottom.
    """
    for eqn in jaxpr.eqns:
        for sub in sub_jaxprs(eqn):
            for e in sub.eqns:
                if e.primitive == "stelling_any":
                    raise NestedDeclaration(
                        f"a declaration sits {_depth + 1} transparent call(s) "
                        f"below top level, where widening cannot reach it"
                    )
            _refuse_nested_declarations(sub, _depth + 1)


def widen(closed: ir.ClosedJaxpr, *, mode: str) -> ir.ClosedJaxpr:
    """Rewrite top-level ``stelling_any`` bounds to (−inf, +inf).

    ``mode`` is required — the two registered procedures answer different
    questions and a silent default would let a harness run the wrong one
    without saying so. Fails loudly if any ``stelling_any`` hides below
    top level (a nested declaration would escape the widening and the
    instrument would silently under-widen).
    """
    if mode not in _MODES:
        raise ValueError(f"widen mode must be one of {_MODES}, got {mode!r}")
    j = closed.jaxpr
    _refuse_nested_declarations(j)
    eqns = []
    for eqn in j.eqns:
        if eqn.primitive == "stelling_any":
            params = dict(eqn.params)
            if mode == "all" or params["lo"] != params["hi"]:
                params["lo"], params["hi"] = -_INF, _INF
                eqn = ir.JaxprEqn(
                    primitive=eqn.primitive,
                    invars=eqn.invars,
                    outvars=eqn.outvars,
                    params=tuple(params.items()),
                    effects=eqn.effects,
                    source_info=eqn.source_info,
                )
        eqns.append(eqn)
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=j.constvars,
            invars=j.invars,
            outvars=j.outvars,
            eqns=tuple(eqns),
            effects=j.effects,
            debug_info=j.debug_info,
        ),
        consts=closed.consts,
    )
