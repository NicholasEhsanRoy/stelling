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

__all__ = ["NestedDeclaration", "declaration_bounds", "unwidened", "widen"]


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


def declaration_bounds(closed) -> list:
    """Every stelling_any's (outvar id, lo, hi), at EVERY depth.

    The population the widen predicate is about. Recursive for the same reason
    the refusal is: a declaration below top level is still a declared bound,
    and a walk that stops at the top reports it absent.
    """
    out = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive == "stelling_any":
                # THE BOUNDS ARE TAKEN RAW HERE AND COMPARED RAW BELOW, and
                # that is sound because of what stores them rather than
                # because of anything this module does — audit 0.2.0 B12,
                # S16. `ir._validate_decl_bounds` refuses a `lo`/`hi` that
                # is not a `float` and INSTALLS the exact `float` it read,
                # in every `JaxprEqn.__post_init__`, so this module's `!=`
                # and the `float(...)` every analysis applies are one read
                # of one value. Before that install they were two: a
                # document carrying `lo:"1.0"`, `hi:1.0` was a POINT by
                # `float()` and not a point by the `!=` in `widen` below,
                # so `mode="inputs-only"` widened a declaration whose whole
                # contract is that a point is held still.
                p = dict(eqn.params)
                out.append((eqn.outvars[0].id, p["lo"], p["hi"]))
            for sub in sub_jaxprs(eqn):
                walk(sub)

    walk(closed.jaxpr)
    return out


def unwidened(before, after) -> str | None:
    """THE ONE RULE. None iff widen moved EVERY declared bound.

    Both known false-qualification defects are the same fact seen twice: the
    verdict claimed the envelope was widened when part of it was not.

      * a declaration inside a transparent call is not reached by the rewrite,
        which walks top-level equations;
      * a POINT declaration (lo == hi) is deliberately held still under
        mode="inputs-only", because a declared point is a stated constant
        rather than a degenerate range -- widening it would make the mode
        answer a different question than its name.

    Neither is a bug in widen. The bug was emitting a load-bearing claim
    anyway. So the predicate is not "is this query shaped oddly" but the
    directly checkable "did every declared bound actually move", which is what
    the claim asserts.

    An EMPTY declaration set returns a reason too: "every bound moved" is
    vacuously true over nothing, and a vacuous truth is exactly what must not
    license the claim.
    """
    if not before:
        return "the query declares no bounded inputs, so widening changes nothing"
    post = {vid: (lo, hi) for vid, lo, hi in after}
    for vid, lo, hi in before:
        if vid not in post:
            return (
                f"declaration {vid} is not present after widening, so the "
                f"re-check is not the same query"
            )
        if post[vid] == (lo, hi):
            if lo == hi:
                allpoint = all(a == b for _, a, b in before)
                scope = ("every declared input is a point interval"
                         if allpoint else
                         f"declaration {vid} is a point interval ({lo})")
                # the hint is load-bearing guidance, not decoration: the
                # other mode DOES widen points, so a caller who needs the
                # measurement has somewhere to go.
                return (
                    f"{scope}, so this mode widens nothing on it; "
                    f"mode='all' would also widen transcribed constants"
                )
            return (
                f"declaration {vid} kept its bounds ({lo}, {hi}) -- the "
                f"rewrite did not reach it; the envelope was not fully widened"
            )
    return None


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
            # the RAW `!=` — see `declaration_bounds` for why it is the same
            # read every analysis makes, and for what it was before it was
            # (audit 0.2.0 B12, S16). A NaN endpoint, which `!=` would call
            # a non-point and `float()` would too, cannot reach here from a
            # document at all: `ir._validate_decl_nonempty` refuses it at
            # the load door as the empty set it declares.
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
