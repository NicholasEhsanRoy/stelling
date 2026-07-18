# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The ⊤-widening test (design/obligation-vacuity.md).

Re-traces each counting harness, rewrites every top-level ``stelling_any``
equation's bounds to (−inf, +inf) in the IR, re-propagates, and reports
which obligations still discharge — those proofs never used the declared
bounds. Judged on obligations only; nonvacuity rows are excluded by
registration. Fails loudly if any ``stelling_any`` hides below top level
(it would silently escape the widening).
"""

import math

import jax

jax.config.update("jax_enable_x64", True)

from stelling import ir
from stelling.coverage import sub_jaxprs
from stelling.propagate import propagate

# importing the harness modules re-runs their cases (deterministic, green);
# we only need their builder functions.
import e2a_hit386
import e2a_417
import cf_run

INF = math.inf


def widen(closed: ir.ClosedJaxpr) -> ir.ClosedJaxpr:
    j = closed.jaxpr
    for eqn in j.eqns:  # loud guard: no nested declarations may escape
        for sub in sub_jaxprs(eqn):
            for e in sub.eqns:
                assert e.primitive != "stelling_any", (
                    "nested stelling_any would escape widening — restructure"
                )
    eqns = []
    for eqn in j.eqns:
        if eqn.primitive == "stelling_any":
            params = dict(eqn.params)
            params["lo"], params["hi"] = -INF, INF
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


from stelling._jax_compat import trace  # noqa: E402

CASES = [
    ("hit386 (control)", e2a_hit386.box_harness(e2a_hit386.X1_HI)),
    ("dfx#417 (counted 1)", e2a_417.box_harness(e2a_417.HI)),
    ("dfx#207 (counted 1)", cf_run.h_207),
    ("npy#249 (counted 1)", cf_run.h_249),
]

print("== ⊤-widening: does each obligation still discharge with all bounds gone?")
for name, harness in CASES:
    p = propagate(widen(trace(harness)))
    stats = [o.status for o in p.obligations]
    tautological = [i for i, s in enumerate(stats) if s == "discharged"]
    verdict = (
        f"TAUTOLOGICAL obligations: {tautological}" if tautological
        else "no obligation survives ⊤ — the declared bounds are load-bearing"
    )
    print(f"  {name}: {stats} -> {verdict}")
