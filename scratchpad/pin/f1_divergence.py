# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""F1's measured divergence: force the shared answer TRUE and watch the
real build and the and-veto mutant give different affine statuses."""
import dataclasses

from stelling import affine, exactness, ir
from stelling.propagate import propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, aval=F64):
    return ir.Var(id=i, aval=aval)


def lit(v):
    return ir.Literal(val=v, aval=F64)


def any_eqn(out, lo, hi):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", "float64"), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out):
    return ir.JaxprEqn(primitive=prim, invars=tuple(ins), outvars=(out,))


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns))
    )


def affine_query():
    x, w, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("sub", [x, x], w),
            eqn("ge", [w, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


qa = affine_query()
pa = propagate(qa)
withheld = dataclasses.replace(pa, assume_dropped=True)

r0, _ = affine.refine_propagation(qa, withheld)
print("UNPATCHED  affine status:", r0.obligations[0].status)

real = exactness.certifies_set_refutation
exactness.certifies_set_refutation = lambda **k: True
try:
    r1, _ = affine.refine_propagation(qa, withheld)
    print("FORCED-TRUE affine status:", r1.obligations[0].status)
finally:
    exactness.certifies_set_refutation = real

# and the interval leg, same forcing, on a run that would otherwise withhold
xi, predi, outi = var(0), var(1, BOOL), var(2, BOOL)
ap, aout, c = var(3, BOOL), var(4, BOOL), var(5, BOOL)
qi = close(
    [
        any_eqn(xi, 0.0, 1.0),
        eqn("ge", [xi, lit(2.0)], c),
        eqn("reduce_and", [c], ap),
        eqn("stelling_assume", [ap], aout),
        eqn("le", [xi, lit(-1.0)], predi),
        eqn("stelling_assert", [predi], outi),
    ],
    [outi],
)
pi0 = propagate(qi)
print("UNPATCHED  interval status:", pi0.obligations[0].status,
      "assume_dropped=", pi0.assume_dropped,
      "region_inhabited=", pi0.region_inhabited)
exactness.certifies_set_refutation = lambda **k: True
try:
    pi1 = propagate(qi)
    print("FORCED-TRUE interval status:", pi1.obligations[0].status)
finally:
    exactness.certifies_set_refutation = real
