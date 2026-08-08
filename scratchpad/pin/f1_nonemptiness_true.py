# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""Does forcing certifies_nonemptiness TRUE lift a withholding?"""
from stelling import exactness, ir
from stelling.propagate import propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")
var = lambda i, a=F64: ir.Var(id=i, aval=a)
lit = lambda v: ir.Literal(val=v, aval=F64)
def any_eqn(out, lo, hi):
    return ir.JaxprEqn(primitive="stelling_any", invars=(), outvars=(out,),
        params=(("shape", ()), ("dtype", "float64"), ("lo", lo), ("hi", hi)))
def eqn(prim, ins, out):
    return ir.JaxprEqn(primitive=prim, invars=tuple(ins), outvars=(out,))
def close(eqns, outvars):
    return ir.ClosedJaxpr(jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns)))

x, w, ap, aout, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL), var(4, BOOL), var(5, BOOL)
q = close([
    any_eqn(x, 0.0, 1.0),
    eqn("mul", [x, x], w),
    eqn("le", [w, lit(0.9)], ap),
    eqn("stelling_assume", [ap], aout),
    eqn("le", [x, lit(-1.0)], pred),
    eqn("stelling_assert", [pred], out),
], [out])

realw = exactness.certifies_point_witness
realn = exactness.certifies_nonemptiness
print("plain:", propagate(q).obligations[0].status, "nu=", propagate(q).narrowing_uncertified)
exactness.certifies_point_witness = lambda **k: False
try:
    p = propagate(q)
    print("cert closed:", p.obligations[0].status, "nu=", p.narrowing_uncertified)
    exactness.certifies_nonemptiness = lambda *a, **k: True
    p2 = propagate(q)
    print("cert closed + nonemptiness FORCED TRUE:", p2.obligations[0].status,
          "nu=", p2.narrowing_uncertified)
    exactness.certifies_nonemptiness = lambda *a, **k: False
    p3 = propagate(q)
    print("cert closed + nonemptiness FORCED FALSE:", p3.obligations[0].status,
          "nu=", p3.narrowing_uncertified)
finally:
    exactness.certifies_point_witness = realw
    exactness.certifies_nonemptiness = realn
