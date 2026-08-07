import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from stelling.harness import any_array, assert_, trace
from stelling.preconditions import check
from stelling.propagate import propagate
from stelling.coverage import sub_jaxprs

def prims(cj):
    out, stack = [], [cj.jaxpr]
    while stack:
        j = stack.pop()
        for e in j.eqns:
            out.append(e.primitive)
            stack.extend(sub_jaxprs(e))
    return out

def d2():
    x = any_array((3,), "float64", (1.0, 2.0))
    def body(c, v):
        return c, assert_(v > 5.0)
    return jax.lax.scan(body, 0.0, x)

def d2w():
    x = any_array((3,), "float64", (1.0, 2.0))
    def cond_f(s):
        i, acc = s
        return i < 3
    def body_f(s):
        i, acc = s
        return i + 1, jnp.where(assert_(acc > 5.0), acc, acc)
    return jax.lax.while_loop(cond_f, body_f, (0, x[0]))

for name, h in [("scan", d2), ("while", d2w)]:
    cj = trace(h)
    print("="*60); print(name)
    print(" top prims:", [e.primitive for e in cj.jaxpr.eqns])
    print(" all prims:", prims(cj))
    p = propagate(cj)
    print(" obligations:", len(p.obligations))
    print(" notes:", p.notes)
    print(" coverage unknown prims:", p.coverage.unknown_primitives)
