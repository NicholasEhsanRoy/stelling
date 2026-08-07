import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from stelling.harness import any_array, assert_, trace
from stelling.preconditions import check
from stelling.propagate import propagate

def d1():
    x = any_array((3,), "float64", (-1.0, 1.0))
    return jax.lax.cond(x[0] - x[0] > 0.0,
                        lambda v: assert_(v > 5.0),
                        lambda v: assert_(v > -9.0),
                        x)

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
        return i + 1, assert_(acc > 5.0)
    return jax.lax.while_loop(cond_f, body_f, (0, x[0]))

for name, h in [("D1 cond-unreachable", d1), ("D2 scan", d2), ("D2 while", d2w)]:
    print("="*70)
    print(name)
    v = check(h, vacuity_mode="all")
    print(" status:", v.status)
    for o in v.obligations:
        print("   obl", o.index, o.status, "|", o.detail[:100])
    print(" nobligations:", len(v.obligations))
    for n in v.notes:
        print("   note:", n[:200])
