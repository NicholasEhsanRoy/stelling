import time, jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
from stelling.harness import any_array, assert_, trace
from stelling.preconditions import check
from stelling.vacuity import _MODES
print("modes:", _MODES)

def h():
    x = any_array((3,), "float64", (-1.0,1.0))
    return jax.lax.cond(x[0]-x[0] > 0.0, lambda v: assert_(v>5.0), lambda v: assert_(v>-9.0), x)

cj = trace(h)
for e in cj.jaxpr.eqns:
    print(e.primitive)
def walk(j, d=0):
    for e in j.eqns:
        if e.primitive=="stelling_assert":
            print("  "*d, "ASSERT si:", e.source_info)
        for k,v in e.params:
            pass
        from stelling.coverage import sub_jaxprs
        for s in sub_jaxprs(e): walk(s.jaxpr if hasattr(s,'jaxpr') else s, d+1)
walk(cj.jaxpr)
t=time.time()
for _ in range(5): check(h, vacuity_mode="all")
print("check time", (time.time()-t)/5)
