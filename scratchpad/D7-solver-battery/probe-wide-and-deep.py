import time, re, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from stelling.harness import any_array, assert_
from stelling.preconditions import check
F = jnp.float64

def wide(n):
    def h():
        a = any_array((n,), F, (0.0, 1.0))
        b = any_array((n,), F, (0.0, 1.0))
        return assert_(jnp.sum(a * b) <= jnp.sum(a))
    return h

def chain_one_var(k):
    def h():
        x = any_array((), F, (-1.0, 2.0))
        p = x
        for _ in range(k - 1):
            p = p * x
        return assert_(p >= -512.0)
    return h

def chain_vars(k):
    def h():
        xs = [any_array((), F, (0.5, 2.0)) for _ in range(k)]
        p = xs[0]
        for x in xs[1:]:
            p = p * x
        return assert_(p * p >= 0.0)
    return h

def report(name, h, only=None):
    kw = {} if only is None else {"solver": only}
    t0 = time.perf_counter()
    try:
        v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=10000, **kw)
    except Exception as e:
        print(f"{name:34s} {str(only):5s} RAISED {type(e).__name__}: {str(e)[:120]}")
        return
    dt = time.perf_counter() - t0
    times = [n for n in v.notes if "answered" in n or "returned" in n]
    frag = "QF_NRA" if any("QF_NRA" in o.detail for o in v.obligations) else (
        "QF_LRA" if any("QF_LRA" in o.detail for o in v.obligations) else "-")
    print(f"{name:34s} {str(only):5s} {v.status:9s} {frag} wall={dt:6.2f}s")
    for t in times: print("        ", t[:150])
    if v.status == "UNKNOWN":
        print("         detail:", v.obligations[0].detail[:200])

for n in (16, 32):
    report(f"wide {2*n} vars {n} products", wide(n), None)
for k in (10, 12):
    report(f"chain1var {k}", chain_one_var(k), None)
for k in (10, 12):
    report(f"chainvars {k}", chain_vars(k), None)
