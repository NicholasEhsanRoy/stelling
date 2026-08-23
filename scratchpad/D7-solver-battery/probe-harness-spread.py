import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from stelling.harness import any_array, assert_
from stelling.preconditions import check
F = jnp.float64

def wide_A(n):   # sum(a*b) <= sum(a),  a,b in [0,1]
    def h():
        a = any_array((n,), F, (0.0, 1.0)); b = any_array((n,), F, (0.0, 1.0))
        return assert_(jnp.sum(a * b) <= jnp.sum(a))
    return h

def wide_B(n):   # sum of squares:  sum(a*a + b*b - 2ab) >= 0
    def h():
        a = any_array((n,), F, (-1.0, 1.0)); b = any_array((n,), F, (-1.0, 1.0))
        return assert_(jnp.sum(a * a + b * b - 2.0 * a * b) >= 0.0)
    return h

def wide_C(n):   # sum(a*b) - sum(b*a) >= 0  (cancellation)
    def h():
        a = any_array((n,), F, (-1.0, 1.0)); b = any_array((n,), F, (-1.0, 1.0))
        return assert_(jnp.sum(a * b) - jnp.sum(b * a) >= 0.0)
    return h

def chainD(k):   # x^k >= 0, one var, k factors
    def h():
        x = any_array((), F, (-1.0, 2.0))
        p = x
        for _ in range(k - 1): p = p * x
        return assert_(p >= 0.0)
    return h

def chainE(k):   # (x0*...*x_{k-1})^2 >= 0
    def h():
        xs = [any_array((), F, (-1.0, 1.0)) for _ in range(k)]
        p = xs[0]
        for x in xs[1:]: p = p * x
        return assert_(p * p >= 0.0)
    return h

def report(name, h, only):
    kw = {} if only is None else {"solver": only}
    t0 = time.perf_counter()
    try:
        v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=10000, **kw)
    except Exception as e:
        print(f"{name:26s} {str(only):5s} RAISED {type(e).__name__}: {str(e)[:100]}"); return
    dt = time.perf_counter() - t0
    times = [n.split(": ",1)[1] for n in v.notes if "answered" in n]
    print(f"{name:26s} {str(only):5s} {v.status:9s} wall={dt:6.2f}s  " + " | ".join(times))

for mk, nm in ((wide_A, "wideA"), (wide_B, "wideB"), (wide_C, "wideC")):
    for n in (16,):
        for only in (None, "z3", "cvc5"):
            report(f"{nm}{n}", mk(n), only)
for mk, nm in ((chainD, "chainD"), (chainE, "chainE")):
    for k in (10, 12):
        for only in (None, "z3", "cvc5"):
            report(f"{nm}{k}", mk(k), only)
