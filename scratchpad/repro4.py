import jax
jax.config.update("jax_enable_x64", True)
from stelling.harness import any_array, assert_
from stelling.preconditions import check

# The neighbour that must NOT change: a genuinely satisfiable guard.
def sat():
    x = any_array((3,), "float64", (-1.0, 1.0))
    return jax.lax.cond(x[0] > 0.0,
                        lambda v: assert_(v > 5.0),   # false, branch REACHABLE
                        lambda v: assert_(v > -9.0), x)

# a definitely-true guard: the OTHER branch is the unreachable one
def dt():
    c = any_array((), "float64", (1.0, 2.0))
    x = any_array((3,), "float64", (-1.0, 1.0))
    return jax.lax.cond(c > 0.0,
                        lambda v: assert_(v > 5.0),   # false, branch FORCED
                        lambda v: assert_(v > -9.0), x)

# hidden-TRUE guard: 'no' branch is the unreachable one
def hidden_true():
    x = any_array((3,), "float64", (-1.0, 1.0))
    return jax.lax.cond(x[0] - x[0] <= 0.0,
                        lambda v: assert_(v > -9.0),
                        lambda v: assert_(v > 5.0),   # unreachable
                        x)

for n, h in [("sat guard (must stay REFUTED)", sat),
             ("definitely-true guard (must stay REFUTED)", dt),
             ("hidden-true guard (must become UNKNOWN)", hidden_true)]:
    v = check(h, vacuity_mode="all")
    print(f"{n:45s} -> {v.status}  {[o.status for o in v.obligations]}")
