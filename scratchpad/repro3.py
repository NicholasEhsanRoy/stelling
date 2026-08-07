import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from stelling.harness import any_array, assert_, trace
from stelling.preconditions import check

def killer():
    x = any_array((3,), "float64", (1.0, 2.0))
    ok = assert_(x[0] > 0.0)          # top-level, true
    def body(c, v):
        return c, assert_(v > 5.0)    # FALSE, swallowed
    _ = jax.lax.scan(body, 0.0, x)
    return ok

v = check(killer, vacuity_mode="all")
print("KILLER status:", v.status)
print("  obligations:", [(o.index, o.status) for o in v.obligations])
print("  notes:", v.notes)
print("  coverage:", v.stamp.coverage if hasattr(v.stamp,'coverage') else None)
print()
print(v.stamp)
