# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""WHY the untaken-branch row declines: required vs witnessed, per probe."""
import jax
jax.config.update("jax_enable_x64", True)
import stelling.propagate as P
from stelling.harness import any_array, assert_, assume, trace


def h():
    x = any_array((), "float64", (0.0, 1.0))
    def has_assume(v):
        assume(v >= 2.0)
        return v * 2.0
    def no_assume(v):
        return v
    y = jax.lax.cond(x >= 0.5, has_assume, no_assume, x)
    return (assert_(y <= -1.0),)


closed = trace(h)
required = P._assume_equation_ids(closed.jaxpr)
print("required (static) assume equations:", len(required))
n = P._declared_element_count(closed.jaxpr)
print("declared elements:", n, "probe count:", P._certificate_probe_count(n))
for k in range(P._certificate_probe_count(n)):
    probe = P._Propagator("constrain", "real")
    probe.pin = k
    try:
        probe.run(closed.jaxpr, list(closed.consts), [])
    except Exception as e:
        print(f"  probe {k}: raised {type(e).__name__}")
        continue
    w = frozenset(key for key, ok in probe.assume_witness.items() if ok)
    print(f"  probe {k}: witness map={dict(probe.assume_witness)} -> witnessed={len(w)}"
          f" ; required<=witnessed = {required <= w}")
