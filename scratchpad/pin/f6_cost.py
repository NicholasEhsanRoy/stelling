# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""F6: what the UNCAPPED older search costs at large declared sizes."""
import os, time
import jax
jax.config.update("jax_enable_x64", True)
import stelling.propagate as P
from stelling.harness import any_array, assert_, trace


def branchy(n):
    """A violation INSIDE a cond branch: what makes the reachability
    search run at all."""
    def h():
        x = any_array((n,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            x[0] >= 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )
    return h


print(f"load: {os.getloadavg()}")
print(f"{'n':>7} {'reach_probes':>13} {'cert_probes':>12} {'propagate ms':>13} {'bare walk ms':>13}")
for n in (16, 64, 256, 1024, 4096, 16384):
    closed = trace(branchy(n))
    P._instr_reset()
    t0 = time.perf_counter()
    P.propagate(closed)
    t1 = time.perf_counter()
    rp, cp = P._INSTR["reach_probes"], P._INSTR["cert_probes"]
    # the bare walk, for the ratio: one _Propagator.run, no searches
    t2 = time.perf_counter()
    pr = P._Propagator("constrain", "real")
    pr.run(closed.jaxpr, list(closed.consts), [])
    t3 = time.perf_counter()
    print(f"{n:>7} {rp:>13} {cp:>12} {(t1-t0)*1e3:>13.1f} {(t3-t2)*1e3:>13.1f}")
print(f"load: {os.getloadavg()}")
