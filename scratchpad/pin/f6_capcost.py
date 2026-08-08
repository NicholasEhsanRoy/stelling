# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""F6: what capping the OLDER search would cost in verdicts.

For every branch-violation row, replay `_reachability_witnesses`'s own
loop probe by probe and record the FIRST probe index that certifies each
assert key. A cap that grants `_certificate_probe_count(n)` probes loses
exactly the keys whose first certifying index is at or beyond it.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import stelling.propagate as P
from stelling.harness import any_array, assert_, trace


def rows():
    def corner_guard(n):
        def h():
            x = any_array((n,), "float64", (-1.0, 1.0))
            return jax.lax.cond(
                x[0] >= 0.0,
                lambda v: assert_(v > 5.0),
                lambda v: assert_(v > -9.0),
                x,
            )
        return h

    def sum_guard(n):
        """A guard on a SUM: no plain corner decides it in the interesting
        direction, so the witnessing probe is a later, per-element one."""
        def h():
            x = any_array((n,), "float64", (-1.0, 1.0))
            return jax.lax.cond(
                (jnp.sum(x) >= -0.25) & (jnp.sum(x) <= 0.25),
                lambda v: assert_(jnp.sum(v) > 5.0),
                lambda v: assert_(jnp.sum(v) > -9.0 * n),
                x,
            )
        return h

    def relational_guard(n):
        """A guard relating two ELEMENTS: the plain anchors put every
        element at the same value, so `x[0] > x[1]` is never witnessed by
        probes 0/1/2."""
        def h():
            x = any_array((n,), "float64", (-1.0, 1.0))
            return jax.lax.cond(
                x[0] > x[1],
                lambda v: assert_(v[0] > 5.0),
                lambda v: assert_(v[0] > -9.0),
                x,
            )
        return h

    out = []
    for n in (4, 64, 256, 1024, 4096, 8192, 16384):
        out.append((f"corner_guard_n{n}", corner_guard(n), n))
        out.append((f"sum_guard_n{n}", sum_guard(n), n))
        out.append((f"relational_guard_n{n}", relational_guard(n), n))
    return out


print(f"{'row':26s} {'n':>6} {'cap':>4} {'keys':>5} {'first_idx':>28} {'lost_at_cap':>11}")
total_keys = total_lost = 0
for name, h, n in rows():
    closed = trace(h)
    p = P._Propagator("constrain", "real")
    p.run(closed.jaxpr, list(closed.consts), [])
    if p.any_constrained or p.assume_dropped:
        print(f"{name:26s} {n:>6} -- reachability search short-circuits")
        continue
    first = {}
    for k in range(P._PROBE_COUNT):
        probe = P._Propagator("constrain", "real")
        probe.pin = k
        try:
            probe.run(closed.jaxpr, list(closed.consts), [])
        except Exception:
            continue
        for key in probe.certain_reached:
            first.setdefault(key, k)
    # the keys the branch pass actually asks about
    asked = {key for _i, key in p.branch_violations}
    idxs = sorted(first.get(key, None) for key in asked) if asked else []
    cap = P._certificate_probe_count(n)
    lost = sum(1 for key in asked if first.get(key) is None or first[key] >= cap)
    total_keys += len(asked)
    total_lost += lost
    print(f"{name:26s} {n:>6} {cap:>4} {len(asked):>5} {str(idxs):>28} {lost:>11}")
print(f"\nbranch-violation keys asked: {total_keys}; lost under a "
      f"_certificate_probe_count cap: {total_lost}")
