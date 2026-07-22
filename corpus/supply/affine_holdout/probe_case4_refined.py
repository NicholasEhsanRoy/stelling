# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

# Held-out evaluation probe: what does refine="affine" do on case-4 hits?
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from stelling._jax_compat import trace
from stelling.harness import any_array, assert_
from stelling.propagate import propagate
from stelling.affine import refine_propagation

def probe(name, h):
    closed = trace(h)
    p = propagate(closed)
    p2, rep = refine_propagation(closed, p)
    print(f"== {name}")
    print("  statuses before:", [o for o in p.statuses] if hasattr(p, "statuses") else "n/a")
    for line in rep.notes if hasattr(rep, "notes") else []:
        print("  note:", line)
    print("  report:", rep.summary() if hasattr(rep, "summary") else rep)

def h_4a():
    s = any_array((), "float64", (0.01, 100.0))
    s_new = jnp.where(True, s * 0.8, s)
    return (assert_(s_new < s),)

def h_4b():
    s = any_array((), "float64", (0.1, 10.0))
    nf = any_array((), "bool", (0, 1)) if False else None
    new = jnp.where(jnp.asarray(True), s * 0.8, s)
    return (assert_(new <= s),)

def h_4c():
    t = any_array((), "float64", (1.0, 1.0))
    dt = any_array((), "float64", (1e-20, 1e-20))
    return (assert_(t + dt > t),)

for name, h in [("4a strict-root 0.8s<s", h_4a), ("4b closed-root select join", h_4b), ("4c strict-root t+dt>t", h_4c)]:
    try:
        probe(name, h)
    except Exception as e:
        print(f"== {name}\n  RAISED {type(e).__name__}: {e}")
