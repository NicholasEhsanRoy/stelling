# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

# Held-out evaluation probe: refine_propagation on the real HeatNode case (5a).
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", False)
from stelling._jax_compat import trace
from stelling.harness import any_array, any_pytree, assert_
from stelling.propagate import propagate
from stelling.affine import refine_propagation
from maddening.nodes.heat import HeatNode

NODE = HeatNode(name="cfl_probe", timestep=0.0625, n_cells=20, length=1.0,
                thermal_diffusivity=0.01)
PROTO = NODE.initial_state()

def h_b():
    state = any_pytree(PROTO, (0.0, 100.0))
    boundary_inputs = {
        "left_temperature": any_array((), "float32", (0.0, 100.0)),
        "right_temperature": any_array((), "float32", (0.0, 100.0)),
        "heat_source": jnp.zeros(20, dtype=jnp.float32),
    }
    new_state = NODE.update(state, boundary_inputs, 0.0625)
    new_T = new_state["temperature"]
    return (assert_(new_T <= 100.0), assert_(new_T >= 0.0))

closed = trace(h_b)
p = propagate(closed)
p2, rep = refine_propagation(closed, p)
print("report:", rep)
