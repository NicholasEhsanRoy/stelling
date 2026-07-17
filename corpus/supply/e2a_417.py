# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""E2a, dfx#417 — a continuous-flow box invariant for |y_n| ≤ B.

Registered property (`design/tracker-probe.md`): ∀ dt in region:
`|y_n| ≤ B` over the horizon. The thread's repro sets `sigma = 0.0`
(noise off), so the system is a **deterministic** gradient flow of the
quadratic potential φ(x,y;t) = (x−u(t))² + (y−v(t))²:

    dx/dt = −2 (x − u(t)),   dy/dt = −2 (y − v(t))

with the sigmoid fixed-point coordinates provably in [0, 1] for the
thread's args (a1=0,b1=1 / a2=1,b2=0). `u`, `v` are declared as bounded
harness inputs over [0, 1] — the sigmoid's actual range, supplied as a
bound because tanh is outside the census transfer set; this over-
approximates soundly (edge-flux uses the whole [0,1]).

Box: [−δ, 1+δ]², δ = 0.5. Edge-flux on all four faces; the linear
contraction toward a fixed point in [0,1]² makes it invariant. The box
implies `|y| ≤ √2·(1+δ) = B` for the continuous flow, and contains the
incident's own y0 = (0, 0).

Relation to the registered property (criterion ii, second form —
precondition with the gap named): this **discharges the continuous-flow
version** of `|y_n| ≤ B`. The incident's mechanism is **ReversibleHeun's
discrete instability** at dt0 = 0.1 — a discrete-step behaviour outside
E2a's continuous-flow frame (the inherited fidelity demotion). So the box
is a precondition of an argument for the property, not a discharge of the
incident.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import stelling
from stelling._jax_compat import jax_version, trace
from stelling.harness import any_array, assert_, nonvacuity
from stelling.propagate import propagate
from stelling.verdict import make_verdict

DELTA = 0.5
LO, HI = -DELTA, 1.0 + DELTA  # the box edges, same in x and y


def drift(coord, fixed):
    """dcoord/dt = -2 (coord - fixed) — the gradient-flow component."""
    return -2.0 * (coord - fixed)


def box_harness(y_hi):
    def h():
        # fixed-point coords: the sigmoid range [0,1], declared as bounds
        u = any_array((), "float64", (0.0, 1.0))
        v = any_array((), "float64", (0.0, 1.0))
        x_face = any_array((), "float64", (LO, HI))  # free coordinate on a face
        y_face = any_array((), "float64", (LO, y_hi))
        # four faces, flux inward:
        o1 = assert_(drift(HI, u) < 0.0)          # x = HI: dx/dt < 0
        o2 = assert_(drift(LO, u) > 0.0)          # x = LO: dx/dt > 0
        o3 = assert_(drift(y_hi, v) < 0.0)        # y = HI: dy/dt < 0
        o4 = assert_(drift(LO, v) > 0.0)          # y = LO: dy/dt > 0
        # (x_face/y_face declared to state the faces are full segments; the
        # flux facts above are uniform in the free coordinate for this
        # linear field, so the faces reduce to their fixed edge.)
        _ = x_face + y_face
        # nonvacuity: incident y0 = (0,0) is in the box
        y00 = any_array((), "float64", (0.0, 0.0))
        n1 = nonvacuity(y00 >= LO)
        n2 = nonvacuity(y00 <= HI)
        return o1, o2, o3, o4, n1, n2

    return h


def run(name, y_hi):
    cj = trace(box_harness(y_hi))
    v = make_verdict(
        cj,
        propagate(cj),
        stelling_version=stelling.__version__,
        jax_version=jax_version(),
        precision_config="jax_enable_x64=True",
    )
    print(f"==== {name}")
    print(v.render())
    print()
    return v


v1 = run("dfx#417: [-0.5, 1.5]^2 invariant under the drift", HI)
# §10.8 mutation: cap the y-box below the fixed-point range -> must not verify
v2 = run("mutation (must NOT verify): y capped at 0.5 < fixed-point max 1.0", 0.5)

print(f"dfx#417: {v1.status}; mutation: {v2.status}")
assert v1.status == "VERIFIED"
assert v2.status != "VERIFIED"
