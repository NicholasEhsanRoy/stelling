# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Three CFL harnesses against the REAL maddening HeatNode (MADD-NODE-005).

Target: maddening @ 849c39126724, ``src/maddening/nodes/heat.py`` — the 1D
explicit-FD heat node whose own metadata names the hazard (heat.py:155):

    "CFL stability limit: dt < dx^2 / (2*alpha) -- violating this produces
    silently incorrect results (MADD-ANO-002)"

A1 poses the CFL predicate at the shipped demo configuration under a ±10%
drift envelope, then re-runs the same query through the ⊤-widening vacuity
transform. A2 is the control: the same predicate over the node's full
validated ``thermal_diffusivity`` regime. B declares the node's real state
pytree and steps the REAL ``HeatNode.update`` inside the harness, asserting
the output temperature field stays inside the declared [0, 100] box.

Everything runs under ``jax_enable_x64=False`` — the node computes in
float32 and is declared and traced as it really is. Consequence for A1/A2
(stated in float64): under x64-off, ANY contact between an f64 tracer and
a Python/numpy scalar routes through ``convert_element_type[f64->f32]``
(verified on jax 0.11.0: jnp's promotion canonicalizes the lattice result;
only the all-jax-values equal-dtype fast path keeps f64), a narrowing
conversion the propagator correctly refuses to pass through — the whole
obligation would fall to ⊤. The one x64-off-legal way to keep the f64
chain analyzable is for every constant the predicate touches to enter as a
traced f64 value, so the constants (n_cells, length, the 0.5 CFL
threshold, and the nonvacuity envelope endpoints) are declared as
``stelling_any`` POINTS (lo == hi) carrying bit-exactly the given values.
The ⊤-widening transform consequently widens those points too — a strictly
stronger widening.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp

# The node is float32: declare and trace it as it really is (no x64).
jax.config.update("jax_enable_x64", False)

import maddening  # noqa: E402  (version report)
from maddening.nodes.heat import HeatNode  # noqa: E402

import stelling  # noqa: E402
from stelling import ir  # noqa: E402
from stelling.vacuity import widen as _widen  # noqa: E402
from stelling._jax_compat import jax_version, trace  # noqa: E402
from stelling.harness import any_array, any_pytree, assert_, nonvacuity  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.verdict import make_verdict  # noqa: E402

PRECISION = "jax_enable_x64=False (float32 node)"
INF = math.inf

print(
    f"pinned: maddening {getattr(maddening, '__version__', 'unversioned')} "
    f"@ 849c39126724 | jax {jax_version()} | stelling {stelling.__version__}"
)


# --- the ⊤-widening transform (the tautology_test.py pattern) ----------------


def widen(closed: ir.ClosedJaxpr) -> ir.ClosedJaxpr:
    """Rewrite every top-level ``stelling_any``'s bounds to (−inf, +inf)
    — the registered all-declarations procedure, extracted to
    :func:`stelling.vacuity.widen` (L12); behaviour byte-identical to
    the local copy this replaced."""
    return _widen(closed, mode="all")


def pose(name, harness):
    """Trace, propagate, and render the stamped verdict; if blocked, report
    the exact error and the stage it happened at."""
    print(f"==== {name}")
    try:
        cj = trace(harness)
    except Exception as e:
        print(f"  BLOCKED at trace/transcribe: {type(e).__name__}: {e}")
        print()
        return None, None
    try:
        p = propagate(cj)
    except Exception as e:
        print(f"  BLOCKED at propagation: {type(e).__name__}: {e}")
        print()
        return cj, None
    v = make_verdict(
        cj,
        p,
        stelling_version=stelling.__version__,
        jax_version=jax_version(),
        precision_config=PRECISION,
    )
    print(v.render())
    print()
    return cj, p


# --- the A-side constants, from the target's own files -----------------------
#
# The shipped demo configuration, heat_diffusion_demo.py:38-39:
#     n_cells = 20
#     length = 1.0
# The CFL regime row the node itself declares, heat.py:163:
#     ValidatedRegime("CFL", 0.0, 0.5, notes="dt * alpha / dx^2 must be < 0.5 for stability"),
# The demo operating point (dt0, alpha0) = (0.0625, 0.01):
# thermal_diffusivity is heat_diffusion_demo.py:40
#     thermal_diffusivity = 0.01  # m^2/s
# and dt follows from heat_diffusion_demo.py:45-47
#     dx = length / n_cells
#     dt_max_stable = dx**2 / (2.0 * thermal_diffusivity)
#     dt = 0.5 * dt_max_stable  # safety factor of 0.5
# i.e. dt = 0.5 * (0.05**2 / 0.02) = 0.0625.
#
# Every constant below enters the trace as an f64 POINT declaration
# (lo == hi) — see the module docstring for the x64-off constraint that
# forces this.


def declare_grid():
    """The shipped grid, with dx computed IN-TRACE — the form at heat.py:326
    (uniform-grid branch of ``_compute_laplacian``):
            dx = L / n
    """
    n_cells = any_array((), "float64", (20.0, 20.0))  # heat_diffusion_demo.py:38
    length = any_array((), "float64", (1.0, 1.0))  # heat_diffusion_demo.py:39
    dx = length / n_cells  # heat.py:326: dx = L / n  (a real traced div)
    return dx


def h_a1():
    dx = declare_grid()
    dt = any_array((), "float64", (0.05625, 0.06875))  # 0.0625 ± 10%
    alpha = any_array((), "float64", (0.009, 0.011))  # 0.01 ± 10%
    # `half` IS the literal 0.5 — the CFL threshold from heat.py:163 —
    # entered as an f64 point (module docstring, x64-off constraint).
    half = any_array((), "float64", (0.5, 0.5))
    # obligation, verbatim: dt * alpha / (dx * dx) < 0.5
    ob = assert_(dt * alpha / (dx * dx) < half)
    # nonvacuity: the demo's own operating point lies strictly inside the
    # drift envelope; each envelope endpoint enters as an f64 point.
    dt0 = any_array((), "float64", (0.0625, 0.0625))
    alpha0 = any_array((), "float64", (0.01, 0.01))
    dt_lo = any_array((), "float64", (0.05625, 0.05625))
    dt_hi = any_array((), "float64", (0.06875, 0.06875))
    a_lo = any_array((), "float64", (0.009, 0.009))
    a_hi = any_array((), "float64", (0.011, 0.011))
    return (
        ob,
        nonvacuity(dt0 > dt_lo),  # nonvacuity(dt0 > 0.05625)
        nonvacuity(dt0 < dt_hi),  # nonvacuity(dt0 < 0.06875)
        nonvacuity(alpha0 > a_lo),  # nonvacuity(alpha0 > 0.009)
        nonvacuity(alpha0 < a_hi),  # nonvacuity(alpha0 < 0.011)
    )


def h_a2():
    dx = declare_grid()
    dt = any_array((), "float64", (0.0625, 0.0625))  # the point 0.0625
    # the node's own validated regime, heat.py:161:
    #     ValidatedRegime("thermal_diffusivity", 1e-6, 1.0, "m^2/s"),
    alpha = any_array((), "float64", (1e-6, 1.0))
    half = any_array((), "float64", (0.5, 0.5))  # heat.py:163, as in A1
    # obligation, verbatim: dt * alpha / (dx * dx) < 0.5
    return assert_(dt * alpha / (dx * dx) < half)


# --- harness B: the real imported node, the real update ----------------------

NODE = HeatNode(
    name="cfl_probe",
    timestep=0.0625,
    n_cells=20,
    length=1.0,
    thermal_diffusivity=0.01,
)  # the demo values (heat_diffusion_demo.py:38-40; dt per lines 45-47)

# The prototype is the node's REAL initial_state() structure, built once
# OUTSIDE the trace: building it inside drags the prototype's own
# construction ops (broadcast_in_dim/copy of the concrete zeros) into the
# query as dead ⊤ noise; only its shapes/dtypes belong to the declaration.
PROTO = NODE.initial_state()
_T0 = np.asarray(PROTO["temperature"])
T0_UNIFORM = bool((_T0 == _T0.flat[0]).all())
T0_POINT = float(_T0.flat[0])
print(
    f"real initial_state(): temperature shape={_T0.shape} dtype={_T0.dtype} "
    f"uniform={T0_UNIFORM} value={T0_POINT}"
)


def h_b():
    # the node's real state structure: temperature (20,) float32 in [0, 100]
    state = any_pytree(PROTO, (0.0, 100.0))
    boundary_inputs = {
        "left_temperature": any_array((), "float32", (0.0, 100.0)),
        "right_temperature": any_array((), "float32", (0.0, 100.0)),
        # the source-free case: a concrete constant, not a declaration
        "heat_source": jnp.zeros(20, dtype=jnp.float32),
    }
    new_state = NODE.update(state, boundary_inputs, 0.0625)
    new_T = new_state["temperature"]
    ob_hi = assert_(new_T <= 100.0)  # every element <= 100
    ob_lo = assert_(new_T >= 0.0)  # every element >= 0
    # nonvacuity: the node's real initial_state() temperature is T=0
    # everywhere (checked concretely above); that value, as a point, lies
    # in the declared box.
    y0 = any_array((), "float32", (T0_POINT, T0_POINT))
    return (
        ob_hi,
        ob_lo,
        nonvacuity(y0 >= 0.0),
        nonvacuity(y0 <= 100.0),
    )


# --- run ---------------------------------------------------------------------

cj_a1, p_a1 = pose(
    "HARNESS A1 — CFL at the shipped configuration, ±10% drift envelope",
    h_a1,
)
if cj_a1 is not None:
    wp = propagate(widen(cj_a1))
    stats = [o.status for o in wp.obligations]
    survived = [i for i, s in enumerate(stats) if s == "discharged"]
    print(
        f"A1 under ⊤-widening (every stelling_any -> (-inf, +inf), the "
        f"constants-as-points included): obligations {stats}"
    )
    if survived:
        msg = f"  -> TAUTOLOGICAL: obligation(s) {survived} discharge without any declared bounds"
    else:
        msg = "  -> the obligation does NOT survive ⊤ — the declared bounds are load-bearing"
    print(msg)
    print()

cj_a2, p_a2 = pose(
    "HARNESS A2 — control: the same predicate over the validated alpha "
    "regime (1e-6, 1.0)",
    h_a2,
)
if p_a2 is not None and p_a2.obligations:
    o = p_a2.obligations[0]
    print(f"A2 classifier: status={o.status!r} — {o.detail}")
    if o.status == "violated-over-set":
        print("  -> violated-over-set: definitely false over the declared box")
    elif o.status != "discharged":
        print(
            "  -> unknown: undecided over the declared box (the analysis's "
            "imprecision, not a refutation)"
        )
    print()

cj_b, p_b = pose(
    "HARNESS B — real HeatNode.update step: [0,100]-box invariance of the "
    "temperature field",
    h_b,
)

print("== harness B census: every ⊤/declined primitive (count | reason)")
if p_b is None:
    print("  (harness B did not propagate — no census)")
else:
    rows = p_b.coverage.unknown_primitives
    if not rows:
        print("  (none — every reached equation had a registered transfer)")
    for name, count in rows:
        reasons = [n for n in p_b.notes if repr(name) in n]
        reason = (
            " | ".join(reasons)
            if reasons
            else "no transfer registered for this primitive; outputs bound to ⊤"
        )
        print(f"  {name:<20} ×{count:<3} {reason}")
    print(f"  coverage: {p_b.coverage.summary()}")
    print(
        "  transfers used: "
        + ", ".join(f"{n} [{t}]" for n, t in p_b.transfers_used)
    )
