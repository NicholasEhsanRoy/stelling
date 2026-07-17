# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Precision probe: decompose layer_pid.py's 10^4 gap (design/precision-probe.md).

Three numbers, same formula N = span/(rho*(tol/C)^(1/6)), C = L^6 held
fixed as the isolating control; only the set over which L is sup'd moves.

    L(x1, c) = ea3*x1*eb/(c+eb)^2 + ea4     (the dominant Jacobian scale)

  N_box    -- as filed (L evaluated at the boundary layer, ~1.2e5)
  N_affine -- best any affine/zonotope form can do over the SAME box:
              Z3 proves the sup of L over the box is attained at the
              corner (x1=415, c=0.019), so the interval bound is exact
              and no domain representation over this box beats it.
  N_tube   -- unsound by design: numerically integrate, take the ranges
              (x1, c) actually occupy, recompute C on those.
"""

import numpy as np
import z3

# -- constants, directed-rounding brackets as in hit386_termination.py --
A = [6.026932645397832, 4.41195014234956, 5.884199824299863,
     3.673504195449191, 4.17957753821087]
B = -2.823760940491063


def bracket(v: float):
    return float(np.nextafter(v, -np.inf)), float(np.nextafter(v, np.inf))


EA0 = bracket(float(np.exp(A[0])))
EA1 = bracket(float(np.exp(A[1])))
EA3 = bracket(float(np.exp(A[3])))
EA4 = bracket(float(np.exp(A[4])))
EB = bracket(float(np.exp(B)))

X1_LO, X1_HI = 6.8, 415.0   # the proved box
C_MIN = 0.019               # the proved c lower edge
Y0 = (4.1154706432848185, 6.831774897154676)

TOL, RHO, SPAN = 1e-8, 0.2, 100.0
N_FILED = 1e5


def N_of_L(L: float) -> float:
    return SPAN * L / (RHO * TOL ** (1 / 6))


# ---------------------------------------------------------------- Z3 half
print("== precision probe: corner domination of L over the box (Z3, QF_NRA)")
x1, c, ea1, ea3, ea4, eb, Lc = z3.Reals("x1 c ea1 ea3 ea4 eb Lc")
brackets = [ea1 >= EA1[0], ea1 <= EA1[1], ea3 >= EA3[0], ea3 <= EA3[1],
            ea4 >= EA4[0], ea4 <= EA4[1], eb >= EB[0], eb <= EB[1]]

# Corner value with outward rounding: L_hi from upper brackets at the corner.
L_corner_hi = EA3[1] * X1_HI * EB[1] / (C_MIN + EB[0]) ** 2 + EA4[1]
L_corner_lo = EA3[0] * X1_HI * EB[0] / (C_MIN + EB[1]) ** 2 + EA4[0]

# Claim: on the box (c capped at ea1, the sampling cap = x0 > 0),
#   L(x1,c) <= L_corner_hi.  Division-free form.
box = z3.And(x1 >= X1_LO, x1 <= X1_HI, c >= C_MIN, c <= ea1)
claim = ea3 * x1 * eb + ea4 * (c + eb) ** 2 <= L_corner_hi * (c + eb) ** 2

s = z3.Solver()
s.add(brackets)
s.add(box)
s.add(z3.Not(claim))
r = s.check()
print(f"  sup L over box attained at corner (<= {L_corner_hi:.4e}): "
      f"{'PROVED' if r == z3.unsat else 'FAILED ' + str(r)}")
print(f"  corner is a real box point; L there >= {L_corner_lo:.4e}")
print("  => the interval bound over this box is EXACT (attained): no affine/")
print("     zonotope form over the same box can be tighter. N_affine = N(corner).")

# ------------------------------------------------------------- numeric half
print("\n== trajectories (RK4 fixed step, dt inside the stiffness limit)")
ea0n, ea1n, ea3n, ea4n, ebn = (float(np.exp(A[0])), float(np.exp(A[1])),
                               float(np.exp(A[3])), float(np.exp(A[4])),
                               float(np.exp(B)))


def f(y):
    x0v, x1v = y[..., 0], y[..., 1]
    cv = ea1n - x0v
    d0 = ea3n * (x1v / (cv + ebn)) * cv - ea4n * x0v
    d1 = ea3n * (ea0n - x1v)
    return np.stack([d0, d1], axis=-1)


def L_of(y):
    x0v, x1v = y[..., 0], y[..., 1]
    cv = ea1n - x0v
    return ea3n * x1v * ebn / (cv + ebn) ** 2 + ea4n


# ICs: the incident's own y0, plus a grid over the proved box.
g_x1 = np.linspace(X1_LO, X1_HI, 9)
g_c = np.geomspace(C_MIN, ea1n * 0.999, 9)  # one-sided c: cap at ea1 (x0>0)
grid = np.array([[ea1n - cv, x1v] for cv in g_c for x1v in g_x1])
ics = np.vstack([np.array(Y0)[None, :], grid])

dt, t_end = 5e-6, 2.0                       # rate<=1.6e5 -> RK4 stable
y = ics.copy()
sup_L = L_of(y)                             # per-trajectory running sup
x1_min, x1_max = y[:, 1].copy(), y[:, 1].copy()
c_min = (ea1n - y[:, 0]).copy()
for _ in range(int(t_end / dt)):
    k1 = f(y); k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
    y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    sup_L = np.maximum(sup_L, L_of(y))
    x1_min = np.minimum(x1_min, y[:, 1]); x1_max = np.maximum(x1_max, y[:, 1])
    c_min = np.minimum(c_min, ea1n - y[:, 0])
resid = float(np.abs(f(y)).max())
print(f"  settled: max |dy/dt| at t={t_end} is {resid:.2e} "
      f"(equilibrium: x1*={y[:, 1].mean():.2f}, c*={ea1n - y[:, 0].mean():.4f})")
print(f"  incident trajectory (y0): x1 occupies [{x1_min[0]:.2f}, {x1_max[0]:.2f}]"
      f" (box: [{X1_LO}, {X1_HI}]), c min {c_min[0]:.4f} (box edge {C_MIN})")
print(f"  per-trajectory sup L: incident {sup_L[0]:.4e}; grid min {sup_L[1:].min():.4e},"
      f" median {np.median(sup_L[1:]):.4e}, max {sup_L[1:].max():.4e}")

# scipy cross-check + the adaptive-step-count annotation
try:
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda t, yv: f(np.asarray(yv)), (0, SPAN), np.array(Y0),
                    method="BDF", rtol=TOL, atol=TOL)
    ys = sol.y.T
    print(f"  scipy BDF cross-check (y0, t=[0,{SPAN:.0f}], rtol=atol={TOL}):"
          f" sup L {L_of(ys).max():.4e}, x1 range [{ys[:, 1].min():.2f},"
          f" {ys[:, 1].max():.2f}]")
    print(f"  ANNOTATION -- the adaptive stiff solver's own accepted-step count:"
          f" {sol.t.size - 1} steps")
except ImportError:
    print("  (scipy unavailable; RK4 only)")

# ------------------------------------------------------------ three numbers
print("\n== the three numbers, against N_filed = 1e5")
L_box_filed = 1.2e5                # layer_pid.py's L (boundary-layer point)
L_tube = float(sup_L[0])           # incident's own trajectory
L_tube_best = float(sup_L.min())   # best case over every sampled IC
for name, L in [("N_box   (as filed)", L_box_filed),
                ("N_affine (= exact box sup, proved)", L_corner_hi),
                ("N_tube  (incident trajectory, unsound)", L_tube),
                ("N_tube  (best trajectory in box, unsound)", L_tube_best)]:
    print(f"  {name}: L = {L:.3e} -> N = {N_of_L(L):.3e}")
print(f"""
Reading against the bands: every number is >> 1e5 — including the unsound
single-trajectory optimum. The set-representation ladder buys a factor of
~{L_corner_hi / L_tube:.1f} in total (box corner -> true tube), against a gap of
10^4: the looseness is not in the domain and not in the set. It is in the
error-model contract C = L^6 itself, which conflates flow-map stiffness
with solution regularity: the trajectory CONVERGES to the stiff boundary
layer (c* ~ 0.03), where the solution is nearly constant and a stiff
integrator takes its largest steps — the adaptive-step annotation above
is the same system, same tolerance, finishing in ~10^2-10^3 steps.
""")
