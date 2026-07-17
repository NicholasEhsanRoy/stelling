# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Supply probe, hit diffrax#386 (termination stratum).

Property (tracker one-liner): ∀ params in region: num_steps ≤ N.
System (hand-transcribed from the issue, x64):

    c = exp(a1) - x0 ;  d = x1 / (c + exp(b))
    dx0/dt = exp(a3)*d*c - exp(a4)*x0
    dx1/dt = exp(a3)*(exp(a0) - x1)

Strategy: (1) prove a state-space box invariant for the continuous flow —
x1 boxed, and c bounded AWAY from the log-singularity c = 0 — by the
edge-flux argument, checked in Z3 over directed-rounding brackets of the
exp constants (sound interval treatment of transcendentals, since Z3 has
no exp). (2) Show what the *termination* claim additionally needs.
"""

import numpy as np
import z3


def bracket(v: float):
    """Directed-rounding bracket of a computed double."""
    return float(np.nextafter(v, -np.inf)), float(np.nextafter(v, np.inf))


A = [6.026932645397832, 4.41195014234956, 5.884199824299863,
     3.673504195449191, 4.17957753821087]
B = -2.823760940491063
EA0 = bracket(float(np.exp(A[0])))  # ~414.52  (x1 equilibrium)
EA1 = bracket(float(np.exp(A[1])))  # ~82.42   (c-boundary location)
EA3 = bracket(float(np.exp(A[3])))  # ~39.39
EA4 = bracket(float(np.exp(A[4])))  # ~65.34
EB = bracket(float(np.exp(B)))      # ~0.0594

Y0 = (4.1154706432848185, 6.831774897154676)

# Candidate invariant after round 2 (round 1 = x1 box alone; round 2 = add
# the c lower bound, which needs the x1 upper bound — see reading):
#   x1 ∈ [x1_lo, x1_hi],  c ≥ c_min
X1_LO, X1_HI = 6.8, 415.0
C_MIN = 0.019

ea0, ea1, ea3, ea4, eb = z3.Reals("ea0 ea1 ea3 ea4 eb")
brackets = [
    ea0 >= EA0[0], ea0 <= EA0[1], ea1 >= EA1[0], ea1 <= EA1[1],
    ea3 >= EA3[0], ea3 <= EA3[1], ea4 >= EA4[0], ea4 <= EA4[1],
    eb >= EB[0], eb <= EB[1],
]


def prove(name: str, claim):
    s = z3.Solver()
    s.add(brackets)
    s.add(z3.Not(claim))
    r = s.check()
    print(f"  {name}: {'PROVED' if r == z3.unsat else 'FAILED ' + str(r)}")
    return r == z3.unsat


print("== hit386: continuous-flow box invariant, edge-flux checks (Z3, QF_NRA)")
x0, x1, c = z3.Reals("x0 x1 c")

# Edge 1: x1 = X1_HI (with e^{a0} < X1_HI): dx1/dt = ea3*(ea0 - x1) < 0
prove("x1 upper edge inward (dx1<0 at x1=415)", z3.Implies(x1 == X1_HI, ea3 * (ea0 - x1) < 0))
# Edge 2: x1 = X1_LO (< y0_1 and below equilibrium): dx1/dt > 0
prove("x1 lower edge inward (dx1>0 at x1=6.8)", z3.Implies(x1 == X1_LO, ea3 * (ea0 - x1) > 0))
# Edge 3: c = C_MIN (x0 = ea1 - C_MIN): need dx0/dt < 0, i.e. dc/dt > 0:
#   dx0 = ea3 * (x1/(c+eb)) * c - ea4*x0  with x1 <= X1_HI, x0 = ea1 - c
inward = z3.Implies(
    z3.And(c == C_MIN, x1 >= X1_LO, x1 <= X1_HI, x0 == ea1 - c),
    ea3 * (x1 / (c + eb)) * c - ea4 * x0 < 0,
)
prove("c lower edge inward (boundary repels at c=0.019)", inward)
# Also need x0 >= 0 on entry region for edge 3's RHS sign; x0 = ea1 - c > 0 given c small:
prove("x0 positive at the c-edge", z3.Implies(c == C_MIN, ea1 - c > 0))

print("""
Conclusion (state-space half): the box
    x1 in [6.8, 415.0]  AND  c = exp(a1) - x0 >= 0.019
is a hand-provable invariant of the continuous flow (2 rounds: the x1 box
alone; then the c lower bound, which requires the x1 upper bound as a
conjunct). It bounds the trajectory away from the log-singularity and is
box-shaped: forward interval propagation could plausibly find it.

Conclusion (termination half): num_steps <= N does NOT follow from any
state-space invariant. On the box, the Jacobian entry
d(dx0)/dx0 = -ea3*x1*eb/(c+eb)^2 - ea4 reaches ~ -1.2e5 near the
equilibrium boundary layer (c* ~ 0.03), i.e. stiffness ratio ~ 3e3, and a
bound on ACCEPTED steps requires (i) a local-error model err ~ C*dt^p with
C = sup of high derivatives on the box, and (ii) a model of the PID
accept/reject policy — a ranking-function/controller-contract argument,
not induction on the step map. TECHNIQUE GAP, exactly the L2 category's
'different machinery' claim.
""")
