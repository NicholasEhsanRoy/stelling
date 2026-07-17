# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Layer probe, target 1: Kvaerno5's Newton iteration on the #368 system.

The incident's own layer. Newton solves F(z) = z - dt*gamma*f(y + z) = 0:

    z_{k+1} = z_k - J^{-1} F(z_k),   J = I - dt*gamma*A,   A = df/dx

Property: isfinite(z_k) ==> isfinite(z_{k+1}) over a region.

For the #368 vector field, A's entries involve exp(f(x,p) - x0 - k2) and
the NN Jacobian df/dx — so every condition below is stated over interval
bounds |f| <= M (the supply probe's contract) AND |df/dx| <= L (a NEW,
first-order learned-component contract this layer forces).

Round log (per the registration's definition):
  r1: candidate `z in box` — FAILS: a Newton step from inside any fixed box
      can exit it (no residual control); recorded as a failed round.
  r2: candidate = Newton–Kantorovich ball: ||J0^{-1}F(z0)|| <= eta,
      Lipschitz(J) <= omega, h = eta*omega*||J0^{-1}|| <= 1/2  ==>  iterates
      stay in B(z0, 2*eta) and stay finite. The invariant is a PRODUCT
      condition (nonlinear), not a box.

This script proves, in Z3, the two arithmetic pillars that make r2 sound
for the 2-D system: (i) invertibility of J from an interval bound on A
(the determinant condition, discharged by a dt-condition); (ii) the
Kantorovich product condition's monotone dependence on dt — i.e. both
"contracts + dt-condition" facts, which are the actual shape of the result.
"""

import z3


def prove(name, claim, assumptions):
    s = z3.Solver()
    s.add(assumptions)
    s.add(z3.Not(claim))
    r = s.check()
    print(f"  {name}: {'PROVED' if r == z3.unsat else 'FAILED ' + str(r)}")
    return r == z3.unsat


print("== layer probe / Newton: the determinant condition (2x2, interval-bounded A)")
a11, a12, a21, a22, s_, a = z3.Reals("a11 a12 a21 a22 s a")  # s = dt*gamma
bounded = [z3.And(x >= -a, x <= a) for x in (a11, a12, a21, a22)]
det = (1 - s_ * a11) * (1 - s_ * a22) - (s_ * a12) * (s_ * a21)
prove(
    "J = I - s*A invertible when s*a <= 0.3  (det >= 1 - 2sa - 2(sa)^2 > 0)",
    det > 0,
    bounded + [a > 0, s_ > 0, s_ * a <= z3.RealVal(3) / 10],
)
# The same claim WITHOUT the dt-condition fails — the condition is load-bearing:
s2 = z3.Solver()
s2.add(bounded + [a > 0, s_ > 0])
s2.add(z3.Not(det > 0))
print(f"  without the dt-condition: {'counterexample exists (sat) — condition is load-bearing' if s2.check() == z3.sat else 'unexpectedly proved'}")

print("== layer probe / Newton: Kantorovich condition is a nonlinear product, monotone in dt")
eta, omega, beta, h = z3.Reals("eta omega beta h")
# h = beta*eta*omega <= 1/2 guarantees finite, ball-confined iterates
# (classical Newton–Kantorovich; stated here, arithmetic checked):
kant = [eta > 0, omega > 0, beta > 0, h == beta * eta * omega]
# (i) it is a product of three region-dependent quantities — not a box on z;
# (ii) all three scale with dt: eta ~ dt*residual, omega ~ dt*Lip(A),
#      beta ~ 1/det(J) — so a sufficiently small dt always satisfies it:
dt, c1, c2, c3 = z3.Reals("dt c1 c2 c3")
scaled = [c1 > 0, c2 > 0, c3 > 0, dt > 0,
          eta == dt * c1, omega == dt * c2, beta <= c3]
prove(
    "exists dt-condition: dt^2 * c1*c2*c3 <= 1/2  ==>  h <= 1/2",
    z3.Implies(dt * dt * c1 * c2 * c3 <= z3.RealVal(1) / 2, beta * eta * omega <= z3.RealVal(1) / 2),
    kant + scaled,
)

print("""
Outcome (per the registered bands): the determinant condition is REAL and
dischargeable — but only via (a) a SECOND learned-component contract
(|df/dx| <= L: the NN Jacobian, one tier deeper than the supply probe's
|f| <= M), and (b) a dt-condition that couples the Newton layer to the
CONTROLLER layer (the two named gaps interlock). The finiteness invariant
that works (r2) is a Kantorovich BALL with a nonlinear PRODUCT side
condition — not a box, not 0-2 rounds. Classical machinery (not a research
gap), but machinery the Stage-2 plan does not contain, resting on
contracts nobody has.
""")
