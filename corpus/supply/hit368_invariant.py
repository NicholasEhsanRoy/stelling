# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Supply probe, hit diffrax#368 (invariant stratum).

Property (tracker one-liner): ∀ (x, p, k) in training region: solve
completes with finite state. System (from the issue):

    dx0/dt = exp(k1) * (exp(f(x, p) - x0 - k2) - 1)      f = pretrained NN
    dx1/dt = exp(k3) * (exp(k4 - x1) - 1)

Both components are mean-reverting: x1 toward k4, x0 toward f(x,p) - k2.
Candidate invariant (box), provable ONLY under a contract on the learned
component:  |f(x, p)| <= M  for x in the box, p in the training region.

Edge-flux checks under symbolic parameter brackets — k's in [KLO, KHI],
f in [-M, M] — so the result is a *family* proof over the whole region.
"""

import z3

KLO, KHI = -3.0, 3.0     # registered training region for k1..k4 (issue: samples of k)
M = 5.0                  # the f-contract bound — an ASSUMPTION, not derivable here
DELTA = 0.1              # box margin

k1, k2, k3, k4, f = z3.Reals("k1 k2 k3 k4 f")
x0, x1 = z3.Reals("x0 x1")
e = z3.Real("e")  # stands for exp(arg) at the edge — handled via sign facts only

region = [k1 >= KLO, k1 <= KHI, k2 >= KLO, k2 <= KHI,
          k3 >= KLO, k3 <= KHI, k4 >= KLO, k4 <= KHI,
          f >= -M, f <= M]


def prove(name, claim, extra=()):
    s = z3.Solver()
    s.add(region)
    s.add(list(extra))
    s.add(z3.Not(claim))
    r = s.check()
    print(f"  {name}: {'PROVED' if r == z3.unsat else 'FAILED ' + str(r)}")
    return r == z3.unsat


print("== hit368: box invariant for the continuous flow, under the f-contract")
# The exp() structure lets every edge check reduce to a SIGN fact:
#   dx0 < 0  iff  exp(f - x0 - k2) < 1  iff  f - x0 - k2 < 0, since exp(k1) > 0.
# So the edge obligations are LINEAR after the monotone-exp rewrite — done by
# hand, checked here as the linear residue (this is the pen-and-paper step).

# x0 upper edge: x0 = M - KLO + DELTA  =>  f - x0 - k2 < 0 for all f, k2 in region
X0_HI = M - KLO + DELTA
prove(f"x0 upper edge inward (x0={X0_HI})", z3.Implies(x0 == X0_HI, f - x0 - k2 < 0))
# x0 lower edge: x0 = -M - KHI - DELTA  =>  f - x0 - k2 > 0
X0_LO = -M - KHI - DELTA
prove(f"x0 lower edge inward (x0={X0_LO})", z3.Implies(x0 == X0_LO, f - x0 - k2 > 0))
# x1 edges: dx1 sign = sign(k4 - x1)
X1_HI, X1_LO = KHI + DELTA, KLO - DELTA
prove(f"x1 upper edge inward (x1={X1_HI})", z3.Implies(x1 == X1_HI, k4 - x1 < 0))
prove(f"x1 lower edge inward (x1={X1_LO})", z3.Implies(x1 == X1_LO, k4 - x1 > 0))

print(f"""
Conclusion: the box  x0 in [{X0_LO}, {X0_HI}],  x1 in [{X1_LO}, {X1_HI}]
is invariant for the continuous flow over the ENTIRE parameter region,
PROVED after the monotone-exp rewrite (each edge reduces to a linear sign
fact). Rounds: 2 (x1 box alone; then the x0 box, which forced the
f-contract into existence). Box-shaped: inference-plausible — but only
AFTER someone supplies |f| <= M, which no interval engine gets for free on
a neural network (CROWN-tier bound propagation = the founding roadmap's
'learned nodes' item). TECHNIQUE GAP #1: a contract on a learned component.

Fidelity finding: the INCIDENT is 'the linear solver returned non-finite
output' — a NaN inside Kvaerno5's Newton iteration. Newton trial iterates
are NOT flow states: exp(f - x0 - k2) overflows f64 once a trial x0 drifts
below f - k2 - 709, which the flow invariant does not prevent. The
tracker's one-line property ('solve completes finite') conflates flow
state with solver-internal state; the proved invariant does NOT discharge
the incident. TECHNIQUE GAP #2: obligations over solver-internal iterates
(no roadmap item covers them).
""")
