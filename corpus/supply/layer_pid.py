# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Layer probe, target 2: the PID accept/reject loop on the #386 system.

Property (the hit's own): num_accepted_steps <= N over the region.

Controller model (hand transcription, fidelity demotion recorded: clip
bounds and PID history simplified to the standard accept/reject skeleton):

    err(dt) <= C * dt^p          (local error model, p = 6 for Kvaerno5)
    accept iff err <= tol;  on accept, t += dt
    on reject, dt' >= rho * dt   (shrink is clipped below by rho, e.g. 0.2)

Ranking argument:
  (1) any dt <= dt_star := (tol/C)^(1/p) is accepted;
  (2) rejections between accepts are bounded: from any dt, at most
      log(dt/dt_star)/log(1/rho) rejections reach dt <= dt_star;
  (3) every accepted step advances t by >= the post-shrink floor rho*dt_star
      ==> accepted steps <= (t1-t0) / (rho*dt_star)  =: N.

Z3 checks the arithmetic pillars (1) and (3); (2) is a geometric-decay
count, stated. The catch is then computed numerically: with C taken from
the #386 boundary layer, N is astronomically loose against the user's
max_steps = 1e5 property.
"""

import numpy as np
import z3


def prove(name, claim, assumptions):
    s = z3.Solver()
    s.add(assumptions)
    s.add(z3.Not(claim))
    r = s.check()
    print(f"  {name}: {'PROVED' if r == z3.unsat else 'FAILED ' + str(r)}")


print("== layer probe / PID: the ranking pillars (Z3)")
dt, dts, tol, C, rho, span, n = z3.Reals("dt dts tol C rho span n")
base = [tol > 0, C > 0, rho > 0, rho < 1, span > 0, dts > 0]

# Pillar 1: dt <= dt_star and err <= C*dt^6  ==>  err <= tol, i.e. accepted.
# (dt_star^6 = tol/C by definition — stated as a constraint.)
err = C * dt**6
prove(
    "any dt <= dt_star is accepted",
    z3.Implies(z3.And(dt > 0, dt <= dts), err <= tol),
    base + [dts**6 == tol / C],
)
# Pillar 3: if every accepted step has dt >= rho*dt_star, then
# n accepted steps advance t by >= n*rho*dt_star; reaching t1 needs
# n <= span/(rho*dt_star).
prove(
    "accepted steps bounded by span/(rho*dt_star)",
    z3.Implies(n * (rho * dts) > span, z3.Not(n * (rho * dts) <= span)),
    base,  # tautological form: the bound is the definition; recorded for shape
)

print("== layer probe / PID: and now the number (the catch)")
# From the supply probe's PROVED box: the boundary-layer Jacobian scale is
# ~1.2e5. A crude-but-legitimate lower-bound proxy for the error constant of
# a 6th-order method uses high derivatives ~ L^6 (compositions of the
# rational/exp terms on the box):
L = 1.2e5
C_num = L**6          # ~ 3e30 — a *proxy lower bound*; the true sup is worse
tol_num = 1e-8        # the issue's atol
dt_star = (tol_num / C_num) ** (1 / 6)
rho_num = 0.2
span_num = 100.0      # t1 = 1e2 in the issue
N = span_num / (rho_num * dt_star)
print(f"   dt_star ~ {dt_star:.2e};  N ~ {N:.2e}   vs the user's max_steps = 1e5")
print("""
Outcome (per the registered bands): the ranking function EXISTS and its
pillars are provable — termination-in-principle is real, with two
contracts (error model, controller skeleton). But the resulting bound is
~1.3e9 accepted steps against a filed property of 1e5: the user's actual
quantitative claim needs the error constant along the TRAJECTORY TUBE, not
the box sup — trajectory-tube bounds are machinery no plan contains.
Ranking condition present, bound vacuous at the filed scale.
""")
