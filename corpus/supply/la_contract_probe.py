# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The LA contract-feasibility probe (design/la-and-stack-probes.md §2).

Hand-work, by the supply-probe instrument (pen + Z3): F3's actual linear
solve is the LSQ gradient's normal system, ``M_P = Σ_f d_f ⊗ d_f`` with
``∇φ_P = M_P⁻¹ · rhs`` (MIME operators.py:131, code :141-144; the inv
routes through lu/custom_linear_solve — F3's tier-9 ⊤). The contract is
assume-guarantee: ``requires`` M well-conditioned over the geometric
region; ``ensures`` a norm-sensitivity bound on the solve output. No
transfer is built here; the probe sizes the layer.
"""

import math

import z3

# --- Part 1: F3's concrete M — the requires can fail, and does ---------------
# The 2x1 Cartesian mesh (make_cartesian_mesh_2d(2,1,1.0,1.0)) has ONE
# interior face; with grad_boundary_values=None the boundary contributes
# nothing, so per cell M = d (x) d for the single delta d = (0.5, 0.0),
# plus the reg=1e-30 diagonal (operators.py:124).
d = (0.5, 0.0)
reg = 1e-30
M = [[d[0] * d[0] + reg, d[0] * d[1]], [d[0] * d[1], d[1] * d[1] + reg]]
tr = M[0][0] + M[1][1]
det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
# symmetric 2x2: cond_2 = lam_max/lam_min, lam = (tr ± sqrt(tr^2-4det))/2
disc = math.sqrt(tr * tr - 4.0 * det)
lam_max = (tr + disc) / 2.0
lam_min = det / lam_max  # stable form: (tr - disc)/2 cancels to 0.0 in float here
print("== Part 1: F3's own per-cell normal matrix")
print(f"   M = [[{M[0][0]:.3e}, 0], [0, {M[1][1]:.3e}]]  (rank-1 stencil + reg)")
print(f"   cond_2(M) ≈ {lam_max / lam_min:.2e}   — the requires FAILS on this")
print("   mesh: a one-face stencil cannot determine a 2-D gradient; the")
print("   failure geometry (boundary-starved / sliver stencils) is real.")

# --- Part 2: the QF_NRA reduction and the Z3 checks --------------------------
# For symmetric positive-definite 2x2 M with entries (a, b; b, c):
#   tr = a + c,  det = a*c - b^2,  and with r = lam_max/lam_min:
#   tr^2 / det = r + 1/r + 2  (exact identity)
# so  cond_2(M) <= kappa  <=>  tr^2 <= det * (kappa + 1/kappa + 2).
# The obligation over a REGION of geometries is polynomial — QF_NRA.
KAPPA = 8.0
RHS_COEFF = KAPPA + 1.0 / KAPPA + 2.0  # 10.125

a, b, c = z3.Reals("a b c")


def check(name, region, expect_unsat):
    s = z3.Solver()
    s.add(region)
    # negation of the obligation: cond > kappa
    s.add((a + c) * (a + c) > (a * c - b * b) * RHS_COEFF)
    r = s.check()
    tag = "PROVED (unsat)" if r == z3.unsat else f"{r}"
    print(f"   {name}: {tag}", end="")
    if r == z3.sat:
        m = s.model()
        print(f"   witness: a={m[a]}, b={m[b]}, c={m[c]}", end="")
    print()
    assert (r == z3.unsat) == expect_unsat


print("\n== Part 2: the conditioning obligation is QF_NRA — Z3 decides it")
well_shaped = [a >= 1.0, a <= 2.0, c >= 1.0, c <= 2.0, b >= -0.5, b <= 0.5]
check("well-shaped region (b in ±0.5): cond <= 8 for ALL", well_shaped, True)
sliver = [a >= 1.0, a <= 2.0, c >= 1.0, c <= 2.0, b >= -1.4, b <= 1.4]
check("sliver-reaching region (b in ±1.4): violated somewhere", sliver, False)

# --- Part 3: the interval straddle — the dependency demonstration ------------
# Independent interval evaluation of tr^2/det over the SAME well-shaped
# region loses the a,c correlation (both appear in tr and det):
tr2_lo, tr2_hi = (1 + 1) ** 2, (2 + 2) ** 2            # [4, 16]
det_lo, det_hi = 1 * 1 - 0.5**2, 2 * 2 - 0.0           # [0.75, 4]
print("\n== Part 3: intervals on the same obligation")
print(f"   tr^2 ∈ [{tr2_lo}, {tr2_hi}], det ∈ [{det_lo}, {det_hi}]")
print(f"   tr^2/det ∈ [{tr2_lo / det_hi:.2f}, {tr2_hi / det_lo:.2f}]  vs  "
      f"threshold {RHS_COEFF}")
print("   STRADDLES: intervals cannot close what Z3 just proved — the")
print("   precondition is dependency-shaped (a, c shared between tr and det;")
print("   det itself is Cauchy-Schwarz slack in the underlying d-vectors),")
print("   and quadratic past plain affine. Q1: dependency-shaped. Q2: the")
print("   closure is a small QF_NRA validity — the solver's shape.")

# --- Part 4 (comment-level, Q3): the ensures ---------------------------------
# A residual-only ensures (||Mx - rhs|| <= eps) does NOT bound the gradient
# (the gradient IS x). The needed ensures is norm-sensitivity:
#   ||grad|| = ||M^-1 rhs|| <= ||M^-1|| * ||rhs|| = (kappa / ||M||) * ||rhs||
# — derived from the SAME conditioning data as the requires. One clean
# layer. Backward error (LAPACK's (M+dM)x = rhs) enters only when the
# R-semantics dial moves; under the stamped dial it is the recorded gap.
print("\n== Part 4 (Q3): ensures = norm-sensitivity, kappa-derived — see comments")
