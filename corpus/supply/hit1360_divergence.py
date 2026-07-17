# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Supply probe, hit numpyro#1360 (invariant stratum).

Property (tracker one-liner): ∀ seeds/inits in region: zero divergent
transitions. Model (from the issue):

    w ~ Exponential(87.29), b ~ Normal(1,1), sigma ~ Chi2(31.63)
    y_i ~ Normal(w*x_i + b, sigma),  n = 100 observations

NUTS runs in unconstrained space: u_w = log w, u_s = log-ish sigma;
init_to_uniform draws u ∈ (-2, 2)^3. A divergence fires when the leapfrog
energy error exceeds 1000.

This script does the pen-and-paper analysis numerically (python as a
calculator — no sampler is run): the potential's curvature and gradient at
registered-region init corners, versus the stability threshold of leapfrog
for any plausible adapted step size.
"""

import numpy as np

x = np.array([0.0970, 2.1020, 0.5840, 1.0394, 3.4375, 1.3102, 1.2863, 4.5382,
              1.2539, 2.9319, 4.7777, 4.5937, 4.0403, 0.7749, 1.8342, 1.5008,
              3.9557, 1.6095, 0.5602, 4.7997])  # first 20 of 100 suffice for scale
y = np.array([4.9404, 45.0399, 14.6792, 23.7890, 71.7504, 29.2036, 28.7251,
              93.7642, 28.0774, 61.6389, 98.5535, 94.8738, 83.8054, 18.4976,
              39.6836, 33.0162, 82.1149, 35.1902, 14.2033, 98.9948])
n = len(x)

print("== hit1360: is 'zero divergences over the init region' even true?")
# Init corner inside the REGISTERED region u ∈ (-2,2)^3:
u_w, b, u_s = -2.0, -2.0 + 3.0, -2.0     # w = e^-2 = 0.135, sigma = e^-2 = 0.135
w, sigma = np.exp(u_w), np.exp(u_s)
resid = y - (w * x + b)
# Gradient of the negative log-likelihood wrt b at this init:
grad_b = -np.sum(resid) / sigma**2
# Curvature wrt b (constant in b): n / sigma^2
curv_b = n / sigma**2
print(f"   at init corner (w={w:.3f}, b={b:.1f}, sigma={sigma:.3f}):")
print(f"   |dU/db| = {abs(grad_b):.3e}   d2U/db2 = {curv_b:.3e}")
# Leapfrog stability: requires eps < 2/sqrt(lambda_max). For ANY eps adapted
# to the posterior bulk (sigma* ~ 4-5 => curvature ~ n/25 ~ 4, eps ~ 1):
eps_bulk = 2 / np.sqrt(n / 5.0**2)
eps_stab = 2 / np.sqrt(curv_b)
print(f"   eps adapted to bulk ~ {eps_bulk:.2f};  stability limit here ~ {eps_stab:.1e}")
print(f"   ratio ~ {eps_bulk / eps_stab:.0f}x above the stability limit -> energy error")
print(f"   grows geometrically; |dH| > 1000 within a few leapfrog steps.")
print("""
Conclusion: the property is FALSE over its own registered region — and it
must be: the incident (2-3 of 10 seeds diverge) IS its counterexample. The
tracker probe's one-line property encoded the *desired* invariant over the
*incident's* region. Recorded loudly, per the registration:

  - The supply question at this hit is not 'prove the property' but
    'characterize the envelope': for WHICH inits/step-sizes do zero
    divergences hold? That envelope is set by curvature ~ n/sigma^2 against
    the adapted eps — a NONLINEAR relation between state components, coupled
    to the ADAPTATION trajectory (eps depends on warmup history, which
    depends on the seed). Not expressible as induction on the step map.
    TECHNIQUE GAP: trajectory/expectation-tier machinery (the founding
    doc's supermartingale bullet — its farthest item).
  - Inferability: not box-shaped. Methodology, if attempted at all.
  - The red direction here is cheap and already served the user (reseed).
""")
