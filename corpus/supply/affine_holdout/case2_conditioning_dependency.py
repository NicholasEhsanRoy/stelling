# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

# Holdout case 2 runner — the dependency-shaped 2x2 conditioning
# obligation: a, c in [1, 2], b in [-0.5, 0.5], kappa = 8, through
# stelling.contracts.conditioning_2x2 — INTERVAL PATH ONLY for the
# primary record (the straddle quote), plus a supplementary escalated run
# so the current solver closure is on record next to it.
#
# Sources this replicates / attaches to:
#   corpus/supply/la_contract_probe.py Part 2 (Z3 proves the region) and
#     Part 3 (the interval straddle + the probe's own non-closure
#     argument for plain affine — quoted in cases.md);
#   tests/test_solver_acceptance.py harness(b_lo, b_hi) — the identical
#     shape as a raw stelling harness.
#
# Run:
#   <venv-jax>/bin/python case2_conditioning_dependency.py

import time

import jax

jax.config.update("jax_enable_x64", True)

import stelling  # noqa: E402
from stelling._jax_compat import jax_version, trace  # noqa: E402
from stelling.contracts import check_contract, conditioning_2x2  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.propagate import propagate  # noqa: E402

print(f"pinned: jax {jax_version()} | stelling {stelling.__version__}")

C = conditioning_2x2("float64", (1.0, 2.0), (1.0, 2.0), (-0.5, 0.5), 8.0)

print("\n==== interval path only (no solver) — the primary record")
t0 = time.perf_counter()
r = check_contract(C, vacuity_mode="inputs-only")
print(f"  wall time: {time.perf_counter() - t0:.3f} s")
print(f"  requires_status: {r.requires_status}")
for i, o in enumerate(r.requires.obligations):
    print(f"  obligation #{i}: {o.status} — {o.detail}")
print("  requires notes, verbatim:")
for n in r.requires.notes:
    print(f"    note: {n}")

print("\n==== supplementary: the same contract, solver_timeout_ms=20000")
t0 = time.perf_counter()
rs = check_contract(C, vacuity_mode="inputs-only", solver_timeout_ms=20000)
print(f"  wall time: {time.perf_counter() - t0:.3f} s")
print(f"  requires_status: {rs.requires_status}")
for i, o in enumerate(rs.requires.obligations):
    print(f"  obligation #{i}: {o.status} — {o.detail}")
print("  requires notes, verbatim:")
for n in rs.requires.notes:
    print(f"    note: {n}")

# the same shape as a RAW harness (the test-suite posing,
# tests/test_solver_acceptance.py:59-76 minus nonvacuity), so the holdout
# has a contract-free replica whose interval statuses can be re-measured
# without the contract layer:
print("\n==== raw-harness replica (test_solver_acceptance shape), interval only")


def h():
    a = any_array((), "float64", (1.0, 2.0))
    c = any_array((), "float64", (1.0, 2.0))
    b = any_array((), "float64", (-0.5, 0.5))
    tr = a + c
    det = a * c - b * b
    return (assert_(tr * tr <= det * 10.125),)  # 10.125 = 81/8 exactly


p = propagate(trace(h))
print(f"  obligation statuses: {[o.status for o in p.obligations]}")
for o in p.obligations:
    print(f"  assert #{o.index}: {o.status} — {o.detail}")
print("  propagation notes, verbatim:")
if not p.notes:
    print("    (none)")
for n in p.notes:
    print(f"    note: {n}")
