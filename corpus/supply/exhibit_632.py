# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""dfx#632 as a labeled exhibit — and the false-VERIFIED that DIDN'T fire.

The work order expected this harness to return VERIFIED on the ℝ-vacuous
property `t_{n+1} > t_n` (`t + dt > t`), demonstrating a false green the
`semantics: real` stamp would then have to disclaim. **It returns UNKNOWN
instead, and the reason is worth more than the expected green** — facts
from the probe.

Two measured mechanisms, both yielding UNKNOWN:

  * **Outward-rounding conservatism (no dependency).** For a point `t` at
    a float-problematic magnitude — `t = 1.0`, `dt = 1e-20 < ulp(t)` —
    the interval `+` transfer computes `fl(1.0 + 1e-20)` as the
    outward-rounded bracket `[0.999…9, 1.000…2]`, which **straddles 1.0**.
    The domain already encodes that in IEEE the sum may be ≤ t, so the
    comparison is UNKNOWN, not a false VERIFIED. The endpoints are
    float-aware even though the *reasoning* is labeled ℝ.
  * **The dependency problem (the ∀-t box).** `t` appears in both `t+dt`
    and `t`; interval arithmetic forgets the correlation and the
    comparison straddles.

So the anticipated false-VERIFIED **does not occur in the MVP** for this
arithmetic shape. It does not follow that `semantics: real` is
decorative: the field is load-bearing exactly where ℝ *reasoning*
diverges from float in ways the endpoints cannot capture — a **branch or
clip** the ℝ model omits (dfx#632's actual bug is the endpoint *clip*
`t_next ← t1`, not the raw addition), or **reassociation** the deployed
XLA program performs and the traced jaxpr does not. Straight-line
monotone arithmetic is guarded by outward rounding; those are not.

The classification and the exclusion stand unchanged
(`design/semantics-classification.md`): the property is ℝ-vacuous, and an
ℝ-vacuous hit is excluded from the E2a denominator regardless — the
exclusion is conservative policy, and this measurement shows it is also
belt-and-suspenders on the arithmetic shape. Two rules still hold: this
never counts (outside the denominator), and the verdict — UNKNOWN — is
correct, not a defect.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import stelling
from stelling import interval as iv
from stelling._jax_compat import jax_version, trace
from stelling.harness import any_array, assert_, nonvacuity
from stelling.propagate import propagate
from stelling.verdict import make_verdict

TAU = 1e-11
DT = TAU / 1000.0


def progress_harness():
    t = any_array((), "float64", (0.0, TAU))
    dt = any_array((), "float64", (DT, DT))
    o = assert_((t + dt) > t)  # t_{n+1} > t_n
    t0 = any_array((), "float64", (0.0, 0.0))
    return o, nonvacuity(t0 >= 0.0), nonvacuity(t0 <= TAU)


cj = trace(progress_harness)
v = make_verdict(
    cj,
    propagate(cj),
    stelling_version=stelling.__version__,
    jax_version=jax_version(),
    precision_config="jax_enable_x64=True",
)

print("==== EXHIBIT: dfx#632  t_{n+1} > t_n")
print("==== out-of-semantics — EXCLUDED from the E2a denominator; never counts")
print(v.render())

# The no-dependency demonstration: outward rounding is float-conservative.
tnext = iv.add(iv.point(1.0), iv.point(1e-20))
print()
print(f"no-dependency probe: fl(1.0 + 1e-20) brackets to "
      f"[{tnext.los[0]!r}, {tnext.his[0]!r}]")
print(f"  straddles 1.0: {tnext.los[0] < 1.0 < tnext.his[0]} "
      f"-> `1.0 + 1e-20 > 1.0` is UNKNOWN, not a false VERIFIED")
print()
print("Finding (facts from the probe): the anticipated false-VERIFIED does NOT")
print("fire in the MVP. Outward rounding brackets the IEEE sum, so monotone")
print("arithmetic is float-conservative. `semantics: real` stays load-bearing")
print("for branch/clip omissions (this bug's real clip logic) and XLA")
print("reassociation — not for `t + dt > t`. Classification and exclusion stand.")

assert v.status == "UNKNOWN", "measured: outward rounding + dependency -> UNKNOWN"
