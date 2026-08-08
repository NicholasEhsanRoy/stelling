# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""F2's cost half: a region inhabited only via the UNTAKEN branch is not
recovered — the static requirement declines a sound refutation."""
import jax

jax.config.update("jax_enable_x64", True)

from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402


def h_inhabited_via_untaken_branch():
    """The only assume sits in the branch taken when `x >= 0.5`, and it is
    UNSATISFIABLE there (`v >= 2` over `[0.5, 1]`). Every `x < 0.5`
    therefore satisfies every assume the program EVALUATES at it — the
    assumed region is `[0, 0.5)`, inhabited — and the assert is definitely
    violated over the whole declared box."""
    x = any_array((), "float64", (0.0, 1.0))

    def has_assume(v):
        assume(v >= 2.0)
        return v * 2.0

    def no_assume(v):
        return v

    y = jax.lax.cond(x >= 0.5, has_assume, no_assume, x)
    return (assert_(y <= -1.0),)


p = propagate(trace(h_inhabited_via_untaken_branch))
print("region_inhabited:", p.region_inhabited)
print("status:", p.obligations[0].status)
print("assume_dropped:", p.assume_dropped, "narrowing_uncertified:", p.narrowing_uncertified)

# the oracle: sample the declared set, execute the program, and count the
# points that satisfy every assume the program evaluates AND violate.
import random  # noqa: E402

random.seed(20260808)
admissible_violating = 0
N = 20000
for _ in range(N):
    xv = random.uniform(0.0, 1.0)
    if xv >= 0.5:
        ok = xv >= 2.0  # the branch's assume, as executed
        yv = xv * 2.0
    else:
        ok = True  # no assume is evaluated on this path
        yv = xv
    if ok and not (yv <= -1.0):
        admissible_violating += 1
print(f"oracle: {admissible_violating}/{N} sampled points are admissible AND violating")
