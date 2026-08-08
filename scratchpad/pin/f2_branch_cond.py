# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""F2: is a BRANCH-SCOPED assume really never certified?"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402


def h_branch_assume():
    """The ONLY assume in the query sits inside a lax.cond branch. The
    probe that pins x to the HIGH corner (1.0) forces the True branch, so
    the assume is evaluated there and witnessed."""
    x = any_array((), "float64", (0.0, 1.0))

    def taken(v):
        assume(v >= 0.25)
        return v * 2.0

    def untaken(v):
        return v

    y = jax.lax.cond(x >= 0.5, taken, untaken, x)
    return (assert_(y <= -1.0),)


p = propagate(trace(h_branch_assume))
print("region_inhabited:", p.region_inhabited)
print("status:", p.obligations[0].status)
print("assume_dropped:", p.assume_dropped, "narrowing_uncertified:", p.narrowing_uncertified)
print("any_constrained-ish coverage.constrained:", p.coverage.constrained)
for n in p.notes:
    if "CERTIFIED NON-EMPTY" in n or "WITHHELD" in n or "UNCERTIFIED" in n:
        print("NOTE:", n[:200])
