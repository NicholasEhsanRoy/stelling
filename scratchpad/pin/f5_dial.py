# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""F5: the boundary paragraph's sentence, on BOTH dials."""
import jax
jax.config.update("jax_enable_x64", True)
from stelling.harness import any_array, assert_, assume, trace
from stelling.propagate import propagate


def h():
    """THE point of the paragraph: x0 pinned to 0.1, x1 pinned to 0.2, so
    the only member of the declared set is (0.1, 0.2)."""
    a = any_array((), "float64", (0.1, 0.1))
    b = any_array((), "float64", (0.2, 0.2))
    s = a * a  # an over-approximated intermediate, so the run withholds
    assume(a + b >= 0.30000000000000004)
    return (assert_(s <= -1.0),)


closed = trace(h)
for sem in ("real", "ieee"):
    p = propagate(closed, semantics=sem)
    print(f"{sem:5s}: region_inhabited={p.region_inhabited} "
          f"narrowing_uncertified={p.narrowing_uncertified} "
          f"status={p.obligations[0].status}")
print()
print("binary64 check: 0.1 + 0.2 ==", repr(0.1 + 0.2),
      ">= 0.30000000000000004 ->", (0.1 + 0.2) >= 0.30000000000000004)
from fractions import Fraction as Fr
print("exact-real check:", Fr(0.1) + Fr(0.2) >= Fr(0.30000000000000004))
