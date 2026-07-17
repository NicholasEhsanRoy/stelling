# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""E2a, case 1: check the hand-proved hit386 box mechanically.

Candidate #1 (design/e2a-registration.md): the exact hand-proved box
    x1 ∈ [6.8, 415.0],  c = exp(a1) - x0 ≥ 0.019   (c unbounded above)
stated in the box's own coordinates. Hand assistance is limited to the
registration's permitted list: the field transcription (the MWE's own
form, constants inline as filed), the box statement, and the harness
declarations. All bracketing, face-flux evaluation, and sign reasoning is
stelling's — the parameters enter as point ``any_array``s so their
``exp`` is traced and outward-rounded by the tool, exactly the demotion
the hand proof's np.nextafter brackets carried.

One trace, three face obligations, one verdict. The §10.8 mutation — the
same harness with the x1 ceiling at 300, below the x1* ≈ 414.5
equilibrium — must NOT verify.
"""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import stelling
from stelling._jax_compat import jax_version, trace
from stelling.harness import any_array, assert_
from stelling.propagate import propagate
from stelling.verdict import make_verdict

# The filed MWE's constants, inline as written there (diffrax#386).
A = [6.026932645397832, 4.41195014234956, 5.884199824299863,
     3.673504195449191, 4.17957753821087]
B = -2.823760940491063

X1_LO, X1_HI, C_MIN = 6.8, 415.0, 0.019


def field(y, a, b):
    x0, x1 = y[0], y[1]
    c = jnp.exp(a[1]) - x0
    d = x1 / (c + jnp.exp(b))
    dx0 = jnp.exp(a[3]) * d * c - jnp.exp(a[4]) * x0
    dx1 = jnp.exp(a[3]) * (jnp.exp(a[0]) - x1)
    return jnp.array([dx0, dx1])


def box_harness(x1_hi):
    """The box's three face-flux obligations, one traced query."""

    def h():
        # parameters as point inputs: their exp is traced, not folded.
        # (jnp.array, not jnp.stack: on jax 0.11 stack is its own primitive,
        # outside the §2 census list — the first run's UNKNOWN was exactly
        # that one equation falling to ⊤, named by the coverage line.)
        a = jnp.array([any_array((), "float64", (ai, ai)) for ai in A])
        b = any_array((), "float64", (B, B))
        # the box, stated in its own (x1, c) coordinates
        cf = any_array((), "float64", (C_MIN, math.inf))
        x1f = any_array((), "float64", (X1_LO, x1_hi))
        # face x1 = x1_hi (c free): flow must point down
        y_hi = jnp.array([jnp.exp(a[1]) - cf, x1_hi * 1.0])
        o1 = assert_(field(y_hi, a, b)[1] < 0.0)
        # face x1 = X1_LO (c free): flow must point up
        y_lo = jnp.array([jnp.exp(a[1]) - cf, X1_LO * 1.0])
        o2 = assert_(field(y_lo, a, b)[1] > 0.0)
        # face c = C_MIN (x1 free): the singularity boundary must repel
        y_c = jnp.array([jnp.exp(a[1]) - C_MIN, x1f])
        o3 = assert_(field(y_c, a, b)[0] < 0.0)
        return o1, o2, o3

    return h


def run(name, x1_hi):
    cj = trace(box_harness(x1_hi))
    v = make_verdict(
        cj,
        propagate(cj),
        stelling_version=stelling.__version__,
        jax_version=jax_version(),
        precision_config="jax_enable_x64=True",
    )
    print(f"==== {name}")
    print(v.render())
    print()
    return v


v1 = run("candidate #1: the hand-proved box", X1_HI)
v2 = run("§10.8 mutation (must NOT verify): x1 capped at 300 < x1*", 300.0)

assert v2.status != "VERIFIED", "the mutation verified — the checker is vacuous"
print(f"candidate #1: {v1.status}; mutation: {v2.status} (red as required)")
