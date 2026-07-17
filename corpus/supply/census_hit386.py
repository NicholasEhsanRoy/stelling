# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Target census: what does hit386's reconstruction actually emit?

The tier assignment is not the build order for this target. This script
points ``stelling.census`` at the E2a build target — the faithful jax
transcription of diffrax#386's vector field plus the three face-flux
obligation functions of the hand-proved box — and prints the primitive
set. That set, exactly, is the MVP's transfer list.

Faithfulness note: the parameters travel as *arguments* (in the incident
they were data — thousands of solves over sampled parameter sets), so the
``exp`` of each parameter is traced, not constant-folded. The obligation
predicates are returned as booleans here; the E2a harness later wraps the
same math in ``assert_``.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from stelling._jax_compat import transcribe
from stelling.census import CensusAccumulator

A = [6.026932645397832, 4.41195014234956, 5.884199824299863,
     3.673504195449191, 4.17957753821087]
B = -2.823760940491063
Y0 = (4.1154706432848185, 6.831774897154676)
X1_LO, X1_HI, C_MIN = 6.8, 415.0, 0.019


def field(y, a, b):
    x0, x1 = y[0], y[1]
    c = jnp.exp(a[1]) - x0
    d = x1 / (c + jnp.exp(b))
    dx0 = jnp.exp(a[3]) * d * c - jnp.exp(a[4]) * x0
    dx1 = jnp.exp(a[3]) * (jnp.exp(a[0]) - x1)
    return jnp.array([dx0, dx1])


def face_x1_hi(x0, a, b):
    """On x1 = X1_HI the flow must point down: dx1/dt < 0."""
    return field(jnp.array([x0, X1_HI]), a, b)[1] < 0.0


def face_x1_lo(x0, a, b):
    """On x1 = X1_LO the flow must point up: dx1/dt > 0."""
    return field(jnp.array([x0, X1_LO]), a, b)[1] > 0.0


def face_c_min(x1, a, b):
    """On c = C_MIN (x0 = e^{a1} - C_MIN) the boundary must repel:
    dc/dt > 0, i.e. dx0/dt < 0."""
    x0 = jnp.exp(a[1]) - C_MIN
    return field(jnp.array([x0, x1]), a, b)[0] < 0.0


# The tier table from design/primitive-census.md (2026-07-17 census).
TIER = {}
for name in ("jit", "custom_jvp_call", "custom_vjp_call", "remat2"):
    TIER[name] = "0"
for name in ("broadcast_in_dim concatenate copy iota pad reshape slice "
             "split squeeze stack stop_gradient").split():
    TIER[name] = "1a"
for name in ("abs add add_any and convert_element_type div eq gt integer_pow "
             "le lt max min mul ne neg not or rem select_n sign sqrt sub").split():
    TIER[name] = "1b"
for name in "cumsum reduce_and reduce_max reduce_or reduce_sum".split():
    TIER[name] = "1c"
for name in ("bitcast_convert_type", "erf_inv", "exp", "is_finite", "log", "nextafter"):
    TIER[name] = "5"

y0 = jnp.array(Y0)
a = jnp.array(A)
b = jnp.array(B)
x0s = jnp.array(80.0)
x1s = jnp.array(200.0)

acc = CensusAccumulator()
acc.add("field", transcribe(jax.make_jaxpr(field)(y0, a, b)))
acc.add("face_x1_hi", transcribe(jax.make_jaxpr(face_x1_hi)(x0s, a, b)))
acc.add("face_x1_lo", transcribe(jax.make_jaxpr(face_x1_lo)(x0s, a, b)))
acc.add("face_c_min", transcribe(jax.make_jaxpr(face_c_min)(x1s, a, b)))
c = acc.freeze()

print(f"== target census: hit386 reconstruction (jax {jax.__version__})")
print(c.markdown_table())
names = sorted(p.name for p in c.primitives)
print(f"\nbuild list ({len(names)} primitives): {', '.join(names)}")
outside = [n for n in names if TIER.get(n, "?") not in ("0", "1a", "1b", "1c")
           and n != "exp"]
print(f"outside tiers 0-1 and exp: {outside if outside else 'NONE'}")
