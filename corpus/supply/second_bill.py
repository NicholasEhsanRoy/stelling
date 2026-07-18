# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The second-bill facts + the boundary probe (design/second-bill.md).

Part 1: extract, mechanically, the per-⊤-equation facts of the clean
case (operand dtypes/shapes, params) so every bucket assignment in the
reading is pointable-at. Part 2: pose the strongest key-independent
sampler property — the MCLMC isokinetic momentum-norm invariant — and
report which wall, if any, its obligation routes through. No count.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from stelling._jax_compat import trace
from stelling.harness import any_array, assert_
from stelling.propagate import TRANSFERS, propagate

import pytree_probe  # the registered clean/hard harnesses, re-used verbatim

# --- part 1: the clean case's ⊤ equations, with facts -----------------------

cj = trace(pytree_probe.h_clean)


def walk(jaxpr, depth=0):
    for eqn in jaxpr.eqns:
        yield eqn, depth
        for _, v in eqn.params:
            from stelling import ir
            if isinstance(v, ir.ClosedJaxpr):
                yield from walk(v.jaxpr, depth + 1)
            elif isinstance(v, ir.Jaxpr):
                yield from walk(v, depth + 1)


print("== clean case: every ⊤ equation, with the facts")
for eqn, depth in walk(cj.jaxpr):
    if eqn.primitive in TRANSFERS or eqn.primitive in (
        "jit", "custom_jvp_call", "custom_vjp_call", "remat2",
        "stelling_any", "stelling_assume", "stelling_assert",
        "stelling_nonvacuity", "cond",
    ):
        continue
    ins = [
        f"{getattr(a.aval, 'dtype', '?')}{list(getattr(a.aval, 'shape', ()))}"
        + ("(lit)" if type(a).__name__ == "Literal" else "")
        for a in eqn.invars
    ]
    params = {
        k: v for k, v in eqn.params
        if not hasattr(v, "jaxpr") and k not in ("sharding",)
    }
    print(f"  d{depth} {eqn.primitive}: ins={ins} params={params}")

# also: which forms did implemented transfers DECLINE?
p_clean = propagate(cj)
for n in p_clean.notes:
    if "declined" in n or "no sound rule" in n:
        print(f"  DECLINED: {n}")

# --- part 2: the boundary probe ----------------------------------------------

from blackjax.mcmc.mclmc import build_kernel, init as mclmc_init  # noqa: E402

BOUND = 5.0


def logdensity_fn(x):
    return -0.5 * jnp.sum(x**2) + jnp.sum(jnp.log(BOUND - jnp.abs(x)))


kernel = build_kernel()


def h_norm():
    """The key-independent candidate: isokinetic momentum norm = 1 for ANY
    key — posed on the real kernel output."""
    bits_i = any_array((2,), "uint32", (0.0, 4294967295.0))
    key_i = jax.random.wrap_key_data(bits_i)
    pos = any_array((2,), "float64", (-1.0, 1.0))
    state = mclmc_init(pos, logdensity_fn, key_i)
    bits_s = any_array((2,), "uint32", (0.0, 4294967295.0))
    key_s = jax.random.wrap_key_data(bits_s)
    step_size = any_array((), "float64", (0.01, 1.0))
    next_state, info = kernel(
        key_s, state, logdensity_fn, jnp.ones(2), 1.0, step_size
    )
    return assert_(jnp.sum(next_state.momentum**2) <= 1.0 + 1e-9)


print("\n== boundary probe: isokinetic momentum-norm invariant (key-independent)")
cj2 = trace(h_norm)
p2 = propagate(cj2)
print(f"  POSED. obligation: {p2.obligations[0].status}")
print(f"  coverage: {p2.coverage.summary()}")
if p2.coverage.unknown_primitives:
    print(f"  ⊤ primitives: {[n for n, c in p2.coverage.unknown_primitives]}")
