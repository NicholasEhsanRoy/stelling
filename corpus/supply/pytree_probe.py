# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The any_pytree target probe (design/any-pytree-probe.md). No count.

Clean case: diffrax PIDController.adapt_step_size — the REAL method at the
installed version, its real _PidState hand-declared leaf by leaf, posing
dfx#207's own property (next-dt >= dtmin, the opt-in dtmin path) on the
real code. Hard case: blackjax MCLMC kernel — state built by the real
init from declared position leaves, stepped under a forall-key
declaration (raw uint32 key data wrapped by the library's own
wrap_key_data). A key leaf gets no invented interval: if its cone falls
to ⊤, that is the finding.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import blackjax  # noqa: F401  (version report)
import diffrax

from stelling._jax_compat import jax_version, trace
from stelling.harness import any_array, assert_
from stelling.propagate import propagate

print(f"pinned: diffrax {diffrax.__version__} | blackjax {blackjax.__version__} "
      f"| jax {jax_version()}")

DTMIN = 1e-6


def run_case(name, harness):
    print(f"==== {name}")
    try:
        cj = trace(harness)
    except Exception as e:  # blocked at trace/transcription — quote it
        print(f"  BLOCKED at trace/transcribe: {type(e).__name__}: {e}")
        print()
        return
    try:
        p = propagate(cj)
    except Exception as e:
        print(f"  BLOCKED at propagation: {type(e).__name__}: {e}")
        print()
        return
    print(f"  POSED. obligations: {[o.status for o in p.obligations]}")
    print(f"  coverage: {p.coverage.summary()}")
    if p.coverage.unknown_primitives:
        print(f"  ⊤ primitives: {[n for n, _ in p.coverage.unknown_primitives]}")
    for note in p.notes[:6]:
        print(f"  note: {note}")
    print()


# --- clean case: pure structure, no keys ------------------------------------

controller = diffrax.PIDController(rtol=1e-3, atol=1e-6, dtmin=DTMIN)


def h_clean():
    t0 = any_array((), "float64", (0.0, 1.0))
    prev_dt = any_array((), "float64", (1e-4, 1e-1))
    t1 = t0 + prev_dt
    y0 = any_array((1,), "float64", (-1.0, 1.0))
    y1c = any_array((1,), "float64", (-1.0, 1.0))
    y_err = any_array((1,), "float64", (-1e-4, 1e-4))
    # the real _PidState, leaf by leaf: (prev_inv_scaled_error,
    # prev_prev_inv_scaled_error, at_dtmin)
    pinv = any_array((), "float64", (0.1, 10.0))
    pprev = any_array((), "float64", (0.1, 10.0))
    at_min = any_array((), "bool", (0.0, 1.0))
    keep, nt0, nt1, made_jump, new_state, result = controller.adapt_step_size(
        t0, t1, y0, y1c, None, y_err, 5.0, (pinv, pprev, at_min)
    )
    # dfx#207's property, on the real code path
    return assert_((nt1 - nt0) >= DTMIN)


run_case("CLEAN: diffrax PIDController.adapt_step_size (real _PidState, hand-declared)",
         h_clean)


# --- hard case: the key wall -------------------------------------------------

from blackjax.mcmc.mclmc import build_kernel, init as mclmc_init  # noqa: E402

BOUND = 5.0


def logdensity_fn(x):
    # the #969 MRE's bounded target, in-thread verbatim shape
    return -0.5 * jnp.sum(x**2) + jnp.sum(jnp.log(BOUND - jnp.abs(x)))


kernel = build_kernel()


def h_hard():
    bits_i = any_array((2,), "uint32", (0.0, 4294967295.0))
    key_i = jax.random.wrap_key_data(bits_i)  # ∀-key, via the library's own wrap
    pos = any_array((2,), "float64", (-1.0, 1.0))
    state = mclmc_init(pos, logdensity_fn, key_i)  # real init builds real state
    bits_s = any_array((2,), "uint32", (0.0, 4294967295.0))
    key_s = jax.random.wrap_key_data(bits_s)
    step_size = any_array((), "float64", (0.01, 1.0))
    next_state, info = kernel(
        key_s, state, logdensity_fn, jnp.ones(2), 1.0, step_size
    )
    # a real-content property of the bounded target: logdensity <= log-normalizer-free max
    return assert_(next_state.logdensity <= 2.0 * jnp.log(BOUND))


run_case("HARD: blackjax MCLMC kernel (real init state, ∀-key declaration)", h_hard)
