"""R5 -- the CHANGELOG's two counts, over the SIX cases the pre-fix page named."""
import warnings
import numpy as np
import jax, jax.numpy as jnp

def st(build):
    jax.clear_caches()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            np.asarray(build()); return "silent"
        except RuntimeWarning: return "WARNS"
        except BaseException as e: return type(e).__name__

X = bool(jax.config.jax_enable_x64)
print(f"jax {jax.__version__}  x64={X}")

EAGER = [
    ("1 jnp.full((2,), 1e300, float32)",  lambda: jnp.full((2,), 1e300, jnp.float32)),
    ("2 jnp.full((2,), 70000.0, float16)",lambda: jnp.full((2,), 70000.0, jnp.float16)),
    ("3 jnp.float16(70000.0)",            lambda: jnp.float16(70000.0)),
    ("4 x_f32 + 1e300",                   lambda: jnp.zeros((2,), jnp.float32) + 1e300),
    ("5 x_f16 + 70000.0",                 lambda: jnp.zeros((2,), jnp.float16) + 70000.0),
    ("6 asarray([1e300,1e300]).astype(float32)",
                                          lambda: jnp.asarray([1e300, 1e300]).astype(jnp.float32)),
]
print("-- EAGER")
n = 0
for lab, b in EAGER:
    r = st(b); n += (r == "WARNS")
    print(f"   {r:8s} {lab}")
print(f"   => {n} of 6 warn eagerly")

# The jit spellings.  Case 6's operand is built OUTSIDE the warning window,
# so what is measured is the astype under the trace and nothing else.
op6 = jnp.asarray([1e300, 1e300])
print(f"-- JIT   (case 6 operand built outside the window: dtype={op6.dtype})")
JIT = [
    ("1 jit(full 1e300 f32)",   lambda: jax.jit(lambda: jnp.full((2,), 1e300, jnp.float32))()),
    ("2 jit(full 70000.0 f16)", lambda: jax.jit(lambda: jnp.full((2,), 70000.0, jnp.float16))()),
    ("3 jit(float16(70000.0))", lambda: jax.jit(lambda: jnp.float16(70000.0))()),
    ("4 jit(x_f32 + 1e300)",    lambda: jax.jit(lambda a: a + 1e300)(jnp.zeros((2,), jnp.float32))),
    ("5 jit(x_f16 + 70000.0)",  lambda: jax.jit(lambda a: a + 70000.0)(jnp.zeros((2,), jnp.float16))),
    ("6 jit(a.astype(float32)) on [1e300,1e300]",
                                lambda: jax.jit(lambda a: a.astype(jnp.float32))(op6)),
]
m = 0
for lab, b in JIT:
    r = st(b); m += (r == "WARNS")
    print(f"   {r:8s} {lab}")
print(f"   => {m} of 6 warn inside jit")
