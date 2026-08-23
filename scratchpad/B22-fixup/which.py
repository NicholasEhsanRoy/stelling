import warnings
import numpy as np
import jax, jax.numpy as jnp

CASES = [
    ("jnp.full((2,), 1e300, jnp.float32)", lambda: jnp.full((2,), 1e300, jnp.float32)),
    ("jnp.full((2,), 70000.0, jnp.float16)", lambda: jnp.full((2,), 70000.0, jnp.float16)),
    ("jnp.float16(70000.0)", lambda: jnp.float16(70000.0)),
    ("jnp.array([1e300], jnp.float32)", lambda: jnp.array([1e300], jnp.float32)),
    ("jnp.full((2,), 100000, jnp.float16)", lambda: jnp.full((2,), 100000, jnp.float16)),
    ("jit(x_f32 + 1e300)", lambda: jax.jit(lambda a: a + 1e300)(jnp.zeros((2,), jnp.float32))),
    ("jit(x_f16 + 70000.0)", lambda: jax.jit(lambda a: a + 70000.0)(jnp.zeros((2,), jnp.float16))),
    ("jit astype f32 of [1e300,1e300]", lambda: jax.jit(lambda a: a.astype(jnp.float32))(
        jnp.asarray([1e300, 1e300], jnp.float64 if jax.config.jax_enable_x64 else jnp.float32))),
]
print(f"jax={jax.__version__} x64={jax.config.jax_enable_x64}")
for label, build in CASES:
    jax.clear_caches()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            np.asarray(build()); r = "SILENT"
        except RuntimeWarning as w: r = f"WARNS ({w})"
        except BaseException as e: r = f"{type(e).__name__}: {e}"
    print(f"   {label:38s} {r}")
