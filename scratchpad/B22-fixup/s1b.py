"""S1b: under simplefilter('error'), eager and inside jit."""
import warnings
import numpy as np
import jax, jax.numpy as jnp

EAGER = [
    ("jnp.full((2,), 1e300, jnp.float32)", lambda: jnp.full((2,), 1e300, jnp.float32)),
    ("jnp.full((2,), 70000.0, jnp.float16)", lambda: jnp.full((2,), 70000.0, jnp.float16)),
    ("jnp.float16(70000.0)", lambda: jnp.float16(70000.0)),
    ("x_f32 + 1e300", lambda: jnp.zeros((2,), jnp.float32) + 1e300),
    ("x_f16 + 70000.0", lambda: jnp.zeros((2,), jnp.float16) + 70000.0),
    ("jnp.asarray([1e300,1e300]).astype(jnp.float32)", lambda: jnp.asarray([1e300, 1e300]).astype(jnp.float32)),
    ("jnp.full((2,), 100000, jnp.float16)", lambda: jnp.full((2,), 100000, jnp.float16)),
    ("np.float64(1e300).astype(np.float32)", lambda: np.float64(1e300).astype(np.float32)),
    ("INT: jnp.full((), 256, jnp.int8)", lambda: jnp.full((), 256, jnp.int8)),
    ("INT: jnp.full((3,), 40000, jnp.int16)", lambda: jnp.full((3,), 40000, jnp.int16)),
    ("INT: x_int16 + 40000", lambda: jnp.zeros((3,), jnp.int16) + 40000),
]

JITTED = [
    ("jit: full 1e300 f32", jax.jit(lambda: jnp.full((2,), 1e300, jnp.float32))),
    ("jit: full 70000.0 f16", jax.jit(lambda: jnp.full((2,), 70000.0, jnp.float16))),
    ("jit: x_f32 + 1e300", lambda: jax.jit(lambda x: x + 1e300)(jnp.zeros((2,), jnp.float32))),
    ("jit: x_f16 + 70000.0", lambda: jax.jit(lambda x: x + 70000.0)(jnp.zeros((2,), jnp.float16))),
    ("jit: full 100000 f16", jax.jit(lambda: jnp.full((2,), 100000, jnp.float16))),
    ("jit INT: x_int8 + 256", lambda: jax.jit(lambda x: x + 256)(jnp.zeros((), jnp.int8))),
]

print(f"jax={jax.__version__} x64={jax.config.jax_enable_x64}")
for group, cases in (("EAGER", EAGER), ("JIT", JITTED)):
    print(f"-- {group}")
    for label, build in cases:
        jax.clear_caches()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                got = np.asarray(build())
                res = f"SILENT -> {got.tolist()!r}"
            except RuntimeWarning as w:
                res = f"RAISES RuntimeWarning: {w}"
            except BaseException as e:
                res = f"{type(e).__name__}: {e}"
        print(f"   {label:50s} {res}")
