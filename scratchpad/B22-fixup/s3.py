"""Is `x_f16 <= 100000` silent? Eager, make_jaxpr, and jit."""
import warnings
import numpy as np
import jax, jax.numpy as jnp

def run(label, fn):
    jax.clear_caches()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            out = fn()
            print(f"   {label:44s} SILENT -> {out}")
        except RuntimeWarning as w:
            print(f"   {label:44s} WARNS: {w}")
        except BaseException as e:
            print(f"   {label:44s} {type(e).__name__}: {e}")

x16 = jnp.zeros((3,), jnp.float16)
x32 = jnp.zeros((3,), jnp.float32)
print(f"jax={jax.__version__} x64={jax.config.jax_enable_x64}")
run("eager  x_f16 <= 100000", lambda: np.asarray(x16 <= 100000).tolist())
run("jaxpr  x_f16 <= 100000", lambda: str(jax.make_jaxpr(lambda a: a <= 100000)(x16)))
run("jit    x_f16 <= 100000", lambda: np.asarray(jax.jit(lambda a: a <= 100000)(x16)).tolist())
run("eager  x_f32 <= 2**31-1", lambda: np.asarray(x32 <= 2**31 - 1).tolist())
run("jaxpr  x_f32 <= 2**31-1", lambda: str(jax.make_jaxpr(lambda a: a <= 2**31 - 1)(x32)))
run("jit    x_f32 <= 2**31-1", lambda: np.asarray(jax.jit(lambda a: a <= 2**31 - 1)(x32)).tolist())
run("eager  x_f16 <= 65505 (inexact)", lambda: np.asarray(x16 <= 65505).tolist())
run("jaxpr  x_f16 <= 65505 (inexact)", lambda: str(jax.make_jaxpr(lambda a: a <= 65505)(x16)))
run("eager  x_int16 + 40000 (INT control)", lambda: np.asarray(jnp.zeros((3,), jnp.int16) + 40000).tolist())
run("jaxpr  x_int16 + 40000 (INT control)", lambda: str(jax.make_jaxpr(lambda a: a + 40000)(jnp.zeros((3,), jnp.int16))))
