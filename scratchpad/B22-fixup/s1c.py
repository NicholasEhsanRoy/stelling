"""S1c: where the residue actually is. Host-side numpy cast vs device-side."""
import warnings
import numpy as np
import jax, jax.numpy as jnp

def big32():
    return jnp.asarray([1e30, 1e30], jnp.float32)

CASES = [
    # arithmetic ON DEVICE between two arrays that overflows
    ("f32 array * f32 array -> inf", lambda: big32() * jnp.asarray([1e30, 1e30], jnp.float32)),
    ("jnp.exp(f32 1000.0)", lambda: jnp.exp(jnp.asarray([1000.0], jnp.float32))),
    ("f32 array squared via **2", lambda: big32() ** 2),
    ("jit(f32*f32)", lambda: jax.jit(lambda a: a * a)(big32())),
    # array -> narrower array conversion, on device
    ("f32 big .astype(float16)", lambda: big32().astype(jnp.float16)),
    ("jnp.asarray(f32 big).astype(bfloat16)", lambda: big32().astype(jnp.bfloat16)),
    ("lax.convert_element_type(f32 big, f16)", lambda: jax.lax.convert_element_type(big32(), jnp.float16)),
    ("jit convert f32->f16", lambda: jax.jit(lambda a: a.astype(jnp.float16))(big32())),
    # host-side entry
    ("jnp.asarray(np.float64([1e300]), jnp.float32)", lambda: jnp.asarray(np.array([1e300]), jnp.float32)),
    ("jnp.array([1e300], jnp.float32)", lambda: jnp.array([1e300], jnp.float32)),
    ("np.float64(1e300).astype(np.float32) [numpy control]", lambda: np.float64(1e300).astype(np.float32)),
]
print(f"jax={jax.__version__} x64={jax.config.jax_enable_x64}")
for label, build in CASES:
    jax.clear_caches()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            got = np.asarray(build())
            res = f"SILENT -> {got.tolist()!r} ({got.dtype})"
        except RuntimeWarning as w:
            res = f"WARNS: {w}"
        except BaseException as e:
            res = f"{type(e).__name__}: {e}"
    print(f"   {label:52s} {res}")
