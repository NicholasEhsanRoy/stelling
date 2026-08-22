"""S1: does a float value that overflows raise a RuntimeWarning?"""
import os, warnings, sys
import numpy as np
import jax, jax.numpy as jnp

CASES = [
    ("jnp.full((2,), 1e300, jnp.float32)", lambda: jnp.full((2,), 1e300, jnp.float32)),
    ("jnp.full((2,), 70000.0, jnp.float16)", lambda: jnp.full((2,), 70000.0, jnp.float16)),
    ("jnp.float16(70000.0)", lambda: jnp.float16(70000.0)),
    ("x_f32 + 1e300", lambda: jnp.zeros((2,), jnp.float32) + 1e300),
    ("x_f16 + 70000.0", lambda: jnp.zeros((2,), jnp.float16) + 70000.0),
    ("jnp.asarray([1e300,1e300]).astype(jnp.float32)", lambda: jnp.asarray([1e300, 1e300]).astype(jnp.float32)),
    ("jnp.full((2,), 100000, jnp.float16)", lambda: jnp.full((2,), 100000, jnp.float16)),
    ("INT ctl: jnp.full((), 256, jnp.int8)", lambda: jnp.full((), 256, jnp.int8)),
    ("INT ctl: jnp.full((3,), 40000, jnp.int16)", lambda: jnp.full((3,), 40000, jnp.int16)),
]

print(f"jax={jax.__version__} numpy={np.__version__} x64={jax.config.jax_enable_x64}")
for label, build in CASES:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            got = np.asarray(build())
            err = None
        except BaseException as e:
            got, err = None, f"{type(e).__name__}: {e}"
    msgs = [f"{w.category.__name__}: {w.message}" for w in rec]
    shown = repr(got) if err is None else err
    print(f"  {label:52s} -> {shown}")
    if msgs:
        for m in msgs:
            print(f"      WARNS  {m}")
    else:
        print(f"      SILENT")
