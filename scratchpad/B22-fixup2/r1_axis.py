"""R1 -- reproduce the S1 finding: HOST => warns is FALSE.

Drives every (dtype, construction door) pair under simplefilter("error"),
with and without jax in the call, and prints WARNS/SILENT plus the image.
"""
import os, warnings, sys
import numpy as np
import ml_dtypes

def probe(label, fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            v = fn()
            return label, "SILENT", repr(np.asarray(v).reshape(-1)[0])
        except RuntimeWarning as e:
            return label, "WARNS", f"RuntimeWarning: {e}"
        except Exception as e:
            return label, type(e).__name__, str(e)[:70]

rows = []

# ---- pure numpy / ml_dtypes: no jax in the call at all -------------------
for nm, dt in [("float16", np.float16), ("float32", np.float32),
               ("bfloat16", ml_dtypes.bfloat16),
               ("float8_e5m2", ml_dtypes.float8_e5m2),
               ("float8_e4m3fn", ml_dtypes.float8_e4m3fn)]:
    rows.append(probe(f"np.array([1e300], dtype={nm})",
                      lambda dt=dt: np.array([1e300], dtype=dt)))
    rows.append(probe(f"np.asarray(1e300).astype({nm})",
                      lambda dt=dt: np.asarray(1e300).astype(dt)))
    rows.append(probe(f"np.full((2,), 1e300, dtype={nm})",
                      lambda dt=dt: np.full((2,), 1e300, dtype=dt)))
    rows.append(probe(f"{nm}(1e300)  [scalar ctor]",
                      lambda dt=dt: np.asarray(dt(1e300))))

print("=== NO JAX IN THE CALL (numpy + ml_dtypes only) ===")
for lab, st, im in rows:
    print(f"  {st:8s}  {lab:48s}  {im}")

# ---- now with jax --------------------------------------------------------
import jax, jax.numpy as jnp
print()
print(f"=== JAX {jax.__version__}  x64={jax.config.jax_enable_x64} ===")
rows = []
for nm in ["float16", "float32", "bfloat16", "float8_e5m2", "float8_e4m3fn"]:
    dt = getattr(jnp, nm)
    rows.append(probe(f"jnp.full((2,), 1e300, {nm})",
                      lambda dt=dt: np.asarray(jnp.full((2,), 1e300, dt))))
    rows.append(probe(f"jnp.array([1e300], {nm})",
                      lambda dt=dt: np.asarray(jnp.array([1e300], dt))))
    rows.append(probe(f"jnp.{nm}(1e300)",
                      lambda dt=dt: np.asarray(dt(1e300))))
    rows.append(probe(f"jnp.asarray(1e300).astype({nm})",
                      lambda dt=dt: np.asarray(jnp.asarray(1e300).astype(dt))))
for lab, st, im in rows:
    print(f"  {st:8s}  {lab:48s}  {im}")
