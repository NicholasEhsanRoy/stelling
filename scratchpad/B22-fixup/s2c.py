"""S2c: the eight quiet formats -- eager AND traced, under simplefilter('error')."""
import warnings
import numpy as np
import ml_dtypes
import jax, jax.numpy as jnp

QUIET = [("float16", 100000), ("float8_e3m4", 33), ("float8_e4m3", 483),
         ("float8_e5m2", 114691), ("float8_e4m3fn", 899),
         ("float8_e4m3b11fnuz", 63), ("float8_e4m3fnuz", 483),
         ("float8_e5m2fnuz", 114691)]

def probe(fn):
    jax.clear_caches()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return "SILENT", fn()
        except RuntimeWarning as w:
            return "WARNS", str(w)
        except BaseException as e:
            return type(e).__name__, str(e)[:40]

print(f"jax={jax.__version__} x64={jax.config.jax_enable_x64}")
print(f"{'dtype':20s} {'lit':>8s} {'image':>6s} {'eager':>8s} {'traced':>8s}  runs")
for name, lit in QUIET:
    dt = getattr(jnp, name)
    x = jnp.zeros((3,), dt)
    ek, ev = probe(lambda: np.asarray(x <= lit).tolist())
    tk, tv = probe(lambda: str(jax.make_jaxpr(lambda a: a <= lit)(x)))
    with np.errstate(over="ignore", invalid="ignore"):
        image = float(np.asarray(lit).astype(np.dtype(dt)))
    print(f"{name:20s} {lit:>8d} {image!s:>6s} {ek:>8s} {tk:>8s}  {ev if ek=='SILENT' else ''}")
