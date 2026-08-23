"""R2 -- is the bfloat16/float8 construction narrowing done on the HOST?

Evidence 1: make_jaxpr already carries the overflowed constant, before any
XLA program exists.  Evidence 2: which numpy API is running underneath.
"""
import warnings
import numpy as np, ml_dtypes
import jax, jax.numpy as jnp

def _copyto(s, dt):
    o = np.empty(s.shape, dtype=dt); np.copyto(o, s, casting="unsafe"); return o

print(f"jax {jax.__version__}  numpy {np.__version__}  ml_dtypes {ml_dtypes.__version__} "
      f" x64={jax.config.jax_enable_x64}")

print("\n=== make_jaxpr: the constant BEFORE any XLA program exists ===")
for nm in ["float16", "float32", "bfloat16", "float8_e5m2"]:
    dt = getattr(jnp, nm)
    for lab, f in [
        (f"jnp.full((), 1e300, {nm})", lambda dt=dt: jnp.full((), 1e300, dt)),
        (f"jnp.array(1e300, {nm})",    lambda dt=dt: jnp.array(1e300, dt)),
        (f"jnp.{nm}(1e300)",           lambda dt=dt: dt(1e300)),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                jx = jax.make_jaxpr(lambda: f())()
                print(f"  {lab:34s} -> {str(jx).replace(chr(10),' ')[:96]}")
            except Exception as e:
                print(f"  {lab:34s} -> {type(e).__name__}: {str(e)[:60]}")

print("\n=== which numpy API: cast loop vs object-conversion ===")
for nm, dt in [("float16", np.float16), ("bfloat16", ml_dtypes.bfloat16),
               ("float8_e5m2", ml_dtypes.float8_e5m2)]:
    src = np.array([1e300], dtype=np.float64)
    cases = [
        ("np.array(list, dtype=)      [object conv]", lambda dt=dt: np.array([1e300], dtype=dt)),
        ("np.array(f64arr, dtype=)    [cast loop]  ", lambda dt=dt, s=src: np.array(s, dtype=dt)),
        ("f64arr.astype(dt)           [cast loop]  ", lambda dt=dt, s=src: s.astype(dt)),
        ("np.empty+copyto             [cast loop]  ", lambda dt=dt, s=src: _copyto(s, dt)),
        ("dt(1e300)                   [scalar ctor]", lambda dt=dt: dt(1e300)),
        ("np.full((1,), 1e300, dt)                 ", lambda dt=dt: np.full((1,), 1e300, dtype=dt)),
    ]
    print(f"  -- {nm}")
    for lab, f in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                f(); st = "SILENT"
            except RuntimeWarning as e:
                st = "WARNS"
            except Exception as e:
                st = type(e).__name__
        print(f"     {st:8s} {lab}")

