"""R3 -- every float dtype jax.numpy exposes x every host construction door.

Derived from dir(jnp) + ml_dtypes.finfo, not from any list.
Doors, each under warnings.simplefilter("error"):
  D1  jnp.full((2,), LIT, dt)          host, object conversion
  D2  jnp.array([LIT], dt)             host, object conversion
  D3  jnp.<dt>(LIT)                    host, object conversion (scalar ctor)
  N1  np.array([LIT], dtype=dt)        numpy alone, object conversion
  N2  np.asarray(LIT).astype(dt)       numpy alone, array->array cast loop
  N3  np.full((1,), LIT, dtype=dt)     numpy alone, empty+copyto = cast loop
LIT is a Python float that overflows the format.
"""
import warnings, math
import numpy as np, ml_dtypes
import jax, jax.numpy as jnp

X64 = jax.config.jax_enable_x64
print(f"jax {jax.__version__} numpy {np.__version__} ml_dtypes {ml_dtypes.__version__} x64={X64}")

names = []
for n in sorted(dir(jnp)):
    o = getattr(jnp, n, None)
    if not isinstance(o, type):
        continue
    try:
        dt = np.dtype(o)
    except Exception:
        continue
    if n != dt.name:
        continue                      # drop aliases (single/double/float_)
    try:
        ml_dtypes.finfo(dt)
    except Exception:
        continue                      # not a float format
    names.append(n)
print(f"{len(names)} concrete float dtypes in jax.numpy: {names}")
print("  kind=='f' (numpy-native binary float): "
      f"{[n for n in names if np.dtype(getattr(jnp,n)).kind=='f']}")
print("  kind=='V' (ml_dtypes extension dtype): "
      f"{[n for n in names if np.dtype(getattr(jnp,n)).kind=='V']}")

def status(f):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            v = f()
            try:
                x = float(np.asarray(v).reshape(-1)[0])
                im = "inf" if math.isinf(x) else ("nan" if math.isnan(x) else f"{x:g}")
            except Exception:
                im = "?"
            return "silent", im
        except RuntimeWarning:
            return "WARNS", "-"
        except UserWarning as e:
            return "UserWarn", str(e)[:20]
        except Exception as e:
            return type(e).__name__[:12], str(e)[:24]

print()
hdr = (f"{'dtype':20s} {'kind':4s} {'max':>12s} | {'D1':>8s} {'D2':>8s} {'D3':>8s} |"
       f" {'N1':>8s} {'N2':>8s} {'N3':>8s} | image")
print(hdr); print("-"*len(hdr))
for n in names:
    dt = getattr(jnp, n)
    ndt = np.dtype(dt)
    mx = float(ml_dtypes.finfo(ndt).max)
    lit = 1e300 if mx * 1e4 == float("inf") else mx * 1e4
    r = {}
    r["D1"], im = status(lambda: jnp.full((2,), lit, dt))
    r["D2"], _ = status(lambda: jnp.array([lit], dt))
    r["D3"], _ = status(lambda: dt(lit))
    r["N1"], imn = status(lambda: np.array([lit], dtype=ndt))
    r["N2"], _ = status(lambda: np.asarray(lit).astype(ndt))
    r["N3"], _ = status(lambda: np.full((1,), lit, dtype=ndt))
    print(f"{n:20s} {ndt.kind:4s} {mx:12.4g} | {r['D1']:>8s} {r['D2']:>8s} {r['D3']:>8s} |"
          f" {r['N1']:>8s} {r['N2']:>8s} {r['N3']:>8s} | jnp={im} np={imn}")
