"""S2b: EVERY float dtype jax.numpy exposes -- which have a finite range an
int64 literal can leave quietly, and what the comparison then runs as."""
import warnings
import numpy as np
import ml_dtypes
import jax, jax.numpy as jnp
from stelling._tripwire import perimeter
from stelling.propagate import _FLOAT_FORMATS

names = sorted(
    n for n in dir(jnp)
    if any(n.startswith(p) for p in ("float", "bfloat"))
    and isinstance(getattr(jnp, n), type)
)
print(f"jax={jax.__version__} x64={jax.config.jax_enable_x64}")
print(f"catalogued in propagate._FLOAT_FORMATS: {sorted(_FLOAT_FORMATS)}")
print(f"{'dtype':20s} {'max':>14s} {'catalogued':>10s} {'lit':>10s} {'runs as':>8s} {'disarmed':>9s} {'armed'}")
INT64_MAX = 2**63 - 1
rows = []
for n in names:
    dt = getattr(jnp, n)
    try:
        fi = ml_dtypes.finfo(dt) if not hasattr(np, n) else np.finfo(dt)
    except Exception:
        try:
            fi = ml_dtypes.finfo(dt)
        except Exception as e:
            print(f"{n:20s} finfo failed: {e}"); continue
    mx = float(fi.max)
    catalogued = n in _FLOAT_FORMATS
    quiet_possible = mx < INT64_MAX
    if not quiet_possible:
        print(f"{n:20s} {mx:>14.6g} {str(catalogued):>10s} {'-':>10s} {'-':>8s} "
              f"{'-':>9s} every int64 is finite here")
        continue
    lit = int(mx) * 2 + 3           # comfortably past max, still an int
    try:
        x = jnp.zeros((3,), dt)
    except BaseException as e:
        print(f'{n:20s} {mx:>14.6g} {str(catalogued):>10s} zeros() failed: {type(e).__name__}'); continue
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            got = np.asarray(x <= lit).tolist()
            disarmed = "SILENT"
        except RuntimeWarning:
            disarmed = "WARNS"; got = None
        except BaseException as e:
            disarmed = type(e).__name__; got = None
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            image = float(np.asarray(lit).astype(np.dtype(dt)))
        except Exception:
            image = float("nan")
    try:
        with perimeter.armed(("array",)) as st:
            assert st.armed, st.explanation
            _ = x <= lit
        armed = "NO REFUSAL"
    except perimeter.NarrowingError as e:
        armed = e.finding.reason
    except BaseException as e:
        armed = f"(escaped {type(e).__name__})"
    print(f"{n:20s} {mx:>14.6g} {str(catalogued):>10s} {lit:>10d} {image!s:>8s} "
          f"{disarmed:>9s} {armed}  {got}")
