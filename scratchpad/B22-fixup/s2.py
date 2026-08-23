"""S2: the float8 formats outside `_FLOAT_FORMATS`, disarmed and armed."""
import warnings
import numpy as np
import ml_dtypes
import jax, jax.numpy as jnp
from stelling._tripwire import perimeter
from stelling.propagate import _FLOAT_FORMATS

CASES = [
    ("float8_e3m4", 100),
    ("float8_e4m3", 500),
    ("float8_e5m2", 100000),
    ("float8_e4m3fn", 1000),
    # the other ml_dtypes float8s, for completeness
    ("float8_e4m3b11fnuz", 100),
    ("float8_e4m3fnuz", 500),
    ("float8_e5m2fnuz", 100000),
]

print(f"jax={jax.__version__} x64={jax.config.jax_enable_x64}")
print(f"propagate._FLOAT_FORMATS = {sorted(_FLOAT_FORMATS)}")
print(f"{'dtype':20s} {'max':>12s} {'written':>9s} {'runs as':>10s}  {'disarmed':10s} {'armed'}")
for name, lit in CASES:
    dt = getattr(ml_dtypes, name)
    fi = ml_dtypes.finfo(dt)
    x = jnp.zeros((3,), dt)
    # DISARMED
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            out = (x <= lit)
            got = np.asarray(out).tolist()
            with np.errstate(over="ignore"):
                image = float(np.asarray(lit).astype(dt))
            disarmed = f"SILENT {got}"
        except RuntimeWarning as w:
            disarmed = f"WARNS: {w}"
            image = float("nan")
        except BaseException as e:
            disarmed = f"{type(e).__name__}: {e}"
            image = float("nan")
    # ARMED
    try:
        with perimeter.armed(("array",)) as st:
            assert st.armed, st.explanation
            _ = x <= lit
        armed = "NO REFUSAL"
    except perimeter.NarrowingError as e:
        armed = f"{e.finding.reason} -> {e.finding.narrowed_to!r}"
    print(f"{name:20s} {fi.max!s:>12s} {'x <= '+str(lit):>9s} {image!r:>10s}  {disarmed:22s} {armed}")
