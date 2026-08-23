"""R4 -- the mechanism: is the warning about the TARGET format's range, or
about a float32 intermediate?"""
import warnings
import numpy as np, ml_dtypes

print("numpy", np.__version__, "ml_dtypes", ml_dtypes.__version__)
print("np.dtype(ml_dtypes.float8_e5m2).kind =", np.dtype(ml_dtypes.float8_e5m2).kind)
print("np.dtype(ml_dtypes.bfloat16).kind    =", np.dtype(ml_dtypes.bfloat16).kind)
print("hasattr(np, 'float8_e5m2')           =", hasattr(np, "float8_e5m2"))

def st(f):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            v = f(); return "silent", f"{float(np.asarray(v).reshape(-1)[0]):g}"
        except RuntimeWarning: return "WARNS", "-"
        except Exception as e: return type(e).__name__[:12], "-"

f8  = ml_dtypes.float8_e5m2      # max 57344; fits inside float32 range
bf  = ml_dtypes.bfloat16
f8b = ml_dtypes.float8_e4m3fn    # max 448

print()
print(f"{'target':16s} {'value':>10s} {'overflows f32?':>15s} | {'np.array(list)':>15s} {'astype':>10s} {'np.full':>10s} {'scalarctor':>11s}")
for name, dt in [("float16", np.float16), ("float8_e5m2", f8),
                 ("float8_e4m3fn", f8b), ("bfloat16", bf)]:
    mx = float(ml_dtypes.finfo(np.dtype(dt)).max)
    for val in [mx * 100.0, 1e300]:
        if val == float("inf"):
            continue
        over32 = val > float(np.finfo(np.float32).max)
        a = st(lambda: np.array([val], dtype=np.dtype(dt)))[0]
        b = st(lambda: np.asarray(val).astype(np.dtype(dt)))[0]
        c = st(lambda: np.full((1,), val, dtype=np.dtype(dt)))[0]
        d = st(lambda: dt(val))[0]
        print(f"{name:16s} {val:10.2e} {str(over32):>15s} | {a:>15s} {b:>10s} {c:>10s} {d:>11s}")

# Does the ml_dtypes cast loop EVER report the target format's own range?
print()
print("float32 -> ext cast loop (no float64 anywhere), value inside float32:")
for name, dt in [("float8_e5m2", f8), ("float8_e4m3fn", f8b), ("float16", np.float16)]:
    src = np.array([1e30], dtype=np.float32)
    print(f"  {name:16s} np.float32([1e30]).astype -> {st(lambda: src.astype(np.dtype(dt)))}")
