"""G6: is the StableHLO control blind to the perturbation it exists to detect?"""
import hashlib
import numpy as np
import jax, jax.numpy as jnp
from stelling._tripwire import perimeter

C32 = jnp.zeros((4,), jnp.float32)

def _lowerable(x, n):
    assume_ok = (n >= 1) & (n <= 3)
    def body(carry, _):
        return carry + x, carry
    total, _ = jax.lax.scan(body, jnp.zeros((4,), jnp.float32), None, length=3)
    picked = jax.lax.cond(n > 1, lambda t: t * 2.0, lambda t: t, total)
    v = jax.vmap(lambda a: a + 1.0)(picked)
    _, w = jax.lax.while_loop(
        lambda s: s[0] < 3, lambda s: (s[0] + 1, s[1] + 1.0), (jnp.int32(0), v)
    )
    return jnp.all(w <= 1e9) & assume_ok

def lower(**kw):
    jax.clear_caches()
    return jax.jit(_lowerable).lower(C32, jnp.int32(2)).as_text(**kw)

def h(s): return hashlib.sha256(s.encode()).hexdigest()[:16]

print(f"jax={jax.__version__} x64={jax.config.jax_enable_x64}")
for kw, label in (({}, "as_text()"), ({"debug_info": True}, "as_text(debug_info=True)")):
    plain = lower(**kw)
    checks = perimeter.snapshot()
    with perimeter.armed(("tracer", "array")) as st:
        assert st.armed, st.explanation
        before = perimeter.snapshot()
        armed_text = lower(**kw)
        after = perimeter.snapshot()
    nloc = plain.count("loc(")
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)
             if isinstance(after.get(k), int) and after.get(k, 0) != before.get(k, 0)}
    print(f"  {label:30s} len={len(plain):6d} loc(={nloc:4d}  "
          f"armed=={('IDENTICAL' if plain == armed_text else 'DIFFERS')}  "
          f"({h(plain)} vs {h(armed_text)})  guard-delta={delta}")
