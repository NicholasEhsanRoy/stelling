"""S1d: where does the RuntimeWarning come from?"""
import traceback, warnings
import numpy as np
import jax, jax.numpy as jnp

def probe(label, build):
    jax.clear_caches()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            build()
            print(f"{label}: SILENT")
        except RuntimeWarning:
            tb = traceback.extract_tb(__import__('sys').exc_info()[2])
            print(f"{label}: WARNS, raised from")
            for fr in tb[-3:]:
                print(f"     {fr.filename}:{fr.lineno} in {fr.name}")

print(f"jax={jax.__version__} x64={jax.config.jax_enable_x64}")
probe("jnp.full((2,), 1e300, f32)", lambda: jnp.full((2,), 1e300, jnp.float32))
probe("jit x_f16 + 70000.0", lambda: jax.jit(lambda x: x + 70000.0)(jnp.zeros((2,), jnp.float16)))
probe("eager x_f16 + 70000.0", lambda: jnp.zeros((2,), jnp.float16) + 70000.0)
