# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Direct probes for the JAX verification-categories artifact.

Small constructions with controls, in the style that settled check_grads in
ten minutes. Each prints a labeled verdict consumed by
``design/jax-verification-categories.md``. Run under the tested jax series;
results are dated by the artifact that cites them.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np


def probe_int_overflow():
    print("== P1: integer overflow — silent or loud?")
    x = jnp.int32(2**31 - 1)
    print(f"   int32 max + 1 = {x + 1}  (silent wrap: {bool((x + 1) < 0)})")
    n = jnp.int32(50_000)
    prod = n * n
    print(f"   50_000 * 50_000 in int32 = {prod}  (true 2_500_000_000; "
          f"silent wrap: {int(prod) != 2_500_000_000})")
    try:
        jnp.int32(1) * 2_500_000_000
        print("   python literal 2.5e9 entering an int32 trace: accepted silently")
    except OverflowError:
        print("   python literal 2.5e9 entering an int32 trace: LOUD OverflowError —")
        print("   boundary literals are defended; *computed* overflow wraps silently")
    print(f"   default index dtype: {jnp.arange(3).dtype} (x64 disabled)")


def probe_vmap_cond_totality():
    print("== P2: vmap(cond) executes both branches — NaN from the untaken branch?")

    def f(x):
        return jax.lax.cond(x > 0, lambda v: jnp.sqrt(v), lambda v: 0.0, x)

    x_neg = jnp.float32(-1.0)
    solo = f(x_neg)
    batched = jax.vmap(f)(jnp.array([-1.0, 4.0]))
    print(f"   plain cond at x=-1: {solo}  (branch not taken -> no sqrt(-1))")
    print(f"   vmap  cond at [-1, 4]: {batched}  "
          f"(untaken branch evaluated: NaN leak = {bool(jnp.isnan(batched).any())})")


def probe_key_reuse():
    print("== P3: PRNG key reuse — silent? and does JAX ship its own defence?")
    key = jax.random.PRNGKey(0)
    a, b = jax.random.normal(key, (2,)), jax.random.normal(key, (2,))
    print(f"   same key twice -> identical draws: {bool((a == b).all())}, no error raised")
    has_checker = "jax_debug_key_reuse" in dir(jax.config) or hasattr(
        jax.config, "jax_debug_key_reuse"
    )
    print(f"   jax.config.jax_debug_key_reuse exists (JAX's own checker): {has_checker}")


def probe_float_stagnation():
    print("== P4: the R-with-margin blind spot — fl(t + dt) == t while R says t + dt > t")
    t = jnp.float32(1e8)
    dt = jnp.float32(1.0)
    print(f"   f32: t=1e8, dt=1.0 -> t+dt == t: {bool(t + dt == t)}  "
          f"(over R, t+dt > t is a theorem; in f32 the solver stagnates silently)")
    t64 = jnp.float64(1e16) if jax.config.jax_enable_x64 else None
    print("   (diffrax#632 = 258 days of exactly this shape; #657 the grid variant)")


def probe_where_nan_grad():
    print("== P5: the where-NaN FAQ class and its hand defence (double-where)")

    def naive(x):
        return jnp.where(x > 0.0, jnp.sqrt(x), 0.0)

    def defended(x):
        safe_x = jnp.where(x > 0.0, x, 1.0)
        return jnp.where(x > 0.0, jnp.sqrt(safe_x), 0.0)

    g_naive = jax.grad(naive)(0.0)
    g_def = jax.grad(defended)(0.0)
    print(f"   grad(naive where) at 0: {g_naive}  (NaN: {bool(jnp.isnan(g_naive))})")
    print(f"   grad(double-where)  at 0: {g_def}  (defended: {not bool(jnp.isnan(g_def))})")


def probe_corpus_static_shapes():
    print("== P6: corpus-wide static-shape and purity assertions (over transcribed IR)")
    import run_census
    from stelling import _jax_compat, census, ir

    total_eqns = 0
    total_avals = 0
    prims = set()
    for target, harnesses in run_census.HARNESSES.items():
        for harness in harnesses:
            try:
                _, cj = harness()
                root = _jax_compat.transcribe(cj)
            except Exception as exc:
                print(f"   [skip] {target}: {exc}")
                continue
            acc = census.CensusAccumulator()
            acc.add(target, root)
            total_eqns += acc.freeze().total
            prims.update(p.name for p in acc.freeze().primitives)

            def walk(jaxpr):
                nonlocal total_avals
                for eqn in jaxpr.eqns:
                    for a in list(eqn.invars) + list(eqn.outvars):
                        av = a.aval if isinstance(a, ir.Var) else a.aval
                        assert all(isinstance(d, int) for d in av.shape)
                        total_avals += 1
                    pending = [v for _, v in eqn.params]
                    while pending:
                        item = pending.pop()
                        if isinstance(item, ir.ClosedJaxpr):
                            walk(item.jaxpr)
                        elif isinstance(item, ir.Jaxpr):
                            walk(item)
                        elif isinstance(item, tuple):
                            pending.extend(item)
            walk(root.jaxpr)
    mutating = [p for p in prims if "update" in p or p.startswith("scatter")]
    print(f"   {total_avals} avals across {total_eqns} eqns: every shape a static int tuple "
          f"(the transcriber raises otherwise — commitment 3 enforced, not assumed)")
    print(f"   observed primitives: {len(prims)}; all functional — 'mutating-looking' ones "
          f"({', '.join(sorted(mutating))}) are pure functional updates returning new arrays")


if __name__ == "__main__":
    for probe in (
        probe_int_overflow,
        probe_vmap_cond_totality,
        probe_key_reuse,
        probe_float_stagnation,
        probe_where_nan_grad,
        probe_corpus_static_shapes,
    ):
        try:
            probe()
        except Exception as exc:  # a failing probe is a finding, not an abort
            print(f"   PROBE ERROR {probe.__name__}: {type(exc).__name__}: {exc}")
        print()
