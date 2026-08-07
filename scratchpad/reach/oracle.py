"""Independent oracle: dense sampling + corners, no stelling involved.

Per obligation (identified by its cases.py source LINE) it reports, over
the sampled points of the DECLARED box:

  n_exec        points at which the obligation was actually EVALUATED
                (real control flow: only the taken branch runs)
  n_false_exec  points at which it was evaluated and false
  n_exec_uncond / n_false_uncond   the same, with BOTH cond/switch legs
                forced to run -- the deliberate control against the
                "scored only points taking the branch" blind spot.

Ground truth used downstream:
  * a harness is SOUNDLY-REFUTABLE iff some obligation has n_false_exec>0
  * a harness is TRUE-ON-THE-BOX iff every obligation has n_false_exec==0
  * an obligation is UNREACHABLE iff n_exec == 0 over every sampled point
"""
from __future__ import annotations

import inspect
import itertools
import os

import numpy as np

CASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.py")


class _Rec:
    def __init__(self):
        self.exec_pt = {}   # line -> bool (this point)
        self.val_pt = {}    # line -> bool (AND over evaluations this point)

    def note(self, line, ok):
        self.exec_pt[line] = True
        self.val_pt[line] = self.val_pt.get(line, True) and bool(ok)


def _line():
    f = inspect.currentframe().f_back.f_back
    return f.f_lineno


class OracleCtl:
    """Real control-flow semantics, in plain Python."""

    force_both = False

    def __init__(self, force_both=False):
        self.force_both = force_both

    def cond(self, pred, tf, ff, *ops):
        if self.force_both:
            a = tf(*ops)
            b = ff(*ops)
            return a if bool(np.asarray(pred)) else b
        return tf(*ops) if bool(np.asarray(pred)) else ff(*ops)

    def switch(self, idx, branches, *ops):
        k = int(np.clip(int(np.asarray(idx)), 0, len(branches) - 1))
        if self.force_both:
            outs = [b(*ops) for b in branches]
            return outs[k]
        return branches[k](*ops)

    def scan(self, f, init, xs):
        carry = init
        ys = []
        for i in range(np.asarray(xs).shape[0]):
            carry, y = f(carry, np.asarray(xs)[i])
            ys.append(y)
        return carry, np.stack([np.asarray(y) for y in ys]) if ys else np.asarray([])

    def while_loop(self, cf, bf, init):
        s = init
        n = 0
        while bool(np.asarray(cf(s))):
            s = bf(s)
            n += 1
            if n > 10000:
                raise RuntimeError("oracle while_loop did not terminate")
        return s


def _sample_points(decls, rng, n_random):
    """decls: list of (name, shape, dtype, (lo, hi)). Corners first."""
    pts = []
    per = {}
    for name, shape, dtype, (lo, hi) in decls:
        size = int(np.prod(shape)) if shape else 1
        mid = 0.5 * (lo + hi)
        base = [lo, hi, mid]
        if lo < 0.0 < hi:
            base.append(0.0)
        vals = []
        # all-elements-equal corners
        for b in base:
            vals.append(np.full(shape, b, dtype=np.float64))
        # per-element independent corners (small arrays only)
        if size <= 4:
            for combo in itertools.product([lo, hi, mid], repeat=size):
                vals.append(np.array(combo, dtype=np.float64).reshape(shape))
        per[name] = vals
    names = [d[0] for d in decls]
    for combo in itertools.product(*[per[n] for n in names]):
        pts.append(dict(zip(names, combo)))
    for _ in range(n_random):
        p = {}
        for name, shape, dtype, (lo, hi) in decls:
            p[name] = rng.uniform(lo, hi, size=shape).astype(np.float64)
        pts.append(p)
    return pts


class _JaxShim:
    """`jax` as the oracle needs it: jit is the identity, lax is unused
    (control flow goes through OracleCtl)."""
    jit = staticmethod(lambda f, *a, **k: f)


def _numpy_mode():
    """Swap cases.py's `jnp`/`jax` for numpy + the shim: the oracle must
    execute the same expressions in plain numpy, ~50x faster than eager
    jax dispatch and with no jax tracing anywhere near it."""
    import cases as C
    old = (C.jnp, C.jax)
    C.jnp, C.jax = np, _JaxShim
    return C, old


def run_case(fn, n_random=400, seed=0, force_both=False):
    C, old = _numpy_mode()
    try:
        return _run_case(fn, n_random, seed, force_both)
    finally:
        C.jnp, C.jax = old


def _run_case(fn, n_random=400, seed=0, force_both=False):
    rng = np.random.default_rng(seed)
    # pass 1: collect declarations (needs one dry evaluation)
    decls = []

    def A_collect(name, shape, dtype, bounds):
        decls.append((name, tuple(shape), dtype, bounds))
        lo, hi = bounds
        return np.full(shape, 0.5 * (lo + hi), dtype=np.float64)

    rec0 = _Rec()

    def S_collect(pred):
        rec0.note(_line(), np.all(np.asarray(pred)))
        return pred

    fn(A_collect, S_collect, OracleCtl(force_both))

    pts = _sample_points(decls, rng, n_random)
    n_exec = {}
    n_false = {}
    for p in pts:
        rec = _Rec()

        def A(name, shape, dtype, bounds, _p=p):
            return _p[name]

        def S(pred, _rec=rec):
            _rec.note(_line(), np.all(np.asarray(pred)))
            return pred

        fn(A, S, OracleCtl(force_both))
        for ln, ok in rec.val_pt.items():
            n_exec[ln] = n_exec.get(ln, 0) + 1
            if not ok:
                n_false[ln] = n_false.get(ln, 0) + 1
    return {"n_points": len(pts), "n_exec": n_exec, "n_false": n_false,
            "decls": decls}
