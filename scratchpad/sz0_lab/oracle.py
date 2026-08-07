# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Independent oracle: dense sampling + corners over the DECLARED boxes.

Knows nothing about stelling's propagation. Each case supplies
(a) the declared input specs, (b) a numpy twin of the assume predicate,
(c) a numpy twin of the assert predicate. `all()` of a bool array is the
universal reading of both, matching numpy/jax semantics for size-0
(vacuously True).
"""
import itertools
import numpy as np

RNG = np.random.default_rng(20260807)


def _structured(shape, lo, hi):
    size = int(np.prod(shape)) if shape else 1
    if size == 0:
        return [np.zeros(shape, dtype=np.float64)]
    out = []
    for v in (lo, hi, 0.0, (lo + hi) / 2.0):
        if lo <= v <= hi:
            out.append(np.full(shape, v, dtype=np.float64))
    if 1 <= size <= 4:          # every per-element corner mixture
        for combo in itertools.product((lo, hi), repeat=size):
            out.append(np.array(combo, dtype=np.float64).reshape(shape))
    return out


def sweep(specs, assume_fn, assert_fn, n_random=100000):
    """(n_points, n_admitted, n_admitted_violating, witnesses).

    Structured points: full cartesian product of per-input corner sets.
    Random points: n_random JOINT uniform draws over the declared box.
    """
    pools = [_structured(sh, lo, hi) for (sh, _dt, (lo, hi)) in specs]
    pts = list(itertools.product(*pools))
    for _ in range(n_random):
        pts.append(tuple(
            RNG.uniform(lo, hi, size=sh).astype(np.float64)
            if (int(np.prod(sh)) if sh else 1) else np.zeros(sh, np.float64)
            for (sh, _dt, (lo, hi)) in specs
        ))
    n = adm = bad = 0
    wit = []
    for pt in pts:
        n += 1
        if assume_fn is not None and not bool(np.all(assume_fn(*pt))):
            continue
        adm += 1
        if not bool(np.all(assert_fn(*pt))):
            bad += 1
            if len(wit) < 3:
                wit.append(tuple(np.asarray(a).tolist() for a in pt))
    return n, adm, bad, wit
