# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The independent oracle: concrete points, numpy, no stelling analysis.

For each corpus row it samples MEMBERS OF THE DECLARED SET — values of the
declaration's own dtype inside its declared box — evaluates every `assume`
and every `assert_` at each sample in binary64, and reports, per row:

* ``nonempty``: at least one sampled member satisfies every assume. A
  POSITIVE existence claim, and sound: the point is exhibited.
* per obligation, ``sat_at_admissible``: an admissible sample at which the
  obligation is TRUE. If stelling says REFUTED, this is a **WRONG
  REFUTED**, definitively.
* per obligation, ``viol_at_admissible``: an admissible sample at which
  the obligation is FALSE. If stelling says VERIFIED, this is a **WRONG
  VERIFIED**, definitively.

**What the oracle CANNOT do**, said plainly: sampling never proves a
region empty and never proves an obligation holds everywhere. Its two
verdicts above are existence claims and those are the only ones it makes.
"Oracle found no admissible sample" is reported as exactly that.
"""

from __future__ import annotations

import math

import numpy as np

_INT_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "uint16",
               "uint32", "uint64", "bool"}


class _Oracle:
    """The backend a corpus row runs against in oracle mode.

    ``mode`` selects how an array declaration is filled. INDEPENDENT
    per-element draws are the natural choice and they are useless on a
    64-element declaration whose assume must hold on EVERY element:
    `2x >= 0.5` over `x ∈ [0,1]^64` has admissible probability
    `0.75^64 ≈ 1e-8`, so a million independent draws find nothing and the
    oracle would report a genuinely inhabited region as "no admissible
    sample". Measured on `r31_wide_declaration`, which is exactly why the
    CORRELATED and CORNER modes exist: they fill every element from one
    draw, which is where an elementwise assume's admissible set lives.
    """

    np = np

    def __init__(self, rng, mode="independent"):
        self._rng = rng
        self._mode = mode
        self.assumes: list[bool] = []
        self.obligations: list[bool] = []

    def any(self, shape, dtype, bounds):
        lo, hi = float(bounds[0]), float(bounds[1])
        n = int(np.prod(shape)) if shape else 1
        draws = 1 if self._mode in ("correlated", "lo", "hi", "mid") else n
        if dtype in _INT_DTYPES:
            a, b = math.ceil(lo), math.floor(hi)
            if self._mode == "lo":
                vals = np.full(draws, a)
            elif self._mode == "hi":
                vals = np.full(draws, b)
            elif self._mode == "mid":
                vals = np.full(draws, (a + b) // 2)
            else:
                vals = self._rng.integers(a, b + 1, size=draws)
            out = np.asarray(np.broadcast_to(vals, (n,)), dtype=np.dtype(dtype))
        else:
            if self._mode == "lo":
                vals = np.full(draws, lo)
            elif self._mode == "hi":
                vals = np.full(draws, hi)
            elif self._mode == "mid":
                vals = np.full(draws, 0.5 * (lo + hi))
            else:
                vals = self._rng.uniform(lo, hi, size=draws)
                # the ENDPOINTS are members too, and narrow regions live
                # at the corners; give them real weight
                mask = self._rng.random(draws) < 0.25
                vals = np.where(mask, self._rng.choice([lo, hi], size=draws), vals)
            out = np.asarray(np.broadcast_to(vals, (n,)), dtype=np.dtype(dtype))
        return out.reshape(shape) if shape else out.reshape(())

    def assume(self, pred):
        self.assumes.append(bool(np.all(np.asarray(pred))))

    def assert_(self, pred):
        self.obligations.append(bool(np.all(np.asarray(pred))))


def measure(row, *, samples=20000, seed=0):
    """Run ``row`` over concrete members of its declared set: the three
    deterministic fills (all-lo, all-hi, all-mid), then ``samples`` draws
    split between independent and correlated per-element filling.

    Returns ``(nonempty, sat_at_admissible, viol_at_admissible,
    n_obligations, n_admissible)``."""
    rng = np.random.default_rng(seed)
    nonempty = False
    n_ob = None
    sat: list[bool] = []
    viol: list[bool] = []
    admissible = 0
    modes = ["lo", "hi", "mid"] + [
        "independent" if k % 2 else "correlated" for k in range(samples)
    ]
    for mode in modes:
        o = _Oracle(rng, mode)
        with np.errstate(all="ignore"):
            row(o)
        if n_ob is None:
            n_ob = len(o.obligations)
            sat = [False] * n_ob
            viol = [False] * n_ob
        if not all(o.assumes):
            continue
        nonempty = True
        admissible += 1
        for i, ok in enumerate(o.obligations):
            if ok:
                sat[i] = True
            else:
                viol[i] = True
    return nonempty, sat, viol, n_ob or 0, admissible
