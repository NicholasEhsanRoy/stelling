# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy <nicholas.roy@formulearn.org>
# SPDX-License-Identifier: Apache-2.0
"""F2 — reproduce the counter-construction to "BRANCH-SCOPED ASSUMES ARE
NEVER CERTIFIED", and its cost twin, from scratch.

Two queries, identical but for the bound inside the branch:

``taken``
    the only assume of the query sits inside the ``x >= 0.5`` branch and
    is SATISFIABLE there.  A probe that pins ``x`` to the box's high
    corner walks INTO the branch, evaluates the assume, finds it
    definitely true and witnesses it.  If the heading were true this
    could not certify.

``untaken``
    the same assume, UNSATISFIABLE inside the branch.  Every admissible
    point walks the side WITHOUT the assume, so the assumed region is
    inhabited and a refutation is owed -- and the STATIC requirement
    declines it, because the assume the IR contains is never witnessed.

Both get a sampling oracle over the executed program, so "the recovery is
sound" and "a sound refutation was lost" are measured and not asserted.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=<tree>/src \\
        python scratchpad/pin/f2_repro.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import stelling  # noqa: E402
from stelling import propagate as P  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402


def _mk(bound):
    def h():
        x = any_array((), "float64", (0.0, 1.0))

        def has_assume(v):
            assume(v >= bound)
            return v * 2.0

        y = jax.lax.cond(x >= 0.5, has_assume, lambda v: v, x)
        return (assert_(y <= -1.0),)

    return h


def _oracle(bound, n=20000, seed=0):
    """Admissible AND violating points of the EXECUTED program.

    A point is admissible when every assume the program EVALUATES at it
    holds -- which for x < 0.5 is vacuously true, the branch carrying the
    assume not being taken.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    took = x >= 0.5
    y = np.where(took, x * 2.0, x)
    admissible = np.where(took, x >= bound, True)
    violating = ~(y <= -1.0)
    return int(np.sum(admissible & violating)), n


def main() -> None:
    print(f"stelling: {stelling.__file__}")
    print(f"jax:      {jax.__version__}")
    for label, bound in (("taken (SATISFIABLE in branch)", 0.25),
                         ("untaken (EMPTY in branch)", 2.0)):
        closed = trace(_mk(bound))
        top = [
            e for e in closed.jaxpr.eqns
            if e.primitive == "stelling_assume"
        ]
        required = P._assume_equation_ids(closed.jaxpr)
        p = P.propagate(closed)
        hit, n = _oracle(bound)
        print(f"\n===== {label} =====")
        print(f"  top-level stelling_assume eqns : {len(top)}")
        print(f"  STATIC required assume ids     : {len(required)}")
        print(f"  narrowing_uncertified          : {p.narrowing_uncertified}")
        print(f"  assume_dropped                 : {p.assume_dropped}")
        print(f"  region_inhabited               : {p.region_inhabited}")
        print(f"  obligation status              : {p.obligations[0].status}")
        cert = [n_ for n_ in p.notes if "CERTIFIED NON-EMPTY" in n_]
        print(f"  certificate note               : "
              f"{cert[0][:96] + '...' if cert else '(none)'}")
        print(f"  oracle admissible AND violating: {hit}/{n}")

        walked_around = walked_into = 0
        for k in range(P._PROBE_COUNT):
            probe = P._Propagator("constrain", "real")
            probe.pin = k
            try:
                probe.run(closed.jaxpr, list(closed.consts), [])
            except Exception:  # noqa: BLE001
                continue
            if probe.assume_witness:
                walked_into += 1
            else:
                walked_around += 1
        print(f"  probes with a NON-EMPTY witness map: {walked_into}"
              f" / EMPTY: {walked_around}  (of {P._PROBE_COUNT})")


if __name__ == "__main__":
    main()
