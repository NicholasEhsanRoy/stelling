# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Which probe index witnesses, per corpus row.

The measurement that decides the cap's SHAPE. Two ways to bound the work
by the declared size are available:

* a hard SIZE CAP — above N declared elements, do not search. Costs every
  recovery on a large declaration and nothing on a small one.
* a WORK BUDGET — run fewer probes as the declaration grows. Costs the
  recoveries whose witness sits late in the grid, at every size.

Which is cheaper is an empirical question about where in the 16-point grid
witnesses actually live, and this answers it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import corpus as C  # noqa: E402
from ledger import _harness  # noqa: E402
from stelling import exactness, propagate as P  # noqa: E402
from stelling.harness import trace  # noqa: E402


def first_witness(closed, assume_mode="constrain", semantics="real"):
    """The lowest probe index that witnesses, or None."""
    required = P._assume_equation_ids(closed.jaxpr)
    if not required:
        return None
    for k in range(P._PROBE_COUNT):
        probe = P._Propagator(assume_mode, semantics)
        probe.pin = k
        try:
            probe.run(closed.jaxpr, list(closed.consts), [])
        except Exception:  # noqa: BLE001
            continue
        if exactness.certifies_point_witness(
            required_assumes=required,
            witnessed_assumes=frozenset(
                key for key, ok in probe.assume_witness.items() if ok
            ),
        ):
            return k
    return None


def main():
    hits = []
    print(f"{'row':38s} {'first witnessing probe index'}")
    for row in C.ROWS:
        try:
            closed = trace(_harness(row))
        except Exception as e:  # noqa: BLE001
            print(f"{row.__name__:38s} trace failed: {type(e).__name__}")
            continue
        try:
            k = first_witness(closed)
        except Exception as e:  # noqa: BLE001
            print(f"{row.__name__:38s} raised: {type(e).__name__}")
            continue
        print(f"{row.__name__:38s} {k if k is not None else '-'}")
        if k is not None:
            hits.append(k)
    print()
    print(f"witnessing rows: {len(hits)}")
    for budget in range(1, P._PROBE_COUNT + 1):
        kept = sum(1 for k in hits if k < budget)
        print(f"  probes={budget:2d}: {kept}/{len(hits)} witnesses still found"
              f"  ({100 * kept / max(len(hits), 1):.0f}%)")


if __name__ == "__main__":
    main()
