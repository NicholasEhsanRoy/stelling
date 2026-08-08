# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy <nicholas.roy@formulearn.org>
# SPDX-License-Identifier: Apache-2.0
"""F5 — the boundary paragraph's dial, reproduced from scratch.

``SOUNDNESS.md`` and ``_region_witness``'s docstring say of the point
``(0.1, 0.2)`` against ``x0 + x1 >= 0.30000000000000004`` that *"the
predicate is INDETERMINATE and no witness is claimed"*.  That is true
under ``semantics="real"``.  This asks the same query on BOTH dials, and
then asks the second, unnamed half: which dial certifies MORE over a
grid of dtypes.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=<tree>/src \\
        python scratchpad/pin/f5_repro.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import stelling  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402

BOUND = 0.30000000000000004


def boundary():
    x0 = any_array((), "float64", (0.1, 0.1))
    x1 = any_array((), "float64", (0.2, 0.2))
    assume(x0 + x1 >= BOUND)
    return (assert_(x0 + x1 <= -1.0),)


def _rows():
    """(name, harness) — rows whose assume narrows an OVER-APPROXIMATED
    intermediate, which is what makes the run withhold and so what makes
    the certificate the thing being scored."""

    def mk(dtype, lo, hi, k):
        def h():
            x = any_array((2,), dtype, (lo, hi))
            assume(x * 2 >= k)
            return (assert_(jnp.sum(x) <= lo - 100),)

        return h

    return [
        ("float64", mk("float64", 0.0, 1.0, 0.5)),
        ("float32", mk("float32", 0.0, 1.0, 0.5)),
        ("int32", mk("int32", 0, 4, 4)),
        ("float16", mk("float16", 0.0, 1.0, 0.5)),
        ("bfloat16", mk("bfloat16", 0.0, 1.0, 0.5)),
    ]


def main() -> None:
    print(f"stelling: {stelling.__file__}")
    print(f"jax:      {jax.__version__}")
    print(f"\nbinary64: 0.1 + 0.2 == {0.1 + 0.2!r}; >= bound -> {0.1 + 0.2 >= BOUND}")
    from fractions import Fraction as Fr

    exact = Fr(1, 10) + Fr(2, 10)
    print(f"exact real: 1/10 + 2/10 = {exact} >= bound -> "
          f"{exact >= Fr(BOUND)}")

    closed = trace(boundary)
    print("\n===== the boundary point, both dials =====")
    for sem in ("real", "ieee"):
        p = propagate(closed, semantics=sem)
        print(f"  {sem:5s}: region_inhabited={p.region_inhabited} "
              f"narrowing_uncertified={p.narrowing_uncertified} "
              f"status={p.obligations[0].status}")

    print("\n===== which dial certifies MORE, per declared dtype =====")
    print(f"  {'dtype':10s} {'real':>6s} {'ieee':>6s}")
    only_real = only_ieee = 0
    for name, h in _rows():
        try:
            c = trace(h)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:10s} trace RAISED {type(e).__name__}")
            continue
        got = {}
        for sem in ("real", "ieee"):
            try:
                got[sem] = propagate(c, semantics=sem).region_inhabited
            except Exception as e:  # noqa: BLE001
                got[sem] = f"RAISED {type(e).__name__}"
        print(f"  {name:10s} {str(got['real']):>6s} {str(got['ieee']):>6s}")
        if got["real"] is True and got["ieee"] is False:
            only_real += 1
        if got["ieee"] is True and got["real"] is False:
            only_ieee += 1
    print(f"\n  certified under real only: {only_real}")
    print(f"  certified under ieee only: {only_ieee}")

    print("\n===== the same split over the whole corpus =====")
    import os
    import sys as _s

    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import corpus_pin

    r_only, i_only, both, neither, rows = [], [], 0, 0, 0
    for name, h in corpus_pin.cases():
        try:
            c = trace(h)
        except Exception:  # noqa: BLE001
            continue
        for mode in ("constrain", "inert"):
            got = {}
            for sem in ("real", "ieee"):
                try:
                    got[sem] = propagate(
                        c, semantics=sem, assume_mode=mode
                    ).region_inhabited
                except Exception:  # noqa: BLE001 — a refusal is an outcome
                    got[sem] = None
            rows += 1
            if got["real"] and got["ieee"]:
                both += 1
            elif got["real"] and not got["ieee"]:
                r_only.append(f"{name}/{mode}")
            elif got["ieee"] and not got["real"]:
                i_only.append(f"{name}/{mode}")
            else:
                neither += 1
    print(f"  corpus rows x assume_mode measured : {rows}")
    print(f"  certified on BOTH dials            : {both}")
    print(f"  certified under REAL only          : {len(r_only)}")
    for n in r_only:
        print(f"      {n}")
    print(f"  certified under IEEE only          : {len(i_only)}")
    for n in i_only:
        print(f"      {n}")
    print(f"  certified on neither               : {neither}")


if __name__ == "__main__":
    main()
