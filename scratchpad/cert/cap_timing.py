# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Gate 2: what the certificate search costs, and what the cap costs back.

Three measurements, all with the load average printed beside them because
the box is loaded:

1. **cost by declared size**, for the two shapes that bound it — a search
   that SUCCEEDS (one probe: the first witness ends it) and one that FAILS
   (the full `_PROBE_COUNT` grid) — against the same query with the
   certificate's route closed, which is the pre-certificate time;
2. **the cap's effect**: the same query at a declared size above
   `_CERT_MAX_ELEMENTS`, where the search does not run;
3. **the cap's COST in recovered refutations** — the number of sound
   refutations that stay withheld purely because the declaration is
   large. Reported, not defined away.

Run: python scratchpad/cert/cap_timing.py
"""

from __future__ import annotations

import os
import time

import jax

jax.config.update("jax_enable_x64", True)

from stelling import exactness, propagate as P  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402


def _load():
    return "load " + ", ".join(f"{v:.2f}" for v in os.getloadavg())


def _succeeds(n):
    def h():
        x = any_array((n,), "float64", (0.0, 1.0))
        y = x * 2.0
        assume(y >= 0.5)          # region [0.25, 1]^n — INHABITED
        return (assert_(x <= -1.0),)

    return h


def _fails(n):
    def h():
        x = any_array((n,), "float64", (0.0, 1.0))
        y = x - x
        assume(y >= 0.5)          # region EMPTY: the full grid is walked
        return (assert_(x <= -1.0),)

    return h


def _time(closed, *, certificate, reps=5):
    """Milliseconds, best of ``reps``.

    ``certificate=False`` disables `_region_witness` OUTRIGHT — the
    pre-certificate behaviour, where no search runs at all. NOT by forcing
    `certifies_point_witness` to False, which was the first version of
    this function and measured the search running the FULL grid and
    failing: it reported the certificate making a 4096-element query four
    times FASTER, which is what a wrong baseline looks like.
    """
    if not certificate:
        saved = P._region_witness
        P._region_witness = lambda *a, **k: False
    try:
        P.propagate(closed)  # warm
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            P.propagate(closed)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
    finally:
        if not certificate:
            P._region_witness = saved
    return best * 1e3


def main():
    print(f"=== certificate search cost by DECLARED SIZE ({_load()}) ===")
    print(f"_CERT_MAX_ELEMENTS = {P._CERT_MAX_ELEMENTS}, "
          f"_PROBE_COUNT = {P._PROBE_COUNT}, "
          f"_CERT_PROBE_BUDGET = {P._CERT_PROBE_BUDGET}, "
          f"_CERT_MIN_PROBES = {P._CERT_MIN_PROBES}")
    print(f"{'n':>7} {'probes':>7} {'shape':10s} {'no cert (ms)':>13} "
          f"{'with cert (ms)':>15} {'x':>6} {'certified':>10}")
    for n in (1, 16, 64, 256, 1024, 2048, 4096, 8192):
        for label, mk in (("succeeds", _succeeds), ("fails", _fails)):
            closed = trace(mk(n))
            base = _time(closed, certificate=False)
            full = _time(closed, certificate=True)
            got = P.propagate(closed).region_inhabited
            k = (0 if n > P._CERT_MAX_ELEMENTS
                 else P._certificate_probe_count(n))
            print(f"{n:7d} {k:7d} {label:10s} {base:13.2f} {full:15.2f} "
                  f"{full / base:6.2f} {str(got):>10}")

    print()
    print(f"=== the cap's COST in recovered refutations ({_load()}) ===")
    print("  BOTH bounds off = full 16-probe grid, no size cap")
    cost_size, cost_budget = 0, 0
    for n in (64, 256, 1024, 2048, 4096, 8192, 16384):
        closed = trace(_succeeds(n))
        capped = P.propagate(closed)
        s_max, s_bud, s_min = (
            P._CERT_MAX_ELEMENTS, P._CERT_PROBE_BUDGET, P._CERT_MIN_PROBES
        )
        # budget off, size cap ON: isolates the probe budget's cost
        P._CERT_PROBE_BUDGET, P._CERT_MIN_PROBES = 10**9, P._PROBE_COUNT
        budget_off = P.propagate(closed)
        # both off: isolates the size cap's cost on top
        P._CERT_MAX_ELEMENTS = 10**9
        t0 = time.perf_counter()
        both_off = P.propagate(closed)
        both_off_ms = (time.perf_counter() - t0) * 1e3
        P._CERT_MAX_ELEMENTS, P._CERT_PROBE_BUDGET, P._CERT_MIN_PROBES = (
            s_max, s_bud, s_min
        )

        def _lost(a, b):
            return sum(
                1
                for x, y in zip(a.obligations, b.obligations)
                if x.status != y.status
            )

        lb, ls = _lost(capped, budget_off), _lost(budget_off, both_off)
        cost_budget += lb
        cost_size += ls
        print(f"  n={n:6d} probes={P._certificate_probe_count(n):2d} "
              f"shipped={str(capped.region_inhabited):5s} "
              f"budget-off={str(budget_off.region_inhabited):5s} "
              f"both-off={str(both_off.region_inhabited):5s} | "
              f"lost to the BUDGET: {lb}, lost to the SIZE CAP: {ls} "
              f"(unbounded search: {both_off_ms:.1f} ms)")
    print(f"  TOTAL recoveries lost to the probe budget: {cost_budget}")
    print(f"  TOTAL recoveries lost to the size cap:     {cost_size}")

    print()
    print(f"=== the search against the pipeline it sits in ({_load()}) ===")
    from stelling.preconditions import check

    for n in (256, 1024, 4096):
        h = _fails(n)  # the WORST case: the full grid, no witness
        t0 = time.perf_counter()
        check(h, vacuity_mode="inputs-only")
        whole = (time.perf_counter() - t0) * 1e3
        closed = trace(h)
        base = _time(closed, certificate=False)
        full = _time(closed, certificate=True)
        print(f"  n={n:5d} whole check(): {whole:8.1f} ms | propagate "
              f"{base:7.1f} -> {full:7.1f} ms | search adds "
              f"{full - base:7.1f} ms ({100 * (full - base) / whole:.0f}% of "
              f"the pipeline)")


if __name__ == "__main__":
    main()
