# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy <nicholas.roy@formulearn.org>
# SPDX-License-Identifier: Apache-2.0
"""F6 — the two witness searches: do they ever run on the same query, and
what is the COMBINED worst case?

``propagate.py`` holds two probe searches.  ``_region_witness`` (the
non-emptiness certificate) is bounded by ``_CERT_MAX_ELEMENTS`` and
``_CERT_PROBE_BUDGET``; ``_reachability_witnesses`` (the branch pass) runs
the full ``_PROBE_COUNT`` grid at any declared size.  The cost sentence in
``SOUNDNESS.md`` says "bounded by the declared size twice over", which is
about the first only.

Run this in the PROBE-COUNTING worktree (``mine/INSTR6``), which adds one
increment to each of the two loops and nothing else.

  ``count``  per-run probe counts over the corpus + the branch grid, so
             "0 queries pay for both" and "the worst combined count is N"
             are measured rather than argued.
  ``time``   wall cost of the uncapped search against a bare walk, at
             declared sizes from 16 to 16384.  Prints the load average
             either side; do not read it off a busy machine.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=<INSTR6>/src \\
        python scratchpad/pin/f6_repro.py count|time
"""

from __future__ import annotations

import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import stelling  # noqa: E402
from stelling import propagate as P  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402


def _branch_violation(n, claim=lambda v: v > 5.0):
    """A violation reachable only inside a `lax.cond` branch, which is
    the shape that makes `_reachability_witnesses` actually probe.

    The OBLIGATION has to be inside the branch, not merely a value that
    flowed out of one: the branch pass runs on recorded branch-scoped
    violations, and an `assert_` written below the `cond` is a top-level
    obligation however branchy its argument was.
    """

    def h():
        x = any_array((n,), "float64", (-1.0, 1.0))
        return (
            jax.lax.cond(
                x[0] >= 0.0,
                lambda v: assert_(claim(v)),
                lambda v: assert_(v > -9.0),
                x,
            ),
        )

    return h


def _cert_row(n):
    """A run the CERTIFICATE fires on: the assume narrows an
    over-approximated intermediate, so the run withholds and looks."""

    def h():
        x = any_array((n,), "float64", (0.0, 1.0))
        assume(x * x <= 0.9)
        return (assert_(jnp.sum(x) <= -1.0),)

    return h


def _both_shot(n):
    """The shape most likely to reach BOTH: a branch-scoped violation
    beside a top-level assume that narrows an over-approximated
    intermediate (so the certificate has something to lift)."""

    def h():
        x = any_array((n,), "float64", (0.0, 1.0))
        assume(x * x <= 0.9)
        return (
            jax.lax.cond(
                x[0] >= 0.0,
                lambda v: assert_(v > 5.0),
                lambda v: assert_(v > -9.0),
                x,
            ),
        )

    return h


def _both_shot_dropped(n):
    """The same, with a DROPPED assume rather than a narrowing one —
    the other of the two states that opens the certificate's gate."""

    def h():
        x = any_array((n,), "float64", (0.0, 1.0))
        y = any_array((n,), "float64", (0.0, 1.0))
        assume(x >= y)
        return (
            jax.lax.cond(
                x[0] >= 0.0,
                lambda v: assert_(v > 5.0),
                lambda v: assert_(v > -9.0),
                x,
            ),
        )

    return h


def _rows():
    rows = []
    for n in (4, 16, 64, 256, 1024, 4096, 8192, 16384):
        rows.append((f"branch_violation_n{n}", _branch_violation(n)))
        rows.append((f"cert_n{n}", _cert_row(n)))
        rows.append((f"both_shot_n{n}", _both_shot(n)))
        rows.append((f"both_shot_dropped_n{n}", _both_shot_dropped(n)))
    return rows


def _corpus_rows():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import corpus_pin

    return corpus_pin.cases()


def count() -> None:
    print(f"stelling: {stelling.__file__}")
    print(f"jax:      {jax.__version__}")
    print(f"_PROBE_COUNT = {P._PROBE_COUNT}  "
          f"_CERT_MAX_ELEMENTS = {P._CERT_MAX_ELEMENTS}  "
          f"_CERT_PROBE_BUDGET = {P._CERT_PROBE_BUDGET}  "
          f"_CERT_MIN_PROBES = {P._CERT_MIN_PROBES}")
    runs = both = 0
    worst = (0, None)
    worst_reach = (0, None)
    worst_region = (0, None)
    detail = []
    allrows = [("corpus/" + n, h) for n, h in _corpus_rows()]
    allrows += [("grid/" + n, h) for n, h in _rows()]
    for name, h in allrows:
        try:
            closed = trace(h)
        except Exception:  # noqa: BLE001
            continue
        for sem in ("real", "ieee"):
            for mode in ("constrain", "inert"):
                P._INSTR_PROBES["region"] = 0
                P._INSTR_PROBES["reach"] = 0
                try:
                    P.propagate(closed, semantics=sem, assume_mode=mode)
                except Exception:  # noqa: BLE001 — a refusal is an outcome
                    pass
                r = P._INSTR_PROBES["region"]
                k = P._INSTR_PROBES["reach"]
                runs += 1
                key = (name, sem, mode)
                if r and k:
                    both += 1
                    detail.append((key, r, k))
                if r + k > worst[0]:
                    worst = (r + k, (key, r, k))
                if k > worst_reach[0]:
                    worst_reach = (k, key)
                if r > worst_region[0]:
                    worst_region = (r, key)
    print(f"\nruns measured                       : {runs}")
    print(f"runs paying for BOTH searches       : {both}")
    for d in detail[:20]:
        print(f"    {d}")
    print(f"worst COMBINED probe count          : {worst[0]}  at {worst[1]}")
    print(f"worst _reachability_witnesses alone : {worst_reach[0]}"
          f"  at {worst_reach[1]}")
    print(f"worst _region_witness alone         : {worst_region[0]}"
          f"  at {worst_region[1]}")
    # the counterfactual worst case: what a query would pay if the two
    # searches were NOT mutually exclusive.  Maximised over the declared
    # size, not read off the size cap -- the certificate's budget is
    # LOOSEST at small n, which is where the sum is largest.
    hyp = max(
        (P._PROBE_COUNT + P._certificate_probe_count(n), n)
        for n in (1, 2, 4, 16, 64, 256, 1024, 4096)
    )
    print(f"\nthe sum they would pay were they NOT exclusive, maximised "
          f"over declared size: {hyp[0]} probes (at n = {hyp[1]}) = "
          f"{P._PROBE_COUNT} + {hyp[0] - P._PROBE_COUNT}; above "
          f"n = {P._CERT_MAX_ELEMENTS} the certificate declines and the "
          f"sum is {P._PROBE_COUNT} again")


def _bare_walk_ms(closed, reps=3):
    """The same query with the branch pass and the certificate both off:
    the propagation cost with neither search in it."""
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        pr = P._Propagator("constrain", "real")
        pr.run(closed.jaxpr, list(closed.consts), [])
        dt = (time.perf_counter() - t0) * 1e3
        best = dt if best is None else min(best, dt)
    return best


def timing() -> None:
    print(f"stelling: {stelling.__file__}")
    print(f"jax:      {jax.__version__}")
    print(f"load before: {os.getloadavg()}")
    print(f"\n{'n':>7} {'reach':>6} {'region':>7} {'propagate ms':>13} "
          f"{'bare walk ms':>13} {'ratio':>7}")
    for n in (16, 64, 256, 1024, 4096, 16384):
        closed = trace(_branch_violation(n))
        P.propagate(closed)  # warm the trace caches
        P._INSTR_PROBES["region"] = 0
        P._INSTR_PROBES["reach"] = 0
        best = None
        for _ in range(3):
            P._INSTR_PROBES["region"] = 0
            P._INSTR_PROBES["reach"] = 0
            t0 = time.perf_counter()
            P.propagate(closed)
            dt = (time.perf_counter() - t0) * 1e3
            best = dt if best is None else min(best, dt)
        bare = _bare_walk_ms(closed)
        print(f"{n:>7} {P._INSTR_PROBES['reach']:>6} "
              f"{P._INSTR_PROBES['region']:>7} {best:>13.1f} "
              f"{bare:>13.1f} {best / bare:>6.1f}x")
    print(f"\nload after: {os.getloadavg()}")


def capcost() -> None:
    """What capping the OLDER search with `_certificate_probe_count`
    would cost in VERDICTS: the reachability keys it would stop finding.

    Reported per row as: the probe index that first certifies each key,
    against the budget the cap would grant.
    """
    print(f"stelling: {stelling.__file__}")
    print(f"jax:      {jax.__version__}")
    shapes = {
        "corner_guard": lambda v: v[0] > 5.0,
        "sum_guard": lambda v: jnp.sum(v) > 5.0,
        "relational_guard": lambda v: v[0] > v[1],
    }
    print(f"{'row':<26}{'n':>7}{'cap':>5}{'keys':>6}"
          f"{'first_idx':>13}{'lost':>6}")
    asked = lost = 0
    for n in (4, 64, 256, 1024, 4096, 8192, 16384):
        for sname, claim in shapes.items():
            closed = trace(_branch_violation(n, claim))
            p = P.propagate(closed)
            cap = P._certificate_probe_count(n)
            firsts = {}
            for k in range(P._PROBE_COUNT):
                probe = P._Propagator("constrain", "real")
                probe.pin = k
                try:
                    probe.run(closed.jaxpr, list(closed.consts), [])
                except Exception:  # noqa: BLE001
                    continue
                for key in probe.certain_reached:
                    firsts.setdefault(key, k)
            real = P._reachability_witnesses(
                closed, p, assume_mode="constrain", semantics="real"
            )
            idxs = sorted(firsts.get(key, -1) for key in real)
            n_lost = sum(1 for i in idxs if i >= cap)
            asked += len(real)
            lost += n_lost
            print(f"{sname + '_n' + str(n):<26}{n:>7}{cap:>5}{len(real):>6}"
                  f"{str(idxs):>11}{n_lost:>6}")
    print(f"\nreachability keys asked: {asked}; "
          f"lost under a _certificate_probe_count cap: {lost}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "count"
    {"count": count, "time": timing, "capcost": capcost}[what]()
