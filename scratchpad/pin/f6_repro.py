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
                # the SAME shape, definitely true: `cond` requires equal
                # output types, and an inert filler in the other branch is
                # what keeps the violation branch-scoped
                lambda v: assert_(claim(v) | True),
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
    would cost in VERDICTS.

    The accounting is over the keys the branch pass ASKS about --
    `p.branch_violations`, the violations recorded inside a cond branch --
    and not over the keys the search happened to find.  A key that is
    asked and never certified within the budget is a `violated-over-set`
    that stays `unknown`: the loss is exactly what a cap would cause, and
    counting found-keys instead would miss every one of them by
    construction.

    Three guard shapes, because WHICH probe first certifies a key is the
    whole question:

      corner_guard      a guard on one element; the plain corner anchors
                        decide it immediately
      sum_guard         a guard on a SUM, which no plain corner satisfies
      relational_guard  a guard relating two ELEMENTS, which the plain
                        anchors cannot witness at all -- they put every
                        element at the same value
    """
    print(f"stelling: {stelling.__file__}")
    print(f"jax:      {jax.__version__}")

    def corner_guard(n):
        def h():
            x = any_array((n,), "float64", (-1.0, 1.0))
            return jax.lax.cond(
                x[0] >= 0.0,
                lambda v: assert_(v > 5.0),
                lambda v: assert_(v > -9.0),
                x,
            )

        return h

    def sum_guard(n):
        def h():
            x = any_array((n,), "float64", (-1.0, 1.0))
            return jax.lax.cond(
                (jnp.sum(x) >= -0.25) & (jnp.sum(x) <= 0.25),
                lambda v: assert_(jnp.sum(v) > 5.0),
                lambda v: assert_(jnp.sum(v) > -9.0 * n),
                x,
            )

        return h

    def relational_guard(n):
        def h():
            x = any_array((n,), "float64", (-1.0, 1.0))
            return jax.lax.cond(
                x[0] > x[1],
                lambda v: assert_(v[0] > 5.0),
                lambda v: assert_(v[0] > -9.0),
                x,
            )

        return h

    print(f"{'row':<24}{'n':>7}{'cap':>5}{'asked':>7}"
          f"{'first_idx':>16}{'lost':>6}")
    asked_total = lost_total = 0
    for n in (4, 64, 256, 1024, 4096, 8192, 16384):
        for sname, mk in (("corner_guard", corner_guard),
                          ("sum_guard", sum_guard),
                          ("relational_guard", relational_guard)):
            closed = trace(mk(n))
            walk = P._Propagator("constrain", "real")
            walk.run(closed.jaxpr, list(closed.consts), [])
            if walk.any_constrained or walk.assume_dropped:
                print(f"{sname + '_n' + str(n):<24}{n:>7}"
                      f"   -- search short-circuits")
                continue
            first = {}
            for k in range(P._PROBE_COUNT):
                probe = P._Propagator("constrain", "real")
                probe.pin = k
                try:
                    probe.run(closed.jaxpr, list(closed.consts), [])
                except Exception:  # noqa: BLE001
                    continue
                for key in probe.certain_reached:
                    first.setdefault(key, k)
            asked = {key for _i, key in walk.branch_violations}
            cap = P._certificate_probe_count(n)
            idxs = sorted(
                (first.get(key) if first.get(key) is not None else -1)
                for key in asked
            )
            lost = sum(
                1 for key in asked
                if first.get(key) is None or first[key] >= cap
            )
            asked_total += len(asked)
            lost_total += lost
            print(f"{sname + '_n' + str(n):<24}{n:>7}{cap:>5}"
                  f"{len(asked):>7}{str(idxs):>16}{lost:>6}")
    print(f"\nbranch-violation keys asked: {asked_total}; lost under a "
          f"_certificate_probe_count cap: {lost_total}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "count"
    {"count": count, "time": timing, "capcost": capcost}[what]()
