"""What the page's `both = z3 + cvc5` identity rules out, and what it cannot.

Two measurements, both about the page's `both` column.

1. **The identity is forced.** `solvers._dispatch_obligation` — reached from
   `solvers.escalate`, and cited here as `solvers._escalate` until 2026-08-23,
   which is a name this repository has never had — runs the admitted backends
   in a plain sequential loop with no short-circuit (`solvers.py:1997`, body
   1998-2046, no `break`), so a two-backend wall IS the two single-backend
   walls for any correct measurement. It cannot fail and it corroborates
   nothing.

2. **What it does rule out** is that the page timed the `check()` wall. This
   prints, per repeat, the published-latency sum against the wall, plus the
   number of solver stamps stelling actually INVOKED — which on a discharged
   row is twice the number of latencies it publishes, because the vacuity
   widen re-check runs the pipeline again and its notes are discarded.

3. **And the ordering can cost the whole wall guard.** Row 7's two-backend run
   prints its invocations in order: cvc5, the `QF_NRA` primary, burns its full
   wall guard before z3 is asked at all.

    JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src \
      SLOW=1 python probe-wall-and-invocation-order.py
"""
import os
import re
import sys
import time

sys.path.insert(0, str(
    __import__("pathlib").Path(__file__).resolve().parents[2] / "tools"))
import solver_battery as battery  # noqa: E402

battery._configure_jax()
from stelling.preconditions import check  # noqa: E402

ANSWERED = re.compile(
    r"^assert #(\d+): (.+?) answered (\S+) in (\d+)ms$")


def drive(build: str, n: int, label: str, repeats: int = 3,
          timeout_ms: int = 10_000) -> None:
    for r in range(repeats):
        t0 = time.monotonic()
        v = check(battery._harness(build, n), vacuity_mode="inputs-only",
                  solver_timeout_ms=timeout_ms)
        wall = int((time.monotonic() - t0) * 1000)
        inv = [(m[2], m[3], int(m[4]))
               for m in (ANSWERED.match(note) for note in v.notes) if m]
        total = sum(x[2] for x in inv)
        stamps = sum(1 for s in v.stamp.solver if s.invoked)
        region = sum(1 for note in v.notes if "admitted-region check —" in note)
        print(f"{label} r{r + 1}: status={v.status} "
              f"published-latency-sum={total} ms  check()-wall={wall} ms  "
              f"ratio={wall / total:.2f}x  "
              f"invoked-stamps={stamps} vs published={len(inv)}  "
              f"admitted-region-notes={region}")
        for i, (who, answer, ms) in enumerate(inv):
            print(f"    [{i}] {who:<14} {answer:<8} {ms:>7} ms")


def main() -> None:
    print("load average:", " ".join(f"{v:.2f}" for v in os.getloadavg()))
    print()
    drive("scalar_linear", 1, "row 1  VERIFIED  QF_LRA")
    drive("array8_linear_false", 3, "row 3  REFUTED   QF_LRA")
    drive("amgm", 4, "row 4  VERIFIED  QF_NRA", repeats=1)
    if os.environ.get("SLOW"):
        # ~20 s per repeat: cvc5's full wall guard, then z3's answer.
        drive("wide_products", 7, "row 7  VERIFIED  both", repeats=2)
    else:
        print("\n(set SLOW=1 for row 7's two-backend ordering — ~40 s)")


if __name__ == "__main__":
    main()
