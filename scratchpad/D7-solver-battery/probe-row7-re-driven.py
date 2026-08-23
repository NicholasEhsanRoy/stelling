"""Row 7's reversal, re-driven: is it the box, the bound, or the budget?

The finding is that the most literal reading of *"32 vars, 16 elementwise
products"* runs the page's direction BACKWARDS — z3 discharges, cvc5 hits its
wall guard. A reversal that only happened at one box, or under one of the two
sums, or at one budget, would be a property of this harness rather than of the
label. So each is varied here, one at a time.

Also driven: two further label-compatible spellings that never reach a backend
at all, which is the constraint the page never states and every row of this
battery has to satisfy (`INTERVAL_UNDECIDED`).

    JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src \
      python probe-row7-re-driven.py            # ~4 minutes
"""
import os
import re
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402

F = jnp.float64
N = 16
ANSWERED = re.compile(r"^assert #(\d+): (.+?) answered (\S+) in (\d+)ms$")


def literal(lo=0.0, hi=1.0, bound="a"):
    """sum(a*b) <= sum(a)  — or <= sum(b), which is equally literal."""
    def h():
        a = any_array((N,), F, (lo, hi))
        b = any_array((N,), F, (lo, hi))
        other = a if bound == "a" else b
        return assert_(jnp.sum(a * b) <= jnp.sum(other))
    return h


def sum_against_a_constant():
    """sum(a*b) <= 16 over [0,1]^16 — interval-EXACT, so nothing escalates."""
    def h():
        a = any_array((N,), F, (0.0, 1.0))
        b = any_array((N,), F, (0.0, 1.0))
        return assert_(jnp.sum(a * b) <= float(N))
    return h


def independent_products():
    """a*b >= -1 elementwise over [0,1]^16 — same, for the same reason."""
    def h():
        a = any_array((N,), F, (0.0, 1.0))
        b = any_array((N,), F, (0.0, 1.0))
        return assert_(a * b >= -1.0)
    return h


def drive(label, h, only, timeout_ms):
    kw = {} if only is None else {"solver": only}
    try:
        v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=timeout_ms, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"{label:<44} {str(only):<5} RAISED {type(e).__name__}: {e}"[:150])
        return
    inv = [(m[2], m[3], int(m[4]))
           for m in (ANSWERED.match(n) for n in v.notes) if m]
    outcome = {"VERIFIED": "unsat", "REFUTED": "sat"}.get(v.status, "UNKNOWN")
    if not inv:
        detail = v.obligations[0].detail if v.obligations else "no obligation"
        print(f"{label:<44} {str(only):<5} NO BACKEND WAS ASKED — {detail[:70]}")
        return
    shown = "  ".join(f"{who.split()[0]} {ans} {ms / 1000:.1f}s"
                      for who, ans, ms in inv)
    print(f"{label:<44} {str(only):<5} {outcome:<8} {shown}")


def main():
    print("load average:", " ".join(f"{v:.2f}" for v in os.getloadavg()))
    print()
    print("ROBUST TO THE BOX (10 s budget)")
    for lo, hi in ((0.0, 1.0), (0.0, 2.0), (0.0, 0.5)):
        for only in ("z3", "cvc5"):
            drive(f"sum(a*b) <= sum(a) over [{lo},{hi}]^16", literal(lo, hi),
                  only, 10_000)
    print()
    print("ROBUST TO WHICH SUM IS THE BOUND (10 s budget)")
    for only in ("z3", "cvc5"):
        drive("sum(a*b) <= sum(b) over [0,1]^16", literal(bound="b"),
              only, 10_000)
    print()
    print("ROBUST TO THE BUDGET: 60 s, six times the page's (this is slow)")
    for only in ("z3", "cvc5"):
        drive("sum(a*b) <= sum(a) over [0,1]^16 @60s", literal(),
              only, 60_000)
    print()
    print("AND TWO SPELLINGS THAT NEVER REACH A BACKEND AT ALL")
    print("(the constraint the page never states: an obligation interval")
    print(" propagation DECIDES cannot be a row of a solver comparison)")
    drive("sum(a*b) <= 16 over [0,1]^16", sum_against_a_constant(), None, 10_000)
    drive("a*b >= -1 elementwise over [0,1]^16", independent_products(),
          None, 10_000)


if __name__ == "__main__":
    sys.exit(main())
