"""Does the freedom a row LABEL leaves actually reach the PUBLISHED NUMBER?

That is the question that decides whether a row of
``docs/choosing-a-solver-backend.md``'s ten-row table is reconstructible —
not the binary "does the label pin a harness", which no label on that page
passes and which therefore grades every row the same.

This sweeps the free parameters of rows 1-6 across every label-compatible
reading that stays TRUE (or, for the two false rows, false) and stays
interval-UNDECIDED, three repeats each, three portfolios each, and prints the
spread against the page's published cell.

    JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src python probe-does-the-freedom-reach-the-number.py

Rows 4 and 5 are the sharp ones: their labels NAME a mathematical object, so
the predicate is not free at all and the only open parameter is the box (plus,
for Motzkin, the association of the degree-6 monomials, which is the same
polynomial and not the same emitted script).
"""
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402

F = jnp.float64
TIMEOUT_MS = 10_000
REPEATS = 3

PUBLISHED = {  # the page's ten-row table, rows 1-6
    1: ("unsat, 78-112 ms", "unsat, 8-9 ms", "unsat, 71-84 ms"),
    2: ("unsat, 86-91 ms", "unsat, 10-12 ms", "unsat, 77-87 ms"),
    3: ("sat, 86-90 ms", "sat, 11-13 ms", "sat, 75-117 ms"),
    4: ("unsat, 80-83 ms", "unsat, 9 ms", "unsat, 75-87 ms"),
    5: ("unsat, 92-106 ms", "unsat, 12-13 ms", "unsat, 81-83 ms"),
    6: ("sat, 87-88 ms", "sat, 11 ms", "sat, 69-71 ms"),
}


# ----------------------------------------------------------- row 1, scalar
def r1(a, b):
    def h():
        x = any_array((), F, (a, b))
        return assert_(2.0 * x - x >= a)
    return h


# ------------------------------------------------------ row 2, 64 elements
def r2_reduced(a, b):
    def h():
        x = any_array((64,), F, (a, b))
        return assert_(jnp.sum(2.0 * x - x) >= 64.0 * a)
    return h


def r2_elementwise(a, b):
    def h():
        x = any_array((64,), F, (a, b))
        return assert_(2.0 * x - x >= a)
    return h


# ------------------------------------------- row 3, 8 elements, false
def r3(a, b, thresh):
    def h():
        x = any_array((8,), F, (a, b))
        return assert_(jnp.sum(2.0 * x - x) >= thresh)
    return h


# ------------------------------------------------------------ row 4, AM-GM
def r4(lo, hi):
    def h():
        x = any_array((), F, (lo, hi))
        y = any_array((), F, (lo, hi))
        return assert_(x * x + y * y >= 2.0 * x * y)
    return h


# ---------------------------------------------------------- row 5, Motzkin
def r5_assoc_a(lo, hi):
    """(x2*x2)*y2 + (x2*y2)*y2 - 3 x2 y2 + 1 — the battery's association."""
    def h():
        x = any_array((), F, (lo, hi))
        y = any_array((), F, (lo, hi))
        x2, y2 = x * x, y * y
        return assert_(x2 * x2 * y2 + x2 * y2 * y2 - 3.0 * x2 * y2 + 1.0 >= 0.0)
    return h


def r5_assoc_b(lo, hi):
    """x2*(x2*y2) + y2*(x2*y2) - 3 x2 y2 + 1 — the other association."""
    def h():
        x = any_array((), F, (lo, hi))
        y = any_array((), F, (lo, hi))
        x2, y2 = x * x, y * y
        x2y2 = x2 * y2
        return assert_(x2 * x2y2 + y2 * x2y2 - 3.0 * x2y2 + 1.0 >= 0.0)
    return h


def r5_rearranged(lo, hi):
    """x2*y2*(x2 + y2 - 3) + 1 — the same polynomial, factored."""
    def h():
        x = any_array((), F, (lo, hi))
        y = any_array((), F, (lo, hi))
        x2, y2 = x * x, y * y
        return assert_(x2 * y2 * (x2 + y2 - 3.0) + 1.0 >= 0.0)
    return h


# --------------------------------------------------- row 6, cubic, false
def r6(cubic, lo, hi):
    def h():
        x = any_array((), F, (lo, hi))
        return assert_(cubic(x) >= 0.0)
    return h


READINGS = [
    (1, "box [1,2]", r1(1.0, 2.0)),
    (1, "box [0,1]", r1(0.0, 1.0)),
    (1, "box [-10,10]", r1(-10.0, 10.0)),
    (1, "box [-1000,1000]", r1(-1000.0, 1000.0)),
    (1, "box [1,100]", r1(1.0, 100.0)),
    (2, "reduced, box [1,2]", r2_reduced(1.0, 2.0)),
    (2, "reduced, box [-10,10]", r2_reduced(-10.0, 10.0)),
    (2, "elementwise, box [1,2]", r2_elementwise(1.0, 2.0)),
    (2, "elementwise, box [-10,10]", r2_elementwise(-10.0, 10.0)),
    (3, "false at one corner (>= 9)", r3(1.0, 2.0, 9.0)),
    (3, "false about half the time (>= 12)", r3(1.0, 2.0, 12.0)),
    (3, "false except at one corner (>= 16)", r3(1.0, 2.0, 16.0)),
    (4, "box [0,1]^2", r4(0.0, 1.0)),
    (4, "box [-1,1]^2", r4(-1.0, 1.0)),
    (4, "box [-2,2]^2", r4(-2.0, 2.0)),
    (4, "box [-10,10]^2", r4(-10.0, 10.0)),
    (4, "box [-100,100]^2", r4(-100.0, 100.0)),
    (4, "box [1,2]^2", r4(1.0, 2.0)),
    (5, "assoc A, box [-1,1]^2", r5_assoc_a(-1.0, 1.0)),
    (5, "assoc A, box [-2,2]^2", r5_assoc_a(-2.0, 2.0)),
    (5, "assoc A, box [-3,3]^2", r5_assoc_a(-3.0, 3.0)),
    (5, "assoc A, box [-10,10]^2", r5_assoc_a(-10.0, 10.0)),
    (5, "assoc A, box [-100,100]^2", r5_assoc_a(-100.0, 100.0)),
    (5, "assoc B, box [-1,1]^2", r5_assoc_b(-1.0, 1.0)),
    (5, "assoc B, box [-2,2]^2", r5_assoc_b(-2.0, 2.0)),
    (5, "assoc B, box [-10,10]^2", r5_assoc_b(-10.0, 10.0)),
    (5, "assoc B, box [-100,100]^2", r5_assoc_b(-100.0, 100.0)),
    (5, "rearranged, box [-2,2]^2", r5_rearranged(-2.0, 2.0)),
    (5, "rearranged, box [-100,100]^2", r5_rearranged(-100.0, 100.0)),
    (6, "x^3, box [-2,2]", r6(lambda x: x * x * x, -2.0, 2.0)),
    (6, "x^3, box [-1,1]", r6(lambda x: x * x * x, -1.0, 1.0)),
    (6, "x^3, box [-100,100]", r6(lambda x: x * x * x, -100.0, 100.0)),
    (6, "x^3 - x, box [-2,2]", r6(lambda x: x * x * x - x, -2.0, 2.0)),
    (6, "x^3 + x^2 - 2x, box [-2,2]",
     r6(lambda x: x * x * x + x * x - 2.0 * x, -2.0, 2.0)),
]

ANSWERED = __import__("re").compile(
    r"^assert #(\d+): (.+?) answered (\S+) in (\d+)ms$")


def drive(h, only):
    kw = {} if only is None else {"solver": only}
    sums, outs = [], []
    for _ in range(REPEATS):
        try:
            v = check(h, vacuity_mode="inputs-only",
                      solver_timeout_ms=TIMEOUT_MS, **kw)
        except Exception as e:  # noqa: BLE001
            return f"RAISED {type(e).__name__}", [], f"{e}"[:120]
        total = 0
        for note in v.notes:
            m = ANSWERED.match(note)
            if m:
                total += int(m[4])
        outs.append({"VERIFIED": "unsat", "REFUTED": "sat"}.get(v.status, "UNKNOWN"))
        sums.append(total)
    outcome = "/".join(sorted(set(outs)))
    return outcome, sums, ""


def main():
    import os
    print("load average:", " ".join(f"{v:.2f}" for v in os.getloadavg()))
    print()
    hdr = f"{'row':>3}  {'reading':<36} {'both':<20} {'z3 alone':<20} {'cvc5 alone':<20}"
    print(hdr)
    print("-" * len(hdr))
    seen = set()
    # per (row, portfolio): every repeat's sum, across every reading; and the
    # worst WITHIN-cell repeat spread, which is the number the whole argument
    # turns on.
    spread: dict = {}
    outcomes: dict = {}
    for n, label, h in READINGS:
        if n not in seen:
            if seen:
                print()
            print(f"     PAGE ROW {n}: both {PUBLISHED[n][0]} | z3 {PUBLISHED[n][1]} "
                  f"| cvc5 {PUBLISHED[n][2]}")
            seen.add(n)
        cells = []
        for which, only in (("both", None), ("z3", "z3"), ("cvc5", "cvc5")):
            outcome, sums, why = drive(h, only)
            if sums:
                cells.append(f"{outcome}, {min(sums)}-{max(sums)} ms")
                key = (n, which)
                spread.setdefault(key, {"all": [], "worst_repeat": 1.0,
                                        "worst_at": ""})
                spread[key]["all"].extend(sums)
                if min(sums) > 0:
                    r = max(sums) / min(sums)
                    if r > spread[key]["worst_repeat"]:
                        spread[key]["worst_repeat"] = r
                        spread[key]["worst_at"] = label
                outcomes.setdefault((n, which), set()).add(outcome)
            else:
                cells.append(f"{outcome} {why}"[:20])
        print(f"{n:>3}  {label:<36} {cells[0]:<20} {cells[1]:<20} {cells[2]:<20}")
    print()
    print("Every cell above is one machine's, at the load printed at the top.")
    print()
    print("DOES THE FREEDOM REACH THE NUMBER?")
    print()
    print("  `across readings` is min..max over every label-compatible reading")
    print("  AND every repeat.  `worst one cell` is the widest min..max INSIDE")
    print("  a single (reading, portfolio) cell -- one harness, one portfolio,")
    print("  three repeats, nothing chosen differently between them.  If the")
    print("  second is not smaller than the first, the choice is not what")
    print("  moved the number.")
    print()
    h2 = (f"{'row':>3}  {'portfolio':<10} {'across readings':<20} "
          f"{'A':>6}  {'worst 1 cell: B':>15}  {'A/B':>6}  outcome")
    print(h2)
    print("-" * len(h2))
    worst_excess = (0.0, None)
    equal = 0
    for n in sorted({k[0] for k in spread}):
        for which in ("both", "z3", "cvc5"):
            d = spread.get((n, which))
            if not d:
                continue
            lo, hi = min(d["all"]), max(d["all"])
            a = hi / lo if lo else float("inf")
            b = d["worst_repeat"]
            excess = a / b if b else float("inf")
            equal += abs(excess - 1.0) < 0.005
            if excess > worst_excess[0]:
                worst_excess = (excess, (n, which, lo, hi, a, b))
            outs = "/".join(sorted(outcomes[(n, which)]))
            print(f"{n:>3}  {which:<10} {f'{lo}-{hi} ms':<20} {a:>5.2f}x  "
                  f"{b:>14.2f}x  {excess:>5.2f}x  {outs}"
                  + (f"   [{d['worst_at']}]" if d["worst_at"] else ""))
    print()
    print("  A is the spread over EVERY reading and every repeat; B is the")
    print("  widest spread inside ONE unchanged (reading, portfolio) cell.")
    print("  A pools three to eighteen times as many samples as B, so A is")
    print("  biased WIDER; A/B near 1 therefore says the reading contributed")
    print("  nothing the clock did not.")
    ex, where = worst_excess
    if where:
        n, which, lo, hi, a, b = where
        print(f"  A/B is exactly 1.00x in {equal} of the 18 pairs, and never")
        print(f"  exceeds {ex:.2f}x — worst at row {n} [{which}], {lo}-{hi} ms,")
        print(f"  where one millisecond of jitter is a {100 / lo:.0f}% move.")


if __name__ == "__main__":
    sys.exit(main())
