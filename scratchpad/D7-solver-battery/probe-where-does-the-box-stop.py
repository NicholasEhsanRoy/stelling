"""The sweep behind ``reconstructed`` had an UNSTATED CEILING. This finds it,
and pushes past it to the one the FORMAT supplies.

``probe-does-the-freedom-reach-the-number.py`` swept rows 1-6 of
``docs/choosing-a-solver-backend.md`` and concluded that *the choice of reading
adds no more spread than re-running the same harness does*.  **Every row-4 and
row-5 reading in that sweep is capped at a box of +/-100**, while row 1 reaches
+/-1000.  Nothing in the sweep, on the page or in the tool states that cap —
and the labels supply no bound at all: AM-GM holds on the whole of R^2 and the
Motzkin polynomial is nonnegative on the whole of R^2.  So the conclusion was a
universal quantified over a box family that had been bounded silently, which is
the one shape of claim that cannot be repaired by widening the sweep: the next
box is always wider.

**There is exactly one non-arbitrary ceiling here, and it is not a judgement
call: the harness declares a float64 box.**  ``any_array(shape, jnp.float64,
(lo, hi))`` cannot be given an endpoint outside float64, so the widest reading
of any of these labels is the widest box the format holds.  This probe sweeps
each row's box from +/-1 to THAT ceiling — the largest scale at which every
constant the row's predicate builds from it is still finite — and reports where
the committed sweep's own statistic survives and where it does not.

    JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src python probe-where-does-the-box-stop.py

Five passes, INTERLEAVED — every (row, box, portfolio) cell is measured once in
pass 1, once in pass 2, ... — so a load drift during the run cannot manufacture
a trend across boxes.  The load average is printed at every pass boundary.

A reading counts only if it stays label-compatible: same published OUTCOME, and
still interval-UNDECIDED, since an obligation interval propagation decides never
reaches a backend and so cannot be a row of a solver comparison at all.  Cells
failing either test are printed and excluded, because excluding them silently is
how the first ceiling got in.
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
TIMEOUT_MS = 10_000
PASSES = 5

ANSWERED = re.compile(r"^assert #(\d+): (.+?) answered (\S+) in (\d+)ms$")

PUBLISHED_OUTCOME = {1: "unsat", 2: "unsat", 3: "sat",
                     4: "unsat", 5: "unsat", 6: "sat"}


# ---------------------------------------------------------------- harnesses
# Each is the committed sweep's own reading with ONLY the box scaled.  Where a
# constant has to move with the box to keep the row's own property (row 3's
# falseness, rows 1-2's true-but-interval-undecided bound), it moves with it,
# and that is what caps the row's ceiling below the format's: the CONSTANT has
# to stay finite too, not just the endpoint.

def r1(s):
    def h():
        x = any_array((), F, (-s, s))
        return assert_(2.0 * x - x >= -s)
    return h


def r2(s):
    def h():
        x = any_array((64,), F, (-s, s))
        return assert_(jnp.sum(2.0 * x - x) >= -64.0 * s)
    return h


def r3(s):
    # false at one corner: the sum of 8 values in [-s, s] reaches -8s and the
    # threshold sits just above it, so the predicate is false exactly there.
    def h():
        x = any_array((8,), F, (-s, s))
        return assert_(jnp.sum(2.0 * x - x) >= -7.0 * s)
    return h


def r4(s):
    def h():
        x = any_array((), F, (-s, s))
        y = any_array((), F, (-s, s))
        return assert_(x * x + y * y >= 2.0 * x * y)
    return h


def r5(s):
    def h():
        x = any_array((), F, (-s, s))
        y = any_array((), F, (-s, s))
        x2, y2 = x * x, y * y
        return assert_(x2 * x2 * y2 + x2 * y2 * y2 - 3.0 * x2 * y2 + 1.0 >= 0.0)
    return h


def r6(s):
    def h():
        x = any_array((), F, (-s, s))
        return assert_(x * x * x >= 0.0)
    return h


FMAX = sys.float_info.max  # 1.7976931348623157e308

#: (builder, widest declarable scale, why that scale and not FMAX).
#: The rule is one line long and is the whole justification for the ceiling:
#: every constant the predicate builds from the box has to be finite too.
ROWS = {
    1: (r1, 8.9e307, "2*s must be finite"),
    2: (r2, 1e306, "the 64*s threshold must be finite"),
    3: (r3, 1e307, "the 7*s threshold must be finite"),
    4: (r4, FMAX, "no constant scales with the box"),
    5: (r5, FMAX, "no constant scales with the box"),
    6: (r6, FMAX, "no constant scales with the box"),
}

LADDER = [1.0, 100.0, 1e20, 1e60, 1e100, 1e150, 1e300]

PORTFOLIOS = [("z3", "z3"), ("cvc5", "cvc5")]

#: The committed sweep's own, unstated, ceiling for the two named-object rows.
COMMITTED_CEILING = 100.0


def scales_for(n):
    _, top, _ = ROWS[n]
    return [s for s in LADDER if s <= top] + [top]


def one(h, only):
    try:
        v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT_MS,
                  solver=only)
    except Exception as e:  # noqa: BLE001
        return None, f"RAISED {type(e).__name__}: {e}"[:90], 0
    total = n_inv = 0
    for note in v.notes:
        m = ANSWERED.match(note)
        if m:
            total += int(m[4])
            n_inv += 1
    return total, {"VERIFIED": "unsat", "REFUTED": "sat"}.get(v.status, "UNKNOWN"), n_inv


def ab(data, n, pname, ceiling):
    allv, worst_b = [], 1.0
    for s in scales_for(n):
        if s > ceiling:
            continue
        v = data.get((n, pname, s))
        if not v:
            continue
        allv.extend(v)
        if min(v) > 0:
            worst_b = max(worst_b, max(v) / min(v))
    if not allv:
        return None
    lo, hi = min(allv), max(allv)
    a = hi / lo if lo else float("inf")
    return lo, hi, a, worst_b, a / worst_b


def main():
    want = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]
    print(f"rows {want}  passes {PASSES}  timeout {TIMEOUT_MS} ms")
    print(f"float64 max = {FMAX!r}")
    print("load average at start:", " ".join(f"{v:.2f}" for v in os.getloadavg()))
    print()
    data: dict = {}
    bad: list = []
    for p in range(PASSES):
        la = " ".join(f"{v:.2f}" for v in os.getloadavg())
        print(f"--- pass {p + 1}/{PASSES}   load {la}")
        sys.stdout.flush()
        for n in want:
            build = ROWS[n][0]
            for s in scales_for(n):
                h = build(s)
                for pname, only in PORTFOLIOS:
                    ms, outcome, n_inv = one(h, only)
                    if ms is None:
                        bad.append((n, s, pname, outcome))
                    elif n_inv == 0:
                        bad.append((n, s, pname,
                                    f"INTERVAL-DECIDED ({outcome}), 0 invocations"))
                    elif outcome != PUBLISHED_OUTCOME[n]:
                        bad.append((n, s, pname, f"outcome {outcome} != published "
                                                 f"{PUBLISHED_OUTCOME[n]}"))
                    else:
                        data.setdefault((n, pname, s), []).append(ms)
    print()
    print("load average at end  :", " ".join(f"{v:.2f}" for v in os.getloadavg()))
    print()

    print("THE CEILING EACH ROW'S OWN READING HAS, AND WHY IT IS NOT A CHOICE")
    for n in want:
        _, top, why = ROWS[n]
        print(f"  row {n}: widest declarable box +/-{top:g}   ({why})")
    print(f"  the committed sweep stopped at +/-{COMMITTED_CEILING:g} on rows 4 and 5,")
    print("  and +/-1000 on row 1, and said so nowhere.")
    print()

    if bad:
        print("EXCLUDED — not label-compatible at that box:")
        for n, s, pname, why in bad:
            print(f"  row {n}  box +/-{s:g}  {pname:<5} {why}")
        print()

    print(f"PER-PASS NOTES-SUM MILLISECONDS ({PASSES} interleaved passes)")
    allscales = sorted({s for n in want for s in scales_for(n)})
    hdr = f"{'row':>3} {'pf':<5} " + " ".join(f"{('+/-%g' % s):>17}" for s in allscales)
    print(hdr)
    print("-" * len(hdr))
    for n in want:
        for pname, _ in PORTFOLIOS:
            cells = []
            for s in allscales:
                v = data.get((n, pname, s))
                cells.append("/".join(str(x) for x in v) if v else ".")
            print(f"{n:>3} {pname:<5} " + " ".join(f"{c:>17}" for c in cells))
    print()

    print("A/B — THE COMMITTED SWEEP'S OWN STATISTIC — AT THREE CEILINGS")
    print()
    print("  A = max/min over every reading AND every repeat of that (row,")
    print("      portfolio) pair.  B = the widest max/min INSIDE one unchanged")
    print("      cell (one box, one portfolio, every pass).  The committed sweep")
    print("      reports A/B <= 1.22x and calls that `the choice of reading adds")
    print("      no more spread than re-running the same harness does`.")
    print("  NOTE A >= B BY CONSTRUCTION: B is a max over cells of a within-cell")
    print("      ratio and A is that ratio over the pooled samples, so the union")
    print("      contains every cell.  A/B >= 1 always, and the statistic is")
    print("      biased toward 1 in the direction that flatters the claim.")
    print()
    for ceiling, label in (
        (COMMITTED_CEILING, "boxes <= +/-100 — the committed sweep's own, unstated ceiling"),
        (1e100, "boxes <= +/-1e100 — well inside float64"),
        (float("inf"), "every declarable box — the ceiling the FORMAT supplies"),
    ):
        print(f"  {label}")
        print(f"    {'row':>3} {'pf':<5} {'A range':<20} {'A':>8} {'B':>8} {'A/B':>8}")
        for n in want:
            for pname, _ in PORTFOLIOS:
                r = ab(data, n, pname, ceiling)
                if not r:
                    continue
                lo, hi, a, b, e = r
                flag = "  <- past 1.22x" if e > 1.22 else ""
                print(f"    {n:>3} {pname:<5} {f'{lo}-{hi} ms':<20} "
                      f"{a:>7.2f}x {b:>7.2f}x {e:>7.2f}x{flag}")
        print()

    print("IS THERE A z3 FLOOR? (the other half of the argument the grade rests on)")
    print("  the page says rows 1-6's cells `sit on two floors this page itself")
    print("  names: the cvc5 spawn (~70 ms) and z3's ~10 ms`, and that `a cell")
    print("  that IS the floor cannot be moved by a choice made above it`.")
    print()
    for pname, _ in PORTFOLIOS:
        allv = [v for (n, p, s), vs in data.items() if p == pname for v in vs]
        narrow = [v for (n, p, s), vs in data.items()
                  if p == pname and s <= COMMITTED_CEILING for v in vs]
        if allv:
            print(f"  {pname:<5} over every row and every declarable box: "
                  f"{min(allv)}-{max(allv)} ms   "
                  f"(at boxes <= +/-100: {min(narrow)}-{max(narrow)} ms)")


if __name__ == "__main__":
    sys.exit(main())
