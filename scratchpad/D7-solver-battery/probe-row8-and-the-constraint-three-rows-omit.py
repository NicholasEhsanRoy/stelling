"""Two things the fixup round had to reproduce before repairing.

**(A) ROW 8'S GRADE DOES NOT MEET ITS OWN STATED CRITERION.**
``GRADES[GRADE_UNSUPPORTED]`` says *"...and at least one runs the published
direction BACKWARDS"*.  Row 7 satisfies that literally: the published row has
z3 timing out and cvc5 answering, and the literal reading has z3 answering and
cvc5 timing out — the same pattern with the backends swapped.  **Row 8's three
readings are driven here to find out whether any of them does the same.**

**(B) THE CONSTRAINT THREE ROWS DO NOT RECORD.**  The page asserts that *"every
row's `chose here` list records"* that the obligation has to be interval-
UNDECIDED or it never reaches a backend.  Rows 4, 5 and 6 do not record it —
and they are reconstruction rows.  Two label-compatible readings that the
constraint excludes are driven here to show the omission is load-bearing rather
than cosmetic: AM-GM over PER-VARIABLE boxes (nothing in *"2 vars"* says the two
share one box) and Motzkin over a square box in the sweep's own family.

    JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src python probe-row8-and-the-constraint-three-rows-omit.py
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
REPEATS = 2

ANSWERED = re.compile(r"^assert #(\d+): (.+?) answered (\S+) in (\d+)ms$")


# ---------------------------------------------------- (A) row 8's readings
# The tool's own three builders at row 8's width, verbatim in shape.

def r8_literal():
    a = any_array((32,), F, (0.0, 1.0))
    b = any_array((32,), F, (0.0, 1.0))
    return assert_(jnp.sum(a * b) <= jnp.sum(a))


def r8_sum_of_squares():
    a = any_array((32,), F, (-1.0, 1.0))
    b = any_array((32,), F, (-1.0, 1.0))
    return assert_(jnp.sum(a * a + b * b - 2.0 * a * b) >= 0.0)


def r8_cancellation():
    a = any_array((32,), F, (-1.0, 1.0))
    b = any_array((32,), F, (-1.0, 1.0))
    return assert_(jnp.sum(a * b) - jnp.sum(b * a) >= 0.0)


# ------------------------------------- (B) readings the constraint excludes

def amgm_per_variable():
    """`2 vars, degree 2 (AM-GM)` with a box PER VARIABLE, not one shared box.

    Nothing in the label says the two variables share a box, and AM-GM holds
    over the whole of R^2, so this is as label-compatible as any box in the
    committed sweep."""
    x = any_array((), F, (0.0, 0.1))
    y = any_array((), F, (10.0, 20.0))
    return assert_(x * x + y * y >= 2.0 * x * y)


def motzkin_tiny_box():
    """Motzkin over `[1e-300, 1e-299]^2` — a square box in the sweep's own
    family, just a small one."""
    x = any_array((), F, (1e-300, 1e-299))
    y = any_array((), F, (1e-300, 1e-299))
    x2, y2 = x * x, y * y
    return assert_(x2 * x2 * y2 + x2 * y2 * y2 - 3.0 * x2 * y2 + 1.0 >= 0.0)


def drive(h, only):
    outs, mss, invs = [], [], []
    for _ in range(REPEATS):
        try:
            v = check(h, vacuity_mode="inputs-only",
                      solver_timeout_ms=TIMEOUT_MS, solver=only)
        except Exception as e:  # noqa: BLE001
            return f"RAISED {type(e).__name__}", [], []
        total = n = 0
        for note in v.notes:
            m = ANSWERED.match(note)
            if m:
                total += int(m[4])
                n += 1
        outs.append({"VERIFIED": "unsat", "REFUTED": "sat"}.get(v.status, "UNKNOWN"))
        mss.append(total)
        invs.append(n)
    return "/".join(sorted(set(outs))), mss, invs


def cell(h, only):
    o, ms, inv = drive(h, only)
    if not ms:
        return f"{o}"
    return f"{o}, {min(ms)}-{max(ms)} ms  [{min(inv)}-{max(inv)} inv]"


def main():
    print("load average:", " ".join(f"{v:.2f}" for v in os.getloadavg()))
    print(f"repeats {REPEATS}   timeout {TIMEOUT_MS} ms")
    print()
    print("(A) ROW 8'S THREE READINGS — does ANY of them reverse the published row?")
    print()
    print("    the page's row 8, as published:  z3 **UNKNOWN** (timeout) | "
          "cvc5 unsat, 772-792 ms")
    print("    so the published DIRECTION is: cvc5 finishes, z3 does not.")
    print("    a REVERSAL is that pattern with the two backends swapped.")
    print()
    hdr = f"    {'reading':<26} {'z3 alone':<34} {'cvc5 alone':<34}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for label, h in (("literal: sum(a*b)<=sum(a)", r8_literal),
                     ("sum-of-squares", r8_sum_of_squares),
                     ("cancellation", r8_cancellation)):
        z = cell(h, "z3")
        c = cell(h, "cvc5")
        print(f"    {label:<26} {z:<34} {c:<34}")
        sys.stdout.flush()
    print()
    print("(B) TWO READINGS THE UNSTATED CONSTRAINT EXCLUDES — and rows 4, 5, 6")
    print("    do not record it.  `inv` is the number of solver invocations:")
    print("    ZERO means interval propagation decided it and no backend was")
    print("    ever asked, so it cannot be a row of a solver comparison.")
    print()
    for label, h in (("AM-GM, per-variable boxes x in [0,0.1], y in [10,20]",
                      amgm_per_variable),
                     ("Motzkin over [1e-300, 1e-299]^2", motzkin_tiny_box)):
        print(f"    {label}")
        for pname in ("z3", "cvc5"):
            print(f"      {pname:<5} {cell(h, pname)}")
        sys.stdout.flush()
    print()
    print("load average at end:", " ".join(f"{v:.2f}" for v in os.getloadavg()))


if __name__ == "__main__":
    sys.exit(main())
