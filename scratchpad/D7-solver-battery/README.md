# D7 — solver battery: raw evidence, and the reconstruction state at handover

Rescued from a scratch directory at a forced stop on 2026-08-22. Nothing here
ships (`/scratchpad` is absent from the sdist allowlist).

## What the files are

| file | what |
|---|---|
| `battery-run-1-2026-08-22T1949Z.txt` | first full `tools/solver_battery.py --variants --repeats 3`, load average 5.67 |
| `battery-run-2-2026-08-22T1959Z.txt` | second full run, load average 3.99. **This is the run the page's re-measured table was copied from.** Every direction agrees with run 1; the seconds move by up to 10% between them |
| `lane-jax-solvers-partial.txt` | whole-suite run in `stelling-jax`, KILLED at ~56% by the shutdown. Green to that point |
| `controls-nojax.sh` / `controls-jax.sh` | the positive controls. The nojax five were driven and all five fired; the jax six were written and NOT driven |
| `nosolvers-plugin.py` | `pytest -p nosolvers`, which hides both wheels from `stelling._optional.available` — the page's own documented method — to simulate the `test-jax-no-solvers` CI lane. NOT driven |
| `probe-*.py` | the harness-space probes that produced the reconstruction finding |

## WHICH ROWS WERE RECONSTRUCTED — the mandate's binding constraint

**None of the ten is reconstructible in the sense of pinning a harness.** Not
one row label on that page fixes a declared box, a predicate or a threshold,
and the page never states the constraint every row must satisfy (an obligation
interval propagation DECIDES never reaches a backend at all). Every harness in
`tools/solver_battery.py` is therefore DECLARED, and every row carries the list
of parameters it had to choose in `Row.chosen`, which a test refuses to let be
empty.

Within that, the rows separate into three grades. **Whoever resumes must not
promote a row between grades without measuring.**

### Grade A — the mathematical object is named, only the box is free (rows 4, 5)

- **row 4, `2 vars, degree 2 (AM–GM)`** — the two-variable degree-2 form of
  AM–GM is `x² + y² >= 2xy`. Named, not chosen. Box chosen: `[-1,1]²`.
- **row 5, `2 vars, degree 6 (Motzkin)`** — the Motzkin polynomial
  `x⁴y² + x²y⁴ − 3x²y² + 1` is a named object. Box chosen: `[-2,2]²`. The
  ASSOCIATION of the degree-6 monomials is still free and is not the same
  emitted script.

### Grade B — shape fixed, everything else chosen; measured direction AGREED with the page (rows 1, 2, 3, 6, 9, 10)

Answers and directions reproduced. Seconds are this machine's.

- row 1 `scalar, linear` — chose `x ∈ [1,2]`, `2x − x >= 1`.
- row 2 `64-element array, linear` — chose `[1,2]⁶⁴`, `sum(2x − x) >= 64`.
- row 3 `8-element array, linear, false` — chose `[1,2]⁸`, `sum(2x − x) >= 9`.
- row 6 `1 var, degree 3, false` — chose `x³ >= 0` over `[-2,2]`.
- row 9 `10-factor product chain` — chose `(x0·…·x9)² >= 0` over `[-1,1]¹⁰`.
- row 10 `12-factor product chain` — same at twelve.

Rows 9 and 10 reproduced the page's DIRECTION (z3 fast, cvc5 slow then
timing out). **But an equally defensible reading — one variable to the tenth,
`x¹⁰ >= 0` — shows no split at all (z3 6–7 ms, cvc5 71–73 ms).** The direction
survived one reading and vanished under another. It did not reverse under
either, which is why these two carry no `‡`.

### Grade C — NOT RECONSTRUCTIBLE, and the page's direction did not reproduce (rows 7, 8)

**This is the finding of the batch.** Three defensible readings of `32 vars,
16 elementwise products`, three repeats each, measured here:

| reading | z3 alone | cvc5 alone |
|---|---|---|
| `sum(a*b) <= sum(a)`, `a,b ∈ [0,1]¹⁶` — the LITERAL one | unsat, 4.4–4.6 s | **UNKNOWN** (16.0 s wall) |
| `sum(a² + b² − 2ab) >= 0`, `a,b ∈ [−1,1]¹⁶` | unsat, 22 ms | unsat, 169–182 ms |
| `sum(a*b) − sum(b*a) >= 0`, `a,b ∈ [−1,1]¹⁶` | unsat, 6–9 ms | unsat, 69–81 ms |
| **the page's published row** | **UNKNOWN** (timeout) | unsat, 166–175 ms |

The literal reading **reverses** the page. The second reproduces the page's
cvc5 cell to within a few milliseconds and still does not reproduce its z3
cell — matching one cell is not identifying a harness. The third shows no
split. At row 8's width (32 products) the literal reading has NEITHER backend
finishing.

**Do not "fix" rows 7 and 8 by searching harness space until something
reproduces the page.** That is fitting to a conclusion. The correct output is
the refusal that is already in the tool: `Row.contested`, the `‡` marks, and
`direction_report`'s `FINDING 2 … NOT DECIDABLE FROM THIS BATTERY`.

## Two things recovered rather than chosen

1. **What the page timed.** Its `both` cells are the SUM of its own two
   single-backend cells. Row 10: 16.0 s + 689–702 ms = 16.69–16.70 s against a
   published `~16.7 s`. Row 7: 10.0–10.1 s + 166–175 ms = 10.2–10.3 s against
   `~10.3 s`. Row 8: same timeout + 772–792 ms = 10.8–10.9 s against `~11.0 s`.
   Row 9 does not fit, by 0.3–0.5 s.
2. **`solver_timeout_ms=10000` is not a ten-second wall.** `solvers._wall_seconds`
   is `timeout*1.5 + 1`, so the wall-guarded cvc5 child is killed at **16.0 s**
   — measured on rows 7, 8 and 10. A two-backend row where both time out costs
   **26 s**, which is what row 8 cost. Nothing on the page said this.
