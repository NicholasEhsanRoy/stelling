<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Choosing a solver backend: z3, cvc5, or both

**Install both (`[solvers]`).** Not because one is unreliable, but because
the portfolio's whole design is that two independent backends answer the
same question and agreement decides. Installing one does not make verdicts
weaker in what they *claim* — a one-backend `VERIFIED` is still a
`VERIFIED` — it removes the only independent check a discharge has, and
stelling says so in the verdict rather than letting you find out later.

If you can only install one, this page says what that costs, measured. It
is not a recommendation derived from the backends' reputations; every
figure below came out of this tree.

**How this page was measured, and what that does not cover.** All figures
are from `stelling 0.1.0`, `jax 0.11.0`, CPU, `jax_enable_x64=True`, z3
`5.0.0` (wheel) and cvc5 `1.3.4-modified` (wheel), Linux x86-64, Python
3.12, `solver_timeout_ms=10000`, three repeats per cell. Single-backend
configurations were produced with `SolverConfig(only=…)` and by hiding a
backend from `stelling._optional.available` — nothing was uninstalled, and
both routes produce the same degraded-portfolio disclosure (below). Wall
times are one machine's; the *decided / timed-out* column is the part that
carries the argument, and even that is a timeout budget away from moving.
Unlike the code blocks in [quickstart](quickstart.md) and
[harness API](harness-api.md), **nothing on this page is re-run by
`tests/test_doc_examples.py`** — it is a hand-check, and it will go stale
the way hand-checks do.

## How an obligation reaches a backend

Only obligations that interval propagation (and, if you asked for it, the
affine refinement) left **undecided** are escalated at all, and only when
you pass `solver_timeout_ms`. Each escalated obligation is classified into
one of exactly two fragments by `stelling.obligation._Slicer._fragment`:

| fragment | when | primary | secondary |
|---|---|---|---|
| `QF_LRA` | every operation on declaration-dependent values is linear, and no `pow` has a non-integer exponent | **z3** | cvc5 |
| `QF_NRA` | some `mul` of two dependent operands, `div` by a dependent operand, `square`, `integer_pow` with exponent ∉ {0, 1}, `pow` with a *dependent* base at an exponent ∉ {0, 1}, **or any `pow` with a non-integer exponent** | **cvc5** | z3 |

The last clause does not mention dependence, and that is deliberate. A
non-integer `pow` exponent emits an **auxiliary-variable encoding**: the
script declares a fresh `aux` and asserts `aux^q = x^p`. `aux^q` is a
product of a fresh symbol with itself, so it is nonlinear whatever the
base is — a *constant* base does not make the script linear, and the
constant fold does not remove it (the value is generally irrational, so
there is nothing exact to fold to). Stamping such a slice `QF_LRA` shipped
`(* aux aux)` under a linear logic, and both backends refused it; that
read as two flaky solvers rather than as one wrong label.

There are no other fragments. "Primary" is **ordering, not selection**:
every installed backend runs on every fragment. Read off the stamp of a
linear obligation:

```text
solver: 2 invocation(s):
  [0] z3 5.0.0 (wheel-bindings (smt2 text)) options={':produce-models': 'true', ':timeout': '10000', 'set-logic': 'QF_LRA', …} — QF_LRA portfolio primary on assert #0
  [1] cvc5 1.3.4-modified (wheel-bindings (smt2 text; wall-guarded child process)) options={':produce-models': 'true', ':tlimit': '10000', 'set-logic': 'QF_LRA', …} — QF_LRA portfolio secondary on assert #0
```

and of a nonlinear one, where the order flips and cvc5 additionally gets
the coverings options:

```text
  [0] cvc5 1.3.4-modified (…) options={…, ':tlimit': '10000', ':nl-cov': 'true', ':nl-ext': 'none', 'set-logic': 'QF_NRA', …} — QF_NRA portfolio primary on assert #0
  [1] z3 5.0.0 (…) options={…, ':timeout': '10000', 'set-logic': 'QF_NRA', …} — QF_NRA portfolio secondary on assert #0
```

An `unsat` on the negated predicate discharges the obligation; a `sat`
becomes `REFUTED` only after its model survives an independent
exact-rational replay; `unknown` or a timeout is `UNKNOWN`, never
`VERIFIED`. A sat/unsat **disagreement** between the two backends raises
`SolverDisagreement` — it is a bug oracle, never a tiebreak.

## What each backend actually decided

Ten obligations, each run three times under the full portfolio and under
each backend alone. `unsat` = discharged, `sat` = refuted with a replayed
witness, `UNKNOWN` = the backend returned a timeout at 10 s.

| obligation | fragment | both | z3 alone | cvc5 alone |
|---|---|---|---|---|
| scalar, linear | `QF_LRA` | unsat, 78–112 ms | unsat, 8–9 ms | unsat, 71–84 ms |
| 64-element array, linear | `QF_LRA` | unsat, 86–91 ms | unsat, 10–12 ms | unsat, 77–87 ms |
| 8-element array, linear, false | `QF_LRA` | sat, 86–90 ms | sat, 11–13 ms | sat, 75–117 ms |
| 2 vars, degree 2 (AM–GM) | `QF_NRA` | unsat, 80–83 ms | unsat, 9 ms | unsat, 75–87 ms |
| 2 vars, degree 6 (Motzkin) | `QF_NRA` | unsat, 92–106 ms | unsat, 12–13 ms | unsat, 81–83 ms |
| 1 var, degree 3, false | `QF_NRA` | sat, 87–88 ms | sat, 11 ms | sat, 69–71 ms |
| 32 vars, 16 elementwise products | `QF_NRA` | unsat, ~10.3 s | **UNKNOWN** (timeout) | unsat, 166–175 ms |
| 64 vars, 32 elementwise products | `QF_NRA` | unsat, ~11.0 s | **UNKNOWN** (timeout) | unsat, 772–792 ms |
| 10-factor product chain | `QF_NRA` | unsat, ~8.1 s | unsat, 123–133 ms | unsat, 8.3–8.5 s |
| 12-factor product chain | `QF_NRA` | unsat, ~16.7 s | unsat, 689–702 ms | **UNKNOWN** (timeout) |

Three things this measured, and one it did not.

**1. On `QF_LRA`, both decided everything, and z3 decided it faster by an
order of magnitude.** Part of that gap is not solving at all: the cvc5
wheel is driven through a wall-guarded child process (its `tlimit` does not
reliably preempt the coverings solver), so every cvc5 invocation pays a
process spawn — visible as the ~70 ms floor on the cheapest rows, where z3
answers in 8.

**2. On `QF_NRA`, the split goes both ways, and that is the finding.**
Wide-and-shallow problems — many independent products, low degree per
variable — were decided by cvc5 in tenths of a second and *timed z3 out*.
Deep-and-narrow problems — one long product chain, high total degree in few
variables — were decided by z3 in tenths of a second and *timed cvc5 out*.
Neither backend dominates the other on the nonlinear fragment. There is no
"install this one" answer that survives both rows.

**3. A full portfolio is not the same as a full-portfolio answer.** On the
16-products row, both backends were installed, invoked, and stamped; only
cvc5 answered. The verdict was `VERIFIED` and said so itself:

```text
  PORTFOLIO DEGRADED — assert #0 was decided by ONE solver backend (cvc5 (wheel)), not the two the portfolio is designed around; the notes say which backend was lost and why
```

**Not measured:** anything above 10 s, anything at `float32` or under
`semantics="ieee"` (which declines escalation wholly), the affine
refinement path (`refine="affine"`, which reduces what reaches a solver at
all), an external cvc5 binary via `STELLING_CVC5`, and any backend other
than the two wheels. The battery is ten small hand-written obligations
plus the declines listed below, not a corpus.

## What one backend alone costs you

Nothing about a verdict's *status* changes: every obligation in the table
that one backend could decide, that backend still decided alone, with the
same `unsat` / `sat` and the same replay of any witness. What changes is
disclosed in three places at once.

On the obligation's own detail line:

```text
  assert #0: discharged — discharged by solver escalation (QF_LRA): the box with the negated predicate is unsat per z3 (wheel) [PORTFOLIO DEGRADED: 1 of 2 backends answered; a discharge has no replay backstop]
```

In the notes, naming *which* backend was lost and *why* — "is not
installed" when it is absent, and a different phrase when a caller
restricted the portfolio, because rendering a configured restriction as a
missing dependency would send you to install something you already have:

```text
note: assert #0: portfolio degraded — only z3 (wheel) ran (cvc5 is not installed)
note: assert #0: and this is the direction with no backstop: a discharge is a universal claim over the whole declared box, so nothing downstream re-derives it the way exact-rational replay re-derives a witness. The second backend was the only independent check on this obligation and it did not answer
```

And in `Verdict.solver_redundancy`, which is the machine-readable form —
`((0, ('z3 (wheel)',)),)` for the run above, against
`((0, ('z3 (wheel)', 'cvc5 (wheel)')),)` for the two-backend run. A CI job
that cares reads it directly:

```text
one_backend = [i for i, who in v.solver_redundancy if len(who) < 2]
```

The asymmetry that note names is the reason this matters more than it
looks. A `sat` reaches `REFUTED` only through an exact-rational replay that
shares no code with the solver, so a lost second backend there costs a
cross-check that something else still performs. An `unsat` is a universal
claim over the whole declared box: nothing re-derives it, and **the second
backend is the only independent check there is.** That is the thing a
one-backend install gives up, and it gives it up on exactly the verdicts
you would most want to trust.

With **neither** backend installed nothing silently degrades either — every
escalated obligation stays `UNKNOWN` and the verdict carries:

```text
note: no SMT solver is installed — pip install "stelling[solvers]" (or set STELLING_CVC5 / put cvc5 on PATH) to enable escalation
```

## If you are installing exactly one

Pick by the shape of your obligations, then re-measure on your own:

- **Mostly linear** — sums, scalings by constants, moving averages,
  concatenations, comparisons against thresholds: `[z3]`. It is the
  `QF_LRA` primary, it decided every linear obligation here, and it did so
  without paying a subprocess spawn per call.
- **Mostly polynomial** — products of two declared arrays, squares,
  `integer_pow`, division by a declared value: `[cvc5]`. It is the
  `QF_NRA` primary, and the wide nonlinear rows above are the ones z3
  could not finish.
- **You do not know, or it is CI**: `[solvers]`. Both nonlinear failure
  directions are real, and CI is exactly where you cannot afford to
  discover which one you have.

## What no backend can reach

Some obligations never reach a solver at all, and *which* backend you
installed makes no difference to any of them — escalation declines before
any invocation, with zero stamps and the reason quoted in the verdict. All
of these were measured; the text is what the tool printed:

| what | quoted decline |
|---|---|
| a primitive with no SMT emission row (`exp`, `log`, `sqrt`, …) | `primitive 'exp' is outside the supported emission set: no SMT emission rule has been built and audited for it — an unbuilt row, not a policy refusal of the form` |
| division whose divisor spans 0 | `'div': divisor may be zero over the declared box — SMT-LIB2 division is underspecified at 0` |
| an integer-dtype computation | `'add' on dtype 'int32': jax integer arithmetic wraps on overflow and SMT-LIB2 Reals are unbounded, so a Real emission does not model it` |
| `dot_general` with two symbolic operands | `'dot_general' has NO constant operand: a sum of products of two symbolic operands is NONLINEAR arithmetic, outside this row's linear scope` |
| an obligation over the emission budget (512 element terms) | `obligation not attempted: it needs 1024 element terms and 256 root conjuncts, and its element terms put it over the per-obligation emission budget of 512` |
| any propagation under `semantics="ieee"`, or one that constrained an assume | escalation is refused wholly, before backend discovery |

Installing the other backend does not move any of these. If your
`UNKNOWN`s look like this, a solver extra is not what you are missing.

## The cvc5 wheel, verified

The `cvc5` extra installs the official PyPI wheel — the non-GPL "BSD
version" build. Two properties of it were checked directly against
cvc5 1.3.4 in this environment rather than taken from its documentation:

- **libpoly is bundled and `nl-cov` works.** A `QF_NRA` unsat that needs
  cylindrical algebraic coverings was solved through the wheel's own
  SMT-LIB2 parser with `nl-cov=true, nl-ext=none` — the exact options
  stelling emits for that fragment — and answered `unsat`. That is
  functional evidence the coverings solver is present and running, not a
  reading of the build configuration, which this wheel does not expose.
- **CoCoALib is absent**, and cvc5 says so itself: requesting
  `nl-cov-lift=lazard` printed `nl-cov::LazardEvaluation is disabled
  because CoCoA is not available. Falling back to regular calculation of
  infeasible regions.` — and the query still solved, on the fallback path.

The other GPL-gated performance components (CLN, glpk-cut-log) are absent
for the same licensing reason; that half is reported from the build's
licensing, not probed here. Only a source build with `./configure.sh --gpl
--auto-download` has them, and stelling can be pointed at one with
`STELLING_CVC5` — see the cvc5 section of the
[README](https://github.com/NicholasEhsanRoy/stelling/blob/main/README.md#cvc5-wheel-vs-external-binary).

None of the nonlinear rows in the table above were re-run against a GPL
build, so nothing here says what those components would be worth.

## z3 and high-degree rational pow (automatic workaround)

When a rational exponent like `x**(1/80)` is emitted, the auxiliary-variable
encoding produces a degree-80 polynomial (`y^80 = x`). z3's default solver
times out on this class of problem (measured: >10s at degree 80 with
perfect-square bounds `[1, 100]`). Since 0.2.0, stelling automatically
detects rational-pow auxiliary variables in the emitted script and switches
z3 to a custom tactic chain:

```
simplify -> solve-eqs -> factor(num_primes=4) -> purify-arith -> tseitin-cnf -> nlsat
```

This restores the z3 cross-check (measured: 0.35-0.6s at degree 80). cvc5
handles these polynomials natively and requires no workaround. The tactic is
transparent to the user: no configuration is needed, the portfolio runs at
full redundancy, and the verdict discloses no difference.

The workaround fires only when the script contains `(declare-const aux_...)`
declarations (the marker for rational-pow auxiliary variables). All other
scripts use z3's default `Solver()` unchanged.

## Related

- [Reading a verdict](reading-a-verdict.md) — the stamp's solver lines and
  the degraded-portfolio disclosure in context
- [Quickstart](quickstart.md) — where `solver_timeout_ms` enters
- [SOUNDNESS.md](https://github.com/NicholasEhsanRoy/stelling/blob/main/SOUNDNESS.md)
  — why a solver is never invoked on defaults, and why the option set is
  stamped
