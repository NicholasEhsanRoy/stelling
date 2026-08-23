<img src="https://raw.githubusercontent.com/NicholasEhsanRoy/stelling/main/assets/stelling_logo.png" alt="the stelling wordmark: three cubes arranged as ∴, therefore, beside the name" width="100%">

[![ci](https://github.com/NicholasEhsanRoy/stelling/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/NicholasEhsanRoy/stelling/actions/workflows/ci.yml)
[![python: >=3.10, the floor this package declares — see "Which Python" for what CI actually runs](https://img.shields.io/badge/python-%3E%3D3.10%20declared-blue.svg)](https://github.com/NicholasEhsanRoy/stelling/blob/main/pyproject.toml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/NicholasEhsanRoy/stelling/blob/main/LICENSE)

# Stelling

An assertion-based verifier for JAX array programs — inspired by
[Kani](https://github.com/model-checking/kani) for Rust. Prove stated
properties of traced computations over declared input regions, with a stamp
on every verdict naming its own assumptions.

*stelling is not affiliated with or endorsed by the JAX project.*

---

## JAX silently narrows integer constants during tracing

<!-- doc-example: illustrative -->
```python
x = jnp.int8(100)
y = x + 256          # you wrote 256
```

The jaxpr — the trace JAX actually executes — contains:

```
add a 0:i8[]         # JAX silently made it 0
```

No error. No warning. Verify it yourself: `jax.make_jaxpr(lambda x: x + 256)(jnp.zeros(1, jnp.int8))` prints `add a 0:i8[]` — the 256 is gone.
([full reproducer](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/overflow-tripwire.md))

Six mechanisms were measured against it (`numpy_dtype_promotion("strict")`,
`enable_checks`, `debug_nans`, `debug_infs`, `np.errstate(over='raise')`,
`warnings.simplefilter('error')`) and **all six leave it silent**.
`checkify.all_checks` returns `None` on it while its out-of-bounds and
divide-by-zero controls throw.

### Check your existing test suite with a single flag

```sh
pip install stelling
pytest -p stelling.overflow
```

The tripwire hooks the exact site where the value dies and reports each
narrowing with your source line, the arithmetic, an independent
recomputation, and a one-line reproducer. It costs nothing until you switch
it on — measured overhead is within noise over 60 cold traces.

### Why this matters

A silently wrapped constant does not crash. The program runs, the loss
decreases, the simulation looks healthy. The integer you wrote and the
integer in the trace are different numbers, and nothing told you. This is
relevant anywhere JAX traces integer arithmetic:

- **Quantized inference** — a scaling factor or zero-point that wraps
  produces outputs in the wrong numerical range, silently
- **Finite-volume solvers** — a grid constant or stencil coefficient that
  wraps causes the discretisation to converge on a physically impossible
  state
- **Sensor pipelines** — an ADC offset or calibration constant that wraps
  inverts the measurement, and downstream checks pass because they verify
  the traced program, not the written one
- **Control systems** — an actuator limit or safety threshold that wraps
  can suppress a protective action

If you then **verify** a property of that trace — "output stays within
bounds" — the verifier is correct about a program you did not write. A
VERIFIED over a corrupted trace is worse than no verification at all: it
actively suppresses the signal that something is wrong.

The tripwire detects this at trace time, and the static verifier refuses to
certify a trace it flagged — the verdict is UNKNOWN with a note naming what
was narrowed. It also refuses, in different words, when it could not watch
the whole trace: "no narrowing was seen" and "no narrowing occurred" are
different claims and only the second one licenses a VERIFIED.

So a VERIFIED with the tripwire armed says the property holds AND that no
narrowing was seen on any route *that* tripwire watches. That clause is
load-bearing, which is why the release ships **three** instruments and not
one. Each is a separate opt-in dial, and none of them turns on either of the
others:

- **the trace-time tripwire** (`-p stelling.overflow`) — the const-fold site
  above, where a constant that survived into the trace dies;
- **the eager construction-site detector**
  (`--stelling-eager-truncation=error`) — array CONSTRUCTION.
  `jnp.full(shape, N, dt)` narrows its constant before any primitive is
  bound, where the hook above cannot see it; this one **raises** at the line
  that wrote the constant. What it does not reach, at that site, is a
  constant numpy destroyed before jax was called at all — two named routes,
  measured;
- **the narrowing perimeter** (`--stelling-narrowing-perimeter=error`) — an
  integer literal that does not survive the conversion into the dtype it
  meets, which need not be out of *range* to be destroyed: `x <= 2**31 - 1`
  on `float32` is a program about `2147483648.0`.

**With all three armed the watched set is still finite**, and the residue is
sharper than "constants at construction" rather than gone. Two spellings of
that same wrong threshold reach VERIFIED today — `jnp.less_equal(x, 2**31 - 1)`,
because a `jnp.*` function carries no Python operator slot to attach to, and
`(x - (2**31 - 1)) <= 0.0`, because inside a traced harness only the
comparison slots are live. And a narrowing the program **computed** rather
than wrote — a float overflow on device — leaves no literal for any of the
three to read. The watched and unwatched routes are enumerated door by door,
and the enumeration is measured rather than asserted, in
[`docs/overflow-tripwire.md`](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/overflow-tripwire.md).

---

## Where Stelling fits

| tool | focus | scope |
|---|---|---|
| **`jax.experimental.checkify`** | runtime checks: NaN, OOB indexing, div-by-zero | operates on the jaxpr after constant folding — integer-literal narrowing has already completed before the transform runs |
| **jaxtyping** | static shape and dtype checking via type hints | verifies types, not values — `int8` is the correct type; whether `256` fits in it is a different question |
| **stelling** | trace-time overflow detection + static SMT-backed proof of algorithmic properties over declared input regions | the paths where the literal is destroyed before the trace exists — array construction (`jnp.full`) and eager execution — across three opt-in instruments, each with its own measured limits ([documented](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/overflow-tripwire.md)) |

These tools are complementary. checkify catches runtime faults stelling
cannot see; jaxtyping catches shape mismatches at definition time; stelling
catches the value corruption both are blind to — in the trace, and at the
sites that destroy a constant before the trace exists — and proves
properties neither attempts.

---

## Quickstart — prove a property

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def harness():
    # ANY float64 array of 8 elements, every element in [0.1, 10.0]
    a = any_array((8,), jnp.float64, (0.1, 10.0))
    a_face = 0.5 * (a + jnp.roll(a, -1))          # your own construction
    return assert_(a_face > 0.0)                  # the obligation


v = check(harness, vacuity_mode="inputs-only")
print(v.status)
print("semantics :", v.stamp.semantics.split(":")[0])
print("solver    :", v.stamp.solver.reason)
print("nonvacuity:", v.stamp.nonvacuity)
print("coverage  :", v.stamp.coverage)
```

prints:

```
VERIFIED
semantics : real (ℝ)
solver    : no solver invoked: escalation was NOT ATTEMPTED (solver_timeout_ms not set); every obligation was judged by outward-rounded interval arithmetic alone
nonvacuity: UNCHECKED — no membership conditions declared
coverage  : 9 eqns: 8 known (89%); 1 transparent
```

VERIFIED here is about *every* array the declaration admits, not a
sample — and the stamp says in what arithmetic, with what help, over how
much of the query, and whether anyone has tied the declared box to data
you actually run on. `v.render()` prints the whole stamp.

**Full walkthrough:** [docs/quickstart.md](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/quickstart.md)

---

## Installation

```sh
pip install stelling              # zero dependencies; never touches JAX
pip install stelling[solvers]     # adds both SMT backends (cvc5 + z3)
```

Stelling has **zero required dependencies**. JAX and the SMT solvers are opt-in
extras, imported lazily on first use, so a bare install never touches the JAX
(or CUDA stack) already managing your environment:

| extra | installs | notes |
|---|---|---|
| `stelling[z3]` | `z3-solver` (MIT) | Z3 backend — the `QF_LRA` (linear) primary |
| `stelling[cvc5]` | `cvc5` (BSD-3-Clause) | cvc5 backend, official PyPI wheels — the `QF_NRA` (polynomial) primary |
| `stelling[solvers]` | both solvers | **the one to install.** The escalation portfolio uses whichever is installed; absence just means UNKNOWNs stay UNKNOWN (`design/solver-integration-build.md`) |
| `stelling[jax]` | `jax` (CPU) | bootstrap **only** — never use it if jax is already installed |
| `stelling[all]` | = `[solvers]` | deliberately excludes jax |

Installing one backend rather than both does not weaken what a verdict
claims — it removes the cross-check behind every discharge, and the verdict
discloses that itself. Measured, in both directions:
[docs/choosing-a-solver-backend.md](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/choosing-a-solver-backend.md).

At runtime Stelling always uses whichever JAX is importable in your
environment — the `[jax]` extra is bootstrap convenience plus a documented
tested-version floor, not a binding. `[all]` deliberately excludes jax: the
extra people type reflexively must never let the resolver touch a working
jax install (a bump can desync CUDA plugin wheels). Nothing is vendored;
both solver backends install from their official PyPI wheels on Linux,
macOS, and Windows.

`python -m stelling` prints which optional dependencies are installed and
runs a one-formula smoke test against each available solver.

### Which Python

The badge above names the **floor**, `requires-python = ">=3.10"`, because
that is the one interpreter fact this repository holds. The floor and the
version CI runs are different claims, and the badge used to conflate them.
Both, measured:

- **The floor is declared, not exercised.** pip enforces `>=3.10` on
  install; **no job in `.github/workflows/` asks for a 3.10 interpreter.**
  What holds the floor instead is a static read —
  `tests/test_zero_dep_import_discipline.py` checks every shipped module's
  imports against the 3.10 standard library — and that check states its own
  limit in as many words: syntax newer than the floor is invisible to it and
  would need a real floor interpreter to catch.
- **What CI provisions is mostly the runner's.** Exactly one job pins an
  interpreter — `acceptance-reproducer`, at 3.12 — and it pins for a
  dependency reason: jaxfluids requires ≥3.11 while stelling declares
  ≥3.10, so the two projects disagree about what a valid interpreter is.
  **Every other job takes whatever interpreter the runner image provides**,
  and nothing here pins it, asserts it or records it.

So a badge saying some version is *tested* would be a claim about the runner
image, which this repository does not hold and cannot promise. It says
`>=3.10 declared` instead, and both bullets above are read back off
`pyproject.toml` and `.github/workflows/` by
`tests/test_readme_claims.py`: a lane that starts or stops pinning, a pin
that moves, or a floor that moves reddens this section rather than leaving
it quietly false.

### cvc5: wheel vs external binary

The `cvc5` extra installs the official PyPI wheel — the non-GPL "BSD
version" build. Verified against cvc5 1.3.4: the wheel bundles libpoly, so
the cylindrical-algebraic-coverings solver (`nl-cov`) that nonlinear real
arithmetic leans on is fully functional. What the wheel lacks is the `cvc5`
CLI and the GPL-gated performance components — CLN (exact-arithmetic
speed), glpk-cut-log (LP acceleration), and CoCoALib (Gröbner-basis
speedups inside coverings, finite-field theory). The official GitHub
release binaries are built the same way (libpoly yes, GPL components no),
so a source build with `./configure.sh --gpl --auto-download` is the only
route to those.

To point stelling at a different cvc5 — a nightly, a distro or custom
build, or (if you genuinely need the GPL components) your own source build:

```sh
export STELLING_CVC5=/path/to/cvc5   # or just put `cvc5` on PATH
```

Plainly: the near-term value of the external-binary route is nightlies and
alternative builds, and the SMT-LIB 2 subprocess transport it rides on is
the same one a later dReal-style backend will use. The GPL source build is
a documented possibility, not an expectation — nobody does it casually.
Either way, an external solver stays a separate program you chose to
install: nothing links into stelling, and its Apache-2.0 licensing is
unaffected. `python -m stelling` reports both transports, including which
optional components a discovered binary was built with.

---

## What it does — and what it doesn't, measured

**Does:**

- Checks **stated box invariants on continuous flows** (edge-flux
  inductiveness): the harness declares bounded inputs
  (`any_array`), obligations (`assert_`), and membership of known data in
  the declared set (`nonvacuity`) — all as traced primitives, so the
  query's content hash covers the declarations, not just the program.
- **Forward interval propagation** over the jax-free IR
  (`stelling.ir`), **outward-rounded** — the exact real result is always
  inside the bracket; how tight the bracket is depends on the operation,
  and `stelling.interval`'s module docstring is the scope
  (`mul([0.25, 0.5], [0.25, 0.5])` is exactly `[0.0625, 0.25]`, no ulp
  spent; `exp` and `pow` do bump one ulp unconditionally, and carry a
  stamped libm assumption for it) — with three-valued verdicts:
  **VERIFIED**, **REFUTED** (set-level: the stated box is not invariant —
  not a witness), **UNKNOWN** (our imprecision, never guessed away).
- **Checks the preconditions your solver assumes** — positivity of a
  coefficient field over its envelope, a nonzero mass/shift scalar over
  its admissible config range — as reusable obligation templates
  (`stelling.preconditions`) with a one-call front door (`check()`), each
  verdict stamped. Guide: [docs/preconditions.md](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/preconditions.md).
- **Escalates undecided obligations to SMT solvers** (optional extras,
  never required): scalar linear/polynomial obligations emit as
  SMT-LIB2 text — exact dyadic rationals, the closed declared box, the
  negated predicate — routed by fragment through a portfolio (cvc5 with
  coverings for nonlinear, Z3 as cross-check). Agreement decides;
  disagreement is a loud error, never a silent pick. `sat` becomes
  **REFUTED with a concrete witness**, checked for box membership and
  predicate violation by exact-rational replay before it is believed;
  timeout or `unknown` stays UNKNOWN — a timeout is never a VERIFIED;
  unsupported fragments stay UNKNOWN with the reason quoted.
- **Every verdict carries a full stamp**: stelling and jax versions,
  query content hash, arithmetic representation *and* semantics,
  precision configuration, solver — recorded absence when intervals
  decided alone, or every invocation (name, version, transport, exact
  option set) when escalation ran — nonvacuity,
  the tier and provenance of every transfer used, its assumptions, and
  ⊤-coverage — including constraints that were *dropped*, which are
  counted and named, never hidden.

**Doesn't (yet, and the stamp or the docs say so in each case):**

<!-- capability-exempt: disclaimer (this is the measured "doesn't" list) -->
- **Derive invariants.** Check mode only: you state the box, stelling
  judges it. Derive mode is designed and deliberately unbuilt.
- **Handle `cond` / `scan` / `while`.** Control flow falls to ⊤ and is
  counted as such in coverage.
- **Say anything about discrete steps.** Every verdict so far is about
  the continuous flow; a solver's stepped trajectory is a different
  object, and no artifact here blurs them.
- **Judge in float semantics by default.** The default stamp says
  `real`: obligations are judged in exact real arithmetic, and a
  predicate can hold in ℝ while failing in floats — that gap held a
  258-day bug upstream, which is why the stamp names its semantics per
  verdict. An opt-in `ieee` mode now judges the censused IEEE
  behaviours (rounding collapse, overflow-as-value, NaN) in all four
  catalogued formats and stamps itself; it treats subnormal-band
  outcomes as indeterminate (measured: this CPU target flushes
  subnormals in three of the four formats; others may not), refuses
  solver escalation (the SMT backends speak ℝ), and **declines `exp` and
  `pow` outright unless the caller declares an accuracy budget** for the
  backend that will execute them — measured, XLA's `float32` `exp` is up
  to 5.5 ulps from the true value, so a bracket built around this
  machine's libm is not a bracket of the compiled program. Every counted
  or recorded verdict to date is a `real`-mode verdict.
- **Discharge the recorded incidents.** Against the 20 long-horizon
  failures this project mined from public trackers, hand proofs
  discharged **0 of 3** attempted; the box invariants it checks are
  preconditions of arguments, not incident closures
  (`design/supply-probe.md`, `design/layer-probe.md`).
<!-- /capability-exempt -->

<!-- capability-exempt: roadmap -->
The roadmap (`design/founding.md`) aims it further: index safety, and
bounds over horizons no test can reach.
<!-- /capability-exempt -->

---

## Disclaimer and recommended practice

Stelling is open-source software provided as-is under the Apache-2.0
license, with no warranty of any kind. A VERIFIED verdict is a statement
about a mathematical model under stated assumptions — it is not a
guarantee about your deployed system, and the stamp exists precisely to
name the gap between the two.

**Do not rely on any single tool for safety-critical decisions.** Stelling
is one layer in a verification stack, not a replacement for the others:

- **Testing** (pytest, unittest) — exercises concrete inputs and catches
  regressions no static tool looks for
- **Property-based testing** ([hypothesis](https://hypothesis.readthedocs.io/))
  — generates adversarial inputs and finds edge cases no author anticipated
- **Runtime checking** (`jax.experimental.checkify`) — catches OOB, NaN,
  and division by zero at execution time, which stelling does not attempt
- **Type checking** (jaxtyping, mypy, pyright) — catches shape and type
  mismatches at definition time
- **Code review and domain expertise** — the only instrument that can judge
  whether the declared envelope matches the physical system

We are actively working to make stelling as correct and useful as
possible, and we disclose its limitations in every verdict it produces. If
you find a defect, please report it.

---

## Documentation

| | |
|---|---|
| [Quickstart](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/quickstart.md) | install, one runnable harness, a stamped verdict |
| [The harness API](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/harness-api.md) | the import path and every primitive: `any_array`, `any_pytree`, `assert_`, `assume`, `nonvacuity`, `trace` |
| [Reading a verdict](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/reading-a-verdict.md) | the statuses, every stamp line, and the two vacuity instruments |
| [Preconditions guide](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/preconditions.md) | ready-made obligation templates and posing guidance |
| [Choosing a solver backend](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/choosing-a-solver-backend.md) | z3, cvc5, or both — how obligations are routed, what each backend decided, and what one alone costs |
| [The overflow tripwire](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/overflow-tripwire.md) | full reference for all three instruments: every door each one watches, every door it does not, xdist aggregation, and the gate that refuses to verify a corrupted trace |
| [Reproducing a witness](https://github.com/NicholasEhsanRoy/stelling/blob/main/docs/reproducing-a-witness.md) | emit a runnable file that executes a REFUTED's witness through **your own program, with stelling uninstalled** — the one check that does not trust this tool |
| [SOUNDNESS.md](https://github.com/NicholasEhsanRoy/stelling/blob/main/SOUNDNESS.md) | what a verdict is permitted to claim |
| [docs/](https://github.com/NicholasEhsanRoy/stelling/tree/main/docs/) | index, including the project-state and ledger records |

## Development

```sh
pip install -e ".[solvers,jax]" --group dev   # pip ≥ 25.1; uv works too
pre-commit install                            # SPDX headers, REUSE, import hygiene
pytest
```

> **`[jax]` here assumes a fresh venv.** It is in this line because a
> contributor's clean environment needs jax to run the suite. **If you are
> installing into an environment that already has jax, drop it** — use
> `".[solvers]" --group dev` instead. The extra exists only to bootstrap an
> environment with no jax at all, and letting it into a resolver that is
> already managing your jax can desync CUDA plugin wheels.

Every source file carries an SPDX header (template in `.license-header.txt`);
the pre-commit hook inserts it into new files automatically. Commits must be
signed off (`git commit -s`) — see [CONTRIBUTING.md](https://github.com/NicholasEhsanRoy/stelling/blob/main/CONTRIBUTING.md) and
[DCO](https://github.com/NicholasEhsanRoy/stelling/blob/main/DCO). Two import rules, both enforced by hooks and tests: (1) only
`stelling/_jax_compat.py` may spell `import jax` / `from jax` — the
churn boundary; (2) `jax._src` is banned everywhere except one pinned
file (`_tripwire/_adapter_jax.py`, which reaches the private
constant-fold registry via `importlib.import_module`). Everything else
consumes the jax-free `stelling.ir`.

## License

**Apache-2.0 for the code; marks reserved.** All source is Apache-2.0 —
deliberately no NOTICE file: nothing is vendored, so there is no
attribution to propagate (one gets added the day third-party code actually
lands in-tree). The stelling name and logo (`assets/`) are **not** under
the code license: they are marks of the maintainer, reserved so a fork
cannot be mistaken for the project
([`LICENSES/LicenseRef-stelling-marks.txt`](https://github.com/NicholasEhsanRoy/stelling/blob/main/LICENSES/LicenseRef-stelling-marks.txt)).
Nominative use — referring to stelling by name or logo — is fine and
expected. This is the same source-open/marks-reserved split Ferrocene
ships, not an open-core arrangement, and it is consistent with Apache-2.0,
whose §6 grants no trademark rights anyway.

**No solver is a required dependency, and none is linked or vendored.**
The SMT backends are optional extras — separate wheels you opt into — and
an external cvc5 binary is driven as a separate process over SMT-LIB2
text. Nothing solver-shaped is compiled into, vendored into, or derived
into stelling. This is true by measurement, not just policy: the full
test surface passes in an environment with no solver installed at all;
verdicts that used no solver stamp that absence explicitly, and verdicts
that escalated stamp every invocation — solver, version, transport, and
the exact emitted option set.

Provenance is machine-verifiable: the repo is REUSE-compliant
([reuse.software](https://reuse.software); `LICENSES/`, `REUSE.toml`,
`reuse lint` in CI), contributions are DCO-signed, and releases are
published via PyPI Trusted Publishing with PEP 740 attestations
([SECURITY.md](https://github.com/NicholasEhsanRoy/stelling/blob/main/SECURITY.md) shows how to verify). The verdict trust policy
lives in [SOUNDNESS.md](https://github.com/NicholasEhsanRoy/stelling/blob/main/SOUNDNESS.md).
