# Stelling
Inspired by Kani, Stelling is an assertion-based verifier for JAX array programs. It lowers jaxpr to SMT queries to mathematically prove invariants, index safety, and robust bounds over infinite horizons, ensuring that scientific code is strictly verified against silent failures.

## Installation

Not yet on PyPI — install from a clone:

```sh
pip install -e ".[solvers]"   # both SMT backends; never touches JAX
pip install -e ".[jax]"       # bootstrap ONLY: for an environment with no JAX at all
```

Stelling has **zero required dependencies**. JAX and the SMT solvers are opt-in
extras, imported lazily on first use, so a bare install never touches the JAX
(or CUDA stack) already managing your environment:

| extra | installs | notes |
|---|---|---|
| `stelling[z3]` | `z3-solver` (MIT) | Z3 backend |
| `stelling[cvc5]` | `cvc5` (BSD-3-Clause) | cvc5 backend, official PyPI wheels |
| `stelling[solvers]` | both solvers | the recommended install |
| `stelling[jax]` | `jax` (CPU) | bootstrap **only** — never use it if jax is already installed |
| `stelling[all]` | = `[solvers]` | deliberately excludes jax |

At runtime Stelling always uses whichever JAX is importable in your
environment — the `[jax]` extra is bootstrap convenience plus a documented
tested-version floor, not a binding. `[all]` deliberately excludes jax: the
extra people type reflexively must never let the resolver touch a working
jax install (a bump can desync CUDA plugin wheels). Nothing is vendored;
both solver backends install from their official PyPI wheels on Linux,
macOS, and Windows.

`python -m stelling` prints which optional dependencies are installed and
runs a one-formula smoke test against each available solver.

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

## Development

```sh
pip install -e ".[solvers,jax]" --group dev   # pip ≥ 25.1; uv works too
pre-commit install                            # SPDX headers, REUSE, import hygiene
pytest
```

Every source file carries an SPDX header (template in `.license-header.txt`);
the pre-commit hook inserts it into new files automatically. Commits must be
signed off (`git commit -s`) — see [CONTRIBUTING.md](CONTRIBUTING.md) and
[DCO](DCO). Only `stelling/_jax_compat.py` may import jax; everything else
consumes the jax-free `stelling.ir`, and both rules are enforced by hooks
and tests.

## License

Apache-2.0. Deliberately no NOTICE file: nothing is vendored, so there is no
attribution to propagate. One gets added the day third-party code actually
lands in-tree.

Provenance is machine-verifiable: the repo is REUSE-compliant
([reuse.software](https://reuse.software); `LICENSES/`, `REUSE.toml`,
`reuse lint` in CI), contributions are DCO-signed, and releases are
published via PyPI Trusted Publishing with PEP 740 attestations
([SECURITY.md](SECURITY.md) shows how to verify). The verdict trust policy
lives in [SOUNDNESS.md](SOUNDNESS.md).
