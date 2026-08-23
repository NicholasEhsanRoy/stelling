"""pytest plugin: hide both solver wheels from stelling._optional.available.

Loaded with `-p nosolvers`, which pytest imports before any conftest or test
module, so `tests/_solver_gate.py`'s HAVE_SOLVER is computed with the backends
hidden. This is the page's own documented method for producing a
single-backend (here: no-backend) configuration -- nothing is uninstalled.

    PYTHONPATH=<worktree>/src:<this directory> \
      python -m pytest -q -p no:randomly -p nosolvers

THE FILE IS NAMED `nosolvers.py` FOR A REASON. It was `nosolvers-plugin.py`,
which `-p nosolvers` cannot load and `-p nosolvers-plugin` cannot either --
`import nosolvers-plugin` is not a Python identifier. The docstring said how
to load it and the filename made that false.

It is a SIMULATION of the `test-jax-no-solvers` CI lane, not that lane: there
is no jax-without-solvers venv on this box and nothing may be installed. The
genuinely wheel-free interpreter here is `/home/nick/venvs/jax051` (jax 0.5.1,
no z3, no cvc5, and no pytest), which is where the TOOL can be driven without
hiding anything; this plugin is how the SUITE is driven.

WHAT THIS SIMULATION CAN AND CANNOT JUDGE -- measured 2026-08-23, and the
previous version of this note was wrong in both directions.

* `tests/test_skip_inventory.py` **can** be judged and passes clean:
  **53 passed, 1 skipped**. This note used to say it could not be, "because
  its legitimacy predicate asks `importlib.util.find_spec` directly rather
  than going through `_optional.available`". It does not: `_wheel()` in that
  module is a one-line wrapper that CALLS `_optional.available`, which is the
  attribute this plugin replaces, so the whole `RULES` table sees the wheels
  hidden exactly as intended.

* What the simulation actually breaks is **nine other modules, 91 failures**,
  and the old note named none of them:

      tests/test_reproduce_acceptance.py        38
      tests/test_pow_row_gauge_jax.py           22
      tests/test_reproduce.py                    8
      tests/test_square_row_gauge_jax.py         7
      tests/test_square_acceptance_jaxfluids.py  5
      tests/test_optional.py                     4
      tests/test_scatter_gauge_jax.py            3
      tests/test_constant_fold_portfolio.py      3
      tests/test_falsify_default_path.py         1

  **They pre-date this branch.** The same nine modules under the same plugin
  against `main` at 6c40ddc give **91 failed, 208 passed, 1 skipped**, with
  the identical per-module counts. Whole-suite figure on this branch:
  4295 passed, 91 failed, 217 skipped.

  They are a property of the simulation rather than of the lane it stands in
  for: hiding a wheel from `_optional.available` does not hide it from
  `importlib`, from a subprocess, or from a module that already holds a
  reference -- so a module that reaches a backend by any route other than that
  one attribute sees a half-hidden environment. Judging them needs the real
  lane, which needs an install.
"""
from stelling import _optional

_real = _optional.available
_optional.available = lambda name: False if name in ("z3", "cvc5") else _real(name)
_optional.cvc5_binary = lambda: None
