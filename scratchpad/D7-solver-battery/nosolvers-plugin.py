"""pytest plugin: hide both solver wheels from stelling._optional.available.

Loaded with `-p nosolvers`, which pytest imports before any conftest or test
module, so `tests/_solver_gate.py`'s HAVE_SOLVER is computed with the backends
hidden. This is the page's own documented method for producing a
single-backend (here: no-backend) configuration -- nothing is uninstalled.

It is a SIMULATION of the `test-jax-no-solvers` CI lane, not that lane: there
is no jax-without-solvers venv on this box and nothing may be installed.
`tests/test_skip_inventory.py` cannot be judged under it, because its
legitimacy predicate asks `importlib.util.find_spec` directly rather than
going through `_optional.available`, so it still sees the wheels.
"""
from stelling import _optional

_real = _optional.available
_optional.available = lambda name: False if name in ("z3", "cvc5") else _real(name)
_optional.cvc5_binary = lambda: None
