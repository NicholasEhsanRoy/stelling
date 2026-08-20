# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Budget and determinism, in one place, selected by one environment variable.

``STELLING_PROPERTY_PROFILE`` picks one of three:

``ci`` (default)
    ``derandomize=True``, small budgets. **Deterministic given the tree AND the
    session's import set**, so a property either fails on every push or on none
    of them. A randomly-seeded property is the wrong instrument for a per-push
    gate — the first spike measured one firing on 6 of 25 seeds, which prices
    the flakiness objection at roughly a quarter of unrelated PRs.

    THE SECOND CONJUNCT IS NOT DECORATION, and this sentence used to omit it
    and say "the same tree gives the same examples" flat. That was FALSE as
    written. Hypothesis mixes a pool of constants harvested from the
    **local non-test modules that happen to be in ``sys.modules``** into
    generation, so the examples a property sees are a function of what else the
    session imported. Measured on this tree, jax 0.11.1, hypothesis 6.165.10::

        pytest tests/property/test_oracle.py                          -> 1 xfailed
        pytest tests/property/test_oracle.py tests/test_obligation_slice.py \
               --deselect tests/test_obligation_slice.py              -> 1 FAILED

    — one ordinary module co-collected and **fully deselected** flips the
    strict-xfail oracle to XPASS and reddens the suite for a reason that has
    nothing to do with the defect it guards. A bigger pool is not a better
    search, which is why whole-tree collection does not flip it and one module
    does.

    The claim held for the LANE all along, because the lane runs one fixed
    command (`pytest -q -ra tests/property`) and therefore one fixed import
    set. It did not hold for the sentence, and a sentence about determinism
    that is true only of one invocation is the kind of claim this project does
    not ship.

    :func:`pin_local_constant_pool` closes it rather than narrowing the
    sentence to the lane: the pool is pinned to :data:`DECLARED_LOCAL_POOL`,
    the import set stops reaching the search at all, and the conjunct above
    becomes true of every invocation. What survives as a qualification is the
    ordinary one — the same tree, the same hypothesis, the same jax.

``dev``
    Randomised, larger budgets, Hypothesis's own ``.hypothesis/`` database.
    What a developer runs while working on a property.

``nightly``
    Randomised, much larger budgets, and a database in
    ``STELLING_PROPERTY_DB`` when set (a cached directory in a scheduled job),
    so that a new failure pins itself for the next run.

**A constraint no recipe can dodge, and it is measured, not read off the
docs**: ``derandomize=True`` implies ``database=None``, and passing both is an
``InvalidArgument``. You get determinism **or** a replaying corpus, never both.
An earlier recommendation in this project's own notes asked for both together;
that recipe does not run. Hence: ``ci`` is derandomised and database-free,
``nightly`` is randomised and database-backed, and they are different profiles
rather than different flags on one.

The example database is worth nothing in an ephemeral CI image in any case —
measured: a run with the database and a run without it found the same failure
at the same example count, because the seed found it unaided before any replay
could matter. ``.hypothesis/`` is gitignored and CI does not cache it, so the
``ci`` profile creating one would be pure noise.

``STELLING_PROPERTY_SCALE`` multiplies every budget (float, default 1.0), which
is how the runner turns one profile into a sweep without editing a test.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

PROFILES = ("ci", "dev", "nightly")

_FACTOR = {"ci": 1.0, "dev": 4.0, "nightly": 40.0}
_DERANDOMIZE = {"ci": True, "dev": False, "nightly": False}


def name() -> str:
    chosen = os.environ.get("STELLING_PROPERTY_PROFILE", "ci")
    if chosen not in PROFILES:
        raise ValueError(
            f"STELLING_PROPERTY_PROFILE must be one of {PROFILES}, got {chosen!r}"
        )
    return chosen


def scale() -> float:
    return float(os.environ.get("STELLING_PROPERTY_SCALE", "1.0"))


@dataclass(frozen=True)
class Profile:
    name: str
    derandomize: bool
    factor: float

    def budget(self, base: int) -> int:
        return max(1, int(round(base * self.factor)))

    def settings(self, base: int, **extra):
        """A ``hypothesis.settings`` for a property whose ``ci`` budget is ``base``."""
        from hypothesis import HealthCheck, settings

        kw = dict(
            max_examples=self.budget(base),
            deadline=None,
            derandomize=self.derandomize,
            print_blob=True,
            suppress_health_check=[
                HealthCheck.too_slow,
                HealthCheck.filter_too_much,
                HealthCheck.data_too_large,
                HealthCheck.large_base_example,
            ],
        )
        if not self.derandomize:
            db_dir = os.environ.get("STELLING_PROPERTY_DB")
            if db_dir:
                from hypothesis.database import DirectoryBasedExampleDatabase

                kw["database"] = DirectoryBasedExampleDatabase(db_dir)
        kw.update(extra)
        return settings(**kw)


def current() -> Profile:
    n = name()
    return Profile(n, _DERANDOMIZE[n], _FACTOR[n] * scale())


# ---------------------------------------------------------------------------
# The local-constant pool, pinned
# ---------------------------------------------------------------------------

#: The constants hypothesis may mix into generation from THIS repository's own
#: source. Empty, and the emptiness is the decision rather than a default.
#:
#: WHAT THE FEATURE DOES. Hypothesis walks the local (non-test, non-stdlib,
#: non-site-packages) modules in ``sys.modules``, harvests every integer,
#: float, bytes and string constant out of their ASTs, and draws from that pool
#: with probability 0.05 per choice. The pool therefore depends on what the
#: SESSION imported, not on what the property under test is about. Measured at
#: the oracle's first input: 2 constants alone, 274 with ``stelling.obligation``
#: imported, 633 with the whole tree collected.
#:
#: WHY EMPTY AND NOT A DECLARED LIST. ``_grammar.py`` chooses this suite's
#: literal pools deliberately and says so — "literals drawn from in-range,
#: just-out-of-range and far-out-of-range pools", sized against the dtype under
#: test, which is the whole mechanism the wrap oracle rests on. Constants
#: harvested from ``stelling``'s own source are not that: they are budget caps,
#: array shapes, sha1 truncation lengths and error-message widths, mixed into a
#: search over integer harnesses because they happened to be in a file someone
#: else's test imported. Nothing here ever asked for them, and no property in
#: this suite is written against them.
#:
#: MEASURED BOTH WAYS, and the marker survives: with the pool pinned empty the
#: oracle's strict xfail holds alone AND with an ordinary module co-collected
#: (``1 xfailed`` in both), where unpinned the second is ``1 FAILED``.
DECLARED_LOCAL_POOL: tuple = ()


def pin_local_constant_pool():
    """Make the search independent of the session's import set. Returns a code.

    RAISES IF THE ATTACHMENT POINT MOVED, and that is the point. This reaches
    into ``hypothesis.internal``, which carries no stability contract — so a
    pin that silently failed to install would leave the docstring above
    claiming a determinism nothing delivers, which is the exact shape of defect
    it was written to remove. It fails closed instead, and
    ``tests/property/test_search_determinism.py`` measures the INSTALLED pin
    behaviourally rather than trusting this function's return value.

    The dev group pins ``hypothesis>=6.165.2,<7``, so the surface this names is
    bounded by a version range this project already controls.
    """
    from hypothesis.internal.conjecture import providers
    from hypothesis.internal.constants_ast import Constants

    pool = Constants()
    for constant in DECLARED_LOCAL_POOL:
        pool.add(constant)

    if not hasattr(providers, "_get_local_constants"):
        raise RuntimeError(
            "hypothesis.internal.conjecture.providers._get_local_constants is "
            "gone, so the local-constant pool cannot be pinned and this "
            "suite's determinism claim is unbacked. Find where the pool is "
            "assembled now and re-point this, or delete the claim."
        )
    providers._get_local_constants = lambda: pool
    # entries computed under the old pool would otherwise survive: the cache is
    # keyed on the choice constraints, not on the pool.
    providers.CONSTANTS_CACHE.cache.clear()
    return "pinned"


#: ``"pinned"`` or ``"no-hypothesis"``. Installed at MODULE SCOPE, on import,
#: which is the earliest point in this directory that is guaranteed to precede
#: generation: every module here imports this one, the pool is read on the
#: first INPUT, and a module import happens before any of them.
#:
#: NOT A ``tests/property/conftest.py``, and that is measured rather than
#: stylistic. pytest's prepend import mode names a conftest after its own
#: directory's basename, so a second ``conftest.py`` anywhere under ``tests/``
#: competes for the module name ``conftest`` — and ``test_skip_inventory.py``
#: reaches the recorder with ``from conftest import ...``. Driven: with a
#: ``tests/property/conftest.py`` present, the session-end pin reports
#: ``ImportError("cannot import name 'CLAIM_MADE' from 'conftest'
#: (tests/property/conftest.py)")`` and the completeness claim FAILS. One
#: conftest under ``tests/``; wiring that needs to run early goes here.
POOL_PIN = "no-hypothesis"

if importlib.util.find_spec("hypothesis") is not None:
    POOL_PIN = pin_local_constant_pool()


def pool_is_pinned() -> bool:
    """Whether the pin above is in force in this process."""
    return POOL_PIN == "pinned"
