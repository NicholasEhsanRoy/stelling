# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Budget and determinism, in one place, selected by one environment variable.

``STELLING_PROPERTY_PROFILE`` picks one of three:

``ci`` (default)
    ``derandomize=True``, small budgets. **Deterministic**: the same tree gives
    the same examples in the same order, so a property either fails on every
    push or on none of them. A randomly-seeded property is the wrong instrument
    for a per-push gate — the first spike measured one firing on 6 of 25 seeds,
    which prices the flakiness objection at roughly a quarter of unrelated PRs.

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
