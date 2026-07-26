# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The EMISSION == REPLAY census.

Replay is what makes REFUTED self-certifying: a solver model becomes a
Witness only after this layer re-derives the violation in exact rational
arithmetic, independently of the solver. That guarantee is only as broad as
replay's coverage, so a primitive that can be emitted but not replayed
yields a witness nobody can independently check.

The two sets are equal today. This pins that equality so the next addition
to the emission set fails here rather than silently creating the first gap.
"""
from __future__ import annotations

import stelling.obligation as ob


def test_emission_and_replay_cover_exactly_the_same_primitives():
    """Neither direction may drift.

    Emittable-but-not-replayable is the dangerous direction — a witness that
    cannot be independently confirmed. Replayable-but-not-emittable is
    harmless but indicates the declaration has rotted away from the dispatch,
    which is the same decay one step earlier.
    """
    assert ob._REPLAY_SUPPORTED == ob._SUPPORTED, (
        f"emittable-but-not-replayable: "
        f"{sorted(ob._SUPPORTED - ob._REPLAY_SUPPORTED)}; "
        f"replayable-but-not-emittable: "
        f"{sorted(ob._REPLAY_SUPPORTED - ob._SUPPORTED)}"
    )


def test_the_census_is_enforced_at_import_and_survives_dash_O():
    """The invariant is enforced at import as well as by this test — and by a
    RAISE, not an `assert`.

    The earlier form of this test checked for an `assert` statement and
    justified it as "a test can be deselected; an import-time assert cannot".
    That is right about pytest and wrong about Python: under `-O` or
    `PYTHONOPTIMIZE=1`, `__debug__` is False and every `assert` is compiled
    out — so the guarantee would have vanished in exactly the deployment mode
    someone reaches for in CI or a container image. An explicit raise is
    unconditional.
    """
    import inspect

    src = inspect.getsource(ob)
    assert "if _REPLAY_SUPPORTED != _SUPPORTED:" in src, (
        "the import-time census check has been removed or weakened; the set "
        "equality would then be pinned only by a test, which can be deselected"
    )
    assert "assert _REPLAY_SUPPORTED" not in src, (
        "the census check is an `assert` again — it is stripped under -O"
    )
