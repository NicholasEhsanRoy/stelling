# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The pin is MEASURED, not trusted.

``_profiles.pin_local_constant_pool`` reaches into ``hypothesis.internal``,
which carries no stability contract. It raises when the attachment point is
gone — but "the assignment ran" and "the search stopped reading the import set"
are different claims, and only the second one is worth anything. So this
imports a local module the property suite does not use and asserts the pool
does not move.

WHY IT IS THIS SUITE'S PROBLEM AND NOT HYPOTHESIS'S. The feature is a good one
in general: constants harvested from the code under test are exactly the values
a bug is likely to sit on. What it collides with here is a **strict xfail**.
``test_oracle.py``'s wrap leg is marked ``xfail(strict=True)`` precisely so that
the day the open defect is fixed the suite goes RED and somebody has to come
and delete the marker — and a search whose examples depend on what else the
session imported turns "the defect was fixed" and "somebody ran pytest with an
extra path on the command line" into the same signal. Measured before the pin,
on this tree::

    pytest tests/property/test_oracle.py                          -> 1 xfailed
    pytest tests/property/test_oracle.py tests/test_obligation_slice.py \\
           --deselect tests/test_obligation_slice.py              -> 1 FAILED
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis", reason="needs hypothesis")

import _profiles  # noqa: E402


def _pool_snapshot():
    """Every constant hypothesis would mix in from this repository's source."""
    from hypothesis.internal.conjecture import providers

    return frozenset(providers._get_local_constants())


def test_the_pin_is_installed():
    assert _profiles.pool_is_pinned(), (
        "_profiles.py did not install the local-constant pin, so the "
        "determinism claim in its docstring is unbacked in this session"
    )


def test_the_pool_is_the_declared_one():
    assert _pool_snapshot() == frozenset(_profiles.DECLARED_LOCAL_POOL)


def test_importing_another_local_module_does_not_move_the_pool():
    """THE MEASUREMENT, and the whole reason this file exists.

    ``stelling.obligation`` is a local, non-test module the property suite does
    not otherwise import — the same one whose mere co-collection flipped the
    oracle's strict xfail. Unpinned, importing it takes the pool from 2
    constants to 274 at the oracle's first input.
    """
    before = _pool_snapshot()
    import stelling.obligation  # noqa: F401
    import stelling.propagate  # noqa: F401

    assert _pool_snapshot() == before, (
        "importing a local module changed the pool hypothesis draws from, so "
        "the pin is not in force and the examples this suite sees are still a "
        "function of what else the session imported"
    )


def test_the_snapshot_reader_is_not_vacuous():
    """A reader that always returned an empty set would make the two tests
    above pass on any tree at all.

    Driven against hypothesis's own GLOBAL pool, which the pin does not touch
    and which is non-empty by construction — so this asserts the private path
    being read is still the path that feeds generation.
    """
    from hypothesis.internal.conjecture import providers

    assert len(providers.GLOBAL_CONSTANTS.integers) > 0


def test_the_provider_reaches_the_pool_through_the_name_the_pin_replaces():
    """THE ASSUMPTION UNDER THE PIN, and it used to be an assertion that
    could not fail.

    ``pin_local_constant_pool`` rebinds the module-level
    ``providers._get_local_constants``. That only pins anything if the code
    that FEEDS GENERATION calls that global on every draw rather than holding
    a reference taken at import — and the line that claimed to check it read
    ``assert providers._get_local_constants() is providers._get_local_constants()``,
    which passes pinned and unpinned alike, because either way the callee
    returns a module-global ``Constants``. It measured the identity of a
    return value, not the route.

    This measures the route: the name is rebound to a sentinel pool and the
    provider's own ``_local_constants`` is read. The consumer is
    ``HypothesisProvider._maybe_draw_constant``, which reaches the pool
    exclusively through that ``cached_property``. If hypothesis ever binds the
    function at import instead, this goes RED — and it is the only thing here
    that would, since ``pin_local_constant_pool``'s ``hasattr`` check would
    still find the name and the two tests above would still see a pinned pool.
    """
    from hypothesis.internal.conjecture import providers
    from hypothesis.internal.constants_ast import Constants

    sentinel = Constants()
    sentinel.add(-987654321)
    original = providers._get_local_constants
    try:
        providers._get_local_constants = lambda: sentinel
        seen = providers.HypothesisProvider(None)._local_constants
    finally:
        providers._get_local_constants = original

    assert seen is sentinel, (
        "the provider did not pick up a replaced "
        "`providers._get_local_constants`, so rebinding that name — which is "
        "all `_profiles.pin_local_constant_pool` does — no longer decides "
        "what generation draws from. The determinism claim in _profiles.py's "
        "docstring is unbacked; find where the pool reaches the provider now "
        "and re-point the pin, or delete the claim."
    )
    # and the restore worked, so this test is not itself a polluter: the pool
    # every later property sees is the DECLARED one again, not the sentinel
    assert providers._get_local_constants is original
    assert _pool_snapshot() == frozenset(_profiles.DECLARED_LOCAL_POOL)
