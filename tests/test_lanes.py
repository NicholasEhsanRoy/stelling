# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The fences over what CI actually runs.

``tests/_lanes.py`` reads ``ci.yml``; this holds every claim in the tree that
depends on a lane to the lane that delivers it. Three claims, one shape — a
coverage statement checked against an enumeration instead of against the
machine that would have to produce it:

* ``TESTED_JAX_SERIES`` — *"an entry with no lane is a claim, not a test"* is
  the rule written above that constant. This is the first thing that enforces
  it.
* the supported install configurations — the ``[jax]``-without-``[solvers]``
  one had no whole-suite lane, and 72 tests were failing in it.
* the randomised-order lane, which is not a coverage claim but an
  *unenumerated* backstop, and which has to exist before anything can say the
  suite is order-independent.

None of these needs the workflow to be RUN. They need it to be READ.
"""

from __future__ import annotations

import pytest

import _lanes
from stelling._optional import TESTED_JAX_SERIES


def test_the_declared_lane_table_is_what_ci_yml_says():
    """The measured/declared pin. Same idiom as the skip inventory: a lane
    added, removed or re-provisioned is a line in a diff, never a silent
    change in what CI measures."""
    measured = {
        lane.job: (lane.jax, lane.solvers, lane.whole_suite, lane.random_order)
        for lane in _lanes.lanes()
    }
    assert measured == _lanes.EXPECTED_LANES, (
        "the CI lanes moved.\n"
        f"  declared {_lanes.EXPECTED_LANES}\n"
        f"  measured {measured}\n"
        "Update EXPECTED_LANES *and* check what else in this file depends on "
        "the lane that changed. This table is a claim about which environment "
        "CI actually provisions; it is not a list to be re-typed until the "
        "suite goes green."
    )


def test_every_series_bearing_job_is_a_whole_suite_lane_that_exists():
    by_job = {lane.job: lane for lane in _lanes.lanes()}
    for job in _lanes.SERIES_BEARING:
        assert job in by_job, f"SERIES_BEARING names {job!r}, which ci.yml has not"
        assert by_job[job].whole_suite, (
            f"{job} is credited with delivering a series, but it does not run "
            f"the whole suite, so `test_doc_example` may never execute in it"
        )


def test_every_tested_series_has_a_lane():
    """THE RULE ``_optional.py`` STATES AND NOTHING ENFORCED.

    ``TESTED_JAX_SERIES`` is a claim about what CI runs. Today ``"0.11"`` is
    delivered by the FLOATING lane alone — no job pins it — so the entry is
    true only for as long as 0.11 is the newest jax there is.

    DRIVEN, the jax-0.12 scenario end to end. Bump the tuple to
    ``("0.10", "0.11", "0.12")``, which is what a maintainer does after
    ``test_tested_jax_series_is_silent`` and the tripwire's known-hash row have
    both gone red and been cleared the obvious way, and re-run::

        AssertionError: TESTED_JAX_SERIES claims a series no lane runs: 0.11 —
        no job pins it and the floating lane resolves 0.12. Add the lane (pin
        the series, the way `test-jax-0-10` does) or drop the entry; an entry
        with no lane is a claim, not a test.

    and, from the doc-hash inventory next door, in the same run::

        AssertionError: a documented query content hash is compared on NO
        tested jax lane … {'quickstart.md#0': 'quickstart.md:66'}
        The lanes resolve ('0.10', '0.12'); TESTED_JAX_SERIES claims
        ('0.10', '0.11', '0.12').

    WHAT IT LOOKS LIKE WITHOUT THIS, measured in the same tree: keyed on
    ``TESTED_JAX_SERIES`` the inventory recomputes ``quickstart.md#0`` as
    ``("0.11",)`` — unchanged, matching what ``EXPECTED_HASH_COVERAGE``
    declares — so that half is green while no lane runs 0.11 at all. Only
    ``harness-api.md#0`` moves, because it carries no stamp and is compared
    everywhere, and updating it is the obvious clearing the failure message
    itself asks for. That is B13's hole one level up: the escape's condition
    widening until it covers every lane there is.

    The remedy both messages ask for is a ``test-jax-0-11`` lane beside
    ``test-jax-0-10``, which is what makes the tuple true again.
    """
    lanes = _lanes.lane_series()
    orphaned = sorted(set(TESTED_JAX_SERIES) - set(lanes))
    assert not orphaned, (
        f"TESTED_JAX_SERIES claims a series no lane runs: {', '.join(orphaned)} "
        f"— no job pins it and the floating lane resolves "
        f"{_lanes._newest(TESTED_JAX_SERIES)}. Add the lane (pin the series, "
        f"the way `test-jax-0-10` does) or drop the entry; an entry with no "
        f"lane is a claim, not a test."
    )
    untested = sorted(set(lanes) - set(TESTED_JAX_SERIES))
    assert not untested, (
        f"a lane runs jax {', '.join(untested)}, which TESTED_JAX_SERIES does "
        f"not claim. Either the constant is behind the workflow or a pin is "
        f"pointed at a series nobody has verified."
    )


def test_the_series_derivation_is_not_vacuous():
    """The anti-vacuity control for :func:`_lanes.lane_series`.

    Without it the fence above passes for free the day the parser stops
    recognising a pin — every lane reads as ``"absent"``, the derived set is
    empty, and an empty set has no orphans to name.
    """
    lanes = _lanes.lanes()
    assert lanes, "no lanes were parsed at all, so nothing above measures anything"
    assert any(l.jax == "floating" for l in lanes), "the floating lane went missing"
    assert any(
        l.jax not in ("absent", "floating", "matrix") for l in lanes
    ), "no lane pins a series any more, so `lane_series` is one inference wide"
    assert _lanes.lane_series(), "the derived lane series is empty"
    # the pin regex must want a CEILING: a floor alone is not a pin, and
    # `.[jax]`'s requirement has a floor of 0.10 today
    assert _lanes._SERIES_PIN.search('"jax>=0.10,<0.11"')
    assert not _lanes._SERIES_PIN.search('"jax>=0.10"')
    # and the newest-series rule really orders numerically, not lexically
    assert _lanes._newest(("0.9", "0.10")) == "0.10"


#: The environments the project's install story tells a user they can have,
#: and which of the two optional pieces each provides. Every one of them must
#: have a whole-suite lane, or "stelling works in this configuration" is a
#: sentence nothing measures.
#:
#: THE FOURTH CELL IS DELIBERATELY ABSENT AND NAMED HERE. Neither jax nor a
#: solver is a configuration in which stelling can decide anything at all —
#: there is no trace to transcribe — so there is no verdict to check and no
#: lane. What that configuration DOES have to satisfy is that the package
#: imports and the CLI runs, and that is measured by
#: `tests/test_zero_dep_import_discipline.py` in a subprocess with both
#: blocked, in every lane there is.
SUPPORTED = {
    ("jax", "solvers"): "test-jax",
    ("jax", "no solvers"): "test-jax-no-solvers",
    ("no jax", "solvers"): "test-no-jax",
}


@pytest.mark.parametrize("config,job", sorted(SUPPORTED.items()))
def test_every_supported_configuration_has_a_whole_suite_lane(config, job):
    """The hole item 1 of this batch came from.

    ``pip install -e ".[jax]" --group dev`` — jax, no solvers — is what a
    contributor gets by following CONTRIBUTING.md without asking for the
    solver extra, and no job ran the suite in it. Measured in that exact
    environment on 2026-08-20, before the lane existed: **72 failed, 3796
    passed, 196 skipped**, every failure a test that needs a solver and never
    said so.
    """
    by_job = {lane.job: lane for lane in _lanes.lanes()}
    assert job in by_job, (
        f"the {config} configuration is supposed to be covered by {job!r}, "
        f"which ci.yml does not have"
    )
    lane = by_job[job]
    assert lane.whole_suite, f"{job} does not run the whole suite"
    assert lane.solvers == (config[1] == "solvers"), (
        f"{job} is meant to be the {config[1]} lane and ci.yml provisions it "
        f"the other way"
    )
    assert (lane.jax != "absent") == (config[0] == "jax"), (
        f"{job} is meant to be the {config[0]} lane and ci.yml provisions it "
        f"the other way"
    )


def test_exactly_one_lane_runs_in_randomised_order():
    """The unenumerated backstop, and why there is exactly one.

    Test order in this repository is deterministic file order in every lane,
    so an order-dependent failure is invisible BY CONSTRUCTION — which is how
    the state-pollution incident that `tests/_state_guard.py` documents
    survived two audit rounds. One lane shuffles.

    Not more than one, and not the merge-bearing lanes: a randomised lane is
    flaky by design and names a symptom rather than a culprit, so it belongs
    beside the state guard's inventory rather than instead of it. Its job is
    to find the pollution nobody enumerated; the inventory's job is to say
    who.
    """
    shuffled = [lane.job for lane in _lanes.lanes() if lane.random_order]
    assert shuffled == ["random-order"], (
        f"expected exactly the `random-order` lane to shuffle, got {shuffled}. "
        f"A second shuffled lane doubles the flakiness for no new information; "
        f"none at all and order-dependent pollution is invisible again."
    )
    for lane in _lanes.lanes():
        if lane.job in _lanes.SERIES_BEARING:
            assert not lane.random_order, (
                f"{lane.job} is a merge-bearing lane and must not shuffle"
            )
