# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The falsification probe: it fires, it declines out loud, and it never
reads as confirmation.

Scope of the fixtures is ``pow`` and ``scatter``, deliberately. ``scatter``
is the primitive currently sitting behind ``VERIFIED_BARRED_PRIMITIVES`` —
a hand-maintained policy the probe is the beginning of replacing — and
``pow`` is the row with the known-hard sampling story, because its emission
is ``aux**q == x**p`` and the discriminating inputs are perfect q-th
powers rather than anything uniform sampling reaches.

**FOUR PROPERTIES ARE PINNED, AND THEY ARE NOT THE SAME PROPERTY.**

*It fires.* An instrument nobody has seen fire is not an instrument, and
this repository has paid for that lesson already (B13). No strategy is in
the budget on the strength of an argument — but the measurement did NOT
come out the flattering way, and the tests say so. Two strategies
(``tight``, ``ulp``) have a fixture no other strategy reaches. The other
three do not: ``tight`` is a margin minimiser and therefore general enough
to reach everything ``endpoints`` and ``exact`` reach, and over the whole
live corpus it does so in FEWER total executions than the five strategies
together. What keeps the others in is cost on particular shapes, which is
a weaker claim than reach and is asserted as the weaker claim.

*It declines out loud.* The skip rate is a first-class result. A probe
that silently declined most of what it was pointed at would read as
coverage while doing nothing, which is worse than no probe, so the decline
reasons are counted, surfaced in the stamp line, and asserted here.

*A null result is not confirmation.* This is the property most easily lost
in a later edit, because "0 violations" is such a natural thing to render
as good news. :func:`test_the_stamp_line_refuses_to_read_as_confirmation`
holds the wording to a statement about WORK DONE.

*A firing raises.* It does not become a REFUTED and it does not become an
UNKNOWN with a note. The argument, and the two rejected dispositions, are
in ``stelling/falsify.py``'s docstring; what is pinned here is the
behaviour, including that the exception carries the report.

**WHERE THE ORACLE COMES FROM.** Every fixture's truth is worked out by
hand in its own docstring or comment, from the declared box and the
arithmetic — never by asking stelling. A test that took the analysis's
word for whether an obligation was true would be measuring the probe
against the thing the probe exists to doubt.
"""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from stelling.falsify import (  # noqa: E402
    SEED_LABEL,
    STRATEGIES,
    Declaration,
    ProbeReport,
    VerifiedFalsified,
    _window,
    probe,
)
from stelling.harness import any_array, assert_, assume  # noqa: E402

SEG = np.asarray([0, 0, 0, 1])  # three duplicates into segment 0, one into 1


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def run(harness, *, strategies=STRATEGIES, semantics="real", n=1):
    """Probe ``harness`` as if the analysis had discharged everything.

    The statuses are supplied rather than propagated, which is the point:
    the probe is being asked to break a claim, and the claim is an input
    to it. Returns ``(falsification_or_None, report)``.
    """
    try:
        report = probe(
            harness,
            statuses=["discharged"] * n,
            strategies=strategies,
            semantics=semantics,
        )
        return None, report
    except VerifiedFalsified as exc:
        return exc.report.falsification, exc.report


# --------------------------------------------------------------- fixtures
#
# Each carries its own hand-worked oracle. Where a docstring below says
# "sole finder", that is asserted by a test rather than claimed — and for
# two of the five it turned out to be false, which the tests record.


def pow_endpoint():
    """``x**2 <= 80`` over ``[0, 9]``. FALSE at 9 (81 > 80). Cheapest
    finder: ``endpoints`` — the violation is exactly at the declared
    corner. NOT a sole finder: ``exact`` samples the integers of the box
    and 9 is one of them, and ``tight`` minimises straight onto it."""
    x = any_array((), "float64", (0.0, 9.0))
    return assert_(jnp.power(x, 2.0) <= 80.0)


def pow_perfect_square_only():
    """``|x**2 - 49| >= 0.5`` over ``[0, 10]``. FALSE only for x within
    0.036 of 7 — no endpoint, no midpoint, no power of two lands there,
    and a 0.7%-wide band is not something 8 uniform draws find. The
    integer 7 does, which is the exactness fact the ``pow`` row turns on.
    Cheapest finder: ``exact`` (measured, less than half the executions
    ``tight`` needs); not a sole finder, because ``tight`` gets there
    too."""
    x = any_array((), "float64", (0.0, 10.0))
    return assert_(jnp.abs(jnp.power(x, 2.0) - 49.0) >= 0.5)


def pow_interior_band():
    """``(x - 3.7)**2 >= 0.01`` over ``[0, 10]``. FALSE only on
    ``(3.6, 3.8)``: no endpoint, no integer, no power of two. Reached by
    driving the margin DOWN, which is what ``tight`` does. Sole finder:
    ``tight``."""
    x = any_array((), "float64", (0.0, 10.0))
    return assert_(jnp.power(x - 3.7, 2.0) >= 0.01)


MID = 4.0
MID_NEXT = math.nextafter(MID, math.inf)
HALF_ULP = (MID_NEXT - MID) / 2


def ulp_only():
    """``|x - MID_NEXT| >= half an ulp`` over ``[0, 8]``. FALSE at exactly
    ONE float, ``nextafter(4.0)``. ``endpoints`` samples lo, hi, mid, 0 and
    the ulp-neighbours of lo and hi — never the neighbour of the MIDPOINT.
    ``tight`` converges onto 4.0 itself, where the predicate holds. Only a
    last-representable-step from where the predicate goes tight reaches it.
    Sole finder: ``ulp``."""
    x = any_array((), "float64", (0.0, 8.0))
    return assert_(jnp.abs(x - MID_NEXT) >= HALF_ULP)


def scatter_dup_accumulate():
    """``segment_sum`` with THREE duplicate indices into segment 0, over
    ``[0, 1]**4``; ``s[0] <= 2.5`` is FALSE when all three are near 1
    (sum 3). Duplicate accumulation is the whole reason ``scatter-add`` is
    its own row, and a constant fill at the upper corner exposes it."""
    d = any_array((4,), "float64", (0.0, 1.0))
    s = jax.ops.segment_sum(d, jnp.asarray(SEG), num_segments=2)
    return assert_(s[0] <= 2.5)


def scatter_spike():
    """The same segmentation; ``s[1] <= 0.5`` is FALSE whenever the single
    element routed to segment 1 exceeds 0.5. Only a SPIKE reaches it — a
    fill that is uniform across the array either violates segment 0's
    bound first or misses entirely."""
    d = any_array((4,), "float64", (0.0, 1.0))
    s = jax.ops.segment_sum(d, jnp.asarray(SEG), num_segments=2)
    return assert_(s[1] <= 0.5)


def scatter_set_row():
    """``x.at[1].set(v)`` — the SET form, the one under the VERIFIED bar.
    ``[1] <= 5.5`` is FALSE for every ``v > 5.5``, and v is declared
    ``[5, 6]``, so it is false on rather more than half the box."""
    x = any_array((3,), "float64", (0.0, 1.0))
    v = any_array((), "float64", (5.0, 6.0))
    return assert_(x.at[1].set(v)[1] <= 5.5)


def scatter_true():
    """The control: same shape, ``s[0] <= 3.0``, TRUE at every point of
    ``[0, 1]**4`` because three elements of at most 1 sum to at most 3."""
    d = any_array((4,), "float64", (0.0, 1.0))
    s = jax.ops.segment_sum(d, jnp.asarray(SEG), num_segments=2)
    return assert_(s[0] <= 3.0)


def pow_true():
    """The control: ``x**2 <= 81`` over ``[0, 9]``, TRUE at every point."""
    x = any_array((), "float64", (0.0, 9.0))
    return assert_(jnp.power(x, 2.0) <= 81.0)


# THE FIXTURES THE PROBE CAN STILL FIRE ON, and it used to be six.
#
# The fire condition now admits a violation ONLY when an exact test refutes
# it (:func:`stelling.falsify._confirm`); an abstention declines. That is
# what stopped the Kahan false alarm reaching through `jnp.where` and a
# fractional `pow`, and it is not free: a program with a step the rational
# replay cannot read is a program this probe cannot fire on, however false
# the obligation is. The three ``scatter`` fixtures below are exactly that
# case and they have moved to :data:`DECLINED_FOR_WANT_OF_AN_EXACT_READING`
# rather than being deleted, because the reach this instrument LOST is a
# fact about it and belongs in its own test file.
LIVE = [
    ("pow_endpoint", pow_endpoint),
    ("pow_perfect_square_only", pow_perfect_square_only),
    ("pow_interior_band", pow_interior_band),
]

# FIXTURES WHOSE OBLIGATION IS FALSE AT A DECLARED POINT AND WHICH THE
# PROBE NO LONGER REPORTS, with the primitive that costs each one.
#
# All three are ``scatter``-family, and the abstention is NOT because a
# scatter has no exact rational reading -- it plainly does, it moves and
# adds exact values -- but because ``_MOVEMENT`` and the arithmetic tables
# in ``falsify.py`` do not contain one. That distinction is the whole
# point of the test below,
# :func:`test_the_exactness_requirement_costs_these_fixtures_and_names_it`
# -- a reader has to be able to tell "this instrument
# cannot see irrational arithmetic", which is inherent, from "this
# instrument has not been taught ``scatter`` yet", which is a table entry.
DECLINED_FOR_WANT_OF_AN_EXACT_READING = [
    ("scatter_dup_accumulate", scatter_dup_accumulate, "scatter-add"),
    ("scatter_spike", scatter_spike, "scatter-add"),
    ("scatter_set_row", scatter_set_row, "scatter"),
]

# The strategies each strategy is measured against below. Two of the five
# turn out to have a fixture NO other strategy reaches; the other two do
# not, and are kept for a different and measured reason. See the two tests
# that say so.
SOLE_FINDER = [
    ("tight", pow_interior_band),
]

# Fixtures reachable by more than one strategy, used to measure COST
# rather than reach: (fixture, cheap strategy, expensive strategy).
CHEAPER = [
    ("pow_perfect_square_only", pow_perfect_square_only, "exact", "tight"),
]


# ------------------------------------------------------------- it FIRES


@pytest.mark.parametrize("name,harness", LIVE, ids=[n for n, _ in LIVE])
def test_the_probe_finds_a_violation_that_is_there(name, harness):
    """Every fixture above is false somewhere in its declared box."""
    found, report = run(harness)
    assert found is not None, (
        f"the probe found nothing on {name}, whose obligation is false by "
        f"hand over its declared box. It executed {report.points_executed} "
        f"point(s) with skips {report.skips}; if the skip rate is high the "
        f"defect is in the sampler's reach, and if it is zero the defect is "
        f"in the strategies."
    )
    assert found.obligation_position == 0
    assert found.strategy in STRATEGIES or found.strategy == SEED_LABEL


@pytest.mark.parametrize(
    "name,harness,primitive",
    DECLINED_FOR_WANT_OF_AN_EXACT_READING,
    ids=[n for n, _, _ in DECLINED_FOR_WANT_OF_AN_EXACT_READING],
)
def test_the_exactness_requirement_costs_these_fixtures_and_names_it(
    name, harness, primitive
):
    """THE PRICE OF THE FIRE CONDITION, DRIVEN RATHER THAN ASSERTED.

    Each of these obligations is false by hand at a declared point, the
    probe EXECUTES that violation, and it declines to report it -- because
    only an exact test may admit a firing and the rational replay has no
    reading of ``scatter``. Three of the six fixtures this file's live
    corpus used to hold.

    Written as a test rather than as a note because the two things a
    reader needs are both facts that can rot: that the violation is still
    reached (if it stopped being reached, the reason would no longer be
    the fire condition) and that the reason is still the missing table
    entry (if ``scatter`` is ever added, this test goes red and the
    fixture goes back into :data:`LIVE`, which is the correct way for that
    change to announce itself).
    """
    found, report = run(harness)
    assert found is None, (
        f"{name} fires again, which is better news than this test records: "
        f"move it back into LIVE and rewrite the reach paragraphs that "
        f"quote three fixtures"
    )
    assert report.violations_seen > 0, (
        f"{name} no longer even REACHES its violation, so this test is no "
        f"longer measuring the fire condition -- it is measuring a sampler "
        f"that stopped looking. skips {report.skips}"
    )
    assert dict(report.skips).get("no-exact-reading-of-this-program") == (
        report.points_declined
    ), report.skips
    reasons = dict(report.abstentions)
    assert any(primitive in text for text in reasons), (
        f"{name} declined for some reason other than the missing "
        f"{primitive!r} reading: {reasons}"
    )


@pytest.mark.parametrize(
    "strategy,harness", SOLE_FINDER, ids=[s for s, _ in SOLE_FINDER]
)
def test_a_strategy_with_a_fixture_no_other_strategy_reaches(strategy, harness):
    """``tight`` is load-bearing on REACH, and this measures it.

    Run alone it finds the interior band; every other strategy run alone
    does not. The negative half is asserted rather than assumed, because a
    fixture that turns out to be reachable by something cheaper is not
    evidence for the expensive thing.

    ``tight`` earned this test the hard way. Its first implementation was
    a bisection toward the box ends, and bisection needs a SIGN CHANGE to
    bracket -- which on a VERIFIED never exists, since the margin is
    positive everywhere the analysis looked. Measured, it hit 0 of 4
    interior fixtures while spending the largest budget of any strategy.
    Replacing it with a margin MINIMISER (no sign change required) took it
    to 3 of 4 at half the cost.
    """
    found, report = run(harness, strategies=(strategy,))
    assert found is not None and found.strategy == strategy, (
        f"{strategy!r} alone did not reach the fixture written for it "
        f"(got {found and found.strategy!r} after "
        f"{report.points_executed} point(s), skips {report.skips})"
    )
    for other in STRATEGIES:
        if other == strategy:
            continue
        got, _ = run(harness, strategies=(other,))
        assert got is None or got.strategy != other, (
            f"{other!r} also reaches the fixture built to isolate "
            f"{strategy!r}, so that fixture no longer shows {strategy!r} "
            f"is load-bearing"
        )


@pytest.mark.parametrize(
    "name,harness,cheap,dear", CHEAPER, ids=[c[0] for c in CHEAPER]
)
def test_a_cheap_strategy_is_kept_for_COST_not_for_reach(
    name, harness, cheap, dear
):
    """AND THE HONEST OTHER HALF: not every strategy is a sole finder.

    Measured on this file's corpus, ``tight`` -- a margin minimiser, and
    therefore very general -- reaches every fixture ``endpoints`` and
    ``exact`` reach. So those two are NOT justified by reach, and saying
    they were would be the easy lie here. What justifies ``exact`` is that
    it gets to the ``pow`` perfect-power shape in less than half the
    executions, and the probe runs on every VERIFIED once the flag is on.

    If that stops being true, ``exact`` should be deleted rather than
    defended.
    """
    cheap_found, cheap_report = run(harness, strategies=(cheap,))
    dear_found, dear_report = run(harness, strategies=(dear,))
    assert cheap_found is not None and cheap_found.strategy == cheap
    assert dear_found is not None, (
        f"{dear!r} no longer reaches {name}, so this test is comparing "
        f"the cost of finding something against the cost of not finding it"
    )
    assert cheap_report.points_executed < dear_report.points_executed, (
        f"{cheap!r} reached {name} in {cheap_report.points_executed} "
        f"executions and {dear!r} in {dear_report.points_executed}. The "
        f"cheap strategy is in the budget ONLY because it is cheaper; "
        f"with this red it has no remaining justification and should go."
    )


def test_endpoints_is_the_CHEAPEST_configuration_and_that_is_its_whole_case():
    """``endpoints`` is in the budget for cost, and nothing measured it.

    The batch's argument was that ``endpoints`` and ``exact`` are
    "justified by cost and not by reach, and the tests say so in those
    words". Half of that was carried: :data:`CHEAPER` has exactly one row
    and it is ``exact``. ``endpoints`` had no test of either kind, so its
    place in a budget that runs on every VERIFIED rested on a sentence.

    Measured here, over this file's whole live corpus, per single-strategy
    configuration (executions, fixtures reached out of three):

    ==========  ==========  ========
    strategy    executions  reached
    ==========  ==========  ========
    endpoints           12       1/3
    ulp                 18       1/3
    uniform             24       0/3
    exact               33       2/3
    tight               41       3/3
    all five            47       3/3
    ==========  ==========  ========

    So ``endpoints`` IS the cheapest configuration, by a margin, and it
    reaches one of the three — which is the claim, and it is a cost claim.
    It is emphatically not a reach claim: ``tight`` alone reaches
    everything ``endpoints`` does. If the cheapest column ever stops being
    ``endpoints``, the strategy should be deleted rather than defended.

    THESE NUMBERS ARE SMALLER THAN THE ONES THIS DOCSTRING USED TO CARRY
    (20/26/28/41/49/55 executions over six fixtures) and the difference is
    not a sampler change: three ``scatter`` fixtures left the live corpus
    when the fire condition stopped admitting anything an exact test had
    not refuted. See :data:`DECLINED_FOR_WANT_OF_AN_EXACT_READING`. The
    ORDER of the column is what this test is about and it is unchanged.
    """
    cost = {}
    reach = {}
    for strategy in STRATEGIES:
        cost[strategy] = sum(
            run(h, strategies=(strategy,))[1].points_executed for _, h in LIVE
        )
        reach[strategy] = sum(
            1 for _, h in LIVE if run(h, strategies=(strategy,))[0] is not None
        )

    cheapest = min(cost, key=lambda k: cost[k])
    assert cheapest == "endpoints", (
        f"the cheapest single-strategy configuration over the live corpus "
        f"is now {cheapest!r} at {cost[cheapest]} execution(s), not "
        f"'endpoints' at {cost['endpoints']}. Cost is the ONLY argument "
        f"`endpoints` has: {cost}"
    )
    assert 0 < reach["endpoints"] < len(LIVE), (
        f"`endpoints` reached {reach['endpoints']} of {len(LIVE)}. At zero "
        f"it is an instrument nobody has seen fire; at all six it is "
        f"justified by REACH and this docstring understates it."
    )


def test_the_ENSEMBLE_does_not_dominate_its_most_general_member():
    """A MEASUREMENT THAT DID NOT COME OUT THE FLATTERING WAY.

    Over this file's whole live corpus, the ``tight``-only CONFIGURATION
    finds every fixture in fewer total executions than all five strategies
    together (41 against 47). The ensemble is not, on this corpus, better
    than its most general member.

    Read "configuration" literally, because the attribution matters and
    flatters ``tight`` if it is skipped. ``tight`` is a SEEDED phase: run
    alone it must still generate its own starting points, and those seeds
    are endpoint points, executed and counted under
    :data:`stelling.falsify.SEED_LABEL`.
    So of the three fixtures the ``tight``-only configuration reaches, one
    is actually reached by its seeds and two by the margin search itself
    -- which is what the per-strategy hit rate in
    :func:`test_a_strategy_with_a_fixture_no_other_strategy_reaches` and
    its neighbours measure, since those count only hits attributed to the
    strategy proper. What this test compares is two whole configurations
    on cost, which is the question a budget is spent against.

    That is recorded as an executed comparison rather than a sentence in a
    report, because it is the kind of fact that quietly stops being true
    (or quietly stays true while the docstring claims otherwise). What
    keeps the other strategies in is measured elsewhere in this file:
    ``ulp`` reaches a fixture ``tight`` does not, and ``exact`` reaches the
    ``pow`` shape more cheaply. Neither of those is "the ensemble wins on
    the corpus", and this test exists so nobody can read it that way.
    """
    ensemble = sum(run(h)[1].points_executed for _, h in LIVE)
    general = sum(
        run(h, strategies=("tight",))[1].points_executed for _, h in LIVE
    )
    ensemble_found = sum(1 for _, h in LIVE if run(h)[0] is not None)
    general_found = sum(
        1 for _, h in LIVE if run(h, strategies=("tight",))[0] is not None
    )

    assert ensemble_found == len(LIVE), "the ensemble must still find them all"
    assert general_found == len(LIVE), (
        f"`tight` alone no longer finds every live fixture ({general_found} "
        f"of {len(LIVE)}). That would make the ensemble load-bearing on "
        f"reach after all -- a better result than the one recorded here, "
        f"and one this docstring must be rewritten to state."
    )
    assert ensemble >= general, (
        f"the ensemble ({ensemble}) is now CHEAPER than `tight` alone "
        f"({general}) over the live corpus. Also a better result than the "
        f"one recorded here; rewrite this docstring rather than deleting "
        f"the test."
    )


def test_the_ulp_strategy_can_fire():
    """``ulp`` gets its own test, and it fires under BOTH semantics now.

    THIS ASSERTION USED TO RUN THE OTHER WAY, and the change is a
    correction rather than a relaxation. The fixture's violation is a
    single float, so it is not stable under an ulp perturbation of the
    INPUT — and while ulp-stability of the input was the real-semantics
    filter, the point was declined as ``precision-ambiguous``.

    But ``nextafter(4.0)`` is an exactly representable rational and
    ``|x - MID_NEXT| >= HALF_ULP`` is FALSE there over ℝ, not only in
    floats: the obligation really is violated at a declared point, so a
    discharge really would be unsound, and the old filter was LOSING a
    refutation rather than suppressing a false alarm. Exact-rational
    replay says so, and the firing records which test admitted it.

    The pair that shows the filter is a filter and not an off switch is
    now :func:`test_the_real_filter_declines_a_violation_that_is_a_pure_
    float_artefact`, whose fixture is true over ℚ at every point.
    """
    found, _ = run(ulp_only, strategies=("ulp",), semantics="ieee")
    assert found is not None and found.strategy == "ulp", (
        "the ulp strategy did not fire on a violation placed one "
        "representable step from where the predicate goes tight; it has "
        "no other fixture, so with this red it is an instrument nobody "
        "has seen fire"
    )
    real_found, report = run(ulp_only, strategies=("ulp",), semantics="real")
    assert real_found is not None, (
        f"the obligation is FALSE over ℝ at nextafter(4.0), which is a "
        f"declared point, so a real-semantics discharge of it is unsound "
        f"and the probe must report it; skips were {report.skips}"
    )
    assert (
        real_found.adjudication == "exact-replay-refutes-over-the-rationals"
    ), (
        f"the firing must be admitted by the exact-rational replay and say "
        f"so, not by the ulp proxy: {real_found.adjudication!r}"
    )


@pytest.mark.parametrize(
    "name,harness",
    [("pow_true", pow_true), ("scatter_true", scatter_true)],
)
def test_the_probe_finds_nothing_where_there_is_nothing(name, harness):
    """The controls. A probe that fires on true obligations is noise."""
    found, report = run(harness)
    assert found is None, (
        f"the probe FIRED on {name}, whose obligation is true at every "
        f"point of its declared box by hand: {found}"
    )
    assert report.points_executed > 0, "and it did look"


# -------------------------------------------------------- it DECLINES loudly


def test_an_unbounded_declaration_is_declined_not_silently_skipped():
    """``(-inf, inf)`` is a legal declaration and an unsampleable one."""

    def harness():
        x = any_array((), "float64", (-math.inf, math.inf))
        return assert_(x * x >= 0.0)

    found, report = run(harness)
    assert found is None
    assert report.declined is not None
    assert "unbounded" in report.declined
    assert report.skip_rate == 1.0, (
        "a probe that sampled nothing must report a skip rate of 1.0; "
        "anything less lets 'declined everything' read like 'found nothing'"
    )
    assert "DECLINED" in report.stamp_line()
    assert "not evidence" in report.stamp_line()


def test_points_outside_the_assumed_region_are_counted_not_hidden():
    """An assume makes part of the box inadmissible, and that is a SKIP.

    ``x >= 9`` over ``[0, 10]``: points below 9 are not counterexamples to
    anything, and a probe that quietly dropped them would report a
    coverage it did not have.
    """

    def harness():
        x = any_array((), "float64", (0.0, 10.0))
        assume(x >= 9.0)
        return assert_(x <= 9.5)  # FALSE on (9.5, 10]

    found, report = run(harness)
    assert found is not None, "the violation on (9.5, 10] is reachable"
    assert dict(report.skips).get("assume-unsatisfied"), (
        f"points below the assume must be counted as skips; got "
        f"{report.skips}"
    )
    assert report.points_admissible < report.points_built
    assert 0.0 < report.skip_rate < 1.0


def test_an_empty_integer_box_is_declined_with_its_own_reason():
    """``int32`` over ``(0.2, 0.8)`` admits no integer at all.

    The declared-set rule for integers is RE-DERIVED in the probe rather
    than imported from the propagator, precisely so the two can disagree;
    this pins the probe's side of it directly.
    """
    decl = Declaration(position=0, shape=(), dtype="int32", lo=0.2, hi=0.8)
    window, why = _window(decl)
    assert window is None and why == "empty-integer-box"

    wide = Declaration(position=0, shape=(), dtype="int32", lo=0.2, hi=2.8)
    window, why = _window(wide)
    assert (window, why) == ((1, 2), None), (
        "int32 over (0.2, 2.8) declares exactly {1, 2}; the probe's own "
        "membership rule has drifted from that"
    )


def test_the_skip_rate_denominator_is_points_BUILT():
    """Because the hidden work is the built-and-discarded point.

    A probe that built a thousand points and used one would have a skip
    rate of ~1.0, which is the honest number. Measuring against points
    EXECUTED would have reported 0.0 and read like full coverage.
    """
    r = ProbeReport(points_built=10, points_executed=10, points_admissible=2)
    assert r.skip_rate == pytest.approx(0.8)
    assert ProbeReport(declined="nothing to do").skip_rate == 1.0
    assert ProbeReport().skip_rate == 1.0


# ------------------------------------------ a null result is NOT confirmation


def test_the_stamp_line_refuses_to_read_as_confirmation():
    """THE WORDING IS A CONSTRAINT, NOT PROSE.

    The line a VERIFIED grows when the probe ran must say what was DONE
    and must carry its own disclaimer. It must not contain a word that a
    reader could take as the verdict having been strengthened — and the
    disclaimer must be in the same sentence, because a caveat a reader
    reaches after the number it qualifies is a caveat that arrives too
    late.
    """
    _, report = run(pow_true)
    line = report.stamp_line()

    assert "point(s) executed" in line
    assert "NOT" in line and "EVIDENCE" in line
    assert "can only refute" in line

    lowered = line.lower()
    for word in (
        "confirm",
        "confidence",
        "validated",
        "corroborat",
        "sound",
        "clean",
        "passed",
        "holds up",
        "stronger",
    ):
        assert word not in lowered, (
            f"the probe's stamp line contains {word!r}. Finding nothing "
            f"says nothing about soundness, and a VERIFIED that reads "
            f"better for having been probed is a verdict above its "
            f"evidence: {line!r}"
        )


def test_the_report_carries_no_summary_of_how_well_the_verdict_held():
    """There is no such quantity, so there must be no such field."""
    _, report = run(pow_true)
    fields = set(vars(report))
    for banned in ("confidence", "score", "coverage", "passed", "clean"):
        assert not any(banned in f for f in fields), (
            f"ProbeReport grew a field matching {banned!r}: {sorted(fields)}"
        )
    assert report.falsification is None


# --------------------------------------------------------- it RAISES


def test_a_firing_raises_and_carries_the_report():
    """The disposition. Not a status, and not a note on one."""
    with pytest.raises(VerifiedFalsified) as caught:
        probe(pow_endpoint, statuses=["discharged"])
    exc = caught.value
    assert isinstance(exc, AssertionError), (
        "VerifiedFalsified must be an AssertionError so that a batch "
        "caller's bare `except Exception` does not swallow a soundness "
        "event by accident"
    )
    assert exc.report.falsification is not None
    text = str(exc)
    assert "UNSOUND" in text
    assert "No verdict is returned" in text
    # it must say whose defect this is, because the whole risk of the other
    # two dispositions was reporting stelling's defect as the user's
    assert "not a finding about the program under test" in text


def test_the_probe_does_not_attack_what_the_analysis_did_not_claim():
    """An obligation the analysis left unknown is not a discharge.

    Falsifying it would be reporting a defect that is not there: the tool
    said it could not decide, and it was right.
    """
    report = probe(pow_endpoint, statuses=["unknown"])
    assert report.falsification is None
    assert report.declined == "no obligation was discharged"


def test_an_unknown_strategy_name_raises_eagerly():
    """Same discipline as every other dial in the pipeline."""
    with pytest.raises(ValueError, match="unknown falsification strateg"):
        probe(pow_true, statuses=["discharged"], strategies=("corners",))


# --------------------------------------------------- it executes the PROGRAM


def test_the_program_raising_is_a_skip_and_never_a_violation():
    """A declared box may contain inputs the program refuses.

    That is a fact about the point, not about the verdict, and treating it
    as a counterexample would let the probe fire on programs that are
    merely partial.
    """

    def harness():
        x = any_array((), "float64", (-4.0, 4.0))
        # jnp does not raise on a negative base with a fractional exponent
        # (it returns nan), so the point of this fixture is the nan path:
        # a nan comparison is FALSE, and that must not be read as a
        # violation of a claim about the reals.
        return assert_(jnp.power(x, 0.5) >= 0.0)

    found, report = run(harness)
    assert report.points_executed > 0
    if found is not None:
        # if it does fire, it must be on a real value and not on a nan
        assert "nan" not in found.values[0].lower()
