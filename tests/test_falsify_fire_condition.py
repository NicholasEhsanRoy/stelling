# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE FIRE CONDITION: when an executed violation may be called UNSOUND.

The probe raises, and the message it raises with says *"stelling is
UNSOUND at this query"*. That is the one message a reader acts on, so the
test that decides whether it is emitted is the most consequential test in
this feature — and the batch that shipped the probe got it wrong in a way
four lines of ordinary numerical code reached:

    y = any_array((), "float64", (0.0, 2.0))
    s = 1e16
    assert_((s + y) - s <= y)          # the Kahan/Neumaier shape

``1e16`` is exactly 10**16, float64's spacing there is 2.0, and over ℝ the
expression is ``y`` — so the obligation is TRUE, both solvers answered
unsat, and the verdict was RIGHT. The probe raised on it anyway, because
the ``real``-semantics filter tested ULP-STABILITY OF THE INPUT, which is
a good proxy for a one-ulp artefact and no proxy at all for coarse
quantisation: the violating band here is about 4.5e15 ulps wide and every
point of it is stable.

**A false-alarming soundness alarm is worse than no alarm** — it reports
our defect as the caller's, in the message nobody can ignore.

So this file pins the fire condition from both sides, because either side
alone is easy to satisfy with an off switch:

* a violation that is a pure float artefact is DECLINED and counted
  (:func:`test_the_kahan_compensation_shape_is_not_a_soundness_event`,
  and the one-ulp family beside it);
* a violation that is real over ℚ is REPORTED, including the ones the old
  proxy lost (:func:`test_a_violation_that_is_real_over_the_rationals_
  still_fires`), and the integer path keeps catching a runtime wrap, which
  is the case exact-rational replay would get WRONG if it were allowed to
  run there (:func:`test_the_integer_branch_is_not_a_rational_replay`).

Every firing also records WHICH test admitted it
(:attr:`stelling.falsify.Falsification.adjudication`), because a firing
adjudicated by exact-rational replay and one adjudicated by the surviving
ulp proxy are not the same claim.
"""

from __future__ import annotations

import math
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import stelling  # noqa: E402
from stelling.falsify import (  # noqa: E402
    DECLINE_REASONS,
    FALSIFY_MODES,
    SEED_LABEL,
    STRATEGIES,
    Declaration,
    VerifiedFalsified,
    _admissible,
    _window,
    probe,
)
from stelling.harness import any_array, assert_, assume  # noqa: E402
from stelling.preconditions import check  # noqa: E402

PROBE_SRC = pathlib.Path(stelling.__file__).resolve().parent / "falsify.py"


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def attack(harness, *, semantics="real", n=1, **kw):
    """Probe ``harness`` as if the analysis had discharged everything."""
    try:
        return None, probe(
            harness, statuses=["discharged"] * n, semantics=semantics, **kw
        )
    except VerifiedFalsified as exc:
        return exc.report.falsification, exc.report


# ------------------------------------------------- the false alarm, both ways


def kahan(op):
    """``(s + y) - s`` against ``y``, the compensated-summation shape.

    TRUE over ℝ for either comparison, because ``(s + y) - s`` IS ``y``
    over ℝ. FALSE in float64 on a band thousands of ulps wide, because
    ``s = 1e16`` is above 2**53 and absorbs y's low bits.
    """

    def harness():
        y = any_array((), "float64", (0.0, 2.0))
        s = 1e16
        return assert_(op((s + y) - s, y))

    return harness


@pytest.mark.parametrize(
    "name,op",
    [("le", lambda a, b: a <= b), ("ge", lambda a, b: a >= b)],
)
def test_the_kahan_compensation_shape_is_not_a_soundness_event(name, op):
    """THE HEADLINE. A correct VERIFIED must not be called UNSOUND.

    Both comparison directions are driven, because the defect had an
    instance in each: ``<=`` fired at y = 1.6888437030500962 and ``>=`` at
    y = 0.5.
    """
    found, report = attack(kahan(op))
    assert found is None, (
        f"the probe reported a violation of an obligation that is TRUE "
        f"over ℝ at every declared point: {found and found.render()}"
    )
    assert report.violations_seen > 0, (
        "the float program really is false on a wide band here, so a run "
        "that saw no violation at all is not evidence that the fire "
        "condition works — it is evidence the sampler stopped looking"
    )
    assert dict(report.adjudications) == {
        "exact-replay-holds-over-the-rationals": report.violations_seen
    }, (
        f"every one of those violations must be declined by the EXACT "
        f"replay, not by the ulp proxy (which cannot see this shape): "
        f"{report.adjudications}"
    )
    assert report.points_declined > 0
    assert report.skip_rate > 0.0, (
        "a run that declined executed violations must not report a skip "
        "rate of 0.0; that reads as 'nothing was skipped' on exactly the "
        "points whose results were dropped"
    )


def test_the_full_check_returns_a_VERIFIED_rather_than_raising():
    """And it must not raise through the public door either.

    The unit-level test above drives :func:`probe` with a supplied status.
    This drives the real pipeline, with the solvers, because that is the
    path the defect was reported on and because the note the VERIFIED
    grows is part of the fix.
    """
    verdict = check(
        kahan(lambda a, b: a <= b),
        vacuity_mode="inputs-only",
        semantics="real",
        solver_timeout_ms=8000,
        falsify="sample",
    )
    assert verdict.status == "VERIFIED"
    line = [n for n in verdict.notes if "falsification probe" in n]
    assert line, f"the probe left no note: {verdict.notes}"
    assert "WERE DECLINED, NOT ABSENT" in line[0], (
        f"a run that executed violations and declined them must say so in "
        f"the same sentence as the counts: {line[0]!r}"
    )


@pytest.mark.parametrize(
    "name,harness",
    [
        (
            "(x/3)*3 <= x",
            lambda: assert_(
                (any_array((), "float64", (0.0, 2.0)) / 3.0) * 3.0
                <= any_array((), "float64", (0.0, 0.0))
            ),
        ),
    ],
)
def test_the_one_ulp_family_is_still_declined(name, harness):
    """The shape the ORIGINAL proxy was right about is still declined.

    The replacement has to keep every decline the proxy earned, not only
    stop the alarms it missed — a fire condition that traded one blind
    spot for another would not be an improvement.
    """
    # a two-declaration spelling would compare different variables; use the
    # single-variable form instead
    def one_ulp():
        x = any_array((), "float64", (0.0, 2.0))
        return assert_((x / 3.0) * 3.0 <= x)

    found, report = attack(one_ulp)
    assert found is None, f"declined shape fired: {found and found.render()}"
    if report.violations_seen:
        assert "exact-replay-holds-over-the-rationals" in dict(
            report.adjudications
        )


def test_the_skip_rate_counts_a_DECLINED_VIOLATION():
    """``x + 1.0 > x`` over ``[0, 2**54]``, and the number it used to hide.

    True over ℝ everywhere; false in float64 above ``2**53``, where the
    ``+ 1.0`` is absorbed. So the probe executes a violation, declines it,
    and reports.

    What it reported before was *120 points executed, skip rate 0.0000* —
    on a run whose 32 most interesting points were the declined ones. The
    denominator was points BUILT and the numerator counted only points the
    sampler could not use, so a violation the FIRE CONDITION would not
    stand behind cost nothing. It costs its own point now, and the stamp
    line names the count in the same sentence as the totals rather than
    after a phrase that reads as "nothing was there".
    """
    def absorbed():
        x = any_array((), "float64", (0.0, 2.0**54))
        return assert_(x + 1.0 > x)

    found, report = attack(absorbed)
    assert found is None, found and found.render()
    assert report.violations_seen > 0
    assert report.points_declined == report.violations_seen
    assert report.skip_rate > 0.0, (
        f"{report.points_declined} executed violation(s) were declined and "
        f"the skip rate is {report.skip_rate}"
    )
    line = report.stamp_line()
    assert f"{report.points_declined} EXECUTED VIOLATION(S) WERE DECLINED" in line
    assert "NOT ABSENT" in line
    assert "NO VIOLATION WAS FOUND" not in line, (
        f"a run that executed and declined violations must not report that "
        f"no violation was found: {line!r}"
    )


# --------------------------------------------- and it still fires when it must


def lying_pow():
    """``x**2 <= 40`` over ``[0, 9]``: FALSE at 9 (81 > 40), over ℝ."""
    x = any_array((), "float64", (0.0, 9.0))
    return assert_(jnp.power(x, 2.0) <= 40.0)


def lying_sum():
    """``sum(v) <= 15`` over ``[0, 10]**4``: FALSE at the upper corner."""
    v = any_array((4,), "float64", (0.0, 10.0))
    return assert_(jnp.sum(v) <= 15.0)


def lying_mul():
    """``x*y <= 50`` over ``[0, 10]**2``: FALSE at (10, 10), over ℝ."""
    x = any_array((), "float64", (0.0, 10.0))
    y = any_array((), "float64", (0.0, 10.0))
    return assert_(x * y <= 50.0)


@pytest.mark.parametrize(
    "name,harness",
    [
        ("pow", lying_pow),
        ("reduce_sum", lying_sum),
        ("mul", lying_mul),
    ],
)
def test_a_violation_that_is_real_over_the_rationals_still_fires(
    name, harness
):
    """Each of these is false over ℝ by hand at a declared point.

    The oracle is worked out in the fixture docstrings from the declared
    box and the arithmetic — never by asking stelling, which is the thing
    under doubt.
    """
    found, report = attack(harness)
    assert found is not None, (
        f"{name}: the obligation is FALSE over ℝ at a declared point and "
        f"the probe did not report it after {report.points_executed} "
        f"execution(s), skips {report.skips}"
    )
    assert found.adjudication == "exact-replay-refutes-over-the-rationals", (
        f"{name}: the firing must be admitted by exact-rational replay, "
        f"which is a proof about ℝ, not by the ulp proxy, which is not: "
        f"{found.adjudication!r}"
    )
    assert "FALSE over ℚ" in found.detail


def test_the_integer_branch_is_not_a_rational_replay():
    """``int8`` arithmetic WRAPS, and ℚ arithmetic does not.

    ``x + y >= 0`` over ``int8 [0, 100]**2`` is true over ℤ and false in
    the program: 100 + 100 wraps to -56. Exact-rational replay would say
    TRUE and decline it, suppressing the one runtime-wrap catch this probe
    was measured to have — so the all-integral branch short-circuits
    BEFORE the replay, exactly as it did before, and the firing records
    that it was integer arithmetic that admitted it.
    """
    def wraps():
        x = any_array((), "int8", (0, 100))
        y = any_array((), "int8", (0, 100))
        return assert_(x + y >= 0)

    found, report = attack(wraps)
    assert found is not None, (
        f"the int8 runtime wrap is no longer caught; skips {report.skips}"
    )
    assert found.adjudication == "exact-integer-arithmetic", found.adjudication


def test_an_unreplayable_primitive_falls_back_to_the_proxy_AND_SAYS_SO():
    """``exp`` is irrational at every rational but 0, so the replay abstains.

    The fall-back is the point: the fire condition degrades to the weaker
    test rather than declining everything it cannot prove. What must never
    happen is that it degrades SILENTLY, so the adjudication names the
    proxy and the abstention reason travels in the detail.
    """
    def transcendental():
        x = any_array((), "float64", (0.0, 2.0))
        return assert_(jnp.exp(x) <= 2.0)  # exp(2) = 7.389...: FALSE

    found, report = attack(transcendental)
    assert found is not None, f"skips {report.skips}"
    assert found.adjudication.startswith("ulp-proxy"), found.adjudication
    assert "exact-rational replay abstained" in found.detail, found.detail


def test_a_replay_too_expensive_to_run_abstains_before_it_starts():
    """The rational replay is a Python loop, and the cost is bounded.

    Measured at 2.8 - 10.5 microseconds per element visited, so a
    declaration large enough turns a fire condition into a hang. The
    budget is checked against a cost read off the AVALS before any
    arithmetic happens, and exceeding it is an abstention like any other:
    the ulp proxy decides and the report says the proxy decided.

    Driven rather than asserted, because a budget nothing has ever
    exceeded is a budget nobody has checked the units of.
    """
    import stelling.falsify as F

    n = F.REPLAY_ELEMENT_BUDGET  # one element per equation output already
    # exceeds it at this width

    def wide():
        v = any_array((n,), "float64", (0.0, 1.0))
        return assert_(jnp.sum(v) <= 0.5)  # FALSE at any point but 0

    census = F._census(wide)
    assert census.replay_cost > F.REPLAY_ELEMENT_BUDGET, census.replay_cost
    with pytest.raises(F._Unreplayable, match="budget"):
        F._replay(census, (np.full((n,), 0.9),))


def test_the_replay_budget_does_not_bind_on_an_ordinary_declaration():
    """The other half: the budget must not be silently on all the time."""
    import stelling.falsify as F

    def ordinary():
        v = any_array((64,), "float64", (0.0, 1.0))
        return assert_(jnp.sum(v) <= 0.5)

    census = F._census(ordinary)
    assert census.replay_cost < F.REPLAY_ELEMENT_BUDGET
    found, _ = attack(ordinary)
    assert found is not None
    assert found.adjudication == "exact-replay-refutes-over-the-rationals"


# ------------------------------------------------------- the firing MESSAGE


def test_the_firing_message_does_not_end_by_saying_nothing_was_found():
    """The module's most important message used to contradict itself.

    :func:`stelling.falsify._fire` appends
    :meth:`ProbeReport.stamp_line`, which had no firing branch — so every
    firing ended with *"NO VIOLATION WAS FOUND, WHICH IS NOT EVIDENCE THAT
    THERE IS NONE"* three lines under *"FALSIFICATION PROBE FIRED"*.
    """
    with pytest.raises(VerifiedFalsified) as caught:
        probe(lying_pow, statuses=["discharged"])
    text = str(caught.value)
    assert "FALSIFICATION PROBE FIRED" in text
    assert "NO VIOLATION WAS FOUND" not in text, (
        f"the firing message still says no violation was found:\n{text}"
    )
    assert "A VIOLATION WAS FOUND AND IS REPORTED ABOVE" in text
    assert caught.value.report.stamp_line() in text


def test_the_stamp_line_still_refuses_to_read_as_confirmation_when_it_fires():
    """The firing branch is under the same wording constraint as the others."""
    with pytest.raises(VerifiedFalsified) as caught:
        probe(lying_pow, statuses=["discharged"])
    lowered = caught.value.report.stamp_line().lower()
    for word in ("confidence", "validated", "corroborat", "clean", "passed"):
        assert word not in lowered, lowered


def test_the_firing_names_which_test_admitted_the_violation():
    """A reader has to be able to tell a ℚ-proof from a proxy."""
    with pytest.raises(VerifiedFalsified) as caught:
        probe(lying_pow, statuses=["discharged"])
    assert "test:" in caught.value.report.falsification.render()


def test_the_seed_label_is_explained_where_the_user_meets_it():
    """``seed`` reaches the firing message and is not in :data:`STRATEGIES`.

    A reader who goes looking for it in the strategy list will not find
    it, so the message it appears in has to say what it is.
    """
    assert SEED_LABEL not in STRATEGIES
    found, _ = attack(lying_pow, strategies=("tight",))
    assert found is not None
    if found.strategy == SEED_LABEL:
        rendered = found.render()
        assert "is not one" in rendered, rendered


# --------------------------------------------------- the dial and its spelling


def test_the_dial_has_ONE_definition_and_the_second_spelling_is_pinned():
    """``preconditions`` re-spells :data:`FALSIFY_MODES` and must not drift.

    It re-spells it on purpose: importing this module imports jax, and the
    dial has to be validated in a jax-less environment too. That is a good
    reason for a second spelling and no reason at all for an unpinned one
    — ``FALSIFY_MODES`` was exported, documented as the single definition,
    and used by nothing.
    """
    def trivial():
        x = any_array((), "float64", (0.0, 1.0))
        return assert_(x <= 1.0)

    from stelling.contracts import Contract, check_contract
    from stelling.inductive import check_inductive_step

    def body(state, constants):
        return {"v": state["v"]}

    contract = Contract(
        name="c",
        requires_description="x stays under its own upper bound",
        harness=trivial,
        ensures=None,
        no_ensures_reason="this contract states no guarantee",
    )
    doors = [
        lambda m: check(trivial, vacuity_mode="inputs-only", falsify=m),
        lambda m: check_contract(
            contract, vacuity_mode="inputs-only", falsify=m
        ),
        lambda m: check_inductive_step(
            body, {"v": ((0.0, 1.0), "float64")}, falsify=m
        ),
    ]
    for door in doors:
        for mode in FALSIFY_MODES:
            door(mode)
        for rejected in ("Sample", "sampled", "yes", 1, True):
            with pytest.raises(ValueError, match="falsify must be"):
                door(rejected)


def test_all_three_public_doors_can_arm_the_probe():
    """The probe used to be reachable from one of the three.

    ``contracts.check_contract`` and ``inductive.check_inductive_step``
    mint VERIFIEDs through the same ``_pipeline``; a probe whose reach
    depended on which function happened to carry the keyword would be an
    instrument bounded by an accident rather than by a decision.
    """
    from stelling.contracts import Contract, check_contract
    from stelling.inductive import check_inductive_step

    def falsely_verified():
        x = any_array((), "float64", (0.0, 9.0))
        return assert_(jnp.power(x, 2.0) <= 40.0)

    import stelling.propagate as P

    original = P.TRANSFERS["pow"]
    P.TRANSFERS["pow"] = (lambda eqn, p, ins: [ins[0]], P.TIER_SOUND)
    try:
        with pytest.raises(VerifiedFalsified):
            check(falsely_verified, vacuity_mode="inputs-only",
                  falsify="sample")
        with pytest.raises(VerifiedFalsified):
            check_contract(
                Contract(
                    name="c",
                    requires_description="x**2 stays under 40 on [0, 9]",
                    harness=falsely_verified,
                    ensures=None,
                    no_ensures_reason="this contract states no guarantee",
                ),
                vacuity_mode="inputs-only",
                falsify="sample",
            )
    finally:
        P.TRANSFERS["pow"] = original

    # the inductive door, on a step that really does leave its invariant
    def body(state, constants):
        return {"v": state["v"] * 2.0}

    with pytest.raises(VerifiedFalsified):
        original_mul = P.TRANSFERS["mul"]
        P.TRANSFERS["mul"] = (lambda eqn, p, ins: [ins[0]], P.TIER_SOUND)
        try:
            check_inductive_step(
                body, {"v": ((1.0, 2.0), "float64")}, falsify="sample"
            )
        finally:
            P.TRANSFERS["mul"] = original_mul


# ------------------------------------------------- the reasons it can decline


REASON_COVERAGE = {
    "dtype-unconstructible": "unit: _window on a dtype numpy cannot build",
    "bound-unreadable": "unit: _window on a non-numeric bound",
    "bound-nan": "unit: _window on a NaN bound",
    "unbounded-declaration": "unit: _window on an infinite endpoint",
    "empty-integer-box": "unit: _window on int32 (0.2, 0.8)",
    "empty-box": "unit: _window on lo > hi",
    "box-outside-the-dtype-range": "unit: _window on float16 (1e5, 2e5)",
    "dtype-not-sampleable": "unit: _window on bfloat16",
    "point-outside-declaration": "driven: a sampler that leaves the box",
    "program-raised": "driven: a program that raises at a declared point",
    "assume-unsatisfied": "driven: a point below the assume",
    "precision-ambiguous": "driven: a proxy-adjudicated knife edge",
    "float-rounding-artefact": "driven: the Kahan shape above",
    "assume-unsatisfied-over-the-rationals": (
        "DEFENCE IN DEPTH, no known reaching input: the assume would have "
        "to hold in floats and fail over ℚ at the same point, which needs "
        "an assume whose own evaluation rounds across the boundary"
    ),
    "no-margin-no-boundary-search": "driven: an assert with no comparison",
    "obligation-count-changed": (
        "DEFENCE IN DEPTH, no known reaching input: `_read` and `_execute` "
        "walk the same equation list, so the counts cannot diverge; the "
        "guard is what stops an IndexError on a soundness path if they "
        "ever do"
    ),
}


def test_every_decline_reason_is_declared_and_accounted_for():
    """Ten of the thirteen shipped reasons appeared in no test at all.

    A decline reason is a user-visible string in a note that is supposed
    to keep the probe honest about what it did NOT do. An unexercised one
    is a sentence nobody has read since it was written.
    """
    source = PROBE_SRC.read_text(encoding="utf-8")
    emitted = set(re.findall(r'skips\.add\("([a-z][a-z0-9-]*)"\)', source))
    emitted |= set(
        re.findall(r'return None, "([a-z][a-z0-9-]*)"', source)
    )
    emitted |= set(
        re.findall(r'^\s+"([a-z][a-z0-9-]*)",\n\s+"(?:exact|ulp|ieee)',
                   source, re.M)
    )
    unlisted = emitted - set(DECLINE_REASONS)
    assert not unlisted, (
        f"stelling/falsify.py can emit decline reason(s) {sorted(unlisted)} "
        f"that DECLINE_REASONS does not list"
    )
    uncovered = set(DECLINE_REASONS) - set(REASON_COVERAGE)
    assert not uncovered, (
        f"decline reason(s) {sorted(uncovered)} have no entry in this "
        f"file's coverage table: either drive one, or record why no input "
        f"reaches it"
    )


@pytest.mark.parametrize(
    "decl,expected",
    [
        (dict(dtype="not-a-dtype", lo=0.0, hi=1.0), "dtype-unconstructible"),
        (dict(dtype="float64", lo="zero", hi=1.0), "bound-unreadable"),
        (dict(dtype="float64", lo=math.nan, hi=1.0), "bound-nan"),
        (
            dict(dtype="float64", lo=-math.inf, hi=1.0),
            "unbounded-declaration",
        ),
        (dict(dtype="int32", lo=0.2, hi=0.8), "empty-integer-box"),
        (dict(dtype="float64", lo=2.0, hi=1.0), "empty-box"),
        (
            dict(dtype="float16", lo=1e5, hi=2e5),
            "box-outside-the-dtype-range",
        ),
        (dict(dtype="bfloat16", lo=0.0, hi=1.0), "dtype-not-sampleable"),
    ],
)
def test_the_window_declines_with_its_own_reason(decl, expected):
    """Each unsampleable declared set is named, not lumped together."""
    window, why = _window(Declaration(position=0, shape=(), **decl))
    assert (window, why) == (None, expected)


def test_the_extension_float_dtypes_are_a_NAMED_reach_gap():
    """``bfloat16`` and the ``float8`` family get ZERO reach, on purpose.

    That is where format-rounding defects are most likely and the probe
    cannot look, because numpy classifies the ml_dtypes extension types as
    ``kind == "V"`` and they carry no ``nextafter`` — and every phase of
    this sampler steps and compares in the declaration's own format. It is
    named in the module docstring's blind-spot section for the same reason
    the integer-literal wrap is: a reach gap a reader has to infer from a
    decline count is a reach gap that reads as coverage.
    """
    for name in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
        try:
            np.dtype(name)
        except TypeError:  # pragma: no cover - depends on ml_dtypes
            continue
        window, why = _window(
            Declaration(position=0, shape=(), dtype=name, lo=0.0, hi=1.0)
        )
        assert (window, why) == (None, "dtype-not-sampleable"), name
    assert "blind" in PROBE_SRC.read_text(encoding="utf-8")
    assert "bfloat16" in PROBE_SRC.read_text(encoding="utf-8"), (
        "the reach gap is not named anywhere in the module that has it"
    )


def test_the_admissibility_guard_rejects_a_point_the_sampler_should_not_build(
    monkeypatch,
):
    """0 of 30,194 points on the live corpus — so it is driven here.

    The guard exists for a sampler defect, and the sampler has none, which
    means nothing on the corpus has ever exercised it. An instrument
    nobody has seen fire is not an instrument, so this gives it a sampler
    that leaves the box and asserts the point is rejected and COUNTED.
    """
    import stelling.falsify as F

    assert _admissible(
        Declaration(position=0, shape=(), dtype="float64", lo=0.0, hi=1.0),
        np.asarray(2.0),
    ) is False

    real_arrays = F._arrays

    def out_of_box(decl, fill, window):
        return [
            np.full(decl.shape, decl.hi + 1.0, dtype=np.dtype(decl.dtype))
        ]

    monkeypatch.setattr(F, "_arrays", out_of_box)

    def h():
        x = any_array((), "float64", (0.0, 1.0))
        return assert_(x <= 1.0)

    report = F.probe(h, statuses=["discharged"])
    assert dict(report.skips).get("point-outside-declaration"), (
        f"every point the sampler built left the declared box and none was "
        f"counted as such: {report.skips}"
    )
    assert report.points_executed == 0, (
        "a point outside the declaration must be rejected BEFORE it is "
        "executed; executing it is how a probe invents a refutation"
    )
    assert F._arrays is out_of_box and real_arrays is not out_of_box


def test_a_program_that_raises_is_counted_under_its_own_reason():
    """A declared box may contain inputs this executor cannot run.

    ``lax.scan`` carries its body as a sub-jaxpr, and a declaration
    primitive inside that body has no concrete implementation to bind, so
    the minimal interpreter raises. That is a fact about the point and the
    interpreter, never about the verdict, and it is the single largest
    skip reason on this repository's own corpus (139 of 181 points, at
    ``test_undescended_assume.py``'s ``while_loop`` fixture alone).
    """
    from jax import lax

    def h():
        x = any_array((), "float64", (0.0, 4.0))

        def cond(state):
            i, _ = state
            return i < 2

        def body(state):
            i, c = state
            assume(x >= 0.0)
            return (i + 1, c)

        lax.while_loop(cond, body, (jnp.int32(0), x))
        return assert_(x * x >= 0.0)

    _, report = attack(h)
    assert dict(report.skips).get("program-raised"), (
        f"skips {report.skips}, declined {report.declined}"
    )


def test_an_assert_with_no_comparison_declines_the_boundary_phases():
    def h():
        x = any_array((), "float64", (0.0, 1.0))
        return assert_(jnp.isfinite(x))

    _, report = attack(h)
    assert dict(report.skips).get("no-margin-no-boundary-search"), report.skips


# -------------------------------------------------------- read-defect repairs


def test_a_literal_operand_in_an_assert_is_not_reported_as_a_TRACE_failure():
    """``_census`` crashed on a folded assert, and blamed the tracer.

    ``src = producer.get(eqn.invars[0])`` raises ``TypeError: unhashable
    type: 'Literal'`` whenever the tracer folded an assert's operand to a
    constant — hit twice by this tree's own corpus. It was fail-safe, but
    ``probe``'s single ``try`` wrapped the read as well as the trace, so
    the user-visible note said *"the harness could not be traced"* about a
    trace that had succeeded.
    """
    def folded():
        x = any_array((), "float64", (0.0, 1.0))
        return assert_(jnp.asarray(True))

    report = probe(folded, statuses=["discharged"])
    assert report.declined is None or "could not be traced" not in (
        report.declined
    ), report.declined
    assert report.points_executed > 0 or report.declined is None, (
        f"the literal-operand assert is readable now: {report.declined}"
    )


def test_a_float16_declaration_survives_W_error_RuntimeWarning():
    """A common CI setting turned a green VERIFIED into a crash.

    ``np.full(shape, v, dtype="float16")`` for a ``v`` the format cannot
    hold, and ``np.nextafter`` past the format's last finite value, both
    emit ``RuntimeWarning``; under ``-W error::RuntimeWarning`` they became
    exceptions that escaped the probe. The declared set is now intersected
    with the format, and the step past the last float is not taken.
    """
    script = textwrap.dedent(
        """
        import jax
        jax.config.update("jax_enable_x64", True)
        from stelling.harness import any_array, assert_
        from stelling.preconditions import check

        for lo, hi in [(-65504.0, 65504.0), (0.0, 1e5), (0.0, 1.0)]:
            def h(lo=lo, hi=hi):
                x = any_array((), "float16", (lo, hi))
                return assert_(x >= lo)
            v = check(h, vacuity_mode="inputs-only", falsify="sample")
            print("OK", lo, hi, v.status)
        """
    )
    src = pathlib.Path(stelling.__file__).resolve().parent
    proc = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(src.parent),
            "PATH": "/usr/bin:/bin",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "HOME": "/tmp",
        },
    )
    assert proc.returncode == 0, (
        f"a plain float16 declaration crashed under "
        f"-W error::RuntimeWarning.\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr[-2500:]}"
    )
    assert proc.stdout.count("OK") == 3, proc.stdout
