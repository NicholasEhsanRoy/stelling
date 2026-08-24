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
import time
import textwrap
from fractions import Fraction

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
    _int_ok,
    _rat_convert,
    _rat_pow,
    _rat_sqrt,
    _Unreplayable,
    _window,
    probe,
)
from stelling.harness import any_array, assert_, assume  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from _solver_gate import need_solver  # noqa: E402

PROBE_SRC = pathlib.Path(stelling.__file__).resolve().parent / "falsify.py"


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def traced(harness):
    """jax's own ``ClosedJaxpr`` for ``harness`` — what :func:`probe` takes.

    The probe no longer traces anything: it is handed the program the
    ANALYSIS judged, because a probe that traces for itself can probe a
    different program than the verdict is about (measured, with the
    overflow tripwire armed). In the pipeline that object comes from
    ``_jax_compat.trace_with_jaxpr``; here, where there is no analysis to
    agree with, one trace is all there is.
    """
    return jax.make_jaxpr(harness)()


def attack(harness, *, semantics="real", n=1, **kw):
    """Probe ``harness`` as if the analysis had discharged everything."""
    try:
        return None, probe(
            traced(harness), statuses=["discharged"] * n, semantics=semantics,
            **kw
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


@need_solver
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


def int_declared_but_rounding():
    """``int16`` in, ``float32`` arithmetic, and it ROUNDS.

    ``(2**24 + y) - 2**24`` is ``y`` over ℝ and over ℚ, so ``y <= 3`` is
    TRUE on the declared box ``{0, 1, 2, 3}`` and a ``VERIFIED`` is right.
    float32's spacing at ``2**24`` is 2, so ``2**24 + 3`` ties up to
    ``2**24 + 4`` and the program computes ``4.0`` at ``x = 3``.
    """

    def harness():
        x = any_array((), "int16", (0, 3))
        y = x.astype("float32")
        b = jnp.float32(2 ** 24)
        return assert_((b + y) - b <= jnp.float32(3.0))

    return harness


def _rounds_in_float32(a):
    """``int -> float32 -> int``, rounding in the middle, ints at both ends."""
    y = a.astype("float32")
    b = jnp.float32(2 ** 24)
    return ((b + y) - b).astype("int32")


def int_declared_but_rounding_out_of_sight():
    """The same rounding, hidden inside a ``jax.checkpoint`` body.

    EVERY dtype the OUTER jaxpr mentions is integral — ``int32`` in,
    ``int32`` out of the ``remat2`` equation, ``bool`` into the assert —
    and every declaration is ``int32``. Only the call body is float, so a
    predicate that reads the declarations calls this program integral, so
    does one that reads the top-level equations, and only one that
    RECURSES gets it right. Driven: with the recursion removed from
    ``_integral_program`` and nothing else changed, this FIRES on both
    supported jax series, adjudicated ``exact-integer-arithmetic``.
    """

    def harness():
        x = any_array((), "int32", (0, 3))
        return assert_(jax.checkpoint(_rounds_in_float32)(x) <= 3)

    return harness


@pytest.mark.parametrize(
    "name,harness",
    [
        ("in plain sight", int_declared_but_rounding()),
        ("inside a checkpoint body", int_declared_but_rounding_out_of_sight()),
    ],
)
def test_an_int_declared_program_that_ROUNDS_is_not_integer_arithmetic(
    name, harness
):
    """THE THIRD APPEARANCE OF ONE MISTAKE, and the shape of the repair.

    The integer branch admits a firing under the sentence *"exact integer
    arithmetic: no rounding involved"*, and its predicate was::

        all(np.dtype(d.dtype).kind in "iub" for d in census.declarations)

    which is a test on the DECLARATIONS licensing a claim about the
    PROGRAM. An int-declared program that converts to float and rounds
    passes it, and the fixtures above are four lines of it: a correct
    ``VERIFIED`` called *"stelling is UNSOUND at this query"* through the
    public API, no mutation, no solver, on jax 0.10.2 and 0.11.0 alike.

    **That is the ulp proxy's mistake for the third time in this module.**
    First ulp-stability of the input stood in for *"this violation is not
    a rounding artefact"*; then a fall-back to that same proxy stood in
    for an exact adjudication; then this. The class, not the instance, is
    what has to be fixed each time: **a predicate that licenses an
    exactness claim must be computed from the object the claim is about.**
    So the predicate is read off every operand and result dtype in the
    jaxpr, at every depth — which is what the second fixture pins, because
    nothing above the ``remat2`` equation there is anything but an
    integer.

    Declining is not the whole of the requirement either. The violation
    must be declined BY BEING READ: the exact replay evaluates this
    program at the same point over ℚ, finds the obligation TRUE, and
    counts it a ``float-rounding-artefact``. A decline for want of a
    reading would hide the same false alarm behind a reach gap.
    """
    import stelling.falsify as F

    census = F._census(harness)
    assert all(
        np.dtype(d.dtype).kind in "iub" for d in census.declarations
    ), (
        f"{name}: every declaration here is integral, and the whole point "
        f"of the fixture is that this is not evidence about the program's "
        f"arithmetic. If it stops being true the test measures nothing."
    )
    assert not census.integral, (
        f"{name}: `_integral_program` called this program integral. It "
        f"converts to float32 and rounds; the sentence the integer branch "
        f"emits — 'exact integer arithmetic: no rounding involved' — is "
        f"contradicted by the program beside it."
    )

    found, report = attack(harness)
    assert found is None, (
        f"{name}: a correct VERIFIED was called UNSOUND. "
        f"{found.render() if found else ''}"
    )
    assert report.violations_seen > 0, (
        f"{name}: no violation was executed, so nothing was adjudicated "
        f"and this test proves nothing; skips {report.skips}"
    )
    assert dict(report.adjudications) == {
        "exact-replay-holds-over-the-rationals": report.violations_seen
    }, (
        f"{name}: the violation must be declined by being READ over ℚ, "
        f"not by an abstention: {report.adjudications}, "
        f"abstentions {dict(report.abstentions)}"
    )


def test_the_int_declared_rounding_program_survives_the_public_door():
    """And it must not raise through :func:`check` either.

    The unit-level test above supplies the status; this drives the real
    pipeline with the solvers, because that is the path the false alarm
    was reachable on — ``check(h, vacuity_mode="inputs-only",
    falsify="sample")``, four lines of harness, raising
    ``VerifiedFalsified``.
    """
    verdict = check(
        int_declared_but_rounding(),
        vacuity_mode="inputs-only",
        semantics="real",
        solver_timeout_ms=8000,
        falsify="sample",
    )
    assert verdict.status == "VERIFIED", verdict.notes


def test_the_integer_admission_IS_A_READING_OF_THE_PROGRAM_at_the_source():
    """The rule, pinned where a refactor meets it — not just an instance.

    The fixtures above are two programs. What went wrong three times in
    this module is not a program, it is a habit: something cheap standing
    in for an exactness claim — ulp-stability of the INPUT for *"not a
    rounding artefact"*, then a fall-back to that proxy for an exact
    adjudication, then the DECLARATIONS for *"this program does integer
    arithmetic"*. Each was defensible-looking, each was correlated with
    the thing it stood for, and each turned a correct ``VERIFIED`` into
    *"stelling is UNSOUND"* in a handful of lines.

    So the rule is asserted at the source, in the same posture as
    :func:`test_the_ulp_proxy_is_gone_from_the_fire_condition_entirely`
    and for the same reason: a behavioural test names the programs someone
    thought of, and the next proxy will be a different shape. **The guard
    on the integer admission must READ THE PROGRAM** — ``census.integral``
    is computed by ``_integral_program`` over every operand and result
    dtype in the jaxpr at every depth — and a predicate that consults only
    the declarations is exactly the defect being repaired.
    """
    import ast

    tree = ast.parse(PROBE_SRC.read_text(encoding="utf-8"))
    confirm = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_confirm"
    )
    assign = next(
        n for n in ast.walk(confirm)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "integral" for t in n.targets
        )
    )
    reads = {ast.unparse(n) for n in ast.walk(assign.value)}
    assert "census.integral" in reads, (
        f"the integer admission is guarded by {ast.unparse(assign.value)!r}, "
        f"which does not read the census's reading of the PROGRAM. That "
        f"branch emits 'exact integer arithmetic: no rounding involved' "
        f"about the program, so its predicate is computed from the "
        f"program; anything else is a proxy, and this module has shipped "
        f"three of those."
    )

    # and the reading itself must recurse, or a float step one `jit` deep
    # is invisible to it -- which is the second fixture above
    prog = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_integral_program"
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_integral_program"
        for n in ast.walk(prog)
    ), "`_integral_program` no longer descends into call bodies"


def test_the_rematerialisation_primitive_the_live_jax_emits_is_read():
    """``remat2``, traced live — the branch the OTHER test does not reach.

    ``_CALL_PRIMITIVES``'s comment claimed for a batch that a live check
    traced ``jnp.where`` *and* ``jax.checkpoint``. It traced ``jnp.where``
    only, and no falsify test named ``jax.checkpoint`` or ``remat2``
    outside a docstring — which is the same defect as a name list nothing
    checks, one level up, on the one name that is load-bearing.

    LOAD-BEARING FOR WHAT, MEASURED. ``_call_jaxpr_of`` accepts a BARE
    ``Jaxpr`` as well as a ``ClosedJaxpr``, and that widening changes
    exactly one thing on the two supported series: on jax 0.10.2 ``jit``
    carries a real ``ClosedJaxpr`` while ``remat2`` carries a bare
    ``Jaxpr`` with no ``.jaxpr``/``.consts`` at all, so the old
    ``hasattr`` test refused ``remat2`` there and nothing else. (On 0.11.0
    ``Jaxpr`` itself answers ``.jaxpr`` with itself and ``.consts`` with
    ``[]``, so the old test matched both.) At ``99abdb0``, with
    ``_CALL_PRIMITIVES`` corrected and ``_call_jaxpr_of`` left as it was,
    the harness below FIRES on jax 0.10.2 under ``ulp-proxy-refutes`` —
    the Kahan false alarm, one more route.
    """
    import stelling.falsify as F

    def kahan_through_checkpoint():
        y = any_array((), "float64", (0.0, 2.0))
        z = jax.checkpoint(lambda a: (1e16 + a) - 1e16)(y)
        return assert_(z <= y)

    traced = jax.make_jaxpr(kahan_through_checkpoint)()
    names = {e.primitive.name for e in traced.jaxpr.eqns}
    call_names = names - {"stelling_any", "stelling_assert", "le"}
    assert call_names, f"no call primitive in {names}"
    assert call_names <= set(F._CALL_PRIMITIVES), (
        f"the live jax lowers `jax.checkpoint` through {sorted(call_names)}, "
        f"which `_CALL_PRIMITIVES` does not name: {F._CALL_PRIMITIVES}"
    )

    # and the body really is handed over, in whichever shape this series
    # wraps it in -- the bare-`Jaxpr` branch of `_call_jaxpr_of` exists
    # for exactly this equation on jax 0.10.2
    eqn = next(e for e in traced.jaxpr.eqns if e.primitive.name in call_names)
    body, consts = F._call_jaxpr_of(eqn)
    assert body.eqns, f"an empty body for {eqn.primitive.name!r}"
    assert len(consts) == len(body.constvars), (body, consts)

    found, report = attack(kahan_through_checkpoint)
    assert found is None, f"the Kahan false alarm is back: {found.render()}"
    assert report.violations_seen > 0, report.skips
    assert dict(report.adjudications) == {
        "exact-replay-holds-over-the-rationals": report.violations_seen
    }, (
        f"the `jax.checkpoint` route declined, but not by being READ: "
        f"{report.adjudications}. Declining is safe; being read is what "
        f"the bare-`Jaxpr` branch buys, and this pins it."
    )


def test_an_unreplayable_primitive_DECLINES_and_names_what_it_could_not_read():
    """``exp`` is irrational at every rational but 0, so the replay abstains.

    THIS TEST RAN THE OTHER WAY ROUND ONE BATCH AGO, and the change is the
    whole point of this one. It used to assert that the fire condition
    "degrades to the weaker test rather than declining everything it
    cannot prove", and named the ulp proxy as the adjudicator. That
    fall-back is what kept the Kahan false alarm alive: the proxy is blind
    to coarse quantisation, so ANY program with a step the replay could
    not read was a route straight back to it — and `jnp.where`, a
    fractional `pow`, `exp`, `sort`, `cumsum`, `rem`, a non-square `sqrt`
    and every matmul are all such steps.

    An alarm whose message is "stelling is UNSOUND" must not be admitted
    by a test already measured to invent refutations. So an abstention
    DECLINES. The obligation here really is false — ``exp(2)`` is 7.389 —
    and the probe still will not report it, because it cannot prove it
    over ℚ. That is a REACH COST and it is paid deliberately; what must
    never happen is that it is paid silently, so the decline is counted
    under its own reason and the reason the exact reading was unavailable
    is in the report and in the stamp line.
    """
    def transcendental():
        x = any_array((), "float64", (0.0, 2.0))
        return assert_(jnp.exp(x) <= 2.0)  # exp(2) = 7.389...: FALSE

    found, report = attack(transcendental)
    assert found is None, (
        f"an executed violation was REPORTED on a program with a step the "
        f"exact replay cannot read, so something other than an exact test "
        f"admitted it: {found.adjudication!r}"
    )
    assert report.violations_seen > 0, (
        "the float program really is false here, so a run that saw no "
        "violation is not evidence about the fire condition"
    )
    assert dict(report.adjudications) == {
        "declined-no-exact-reading": report.violations_seen
    }, report.adjudications
    assert dict(report.skips) == {
        "no-exact-reading-of-this-program": report.points_declined
    }, report.skips
    assert any(
        "'exp' has no exact rational reading" in text
        for text, _ in report.abstentions
    ), report.abstentions
    assert "THE EXACT READING WAS UNAVAILABLE" in report.stamp_line()
    assert "'exp' has no exact rational reading" in report.stamp_line()


def test_the_ulp_proxy_is_gone_from_the_fire_condition_entirely():
    """Not demoted, not kept as a filter — gone from the firing path.

    An instrument is not made safe by putting a correct adjudicator in
    FRONT of an unsafe one, which is what the previous batch did: the
    exact replay went first and the ulp proxy stayed as the fall-back, and
    the false alarm the replay was added to kill was still four lines away
    through ``jnp.where``. This asserts the removal at the source, because
    the alternative — asserting it only through behaviour — leaves the
    door open for it to come back as a "filter" that admits.
    """
    import ast

    tree = ast.parse(PROBE_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                   ast.ClassDef)
        ):
            # the prose is allowed to say why the proxy is gone; this is
            # about what the module DOES, so every docstring is dropped
            # before the scan and only executable code is read
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    code = ast.unparse(tree)
    for banned in ("ulp-proxy", "precision-ambiguous"):
        assert banned not in code, (
            f"stelling/falsify.py can still emit {banned!r}. Only an exact "
            f"test may admit a firing; every other outcome declines, and a "
            f"proxy kept as a 'filter' is one refactor from admitting."
        )

    confirm = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_confirm"
    )
    called = {
        n.func.id for n in ast.walk(confirm)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_step" not in called and "_execute" not in called, (
        f"the fire condition executes the program again ({sorted(called)}). "
        f"Re-execution at a neighbouring point is the ulp proxy however it "
        f"is spelled: the exact replay decides, or the violation declines."
    )


def test_the_call_primitive_the_live_jax_emits_is_replayed():
    """THE OTHER HALF OF THE SAME DEFECT: a name list nothing checked.

    ``_CALL_PRIMITIVES`` shipped as ``("pjit", "closed_call", "remat",
    "checkpoint")`` with a comment saying ``jax.numpy`` routes almost
    everything through ``pjit``. On both supported jax series the
    primitive is called ``jit``, so not one of the four ever matched and
    the replay abstained on essentially every ``jnp`` program — which,
    with the ulp proxy behind it, meant the weaker test adjudicated them.

    A name list is only as good as the thing that checks it against the
    live library, so this traces the LIVE jax and asserts twice: that the
    primitive is one this module claims to replay, and that a whole
    ``jnp.where`` program actually round-trips through the replay rather
    than abstaining on it.
    """
    import stelling.falsify as F

    def where_program(x):
        return jnp.where(x >= 0.0, x + 1e16 - 1e16, x)

    traced = jax.make_jaxpr(where_program)(jnp.float64(1.0))
    names = {e.primitive.name for e in traced.jaxpr.eqns}
    call_names = names - {"ge", "add", "sub", "neg", "select_n", "lt"}
    assert call_names, f"no call primitive in {names}"
    assert call_names <= set(F._CALL_PRIMITIVES), (
        f"the live jax lowers `jnp.where` through {sorted(call_names)}, "
        f"which `_CALL_PRIMITIVES` does not name: {F._CALL_PRIMITIVES}. "
        f"The replay abstains on every such program, and an abstention "
        f"declines — so this makes the probe inert rather than unsafe, "
        f"which is the failure that reads as coverage."
    )

    def kahan_through_where():
        y = any_array((), "float64", (0.0, 2.0))
        z = jnp.where(y >= 0.0, (1e16 + y) - 1e16, y)
        return assert_(z <= y)

    found, report = attack(kahan_through_where)
    assert found is None, f"the Kahan false alarm is back: {found.render()}"
    assert report.violations_seen > 0, report.skips
    assert dict(report.adjudications) == {
        "exact-replay-holds-over-the-rationals": report.violations_seen
    }, (
        f"the `jnp.where` route declined, but not by being READ: "
        f"{report.adjudications}. Declining is safe; being read is what "
        f"naming the call primitive buys, and this pins it."
    )


def test_a_bitwise_integer_and_is_not_replayed_as_a_boolean_one():
    """THE ONE PLACE THE REPLAY COULD INVENT A REFUTATION, closed.

    ``_BOOLEAN`` and the ``reduce_and``/``reduce_or`` folds read jax's
    ``and``/``or``/``xor``/``not`` as boolean connectives. Over an INTEGER
    operand those primitives are bitwise, and the two disagree at the
    first argument anyone would try: ``5 & 2`` is 0, while ``bool(5) and
    bool(2)`` is True. Driven before the guard existed, ``_confirm``
    returned ``exact-replay-refutes-over-the-rationals`` on an obligation
    that is TRUE over ℚ — an INVENTED refutation, in the one direction
    this module has no other defence against.

    It was unreachable through the public API only because
    ``propagate._t_bool_logic`` refuses bitwise-int. That is an invariant
    in the one module ``falsify.py`` is forbidden to import and does not
    re-derive, so it is not this module's to rely on: every other integer
    path is guarded by ``_int_ok`` and these returned early past it.

    Driven at the evaluator, since the analysis will not route one here.
    """
    import stelling.falsify as F

    def bitwise(a, b):
        return jax.lax.bitwise_and(a, b)

    traced = jax.make_jaxpr(bitwise)(np.int32(5), np.int32(2))
    eqn = traced.jaxpr.eqns[0]
    assert eqn.primitive.name == "and", eqn.primitive.name
    with pytest.raises(F._Unreplayable, match="BITWISE"):
        F._boolean_only("and", eqn, np.dtype("int32"))

    # and the fold, which reaches the same guard from `_apply`
    def fold(v):
        return jax.lax.reduce_and(v, axes=(0,))

    traced = jax.make_jaxpr(fold)(np.zeros((3,), "int32"))
    eqn = traced.jaxpr.eqns[0]
    assert eqn.primitive.name == "reduce_and", eqn.primitive.name
    with pytest.raises(F._Unreplayable, match="BITWISE"):
        F._boolean_only("reduce_and", eqn, np.dtype("int32"))

    # the honest other half: a real boolean `and` still replays
    def boolean(a, b):
        return jax.lax.bitwise_and(a, b)

    traced = jax.make_jaxpr(boolean)(np.bool_(True), np.bool_(False))
    eqn = traced.jaxpr.eqns[0]
    F._boolean_only("and", eqn, np.dtype("bool"))


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


def deep_squarings(n):
    """``y`` squared ``n`` times, which is ``n + 2`` element-visits and an
    exact value ``53 * 2**n`` bits wide.  The two numbers diverge fast, and
    that divergence is the whole finding below."""

    def harness():
        y = any_array((), "float64", (0.5, 0.9))
        for _ in range(n):
            y = y * y
        return assert_(y >= 0.25)  # 0.7**65536 is 0: FALSE

    return harness


def test_the_replay_budget_bounds_RATIONAL_GROWTH_and_not_only_element_count():
    """THE ELEMENT BUDGET COUNTED THE WRONG THING, AND SAID SO IN SECONDS.

    ``REPLAY_ELEMENT_BUDGET`` is read off the AVALS, so it sees SHAPES and
    is blind to the size of the rationals flowing through them. A
    ``Fraction`` doubles in width every squaring and Python's ``gcd`` is
    quadratic in that width, so the cost goes up 4x per step at a shape
    the element budget reads as free. Measured on this tree at 99abdb0,
    one declared float64 point, jax 0.11.0:

    =========  ==============  =================
    squarings  element-visits  replay wall time
    =========  ==============  =================
           12              15            0.031 s
           14              17            0.454 s
           16              19            7.728 s
           17              20           28.117 s
    =========  ==============  =================

    Nineteen element-visits is 0.008% of a budget documented at "2.8 -
    10.5 microseconds per element visited"; it charged 4.7 SECONDS each.

    So the missing term is bounded directly, by the width of the values
    themselves. Both halves are driven: the cascade abstains in
    milliseconds, and the WIDE-and-shallow shape -- the half of the
    original claim that did hold -- still replays.
    """
    import stelling.falsify as F

    census = F._census(deep_squarings(16))
    assert census.replay_cost < F.REPLAY_ELEMENT_BUDGET, (
        f"the element budget already refuses this program "
        f"({census.replay_cost}), so it is no longer the case that the "
        f"element count is blind to rational growth and this test is "
        f"measuring nothing"
    )
    started = time.perf_counter()
    with pytest.raises(F._Unreplayable, match="bits wide"):
        F._replay(census, (np.float64(0.7),))
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, (
        f"the width budget took {elapsed:.2f}s to refuse a program the "
        f"unbounded version needed 7.7s for; it is supposed to refuse "
        f"BEFORE paying, not after"
    )

    # and the wide-shallow half, which is the shape the element budget was
    # calibrated on and which must keep replaying
    def wide():
        v = any_array((4096,), "float64", (0.0, 1.0))
        return assert_(jnp.sum(v * v) <= 0.5)

    census = F._census(wide)
    assumes, asserts = F._replay(census, (np.full((4096,), 0.9),))
    assert asserts == [False], asserts


def test_the_wall_clock_backstop_is_a_backstop_and_not_the_operative_bound():
    """A wall clock makes a firing depend on the machine, so it must not bind.

    The direction is safe -- a slow machine DECLINES where a fast one
    fires, never the reverse, because only an exact refutation may admit
    -- but a bound that routinely decided things would make this
    instrument's results irreproducible. It is calibrated against the
    deterministic pair: at most ``REPLAY_ELEMENT_BUDGET`` values, each at
    most ``REPLAY_BIT_BUDGET`` bits wide, and one ``Fraction`` multiply at
    4,096 bits is about 19 microseconds here.

    **AND "CALIBRATED ABOVE" IS AS MUCH AS THIS TEST ESTABLISHES, WHICH IS
    LESS THAN IT SOUNDS.** The pair permits about 4.75 seconds against a
    5.0-second clock, so the assertion below passes on a margin of 5%, and
    ordinary programs land inside the clock rather than far below it: a
    ``(35000,)`` float64 declaration squared six times replays in 1.55 s
    on the machine this was written on, so a machine three times slower
    declines it. The honest statement, which is in this module's docstring
    and in the constant's comment rather than only here, is that **whether
    this probe FIRES on a given program can depend on the machine**. What
    the safe direction buys is that only its REACH varies: nothing the
    clock does can admit a firing, so slow hardware costs refutations and
    never invents them.
    """
    import stelling.falsify as F

    worst = F.REPLAY_ELEMENT_BUDGET * 19e-6
    assert F.REPLAY_SECONDS_BUDGET > worst * 0.9, (
        f"the wall clock ({F.REPLAY_SECONDS_BUDGET}s) is below what the "
        f"deterministic budgets already allow ({worst:.1f}s), so it, and "
        f"not they, is what decides -- and which replays get decided now "
        f"depends on the machine"
    )

    # and it really does stop a runaway: a guard past its deadline refuses
    guard = F._Guard()
    guard._deadline = time.monotonic() - 1.0
    with pytest.raises(F._Unreplayable, match="wall-clock"):
        guard.tick()


# ------------------------------------------------------- the firing MESSAGE


def test_the_firing_message_does_not_end_by_saying_nothing_was_found():
    """The module's most important message used to contradict itself.

    :func:`stelling.falsify._fire` appends
    :meth:`ProbeReport.stamp_line`, which had no firing branch — so every
    firing ended with *"NO VIOLATION WAS FOUND, WHICH IS NOT EVIDENCE THAT
    THERE IS NONE"* three lines under *"FALSIFICATION PROBE FIRED"*.
    """
    with pytest.raises(VerifiedFalsified) as caught:
        probe(traced(lying_pow), statuses=["discharged"])
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
        probe(traced(lying_pow), statuses=["discharged"])
    lowered = caught.value.report.stamp_line().lower()
    for word in ("confidence", "validated", "corroborat", "clean", "passed"):
        assert word not in lowered, lowered


def test_the_firing_names_which_test_admitted_the_violation():
    """A reader has to be able to tell a ℚ-proof from a proxy."""
    with pytest.raises(VerifiedFalsified) as caught:
        probe(traced(lying_pow), statuses=["discharged"])
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


def test_a_movement_primitive_that_reads_an_operand_by_VALUE_is_refused():
    """``_apply`` used to substitute ``np.zeros`` for what it could not see.

    ``_movement`` replays a data-movement primitive by binding it to INDEX
    arrays in place of the data, which is exact -- jax's own shape rules
    answer "where did this element come from". Every operand that is NOT
    data is passed through verbatim, because its VALUE is what the
    primitive reads: a gather's start indices, a dynamic_slice's offsets.

    For a non-literal operand there was no value to pass, and the code
    substituted a float64 array of ZEROS. That is dead for today's
    ``_MOVEMENT`` table -- every entry in it is all-data -- and silently
    wrong the moment ``gather`` or ``dynamic_slice`` is added: the replay
    would then read a DIFFERENT program from the one that executed, at a
    different offset, and could report a violation that is not there.

    A refusal is loud and a zero is silent, so it refuses. Driven by
    adding a table entry, because the table has no such entry yet and an
    unexercised guard on a soundness path is a guard nobody has run.
    """
    import stelling.falsify as F

    def gathering():
        x = any_array((4,), "float64", (0.0, 1.0))
        i = any_array((), "int32", (0, 3))
        return assert_(x[i] <= 2.0)

    census = F._census(gathering)
    names = [e.primitive.name for e in census.closed.jaxpr.eqns]
    assert "dynamic_slice" in names, names
    assert "dynamic_slice" not in F._MOVEMENT, (
        "`dynamic_slice` is in the movement table now, so this test must "
        "be rewritten to drive whatever operand it reads by value"
    )
    original = dict(F._MOVEMENT)
    F._MOVEMENT["dynamic_slice"] = (0,)  # operand 1 is the START, by VALUE
    try:
        with pytest.raises(F._Unreplayable, match="by VALUE"):
            F._replay(census, (np.full((4,), 0.5), np.int32(2)))
    finally:
        F._MOVEMENT.clear()
        F._MOVEMENT.update(original)


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
    "assume-not-fully-executed": (
        "driven: `assume()` inside a `jit` or a `remat2` body, which "
        "`_execute` binds whole — four shapes, and at all three public "
        "doors"
    ),
    "float-rounding-artefact": "driven: the Kahan shape above",
    "no-exact-reading-of-this-program": (
        "driven: `exp`, and the three `scatter` fixtures in "
        "tests/test_falsify_probe.py"
    ),
    "assume-unsatisfied-over-the-rationals": (
        "driven: an assume whose own evaluation rounds across the "
        "boundary — `assume(y * 0.1 * 10.0 <= y)` under the Kahan assert"
    ),
    "no-margin-no-boundary-search": "driven: an assert with no comparison",
    "executed-float-depends-on-granularity": (
        "driven: `jnp.mean` under `ieee` — the trace inlines the `jit` "
        "`jnp.mean` is built out of, so `_execute` walks op by op what the "
        "caller's own call compiles whole"
    ),
    "whole-program-route-unavailable": (
        "driven with a route that raises: no program is known that traces "
        "and executes op by op and then cannot be staged under one `jit`, "
        "and 'not shown to move' is not 'shown not to move'"
    ),
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
    # the decline reasons that travel as the SECOND member of `_confirm`'s
    # return tuple, recognised by the adjudication name that follows them
    emitted |= set(
        re.findall(r'^\s+"([a-z][a-z0-9-]*)",\n\s+"(?:exact|ieee|declined)',
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

    report = F.probe(traced(h), statuses=["discharged"])
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

    report = probe(traced(folded), statuses=["discharged"])
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


def test_the_window_is_snapped_onto_the_dtypes_own_grid():
    """``_admissible`` was right and the SAMPLER was refusing its own corners.

    ``_admissible`` reads the declaration's endpoints AS DECLARED, widens
    the sampled array to float64 and compares over ℝ — which is correct
    and must stay strict, because it is the last thing between a sampler
    defect and a firing. ``_window`` did not match it: it returned the
    declared endpoints unchanged, and ``np.full((), 0.3, "float32")`` is
    0.30000001192092896, which is OUTSIDE ``(-0.3, 0.3)`` over ℝ.

    Measured at 99abdb0 on ``float32`` and ``float16`` declarations of
    ``(-0.3, 0.3)``: 69 points built, 59 executed, ``skips
    {'point-outside-declaration': 10}`` — ten points thrown away at
    exactly the corners the ``endpoints`` strategy exists to reach. The
    window now snaps INWARD onto the dtype's own grid, which is exact and
    is the identity at float64.
    """
    import stelling.falsify as F

    for dtype in ("float32", "float16", "float64"):
        decl = F.Declaration(
            position=0, shape=(), dtype=dtype, lo=-0.3, hi=0.3
        )
        window, reason = F._window(decl)
        assert reason is None, (dtype, reason)
        lo, hi = window
        for end in (lo, hi):
            built = np.full((), end, dtype=dtype)
            assert F._admissible(decl, built), (
                f"{dtype}: the window's own endpoint {end!r} builds a point "
                f"the declaration does not admit ({built!r}), which is the "
                f"mismatch this test exists for"
            )
        assert lo >= -0.3 and hi <= 0.3, (dtype, window)

    # float64 is untouched: the cast is the identity there
    decl = F.Declaration(
        position=0, shape=(), dtype="float64", lo=-0.3, hi=0.3
    )
    assert F._window(decl)[0] == (-0.3, 0.3)

    # and end to end, the ten rejections are gone
    def narrow():
        x = any_array((), "float32", (-0.3, 0.3))
        return assert_(x * x <= 1.0)

    _, report = attack(narrow)
    assert "point-outside-declaration" not in dict(report.skips), report.skips
    assert report.points_executed == report.points_built, (
        report.points_built, report.points_executed
    )


def test_an_assume_that_rounds_across_its_boundary_declines_over_Q():
    """``assume-unsatisfied-over-the-rationals`` had no reaching input.

    The coverage table called it "DEFENCE IN DEPTH, no known reaching
    input: the assume would have to hold in floats and fail over ℚ at the
    same point". It is four lines away — ``y * 0.1 * 10.0 <= y`` is FALSE
    over ℚ for most y (0.1 is not a tenth) and TRUE in float64 wherever
    the two roundings cancel — and it is the exact failure the entry was
    written to prevent: a decline reason nobody has driven is a sentence
    nobody has checked.
    """
    def rounding_assume():
        y = any_array((), "float64", (0.0, 2.0))
        assume(y * 0.1 * 10.0 <= y)
        s = 1e16
        return assert_((s + y) - s <= y)

    found, report = attack(rounding_assume)
    assert found is None, found and found.render()
    counts = dict(report.skips)
    assert counts.get("assume-unsatisfied-over-the-rationals", 0) > 0, counts
    # AND IT IS DECLINED AT THE GATE, NOT ADJUDICATED AS A VIOLATION.
    # The exact re-reading of the assumes now runs on EVERY admissible
    # point rather than only where the obligation had already evaluated
    # FALSE, so a point outside the assumed region over ℚ never becomes a
    # violation to adjudicate -- which is what the count is FOR. It used
    # to arrive through `_confirm` and be counted under
    # `exact-replay-outside-the-assumed-region`, i.e. as an executed
    # violation the probe had declined, which those points were not.
    assert report.points_declined == 0, (
        f"a point outside the assumed region over ℚ is not a declined "
        f"violation: {report.points_declined}, {report.adjudications}"
    )
    assert dict(report.adjudications).get(
        "exact-replay-outside-the-assumed-region", 0
    ) == 0, report.adjudications
    # and the number that used to read as coverage now carries its evidence
    assert report.points_admissible_unconfirmed == 0, report
    assert "re-read over ℚ and confirmed" in report.stamp_line()


# ------------------------------------- a reading of the program can be PARTIAL
#
# THE FOURTH DEFECT, AND THE RULE THAT CLOSES THE CLASS. Four audits found
# four defects and they are one defect: a predicate licensing a claim about
# the PROGRAM while being computed from something else. The first three are
# pinned above (`test_the_ulp_proxy_is_gone_from_the_fire_condition_entirely`,
# `test_an_int_declared_program_that_ROUNDS_is_not_integer_arithmetic`,
# `test_the_integer_admission_IS_A_READING_OF_THE_PROGRAM_at_the_source`).
#
# The fourth adds a second half to the rule. Its predicate WAS computed
# from the program — it was computed from PART of it. `_execute` iterates
# `jaxpr.eqns` at the top level and hands a call equation whole to
# `Primitive.bind`, so a `stelling_assume` inside a `jit` or `remat2` body
# executes and never reaches `run.assumes`, and the gate `if run.assumes
# and not all(...)` on an empty list admitted every point. `propagate` DOES
# narrow on that assume — it is the whole reason the VERIFIED exists — so
# the probe attacked points the analysis had claimed nothing about.
#
# So: a reading that may be partial is checked against what is in the
# program before it licenses anything. Every quantity, not just this one.


def nested_assume_harness(wrapper, dtype, lo, hi, bound):
    """``assume(a >= bound)`` one call deep, and an assert that repeats it.

    Over the declared box the obligation is TRUE **on the assumed region**
    and false below it, so a ``VERIFIED`` is CORRECT and any firing is a
    false alarm. Worked out by hand from the box and the arithmetic, not
    by asking stelling: the assume admits exactly ``[bound, hi]`` and the
    assert asserts exactly ``>= bound``, so the two coincide.
    """

    def harness():
        x = any_array((), dtype, (lo, hi))
        y = wrapper(lambda a: (assume(a >= bound), a)[1])(x)
        return assert_(y >= bound)

    return harness


NESTED_ASSUME_SHAPES = [
    # (label, wrapper, dtype, lo, hi, bound, semantics)
    # the two branches of `_confirm` that return BEFORE the replay is
    # consulted are the exposure, and they are reached by different
    # programs: `exact-integer-arithmetic` by an integral one under
    # `real`, `ieee-executed-float` by any program under `ieee`.
    ("jit, integral, real", jax.jit, "int32", 0, 10, 9, "real"),
    ("remat2, integral, real", jax.checkpoint, "int32", 0, 10, 9, "real"),
    ("jit, float, ieee", jax.jit, "float32", 0.0, 10.0, 9.0, "ieee"),
    ("remat2, float, ieee", jax.checkpoint, "float32", 0.0, 10.0, 9.0, "ieee"),
]


@pytest.mark.parametrize(
    "label,wrapper,dtype,lo,hi,bound,semantics", NESTED_ASSUME_SHAPES
)
def test_an_assume_the_executed_walk_CANNOT_SEE_does_not_admit_a_point(
    label, wrapper, dtype, lo, hi, bound, semantics
):
    """The fourth false alarm, in all four shapes that reach it.

    ``_execute`` cannot see this assume, so the probe does not know whether
    the point is inside the assumed region — and *"does not know"* is not
    *"admitted"*. It declines by name, and the count says how many points
    it declined, which is what stops the decline from reading as coverage.
    """
    harness = nested_assume_harness(wrapper, dtype, lo, hi, bound)
    found, report = attack(harness, semantics=semantics)
    assert found is None, (
        f"[{label}] the probe raised 'stelling is UNSOUND' on a CORRECT "
        f"VERIFIED: {found and found.render()}"
    )
    counts = dict(report.skips)
    assert counts.get("assume-not-fully-executed", 0) > 0, (
        f"[{label}] the point was not declined for the right reason: "
        f"{counts}"
    )
    assert report.points_executed > 0, "nothing was executed; wrong fixture"
    assert report.points_admissible == 0, (
        f"[{label}] {report.points_admissible} point(s) counted as "
        f"'admitted by every assume' while an assume went unread"
    )


@pytest.mark.parametrize(
    "label,wrapper,dtype,lo,hi,bound,semantics", NESTED_ASSUME_SHAPES
)
def test_the_nested_assume_false_alarm_is_gone_through_the_public_door(
    label, wrapper, dtype, lo, hi, bound, semantics
):
    """Five lines, no mutation, no solver, and it used to RAISE.

    The probe's own entry point is where the repair lives, but the cost of
    getting this wrong is paid at ``check()``, so that is where it is
    pinned too: same verdict with the probe on as with it off.
    """
    harness = nested_assume_harness(wrapper, dtype, lo, hi, bound)
    quiet = check(harness, vacuity_mode="inputs-only", semantics=semantics)
    assert quiet.status == "VERIFIED", (
        f"[{label}] the fixture does not reach the branch under test: "
        f"{quiet.status}"
    )
    probed = check(
        harness,
        vacuity_mode="inputs-only",
        semantics=semantics,
        falsify="sample",
    )
    assert probed.status == "VERIFIED", probed.status


def test_the_nested_assume_false_alarm_is_gone_at_the_other_two_doors():
    """``check_contract`` and ``check_inductive_step`` mint VERIFIEDs too.

    All three doors run the same ``_pipeline``, so all three reached the
    false alarm. A repair pinned at one of them would be a repair whose
    reach was an accident of which function carried the keyword.

    **BOTH FIXTURES REACH IT, AND THE FIRST TWO DID NOT.** The inductive
    step this test was written with had a float state and an assume that
    the step did not need, and it returned VERIFIED at ``cefc4a9`` too —
    a door assertion that could not fail, which is the shape this
    repository keeps paying for. The float version is saved by the exact
    replay (which descends and DOES see the nested assume, declining under
    ``assume-unsatisfied-over-the-rationals``), so reaching the two
    branches that return before the replay needs an INTEGRAL step: ``v *
    2`` over ``int32 [0, 10]``, where the assume ``v <= 5`` is what keeps
    the result inside the invariant. Both fixtures below are driven to
    RAISE at ``cefc4a9``.
    """
    from stelling.contracts import Contract, check_contract
    from stelling.inductive import check_inductive_step

    harness = nested_assume_harness(jax.jit, "int32", 0, 10, 9)
    verdict = check_contract(
        Contract(
            name="nested-assume",
            requires_description="x >= 9 is assumed inside a jit body",
            harness=harness,
            ensures=None,
            no_ensures_reason="this contract states no guarantee",
        ),
        vacuity_mode="inputs-only",
        falsify="sample",
    )
    assert verdict.requires.status == "VERIFIED", verdict.requires.status

    def step(state, constants):
        # the assume is LOAD-BEARING: without it `v * 2` leaves [0, 10] at
        # v = 10, and the step is integral throughout, so the violation is
        # admitted by `exact-integer-arithmetic` before any replay
        v = jax.jit(lambda a: (assume(a <= 5), a)[1])(state["v"])
        return {"v": v * 2}

    verdict = check_inductive_step(
        step, {"v": ((0, 10), "int32")}, falsify="sample"
    )
    assert verdict.status == "VERIFIED", verdict.status
    assert check_inductive_step(
        step, {"v": ((0, 10), "int32")}
    ).status == "VERIFIED", "the fixture does not reach the branch under test"


def test_a_TOP_LEVEL_assume_still_gates_points_exactly_as_before():
    """The repair must not be an off switch for the gate it repairs.

    The same program with the assume left where users write it: points
    below the assumed region are still declined under
    ``assume-unsatisfied``, points inside it are still counted admissible,
    and nothing declines under the new reason because nothing is unread.
    """

    def harness():
        x = any_array((), "int32", (0, 10))
        assume(x >= 9)
        return assert_(x >= 9)

    found, report = attack(harness)
    assert found is None, found and found.render()
    counts = dict(report.skips)
    assert counts.get("assume-unsatisfied", 0) > 0, counts
    assert "assume-not-fully-executed" not in counts, counts
    assert report.points_admissible > 0, report.points_admissible


def test_the_gate_still_FIRES_when_the_assume_admits_the_violating_point():
    """And the other side: a genuinely false obligation still fires.

    ``assume(x >= 5)`` admits ``x = 5`` and the obligation ``x >= 9`` is
    false there, so the analysis discharging it would be a real
    unsoundness and the probe must say so.
    """

    def harness():
        x = any_array((), "int32", (0, 10))
        assume(x >= 5)
        return assert_(x >= 9)

    found, report = attack(harness)
    assert found is not None, f"the gate swallowed a real one: {report.skips}"
    assert found.adjudication == "exact-integer-arithmetic", found.adjudication


def test_a_declaration_the_probe_cannot_VARY_declines_the_whole_probe():
    """``stelling_any`` one ``jit`` deep, which has no value to substitute.

    ``_execute`` substitutes the sampled point at top-level ``stelling_any``
    equations only; one inside a call body is bound instead, and
    ``stelling_any`` has no implementation by design, so the program raises
    at EVERY point. That was already a decline — under ``program-raised``,
    a reason that names the USER's program for a limit of this walk, once
    per point. The reading is partial, so it declines as a partial reading,
    once, saying which.
    """

    def harness():
        x = any_array((), "float64", (0.0, 1.0))
        y = jax.jit(lambda: any_array((), "float64", (0.0, 1.0)))()
        return assert_(x + y <= 2.0)

    found, report = attack(harness)
    assert found is None, found and found.render()
    assert report.declined is not None, report.skips
    assert "at the top level" in report.declined, report.declined
    assert report.points_built == 0, report.points_built
    assert "program-raised" not in dict(report.skips), report.skips


def test_an_obligation_the_probe_cannot_SEE_declines_the_whole_probe():
    """An ``assert_`` one ``jit`` deep: the probe cannot pair the indices.

    This declines today through the OTHER guard — the analysis descends,
    reports two obligations, and the pairing check refuses one top-level
    assert against two statuses. That guard is computed from what the
    ANALYSIS did; this one is computed from the PROGRAM, and it is the
    second that survives the analysis changing its mind about nested
    obligations. Both are asserted, in that order, because a guard that
    only ever fires behind another guard is a guard nobody has driven.
    """

    def harness():
        x = any_array((), "float64", (0.0, 10.0))
        y = jax.jit(lambda a: (assert_(a >= 0.0), a)[1])(x)
        return assert_(y <= 10.0)

    import stelling.falsify as F

    census = F._census(harness)
    assert len(census.assert_positions) == 1, census.assert_positions
    assert census.obligations_in_program == 2, census.obligations_in_program

    # the pairing guard, when the analysis's count is what disagrees
    found, report = attack(harness, n=2)
    assert found is None and report.declined is not None
    assert "cannot pair them" in report.declined, report.declined

    # and the program-side guard, when it is not
    found, report = attack(harness, n=1)
    assert found is None and report.declined is not None
    assert "at the top level" in report.declined, report.declined


def test_every_census_quantity_has_a_totality_guard():
    """THE TEST THAT FAILS WHEN A FIFTH QUANTITY ARRIVES WITHOUT A GUARD.

    Four audits, four defects, one class — and the fourth was reachable
    because two of this module's readings had a totality guard and nobody
    had asked for the rest. Asserts had ``obligation-count-changed``;
    declarations had *"the harness declares no inputs to vary"*, which
    catches zero and not partial; assumes had nothing at all.

    Patching the fourth instance does not close the class, so what is
    pinned here is the RULE rather than the instance: every field of
    ``_Census`` and of ``_Run`` appears in ``_READINGS`` with either the
    decline a partial reading produces, or a written argument that this
    reading cannot license anything. A new field arrives red.

    This test cannot know whether a `why` is a GOOD argument. It can know
    that somebody had to write one down, in the file, next to the field,
    where a reviewer meets it — which is the difference between a decision
    and an omission.
    """
    from dataclasses import fields

    import stelling.falsify as F

    declared = {(r.subject, r.name) for r in F._READINGS}
    actual = {("census", f.name) for f in fields(F._Census)}
    actual |= {("run", f.name) for f in fields(F._Run)}

    unguarded = actual - declared
    assert not unguarded, (
        f"{sorted(unguarded)} is read off the program and has no entry in "
        f"stelling.falsify._READINGS. Give it the totality guard that "
        f"declines when the reading is partial, or record in `why` the "
        f"argument that it cannot license a claim. This module has shipped "
        f"four predicates that licensed a claim about the program while "
        f"being computed from something else; the fourth was a reading "
        f"taken at the top level of a jaxpr that has more than one level."
    )
    stale = declared - actual
    assert not stale, f"_READINGS names {sorted(stale)}, which is not a field"

    for r in F._READINGS:
        assert (r.guard is None) != (not r.why), (
            f"{r.subject}.{r.name} must have exactly one of a guard and a "
            f"written exemption, and has guard={r.guard!r} why={r.why!r}"
        )
        if r.guard is not None:
            assert (
                r.guard in DECLINE_REASONS or r.guard in F._WHOLE_PROBE_GUARDS
            ), (
                f"{r.subject}.{r.name}'s guard {r.guard!r} is neither a "
                f"listed decline reason nor a listed whole-probe decline"
            )
        else:
            # a floor, so that "no guard needed" cannot be discharged with
            # a word. It is not a quality bar and does not pretend to be;
            # it is the difference between an argument and a shrug.
            assert len(r.why) >= 60, (
                f"{r.subject}.{r.name} claims it needs no totality guard "
                f"on {r.why!r}, which is too short to be the argument"
            )
        assert r.depth in ("top-level", "every-depth", "the-program-itself")


def test_the_guards_named_in_the_readings_table_are_LIVE_in_the_source():
    """A table of guards is worth exactly as much as a name list nothing checks.

    This module has been here before: ``_CALL_PRIMITIVES`` named four jax
    primitives that do not exist, under a comment describing what it
    would do if they did, and nothing failed for a batch.

    **AND "SOMEWHERE IN THE FILE" IS NOT A BINDING.** This test used to
    grep each guard as a bare literal anywhere in ``falsify.py``, and only
    three of the five guarded fields had a comparison of their own pinned
    beside it. So a new field could declare ANY decline reason the file
    already emits and be green in both tests — driven: a ``_Census`` field
    added with ``guard="bound-nan"``, a ``_window`` decline about a NaN
    endpoint that has nothing to do with any field, passed. The docstring
    of ``_Reading`` was honest about this for ``why`` and read stronger
    than it was for ``guard``.

    Each reading now carries the SOURCE TEXT of the ``if`` that takes its
    guard, and this test parses ``falsify.py`` and demands that the ``if``
    exist and that the guard be emitted INSIDE it. A guard is then bound
    to one line, not to a file.
    """
    import ast

    import stelling.falsify as F

    tree = ast.parse(PROBE_SRC.read_text(encoding="utf-8"))
    ifs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            ifs.setdefault(ast.unparse(node.test), []).append(node)

    guarded = [r for r in F._READINGS if r.guard is not None]
    assert guarded, "no reading in _READINGS carries a guard at all"
    for r in guarded:
        assert r.site, (
            f"{r.subject}.{r.name} declares guard {r.guard!r} and no "
            f"`site`. A guard name on its own is satisfied by any decline "
            f"reason already spelled anywhere in falsify.py; name the "
            f"`if` that takes THIS field's reading."
        )
        sites = ifs.get(r.site)
        assert sites, (
            f"_READINGS says {r.subject}.{r.name} is guarded by "
            f"`if {r.site}:` and falsify.py has no such `if`. Either the "
            f"guard was moved and the table was not, or it was taken out."
        )
        assert len(sites) == 1, (
            f"`if {r.site}:` appears {len(sites)} times in falsify.py, so "
            f"the table cannot say which one guards {r.subject}.{r.name}"
        )
        if r.guard in F._WHOLE_PROBE_GUARDS:
            # a SENTENCE rather than a code: what is pinned is that the
            # comparison is still taken and still declines the whole probe
            body = ast.unparse(sites[0])
            assert "ProbeReport(" in body and "declined=" in body, (
                f"{r.subject}.{r.name}'s guard no longer declines the "
                f"whole probe at `if {r.site}:`"
            )
            continue
        emitted = {
            n.value
            for n in ast.walk(sites[0])
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        assert r.guard in emitted, (
            f"_READINGS says {r.subject}.{r.name} declines under "
            f"{r.guard!r}, and `if {r.site}:` emits {sorted(emitted)!r}. "
            f"The guard must be taken where the reading is checked; a "
            f"reason emitted somewhere else in the file binds nothing."
        )

    # and the reverse: a `site` that is not attached to a guard is a
    # sentence in the table nothing holds
    for r in F._READINGS:
        assert (r.guard is None) == (not r.site), (
            f"{r.subject}.{r.name} has guard={r.guard!r} and site={r.site!r}"
        )


def test_the_EXECUTED_walk_does_not_descend_into_call_bodies():
    """AND HERE IS WHY, BECAUSE THE OBVIOUS REPAIR IS TO MAKE IT DESCEND.

    ``_execute`` and ``_replay`` are two walkers with different depth
    behaviour and all four of this module's defects have lived in the gap.
    The tempting reconciliation is to make ``_execute`` descend the way
    ``_replay`` already does, which would keep the reach the totality guard
    costs. **It is not available: descending changes the program.**

    ``Primitive.bind`` on a call equation compiles the WHOLE body and XLA
    may contract across the equations inside it; walking those equations
    one at a time asks jax for each primitive separately and gets no
    contraction. This test drives the disagreement rather than describing
    it: with ``c = -fl(a*b)``, the descended walk returns exactly ``0.0``
    and the bound call returns the rounding error of the product — **a
    SIGN disagreement**, and a sign is what an obligation ``>= 0`` reads.

    If ``_execute`` descended, the ``ieee`` branch of ``_confirm`` would be
    firing on *"the executed float IS the subject of the claim"* about a
    float the user's program never computes: the fifth instance of the
    class, manufactured by the repair for the fourth. So the walkers are
    reconciled by MEASUREMENT — each reading checked against a census taken
    at every depth — and not by merging.

    **AND THAT IS DRIVEN AND NOT ARGUED.** A descending ``_execute`` was
    written out in full and run: it does fix the four false alarms this
    batch is about, and on the fixture in the second half of this test it
    raises *"FALSIFICATION PROBE FIRED — stelling is UNSOUND at this
    query"* under ``ieee-executed-float`` at a point where the user's
    program satisfies the obligation. Both halves of this test kill it.
    """
    import stelling.falsify as F

    def walk_descending(jaxpr, consts, args):
        env = {}

        def read(a):
            return a.val if isinstance(a, F.jex_core.Literal) else env[a]

        for v, c in zip(jaxpr.constvars, consts):
            env[v] = c
        for v, a in zip(jaxpr.invars, args):
            env[v] = a
        for eqn in jaxpr.eqns:
            ins = [read(a) for a in eqn.invars]
            if eqn.primitive.name in F._CALL_PRIMITIVES:
                body, body_consts = F._call_jaxpr_of(eqn)
                outs = walk_descending(body, body_consts, ins)
            else:
                out = eqn.primitive.bind(*ins, **eqn.params)
                outs = out if eqn.primitive.multiple_results else [out]
            for var, o in zip(eqn.outvars, outs):
                env[var] = o
        return [read(a) for a in jaxpr.outvars]

    flips = 0
    tried = 0
    for a_, b_ in ((0.61446244, 1.6698782), (1.1576139, 1.5851978),
                   (1.9669843, 1.3077438)):
        a = jnp.asarray(a_, "float32")
        b = jnp.asarray(b_, "float32")
        c = jnp.asarray(-np.asarray(a * b), "float32")
        closed = jax.make_jaxpr(jax.jit(lambda p, q, r: p * q + r))(a, b, c)
        # `_execute`'s policy: the call goes whole to `bind`
        env = {}
        for v, x in zip(closed.jaxpr.invars, (a, b, c)):
            env[v] = x
        eqn = closed.jaxpr.eqns[0]
        assert eqn.primitive.name in F._CALL_PRIMITIVES, eqn.primitive.name
        bound = np.asarray(
            eqn.primitive.bind(*[env[v] for v in eqn.invars], **eqn.params)
        )
        descended = np.asarray(
            walk_descending(closed.jaxpr, closed.consts, (a, b, c))[0]
        )
        tried += 1
        flips += np.sign(bound) != np.sign(descended)

    assert flips == tried, (
        f"only {flips} of {tried} of these bodies still disagree in SIGN "
        f"between the compiled call and an op-by-op walk of its body. That "
        f"is the measurement this module's depth policy rests on; if this "
        f"jax no longer contracts, re-derive the argument in `_execute` "
        f"before changing the policy, do not just delete this test."
    )

    # AND THE COST OF DESCENDING, AT THE PROBE, ON A PROGRAM WHOSE
    # OBLIGATION IS TRUE.  `c` is exactly `-fl32(a0 * b)`, so the body
    # computes the rounding error of the product: nonzero when the call is
    # compiled and exactly zero when its body is walked op by op. `y != 0`
    # therefore HOLDS at every declared point of the real program and FAILS
    # at every point of the descended one. Measured with a descending
    # `_execute` in place: FIRED, `ieee-executed-float`, at
    # a = 1.9669841527938843, where the program computes -9.491779e-08.
    a0 = np.float32(1.9669843)
    b0 = np.float32(1.3077438)
    c0 = np.float32(-np.float32(a0 * b0))
    box = (float(np.nextafter(a0, np.float32(-np.inf))), float(a0))
    prod = jax.jit(lambda p, q, r: p * q + r)

    def rounding_error_of_a_product():
        a = any_array((), "float32", box)
        return assert_(prod(a, jnp.float32(b0), jnp.float32(c0)) != 0.0)

    for endpoint in box:
        computed = np.asarray(
            prod(np.float32(endpoint), b0, c0)
        )
        assert computed != 0, (
            f"at a = {endpoint!r} this jax computes {float(computed)!r} for "
            f"the rounding error of a product, so the fixture no longer "
            f"states something the program satisfies"
        )

    found, report = attack(rounding_error_of_a_product, semantics="ieee")
    assert found is None, (
        f"the probe raised 'stelling is UNSOUND' about an obligation the "
        f"program SATISFIES at that point: {found and found.render()}"
    )
    assert report.points_executed > 0, report.declined

    # and the policy itself, at the source: the loop binds the call
    src = PROBE_SRC.read_text(encoding="utf-8")
    body = src.split("def _execute(", 1)[1].split("\ndef ", 1)[0]
    assert "_call_jaxpr_of" not in body, (
        "`_execute` now descends into call bodies. Read its docstring: the "
        "executed float stops being the one the user's program computes, "
        "and `ieee-executed-float` licenses a firing on exactly that."
    )


def test_a_float_step_hidden_in_a_STRUCTURED_body_is_not_integer_arithmetic():
    """``_integral_program`` recurses UNCONDITIONALLY, and that is the pin.

    ``test_the_integer_admission_IS_A_READING_OF_THE_PROGRAM_at_the_source``
    asserts that ``_integral_program`` calls itself somewhere. It does not
    assert that the recursion is unconditional, and a mutant that restricts
    it to ``if eqn.primitive.name in _CALL_PRIMITIVES`` survives the whole
    falsify suite while answering **True** for each of the three programs
    below — every one of which converts to float32 and rounds inside a
    ``cond`` arm, a ``scan`` body or a ``while`` body, behind an
    all-integer signature.

    Not exploitable today: ``propagate`` returns UNKNOWN on all three, so
    no VERIFIED exists for the probe to attack. That is a fact about
    another module, which is exactly the kind of fact this file does not
    lean on — the third defect was a predicate that was *"correlated with
    the thing it stood for"*, and *"nothing reaches it"* is the same
    argument one level out.
    """
    import stelling.falsify as F

    def rounds(a):
        y = a.astype("float32")
        b = jnp.float32(2 ** 24)
        return ((b + y) - b).astype("int32")

    def in_a_cond():
        x = any_array((), "int32", (0, 3))
        return assert_(jax.lax.cond(x > 1, rounds, lambda a: a, x) <= 3)

    def in_a_scan():
        x = any_array((), "int32", (0, 3))
        out, _ = jax.lax.scan(
            lambda c, _: (rounds(c), 0), x, jnp.zeros((2,), "int32")
        )
        return assert_(out <= 3)

    def in_a_while():
        x = any_array((), "int32", (0, 3))
        out = jax.lax.while_loop(
            lambda s: s[1] < 1,
            lambda s: (rounds(s[0]), s[1] + 1),
            (x, jnp.int32(0)),
        )[0]
        return assert_(out <= 3)

    for name, harness in (
        ("cond", in_a_cond), ("scan", in_a_scan), ("while", in_a_while)
    ):
        closed = jax.make_jaxpr(harness)()
        # the fixture is only interesting if the OUTER program is integral
        assert all(
            F._integral_atom(atom)
            for eqn in closed.jaxpr.eqns
            for atom in (*eqn.invars, *eqn.outvars)
        ), f"[{name}] the float step is not hidden; wrong fixture"
        assert F._integral_program(closed.jaxpr) is False, (
            f"[{name}] a float32 rounding step inside a {name} body is "
            f"called 'exact integer arithmetic: no rounding involved'"
        )
        assert F._census(harness).integral is False, name

    # and at the source, so the shape of the mutant is refused and not just
    # the three programs someone thought of: the recursion is a direct
    # statement of the equation loop, never guarded by a primitive name
    import ast

    tree = ast.parse(PROBE_SRC.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_integral_program"
    )
    eqn_loop = next(
        n for n in fn.body
        if isinstance(n, ast.For) and "eqns" in ast.unparse(n.iter)
    )
    assert any(
        isinstance(stmt, ast.For) and "_sub_jaxprs" in ast.unparse(stmt.iter)
        for stmt in eqn_loop.body
    ), (
        "`_integral_program`'s recursion is no longer an unconditional "
        "statement of its equation loop. A recursion restricted to "
        "`_CALL_PRIMITIVES` answers True for a float step inside a "
        "`cond`/`scan`/`while` body."
    )


def test_a_point_no_assume_admitted_is_not_counted_as_ADMITTED_BY_EVERY_ASSUME():
    """The stamp line contradicted itself in one sentence, and this is it.

    Measured on the fixture below at ``cefc4a9``: *"74 point(s) executed,
    65 inside the declared set and admitted by every assume ... declined 39
    assume-unsatisfied-over-the-rationals"*. Thirty-nine of the
    sixty-five were points the exact replay had just said NO assume
    admitted — the reads-as-coverage failure this module's docstring exists
    to prevent, in the module's own headline number.

    The two readings run at different times: the float gate admits the
    point, and only the replay, which runs where the float found a
    violation, can say otherwise. So the count is TAKEN BACK. It is not
    moved to ``points_declined`` either — that number means *"an admissible
    violation the probe would not stand behind"*, and this point was never
    admissible.
    """

    def rounding_assume():
        y = any_array((), "float64", (0.0, 2.0))
        assume(y * 0.1 * 10.0 <= y)
        s = 1e16
        return assert_((s + y) - s <= y)

    found, report = attack(rounding_assume)
    assert found is None, found and found.render()
    counts = dict(report.skips)
    outside = counts.get("assume-unsatisfied-over-the-rationals", 0)
    assert outside > 0, counts

    # EVERY skip this fixture produces is a whole-POINT decline — no
    # obligation-level one, which could legitimately fire more than once at
    # a point that IS admissible — so the three numbers have to add up
    # exactly. Asserted rather than assumed, so that a fixture drifting
    # into a mixed regime turns this red instead of passing by luck.
    assert set(counts) <= {
        "assume-unsatisfied", "assume-unsatisfied-over-the-rationals"
    }, counts
    accounted = report.points_admissible + sum(counts.values())
    assert accounted == report.points_built, (
        f"{report.points_built} built, {report.points_admissible} counted "
        f"admissible and {sum(counts.values())} declined: {accounted}"
    )
    assert report.points_declined == 0, (
        f"{report.points_declined} point(s) reported as admissible "
        f"violations the probe would not stand behind, when no assume "
        f"admitted them"
    )
    line = report.stamp_line()
    assert "EXECUTED VIOLATION(S) WERE DECLINED" not in line, line
    assert f"{report.points_admissible} inside the declared set" in line, line


ORDINARY_WRAPPERS = [
    ("plain", lambda a: a * 2.0),
    ("jit", jax.jit(lambda a: a * 2.0)),
    ("jit in jit", jax.jit(lambda a: jax.jit(lambda b: b * 2.0)(a))),
    ("checkpoint", jax.checkpoint(lambda a: a * 2.0)),
    ("cond", lambda a: jax.lax.cond(
        a > 0.5, lambda b: b * 2.0, lambda b: b, a)),
    ("scan", lambda a: jax.lax.scan(
        lambda c, _: (c * 1.0, 0.0), a, jnp.zeros((2,)))[0]),
    ("while", lambda a: jax.lax.while_loop(
        lambda s: s[1] < 1, lambda s: (s[0] * 1.0, s[1] + 1), (a, 0))[0]),
    ("grad", jax.grad(lambda a: (a * 2.0).sum())),
]


@pytest.mark.parametrize("name,wrapper", ORDINARY_WRAPPERS)
def test_the_totality_walk_counts_each_declaration_ONCE(name, wrapper):
    """A guard that OVER-counts declines programs that are perfectly fine.

    ``_declaration_totals`` walks through ``_sub_jaxprs``, which follows a
    jaxpr wherever a PARAMETER carries one — and a primitive that carried
    the same body under two parameter keys would be counted twice, so the
    executed run's reading would always come up short and the probe would
    decline every such program under ``assume-not-fully-executed``. That is
    the conservative direction and therefore the direction nobody notices:
    it costs reach silently.

    So the total is compared against the top-level count on ordinary jax
    constructs, where they must agree exactly because nothing is nested,
    and against what ``_execute`` actually reads.
    """
    import stelling.falsify as F

    def harness():
        x = any_array((), "float64", (0.0, 1.0))
        assume(x >= 0.0)
        return assert_(jnp.sum(wrapper(x)) >= 0.0)

    closed = jax.make_jaxpr(harness)()
    totals = F._declaration_totals(closed.jaxpr)
    top = {
        n: sum(1 for e in closed.jaxpr.eqns if e.primitive.name == n)
        for n in totals
    }
    assert totals == top, (
        f"[{name}] the every-depth walk counts {totals} where the top level "
        f"has {top}; a declaration counted twice declines a program the "
        f"probe could read"
    )

    census = F._read(closed)
    run = F._execute(census, [jnp.asarray(np.float64(0.5))])
    assert run.raised is None, run.raised
    assert len(run.assumes) == census.assumes_in_program == 1, (
        f"[{name}] the executed run read {len(run.assumes)} assume(s) "
        f"against a census of {census.assumes_in_program}"
    )

    found, report = attack(harness)
    assert found is None, found and found.render()
    assert "assume-not-fully-executed" not in dict(report.skips), report.skips
    assert report.declined is None, report.declined


# ------------------------- a reading of the program can be at the WRONG GRAIN


MEAN_X0 = 1.3102272059107631


def mean_of_a_stack(x):
    """``mean([x, 2x, 3x])`` — a program with no ``jit`` written in it."""
    return jnp.mean(jnp.stack([x, x * 2.0, x * 3.0]))


def mean3_at_its_own_value():
    """An obligation the program satisfies at the only point it declares.

    ``C`` is the value the PROGRAM computes at ``MEAN_X0``, read back from
    the program itself, so ``mean3(x) <= C`` is true there by construction
    — eagerly and under ``jax.jit`` alike. A firing on it is a false alarm
    whatever a solver thinks, because the bound came from the program.
    """
    c = float(np.asarray(mean_of_a_stack(jnp.asarray(MEAN_X0, "float64"))))

    def harness():
        x = any_array((), "float64", (MEAN_X0, MEAN_X0))
        return assert_(mean_of_a_stack(x) <= c)

    return harness


def test_the_EXECUTED_walk_reads_the_program_at_the_TRACES_granularity():
    """THE FIFTH INSTANCE, INSIDE THE SENTENCE THAT JUSTIFIED THE FOURTH.

    ``_execute`` binds one equation at a time, so XLA never sees two of
    them together. The docstring that justified this defended it with
    *"what this loop reproduces is what the user's own code does: their
    un-jitted top level op by op, their ``jit`` compiled whole."* **jax
    does not divide the two that way.** ``jnp.mean`` is a compiled region
    on the eager path and ``jax.make_jaxpr`` INLINES it, so the traced top
    level carries a bare ``reduce_sum ; div`` for a call the caller
    compiles whole — and the two give different floats, one ulp apart.

    So this test drives the three facts in order: the trace has no call
    equation to hand whole to ``bind``; the two routes disagree; and the
    probe, which used to raise *"stelling is UNSOUND at this query"* on
    the obligation below, declines it by name.
    """
    import stelling.falsify as F

    x = jnp.asarray(MEAN_X0, "float64")
    closed = jax.make_jaxpr(mean_of_a_stack)(x)
    names = [e.primitive.name for e in closed.jaxpr.eqns]
    assert not any(n in F._CALL_PRIMITIVES for n in names), (
        f"this jax no longer inlines the `jit` inside `jnp.mean`: {names}. "
        f"Re-derive the argument in `_execute` before relaxing the guard — "
        f"a top-level call equation is one `_execute` hands whole to `bind`"
    )
    assert "reduce_sum" in names and "div" in names, names

    # the two routes, on the same jaxpr at the same point
    eager = float(np.asarray(mean_of_a_stack(x)))
    census = F._read(closed)
    env = {v: c for v, c in zip(closed.jaxpr.constvars, closed.consts)}
    env[closed.jaxpr.invars[0]] = x
    for eqn in closed.jaxpr.eqns:
        ins = [
            a.val if isinstance(a, F.jex_core.Literal) else env[a]
            for a in eqn.invars
        ]
        out = eqn.primitive.bind(*ins, **eqn.params)
        for var, o in zip(
            eqn.outvars, out if eqn.primitive.multiple_results else [out]
        ):
            env[var] = o
    op_by_op = float(np.asarray(env[closed.jaxpr.outvars[0]]))
    assert op_by_op != eager, (
        f"the op-by-op walk and the eager call now agree at {MEAN_X0!r} "
        f"({eager!r}); this fixture no longer states the defect it was "
        f"written for"
    )

    # and the probe: a violation at the TRACE's granularity, declined
    found, report = attack(mean3_at_its_own_value(), semantics="ieee")
    assert found is None, (
        f"the probe raised 'stelling is UNSOUND' about an obligation the "
        f"program SATISFIES at its only declared point, because it "
        f"evaluated the program one equation at a time: "
        f"{found and found.render()}"
    )
    counts = dict(report.skips)
    assert counts.get("executed-float-depends-on-granularity", 0) > 0, (
        f"the violation was not declined for the right reason: {counts}"
    )
    assert report.violations_seen > 0, (
        "nothing violated at all; the fixture no longer reaches the branch"
    )
    assert dict(report.adjudications) == {
        "declined-executed-routes-disagree": report.violations_seen
    }, report.adjudications


def test_the_whole_program_route_agrees_with_the_CALLERS_OWN_CALL():
    """The second route is not just a second answer; it is the right one.

    ``_granularity_stable`` only ever DECLINES, so it would be safe even
    if its route were arbitrary — but a guard that declines for a reason
    that is not the real one declines the wrong programs. On this fixture
    the whole-program route computes what the caller's own eager call
    computes, and the op-by-op walk does not, which is the whole content
    of the claim that ``_execute`` reads at the wrong grain.
    """
    import stelling.falsify as F

    harness = mean3_at_its_own_value()
    census = F._census(harness)
    point = [jnp.asarray(MEAN_X0, "float64")]
    run = F._execute(census, point)
    _, asserts = F._whole_program_route(census)(*point)

    assert bool(np.all(run.asserts[0])) is False, (
        "the op-by-op walk no longer violates the obligation here"
    )
    assert bool(np.all(np.asarray(asserts[0]))) is True, (
        "the whole-program route no longer agrees with the caller's own "
        "call; the fixture has stopped separating the two granularities"
    )
    assert F._granularity_stable(F._whole_program_route(census), point,
                                 run, 0) is False


def test_the_granularity_guard_costs_the_JIT_FIXTURE_NOTHING():
    """The other fixture in this file goes through the new guard unchanged.

    The ``a * b + c`` program is the one that refuses a DESCENDING
    ``_execute``. Its call equation is handed whole to ``bind`` by both
    routes, so they agree, and the guard must not decline it — a guard
    that declined everything would pass the test above and be worthless.
    """
    import stelling.falsify as F

    a0 = np.float32(1.9669843)
    b0 = np.float32(1.3077438)
    c0 = np.float32(-np.float32(a0 * b0))
    box = (float(np.nextafter(a0, np.float32(-np.inf))), float(a0))
    prod = jax.jit(lambda p, q, r: p * q + r)

    def harness():
        a = any_array((), "float32", box)
        return assert_(prod(a, jnp.float32(b0), jnp.float32(c0)) != 0.0)

    census = F._census(harness)
    route = F._whole_program_route(census)
    for endpoint in box:
        point = [jnp.asarray(endpoint, "float32")]
        run = F._execute(census, point)
        assert F._granularity_stable(route, point, run, 0) is True, (
            f"the two routes disagree at {endpoint!r} on a program whose "
            f"only call equation both of them hand whole to `bind`"
        )

    found, report = attack(harness, semantics="ieee")
    assert found is None, found and found.render()
    assert "executed-float-depends-on-granularity" not in dict(report.skips), (
        f"the guard declined a program it must not: {report.skips}"
    )


def test_a_violation_that_survives_BOTH_ROUTES_still_fires():
    """The guard declines a moving reading and nothing else.

    ``x + 1.0 > x`` over ``[0, 2**54]`` is FALSE in float64 above the
    doubling point, and it is false at the same points however much of the
    program XLA compiles together — there is nothing here to contract. So
    the ``ieee`` firing this instrument is for must survive the guard.
    """

    def harness():
        x = any_array((), "float64", (0.0, 2.0 ** 54))
        return assert_(x + 1.0 > x)

    found, report = attack(harness, semantics="ieee")
    assert found is not None, (
        f"the granularity guard swallowed a genuine `ieee` refutation: "
        f"{report.stamp_line()}"
    )
    assert found.adjudication == "ieee-executed-float", found.adjudication


def test_a_second_route_that_cannot_be_run_is_not_agreement(monkeypatch):
    """``None`` is not ``True``, and the code must not read it as one.

    No program is known that ``_execute`` can walk and that then fails to
    stage under a single ``jit``, so the branch is driven with a route
    that raises. It is the direction that matters: an unchecked
    granularity declines under its own reason and never admits.
    """
    import stelling.falsify as F

    def unavailable(census):
        def raises(*_):
            raise RuntimeError("no")

        return raises

    monkeypatch.setattr(F, "_whole_program_route", unavailable)

    def harness():
        x = any_array((), "float64", (0.0, 2.0 ** 54))
        return assert_(x + 1.0 > x)

    found, report = attack(harness, semantics="ieee")
    assert found is None, (
        f"a violation was admitted although the second route could not be "
        f"run: {found.render()}"
    )
    assert dict(report.skips).get("whole-program-route-unavailable", 0) > 0, (
        report.skips
    )
    assert dict(report.adjudications) == {
        "declined-whole-program-route-unavailable": report.violations_seen
    }, report.adjudications


def test_the_granularity_guard_is_handed_THE_SAME_POINT(monkeypatch):
    """Re-execution at a NEIGHBOURING point is the ulp proxy, respelled.

    ``test_the_ulp_proxy_is_gone_from_the_fire_condition_entirely`` bans
    re-execution from ``_confirm`` for exactly that reason. This guard
    does re-execute, so the distinction it rests on is asserted rather
    than described: the second route is handed the SAME point object the
    executed run was handed, so it can only ever say that the two readings
    of ONE point disagree — never that a neighbouring point behaves
    differently, which is a fact about the neighbourhood and not about
    this violation.
    """
    import stelling.falsify as F

    seen = []
    real = F._whole_program_route

    def recording(census):
        route = real(census)

        def wrapped(*vals):
            seen.append(tuple(float(np.asarray(v)) for v in vals))
            return route(*vals)

        return wrapped

    monkeypatch.setattr(F, "_whole_program_route", recording)
    monkeypatch.setattr(
        F, "_execute",
        lambda census, point, _f=F._execute: (
            seen.append(tuple(float(np.asarray(v)) for v in point)) or None
        ) or _f(census, point),
    )

    found, _ = attack(mean3_at_its_own_value(), semantics="ieee")
    assert found is None
    assert seen, "neither walker ran"
    assert len(set(seen)) == 1, (
        f"the two routes were run at different points {sorted(set(seen))}; "
        f"a second execution at a NEIGHBOURING point is the ulp proxy "
        f"under another name"
    )


# --------------- the evaluator's ARITHMETIC, which its name tables do not pin


RATIONAL_READINGS_AGAINST_JAX = [
    # (label, the exact-rational reading, the same computation in jax)
    #
    # Every jax result below is exactly representable, so "agrees with jax"
    # is a statement about the READING and not about float error. Values
    # are chosen to separate the readings from their near neighbours:
    # truncation from rounding, an integer exponent from a fractional one,
    # a perfect square from an irrational root.
    (
        "convert float->int32 at 7/4 truncates, and does NOT round",
        lambda: _rat_convert(Fraction(7, 4), "f", np.dtype("int32")),
        lambda: jnp.asarray(1.75, "float64").astype("int32"),
    ),
    (
        "convert float->int32 at -7/4 truncates TOWARD ZERO",
        lambda: _rat_convert(Fraction(-7, 4), "f", np.dtype("int32")),
        lambda: jnp.asarray(-1.75, "float64").astype("int32"),
    ),
    (
        "convert float->int32 at 3/2, where round() goes the other way",
        lambda: _rat_convert(Fraction(3, 2), "f", np.dtype("int32")),
        lambda: jnp.asarray(1.5, "float64").astype("int32"),
    ),
    (
        "convert float->int64 at -1/3, where trunc and floor differ",
        lambda: _rat_convert(Fraction(-1, 3), "f", np.dtype("int64")),
        lambda: jnp.asarray(-1.0 / 3.0, "float64").astype("int64"),
    ),
    (
        "convert float->float32 is the IDENTITY over R, not a rounding",
        lambda: _rat_convert(Fraction(1, 3), "f", np.dtype("float32")),
        None,  # jax rounds here; R has no such operation. See below.
    ),
    (
        "integer exponent: 3 ** 2",
        lambda: _rat_pow(Fraction(3), Fraction(2)),
        lambda: jnp.power(jnp.asarray(3.0, "float64"), 2.0),
    ),
    (
        "integer exponent, negative: 2 ** -3",
        lambda: _rat_pow(Fraction(2), Fraction(-3)),
        lambda: jnp.power(jnp.asarray(2.0, "float64"), -3.0),
    ),
    (
        "FRACTIONAL exponent on a perfect square: 4 ** (1/2) is 2, not 4",
        lambda: _rat_pow(Fraction(4), Fraction(1, 2)),
        lambda: jnp.power(jnp.asarray(4.0, "float64"), 0.5),
    ),
    (
        "FRACTIONAL exponent, cube root of a cube: 8 ** (1/3) is 2, not 8",
        lambda: _rat_pow(Fraction(8), Fraction(1, 3)),
        lambda: jnp.power(jnp.asarray(8.0, "float64"), 1.0 / 3.0),
    ),
    (
        "sqrt of a perfect square: 4",
        lambda: _rat_sqrt(Fraction(4)),
        lambda: jnp.sqrt(jnp.asarray(4.0, "float64")),
    ),
    (
        "sqrt of a perfect rational square: 9/4",
        lambda: _rat_sqrt(Fraction(9, 4)),
        lambda: jnp.sqrt(jnp.asarray(2.25, "float64")),
    ),
]


def test_the_rational_readings_AGREE_WITH_JAX_where_jax_is_exact():
    """THE TABLES' NAMES ARE PINNED TO A LIVE TRACE; THE READINGS WERE NOT.

    ``_CALL_PRIMITIVES`` taught this module that a name list is worth what
    checks it, and the answer was a test that traces live jax. The other
    predicate a firing rests on — *"and it is false over ℚ"* — had no
    such test at all: which primitive names the replay claims to read is
    pinned, what it reads them AS is not.

    That is the direction this evaluator's own comment names as the one
    place it could INVENT a refutation — *"a name that matches with the
    WRONG READING invents a refutation"* — and three one-token mutations
    reach it with the whole falsify suite green: ``math.trunc`` to
    ``round`` in ``_rat_convert``, dropping the integer-exponent guard in
    ``_rat_pow``, and ``Fraction(math.sqrt(a))`` in ``_rat_sqrt``. Each
    one raises *"stelling is UNSOUND at this query"* on an obligation that
    is TRUE over ℝ; the ``_rat_pow`` one does it on a real ``VERIFIED``
    through ``check(semantics="real")``, and the ``_rat_sqrt`` one
    survives the entire repository.

    So the readings are checked against jax's own arithmetic, at values
    where jax's answer is exact and where the mutant's answer is not the
    same number. **An abstention is always allowed** — it costs reach and
    can never invent a firing — so what is asserted is the implication: if
    a reading is produced, it is the number jax produces.
    """
    for label, reading, in_jax in RATIONAL_READINGS_AGAINST_JAX:
        if in_jax is None:
            continue
        try:
            got = reading()
        except _Unreplayable:
            continue
        raw = np.asarray(in_jax())
        want = Fraction(raw.item())
        assert got == want, (
            f"[{label}] the exact-rational replay reads this as {got!r} "
            f"and jax computes {raw.item()!r}. A reading that is not what "
            f"the program computes is how this evaluator INVENTS a "
            f"refutation, which is the one direction it can be wrong in "
            f"that costs more than reach."
        )


def test_the_rational_readings_obey_their_own_ALGEBRA():
    """And the roots, where jax's own answer is NOT exact and cannot arbitrate.

    ``jnp.sqrt(2.0)`` is a float, and ``Fraction`` of that float is a
    perfectly good rational — just not one whose square is 2. So the
    ``_rat_sqrt`` mutant agrees with jax to every bit jax has and is still
    wrong, and only the algebra catches it: **whatever this evaluator
    returns for a root must BE a root**, exactly, over ℚ. Same for a
    power: ``v ** k.denominator == a ** k.numerator``, which is the
    defining property and holds for the negative and fractional cases too.
    """
    for a in (Fraction(4), Fraction(9, 4), Fraction(1), Fraction(0),
              Fraction(2), Fraction(3), Fraction(1, 3), Fraction(5, 7)):
        try:
            v = _rat_sqrt(a)
        except _Unreplayable:
            continue
        assert v * v == a, (
            f"_rat_sqrt({a}) returned {v}, whose square is {v * v}. An "
            f"approximate root read as exact makes a TRUE obligation false "
            f"over ℚ, which is a firing this instrument invented."
        )

    for a, k in ((Fraction(3), Fraction(2)), (Fraction(2), Fraction(-3)),
                 (Fraction(4), Fraction(1, 2)), (Fraction(8), Fraction(1, 3)),
                 (Fraction(9, 4), Fraction(3, 2)), (Fraction(5), Fraction(1)),
                 (Fraction(-2), Fraction(3))):
        try:
            v = _rat_pow(a, k)
        except _Unreplayable:
            continue
        assert v ** k.denominator == a ** k.numerator, (
            f"_rat_pow({a}, {k}) returned {v}, and {v} ** {k.denominator} "
            f"is not {a} ** {k.numerator}. A fractional exponent read as "
            f"an integer one is the mutation that fires on "
            f"`power(x, 0.5) ** 2 <= x` over [0, 4], where the obligation "
            f"is true over ℝ at every point."
        )


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "int32"])
def test_the_integer_range_guard_agrees_with_where_jax_WRAPS(dtype):
    """``_int_ok`` is the other half: rational arithmetic does not wrap.

    Its boundary is not a convention — jax's own arithmetic decides where
    it is, and it is asserted here by making jax wrap rather than by
    reading ``iinfo`` twice. Both halves of the guard are driven: a value
    outside the dtype, and a value that is not an integer at all, which is
    a value the program could not have produced in an integer register.
    """
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    lo, hi = int(info.min), int(info.max)

    _int_ok([Fraction(lo), Fraction(0), Fraction(hi)], dt)

    wrapped = int(np.asarray(jnp.asarray(hi, dtype) + jnp.asarray(1, dtype)))
    assert wrapped != hi + 1, (
        f"{dtype} no longer wraps at its maximum in this jax ({wrapped}); "
        f"the guard's boundary is defined by what the program does there"
    )
    for outside in (Fraction(lo - 1), Fraction(hi + 1)):
        with pytest.raises(_Unreplayable, match="left its dtype"):
            _int_ok([outside], dt)
    with pytest.raises(_Unreplayable, match="left its dtype"):
        _int_ok([Fraction(1, 2)], dt)


def test_the_float_to_float_reading_is_the_IDENTITY_and_says_why():
    """The one entry above with no jax arbiter, asserted on its own terms.

    ``convert_element_type`` from float64 to float32 ROUNDS, and ℝ has no
    rounding — so under ``semantics="real"`` the exact reading is the
    identity, and jax's float32 answer is exactly the thing the claim
    under attack is not about. Pinned here rather than left implicit,
    because "agrees with jax" is the rule everywhere else in this section
    and this is the documented exception.
    """
    a = Fraction(1, 3)
    assert _rat_convert(a, "f", np.dtype("float32")) == a
    assert _rat_convert(a, "f", np.dtype("float64")) == a
    assert _rat_convert(a, "f", np.dtype("float16")) == a
    narrowed = Fraction(float(np.asarray(jnp.asarray(1.0, "float64") / 3.0,
                                         dtype="float32")))
    assert narrowed != a, (
        "float32 no longer rounds 1/3, so this test no longer states the "
        "distinction it exists for"
    )


# ==========================================================================
# THE PROBE'S PROGRAM IS THE VERDICT'S PROGRAM
# ==========================================================================
#
# The probe used to call `jax.make_jaxpr(harness)()` for ITSELF, and
# whether that re-ran the harness body was decided by jax's own trace
# memo. `preconditions._pipeline` defeats that memo on purpose when the
# overflow tripwire is armed -- `jax.clear_caches()` and a fresh closure,
# so the trace happens under the instrument -- so an impure harness handed
# the probe a genuinely different program from the one the verdict is
# about, and NOTHING compared the two. Every totality guard in the module
# compares the probe's reading against the probe's own second trace.
#
# The subprocess is not decoration: arming the tripwire is process-global
# and reaches a private jax registry, so a test that armed it in-process
# would leave four other test files running under an instrument they did
# not ask for.

_IMPURE = '''
import jax
jax.config.update("jax_enable_x64", True)
from stelling.harness import any_array, assert_
from stelling.preconditions import check
from stelling.falsify import VerifiedFalsified

CALLS = []

def harness():
    x = any_array((), "float64", (0.0, 10.0))
    CALLS.append(1)
    if len(CALLS) == 1:
        return assert_(x >= 0.0)      # TRUE on [0, 10]: the VERIFIED is right
    return assert_(x <= 1.0)          # a DIFFERENT program, false on (1, 10]

if %(arm)r:
    from stelling import _tripwire
    _tripwire.arm()
    assert _tripwire.fires_count() is not None, "the tripwire did not arm"

try:
    v = check(harness, vacuity_mode="inputs-only", falsify="sample")
    print("STATUS", v.status)
except VerifiedFalsified as exc:
    print("FIRED", str(exc).splitlines()[0])
print("BODY-CALLS", len(CALLS))
'''


def _run_isolated(script):
    src = pathlib.Path(stelling.__file__).resolve().parent
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(src.parent),
            "PATH": "/usr/bin:/bin",
            "JAX_PLATFORMS": "cpu",
            "HOME": "/tmp",
        },
    )
    return proc


@pytest.mark.parametrize("arm", [False, True])
def test_the_probe_reads_the_program_the_VERDICT_is_about(arm):
    """Same call, same stelling: the answer may not depend on an instrument.

    A harness that returns a different obligation on its second call is
    VERIFIED, correctly, on its first program. With the probe tracing for
    itself, arming the overflow tripwire — which ``check()``'s own
    docstring recommends — turned that into *"FALSIFICATION PROBE FIRED —
    stelling is UNSOUND at this query"*, because the tripwire's gate
    evicts jax's caches and traces through a fresh closure and the probe's
    ``make_jaxpr`` then genuinely re-ran the body.

    The impurity is the INSTRUMENT here and not the subject: what is being
    measured is that the probe reads the analysis's program, and an impure
    harness is simply the only way to make "a second trace" visible from
    outside.

    ``_tripwire.arm()`` rather than a nested ``pytest -p
    stelling.overflow`` because it is the same call —
    ``stelling._tripwire.plugin`` makes it at ``pytest_configure`` — and a
    subprocess is cheaper than a session inside a session. The documented
    spelling was driven by hand on both trees and gives the same two
    answers: on ``115d771``, ``pytest`` -> VERIFIED with one body call and
    ``pytest -p stelling.overflow`` -> *"FALSIFICATION PROBE FIRED —
    stelling is UNSOUND at this query"* with two; here, VERIFIED with one
    body call under both.
    """
    proc = _run_isolated(_IMPURE % {"arm": arm})
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "STATUS VERIFIED" in proc.stdout, (
        f"arm={arm}: the correct VERIFIED did not survive.\n{proc.stdout}\n"
        f"{proc.stderr[-2000:]}"
    )
    assert "FIRED" not in proc.stdout, proc.stdout
    assert "BODY-CALLS 1" in proc.stdout, (
        f"arm={arm}: the harness body ran more than once, so something "
        f"traced a second program.\n{proc.stdout}"
    )


def test_the_probe_refuses_a_harness_and_says_what_to_pass_instead():
    """The parameter change is the fix; the door has to say so.

    The minimum acceptable outcome for this defect was a guard that
    compares the probe's program with the analysis's and declines when
    they differ. What shipped is stronger and is why no such guard exists:
    the probe cannot obtain a second program, because it is handed one and
    has no tracer call left in it.
    """
    def h():
        x = any_array((), "float64", (0.0, 1.0))
        return assert_(x <= 1.0)

    with pytest.raises(TypeError, match="not a harness"):
        probe(h, statuses=["discharged"])
    with pytest.raises(TypeError, match="trace_with_jaxpr"):
        probe(h, statuses=["discharged"])
    # and the object the pipeline hands it is accepted
    from stelling._jax_compat import trace_with_jaxpr

    query, closed = trace_with_jaxpr(h)
    report = probe(closed, statuses=["discharged"])
    assert report.declined is None and report.points_executed > 0, report


def test_one_trace_feeds_the_transcription_and_the_probe():
    """``trace_with_jaxpr`` returns both halves of ONE trace.

    The transcription is what the analysis judges and jax's object is what
    the probe executes. Two traces of the same callable are not
    independence — it is the same tracer on the same function — and they
    are the failure mode above, so the pipeline takes one.
    """
    from stelling._jax_compat import trace, trace_with_jaxpr

    calls = []

    def h():
        calls.append(1)
        x = any_array((), "float64", (0.0, 1.0))
        return assert_(x <= 1.0)

    query, closed = trace_with_jaxpr(lambda: h())
    assert len(calls) == 1, calls
    # jax's object, not the transcription: the probe may not read `ir`
    assert hasattr(closed, "jaxpr") and not hasattr(closed, "content_hash")
    assert hasattr(query, "content_hash")
    # content hash, not object equality: `source_info` records the frame
    # the declaration was written in and the two lambdas here are on
    # different lines
    assert query.content_hash() == trace(lambda: h()).content_hash()


# ==========================================================================
# A FIRING NAMES THE RIGHT PARTY
# ==========================================================================


def _under_declared_exp():
    """A harness whose ``ieee`` VERIFIED exists ONLY because of a bad budget.

    ``exp`` at ``X`` in float32: XLA's answer is about six ulps below the
    correctly-rounded one, which is what the shipped profile declares. A
    hand-written 0-ulp profile brackets ``exp`` too tightly, discharges an
    obligation the executed program violates, and the honest profile
    returns UNKNOWN on the same harness — so the counterexample is real
    and it is the DECLARATION that is wrong, not stelling.
    """
    from stelling.propagate import LibmBudget

    X = np.float32(88.72167205810547)
    L = np.float32(3.398854604474046e+38)

    def harness():
        x = any_array((), "float32", (float(X), float(X)))
        return assert_(jnp.exp(x) >= jnp.float32(L))

    wishful = LibmBudget(
        name="wishful-exp-0ulp",
        basis="written by hand for this test; NOT a measurement of any backend",
        ulps={("exp", "float32"): 0.0},
    )
    return harness, wishful


def test_the_shipped_libm_budget_line_carries_the_phrase_the_probe_splits_on():
    """The coupling is a PHRASE in data, so a test holds it.

    ``falsify`` may not import ``propagate``; it recognises an unverified
    caller declaration by the words the stamp uses. A test is under no
    such constraint, so it reads the real line and asserts the phrase is
    in it — otherwise the split below would silently stop happening.
    """
    from stelling.falsify import DECLARED_NOT_VERIFIED, unverified_declarations
    from stelling.propagate import LIBM_PROFILES

    profile = LIBM_PROFILES["xla-cpu-2026-08"]
    line = profile.render({op for (op, _d), _u in profile.ulps})
    assert DECLARED_NOT_VERIFIED in line, line
    assert unverified_declarations((line, "some other assumption")) == (line,)
    assert unverified_declarations(("some other assumption",)) == ()


def test_a_firing_on_a_DECLARED_verdict_does_not_accuse_stelling():
    """The counterexample is real; the attribution was not.

    ``_confirm`` returns on ``ieee-executed-float`` before any exact test,
    and an ``ieee`` VERIFIED can rest on a ``libm_budget`` the stamp marks
    *"DECLARED, NOT VERIFIED … stelling checks NEITHER"*. Under-declaring
    it made the probe raise *"stelling is UNSOUND at this query"* — an
    ``AssertionError`` that stops the caller's CI and points them at
    ``stelling/falsify.py`` — for the caller's own declaration.
    """
    harness, wishful = _under_declared_exp()

    # the honest profile does not mint the VERIFIED at all, which is what
    # makes this a defect in the declaration rather than in the analysis
    honest = check(harness, vacuity_mode="inputs-only", semantics="ieee",
                   libm_budget="xla-cpu-2026-08", falsify="sample")
    assert honest.status == "UNKNOWN", honest.status

    with pytest.raises(VerifiedFalsified) as caught:
        check(harness, vacuity_mode="inputs-only", semantics="ieee",
              libm_budget=wishful, falsify="sample")
    text = str(caught.value)
    assert "FALSIFICATION PROBE FIRED" in text
    assert "RESTS ON A DECLARATION stelling does not check" in text
    assert "stelling is UNSOUND at this query" not in text, text
    assert "THIS IS NOT A REPORT THAT STELLING IS UNSOUND" in text
    # and it names the profile, which is the thing to go and check
    assert "wishful-exp-0ulp" in text, text
    # the violation is still REPORTED and still RAISES: declining would
    # throw away the point at which to check the declaration
    assert caught.value.report.falsification is not None
    assert isinstance(caught.value, AssertionError)


def test_an_unconditioned_firing_still_says_stelling_is_unsound():
    """The split must not soften the message it was carved out of.

    A verdict carrying no unverified caller declaration is stelling's own
    claim, and a firing on it is a soundness event in this tool.
    """
    with pytest.raises(VerifiedFalsified) as caught:
        probe(traced(lying_pow), statuses=["discharged"])
    text = str(caught.value)
    assert "stelling is UNSOUND at this query" in text
    assert "RESTS ON A DECLARATION" not in text


# ==========================================================================
# points_admissible CARRIES ITS EVIDENCE
# ==========================================================================


def test_the_admissible_count_is_re_read_over_Q_on_a_CLEAN_run():
    """The take-back used to run only where a violation was already found.

    ``_confirm``'s exact re-reading of the assumes is consulted at a point
    where the obligation evaluated FALSE — so on a clean run, the common
    case, it never ran and ``points_admissible`` stood as a count of
    points *"admitted by every assume"* with no exact evidence behind it.
    Measured before this batch: 55 admissible, **47 of them not in the
    assumed region over ℚ**, under a stamp line ending *"NO VIOLATION WAS
    FOUND"*.
    """
    def clean():
        y = any_array((), "float64", (0.0, 2.0))
        # TRUE in floats, FALSE over ℚ for every y > 0
        a = assume(y * 0.1 * 10.0 <= y)
        # trivially true, so no violation anywhere and no take-back
        return a, assert_(y >= 0.0)

    import stelling.falsify as F

    found, report = attack(clean)
    assert found is None and report.declined is None, report
    counts = dict(report.skips)
    assert counts.get("assume-unsatisfied-over-the-rationals", 0) > 0, counts
    assert report.points_declined == 0, report.points_declined
    assert report.violations_seen == 0, report.violations_seen

    # every counted point is one an exact reading placed inside the region
    assert report.points_admissible_unconfirmed == 0, report
    census = F._read(traced(clean))
    checked = 0
    for point in _points_of(clean, report):
        assumes, _ = F._replay(census, point)
        assert all(assumes), (
            f"a point counted as admissible is outside the assumed region "
            f"over ℚ: {point}"
        )
        checked += 1
    assert checked == report.points_admissible, (checked, report)

    line = report.stamp_line()
    assert f"{report.points_admissible} inside the declared set" in line
    assert "re-read over ℚ and confirmed" in line


def _points_of(harness, report):
    """Re-run the probe recording every point it counted as admissible.

    Re-run rather than instrumented in place: the probe is deterministic
    for a fixed seed, and a report that carried its points would be a
    report that grew a field for a test's convenience.
    """
    import stelling.falsify as F

    kept = []
    census = F._read(traced(harness))
    original = F._execute

    def spy(c, point):
        run = original(c, point)
        if run.raised is None and len(run.assumes) == c.assumes_in_program:
            if all(bool(np.all(a)) for a in run.assumes):
                exact = None
                try:
                    assumes, _ = F._replay(c, [np.asarray(a) for a in point])
                    exact = (len(assumes) == c.assumes_in_program
                             and all(assumes))
                except Exception:  # noqa: BLE001
                    exact = None
                if exact is not False:
                    kept.append([np.asarray(a) for a in point])
        return run

    F._execute = spy
    try:
        try:
            F.probe(traced(harness), statuses=["discharged"] * 1)
        except VerifiedFalsified:
            pass
    finally:
        F._execute = original
    return kept


def test_the_stamp_line_says_which_reading_the_assume_half_rests_on():
    """Four cases, four different facts, and none of them is silence."""
    from stelling.falsify import ProbeReport

    none = ProbeReport(points_admissible=5, assumes_in_program=0)
    assert "states no assume" in none.stamp_line()

    ieee = ProbeReport(points_admissible=5, assumes_in_program=1,
                       semantics="ieee")
    assert "AS THE PROGRAM EXECUTED THEM IN FLOATS" in ieee.stamp_line()

    exact = ProbeReport(points_admissible=5, assumes_in_program=1)
    assert "every one of them re-read over ℚ" in exact.stamp_line()

    mixed = ProbeReport(points_admissible=5, assumes_in_program=1,
                        points_admissible_unconfirmed=2)
    line = mixed.stamp_line()
    assert "3 of them re-read over ℚ and confirmed" in line
    assert "2 on the executed float reading" in line
    assert "not evidence that those 2 are in the assumed region" in line


# ==========================================================================
# the two smaller ones
# ==========================================================================


def test_the_bool_window_intersects_with_the_declaration():
    """The one ``_window`` branch that ignored the numbers it was handed.

    ``any_array((), "bool", (0.0, 0.0))`` declares FALSE and nothing else.
    The bool branch returned ``(0, 1)`` flat, the sampler built ``True``,
    and only ``_admissible`` stopped it — at four wasted
    ``point-outside-declaration`` skips per run, which are also the only
    rejections that guard had ever made on a live corpus.
    """
    false_only = Declaration(position=0, shape=(), dtype="bool",
                             lo=0.0, hi=0.0)
    assert _window(false_only) == ((0, 0), None)
    assert _window(Declaration(position=0, shape=(), dtype="bool",
                               lo=1.0, hi=1.0)) == ((1, 1), None)
    assert _window(Declaration(position=0, shape=(), dtype="bool",
                               lo=0.0, hi=1.0)) == ((0, 1), None)
    # an infinite endpoint never leaves a two-element set unbounded
    assert _window(Declaration(position=0, shape=(), dtype="bool",
                               lo=float("-inf"),
                               hi=float("inf"))) == ((0, 1), None)
    # and a box with no bool in it declines by name rather than sampling
    assert _window(Declaration(position=0, shape=(), dtype="bool",
                               lo=2.0, hi=3.0)) == (None, "empty-integer-box")

    def only_false():
        b = any_array((), "bool", (0.0, 0.0))
        return assert_(jnp.logical_or(b, jnp.logical_not(b)))

    _, report = attack(only_false)
    assert dict(report.skips).get("point-outside-declaration", 0) == 0, (
        f"the sampler still builds points the declaration excludes: "
        f"{report.skips}"
    )
    assert report.points_built == report.points_executed, report


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
def test_the_sampled_point_must_survive_conversion_to_jax():
    """An invariant that holds because of a decline in ANOTHER module.

    ``run_one`` hands ``jnp.asarray`` the numpy point and ``_confirm`` and
    ``_replay`` the un-narrowed one. Under ``jax_enable_x64=0`` that
    conversion NARROWS float64 to float32, and the two halves of the fire
    condition would then be about different programs. Nothing reaches it
    today — ``propagate`` refuses the value-changing convert a float64
    declaration produces with x64 off, so no VERIFIED exists — but that is
    a fact about another module, and an invariant that rests on one is an
    invariant one line elsewhere can remove.
    """
    from stelling.falsify import ProbeInvariantViolated

    def h():
        x = any_array((), "float64", (0.0, 9.0))
        return assert_(x * x <= 40.0)

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        closed = traced(h)
        assert str(np.asarray(jnp.asarray(np.asarray(1.5, "float64")).dtype)) \
            == "float32", "jax no longer narrows here; the test is inert"
        with pytest.raises(ProbeInvariantViolated, match="did not survive"):
            probe(closed, statuses=["discharged"])
    finally:
        jax.config.update("jax_enable_x64", old)

    assert not issubclass(ProbeInvariantViolated, VerifiedFalsified), (
        "a broken invariant must not be catchable as a firing"
    )
    assert issubclass(ProbeInvariantViolated, AssertionError)


def test_the_assume_confirmation_stops_at_the_last_assume():
    """An abstention AFTER the last assume must not cost a reading already made.

    The gate needs the assumes and nothing else. A whole-program replay
    would abstain on ``exp`` and report every point of this program
    unconfirmed — while the assume itself has a perfectly exact reading
    over ℚ, and it is FALSE, so those points are outside the assumed
    region and must not be attacked at all.
    """
    import stelling.falsify as F

    def exp_after_assume():
        x = any_array((), "float64", (0.5, 2.0))
        a = assume(x * 0.1 * 10.0 <= x)      # exactly readable; false over ℚ
        return a, assert_(jnp.exp(x) >= 0.0)  # no exact rational reading

    census = F._read(traced(exp_after_assume))
    point = [np.asarray(1.0)]
    assert F._replay(census, point, assumes_only=True)[0] == [False]
    with pytest.raises(_Unreplayable, match="no exact rational reading"):
        F._replay(census, point)

    _, report = attack(exp_after_assume)
    assert dict(report.skips).get(
        "assume-unsatisfied-over-the-rationals", 0
    ) > 0, report.skips
    assert report.points_admissible_unconfirmed == 0, report
    assert report.points_admissible == 0, report
