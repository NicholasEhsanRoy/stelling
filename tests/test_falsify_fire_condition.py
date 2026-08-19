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
    assert dict(report.adjudications).get(
        "exact-replay-outside-the-assumed-region", 0
    ) == counts["assume-unsatisfied-over-the-rationals"], report.adjudications
