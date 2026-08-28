# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The boundary dial against the REAL jax decorators, and through `check`.

`tests/test_boundary_dial.py` is the hand-built-IR half and carries the
argument, the baseline and the cond-out prohibition; it runs on every lane.
This half needs jax, and exists for the two things hand-built IR cannot
establish:

* that the four wrappers `stelling.coverage.DEFAULT_TRANSPARENT` names are
  what `jax.jit`, `jax.checkpoint`, `jax.custom_jvp` and `jax.custom_vjp`
  ACTUALLY trace to — read off the registry through
  `tests/test_assume_scope_identity.py`'s `_wrappers`, which is the one
  place in this suite that maps a primitive name to a real decorator;
* that the dial reaches the public front door `stelling.preconditions.check`
  and its widen re-check, rather than stopping at `propagate`.

**WHAT IT DOES NOT REACH.** It does not exercise the solver, the affine
refinement or the falsification probe under the dial: those layers do not
read the strict-sign certificate at all, so there is nothing there for the
dial to change — which is a statement about today's code and would stop
being true the day one of them learned to. It does not compare against a
base tree; the byte-for-byte baseline is the hand-built one next door,
because a traced query's content hash depends on the host's jax and x64
state and a baseline built on those would report a different truth to
different people.

**AND IT IS THE ONLY HALF THAT CAN DECIDE WHAT THE PROGRAM DOES**, which is
why the disclosure's own measurement lives here rather than next door.
Three tests, one per clause of the stamped sentence:
`::test_the_carry_reaches_a_VERIFIED_the_compiled_program_contradicts`
executes the `reduce_sum` query and finds a point of the declared box at
which the verdict is false;
`::test_the_reduction_seed_is_the_mechanism_and_it_is_measured` measures the
`+0.0` seed that makes it false; and
`::test_dot_general_reaches_it_too_and_the_CONTRACTION_LENGTH_decides`
measures the OTHER reduction the sentence names, which is not assumed to
behave like the first. All three are skipped on the zero-dep lane, so on
that lane NOTHING in this repository checks that the stamped reach
disclosure is TRUE — only that it is stamped.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import math  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

from stelling.coverage import DEFAULT_TRANSPARENT, sub_jaxprs  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import interval_env, propagate  # noqa: E402

from test_assume_scope_identity import _wrappers  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """This module declares float64 inputs, so it asks for x64 ITSELF and
    restores it — the house pattern; inheriting it from another module's
    module-scope call is the defect `tests/test_assume_scope_identity.py`'s
    own fixture docstring records."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _wrapped_harness(wrapper, *, assume_inside):
    """`assume(x>0)`; `assert_(1 / W(lambda v: sum(v*v))(x) > 0)`.

    `assume_inside` moves the precondition into the wrapper body, which is
    the other direction the certificate can cross. Both placements are
    named as UNKNOWN in the shipped 0.2.x disclosure.
    """
    def h():
        x = any_array((4,), "float64", (0.0, 2.0))
        if not assume_inside:
            assume(x > 0.0)

        def body(v):
            if assume_inside:
                assume(v > 0.0)
            return jnp.sum(v * v)

        return assert_(1.0 / wrapper(body)(x) > 0.0)

    return h


def _unwrapped_harness():
    """THE ORACLE: the same arithmetic with no wrapper at all."""
    def h():
        x = any_array((4,), "float64", (0.0, 2.0))
        assume(x > 0.0)
        return assert_(1.0 / jnp.sum(x * x) > 0.0)

    return h


def test_the_wrapper_registry_is_the_one_this_suite_already_has():
    """`_wrappers` is imported rather than rebuilt. Pinned so a second
    copy cannot appear here and drift from the registry."""
    assert set(_wrappers()) == set(DEFAULT_TRANSPARENT), sorted(_wrappers())


@pytest.mark.parametrize("wrapper_name", sorted(_wrappers()))
@pytest.mark.parametrize("assume_inside", [False, True])
def test_the_real_decorator_carries_the_certificate_both_ways(
    wrapper_name, assume_inside
):
    """Every member of `DEFAULT_TRANSPARENT`, as the decorator a user
    actually writes, in both `assume` placements.

    The assertion is AGREEMENT WITH THE UNWRAPPED CONTROL, not a literal
    `discharged`: the wrapped and unwrapped programs compute the same real
    number, so a boundary that carries the certificate must make the two
    verdicts equal. That equality is the strongest acceptance criterion
    available here — a dial that discharged the wrapped query for some
    unrelated reason would not satisfy it.

    The `"opaque"` half of each case is the shipped 0.2.x measurement,
    re-driven through the real decorator rather than through hand-built IR.
    """
    wrapper = _wrappers()[wrapper_name]
    control = propagate(
        trace(_unwrapped_harness()), semantics="real", boundary="transparent"
    )
    assert control.obligations[0].status == "discharged", (
        f"the unwrapped control is not green: {control.obligations[0].detail}"
    )
    cj = trace(_wrapped_harness(wrapper, assume_inside=assume_inside))
    opaque = propagate(cj, semantics="real")
    assert opaque.obligations[0].status != "discharged", (
        f"{wrapper_name}: the DEFAULT discharged a wrapped query — the "
        f"default has moved"
    )
    assert opaque.boundary_crossings == 0
    transparent = propagate(cj, semantics="real", boundary="transparent")
    assert (
        transparent.obligations[0].status == control.obligations[0].status
    ), (
        f"{wrapper_name} (assume_inside={assume_inside}): wrapped is "
        f"{transparent.obligations[0].status!r}, unwrapped is "
        f"{control.obligations[0].status!r} — "
        f"{transparent.obligations[0].detail}"
    )
    assert transparent.boundary_crossings > 0


def test_a_real_jit_inside_a_real_jit():
    """Nesting, through jax's own tracer rather than through hand-built
    bodies: two `call` steps on the scope path, extended and not reset."""
    def h():
        x = any_array((4,), "float64", (0.0, 2.0))
        assume(x > 0.0)
        inner = jax.jit(lambda v: jnp.sum(v * v))
        return assert_(1.0 / jax.jit(lambda v: inner(v))(x) > 0.0)

    cj = trace(h)
    assert propagate(cj, semantics="real").obligations[0].status != "discharged"
    p = propagate(cj, semantics="real", boundary="transparent")
    assert p.obligations[0].status == "discharged", p.obligations[0].detail


def test_a_real_cond_carries_IN_but_a_real_branch_assume_stays_INSIDE():
    """Both halves of the `cond` rule, on jax's own `lax.cond`.

    IN: the outer `assume` reaches a division inside a branch, so the
    branch obligation discharges under the dial and not under the default.

    NOT OUT: the branch's own `assume` certifies nothing outside it — the
    division on the cond's RESULT stays UNKNOWN in both positions. The
    hand-built version of that prohibition, with its unconditional-wrapper
    control and the broken build it was driven against, is
    `tests/test_boundary_dial.py::test_a_branch_body_assume_NEVER_certifies_anything_outside_its_branch`.
    """
    def carries_in():
        x = any_array((), "float64", (0.0, 2.0))
        s = any_array((), "float64", (0.0, 1.0))
        assume(x > 0.0)
        return assert_(
            lax.cond(s > 0.5, lambda v: 1.0 / v, lambda v: 1.0 / v, x) > -1e9
        )

    cj = trace(carries_in)
    assert propagate(cj, semantics="real").obligations[0].status != "discharged"
    assert (
        propagate(cj, semantics="real", boundary="transparent")
        .obligations[0].status == "discharged"
    )

    def branch_assume_escapes():
        x = any_array((), "float64", (0.0, 1.0))
        s = any_array((), "float64", (0.0, 1.0))

        def certifying(v):
            assume(v > 0.0)
            return v

        # the OTHER branch must not be `lambda v: v`. MEASURED on jax
        # 0.11.0: `lax.cond(p, certifying, lambda v: v, x)` traces to a
        # `cond` equation with NO OUTVARS AT ALL — both branches are the
        # identity, jax forwards the operand, and the division downstream
        # is then on the DECLARED input rather than on a cond output. The
        # test would still have passed and would have been testing
        # nothing. `v * 2.0` keeps the cond's output a real value while
        # leaving the joined box one-sided at zero, which is the shape the
        # `div` boundary gate is about.
        d = lax.cond(s > 0.5, certifying, lambda v: v * 2.0, x)
        return assert_(1.0 / d > 0.0)

    cj2 = trace(branch_assume_escapes)
    assert any(
        e.primitive == "cond" and e.outvars for e in cj2.jaxpr.eqns
    ), "jax forwarded the operand and left no cond output to divide by"
    for boundary in ("opaque", "transparent"):
        r = propagate(cj2, semantics="real", boundary=boundary)
        assert r.obligations[0].status != "discharged", (
            f"boundary={boundary!r}: a branch-local assume discharged a "
            f"division OUTSIDE its branch — {r.obligations[0].detail}"
        )


# ---------------------------------------------------------------------------
# what the carry reaches
# ---------------------------------------------------------------------------

_UNDERFLOWING = 1e-200


def _reaching_chain(v):
    """``1 / Σ(v · 1e-200 · 1e-200)``.

    The two multiplications underflow, so the propagated box of the product
    is ``[-5e-324, -0.0]`` — zero at ONE boundary rather than merely small —
    which is the shape `stelling.interval.boundary_div` is consulted for and
    the shape a strict-sign certificate licenses it to tighten.
    """
    return 1.0 / jnp.sum(v * _UNDERFLOWING * _UNDERFLOWING)


def _plain_chain(v):
    """``1 / Σv``: the SAME division, with no underflow anywhere."""
    return 1.0 / jnp.sum(v)


def _reaching_harness(chain, *, wrapped, assumed=True):
    def h():
        x = any_array((2,), "float64", (-1.0, -0.25))
        if assumed:
            assume(x < 0.0)
        f = jax.jit(chain) if wrapped else chain
        return assert_(f(x) < 0.0)

    return h


_DECLARED_POINTS = ((-1.0, -1.0), (-0.25, -0.25), (-1.0, -0.25),
                    (-0.6, -0.4), (-0.9, -0.3))


def _the_program_says(chain):
    """Does the COMPILED program satisfy ``chain(x) < 0`` at every point of
    the declared box we try — eager and under `jax.jit`?

    Returns the first point where it does not, or ``None``. Deciding this
    by running jax rather than by modelling it is the point: the question
    is what the executable does, and a model of that is one indirection
    behind the answer.
    """
    for pt in _DECLARED_POINTS:
        x = jnp.array(pt, dtype=jnp.float64)
        for f in (chain, jax.jit(chain)):
            out = f(x)
            if not bool(out < 0.0):
                return pt, float(out)
    return None


def test_the_carry_reaches_a_VERIFIED_the_compiled_program_contradicts():
    """**THE DISCLOSURE'S OWN MEASUREMENT, TAKEN BY RUNNING THE PROGRAM.**

    `stelling.propagate.BOUNDARY_TRANSPARENT_REACH_DISCLOSURE` is stamped on
    every live-carry `transparent` run and says that a certificate reaching
    `reduce_sum` or `dot_general` can license a VERIFIED the compiled
    program contradicts. This is the run of that claim, three rows, at
    `vacuity_mode="all"`, `semantics="real"`, `jax_enable_x64=True`:

        query                                    opaque      transparent
        1/Σ(x·1e-200·1e-200) < 0, no wrapper     VERIFIED    VERIFIED
        ...the same chain inside a `jax.jit`     UNKNOWN     VERIFIED
        1/Σx < 0 inside a `jax.jit`              VERIFIED    VERIFIED

    and the three readings, each asserted below rather than described:

    1. ROW 1 IS VERIFIED AT THE DEFAULT, with zero crossings. The defect is
       NOT an artifact of the dial and is not confined to `transparent`;
       `opaque` reaches it wherever the chain sits in the assume's own
       scope. This is why the stamped sentence says so in its own words and
       why the absence of that sentence at `"opaque"` is not a claim that
       `"opaque"` is safe.
    2. ROW 2 MOVES UNKNOWN → VERIFIED. What the opt-in adds is REACH: a
       chain the default DECLINED, because the reduction sits behind a
       wrapper, now gets a verdict — and the program contradicts it.
    3. ROW 3 IS NOT A DISCRIMINATOR, and the reason is CHECKED and not
       assumed: the same query with the `assume` DELETED — so that no
       strict-sign certificate exists anywhere in the walk — is still
       VERIFIED at `opaque`. `Σx` over `[-1, -0.25]²` has interval
       `[-2, -0.5]`, which excludes zero with no certificate involved. Row
       3 is in the table so that nobody later reads it as evidence of a
       carry.

    **IF THIS TEST REDDENS BECAUSE THE REPAIR LANDED, THE DISCLOSURE MUST
    BE RE-READ — AND NOT DELETED.** This paragraph used to say the day the
    reduction taint lands `BOUNDARY_TRANSPARENT_REACH_DISCLOSURE` *"stops
    being true and must go out with the same commit"*, and that commitment
    was a net regression the tree had written down (0.3.0 P1 second
    re-audit, F1b). The taint keys on a CERTIFICATE and a REDUCTION; the
    wider route the same sentence now names —
    `::test_the_WIDER_route_needs_neither_certificate_nor_reduction` — has
    neither, so the taint closes the narrow sub-route and leaves the wider
    one open. Deleting the line then removes the only sentence in the stamp
    that gestures at what is still open. The coupling this test buys is
    that the sentence cannot go stale unnoticed, which is a reason to
    RE-READ it, not to retire it.
    """
    from stelling.propagate import BOUNDARY_TRANSPARENT_REACH_DISCLOSURE

    rows = {
        "1 no wrapper": (_reaching_chain, False),
        "2 in a jit": (_reaching_chain, True),
        "3 plain sum in a jit": (_plain_chain, True),
    }
    table = {}
    for name, (chain, wrapped) in rows.items():
        for boundary in ("opaque", "transparent"):
            table[name, boundary] = check(
                _reaching_harness(chain, wrapped=wrapped),
                vacuity_mode="all", semantics="real", boundary=boundary,
            ).status
    assert table == {
        ("1 no wrapper", "opaque"): "VERIFIED",
        ("1 no wrapper", "transparent"): "VERIFIED",
        ("2 in a jit", "opaque"): "UNKNOWN",
        ("2 in a jit", "transparent"): "VERIFIED",
        ("3 plain sum in a jit", "opaque"): "VERIFIED",
        ("3 plain sum in a jit", "transparent"): "VERIFIED",
    }, table

    # READING 1: row 1 is the DEFAULT's own reach, and no boundary carried.
    p1 = propagate(
        trace(_reaching_harness(_reaching_chain, wrapped=False)),
        semantics="real",
    )
    assert p1.obligations[0].status == "discharged"
    assert p1.boundary_crossings == 0

    # READING 2: the carry is what moved row 2, and it is COUNTED.
    p2 = propagate(
        trace(_reaching_harness(_reaching_chain, wrapped=True)),
        semantics="real", boundary="transparent",
    )
    assert p2.obligations[0].status == "discharged"
    assert p2.boundary_crossings > 0, (
        "row 2 discharged under the dial with nothing crossing, so this "
        "row measures something other than the carry"
    )

    # READING 3: row 3's mechanism, CHECKED. With the assume deleted there
    # is no certificate anywhere and the verdict does not move, so nothing
    # a carry could have supplied is what decided it.
    bare = check(
        _reaching_harness(_plain_chain, wrapped=True, assumed=False),
        vacuity_mode="all", semantics="real", boundary="opaque",
    )
    assert bare.status == "VERIFIED", (
        f"row 3 stopped being VERIFIED once the certificate was removed, so "
        f"it IS a discriminator after all and the table's third row means "
        f"something this docstring denies: {bare.render()}"
    )

    # AND WHAT THE PROGRAM DOES. Rows 1 and 2 are the same chain, so one
    # execution answers for both.
    contradiction = _the_program_says(_reaching_chain)
    assert contradiction is not None, (
        "the compiled program satisfies the obligation at every point "
        "tried, so rows 1 and 2 are not false VERIFIEDs and the stamped "
        "reach disclosure is claiming something this tree cannot show"
    )
    pt, out = contradiction
    assert out > 0.0, (pt, out)

    # ...and the ANTI-VACUITY half: the same executor on row 3's chain
    # AGREES with the verdict, so the check above is discriminating and not
    # a probe that fails on everything.
    assert _the_program_says(_plain_chain) is None, _the_program_says(_plain_chain)

    # the verdict a caller actually reads carries the disclosure
    moved = check(
        _reaching_harness(_reaching_chain, wrapped=True),
        vacuity_mode="all", semantics="real", boundary="transparent",
    )
    assert BOUNDARY_TRANSPARENT_REACH_DISCLOSURE in moved.stamp.assumptions, (
        moved.render()
    )


def test_the_reduction_seed_is_the_mechanism_and_it_is_measured():
    """WHY the reach exists, decided by running jax rather than by citing
    a lowering.

    `reduce_sum` accumulates from a `+0.0` seed, so `(+0) + (-0) = +0`: a
    value the strict-sign certificate calls NEGATIVE, all of whose elements
    are `-0.0` at run time, reduces to `+0.0`. `+0.0` is outside every arm
    of the box `stelling.interval.boundary_div` returns from a negative
    certificate, and `1.0 / (+0.0)` is `+inf`.

    The n = 1 case is the control: with no addition performed there is no
    seed to see, and the reduction is `-0.0`. Everything from n = 2 up is
    `+0.0`. If a future jax lowered the reduction so that the sign survived,
    THIS is the test that would redden — and the disclosure would need
    re-reading, because the mechanism it names would have changed.
    """
    def sign_bit(x):
        return math.copysign(1.0, float(x)) < 0.0

    minus_zero_sum = {
        n: jnp.sum(jnp.full((n,), -0.0, dtype=jnp.float64))
        for n in (1, 2, 3, 8, 64)
    }
    assert sign_bit(minus_zero_sum[1]), (
        f"n=1 lost the sign with no addition performed: "
        f"{minus_zero_sum[1]!r}"
    )
    for n in (2, 3, 8, 64):
        assert float(minus_zero_sum[n]) == 0.0, minus_zero_sum[n]
        assert not sign_bit(minus_zero_sum[n]), (
            f"a reduction of {n} copies of -0.0 kept the sign bit; the +0.0 "
            f"seed the reach disclosure names is not what this target does"
        )

    # NUMPY DISAGREES ON THE n = 1 ROW AND AGREES ON EVERY OTHER, and both
    # halves are measured because the shipped comment used to say the two
    # were alike across the whole table (0.3.0 P1 re-audit, F6). n = 1 is
    # the control above, and it is a JAX fact; from n = 2 the two agree, and
    # that is the row the defect rides.
    assert math.copysign(1.0, float(np.sum(np.full((1,), -0.0)))) > 0.0, (
        "numpy kept the sign at n=1; the n=1 control above is documented as "
        "a jax-only fact and that documentation has gone stale"
    )
    for n in (2, 3, 8, 64):
        assert math.copysign(1.0, float(np.sum(np.full((n,), -0.0)))) > 0.0, n

    # and the whole chain, end to end, on the query the table uses
    x = jnp.array((-1.0, -0.25), dtype=jnp.float64)
    product = x * _UNDERFLOWING * _UNDERFLOWING
    assert all(sign_bit(e) and float(e) == 0.0 for e in product), product
    assert not sign_bit(jnp.sum(product)), jnp.sum(product)
    assert float(_reaching_chain(x)) == math.inf, _reaching_chain(x)


def _power_harness(k, lo, hi, *, assumed=True, wrapped=False):
    """``1 / Σ(x**k) < 0`` over ``x`` declared ``[lo, hi]``.

    No exotic constant anywhere: the only thing that grows is the exponent,
    and what underflows is the ANALYSIS'S OWN box.
    """
    def h():
        x = any_array((2,), "float64", (lo, hi))
        if assumed:
            assume(x < 0.0)
        g = lambda v: 1.0 / jnp.sum(v ** k)  # noqa: E731
        f = jax.jit(g) if wrapped else g
        return assert_(f(x) < 0.0)

    return h


def _power_runs(k, at):
    x = jnp.full((2,), at, dtype=jnp.float64)
    return float(1.0 / jnp.sum(x ** k))


def test_an_ORDINARY_magnitude_chain_reaches_it_at_the_DEFAULT():
    """**THE REACHING CONDITION IS NOT ABOUT MAGNITUDES, AND THE FIRST PASS
    OF THIS ITEM CONCLUDED THAT IT WAS.**

    That pass searched a cube-and-scale grammar at ordinary magnitudes,
    found 0 contradicted in 10,164 queries, and read the null as a bound on
    the reach. It was a bound on the GRAMMAR. The condition is only that
    the analysis's own binary64 box underflows onto zero at one boundary,
    and any long enough chain gets there from ordinary declared values —
    `1e-200` is a way to reach it in two multiplications, not a
    prerequisite (0.3.0 P1 re-audit, F1).

    `x` declared `[-0.4, -0.2]` — nothing unusual about either endpoint —
    with `assume(x < 0)` and `1.0 / jnp.sum(x ** 1001) < 0.0`:

        default             VERIFIED, 0 boundary crossings, clean stamp
        no-assume control   UNKNOWN  (the certificate is load-bearing)
        the same in a jit   opaque UNKNOWN -> transparent VERIFIED
        the program         +inf, eager and jitted

    Bisected at `1b34d25` on 2026-08-28 over `x` in `(-m, -m/2)`, largest
    reaching `m`: k=3 → 3.4e-108, k=11 → 8.1e-30, k=51 → 9.2e-07,
    k=201 → 0.049, k=501 → 0.45, k=1001 → 0.95. The frontier is a function
    of the CHAIN, not of the inputs.

    THE ANTI-VACUITY HALF is the small exponent: at k = 3 the same envelope
    is VERIFIED *without* any certificate and the program AGREES, so this
    test is about the chain length and not about the envelope being
    somehow special.
    """
    v = check(_power_harness(1001, -0.4, -0.2), vacuity_mode="all",
              semantics="real")
    assert v.status == "VERIFIED", v.render()
    assert not any("boundary=" in a for a in v.stamp.assumptions), (
        "the DEFAULT grew a boundary line; this row is supposed to show the "
        "defect arriving with NOTHING in the stamp about it"
    )
    bare = check(_power_harness(1001, -0.4, -0.2, assumed=False),
                 vacuity_mode="all", semantics="real")
    assert bare.status == "UNKNOWN", (
        f"the ordinary-magnitude row discharged with no `assume` anywhere, "
        f"so no strict-sign certificate was needed and it is not about the "
        f"certificate at all: {bare.render()}"
    )
    assert _power_runs(1001, -0.2) > 0.0, _power_runs(1001, -0.2)

    cj = trace(_power_harness(1001, -0.4, -0.2, wrapped=True))
    assert propagate(cj, semantics="real").obligations[0].status != "discharged"
    moved = propagate(cj, semantics="real", boundary="transparent")
    assert moved.obligations[0].status == "discharged", (
        moved.obligations[0].detail
    )
    assert moved.boundary_crossings > 0

    # the anti-vacuity half: a SHORT chain on the same envelope
    small = check(_power_harness(3, -0.4, -0.2), vacuity_mode="all",
                  semantics="real")
    small_bare = check(_power_harness(3, -0.4, -0.2, assumed=False),
                       vacuity_mode="all", semantics="real")
    assert small.status == "VERIFIED" and small_bare.status == "VERIFIED", (
        f"k=3 on this envelope needs the certificate, so the envelope is "
        f"doing the work and not the chain length: "
        f"{small.status}/{small_bare.status}"
    )
    assert _power_runs(3, -0.2) < 0.0, (
        f"the program contradicts the SHORT chain too, so this control does "
        f"not separate chain length from envelope: {_power_runs(3, -0.2)}"
    )


def _wider_route_harness():
    """The whole defect class in four equations, and the stamped sentence's
    own counterexample.

    `x * 2**-512 * 2**-511` over `x` declared `[-1, -0.5]`. No `assume`, no
    strict-sign certificate, no reduction, no division, no `boundary_div` —
    just two multiplications whose exact result lands in the target's
    SUBNORMAL BAND.
    """
    def h():
        x = any_array((), "float64", (-1.0, -0.5))
        return assert_(x * 2.0**-512 * 2.0**-511 < 0.0)

    return h


def test_the_WIDER_route_needs_neither_certificate_nor_reduction():
    """**THE STAMPED SENTENCE ONCE READ "THE CONDITION IS ONLY THAT THE
    ANALYSIS'S OWN BOX UNDERFLOWS ONTO ZERO", AND THE WORD *ONLY* MADE IT A
    BOUND IT IS NOT** (0.3.0 P1 second re-audit, F1).

    That is the condition for the route the dial widens — `boundary_div` is
    consulted only on a divisor box that reaches zero. It is NOT the
    condition for a real-mode VERIFIED the compiled program contradicts,
    and a reader meeting `only` in a stamp concludes that a verdict whose
    boxes never touch zero is safe.

    The witness below has ZERO EXCLUDED from every box, no certificate, no
    reduction and no division, and it is VERIFIED at the default while the
    program returns `-0.0` at every declared point. The wider condition is
    the box entering the target's subnormal band: this target flushes a
    subnormal RESULT of an ordinary `mul`, while the real-mode transfer
    computes it exactly — `python` and `numpy` both return
    `-1.1125369292536007e-308` for the same arithmetic.

    Three controls, so this is a reading and not an anecdote: the query has
    none of the machinery the reach sentence names; `semantics="ieee"`
    declines it; and a sibling one binade up, whose result stays NORMAL, is
    VERIFIED *and* agrees with the program.
    """
    cj = trace(_wider_route_harness())
    prims = _all_primitives(cj.jaxpr)
    assert prims == {"stelling_any", "mul", "lt", "stelling_assert"}, sorted(
        prims
    )

    p = propagate(cj, semantics="real")
    assert p.obligations[0].status == "discharged", p.obligations[0].detail
    assert p.boundary_crossings == 0
    assert p.relational_assumes == ()
    assert p.coverage.constrained == 0

    boxes = interval_env(cj, assume_mode="constrain")
    producers = {o.id: e.primitive for e in cj.jaxpr.eqns for o in e.outvars}
    divisor_like = [
        (b.los[0], b.his[0])
        for vid, b in boxes.items()
        if producers.get(vid) == "mul"
    ]
    assert all(lo < 0.0 and hi < 0.0 for lo, hi in divisor_like), divisor_like
    assert any(
        abs(hi) < 2.2250738585072014e-308 for _, hi in divisor_like
    ), (
        f"no box reaches the binary64 subnormal band, so this witness is not "
        f"about the band at all: {divisor_like}"
    )

    v = check(_wider_route_harness(), vacuity_mode="all", semantics="real")
    assert v.status == "VERIFIED", v.render()
    assert not any("boundary=" in a for a in v.stamp.assumptions), (
        "this row is supposed to arrive with NOTHING in the stamp about a "
        "boundary; the defect it shows is the default's"
    )

    for at in (-1.0, -0.75, -0.5):
        x = jnp.array(at, dtype=jnp.float64)
        for f in (lambda z: z * 2.0**-512 * 2.0**-511,
                  jax.jit(lambda z: z * 2.0**-512 * 2.0**-511)):
            assert not bool(f(x) < 0.0), (
                f"the program satisfies the obligation at x={at}, so the "
                f"target no longer flushes this result and the wider route "
                f"named in the stamp has closed: {float(f(x))!r}"
            )
    # ...and the same arithmetic OFF this target keeps the value
    assert float(np.float64(-1.0) * 2.0**-512 * 2.0**-511) < 0.0, (
        "numpy flushes it too, so this is not a target-flush finding"
    )

    # CONTROL 1: ieee declines the whole thing
    assert check(_wider_route_harness(), vacuity_mode="all",
                 semantics="ieee").status == "UNKNOWN"

    # CONTROL 2: one binade up, the result is NORMAL and everything agrees
    def normal():
        x = any_array((), "float64", (-1.0, -0.5))
        return assert_(x * 2.0**-512 * 2.0**-509 < 0.0)

    assert check(normal, vacuity_mode="all", semantics="real").status == (
        "VERIFIED"
    )
    assert float(jnp.float64(-0.5) * 2.0**-512 * 2.0**-509) < 0.0, (
        "the control's result is flushed too, so it does not separate the "
        "subnormal band from arithmetic in general"
    )


_REFUTING_CONST = np.array([-0.3, -0.35])


def _refuting_harness(*, keep_sign=True):
    """A false REFUTED out of the same licence.

    `q = 1 / Σ(W**1001)` boxes to `[-inf, -1.79e308]` and RUNS to `+inf`, so
    the analysis decides `q > 0.0` FALSE, takes only the `v + 10` branch,
    and finds a definite violation there. The program takes the other one.
    """
    def h():
        x = any_array((), "float64", (0.0, 1.0))
        w = jnp.asarray(_REFUTING_CONST) ** 1001
        if not keep_sign:
            # `sub` is not a strict-sign rule, so the certificate is dropped
            # and the identical query goes UNKNOWN
            w = w - jnp.zeros((2,), dtype=jnp.float64)
        q = 1.0 / jnp.sum(w)
        return assert_(
            lax.cond(q > 0.0, lambda v: v - 10.0, lambda v: v + 10.0, x) < 0.0
        )

    return h


def test_the_same_licence_mints_a_false_REFUTED():
    """**THE STAMPED SENTENCE SAID "a VERIFIED", AND THE DEFECT IS NOT
    DIRECTIONAL** (0.3.0 P1 re-audit, F2).

    `boundary_div`'s half-infinite box can decide a COMPARISON that feeds a
    `cond` selector. The analysis then walks only the branch the program
    does not take, and a definite violation inside that forced branch is
    not withheld: the verdict is REFUTED and the obligation is TRUE at
    every point of the declared box.

    Driven at the DEFAULT — no dial involved, no `assume` involved — with a
    control that drops the certificate through a `sub` and returns UNKNOWN,
    so the REFUTED is attributable to the certificate and not to the shape.
    """
    v = check(_refuting_harness(), vacuity_mode="all", semantics="real")
    assert v.status == "REFUTED", v.render()

    q = float(1.0 / jnp.sum(jnp.asarray(_REFUTING_CONST) ** 1001))
    assert q > 0.0, f"the analysis and the program agree about q: {q}"
    for at in (0.0, 0.5, 1.0):
        x = jnp.array(at, dtype=jnp.float64)
        r = lax.cond(jnp.asarray(q) > 0.0,
                     lambda v: v - 10.0, lambda v: v + 10.0, x)
        assert float(r) < 0.0, (
            f"the obligation is FALSE at x={at} in the executed program, so "
            f"this REFUTED is not false and this test has nothing to say: "
            f"{float(r)}"
        )

    control = check(_refuting_harness(keep_sign=False), vacuity_mode="all",
                    semantics="real")
    assert control.status == "UNKNOWN", (
        f"the control REFUTED too, with the certificate dropped by a `sub` — "
        f"so the certificate is not what minted it: {control.render()}"
    )


def _all_primitives(jaxpr):
    """Every primitive in a jaxpr AND in every sub-jaxpr it carries, at any
    depth, via `stelling.coverage.sub_jaxprs`.

    A top-level `{e.primitive for e in jaxpr.eqns}` census answers a
    question about the OUTERMOST equation list, and "does this query
    contain a wrapper" is not that question — a wrapper nested inside a
    `cond` branch is invisible to it and is exactly what would spoil the
    separation the caller is measuring.

    **AND THE FIRST DESCENT WRITTEN HERE WAS `getattr(sub, "jaxpr", None)`,
    WHICH IS A FACT ABOUT ONE JAX SERIES** (0.3.0 P1 second re-audit, F3).
    It finds a `ClosedJaxpr` and misses a bare `ir.Jaxpr`, and
    `stelling.coverage.call_body`'s own table records that `remat2`'s body
    is a bare `Jaxpr` on jax 0.10.2 and a `ClosedJaxpr` on 0.11.0 — jax 0.11
    merged the two classes. Measured over 8 nested wrapper queries: 0
    disagreements on 0.11.0, and on **0.10.2** the hand-rolled descent
    missed `['mul', 'reduce_sum']` under `remat2 alone`, `remat2 in jit` and
    `remat2 in remat2`, and `['jit', 'mul', 'reduce_sum']` under `jit in
    remat2`. `sub_jaxprs` is the project's canonical accessor for exactly
    this reason and its docstring says so.

    **WHAT THAT BLINDNESS DID *NOT* DO IS OPEN A HOLE IN THE ONE ASSERTION
    THAT CONSUMES THIS TODAY**, and saying otherwise would be claiming a
    repair nobody needed. `::test_the_carry_reaches_it_through_a_cond_
    BRANCH_and_not_only_a_wrapper` asks whether ANY member of
    `DEFAULT_TRANSPARENT` appears, and the outermost wrapper is visible to
    either descent — a wrapper nested inside a `remat2` body implies a
    `remat2` at an outer level, which is itself a member. Driven: a
    `jax.checkpoint` smuggled into each cond branch reddens that test on
    both series under BOTH descents. The defect was in this helper's own
    contract — its first line promises *every* primitive at *any* depth —
    and it is pinned directly by
    `::test_the_primitive_census_descends_a_remat2_body`, which is the
    assertion that does observe the difference.
    """
    seen = set()

    def walk(jx):
        for e in jx.eqns:
            seen.add(e.primitive)
            for sub in sub_jaxprs(e):
                walk(sub)

    walk(jaxpr)
    return seen


def test_the_primitive_census_descends_a_remat2_body():
    """`_all_primitives` promises every primitive at any depth, and the
    `remat2` body is where that promise is series-dependent.

    `stelling.coverage.call_body`'s measured table: `remat2` carries a bare
    `ir.Jaxpr` on jax 0.10.2 and a `ClosedJaxpr` on 0.11.0, because 0.11
    merged the two classes. A descent keyed on `getattr(sub, "jaxpr")`
    therefore finds the body on one series and not the other, and this
    module runs on both.

    Asserted against CONCRETE primitives rather than against
    `sub_jaxprs`'s own answer: comparing the helper to the function it
    calls would be a restatement, not a check.
    """
    def h():
        x = any_array((3,), "float64", (0.0, 2.0))
        assume(x > 0.0)
        return assert_(1.0 / jax.checkpoint(lambda v: jnp.sum(v * v))(x) > 0.0)

    prims = _all_primitives(trace(h).jaxpr)
    assert "remat2" in prims, sorted(prims)
    assert {"mul", "reduce_sum"} <= prims, (
        f"the census stopped at the remat2 equation and never entered its "
        f"body, so it is reporting the outer equation list and not the "
        f"query: {sorted(prims)}"
    )


def _cond_harness(*, forced):
    """The reaching chain inside a `lax.cond` BRANCH and nowhere else."""
    def h():
        x = any_array((2,), "float64", (-1.0, -0.25))
        s = any_array((), "float64", (0.0, 1.0))
        assume(x < 0.0)
        pred = jnp.asarray(True) if forced else (s > 0.5)
        return assert_(lax.cond(pred, _reaching_chain, _reaching_chain, x)
                       < 0.0)

    return h


@pytest.mark.parametrize("forced", [False, True])
def test_the_carry_reaches_it_through_a_cond_BRANCH_and_not_only_a_wrapper(
    forced
):
    """**THE STAMPED SENTENCE SAID "behind a wrapper", AND "wrapper" IS A
    TERM OF ART HERE THAT EXCLUDES THIS ROUTE** (0.3.0 P1 re-audit, F3).

    `stelling.propagate.BOUNDARY_TRANSPARENT_POSITION` itself separates
    *"the unconditional wrappers (jit/remat2/custom_jvp_call/
    custom_vjp_call)"* from *"a cond branch"*, so a sentence saying
    "wrapper" names half the boundaries the dial opens. A `cond` branch is
    the other half and the carry reaches through it, for a live selector
    and for a forced one alike.

    The query is checked to contain a `cond` and NO member of
    `stelling.coverage.DEFAULT_TRANSPARENT`, so "not a wrapper" is measured
    off the traced IR rather than asserted about the source — and the
    census is RECURSIVE. **A top-level-only census passed a query with a
    `jax.jit` inside each cond branch**, which is exactly the query this
    assertion has to reject: the carry would then have a wrapper boundary
    available to it and the row would prove nothing about `cond`. Found by
    driving this test against that build while it was being written.
    """
    cj = trace(_cond_harness(forced=forced))
    prims = _all_primitives(cj.jaxpr)
    assert "cond" in prims, sorted(prims)
    assert not (prims & set(DEFAULT_TRANSPARENT)), (
        f"the traced query carries a wrapper as well as the cond, at some "
        f"depth, so it cannot separate the two routes: {sorted(prims)}"
    )
    assert propagate(cj, semantics="real").obligations[0].status != "discharged"
    moved = propagate(cj, semantics="real", boundary="transparent")
    assert moved.obligations[0].status == "discharged", (
        moved.obligations[0].detail
    )
    assert moved.boundary_crossings > 0
    assert _the_program_says(_reaching_chain) is not None


_CONSTVAR_W = np.array([-1e-120, -2e-120, -3e-120])


def _constvar_harness(*, wrapped=False):
    """The reaching chain with NO `assume` anywhere: the certificate comes
    from the CONSTVAR writer, off an array constant's own box."""
    def h():
        g = lambda v: 1.0 / jnp.sum(v ** 3)  # noqa: E731
        f = jax.jit(g) if wrapped else g
        return assert_(f(jnp.asarray(_CONSTVAR_W)) < 0.0)

    return h


def test_the_reach_needs_no_assume_the_CONSTVAR_writer_is_enough():
    """**THE STAMPED SENTENCE SAID "the assume's own scope", AND NO ASSUME
    IS REQUIRED** (0.3.0 P1 re-audit, F4).

    The strict-sign table has three sources and only one of them is a
    strict `assume`; an array CONSTANT is certified from its own box by the
    constvar writer in `_Propagator.run`. A query with no `assume` anywhere
    is VERIFIED at the default and contradicted by the program, and the
    same query behind a `jit` is dial-moved — with `relational_assumes` and
    `coverage.constrained` both EMPTY, which is the fact
    `stelling.verdict`'s bar-scope argument 2 said could not exist.
    """
    v = check(_constvar_harness(), vacuity_mode="all", semantics="real")
    assert v.status == "VERIFIED", v.render()
    assert float(1.0 / jnp.sum(jnp.asarray(_CONSTVAR_W) ** 3)) > 0.0

    cj = trace(_constvar_harness(wrapped=True))
    opaque = propagate(cj, semantics="real")
    assert opaque.obligations[0].status != "discharged"
    assert opaque.boundary_crossings == 0
    moved = propagate(cj, semantics="real", boundary="transparent")
    assert moved.obligations[0].status == "discharged", (
        moved.obligations[0].detail
    )
    assert moved.boundary_crossings > 0
    assert moved.relational_assumes == (), moved.relational_assumes
    assert moved.coverage.constrained == 0, moved.coverage.constrained


def _dot_harness(n, *, wrapped=False, assumed=True):
    """``1 / (a·b) < 0`` with `a` certified −1 and `b` certified +1, every
    product underflowing to ``-0.0``. `dot_general` is the OTHER reduction
    the disclosure names, so it is measured and not assumed to behave like
    `reduce_sum`."""
    def h():
        a = any_array((n,), "float64", (-1e-200, -1e-210))
        b = any_array((n,), "float64", (1e-200, 1e-190))
        if assumed:
            assume(a < 0.0)
            assume(b > 0.0)
        f = jax.jit(jnp.dot) if wrapped else jnp.dot
        return assert_(1.0 / f(a, b) < 0.0)

    return h


def _dot_runs(n):
    """``1 / (a·b)`` at a point of the declared box, eager and jitted."""
    a = jnp.full((n,), -1e-210, dtype=jnp.float64)
    b = jnp.full((n,), 1e-200, dtype=jnp.float64)
    return float(1.0 / jnp.dot(a, b)), float(1.0 / jax.jit(jnp.dot)(a, b))


def test_dot_general_reaches_it_too_and_the_CONTRACTION_LENGTH_decides():
    """**THE SECOND REDUCTION THE DISCLOSURE NAMES, AND THE SHARPEST FORM
    OF THE DEFECT THIS TREE HAS.**

    :data:`stelling.propagate.BOUNDARY_TRANSPARENT_REACH_DISCLOSURE` names
    `reduce_sum` AND `dot_general`. A sentence naming a primitive nothing
    measures is the defect class this project keeps finding, so the second
    one is measured here.

    MEASURED on jax 0.11.0 and on jax 0.10.2, CPU, ``jax_enable_x64=True``,
    over `a` declared `[-1e-200, -1e-210]` (certified −1) and `b` declared
    `[1e-200, 1e-190]` (certified +1), so that every product underflows to
    `-0.0`:

        n ≤ 32   a·b = -0.0   1/(a·b) = -inf   the VERIFIED holds
        n ≥ 33   a·b = +0.0   1/(a·b) = +inf   the VERIFIED is FALSE

    **and the verdict is the same at every n.** The analysis sees no
    contraction length; the contraction length is what decides whether its
    VERIFIED is true. At n = 32 the verdict is right BY ACCIDENT of a
    lowering — a fused chain that happens to keep the sign — and that half
    is asserted too, because "the analysis is merely conservative here"
    would be the comfortable reading and it is false. The boundary is
    asserted at 32/33, ADJACENT, rather than at two comfortable distances
    from it: a pin two octaves away would survive the threshold moving,
    which is the one event this test exists to report. (**THE FIRST PASS
    WROTE "n ≤ 32 / n ≥ 64" AND ASSERTED 8 AND 64** — the table was right
    about the threshold and the assertions did not touch it. 0.3.0 P1
    re-audit, F8; re-derived here over n = 1..129.)

    Swept independently at `1b34d25` on 2026-08-28, n = 1..129: the sign is
    kept 1..32 and lost from 33, on jax 0.11.0 and jax 0.10.2, eager and
    jitted, float32 identical, and 2-D `A @ B` identical. **`numpy.dot`
    loses it from n = 2**, so unlike the `reduce_sum` seed — where numpy
    agrees with jax from n = 2 and the `+0` is the sum's identity element —
    THIS half really is one backend's lowering choice, and the assertion
    below says so by measuring numpy beside jax.

    A jax whose blocking threshold moves reddens this with the numbers
    above; that is a request to re-measure the table, not a defect in the
    dial.
    """
    small, large = 32, 33
    for n in (small, large):
        assert check(_dot_harness(n), vacuity_mode="all",
                     semantics="real").status == "VERIFIED", n
        assert check(_dot_harness(n, assumed=False), vacuity_mode="all",
                     semantics="real").status == "UNKNOWN", (
            f"n={n}: the query discharged with no `assume` anywhere, so no "
            f"strict-sign certificate was needed and this row is not about "
            f"the certificate at all"
        )

    eager_small, jit_small = _dot_runs(small)
    eager_large, jit_large = _dot_runs(large)
    assert eager_small < 0.0 and jit_small < 0.0, (
        f"n={small}: the compiled dot already seeded +0 at this contraction "
        f"length, so the verdict is not right-by-accident anywhere and the "
        f"measured threshold has moved: {eager_small}, {jit_small}"
    )
    assert eager_large > 0.0 and jit_large > 0.0, (
        f"n={large}: the compiled dot kept the sign of its terms, so the "
        f"+0.0 accumulation seed did not appear at this contraction "
        f"length and the measured table above has moved: "
        f"{eager_large}, {jit_large}"
    )

    # NUMPY, BESIDE IT, because the contrast is the finding: numpy's dot
    # loses the sign at n = 2, so the threshold above is a lowering choice
    # and not an identity of the operation (unlike the `reduce_sum` seed,
    # where the two backends agree from n = 2 — measured in
    # `::test_the_reduction_seed_is_the_mechanism_and_it_is_measured`).
    np_two = np.dot(np.full((2,), -1e-210), np.full((2,), 1e-200))
    assert math.copysign(1.0, float(np_two)) > 0.0, (
        f"numpy kept the sign at n=2, so it now agrees with jax's 1..32 "
        f"band and this contrast no longer holds: {np_two!r}"
    )

    # ...and the dial moves it exactly as it moves the `reduce_sum` chain
    cj = trace(_dot_harness(large, wrapped=True))
    assert propagate(cj, semantics="real").obligations[0].status != "discharged"
    moved = propagate(cj, semantics="real", boundary="transparent")
    assert moved.obligations[0].status == "discharged", moved.obligations[0].detail
    assert moved.boundary_crossings > 0


# ---------------------------------------------------------------------------
# the public front door
# ---------------------------------------------------------------------------


def test_check_passes_the_dial_through_and_stamps_it():
    """`stelling.preconditions.check` is a front door, so the dial has to
    reach it and the verdict has to say which position it was in."""
    from stelling.propagate import BOUNDARY_TRANSPARENT_POSITION

    h = _wrapped_harness(jax.jit, assume_inside=False)
    default = check(h, vacuity_mode="inputs-only")
    assert default.status == "UNKNOWN", default.render()
    assert not any("boundary=" in a for a in default.stamp.assumptions), (
        "the DEFAULT front door grew a stamp line"
    )

    moved = check(h, vacuity_mode="inputs-only", boundary="transparent")
    assert moved.status == "VERIFIED", moved.render()
    assert BOUNDARY_TRANSPARENT_POSITION in moved.stamp.assumptions
    assert any(
        a.startswith("boundary='transparent' CARRIED")
        for a in moved.stamp.assumptions
    ), moved.stamp.assumptions
    # the widen re-check ran at the SAME position — it is the only way a
    # vacuity line exists on this VERIFIED at all
    assert any(
        a.startswith("vacuity checked") or a.startswith("vacuity instrument")
        for a in moved.stamp.assumptions
    ), moved.stamp.assumptions


def test_check_refuses_an_unknown_boundary_before_it_traces():
    """Eagerly, by name, like every other dial on this door. The harness
    raises if it is ever called, so a refusal that arrived after tracing
    would show as that error instead of the ValueError."""
    def h():  # pragma: no cover - reaching this body IS the failure
        raise AssertionError("traced before the dial was validated")

    with pytest.raises(ValueError) as e:
        check(h, vacuity_mode="inputs-only", boundary="see-through")
    assert "'see-through'" in str(e.value), str(e.value)


def test_the_two_SIBLING_doors_deliberately_do_not_take_the_dial():
    """A DECISION, pinned so it is a decision and not a drift.

    `stelling.contracts.check_contract` and
    `stelling.inductive.check_inductive_step` mint VERIFIEDs through the
    same `_pipeline`, and they DO take `falsify` — the downstream check on
    a VERIFIED — for exactly that reason. They take neither `semantics` nor
    `boundary`, which are ANALYSIS-MODE dials that change what the walk is
    allowed to conclude. The two questions are different and this suite
    should notice if the answer to one of them changes silently.

    A caller who wants a boundary-transparent contract runs the trace
    through `check` or through `propagate` directly.
    """
    import inspect

    from stelling.contracts import check_contract
    from stelling.inductive import check_inductive_step
    from stelling.preconditions import check as _check

    assert "boundary" in inspect.signature(_check).parameters
    for door in (check_contract, check_inductive_step):
        params = inspect.signature(door).parameters
        assert "falsify" in params, door.__name__
        assert "semantics" not in params, (
            f"{door.__name__} grew a `semantics` keyword; if the "
            f"analysis-mode dials are being plumbed to the sibling doors, "
            f"`boundary` belongs there too and this pin is the record of "
            f"why it was not"
        )
        assert "boundary" not in params, (
            f"{door.__name__} grew a `boundary` keyword while still not "
            f"taking `semantics`; the two are the same kind of dial and "
            f"plumbing one without the other is the accident this pin "
            f"exists to catch"
        )
