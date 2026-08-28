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
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from stelling.coverage import DEFAULT_TRANSPARENT  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import propagate  # noqa: E402

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
