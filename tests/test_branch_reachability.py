# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Two sub-jaxpr defects, one direction: claims about code nothing ran.

**Branch reachability.** Refuting inside a branch presumes a reachability
that nothing certifies. The index interval ADMITTING a branch is not
evidence that any point of the declared set takes it — ``x - x > 0``
admits both legs of a ``cond`` while being false at every point — so a
definite violation found inside an admitted-only branch is withheld from
REFUTED. What certifies a branch is a POINT of the declared box that
takes it, which is what the pinned probe runs search for; that is why a
genuinely satisfiable guard keeps refuting and a definitely-false one
does not.

**Unexamined obligations.** An ``assert_`` inside a ``scan`` or ``while``
body is transcribed into the IR and never descended into. Before this it
vanished: ``obligations = 0``, no note. Measured, it could ride under a
**VERIFIED** — one true top-level assert beside a false one inside a
scan stamped VERIFIED. It is now recorded ``unknown`` with its location
and the swallowing primitive named.

The oracle for every reachability claim below is stated in the test: the
declared box and the guard, worked out by hand. No test here asks
stelling what the answer is.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import (  # noqa: E402
    UNCERTIFIED_REACHABILITY_REFUSAL,
    propagate,
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def statuses(harness):
    return [o.status for o in propagate(trace(harness)).obligations]


# --- the reported defect, both polarities ----------------------------------


def test_violation_in_a_definitely_false_branch_is_withheld():
    # guard is 0 > 0: FALSE at every point of the declared box, so the
    # 'yes' branch never executes and `v > 5.0` is never evaluated.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            x[0] - x[0] > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    p = propagate(trace(h))
    assert sorted(o.status for o in p.obligations) == ["discharged", "unknown"]
    assert not p.any_violated
    withheld = [o for o in p.obligations if o.status == "unknown"][0]
    assert "UNCERTIFIED" in withheld.detail
    assert any(UNCERTIFIED_REACHABILITY_REFUSAL in n for n in p.notes)
    assert check(h, vacuity_mode="all").status == "UNKNOWN"


def test_the_other_polarity_a_definitely_true_guard_strands_the_no_branch():
    # guard is 0 <= 0: TRUE everywhere, so the 'no' branch never executes.
    # Checked because a fix that looked only at the 'yes' side would pass
    # the test above and leave this one refuting.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            x[0] - x[0] <= 0.0,
            lambda v: assert_(v > -9.0),
            lambda v: assert_(v > 5.0),
            x,
        )

    assert sorted(statuses(h)) == ["discharged", "unknown"]


def test_the_rule_is_reachability_not_cancellation():
    # No `x - x` anywhere: the guard runs through `sin`, which has no
    # registered transfer, so the index is ⊤ and neither branch is
    # certified. A fix that special-cased cancellation would refute here.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            jnp.sin(x[0] * 0.0) > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert "violated-over-set" not in statuses(h)


def test_the_withheld_obligation_is_never_discharged():
    # Vacuous truth is a DIFFERENT claim and it is reserved: an
    # unreachable obligation must not become the evidence for a VERIFIED.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            x[0] - x[0] > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    v = check(h, vacuity_mode="all")
    assert v.status == "UNKNOWN"
    assert v.status != "VERIFIED"
    assert [o.status for o in v.obligations].count("discharged") == 1


def test_nested_cond_inner_unreachable_branch_is_withheld():
    # outer guard satisfiable, inner guard definitely false: only the
    # inner obligation is stranded, and the depth must not lose it.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        inner_y = lambda v: assert_(v > 5.0)  # noqa: E731
        inner_n = lambda v: assert_(v > -9.0)  # noqa: E731
        outer_y = lambda v: jax.lax.cond(  # noqa: E731
            x[1] - x[1] > 0.0, inner_y, inner_n, v
        )
        outer_n = lambda v: assert_(v > -8.0)  # noqa: E731
        return jax.lax.cond(x[0] > 0.0, outer_y, outer_n, x)

    assert "violated-over-set" not in statuses(h)


# --- the neighbours that must NOT move -------------------------------------


def test_a_satisfiable_guard_keeps_refuting():
    # x[0] > 0 is TRUE at x = (1, 1, 1), a point of the declared box, so
    # the 'yes' branch really runs and `v > 5.0` really fails there. The
    # pinned corner probe finds exactly that witness.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            x[0] > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert "violated-over-set" in statuses(h)
    assert check(h, vacuity_mode="all").status == "REFUTED"


def test_a_satisfiable_guard_through_registered_arithmetic_keeps_refuting():
    # x[0]*x[0] > 0.25 straddles over [-1, 1] and is satisfied at the
    # corner: the certificate is a witness, not a syntactic shape.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            x[0] * x[0] > 0.25,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert "violated-over-set" in statuses(h)


def test_a_forced_branch_keeps_refuting():
    # the index interval determines the branch: it runs whenever the cond
    # runs, so nothing about reachability is being presumed.
    def h():
        c = any_array((), "float64", (1.0, 2.0))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            c > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert statuses(h) == ["violated-over-set"]
    assert check(h, vacuity_mode="all").status == "REFUTED"


def test_a_literal_guard_keeps_refuting():
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            True, lambda v: assert_(v > 5.0), lambda v: assert_(v > -9.0), x
        )

    assert statuses(h) == ["violated-over-set"]


def test_a_top_level_violation_is_untouched():
    # the withholding is scoped to branches; nothing at top level moves.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        a = assert_(x > 5.0)
        b = jax.lax.cond(
            x[0] - x[0] > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )
        return a, b

    assert statuses(h).count("violated-over-set") == 1
    assert check(h, vacuity_mode="all").status == "REFUTED"


def test_a_branch_the_guard_forces_the_other_way_still_refutes():
    # The SAME guard shape as the reported defect, other leg: `x - x > 0`
    # is false everywhere, so the 'no' branch is taken at every point and
    # its violation is a real one. The witness probe certifies it.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            x[0] - x[0] > 0.0,
            lambda v: assert_(v > -9.0),
            lambda v: assert_(v > 5.0),
            x,
        )

    assert "violated-over-set" in statuses(h)


def test_an_obligation_inside_a_jit_wrapper_is_unmoved():
    # jit is a TRANSPARENT primitive: it is descended into and executes
    # unconditionally, so nothing here is branch-scoped.
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        return jax.jit(lambda v: assert_(v > 5.0))(x)

    assert statuses(h) == ["violated-over-set"]


# --- obligations inside sub-jaxprs nothing descends into --------------------


def _scan_harness():
    x = any_array((3,), "float64", (1.0, 2.0))

    def body(c, v):
        return c, assert_(v > 5.0)

    return jax.lax.scan(body, 0.0, x)


def test_the_scan_body_assert_is_in_the_ir_and_used_to_vanish():
    from stelling.coverage import sub_jaxprs

    cj = trace(_scan_harness)
    found = []
    stack = [cj.jaxpr]
    while stack:
        j = stack.pop()
        for e in j.eqns:
            found.append(e.primitive)
            stack.extend(sub_jaxprs(e))
    # the mechanism: transcribed, never descended into
    assert "scan" in found and "stelling_assert" in found
    assert [e.primitive for e in cj.jaxpr.eqns].count("stelling_assert") == 0


def test_an_assert_inside_a_scan_is_recorded_and_named():
    p = propagate(trace(_scan_harness))
    assert [o.status for o in p.obligations] == ["unknown"]
    (o,) = p.obligations
    assert "NOT EXAMINED" in o.detail
    assert "scan" in o.detail
    note = [n for n in p.notes if "NOT EXAMINED" in n]
    assert note, p.notes
    # the note names the SOURCE LOCATION and the swallowing primitive
    assert "'scan'" in note[0]
    assert o.source_info and o.source_info[-1].split(" (")[0] in note[0]


def test_an_assert_inside_a_while_is_recorded_and_named():
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))

        def body(s):
            i, a = s
            return i + 1, jnp.where(assert_(a > 5.0), a, a)

        return jax.lax.while_loop(lambda s: s[0] < 3, body, (0, x[0]))

    p = propagate(trace(h))
    assert [o.status for o in p.obligations] == ["unknown"]
    assert "while" in p.obligations[0].detail
    assert any("NOT EXAMINED" in n for n in p.notes)


def test_a_swallowed_obligation_cannot_ride_under_a_verified():
    # THE measured false VERIFIED on c4133f8: one true top-level assert
    # beside one false assert inside a scan stamped VERIFIED, with the
    # scan obligation nowhere in the report.
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        ok = assert_(x[0] > 0.0)

        def body(c, v):
            return c, assert_(v > 5.0)

        jax.lax.scan(body, 0.0, x)
        return ok

    v = check(h, vacuity_mode="all")
    assert v.status == "UNKNOWN"
    assert len(v.obligations) == 2
    assert sorted(o.status for o in v.obligations) == ["discharged", "unknown"]


def test_an_unexamined_obligation_is_never_discharged():
    # the whole point: no evidence in either direction, so no definite
    # face — an unexamined check must not complete a VERIFIED.
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))

        def body(c, v):
            return c, assert_(v > -9.0)  # TRUE over the box, still unexamined

        return jax.lax.scan(body, 0.0, x)

    assert [o.status for o in propagate(trace(h)).obligations] == ["unknown"]


# --- the cost clause, with its positive control ----------------------------


def _count_propagators(harness):
    import stelling.propagate as P

    calls = []
    original = P._Propagator.__init__

    def counting(self, *a, **k):
        calls.append(1)
        return original(self, *a, **k)

    P._Propagator.__init__ = counting
    try:
        propagate(trace(harness))
    finally:
        P._Propagator.__init__ = original
    return len(calls)


def test_the_witness_search_is_not_run_without_a_candidate():
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        return assert_(x > 0.0)

    assert _count_propagators(h) == 1


def test_the_witness_search_does_run_when_there_is_a_candidate():
    # the POSITIVE CONTROL for the clause above: without it, the zero it
    # measures could be a search that never runs at all.
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.cond(
            x[0] - x[0] > 0.0,
            lambda v: assert_(v > 5.0),
            lambda v: assert_(v > -9.0),
            x,
        )

    assert _count_propagators(h) > 1


# --- the measured capability cost, pinned so a future fix is visible --------


def test_no_switch_branch_can_refute_while_clamp_is_unregistered():
    """``jax.lax.switch`` clamps its index, and ``clamp`` has no transfer.

    The index is therefore ⊤ on every switch, no branch is ever forced,
    and no point probe can evaluate the clamp either — so every
    branch-scoped violation under a ``switch`` is withheld. That is the
    honest posture given the registry, and it is pinned here rather than
    left implicit: registering a ``clamp`` transfer flips this test, which
    is exactly the notice such a change should give.
    """

    def h():
        k = any_array((), "int32", (0.0, 2.0))
        x = any_array((3,), "float64", (-1.0, 1.0))
        return jax.lax.switch(
            k,
            [lambda v: assert_(v > 5.0), lambda v: v > 1.0, lambda v: v > 2.0],
            x,
        )

    p = propagate(trace(h))
    assert [o.status for o in p.obligations] == ["unknown"]
    assert "clamp" in dict(p.coverage.unknown_primitives)
