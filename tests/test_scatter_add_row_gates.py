# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Soundness gates on the static-index scatter ADD row.

The row models `x.at[k].add(v)` as accumulation: element k's term is the
operand's PLUS the update's. The gate below marks a form for which that
model would be WRONG, so admitting it emits a formula describing a
different program than the one traced.

The combiner gate is here because an adversarial audit found it missing.

Scoped to the ADD row deliberately. The audit that found this defect ran
on a branch that also carried a scatter SET row; that row is PARKED and
unmerged, so its gates are not testable here and are not in this file.
This defect is not scatter-set work and was parked only by accident.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check
from _solver_gate import need_solver  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    # scoped, not set at import: a module-level flip leaks float64 into every
    # later-run module in the same process
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def test_scatter_add_with_a_present_none_combiner_is_not_modelled_as_add():
    """`update_jaxpr` PRESENT with value None is jax's SET combiner.

    Found by adversarial audit, and it is the mirror of the SET row's defect:
    there the distinguishing field was a non-None `update_jaxpr`; here it is a
    None one.

    Key ABSENT and key PRESENT-with-value-None are different facts, and jax
    uses the difference. An absent key is the hand-built IR form, where the
    primitive name is the semantic authority. A present-None key is what a
    jax-produced equation carries, and jax's `_scatter_lower` substitutes
    `lambda x, y: y` for it — REPLACE, last-wins, operand discarded and
    duplicates NOT accumulated. Measured on jax 0.11.0: operand zeros(3),
    indices [[0],[2],[0],[0]], updates [1,10,100,1000] gives [1101,0,10] under
    the add combiner and [1000,0,10] under update_jaxpr=None.

    Reading it with `.get()` conflated the two and modelled a set as an add.
    Before the fix this query returned VERIFIED with BOTH solvers answering
    unsat, on a property false at every point of the declared box — and
    scatter-add is not covered by the VERIFIED bar, so nothing downstream
    would have caught it.
    """
    # public surface only. Private jax modules are banned everywhere in
    # `tests/` and everywhere under `src/` except `_tripwire/_adapter_jax.py`,
    # which carries the one exemption -- see `design/private-jax-boundary.md`.
    from jax.lax import (
        GatherScatterMode,
        ScatterDimensionNumbers,
        scatter_add_p,
    )

    dn = ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    idx = jnp.array([[0], [2], [0], [0]], dtype=jnp.int32)
    upd = jnp.array([1.0, 10.0, 100.0, 1000.0])

    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = scatter_add_p.bind(
            x, idx, upd, update_jaxpr=None, update_consts=(),
            dimension_numbers=dn, indices_are_sorted=False,
            unique_indices=False, mode=GatherScatterMode.FILL_OR_DROP,
        )
        # false at EVERY point: y[0] is the constant 1000.0 and x[0] >= 0
        return (assert_(y[0] - x[0] >= 1100.0),)

    v = check(q, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert v.status != "VERIFIED", (
        "a property false at every point was VERIFIED: the present-None "
        "combiner is being modelled as addition again"
    )


@need_solver
def test_plain_scatter_add_still_verifies():
    """Anti-vacuity for the test above: the add row must still work.

    If the gate closed the row entirely, that test would pass for the wrong
    reason.
    """
    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        return (assert_(x.at[0].add(5.0)[0] >= 5.0),)

    v = check(q, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert v.status == "VERIFIED", f"genuine .at[].add regressed: {v.status}"


def test_present_none_combiner_box_contains_the_executed_truth():
    """The SOLVER-INDEPENDENT face of the same defect.

    The sibling test above asks whether a solver-backed VERIFIED can be
    minted. This one asks the prior question: does the interval box CONTAIN
    the value the program actually computes? That is the claim the whole
    abstraction rests on, and it needs no solver, no emission and no replay
    to check — run the program, read the box, test membership.

    It is the leg that matters most. stelling's refute side is a search
    procedure whose output is checkable by execution, and that asymmetry is
    what FLAGSHIP leads with; an unsound box breaks it at the root rather
    than at the solver.

    Measured before the fix, on main: executed truth y[0] = 1000.0 against a
    box of [1100.9999999999998, 1102.0000000000002]. The box did not contain
    the truth.
    """
    import numpy as np
    from jax.lax import (
        GatherScatterMode,
        ScatterDimensionNumbers,
        scatter_add_p,
    )

    from stelling._jax_compat import transcribe
    from stelling.propagate import interval_env

    dn = ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    idx = jnp.array([[0], [2], [0], [0]], dtype=jnp.int32)
    upd = jnp.array([1.0, 10.0, 100.0, 1000.0])
    kw = dict(update_jaxpr=None, update_consts=(), dimension_numbers=dn,
              indices_are_sorted=False, unique_indices=False,
              mode=GatherScatterMode.FILL_OR_DROP)

    # ground truth by EXECUTION at a concrete point inside the declared box
    truth = np.asarray(scatter_add_p.bind(jnp.zeros(3), idx, upd, **kw))

    def build():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = scatter_add_p.bind(x, idx, upd, **kw)
        return (assert_(y[0] >= -1e30),)

    cj = transcribe(jax.make_jaxpr(build)())
    env = interval_env(cj)
    boxes = [env.get(e.outvars[0].id) for e in cj.jaxpr.eqns
             if "scatter" in str(e.primitive)]
    assert boxes and boxes[0] is not None, "no box for the scatter equation"
    box = boxes[0]

    # x = zeros is in the declared box [0,1]^3, so the executed value must be
    # inside the propagated interval for every element. If the row models a
    # replace as an add, element 0 is off by the operand's contribution.
    for i, t in enumerate(truth):
        assert box.los[i] <= t <= box.his[i], (
            f"UNSOUND BOX: element {i} executes to {t} but the interval is "
            f"[{box.los[i]}, {box.his[i]}] — the present-None combiner is "
            f"being modelled as addition again"
        )
