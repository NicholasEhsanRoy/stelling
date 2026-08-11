# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Soundness gates on the static-index scatter SET and ADD rows.

The row models `x.at[k].set(v)` as pure data movement: element k's term IS
the update's, every other element's IS the operand's. Each gate below marks
a form for which that model would be WRONG, so admitting it would emit a
formula describing a different program than the one traced.

The combiner gate is here because an adversarial audit found it missing.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    # scoped, not set at import: a module-level flip leaks float64 into every
    # later-run module in the same process
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def test_apply_is_not_admitted_as_set():
    """`x.at[k].apply(f)` must never be modelled as `x.at[k].set(...)`.

    Found by adversarial audit. `.apply` traces to the SAME primitive with
    the SAME dimension numbers, shapes, mode and static index as `.set` —
    the only distinguishing field is a non-None `update_jaxpr` carrying f,
    beside a DUMMY updates operand. A form test that does not read it admits
    `.apply` as a set and models `out[k] = <dummy>` where the program
    computes `out[k] = f(operand[k])`.

    Before the gate this query returned REFUTED with witness x=(2,2,2), and
    the exact-rational replay CONFIRMED that witness — because replay drives
    the same plan and so re-derived the same wrong value. Replay is
    independent of the SOLVER, not of the plan, so it cannot catch a wrong
    plan. That is what makes this class of defect worth a gate rather than a
    downstream check.

    The property below is trivially true on the declared box: y == x and
    every x element is at least 2.
    """
    def q():
        x = any_array((3,), "float32", (2.0, 3.0))
        y = x.at[0].apply(lambda t: t)
        return (assert_(y[0] >= 1.0),)

    v = check(q, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert v.status != "REFUTED", (
        "a true property was refuted: the combiner gate has been removed and "
        "`.apply` is being modelled as `.set`"
    )
    assert not v.witnesses
    assert any("combiner" in n for n in v.notes), (
        "the decline must name the combiner; blaming the shapes sends the "
        "reader to fields that are identical to a legitimate `.set`"
    )


def test_plain_set_still_reaches_the_solver():
    """The gate must not close the row it exists to protect.

    Anti-vacuity for the test above: if `.set` also stopped being admitted,
    that test would pass for the wrong reason.
    """
    def q():
        x = any_array((3,), "float32", (0.0, 1.0))
        y = x.at[0].set(5.0)
        return (assert_(y[0] >= 4.0),)

    v = check(q, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert not any("combiner" in n for n in v.notes)
    assert not any("outside the measured static-index set row form" in n
                   for n in v.notes), "a plain .set must be admitted by the row"


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
