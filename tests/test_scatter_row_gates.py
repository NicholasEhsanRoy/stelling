# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Soundness gates on the static-index scatter SET row.

The row models `x.at[k].set(v)` as pure data movement: element k's term IS
the update's, every other element's IS the operand's. Each gate below marks
a form for which that model would be WRONG, so admitting it would emit a
formula describing a different program than the one traced.

The combiner gate is here because an adversarial audit found it missing.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from stelling.harness import any_array, assert_
from stelling.preconditions import check


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
