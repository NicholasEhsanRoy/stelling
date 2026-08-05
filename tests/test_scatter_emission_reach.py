# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""WHICH USER-FACING IDIOMS ACTUALLY REACH THE SCATTER SMT EMISSION ROWS.

A tripwire, not a capability test. The scatter rows are the ones
``verdict.VERIFIED_BARRED_PRIMITIVES`` and the ``set-row-agreement`` gate in
``tests/test_scatter_gauge_jax.py`` are built around, and how much those
guards are worth depends entirely on how much traffic the rows see. Today
most of that traffic is stopped somewhere else:

    ``x.at[idx].add(u)`` and ``x.at[idx].set(u)`` with an ARRAY index never
    reach emission. They decline earlier, on ``'add' on dtype 'int32'`` —
    jax's traced negative-index normalisation arithmetic, refused by the
    integer-overflow posture in ``obligation.py``, which has nothing to do
    with scatter.

THAT DECLINE LIVES IN A DIFFERENT PART OF THE SYSTEM FROM THE SCATTER ROWS.
If it is ever improved — an exact model for in-range integer index
arithmetic is an obvious and desirable thing to build — then the most
idiomatic scattering code people write starts reaching the emission row
overnight, with no new test and no signal. This project has been bitten
twice by exactly that shape: a predicate that was correct only because of a
limitation somewhere else, and stopped being correct the day the limitation
was lifted. Hence a build failure instead.

**The norm this rests on: evidence of non-occurrence licenses "unreached",
never "unreachable".** Nothing below says an idiom CANNOT reach the row.
Each line says only that on this build, at this commit, it does not — and
pins that so a change is loud.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.obligation import (  # noqa: E402
    DeclinedObligation,
    slice_obligation,
)
from stelling.propagate import interval_env  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


_DN = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(), inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,),
)
_IDX = np.array([0, 2, 0], dtype=np.int32)
_SEG = np.array([0, 0, 1], dtype=np.int32)


# ---- the census. Every entry is a scattering idiom posed RELATIONALLY, so
# ---- intervals cannot settle it and the question "does it reach emission?"
# ---- is the one being asked.

def i_set_scalar_index():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[0].set(0.5)[1] - x[1] <= 0.0),)


def i_set_negative_scalar_index():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[-1].set(0.5)[1] - x[1] <= 0.0),)


def i_set_scalar_index_traced_update():
    x = any_array((3,), "float64", (0.0, 1.0))
    u = any_array((), "float64", (2.0, 3.0))
    return (assert_(x.at[0].set(u)[1] - x[1] <= 0.0),)


def i_add_scalar_index():
    x = any_array((3,), "float64", (0.0, 1.0))
    u = any_array((), "float64", (0.0, 1.0))
    return (assert_(x.at[0].add(u)[0] - x[0] >= 0.0),)


def i_segment_sum():
    d = any_array((3,), "float64", (0.0, 1.0))
    s = jax.ops.segment_sum(d, jnp.asarray(_SEG), num_segments=2)
    return (assert_(s[0] - d[0] >= 0.0),)


def i_lax_scatter_add():
    x = any_array((3,), "float64", (0.0, 1.0))
    v = any_array((1,), "float64", (0.0, 1.0))
    out = jax.lax.scatter_add(
        x, jnp.asarray(np.array([[0]], dtype=np.int32)), v, _DN,
        indices_are_sorted=False, unique_indices=False,
        mode=jax.lax.GatherScatterMode.FILL_OR_DROP)
    return (assert_(out[0] - x[0] >= 0.0),)


def i_set_array_index():
    x = any_array((3,), "float64", (0.0, 1.0))
    i = jnp.asarray(np.array([0], dtype=np.int32))
    return (assert_(x.at[i].set(0.5)[1] - x[1] <= 0.0),)


def i_add_array_index():
    x = any_array((3,), "float64", (0.0, 1.0))
    v = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[jnp.asarray(_IDX)].add(v)[0] - x[0] >= 0.0),)


def i_zeros_add_array_index():
    v = any_array((3,), "float64", (0.0, 1.0))
    z = jnp.zeros((3,))
    return (assert_(z.at[jnp.asarray(_IDX)].add(v)[0] - v[0] >= 0.0),)


def i_set_multi_axis_index():
    x = any_array((2, 2), "float64", (0.0, 1.0))
    return (assert_(x.at[0, 1].set(0.5)[1, 1] - x[1, 1] <= 0.0),)


def i_set_slice_index():
    x = any_array((4,), "float64", (0.0, 1.0))
    return (assert_(x.at[0:2].set(0.5)[3] - x[3] <= 0.0),)


def i_lax_scatter_window():
    x = any_array((3,), "float64", (0.0, 1.0))
    v = any_array((1,), "float64", (0.0, 1.0))
    out = jax.lax.scatter(
        x, jnp.asarray(np.array([[0]], dtype=np.int32)), v, _DN,
        indices_are_sorted=False, unique_indices=False,
        mode=jax.lax.GatherScatterMode.FILL_OR_DROP)
    return (assert_(out[1] - x[1] <= 0.0),)


def i_apply():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[0].apply(jnp.negative)[1] - x[1] <= 0.0),)


def i_mul():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[0].mul(2.0)[1] - x[1] <= 0.0),)


def i_max():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[0].max(0.5)[1] - x[1] <= 0.0),)


def i_min():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[0].min(0.5)[1] - x[1] <= 0.0),)


def i_segment_max():
    d = any_array((3,), "float64", (0.0, 1.0))
    s = jax.ops.segment_max(d, jnp.asarray(_SEG), num_segments=2)
    return (assert_(s[0] - d[0] >= 0.0),)


CENSUS = [
    "x.at[k].set(const), scalar index",
    "x.at[-k].set(const), negative scalar index",
    "x.at[k].set(traced), scalar index",
    "x.at[k].add(traced), scalar index",
    "jax.ops.segment_sum",
    "jax.lax.scatter_add, hand-written",
    "x.at[arr].set(const), array index",
    "x.at[arr].add(vec), array index",
    "jnp.zeros(...).at[arr].add(vec)",
    "x.at[i, j].set(const), multi-axis index",
    "x.at[a:b].set(const), slice index",
    "jax.lax.scatter, hand-written window form",
    "x.at[k].apply(f)",
    "x.at[k].mul(const)",
    "x.at[k].max(const)",
    "x.at[k].min(const)",
    "jax.ops.segment_max",
]

BUILDS = dict(zip(CENSUS, [
    i_set_scalar_index,
    i_set_negative_scalar_index,
    i_set_scalar_index_traced_update,
    i_add_scalar_index,
    i_segment_sum,
    i_lax_scatter_add,
    i_set_array_index,
    i_add_array_index,
    i_zeros_add_array_index,
    i_set_multi_axis_index,
    i_set_slice_index,
    i_lax_scatter_window,
    i_apply,
    i_mul,
    i_max,
    i_min,
    i_segment_max,
]))

# ── THE PIN ──────────────────────────────────────────────────────────────
#
# Measured on jax 0.11.0. Exactly these idioms put a `scatter` or
# `scatter-add` equation on a slice the SMT emission is asked to encode.
REACHES_THE_EMISSION_ROW = frozenset({
    "x.at[k].set(const), scalar index",
    "x.at[-k].set(const), negative scalar index",
    "x.at[k].set(traced), scalar index",
    "x.at[k].add(traced), scalar index",
    "jax.ops.segment_sum",
    "jax.lax.scatter_add, hand-written",
})

# Why each of the others stops short, quoted from its decline. The FIRST
# group is the one this module exists for: the decline is not about scatter
# at all, and the day it improves those idioms arrive at the row.
STOPS_SHORT_ON = {
    "x.at[arr].set(const), array index": "'add' on dtype 'int32'",
    "x.at[arr].add(vec), array index": "'add' on dtype 'int32'",
    "jnp.zeros(...).at[arr].add(vec)": "'add' on dtype 'int32'",
    "x.at[i, j].set(const), multi-axis index": "outside the measured",
    "x.at[a:b].set(const), slice index": "outside the measured",
    "jax.lax.scatter, hand-written window form": "outside the measured",
    "x.at[k].apply(f)": "carries a combiner",
    "x.at[k].mul(const)": "'scatter-mul' is outside the supported emission set",
    "x.at[k].max(const)": "'scatter-max' is outside the supported emission set",
    "x.at[k].min(const)": "'scatter-min' is outside the supported emission set",
    "jax.ops.segment_max": "non-finite constant",
}

# The subgroup whose decline is OWNED BY ANOTHER PART OF THE SYSTEM: jax
# emits int32 `add`/`select_n` to normalise a possibly-negative traced index
# before the scatter, and the integer-overflow posture in obligation.py
# refuses it. Nothing here is a judgement about the scatter rows.
BLOCKED_BY_INDEX_NORMALISATION = frozenset({
    "x.at[arr].set(const), array index",
    "x.at[arr].add(vec), array index",
    "jnp.zeros(...).at[arr].add(vec)",
})

_WHAT_TO_DO = (
    "\n\nWHAT TO DO: this is not a test to re-pin and move on. The set of "
    "idioms that reach the scatter SMT emission rows has changed, so the "
    "coverage those rows rest on has to be revisited before the pin is "
    "updated:\n"
    "  1. tests/test_scatter_gauge_jax.py — does the gauge battery exercise "
    "the newly-arriving shape? A mutation gauge measures only the shapes it "
    "drives.\n"
    "  2. verdict.VERIFIED_BARRED_PRIMITIVES and design/scatter-rows.md — "
    "the adversarial pass that cleared the accumulate rows was point-in-time "
    "and covered the forms reachable THEN.\n"
    "  3. docs/supported-primitives.md and the census — the reach limits "
    "recorded there are now wrong.\n"
    "Then re-pin, with the measurement in the commit message."
)


def _reach(build):
    """(reached, reason, scatter-family primitives in the query).

    ``reached`` is "the SMT emission is asked to encode a scatter equation":
    the obligation slices without declining AND a scatter-family equation
    survives onto the slice. Measured at the slice, not at the verdict, so
    the VERIFIED bar — which acts later — cannot confound it."""
    closed = trace(build)
    family = tuple(sorted({
        str(e.primitive) for e in closed.jaxpr.eqns
        if str(e.primitive).startswith("scatter")
    }))
    sl = slice_obligation(closed, 0, interval_env(closed))
    if isinstance(sl, DeclinedObligation):
        return False, sl.reason, family
    on_slice = {str(e.primitive) for e in sl.eqns} & {"scatter", "scatter-add"}
    return bool(on_slice), "", family


def test_the_set_of_idioms_reaching_the_emission_row_is_pinned():
    measured = {name for name in CENSUS if _reach(BUILDS[name])[0]}
    arrived = measured - REACHES_THE_EMISSION_ROW
    left = REACHES_THE_EMISSION_ROW - measured
    assert measured == REACHES_THE_EMISSION_ROW, (
        f"the reach of the scatter SMT emission rows moved.\n"
        f"  newly REACHING: {sorted(arrived)}\n"
        f"  no longer reaching: {sorted(left)}" + _WHAT_TO_DO
    )


def test_the_array_index_idioms_are_stopped_by_something_else_entirely():
    """The load-bearing half of the pin, and the reason this file exists.

    These are the idiomatic spellings — ``x.at[idx].add(v)`` is how people
    scatter — and they are held back by the integer-overflow posture on jax's
    index-normalisation arithmetic, a rule with no opinion about scatter. It
    is a limitation, not a design, and improving it is a reasonable thing for
    someone to do without ever reading the scatter rows.
    """
    for name in sorted(BLOCKED_BY_INDEX_NORMALISATION):
        reached, reason, _family = _reach(BUILDS[name])
        assert not reached, (
            f"{name!r} now REACHES the scatter emission row. The index-"
            f"normalisation decline that used to stop it has been lifted or "
            f"narrowed." + _WHAT_TO_DO
        )
        assert "'add' on dtype 'int32'" in reason, (
            f"{name!r} still stops short, but for a DIFFERENT reason than "
            f"the index-normalisation decline this pin tracks:\n  {reason}\n"
            f"Re-derive which rule is now holding it back before re-pinning "
            f"— the guarantee is only as good as the rule behind it."
        )


def test_every_non_reaching_idiom_declines_for_its_recorded_reason():
    for name, expect in sorted(STOPS_SHORT_ON.items()):
        reached, reason, _family = _reach(BUILDS[name])
        assert not reached, (
            f"{name!r} now reaches the scatter emission row." + _WHAT_TO_DO
        )
        assert expect in reason, (
            f"{name!r} declines for a reason other than the recorded one.\n"
            f"  recorded: {expect!r}\n  measured: {reason}"
        )


def test_the_census_is_not_vacuous():
    """ANTI-VACUITY (Norm C). A census of idioms that turned out not to be
    scattering idioms at all would pin nothing: every entry would 'not reach'
    for reasons having nothing to do with the rows. So prove each entry
    really does lower to a scatter-family primitive, that the census covers
    both answers, and that its two halves partition it."""
    assert set(BUILDS) == set(CENSUS)
    assert set(STOPS_SHORT_ON) | REACHES_THE_EMISSION_ROW == set(CENSUS), (
        "an idiom is in neither the reaching pin nor the stops-short table"
    )
    assert not (set(STOPS_SHORT_ON) & REACHES_THE_EMISSION_ROW)
    assert BLOCKED_BY_INDEX_NORMALISATION <= set(STOPS_SHORT_ON)
    for name in CENSUS:
        _reached, _reason, family = _reach(BUILDS[name])
        assert family, (
            f"{name!r} lowers to no scatter-family primitive at all "
            f"({family}) — it is not a scattering idiom and pins nothing"
        )
