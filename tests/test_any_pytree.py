# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""any_pytree: tracing-time sugar over any_array, and nothing more.

"Faithful sugar" is defined by content-hash equality: a harness declared
through :func:`stelling.harness.any_pytree` must trace to the *identical*
query — hash-equal — as its hand-declared original. The acceptance tests
here hold that bar against the two worked examples in
``corpus/supply/pytree_probe.py`` (h_clean and h_hard), rebuilt with
sugar declarations inside the tests (the probe file itself is read-only).

Skipped without jax; the acceptance tests additionally need diffrax and
blackjax (the probe's own pinned libraries).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

jax = pytest.importorskip("jax")

import numpy as np  # noqa: E402

from stelling.harness import any_array, any_pytree, assert_, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402

PROBE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "corpus"
    / "supply"
    / "pytree_probe.py"
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def eqns(cj):
    return [e.primitive for e in cj.jaxpr.eqns]


# --- leaf rules --------------------------------------------------------------


def test_sugar_traces_hash_identical_to_hand_declaration():
    # dict pytrees flatten in sorted-key order: "b" before "x"
    def hand():
        b = any_array((), "bool", (0.0, 1.0))
        x = any_array((2,), "float64", (0.0, 1.0))
        return assert_(x <= 1.0), b

    def sugar():
        t = any_pytree(
            {"b": np.zeros((), bool), "x": np.zeros((2,))},
            {"b": (0.0, 1.0), "x": (0.0, 1.0)},
        )
        return assert_(t["x"] <= 1.0), t["b"]

    assert trace(sugar).content_hash() == trace(hand).content_hash()


def test_broadcast_bounds_reach_every_array_leaf():
    def h():
        t = any_pytree({"a": np.zeros((2,)), "b": np.zeros(())}, (0.0, 1.0))
        return assert_(t["a"] <= 1.0), assert_(t["b"] <= 1.0)

    cj = trace(h)
    assert eqns(cj).count("stelling_any") == 2
    any_params = [
        e.params_dict() for e in cj.jaxpr.eqns if e.primitive == "stelling_any"
    ]
    assert all((p["lo"], p["hi"]) == (0.0, 1.0) for p in any_params)
    assert propagate(cj).all_discharged


def test_aliased_leaf_declared_once_and_shared():
    seen = {}

    def h():
        z = np.zeros(())
        t = any_pytree({"first": z, "second": z}, (0.0, 1.0))
        seen["shared"] = t["first"] is t["second"]
        return assert_(t["first"] >= 0.0)

    cj = trace(h)
    assert seen["shared"] is True  # one traced value at both positions
    assert eqns(cj).count("stelling_any") == 1  # never two declarations


def test_distinct_equal_leaves_get_independent_declarations():
    seen = {}

    def h():
        t = any_pytree({"first": np.zeros(()), "second": np.zeros(())}, (0.0, 1.0))
        seen["shared"] = t["first"] is t["second"]
        return assert_(t["first"] >= 0.0)

    cj = trace(h)
    assert seen["shared"] is False  # never one declaration for two objects
    assert eqns(cj).count("stelling_any") == 2


def test_aliased_leaf_with_conflicting_bounds_refused():
    z = np.zeros(())

    def h():
        return any_pytree((z, z), ((0.0, 1.0), (0.0, 2.0)))

    with pytest.raises(ValueError, match="aliases"):
        trace(h)


def test_static_leaves_pass_through_verbatim():
    marker = "configuration string"
    seen = {}

    def h():
        t = any_pytree(
            {"w": np.zeros(()), "c": 2.5, "s": marker, "n": None},
            {"w": (0.0, 1.0), "c": None, "s": None, "n": None},
        )
        seen["c"], seen["s"], seen["n"] = t["c"], t["s"], t["n"]
        return assert_(t["w"] + t["c"] <= 4.0)

    cj = trace(h)
    assert seen["c"] == 2.5 and seen["s"] is marker and seen["n"] is None
    assert eqns(cj).count("stelling_any") == 1  # only the array leaf declares
    assert propagate(cj).all_discharged  # [0,1] + 2.5 <= 4


def test_prng_key_leaf_refused_pointing_at_wrap_key_data():
    key = jax.random.key(0)
    with pytest.raises(TypeError, match="wrap_key_data"):
        any_pytree({"k": key}, (0.0, 1.0))


def test_bounds_structure_mismatch_refused():
    with pytest.raises(ValueError, match="any_pytree"):
        any_pytree({"a": np.zeros(())}, {"wrong_key": (0.0, 1.0)})


def test_static_leaf_given_bounds_refused():
    # "a" sorts before "b", so the static leaf is checked before anything binds
    with pytest.raises(ValueError, match="static"):
        any_pytree(
            {"a": 2.5, "b": np.zeros(())}, {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        )


def test_malformed_bounds_pair_refused():
    with pytest.raises(ValueError, match="pair"):
        any_pytree({"a": np.zeros(())}, {"a": (0.0, 1.0, 2.0)})


def test_empty_bounds_refused_like_any_array():
    # the empty-set refusal is any_array's own, reached through the sugar
    def h():
        return any_pytree({"a": np.zeros(())}, (5.0, 3.0))

    with pytest.raises(ValueError, match="empty"):
        trace(h)


# --- acceptance: the pytree probe, sugar-declared, hash-equal ----------------


@pytest.fixture(scope="module")
def probe(_x64):
    pytest.importorskip("diffrax")
    pytest.importorskip("blackjax")
    spec = importlib.util.spec_from_file_location(
        "stelling_test_pytree_probe", PROBE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # the probe runs (and prints) its own cases
    return mod


def test_h_clean_sugar_hash_equals_hand_declaration(probe):
    """h_clean with any_pytree declarations: the real _PidState tuple as one
    tree, the scalars grouped by declaration phase. Hash equality is the
    acceptance bar for 'faithful sugar'."""

    def h_clean_sugar():
        t0, prev_dt = any_pytree(
            (np.zeros(()), np.zeros(())),
            ((0.0, 1.0), (1e-4, 1e-1)),
        )
        t1 = t0 + prev_dt
        y0, y1c, y_err = any_pytree(
            (np.zeros((1,)), np.zeros((1,)), np.zeros((1,))),
            ((-1.0, 1.0), (-1.0, 1.0), (-1e-4, 1e-4)),
        )
        # the real _PidState, declared as the pytree it is:
        # (prev_inv_scaled_error, prev_prev_inv_scaled_error, at_dtmin)
        state = any_pytree(
            (np.zeros(()), np.zeros(()), np.zeros((), dtype=bool)),
            ((0.1, 10.0), (0.1, 10.0), (0.0, 1.0)),
        )
        keep, nt0, nt1, made_jump, new_state, result = (
            probe.controller.adapt_step_size(
                t0, t1, y0, y1c, None, y_err, 5.0, state
            )
        )
        return assert_((nt1 - nt0) >= probe.DTMIN)

    hand = trace(probe.h_clean).content_hash()
    sugar = trace(h_clean_sugar).content_hash()
    assert sugar == hand


def test_h_hard_sugar_hash_equals_hand_declaration(probe):
    """h_hard with any_pytree declarations. The forall-key declarations stay
    exactly the probe's: raw uint32 bits declared (bare-leaf pytrees), keys
    built by the library's own wrap_key_data — any_pytree refuses key-dtype
    leaves, so this is also the shape the refusal points the caller at."""
    import jax.numpy as jnp

    def h_hard_sugar():
        bits_i = any_pytree(np.zeros((2,), np.uint32), (0.0, 4294967295.0))
        key_i = jax.random.wrap_key_data(bits_i)
        pos = any_pytree(np.zeros((2,)), (-1.0, 1.0))
        state = probe.mclmc_init(pos, probe.logdensity_fn, key_i)
        bits_s = any_pytree(np.zeros((2,), np.uint32), (0.0, 4294967295.0))
        key_s = jax.random.wrap_key_data(bits_s)
        step_size = any_pytree(np.zeros(()), (0.01, 1.0))
        next_state, info = probe.kernel(
            key_s, state, probe.logdensity_fn, jnp.ones(2), 1.0, step_size
        )
        return assert_(next_state.logdensity <= 2.0 * jnp.log(probe.BOUND))

    hand = trace(probe.h_hard).content_hash()
    sugar = trace(h_hard_sugar).content_hash()
    assert sugar == hand
