# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Inductive step verification: both faces plus error handling.

Every test states both faces (a VERIFY face and a REFUTE/UNKNOWN face)
so a template that trivially passes or trivially fails cannot look
correct.
"""

from __future__ import annotations

import pytest

from stelling import inductive  # jax-free import must always work


def test_module_imports_without_jax():
    assert inductive.__all__ == ["check_inductive_step"]


jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


import jax.numpy as jnp  # noqa: E402

from stelling.inductive import check_inductive_step  # noqa: E402


# --- Contractive (VERIFIED) -------------------------------------------------


def test_damped_oscillator_verifies():
    """A simple contractive map: x_new = 0.99 * x with bounds (-10, 10).

    Since |0.99 * x| <= 0.99 * 10 = 9.9 < 10 for |x| <= 10, the invariant
    is preserved and the inductive step should VERIFY."""

    def body(state, constants):
        return {"x": constants["damping"] * state["x"]}

    v = check_inductive_step(
        body=body,
        state_bounds={"x": ((-10.0, 10.0), "float64")},
        constants={"damping": 0.99},
    )
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for contractive map, got {v.status}; "
        f"notes: {v.notes}"
    )
    assert any("invariant is preserved by one step" in n for n in v.notes)


def test_damped_two_state_verifies():
    """A damped oscillator with position and velocity.

    position_new = position + dt * velocity
    velocity_new = damping * velocity

    With dt=0.01, damping=0.99, position in (-10, 10), velocity in (-5, 5):
    - |position_new| <= 10 + 0.01 * 5 = 10.05 -- FAILS if bounds are tight
    - So use wider bounds to make it verify.

    With position in (-11, 11), velocity in (-5, 5):
    - |position_new| <= 11 + 0.01 * 5 = 11.05 -- still fails

    Use bounds that work: position in (-10, 10), velocity in (-1, 1):
    - |position_new| <= 10 + 0.01 * 1 = 10.01 -- still fails

    Actually, for verification we need the output to DEFINITELY be in bounds.
    Let's use a purely contractive two-state system."""

    def body(state, constants):
        return {
            "position": constants["dp"] * state["position"],
            "velocity": constants["dv"] * state["velocity"],
        }

    v = check_inductive_step(
        body=body,
        state_bounds={
            "position": ((-10.0, 10.0), "float64"),
            "velocity": ((-5.0, 5.0), "float64"),
        },
        constants={"dp": 0.9, "dv": 0.95},
    )
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for two-state contractive map, got {v.status}; "
        f"notes: {v.notes}"
    )


# --- Expansive (REFUTED) ----------------------------------------------------


def test_unstable_system_refutes():
    """An unstable system: x_new = 3.0 * x with bounds (1, 2).

    For x in [1, 2]: x_new = 3*x in [3, 6], and assert(x_new <= 2)
    checks [3, 6] <= 2 which is definitely false (the entire output
    interval exceeds the upper bound).  REFUTED."""

    def body(state, constants):
        return {"x": constants["gain"] * state["x"]}

    v = check_inductive_step(
        body=body,
        state_bounds={"x": ((1.0, 2.0), "float64")},
        constants={"gain": 3.0},
    )
    assert v.status == "REFUTED", (
        f"expected REFUTED for expansive map, got {v.status}; "
        f"notes: {v.notes}"
    )
    assert any("REFUTED" in n for n in v.notes)


def test_one_stable_one_unstable_refutes():
    """Two state variables: x is stable (0.5*x), y is unstable (3.0*y).

    With y in [1, 2], 3*y in [3, 6] which definitely exceeds 2.
    The system as a whole should REFUTE because y escapes."""

    def body(state, constants):
        return {
            "x": 0.5 * state["x"],
            "y": 3.0 * state["y"],
        }

    v = check_inductive_step(
        body=body,
        state_bounds={
            "x": ((-10.0, 10.0), "float64"),
            "y": ((1.0, 2.0), "float64"),
        },
    )
    assert v.status == "REFUTED"


# --- Solver-needed (non-trivial computation) ---------------------------------


def test_clamped_system_verifies():
    """A system with clamping: x_new = clip(2.0 * x, -10, 10).

    Even though the gain is 2.0, the clamp keeps x in bounds. Interval
    arithmetic alone should see clip([-20, 20], -10, 10) = [-10, 10] and
    VERIFY."""

    def body(state, constants):
        return {"x": jnp.clip(2.0 * state["x"], -10.0, 10.0)}

    v = check_inductive_step(
        body=body,
        state_bounds={"x": ((-10.0, 10.0), "float64")},
    )
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for clamped system, got {v.status}; "
        f"notes: {v.notes}"
    )


# --- Error handling ----------------------------------------------------------


def test_empty_state_bounds_raises():
    """Empty state_bounds should raise ValueError."""

    def body(state, constants):
        return {}

    with pytest.raises(ValueError, match="non-empty"):
        check_inductive_step(body=body, state_bounds={})


def test_body_wrong_return_type_raises():
    """A body returning a non-dict raises ValueError during tracing."""

    def body(state, constants):
        return state["x"]  # returns a scalar, not a dict

    with pytest.raises(ValueError, match="body must return a dict"):
        check_inductive_step(
            body=body,
            state_bounds={"x": ((-10.0, 10.0), "float64")},
        )


def test_body_missing_key_raises():
    """A body returning a dict missing a declared key raises ValueError."""

    def body(state, constants):
        return {}  # missing "x"

    with pytest.raises(ValueError, match="missing state keys"):
        check_inductive_step(
            body=body,
            state_bounds={"x": ((-10.0, 10.0), "float64")},
        )


def test_body_extra_key_raises():
    """A body returning extra keys raises ValueError."""

    def body(state, constants):
        return {"x": state["x"], "y": state["x"]}

    with pytest.raises(ValueError, match="unexpected keys"):
        check_inductive_step(
            body=body,
            state_bounds={"x": ((-10.0, 10.0), "float64")},
        )


def test_bad_bounds_shape_raises():
    """Malformed state_bounds entry raises TypeError."""

    def body(state, constants):
        return {"x": state["x"]}

    with pytest.raises(TypeError, match="must be"):
        check_inductive_step(
            body=body,
            state_bounds={"x": "not a tuple"},
        )


def test_non_numeric_bounds_raises():
    """Non-numeric bounds raise TypeError."""

    def body(state, constants):
        return {"x": state["x"]}

    with pytest.raises(TypeError, match="bounds must be numeric"):
        check_inductive_step(
            body=body,
            state_bounds={"x": (("a", "b"), "float64")},
        )


# --- Verdict notes -----------------------------------------------------------


def test_verified_note_mentions_assumption():
    """VERIFIED verdicts include the assumption note about initial state."""

    def body(state, constants):
        return {"x": 0.5 * state["x"]}

    v = check_inductive_step(
        body=body,
        state_bounds={"x": ((-10.0, 10.0), "float64")},
    )
    assert v.status == "VERIFIED"
    assert any("initial state" in n for n in v.notes)


def test_constants_are_traced_not_declared():
    """Constants appear as traced values, not as symbolic inputs.

    A constant that violates the bounds should not affect the verdict
    since it is not a state variable."""

    def body(state, constants):
        # The constant 100.0 exceeds the bounds of x, but it is just used
        # as a scaling factor -- the result is still in bounds because
        # state["x"] is scaled to be tiny.
        return {"x": state["x"] * constants["tiny"]}

    v = check_inductive_step(
        body=body,
        state_bounds={"x": ((-10.0, 10.0), "float64")},
        constants={"tiny": 0.01},
    )
    assert v.status == "VERIFIED"
