# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The assumed-precondition templates: both faces of each, plus the
module's jax-free import claim.

Every template test states both faces (a VERIFY face and a
REFUTE/UNKNOWN face) so a template that trivially passes or trivially
fails cannot look correct — the L2 discipline applied to test design.
"""

from __future__ import annotations

import pytest

from stelling import preconditions  # jax-free import must always work


def test_module_imports_without_jax():
    assert preconditions.__all__ == ["check", "field_positive", "scalar_nonzero"]


jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True, scope="module")
def _x64():
    # module-scoped and restored: setting the flag at import time leaks
    # float64 into every later-run module in the same process (measured —
    # it flipped test_transcribe's cross-process hash-stability test)
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


import jax.numpy as jnp  # noqa: E402

import stelling  # noqa: E402
from stelling._jax_compat import jax_version, trace  # noqa: E402
from stelling.preconditions import field_positive, scalar_nonzero  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.verdict import make_verdict  # noqa: E402

VER = dict(
    stelling_version="test",
    jax_version="test",
    precision_config="jax_enable_x64=True",
)


def verdict(h):
    cj = trace(h)
    return make_verdict(cj, propagate(cj), **VER)


# --- field_positive ----------------------------------------------------------


def test_field_positive_verify_face():
    """Coefficient 1+x over a strictly positive envelope: VERIFIED."""

    def h():
        _, o = field_positive((), "float64", (1e-6, 1e2),
                              transform=lambda x: 1.0 + x)
        return (o,)

    assert verdict(h).status == "VERIFIED"


def test_field_positive_refute_face():
    """The same coefficient over a sign-spanning envelope that reaches
    1+x <= 0: definitely false at the low corner region -> not VERIFIED,
    and the interval straddles so the honest interval verdict is UNKNOWN."""

    def h():
        _, o = field_positive((), "float64", (-2.0, 1e2),
                              transform=lambda x: 1.0 + x)
        return (o,)

    assert verdict(h).status == "UNKNOWN"


def test_field_positive_definite_refute():
    """An envelope wholly past the singularity is definitely false."""

    def h():
        _, o = field_positive((), "float64", (-3.0, -2.0),
                              transform=lambda x: 1.0 + x)
        return (o,)

    assert verdict(h).status == "REFUTED"


def test_field_positive_elementwise_array():
    """The pointwise reading: an array field is judged at every point."""

    def h():
        f, o = field_positive((3,), "float64", (0.5, 2.0))
        return (o,)

    assert verdict(h).status == "VERIFIED"


def test_field_positive_default_transform_and_bound():
    def h():
        _, o = field_positive((), "float64", (-1.0, 1.0), bound=-2.0)
        return (o,)

    assert verdict(h).status == "VERIFIED"


# --- scalar_nonzero ----------------------------------------------------------


def test_scalar_nonzero_verify_face():
    """A point default away from zero: VERIFIED."""

    def h():
        _, o = scalar_nonzero("float64", (1.0, 1.0))
        return (o,)

    assert verdict(h).status == "VERIFIED"


def test_scalar_nonzero_range_admitting_zero_is_unknown_interval_only():
    """A config range containing 0 and non-0: ne straddles -> UNKNOWN
    (the escalated REFUTED-with-witness face is exercised in the
    instance harness, where a solver names the witness)."""

    def h():
        _, o = scalar_nonzero("float64", (0.0, 1.0))
        return (o,)

    assert verdict(h).status == "UNKNOWN"


def test_scalar_nonzero_zero_point_refutes():
    """The singular configuration itself: definitely false."""

    def h():
        _, o = scalar_nonzero("float64", (0.0, 0.0))
        return (o,)

    assert verdict(h).status == "REFUTED"


def test_templates_cover_negative_axis():
    """nonzero is two-sided: a strictly negative range verifies too."""

    def h():
        _, o = scalar_nonzero("float64", (-2.0, -1.0))
        return (o,)

    assert verdict(h).status == "VERIFIED"


# --- the multi-value transform (the magnetics template gap, closed) ----------


def test_field_positive_multi_value_transform():
    """A transform returning a tuple states one obligation per value."""

    def h():
        _, obs = field_positive((), "float64", (0.5, 2.0),
                                transform=lambda x: (x + 1.0, x * 2.0))
        assert isinstance(obs, tuple) and len(obs) == 2
        return obs

    assert verdict(h).status == "VERIFIED"


def test_field_positive_multi_value_one_bad_face_refutes():
    """Each produced value carries its own obligation: one definitely
    non-positive value refutes even when the sibling verifies."""

    def h():
        _, obs = field_positive((), "float64", (0.5, 2.0),
                                transform=lambda x: (x + 1.0, -x))
        return obs

    assert verdict(h).status == "REFUTED"


def test_field_positive_poses_the_magnetics_face_shape_without_hand_adaptation():
    """The §1a acceptance: the magnetics SPD check's real shape — one cell
    field, the code's default a = θ path, conservative face averaging via
    roll producing TWO face arrays — poses through the template directly.
    This exact shape needed a hand-written harness before the
    generalization (design/precondition-class.md, §6)."""
    inv_h2 = 1.0 / 0.25**2

    def faces(theta):
        a = theta                                   # _coeff_field default path
        a_plus = 0.5 * (a + jnp.roll(a, -1)) * inv_h2
        a_minus = 0.5 * (a + jnp.roll(a, 1)) * inv_h2
        return a_plus, a_minus

    def h_supported():
        _, obs = field_positive((4,), "float64", (1e-6, 1e2), transform=faces)
        return obs

    def h_sign_spanning():
        _, obs = field_positive((4,), "float64", (-2.0, 1e2), transform=faces)
        return obs

    assert verdict(h_supported).status == "VERIFIED"
    assert verdict(h_sign_spanning).status == "UNKNOWN"  # honest straddle


# --- check(), the front door -------------------------------------------------


def test_check_interval_only_fills_stamps_and_records_absence():
    from stelling.preconditions import check
    from stelling.verdict import SolverStamp

    def h():
        _, o = field_positive((), "float64", (1e-6, 1e2),
                              transform=lambda x: 1.0 + x)
        return (o,)

    v = check(h, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"
    assert v.stamp.stelling_version == stelling.__version__
    assert v.stamp.precision_config == "jax_enable_x64=True"
    assert isinstance(v.stamp.solver, SolverStamp) and not v.stamp.solver.invoked


def test_check_escalates_only_with_an_explicit_timeout():
    from stelling._optional import available
    from stelling.preconditions import check

    def h():
        _, o = scalar_nonzero("float64", (0.0, 1.0))
        return (o,)

    v = check(h, vacuity_mode="inputs-only")  # no timeout: interval-only, honest UNKNOWN
    assert v.status == "UNKNOWN"
    if not (available("z3") or available("cvc5")):
        pytest.skip("no solver installed")
    v2 = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert v2.status == "REFUTED"
    assert v2.witnesses and dict(v2.witnesses[0].values)["x0"] == "0"


# --- the vacuity wiring (check() cannot return an unchecked VERIFIED) --------


def test_check_requires_vacuity_mode():
    from stelling.preconditions import check

    def h():
        _, o = scalar_nonzero("float64", (1.0, 1.0))
        return (o,)

    with pytest.raises(TypeError):
        check(h)  # no silent mode, no silent skip


def test_check_flags_envelope_not_load_bearing_verified():
    """A range-theorem obligation (|x| >= bound below zero) discharges with
    the bounds gone: the entry point itself must say so — a note per
    obligation and a stamped vacuity line."""
    from stelling.preconditions import check

    def h():
        _, o = field_positive((3,), "float64", (-10.0, 10.0),
                              transform=lambda d: jnp.abs(d), bound=-1e-300)
        return (o,)

    v = check(h, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"
    assert any("envelope not load-bearing" in n for n in v.notes)
    assert any("vacuity checked (mode=inputs-only)" in a
               for a in v.stamp.assumptions)


def test_check_stamps_load_bearing_when_bounds_matter():
    from stelling.preconditions import check

    def h():
        _, o = scalar_nonzero("float64", (1.0, 1024.0))
        return (o,)

    v = check(h, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"
    assert not any("envelope not load-bearing" in n for n in v.notes)
    assert any("envelope is load-bearing" in a for a in v.stamp.assumptions)


def test_check_non_verified_returns_unchanged():
    """Widening cannot rescue UNKNOWN/REFUTED; no vacuity line is added."""
    from stelling.preconditions import check

    def h():
        _, o = scalar_nonzero("float64", (0.0, 0.0))
        return (o,)

    v = check(h, vacuity_mode="inputs-only")
    assert v.status == "REFUTED"
    assert not any("vacuity checked" in a for a in v.stamp.assumptions)
