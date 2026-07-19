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
    assert preconditions.__all__ == ["field_positive", "scalar_nonzero"]


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
