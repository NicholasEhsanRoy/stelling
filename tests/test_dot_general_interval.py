# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Containment and gate tests for the `dot_general` interval face.

Face verified STANDALONE, before registration: the transfer is written but
`dot_general` is not yet in TRANSFERS, so nothing here goes through the
propagation pipeline. That ordering is deliberate — a half-registered row
trips a census raise at import and breaks the package for everyone.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.lax import dot_general_p

from stelling.interval import IntervalArray, IntervalError
from stelling.interval import dot_general as iv_dot
from stelling.propagate import _dot_general_row_form


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


CONFIGS = [
    ((3, 4), (4,), (((1,), (0,)), ((), ())), "matvec"),
    ((3, 4), (4, 5), (((1,), (0,)), ((), ())), "matmul"),
    ((3, 5), (5,), (((1,), (0,)), ((), ())), "contract#4 shape"),
    ((2, 3, 4), (3, 4), (((1, 2), (0, 1)), ((), ())), "two-contractions"),
    ((4, 3, 5), (4, 5, 2), (((2,), (1,)), ((0,), (0,))), "batched"),
    ((3,), (4,), (((), ()), ((), ())), "zero-contraction (outer product)"),
    ((8, 4, 19), (19, 3), (((2,), (0,)), ((), ())), "lbm-shaped"),
]


@pytest.mark.parametrize("lshape,rshape,dn,label", CONFIGS,
                         ids=[c[3] for c in CONFIGS])
def test_true_value_is_inside_the_box(lshape, rshape, dn, label):
    """THE SOUNDNESS PROPERTY, and it needs no solver to check.

    Draw a box, contract it, then execute jax at concrete points inside that
    box: every output element must land within the interval. This is the
    plain-execution check, so it holds whether or not anything else in
    stelling is sound.
    """
    rng = np.random.default_rng(20260726)
    for _ in range(12):
        alo = rng.uniform(-3, 3, lshape)
        awid = rng.uniform(0, 2, lshape)
        blo = rng.uniform(-3, 3, rshape)
        bwid = rng.uniform(0, 2, rshape)
        box = iv_dot(
            IntervalArray(shape=lshape, los=tuple(alo.ravel()),
                          his=tuple((alo + awid).ravel())),
            IntervalArray(shape=rshape, los=tuple(blo.ravel()),
                          his=tuple((blo + bwid).ravel())),
            dn,
        )
        for _ in range(6):
            av = alo + rng.uniform(0, 1, lshape) * awid
            bv = blo + rng.uniform(0, 1, rshape) * bwid
            got = np.asarray(dot_general_p.bind(
                jnp.asarray(av), jnp.asarray(bv), dimension_numbers=dn,
                precision=None,
                preferred_element_type=jnp.dtype(jnp.float64),
                out_sharding=None,
            ))
            assert got.shape == box.shape, f"{label}: shape disagrees with jax"
            for k, v in enumerate(got.ravel()):
                assert box.los[k] <= v <= box.his[k], (
                    f"{label}: element {k} executes to {v}, outside "
                    f"[{box.los[k]}, {box.his[k]}]"
                )


def test_single_occurrence_holds_so_the_box_is_exact_on_a_point():
    """A degenerate box (lo == hi) must produce a POINT, not a widened
    bracket.

    This is the observable consequence of the single-occurrence property: no
    operand element appears twice in any one output element's sum, so there
    is no dependency problem to inflate the result. If an operand were
    double-counted, a point input would still yield a spread.
    """
    rng = np.random.default_rng(11)
    a = rng.uniform(-2, 2, (3, 4))
    b = rng.uniform(-2, 2, (4,))
    dn = (((1,), (0,)), ((), ()))
    box = iv_dot(
        IntervalArray(shape=(3, 4), los=tuple(a.ravel()), his=tuple(a.ravel())),
        IntervalArray(shape=(4,), los=tuple(b.ravel()), his=tuple(b.ravel())),
        dn,
    )
    truth = np.asarray(dot_general_p.bind(
        jnp.asarray(a), jnp.asarray(b), dimension_numbers=dn, precision=None,
        preferred_element_type=jnp.dtype(jnp.float64), out_sharding=None))
    for k, v in enumerate(truth.ravel()):
        assert box.los[k] <= v <= box.his[k]
        # a 4-term sum earns 3 outward bumps; anything wider means a term
        # was counted more than once
        assert box.his[k] - box.los[k] < 1e-12 * max(abs(v), 1.0), (
            f"element {k} spread {box.his[k] - box.los[k]} on a POINT input"
        )


def test_empty_contraction_is_the_outer_product():
    """No contracted axes: each output element is a single product, so no
    accumulation happens and the outer product is what jax computes.

    Asserted as containment plus one-ulp tightness rather than as exact
    equality. Each corner product carries the same unconditional outward
    bump :func:`stelling.interval.mul` applies, so `1.0 * 1.0` brackets as
    [0.9999999999999999, 1.0000000000000002]. That is the established
    convention for a product in this domain, and matching it is the point —
    a dot_general that came out exact here would be using a different
    rounding rule from `mul` for the same operation.
    """
    box = iv_dot(
        IntervalArray(shape=(2,), los=(1.0, 2.0), his=(1.0, 2.0)),
        IntervalArray(shape=(3,), los=(1.0, 2.0, 3.0), his=(1.0, 2.0, 3.0)),
        (((), ()), ((), ())),
    )
    assert box.shape == (2, 3)
    for k, want in enumerate((1.0, 2.0, 3.0, 2.0, 4.0, 6.0)):
        assert box.los[k] <= want <= box.his[k]
        assert box.his[k] - box.los[k] <= 4 * np.spacing(want), (
            f"element {k} spread {box.his[k] - box.los[k]} on a single product"
        )


def test_mismatched_contracted_dims_raise():
    with pytest.raises(IntervalError, match="contracted dims disagree"):
        iv_dot(
            IntervalArray(shape=(3, 4), los=(0.0,) * 12, his=(1.0,) * 12),
            IntervalArray(shape=(5,), los=(0.0,) * 5, his=(1.0,) * 5),
            (((1,), (0,)), ((), ())),
        )


# -- the shared oracle. Each decline names the magnitude motivating it. ------

def _p(**over):
    base = dict(dimension_numbers=(((1,), (0,)), ((), ())), precision=None,
                preferred_element_type=jnp.dtype(jnp.float64),
                out_sharding=None)
    base.update(over)
    return base


ORACLE_CASES = [
    (_p(), "float64", "float64", True, "baseline"),
    (_p(dimension_numbers=(((2,), (1,)), ((0,), (0,)))), "float64", "float64",
     True, "batch dims are BENIGN, measured — admitting them is not laxity"),
    (_p(), "int32", "int32", True, "int operands, float64 accumulation: 5.9e-16"),
    (_p(preferred_element_type=jnp.dtype(jnp.float32)), "float64", "float64",
     False, "accumulation narrower than operands: 3.9e-08"),
    (_p(preferred_element_type=jnp.dtype(jnp.int32)), "int32", "int32",
     False, "integer accumulation wraps: 1.0"),
    (_p(preferred_element_type=None), "int32", "int32", False,
     "ABSENT pet over integer operands still accumulates in int32"),
    (_p(preferred_element_type=jnp.dtype(jnp.float32)), "int32", "int32",
     False, "int operands in float32: 2.0e-08"),
    (_p(preferred_element_type=jnp.dtype(jnp.complex64)), "complex64",
     "complex64", False, "complex is outside the DOMAIN"),
    (_p(precision="weird"), "float64", "float64", False,
     "unrecognised precision"),
    (_p(out_sharding="mesh"), "float64", "float64", False, "out_sharding set"),
]


@pytest.mark.parametrize("params,ld,rd,admit,label", ORACLE_CASES,
                         ids=[c[4][:40] for c in ORACLE_CASES])
def test_oracle_admits_exactly_the_measured_benign_forms(
    params, ld, rd, admit, label
):
    dn, reason = _dot_general_row_form(params, ld, rd)
    if admit:
        assert dn is not None, f"{label}: declined, but measured benign"
        assert reason is None
    else:
        assert dn is None, f"{label}: admitted, but measured divergent"
        assert reason, "a decline must carry its reason"


def test_absent_pet_over_float_operands_is_still_admitted():
    """Anti-vacuity for the absent-pet gate: it must decline on the INTEGER
    operands, not on the absence itself, or the gate passes by closing the
    row."""
    dn, reason = _dot_general_row_form(
        _p(preferred_element_type=None), "float64", "float64")
    assert dn is not None, f"absent pet over float operands declined: {reason}"
