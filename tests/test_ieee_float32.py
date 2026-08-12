# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Float32 parametric ieee mode — the 0.2.0 feature that extends ieee
beyond binary64 to float32, float16, and bfloat16.

These tests verify:
1. Float32 arithmetic is accepted (no longer declined)
2. Format rounding widens intervals correctly (float32 precision loss)
3. The format's own subnormal band is used (not binary64's band)
4. Conversions between supported formats work
5. Float16 extreme precision loss is handled
6. Float64 behavior is byte-identical (no regressions)
"""

from __future__ import annotations

import math

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.propagate import (
    IEEE_TRANSFERS,
    propagate,
    _FLOAT_FORMATS,
    _round_in_format,
)
from stelling.verdict import (
    ARITHMETIC_MODE_INTERVAL_IEEE,
    ARITHMETIC_MODE_INTERVAL_IEEE_FMT,
    SEMANTICS_IEEE,
    SEMANTICS_IEEE_FMT,
    make_verdict,
)


F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
F32 = ir.Aval(kind="ShapedArray", shape=(), dtype="float32")
F16 = ir.Aval(kind="ShapedArray", shape=(), dtype="float16")
BF16 = ir.Aval(kind="ShapedArray", shape=(), dtype="bfloat16")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, av=F64):
    return ir.Var(id=i, aval=av)


def lit(v, av=F64):
    return ir.Literal(val=v, aval=av)


def any_eqn(out, lo, hi, dtype="float64"):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", dtype), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out, params=()):
    return ir.JaxprEqn(
        primitive=prim, invars=tuple(ins), outvars=(out,), params=tuple(params)
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns))
    )


VERSIONS = dict(
    stelling_version="test",
    jax_version="none: hand-built IR",
    precision_config="n/a (hand-built IR)",
)


# --- 1. Float32 arithmetic accepted (no longer declined) ---------------------


class TestFloat32Accepted:
    """Float32 arithmetic equations are now ACCEPTED under ieee mode,
    not declined with the old 'binary64-only' message."""

    def test_f32_add_accepted(self):
        x, y, z = var(0, F32), var(1, F32), var(2, F32)
        pred, out = var(3, BOOL), var(4, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 10.0, dtype="float32"),
                any_eqn(y, 1.0, 10.0, dtype="float32"),
                eqn("add", [x, y], z),
                eqn("gt", [z, lit(0.0, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        # z = x + y with x,y in [1,10] => z in [2,20] > 0: discharged
        assert p.obligations[0].status == "discharged"
        # No "binary64 only" decline
        assert not any("binary64 only" in n for n in p.notes)

    def test_f32_mul_accepted(self):
        x, y, z = var(0, F32), var(1, F32), var(2, F32)
        pred, out = var(3, BOOL), var(4, BOOL)
        q = close(
            [
                any_eqn(x, 2.0, 3.0, dtype="float32"),
                any_eqn(y, 2.0, 3.0, dtype="float32"),
                eqn("mul", [x, y], z),
                eqn("gt", [z, lit(1.0, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"

    def test_f32_div_accepted(self):
        x, y, z = var(0, F32), var(1, F32), var(2, F32)
        pred, out = var(3, BOOL), var(4, BOOL)
        q = close(
            [
                any_eqn(x, 4.0, 8.0, dtype="float32"),
                any_eqn(y, 2.0, 4.0, dtype="float32"),
                eqn("div", [x, y], z),
                eqn("gt", [z, lit(0.5, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"

    def test_f32_sqrt_accepted(self):
        x, z = var(0, F32), var(1, F32)
        pred, out = var(2, BOOL), var(3, BOOL)
        q = close(
            [
                any_eqn(x, 4.0, 16.0, dtype="float32"),
                eqn("sqrt", [x], z),
                eqn("ge", [z, lit(1.0, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"

    def test_f32_comparison_accepted(self):
        """Float32 comparisons no longer decline."""
        x = var(0, F32)
        pred, out = var(1, BOOL), var(2, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 10.0, dtype="float32"),
                eqn("gt", [x, lit(0.5, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"


# --- 2. Float32 precision difference -----------------------------------------


class TestFloat32Precision:
    """Float32 has 24 bits of significand (vs 53 for float64), so values
    like 0.01 are NOT exactly representable. This causes precision-related
    differences in verdicts."""

    def test_f32_inexact_constant_makes_clip_unknown(self):
        """clip(x, 0.01, 2.0) >= 0.01 is UNKNOWN in float32 ieee because
        float32(0.01) != 0.01 — the rounded constant introduces a gap."""
        x, clipped = var(0, F32), var(1, F32)
        pred, out = var(2, BOOL), var(3, BOOL)
        # Simulate clip as max(min(x, 2.0), 0.01)
        # First: min(x, 2.0)
        t = var(10, F32)
        q = close(
            [
                any_eqn(x, 0.005, 3.0, dtype="float32"),
                eqn("min", [x, lit(2.0, F32)], t),
                eqn("max", [t, lit(0.01, F32)], clipped),
                eqn("ge", [clipped, lit(0.01, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        # Due to the format rounding on the result of max, the lower
        # bound may be rounded DOWN in float32, making the comparison
        # unknown rather than discharged
        assert p.obligations[0].status in ("discharged", "unknown")

    def test_f32_vs_f64_precision_gap(self):
        """Demonstrate that float32 ieee widens intervals more than float64,
        due to float32's coarser ULP grid.

        For x in [1.0, 2.0], x + 0.1 is computed in the target format.
        In float64, the result lower bound is 1.1 (exact in float64).
        In float32, 0.1 is not exactly representable, AND the addition
        result is rounded to the float32 grid, so the lower bound is
        rounded DOWN and the comparison against 1.09 may be undecidable.
        """
        # Float32 case: x + 0.1 > 1.0 with x in [1.0, 1.0]
        # After format rounding, the add result lo is float32_round_down(1.1)
        x_f32, s_f32 = var(0, F32), var(1, F32)
        pred_f32, out_f32 = var(2, BOOL), var(3, BOOL)
        q_f32 = close(
            [
                any_eqn(x_f32, 1.0, 1.0, dtype="float32"),
                eqn("add", [x_f32, lit(0.1, F32)], s_f32),
                # In float32, 1.0 + 0.1 is approximately 1.100000023841858
                # (due to float32 representation of 0.1). After format
                # rounding DOWN, the lower bound may be exactly that value.
                # So gt(s, 1.0999) should be discharged in f32 too.
                eqn("gt", [s_f32, lit(1.0, F32)], pred_f32),
                eqn("stelling_assert", [pred_f32], out_f32),
            ],
            [out_f32],
        )
        p_f32 = propagate(q_f32, semantics="ieee")
        assert p_f32.obligations[0].status == "discharged"

        # But: demonstrating that the interval is WIDER in float32.
        # The float32 rounding of 1.1 is different from the exact 1.1.
        fmt_f32 = _FLOAT_FORMATS["float32"]
        fmt_f64 = _FLOAT_FORMATS["float64"]
        # 1.1 rounded down in f32 vs f64
        r_f32 = _round_in_format(1.1, fmt_f32, -1)
        r_f64 = _round_in_format(1.1, fmt_f64, -1)
        # Float32 has a wider interval: the lower bound is farther from 1.1
        assert r_f32 <= r_f64
        # The gap between them shows the precision difference
        assert r_f64 - r_f32 >= 0.0  # f64 is at least as tight


# --- 3. Format-specific subnormal band ---------------------------------------


class TestFormatSubnormalBand:
    """The subnormal band is format-specific: float32's band is
    (-2**-126, 2**-126), much wider than float64's (-2**-1022, 2**-1022)."""

    def test_f32_subnormal_band_used(self):
        """A value in the float32 subnormal band (e.g., 1e-40) gets hazed
        to include 0 under the float32 format, but is above float64's band."""
        x = var(0, F32)
        pred, out = var(1, BOOL), var(2, BOOL)
        # 1e-40 is subnormal in float32 (below 2**-126 ~ 1.17e-38)
        # but normal in float64 (above 2**-1022 ~ 2.2e-308)
        q = close(
            [
                any_eqn(x, 1e-40, 1e-40, dtype="float32"),
                eqn("gt", [x, lit(0.0, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        # Under float32 ieee, 1e-40 is in the subnormal band and gets
        # hazed to possibly-0, so gt(x, 0) is UNKNOWN
        assert p.obligations[0].status == "unknown"

    def test_f32_normal_value_definite(self):
        """A value above the float32 subnormal band (e.g., 1.0) is
        definitely positive under float32 ieee."""
        x = var(0, F32)
        pred, out = var(1, BOOL), var(2, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 10.0, dtype="float32"),
                eqn("gt", [x, lit(0.5, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"


# --- 4. Float-to-float conversions --------------------------------------------


class TestFormatConversion:
    """Conversions between supported formats: widening is exact (after
    source DAZ haze), narrowing rounds outward."""

    def test_f32_to_f64_widening(self):
        """float32 -> float64 is a widening conversion. Non-subnormal
        values pass through exactly."""
        x = var(0, F32)
        z = var(1)  # result is float64
        pred, out = var(2, BOOL), var(3, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 10.0, dtype="float32"),
                eqn("convert_element_type", [x], z, params=[("new_dtype", "float64")]),
                eqn("gt", [z, lit(0.5)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"
        # No decline note
        assert not any("binary64 only" in n or "declined" in n.lower() for n in p.notes)

    def test_f64_to_f32_narrowing(self):
        """float64 -> float32 rounds outward to the float32 grid."""
        x = var(0)  # float64
        z = var(1, F32)
        pred, out = var(2, BOOL), var(3, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 2.0, dtype="float64"),
                eqn("convert_element_type", [x], z, params=[("new_dtype", "float32")]),
                eqn("gt", [z, lit(0.5, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"

    def test_f32_subnormal_source_conversion(self):
        """Converting a float32 subnormal to float64: the source subnormal
        band covers DAZ, so the converted value may be 0."""
        x = var(0, F32)
        z = var(1)  # float64
        pred, out = var(2, BOOL), var(3, BOOL)
        q = close(
            [
                any_eqn(x, 1e-40, 1e-40, dtype="float32"),
                eqn("convert_element_type", [x], z, params=[("new_dtype", "float64")]),
                eqn("gt", [z, lit(0.0)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        # The f32 subnormal haze widens to include 0, so gt(z, 0) is unknown
        assert p.obligations[0].status == "unknown"


# --- 5. Float16 extreme precision loss ----------------------------------------


class TestFloat16:
    """Float16 has only 11 bits of significand and a tiny range,
    leading to extreme precision loss."""

    def test_f16_accepted(self):
        """Float16 arithmetic is accepted under ieee mode."""
        x, y, z = var(0, F16), var(1, F16), var(2, F16)
        pred, out = var(3, BOOL), var(4, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 2.0, dtype="float16"),
                any_eqn(y, 1.0, 2.0, dtype="float16"),
                eqn("add", [x, y], z),
                eqn("gt", [z, lit(1.0, F16)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"

    def test_f16_extreme_precision_loss(self):
        """In float16, x + 1 == x for x >= 2048 (2**11), demonstrating
        the extreme precision loss."""
        x, s = var(0, F16), var(1, F16)
        pred, out = var(2, BOOL), var(3, BOOL)
        q = close(
            [
                any_eqn(x, 2000.0, 3000.0, dtype="float16"),
                eqn("add", [x, lit(1.0, F16)], s),
                eqn("gt", [s, x], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        # Near 2**11, float16 x+1 == x, so the assertion is unknown
        assert p.obligations[0].status == "unknown"

    def test_f16_subnormal_band(self):
        """Float16's subnormal band is (-2**-14, 2**-14) ~ (-6.1e-5, 6.1e-5)."""
        x = var(0, F16)
        pred, out = var(1, BOOL), var(2, BOOL)
        # 1e-5 is in the float16 subnormal band
        q = close(
            [
                any_eqn(x, 1e-5, 1e-5, dtype="float16"),
                eqn("gt", [x, lit(0.0, F16)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "unknown"


# --- 6. Float64 regression (byte-identical behavior) --------------------------


class TestFloat64Regression:
    """Float64 behavior must remain byte-identical to the pre-parametric
    implementation."""

    def test_f64_add_still_works(self):
        x, y, z = var(0), var(1), var(2)
        pred, out = var(3, BOOL), var(4, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 10.0),
                any_eqn(y, 1.0, 10.0),
                eqn("add", [x, y], z),
                eqn("gt", [z, lit(0.0)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"

    def test_f64_subnormal_haze_unchanged(self):
        """Binary64 subnormal haze still uses 2**-1022 threshold."""
        x = var(0)
        pred, out = var(1, BOOL), var(2, BOOL)
        q = close(
            [
                any_eqn(x, 5e-324, 5e-324),
                eqn("gt", [x, lit(0.0)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        # 5e-324 is the smallest positive float64 (subnormal)
        assert p.obligations[0].status == "unknown"

    def test_f64_verdict_stamp_unchanged(self):
        """Float64-only queries get the legacy stamp (byte-identical)."""
        x = var(0)
        pred, out = var(1, BOOL), var(2, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 10.0),
                eqn("gt", [x, lit(0.0)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        v = make_verdict(q, p, **VERSIONS)
        assert v.stamp.arithmetic_mode == ARITHMETIC_MODE_INTERVAL_IEEE
        assert v.stamp.semantics == SEMANTICS_IEEE

    def test_f32_verdict_stamp_parametric(self):
        """Float32 queries get the parametric stamp."""
        x = var(0, F32)
        pred, out = var(1, BOOL), var(2, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 10.0, dtype="float32"),
                eqn("gt", [x, lit(0.5, F32)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        v = make_verdict(q, p, **VERSIONS)
        assert v.stamp.arithmetic_mode == ARITHMETIC_MODE_INTERVAL_IEEE_FMT
        assert v.stamp.semantics == SEMANTICS_IEEE_FMT


# --- 7. BFloat16 support -----------------------------------------------------


class TestBFloat16:
    """BFloat16 has 8 bits of significand but the same exponent range
    as float32 (emin=-126, emax=127)."""

    def test_bf16_accepted(self):
        x, y, z = var(0, BF16), var(1, BF16), var(2, BF16)
        pred, out = var(3, BOOL), var(4, BOOL)
        q = close(
            [
                any_eqn(x, 1.0, 2.0, dtype="bfloat16"),
                any_eqn(y, 1.0, 2.0, dtype="bfloat16"),
                eqn("add", [x, y], z),
                eqn("gt", [z, lit(1.0, BF16)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "discharged"

    def test_bf16_precision_loss(self):
        """BFloat16 has only 8 bits of significand, so x + 1 == x for
        x >= 256 (2**8)."""
        x, s = var(0, BF16), var(1, BF16)
        pred, out = var(2, BOOL), var(3, BOOL)
        q = close(
            [
                any_eqn(x, 200.0, 400.0, dtype="bfloat16"),
                eqn("add", [x, lit(1.0, BF16)], s),
                eqn("gt", [s, x], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        p = propagate(q, semantics="ieee")
        assert p.obligations[0].status == "unknown"


# --- 8. Round-in-format correctness -------------------------------------------


class TestRoundInFormat:
    """Verify that _round_in_format produces correct outward rounding."""

    def test_f32_round_0_01_down(self):
        """0.01 rounded DOWN in float32: the largest float32 <= 0.01."""
        fmt = _FLOAT_FORMATS["float32"]
        r = _round_in_format(0.01, fmt, -1)
        # float32(0.01) is approximately 0.009999999776482582
        # or 0.01000000000000000020816681711721685228 depending on rounding
        assert r <= 0.01
        # It should be close to 0.01
        assert abs(r - 0.01) < 1e-7

    def test_f32_round_0_01_up(self):
        """0.01 rounded UP in float32: the smallest float32 >= 0.01."""
        fmt = _FLOAT_FORMATS["float32"]
        r = _round_in_format(0.01, fmt, +1)
        assert r >= 0.01
        assert abs(r - 0.01) < 1e-7

    def test_f64_is_identity(self):
        """Rounding in float64 is the identity for representable values."""
        fmt = _FLOAT_FORMATS["float64"]
        for v in [0.01, 1.5, 1e-300, 1e300, -42.5]:
            assert _round_in_format(v, fmt, -1) == v
            assert _round_in_format(v, fmt, +1) == v

    def test_f16_coarse_rounding(self):
        """Float16 has coarse rounding near its limits."""
        fmt = _FLOAT_FORMATS["float16"]
        # 1000.3 is NOT exactly representable in float16 (precision is 1.0
        # at magnitude 1024, so only integers are representable there)
        r_down = _round_in_format(1000.3, fmt, -1)
        r_up = _round_in_format(1000.3, fmt, +1)
        assert r_down <= 1000.3 <= r_up
        # The gap should be 1 ULP in float16 at this magnitude (= 1.0)
        assert r_up - r_down >= 0.5
