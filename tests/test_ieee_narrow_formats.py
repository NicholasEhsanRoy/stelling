# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The four narrow-format repairs of audit 0.2.0 — M12, M13, M14, M15.

Each is a different way the format-parametric ieee mode was still speaking
binary64:

* **M12** — no literal decoder for float16 (`<f2`) or bfloat16 (`<V2`), so
  any harness in those formats that mentions a scalar constant topped out.
* **M13** — the comparison band was picked ALPHABETICALLY among the
  operands' formats, which is the narrowest one for anything paired with
  bfloat16.
* **M14** — two binary64 assumption sentences stamped verbatim on
  narrow-format verdicts, where both are false.
* **M15** — `_ieee_arith`'s fallback would have hazed a narrow format with
  the binary64 subnormal band, and no later outward rounding can put the
  missing hull back.

The M13/M15 rows run on hand-built IR, which is the only way to reach
them: jax promotes before it computes, so a traced harness cannot present
a mixed equation, and all four registered ieee binary kernels have a
format-parametric row. Hand-built and deserialized IR is in scope by this
codebase's own rules (`_ieee_select_n`: "hand-built/deserialized IR
arrives here unchecked").
"""

import struct

import pytest

import stelling.interval as iv
from stelling import ir
from stelling.propagate import (
    _FLOAT_FORMATS,
    _FMT_BINARY_OPS,
    _assert_ieee_binary_kernels_are_format_parametric,
    _decode_array,
    _ieee_arith,
    _ieee_cmp_get_min_normal,
    _ieee_format_min_normal,
    _ieee_round_box,
    propagate,
)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402

# x64 is process-global in jax and an unrestored set leaks into every test
# that runs after this module — invisible to anyone running with
# JAX_ENABLE_X64=1, which CI does not set.
@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


FOUR = ("float16", "bfloat16", "float32", "float64")


def _aval(dtype, shape=()):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype=dtype)


class _FakeEqn:
    """The smallest thing the format helpers read: invars/outvars with
    avals. Hand-built IR reaches them exactly this way."""

    def __init__(self, in_dtypes, out_dtypes=("bool",)):
        self.primitive = "le"
        self.invars = tuple(ir.Var(id=i, aval=_aval(d))
                            for i, d in enumerate(in_dtypes))
        self.outvars = tuple(ir.Var(id=100 + i, aval=_aval(d))
                             for i, d in enumerate(out_dtypes))


# -- M12: the two narrow formats can see a constant ---------------------------


def _const_harness(dtype):
    def h():
        x = any_array((), dtype, (1.0, 2.0))
        y = x * 2.0 + 1.0
        return assert_(y > 0.0)

    return h


@pytest.mark.parametrize("dtype", FOUR)
@pytest.mark.parametrize("semantics", ("real", "ieee"))
def test_a_scalar_constant_is_visible_in_every_catalogued_format(
    dtype, semantics
):
    """`assert_(y > 0.0)` — the most ordinary harness there is.

    Before M12, float16 and bfloat16 answered UNKNOWN with a "literal
    outside the domain" note, because `_STRUCT_FMT` had no `<f2` and no
    route for bfloat16's `<V2`. CHANGELOG advertised "all four catalogued
    formats" while two of them could not read a threshold.
    """
    v = check(_const_harness(dtype), vacuity_mode="inputs-only",
              semantics=semantics)
    assert v.status == "VERIFIED", (dtype, semantics, v.notes)
    assert [o.status for o in v.obligations] == ["discharged"]
    assert not any("outside the domain" in n for n in v.notes)


def test_float16_constants_decode_to_their_exact_values():
    """A float16 value is exactly a binary64, so the decoded interval is a
    POINT at that value — not a bracket."""
    import numpy as np

    vals = [1.0, -2.5, 65504.0, 6.103515625e-05, 5.960464477539063e-08]
    arr = np.asarray(vals, dtype=np.float16)
    box = _decode_array(
        ir.Array(dtype="<f2", shape=(len(vals),), data=arr.tobytes()),
        "float16",
    )
    assert box.los == box.his == tuple(float(v) for v in vals)


def test_bfloat16_constants_decode_to_their_exact_values():
    """bfloat16 is binary32 with the low 16 significand bits removed, so
    the decode is a bit shift and is exact — checked against ml_dtypes,
    which stelling itself never imports."""
    import numpy as np

    ml = pytest.importorskip("ml_dtypes")
    vals = [1.0, -2.5, 3.0, 1.5, 0.0, 128.0, 1.1754943508222875e-38]
    arr = np.asarray(vals, dtype=ml.bfloat16)
    box = _decode_array(
        ir.Array(dtype="<V2", shape=(len(vals),), data=arr.tobytes()),
        "bfloat16",
    )
    expect = tuple(float(v) for v in arr.astype(np.float64))
    assert box.los == box.his == expect


def test_a_two_byte_void_is_decoded_only_under_the_bfloat16_name():
    """`<V2` is an ANONYMOUS 2-byte void — every 2-byte structured dtype
    spells it. Decoding one as bfloat16 on the strength of the byte string
    alone would read an arbitrary record as a float, which is a wrong
    VALUE where the old behaviour was only a ⊤."""
    a = ir.Array(dtype="<V2", shape=(), data=b"\x80\x3f")
    assert _decode_array(a, "bfloat16").los == (1.0,)
    for name in (None, "float16", "void16", "int16"):
        with pytest.raises(ir.TranscriptionError) as e:
            _decode_array(a, name)
        assert "bfloat16" in str(e.value)


def test_a_malformed_bfloat16_payload_is_refused_at_construction():
    """`ir._load_itemsize` already reads `2` out of `<V2`, so a truncated
    bfloat16 payload never reaches the decoder — it is refused where every
    other dtype's is. The decoder carries the same predicate as a second
    guard for a construction that bypasses `__post_init__`."""
    with pytest.raises(ir.TranscriptionError) as e:
        ir.Array(dtype="<V2", shape=(3,), data=b"\x80\x3f\x00")
    assert "3 byte(s), expected 6" in str(e.value)
    bypass = object.__new__(ir.Array)
    object.__setattr__(bypass, "dtype", "<V2")
    object.__setattr__(bypass, "shape", (3,))
    object.__setattr__(bypass, "data", b"\x80\x3f\x00")
    with pytest.raises(iv.IntervalError) as e2:
        _decode_array(bypass, "bfloat16")
    assert "3 byte(s), expected 6" in str(e2.value)


def test_an_undecodable_constant_still_tops_out_rather_than_crashing():
    """The degrade-don't-crash posture M12 must not have traded away."""
    from stelling.propagate import _Propagator

    p = _Propagator("constrain")
    atom = ir.Literal(
        val=ir.Array(dtype="<V2", shape=(), data=b"\x00\x00"),
        aval=_aval("void16"),
    )
    box = p.read(atom)
    assert (box.los, box.his) == ((-iv._INF,), (iv._INF,))
    assert p.read_flag(atom) is True
    assert any("outside the domain" in n for n in p.notes)


# -- M13: the comparison band is the WIDEST operand band ----------------------


def test_a_bfloat16_float16_comparison_uses_float16s_band():
    """Alphabetically bfloat16 comes first and its band is `2**-126`; the
    float16 operand needs `2**-14`, 112 decades wider. The band is what
    keeps a verdict sound for a flushing target, so the narrow pick
    dropped the flushing half (audit 0.2.0 M13)."""
    got = _ieee_cmp_get_min_normal(_FakeEqn(("bfloat16", "float16")))
    assert got == 2.0 ** -14
    assert got == _ieee_format_min_normal(_FLOAT_FORMATS["float16"])


def test_every_ordered_format_pair_gets_a_band_no_operand_outgrows():
    """The whole 4x4 grid, both orders — the rule is a MAXIMUM, so no
    ordering of the same pair can change the answer."""
    checked = 0
    for a in FOUR:
        for b in FOUR:
            need = max(_ieee_format_min_normal(_FLOAT_FORMATS[a]),
                       _ieee_format_min_normal(_FLOAT_FORMATS[b]))
            assert _ieee_cmp_get_min_normal(_FakeEqn((a, b))) == need
            assert _ieee_cmp_get_min_normal(_FakeEqn((b, a))) == need
            checked += 1
    assert checked == len(FOUR) ** 2 == 16


def test_a_comparison_of_an_unsupported_float_format_still_declines():
    with pytest.raises(iv.IntervalError) as e:
        _ieee_cmp_get_min_normal(_FakeEqn(("float32", "float8_e4m3fn")))
    assert "not a supported format" in str(e.value)


def test_an_integer_comparison_keeps_the_binary64_band():
    assert _ieee_cmp_get_min_normal(_FakeEqn(("int32", "int32"))) == iv.MIN_NORMAL


# -- M14: the two mode-wide stamps say what the RUN did -----------------------


def _ieee_assumptions(dtype):
    def h():
        x = any_array((), dtype, (1.0, 2.0))
        return assert_(x + x > 0.0)

    return check(h, vacuity_mode="inputs-only",
                 semantics="ieee").stamp.assumptions


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
def test_a_narrow_format_verdict_does_not_stamp_the_binary64_sentences(dtype):
    """Both sentences are FALSE of a narrow-format run: the endpoints WERE
    outward-rounded to the target grid (that is all of `_ieee_round_box`),
    and the band applied was the format's, not `2**-1022`."""
    got = _ieee_assumptions(dtype)
    assert iv.IEEE_ENDPOINT_ASSUMPTION not in got
    assert iv.SUBNORMAL_INDETERMINACY_ASSUMPTION not in got
    endpoint = [a for a in got if a.startswith("ieee endpoint arithmetic")]
    band = [a for a in got if a.startswith("subnormal indeterminacy")]
    assert len(endpoint) == len(band) == 1
    assert dtype in endpoint[0] and dtype in band[0]
    assert "rounded OUTWARD onto the target format's own ulp grid" in endpoint[0]
    assert "2**-1022" not in band[0]
    assert {"float16": "2**-14", "bfloat16": "2**-126",
            "float32": "2**-126"}[dtype] in band[0]


def test_a_binary64_verdict_stamps_the_unchanged_binary64_sentences():
    """The case those sentences were written for keeps them verbatim —
    a reworded stamp on an unchanged run is its own disclosure noise."""
    got = _ieee_assumptions("float64")
    assert iv.IEEE_ENDPOINT_ASSUMPTION in got
    assert iv.SUBNORMAL_INDETERMINACY_ASSUMPTION in got


def test_a_mixed_format_query_names_every_format_it_contains():
    def h():
        x = any_array((), "float32", (1.0, 2.0))
        y = any_array((), "float64", (1.0, 2.0))
        return (assert_(x + x > 0.0), assert_(y + y > 0.0))

    got = check(h, vacuity_mode="inputs-only", semantics="ieee").stamp.assumptions
    band = [a for a in got if a.startswith("subnormal indeterminacy")][0]
    assert "float32" in band and "float64" in band
    assert "2**-126" in band and "2**-1022" in band


def test_the_real_mode_stamp_carries_neither_ieee_sentence():
    def h():
        x = any_array((), "float32", (1.0, 2.0))
        return assert_(x + x > 0.0)

    got = check(h, vacuity_mode="inputs-only").stamp.assumptions
    assert not [a for a in got if a.startswith("ieee endpoint arithmetic")]
    assert not [a for a in got if a.startswith("subnormal indeterminacy")]


# -- M15: a binary kernel with no format row declines, and cannot be added ----


def _one_ulp_subnormal_box():
    """float32 `x + x` at a float32 SUBNORMAL: jax computes 0.0 (FTZ), the
    format-parametric kernel hulls with 0, the binary64 kernel does not."""
    v = 2.0 ** -140
    return iv.IntervalArray(shape=(), los=(v,), his=(v,))


def test_the_binary64_kernel_on_a_narrow_format_would_exclude_what_jax_computes():
    """The premise M15 rests on, measured rather than asserted: this is
    what the removed fallback would have produced."""
    f32 = _FLOAT_FORMATS["float32"]
    a = _one_ulp_subnormal_box()
    mapped, _ = iv.ieee_add_fmt(a, a, _ieee_format_min_normal(f32))
    mapped = _ieee_round_box(mapped, f32)
    fallback, _ = iv.ieee_add(a, a)
    fallback = _ieee_round_box(fallback, f32)
    executed = float(jnp.asarray(2.0 ** -140, "float32")
                     + jnp.asarray(2.0 ** -140, "float32"))
    assert executed == 0.0
    assert mapped.los[0] <= executed <= mapped.his[0]
    assert not (fallback.los[0] <= executed <= fallback.his[0])


def test_an_unmapped_binary_kernel_declines_on_a_narrow_format():
    """A fifth binary kernel registered without a `_FMT_BINARY_OPS` row
    now DECLINES rather than silently using the binary64 band."""

    def unmapped(a, b):  # pragma: no cover - never reached past the guard
        return iv.ieee_add(a, b)

    t = _ieee_arith(unmapped)
    eqn = _FakeEqn(("float32", "float32"), ("float32",))
    eqn.primitive = "add"
    a = _one_ulp_subnormal_box()
    with pytest.raises(iv.IntervalError) as e:
        t(eqn, {}, [a, a], [False, False])
    assert "no format-parametric kernel" in str(e.value)
    assert "float32" in str(e.value)


def test_an_unmapped_binary_kernel_is_still_fine_on_binary64():
    """The decline is about the BAND, and binary64 is the band the
    binary64 kernel already uses."""

    def unmapped(a, b):
        return iv.ieee_add(a, b)

    t = _ieee_arith(unmapped)
    eqn = _FakeEqn(("float64", "float64"), ("float64",))
    eqn.primitive = "add"
    a = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    (box,), _flags = t(eqn, {}, [a, a], [False, False])
    assert (box.los[0], box.his[0]) == (2.0, 4.0)


def test_the_import_time_census_bites_when_a_format_row_goes_missing(
    monkeypatch
):
    """The check that makes the coupling structural: `_FMT_BINARY_OPS` and
    `IEEE_TRANSFERS` are two hand-written lists, and this is what refuses
    the import when they stop agreeing."""
    thinned = dict(_FMT_BINARY_OPS)
    thinned.pop(iv.ieee_mul)
    monkeypatch.setattr(
        "stelling.propagate._FMT_BINARY_OPS", thinned, raising=True
    )
    with pytest.raises(RuntimeError) as e:
        _assert_ieee_binary_kernels_are_format_parametric()
    assert "mul" in str(e.value)


def test_the_import_time_census_bites_when_the_hook_stops_being_attached(
    monkeypatch
):
    """...and it cannot pass by inspecting nothing."""
    import stelling.propagate as P

    stripped = dict(P.IEEE_TRANSFERS)

    def bare(eqn, params, ins, flags):  # pragma: no cover
        raise AssertionError("never called")

    stripped["add"] = (bare, P.TIER_EXACT)
    stripped["add_any"] = (bare, P.TIER_EXACT)
    monkeypatch.setattr("stelling.propagate.IEEE_TRANSFERS", stripped)
    with pytest.raises(RuntimeError) as e:
        _assert_ieee_binary_kernels_are_format_parametric()
    assert "census" in str(e.value)


def test_the_live_registry_passes_its_own_census():
    _assert_ieee_binary_kernels_are_format_parametric()
    assert len(_FMT_BINARY_OPS) == 4


def test_a_narrow_format_add_still_hulls_with_its_own_band_end_to_end():
    """The property the fallback would have broken, through the public
    entry point: a float32 subnormal sum is INDETERMINATE, not definite."""

    def h():
        x = any_array((), "float32", (2.0 ** -140, 2.0 ** -140))
        return assert_(x + x > 0.0)

    v = check(h, vacuity_mode="inputs-only", semantics="ieee")
    assert v.status == "UNKNOWN"
    assert float(jnp.asarray(2.0 ** -140, "float32")
                 + jnp.asarray(2.0 ** -140, "float32")) == 0.0
