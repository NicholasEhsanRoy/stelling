# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The convert decline notes claim only measured truths.

Three assertions in ``_t_convert``'s user-facing decline text were false;
each is fixed and pinned here by a test that goes red if it comes back:

1. The terminal decline called EVERY unlisted cast "not exact" and closed
   with "every other cast may change the value it carries" — false for
   uint32->uint64, uint8->int32, bool->float16 and uint32->float64, each
   exact at every representable source value (measured below, on the
   values themselves). The decline is a whitelist gap and STANDS; the
   sentence about it may not invent a value change.
2. The same sentence named "bool->any" among the admitted casts, while
   ("bool", "float16") is not in ``_EXACT_CONVERSIONS`` — a traced
   bool->float16 cast was declined by the very sentence asserting bool
   casts are admitted. The examples are now built from the whitelist,
   and the test re-derives every cast the emitted text names.
3. The float->int range refusal printed the target's range computed in
   floats: ``float(2**63) - 1`` rounds back to ``2**63``, so the note
   named 9.223372036854776e+18 (= 2**63) as int64's maximum — a value
   int64 cannot hold. The printed bounds now come from
   ``_INT_DTYPE_BOUNDS``, and the test walks every integer target the
   refusal can reach, int64 included.

Text-truth tests only: every decline exercised here must still decline,
and one test pins the corrected decline's coverage accounting on hand IR.
Skipped without jax — the point of most of these is measuring the claims
against the real target's conversions.
"""

from __future__ import annotations

import re

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling import ir  # noqa: E402
from stelling import propagate as P  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _decline_note(src: str, dst: str, bounds=(0, 1)):
    """Drive ``src -> dst`` through a real traced walk; return the
    propagation result and the terminal convert note naming this pair."""

    def h():
        x = any_array((2,), src, bounds)
        return assert_(
            jnp.sum(jnp.asarray(x.astype(dst), jnp.float64)) <= 1e30
        )

    p = P.propagate(trace(h))
    notes = [n for n in p.notes if f"{src!r} -> {dst!r}" in n]
    assert notes, p.notes
    return p, notes[0]


def _int_probes(name: str):
    """Boundary and interior values of an integer/bool dtype — the values
    an exactness claim is measured on."""
    lo, hi = P._INT_DTYPE_BOUNDS[name]
    vals = {lo, lo + 1, 0, 1, hi - 1, hi, (hi // 3) | 1}
    return sorted(v for v in vals if lo <= v <= hi)


def _measured_roundtrip(src: str, dst: str, values):
    """Convert each value on the real target; return the (value, image)
    pairs that fail to cross unchanged. ``tolist()`` yields python
    scalars, so the comparisons are exact (no numpy mixed-width float
    detour, which is the very trap finding 3 is about)."""
    vals = list(values)
    got = jnp.asarray(vals, src).astype(dst).tolist()
    return [(v, g) for v, g in zip(vals, got) if not (g == v)]


def _constructible(name: str) -> bool:
    try:
        jnp.asarray([0], name).astype(name)
        return True
    except Exception:
        return False


# --- finding 1: exact-but-unlisted casts may not be called inexact ----------

_EXACT_BUT_UNLISTED = [
    ("uint32", "uint64"),
    ("uint8", "int32"),
    ("bool", "float16"),
    ("uint32", "float64"),
]


@pytest.mark.parametrize("src,dst", _EXACT_BUT_UNLISTED)
def test_exact_but_unlisted_cast_note_claims_only_truths(src, dst):
    """The decline stands; the note's every claim is checked against
    ground truth: exactness measured on the source dtype's own values,
    unlistedness against the whitelist as it is, and the printed numbers
    against the tables they must come from."""
    # the finding's precondition, re-checked so this test cannot rot into
    # exercising a whitelisted (never-declining) pair
    assert (src, dst) not in P._EXACT_CONVERSIONS
    p, note = _decline_note(src, dst)
    # the DECISION is untouched: still declined, obligation still unknown
    assert p.obligations[0].status == "unknown"
    # the measured-false claims are gone
    assert "not exact" not in note, note
    assert "may change the value" not in note, note
    # claim (i): "exact at every representable value" — measured (uint8
    # and bool exhaustively; boundary+interior probes for the 2**32 case)
    assert "is exact at every representable" in note, note
    probes = (
        list(range(256)) if src == "uint8"
        else [0, 1] if src == "bool"
        else _int_probes(src)
    )
    assert _measured_roundtrip(src, dst, probes) == []
    # claim (ii): "not listed" — true of the whitelist as it actually is
    assert "not listed in propagate._EXACT_CONVERSIONS" in note, note
    # claim (iii): the numbers printed are the true table numbers, each
    # BOUND to its dtype — the whole reason phrase must match, because
    # mere per-number substring checks passed a mutant that swapped the
    # source range with the destination range (audit repair R4)
    slo, shi = P._INT_DTYPE_BOUNDS[src]
    if dst in P._INT_DTYPE_BOUNDS:
        dlo, dhi = P._INT_DTYPE_BOUNDS[dst]
        assert (
            f"every {src} value lies in [{slo}, {shi}], inside {dst}'s "
            f"[{dlo}, {dhi}], where the conversion is the identity"
        ) in note, note
    else:
        p_bits, _, _ = P._FLOAT_FORMATS[dst]
        assert (
            f"{dst} represents every integer of magnitude at most "
            f"2**{p_bits} exactly, and {src} spans [{slo}, {shi}]"
        ) in note, note


def _witness(src: str, dst: str):
    """One value the classifier's "inexact" verdict promises will change,
    derived from the same tables the verdict came from. Raises if a
    decided-inexact pair has no witness rule — that is a test gap, and it
    must fail loudly rather than skip."""
    s_int, d_int = P._INT_DTYPE_BOUNDS.get(src), P._INT_DTYPE_BOUNDS.get(dst)
    s_flt, d_flt = P._FLOAT_FORMATS.get(src), P._FLOAT_FORMATS.get(dst)
    if s_int and d_int:
        (slo, shi), (dlo, dhi) = s_int, d_int
        return dhi + 1 if shi > dhi else dlo - 1
    if s_int and d_flt:
        (slo, shi), (p_bits, _, _) = s_int, d_flt
        # every table row's hi is odd, so past 2**p it cannot fit p
        # significand bits; the low side alone never decides (its
        # magnitude is a power of two, which any float holds)
        assert shi > 2**p_bits, (src, dst)
        return shi
    if s_flt and d_int:
        return 0.5
    if s_flt and d_flt:
        (ps, _, xs), (pd, _, xd) = s_flt, d_flt
        if pd < ps:
            return 1.0 + 2.0 ** (1 - ps)  # needs ps bits; rounds in pd
        assert xd < xs, (src, dst)  # the remaining decided case today
        return (2.0 - 2.0 ** (1 - ps)) * 2.0**xs  # src max finite
    raise AssertionError(f"no witness rule for {src}->{dst}")


def test_classifier_verdicts_are_measured_on_the_target():
    """Every "exact" verdict is measured value-preserving and every
    "inexact" verdict is measured value-changing on the real target, over
    all classifiable dtype pairs jax can construct. This is what keeps
    the note's exact/inexact split from ever asserting a change that
    cannot happen (finding 1) — or exactness that is not there."""
    names = sorted(
        n
        for n in set(P._INT_DTYPE_BOUNDS) | set(P._FLOAT_FORMATS)
        if _constructible(n)
    )
    core = {
        "bool", "uint8", "uint32", "uint64", "int32", "int64",
        "float16", "float32", "float64",
    }
    assert core <= set(names)  # the sweep must not silently shrink
    checked_exact = checked_inexact = 0
    for src in names:
        for dst in names:
            if src == dst:
                continue
            verdict, why = P._conversion_exactness(src, dst)
            assert why.strip(), (src, dst)
            if verdict == "exact":
                assert src in P._INT_DTYPE_BOUNDS, (src, dst, why)
                changed = _measured_roundtrip(src, dst, _int_probes(src))
                assert changed == [], (src, dst, changed)
                checked_exact += 1
            elif verdict == "inexact":
                w = _witness(src, dst)
                changed = _measured_roundtrip(src, dst, [w])
                assert changed, (src, dst, w)
                checked_inexact += 1
    # both arms must have been exercised many times over, or the sweep
    # proved nothing
    assert checked_exact >= 10 and checked_inexact >= 10, (
        checked_exact,
        checked_inexact,
    )


def test_whitelist_never_holds_a_provably_inexact_cast():
    """The other coupling direction: _EXACT_CONVERSIONS may never list a
    cast the classifier can prove value-changing. Integer and bool
    sources classify exact outright; the float widenings classify
    "unknown" — the format embeds, and the note says only that, because
    per-dtype DAZ was measured flushing an f32 subnormal to 0.0 across a
    convert on this target (see the ieee convert rule)."""
    for pair in sorted(P._EXACT_CONVERSIONS):
        verdict, why = P._conversion_exactness(*pair)
        assert verdict != "inexact", (pair, why)
        if pair[0] in P._INT_DTYPE_BOUNDS:
            assert verdict == "exact", (pair, why)
        else:
            # the embed fact must bind each significand count to ITS
            # format — a mutant swapping the two numbers read "24
            # significand bits into 11" and passed a bare "embeds"
            # check (audit repair R4)
            assert verdict == "unknown", (pair, why)
            ps, _, _ = P._FLOAT_FORMATS[pair[0]]
            pd, _, _ = P._FLOAT_FORMATS[pair[1]]
            assert (
                f"{pair[1]} embeds {pair[0]} ({ps} significand bits "
                f"into {pd}, exponent range covered)"
            ) in why, (pair, why)


# --- finding 2: casts the note names as admitted must be whitelisted --------

_ADMITTED_RE = re.compile(r"_EXACT_CONVERSIONS \(([^)]*)\)")
_ARROW_RE = re.compile(r"(\w+)->(\w+)")


def test_note_names_admitted_casts_only_from_the_whitelist():
    """Re-derive every cast the emitted note presents as admitted (the
    parenthesized examples after the _EXACT_CONVERSIONS mention) and
    check each against the whitelist as it actually is. The old text
    hard-coded "bool->any" while declining bool->float16 — the first
    driver below is that exact cast, so the regression cannot return
    unnoticed. The int64->int32 narrowing clause sits outside the
    parentheses: it is admitted conditionally, not via the whitelist."""
    for src, dst, bounds in [
        ("bool", "float16", (0, 1)),  # the measured "bool->any" falsehood
        ("float64", "float32", (0.1, 0.1)),  # an inexact-branch note
    ]:
        _, note = _decline_note(src, dst, bounds)
        groups = _ADMITTED_RE.findall(note)
        assert groups, note
        named = [m for g in groups for m in _ARROW_RE.findall(g)]
        assert named, note  # the sentence must actually name examples
        for pair in named:
            assert pair in P._EXACT_CONVERSIONS, (pair, note)
    # and at the source: the builder the message calls
    named = _ARROW_RE.findall(P._admitted_examples())
    assert len(named) >= 1
    assert all(pair in P._EXACT_CONVERSIONS for pair in named)


# --- finding 3: the range refusal prints exact integer bounds ---------------

def test_float_to_int_range_refusal_prints_the_exact_bounds():
    """For EVERY integer target the truncation rule can reach — int64
    included, where float(2**63) - 1 rounds back to 2**63 — the printed
    representable range must be exactly the integer pair from
    _INT_DTYPE_BOUNDS, character for character: a float-formatted or
    off-by-rounding range cannot match the extracted segment."""
    assert "int64" in P._INT_RANGE  # the finding's dtype; never int32-only
    for dst in sorted(P._INT_RANGE):
        lo_b, hi_b = P._INT_DTYPE_BOUNDS[dst]

        def h(dst=dst):
            x = any_array((2,), "float64", (0.0, float(2**70)))
            return assert_(
                jnp.sum(jnp.asarray(x.astype(dst), jnp.float64)) <= 1e40
            )

        p = P.propagate(trace(h))
        assert p.obligations[0].status == "unknown"  # the refusal stands
        notes = [n for n in p.notes if "truncates toward zero" in n]
        assert notes, p.notes
        m = re.search(r"representable range \[([^\]]+)\]", notes[0])
        assert m, notes[0]
        assert m.group(1) == f"{lo_b}, {hi_b}", notes[0]


# --- the fix is text-only: decision and accounting pinned on hand IR --------

def _convert_query(src_dt: str, dst_dt: str) -> ir.ClosedJaxpr:
    """Hand IR with exactly ONE convert equation — any(src) -> convert ->
    le(1e30) -> assert — so a decline's coverage count is exact, and any
    dtype STRING can ride in the avals (the walk must handle dtypes no
    table has heard of)."""

    def aval(d):
        return ir.Aval(kind="ShapedArray", shape=(), dtype=d)

    x, y = ir.Var(id=0, aval=aval(src_dt)), ir.Var(id=1, aval=aval(dst_dt))
    pred, ob = ir.Var(id=2, aval=aval("bool")), ir.Var(id=3, aval=aval("bool"))
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(),
            outvars=(ob,),
            eqns=(
                ir.JaxprEqn(
                    primitive="stelling_any",
                    invars=(),
                    outvars=(x,),
                    params=(
                        ("shape", ()),
                        ("dtype", src_dt),
                        ("lo", 0.0),
                        ("hi", 5.0),
                    ),
                ),
                ir.JaxprEqn(
                    primitive="convert_element_type",
                    invars=(x,),
                    outvars=(y,),
                    params=(("new_dtype", dst_dt),),
                ),
                ir.JaxprEqn(
                    primitive="le",
                    invars=(y, ir.Literal(val=1e30, aval=aval("float64"))),
                    outvars=(pred,),
                ),
                ir.JaxprEqn(
                    primitive="stelling_assert",
                    invars=(pred,),
                    outvars=(ob,),
                ),
            ),
        )
    )


def test_exact_unlisted_decline_keeps_decision_and_accounting():
    """The truth fix must never become an admission: an exact-but-unlisted
    cast still declines, still counts unknown, and still tops out. Hand
    IR with exactly one convert equation, so the count is exact."""
    p = P.propagate(_convert_query("uint32", "uint64"))
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("convert_element_type", 1),)


def test_unknown_verdict_casts_still_decline_with_accounting():
    """The UNKNOWN-verdict arm is an arm of the decline, not of admission.
    A mutant returning the operand for unknown verdicts survived the whole
    prior suite while flipping bfloat16->float32 — and a dtype no table
    has heard of — to discharged with coverage.unknown == 0, a
    false-VERIFIED shape (audit repair R3). Pinned through the real walk,
    traced and hand-built, with exact accounting each time."""
    # traced: bfloat16 -> float32, the embeds-unknown pair
    def h():
        x = any_array((2,), "bfloat16", (0.0, 2.0))
        return assert_(
            jnp.sum(jnp.asarray(x.astype("float32"), jnp.float64)) <= 1e30
        )

    p = P.propagate(trace(h))
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("convert_element_type", 1),)
    note = next(n for n in p.notes if "'bfloat16' -> 'float32'" in n)
    assert "not listed in propagate._EXACT_CONVERSIONS" in note, note
    # the embed fact binds each significand count to ITS format (R4)
    ps, _, _ = P._FLOAT_FORMATS["bfloat16"]
    pd, _, _ = P._FLOAT_FORMATS["float32"]
    assert (
        f"float32 embeds bfloat16 ({ps} significand bits into {pd}, "
        f"exponent range covered)"
    ) in note, note
    # hand IR: an out-of-table dtype (real, and outright garbage) must
    # decline identically — the walk may never vouch for a dtype nothing
    # here can classify
    for src_dt in ("float8_e4m3fn", "notadtype9"):
        p2 = P.propagate(_convert_query(src_dt, "float16"))
        assert p2.obligations[0].status == "unknown", src_dt
        assert p2.coverage.unknown == 1, src_dt
        assert p2.coverage.unknown_primitives == (
            ("convert_element_type", 1),
        ), src_dt
        note2 = next(n for n in p2.notes if f"{src_dt!r} -> 'float16'" in n)
        assert "neither an exactness proof nor a value-change witness" in note2
