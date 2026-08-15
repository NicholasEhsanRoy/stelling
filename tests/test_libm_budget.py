# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The DECLARED libm accuracy budget — audit 0.2.0 S9 and S11.

Under ``semantics="ieee"`` a verdict is a claim about the float value the
program computes. ``iv.exp`` brackets CPython's ``math.exp`` — glibc on the
host running the analysis — with a ±1-binary64-ulp bump. The program runs
whatever the compiler emitted. Two different functions, and a bracket of
one is not a bracket of the other.

**S11 reaches the released 0.1.0**, where ``propagate(closed,
semantics="ieee")`` was the door to ieee mode.

Every escaping argument the audit measured is re-measured here against the
live backend inside the test that uses it, so no row can pass by agreeing
with a stale number.
"""

import math

import pytest

import stelling.interval as iv
from stelling.propagate import (
    LIBM_BUDGET_OPS,
    LIBM_MEASURED,
    LIBM_PROFILES,
    LibmBudget,
    _FLOAT_FORMATS,
    _ieee_format_min_normal,
    _ieee_round_box,
    _libm_ulp_at,
    _libm_widen_box,
    resolve_libm_budget,
)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402

PROFILE = "xla-cpu-2026-08"


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _ieee(h, **kw):
    return check(h, vacuity_mode="inputs-only", semantics="ieee", **kw)


# -- S11: the binary64 escape, at the audit's own argument --------------------

# `iv.exp([X, X])`'s upper endpoint on the pre-fix tree. jnp.exp(X) is one
# binary64 ulp ABOVE it, so `assert_(y <= K)` was VERIFIED and false.
S11_X = -217.29998556254742
S11_K = 4.2443906829988504e-95


def test_s11_the_binary64_exp_escape_is_not_verified_without_a_budget():
    """THE FINDING THAT REACHES 0.1.0. The bracket is glibc's; the program
    runs XLA's; they differ by an ulp at an entirely ordinary argument."""
    executed = float(jnp.exp(jnp.asarray(S11_X, "float64")))
    assert executed > S11_K, (
        "the backend no longer escapes at this argument — re-measure "
        "before trusting this row"
    )
    assert executed != math.exp(S11_X), (
        "jnp.exp and math.exp agree here now; the escape this row pins is "
        "the disagreement between them"
    )

    def h():
        x = any_array((), "float64", (S11_X, S11_X))
        return assert_(jnp.exp(x) <= S11_K)

    v = _ieee(h)
    assert v.status == "UNKNOWN"
    assert any("no DECLARED accuracy budget" in n for n in v.notes)


def test_s11_stays_undecided_with_the_shipped_budget_declared():
    """The budget does not restore the old answer — it widens the bracket
    until it contains the value the program really computes, which makes
    the obligation UNDECIDED rather than false."""

    def h():
        x = any_array((), "float64", (S11_X, S11_X))
        return assert_(jnp.exp(x) <= S11_K)

    v = _ieee(h, libm_budget=PROFILE)
    assert v.status == "UNKNOWN"
    assert [o.status for o in v.obligations] == ["unknown"]


def test_the_widened_binary64_box_contains_what_the_backend_computes():
    """The direct statement of the repair, at the kernel."""
    fmt = _FLOAT_FORMATS["float64"]
    u = LIBM_PROFILES[PROFILE].get("exp", "float64")
    a = iv.IntervalArray(shape=(), los=(S11_X,), his=(S11_X,))
    before = iv.exp(a)
    after = _libm_widen_box(before, fmt, u, floor=0.0)
    executed = float(jnp.exp(jnp.asarray(S11_X, "float64")))
    assert not (before.los[0] <= executed <= before.his[0])
    assert after.los[0] <= executed <= after.his[0]


# -- S9: the float32 escape, both directions ----------------------------------

S9_C = 88.71259307861328          # exactly a float32
S9_LO = 3.3681360838329314e+38    # exactly a float32, and the box's old lo
S9_K_REFUTED = 3.3681358810088354e+38  # between the executed value and the box


def test_s9_the_float32_exp_escape_is_not_verified_without_a_budget():
    executed = float(jnp.exp(jnp.asarray(S9_C, "float32")))
    assert executed < S9_LO, (
        "the backend no longer falls short here — re-measure before "
        "trusting this row"
    )

    def h():
        x = any_array((), "float32", (S9_C, S9_C))
        return assert_(jnp.exp(x) >= jnp.asarray(S9_LO, "float32"))

    v = _ieee(h)
    assert v.status == "UNKNOWN"
    assert any("no DECLARED accuracy budget" in n for n in v.notes)


def test_s9_stays_undecided_with_the_shipped_budget_declared():
    def h():
        x = any_array((), "float32", (S9_C, S9_C))
        return assert_(jnp.exp(x) >= jnp.asarray(S9_LO, "float32"))

    assert _ieee(h, libm_budget=PROFILE).status == "UNKNOWN"


def test_s9_the_false_refuted_half_closes_too():
    """A wider bracket makes VERIFIED and REFUTED both harder, so the same
    change closes the other face: with the threshold between the executed
    value and the box, ieee mode called the obligation DEFINITELY FALSE
    where the float32 execution makes it true."""
    executed = float(jnp.exp(jnp.asarray(S9_C, "float32")))
    assert executed < S9_K_REFUTED, (
        "the predicate (y < K) is no longer TRUE at the executed value — "
        "re-measure"
    )

    def h():
        x = any_array((), "float32", (S9_C, S9_C))
        return assert_(jnp.exp(x) < jnp.asarray(S9_K_REFUTED, "float32"))

    assert _ieee(h).status == "UNKNOWN"
    assert _ieee(h, libm_budget=PROFILE).status == "UNKNOWN"


def test_s9_is_not_an_artifact_of_a_point_declaration():
    """The audit's F1c: a box holding 2,622 distinct float32 values was
    VERIFIED while a point of it violated."""
    lo, hi = 88.70000457763672, 88.72000122070312
    k = float(jnp.exp(jnp.asarray(hi, "float32")))

    def h():
        x = any_array((), "float32", (lo, hi))
        return assert_(jnp.exp(x) >= jnp.asarray(3.3260020145945936e+38,
                                                 "float32"))

    assert _ieee(h).status == "UNKNOWN"
    assert _ieee(h, libm_budget=PROFILE).status == "UNKNOWN"
    assert math.isfinite(k)


# -- the containment sweep behind the profile ---------------------------------

# The exact number of (argument, format) containment checks the sweep below
# executes. Asserted rather than bounded, the way
# `BOUNDARY_DIV_SWEEP_QUOTIENTS` is: a figure quoted in SOUNDNESS.md that no
# run in the tree reproduces is worth less than no figure.
LIBM_EXP_SWEEP_CHECKS = 4003


def _exp_box(arg, fmt_name, ulps):
    fmt = _FLOAT_FORMATS[fmt_name]
    mn = _ieee_format_min_normal(fmt)
    a = iv.IntervalArray(shape=(), los=(arg,), his=(arg,))
    box = _libm_widen_box(iv.exp(a), fmt, ulps, floor=0.0)
    box = iv.subnormal_haze_fmt(box, mn)[0]
    return _ieee_round_box(box, fmt)


def _exp_sweep_args(fmt_name):
    """Arguments crowding the places the audit found escapes, plus a
    uniform spread. Built from the FORMAT's own values so a point is a
    value of the declared set rather than a number near one."""
    import numpy as np
    import ml_dtypes

    dt = {"float16": np.float16, "bfloat16": ml_dtypes.bfloat16,
          "float32": np.float32, "float64": np.float64}[fmt_name]
    lo, hi = {"float16": (-16.0, 11.0), "bfloat16": (-88.0, 88.0),
              "float32": (-104.0, 88.72), "float64": (-745.0, 709.0)}[fmt_name]
    xs = list(np.linspace(lo, hi, 900))
    if fmt_name == "float32":                 # the measured escape band
        xs += list(np.linspace(88.5463, 88.7228, 300))
    if fmt_name == "float64":
        xs += [S11_X, -468.9064012972897, -22.486606621413898]
        xs += list(np.linspace(700.0, 709.7, 100))
    return [x for x in np.asarray(xs, dt).astype(np.float64)
            if math.isfinite(x)]


def test_the_declared_budget_brackets_what_the_backend_computes():
    """The property the profile's numbers are FOR, executed rather than
    asserted: at every sampled argument the widened, hazed, format-rounded
    box contains the value jax computes."""
    import numpy as np
    import ml_dtypes

    npdt = {"float16": np.float16, "bfloat16": ml_dtypes.bfloat16,
            "float32": np.float32, "float64": np.float64}
    budget = LIBM_PROFILES[PROFILE]
    checked = 0
    for fmt_name in ("float16", "bfloat16", "float32", "float64"):
        u = budget.get("exp", fmt_name)
        args = _exp_sweep_args(fmt_name)
        vals = np.asarray(
            jnp.exp(jnp.asarray(np.asarray(args, npdt[fmt_name])))
        ).astype(np.float64)
        for arg, v in zip(args, vals.tolist()):
            box = _exp_box(float(arg), fmt_name, u)
            checked += 1
            assert box.los[0] <= v <= box.his[0], (
                f"{fmt_name} exp({arg!r}) = {v!r} escapes the declared "
                f"box [{box.los[0]!r}, {box.his[0]!r}] at {u} ulps"
            )
    assert checked == LIBM_EXP_SWEEP_CHECKS, (
        f"the sweep executed {checked} containment checks; SOUNDNESS.md "
        f"quotes {LIBM_EXP_SWEEP_CHECKS}. Update both or neither."
    )


def test_the_pre_fix_bracket_would_have_failed_the_same_sweep():
    """The sweep BITES: with the widening removed the same arguments
    escape, so a green above is a property of the fix and not of the
    sample."""
    escapes = 0
    import numpy as np

    for arg in (S11_X, -468.9064012972897, -22.486606621413898):
        box = _exp_box(arg, "float64", 0.5)   # 0.5 = no widening at all
        v = float(jnp.exp(jnp.asarray(arg, "float64")))
        if not (box.los[0] <= v <= box.his[0]):
            escapes += 1
    for arg in np.asarray(np.linspace(88.5463, 88.7228, 300), np.float32):
        box = _exp_box(float(arg), "float32", 0.5)
        v = float(jnp.exp(jnp.asarray(arg, "float32")))
        if not (box.los[0] <= v <= box.his[0]):
            escapes += 1
    assert escapes >= 100, escapes


# -- the decline is the feature -----------------------------------------------


def _decline_note(dtype="float64", op="exp"):
    def h():
        x = any_array((), dtype, (1.0, 2.0))
        y = jnp.exp(x) if op == "exp" else x ** 1.5
        return assert_(y > 0.0)

    v = _ieee(h)
    notes = [n for n in v.notes if "DECLARED accuracy budget" in n]
    assert len(notes) == 1, v.notes
    return v, notes[0]


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64"))
@pytest.mark.parametrize("op", ("exp", "pow"))
def test_every_libm_op_and_format_declines_without_a_declaration(dtype, op):
    """Fail closed, uniformly. bfloat16 exp is exhaustively CORRECTLY
    ROUNDED on the measured backend over every normal finite result, and
    it still declines: "measured well once, on one jaxlib, on one device"
    is not something a verdict may rest on silently — and float16, which
    the same measurement first read as correctly rounded and is not, is
    why."""
    v, note = _decline_note(dtype, op)
    assert v.status == "UNKNOWN"
    assert op in note and dtype in note


def test_the_decline_carries_the_evidence_that_justifies_it():
    _v, note = _decline_note("float32", "exp")
    assert "EXHAUSTIVE over every float32 argument" in note
    assert "5.5112 ulps" in note
    assert "88.5463" in note and "88.7228" in note
    assert "12,542" in note


def test_the_float32_population_is_derived_rather_than_transcribed():
    """The row said 2,237,668,968 — one too many, an inclusive/exclusive
    fencepost at the low edge (audit 0.2.0 B4). The figure is the count of
    float32 values whose `exp` is normal and finite, which the bit
    ordering computes in closed form, so it is COMPUTED here instead of
    believed. The band's two edges are pinned the same way: one step out
    on either side leaves it."""
    import numpy as np

    def ordinal(x):
        u = np.array([x], dtype=np.float32).view(np.int32)[0].item()
        return u if u >= 0 else -(u & 0x7FFFFFFF)

    min_normal = 2.0 ** -126
    lo, hi = np.float32(-87.33654022216797), np.float32(88.72283172607422)
    with np.errstate(over="ignore"):
        assert math.exp(float(lo)) >= min_normal
        assert math.exp(float(np.nextafter(lo, np.float32(-np.inf)))) \
            < min_normal
        assert math.isfinite(float(np.float32(math.exp(float(hi)))))
        assert math.isinf(float(np.float32(
            math.exp(float(np.nextafter(hi, np.float32(np.inf)))))))

    n = ordinal(hi) - ordinal(lo) + 1
    assert n == 2_237_668_967, n
    assert f"{n:,} of them" in LIBM_MEASURED[("exp", "float32")]
    # ... and the wider interval the row also names, which is 2,185,053
    # arguments larger — precisely the subnormal-result region
    wide = (ordinal(np.float32(88.73)) - ordinal(np.float32(-104.0)) + 1)
    assert wide == 2_239_854_020 and wide - n == 2_185_053, (wide, n)


def test_the_decline_carries_the_exact_incantation():
    _v, note = _decline_note("float32", "exp")
    assert f"libm_budget='{PROFILE}'" in note
    assert "from stelling.propagate import LibmBudget" in note
    assert "name='my-backend-2026-08'" in note
    assert "('exp', 'float32'): 6" in note
    assert "6 ulps" in note  # the shipped number for this pair
    # BOTH doors, not just check(): the S11 exposure this gate closes runs
    # through `propagate`, which takes the same keyword (audit 0.2.0 B4)
    assert f"propagate(closed, semantics='ieee', libm_budget='{PROFILE}')" \
        in note


def test_the_decline_says_a_fixed_wider_bracket_is_not_the_fix():
    """The audit's own suggested remedy — ±2 ulps "to cover any
    faithfully-rounded implementation" — is refuted by its own float32
    measurement, and the decline says so where someone would otherwise
    reach for it."""
    _v, note = _decline_note("float32", "exp")
    assert "A FIXED WIDER BRACKET IS NOT THE FIX" in note


def test_the_incantation_in_the_decline_actually_works():
    """The message is a promise; this is the promise kept."""

    def h():
        x = any_array((), "float32", (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    assert _ieee(h).status == "UNKNOWN"
    assert _ieee(h, libm_budget=PROFILE).status == "VERIFIED"


def test_every_line_the_decline_prints_RUNS_AS_WRITTEN():
    """A template is not an incantation (audit 0.2.0 B4).

    The first draft printed ``check(harness, vacuity_mode=..., ...)`` —
    ``...`` is ``Ellipsis`` and raises *"widen mode must be one of ('all',
    'inputs-only')"* — and ``ulps={('exp','float32'): <ulps>}``, which is a
    ``SyntaxError``. Both were templates wearing an incantation's clothes,
    and the test that was supposed to cover them checked a DIFFERENT line
    that happened to work. So every backticked call the decline prints is
    pulled out of a LIVE decline here and executed verbatim."""
    import re

    from stelling.harness import trace
    from stelling.propagate import LibmBudget, propagate

    _v, note = _decline_note("float32", "exp")
    printed = [m for m in re.findall(r"`([^`]+)`", note)
               if m.startswith(("check(", "propagate("))]
    assert len(printed) == 3, printed

    def harness():
        x = any_array((), "float32", (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    closed = trace(harness)
    env = {"harness": harness, "closed": closed, "check": check,
           "propagate": propagate, "LibmBudget": LibmBudget}
    for src in printed:
        out = eval(compile(src, "<the decline>", "eval"), env)  # noqa: S307
        # each one has to actually OPEN the gate, not merely parse
        status = getattr(out, "status", None)
        assert status == "VERIFIED" if status is not None \
            else out.all_discharged, (src, out)


# -- per (op, format), never extrapolated -------------------------------------


def test_a_budget_for_one_format_does_not_cover_another():
    only64 = LibmBudget(
        name="only-f64", basis="a test fixture, measured nowhere",
        ulps={("exp", "float64"): 2.0},
    )

    def h32():
        x = any_array((), "float32", (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    def h64():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    assert _ieee(h32, libm_budget=only64).status == "UNKNOWN"
    assert _ieee(h64, libm_budget=only64).status == "VERIFIED"


def test_a_budget_for_one_op_does_not_cover_another():
    only_exp = LibmBudget(
        name="only-exp", basis="a test fixture, measured nowhere",
        ulps={("exp", "float64"): 2.0},
    )

    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(x ** 1.5 > 0.0)

    v = _ieee(h, libm_budget=only_exp)
    assert v.status == "UNKNOWN"
    assert any("pow" in n and "DECLARED accuracy budget" in n for n in v.notes)


# -- 0.5 ulp stays legitimate -------------------------------------------------


@pytest.mark.parametrize(
    "arg,fmt_name",
    [(3.0, "float16"), (3.0, "bfloat16"), (3.0, "float32"), (3.0, "float64"),
     (-7.5, "float32"), (0.25, "float64"), (88.0, "float32")],
)
def test_a_correctly_rounded_budget_costs_exactly_nothing(arg, fmt_name):
    """A correctly-rounded libm needs no slack: round-to-nearest is
    monotone and the box is rounded outward onto the format grid anyway.
    So a 0.5-ulp declaration must leave the bracket byte-identical to what
    the pre-fix code produced — the mechanism must not punish a good
    platform. This is `interval.sqrt`'s argument generalised."""
    fmt = _FLOAT_FORMATS[fmt_name]
    mn = _ieee_format_min_normal(fmt)
    a = iv.IntervalArray(shape=(), los=(arg,), his=(arg,))
    pre_fix = _ieee_round_box(iv.subnormal_haze_fmt(iv.exp(a), mn)[0], fmt)
    at_half = _exp_box(arg, fmt_name, 0.5)
    assert (at_half.los, at_half.his) == (pre_fix.los, pre_fix.his)


def test_the_shipped_profile_declares_only_bfloat16_correctly_rounded():
    """float16 `exp` is NOT correctly rounded on this backend and the
    profile used to say it was (audit 0.2.0 B4). 2 of the 63,487 arguments
    measure 0.500028 ulps, so a 0.5 declaration — which
    `_libm_widen_box` honours by widening NOTHING — states a bound the
    backend it was measured on violates. The re-measurement is in
    `test_the_float16_exp_escapes_are_re_measured_against_the_backend`;
    this pins what the profile concluded from it."""
    b = LIBM_PROFILES[PROFILE]
    assert b.get("exp", "float16") == 1.0
    assert b.get("exp", "bfloat16") == 0.5
    assert "NOT correctly rounded" in LIBM_MEASURED[("exp", "float16")]
    assert "CORRECTLY ROUNDED" in LIBM_MEASURED[("exp", "bfloat16")]
    # and every exhaustive exp row carries the qualifier that makes it true
    for d in ("float16", "bfloat16", "float32"):
        prose = LIBM_MEASURED[("exp", d)]
        assert "EXHAUSTIVE" in prose, d
        assert "result is normal and finite" in prose, d


@pytest.mark.parametrize("x", (0.0226898193359375, 0.007297515869140625))
def test_the_float16_exp_escapes_are_re_measured_against_the_backend(x):
    """Re-measured live against a 60-decimal-digit reference, like every
    other escaping argument in this file: a row cannot pass by agreeing
    with a stale number. These two arguments are why `("exp","float16")`
    cannot declare 0.5 — and `_libm_widen_box` honours a 0.5 declaration
    by widening NOTHING, so the shortfall would have had no cover at all
    beyond the ±1-binary64-ulp bump."""
    import numpy as np
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = 60
        t = Decimal(x).exp()
    fmt = _FLOAT_FORMATS["float16"]
    b = float(np.asarray(jnp.exp(jnp.asarray(np.float16(x)))))
    err = abs(Decimal(b) - t) / Decimal(_libm_ulp_at(float(t), fmt))
    assert err > Decimal("0.5"), (x, err)
    assert err < Decimal("0.5001"), (x, err)      # it misses by a hair
    assert LIBM_PROFILES[PROFILE].get("exp", "float16") >= float(err)


def test_the_bfloat16_row_needs_its_normal_and_finite_qualifier():
    """The row claims correct rounding, and it is true only over results
    that are NORMAL and finite. At x=-87.5 the true value is a bfloat16
    SUBNORMAL and this backend flushes it to 0.0 — 108.7 ulps out, 217
    times the declared 0.5. What covers that is `subnormal_haze_fmt`,
    which hulls the box with 0 whenever it reaches the format's subnormal
    band, and not the accuracy budget (audit 0.2.0 B4)."""
    import numpy as np
    import ml_dtypes
    from decimal import Decimal, localcontext

    fmt = _FLOAT_FORMATS["bfloat16"]
    x = -87.5
    with localcontext() as ctx:
        ctx.prec = 60
        t = Decimal(x).exp()
    b = float(np.asarray(jnp.exp(jnp.asarray(np.asarray(x, ml_dtypes.bfloat16)))))
    assert b == 0.0 and float(t) > 0.0            # flushed
    assert float(t) < _ieee_format_min_normal(fmt)  # ... because subnormal
    err = abs(Decimal(b) - t) / Decimal(_libm_ulp_at(float(t), fmt))
    assert err > Decimal("100"), err
    # the budget does not cover it; the haze does
    box = _exp_box(x, "bfloat16", LIBM_PROFILES[PROFILE].get("exp", "bfloat16"))
    assert box.los[0] == 0.0 <= b <= box.his[0], (box.los[0], box.his[0])


def test_a_half_ulp_budget_is_read_as_CORRECT_ROUNDING_not_as_the_inequality():
    """`ulps <= 0.5` widens by nothing, and the justification is CORRECT
    ROUNDING — not "<= 0.5 ulps" read through this module's binade
    convention, which is strictly weaker (audit 0.2.0 B4).

    `_libm_ulp_at(2**k)` is the spacing ABOVE `2**k`, while the float
    BELOW it is only `2**(k-p)` away — half such an ulp. So the LITERAL
    inequality at u=0.5 also admits a backend returning `nextdown(2**k)`
    where the true value is `2**k`, which correct rounding does not.
    `exp(0) = 1.0` reaches it. The residual is covered for THIS module's
    callers, and only by them: `iv.exp` hands over a box already bumped a
    binary64 ulp outward, so the format-rounded lower endpoint sits one
    format step below."""
    import numpy as np
    import ml_dtypes

    npdt = {"float16": np.float16, "bfloat16": ml_dtypes.bfloat16,
            "float32": np.float32, "float64": np.float64}
    for fmt_name in ("float16", "bfloat16", "float32", "float64"):
        fmt = _FLOAT_FORMATS[fmt_name]
        dt = npdt[fmt_name]
        one = dt(1.0)
        below = float(np.nextafter(one, dt(-np.inf)))
        # the literal reading admits it: it is exactly half an ulp away
        assert abs(below - 1.0) == 0.5 * _libm_ulp_at(1.0, fmt), fmt_name
        # the widening itself does NOT reach it — the branch is a no-op
        w = _libm_widen_box(
            iv.IntervalArray(shape=(), los=(1.0,), his=(1.0,)), fmt, 0.5,
            floor=0.0)
        assert w.los[0] == 1.0 > below, fmt_name
        # what covers it is the caller's outward bump plus the format round
        box = _exp_box(0.0, fmt_name, 0.5)
        assert box.los[0] <= below, (fmt_name, box.los[0], below)


def test_a_faithful_budget_does_cost_and_a_measured_one_costs_more():
    """The price, in float32 grid steps at a point argument. Monotone in
    the declared ulps, which is what a budget is supposed to buy."""
    widths = [
        _steps32(_exp_box(3.0, "float32", u))
        for u in (0.5, 1.0, 2.0, 6.0)
    ]
    assert widths == sorted(widths)
    assert widths[0] < widths[-1]


def _steps32(box):
    import numpy as np

    n, x = 0, np.float32(box.los[0])
    while x < np.float32(box.his[0]) and n < 10000:
        x = np.nextafter(x, np.float32(np.inf))
        n += 1
    return n


# -- real mode is untouched ---------------------------------------------------


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64"))
def test_real_mode_still_judges_exp_with_no_budget_at_all(dtype):
    """Real mode's bracket is about the TRUE REAL value, the host's math
    module does satisfy the ±1-ulp assumption it rides on, and the
    divergence from the compiled program is the ℝ-versus-float gap the
    stamp already names. Nothing here touches it."""

    def h():
        x = any_array((), dtype, (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    v = check(h, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"
    assert any("faithfully-rounded libm exp" in a for a in v.stamp.assumptions)
    assert not any("DECLARED, NOT VERIFIED" in a for a in v.stamp.assumptions)


def test_a_budget_under_real_semantics_is_refused_not_ignored():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    with pytest.raises(ValueError) as e:
        check(h, vacuity_mode="inputs-only", libm_budget=PROFILE)
    assert "no meaning under semantics='real'" in str(e.value)


def test_real_mode_exp_boxes_are_byte_identical_to_the_pre_fix_kernel():
    """`iv.exp` itself is untouched: the widening lives in the ieee
    transfer, and real mode never calls it."""
    a = iv.IntervalArray(shape=(), los=(-3.0,), his=(2.5,))
    box = iv.exp(a)
    assert box.los[0] == math.nextafter(math.exp(-3.0), 0.0)
    assert box.his[0] == math.nextafter(math.exp(2.5), math.inf)


# -- the stamp is blunt about what it transfers -------------------------------


def test_the_stamp_says_declared_not_verified():
    """**The single most important sentence in this change.** A budget that
    is too small mints a false VERIFIED and stelling cannot check it."""

    def h():
        x = any_array((), "float32", (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    v = _ieee(h, libm_budget=PROFILE)
    line = [a for a in v.stamp.assumptions if "libm accuracy" in a]
    assert len(line) == 1
    (line,) = line
    assert "DECLARED, NOT VERIFIED" in line
    assert PROFILE in line
    assert "exp@float32 <= 6 ulps" in line
    assert "stelling checks NEITHER" in line
    assert "may EXCLUDE the value the program computes" in line
    assert "a VERIFIED resting on it is FALSE" in line
    assert "measured 2026-08-15 on jax 0.11.0" in line


def test_the_stamp_names_only_the_pairs_the_run_consumed():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    (line,) = [a for a in _ieee(h, libm_budget=PROFILE).stamp.assumptions
               if "libm accuracy" in a]
    # the DECLARED list is the prefix before the explanation; the basis
    # after it names every format the profile measured, on purpose
    declared = line.split(". TWO claims")[0]
    assert "exp@float64 <= 2 ulps" in declared
    assert "float32" not in declared and "pow@" not in declared


def test_a_run_that_uses_no_libm_op_stamps_no_budget_line():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(x + x > 0.0)

    got = _ieee(h, libm_budget=PROFILE).stamp.assumptions
    assert not [a for a in got if "libm accuracy" in a]


def test_the_ieee_stamp_no_longer_carries_the_bare_binary64_libm_sentence():
    """The sentence audit 0.2.0 S9 called out: `EXP_LIBM_ASSUMPTION`
    asserts a property of the ANALYSIS HOST's libm and was stamped, alone,
    under a verdict about the target's."""

    def h():
        x = any_array((), "float32", (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    got = _ieee(h, libm_budget=PROFILE).stamp.assumptions
    assert iv.EXP_LIBM_ASSUMPTION not in got


# -- the object, and the profile ----------------------------------------------


def test_the_shipped_profile_covers_every_op_and_format():
    b = LIBM_PROFILES[PROFILE]
    want = {(op, d) for op in LIBM_BUDGET_OPS for d in _FLOAT_FORMATS}
    assert {k for k, _ in b.ulps} == want
    assert set(LIBM_MEASURED) == want


def test_every_shipped_budget_is_at_least_its_measured_maximum():
    """The profile's numbers and the measurements they came from are two
    tables, and this is the one place they are made to agree. The maxima
    are parsed out of the measurement prose so that editing one without
    the other reddens here."""
    import re

    b = LIBM_PROFILES[PROFILE]
    checked = 0
    for key, prose in LIBM_MEASURED.items():
        found = re.findall(r"max error ([0-9.]+) ulps", prose)
        # exactly one, so a row cannot state a second, larger maximum for
        # a wider population and have the parse silently take the first —
        # the bfloat16 row quotes 108.698176 for the flushed-subnormal
        # population and 0.499988 for the one the budget covers
        assert len(found) == 1, (key, found)
        measured = float(found[0])
        declared = b.get(*key)
        assert declared >= measured, (key, declared, measured)
        checked += 1
    assert checked == 8


# -- the signature census that decides who gets handed a budget ---------------
#
# `_assert_libm_transfers_take_a_budget` had ZERO test references (audit
# 0.2.0 B4): functional, load-bearing at import, and unpinned. The
# dispatcher picks the calling convention from the TIER, so both directions
# of disagreement are real bugs and both are import-time errors here.


def test_the_signature_census_passes_on_the_live_registry():
    from stelling.propagate import _assert_libm_transfers_take_a_budget

    _assert_libm_transfers_take_a_budget()


def test_the_signature_census_bites_on_a_libm_transfer_with_no_budget_param(
    monkeypatch
):
    """A `sound-libm` transfer with the four-argument signature would raise
    `TypeError` out of the walk on first contact."""
    import stelling.propagate as P

    def four(eqn, params, ins, flags):  # pragma: no cover - never called
        raise AssertionError

    monkeypatch.setattr(
        "stelling.propagate.IEEE_TRANSFERS",
        {**P.IEEE_TRANSFERS, "exp": (four, P.TIER_SOUND_LIBM)},
    )
    with pytest.raises(RuntimeError) as e:
        P._assert_libm_transfers_take_a_budget()
    assert "exp" in str(e.value) and "4 params" in str(e.value)


def test_the_signature_census_bites_on_a_non_libm_transfer_taking_a_budget(
    monkeypatch
):
    """...and the other direction: registered at another tier while taking
    a budget, it would never be handed one and would ride whatever default
    it wrote."""
    import stelling.propagate as P

    def five(eqn, params, ins, flags, budget):  # pragma: no cover
        raise AssertionError

    monkeypatch.setattr(
        "stelling.propagate.IEEE_TRANSFERS",
        {**P.IEEE_TRANSFERS, "add": (five, P.TIER_EXACT)},
    )
    with pytest.raises(RuntimeError) as e:
        P._assert_libm_transfers_take_a_budget()
    assert "add" in str(e.value) and "5 params" in str(e.value)


def test_the_signature_census_bites_when_no_transfer_rides_the_tier(
    monkeypatch
):
    """The gate guarding nothing is the failure mode a rename produces:
    `LIBM_BUDGET_OPS` stays a description of the registry, not a wish."""
    import stelling.propagate as P

    monkeypatch.setattr("stelling.propagate.LIBM_BUDGET_OPS", frozenset())
    with pytest.raises(RuntimeError) as e:
        P._assert_libm_transfers_take_a_budget()
    assert "guards nothing" in str(e.value)


def test_every_sampled_row_names_its_draw():
    """A SAMPLED maximum is a property of the SAMPLE, not of the backend,
    so the row has to say which sample. Without that a reader who re-runs
    and gets a different number cannot tell *"the row is stale"* from
    *"your draw differs"* — which is exactly how `exp@float64` was
    mis-flagged in the first round of this audit (audit 0.2.0 B4).

    Every sampled row names the seed of the draw that can be re-run and
    names the other draw. The two rows whose carried figure comes from the
    draw that CANNOT be re-run say so, in the row, beside the figure."""
    import re

    sampled = {k: v for k, v in LIBM_MEASURED.items()
               if not v.startswith("EXHAUSTIVE")}
    assert set(sampled) == {("exp", "float64"), ("pow", "float16"),
                            ("pow", "bfloat16"), ("pow", "float32"),
                            ("pow", "float64")}, sorted(sampled)
    for key, prose in sampled.items():
        assert "sampled" in prose, key
        assert "20260815" in prose, key
        assert re.search(r"[Dd]raw [AB]|earlier draw", prose), key

    # the larger of the two draws is what each row keeps; on these two that
    # is the unreproducible one, and the row must not hide it
    for key in (("pow", "float32"), ("pow", "float64"), ("exp", "float64")):
        assert "not recorded" in LIBM_MEASURED[key].lower(), key


def test_an_unknown_profile_name_raises_where_it_was_written():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(jnp.exp(x) > 0.0)

    with pytest.raises(ValueError) as e:
        _ieee(h, libm_budget="xla-gpu-2099-01")
    assert PROFILE in str(e.value)


def test_resolve_accepts_none_a_name_and_a_budget_and_nothing_else():
    b = LIBM_PROFILES[PROFILE]
    assert resolve_libm_budget(None) is None
    assert resolve_libm_budget(PROFILE) is b
    assert resolve_libm_budget(b) is b
    with pytest.raises(TypeError):
        resolve_libm_budget(6.0)


@pytest.mark.parametrize(
    "kwargs,exc,fragment",
    [
        (dict(name="", basis="measured somewhere", ulps={("exp", "float64"): 1}),
         ValueError, "non-empty string"),
        (dict(name="n", basis="short", ulps={("exp", "float64"): 1}),
         ValueError, "must say what was measured"),
        (dict(name="n", basis="measured somewhere", ulps={}),
         ValueError, "declares nothing"),
        (dict(name="n", basis="measured somewhere", ulps={("sqrt", "float64"): 1}),
         ValueError, "does not ride a libm accuracy assumption"),
        (dict(name="n", basis="measured somewhere", ulps={("exp", "float128"): 1}),
         ValueError, "not a catalogued float format"),
        (dict(name="n", basis="measured somewhere", ulps={("exp", "float64"): -1}),
         ValueError, "finite non-negative"),
        (dict(name="n", basis="measured somewhere",
              ulps={("exp", "float64"): float("inf")}),
         ValueError, "finite non-negative"),
        (dict(name="n", basis="measured somewhere", ulps={("exp", "float64"): "6"}),
         TypeError, "must be a number of ulps"),
        (dict(name="n", basis="measured somewhere", ulps={("exp", "float64"): True}),
         TypeError, "must be a number of ulps"),
    ],
)
def test_a_budget_refuses_what_it_cannot_be(kwargs, exc, fragment):
    with pytest.raises(exc) as e:
        LibmBudget(**kwargs)
    assert fragment in str(e.value)


def test_a_budget_is_hashable_and_normalised():
    a = LibmBudget(name="n", basis="measured somewhere",
                   ulps={("pow", "float64"): 1, ("exp", "float64"): 2})
    b = LibmBudget(name="n", basis="measured somewhere",
                   ulps=[(("exp", "float64"), 2.0), (("pow", "float64"), 1.0)])
    assert a == b and hash(a) == hash(b)
    assert a.ulps == ((("exp", "float64"), 2.0), (("pow", "float64"), 1.0))


def test_sqrt_needs_no_budget_and_that_is_the_line_this_draws():
    """`interval.sqrt`'s docstring already draws the correct line: sqrt is
    a CORRECTLY-ROUNDED IEEE-754 basic operation, so a 1-ulp bump contains
    the true root with a full ulp to spare and no libm-fidelity demotion
    applies. It is registered at `TIER_EXACT` under ieee, so the budget
    gate does not reach it — which is the generalisation, not an
    exception: the gate is exactly the set of `TIER_SOUND_LIBM` rows."""
    assert LIBM_BUDGET_OPS == frozenset({"exp", "pow"})

    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return assert_(jnp.sqrt(x) > 0.0)

    assert _ieee(h).status == "VERIFIED"


# -- the widening math --------------------------------------------------------


@pytest.mark.parametrize(
    "fmt_name,x,want",
    [("float64", 1.0, 2.0 ** -52), ("float64", 2.0, 2.0 ** -51),
     ("float32", 1.0, 2.0 ** -23), ("float32", 3.0, 2.0 ** -22),
     ("float16", 1.0, 2.0 ** -10), ("bfloat16", 1.0, 2.0 ** -7),
     ("float64", 0.0, 2.0 ** -1074), ("float32", 1e-45, 2.0 ** -149)],
)
def test_the_ulp_helper_is_the_binade_spacing(fmt_name, x, want):
    assert _libm_ulp_at(x, _FLOAT_FORMATS[fmt_name]) == want


def test_widening_never_narrows_and_never_crosses_zero_downward():
    fmt = _FLOAT_FORMATS["float32"]
    for lo, hi in ((0.0, 1.0), (1e-40, 1e-38), (1.0, 1e30), (0.0, 0.0)):
        a = iv.IntervalArray(shape=(), los=(lo,), his=(hi,))
        w = _libm_widen_box(a, fmt, 6.0, floor=0.0)
        assert w.los[0] <= lo and w.his[0] >= hi
        assert w.los[0] >= 0.0


def test_widening_leaves_infinite_endpoints_alone():
    """An infinite endpoint stays infinite — AND the FINITE one still owes
    the contract, which is the half this test never checked (audit 0.2.0
    B4). A half-infinite box holds every ``t`` beyond its finite end, so
    that endpoint has to clear ``t -+ u*ulp(t)`` for ALL of them; asserting
    only ``his[0] == inf`` is a scope that does not cover the claim.

    **And the box has to be one where the defect can appear.** ``[0, inf]``
    — the box this test used — is not: at ``lo = 0`` the infimum of
    ``t - u*ulp(t)`` really is at ``t = 0``, so the rule with the bug in it
    passes. The box that bites is anchored just BELOW a power of two,
    where the first binade boundary above ``lo`` carries twice ``lo``'s
    spacing."""
    for fmt_name in ("float16", "bfloat16", "float32", "float64"):
        fmt = _FLOAT_FORMATS[fmt_name]
        for k in (-8, 0, 1, 8, 64, 512):
            lo = math.nextafter(2.0 ** k, 0.0)
            a = iv.IntervalArray(shape=(), los=(lo,), his=(math.inf,))
            w = _libm_widen_box(a, fmt, 6.0, floor=None)
            assert w.his[0] == math.inf
            for j in range(k, 1024, 137):
                t = 2.0 ** j
                assert w.los[0] <= t - 6.0 * _libm_ulp_at(t, fmt), (
                    fmt_name, k, j)
            # the mirror arm, latent today because exp/pow pass floor=0.0
            hi = -lo
            b = iv.IntervalArray(shape=(), los=(-math.inf,), his=(hi,))
            w2 = _libm_widen_box(b, fmt, 6.0, floor=None)
            assert w2.los[0] == -math.inf
            for j in range(k, 1024, 137):
                t = -(2.0 ** j)
                assert w2.his[0] >= t + 6.0 * _libm_ulp_at(t, fmt), (
                    fmt_name, k, j)


def test_a_budget_past_two_to_the_p_minus_one_unbounds_a_half_infinite_box():
    """The side condition on the doubling. ``g(2**k) = 2**k*(1 -
    u*2**(1-p))`` rises with ``k`` only while ``u <= 2**(p-1)``; past that
    it falls without bound and NO finite lower endpoint is sound.

    A caller reaches that threshold by ROUNDING UP, never by measuring —
    the checkable half of which is
    `test_a_flush_to_zero_can_never_measure_up_to_the_threshold`."""
    for fmt_name in ("float16", "bfloat16", "float32", "float64"):
        fmt = _FLOAT_FORMATS[fmt_name]
        thr = 2.0 ** (fmt[0] - 1)
        a = iv.IntervalArray(shape=(), los=(1.0,), his=(math.inf,))
        at = _libm_widen_box(a, fmt, thr, floor=None)
        assert math.isfinite(at.los[0]), (fmt_name, "finite at the threshold")
        over = _libm_widen_box(a, fmt, math.nextafter(thr, math.inf),
                               floor=None)
        assert over.los[0] == -math.inf, (fmt_name, over.los[0])
        # ... and `floor` still states the RANGE rather than being lost
        floored = _libm_widen_box(a, fmt, thr * 4.0, floor=0.0)
        assert floored.los[0] == 0.0, (fmt_name, floored.los[0])
        # the mirror arm saturates upward
        b = iv.IntervalArray(shape=(), los=(-math.inf,), his=(-1.0,))
        assert _libm_widen_box(b, fmt, thr * 4.0, floor=None).his[0] == math.inf


def test_a_flush_to_zero_can_never_measure_up_to_the_threshold():
    """Which way round the ``2**(p-1)`` threshold sits, checked in exact
    rationals rather than asserted in prose — the sentence arguing its
    reachability was inverted twice before this test existed.

    The worst error this module names is a subnormal result FLUSHED to
    zero. `_libm_ulp_at` floors at ``tiny``, so that error is ``t/tiny``
    for a true value ``t`` below ``2**emin``. The largest REPRESENTABLE
    subnormal is ``tiny*(2**(p-1) - 1)``, so a flush measures at most
    ``2**(p-1) - 1`` ulps — exactly one ulp UNDER the threshold, in every
    format; over the open subnormal band the bound is ``< 2**(p-1)``, a
    supremum that is not attained. The threshold is therefore just ABOVE
    anything observable, and a caller crosses it only by rounding a
    measurement up."""
    from fractions import Fraction

    caps = {}
    for fmt_name, fmt in _FLOAT_FORMATS.items():
        p, emin, _emax = fmt
        tiny = Fraction(2) ** (emin - p + 1)
        min_normal = Fraction(2) ** emin
        threshold = Fraction(2) ** (p - 1)

        # the helper really does floor a subnormal's ulp at `tiny`, which
        # is what makes t/tiny the error in ulps
        largest_sub = min_normal - tiny
        assert _libm_ulp_at(float(largest_sub), fmt) == float(tiny), fmt_name
        assert _libm_ulp_at(float(min_normal), fmt) == float(tiny), fmt_name

        cap = largest_sub / tiny                    # exact, in ulps
        assert cap == threshold - 1, (fmt_name, cap, threshold)
        assert cap < threshold
        assert min_normal / tiny == threshold       # the open-band supremum
        caps[fmt_name] = (int(cap), int(threshold))

    assert caps == {
        "float16": (1023, 1024),
        "bfloat16": (127, 128),
        "float32": (8388607, 8388608),
        "float64": (4503599627370495, 4503599627370496),
    }, caps

    # and the flush this backend actually produces is under BOTH
    bf = _FLOAT_FORMATS["bfloat16"]
    measured = 9.982350930569248e-39 / _libm_ulp_at(9.982350930569248e-39, bf)
    assert 108.0 < measured < 109.0, measured
    assert measured < caps["bfloat16"][0] < caps["bfloat16"][1]


def test_widening_saturates_rather_than_producing_a_nan_endpoint():
    fmt = _FLOAT_FORMATS["float64"]
    big = math.nextafter(math.inf, 0.0)
    a = iv.IntervalArray(shape=(), los=(big,), his=(big,))
    w = _libm_widen_box(a, fmt, 6.0, floor=0.0)
    assert w.his[0] == math.inf and math.isfinite(w.los[0])


# -- ulp is a STEP function, and the widening has to know it ------------------

# The exact number of endpoint obligations the binade sweep below
# discharges. Asserted rather than bounded, like LIBM_EXP_SWEEP_CHECKS.
LIBM_BINADE_SWEEP_CHECKS = 372


def _binade_exponents(fmt):
    p, emin, emax = fmt
    ks = [emin + p, emin + p + 8, -8, 0, 1, 8, emax - 8, emax - 1]
    return sorted({k for k in ks if emin + p <= k <= emax - 1})


def test_the_widening_covers_every_binade_boundary_a_box_straddles():
    """**ulp is a STEP function of the magnitude.** It doubles at every
    power of two, so ``t - u*ulp(t)`` is NOT monotone in ``t`` — it drops
    by half an ulp each time ``t`` crosses one upward. A box straddling
    ``2**k`` from below therefore admits values as low as
    ``2**k - u*2**(k-p+1)``, while ``ulp(lo)`` just under the boundary is
    only ``2**(k-p)`` — half of what the widening needs. So one spacing
    has to serve both endpoints and it has to be the LARGER.
    """
    checked = 0
    for fmt_name in ("float16", "bfloat16", "float32", "float64"):
        fmt = _FLOAT_FORMATS[fmt_name]
        for k in _binade_exponents(fmt):
            b = 2.0 ** k
            lo = math.nextafter(b, 0.0)
            for u in (1.0, 2.0, 6.0):
                w = _libm_widen_box(
                    iv.IntervalArray(shape=(), los=(lo,), his=(b,)),
                    fmt, u, floor=None,
                )
                for t in (lo, b):
                    assert w.los[0] <= t - u * _libm_ulp_at(t, fmt), (
                        fmt_name, k, u, t)
                    assert w.his[0] >= t + u * _libm_ulp_at(t, fmt), (
                        fmt_name, k, u, t)
                    checked += 2
    assert checked == LIBM_BINADE_SWEEP_CHECKS, checked


def test_the_binade_sweep_bites_against_a_per_endpoint_widening():
    """The positive control. The natural-looking rule — each endpoint
    widened by its OWN ulp — fails the row above at every boundary, so a
    green there is a property of the rule and not of the sample. This is
    the shape the first draft of `_libm_widen_box` had."""

    def per_endpoint(lo, hi, fmt, ulps):
        return (
            math.nextafter(lo - ulps * _libm_ulp_at(lo, fmt), -math.inf),
            math.nextafter(hi + ulps * _libm_ulp_at(hi, fmt), math.inf),
        )

    boundaries = [
        (f, k)
        for f in ("float16", "bfloat16", "float32", "float64")
        for k in _binade_exponents(_FLOAT_FORMATS[f])
    ]
    misses = 0
    for fmt_name, k in boundaries:
        fmt = _FLOAT_FORMATS[fmt_name]
        b = 2.0 ** k
        wlo, _whi = per_endpoint(math.nextafter(b, 0.0), b, fmt, 6.0)
        if not wlo <= b - 6.0 * _libm_ulp_at(b, fmt):
            misses += 1
    assert misses == len(boundaries) == 31, (misses, len(boundaries))


# -- the same step function, on the arm the row above does not reach ----------
#
# `test_the_widening_covers_every_binade_boundary_a_box_straddles` uses only
# FINITE boxes, and `test_widening_leaves_infinite_endpoints_alone` used the
# one half-infinite box where the defect cannot appear. So the rule shipped
# with a hole in it and nothing went red (audit 0.2.0 B4). What closes that
# is not another example: it is enumerating the whole extremiser set.
#
# `t -> t -+ u*ulp(t)` is affine with slope 1 inside a binade, so an extremum
# over a box sits at an endpoint or at a jump of `ulp`. `_extremisers` below
# holds EVERY jump the binary64 grid can express — +-2**k for every k the
# format's ulp formula distinguishes, up to 2**1023, with the float on each
# side of it — so a box's true infimum and supremum are exactly a min/max
# over (its endpoints + the candidates it contains). For a half-infinite box
# that is a SUFFIX min / PREFIX max, which is what makes the +-inf arm
# checkable at all instead of samplable.

# Pinned, like LIBM_BINADE_SWEEP_CHECKS: the extremiser points the sweep
# reduces over, per arm. Asserted rather than bounded.
LIBM_EXTREMISER_POINTS = {"finite": 1551584, "half": 34844544,
                          "both-inf": 525888}


def _extremisers(fmt):
    """Every ulp jump the binary64 grid can express, both sides, sorted,
    with its ulp — computed once per format."""
    p, emin, _emax = fmt
    big = math.nextafter(math.inf, 0.0)
    pts = {0.0, big, -big}
    for k in range(emin - p + 1, 1024):
        b = math.ldexp(1.0, k)
        for s in (b, -b):
            pts.add(s)
            pts.add(math.nextafter(s, -math.inf))
            pts.add(math.nextafter(s, math.inf))
    ts = sorted(pts)
    return ts, [_libm_ulp_at(t, fmt) for t in ts]


def _sweep_anchors(fmt):
    p, emin, emax = fmt
    out = {0.0}
    for k in (emin - p + 1, emin - p + 4, emin, emin + 4, -8, -1, 0, 1, 8,
              emax - 4, emax):
        b = math.ldexp(1.0, k)
        for v in (b, math.nextafter(b, 0.0), math.nextafter(b, math.inf)):
            out.add(v)
            out.add(-v)
    return sorted(out)


def _sweep_boxes(fmt):
    a = _sweep_anchors(fmt)
    out = []
    for i, lo in enumerate(a):
        out.append((lo, lo))
        out.append((lo, a[min(i + 1, len(a) - 1)]))
        out.append((lo, a[min(i + 5, len(a) - 1)]))
    out += [(lo, math.inf) for lo in a]
    out += [(-math.inf, hi) for hi in a]
    out.append((-math.inf, math.inf))
    return [(lo, hi) for lo, hi in out if lo <= hi]


def _sweep_budgets(fmt):
    """Ordinary budgets, then the ``2**(p-1)`` threshold the doubling's
    side condition turns on, then past it."""
    thr = math.ldexp(1.0, fmt[0] - 1)
    return (0.75, 1.0, 2.0, 6.0, thr, math.nextafter(thr, math.inf),
            thr * 1.5, thr * 4096.0)


def _run_widen_sweep(rule):
    """(points driven, violations) per arm, over every format, budget and
    box. `rule(lo, hi, fmt, u) -> (widened_lo, widened_hi)`."""
    import bisect

    driven = {"finite": 0, "half": 0, "both-inf": 0}
    bad = {"finite": 0, "half": 0, "both-inf": 0}
    first = None
    for fmt in _FLOAT_FORMATS.values():
        ts, us = _extremisers(fmt)
        n = len(ts)
        for u in _sweep_budgets(fmt):
            g = [t - u * uu for t, uu in zip(ts, us)]
            h = [t + u * uu for t, uu in zip(ts, us)]
            suf_g = [math.inf] * (n + 1)
            for i in range(n - 1, -1, -1):
                suf_g[i] = min(g[i], suf_g[i + 1])
            pre_h = [-math.inf] * (n + 1)
            for i in range(n):
                pre_h[i + 1] = max(h[i], pre_h[i])
            for lo, hi in _sweep_boxes(fmt):
                lo_inf, hi_inf = math.isinf(lo), math.isinf(hi)
                kind = ("both-inf" if lo_inf and hi_inf
                        else "finite" if not (lo_inf or hi_inf) else "half")
                i = bisect.bisect_left(ts, lo)
                j = bisect.bisect_right(ts, hi)
                need_lo, need_hi = math.inf, -math.inf
                for v in (lo, hi):
                    if math.isfinite(v):
                        uu = _libm_ulp_at(v, fmt)
                        need_lo = min(need_lo, v - u * uu)
                        need_hi = max(need_hi, v + u * uu)
                if hi_inf:
                    need_lo, need_hi = min(need_lo, suf_g[i]), math.inf
                elif lo_inf:
                    need_lo, need_hi = -math.inf, max(need_hi, pre_h[j])
                else:
                    for k in range(i, j):
                        need_lo = min(need_lo, g[k])
                        need_hi = max(need_hi, h[k])
                driven[kind] += 2 * max(
                    (j - i) if kind == "finite"
                    else ((n - i) if hi_inf else j) + 2, 1)
                wlo, whi = rule(lo, hi, fmt, u)
                if not (wlo <= need_lo and whi >= need_hi):
                    bad[kind] += 1
                    if first is None:
                        first = (fmt, u, (lo, hi), (wlo, whi),
                                 (need_lo, need_hi))
    return driven, bad, first


def test_the_widening_covers_the_whole_extremiser_set_on_every_arm():
    """The obligation, enumerated rather than sampled, over finite,
    half-infinite and doubly-infinite boxes, all four formats, budgets from
    0.75 up past ``2**(p-1)``."""
    driven, bad, first = _run_widen_sweep(
        lambda lo, hi, fmt, u: (
            lambda b: (b.los[0], b.his[0])
        )(_libm_widen_box(
            iv.IntervalArray(shape=(), los=(lo,), his=(hi,)),
            fmt, u, floor=None))
    )
    assert bad == {"finite": 0, "half": 0, "both-inf": 0}, first
    assert driven == LIBM_EXTREMISER_POINTS, driven


def test_the_extremiser_sweep_bites_against_all_three_wrong_rules():
    """The positive control, and it is the load-bearing half of this
    finding: the shipped rule going green proves nothing unless the rules
    it replaced go RED on the same obligations. All three did ship or were
    drafted.

    A: ``max`` over the FINITE endpoints only — correct on a finite box and
       silently wrong the moment one endpoint is ``inf``. This is what
       B4 found.
    B: each endpoint widened by its OWN ulp — the first draft; wrong on
       BOTH arms.
    C: the doubling WITHOUT the ``u <= 2**(p-1)`` side condition — the
       shape proposed with this finding; sound up to the threshold and
       unbounded-wrong past it.
    """
    nxt = math.nextafter

    def finite_max_only(lo, hi, fmt, u):
        fin = [v for v in (lo, hi) if math.isfinite(v)]
        if not fin:
            return lo, hi
        w = u * max(_libm_ulp_at(v, fmt) for v in fin)
        return (nxt(lo - w, -math.inf) if math.isfinite(lo) else lo,
                nxt(hi + w, math.inf) if math.isfinite(hi) else hi)

    def per_endpoint(lo, hi, fmt, u):
        return (nxt(lo - u * _libm_ulp_at(lo, fmt), -math.inf)
                if math.isfinite(lo) else lo,
                nxt(hi + u * _libm_ulp_at(hi, fmt), math.inf)
                if math.isfinite(hi) else hi)

    def doubled_without_side_condition(lo, hi, fmt, u):
        fin = [v for v in (lo, hi) if math.isfinite(v)]
        if not fin:
            return lo, hi
        s = max(_libm_ulp_at(v, fmt) for v in fin)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            s *= 2.0
        w = u * s
        return (nxt(lo - w, -math.inf) if math.isfinite(lo) else lo,
                nxt(hi + w, math.inf) if math.isfinite(hi) else hi)

    _d, bad_a, _f = _run_widen_sweep(finite_max_only)
    assert bad_a["half"] == 1886, bad_a
    assert bad_a["finite"] == 0, bad_a      # correct on the arm it covered

    _d, bad_b, _f = _run_widen_sweep(per_endpoint)
    assert bad_b["finite"] == 1552 and bad_b["half"] == 1886, bad_b

    _d, bad_c, first_c = _run_widen_sweep(doubled_without_side_condition)
    assert bad_c["finite"] == 0 and bad_c["half"] == 1538, bad_c
    # and it first goes wrong at the very first budget past the threshold
    assert first_c[1] == math.nextafter(2.0 ** (first_c[0][0] - 1), math.inf)


def test_a_backend_inside_the_declared_budget_is_not_excluded_over_inf():
    """End to end, on the shipped profile: an ordinary float32 envelope
    whose `iv.exp` overflows to ``+inf`` (any envelope reaching past
    709.78 does). Before B4 the propagated lower endpoint sat ABOVE values
    the declared 6-ulp budget permits, and the obligation below came back
    VERIFIED. The escaping value is re-measured here against the live
    reference rather than quoted."""
    import numpy as np

    lo_arg = 0.6931471805599453
    xp = float(np.nextafter(np.float32(lo_arg), np.float32(np.inf)))
    true = float(jnp.exp(jnp.asarray(xp, "float64")))
    fmt = _FLOAT_FORMATS["float32"]
    u = LIBM_PROFILES[PROFILE].get("exp", "float32")

    def _finish(b):
        return _ieee_round_box(
            iv.subnormal_haze_fmt(b, _ieee_format_min_normal(fmt))[0], fmt)

    raw = iv.exp(iv.IntervalArray(shape=(), los=(lo_arg,), his=(1e6,)))
    assert raw.his[0] == math.inf          # the half-infinite arm
    box = _finish(_libm_widen_box(raw, fmt, u, floor=0.0))

    # the rule B4 replaced: max over the FINITE endpoints only, so the
    # +inf endpoint drops out and the spacing falls back to ulp(lo)
    w_pre = u * _libm_ulp_at(raw.los[0], fmt)
    pre = _finish(iv.IntervalArray(
        shape=(), los=(math.nextafter(raw.los[0] - w_pre, -math.inf),),
        his=(math.inf,)))

    # the LEAST-WRONG float32 the pre-fix box excluded
    esc = np.float32(pre.los[0])
    while float(esc) >= pre.los[0]:
        esc = np.nextafter(esc, np.float32(-np.inf))
    err = abs(float(esc) - true) / _libm_ulp_at(true, fmt)

    assert err < u, err                    # inside the DECLARED budget
    assert err < 5.5112, err               # and inside the MEASURED maximum
    assert box.los[0] <= float(esc) < pre.los[0], (
        f"a backend only {err:.4f} ulps out — inside both the declared {u} "
        f"and the 5.5112 this profile measured exhaustively — returns "
        f"{float(esc)!r}, which the pre-B4 lower endpoint {pre.los[0]!r} "
        f"excluded and the fixed one {box.los[0]!r} must admit"
    )

    def h():
        x = any_array((), "float32", (lo_arg, 1e6))
        return assert_(jnp.exp(x) >= np.float32(1.9999991655349731))

    # the PRE-FIX lower endpoint, which this used to mint as VERIFIED
    assert _ieee(h, libm_budget=PROFILE).status != "VERIFIED"
