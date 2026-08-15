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
    """Fail closed, uniformly. float16 and bfloat16 are exhaustively
    CORRECTLY ROUNDED on the measured backend and still decline: "measured
    well once, on one jaxlib, on one device" is not something a verdict may
    rest on silently."""
    v, note = _decline_note(dtype, op)
    assert v.status == "UNKNOWN"
    assert op in note and dtype in note


def test_the_decline_carries_the_evidence_that_justifies_it():
    _v, note = _decline_note("float32", "exp")
    assert "EXHAUSTIVE over every float32 argument" in note
    assert "5.5112 ulps" in note
    assert "88.5463" in note and "88.7228" in note
    assert "12,542" in note


def test_the_decline_carries_the_exact_incantation():
    _v, note = _decline_note("float32", "exp")
    assert f"libm_budget='{PROFILE}'" in note
    assert "from stelling.propagate import LibmBudget" in note
    assert "name='my-backend-YYYY-MM'" in note
    assert "('exp', 'float32'): <ulps>" in note
    assert "6 ulps" in note  # the shipped number for this pair


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


def test_the_shipped_profile_declares_the_narrow_formats_correctly_rounded():
    b = LIBM_PROFILES[PROFILE]
    assert b.get("exp", "float16") == 0.5
    assert b.get("exp", "bfloat16") == 0.5
    assert "CORRECTLY ROUNDED" in LIBM_MEASURED[("exp", "float16")]
    assert "CORRECTLY ROUNDED" in LIBM_MEASURED[("exp", "bfloat16")]


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
        m = re.search(r"max error ([0-9.]+) ulps", prose)
        assert m, (key, prose)
        measured = float(m.group(1))
        declared = b.get(*key)
        assert declared >= measured, (key, declared, measured)
        checked += 1
    assert checked == 8


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
    fmt = _FLOAT_FORMATS["float64"]
    a = iv.IntervalArray(shape=(), los=(0.0,), his=(math.inf,))
    w = _libm_widen_box(a, fmt, 6.0, floor=0.0)
    assert w.his[0] == math.inf


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
