# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The non-emptiness certificate's corpus — built for this build, by me.

Each row is written ONCE against a tiny backend protocol and run two ways:

* through **stelling** (`any_array` / `assume` / `assert_` over `jnp`),
  which is the thing under test;
* through the **oracle** (`numpy` over concrete members of the declared
  set), which never touches stelling's analysis at all.

**What that shared source does and does not make independent.** It makes
the two runs agree about WHAT PROGRAM is being judged — deliberately: a
second transcription of each row by hand would add a transcription-bug
channel that has nothing to do with the claim. What is independent is the
only thing the claim is about: the DECISION PROCEDURE. stelling decides by
outward-rounded interval propagation over boxes; the oracle decides by
evaluating the program at concrete points in binary64 and counting. A
recovered refutation the oracle can satisfy at an admissible point is a
wrong refutation, and nothing shared between the two can hide that.

**Ground truth per row is a LABEL I wrote plus what the oracle measured**,
and the two are reported side by side. Where they disagree the row is a
finding, not a rounding error.

THE BALANCE IS DELIBERATE. Roughly half the withheld rows have an
INHABITED assumed region (the withholding costs a sound refutation) and
half an EMPTY one (the withholding is the whole point). A corpus of
inhabited rows only would measure the certificate's firing rate and call
it accuracy.
"""

from __future__ import annotations

# --- the backend protocol -----------------------------------------------------
#
# A row is `f(A)` where `A` carries `A.np` (jnp or numpy), `A.any(...)`,
# `A.assume(pred)` and `A.assert_(pred)`. Rows return nothing; obligations
# are whatever `A.assert_` was called with, in order.


def r01_narrow_inhabited(A):
    """UNCERTIFIED narrowing, region INHABITED: {x in [0,1] : 2x >= 0.5}
    is [0.25, 1]. One violated obligation."""
    x = A.any((), "float64", (0.0, 1.0))
    y = x * 2.0
    A.assume(y >= 0.5)
    A.assert_(x <= -1.0)


def r02_narrow_empty(A):
    """UNCERTIFIED narrowing, region EMPTY: x - x is exactly 0 while its
    box is [-1, 1]."""
    x = A.any((), "float64", (0.0, 1.0))
    y = x - x
    A.assume(y >= 0.5)
    A.assert_(x <= -1.0)


def r03_narrow_inhabited_pair(A):
    """Two obligations under one uncertified narrowing: one DISCHARGED,
    one violated. The row that makes the per-obligation ledger say
    something a per-query one could not."""
    x = A.any((), "float64", (0.0, 1.0))
    y = x * 2.0
    A.assume(y >= 0.5)
    A.assert_(x >= -5.0)
    A.assert_(x <= -1.0)


def r04_narrow_empty_pair(A):
    """The same two faces over an EMPTY region."""
    x = A.any((), "float64", (0.0, 1.0))
    y = x - x
    A.assume(y >= 0.5)
    A.assert_(x >= -5.0)
    A.assert_(x <= -1.0)


def r05_square_inhabited(A):
    """{x in [-1,1] : x^2 <= 0.25} is [-0.5, 0.5]; the box of x*x is
    [-1, 1] (correlation-blind), so the narrowing is uncertified."""
    x = A.any((), "float64", (-1.0, 1.0))
    A.assume(x * x <= 0.25)
    A.assert_(x >= 100.0)


def r06_square_empty(A):
    """{x in [-1,1] : x^2 <= -0.5} is EMPTY, and the BOX of x*x reaches
    -1, so the meet is nonempty and nothing at the box level sees it."""
    x = A.any((), "float64", (-1.0, 1.0))
    A.assume(x * x <= -0.5)
    A.assert_(x >= 100.0)


def r07_relational_inhabited(A):
    """RELATIONAL assume — dropped over boxes, decidable at a point."""
    x = A.any((), "float64", (0.0, 1.0))
    y = A.any((), "float64", (0.0, 1.0))
    A.assume(x >= y)
    A.assert_(x + y >= 100.0)


def r08_relational_empty(A):
    """The same over disjoint boxes: EMPTY."""
    x = A.any((), "float64", (0.0, 1.0))
    y = A.any((), "float64", (2.0, 3.0))
    A.assume(x >= y)
    A.assert_(x + y >= 100.0)


def r09_relational_array_inhabited(A):
    """The vacuous-refutation file's own shape, at array rank."""
    a = A.any((3,), "float64", (0.0, 10.0))
    b = A.any((3,), "float64", (5.0, 6.0))
    A.assume(a >= 0.0)
    A.assume(a <= b)
    A.assert_(a > 50.0)


def r10_relational_array_empty(A):
    """The same with the boxes moved so no member satisfies both."""
    a = A.any((3,), "float64", (7.0, 10.0))
    b = A.any((3,), "float64", (5.0, 6.0))
    A.assume(a >= 0.0)
    A.assume(a <= b)
    A.assert_(a > 50.0)


def r11_all_reduction_inhabited(A):
    """`jnp.all(...)` — `reduce_and` has no interval transfer in either
    registry, so the assumed predicate is TOP over a box AND at a point.
    Region [0.5, 1]^3, INHABITED, and the certificate cannot see it: the
    row that measures this build's largest blind spot."""
    x = A.any((3,), "float64", (-1.0, 1.0))
    A.assume(A.np.all(x >= 0.5))
    A.assert_(x > 5.0)


def r12_all_reduction_empty(A):
    """The same with an EMPTY region."""
    x = A.any((3,), "float64", (-1.0, 1.0))
    A.assume(A.np.all(x >= 2.0))
    A.assert_(x > 5.0)


def r13_or_predicate_inhabited(A):
    """An `or` predicate: dropped (not a comparison producer), decidable
    at a point. {x in [0,1] : x >= 0.9 or x <= 0.1} is INHABITED."""
    x = A.any((), "float64", (0.0, 1.0))
    A.assume((x >= 0.9) | (x <= 0.1))
    A.assert_(x >= 100.0)


def r14_or_predicate_empty(A):
    """The same, EMPTY: neither disjunct is satisfiable in [0.3, 0.7]."""
    x = A.any((), "float64", (0.3, 0.7))
    A.assume((x >= 0.9) | (x <= 0.1))
    A.assert_(x >= 100.0)


def r15_int32_inhabited(A):
    """A non-float64 declaration. {n in int32 [0,10] : 2n >= 4} is
    {2..10}, INHABITED — and the witness must be an INTEGER."""
    n = A.any((), "int32", (0.0, 10.0))
    m = n * 2
    A.assume(m >= 4)
    A.assert_(n >= 100)


def r16_int32_empty_off_grid(A):
    """{n in int32 [0,10] : 2n == 3} is EMPTY, and it is empty ONLY
    because of the dtype: a real n = 1.5 satisfies it. A search that
    probed the interval rather than the dtype's values would certify."""
    n = A.any((), "int32", (0.0, 10.0))
    m = n * 2
    A.assume(m >= 3)
    A.assume(m <= 3)
    A.assert_(n >= 100)


def r17_float32_inhabited(A):
    """A float32 declaration — the half a float64-only corpus is
    structurally blind to, since float64 is the one format that IS its own
    interval."""
    x = A.any((), "float32", (0.0, 1.0))
    y = x * 2.0
    A.assume(y >= 0.5)
    A.assert_(x <= -1.0)


def r18_float32_pair(A):
    """float32 with a discharge beside the violation."""
    x = A.any((), "float32", (0.0, 1.0))
    y = x * 2.0
    A.assume(y >= 0.5)
    A.assert_(x >= -5.0)
    A.assert_(x <= -1.0)


def r19_sqrt_slack(A):
    """A transcendental WITH slack: {x in [0,1] : sqrt(x+1) >= 1.2} is
    [0.44, 1], and the enclosure at a pinned point clears 1.2."""
    x = A.any((), "float64", (0.0, 1.0))
    A.assume(A.np.sqrt(x + 1.0) >= 1.2)
    A.assert_(x <= -1.0)


def r20_sqrt_no_slack(A):
    """A transcendental with NO slack: x is pinned at 0.25 by declaration,
    sqrt(0.25) is exactly 0.5, and stelling's enclosure straddles it."""
    x = A.any((), "float64", (0.25, 0.25))
    A.assume(A.np.sqrt(x) >= 0.5)
    A.assert_(x <= -1.0)


def r21_exp_slack(A):
    """exp, with slack. {x in [0,1] : exp(x) >= 1.5} is [0.405, 1]."""
    x = A.any((), "float64", (0.0, 1.0))
    A.assume(A.np.exp(x) >= 1.5)
    A.assert_(x <= -1.0)


def r22_exp_empty(A):
    """exp, EMPTY, and invisible at the box level. `exp(x) - exp(x)` is
    exactly 0 at every point while its BOX over [0,1] is [1-e, e-1], so
    the meet with `>= 0.5` is nonempty and only a point can see it.

    NOT `exp(x) >= 2.8`, which stelling detects at declaration time: the
    box of exp over [0,1] is [1, e] and the meet is EMPTY, so the run
    raises `UnsatisfiableAssumptionError` and never reaches the
    withholding this corpus is about. Measured, not assumed — that row
    was written first and killed the harness."""
    x = A.any((), "float64", (0.0, 1.0))
    A.assume(A.np.exp(x) - A.np.exp(x) >= 0.5)
    A.assert_(x <= -1.0)


def r23_two_assumes_one_empty(A):
    """One satisfiable assume beside an EMPTY one. A certificate that
    required only SOME assume would certify this; requiring ALL declines."""
    x = A.any((), "float64", (0.0, 1.0))
    y = x - x
    A.assume(x * 2.0 >= 0.5)
    A.assume(y >= 0.5)
    A.assert_(x <= -1.0)


def r24_two_assumes_both_live(A):
    """Two satisfiable assumes: {x in [0,1] : 2x >= 0.5 and 3x <= 2.5} is
    [0.25, 0.833], INHABITED."""
    x = A.any((), "float64", (0.0, 1.0))
    A.assume(x * 2.0 >= 0.5)
    A.assume(x * 3.0 <= 2.5)
    A.assert_(x <= -1.0)


def r25_no_assume_refuted(A):
    """No assume at all: nothing is withheld and nothing may change."""
    x = A.any((), "float64", (-1.0, 1.0))
    A.assert_(x >= 5.0)


def r26_no_assume_verified(A):
    """No assume, VERIFIED. The corpus must contain one, or it cannot see
    a move toward VERIFIED at all."""
    x = A.any((), "float64", (-1.0, 1.0))
    A.assert_(x >= -5.0)


def r27_certified_assume_verified(A):
    """A CERTIFIED assume (the target is the declared input, whose box IS
    its value set) with a VERIFIED obligation: the certificate must not
    run and must change nothing."""
    x = A.any((), "float64", (0.0, 1.0))
    A.assume(x >= 0.9)
    A.assert_(x >= -5.0)


def r28_certified_assume_refuted(A):
    """The same, REFUTED on its own without any certificate."""
    x = A.any((), "float64", (0.0, 1.0))
    A.assume(x >= 0.9)
    A.assert_(x <= 0.5)


def r29_uncertified_then_verified(A):
    """An UNCERTIFIED narrowing over an INHABITED region whose obligation
    DISCHARGES. Nothing here may move: the discharge was never withheld,
    and the certificate has no way to touch it."""
    x = A.any((), "float64", (0.0, 1.0))
    y = x * 2.0
    A.assume(y >= 0.5)
    A.assert_(x >= -100.0)


def r30_uncertified_empty_verified(A):
    """The same over an EMPTY region — a VERIFIED that is VACUOUSLY true.
    stelling keeps it (a discharge over a superset implies the discharge
    over the intended set) and the certificate must not touch it either."""
    x = A.any((), "float64", (0.0, 1.0))
    y = x - x
    A.assume(y >= 0.5)
    A.assert_(x >= -100.0)


def r31_wide_declaration(A):
    """A declaration at the cap boundary is measured separately (see
    cap_timing.py); this one is merely WIDE — 64 elements — so the corpus
    is not entirely scalars."""
    x = A.any((64,), "float64", (0.0, 1.0))
    y = x * 2.0
    A.assume(y >= 0.5)
    A.assert_(x <= -1.0)


def r32_narrow_inhabited_but_assert_holds(A):
    """THE ROW THAT CATCHES A WRONG REFUTED. The region is INHABITED and
    the obligation is TRUE on it, so no refutation is available at all —
    if the certificate ever turned this into REFUTED the oracle would
    find an admissible point satisfying the obligation."""
    x = A.any((), "float64", (0.0, 1.0))
    y = x * 2.0
    A.assume(y >= 0.5)
    A.assert_(x >= 0.2)


ROWS = [
    r01_narrow_inhabited, r02_narrow_empty,
    r03_narrow_inhabited_pair, r04_narrow_empty_pair,
    r05_square_inhabited, r06_square_empty,
    r07_relational_inhabited, r08_relational_empty,
    r09_relational_array_inhabited, r10_relational_array_empty,
    r11_all_reduction_inhabited, r12_all_reduction_empty,
    r13_or_predicate_inhabited, r14_or_predicate_empty,
    r15_int32_inhabited, r16_int32_empty_off_grid,
    r17_float32_inhabited, r18_float32_pair,
    r19_sqrt_slack, r20_sqrt_no_slack,
    r21_exp_slack, r22_exp_empty,
    r23_two_assumes_one_empty, r24_two_assumes_both_live,
    r25_no_assume_refuted, r26_no_assume_verified,
    r27_certified_assume_verified, r28_certified_assume_refuted,
    r29_uncertified_then_verified, r30_uncertified_empty_verified,
    r31_wide_declaration, r32_narrow_inhabited_but_assert_holds,
]

# My LABEL for each row's assumed region, written before any run. The
# oracle measures it independently and the two are reported side by side.
LABELLED_INHABITED = {
    "r01_narrow_inhabited", "r03_narrow_inhabited_pair",
    "r05_square_inhabited", "r07_relational_inhabited",
    "r09_relational_array_inhabited", "r11_all_reduction_inhabited",
    "r13_or_predicate_inhabited", "r15_int32_inhabited",
    "r17_float32_inhabited", "r18_float32_pair", "r19_sqrt_slack",
    "r20_sqrt_no_slack", "r21_exp_slack", "r24_two_assumes_both_live",
    "r25_no_assume_refuted", "r26_no_assume_verified",
    "r27_certified_assume_verified", "r28_certified_assume_refuted",
    "r29_uncertified_then_verified", "r31_wide_declaration",
    "r32_narrow_inhabited_but_assert_holds",
}
