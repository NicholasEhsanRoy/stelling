# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""AUDIT 0.2.0 S12 — the emission face may not accept what the propagation
face rejects.

`interval.dot_general` checked contracted-dimension agreement and RAISED.
`obligation._dot_general_plan` re-derived the same geometry from the LHS
alone and never cross-checked the RHS, so on a shorter symbolic operand it
emitted a TRUNCATED linear combination with no decline, and on a shorter
constant operand it raised a raw `IndexError` out of a slicer that catches
only `_Decline`. Because the interval leg's refusal lands the obligation at
⊤ → `unknown` → escalation, the truncating plan is exactly what ran.

Measured on `main` (`dee8bc2`), the query in
`test_the_from_dict_door_no_longer_yields_a_truncated_verdict`:

    from_dict ACCEPTED the mismatched query
    ESCALATION assert #0 -> discharged
    interval transfer:  IntervalError dot_general contracted dims disagree:
                        lhs[0]=2 vs rhs[0]=4
    emission plan:      (0, [[(Fraction(1,1), 0), (Fraction(1,1), 1)]])
                        <- 2 of the constant operand's 4 elements DROPPED

THE TRUTH HERE IS ARITHMETIC, NOT THE TOOL. Every expectation below is
computed in exact `fractions.Fraction` over the FOUR-term reading (the one
the constant operand's own four elements state), independently of anything
in `stelling.obligation`.

The fix is a shared oracle — `interval.dot_general_geometry` — driven by
both faces, so a shape predicate cannot exist in one and not the other.
That is what the first test measures, and it is the one that would catch a
future check added to only one side.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.obligation import _Decline, _dot_general_plan

# ── the three shapes the audit measured, plus the rest of the malformed
#    space. Each is (lhs_shape, rhs_shape, dimension_numbers, label).
WELL_FORMED = [
    ((3,), (3,), (((0,), (0,)), ((), ())), "equal 1-D contraction"),
    ((3, 4), (4,), (((1,), (0,)), ((), ())), "matvec"),
    ((3, 4), (4, 5), (((1,), (0,)), ((), ())), "matmul"),
    ((2, 3, 4), (3, 4), (((1, 2), (0, 1)), ((), ())), "two contractions"),
    ((4, 3, 5), (4, 5, 2), (((2,), (1,)), ((0,), (0,))), "batched"),
    ((3,), (4,), (((), ()), ((), ())), "outer product, no contraction"),
]

MALFORMED = [
    ((2,), (4,), (((0,), (0,)), ((), ())), "S12: lhs SHORTER — truncated"),
    ((4,), (2,), (((0,), (0,)), ((), ())), "S12: rhs SHORTER — IndexError"),
    ((3, 4), (5,), (((1,), (0,)), ((), ())), "matvec, extent disagrees"),
    ((2, 3, 4), (3, 9), (((1, 2), (0, 1)), ((), ())), "second contraction"),
    ((4, 3, 5), (2, 5, 2), (((2,), (1,)), ((0,), (0,))), "BATCH disagrees"),
    ((3,), (3,), (((0,), (0, 0)), ((), ())), "lists do not pair up"),
    ((3, 3), (3,), (((0, 0), (0,)), ((), ())), "lhs names a dim twice"),
    ((3,), (3,), (((5,), (0,)), ((), ())), "lhs dim out of range"),
    ((3,), (3,), (((0,), (5,)), ((), ())), "rhs dim out of range"),
    # audit 0.2.0 B6/S12″: a NON-INTEGER dim. It passes the range test
    # (`0 <= 0.0 < 3` is True) and then indexes a tuple with a float, which
    # raised a raw `TypeError` — out of the public `propagate()` on the
    # transfer side, and past `_dot_general_plan`'s `except IntervalError`
    # on the emission side, where the blanket net absorbed it as an
    # "internal error" decline. Two faces, two behaviours, one
    # malformation: the S12 shape again, in the oracle's own contract.
    ((3,), (3,), (((0.0,), (0,)), ((), ())), "lhs dim is a float"),
    ((3,), (3,), (((0,), (0.0,)), ((), ())), "rhs dim is a float"),
    ((3, 3), (3, 3), (((1,), (1,)), ((0.0,), (0,))), "batch dim is a float"),
    ((3,), (3,), ((("0",), (0,)), ((), ())), "lhs dim is a string"),
]


def _aval(shape):
    return ir.Aval(kind="ShapedArray", shape=tuple(shape), dtype="float64")


def _bool_aval():
    return ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def _size(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _dot_eqn(lhs_shape, rhs_shape, dn, out_shape):
    """A `dot_general` equation with operand 1 constant (var id 1)."""
    return ir.JaxprEqn(
        primitive="dot_general",
        invars=(
            ir.Var(id=0, aval=_aval(lhs_shape)),
            ir.Var(id=1, aval=_aval(rhs_shape)),
        ),
        outvars=(ir.Var(id=2, aval=_aval(out_shape)),),
        params=(
            ("dimension_numbers", dn),
            ("out_sharding", None),
            ("precision", None),
            ("preferred_element_type", "float64"),
        ),
    )


def _transfer(lhs_shape, rhs_shape, dn):
    """What the PROPAGATION face does: accept, or refuse with a sentence."""
    try:
        iv.dot_general(
            iv.from_bounds(tuple(lhs_shape), 1.0, 2.0),
            iv.from_bounds(tuple(rhs_shape), 1.0, 1.0),
            dn,
        )
    except iv.IntervalError as e:
        return str(e)
    return None


def _pre_fix_out_shape(lhs_shape, rhs_shape, dn):
    """The output shape `_dot_general_plan` derived BEFORE the fix — batch
    and free dims read off the operands directly, with no extent agreement
    anywhere in it.

    Used to build the equations below so that reverting the fix reproduces
    the audit's numbers exactly: the plan's own `out_shape` vs outvar-aval
    check must not be what refuses a malformed form, or these tests would
    pass for the wrong reason and stay green over a re-broken oracle.
    """
    try:
        (lc, rc), (lb, rb) = dn
        lfree = [i for i in range(len(lhs_shape)) if i not in lb and i not in lc]
        rfree = [j for j in range(len(rhs_shape)) if j not in rb and j not in rc]
        return (
            tuple(lhs_shape[i] for i in lb)
            + tuple(lhs_shape[i] for i in lfree)
            + tuple(rhs_shape[j] for j in rfree)
        )
    except (IndexError, TypeError, ValueError):
        return ()


def _emission(lhs_shape, rhs_shape, dn, out_shape=None):
    """What the EMISSION face does, on the same equation. Any exception
    that is not a `_Decline` escapes deliberately: `slice_obligation`'s
    caller handles `_Decline` and nothing else, so an `IndexError` here is
    the S12 crash and this helper must not disguise it."""
    if out_shape is None:
        out_shape = _pre_fix_out_shape(lhs_shape, rhs_shape, dn)
    eqn = _dot_eqn(lhs_shape, rhs_shape, dn, out_shape)
    consts = {1: tuple(Fraction(1) for _ in range(_size(rhs_shape)))}
    try:
        return None, _dot_general_plan([eqn], consts, eqn)
    except _Decline as d:
        return d.reason, None


# ── the property, stated once ────────────────────────────────────────────


@pytest.mark.parametrize(
    "lhs,rhs,dn,label",
    WELL_FORMED + MALFORMED,
    ids=[c[3] for c in WELL_FORMED + MALFORMED],
)
def test_the_two_faces_agree_about_well_formedness(lhs, rhs, dn, label):
    """THE PROPERTY S12 VIOLATED, over both halves of the space.

    Not "the emission declines every malformed form" — that would be
    satisfied by a second copy of the predicate, which is the arrangement
    that produced the defect. What is asserted is AGREEMENT: for every
    form, both faces accept or both refuse, with the SAME sentence. A check
    added to one side and not the other fails this test on the row it was
    added for — which is how the non-integer-dim rows below arrived, since
    on `4d793cf` the transfer crashed on them and the emission declined.

    The well-formed half is what stops the test being satisfiable by
    declining everything.
    """
    refusal = _transfer(lhs, rhs, dn)
    reason, plan = _emission(lhs, rhs, dn)
    if refusal is None:
        assert reason is None, (
            f"{label}: the transfer ACCEPTED this equation and the emission "
            f"declined it — {reason}"
        )
        assert plan is not None
    else:
        assert reason is not None, (
            f"{label}: the transfer refused this equation ({refusal}) and "
            f"the emission produced a plan anyway: {plan!r}"
        )
        # and the emission quotes the transfer's own sentence, because it is
        # the transfer's own function that produced it
        assert refusal in reason, (
            f"{label}: the two faces refuse for different stated reasons —\n"
            f"  transfer: {refusal}\n  emission: {reason}"
        )


def test_the_three_shapes_the_audit_measured():
    """S12's headline table, with the two directions named apart.

    `lhs=(2,) rhs=(4,)` returned `(0, [[(1,0),(1,1)]])` — a TWO-term
    combination over a FOUR-element constant operand, no decline.
    `lhs=(4,) rhs=(2,)` raised `IndexError: tuple index out of range`.
    `lhs=(3,) rhs=(3,)` was, and remains, correct.
    """
    dn = (((0,), (0,)), ((), ()))

    short_lhs_reason, short_lhs_plan = _emission((2,), (4,), dn)
    assert short_lhs_plan is None, (
        "the shorter-LHS form still produces a plan: " f"{short_lhs_plan!r}"
    )
    assert "contracted dims disagree: lhs[0]=2 vs rhs[0]=4" in short_lhs_reason

    # the shorter-RHS form must DECLINE, not raise: `_emission` lets any
    # non-_Decline exception through, so an IndexError fails this line
    short_rhs_reason, short_rhs_plan = _emission((4,), (2,), dn)
    assert short_rhs_plan is None
    assert "contracted dims disagree: lhs[0]=4 vs rhs[0]=2" in short_rhs_reason

    equal_reason, equal_plan = _emission((3,), (3,), dn)
    assert equal_reason is None, equal_reason
    sym_operand, groups = equal_plan
    assert sym_operand == 0
    assert groups == [[(Fraction(1), 0), (Fraction(1), 1), (Fraction(1), 2)]]


def test_the_plan_a_well_formed_contraction_produces_is_the_exact_sum():
    """The positive half, checked against exact `Fraction` arithmetic that
    never calls the code under test.

    A `(4,) · (4,)` contraction against the constant `[1, 2, 3, 4]`: the
    output element is `Σ c[k]·x[k]`, and the oracle for that is written out
    here as a dict from symbolic index to coefficient. The plan must state
    the same map — all four terms, each with its own coefficient.
    """
    dn = (((0,), (0,)), ((), ()))
    consts = {1: (Fraction(1), Fraction(2), Fraction(3), Fraction(4))}
    eqn = _dot_eqn((4,), (4,), dn, ())
    sym_operand, groups = _dot_general_plan([eqn], consts, eqn)

    assert sym_operand == 0
    (terms,) = groups
    # the ORACLE: the four-term reading, in exact rationals, written here
    oracle = {0: Fraction(1), 1: Fraction(2), 2: Fraction(3), 3: Fraction(4)}
    assert dict((idx, coeff) for coeff, idx in terms) == oracle
    assert len(terms) == 4, "an addend was dropped"


def test_a_truncated_plan_would_have_changed_the_verdict_and_here_is_why():
    """The independent-confirmation arithmetic from the S12 write-up, with
    NOTHING from `stelling` in it.

    Four declared elements in `[1, 2]`, constant operand all ones, threshold
    `9/2`. Over the FOUR-term reading the sum ranges over `[4, 8]`, which the
    threshold does not bound. Over the truncated TWO-term reading it ranges
    over `[2, 4]`, which it does — so the truncation is exactly the
    difference between REFUTED-in-principle and a false VERIFIED. Written
    out so a reader can check the claim the end-to-end test rests on
    without running the pipeline.
    """
    lo, hi, threshold = Fraction(1), Fraction(2), Fraction(9, 2)
    four_lo, four_hi = 4 * lo, 4 * hi
    two_lo, two_hi = 2 * lo, 2 * hi
    assert (four_lo, four_hi) == (Fraction(4), Fraction(8))
    assert (two_lo, two_hi) == (Fraction(2), Fraction(4))
    assert not (four_hi <= threshold), "the four-term claim is NOT valid"
    assert two_hi <= threshold, "the two-term claim IS valid — hence VERIFIED"


def test_BOTH_faces_actually_ask_the_oracle(monkeypatch):
    """The structural half of the repair, asserted rather than described.

    Agreement can be reached by two copies of a predicate that happen to
    match today, and that arrangement is what produced S12. What makes the
    repair durable is that neither face owns a shape rule any more. This
    pins the thing a private copy could not fake: a spy on the oracle
    records ONE call from each face on a well-formed contraction, so a face
    that went back to deriving its own geometry would record none.
    """
    dn = (((0,), (0,)), ((), ()))
    calls: list[str] = []
    real = iv.dot_general_geometry

    def spy(lhs_shape, rhs_shape, dimension_numbers):
        calls.append("call")
        return real(lhs_shape, rhs_shape, dimension_numbers)

    monkeypatch.setattr(iv, "dot_general_geometry", spy)

    iv.dot_general(
        iv.from_bounds((3,), 1.0, 2.0), iv.from_bounds((3,), 1.0, 1.0), dn
    )
    assert len(calls) == 1, (
        "the interval transfer computed its geometry without asking the "
        "oracle"
    )

    eqn = _dot_eqn((3,), (3,), dn, ())
    _dot_general_plan([eqn], {1: (Fraction(1),) * 3}, eqn)
    assert len(calls) == 2, (
        "the emission plan computed its geometry without asking the oracle"
    )


# ── the oracle's own CONTRACT (audit 0.2.0 B6/S12″) ──────────────────────


MALFORMED_DIMENSION_NUMBERS = [
    (0, "a bare int"),
    ((0, 0), "a 2-tuple of scalars"),
    ((((0,), (0,)), ((), ()), ((), ())), "three groups"),
    (None, "None"),
    ("xy", "a bare string"),
    ((((0.0,), (0,)), ((), ())), "a float contracting dim"),
    ((((0,), (0.0,)), ((), ())), "a float contracting dim on the rhs"),
    # the BATCH extent loop specifically: the contracting dims here are
    # well-formed and integral, so nothing refuses before the batch pair is
    # indexed. (`(((0,), (0,)), ((0.5,), (0,)))` would NOT do — its rhs
    # names dimension 0 twice and is refused two checks earlier, which
    # would leave this row green over a broken batch path.)
    ((((1,), (1,)), ((0.0,), (0,))), "a float batch dim"),
    (((("0",), (0,)), ((), ())), "a string dim"),
    ((((5,), (0,)), ((), ())), "a dim out of range"),
    ((((0, 0), (0, 0)), ((), ())), "a duplicated dim"),
    ((((0,), ()), ((), ())), "unpaired lists"),
]


@pytest.mark.parametrize(
    "dn,label",
    MALFORMED_DIMENSION_NUMBERS,
    ids=[c[1] for c in MALFORMED_DIMENSION_NUMBERS],
)
def test_the_oracle_raises_IntervalError_AND_NOTHING_ELSE(dn, label):
    """`dot_general_geometry` promises "Raises IntervalError on any
    malformation", and the promise was false in exactly one line.

    A non-integer entry in `dimension_numbers` passes the range test
    (`0 <= 0.0 < 3` is True) and reaches `lhs_shape[i]`, where python
    raises `TypeError: tuple indices must be integers or slices, not
    float`. Both consumers catch `IntervalError` and nothing else, so on
    `4d793cf` this was a raw crash out of the public `propagate()` — and,
    on the emission side, an "internal error" decline. The docstring
    asserting it could not happen was itself new in that commit, which is
    why the finding is the CLAIM as much as the code.

    Asserted as a type discipline rather than as a list of messages: any
    exception that is not an `IntervalError` fails, so a future predicate
    added here cannot reintroduce the class by raising something else.
    """
    with pytest.raises(iv.IntervalError):
        iv.dot_general_geometry((3, 3), (3, 3), dn)
    # and through the public transfer, which is the entry point that crashed
    with pytest.raises(iv.IntervalError):
        iv.dot_general(
            iv.from_bounds((3, 3), 1.0, 2.0),
            iv.from_bounds((3, 3), 1.0, 1.0),
            dn,
        )


def test_a_float_dim_no_longer_crashes_the_public_propagation():
    """The end-to-end half: `propagate()` is the public entry point, and
    on `4d793cf` this document took it down with a `TypeError`. It now
    returns, having declined the equation to ⊤ through the ordinary
    no-sound-rule channel — and the emission quotes the SAME sentence,
    which is the property this whole file is about."""
    from stelling.propagate import propagate

    dn = (((0.0,), (0,)), ((), ()))
    q = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(),
            outvars=(ir.Var(id=4, aval=_bool_aval()),),
            eqns=(
                ir.JaxprEqn(
                    primitive="stelling_any",
                    invars=(),
                    outvars=(ir.Var(id=0, aval=_aval((3,))),),
                    params=(
                        ("shape", (3,)), ("dtype", "float64"),
                        ("lo", 1.0), ("hi", 2.0),
                    ),
                ),
                ir.JaxprEqn(
                    primitive="stelling_any",
                    invars=(),
                    outvars=(ir.Var(id=1, aval=_aval((3,))),),
                    params=(
                        ("shape", (3,)), ("dtype", "float64"),
                        ("lo", 1.0), ("hi", 1.0),
                    ),
                ),
                ir.JaxprEqn(
                    primitive="dot_general",
                    invars=(
                        ir.Var(id=0, aval=_aval((3,))),
                        ir.Var(id=1, aval=_aval((3,))),
                    ),
                    outvars=(ir.Var(id=2, aval=_aval(())),),
                    params=(
                        ("dimension_numbers", dn),
                        ("out_sharding", None),
                        ("precision", None),
                        ("preferred_element_type", "float64"),
                    ),
                ),
                ir.JaxprEqn(
                    primitive="le",
                    invars=(
                        ir.Var(id=2, aval=_aval(())),
                        ir.Literal(val=99.0, aval=_aval(())),
                    ),
                    outvars=(ir.Var(id=3, aval=_bool_aval()),),
                ),
                ir.JaxprEqn(
                    primitive="stelling_assert",
                    invars=(ir.Var(id=3, aval=_bool_aval()),),
                    outvars=(ir.Var(id=4, aval=_bool_aval()),),
                ),
            ),
        )
    )
    p = propagate(q)  # must not raise
    assert [o.status for o in p.obligations] == ["unknown"]
    assert any("dot_general" in n for n in p.notes), p.notes


# ── FIX 5: what the S12 extraction ACTUALLY changed, measured ────────────


EXTRACTION_DELTAS = [
    (0, "a bare int"),
    ((0, 0), "a 2-tuple of scalars"),
    ((((0,), (0,)), ((), ()), ((), ())), "three groups"),
    (None, "None"),
    ("xy", "a bare string"),
]


@pytest.mark.parametrize(
    "dn,label", EXTRACTION_DELTAS, ids=[c[1] for c in EXTRACTION_DELTAS]
)
def test_the_extraction_DID_change_these_predicates_and_here_they_are(dn, label):
    """"A pure extraction: no predicate changed" was not true, and the
    honest sentence is this list.

    Measured on `dee8bc2` against `4d793cf`, `interval.dot_general` with
    each of these `dimension_numbers`:

        dee8bc2:  TypeError: cannot unpack non-iterable int object
                  TypeError: cannot unpack non-iterable int object
                  ValueError: too many values to unpack (expected 2)
                  TypeError: cannot unpack non-iterable NoneType object
                  ValueError: not enough values to unpack (expected 2, got 1)
        4d793cf:  IntervalError, all five

    The extraction wrapped the `dimension_numbers` unpack in a `try` and
    converted five raw unpack failures into the decline channel. Every one
    is in the SAFE direction (a raw crash became a quoted refusal) and none
    is reachable through `_t_dot_general`, whose `dimension_numbers` comes
    from `_dot_general_row_form` and is a well-formed 2×2 by then — but
    "no predicate changed" is a claim about the FUNCTION, and the function
    is public. Pinned here so the narrowed claim has a measurement behind
    it and so a later reader does not re-broaden it.
    """
    with pytest.raises(iv.IntervalError):
        iv.dot_general(
            iv.from_bounds((3,), 1.0, 2.0), iv.from_bounds((3,), 1.0, 1.0), dn
        )


def test_the_two_new_check_shape_calls_are_unreachable_for_an_IntervalArray():
    """The other half of the narrowed Fix-5 claim, and it goes the other
    way: the extraction added `check_shape(lhs_shape)` / `check_shape(
    rhs_shape)` to the oracle, and for any `IntervalArray` operand those
    cannot fire, because `IntervalArray.__post_init__` has already run the
    identical check. So they are neutral for every caller of
    `interval.dot_general` — the behavioural delta is only in
    `dot_general_geometry`, which did not exist before the extraction and
    therefore changed nothing.
    """
    with pytest.raises(iv.IntervalError):
        iv.IntervalArray(shape=(-2,), los=(), his=())
    with pytest.raises(iv.IntervalError):
        iv.IntervalArray(shape=("x",), los=(), his=())
    # the oracle refuses them directly, which is what those two calls buy
    dn = (((0,), (0,)), ((), ()))
    with pytest.raises(iv.IntervalError):
        iv.dot_general_geometry((-2,), (3,), dn)
    with pytest.raises(iv.IntervalError):
        iv.dot_general_geometry((3,), ("x",), dn)
