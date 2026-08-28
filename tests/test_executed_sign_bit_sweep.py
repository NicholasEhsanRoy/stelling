# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE SIGN BIT OF AN EXECUTED ZERO, SWEPT OVER THE WHOLE CARRYING SET.

`_t_div`'s standing-constraint paragraph says the certificate's real-vs-float
gap costs no verdict *"while the zero the program actually computes carries
the sign bit the certificate claims"*. **That sentence is FALSE on this tree
and on `main`**, and this module is the instrument that says so by primitive
rather than by anecdote.

**WHERE IT IS FALSE IS NOT WHERE A READER WOULD LOOK.** The paragraph names
three rows — `mul`, `sqrt`, `max`/`min` — as the ones "whose ℝ value is
nonzero where the executable's is zero", and every one of those four
primitives comes back CLEAN below: their executed zeros carry the certified
sign bit at every value this sweep tries. What is wrong is the LIST.
`reduce_sum` and `dot_general` have the same ℝ-versus-executable zero gap — a
certified operand whose elements flush leaves them computing a zero where ℝ
has a nonzero — and the paragraph does not name them, so the sentence that
follows was never held against the two members it fails for. The blinded
audit it cites (500 traced programs, **2 097 executed zeros under a
certificate, 0 with the wrong sign bit**) is a null of that GENERATOR, not of
the class: what it takes is a program that reduces an all-flushed certified
vector, and the `reduce_sum` row below finds nine wrong-signed zeros in
seventy-eight executions.

THE QUESTION, STATED SO IT CAN BE ASKED OF EVERY ROW. The certificate is a
claim about the **ℝ sign** of a value. Its one consumer,
:func:`stelling.interval.boundary_div`, is sensitive to the **IEEE sign bit
of a zero** — an opposite-signed zero divisor falls outside all four of its
arms, which is a false VERIFIED on a lower-bound obligation and a false
REFUTED on an upper one
(`test_boundary_div_tolerates_only_a_MATCHING_signed_zero`). The two come
apart wherever the target computes a zero whose sign bit is not the ℝ sign
the certificate asserts. So of every carrying primitive this module asks:

    given operands whose executed values are consistent with THEIR
    certificates — a nonzero of the certified sign, or a zero whose sign bit
    IS the certified sign — can this primitive produce an output that is a
    zero whose sign bit is not the sign the rule mints for it?

**IT IS A LOCAL QUESTION ON PURPOSE, AND THAT IS WHAT MAKES IT COMPOSE.** A
certified value is built by a DAG of these rules over a base of declarations,
literals and constvars. If every primitive is locally clean and every base
case is clean, then by induction over the DAG no certified value's executed
zero can carry the wrong sign bit. One dirty primitive breaks the induction,
and everything downstream of it inherits the break — which is why `neg` is
locally CLEAN here and still carries the defect onward in
`tests/test_strict_sign_census.py::
test_an_executed_zero_under_a_certificate_carries_the_CERTIFIED_sign_bit`.
The induction's two premises are pinned here as well: that a certified value
really CAN execute as a matching-signed zero (so the hypothesis is not
vacuous), and that the three certificate SOURCES are clean.

**DECIDED BY RUNNING THE PROGRAM, NEVER BY READING THE LOWERING.** Every row
executes jax at concrete operand values, eagerly AND under `jit`, and reads
`math.copysign` off what came back. Nothing here reasons about what XLA
"should" do; L28 — *a check that models a behaviour is one indirection behind
it* — is the whole reason the two guards that existed could not draw this
class.

**THE ANSWER IS REPORTED FOR TWO CARRYING SETS.** `main` carries the
certificate on ten primitives and the 0.3.0 census carries it on thirty-one.
The thirty-one are DERIVED from the census at test time; the ten are a frozen
record of another tree, and :data:`MAIN_CARRIERS` says why it cannot be
derived here and what was done instead. A sibling under the wider set only
would be a 0.3.0 blocker; one under the narrower set is live on `main` today.
The two answers are separate tests.

**AND THE ANSWER IS ABOUT PRIMITIVES, NOT ABOUT REACH — a distinction this
module got wrong and now states twice.** "The twenty-one carriers 0.3.0 added
are all clean" is true, and it does NOT mean the release left this class
where it found it. Widening the carrier set widens the set of PROGRAMS that
arrive at the two diverging primitives still holding a certificate, and that
is a change in blast radius even though no new primitive diverges. MEASURED,
same `src/`, only `_STRICT_SIGN_PRIMITIVES` swapped between the two sets::

    u = any_array((), f64, (-1.0, -0.25));  assume(u < 0)
    v = jnp.stack([u, u])                   # `stack`: a 0.3.0 carrier
    assert_(1.0 / jnp.sum(v * 1e-200 * 1e-200) < 0.0)
        main's ten  -> unknown
        census's 31 -> discharged, and the program says False at every point

`jnp.broadcast_to(u, (3,))` splits the same way through
`broadcast_in_dim`. So the honest pair of sentences is: **0.3.0 admitted no
new diverging primitive, and 0.3.0 did widen the reach of the two that
diverge.** Both are checked —
`test_THE_ANSWER_for_THE_CENSUSS_THIRTY_ONE_carrying_primitives` for the
first, `test_the_0_3_0_widening_admitted_no_new_SIBLING_but_did_widen_the_REACH`
for the second.

**WHAT THIS SWEEP DOES NOT REACH.**

* It is a **finite enumeration**, not a proof. Each row runs a declared
  cross-product of operand values; a divergence that needs a value outside
  that product is invisible here. The values are chosen adversarially rather
  than sampled — every certified operand list contains the matching-signed
  zero, the flushed subnormal and the underflowing magnitude — but "no
  divergence found" is a statement about the enumeration, not about the
  primitive.
* It asks the LOCAL question. It says nothing about which local
  configurations a real query can REACH; `tests/test_strict_sign_census.py`'s
  `ZERO_UNDER_CERTIFICATE` table is the end-to-end half, and the one
  reachability premise this induction needs is pinned below.
* It reads one target — jax on this machine, whose version and backend the
  rows do not pin. A divergence is a fact about what was executed here.
* It says nothing about VERDICTS. Whether a wrong-signed zero reaches a
  wrong answer is `boundary_div`'s four-arm asymmetry, pinned in the census
  module.
* **rank > 1 and non-trivial params.** Every row is rank 0 or rank 1 with a
  small static extent and the simplest admitted dimension numbers. A
  reduction over one axis of a rank-3 operand, a batched `dot_general`, a
  `gather` with several offset dims — none of those forms is executed here.
  The one rank-2 configuration that mattered to the answer, a `reduce_sum`
  whose output cells each sum ONE term, is driven end to end by
  `tests/test_strict_sign_census.py`'s
  `"reduce_sum n=6 reduced to 1 term per cell"` row.
* **It is not the exact-rational property**, and the exact-rational property
  is not this. `tests/property/test_strict_sign_property.py` searches in
  `Fraction` arithmetic, where there is no signed zero at all: `(+0) + (−0)`
  is `0`, full stop. That search is structurally incapable of expressing a
  defect whose subject is a float sign bit, so a null from it is not evidence
  about this class and must never be read as one.

**A GENERATED SEARCH WAS CONSIDERED FOR THIS QUESTION AND DECLINED, and the
reason is not cost.** The obvious alternative to this module is a
`hypothesis` property in `tests/property/` over shapes, extents, bounds and
chains, with jax executing and `copysign` as the oracle — a different
instrument from the exact-rational one and not subject to the ℚ objection
above. Two things decided against it.

The smaller one: `hypothesis` is in NONE of the three merge lanes (measured
2026-08-28: `import hypothesis` raises `ModuleNotFoundError` under
`stelling-jax`, `stelling-nojax` and `stelling-jax010`). A property-suite
guard for this class would run only in the `property` job, and would be
absent from every lane that gates a merge, while this module runs in the two
that have jax. That is a real cost and it is not decisive on its own —
`tests/property/` exists and is gated by its own job and its own positive
controls.

The decisive one is the SHAPE of the question. The operand values that can
separate a sign bit are a handful of distinguished floats — the two zeros,
the smallest subnormal, a magnitude that underflows under the operation —
and the shape distinction that matters is "reduced extent 1" against
"reduced extent ≥ 2". That space is small enough to EXHAUST, and this module
exhausts it: 31 rows and 72 cases, 4 130 executions producing 16 228
read output elements (measured 2026-08-28 on jax 0.11.0 by summing
`_measure(...).executions` and `.elements` over `SWEEP`), every certified
operand list containing the matching-signed zero. A random search over a
space you can enumerate is strictly weaker than enumerating it, and it would
report its coverage as a generator FLOOR where this reports a total. Where a
generated search would genuinely buy reach is the shape and param space in
the bullet above — rank > 1, batch dims, dimension numbers — and that reach
is NOT bought here. It is the residual, it is stated, and it is the thing a
future item would build the search for.
"""

from __future__ import annotations

import itertools
import math

import pytest

from stelling import interval as iv
from stelling import ir
from stelling import propagate as P
from stelling.propagate import _Propagator

_TESTS_ROOT = __import__("pathlib").Path(__file__).resolve().parent


# --- what `main` carries, recorded rather than derived -----------------------

MAIN_CARRIERS = frozenset({
    "mul", "div", "add", "add_any", "neg", "abs", "square", "integer_pow",
    "reduce_sum", "dot_general",
})
"""The ten primitives `_STRICT_SIGN_PRIMITIVES` names on `main`.

A FROZEN RECORD OF ANOTHER TREE. It is TYPED rather than derived — the one
set in this module that is, because it cannot be derived from the tree this
file is in — and its CONTENT is measured, by the command below. Measured on 2026-08-28 at `a90862b` by reading the other
tree out of git rather than the developer's checkout::

    git show main:src/stelling/propagate.py

where `_STRICT_SIGN_PRIMITIVES` is a hand-written `frozenset` of exactly
these ten names.

**AND WHY ONE SWEEP, RUN ON THIS TREE, ANSWERS FOR THAT ONE.** The ten rule
branches of `_Propagator._strict_sign_out` were compared line by line
between the two files (the `main` copy taken from the command above, the
branches split on `if prim ==`/`in` and stripped): `("mul", "div")`,
`("add", "add_any")`, `"neg"`, `("abs", "square")`, `"integer_pow"`,
`"reduce_sum"` and `"dot_general"` are IDENTICAL, and the branch tree adds
`"sub"`, `"sqrt"`, `"scatter-add"`, `"max"`, `"min"` and the
`_SIGN_ROUTING` fallthrough that `main` does not have. Same rules, same
target, so the answer transfers.

TWO DIFFERENCES THAT ARE REAL AND ARE NOT ABOUT THIS CLASS, named so a
reader does not have to take "identical" as a claim about the whole
function. `main`'s gate reads
`if prim not in _STRICT_SIGN_PRIMITIVES or len(eqn.outvars) != 1`, refusing
every multi-output equation rather than permitting the routing class; and
`main` has no `_record_strict_sign` at all, so it carries no empty-value
guard. Neither touches the sign bit of an executed zero: a value with no
elements has no zero to be wrong about, and every one of `main`'s ten is
single-output anyway.

It is NOT read back out of git at test time, deliberately. A check whose
input is the developer's git state reports a different truth to different
people, and a worktree with no `main` would make this module's answer depend
on the checkout rather than on the code.
"""


# --- the executed-value candidates -------------------------------------------
#
# The inductive hypothesis, made concrete. An operand certified `s` may
# execute as any nonzero of sign `s` — including one small enough that the
# operation on it underflows — or as a zero whose sign bit is `s`. An
# UNCERTIFIED operand may execute as anything, and the two zeros are the
# values that matter, so both are in its list.
#
# Two widths because the cross-product is over every value slot a row has:
# a row with four slots (a `dot_general` over two extent-2 operands) would
# be 1296 executions at the wide width and is 256 at the narrow one.

_WIDE = {
    1: (1.0, 3.0, 0.0, 5e-324, 1e-320, 1e-200),
    -1: (-1.0, -3.0, -0.0, -5e-324, -1e-320, -1e-200),
    0: (1.0, -1.0, 0.0, -0.0, 1e-320, -1e-320),
}
_NARROW = {
    1: (1.0, 0.0, 1e-320, 1e-200),
    -1: (-1.0, -0.0, -1e-320, -1e-200),
    0: (1.0, -1.0, 0.0, -0.0),
}


def _candidates(sign: int, slots: int) -> tuple[float, ...]:
    return (_WIDE if slots <= 2 else _NARROW)[sign]


# --- asking the RULE what it mints -------------------------------------------


F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")


def _aval(shape):
    return ir.Aval(kind="ShapedArray", shape=tuple(shape), dtype="float64")


def _box(sign, shape):
    """A box consistent with `sign`. The rules read boxes only for the two
    SIZE guards, so the endpoints are illustrative and the size is not."""
    n = 1
    for d in shape:
        n *= d
    lo, hi = {1: (1.0, 2.0), -1: (-2.0, -1.0), 0: (-1.0, 1.0)}[sign]
    return iv.IntervalArray(shape=tuple(shape), los=(lo,) * n, his=(hi,) * n)


def _minted(prim, signs, shapes, params, outvars=1) -> int:
    """What `_Propagator._strict_sign_out` answers for this equation.

    A near-twin of `tests/test_strict_sign_census.py::_run_case`, and
    deliberately a SECOND copy rather than an import: this module's claim is
    "here is what the shipped rule mints and here is what the target
    computes", and an auditor should be able to read the whole of the first
    half in one file. What is duplicated is twenty lines of IR
    construction, not a fact — the rule itself is called, never restated.
    """
    p = _Propagator("constrain")
    invars, ins = [], []
    for i, (sgn, shape) in enumerate(zip(signs, shapes)):
        v = ir.Var(id=i + 1, aval=_aval(shape))
        invars.append(v)
        ins.append(_box(sgn, shape))
        if sgn:
            p.strict_sign[v.id] = sgn
    outs = tuple(ir.Var(id=900 + k, aval=F64) for k in range(outvars))
    eqn = ir.JaxprEqn(
        primitive=prim, invars=tuple(invars), outvars=outs,
        params=tuple(params),
    )
    return p._strict_sign_out(eqn, dict(params), ins)


# --- executing ---------------------------------------------------------------


def _f64_lax():
    import jax
    import jax.numpy as jnp
    from jax import lax

    return jax, jnp, lax


def _at_x64(fn):
    import jax

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return fn()
    finally:
        jax.config.update("jax_enable_x64", old)


class Case:
    """One probe of one primitive.

    ``signs``/``shapes``/``params`` are what the RULE is asked; ``slots`` is
    how many scalar operand VALUES ``run`` takes and ``slot_signs`` is the
    certificate each of those scalars must be consistent with. The two are
    separate because an operand can be an ARRAY whose elements are several
    slots (a `reduce_sum` over an extent-2 operand is one operand and two
    slots) and because an INDEX operand has a certificate the rule reads and
    no float value the sign bit question applies to.
    """

    def __init__(self, signs, slot_signs, run, *, shapes=None, params=(),
                 outvars=1, why=""):
        self.signs = tuple(signs)
        self.slot_signs = tuple(slot_signs)
        self.run = run
        self.shapes = tuple(shapes) if shapes is not None else ((),) * len(self.signs)
        self.params = tuple(params)
        self.outvars = outvars
        self.why = why


class Row:
    """One carrying primitive's answer, declared. ``bucket`` is one of:

    ``"no-zero"``   the enumeration produced no executed zero at all;
    ``"matching"``  it produced zeros and every one carried the certified
                    sign bit;
    ``"diverges"``  it produced at least one zero whose sign bit is not the
                    certified one — a SIBLING, with the witness in the
                    failure message and in `_measure`'s record.

    The bucket is COMPARED against what the executions actually did, so all
    three arms are live: a row cannot be parked in a bucket by assertion.

    ``jit_disagrees`` is about the TWO RUNTIME LEGS this row executes —
    eager, and `jit` over the drawn scalar arguments. It is NOT a claim that
    every lowering agrees: a third, constant-folded lowering exists and
    differs on many of these rows — how many is version-dependent — which is
    a separate question with its own check,
    `test_the_THIRD_lowering_changes_the_bits_and_changes_NO_CLASSIFICATION`.
    """

    def __init__(self, bucket, cases, why, *, nans=False,
                 jit_disagrees=False):
        self.bucket = bucket
        self.cases = tuple(cases)
        self.why = why
        self.nans = nans
        self.jit_disagrees = jit_disagrees


def _vec(jnp, vals):
    """A rank-1 float64 array from the drawn scalar slots. `stack` rather
    than `jnp.array` so the same expression works on concrete values and on
    `jit` tracers alike."""
    return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in vals])


def _one(jnp, v):
    return jnp.asarray(v, dtype=jnp.float64)


_IDX0 = ((0,), ())
"""Static index tuples are built inside the runs; named only where a reader
would otherwise have to count parentheses."""


# The sweep. One entry per carrying primitive, asserted TOTAL over
# `_STRICT_SIGN_PRIMITIVES` below, so a primitive admitted to the census
# without a row here reds — the census's own rule, applied to the census's
# own blind spot.
SWEEP: dict[str, Row] = {

    # ── ARITHMETIC ─────────────────────────────────────────────────────────

    "add": Row(
        "matching",
        (
            Case((1, 1), (1, 1), lambda m, v: m[2].add(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((-1, -1), (-1, -1), lambda m, v: m[2].add(_one(m[1], v[0]), _one(m[1], v[1]))),
        ),
        "IEEE addition of two operands of the same sign is that sign, and "
        "the only way it is zero is if both operands are zeros of that sign "
        "— `(−0) + (−0) = −0`, `(+0) + (+0) = +0`. There is no identity to "
        "be seeded with here: `add` is a binary operation on the two "
        "operands and nothing else. Two same-signed NONZEROS cannot cancel, "
        "so the magnitudes cannot reach zero either.",
    ),
    "add_any": Row(
        "matching",
        (
            Case((1, 1), (1, 1),
                 lambda m, v: m[0].grad(
                     lambda t: _one(m[1], v[0]) * t + _one(m[1], v[1]) * t
                 )(m[1].float64(1.0))),
            Case((-1, -1), (-1, -1),
                 lambda m, v: m[0].grad(
                     lambda t: _one(m[1], v[0]) * t + _one(m[1], v[1]) * t
                 )(m[1].float64(1.0))),
        ),
        "`add`'s argument, on jax's autodiff-only cotangent sum. Reached "
        "through `jax.grad` rather than by binding the primitive directly, "
        "because `add_any_p` is exported only from a PRIVATE jax module that "
        "`tests/test_import_hygiene.py::"
        "test_private_jax_modules_banned_everywhere` forbids every file "
        "under `tests/` and `src/` to name — which is why this sentence does "
        "not name it either. Measured: the traced program really contains "
        "`add_any` (`test_every_run_really_EXECUTES_the_primitive_it_is_a_"
        "row_for` asks jax rather than taking the word of this comment).",
    ),
    "sub": Row(
        "matching",
        (
            Case((1, -1), (1, -1), lambda m, v: m[2].sub(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((-1, 1), (-1, 1), lambda m, v: m[2].sub(_one(m[1], v[0]), _one(m[1], v[1]))),
        ),
        "The rule fires only on OPPOSITE-signed operands, and `a − b` there "
        "is `a + (−b)` with both terms of `a`'s sign — `add`'s argument "
        "again. The two zero cases are the ones IEEE 754 fixes explicitly: "
        "`(+0) − (−0) = +0` and `(−0) − (+0) = −0`, both the certified sign. "
        "SAME-signed operands, where cancellation to a zero of either sign "
        "is possible, mint nothing (the `Σx² − c` shape), so they are not a "
        "configuration this question can be asked of.",
    ),
    "mul": Row(
        "matching",
        (
            Case((1, 1), (1, 1), lambda m, v: m[2].mul(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((1, -1), (1, -1), lambda m, v: m[2].mul(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((-1, -1), (-1, -1), lambda m, v: m[2].mul(_one(m[1], v[0]), _one(m[1], v[1]))),
        ),
        "IEEE 754 fixes the sign of a product as the XOR of the operands' "
        "sign bits, for every finite result INCLUDING an underflow to zero "
        "and including a zero operand. The rule mints exactly that XOR, so "
        "the two agree by construction and not by luck. This is the row the "
        "whole standing-constraint paragraph was written about — `x*x*x` "
        "underflowing to `-0.0` — and it is clean.",
    ),
    "div": Row(
        "matching",
        (
            Case((1, 1), (1, 1), lambda m, v: m[2].div(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((1, -1), (1, -1), lambda m, v: m[2].div(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((-1, -1), (-1, -1), lambda m, v: m[2].div(_one(m[1], v[0]), _one(m[1], v[1]))),
        ),
        "The same XOR, and the same argument. A zero DIVISOR gives an "
        "infinity rather than a zero, so it cannot produce a wrong-signed "
        "zero; a zero DIVIDEND gives a zero whose sign bit is the XOR. The "
        "one thing this row produces that is not a number IS reachable end "
        "to end from an ordinary declaration — measured in "
        "`test_a_certified_div_under_an_ORDINARY_declaration_executes_as_NaN`.",
        nans=True,
    ),
    "neg": Row(
        "matching",
        (
            Case((1,), (1,), lambda m, v: m[2].neg(_one(m[1], v[0]))),
            Case((-1,), (-1,), lambda m, v: m[2].neg(_one(m[1], v[0]))),
        ),
        "`neg` flips the sign bit, unconditionally and including on a zero "
        "(`neg(+0.0) = -0.0`), and the rule flips the certificate. LOCALLY "
        "CLEAN AND NOT HARMLESS: it is exactly this faithfulness that "
        "carries a broken certificate onward, turning a wrong `-1` into a "
        "wrong `+1`. That is the induction breaking downstream of a dirty "
        "primitive, not `neg` being dirty.",
    ),
    "abs": Row(
        "matching",
        (
            Case((1,), (1,), lambda m, v: m[2].abs(_one(m[1], v[0]))),
            Case((-1,), (-1,), lambda m, v: m[2].abs(_one(m[1], v[0]))),
        ),
        "`abs` clears the sign bit — `abs(-0.0)` is `+0.0`, measured — and "
        "the rule mints `+1`.",
    ),
    "square": Row(
        "matching",
        (
            Case((1,), (1,), lambda m, v: m[2].square(_one(m[1], v[0]))),
            Case((-1,), (-1,), lambda m, v: m[2].square(_one(m[1], v[0]))),
        ),
        "`x*x` is `mul`'s XOR with equal operands, so the sign bit is "
        "always clear; the rule mints `+1`.",
    ),
    "sqrt": Row(
        "matching",
        (
            Case((1,), (1,), lambda m, v: m[2].sqrt(_one(m[1], v[0]))),
        ),
        "The rule is one-sided: only a `+1` operand mints, and it mints "
        "`+1`. IEEE `sqrt(+0)` is `+0`, and the target's DAZ flush of a "
        "subnormal operand produces `+0` from a positive subnormal. "
        "`sqrt(-0.0)` is `-0.0` — faithful, and unreachable from a `+1` "
        "certificate under the hypothesis, which is why this row is clean "
        "and would stop being clean the moment an upstream `+1` executed as "
        "`-0.0`.",
    ),
    "integer_pow": Row(
        "matching",
        (
            Case((-1,), (-1,), lambda m, v: m[2].integer_pow(_one(m[1], v[0]), 2),
                 params=(("y", 2),)),
            Case((-1,), (-1,), lambda m, v: m[2].integer_pow(_one(m[1], v[0]), 3),
                 params=(("y", 3),)),
            Case((1,), (1,), lambda m, v: m[2].integer_pow(_one(m[1], v[0]), 2),
                 params=(("y", 2),)),
            Case((-1,), (-1,), lambda m, v: m[2].integer_pow(_one(m[1], v[0]), -2),
                 params=(("y", -2),)),
            Case((-1,), (-1,), lambda m, v: m[2].integer_pow(_one(m[1], v[0]), -3),
                 params=(("y", -3),)),
        ),
        "`x**n` for integer `n` is a chain of `mul`s, so its sign bit is "
        "`sign(x)**n` — `+` for even `n`, `sign(x)` for odd — which is "
        "exactly the parity rule. Measured on both zeros and on underflowing "
        "magnitudes, and on NEGATIVE exponents, where a large magnitude "
        "underflows the other way: `(-1e200)**-3` is `-0.0` and `**-2` is "
        "`+0.0`.",
    ),
    "reduce_sum": Row(
        "diverges",
        (
            Case((-1,), (-1, -1),
                 lambda m, v: m[1].sum(_vec(m[1], v)),
                 shapes=((2,),), params=(("axes", (0,)),)),
            Case((1,), (1, 1),
                 lambda m, v: m[1].sum(_vec(m[1], v)),
                 shapes=((2,),), params=(("axes", (0,)),)),
            Case((-1,), (-1,),
                 lambda m, v: m[1].sum(_vec(m[1], v)),
                 shapes=((1,),), params=(("axes", (0,)),),
                 why="one term: the control inside the row"),
        ),
        "**SIBLING, AND THE ONE THE INSTRUMENTS WERE BUILT FOR.** XLA lowers "
        "the reduction as an accumulation SEEDED WITH `+0.0`, and in "
        "round-to-nearest `(+0) + (−0) = +0`. A `-1`-certified operand whose "
        "every element executes as `-0.0` therefore reduces to `+0.0`. "
        "MEASURED, eager and under `jit`: `sum([-0.0])` is `-0.0` (one term, "
        "the seed never has to absorb a second) and `sum([-0.0, -0.0])` is "
        "`+0.0`. ASYMMETRIC: a `+1` certificate survives, because the seed "
        "is the sign it claims. This row's two RUNTIME legs disagree, and "
        "a third, constant-folded lowering disagrees with both — see "
        "`test_the_SAME_reduction_compiles_AT_LEAST_THREE_WAYS_and_they_"
        "disagree_on_the_sign_bit`. `dot_general` splits the same way one "
        "leg further out.",
        jit_disagrees=True,
    ),
    "dot_general": Row(
        "diverges",
        (
            Case((-1, 1), (-1, -1, 1, 1),
                 lambda m, v: m[2].dot_general(
                     _vec(m[1], v[:2]), _vec(m[1], v[2:]),
                     (((0,), (0,)), ((), ()))),
                 shapes=((2,), (2,))),
            Case((1, 1), (1, 1, 1, 1),
                 lambda m, v: m[2].dot_general(
                     _vec(m[1], v[:2]), _vec(m[1], v[2:]),
                     (((0,), (0,)), ((), ()))),
                 shapes=((2,), (2,))),
            Case((-1, -1), (-1, -1, -1, -1),
                 lambda m, v: m[2].dot_general(
                     _vec(m[1], v[:2]), _vec(m[1], v[2:]),
                     (((0,), (0,)), ((), ()))),
                 shapes=((2,), (2,))),
            Case((-1, 1), (-1, 1),
                 lambda m, v: m[2].dot_general(
                     _vec(m[1], v[:1]), _vec(m[1], v[1:]),
                     (((0,), (0,)), ((), ()))),
                 shapes=((1,), (1,)),
                 why="an extent-1 contraction: the control inside the row"),
        ),
        "**SIBLING.** `reduce_sum`'s seed reached through a contraction: "
        "each output element is a sum of products, and under the lowerings "
        "this row executes that sum is seeded with `+0.0` as `reduce_sum`'s "
        "is. **THAT SENTENCE USED TO END \"exactly as `reduce_sum`'s is\" "
        "WITH NO QUALIFIER, AND IT IS THE SAME UNQUALIFIED SHAPE THIS "
        "MODULE EXISTS TO REFUSE.** The seed is a property of the LOWERING: "
        "`dot_general` splits across lowerings exactly as `reduce_sum` does, "
        "just one leg further out — measured at this row's own case-0 values "
        "`(-1.0, -1.0, 0.0, 0.0)` on jax 0.11.0 and again on 0.10.2, eager "
        "`+0.0`, jit-from-scalars `+0.0`, jit-over-constants `-0.0`. The two "
        "legs this row executes agree; a third does not. See "
        "`test_the_SAME_reduction_compiles_AT_LEAST_THREE_WAYS_and_they_"
        "disagree_on_the_sign_bit`. MEASURED: "
        "`dot([-0.,-0.], [1.,1.])` is `+0.0` under a `-1` certificate, and "
        "the extent-1 contraction `dot([-0.], [1.])` is `-0.0`. DATA "
        "DEPENDENT in a way `reduce_sum` is not — `dot([-1e-200,-1e-200], "
        "[1e-200,1e-200])`, whose products UNDERFLOW rather than arriving as "
        "zeros, comes back `-0.0` on this target — which is why the row "
        "enumerates values rather than asserting one.",
    ),
    "scatter-add": Row(
        "matching",
        (
            Case((1, 0, 1), (1, 1, 1),
                 lambda m, v: _vec(m[1], v[:2]).at[0].add(_one(m[1], v[2])),
                 shapes=((2,), (1,), ())),
            Case((-1, 0, -1), (-1, -1, -1),
                 lambda m, v: _vec(m[1], v[:2]).at[0].add(_one(m[1], v[2])),
                 shapes=((2,), (1,), ())),
            Case((-1, 0, -1), (-1, -1, -1, -1),
                 lambda m, v: _vec(m[1], v[:2]).at[
                     m[1].array([0, 0])].add(_vec(m[1], v[2:])),
                 shapes=((2,), (2,), (2,)),
                 why="DUPLICATE INDICES, which is the accumulate form's "
                     "defining semantic and the only place a seed could hide"),
        ),
        "THE ACCUMULATION THAT IS NOT SEEDED WITH THE IDENTITY, and it had "
        "to be measured rather than assumed because it is the same shape as "
        "the two siblings. `out[i] = operand[i] + Σ updates[j]` starts from "
        "the OPERAND, which carries a certificate, so there is no `+0.0` in "
        "the sum: measured, an operand of `-0.0` accumulating eight `-0.0` "
        "updates at one duplicated index comes back `-0.0`, eager and under "
        "`jit`. The seeded shape does exist in jax — `jax.ops.segment_sum` "
        "over implicit `zeros` reduces `[-0.,-0.]` to `+0.0` — but that "
        "operand is a zeros array, whose box is `[0, 0]`, which "
        "`_box_strict_sign` refuses to certify, so the rule never mints "
        "there. The rule ALSO needs both operand and updates certified to "
        "the same sign, so a mixed accumulation mints nothing.",
    ),

    # ── ROUTING ────────────────────────────────────────────────────────────

    "max": Row(
        "matching",
        (
            Case((1, 0), (1, 0), lambda m, v: m[2].max(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((0, 1), (0, 1), lambda m, v: m[2].max(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((-1, -1), (-1, -1), lambda m, v: m[2].max(_one(m[1], v[0]), _one(m[1], v[1]))),
        ),
        "**THE ROW THE SPEC ASKED ABOUT BY NAME, AND THE ANSWER IS THE "
        "CERTIFIED ONE.** MEASURED, eager and under `jit`: "
        "`lax.max(+0.0, -0.0)` is `+0.0` and `lax.max(-0.0, +0.0)` is "
        "`+0.0` — the target's maximum returns the POSITIVE zero whichever "
        "side it is on. That is the sign the `+1` arm mints, and the `+1` "
        "arm is the asymmetric one that needs only ONE certified operand, "
        "so the other operand being an arbitrary `-0.0` is exactly the case "
        "this row had to cover. The `(-1, -1)` arm is `max(-0.0, -0.0)`, "
        "which is `-0.0`. The DAZ exception the census names — "
        "`lax.max(5e-324, -1.0)` is `+0.0`, a value neither operand held — "
        "is in the enumeration and is matching.",
    ),
    "min": Row(
        "matching",
        (
            Case((-1, 0), (-1, 0), lambda m, v: m[2].min(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((0, -1), (0, -1), lambda m, v: m[2].min(_one(m[1], v[0]), _one(m[1], v[1]))),
            Case((1, 1), (1, 1), lambda m, v: m[2].min(_one(m[1], v[0]), _one(m[1], v[1]))),
        ),
        "The mirror, and it mirrors: `lax.min(-0.0, +0.0)` and "
        "`lax.min(+0.0, -0.0)` are both `-0.0`, which is the sign the `-1` "
        "arm mints. So the two exceptional routing members return the zero "
        "their own asymmetric rule claims, on both sides.",
    ),
    "select_n": Row(
        "matching",
        (
            Case((0, 1, 1), (1, 1, 1, 1),
                 lambda m, v: m[2].select_n(
                     m[1].array([0, 1]), _vec(m[1], v[:2]), _vec(m[1], v[2:])),
                 shapes=((2,), (2,), (2,))),
            Case((0, -1, -1), (-1, -1, -1, -1),
                 lambda m, v: m[2].select_n(
                     m[1].array([0, 1]), _vec(m[1], v[:2]), _vec(m[1], v[2:])),
                 shapes=((2,), (2,), (2,))),
        ),
        "A BIT COPY, PROVED RATHER THAN ASSUMED. `out[i]` is `cases[w[i]][i]` "
        "and the copy carries the sign bit with it: a `-0.0` selected out of "
        "either case arrives as `-0.0`. Both branches of the selector are "
        "taken by the constant `[0, 1]` in every execution, so neither arm "
        "is untested. The selector is an INDEX, not a value operand.",
    ),
    "gather": Row(
        "matching",
        (
            Case((1, 0), (1, 1),
                 lambda m, v: _vec(m[1], v)[m[1].array([0, 1])],
                 shapes=((2,), (1,))),
            Case((-1, 0), (-1, -1),
                 lambda m, v: _vec(m[1], v)[m[1].array([0, 1])],
                 shapes=((2,), (1,))),
        ),
        "A bit copy of whole rows of the data operand at in-range indices; "
        "no fill and no clamp is reachable in the admitted row form, so "
        "nothing but an operand element can come out. Executed on `-0.0` "
        "rows.",
    ),
    "scatter": Row(
        "matching",
        (
            Case((1, 0, 1), (1, 1, 1),
                 lambda m, v: _vec(m[1], v[:2]).at[0].set(_one(m[1], v[2])),
                 shapes=((2,), (1,), ())),
            Case((-1, 0, -1), (-1, -1, -1),
                 lambda m, v: _vec(m[1], v[:2]).at[0].set(_one(m[1], v[2])),
                 shapes=((2,), (1,), ())),
        ),
        "The admitted `x.at[k].set(v)` form at a static in-range `k`: the "
        "written position is a bit copy of the update and every other is a "
        "bit copy of the operand. Both are value operands and both must be "
        "certified, so both are enumerated over the matching zero.",
    ),
    "dynamic_slice": Row(
        "matching",
        (
            Case((1, 0), (1, 1),
                 lambda m, v: m[2].dynamic_slice(
                     _vec(m[1], v), (m[1].int32(0),), (2,)),
                 shapes=((2,), ())),
            Case((-1, 0), (-1, -1),
                 lambda m, v: m[2].dynamic_slice(
                     _vec(m[1], v), (m[1].int32(0),), (2,)),
                 shapes=((2,), ())),
        ),
        "A window of the data operand at a start the transfer has proved "
        "in-range, so jax's clamp is the identity and the output is a bit "
        "copy. The START is an index, not a value operand.",
    ),
    "dynamic_update_slice": Row(
        "matching",
        (
            Case((1, 1, 0), (1, 1, 1),
                 lambda m, v: m[2].dynamic_update_slice(
                     _vec(m[1], v[:2]), m[1].reshape(_one(m[1], v[2]), (1,)),
                     (m[1].int32(0),)),
                 shapes=((2,), (1,), ())),
            Case((-1, -1, 0), (-1, -1, -1),
                 lambda m, v: m[2].dynamic_update_slice(
                     _vec(m[1], v[:2]), m[1].reshape(_one(m[1], v[2]), (1,)),
                     (m[1].int32(0),)),
                 shapes=((2,), (1,), ())),
        ),
        "Written positions are bit copies of the update, unwritten ones bit "
        "copies of the operand; both are value operands.",
    ),
    "concatenate": Row(
        "matching",
        (
            Case((1, 1), (1, 1),
                 lambda m, v: m[1].concatenate(
                     [_vec(m[1], v), _vec(m[1], v)]),
                 shapes=((2,), (2,)), params=(("dimension", 0),)),
            Case((-1, -1), (-1, -1),
                 lambda m, v: m[1].concatenate(
                     [_vec(m[1], v), _vec(m[1], v)]),
                 shapes=((2,), (2,)), params=(("dimension", 0),)),
        ),
        "Every output element is an operand element at a static position, "
        "copied bitwise: a `-0.0` operand element arrives as `-0.0`, "
        "executed. A variadic member is poisoned by an EMPTY operand — no "
        "size-0 value is ever certified, and the agreement rule needs every "
        "value operand certified — so the empty case mints nothing and is "
        "not a configuration this question can be asked of.",
    ),
    "stack": Row(
        "matching",
        (
            Case((1, 1), (1, 1),
                 lambda m, v: m[1].stack([_vec(m[1], v), _vec(m[1], v)]),
                 shapes=((2,), (2,)), params=(("axis", 0),)),
            Case((-1, -1), (-1, -1),
                 lambda m, v: m[1].stack([_vec(m[1], v), _vec(m[1], v)]),
                 shapes=((2,), (2,)), params=(("axis", 0),)),
        ),
        "As `concatenate`, one axis up: the output is the two operands laid "
        "side by side along a new leading axis, every element a bitwise copy "
        "of an operand element. Executed on `-0.0` operands in both arms.",
    ),
    "split": Row(
        "matching",
        (
            Case((1,), (1, 1, 1, 1),
                 lambda m, v: m[1].split(_vec(m[1], v), 2),
                 shapes=((4,),), outvars=2, params=(("sizes", (2, 2)), ("axis", 0))),
            Case((-1,), (-1, -1, -1, -1),
                 lambda m, v: m[1].split(_vec(m[1], v), 2),
                 shapes=((4,),), outvars=2, params=(("sizes", (2, 2)), ("axis", 0))),
        ),
        "Multi-output, and the routing class's claim is quantified over "
        "every element of every output — so BOTH outputs are read here. "
        "Static slices of the operand; bit copies.",
    ),
    "unstack": Row(
        "matching",
        (
            Case((1,), (1, 1),
                 lambda m, v: m[1].unstack(_vec(m[1], v)),
                 shapes=((2,),), outvars=2),
            Case((-1,), (-1, -1),
                 lambda m, v: m[1].unstack(_vec(m[1], v)),
                 shapes=((2,),), outvars=2),
        ),
        "As `split`, one rank down: each output is one static index of the "
        "operand's leading axis, a bitwise copy. Multi-output, so both "
        "outputs are read — the routing class is the only class allowed "
        "to mint for an equation with more than one outvar, and its claim "
        "covers every element of every one of them.",
    ),
    "reshape": Row(
        "matching",
        (
            Case((1,), (1, 1), lambda m, v: m[1].reshape(_vec(m[1], v), (2, 1)),
                 shapes=((2,),)),
            Case((-1,), (-1, -1), lambda m, v: m[1].reshape(_vec(m[1], v), (2, 1)),
                 shapes=((2,),)),
        ),
        "A permutation of the element order; every element is copied bitwise. "
        "Executed on a subnormal too, because the class comment's own control "
        "for 'exactly two exceptions' is that `reshape` of `5e-324` is exact.",
    ),
    "transpose": Row(
        "matching",
        (
            Case((1,), (1, 1),
                 lambda m, v: m[1].transpose(m[1].reshape(_vec(m[1], v), (2, 1))),
                 shapes=((2,),)),
            Case((-1,), (-1, -1),
                 lambda m, v: m[1].transpose(m[1].reshape(_vec(m[1], v), (2, 1))),
                 shapes=((2,),)),
        ),
        "A permutation of the element ORDER and nothing else: the transfer "
        "reads `a.los[i]`/`a.his[i]` and writes them at a permuted position, "
        "and the target copies the bits. Executed on `-0.0` and on the "
        "smallest subnormal.",
    ),
    "broadcast_in_dim": Row(
        "matching",
        (
            Case((1,), (1, 1),
                 lambda m, v: m[1].broadcast_to(_vec(m[1], v), (2, 2)),
                 shapes=((2,),)),
            Case((-1,), (-1, -1),
                 lambda m, v: m[1].broadcast_to(_vec(m[1], v), (2, 2)),
                 shapes=((2,),)),
        ),
        "Each output element is a copy of one operand element. A broadcast "
        "onto an extent of 0 would be a size-0 output, which "
        "`_record_strict_sign` refuses to certify at all, so it is not a "
        "configuration this question can be asked of.",
    ),
    "slice": Row(
        "matching",
        (
            Case((1,), (1, 1),
                 lambda m, v: m[2].slice(_vec(m[1], v), (0,), (2,)),
                 shapes=((2,),)),
            Case((-1,), (-1, -1),
                 lambda m, v: m[2].slice(_vec(m[1], v), (0,), (2,)),
                 shapes=((2,),)),
        ),
        "A static window; bit copies. The empty window is a size-0 output "
        "and is refused certification, per "
        "`test_a_ROUTING_rule_that_produces_an_EMPTY_output_certifies_nothing`.",
    ),
    "squeeze": Row(
        "matching",
        (
            Case((1,), (1, 1),
                 lambda m, v: m[1].squeeze(
                     m[1].reshape(_vec(m[1], v), (2, 1)), axis=1),
                 shapes=((2,),)),
            Case((-1,), (-1, -1),
                 lambda m, v: m[1].squeeze(
                     m[1].reshape(_vec(m[1], v), (2, 1)), axis=1),
                 shapes=((2,),)),
        ),
        "A shape change with no element change — the flat element sequence "
        "is identical on both sides, so every output element is a bitwise "
        "copy of the operand element at the same flat index. Executed on "
        "`-0.0` and on the smallest subnormal.",
    ),
    "copy": Row(
        "matching",
        (
            Case((1,), (1, 1), lambda m, v: m[1].array(_vec(m[1], v)),
                 shapes=((2,),)),
            Case((-1,), (-1, -1), lambda m, v: m[1].array(_vec(m[1], v)),
                 shapes=((2,),)),
        ),
        "The identity. Measured rather than assumed because 'is a bit copy' "
        "is exactly the kind of claim this project keeps finding one "
        "indirection away from what the code does.",
    ),
    "stop_gradient": Row(
        "matching",
        (
            Case((1,), (1, 1), lambda m, v: m[2].stop_gradient(_vec(m[1], v)),
                 shapes=((2,),)),
            Case((-1,), (-1, -1), lambda m, v: m[2].stop_gradient(_vec(m[1], v)),
                 shapes=((2,),)),
        ),
        "The identity on the primal (`[ins[0]]` in the transfer, a bit copy "
        "on the target); it changes only what autodiff does downstream, "
        "which is not a value question at all. Executed anyway, because "
        "'this one is obviously the identity' is how the two `max`/`min` "
        "exceptions got into an unqualified class comment.",
    ),
}


# --- running one row ---------------------------------------------------------


class Measurement:
    """What one row's enumeration actually did. Every field is a count, so a
    row that quietly stopped producing zeros is visible as a number and not
    only as a bucket.

    ``executions`` counts operand assignments; ``elements`` and every counter
    below it count OUTPUT ELEMENTS ACROSS BOTH LEGS — each assignment is run
    eagerly and under `jit` and both results are read, so `elements` is twice
    the number of distinct output elements. Deliberate: the classification is
    over the union of the two lowerings, because a divergence under either is
    a divergence."""

    def __init__(self):
        self.executions = 0
        self.elements = 0
        self.zeros = 0
        self.wrong_signed_zeros = 0
        self.wrong_signed_nonzeros = 0
        self.nans = 0
        self.infinities = 0
        self.jit_disagreements: list[str] = []
        self.witnesses: list[str] = []
        self.minted: dict[int, int] = {}

    @property
    def bucket(self) -> str:
        if self.wrong_signed_zeros:
            return "diverges"
        return "matching" if self.zeros else "no-zero"


def _flat(jnp, out) -> list[float]:
    if isinstance(out, (list, tuple)):
        vals: list[float] = []
        for o in out:
            vals.extend(_flat(jnp, o))
        return vals
    return [float(v) for v in jnp.ravel(out)]


def _show(v: float) -> str:
    if v == 0.0:
        return "-0.0" if math.copysign(1.0, v) < 0 else "+0.0"
    return repr(v)


_CACHE: dict[str, Measurement] = {}


def _measure(prim: str) -> Measurement:
    """Execute every case of `prim` at every value assignment, eagerly and
    under `jit`, and read the sign bit off what came back.

    Memoized because three tests ask for the same answer and jax dispatch is
    the cost here; the memo holds a value, never a verdict."""
    if prim not in _CACHE:
        _CACHE[prim] = _run_row(prim, SWEEP[prim])
    return _CACHE[prim]


def _run_row(prim: str, row: "Row") -> Measurement:
    """The classifier itself, with the row passed in.

    Separated from :func:`_measure` so that
    `test_the_CLASSIFIER_separates_the_three_buckets` can drive it against
    rows built to land in each of them. A classifier only ever run on the
    shipped table is a classifier whose discrimination nobody has seen."""
    jax, jnp, lax = _f64_lax()
    mods = (jax, jnp, lax)
    m = Measurement()

    def go():
        for index, case in enumerate(row.cases):
            minted = _minted(prim, case.signs, case.shapes, case.params,
                             case.outvars)
            m.minted[index] = minted
            if not minted:
                continue
            jrun = jax.jit(lambda *a, case=case: case.run(mods, a))
            pools = [
                _candidates(s, len(case.slot_signs)) for s in case.slot_signs
            ]
            for vals in itertools.product(*pools):
                args = tuple(jnp.float64(v) for v in vals)
                eager = _flat(jnp, case.run(mods, args))
                jitted = _flat(jnp, jrun(*args))
                m.executions += 1
                if len(eager) != len(jitted) or any(
                    (a != b and not (math.isnan(a) and math.isnan(b)))
                    or (a == 0.0 and math.copysign(1.0, a)
                        != math.copysign(1.0, b))
                    for a, b in zip(eager, jitted)
                ):
                    m.jit_disagreements.append(
                        f"{prim} case {index} at {vals!r}: eager "
                        f"{[_show(v) for v in eager]} vs jit "
                        f"{[_show(v) for v in jitted]}"
                    )
                for v in eager + jitted:
                    m.elements += 1
                    if math.isnan(v):
                        m.nans += 1
                    elif math.isinf(v):
                        m.infinities += 1
                        if (v > 0) != (minted > 0):
                            m.wrong_signed_nonzeros += 1
                            m.witnesses.append(
                                f"{prim} case {index} at {vals!r}: certified "
                                f"{minted:+d}, executed {v!r}"
                            )
                    elif v == 0.0:
                        m.zeros += 1
                        if math.copysign(1.0, v) != float(minted):
                            m.wrong_signed_zeros += 1
                            m.witnesses.append(
                                f"{prim} case {index} at {vals!r}: certified "
                                f"{minted:+d}, executed {_show(v)}"
                            )
                    elif (v > 0) != (minted > 0):
                        m.wrong_signed_nonzeros += 1
                        m.witnesses.append(
                            f"{prim} case {index} at {vals!r}: certified "
                            f"{minted:+d}, executed {v!r}"
                        )

    _at_x64(go)
    return m


# --- the census's own rule, applied to the census's blind spot ---------------


def test_the_sweep_is_TOTAL_over_the_carrying_set():
    """A primitive with no row is not allowed.

    Derived from `_STRICT_SIGN_PRIMITIVES` at test time, so a primitive
    admitted to the census without an answer to this question reds here.
    Same rule as the census itself: absence must be impossible.

    **AND WHY THE CARRYING SET IS THE WHOLE SURFACE.** A primitive outside it
    cannot be a sibling, because a sibling needs a CERTIFIED value whose
    executed zero is wrong-signed and a non-carrier mints no certificate at
    all — `_strict_sign_out` returns 0 on its first line for anything not in
    this set. That is not taken on the code's word either: every member of
    `_SIGN_BOOLEAN` and `_SIGN_NO_RULE` is probed with EVERY operand
    certified and must still answer 0, in
    `tests/test_strict_sign_census.py::
    test_an_exempt_primitive_mints_nothing_even_with_every_operand_certified`,
    and that probe table is asserted total over both classes. So `sign`,
    `exp`, `pow`, `rem` and `convert_element_type` are out of this sweep by a
    checked argument rather than by omission — `convert_element_type` in
    particular can underflow a certified value to a zero on a narrowing
    conversion, and it drops the certificate at the same equation, so there
    is no certified value left for the zero to disagree with."""
    assert set(SWEEP) == set(P._STRICT_SIGN_PRIMITIVES), (
        f"unswept {sorted(set(P._STRICT_SIGN_PRIMITIVES) - set(SWEEP))}, "
        f"stale {sorted(set(SWEEP) - set(P._STRICT_SIGN_PRIMITIVES))}"
    )


def test_MAINS_TEN_are_all_still_carried_on_this_branch():
    """The frozen record is only usable while it is a SUBSET of what this
    tree carries — otherwise the sweep has no row for one of `main`'s
    carriers and the narrower answer would be silently short."""
    missing = sorted(MAIN_CARRIERS - set(P._STRICT_SIGN_PRIMITIVES))
    assert not missing, (
        f"{missing} carried the certificate on `main` at `a90862b` and are "
        f"not in this tree's carrying set, so the `main` answer below cannot "
        f"be read off this sweep. Re-derive both."
    )
    assert MAIN_CARRIERS < set(P._STRICT_SIGN_PRIMITIVES), (
        "the two carrying sets are no longer different, which makes the two "
        "answers below one answer told twice"
    )


def test_every_row_carries_an_ARGUMENT():
    """A bucket without a reason is a bucket nobody decided. Same rule
    `_SIGN_NO_RULE` is held to."""
    for prim, row in sorted(SWEEP.items()):
        assert row.bucket in ("no-zero", "matching", "diverges"), (prim, row.bucket)
        assert len(row.why) > 80, f"{prim}'s argument is too thin: {row.why!r}"
        assert row.cases, f"{prim} has no case"


@pytest.mark.parametrize("prim", sorted(SWEEP))
def test_the_declared_bucket_is_the_one_the_TARGET_puts_in_memory(prim):
    """THE SWEEP.

    One assertion, three possible answers, always live: the bucket the table
    declares must be the bucket the executions produced. A row cannot be
    parked in `"matching"` by assertion, and a row that stops producing
    zeros drops to `"no-zero"` and reds rather than passing quietly — which
    is this instrument's vacuity guard, since a green `"matching"` over zero
    executed zeros would be a claim about nothing.

    Every case must also MINT: a case whose certificate the rule stopped
    writing is a case measuring nothing, and it fails here by name."""
    pytest.importorskip("jax")
    row = SWEEP[prim]
    m = _measure(prim)
    unminted = sorted(i for i, s in m.minted.items() if not s)
    assert not unminted, (
        f"{prim}: case(s) {unminted} mint no certificate, so this row's "
        f"question is not being asked of them. Either the rule changed or "
        f"the case's operand certificates no longer reach it."
    )
    assert m.executions, f"{prim}: nothing was executed"
    assert not m.wrong_signed_nonzeros, (
        f"{prim}: a certified value executed as a NONZERO of the wrong "
        f"sign, which is a stronger defect than this sweep is looking "
        f"for:\n  " + "\n  ".join(m.witnesses[:5])
    )
    if row.nans:
        assert m.nans, (
            f"{prim} declares that it can execute as NaN under a "
            f"certificate and none of {m.elements} executed elements was "
            f"one, so the declaration is unfalsified prose"
        )
    else:
        assert not m.nans, (
            f"{prim}: {m.nans} of {m.elements} executed elements were NaN "
            f"under a certificate, and this row does not declare that. In ℝ "
            f"a certified value is a nonzero real; a NaN is not one."
        )
    assert m.bucket == row.bucket, (
        f"{prim}: declared {row.bucket!r}, measured {m.bucket!r} over "
        f"{m.executions} executions / {m.elements} elements "
        f"({m.zeros} executed zeros, {m.wrong_signed_zeros} of them "
        f"wrong-signed).\n  " + "\n  ".join(m.witnesses[:5] or ["(no witness)"])
    )


def test_every_row_actually_EXECUTED_a_zero_and_the_counts_are_reported():
    """VACUITY, per row, as a number rather than as a bucket.

    A row could hold its bucket while its enumeration produced no zeros at
    all — a jax version that folds differently, a candidate list that lost
    its zero — and `"matching"` would then be a claim about nothing. The
    bucket already collapses to `"no-zero"` in that case; this states the
    same thing as a count and prints the whole census when it fires, which
    is what a reader wants when one row moves."""
    pytest.importorskip("jax")
    report = []
    barren = []
    for prim in sorted(SWEEP):
        m = _measure(prim)
        report.append(
            f"{prim:22s} {m.executions:5d} executions  {m.elements:6d} "
            f"elements  {m.zeros:5d} zeros  {m.wrong_signed_zeros:4d} "
            f"wrong-signed  {m.nans:4d} NaN  {m.infinities:4d} inf  "
            f"-> {m.bucket}"
        )
        if not m.zeros:
            barren.append(prim)
    assert not barren, (
        f"{barren} executed no zero at all, so their bucket measures "
        f"nothing:\n" + "\n".join(report)
    )


def test_the_jit_leg_agrees_with_the_eager_one_except_where_DECLARED():
    """Two lowerings of one equation, and they do not always agree.

    Both halves: a row that declares a disagreement must SHOW one, and a row
    that does not declare one must not have any. The declaration is not a
    tolerance — it is where the tree records that the sign bit of an
    executed zero is not a function of the stelling IR alone."""
    pytest.importorskip("jax")
    undeclared, unrealised = [], []
    for prim in sorted(SWEEP):
        m = _measure(prim)
        if m.jit_disagreements and not SWEEP[prim].jit_disagrees:
            undeclared.append(f"{prim}: {m.jit_disagreements[0]}")
        if SWEEP[prim].jit_disagrees and not m.jit_disagreements:
            unrealised.append(prim)
    assert not undeclared, (
        "eager and `jit` returned different bits for the same equation, "
        "undeclared:\n  " + "\n  ".join(undeclared)
    )
    assert not unrealised, (
        f"{unrealised} declare an eager/jit disagreement and produced none, "
        f"so the declaration is unfalsified prose"
    )


# --- the two answers ---------------------------------------------------------


def _diverging(carriers) -> list[str]:
    return sorted(p for p in carriers if _measure(p).bucket == "diverges")


def test_THE_ANSWER_for_MAINS_TEN_carrying_primitives():
    """The narrower carrying set, and therefore the URGENT one: a sibling
    here is live on `main` today and is not a 0.3.0 artefact.

    Both directions. The named two must be there — a run that found neither
    would mean the class had been repaired somewhere nobody recorded — and
    nothing else may be, because an eleventh diverging primitive under
    `main`'s ten is a finding this branch has not made."""
    pytest.importorskip("jax")
    got = _diverging(MAIN_CARRIERS)
    assert got == ["dot_general", "reduce_sum"], (
        f"the answer for `main`'s ten carriers is {got}, and this branch "
        f"measured ['dot_general', 'reduce_sum']. A NEW name here is a "
        f"sibling live on `main`; a MISSING one means the class moved."
    )
    clean = sorted(set(MAIN_CARRIERS) - set(got))
    assert len(clean) == len(MAIN_CARRIERS) - 2, clean


def test_THE_ANSWER_for_THE_CENSUSS_THIRTY_ONE_carrying_primitives():
    """The wider carrying set. A sibling here that is NOT in the answer
    above would be a 0.3.0 blocker — a defect the census's widening
    introduced. There is none: the two diverging primitives are both
    `main`'s, and the twenty-one carriers 0.3.0 added are all clean.

    **THE CLAIM IS ABOUT THE PRIMITIVE SET AND ONLY ABOUT IT.** This
    docstring used to close *"this branch did not make the class bigger"*,
    which reads as a statement about the release and is false of the REACH:
    the widening lets programs that could not previously carry a certificate
    into the two diverging primitives carry one there now. Measured in
    `test_the_0_3_0_widening_admitted_no_new_SIBLING_but_did_widen_the_REACH`
    and in this module's docstring. The sentence that survives is the narrow
    one: **0.3.0 admitted no new diverging primitive.**"""
    pytest.importorskip("jax")
    got = _diverging(P._STRICT_SIGN_PRIMITIVES)
    assert got == ["dot_general", "reduce_sum"], (
        f"the answer for the census's carrying set is {got}, measured "
        f"['dot_general', 'reduce_sum'] on this branch"
    )
    new_in_0_3_0 = sorted(set(P._STRICT_SIGN_PRIMITIVES) - MAIN_CARRIERS)
    assert new_in_0_3_0, "the two carrying sets are the same set"
    assert not (set(got) & set(new_in_0_3_0)), (
        f"{sorted(set(got) & set(new_in_0_3_0))} diverge and were admitted "
        f"by the 0.3.0 census, so this release introduced a member of the "
        f"class rather than only inheriting it"
    )


def test_every_run_really_EXECUTES_the_primitive_it_is_a_row_for():
    """A row whose program does not contain its own primitive measures a
    neighbour and reports it under the wrong name.

    Decided by asking jax what it traced, not by reading the expression: the
    primitive's name must appear among the equations of
    `jax.make_jaxpr(run)`. Measured on jax 0.11.0 — all thirty-one of these
    names are jax's own primitive names, so no translation table stands
    between the census and the trace, and if one ever becomes necessary this
    reds rather than quietly matching nothing."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()
    mods = (jax, jnp, lax)

    def go():
        missing = []
        for prim, row in sorted(SWEEP.items()):
            for index, case in enumerate(row.cases):
                args = tuple(
                    jnp.float64(_candidates(s, len(case.slot_signs))[0])
                    for s in case.slot_signs
                )
                jx = jax.make_jaxpr(
                    lambda *a, case=case: case.run(mods, a)
                )(*args)
                names = {str(e.primitive) for e in jx.jaxpr.eqns}
                if prim not in names:
                    missing.append(f"{prim} case {index}: traced {sorted(names)}")
        return missing

    missing = _at_x64(go)
    assert not missing, (
        "row(s) whose executed program does not contain the primitive they "
        "answer for:\n  " + "\n  ".join(missing)
    )


def test_the_CLASSIFIER_separates_the_three_buckets():
    """THE POSITIVE CONTROL for this module's one assertion.

    `test_the_declared_bucket_is_the_one_the_TARGET_puts_in_memory` compares
    a declared bucket to a measured one, and it is worth exactly as much as
    the measurement's ability to come out differently. So the classifier is
    driven against three rows built to land in each of the three buckets,
    none of them the shipped table's:

    * `"diverges"` — `add`'s two same-signed operands summed by a
      two-element `reduce_sum` instead of by `add`. The certificate is still
      `add`'s `-1`; the execution is the seeded reduction.
    * `"matching"` — the shipped `add`, which is what the row says.
    * `"no-zero"` — `neg` of `x + 1.0`, which is bounded away from zero over
      the whole positive candidate list.

    Without this, a classifier that answered `"matching"` unconditionally
    would pass all thirty-one rows except the two that are declared to
    diverge, and would have been caught by nothing else."""
    pytest.importorskip("jax")
    jnp_diverges = Row(
        "diverges",
        (Case((-1, -1), (-1, -1),
              lambda m, v: m[1].sum(_vec(m[1], v))),),
        "a control, not a row of the sweep: `add`'s certificate over a "
        "seeded two-element reduction",
    )
    never_zero = Row(
        "no-zero",
        (Case((1,), (1,),
              lambda m, v: m[2].neg(
                  m[2].add(_one(m[1], v[0]), _one(m[1], 1.0)))),),
        "a control, not a row of the sweep: bounded away from zero",
    )
    got = {
        "diverges": _run_row("add", jnp_diverges).bucket,
        "matching": _measure("add").bucket,
        "no-zero": _run_row("neg", never_zero).bucket,
    }
    assert got == {"diverges": "diverges", "matching": "matching",
                   "no-zero": "no-zero"}, (
        f"the classifier does not separate the three buckets: {got}. Every "
        f"bucket assertion in this module is worth what this is."
    )


# --- the induction's two premises --------------------------------------------


def test_a_certified_value_really_CAN_execute_as_a_matching_signed_zero():
    """THE HYPOTHESIS IS NOT VACUOUS, driven end to end.

    Every row above assumes an operand certified `s` may execute as a zero
    whose sign bit is `s`. If that were unreachable, the whole sweep would
    be asking a question about nothing and its `"matching"` answers would be
    worth nothing. It is reachable through ordinary arithmetic on an
    ordinary declared box — no subnormal declaration required — and this is
    the chain every diverging row in
    `tests/test_strict_sign_census.py::ZERO_UNDER_CERTIFICATE` is built
    from."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()
    from stelling.harness import any_array, assert_, assume, trace

    def h():
        x = any_array((2,), jnp.float64, (-1.0, -0.25))
        assume(x < 0)
        return assert_(x * 1e-200 * 1e-200 < 0)

    cj = _at_x64(lambda: trace(h))
    p = _Propagator("constrain")
    p.run(cj.jaxpr, list(cj.consts), [])
    asserts = [e for e in cj.jaxpr.eqns if e.primitive == "stelling_assert"]
    pred = asserts[0].invars[0]
    cmp_eqn = next(
        e for e in cj.jaxpr.eqns if e.outvars and e.outvars[0].id == pred.id
    )
    value = cmp_eqn.invars[0]
    assert p.strict_sign.get(value.id) == -1, (
        f"the underflow chain is certified {p.strict_sign.get(value.id)}, "
        f"not -1; the sweep's hypothesis has lost its witness"
    )
    vals = _at_x64(lambda: [
        float(v) for v in jnp.ravel(
            jnp.array([-1.0, -0.25], dtype=jnp.float64) * 1e-200 * 1e-200
        )
    ])
    assert vals and all(v == 0.0 for v in vals), vals
    assert all(math.copysign(1.0, v) < 0 for v in vals), (
        f"the chain executed as {[_show(v) for v in vals]}; the hypothesis "
        f"is that a certified value can execute as a MATCHING-signed zero"
    )


def test_the_three_certificate_SOURCES_cannot_mint_a_wrong_signed_zero():
    """THE INDUCTION'S BASE CASE, which the per-primitive sweep does not
    reach and without which it proves nothing about a whole program.

    A certificate enters `strict_sign` from three places
    (`_Propagator._strict_sign_out`'s docstring names four writers; the
    fourth re-keys an already-minted one across a sub-jaxpr boundary and
    mints nothing new). Each is checked here in the only direction that can
    be wrong — that it never certifies a value whose executed representation
    is a zero of the OPPOSITE sign bit:

    * **a strict `assume` on a declared value.** The certified fact is
      `x < 0` (or `> 0`) over ℝ, so every point of the assumed region is a
      strictly signed real, and binary64's round-to-nearest carries a
      negative real to a negative float or to `-0.0` — never to `+0.0`.
      Driven with a real that underflows the format entirely.
    * **a LITERAL.** `_literal_strict_sign` reads the decoded value, and
      `-0.0 < 0` is False in ℝ and in Python, so a signed-zero literal is
      certified 0 and mints nothing. Both zeros checked.
    * **a CONSTVAR bind**, which goes through `_box_strict_sign` on the
      constant's own box; a box that touches zero answers 0.

    The one case that IS certified and executes as a zero — a subnormal
    literal the target flushes — keeps its sign bit, which is the same
    statement the sweep makes about every primitive."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()

    # 1. round-to-nearest never turns a strictly negative real into +0.0
    for tiny in (-1e-200 * 1e-200, -5e-324 / 2.0, float("-1e-400")):
        assert tiny == 0.0, tiny
        assert math.copysign(1.0, tiny) < 0, (
            f"a strictly negative real rounded to {_show(tiny)}; the assume "
            f"source's whole argument is that this cannot happen"
        )
    # 2. a literal
    for zero in (0.0, -0.0):
        got = P._literal_strict_sign(ir.Literal(val=zero, aval=F64))
        assert got == 0, (
            f"the literal {_show(zero)} is certified {got}; a signed zero is "
            f"not a strictly signed value and must mint nothing"
        )
    assert P._literal_strict_sign(ir.Literal(val=-1e-320, aval=F64)) == -1, (
        "a subnormal negative literal is no longer certified, so the "
        "interesting half of this case is gone"
    )
    # ...and the certified subnormal literal executes as a MATCHING zero
    flushed = _at_x64(
        lambda: float(lax.mul(jnp.float64(-1e-320), jnp.float64(1.0)))
    )
    assert flushed == 0.0 and math.copysign(1.0, flushed) < 0, (
        f"a certified `-1` subnormal literal executed as {_show(flushed)}"
    )
    # 3. a constvar's box
    for lo, hi in ((-0.0, -0.0), (0.0, 0.0), (-1.0, 0.0), (0.0, 1.0)):
        box = iv.IntervalArray(shape=(1,), los=(lo,), his=(hi,))
        assert P._box_strict_sign(box) == 0, (
            f"the box [{lo}, {hi}] touches zero and is certified "
            f"{P._box_strict_sign(box)}"
        )
    assert P._box_strict_sign(
        iv.IntervalArray(shape=(1,), los=(-1e-310,), his=(-1e-320,))
    ) == -1, "a wholly-negative subnormal box is no longer certified"


def test_a_certified_div_under_an_ORDINARY_declaration_executes_as_NaN():
    """AN ADJACENT FINDING, and its reachability is MEASURED, not disclaimed.

    `div` mints the XOR of its operands' certificates, and both operands may
    execute as matching-signed zeros. `0/0` is a NaN, and a NaN is not a
    value ℝ has: the certificate says the quotient is a nonzero real of a
    definite sign, and the target computes something that compares false
    against everything.

    **THIS TEST USED TO SAY REACHABILITY WAS NOT ESTABLISHED, AND THE ROUTE
    IT NAMED WAS THE WRONG ONE.** It read: *"WHAT IS NOT ESTABLISHED HERE IS
    REACHABILITY. For a query to reach this, the divisor's own interval would
    have to be certified nonzero while containing the zero its value flushes
    to, and `_t_div` declines a divisor box that straddles zero unless a
    certificate excludes it. This module asks the LOCAL question and answers
    it; whether a whole query can get here is not measured, and this
    docstring is the disclosure rather than the claim."*

    Two things were wrong with that. The route it named is **structurally
    impossible** — a certified value that flushes has the box `[0, 0]`, and
    :func:`stelling.interval.boundary_div` raises on exactly that, so every
    construction of that shape comes back UNKNOWN. And the route that DOES
    work needs none of it: the divisor's box can exclude zero entirely and
    still flush at runtime, because the flush belongs to the KERNEL and not
    to the box. An ordinary declaration scaled into the subnormal band is
    enough.

    DRIVEN, jax 0.11.0 CPU binary64, this exact query::

        x = any_array((), float64, (-1.0, -0.25));  assume(x < 0)
        y = x * 1e-200 * 1e-108     box [-1.0000000000000004e-308,
                                         -2.499999999999997e-309]
                                    SUBNORMAL, and it EXCLUDES zero
        assert_(y / y > 0)          ->  discharged

    `y` is certified `-1` and the quotient `+1` — this row's configuration
    exactly — the divisor box never straddles zero so `boundary_div` is not
    involved at all, and at every sampled point of the assumed region the
    target computes `y = -0.0` and `y / y = nan`. **No note fires**:
    `_SUBNORMAL_TELL_ROWS` holds the six COMPARISONS only, so nothing in the
    verdict, the notes or the assumptions tells the user a subnormal was
    reached.

    NOT REPAIRED HERE, and not this item's to repair: it is a different class
    from the wrong-signed zero (the executed value is not a zero at all), it
    needs no change to any sign rule, and the surface it would want is
    `_subnormal_flush_tell`, whose predicate and wording are
    comparison-shaped. This is the disclosure, and it is now a driven one."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()
    from stelling.harness import any_array, assert_, assume, trace

    # 1. the LOCAL fact
    got = _at_x64(lambda: float(lax.div(jnp.float64(0.0), jnp.float64(0.0))))
    assert math.isnan(got), (
        f"lax.div(+0.0, +0.0) is {got!r}; the `div` row declares that it can "
        f"execute as NaN under a certificate and that declaration now rests "
        f"on nothing"
    )
    m = _measure("div")
    assert m.nans and m.infinities, (
        f"the `div` sweep saw {m.nans} NaN and {m.infinities} infinities; "
        f"both are what a certified-nonzero operand executing as a zero "
        f"produces, and a run with neither is not exercising the case"
    )

    # 2. the REACHABILITY, end to end, from an ordinary declaration.
    #
    # ONE `chain`, used by the traced harness AND by the point evaluation.
    # They were written twice, and a mutation that changed only the harness
    # left this test green while the executed half went on measuring the
    # ORIGINAL expression — the "a check whose two halves are two copies"
    # shape. Driven: `1e-200 * 1e-108` -> `1e-20 * 1e-10` in the harness
    # alone used to pass.
    def chain(v):
        return v * 1e-200 * 1e-108

    def h():
        x = any_array((), jnp.float64, (-1.0, -0.25))
        assume(x < 0)
        y = chain(x)
        return assert_(y / y > 0)

    cj = _at_x64(lambda: trace(h))
    report = _at_x64(lambda: P.propagate(cj, semantics="real"))
    statuses = [o.status for o in report.obligations]
    assert statuses == ["discharged"], (
        f"the NaN query now reports {statuses}. If it stopped discharging, "
        f"this disclosure is out of date in the GOOD direction — re-measure "
        f"the paragraph above rather than deleting it."
    )

    p = _Propagator("constrain")
    p.run(cj.jaxpr, list(cj.consts), [])
    asserts = [e for e in cj.jaxpr.eqns if e.primitive == "stelling_assert"]
    pred = asserts[0].invars[0]
    cmp_eqn = next(
        e for e in cj.jaxpr.eqns if e.outvars and e.outvars[0].id == pred.id
    )
    quotient = cmp_eqn.invars[0]
    producer = next(
        e for e in cj.jaxpr.eqns
        if any(o.id == quotient.id for o in e.outvars)
    )
    assert producer.primitive == "div", (
        f"the compared value is produced by {producer.primitive!r}, so this "
        f"reproduction no longer exercises the `div` row"
    )
    assert p.strict_sign.get(quotient.id) == 1, (
        f"the quotient is certified {p.strict_sign.get(quotient.id)}, not +1 "
        f"— this is no longer the `div` row's configuration"
    )
    divisor_box = p.env[producer.invars[1].id]
    assert not iv.straddles_zero(divisor_box), (
        f"the divisor box {(divisor_box.los[0], divisor_box.his[0])} "
        f"straddles zero, which would route this through `boundary_div`. "
        f"The whole point of this reproduction is that it does NOT: the box "
        f"excludes zero and the runtime flushes anyway."
    )

    points = (-1.0, -0.5, -0.25)
    flushed = _at_x64(lambda: [
        float(chain(jnp.float64(pt))) for pt in points
    ])
    quotients = _at_x64(lambda: [
        float(chain(jnp.float64(pt)) / chain(jnp.float64(pt)))
        for pt in points
    ])
    assert all(v == 0.0 and math.copysign(1.0, v) < 0 for v in flushed), (
        f"the certified value executes as {[_show(v) for v in flushed]}, not "
        f"as a negative zero, so the 0/0 configuration is not reached"
    )
    assert all(math.isnan(q) for q in quotients), (
        f"the executed quotients are {quotients}; the disclosure is that a "
        f"DISCHARGED obligation is NaN at every sampled point of the assumed "
        f"region"
    )
    told = [n for n in report.notes if "subnormal" in n.lower()]
    assert not told, (
        f"a subnormal note now fires for this query: {told}. That is an "
        f"improvement, and the claim above that nothing tells the user must "
        f"be corrected rather than left standing."
    )


def test_the_0_3_0_widening_admitted_no_new_SIBLING_but_did_widen_the_REACH():
    """THE SENTENCE A RELEASE NOTE GETS BUILT FROM, measured in both halves.

    The sweep's two answers are about PRIMITIVES: no carrier 0.3.0 added
    diverges. That is a real and useful fact and it is the whole of what the
    sweep can say, because the sweep asks a LOCAL question and says nothing
    about which local configurations a query can reach.

    Reach is a separate question with a separate answer, and it is not the
    reassuring one. Admitting `stack`, `broadcast_in_dim` and the rest of
    the routing class means a certificate now SURVIVES constructions it used
    to die in, and some of those constructions end at `reduce_sum`. The
    diverging primitive is `main`'s; the program that reaches it is not.

    Driven here rather than argued: the same `src/`, the same traced query,
    with only `_STRICT_SIGN_PRIMITIVES` swapped between the two carrying
    sets. The certificate's presence is what flips the verdict, and the
    verdict flips to a DISCHARGED obligation the compiled program makes
    false at every sampled point of the assumed region."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()
    from stelling.harness import any_array, assert_, assume, trace

    def build(reshape):
        def h():
            u = any_array((), jnp.float64, (-1.0, -0.25))
            assume(u < 0)
            v = reshape(u)
            return assert_(1.0 / jnp.sum(v * 1e-200 * 1e-200) < 0.0)
        return h

    routes = {
        "stack": lambda u: jnp.stack([u, u]),
        "broadcast_in_dim": lambda u: jnp.broadcast_to(u, (3,)),
    }
    widened = []
    for name, reshape in sorted(routes.items()):
        cj = _at_x64(lambda reshape=reshape: trace(build(reshape)))
        assert name in {e.primitive for e in cj.jaxpr.eqns}, (
            f"the {name!r} route no longer traces to {name!r}; it cannot "
            f"demonstrate anything about that carrier"
        )
        assert name in P._STRICT_SIGN_PRIMITIVES - MAIN_CARRIERS, (
            f"{name!r} is not one of the carriers 0.3.0 added, so this route "
            f"is not about the widening"
        )
        verdicts = {}
        for label, carriers in (("main", MAIN_CARRIERS),
                                ("census", frozenset(P._STRICT_SIGN_PRIMITIVES))):
            saved = P._STRICT_SIGN_PRIMITIVES
            P._STRICT_SIGN_PRIMITIVES = carriers
            try:
                verdicts[label] = _at_x64(
                    lambda: P.propagate(cj, semantics="real")
                ).obligations[0].status
            finally:
                P._STRICT_SIGN_PRIMITIVES = saved
        if verdicts["main"] != verdicts["census"]:
            widened.append((name, verdicts["main"], verdicts["census"]))
        # ...and the program disagrees with the wider verdict
        executed = _at_x64(lambda reshape=reshape: [
            float(1.0 / jnp.sum(reshape(jnp.float64(pt)) * 1e-200 * 1e-200))
            for pt in (-1.0, -0.5, -0.25)
        ])
        assert all(not (q < 0.0) for q in executed), (
            f"{name}: the obligation `1/sum < 0` is TRUE somewhere in "
            f"{executed}, so this route no longer exhibits the defect"
        )

    assert widened, (
        "neither route's verdict differs between the two carrying sets, so "
        "0.3.0's widening buys no reach here and this module's docstring "
        "overstates it. Re-measure the paragraph rather than deleting it."
    )
    for name, before, after in widened:
        assert (before, after) == ("unknown", "discharged"), (
            f"{name}: the widening moves the verdict {before!r} -> {after!r}, "
            f"and this branch measured 'unknown' -> 'discharged'"
        )
    # the OTHER half, and the one the sweep's answers assert: no new sibling
    added = frozenset(P._STRICT_SIGN_PRIMITIVES) - MAIN_CARRIERS
    assert not [p for p in added if SWEEP[p].bucket == "diverges"], (
        "a carrier 0.3.0 added now diverges, which makes the release the "
        "author of a sibling rather than only of its reach"
    )


# --- the THIRD lowering, and what it does and does not change ---------------

def _fold_row(prim: str, row: "Row"):
    """Run one row's whole enumeration a third way — `jit` over a closure
    taking NO arguments, so XLA constant-folds the operands into the graph.

    Returns ``(executions whose bits differ from eager, wrong-signed zeros)``.
    """
    jax, jnp, lax = _f64_lax()
    mods = (jax, jnp, lax)
    differing = wrong = 0
    for case in row.cases:
        minted = _minted(prim, case.signs, case.shapes, case.params,
                         case.outvars)
        if not minted:
            continue
        pools = [
            _candidates(s, len(case.slot_signs)) for s in case.slot_signs
        ]
        for vals in itertools.product(*pools):
            args = tuple(jnp.float64(v) for v in vals)
            eager = _flat(jnp, case.run(mods, args))
            folded = _flat(jnp, jax.jit(
                lambda case=case, vals=vals: case.run(
                    mods, tuple(jnp.float64(v) for v in vals))
            )())
            if len(eager) != len(folded) or any(
                (a != b and not (math.isnan(a) and math.isnan(b)))
                or (a == 0.0
                    and math.copysign(1.0, a) != math.copysign(1.0, b))
                for a, b in zip(eager, folded)
            ):
                differing += 1
            for v in folded:
                if v == 0.0 and math.copysign(1.0, v) != float(minted):
                    wrong += 1
    return differing, wrong


_FOLD_CACHE: dict[str, tuple[int, int]] = {}


def _folded(prim: str) -> tuple[int, int]:
    if prim not in _FOLD_CACHE:
        _FOLD_CACHE[prim] = _at_x64(lambda: _fold_row(prim, SWEEP[prim]))
    return _FOLD_CACHE[prim]


def test_the_THIRD_lowering_changes_the_bits_and_changes_NO_CLASSIFICATION():
    """The sweep executes two lowerings; a third exists; this is what it does.

    **THE SOUNDNESS QUESTION AND THE CURIOSITY ARE DIFFERENT QUESTIONS AND
    ARE ANSWERED SEPARATELY.** Over the whole enumeration, run a third way:

    * the BITS differ on many rows, and **WHICH rows is a fact about the jax
      in the room, not about this tree**;
    * a wrong-signed zero under a certificate appears only in rows this table
      already classifies `"diverges"`. So **the third lowering adds no
      sibling**: the answer over three lowerings is the answer over two, and
      both reported answers stand.

    Only the second is asserted, and that is a correction. This check first
    named the differing rows in a frozen set and asserted set EQUALITY —
    which made it a check whose input is the developer's jax version, the
    shape this project's rules forbid. MEASURED, same tree, same values, the
    two lanes disagree:

        jax 0.11.0   11 rows differ: add add_any div dot_general max min mul
                                     reduce_sum scatter-add sqrt sub
        jax 0.10.2   20 rows differ: the above minus scatter-add and sqrt,
                                     plus broadcast_in_dim copy dynamic_slice
                                     dynamic_update_slice gather scatter
                                     select_n slice split squeeze
                                     stop_gradient

    The equality assertion passed on 0.11.0 and reddened the whole jax-0.10
    lane. The count is not the point and never was: DAZ/FTZ is a property of
    the runtime kernel, the constant folder does not share it, and HOW MUCH
    of the graph a given XLA folds is a version-to-version implementation
    choice. What must hold on every version is the subset below, and it does
    — on 0.11.0 the folded leg's wrong-signed zeros are `{dot_general}`, on
    0.10.2 `{dot_general, reduce_sum}`, and both are inside the declared
    diverging set.

    That subset is also why the third leg is a check of its own rather than a
    third leg inside `_measure`: folding it in would have merged a benign,
    version-dependent fact with the load-bearing one, and left every row's
    counts unable to tell a reader which was which."""
    pytest.importorskip("jax")
    differ, wrong = set(), {}
    for prim in sorted(SWEEP):
        d, w = _folded(prim)
        if d:
            differ.add(prim)
        if w:
            wrong[prim] = w
    assert differ, (
        "the constant-folded lowering returned bit-identical results to the "
        "runtime legs on all 31 rows, so this third leg is measuring nothing "
        "and the classification below is a subtraction over an empty set"
    )
    already = {p for p in SWEEP if SWEEP[p].bucket == "diverges"}
    assert set(wrong) <= already, (
        f"the constant-folded lowering produces a wrong-signed zero under a "
        f"certificate in {sorted(set(wrong) - already)}, which this table "
        f"classifies as clean. That is a SIBLING the two executed legs cannot "
        f"see, and both reported answers are short by it."
    )
    assert wrong, (
        "the folded leg produced no wrong-signed zero anywhere, so this "
        "check's subtraction is over an empty set and demonstrates nothing"
    )


def test_the_SAME_reduction_compiles_AT_LEAST_THREE_WAYS_and_they_disagree_on_the_sign_bit():
    """THE SHARPEST STATEMENT IN THIS MODULE of why the question had to be
    decided by running the program.

    **THIS TEST USED TO SAY "TWO WAYS" AND NAME ONLY `reduce_sum`.** It read
    *"One stelling-level `reduce_sum` over an extent-2 operand of `-0.0`
    compiles two ways"*, and the module declared `jit_disagrees` for
    `reduce_sum` alone while `dot_general`'s row carried the unqualified
    claim that its sum "is seeded with `+0.0` exactly as `reduce_sum`'s is".
    Both primitives split across lowerings. They split at DIFFERENT legs, and
    the two legs the sweep executes both land on the same side for
    `dot_general` — so the sweep's own `jit_disagrees` column gave it a clean
    bill. The count is "at least three" and not "three" because nothing here
    enumerates the lowerings; it exhibits them.

    MEASURED on jax 0.11.0 CPU binary64, and again on 0.10.2 with identical
    answers, under a `-1` certificate in every cell:

        lowering                  reduce_sum([-0.,-0.])   dot_general(...)
        eager                           +0.0                   +0.0
        jit, scalar parameters          -0.0                   +0.0
        jit, operand a parameter        +0.0                   +0.0
        jit, constant-folded            -0.0                   -0.0

    The scalar-parameter form of the reduction compiles to
    ``ROOT %reduce_sum.1 = f64[] add(%param_0, %param_1)`` inside a fusion,
    with no `constant(0)` operand: the reduction, and its seed, are gone. So
    the sign bit of an executed zero is NOT a function of the IR the
    propagator sees, for either primitive. A repair that reasons "XLA seeds
    reductions with `+0.0`, therefore ..." is reasoning about one of at least
    three lowerings of the same equation.

    This reddens if any cell of that table moves, which is the point."""
    pytest.importorskip("jax")
    jax, jnp, lax = _f64_lax()

    def dot(a, b):
        return lax.dot_general(a, b, (((0,), (0,)), ((), ())))

    def go():
        neg2 = jnp.array([-0.0, -0.0], dtype=jnp.float64)
        lhs = jnp.array([-1.0, -1.0], dtype=jnp.float64)
        rhs = jnp.array([0.0, 0.0], dtype=jnp.float64)
        z = jnp.float64(-0.0)
        return {
            ("reduce_sum", "eager"): float(jnp.sum(neg2)),
            ("reduce_sum", "jit/scalars"): float(
                jax.jit(lambda a, b: jnp.sum(jnp.stack([a, b])))(z, z)),
            ("reduce_sum", "jit/array"): float(jax.jit(jnp.sum)(neg2)),
            ("reduce_sum", "jit/folded"): float(
                jax.jit(lambda: jnp.sum(jnp.array([-0.0, -0.0])))()),
            ("dot_general", "eager"): float(dot(lhs, rhs)),
            ("dot_general", "jit/scalars"): float(
                jax.jit(lambda a, b, c, d: dot(
                    jnp.stack([a, b]), jnp.stack([c, d])))(
                        jnp.float64(-1.0), jnp.float64(-1.0),
                        jnp.float64(0.0), jnp.float64(0.0))),
            ("dot_general", "jit/array"): float(jax.jit(dot)(lhs, rhs)),
            ("dot_general", "jit/folded"): float(jax.jit(
                lambda: dot(jnp.array([-1.0, -1.0]),
                            jnp.array([0.0, 0.0])))()),
        }

    got = _at_x64(go)
    want = {
        ("reduce_sum", "eager"): +1.0,
        ("reduce_sum", "jit/scalars"): -1.0,
        ("reduce_sum", "jit/array"): +1.0,
        ("reduce_sum", "jit/folded"): -1.0,
        ("dot_general", "eager"): +1.0,
        ("dot_general", "jit/scalars"): +1.0,
        ("dot_general", "jit/array"): +1.0,
        ("dot_general", "jit/folded"): -1.0,
    }
    moved = []
    for key, value in sorted(got.items()):
        if value != 0.0:
            moved.append(f"{key[0]} {key[1]}: {value!r} is not a zero at all")
        elif math.copysign(1.0, value) != want[key]:
            moved.append(
                f"{key[0]} {key[1]}: {_show(value)}, and this tree measured "
                f"{'+0.0' if want[key] > 0 else '-0.0'}"
            )
    assert not moved, "the lowering table moved:\n  " + "\n  ".join(moved)
    # BOTH primitives must actually SPLIT, or "at least three" is a claim
    # about a table in which nothing disagrees
    for prim in ("reduce_sum", "dot_general"):
        bits = {math.copysign(1.0, v) for k, v in got.items() if k[0] == prim}
        assert bits == {-1.0, 1.0}, (
            f"{prim} returns the same sign bit under every lowering here, so "
            f"it no longer demonstrates that the sign bit is not a function "
            f"of the equation"
        )


def test_every_test_name_THIS_ITEM_cites_in_its_own_prose_resolves():
    """A citation nobody reads is a claim about nothing, and this one bit.

    `tests/test_prose_hygiene.py` checks backticked test names in SHIPPED
    prose — `SOUNDNESS.md`, `docs/`, the root `README.md`. It does not read
    test-module docstrings, so a name cited by one test file about another is
    unchecked.

    THIS IS NOT HYPOTHETICAL AND IT IS WHY THE CHECK EXISTS. While this
    item's fixes were being written, an edit that replaced a block by index
    truncated the tail of this module: the third-lowering machinery and two
    whole tests went with it, and **three citations of the
    three-lowerings test — one of them the `xfail` reason string pytest
    prints in the summary line — were left pointing at nothing.** (Its name
    is not spelled here: this scan is strict about line wraps, deliberately,
    and the rule that buys the strictness is below.) The suite stayed
    green: the file still imported, the surviving tests still passed, and no
    gate in the tree reads a docstring. It was found by a red-drive anchor
    failing to match, which is luck.

    Scope: the three modules this item owns. Tree-wide is
    `test_prose_hygiene.py`'s business and widening it is not this item's."""
    import ast
    import re

    owned = (
        _TESTS_ROOT / "test_executed_sign_bit_sweep.py",
        _TESTS_ROOT / "test_strict_sign_census.py",
        _TESTS_ROOT / "property" / "test_strict_sign_property.py",
    )
    # a citation resolves to a test FUNCTION or to a test MODULE — both are
    # things this item's prose points at, and `test_prose_hygiene.py`'s own
    # resolver is built the same way (`names.add(path.stem)` beside the
    # defined function names)
    defined = set()
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        defined.add(path.stem)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file reds first
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)

    # a cited name may be wrapped across lines inside a docstring or split
    # across adjacent string literals, so the source is de-wrapped first
    cited, dangling = set(), []
    for path in owned:
        text = path.read_text(encoding="utf-8")
        flat = re.sub(r'"\s*\n\s*(?:f?")?', "", text)
        flat = re.sub(r"\s*\n\s*#?\s*", " ", flat)
        for name in re.findall(r"\btest_[A-Za-z0-9_]{8,}\b", flat):
            cited.add(name)
            if name not in defined:
                dangling.append(f"{path.name}: {name}")

    assert not dangling, (
        "test name(s) cited in this item's prose that no test defines:\n  "
        + "\n  ".join(sorted(set(dangling)))
        + "\n\nEither the citation is stale — the thing it names is gone, "
        "which is the case this check exists for — or the NAME IS WRAPPED "
        "ACROSS A LINE. This scan de-wraps across adjacent string literals "
        "(`\"...foo_\"` / `\"bar...\"`) and NOT across a plain line break "
        "inside a docstring, because a fuzzy join is a scan that can excuse "
        "a real stale citation by accident. So the house rule this check "
        "buys is: a cited symbol goes on one line, or is split only at a "
        "string-literal boundary."
    )
    assert len(cited) > 10, (
        f"only {len(cited)} test name(s) were found cited across "
        f"{[p.name for p in owned]}; the scan has stopped reading the prose "
        f"it is supposed to check"
    )
