# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE STRICT-SIGN CERTIFICATE, SEARCHED.

The property is one sentence: **if `_Propagator.strict_sign` says a value is
strictly positive, then it is strictly positive at every point of the assumed
region** — and the same for negative. It is checked by building a small
program over the census's rules, running the propagator, and evaluating the
same program in exact `Fraction` arithmetic at points of the assumed region.
Exact rationals, so no rounding can mask a violation and none can invent one.

The certificate is real-mode only and reaches exactly one consumer, `div`'s
boundary gate — which returns a HALF-INFINITE box, so a wrong certificate
can mint a false REFUTED as well as a false VERIFIED. That is why this
property checks the CERTIFICATE and not a verdict: the certificate is
upstream of both directions.

**THE FIRST ENTRY IN THE LIST BELOW IS THE ONE THAT WAS MISSING, AND IT IS
THE ONE THAT MATTERS MOST.** This module's NOT-covered list used to open at
`sqrt` and say nothing at all about what an EXACT-RATIONAL oracle is
structurally unable to express. A reader who took a null from this search as
a statement about signed zeros was being misled by an instrument that could
not have spoken, and that is exactly what happened: a seeded-reduction
defect — a `-1`-certified value whose executed zero carries the `+` sign bit
— sat under this search and under
`tests/test_strict_sign_census.py::ZERO_UNDER_CERTIFICATE` at the same time,
and neither could draw it.

NOT covered:

* **ANY DEFECT WHOSE SUBJECT IS A FLOAT SIGN BIT, AND NO WIDENING OF THIS
  SEARCH CAN CHANGE THAT.** The oracle evaluates in `Fraction`, and ℚ has no
  signed zero: `(+0) + (−0)` is `0`, full stop, and `Fraction(0) < 0` is
  False whichever way the bits went. So this property cannot express the
  proposition "the executed zero carries the wrong sign bit", let alone
  falsify it. Drawing more primitives here — which the entry below does —
  closes an ℝ-side coverage gap and does NOT make this the guard for that
  class. The guard for it must EXECUTE on the target and read the sign bit:
  `tests/test_executed_sign_bit_sweep.py` does that for every carrying
  primitive, and `tests/test_strict_sign_census.py::
  test_an_executed_zero_under_a_certificate_carries_the_CERTIFIED_sign_bit`
  does it end to end. **A null from this file is evidence about ℝ and about
  nothing else.**
* **`sqrt`, `exp`, `pow`** — anything whose value leaves the rationals. The
  evaluator here is exact or it is nothing, and a floating `math.sqrt` would
  put the propagator's own rounding question back inside the instrument. The
  `sqrt` rule is covered by hand cases in
  `tests/test_assume_bump_boundary_div.py` and by an executed-jax case in
  `tests/test_strict_sign_census.py`; `exp` and `pow` carry no rule.
* **the INDEXING members of the routing class** — `gather`, `scatter`,
  `scatter-add`, `dynamic_slice`, `dynamic_update_slice`. Their admitted
  forms carry dimension-number params and row geometry, and re-implementing
  that geometry inside this evaluator would make the instrument a second
  copy of the thing under test. They are checked against EXECUTED jax in
  `tests/test_strict_sign_census.py`.
* *`dot_general` and `reduce_sum` USED TO BE HERE.* The entry read:
  *"**`dot_general`, `reduce_sum` over generated shapes** — every value here
  has one fixed extent, so no contraction or reduction is drawn. Both rules
  have hand cases elsewhere."* That was true and it is no longer: the extent
  is DRAWN (`k` in `_specs`, 1 to 4), and `sum_broadcast` and
  `dot_broadcast` emit a real `reduce_sum` / `dot_general` down to a scalar
  and a `broadcast_in_dim` back to `(k,)`, so both rules are searched over
  generated extents, and each rule's SIZE GUARD is searched **on its
  admitting side only, at `k = 1`**. *This entry first said "both of their
  SIZE guards are searched", which is not supported:* `EXTENTS` starts at 1,
  the refusing side is `size == 0`, and DRIVEN — with `reduce_sum`'s
  `ins[0].size > 0` guard deleted outright, this property still PASSES. The
  refusing side is unreachable from this grammar by construction and not by
  omission: no size-0 value is ever certified
  (`_Propagator._record_strict_sign`), and `_box_strict_sign` answers 0 on an
  empty box, so a drawn chain cannot present a certified empty operand to a
  reduction. Those two are driven in
  `tests/test_strict_sign_census.py::
  test_the_one_writer_refuses_a_size_0_value_and_accepts_a_sized_one` and
  `tests/test_strict_sign_census.py::
  test_a_ROUTING_rule_that_produces_an_EMPTY_output_certifies_nothing`.
  What the widening does not buy is the entry above it: in ℚ a reduction of
  negatives is negative at every extent, so the sign-bit class stays
  invisible here by construction.
* **ieee semantics.** The certificate is never written under ieee (the call
  site short-circuits), so there is nothing here to search.
* **anything about VERDICTS.** This property never runs
  `preconditions.check`; a rule that is sound and useless passes it.
* **rank > 1.** Every drawn value is rank 1, so a routing rule that is wrong
  only on a multi-axis operand is not reached. *This entry used to add "with
  a fixed extent"; the extent is drawn now and the rank is not.*

The search is over PROGRAMS and BOXES both: the declared extent, the declared
upper bound, the operator chain, the constants, the operand picks and the
sample points are all drawn.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

pytest.importorskip("hypothesis", reason="needs hypothesis")

from hypothesis import example, given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import _profiles  # noqa: E402
import _runner  # noqa: E402

from stelling import ir  # noqa: E402
from stelling.propagate import _Propagator  # noqa: E402


EXTENTS = (1, 2, 3, 4)
"""The rank-1 extents a spec may draw.

**THIS USED TO BE `K = 3`, A MODULE CONSTANT**, whose docstring read *"The
one extent every drawn value has. Rank 1, fixed, by construction — see the
module docstring's NOT-covered list."* Fixed by construction is what kept
`reduce_sum` and `dot_general` out of the grammar: with one extent there is
no contraction or reduction to draw. `1` is in the list deliberately — an
extent-1 reduction is the SIZE guard's boundary on the admitting side, and
an empty one cannot be drawn at all because no size-0 value is ever
certified.
"""

SCALAR = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")


def _avals(k):
    """``(value, concatenated, boolean)`` avals for extent ``k``."""
    return (
        ir.Aval(kind="ShapedArray", shape=(k,), dtype="float64"),
        ir.Aval(kind="ShapedArray", shape=(2 * k,), dtype="float64"),
        ir.Aval(kind="ShapedArray", shape=(k,), dtype="bool"),
    )

# Every op preserves the rank-1 extent, which is what lets the chain be
# drawn freely without a shape solver. `concat_slice` and `concat_split`
# are round trips: they emit the shape-changing routing equations and come
# back to `(K,)`, so `concatenate`, `slice` and `split` are all exercised.
UNARY = ("neg", "abs", "square", "copy", "stop_gradient", "reshape")
BINARY = ("add", "sub", "mul", "max", "min")
OPS = UNARY + BINARY + (
    "div_by_const", "integer_pow", "select_n", "concat_slice", "concat_split",
    # The two REDUCTIONS, each a round trip: down to a scalar and back to
    # `(k,)` through `broadcast_in_dim`, so the pool stays homogeneous the
    # way `concat_slice` keeps it. The scalar itself is certified and is
    # checked — it is in `produced` — so the reduction's own output is what
    # the property reads, not only the broadcast copy of it.
    "sum_broadcast", "dot_broadcast",
    # `sub_opposite` is a BIASED draw and is here for a measured reason:
    # `sub`'s rule fires only on OPPOSITE-signed operands, and the unbiased
    # grammar produced that pair 2 times in 401 examples (measured
    # 2026-08-28, budget 400) — under the floor, and under it for the one
    # rule this item is most about. This op emits `neg(b)` and then
    # `sub(a, neg b)`, which is a shape the unbiased grammar can already
    # build; drawing it directly is the "pin it and say so" rule applied to
    # a grammar rather than to an `@example`.
    "sub_opposite",
)

CONSTS = (Fraction(1, 2), Fraction(1), Fraction(2), Fraction(-1),
          Fraction(-3), Fraction(0), Fraction(4))
NONZERO_CONSTS = tuple(c for c in CONSTS if c)


@st.composite
def _specs(draw):
    """One drawn program, box and sample set, as a single tuple.

    Drawn as a tuple rather than through `st.data()` so that `@example`
    below is possible — the suite's rule 3."""
    sign = draw(st.sampled_from((1, -1)))
    k = draw(st.sampled_from(EXTENTS))
    hi = draw(st.sampled_from((1, 2, 4)))
    depth = draw(st.integers(min_value=1, max_value=6))
    ops = tuple(draw(st.sampled_from(OPS)) for _ in range(depth))
    consts = tuple(draw(st.sampled_from(CONSTS)) for _ in range(depth))
    nz = tuple(draw(st.sampled_from(NONZERO_CONSTS)) for _ in range(depth))
    ys = tuple(draw(st.sampled_from((0, 2, 3))) for _ in range(depth))
    picks = tuple(draw(st.integers(0, 30)) for _ in range(2 * depth))
    pts = tuple(
        tuple(draw(st.integers(1, 8)) for _ in range(k))
        for _ in range(draw(st.integers(min_value=2, max_value=4)))
    )
    return sign, k, hi, ops, consts, nz, ys, picks, pts


EXAMPLE = (
    1, 3, 2,
    ("neg", "sub", "concat_slice", "max"),
    (Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    (Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    (2, 2, 2, 2),
    (0, 0, 1, 0, 0, 1, 0, 1),
    ((1, 4, 8), (8, 1, 4)),
)
"""PINNED: `x`, `-x`, `x - (-x)`, a concat/slice round trip and a `max`, at
the extent this file used to fix for every spec. The unbiased search does
draw `sub` on opposite-signed operands, but not reliably at depth 4 with the
round trip present, so the shape that motivated the `sub` rule is pinned
rather than left to the draw."""

EXAMPLE_REDUCTIONS = (
    -1, 2, 2,
    ("sum_broadcast", "dot_broadcast", "neg"),
    (Fraction(1), Fraction(1), Fraction(1)),
    (Fraction(1), Fraction(1), Fraction(1)),
    (2, 2, 2),
    (0, 0, 0, 1, 0, 1),
    ((1, 4), (8, 1)),
)
"""PINNED: a NEGATIVE declaration reduced by `reduce_sum` at extent 2, then
contracted by `dot_general`, then negated.

This is the exact ℝ shape of the executed-sign-bit defect
`tests/test_executed_sign_bit_sweep.py` reports — and it PASSES here, at
every drawn point, because in ℚ the sum of two negatives is negative and no
`Fraction` carries a sign bit. Pinned for that reason: the reader who wants
to know why this search says nothing about signed zeros can run this one
example and watch it come back clean."""


def _lit(v):
    return ir.Literal(val=float(v), aval=SCALAR)


def _build(spec):
    """The IR for one spec: `assume(x > 0)` (or `< 0`), then the chain.

    Returns ``(ClosedJaxpr, eqns, {var id: producing primitive})``."""
    sign, k, hi, ops, consts, nz, ys, picks, _pts = spec
    F64, CAT, BOOLK = _avals(k)
    counter = [0]

    def nxt(aval=None):
        counter[0] += 1
        return ir.Var(id=counter[0], aval=F64 if aval is None else aval)

    def eqn(prim, ins, outs, params=()):
        return ir.JaxprEqn(primitive=prim, invars=tuple(ins),
                           outvars=tuple(outs), params=tuple(params))

    x = ir.Var(id=0, aval=F64)
    eqns = [eqn("stelling_any", (), (x,),
                (("shape", (k,)), ("dtype", "float64"),
                 ("lo", 0.0 if sign > 0 else float(-hi)),
                 ("hi", float(hi) if sign > 0 else 0.0)))]
    pa, ao = nxt(BOOLK), nxt(BOOLK)
    eqns.append(eqn("gt" if sign > 0 else "lt", (x, _lit(0)), (pa,)))
    eqns.append(eqn("stelling_assume", (pa,), (ao,)))

    pool = [x]
    produced: dict[int, str] = {}

    def pick(i):
        return pool[picks[i] % len(pool)]

    # `step`, not `k`: `k` is the drawn EXTENT and it is read inside this
    # loop (the `slice` limit, the `split` sizes, the `broadcast_in_dim`
    # shape). Shadowing it with the loop index made every `slice` a size-0
    # window, which certifies nothing — measured as `slice=0` in the census
    # while `concatenate` was 82.
    for step, op in enumerate(ops):
        a = pick(2 * step)
        b = pick(2 * step + 1)
        if op in UNARY:
            o = nxt()
            params = ()
            if op == "reshape":
                params = (("new_sizes", (k,)), ("dimensions", None))
            eqns.append(eqn(op, (a,), (o,), params))
            pool.append(o)
            produced[o.id] = op
        elif op in BINARY:
            o = nxt()
            eqns.append(eqn(op, (a, b), (o,)))
            pool.append(o)
            produced[o.id] = op
        elif op == "div_by_const":
            o = nxt()
            eqns.append(eqn("div", (a, _lit(nz[step])), (o,)))
            pool.append(o)
            produced[o.id] = "div"
        elif op == "integer_pow":
            o = nxt()
            eqns.append(eqn("integer_pow", (a,), (o,), (("y", ys[step]),)))
            pool.append(o)
            produced[o.id] = "integer_pow"
        elif op == "select_n":
            pr = nxt(BOOLK)
            eqns.append(eqn("gt", (a, _lit(consts[step])), (pr,)))
            o = nxt()
            eqns.append(eqn("select_n", (pr, a, b), (o,)))
            pool.append(o)
            produced[o.id] = "select_n"
        elif op == "concat_slice":
            c = nxt(CAT)
            eqns.append(eqn("concatenate", (a, b), (c,), (("dimension", 0),)))
            produced[c.id] = "concatenate"
            o = nxt()
            eqns.append(eqn(
                "slice", (c,), (o,),
                (("start_indices", (0,)), ("limit_indices", (k,)),
                 ("strides", None)),
            ))
            pool.append(o)
            produced[o.id] = "slice"
        elif op == "sub_opposite":
            nb = nxt()
            eqns.append(eqn("neg", (b,), (nb,)))
            produced[nb.id] = "neg"
            pool.append(nb)
            o = nxt()
            eqns.append(eqn("sub", (a, nb), (o,)))
            pool.append(o)
            produced[o.id] = "sub"
        elif op == "sum_broadcast":
            sc = nxt(SCALAR)
            eqns.append(eqn("reduce_sum", (a,), (sc,), (("axes", (0,)),)))
            produced[sc.id] = "reduce_sum"
            o = nxt()
            eqns.append(eqn("broadcast_in_dim", (sc,), (o,),
                            (("shape", (k,)), ("broadcast_dimensions", ()))))
            pool.append(o)
            produced[o.id] = "broadcast_in_dim"
        elif op == "dot_broadcast":
            sc = nxt(SCALAR)
            eqns.append(eqn(
                "dot_general", (a, b), (sc,),
                (("dimension_numbers", (((0,), (0,)), ((), ()))),
                 ("precision", None), ("preferred_element_type", None)),
            ))
            produced[sc.id] = "dot_general"
            o = nxt()
            eqns.append(eqn("broadcast_in_dim", (sc,), (o,),
                            (("shape", (k,)), ("broadcast_dimensions", ()))))
            pool.append(o)
            produced[o.id] = "broadcast_in_dim"
        else:  # concat_split
            c = nxt(CAT)
            eqns.append(eqn("concatenate", (a, b), (c,), (("dimension", 0),)))
            produced[c.id] = "concatenate"
            o1, o2 = nxt(), nxt()
            eqns.append(eqn("split", (c,), (o1, o2),
                            (("sizes", (k, k)), ("axis", 0))))
            pool.extend((o1, o2))
            produced[o1.id] = "split"
            produced[o2.id] = "split"

    out = nxt(ir.Aval(kind="ShapedArray", shape=(), dtype="bool"))
    eqns.append(eqn("stelling_assert", (ao,), (out,)))
    closed = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=(out,),
                       eqns=tuple(eqns))
    )
    return closed, eqns, produced


def _evaluate(eqns, point, k):
    """The same program in exact `Fraction`s. `var id -> list[Fraction]`,
    flat, C order. Raises on any primitive it does not implement, so the
    grammar cannot grow past the instrument in silence.

    **EXACT, AND THEREFORE BLIND TO ONE WHOLE CLASS.** `Fraction(0)` has no
    sign bit: the reduction below sums `-0` and `-0` to `0`, exactly as it
    sums `+0` and `+0`, and the propagator's certificate is judged against
    that. The target does not agree with either — see the module docstring's
    first NOT-covered entry."""
    env: dict[int, list[Fraction]] = {0: list(point)}

    def val(atom):
        if isinstance(atom, ir.Literal):
            return [Fraction(atom.val)]
        return env[atom.id]

    def pair(a, b):
        if len(a) == 1:
            a = a * len(b)
        if len(b) == 1:
            b = b * len(a)
        return a, b

    for e in eqns:
        prim = e.primitive
        if prim in ("stelling_any", "stelling_assume", "stelling_assert"):
            continue
        o = e.outvars[0].id
        if prim == "neg":
            env[o] = [-v for v in val(e.invars[0])]
        elif prim == "abs":
            env[o] = [abs(v) for v in val(e.invars[0])]
        elif prim == "square":
            env[o] = [v * v for v in val(e.invars[0])]
        elif prim in ("copy", "stop_gradient", "reshape"):
            env[o] = list(val(e.invars[0]))
        elif prim == "integer_pow":
            y = dict(e.params)["y"]
            env[o] = [v ** y for v in val(e.invars[0])]
        elif prim in ("add", "sub", "mul", "div", "max", "min", "gt", "lt"):
            a, b = pair(val(e.invars[0]), val(e.invars[1]))
            if prim == "add":
                env[o] = [p + q for p, q in zip(a, b)]
            elif prim == "sub":
                env[o] = [p - q for p, q in zip(a, b)]
            elif prim == "mul":
                env[o] = [p * q for p, q in zip(a, b)]
            elif prim == "div":
                env[o] = [p / q for p, q in zip(a, b)]
            elif prim == "max":
                env[o] = [p if p > q else q for p, q in zip(a, b)]
            elif prim == "min":
                env[o] = [p if p < q else q for p, q in zip(a, b)]
            elif prim == "gt":
                env[o] = [Fraction(int(p > q)) for p, q in zip(a, b)]
            else:
                env[o] = [Fraction(int(p < q)) for p, q in zip(a, b)]
        elif prim == "select_n":
            w = val(e.invars[0])
            cases = [val(c) for c in e.invars[1:]]
            n = len(cases[0])
            if len(w) == 1:
                w = w * n
            env[o] = [cases[int(w[i])][i] for i in range(n)]
        elif prim == "concatenate":
            flat: list[Fraction] = []
            for a in e.invars:
                flat.extend(val(a))
            env[o] = flat
        elif prim == "slice":
            pr = dict(e.params)
            env[o] = list(val(e.invars[0])[
                tuple(pr["start_indices"])[0]:tuple(pr["limit_indices"])[0]
            ])
        elif prim == "split":
            src = val(e.invars[0])
            off = 0
            for v in e.outvars:
                env[v.id] = list(src[off:off + k])
                off += k
        elif prim == "reduce_sum":
            env[o] = [sum(val(e.invars[0]), Fraction(0))]
        elif prim == "dot_general":
            a, b = val(e.invars[0]), val(e.invars[1])
            env[o] = [sum(
                (p * q for p, q in zip(a, b)), Fraction(0)
            )]
        elif prim == "broadcast_in_dim":
            src = val(e.invars[0])
            n = 1
            for d in dict(e.params)["shape"]:
                n *= d
            assert len(src) == 1, (
                "this evaluator only broadcasts a scalar; the grammar has "
                "outgrown it"
            )
            env[o] = list(src) * n
        else:
            raise AssertionError(f"no exact rule for {prim!r}")
    return env


def test_a_certified_sign_is_TRUE_at_every_assumed_point():
    """THE property.

    **THE BUDGET MOVED FROM 400 TO 800 AND THE REASON IS A MEASUREMENT, not
    a preference.** This docstring recorded, at the `ci` profile (budget 400,
    derandomized), by wrapping `_runner.Census.require` to print its report:

        drawn=401 examined=401 compared=16080
        tags={declaration 1178, neg 407, add 310, reshape 310, split 304,
              concatenate 275, copy 234, integer_pow 234, mul 210,
              select_n 207, stop_gradient 207, abs 205, div 203, max 202,
              square 174, min 172, sub 130, slice 123}

    Adding `sum_broadcast` and `dot_broadcast` to `OPS` and drawing the
    extent changes the whole derandomized stream, and at budget 400 the new
    stream gives **`slice: 4`** — the search still reaches the op, but four
    certified slices is not a search of it, and the floor caught it.

    **400 IS AN ISOLATED POTHOLE FOR `slice`, NOT A COVERAGE FLOOR, and this
    paragraph used to imply otherwise.** It closed *"not 'more is better' but
    'the ci stream needs this much to reach the rarest op in the grammar'"*,
    which reads as a threshold. There is no threshold. MEASURED, nine
    budgets, each a whole pytest run of this module with the pinned constant
    pool, rarest tag over the whole census reported:

        budget   result   rarest tag   slice
          400     FAIL          4        4      <- `slice` alone, under floor 5
          500     pass        131      215
          600     pass        162      180
          700     pass        207      411
          800     pass        156      485
          900     pass        264      272
         1000     pass        302      366
         1100     pass        390      390
         1200     pass        349      669

    Every budget from 500 up clears every floor by at least 26x, and no other
    op is anywhere near collapse at 400 — the failure is one op in one
    stream. 800 is kept because it is the budget whose census is recorded
    below tag for tag, not because anything needs it.

    MEASURED on 2026-08-28 on this branch, `ci` profile, budget 800,
    derandomized, hypothesis 6.165.10, by the same wrapping:

        drawn=802 examined=802 compared=30927
        tags={declaration 2290, neg 977, broadcast_in_dim 924, concatenate
              844, split 718, reduce_sum 573, div 561, slice 485, max 485,
              mul 427, abs 426, square 392, sub 364, dot_general 351,
              reshape 343, stop_gradient 330, add 320, min 307, copy 299,
              integer_pow 276, select_n 156}

    and no violation — which is the OTHER half of the correction: the two
    reduction rules are now drawn, in ℚ, and ℚ says they are sound. That is
    not a statement about the executed sign bit, and the module docstring's
    first NOT-covered entry is where a reader is told so.

    The floors below are TRIPWIRES for a search that collapsed, not claims
    of thoroughness — they are set an order of magnitude under every one of
    those numbers on purpose, and they are unchanged in value from before
    the widening, so this change cannot have loosened one."""
    census = _runner.Census("strict-sign/certified-is-true")

    @_profiles.current().settings(800)
    @given(_specs())
    @example(EXAMPLE)
    @example(EXAMPLE_REDUCTIONS)
    def search(spec):
        census.draw()
        sign, k, hi, _ops, _c, _nz, _ys, _picks, pts = spec
        closed, eqns, produced = _build(spec)
        p = _Propagator("constrain")
        p.run(closed.jaxpr, list(closed.consts), [])
        signs = dict(p.strict_sign)
        if not signs:
            census.skip("nothing certified")
            return
        points = [
            [Fraction(sign * n, 8) * hi for n in pt] for pt in pts
        ]
        for point in points:
            env = _evaluate(eqns, point, k)
            for vid, sgn in signs.items():
                assert vid in env, (
                    f"var {vid} was certified but this evaluator does not "
                    f"know it — the grammar has outgrown the instrument"
                )
                for cell in env[vid]:
                    ok = (cell > 0) if sgn > 0 else (cell < 0)
                    assert ok, (
                        "CERTIFIED SIGN IS FALSE: var "
                        f"{vid} (from {produced.get(vid, 'the declaration')}) "
                        f"was certified sign={sgn} and is {cell} at the "
                        f"assumed point {point}"
                    )
                    census.compare()
                census.tag(produced.get(vid, "declaration"))
        census.verdict("checked")

    search()
    census.require(
        drawn=120, examined=60, compared=400,
        add=5, sub=5, mul=5, neg=5, abs=5, square=5, max=5, min=5,
        select_n=5, concatenate=5, slice=5, split=5, div=5, integer_pow=5,
        copy=5, stop_gradient=5, reshape=5,
        # the two the NOT-covered list used to exclude, plus the round
        # trip's return leg — floored on the same terms as the rest
        reduce_sum=5, dot_general=5, broadcast_in_dim=5,
    )
