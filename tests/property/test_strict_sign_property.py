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

NOT covered:

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
* **`dot_general`, `reduce_sum` over generated shapes** — every value here
  has one fixed extent, so no contraction or reduction is drawn. Both rules
  have hand cases elsewhere.
* **ieee semantics.** The certificate is never written under ieee (the call
  site short-circuits), so there is nothing here to search.
* **anything about VERDICTS.** This property never runs
  `preconditions.check`; a rule that is sound and useless passes it.
* **rank > 1.** Every drawn value is rank 1 with a fixed extent, so a
  routing rule that is wrong only on a multi-axis operand is not reached.

The search is over PROGRAMS and BOXES both: the declared upper bound, the
operator chain, the constants, the operand picks and the sample points are
all drawn.
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


K = 3
"""The one extent every drawn value has. Rank 1, fixed, by construction —
see the module docstring's NOT-covered list."""

F64 = ir.Aval(kind="ShapedArray", shape=(K,), dtype="float64")
CAT = ir.Aval(kind="ShapedArray", shape=(2 * K,), dtype="float64")
BOOLK = ir.Aval(kind="ShapedArray", shape=(K,), dtype="bool")
SCALAR = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")

# Every op preserves the rank-1 extent, which is what lets the chain be
# drawn freely without a shape solver. `concat_slice` and `concat_split`
# are round trips: they emit the shape-changing routing equations and come
# back to `(K,)`, so `concatenate`, `slice` and `split` are all exercised.
UNARY = ("neg", "abs", "square", "copy", "stop_gradient", "reshape")
BINARY = ("add", "sub", "mul", "max", "min")
OPS = UNARY + BINARY + (
    "div_by_const", "integer_pow", "select_n", "concat_slice", "concat_split",
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
    hi = draw(st.sampled_from((1, 2, 4)))
    depth = draw(st.integers(min_value=1, max_value=6))
    ops = tuple(draw(st.sampled_from(OPS)) for _ in range(depth))
    consts = tuple(draw(st.sampled_from(CONSTS)) for _ in range(depth))
    nz = tuple(draw(st.sampled_from(NONZERO_CONSTS)) for _ in range(depth))
    ys = tuple(draw(st.sampled_from((0, 2, 3))) for _ in range(depth))
    picks = tuple(draw(st.integers(0, 30)) for _ in range(2 * depth))
    pts = tuple(
        tuple(draw(st.integers(1, 8)) for _ in range(K))
        for _ in range(draw(st.integers(min_value=2, max_value=4)))
    )
    return sign, hi, ops, consts, nz, ys, picks, pts


EXAMPLE = (
    1, 2,
    ("neg", "sub", "concat_slice", "max"),
    (Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    (Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    (2, 2, 2, 2),
    (0, 0, 1, 0, 0, 1, 0, 1),
    ((1, 4, 8), (8, 1, 4)),
)
"""PINNED: `x`, `-x`, `x - (-x)`, a concat/slice round trip and a `max`.
The unbiased search does draw `sub` on opposite-signed operands, but not
reliably at depth 4 with the round trip present, so the shape that motivated
the `sub` rule is pinned rather than left to the draw."""


def _lit(v):
    return ir.Literal(val=float(v), aval=SCALAR)


def _build(spec):
    """The IR for one spec: `assume(x > 0)` (or `< 0`), then the chain.

    Returns ``(ClosedJaxpr, eqns, {var id: producing primitive})``."""
    sign, hi, ops, consts, nz, ys, picks, _pts = spec
    counter = [0]

    def nxt(aval=F64):
        counter[0] += 1
        return ir.Var(id=counter[0], aval=aval)

    def eqn(prim, ins, outs, params=()):
        return ir.JaxprEqn(primitive=prim, invars=tuple(ins),
                           outvars=tuple(outs), params=tuple(params))

    x = ir.Var(id=0, aval=F64)
    eqns = [eqn("stelling_any", (), (x,),
                (("shape", (K,)), ("dtype", "float64"),
                 ("lo", 0.0 if sign > 0 else float(-hi)),
                 ("hi", float(hi) if sign > 0 else 0.0)))]
    pa, ao = nxt(BOOLK), nxt(BOOLK)
    eqns.append(eqn("gt" if sign > 0 else "lt", (x, _lit(0)), (pa,)))
    eqns.append(eqn("stelling_assume", (pa,), (ao,)))

    pool = [x]
    produced: dict[int, str] = {}

    def pick(i):
        return pool[picks[i] % len(pool)]

    for k, op in enumerate(ops):
        a = pick(2 * k)
        b = pick(2 * k + 1)
        if op in UNARY:
            o = nxt()
            params = ()
            if op == "reshape":
                params = (("new_sizes", (K,)), ("dimensions", None))
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
            eqns.append(eqn("div", (a, _lit(nz[k])), (o,)))
            pool.append(o)
            produced[o.id] = "div"
        elif op == "integer_pow":
            o = nxt()
            eqns.append(eqn("integer_pow", (a,), (o,), (("y", ys[k]),)))
            pool.append(o)
            produced[o.id] = "integer_pow"
        elif op == "select_n":
            pr = nxt(BOOLK)
            eqns.append(eqn("gt", (a, _lit(consts[k])), (pr,)))
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
                (("start_indices", (0,)), ("limit_indices", (K,)),
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
        else:  # concat_split
            c = nxt(CAT)
            eqns.append(eqn("concatenate", (a, b), (c,), (("dimension", 0),)))
            produced[c.id] = "concatenate"
            o1, o2 = nxt(), nxt()
            eqns.append(eqn("split", (c,), (o1, o2),
                            (("sizes", (K, K)), ("axis", 0))))
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


def _evaluate(eqns, point):
    """The same program in exact `Fraction`s. `var id -> list[Fraction]`,
    flat, C order. Raises on any primitive it does not implement, so the
    grammar cannot grow past the instrument in silence."""
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
                env[v.id] = list(src[off:off + K])
                off += K
        else:
            raise AssertionError(f"no exact rule for {prim!r}")
    return env


def test_a_certified_sign_is_TRUE_at_every_assumed_point():
    """THE property.

    MEASURED on 2026-08-28 on this branch, at the `ci` profile (budget 400,
    derandomized), by wrapping `_runner.Census.require` to print its report:

        drawn=401 examined=401 compared=16080
        tags={declaration 1178, neg 407, add 310, reshape 310, split 304,
              concatenate 275, copy 234, integer_pow 234, mul 210,
              select_n 207, stop_gradient 207, abs 205, div 203, max 202,
              square 174, min 172, sub 130, slice 123}

    and no violation. The floors below are TRIPWIRES for a search that
    collapsed, not claims of thoroughness — they are set an order of
    magnitude under every one of those numbers on purpose. The nightly
    profile also passes (68 s, same date, same tree)."""
    census = _runner.Census("strict-sign/certified-is-true")

    @_profiles.current().settings(400)
    @given(_specs())
    @example(EXAMPLE)
    def search(spec):
        census.draw()
        sign, hi, _ops, _c, _nz, _ys, _picks, pts = spec
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
            env = _evaluate(eqns, point)
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
    )
