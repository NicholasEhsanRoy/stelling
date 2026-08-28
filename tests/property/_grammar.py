# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The harness grammar the property suite generates over, and its exact oracle.

The unit of generation is a **whole harness** — a list of ``any_array``
declarations plus a list of ``assume``/``assert_`` statements over an
expression grammar — not a value. Three things follow from that choice and
each of them is load-bearing:

* a generated harness can be **rendered as Python source**, so a shrunk
  counter-example is a hand reproducer rather than a pile of integers. This is
  the single mechanism that justified the Hypothesis dependency: measured
  median 3 AST nodes over 25 seeds, machine-reduced, no hand editing;
* it can be **mutated structurally**, which is what the metamorphic properties
  need (conjoin a redundant clause, swap two statements, widen a box);
* at integer dtypes it can be **evaluated exactly**, in unbounded Python
  integers, over every point of the declared box — which is what makes the
  oracle property sharp.

**Why the exact oracle is integer-only.** ``SOUNDNESS.md`` records that real
mode judges floats in exact real arithmetic while integers are judged
execution-faithfully. At integer dtypes the two readings agree, so a VERIFIED
whose predicate is false at an enumerated point of the declared box is a
genuine defect with no ℝ-vs-IEEE confound to argue about — and the machinery
below, :func:`eval_expr_exact`, works in unbounded Python integers and has no
float reading to offer.

THIS PARAGRAPH USED TO CARRY A SECOND CLAIM AND IT IS WITHDRAWN. It read
*"...that is the declared posture, not a defect, and an oracle pointed at
floats measures the documented gap rather than a bug."* The posture is
declared, and a verdict the compiled program contradicts at a
dtype-representable point of its own declared box is a defect regardless:
``tests/property/test_float_oracle.py`` pins nine such programs, five of which
reach ``v0.1.0``. What is integer-only is THIS oracle's arithmetic, not the
question.

**Why the oracle evaluates the source text and not jax.** An execution oracle
runs the predicate through the same jax that introduced the defect this suite
was built around (an out-of-dtype-range integer literal wrapping before
tracing), so it is structurally blind to it. Evaluating the predicate *as the
user wrote it*, in exact Python integers, is strictly stronger for any defect
that lives in the translation to the jaxpr. :func:`counterexamples` never wraps
a constant; that is the whole point.

Two grammars are exported:

``integer_specs``
    One declaration, integer dtype, box small enough to enumerate, elementwise
    operations only. Sharp oracle. Used by the oracle property.

``general_specs``
    One to three declarations, floats and integers and bool, casts, reductions,
    ``where``, boolean connectives. **No oracle** — used only by the
    metamorphic properties, which relate two runs of the tool and need no
    ground truth.

The value pools are not decorative. Every entry is a shape ``SOUNDNESS.md``
records as having broken something: wide boxes, sub-ulp boxes, boxes containing
no value of their dtype, integer boxes with non-integral endpoints, int8-range
overflow, subnormals, signed zero, infinities, size-0 arrays, ``x - x``
cancellation. ``tests/property/test_generator_floor.py`` asserts that they are
actually drawn, because a strategy that silently degenerates and a property
that finds nothing look identical from the outside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from hypothesis import strategies as st

# ── dtypes ───────────────────────────────────────────────────────────────────

# (name, bits, signed) for every integer dtype stelling's scalar decoder reads.
_INT_TABLE = (
    ("int8", 8, True),
    ("uint8", 8, False),
    ("int16", 16, True),
    ("uint16", 16, False),
    ("int32", 32, True),
    ("uint32", 32, False),
    ("int64", 64, True),
    ("uint64", 64, False),
)
INT_DTYPES = tuple(n for n, _, _ in _INT_TABLE)
FLOAT_DTYPES = ("float64", "float32", "float16", "bfloat16")
ALL_DTYPES = INT_DTYPES + FLOAT_DTYPES + ("bool",)

_BITS = {n: (b, s) for n, b, s in _INT_TABLE}


def dtype_range(name: str) -> tuple[int, int]:
    """The closed range of values representable in an integer dtype."""
    bits, signed = _BITS[name]
    if signed:
        return (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return (0, 2**bits - 1)


def in_dtype_range(name: str, value) -> bool:
    """Is ``value`` representable in ``name`` without wrapping?

    Non-integer dtypes answer ``True``: floats do not wrap, they overflow to
    ``±inf``, which is a different (and disclosed) phenomenon.
    """
    if name not in _BITS:
        return True
    lo, hi = dtype_range(name)
    return isinstance(value, int) and lo <= value <= hi


# ── value pools ──────────────────────────────────────────────────────────────

FLOAT_POOL = (
    0.0, 1.0, -1.0, -0.0, 2.0, 0.5, 10.0, -2.0,
    1.0 + 2.0**-52,            # one ulp above 1.0 in binary64
    2.0**-24,                  # below the f32 ulp of 1.0
    2.0**-53,
    9007199254740992.0,        # 2**53
    9007199254740994.0,        # 2**53 + 2
    1e-8, 1e8, 1e300, -1e300,
    2.2250738585072014e-308,   # smallest binary64 normal
    5e-324,                    # smallest binary64 subnormal
    1e-320,                    # a subnormal with room either side
    1.7976931348623157e308,    # binary64 max
    65504.0,                   # float16 max
    3.4028234663852886e38,     # float32 max
    6.103515625e-05,           # float16 min normal
    3.141592653589793,
    math.inf, -math.inf,
)

INT_POOL = (
    0, 1, -1, 2, 7, -7, 10,
    127, 128, -128, -129, 255, 256,
    32767, 65535,
    2**31 - 1, 2**31, -(2**31),
    2**53, 2**53 + 1, -(2**53) - 1,
    2**63 - 1, -(2**63), 2**64 - 1,
    10**23,
)

# Integer dtypes declared with non-integral endpoints: a shape the declaration
# guard has to have an opinion about, and where route agreement has broken.
NON_INTEGRAL = (0.5, -0.5, 1.5, 127.5, -128.5, 2.5)

SHAPES = ((), (1,), (2,), (0,), (3,), (2, 2), (0, 3))

CMP_OPS = ("<=", ">=", "<", ">", "==", "!=")
UNARY_ANY = ("neg", "abs", "sign", "square", "copy")
UNARY_FLOAT = ("sqrt", "exp")
BINARY_ANY = ("add", "sub", "mul", "max", "min")
BINARY_FLOAT = ("div",)


# ── the harness IR ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Decl:
    name: str
    shape: tuple
    dtype: str
    lo: object
    hi: object


@dataclass(frozen=True)
class Stmt:
    kind: str  # "assert" | "assume"
    pred: tuple


@dataclass(frozen=True)
class Spec:
    decls: tuple
    stmts: tuple

    def render(self) -> str:
        """The harness as runnable Python. This is what a reader acts on."""
        lines = ["def harness():"]
        for d in self.decls:
            lines.append(
                f"    {d.name} = any_array({d.shape!r}, {d.dtype!r}, "
                f"({_lit(d.lo)}, {_lit(d.hi)}))"
            )
        outs = []
        for i, s in enumerate(self.stmts):
            expr = render_pred(s.pred)
            if s.kind == "assume":
                lines.append(f"    assume({expr})")
            else:
                lines.append(f"    o{i} = assert_({expr})")
                outs.append(f"o{i}")
        lines.append("    return " + (", ".join(outs) if outs else "()"))
        return "\n".join(lines)

    @property
    def asserts(self) -> tuple:
        return tuple(s for s in self.stmts if s.kind == "assert")

    @property
    def n_asserts(self) -> int:
        return len(self.asserts)


def _lit(v) -> str:
    if isinstance(v, float):
        if v == math.inf:
            return "math.inf"
        if v == -math.inf:
            return "-math.inf"
        if v != v:
            return "math.nan"
    return repr(v)


# Expr nodes:  ("var", name) ("const", v) ("un", op, e) ("bin", op, a, b)
#              ("cast", dtype, e) ("sum", e) ("where", pred, a, b)
#              ("cancel", e)  -- renders as (x - x)
# Pred nodes:  ("cmp", op, a, b) ("and", p, q) ("or", p, q) ("not", p)


def render_expr(e) -> str:
    tag = e[0]
    if tag == "var":
        return e[1]
    if tag == "const":
        return _lit(e[1])
    if tag == "un":
        x = render_expr(e[2])
        return {
            "neg": f"(-{x})", "abs": f"jnp.abs({x})", "sign": f"jnp.sign({x})",
            "square": f"jnp.square({x})", "copy": f"({x} + 0)",
            "sqrt": f"jnp.sqrt({x})", "exp": f"jnp.exp({x})",
        }[e[1]]
    if tag == "bin":
        a, b = render_expr(e[2]), render_expr(e[3])
        return {
            "add": f"({a} + {b})", "sub": f"({a} - {b})", "mul": f"({a} * {b})",
            "div": f"({a} / {b})",
            "max": f"jnp.maximum({a}, {b})", "min": f"jnp.minimum({a}, {b})",
        }[e[1]]
    if tag == "pow":
        return f"({render_expr(e[1])} ** {e[2]})"
    if tag == "cast":
        return f"jnp.asarray({render_expr(e[2])}).astype(jnp.{e[1]})"
    if tag == "sum":
        return f"jnp.sum({render_expr(e[1])})"
    if tag == "where":
        return (f"jnp.where({render_pred(e[1])}, {render_expr(e[2])}, "
                f"{render_expr(e[3])})")
    if tag == "cancel":
        x = render_expr(e[1])
        return f"({x} - {x})"
    if tag == "at_add":
        # The scatter-add form. Present because the cross-series property's
        # positive control needs it: `_is_add_combiner` read the combiner by
        # jaxpr CONTAINER CLASS, and jax 0.11 merged `ClosedJaxpr` into
        # `Jaxpr` while 0.10.2 did not, so this one node was VERIFIED on one
        # series and UNKNOWN on the other with `TESTED_JAX_SERIES` claiming
        # both.
        return f"{render_expr(e[1])}.at[{e[2]}].add({_lit(e[3])})"
    raise AssertionError(tag)


def render_pred(p) -> str:
    if p[0] == "cmp":
        return f"({render_expr(p[2])} {p[1]} {render_expr(p[3])})"
    if p[0] == "and":
        return f"({render_pred(p[1])} & {render_pred(p[2])})"
    if p[0] == "or":
        return f"({render_pred(p[1])} | {render_pred(p[2])})"
    if p[0] == "not":
        return f"(~{render_pred(p[1])})"
    if p[0] == "all":
        return f"jnp.all({render_pred(p[1])})"
    raise AssertionError(p[0])


# ── building the callable stelling.preconditions.check wants ─────────────────


def eval_expr(e, env, jnp):
    tag = e[0]
    if tag == "var":
        return env[e[1]]
    if tag == "const":
        return e[1]
    if tag == "un":
        x = eval_expr(e[2], env, jnp)
        return {
            "neg": lambda: -x, "abs": lambda: jnp.abs(x),
            "sign": lambda: jnp.sign(x), "square": lambda: jnp.square(x),
            "copy": lambda: x + 0, "sqrt": lambda: jnp.sqrt(x),
            "exp": lambda: jnp.exp(x),
        }[e[1]]()
    if tag == "bin":
        a = eval_expr(e[2], env, jnp)
        b = eval_expr(e[3], env, jnp)
        return {
            "add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
            "div": lambda: a / b,
            "max": lambda: jnp.maximum(a, b), "min": lambda: jnp.minimum(a, b),
        }[e[1]]()
    if tag == "pow":
        return eval_expr(e[1], env, jnp) ** e[2]
    if tag == "cast":
        return jnp.asarray(eval_expr(e[2], env, jnp)).astype(getattr(jnp, e[1]))
    if tag == "sum":
        return jnp.sum(eval_expr(e[1], env, jnp))
    if tag == "where":
        return jnp.where(eval_pred(e[1], env, jnp),
                         eval_expr(e[2], env, jnp), eval_expr(e[3], env, jnp))
    if tag == "cancel":
        x = eval_expr(e[1], env, jnp)
        return x - x
    if tag == "at_add":
        return eval_expr(e[1], env, jnp).at[e[2]].add(e[3])
    raise AssertionError(tag)


_CMP_FN = {
    "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b, ">": lambda a, b: a > b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


def eval_pred(p, env, jnp):
    if p[0] == "cmp":
        return _CMP_FN[p[1]](eval_expr(p[2], env, jnp), eval_expr(p[3], env, jnp))
    if p[0] == "and":
        return eval_pred(p[1], env, jnp) & eval_pred(p[2], env, jnp)
    if p[0] == "or":
        return eval_pred(p[1], env, jnp) | eval_pred(p[2], env, jnp)
    if p[0] == "not":
        return jnp.logical_not(eval_pred(p[1], env, jnp))
    if p[0] == "all":
        # `jnp.all` over a predicate is the form that makes an assume DROPPED
        # rather than constraining, which is the shape the affine
        # vacuous-refutation defect lived in. It is in the grammar for that.
        return jnp.all(eval_pred(p[1], env, jnp))
    raise AssertionError(p[0])


def build(spec: Spec):
    """Turn a :class:`Spec` into the zero-argument harness ``check()`` expects."""
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_, assume

    def harness():
        env = {}
        for d in spec.decls:
            env[d.name] = any_array(d.shape, d.dtype, (d.lo, d.hi))
        outs = []
        for s in spec.stmts:
            pred = eval_pred(s.pred, env, jnp)
            if s.kind == "assume":
                assume(pred)
            else:
                outs.append(assert_(pred))
        return tuple(outs)

    return harness


# ── the exact oracle (integer dtypes only) ───────────────────────────────────

_EXACT_UNARY = {
    "neg": lambda v: -v,
    "abs": abs,
    "sign": lambda v: (v > 0) - (v < 0),
    "square": lambda v: v * v,
    "copy": lambda v: v,
}
_EXACT_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": max,
    "min": min,
}


def exact_supported(spec: Spec) -> bool:
    """Can every point of this harness be decided in exact Python integers?

    Requires: every declaration an integer dtype with integral, in-range,
    non-empty bounds; every operation elementwise and exactly modelled. The
    elementwise restriction is what lets a *scalar* counterexample stand for a
    whole shape-``(n,)`` obligation: the point is the constant vector.
    """
    for d in spec.decls:
        if d.dtype not in INT_DTYPES:
            return False
        if not (isinstance(d.lo, int) and isinstance(d.hi, int)):
            return False
        if d.lo > d.hi:
            return False
        if not (in_dtype_range(d.dtype, d.lo) and in_dtype_range(d.dtype, d.hi)):
            return False
    return all(_exact_pred_ok(s.pred) for s in spec.stmts)


def _exact_expr_ok(e) -> bool:
    tag = e[0]
    if tag in ("var", "const"):
        return True
    if tag == "un":
        return e[1] in _EXACT_UNARY and _exact_expr_ok(e[2])
    if tag == "bin":
        return e[1] in _EXACT_BINARY and _exact_expr_ok(e[2]) and _exact_expr_ok(e[3])
    if tag == "pow":
        return _exact_expr_ok(e[1])
    return False


def _exact_pred_ok(p) -> bool:
    if p[0] == "cmp":
        return _exact_expr_ok(p[2]) and _exact_expr_ok(p[3])
    if p[0] in ("and", "or"):
        return _exact_pred_ok(p[1]) and _exact_pred_ok(p[2])
    if p[0] == "not":
        return _exact_pred_ok(p[1])
    return False


def eval_expr_exact(e, point):
    """The value of the expression AS WRITTEN, in unbounded Python integers.

    This never wraps a constant. That is the whole point: the defect class this
    suite was built around destroys an out-of-dtype-range literal *before* any
    stelling primitive binds it, so an oracle that goes through jax cannot see
    it.
    """
    tag = e[0]
    if tag == "var":
        return point[e[1]]
    if tag == "const":
        return e[1]
    if tag == "un":
        return _EXACT_UNARY[e[1]](eval_expr_exact(e[2], point))
    if tag == "bin":
        return _EXACT_BINARY[e[1]](
            eval_expr_exact(e[2], point), eval_expr_exact(e[3], point)
        )
    if tag == "pow":
        return eval_expr_exact(e[1], point) ** e[2]
    raise AssertionError(tag)


def eval_pred_exact(p, point) -> bool:
    if p[0] == "cmp":
        return bool(
            _CMP_FN[p[1]](eval_expr_exact(p[2], point), eval_expr_exact(p[3], point))
        )
    if p[0] == "and":
        return eval_pred_exact(p[1], point) and eval_pred_exact(p[2], point)
    if p[0] == "or":
        return eval_pred_exact(p[1], point) or eval_pred_exact(p[2], point)
    if p[0] == "not":
        return not eval_pred_exact(p[1], point)
    raise AssertionError(p[0])


ENUMERATION_CAP = 4096


def declared_points(spec: Spec, cap: int = ENUMERATION_CAP):
    """Every point of the declared box, or ``None`` if there are too many.

    ``None`` is a refusal to answer, not an empty answer — the callers treat it
    as "this example proves nothing" rather than "no counterexamples".
    """
    import itertools

    ranges = []
    total = 1
    for d in spec.decls:
        n = d.hi - d.lo + 1
        total *= n
        if total > cap:
            return None
        ranges.append(range(d.lo, d.hi + 1))
    return [
        dict(zip([d.name for d in spec.decls], combo))
        for combo in itertools.product(*ranges)
    ]


def admitted_points(spec: Spec, cap: int = ENUMERATION_CAP):
    """Declared points the harness's own ``assume``s admit, forward-scoped.

    Forward scoping is stelling's deliberate posture: an ``assume`` constrains
    the obligations written *after* it, not before. So "admitted" is per
    obligation, and this returns the points admitted for the *last* obligation
    — the strongest precondition any obligation in the harness sees. Callers
    that need per-obligation admission use :func:`admitted_for_obligation`.
    """
    pts = declared_points(spec, cap)
    if pts is None:
        return None
    assumes = [s.pred for s in spec.stmts if s.kind == "assume"]
    return [p for p in pts if all(eval_pred_exact(a, p) for a in assumes)]


def admitted_for_obligation(spec: Spec, index: int, points):
    """Points admitted for the ``index``-th ``assert_``, forward-scoped."""
    seen = 0
    active = []
    for s in spec.stmts:
        if s.kind == "assume":
            active.append(s.pred)
            continue
        if seen == index:
            break
        seen += 1
    return [p for p in points if all(eval_pred_exact(a, p) for a in active)]


def counterexamples(spec: Spec, limit: int = 8):
    """Admitted points at which some obligation, AS WRITTEN, is false.

    Returns ``(obligation_index, point)`` pairs, or ``None`` when the box is
    too large to enumerate. One-sided by construction: finding a point refutes
    a VERIFIED; finding none confirms nothing.
    """
    pts = declared_points(spec)
    if pts is None:
        return None
    out = []
    seen = 0
    for s in spec.stmts:
        if s.kind != "assert":
            continue
        admitted = admitted_for_obligation(spec, seen, pts)
        for p in admitted:
            if not eval_pred_exact(s.pred, p):
                out.append((seen, p))
                if len(out) >= limit:
                    return out
        seen += 1
    return out


def simple_admitted_region_is_empty(spec: Spec, upto: int):
    """Is the admitted region for assert #``upto`` EXACTLY empty? Or unknown.

    Returns ``True``, ``False``, or ``None`` — and ``None`` is a refusal to
    answer, never "no".

    Deliberately narrow, and narrow in the direction that keeps it exact. It
    answers only for harnesses where every declaration is a **float** dtype
    with finite, non-NaN bounds and at least one element, and where every
    ``assume`` in scope is a conjunction (optionally under ``jnp.all``) of
    atoms comparing one variable to one constant. On that fragment the admitted
    region is an intersection of half-spaces with a box, so emptiness is a
    comparison of two real numbers and nothing is approximated.

    Why the float fragment specifically: this is the shape of the affine defect
    this project shipped — ``assume(jnp.all(x >= 2.0))`` over a box of
    ``(-1.0, 1.0)``, where the refinement judged over the DECLARED box rather
    than the assumed region and re-minted a violation the interval leg had
    withheld. The integer grammar cannot reach it, because the affine
    refinement supports float declarations only.
    """
    box = {}
    for d in spec.decls:
        if d.dtype not in FLOAT_DTYPES:
            return None
        if n_elements(d.shape) == 0:
            return None
        try:
            lo, hi = float(d.lo), float(d.hi)
        except (TypeError, ValueError):
            return None
        if lo != lo or hi != hi or not (math.isfinite(lo) and math.isfinite(hi)):
            return None
        if lo > hi:
            return None
        box[d.name] = (lo, hi, True, True)  # lo, hi, lo_closed, hi_closed

    seen = 0
    for s in spec.stmts:
        if s.kind == "assert":
            if seen == upto:
                break
            seen += 1
            continue
        atoms = _conjuncts(s.pred)
        if atoms is None:
            return None
        for op, name, const, var_on_left in atoms:
            if name not in box:
                return None
            lo, hi, lc, hc = box[name]
            try:
                c = float(const)
            except (TypeError, ValueError):
                return None
            if c != c:
                return None
            o = op if var_on_left else _FLIP[op]
            if o in (">=", ">"):
                if c > lo or (c == lo and o == ">"):
                    lo, lc = c, o == ">="
            elif o in ("<=", "<"):
                if c < hi or (c == hi and o == "<"):
                    hi, hc = c, o == "<="
            elif o == "==":
                lo = max(lo, c)
                hi = min(hi, c)
                lc = hc = True
            else:  # "!=" cannot be expressed as a box, and is not guessed at
                return None
            box[name] = (lo, hi, lc, hc)

    for lo, hi, lc, hc in box.values():
        if lo > hi or (lo == hi and not (lc and hc)):
            return True
    return False


_FLIP = {"<=": ">=", ">=": "<=", "<": ">", ">": "<", "==": "==", "!=": "!="}


def _conjuncts(p):
    """Flatten an ``and``-tree (optionally under ``all``) into simple atoms."""
    if p[0] == "all":
        return _conjuncts(p[1])
    if p[0] == "and":
        a, b = _conjuncts(p[1]), _conjuncts(p[2])
        return None if a is None or b is None else a + b
    if p[0] != "cmp":
        return None
    _, op, left, right = p
    if left[0] == "var" and right[0] == "const":
        return [(op, left[1], right[1], True)]
    if left[0] == "const" and right[0] == "var":
        return [(op, right[1], left[1], False)]
    return None


def any_obligation_is_admitted(spec: Spec) -> bool:
    """Does at least one obligation see a non-empty admitted set?

    ``∀ x ∈ ∅`` is vacuously true and the tool is entitled to VERIFIED, so a
    harness whose preconditions admit nothing is not evidence of anything.
    """
    pts = declared_points(spec)
    if pts is None:
        return False
    seen = 0
    for s in spec.stmts:
        if s.kind != "assert":
            continue
        if admitted_for_obligation(spec, seen, pts):
            return True
        seen += 1
    return False


# ── the wrap fingerprint ─────────────────────────────────────────────────────
#
# The unit is the maximal CONSTANT SUBEXPRESSION, not the literal token.
# Python folds `-abs(1)` to `-1` and `-129` to a single constant before jax
# sees anything, so a harness whose only literal token is the in-range `1` can
# still hand jax an out-of-dtype-range constant. Measured in the second spike:
# a mask written on literal tokens alone left the defect in and the "masked"
# property still failed at every budget.


def _is_constant(e) -> bool:
    if e[0] == "var":
        return False
    if e[0] == "const":
        return True
    return all(_is_constant(c) for c in e[1:] if isinstance(c, tuple))


_FOLD_ANY_UNARY = {
    "neg": lambda v: -v,
    "abs": abs,
    "sign": lambda v: (v > 0) - (v < 0),
    "square": lambda v: v * v,
    "copy": lambda v: v,
    "sqrt": lambda v: math.sqrt(v),
    "exp": lambda v: math.exp(v),
}
_FOLD_ANY_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": max,
    "min": min,
    "div": lambda a, b: a / b,
}


def _fold(e):
    """The value of a CONSTANT subexpression, as Python computes it.

    Wider than :func:`eval_expr_exact`, and it has to be: the general grammar
    carries ``div``/``sqrt``/``exp``/``cast``/``sum``/``cancel``/``where``, and
    a folder that raised on those made :func:`wrappable_constants` — which is
    the MASK the oracle properties use — raise instead of answering.

    That was a defect in this harness, not in the tool, and the ``dev`` profile
    found it: 250 derandomised examples never built ``x <= (0.0 / 0.0)`` beside
    an ``int8`` declaration, and 1000 randomised ones did, three runs out of
    three. Recorded because it is the same lesson the properties are for — a
    green run at one budget says nothing about another.

    ``cast`` to an integer dtype is folded WITH its wrap, because that is a
    genuine second face of the wrap class and the mask must be able to see it.
    """
    tag = e[0]
    if tag == "const":
        return e[1]
    if tag == "un":
        return _FOLD_ANY_UNARY[e[1]](_fold(e[2]))
    if tag == "bin":
        return _FOLD_ANY_BINARY[e[1]](_fold(e[2]), _fold(e[3]))
    if tag == "pow":
        return _fold(e[1]) ** e[2]
    if tag == "cancel":
        v = _fold(e[1])
        return v - v
    if tag == "sum":
        return _fold(e[1])
    if tag == "cast":
        v = _fold(e[1])
        if e[1] in INT_DTYPES:
            bits, _ = _BITS[e[1]]
            lo, _hi = dtype_range(e[1])
            return lo + ((int(v) - lo) % (2**bits))
        return float(v)
    if tag == "where":
        return _fold(e[2])
    raise AssertionError(tag)


def folded_constants(spec: Spec):
    """Every maximal constant subexpression's value, folded as Python folds it."""
    out = []

    def walk(e):
        if _is_constant(e):
            try:
                out.append(_fold(e))
            except Exception:  # noqa: BLE001
                # A constant subexpression this folder cannot evaluate is one
                # it cannot classify. The catch is broad DELIBERATELY and the
                # cost is stated: an unclassifiable constant is not masked, so
                # the oracle properties see the harness rather than skip it.
                # That direction is the safe one — it can produce a report to
                # read, never a silent skip — and it is the direction a mask
                # must fail in.
                pass
            return
        for c in e[1:]:
            if isinstance(c, tuple):
                walk(c)

    def walk_pred(p):
        if p[0] == "cmp":
            walk(p[2])
            walk(p[3])
        elif p[0] in ("and", "or"):
            walk_pred(p[1])
            walk_pred(p[2])
        elif p[0] == "not":
            walk_pred(p[1])

    for s in spec.stmts:
        walk_pred(s.pred)
    return out


def wrappable_constants(spec: Spec):
    """Folded constants outside the range of some integer declaration's dtype.

    These are the constants jax silently reduces mod 2**bits before stelling
    ever sees them. A harness carrying one is in the known open defect class
    (HANDOFF6 §71/§74: the arithmetic-entry integer literal wrap); a harness
    carrying none is not, and a wrong VERIFIED from one of *those* is a new
    finding rather than a rediscovery.
    """
    dtypes = {d.dtype for d in spec.decls if d.dtype in INT_DTYPES}
    if not dtypes:
        return []
    return [
        v
        for v in folded_constants(spec)
        if isinstance(v, int)
        and any(not in_dtype_range(dt, v) for dt in dtypes)
    ]


# ── structural mutations used by the metamorphic properties ──────────────────


def box_implied_pred(decl: Decl):
    """A predicate true of every member of ``decl``'s declared box.

    Deliberately box-shaped (``x >= lo``): the interval domain already knows
    it, so it can add no information, so any *gain* in proving power from
    conjoining it is suspicious rather than expected.
    """
    return ("cmp", ">=", ("var", decl.name), ("const", decl.lo))


def redundant_conjunct_is_sound_for(decl: Decl) -> bool:
    """Is ``x >= lo`` genuinely redundant for this declaration?

    Three ways it is not, and each one would make the property cry wolf:

    * ``lo`` is NaN — nothing compares true against it;
    * the declaration is ``bool`` — the comparison is a different sort;
    * ``lo`` is an integer outside the declared integer dtype's own range, so
      jax reduces it mod ``2**bits`` and the conjunct the tool sees is *not*
      the one written. That is the open wrap defect, and conflating it with a
      redundancy violation would report the wrong thing.
    """
    if decl.dtype == "bool":
        return False
    if isinstance(decl.lo, float) and decl.lo != decl.lo:
        return False
    if decl.dtype in INT_DTYPES and not in_dtype_range(decl.dtype, decl.lo):
        return False
    return True


def with_redundant_conjunct(spec: Spec, stmt_index: int, decl: Decl) -> Spec:
    s = spec.stmts[stmt_index]
    stmts = list(spec.stmts)
    stmts[stmt_index] = Stmt(s.kind, ("and", s.pred, box_implied_pred(decl)))
    return replace(spec, stmts=tuple(stmts))


def with_redundant_assume(spec: Spec, at: int, decl: Decl) -> Spec:
    stmts = list(spec.stmts)
    stmts.insert(at, Stmt("assume", box_implied_pred(decl)))
    return replace(spec, stmts=tuple(stmts))


def swapped(spec: Spec, i: int) -> Spec:
    stmts = list(spec.stmts)
    stmts[i], stmts[i + 1] = stmts[i + 1], stmts[i]
    return replace(spec, stmts=tuple(stmts))


def widened(spec: Spec, decl_index: int, factor: float = 2.0):
    """Widen one declaration's box, or ``None`` if it cannot be widened.

    Integer boxes are widened **within the dtype's own range**: a widening that
    put the bound outside the dtype would be refused at the declaration door,
    or worse, would introduce the wrap the metamorphic property is not about.
    """
    d = spec.decls[decl_index]
    if d.dtype == "bool":
        return None
    if d.dtype in FLOAT_DTYPES:
        try:
            lo, hi = float(d.lo), float(d.hi)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            return None
        span = max(abs(hi - lo), 1.0)
        nlo, nhi = lo - span * factor, hi + span * factor
        if not (math.isfinite(nlo) and math.isfinite(nhi)):
            return None
    else:
        if not (isinstance(d.lo, int) and isinstance(d.hi, int)):
            return None
        if d.lo > d.hi:
            return None
        dlo, dhi = dtype_range(d.dtype)
        span = max(abs(d.hi - d.lo), 1)
        # CLAMPED TO THE DTYPE, AND THEN CLAMPED AGAIN TO THE ORIGINAL BOX.
        # The second clamp is not belt and braces. A declaration whose bound is
        # already OUTSIDE its own dtype's range — `any_array((), "uint8",
        # (-1, 0))`, which this grammar draws and stelling accepts — had its
        # `lo` moved UP by the dtype clamp, so the "widened" box was a strict
        # SUBSET of the original and the monotonicity property was comparing a
        # narrowing against a narrowing. It reported `UNKNOWN -> VERIFIED` and
        # was right to: `x0 <= x1` over `x1 in [-1, 0]` is undecided and over
        # `x1 in [0, 255]` it is discharged. That was a defect in this
        # function, found by the randomised profile at 1000 examples where 250
        # derandomised ones never built it.
        nlo = min(d.lo, max(dlo, d.lo - span))
        nhi = max(d.hi, min(dhi, d.hi + span))
        if (nlo, nhi) == (d.lo, d.hi):
            return None
    decls = list(spec.decls)
    decls[decl_index] = replace(d, lo=nlo, hi=nhi)
    # The property's whole premise, asserted rather than assumed: the new box
    # must CONTAIN the old one. A "widening" that does not is a different
    # mutation and any report from it is a false one.
    if not (nlo <= d.lo and nhi >= d.hi):
        return None
    return replace(spec, decls=tuple(decls))


# ── strategies ───────────────────────────────────────────────────────────────


def _int_literal_pool(dtype: str):
    """Literals worth trying: in-range, just-out-of-range, and far out.

    The out-of-range half is the class ``hypothesis.extra.numpy.from_dtype``
    is structurally incapable of producing — measured, 0 of 3000 draws — and it
    is exactly the class this project's open wrap defect lives in. That is why
    the pools are hand-written and why the generator floor asserts they fire.
    """
    bits, _ = _BITS[dtype]
    lo, hi = dtype_range(dtype)
    m = 2**bits
    pool = [0, 1, -1, 2, 3, -2, 10, -10, 100, -100]
    for base in (m, 2 * m, m // 2):
        pool += [base, -base, base + 1, base - 1, -base + 1, -base - 1]
    pool += [lo, hi, lo - 1, hi + 1, lo - m, hi + m]
    return sorted(set(pool))


def int_literals(dtype: str):
    return st.one_of(
        st.sampled_from(_int_literal_pool(dtype)),
        st.integers(min_value=-8, max_value=8),
        st.integers(min_value=-(2**34), max_value=2**34),
    )


def _int_expressions(dtype: str, max_leaves: int = 8):
    lit = int_literals(dtype)
    base = st.one_of(st.just(("var", "x0")), lit.map(lambda k: ("const", k)))

    def extend(children):
        return st.one_of(
            st.tuples(st.sampled_from(("add", "sub", "mul", "max", "min")),
                      children, children).map(lambda t: ("bin",) + t),
            children.map(lambda e: ("un", "neg", e)),
            children.map(lambda e: ("un", "abs", e)),
            children.map(lambda e: ("un", "square", e)),
            st.tuples(children, st.integers(1, 3)).map(
                lambda t: ("pow", t[0], t[1])),
        )

    return st.recursive(base, extend, max_leaves=max_leaves)


def _int_predicates(dtype: str, max_leaves: int = 6):
    e = _int_expressions(dtype, max_leaves=max_leaves)
    return st.tuples(st.sampled_from(CMP_OPS), e, e).map(
        lambda t: ("cmp",) + t
    )


@st.composite
def integer_specs(draw, *, allow_assume=True, max_box=48, max_leaves=6,
                  shapes=((), (2,)), dtypes=INT_DTYPES):
    """The sharp-oracle grammar: one integer declaration, elementwise ops.

    The box is drawn **inside the dtype's own range** and small enough to
    enumerate exactly, so :func:`counterexamples` is total on what this
    strategy produces. Out-of-range values appear as *literals in the
    expression*, which is where they wrap.
    """
    dtype = draw(st.sampled_from(dtypes))
    dlo, dhi = dtype_range(dtype)
    lo = draw(st.integers(min_value=max(dlo, -64), max_value=min(dhi, 64)))
    hi = draw(st.integers(min_value=lo, max_value=min(dhi, lo + max_box)))
    shape = draw(st.sampled_from(shapes))
    decls = (Decl("x0", shape, dtype, lo, hi),)
    stmts = []
    if allow_assume and draw(st.booleans()):
        stmts.append(Stmt("assume", draw(_int_predicates(dtype, max_leaves=4))))
    stmts.append(Stmt("assert", draw(_int_predicates(dtype, max_leaves=max_leaves))))
    return Spec(decls, tuple(stmts))


@st.composite
def integer_program_specs(draw, *, max_stmts=3, max_box=32, max_leaves=4,
                          shapes=((), (2,)), dtypes=INT_DTYPES):
    """Several statements over the sharp-integer grammar.

    The one-declaration :func:`integer_specs` has at most one ``assume`` and one
    ``assert_``, which is all an oracle needs and not enough to reorder. This
    draws two to ``max_stmts`` statements so that the reordering and
    redundant-conjunct properties keep the exact machinery — in particular
    :func:`any_obligation_is_admitted`, which is what lets those properties
    exclude the vacuous case *exactly* rather than by guessing.
    """
    dtype = draw(st.sampled_from(dtypes))
    dlo, dhi = dtype_range(dtype)
    lo = draw(st.integers(min_value=max(dlo, -32), max_value=min(dhi, 32)))
    hi = draw(st.integers(min_value=lo, max_value=min(dhi, lo + max_box)))
    shape = draw(st.sampled_from(shapes))
    decls = (Decl("x0", shape, dtype, lo, hi),)
    n = draw(st.integers(2, max_stmts))
    kinds = [draw(st.sampled_from(("assert", "assert", "assume"))) for _ in range(n)]
    if "assert" not in kinds:
        kinds[-1] = "assert"
    stmts = tuple(
        Stmt(k, draw(_int_predicates(dtype, max_leaves=max_leaves))) for k in kinds
    )
    return Spec(decls, stmts)


def static_shape(e, decls):
    """The shape an expression will have, or ``None`` if it cannot be built.

    Needed by the vacuous-conjunct property, which has to know *before running
    anything* whether conjoining a predicate collapses the whole conjunction to
    zero elements. Broadcasting rules only; a ``None`` means jax would refuse
    the harness, and the example is discarded rather than reported.
    """
    shapes = {d.name: d.shape for d in decls}
    tag = e[0]
    if tag == "var":
        return shapes[e[1]]
    if tag == "const":
        return ()
    if tag in ("un", "cast"):
        return static_shape(e[2], decls)
    if tag in ("cancel", "copy"):
        return static_shape(e[1], decls)
    if tag == "sum":
        return () if static_shape(e[1], decls) is not None else None
    if tag in ("pow", "at_add"):
        return static_shape(e[1], decls)
    if tag == "bin":
        return _broadcast(static_shape(e[2], decls), static_shape(e[3], decls))
    if tag == "where":
        a = static_shape_pred(e[1], decls)
        b = _broadcast(static_shape(e[2], decls), static_shape(e[3], decls))
        return _broadcast(a, b)
    return None


def static_shape_pred(p, decls):
    if p[0] == "cmp":
        return _broadcast(static_shape(p[2], decls), static_shape(p[3], decls))
    if p[0] in ("and", "or"):
        return _broadcast(static_shape_pred(p[1], decls),
                          static_shape_pred(p[2], decls))
    if p[0] == "not":
        return static_shape_pred(p[1], decls)
    if p[0] == "all":
        return () if static_shape_pred(p[1], decls) is not None else None
    return None


def _broadcast(a, b):
    if a is None or b is None:
        return None
    out = []
    for x, y in zip(reversed(a or ()), reversed(b or ())):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            return None
    longer = a if len(a) >= len(b) else b
    out += list(reversed(longer[: len(longer) - len(out)]))
    return tuple(reversed(out))


def n_elements(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def wrap_biased_integer_specs(**kw):
    """:func:`integer_specs` restricted to harnesses jax can wrap a constant in.

    **Why a biased generator is the right instrument here, and where the bias
    must NOT go.** The unbiased grammar reaches the wrap class, but thinly:
    measured, 5 of 25 randomly-seeded 200-example runs found a wrong VERIFIED,
    which is far too thin for a deterministic per-push gate that is supposed to
    fail *every* time until the defect is fixed. Restricting the draw to
    harnesses that carry an out-of-dtype-range constant is generator design
    aimed at a known open defect, and every measurement this project has made
    says generator design dominates search budget.

    The bias is confined to the leg that is ``xfail``-marked for that defect.
    The residual leg — the one that must stay green, and the one that would
    catch a *new* wrong VERIFIED — runs the unbiased grammar with this class
    masked out. Biasing that one would be marking one's own homework.
    """
    return integer_specs(**kw).filter(lambda s: bool(wrappable_constants(s)))


def _bounds(dtype: str):
    if dtype == "bool":
        return st.sampled_from([(False, True), (False, False), (True, True)])
    if dtype in FLOAT_DTYPES:
        pool = st.sampled_from(FLOAT_POOL)
    else:
        pool = st.one_of(
            st.sampled_from(INT_POOL),
            st.sampled_from(NON_INTEGRAL),
            st.sampled_from(tuple(dtype_range(dtype))),
        )
    return st.tuples(pool, pool).map(_order)


def _order(pair):
    lo, hi = pair
    try:
        if lo > hi:
            return (hi, lo)
    except TypeError:  # pragma: no cover - defensive
        pass
    return (lo, hi)


@st.composite
def general_specs(draw, *, max_decls=3, max_stmts=4, dtypes=ALL_DTYPES):
    """The wide grammar: floats, casts, reductions, ``where``, connectives.

    **No EXACT oracle.** This grammar exists for the metamorphic properties,
    which relate two runs of the tool to each other and need no ground truth,
    and for ``test_float_oracle.py``, which does point an execution oracle at
    it. This docstring used to end *"Pointing an execution oracle at it would
    measure the declared ℝ-vs-IEEE posture, not a defect"* — **withdrawn**.
    Measured 2026-08-28 over 1500 derandomised draws of THIS strategy alone,
    the executed value falls outside the propagator's box in **146** of them.
    What this grammar has no oracle for is the EXACT reading:
    :func:`eval_expr_exact` cannot evaluate a float harness, which is why the
    float question is answered by running the program instead.
    """
    n = draw(st.integers(1, max_decls))
    shape = draw(st.sampled_from(SHAPES))
    primary = draw(st.sampled_from(dtypes))
    decls = []
    for i in range(n):
        dt = primary if i == 0 or draw(st.booleans()) else draw(st.sampled_from(dtypes))
        # MIXED RANKS ON PURPOSE. The size-0 conjunct defect needs a rank-0
        # sibling next to a `(0,)` declaration: a `bool[0]` conjunct is
        # vacuously true over the whole box, and the tool narrowed the sibling
        # to a strict SUBSET anyway. A grammar that gave every declaration the
        # same shape could not build that harness.
        sh = shape if i == 0 or draw(st.booleans()) else draw(st.sampled_from(SHAPES))
        lo, hi = draw(_bounds(dt))
        decls.append(Decl(f"x{i}", sh, dt, lo, hi))
    decls = tuple(decls)

    names = [d.name for d in decls]
    floaty = any(d.dtype in FLOAT_DTYPES for d in decls)
    cast_targets = sorted({d.dtype for d in decls} - {"bool"}) or ["int32"]
    const_pool = st.sampled_from(FLOAT_POOL if floaty else INT_POOL)

    base = st.one_of(
        st.sampled_from(names).map(lambda n_: ("var", n_)),
        const_pool.map(lambda v: ("const", v)),
    )

    def extend(children):
        unary = UNARY_ANY + (UNARY_FLOAT if floaty else ())
        binary = BINARY_ANY + (BINARY_FLOAT if floaty else ())
        return st.one_of(
            st.tuples(st.sampled_from(unary), children).map(
                lambda t: ("un", t[0], t[1])),
            st.tuples(st.sampled_from(binary), children, children).map(
                lambda t: ("bin",) + t),
            children.map(lambda e: ("cancel", e)),
            children.map(lambda e: ("sum", e)),
            st.tuples(st.sampled_from(cast_targets), children).map(
                lambda t: ("cast", t[0], t[1])),
        )

    exprs = st.recursive(base, extend, max_leaves=5)
    cmps = st.tuples(st.sampled_from(CMP_OPS), exprs, exprs).map(
        lambda t: ("cmp",) + t)

    def extend_pred(children):
        return st.one_of(
            st.tuples(children, children).map(lambda t: ("and",) + t),
            st.tuples(children, children).map(lambda t: ("or",) + t),
            children.map(lambda p: ("not", p)),
        )

    preds = st.recursive(cmps, extend_pred, max_leaves=3)

    k = draw(st.integers(1, max_stmts))
    stmts = []
    for _ in range(k):
        kind = draw(st.sampled_from(("assert", "assert", "assume")))
        stmts.append(Stmt(kind, draw(preds)))
    if not any(s.kind == "assert" for s in stmts):
        stmts.append(Stmt("assert", draw(preds)))
    return Spec(decls, tuple(stmts))
