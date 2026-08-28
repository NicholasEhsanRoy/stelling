# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Running a generated float harness BOTH WAYS, and comparing the two.

The propagator computes a box for every value in the query. The compiled
program computes a float for every value in the query. This module runs both
and asks the one question the rest of this repository never asks:

    **is the executed value inside the box the propagator computed?**

That sentence is the invariant every VERIFIED rests on. It is the reason a
discharge over a declared set means anything at all about the program the user
will run. Nothing in ``src/`` asserts it and nothing in ``tests/`` asserted it
before this module, and it is FALSE on this tree in eight measured places
(:data:`MEMBERS`) — the seven this module was commissioned to pin
(:data:`SEVEN`) and one it found itself while its own classifier was being
written.

The nearest thing ``src/`` does have is ``stelling.falsify.probe``, which also
executes the program but asks whether the OBLIGATION is false rather than
whether each value is inside its box; driven over all eight it reports none of
them, and for four it says in its own note that it SAW the executed violation
and declined it. ``test_float_oracle.py``'s docstring carries that
measurement.

────────────────────────────────────────────────────────────────────────────
THE THREE THINGS THAT MAKE THIS INSTRUMENT WORTHLESS IF THEY ARE MISSED
────────────────────────────────────────────────────────────────────────────

Each was got wrong on a first attempt, each is measured, and each is a
separate mechanism in this file rather than a paragraph asking a reader to be
careful.

**1. ⊤ IS UNFALSIFIABLE, SO THE PASS RATE IS NOT A SAFETY SIGNAL.** A box of
``[-inf, inf]`` contains every finite float, so a query the propagator
declined on — every decline lands as ⊤ — reports "no violation" precisely
where the analysis already gave up. Two consequences, both acted on here:
:func:`read` counts ⊤ boxes separately (:attr:`Reading.top_boxes`) and the
properties assert a floor on NON-⊤ boxes examined; and the assumes are KEPT
(``assume_mode="constrain"``), because dropping them widens every box a
narrowing would have tightened. Measured 2026-08-28 at ``a90862b`` on jax
0.11.0, by the shipped residual leg at
``STELLING_PROPERTY_SCALE=12.5`` (1500 examples), of 11488 boxes read:
**1603 are ⊤ (14 %)** and a further **1177 are EMPTY (10 %)** — a size-0
declaration, a value with no elements to violate anything — leaving **8708
(76 %)** that a finite value could be caught outside of. A quarter of this
instrument's own field of view is a place where it cannot see a finite
violation at all, for two different reasons which are counted apart. NaN is
the one exception — no box contains a NaN, ⊤ included — which is why the
``nan`` cause survives a ⊤ and the other four do not.

**2. A DECLARED ENDPOINT IS NOT NECESSARILY A POINT OF THE PROGRAM'S DTYPE.**
``np.float32(1e-20)`` is ``9.9999997e-21``, which is BELOW a box declared
``(1e-20, 1e-10)``. Executing at it and comparing against a box the propagator
built for ``[1e-20, ...]`` reports a violation with no arithmetic in it at all.
:func:`snap_inward` fixes the direction: the sampled low endpoint is the
SMALLEST value of the program's dtype that is still ``>= lo``, and the high
endpoint the largest still ``<= hi``, by ``nextafter`` in the target format. A
declaration with no such value at all is not sampled and is reported as
``unsampleable`` — a refusal to answer, never a silent "no violation".

DRIVEN BOTH WAYS on 2026-08-28 at ``a90862b``, with the two ``nextafter``
steps deleted on a copy of this tree. On the identity harness
``x0 : float32 (1e-20, 1e-10) |- x0 >= 1e-20``, which contains no
arithmetic whatever, the unsnapped sampler reports **two** violations — one
per endpoint, ``9.999999682655225e-21`` and ``1.000000013351432e-10``, both
against the declaration's own box — and the snapped one reports none. Over 600
derandomised draws of :func:`uniform_float_programs`:

    unsnapped   330 read, 139 violations, **99 of them (71 %) on
                ``stelling_any`` itself** (61 overflow-to-inf, 38
                narrow-format-rounding)
    snapped     322 read,  84 violations, **0** on ``stelling_any``

Seventy-one per cent of the unsnapped instrument's findings are its own
sampler's rounding. That is not a bias to be corrected in the reading; it is a
different instrument.

RELATED, NOT REPAIRED HERE, AND REPORTED: the declared endpoints stelling
stores are binary64 images (``_bound_spelling.binary64_image``) and are not
snapped to the program's dtype grid, so the propagator's ``float32`` box for
``(1e-20, 1e-10)`` has endpoints no float32 can take. That is a defect in the
declaration layer, this instrument only works around it, and working around it
is what item 2 is.

**3. A UNIFORM ENVELOPE CANNOT REACH THE REASSOCIATION CLASS.** ``jnp.sum``
over ``f64[n]`` lowers to a two-window split at n >= 33 and to a fold at
n <= 32, and ``interval.reduce_sum`` models one order. A generator that draws
random envelopes and random data does not build a cancelling sum, and here
that is not a measurement but a PROOF about the generator:
``_grammar.SHAPES`` is ``((), (1,), (2,), (0,), (3,), (2, 2), (0, 3))``, whose
largest member has four elements, so no draw of
:func:`uniform_float_programs` can cross n >= 33 and the class is
UNREACHABLE there rather than merely unobserved. A count would have been the
weaker statement. Under the deliberate construction below, 200 programs per
size, same seed (driver in the builder's scratchpad, not in the tree):

    n = 16, 30, 31, 32 -> 0/200 each      n = 33 -> 18/200
    n = 34 -> 15/200                      n = 64 -> 9/200
    n = 128 -> 0/200

which is the n <= 32 fold / n >= 33 split boundary, re-derived from this
instrument rather than quoted — and **byte-identical on jax 0.10.2**, every
row, so the boundary is a property of the XLA lowering and not of one series.
All eight pinned members fire on 0.10.2 too, and all seven of the
box-genuinely-contains-it controls stay green there. Reaching the class needs three things AT ONCE —
a DEGENERATE envelope (so the box of each
element is a point and the box of the sum is the outward-rounded exact sum),
PER-ELEMENT CONSTANTS (so the 33 elements are not all the same number), and
FORCED CANCELLATION (so the exact sum is small while the partial sums are
huge). :func:`cancelling_sum_programs` builds exactly that, and the cost is
stated at that function: it is generator design aimed at a known class, it
finds nothing else, and it is confined to its own strategy so the uniform leg
stays unbiased.

────────────────────────────────────────────────────────────────────────────
WHAT THIS MODULE DOES NOT REACH
────────────────────────────────────────────────────────────────────────────

* **Integer and unsigned outputs are not compared.** A box endpoint is a
  binary64 float; ``int64`` values above 2**53 have no exact binary64 image,
  so a comparison there would invent violations out of the comparison's own
  rounding. Integer-dtype equations are counted (``skipped_integer``) and not
  judged. The wrap class those dtypes carry is ``test_oracle.py``'s subject
  and is judged there, exactly, in unbounded Python integers.
* **One point per round, and a handful of rounds.** :func:`sample_points`
  builds a small deterministic candidate set per declaration (both snapped
  endpoints, and 0, ±1 and the midpoint where the box admits them) plus one
  drawn interior point. A box is a set; this looks at up to six of its points.
  Finding one violating point refutes the containment claim; finding none
  confirms nothing.
* **The assumes are satisfied, not solved.** A sampled point that any
  ``assume`` rejects is discarded rather than repaired, so a harness whose
  admitted region is a thin slice of its declared box may contribute no points
  at all. That direction is safe (it costs evidence, never soundness) and it
  is counted: ``Reading.status == "no-admitted-point"``.
* **The comparison is against the FINAL environment.** ``interval_env``
  returns the environment after the whole forward walk, so an equation
  upstream of an ``assume`` is compared against the box that assume narrowed.
  Every point this module executes satisfies every assume, so it is inside
  the narrowed set too, and the comparison stays sound in the direction that
  matters: it can only under-report.
* **``semantics="ieee"`` is not exercised.** Every box here is a real-mode
  box, which is the mode every published verdict in this repository was
  stamped with. What an ieee-mode box would say about the same eight programs
  is not measured here.

────────────────────────────────────────────────────────────────────────────
THE MECHANISM, AND THE ONE COUPLING IT RESTS ON
────────────────────────────────────────────────────────────────────────────

``_jax_compat.trace_with_jaxpr`` returns the transcribed
:class:`stelling.ir.ClosedJaxpr` and jax's own ``ClosedJaxpr`` **from one
trace**, so the two objects describe the same program by construction rather
than by a second tracing that might not agree. The transcription is
mechanical, so equation ``i`` of one is equation ``i`` of the other — and
:func:`read` ASSERTS that (same length, same primitive at each index) instead
of assuming it, because the whole comparison is keyed on it.

Boxes come from :func:`stelling.propagate.interval_env`, the read-only
accessor that re-runs the forward walk and hands back ``var id -> box``. It is
pure: nothing observable by :func:`stelling.propagate.propagate` changes.

Execution is a plain jaxpr interpreter (:func:`execute`) binding each
primitive eagerly, with the three stelling primitives handled by hand:
``stelling_any`` takes the sampled array, and ``assume``/``assert_`` are the
identity on their operand (which is what their own ``def_impl`` says). It is
op-by-op rather than compiled ON PURPOSE — a compiled whole-program run is
free to fuse and reassociate ACROSS equations, and this instrument's claim is
about the value at each equation.

**AND THE HARNESS ITSELF IS BUILT BY ``_grammar.build`` AND NOT BY ANYTHING
HERE.** :func:`build` lowers this module's one extra node away
(:func:`_lowered_expr`) and hands the result to the same builder every other
property in this suite uses. A second evaluator living next door would make
every reading a comparison between the propagator and THAT evaluator, and its
disagreements would be reported as defects in stelling. The one thing kept
local is :func:`Program.render`, because a rank-1 constant has to be printed
at full precision to be a reproducer and ``repr`` of a numpy array is not:
``repr(np.array([0.1234567890123]))`` is ``array([0.12345679])``, which does
not round-trip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

import _grammar
from _grammar import Decl, Stmt

# ── formats ──────────────────────────────────────────────────────────────────

#: The float formats a declaration may name, and the numpy type that rounds to
#: each. ``bfloat16`` comes from ``ml_dtypes``, which jax depends on; it is
#: looked up rather than assumed so that a jax build without it degrades to
#: "this dtype is not sampled" instead of an import error at collection.
FLOAT_TYPES: dict = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
}
try:  # pragma: no cover - exercised by whichever branch this environment takes
    import ml_dtypes

    FLOAT_TYPES["bfloat16"] = ml_dtypes.bfloat16
except Exception:  # pragma: no cover - ml_dtypes is a jax dependency
    pass

#: Smallest positive NORMAL of each format. The flush classification reads it,
#: and it is written per format rather than as binary64's because the whole
#: point of that class is that the program's format is not the analysis's.
MIN_NORMAL = {
    "float64": 2.0**-1022,
    "float32": 2.0**-126,
    "float16": 2.0**-14,
    "bfloat16": 2.0**-126,
}

INT_DTYPES = (
    "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
)

# ── causes ───────────────────────────────────────────────────────────────────

NAN = "nan"
OVERFLOW = "overflow-to-inf"
NARROW = "narrow-format-rounding"
FLUSH = "flush-or-subnormal"
REASSOCIATION = "reduction-reassociation"
UNEXPLAINED = "UNEXPLAINED"

#: Every cause this module can name, in the order :func:`classify` tries them.
#: ``UNEXPLAINED`` is last and is the residual: a violation with none of the
#: five IEEE explanations is either a defect in the interval domain itself or a
#: sixth class nobody has written down, and either is a finding.
CAUSES = (NAN, OVERFLOW, NARROW, FLUSH, REASSOCIATION, UNEXPLAINED)


# ── the program under test ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Program:
    """A harness, in the same IR ``_grammar`` uses plus one extra node.

    ``decls`` are :class:`_grammar.Decl`, ``stmts`` are
    :class:`_grammar.Stmt`. The extra expression node is
    ``("aconst", values)`` — a rank-1 numpy constant, which is what the
    reassociation class needs and what a scalar-constant grammar cannot build.

    A DISTINCT TYPE FROM ``_grammar.Spec`` AND NOT A SUBCLASS, deliberately:
    ``Spec.render`` does not know ``aconst`` and raises on one, and a type that
    inherits a method which cannot answer for every value of the type is a trap
    for the next reader. :meth:`render` here does answer for every value of it.
    ``_grammar.build`` does not know the node either and never sees it —
    :func:`build` lowers it to a plain ``const`` first — which is what lets the
    program this module executes be the program the rest of this suite builds.
    The fields are the same, so :meth:`from_spec` is a copy.
    """

    decls: tuple
    stmts: tuple
    #: A pinned member's name, or ``""`` for a drawn program. The properties
    #: tag their census with it, which is what turns "they are all still
    #: found" into an assertion instead of a claim.
    label: str = ""

    @staticmethod
    def from_spec(spec) -> "Program":
        return Program(spec.decls, spec.stmts)

    def render(self) -> str:
        """The harness as runnable Python. This is what a reader acts on."""
        lines = ["def harness():"]
        for d in self.decls:
            lines.append(
                f"    {d.name} = any_array({d.shape!r}, {d.dtype!r}, "
                f"({d.lo!r}, {d.hi!r}))"
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


def _lit(v) -> str:
    if isinstance(v, float):
        if v == math.inf:
            return "math.inf"
        if v == -math.inf:
            return "-math.inf"
        if v != v:
            return "math.nan"
    return repr(v)


def render_expr(e) -> str:
    tag = e[0]
    if tag == "var":
        return e[1]
    if tag == "const":
        return _lit(e[1])
    if tag == "aconst":
        # THE WHOLE VECTOR, never an abbreviation. A shrunk counter-example is
        # only a reproducer if every number in it is there.
        return "np.array([" + ", ".join(_lit(v) for v in e[1]) + "])"
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


# ── lowering to the grammar the rest of this suite executes ─────────────────


def _lowered_expr(e):
    """``e`` with every ``aconst`` turned into a ``_grammar`` ``const``.

    THE ONE THING THIS MODULE MUST NOT DO IS EVALUATE THE HARNESS ITSELF. The
    program executed here has to be the program ``_grammar.build`` would build,
    or this instrument is comparing the propagator against a second evaluator
    written next door and reporting the difference between the two as a defect
    in stelling. So the only extension — a rank-1 constant, which the
    reassociation class needs and a scalar-constant grammar cannot express — is
    REWRITTEN AWAY here rather than interpreted, and everything downstream is
    ``_grammar``'s own :func:`_grammar.build`.

    Why the node exists at all rather than storing the array directly: a
    :class:`Program` is drawn by Hypothesis and pinned with ``@example``, and a
    frozen dataclass holding a numpy array is unhashable. ``("aconst",
    tuple_of_floats)`` is hashable; ``("const", ndarray)`` is what jax needs.
    """
    if not isinstance(e, tuple):
        return e
    if e[0] == "aconst":
        return ("const", np.asarray(e[1], dtype=np.float64))
    return tuple(_lowered_expr(c) for c in e)


def _lowered_stmts(stmts):
    return tuple(
        _grammar.Stmt(s.kind, _lowered_expr(s.pred)) for s in stmts
    )


def build(program: "Program"):
    """The zero-argument harness ``check``/``trace_with_jaxpr`` expect."""
    return _grammar.build(
        _grammar.Spec(program.decls, _lowered_stmts(program.stmts))
    )


# ── sampling: dtype-representable points INSIDE the declared box ─────────────


def snap_inward(dtype: str, lo: float, hi: float):
    """``(lo', hi')`` — the extreme values OF ``dtype`` inside ``[lo, hi]``.

    ``None`` when the dtype has no value in the box at all, which is a refusal
    to answer and not an empty answer.

    THE DIRECTION IS THE WHOLE POINT. ``np.float32(1e-20)`` rounds to
    ``9.9999997e-21``, which is outside a box declared ``(1e-20, 1e-10)``;
    executing there and comparing against a box built for ``[1e-20, ...]``
    reports a violation of the IDENTITY, with no arithmetic in it. So a cast
    that lands outside is stepped one ``nextafter`` back in, in the target
    format, and the result is checked against the binary64 bound rather than
    assumed correct.
    """
    if dtype == "bool":
        vals = [v for v in (False, True) if lo <= float(v) <= hi]
        return (vals[0], vals[-1]) if vals else None
    if dtype in INT_DTYPES:
        info = np.iinfo(getattr(np, dtype))
        a = info.min if lo == -math.inf else max(info.min, math.ceil(lo))
        b = info.max if hi == math.inf else min(info.max, math.floor(hi))
        return (a, b) if a <= b else None
    T = FLOAT_TYPES.get(dtype)
    if T is None:
        return None
    # `over="ignore"`: a bound of 1e300 cast to float32 IS +inf, and that is
    # the right answer — the check below then refuses the declaration as
    # unsampleable. numpy's warning would be a per-example RuntimeWarning on a
    # correct path, and a suite that prints warnings on correct paths teaches
    # its readers to ignore warnings.
    with np.errstate(over="ignore", invalid="ignore"):
        a, b = T(lo), T(hi)
        if float(a) < lo:
            a = np.nextafter(a, T(math.inf))
        if float(b) > hi:
            b = np.nextafter(b, T(-math.inf))
    if not (lo <= float(a) <= float(b) <= hi):
        return None
    return (a, b)


def sample_points(shape, dtype: str, lo: float, hi: float, interior: float):
    """Candidate concrete arrays for one declaration, or ``None``.

    The candidates, in order: the two snapped endpoints; then ``0.0``,
    ``1.0``, ``-1.0`` and the midpoint where the snapped box admits them; then
    one INTERIOR point at fraction ``interior`` of the box, which is the only
    part of the sampling the search gets to move. Duplicates are dropped.

    ``0.0`` is in the list for a named reason: ``y / y`` is NaN at exactly one
    point of ``[-1, 1]`` and an endpoint-only sampler measured **no
    violation** on it — the ``nan`` class is reachable only by sampling the
    interior, and 0 is the interior point that matters most often.
    """
    snapped = snap_inward(dtype, lo, hi)
    if snapped is None:
        return None
    a, b = snapped
    fa, fb = float(a), float(b)
    wanted = [fa, fb]
    for c in (0.0, 1.0, -1.0, fa + 0.5 * (fb - fa),
              fa + max(0.0, min(1.0, interior)) * (fb - fa)):
        if fa <= c <= fb and math.isfinite(c):
            wanted.append(c)
    npd = np.dtype(dtype) if dtype != "bfloat16" else np.dtype(FLOAT_TYPES["bfloat16"])
    out, seen = [], set()
    for v in wanted:
        try:
            # SILENCED, AND THE SILENCE IS THE POINT. This instrument's whole
            # job is to run programs at the edges of their formats, so a cast
            # that overflows or lands on an invalid value is a CORRECT PATH
            # here — the reading it produces is the finding. numpy would emit
            # a RuntimeWarning per example, and a suite that prints warnings
            # on correct paths teaches its readers to ignore warnings.
            with np.errstate(all="ignore"):
                arr = np.full(shape, v, dtype=npd)
        except (ValueError, OverflowError):  # pragma: no cover - defensive
            continue
        key = arr.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(arr)
    return out or None


# ── the two runs ─────────────────────────────────────────────────────────────


def boxes_of(closed):
    """``[(eqn index, primitive, outvar position, box)]`` for a transcribed query.

    ``assume_mode="constrain"`` and NOT the accessor's own ``"inert"``
    default. The default is right for its first consumer (the SMT emission
    reasons over the DECLARED box); it is wrong here, because dropping the
    assumes widens every box a narrowing would have tightened and this
    instrument's whole job is to find a box that is too NARROW.
    """
    from stelling.propagate import interval_env

    env = interval_env(closed, assume_mode="constrain")
    out = []
    for i, eqn in enumerate(closed.jaxpr.eqns):
        for j, ov in enumerate(eqn.outvars):
            box = env.get(ov.id)
            if box is not None:
                out.append((i, eqn.primitive, j, box))
    return out


def is_top(box) -> bool:
    """Is every element of this box ``[-inf, +inf]``? ⊤ contains every float.

    **AN EMPTY BOX IS NOT ⊤ AND THIS FUNCTION SAID IT WAS.** ``all()`` over no
    elements is ``True``, so a size-0 declaration — ``_grammar.SHAPES`` draws
    ``(0,)`` and ``(0, 3)`` — answered ⊤ here. Both are unfalsifiable, and for
    entirely different reasons: a ⊤ box is a value the analysis declined to
    bound, an empty one is a value that does not exist. Conflating them
    inflated the headline ⊤ figure with array shapes. Measured by the shipped
    residual leg at ``STELLING_PROPERTY_SCALE=12.5``: of 11488 boxes read,
    **1177 are size-0** against **1603** that are genuinely ⊤ — so the
    conflated figure was 2780, and 42 % of it was array shapes. The counters
    are three-way now and :attr:`Reading.empty_boxes` is the third.
    """
    return box.size > 0 and all(
        lo == -math.inf and hi == math.inf
        for lo, hi in zip(box.los, box.his)
    )


def declarations(jax_closed):
    """``[(shape, dtype, lo, hi)]``, one per ``stelling_any``, in walk order.

    Read out of the equation's own params rather than out of the
    :class:`Program`, so the sampler is driven by the query that was actually
    traced. A harness whose declaration jax rewrote would be sampled as jax
    left it.
    """
    out = []
    for eqn in jax_closed.jaxpr.eqns:
        if eqn.primitive.name == "stelling_any":
            p = dict(eqn.params)
            out.append((tuple(p["shape"]), str(p["dtype"]),
                        float(p["lo"]), float(p["hi"])))
    return out


def execute(jax_closed, points):
    """Run the jaxpr op by op; ``(values, operands, True)`` or ``(None, None, False)``.

    ``points[k]`` is bound at the k-th ``stelling_any``. ``False`` means a
    sampled point failed one of the harness's own ``assume``s, so it is not an
    admitted point and nothing about it is evidence.

    ``operands[i]`` is the tuple of concrete values equation ``i`` was applied
    to, LITERALS RESOLVED. It is taken from this walk rather than decoded back
    out of the transcription, because the transcription stores a literal as a
    :class:`stelling.ir.Array` byte payload and a classifier that decoded that
    itself would be modelling a representation instead of reading the values
    the program ran on. That distinction is not academic: it is exactly the
    bug that made the first version of :func:`classify` report the subnormal
    comparison as ``UNEXPLAINED`` — the literal was there, and unreadable.

    OP BY OP, NOT COMPILED, and that is the claim's shape rather than a
    convenience: a compiled whole-program run may fuse and reassociate across
    equations, and what is being checked here is the value AT each equation
    against the box AT that equation.
    """
    import jax.extend.core as jex

    env = {}
    for var, val in zip(jax_closed.jaxpr.constvars, jax_closed.consts):
        env[var] = val
    values = {}
    operands = {}
    k = 0
    # Same reason as :func:`sample_points`: an overflowing cast, an invalid
    # operation and a division by zero are what this walk is here to OBSERVE,
    # so numpy's and jax's warnings about them are noise on a correct path.
    with np.errstate(all="ignore"):
        for i, eqn in enumerate(jax_closed.jaxpr.eqns):
            name = eqn.primitive.name
            if name == "stelling_any":
                outs = [points[k]]
                operands[i] = ()
                k += 1
            else:
                ins = [v.val if isinstance(v, jex.Literal) else env[v]
                       for v in eqn.invars]
                operands[i] = tuple(ins)
                if name in ("stelling_assume", "stelling_assert",
                            "stelling_nonvacuity"):
                    if name == "stelling_assume" and not bool(
                        np.all(np.asarray(ins[0]))
                    ):
                        return None, None, False
                    outs = [ins[0]]
                else:
                    r = eqn.primitive.bind(*ins, **dict(eqn.params))
                    outs = list(r) if eqn.primitive.multiple_results else [r]
            for j, (ov, v) in enumerate(zip(eqn.outvars, outs)):
                env[ov] = v
                values[(i, j)] = v
    return values, operands, True


# ── classification ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Violation:
    eqn: int
    primitive: str
    dtype: str
    element: int
    executed: float
    lo: float
    hi: float
    cause: str
    point: tuple


def _prev(T, x):
    return float(np.nextafter(T(x), T(-math.inf)))


def _next(T, x):
    return float(np.nextafter(T(x), T(math.inf)))


def _exact_reduction(primitive: str, inputs) -> Fraction | None:
    """The EXACT real value of a reduction over its executed inputs, or ``None``.

    Only ``reduce_sum`` is answered. That is a deliberate floor rather than an
    oversight: the reassociation cause claims *"the analysis's box is right
    about ℝ and the execution reassociated"*, and it may only be claimed where
    the real value is computed exactly. Every other primitive falls through to
    ``UNEXPLAINED``, which is the direction a classifier has to fail in — it
    can produce a report to read, never a silent explanation.
    """
    if primitive != "reduce_sum":
        return None
    flat = np.asarray(inputs[0]).astype(np.float64).reshape(-1)
    if not np.all(np.isfinite(flat)):
        return None
    total = Fraction(0)
    for v in flat:
        total += Fraction(float(v))
    return total


def classify(primitive: str, dtype: str, v: float, lo: float, hi: float,
             inputs) -> str:
    """Why is ``v`` outside ``[lo, hi]``? One of :data:`CAUSES`.

    THE ORDER IS LOAD-BEARING AND IT WAS GOT WRONG ONCE, in the direction that
    flatters the instrument. ``reduction-reassociation`` was tried before
    ``flush-or-subnormal``, and its test — *"the exact real value of this
    reduction over its executed operands is INSIDE the box"* — is satisfied by
    ANY rounding a reduction does, flush-to-zero included. So
    ``ftz-subnormal-sum``, whose ``reduce_sum`` of two binary64 subnormals
    executes ``0.0`` against a strictly negative box, was reported as a
    reassociation of a two-element sum, which has no association to choose.
    The specific explanation has to be tried before the general one, and
    ``reassociation`` is the most general of the five.

    Each step below says what it excludes as well as what it claims.
    """
    if v != v:
        # No box contains a NaN, ⊤ included. This is the one cause that
        # survives a declined equation, and the reason the ⊤ count does not
        # bound this instrument's whole reach.
        return NAN
    if math.isinf(v) and math.isfinite(lo) and math.isfinite(hi):
        # An infinity the analysis said could not happen. Where the box is
        # already unbounded on that side there is no violation to classify.
        return OVERFLOW
    T = FLOAT_TYPES.get(dtype)
    if T is not None and dtype != "float64" and _prev(T, lo) <= v <= _next(T, hi):
        # Boxes are computed in binary64 whatever the program's dtype, so a
        # narrower output is one rounding away from the box by construction.
        # Claimed only within ONE ulp of the format either side; anything
        # wider is not a single rounding and is not explained by one.
        return NARROW
    mn = MIN_NORMAL.get(dtype)
    if mn is not None and min(abs(lo), abs(hi)) <= mn and abs(v) <= mn:
        # The box comes into the format's subnormal band and the executed
        # value is in it too: flush-to-zero, or gradual underflow the ℝ box
        # does not model. BOTH endpoints are checked — a box that merely
        # CONTAINS zero (say [-1, 1]) is not in the band, and an executed 0
        # there is not explained by underflow.
        return FLUSH
    if _an_operand_is_subnormal(inputs):
        # THE OPERAND SIDE, AND IT IS NOT A REFINEMENT — IT IS THE ONLY WAY A
        # COMPARISON'S VIOLATION CAN BE EXPLAINED AT ALL. A ``bool`` output
        # has no format and no subnormal band of its own, so the rule above
        # cannot answer for one, and the first thing this instrument found
        # that the rule above could not answer for was a comparison:
        # ``x0 >= 5e-324`` over ``x0 ∈ [0, 0]`` has the box ``(0, 0)`` —
        # definitely false, and right over ℝ — and **executes True** on this
        # backend, measured at ``a90862b`` on jax 0.11.0 / CPU. The subnormal
        # operand is flushed before the comparison, so the compiled program is
        # answering ``0 >= 0``. Same cause, one indirection out.
        return FLUSH
    exact = _exact_reduction(primitive, inputs)
    if exact is not None and Fraction(lo) <= exact <= Fraction(hi):
        # THE MOST GENERAL OF THE FIVE, hence last of them. The box is RIGHT
        # about the real value of this reduction and the executed float is not
        # it, and no narrower explanation above applied: the compiler chose an
        # association order the interval domain does not model.
        return REASSOCIATION
    if _an_operand_is_nan(inputs):  # pragma: no cover - a NaN operand
        # normally makes the output NaN and is caught above; kept because a
        # comparison against a NaN is a bool, not a NaN, and would otherwise
        # arrive here unexplained.
        return NAN
    return UNEXPLAINED


def _float_elements(inputs):
    """Every float element of an equation's operands, with its format name.

    NaNs and infinities included — :func:`_an_operand_is_nan` is one of the two
    readers and would see nothing without them.
    """
    for operand in inputs:
        if operand is None:
            continue
        arr = np.asarray(operand)
        name = str(arr.dtype)
        if name not in FLOAT_TYPES:
            continue
        for w in arr.reshape(-1):
            yield name, float(w)


def _an_operand_is_subnormal(inputs) -> bool:
    return any(
        0.0 < abs(w) <= MIN_NORMAL[name] for name, w in _float_elements(inputs)
    )


def _an_operand_is_nan(inputs) -> bool:
    return any(w != w for _, w in _float_elements(inputs))


# ── one reading ──────────────────────────────────────────────────────────────


@dataclass
class Reading:
    status: str = "read"
    violations: list = field(default_factory=list)
    boxes_read: int = 0
    top_boxes: int = 0
    empty_boxes: int = 0
    finite_boxes: int = 0
    skipped_integer: int = 0
    points: int = 0
    admitted_points: int = 0
    falsified_discharges: int = 0
    contradicted_refutations: int = 0
    detail: str = ""

    @property
    def first(self):
        return self.violations[0] if self.violations else None


def _refusals() -> frozenset:
    """What :func:`read` treats as "the tool, or jax, declined this input".

    ``_runner.REFUSALS`` verbatim, read at call time rather than copied: two
    lists is one place for them to disagree, and this module provokes the same
    refusals the rest of the suite does — an empty box, a bound with no value
    of its dtype, an unregistered primitive, a declaration jax will not build.
    A refusal is never a finding; :class:`Reading` records which one it was and
    the properties census it, so an over-broad entry cannot quietly turn this
    into a no-op.
    """
    from _runner import REFUSALS

    return REFUSALS


def read(program: Program, *, interior: float = 0.5, max_rounds: int = 6):
    """Propagate, execute, compare. Never raises for an input either side declined."""
    from stelling._jax_compat import trace_with_jaxpr

    refusals = _refusals()
    reading = Reading()
    try:
        # `errstate` for the same reason :func:`execute` uses it, one phase
        # earlier: jax CONSTANT-FOLDS a literal cast during tracing, so
        # `jnp.asarray(1e300).astype(jnp.float32)` — which this grammar draws
        # on purpose — emits an overflow RuntimeWarning before a single point
        # has been sampled.
        with np.errstate(all="ignore"):
            closed, jax_closed = trace_with_jaxpr(build(program))
    except Exception as exc:  # noqa: BLE001 — the type is the classification
        if type(exc).__name__ in refusals:
            reading.status = f"refused-trace:{type(exc).__name__}"
            return reading
        raise
    try:
        boxes = boxes_of(closed)
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in refusals:
            reading.status = f"refused-propagate:{type(exc).__name__}"
            return reading
        raise

    # THE ONE COUPLING, ASSERTED. The transcription is mechanical, so equation
    # i of the IR query is equation i of jax's; every box read below is keyed
    # on that. It is checked rather than trusted because a transcription that
    # ever reorders or drops an equation would make this whole module compare
    # one program's box against another program's value, silently.
    ir_prims = [e.primitive for e in closed.jaxpr.eqns]
    jax_prims = [e.primitive.name for e in jax_closed.jaxpr.eqns]
    assert ir_prims == jax_prims, (
        "THE TRANSCRIPTION NO LONGER MIRRORS THE JAXPR EQUATION FOR "
        "EQUATION, so this instrument would be comparing one program's box "
        f"against another program's value:\n  ir  {ir_prims}\n  jax {jax_prims}"
    )

    decls = declarations(jax_closed)
    candidates = []
    for shape, dtype, lo, hi in decls:
        c = sample_points(shape, dtype, lo, hi, interior)
        if c is None:
            reading.status = "unsampleable"
            reading.detail = (
                f"declaration {dtype}{shape} ({lo!r}, {hi!r}) has no value of "
                f"its own dtype inside the declared box"
            )
            return reading
        candidates.append(c)

    rounds = min(max_rounds, max((len(c) for c in candidates), default=1))
    for r in range(rounds):
        points = [c[min(r, len(c) - 1)] for c in candidates]
        reading.points += 1
        try:
            values, operands, admitted = execute(jax_closed, points)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ in refusals:
                reading.status = f"refused-execute:{type(exc).__name__}"
                return reading
            raise
        if not admitted:
            continue
        reading.admitted_points += 1
        # Boxes are counted ONCE per program, on the FIRST ADMITTED round and
        # not on round 0. A harness whose assume rejects the low endpoint —
        # `assume(x >= 1.0)` over a box of (-10, 10) is the shape — has no
        # admitted point in round 0, so counting there recorded `boxes_read =
        # 0` for a program this instrument had in fact read four boxes of.
        # Measured on that harness before the fix: `boxes=0`, and the ⊤
        # fraction those counters feed was therefore computed over a
        # population that silently excluded every assume-narrowed program.
        _compare(reading, boxes, values, operands, points,
                 reading.admitted_points == 1)
    if reading.admitted_points == 0 and reading.status == "read":
        reading.status = "no-admitted-point"
    return reading


def _compare(reading, boxes, values, operands, points, count_boxes):
    """Compare every box against its executed value; record the EARLIEST miss.

    Only the earliest violating equation is recorded per point. Once a value
    is outside its box, nothing downstream of it is guaranteed by anything —
    reporting the derived misses as findings would count one defect three
    times and would classify a boolean comparison's box as the cause of a
    float multiply's underflow.
    """
    point = tuple(float(np.asarray(p).reshape(-1)[0]) if np.asarray(p).size
                  else None for p in points)
    earliest = None
    for i, primitive, j, box in boxes:
        v = values.get((i, j))
        if v is None:  # pragma: no cover - every outvar is bound above
            continue
        arr = np.asarray(v)
        dtype = str(arr.dtype)
        if count_boxes:
            # THREE BUCKETS, NOT TWO. A ⊤ box is a value the analysis declined
            # to bound and contains every float; an EMPTY box is a value that
            # has no elements at all. Both are unfalsifiable and neither is
            # the other; only the third bucket is a place a finite value can
            # be found outside its box, and only the third is floored on.
            reading.boxes_read += 1
            if box.size == 0:
                reading.empty_boxes += 1
            elif is_top(box):
                reading.top_boxes += 1
            else:
                reading.finite_boxes += 1
        if dtype in INT_DTYPES:
            # A box endpoint is a binary64 float and an int64 above 2**53 has
            # no exact binary64 image, so a comparison here would report the
            # comparison's own rounding as a violation. Counted, not judged.
            if count_boxes:
                reading.skipped_integer += 1
            continue
        if dtype not in FLOAT_TYPES and dtype != "bool":
            continue
        flat = arr.astype(np.float64).reshape(-1)
        if flat.size != box.size:  # pragma: no cover - shapes agree by trace
            continue
        for idx in range(flat.size):
            x = float(flat[idx])
            lo, hi = box.los[idx], box.his[idx]
            if lo <= x <= hi:
                continue
            if earliest is not None and i >= earliest[0]:
                continue
            ins = operands.get(i, ())
            earliest = (
                i,
                Violation(
                    eqn=i, primitive=primitive, dtype=dtype, element=idx,
                    executed=x, lo=lo, hi=hi,
                    cause=classify(primitive, dtype, x, lo, hi, ins),
                    point=point,
                ),
            )
            break
    if earliest is not None:
        reading.violations.append(earliest[1])
    d, r = _contradicted_obligations(boxes, values)
    reading.falsified_discharges += d
    reading.contradicted_refutations += r


#: The two boxes the propagator reads as definite. Mirrors
#: ``propagate._bool_status``'s ``n_true == n`` / ``n_false`` tests, written
#: out rather than imported because they are two pairs of floats and importing
#: a private helper to compare four floats couples this file to a name it does
#: not need.
_BOOL_TRUE = (1.0, 1.0)
_BOOL_FALSE = (0.0, 0.0)


def _contradicted_obligations(boxes, values):
    """``(falsified discharges, contradicted refutations)`` at this point.

    THIS IS THE SENTENCE THE WHOLE ANALYSIS RESTS ON, counted, and it has two
    directions. A box violation anywhere is a defect in the bracket; a
    violation that reaches an OBLIGATION is a verdict the compiled program
    contradicts at a dtype-representable, admitted point of its own declared
    box:

    * every element of the box ``(1, 1)`` is ``discharged`` — the obligation
      is *definitely true for all elements* — and the predicate executes
      FALSE. That is a VERIFIED a user acts on, contradicted;
    * an element ``(0, 0)`` is ``violated-over-set`` — *definitely false for
      every element of the declared set* — and the predicate executes TRUE.
      That is a REFUTED contradicted, which is the recoverable direction and
      is counted separately rather than folded in.

    READ FROM THE BOX, NOT FROM THE ASSEMBLED VERDICT, and the two are not the
    same for the second bullet. ``propagate`` may WITHHOLD a
    ``violated-over-set`` after the walk — ``_withhold_uncertified_refutations``
    does exactly that where the assume state does not certify the region — so a
    box this counts as definitely-false may not reach the caller as REFUTED.
    The discharge direction has no such step and the two readings coincide
    there. Neither number is a count of verdicts; both are counts of what the
    propagator's own box says about an obligation at a point the program was
    executed at.
    """
    discharged = refuted = 0
    for i, primitive, j, box in boxes:
        if primitive != "stelling_assert":
            continue
        v = values.get((i, j))
        if v is None:  # pragma: no cover
            continue
        executed = bool(np.all(np.asarray(v)))
        pairs = list(zip(box.los, box.his))
        if pairs and all(pair == _BOOL_TRUE for pair in pairs):
            if not executed:
                discharged += 1
        elif any(pair == _BOOL_FALSE for pair in pairs):
            if bool(np.any(np.asarray(v))):
                refuted += 1
    return discharged, refuted


# ── the pinned members of the class ──────────────────────────────────────────
#
# Each is a program measured, on 2026-08-28 at `a90862b` on jax 0.11.0 /
# jaxlib 0.11.0 / numpy 2.5.2 / CPython 3.12.3, `JAX_PLATFORMS=cpu` and x64
# forced on by the property module's own fixture, to execute a value OUTSIDE
# the box the propagator computed for it.
#
# WHAT IS ASSERTED AND WHAT IS DATED, because they are not the same thing.
# `test_float_oracle.py` re-runs every entry on every run and asserts TWO
# things about it: that it still produces a violation at all (the per-member
# census floor) and that the violation's CAUSE is still the one registered
# here (the drift check in the residual leg's body). The box and the executed
# value in each comment are a DATED MEASUREMENT and are not asserted — an ulp
# of them is a property of the jax and the backend, and pinning that would
# turn a version bump into a red suite that says nothing about stelling.
#
# SEVEN OF THESE ARE THE SEVEN THIS MODULE WAS BUILT TO PIN. The eighth,
# `subnormal-comparison`, was found BY this instrument from an unbiased draw
# while the classifier was being written, and is registered here because a
# member found and not pinned is a member that gets lost.
#
# HOW FAR BACK THEY GO, RE-DERIVED RATHER THAN QUOTED. This module was
# commissioned with the sentence "four of the seven need no strict-sign
# certificate and reach v0.1.0, the first published release". Driven, with
# `git archive v0.1.0 src` on PYTHONPATH and v0.1.0's two-trace API
# (`harness.trace` plus `jax.make_jaxpr`, since `trace_with_jaxpr` did not
# exist yet) standing in for `read`'s one:
#
#   eight of the nine pinned programs violate box containment at v0.1.0 too;
#   FOUR of them falsify a DISCHARGE there — `ftz-subnormal-sum` (2 points),
#   `f32-underflow`, `f32-single-multiply` and `f32-exp` (1 each) — and
#   `subnormal-comparison` contradicts a refutation. The figure re-derives.
#
# THE ONE THAT DOES NOT REACH v0.1.0 IS `reassociation-n33`, AND THE REASON IS
# WORTH KNOWING: v0.1.0's `interval.mul` was not exact, so `x * K` at the
# pinned point boxed to `[7.205759403792793e+16, 7.205759403792795e+16]`
# instead of to a point, and `reduce_sum` of those came out
# `[-570.0019531250001, 694.0029296875001]` — wide enough to contain the
# executed 9.0. Tightening the multiply to the exact rational corner (audit
# 0.2.0 M16) is what made this class VISIBLE. The slack was never soundness;
# it was slack.


def _assert(pred):
    return Stmt("assert", pred)


def cancelling_vector(exponents, offsets, residue):
    """A per-element constant vector whose exact sum is small and whose partial
    sums are huge.

    ``(2**e, -(2**e) + d)`` pairs, then one residue. Every entry is exactly
    representable in binary64 — a power of two, and a power of two plus a
    small integer offset — and every PAIR sums exactly, so a left-to-right
    fold of the whole vector is exact and the interval domain's box for it is
    a point-to-within-outward-rounding. Any other association order rounds at
    the huge partial sums, and that difference is the whole of the
    reassociation class.
    """
    out = []
    for e, d in zip(exponents, offsets):
        out.append(2.0**e)
        out.append(-(2.0**e) + float(d))
    out.append(float(residue))
    return tuple(out)


#: The pinned reassociation witness. Found by 300 draws of the shape
#: :func:`cancelling_sum_programs` builds, of which 22 violated; this is the
#: first. Its measured reading is in :data:`MEMBERS`.
REASSOC_EXPONENTS = (56, 43, 51, 60, 44, 45, 37, 45, 37, 33, 49, 44, 43, 47,
                     57, 43)
REASSOC_OFFSETS = (-1, -1, 1, -1, 4, 2, -3, 0, -4, 4, -1, -3, 2, -4, -4, 4)
REASSOC_RESIDUE = 2
REASSOC_VECTOR = cancelling_vector(REASSOC_EXPONENTS, REASSOC_OFFSETS,
                                   REASSOC_RESIDUE)

#: ``float(np.float32(1.0000001))``. Written as the binary64 image of an
#: actual float32 rather than as ``1.0000001``, because ``(1.0000001,
#: 1.0000001)`` is REFUSED at the declaration door — no float32 lies in that
#: interval, and stelling says so. The refusal is correct and it is the reason
#: this member is spelled with the neighbour instead.
F32_JUST_ABOVE_ONE = float(np.float32(1.0000001))


def _member(label, decls, pred):
    return Program(decls, (_assert(pred),), label)


#: ``label -> (Program, cause, what was measured)``. The properties iterate
#: this; nothing else in this file mentions a member by name.
MEMBERS = {}


def _register(label, program, cause, measured):
    MEMBERS[label] = (program, cause, measured)


# (1) 1 / Σ(x·1e-200·1e-200) — the product underflows where the ℝ box does not.
#
# THE ENVELOPE IS PART OF THE MEMBER AND THE FIRST ONE TRIED WAS WRONG. With
# `x ∈ [1, 2]` the interval product's LOWER endpoint rounds outward to 0.0 as
# well, the box is `[0, 5e-324]`, it CONTAINS the executed 0.0, and there is no
# violation at all — measured, every equation IN. The class needs an envelope
# whose exact product lands in the subnormal band STRICTLY ABOVE zero, which
# `x ∈ [1e80, 1e85]` does: `x·1e-400 ∈ [1e-320, 1.0e-315]`, all subnormal, all
# positive.
_register(
    "underflow-reciprocal",
    _member(
        "underflow-reciprocal",
        (Decl("x0", (2,), "float64", 1e80, 1e85),),
        ("cmp", ">",
         ("bin", "div", ("const", 1.0),
          ("sum", ("bin", "mul",
                   ("bin", "mul", ("var", "x0"), ("const", 1e-200)),
                   ("const", 1e-200)))),
         ("const", 0.0)),
    ),
    FLUSH,
    "mul executes 0.0 against box [1e-320, 1.000000003e-315]; the reciprocal "
    "then executes +inf",
)

# (2) sum of two binary64 subnormals — flushed to zero by the backend.
_register(
    "ftz-subnormal-sum",
    _member(
        "ftz-subnormal-sum",
        (Decl("x0", (2,), "float64", -1e-323, -5e-324),),
        ("cmp", "<", ("sum", ("var", "x0")), ("const", 0.0)),
    ),
    FLUSH,
    "reduce_sum executes 0.0 against box [-2e-323, -1e-323], a box that is "
    "strictly negative",
)

# (3) a float32 square underflowing through a binary64 box.
_register(
    "f32-underflow",
    _member(
        "f32-underflow",
        (Decl("x0", (), "float32", 1e-20, 1e-10),),
        ("cmp", ">", ("bin", "mul", ("var", "x0"), ("var", "x0")),
         ("const", 0.0)),
    ),
    FLUSH,
    "mul executes 0.0f32 against box [9.999999999999997e-41, "
    "1.0000000000000001e-20]",
)

# (4) y / y at the one point of the box where it is NaN.
_register(
    "nan-from-y-over-y",
    _member(
        "nan-from-y-over-y",
        (Decl("x0", (), "float64", -1.0, 1.0),),
        ("cmp", "<=", ("bin", "div", ("var", "x0"), ("var", "x0")),
         ("const", 1.0)),
    ),
    NAN,
    "div executes NaN at x0 = 0.0; the box is ⊤ and no box contains a NaN",
)

# (5) reassociation at NORMAL magnitudes: n = 33, degenerate envelope,
#     per-element constants, forced cancellation.
_register(
    "reassociation-n33",
    _member(
        "reassociation-n33",
        (Decl("x0", (33,), "float64", 1.0, 1.0),),
        ("cmp", ">",
         ("sum", ("bin", "mul", ("var", "x0"), ("aconst", REASSOC_VECTOR))),
         ("const", 0.0)),
    ),
    REASSOCIATION,
    "reduce_sum executes 9.0 against box [-10.0, 6.0]; the EXACT real sum is "
    "3.0 and is inside the box, so the box is right about ℝ",
)

# (6) one float32 multiply, judged in binary64.
_register(
    "f32-single-multiply",
    _member(
        "f32-single-multiply",
        (Decl("x0", (), "float32", F32_JUST_ABOVE_ONE, F32_JUST_ABOVE_ONE),),
        ("cmp", ">", ("bin", "mul", ("var", "x0"), ("var", "x0")),
         ("const", 1.0000002)),
    ),
    NARROW,
    "mul executes 1.000000238418579 against the point box "
    "[1.0000002384185933, 1.0000002384185933]",
)

# (7) float32 exp on [-100, -50].
_register(
    "f32-exp",
    _member(
        "f32-exp",
        (Decl("x0", (), "float32", -100.0, -50.0),),
        ("cmp", ">", ("un", "exp", ("var", "x0")), ("const", 0.0)),
    ),
    FLUSH,
    "exp executes 0.0f32 at x0 = -100 against box [3.7200759760208356e-44, "
    "1.928749847963918e-22]; at x0 = -50 the same equation executes "
    "1.9287498933537385e-22, one float32 ulp above the box's upper endpoint",
)

# (8) NOT ONE OF THE SEVEN. Found by this instrument, from an unbiased draw of
#     `uniform_float_programs`, while `classify` was being written — the
#     violation it could not explain was this one, and the reason it could not
#     is that a comparison's output is a `bool` and has no subnormal band of
#     its own. The cause is one indirection out, in the OPERAND.
#
#     `jnp.asarray(0.0) >= jnp.asarray(5e-324)` is **True** on this backend.
#     Python says False and numpy says False; XLA CPU flushes the subnormal
#     operand before comparing, so the compiled program is answering
#     `0 >= 0`. The propagator's box for the comparison is `(0, 0)` —
#     definitely false, and correct over ℝ — so the obligation is
#     `violated-over-set` and the execution contradicts a REFUTED rather than
#     a VERIFIED. The recoverable direction, and a member of the same class.
_register(
    "subnormal-comparison",
    _member(
        "subnormal-comparison",
        (Decl("x0", (), "float64", 0.0, 0.0),),
        ("cmp", ">=", ("var", "x0"), ("const", 5e-324)),
    ),
    FLUSH,
    "ge executes True (1.0) against the box [0.0, 0.0]; the subnormal "
    "operand 5e-324 is flushed to zero before the comparison",
)

#: The seven this module was commissioned to find, in the order they were
#: handed over. Named separately from :data:`MEMBER_NAMES` so that "all seven"
#: stays a checkable claim about a fixed list after an eighth was added.
SEVEN = (
    "underflow-reciprocal",
    "ftz-subnormal-sum",
    "f32-underflow",
    "nan-from-y-over-y",
    "reassociation-n33",
    "f32-single-multiply",
    "f32-exp",
)

#: Every pinned member, the seven plus what this instrument found itself.
MEMBER_NAMES = tuple(MEMBERS)

assert set(SEVEN) <= set(MEMBER_NAMES), sorted(set(SEVEN) - set(MEMBER_NAMES))


# ── strategies ───────────────────────────────────────────────────────────────

from hypothesis import strategies as st  # noqa: E402


def member_programs():
    """Every pinned member, as a strategy, so the search draws them like
    anything else — and so the containment leg reaches one within a few
    examples instead of relying on the unbiased grammar stumbling into the
    class."""
    return st.sampled_from([p for p, _, _ in MEMBERS.values()])


def uniform_float_programs():
    """``_grammar.general_specs`` as-is — the unbiased leg, and the control.

    UNBIASED IS THE POINT. This is the leg whose partition can be compared
    against a run of any other generator, and the leg that would catch a class
    nobody has thought of. It reaches NaN, flush-or-subnormal and
    narrow-format DENSELY; it reaches overflow-to-inf only BY LUCK — 0 in one
    1500-draw derandomised run and 7 in another, which is why
    ``test_float_oracle.OVERFLOW_PROBE`` is pinned; and it reaches the
    reassociation class **never**, because ``SHAPES`` tops out at four
    elements and no draw of it can cross the n >= 33 boundary. The last of
    those three is a proof about the generator rather than a zero in a table,
    and it is stated here rather than left to be inferred.
    """
    return _grammar.general_specs().map(Program.from_spec)


@st.composite
def cancelling_sum_programs(draw):
    """The deliberate reassociation shape, and what it costs.

    Three ingredients at once, none of which an envelope-only generator
    supplies:

    * a **degenerate envelope** — ``x`` is pinned to one value, so the box of
      each element is a point and the box of the sum is the outward-rounded
      exact sum rather than a wide interval that would swallow anything;
    * **per-element constants** — a rank-1 ``aconst``, so the 33 summands are
      33 different numbers. A scalar-constant grammar makes every element
      equal and every association order agree;
    * **forced cancellation** — ``(2**e, -(2**e) + d)`` pairs with ``e`` drawn
      from 30 to 60, so the partial sums reach ``2**60`` while the exact total
      is single digits.
      Without it the rounding difference between two association orders is far
      below the box's own width and nothing shows.

    THE COST, SAID PLAINLY. This strategy is generator design aimed at ONE
    known class. It finds that class and nothing else: every program it builds
    is ``sum(x * K) > 0`` over a pinned box, so it cannot report a defect in
    ``div``, in a cast, in a connective or in an assume. It is a separate
    strategy from :func:`uniform_float_programs` for that reason — mixing it
    in would make the unbiased leg's partition a statement about this shape's
    density rather than about the tool.

    ``n`` is drawn with 33 three times in the pool and 32 once, so the
    measured boundary is exercised from BOTH sides in every run. Measured
    2026-08-28 at ``a90862b``, 200 programs of this exact construction per
    size: **0/200** at each of n = 16, 30, 31 and 32, **18/200** at n = 33,
    **15/200** at n = 34, **9/200** at n = 64 and **0/200** at n = 128. The
    density is low on purpose — the pairs have to straddle the window boundary
    for the two orders to differ — and 400 derandomised draws of this strategy
    alone produced **15** violations, every one of them
    ``reduction-reassociation``, at 4.9 ms/example. That n = 128 is back to
    zero is recorded and NOT explained: 64 exact pairs may simply re-cancel
    under whatever split XLA picks there. It is a fact about this
    construction, not about the boundary, and the n <= 32 rows are what
    establish the boundary.
    """
    n = draw(st.sampled_from((33, 33, 33, 32, 34, 64)))
    pairs = n // 2
    exponents = draw(st.lists(st.integers(30, 60), min_size=pairs,
                              max_size=pairs))
    offsets = draw(st.lists(st.integers(-4, 4), min_size=pairs,
                            max_size=pairs))
    residue = draw(st.integers(-4, 4))
    vector = cancelling_vector(exponents, offsets, residue)
    vector = vector[:n] if len(vector) >= n else vector + (0.0,) * (n - len(vector))
    pinned = draw(st.sampled_from((1.0, -1.0, 2.0)))
    return Program(
        (Decl("x0", (n,), "float64", pinned, pinned),),
        (_assert(("cmp", ">",
                  ("sum", ("bin", "mul", ("var", "x0"), ("aconst", vector))),
                  ("const", 0.0))),),
    )


def float_oracle_inputs():
    """One strategy, as a tuple: ``(program, interior fraction)``.

    Everything the properties draw comes from here, because ``st.data()``
    makes ``@example`` impossible and the residual property pins ten of them —
    every member, and the two probes for the causes the search reaches only by
    luck or not at all.
    """
    programs = st.one_of(
        uniform_float_programs(),
        cancelling_sum_programs(),
        member_programs(),
    )
    return st.tuples(programs, st.floats(0.0, 1.0))
