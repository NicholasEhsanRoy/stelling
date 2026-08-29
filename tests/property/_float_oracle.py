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
before this module, and it is FALSE on this tree in **nine** measured places
(:data:`MEMBERS`) — the seven this module was commissioned to pin
(:data:`SEVEN`) and two it found itself: a comparison against a subnormal
constant, found while its own classifier was being written, and a precondition
that narrows away a point the compiled program admits, found by an independent
audit of this file.

The nearest thing ``src/`` does have is ``stelling.falsify.probe``, which also
executes the program but asks whether the OBLIGATION is false rather than
whether each value is inside its box; driven over all nine it reports none of
them, and for four it says in its own note that it SAW the executed violation
and declined it. ``test_float_oracle.py``'s docstring carries that
measurement.

────────────────────────────────────────────────────────────────────────────
THE THREE THINGS THAT MAKE THIS INSTRUMENT WORTHLESS IF THEY ARE MISSED
────────────────────────────────────────────────────────────────────────────

Each was got wrong on a first attempt, each is measured, and each is a
separate mechanism in this file rather than a paragraph asking a reader to be
careful.

**1. ⊤ IS UNFALSIFIABLE, SO THE PASS RATE IS NOT A SAFETY SIGNAL.** A ⊤ box
admits every value its dtype can take, so a query the propagator declined
on — every decline lands as ⊤ — reports "no violation" precisely where the
analysis already gave up. Two consequences, both acted on here: :func:`read`
counts ⊤ boxes separately (:attr:`Reading.top_boxes`) and the properties
assert a floor on NON-⊤ boxes examined; and the assumes are KEPT
(``assume_mode="constrain"``), because dropping them widens every box a
narrowing would have tightened.

The bucket table is a DATED RECORD and the figure in it has been wrong twice.
Its current value is in :data:`test_float_oracle.FLOAT_ORACLE_MEASURED`, which
is anchored to a commit and to a command; what belongs HERE is the shape of
the partition and the two corrections that produced it, because those do not
rot.

**THE FOURTH BUCKET IS A CORRECTION AND THE THIRD WAS HIDING IN IT.** The
table read "8708 (76 %) that a finite value could be caught outside of", and
1327 of that 8708 were integer boxes the comparison ``continue``s past —
counted as falsifiable and never compared. Integer-dtype outputs are skipped
because a box endpoint is a binary64 float and an ``int64`` above 2**53 has no
exact binary64 image, so the comparison would report its own rounding; that is
a sound decision and it was not disclosed anywhere a reader could find it.

**AND THE ``compared`` BUCKET THEN HID THE BOOL LATTICE'S OWN ⊤, WHICH IS THE
SAME MISTAKE A THIRD TIME.** :func:`is_top` tested for ``±inf`` and nothing
else, so a ``bool`` box of ``[0, 1]`` — which admits both values a bool can
take, and which every declined comparison, connective and obligation in this
grammar produces — was counted as a place a violation could be found. Measured
2026-08-29 by the residual leg at ``STELLING_PROPERTY_SCALE=12.5``, with
:func:`is_top` instrumented to report the split it was hiding, of the 7719
boxes the four-bucket table called ``compared``: 2228 were bool ``[0, 1]``,
1777 were a definite bool, 3714 were float. The
falsifiable field was **5491 of 13041**, and the disclosure in this file said
39 % blind, ``README.md`` and ``ci.yml`` said 41 %, and the true figure is
**58 %**. :func:`is_top` asks *"does this box admit every value the executed
dtype can take"* now, which is one rule and not a list of dtype spellings.

So a large majority of this instrument's own field of view is a place where it
cannot see a finite violation at all, for three different reasons which are
counted apart. NaN is the one exception — no box contains a NaN, ⊤ included —
which is why the ``nan`` cause survives a ⊤ and the other four do not, and why
a float ⊤ is very slightly less blind than a bool one.

**2. A DECLARED ENDPOINT IS NOT NECESSARILY A POINT OF THE PROGRAM'S DTYPE.**
``np.float32(1e-20)`` is ``9.9999997e-21``, which is BELOW a box declared
``(1e-20, 1e-10)``. Executing at it and comparing against a box the propagator
built for ``[1e-20, ...]`` reports a violation with no arithmetic in it at all.
:func:`snap_inward` fixes the direction: the sampled low endpoint is the
SMALLEST value of the program's dtype that is still ``>= lo``, and the high
endpoint the largest still ``<= hi``, by ``nextafter`` in the target format. A
declaration with no such value at all is not sampled and is reported as
``unsampleable`` — a refusal to answer, never a silent "no violation".

DRIVEN BOTH WAYS on 2026-08-28 at ``874d8ba``, with the two ``nextafter``
steps deleted on a copy of this tree. On the identity harness
``x0 : float32 (1e-20, 1e-10) |- x0 >= 1e-20``, which contains no
arithmetic whatever, the unsnapped sampler reports **two** violations — one
per endpoint, ``9.999999682655225e-21`` and ``1.000000013351432e-10``, both
against the declaration's own box — and the snapped one reports none. Over 600
derandomised draws of :func:`uniform_float_programs`:

    unsnapped   330 read, 246 violations, **162 of them (66 %) on
                ``stelling_any`` itself** (99 overflow-to-inf, 46
                narrow-format-rounding, 17 flush-or-subnormal)
    snapped     322 read, 148 violations, **0** on ``stelling_any``

Two thirds of the unsnapped instrument's findings are its own sampler's
rounding. That is not a bias to be corrected in the reading; it is a different
instrument. **AND NOTHING ASSERTED IT UNTIL AN AUDIT SAID SO**: with the snap
deleted, the containment leg still XFAILed and the residual leg failed only
incidentally, through one member's cause-drift check. The guard is now
explicit — see :attr:`Reading.sampler_artefacts` and the residual leg — and it
is not a blanket ban, because a ``stelling_any`` violation under a NARROWING
ASSUME is a genuine member of the class (``assume-narrows-past-the-program``).

THE SAME MISTAKE, ONE DTYPE OVER, WAS LIVE IN THIS FILE UNTIL THE SAME AUDIT.
The candidates were built as binary64 and handed to ``np.full(..., dtype=...)``
for integer declarations too. Above 2**53 that is lossy and at 2**63 it WRAPS:
``any_array((), "int64", (1, 2**63 - 1))`` sampled ``-9223372036854775808``,
outside its own declared box, and the comparison one equation downstream then
executed True against a box the propagator had proved definitely false — this
module reporting its own sampler as a defect in stelling. Integer candidates
stay Python integers now; :func:`sample_points` carries the record.

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
Every pinned member fires on 0.10.2 too, and the seven programs the builder's
red/green driver runs as boxes that GENUINELY contain their value stay green
there. That driver is in the builder's scratchpad and is NOT in this tree,
which is said here because the sentence beside it names a table that is not in
the tree either and said so. Reaching the class needs three things AT ONCE —
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

  **AND A NULL FROM IT IS NOT MARKED AS UNINFORMATIVE, WHICH IS MEASURED.**
  The candidate set is built from the DECLARATION and knows nothing about the
  harness's assumes, so a narrowing whose boundary is not one of those six
  points is invisible. Driven on a tree carrying the documented ieee-only
  endpoint bump in real mode (the plant the ``assume-narrowing`` fix was
  red-driven against):
  ``x0: float64 (0,1); assume(x0 >= 0.0); assert_(x0 > 0.0)`` is a wrong
  VERIFIED and this module reports it, because 0.0 is a candidate; the SAME
  wrong VERIFIED with ``assume(x0 >= 0.25)`` is reported as **0 violations**
  with ``Reading.status == "read"``. Nothing distinguishes that null from a
  null on a correct program. Repairing it means sampling the assume bounds
  too, which is a change to what the sampler is given rather than to how it
  rounds, and it is not made here.
* **The assumes are satisfied, not solved.** A sampled point that any
  ``assume`` rejects is discarded rather than repaired, so a harness whose
  admitted region is a thin slice of its declared box may contribute no points
  at all. That direction is safe (it costs evidence, never soundness) and it
  is counted: ``Reading.status == "no-admitted-point"``.
* **The comparison is against the NARROWED environment, and the claim that
  this can only under-report is WITHDRAWN.** ``interval_env`` returns the
  environment after the whole forward walk, so an equation is compared against
  the box the assumes narrowed. This paragraph used to conclude *"every point
  this module executes satisfies every assume, so it is inside the narrowed
  set too, and the comparison stays sound in the direction that matters: it
  can only under-report."* The first clause is true and the conclusion needs a
  premise it does not have — that the narrowing is sound against EXECUTED
  semantics — and ``assume-narrows-past-the-program``, this module's own ninth
  member, is precisely where it is not: the assume narrows over ℝ and the
  compiled comparison flushes and admits the point. Such a violation is
  CREATED by comparing against the narrowed box. It is a real finding and it
  is a different one, so it is separated rather than assumed away: every
  violation is re-read against the ``inert`` environment and one that
  disappears there is :data:`NARROWING` — but ONLY where the sampled point is
  proved to be outside the assumed set over ℝ, because the two-box comparison
  alone cannot tell a narrowing that is right about ℝ from one that is wrong
  about it. See :func:`assumes_hold_over_reals`, which is the third fact, and
  :func:`classify`, which spends it.
* **``semantics="ieee"`` is not exercised.** Every box here is a real-mode
  box, which is the mode every published verdict in this repository was
  stamped with. What an ieee-mode box would say about the same nine programs
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
#: ``finfo`` implementations, asked in order. numpy owns the three IEEE
#: formats and refuses ``bfloat16``; ``ml_dtypes`` owns that one.
_FINFOS = [np.finfo]
try:  # pragma: no cover - exercised by whichever branch this environment takes
    _FINFOS.append(ml_dtypes.finfo)
except NameError:  # pragma: no cover - ml_dtypes is a jax dependency
    pass

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
UNCLASSIFIED = "unclassified"
NARROWING = "assume-narrowing"

#: How a cause was reached. ``proved`` means an exact rational reading of the
#: equation backed it; ``sound-by-construction`` means the rule needs no
#: reading (a NaN is in no box, an infinity the box excludes is in no box, a
#: narrowing is decided by comparing two boxes); ``heuristic`` means a rule
#: that names a plausible cause and proves nothing. The last one is the row
#: "0 unclassified" was silently standing in for.
PROVED = "proved"
STRUCTURAL = "sound-by-construction"
HEURISTIC = "heuristic"
BASES = (PROVED, STRUCTURAL, HEURISTIC)

#: Every cause this module can name, in the order :func:`classify` tries them.
#:
#: THE LAST TWO ARE DIFFERENT ANSWERS AND THEY USED TO BE ONE.
#: :data:`UNEXPLAINED` is a **proof**: the exact real value of the equation on
#: the values it ran on is computable, it is OUTSIDE the box, and no IEEE
#: difference accounts for the miss — so the box is wrong about the REALS,
#: which is a defect in the interval domain and is what the residual leg
#: forbids. :data:`UNCLASSIFIED` is a **refusal**, and it covers two shapes:
#: this classifier has no exact reading of that primitive, so it can neither
#: explain the violation nor prove it unexplainable; or it has one, the box is
#: RIGHT about ℝ, and the rounding that moved the float out is one none of the
#: four IEEE rules names. Neither is a defect in the interval domain, and
#: neither may redden the residual leg. Returning ``UNEXPLAINED`` for them
#: would make the leg cry wolf on ``sqrt``; returning ``FLUSH``, which is what
#: it did, made "0 unexplained" mean less than it looked.
CAUSES = (NARROWING, NAN, OVERFLOW, NARROW, FLUSH, REASSOCIATION,
          UNEXPLAINED, UNCLASSIFIED)


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
    #: WHICH STRATEGY BUILT THIS, and it is not derivable from ``label``.
    #: ``read/unlabelled`` was documented as "THE UNBIASED HALF, COUNTED
    #: SEPARATELY" and counted the AIMED strategy too, because
    #: :func:`cancelling_sum_programs` also yields ``label == ""``. The three
    #: generators have wildly different densities — measured over 1500
    #: examples, 326 violations from 672 uniform draws, 19 from 724 cancelling
    #: draws and 218 from 115 member re-draws — so a partition that does not
    #: separate them is a statement about how often an ``@example`` was
    #: re-drawn.
    #:
    #: **AND IT WAS SPENT ON THE CAUSE PARTITION AND NOT ON THE FIGURE EVERY
    #: CONSUMER QUOTES**, which is the harder half of the same mistake. The
    #: falsified-discharge count — "the denominator a repair is scoped on" —
    #: and the distinct-program column of the cause table were both left
    #: uncrossed for a round. Crossed, "116 violations over 21 distinct
    #: programs" is five pinned ``@example``s re-drawn plus one aimed
    #: strategy's shrink neighbours, and **nothing at all from the unbiased
    #: leg**. ``test_float_oracle._record`` does the crossing and
    #: :data:`test_float_oracle.FLOAT_ORACLE_MEASURED` carries the table.
    source: str = ""

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


def _finite_max(dtype: str):
    """The largest finite magnitude of ``dtype``, or ``None``.

    ``np.finfo`` refuses ``bfloat16`` (``data type dtype(bfloat16) not
    compatible with finfo``), which is an extension dtype numpy does not own,
    so ``ml_dtypes`` is asked for its own formats. ``None`` is a refusal and
    makes the declaration unsampleable rather than sampled at a guess.
    """
    T = FLOAT_TYPES.get(dtype)
    if T is None:
        return None
    for finfo in _FINFOS:
        try:
            return float(finfo(T).max)
        except (ValueError, TypeError):  # pragma: no cover - one of the two answers
            continue
    return None  # pragma: no cover - every shipped format is covered


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

    AND AN INFINITE ENDPOINT IS A BOUND, NOT A MEMBER — THE THIRD DEFECT OF
    THAT SHAPE IN THIS FUNCTION. The guard below reads
    ``lo <= float(a) <= float(b) <= hi``, which ``-inf <= -inf`` satisfies, so
    an unbounded declaration was sampled AT ±inf. Under the stamped ℝ
    semantics an infinite endpoint means *unbounded*, which
    ``_jax_compat.any_array`` says in its own words when it refuses
    ``(inf, inf)`` as "an infinite point has no members under ℝ semantics" —
    so ±inf is not in the declared set, and executing there is the float32
    endpoint problem again, one bound out. Measured before the fix,
    exhaustively over the shipped run's declarations: **71 of 33684 candidates
    were non-finite**, and 51 of the 563 violations occurred at one, every one
    of them in the ``nan`` row.

    CLAMPED INWARD RATHER THAN REFUSED, and the two precedents in ``src/``
    point opposite ways for a reason. ``stelling.falsify.probe`` REFUSES a
    float declaration with an infinite bound (``unbounded-declaration``)
    because it needs a uniform sample over the declared set and there is no
    such distribution on an unbounded one; its BOOL branch clamps instead —
    "replaced by a finite value on the same side" — because on a two-element
    set clamping loses nothing. This sampler is the second case in spirit: it
    draws named candidate points rather than a distribution, and the format's
    largest finite magnitude is an ordinary point of an unbounded set.
    Refusing would throw away every reading from such a declaration in order
    to remove a point that should never have been sampled.
    """
    if dtype == "bool":
        vals = [v for v in (False, True) if lo <= float(v) <= hi]
        return (vals[0], vals[-1]) if vals else None
    if dtype in INT_DTYPES:
        info = np.iinfo(getattr(np, dtype))
        # `math.ceil(inf)` and `math.floor(-inf)` RAISE `OverflowError`, and
        # `math.ceil(nan)` raises `ValueError` — so an integer declaration of
        # `(inf, inf)` came out of this function as an exception rather than as
        # a refusal, and an exception from a sampler is a red suite that names
        # the wrong subject. Found by 8000 hypothesis examples over all 13
        # dtypes and all seven `_grammar.SHAPES`; unreachable from the shipped
        # generators, which is why it was latent. A low bound of `+inf`, a high
        # bound of `-inf` and a NaN bound all name NO INTEGER AT ALL, which is
        # the same answer `a > b` gives below, so they take the same exit.
        if math.isnan(lo) or math.isnan(hi):
            return None
        if lo == math.inf or hi == -math.inf:
            return None
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
    finite_max = _finite_max(dtype)
    if finite_max is None:
        return None
    if lo == -math.inf:
        lo = -finite_max
    if hi == math.inf:
        hi = finite_max
    with np.errstate(over="ignore", invalid="ignore"):
        a, b = T(lo), T(hi)
        if float(a) < lo:
            a = np.nextafter(a, T(math.inf))
        if float(b) > hi:
            b = np.nextafter(b, T(-math.inf))
    if not (lo <= float(a) <= float(b) <= hi):
        return None
    if not (math.isfinite(float(a)) and math.isfinite(float(b))):
        # belt and braces on the clamp above: every candidate this function
        # returns is a point OF the declared set, and ±inf is not one.
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

    ────────────────────────────────────────────────────────────────────────
    AND EVERY ARRAY IT RETURNS IS CHECKED TO BE A POINT OF ``(lo, hi)``,
    HERE, AT THE SOURCE
    ────────────────────────────────────────────────────────────────────────

    THREE DEFECTS OF ONE SHAPE HAVE BEEN FOUND IN THIS SAMPLER — a float32
    endpoint below its own binary64 bound, integer candidates routed through
    ``float`` so ``any_array((), "int64", (1, 2**63 - 1))`` sampled
    ``-9223372036854775808``, and an INFINITE endpoint sampled as if it were a
    member of an unbounded set — and each was caught by a reader noticing a
    strange violation, not by a check.

    **THE CHECK THAT WAS SUPPOSED TO CATCH THEM COULD NOT SEE THE THIRD.**
    :attr:`Reading.sampler_artefacts` fires on a violation AT
    ``stelling_any`` — and an unbounded declaration's own box is ⊤, so ±inf is
    INSIDE it and there is no violation to fire on. It was structurally blind
    to the very class it had just been extended for. Driven 2026-08-29, same
    seed at ``STELLING_PROPERTY_SCALE=12.5``, with the ±inf clamp removed from
    :func:`snap_inward` AND this assertion removed (without which the run
    cannot finish, which is the point): the ``nan`` cause goes from **162 to
    164 events** and from 30 to 35 distinct unbiased programs, every other
    cause is identical, and ``sampler_artefacts`` is **0 in both runs**. An
    earlier run of the same A/B, reported by the audit that found this and
    taken before the classifier reorder, gave 177 -> 252; it does not
    reproduce here and the figure above is the one this tree gives. The delta
    is smaller and the conclusion is the one that matters and is unchanged:
    the counter written to catch this class reports zero on a tree that has
    it.

    So membership is asserted where the point is BUILT rather than inferred
    from what the program did with it later. For a float dtype every element
    must be FINITE and within ``[lo, hi]`` — an infinite endpoint is a bound
    and not a member, which is what ``_jax_compat.any_array`` says in its own
    words when it refuses ``(inf, inf)``; for an integer or ``bool`` dtype
    every element must be within the bounds in exact Python arithmetic. This
    is an ``AssertionError`` and NOT a counted violation: a sampler outside
    its own declared box is this module being wrong, not stelling, and it must
    stop the run rather than be censused as a finding.

    WHAT IT DOES NOT REACH: the declaration stelling itself builds. These
    bounds are the ones the harness asked for; the box the propagator computes
    from them holds the ``binary64`` IMAGE of each bound
    (``_bound_spelling.binary64_image``), which for a ``float32`` declaration
    can be an endpoint no float32 can take. This assertion says the sampled
    point is in the DECLARED set, not that it is in the propagator's box for
    it, and the gap between those two is a declaration-layer defect this
    module only works around.
    """
    snapped = snap_inward(dtype, lo, hi)
    if snapped is None:
        return None
    a, b = snapped
    t = max(0.0, min(1.0, interior))
    if dtype in INT_DTYPES:
        # INTEGERS NEVER GO THROUGH `float`, AND THAT IS A BUG THIS FILE HAD.
        # The candidates used to be built as binary64 and handed to
        # `np.full(..., dtype=int64)`. Above 2**53 that is lossy and at
        # 2**63 it WRAPS: `any_array((), "int64", (1, 2**63 - 1))` sampled
        # `-9223372036854775808` — outside its own declared box — and the
        # comparison one equation downstream then executed True against a box
        # the propagator had proved definitely false. Measured, and it was
        # this module reporting its own sampler as a defect in stelling: the
        # exact analogue, one dtype over, of the float32 endpoint problem
        # :func:`snap_inward` was written for. Python integers are exact and
        # numpy stores them exactly, so the arithmetic below stays in ℤ.
        wanted = [a, b]
        for c in (0, 1, -1, (a + b) // 2, a + int(t * (b - a))):
            if a <= c <= b:
                wanted.append(c)
    else:
        fa, fb = float(a), float(b)
        wanted = [fa, fb]
        for c in (0.0, 1.0, -1.0, fa + 0.5 * (fb - fa), fa + t * (fb - fa)):
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
        _assert_points_of(arr, dtype, lo, hi)
        out.append(arr)
    return out or None


def _assert_points_of(arr, dtype: str, lo: float, hi: float) -> None:
    """Every element of ``arr`` is a point of the declared set ``(lo, hi)``.

    The membership assertion :func:`sample_points` ends on. Split out so the
    condition is readable and so the message can name the three defects it
    exists for. A size-0 array has no element and passes, which is right: an
    empty declaration contributes no point rather than a bad one.

    WHAT IT DOES NOT REACH, AT THE INTEGER DTYPES. ``lo`` and ``hi`` arrive
    here as binary64 floats — that is what ``declarations()`` reads out of the
    equation's params and what stelling stores — so for an ``int64`` bound
    above 2**53 the interval this checks against is the bound's binary64
    IMAGE. ``src``'s ``any_array`` refuses any integer bound whose image would
    NARROW the declared set and admits only the widening direction, so the
    check is a WIDENED one: it can pass a value that is outside the true
    declared set and inside its image. The comparison in ``_violating_elements``
    has the same limit for the same reason, which is why integer-dtype outputs
    are never compared at all. What closes the gap in practice is
    :func:`snap_inward`, which clamps integer candidates to
    ``np.iinfo``'s own ``min``/``max`` in exact Python integers before this
    function ever sees them.
    """
    flat = np.asarray(arr).reshape(-1)
    if flat.size == 0:
        return
    if dtype in INT_DTYPES or dtype == "bool":
        # EXACT PYTHON ARITHMETIC, because `float(2**63 - 1)` is `2**63` and a
        # check that rounds its own subject is the defect it is checking for.
        bad = [int(w) for w in flat.tolist()
               if not (lo <= int(w) <= hi)]
    else:
        vals = [float(w) for w in flat]
        bad = [w for w in vals
               if not (math.isfinite(w) and lo <= w <= hi)]
    assert not bad, (
        "THIS MODULE'S OWN SAMPLER LEFT THE DECLARED SET. %r is not a point "
        "of the %s declaration (%r, %r) — %d of %d element(s) are outside it "
        "or non-finite. Everything downstream of this would be a violation "
        "of a box the propagator computed for a set this value is not in, "
        "i.e. this instrument reporting itself as a defect in stelling. "
        "Three defects of exactly this shape have already been found here "
        "(a float32 endpoint below its own binary64 bound; integer "
        "candidates routed through `float`; an infinite endpoint sampled as "
        "a member of an unbounded set), and `Reading.sampler_artefacts` "
        "could not see the third."
        % (bad[:4], dtype, lo, hi, len(bad), flat.size)
    )


# ── the two runs ─────────────────────────────────────────────────────────────


def boxes_of(closed, assume_mode: str = "constrain"):
    """``[(eqn index, primitive, outvar position, box)]`` for a transcribed query.

    ``assume_mode="constrain"`` by default and NOT the accessor's own
    ``"inert"``. The accessor's default is right for its first consumer (the
    SMT emission reasons over the DECLARED box); it is wrong here, because
    dropping the assumes widens every box a narrowing would have tightened and
    this instrument's whole job is to find a box that is too NARROW.

    The ``"inert"`` reading is asked for as well, only where a violation was
    found, for two questions that neither the constrained boxes nor the
    harness's TEXT can answer: whether the violation exists at all against the
    UN-NARROWED box (:data:`NARROWING`), and whether an ``assume`` narrowed
    the declaration a ``stelling_any`` violation sits on
    (:attr:`Reading.sampler_artefacts`).
    """
    from stelling.propagate import interval_env

    env = interval_env(closed, assume_mode=assume_mode)
    out = []
    for i, eqn in enumerate(closed.jaxpr.eqns):
        for j, ov in enumerate(eqn.outvars):
            box = env.get(ov.id)
            if box is not None:
                out.append((i, eqn.primitive, j, box))
    return out


def is_top(box, dtype: str) -> bool:
    """Is this box ⊤ — does it admit EVERY VALUE THE EXECUTED DTYPE CAN TAKE?

    A ⊤ box cannot be violated by anything the program computes, so it is a
    place this instrument can see nothing, and counting it as a place a
    violation could have been caught is the difference between a field of view
    and a pass rate. The one exception is a NaN, which no box contains, ⊤
    included — which is why the ``nan`` cause survives a ⊤ and the others do
    not, and why a FLOAT ⊤ is very slightly less blind than a bool one.

    **IT TESTED ``±inf`` AND NOTHING ELSE, SO THE BOOL LATTICE'S OWN TOP WAS
    COUNTED AS FALSIFIABLE.** A ``bool`` output takes exactly two values, so a
    box of ``[0, 1]`` admits both of them and NO executed bool can fall
    outside it — it is exactly as unfalsifiable as ``[-inf, inf]``, and for
    exactly the same reason: the analysis declined to say anything. Every
    comparison, every connective and every obligation in this grammar has a
    bool output, so that is not a corner. Measured 2026-08-29 by the residual
    leg at ``STELLING_PROPERTY_SCALE=12.5``, with this function instrumented
    to report the split the four-bucket table was hiding, of the **7719**
    boxes that table called ``compared``:

        2228  bool ``[0, 1]``   the bool lattice's ⊤: no executed bool is
                                outside it, ever
        1777  bool definite     ``[0, 0]`` or ``[1, 1]`` — falsifiable, and
                                where five of the nine members are caught
        3714  float            falsifiable

    so the falsifiable field was **5491 of 13041 = 42 %**, and the disclosure
    everywhere read 39 %, 41 % or 24 % blind against a true **58 %**.

    THE RULE IS ONE RULE AND NOT A LIST OF DTYPE SPELLINGS. *Admits every
    value of its dtype* collapses to ``[-inf, +inf]`` for every float format
    on its own — a float32 can take ``±inf``, so a box of
    ``[-3.4e38, 3.4e38]`` does NOT admit every float32 and is falsifiable —
    and to ``lo <= 0 and hi >= 1`` for ``bool``, whose value set is ``{0, 1}``
    with no infinity and no NaN in it. Integer dtypes are never compared at
    all (see :func:`_violating_elements`) and reach this function only through
    the float branch, which is what they did before.

    **AN EMPTY BOX IS NOT ⊤ AND THIS FUNCTION SAID IT WAS.** ``all()`` over no
    elements is ``True``, so a size-0 declaration — ``_grammar.SHAPES`` draws
    ``(0,)`` and ``(0, 3)`` — answered ⊤ here. Both are unfalsifiable, and for
    entirely different reasons: a ⊤ box is a value the analysis declined to
    bound, an empty one is a value that does not exist. Conflating them
    inflated the headline ⊤ figure with array shapes. Measured by the residual
    leg at ``STELLING_PROPERTY_SCALE=12.5`` on the tree that carried it: of
    13727 boxes read,
    **1697 are size-0** against **2042** that are genuinely ⊤ — so the
    conflated figure would be 3739, and 45 % of it array shapes. The counters
    are four-way (⊤ / empty / integer / compared) and
    :attr:`Reading.empty_boxes` is the second.
    """
    if box.size == 0:
        return False
    if dtype == "bool":
        return all(lo <= 0.0 and hi >= 1.0
                   for lo, hi in zip(box.los, box.his))
    return all(
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


# ── the second route: the same jaxpr as ONE compiled region ─────────────────
#
# THE OP-BY-OP WALK IS A GRANULARITY, AND IT IS THE TRACE'S RATHER THAN THE
# PROGRAM'S. `Primitive.bind` runs each equation on its own, so XLA never sees
# two of them together — and an equation whose operands are all compile-time
# constants is then executed at RUNTIME, where XLA:CPU flushes a subnormal
# operand, instead of being CONSTANT-FOLDED in full IEEE precision the way the
# compiled program folds it. Measured 2026-08-28 at `874d8ba`:
#
#     jax.jit(lambda: jnp.sqrt(5e-324))()    2.2227587494850775e-162
#     np.sqrt(5e-324)                        2.2227587494850775e-162
#     jnp.sqrt(jnp.asarray(5e-324))          0.0            <- the eager route
#
# and the box for that equation is `[2.2227587494850772e-162,
# 2.222758749485078e-162]`, which CONTAINS the compiled answer. So the eager
# route alone reports a violation the compiled program does not have, and this
# module's headline noun is *the value the COMPILED PROGRAM computes*.
#
# `src/stelling/falsify.py` already ships the guard for this class — a second
# compilation route, decline reason `executed-float-depends-on-granularity` —
# and this is that guard one axis over. Two properties of it are copied
# deliberately from :func:`stelling.falsify._whole_program_route`:
#
# * **it is consulted only after the eager route found a violation, and it can
#   only DECLINE.** It never promotes anything to a violation. That is
#   one-sided ON PURPOSE and it is a blindness, stated here: a violation only
#   the compiled route would show is not looked for by this module;
# * **it is built once per program**, so jax's jit cache keys on one function
#   object and compiles one module rather than one per sampled point.


def staged_runner(jax_closed):
    """``(jitted, keys)`` — the jaxpr interpreted inside ONE :func:`jax.jit`.

    The sampled points are jit ARGUMENTS and the jaxpr's literals stay
    literals, which is the same split the compiled program has: an
    ``any_array`` declaration is an input to the program the user runs and a
    literal is a constant in it. Getting that split wrong in either direction
    would make this route answer about a different program.

    ────────────────────────────────────────────────────────────────────────
    AND IT IS LOAD-BEARING FOR THE REPAIR, NOT ONLY FOR THIS MODULE. READ
    THIS BEFORE BUILDING A REPRODUCER
    ────────────────────────────────────────────────────────────────────────

    **MOST OF THESE VIOLATIONS DISAPPEAR IF THE DECLARATION IS FOLDED IN AS A
    CONSTANT**, because XLA then constant-folds the arithmetic in full
    precision and the value the box excludes is never computed. *Any repair
    validated against a constant-folded reproducer will look fixed and will
    not be.*

    Measured 2026-08-29 on this tree (jax 0.11.0, jaxlib 0.11.0, CPU, x64 on),
    by running every pinned member through this function and through a second
    interpreter identical to it except that the points are CLOSED OVER instead
    of passed in — driver in the builder's scratchpad, not in the tree — and
    counting box-violating ``(round, equation, output, element)`` sites and
    :func:`_contradicted_obligations` under each. **EACH ROUTE IS GATED ON ITS
    OWN ``assume``s**, which is the correction below:

        member                          adm ARG  adm FLD  sites  FOLDED  disch
        underflow-reciprocal                  3        3      9       0  0 -> 0
        ftz-subnormal-sum                     2        2      6       0  2 -> 0
        f32-underflow                         3        3      3       1  1 -> 0
        nan-from-y-over-y                     3        3      1       1  0 -> 0
        reassociation-n33                     1        1      1       1  0 -> 0
        f32-single-multiply                   1        1      3       3  1 -> 1
        f32-exp                               3        3      4       1  1 -> 0
        subnormal-comparison                  1        1      2       0  0 -> 0
        assume-narrows-past-the-program       3        2      3       0  1 -> 0

    **4 of the 5 discharge-falsifying members lose the falsified discharge
    entirely**, 6 of the 9 lose violating sites, and 4 of those lose every
    site they had.

    THE LAST ROW READ ``1 -> 1`` AND IT IS THE ROW THIS TABLE EXISTS FOR. The
    first run of this driver gated BOTH routes on the EAGER walk's admission —
    it mirrored :func:`_compare`'s asymmetry, which gates on the eager route
    and counts on the compiled one, and that asymmetry is wrong for the
    question THIS table asks. ``assume-narrows-past-the-program``'s whole
    mechanism is that the compiled comparison flushes the subnormal:

        route with ``x0`` a jit ARGUMENT   ``x0 >= 5e-324``  ->  True
        route with ``x0`` CLOSED OVER      ``x0 >= 5e-324``  ->  False

    Folded, ``0.0 >= 5e-324`` is constant-folded in full precision, the point
    LEAVES the admitted set, and the folded program has no violation there and
    no falsified discharge — so the row is ``1 -> 0`` and the count is 4, not
    3. Every other row is byte-identical under both gatings. The ungated
    reading told a repair team that this member survives constant folding; it
    does not, and they would have folded it in, seen nothing, and had no way
    to tell that from a fix. The conclusion is unchanged and gets stronger.
    """
    import jax
    import jax.extend.core as jex

    jaxpr = jax_closed.jaxpr
    keys = [(i, j) for i, eqn in enumerate(jaxpr.eqns)
            for j in range(len(eqn.outvars))]

    def staged(*pts):
        env = {}
        for var, val in zip(jaxpr.constvars, jax_closed.consts):
            env[var] = val
        out = {}
        k = 0
        for i, eqn in enumerate(jaxpr.eqns):
            name = eqn.primitive.name
            if name == "stelling_any":
                outs = [pts[k]]
                k += 1
            else:
                ins = [a.val if isinstance(a, jex.Literal) else env[a]
                       for a in eqn.invars]
                if name in ("stelling_assume", "stelling_assert",
                            "stelling_nonvacuity"):
                    outs = [ins[0]]
                else:
                    r = eqn.primitive.bind(*ins, **dict(eqn.params))
                    outs = list(r) if eqn.primitive.multiple_results else [r]
            for j, (ov, v) in enumerate(zip(eqn.outvars, outs)):
                env[ov] = v
                out[(i, j)] = v
        return tuple(out[key] for key in keys)

    return jax.jit(staged), keys


def operands_of(jax_closed, values):
    """``eqn index -> its operands``, read back out of a values map.

    Used for the compiled route, whose interpreter runs inside ``jit`` and
    cannot hand intermediate operands back cheaply. Literals answer their own
    value, exactly as :func:`execute` resolves them.
    """
    import jax.extend.core as jex

    var_of = {}
    for i, eqn in enumerate(jax_closed.jaxpr.eqns):
        for j, ov in enumerate(eqn.outvars):
            var_of[ov] = (i, j)
    consts = dict(zip(jax_closed.jaxpr.constvars, jax_closed.consts))
    out = {}
    for i, eqn in enumerate(jax_closed.jaxpr.eqns):
        ins = []
        for atom in eqn.invars:
            if isinstance(atom, jex.Literal):
                ins.append(atom.val)
            elif atom in var_of:
                ins.append(values.get(var_of[atom]))
            else:
                ins.append(consts.get(atom))
        out[i] = tuple(ins)
    return out


def ancestors(jax_closed):
    """``eqn index -> every equation whose output reaches its inputs``.

    THE CUT THIS EXISTS FOR IS DATA DEPENDENCE AND IT USED TO BE EQUATION
    INDEX. :func:`_compare` reports the earliest violation and drops the rest,
    on the rationale that *once a value is outside its box nothing DOWNSTREAM
    of it is guaranteed by anything*. Keyed on the index, that rationale is
    false for a SIBLING: `assert_((y/y <= 1.0) & (x + x >= 5.0))` has the
    division and the addition under one `&`, neither downstream of the other,
    and an index cut reports the NaN and silently drops an addition that may
    be violating for a completely different reason. Driven on a copy of this
    tree with `interval.add` shifted by +1.0, one wrong box: alone it reports
    `add executed 5.0 -> UNEXPLAINED`; with one NaN equation placed EARLIER it
    reports the NaN and the add is never seen. A second false-green route in
    the leg whose whole job is to forbid an unexplained violation.

    A jaxpr's equations are in topological order, so one forward pass is
    enough and this asserts nothing it has not seen.
    """
    import jax.extend.core as jex

    producer = {}
    for i, eqn in enumerate(jax_closed.jaxpr.eqns):
        for ov in eqn.outvars:
            producer[ov] = i
    out: dict = {}
    for i, eqn in enumerate(jax_closed.jaxpr.eqns):
        seen: set = set()
        for atom in eqn.invars:
            # a jax Literal is UNHASHABLE, so it cannot even be looked up here
            # — and it has no producer to be downstream of.
            if isinstance(atom, jex.Literal):
                continue
            producing = producer.get(atom)
            if producing is None:
                continue
            seen.add(producing)
            seen |= out.get(producing, set())
        out[i] = seen
    return out


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
    #: One of :data:`BASES`. A cause and the strength of the evidence for it
    #: are different facts, and reporting only the first is what let 48
    #: violations named by a rule that proves nothing be counted as part of
    #: "0 unclassified".
    basis: str = HEURISTIC


def _prev(T, x):
    # `errstate` for the reason :func:`sample_points` gives: a box endpoint of
    # 1e300 cast to float32 IS an overflow, and the answer that produces is the
    # right one for the window below. A RuntimeWarning on a correct path
    # teaches its readers to ignore warnings.
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.nextafter(T(x), T(-math.inf)))


def _next(T, x):
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.nextafter(T(x), T(math.inf)))


# ── the exact reading, which is what makes UNEXPLAINED a PROOF ──────────────

_EXACT_UNARY = {
    "neg": lambda a: -a,
    "abs": abs,
    "copy": lambda a: a,
}

#: ``convert_element_type`` IS NOT IN THE TABLE ABOVE, and it was, as the
#: identity. See :func:`_exact_convert` — an entry that needs the dtypes on
#: both sides cannot be a one-argument lambda, and the identity is FALSE for
#: half the conversions this grammar draws.
_EXACT_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": max,
    "min": min,
}
_EXACT_CMP = {
    "ge": lambda a, b: a >= b, "gt": lambda a, b: a > b,
    "le": lambda a, b: a <= b, "lt": lambda a, b: a < b,
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
}


class _Exact:
    """One operand, read as exact rationals — ``None`` per element it refuses.

    A flat tuple of :class:`~fractions.Fraction`, with ``None`` in every
    position this reader will not answer for: a non-arithmetic dtype, a
    non-finite value, an operand that is not there at all.

    IT EXISTS SO THERE IS ONE TABLE AND NOT TWO. :func:`exact_value` reads a
    numpy operand and :func:`exact_walk` reads an operand that is already
    rational, and both have to dispatch on the same primitive names with the
    same broadcast rule. Two dispatches would be a second evaluator living next
    door, which is the thing this module's own docstring forbids at the top —
    so the dispatch (:func:`_exact_apply`) takes these and the two callers
    differ only in how they build one.
    """

    __slots__ = ("vals", "dtype")

    def __init__(self, vals, dtype: str = ""):
        self.vals = tuple(vals)
        #: The operand's own dtype NAME, or ``""`` where it is not known.
        #: Carried because one entry in the table — the conversion — is a
        #: claim about the pair of dtypes and not about the value alone, and
        #: a dispatch that cannot see the source dtype has to guess.
        self.dtype = dtype

    @staticmethod
    def of(operand) -> "_Exact":
        """One numpy operand (or ``None``, or a python scalar) as ``_Exact``.

        INTEGERS DO NOT GO THROUGH ``float`` HERE, AND THEY DID. ``float(w)``
        is lossy above 2**53 and at 2**63 it rounds to a value the dtype
        cannot hold, so an ``int64`` operand's "exact" rational was the
        rounded one — the same defect :func:`sample_points` records having had
        one function over, in the sampler that builds these very arrays.
        Python integers are exact and :class:`~fractions.Fraction` takes them
        directly.
        """
        if operand is None:
            return _Exact((None,))
        arr = np.asarray(operand)
        name = str(arr.dtype)
        if name not in FLOAT_TYPES and name != "bool" and name not in INT_DTYPES:
            return _Exact((None,) * max(1, arr.size), name)
        out = []
        if name in INT_DTYPES or name == "bool":
            for w in arr.reshape(-1).tolist():
                out.append(Fraction(int(w)))
        else:
            for w in arr.reshape(-1):
                w = float(w)
                out.append(Fraction(w) if math.isfinite(w) else None)
        return _Exact(out, name)

    def element(self, idx: int, n_out: int):
        """This operand's contribution to output element ``idx``, or ``None``.

        The broadcast rule is *size 1 or size n_out and nothing else*: a shape
        this reader cannot align is a refusal, never a guess.
        """
        n = len(self.vals)
        if n == 1:
            return self.vals[0]
        if n == n_out and idx < n:
            return self.vals[idx]
        return None

    def whole(self):
        """Every element, or ``None`` if any one of them is a refusal."""
        return None if any(v is None for v in self.vals) else self.vals


def _int_bounds(dtype: str):
    """``(lo, hi)`` as exact Python integers for an integer-valued dtype.

    READ OUT OF ``numpy`` RATHER THAN TYPED. A table of per-dtype ranges here
    would be a second copy of one ``src/`` also keeps, and two copies is one
    place for them to disagree — but the range of ``int32`` is not a fact
    about stelling either, so the answer comes from ``np.iinfo`` and belongs
    to neither. ``bool`` is the one numpy will not answer for.
    """
    if dtype == "bool":
        return 0, 1
    if dtype in INT_DTYPES:
        info = np.iinfo(getattr(np, dtype))
        return int(info.min), int(info.max)
    return None


def _exact_convert(a, src: str, dst: str):
    """The exact real value of ``convert_element_type`` on one element.

    ────────────────────────────────────────────────────────────────────────
    THIS WAS ``lambda a: a`` IN THE UNARY TABLE, AND THE IDENTITY IS FALSE
    FOR HALF THE CONVERSIONS THIS GRAMMAR DRAWS
    ────────────────────────────────────────────────────────────────────────

    A float→int conversion TRUNCATES TOWARD ZERO and a narrowing int→int
    conversion WRAPS; neither is the identity, and both were being reported as
    one. :func:`exact_walk` promises *"where this walk answers at all, it
    answers exactly"* and :func:`_exact_apply` promises every entry *"computes
    the same real number the primitive is specified to compute"*. Neither
    promise survived the entry.

    IT IS NOT A JUDGEMENT CALL ABOUT ℝ, WHICH IS WHAT MAKES IT A DEFECT AND
    NOT A POSTURE. ``stelling.propagate._t_convert`` models this same
    primitive as ``math.trunc`` on both endpoints, so the instrument and the
    analysis it audits disagreed about the real value of the same equation, by
    construction. Driven at the classifier, before the fix::

        cast(int32, x0) <= 0  at x0 = 0.5   [x0 : float64 (0.0, 1.0)]
          assumes_hold_over_reals -> False
          TRUTH over ℝ            -> True    (trunc(0.5) = 0, and 0 <= 0)
          the compiled assume admits the point too

    A wrong ``False`` there mints ``(assume-narrowing, proved)`` and
    SUPPRESSES the fall-through that would have proved ``UNEXPLAINED`` — the
    one answer the residual leg forbids. A wrong ``True`` runs the other way.

    REACH, MEASURED RATHER THAN GUESSED: of 573 traced draws of
    :func:`uniform_float_programs`, 271 carry an ``assume``, 8 carry a
    float→int cast, and **4 carry one inside an assume's dependency cone — in
    all 4 the walk refused for another reason.** Every live
    ``assume-narrowing/proved`` event is the pinned member, whose assume
    contains no cast. So the defect is PLAUSIBLE and was not driven end to end
    (reaching a wrong ``PROVED`` also needs a live ``propagate`` narrowing
    defect, which is not constructible on a clean tree). It is repaired
    because a table entry that is false is false whether or not today's search
    reaches it.

    THE RULE, AND IT IS DERIVED FROM THE PRIMITIVE RATHER THAN FROM
    ``_t_convert``. What ``convert_element_type`` computes is a fact about
    jax, not about stelling, so this reader states it independently and agrees
    with that transfer rather than importing it — an instrument that imported
    the analysis's own whitelist could not see a defect in it:

    * **into a float format** — the identity. Converting a real number into a
      binary format does not change the real number; the rounding into the
      format is the IEEE difference :data:`NARROW` names, and the whole
      ``narrow-format-rounding`` rule depends on this reading;
    * **into ``bool``** — ``a != 0``, which is what the primitive computes;
    * **from a float into an integer dtype** — ``trunc(a)``, and only while
      that lands inside the target's range. Outside it jax clamps or wraps
      rather than truncating, and this reader will not guess which;
    * **from an integer or ``bool`` into an integer dtype** — the identity
      while the value fits the target, a REFUSAL when it does not, because
      the narrowing wraps;
    * anything else, or either dtype unknown — a refusal.
    """
    if not src or not dst:
        return None
    if src == dst:
        return a
    if dst in FLOAT_TYPES:
        return a
    if dst == "bool":
        return Fraction(0) if a == 0 else Fraction(1)
    bounds = _int_bounds(dst)
    if bounds is None:
        return None
    lo_b, hi_b = bounds
    if src in FLOAT_TYPES:
        out = Fraction(math.trunc(a))
    elif src in INT_DTYPES or src == "bool":
        out = a
    else:
        return None
    return out if lo_b <= out <= hi_b else None


def _exact_apply(primitive: str, ops, idx: int, n_out: int,
                 out_dtype: str = ""):
    """The EXACT REAL value of one output element from ``_Exact`` operands.

    The one dispatch. The table is deliberately small — the four field
    operations, the two lattice operations, magnitude and negation, the
    format conversion (:func:`_exact_convert`, which needs the dtypes on both
    sides and is therefore not a lambda), the six comparisons, and
    ``reduce_sum`` — because every entry is a claim that this function
    computes the same real number the primitive is specified to compute, and a
    wrong entry would manufacture an ``UNEXPLAINED`` out of its own
    arithmetic. ``sqrt``, ``exp``, ``sign``, ``integer_pow``, the boolean
    connectives, every reduction other than ``reduce_sum`` and everything not
    listed answer ``None``.

    ``out_dtype`` is the dtype of the output element under judgement. Only the
    conversion reads it, and without it the conversion REFUSES rather than
    assuming the identity — which is the direction that costs evidence instead
    of soundness.
    """
    if primitive == "convert_element_type" and len(ops) == 1:
        a = ops[0].element(idx, n_out)
        return None if a is None else _exact_convert(a, ops[0].dtype,
                                                     out_dtype)
    if primitive == "reduce_sum":
        if n_out != 1 or len(ops) != 1:
            return None
        vals = ops[0].whole()
        if vals is None:
            return None
        total = Fraction(0)
        for w in vals:
            total += w
        return total
    if primitive in _EXACT_UNARY and len(ops) == 1:
        a = ops[0].element(idx, n_out)
        return None if a is None else Fraction(_EXACT_UNARY[primitive](a))
    if len(ops) == 2:
        a = ops[0].element(idx, n_out)
        b = ops[1].element(idx, n_out)
        if a is None or b is None:
            return None
        if primitive in _EXACT_BINARY:
            return Fraction(_EXACT_BINARY[primitive](a, b))
        if primitive in _EXACT_CMP:
            return Fraction(1) if _EXACT_CMP[primitive](a, b) else Fraction(0)
        if primitive == "div" and b != 0:
            return a / b
    return None


def exact_value(primitive: str, operands, idx: int, n_out: int,
                out_dtype: str = ""):
    """The EXACT REAL value of one output element, or ``None``.

    ``None`` is a refusal to answer and it is what separates
    :data:`UNEXPLAINED` from :data:`UNCLASSIFIED`. See :func:`_exact_apply`
    for the table and for why it is small, and :func:`_exact_convert` for why
    ``out_dtype`` is not optional in spirit even though it defaults: a caller
    that omits it gets a REFUSAL from the conversion rather than a guess.
    """
    return _exact_apply(primitive, [_Exact.of(o) for o in operands], idx,
                        n_out, out_dtype)


def exact_walk(jax_closed, points):
    """``(i, j) -> _Exact`` — every equation's output, over ℝ, at this point.

    The same reading :func:`exact_value` makes of ONE equation, chained down
    the whole jaxpr, and it exists for exactly one question:
    :data:`NARROWING`'s third fact — *does the sampled point satisfy the
    harness's ``assume``s OVER THE REALS?* (:func:`assumes_hold_over_reals`).

    A refusal PROPAGATES. An equation whose primitive has no entry in the
    table, or whose operands are not all exactly known, has no exact reading —
    and neither has anything computed from it. That is what makes the answer
    usable as a premise: where this walk answers at all, it answers exactly.

    The three stelling primitives are the identity on their operand, which is
    what their own ``def_impl`` says and what :func:`execute` does with them.
    ``stelling_any`` takes the sampled array, which is a float and therefore
    its own exact rational.
    """
    import jax.extend.core as jex

    env = {}
    for var, val in zip(jax_closed.jaxpr.constvars, jax_closed.consts):
        env[var] = _Exact.of(val)
    out = {}
    k = 0
    for i, eqn in enumerate(jax_closed.jaxpr.eqns):
        name = eqn.primitive.name
        if name == "stelling_any":
            res = [_Exact.of(points[k])]
            k += 1
        else:
            ops = [_Exact.of(a.val) if isinstance(a, jex.Literal)
                   else env.get(a, _Exact((None,)))
                   for a in eqn.invars]
            if name in ("stelling_assume", "stelling_assert",
                        "stelling_nonvacuity"):
                res = [ops[0]]
            elif len(eqn.outvars) != 1:
                # A multi-output primitive has no per-output entry in the
                # table, so there is nothing to align an answer to. Refuse.
                res = [_Exact((None,)) for _ in eqn.outvars]
            else:
                size = 1
                for d in eqn.outvars[0].aval.shape:
                    size *= int(d)
                out_dtype = str(eqn.outvars[0].aval.dtype)
                res = [_Exact(
                    (_exact_apply(name, ops, e, size, out_dtype)
                     for e in range(size)),
                    out_dtype,
                )]
        for j, (ov, val) in enumerate(zip(eqn.outvars, res)):
            env[ov] = val
            out[(i, j)] = val
    return out


def assumes_hold_over_reals(jax_closed, points):
    """``True``, ``False`` or ``None`` for the harness's ``assume``s at a point.

    ``True``  — every ``assume`` is EXACTLY true over ℝ at this point;
    ``False`` — at least one is EXACTLY false, so the point is not in the
                assumed set over ℝ however the compiled program answered;
    ``None``  — a refusal: no assume is provably false and at least one has no
                exact reading (a connective, ``jnp.all``, a ``sqrt``).

    THE FACT :data:`NARROWING` NEEDED AND DID NOT HAVE. Comparing the narrowed
    box against the un-narrowed one decides only that the violation was
    CREATED by the narrowing; it cannot tell *"the narrowing is right over ℝ
    and the compiled comparison admitted a point ℝ excludes"* — the disclosed
    posture, and what ``assume-narrows-past-the-program`` is — from *"the
    narrowing is wrong over ℝ"*, which is a soundness defect in ``propagate``.
    The two differ in exactly this predicate, and the first is ``False`` here.

    A harness with no ``assume`` answers ``True`` vacuously, which is the
    right answer and is never asked: a violation that vanishes against the
    inert boxes requires a narrowing, and a narrowing requires an assume.
    """
    walk = exact_walk(jax_closed, points)
    unknown = False
    for i, eqn in enumerate(jax_closed.jaxpr.eqns):
        if eqn.primitive.name != "stelling_assume":
            continue
        vals = walk.get((i, 0))
        whole = None if vals is None else vals.whole()
        if whole is None:
            unknown = True
            continue
        if any(w == 0 for w in whole):
            return False
    return None if unknown else True


def _exact_in_box(exact, lo: float, hi: float) -> bool:
    """Is the exact real ``exact`` inside ``[lo, hi]``? Infinities included.

    ``Fraction(math.inf)`` RAISES ``OverflowError``, and an infinite box
    endpoint is ordinary here — ``any_array((), "float32", (1.0, math.inf))``
    is a declaration this grammar draws and stelling accepts. Comparing
    against an infinity is a comparison in the extended reals and needs no
    rational at all, so the infinite side is answered structurally and only
    the finite side is converted.
    """
    if math.isinf(lo):
        if lo > 0:
            return False
    elif exact < Fraction(lo):
        return False
    if math.isinf(hi):
        if hi < 0:
            return False
    elif exact > Fraction(hi):
        return False
    return True


def _rounds_to(T, exact, v: float) -> bool:
    """Does ``exact`` round into ``T`` as exactly ``v``? ``False`` if it cannot
    be asked — a Fraction too large for a float raises rather than answering."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(T(float(exact))) == v
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        return False


def _subnormal_elements(operands):
    """``[(operand position, format, value)]`` for every nonzero subnormal."""
    out = []
    for pos, operand in enumerate(operands):
        if operand is None:
            continue
        arr = np.asarray(operand)
        name = str(arr.dtype)
        if name not in FLOAT_TYPES:
            continue
        mn = MIN_NORMAL[name]
        for w in arr.reshape(-1):
            w = float(w)
            if 0.0 < abs(w) <= mn:
                out.append((pos, name, w))
    return out


def _flushed(operands):
    """``operands`` with every nonzero subnormal replaced by a signed zero."""
    out = []
    for operand in operands:
        if operand is None:
            out.append(operand)
            continue
        arr = np.asarray(operand)
        name = str(arr.dtype)
        if name not in FLOAT_TYPES:
            out.append(operand)
            continue
        mn = MIN_NORMAL[name]
        flat = np.array(arr, dtype=np.float64).reshape(-1)
        for k, w in enumerate(flat):
            if 0.0 < abs(float(w)) <= mn:
                flat[k] = math.copysign(0.0, float(w))
        out.append(flat.reshape(arr.shape))
    return tuple(out)


def _operand_flush_explains(primitive, operands, idx, n_out, v, lo, hi,
                            out_dtype="") -> bool:
    """Does flushing the subnormal OPERANDS to zero explain ``v``?

    TWO CONDITIONS, AND THE FIRST ONE USED TO BE MISSING. This rule read
    *"some operand element is a nonzero subnormal"* and nothing else, so an
    operand-side fact licensed an UNBOUNDED output-side error: on a copy of
    this tree with ``interval.add`` shifted by +1.0, a program whose addend was
    a subnormal literal reported ``executed -1.0, box [0.0, 2.0]`` as
    ``flush-or-subnormal`` while the SAME injected defect with the addend
    ``1.0`` reported ``UNEXPLAINED``. Driven straight at the classifier on the
    clean tree, ``classify("add", "float64", v=1e300, box=[0.0, 1.0],
    operands=(5e-324, 1e300))`` answered ``flush-or-subnormal``. **21 of the
    384 violations this module shipped were classified by that rule.**

    So the rule now asks two things and can prove both:

    * the EXACT REAL value of this primitive on its UNFLUSHED operands is
      INSIDE the box — the box is right about ℝ, exactly as the reassociation
      rule requires;
    * flushing those subnormal elements to a signed zero REPRODUCES the
      executed value — the flush is what moved it out, rather than something
      else that happened to have a subnormal nearby.

    Either one unanswerable makes this ``False``, and the violation falls
    through to a rule that can answer or to :data:`UNCLASSIFIED`.
    """
    if not _subnormal_elements(operands):
        return False
    exact = exact_value(primitive, operands, idx, n_out, out_dtype)
    if exact is None or not _exact_in_box(exact, lo, hi):
        return False
    flushed = exact_value(primitive, _flushed(operands), idx, n_out, out_dtype)
    if flushed is None:
        return False
    return float(flushed) == v


def _same_side_of_zero(v: float, lo: float, hi: float) -> bool:
    """A single rounding cannot cross zero, and the window must not either."""
    if lo > 0.0 and v <= 0.0:
        return False
    if hi < 0.0 and v >= 0.0:
        return False
    return True


def _exact_of_equation(primitive: str, inputs, idx: int, n_out: int, v: float,
                       out_dtype: str = ""):
    """:func:`exact_value`, plus the one equation whose operands cannot say it.

    A ``stelling_any`` COMPUTES NOTHING: its value is the sampled point, and
    :func:`sample_points` asserts that every point it returns is a member of
    the declared set (:func:`_assert_points_of`). So the exact real value of
    that equation is the float that ran, exactly; it needs no table entry, and
    it has no operands to read — ``execute`` records ``()`` for it, which is
    why :func:`exact_value` alone answers ``None`` here.

    That matters for one specific proof, and it is what :data:`NARROWING`'s
    third fact is FOR. When an ``assume`` narrows a DECLARATION's box past a
    point that satisfies every assume over ℝ, there is no arithmetic anywhere
    in the finding — the box excludes a member of its own assumed set — and
    without this reading the violation has no exact value at all and falls out
    as a heuristic rather than as the proof it is.

    IT ALSO WIDENS WHAT ``UNEXPLAINED`` CAN PROVE, and that is deliberate: a
    ``stelling_any`` violation in a harness with NO narrowing is now
    ``UNEXPLAINED`` too, because the sampled point is asserted to be a member
    of the declared set and the propagator's box for that very declaration
    excludes it. That is a box wrong about its own declared set, which is a
    stronger finding than the sampler-artefact counter beside it and is
    measured at 0 on this tree.

    :func:`exact_walk` says the same thing in its own way, reading the point
    out of ``points`` because it has them and this does not. One fact, two
    callers, and neither can borrow the other's spelling.
    """
    if primitive == "stelling_any":
        return Fraction(v) if math.isfinite(v) else None
    return exact_value(primitive, inputs, idx, n_out, out_dtype)


def _left_to_right_fold(operand, dtype: str):
    """``sum(operand)`` folded IN ORDER in ``dtype``, or ``None``.

    The one association order a reader can name without asking the compiler,
    and therefore the only thing that makes ``reduction-reassociation`` a
    PROOF rather than an attribution read off the primitive's identity.
    ``None`` where the format is not one this module can accumulate in, which
    is a refusal and drops the rule to ``heuristic``.
    """
    T = FLOAT_TYPES.get(dtype)
    if T is None or operand is None:
        return None
    arr = np.asarray(operand)
    if str(arr.dtype) not in FLOAT_TYPES:
        return None
    with np.errstate(all="ignore"):
        total = T(0.0)
        for w in arr.reshape(-1):
            total = T(total + T(w))
    return float(total)


def classify(primitive: str, dtype: str, v: float, lo: float, hi: float,
             inputs, element: int = 0, n_out: int = 1,
             inert: tuple | None = None, assumes: bool | None = None) -> tuple:
    """Why is ``v`` outside ``[lo, hi]``? ``(cause, basis)``.

    ────────────────────────────────────────────────────────────────────────
    A PROVABLE ``UNEXPLAINED`` WAS BEING PREEMPTED BY AN UNPROVED RULE, AND
    THAT IS THE THIRD TIME THE ORDER HAS BEEN WRONG
    ────────────────────────────────────────────────────────────────────────

    :func:`_operand_flush_explains` was hardened to prove two things, exactly
    because an operand-side fact must not license an unbounded output-side
    error. **The physical-band rule sitting immediately above it was left
    unguarded** — box reaches the format's subnormal band, ``|v| <= mn``,
    return FLUSH, no ℝ check at all — and it ran BEFORE the final
    ``_exact_in_box`` test that would have proved the box wrong. Driven with
    this project's own registered ``float-oracle-unexplained`` mutation, on
    the control's own harness and one envelope over:

        x0 in [-1.0, 1.0]      mul executed 0.0 box [1.0, 1.0]     UNEXPLAINED
        x0 in [-1e-160, 1.0]   mul executed 0.0 box [1e-320, 1.0]  flush
        x0 in [-1.0, 1e-160]   mul executed 0.0 box [1e-320, 1.0]  flush

    ``exact_value("mul", (0.0, 0.0))`` is ``0``, which is outside
    ``[1e-320, 1.0]``: the proof was in hand and an earlier rule took the
    answer. **161 of 563 violations went through that rule.** So every rule
    that CAN be proved is now tried before ``UNEXPLAINED``, every rule that
    cannot is tried after it, and each answer carries which it was.

    ────────────────────────────────────────────────────────────────────────
    AND THE REORDER LEFT THREE RULES ABOVE THE EXACT READING THAT ASK
    NOTHING ABOUT ℝ. ALL THREE ARE BELOW IT NOW
    ────────────────────────────────────────────────────────────────────────

    The rule was *"every provable rule above ``UNEXPLAINED``, every unprovable
    one below"*, and the first pass sorted by which rules had been CAUGHT
    rather than by which ones could prove anything. An independent audit drove
    each of the three:

    * **``overflow-to-inf``** was ``sound-by-construction`` and it is not.
      What is sound by construction is *"no finite box contains an
      infinity"* — the violation. The ATTRIBUTION, that an overflow of a box
      the propagator got RIGHT is what produced the infinity, needs the same
      ℝ reading every other cause needs. Driven, on a tree where
      ``interval._mul_corners``' upper endpoint is halved above ``1e50``, a
      box provably wrong about ℝ::

          exact_value("mul", (1e300, 1e300))  ~ 1e600
          _exact_in_box(exact, 0.0, 1.0)      False
          classify("mul", "float64", inf, 0.0, 1.0, (1e300, 1e300))
             -> ('overflow-to-inf', 'sound-by-construction')

      — and end to end, ``OVERFLOW_PROBE`` reported ``overflow-to-inf /
      sound-by-construction`` with its exact value ``8.99e76`` OUTSIDE the
      box, ``Any UNEXPLAINED? False``, and the residual leg ``1 passed``. It
      is now ``PROVED`` inside the ``in_box`` block and ``HEURISTIC`` below
      ``UNEXPLAINED``;
    * **``assume-narrowing``** was the first act of this function and decided
      on a comparison of two BOXES, which cannot tell *"the narrowing is right
      over ℝ and the floats disagree"* — ``assume-narrows-past-the-program``,
      the disclosed posture — from *"the narrowing is wrong over ℝ"*, which is
      a soundness defect in ``propagate``. It needs a third fact, and
      :func:`assumes_hold_over_reals` is it: the cause is PROVED only where an
      ``assume`` is EXACTLY FALSE at the sampled point over ℝ, so the compiled
      program admitted a point the reals exclude. Where every assume is
      exactly TRUE there and the narrowing dropped the point anyway, this
      returns nothing and the ladder below proves ``UNEXPLAINED``. Driven on a
      tree carrying the documented ieee-only endpoint bump in real mode,
      ``x0: float64 (0,1); assume(x0 >= 0.0); assert_(x0 > 0.0)`` was VERIFIED
      with ``cause=assume-narrowing basis=sound-by-construction`` and the whole
      module ``1 passed, 1 xfailed``;
    * **the operand NaN** scanned every element of every operand, so a NaN
      anywhere answered for an element whose own exact reading is computable
      and outside its box. It is asked of the operand elements the output
      element under judgement DEPENDS ON now (:func:`_an_operand_is_nan`), and
      it sits below ``UNEXPLAINED``. That is safe rather than lucky: an
      aligned operand element that is a NaN makes :func:`exact_value` refuse,
      so ``UNEXPLAINED`` — which requires an exact reading — cannot fire on
      the cases this rule answers.

    ``reduction-reassociation`` was the fourth: PROVED from the primitive's
    identity alone. :func:`_left_to_right_fold` is the missing half.

    THE ORDER, AND WHAT EACH STEP EXCLUDES:

    1. a NaN — sound by construction: no box contains one, ⊤ included;
    2. the violation does not exist against the UN-NARROWED box AND an
       ``assume`` is exactly false at this point over ℝ (:data:`NARROWING`,
       proved);
    3. every other PROVED rule, each of which requires an exact rational
       reading of this equation on the values it ran on to be INSIDE the box —
       the box is right about ℝ and an IEEE difference moved the float out:
       the overflow, the physical band, the narrow format, the operand flush,
       the reduction;
    4. :data:`UNEXPLAINED` — an exact reading exists and is OUTSIDE the box.
       The box is wrong about the reals. This is the only answer the residual
       leg forbids, and it is a proof;
    5. the HEURISTIC rules, reachable only where no exact reading exists: an
       undecided narrowing, an operand NaN, the overflow, the physical band
       and the one-ulp window. Each names a plausible cause and proves
       nothing, and they are counted as their own row.
    """
    if v != v:
        # No box contains a NaN, ⊤ included. The one cause that survives a
        # declined equation, and the reason the ⊤ count does not bound this
        # instrument's whole reach. FIRST, and above the narrowing: an
        # `ilo <= nan <= ihi` test is False for every box, so no inert reading
        # could answer this one anyway.
        return NAN, STRUCTURAL

    if inert is not None and inert[0] <= v <= inert[1]:
        # THE VIOLATION IS CREATED BY THE NARROWING, AND WHICH FINDING THAT IS
        # DEPENDS ON A THIRD FACT. Against the DECLARED box this value is
        # inside; it is outside only the box the assume tightened. Two
        # completely different things do that, and the box comparison alone
        # cannot tell them apart:
        if assumes is False:
            # the harness's own precondition is EXACTLY FALSE at this point
            # over ℝ, so the analysis was RIGHT to drop it and the compiled
            # program admitted it anyway. The disclosed posture, and what
            # `assume-narrows-past-the-program` is. Proved, not asserted.
            return NARROWING, PROVED
        if assumes is None:
            # No assume in this harness is provably false here and at least
            # one has no exact reading — a connective, `jnp.all`, a `sqrt`.
            # PLAUSIBLE AND UNPROVED, and it returns HERE rather than falling
            # through on purpose: every rule below reads the NARROWED box, so
            # an `UNEXPLAINED` derived from it would be a proof about a
            # narrowing this function has just failed to justify. A narrowing
            # that is WRONG about ℝ lands in this row too. It is 0 on this
            # tree and it is the honest place for it to be.
            return NARROWING, HEURISTIC
        # `assumes is True`: the point satisfies every assume over ℝ and the
        # narrowing dropped it anyway — so the narrowed box is wrong about the
        # reals, and the ladder below proves it. Fall through.

    exact = _exact_of_equation(primitive, inputs, element, n_out, v, dtype)
    in_box = exact is not None and _exact_in_box(exact, lo, hi)
    mn = MIN_NORMAL.get(dtype)
    T = FLOAT_TYPES.get(dtype)

    if in_box:
        if math.isinf(v) and math.isfinite(lo) and math.isfinite(hi):
            # PROVED overflow: the box is right about ℝ — the exact real value
            # of this equation is inside it — and the program's arithmetic
            # left the format's range anyway. FIRST inside this block, because
            # every rule below it would also fire on an infinity and the
            # infinity is the more specific answer.
            return OVERFLOW, PROVED
        if mn is not None and abs(v) <= mn and abs(float_or_inf(exact)) <= mn:
            # PROVED underflow: the box is right about ℝ, the true value is
            # itself inside the format's subnormal band, and the executed
            # float is too. Flush-to-zero, or gradual underflow the ℝ box does
            # not model.
            return FLUSH, PROVED
        if T is not None and dtype != "float64" and _rounds_to(T, exact, v):
            # PROVED narrow-format rounding: the box is right about ℝ and
            # rounding that real into the program's own format is exactly
            # what ran.
            return NARROW, PROVED
        if _operand_flush_explains(primitive, inputs, element, n_out, v,
                                   lo, hi, dtype):
            # PROVED operand flush, and the only way a COMPARISON's violation
            # can be explained at all: a ``bool`` output has no format and no
            # subnormal band of its own.
            return FLUSH, PROVED
        if primitive == "reduce_sum" and len(inputs) == 1:
            # THE ATTRIBUTION, PROVED, AND IT USED TO BE READ OFF THE
            # PRIMITIVE'S NAME. The box being right about ℝ and the executed
            # float not being the real sum says a reduction rounded; it does
            # NOT say the compiler reassociated. Driven at the classifier:
            # `classify("reduce_sum", "float64", 0.0, [1.0, 1e100, -1e100])`
            # answered `('reduction-reassociation', 'proved')`, while a plain
            # left-to-right float64 fold of that vector is 0.0 as well — the
            # naive order explains the value and no reassociation is needed.
            # 46 events and 30 programs sat in the "303 proved" row on that
            # reading. So the rule asks for the order it can name:
            folded = _left_to_right_fold(inputs[0], dtype)
            if folded is None:
                # No format to accumulate in. Plausible, unproved, censused.
                return REASSOCIATION, HEURISTIC
            if folded != v:  # `v` is not a NaN: that is answered at the top
                # PROVED: the box is right about the real value of this
                # reduction, and the executed float is not what summing the
                # operand IN ORDER produces either — so some other association
                # order ran, which is the one the interval domain does not
                # model.
                return REASSOCIATION, PROVED
            # The in-order fold reproduces the executed value exactly. The box
            # is right about ℝ and the miss is the ordinary accumulated
            # rounding of a reduction, which is none of the five named causes.
            # A refusal, censused — this is the second shape of UNCLASSIFIED
            # that :data:`CAUSES` describes, and it is 0 on this tree because
            # every reassociation violation found so far is an n >= 33 draw.
            return UNCLASSIFIED, HEURISTIC

    if exact is not None and not in_box:
        # PROVED: the exact real value of this equation on the values it ran
        # on is OUTSIDE the box the propagator computed for it, and no IEEE
        # difference accounts for the miss. The box is wrong about the REALS.
        return UNEXPLAINED, PROVED

    # ── below here nothing is proved, and the count of it is a disclosure ──
    if _an_operand_is_nan(primitive, inputs, element, n_out):
        # A comparison against a NaN is a bool, not a NaN, so it would
        # otherwise arrive below unexplained. Sound by construction for the
        # same reason the output-NaN rule is: the operand this element depends
        # on is not a real number, so no box computed over ℝ says anything
        # about it.
        return NAN, STRUCTURAL
    if math.isinf(v) and math.isfinite(lo) and math.isfinite(hi):
        # An infinity the analysis said could not happen, with no exact
        # reading to say whether the box was right about ℝ in the first place.
        # PLAUSIBLE AND UNPROVED. Where the box is already unbounded on that
        # side there is no violation to classify.
        return OVERFLOW, HEURISTIC
    if mn is not None and min(abs(lo), abs(hi)) <= mn and abs(v) <= mn:
        # The box comes into the format's subnormal band and the executed
        # value is in it too. PLAUSIBLE AND UNPROVED: with no exact reading of
        # this primitive (``exp``, ``sqrt``) there is nothing here that rules
        # out a box which is simply wrong.
        return FLUSH, HEURISTIC
    if T is not None and dtype != "float64" and (
        _prev(T, lo) <= v <= _next(T, hi) and _same_side_of_zero(v, lo, hi)
    ):
        # One ulp of the program's own format either side of a box computed in
        # binary64. PLAUSIBLE AND UNPROVED, and guarded so the window cannot
        # cross zero — a single rounding cannot turn a strictly-signed real
        # into a zero of the other sign, and the case where it CAN produce a
        # zero is underflow, which the rule above takes.
        return NARROW, HEURISTIC
    # No exact reading exists for this primitive and no rule named it. A
    # refusal, censused.
    return UNCLASSIFIED, HEURISTIC


def float_or_inf(exact) -> float:
    """``float(exact)``, saturating instead of raising on a huge Fraction."""
    try:
        return float(exact)
    except OverflowError:  # pragma: no cover - defensive
        return math.inf if exact > 0 else -math.inf


def _an_operand_is_nan(primitive: str, inputs, idx: int, n_out: int) -> bool:
    """Is one of the operand elements THIS OUTPUT ELEMENT DEPENDS ON a NaN?

    IT ASKED OF THE WHOLE OPERAND AND THAT IS A DIFFERENT QUESTION. The rule
    read *"a NaN is present anywhere in any operand"*, so element 0 of a
    two-element ``add`` whose element 1 is a NaN answered ``NAN`` — for a
    violation whose own exact reading is computable and outside its box, i.e.
    for a box that is wrong about ℝ. 0 of the 563 violations in the shipped
    run took that path, so it was latent rather than live; a latent
    misattribution in the rule that stands between a violation and
    ``UNEXPLAINED`` is not one to leave.

    THE ALIGNMENT IS :meth:`_Exact.element`'S, plus reductions. An operand of
    size 1 broadcasts, an operand of size ``n_out`` is element-wise, and every
    element of a full reduction's operand reaches its single output. Any other
    shape is a REFUSAL: this reader does not guess which elements fed which,
    and a NaN it cannot align contributes nothing rather than answering for a
    neighbour. That refusal costs evidence and never soundness — the caller's
    next step is ``UNCLASSIFIED``, which is censused.
    """
    reducing = primitive.startswith("reduce_")
    for operand in inputs:
        if operand is None:
            continue
        arr = np.asarray(operand)
        if str(arr.dtype) not in FLOAT_TYPES:
            continue
        flat = arr.reshape(-1)
        if reducing:
            here = flat
        elif flat.size == 1:
            here = flat[:1]
        elif flat.size == n_out and idx < flat.size:
            here = flat[idx:idx + 1]
        else:
            continue
        if any(float(w) != float(w) for w in here):
            return True
    return False


# ── one reading ──────────────────────────────────────────────────────────────


@dataclass
class Reading:
    status: str = "read"
    violations: list = field(default_factory=list)
    boxes_read: int = 0
    top_boxes: int = 0
    empty_boxes: int = 0
    integer_boxes: int = 0
    compared_boxes: int = 0
    points: int = 0
    admitted_points: int = 0
    falsified_discharges: int = 0
    contradicted_refutations: int = 0
    route_declined: int = 0
    route_unavailable: int = 0
    route_obligations_compared: int = 0
    route_obligations_disagreed: int = 0
    sampler_artefacts: int = 0
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

    anc = ancestors(jax_closed)
    inert = _InertOnce(closed, refusals)
    staged = _StagedOnce(jax_closed)

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
        _compare(reading, boxes, values, operands, points,
                 reading.admitted_points == 1, jax_closed, anc, inert,
                 staged, refusals)
    if reading.admitted_points == 0 and reading.status == "read":
        reading.status = "no-admitted-point"
    return reading


class _InertOnce:
    """The un-narrowed boxes, computed at most once and only if needed.

    A second forward walk is the price of two questions the constrained boxes
    cannot answer — whether a violation exists at all against the DECLARED box
    (:data:`NARROWING`), and whether an ``assume`` narrowed the declaration a
    ``stelling_any`` violation sits on — and it is paid only where the first
    route found something to ask them about.
    """

    def __init__(self, closed, refusals):
        self._closed = closed
        self._refusals = refusals
        self._made = False
        self._boxes = None

    def boxes(self):
        if not self._made:
            self._made = True
            try:
                self._boxes = {
                    (i, j): box
                    for i, _prim, j, box in boxes_of(self._closed, "inert")
                }
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ not in self._refusals:
                    raise
                self._boxes = None
        return self._boxes


class _StagedOnce:
    """:func:`staged_runner`, built at most once and only if it is needed.

    Compiling is the expensive half of this module, so the second route is
    paid for only where the first found something to check.
    """

    def __init__(self, jax_closed):
        self._jax_closed = jax_closed
        self._made = None

    def values(self, points):
        if self._made is None:
            self._made = staged_runner(self._jax_closed)
        jitted, keys = self._made
        with np.errstate(all="ignore"):
            got = jitted(*points)
        return dict(zip(keys, got))


def _violating_elements(boxes, values, reading, count_boxes):
    """Every ``(eqn, primitive, dtype, element, v, lo, hi)`` outside its box.

    Buckets the boxes on the way past, in FOUR buckets and not three.
    ``finite_boxes`` used to be incremented and then ``continue``d on an
    integer dtype, so 1327 of the 8708 boxes it reported as falsifiable were
    never compared at all and the "76 % falsifiable" sentence was 64 %.
    """
    out = []
    for i, primitive, j, box in boxes:
        v = values.get((i, j))
        if v is None:  # pragma: no cover - every outvar is bound above
            continue
        arr = np.asarray(v)
        dtype = str(arr.dtype)
        comparable = dtype in FLOAT_TYPES or dtype == "bool"
        if count_boxes:
            reading.boxes_read += 1
            if box.size == 0:
                reading.empty_boxes += 1
            elif is_top(box, dtype):
                reading.top_boxes += 1
            elif dtype in INT_DTYPES:
                # A box endpoint is a binary64 float and an int64 above 2**53
                # has no exact binary64 image, so a comparison here would
                # report the comparison's own rounding as a violation.
                reading.integer_boxes += 1
            elif comparable:
                reading.compared_boxes += 1
            else:
                reading.integer_boxes += 1
        if dtype in INT_DTYPES or not comparable or box.size == 0:
            continue
        flat = arr.astype(np.float64).reshape(-1)
        if flat.size != box.size:  # pragma: no cover - shapes agree by trace
            continue
        for idx in range(flat.size):
            x = float(flat[idx])
            lo, hi = box.los[idx], box.his[idx]
            if lo <= x <= hi:
                continue
            out.append((i, j, primitive, dtype, idx, x, lo, hi, flat.size))
    return out


def _narrowed_here(inert, lo: float, hi: float) -> bool:
    """Did an assume tighten THIS element's box? ``None`` means "not read"."""
    if inert is None:
        return False
    return (inert[0], inert[1]) != (lo, hi)


def _compare(reading, boxes, values, operands, points, count_boxes, jax_closed,
             anc, inert, staged, refusals):
    """Compare every box against its executed value and record what survives.

    THREE CUTS, and each one is a place this instrument was wrong:

    * violations at an equation that DEPENDS on another violating equation are
      dropped, because once a value is outside its box nothing computed from
      it is guaranteed by anything. The cut is by data dependence
      (:func:`ancestors`) and was by equation INDEX, which dropped siblings;
    * every survivor is re-checked against the SECOND ROUTE — the same jaxpr
      as one compiled region — and one the compiled program does not have is
      declined, not reported. See :func:`staged_runner`;
    * a survivor on ``stelling_any`` in a harness with NO assume can only be
      this module's own sampler rounding, and is counted as such.
    """
    point = tuple(float(np.asarray(p).reshape(-1)[0]) if np.asarray(p).size
                  else None for p in points)
    compiled = None
    inert_boxes = None
    candidates = _violating_elements(boxes, values, reading, count_boxes)
    if candidates:
        inert_boxes = inert.boxes()
    if candidates:
        violating_eqns = {c[0] for c in candidates}
        candidates = [c for c in candidates
                      if not (anc.get(c[0], set()) & violating_eqns)]
        # The topologically earliest violating equation has no violating
        # ancestor, so this cut can never empty a non-empty list — measured,
        # 0 of 1511 programs. Every path below that needs the compiled route
        # therefore gets it whenever any violation exists.
    if candidates:
        try:
            compiled = staged.values(points)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ in refusals:
                compiled = None
            else:
                raise
        if compiled is not None:
            # THE STRONGEST FIGURE THIS MODULE REPORTS IS THE FALSIFIED
            # DISCHARGE, AND IT IS READ FROM THE EAGER ROUTE. So wherever the
            # second route runs at all, the two routes' OBLIGATION TRUTH
            # VALUES are compared and any disagreement is counted — which
            # turns "the discharge counts are route-independent" from a
            # sentence into a number the census carries on every run.
            eager_truths = _obligation_truths(boxes, values)
            compiled_truths = _obligation_truths(boxes, compiled)
            reading.route_obligations_compared += len(eager_truths)
            reading.route_obligations_disagreed += sum(
                1 for a, b in zip(eager_truths, compiled_truths) if a != b
            )
            compiled_ops = operands_of(jax_closed, compiled)
            kept = []
            for (i, j, prim, dtype, idx, _x, lo, hi, n_out) in candidates:
                # `(i, j)`, NOT `(i, 0)`. The eager route is j-correct and
                # this one was not; no equation in 1511 programs had more than
                # one outvar, so it was latent rather than live, and a latent
                # index bug in the route that DECLINES violations is not one
                # to leave for the primitive that grows a second output.
                cv = compiled.get((i, j))
                if cv is None:  # pragma: no cover - every outvar is bound
                    continue
                cflat = np.asarray(cv).astype(np.float64).reshape(-1)
                if idx >= cflat.size:  # pragma: no cover
                    continue
                x = float(cflat[idx])
                if lo <= x <= hi:
                    # THE COMPILED PROGRAM DOES NOT HAVE THIS VIOLATION.
                    reading.route_declined += 1
                    continue
                kept.append((i, j, prim, dtype, idx, x, lo, hi, n_out,
                             compiled_ops.get(i, ())))
            candidates = kept
        else:  # pragma: no cover - a refusal inside the compiled route
            reading.route_unavailable += 1
            candidates = [c + (operands.get(c[0], ()),) for c in candidates]
    # THE EXACT READING OF THE HARNESS'S ASSUMES, AT MOST ONCE PER POINT AND
    # ONLY WHERE A NARROWING IS ACTUALLY IN QUESTION. It is a whole-jaxpr walk
    # in rationals, so it is paid for on the same terms as the inert boxes and
    # the compiled route: after a violation, and only for the violations that
    # need it. :func:`classify` reads it as :data:`NARROWING`'s third fact.
    memo: list = []

    def assumes_over_reals():
        if not memo:
            memo.append(assumes_hold_over_reals(jax_closed, points))
        return memo[0]

    for (i, j, prim, dtype, idx, x, lo, hi, n_out, ins) in candidates:
        inert = None
        if inert_boxes is not None:
            ibox = inert_boxes.get((i, j))
            if ibox is not None and idx < ibox.size:
                inert = (ibox.los[idx], ibox.his[idx])
        if prim == "stelling_any" and not _narrowed_here(inert, lo, hi):
            # THE GUARD TESTS WHETHER THIS DECLARATION WAS NARROWED, NOT
            # WHETHER THE HARNESS SAYS `assume` ANYWHERE. It read
            # `has_assume`, which is well-formedness standing in for truth —
            # the shape this project keeps re-finding. Driven on the
            # snap-deleted copy: with no assume, 2 artefacts and the leg
            # FAILS; with a VACUOUS `assume(x0 >= -1e300)`, 0 artefacts and
            # the leg is GREEN, and 172 of 1206 read programs were exempt on
            # that reading. Comparing the inert and constrained boxes for
            # THIS declaration answers the question that was being asked.
            #
            # IT IS NO LONGER THE ONLY GUARD, AND IT WAS NEVER THE RIGHT
            # PLACE. This one can only fire on a VIOLATION at `stelling_any`,
            # and an unbounded declaration's own box is ⊤ — so ±inf was inside
            # it, there was no violation to fire on, and this counter was
            # structurally blind to the very class it had just been extended
            # for. `sample_points` asserts membership where the point is BUILT
            # now (`_assert_points_of`), which is the check that can see it.
            # What survives here is the OTHER half: a declaration whose box
            # the propagator computed differently from the set the harness
            # declared. Same-seed A/B at STELLING_PROPERTY_SCALE=12.5 with the
            # ±inf clamp removed AND the membership assertion removed (without
            # which the run cannot finish): `nan` 162 -> 164 events and 30 ->
            # 35 distinct uniform programs, every other cause identical, and
            # **`sampler_artefacts` 0 in BOTH**.
            reading.sampler_artefacts += 1
        narrowed_away = inert is not None and inert[0] <= x <= inert[1]
        cause, basis = classify(
            prim, dtype, x, lo, hi, ins, idx, n_out, inert,
            assumes_over_reals() if narrowed_away else None,
        )
        reading.violations.append(
            Violation(
                eqn=i, primitive=prim, dtype=dtype, element=idx,
                executed=x, lo=lo, hi=hi, cause=cause, basis=basis,
                point=point,
            )
        )
    # READ FROM THE COMPILED ROUTE WHEREVER IT RAN, because the claim these
    # two numbers make is about the program the user COMPILES, and that is the
    # whole of the reason. **THE SENTENCE HERE USED TO REST ON A NUMBER AND
    # THE NUMBER IS NOW ZERO.** It read *"the two routes DO disagree here, and
    # the disagreement is not rare on the points that matter: measured over
    # 1500 examples, 17 of the 537 obligation readings taken at a violating
    # point differ between the eager walk and one compiled region … it is why
    # this cannot be left on the eager reading."* It was CORRECT at `03b2dbe`
    # and is re-derived there — `git archive 03b2dbe src tests` into a scratch
    # tree, same command: 537 compared, 17 disagreeing — but that is before
    # the reorder and before the sampler stopped executing at ±inf, and the
    # census on the same command at this tree reports
    # `route_obligations_compared` and `route_obligations_disagreed` and the
    # second of them is 0. The DESIGN choice is unchanged and does not need
    # the number — an obligation truth read from a route the user does not run
    # is the wrong reading whether or not the two happen to agree today — and
    # the census carries both figures on every run, so the day they diverge
    # again nobody has to remember this paragraph. Where no violation was
    # found the two agree by construction: a falsified discharge IS a
    # violation at the obligation's own equation, so the compiled route has
    # run wherever one exists.
    d, r = _contradicted_obligations(
        boxes, values if compiled is None else compiled
    )
    reading.falsified_discharges += d
    reading.contradicted_refutations += r


#: The two boxes the propagator reads as definite. Mirrors
#: ``propagate._bool_status``'s ``n_true == n`` / ``n_false`` tests, written
#: out rather than imported because they are two pairs of floats and importing
#: a private helper to compare four floats couples this file to a name it does
#: not need.
_BOOL_TRUE = (1.0, 1.0)
_BOOL_FALSE = (0.0, 0.0)


def _obligation_truths(boxes, values):
    """``True``/``False`` per ``stelling_assert``, in walk order.

    The quantity :func:`_contradicted_obligations` reads. Extracted so the two
    execution routes can be compared on it directly.
    """
    out = []
    for i, primitive, j, _box in boxes:
        if primitive != "stelling_assert":
            continue
        v = values.get((i, j))
        out.append(None if v is None else bool(np.all(np.asarray(v))))
    return out


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

        ELEMENT-ALIGNED, AND THE REFUTATION HALF WAS NOT. It asked
    ``any(pair == BOOL_FALSE)`` of the box and then ``np.any(v)`` of the
    value, which are questions about DIFFERENT elements: a box whose element 0
    is definitely-false beside a value whose element 1 is true fired the
    counter while element 0 agreed with the refutation perfectly. Driven with
    this module's own ``aconst`` node, every element inside its own box and
    ``np.all`` False: the counter fired anyway. Bounded before the fix — 476
    multi-element assert boxes in the shipped run, **0 with a mixed element
    pair**, and an element-aligned recount reproduced 95/13 exactly — so it
    was latent, not live. It is asked per element now.
    """
    discharged = refuted = 0
    for i, primitive, j, box in boxes:
        if primitive != "stelling_assert":
            continue
        v = values.get((i, j))
        if v is None:  # pragma: no cover
            continue
        flat = np.asarray(v).reshape(-1)
        pairs = list(zip(box.los, box.his))
        if len(pairs) != flat.size:  # pragma: no cover - shapes agree
            continue
        if pairs and all(pair == _BOOL_TRUE for pair in pairs):
            if not bool(np.all(flat)):
                discharged += 1
        elif any(
            pair == _BOOL_FALSE and bool(flat[k])
            for k, pair in enumerate(pairs)
        ):
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
# SEVEN OF THESE ARE THE SEVEN THIS MODULE WAS BUILT TO PIN. The other two
# were found BY this instrument and are registered here because a member found
# and not pinned is a member that gets lost: `subnormal-comparison`, from an
# unbiased draw while the classifier was being written, and
# `assume-narrows-past-the-program`, which an independent audit of this file
# drove while showing that nothing asserted a violation was not the sampler's
# own rounding. The second is a class the other eight do not cover, and it is
# the one whose CAUSE is the narrowing rather than the arithmetic.
#
# HOW FAR BACK THEY GO, RE-DERIVED RATHER THAN QUOTED. This module was
# commissioned with the sentence "four of the seven need no strict-sign
# certificate and reach v0.1.0, the first published release". Driven, with
# `git archive v0.1.0 src` on PYTHONPATH and v0.1.0's two-trace API
# (`harness.trace` plus `jax.make_jaxpr`, since `trace_with_jaxpr` did not
# exist yet) standing in for `read`'s one:
#
#   nine of the ten pinned programs violate box containment at v0.1.0 too;
#   FIVE of them falsify a DISCHARGE there — `ftz-subnormal-sum` (2 points),
#   `f32-underflow`, `f32-single-multiply`, `f32-exp` and
#   `assume-narrows-past-the-program` (1 each) — and `subnormal-comparison`
#   contradicts a refutation. The figure re-derives, and it went from four to
#   five when the ninth member was pinned, not when it was re-measured.
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
    return Program(decls, (_assert(pred),), label, "member")


#: ``label -> (Program, cause, what was measured, which verdict it
#: contradicts)``. The properties iterate this; nothing else in this file
#: mentions a member by name.
#:
#: THE FOURTH FIELD IS THE ONE A REPAIR TEAM READS FIRST, and it was a
#: sentence in three docstrings before it was a field. "Five of them falsify a
#: discharge" appeared in this file, in ``test_float_oracle.py`` and in
#: ``tests/property/README.md``, was quoted from a v0.1.0 run, and was
#: asserted nowhere — so which five was not answerable without re-running the
#: v0.1.0 experiment. It is a per-member fact now, measured at HEAD by
#: :func:`read` and floored on in the residual leg through
#: :data:`DISCHARGE_FALSIFYING`, so the count in the prose is DERIVED from the
#: registry rather than typed beside it.
MEMBERS = {}

#: ``""``, ``"discharge"`` or ``"refutation"`` — the permitted fourth field.
#: A member that contradicts no verdict is still a box-containment violation;
#: it just does not reach an obligation, which is the distinction
#: :func:`_contradicted_obligations` is written around.
CONTRADICTS = ("", "discharge", "refutation")


def _register(label, program, cause, measured, contradicts=""):
    assert contradicts in CONTRADICTS, contradicts
    MEMBERS[label] = (program, cause, measured, contradicts)


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
    "",
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
    "discharge",
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
    "discharge",
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
    "",
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
    "",
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
    "discharge",
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
    "discharge",
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
    "refutation",
)

# (9) NOT ONE OF THE SEVEN EITHER, and a class the other eight do not cover:
#     A PRECONDITION THAT NARROWS AWAY A POINT THE PROGRAM ADMITS. The assume
#     `x0 >= 5e-324` narrows the declared box over ℝ to `[5e-324, 1.0]`, and
#     the compiled comparison flushes the subnormal and ADMITS `x0 = 0.0` —
#     so the narrowed box excludes a point the program's own precondition lets
#     through, and the violation is on `stelling_any` itself. Every other
#     violation on that primitive can only be this module's sampler rounding
#     (:func:`snap_inward`), which is why `_compare` counts those separately;
#     this one is the reason the guard is "no `stelling_any` violation in a
#     harness with NO narrowing assume" rather than a blanket ban.
_register(
    "assume-narrows-past-the-program",
    Program(
        (Decl("x0", (), "float64", 0.0, 1.0),),
        (Stmt("assume", ("cmp", ">=", ("var", "x0"), ("const", 5e-324))),
         _assert(("cmp", ">", ("var", "x0"), ("const", 0.0)))),
        "assume-narrows-past-the-program",
        "member",
    ),
    NARROWING,
    "stelling_any executes 0.0 against the assume-narrowed box [5e-324, 1.0] "
    "— and INSIDE the declared box [0.0, 1.0], which is what makes the cause "
    "the narrowing rather than the flush. The mechanism behind the narrowing "
    "overshooting is still a flush: the compiled comparison flushes the "
    "subnormal and admits the point the analysis removed",
    "discharge",
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

#: The members whose box violation reaches an OBLIGATION and contradicts a
#: VERIFIED. Derived from the registry, never typed: the residual leg floors
#: on ``len()`` of this, so "five of them falsify a discharge" is a count the
#: file computes rather than a numeral beside a list that moves.
#:
#: Measured 2026-08-28 on this tree by :func:`read` at ``interior=0.5``, one
#: member at a time — ``ftz-subnormal-sum`` 2 discharges, ``f32-underflow``,
#: ``f32-single-multiply``, ``f32-exp`` and ``assume-narrows-past-the-program``
#: 1 each; ``subnormal-comparison`` 1 contradicted refutation; the other three
#: violate box containment at an equation UPSTREAM of the obligation and
#: contradict no verdict at all. The same five, and the same one, that the
#: v0.1.0 re-derivation below names — so the class does not merely still
#: exist, it still reaches the same verdicts through today's ``src/``.
#:
#: **READ THE CAUSE COLUMN BESIDE IT, BECAUSE FIVE PROGRAMS ARE NOT FIVE
#: REPAIRS.** These five carry THREE distinct causes —
#: ``flush-or-subnormal`` three times (``ftz-subnormal-sum``,
#: ``f32-underflow``, ``f32-exp``), ``narrow-format-rounding`` once
#: (``f32-single-multiply``) and ``assume-narrowing`` once
#: (``assume-narrows-past-the-program``) — and the sixth independent situation
#: the census reports, the aimed cancelling-sum construction, adds
#: ``reduction-reassociation``. **Four causes over six situations, and half of
#: the six are one repair.** Both facts are computed from this registry rather
#: than typed beside it: ``Counter(MEMBERS[n][1] for n in
#: DISCHARGE_FALSIFYING)``.
DISCHARGE_FALSIFYING = tuple(
    name for name, entry in MEMBERS.items() if entry[3] == "discharge"
)

#: The members that contradict a REFUTED rather than a VERIFIED — the
#: recoverable direction, counted apart for the reason
#: :func:`_contradicted_obligations` gives.
REFUTATION_CONTRADICTING = tuple(
    name for name, entry in MEMBERS.items() if entry[3] == "refutation"
)


# ── strategies ───────────────────────────────────────────────────────────────

from hypothesis import strategies as st  # noqa: E402

#: The element count at which ``jnp.sum`` stops folding left to right and
#: splits into two windows on this backend — re-derived from this instrument
#: rather than read off a lowering, by the 200-programs-per-size table in this
#: module's docstring (0/200 at n = 16, 30, 31 and 32; 18/200 at n = 33), and
#: byte-identical on jax 0.10.2.
#:
#: WRITTEN ONCE BECAUSE THREE PLACES NEED IT AND TWO OF THEM ARE ASSERTIONS.
#: :func:`cancelling_sum_programs` draws either side of it;
#: :func:`uniform_float_programs` states that ``_grammar.SHAPES`` cannot reach
#: it, which is the proof the unbiased leg's zero in the reassociation row
#: rests on; and ``test_float_oracle.py``'s residual leg asserts that premise
#: against ``SHAPES`` on every run, because the proof was stated in three
#: docstrings and checked in none.
REDUCTION_SPLIT_N = 33


def member_programs():
    """Every pinned member, as a strategy, so the search draws them like
    anything else — and so the containment leg reaches one within a few
    examples instead of relying on the unbiased grammar stumbling into the
    class."""
    return st.sampled_from([entry[0] for entry in MEMBERS.values()])


def uniform_float_programs():
    """``_grammar.general_specs`` as-is — the unbiased leg, and the control.

    UNBIASED IS THE POINT. This is the leg whose partition can be compared
    against a run of any other generator, and the leg that would catch a class
    nobody has thought of. It reaches NaN, flush-or-subnormal, narrow-format
    and overflow DENSELY ENOUGH TO RANK — its distinct-program counts are the
    only ones in the census that are a statement about the tool rather than
    about a strategy someone aimed. It reaches overflow-to-inf unreliably — 0
    in one 1500-draw derandomised run, 7 in another, which is why
    ``test_float_oracle.OVERFLOW_PROBE`` is pinned. And it reaches the
    reassociation class **never**, because ``SHAPES`` tops out at four
    elements and no draw of it can cross the n >= :data:`REDUCTION_SPLIT_N`
    boundary. The last of those is a proof about the generator rather than a
    zero in a table, and ``test_float_oracle.py``'s residual leg now ASSERTS
    its premise against ``_grammar.SHAPES`` on every run — it was stated in
    three docstrings and checked in none, while the whole weight of
    "reassociation is second by distinct programs" rested on it.

    **AND IT CONTRIBUTES NOTHING TO THE FIGURE EVERY CONSUMER QUOTES.** The
    census crosses the falsified-discharge count with
    :attr:`Program.source`, and this leg's cell is **0** — every discharge a
    repair would be scoped on comes from a pinned ``@example`` or from the
    aimed cancelling-sum strategy. That is not floorable (a floor on zero is
    not a floor) and it is not a defect in the generator either; it is the
    single most important thing a reader of the headline needs to know, which
    is why it is written here and in :data:`test_float_oracle.FLOAT_ORACLE_MEASURED`
    rather than left to be noticed.

    **NOTHING FLOORED THIS LEG UNTIL AN AUDIT SAID SO.** Every floor in
    ``test_float_oracle.py`` — the read count, the box count, the admitted
    points, the discharges, the per-member and per-cause tags — was satisfied
    by the pinned ``@example``s alone. Driven with both unbiased generators
    replaced by one declaration stelling refuses: BOTH LEGS GREEN, on
    ``read=39`` and 29 falsified discharges, every number of them from the
    pins.

    THE FLOOR THAT CLOSED IT IS ``read/uniform`` AND THIS SENTENCE NAMED THE
    WRONG ONE. It read *"the residual leg now floors on ``read/unlabelled``,
    which only a drawn program can raise"*. Both legs floor on
    ``read/uniform``; ``read/unlabelled`` is floored NOWHERE, and it is the
    weaker tag in any case — it counts this generator and
    :func:`cancelling_sum_programs` together, because that one also yields
    ``label == ""``, which is the very conflation the ``source`` field was
    added to undo. The floor in the tree is the right one and the sentence
    describing it was not.
    """
    return _grammar.general_specs().map(
        lambda s: Program(s.decls, s.stmts, "", "uniform")
    )


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
    alone produced **24** violations, every one of them
    ``reduction-reassociation``, none of them declined by the second route, at
    7.8 ms/example. That n = 128 is back to
    zero is recorded and NOT explained: 64 exact pairs may simply re-cancel
    under whatever split XLA picks there. It is a fact about this
    construction, not about the boundary, and the n <= 32 rows are what
    establish the boundary.
    """
    n = draw(st.sampled_from((
        REDUCTION_SPLIT_N, REDUCTION_SPLIT_N, REDUCTION_SPLIT_N,
        REDUCTION_SPLIT_N - 1, REDUCTION_SPLIT_N + 1, 64,
    )))
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
        "",
        "cancelling",
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
