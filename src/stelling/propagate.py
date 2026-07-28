# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Forward interval propagation over :mod:`stelling.ir`.

The autodidax pattern: walk the equations of a transcribed query in order,
mapping each primitive through a transfer function over
:class:`stelling.interval.IntervalArray`. Consumes the IR, never jaxprs.

Scope, held deliberately (design/e2a-registration.md): no widening, no
fixpoints, no cond/scan descent, no solver. The transfer registry contains
exactly the primitives the target census returned
(`design/primitive-census.md`, "The target census") plus the three harness
primitives, plus the closed pytree-probe registration round (abs, eq, ne,
and, or, stop_gradient, reshape, pow, reduce_or, and the scalar-selector /
rank-broadcast forms of already-registered transfers), plus one
allowed-by-census structural addition from the maddening HeatNode trace
(``scatter`` in its static-index ``x.at[k].set(v)`` form only), plus two
allowed-by-census structural additions from the MIME fvm laplacian trace
(``gather`` in its static-index leading-axis row form only, and
``transpose`` — both pure data movement with exact semantics), plus the
closed three-row round measured by attribution against a real trace
(``reduce_sum`` and ``integer_pow``; that round's third row was
emission-only), plus the two allowed-by-census additions of the
scatter-add/stack census round (the MIME LSQ normal-system census
contact: a real ``jax.ops.segment_sum`` assembly traced to a
``scatter-add`` equation with no row — coverage ``1 ⊤ (scatter-add
×1)``, obligation UNKNOWN, escalation declined naming the primitive —
so ``scatter-add`` lands in its static-index accumulate row forms only,
and ``stack``, what ``jnp.stack`` traces to on jax 0.11.0, lands as
pure element routing). Everything else falls to ⊤ — soundly, with
coverage recording exactly how much fell.

The three-row round is also where the ieee census first had to say **no**
to arithmetic it can state in ℝ. Both new rows contract more than one
float operation into a single equation while the jaxpr records no
evaluation ORDER for them, and neither float addition nor float
multiplication is associative — so under ``semantics="ieee"`` each is
censused down to the sub-cases that perform no arithmetic at all (a
reduction of at most 2 elements; exponent 0 or 1) and declines the rest
with the gap quoted. The ℝ transfers are unaffected: there, every
association order denotes the same number.

Every transfer declares an assumption tier (design commitment 5):
``exact`` (no arithmetic, or arithmetic with no rounding), ``sound``
(outward-rounded interval arithmetic), or ``sound-libm`` (outward-rounded
around a faithfully-rounded libm call — carries
:data:`stelling.interval.EXP_LIBM_ASSUMPTION`). The tiers of every
transfer *used* ride into the verdict stamp.

Obligations are ``stelling_assert`` equations. Their statuses:
``discharged`` (predicate definitely true over the declared box),
``unknown`` (interval too wide to decide — *our* imprecision), or
``violated-over-set`` (definitely false over the propagated superset of
the judged set — the declared box when no assume has constrained, the
precondition-narrowed set once one has; a **sound set-level
refutation** of that set, rendered as a REFUTED verdict per
`design/e2a-registration.md` amendment 1 with the conditional wording
when an assume constrained; not a witness, not a counterexample to the
program).

``stelling_assume`` **constrains** by default (``assume_mode="constrain"``):
where the predicate's producing equation is a comparison against a
point-interval side, the compared variable's own propagated interval is
narrowed by an exact meet with the *closed* half-space. The soundness
direction is the whole design: the narrowed domain must be a SUPERSET of
the true assumed region intersected with the current domain, never a
subset — so strict inequalities close (``x > k`` narrows to ``[k, hi]``,
never ``[nextafter(k), hi]``), nothing is ever inverted through
arithmetic (``assume(x*x <= 4)`` narrows the square's interval only,
never ``x``), and both-sides-varying comparisons stay inert (intervals
cannot represent relational coupling). Conjunctions recurse into their
conjuncts; every other predicate shape stays inert exactly as before —
dropped, which is sound (propagation runs over a superset) and never
silent: counted in coverage as ``inert`` (outside the "known" fraction)
and noted with its source address plus the reason (amendment 2).
Narrowing is forward-only: later equations read the narrowed interval,
earlier results are not revisited (conservative). Every applied
narrowing is disclosed three ways: a ``constrained`` coverage category,
an ``assume CONSTRAINED`` note, and a stamped assumption (the verdict
holds where the precondition holds).

Definite-verdict licensing splits by BOX EXACTNESS (audit F7): a box is
exact iff it equals the variable's true value set — ``stelling_any``
outputs and exact-point consts only; every transfer output is an
over-approximation (rounding pads, correlation-blind arithmetic). An
assume constraining an exact-box variable certifies its region's
satisfiability by the nonempty meet, and definite verdicts stand. An
assume constraining a NON-exact variable is still applied (the meet
over-approximates true-region ∩ reachable — sound) and the emptiness
refusals still fire (they prove emptiness from the over-approximation),
but the precondition's satisfiability is UNCERTIFIED: every subsequent
definite violation is withheld from REFUTED (status ``unknown``, note
quoted — a possibly-vacuous refutation is not a refutation), while
VERIFIED remains allowed carrying a may-be-vacuous note and stamped
line; the inert-mode control is the visibility instrument. An assume whose region is provably
EMPTY — an empty meet, a definitely-false constant comparison, or a
STRICT constraint whose meet collapses onto the closed boundary point
(``x > k`` narrowing to ``[k, k]``: the true region ``(k, k]`` is empty
though the closed stand-in is not) — is a harness defect at top level
and inside transparent call scopes: the precondition is definitely false
on the whole over-approximated domain, so every downstream obligation
would be vacuous — :class:`UnsatisfiableAssumptionError`, never a
verdict, never silent (degrade-don't-crash does not apply to harness
defects, exactly as for the unbound-var ``TranscriptionError``). Inside
a possibly-untaken ``cond`` BRANCH the same unsatisfiability is
branch-scoped (the other branch is real): the constraint is not applied
and a note discloses that obligations in that branch may be vacuous
under the branch's precondition — no raise, no narrowing.
``assume_mode="inert"`` reproduces the drop-everything MVP behavior
byte-identically — the comparability control for the vacuity instrument.

``semantics`` is the dial SOUNDNESS.md's stamp contract names: ``"real"``
(the default, byte-identical to everything above) judges obligations in
exact real arithmetic; ``semantics="ieee"`` judges them about the traced
program's **IEEE binary64 round-to-nearest float execution**. The ieee
domain is the same :class:`stelling.interval.IntervalArray` plus a
per-array ``maybe_nan`` flag (a parallel ``var id -> bool`` table beside
the interval env): endpoint arithmetic for the monotone core is native
binary64 with NO outward rounding (the float value itself is computable
— :data:`stelling.interval.IEEE_ENDPOINT_ASSUMPTION`), overflow
saturates to the VALUE ±inf, NaN-producing corner classes (``inf−inf``,
``0·±inf``, ``0/0``, ``inf/inf``) set ``maybe_nan``, ⊤ is
maybe-NaN, and a predicate over a maybe-NaN operand is never definitely
true (NaN falsifies every comparison except ``ne``, which it satisfies).
Every registered transfer is censused for ieee in
:data:`IEEE_TRANSFERS` — sound as-is, given an ieee variant, or declined
to ⊤-maybe-NaN with the gap quoted; the real mode's extended-real
conventions (the ``0·∞ = 0`` endpoint rule) are never reused. The
``domain`` parameter (only registered value ``"interval"``) is the
integration point for future tightened domains, which are refused under
``semantics="real"`` outright: tightening ℝ arithmetic without float
semantics converts accidental UNKNOWNs into false VERIFIEDs.

The exactness-certification machinery (the per-var exact set and the
box-nonemptiness-certifies-region-nonemptiness decision) lives in
:mod:`stelling.exactness` and is consumed here; the principle it holds —
a sound over-approximation certifies emptiness but never nonemptiness —
is that module's docstring.
"""

from __future__ import annotations

import math
from operator import index as _op_index
import struct
from dataclasses import dataclass

from stelling import exactness
from stelling import interval as iv
from stelling import ir
from stelling.coverage import DEFAULT_TRANSPARENT, Coverage, CoverageCounter, sub_jaxprs

__all__ = [
    "IEEE_TRANSFERS",
    "ObligationReport",
    "Propagation",
    "TIGHTENED_DOMAIN_REAL_REFUSAL",
    "TRANSFERS",
    "UnsatisfiableAssumptionError",
    "interval_env",
    "propagate",
]

TIER_EXACT = "exact"
TIER_SOUND = "sound"
TIER_SOUND_LIBM = "sound-libm"


class UnsatisfiableAssumptionError(ValueError):
    """An assume's precondition is definitely false on the whole
    over-approximated domain: the meet of the constrained variable's
    propagated interval with the assumed half-space is empty, so the
    declared set contains no point satisfying the precondition and every
    downstream obligation would be vacuously "verified".

    This is the empty-declared-set refusal class — a harness defect, like
    the unbound-var :class:`stelling.ir.TranscriptionError` — so
    degrade-don't-crash does not apply: raised loudly, never a VERIFIED,
    never silent.
    """

# 2**53: the largest magnitude below which every int is exactly a double.
_EXACT_INT = 9007199254740992


@dataclass(frozen=True)
class ObligationReport:
    """One ``stelling_assert`` equation, judged over the declared box."""

    index: int  # traversal order among obligations
    status: str  # "discharged" | "unknown" | "violated-over-set"
    detail: str
    source_info: tuple[str, ...]


@dataclass(frozen=True)
class Propagation:
    obligations: tuple[ObligationReport, ...]
    nonvacuity_checks: tuple[ObligationReport, ...]  # y0-membership conditions
    coverage: Coverage
    transfers_used: tuple[tuple[str, str], ...]  # (primitive, tier), sorted
    assumptions: tuple[str, ...]
    notes: tuple[str, ...]
    # which arithmetic the obligations were judged about ("real" | "ieee");
    # the verdict assemblers stamp from this field, never from a guess
    semantics: str = "real"
    # an assume was DROPPED in constrain mode: its predicate had no decidable
    # box, so the query ran over a SUPERSET of the intended set. Carried out
    # to the solver layer because the escalation refusal keys on a CONSTRAINED
    # assume being present, and a dropped one is not present at all — leaving
    # the solver free to emit a sat witness outside the precondition.
    assume_dropped: bool = False

    @property
    def all_discharged(self) -> bool:
        return bool(self.obligations) and all(
            o.status == "discharged" for o in self.obligations
        )

    @property
    def any_violated(self) -> bool:
        return any(o.status == "violated-over-set" for o in self.obligations)

    @property
    def dropped_constraints(self) -> int:
        return self.coverage.inert


# -- literals and consts ------------------------------------------------------
#
# The unsigned formats joined with the pytree-probe registration round:
# registering `and`/`or`/`eq`/`ne` made the propagator *read* the uint mask
# literals of RNG plumbing (threefry constants and friends) that previously
# sat behind unknown primitives — without decoders the read raised and the
# whole analysis died on a legal trace (h_hard), the exact failure the
# guard rule forbids. Unsigned ints decode to python ints, which
# `_int_bracket` already brackets soundly above 2**53.
_STRUCT_FMT = {
    "<f8": "d", "<f4": "f", "<i8": "q", "<i4": "i", "|b1": "?",
    "|u1": "B", "<u2": "H", "<u4": "I", "<u8": "Q",
    "|i1": "b", "<i2": "h",
}


def _int_bracket(v: int) -> tuple[float, float]:
    """A sound double bracket of an arbitrary python int (audit findings 3/5:
    int64 above 2**53 must widen; ints beyond the double range must saturate
    rather than crash on float())."""
    if abs(v) <= _EXACT_INT:
        x = float(v)
        return x, x
    try:
        x = float(v)
    except OverflowError:
        maxf = math.nextafter(math.inf, 0.0)
        return (maxf, math.inf) if v > 0 else (-math.inf, -maxf)
    return iv._down(x), iv._up(x)


def _decode_array(a: ir.Array) -> iv.IntervalArray:
    fmt = _STRUCT_FMT.get(a.dtype)
    if fmt is None:
        raise ir.TranscriptionError(
            f"no zero-dep decoder for array dtype {a.dtype!r}; add one before "
            f"propagating this query"
        )
    # shape predicates first (fix re-attacks R1/N2): integral nonnegative
    # extents (a negative or string extent would reach struct.unpack as a
    # malformed format and raise raw), through the decline channel
    iv.check_shape(a.shape)
    n = 1
    for d in a.shape:
        n *= d
    # ... and the PAYLOAD LENGTH against the shape (N2): a truncated,
    # oversized, or empty buffer under a positive shape raised raw
    # struct.error out of the walk — the exact sibling of the negative
    # route, one predicate away ("what else does unpack assume")
    expect = struct.calcsize(f"<{n}{fmt}")
    if len(a.data) != expect:
        raise iv.IntervalError(
            f"array constant of shape {a.shape} dtype {a.dtype!r} carries "
            f"{len(a.data)} byte(s), expected {expect} — truncated or "
            f"oversized payload (malformed IR)"
        )
    los, his = [], []
    for v in struct.unpack(f"<{n}{fmt}", a.data):
        if isinstance(v, int):
            lo, hi = _int_bracket(v)
        else:
            lo = hi = float(v)
        los.append(lo)
        his.append(hi)
    return iv.IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))


def _value_to_interval(v, shape: tuple[int, ...]) -> iv.IntervalArray:
    if isinstance(v, ir.Array):
        return _decode_array(v)
    if isinstance(v, bool):
        return iv.point(1.0 if v else 0.0, shape)
    if isinstance(v, int):
        lo, hi = _int_bracket(v)
        n = 1
        for d in shape:
            n *= d
        return iv.IntervalArray(shape=shape, los=(lo,) * n, his=(hi,) * n)
    if isinstance(v, float):
        return iv.point(v, shape)
    raise ir.TranscriptionError(
        f"no interval meaning for literal/const of type {type(v).__name__}"
    )


# -- the transfer registry ----------------------------------------------------
#
# Exactly the target census list + the harness primitives, plus the closed
# pytree-probe registration round (abs, eq, ne, and, or, stop_gradient,
# reshape, pow, reduce_or — the primitives the probe's own ⊤ list named),
# plus the maddening heat-node census round's one allowed structural
# addition (scatter, static-index set form only), plus the two allowed
# structural additions of the MIME fvm census round (gather,
# static-index row form only; transpose).
# Signature:
# transfer(eqn, params: dict, ins: list[IntervalArray]) -> list[IntervalArray]


# Conversions that are exact for EVERY representable source value — the only
# ones the pass-through is sound for (audit finding 1: f64->f32 rounds,
# int64->int32 wraps, int->bool collapses; passing those through produced a
# verified false VERIFIED on a real f32 round-trip trace).
_EXACT_CONVERSIONS = frozenset(
    {
        ("float32", "float64"),
        ("float16", "float32"),
        ("float16", "float64"),
        ("int8", "int16"), ("int8", "int32"), ("int8", "int64"),
        ("int16", "int32"), ("int16", "int64"),
        ("int32", "int64"),
        ("int8", "float32"), ("int16", "float32"),
        ("int8", "float64"), ("int16", "float64"), ("int32", "float64"),
        ("bool", "int8"), ("bool", "int16"), ("bool", "int32"),
        ("bool", "int64"), ("bool", "float32"), ("bool", "float64"),
        # bool -> unsigned was missing, which left a uint counter ⊤ for a
        # reason that had nothing to do with its arithmetic (audit
        # COSMETIC 5, case 2). {0, 1} is representable in every unsigned
        # width — MEASURED on jax 0.11.0, both values, eager and jit, for
        # uint4/8/16/32/64.
        ("bool", "uint4"), ("bool", "uint8"), ("bool", "uint16"),
        ("bool", "uint32"), ("bool", "uint64"),
    }
)

# Exact representable ranges per integer dtype: intN spans
# [-2**(N-1), 2**(N-1)-1], uintN spans [0, 2**N - 1], bool spans [0, 1].
# Kept as exact python ints, never floats: an int64 bound is not
# representable as a double and the whole point of the overflow guard is to
# be exact at the boundary (python compares int to float exactly).
_INT_DTYPE_BOUNDS: dict[str, tuple[int, int]] = {
    "bool": (0, 1),
    # int4/uint4 exist in jax 0.11.0. Their absence made the guard a silent
    # no-op for them, which was harmless only because no int4 conversion is
    # whitelisted — a coincidence, not a guarantee (audit, latent note).
    **{f"int{n}": (-(2 ** (n - 1)), 2 ** (n - 1) - 1) for n in (4, 8, 16, 32, 64)},
    **{f"uint{n}": (0, 2**n - 1) for n in (4, 8, 16, 32, 64)},
}


def _is_integer_dtype(dtype: str) -> bool:
    """Whether a dtype name denotes integer semantics — by NAME, so a dtype
    the bounds table has never heard of still reads as an integer and
    declines, instead of silently skipping the guard."""
    return dtype == "bool" or dtype.startswith(("int", "uint"))

# The float->int conversion guard's half-open bound, DERIVED from the table
# above so the two cannot drift: `-bound <= x < bound` is exactly
# [min, max] for a signed dtype. Behaviour unchanged — the strict upper
# comparison the second audit pinned (finding 4-B) is what makes the
# derivation exact.
_INT_RANGE = {d: float(_INT_DTYPE_BOUNDS[d][1] + 1) for d in ("int32", "int64")}


def _safe_top(shape) -> iv.IntervalArray:
    """⊤ of the shape — or the scalar ⊤ stand-in when the shape is
    uninhabited (a negative extent, which :class:`interval.IntervalArray`
    refuses to construct). The stand-in exists so a DECLINE over an
    impossible shape can still bind the outvar and keep the walk alive
    (the guard rule: declines never raise): no value of the variable
    exists, every equation CONSUMING it is itself declined by the
    negative-shape screen (its invar aval carries the negative dim), and
    the stand-in is read only by the assert/nonvacuity bookkeeping, where
    a full ⊤ box supports no definite face — so nothing can launder it
    into a definite verdict."""
    try:
        return iv.top(tuple(shape))
    except iv.IntervalError:
        return iv.top(())


def _refused_value_problem(aval_shape, value) -> str | None:
    """The refused-class problem of a constvar/literal binding, or None.

    Refused-class means NO coherent value can exist: a malformed or
    uninhabited shape on either coordinate (the recorded aval OR an
    ir.Array payload's own shape — a from_dict query can lie on one and
    not the other), or a payload whose byte length contradicts its shape.
    Deliberately NOT the inhabited-but-unbracketable class (NaN
    sentinels, undecodable dtypes), which keeps the registered
    ⊤-with-note treatment: those values EXIST."""
    try:
        iv.check_shape(aval_shape)
    except iv.IntervalError as e:
        return str(e)
    if isinstance(value, ir.Array):
        try:
            iv.check_shape(value.shape)
        except iv.IntervalError as e:
            return str(e)
        fmt = _STRUCT_FMT.get(value.dtype)
        if fmt is not None:
            n = 1
            for d in value.shape:
                n *= d
            expect = struct.calcsize(f"<{n}{fmt}")
            if len(value.data) != expect:
                return (
                    f"payload of {len(value.data)} byte(s) contradicts "
                    f"shape {value.shape} dtype {value.dtype!r} "
                    f"(expected {expect})"
                )
    return None


def _req(params, name: str, prim: str):
    """A required equation param, or an :class:`interval.IntervalError`
    decline. Transfers must never read a required param by bare
    subscript: a malformed ``from_dict`` query with the param missing
    would escape the transfer-call guard as a raw ``KeyError`` and kill
    the whole propagation walk (first-contact audit F4 — the guard rule
    is "declines never raise", and the transfer-call site catches exactly
    ``IntervalError``)."""
    if name not in params:
        raise iv.IntervalError(
            f"{prim!r} equation is missing its required param {name!r}"
        )
    return params[name]


def _in_range_int_narrowing(a, src: str, dst: str) -> bool:
    """Whether this ``int64 -> int32`` conversion is STATICALLY in range:
    every element interval lies inside int32's representable range, so
    the narrowing is the identity on every value the operand can hold —
    exact, no wraparound reachable. The census contact is INDEX data
    (third audit, F5b: the default-dtype ``at[].add`` sugar under x64
    declares its index constants int64 and narrows them to int32 before
    the scatter; without this the index column fell to ⊤ and the
    accumulate row declined for a reason that had nothing to do with its
    semantics). The rule is value-based, not use-based — a transfer
    cannot see what its output feeds — and is sound generically: an
    in-range int64 value IS its int32 image. The range check is static
    (the propagated interval) and exact at the boundary: float(2**31 - 1)
    is exactly representable, so `hi <= 2**31 - 1` admits the boundary
    value and `2**31` fails it (both sides pinned by test)."""
    if (src, dst) != ("int64", "int32"):
        return False
    lo_b, hi_b = _INT_DTYPE_BOUNDS["int32"]
    return all(lo_b <= x <= hi_b for x in (*a.los, *a.his))


def _t_convert(eqn, params, ins):
    (a,) = ins
    src = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    dst = str(params.get("new_dtype"))
    if src == dst or (src, dst) in _EXACT_CONVERSIONS:
        return [a]
    if _in_range_int_narrowing(a, src, dst):
        return [a]
    if "float" in src and dst in _INT_RANGE:
        # float -> integer truncates toward zero; trunc is monotone, so
        # [trunc(lo), trunc(hi)] brackets — but only while the values fit the
        # target's range (outside it jax clamps/wraps, which trunc does not
        # model). The upper check is STRICT: +2**n-1 itself is not
        # representable (intN max is 2**(n-1) - 1), and for int64 the float
        # `bound - 1` rounds back to `bound` — second audit, finding 4-B: the
        # inclusive check admitted exactly ±2**31 and claimed a value int32
        # cannot hold.
        bound = _INT_RANGE[dst]
        if any(not (-bound <= x < bound) for x in (*a.los, *a.his)):
            return None
        return [iv.IntervalArray(
            shape=a.shape,
            los=tuple(float(math.trunc(x)) for x in a.los),
            his=tuple(float(math.trunc(x)) for x in a.his),
        )]
    return None  # value-changing or unrecognized conversion -> ⊤, noted


# The primitives whose operand a product may be CONTRACTED into, as one
# fused multiply-add. Named rather than inlined because it was a by-hand
# tuple at the gate AND a second by-hand tuple in the test that covers the
# gate, so a primitive could be added to the registry, pass every census
# constraint, and be missing from both. `add_any` was, for one commit.
# The test now iterates this constant, so the gate and its coverage cannot
# disagree about what is gated.
IEEE_CONTRACTION_ADDENDS = ("add", "sub", "add_any")


def _t_split(eqn, params, ins):
    """``jax.lax.split``: cut one operand along ``axis`` into ``sizes`` pieces.

    EXACT, and exactly so: every output element IS an input element, at a
    statically known index. There is no arithmetic, no rounding, and no
    dtype question — which is why this is classified non-computing.

    Built on :func:`interval.slice_` rather than on fresh index arithmetic:
    the offsets are a running sum along one axis and the extents are the
    operand's on every other, which is precisely a slice. Reusing the
    audited helper is the "don't hand-roll a traversal" norm applied to
    indexing — a second implementation of the same bounds arithmetic is a
    second place for it to be wrong.

    Declines (⊤) when the params do not describe the operand it was handed:
    sizes that do not sum to the axis extent, an axis outside the operand's
    rank, a negative size, or an output count the params disagree with.
    """
    if len(ins) != 1:
        return None
    (a,) = ins
    sizes = params.get("sizes")
    axis = params.get("axis")
    if sizes is None or axis is None:
        return None  # absent params are not a traced form; never guessed
    try:
        sizes = tuple(int(s) for s in sizes)
        axis = int(axis)
    except (TypeError, ValueError):
        return None
    rank = len(a.shape)
    if not 0 <= axis < rank or any(s < 0 for s in sizes):
        return None
    if sum(sizes) != a.shape[axis]:
        return None  # the cut does not partition the axis
    if len(sizes) != len(eqn.outvars):
        return None  # params and the equation's own arity disagree

    out, start = [], 0
    for s in sizes:
        starts = tuple(start if d == axis else 0 for d in range(rank))
        limits = tuple(start + s if d == axis else a.shape[d]
                       for d in range(rank))
        out.append(iv.slice_(a, starts, limits, None))
        start += s
    return out


def _t_reshape(eqn, params, ins):
    if params.get("dimensions") is not None:
        # a dimensions= reshape permutes before reshaping — not the C-order
        # flat identity; no rule yet, so decline (⊤ with the params noted).
        return None
    return [iv.reshape(ins[0], tuple(_req(params, "new_sizes", "reshape")))]


def _t_bool_logic(name, op):
    """jax's ``and``/``or`` are *bitwise*: on bool operands they are the
    logical connectives our three-valued encoding models; on integers they
    are bit arithmetic, which no interval rule here covers — decline."""

    def t(eqn, params, ins):
        dtypes = [v.aval.dtype for v in eqn.invars]
        if any(d != "bool" for d in dtypes):
            raise iv.IntervalError(
                f"{name!r} transfer covers bool operands only (three-valued "
                f"logic); got dtypes {dtypes} — bitwise integer {name} has "
                f"no interval rule"
            )
        return [op(*ins)]

    return t


def _t_reduce_or(eqn, params, ins):
    dtypes = [v.aval.dtype for v in eqn.invars]
    if any(d != "bool" for d in dtypes):
        raise iv.IntervalError(
            f"'reduce_or' transfer covers bool operands only; got dtypes "
            f"{dtypes}"
        )
    return [iv.reduce_or(ins[0], tuple(_req(params, "axes", "reduce_or")))]


def _scatter_index_dtype_covers(indices_dtype: str | None, axis_len: int) -> bool:
    """Whether the operand's leading-axis bound is EXACTLY representable in the
    scatter index array's element type — the one authority for the index-dtype
    admissibility of both scatter rows, shared by both oracles and quoted by
    the emission faces.

    XLA computes a scatter's out-of-bounds bound IN THE INDEX ARRAY'S ELEMENT
    TYPE. For the covered rows the window along the scattered axis is one
    element (``inserted_window_dims = (0,)``), so that bound is
    ``operand.shape[0] - 1``. When it does not fit the index dtype it WRAPS,
    and the comparison XLA performs is not the comparison the rows model:
    every index failing the wrapped test is silently DROPPED under
    FILL_OR_DROP while the row models the write as landing. No error, no
    ⊤ — a wrong value.

    Measured on jax 0.11.0, `scatter` and `scatter-add` alike, index k=1,
    FILL_OR_DROP: int8 column writes at operand length 128 (bound 127, the
    int8 max) and DROPS at 129 (bound 128, wrapping to -128); int16 writes at
    32768 and drops at 32769; uint8 writes at 256 and drops at 257 (bound 256
    wrapping to 0). The unsigned case is the one that looks benign under a
    careless probe: at 257 the wrapped bound is 0, so an index of 0 still
    writes and only k >= 1 is dropped — the boundary is the dtype's MAXIMUM,
    not its signedness.

    A dtype the integer-bounds table does not name (including an absent aval
    dtype) yields False: there is then no basis on which to claim the bound is
    representable, and the rows never guess.
    """
    bounds = _INT_DTYPE_BOUNDS.get(indices_dtype or "")
    if bounds is None:
        return False
    lo, hi = bounds
    return lo <= axis_len - 1 <= hi


def _scatter_indices_dtype(eqn) -> str | None:
    """The index operand's aval dtype for a scatter-shaped equation, or None
    when there is no such operand.

    The interval domain does not carry dtypes — :class:`interval.IntervalArray`
    is bounds only — so the propagation face reads the index dtype from the
    EQUATION, which is where the emission face reads it too. None (and any
    dtype the bounds table does not name) declines at
    :func:`_scatter_index_dtype_covers`.
    """
    return eqn.invars[1].aval.dtype if len(eqn.invars) > 1 else None


# The single-element scatter dimension_numbers of ``x.at[k].set(v)`` on a
# rank-1 operand; any OTHER field a jax version adds (batching dims today)
# must be empty or the transfer declines.
_SCATTER_SET_CORE = {
    "update_window_dims": (),
    "inserted_window_dims": (0,),
    "scatter_dims_to_operand_dims": (0,),
}


def _scatter_set_row_form(
    params, operand_shape, indices_shape, updates_shape, indices_dtype
):
    """The ONE admissibility oracle for the static-index ``scatter`` set-form,
    shared by the propagation transfer and the SMT emission — the same posture
    :func:`_scatter_add_row_form` holds for the accumulate form.

    Returns True for the single covered form: canonical single-element
    dimension numbers, rank-1 operand, one index row, scalar update, and an
    index dtype that exactly represents the operand's leading-axis bound.
    Everything else is False and the caller declines with its own wording.
    Sharing it is the point: a bounds or shape rule that lived in two places
    could be tightened in one and not the other, and the emission is the face
    where getting it wrong mints a false model rather than a ⊤.

    ``indices_dtype`` is the index array's aval dtype and is REQUIRED, not
    defaulted: the range check the callers perform is against the operand's
    leading axis, but XLA performs it in the index element type, and a caller
    that forgot to pass the dtype would silently get the old, wrong rule back
    (:func:`_scatter_index_dtype_covers` carries the measurement).

    NOTE what this does NOT check: the index VALUE (each caller reads it from
    its own domain — intervals on the transfer side, static constants on the
    emission side) and the scatter ``mode``. Both are the caller's, and both
    are load-bearing; see the callers.
    """
    # THE COMBINER GATE, and it belongs HERE rather than in either caller.
    # `x.at[k].apply(f)` traces to the SAME primitive with the SAME dimension
    # numbers, shapes, mode and static index as `.set` — the ONLY thing
    # distinguishing them is a non-None `update_jaxpr` carrying f, alongside a
    # DUMMY 0.0 updates operand. A form test that does not look at it admits
    # `.apply` as if it were `.set` and models `out[k] = 0.0` where the program
    # computes `out[k] = f(operand[k])`.
    #
    # This check lived in the transfer while the emission used only this
    # oracle, so the emission never saw it — which is precisely the asymmetry
    # a shared oracle exists to prevent, reintroduced by extracting the shape
    # rules and leaving this one behind. `_scatter_add_row_form` gates its own
    # combiner via `_is_add_combiner`; the SET form's combiner must be ABSENT,
    # because "set" means there is no combining.
    if params.get("update_jaxpr") is not None:
        return False
    if params.get("update_consts"):
        return False
    dn = params.get("dimension_numbers")
    if not isinstance(dn, ir.NamedTupleParam):
        return False
    fields = dict(dn.fields)
    if any(fields.get(k) != v for k, v in _SCATTER_SET_CORE.items()):
        return False
    if any(v != () for k, v in fields.items() if k not in _SCATTER_SET_CORE):
        return False
    if len(operand_shape) != 1 or indices_shape != (1,) or updates_shape != ():
        return False
    # THE INDEX-DTYPE GATE. The callers range-check the index against
    # operand_shape[0]; XLA range-checks it against a bound computed in the
    # INDEX dtype, and the two are the same comparison only while that bound
    # is representable there.
    if not _scatter_index_dtype_covers(indices_dtype, operand_shape[0]):
        return False
    return True


def _t_scatter(eqn, params, ins):
    """``x.at[k].set(v)`` in its static-index form — the allowed-by-census
    structural addition from the maddening HeatNode census trace (the
    Dirichlet boundary writes ``T_new.at[0].set(T_left)`` /
    ``.at[-1].set(T_right)``).

    Covered form, exactly: rank-1 operand, one index row holding one
    definite in-range integer, scalar update, no update computation
    (``update_jaxpr=None``), canonical single-element dimension numbers.
    The output is the operand's box with element ``k``'s interval replaced
    by the update's — pure data movement, no arithmetic, no rounding.

    Everything else declines to a noted ⊤: dynamic (non-point) or
    out-of-range indices (mode-dependent clamp/drop is the census's wedge
    bug class, never guessed), update windows, batching dims, higher
    ranks, computed updates.
    """
    # WHAT THIS FUNCTION RETAINS BEYOND THE SHARED ORACLE, enumerated because
    # an unenumerated retained check is how the two faces drifted apart once
    # already: the combiner gate lived here while the emission used only the
    # oracle, so `.apply` was admitted as `.set`. Everything below is either a
    # precondition for CALLING the oracle or genuinely interval-domain.
    #
    #   len(ins) != 3        — arity. The oracle takes three shapes, so having
    #                          three operands is a precondition for calling it.
    #   update_jaxpr         — DELIBERATE DEFENSIVE DUPLICATE. The oracle now
    #                          gates this and is the authority; this copy is
    #                          kept, not removed, because it is the check whose
    #                          absence caused the defect and a redundant gate
    #                          costs nothing.
    #   index point/finite/integral, and in-range — interval-domain. This face
    #                          reads the index from a propagated INTERVAL; the
    #                          emission reads it from static constants. Same
    #                          question, two representations, so it cannot live
    #                          in a shape-and-params oracle.
    if len(ins) != 3 or params.get("update_jaxpr") is not None:
        return None
    operand, indices, updates = ins
    if not _scatter_set_row_form(
        params, operand.shape, indices.shape, updates.shape,
        _scatter_indices_dtype(eqn),
    ):
        return None  # form outside the covered row — the shared oracle's call
    lo, hi = indices.los[0], indices.his[0]
    if lo != hi or not math.isfinite(lo) or lo != math.floor(lo):
        return None  # dynamic or non-integral index: no exact rule
    k = int(lo)
    if not 0 <= k < operand.shape[0]:
        return None  # out of range: FILL_OR_DROP drops, CLIP clamps — decline
    los = list(operand.los)
    his = list(operand.his)
    los[k] = updates.los[0]
    his[k] = updates.his[0]
    return [iv.IntervalArray(shape=operand.shape, los=tuple(los), his=tuple(his))]


def _t_gather(eqn, params, ins):
    """``x[idx]`` in its static-index leading-axis row form — the
    allowed-by-census structural addition from the MIME fvm laplacian
    trace (the gather half of the operators' gather→compute→scatter
    pattern: ``phi[mesh.owner]`` / ``phi[mesh.neighbour]`` on rank-1
    fields and ``grad[mesh.owner]`` on rank-2, with the mesh topology
    entering as definite const indices).

    Covered form, exactly: operand of rank r >= 1; indices ``(N, 1)``
    holding definite integral in-range points; dimension numbers that
    collapse exactly the leading axis (``offset_dims = (1, …, r-1)``,
    ``collapsed_slice_dims = (0,)``, ``start_index_map = (0,)``, every
    batching field empty); ``slice_sizes = (1, *operand.shape[1:])``.
    The output stacks the selected rows: ``out[i] = operand[k_i]`` —
    pure data movement, no arithmetic, no rounding. All
    ``GatherScatterMode``\\ s agree on definitely-in-range indices, so
    the mode is not constrained here.

    Everything else declines to a noted ⊤: dynamic (non-point) or
    out-of-range indices (mode-dependent clamp/drop/fill is the census's
    wedge bug class, never guessed), batching dims, window offsets not
    covering the full trailing block, multi-column index vectors.
    """
    if len(ins) != 2:
        return None
    operand, indices = ins
    r = len(operand.shape)
    if r < 1 or len(indices.shape) != 2 or indices.shape[1] != 1:
        return None
    dn = params.get("dimension_numbers")
    if not isinstance(dn, ir.NamedTupleParam):
        return None
    fields = dict(dn.fields)
    want = {
        "offset_dims": tuple(range(1, r)),
        "collapsed_slice_dims": (0,),
        "start_index_map": (0,),
    }
    if any(fields.get(k) != v for k, v in want.items()):
        return None
    if any(v != () for k, v in fields.items() if k not in want):
        return None
    if tuple(params.get("slice_sizes", ())) != (1,) + operand.shape[1:]:
        return None
    ks = []
    for lo, hi in zip(indices.los, indices.his):
        if lo != hi or not math.isfinite(lo) or lo != math.floor(lo):
            return None  # dynamic or non-integral index: no exact rule
        k = int(lo)
        if not 0 <= k < operand.shape[0]:
            return None  # out of range: clamp/drop/fill is mode-dependent
        ks.append(k)
    return [iv.take_rows(operand, ks)]


# The core scatter-add dimension-number fields shared by every measured
# form; any OTHER field a jax version adds (batching dims today) must be
# empty or the form oracle returns None (the transfer declines).
_SCATTER_ADD_CORE = {
    "inserted_window_dims": (0,),
    "scatter_dims_to_operand_dims": (0,),
}


def _is_add_combiner(update_jaxpr) -> bool:
    """Whether a ``scatter-add`` equation's recorded combining function is
    the measured single-``add`` form: a :class:`stelling.ir.ClosedJaxpr`
    with no consts, two scalar invars, and exactly one ``add`` equation
    combining them into the single outvar (measured on jax 0.11.0 — every
    traced ``scatter-add`` carries exactly this). The primitive NAME
    declares the accumulate semantic; a recorded combiner that
    contradicts the name is malformed IR and the form oracle refuses it
    rather than trusting either self-description."""
    if not isinstance(update_jaxpr, ir.ClosedJaxpr):
        return False
    j = update_jaxpr.jaxpr
    if update_jaxpr.consts or len(j.invars) != 2 or len(j.outvars) != 1:
        return False
    if len(j.eqns) != 1:
        return False
    e = j.eqns[0]
    if e.primitive != "add" or len(e.invars) != 2 or len(e.outvars) != 1:
        return False
    invar_ids = {v.id for v in j.invars}
    got_ids = {a.id for a in e.invars if isinstance(a, ir.Var)}
    if got_ids != invar_ids:
        return False  # the add must combine exactly the two operands
    out = j.outvars[0]
    return isinstance(out, ir.Var) and out.id == e.outvars[0].id


def _scatter_add_row_form(
    params,
    operand_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    updates_shape: tuple[int, ...],
    indices_dtype: str | None,
) -> int | None:
    """The pinned static-shape FORM of a ``scatter-add`` equation, or None.

    Exactly the dimension_numbers configurations measured on jax 0.11.0
    (the scatter-add/stack census round: ``jax.ops.segment_sum`` 1-D and
    with trailing dims, ``x.at[idx].add(v)`` with array and static scalar
    idx), and no generalization past them. For operand rank ``r >= 1``:

    * **index-column form** — indices ``(n, 1)``, updates
      ``(n, *operand.shape[1:])``, ``update_window_dims = (1, …, r-1)``:
      updates row ``j`` accumulates into operand row ``k_j``
      (``segment_sum`` and array-index ``at[].add``);
    * **scalar-index form** — indices ``(1,)``, updates
      ``operand.shape[1:]``, ``update_window_dims = (0, …, r-2)``: the
      single updates block accumulates into row ``k_0`` (the
      ``x.at[1].add(v)`` sugar).

    Both share ``inserted_window_dims = (0,)`` and
    ``scatter_dims_to_operand_dims = (0,)``; every other
    dimension-numbers field (batching dims) must be empty. The ``mode``
    param is deliberately NOT constrained: the registered transfer
    refuses any index that is not definitely in range, and all
    ``GatherScatterMode``\\ s agree on in-range indices (out-of-range is
    where they diverge — measured: FILL_OR_DROP drops the update, clip
    accumulates into the clamped row). The recorded combiner
    (``update_jaxpr``) must be absent or the measured single-``add``
    form (:func:`_is_add_combiner`); ``update_consts`` must be empty.

    ``indices_dtype`` (the index array's aval dtype, REQUIRED — see
    :func:`_scatter_set_row_form` on why it is not defaulted) must exactly
    represent the operand's leading-axis bound: XLA computes the
    out-of-bounds bound in the index element type, so an index column too
    narrow for that bound has its updates silently DROPPED where this row
    models them as accumulating (:func:`_scatter_index_dtype_covers`).

    Returns the number of scattered index rows ``n``, or None (the
    caller declines).
    """
    r = len(operand_shape)
    if r < 1:
        return None
    if not _scatter_index_dtype_covers(indices_dtype, operand_shape[0]):
        return None
    dn = params.get("dimension_numbers")
    if not isinstance(dn, ir.NamedTupleParam):
        return None
    fields = dict(dn.fields)
    if any(fields.get(k) != v for k, v in _SCATTER_ADD_CORE.items()):
        return None
    known = set(_SCATTER_ADD_CORE) | {"update_window_dims"}
    if any(v != () for k, v in fields.items() if k not in known):
        return None
    # KEY PRESENCE, not `.get()`. The two are different facts and jax uses the
    # difference: an ABSENT key is the hand-built IR form, where the primitive
    # name is the semantic authority (blessed by
    # test_absent_combiner_is_accepted_hand_built_form). A key PRESENT with
    # value None is what a jax-produced equation carries, and jax's
    # `_scatter_lower` substitutes `lambda x, y: y` for it — REPLACE,
    # last-wins, operand discarded, duplicates NOT accumulated. Reading it with
    # `.get()` conflates the two and models a set as an add.
    #
    # Measured on jax 0.11.0: operand zeros(3), indices [[0],[2],[0],[0]],
    # updates [1,10,100,1000] gives [1101,0,10] with the add combiner and
    # [1000,0,10] with update_jaxpr=None.
    if "update_jaxpr" in params:
        uj = params["update_jaxpr"]
        if uj is None:
            return None  # jax's set/last-wins combiner, not accumulation
        if not _is_add_combiner(uj):
            return None  # a combiner contradicting the primitive name
    if params.get("update_consts") not in ((), None):
        return None
    uwd = fields.get("update_window_dims")
    tail = tuple(operand_shape[1:])
    if (
        len(indices_shape) == 2
        and indices_shape[1] == 1
        and uwd == tuple(range(1, r))
        and updates_shape == (indices_shape[0],) + tail
    ):
        return indices_shape[0]
    if (
        indices_shape == (1,)
        and uwd == tuple(range(0, r - 1))
        and updates_shape == tail
    ):
        return 1
    return None


# Relative precision of an accumulator, in mantissa bits. Used ONLY to
# compare accumulation width against operand width; every threshold below is
# backed by a measured discrepancy, never by caution (CONTRIBUTING.md: "a
# decline rule must trace to a measured discrepancy with a magnitude").
_MANTISSA_BITS = {
    "bfloat16": 8, "float16": 11, "float32": 24, "float64": 53,
}


def _dot_general_row_form(params, lhs_dtype: str, rhs_dtype: str):
    """THE SHARED ADMISSIBILITY ORACLE for the ``dot_general`` row.

    Built shared from the start and driven by BOTH faces — the interval
    transfer (:func:`_t_dot_general`) and the SMT emission
    (:func:`stelling.obligation._dot_general_plan`). The scatter defect this
    design exists to prevent was an oracle *extracted* after the fact, which
    moved the shape rules and left the combiner check behind in the transfer;
    a row admitting ``.apply(f)`` as ``.set`` was the result.

    Returns ``(dimension_numbers, None)`` when the equation is inside the
    modelled form, or ``(None, reason)`` when it is not. The reason is
    returned rather than logged so each face can surface it in its own
    vocabulary without either inventing wording the other does not use.

    All FOUR semantics-changing params are read. Scatter had one such param
    and three independent audits found three defects in it; an unread param
    is not a benign param.

    Every decline below names the magnitude that motivates it, measured on
    jax 0.11.0 against an independently-computed model:

    ``preferred_element_type`` complex, or a complex operand
        DECLINED ON DOMAIN, not precision — measured 3.62 relative, but the
        number is beside the point. :class:`~stelling.interval.IntervalArray`
        carries ``los``/``his``, which presuppose a total order; ℂ has none,
        so there is no interval representation here to be imprecise with.

    ``preferred_element_type`` integral
        3.9e-08 … 1.0 — jax accumulates in the integer type and WRAPS, while
        the row models an exact real contraction. Measured 1.0000000002
        (int32) and 1.0 (int64): the result is not approximately wrong, it
        is unrelated.

    ``preferred_element_type`` narrower than a float operand
        float32 3.9e-08, float16 1.6e-04, bfloat16 2.1e-03.

        AND THE RULE BOUNDS NARROWING, NOT ERROR — the earlier wording
        implied otherwise and a blinded audit measured the gap. On matched
        data this gate DECLINES float64 operands accumulated in float32 at
        a relative error of 2.3e-08 while ADMITTING bfloat16 operands
        accumulated in bfloat16 at 2.1e-03, ninety thousand times larger.
        That is not incoherent once stated correctly: under ℝ semantics a
        float16 program diverges from exact real arithmetic no matter which
        primitive runs it, and stelling admits float16 ``add`` on exactly
        the same footing. Tightening this to "accumulation must be
        float64" would refuse ordinary ``float32 @ float32`` that every
        other row in the engine accepts. What this rule enforces is that
        the accumulation is no narrower than the operands the caller chose;
        the declared ℝ/IEEE gap is the framework's, recorded in every stamp,
        and is not this row's to close.

    integer operands accumulated in anything but ``float64``
        int32 with float32 accumulation measures 2.0e-08. With float64 it
        measures 5.9e-16 and int64 with float64 measures 1.9e-16 — both far
        below the gauge's 1e-12 threshold, so integer operands are ADMITTED
        under float64 accumulation. Declining integer *operands* outright
        would be a capability loss with no discrepancy behind it, which is
        the failure mode the norm forbids; the decline is on the
        ACCUMULATION dtype.

    ``out_sharding``
        Admitted only when ``None``. Not measured, and that is exactly why:
        a sharded contraction is a distributed program whose semantics this
        row has never been compared against, and an unread param is not a
        benign one.

    ``precision``
        Read, and unrecognised values decline. Recognised values are
        admitted: all of DEFAULT/HIGH/HIGHEST and the per-operand 2-tuple
        measure identical to ``None`` on this CPU backend, and under ℝ
        semantics ``precision`` selects float rounding rather than a
        different real-valued function. TF32 paths on accelerators round
        differently, which ℝ semantics does not model either — the stamp
        records ``device_class``.
    """
    dn = params.get("dimension_numbers")
    if dn is None:
        return None, "dot_general has no dimension_numbers"
    try:
        (lc, rc), (lb, rb) = dn
        dn = ((tuple(lc), tuple(rc)), (tuple(lb), tuple(rb)))
    except Exception:
        return None, f"dot_general dimension_numbers not in jax's form: {dn!r}"

    if "out_sharding" in params and params["out_sharding"] is not None:
        return None, (
            "dot_general carries out_sharding="
            f"{params['out_sharding']!r}: a sharded contraction is a "
            "distributed program this row has not been compared against"
        )

    prec = params.get("precision")
    if not _recognised_precision(prec):
        return None, (
            f"dot_general precision={prec!r} is not a form this row has "
            "measured"
        )

    for name, dt in (("lhs", lhs_dtype), ("rhs", rhs_dtype)):
        if "complex" in dt:
            return None, (
                f"dot_general {name} operand is {dt}: complex is outside "
                "this row's DOMAIN, not merely its precision — interval "
                "endpoints presuppose a total order and ℂ has none"
            )

    pet = params.get("preferred_element_type")
    if pet is None:
        # ABSENT ACCUMULATION TYPE IS NOT AN ABSENT CONSTRAINT. With no
        # preferred_element_type jax derives the output dtype from the
        # operands, so int32 x int32 accumulates in int32 and WRAPS -- the
        # 1.0-relative class -- while a row that skipped the dtype checks
        # here would model it as exact real arithmetic. This is the
        # present-vs-absent distinction that produced the scatter-add false
        # VERIFIED, at the parameter level rather than the key level.
        for name, dt in (("lhs", lhs_dtype), ("rhs", rhs_dtype)):
            if dt not in _MANTISSA_BITS:
                return None, (
                    f"dot_general has no preferred_element_type and a {dt} "
                    f"{name} operand: accumulation follows the operands, and "
                    "integer accumulation WRAPS where this row models exact "
                    "real arithmetic (measured 1.0 relative)"
                )
        return dn, None
    if pet is not None:
        pet = str(pet)
        if "complex" in pet:
            return None, (
                f"dot_general accumulates in {pet}: complex is outside this "
                "row's domain — interval endpoints presuppose a total order"
            )
        if pet not in _MANTISSA_BITS:
            return None, (
                f"dot_general accumulates in {pet}, an integer or "
                "unrecognised type: integer accumulation WRAPS where this "
                "row models exact real arithmetic (measured 1.0 relative)"
            )
        acc_bits = _MANTISSA_BITS[pet]
        for name, dt in (("lhs", lhs_dtype), ("rhs", rhs_dtype)):
            if dt in _MANTISSA_BITS:
                if acc_bits < _MANTISSA_BITS[dt]:
                    return None, (
                        f"dot_general accumulates in {pet} from a {dt} "
                        f"{name} operand: the accumulation is narrower than "
                        "the operands (measured 3.9e-08 for float32)"
                    )
            elif pet != "float64":
                # integral operand: only float64 accumulation measured benign
                return None, (
                    f"dot_general accumulates a {dt} {name} operand in "
                    f"{pet}: only float64 accumulation of integer operands "
                    "is measured benign (5.9e-16); float32 measures 2.0e-08"
                )
    return dn, None


def _recognised_precision(prec) -> bool:
    """``precision`` forms this row has measured. A 2-tuple is jax's
    per-operand form, which is what a string like ``"highest"`` normalises
    to.

    READS THE TRANSCRIBED FORM, which is the only form this ever sees. A
    blinded audit found the first version broken here: it did
    ``str(prec).split(".")[-1].upper()``, which works on a raw
    ``jax.lax.Precision`` member and NEVER matches the transcriber's
    ``ir.EnumParam(cls='Precision', member='HIGHEST')``. Every non-``None``
    precision from a real trace declined.

    The decline direction is safe, so nothing unsound shipped — but the
    cost was real: an explicit ``precision="highest"`` on a contraction is
    ordinary numerical-code practice, and the row refused every one of them
    while its docstring said otherwise.

    The reason the unit tests missed it is worth more than the bug: they
    constructed params from RAW JAX OBJECTS and the engine only ever
    receives TRANSCRIBED ones. A test that builds its own input in a form
    the system never produces is testing a path that does not exist.
    """
    if prec is None:
        return True
    if isinstance(prec, tuple):
        return len(prec) == 2 and all(_recognised_precision(p) for p in prec)
    if isinstance(prec, ir.EnumParam):
        return prec.cls == "Precision" and prec.member.upper() in (
            "DEFAULT", "HIGH", "HIGHEST"
        )
    return str(prec).split(".")[-1].upper() in ("DEFAULT", "HIGH", "HIGHEST")


def _check_unique_indices_promise(params, ks, exc_type) -> None:
    """The ``unique_indices`` promise check (third audit, F2): an equation
    carrying ``unique_indices=True`` whose static indices measurably
    contain duplicates has VIOLATED its own promise, and what jax computes
    then is implementation-defined — the promise exists precisely so
    backends may exploit it (skip the atomic/combine path). The measured
    CPU happening to accumulate is a coincidence of one backend, not a
    semantic; modelling accumulate here would be a guess on a
    self-described-unreliable input, the same never-guess posture as the
    out-of-range mode dependence. So it declines loudly, at the transfer
    AND at the emission (both call here with their own decline type).

    ``unique_indices=True`` with actually-unique indices proceeds — the
    promise holds and accumulate degenerates to at-most-one contribution
    per element, where every backend agrees. ``indices_are_sorted``
    deliberately needs NO action: sorting permutes the contribution order
    only, and the ℝ accumulate is order-free (addition is associative and
    commutative there), so a kept or violated sort promise cannot change
    the value this transfer brackets."""
    if params.get("unique_indices") is not True:
        return
    if len(set(ks)) == len(ks):
        return
    from collections import Counter

    k, c = Counter(ks).most_common(1)[0]
    raise exc_type(
        f"scatter-add carries unique_indices=True but its static indices "
        f"contain duplicates (index {k} appears {c} times): the promise "
        f"licenses backends to assume no index is repeated, so duplicate "
        f"behaviour under it is implementation-defined — modelling "
        f"accumulate here would be a guess on a self-described-unreliable "
        f"input; declined (the never-guess posture of the out-of-range "
        f"mode dependence, applied to the uniqueness promise)"
    )


def _t_scatter_add(eqn, params, ins):
    """``scatter-add`` in its static-index accumulate row forms — the
    allowed-by-census addition from the MIME LSQ normal-system census
    round (``jax.ops.segment_sum`` assembling the LSQ normal matrix
    ``M = Σ_f d_f ⊗ d_f`` traced to exactly this equation; the same
    primitive is what ``x.at[idx].add(v)`` lowers to).

    The defining semantic — the reason this is NOT the registered
    set-form ``scatter`` row with a different name: **duplicate indices
    ACCUMULATE**. ``out[i] = operand[i] + Σ_j updates[j]`` over every
    index row ``j`` mapping to ``i`` (measured:
    ``zeros(3).at[[0,2,0,0]].add([1,10,100,1000])`` is ``[1101, 0, 10]``;
    the set form's last-wins would be ``[1000, 0, 10]``). With static
    indices the contributing set per output element is statically known,
    so the transfer is the outward-rounded interval sum of the operand
    element and its contributing update elements — exact in ℝ, sound for
    every accumulation order at once (:func:`stelling.interval
    .scatter_add_rows` carries the argument), one outward bump per real
    addition, untouched elements copied exactly.

    Covered forms, exactly (:func:`_scatter_add_row_form`): the two
    measured static-row configurations. Everything else declines:
    non-point (traced/dynamic) indices and out-of-range indices decline
    loudly with the reason quoted (out-of-range handling is
    mode-dependent — FILL_OR_DROP drops, clip clamp-accumulates, both
    measured — and is never guessed); a ``unique_indices=True`` equation
    whose static indices measurably contain duplicates declines loudly
    (the violated promise makes duplicate behaviour
    implementation-defined — :func:`_check_unique_indices_promise`, same
    check at the emission); unrecognized dimension numbers,
    batching dims, a combiner that is not the single ``add``, and
    mismatched shapes decline to a noted ⊤ via the form oracle. Integer
    dtypes route through the overflow-reachability guard exactly as
    ``add``/``reduce_sum`` do: in-range integer accumulation keeps its
    exact result where the bracket resolves it (exact snapping is a
    magnitude-conditional claim — see :func:`_int_overflow_guard`),
    wraparound-reachable accumulation declines with the range quoted.
    """
    if len(ins) != 3:
        return None
    operand, indices, updates = ins
    n = _scatter_add_row_form(
        params, operand.shape, indices.shape, updates.shape,
        _scatter_indices_dtype(eqn),
    )
    if n is None:
        return None
    ks = []
    for lo, hi in zip(indices.los, indices.his):
        if lo != hi or not math.isfinite(lo) or lo != math.floor(lo):
            raise iv.IntervalError(
                f"scatter-add indices are not definite integers over the "
                f"declared box (an index element spans [{lo}, {hi}]) — the "
                f"static-index accumulate rule needs static indices; "
                f"traced/dynamic indices have no row"
            )
        k = int(lo)
        if not 0 <= k < operand.shape[0]:
            raise iv.IntervalError(
                f"scatter-add index {k} is out of range for the operand's "
                f"leading axis {operand.shape[0]} — out-of-range handling "
                f"is mode-dependent (measured on jax 0.11.0: FILL_OR_DROP "
                f"drops the update, clip accumulates into the clamped row) "
                f"and is never guessed"
            )
        ks.append(k)
    _check_unique_indices_promise(params, ks, iv.IntervalError)
    if updates.shape == operand.shape[1:]:
        # the scalar-index sugar: one updates block, normalized to the
        # one-row form (a flat C-order reshape is the identity on elements)
        updates = iv.reshape(updates, (1,) + operand.shape[1:])
    outs = [iv.scatter_add_rows(operand, updates, ks)]
    return _int_overflow_guard(eqn, "scatter-add", outs)


# jax integer arithmetic WRAPS on overflow; every arithmetic transfer here
# computes over ℝ, which does not model wraparound. Modelling int32 as an
# unbounded real is a false-VERIFIED generator — measured, `v * v > 0`
# discharges while jax computes -1794967296 (audit UNSOUND 1).
#
# The fix is an overflow-REACHABILITY guard, not a blanket refusal of
# integers: refusing integers wholesale would ⊤ every index and counter
# computation in every trace. The exact result interval is already in
# hand — it is outward-rounded, hence a SUPERSET of the true integer
# result set — so it is checked against the dtype's representable range:
#
#   * fits  -> no wraparound is reachable over the declared box, the
#              real-arithmetic result IS the integer result, it stands;
#   * escapes -> wraparound is reachable, ⊤ with the range quoted.
#
# Sound in the direction that matters: the check can only be pessimistic
# (the superset may escape a range the true values stay inside), never
# optimistic.
_INT_WRAPAROUND_DECLINE = (
    "{prim!r} on dtype {dtype!r}: jax integer arithmetic wraps on overflow, "
    "which this transfer's real-arithmetic rule does not model — declined"
)

# Three DISTINCT causes, each saying what actually happened (audit
# COSMETIC 5: one sentence was doing duty for all three, and was false for
# two of them).
_INT_OVERFLOW_DECLINE = (
    "{prim!r} on dtype {dtype!r}: integer wraparound is not excluded over "
    "the declared box — the result range reaches [{rlo}, {rhi}], outside "
    "the representable range [{lo}, {hi}]; jax wraps, this domain does not "
    "model it — declined"
)

_INT_UNBOUNDED_DECLINE = (
    "{prim!r} on dtype {dtype!r}: an operand's range is unbounded (⊤, from "
    "an unmodelled producer or an earlier decline), so whether the result "
    "stays inside the representable range [{lo}, {hi}] cannot be decided "
    "at all — declined. This is NOT a wraparound finding"
)

_INT_BRACKET_DECLINE = (
    "{prim!r} on dtype {dtype!r}: the result cannot be resolved against the "
    "representable range [{lo}, {hi}] — it lands within {width} of the "
    "boundary, and the double bracket at this magnitude is itself {width} "
    "wide, so wraparound can be neither shown nor excluded — declined. "
    "This is a BRACKET limit, not a demonstrated overflow"
)


def _snap_int(lo: float, hi: float) -> tuple[float, float]:
    """Tighten an integer-valued result bracket to the integers it can
    actually contain. The true values are integers inside ``[lo, hi]``, so
    ``[ceil(lo), floor(hi)]`` still contains every one of them — exact, and
    it removes the one-ulp outward bump that the arithmetic kernels added
    (audit COSMETIC 5, case 1: a result landing exactly on the dtype
    maximum used to decline although wraparound was provably impossible)."""
    if lo == -math.inf or hi == math.inf or lo != lo or hi != hi:
        return lo, hi
    return math.ceil(lo), math.floor(hi)


def _require_float_dtype(eqn, prim: str) -> None:
    """The blanket float-only guard. Retained for the cases the
    reachability guard cannot decide — a NEGATIVE integer exponent, whose
    jax semantics are not real division at all — never for ordinary
    integer arithmetic, which now goes through
    :func:`_int_overflow_guard`."""
    dt = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    if not dt.startswith("float"):
        raise iv.IntervalError(
            _INT_WRAPAROUND_DECLINE.format(prim=prim, dtype=dt)
        )


def _int_overflow_guard(eqn, prim: str, outs):
    """Guard AND tighten the result of a computing transfer on integers.

    Raises unless every output element provably stays inside the result
    dtype's representable range — three distinct refusals, each attributed
    to what actually happened: an unbounded operand, a bracket too wide to
    resolve at this magnitude, or a genuine escape from the range.

    Returns the outputs with each integer bracket SNAPPED to the integers
    it can contain. The true values of an integer-dtyped result are
    integers, so ``[ceil(lo), floor(hi)]`` still contains every one of them
    — an exact tightening, not an approximation, and it removes the
    one-ulp outward bump the arithmetic kernels added. Without it a counter
    bounded by ``n + 1 <= 2`` could not be discharged although it is
    provably true (audit COSMETIC 5 / over-guard reports: deciding it
    exactly beats declining it).

    The snap yields a POINT (and thereby definite equality verdicts) only
    where one double ulp at the result's magnitude spans at most one
    integer — |result| < 2**53. At int64 magnitudes the arithmetic
    kernels' outward bump itself spans many integers (measured, third
    audit F6: an in-range ``2**62 + 0`` accumulate keeps the snapped
    bracket ``[2**62 - 512, 2**62 + 1024]`` — one outward ulp each way
    at that magnitude — and stays undecided, while ``2**62 + 2**62``
    escapes the range and declines correctly), so "in-range integer
    arithmetic keeps its exact result" is a magnitude-conditional claim:
    exact below 2**53, a sound-but-wide in-range bracket above.
    Deliberate: tightness is never bought at the price of the bracket.

    A no-op for float dtypes, which are returned untouched.
    """
    dtype = (eqn.outvars[0].aval.dtype or "") if eqn.outvars else ""
    if not _is_integer_dtype(dtype):
        return outs  # float (or dtypeless): nothing wraps, nothing to snap
    bounds = _INT_DTYPE_BOUNDS.get(dtype)
    if bounds is None:
        # an integer dtype the table has never heard of: refuse rather than
        # skip the guard silently (the audit's int4/uint4 latent note — the
        # table is now complete, and this keeps it honest if jax adds more)
        raise iv.IntervalError(
            f"{prim!r} on integer dtype {dtype!r}: no representable range is "
            f"registered for it, so integer wraparound cannot be excluded — "
            f"declined"
        )
    lo_b, hi_b = bounds
    snapped = []
    for box in outs:
        los, his = [], []
        for raw_lo, raw_hi in zip(box.los, box.his):
            if raw_lo == -math.inf or raw_hi == math.inf:
                raise iv.IntervalError(
                    _INT_UNBOUNDED_DECLINE.format(
                        prim=prim, dtype=dtype, lo=lo_b, hi=hi_b
                    )
                )
            # the true values are integers: snap the bracket to them first,
            # so a one-ulp bump cannot masquerade as an overflow
            lo, hi = _snap_int(raw_lo, raw_hi)
            # int-vs-float comparison is exact in python, which is why the
            # bounds are kept as ints: float(2**63) would round
            if lo >= lo_b and hi <= hi_b:
                los.append(float(lo))
                his.append(float(hi))
                continue
            # distinguish "the bracket is too wide to resolve here" from
            # "the result genuinely escapes": at int64/uint64 magnitudes one
            # double ulp spans thousands of integers, and that is a bracket
            # limit rather than a demonstrated overflow
            width = max(math.ulp(float(lo_b)), math.ulp(float(hi_b)))
            if width > 1 and (
                (hi > hi_b and hi - hi_b < width)
                or (lo < lo_b and lo_b - lo < width)
            ):
                raise iv.IntervalError(
                    _INT_BRACKET_DECLINE.format(
                        prim=prim, dtype=dtype, lo=lo_b, hi=hi_b,
                        width=int(width),
                    )
                )
            raise iv.IntervalError(
                _INT_OVERFLOW_DECLINE.format(
                    prim=prim, dtype=dtype, rlo=lo, rhi=hi, lo=lo_b, hi=hi_b,
                )
            )
        snapped.append(
            iv.IntervalArray(shape=box.shape, los=tuple(los), his=tuple(his))
        )
    return snapped


def _int_guarded(prim: str, compute):
    """Wrap an arithmetic transfer with the overflow-reachability guard.
    The marker attribute is what the census assert below checks, so
    membership is enforced rather than declared."""

    def t(eqn, params, ins):
        outs = compute(eqn, params, ins)
        if outs is None:
            return None
        return _int_overflow_guard(eqn, prim, outs)

    t._int_guarded = True
    return t


def _integer_exponent(params) -> int | None:
    """The ``y`` param as a genuine integer, or ``None`` to decline. bool is
    an int subclass in Python and is NOT an exponent."""
    y = params.get("y")
    if isinstance(y, bool) or not isinstance(y, int):
        return None
    return y


# Primitives that do NOT pass the product-derived taint on. The default is
# to propagate through EVERYTHING: an exemption is a claim that no
# compiler, under any simplification set, could present the upstream
# product to a later add/sub as a raw addend through this primitive. Each
# is a separate flagged judgement call, and the standing posture where the
# argument is not airtight is to propagate.
#
#  * `exp` — a transcendental. `exp(a*b)` is not a product of anything the
#    compiler holds; no fused multiply-add can absorb the original
#    multiply through it. (Deliberately NOT extended to `pow` or
#    `integer_pow`: `pow(x, 2)` is a multiply after expansion, which is
#    precisely the kind of simplification this finding is about.)
#
#  * the comparisons and boolean logic — their outputs are BOOLEANS, not
#    float addends at all. A bool cannot be an fma operand; any later
#    float use goes through a conversion whose value is 0 or 1, not a
#    product. The chain genuinely restarts.
_TAINT_STOPS = frozenset({
    "exp",
    "lt", "gt", "le", "ge", "eq", "ne", "and", "or", "reduce_or",
})

# The primitives whose OUTPUT can be a product XLA is free to contract into a
# following addition. This is the source end of the same hazard
# IEEE_CONTRACTION_ADDENDS guards at the sink end, and it was `mul` alone.
#
# `dot_general` and `integer_pow` are here because they are products too, and
# leaving them out was LATENT rather than safe. Measured on jax 0.11.0 CPU
# under jit: `dot_general(a, b) + c` with a size-1 contraction compiles to
# `ROOT %bitcast_add_fusion` and `a**2 + c` to `ROOT %multiply_add_fusion`,
# both executing to 4.930380657631324e-32 where two separate roundings give
# 0.0 — the same value that made `add_any`'s omission a false discharge.
#
# They did not produce one only because neither has an ieee transfer today, so
# both decline to ⊤ before the taint is ever consulted. That is protection by
# an unrelated decline, which is exactly the shape the scatter bar's
# unreachability had: correct today, and silently wrong the moment someone
# adds the missing ieee rule. Over-tainting is SOUND — it makes more hulls
# fire, never fewer — so they go in now rather than as a comment for later.
# The DROPPED-assume disclosure, in ONE place because inert mode and constrain
# mode must emit the same base text (inert adds no reason parenthetical) and
# they were two hand-written copies with a comment asking the next editor to
# keep them byte-identical. A test pinned the string a third time. Three copies
# of one sentence is the by-name-gate shape in prose.
#
# It names REFUTED explicitly. It used to cover VERIFIED and UNKNOWN only, and
# REFUTED is the case where the drop actually costs the reader something:
# measured, `assume(jnp.all(x >= 0))` over x in [-10, 10] asserting sum(x) >= 0
# returns REFUTED with the replay-confirmed witness [0, 0, -1] — a
# counterexample that VIOLATES the dropped precondition. Sound for what it
# claims (a counterexample to the query without the assumption) and useless as
# a counterexample to the query the author wrote, which is what a reader takes
# a witness to be.
ASSUME_DROP_NOTE = (
    "assume constraint DROPPED (inert in MVP propagation) at {where}: "
    "VERIFIED proves a superset; UNKNOWN may be confounded by this drop; "
    "and a REFUTED WITNESS MAY VIOLATE THE DROPPED PRECONDITION \u2014 it is a "
    "counterexample to the query WITHOUT the assumption, so check it against "
    "the precondition before treating it as one"
)

IEEE_PRODUCT_SOURCES = frozenset({"mul", "dot_general", "integer_pow"})

# Every registered transfer that PERFORMS an addition, and therefore every
# place a product can be contracted into. This is the complete candidate set
# for the sink gate, kept separate from IEEE_CONTRACTION_ADDENDS because the
# two are not the same list and the difference is the whole point:
#
#   add, sub, add_any   in the sink gate; hulled
#   reduce_sum          handled by its OWN branch (a two-element reduction IS
#                       an addition), which is a third site for one hazard
#   dot_general         NOT hulled -- it declines in ieee mode
#   scatter-add         NOT hulled -- it declines in ieee mode
#
# The last two are LATENT, measured not assumed: `c.at[0].add(a*b)` compiles to
# ROOT bitcast_add_fusion and executes to 4.930380657631324e-32 against a
# two-rounding 0.0. They are safe today only because their ieee transfers
# decline before the hull is reached, which is protection by an unrelated
# decline -- the scatter bar's shape, and it evaporates the moment either gets
# an ieee rule.
#
# So the invariant a new adding transfer must satisfy is a DISJUNCTION:
# hulled, or declining in ieee mode. tests/test_by_name_gates.py checks it
# behaviourally, because "declines in ieee" is not a membership fact and no
# import-time census can see it.
IEEE_ADDITION_PERFORMERS = frozenset({
    "add", "sub", "add_any", "reduce_sum", "dot_general", "scatter-add",
})


def _integer_pow_budget(box, y: int) -> None:
    """Degrade-don't-HANG, both dimensions. The exponent cap bounds the
    per-element cost; the work cap bounds ``size x |y|``, which is what an
    elementwise transfer actually pays (audit FRAGILE 3)."""
    if abs(y) > iv.INTEGER_POW_EXACT_CAP:
        raise iv.IntervalError(
            iv.INTEGER_POW_CAP_DECLINE.format(
                n=abs(y), cap=iv.INTEGER_POW_EXACT_CAP
            )
        )
    work = box.size * abs(y)
    if work > iv.INTEGER_POW_WORK_CAP:
        raise iv.IntervalError(
            iv.INTEGER_POW_WORK_DECLINE.format(
                size=box.size, n=abs(y), work=work,
                cap=iv.INTEGER_POW_WORK_CAP,
            )
        )


def _t_div(eqn, params, ins):
    """``div``. On floats this is real division, unchanged. On INTEGERS it
    is not: jax integer division TRUNCATES toward zero (measured:
    ``lax.div(-7, 2) = -3``, not −3.5) and ``INT_MIN / -1`` WRAPS (measured:
    ``lax.div(-2**31, -1) = -2147483648``, not +2³¹). Modelling either as
    real division mints false definite verdicts in both directions — audit
    UNSOUND 3, and the second of those is literally the wraparound class
    the overflow guard exists for, so it routes through the same guard."""
    dtype = (eqn.outvars[0].aval.dtype or "") if eqn.outvars else ""
    if not _is_integer_dtype(dtype):
        return [iv.div(*ins)]
    return _int_overflow_guard(eqn, "div", [iv.int_div(*ins)])


def _t_reduce_sum(eqn, params, ins):
    outs = [iv.reduce_sum(ins[0], tuple(_req(params, "axes", "reduce_sum")))]
    return _int_overflow_guard(eqn, "reduce_sum", outs)


def _t_dot_general(eqn, params, ins):
    """Interval transfer for ``dot_general``, gated by the shared oracle.

    Declines to ⊤ (``None``) on anything the oracle refuses. The oracle's
    reason is not narrated here: a ⊤ is recorded by the coverage tool with
    the primitive named, and the *emission* face is where a caller sees the
    prose, so surfacing it twice in two vocabularies would invite them to
    drift apart.

    Censused ``_INT_COMPUTING``, following ``sqrt``'s precedent exactly: a
    primitive that COMPUTES a new value is classified computing even when
    it is float-only, and its gate is what closes the integer class. Here
    the oracle declines integral ``preferred_element_type`` outright and
    declines an absent one over integer operands, so an integer-dtyped
    output cannot reach the arithmetic — the honest way a computing float
    op closes the class, rather than by claiming it computes nothing.
    """
    if len(ins) != 2 or len(eqn.invars) != 2:
        return None
    lhs_dt = eqn.invars[0].aval.dtype or ""
    rhs_dt = eqn.invars[1].aval.dtype or ""
    dn, _reason = _dot_general_row_form(params, lhs_dt, rhs_dt)
    if dn is None:
        return None
    try:
        return [iv.dot_general(ins[0], ins[1], dn)]
    except iv.IntervalError:
        # shapes the oracle admitted but the domain refuses: ⊤ rather than
        # a crash, same posture as every other transfer
        return None


def _t_integer_pow(eqn, params, ins):
    y = _integer_exponent(params)
    if y is None:
        return None  # non-integer exponent: no rule, ⊤ with the params noted
    # degrade-don't-HANG in BOTH dimensions: the exact-rational endpoints
    # cost time linear in the exponent (jax bounds it nowhere) and are paid
    # per element (audit FRAGILE 2 and 3)
    _integer_pow_budget(ins[0], y)
    if y < 0:
        # a negative exponent over integers is not real division: jax's
        # integer semantics here are not modelled at all, so the
        # reachability guard has nothing to decide and the blanket float
        # guard is the honest one
        _require_float_dtype(eqn, "integer_pow")
    outs = [iv.integer_pow(ins[0], y)]
    return _int_overflow_guard(eqn, "integer_pow", outs)


def _t_sqrt(eqn, params, ins):
    """``sqrt`` — real square root, a FLOAT-only primitive (jax's ``sqrt``
    rejects integer dtypes at trace time, so an integer sqrt is IR no trace
    can carry). The blanket float-only guard declines an integer/bool
    operand loudly rather than modelling a wrapping integer as a real — the
    same posture the negative integer_pow exponent takes, and the reason
    this transfer is classified computing yet declines every integer probe.

    On floats the monotone outward-rounded transfer runs; the domain refusal
    (``arg >= 0``, the obligation a sqrt call carries) is raised inside
    :func:`stelling.interval.sqrt` where a below-0 lower bound reaches the
    out-of-domain region, and the walk turns that :class:`IntervalError`
    into a noted top-decline."""
    _require_float_dtype(eqn, "sqrt")
    return [iv.sqrt(ins[0])]


TRANSFERS = {
    # the arithmetic core carries the integer overflow-reachability guard
    # (audit UNSOUND 1): in-range integer arithmetic keeps its exact real
    # result, wraparound-reachable arithmetic declines with the range
    # quoted. A no-op on every float dtype.
    "add": (_int_guarded("add", lambda eqn, p, ins: [iv.add(*ins)]), TIER_SOUND),
    # `add_any` is NOT an alias of `add` in jax -- it is a separate
    # Primitive("add_any") whose abstract_eval asserts core.typematch(x, y)
    # and returns x, so it admits NO promotion and NO broadcasting where
    # `add` admits both. What it does to the VALUES is addition (jax binds
    # it to accumulate cotangents), so the interval rule is `add`'s rule
    # and it carries the same overflow guard under its own name.
    "add_any": (
        _int_guarded("add_any", lambda eqn, p, ins: [iv.add(*ins)]),
        TIER_SOUND,
    ),
    "sub": (_int_guarded("sub", lambda eqn, p, ins: [iv.sub(*ins)]), TIER_SOUND),
    "mul": (_int_guarded("mul", lambda eqn, p, ins: [iv.mul(*ins)]), TIER_SOUND),
    # real division on floats; TRUNCATING integer division, exactly, on
    # integers — with INT_MIN/-1 routed through the overflow guard
    "div": (_t_div, TIER_SOUND),
    # neg/abs wrap at INT_MIN only (-(-2**31) is -2**31 in two's
    # complement), which the same guard catches
    "neg": (_int_guarded("neg", lambda eqn, p, ins: [iv.neg(ins[0])]), TIER_EXACT),
    "abs": (_int_guarded("abs", lambda eqn, p, ins: [iv.abs_(ins[0])]), TIER_EXACT),
    "max": (lambda eqn, p, ins: [iv.maximum(*ins)], TIER_EXACT),
    "min": (lambda eqn, p, ins: [iv.minimum(*ins)], TIER_EXACT),
    # pow's corner rule holds for strictly positive bases only; anything
    # else declines inside iv.pow_ (IntervalError -> noted ⊤).
    "pow": (_int_guarded("pow", lambda eqn, p, ins: [iv.pow_(*ins)]),
            TIER_SOUND_LIBM),
    "stop_gradient": (lambda eqn, p, ins: [ins[0]], TIER_EXACT),
    "reshape": (_t_reshape, TIER_EXACT),
    # select_n(which, *cases): the masked case-split. Exact where `which` is
    # definite; a sound join where it straddles (design/control-flow-*).
    "select_n": (lambda eqn, p, ins: [iv.select_n(ins[0], ins[1:])], TIER_EXACT),
    "exp": (_int_guarded("exp", lambda eqn, p, ins: [iv.exp(ins[0])]),
            TIER_SOUND_LIBM),
    # sqrt: real square root — monotone increasing on [0, inf), outward-
    # rounded. A CORRECTLY-ROUNDED IEEE basic op (tier sound, NO libm
    # demotion, unlike exp/pow). Float-only (declines on integers via the
    # blanket guard, so it declines every integer probe rather than snapping
    # a non-integer result). Domain arg >= 0 is the OBLIGATION: a below-0
    # lower bound declines inside iv.sqrt (the pow domain posture). census
    # tier-1b addition — sqrt is the root of the L2 residual norm in every
    # convergence diagnostic and a node hazard (face-area/two-norm) across
    # the corpus; the authorized core-row build.
    "sqrt": (_t_sqrt, TIER_SOUND),
    "lt": (lambda eqn, p, ins: [iv.lt(*ins)], TIER_EXACT),
    "gt": (lambda eqn, p, ins: [iv.gt(*ins)], TIER_EXACT),
    "le": (lambda eqn, p, ins: [iv.le(*ins)], TIER_EXACT),
    "ge": (lambda eqn, p, ins: [iv.ge(*ins)], TIER_EXACT),
    "eq": (lambda eqn, p, ins: [iv.eq(*ins)], TIER_EXACT),
    "ne": (lambda eqn, p, ins: [iv.ne(*ins)], TIER_EXACT),
    "and": (_t_bool_logic("and", iv.logical_and), TIER_EXACT),
    "or": (_t_bool_logic("or", iv.logical_or), TIER_EXACT),
    "reduce_or": (_t_reduce_or, TIER_EXACT),
    # the sum over the reduced axes. Sound under ℝ for EVERY association
    # order at once (in ℝ they all denote the same number); the ieee
    # counterpart cannot reuse that and declines — see IEEE_TRANSFERS.
    "reduce_sum": (_t_reduce_sum, TIER_SOUND),
    "dot_general": (_t_dot_general, TIER_SOUND),
    # x ** y for integer y. Even y > 0 PRODUCES non-negativity; y < 0
    # routes through div's zero-in-divisor discipline (⊤ when the base
    # straddles 0 — the pole is real and nothing here papers over it).
    "integer_pow": (_t_integer_pow, TIER_SOUND),
    "squeeze": (
        lambda eqn, p, ins: [iv.squeeze(ins[0], tuple(p.get("dimensions", ())))],
        TIER_EXACT,
    ),
    "split": (_t_split, TIER_EXACT),
    "slice": (
        lambda eqn, p, ins: [
            iv.slice_(
                ins[0],
                tuple(_req(p, "start_indices", "slice")),
                tuple(_req(p, "limit_indices", "slice")),
                tuple(p["strides"]) if p.get("strides") else None,
            )
        ],
        TIER_EXACT,
    ),
    # x.at[k].set(v), static index only — census addition from the maddening
    # HeatNode trace; every other scatter configuration declines (see
    # _t_scatter).
    "scatter": (_t_scatter, TIER_EXACT),
    # the ACCUMULATE scatter (what jax.ops.segment_sum and x.at[idx].add(v)
    # lower to), static indices only — census addition from the MIME LSQ
    # normal-system census round; duplicate indices accumulate, which is
    # why this is its own row and NOT the set-form 'scatter' above (see
    # _t_scatter_add). Outward-rounded accumulation: tier sound.
    "scatter-add": (_t_scatter_add, TIER_SOUND),
    # k same-shape arrays joined along a new axis (what jnp.stack traces
    # to on jax 0.11.0) — same census round as scatter-add; pure element
    # routing, no arithmetic; malformed axes/shapes decline inside
    # iv.stack (IntervalError -> noted ⊤).
    "stack": (
        lambda eqn, p, ins: [iv.stack(list(ins), int(_req(p, "axis", "stack")))],
        TIER_EXACT,
    ),
    # x[idx], static-index leading-axis row form only — census addition from
    # the MIME fvm laplacian census trace; every other gather configuration
    # declines (see _t_gather).
    "gather": (_t_gather, TIER_EXACT),
    # axis permutation: pure data movement (MIME fvm census round, reached
    # inside the transparent jnp.linalg.inv jit); malformed permutations
    # decline inside iv.transpose (IntervalError -> noted ⊤).
    "transpose": (
        lambda eqn, p, ins: [
            iv.transpose(ins[0], tuple(p.get("permutation", ()) or ()))
        ],
        TIER_EXACT,
    ),
    "broadcast_in_dim": (
        lambda eqn, p, ins: [
            iv.broadcast_in_dim(
                ins[0],
                tuple(_req(p, "shape", "broadcast_in_dim")),
                tuple(_req(p, "broadcast_dimensions", "broadcast_in_dim")),
            )
        ],
        TIER_EXACT,
    ),
    "concatenate": (
        lambda eqn, p, ins: [
            iv.concatenate(list(ins), int(_req(p, "dimension", "concatenate")))
        ],
        TIER_EXACT,
    ),
    "convert_element_type": (_t_convert, TIER_EXACT),
    "stelling_any": (
        lambda eqn, p, ins: [
            iv.from_bounds(
                tuple(_req(p, "shape", "stelling_any")),
                float(_req(p, "lo", "stelling_any")),
                float(_req(p, "hi", "stelling_any")),
            )
        ],
        TIER_EXACT,
    ),
    "stelling_assert": (lambda eqn, p, ins: [ins[0]], TIER_EXACT),
    "stelling_nonvacuity": (lambda eqn, p, ins: [ins[0]], TIER_EXACT),
}


# -- the integer-semantics census over the TRANSFER sites ---------------------
#
# The first sweep for this defect class was run over the EMISSION sites and
# not over the transfer sites, and `div` fell through the gap (audit
# UNSOUND 3): the cleared-list entry "div — already stricter" was true of
# the emission and false of the transfer, and interval propagation mints
# definite statuses without ever reaching the emission. A review found the
# siblings once and then did not find them again.
#
# So the classification is mechanised instead of remembered. Every
# registered transfer is either COMPUTING — it can produce a numeric value
# its operands did not contain, so it carries the overflow-reachability
# guard — or NON-COMPUTING, with the reason recorded here. The assert makes
# the census TOTAL: a new transfer cannot be registered without landing in
# one of the two sets, exactly as IEEE_TRANSFERS cannot be left short.
_INT_COMPUTING = frozenset({
    "add", "sub", "mul", "div", "neg", "abs",
    "pow", "exp", "reduce_sum", "integer_pow",
    # sqrt COMPUTES a new value on floats (it is not a selector, router, or
    # identity), so it is classified computing and probed. It is float-only
    # (jax's sqrt rejects integer dtypes), so its transfer DECLINES every
    # integer probe via the blanket float-only guard — the honest way a
    # computing float op closes the integer class, and strictly the
    # negative-integer_pow-exponent posture applied to the whole domain.
    "sqrt",
    # the accumulate scatter SUMS (operand element + contributing update
    # elements), so it can produce a value its operands did not contain —
    # exactly the add/reduce_sum class, same guard
    "scatter-add",
    # `add_any` sums two operands exactly as `add` does, so it can produce a
    # value neither contained; same class, same guard, probed the same way
    "add_any",
    # the contraction SUMS products, so it too can produce a value its
    # operands did not contain. Float-only in practice because the shared
    # oracle refuses integral accumulation (and an ABSENT
    # preferred_element_type over integer operands, which follows the
    # operands and would wrap) — sqrt's posture, applied to a binary op.
    "dot_general",
})

# Why each of these cannot introduce an out-of-range integer:
#   max/min/select_n  select an operand value; they compute nothing
#   comparisons/and/or/reduce_or  produce bools, exact for integers
#   convert_element_type  carries its own whitelist + range guard
#   the structural ops  are pure data movement (copies of in-range values)
#   the harness primitives  are declarations and identities
_INT_NON_COMPUTING = frozenset({
    # `split` is pure data movement: every output element IS an input element
    # at a static index, so it cannot introduce an out-of-range integer
    "split",
    "max", "min", "select_n",
    "lt", "gt", "le", "ge", "eq", "ne", "and", "or", "reduce_or",
    "convert_element_type",
    "stop_gradient", "reshape", "squeeze", "slice", "scatter", "gather",
    "transpose", "broadcast_in_dim", "concatenate", "stack",
    "stelling_any", "stelling_assert", "stelling_nonvacuity",
})

if _INT_COMPUTING | _INT_NON_COMPUTING != set(TRANSFERS):
    raise RuntimeError(
    "the integer-semantics census must stay total over TRANSFERS: "
    f"unclassified {set(TRANSFERS) - _INT_COMPUTING - _INT_NON_COMPUTING}, "
    f"stale {_INT_COMPUTING | _INT_NON_COMPUTING - set(TRANSFERS)}"
)
if _INT_COMPUTING & _INT_NON_COMPUTING:
    raise RuntimeError(
        "a primitive cannot both compute and not compute an integer "
        f"value: {sorted(_INT_COMPUTING & _INT_NON_COMPUTING)}"
    )

# -- the probe-or-exempt census over the CLASSIFICATION itself ----------------
#
# Third audit, F3: the behavioural boundary sweep probes only the names
# CLASSIFIED computing, so a future two-edit misfiling — an arithmetic
# primitive given a transfer and filed _INT_NON_COMPUTING (plus its ieee
# row) — passed every import-time assert and would mint false VERIFIEDs
# on wrapping integer arithmetic (demonstrated on an int32 cumsum, scratch
# edits reverted). The classification is therefore itself censused: every
# registered transfer is either PROBED (in _INT_COMPUTING, swept at every
# dtype boundary in both directions) or EXEMPT with a written soundness
# reason below. A silent two-edit misfiling is now a conscious three-edit
# act whose third edit is a soundness claim in this registry.
_INT_NON_COMPUTING_EXEMPT: dict[str, str] = {
    "split": (
        "cuts one operand along one axis at statically known offsets; every "
        "output element IS an input element, so no arithmetic occurs and an "
        "in-range integer cannot be moved out of range by being copied"
    ),
    "max": (
        "selects one of its operands' values elementwise; no arithmetic "
        "creates a value its in-range operands did not already contain"
    ),
    "min": (
        "selects one of its operands' values elementwise; no arithmetic "
        "creates a value its in-range operands did not already contain"
    ),
    "select_n": (
        "selects/joins operand values elementwise (measured clamp "
        "semantics); it computes no new numeric value"
    ),
    "lt": "produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap",
    "gt": "produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap",
    "le": "produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap",
    "ge": "produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap",
    "eq": "produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap",
    "ne": "produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap",
    "and": (
        "bool-only by its own dtype guard (the bitwise integer form "
        "declines inside the transfer); Kleene logic on {0, 1}"
    ),
    "or": (
        "bool-only by its own dtype guard (the bitwise integer form "
        "declines inside the transfer); Kleene logic on {0, 1}"
    ),
    "reduce_or": (
        "bool-only by its own dtype guard; a three-valued OR-fold whose "
        "outputs are booleans"
    ),
    "convert_element_type": (
        "carries its own exact-conversions whitelist plus the float->int "
        "range guard and the interval-based int64->int32 index-narrowing "
        "range check; every value-changing conversion declines"
    ),
    "stop_gradient": "the identity on its operand",
    "reshape": (
        "pure element routing (flat C-order identity) — misclassification "
        "would require the routing kernel itself to compute, which it "
        "cannot: it only copies in-range values"
    ),
    "squeeze": (
        "pure element routing (axis removal) — copies of in-range values, "
        "no arithmetic performed on them"
    ),
    "slice": (
        "pure element routing (static selection) — copies of in-range "
        "values, no arithmetic performed on them"
    ),
    "scatter": (
        "element REPLACEMENT (the set form): the output holds only values "
        "its operand and update already contained; the accumulate sibling "
        "'scatter-add' computes and is probed"
    ),
    "gather": (
        "pure element routing (static row take) — copies of in-range "
        "values, no arithmetic performed on them"
    ),
    "transpose": (
        "pure element routing (axis permutation) — copies of in-range "
        "values, no arithmetic performed on them"
    ),
    "broadcast_in_dim": (
        "pure element routing (replication) — copies of in-range values, "
        "no arithmetic performed on them"
    ),
    "concatenate": (
        "pure element routing (adjacency) — copies of in-range values, no "
        "arithmetic performed on them"
    ),
    "stack": (
        "pure element routing (new-axis join) — copies of in-range "
        "values, no arithmetic performed on them"
    ),
    "stelling_any": (
        "a declaration: its output box IS the declared bounds; nothing is "
        "computed from operand values"
    ),
    "stelling_assert": "the identity on its predicate operand",
    "stelling_nonvacuity": "the identity on its membership operand",
}


def _assert_integer_classification_censused(
    registered=None, probed=None, exemptions=None
) -> None:
    """Probe-or-exempt (third audit, F3): every primitive with a
    registered transfer must be either covered by the behavioural integer
    boundary sweep (classified ``_INT_COMPUTING``) or present in
    :data:`_INT_NON_COMPUTING_EXEMPT` with a non-empty written reason —
    and every exemption must name a live, unprobed primitive (a stale
    exemption is a soundness claim about nothing, the fidelity module's
    stale-residual discipline applied here). Callable with explicit
    arguments so the regression test can doctor a copy in-process; the
    import-time call runs on the live registries."""
    registered = set(TRANSFERS) if registered is None else set(registered)
    probed = set(_INT_COMPUTING) if probed is None else set(probed)
    exemptions = (
        dict(_INT_NON_COMPUTING_EXEMPT)
        if exemptions is None
        else dict(exemptions)
    )
    for prim in sorted(registered):
        if prim in probed:
            continue
        reason = exemptions.get(prim)
        if not isinstance(reason, str) or not reason.strip():
            raise AssertionError(
                f"integer-classification census: primitive {prim!r} has a "
                f"registered transfer, is not covered by the behavioural "
                f"integer boundary sweep, and carries no written exemption "
                f"reason — the CLASSIFICATION itself must be censused: "
                f"either classify it computing (probed at every dtype "
                f"boundary in both directions) or write its soundness "
                f"claim into _INT_NON_COMPUTING_EXEMPT"
            )
    stale = sorted(set(exemptions) - (registered - probed))
    if stale:
        raise AssertionError(
            f"integer-classification census: exemption entr"
            f"{'ies' if len(stale) > 1 else 'y'} {stale} name no live "
            f"unprobed primitive — a stale exemption is a soundness claim "
            f"about nothing; delete or correct it"
        )


_assert_integer_classification_censused()

# and every computing transfer must actually be wearing the guard — the
# census is only worth having if membership is enforced rather than
# declared — and DECLARED is what the first version of this assert was
# (audit COSMETIC 6): it read a settable marker attribute plus a
# hand-maintained escape list, either of which a computing transfer could
# satisfy while leaving the class wide open. An assert that can pass while
# the invariant fails is worse than no assert, because it licenses trust.
#
# So the check is BEHAVIOURAL: each computing transfer is actually run on
# an out-of-range integer operand, and must either decline or return a
# result inside the dtype's range. Nothing about how it is implemented,
# labelled or wrapped can satisfy this — only doing the right thing can.
# RETIRED, deliberately kept and deliberately empty. This was the
# hand-maintained escape list the declarative assert consulted; the escape
# list WAS the defect, so it is not repaired but abolished — and a name
# that once granted exemptions is left here at zero so that re-adding one
# is a visible act rather than a quiet edit to a live list.
_INT_GUARDED_INSIDE: frozenset[str] = frozenset()

def _probe_operands(prim: str, lo_b: int, hi_b: int, high: bool):
    """Operand values chosen to push ``prim`` past the named boundary.
    ``None`` for the second slot means the primitive is unary. For
    ``scatter-add`` the two slots are (operand element, update element) —
    the accumulation ``boundary + boundary`` escapes the range in both
    directions except at an unsigned lower bound of 0, where the in-range
    exact result must stand."""
    hi, lo = float(hi_b), float(lo_b)
    if high:
        return {
            "add": (hi, hi), "sub": (hi, lo), "mul": (hi, hi),
            "div": (lo, -1.0), "neg": (lo, None), "abs": (lo, None),
            "pow": (hi, 2.0), "exp": (hi, None), "sqrt": (hi, None),
            "reduce_sum": (hi, None), "integer_pow": (hi, None),
            "scatter-add": (hi, hi),
            "add_any": (hi, hi),
            # the contraction's products escape exactly as `mul`'s do, so
            # it takes mul's operand values
            "dot_general": (hi, hi),
        }[prim]
    return {
        "add": (lo, lo), "sub": (lo, hi), "mul": (lo, hi),
        "div": (hi, -1.0), "neg": (hi, None), "abs": (hi, None),
        "pow": (lo, 3.0), "exp": (lo, None), "sqrt": (lo, None),
        "reduce_sum": (lo, None), "integer_pow": (lo, None),
        "scatter-add": (lo, lo),
        "add_any": (lo, lo),
        "dot_general": (lo, hi),
    }[prim]


def _probe_slice(dtype: str, high: bool):
    """One (dtype, direction) slice of the sweep as concrete operand
    boxes-to-be — a readable summary of the shapes :func:`_probe_operands`
    generates, DERIVED from it so the two cannot drift."""
    lo_b, hi_b = _INT_DTYPE_BOUNDS[dtype]
    out = {}
    for prim in sorted(_INT_COMPUTING):
        first, second = _probe_operands(prim, lo_b, hi_b, high)
        out[prim] = (
            (first, first),
            (second, second) if second is not None else None,
        )
    return out


# The int32 / upper-boundary slice, materialised under its own name. This
# used to BE the census — one dtype, one direction per primitive — which is
# exactly the narrowness the audit named: a transfer could decline the one
# probed value while accepting a sibling. It is kept only as a readable
# summary; the census itself sweeps every dtype in both directions.
_INT_PROBE_OPERANDS = _probe_slice("int32", True)


def _assert_computing_transfers_close_the_integer_class() -> None:
    """The behavioural census: run every computing transfer at every
    integer dtype's boundary, in BOTH directions, and require it to decline
    or return a result inside that dtype's representable range.

    Nothing about how a transfer is implemented, labelled or wrapped can
    satisfy this — only doing the right thing can. Widened from a single
    int32 probe per primitive to the whole invariant (audit: a transfer
    could decline the one probed value while accepting a sibling; the
    assert should test the invariant, not a representative of it). Kept at
    import rather than in a test because it runs for every consumer of the
    module, not only when the suite runs — measured cost of the full sweep
    is ~2 ms.
    """
    declines = 0
    for prim in sorted(_INT_COMPUTING):
        exponents = ((2,), (3,)) if prim == "integer_pow" else ((None,),)
        for dtype, (lo_b, hi_b) in sorted(_INT_DTYPE_BOUNDS.items()):
            for high in (True, False):
                y = (2 if high else 3) if prim == "integer_pow" else None
                first, second = _probe_operands(prim, lo_b, hi_b, high)
                if prim == "scatter-add":
                    # the 3-operand accumulate form: one operand row at the
                    # boundary, one in-range index (0), one update at the
                    # boundary — the accumulation boundary+boundary escapes
                    # the range (except unsigned low, where 0+0 must stand
                    # exactly), and the transfer must decline it
                    boxes = [
                        iv.from_bounds((1,), first, first),
                        iv.point(0.0, (1, 1)),
                        iv.from_bounds((1,), second, second),
                    ]
                    avals = (
                        ir.Aval(kind="ShapedArray", shape=(1,), dtype=dtype),
                        ir.Aval(
                            kind="ShapedArray", shape=(1, 1), dtype="int32"
                        ),
                        ir.Aval(kind="ShapedArray", shape=(1,), dtype=dtype),
                    )
                    invars = tuple(
                        ir.Var(id=i, aval=a) for i, a in enumerate(avals)
                    )
                    eqn = ir.JaxprEqn(
                        primitive=prim, invars=invars,
                        outvars=(ir.Var(id=99, aval=avals[0]),),
                        params=(
                            (
                                "dimension_numbers",
                                ir.NamedTupleParam(
                                    cls="ScatterDimensionNumbers",
                                    fields=(
                                        ("update_window_dims", ()),
                                        ("inserted_window_dims", (0,)),
                                        ("scatter_dims_to_operand_dims", (0,)),
                                        ("operand_batching_dims", ()),
                                        ("scatter_indices_batching_dims", ()),
                                    ),
                                ),
                            ),
                        ),
                    )
                elif prim == "dot_general":
                    # the 2-operand contraction form, at integer dtypes and
                    # with NO preferred_element_type -- which is what jax
                    # emits for an integer matmul, and which makes the
                    # accumulation follow the operands and WRAP. The shared
                    # oracle must decline it. Probing the absent-param form
                    # deliberately: an absent param is the class that has
                    # produced three defects in this codebase, and the one
                    # this row's own oracle nearly shipped a hole on.
                    avals = (
                        ir.Aval(kind="ShapedArray", shape=(1,), dtype=dtype),
                        ir.Aval(kind="ShapedArray", shape=(1,), dtype=dtype),
                    )
                    boxes = [
                        iv.from_bounds((1,), first, first),
                        iv.from_bounds((1,), second, second),
                    ]
                    invars = tuple(
                        ir.Var(id=i, aval=a) for i, a in enumerate(avals)
                    )
                    eqn = ir.JaxprEqn(
                        primitive=prim, invars=invars,
                        outvars=(
                            ir.Var(
                                id=99,
                                aval=ir.Aval(
                                    kind="ShapedArray", shape=(), dtype=dtype
                                ),
                            ),
                        ),
                        params=(
                            ("dimension_numbers", (((0,), (0,)), ((), ()))),
                            ("precision", None),
                            ("preferred_element_type", None),
                            ("out_sharding", None),
                        ),
                    )
                else:
                    shape = (3,) if prim == "reduce_sum" else ()
                    aval_in = ir.Aval(
                        kind="ShapedArray", shape=shape, dtype=dtype
                    )
                    boxes = [iv.from_bounds(shape, first, first)]
                    if second is not None:
                        boxes.append(iv.from_bounds((), second, second))
                    invars = tuple(
                        ir.Var(
                            id=i,
                            aval=aval_in if i == 0 else ir.Aval(
                                kind="ShapedArray", shape=(), dtype=dtype
                            ),
                        )
                        for i in range(len(boxes))
                    )
                    params = {
                        "integer_pow": (("y", y),),
                        "reduce_sum": (("axes", (0,)),),
                    }
                    eqn = ir.JaxprEqn(
                        primitive=prim, invars=invars,
                        outvars=(ir.Var(
                            id=99,
                            aval=ir.Aval(
                                kind="ShapedArray", shape=(), dtype=dtype
                            ),
                        ),),
                        params=params.get(prim, ()),
                    )
                try:
                    outs = TRANSFERS[prim][0](eqn, eqn.params_dict(), boxes)
                except iv.IntervalError:
                    declines += 1
                    continue  # declined: the class is closed here
                if outs is None:
                    declines += 1
                    continue  # no rule for this configuration: also closed
                for box in outs:
                    for lo, hi in zip(box.los, box.his):
                        assert lo >= lo_b and hi <= hi_b, (
                            f"computing transfer {prim!r} on {dtype!r} "
                            f"returned {[lo, hi]}, outside its range "
                            f"[{lo_b}, {hi_b}], without declining — the "
                            f"integer class is open"
                        )
    # the sweep must actually BITE, or it would pass by never reaching the
    # guard at all
    assert declines > 0, "the integer census probed nothing that declined"


_assert_computing_transfers_close_the_integer_class()


# Every sound-libm transfer names the exact libm-fidelity assumption it
# rides on; the tier check below stamps it into the verdict. A transfer at
# that tier without an entry still stamps a (generic) assumption — the
# demotion is never silent.
_LIBM_ASSUMPTIONS = {
    "exp": iv.EXP_LIBM_ASSUMPTION,
    "pow": iv.POW_LIBM_ASSUMPTION,
}


# -- the ieee transfer registry -----------------------------------------------
#
# The censused counterpart of TRANSFERS for semantics="ieee": every key of
# TRANSFERS appears here explicitly — (i) verified sound as-is for float
# (pure data movement, structural ops — wrapped only to route the
# maybe-NaN flag), (ii) given an ieee variant (the arithmetic core with
# native binary64 endpoints and NaN-corner routing; comparisons with the
# NaN-falsification rule; anything that touched the extended-real
# conventions — the real-mode 0·∞ = 0 inside iv.mul is exactly what is
# NOT reused), or (iii) declined to ⊤-maybe-NaN with the gap quoted.
# No silent reuse; the registry is the census, mirrored in the build
# report.
#
# Signature: t(eqn, params, ins, flags) -> (outs, out_flags) | None
# where flags/out_flags are the per-array maybe-NaN flags. None and
# IntervalError decline exactly as in real mode, except ⊤ outputs carry
# maybe_nan=True (⊤ under ieee is maybe-NaN: an unmodeled value could be
# anything a float can be, including NaN).


def _ieee_f64_only(eqn) -> None:
    """The binary64 guard for the arithmetic core: ieee endpoints are
    native binary64, which does not model float32/float16 rounding or
    integer wraparound — any other dtype declines with the gap quoted."""
    bad = sorted(
        {v.aval.dtype for v in (*eqn.invars, *eqn.outvars)} - {"float64"}
    )
    if bad:
        raise iv.IntervalError(
            f"ieee endpoint arithmetic is binary64-only; operand/result "
            f"dtypes {bad} are not modeled (float32/float16 round "
            f"differently, integer arithmetic wraps) — declined"
        )


def _ieee_arith(op):
    """The monotone core (add/sub/mul/div): native binary64 corner
    endpoints, NaN corners routed to the flag (never into an interval).
    A maybe-NaN operand poisons the result (NaN propagates through all
    four ops), so operand flags OR into the output flag."""

    def t(eqn, params, ins, flags):
        _ieee_f64_only(eqn)
        box, made_nan = op(ins[0], ins[1])
        return [box], [made_nan or any(flags)]

    return t


def _ieee_unary_exact(fn):
    """neg/abs: exact sign arithmetic — the float result IS the real
    result, no rounding, no new NaN (neg(nan)/abs(nan) stay NaN: flag
    propagates)."""

    def t(eqn, params, ins, flags):
        _ieee_f64_only(eqn)
        # subnormal haze on the result: whether a sign operation is
        # flushed is target-dependent; the hull with 0 covers both
        return [iv.subnormal_haze(fn(ins[0]))[0]], [flags[0]]

    return t


def _ieee_minmax(fn):
    """max/min: float max/min of non-NaN operands is exact (no rounding),
    so the real transfer's endpoint rule is float-exact when neither
    operand is maybe-NaN. With a maybe-NaN operand the result is the
    other operand, the extremum, or NaN depending on the backend's NaN
    ordering (measured on jax 0.11.0 cpu: lax.max/min PROPAGATE NaN in
    both operand orders) — the operand hull covers every one of those
    non-NaN outcomes, so hull + flag is sound without leaning on the
    measurement."""

    def t(eqn, params, ins, flags):
        _ieee_f64_only(eqn)
        a, b = ins
        if any(flags):
            mn, mx = iv.minimum(a, b), iv.maximum(a, b)
            hull = iv.IntervalArray(shape=mn.shape, los=mn.los, his=mx.his)
            return [iv.subnormal_haze(hull)[0]], [True]
        return [iv.subnormal_haze(fn(a, b))[0]], [False]

    return t


def _ieee_exp(eqn, params, ins, flags):
    """exp keeps its 1-ulp outward libm bracket — a faithfully-rounded
    libm result lands within 1 ulp of the true value, so the outward
    bracket contains the float the program computes (the same
    EXP_LIBM_ASSUMPTION stamps, via the sound-libm tier). exp(NaN) is
    NaN and nothing else: flag propagation is exact. The subnormal haze
    covers flushing libm targets (measured jax 0.11.0 CPU:
    exp(-720) = 0.0 while IEEE exp is subnormal — the 1-ulp bracket alone
    cannot absorb a flush to 0)."""
    _ieee_f64_only(eqn)
    return [iv.subnormal_haze(iv.exp(ins[0]))[0]], [flags[0]]


def _ieee_pow(eqn, params, ins, flags):
    """pow keeps its libm corner brackets (strictly positive base — the
    same decline otherwise), but a maybe-NaN operand DECLINES rather than
    flag-propagating: IEEE pow has non-NaN results at NaN inputs
    (pow(NaN, 0) = 1 and pow(1, NaN) = 1 — measured on jax 0.11.0), and 1
    may lie outside the corner bracket, so flag propagation alone would
    be unsound."""
    _ieee_f64_only(eqn)
    if any(flags):
        raise iv.IntervalError(
            "pow over a maybe-NaN operand: IEEE pow(NaN, 0) = 1 and "
            "pow(1, NaN) = 1 escape both the corner bracket and the NaN "
            "flag — no sound rule here, declined"
        )
    # subnormal haze on the result: a flushing libm pow may return 0
    # where the bracket is subnormal
    return [iv.subnormal_haze(iv.pow_(*ins))[0]], [False]


def _ieee_sqrt(eqn, params, ins, flags):
    """sqrt under ieee — native binary64, CORRECTLY rounded, so the float
    root is bracketed exactly (no outward rounding), unlike the 1-ulp libm
    bracket exp/pow carry. A NEGATIVE argument produces NaN, routed into the
    flag by :func:`stelling.interval.ieee_sqrt` (never leaked as an
    endpoint); a maybe-NaN operand poisons the result (``sqrt(NaN) = NaN``),
    so the operand flag ORs into the output flag. binary64-only, like the
    rest of the ieee arithmetic core (the subnormal haze is applied inside
    the kernel)."""
    _ieee_f64_only(eqn)
    box, made_nan = iv.ieee_sqrt(ins[0])
    return [box], [made_nan or flags[0]]


def _non_f64_float_dtypes(atoms) -> list[str]:
    """The non-binary64 FLOAT dtypes among these atoms' avals. The ieee
    mode models binary64 only, and the measured flush is PER-DTYPE
    (jax 0.11.0 CPU flushes float32 subnormals — |x| < 2**-126, which are
    NORMAL binary64 numbers invisible to the binary64 haze — while
    float16 is not flushed on this target): any surface that would judge
    a non-f64 float definitely must decline instead of mismodelling."""
    return sorted(
        {
            v.aval.dtype
            for v in atoms
            if "float" in (v.aval.dtype or "") and v.aval.dtype != "float64"
        }
    )


def _ieee_cmp_f64_only(eqn) -> None:
    """The comparison face of the binary64-only guard (re-attack U2): DAZ
    reaches comparisons per-dtype (measured jax 0.11.0 CPU:
    ``float32(1e-45) > 0`` is False, ``float32(1e-39) == float32(1e-40)``
    is True), and the binary64 haze cannot see the f32 band — so any
    comparison with a non-f64 float operand declines with the gap
    quoted. Integer/bool comparisons are unaffected (no flush hazard)."""
    bad = _non_f64_float_dtypes(eqn.invars)
    if bad:
        raise iv.IntervalError(
            f"ieee mode models binary64 only; {'/'.join(bad)} comparison "
            f"semantics (incl. per-dtype subnormal flush — measured on jax "
            f"0.11.0 CPU: float32 subnormals flush, float32(1e-45) > 0 is "
            f"False) are not modelled — declined"
        )


def _ieee_cmp(fn, nan_answer):
    """Comparisons under ieee: exact on non-NaN operands (float compare
    agrees with the real order on ±inf), three-valued as in real mode.
    NaN falsifies lt/gt/le/ge/eq and satisfies ne, so with a maybe-NaN
    operand the elements whose definite answer disagrees with the NaN
    answer degrade to unknown: definite-true is blocked for the
    falsified comparisons (never definitely true), definite-false is
    blocked for ne — the other face stands (NaN agrees with it).
    Comparison outputs are bools: never NaN, flag False.

    Operands are subnormal-hazed before judging: DAZ reaches
    comparisons themselves (measured jax 0.11.0 CPU: ``5e-324 > 0`` is
    False, ``5e-324 == 1e-320`` is True), so a band-touching operand
    must be judged as possibly 0."""
    blocked = iv.BOOL_FALSE if nan_answer else iv.BOOL_TRUE

    def t(eqn, params, ins, flags):
        _ieee_cmp_f64_only(eqn)
        r = fn(*(iv.subnormal_haze(x)[0] for x in ins))
        if any(flags):
            los, his = [], []
            for lo, hi in zip(r.los, r.his):
                if (lo, hi) == blocked:
                    lo, hi = iv.BOOL_UNKNOWN
                los.append(lo)
                his.append(hi)
            r = iv.IntervalArray(shape=r.shape, los=tuple(los), his=tuple(his))
        return [r], [False]

    return t


def _ieee_bool_logic(name, op):
    """Kleene and/or on bools (same bool-dtype guard as real mode —
    bitwise integer forms decline). Bools are never NaN; a maybe-NaN
    flag on a bool operand is a decline artifact (⊤-maybe-NaN), so that
    operand's elements read as unknown before the Kleene op — false ∧
    unknown is still false, nothing definite leaks from a flagged
    operand."""

    def t(eqn, params, ins, flags):
        dtypes = [v.aval.dtype for v in eqn.invars]
        if any(d != "bool" for d in dtypes):
            raise iv.IntervalError(
                f"{name!r} transfer covers bool operands only (three-valued "
                f"logic); got dtypes {dtypes} — bitwise integer {name} has "
                f"no interval rule"
            )
        ops = [
            iv.IntervalArray(
                shape=b.shape, los=(0.0,) * b.size, his=(1.0,) * b.size
            )
            if f
            else b
            for b, f in zip(ins, flags)
        ]
        return [op(*ops)], [False]

    return t


def _ieee_reduce_sum(eqn, params, ins, flags):
    """reduce_sum under ieee — the ONE row of this build whose real-mode
    argument does not survive the dial.

    The real transfer is sound for every association order because ℝ
    addition is associative, so all orders denote one number. Float
    addition is not, XLA may reassociate a reduction, and the jaxpr
    records no order: the association freedom lives INSIDE the equation,
    where no equation-faithful model can reach it. So only the
    association-free reductions are modelled (0, 1 or 2 contributors —
    zero or one addition, and IEEE addition is commutative); 3 or more
    declines with the gap quoted (:data:`stelling.interval
    .REDUCE_SUM_IEEE_ORDER_DECLINE`), which the dispatcher turns into
    ⊤-maybe-NaN.

    A maybe-NaN operand poisons the sum (NaN + anything is NaN) — except
    over an EMPTY reduction range, which reads no element at all and is
    exactly 0.0 whatever flag the (elementless) operand carries.
    """
    _ieee_f64_only(eqn)
    box, made_nan = iv.ieee_reduce_sum(ins[0], tuple(_req(params, "axes", "reduce_sum")))
    reads_an_element = ins[0].size > 0
    return [box], [made_nan or (flags[0] and reads_an_element)]


def _ieee_integer_pow(eqn, params, ins, flags):
    """integer_pow under ieee — the same defect class as reduce_sum above,
    with float MULTIPLICATION in place of addition.

    The jaxpr fixes the exponent but not the evaluation schedule, and the
    candidate lowerings disagree in the last ulps (measured: ``((x*x)*x)*x
    != (x*x)*(x*x)`` for 34% of x in [0.5, 2.0]; ``1/(x*x) !=
    (1/x)*(1/x)`` for 53%; a correctly-rounded libm ``pow`` is a third
    answer). So only the arithmetic-free exponents are modelled:

    * ``y = 0`` → exactly 1.0 for every base. MEASURED on jax 0.11.0 CPU
      binary64 at 0.0, -0.0, ±inf and **NaN** — so this is the one ieee
      transfer that legitimately CLEARS a maybe-NaN flag: the result does
      not depend on the operand's value at all.
    * ``y = 1`` → the identity (flag rides along, DAZ haze applied).

    Every other exponent declines with the gap quoted. That is strictly
    MORE conservative than the real transfer's zero-in-divisor rule for
    negative y — under ieee the pole is not even reached, because the
    schedule question bites first.
    """
    _ieee_f64_only(eqn)
    y = _integer_exponent(params)
    if y is None:
        return None
    _integer_pow_budget(ins[0], y)
    if y == 0:
        return [iv.point(1.0, ins[0].shape)], [False]
    if y == 1:
        return [iv.subnormal_haze(ins[0])[0]], [flags[0]]
    raise iv.IntervalError(iv.INTEGER_POW_IEEE_SCHEDULE_DECLINE.format(y=y))


def _ieee_reduce_or(eqn, params, ins, flags):
    dtypes = [v.aval.dtype for v in eqn.invars]
    if any(d != "bool" for d in dtypes):
        raise iv.IntervalError(
            f"'reduce_or' transfer covers bool operands only; got dtypes "
            f"{dtypes}"
        )
    a = ins[0]
    if flags[0]:
        a = iv.IntervalArray(
            shape=a.shape, los=(0.0,) * a.size, his=(1.0,) * a.size
        )
    return [iv.reduce_or(a, tuple(_req(params, "axes", "reduce_or")))], [False]


# the selector dtypes jax's select_n accepts (measured: lax.select_n
# rejects float selectors at trace time) — the ONLY dtypes for which
# "a selector value is never NaN" is a fact rather than an assumption
_SELECTOR_DTYPES = frozenset(
    {"bool", "int8", "int16", "int32", "int64",
     "uint8", "uint16", "uint32", "uint64"}
)


def _ieee_select_n(eqn, params, ins, flags):
    """Selection/join is pure data movement — the real transfer's measured
    clamp semantics and straddle join are float-sound as-is. The output
    flag is the OR of every case's flag — conservative (a case the
    selector excludes may still set it), sound.

    The selector invariant is ENFORCED, not assumed (audit F1): only
    bool/integer selectors are accepted (jax rejects float selectors at
    trace time, but hand-built/deserialized IR arrives here unchecked —
    a float selector's value could be NaN, which no selection rule
    models), and a maybe-NaN-flagged selector declines outright (its
    provenance is untrusted; picking a case from its interval would
    silently drop the NaN possibility)."""
    sel_dtype = eqn.invars[0].aval.dtype if eqn.invars else ""
    if sel_dtype not in _SELECTOR_DTYPES:
        raise iv.IntervalError(
            f"select_n selector dtype {sel_dtype!r} is not bool/integer "
            f"(jax rejects float selectors at trace time; a float selector "
            f"may be NaN) — no selection rule; declined"
        )
    if flags[0]:
        raise iv.IntervalError(
            "select_n selector carries maybe-NaN under ieee semantics — "
            "its provenance is untrusted and selection would drop the NaN "
            "possibility; declined"
        )
    r = iv.select_n(ins[0], list(ins[1:]))
    return [r], [any(flags[1:])]


def _ieee_passthrough(real_transfer):
    """Category (i): pure data movement with exact semantics — element
    routing only, no arithmetic, no rounding, dtype-agnostic, NaN moves
    like any other value. The real transfer is reused with the operand
    flags OR-ed onto every output."""

    def t(eqn, params, ins, flags):
        outs = real_transfer(eqn, params, ins)
        if outs is None:
            return None
        return outs, [any(flags)] * len(outs)

    return t


def _ieee_scatter(eqn, params, ins, flags):
    """The static-index x.at[k].set(v) form: data movement (sound as-is);
    output flag = operand ∨ updates. Maybe-NaN INDICES decline — the
    static-index rule needs definite integer indices, and a NaN index is
    mode-dependent garbage."""
    if len(ins) == 3 and flags[1]:
        raise iv.IntervalError(
            "scatter indices carry maybe-NaN under ieee semantics — the "
            "static-index rule needs definite non-NaN indices; declined"
        )
    outs = _t_scatter(eqn, params, ins)
    if outs is None:
        return None
    return outs, [flags[0] or flags[2]]


def _ieee_gather(eqn, params, ins, flags):
    """The static-index leading-axis row form: data movement (sound
    as-is); output flag = operand's. Maybe-NaN indices decline, as for
    scatter."""
    if len(ins) == 2 and flags[1]:
        raise iv.IntervalError(
            "gather indices carry maybe-NaN under ieee semantics — the "
            "static-index rule needs definite non-NaN indices; declined"
        )
    outs = _t_gather(eqn, params, ins)
    if outs is None:
        return None
    return outs, [flags[0]]


def _ieee_scatter_add(eqn, params, ins, flags):
    """Category (iii), whole-primitive: the censused ieee REFUSAL for
    scatter-add — the honest floor, chosen over an order-independent
    enclosure. The real transfer's soundness argument is ℝ-associativity
    of the per-element accumulation, which is exactly what float addition
    does not offer (the reduce_sum defect class), and the contraction
    freedom (a product-derived update fused into the accumulate's add)
    has no taint-hull built for the scatter path — so EVERY form declines
    with the gap quoted, including duplicate-free and empty-update forms.
    The dispatcher turns the raise into a noted ⊤ maybe-NaN, counted in
    coverage: a refusal is a censused entry, not an omission."""
    raise iv.IntervalError(iv.SCATTER_ADD_IEEE_DECLINE)


def _ieee_dot_general(eqn, params, ins, flags):
    """dot_general under ieee: a whole-primitive censused REFUSAL.

    Same shape of refusal as ``scatter-add`` and for the sharper version of
    the same reason — see :data:`stelling.interval.DOT_GENERAL_IEEE_DECLINE`.
    The dispatcher turns the raise into a noted ⊤ maybe-NaN counted in
    coverage: a refusal is a censused entry, not an omission.
    """
    raise iv.IntervalError(iv.DOT_GENERAL_IEEE_DECLINE)


def _ieee_convert(eqn, params, ins, flags):
    """convert_element_type under ieee. A non-f64 FLOAT source declines
    outright (re-attack U2): the whitelist's value-preservation claim is
    a gradual-semantics fact, measured FALSE under this target's
    per-dtype DAZ — jax 0.11.0 CPU converts an f32 subnormal to f64 as
    0.0 (storage keeps the bits; compute flushes), which would carry the
    flushed 0 into f64 dataflow past the binary64 haze and the f64-only
    arithmetic guard. The remaining whitelisted conversions (int/bool
    sources, f64 identity) really are value-preserving for every source
    value including ±inf and NaN (NaN converts to NaN: flag propagates).
    The float→int trunc path is float-exact (trunc is the measured
    conversion semantics, the in-range guard keeps it defined, integer
    results are exact doubles) but declines a maybe-NaN input —
    converting NaN to int is target-dependent garbage. Everything else
    declines exactly as in real mode."""
    src = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    dst = str(params.get("new_dtype"))
    if "float" in src and src != "float64":
        raise iv.IntervalError(
            f"ieee mode models binary64 only; {src} conversion semantics "
            f"(incl. per-dtype subnormal flush — measured on jax 0.11.0 "
            f"CPU: f32→f64 convert of a subnormal yields 0.0, so the "
            f"exact-conversion whitelist's value-preservation claim fails "
            f"under DAZ) are not modelled — declined"
        )
    if src == dst or (src, dst) in _EXACT_CONVERSIONS:
        return [ins[0]], [flags[0]]
    if _in_range_int_narrowing(ins[0], src, dst):
        # the statically-in-range int64->int32 narrowing (third audit,
        # F5b) is an exact integer identity — no float semantics, no
        # flush hazard — so it passes under ieee exactly as in real mode
        return [ins[0]], [flags[0]]
    if "float" in src and dst in _INT_RANGE:
        if flags[0]:
            raise iv.IntervalError(
                "float->int conversion of a maybe-NaN value is "
                "target-dependent garbage under ieee semantics — declined"
            )
        outs = _t_convert(eqn, params, ins)
        if outs is None:
            return None
        return outs, [False]
    return None  # value-changing or unrecognized conversion -> ⊤, noted


def _ieee_any(eqn, params, ins, flags):
    """Declared inputs start maybe_nan=False: the declared closed box is
    a set of floats, and NaN bounds are refused (iv.from_bounds raises on
    NaN, which declines the declaration loudly rather than admitting an
    unbounded-NaN input). The box is subnormal-hazed on entry — DAZ
    flushes inputs, so a band-located declared value may be consumed as
    0; the dispatcher withholds exactness from declarations the haze
    changed."""
    box = iv.from_bounds(
        tuple(_req(params, "shape", "stelling_any")),
        float(_req(params, "lo", "stelling_any")),
        float(_req(params, "hi", "stelling_any")),
    )
    return [iv.subnormal_haze(box)[0]], [False]


IEEE_TRANSFERS = {
    # (ii) ieee variants: the monotone arithmetic core — native binary64
    # endpoints, NaN corners routed to the flag; the real-mode 0·∞ = 0
    # convention (iv._prod inside iv.mul) is NOT reused.
    "add": (_ieee_arith(iv.ieee_add), TIER_EXACT),
    "sub": (_ieee_arith(iv.ieee_sub), TIER_EXACT),
    "add_any": (_ieee_arith(iv.ieee_add), TIER_EXACT),
    "mul": (_ieee_arith(iv.ieee_mul), TIER_EXACT),
    "div": (_ieee_arith(iv.ieee_div), TIER_EXACT),
    # (i)/(ii) exact sign arithmetic; flag propagates (NaN stays NaN)
    "neg": (_ieee_unary_exact(iv.neg), TIER_EXACT),
    "abs": (_ieee_unary_exact(iv.abs_), TIER_EXACT),
    # (ii) exact on non-NaN operands; NaN-ordering ambiguity covered by
    # the operand hull + flag
    "max": (_ieee_minmax(iv.maximum), TIER_EXACT),
    "min": (_ieee_minmax(iv.minimum), TIER_EXACT),
    # (ii) libm brackets kept (faithful rounding lands within 1 ulp);
    # pow declines maybe-NaN operands (pow(NaN,0)=1 escapes the flag)
    "pow": (_ieee_pow, TIER_SOUND_LIBM),
    "exp": (_ieee_exp, TIER_SOUND_LIBM),
    # (ii) native binary64, CORRECTLY rounded — the float root bracketed
    # exactly (no outward bump, tier sound not sound-libm); a negative arg
    # is NaN routed to the flag, a maybe-NaN operand poisons the result
    "sqrt": (_ieee_sqrt, TIER_EXACT),
    # (i) identity
    "stop_gradient": (_ieee_passthrough(lambda eqn, p, ins: [ins[0]]), TIER_EXACT),
    # (i) data movement (the dimensions= reshape declines as in real mode)
    "reshape": (_ieee_passthrough(_t_reshape), TIER_EXACT),
    # (ii) selection/join reused; flag = OR over cases
    "select_n": (_ieee_select_n, TIER_EXACT),
    # (ii) comparisons: NaN falsifies lt/gt/le/ge/eq (definite-true
    # blocked under maybe-NaN), satisfies ne (definite-false blocked)
    "lt": (_ieee_cmp(iv.lt, nan_answer=False), TIER_EXACT),
    "gt": (_ieee_cmp(iv.gt, nan_answer=False), TIER_EXACT),
    "le": (_ieee_cmp(iv.le, nan_answer=False), TIER_EXACT),
    "ge": (_ieee_cmp(iv.ge, nan_answer=False), TIER_EXACT),
    "eq": (_ieee_cmp(iv.eq, nan_answer=False), TIER_EXACT),
    "ne": (_ieee_cmp(iv.ne, nan_answer=True), TIER_EXACT),
    # (ii) Kleene logic; flagged bool operands read as unknown
    "and": (_ieee_bool_logic("and", iv.logical_and), TIER_EXACT),
    "or": (_ieee_bool_logic("or", iv.logical_or), TIER_EXACT),
    "reduce_or": (_ieee_reduce_or, TIER_EXACT),
    # (ii) censused DOWN to the association-free cases: <=2 contributors
    # are exact (0 or 1 addition; IEEE add is commutative), >=3 declines —
    # float addition is not associative and the jaxpr fixes no order
    "reduce_sum": (_ieee_reduce_sum, TIER_EXACT),
    # (ii) censused down the same way: y in {0, 1} perform NO arithmetic
    # and are exact (y=0 is measured 1.0 even at NaN, so it CLEARS the
    # flag); every other exponent declines — no fixed multiply schedule
    "integer_pow": (_ieee_integer_pow, TIER_EXACT),
    # (i) pure data movement, dtype-agnostic, flags ride along
    "squeeze": (
        _ieee_passthrough(
            lambda eqn, p, ins: [iv.squeeze(ins[0], tuple(p.get("dimensions", ())))]
        ),
        TIER_EXACT,
    ),
    "split": (_ieee_passthrough(_t_split), TIER_EXACT),
    "slice": (
        _ieee_passthrough(
            lambda eqn, p, ins: [
                iv.slice_(
                    ins[0],
                    tuple(_req(p, "start_indices", "slice")),
                    tuple(_req(p, "limit_indices", "slice")),
                    tuple(p["strides"]) if p.get("strides") else None,
                )
            ]
        ),
        TIER_EXACT,
    ),
    "scatter": (_ieee_scatter, TIER_EXACT),
    "gather": (_ieee_gather, TIER_EXACT),
    # (iii) the whole-primitive censused refusal: the accumulate's ℝ-
    # associativity argument does not survive the dial, and no all-orders
    # bound or contraction hull is built for the scatter path — every
    # form declines with iv.SCATTER_ADD_IEEE_DECLINE quoted
    "scatter-add": (_ieee_scatter_add, TIER_EXACT),
    "dot_general": (_ieee_dot_general, TIER_EXACT),
    # (i) pure element routing, dtype-agnostic, flags ride along
    "stack": (
        _ieee_passthrough(
            lambda eqn, p, ins: [
                iv.stack(list(ins), int(_req(p, "axis", "stack")))
            ]
        ),
        TIER_EXACT,
    ),
    "transpose": (
        _ieee_passthrough(
            lambda eqn, p, ins: [
                iv.transpose(ins[0], tuple(p.get("permutation", ()) or ()))
            ]
        ),
        TIER_EXACT,
    ),
    "broadcast_in_dim": (
        _ieee_passthrough(
            lambda eqn, p, ins: [
                iv.broadcast_in_dim(
                    ins[0],
                    tuple(_req(p, "shape", "broadcast_in_dim")),
                    tuple(_req(p, "broadcast_dimensions", "broadcast_in_dim")),
                )
            ]
        ),
        TIER_EXACT,
    ),
    "concatenate": (
        _ieee_passthrough(
            lambda eqn, p, ins: [
                iv.concatenate(list(ins), int(_req(p, "dimension", "concatenate")))
            ]
        ),
        TIER_EXACT,
    ),
    # (ii) exact whitelist passes with flag propagation; trunc path
    # declines maybe-NaN inputs; everything else declines as before
    "convert_element_type": (_ieee_convert, TIER_EXACT),
    # (i) declarations: flag starts False, NaN bounds refused
    "stelling_any": (_ieee_any, TIER_EXACT),
    # (i) identity pass-throughs (judging is flag-aware in the dispatcher)
    "stelling_assert": (
        _ieee_passthrough(lambda eqn, p, ins: [ins[0]]), TIER_EXACT
    ),
    "stelling_nonvacuity": (
        _ieee_passthrough(lambda eqn, p, ins: [ins[0]]), TIER_EXACT
    ),
}

# the census must stay total: a registered transfer with no ieee census
# entry would be silent reuse, the exact thing rule 6 forbids
if set(IEEE_TRANSFERS) != set(TRANSFERS):
    raise RuntimeError(
    "every registered transfer must be censused for ieee semantics"
)


# -- constraining assume ------------------------------------------------------

_ASSUME_MODES = ("constrain", "inert")

# The comparisons a point-bounded assume can narrow through. `ne` is a
# comparison but NOT here: excluding a single point from an interval does
# not narrow it (the hull of [lo, hi] \ {k} is [lo, hi]) — it stays inert
# with the reason quoted.
_ASSUME_CMPS = frozenset({"ge", "gt", "le", "lt", "eq"})

# cmp(k, v) === flipped-cmp(v, k): normalization when the point side is on
# the left.
_CMP_FLIP = {"ge": "le", "le": "ge", "gt": "lt", "lt": "gt", "eq": "eq"}

_CMP_SYMBOL = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "=="}

# three-valued comparison transfers, for deciding point-vs-point assumes
_CMP_FN = {"ge": iv.ge, "gt": iv.gt, "le": iv.le, "lt": iv.lt, "eq": iv.eq}

# The spec-fixed relational disclosure, appended verbatim to the DROPPED
# note when neither comparison side's interval is a finite point AND both
# sides genuinely vary.
_RELATIONAL_REASON = (
    "relational: both sides vary — constraining needs relational domains"
)

# Audit F6: a CONSTANT bound that is not a finite point (±inf, NaN, an
# undecodable literal) must be disclosed as such — "both sides vary" is
# false of it, and drop reasons are quoted into verdict notes as fact.
_NONFINITE_BOUND_REASON = (
    "the comparison bound is a constant but not a finite point (infinite "
    "or undecodable, e.g. ±inf/NaN) — no closed half-space represents it"
)


def _finite_point(box: iv.IntervalArray | None) -> bool:
    """Every element is a degenerate finite interval (``lo == hi``,
    finite) — the shape of a usable assume bound."""
    return box is not None and all(
        lo == hi and math.isfinite(lo) for lo, hi in zip(box.los, box.his)
    )


def _subnormal_const_literal(atom: ir.Atom) -> bool:
    """A literal constant whose raw decode the subnormal haze changes —
    i.e. a subnormal-band constant. Under ieee its as-consumed value is
    flush-indeterminate, so the drop reason must name that shape rather
    than claim both sides vary (the audit-F6 discipline)."""
    if not isinstance(atom, ir.Literal):
        return False
    try:
        raw = _value_to_interval(atom.val, atom.aval.shape)
    except (iv.IntervalError, ir.TranscriptionError):
        return False
    return iv.subnormal_haze(raw)[1]


_SUBNORMAL_BOUND_REASON = (
    "a comparison side is a subnormal-band constant — its as-consumed "
    "value is flush-vs-gradual indeterminate under ieee semantics "
    "(FTZ/DAZ targets read it as 0); no certified half-space represents it"
)


def _strict_flush_witness(cmp: str, k: float, nlo: float, nhi: float) -> bool:
    """Flush-robust certification witness for a STRICT assume under ieee:
    does the closed meet stand-in contain a NON-subnormal point strictly
    satisfying the comparison? A strict region whose only content is
    subnormal may be EMPTY at runtime on a DAZ target (every member reads
    as 0, which the strict comparison excludes), so certification demands
    a witness whose runtime reading is itself (0, a normal, or ±inf)."""
    if cmp == "gt":  # region (k, nhi]
        if k < 0.0 <= nhi:
            return True  # w = 0
        if nhi > k and abs(nhi) >= iv.MIN_NORMAL:
            return True  # w = nhi (normal or ±inf)
        if k < -iv.MIN_NORMAL <= nhi:
            return True  # w = -MIN_NORMAL
        if k < iv.MIN_NORMAL <= nhi:
            return True  # w = +MIN_NORMAL
        return False
    # lt: region [nlo, k)
    if nlo <= 0.0 < k:
        return True  # w = 0
    if nlo < k and abs(nlo) >= iv.MIN_NORMAL:
        return True  # w = nlo (normal or ±inf)
    if nlo <= iv.MIN_NORMAL < k:
        return True  # w = +MIN_NORMAL
    if nlo <= -iv.MIN_NORMAL and k > -iv.MIN_NORMAL:
        return True  # w = -MIN_NORMAL
    return False


def _nonfinite_const(atom: ir.Atom, box: iv.IntervalArray | None) -> bool:
    """A constant comparison side that is not a finite point: an
    undecodable literal (NaN, complex, …) or a degenerate interval whose
    value is non-finite (±inf). Distinguished from a genuinely varying
    side so the drop reason names the actual shape (audit F6)."""
    if box is None:
        return isinstance(atom, ir.Literal)
    return all(lo == hi for lo, hi in zip(box.los, box.his)) and not all(
        math.isfinite(lo) for lo in box.los
    )


def _render_box(box: iv.IntervalArray) -> str:
    """Concise interval rendering for notes/assumptions: the scalar
    interval, a per-element list for small arrays, the hull for large."""
    if box.size == 1:
        return f"[{box.los[0]}, {box.his[0]}]"
    if box.size <= 4:
        return "[" + ", ".join(
            f"[{lo}, {hi}]" for lo, hi in zip(box.los, box.his)
        ) + "]"
    return f"hull [{min(box.los)}, {max(box.his)}] ({box.size} elements)"


def _render_bound(bound: iv.IntervalArray) -> str:
    """Render a point-interval bound as its value(s)."""
    if bound.size == 1:
        return f"{bound.los[0]}"
    if bound.size <= 4:
        return "[" + ", ".join(f"{v}" for v in bound.los) + "]"
    return f"values in [{min(bound.los)}, {max(bound.los)}] ({bound.size} elements)"


def _bool_status(b: iv.IntervalArray, *, constrained: bool = False) -> tuple[str, str]:
    n_true = sum(1 for lo, hi in zip(b.los, b.his) if (lo, hi) == iv.BOOL_TRUE)
    n_false = sum(1 for lo, hi in zip(b.los, b.his) if (lo, hi) == iv.BOOL_FALSE)
    n = b.size
    if n_true == n:
        return "discharged", f"definitely true for all {n} element(s)"
    if n_false:
        # audit F4: once an assume has constrained, the judgment set is no
        # longer the declared box — the detail must name the set actually
        # judged, or the refutation sentence misstates its own claim.
        judged = (
            "the precondition-narrowed set"
            if constrained
            else "the declared box"
        )
        return (
            "violated-over-set",
            f"definitely false for {n_false}/{n} element(s) over {judged}",
        )
    return "unknown", f"undecided for {n - n_true}/{n} element(s)"


class _Propagator:
    def __init__(self, assume_mode: str, semantics: str = "real") -> None:
        self.assume_mode = assume_mode
        self.semantics = semantics
        self.env: dict[int, iv.IntervalArray] = {}
        # ieee mode's parallel table: var id -> maybe_nan. Real mode never
        # writes it (the flag machinery is fenced behind semantics checks),
        # so the real path's behavior is untouched.
        self.nan: dict[int, bool] = {}
        # ieee mode's second parallel table: var id -> PRODUCT-DERIVED.
        # XLA contracts a multiply feeding an add/sub into one fused
        # multiply-add, and it does so AFTER its own simplification passes
        # — so a syntactic "is my operand's producer a mul?" test is a bet
        # that the jaxpr shape survives compilation, and that bet loses
        # (audit UNSOUND 5: ten forms, from `neg` to a one-element
        # `reduce_sum`, break the match while the contraction still
        # happens). The taint closes the CLASS instead: it flows from every
        # mul output through the dataflow, and every add/sub that meets it
        # hulls both roundings, whatever lies between. Soundness does not
        # depend on recognising the intervening shape, which is exactly
        # what a simplification set makes unrecognisable.
        self.taint: dict[int, bool] = {}
        # var ids whose value comes from a REFUSED negative/malformed
        # shape (fix re-attack N1): membership, not aval classification,
        # is what gates every env reference — ids are globally unique per
        # transcription, so the set is deliberately NOT scope-swapped
        # (refusal follows the value across call boundaries).
        self.refused_shape: set[int] = set()
        # var id -> producing equation, for the jaxpr currently being run
        # (scoped in run(); assume classification only — never a transfer
        # input)
        self.producers: dict[int, ir.JaxprEqn] = {}
        self.counter = CoverageCounter()
        self.obligations: list[ObligationReport] = []
        self.nonvacuity_checks: list[ObligationReport] = []
        self.used: dict[str, str] = {}
        self.assumptions: set[str] = set()
        self.notes: list[str] = []
        # > 0 while executing a possibly-untaken cond branch: an
        # unsatisfiable assume there is a BRANCH-scoped vacuity (the other
        # branch is real), not a whole-domain harness defect — it degrades
        # with a note instead of raising (audit F2). Transparent call
        # scopes (jit) inherit the current value: they execute
        # unconditionally, so at depth 0 the raise stands.
        self.branch_depth = 0
        # set once any assume narrows: violated-over-set details are then
        # judged over the precondition-narrowed set, and must say so
        # (audit F4)
        self.any_constrained = False
        # var ids whose box is EXACT — equal to the variable's true value
        # set, not an over-approximation: stelling_any outputs (the
        # declared closed box IS the declared set) and consts decoded to
        # exact points. EVERY transfer output is non-exact (minimal,
        # conservative rule — audit F7): rounding pads and
        # correlation-blind joins can inflate a box past the true image,
        # so box-nonemptiness certifies true-region nonemptiness only for
        # exact boxes. Scope-local (swapped with env at every sub-jaxpr
        # descent): a scope certifies only its own declarations. The set,
        # its maintenance rules, and the certification decision live in
        # stelling.exactness (the shared primitive future layers import).
        self.exact = exactness.ExactSet()
        # set once a constraining assume narrows a NON-exact variable: an
        # uncertified-precondition constraint is then in force, and every
        # subsequent definite violation is withheld from REFUTED (a
        # possibly-vacuous refutation is not a refutation — audit F7).
        self.uncertified = False
        # set when an assume was DROPPED rather than applied. Distinct from
        # `uncertified` because it must also reach the SOLVER path: the
        # escalation decline keys on a constrained assume being present, and
        # a dropped one is not present at all.
        self.assume_dropped = False

    def read(self, atom: ir.Atom) -> iv.IntervalArray:
        if isinstance(atom, ir.Literal):
            # a constant the domain cannot represent (a NaN sentinel — the
            # ubiquitous `where(pred, x, nan)` pattern — an undecodable
            # dtype, a complex literal) is a LEGAL program value outside the
            # ℝ model: it degrades to ⊤ with a note, it does not kill the
            # analysis (audit-gate finding 1). The unbound-var raise below
            # is untouched — that one is a transcription defect, not a
            # value.
            try:
                box = _value_to_interval(atom.val, atom.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError) as e:
                self.notes.append(f"literal outside the domain ({e}); ⊤")
                return _safe_top(atom.aval.shape)
            if self.semantics == "ieee":
                # DAZ flushes inputs: literal constants entering ieee
                # propagation are subnormal-hazed like every other value
                box = iv.subnormal_haze(box)[0]
            return box
        got = self.env.get(atom.id)
        if got is None:
            # not a coverage gap: an equation reading a never-bound var is a
            # transcription/traversal defect, and ⊤-ing it would soundly hide
            # a bug in the one module that must not have bugs. Same discipline
            # as the params whitelist: unknown primitives are fine, unknown
            # *structure* is not.
            raise ir.TranscriptionError(
                f"equation reads var {atom.id} before any binding — "
                f"transcription defect, refusing to widen it away"
            )
        return got

    def read_flag(self, atom: ir.Atom) -> bool:
        """The atom's maybe-NaN flag (ieee mode). Decodable literals are
        definite non-NaN values; undecodable ones (the NaN sentinel above
        all) may be NaN. Vars default to False — every flag-True binding
        is written explicitly."""
        if isinstance(atom, ir.Literal):
            try:
                _value_to_interval(atom.val, atom.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError):
                return True
            return False
        return self.nan.get(atom.id, False)

    def read_taint(self, atom: ir.Atom) -> bool:
        """Whether this atom's value is PRODUCT-DERIVED (ieee mode). A
        literal never is; vars default to False and every taint is written
        explicitly."""
        if isinstance(atom, ir.Literal):
            return False
        return self.taint.get(atom.id, False)

    def top_out(self, eqn: ir.JaxprEqn) -> None:
        for out in eqn.outvars:
            self.env[out.id] = _safe_top(out.aval.shape)
            if self.semantics == "ieee":
                # ⊤ under ieee is maybe-NaN: an unknown/declined value
                # could be anything a float can be, including NaN
                self.nan[out.id] = True
                # ...and a DECLINED equation must not launder the taint:
                # whatever the compiler did with the product, the result is
                # still product-derived
                self.taint[out.id] = any(
                    self.read_taint(a) for a in eqn.invars
                )

    def _contraction_hull(self, eqn: ir.JaxprEqn, outs, out_flags):
        """Cover BOTH roundings of a product feeding an add/sub (ieee only).

        XLA contracts ``a*b + c`` into a single fused multiply-add — the
        compiled HLO for this target contains ``multiply_add_fusion``, and
        measured, ``(a*b) - 1`` at ``a = 1+2**-27, b = 1-2**-27`` is
        ``0.0`` eager but ``-2**-54`` under jit. Contraction leaves the
        equation ORDER untouched, so the equation-order reliance does not
        reach it, and an assumption measured false on the target is not
        stampable. The mode therefore models both: the contracted value is
        computed exactly (:func:`stelling.interval.ieee_fma_hull`) and
        hulled with the uncontracted one, so results stay definite
        wherever the two roundings agree and go indeterminate only where
        they differ.

        Forms whose contracted value cannot be bracketed here — an
        infinite operand endpoint, an unreadable producer, shapes that do
        not broadcast — raise :class:`stelling.interval.IntervalError` and
        the caller turns that into a quoted ⊤ decline.
        """
        candidates = [
            i for i, atom in enumerate(eqn.invars) if self.read_taint(atom)
        ]
        if not candidates:
            return outs, out_flags
        box = outs[0]
        made_nan = bool(out_flags and out_flags[0])
        for i in candidates:
            atom = eqn.invars[i]
            other = eqn.invars[1 - i]
            # sub: `p - c` negates the addend; `c - p` negates the product.
            negate_product = eqn.primitive == "sub" and i == 1
            negate_addend = eqn.primitive == "sub" and i == 0
            prod = (
                self.producers.get(atom.id)
                if isinstance(atom, ir.Var)
                else None
            )
            c = self.read(other)
            if (
                prod is not None
                and prod.primitive == "mul"
                and len(prod.invars) == 2
            ):
                # PRECISION path: the product is right there, so the
                # contracted value is computed exactly and a form whose two
                # roundings agree stays definite. Soundness never rests on
                # reaching this branch — the taint is what guarantees we
                # are here at all.
                a, b = (self.read(x) for x in prod.invars)
                fused, fused_nan = iv.ieee_fma_hull(
                    a, b, c,
                    negate_product=negate_product,
                    negate_addend=negate_addend,
                )
            else:
                # CONSTRAINT ON FUTURE TIGHTENING (read before touching
                # the ieee add/sub kernels): the last clause below leans on
                # the ieee add ROUNDING OUTWARD to absorb the fma's single
                # rounding. If ieee add/sub are ever moved to the
                # exact-when-representable discipline the real-mode
                # transfers now use, that slack disappears and this hull
                # stops covering the fused form — a silent soundness
                # regression with no test that would notice, because the
                # real-mode tightening (which is safe) and the ieee one
                # (which is not) look identical at the call site. The
                # real-mode tightening is clean precisely BECAUSE this
                # whole path is gated on `ieee` at its caller.
                #
                # SOUND path: something lies between the multiply and here
                # — and what that something is, after XLA's simplification
                # passes, is not knowable from the jaxpr. The tainted
                # operand is a rounded product `fl(p)`, so `p` sits within
                # half an ulp of it; widening by a full ulp each way covers
                # every `p` the compiler could still be holding, and the
                # outward-rounded add covers the single rounding an fma
                # applies to `p ± c`.
                fused, fused_nan = self._unrecovered_contraction(
                    self.read(atom), c,
                    negate_product=negate_product,
                    negate_addend=negate_addend,
                )
            if fused.shape != box.shape:
                raise iv.IntervalError(
                    iv.IEEE_CONTRACTION_DECLINE.format(
                        why=(
                            f"the contracted operands broadcast to "
                            f"{fused.shape} but the equation's result is "
                            f"{box.shape}"
                        )
                    )
                )
            box = iv.hull(box, fused)
            made_nan = made_nan or fused_nan
        return [box, *outs[1:]], [made_nan, *(out_flags or [])[1:]]

    @staticmethod
    def _unrecovered_contraction(
        prod_box, c, *, negate_product: bool, negate_addend: bool
    ):
        """Sound bracket of the contracted value when the multiply cannot
        be recovered from the jaxpr (see :meth:`_contraction_hull`)."""
        widened = iv.IntervalArray(
            shape=prod_box.shape,
            los=tuple(math.nextafter(v, -math.inf) for v in prod_box.los),
            his=tuple(math.nextafter(v, math.inf) for v in prod_box.his),
        )
        if negate_product:
            widened = iv.neg(widened)
        if negate_addend:
            c = iv.neg(c)
        try:
            # the real (outward-rounded) add: its one-ulp bump is what
            # covers the fma's own final rounding
            return iv.add(widened, c), False
        except iv.IntervalError as e:
            raise iv.IntervalError(
                iv.IEEE_CONTRACTION_DECLINE.format(
                    why=f"the widened product bracket is not addable ({e})"
                )
            ) from None

    def mark_unreached(self, eqn: ir.JaxprEqn) -> None:
        stack = list(sub_jaxprs(eqn))
        while stack:
            j = stack.pop()
            for e in j.eqns:
                self.counter.record_unreached(e.primitive)
                stack.extend(sub_jaxprs(e))

    # -- constraining assume --------------------------------------------------

    def _quiet_interval(self, atom: ir.Atom) -> iv.IntervalArray | None:
        """The atom's current interval for assume classification only:
        never notes, never raises — ``None`` where no interval is
        readable (an undecodable literal, an unbound var). Classification
        failures make the assume inert, which is always sound."""
        if isinstance(atom, ir.Literal):
            try:
                box = _value_to_interval(atom.val, atom.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError):
                return None
            if self.semantics == "ieee":
                # the same haze read() applies: assume classification must
                # see the interval the judging paths see (a subnormal-band
                # bound is then not a finite point and cannot narrow)
                box = iv.subnormal_haze(box)[0]
            return box
        return self.env.get(atom.id)

    def _unsatisfiable(
        self, where: str, desc: str, vacuous: list[str], message: str
    ) -> bool:
        """One unsatisfiability posture for all three detection sites
        (empty meet, strict-boundary collapse, definitely-false constant).

        At top level and inside transparent call scopes the whole-domain
        claim is true: raise :class:`UnsatisfiableAssumptionError` with
        ``message``. Inside a possibly-untaken cond branch it is NOT — the
        assume is branch-scoped, the other branch is real (audit F2): the
        constraint degrades to a branch-local vacuity note (``desc`` rides
        into it) and the caller applies no narrowing. Returns True in the
        branch-local case."""
        if self.branch_depth:
            vacuous.append(desc)
            return True
        raise UnsatisfiableAssumptionError(message)

    def _assume_constrain(self, eqn: ir.JaxprEqn, where: str) -> None:
        """Classify the assumed predicate by its producing equation and
        narrow where — and only where — the narrowing is provably a
        superset of the true assumed region. Everything else stays inert
        with the reason disclosed."""
        narrowed: list[tuple[int, iv.IntervalArray, bool, bool]] = []
        dropped: list[str] = []
        vacuous: list[str] = []
        self._apply_assumed_pred(eqn.invars[0], where, narrowed, dropped, vacuous)
        for desc in vacuous:
            # audit F2: the branch-scoped unsatisfiability disclosure
            self.notes.append(
                f"assume unsatisfiable within this cond branch at {where}: "
                f"the precondition is definitely false whenever this branch "
                f"is taken — constraint not applied; obligations in this "
                f"branch may be vacuous under the branch's precondition "
                f"({desc})"
            )
        if narrowed:
            self.any_constrained = True
            self.counter.record_constrained(eqn.primitive)
            # audit F5: conjuncts that did NOT apply inside a constrained
            # assume must be visible in the coverage summary, not only in
            # the notes
            for _ in (*dropped, *vacuous):
                self.counter.record_dropped_conjunct()
            for var_id, box, changed, certified in narrowed:
                if changed:
                    self.notes.append(
                        f"assume CONSTRAINED at {where}: narrowed var "
                        f"{var_id} to {_render_box(box)}"
                    )
                else:
                    self.notes.append(
                        f"assume CONSTRAINED at {where}: var {var_id} "
                        f"already within the assumed region {_render_box(box)}"
                    )
                self.assumptions.add(
                    f"constrained assume at {where}: the verdict holds "
                    f"where the precondition holds — narrowed var {var_id} "
                    f"to {_render_box(box)}"
                )
                if not certified:
                    # audit F7: the target's box is an over-approximation
                    # (a transfer output / bracketed const), so a nonempty
                    # meet does NOT certify the true assumed region
                    # nonempty. The narrowing itself is sound (superset of
                    # true-region ∩ reachable) and stays applied; the
                    # conditional claim may be vacuous, and every
                    # subsequent definite violation is withheld from
                    # REFUTED.
                    self.uncertified = True
                    self.notes.append(
                        f"precondition satisfiability UNCERTIFIED at "
                        f"{where}: var {var_id} is an over-approximated "
                        f"intermediate (its box may exceed its true image) "
                        f"— the conditional claim may be vacuous"
                    )
                    self.assumptions.add(
                        "precondition satisfiability uncertified: a "
                        "constraining assume narrowed an over-approximated "
                        "intermediate whose box may exceed its true image "
                        "— the conditional claim may be vacuous; the "
                        "inert-mode control is the visibility instrument"
                    )
            for reason in dropped:
                # a conjunction can mix constrainable and inert conjuncts;
                # the un-narrowable part is still a drop and still says so
                self.notes.append(
                    f"assume conjunct DROPPED at {where}: constraining "
                    f"proceeded without this conjunct — a superset ({reason})"
                )
        else:
            self.counter.record_inert(eqn.primitive)
            if dropped or not vacuous:
                # an assume whose only content was a branch-scoped
                # unsatisfiable conjunct already carries the directed note
                # above; everything else gets the pre-existing DROPPED
                # disclosure with the reason(s) appended
                reasons = "; ".join(dropped) if dropped else "unclassified predicate"
                self.notes.append(
                    ASSUME_DROP_NOTE.format(where=where) + f" ({reasons})"
                )
                # F7's NO-OP HALF. The narrowing path sets `uncertified` when
                # it constrains an over-approximated variable; a DROPPED
                # assume never reached that branch, so neither the interval
                # withhold nor solvers.py's decline-when-constrained ever
                # engaged — both are conditioned on the assume having TAKEN
                # EFFECT, and an assume that no-ops is invisible to both.
                #
                # A dropped assume means the query ran over a SUPERSET of the
                # intended set, so the disposition is ONE-SIDED, exactly as
                # F7 already is (it gates `violated-over-set` only, and
                # nothing withholds `discharged`):
                #
                #   VERIFIED over a superset IMPLIES VERIFIED over the subset
                #                                       -> keep, disclose
                #   REFUTED  over a superset does NOT   -> withhold; the
                #                                          witness may lie
                #                                          outside the set
                #
                # Measured before this: `assume(jnp.all(x >= 0))` over
                # x in [-10, 10]^3 asserting sum(x) >= 0 returned REFUTED with
                # the replay-confirmed witness [0, 0, -1], which violates the
                # dropped precondition. Two-sided would over-fire the way the
                # scatter bar did for its whole history.
                self.uncertified = True
                self.assume_dropped = True

    def _apply_assumed_pred(
        self,
        atom: ir.Atom,
        where: str,
        narrowed: list[tuple[int, iv.IntervalArray, bool, bool]],
        dropped: list[str],
        vacuous: list[str],
    ) -> None:
        if isinstance(atom, ir.Literal):
            dropped.append(
                f"predicate is a literal ({atom.val!r}), not a traced comparison"
            )
            return
        producer = self.producers.get(atom.id)
        if producer is None:
            dropped.append(
                "the predicate's producing equation is not visible in this "
                "scope (constvar, invar, or cross-scope value)"
            )
            return
        prim = producer.primitive
        if prim == "and":
            # a conjunction holds iff both conjuncts hold: recurse,
            # classifying each independently. jax's `and` is BITWISE — it
            # is the logical connective on bool operands only.
            if any(v.aval.dtype != "bool" for v in producer.invars):
                dropped.append(
                    "'and' on non-bool operands is bit arithmetic, not "
                    "conjunction"
                )
                return
            for conj in producer.invars:
                self._apply_assumed_pred(conj, where, narrowed, dropped, vacuous)
            return
        if prim not in _ASSUME_CMPS:
            dropped.append(
                f"predicate produced by {prim!r} admits no sound box narrowing"
            )
            return
        if len(producer.invars) != 2:
            dropped.append(
                f"comparison {prim!r} with {len(producer.invars)} operand(s)"
            )
            return
        if self.semantics == "ieee":
            # re-attack U2's swept surface: assume classification consumes
            # the comparison EQUATION directly, bypassing the guarded
            # comparison transfer — the same invariant must be enforced
            # here too. A non-f64-float comparison must neither narrow,
            # nor certify satisfiability, nor raise the unsatisfiable-
            # precondition oracle (a "definitely false" f32-band
            # comparison can be TRUE at runtime under per-dtype DAZ:
            # measured float32(1e-45) == float32(1e-40) is True) — inert
            # with the gap quoted is the only sound posture.
            bad = _non_f64_float_dtypes(producer.invars)
            if bad:
                dropped.append(
                    f"ieee mode models binary64 only; {'/'.join(bad)} "
                    f"comparison semantics (incl. per-dtype subnormal "
                    f"flush) are not modelled — no narrowing, no "
                    f"satisfiability claim"
                )
                return
        a, b = producer.invars
        box_a, box_b = self._quiet_interval(a), self._quiet_interval(b)
        point_a, point_b = _finite_point(box_a), _finite_point(box_b)
        if point_a and point_b:
            # no variable to narrow — but a point-vs-point comparison is
            # DECIDABLE (never indeterminate: order comparisons of two
            # degenerate finite intervals are definite, and two points are
            # either equal or disjoint). Definitely false means the
            # precondition is unsatisfiable over the declared set — the
            # same harness defect the empty meet names, so the same loud
            # refusal. Definitely true stays a no-op inert drop.
            try:
                result = _CMP_FN[prim](box_a, box_b)
            except iv.IntervalError as e:
                dropped.append(
                    f"point comparison sides do not broadcast ({e})"
                )
                return
            if any(
                (lo, hi) == iv.BOOL_FALSE
                for lo, hi in zip(result.los, result.his)
            ):
                # under ieee a NaN side would ALSO falsify the comparison,
                # so the definitely-false refusal stands with or without a
                # maybe-NaN flag on either side
                self._unsatisfiable(
                    where,
                    f"constant comparison {_render_bound(box_a)} "
                    f"{_CMP_SYMBOL[prim]} {_render_bound(box_b)} is "
                    f"definitely false",
                    vacuous,
                    f"unsatisfiable assume at {where}: the assumed "
                    f"comparison {_render_bound(box_a)} {_CMP_SYMBOL[prim]} "
                    f"{_render_bound(box_b)} is definitely false — the "
                    f"precondition is unsatisfiable over the declared set "
                    f"and every downstream obligation would be vacuous "
                    f"(harness defect; nothing was verified)",
                )
                return
            if self.semantics == "ieee" and (
                self.read_flag(a) or self.read_flag(b)
            ):
                # the interval parts compare definitely true, but a NaN
                # side would falsify — "definitely true" cannot be claimed
                dropped.append(
                    "a comparison side may be NaN under ieee semantics "
                    "(NaN falsifies the comparison) — the assumed "
                    "comparison is not certified true; dropped"
                )
                return
            dropped.append(
                "both comparison sides are point intervals — nothing to "
                "narrow (the assumed comparison is definitely true)"
            )
            return
        if not point_a and not point_b:
            # audit F6: a constant-but-not-finite bound (±inf, NaN, an
            # undecodable literal) is not a varying side — the disclosed
            # reason must name the actual shape. Under ieee a subnormal-
            # band literal is hazed to a non-point for the same reason,
            # and gets its own truthful reason.
            if self.semantics == "ieee" and (
                _subnormal_const_literal(a) or _subnormal_const_literal(b)
            ):
                dropped.append(_SUBNORMAL_BOUND_REASON)
            elif _nonfinite_const(a, box_a) or _nonfinite_const(b, box_b):
                dropped.append(_NONFINITE_BOUND_REASON)
            else:
                dropped.append(_RELATIONAL_REASON)
            return
        if point_a:
            target_atom, target_box, bound, bound_atom = b, box_b, box_a, a
            cmp = _CMP_FLIP[prim]  # cmp(k, v) === flipped-cmp(v, k)
        else:
            target_atom, target_box, bound, bound_atom = a, box_a, box_b, b
            cmp = prim
        if self.semantics == "ieee" and self.read_flag(bound_atom):
            # a maybe-NaN bound: if the bound IS NaN the true assumed
            # region is empty (NaN falsifies every _ASSUME_CMPS
            # comparison), so a half-space built from its interval could
            # certify a vacuous precondition — no certified half-space
            # represents a maybe-NaN bound; inert is the sound posture
            dropped.append(
                "the comparison bound may be NaN under ieee semantics "
                "(its producer carries maybe-NaN) — no certified "
                "half-space represents it"
            )
            return
        if not isinstance(target_atom, ir.Var):
            dropped.append(
                "the varying comparison side is a literal, not an "
                "environment variable"
            )
            return
        if target_box is None:
            dropped.append(
                f"var {target_atom.id} has no propagated interval to narrow"
            )
            return
        # bound shape: a scalar broadcast over the variable, or an exact
        # elementwise match — nothing else (general broadcasting between
        # the bound and the variable is not attempted; inert is sound).
        if not (bound.is_scalar() or bound.shape == target_box.shape):
            dropped.append(
                f"bound shape {bound.shape} is neither scalar nor equal to "
                f"the constrained variable's shape {target_box.shape}"
            )
            return
        n = target_box.size
        ks = bound.los * n if bound.is_scalar() else bound.los
        # the CLOSED half-space, deliberately: for strict cmps the true
        # ℝ-region (k, hi] must be contained in the narrowing, and
        # [nextafter(k), hi] excludes the reals in (k, nextafter(k)) — an
        # unsound under-approximation (binding rule 1).
        if cmp in ("ge", "gt"):
            half = iv.IntervalArray(
                shape=target_box.shape, los=ks, his=(math.inf,) * n
            )
        elif cmp in ("le", "lt"):
            half = iv.IntervalArray(
                shape=target_box.shape, los=(-math.inf,) * n, his=ks
            )
        else:  # eq
            half = iv.IntervalArray(shape=target_box.shape, los=ks, his=ks)
        try:
            new = iv.meet(target_box, half)  # exact: no outward rounding
        except iv.IntervalError:
            self._unsatisfiable(
                where,
                f"var {target_atom.id} ∈ {_render_box(target_box)} cannot "
                f"satisfy {_CMP_SYMBOL[cmp]} {_render_bound(bound)} (empty "
                f"meet)",
                vacuous,
                f"unsatisfiable assume at {where}: var {target_atom.id} has "
                f"propagated interval {_render_box(target_box)}, but the "
                f"assumed constraint requires var {target_atom.id} "
                f"{_CMP_SYMBOL[cmp]} {_render_bound(bound)}; the "
                f"precondition is definitely false on the whole "
                f"over-approximated domain, so the declared set as assumed "
                f"is empty and every downstream obligation would be vacuous "
                f"(harness defect; nothing was verified)",
            )
            return
        if cmp in ("gt", "lt") and any(
            new.los[i] == new.his[i] == ks[i] for i in range(n)
        ):
            # audit F1: a STRICT constraint whose meet collapses onto the
            # closed boundary point has an EMPTY true region ((k, k] or
            # [k, k)) — the closed narrowing cannot represent the
            # exclusion, but the emptiness is decidable right here, and a
            # definite verdict judged over the nonempty stand-in [k, k]
            # would misstate a claim about an empty admitted set (the
            # REFUTED face mints a false refutation). Same refusal class
            # as the empty meet. Elementwise: any collapsed element
            # empties the universal predicate.
            self._unsatisfiable(
                where,
                f"strict constraint var {target_atom.id} {_CMP_SYMBOL[cmp]} "
                f"{_render_bound(bound)} collapses onto the boundary point "
                f"— the true region is empty",
                vacuous,
                f"unsatisfiable assume at {where}: the strict constraint "
                f"var {target_atom.id} {_CMP_SYMBOL[cmp]} "
                f"{_render_bound(bound)} collapses the narrowed interval "
                f"onto the closed boundary point {_render_box(new)}, which "
                f"the strict comparison itself excludes — the true assumed "
                f"region is empty, so the precondition is definitely false "
                f"on the whole over-approximated domain and every "
                f"downstream obligation would be vacuous (harness defect; "
                f"nothing was verified)",
            )
            return
        # audit F8: an assume whose predicate is DEFINITELY TRUE over the
        # target's whole box certifies itself, exactness aside — pred true
        # on the box superset is true on the whole current domain, so the
        # assumed region EQUALS the (nonempty) reachable set and no
        # vacuity can arise from this assume. Checked with the
        # comparator's own strictness: a no-op meet does NOT suffice for
        # gt/lt at the boundary (box lo == k leaves x == k reachable and
        # excluded — pred not definitely true there), so the strict forms
        # demand strict inequality against the bound.
        if cmp == "ge":
            def_true = all(lo >= k for lo, k in zip(target_box.los, ks))
        elif cmp == "gt":
            def_true = all(lo > k for lo, k in zip(target_box.los, ks))
        elif cmp == "le":
            def_true = all(hi <= k for hi, k in zip(target_box.his, ks))
        elif cmp == "lt":
            def_true = all(hi < k for hi, k in zip(target_box.his, ks))
        else:  # eq: definitely true only if the box IS the bound point
            def_true = all(
                lo == hi == k
                for lo, hi, k in zip(target_box.los, target_box.his, ks)
            )
        if self.semantics == "ieee" and self.read_flag(target_atom):
            # a maybe-NaN target: NaN falsifies the assumed comparison, so
            # the predicate is NOT definitely true over the value set even
            # when the interval part is — the audit-F8 self-certification
            # channel stays closed for flagged targets
            def_true = False
        self.env[target_atom.id] = new
        if self.semantics == "ieee" and self.nan.get(target_atom.id):
            # an assumed-true comparison excludes NaN (NaN would falsify
            # it), so the narrowed target's maybe-NaN flag is soundly
            # cleared — the clearing is a judgement call the spec flags;
            # disclosed here so no verdict rests on it silently
            self.nan[target_atom.id] = False
            self.notes.append(
                f"assume cleared maybe-NaN on var {target_atom.id} at "
                f"{where}: an assumed-true comparison excludes NaN"
            )
        narrowed.append(
            (
                target_atom.id,
                new,
                new != target_box,
                # certified iff the target's box is EXACT (its true value
                # set: a stelling_any output or an exact-point const,
                # narrowed only by other assumes — meets of exact sets are
                # exact), OR the predicate is definitely true over the box
                # (audit F8). Everything else — a cutting assume on a
                # transfer output — is an over-approximation (audit F7).
                # The decision is the shared primitive in
                # stelling.exactness (module-attr call, so tests can pin
                # the routing). Under ieee, a STRICT certification
                # additionally needs a flush-robust witness: a strict
                # region whose only content is subnormal may be empty at
                # runtime on a DAZ target (its members read as 0, which
                # the strict comparison excludes) — without one the
                # precondition stays uncertified (indeterminate, never
                # definite).
                exactness.certifies_nonemptiness(
                    self.exact, target_atom.id, definitely_true=def_true
                )
                and not (
                    self.semantics == "ieee"
                    and cmp in ("gt", "lt")
                    and not def_true
                    and not all(
                        _strict_flush_witness(cmp, k, nlo, nhi)
                        for k, nlo, nhi in zip(ks, new.los, new.his)
                    )
                ),
            )
        )

    def run(
        self, jaxpr: ir.Jaxpr, consts, args, arg_flags=None, arg_taints=None
    ) -> list[iv.IntervalArray]:
        ieee = self.semantics == "ieee"
        for var, c in zip(jaxpr.constvars, consts):
            if isinstance(c, iv.IntervalArray):
                self.env[var.id] = c  # pre-boxed const: provenance unknown,
                if ieee:  # so non-exact (conservative — audit F7) and,
                    # under ieee, maybe-NaN (a box of unknown provenance
                    # carries no NaN-freedom claim)
                    self.nan[var.id] = True
                continue
            refused = _refused_value_problem(var.aval.shape, c)
            if refused is not None:
                # the REFUSED class (no coherent value exists): bind and
                # register as one operation, so the read gate declines
                # every consumer regardless of what their avals claim
                # (fix re-attack P1 — the constvar route was half-gated)
                self.notes.append(
                    f"constvar {var.id} refused: {refused}; ⊤"
                )
                self._bind_refused(var)
                continue
            try:
                box = _value_to_interval(c, var.aval.shape)
            except (iv.IntervalError, ir.TranscriptionError) as e:
                # same posture as literals: an unrepresentable const binds ⊤
                # (audit-gate finding 1 — a NaN closure const killed the run)
                # — the INHABITED-unknown class: the value exists, only its
                # bracket does not, so it participates as ⊤ (registered)
                self.notes.append(f"const outside the domain ({e}); ⊤")
                self.env[var.id] = _safe_top(var.aval.shape)
                if ieee:  # ⊤ under ieee is maybe-NaN
                    self.nan[var.id] = True
                continue
            if ieee:
                # DAZ flushes inputs: constants entering ieee propagation
                # are subnormal-hazed; a band const stops being a point,
                # so mark_if_point below withholds exactness by itself
                box = iv.subnormal_haze(box)[0]
            self.env[var.id] = box
            # exact iff the decoded box is a point per element — a >2**53
            # int decodes to a genuine bracket, which is NOT its value set
            self.exact.mark_if_point(var.id, box.los, box.his)
        for i, (var, a) in enumerate(zip(jaxpr.invars, args)):
            self.env[var.id] = a
            if ieee and arg_flags is not None and arg_flags[i]:
                self.nan[var.id] = True
            # taint crosses scope boundaries with the value it marks
            if ieee and arg_taints is not None and arg_taints[i]:
                self.taint[var.id] = True
        # assume classification looks up the predicate's producing equation
        # at the CURRENT jaxpr level only; sub-jaxpr runs (transparent
        # wrappers, cond branches) get their own map, restored on exit —
        # same scoping discipline as the env swap the callers perform.
        prev_producers = self.producers
        self.producers = {
            out.id: e for e in jaxpr.eqns for out in e.outvars
        }
        try:
            for eqn in jaxpr.eqns:
                self.eqn(eqn)
        finally:
            self.producers = prev_producers
        return [self.read(o) for o in jaxpr.outvars]

    def _shape_refusal(self, eqn: ir.JaxprEqn) -> str | None:
        """The uninhabited/malformed-shape refusal reason for an equation,
        or None. THE property gate (fix re-attacks R1/N1): a value from a
        refused negative-shape declaration is identified by its VAR ID at
        the env reference — never by classifying the consumer, whose
        recorded avals a from_dict query can simply lie about (the N1
        laundering: a consumer claiming a scalar aval for a refused id
        read the ⊤ stand-in as a real value, and ⊤·0 = [0,0] minted a
        definite face over an empty declared set). Every equation's env
        reads go through its invars, checked here before any read, so no
        aval lie can route around the gate. Also refused here: any
        negative or non-integer extent in the equation's own recorded
        avals or in an inline array literal's payload shape (from_dict
        does not coerce shape entry types, so `d < 0` on a string raised
        raw before the integral check)."""
        for atom in eqn.invars:
            if isinstance(atom, ir.Var) and atom.id in self.refused_shape:
                return (
                    "reads a value from a refused negative-shape "
                    "declaration"
                )
        for atom in (*eqn.invars, *eqn.outvars):
            if isinstance(atom, ir.Literal):
                # the shared refused-class predicate covers the recorded
                # aval, the payload's own shape, AND the payload length
                # (a length-lying literal has no coherent value either —
                # the same class one predicate over, unified here so the
                # interval path cannot ⊤-launder what the slicer refuses)
                problem = _refused_value_problem(atom.aval.shape, atom.val)
                if problem is not None:
                    return f"carries a refused literal ({problem})"
                continue
            for d in atom.aval.shape:
                try:
                    k = _op_index(d)
                except TypeError:
                    return (
                        f"carries a non-integer shape extent {d!r} "
                        f"(malformed IR: from_dict does not coerce "
                        f"shape entries)"
                    )
                if k < 0:
                    return (
                        "touches a negative-extent shape (no jax "
                        "program constructs such a value)"
                    )
        return None

    def _bind_refused(self, var: ir.Var) -> None:
        """Bind a refused-shape value: the stand-in env entry and the
        ``refused_shape`` membership are ONE operation. The separable
        pair was P1's defect — a constvar decline bound the stand-in
        without registering the id, and the read gate then cleared its
        consumers; no call site can now do one without the other (L7).
        Under ieee the stand-in is maybe-NaN, like every ⊤-out."""
        self.env[var.id] = _safe_top(var.aval.shape)
        self.refused_shape.add(var.id)
        if self.semantics == "ieee":
            self.nan[var.id] = True

    def eqn(self, eqn: ir.JaxprEqn) -> None:
        params = eqn.params_dict()
        refusal = self._shape_refusal(eqn)
        if refusal is not None:
            # the negative/malformed-shape screen (fix-re-attacks R1/N1).
            # stelling_assert/stelling_nonvacuity stay EXEMPT from
            # declining so the obligation/check is still RECORDED (judged
            # below over the ⊤ stand-in, which supports no definite
            # face; the stand-in never participates in arithmetic because
            # this gate declines every computing consumer first) — but
            # their OUTPUTS join the refused set: refusal is a property
            # of the VALUE and follows its id through every binding.
            if eqn.primitive in ("stelling_assert", "stelling_nonvacuity"):
                for out in eqn.outvars:
                    self.refused_shape.add(out.id)
            else:
                self.notes.append(f"{eqn.primitive!r} {refusal}; ⊤")
                self.counter.record_unknown(eqn.primitive)
                # sub-jaxprs of a refused equation were never analyzed:
                # unreached, so the denominator stays a function of the
                # program on this decline path too (third audit, F1)
                self.mark_unreached(eqn)
                for out in eqn.outvars:
                    self._bind_refused(out)
                return
        if eqn.primitive in DEFAULT_TRANSPARENT:
            inner = next(
                (v for _, v in eqn.params if isinstance(v, ir.ClosedJaxpr)), None
            )
            ins = [self.read(a) for a in eqn.invars]
            if inner is not None and len(inner.jaxpr.invars) == len(ins):
                self.counter.record_transparent(eqn.primitive)
                ieee = self.semantics == "ieee"
                in_flags = (
                    [self.read_flag(a) for a in eqn.invars] if ieee else None
                )
                in_taints = (
                    [self.read_taint(a) for a in eqn.invars] if ieee else None
                )
                outer_env = self.env  # isolated scope, as for cond branches
                outer_exact = self.exact
                outer_nan = self.nan
                outer_taint = self.taint
                self.env = {}
                self.exact = exactness.ExactSet()
                self.nan = {}
                self.taint = {}
                out_flags = None
                out_taints = None
                try:
                    outs = self.run(
                        inner.jaxpr, inner.consts, ins, in_flags, in_taints
                    )
                    if ieee:
                        out_flags = [
                            self.read_flag(o) for o in inner.jaxpr.outvars
                        ]
                        out_taints = [
                            self.read_taint(o) for o in inner.jaxpr.outvars
                        ]
                finally:
                    self.env = outer_env
                    self.exact = outer_exact
                    self.nan = outer_nan
                    self.taint = outer_taint
                for out, val, iout in zip(
                    eqn.outvars, outs, inner.jaxpr.outvars
                ):
                    # refusal follows the value across the call boundary:
                    # an inner outvar refused for its shape refuses the
                    # call's outvar too (fix re-attack N1/P1 — bind and
                    # register as one operation)
                    if (
                        isinstance(iout, ir.Var)
                        and iout.id in self.refused_shape
                    ):
                        self._bind_refused(out)
                    else:
                        self.env[out.id] = val
                if ieee:
                    for out, f in zip(eqn.outvars, out_flags):
                        self.nan[out.id] = f
                    for out, t in zip(eqn.outvars, out_taints):
                        self.taint[out.id] = t
                return
            self.notes.append(
                f"transparent {eqn.primitive!r}: arity mismatch or no sub-jaxpr; ⊤"
            )
            self.counter.record_unknown(eqn.primitive)
            self.mark_unreached(eqn)
            self.top_out(eqn)
            return

        if eqn.primitive == "cond":
            # cond(index, *operands)[branches=(jaxpr, ...)]: run every branch
            # the index interval can select and JOIN outputs (sound
            # over-approximation of an undetermined branch). Out-of-range
            # indices select the FINAL branch — the jax convention, verified
            # by binding cond_p directly (index -1 -> last, index 5 -> last;
            # audit finding 2: clamping negative indices to branch 0 produced
            # a verified false VERIFIED at the [-1, 0] boundary).
            branches = next(
                (v for k, v in eqn.params if k == "branches"), None
            )
            ins = [self.read(a) for a in eqn.invars]
            if branches is None or not ins:
                self.notes.append("cond without branches/operands; ⊤")
                self.counter.record_unknown(eqn.primitive)
                self.mark_unreached(eqn)
                self.top_out(eqn)
                return
            ieee = self.semantics == "ieee"
            op_flags = (
                [self.read_flag(a) for a in eqn.invars[1:]] if ieee else None
            )
            op_taints = (
                [self.read_taint(a) for a in eqn.invars[1:]] if ieee else None
            )
            # under ieee the index invariant is enforced, not assumed
            # (the F1/U2 lesson): jax's cond index is always int32, so a
            # FLOAT-dtyped index is out-of-contract hand-built IR whose
            # value could flush per-dtype — its interval is untrusted and
            # every branch is joined; same for a maybe-NaN-flagged index
            index_untrusted = ieee and (
                self.read_flag(eqn.invars[0])
                or eqn.invars[0].aval.dtype not in _SELECTOR_DTYPES
            )
            index, operands = ins[0], ins[1:]
            last = len(branches) - 1
            w_lo, w_hi = index.los[0], index.his[0]
            if w_lo == -math.inf or w_hi == math.inf or index_untrusted:
                # ⊤ index, or an untrusted (flagged / non-int) index
                # under ieee: any branch
                possible = set(range(last + 1))
            else:
                lo_i, hi_i = int(math.floor(w_lo)), int(math.floor(w_hi))
                possible = {i for i in range(last + 1) if lo_i <= i <= hi_i}
                if lo_i < 0 or hi_i > last:
                    possible.add(last)  # out-of-range mass -> the final branch
            if not possible:
                possible = {last}
            self.counter.record_known(eqn.primitive)
            self.used[eqn.primitive] = TIER_EXACT
            # untaken branches are still part of the query: their equations
            # count as unreached, or the coverage denominator lies (audit
            # finding 4)
            for i, b in enumerate(branches):
                if i not in possible:
                    stack = [b.jaxpr]
                    while stack:
                        jx = stack.pop()
                        for e in jx.eqns:
                            self.counter.record_unreached(e.primitive)
                            stack.extend(sub_jaxprs(e))
            # each branch runs in an isolated scope: branch jaxprs are closed
            # (invars + consts only), and a shared flat env would let one
            # branch read another's internals, defeating the unbound-var
            # check (audit finding 6)
            outer_env = self.env
            outer_exact = self.exact
            outer_nan = self.nan
            outer_taint = self.taint
            results = []
            branch_flags = []  # ieee: per-branch outvar flags
            branch_taints = []  # ieee: per-branch outvar product-taints
            # every branch here is possibly-untaken from the analysis's
            # view (the selector interval admits it, nothing more): an
            # unsatisfiable assume inside is a branch-scoped vacuity, not
            # a whole-domain harness defect (audit F2) — branch_depth
            # switches the unsatisfiability posture from raise to a
            # branch-local note. Conservative even for a definite
            # single-branch selector: not raising is always sound.
            self.branch_depth += 1
            try:
                for i in sorted(possible):
                    b = branches[i]
                    self.env = {}
                    self.exact = exactness.ExactSet()
                    self.nan = {}
                    self.taint = {}
                    results.append(
                        self.run(
                            b.jaxpr, list(b.consts), operands, op_flags,
                            op_taints,
                        )
                    )
                    if ieee:
                        branch_flags.append(
                            [self.read_flag(o) for o in b.jaxpr.outvars]
                        )
                        branch_taints.append(
                            [self.read_taint(o) for o in b.jaxpr.outvars]
                        )
            finally:
                self.branch_depth -= 1
                self.env = outer_env
                self.exact = outer_exact
                self.nan = outer_nan
                self.taint = outer_taint
            branch_refused = [
                any(
                    isinstance(branches[i].jaxpr.outvars[j], ir.Var)
                    and branches[i].jaxpr.outvars[j].id in self.refused_shape
                    for i in possible
                )
                for j in range(len(eqn.outvars))
            ]
            for j, out in enumerate(eqn.outvars):
                if branch_refused[j]:
                    # a possible branch's output comes from a refused
                    # shape: the join would mix a stand-in with real
                    # boxes — the cond output is refused too (N1)
                    self.notes.append(
                        f"cond output {j} comes from a refused "
                        f"negative-shape value in a possible branch; ⊤"
                    )
                    self._bind_refused(out)
                    continue
                try:
                    self.env[out.id] = iv.join([r[j] for r in results])
                except iv.IntervalError as e:
                    # branches returning mismatched shapes (malformed IR)
                    # previously escaped as a raw raise from the join —
                    # degrade-don't-crash, quoted
                    self.notes.append(f"cond output join declined: {e}; ⊤")
                    self.env[out.id] = _safe_top(out.aval.shape)
                    if ieee:
                        self.nan[out.id] = True
                        self.taint[out.id] = False
                    continue
                if ieee:
                    # the output is SOME branch's output whatever the
                    # index value is (out-of-range selects the final
                    # branch), so the join's flag is the OR over possible
                    # branches — the index's own flag does not propagate
                    self.nan[out.id] = any(f[j] for f in branch_flags)
                    # the product taint joins exactly the same way, and for
                    # the same reason: the output IS some branch's output,
                    # so if any possible branch produced it from a multiply
                    # the join is product-derived. Writing this was stated
                    # in the build and NOT done (audit COSMETIC 7) — the
                    # join defaulted the taint to False and laundered it.
                    # It was harmless only because XLA does not currently
                    # contract across a cond boundary on this target, which
                    # is precisely the kind of compiler behaviour the taint
                    # exists so as not to depend on.
                    self.taint[out.id] = any(t[j] for t in branch_taints)
            return

        if eqn.primitive == "stelling_assume":
            # value semantics: the identity on the predicate — the assume's
            # output passes its input through unchanged in BOTH modes
            # (conservative: the output interval computed before any
            # narrowing is a superset of the predicate's value under the
            # assumption). The constraint semantics differ by mode.
            ins = [self.read(a) for a in eqn.invars]
            in_flags = (
                [self.read_flag(a) for a in eqn.invars]
                if self.semantics == "ieee"
                else None
            )
            where = eqn.source_info[-1] if eqn.source_info else "unknown location"
            if self.assume_mode == "constrain" and eqn.invars:
                self._assume_constrain(eqn, where)
            else:
                # inert (amendment 2): sound, counted, addressed — never
                # silent, never "known". Byte-identical to the MVP note.
                self.counter.record_inert(eqn.primitive)
                self.notes.append(
                    ASSUME_DROP_NOTE.format(where=where)
                )
            in_taints = (
                [self.read_taint(a) for a in eqn.invars]
                if self.semantics == "ieee"
                else None
            )
            for i, (out, val) in enumerate(zip(eqn.outvars, ins)):
                self.env[out.id] = val
                if in_flags is not None:
                    self.nan[out.id] = in_flags[i]
                if in_taints is not None:
                    self.taint[out.id] = in_taints[i]
            return

        ieee = self.semantics == "ieee"
        entry = (IEEE_TRANSFERS if ieee else TRANSFERS).get(eqn.primitive)
        if entry is None:
            self.counter.record_unknown(eqn.primitive)
            self.mark_unreached(eqn)
            self.top_out(eqn)
            return

        transfer, tier = entry
        ins = [self.read(a) for a in eqn.invars]
        in_flags = [self.read_flag(a) for a in eqn.invars] if ieee else None
        try:
            result = (
                transfer(eqn, params, ins, in_flags)
                if ieee
                else transfer(eqn, params, ins)
            )
        except iv.IntervalError as e:
            # a transfer whose domain doesn't cover this legal form (rank
            # broadcasting, batched selectors, …) DECLINES: sound ⊤
            # degradation with the reason quoted — the registered
            # degrade-don't-crash posture (second audit, FRAGILE 5; the
            # shape guards previously killed the whole analysis here).
            # Under ieee, top_out marks the outputs maybe-NaN.
            self.notes.append(f"{eqn.primitive!r} declined this form: {e}; ⊤")
            self.counter.record_unknown(eqn.primitive)
            # the coverage denominator is a function of the PROGRAM, never
            # of the outcome (third audit, F1): an equation carrying a
            # sub-jaxpr (scatter-add's recorded combiner) whose transfer
            # declines never analyzed those inner equations — they count
            # UNREACHED, exactly as under an unregistered primitive, so
            # the same program reports the same total on the success path
            # (inner add: known), every decline path (unreached), and
            # under every semantics dial. Before this, the inner equation
            # silently vanished from the total on declines (real 6 vs
            # ieee 5 on one program). A no-op for every sub-jaxpr-free
            # equation.
            self.mark_unreached(eqn)
            self.top_out(eqn)
            return
        if result is None:  # a known transfer declining this configuration
            self.notes.append(
                f"{eqn.primitive!r} has no sound rule for params "
                f"{ {k: v for k, v in params.items() if not isinstance(v, ir.ClosedJaxpr)} }; ⊤"
            )
            self.counter.record_unknown(eqn.primitive)
            # same accounting as the IntervalError decline above: inner
            # equations of a declined form count unreached, keeping the
            # denominator outcome-independent (third audit, F1)
            self.mark_unreached(eqn)
            self.top_out(eqn)
            return
        outs, out_flags = result if ieee else (result, None)
        if (
            ieee
            and eqn.primitive == "reduce_sum"
            and self.read_taint(eqn.invars[0])
            and ins[0].size > 1
        ):
            # a 2-element reduce_sum IS an addition, so a product among its
            # elements can be contracted exactly as at an `add`. The
            # per-array taint says the array is product-derived but not
            # WHICH element is, so the contracted value cannot be bracketed
            # here — decline with the reason quoted (audit UNSOUND 5: a
            # one-element reduce_sum was one of the ten forms, and the
            # two-element case is where the addition itself appears).
            self.notes.append(
                f"'reduce_sum' declined this form: "
                f"{iv.IEEE_CONTRACTION_DECLINE.format(why='the reduction performs an addition over a product-derived array, and the per-array taint does not say which element carries the product')}; ⊤"
            )
            self.counter.record_unknown(eqn.primitive)
            self.top_out(eqn)
            return
        if ieee and eqn.primitive in IEEE_CONTRACTION_ADDENDS:
            # UNSOUND 4: XLA may CONTRACT a product feeding this add/sub
            # into one fused multiply-add. Both roundings are legal for the
            # same jaxpr, so the result must cover both.
            #
            # `add_any` is HERE, and was missing for one commit. It is a
            # by-NAME gate, and registering `add_any` with the same IEEE
            # function object as `add` satisfied every census constraint
            # while leaving this tuple — and therefore the contraction hull
            # — untouched. Measured consequence, jax 0.11.0 CPU under jit,
            # HLO `multiply_add_fusion`: the box was [0.0, 0.0] where the
            # compiled program returns 4.930380657631324e-32, and the
            # obligation `out <= 0.0` DISCHARGED. A false verdict, with
            # IEEE_CONTRACTION_ASSUMPTION stamped on the run — the stamp
            # asserting the hazard was modelled at the equation where it
            # was not. `add` on the identical program returned unknown.
            #
            # `negate_product`/`negate_addend` below test `== "sub"`, so
            # both stay False for `add_any`, which is correct: it is an
            # addition.
            try:
                outs, out_flags = self._contraction_hull(eqn, outs, out_flags)
            except iv.IntervalError as e:
                self.notes.append(
                    f"{eqn.primitive!r} declined this form: {e}; ⊤"
                )
                self.counter.record_unknown(eqn.primitive)
                self.top_out(eqn)
                return
        self.counter.record_known(eqn.primitive)
        if eqn.primitive == "scatter-add":
            # the accumulate row is the first registered transfer whose
            # equation CARRIES a sub-jaxpr (the recorded add combiner).
            # Its inner equation was honored — the form oracle pinned it
            # to the single `add` the transfer just performed — so it
            # counts known here: the coverage denominator must neither
            # drop it silently (the audit-finding-4 vanishing class) nor
            # call it unreached (before this row it WAS unreached, and
            # the before/after totals must stay comparable)
            for j in sub_jaxprs(eqn):
                for e in j.eqns:
                    self.counter.record_known(e.primitive)
        self.used[eqn.primitive] = tier
        if tier == TIER_SOUND_LIBM:
            self.assumptions.add(
                _LIBM_ASSUMPTIONS.get(
                    eqn.primitive,
                    f"{eqn.primitive} endpoints assume a faithfully-rounded "
                    f"libm (error <= 1 ulp), bumped 1 ulp outward",
                )
            )
        if eqn.primitive == "stelling_assert":
            status, detail = _bool_status(ins[0], constrained=self.any_constrained)
            if ieee and in_flags and in_flags[0] and status != "unknown":
                # the predicate VALUE arrived flagged maybe-NaN (a decline
                # artifact ⊤ reaching the assert, or a flagged selector
                # path): a bool cannot BE NaN, so the flag marks untrusted
                # provenance — neither definite face is claimed
                where = (
                    eqn.source_info[-1] if eqn.source_info else "unknown location"
                )
                self.notes.append(
                    f"obligation withheld from a definite status at {where}: "
                    f"the predicate value carries maybe-NaN under ieee "
                    f"semantics (its producer was declined/unmodeled) — "
                    f"judged unknown"
                )
                status = "unknown"
                detail = (
                    "predicate value carries maybe-NaN under ieee semantics "
                    "(see notes); no definite status is claimed"
                )
            if status == "violated-over-set" and self.uncertified:
                # audit F7: a definite violation judged while an
                # UNCERTIFIED-precondition constraint is in force is
                # withheld from REFUTED — the narrowed superset it was
                # judged over may over-approximate an EMPTY true region,
                # and a possibly-vacuous refutation is not a refutation.
                where = (
                    eqn.source_info[-1] if eqn.source_info else "unknown location"
                )
                self.notes.append(
                    f"violation WITHHELD from REFUTED at {where}: a "
                    f"definite violation was found over the narrowed "
                    f"superset, but the precondition's satisfiability is "
                    f"uncertified (it constrains an over-approximated "
                    f"intermediate whose box may exceed its true image) — "
                    f"a possibly-vacuous refutation is not a refutation; "
                    f"REFUTED under constraining assumes requires "
                    f"certified-satisfiable preconditions"
                )
                status = "unknown"
                detail = (
                    "definite violation over the precondition-narrowed "
                    "superset WITHHELD from REFUTED (precondition "
                    "satisfiability uncertified; see notes)"
                )
            self.obligations.append(
                ObligationReport(
                    index=len(self.obligations),
                    status=status,
                    detail=detail,
                    source_info=eqn.source_info,
                )
            )
        if eqn.primitive == "stelling_nonvacuity":
            status, detail = _bool_status(ins[0], constrained=self.any_constrained)
            if ieee and in_flags and in_flags[0] and status != "unknown":
                # same posture as the assert above: a maybe-NaN membership
                # condition supports neither the checked nor the FAILED face
                where = (
                    eqn.source_info[-1] if eqn.source_info else "unknown location"
                )
                self.notes.append(
                    f"nonvacuity condition withheld from a definite status "
                    f"at {where}: the membership predicate carries "
                    f"maybe-NaN under ieee semantics — judged unknown"
                )
                status = "unknown"
                detail = (
                    "membership condition carries maybe-NaN under ieee "
                    "semantics (see notes); no definite status is claimed"
                )
            if status == "violated-over-set" and self.uncertified:
                # audit F9: the definite FAILED stamp sentence is the same
                # claim class the assert withholding guards — a membership
                # condition judged definitely false over an
                # uncertified-narrowed region may be vacuously so. Same
                # forward scoping as the assert withholding; certified and
                # exact runs keep FAILED byte-identically.
                where = (
                    eqn.source_info[-1] if eqn.source_info else "unknown location"
                )
                self.notes.append(
                    f"nonvacuity FAILED face WITHHELD at {where}: the "
                    f"membership condition was judged definitely false over "
                    f"an uncertified-narrowed region (a constraining assume "
                    f"cut an over-approximated intermediate whose box may "
                    f"exceed its true image) — the FAILED sentence is "
                    f"reserved for judgments not confounded by an "
                    f"uncertified constraint"
                )
                status = "unknown"
                detail = (
                    "membership condition definitely false over the "
                    "precondition-narrowed superset WITHHELD from FAILED "
                    "(precondition satisfiability uncertified; see notes)"
                )
            self.nonvacuity_checks.append(
                ObligationReport(
                    index=len(self.nonvacuity_checks),
                    status=status,
                    detail=detail,
                    source_info=eqn.source_info,
                )
            )
        for i, (out, val) in enumerate(zip(eqn.outvars, outs)):
            self.env[out.id] = val
            if ieee:
                self.nan[out.id] = out_flags[i]
                # IEEE_PRODUCT_SOURCES are the taint SOURCES; everything
                # else passes what it was given unless it is a registered stop
                self.taint[out.id] = eqn.primitive in IEEE_PRODUCT_SOURCES or (
                    eqn.primitive not in _TAINT_STOPS
                    and any(self.read_taint(a) for a in eqn.invars)
                )
        if eqn.primitive == "stelling_any":
            # the ONE transfer whose output box is exact: the declared
            # closed box IS the declared value set (no rounding at
            # declaration). Every other transfer output stays non-exact
            # (audit F7: rounding pads and correlation-blind arithmetic
            # can inflate a box past the true image). Under ieee, a
            # declaration the subnormal haze CHANGED is not exact: its
            # as-consumed value set is flush-indeterminate (the hazed box
            # is a sound hull of both semantics, not the declared set),
            # so it must not certify assume-satisfiability.
            if ieee:
                raw = iv.from_bounds(
                    tuple(params["shape"]),
                    float(params["lo"]),
                    float(params["hi"]),
                )
                if iv.subnormal_haze(raw)[1]:
                    return
            for out in eqn.outvars:
                self.exact.mark_declared(out.id)


def _check_assume_mode(assume_mode: str) -> None:
    if assume_mode not in _ASSUME_MODES:
        raise ValueError(
            f"assume_mode must be one of {_ASSUME_MODES}, got {assume_mode!r}"
        )


_SEMANTICS_MODES = ("real", "ieee")
_DOMAINS = ("interval",)

# The mechanical guard on the domain dial: quoted verbatim when a
# tightened (non-interval) domain is requested under semantics="real".
TIGHTENED_DOMAIN_REAL_REFUSAL = (
    "a tightened domain under semantics='real' is refused outright: "
    "tightening ℝ arithmetic without float semantics converts accidental "
    "UNKNOWNs into false VERIFIEDs (the interval slack was masking the "
    "ℝ-vs-float gap, not modeling it) — tightened domains run only under "
    "semantics='ieee'"
)


def _check_semantics(semantics: str) -> None:
    if semantics not in _SEMANTICS_MODES:
        raise ValueError(
            f"semantics must be one of {_SEMANTICS_MODES}, got {semantics!r}"
        )


def _check_domain(domain: str, semantics: str) -> None:
    """Guard 1: ``"interval"`` is the only registered domain. A
    non-interval value always raises; under ``semantics="real"`` the
    refusal carries the rationale a future tightened domain must not be
    allowed to erode (:data:`TIGHTENED_DOMAIN_REAL_REFUSAL`)."""
    if domain in _DOMAINS:
        return
    if semantics == "real":
        raise ValueError(
            f"domain must be one of {_DOMAINS} (the only registered "
            f"domain), got {domain!r}. {TIGHTENED_DOMAIN_REAL_REFUSAL}"
        )
    raise ValueError(
        f"domain must be one of {_DOMAINS} (the only registered domain), "
        f"got {domain!r}"
    )


def interval_env(
    closed: ir.ClosedJaxpr, *, assume_mode: str = "inert"
) -> dict[int, iv.IntervalArray]:
    """Read-only accessor: the top-level ``var id -> interval`` environment
    after the forward walk :func:`propagate` performs under ``assume_mode``.

    Exists for the escalation layer, whose division guard needs the
    propagated interval of a divisor ("emit ``div`` only if the divisor's
    interval definitely excludes 0"). Pure: it re-runs the identical
    traversal on a fresh propagator and returns a copy of the resulting
    environment; nothing observable by :func:`propagate` changes.

    The default here is ``"inert"``, deliberately NOT :func:`propagate`'s
    ``"constrain"`` default: this accessor's consumer reasons over the
    DECLARED box. The emitted SMT problem carries the ``stelling_any``
    bounds and never the assume constraints, so its guards' premises
    ("the divisor definitely excludes 0 *over the declared box*") must be
    judged on the un-narrowed environment — an assume-narrowed divisor
    interval would let ``div`` be emitted into a script whose domain
    still contains the zero, and a model at that zero would be
    misdiagnosed as emission infidelity. Un-narrowed intervals are wider,
    so every guard judged on them is conservative — inert is always
    sound. Callers that want the constrained view ask for it explicitly.
    """
    _check_assume_mode(assume_mode)
    p = _Propagator(assume_mode)
    if closed.jaxpr.invars:
        raise ir.TranscriptionError(
            "propagate expects a self-contained harness query (inputs declared "
            f"via any_array), got {len(closed.jaxpr.invars)} free invar(s)"
        )
    p.run(closed.jaxpr, list(closed.consts), [])
    return dict(p.env)


def propagate(
    closed: ir.ClosedJaxpr,
    *,
    semantics: str = "real",
    assume_mode: str = "constrain",
    domain: str = "interval",
) -> Propagation:
    """Forward-propagate the declared boxes through a transcribed query and
    judge every ``stelling_assert`` obligation.

    ``semantics="real"`` (the default) judges obligations in exact real
    arithmetic — byte-identical to the pre-dial behavior;
    ``semantics="ieee"`` judges them about the traced program's IEEE
    binary64 round-to-nearest execution (see the module docstring). Any
    other value raises :class:`ValueError`. The returned
    :class:`Propagation` records which semantics ran.

    ``assume_mode="constrain"`` (the default) narrows the propagated
    domain at each soundly-constrainable ``stelling_assume`` (see the
    module docstring); ``assume_mode="inert"`` reproduces the
    drop-every-assume MVP behavior byte-identically (notes, coverage,
    env) — the comparability control for the vacuity instrument. Any
    other value raises :class:`ValueError`.

    ``domain="interval"`` is the only registered abstract domain. Any
    other value raises :class:`ValueError`; under ``semantics="real"``
    the refusal quotes why a tightened domain may never run there
    (:data:`TIGHTENED_DOMAIN_REAL_REFUSAL`).
    """
    _check_semantics(semantics)
    _check_assume_mode(assume_mode)
    _check_domain(domain, semantics)
    p = _Propagator(assume_mode, semantics)
    if closed.jaxpr.invars:
        raise ir.TranscriptionError(
            "propagate expects a self-contained harness query (inputs declared "
            f"via any_array), got {len(closed.jaxpr.invars)} free invar(s)"
        )
    p.run(closed.jaxpr, list(closed.consts), [])
    assumptions = set(p.assumptions)
    if semantics == "ieee":
        # the mode-wide stamped assumptions: how ieee endpoints are
        # computed and what their soundness relies on, and the
        # subnormal-band indeterminacy (flush-vs-gradual is
        # target-dependent; band outcomes are never definite)
        assumptions.add(iv.IEEE_ENDPOINT_ASSUMPTION)
        assumptions.add(iv.SUBNORMAL_INDETERMINACY_ASSUMPTION)
        # the equation-order reliance, disclosed once the three-row round's
        # reduce_sum decline made the contrast load-bearing (audit
        # COSMETIC 4): modelling an `add` chain while refusing a reduction
        # rests on the compiler not reassociating ACROSS equations
        assumptions.add(iv.IEEE_EQUATION_ORDER_ASSUMPTION)
        # contraction is the OTHER compiler freedom over the same equation
        # order, and it is measured TRUE on this target — so it is modelled
        # (a hull over both roundings), never assumed away
        assumptions.add(iv.IEEE_CONTRACTION_ASSUMPTION)
        # the mode's measured precision boundary, disclosed so a non-green
        # under ieee is read against it rather than as a float finding
        assumptions.add(iv.IEEE_NAN_HYGIENE_SCOPE)
    return Propagation(
        obligations=tuple(p.obligations),
        nonvacuity_checks=tuple(p.nonvacuity_checks),
        coverage=p.counter.freeze(),
        transfers_used=tuple(sorted(p.used.items())),
        assumptions=tuple(sorted(assumptions)),
        notes=tuple(p.notes),
        semantics=semantics,
        assume_dropped=p.assume_dropped,
    )
