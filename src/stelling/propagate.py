# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Forward interval propagation over :mod:`stelling.ir`.

The autodidax pattern: walk the equations of a transcribed query in order,
mapping each primitive through a transfer function over
:class:`stelling.interval.IntervalArray`. Consumes the IR, never jaxprs.

It also OWNS THE QUERY IDENTITY that binds a :class:`Propagation` to the
query it is about — :func:`query_identity`, the field
:attr:`Propagation.query_sha256` it writes, and :func:`unpaired_propagation`,
the one comparison every consumer of a propagation makes. That is not a
widening of the scope claim below (it decides nothing about any program); it
is here because the identity belongs on the object that carries the
judgements, and because one derivation read by five call sites cannot drift
the way five copies can. See :func:`unpaired_propagation` for what each site
does with the answer.

Scope, held deliberately (design/e2a-registration.md): no widening, no
fixpoints, no ``scan``/``while_loop`` descent, no solver.

**THAT CLAUSE READ "no cond/scan descent", AND THE ``cond`` HALF OF IT WAS
FALSE.** It was false against this same docstring, which says below that
*"the walk enters the transparent wrappers and ``cond``"*, and false against
the code, where :meth:`_Propagator.eqn` takes a ``cond`` arm that walks every
branch under a ``("cond", eqn_index, branch_index)`` scope step. It has been
false since control-flow descent was built; the half that stayed true is
``scan`` and ``while_loop``, which is the half the assume-recording rule
below rests on — an assume the walk never reaches is unclassified, and the
paragraph beginning "AN ASSUME THE WALK NEVER REACHES" is what says so.

The transfer registry contains exactly the primitives the target census
returned (`design/primitive-census.md`, "The target census") plus the three
harness primitives, plus the closed pytree-probe registration round
(abs, eq, ne, and, or, stop_gradient, reshape, pow, reduce_or, and the
scalar-selector / rank-broadcast forms of already-registered transfers),
plus one allowed-by-census structural addition from the maddening HeatNode
trace (``scatter`` in its static-index ``x.at[k].set(v)`` form only), plus
two allowed-by-census structural additions from the MIME fvm laplacian
trace (``gather`` in its static-index leading-axis row form only, and
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
pure element routing), plus the two rows of the index-bounds round
(``dynamic_slice`` and ``dynamic_update_slice`` — what ``u[i]`` with a
traced ``i`` and an out-of-range static ``u[30]`` actually lower to,
measured on both tested series; the same round widens ``gather``'s row
form from a point index to a range). Everything else falls to ⊤ —
soundly, with coverage recording exactly how much fell.

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

**AN ASSUME THE WALK NEVER REACHES IS STILL RECORDED** (audit 0.2.0
S13). The walk enters the transparent wrappers and ``cond``; it does not
enter a ``scan`` or ``while_loop`` body, so a ``stelling_assume`` written
in one is never classified — it narrows nothing and is not forwarded.
Being unclassified is a precision limit; leaving NO RECORD of it was a
soundness defect, because every rule that withholds a violation keys on
the ledger or on the flag the ledger's writers set, and an empty ledger
read as "no assume" released a witness the user's precondition excludes.
:func:`_record_undescended_assumes` reconciles the ledger against the
STATIC assume set (:func:`_assume_equations`) before anything reads the
run's assume state, so the ledger is total over the assumes the query
CONTAINS rather than over the ones the walk saw. The loop body is still
not descended: a loop body's assume is a per-iteration statement about a
carry this domain does not model, and reading one is a feature rather
than a repair.

Definite-verdict licensing splits by BOX EXACTNESS (audit F7): a box is
exact iff it equals the variable's true value set — ``stelling_any``
outputs and exact-point consts only; every transfer output is an
over-approximation (rounding pads, correlation-blind arithmetic). An
assume constraining an exact-box variable certifies its region's
satisfiability by the nonempty meet, and definite verdicts stand. An
assume constraining a NON-exact variable is still applied (the meet
over-approximates true-region ∩ reachable — sound) and the emptiness
refusals still fire (they prove emptiness from the over-approximation),
but the precondition's satisfiability is UNCERTIFIED: every definite
violation of the run is withheld from REFUTED (status ``unknown``, note
quoted — a possibly-vacuous refutation is not a refutation), while
VERIFIED remains allowed carrying a may-be-vacuous note and stamped
line; the inert-mode control is the visibility instrument. **Unless a
POINT WITNESS is found** — one member of the declared set at which every
``stelling_assume`` of the query is definitely true, which settles the
one thing the withholding was waiting on (the assumed region is
inhabited, so no obligation is vacuous) and lets the violations stand:
:func:`_region_witness`, one-sided, and reaching the withholding through
:func:`stelling.exactness.certifies_set_refutation` like everything else.
An assume whose region is provably
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
program's **IEEE round-to-nearest float execution in the format the
jaxpr actually carries** — float16, bfloat16, float32 or binary64, each
judged in its OWN format rather than all four as binary64. The ieee
domain is the same :class:`stelling.interval.IntervalArray` plus a
per-array ``maybe_nan`` flag (a parallel ``var id -> bool`` table beside
the interval env): endpoint arithmetic for the monotone core is native
binary64 with no outward rounding IN BINARY64 (the float value itself is
computable — :data:`stelling.interval.IEEE_ENDPOINT_ASSUMPTION`); in the
three narrow formats the endpoints are rounded OUTWARD onto that
format's grid and hazed with that format's own subnormal band, because
binary64 endpoint arithmetic does not land on a narrow format's values.
Overflow saturates to the VALUE ±inf, NaN-producing corner classes
(``inf−inf``, ``0·±inf``, ``0/0``, ``inf/inf``) set ``maybe_nan``, ⊤ is
maybe-NaN, and a predicate over a maybe-NaN operand is never definitely
true (NaN falsifies every comparison except ``ne``, which it satisfies).
``exp`` and ``pow`` are the second exception to the no-outward-rounding
line and a larger one: they bracket the TRUE REAL value and DECLINE
outright unless the caller declares a :class:`LibmBudget` for the
backend that will execute them (``libm_budget=``, accepted by both
:func:`propagate` and :func:`stelling.preconditions.check`), because the
bracket of the analysis host's ``math.exp`` is not a bracket of the one
the compiler emits. Every registered transfer is censused for ieee in
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

import dataclasses
import math
import re
import struct
from dataclasses import dataclass

from stelling import exactness
from stelling import interval as iv
from stelling import ir
from stelling.coverage import (
    DEFAULT_TRANSPARENT,
    Coverage,
    CoverageCounter,
    call_body,
    declaration_name,
    sub_jaxprs,
)

__all__ = [
    "IEEE_TRANSFERS",
    "LIBM_MEASURED",
    "LIBM_PROFILES",
    "LibmBudget",
    "ObligationReport",
    "Propagation",
    "RelationalAssume",
    "TIGHTENED_DOMAIN_REAL_REFUSAL",
    "TRANSFERS",
    "UnsatisfiableAssumptionError",
    "interval_env",
    "propagate",
    "query_identity",
    "resolve_libm_budget",
    "unpaired_propagation",
]

TIER_EXACT = "exact"
TIER_SOUND = "sound"
TIER_SOUND_LIBM = "sound-libm"


class UnsatisfiableAssumptionError(ValueError):
    """An assume's precondition is definitely false on the whole
    over-approximated domain: the declared set contains no point satisfying
    the precondition, so every downstream obligation would be vacuously
    "verified".

    This is the empty-declared-set refusal class — a harness defect, like
    the unbound-var :class:`stelling.ir.TranscriptionError` — so
    degrade-don't-crash does not apply: raised loudly, never a VERIFIED,
    never silent. :func:`stelling.preconditions.check` names it among the
    two classes it deliberately does NOT convert to a status.

    **TWO DETECTORS, ONE CLASS, AND THE SECOND IS WHY THIS DOCSTRING
    CHANGED.** The original is the interval one, in this module: the meet of
    a constrained variable's propagated interval with the assumed half-space
    is empty (also the strict-boundary collapse and the definitely-false
    constant comparison). It works on BOXES, so it cannot see a RELATIONAL
    assume — `x < y` is not a half-space on either box — and 0.2.0 began
    forwarding exactly those to the solver as positive axioms, where an
    unsatisfiable axiom set discharged every obligation instead (audit 0.2.0
    S7). The second detector is :func:`stelling.solvers._dispatch_obligation`,
    which asks the backend that answered ``unsat`` whether the declared boxes
    and the forwarded axioms ALONE are also ``unsat``.

    Same class deliberately, and not a new one: it is the same defect, the
    same sentence is right for it ("harness defect; nothing was verified"),
    and a caller who already handles the non-relational form must not have
    to learn a second name for the relational one. The scope is the same
    too — an assume is a precondition on the WHOLE QUERY, and a slice's
    axioms are a subset of the query's assumes, so a slice whose region is
    empty proves the query's is.
    """

# 2**53: the largest magnitude below which every int is exactly a double.
_EXACT_INT = 9007199254740992


# -- scope-qualified variable identity ----------------------------------------
#
# THE DEFECT THIS TYPE EXISTS FOR (audit 0.2.0 S5-B2, a FALSE VERIFIED).
#
# An IR var id is unique only WITHIN ONE SCOPE. `_jax_compat._Transcriber`
# numbers with one global counter keyed by `id(jax Var)`, and jax REUSES jaxpr
# objects, so the same id is legitimately bound in several scopes; and
# `obligation._Slicer` renumbers every inlined scope from
# `max(top-level ids) + 1`, a maximum that does not see sub-jaxpr ids at all.
#
# A relational `assume` traced inside a `jit`/`custom_jvp` body is recorded by
# its producing comparison equation, whose operands carry THAT BODY'S ids. The
# emission used to resolve them with a bare `names.get(atom.id)` against the
# slice's renumbered table — an integer lookup with no scope check. When the
# two ranges met, a foreign id resolved to an unrelated term and the axiom was
# emitted about the wrong values: measured as the CONVERSE of the user's own
# precondition, discharging an obligation that is false at every admitted
# point.
#
# A raw `int` cannot carry the missing information, which is why the fix is a
# type and not a bounds check. Raising the fresh-id base above every scope's
# ids would remove the collision and leave the defect: the lookup would simply
# MISS, and the run would lose a user-stated constraint with nothing said
# about it.
#
# A scope path addresses one scope INSTANCE in the IR tree by position, not by
# object identity and not by traversal convention: each step names the index of
# the equation whose body is being entered, tagged with the kind of descent, so
# `("call", 4)` (the body of top-level equation 4) can never denote the same
# scope as `("cond", 4, 1)` (branch 1 of the cond at equation 4). The
# propagator records the path it was in when it forwarded; the slicer records
# the same path when it renumbers; and a key that does not match resolves to
# NOTHING rather than to something else. Divergence therefore costs a
# disclosed skip, never a forged axiom.
ScopeStep = tuple  # ("call", eqn_index) | ("cond", eqn_index, branch_index)
ScopePath = tuple  # tuple[ScopeStep, ...]; () is the top-level scope


def _declaration_names(jaxpr: ir.Jaxpr) -> dict[tuple, str] | None:
    """``(scope path, var id) -> "x{k}"`` for every declaration this query's
    EMISSION will name that way, or ``None`` when the numbering cannot be
    derived without risk of naming the wrong one.

    THE POINT IS THAT IT IS THE SAME NUMBERING, not a second one (audit
    0.2.0 B8a, item 5 / M3). :meth:`stelling.obligation._Slicer._flatten`
    assigns ``any_order[outvar.id] = len(any_order)`` while walking the
    query's equations in order and inlining each SOUNDLY INLINABLE
    transparent wrapper IN PLACE; :attr:`stelling.obligation.SliceInput.name`
    is :func:`stelling.coverage.declaration_name` of that index. This
    function performs the identical walk over the identical wrapper set and
    counts the identical equations, so the INDEX it hands a message is the
    index the witness will carry — and it is derived from the IR alone, so
    no message can be aimed by anything a caller supplies.

    IT IS A PREFIX, NOT ALWAYS THE WHOLE NAME (audit 0.2.0 B8a FIXUP). For
    a scalar declaration the witness constant is `x{k}` and the two agree
    exactly. For an ARRAY declaration the emission mints one constant per
    flat element, `x{k}_{i}`, and there is no single value for a message to
    name — the message names the declaration, `x{k}`, which is the common
    prefix of every element's name and is what a reader needs to find them.

    Where the two walks COULD diverge, this returns ``None`` and every
    message falls back to the internal id, spelled so that it cannot be read
    as a declaration index:

    * the propagator descends a ``cond``; the slicer never does. A
      declaration inside a branch has no ``x{k}`` at all, so meeting one
      abandons the numbering rather than guessing which of the survivors is
      #0.
    * the propagator's own descent test is WEAKER than the slicer's (it
      checks invar arity; the slicer also checks outvar arity and that the
      body's constvars pair with its consts). A wrapper only one of them
      inlines shifts every later index, so a wrapper that fails the
      SLICER's test abandons the numbering here too — this walk uses the
      strict test on purpose, and disagreeing with the propagator's descent
      is exactly what it is detecting.
    * a declaration's ``(scope, id)`` key bound TWICE in that scope — by a
      second declaration or by any ordinary equation, invar or constvar.
      Ids are unique per jaxpr under transcription and are whatever a
      document says under ``from_dict``; a key two bindings share does not
      identify a declaration, and :meth:`_Propagator._name_of_id` would
      hand the name to whichever value was written there last, so nothing
      is named.

    Keyed by SCOPE PATH AND ID, never by id alone, for the reason
    :data:`ScopePath` gives: an inner scope's ids are not the outer scope's,
    and a lookup that ignores the scope resolves to a different variable
    rather than to nothing.
    """
    names: dict[tuple, str] = {}
    bound: set[tuple] = set()
    count = 0

    def collides(key: tuple, *, declaring: bool) -> bool:
        """Whether binding ``key`` here would make a published name a guess.

        A scope binds each id exactly once under transcription, so this is
        a ``from_dict``/hand-built-IR question only. Where a document binds
        one id twice, :meth:`_Propagator._name_of_id` — which resolves
        ``(scope, id)`` and nothing else — hands the declaration's name to
        whichever value the walk wrote there LAST. So a declaration binding
        an id this scope already binds is a collision, and so is any later
        binding of an id a declaration already holds. Two non-declaration
        bindings that meet name nothing between them and are left alone."""
        return key in bound and (declaring or key in names)

    def walk(j: ir.Jaxpr, path: tuple) -> bool:
        nonlocal count
        # ids this scope binds before its first equation: an invar or a
        # constvar sharing a declaration's id is the same ambiguity as a
        # rebinding equation, and reads the same way in a message.
        for v in (*j.constvars, *j.invars):
            if isinstance(v, ir.Var):
                bound.add((path, v.id))
        for pos, e in enumerate(j.eqns):
            transparent = e.primitive in DEFAULT_TRANSPARENT
            inner = call_body(e) if transparent else None
            if transparent:
                if (
                    inner is None
                    or len(inner.jaxpr.invars) != len(e.invars)
                    or len(inner.jaxpr.outvars) != len(e.outvars)
                    or len(inner.jaxpr.constvars) != len(inner.consts)
                ):
                    # the slicer leaves this wrapper opaque; the propagator
                    # may still descend it — the two counts can differ from
                    # here on, so neither is a name
                    return False
            elif _holds_a_declaration(e):
                # a declaration this walk cannot reach and the slicer
                # cannot name: a `cond` branch (the propagator descends,
                # the slicer does not), a `scan`/`while_loop` body, or any
                # other sub-jaxpr that is not an inlinable wrapper. Which
                # of the survivors is #0 is then a guess.
                return False
            declares = (
                not transparent
                and e.primitive == "stelling_any"
                and bool(e.outvars)
            )
            # EVERY outvar this equation binds in this scope is recorded,
            # declaration or not (audit 0.2.0 B8a fixup, item 1): keying the
            # collision test on declarations alone left a declaration's id
            # reused by an ordinary equation undetected, and the message
            # then named `-x0` as `x0`.
            for i, out in enumerate(e.outvars):
                naming = declares and i == 0
                if not isinstance(out, ir.Var):
                    if naming:
                        return False
                    continue
                key = (path, out.id)
                if collides(key, declaring=naming):
                    return False
                bound.add(key)
                if naming:
                    names[key] = declaration_name(count)
                    count += 1
            if transparent and not walk(
                inner.jaxpr, path + (("call", pos),)
            ):
                return False
        return True

    return names if walk(jaxpr, ()) else None


_UNSET = object()  # "not derived yet", distinct from "not derivable"


def _holds_a_declaration(eqn: ir.JaxprEqn) -> bool:
    """Whether any sub-jaxpr of ``eqn``, at any depth, declares an input.

    Read through :func:`stelling.coverage.sub_jaxprs`, which is the walk
    that enters a param holding a sub-jaxpr in a ``list`` as well as a
    ``tuple`` — audit B9 measured what a narrower one misses."""
    stack = list(sub_jaxprs(eqn))
    while stack:
        j = stack.pop()
        for e in j.eqns:
            if e.primitive == "stelling_any":
                return True
            stack.extend(sub_jaxprs(e))
    return False


@dataclass(frozen=True)
class RelationalAssume:
    """One relational ``assume`` forwarded for solver emission, WITH THE
    SCOPE ITS OPERAND IDS BELONG TO.

    ``eqn`` is the comparison equation the assume consumed (primitive in
    ``_ASSUME_CMPS``), exactly as the propagator met it — so its operand ids
    are ids of ``scope``, and are meaningless anywhere else. Nothing may
    resolve them without first mapping ``scope`` to the namespace it is being
    resolved into; :meth:`stelling.obligation._Slicer._carry_assumes` is the
    one place that mapping happens.

    ``where`` is the assume's source location, quoted into the disclosure when
    the assume cannot be carried into a given obligation's slice.
    """

    scope: ScopePath
    eqn: ir.JaxprEqn
    where: str = ""


# -- the assume disposition ledger --------------------------------------------
#
# THE FOUR DISPOSITIONS. Every assumed conjunct the propagator classifies
# leaves exactly one :class:`AssumeDisposition` on
# :attr:`Propagation.assume_ledger`, recorded AT THE POINT OF CLASSIFICATION,
# and the four names below are the whole space a classification may land in.
# The question a reader downstream is asking is: is the region this run
# JUDGED contained in the region the user's `assume`s describe?
#
#   ``applied``   the interval domain narrowed (or confirmed already within)
#                 the target variable with this conjunct, so every point the
#                 run judges satisfies it by construction.
#   ``no-op``     the conjunct was not applied, but its own value is
#                 definitely TRUE over the boxes in force
#                 (:meth:`_Propagator._conjunct_certainly_true`), so it excluded
#                 nothing and its absence widened nothing: the judged set IS
#                 the assumed region for this conjunct.
#   ``forwarded`` relational (both sides vary), so the interval domain cannot
#                 represent it and it was handed to the solver layer as
#                 element ``forwarded_index`` of
#                 :attr:`Propagation.relational_assumes`. Accounted for on a
#                 given obligation ONLY IF that obligation's script emitted
#                 it — see :func:`unaccounted_assumes`.
#   ``dropped``   anything else: no narrowing, no certainty, nothing given to
#                 the solver. The run judged a SUPERSET of the assumed region.
#                 It covers TWO situations a reader must be able to tell
#                 apart, and the ``reason`` is where they are told apart: a
#                 classifier looked at the conjunct and gave up, or NOBODY
#                 LOOKED — the assume sits inside a sub-jaxpr the walk does
#                 not enter (:func:`_record_undescended_assumes`, audit 0.2.0
#                 S13). The kind is the same because what the verdict rests on
#                 is the same; the sentence is not.
#
# THE LEDGER IS TOTAL OVER THE ASSUMES THE QUERY CONTAINS, not over the ones
# the walk reached, and the difference is a released soundness finding. Until
# audit 0.2.0 S13 the totality claim was scoped to the traversal, and it was
# literally true: an assume inside a `scan` body left no entry because the
# walk never saw it, so the ledger was empty AND consistent, `assume_dropped`
# stayed False, and a definite violation was released with a witness the
# user's own precondition excludes. :func:`propagate` now reconciles the
# ledger against the STATIC set (:func:`_assume_equations`) before anything
# reads it, so :func:`ledger_covers` is a POSTCONDITION of a propagation
# rather than a question about the walk's reach.
#
# WHY A LEDGER AND NOT A COUNT. The rule that releases a definite violation
# from withholding used to compare two integers — how many relational assumes
# the propagation forwarded against how many a script emitted — and that shape
# failed twice. Once because the denominator counted only the RELATIONAL
# subset while the flag it was gated on (``assume_dropped``) is set by ANY
# drop reason at all (audit 0.2.0 S6), so one satisfied relational assume
# released a violation whose witness broke a differently-dropped one. Once
# because a change on the propagator's side — no longer forwarding a
# branch-scoped assume — silently moved the denominator, and `1 == 1` released
# a violation whose branch-scoped precondition the solver had never been told.
#
# Both are one failure: an arithmetic comparison between two populations that
# nothing forces to be the same population. A ledger cannot fail that way,
# because the release test counts nothing — it asks whether EVERY entry names
# an accounted-for disposition. A drop reason added later has to name a
# disposition to be recorded at all, and a name this module does not know is
# UNACCOUNTED (:func:`unaccounted_assumes` whitelists, never blacklists), so a
# new reason fails closed instead of shrinking a denominator nobody
# re-derived.
ASSUME_APPLIED = "applied"
ASSUME_NOOP = "no-op"
ASSUME_FORWARDED = "forwarded"
ASSUME_DROPPED = "dropped"


@dataclass(frozen=True)
class AssumeDisposition:
    """What became of ONE assumed conjunct, recorded where it was decided.

    ``kind`` is one of :data:`ASSUME_APPLIED`, :data:`ASSUME_NOOP`,
    :data:`ASSUME_FORWARDED`, :data:`ASSUME_DROPPED`. ``reason`` is the
    sentence already written into the run's notes for this conjunct, so a
    withholding can NAME the conjunct that caused it rather than restate the
    rule. ``where`` is the assume's source location.

    ``forwarded_index`` is meaningful only for :data:`ASSUME_FORWARDED`: it is
    this conjunct's index in :attr:`Propagation.relational_assumes`, and it is
    the identity a downstream join is keyed on. It is ``-1`` on every other
    kind, and ``-1`` matches no emitted origin.

    ``eqn_id`` is ``id()`` of the ``stelling_assume`` EQUATION this conjunct
    came out of — the same identity :func:`_assume_equation_ids` collects
    statically and :attr:`_Propagator.assume_witness` is keyed on. It is what
    :func:`ledger_covers` joins on, and it is why an empty
    :func:`unaccounted_assumes` result can be told apart from an assume the
    walk never reached: a filter over the ledger cannot see an assume that
    left no entry, and the static set can. ``-1`` on a hand-built entry that
    names no equation, which matches no static id.

    **``eqn_id`` IS OUT OF THE EQUALITY, and that is not a convenience.** It
    is a process-local join key — ``id()`` of an object in this run's memory —
    and not part of what became of the conjunct: two dispositions with the
    same kind, reason, location and forwarded index record the same
    classification whichever trace of the same harness produced them. Leaving
    it in ``__eq__`` would make every :class:`Propagation` unequal to a
    propagation of a SECOND trace of the same harness, which is precisely the
    comparison the certificate's one-sidedness is pinned by
    (``test_a_failed_certificate_search_changes_nothing_at_all`` compares two
    whole runs byte-for-byte). A field that cannot be compared across runs
    must not be what a cross-run comparison fails on.
    """

    kind: str
    reason: str = ""
    where: str = ""
    forwarded_index: int = -1
    eqn_id: int = dataclasses.field(default=-1, compare=False)


def unaccounted_assumes(
    ledger: tuple["AssumeDisposition", ...],
    emitted_origins: tuple[int, ...],
) -> tuple["AssumeDisposition", ...]:
    """The ledger entries this obligation's solver query did NOT account for.

    ``emitted_origins`` are the ``relational_assumes`` INDICES the
    obligation's emitted script states as axioms
    (:attr:`stelling.smt.Script.emitted_origins`) — not a count of them, and
    not re-derived here from anything.

    **THE TWO ARGUMENTS MUST INDEX THE SAME TUPLE**, and nothing in this
    signature enforces it. Both are indices into ONE propagation's
    ``relational_assumes``: the ledger's ``forwarded_index`` is written by
    the propagator, and the origins reach the script through
    ``obligation.SliceAssume.origin``, which
    :meth:`stelling.obligation._Slicer._carry_assumes` fills from the tuple
    it was HANDED. Inside :func:`stelling.solvers.escalate` that is the same
    tuple by construction — ``slice_unknown_obligations`` passes
    ``propagation.relational_assumes`` and nothing else can. A caller
    driving :func:`stelling.obligation.slice_obligation`'s public
    ``relational_assumes=`` keyword with some OTHER tuple and then joining
    against this propagation's ledger would be comparing indices into two
    different lists; do not. (`main` had the same exposure through
    ``smt.emit``'s removed ``relational_assumes`` parameter.)

    An entry is accounted for when the region the solver ran over is inside
    the region that entry describes:

      * ``applied`` and ``no-op`` — the judged set is already inside the
        conjunct's region, so the solver needs to be told nothing;
      * ``forwarded`` — only when THIS obligation's script emitted THAT
        assume, so the model satisfies it by construction.

    Everything else — including a ``kind`` this function does not recognise —
    is unaccounted, and an empty result is the only thing that may release a
    definite violation from withholding. THE MEMBERSHIP TEST IS A WHITELIST
    AND MUST STAY ONE: a disposition added later that this function has not
    been taught is a disposition nobody has shown the judged set to be inside
    of, and the failure direction of guessing is a witness outside the
    precondition presented as a refutation.
    """
    emitted = frozenset(emitted_origins)
    return tuple(
        e for e in ledger
        if not (
            e.kind in (ASSUME_APPLIED, ASSUME_NOOP)
            or (e.kind == ASSUME_FORWARDED and e.forwarded_index in emitted)
        )
    )


def ledger_covers(
    ledger: tuple["AssumeDisposition", ...],
    jaxpr,
) -> bool:
    """Whether the ledger has a record for EVERY ``stelling_assume`` equation
    ``jaxpr`` contains — sub-jaxprs included.

    **THE QUESTION :func:`unaccounted_assumes` CANNOT ANSWER, AND MUST NOT BE
    READ AS ANSWERING.** That function is a FILTER over the ledger, so its
    empty result has two causes that a caller reading a single value cannot
    tell apart: every recorded assume is accounted for, or *nothing is
    recorded* for an assume that exists. The second was reachable — the
    propagator does not descend ``scan`` or ``while_loop`` bodies, so a
    ``stelling_assume`` inside one was never classified and left no ledger
    entry at all. Measured, before the repair: ``assume(x < y)`` at top level
    plus ``assume(y < x)`` inside a ``lax.scan`` body gave a one-entry ledger,
    an empty ``unaccounted_assumes``, and a CLEAN VERIFIED over a precondition
    that admits no point of any strict order.

    So the positive claim "the region the solver ran over is inside the region
    EVERY assume of the query describes" is the CONJUNCTION of the two: the
    filter says the recorded ones are accounted for, and this says the record
    is complete. Neither alone is that claim.

    **WHAT THIS NOW ANSWERS, AND WHAT IT DOES NOT.**
    :func:`_record_undescended_assumes` closed the hole at its root: a
    propagation's ledger is reconciled against the static set before anything
    reads it, so on a :class:`Propagation` produced by :func:`propagate` this
    returns ``True`` BY CONSTRUCTION and the un-recorded assume reaches every
    rule as an unaccounted-for ``dropped`` entry instead — which is what the
    filter can see and name. This is therefore no longer the discriminator
    for that case; it is the check that the postcondition holds, and it still
    answers ``False`` for the two ways a caller can break the join: a ledger
    from a DIFFERENT (or re-decoded) jaxpr than the one passed here, and a
    hand-built or emptied ledger. Both are the conservative direction.

    The requirement is the STATIC set (:func:`_assume_equation_ids`) — every
    assume equation the query contains, whether or not any walk reached it —
    which is the same total machinery the non-emptiness certificate's
    requirement already rests on, and the reason that path never had this
    hole. Over-collecting is the safe direction: an equation nothing evaluates
    simply fails the subset test, and the caller falls back to
    :data:`stelling.solvers.REGION_UNCERTIFIED` and a disclosed caveat.

    The join is on ``AssumeDisposition.eqn_id``, so ``ledger`` must be the
    ledger of a propagation of THIS ``jaxpr`` object — the same discipline
    :func:`unaccounted_assumes` states for its two arguments, and the same
    object identity :func:`stelling.propagate._region_witness` already joins
    its static requirement to its witness map on. A ledger from a different
    (or re-decoded) jaxpr matches nothing and answers False, which is the
    conservative direction.
    """
    return _assume_equation_ids(jaxpr) <= frozenset(
        e.eqn_id for e in ledger if e.eqn_id != -1
    )


def _drop(reason: str) -> AssumeDisposition:
    """The ordinary disposition: this conjunct was not applied, not certainly
    true, and not given to the solver.

    Every classification site that gives up on a conjunct goes through here,
    so `dropped` is the default and a site has to say something OTHER than
    this to be treated as accounted for. That is the direction the whitelist
    in :func:`unaccounted_assumes` is built to fail in: a reason added later
    that nobody thought about is a drop."""
    return AssumeDisposition(kind=ASSUME_DROPPED, reason=reason)


# -- the assume the walk never reached (audit 0.2.0 S13) ----------------------
#
# `dropped` IS THE RIGHT KIND AND IT IS NOT THE WHOLE TRUTH. The kind says
# what the reader's verdict rests on — no narrowing, no certainty, nothing
# given to the solver, so the run judged a SUPERSET of the assumed region —
# and that is exactly this case, which is why the whitelist in
# :func:`unaccounted_assumes` and every rule keyed on it are already correct
# for it without being taught anything. But a reader who is told only
# "dropped" will look for the classifier that gave up, and there was none:
# the propagator's walk does not enter a `scan` or `while_loop` body, so an
# assume written inside one was never looked at. The reason has to say that,
# and NAME THE CONSTRUCT, because the reader's next question is which of
# their assumes this was and what to do about it (lift it out of the loop
# body; the tool does not descend one).
#
# THE CHAIN, NOT THE INNERMOST NAME. `_record_undescended_assumes` renders
# every enclosing primitive from the top down, so `jit` inside `scan` reads
# `'scan' -> 'jit'`: the innermost name alone matches the source line but
# misattributes the cause (it is the `scan` that is not entered), and the
# outermost name alone disagrees with the source line. Both are in the
# sentence, in order, and the reader can see which is which.
UNDESCENDED_ASSUME_REASON = (
    "NEVER CLASSIFIED: this assume sits inside {inside}, which the "
    "propagation's walk does not enter, so no classifier ever saw it — it "
    "narrowed nothing, was not forwarded to the solver, and had NO EFFECT on "
    "the analysis; the judged set is a SUPERSET of the assumed region"
)
UNDESCENDED_ASSUME_NOTE = (
    "assume NEVER CLASSIFIED at {where}: it sits inside {inside}, and the "
    "propagation does not descend that construct — a loop body's assume is a "
    "per-iteration statement about a carry the analysis does not track, so it "
    "is recorded as DROPPED rather than silently ignored. It narrowed "
    "nothing and was not given to the solver: VERIFIED still proves a "
    "superset, and every definite violation is WITHHELD from REFUTED because "
    "a witness of the superset may violate this assume. Write the "
    "precondition at the top level of the harness to have it honoured"
)


@dataclass(frozen=True)
class ObligationReport:
    """One ``stelling_assert`` equation, judged over the declared box."""

    index: int  # traversal order among obligations
    status: str  # "discharged" | "unknown" | "violated-over-set"
    detail: str
    source_info: tuple[str, ...]
    # The Var IDs of the stelling_assert equation's invars (the predicate
    # operand), IN THE SCOPE THE WALK SAW THEM IN.  Populated by the
    # propagator for top-level and descended-into obligations; empty for
    # unexamined obligations (inside opaque sub-jaxprs).
    #
    # A RECORD KEPT FOR READERS, CONSUMED BY NOTHING — audit 0.2.0 B8a,
    # item 4, and stated plainly here at the FIXUP's request because a
    # field that is only written is easy to mistake for one that is used.
    # It is set once, in this module's `stelling_assert` handler, and
    # copied once, in `solvers.make_solver_verdict`'s re-statusing loop.
    # Nothing in the library reads it: no transfer, judgment, status,
    # counter, hash or message. It survives the removal of its one former
    # consumer because it is the only place an obligation records WHICH
    # values its predicate was about, which is what a human tracing a
    # verdict back into the IR needs.
    #
    # That former consumer was `verdict._apply_reachability_conjunct`,
    # which read these ids against a TOP-LEVEL live set; that conjunct is
    # removed and the block comment where it stood says why. The ids are
    # scope-local, so a reader comparing them against another scope's is
    # comparing two different variables — which is exactly how the removed
    # conjunct silenced a genuine REFUTED. Any future consumer must join on
    # `top_level_eqn_pos` (or a scope path) first.
    operand_var_ids: tuple[int, ...] = ()
    # WHERE THIS OBLIGATION'S ASSERT LIVES, recorded by the walk that saw it:
    # the position of the `stelling_assert` equation in the TOP-LEVEL
    # `jaxpr.eqns`, or None when the obligation was recorded from inside a
    # sub-jaxpr (a transparent call body, a `cond` branch, or an undescended
    # `scan`/`while_loop` body via `_record_unexamined`).
    #
    # THE ASSOCIATION IS CARRIED, NOT INFERRED — audit 0.2.0 M17. Solver
    # escalation slices a top-level `stelling_assert`, so it needs to know
    # which one an obligation came from. It used to infer that by COUNTING:
    # if the number of top-level asserts equalled the number of obligations,
    # index k meant assert k; otherwise nothing could be mapped and EVERY
    # unknown obligation declined escalation — so one `assert_` written
    # inside a `jax.jit` helper silently cost solver escalation for every
    # other obligation in the query.
    #
    # The count check was SOUND (this walk records exactly one obligation per
    # top-level assert — the malformed-shape screen above EXEMPTS asserts so
    # that it still does — so equal totals really do mean index k is assert
    # k). It was simply the wrong SHAPE of instrument: a per-obligation
    # question answered with a whole-query number. The walk knows the answer
    # exactly, per obligation, at the moment it records one; this field is
    # that answer, and `stelling.obligation.slice_unknown_obligations`
    # VERIFIES it against the IR rather than trusting it.
    top_level_eqn_pos: int | None = None


@dataclass(frozen=True)
class Propagation:
    obligations: tuple[ObligationReport, ...]
    nonvacuity_checks: tuple[ObligationReport, ...]  # y0-membership conditions
    coverage: Coverage
    transfers_used: tuple[tuple[str, str], ...]  # (primitive, tier), sorted
    assumptions: tuple[str, ...]
    notes: tuple[str, ...]
    # WHICH QUERY THIS PROPAGATION IS ABOUT — audit 0.2.0 B6 re-audit
    # UNSOUND-3, closed in B11. :meth:`stelling.ir.ClosedJaxpr.content_hash`
    # of the ``closed`` :func:`propagate` walked, or ``""`` when that hash
    # could not be taken (see :func:`query_identity`).
    #
    # WHY IT IS HERE AND NOT AT THE CONSUMER. Until this field existed the
    # `Propagation` carried no query identity at all, and every function
    # that consumes one against a query therefore had nothing to check. The
    # measured consequence was a live false VERIFIED on `main` and on the
    # released 0.1.0: two queries traced from ONE factory carry identical
    # `source_info` at identical top-level positions, so the per-obligation
    # association check in
    # :func:`stelling.obligation.slice_unknown_obligations` passes, and an
    # assembly of (query B, propagation of A) reports B's obligations with
    # A's statuses. `stelling.solvers.MispairedEscalationError` bound the
    # ESCALATION leg and only that leg — `escalate` hashes the `closed` IT
    # was handed, so the (query, escalation) pair genuinely matches while
    # the propagation is a stranger, and with `carries_work=False` the
    # escalation gate is not consulted at all.
    #
    # NO DEFAULT, deliberately, and the reason is the sibling field's own
    # history: `stelling.solvers.Escalation.query_sha256` defaults to `""`,
    # which makes a return site that forgot it a SILENT hole and costs a
    # whole test (`test_every_escalate_return_site_records_the_query`) to
    # pin. A `Propagation` has exactly ONE construction site in this
    # library, so a required field makes that class of hole unconstructible
    # rather than merely tested for. A caller who hand-builds one must say
    # which query it is about; `""` is still a legal value and still means
    # "not recorded", and every consumer REFUSES it rather than pairing it
    # with anything (two absences are not a match).
    query_sha256: str
    # which arithmetic the obligations were judged about ("real" | "ieee");
    # the verdict assemblers stamp from this field, never from a guess
    semantics: str = "real"
    # WHICH POSITION THE BOUNDARY DIAL WAS IN (:data:`_BOUNDARY_MODES`),
    # recorded on every propagation the way `semantics` is, from the run
    # that happened and never from a guess. The DEFAULT here repeats the
    # module default deliberately: a `Propagation` a caller hand-builds,
    # and every one recorded before this field existed, IS a
    # boundary-opaque propagation, so the default is the true value
    # rather than a placeholder.
    #
    # WHAT READS IT. :func:`propagate` itself, to decide whether to write
    # the stamped position line (:data:`BOUNDARY_TRANSPARENT_POSITION`)
    # into `assumptions`; no transfer, judgment or counter does.
    boundary: str = "opaque"
    # How many strict-sign certificates actually CROSSED a sub-jaxpr
    # boundary on this run — see `_Propagator.boundary_crossings` for what
    # the count is and, more importantly, what it is not. Always 0 under
    # the `"opaque"` default.
    boundary_crossings: int = 0
    # an assume was DROPPED in constrain mode: its predicate had no decidable
    # box, so the query ran over a SUPERSET of the intended set. Carried out
    # to the solver layer because the escalation refusal keys on a CONSTRAINED
    # assume being present, and a dropped one is not present at all — leaving
    # the solver free to emit a sat witness outside the precondition.
    assume_dropped: bool = False
    # a constraining assume NARROWED a variable whose box is an
    # over-approximation, so the narrowed region was never certified
    # inhabited (audit F7). The OTHER half of this run's assume state, and
    # carried out for the same reason `assume_dropped` is: the refinement
    # layer decides the same question on its own leg and must consult the
    # whole state through the same shared point
    # (:func:`stelling.exactness.certifies_set_refutation`), not a private
    # subset of it. Kept a SEPARATE field rather than merged into one
    # "uncertified" bit because the two name different mechanisms, and the
    # sentence explaining a withholding has to name the one that fired.
    narrowing_uncertified: bool = False
    # a POINT WITNESS was found: one member of the declared set at which
    # every ``stelling_assume`` of the query is definitely true, so the
    # assumed region is INHABITED and the vacuity the two flags above
    # guard against cannot arise on this run. The positive channel to the
    # same fact those two flags approach negatively, carried out for
    # exactly the reason they are: the refinement layer decides the same
    # question on its own leg and must consult the whole assume state
    # through the same shared point
    # (:func:`stelling.exactness.certifies_set_refutation`), not a private
    # subset of it. False means NO WITNESS WAS FOUND — never "the region
    # is empty"; see :func:`_region_witness`.
    region_inhabited: bool = False
    # Values this walk bound to ⊤ — every element [-inf, inf], the widest
    # box there is — in the top-level scope, as (producing primitive,
    # count), most frequent first. THE FACT THE COVERAGE CENSUS CANNOT
    # HOLD, and the reason it is recorded here instead of there:
    # :func:`stelling.coverage.measure` is a static census over the IR
    # and a set of primitive NAMES, so it can only answer "is a transfer
    # registered for this primitive". A registered transfer that runs and
    # returns ⊤ on the values it was handed is `known` to the census and
    # invisible in its counts. Measured: `exp(x) - exp(x)` over
    # x ∈ [-1000, 1000] counts 4/4 known, 0 fallen to ⊤, and propagates
    # [-inf, inf]. The verdict assemblers pair this with the census's own
    # zero-gap counts to state what the coverage line did NOT establish.
    #
    # WHAT THIS DOES NOT SEE, by name rather than by rule. An earlier
    # version of this comment said "top-level scope only, and that is the
    # whole claim". The scope half is true; "the whole claim" was an
    # EXHAUSTIVENESS claim and it was false in four ways, three of them
    # nothing to do with scope. A list of gaps ranges over a closed set;
    # a sentence claiming there are no others ranges over everything the
    # author has not looked at.
    #
    #   * A ⊤ inside a sub-jaxpr (jit/custom_jvp wrapper, cond branch)
    #     that never reaches the wrapping equation's outvars. Those envs
    #     are isolated and discarded on exit. This one is deliberate.
    #   * A box that is ⊤ on SOME elements. `_is_top` requires every
    #     element, so a partly-unbounded array reads as bounded here.
    #   * A ⊤ that a later constraining `assume` narrows in place. This
    #     reads the finished env, so what a transfer RETURNED is not what
    #     is recorded if something overwrote it afterwards.
    #   * Anything at all when the census reports a gap of its own: the
    #     assemblers suppress this field then, so a query carrying both an
    #     unregistered primitive AND registered-transfer ⊤s discloses only
    #     the former. `scan`, `while` and `iota` are unregistered, so a
    #     query using lax control flow never receives this disclosure.
    #
    # And one it reports that the rationale above would not predict: a
    # top-level ⊤ bound to a value nothing reads is counted, because this
    # walks the env rather than the live set.
    top_boxes: tuple[tuple[str, int], ...] = ()
    # Relational assumes the interval domain could not apply: comparison
    # equations whose BOTH operands are non-constant (non-point intervals),
    # so no variable can be narrowed independently. Recorded so the solver
    # escalation can emit them as additional axioms — the constraint the
    # user stated is sound, it merely cannot be represented in the
    # non-relational interval domain.
    #
    # Each entry is a :class:`RelationalAssume`: the comparison equation the
    # assume consumed AND the scope path its operand ids belong to. The scope
    # is not decoration — see that type's own note. A BRANCH-SCOPED assume is
    # never here: a precondition that holds only when a `cond` branch is taken
    # is not a fact about the query, and asserting it globally would constrain
    # the whole query by something the program does not guarantee (audit
    # 0.2.0 S5-B1). Those are dropped with the reason quoted, exactly as
    # `_unsatisfiable` already degrades inside a branch.
    relational_assumes: tuple[RelationalAssume, ...] = ()
    # EVERY assumed conjunct this run classified, with what became of it —
    # one :class:`AssumeDisposition` per classification, in classification
    # order, filled in by the propagator as it decides. See the block comment
    # above that type for the four dispositions and for why the rule that
    # releases a withheld violation reads this and not a pair of counts.
    #
    # It is a SUPERSET of `relational_assumes` in the sense that matters: an
    # entry of kind `forwarded` carries the index of its `relational_assumes`
    # element, and every OTHER assumed conjunct — dropped for any reason at
    # all, branch-scoped, inert-mode, unclassified — has an entry here and
    # has none there. That is exactly the gap audit 0.2.0 S6 fell through.
    #
    # Empty when the run classified no assume. It is NOT empty merely because
    # no assume constrained anything: a run that dropped every assume has an
    # entry per drop, which is the case the release test must refuse.
    assume_ledger: tuple[AssumeDisposition, ...] = ()

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


# -- query identity -----------------------------------------------------------
#
# ONE DERIVATION, READ BY EVERY SITE THAT PAIRS A PROPAGATION WITH A QUERY.
# `Propagation.query_sha256` is written here (at `propagate`'s single
# construction site) and compared here (at `unpaired_propagation`), so the two
# halves cannot drift apart the way a hash copied at one site and recomputed at
# another can. `stelling.solvers._query_sha256` — the escalation leg's
# derivation, which predates this one — delegates to `query_identity` for the
# same reason: the escalation gate and the propagation gate must agree about
# what "the same query" means, or a caller can satisfy one and not the other.


def query_identity(closed) -> str:
    """:meth:`stelling.ir.ClosedJaxpr.content_hash` of ``closed``, or ``""``
    if it cannot be taken.

    NEVER RAISES, and the empty string is not a pass — it is the value
    :func:`unpaired_propagation` REFUSES on either leg. An unhashable
    ``closed`` and a propagation that recorded nothing both produce ``""``,
    and an equality test would pass them: two absences are not a match.
    That exact hole was live in the escalation leg until `e35de13` and is
    not being rebuilt here.
    """
    try:
        return str(closed.content_hash())
    except Exception:  # noqa: BLE001 — an unhashable query is refused, not excused
        return ""


# The one refusal that composes no message at all, for the object on which
# even composing one raises. A module-level literal: returning it cannot
# fail, which is the whole property :func:`unpaired_propagation` needs.
UNREADABLE_PROPAGATION_IDENTITY = (
    "unpaired propagation: the supplied propagation could not be examined "
    "at all — reading its query identity, and then describing what was "
    "read, both raised. It cannot be shown to be about the query being "
    "judged, and an identity that cannot be read is not a match. Judge the "
    "query with the propagation `propagate()` produced from it."
)


def unpaired_propagation(propagation, query_sha256: str) -> str | None:
    """``None`` when ``propagation`` is PROVABLY about the query whose
    identity is ``query_sha256``; otherwise the sentence saying why it
    cannot be shown to be.

    ``query_sha256`` is the caller's own :func:`query_identity` of the
    ``closed`` it is judging — passed in rather than recomputed here
    because three of the five consumption sites already hold it (the two
    verdict assemblers stamp it, and :func:`stelling.solvers.escalate`
    records it on the :class:`stelling.solvers.Escalation`), and a second
    hash of the same query would be pure cost.

    THE PROPERTY IS "BOTH OF THEM SAID SO", NOT EQUALITY. A non-string, an
    empty string on either leg, or two different strings are all refusals,
    and each gets its own sentence: a reader who has just lost a VERIFIED
    needs to know whether the propagation named another query or named
    none.

    **AND "TWO DIFFERENT STRINGS" MEANS "TWO STRINGS THAT SAY THEY DIFFER"**
    — audit 0.2.0 B11 audit, fix 6. That last test is ``!=``, which
    dispatches to the recorded value's own ``__ne__``, so a ``str`` SUBCLASS
    whose comparison always answers equal satisfies the ``isinstance`` test
    and the emptiness test honestly and then decides the third itself: it
    PAIRS, with any query. Its two neighbours do not — an always-equal object
    of any OTHER type fails ``isinstance``, and an EMPTY always-equal
    subclass fails ``not recorded``, because a plain subclass inherits
    ``str.__len__`` — so the exposure is exactly non-empty ``str``
    subclasses, and it is driven AS A PAIRING in
    ``tests/test_propagation_identity.py::test_an_ALWAYS_EQUAL_str_SUBCLASS*``.
    It is disclosed rather than closed because it buys nothing: defining such
    a class is attacker Python in the caller's own process, and a caller who
    can do that can equally pass the query's true hash, which pairs by the
    honest rule.

    IT DOES NOT DECIDE WHAT HAPPENS NEXT. Each site fails closed in its own
    vocabulary — a verdict assembler returns UNKNOWN carrying this
    sentence, :func:`stelling.affine.refine_propagation` declines every
    unknown obligation with it, :func:`stelling.obligation.
    slice_unknown_obligations` returns a
    :class:`stelling.obligation.DeclinedObligation` per obligation quoting
    it — because three of the five sites MAY NOT RAISE (they are iterated
    in a ``for`` header by callers whose per-obligation nets sit inside it;
    see `slice_unknown_obligations`' docstring, audit 0.2.0 B6/M17′) and a
    fix that raised at two of them and degraded at three would be two
    behaviours for one fact.

    **AND THIS FUNCTION MAY NOT RAISE EITHER — the same lesson, one layer
    in** (audit 0.2.0 B6 re-audit R7: *"a net that re-raises while composing
    its own message is not a net"*). It is handed whatever the caller passed
    as a propagation, and both reads it makes of that object can raise on a
    hostile one: ``query_sha256`` may be a property that raises something
    other than ``AttributeError``, and the value it returns may have a
    ``__repr__`` that raises. So the read is netted, and the value is
    quoted only when its EXACT type is a builtin scalar — anything else is
    described by type name. A refusal whose message raises is not a
    refusal; it is the raw escape the gate exists to prevent, at the one
    site (:func:`stelling.obligation.slice_unknown_obligations`) where it
    would cost every obligation's verdict rather than one.

    The two netted reads below are the ones with a NAMED hostile shape. The
    outer net is for the ones there is no point enumerating — ``repr`` of a
    builtin subclass is excluded by an exact type test, but
    ``type(x).__name__`` goes through a metaclass, and the list of ways a
    Python object can refuse to be described is not closed. Any escape at
    all becomes :data:`UNREADABLE_PROPAGATION_IDENTITY`, which is a literal
    and so cannot fail; an escape on the HONEST path becomes a refusal too,
    which is the direction to fail in.
    """
    try:
        return _unpaired_reason(propagation, query_sha256)
    except Exception:  # noqa: BLE001 — a refusal may not itself raise
        return UNREADABLE_PROPAGATION_IDENTITY


def _unpaired_reason(propagation, query_sha256: str) -> str | None:
    """The body of :func:`unpaired_propagation`, netted by it."""
    try:
        recorded = getattr(propagation, "query_sha256", None)
    except Exception as e:  # noqa: BLE001 — a refusal may not itself raise
        return (
            f"unpaired propagation: reading the supplied propagation's "
            f"`query_sha256` raised "
            f"{type(e).__name__}, so it cannot be shown to be about the "
            f"query being judged ({query_sha256 or '<unhashable>'}) — an "
            f"identity that cannot be read is not a match. Judge the query "
            f"with the propagation `propagate()` produced from it."
        )
    if not isinstance(recorded, str) or not recorded:
        # EXACT type, not `isinstance`: an `int`/`str` SUBCLASS may carry a
        # `__repr__` that raises, and this string is a refusal's message
        shown = (
            repr(recorded)
            if type(recorded) in (str, bytes, int, float, bool, type(None))
            else f"<{type(recorded).__name__}>"
        )
        return (
            f"unpaired propagation: the supplied propagation records no "
            f"query identity ({shown}), so it cannot be shown to be "
            f"about the query being judged ({query_sha256 or '<unhashable>'}) "
            f"— an absent identity is not a match, it is an absence, and a "
            f"propagation of another query reports THAT query's obligation "
            f"statuses under this query's name (a stale cached propagation, "
            f"or one built before this field existed, has exactly this "
            f"shape). Judge the query with the propagation `propagate()` "
            f"produced from it."
        )
    if not query_sha256:
        return (
            f"unpaired propagation: the supplied propagation is about the "
            f"query {recorded}, but the query being judged cannot be hashed "
            f"at all (ClosedJaxpr.content_hash() raised), so the two cannot "
            f"be shown to be the same query — refusing to pair them. Judge "
            f"the query with the propagation `propagate()` produced from it."
        )
    if recorded != query_sha256:
        return (
            f"unpaired propagation: the supplied propagation was produced by "
            f"propagate() on the query {recorded}, but the query being judged "
            f"hashes to {query_sha256} — the two are not the same query, so "
            f"this propagation's obligation statuses are judgements about a "
            f"program other than the one being judged. An obligation the "
            f"interval leg DISCHARGED on the other query arrives already "
            f"discharged here and is reported by INDEX alone, so a mispaired "
            f"assembly can mint VERIFIED on a query whose honest verdict is "
            f"REFUTED. Judge the query with the propagation `propagate()` "
            f"produced from it."
        )
    return None


# -- literals and consts ------------------------------------------------------
#
# The unsigned formats joined with the pytree-probe registration round:
# registering `and`/`or`/`eq`/`ne` made the propagator *read* the uint mask
# literals of RNG plumbing (threefry constants and friends) that previously
# sat behind unknown primitives — without decoders the read raised and the
# whole analysis died on a legal trace (h_hard), the exact failure the
# guard rule forbids. Unsigned ints decode to python ints, which
# `_int_bracket` already brackets soundly above 2**53.
#
# `<f2` (float16) joined with audit 0.2.0 M12: without it EVERY float16
# harness that mentions a scalar constant — the ubiquitous
# `assert_(y > 0.0)` above all — bound ⊤ and topped out, so one of the
# four catalogued formats did not work on the ordinary shape of a
# harness. `struct`'s `e` code IS IEEE binary16, so the decode is exact
# and costs no dependency. (`obligation._SCALAR_STRUCT_FMT` has carried
# `<f2` all along; only the interval leg was short, and the two legs
# therefore disagreed about the same constant.)
_STRUCT_FMT = {
    "<f8": "d", "<f4": "f", "<f2": "e", "<i8": "q", "<i4": "i", "|b1": "?",
    "|u1": "B", "<u2": "H", "<u4": "I", "<u8": "Q",
    "|i1": "b", "<i2": "h",
}

# bfloat16 is the one catalogued format with no numpy scalar type and no
# `struct` code, and its dtype `.str` is **`<V2`: an anonymous 2-byte
# VOID**, which every other 2-byte structured dtype also spells. The byte
# string therefore does NOT identify the format, and a decoder keyed on
# `<V2` alone would read an arbitrary record type as a float — a wrong
# VALUE, which is worse than the ⊤ it replaces. The `ir.Aval` beside the
# value carries the dtype's NAME (`str(np.dtype(...))` -> `"bfloat16"`),
# and that does identify it, so :func:`_decode_array` takes the name as a
# disambiguator and decodes `<V2` **only** under it. A `<V2` payload with
# no name, or with a different one, stays undecodable — ⊤ with a note and
# `read_flag` True, exactly as before (audit 0.2.0 M12).
_BFLOAT16_DTYPE_STR = "<V2"
_BFLOAT16_DTYPE_NAME = "bfloat16"


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


def _elements(extents: tuple[int, ...]) -> int:
    """The element count of extents ALREADY NORMALISED by
    :func:`stelling.interval.check_shape` — never of a raw shape.

    The argument name says what the type does not. ``for d in shape: n *=
    d`` over the objects a caller supplied reaches ``__mul__``, a THIRD
    protocol beside the ``__index__`` every shape guard in this library
    validates with and the ``__eq__`` the shape comparisons use, so an
    extent could be validated at 2 and counted as 1 (audit 0.2.0 B8a,
    item 1 — the identical repair as :func:`stelling.obligation._size`,
    which states the same rule for the emission side). Taking the guard's
    RETURN VALUE rather than the shape is what makes the count and the
    guard that validated it one read.

    Its callers hand it :func:`stelling.interval.check_shape`'s return
    directly, and the shapes they check are ``ir.Array`` fields
    :meth:`stelling.ir.Array.__post_init__` has already installed as
    plain ``int``s — so this is a containment of the READING, not a fix
    for a live route through those two."""
    n = 1
    for d in extents:
        n *= d
    return n


def _decode_bfloat16(a: ir.Array) -> iv.IntervalArray:
    """The `<V2` payload of a bfloat16 array, decoded EXACTLY.

    bfloat16 is binary32 with the low 16 significand bits removed — same
    exponent field, same bias — so a bfloat16 value is exactly the binary32
    whose top half it is, and exactly the binary64 that binary32 is. The
    decode is therefore a bit shift and a `struct` reinterpret, with no
    rounding anywhere: `lo == hi`, like every other float row here.

    Called only under the aval's dtype NAME (see
    :data:`_BFLOAT16_DTYPE_STR`) — the `.str` alone cannot say a 2-byte
    void is this format.
    """
    # ONE READ: the guard hands back the extents it validated and the
    # count below is of THOSE (audit 0.2.0 B8a, item 1)
    n = _elements(iv.check_shape(a.shape))
    if len(a.data) != 2 * n:
        raise iv.IntervalError(
            f"array constant of shape {a.shape} dtype "
            f"{_BFLOAT16_DTYPE_NAME!r} carries {len(a.data)} byte(s), "
            f"expected {2 * n} — truncated or oversized payload "
            f"(malformed IR)"
        )
    los = []
    for u16 in struct.unpack(f"<{n}H", a.data):
        (v,) = struct.unpack("<f", struct.pack("<I", u16 << 16))
        los.append(float(v))
    # NaN payloads reach IntervalArray's own NaN refusal, which the
    # read site already turns into ⊤-with-maybe-NaN — the float64 NaN
    # sentinel's route, unchanged.
    return iv.IntervalArray(shape=a.shape, los=tuple(los), his=tuple(los))


def _decode_array(a: ir.Array, dtype_name: str | None = None) -> iv.IntervalArray:
    fmt = _STRUCT_FMT.get(a.dtype)
    if fmt is None:
        if (a.dtype == _BFLOAT16_DTYPE_STR
                and dtype_name == _BFLOAT16_DTYPE_NAME):
            return _decode_bfloat16(a)
        extra = ""
        if a.dtype == _BFLOAT16_DTYPE_STR:
            extra = (
                f" ({_BFLOAT16_DTYPE_STR!r} is an anonymous 2-byte void; "
                f"stelling decodes it as bfloat16 only when the aval names "
                f"that dtype, and this one names {dtype_name!r})"
            )
        raise ir.TranscriptionError(
            f"no zero-dep decoder for array dtype {a.dtype!r}{extra}; add one "
            f"before propagating this query"
        )
    # shape predicates first (fix re-attacks R1/N2): integral nonnegative
    # extents (a negative or string extent would reach struct.unpack as a
    # malformed format and raise raw), through the decline channel
    n = _elements(iv.check_shape(a.shape))
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


def _value_to_interval(
    v, shape: tuple[int, ...], dtype_name: str | None = None
) -> iv.IntervalArray:
    """The interval a literal/const value denotes.

    ``dtype_name`` is the *aval's* dtype name beside the value. It is a
    disambiguator, never an override: exactly one dtype (`bfloat16`) has a
    `.str` that does not identify it, and only that one row reads this
    argument (see :data:`_BFLOAT16_DTYPE_STR`). Omitting it can only lose
    precision — the value stays undecodable and binds ⊤ — never change a
    decoded value.
    """
    if isinstance(v, ir.Array):
        return _decode_array(v, dtype_name)
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

# Binary float formats this module knows, as (significand bits INCLUDING
# the implicit leading one, minimum normal exponent, maximum exponent).
#
# TWO consumers, and the second one JUDGES. The conversion-exactness
# classifier below reads it to word a decline NOTE, and CONVERSION
# admission really does not consult it — that stays gated on
# `_EXACT_CONVERSIONS` membership. But `_FLOAT_MAX` is derived from this
# table and `_member_bounds` reads both to decide whether a probe witness
# may be admitted AT ALL: a dtype absent here yields no member, hence no
# witness, hence no branch-scoped refutation. The comment that stood here
# said "admission never consults this table" without that qualification,
# and the member-witness repair made it false — a row added or removed
# here now moves verdicts.
_FLOAT_FORMATS: dict[str, tuple[int, int, int]] = {
    "float16": (11, -14, 15),
    "bfloat16": (8, -126, 127),
    "float32": (24, -126, 127),
    "float64": (53, -1022, 1023),
}

# The reverse view, for messages that hold a format tuple and owe the
# reader a NAME. bfloat16 and float32 share (emin, emax) and differ only
# in `p`, so the whole tuple is the key and the map is injective.
_FLOAT_FORMATS_BY_TUPLE: dict[tuple[int, int, int], str] = {
    v: k for k, v in _FLOAT_FORMATS.items()
}
if len(_FLOAT_FORMATS_BY_TUPLE) != len(_FLOAT_FORMATS):  # pragma: no cover
    # `raise`, not `assert`: a module-level assert is stripped under -O,
    # and a census that stops being enforced in an optimised deployment
    # is not a census (tests/test_optimize_mode_guards.py).
    raise RuntimeError(
        "two float formats share a (p, emin, emax) tuple — the reverse "
        "view would name the wrong one"
    )


def _conversion_exactness(src: str, dst: str) -> tuple[str, str]:
    """``(verdict, why)``: whether the cast ``src -> dst`` preserves EVERY
    representable source value — ``"exact"``, ``"inexact"``, or
    ``"unknown"`` — with ``why`` a fragment that is true on its own.

    This classifies for the decline NOTE only; admission stays gated on
    :data:`_EXACT_CONVERSIONS` membership, and the two must not be
    conflated. The note used to render every unlisted cast as "not exact"
    and add that "every other cast may change the value it carries" —
    measured false on uint32->uint64, uint8->int32, bool->float16 and
    uint32->float64, each exact at every representable source value and
    each declined by a sentence asserting a change that cannot happen.

    Verdicts are claims, so each is backed: "exact" and "inexact" only
    with a proof from the tables (integer ranges, float capacities).
    Everything else says "unknown" — including float->float format
    embeddings, where the format fact is not the whole story on this
    target: per-dtype DAZ was MEASURED flushing an f32 subnormal to 0.0
    across an f32->f64 convert (jax 0.11.0 CPU; see the ieee convert
    rule), so bit-format containment alone does not vouch for subnormal
    values. Integer and bool sources have no subnormals, so no such
    caveat dilutes their verdicts.
    """
    if src == dst:
        return "exact", "it is the identity"
    s_int, d_int = _INT_DTYPE_BOUNDS.get(src), _INT_DTYPE_BOUNDS.get(dst)
    s_flt, d_flt = _FLOAT_FORMATS.get(src), _FLOAT_FORMATS.get(dst)
    if s_int is not None and d_int is not None:
        (slo, shi), (dlo, dhi) = s_int, d_int
        if dlo <= slo and shi <= dhi:
            return "exact", (
                f"every {src} value lies in [{slo}, {shi}], inside {dst}'s "
                f"[{dlo}, {dhi}], where the conversion is the identity"
            )
        return "inexact", (
            f"{src} spans [{slo}, {shi}] but {dst} holds only "
            f"[{dlo}, {dhi}], and jax wraps or collapses what falls outside"
        )
    if s_int is not None and d_flt is not None:
        (slo, shi), (p, _, emax) = s_int, d_flt
        if p <= emax and -(2**p) <= slo and shi <= 2**p:
            return "exact", (
                f"{dst} represents every integer of magnitude at most "
                f"2**{p} exactly, and {src} spans [{slo}, {shi}]"
            )
        # One-directional, like the exact branch: magnitude <= 2**p is
        # SUFFICIENT for exactness, not necessary — float64 also carries
        # 2**54 exactly — so the sentence may not equate "the integers
        # {dst} carries exactly" with that band (audit repair R1).
        return "inexact", (
            f"{src} spans [{slo}, {shi}], while {dst} represents every "
            f"integer of magnitude at most 2**{p} exactly but not every "
            f"integer beyond (2**{p} + 1 already is not), so conversion "
            f"rounds — or overflows — some {src} values"
        )
    if s_flt is not None and d_int is not None:
        return "inexact", (
            f"{src} carries non-integers — 0.5 for one — which have no "
            f"exact {dst} image"
        )
    if s_flt is not None and d_flt is not None:
        (ps, ns, xs), (pd, nd, xd) = s_flt, d_flt
        if pd < ps:
            return "inexact", (
                f"{dst} keeps {pd} significand bits to {src}'s {ps}, so "
                f"some {src} values round"
            )
        if xd < xs:
            return "inexact", (
                f"{dst}'s exponents stop at {xd} while {src}'s reach "
                f"{xs}, so some finite {src} values overflow"
            )
        if nd - (pd - 1) > ns - (ps - 1):
            return "inexact", (
                f"{src}'s smallest subnormals fall below {dst}'s, so "
                f"they underflow"
            )
        return "unknown", (
            f"as formats {dst} embeds {src} ({ps} significand bits into "
            f"{pd}, exponent range covered), but per-dtype DAZ was "
            f"measured flushing a float32 subnormal to 0.0 across a "
            f"convert on this target (jax 0.11.0 CPU), so the embedding "
            f"alone does not establish that every subnormal {src} value "
            f"crosses unchanged"
        )
    return "unknown", (
        f"neither an exactness proof nor a value-change witness is "
        f"established here for it (a dtype outside the integer-bounds "
        f"and float-format tables)"
    )


# The example casts the decline note names as admitted — each checked
# against the whitelist AT THE MOMENT THE SENTENCE IS BUILT. The note
# used to hard-code "bool->any", and a traced bool->float16 — which is
# NOT in _EXACT_CONVERSIONS — was declined by the very sentence asserting
# bool casts are admitted. Preferred picks first, for a varied showing;
# should every pick be delisted someday, the first whitelist entries
# stand in, so the sentence never names an unlisted cast and never goes
# empty while the whitelist is nonempty.
_ADMITTED_EXAMPLE_PICKS = (
    ("float32", "float64"),
    ("int32", "float64"),
    ("bool", "int64"),
)


def _admitted_examples() -> str:
    picks = [p for p in _ADMITTED_EXAMPLE_PICKS if p in _EXACT_CONVERSIONS]
    if not picks:
        picks = sorted(_EXACT_CONVERSIONS)[:3]
    return ", ".join(f"{s}->{d}" for s, d in picks)


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
            extents = iv.check_shape(value.shape)
        except iv.IntervalError as e:
            return str(e)
        fmt = _STRUCT_FMT.get(value.dtype)
        if fmt is not None:
            # the count is of the extents the guard just validated, never
            # a second raw read (audit 0.2.0 B8a, item 1)
            n = _elements(extents)
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
    if (src, dst) == ("int64", "float64") and _finite_point(a):
        # A point interval at an integer value float64 represents exactly
        # (any integer in [-2**53, 2**53]) passes through: the conversion
        # is the identity on that value. Outside that range float64 rounds,
        # so non-point intervals and out-of-range points decline below.
        bound = 2**53
        if all(-bound <= x <= bound for x in (*a.los, *a.his)):
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
            # The PRINTED range is the exact integer pair from the bounds
            # table, never the float comparison bound: `bound - 1` is float
            # arithmetic, and for int64 it rounds straight back to 2**63
            # (the same trap the comment above pins for the comparison), so
            # the note used to name 9.223372036854776e+18 = 2**63 as
            # int64's maximum — a value int64 cannot hold.
            lo_b, hi_b = _INT_DTYPE_BOUNDS[dst]
            raise iv.IntervalError(
                f"{src} -> {dst} truncates toward zero, which this transfer "
                f"models — but the operand "
                f"spans [{min(a.los)}, {max(a.his)}], which leaves {dst}'s "
                f"representable range [{lo_b}, {hi_b}]. Outside it jax "
                f"clamps or wraps rather than truncating, and no interval rule "
                f"here models that. Narrowing the operand admits this form"
            )
        return [iv.IntervalArray(
            shape=a.shape,
            los=tuple(float(math.trunc(x)) for x in a.los),
            his=tuple(float(math.trunc(x)) for x in a.his),
        )]
    # THE REASON IS NAMED, and the source dtype with it. A `return None` here
    # produced the note "'convert_element_type' has no sound rule for params
    # {'new_dtype': 'float64', ...}; ⊤" — which prints the DESTINATION and
    # never the SOURCE, and the source is the load-bearing half. Measured: this
    # is a terminal in independently-authored external code, where a python int
    # literal promotes through `int64 -> float64`, and a reader of that note
    # cannot tell which side of the cast is the problem.
    #
    # WHAT the reason says is classified first: "unlisted" and
    # "value-changing" are different facts, and the previous single sentence
    # asserted the second of every decline — measured false on
    # uint32->uint64 and every other exact-but-unlisted cast. The DECISION
    # is unchanged in all three branches: this line declines exactly what it
    # declined; only the sentence now matches the cast it describes.
    verdict, why = _conversion_exactness(src, dst)
    if verdict == "exact":
        head = (
            f"the conversion {src!r} -> {dst!r} is exact at every "
            f"representable {src} value ({why}), yet the pair is not listed "
            f"in propagate._EXACT_CONVERSIONS — declined as unlisted, a "
            f"whitelist gap, not as value-changing"
        )
    elif verdict == "inexact":
        head = (
            f"the conversion {src!r} -> {dst!r} is not exact: {why} — and "
            f"this transfer declines rather than modelling the change"
        )
    else:
        head = (
            f"the conversion {src!r} -> {dst!r} is not listed in "
            f"propagate._EXACT_CONVERSIONS, and {why} — declined: only "
            f"casts vouched exact pass through"
        )
    # The admissions sentence enumerates EVERY return path above — the
    # identity, the whitelist, the in-range narrowing, and the in-range
    # float-source truncation (measured admitted with no note:
    # float64 -> int32 over (0.25, 100.75) discharges) — with the
    # truncation targets derived from _INT_RANGE so the sentence cannot
    # drift from the branch it describes (audit repair R2: the previous
    # sentence omitted the truncation admission).
    trunc_dsts = "/".join(sorted(_INT_RANGE))
    raise iv.IntervalError(
        f"{head}. NOTE THE SOURCE DTYPE, {src!r} — it is the half the "
        f"generic params note does not show. This transfer admits the "
        f"identity, the casts in propagate._EXACT_CONVERSIONS "
        f"({_admitted_examples()}, and the rest of that list), an "
        f"int64->int32 narrowing whose interval provably fits, and a "
        f"float-source truncation to {trunc_dsts} whose interval provably "
        f"fits the target's range"
    )


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
    Every decline names its reason and prints the numbers that made it
    decline — the tracer validates all of these before binding (measured:
    mis-summing sizes, a negative size and an out-of-range axis each raise
    at trace time), so each path below marks IR that did not come from
    that trace path: hand-built or edited-after-serialization.
    """
    if len(ins) != 1:
        raise iv.IntervalError(
            f"split cuts ONE operand along one axis, and this equation "
            f"binds {len(ins)} operands — there is no single array here "
            f"for the sizes to partition; check the hand-built or "
            f"deserialized IR that produced it"
        )
    (a,) = ins
    sizes = params.get("sizes")
    axis = params.get("axis")
    if sizes is None or axis is None:
        gone = " and ".join(
            repr(n) for n in ("sizes", "axis") if params.get(n) is None
        )
        raise iv.IntervalError(
            f"split carries no usable {gone} param (absent or None) — the "
            f"row cuts at the offsets 'sizes' gives along 'axis', and an "
            f"absent param is never guessed"
        )
    try:
        sizes = tuple(int(s) for s in sizes)
        axis = int(axis)
    except (TypeError, ValueError):
        raise iv.IntervalError(
            f"split params do not read as integers: "
            f"sizes={params.get('sizes')!r}, axis={params.get('axis')!r} — "
            f"the cut offsets are integer arithmetic on these, and reading "
            f"them raised"
        ) from None
    rank = len(a.shape)
    if not 0 <= axis < rank:
        raise iv.IntervalError(
            f"split axis {axis} lies outside the operand's rank {rank} "
            f"(operand shape {a.shape}) — there is no axis {axis} to cut"
        )
    if any(s < 0 for s in sizes):
        bad = tuple(s for s in sizes if s < 0)
        raise iv.IntervalError(
            f"split sizes {sizes} contain negative piece extents {bad} — "
            f"no piece holds a negative number of elements"
        )
    if sum(sizes) != a.shape[axis]:
        raise iv.IntervalError(
            f"split sizes {sizes} sum to {sum(sizes)}, but the operand's "
            f"axis {axis} has extent {a.shape[axis]} (operand shape "
            f"{a.shape}) — the cut does not partition the axis"
        )
    if len(sizes) != len(eqn.outvars):
        raise iv.IntervalError(
            f"split params name {len(sizes)} pieces but the equation binds "
            f"{len(eqn.outvars)} output(s) — the params and the equation's "
            f"own arity disagree about how many pieces exist"
        )

    out, start = [], 0
    for s in sizes:
        starts = tuple(start if d == axis else 0 for d in range(rank))
        limits = tuple(start + s if d == axis else a.shape[d]
                       for d in range(rank))
        out.append(iv.slice_(a, starts, limits, None))
        start += s
    return out


def _t_square(eqn, params, ins):
    """``jnp.square`` — a first-class primitive on jax 0.11.0, not sugar.

    DELEGATES to the exponent-2 case of :func:`interval.integer_pow`, which
    already carries the even-exponent rule and its audit: a straddling box maps
    to ``[0, max(lo², hi²)]``, a one-sided box to the tight monotone bracket.
    Measured: ``integer_pow([-2, 3], 2) = [0, 9]``.

    WHY THIS IS NOT THE REVERTED same-operand `mul` WORK. That change tried to
    detect ``mul(x, x)`` as a PATTERN and route it, which meant inspecting
    operand identity inside a binary transfer — invasive, and reverted. Here
    the primitive IS the square: **one operand, so the dependency problem does
    not arise at all.** The correlation ``mul`` cannot see is not present to be
    seen.

    External demand, measured: `jnp.square` appears at 137 call sites across 40
    files in jaxfluids alone, and its absence there poisoned a defensive
    ``x*x + eps`` division into a spurious divisor-may-be-zero decline — a
    `div` situation the where-refinement does not touch.
    """
    if len(ins) != 1:
        return None
    _refuse_complex(eqn, "square")
    _integer_pow_budget(ins[0], 2)
    return _int_overflow_guard(eqn, "square", [iv.integer_pow(ins[0], 2)])


def _t_copy(eqn, params, ins):
    """``copy_p`` is the identity. ``jnp.array(x)`` emits it, which is why a
    kinetic-energy contract was unreachable for want of a no-op."""
    if len(ins) != 1:
        raise iv.IntervalError(
            f"copy is unary and this equation carries {len(ins)} operand"
            f"{'' if len(ins) == 1 else 's'}. A traced `copy_p` always has "
            f"exactly one, so this IR was not produced by tracing — check the "
            f"serialization or the hand-built equation that reached here"
        )
    return [ins[0]]


def _t_unstack(eqn, params, ins):
    """``jnp.unstack`` — split along ``axis`` into one output per index.

    EXACT: every output element IS an input element at a static index, so this
    is built on :func:`interval.slice_` rather than fresh index arithmetic —
    the "don't hand-roll a traversal" norm applied to indexing. Declines when
    the params do not describe the operand it was handed; every decline
    names its reason with the numbers. The tracer validates the axis and
    fixes the output count before binding (measured: an out-of-range axis
    raises at trace time, a negative one is normalized, and the abstract
    eval binds one output per index), so each path below marks IR that
    did not come from that trace path: hand-built or
    edited-after-serialization.
    """
    if len(ins) != 1:
        raise iv.IntervalError(
            f"unstack routes ONE operand's indices along one axis to its "
            f"outputs, and this equation binds {len(ins)} operands — there "
            f"is no single array here to unstack; check the hand-built or "
            f"deserialized IR that produced it"
        )
    (a,) = ins
    axis = params.get("axis")
    if axis is None:
        raise iv.IntervalError(
            "unstack carries no usable 'axis' param (absent or None) — the "
            "row cuts along that axis, and an absent param is never guessed"
        )
    try:
        axis = int(axis)
    except (TypeError, ValueError):
        raise iv.IntervalError(
            f"unstack's axis param does not read as an integer: "
            f"axis={params.get('axis')!r} — the cut offsets are integer "
            f"arithmetic on it, and reading it raised"
        ) from None
    shape = tuple(a.shape)
    rank = len(shape)
    if not 0 <= axis < rank:
        raise iv.IntervalError(
            f"unstack axis {axis} lies outside the operand's rank {rank} "
            f"(operand shape {shape}) — there is no axis {axis} to cut"
        )
    if len(eqn.outvars) != shape[axis]:
        raise iv.IntervalError(
            f"unstack along axis {axis} of operand shape {shape} yields "
            f"{shape[axis]} piece(s), but the equation binds "
            f"{len(eqn.outvars)} output(s) — the operand and the equation's "
            f"own arity disagree about how many pieces exist"
        )
    out = []
    for i in range(shape[axis]):
        starts = tuple(i if d == axis else 0 for d in range(rank))
        limits = tuple(i + 1 if d == axis else shape[d] for d in range(rank))
        sl = iv.slice_(a, starts, limits, None)
        v = eqn.outvars[i]
        out.append(iv.IntervalArray(shape=tuple(v.aval.shape),
                                    los=sl.los, his=sl.his))
    return out


def _t_sign(eqn, params, ins):
    """``sign`` — and the obvious rule is UNSOUND, which is this row's point.

    The rule everyone writes is ``lo > 0 -> [1, 1]``. MEASURED on jax 0.11.0
    CPU binary64, eager AND under jit::

        lax.sign(1e-320) = 0.0        numpy.sign(1e-320) = 1.0

    XLA flushes subnormals (FTZ on results, DAZ on operands) in arithmetic,
    comparisons and libm alike — the same device-dependence ieee mode already
    discloses in :data:`interval.SUBNORMAL_INDETERMINACY_ASSUMPTION`. So a
    declared box like ``[1e-320, 1e-300]`` has ``lo > 0`` while the target
    evaluates its lower elements to ``0.0``, and the obvious rule records a
    DEFINITE ``[1, 1]`` that the execution does not satisfy — a false box of
    exactly the `square` class, and one a gauge sampling ``[-3, 3]`` never
    reaches. It is also why the oracle clause now reads *"jax, not stelling AND NOT
    numpy"* (CONTRIBUTING.md): a numpy oracle would have certified it. The
    executed-value containment sweep lives in the campaign repo, not here;
    what is in THIS tree is tests/test_sign_rem_rows.py, which pins the
    oracle disagreement and the load-bearingness of the floor.

    So the definite branches are gated on :data:`interval.MIN_NORMAL` rather
    than on zero, and the open band ``(-MIN_NORMAL, MIN_NORMAL)`` admits
    ``0``. The constant is READ from its definition site, not re-derived: it
    is already in the tree, already carries its derivation, and is already
    load-bearing for this exact device-dependence in ieee mode. Admitting 0
    across the band is sound under BOTH flush and gradual underflow, which is
    what makes one rule serve both semantics modes.

    STRICTER THAN ITS NEIGHBOURS, deliberately, and recorded rather than
    hidden: real-mode ``gt([1e-320, 1e-300], 0)`` returns definite TRUE today
    — the same device-dependence answered the other way. Whether real mode
    brackets the EXECUTED program or the real-arithmetic idealization is a
    MODE question that predates this row and reaches every comparison; this
    transfer takes the answer that is sound under either reading and changes
    nothing else to match.

    Integers have no subnormals, so their positive floor is 1. Unsigned needs
    no branch of its own: a uint box has ``lo >= 0``, so the negative cases
    are unreachable and there is no dead ``-1`` branch here to be misread as
    a live one.

    BINARY64 AND INTEGERS ONLY. The floor is `MIN_NORMAL = 2**-1022`, which
    is a *binary64* boundary; every other float dtype flushes somewhere else
    and declines via :func:`_refuse_non_f64_float`. That restriction was
    bought by a blinded audit, which found this row minting a **discharged
    obligation at 4/4 KNOWN coverage** on an f32 box that execution refutes.

    SOUND, NOT EXACT — and the tier says so because the row cannot be exact
    under both readings at once. Under gradual underflow the image of
    ``[-1e-320, 1e-320]`` is {-1, 0, 1} and ``[-1, 1]`` IS its hull; under
    the measured flush the image is {-0.0} and the hull is ``[0, 0]``. The
    returned box is sound under both, which is the point, and is the tight
    hull under neither. An earlier draft registered this EXACT on the
    gradual reading alone; the tier rides into the verdict stamp, so it
    states the weaker claim.
    """
    if len(ins) != 1:
        return None
    _refuse_complex(eqn, "sign", _SIGN_COMPLEX_REASON)
    _refuse_non_f64_float(eqn, "sign")
    (a,) = ins
    dtype = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    # the smallest magnitude whose sign is DEFINITE on the target
    floor_ = 1.0 if _is_integer_dtype(dtype) else iv.MIN_NORMAL
    los, his = [], []
    # NO NaN-ENDPOINT BRANCH HERE, and the absence is deliberate. Falling
    # through on a NaN endpoint WOULD be silently wrong — every comparison
    # below is False, so the box would read [-1, 1], which does not contain
    # sign(NaN) = NaN. But `IntervalArray` refuses a NaN endpoint at
    # CONSTRUCTION ("NaN endpoint in interval arithmetic"), so a guard here
    # could never fire, and a guard that cannot fire reads as protection
    # while providing none. The property is pinned at the constructor
    # instead; see tests/test_sign_rem_rows.py.
    for lo, hi in zip(a.los, a.his):
        if lo >= floor_:
            l, h = 1.0, 1.0
        elif hi <= -floor_:
            l, h = -1.0, -1.0
        elif lo >= 0.0 and hi <= 0.0:
            # the exact-zero box, tight. Without this branch the `lo >= 0`
            # case below fires first and returns [0, 1] for [0, 0], admitting
            # a 1 no operand in the box can produce (blinded audit).
            l, h = 0.0, 0.0
        elif lo >= 0.0:
            l, h = 0.0, 1.0
        elif hi <= 0.0:
            l, h = -1.0, 0.0
        else:
            l, h = -1.0, 1.0
        los.append(l)
        his.append(h)
    outs = [iv.IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))]
    if not _is_integer_dtype(dtype):
        return outs
    # Route integers through the shared guard like every other computing row
    # that accepts them. The `-1` branches are only unreachable for unsigned
    # dtypes if `lo >= 0` is guaranteed, and it is NOT: `any_array` validates
    # shape and ordering but not bounds-against-dtype, so a declared
    # uint8 box of (-3, -1) reached this and returned [-1, -1] — out of
    # uint8's range, at 100% coverage, minting a REFUTED. Found by a blinded
    # audit. The invariant is now enforced rather than asserted in a comment.
    return _int_overflow_guard(eqn, "sign", outs)


def _t_rem(eqn, params, ins):
    """``rem`` — TRUNCATED remainder, which is what ``lax.rem`` is.

    MEASURED, jax 0.11.0, floats and integers alike::

        lax.rem(-7, 3) = -1     truncated: the sign follows the DIVIDEND
        jnp.mod(-7, 3) =  2     floored:   the sign follows the DIVISOR

    They disagree on every mixed-sign case with a nonzero remainder, so
    modelling the floored rule here would be wrong on all of them. But the
    choice is not a trade-off, because ``jnp.mod`` / ``jnp.remainder`` / ``%``
    do not lower to this primitive at all. They lower to a ``jit`` composite
    whose body is::

        rem ; ne ; lt ; lt ; ne ; and ; add ; select_n

    — truncation plus seven primitives that already have rows. MEASURED with
    ``coverage.measure``: before this row ``jnp.mod`` is ``total=9 known=7
    transparent=1 unknown=1``, and the single unknown IS ``rem``. After it:
    ``known=8 transparent=1 unknown=0``. So the truncated row is the only one
    to build, and building it takes ``jnp.mod``'s unknowns to zero — stated
    as 7→8 known of 9 equations rather than "9/9", because the ninth is the
    transparent ``jit`` wrapper, which is counted in its own column and is
    neither known nor unknown.

    THE BOUND. Truncated remainder satisfies ``|rem(a,b)| <= min(|a|, |b|)``
    and carries the sign of ``a`` (or is zero): if ``|a| < |b|`` the result IS
    ``a``, otherwise its magnitude is strictly below ``|b|``. So the dividend
    box picks the side and the divisor's larger magnitude bounds the width.

    THE DIVISOR GUARD, and why it REFUSES rather than widening: ``rem(a, 0)``
    is NaN on floats, and NO INTERVAL CONTAINS NaN — a ⊤ here would be a false
    box, not merely a weak one. (On integers the consequence is different and
    the message says so: measured, ``lax.rem(int32(7), 0) = 7``. Refusing is
    over-conservative there rather than necessary.) Same for a non-finite
    dividend (``rem(inf, 2)`` is NaN). Both refuse loudly with the offending
    interval printed — the three properties two independently-blinded external
    agents rated 9/10 on stelling's **escalation-face** `div` guard: names the
    primitive, gives the reason, PRINTS THE BOX. Note that is the *emission*
    guard; real-mode `_t_div` now declines with DIV_STRADDLE_DECLINE when the
    divisor straddles zero (previously it returned ⊤ silently).
    This row is deliberately stricter than its sibling, and the attribution is
    to the message that earned the rating, not to `div`'s transfer.

    The guard's threshold is MIN_NORMAL rather than zero, for `sign`'s reason
    read on the divisor: under DAZ a subnormal divisor reads as ``0`` and the
    result is NaN, so a divisor box inside the open subnormal band is refused
    even though it excludes zero exactly. Same mode-boundary caveat as `sign`,
    same resolution — the reading that is sound under both semantics.

    SOUND, not exact: for ``a = [6, 7]``, ``b = [3, 3]`` the achievable image
    is {0, 1} and this returns ``[0, 3]``.
    """
    if len(ins) != 2:
        return None
    _refuse_non_f64_float(eqn, "rem")
    a, b = ins
    # THE CANONICAL ELEMENTWISE PAIRING, not a hand-rolled one. An earlier
    # draft required equal shapes and declined everything else "because jaxpr
    # rem is elementwise on equal shapes" — which is false: jaxprs carry
    # scalar literals as RANK-0 operands of elementwise equations, so
    # `jnp.mod(x, 2.0)` on an array and jax-md's `space.periodic` shift both
    # produce `rem` with shapes [(n,), ()] and both were declined outright.
    # `add`, `mul` and `div` have always routed through this helper. Found by
    # a blinded audit; the static coverage census read 0 unknown throughout,
    # which is the same census-cannot-see-guards mechanism recorded in
    # docs/state-0.1.0.md, hitting the row that documents it.
    try:
        shape, xs, ys = iv._pair_elements(a, b)
    except iv.IntervalError:
        return None
    los, his = [], []
    for (alo, ahi), (blo, bhi) in zip(xs, ys):
        if not (math.isfinite(alo) and math.isfinite(ahi)):
            raise iv.IntervalError(
                f"'rem' declined: the dividend's interval [{alo}, {ahi}] is "
                f"not bounded, and rem of an infinity is NaN (measured: "
                f"lax.rem(inf, 2) = nan). No interval contains NaN, so this "
                f"refuses rather than widening — a wide box here would be "
                f"FALSE, not weak. Bound the dividend to admit this form"
            )
        if blo < iv.MIN_NORMAL and bhi > -iv.MIN_NORMAL:
            int_out = _is_integer_dtype(
                (eqn.outvars[0].aval.dtype or "") if eqn.outvars else ""
            )
            why = (
                "contains zero"
                if blo <= 0.0 <= bhi
                else (
                    f"lies inside the open subnormal band "
                    f"(|b| < {iv.MIN_NORMAL}), where the measured target "
                    f"flushes the divisor to zero (DAZ)"
                )
            )
            # The consequence is dtype-dependent and the message says which:
            # measured, lax.rem(int32(7), 0) = 7, NOT NaN. A blinded audit
            # caught the float reason being printed at integer dtypes.
            consequence = (
                "integer rem by zero returns the DIVIDEND unchanged "
                "(measured: lax.rem(int32(7), 0) = 7), which no interval "
                "rule here models"
                if int_out
                else "rem(a, 0) is NaN, and no interval contains NaN"
            )
            raise iv.IntervalError(
                f"'rem' declined: the divisor's interval [{blo}, {bhi}] "
                f"{why}, and {consequence}. This refuses rather than "
                f"returning a wide box — a wide box here would be FALSE, not "
                f"weak. A precondition bounding the divisor away from zero "
                f"admits it"
            )
        m = max(abs(blo), abs(bhi))     # +inf is sound here: min() absorbs it
        if alo >= 0.0:
            lo, hi = 0.0, min(ahi, m)
        elif ahi <= 0.0:
            lo, hi = max(alo, -m), 0.0
        else:
            lo, hi = max(alo, -m), min(ahi, m)
        los.append(lo)
        his.append(hi)
    outs = [iv.IntervalArray(shape=shape, los=tuple(los), his=tuple(his))]
    dtype = (eqn.outvars[0].aval.dtype or "") if eqn.outvars else ""
    if not _is_integer_dtype(dtype):
        return outs
    # `div`'s posture on its integer sibling. Nothing can escape (|rem| < |b|
    # and |rem| <= |a|, both in range — measured at the boundary:
    # rem(INT_MIN, -1) = 0, where div WRAPS), and the guard's SNAP is a no-op
    # here because every endpoint is already integral. Its live residual
    # effect is the refusal on an unregistered or out-of-range integer dtype,
    # which is why it is called rather than skipped — stated accurately after
    # a blinded audit pointed out that "for the snap" was not the reason.
    return _int_overflow_guard(eqn, "rem", outs)


def _t_reshape(eqn, params, ins):
    if params.get("dimensions") is not None:
        raise iv.IntervalError(
            f"this reshape carries dimensions={tuple(params['dimensions'])!r}, "
            f"which PERMUTES the operand before reshaping it. The row models "
            f"the C-order flat identity only — the permuting form needs its "
            f"own rule and does not have one, so it declines rather than "
            f"reshaping as if the permutation were absent"
        )
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


def _t_not(eqn, params, ins):
    """Kleene NOT on bools — ``~a`` on boolean arrays."""
    dtypes = [v.aval.dtype for v in eqn.invars]
    if any(d != "bool" for d in dtypes):
        raise iv.IntervalError(
            f"'not' transfer covers bool operands only; got dtypes "
            f"{dtypes}"
        )
    return [iv.logical_not(ins[0])]


def _t_reduce_or(eqn, params, ins):
    dtypes = [v.aval.dtype for v in eqn.invars]
    if any(d != "bool" for d in dtypes):
        raise iv.IntervalError(
            f"'reduce_or' transfer covers bool operands only; got dtypes "
            f"{dtypes}"
        )
    return [iv.reduce_or(ins[0], tuple(_req(params, "axes", "reduce_or")))]


def _t_is_finite(eqn, params, ins):
    return [iv.is_finite(ins[0])]


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

    Everything else declines to a noted ⊤ that names its reason and
    prints the numbers: dynamic (non-point) or out-of-range indices
    (mode-dependent clamp/drop is the census's wedge bug class, never
    guessed), update windows, batching dims, higher ranks, computed
    updates. The legible causes — a combiner, leftover combiner consts,
    unreadable dimension numbers, an index dtype that cannot hold the
    leading-axis bound — are named BEFORE the general form failure, the
    same order and for the same reason as the emission face
    (:func:`stelling.obligation._scatter_set_plan`): each is a case the
    shared oracle also rejects, so the pre-checks change no decision,
    only the sentence.
    """
    # WHAT THIS FUNCTION RETAINS BEYOND THE SHARED ORACLE, enumerated because
    # an unenumerated retained check is how the two faces drifted apart once
    # already: the combiner gate lived here while the emission used only the
    # oracle, so `.apply` was admitted as `.set`. Everything below is a
    # precondition for CALLING the oracle, genuinely interval-domain, or a
    # named pre-check of a case the oracle also rejects (reason-order only,
    # never a decision).
    #
    #   len(ins) != 3        — arity. The oracle takes three shapes, so having
    #                          three operands is a precondition for calling it.
    #   update_jaxpr         — DELIBERATE DEFENSIVE DUPLICATE. The oracle now
    #                          gates this and is the authority; this copy is
    #                          kept, not removed, because it is the check whose
    #                          absence caused the defect and a redundant gate
    #                          costs nothing — and it now carries the legible
    #                          combiner sentence.
    #   update_consts / dimension_numbers readability / index dtype —
    #                          named pre-checks mirroring the emission face:
    #                          each fires only where the oracle would return
    #                          False anyway, so the decline set is the
    #                          oracle's; the pre-check owns the sentence.
    #   index point/finite/integral, and in-range — interval-domain. This face
    #                          reads the index from a propagated INTERVAL; the
    #                          emission reads it from static constants. Same
    #                          question, two representations, so it cannot live
    #                          in a shape-and-params oracle.
    if len(ins) != 3:
        raise iv.IntervalError(
            f"scatter takes an operand, one index array and one updates "
            f"array, and this equation binds {len(ins)} operand(s) — "
            f"check the hand-built or deserialized IR that produced it"
        )
    if params.get("update_jaxpr") is not None:
        raise iv.IntervalError(
            "scatter carries a combiner (update_jaxpr): this is an "
            "`x.at[k].apply(f)`-shaped equation, not the covered "
            "`x.at[k].set(v)`. It traces with the same dimension numbers, "
            "shapes, mode and static index as a set, beside a DUMMY "
            "updates operand (measured: 0.0 regardless of f) — modelling "
            "it as a set would write the dummy where the program computes "
            "f(operand[k])"
        )
    operand, indices, updates = ins
    indices_dtype = _scatter_indices_dtype(eqn)
    if params.get("update_consts"):
        raise iv.IntervalError(
            f"scatter carries non-empty update_consts "
            f"{params.get('update_consts')!r} with no update_jaxpr — "
            f"combiner state without a combiner; the traced set form "
            f"carries update_consts=() (measured), so check the hand-built "
            f"or deserialized IR that produced this"
        )
    dn = params.get("dimension_numbers")
    if not isinstance(dn, ir.NamedTupleParam):
        raise iv.IntervalError(
            f"scatter carries no readable dimension_numbers (got {dn!r}) "
            f"— the covered set form is recognized by those fields, and "
            f"without them the write's geometry is unknown; the traced "
            f"form records them, so check the hand-built or deserialized "
            f"IR"
        )
    if len(operand.shape) == 1 and not _scatter_index_dtype_covers(
        indices_dtype, operand.shape[0]
    ):
        bound = operand.shape[0] - 1
        table = _INT_DTYPE_BOUNDS.get(indices_dtype or "")
        if indices_dtype is None:
            why = (
                "no dtype is recorded for the index operand, so nothing "
                "vouches the bound is representable in it"
            )
        elif table is None:
            why = (
                f"dtype {indices_dtype!r} is not in the integer-bounds "
                f"table here, so nothing vouches the bound fits it"
            )
        else:
            why = (
                f"{indices_dtype} represents [{table[0]}, {table[1]}], "
                f"which does not contain {bound}"
            )
        raise iv.IntervalError(
            f"scatter's index dtype cannot be vouched to represent the "
            f"operand's leading-axis bound {bound} exactly: {why}. XLA "
            f"computes the out-of-bounds bound in the INDEX element type, "
            f"so the range check it performs is not the one this row "
            f"models — measured on jax 0.11.0: an int8 index column "
            f"writes at operand length 128 and silently DROPS an "
            f"in-range-looking write at 129"
        )
    if not _scatter_set_row_form(
        params, operand.shape, indices.shape, updates.shape,
        indices_dtype,
    ):
        # form outside the covered row — the shared oracle's call. The
        # covered core is derived from the SAME constant the oracle
        # reads, so this sentence cannot drift from the check.
        core = ", ".join(f"{k}={v!r}" for k, v in _SCATTER_SET_CORE.items())
        dn_got = ", ".join(f"{k}={v!r}" for k, v in dict(dn.fields).items())
        raise iv.IntervalError(
            f"scatter configuration (operand {operand.shape}, indices "
            f"{indices.shape}, updates {updates.shape}, {dn_got}) is "
            f"outside the covered static-index set row form: rank-1 "
            f"operand, one index row (1,), scalar update (), {core}, and "
            f"every other dimension-number field empty"
        )
    lo, hi = indices.los[0], indices.his[0]
    if lo != hi:
        raise iv.IntervalError(
            f"scatter's index spans [{lo}, {hi}] over the declared box — "
            f"not a single point, so no one element is THE written one, "
            f"and nothing here brackets a data-dependent write"
        )
    if not math.isfinite(lo) or lo != math.floor(lo):
        raise iv.IntervalError(
            f"scatter's index is the point {lo}, which is not a finite "
            f"integer — element positions are integers, so this index "
            f"names no element"
        )
    k = int(lo)
    if not 0 <= k < operand.shape[0]:
        raise iv.IntervalError(
            f"scatter index {k} is out of range for the operand's leading "
            f"axis: 0 <= {k} < {operand.shape[0]} fails — out-of-range "
            f"handling is mode-dependent (measured on jax 0.11.0: "
            f"FILL_OR_DROP drops the write and the operand passes through "
            f"unchanged; clip clamps the index into range from both "
            f"sides, 7 onto the last element and -2 onto the first) and "
            f"is never guessed"
        )
    los = list(operand.los)
    his = list(operand.his)
    los[k] = updates.los[0]
    his[k] = updates.his[0]
    return [iv.IntervalArray(shape=operand.shape, los=tuple(los), his=tuple(his))]


def _t_gather(eqn, params, ins):
    """``x[idx]`` in its leading-axis row form — the allowed-by-census
    structural addition from the MIME fvm laplacian census trace (the
    gather half of the operators' gather→compute→scatter pattern:
    ``phi[mesh.owner]`` / ``phi[mesh.neighbour]`` on rank-1 fields and
    ``grad[mesh.owner]`` on rank-2, with the mesh topology entering as
    definite const indices), widened by the index-bounds round to indices
    known only to a RANGE.

    Covered GEOMETRY, exactly, and unchanged by that round: operand of
    rank r >= 1; indices ``(N, 1)``; dimension numbers that collapse
    exactly the leading axis (``offset_dims = (1, …, r-1)``,
    ``collapsed_slice_dims = (0,)``, ``start_index_map = (0,)``, every
    batching field empty); ``slice_sizes = (1, *operand.shape[1:])``.

    Covered INDICES: any integral interval lying inside the leading axis.
    A point reproduces the exact row take, ``out[i] = operand[k_i]``; a
    range takes the elementwise hull of the rows it can reach. Pure data
    movement either way — every output element IS an operand element, so
    there is no arithmetic and no rounding, and a range only widens WHICH
    in-range elements are copied. All ``GatherScatterMode``\\ s agree on
    definitely-in-range indices, so the mode is still not constrained
    here: this transfer computes a value only where every admitted index
    is in range, which is exactly the condition under which the modes
    cannot disagree.

    Everything else declines to a noted ⊤ that names its reason and prints
    the numbers: an index range STRADDLING the axis (the out-of-range
    inputs would take a clamped or filled element — the census's wedge bug
    class, never guessed), a non-integral or unbounded index interval, an
    index dtype too narrow to hold the axis' bound, batching dims, window
    offsets not covering the full trailing block, multi-column index
    vectors. An index range DISJOINT from the axis is out of bounds for
    every declared input and is reported as a finding rather than a
    decline (:class:`interval.IndexOutOfBoundsError`).
    """
    if len(ins) != 2:
        raise iv.IntervalError(
            f"gather takes an operand and one index array, and this "
            f"equation binds {len(ins)} operand(s) — check the hand-built "
            f"or deserialized IR that produced it"
        )
    operand, indices = ins
    r = len(operand.shape)
    if r < 1:
        raise iv.IntervalError(
            "gather's covered row form takes rows of the operand's leading "
            "axis, and this operand is rank-0 (shape ()) — there is no "
            "leading axis to take rows from"
        )
    if len(indices.shape) != 2 or indices.shape[1] != 1:
        raise iv.IntervalError(
            f"gather indices have shape {indices.shape} — the covered row "
            f"form reads them as an (N, 1) column of leading-axis row "
            f"numbers, and this index array is not such a column"
        )
    dn = params.get("dimension_numbers")
    if not isinstance(dn, ir.NamedTupleParam):
        raise iv.IntervalError(
            f"gather carries no readable dimension_numbers (got {dn!r}) — "
            f"the covered row form is recognized by those fields, and "
            f"without them the take's geometry is unknown; the traced form "
            f"records them, so check the hand-built or deserialized IR"
        )
    fields = dict(dn.fields)
    want = {
        "offset_dims": tuple(range(1, r)),
        "collapsed_slice_dims": (0,),
        "start_index_map": (0,),
    }
    if any(fields.get(k) != v for k, v in want.items()):
        got = ", ".join(f"{k}={fields.get(k)!r}" for k in want)
        cov = ", ".join(f"{k}={v!r}" for k, v in want.items())
        raise iv.IntervalError(
            f"gather dimension numbers do not collapse exactly the leading "
            f"axis of this rank-{r} operand: got {got}, where the covered "
            f"leading-axis row form is {cov} — a different take geometry "
            f"has no rule here"
        )
    extra = {k: v for k, v in fields.items() if k not in want and v != ()}
    if extra:
        raise iv.IntervalError(
            f"gather carries non-empty dimension-number field(s) beyond "
            f"the covered three: {extra!r} — the covered row form has "
            f"every such field (the batching fields today) empty"
        )
    got_ss = tuple(params.get("slice_sizes", ()))
    want_ss = (1,) + operand.shape[1:]
    if got_ss != want_ss:
        raise iv.IntervalError(
            f"gather slice_sizes {got_ss} do not take one full row: the "
            f"covered row form takes (1, *operand.shape[1:]) = {want_ss} "
            f"of the operand (shape {operand.shape}) per index"
        )
    ranges = []
    for i, (lo, hi) in enumerate(zip(indices.los, indices.his)):
        ranges.append(
            _classify_index_range(
                lo, hi, operand.shape[0] - 1, f"gather index element {i}",
                f"the operand's leading axis (shape {operand.shape})",
            )
        )
    # AFTER the classification, deliberately: this gate protects the VALUE
    # about to be computed, and a value is computed on the in-range path
    # only. Asking it first would answer a malformed float-dtype index with
    # a dtype-range complaint when the informative decline is that the
    # index is not an integer at all.
    _index_dtype_covers_or_decline(
        _index_operand_dtype(eqn, 1), operand.shape[0] - 1, "gather"
    )
    return [iv.take_row_ranges(operand, ranges)]


# -- index-bounds reasoning for dynamic indexing -----------------------------
#
# WHAT THIS BUYS, and the one thing it must never do.
#
# `u[i]` with a traced `i` is not a gather. MEASURED on jax 0.11.0 and
# 0.10.2: `jnp`'s `__getitem__` emits `lt/add/select_n` (the from-the-end
# normalisation) and then a `dynamic_slice`, and so does an out-of-range
# STATIC index — `u[3]` on a length-10 array lowers to a static `slice`,
# but `u[30]` and `u[-11]` fall back to the same normalise-then-
# `dynamic_slice` path. `dynamic_slice` had no transfer, so every dynamic
# index — and every statically-out-of-range one — collapsed the value to ⊤
# and killed all reasoning downstream of it. Indexing is everywhere in
# scientific code; this is the largest single power gap the census left.
#
# THE CLAMP, AND WHY IT IS NOT MODELLED. MEASURED, primitive-level, by
# binding `dynamic_slice_p` directly (which skips the wrapper's
# normalisation): jax CLAMPS the start of a gather/slice into the legal
# window — `dynamic_slice(arange(10), 30, (1,))` reads element 9, and
# `dynamic_slice(arange(10), -1, (1,))` reads element 0 — while a scatter
# with an out-of-range index is silently DROPPED (`x.at[30].set(v)` on a
# length-10 `x` is a no-op). A transfer that modelled the clamp would be
# sound about the *executed* program and wrong about the program the user
# WROTE: it would answer `u[i] == u[9]` with "definitely true" for `i` in
# [12, 20]. That is precisely the shape of the integer-literal wrap defect,
# and the tree's stated posture — *"integers and converts are
# execution-faithful"* (SOUNDNESS.md, the fixed-width boundary) — makes the
# tension real rather than rhetorical, so the choice is argued here and in
# SOUNDNESS.md rather than assumed. What decides it is that THERE IS NO
# SINGLE CLAMP TO BE FAITHFUL TO — measured, not argued:
#
#   1. One gather, one out-of-range index, TWO values. Measured on jax
#      0.11.0, index 30 into a 10-element operand: mode CLIP returns
#      element 9, mode FILL_OR_DROP returns the fill value. In range, all
#      three modes agree. So "the clamp" is not a property of the
#      operation; it is a property of a param, and modelling it would mean
#      picking one of two answers the same jaxpr can carry.
#   2. Read and write disagree too: the gather clamps, the scatter DROPS
#      (`x.at[30].set(v)` on a length-10 `x` is a no-op, measured), and the
#      same source-level `x[i]` picks one or the other by which side of an
#      assignment it lands on.
#
# An int32 `add`'s wrap has neither property: it is one defined,
# reproducible answer, which is why THAT is modelled and this is not.
#
# MEASURED AND NOT A REASON, corrected TWICE, because the first version of
# this paragraph asserted it without running it and the second generalised
# past what it ran. jax's non-inverse property out of bounds IS real, on
# 0.11.0 and 0.10.2: under GatherScatterMode.PROMISE_IN_BOUNDS, XLA's
# gather CLAMPS and its scatter DROPS, and the transpose of a gather is a
# scatter — so `u.at[array([30])].get()` reads element 9 while its
# cotangent is identically ZERO (the true d/du_9 is 1), and
# `x.at[30].set(v, mode="promise_in_bounds")` drops the write (the true
# d/dv is 0) while AD answers 1. The READ half reproduces at the DEFAULT
# indexing mode; the write half needs the mode spelled out, because
# `.at[...].set()` defaults to FILL_OR_DROP, which agrees.
#
# It does NOT reach the pair this rule sits on. `u[i]` is a dynamic_slice
# transposing to dynamic_update_slice, and both CLAMP: the cotangent of
# `u[30]` lands on element 9, exactly where the clamped read came from.
# The retraction measured that pair and the FILL_OR_DROP scatter/gather
# pair — the two SELF-CONSISTENT ones — and wrote a claim about all of AD,
# which is the failure it was written to correct. Either way it decides
# nothing here: the two reasons above stand without it.
#
# So the rule below computes a value ONLY where jax's clamp is provably the
# IDENTITY. That is the whole soundness argument in one line, and it is why
# minting a false VERIFIED through this path requires the hull to be wrong,
# not the clamp story.
#
# Three cases, and only the first produces a value:
#
#   (1) index range inside the legal window  -> the hull over every
#       reachable slice. The power gain.
#   (2) range straddles the window           -> DECLINE, named: the value
#       depends on the clamp for some declared inputs and on nothing else
#       for the rest, and no box states that.
#   (3) range disjoint from the window       -> PROVABLY out of bounds for
#       every input the user declared. Reported as a finding
#       (`iv.IndexOutOfBoundsError`, its own shouted note), still ⊤.


def _index_operand_dtype(eqn, position: int) -> str | None:
    """The aval dtype of an equation's index operand, or None when the
    equation does not carry one. The interval domain is bounds-only, so
    every dtype question is asked of the EQUATION — the same place
    :func:`_scatter_indices_dtype` asks it."""
    if len(eqn.invars) <= position:
        return None
    return eqn.invars[position].aval.dtype


def _index_dtype_covers_or_decline(
    dtype: str | None, bound: int, prim: str
) -> None:
    """Decline unless the largest legal index is exactly representable in
    the index operand's element type.

    XLA computes an index's out-of-bounds bound IN THE INDEX ARRAY'S
    ELEMENT TYPE — the fact :func:`_scatter_index_dtype_covers` was written
    for, where a bound that does not fit WRAPS and silently changes which
    indices are treated as in range. Probing jax 0.11.0's `dynamic_slice`
    with an `int8` start over operands of length 100/127/128/129/200 did
    NOT exhibit a wrapped clamp bound (every reachable start read the
    element it should), so the failure is UNCONFIRMED for this primitive —
    but "I could not make it happen" is not "it cannot happen", and the
    gate costs nothing where it matters: every index dtype jax's own
    indexing produces is `int32`/`int64`, for which any array length jax
    can allocate fits. Refusing the unconfirmed case is free; guessing it
    is not."""
    bounds = _INT_DTYPE_BOUNDS.get(dtype or "")
    if bounds is None:
        raise iv.IntervalError(
            f"{prim} index dtype {dtype!r} is not one this build knows the "
            f"range of, so whether the largest legal index {bound} is "
            f"representable in it cannot be checked — and XLA computes the "
            f"out-of-bounds comparison in the index's own element type"
        )
    lo, hi = bounds
    if not lo <= bound <= hi:
        raise iv.IntervalError(
            f"{prim} index dtype {dtype!r} spans [{lo}, {hi}], which does "
            f"not hold the largest legal index {bound} — XLA computes the "
            f"out-of-bounds comparison in the index's element type, where "
            f"that bound WRAPS, so the comparison performed is not the one "
            f"modelled here"
        )


def _classify_index_range(
    lo: float, hi: float, top: int, what: str, where: str
) -> tuple[int, int]:
    """The three-case classifier shared by every dynamic-index transfer.

    ``[lo, hi]`` is a propagated index interval and ``[0, top]`` the legal
    window (``top`` is ``n - 1`` for a gather row, ``n - slice_size`` for a
    dynamic slice start). Returns the integer range on the inside case, and
    raises on the other two: :class:`interval.IndexOutOfBoundsError` when
    the interval cannot contain a legal index at all, a plain
    :class:`interval.IntervalError` when it straddles.

    Endpoints must be finite integers. They are NOT rounded inward to the
    integers they contain: this layer is handed bounds with no dtype, and
    narrowing ``[0.5, 5.5]`` to ``[1, 5]`` would be sound only if the value
    really is integral. jax's own typing rule says index operands are
    integral, but a hand-built or deserialised query is not bound by it,
    and the cost of the strict reading is a decline on a form no trace
    produces."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise iv.IntervalError(
            f"{what} spans [{lo}, {hi}], which is not finite — an index "
            f"with an unbounded side names no position in {where}"
        )
    if lo != math.floor(lo) or hi != math.floor(hi):
        raise iv.IntervalError(
            f"{what} spans [{lo}, {hi}], whose endpoints are not integers "
            f"— positions in {where} are integers, and this layer does not "
            f"round an index inward to the ones it contains"
        )
    ilo, ihi = int(lo), int(hi)
    if ihi < 0 or ilo > top:
        raise iv.IndexOutOfBoundsError(
            f"{what} spans [{ilo}, {ihi}], and EVERY value in it is outside "
            f"{where}: the legal positions are [0, {top}]. This is not a "
            f"gap in stelling — over the whole declared set there is no "
            f"input for which this index is in bounds. jax will not raise: "
            f"measured on jax 0.11.0, a read clamps into range and a write "
            f"is dropped, so the program silently computes with the wrong "
            f"element. stelling withholds the value rather than modelling "
            f"the clamp. (jnp normalises a from-the-end index upstream of "
            f"this equation, so the span printed is the position actually "
            f"asked for: on a length-10 axis, u[-11] arrives here as -1)"
        )
    if ilo < 0 or ihi > top:
        raise iv.IntervalError(
            f"{what} spans [{ilo}, {ihi}], which straddles the legal "
            f"positions [0, {top}] of {where} — some inputs the declared "
            f"set admits index in bounds and some do not, and the "
            f"out-of-bounds ones take a clamped (read) or dropped (write) "
            f"element that is not the one written. No box states that, so "
            f"no value is claimed here"
        )
    return ilo, ihi


def _start_ranges(eqn, ins, first: int, sizes, prim: str):
    """Classify one scalar start index per axis for the dynamic-slice
    family, and check each index operand's dtype covers its axis' bound."""
    operand = ins[0]
    rank = len(operand.shape)
    starts = ins[first:]
    if len(starts) != rank:
        raise iv.IntervalError(
            f"{prim} takes one scalar start index per axis of its rank-"
            f"{rank} operand (shape {operand.shape}) and this equation "
            f"binds {len(starts)}"
        )
    out = []
    for d, (start, n, s) in enumerate(zip(starts, operand.shape, sizes)):
        if start.shape != ():
            raise iv.IntervalError(
                f"{prim} start index for axis {d} has shape {start.shape}; "
                f"the primitive takes one SCALAR start per axis"
            )
        if not 0 <= s <= n:
            raise iv.IntervalError(
                f"{prim} takes {s} element(s) along axis {d} of an extent-"
                f"{n} axis (operand shape {operand.shape})"
            )
        out.append(
            _classify_index_range(
                start.los[0], start.his[0], n - s,
                f"{prim} start index for axis {d}",
                f"axis {d} of the operand (shape {operand.shape}, taking "
                f"{s} element(s) there)",
            )
        )
        # after the classification, for the reason given in _t_gather
        _index_dtype_covers_or_decline(
            _index_operand_dtype(eqn, first + d), n - s, prim
        )
    return tuple(out)


def _t_dynamic_slice(eqn, params, ins):
    """``lax.dynamic_slice`` — what ``u[i]`` with a traced ``i`` lowers to,
    and what an out-of-range static index lowers to as well.

    Covered: any rank, any ``slice_sizes``, start indices known to any
    integer interval that lies inside the axis' legal start window
    ``[0, n_d - s_d]``. The result is the elementwise hull over every start
    the declared set admits (:func:`interval.dynamic_slice_hull`), which is
    tight on a single axis. A point start reproduces the exact slice.

    Declines, each naming its reason: a start straddling the legal window
    (the clamp would decide the value for some inputs), a start range
    disjoint from it (reported as an out-of-bounds FINDING), a non-integral
    or unbounded start interval, an index dtype too narrow to hold the
    axis' bound, and a hull whose enumeration would exceed the work budget.

    Data movement only: every output element IS an input element, so there
    is no arithmetic and no rounding — tier exact."""
    if not ins:
        raise iv.IntervalError(
            "dynamic_slice takes an operand and one start index per axis, "
            "and this equation binds none"
        )
    operand = ins[0]
    sizes = tuple(int(s) for s in _req(params, "slice_sizes", "dynamic_slice"))
    if len(sizes) != len(operand.shape):
        raise iv.IntervalError(
            f"dynamic_slice slice_sizes {sizes} do not match the rank of "
            f"its operand (shape {operand.shape})"
        )
    ranges = _start_ranges(eqn, ins, 1, sizes, "dynamic_slice")
    return [iv.dynamic_slice_hull(operand, ranges, sizes)]


def _t_dynamic_update_slice(eqn, params, ins):
    """``lax.dynamic_update_slice`` — the write sibling of
    :func:`_t_dynamic_slice`, and what ``u.at[i].set(v)`` lowers to when
    the update is a contiguous block (a scalar ``.at[i].set`` traces to a
    ``scatter`` instead, measured on jax 0.11.0).

    The operand keeps its own value everywhere some admitted start does not
    write, and takes the update's values everywhere some admitted start
    does; positions where both are possible get the hull of the two. With a
    point start that is exact.

    Same three cases and the same declines as the read side — the start
    must lie inside ``[0, n_d - s_d]``, because jax clamps a write's start
    exactly as it clamps a read's (measured, primitive-level: a length-2
    update at start 20 of a length-10 operand lands at index 8)."""
    if len(ins) < 2:
        raise iv.IntervalError(
            f"dynamic_update_slice takes an operand, an update and one "
            f"start index per axis, and this equation binds {len(ins)}"
        )
    operand, update = ins[0], ins[1]
    if len(update.shape) != len(operand.shape):
        raise iv.IntervalError(
            f"dynamic_update_slice update shape {update.shape} does not "
            f"match the rank of its operand (shape {operand.shape})"
        )
    ranges = _start_ranges(
        eqn, ins, 2, update.shape, "dynamic_update_slice"
    )
    return [iv.dynamic_update_slice_hull(operand, update, ranges)]


# The core scatter-add dimension-number fields shared by every measured
# form; any OTHER field a jax version adds (batching dims today) must be
# empty or the form oracle returns None (the transfer declines).
_SCATTER_ADD_CORE = {
    "inserted_window_dims": (0,),
    "scatter_dims_to_operand_dims": (0,),
}


def _is_add_combiner(update_jaxpr) -> bool:
    """Whether a ``scatter-add`` equation's recorded combining function is
    the measured single-``add`` form: no consts, two scalar invars, and
    exactly one ``add`` equation combining them into the single outvar
    (measured on jax 0.10.2 and 0.11.0 — every traced ``scatter-add``
    carries exactly this). The primitive NAME declares the accumulate
    semantic; a recorded combiner that contradicts the name is malformed
    IR and the form oracle refuses it rather than trusting either
    self-description.

    Read STRUCTURALLY, across BOTH jaxpr container shapes, because the
    container is a jax-series artifact and not a semantic difference. jax
    0.10 records this param as a bare ``Jaxpr``, which stelling
    transcribes to :class:`stelling.ir.Jaxpr`; jax 0.11 merged ``Jaxpr``
    and ``ClosedJaxpr`` into one class, so the identical combiner now
    satisfies ``isinstance(v, jex_core.ClosedJaxpr)`` first and
    transcribes to :class:`stelling.ir.ClosedJaxpr` closed over an empty
    const tuple. Measured: the same ``x.at[0].add(5.0)`` yields
    ``update_jaxpr`` as ``ir.Jaxpr`` on 0.10.2 and ``ir.ClosedJaxpr`` on
    0.11.0. An ``isinstance`` test against either class alone therefore
    reads as "no combiner" on the other series and silently declines a
    genuine ``.at[].add`` — which is exactly how this was found. Both
    shapes are held to the SAME emptiness requirement: a closed form must
    carry no consts, an open one no constvars."""
    if isinstance(update_jaxpr, ir.ClosedJaxpr):
        consts, j = update_jaxpr.consts, update_jaxpr.jaxpr
    elif isinstance(update_jaxpr, ir.Jaxpr):
        consts, j = update_jaxpr.constvars, update_jaxpr
    else:
        return False
    if consts or len(j.invars) != 2 or len(j.outvars) != 1:
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
    .scatter_add_rows` carries the argument), each step bumped outward
    ONLY where that step is inexact, untouched elements copied exactly.

    *That read "one outward bump per real addition", which was the rule
    before the accumulation moved onto the exact-``Fraction`` route and is
    not the rule now: a step whose exact sum is representable is not
    bumped at all, so the number of bumps is a property of the DATA and
    not of the number of contributions. Measured: a single ``[0] += [1]``
    row comes back ``(1.0, 1.0)`` — one real addition, no bump.*
    :func:`stelling.interval.reduce_sum` *retracts this same sentence
    about itself, and this copy of it was left standing one module over.*

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


def _refuse_non_f64_float(eqn, prim: str) -> None:
    """A real-mode guard whose threshold is :data:`interval.MIN_NORMAL` is a
    BINARY64 guard, and must say so by declining every other float dtype.

    A DELIBERATE DEPARTURE FROM A STATED POSTURE — not a defect fix, and the
    distinction was got wrong once already. SOUNDNESS.md had adjudicated this
    before these rows existed: *"real-mode boxes excluding the executed value
    at every non-binary64 float dtype, shared by `add`, `mul` and every
    arithmetic row… **is not a defect in any row**: ieee mode is gated to
    binary64, and real mode has no dtype gate at all, so **ℝ-judgement of a
    narrower float is the stated posture**."* Under that posture the f32 case
    below is the posture behaving as documented.

    WHAT WAS MEASURED. `MIN_NORMAL` is ``2**-1022``; float32's smallest
    normal is ``2**-126``, ~270 orders of magnitude above it. So an f32
    operand box like ``[1e-40, 1e-30]`` clears the f64 floor while the target
    evaluates its lower elements to ``0.0``: `sign` returned a definite
    ``[1, 1]`` where jax returns ``0.0``, and end to end that was a
    **discharged obligation at 4/4 KNOWN coverage** which execution refutes.
    For `rem` the executed value is **NaN** (``rem(1.0, 5e-39)``), which no
    interval contains under any reading.

    WHY THE DEPARTURE IS DEFENSIBLE: **declining is safe under BOTH readings.**
    Under execution-bracketing it removes a false box; under ℝ-judgement it is
    merely over-conservative, and over-conservative never mints a verdict. The
    reverse — keeping the floor and admitting f32 — is safe under only one.
    A row that must be sound before the mode question is settled takes the
    option that does not depend on the answer.

    KNOWN INCONSISTENCY, recorded rather than hidden: **`add` and `mul` box
    f32 the same way and have no such gate.** Two rows now depart from the
    stated posture while their siblings follow it. That is a divergence with
    a known cause, not a defect in either place, and closing it is a MODE-WIDE
    decision — either every arithmetic row takes the gate, or these two drop
    it — which needs the per-dtype table of smallest normals (numeric-constant
    work) or the ℝ reading confirmed. Neither is done here.

    The SHAPE of the decline is not a new adjudication: the ieee face answered
    the same question the same way (re-attack U2, see
    :func:`_non_f64_float_dtypes` and tests/test_ieee_f32_band.py), pinning
    float16 as a decline *"not as target behavior"* even though f16 is
    measured NOT to flush here.
    """
    bad = _non_f64_float_dtypes(eqn.invars)
    if bad:
        raise iv.IntervalError(
            f"{prim!r} declined: its subnormal guard is a BINARY64 guard "
            f"(threshold {iv.MIN_NORMAL}, which is 2**-1022), and operand "
            f"dtype{'s' if len(bad) > 1 else ''} {'/'.join(bad)} flush at a "
            f"different boundary — float32's smallest normal is 2**-126, so "
            f"values this guard reads as definitely-nonzero are flushed to "
            f"zero by the target. Declined rather than mismodelled; float64 "
            f"and integer operands are unaffected"
        )


_EVEN_POWER_COMPLEX_REASON = (
    "the even-power non-negativity rule is a real-arithmetic fact and "
    "complex squaring does not satisfy it (measured: square of a pure "
    "imaginary is a NEGATIVE real)"
)

# `sign` shares the guard and NOT the reason: it uses no even-power rule.
# A blinded audit found it emitting `square`'s message, which named a rule
# the row does not have — the refusal was right and its stated cause was
# about something else entirely.
_SIGN_COMPLEX_REASON = (
    "sign(z) for complex z is z/|z|, a point on the unit circle, not a real "
    "-1/0/+1 (measured: lax.sign(3+4j) = 0.6+0.8j) — the sign rule's three "
    "definite values do not describe it"
)


def _refuse_complex(eqn, prim: str, reason: str = _EVEN_POWER_COMPLEX_REASON) -> None:
    """The even-power non-negativity rule is a REAL-arithmetic fact, and this
    is the guard that keeps it from being applied where it is false.

    Found by a blinded audit of the `square` row. jax declares `square_p` as
    ``standard_unop(_int | _float | _complex, 'square')`` — complex IS in its
    domain — and `interval.integer_pow`'s even-exponent branch returns
    ``[0, +inf]`` for a ⊤ operand. Measured: for ``square(x.astype(complex128)
    * 1j)`` over a declared real box ``[1, 2]``, stelling claimed ``[0, inf]``
    while jax returns ``-1``, ``-2.25``, ``-4`` — genuinely REAL negatives,
    outside the box, recorded as KNOWN coverage rather than declined.

    It applies to `integer_pow` identically, which is where `square` inherited
    it by delegating: `_t_integer_pow` guards dtype only for a NEGATIVE
    exponent, so `z ** 2` on a complex operand had the same false box. Both
    are guarded here rather than only the row under audit, because a known
    false box is not made acceptable by being pre-existing.

    The codebase already holds this posture elsewhere — `_t_sqrt` refuses
    non-float loudly and `_dot_general_row_form` declines complex operands
    explicitly. This closes the gap those two already stood in.
    """
    dt = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    if dt.startswith("complex"):
        raise iv.IntervalError(
            f"{prim!r} on dtype {dt!r}: {reason}. Complex is outside this "
            f"transfer's domain — declined"
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
    integer — |result| < 2**53. Above that a result the double cannot
    represent exactly keeps a bracket many integers wide (measured on an
    int64 ``scatter-add``: an in-range ``2**62 + 1`` accumulate, whose
    exact total needs 63 bits, keeps the snapped bracket
    ``[2**62, 2**62 + 1024]`` — one binade's ulp, 1024 integers — and
    stays undecided), so "in-range integer arithmetic keeps its exact
    result" is a magnitude-conditional claim: exact below 2**53, a
    sound-but-wide in-range bracket above. Deliberate: tightness is never
    bought at the price of the bracket.

    *This read the claim off ``2**62 + 0``, citing*
    ``[2**62 - 512, 2**62 + 1024]`` *and "stays undecided", with*
    ``2**62 + 2**62`` *beside it "escaping the range and declining
    correctly". NEITHER HALF MEASURES WHAT IT SAYS ANY MORE. Adding zero
    has an exact representable total, so it was never a witness for
    magnitude at all: the 512 below it was the arithmetic kernel's
    unconditional outward bump, and with the kernels on the
    exact-``Fraction`` route that case now comes back exactly*
    ``[2**62, 2**62]`` *and DISCHARGES —* ``2**62 + 1`` *replaces it
    above, because its exact total genuinely needs 63 bits. And*
    ``2**62 + 2**62`` *does still decline, but the class moved:
    :data:`_INT_OVERFLOW_DECLINE` fired for it while the bracket carried a
    bump, and* :data:`_INT_BRACKET_DECLINE` *fires now that it does not. So
    "declines correctly" no longer means the overflow class — see the
    width test below, whose calibration this citation was resting on.*

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
            #
            # KNOWN MISCALIBRATED, AND NOT FIXED HERE. The test below is a
            # PROXY: it asks how far the escaping END is past the boundary
            # and ignores where the bracket's OTHER end sits. A bracket
            # that snapped to a single integer carries no slack at all, so
            # its escape IS demonstrated however close to the boundary it
            # lands -- and this test declines it as unresolvable anyway.
            # The predicate that says what the comment above means is
            # `lo > hi_b or hi < lo_b`: the WHOLE snapped bracket outside
            # the representable range, which no amount of bracket width
            # can explain away.
            #
            # Measured on `61de794`, before the scatter-add accumulation
            # moved to the exact route: `add(2**62, 2**62)` on int64
            # already declines here, under `_INT_BRACKET_DECLINE` and this
            # same 2048-wide message, for an exact bracket of
            # `[2**63, 2**63]` that exceeds int64's top by exactly 1. So
            # this is not a consequence of any one operation getting
            # tighter: it is reachable from ANY operation whose in-range
            # bracket is exact, and `_add_lo`/`_add_hi` have been
            # exact-when-representable since `1be900d`. Each later
            # conversion only joins that set. Sound either way -- the
            # verdict is a decline in both classes, and only the REASON is
            # less informative than the truth. Left alone deliberately: it
            # is an integer-overflow classification rule with its own
            # soundness argument and its own callers, and its own dispatch.
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
    "lt", "gt", "le", "ge", "eq", "ne", "and", "or", "not", "reduce_or",
    "is_finite",
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

# Naming a WORKING ALTERNATIVE, not just the missing primitive. "no transfer for
# reduce_and" tells a reader the tool is incomplete; this tells them what to
# write instead. The cause is a REGISTRY ASYMMETRY — `reduce_or` has a transfer
# in both registries and `reduce_and` has one in neither — so `jnp.any` decides
# and `jnp.all` does not; tests/test_membership_idiom_hint.py pins that, because
# the moment a `reduce_and` row lands the first sentence here becomes false.
#
# THREE forms, and the ordering is measured, not stylistic. Measured on
# x declared (n,) float64, spelled EXACTLY as the string below spells them
# (the two-sided hinge and the one-sided count — a comment that paraphrases the
# string it annotates is a second copy that drifts, and this one had):
#
#                                     real          ieee
#   x >= lo                           decides       decides, every n
#   sum(max(lo-x,0)+max(x-hi,0))<=0   decides       ⊤ at reduce_sum for n >= 3
#   sum((x<lo).astype(int32)) == 0    decides       ⊤ at reduce_sum at EVERY n
#
# The ieee column is the whole reason this comment exists. `_ieee_reduce_sum`
# declines three or more float contributors (association freedom), and
# `_ieee_f64_only` declines the counting form's int64 accumulator at any size —
# so under semantics="ieee" the elementwise form is the ONLY one of the three
# that decides at every array size, and a hint that said "all three decide"
# handed a stuck reader a second dead end in the mode docs/preconditions.md
# tells them to use for float-boundary facts. The string says which.
#
# As an `assume` they differ again, and this is the part nobody would guess.
# Measured on x ∈ [-10, 10]^3, real mode, with the shipped spellings:
#
#   hinge  (two-sided)  CONSTRAINED, narrows var 7 -- the reduction's own
#   count  (one-sided)  CONSTRAINED, narrows var 5 -- intermediate, an
#                       over-approximated value, so each raises
#                       satisfiability-UNCERTIFIED: `assert_(sum(x) <= -100)`
#                       comes back `unknown` (withheld from REFUTED) and
#                       `assert_(sum(x) >= 0)` stays `unknown`
#   x >= 0              CONSTRAINED, narrows var 1 -- the DECLARED input --
#                       to [0, 10]^3, no UNCERTIFIED: `sum(x) <= -100` reaches
#                       `violated-over-set` and `sum(x) >= 0` `discharged`
#
# So the elementwise form is named FIRST on two independent measurements: it is
# the only one that survives ieee at every size, and it is the only one that
# leaves the REFUTED face reachable. The arithmetic pair stays because a genuine
# reduction over an array is not a pointwise fact and has nowhere else to go.
#
# THE ATTRIBUTION IS STRUCTURE, NOT PROSE. `_REAL_ONLY_MARKER` below is a
# sentinel the test splits this string on: every rewrite named AFTER it is
# claimed real-only, every rewrite named before it is claimed to hold in both
# modes, and tests/test_membership_idiom_hint.py requires that split to equal
# the measured one. Inverting the attribution — swapping which forms sit after
# the marker — fails there. Before this it did not: the test asserted only that
# the words "ieee" and "reduce_sum" appeared somewhere, so a text that blamed
# the wrong form passed. Move a rewrite across the marker only with a
# measurement in hand.
_REAL_ONLY_MARKER = "decide under semantics='real' but NOT under semantics='ieee':"

# The second sentinel, for the second claim a test would otherwise only check
# BEHAVIOURALLY: "delete the reduction" applied verbatim to a conjunction of
# differently-shaped arrays does not trace. Measured, `jnp.all(w >= 0) &
# jnp.all(b >= 0)` with w:(3,) and b:(5,) — the weights-and-biases shape, the
# most likely spelling a reader arrives with — raises `TypeError: and got
# incompatible shapes for broadcasting: (3,), (5,)` once the reductions are
# gone, while two calls decide in both semantics with no shape condition. A
# test that measured only the TypeError left the SENTENCE droppable; this is
# the string it must keep.
_SEPARATE_CALLS_MARKER = "Keep them separate calls rather than one `&`"
MEMBERSHIP_IDIOM_HINT = (
    " — `jnp.all(...)` lowers to `reduce_and`, which has no interval transfer "
    "in either registry (`reduce_or` has one in both, which is why `jnp.any` "
    "decides), so its box is ⊤. Usually the fix is to DELETE THE REDUCTION and "
    "state each conjunct as its OWN call: `assert_`, `assume` and `nonvacuity` "
    "judge an array predicate ELEMENTWISE, so the bare `k >= lo` over an "
    "n-element array already IS the conjunction over all n, and two calls are "
    "the conjunction of both. Keep them separate calls rather than one `&` — "
    "`jnp.all(w >= 0.0) & jnp.all(b >= 0.0)` over differently-shaped arrays "
    "stops broadcasting once the reductions are gone. Where the condition is "
    "genuinely a reduction, these decide under semantics='real' but NOT under "
    "semantics='ieee': `jnp.sum(jnp.maximum(lo - k, 0.0) + jnp.maximum(k - hi, "
    "0.0)) <= 0.0`, which falls to ⊤ at `reduce_sum` for three or more "
    "contributors, and `jnp.sum((k < lo).astype(jnp.int32)) == 0`, which falls "
    "there at every size because its accumulator is an integer. As an `assume` "
    "the three are not "
    "interchangeable either: all three CONSTRAIN rather than DROP, but the two "
    "arithmetic forms narrow the reduction's own intermediate — an "
    "over-approximated value — so the precondition is stamped "
    "satisfiability-UNCERTIFIED and every definite violation is then withheld "
    "from REFUTED unless a probed point of the declared set is found to "
    "satisfy every assume of the query, whereas the elementwise form narrows "
    "the compared value itself and stays certified where that value is a "
    "declared input"
)

# The hint is ~1.2k characters and one run can state a property on three faces.
# Printing it three times is the complaint `_note_decline`'s dedupe already
# exists for ("a ~120-word decline note printed verbatim twice was the measured
# complaint") — and verbatim dedupe cannot fire here, because each face names
# its own site. So the BODY is printed once per propagation and every later
# face gets its own locator plus this pointer. Deliberately order-free: notes
# ride into the verdict in emission order, but "the note above" would be a
# claim about a tuple index that no test pins.
MEMBERSHIP_IDIOM_POINTER = (
    " — same cause (`jnp.all(...)` → `reduce_and` → ⊤) and the same rewrites as "
    "the membership-idiom note printed elsewhere on this run; the text is "
    "printed once per run rather than once per face"
)

# What a `jnp.all` result can pass through and still BE that conjunction, for
# the gate below. `and` qualifies on the meaning: `assert_(a & b)` judged
# elementwise is exactly `all(a) & all(b)`, so deleting both reductions
# preserves the property. The rest are shape-only moves — `keepdims=True`
# lowers to a broadcast_in_dim/squeeze or reshape pair.
#
# `or` and `not` are deliberately ABSENT although they are boolean too:
# `all(a) | all(b)` is NOT elementwise `a | b`, and `~all(a)` is not `~a`, so
# "delete the reduction" would change the stated property. The hint would be
# wrong there, and no hint beats a wrong hint. `select_n` is absent for the
# same reason at one remove: in `jnp.where(jnp.all(...), a, b)` the reduction
# is a SELECTOR, not the judged property.
_MEMBERSHIP_HINT_TRANSPARENT = frozenset({
    "and", "reshape", "squeeze", "broadcast_in_dim", "convert_element_type",
})

# Node cap for both membership-hint walks. Message-content code runs only on
# undecided faces, but "only" is not "never" and an unbounded backward walk
# over a wide jaxpr is a hang. Exhausting the cap degrades to NO HINT, never to
# a hint asserted without the evidence (degrade-don't-hang, the same posture
# _integer_pow_budget takes).
_MEMBERSHIP_HINT_WALK_CAP = 512

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


DIV_STRADDLE_DECLINE = (
    "div: the divisor interval {divisor} straddles zero — real division is "
    "undefined at 0 and the quotient's image is unbounded over this box. "
    "Remedies: narrow the divisor's declared envelope to exclude zero, or "
    "add assume(divisor > 0) / assume(divisor < 0) before the division"
)

DIV_BOUNDARY_ZERO_DECLINE = (
    "div: the divisor interval {divisor} REACHES zero at a boundary, and "
    "nothing in this query excludes that point — real division is undefined "
    "at 0, so no bound on the quotient holds over the whole box. The "
    "boundary-aware tightening applies only when a strict assume certifies "
    "the divisor is nonzero: `assume(d > 0)` / `assume(d < 0)` on the "
    "divisor itself, or on a value the divisor is built from by * / neg abs "
    "square x**n sum dot, with nonzero finite constants allowed anywhere in "
    "that chain (a subtraction breaks it — two positives can differ by "
    "zero — and so does any other operation, sqrt and exp included). "
    "Remedies: narrow the divisor's declared envelope to exclude zero, or "
    "add that assume"
)

# The primitives :meth:`_Propagator._strict_sign_out` has a rule for. A
# by-NAME census, kept beside the rules it guards so a reader sees the
# closed set: anything absent DROPS the certificate, which is the
# conservative direction (a dropped fact can only turn a VERIFIED into an
# UNKNOWN).
_STRICT_SIGN_PRIMITIVES = frozenset({
    "mul", "div", "add", "add_any", "neg", "abs", "square", "integer_pow",
    "reduce_sum", "dot_general",
})

# Real-mode transfers that READ the strict-sign certificate, and therefore
# take it as a fourth positional argument. The dispatcher passes
# `tuple(sign per invar)` to exactly these, mirroring the `in_flags`
# argument every ieee transfer already receives. One member today; a
# registry rather than an `if` so the coupling is greppable from both ends.
_REAL_TRANSFERS_READING_STRICT_SIGN = frozenset({"div"})


def _box_strict_sign(box: iv.IntervalArray) -> int:
    """The strict sign of a box that IS a value: ``+1`` when every element
    is certainly positive, ``-1`` when every element is certainly negative,
    ``0`` otherwise.

    **The premise is that the box is the value**, which is true of a
    decoded LITERAL and of a decoded CONSTVAR and of nothing else in this
    module — a var's box is an over-approximation, and `assume(x > 0)`
    narrows to `[0, hi]`, which is exactly the box this function would call
    unsigned. Callers must establish that premise; the strict-sign table
    exists because a box in general cannot.

    Three conditions, each load-bearing:

    * **NON-EMPTY.** A size-0 value certifies nothing about "every
      element" — both quantifiers below are vacuously true over an empty
      box, which would mint a sign for a value that has none.
    * **FINITE, both endpoints.** ``±inf`` is nonzero but breaks the chain
      rules that consume this: ``a / inf = 0``, so a certificate minted off
      an infinite operand would claim NONZERO of a value that is zero. Not
      hypothetical plumbing — :func:`_int_bracket` saturates an int beyond
      the double range to ``(maxf, inf)``, and that is a literal.
    * **STRICTLY NONZERO, every element.** An ARRAY value must be all-one
      sign, not merely first-element signed, because the fact this mints is
      quantified over every element (see :meth:`_strict_sign_out`). A
      ``0`` fails both halves, which is what keeps the divisor case exactly
      as strong as it was: `div`'s boundary gate is consulted only when
      :func:`stelling.interval.straddles_zero` holds of the divisor, i.e.
      some element has ``lo <= 0 <= hi``, and a value answering ``+1`` here
      has ``lo > 0`` for EVERY element (``-1``: ``hi < 0``, so
      ``lo <= hi < 0``). Nothing that answers nonzero can straddle, so no
      constant divisor can reach the gate this widens. Enumerated in
      ``tests/test_assume_bump_boundary_div.py`` rather than asserted.
    """
    if box.size == 0:
        return 0
    if not all(
        math.isfinite(lo) and math.isfinite(hi)
        for lo, hi in zip(box.los, box.his)
    ):
        return 0
    if all(lo > 0.0 for lo in box.los):
        return 1
    if all(hi < 0.0 for hi in box.his):
        return -1
    return 0


def _literal_strict_sign(atom: ir.Literal) -> int:
    """The strict sign of a LITERAL, from its own decoded value.

    REAL MODE ONLY by construction — every call path is gated on
    `semantics != "ieee"`, and that gating is what makes reading the RAW
    decoded box correct here. **THIS SENTENCE READ "both call paths" AND
    NAMED TWO**; they were :meth:`_Propagator._strict_sign_out` and the
    `div` transfer's `in_signs`, and both are still gated exactly as it
    said. There are now THREE: the boundary carry
    (:meth:`_Propagator._carries_signs`) also reaches
    :meth:`_Propagator.read_strict_sign`, and a wrapper operand or a
    sub-jaxpr outvar can be a Literal, so an ungated third path would
    have written an ieee table off this raw box. Its gate is
    `_carries_signs`'s own `semantics != "ieee"` conjunct, which exists
    for exactly this reason and says so. Under ieee a literal is
    subnormal-hazed by :meth:`_Propagator.read` (DAZ flushes a literal like
    every other value), so a tiny positive literal's runtime value IS zero
    and this box would be lying; in real mode the literal's value simply is
    its box.

    A literal that does not DECODE keeps answering 0: the tree carries
    dtypes with no zero-dep decoder and an undecodable-literal NaN
    sentinel, guarded with the same idiom as :meth:`read_flag` and
    :meth:`_quiet_box`.
    """
    try:
        box = _value_to_interval(atom.val, atom.aval.shape, atom.aval.dtype)
    except (iv.IntervalError, ir.TranscriptionError):
        return 0
    return _box_strict_sign(box)


def _t_div(eqn, params, ins, in_signs=None):
    """``div``. On floats this is real division, unchanged. On INTEGERS it
    is not: jax integer division TRUNCATES toward zero (measured:
    ``lax.div(-7, 2) = -3``, not −3.5) and ``INT_MIN / -1`` WRAPS (measured:
    ``lax.div(-2**31, -1) = -2147483648``, not +2³¹). Modelling either as
    real division mints false definite verdicts in both directions — audit
    UNSOUND 3, and the second of those is literally the wraparound class
    the overflow guard exists for, so it routes through the same guard.

    **A divisor box that CONTAINS zero declines in all four of its shapes
    unless the zero is certified excluded** (audit 0.2.0 B5-1). The four
    shapes used to be judged by WHERE the zero sat — ``[0,0]``, a true
    straddle and a negative ``sqrt``-style domain all declined citing "ℝ
    has no value there", while ``[lo,0]``/``[0,hi]`` silently dropped the
    endpoint and minted a definite quotient from the rest. That fourth arm
    was a false VERIFIED: with `x` declared ``[0,2]``, ``1/(Σxᵢ² − 8)``
    boxed to ``(-inf, -0.125]`` and discharged ``q <= -0.125``, while jax
    at ``x = [2,2]`` — a point of the DECLARED box — returns ``+inf``.
    Nothing in the verdict disclosed that a point had been dropped.

    The honest condition is not which endpoint is zero but **whether zero
    is a value the divisor can take**, and the propagator answers that from
    ``in_signs``: a ``+1``/``-1`` there means a strict ``assume`` certifies
    the divisor is nonzero over the assumed region, so the closed box's
    zero endpoint is the unrepresentable-open-bound artifact
    :func:`stelling.interval.boundary_div` was written for and dropping it
    drops nothing. Without that certificate the endpoint is a real value of
    a real point and the transfer declines, exactly as the other three
    shapes do.

    ``in_signs is None`` — the shape every direct caller and every test
    that builds this transfer by hand takes — means "no certificate",
    the conservative reading.
    """
    dtype = (eqn.outvars[0].aval.dtype or "") if eqn.outvars else ""
    if not _is_integer_dtype(dtype):
        divisor = ins[1]
        if iv.straddles_zero(divisor):
            # Decompose straddle: point-at-zero, one-sided boundary, or
            # true straddle (lo < 0 < hi).
            has_point_zero = False
            has_true_straddle = False
            for lo, hi in zip(divisor.los, divisor.his):
                if lo <= 0.0 <= hi:
                    if lo == 0.0 and hi == 0.0:
                        has_point_zero = True
                    elif lo < 0.0 and hi > 0.0:
                        has_true_straddle = True
            if has_point_zero:
                # Division by literal zero: decline
                raise iv.IntervalError(
                    "div: the divisor interval contains the point [0, 0] "
                    "— division by zero is undefined"
                )
            if has_true_straddle:
                # True straddle: decline with the existing message
                for i, (lo, hi) in enumerate(
                    zip(divisor.los, divisor.his)
                ):
                    if lo < 0.0 and hi > 0.0:
                        span = f"[{lo}, {hi}]"
                        if divisor.size > 1:
                            span = f"element {i} spans {span}"
                        break
                raise iv.IntervalError(
                    DIV_STRADDLE_DECLINE.format(divisor=span)
                )
            # Every straddling element is a one-sided boundary. The zero is
            # dropped by `boundary_div` ONLY where a strict assume put it
            # there; otherwise it is a point of the box the program reaches
            # and ℝ has no quotient at it.
            if not (in_signs and len(in_signs) > 1 and in_signs[1]):
                for i, (lo, hi) in enumerate(zip(divisor.los, divisor.his)):
                    if lo <= 0.0 <= hi:
                        span = f"[{lo}, {hi}]"
                        if divisor.size > 1:
                            span = f"element {i} spans {span}"
                        break
                raise iv.IntervalError(
                    DIV_BOUNDARY_ZERO_DECLINE.format(divisor=span)
                )
            return [iv.boundary_div(ins[0], divisor)]
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
    _refuse_complex(eqn, "integer_pow")
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
    out-of-domain region, and the walk turns that
    :class:`interval.IntervalError` into a noted top-decline."""
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
    "not": (_t_not, TIER_EXACT),
    "reduce_or": (_t_reduce_or, TIER_EXACT),
    "is_finite": (_t_is_finite, TIER_EXACT),
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
    "square": (_t_square, TIER_SOUND),
    "copy": (_t_copy, TIER_EXACT),
    "unstack": (_t_unstack, TIER_EXACT),
    # sound, not exact: the box is the achievable hull under GRADUAL
    # underflow and a strict superset of it under the measured FLUSH, so it
    # cannot be exact under both — and it is registered for the weaker of
    # the two because the tier rides into the stamp. See _t_sign.
    "sign": (_t_sign, TIER_SOUND),
    # sound, not exact: |rem| <= min(|a|, |b|) bounds the box, but the
    # achievable image inside it is not swept — see _t_rem.
    "rem": (_t_rem, TIER_SOUND),
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
    # x[idx], leading-axis row form only — census addition from the MIME
    # fvm laplacian census trace, widened by the index-bounds round to an
    # index known only to a range; every other gather GEOMETRY declines
    # (see _t_gather).
    "gather": (_t_gather, TIER_EXACT),
    # u[i] with a traced i, and any out-of-range static index: the
    # index-bounds round. Dynamic start indices are propagated as intervals
    # and compared against the axis' legal start window; a value is
    # computed only where jax's clamp is provably the identity, so the
    # clamp is never modelled (see _t_dynamic_slice). Pure data movement:
    # tier exact.
    "dynamic_slice": (_t_dynamic_slice, TIER_EXACT),
    # the write sibling, same round, same three cases, same clamp posture
    "dynamic_update_slice": (_t_dynamic_update_slice, TIER_EXACT),
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
    # `square` multiplies its operand by itself, so it can produce a value the
    # operand did not contain -- the same class as `mul`, same guard
    "square",
    # `sign` maps its operand to -1/0/+1, values the operand need not have
    # contained, so it computes and is probed. Its arithmetic CANNOT escape
    # any integer range ({-1,0,1} fits every width down to int4/uint4), so
    # the probe passes structurally rather than by a guard — stated here
    # because a probe that cannot fail is not the load-bearing check for
    # this row. The load-bearing checks are the executed-value containment
    # tests in tests/test_sign_rem_rows.py and the sweep in the campaign repo
    # (Norm G: the instrument must reach the claim, and this probe reaches
    # only the range claim).
    "sign",
    # truncated remainder computes a value neither operand contained
    # (rem(7,3) = 1), so it is the same class. It cannot escape the range
    # either — |rem| < |b| and |rem| <= |a| — and MEASURED at the boundary
    # where its sibling does escape: rem(INT_MIN, -1) = 0 while
    # div(INT_MIN, -1) WRAPS. It routes through the shared guard anyway,
    # for the integer snap.
    "rem",
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
    # `copy` is the identity and `unstack` routes elements to outputs; neither
    # computes anything, so neither can introduce an out-of-range integer
    "copy",
    "unstack",
    # `split` is pure data movement: every output element IS an input element
    # at a static index, so it cannot introduce an out-of-range integer
    "split",
    "max", "min", "select_n",
    "lt", "gt", "le", "ge", "eq", "ne", "and", "or", "not", "reduce_or",
    "is_finite",
    "convert_element_type",
    "stop_gradient", "reshape", "squeeze", "slice", "scatter", "gather",
    "dynamic_slice", "dynamic_update_slice",
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
    "copy": (
        "the identity primitive: its output IS its input, so no arithmetic "
        "occurs and an in-range integer cannot leave the range by being copied"
    ),
    "unstack": (
        "routes each index along one axis to its own output; every output "
        "element IS an input element, so no arithmetic occurs"
    ),
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
    "not": (
        "bool-only by its own dtype guard (the bitwise integer form "
        "declines inside the transfer); Kleene NOT on {0, 1}"
    ),
    "reduce_or": (
        "bool-only by its own dtype guard; a three-valued OR-fold whose "
        "outputs are booleans"
    ),
    "is_finite": (
        "produces booleans; tests whether its operand is finite, cannot "
        "introduce an out-of-range integer value"
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
        "pure element routing (row take) — copies of in-range values, no "
        "arithmetic performed on them; a range-valued index only widens "
        "WHICH in-range values are copied, never computes a new one"
    ),
    "dynamic_slice": (
        "pure element routing (window take) — every output element IS an "
        "operand element, and the transfer computes a value only where the "
        "start index provably lands inside the legal window, so no index "
        "arithmetic (jax's clamp) is performed either"
    ),
    "dynamic_update_slice": (
        "element REPLACEMENT (the write form): the output holds only "
        "values its operand and update already contained, hulled where "
        "both are reachable — no arithmetic on any of them"
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
            "square": (hi, None),
            # sign/rem sit at the dtype boundary like the rest. The divisor
            # is `hi` rather than div's `-1`: -1 is UNREPRESENTABLE at every
            # unsigned dtype, so that cell corresponded to no executable
            # program (blinded audit). Both probes are satisfied
            # STRUCTURALLY -- neither row's arithmetic can leave the range --
            # so the load-bearing checks for them are the containment tests,
            # not this sweep. Named rather than left to be assumed.
            "sign": (lo, None),
            "rem": (lo, hi),
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
        "square": (lo, None),
        "sign": (hi, None),
        "rem": (hi, lo if lo != 0.0 else hi),
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


def _the_float_format(dtypes, what: str) -> tuple[str, tuple[int, int, int]]:
    """**The one place a set of dtypes becomes an ieee float format.**

    Returns ``(dtype_name, (p, emin, emax))``. Raises
    :class:`~stelling.interval.IntervalError` when there is no float dtype
    at all, when one is not a supported format (integer dtypes, the float8
    variants), or when two supported ones DISAGREE.

    The agreement check is not decoration and it is not optional. Sorting
    the float dtypes and taking ``[0]`` picks a format ALPHABETICALLY, and
    ``bfloat16 < float16 < float32 < float64``, so a `{bfloat16, float16}`
    pair silently resolves to bfloat16 — whose subnormal band is
    ``2**-126`` where float16 needs ``2**-14``, 112 decades too narrow, and
    the band is what keeps a verdict sound for a flushing target. Two
    call sites did the sort-and-take and only one did the check, so the
    comparison side judged mixtures with the wrong band while the
    arithmetic side declined them (audit 0.2.0 M13). One implementation
    now, so the two cannot drift again: jax promotes before computing, so
    a disagreement means hand-built or deserialized IR — which this
    module's own rules treat as in scope — and a mixed equation is
    DECLINED, in both faces, rather than judged with either operand's
    band.

    ``what`` names the caller in the refusal ("ieee arithmetic", "ieee
    comparison"), which is the only thing the two faces do differently.
    """
    float_dtypes = sorted(d for d in dtypes if "float" in (d or ""))
    if not float_dtypes:
        raise iv.IntervalError(
            f"{what} requires float operands; dtypes {sorted(dtypes)} "
            f"contain no supported float format — declined"
        )
    fmt = _FLOAT_FORMATS.get(float_dtypes[0])
    if fmt is None:
        raise iv.IntervalError(
            f"{what}: dtype {float_dtypes[0]!r} is not a supported "
            f"format (supported: {sorted(_FLOAT_FORMATS)}) — declined"
        )
    for d in float_dtypes[1:]:
        other = _FLOAT_FORMATS.get(d)
        if other is None:
            raise iv.IntervalError(
                f"{what}: dtype {d!r} is not a supported format "
                f"(supported: {sorted(_FLOAT_FORMATS)}) — declined"
            )
        if other != fmt:
            raise iv.IntervalError(
                f"{what}: mixed float formats "
                f"{float_dtypes} — declined"
            )
    return float_dtypes[0], fmt


def _ieee_get_format(eqn) -> tuple[int, int, int]:
    """Look up the float format for an equation's operand/result dtypes.

    All float operands and results of an arithmetic equation share a single
    format (JAX promotes to a common dtype before computing). Returns the
    format tuple ``(p, emin, emax)`` from :data:`_FLOAT_FORMATS`. Raises
    :class:`~stelling.interval.IntervalError` if any dtype is not a
    supported float format (e.g., integer dtypes, float8 variants), or if
    the dtypes disagree (a mixed-dtype equation that bypassed promotion).
    """
    return _ieee_get_dtype_format(eqn)[1]


def _ieee_get_dtype_format(eqn) -> tuple[str, tuple[int, int, int]]:
    """:func:`_ieee_get_format` plus the dtype NAME it resolved.

    The name is what the libm accuracy budget is keyed on — a budget is
    per ``(op, format)`` and the format tuple is not what a user writes.
    """
    dtypes = {v.aval.dtype for v in (*eqn.invars, *eqn.outvars)}
    # Non-float dtypes that are acceptable in some contexts (bool selectors
    # in select_n, int indices) are the caller's business to validate; here
    # we extract THE float format.
    return _the_float_format(dtypes, "ieee arithmetic")


def _ieee_float_dtype_or_none(eqn) -> str | None:
    """The equation's float dtype NAME, or None when there is not exactly
    one. Never raises: it feeds the propagator's record of which
    ``(op, format)`` budgets a run consumed, and a record is not worth a
    second failure path — a pair it cannot name simply is not recorded,
    and the transfer that just ran had already resolved the same dtype."""
    try:
        return _ieee_get_dtype_format(eqn)[0]
    except iv.IntervalError:
        return None


def _ieee_f64_only(eqn) -> None:
    """Legacy binary64 guard — now delegates to _ieee_get_format and
    rejects anything that isn't float64. Retained for call sites that
    genuinely need binary64 only (comparisons with per-dtype DAZ bands)."""
    fmt = _ieee_get_format(eqn)
    if fmt != _FLOAT_FORMATS["float64"]:
        float_dtypes = sorted(
            d for d in {v.aval.dtype for v in (*eqn.invars, *eqn.outvars)}
            if "float" in (d or "") and d != "float64"
        )
        raise iv.IntervalError(
            f"ieee endpoint arithmetic is binary64-only; operand/result "
            f"dtypes {float_dtypes} are not modeled (float32/float16 round "
            f"differently, integer arithmetic wraps) — declined"
        )


def _ieee_format_min_normal(fmt: tuple[int, int, int]) -> float:
    """The smallest positive normal for a format: 2**emin."""
    _, emin, _ = fmt
    return iv._MIN_NORMAL_FOR_EMIN.get(emin, 2.0**emin)


def _ieee_format_min_positive(fmt: tuple[int, int, int]) -> float:
    """The smallest positive value (subnormal) for a format: 2**(emin-p+1).

    In ieee mode, ``x > 0`` genuinely means ``x >= min_positive`` because
    there is no representable value between 0 and this threshold.
    """
    p, emin, _ = fmt
    return math.ldexp(1.0, emin - p + 1)


def _format_nextafter(k: float, direction: int, fmt: tuple[int, int, int]) -> float:
    """The smallest representable value in ``fmt`` strictly past ``k`` in
    ``direction`` (+1 up, -1 down).

    Strategy: step once in binary64 (always at least as fine as any
    supported format), then round to the target format's grid in the same
    direction. The result is always on the far side of k in the format,
    so the closed interval ``[result, hi]`` exactly represents the open
    ``(k, hi]`` restricted to the format's values.

    For ``k == 0`` upward this produces the format's smallest positive
    subnormal. For ``k == 0`` downward, the negative counterpart.
    """
    if not math.isfinite(k):
        return k  # nothing beyond +-inf
    # Step once in float64 (the finest grid we can represent)
    stepped = math.nextafter(k, math.inf if direction > 0 else -math.inf)
    # Round to the target format's grid in the same direction. This is
    # >= stepped for direction +1 (so >= the true next) and <= stepped
    # for direction -1 — always on the sound side.
    return _round_in_format(stepped, fmt, direction)


def _ieee_round_box(box: iv.IntervalArray, fmt: tuple[int, int, int]) -> iv.IntervalArray:
    """Round an interval box's endpoints OUTWARD to the target format's ULP
    grid. For float64, this is the identity (native endpoints ARE binary64).
    For narrower formats, the lo is rounded DOWN and hi is rounded UP in the
    target format, ensuring the interval still contains every value the
    program could compute.

    Infinite endpoints are preserved (they represent overflow to ±inf, which
    IS a value in every IEEE format).
    """
    if fmt == _FLOAT_FORMATS["float64"]:
        return box  # identity — native endpoints are already binary64
    los, his = [], []
    for lo, hi in zip(box.los, box.his):
        # Round lo DOWN (direction=-1) and hi UP (direction=+1)
        rlo = lo if lo == -math.inf else _round_in_format(lo, fmt, -1)
        rhi = hi if hi == math.inf else _round_in_format(hi, fmt, +1)
        los.append(rlo)
        his.append(rhi)
    return iv.IntervalArray(shape=box.shape, los=tuple(los), his=tuple(his))


# -- the declared libm accuracy budget: the widening math ---------------------
#
# **WHAT THIS EXISTS FOR.** Under ``semantics="ieee"`` a verdict is a claim
# about the float value THE PROGRAM COMPUTES. :func:`stelling.interval.exp`
# brackets CPython's ``math.exp`` — glibc on the host running the analysis —
# with a ±1-binary64-ulp bump, and :func:`stelling.interval.pow_` does the
# same around ``math.pow``. The program does not run glibc's ``exp``. It runs
# whatever XLA compiled for the device, and the bracket of one function is
# not a bracket of another (audit 0.2.0 S9 and S11; S11 reaches the released
# 0.1.0, where ``propagate(closed, semantics="ieee")`` was the door).
#
# **MEASURED** on jax 0.11.0 / jaxlib 0.11.0, CPU, x86_64 Linux, glibc 2.39,
# as |backend(x) − true(x)| in ulps of the target format (:data:`LIBM_MEASURED`
# carries each figure with the population it came from): exp in float32 is
# out by up to **5.51 ulps**, exhaustively over every argument whose result
# is normal and finite; exp in binary64 by up to **1.67 ulps** over three
# million samples; exp in bfloat16 is **correctly rounded** over every
# normal finite result, and exp in float16 misses correct rounding on 2 of
# 63,487 arguments by 3e-5 of an ulp — both narrow formats are evaluated in
# float32 and rounded, which is what makes them so close and what makes
# float16, with only 11 significand bits, occasionally round twice the
# wrong way.
#
# **WHY NOT A WIDER FIXED BRACKET.** The audit that found this proposed
# ±2 ulps, "enough to cover any faithfully-rounded implementation". At 5.51
# measured ulps this backend's float32 ``exp`` is not faithfully rounded AT
# ALL — it is five and a half times worse than faithful — so no fixed
# number is sound: the quantity is a property of a compiled function
# stelling cannot see. Nor is one number right across formats. The SAME
# backend, on the SAME op, is 0.500028 ulps out in float16 and 5.5112 out
# in float32: a factor of ELEVEN between two formats, which is why the
# budget is per ``(op, format)`` and never extrapolated.
#
# **SO: FAIL CLOSED, AND OPEN IT WITH A DECLARATION.** A transfer whose
# backend accuracy stelling cannot establish declines, carrying the evidence
# and the exact incantation. The caller re-enables it by DECLARING a budget,
# which makes the assumption a deliberate engineering choice with a name and
# a date on it rather than a silent default — and the stamp then says, in
# those words, that the accuracy was declared and not verified.
#
# **REAL MODE IS UNTOUCHED.** There the bracket is about the true real value,
# CPython's own ``math`` module does satisfy the ±1-ulp assumption it is
# built on, and the divergence from XLA is the ℝ-versus-float gap the stamp
# already names. A budget passed under ``semantics="real"`` is REFUSED at
# entry rather than silently ignored.
#
# The class, the shipped profiles and the resolver live further down, after
# `IEEE_TRANSFERS`, because the set of ops a budget may name is derived from
# the registry's own tiers rather than hand-written beside it.

_DEFAULT_LIBM_PROFILE = "xla-cpu-2026-08"


def _libm_ulp_at(x: float, fmt: tuple[int, int, int]) -> float:
    """The format's ulp at ``|x|``: the spacing of the binade containing it,
    floored at the format's smallest subnormal.

    The binade convention — ``ulp(2**e) = 2**(e-p+1)``, the spacing ABOVE a
    power of two — is the one the sweeps behind :data:`LIBM_MEASURED` used,
    so a declared budget means the same thing on both sides of the
    comparison. Getting those two conventions out of step would make every
    number in the profile mean something slightly different from what it
    measured.
    """
    p, emin, _emax = fmt
    tiny = math.ldexp(1.0, emin - p + 1)
    if x == 0.0 or not math.isfinite(x):
        return tiny
    _m, e = math.frexp(abs(x))  # |x| = m * 2**e with 0.5 <= m < 1
    return max(math.ldexp(1.0, e - p), tiny)


def _libm_widen_box(
    box: iv.IntervalArray,
    fmt: tuple[int, int, int],
    ulps: float,
    *,
    floor: float | None,
) -> iv.IntervalArray:
    """Widen a libm bracket to admit a backend within ``ulps`` format ulps
    of the true real value.

    ``box`` must already bracket the TRUE REAL image of the argument box —
    which is what ``iv.exp`` and ``iv.pow_`` produce, under their own
    assumption that the host's ``math`` module is faithfully rounded. The
    declared claim is ``|backend(x) − true(x)| ≤ ulps · ulp_fmt(true(x))``,
    so what the widened box owes is ``lo_out ≤ t − u·ulp(t)`` and
    ``hi_out ≥ t + u·ulp(t)`` for EVERY ``t`` in the box.

    **ONE spacing serves both endpoints, and it is the LARGEST the box
    reaches — widening each endpoint by its own ulp is NOT enough.** ``ulp``
    is a step function of the magnitude: it doubles at every binade
    boundary, so ``t ↦ t − u·ulp(t)`` is *not* monotone — it drops by
    ``u·ulp/2`` each time ``t`` crosses a power of two upward. A box
    straddling ``2**k`` from below therefore admits values as low as
    ``2**k − u·2**(k-p+1)`` while ``ulp(lo)`` is only ``2**(k-p)``, half of
    what the widening needs. Taking ``U = max(ulp(lo), ulp(hi))`` restores
    the property for every ``t``: ``t ≥ lo`` and ``ulp(t) ≤ U`` give
    ``t − u·ulp(t) ≥ lo − u·U``, and symmetrically above. Each endpoint
    then takes an outward binary64 bump, paying for the arithmetic's own
    rounding. A FINITE box needs nothing beyond that maximum, for any
    ``u``: ``ulp`` is monotone in ``|t|``, so ``max(ulp(lo), ulp(hi))``
    dominates ``ulp(t)`` everywhere between — including across zero.

    **AN INFINITE ENDPOINT DOUBLES THAT SPACING; DROPPING IT FROM THE
    MAXIMUM IS HOW THIS WAS WRONG THE FIRST TIME.** ``iv.exp`` returns
    ``hi = +inf`` whenever ``math.exp`` overflows binary64, so an ordinary
    float32 envelope reaching past 709.78 arrives here half-infinite.
    Taking ``U`` over the FINITE endpoints alone then falls back to
    ``ulp(lo)`` while the box still holds every ``t ≥ lo``, in binades
    whose spacing is 2×, 4×, … larger: exactly the class the paragraph
    above is about, surviving on the arm it did not cover (audit 0.2.0
    B4). ``U = 2·ulp(lo)`` is enough, and is exactly enough. Enough:
    write ``g(t) = t − u·ulp(t)``; for ``t ≥ lo`` either
    ``ulp(t) ≤ 2·ulp(lo)``, giving ``g(t) ≥ lo − 2u·ulp(lo)`` directly, or
    ``ulp(t) > 2·ulp(lo)``, and then ``ulp(t)`` — a power of two — is at
    least ``4·ulp(lo)`` and is not the subnormal floor, so
    ``t ≥ 2**(p-1)·ulp(t)`` and
    ``g(t) ≥ ulp(t)·(2**(p-1) − u) ≥ 4·ulp(lo)·(2**(p-1) − u)``, while
    ``lo < 2**p·ulp(lo)`` gives ``lo − 2u·ulp(lo) < 2·ulp(lo)·(2**(p-1) −
    u)``. Exactly enough: ``g(2**k) = 2**k·(1 − u·2**(1-p))``, the first
    binade boundary above ``lo`` has ``ulp = 2·ulp(lo)`` and is where
    ``g`` dips lowest, and for ``u < 2**(p-1)`` every boundary after it
    dips less.

    **AND THAT SIDE CONDITION IS REACHABLE — BY ROUNDING UP, NOT BY
    MEASURING.** Past ``u = 2**(p-1)`` ulps — 1024 for float16, **128 for
    bfloat16**, 8388608 for float32 — ``1 − u·2**(1-p)`` turns
    non-positive, ``g(2**k) → −∞``, and the infimum over a half-infinite
    box is UNBOUNDED: no finite lower endpoint is sound, and the doubled
    ``U`` would mint one. That arm returns ``-inf`` (then ``floor``),
    which is the true bound rather than a giving-up.

    **A MEASUREMENT CANNOT REACH THAT THRESHOLD**, and which way round
    that goes is worth stating rather than waving at. The worst error this
    module has a name for — a subnormal result FLUSHED to zero — is
    ``t/tiny`` for a true value ``t`` under ``2**emin``, because
    :func:`_libm_ulp_at` floors at ``tiny``. The largest representable
    subnormal is ``tiny·(2**(p-1) − 1)``, so a flush measures at most
    ``2**(p-1) − 1`` ulps: exactly ONE ulp under the threshold, in every
    format — 1023 vs 1024, 127 vs 128, 8388607 vs 8388608,
    4503599627370495 vs 4503599627370496. For a true value anywhere in the
    open subnormal band the bound is ``< 2**(p-1)``, a supremum that is
    not attained. **The threshold therefore sits just ABOVE anything that
    can be observed**, and this backend's bfloat16 ``exp`` flush reaches
    108.7 — under 127 and so under 128. What crosses the threshold is a
    caller ROUNDING a measurement up, which is real and likely: 108.7
    rounds to 128 as readily as to 109, and rounding up is this profile's
    own stated convention. The side condition is necessary either way —
    the extremiser sweep's control C fails 1,538 obligations without it —
    but the route to it is a DECLARATION, not an observation.

    The upper side is the mirror image under ``t ↦ −t``, ``ulp`` being
    even: ``lo = -inf`` needs ``U = 2·ulp(hi)`` and saturates to ``+inf``
    past the same threshold. Nothing in this module produces ``lo = -inf``
    today — ``exp`` and ``pow`` have range in ``[0, ∞)`` and pass
    ``floor=0.0`` — so that arm is latent, and it is closed here rather
    than left for whichever transfer makes it reachable. ``[-inf, +inf]``
    is returned unchanged: it already admits every value.

    **``ulps ≤ 0.5`` widens by NOTHING, and what makes that sound is
    CORRECT ROUNDING — not "≤ 0.5 ulps" read through the binade convention
    above.** Those are not the same statement and they come apart at
    exactly one place: powers of two. ``ulp(2**k)`` here is the spacing
    ABOVE ``2**k``, while the float BELOW it is only ``2**(k-p)`` away —
    half an ulp by this convention — so the LITERAL reading of a ``u =
    0.5`` declaration also admits a backend returning ``nextdown(2**k)``
    where the true value is ``2**k``, which correct rounding does not.
    ``exp(0) = 1.0`` reaches it. So the branch is read as the declaration
    it is meant to be: :class:`LibmBudget` defines ``0.5`` as "correctly
    rounded", and for a correctly-rounded backend the no-op IS a theorem —
    the result is ``RN_fmt(t)``, round-to-nearest is monotone, and the
    caller rounds the box outward onto the format's grid immediately
    after, so ``RN_fmt(t)`` is already inside ``[floor_fmt(lo),
    ceil_fmt(hi)]`` for every ``t`` in the box. This is
    :func:`stelling.interval.sqrt`'s own argument, generalised: sqrt
    carries no libm demotion precisely because it is correctly rounded,
    and the mechanism here must not punish a platform that has that
    property for ``exp`` too.

    The residual — the literal reading's ``nextdown(2**k)`` — is covered
    for THIS module's callers, and only by them: ``iv.exp`` and
    ``iv.pow_`` hand over a box already bumped a binary64 ulp OUTWARD, so
    ``floor_fmt(lo)`` sits at least one format step below any grid-point
    true value, which is exactly where ``nextdown(2**k)`` is. A future
    caller passing a box that TOUCHES the true value may not read ``u =
    0.5`` as free. A budget of 1 ulp — merely FAITHFUL — does cost, and
    pays for the case where the true value sits exactly on a grid point
    and the backend lands one step beyond it.

    ``floor`` clamps the widened lower endpoint. ``exp``, and ``pow`` over a
    strictly positive base, have range in ``[0, ∞)``, so no backend value
    can be negative and ``floor=0.0`` states the RANGE rather than
    narrowing anything (``iv.exp`` already floors there). It has no default:
    a future transfer whose range is not half-open must say so out loud.
    """
    if ulps <= 0.5:
        return box
    p = fmt[0]
    # past 2**(p-1) declared ulps the infimum of t - u*ulp(t) over a
    # half-infinite box is unbounded — see the docstring's side condition
    unbounded = ulps > math.ldexp(1.0, p - 1)
    los, his = [], []
    for lo, hi in zip(box.los, box.his):
        lo_finite, hi_finite = math.isfinite(lo), math.isfinite(hi)
        if not (lo_finite or hi_finite):  # [-inf, inf]: nothing to widen
            los.append(lo)
            his.append(hi)
            continue
        spacing = max(
            _libm_ulp_at(v, fmt) for v in (lo, hi) if math.isfinite(v)
        )
        if not (lo_finite and hi_finite):
            # the first binade boundary past the finite end doubles the
            # spacing, and for u <= 2**(p-1) that boundary is the infimum
            spacing *= 2.0
        w = ulps * spacing
        if lo_finite:
            lo = (
                -math.inf if (unbounded and not hi_finite)
                else math.nextafter(lo - w, -math.inf)
            )
            if floor is not None and lo < floor:
                lo = floor
        if hi_finite:
            hi = (
                math.inf if (unbounded and not lo_finite)
                else math.nextafter(hi + w, math.inf)
            )
        los.append(lo)
        his.append(hi)
    return iv.IntervalArray(shape=box.shape, los=tuple(los), his=tuple(his))


# Every figure here is from a run recorded in this campaign's scratch and is
# regenerable: enumerate the format's bit patterns, evaluate the op through
# jax, compare against a higher-precision reference. A figure a reader can
# regenerate is worth more than a larger one they cannot, so these are the
# measured maxima and :data:`XLA_CPU_2026_08` rounds each UP to its budget.
#
# **THE SAMPLED ROWS NAME THEIR DRAW, because a sampled maximum is not a
# property of the backend — it is a property of the sample, and a reader who
# re-runs and gets a different number must be able to tell "the row is stale"
# from "your draw differs".** Two campaigns measured these:
#
#   DRAW A (c322cec) — the first campaign. Its regions are described below,
#     but ITS SEED WAS NOT RECORDED, so draw A cannot be re-run. Where it is
#     the larger of the two it is still what the row carries, because the
#     budget must clear anything either draw saw; it is flagged as
#     unreproducible each time.
#   DRAW B (B4, 2026-08-15) — ``numpy.random.default_rng(20260815)``,
#     re-measured from scratch for the amendment and re-run to confirm the
#     seed reproduces it.
#
# Draw B's sampled designs, in full, because "four regions" is not a design:
#
#   exp@float64 — 3,000,000 arguments: 1,000,000 uniform on each of
#     [-708, 709], [-40, 40], [-1, 1], drawn in that order.
#   pow@* — n pairs, n/4 per region, drawn in this order, each cast to the
#     target format, keeping the pairs whose true result is normal and
#     finite:
#       1. broad            base = 2**U(-20, 20),   exponent = U(-8, 8)
#       2. under overflow   base = 2**U(0.5, 12),
#                           exponent = U(0.80, 0.999)*emax / log2(base)
#       3. bases near 1     base = 1 + U(-0.05, 0.05),
#                           exponent = U(-400, 400)
#       4. powers of two    base = 2**k, k ~ randint(-12, 13) (0 mapped to 1),
#                           exponent = U(-emax/12, emax/12)
#
# References: binary64 for the three narrow formats; a 60-decimal-digit
# ``decimal`` context (``prec=70``) for binary64 targets.
LIBM_MEASURED: dict[tuple[str, str], str] = {
    ("exp", "float16"): (
        "EXHAUSTIVE over all 63,487 distinct finite float16 arguments, "
        "over the 37,479 of them whose result is normal and finite: max "
        "error 0.500028 ulps, and 2 arguments exceed 0.5 — NOT correctly "
        "rounded, though it misses by 3e-5 of an ulp. The backend "
        "evaluates float16 exp in float32 and rounds twice; at "
        "x=0.0226898193359375 and x=0.007297515869140625 the true value "
        "sits a hair BELOW the float16 midpoint and the float32 "
        "intermediate lands above it, so the second rounding goes to the "
        "far neighbour. Over the 6,392 arguments with a subnormal result: "
        "0.499849, none above 0.5"
    ),
    ("exp", "bfloat16"): (
        "EXHAUSTIVE over all 65,279 distinct finite bfloat16 arguments, "
        "over the 34,145 of them whose result is normal and finite: max "
        "error 0.499988 ulps and none above it — CORRECTLY ROUNDED there, "
        "the backend evaluating bfloat16 exp in float32 and rounding, and "
        "8 significand bits leaving room the 11 of float16 do not. THE "
        "QUALIFIER IS LOAD-BEARING: over ALL finite arguments the maximum "
        "is 108.698176 ulps and 11 exceed 0.5, every one of them a "
        "subnormal result FLUSHED to zero (worst x=-87.5, true "
        "9.982350930569248e-39, backend 0.0). That flush is covered by "
        "interval.subnormal_haze_fmt, which hulls the box with 0 whenever "
        "it reaches the format's subnormal band — not by this budget"
    ),
    ("exp", "float32"): (
        "EXHAUSTIVE over every float32 argument in [-104, 88.73] whose "
        "result is normal and finite (2,237,668,967 of them, which is "
        "exactly the arguments in [-87.33654022216797, "
        "88.72283172607422]): max error 5.5112 ulps, and 12,542 arguments "
        "exceed 1 ulp. They concentrate in [88.54634857177734, "
        "88.72283172607422] — 12,520 of the 12,542, and that band holds "
        "exactly 23,133 float32 values, so 54.12% of the arguments there "
        "escape. XLA's exp is not faithfully rounded at all in that band, "
        "and every escape is on the low side (12,542 low, 0 high)"
    ),
    ("exp", "float64"): (
        "3,000,000 sampled arguments — a million uniform on each of "
        "[-708,709], [-40,40] and [-1,1], numpy default_rng(20260815) — "
        "against a 60-digit decimal reference: max error 1.6660 ulps, with "
        "10,559 (0.35%) above 1 ulp, so XLA's binary64 exp is not "
        "faithfully rounded either and a 1-ulp bracket around glibc's "
        "leaks in both directions. SAMPLED, NOT EXHAUSTIVE, and this row "
        "is the demonstration of what that costs: draw A, an earlier draw "
        "of the same size whose SEED WAS NOT RECORDED, reached 1.6470 and "
        "this one beat it. Here the reproducible draw is also the larger, "
        "so the row carries it. A sampled row bounds what was sampled and "
        "nothing more, which is why the declared budget rounds up to the "
        "next integer rather than to the figure above"
    ),
    ("pow", "float16"): (
        "TWO INDEPENDENT DRAWS and the row carries the larger. Draw B "
        "(seed 20260815, re-run to confirm the seed reproduces it exactly): "
        "16,000,000 sampled (base, exponent) pairs over the four regions "
        "above, 12,642,619 of them with a normal finite result, reaching "
        "0.5001 with 509 above half an ulp. Draw A (the first campaign, "
        "16,000,000 pairs, SEED NOT RECORDED so it cannot be re-run): also "
        "0.5001. The two agree. max error 0.5001 ulps, none above 1 ulp"
    ),
    ("pow", "bfloat16"): (
        "Draw B (seed 20260815, reproduced): 16,000,000 sampled pairs over "
        "the same four regions, 15,907,789 with a normal finite result, "
        "reaching 0.5000 with 74 above half an ulp. Draw A (first "
        "campaign, 16,000,000 pairs, SEED NOT RECORDED): also 0.5000. The "
        "two agree. max error 0.5000 ulps, none above 1 ulp"
    ),
    ("pow", "float32"): (
        "THE TWO DRAWS DISAGREE AND THE ROW CARRIES THE LARGER, WHICH IS "
        "THE ONE THAT CANNOT BE RE-RUN. Draw A (first campaign, "
        "16,000,000 sampled pairs over the same four regions, SEED NOT "
        "RECORDED) reached 0.5380. Draw B (seed 20260815, reproduced): "
        "16,000,000 pairs, 15,907,360 with a normal finite result, only "
        "0.5290, with 24,511 above half an ulp. A budget must clear "
        "anything either draw saw, so the row states A's figure and the "
        "declared 1.0 covers both with room. max error 0.5380 ulps, none "
        "above 1 ulp in either draw — pow does not share exp's "
        "overflow-band path"
    ),
    ("pow", "float64"): (
        "The two draws disagree in the fourth decimal and the row carries "
        "the larger, which again is the one that cannot be re-run. Draw A "
        "(first campaign, 1,045,976 sampled pairs, 60-digit decimal "
        "reference, SEED NOT RECORDED) reached 0.5059. Draw B (seed 20260815, "
        "reproduced): 1,000,000 pairs against a 60-digit reference, 999,989 "
        "with a normal finite result, reached 0.5056 with 853 above half "
        "an ulp. max error 0.5059 ulps, none above 1 ulp in either draw"
    ),
}

# THE DECLINE. It is the feature, not the error: a halt a reader cannot act
# on is an obstacle to be worked around by trial and error, and what makes
# this one a deliberate choice instead is that it carries the evidence that
# justifies it and a line that RUNS AS WRITTEN.
LIBM_BUDGET_DECLINE = (
    "{op} under semantics='ieee' has no DECLARED accuracy budget for "
    "{dtype} — declined rather than judged against an assumption stelling "
    "cannot check. "
    "WHY: the bracket here is built from CPython's math.{op} (glibc on the "
    "host running the analysis) bumped 1 ulp outward. Under ieee semantics "
    "the verdict is a claim about the float value YOUR PROGRAM computes, "
    "and your program runs the {op} the compiler emitted for your device — "
    "a different function, which stelling cannot see, execute or measure. "
    "MEASURED for ({op}, {dtype}) on jax 0.11.0 / jaxlib 0.11.0, CPU, "
    "x86_64: {evidence}. "
    "A FIXED WIDER BRACKET IS NOT THE FIX: the error is a property of a "
    "compiled function, and on this very backend the same op measures "
    "0.500028 ulps in float16 and 5.5112 in float32 — a factor of eleven "
    "between two formats, which no single number spans. "
    "{incantation}. "
    "The budget is DECLARED, NEVER VERIFIED: one smaller than your "
    "backend's real error mints a VERIFIED nothing here can catch, and the "
    "verdict's stamp says exactly that."
)


def _libm_decline_incantation(
    op: str, dtype: str, profile: str, ship_u: float | None
) -> str:
    """The decline's TO PROCEED sentence, built so that what it prints
    RUNS AS WRITTEN.

    **A template is not an incantation.** The first draft printed
    ``check(harness, vacuity_mode=..., semantics='ieee', ...)`` — where
    ``...`` is ``Ellipsis`` and raises *"widen mode must be one of ('all',
    'inputs-only')"* — and ``ulps={('exp','float32'): <ulps>}``, which is a
    ``SyntaxError``. A reader who pastes what a halt told them to paste and
    gets a SECOND error learns that the halt was decoration. So the shipped
    number goes in literally, ``vacuity_mode`` gets a real value, and the
    only thing left for the reader to supply is their own harness.

    It also names BOTH doors. The decline used to mention only ``check``,
    while the exposure this whole gate exists to close (audit 0.2.0 S11)
    runs through :func:`propagate` — which takes the same keyword and is
    the entry point the released 0.1.0 reached ieee mode through.
    """
    number = ("%g" % ship_u) if ship_u is not None else "<the ulps you measured>"
    own = (
        "`from stelling.propagate import LibmBudget` then "
        "`check(harness, vacuity_mode='inputs-only', semantics='ieee', "
        "libm_budget=LibmBudget(name='my-backend-2026-08', "
        "basis='what you measured, on what, and when', "
        "ulps={%r: %s}))`" % ((op, dtype), number)
    )
    if ship_u is None:
        return (
            "TO PROCEED, declare what you are willing to assume about your "
            "backend. The shipped profile %r does NOT cover (%s, %s), so "
            "there is no number here to borrow: measure yours and write it "
            "where the placeholder is — %s" % (profile, op, dtype, own)
        )
    return (
        "TO PROCEED, declare what you are willing to assume about your "
        "backend. EITHER OF THESE LINES RUNS AS WRITTEN once the first "
        "argument is yours: `check(harness, vacuity_mode='inputs-only', "
        "semantics='ieee', libm_budget=%r)` or `propagate(closed, "
        "semantics='ieee', libm_budget=%r)`. Both declare %s@%s <= %s ulps "
        "on the measurement above; both entry points take the keyword, and "
        "`propagate` is the one the released 0.1.0 reached ieee mode "
        "through. To declare your OWN number instead of borrowing that "
        "one, %s, with your measurement in place of the %s"
        % (profile, profile, op, dtype, number, own, number)
    )

LIBM_BUDGET_REAL_MODE_REFUSAL = (
    "a libm accuracy budget has no meaning under semantics='real' and is "
    "refused rather than silently ignored: there the bracket is about the "
    "TRUE REAL value, the ±1-ulp assumption it rides on is about CPython's "
    "own math module — which satisfies it — and the divergence from what "
    "your program computes is the ℝ-versus-float gap the stamp already "
    "names. No accuracy budget closes that gap. Drop libm_budget, or pass "
    "semantics='ieee'."
)


def _libm_budget_ulps(op: str, dtype: str, budget) -> float:
    """The declared budget for ``(op, dtype)``, or the decline.

    **THE CHOKE POINT.** Every ieee transfer that rides a libm accuracy
    claim comes through here, and there is no path past it that does not
    carry a number the caller wrote down.
    """
    if budget is not None:
        got = budget.get(op, dtype)
        if got is not None:
            return got
    shipped = LIBM_PROFILES.get(_DEFAULT_LIBM_PROFILE)
    ship_u = shipped.get(op, dtype) if shipped is not None else None
    covered = (
        ", ".join(f"{o}@{d}" for (o, d), _u in shipped.ulps)
        if shipped is not None else "nothing"
    )
    evidence = LIBM_MEASURED.get(
        (op, dtype),
        f"stelling ships no measurement for this pair — the shipped "
        f"profile covers {covered}",
    )
    raise iv.IntervalError(
        LIBM_BUDGET_DECLINE.format(
            op=op,
            dtype=dtype,
            evidence=evidence,
            incantation=_libm_decline_incantation(
                op, dtype, _DEFAULT_LIBM_PROFILE, ship_u
            ),
        )
    )


def _ieee_arith(op):
    """The monotone core (add/sub/mul/div): native binary64 corner
    endpoints, NaN corners routed to the flag (never into an interval).
    A maybe-NaN operand poisons the result (NaN propagates through all
    four ops), so operand flags OR into the output flag.

    Format-parametric: for non-float64 formats, uses the format's own
    subnormal band for the DAZ/FTZ haze, and rounds the result endpoints
    outward to the format's ULP grid.

    **A kernel with no :data:`_FMT_BINARY_OPS` row DECLINES on a narrow
    format** — it does not fall back to the binary64 kernel. The fallback
    that used to stand here hazed with ``iv.MIN_NORMAL`` (``2**-1022``)
    and `_ieee_round_box` afterwards CANNOT recover the missing haze:
    outward rounding onto the format grid does not hull with 0. Measured
    on float32 ``x + x`` at ``x = 2**-140`` the fallback box was
    ``[1.4349e-42, 1.4349e-42]`` where jax computes ``0.0`` — a box
    excluding the executed value, i.e. a false VERIFIED in waiting (audit
    0.2.0 M15). It was unreachable when it was written and the hazard was
    that the NEXT binary kernel would make it reachable silently. Two
    guards, because one of them fails before anything runs:
    :func:`_assert_ieee_binary_kernels_are_format_parametric` refuses at
    IMPORT if a registered ieee binary transfer has no row, and this arm
    declines at RUN if one somehow gets past it."""

    def t(eqn, params, ins, flags):
        fmt = _ieee_get_format(eqn)
        min_normal = _ieee_format_min_normal(fmt)
        if fmt == _FLOAT_FORMATS["float64"]:
            box, made_nan = op(ins[0], ins[1])
        else:
            # Use the format-parametric binary kernel with format's band
            op_fmt = _FMT_BINARY_OPS.get(op)
            if op_fmt is None:
                raise iv.IntervalError(
                    f"ieee {eqn.primitive!r} has no format-parametric kernel "
                    f"for {_FLOAT_FORMATS_BY_TUPLE.get(fmt, fmt)}: the "
                    f"binary64 kernel hazes with the binary64 subnormal band "
                    f"(2**-1022) and outward rounding onto the format grid "
                    f"cannot add the hull-with-0 this format's own band "
                    f"(2**{fmt[1]}) requires — declined rather than judged "
                    f"with the wrong band"
                )
            box, made_nan = op_fmt(ins[0], ins[1], min_normal)
            box = _ieee_round_box(box, fmt)
        return [box], [made_nan or any(flags)]

    # The census hook: the closure hides `op`, so the import-time check
    # below could not otherwise see WHICH kernel this transfer runs.
    t._ieee_binary_kernel = op
    return t


def _ieee_div(eqn, params, ins, flags):
    """``div`` under ieee: a zero-containing divisor DECLINES, with the
    reason quoted, rather than returning ⊤ silently.

    The kernel's answer is unchanged and correct — ⊤ is the exact hull, for
    the reasons :data:`stelling.interval.IEEE_ZERO_DIVISOR_TOP` gives — but
    a ⊤ returned as an ordinary result is counted KNOWN, and the undecided
    note then told the reader "none fell to ⊤ … compatible with a precision
    near-miss" about a `[-inf, +inf]` box, while the stamp's
    ``top_despite_coverage`` line correctly named `div ×1` in the same
    verdict. One of those two sentences had to go, and the honest one to
    keep is the stamp's: the transfer has no informative answer here, which
    is what a decline says.

    It also puts the constant in front of the person the CHANGELOG sends
    to it. `IEEE_ZERO_DIVISOR_TOP` was reachable only from docstrings while
    the release note said "``interval.IEEE_ZERO_DIVISOR_TOP`` says why".

    **The divisor is hazed FIRST.** DAZ flushes a subnormal operand to
    zero, so a box that clears zero on paper can contain it as executed —
    which is the third and least obvious of the three shapes that reach
    this rule (audit 0.2.0 S10). `_ieee_binary`/`_ieee_binary_fmt` apply
    exactly this haze to the operands before the corner rule, so testing
    the hazed box is testing what the kernel would have seen.
    """
    fmt = _ieee_get_format(eqn)
    hazed, _ = iv.subnormal_haze_fmt(ins[1], _ieee_format_min_normal(fmt))
    if iv.straddles_zero(hazed):
        span = None
        for i, (lo, hi) in enumerate(zip(hazed.los, hazed.his)):
            if lo <= 0.0 <= hi:
                span = f"[{lo}, {hi}]"
                if hazed.size > 1:
                    span = f"element {i} spans {span}"
                break
        raise iv.IntervalError(
            f"div: the divisor interval {span} contains zero (after the "
            f"subnormal haze). {iv.IEEE_ZERO_DIVISOR_TOP}"
        )
    return _IEEE_DIV_ARITH(eqn, params, ins, flags)


# Mapping from the binary64-native ieee kernel to its format-parametric
# counterpart. Used by _ieee_arith to dispatch the correct kernel when
# the equation's format is not float64.
_FMT_BINARY_OPS: dict = {
    iv.ieee_add: iv.ieee_add_fmt,
    iv.ieee_sub: iv.ieee_sub_fmt,
    iv.ieee_mul: iv.ieee_mul_fmt,
    iv.ieee_div: iv.ieee_div_fmt,
}

# the plain arithmetic face `_ieee_div` delegates to once its zero-divisor
# gate has passed (built here, after _FMT_BINARY_OPS, so the format
# dispatch inside it is populated)
_IEEE_DIV_ARITH = _ieee_arith(iv.ieee_div)
# `div` reaches the binary core through a wrapper, so the wrapper has to
# carry the census hook too — otherwise the import-time check below reads
# `div` as "not a binary transfer" and stops covering the one row whose
# kernel is hardest to see (audit 0.2.0 M15).
_ieee_div._ieee_binary_kernel = _IEEE_DIV_ARITH._ieee_binary_kernel


def _ieee_unary_exact(fn):
    """neg/abs: exact sign arithmetic — the float result IS the real
    result, no rounding, no new NaN (neg(nan)/abs(nan) stay NaN: flag
    propagates). Format-parametric: uses the format's own subnormal band."""

    def t(eqn, params, ins, flags):
        fmt = _ieee_get_format(eqn)
        min_normal = _ieee_format_min_normal(fmt)
        # subnormal haze on the result: whether a sign operation is
        # flushed is target-dependent; the hull with 0 covers both
        return [iv.subnormal_haze_fmt(fn(ins[0]), min_normal)[0]], [flags[0]]

    return t


def _ieee_minmax(fn):
    """max/min: float max/min of non-NaN operands is exact (no rounding),
    so the real transfer's endpoint rule is float-exact when neither
    operand is maybe-NaN. With a maybe-NaN operand the result is the
    other operand, the extremum, or NaN depending on the backend's NaN
    ordering (measured on jax 0.11.0 cpu: lax.max/min PROPAGATE NaN in
    both operand orders) — the operand hull covers every one of those
    non-NaN outcomes, so hull + flag is sound without leaning on the
    measurement. Format-parametric: uses the format's own subnormal band."""

    def t(eqn, params, ins, flags):
        fmt = _ieee_get_format(eqn)
        min_normal = _ieee_format_min_normal(fmt)
        a, b = ins
        if any(flags):
            mn, mx = iv.minimum(a, b), iv.maximum(a, b)
            hull = iv.IntervalArray(shape=mn.shape, los=mn.los, his=mx.his)
            return [iv.subnormal_haze_fmt(hull, min_normal)[0]], [True]
        return [iv.subnormal_haze_fmt(fn(a, b), min_normal)[0]], [False]

    return t


def _ieee_exp(eqn, params, ins, flags, budget):
    """exp under ieee — a bracket of the TRUE REAL value, widened by the
    caller's DECLARED budget for the backend that will execute it.

    Three steps, and the middle one is what audit 0.2.0 S9 and S11 bought:

    1. ``iv.exp`` brackets the true real image of the argument box, riding
       the assumption that the host's ``math.exp`` is faithfully rounded
       (the ±1-binary64-ulp bump). That assumption is about *this*
       process's libm and it is the right one for step 1.
    2. :func:`_libm_widen_box` widens by the declared budget for
       ``(exp, this format)``. **With no budget this DECLINES** — the
       bracket of glibc's exp is not a bracket of XLA's, and the gap is up
       to 5.5 float32 ulps on the measured backend.
    3. the subnormal haze (a flushing libm returns 0 where the bracket is
       subnormal — measured, ``exp(-720) = 0.0``) and the outward round
       onto the format's grid, which is the identity for binary64.

    ``exp(NaN)`` is NaN and nothing else, so flag propagation is exact.
    """
    dtype, fmt = _ieee_get_dtype_format(eqn)
    ulps = _libm_budget_ulps("exp", dtype, budget)
    min_normal = _ieee_format_min_normal(fmt)
    box = _libm_widen_box(iv.exp(ins[0]), fmt, ulps, floor=0.0)
    box = iv.subnormal_haze_fmt(box, min_normal)[0]
    return [_ieee_round_box(box, fmt)], [flags[0]]


def _ieee_pow(eqn, params, ins, flags, budget):
    """pow under ieee — :func:`_ieee_exp`'s three steps over the four
    monotone corners, and the same DECLARED budget gate.

    A maybe-NaN operand DECLINES rather than flag-propagating: IEEE pow has
    non-NaN results at NaN inputs (``pow(NaN, 0) = 1`` and
    ``pow(1, NaN) = 1`` — measured on jax 0.11.0), and 1 may lie outside
    the corner bracket, so flag propagation alone would be unsound.

    The budget gate runs FIRST, before that check: a harness with no
    declared budget needs one whatever its operands look like, and the
    decline that says so is the more actionable of the two.

    ``pow`` measures far better than ``exp`` on this backend (0.5–0.54 ulps
    against 5.5) and it still declines by default, because "measured well
    once" is not something a verdict can rest on silently. The profile
    carries the number and the stamp carries the profile.
    """
    dtype, fmt = _ieee_get_dtype_format(eqn)
    ulps = _libm_budget_ulps("pow", dtype, budget)
    min_normal = _ieee_format_min_normal(fmt)
    if any(flags):
        raise iv.IntervalError(
            "pow over a maybe-NaN operand: IEEE pow(NaN, 0) = 1 and "
            "pow(1, NaN) = 1 escape both the corner bracket and the NaN "
            "flag — no sound rule here, declined"
        )
    box = _libm_widen_box(iv.pow_(*ins), fmt, ulps, floor=0.0)
    box = iv.subnormal_haze_fmt(box, min_normal)[0]
    return [_ieee_round_box(box, fmt)], [False]


def _ieee_sqrt(eqn, params, ins, flags):
    """sqrt under ieee — native binary64, CORRECTLY rounded, so the float
    root is bracketed exactly (no outward rounding), unlike the 1-ulp libm
    bracket exp/pow carry. A NEGATIVE argument produces NaN, routed into the
    flag by :func:`stelling.interval.ieee_sqrt` (never leaked as an
    endpoint); a maybe-NaN operand poisons the result (``sqrt(NaN) = NaN``),
    so the operand flag ORs into the output flag. Format-parametric: uses the
    format's own subnormal band for sqrt."""
    fmt = _ieee_get_format(eqn)
    min_normal = _ieee_format_min_normal(fmt)
    if fmt == _FLOAT_FORMATS["float64"]:
        box, made_nan = iv.ieee_sqrt(ins[0])
    else:
        box, made_nan = iv.ieee_sqrt_fmt(ins[0], min_normal)
        box = _ieee_round_box(box, fmt)
    return [box], [made_nan or flags[0]]


def _ieee_cmp_get_min_normal(eqn) -> float:
    """Get the format's min_normal for comparison operands. Comparisons
    produce booleans but consume floats — the subnormal haze applies to the
    INPUT dtypes, not the output. Integer/bool comparisons have no subnormal
    band. Returns the min_normal for the input float format, or the binary64
    min_normal if inputs are non-float (integers/bools are unaffected).

    **THE WIDEST operand band, never the alphabetically-first one.** This
    used to sort the operands' float dtypes and take ``[0]``, and
    ``bfloat16 < float16 < float32 < float64``, so a `{bfloat16, float16}`
    comparison was hazed with bfloat16's ``2**-126`` where the float16
    operand needs ``2**-14`` — 112 decades too narrow, and the band is
    exactly what keeps the verdict sound for a flushing target (audit
    0.2.0 M13).

    **Why the widest band and NOT the agreement check the finding
    proposed.** The arithmetic face declines a mixed equation because a
    mixed arithmetic equation has no result format — there is no grid to
    round the sum onto. A comparison has no result format to pick: it
    consumes two values and produces a bool, and the only per-format
    quantity it uses is the DAZ band. Taking the MAXIMUM band over the
    operands is sound for every one of them, because the haze HULLS with
    0 rather than replacing (`_elt_haze_fmt`) — a band wider than an
    operand needs costs precision and can never cost soundness. So the
    general rule is available here and it is strictly more capable than
    declining, which would additionally have withdrawn the
    `{float32, float64}` mixture that hand-built and deserialized IR
    routinely carries (a float32 value compared against a binary64
    literal) and that the old code happened to get right. Agreement stays
    REQUIRED on the arithmetic face, where it is load-bearing for a
    different reason.
    """
    dtypes = {v.aval.dtype or "" for v in eqn.invars}
    floats = sorted(d for d in dtypes if "float" in d)
    if not floats:
        # pure integer/bool comparison — no subnormal hazard
        return iv.MIN_NORMAL
    bands = []
    for d in floats:
        fmt = _FLOAT_FORMATS.get(d)
        if fmt is None:
            raise iv.IntervalError(
                f"ieee comparison: dtype {d!r} is not a supported "
                f"format (supported: {sorted(_FLOAT_FORMATS)}) — declined"
            )
        bands.append(_ieee_format_min_normal(fmt))
    return max(bands)


# --- the real-mode subnormal TELL -------------------------------------------
#
# THE ROWS THAT TOOK THE DEPARTURE, AND THE ROW THAT DID NOT. `_t_sign` and
# `_t_rem` gate their definite branches on `interval.MIN_NORMAL` rather than
# on zero, and `_refuse_non_f64_float` calls that what it is: *"A DELIBERATE
# DEPARTURE FROM A STATED POSTURE"*. The comparisons never took it, and
# `_t_sign`'s own docstring records the residue in advance:
#
#     STRICTER THAN ITS NEIGHBOURS, deliberately, and recorded rather than
#     hidden: real-mode ``gt([1e-320, 1e-300], 0)`` returns definite TRUE
#     today — the same device-dependence answered the other way. […] this
#     transfer takes the answer that is sound under either reading and
#     changes nothing else to match.
#
# THE POSTURE IS KEPT AND NOTHING BELOW JUDGES. The departure is NOT
# extended to comparisons: a real-mode comparison still returns the ℝ
# answer, at the same tier, with the same coverage accounting, and the
# `semantics: real` stamp's own disclaimer ("a predicate can hold in ℝ and
# fail in floats") still says what mode this is. What is added is a NOTE, on
# the one shape where that sentence stops being a general disclaimer and
# becomes a specific, known divergence — so that a 100%-coverage VERIFIED
# with no decline and no ⊤, over a box whose values the target reads as
# zero, SAYS WHICH SEMANTICS IT IS SPEAKING and cannot be mistaken for a
# claim about what the hardware computes.
#
# The mode question itself — whether real mode brackets the EXECUTED program
# or the real-arithmetic idealization — is not answered here. It predates
# these rows, it reaches every comparison, and it is the principal's to
# settle. A tell is what an unsettled question is allowed to ship with.
_SUBNORMAL_TELL_ROWS = frozenset({"lt", "gt", "le", "ge", "eq", "ne"})


def _subnormal_tell_band(dtype):
    """The band the TELL applies to ONE operand, or ``None`` for "say
    nothing about this operand".

    Three ways to get ``None``, and they are three different silences:
    the operand is not a float (no band exists); the measured target is not
    known to flush that format (:func:`stelling.interval.
    target_flushes_subnormals` returns ``False`` or ``None`` — the second
    being "not measured", which is not "does not flush"); or the format has
    no entry in :data:`_FLOAT_FORMATS` and therefore no band on file.

    **PER OPERAND, AT ITS OWN FORMAT'S BAND — NOT the widest band in the
    equation.** :func:`_ieee_cmp_get_min_normal` takes the maximum over the
    operands, and that is RIGHT for the `ieee` haze, where the haze HULLS
    and a band wider than an operand needs costs precision and never
    soundness. It is WRONG here, because this function does not widen a
    box: it decides whether to ASSERT that the target's flush removes an
    answer. Over-hazing an assertion makes it false. Measured (0.2.0 B18
    fixup): a float64 ``[1e-10, 1e-9]``, 298 decades clear of its own band,
    compared against a `float16` zero was hazed at float16's ``2**-14`` and
    fired — with the note's own first clause, *"an operand box reaches into
    the subnormal band of its own format"*, false on that very run.
    """
    if not dtype or "float" not in dtype:
        return None
    if iv.target_flushes_subnormals(dtype) is not True:
        return None
    fmt = _FLOAT_FORMATS.get(dtype)
    if fmt is None:
        return None
    return _ieee_format_min_normal(fmt)


def _subnormal_flush_tell(eqn, params, ins, out):
    """Would the target's DAZ flush take this real-mode comparison's
    definite answer away? ``(lost, fired)`` if it would, ``None`` if it
    would not, where ``fired`` is the sorted ``(dtype, min_normal)`` pairs
    of the operands the flush actually moved.

    ``lost`` counts elements answered DEFINITELY in ℝ whose answer is not
    established once the operands are read the way the measured target reads
    them. ``None`` is the ordinary case and the one that must stay
    byte-identical: this function is the whole of the tell's trigger, and a
    ``None`` here means not one byte of the verdict moves.

    **THE PREDICATE IS A DIFFERENCE, NOT A SIZE TEST.** "Is the box small"
    would fire on everything with a zero in it. What is compared here is two
    runs of THE SAME ROW — the row's ℝ answer, which is what the verdict
    carries, against the row's answer over operands hulled with 0 across the
    subnormal band of their own format. The haze is
    :func:`stelling.interval.subnormal_haze_fmt`, the DAZ model ieee mode
    already carries. The row is fetched from :data:`TRANSFERS` rather than
    re-derived, for the same reason.

    **THE HAZE IS PER OPERAND AND ONLY WHERE THE TARGET IS MEASURED TO
    FLUSH.** Both halves are load-bearing and both were wrong for one
    commit; see :func:`_subnormal_tell_band` for the band half. The flush
    half: whether the target flushes is a PER-FORMAT measured fact, and on
    this one float16 keeps gradual underflow while the other three flush
    (:data:`stelling.interval._FORMAT_TARGET_FLUSHES`, the one table every
    sentence about the flush is derived from — including the `ieee` stamp's,
    so the two faces of a run cannot disagree about it). A float16 box
    strictly inside float16's band is a box the target reads exactly as
    written, and a tell there would be asserting a flush that does not
    happen.

    So it is SILENT wherever the flush cannot change an answer, which is
    where the ordinary harness lives:

      * a box clear of the band OF ITS OWN FORMAT — the haze is the
        identity;
      * a box that already contains 0 — the haze is the identity too, which
        is why a declaration like ``[-10, 10]`` never reaches this, and why
        ⊤ never does;
      * an integer or bool comparison — no subnormal band exists;
      * a float16 operand — the measured target does not flush it;
      * a claim the flush does not decide differently — ``[1e-320, 1e-300]
        < 1.0`` is still definitely true with the low end read as ``0.0``.

    And it is LOUD on the shape that motivated it: ``x > 0.0`` over a
    float64 ``[1e-320, 1e-300]``, VERIFIED at 100% coverage with no decline,
    no ⊤ and — before this — nothing in the verdict to distinguish it from a
    statement about the running program. Both directions are pinned in
    tests/test_subnormal_tell.py; a tell that cannot fire and a tell that
    fires on everything are the same defect, and neither is caught by an
    example that only fires — nor by one that only ever uses float64, which
    is how the two defects above shipped.

    **THE ``fired`` EXIT IS A FAST PATH AND NOT A GUARD, and it is written
    down that way because a comment claiming otherwise is the failure this
    file is full of warnings about.** When the haze is the identity on every
    operand the hazed answer IS the ℝ answer, so ``lost`` is 0 and the
    function returns ``None`` by the last gate anyway. Measured: with that
    exit deleted, tests/test_subnormal_tell.py is unchanged — no test moves,
    because no OUTCOME moves. What it buys is that an ordinary program does
    not pay for a second transfer call at every comparison it contains.
    """
    if eqn.primitive not in _SUBNORMAL_TELL_ROWS:
        return None
    hazed, fired = [], {}
    for v, x in zip(eqn.invars, ins):
        band = _subnormal_tell_band(v.aval.dtype)
        if band is None:
            # not a float, not measured to flush, or no band on file. The
            # operand goes through unhazed: silence about an operand is
            # never a claim that it is clear of its band, and hazing it
            # anyway would put a flush the target does not perform into an
            # assertion about the target.
            hazed.append(x)
            continue
        h, changed = iv.subnormal_haze_fmt(x, band)
        hazed.append(h)
        if changed:
            fired[v.aval.dtype] = band
    if not fired:
        return None
    try:
        flushed = TRANSFERS[eqn.primitive][0](eqn, params, hazed)[0]
    except iv.IntervalError:
        # the row declining over the hazed operands says nothing about the
        # answer it gave over the real ones; no tell rather than a guess.
        return None
    lost = sum(
        1
        for lo, hi, flo, fhi in zip(out.los, out.his, flushed.los, flushed.his)
        if lo == hi and (flo, fhi) != (lo, hi)
    )
    if not lost:
        return None
    return lost, tuple(sorted(fired.items()))


def _subnormal_tell_text(eqn, where, lost, fired, provenance):
    """The tell's sentence. Split out from the walk so a test can read it
    without driving a whole propagation, and so the bands it names are the
    bands :func:`_subnormal_flush_tell` measured with — ``fired`` is that
    function's own list of the operand formats the flush moved, so the
    sentence cannot name a format the run did not haze.

    **THE FLUSH CLAIM IS DERIVED, NOT RE-WRITTEN.** Which formats this
    target flushes comes from :func:`stelling.interval.
    measured_flush_clause`, the same builder the `ieee` stamp uses. For one
    commit this sentence carried its own hard-coded float64 evidence
    (*"reads 5e-324 > 0 as False"*) and stamped it on float16 runs, where
    the `ieee` face of the SAME run said the opposite and said it correctly.
    A second spelling of a measured fact is a second thing to be wrong.
    """
    bands = "; ".join(
        f"{d}: 0 < |x| < {iv._FORMAT_MIN_NORMAL_TEXT.get(d, '?')} = "
        f"{mn!r}, and {iv._TARGET_MEASURED} reads "
        f"{_ieee_format_min_positive(_FLOAT_FORMATS[d])!r} > 0 as False"
        for d, mn in fired
    )
    return (
        f"SUBNORMAL-SENSITIVE DEFINITE ANSWER: {eqn.primitive!r} at {where} "
        f"answers definitely for {lost} element(s) under semantics='real', "
        f"which judges the declared set in exact real arithmetic — and an "
        f"operand box reaches into the subnormal band of its own format, in "
        f"a format this target is MEASURED to flush ({bands}). Whether a "
        f"target flushes subnormals is device/compiler-dependent AND "
        f"PER-FORMAT — {iv.measured_flush_clause()} — so on the formats "
        f"named above it answers from 0.0 where ℝ reads a nonzero value, "
        f"and {lost} of those element(s) then have no definite answer at "
        f"all. THE VERDICT IS UNCHANGED AND IS AN ℝ CLAIM over the declared "
        f"set, which is the registered posture of this mode; it is NOT a "
        f"claim about what the hardware computes. semantics='ieee' is the "
        f"dial that models the flush — it hazes these same operands at a "
        f"band at least this wide (it takes the widest band in the "
        f"equation, which only ever hulls more) and does not answer these "
        f"element(s) definitely{provenance}"
    )


def _ieee_cmp_f64_only(eqn) -> None:
    """Legacy comparison guard — kept for backward compatibility with test
    assertions that check for 'binary64 only' in decline notes. Now only
    used internally when we genuinely need f64-only behavior."""
    float_dtypes = sorted(
        d for v in eqn.invars
        for d in [v.aval.dtype or ""]
        if "float" in d and d != "float64"
    )
    if float_dtypes:
        raise iv.IntervalError(
            f"ieee mode models binary64 only; {'/'.join(float_dtypes)} comparison "
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
    must be judged as possibly 0. Format-parametric: uses the format's own
    subnormal band for the operand haze."""
    blocked = iv.BOOL_FALSE if nan_answer else iv.BOOL_TRUE

    def t(eqn, params, ins, flags):
        min_normal = _ieee_cmp_get_min_normal(eqn)
        r = fn(*(iv.subnormal_haze_fmt(x, min_normal)[0] for x in ins))
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


def _ieee_not(eqn, params, ins, flags):
    """Kleene NOT on bools under ieee semantics. A maybe-NaN flag on a
    bool operand is a decline artifact (⊤-maybe-NaN); that operand's
    elements read as unknown before the NOT."""
    dtypes = [v.aval.dtype for v in eqn.invars]
    if any(d != "bool" for d in dtypes):
        raise iv.IntervalError(
            f"'not' transfer covers bool operands only; got dtypes "
            f"{dtypes}"
        )
    [a] = ins
    [f] = flags
    if f:
        a = iv.IntervalArray(
            shape=a.shape, los=(0.0,) * a.size, his=(1.0,) * a.size
        )
    return [iv.logical_not(a)], [False]


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
    fmt = _ieee_get_format(eqn)
    min_normal = _ieee_format_min_normal(fmt)
    if fmt == _FLOAT_FORMATS["float64"]:
        box, made_nan = iv.ieee_reduce_sum(ins[0], tuple(_req(params, "axes", "reduce_sum")))
    else:
        box, made_nan = iv.ieee_reduce_sum_fmt(
            ins[0], tuple(_req(params, "axes", "reduce_sum")), min_normal
        )
        box = _ieee_round_box(box, fmt)
    reads_an_element = ins[0].size > 0
    return [box], [made_nan or (flags[0] and reads_an_element)]


def _ieee_square(eqn, params, ins, flags):
    """`square` under ieee — DECLINES, and the reason is not schedule ambiguity.

    Unlike `integer_pow`, `square_p` fixes the evaluation completely: it is ONE
    correctly-rounded multiplication, so there is no `(x*x)*x` versus `x*(x*x)`
    disagreement to model.

    It declines for a different reason. The real-mode transfer's whole value is
    the SAME-OPERAND rule — a straddling box maps to `[0, max(lo², hi²)]`
    because x·x cannot be negative. Reusing `ieee_mul(a, a)` would discard
    exactly that: it treats the operands as independent and returns `[-6, 9]`
    for `[-2, 3]`, which is sound and useless. An ieee square transfer needs the
    same-operand rule restated in ieee terms (signed zeros, the maybe-NaN flag,
    and the DAZ haze on the endpoints), which is a separate piece of work.

    So: ⊤ with the reason quoted, rather than a silently weaker box. The row is
    REAL-MODE ONLY and the acceptance record says so.
    """
    return None


def _ieee_sign(eqn, params, ins, flags):
    """`sign` under ieee — and unlike `square`, this one does NOT decline.

    The reason is that the real-mode rule was already built to be sound here.
    Its definite branches are gated on MIN_NORMAL, so the open subnormal band
    admits 0 — which IS the DAZ answer, applied to the operand where it
    belongs rather than hulled onto the result afterwards. (Hazing the OUTPUT
    would be a no-op that looks like a safeguard: {-1, 0, 1} contains no
    subnormal, so `subnormal_haze` on the result would change nothing while
    reading as though it had handled the band. It is handled by the floor.)

    NaN routes to the flag exactly as `neg` and `abs` do: sign(NaN) = NaN, so
    a maybe-NaN operand yields a maybe-NaN result and the box below covers
    the non-NaN part. Endpoints are exact — no rounding occurs in a sign — so
    the ieee endpoint assumption is satisfied trivially.

    Format-parametric: the subnormal floor is the format's own min_normal.
    """
    fmt = _ieee_get_format(eqn)
    min_normal = _ieee_format_min_normal(fmt)
    if len(ins) != 1:
        return None
    (a,) = ins
    dtype = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    if "complex" in dtype:
        return None
    floor_ = 1.0 if _is_integer_dtype(dtype) else min_normal
    los, his = [], []
    for lo, hi in zip(a.los, a.his):
        if lo >= floor_:
            l, h = 1.0, 1.0
        elif hi <= -floor_:
            l, h = -1.0, -1.0
        elif lo >= 0.0 and hi <= 0.0:
            l, h = 0.0, 0.0
        elif lo >= 0.0:
            l, h = 0.0, 1.0
        elif hi <= 0.0:
            l, h = -1.0, 0.0
        else:
            l, h = -1.0, 1.0
        los.append(l)
        his.append(h)
    outs = [iv.IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))]
    if _is_integer_dtype(dtype):
        outs = _int_overflow_guard(eqn, "sign", outs)
        if outs is None:
            return None
    return outs, [flags[0]]


def _ieee_rem(eqn, params, ins, flags):
    """`rem` under ieee — also does not decline, and for a sharper reason:
    truncated remainder is EXACT in the target format. `fmod` introduces no
    rounding at all (the result is representable whenever the operands are),
    so there is no schedule to fix, no association to worry about, and no
    outward bump to justify — the contrast with `reduce_sum` and
    `integer_pow`, which decline precisely because the jaxpr does not fix
    their arithmetic.

    The DAZ hazard is refused by the divisor guard, which tests against the
    format's own min_normal rather than zero: a subnormal divisor reads as 0
    on the measured target and rem(a, 0) is NaN. Format-parametric.
    """
    fmt = _ieee_get_format(eqn)
    min_normal = _ieee_format_min_normal(fmt)
    # For non-float64, we can't reuse _t_rem (it calls _refuse_non_f64_float).
    # Inline the rem logic with the format's own min_normal.
    if len(ins) != 2:
        return None
    (a, b) = ins
    dtype = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    if "complex" in dtype:
        return None
    try:
        shape, xs, ys = iv._pair_elements(a, b)
    except iv.IntervalError:
        return None
    los, his = [], []
    for (alo, ahi), (blo, bhi) in zip(xs, ys):
        if not (math.isfinite(alo) and math.isfinite(ahi)):
            raise iv.IntervalError(
                f"'rem' declined: the dividend's interval [{alo}, {ahi}] is "
                f"not bounded, and rem of an infinity is NaN"
            )
        if blo < min_normal and bhi > -min_normal:
            why = (
                "contains zero"
                if blo <= 0.0 <= bhi
                else (
                    f"lies inside the open subnormal band "
                    f"(|b| < {min_normal}), where the measured target "
                    f"flushes the divisor to zero (DAZ)"
                )
            )
            raise iv.IntervalError(
                f"'rem' declined: the divisor's interval [{blo}, {bhi}] "
                f"{why}, and rem(a, 0) is NaN"
            )
        m = max(abs(blo), abs(bhi))
        if alo >= 0.0:
            lo, hi = 0.0, min(ahi, m)
        elif ahi <= 0.0:
            lo, hi = max(alo, -m), 0.0
        else:
            lo, hi = max(alo, -m), min(ahi, m)
        los.append(lo)
        his.append(hi)
    outs = [iv.IntervalArray(shape=shape, los=tuple(los), his=tuple(his))]
    return outs, [any(flags)]


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
    fmt = _ieee_get_format(eqn)
    min_normal = _ieee_format_min_normal(fmt)
    y = _integer_exponent(params)
    if y is None:
        return None
    _integer_pow_budget(ins[0], y)
    if y == 0:
        return [iv.point(1.0, ins[0].shape)], [False]
    if y == 1:
        return [iv.subnormal_haze_fmt(ins[0], min_normal)[0]], [flags[0]]
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


def _ieee_is_finite(eqn, params, ins, flags):
    """``is_finite`` under ieee: same bounded-interval logic as real mode,
    PLUS: if the operand carries maybe-NaN, ``isfinite(NaN)`` is False, so
    every element that was definite-true degrades to unknown — a maybe-NaN
    operand's finiteness is not decidable."""
    r = iv.is_finite(ins[0])
    if flags[0]:
        # maybe-NaN: isfinite(NaN) is False, so definite-true ([1,1])
        # degrades to unknown ([0,1])
        los, his = [], []
        for lo, hi in zip(r.los, r.his):
            if lo == 1.0 and hi == 1.0:
                lo, hi = iv.BOOL_UNKNOWN
            los.append(lo)
            his.append(hi)
        r = iv.IntervalArray(shape=r.shape, los=tuple(los), his=tuple(his))
    return [r], [False]


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


def _ieee_dynamic_slice(eqn, params, ins, flags):
    """Data movement, sound as-is under ieee: every output element IS an
    operand element, so the output is maybe-NaN exactly when the operand
    is. Maybe-NaN START INDICES decline, as for gather and scatter — the
    bounds classification needs definite integers, and a NaN start is
    clamp-dependent garbage."""
    if any(flags[1:]):
        raise iv.IntervalError(
            "dynamic_slice start indices carry maybe-NaN under ieee "
            "semantics — the index-bounds rule needs definite non-NaN "
            "indices; declined"
        )
    outs = _t_dynamic_slice(eqn, params, ins)
    if outs is None:
        return None
    return outs, [flags[0]]


def _ieee_dynamic_update_slice(eqn, params, ins, flags):
    """Data movement, sound as-is under ieee: every output element is an
    operand element or an update element, so the output is maybe-NaN when
    either side is — the same OR the scatter set-form takes. Maybe-NaN
    start indices decline."""
    if len(ins) >= 2 and any(flags[2:]):
        raise iv.IntervalError(
            "dynamic_update_slice start indices carry maybe-NaN under ieee "
            "semantics — the index-bounds rule needs definite non-NaN "
            "indices; declined"
        )
    outs = _t_dynamic_update_slice(eqn, params, ins)
    if outs is None:
        return None
    return outs, [flags[0] or flags[1]]


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
    """convert_element_type under ieee. Format-parametric: supports
    conversions between supported float formats.

    Float-to-float conversions:
    - WIDENING (e.g., float32->float64): the subnormal haze on the SOURCE
      value covers DAZ on the source format (a subnormal source value may
      read as 0 during the conversion — measured jax 0.11.0 CPU). After
      hazing, every source value is exactly representable in the wider
      format, so the interval passes through unchanged.
    - NARROWING (e.g., float64->float32): the interval endpoints are
      rounded OUTWARD to the destination format's grid, soundly covering
      every possible rounded result. The subnormal haze on the source
      covers DAZ on the source side.

    Int/bool sources to float: value-preserving for every source value
    (same as before). Float->int trunc: exact, declines maybe-NaN.
    """
    src = (eqn.invars[0].aval.dtype or "") if eqn.invars else ""
    dst = str(params.get("new_dtype"))

    # Float-to-float conversion between supported formats
    if "float" in src and "float" in dst:
        src_fmt = _FLOAT_FORMATS.get(src)
        dst_fmt = _FLOAT_FORMATS.get(dst)
        if src_fmt is None or dst_fmt is None:
            return None  # unsupported format -> decline to top
        # Apply subnormal haze for the SOURCE format's band (DAZ on source)
        src_min_normal = _ieee_format_min_normal(src_fmt)
        box = iv.subnormal_haze_fmt(ins[0], src_min_normal)[0]
        # If narrowing, round outward to the destination format's grid
        if src_fmt != dst_fmt:
            box = _ieee_round_box(box, dst_fmt)
            # Apply FTZ haze for the DESTINATION format (result may flush)
            dst_min_normal = _ieee_format_min_normal(dst_fmt)
            box = iv.subnormal_haze_fmt(box, dst_min_normal)[0]
        return [box], [flags[0]]

    if src == dst or (src, dst) in _EXACT_CONVERSIONS:
        return [ins[0]], [flags[0]]
    if _in_range_int_narrowing(ins[0], src, dst):
        # the statically-in-range int64->int32 narrowing (third audit,
        # F5b) is an exact integer identity — no float semantics, no
        # flush hazard — so it passes under ieee exactly as in real mode
        return [ins[0]], [flags[0]]
    if "int" in src and "float" in dst:
        # Integer to float: check if all values are exactly representable
        dst_fmt = _FLOAT_FORMATS.get(dst)
        if dst_fmt is not None:
            p, _, _ = dst_fmt
            bound = 2**p
            if all(-bound <= x <= bound for x in (*ins[0].los, *ins[0].his)):
                return [ins[0]], [flags[0]]
    if (src, dst) == ("int64", "float64") and _finite_point(ins[0]):
        # int64 point at an exactly-representable value: integer identity,
        # no float semantics involved — same as the real-mode rule
        bound = 2**53
        if all(-bound <= x <= bound for x in (*ins[0].los, *ins[0].his)):
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
    changed. Format-parametric: uses the format's own subnormal band."""
    dtype = str(_req(params, "dtype", "stelling_any"))
    fmt = _FLOAT_FORMATS.get(dtype)
    box = iv.from_bounds(
        tuple(_req(params, "shape", "stelling_any")),
        float(_req(params, "lo", "stelling_any")),
        float(_req(params, "hi", "stelling_any")),
    )
    if fmt is not None:
        min_normal = _ieee_format_min_normal(fmt)
        return [iv.subnormal_haze_fmt(box, min_normal)[0]], [False]
    # Non-float declarations (int, bool) pass through without haze
    return [box], [False]


IEEE_TRANSFERS = {
    # (ii) ieee variants: the monotone arithmetic core — native binary64
    # endpoints, NaN corners routed to the flag; the real-mode 0·∞ = 0
    # convention (iv._prod inside iv.mul) is NOT reused.
    "add": (_ieee_arith(iv.ieee_add), TIER_EXACT),
    "sub": (_ieee_arith(iv.ieee_sub), TIER_EXACT),
    "add_any": (_ieee_arith(iv.ieee_add), TIER_EXACT),
    "square": (_ieee_square, TIER_EXACT),
    "mul": (_ieee_arith(iv.ieee_mul), TIER_EXACT),
    # a zero-containing divisor DECLINES here rather than returning ⊤ as an
    # ordinary result — same box, but counted and narrated (IEEE_ZERO_DIVISOR_TOP)
    "div": (_ieee_div, TIER_EXACT),
    # (i)/(ii) exact sign arithmetic; flag propagates (NaN stays NaN)
    "neg": (_ieee_unary_exact(iv.neg), TIER_EXACT),
    "abs": (_ieee_unary_exact(iv.abs_), TIER_EXACT),
    # (ii) exact on non-NaN operands; NaN-ordering ambiguity covered by
    # the operand hull + flag
    "max": (_ieee_minmax(iv.maximum), TIER_EXACT),
    "min": (_ieee_minmax(iv.minimum), TIER_EXACT),
    # (ii) libm brackets kept, but under ieee they bracket the WRONG
    # FUNCTION on their own — the host's math module, not the one the
    # compiler emits — so both require a DECLARED per-(op, format)
    # accuracy budget and DECLINE without one, widening the bracket by
    # the declared ulps when they have it (audit 0.2.0 S9/S11;
    # LibmBudget). pow additionally declines maybe-NaN operands
    # (pow(NaN,0)=1 escapes the flag).
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
    "not": (_ieee_not, TIER_EXACT),
    "reduce_or": (_ieee_reduce_or, TIER_EXACT),
    "is_finite": (_ieee_is_finite, TIER_EXACT),
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
    "copy": (_ieee_passthrough(_t_copy), TIER_EXACT),
    "unstack": (_ieee_passthrough(_t_unstack), TIER_EXACT),
    # (ii) exact sign arithmetic with the subnormal band handled on the
    # OPERAND by the real rule's MIN_NORMAL floor; flag propagates
    "sign": (_ieee_sign, TIER_SOUND),
    # (ii) truncated remainder is exact in binary64 — no rounding, no
    # schedule; the DAZ divisor hazard is refused by the real rule's guard
    "rem": (_ieee_rem, TIER_SOUND),
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
    "dynamic_slice": (_ieee_dynamic_slice, TIER_EXACT),
    "dynamic_update_slice": (_ieee_dynamic_update_slice, TIER_EXACT),
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


def _assert_ieee_binary_kernels_are_format_parametric() -> None:
    """Every registered ieee binary transfer has a `_FMT_BINARY_OPS` row.

    :data:`_FMT_BINARY_OPS` and :data:`IEEE_TRANSFERS` are two hand-written
    lists that must agree, the same coupling ``affine.py``'s
    ``AFFINE_SUPPORTED`` already names as load-bearing. They agreed by
    luck: the fallback for a missing row hazed a NARROW format with the
    BINARY64 subnormal band, and `_ieee_round_box` cannot put the missing
    hull-with-0 back, so the fifth binary kernel registered without a row
    would have shipped a box that excludes the value jax computes — with
    nothing in the suite to notice, because the four that exist all have
    rows (audit 0.2.0 M15).

    Checked at import, where a mismatch costs a failed import instead of a
    wrong verdict. The runtime arm in :func:`_ieee_arith` is the second
    guard, not the first.
    """
    missing = []
    for prim, (transfer, _tier) in sorted(IEEE_TRANSFERS.items()):
        kernel = getattr(transfer, "_ieee_binary_kernel", None)
        if kernel is not None and kernel not in _FMT_BINARY_OPS:
            missing.append(prim)
    if missing:
        raise RuntimeError(
            f"ieee binary transfer(s) {missing} have no _FMT_BINARY_OPS "
            f"row: on a narrow format they would haze with the binary64 "
            f"subnormal band (2**-1022) and no later outward rounding can "
            f"add the hull-with-0 the format's own band requires. Add the "
            f"format-parametric kernel before registering the transfer."
        )
    # ... and the census must BITE, in both directions. If the hook
    # stopped being attached the loop above would pass by inspecting
    # nothing; a row for a kernel no transfer runs is a dead entry that
    # makes the table look more complete than it is. `add` and `add_any`
    # share one kernel, so this compares SETS, not counts.
    seen = {
        k for t, _ in IEEE_TRANSFERS.values()
        for k in [getattr(t, "_ieee_binary_kernel", None)]
        if k is not None
    }
    if seen != set(_FMT_BINARY_OPS):
        raise RuntimeError(
            f"the ieee binary census and _FMT_BINARY_OPS disagree: "
            f"{len(seen)} kernel(s) reached from IEEE_TRANSFERS against "
            f"{len(_FMT_BINARY_OPS)} rows — either the census hook is not "
            f"being attached (so the check above inspects nothing) or a row "
            f"exists for a kernel no registered transfer runs"
        )


_assert_ieee_binary_kernels_are_format_parametric()


# -- the declared libm accuracy budget: the object a caller writes ------------
#
# Placed after the registry because the set of ops a budget may name is
# DERIVED from the registry's own tiers rather than hand-written beside it:
# the sixth transfer registered at `TIER_SOUND_LIBM` becomes budget-gated by
# being registered, with no second list to keep in step. That is the same
# coupling `_assert_ieee_binary_kernels_are_format_parametric` closes one
# table over, and it is closed here by construction instead of by a check.
LIBM_BUDGET_OPS = frozenset(
    prim for prim, (_t, tier) in IEEE_TRANSFERS.items()
    if tier == TIER_SOUND_LIBM
)


@dataclass(frozen=True)
class LibmBudget:
    """A DECLARED bound on the execution backend's libm error, per
    ``(op, format)`` — the thing that re-enables ``exp`` and ``pow`` under
    ``semantics="ieee"``.

    ``ulps[(op, dtype)] = u`` declares: **for every argument, the value the
    execution target produces for** ``op`` **in** ``dtype`` **is within**
    ``u`` **ulps of the true real value**, where an ulp is the spacing of
    the binade containing that value in ``dtype``. ``u = 0.5`` is exactly
    "correctly rounded" and costs nothing; ``u = 1`` is "faithfully
    rounded"; anything larger is a measurement of one compiled function.

    **stelling does not verify this and cannot.** The declaration is about
    a function the compiler emits for a device, and nothing in this process
    can see it. What stelling does with the number is widen the bracket by
    exactly that much before rounding it onto the format's grid, and stamp
    the declaration on the verdict in the words *declared, not verified*. A
    budget smaller than the backend's real error mints a VERIFIED stelling
    has no way to catch — which is why the number has to be written down by
    someone rather than defaulted to by the tool.

    **PER (op, format), and never extrapolated.** A budget naming
    ``("exp", "float64")`` does nothing for float32 exp: on the measured
    backend those two differ by more than a factor of three, and float16
    differs from float32 by a factor of eleven in the other direction. A
    pair this budget does not name DECLINES, exactly as if no budget had
    been passed.

    ``name`` and ``basis`` are REQUIRED, and not as decoration. A budget is
    a citable claim: the name is what the stamp carries and what a reader
    looks up; the basis is what they read to decide whether to believe it.
    A bare number in a stamp is a number nobody can audit — the same reason
    a solver invocation may not be made on defaults.

    Named profiles live in :data:`LIBM_PROFILES` and are passed by name, so
    the common case needs no import::

        check(harness, vacuity_mode="inputs-only", semantics="ieee",
              libm_budget="xla-cpu-2026-08")

    **A profile name is DATED on purpose.** The number it carries is a
    property of one jaxlib, on one device class, on one day. When jaxlib
    moves, the name stays honest about what it measured and when — which is
    the one thing a bare number can never be, and it is what makes the
    durability problem survivable rather than solved.
    """

    name: str
    basis: str
    ulps: object = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "LibmBudget.name must be a non-empty string: it is what the "
                "verdict's stamp carries, and an unnamed accuracy claim is "
                "one no reader can look up"
            )
        if not isinstance(self.basis, str) or len(self.basis.strip()) < 8:
            raise ValueError(
                f"LibmBudget.basis must say what was measured, on what, and "
                f"when — a budget is a claim about a backend, and the basis "
                f"is the only thing a reader of the stamp has to judge it "
                f"by; got {self.basis!r}"
            )
        raw = self.ulps
        items: list[tuple[tuple[str, str], float]] = []
        pairs = raw.items() if hasattr(raw, "items") else raw
        for entry in pairs:
            try:
                key, value = entry
                op, dtype = key
            except (TypeError, ValueError):
                raise ValueError(
                    f"LibmBudget.ulps entries are ((op, dtype), ulps) pairs; "
                    f"got {entry!r}"
                ) from None
            if op not in LIBM_BUDGET_OPS:
                raise ValueError(
                    f"LibmBudget: {op!r} does not ride a libm accuracy "
                    f"assumption under ieee semantics, so a budget for it "
                    f"would never be read. The ops that do: "
                    f"{sorted(LIBM_BUDGET_OPS)}"
                )
            if dtype not in _FLOAT_FORMATS:
                raise ValueError(
                    f"LibmBudget: {dtype!r} is not a catalogued float format "
                    f"(known: {sorted(_FLOAT_FORMATS)})"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"LibmBudget: the budget for ({op!r}, {dtype!r}) must be "
                    f"a number of ulps, got {type(value).__name__} "
                    f"({value!r})"
                )
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"LibmBudget: the budget for ({op!r}, {dtype!r}) must be "
                    f"a finite non-negative number of ulps, got {value!r}"
                )
            items.append(((op, dtype), value))
        if not items:
            raise ValueError(
                "LibmBudget.ulps is empty: a budget that declares nothing "
                "re-enables nothing, and every (op, format) it does not "
                "name still declines. Name the pairs your harness uses."
            )
        seen: dict[tuple[str, str], float] = {}
        for key, value in items:
            if key in seen and seen[key] != value:
                raise ValueError(
                    f"LibmBudget: two different budgets declared for {key}: "
                    f"{seen[key]} and {value}"
                )
            seen[key] = value
        object.__setattr__(self, "ulps", tuple(sorted(seen.items())))

    def get(self, op: str, dtype: str) -> float | None:
        """The declared budget for ``(op, dtype)``, or ``None`` when this
        budget does not name that pair. **None means decline.**"""
        for (o, d), u in self.ulps:
            if o == op and d == dtype:
                return u
        return None

    def render(self, used) -> str:
        """The stamped assumption line, over the pairs this run actually
        consumed.

        **The most important sentence a budgeted ieee verdict carries.** A
        budget that is too small mints a false VERIFIED and stelling cannot
        check it, so the line has to say that the accuracy was DECLARED and
        not verified, and what a wrong declaration costs — otherwise the
        stamp reads as though something had been established.
        """
        want = set(used)
        declared = "; ".join(
            f"{op}@{dtype} <= {u:g} ulp{'' if u == 1 else 's'}"
            for (op, dtype), u in self.ulps
            if (op, dtype) in want
        )
        return (
            f"ieee libm accuracy DECLARED, NOT VERIFIED — profile "
            f"{self.name!r}: {declared}. TWO claims compose to make this "
            f"verdict and stelling checks NEITHER. (1) the math module of "
            f"the host that ran this analysis (CPython -> the platform "
            f"libm) is faithfully rounded, which is what makes the "
            f"1-binary64-ulp bracket contain the TRUE REAL value. (2) the "
            f"function the compiler emits FOR YOUR TARGET is within the "
            f"declared ulps of that true value; the bracket is widened by "
            f"exactly that much before being rounded onto the format's "
            f"grid. Claim (2) is a DECLARATION about a compiled function "
            f"stelling cannot see, execute or measure: if the target is "
            f"worse than declared, the bracket may EXCLUDE the value the "
            f"program computes, and a VERIFIED resting on it is FALSE with "
            f"nothing here able to notice. It also makes exp/pow endpoints "
            f"the exception to the ieee endpoint-arithmetic line's "
            f"'no outward rounding' clause: these are an outward-rounded "
            f"bracket, widened. Basis given for the declaration: "
            f"{self.basis}"
        )


XLA_CPU_2026_08 = LibmBudget(
    name="xla-cpu-2026-08",
    basis=(
        "measured 2026-08-15 on jax 0.11.0 / jaxlib 0.11.0, CPU backend, "
        "x86_64 Linux (glibc 2.39), CPython 3.12.3, eager and under jit "
        "(identical), as |jnp.op(x) - true(x)| in ulps of the target "
        "format, against a binary64 reference for the three narrow formats "
        "and a 60-digit decimal reference for binary64. exp is EXHAUSTIVE "
        "in all three narrow formats and, in float32, over every argument "
        "whose result is normal and finite; exp in float64 and pow "
        "everywhere are SAMPLED and bound only what was sampled — those "
        "five rows were drawn twice, the row keeps the larger maximum of "
        "the two draws, and LIBM_MEASURED names each draw's seed or says "
        "outright that it was not recorded. EVERY exp "
        "row is over the arguments whose RESULT IS NORMAL AND FINITE: a "
        "result that underflows to a subnormal is flushed to zero by this "
        "backend and is covered by the subnormal haze, not by an accuracy "
        "budget — in bfloat16 that flush reaches 108.7 ulps and would "
        "otherwise dominate the row. Each budget is the measured maximum "
        "rounded UP TO THE NEXT INTEGER, except where the measurement is "
        "correct rounding and 0.5 is exact; the maxima are in "
        "stelling.propagate.LIBM_MEASURED. This profile describes ONE "
        "jaxlib on ONE device class on ONE day, and is named so that it "
        "cannot quietly outlive that"
    ),
    ulps={
        # NOT correctly rounded, by 3e-5 of an ulp: 2 of the 63,487 float16
        # arguments measure 0.500028, so 0.5 would declare a bound this
        # backend violates and `_libm_widen_box` would widen by nothing to
        # cover it. 1 is that rounded up, and it is what every other row
        # short of correct rounding already declares. This costs the
        # "float16 exp is free" property, and the property was not true.
        ("exp", "float16"): 1.0,
        # Correctly rounded over every normal finite result (0.499988,
        # exhaustively) — the backend evaluates this in float32 and rounds,
        # leaving the bfloat16 grid thousands of backend ulps of slack. 0.5
        # costs NOTHING: round-to-nearest is monotone and the box is
        # rounded outward onto the format grid anyway.
        ("exp", "bfloat16"): 0.5,
        # measured 5.5112 exhaustively; 6 is that rounded up
        ("exp", "float32"): 6.0,
        # measured 1.6660 over 3M samples (and 1.6470 over an earlier,
        # independent 3M); 2 is that rounded up
        ("exp", "float64"): 2.0,
        # measured 0.5001 / 0.5000 / 0.5380 / 0.5059, each the larger of
        # two independent draws (the other gave 0.5001 / 0.5000 / 0.5290 /
        # 0.5056) — every one a hair above correctly rounded and nowhere
        # near faithful. 1.0 is the classic faithful-rounding claim and
        # covers every figure in both draws with room, which is what makes
        # the unrecorded seed on the larger draw survivable here.
        ("pow", "float16"): 1.0,
        ("pow", "bfloat16"): 1.0,
        ("pow", "float32"): 1.0,
        ("pow", "float64"): 1.0,
    },
)

# The shipped profiles, by name. **A released profile may never be edited
# in place** — the name is what a stamp carries, so changing what it means
# would retroactively change what old verdicts claimed. Add a new dated one.
LIBM_PROFILES: dict[str, LibmBudget] = {
    XLA_CPU_2026_08.name: XLA_CPU_2026_08,
}

if _DEFAULT_LIBM_PROFILE not in LIBM_PROFILES:  # pragma: no cover
    raise RuntimeError(
        f"the decline message points at profile {_DEFAULT_LIBM_PROFILE!r}, "
        f"which is not shipped: the incantation it tells a reader to write "
        f"would raise"
    )


def resolve_libm_budget(value) -> "LibmBudget | None":
    """``None`` | a profile NAME | a :class:`LibmBudget` -> the budget.

    Resolved EAGERLY at the entry point, so a typo'd profile name raises
    where it was written rather than arriving three layers down as a
    decline that reads like a stelling limitation.
    """
    if value is None or isinstance(value, LibmBudget):
        return value
    if isinstance(value, str):
        try:
            return LIBM_PROFILES[value]
        except KeyError:
            raise ValueError(
                f"unknown libm profile {value!r}; shipped profiles: "
                f"{sorted(LIBM_PROFILES)}. Pass a "
                f"stelling.propagate.LibmBudget to declare your own."
            ) from None
    raise TypeError(
        f"libm_budget must be None, a profile name from "
        f"{sorted(LIBM_PROFILES)}, or a stelling.propagate.LibmBudget; got "
        f"{type(value).__name__}"
    )


def _assert_libm_transfers_take_a_budget() -> None:
    """Every ieee transfer at `TIER_SOUND_LIBM` accepts the budget, and no
    other one does.

    The dispatcher decides which calling convention to use from the TIER,
    so a transfer registered at that tier with a four-argument signature
    would raise a `TypeError` out of the walk on first contact — and one
    registered at another tier while taking a budget would never be handed
    one and would ride whatever default it wrote. Both are import-time
    errors here rather than runtime surprises, which is what keeps
    `LIBM_BUDGET_OPS` a description of the registry instead of a wish.
    """
    import inspect

    wrong = []
    for prim, (transfer, tier) in sorted(IEEE_TRANSFERS.items()):
        try:
            n = len(inspect.signature(transfer).parameters)
        except (TypeError, ValueError):  # pragma: no cover - builtins only
            continue
        want = 5 if tier == TIER_SOUND_LIBM else 4
        if n != want:
            wrong.append(f"{prim} (tier {tier}, {n} params, wants {want})")
    if wrong:
        raise RuntimeError(
            f"ieee transfer signature census failed for {wrong}: a "
            f"{TIER_SOUND_LIBM!r} transfer takes "
            f"(eqn, params, ins, flags, budget) and every other takes "
            f"(eqn, params, ins, flags)"
        )
    if not LIBM_BUDGET_OPS:
        raise RuntimeError(
            "no ieee transfer is registered at tier "
            f"{TIER_SOUND_LIBM!r}, so the budget gate guards nothing — "
            "either the tier was renamed or the census is inspecting the "
            "wrong registry"
        )


_assert_libm_transfers_take_a_budget()


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

# Audit 0.2.0 S5-B1: a relational assume traced inside a `cond` branch is a
# BRANCH-SCOPED precondition. Forwarding it to the solver would assert it over
# the whole query, which the program does not guarantee — the other branch is
# real. It is dropped, and the drop says which of the two facts about it
# stopped the forwarding.
_BRANCH_SCOPED_REASON = (
    "and it is stated inside a cond branch, so it is a branch-scoped "
    "precondition, not a fact about the query — a branch-scoped precondition "
    "is not forwarded to the solver as an axiom (it would constrain the "
    "whole query by something that holds only when the branch is taken), and "
    "this build does not emit it as an implication either"
)


def _finite_point(box: iv.IntervalArray | None) -> bool:
    """Every element is a degenerate finite interval (``lo == hi``,
    finite) — the shape of a usable assume bound.

    Zero-element boxes answer True vacuously, which is why assume
    classification cannot lean on this predicate alone to recognise a
    size-0 predicate: `_atom_element_count` is the one that does."""
    return box is not None and all(
        lo == hi and math.isfinite(lo) for lo, hi in zip(box.los, box.his)
    )


def _atom_element_count(atom: ir.Atom) -> int:
    """How many elements this atom's abstract value carries.

    Read from the STATIC aval shape, not from a propagated box: assume
    classification must be able to say "this predicate has no elements"
    before it has read any interval, and an atom whose box is missing
    (an unbound var, an undecodable literal) still has a shape."""
    n = 1
    for d in atom.aval.shape:
        n *= d
    return n


def _subnormal_const_literal(atom: ir.Atom) -> bool:
    """A literal constant whose raw decode the subnormal haze changes —
    i.e. a subnormal-band constant. Under ieee its as-consumed value is
    flush-indeterminate, so the drop reason must name that shape rather
    than claim both sides vary (the audit-F6 discipline).
    Format-parametric: uses the literal's own format band."""
    if not isinstance(atom, ir.Literal):
        return False
    try:
        raw = _value_to_interval(atom.val, atom.aval.shape, atom.aval.dtype)
    except (iv.IntervalError, ir.TranscriptionError):
        return False
    lit_dtype = atom.aval.dtype or ""
    lit_fmt = _FLOAT_FORMATS.get(lit_dtype)
    if lit_fmt is not None:
        return iv.subnormal_haze_fmt(raw, _ieee_format_min_normal(lit_fmt))[1]
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


# The obligation-quote comparison tables (docs/proposed-decline-messages.md
# #1). Deliberately separate from the assume-classification maps above:
# these are read only when WORDING an undecided obligation's detail, never
# when deciding anything, and they cover all six comparisons (the assume
# tables exclude `ne` for narrowing reasons that do not apply to quoting).
_OBL_CMP_SYMBOL = {
    "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=",
}

# cmp(k, v) === flipped-cmp(v, k): normalization when the point side is on
# the left, so the quote can always read "operand <cmp> bound".
_OBL_CMP_FLIP = {
    "lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "ne": "ne",
}


# The largest finite binary64 value, as an exact int: (2 - 2**-52)*2**1023.
# The miss-distance fragment compares its EXACT Fraction against this
# BEFORE any float conversion — the exact difference of two finite doubles
# can exceed the binary64 range (opposite-sign endpoint and bound), and
# Fraction.__float__ raises OverflowError there (blinded-lens audit R1: a
# legal query crashed the whole analysis on exactly that shape).
_MAX_BINARY64 = (2**53 - 1) * 2**971


def _ulp_steps(a: float, b: float, cap: int = 3) -> int | None:
    """The EXACT number of nextafter steps from ``a`` to ``b``, or None
    when it exceeds ``cap`` (a big count carries no more meaning than the
    distance itself). Message wording only."""
    if not (math.isfinite(a) and math.isfinite(b)) or a == b:
        return None
    x, steps = a, 0
    while x != b:
        x = math.nextafter(x, b)
        steps += 1
        if steps > cap:
            return None
    return steps


# WHAT A DISCHARGE OVER ZERO ELEMENTS SAYS, and where a reader meets it.
#
# Audit 0.2.0 B8a, item 6 (M18). A universal claim over a size-0 array is
# vacuously true, and stelling said so — but only inside
# `ObligationReport.detail`, as the words "for all 0 element(s)". Driven on
# `aabb58d`: `any_array((0,), float64, (1.0, 2.0))` with `assert_(x <= 0.5)`
# returns
#
#     STATUS: VERIFIED   coverage: 3 eqns: 3 known (100%)   notes: (none
#     about the emptiness)
#
# so a consumer reading `status` and `notes` — which is what the rendered
# verdict leads with, and what a CI gate reads — sees an ordinary VERIFIED
# over a program about which nothing was checked. A shape that went to zero
# by accident (a batch dimension, a filtered index set, an `n - 1` that hit
# zero) verifies everything.
#
# The status is NOT changed: over the empty set the claim is true, and an
# UNKNOWN there would be a false one in the other direction. What changes is
# that the reader is told, in the channel they read.
EMPTY_UNIVERSAL_DETAIL = (
    "vacuously true: the predicate has ZERO elements, so the universal "
    "claim holds with nothing to check"
)

EMPTY_UNIVERSAL_NOTE = (
    "{what} #{index} at {where} is VACUOUSLY discharged: its predicate has "
    "ZERO elements, so \"true for every element\" holds because there is no "
    "element. Nothing about the program's arithmetic was checked by it. If "
    "the empty shape is not deliberate, this VERIFIED is about a query you "
    "did not mean to write"
)


def nonvacuity_summary(checks: tuple[ObligationReport, ...]) -> str:
    """The stamp's one-line `nonvacuity` field for a run's membership
    conditions — :attr:`stelling.verdict.Stamp.nonvacuity`, which the render
    prints ABOVE the notes.

    ONE MINTER, and that is the point. This sentence was spelled twice, in
    :func:`stelling.verdict.make_verdict` and in
    :func:`stelling.solvers.make_solver_verdict`, with a comment on the
    second requiring the two to stay byte-identical. Two hand-kept copies of
    a sentence that must not differ is the same shape of defect as the two
    numberings of item 5 (M3): it holds until someone edits one. It is
    derived in one place now, and the requirement is structural.

    THE ZERO-ELEMENT CASE, and why it is not simply "checked" (audit 0.2.0
    B8a FIXUP, item 2). A membership condition over a size-0 array is
    discharged VACUOUSLY — `jnp.all` of an empty array is true because
    there is no element — and the summary read

        checked — 1 membership condition(s) definitely true
        (the declared set contains the stated point)

    for it. No point was tested and the parenthetical is FALSE: nothing
    established that the declared set contains anything. Item 6 (M18) put
    the disclosure BESIDE this line as a note; the line itself still said
    the false thing, and it is the line a reader meets first.

    So a run whose membership conditions are ALL vacuous does not say
    "checked" — which also means
    :func:`stelling.verdict.make_verdict`'s VERIFIED caveat, gated on that
    prefix, now fires for it. A run with at least one non-vacuous condition
    decided true HAS tied the declared set to a stated point, so it stays
    "checked", with the vacuous ones counted out of the total rather than
    silently folded into it."""
    if not checks:
        return "UNCHECKED — no membership conditions declared"
    if all(c.status == "discharged" for c in checks):
        n = len(checks)
        vacuous = sum(1 for c in checks if c.detail == EMPTY_UNIVERSAL_DETAIL)
        if vacuous == n:
            return (
                f"VACUOUS — {n} membership condition(s) hold over ZERO "
                f"elements: no point was tested, so nothing ties the "
                f"declared set to the incident's data"
            )
        if vacuous:
            return (
                f"checked in part — {n - vacuous} of {n} membership "
                f"condition(s) definitely true (the declared set contains "
                f"the stated point); the other {vacuous} hold VACUOUSLY "
                f"over ZERO elements and tested no point"
            )
        return (
            f"checked — {n} membership condition(s) definitely true "
            f"(the declared set contains the stated point)"
        )
    if any(c.status == "violated-over-set" for c in checks):
        return (
            "FAILED — a membership condition is definitely false: the stated "
            "point is NOT in the declared set (harness defect, not a box fact)"
        )
    return "undecided — a membership condition could not be decided"


def _bool_status(b: iv.IntervalArray, *, constrained: bool = False) -> tuple[str, str]:
    """``(status, detail)`` for a judged boolean box.

    ``constrained`` is a DETAIL-ONLY input and deliberately so. The status
    is computed from ``n_true``/``n_false``/``n`` above the only line that
    reads the flag, so no value of it can change which of the three
    statuses comes back. MEASURED, not asserted: over the whole suite this
    function is called **7392** times and computing the status both ways
    disagrees **0** times — that half re-drives from the shipped tree —
    and over the 184-row `scratchpad/mechc` ledger and
    the 464-row `scratchpad/claims/corpus_b3.py` ledger, both a historical
    measurement, tracked and not in the sdist, forcing the flag
    to ``True`` and to ``False`` moves **0** of 648 obligation statuses
    (details move 62/8 and 86/4 respectively).

    Its two callers pass ``self.any_constrained`` DURING the walk, so the
    sentence is positional while the withholding decision
    (:func:`stelling.exactness.certifies_set_refutation`) is run-scoped.
    That is the intended split and not an oversight of the query-scoping
    change — see the note at the ``stelling_assert`` call site.
    """
    n_true = sum(1 for lo, hi in zip(b.los, b.his) if (lo, hi) == iv.BOOL_TRUE)
    n_false = sum(1 for lo, hi in zip(b.los, b.his) if (lo, hi) == iv.BOOL_FALSE)
    n = b.size
    if n == 0:
        # THE EMPTY UNIVERSAL, SAID AS ONE (audit 0.2.0 B8a, item 6 / M18).
        # `n_true == n` is satisfied by 0 == 0 and this used to fall through
        # to it, so a claim over a zero-element array discharged with the
        # words "definitely true for all 0 element(s)" — true, and read by
        # every consumer as an ordinary discharge. It is a DIFFERENT FACT
        # about the program: nothing was checked. The status is unchanged
        # (a universal over the empty set IS true, and downgrading it would
        # be a false UNKNOWN), the DETAIL says which kind of true it is, and
        # the walk raises a note beside it — see :data:`EMPTY_UNIVERSAL_NOTE`.
        return "discharged", EMPTY_UNIVERSAL_DETAIL
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


# The two primitives that STATE something. Found inside a sub-jaxpr the
# analysis never descends into, each is an unexamined claim rather than an
# undecided one, and :meth:`_Propagator._record_unexamined` says so.
_UNEXAMINABLE_OBLIGATIONS = frozenset({"stelling_assert", "stelling_nonvacuity"})

# ---------------------------------------------------------------------------
# Reachability witnesses for branch-scoped refutations.
#
# Refuting inside a branch presumes a reachability that nothing certifies.
# The index interval ADMITTING branch i is not evidence that branch i runs:
# interval arithmetic over-approximates, so `x - x > 0` admits both legs of
# a cond whose guard is false at every point of the declared box, and a
# definite violation found in the leg that never executes is not a violation
# of the program.
#
# What DOES certify a branch is a point of the declared box that takes it.
# Interval propagation over a box pinned to a single point brackets the
# program's values AT that point, so a cond whose index box is a singleton
# there is a cond whose real index at that point is determined — and an
# assert reached under a chain of such conds is an assert the program
# really evaluates somewhere in the declared box.
#
# The grid below is that witness search. It is DETERMINISTIC by
# construction (a verdict must not depend on a seed), SUFFICIENT rather
# than necessary (a satisfiable guard whose true region the grid misses
# simply goes uncertified, and its refutation is withheld), and it is
# never consulted unless a branch-scoped violation is actually on the
# table — a query without one runs exactly the propagation it ran before.
_PROBE_ANCHORS = (0.0, 1.0, 0.5)  # every element at lo / at hi / at the middle
_PROBE_COUNT = 16
# golden-ratio conjugate and its R2 companion: a low-discrepancy pair, so
# successive probes spread over the box instead of clustering
_PROBE_STRIDE = 0.6180339887498949
_PROBE_ELEMENT_STRIDE = 0.7548776662466927
# a declaration with an infinite endpoint has no fractions; it gets a
# ladder of finite magnitudes instead
_PROBE_LADDER = (0.0, 1.0, -1.0, 1000.0, -1000.0, 0.001, -0.001, 1e6, -1e6)


def _probe_fraction(k: int, element: int) -> float:
    """Where in its declared interval probe ``k`` puts element ``element``.

    The first probes are the plain anchors — every element at the same
    end — because the guards that matter most (``x[0] > 0`` over a box
    straddling zero) are witnessed by a corner. Later probes vary per
    element so a guard relating two elements of one array can still be
    witnessed.
    """
    if k < len(_PROBE_ANCHORS):
        return _PROBE_ANCHORS[k]
    return (
        (k + 1) * _PROBE_STRIDE + element * _PROBE_ELEMENT_STRIDE
    ) % 1.0


# The largest finite value of each binary format, DERIVED from the format
# table so the two cannot drift: (2**p - 1) * 2**(emax - p + 1). Every
# value of every format here is exactly representable in binary64 (each has
# at most 53 significand bits and an exponent range inside binary64's), so
# every comparison and every rounding below is EXACT in python floats.
_FLOAT_MAX: dict[str, float] = {
    d: math.ldexp(float(2**p - 1), emax - p + 1)
    for d, (p, _emin, emax) in _FLOAT_FORMATS.items()
}


def _round_in_format(v: float, fmt: tuple[int, int, int], direction: int) -> float:
    """``v`` rounded to a value of the binary format ``fmt = (p, emin,
    emax)``, TOWARD ``direction`` (``+1`` up, ``-1`` down) — the DIRECTED
    rounding, never round-to-nearest.

    The direction is the whole point. Round-to-nearest can cross the
    endpoint it is narrowing (``float32`` nearest of a value just above
    `lo` sits just BELOW `lo`, outside the declared interval), and the
    obvious repair — nearest, then step with :func:`math.nextafter` — is
    the float face of the trap :data:`_INT_DTYPE_BOUNDS` documents for
    ``int64``: ``nextafter`` steps to the next **binary64**, which for
    every format narrower than binary64 is not a value of the format at
    all. So the step is taken in the format's own grid, by rounding the
    exact scaled significand.

    The result is ``±inf`` exactly when the format holds no value on that
    side of ``v`` — rounding `1e308` UP in `float32` has nowhere to go —
    and the format's largest finite value when it does: rounding `-1e308`
    UP in `float32` gives `-3.4e38`, the smallest `float32` there is.
    Overflowing the SIGNIFICAND is not overflow: rounding `65503.9` up in
    `float16` carries to `65504.0`, which is a value.
    """
    if v == 0.0 or not math.isfinite(v):
        return v
    p, emin, emax = fmt
    big = math.ldexp(float(2**p - 1), emax - p + 1)
    _m, e = math.frexp(v)  # v == _m * 2**e with 0.5 <= |_m| < 1
    # the exponent of the local ulp: `e - p` in the normal range, floored
    # at the subnormal spacing so the grid never gets finer than the
    # format's smallest step
    q = max(e - p, emin - p + 1)
    scaled = math.ldexp(v, -q)  # exact: scaling by a power of two
    s = math.ceil(scaled) if direction > 0 else math.floor(scaled)
    try:
        out = math.ldexp(float(s), q)
    except OverflowError:  # far outside binary64's own exponent range
        out = math.inf if s > 0 else -math.inf
    if out > big:
        return math.inf if direction > 0 else big
    if out < -big:
        return -big if direction > 0 else -math.inf
    return out


def _member_bounds(lo: float, hi: float, dtype: str):
    """``[lo, hi]`` narrowed until its own endpoints are MEMBERS of the
    declared set, or ``(None, None)`` when the set is empty.

    The declared set of a declaration is the set of that dtype's VALUES
    inside ``[lo, hi]`` — stelling says so itself when it refuses `int32
    (0.2, 0.8)`: "int32 represents the integers ... and the interval
    contains none of them". So the set declared by `int32 (0.2, 2.8)` is
    `{1, 2}`, and `0.2` is not in it. TWO constraints, and the clamp needs
    both: the interval's own endpoints rounded INWARD to the dtype's
    grid, and the dtype's representable range (an `int8` declared over
    `(-1e9, 1e9)` declares `[-128, 127]`, nothing wider).

    A FLOAT declaration is bounded the same way twice, and reading its
    interval as its declared set is the same error one dtype family over:
    `float32 (-1e308, 1e308)` declares `[-3.4e38, 3.4e38]`, not the
    binary64 range, and `float32 (v, (v + nextafter(v))/2)` declares the
    single value `{v}` — no `float32` lies strictly inside. Only
    `float64` is its own interval, which is why a float64-only corpus
    cannot see this half at all.

    Returning the pair rather than clamping in place is what lets the
    caller round toward the dtype's grid and then clamp INTO the member
    set — clamping to the raw `[lo, hi]` after rounding puts the
    non-member endpoint straight back, which is the defect this exists to
    close.

    A dtype whose grid this module cannot name yields ``(None, None)``:
    no member it can vouch for, therefore no witness. That is
    DEFAULT-DENY and it is the only safe default here, because "skip the
    clamp" would return the raw interval and hand back a witness that is
    not a member. `any_array` accepts `int2`, `uint2`, five `float8`/
    `float4` formats and the two complex dtypes (measured, jax 0.11.0 and
    0.10.2), none of which either table names.

    A NaN endpoint yields ``(None, None)`` too, and that guard is FIRST
    because neither branch below can catch it: every comparison with NaN
    is False, so ``m_lo > m_hi`` does not fire and the pair is returned
    with the NaN still in it, and the integer branch raises
    ``ValueError`` on ``math.ceil(nan)`` before getting that far. Both
    were measured; neither is reachable through the public API, which is
    why this is a hardening and not a fix — ``any_array`` refuses NaN
    bounds at declaration ("declare an empty set; refusing at declaration
    time") and :func:`_probe_point` drops any non-finite value it forms.
    Recorded rather than left to be re-derived: an internal caller added
    later would meet the raise, not the withholding.
    """
    if math.isnan(lo) or math.isnan(hi):
        return None, None
    if _is_integer_dtype(dtype):
        d_lo, d_hi = _INT_DTYPE_BOUNDS.get(dtype, (None, None))
        if d_lo is None:
            return None, None
        m_lo = float(math.ceil(lo)) if lo != -math.inf else -math.inf
        m_hi = float(math.floor(hi)) if hi != math.inf else math.inf
        # int64/uint64 end where binary64 goes inexact: `float(2**63 - 1)`
        # rounds UP to 2**63, which is not an int64 value. Step back onto
        # the largest float that the dtype really holds — the clamp must
        # land on a member, not near one.
        f_lo, f_hi = float(d_lo), float(d_hi)
        if f_lo < d_lo:
            f_lo = math.nextafter(f_lo, math.inf)
        if f_hi > d_hi:
            f_hi = math.nextafter(f_hi, -math.inf)
        m_lo, m_hi = max(m_lo, f_lo), min(m_hi, f_hi)
        if m_lo > m_hi:
            return None, None
        return m_lo, m_hi
    fmt = _FLOAT_FORMATS.get(dtype)
    if fmt is None:
        return None, None
    big = _FLOAT_MAX[dtype]
    m_lo = -big if lo == -math.inf else _round_in_format(lo, fmt, +1)
    m_hi = big if hi == math.inf else _round_in_format(hi, fmt, -1)
    # No clamp into `[-big, big]` here, DELIBERATELY: the rounding already
    # returns a value of the format or the ±inf that says the format has
    # nothing on that side, so a clamp would be dead. An endpoint past the
    # top (`float32 (1e300, 1e308)`) makes `m_lo` +inf, the two cross, and
    # the declared set is empty — which is right, no `float32` is up there.
    # A clamp did stand here; the mutation survey deleted it and no test
    # moved, and it was removed rather than left looking load-bearing.
    if m_lo > m_hi:
        return None, None
    return m_lo, m_hi


def _probe_point(k: int, shape, lo: float, hi: float, dtype: str, base: int):
    """A point of the declared box ``[lo, hi]^shape``, as flat values.

    ``base`` offsets the per-element index by the declaration's position,
    so two declarations are not pinned to the same fraction. Every
    declaration is pinned to values OF ITS DTYPE inside the box
    (:func:`_member_bounds`): a point the dtype cannot hold is not a
    member of the declared set, and a witness that is not a member
    certifies nothing. That is one rule for integers (a non-integral
    point) and for floats (a point off the format's grid, or past its
    finite range). A box holding no value of its dtype yields ``None`` —
    no member, no witness.
    """
    n = 1
    for d in shape:
        n *= d
    m_lo, m_hi = _member_bounds(lo, hi, dtype)
    if m_lo is None:
        return None
    fmt = None if _is_integer_dtype(dtype) else _FLOAT_FORMATS.get(dtype)
    vals = []
    for j in range(n):
        f = _probe_fraction(k, base + j)
        if lo == -math.inf or hi == math.inf:
            v = _PROBE_LADDER[k % len(_PROBE_LADDER)]
            if lo != -math.inf:
                v = lo + abs(v)
            elif hi != math.inf:
                v = hi - abs(v)
        else:
            # NOT `lo + f * (hi - lo)`: on a box wider than half the float
            # range `hi - lo` overflows to +inf, `0 * inf` is NaN and
            # every other fraction saturates — measured on
            # `[-1e308, 1e308]`, 15 of the 16 probes collapse onto `hi`
            # and the 16th is no probe at all. The convex combination
            # never forms the difference, so the grid stays 16 points
            # wide; it still returns `lo` at f=0 and `hi` at f=1 exactly.
            v = lo * (1.0 - f) + hi * f
        if _is_integer_dtype(dtype) or dtype == "bool":
            v = float(math.floor(v + 0.5))
        v = min(max(v, m_lo), m_hi)
        if fmt is not None:
            # The clamp alone is not enough for floats: it fixes the two
            # ENDPOINTS and leaves every interior point on binary64's grid
            # — measured on the shipped sweep, 6716 of 6880 float32 probe
            # values were not float32 values, and only 90 of those were
            # out-of-range. `v` is already inside `[m_lo, m_hi]`, whose
            # endpoints ARE values of the format, and directed rounding is
            # monotone and fixes them, so rounding down cannot leave the
            # member set. For float64 this is the identity.
            v = _round_in_format(v, fmt, -1)
        if not math.isfinite(v):
            return None
        vals.append(v)
    return vals


# **TWO NAMES THIS CLASS HAS NEVER HAD, AND EIGHT SENTENCES BUILT ON THEM.**
# Prose in this file and in `src/stelling/obligation.py` called this class
# `_Walker`, and called its `_classify_assumed_pred` below `_classify_cmp`.
# NEITHER NAME HAS EVER BEEN DEFINED IN THIS REPOSITORY. Measured on
# 2026-08-28: `git log --all -S "class _Walker" -- src/` and `git log --all
# -S "def _classify_cmp" -- src/` each return no commits, while this class
# has been `_Propagator` since the MVP commit `4f25390` (2026-07-17) and
# `_classify_assumed_pred` was added in `8106a55` (2026-08-07). The invented
# spellings entered the prose afterwards — `f116890` (2026-08-07) and
# `0874dd1` (2026-08-14) — and reached six sites and two.
#
# So this is not a rename that was half-finished. A rename leaves a trail a
# reader can date; these were written down as if they existed. Five of the
# eight were `:meth:` roles, which do not decorate a sentence — a role
# asserts that a method of that name exists, to a reader who will go looking
# for it — and they were in the file whose subject is a soundness
# certificate.
#
# THE EIGHT ARE CORRECTED WHERE THEY STAND and this comment is the record of
# what they said, rather than the same paragraph repeated eight times.
#
# WHAT HOLDS IT NOW, AND WHAT THAT DOES NOT REACH:
# `tests/test_referenced_names_resolve.py::test_every_cross_reference_
# role_in_src_resolves` resolves every Sphinx cross-reference role in `src/`
# against the parsed tree, so a role naming a method this class does not
# have is a red. It reads ROLES ONLY. Three of the eight were plain
# backticked identifiers — one further down this file, one in
# `obligation.py`, one in `tests/test_strict_sign_algebra.py` — and those
# were found by reading; nothing catches the next one, for the
# false-positive reason that gate's docstring gives.
class _Propagator:
    def __init__(
        self,
        assume_mode: str,
        semantics: str = "real",
        libm_budget=None,
        boundary: str = "opaque",
    ) -> None:
        self.assume_mode = assume_mode
        self.semantics = semantics
        # WHETHER A SUB-JAXPR BOUNDARY CARRIES THE STRICT-SIGN CERTIFICATE
        # (:data:`_BOUNDARY_MODES`). ``"opaque"`` is the default and is
        # today's behaviour with nothing added to it; ``"transparent"``
        # carries the certificate across the boundaries this walk already
        # enters. Resolved and validated by :func:`propagate`; the class
        # default repeats the module default so a hand-built propagator —
        # the shape half of `tests/` uses — is a boundary-opaque one.
        self.boundary = boundary
        # The caller's DECLARED libm accuracy budget (ieee only; None is
        # the default and makes every libm-riding transfer decline). It is
        # already resolved to a `LibmBudget` or None by `propagate` —
        # nothing below this line accepts a profile NAME, so there is one
        # place a bad name can raise and it is the entry point.
        self.libm_budget = libm_budget
        # the (op, dtype) pairs whose budget this run actually CONSUMED —
        # the stamp names those and not the whole profile, because a
        # verdict rests on what it used
        self.libm_declared: set[tuple[str, str]] = set()
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
        # REAL mode's parallel table: var id -> +1 / -1, read as "every
        # element of this value is certainly > 0 (resp. < 0) at every point
        # of the ASSUMED region". A fact about the true real values, never
        # about the box — the box is a closed over-approximation whose
        # endpoint may sit exactly at 0 while no point of the assumed
        # region does. That gap is the whole reason the table exists: an
        # interval cannot represent an OPEN bound, so `assume(x > 0)`
        # narrows to the CLOSED `[0, hi]` and the exclusion of zero
        # survives only here (audit 0.2.0 B5-1).
        #
        # Absent id = 0 = "unknown", the conservative reading. This TABLE
        # is keyed by var id, propagated by exactly one rule set
        # (:meth:`_strict_sign_out`), and reaches exactly one consumer
        # (the `div` transfer's boundary gate). Every other primitive
        # drops the fact, which is sound in the only direction that
        # matters.
        #
        # SCOPE-SWAPPED at every sub-jaxpr descent, like the env it
        # annotates, because var ids are unique per JAXPR and not per
        # transcription. Whether anything is CARRIED across such a
        # descent is the ``boundary`` dial (:data:`_BOUNDARY_MODES`);
        # under its ``"opaque"`` default nothing is, in either direction.
        #
        # TWO WRITERS THAT MINT, both real-mode only, and each writes a
        # fact it can actually establish:
        #
        #   1. a strict `gt`/`lt` assume in
        #      :meth:`_classify_assumed_pred` — the half-space the closed
        #      meet could not represent;
        #   2. a CONSTVAR bound in :meth:`run` — whose decoded box IS its
        #      value, so its sign needs no assume at all.
        #
        # A LITERAL is in neither: it has no var id, and its sign is
        # recomputed from its value on each read
        # (:func:`_literal_strict_sign`). So the CERTIFICATE has three
        # sources and the TABLE has two MINTING writers, and an ARRAY
        # constant needs writer 2 specifically — it is traced as a
        # constvar and never reaches a rule as a Literal.
        #
        # **THIS CENSUS READ "TWO WRITERS" FLAT, AND UNDER
        # ``boundary="transparent"`` THERE IS A THIRD ENTRY POINT TO THE
        # TABLE.** It is called out as a distinct kind rather than
        # renumbered into the list, because it is a different kind:
        #
        #   3. THE BOUNDARY CARRY — the ``arg_signs`` bind in
        #      :meth:`run` and the out-carry in the transparent-wrapper
        #      arm of :meth:`eqn`, gated by :meth:`_carries_signs` and
        #      therefore INERT under the ``"opaque"`` default. It MINTS
        #      NOTHING. Every value it writes was already certified in
        #      another scope's table by writer 1 or writer 2 (or is a
        #      literal's recomputed sign), and all it does is re-key that
        #      same fact onto the var id the SAME VALUE has on the other
        #      side of a sub-jaxpr boundary. Its licence is therefore not
        #      an algebraic rule but a value identity, and that identity
        #      is argued in full at the two sites — including why the
        #      `cond` arm carries IN and never OUT.
        #
        # So under the default the table has two writers and three
        # sources, exactly as it always did; under ``"transparent"`` it
        # has a third writer and still three sources.
        #
        # REAL MODE ONLY, and that is not tidiness. Under ieee the strict
        # narrowing bumps to the format's smallest subnormal and the DAZ
        # haze hulls that straight back to a box containing 0 — the runtime
        # value IS zero there — so `x > 0` does NOT imply "certainly
        # nonzero" on a flush-to-zero target (audit 0.2.0 S10). The ieee
        # `div` kernel divides every zero-containing divisor to ⊤ with no
        # case split and never reads this table; nothing writes it under
        # ieee either.
        self.strict_sign: dict[int, int] = {}
        # HOW MANY CERTIFICATES ACTUALLY CROSSED A SUB-JAXPR BOUNDARY on
        # this walk — one per WRITE performed by a boundary carry, in
        # either direction, counted where the write happens
        # (:meth:`run` for the IN direction, the transparent-wrapper arm
        # of :meth:`eqn` for the OUT one). Always 0 under
        # ``boundary="opaque"``, because nothing carries there.
        #
        # WHAT IT IS NOT. It is not a count of verdicts that depended on
        # a carried certificate, and no arithmetic on it can become one:
        # a certificate can cross and never be read by a `div` gate, and
        # a single carried certificate can be read many times. It is the
        # count of the ACT, which is exactly what the run's disclosure
        # reports (:data:`BOUNDARY_CROSSED_DISCLOSURE`) — a disclosure
        # that claimed the stronger thing would be claiming a
        # measurement nobody made.
        self.boundary_crossings = 0
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
        # var id -> (primitive, where, cause) for values minted as artifact
        # ⊤ by top_out: MESSAGE PROVENANCE ONLY (docs/proposed-decline-
        # messages.md #2 — "a decline that reports a box must say where the
        # box came from"). Never read by any transfer, judgment, or
        # counter; ids are globally unique per transcription, so the map is
        # deliberately not scope-swapped (provenance follows the value
        # across call boundaries, like refused_shape).
        self.top_origin: dict[int, tuple[str, str, str]] = {}
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
        # > 0 while executing a cond branch the index interval only
        # ADMITS. Deliberately NOT the same counter as ``branch_depth``,
        # which counts every branch: a branch the index box FORCES runs
        # whenever the cond runs, so a definite violation found there is a
        # violation of the program, while one found under an admitted-only
        # branch presumes a reachability nothing has certified.
        self.unforced_depth = 0
        # (branch path, assert equation) keys this walk reached with
        # ``unforced_depth == 0`` — under a chain of conds every one of
        # which the index box FORCED. On a point-pinned probe run this set
        # IS the reachability certificate: the real program evaluates that
        # assert, by that route, at that point of the declared box.
        #
        # The key is the PATH and not the assert's outvar id, because the
        # id is not unique to a dynamic occurrence. Measured: jax caches
        # traced branch jaxprs, so `lax.cond(p, f, f, x)` gives both
        # branches the SAME body and the transcriber gives both asserts
        # the same outvar id (10 and 10) — an id-keyed certificate would
        # let a witness for one branch certify an occurrence in another.
        self.certain_reached: set[tuple] = set()
        # (obligation index, path key) for every violation recorded while
        # ``unforced_depth > 0`` — the candidates for withholding, paired
        # with the identity a probe run can certify.
        self.branch_violations: list[tuple[int, tuple]] = []
        # the same, for nonvacuity conditions: their definite face is the
        # FAILED stamp sentence, which is the same claim class.
        self.branch_nonvacuity_violations: list[tuple[int, tuple]] = []
        # (cond equation, branch index) for every branch currently open.
        # The equation identity is the object itself: the probe runs walk
        # the SAME `ir` query object the real run walked, so identity is
        # stable across them and needs no synthetic numbering.
        self._branch_path: list[tuple[int, int]] = []
        # THE SCOPE THIS WALK IS CURRENTLY INSIDE, as a positional address in
        # the IR tree (:data:`ScopePath`). Empty at top level; one step is
        # pushed per descent, tagged with the kind of scope it enters, and
        # popped in a `finally`. Deliberately NOT `_branch_path`: that one is
        # keyed on `id(eqn)` and exists to distinguish DYNAMIC occurrences of
        # one assert within a single process, while this must be reproducible
        # by a second walker (the slicer) that never sees the propagator's
        # objects. Positions in a fixed `eqns` list are a property of the IR;
        # object addresses are not.
        self._scope_path: ScopePath = ()
        # `(scope path, var id) -> "x{k}"` for the declarations this query's
        # EMISSION names that way, or None when the numbering is not
        # derivable (`_declaration_names` says exactly when). Filled at the
        # first `run`, which is always the outermost jaxpr, and read only by
        # `_name_of`/`_name_of_id` when WORDING a message. `_UNSET` rather
        # than `None` because `None` is a decided answer here.
        self._decl_names: dict[tuple, str] | None | object = _UNSET
        # probe index, or None on a real run. When set, every declaration's
        # box is replaced by a single POINT of that box (:func:`_probe_point`),
        # which is what turns propagation into a witness evaluator.
        self.pin: int | None = None
        self._pin_ordinal = 0
        # PROBE RUNS ONLY: assume equation id -> was this assume's predicate
        # DEFINITELY TRUE at the pinned point, on every occasion this walk
        # evaluated it. The conjunction over occasions, so a query whose
        # cond stayed unforced (both branch bodies walked) needs the assume
        # true on both. Empty on a real run: nothing writes it unless
        # `pin` is set. Read by :func:`_region_witness` and handed to
        # :func:`stelling.exactness.certifies_point_witness`.
        self.assume_witness: dict[int, bool] = {}
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
        # set once a constraining assume narrows a NON-exact variable: the
        # narrowed region was not certified inhabited, so no definite
        # violation on this run is a refutation (a possibly-vacuous
        # refutation is not a refutation — audit F7). NAMED FOR ITS
        # MECHANISM, not merged into a generic "uncertified": the sentence
        # that explains a withholding must quote the mechanism that
        # actually fired, and this one is "narrowed an over-approximated
        # intermediate". Read ONCE, at the end of the run, through
        # exactness.certifies_set_refutation — never at an obligation.
        self.narrowing_uncertified = False
        # set once MEMBERSHIP_IDIOM_HINT's body has been printed on this run.
        # One property can be stated on three faces and the hint is ~1.2k
        # characters; every later face gets the pointer instead (see
        # MEMBERSHIP_IDIOM_POINTER). Message content only.
        self.membership_hint_emitted = False
        # set when an assume was DROPPED rather than applied: the judged set
        # is a SUPERSET of the assumed region and that region was never
        # shown inhabited. Distinct from `narrowing_uncertified` because it
        # is a different mechanism (see above) and because it must also
        # reach the SOLVER path: the escalation decline keys on a
        # constrained assume being present, and a dropped one is not
        # present at all.
        self.assume_dropped = False
        # comparison equations from relational assume drops: both operands
        # vary, so the interval domain cannot narrow, but the solver can use
        # the constraint as an axiom. Accumulated during the walk; surfaced
        # on the Propagation result for the emission to read. Each entry
        # carries the SCOPE its operand ids belong to — see
        # :class:`RelationalAssume`.
        self.relational_assumes: list[RelationalAssume] = []
        # ONE :class:`AssumeDisposition` PER ASSUMED CONJUNCT THIS WALK
        # CLASSIFIES, in classification order, written by
        # :meth:`_assume_constrain` (and by the inert-mode arm of the
        # `stelling_assume` handler, which classifies nothing and says so).
        # `relational_assumes` above is the FORWARDED SUBSET of this; the
        # gap between the two is what a rule reading only the subset cannot
        # see (audit 0.2.0 S6).
        self.assume_ledger: list[AssumeDisposition] = []
        # the NON-EMPTINESS CERTIFICATE's answer for this run, written by
        # :func:`propagate` after the walk finishes and before the
        # withholding reads it (:func:`_region_witness`). False here means
        # "no witness found" and is the initial value precisely because
        # that is also the conservative one: a walk that never reaches the
        # search behaves exactly as it did before the search existed.
        self.region_inhabited = False

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
                box = _value_to_interval(atom.val, atom.aval.shape, atom.aval.dtype)
            except (iv.IntervalError, ir.TranscriptionError) as e:
                self.notes.append(f"literal outside the domain ({e}); ⊤")
                return _safe_top(atom.aval.shape)
            if self.semantics == "ieee":
                # DAZ flushes inputs: literal constants entering ieee
                # propagation are subnormal-hazed like every other value,
                # using the literal's own format's band
                lit_dtype = atom.aval.dtype or ""
                lit_fmt = _FLOAT_FORMATS.get(lit_dtype)
                if lit_fmt is not None:
                    lit_mn = _ieee_format_min_normal(lit_fmt)
                    box = iv.subnormal_haze_fmt(box, lit_mn)[0]
                else:
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
                f"equation reads IR var {atom.id} before any binding — "
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
                _value_to_interval(atom.val, atom.aval.shape, atom.aval.dtype)
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

    def read_strict_sign(self, atom: ir.Atom) -> int:
        """``+1``/``-1`` when every element of this atom's value is
        certainly positive / certainly negative over the assumed region;
        ``0`` when nothing certifies either (see ``self.strict_sign``).

        A LITERAL answers from its own decoded value
        (:func:`_literal_strict_sign`), because it is not only the `div`
        boundary gate that reads this: :meth:`_strict_sign_out` reads
        EVERY operand of every rule, so a literal answering 0 zeroed the
        whole chain through it. The boundary carry
        (:meth:`_carries_signs`) is the third reader, and it is the one
        that makes this method's silence about ``semantics`` matter — see
        :func:`_literal_strict_sign`. `0.5 * Σx²`, `2.0 * x`, `x / 2.0` and the
        `/n` inside `jnp.mean` are all literal-COEFFICIENT shapes whose
        certificate died on the coefficient, not on anything about the
        divisor (B5 follow-up audit).
        """
        if isinstance(atom, ir.Literal):
            return _literal_strict_sign(atom)
        return self.strict_sign.get(atom.id, 0)

    def _carries_signs(self) -> bool:
        """Does a sub-jaxpr boundary carry the strict-sign certificate?

        THE ONE GATE, read by every descent that could carry one, so the
        dial and the real-mode restriction cannot end up spelled
        differently at two sites. Both conjuncts are load-bearing:

        * ``boundary == "transparent"`` — the caller asked for it. Under
          the ``"opaque"`` default this returns False at every descent
          and no line of the carry runs at all.
        * ``semantics != "ieee"`` — the certificate is REAL-MODE ONLY
          (see ``self.strict_sign``), and the restriction has to be
          re-stated *here* rather than left to the writers, because
          :meth:`read_strict_sign` answers for a LITERAL from its own
          decoded value with no semantics test in it. A wrapper operand
          that is a literal would therefore hand a nonzero sign across
          the boundary under ieee, writing the table in a mode where
          nothing may write it and minting a certificate that is FALSE
          on a flush-to-zero target (audit 0.2.0 S10, and the invariant
          ``tests/test_assume_bump_boundary_div.py::test_the_literal_sign_is_REAL_MODE_ONLY``
          states).
        """
        return self.boundary == "transparent" and self.semantics != "ieee"

    def _strict_sign_out(self, eqn: ir.JaxprEqn, params, ins) -> int:
        """The strict sign of this equation's output, from its operands'.

        Every rule below is a statement about REAL values under the
        premise that each named operand is certainly nonzero of the given
        sign; none of them reads a box endpoint, so none of them can be
        defeated by outward rounding. Any primitive not listed answers 0
        — the fact is dropped, never guessed.

        An operand's premise comes from :meth:`read_strict_sign`, and
        has exactly two shapes: a VAR reads ``self.strict_sign``, a
        LITERAL reads its own decoded value
        (:func:`_literal_strict_sign`). The table itself has two MINTING
        writers — a strict assume in :meth:`_classify_assumed_pred`, and
        a CONSTVAR bound in :meth:`run` — and both are real-mode only.
        (This sentence read *"has two writers"* with no qualifier; under
        ``boundary="transparent"`` a third entry point re-keys an
        already-minted certificate across a sub-jaxpr boundary, and the
        census on ``self.strict_sign`` states all three and what
        separates them. It is real-mode only too, and inert under the
        ``"opaque"`` default.) The constant sources, literal and
        constvar, are why a coefficient no longer zeroes the chain it
        sits in.

        * ``mul``/``div``: ``sign(a·b) = sign(a)·sign(b)`` and a product or
          quotient of nonzeros is nonzero. Broadcasting is irrelevant
          because the fact is quantified over ALL elements of each operand.
        * ``add``/``add_any``: two same-signed nonzeros sum to that sign.
          (``sub`` is absent on purpose — ``a − b`` with both positive can
          be anything, and that is exactly the `Σx² − c` shape whose
          missing decline was the false VERIFIED.)
        * ``neg`` flips it; ``abs`` and ``square`` make it ``+1``.
        * ``integer_pow``: ``x ≠ 0`` gives ``x**y ≠ 0`` for every integer
          ``y`` including negative ones, with sign ``+1`` for even ``y``
          (``y = 0`` included: ``x**0 = 1``) and ``sign(x)`` for odd.
        * ``reduce_sum``: a sum of same-signed nonzeros keeps the sign,
          PROVIDED each output cell sums at least one term. ``size > 0`` on
          the operand is exactly that guarantee — every extent is nonzero,
          so every reduced extent is too. A size-0 operand sums to the
          IDENTITY 0 and must drop the fact, which the guard does.
        * ``dot_general``: each output element is a sum of products of one
          lhs and one rhs element, so it is the two rules above composed;
          the same size guard covers an empty contraction.

        **WHY THE ALGEBRA IS VALID, AND EXACTLY WHAT IT IS VALID ABOUT** —
        the B5 follow-up's first item, recorded here beside the rules it is
        about rather than in a log.

        Every rule above is a theorem of the ORDERED FIELD ℝ, and of nothing
        narrower. ℝ's nonzero elements are closed under multiplication and
        under division, and its positive cone is closed under addition, so
        "a product of nonzeros is nonzero" and "two same-signed nonzeros sum
        to that sign" hold with no side condition at all — no magnitude
        enters any of them, which is why none of these rules reads a box
        endpoint and why outward rounding cannot defeat one. The two rules
        that DO carry a side condition carry it for a reason that is also
        algebraic and not numeric: ``reduce_sum`` and ``dot_general`` are
        empty-sum IDENTITIES at size 0, and the identity of ``+`` is 0,
        which is the one element the certificate excludes.

        **THE FLOAT DOUBLE OF THAT FIELD IS NOT CLOSED UNDER
        MULTIPLICATION, AND THAT IS NOT A GAP HERE — IT IS THE REASON THE
        IEEE FACE DOES NOT CALL THIS FUNCTION AT ALL.** binary64's nonzero
        values are NOT closed under ``*``: a product of two nonzeros can
        underflow to zero, and the certificate would then be a false claim
        about the executable. MEASURED, so a lowering change reddens the
        suite rather than the log
        (``tests/test_strict_sign_algebra.py``):

            x  = -1e-120                       (declared, certified -1)
            x*x   = 1e-240                     (this table: +1)
            x*x*x = -0.0                       (this table: -1)

        — in python's binary64 and in ``jnp.float64`` alike, both faces of
        the ``mul`` chain certified nonzero and the third exactly ``-0.0``.
        The real-mode verdict built on that certificate is not wrong: real
        mode's whole posture is that obligations are judged *in exact real
        arithmetic over the declared sets*, and its stamp says so in the
        sentence a reader meets first. What would be wrong is the same
        certificate under ieee semantics, where the value IS ``fl(x*y)``,
        and that is exactly what ``0 if ieee else self._strict_sign_out(...)``
        at this function's one call site refuses — the short-circuit is the
        boundary between "a theorem of ℝ" and "a claim about the program's
        floats", and this paragraph is the argument for why the boundary is
        where it is.
        """
        prim = eqn.primitive
        if prim not in _STRICT_SIGN_PRIMITIVES or len(eqn.outvars) != 1:
            # every member is single-output; a multi-output member would
            # have to say WHICH output its rule speaks about before the
            # caller could write the fact anywhere
            return 0
        sgn = [self.read_strict_sign(a) for a in eqn.invars]
        if prim in ("mul", "div"):
            return sgn[0] * sgn[1] if len(sgn) == 2 else 0
        if prim in ("add", "add_any"):
            return sgn[0] if len(sgn) == 2 and sgn[0] == sgn[1] else 0
        if prim == "neg":
            return -sgn[0]
        if prim in ("abs", "square"):
            return 1 if sgn[0] else 0
        if prim == "integer_pow":
            y = _integer_exponent(params)
            if y is None or not sgn[0]:
                return 0
            return sgn[0] if y % 2 else 1
        if prim == "reduce_sum":
            return sgn[0] if ins and ins[0].size > 0 else 0
        if prim == "dot_general":
            if len(sgn) != 2 or not (sgn[0] and sgn[1]):
                return 0
            if len(ins) != 2 or not (ins[0].size and ins[1].size):
                return 0
            return sgn[0] * sgn[1]
        return 0  # pragma: no cover - the membership test above precedes it

    def top_out(
        self, eqn: ir.JaxprEqn, cause: str = "its result was not modeled"
    ) -> None:
        where = eqn.source_info[-1] if eqn.source_info else "unknown location"
        for out in eqn.outvars:
            self.env[out.id] = _safe_top(out.aval.shape)
            # message provenance only (see __init__): the ⊤ minted here is
            # stelling's own artifact, and every downstream decline that
            # reports this value's box may truthfully name this equation
            # as its origin
            self.top_origin[out.id] = (eqn.primitive, where, cause)
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

    def _name_of_id(self, var_id: int) -> str:
        """How a MESSAGE names the value bound to ``var_id`` IN THE SCOPE
        THIS WALK IS CURRENTLY IN.

        A declared input is named the way the user will meet it again — by
        the published witness name :func:`stelling.coverage.declaration_name`
        mints and :class:`stelling.obligation.SliceInput` emits under — with
        the internal id kept beside it so a reader tracing a note back into
        the IR still can. Anything else is named "IR var {id}", which cannot
        be read as a declaration index.

        For an ARRAY declaration `x{k}` is the PREFIX of the witness's
        names, not one of them: the emission mints `x{k}_{i}` per flat
        element. The message is about the declaration — one box, one
        assumed constraint — so the declaration's name is the right one to
        print, and it is the prefix a reader searches the witness for.

        Audit 0.2.0 B8a, item 5 (M3): these messages used to say
        `var {atom.id}` flat, and that number is NOT the witness's. Measured
        on `aabb58d`, a two-declaration query numbers declaration #0 as IR
        var 1 and declaration #1 as IR var 2 — so an unsatisfiable-assume
        message read "var 2" for the input the witness calls `x1`, and a
        message about declaration #0 read "var 1" for the input the witness
        calls `x0`. Two 0-based namespaces, one reader, nothing relating
        them.

        MESSAGE TEXT ONLY. Nothing here is consulted by a transfer, a
        judgment, a counter or a hash, and the fallback is total: a scope
        the numbering could not cover simply gets the id."""
        if isinstance(self._decl_names, dict):
            name = self._decl_names.get((self._scope_path, var_id))
            if name is not None:
                return f"{name} (IR var {var_id})"
        return f"IR var {var_id}"

    def _name_of(self, atom: ir.Atom) -> str:
        """:meth:`_name_of_id` for an atom that is a ``Var``."""
        return self._name_of_id(atom.id)

    def _quiet_box(self, atom: ir.Atom) -> iv.IntervalArray | None:
        """The atom's box for MESSAGE TEXT only, or None: never raises and
        never appends a note (:meth:`read` notes an undecodable literal,
        and a message-builder must not double it)."""
        if isinstance(atom, ir.Literal):
            try:
                return _value_to_interval(atom.val, atom.aval.shape, atom.aval.dtype)
            except (iv.IntervalError, ir.TranscriptionError):
                return None
        return self.env.get(atom.id)

    def _operand_provenance(self, eqn: ir.JaxprEqn) -> str:
        """Where each operand of a DECLINING equation came from, as a
        bracketed message fragment, or "" (docs/proposed-decline-messages.md
        #2: a decline that reports a box must say where the box came from).

        Message content only — never consulted by a transfer, judgment, or
        counter. Each clause states only what is recorded: a literal is
        quoted with its dtype; a value minted as artifact ⊤ by
        :meth:`top_out` names its originating equation and cause ("it did
        not come from your declaration" is literally true of it — the box
        was minted here, not decoded from any declared bound); a declared
        input names its declaration; anything else names its producing
        equation and propagated span, or is silently omitted — no guessing.
        Equations with more than two operands get only the artifact-⊤
        clauses (the load-bearing ones), so wide forms stay readable.
        """
        clauses = []
        brief = len(eqn.invars) > 2
        for i, atom in enumerate(eqn.invars):
            if isinstance(atom, ir.Literal):
                if brief:
                    continue
                if isinstance(atom.val, (bool, int, float)):
                    clauses.append(
                        f"operand {i} is the literal {atom.val!r} "
                        f"({atom.aval.dtype})"
                    )
                else:
                    clauses.append(
                        f"operand {i} is a {atom.aval.dtype} literal"
                    )
                continue
            origin = self.top_origin.get(atom.id)
            if origin is not None:
                prim, owhere, cause = origin
                if prim == "stelling_any":
                    clauses.append(
                        f"operand {i} is ⊤ because its own declaration "
                        f"declined at {owhere} ({cause})"
                    )
                else:
                    clauses.append(
                        f"operand {i} is stelling's own ⊤ from {prim!r} at "
                        f"{owhere} ({cause}) — it did not come from your "
                        f"declaration; resolve that upstream decline first, "
                        f"this one is downstream of it"
                    )
                continue
            if brief:
                continue
            box = self._quiet_box(atom)
            if box is None:
                continue
            producer = self.producers.get(atom.id)
            if producer is None:
                clauses.append(f"operand {i} spans {_render_box(box)}")
            elif producer.primitive == "stelling_any":
                pwhere = (
                    producer.source_info[-1]
                    if producer.source_info
                    else "unknown location"
                )
                clauses.append(
                    f"operand {i} is the declared input itself (declared "
                    f"at {pwhere}), spanning {_render_box(box)}"
                )
            else:
                pwhere = (
                    producer.source_info[-1]
                    if producer.source_info
                    else "unknown location"
                )
                clauses.append(
                    f"operand {i} was produced by {producer.primitive!r} "
                    f"at {pwhere}, spanning {_render_box(box)}"
                )
        if not clauses:
            return ""
        return " [" + "; ".join(clauses) + "]"

    def _straddle_suffix(self, eqn: ir.JaxprEqn) -> str:
        """The quoted straddle for an interval-undecided obligation whose
        operand is a top-level comparison, appended to the obligation
        detail — or "" when no honest quote exists (a non-comparison
        operand, a cross-scope value): the standard detail then stands
        alone, no guessing (docs/proposed-decline-messages.md #1).

        Message content only, produced at the ONE place the judgment is
        made — every front door (check(), check_contract, direct
        propagate callers, both semantics) inherits it from here, which
        is the one-pipeline principle applied to the quote itself. The
        boxes quoted are the LIVE judged boxes (the same env the verdict
        was computed from — constrain-mode narrowing included), read
        through :meth:`_quiet_box` so no note is ever doubled.

        Every numeric fragment is measured: the spans are the propagated
        intervals; the miss distance is computed exactly (Fraction) and
        marked ≈ when its float rendering rounds; the ulp-step count is
        an exact nextafter walk. When a strict bound's failing endpoint
        EQUALS the bound, the text says that instead (the exactly-stated
        threshold shape). An operand that is stelling's own artifact ⊤
        names its origin — the same top_origin record the decline notes
        use."""
        pred = eqn.invars[0] if eqn.invars else None
        if not isinstance(pred, ir.Var):
            return ""
        prod = self.producers.get(pred.id)
        if (
            prod is None
            or prod.primitive not in _OBL_CMP_SYMBOL
            or len(prod.invars) != 2
        ):
            return ""
        lb, rb = (self._quiet_box(a) for a in prod.invars)
        if lb is None or rb is None or lb.size == 0 or rb.size == 0:
            return ""
        cmp = prod.primitive
        origin = self._cmp_origin_clause(prod)

        def _is_point(b: iv.IntervalArray) -> bool:
            return all(
                lo == hi and math.isfinite(lo)
                for lo, hi in zip(b.los, b.his)
            ) and len(set(b.los)) == 1

        l_pt, r_pt = _is_point(lb), _is_point(rb)
        if l_pt == r_pt:
            # both sides vary (or both are points): quote the straddle
            return (
                f"; the comparison straddles: lhs in {_render_box(lb)} "
                f"{_OBL_CMP_SYMBOL[cmp]} rhs in {_render_box(rb)}{origin}"
            )
        if r_pt:
            vb, k, cmp_v = lb, rb.los[0], cmp
        else:
            vb, k, cmp_v = rb, lb.los[0], _OBL_CMP_FLIP[cmp]
        out = (
            f"; the operand spans {_render_box(vb)} and the asserted "
            f"bound is operand {_OBL_CMP_SYMBOL[cmp_v]} {k}"
        )
        if vb.size == 1 and cmp_v in ("ge", "gt", "le", "lt"):
            if cmp_v in ("ge", "gt"):
                endpoint, word = vb.los[0], "lower"
            else:
                endpoint, word = vb.his[0], "upper"
            if math.isfinite(endpoint):
                from fractions import Fraction

                miss = abs(Fraction(k) - Fraction(endpoint))
                if miss == 0:
                    out += (
                        f"; the operand's {word} endpoint equals the "
                        f"bound, which strict {_OBL_CMP_SYMBOL[cmp_v]} "
                        f"does not admit"
                    )
                elif miss > _MAX_BINARY64:
                    # totality (blinded-lens audit R1): printability is
                    # decided EXACTLY, before converting — float(miss)
                    # raises OverflowError on this class, and the class
                    # is stated as a class, never as a number
                    out += (
                        f"; the operand's {word} endpoint misses the "
                        f"bound by more than the largest finite binary64 "
                        f"value ({float(_MAX_BINARY64)})"
                    )
                else:
                    miss_f = float(miss)
                    approx = "" if Fraction(miss_f) == miss else "≈ "
                    steps = _ulp_steps(endpoint, k)
                    ulp = (
                        f" ({steps} ulp step{'s' if steps != 1 else ''} "
                        f"at this magnitude)"
                        if steps
                        else ""
                    )
                    out += (
                        f"; the operand's {word} endpoint misses the "
                        f"bound by {approx}{miss_f}{ulp}"
                    )
        return out + origin

    def _foreign_top_in_cone(self, atom) -> bool:
        """Whether the judged predicate's cone holds an artifact ⊤ minted by
        anything OTHER than ``reduce_and`` — an unrestricted backward walk,
        unlike :meth:`_membership_reduce_and`'s.

        The hint's dead-end guard, and the scope is the WHOLE PREDICATE, which
        cost two measured defects to learn. Scoped to one reduction's own
        operands it was wrong in both directions:

        * it withdrew CORRECT guidance from a nested ``jnp.all``. Measured,
          ``jnp.all(jnp.all(M >= 0, axis=1))`` has a ``reduce_and`` ⊤ under its
          ``reduce_and`` ⊤ — the one thing the rewrite fixes — and got nothing,
          although deleting both reductions reaches ``discharged`` at 100%
          coverage in both semantics, and the program after ONE application of
          the advice hints. A conjunction holding a nested ``all`` lost the
          hint for its fixable half too.
        * and it kept WRONG guidance where a sibling conjunct was the dead
          end: ``assert_(jnp.all(x >= 0) & (jnp.max(y) >= 0))`` hinted while
          ``assert_(jnp.all(jnp.max(x) >= 0))`` did not — same two primitives,
          same dead end, opposite decision decided by spelling.

        One question over the whole predicate answers both: is every ⊤ in the
        way a ``reduce_and`` ⊤? Deleting the reductions removes exactly those,
        and nothing else, so the hint may claim what it claims iff nothing
        else is there."""
        stack = [atom]
        seen: set[int] = set()
        budget = _MEMBERSHIP_HINT_WALK_CAP
        while stack:
            a = stack.pop()
            if not isinstance(a, ir.Var) or a.id in seen:
                continue
            seen.add(a.id)
            budget -= 1
            if budget < 0:
                # cap exhausted: the evidence is unknown, and the hint is only
                # ever emitted on evidence — report the dead-end shape so the
                # caller stays silent
                return True
            origin = self.top_origin.get(a.id)
            if origin is not None and origin[0] != "reduce_and":
                return True
            producer = self.producers.get(a.id)
            if producer is not None:
                stack.extend(producer.invars)
        return False

    def _membership_reduce_and(self, atom) -> tuple[str, str, str] | None:
        """The ``top_origin`` record of a ``reduce_and`` ⊤ that DELETING would
        fix, reached from a judged predicate through meaning-preserving ops —
        or ``None``.

        The ONE gate behind every membership-idiom emission. It was three
        different instruments: the assume path substring-matched its own
        dropped-conjunct reasons, and the assert/nonvacuity paths read
        ``top_origin`` on the predicate itself. Measured, the three disagreed
        on the canonical two-sided spelling — ``jnp.all(k >= lo) &
        jnp.all(k <= hi)`` hinted as an assume and not as an assert, because
        the judged predicate there is the ``and`` output — and on
        ``keepdims``, which no gate reached. One instrument or the docs
        describe none of them.

        Two conditions, each measured into existence:

        * REACHABLE through :data:`_MEMBERSHIP_HINT_TRANSPARENT` only. Walking
          arbitrary producers would reach the reduction in
          ``jnp.where(jnp.all(...), a, b)``, where it is a selector and
          "delete it" is wrong.
        * ACTIONABLE: no ⊤ anywhere in the judged predicate's cone comes from
          anything but ``reduce_and`` (:meth:`_foreign_top_in_cone`) — the
          whole predicate, not one reduction's operands, and never counting a
          nested ``reduce_and`` against itself.

        Returns the first actionable origin, for the locator; the caller
        supplies the text. The cap is a node budget over the PREDICATE'S CONE,
        not the program: measured, a 407-equation jaxpr whose predicate is one
        ``jnp.all`` still hints, while a predicate that is 300 chained ``&``
        conjuncts does not."""
        if self._foreign_top_in_cone(atom):
            return None
        stack = [atom]
        seen: set[int] = set()
        budget = _MEMBERSHIP_HINT_WALK_CAP
        found: tuple[str, str, str] | None = None
        while stack:
            a = stack.pop()
            if not isinstance(a, ir.Var) or a.id in seen:
                continue
            seen.add(a.id)
            budget -= 1
            if budget < 0:
                return None
            producer = self.producers.get(a.id)
            if producer is None:
                continue
            if producer.primitive == "reduce_and":
                origin = self.top_origin.get(a.id)
                if origin is None:
                    # a reduce_and that did NOT fall to ⊤ (a registered row
                    # landed, or a future scope handled it): nothing to fix
                    continue
                if found is None:
                    found = origin
                continue
            if producer.primitive in _MEMBERSHIP_HINT_TRANSPARENT:
                stack.extend(producer.invars)
        return found

    def _membership_hint_for(self, atom) -> str:
        """:data:`MEMBERSHIP_IDIOM_HINT` the first time this propagation earns
        it, :data:`MEMBERSHIP_IDIOM_POINTER` after that, ``""`` when the gate
        says no.

        The text is in the NOTES at every site rather than in an obligation
        detail because the detail does not survive: measured, escalating the
        same query replaces the propagation's detail wholesale with the solver
        record's own (``escalation declined: primitive 'reduce_and' …``), so a
        detail-only hint would vanish for exactly the reader who tried harder,
        while ``solvers.make_solver_verdict`` carries ``propagation.notes``
        through unchanged."""
        if self._membership_reduce_and(atom) is None:
            return ""
        if self.membership_hint_emitted:
            return MEMBERSHIP_IDIOM_POINTER
        self.membership_hint_emitted = True
        return MEMBERSHIP_IDIOM_HINT

    def _note_membership_idiom(self, eqn: ir.JaxprEqn, subject: str) -> None:
        """The membership hint as a standalone note, for the two faces that
        had no note of their own to ride on.

        The hint existed at ONE call site, the dropped-assume note, while the
        same ⊤ makes ``assert_(jnp.all(...))`` an undecided obligation and
        ``nonvacuity(jnp.all(...))`` an undecided membership condition with
        NOTHING printed (measured: no propagation note at all on either path,
        and the nonvacuity face does not even reach
        :func:`stelling.verdict.undecided_cause_note`, which fires only on an
        undecided *obligation*)."""
        pred = eqn.invars[0] if eqn.invars else None
        origin = self._membership_reduce_and(pred)
        if origin is None:
            return
        where = eqn.source_info[-1] if eqn.source_info else "unknown location"
        head = (
            f"{subject} UNDECIDED at {where}: its predicate is stelling's own "
            f"⊤ from 'reduce_and' at {origin[1]}"
        )
        if any(n.startswith(head) for n in self.notes):
            # two faces of one kind on ONE source line say exactly one thing:
            # same subject, same site, same origin. `_note_decline` dedupes
            # verbatim notes for this reason ("a ~120-word decline note printed
            # verbatim twice was the measured complaint") and the check is the
            # same one — on the head, because the tail differs only by whether
            # this call would have drawn the body or the pointer.
            return
        self.notes.append(head + self._membership_hint_for(pred))

    def _cmp_origin_clause(self, prod: ir.JaxprEqn) -> str:
        """Artifact-⊤ origins of a quoted comparison's sides — the #2
        provenance rule applied to the straddle quote: a quoted unbounded
        side that is stelling's own ⊤ says so, with its origin. Mirrors
        :meth:`_operand_provenance`'s declined-declaration branch
        (blinded-lens audit R4): a ⊤ minted at the side's OWN declaration
        must point at the declaration, never read as 'not
        declaration-derived' — literally-true fragments that send the
        reader away from the one thing to fix are the #2 defect class."""
        parts = []
        for label, atom in zip(("lhs", "rhs"), prod.invars):
            if isinstance(atom, ir.Var):
                origin = self.top_origin.get(atom.id)
                if origin is not None:
                    prim, owhere, cause = origin
                    if prim == "stelling_any":
                        parts.append(
                            f"{label} is ⊤ because its own declaration "
                            f"declined at {owhere} ({cause})"
                        )
                    else:
                        parts.append(
                            f"{label} is stelling's own ⊤ from {prim!r} at "
                            f"{owhere} ({cause}), not a declaration-derived "
                            f"range"
                        )
        return " — " + "; ".join(parts) if parts else ""

    def _note_decline(self, note: str) -> None:
        """Append a TRANSFER-DECLINE note unless the identical note is
        already present (docs/proposed-decline-messages.md: a ~120-word
        decline note printed verbatim twice was the measured complaint).
        Verbatim-identical decline notes say exactly what one says — the
        site and operand provenance are in the text, so notes that differ
        in ANY byte all stay. Scoped to the decline classes deliberately:
        assume/withhold notes keep their multiplicity byte-identically
        (the inert-mode comparability contract pins them). Accounting is
        untouched — the coverage counters are recorded per equation at the
        call sites, never derived from the notes."""
        if note not in self.notes:
            self.notes.append(note)

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
        # each frame carries the primitive whose sub-jaxpr it IS, so a
        # nested body names the INNERMOST primitive that swallowed the
        # obligation: an assert inside a `while` inside a `scan` sits in
        # the `while` body, and telling the reader `'scan'` sends them to
        # the wrong construct (its source location is right, so the two
        # disagree). The outermost name is what the single `eqn.primitive`
        # gave for every depth.
        stack = [(j, eqn.primitive) for j in sub_jaxprs(eqn)]
        while stack:
            j, swallower = stack.pop()
            for e in j.eqns:
                self.counter.record_unreached(e.primitive)
                # THE choke point for "this sub-jaxpr was never analysed",
                # and therefore the one place an obligation inside one can
                # be caught. `scan`, `while` and every other unregistered
                # primitive carrying a body route through here; so do the
                # decline paths (a transfer that refused this form, a
                # refused shape). An assert down there is transcribed —
                # it IS in the IR — and nothing ever propagated a box to
                # its predicate.
                if e.primitive in _UNEXAMINABLE_OBLIGATIONS:
                    self._record_unexamined(e, swallower)
                stack.extend((sj, e.primitive) for sj in sub_jaxprs(e))

    def _reach_key(self, eqn: ir.JaxprEqn) -> tuple:
        """The identity of one DYNAMIC occurrence of an assert equation:
        the chain of (cond, branch) choices that led here, then the
        equation itself. Two occurrences of a shared branch body differ
        in the chain even when they share every var id."""
        return (tuple(self._branch_path), id(eqn))

    def _pinned(self, eqn: ir.JaxprEqn, outs):
        """This declaration's box, collapsed to one point of itself.

        Sound as a witness because the point is a MEMBER of the declared
        set: integer and boolean declarations are pinned to integers, and
        a declaration whose probe point cannot be formed keeps its full
        box (which simply fails to force any downstream branch — the
        certificate is withheld, never faked). Under ieee the subnormal
        haze is re-applied: a hazed point box is a sound hull of both
        flush semantics, so any forcing conclusion drawn from it still
        holds.
        """
        params = eqn.params_dict()
        base = self._pin_ordinal
        pinned = []
        for out, box in zip(eqn.outvars, outs):
            self._pin_ordinal += box.size
            vals = _probe_point(
                self.pin,
                box.shape,
                float(params["lo"]),
                float(params["hi"]),
                out.aval.dtype,
                base,
            )
            if vals is None:
                pinned.append(box)
                continue
            try:
                point = iv.from_values(box.shape, vals)
            except iv.IntervalError:
                pinned.append(box)
                continue
            if self.semantics == "ieee":
                # Format-parametric haze on the pinned point
                out_dtype = out.aval.dtype or ""
                out_fmt = _FLOAT_FORMATS.get(out_dtype)
                if out_fmt is not None:
                    point = iv.subnormal_haze_fmt(
                        point, _ieee_format_min_normal(out_fmt)
                    )[0]
                else:
                    point = iv.subnormal_haze(point)[0]
            pinned.append(point)
        return pinned

    def _record_unexamined(self, eqn: ir.JaxprEqn, swallower: str) -> None:
        """Record an obligation the analysis never looked at.

        A dropped obligation is indistinguishable from an undecided one:
        the user writes a check, the tool never examines it, and the
        verdict says UNKNOWN — or, measured on ``c4133f8`` with one true
        top-level assert beside a false one inside a ``scan``, says
        **VERIFIED**. So the obligation is recorded rather than dropped.
        It is recorded ``unknown``, never ``discharged``: the analysis
        has no evidence in either direction, and an unexamined check must
        not be able to complete a VERIFIED.

        The detail and the note both name the SOURCE LOCATION of the
        assert and the PRIMITIVE that swallowed it, because "unknown"
        alone is exactly the word the reader would misread.
        """
        where = eqn.source_info[-1] if eqn.source_info else "unknown location"
        kind = (
            "obligation"
            if eqn.primitive == "stelling_assert"
            else "nonvacuity condition"
        )
        sink = (
            self.obligations
            if eqn.primitive == "stelling_assert"
            else self.nonvacuity_checks
        )
        sink.append(
            ObligationReport(
                index=len(sink),
                status="unknown",
                detail=(
                    f"NOT EXAMINED: this {kind} sits inside the sub-jaxpr "
                    f"of {swallower!r}, which propagation does not descend "
                    f"into — no box ever reached its predicate. This is not "
                    f"an undecided {kind}, it is an unexamined one"
                ),
                source_info=eqn.source_info,
            )
        )
        self.notes.append(
            f"{kind} at {where} was NOT EXAMINED: it sits inside a "
            f"{swallower!r} sub-jaxpr that no transfer descends into, so "
            f"nothing was ever propagated to its predicate. It is recorded "
            f"unknown rather than dropped — a dropped {kind} is "
            f"indistinguishable from an undecided one, and it would let a "
            f"VERIFIED stand over a check nobody made"
        )

    # -- constraining assume --------------------------------------------------

    def _quiet_interval(self, atom: ir.Atom) -> iv.IntervalArray | None:
        """The atom's current interval for assume classification only:
        never notes, never raises — ``None`` where no interval is
        readable (an undecodable literal, an unbound var). Classification
        failures make the assume inert, which is always sound."""
        if isinstance(atom, ir.Literal):
            try:
                box = _value_to_interval(atom.val, atom.aval.shape, atom.aval.dtype)
            except (iv.IntervalError, ir.TranscriptionError):
                return None
            if self.semantics == "ieee":
                # the same haze read() applies: assume classification must
                # see the interval the judging paths see (a subnormal-band
                # bound is then not a finite point and cannot narrow).
                # Format-parametric: use the literal's own format band.
                lit_dtype = atom.aval.dtype or ""
                lit_fmt = _FLOAT_FORMATS.get(lit_dtype)
                if lit_fmt is not None:
                    box = iv.subnormal_haze_fmt(
                        box, _ieee_format_min_normal(lit_fmt)
                    )[0]
                else:
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
        # ONE ENTRY PER CLASSIFIED CONJUNCT THAT DID NOT NARROW, CARRYING ITS
        # OWN DISPOSITION. This used to be `dropped: list[str]` alongside a
        # parallel `harmless: list[bool]`, with a fail-safe that rebuilt
        # `harmless` as all-False whenever the two lengths disagreed —
        # because `harmless[i]` is only meaningful as the verdict on
        # `dropped[i]`, and a misalignment could mark a RESTRICTING drop
        # harmless, the one direction that produces a wrong REFUTED.
        #
        # The verdict now rides ON the record it is a verdict about, so there
        # are no two lists to come apart and the fail-safe has nothing left to
        # be safe against: an :class:`AssumeDisposition` cannot be handed
        # another conjunct's kind, because it is not addressed by an index at
        # all. This is also what the solver-side release rule reads (see
        # :func:`unaccounted_assumes`) — the same records, not a count taken
        # over them.
        dropped: list[AssumeDisposition] = []
        vacuous: list[str] = []
        self._apply_assumed_pred(eqn.invars[0], where, narrowed, dropped,
                                 vacuous)
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
                        f"assume CONSTRAINED at {where}: narrowed "
                        f"{self._name_of_id(var_id)} to {_render_box(box)}"
                    )
                else:
                    self.notes.append(
                        f"assume CONSTRAINED at {where}: "
                        f"{self._name_of_id(var_id)} is already within the "
                        f"assumed region {_render_box(box)}"
                    )
                self.assumptions.add(
                    f"constrained assume at {where}: "
                    f"{CONDITIONAL_ON_PRECONDITION} — narrowed "
                    f"{self._name_of_id(var_id)} to {_render_box(box)}"
                )
                if not certified:
                    # audit F7: the target's box is an over-approximation
                    # (a transfer output / bracketed const), so a nonempty
                    # meet does NOT certify the true assumed region
                    # nonempty. The narrowing itself is sound (superset of
                    # true-region ∩ reachable) and stays applied; the
                    # conditional claim may be vacuous, so EVERY definite
                    # violation on this run is withheld from REFUTED —
                    # every, not every subsequent: the flag is read once at
                    # the end of the run (an assume is a precondition on
                    # the whole query), never here.
                    self.narrowing_uncertified = True
                    self.notes.append(
                        f"precondition satisfiability UNCERTIFIED at "
                        f"{where}: {self._name_of_id(var_id)} is an "
                        f"over-approximated intermediate (its box may exceed "
                        f"its true image) "
                        f"— the conditional claim may be vacuous"
                    )
                    self.assumptions.add(UNCERTIFIED_NARROWING_ASSUMPTION)
            for i, disp in enumerate(dropped):
                # a conjunction can mix constrainable and inert conjuncts;
                # the un-narrowable part is still a drop and still says so.
                # The hint rides the FIRST such conjunct: it is a fact about
                # the assumed predicate, not about one reason, and this branch
                # reached NO hint at all before — `assume((x >= lo) &
                # jnp.all(x <= hi))` narrows on the first conjunct and dropped
                # the second in silence.
                # "a superset" is the general case and stops being TRUE
                # once the drop is known to be a no-op: a conjunct that is
                # definitely true over the boxes in force excluded nothing,
                # so the set constrained without it IS the assumed region.
                # Saying "superset" there would be a false statement in the
                # one sentence whose job is to say what was given up — and
                # the reader would have no way to tell it from the case
                # where the region really did widen, which is the case that
                # withholds every refutation.
                widened = "a superset"
                if disp.kind == ASSUME_NOOP:
                    widened = (
                        "NOT a superset: this conjunct is definitely TRUE "
                        "over the boxes in force, so it excluded nothing "
                        "and dropping it widened nothing"
                    )
                self.notes.append(
                    f"assume conjunct DROPPED at {where}: constraining "
                    f"proceeded without this conjunct — {widened} "
                    f"({disp.reason})"
                    + (self._membership_hint_for(eqn.invars[0]) if i == 0 else "")
                )
            # a conjunct whose own value is definitely TRUE over the boxes
            # in force restricted nothing, so its absence widened nothing
            # (:meth:`_conjunct_certainly_true`). Only the rest put the
            # assumed region's non-emptiness in doubt. `vacuous` is never
            # harmless: a branch-scoped unsatisfiable conjunct is the
            # region being EMPTY in that branch, which is the very thing
            # being guarded against.
            restricting = [d for d in dropped if d.kind != ASSUME_NOOP]
            if restricting or vacuous:
                # F7's NO-OP HALF, on the MIXED-CONJUNCTION path. The `else:`
                # branch below applies exactly this rule to an assume that
                # dropped WHOLLY; this branch reached it for no conjunct at
                # all, because `narrowed` being non-empty routed the whole
                # assume here — so `assume((x >= lo) & jnp.all(x >= hi))`
                # narrowed on a conjunct that may narrow NOTHING (the note
                # two lines up can read "already within the assumed region")
                # and then refuted over a set the note itself calls "a
                # SUPERSET".
                #
                # The rule is the branch's own: the superset is one-sided.
                # A definite violation over it is a violation at every point
                # of the intended region, which is a refutation only if that
                # region is NON-EMPTY — and a dropped conjunct is exactly the
                # part of the precondition whose satisfiability was not
                # established. Measured on `main` at 9efea6f:
                # `assume((x >= -1.) & jnp.all(x >= 2.))` over x in [-1,1]^3
                # asserting `x > 5.` returned REFUTED with witnesses=(), and
                # the assumed region is EMPTY — the implication is vacuously
                # TRUE. `x >= -1.` IS the declared lower bound; it narrows
                # nothing, and its mere presence flipped a correct UNKNOWN
                # into a wrong REFUTED.
                #
                # Same flag, same reason, as the whole-drop path.
                # `assume_dropped` is the DROP's own mechanism name: it
                # withholds the interval leg's `violated-over-set` and the
                # affine leg's (both through
                # `exactness.certifies_set_refutation`), and it reaches the
                # solver leg's DROPPED_ASSUME_REFUSAL — which the
                # constrained-assume refusal already covers here, but the
                # flag is the record of WHY. It does NOT also set
                # `narrowing_uncertified`: nothing here narrowed an
                # over-approximated intermediate, and a run that quoted that
                # mechanism would be stating a mechanism that did not fire.
                self.assume_dropped = True
                self.notes.append(
                    f"precondition satisfiability UNCERTIFIED at {where}: "
                    f"{len(restricting) + len(vacuous)} conjunct(s) of this "
                    f"assume were dropped, so the narrowed set is a "
                    f"SUPERSET of the assumed region and that region was "
                    f"not shown non-empty HERE — the conditional claim may "
                    f"be vacuous, and every definite violation is withheld "
                    f"from REFUTED unless the run's non-emptiness "
                    f"certificate finds a point of the declared set "
                    f"satisfying every assume of this query"
                )
                self.assumptions.add(UNCERTIFIED_DROP_ASSUMPTION)
        else:
            self.counter.record_inert(eqn.primitive)
            if dropped or not vacuous:
                # an assume whose only content was a branch-scoped
                # unsatisfiable conjunct already carries the directed note
                # above; everything else gets the pre-existing DROPPED
                # disclosure with the reason(s) appended
                reasons = (
                    "; ".join(d.reason for d in dropped) if dropped
                    else "unclassified predicate"
                )
                self.notes.append(
                    ASSUME_DROP_NOTE.format(where=where) + f" ({reasons})"
                    + self._membership_hint_for(eqn.invars[0])
                )
            # F7's NO-OP HALF. The narrowing path sets
            # `narrowing_uncertified` when it constrains an
            # over-approximated variable; a DROPPED assume never
            # reached that branch, so neither the interval withhold nor
            # solvers.py's decline-when-constrained ever engaged — both are
            # conditioned on the assume having TAKEN EFFECT, and an assume
            # that no-ops is invisible to both.
            #
            # A dropped assume means the query ran over a SUPERSET of the
            # intended set, so the disposition is ONE-SIDED, exactly as F7
            # already is (it gates `violated-over-set` only, and nothing
            # withholds `discharged`):
            #
            #   VERIFIED over a superset IMPLIES VERIFIED over the subset
            #                                       -> keep, disclose
            #   REFUTED  over a superset does NOT   -> withhold; the witness
            #                                          may lie outside the set
            #
            # Measured before this: `assume(jnp.all(x >= 0))` over
            # x in [-10, 10]^3 asserting sum(x) >= 0 returned REFUTED with the
            # replay-confirmed witness [0, 0, -1], which violates the dropped
            # precondition. Two-sided would over-fire the way the scatter bar
            # did for its whole history.
            #
            # THE NOTE GATE IS NOT THE FLAG GATE. They were one `if` until a
            # branch-scoped unsatisfiable conjunct walked between them.
            # `_unsatisfiable` degrades inside a cond branch (audit F2: the
            # assume is branch-scoped and the other branch is real), so it
            # appends to `vacuous` and NOT to `dropped` — and `dropped or not
            # vacuous` is then FALSE. Nothing narrowed, the assume was not
            # applied, and neither flag was set: the query refuted over the
            # DECLARED box with the assume contributing nothing.
            #
            # Measured at 9efea6f AND at 3afbf01, x ∈ [-1,1]^3,
            # `cond(x[0] > 0, yes, no)` with `yes: assume(v >= 2.);
            # assert_(v > 5.)` -> REFUTED, witnesses=(); deleting the assume
            # gives the same REFUTED, and the same assume at top level RAISES
            # UnsatisfiableAssumptionError. All three detection sites reach
            # this (empty meet, strict-boundary collapse, definitely-false
            # constant comparison), and the run's own note already says
            # "obligations in this branch may be vacuous under the branch's
            # precondition" while the verdict says REFUTED.
            #
            # The mixed-conjunction path above already spells the rule:
            # `if restricting or vacuous:` — vacuous is never harmless,
            # because a branch-scoped unsatisfiable conjunct IS the assumed
            # region being empty in that branch. Here nothing narrowed at
            # all, so every reason to withhold that the mixed path has, this
            # path has; the flag is unconditional.
            #
            # ONE flag, not two, and the difference is a claim about
            # mechanism: nothing on this path narrowed anything, so
            # `narrowing_uncertified` — "a constraining assume narrowed an
            # over-approximated intermediate" — would be false here. The
            # withholding they both cause is the same withholding; the
            # sentence that explains it is not the same sentence.
            self.assume_dropped = True
        # -- THE LEDGER, WRITTEN ONCE PER ASSUME AND FROM BOTH PATHS ---------
        #
        # Every conjunct this assume produced leaves exactly one entry, and
        # the three sources below are the whole classification: what narrowed,
        # what did not (already carrying its own kind), and what was
        # branch-scoped unsatisfiable. It is deliberately OUTSIDE the
        # `if narrowed:` / `else:` split — a mixed assume writes from both —
        # and deliberately AFTER the flag decisions, so no path can set
        # `assume_dropped` and forget to say which conjunct did it.
        #
        # A `vacuous` conjunct is `dropped`, never `no-op`: it is the assumed
        # region being EMPTY on the branch it was written on, which is the
        # opposite of excluding nothing.
        #
        # `eqn_id` is stamped from the SAME three sources and for the same
        # reason `where` is: it is a property of the ASSUME, not of a conjunct
        # of it. It is what lets :func:`ledger_covers` ask whether the ledger
        # has a record for every assume equation the query CONTAINS — the
        # question a filter over the ledger cannot answer, because an assume
        # nobody classified leaves nothing to filter.
        self.assume_ledger.extend(
            AssumeDisposition(
                kind=ASSUME_APPLIED,
                reason=(
                    f"narrowed {self._name_of_id(var_id)} to "
                    f"{_render_box(box)}"
                ),
                where=where,
                eqn_id=id(eqn),
            )
            for var_id, box, _changed, _certified in narrowed
        )
        # `where` is a property of the ASSUME, not of a conjunct of it, so it
        # is stamped once here rather than threaded through every
        # classification site
        self.assume_ledger.extend(
            dataclasses.replace(d, where=where, eqn_id=id(eqn)) for d in dropped
        )
        self.assume_ledger.extend(
            AssumeDisposition(
                kind=ASSUME_DROPPED,
                reason=f"unsatisfiable within this cond branch ({desc})",
                where=where,
                eqn_id=id(eqn),
            )
            for desc in vacuous
        )

    def _conjunct_certainly_true(self, atom: ir.Atom) -> bool:
        """Whether this assumed conjunct's OWN value is definitely true at
        every point of the boxes in force — in which case dropping it
        removed nothing and introduced no superset.

        A predicate whose box is ``[1, 1]`` on every element is true at
        every point of that box, so ``{x : others ∧ this} == {x : others}``:
        the conjunct is a no-op, the narrowed set is not widened by its
        absence, and a definite violation over that set is a violation at
        every point of the assumed region — a refutation, not a possibly
        vacuous one.

        **What this does NOT rest on.** An earlier version of this
        docstring justified the step by "the propagated boxes
        over-approximate the reachable values". **That premise is false
        here**, and this method is one of the few places it can be. The
        conjuncts of one ``assume`` are classified in sequence and each
        narrowing is written straight back —
        ``self.env[target_atom.id] = new`` in
        :meth:`_classify_assumed_pred` is the one env writer in this file
        that makes a box *smaller* than the values reaching that point.
        A conjunct read after a sibling narrowed its variable is read
        against a box that is deliberately NOT an over-approximation of
        the reachable set.

        **What it does rest on**, in order:

        1. The box still over-approximates the *assumed region*, which is
           the only set the conclusion quantifies over. Every narrowing is
           a meet with the CLOSED half-space (chosen deliberately in
           :meth:`_classify_assumed_pred` — the strict form would
           under-approximate, binding rule 1), so it can only remove
           points that no point of the assumed region occupies.
        2. Emptiness — the case where ``[1, 1]`` is true but useless — is
           closed by an INDEPENDENT gate, not by anything here: the F7
           exactness decision on the ``narrowed`` tuple below
           (``exactness.certifies_nonemptiness(...)``, with audit F8's
           definitely-true half). A narrowing whose target is neither
           exact-declared nor definitely-true over its box sets
           ``narrowing_uncertified``, which withholds every
           ``violated-over-set`` of the run from REFUTED — through
           :func:`stelling.exactness.certifies_set_refutation`, at the end
           of the run, regardless of what this method answers or of where
           in the trace the obligation sits. Measured, three rows: a
           sibling narrowing an
           exact declared input does not raise it; the same narrowing on
           ``y = x * 2.`` does; the same again with the predicate
           definitely true over ``y``'s box does not.

        So an INDETERMINATE box (⊤ included) and a definitely-FALSE one
        both answer no, and reading a transfer output rather than only a
        declaration is safe: a wider box makes ``[1, 1]`` harder to reach,
        never easier.

        Three refusals below. Deleting any one of them reddens no test in
        the suite, and an instrumented full-suite run recorded all 93
        calls this method receives as
        ``(dtype, ieee_flagged, size0, answer)``: ``size0`` is False in
        every one, ``int32`` appears twice and never inside an assume that
        narrowed (so its answer is never read), and every
        ``ieee_flagged=True`` call already answers False on its own.

        **That instrument's population is the SUITE, and the suite has no
        size-0 assume in it** — every size-0 declaration it contains flows
        into an ``assert_`` or a ``jnp.all`` reduction. "0 of 93" is
        therefore a fact about the tests, not about the method, and the
        earlier reading of it here ("measured UNREACHED … never asked")
        was wrong. Re-measured over 21 constructions that DO put a size-0
        value in an assume: 36 of the 38 calls this method receives are
        size-0, and deleting the size-0 refusal changes an outcome (see
        that bullet). The other two refusals' readings survive the
        re-measurement unchanged.

        * **non-bool** (measured unreachable IN EFFECT). ``and`` on
          integer operands is bit arithmetic and its ``[1, 1]`` means the
          integer one — but jax promotes ``bool & int32`` to an *int32*
          ``and``, so a non-bool operand promotes the whole tree and
          :meth:`_classify_assumed_pred` refuses it at the top with
          "'and' on non-bool operands is bit arithmetic, not conjunction",
          before the recursion that fills ``harmless`` is ever entered.
          That top-level refusal is the reachable one and is what
          ``test_a_non_bool_and_is_refused_as_a_conjunction_and_says_so``
          pins; deleting this gate reddens no test in the suite.
        * **maybe-NaN under ieee** (measured DEAD). ``_ieee_cmp`` already
          returns flag ``False`` on comparison outputs and degrades a
          would-be definite TRUE to ``BOOL_UNKNOWN`` whenever an operand
          is flagged, and ``_ieee_bool_logic`` reads a flagged bool as
          ⊤-maybe-NaN. A flagged bool is therefore ``[0, 1]``, never
          ``[1, 1]``, so the final ``all(...)`` would already answer no.
        * **size-0** (measured REACHED, and it moves a verdict). ``all()``
          over no elements is vacuously true; this refusal answers no
          instead, and that answer is read. Measured over the 21-case
          sibling sweep: 36 of 38 calls arrive with a ``bool[0]`` atom,
          and deleting the refusal flips ``assume((k >= 0.5) & (z >=
          2.0)); assert_(k < 0.0)`` — with ``k`` declared ``[-1, 1]`` and
          ``z`` at shape ``(0,)`` — from UNKNOWN to REFUTED in all four
          (vacuity mode × refine) configurations, while reddening no
          test.

          The argument this bullet used to give for unreachability is
          **false**: an ``and``'s operands must broadcast, but broadcast
          forces the zero-element shape onto the OUTPUT, not onto the
          siblings — ``bool[]`` against ``bool[0]`` is a legal pair whose
          result is ``bool[0]``. So a rank-0 sibling of a size-0 conjunct
          is read as a standalone comparison and CAN narrow.

          That is the same defect the vacuous-predicate gate at the top
          of :meth:`_apply_assumed_pred` exists to stop, and with that
          gate in force nothing below a ``bool[0]`` predicate narrows, so
          ``narrowed`` is empty and this method's answer is not read for
          the verdict. The refusal is kept regardless: withholding is
          sound whatever the answer.

          The flipped REFUTED above is, on inspection, SOUND — a size-0
          conjunct forces the whole assumed predicate to ``bool[0]``,
          which admits every point of the declared box, so a witness
          drawn from any narrowed subset is admitted. The refusal
          therefore costs a refutation rather than buying a soundness
          guarantee, which is the opposite of how the sentence above
          used to read.
        """
        if getattr(atom.aval, "dtype", None) != "bool":
            return False
        if self.semantics == "ieee" and self.read_flag(atom):
            return False
        box = self._quiet_interval(atom)
        if box is None or not len(box.los):
            return False
        return all(
            (lo, hi) == iv.BOOL_TRUE for lo, hi in zip(box.los, box.his)
        )

    def _apply_assumed_pred(
        self,
        atom: ir.Atom,
        where: str,
        narrowed: list[tuple[int, iv.IntervalArray, bool, bool]],
        dropped: list["AssumeDisposition"],
        vacuous: list[str],
    ) -> None:
        """Classify one assumed conjunct, and upgrade to :data:`ASSUME_NOOP`
        every disposition it produced that its own value makes a NO-OP.

        THE SUBTREE CLASSIFIES INTO ITS OWN LIST, and only that list is
        upgraded. Not a tidiness choice: the upgrade is the one place a
        conjunct's verdict is written onto records, and giving it a private
        list means there is no index into a shared one, hence no arithmetic
        that could reach a SIBLING's disposition. A conjunct already recorded
        cannot be re-verdicted by a later one, whatever the recursion does
        below.

        The entries in that private list are the ones produced anywhere in
        THIS subtree — this node's own leaf classification, or a descendant's
        if the node is an ``and`` that decomposed. Both are upgraded by the
        same test, and the justification is one argument that does not go
        through the ``and`` transfer's soundness: if ``A ∧ B`` is
        ``[1, 1]`` at every element of the boxes in force then A and B are
        each true at every point of those boxes, so dropping any conjunct of
        that subtree excluded nothing. (The one shape where ``all(A ∧ B)``
        would NOT imply ``all(A)`` is a broadcast that loses an operand
        element, which happens only into a zero-size output — measured over
        all 256 ordered pairs of a 16-shape set at the vacuous-predicate gate
        in :meth:`_classify_assumed_pred` — and a zero-size box is never
        certainly true, because :meth:`_conjunct_certainly_true` refuses an
        empty box.)

        THE VERDICT RIDES ON THE RECORD. This used to extend a parallel
        ``harmless: list[bool]`` and the caller read it by index, with a
        fail-safe for the case where the two lengths disagreed. The verdict
        is now written onto the disposition it is a verdict about, so there
        is no index to misread and a RESTRICTING drop cannot be handed
        another conjunct's "harmless" — the direction that produces a wrong
        REFUTED.

        A :data:`ASSUME_FORWARDED` entry is upgraded too, and keeps its
        ``forwarded_index``. That is deliberate and it is the pre-existing
        rule, not a new one: a relational conjunct that is definitely true
        over the boxes in force excluded nothing, so the judged set is inside
        it whether or not any script emits it, and treating it otherwise
        would withhold refutations on assumes that constrain nothing.
        """
        mine: list[AssumeDisposition] = []
        self._classify_assumed_pred(atom, where, narrowed, mine, vacuous)
        if mine and self._conjunct_certainly_true(atom):
            mine = [
                dataclasses.replace(d, kind=ASSUME_NOOP) for d in mine
            ]
        dropped.extend(mine)

    def _classify_assumed_pred(
        self,
        atom: ir.Atom,
        where: str,
        narrowed: list[tuple[int, iv.IntervalArray, bool, bool]],
        dropped: list["AssumeDisposition"],
        vacuous: list[str],
    ) -> None:
        if _atom_element_count(atom) == 0:
            # THE VACUOUS-PREDICATE GATE. `assume` reads its predicate
            # universally, and a universal over NO elements is true at
            # every point of the declared set: a zero-element predicate
            # admits the whole domain and states nothing about its own
            # subterms. So nothing below this node may narrow, certify
            # satisfiability, or raise the unsatisfiable-precondition
            # oracle — the node is disclosed as an unconstraining drop and
            # the recursion stops here.
            #
            # Without this, `assume((k >= 0.5) & (z >= 2.0))` with `z`
            # declared at shape (0,) narrowed `k` to [0.5, 1.0] — a SUBSET
            # of its declared [-1, 1] — and minted VERIFIED for `k > 0`,
            # while the assume admitted every k including -1. jax
            # broadcasts the rank-0 `k >= 0.5` against the size-0 sibling
            # to `bool[0]`, so the `and`'s truth implies nothing about the
            # rank-0 conjunct, and the `and` recursion below classified it
            # as if standing alone. Eleven constructions of that shape
            # returned VERIFIED over a strict subset (measured against
            # dense independent sampling), and the drop note said "a
            # superset" while the narrowing had gone the other way.
            #
            # The rule is general and the gate is where the rule lives:
            # `all(A & B)` implies `all(A)` only if every element of `A`
            # survives the broadcast into the output, and over numpy/jax
            # broadcasting that fails exactly when the output has zero
            # elements and `A` does not. Measured over all 256 ordered
            # pairs from a 16-shape set (rank 0-3, unit axes, zero axes):
            # 31 pairs lose an operand element, every one of them with a
            # zero-size output, and no size-0 operand ever broadcasts to a
            # nonzero-size output. That second half is why the gate at
            # this ONE node covers the whole tree: a size-0 node forces
            # every ancestor size-0, so the root check and the per-node
            # check cannot disagree, and a leaf is covered by the same
            # line.
            #
            # It also closes the opposite-direction face: a satisfiable
            # `bool[0]` assume whose sibling conjunct has an empty meet
            # (`(k >= 2.0) & (z >= 2.0)` on k in [-1, 1]) used to raise
            # UnsatisfiableAssumptionError — "harness defect; nothing was
            # verified" — about a precondition that is true everywhere.
            dropped.append(_drop(
                f"the assumed predicate has shape "
                f"{tuple(atom.aval.shape)} with zero elements: a universal "
                f"over no elements is true at every point of the declared "
                f"set, so it constrains nothing and licenses no narrowing "
                f"of the values it is written about"
            ))
            return
        if isinstance(atom, ir.Literal):
            dropped.append(_drop(
                f"predicate is a literal ({atom.val!r}), not a traced comparison"
            ))
            return
        producer = self.producers.get(atom.id)
        if producer is None:
            dropped.append(_drop(
                "the predicate's producing equation is not visible in this "
                "scope (constvar, invar, or cross-scope value)"
            ))
            return
        prim = producer.primitive
        if prim == "and":
            # a conjunction holds iff both conjuncts hold: recurse,
            # classifying each independently. jax's `and` is BITWISE — it
            # is the logical connective on bool operands only.
            if any(v.aval.dtype != "bool" for v in producer.invars):
                dropped.append(_drop(
                    "'and' on non-bool operands is bit arithmetic, not "
                    "conjunction"
                ))
                return
            for conj in producer.invars:
                self._apply_assumed_pred(conj, where, narrowed, dropped,
                                         vacuous)
            return
        if prim not in _ASSUME_CMPS:
            dropped.append(_drop(
                f"predicate produced by {prim!r} admits no sound box narrowing"
            ))
            return
        if len(producer.invars) != 2:
            dropped.append(_drop(
                f"comparison {prim!r} with {len(producer.invars)} operand(s)"
            ))
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
            # Check for unsupported float dtypes (those not in the format
            # table). Supported formats (float32, float16, bfloat16, float64)
            # are now handled parametrically.
            bad = [
                d for d in sorted(
                    {v.aval.dtype for v in producer.invars
                     if "float" in (v.aval.dtype or "")}
                )
                if d not in _FLOAT_FORMATS
            ]
            if bad:
                dropped.append(_drop(
                    f"ieee mode: {'/'.join(bad)} comparison semantics are "
                    f"not modelled (unsupported format) — no narrowing, no "
                    f"satisfiability claim"
                ))
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
                dropped.append(_drop(
                    f"point comparison sides do not broadcast ({e})"
                ))
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
                dropped.append(_drop(
                    "a comparison side may be NaN under ieee semantics "
                    "(NaN falsifies the comparison) — the assumed "
                    "comparison is not certified true; dropped"
                ))
                return
            dropped.append(_drop(
                "both comparison sides are point intervals — nothing to "
                "narrow (the assumed comparison is definitely true)"
            ))
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
                dropped.append(_drop(_SUBNORMAL_BOUND_REASON))
            elif _nonfinite_const(a, box_a) or _nonfinite_const(b, box_b):
                dropped.append(_drop(_NONFINITE_BOUND_REASON))
            else:
                # Record for solver forwarding: both operands genuinely
                # vary, so this is a constraint the solver CAN use as an
                # axiom even though the interval domain cannot represent it.
                # Only under real semantics (ieee declines escalation
                # wholly) and only when both sides are ir.Var (not literals).
                #
                # AND ONLY OUTSIDE A COND BRANCH (audit 0.2.0 S5-B1). What is
                # forwarded is a POSITIVE, QUERY-GLOBAL axiom: the solver is
                # told the comparison holds everywhere. Inside a possibly
                # untaken branch that is a claim the program does not make —
                # the assume is branch-scoped and the other branch is real,
                # which is the same reading `_unsatisfiable` already applies
                # one screen up when it degrades a branch-local
                # unsatisfiability instead of raising. Asserting it globally
                # constrains the whole query by a precondition that holds
                # only on the branch, and there is nothing downstream that
                # withholds a discharge obtained that way.
                #
                # The branch-scoped case is NOT emitted as an implication in
                # this build: an implication needs the branch predicate in
                # the slice's own namespace, which the slicer does not carry
                # (it never descends a cond). Saying so in the reason is
                # honest; emitting the antecedent-free axiom is not. The drop
                # still sets `assume_dropped` through the caller, so every
                # definite violation stays withheld — the conservative
                # direction.
                #
                # THE DISPOSITION IS RECORDED HERE, WHERE THE DECISION IS
                # MADE, and it is `forwarded` in exactly the case something
                # was appended to `relational_assumes` — carrying THAT
                # element's index, which is the identity the solver-side
                # release rule joins on. The branch-scoped case takes the
                # ordinary `dropped` disposition like every other drop, and
                # that is the whole repair for the false REFUTED the S5-B1
                # fix above opened: the release rule can no longer be
                # satisfied by a count that quietly stopped including it.
                reason = _RELATIONAL_REASON
                kind, fwd = ASSUME_DROPPED, -1
                if (
                    self.semantics == "real"
                    and isinstance(a, ir.Var)
                    and isinstance(b, ir.Var)
                ):
                    if self.branch_depth:
                        reason = (
                            f"{_RELATIONAL_REASON}; {_BRANCH_SCOPED_REASON}"
                        )
                    else:
                        kind = ASSUME_FORWARDED
                        fwd = len(self.relational_assumes)
                        self.relational_assumes.append(
                            RelationalAssume(
                                scope=self._scope_path,
                                eqn=producer,
                                where=where,
                            )
                        )
                dropped.append(AssumeDisposition(
                    kind=kind, reason=reason, forwarded_index=fwd,
                ))
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
            dropped.append(_drop(
                "the comparison bound may be NaN under ieee semantics "
                "(its producer carries maybe-NaN) — no certified "
                "half-space represents it"
            ))
            return
        if not isinstance(target_atom, ir.Var):
            dropped.append(_drop(
                "the varying comparison side is a literal, not an "
                "environment variable"
            ))
            return
        if target_box is None:
            dropped.append(_drop(
                f"{self._name_of(target_atom)} has no propagated interval "
                f"to narrow"
            ))
            return
        # bound shape: a scalar broadcast over the variable, or an exact
        # elementwise match — nothing else (general broadcasting between
        # the bound and the variable is not attempted; inert is sound).
        if not (bound.is_scalar() or bound.shape == target_box.shape):
            dropped.append(_drop(
                f"bound shape {bound.shape} is neither scalar nor equal to "
                f"the constrained variable's shape {target_box.shape}"
            ))
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
                f"{self._name_of(target_atom)} ∈ "
                f"{_render_box(target_box)} cannot satisfy "
                f"{_CMP_SYMBOL[cmp]} {_render_bound(bound)} (empty meet)",
                vacuous,
                f"unsatisfiable assume at {where}: "
                f"{self._name_of(target_atom)} has propagated interval "
                f"{_render_box(target_box)}, but the assumed constraint "
                f"requires {self._name_of(target_atom)} "
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
                f"strict constraint {self._name_of(target_atom)} "
                f"{_CMP_SYMBOL[cmp]} {_render_bound(bound)} collapses onto "
                f"the boundary point "
                f"— the true region is empty",
                vacuous,
                f"unsatisfiable assume at {where}: the strict constraint "
                f"{self._name_of(target_atom)} {_CMP_SYMBOL[cmp]} "
                f"{_render_bound(bound)} collapses the narrowed interval "
                f"onto the closed boundary point {_render_box(new)}, which "
                f"the strict comparison itself excludes — the true assumed "
                f"region is empty, so the precondition is definitely false "
                f"on the whole over-approximated domain and every "
                f"downstream obligation would be vacuous (harness defect; "
                f"nothing was verified)",
            )
            return
        # IEEE strict-inequality auto-bump: in ieee mode, the value set IS
        # the format's representable floats, and there is no value between
        # k and the next representable value above/below k in the format.
        # So x > k genuinely means x >= nextafter(k) in that format, and
        # we can bump the closed boundary to exclude k EXACTLY. This is
        # UNSOUND in real mode (reals in (k, nextafter(k)) would be
        # excluded) and is only applied when the target has a known float
        # format. The bump uses _format_nextafter which handles all k
        # values correctly, including k=0.
        if self.semantics == "ieee" and cmp in ("gt", "lt"):
            target_dtype = target_atom.aval.dtype or ""
            target_fmt = _FLOAT_FORMATS.get(target_dtype)
            if target_fmt is not None:
                if cmp == "gt":
                    bumped_los = tuple(
                        _format_nextafter(k, +1, target_fmt)
                        if lo == k else lo
                        for lo, k in zip(new.los, ks)
                    )
                    if bumped_los != new.los:
                        new = iv.IntervalArray(
                            shape=new.shape, los=bumped_los, his=new.his
                        )
                else:  # lt
                    bumped_his = tuple(
                        _format_nextafter(k, -1, target_fmt)
                        if hi == k else hi
                        for hi, k in zip(new.his, ks)
                    )
                    if bumped_his != new.his:
                        new = iv.IntervalArray(
                            shape=new.shape, los=new.los, his=bumped_his
                        )
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
        if self.semantics == "real" and cmp in ("gt", "lt"):
            # THE FIRST OF THE TABLE'S TWO WRITERS. **THIS SENTENCE READ
            # "THE ONE SOURCE of the strict-sign certificate", AND IT WAS
            # STALE AGAINST BOTH ITS NEIGHBOURS**: the block comment on
            # `self.strict_sign` already said "TWO WRITERS" and "the
            # CERTIFICATE has three sources and the TABLE has two", and
            # the constvar writer below already called itself "THE SECOND
            # SOURCE". One table, two writers — this one and the constvar
            # bind in :meth:`run`; one certificate, three sources — those
            # two plus :func:`_literal_strict_sign`, which recomputes a
            # literal's sign on every read and stores nothing, so it is a
            # source of the certificate and not of the table. (Under
            # ``boundary="transparent"`` a THIRD entry point re-keys an
            # already-minted certificate across a sub-jaxpr boundary; it
            # mints nothing, so "one certificate, three sources" is
            # unchanged and the writer census on `self.strict_sign` says
            # what separates a minting writer from a copying one.) The
            # closed meet just written cannot represent `x > k`; this records the
            # part of it the box lost, and only the part that is a
            # NONZERO-ness claim: `x > k` with `k >= 0` gives `x > 0`
            # (transitively, and for every element, since `ks` is the
            # per-element bound), `x < k` with `k <= 0` gives `x < 0`.
            # A bound of the wrong sign — `assume(x > -5)` — records
            # nothing, which is right: it excludes no zero.
            #
            # LATER ASSUMES CANNOT INVALIDATE IT. Narrowing is a meet, so
            # every subsequent constraint on this var only shrinks its box,
            # and the assumed region is the CONJUNCTION of all of them — a
            # fact true of the region at this point is true of every
            # sub-region after it. That is why the record needs no
            # re-validation downstream, and why it is keyed by var id
            # rather than by box.
            #
            # It is NOT gated on `exactness.certifies_nonemptiness`, and
            # that is deliberate: the claim a certified sign licenses is
            # "the quotient is bounded WHERE THE PRECONDITION HOLDS", which
            # is what a VERIFIED under a constraining assume already says
            # in its stamp. Whether that region is inhabited is the vacuity
            # question, decided by the machinery above and disclosed
            # separately; it is not this fact's job and folding it in here
            # would answer it twice, inconsistently.
            if cmp == "gt" and all(k >= 0.0 for k in ks):
                self.strict_sign[target_atom.id] = 1
            elif cmp == "lt" and all(k <= 0.0 for k in ks):
                self.strict_sign[target_atom.id] = -1
        if self.semantics == "ieee" and self.nan.get(target_atom.id):
            # an assumed-true comparison excludes NaN (NaN would falsify
            # it), so the narrowed target's maybe-NaN flag is soundly
            # cleared — the clearing is a judgement call the spec flags;
            # disclosed here so no verdict rests on it silently
            self.nan[target_atom.id] = False
            self.notes.append(
                f"assume cleared maybe-NaN on "
                f"{self._name_of(target_atom)} at "
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
        self,
        jaxpr: ir.Jaxpr,
        consts,
        args,
        arg_flags=None,
        arg_taints=None,
        arg_signs=None,
    ) -> list[iv.IntervalArray]:
        if self._decl_names is _UNSET:
            # the FIRST `run` is the outermost jaxpr on every entry point
            # (`propagate`, `interval_env`, the probe runs); every later one
            # is a descent this walk performs itself. Derived once, from the
            # IR and from nothing a caller supplies.
            self._decl_names = _declaration_names(jaxpr)
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
                box = _value_to_interval(c, var.aval.shape, var.aval.dtype)
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
                # so mark_if_point below withholds exactness by itself.
                # Format-parametric: use the var's own format band.
                var_dtype = var.aval.dtype or ""
                var_fmt = _FLOAT_FORMATS.get(var_dtype)
                if var_fmt is not None:
                    box = iv.subnormal_haze_fmt(
                        box, _ieee_format_min_normal(var_fmt)
                    )[0]
                else:
                    box = iv.subnormal_haze(box)[0]
            self.env[var.id] = box
            # exact iff the decoded box is a point per element — a >2**53
            # int decodes to a genuine bracket, which is NOT its value set
            self.exact.mark_if_point(var.id, box.los, box.his)
            if not ieee:
                # THE SECOND SOURCE of the strict-sign certificate, and it
                # is the same source as the literal one: a CONSTVAR's box
                # IS its value, so its sign is as certifiable as a
                # literal's. It has to be written here because a constant
                # only reaches the rules as a Var — an ARRAY constant is
                # traced as a constvar, never as a Literal, so
                # `jnp.array([1.,2.]) * x` would otherwise drop the chain
                # while the scalar `2.0 * x` kept it. Skipped for the
                # pre-boxed IntervalArray branch above, which `continue`s
                # before this: that box is of unknown provenance and is
                # NOT a value.
                sgn = _box_strict_sign(box)
                if sgn:
                    self.strict_sign[var.id] = sgn
        for i, (var, a) in enumerate(zip(jaxpr.invars, args)):
            self.env[var.id] = a
            if ieee and arg_flags is not None and arg_flags[i]:
                self.nan[var.id] = True
            # taint crosses scope boundaries with the value it marks
            if ieee and arg_taints is not None and arg_taints[i]:
                self.taint[var.id] = True
            # THE IN DIRECTION OF THE BOUNDARY CARRY, and the only place
            # the strict-sign table is written from outside this scope.
            # ``arg_signs`` is None unless the descent that called this
            # asked for the carry, and the ONE place that decision is
            # made is :meth:`_carries_signs` — so the mode test and the
            # real-mode restriction are not re-spelled here. The licence
            # for each caller is written at that caller: the
            # unconditional-wrapper argument at the transparent arm of
            # :meth:`eqn`, the restriction-to-a-subset argument at its
            # `cond` arm.
            #
            # 0 means "unknown" and is never written: the table's
            # absent-is-unknown reading is what makes a missing entry
            # conservative, and an explicit 0 would be a second spelling
            # of it.
            if arg_signs is not None and arg_signs[i]:
                self.strict_sign[var.id] = arg_signs[i]
                self.boundary_crossings += 1
        # assume classification looks up the predicate's producing equation
        # at the CURRENT jaxpr level only; sub-jaxpr runs (transparent
        # wrappers, cond branches) get their own map, restored on exit —
        # same scoping discipline as the env swap the callers perform.
        prev_producers = self.producers
        self.producers = {
            out.id: e for e in jaxpr.eqns for out in e.outvars
        }
        try:
            # the POSITION rides with the equation: it is the only part of a
            # scope path that this level can supply, and a descent one frame
            # down needs it to address the scope it is entering
            # (:data:`ScopePath`).
            for pos, eqn in enumerate(jaxpr.eqns):
                self.eqn(eqn, pos)
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
            # ONE PREDICATE FOR BOTH ATOM KINDS. The literal arm above has
            # asked `iv.check_shape` (through `_refused_value_problem`)
            # since it was written; this arm re-implemented the same two
            # tests by hand, and the copy was wrong in both of the ways
            # audit 0.2.0 B6 audit 3 closed one module over (item 2 of B8a):
            #
            #   * it caught only `TypeError`, and `operator.index` raises
            #     whatever `__index__` raises — measured on `aabb58d`, an
            #     extent whose `__index__` raises `ValueError` left
            #     `interval_env()` and `propagate()` RAW, past every
            #     `except iv.IntervalError` in the library;
            #   * it interpolated `{d!r}` UNGUARDED, inside the screen's
            #     own message path — so a hostile `__repr__` raised
            #     `RuntimeError` out of the code that was trying to explain
            #     why it refused the object. Measured the same way.
            #
            # `iv.check_shape` catches `Exception` and quotes both the
            # shape and the extent through `iv._safe_repr`, so a second
            # spelling of this predicate is a second thing to keep total.
            try:
                iv.check_shape(atom.aval.shape)
            except iv.IntervalError as e:
                return f"carries a refused shape ({e})"
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

    def eqn(self, eqn: ir.JaxprEqn, pos: int = 0) -> None:
        """Judge one equation. ``pos`` is this equation's index in its own
        scope's ``eqns`` list — the address component a descent from here
        needs to name the scope it enters (:data:`ScopePath`)."""
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
            # coverage.call_body, the canonical accessor: a wrapper's body
            # is a bare Jaxpr for remat2 on jax 0.10 and a ClosedJaxpr on
            # 0.11, and that is a fact about the series, not the callee.
            inner = call_body(eqn)
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
                outer_scope = self._scope_path
                # var ids are unique per JAXPR, not per transcription, so
                # the strict-sign table is scope-swapped exactly like the
                # env it annotates.
                #
                # **THIS COMMENT USED TO END:** *"Nothing is carried IN or
                # OUT across the boundary: an inner value inherits no
                # certificate from its caller and hands none back, which
                # loses precision on a `remat`-wrapped division and cannot
                # lose soundness."* That was true of every release up to
                # and including 0.2.1, and it is still true under the
                # ``boundary="opaque"`` DEFAULT, which is what
                # :meth:`_carries_signs` returns False for. It is no
                # longer true of the whole function, so it does not stand
                # as an unqualified sentence.
                #
                # UNDER ``boundary="transparent"`` THE CERTIFICATE CROSSES
                # THIS BOUNDARY IN BOTH DIRECTIONS, AND HERE IS THE WHOLE
                # LICENCE FOR IT — it is short, and a reader who cannot
                # find it cannot audit the change.
                #
                # The certificate says: *every element of this value is
                # certainly > 0 (resp. < 0) at every point of the assumed
                # region*. It is a claim about a VALUE over a REGION, and
                # this boundary changes neither.
                #
                #   * The wrapper is UNCONDITIONAL. `jit`, `remat2`,
                #     `custom_jvp_call` and `custom_vjp_call` all run their
                #     body exactly when the equation runs; there is no
                #     predicate, no branch, and nothing that can make the
                #     body execute on a subset of the region the caller is
                #     being analysed over. Contrast `cond` below, where
                #     precisely that is what happens and why the OUT
                #     direction is refused there.
                #   * The inner invar IS the outer operand, as a real
                #     value — that is what the positional binding two
                #     lines down means, and it is the same identity the
                #     boxes, the maybe-NaN flags and the product-taints
                #     already cross on. So a fact true of the outer
                #     operand's value is true of the inner invar's value,
                #     because they are one value.
                #   * The inner outvar IS the outer outvar, by the same
                #     positional identity, so the converse holds and the
                #     fact may come back.
                #   * The REGION is unchanged in both directions: the
                #     assumed region is a property of the top-level
                #     declarations and the assumes that narrowed them, and
                #     descending an unconditional wrapper narrows nothing.
                #
                # What is NOT carried, deliberately: `self.exact` (the
                # :class:`stelling.exactness.ExactSet`) is swapped fresh
                # below and stays that way. It governs WITHHOLDING —
                # whether a definite violation may become a REFUTED — so
                # carrying it would widen what can be refuted, which is
                # the dangerous direction and a different argument from
                # this one. Out of scope here by decision, not by
                # oversight.
                outer_sign = self.strict_sign
                # ONE read, before the table is swapped, of the OUTER
                # operands' certificates — positional, exactly like
                # `in_flags` and `in_taints` above. None (never an empty
                # list) when nothing is carried, so `run` has one test to
                # make and not two.
                in_signs_cert = (
                    [self.read_strict_sign(a) for a in eqn.invars]
                    if self._carries_signs()
                    else None
                )
                self.env = {}
                self.exact = exactness.ExactSet()
                self.nan = {}
                self.taint = {}
                # the ids bound below belong to THIS scope and to no other:
                # the path is what lets a relational assume recorded down
                # here be resolved later, and what stops it being resolved
                # against a namespace it does not belong to.
                self._scope_path = outer_scope + (("call", pos),)
                self.strict_sign = {}
                out_flags = None
                out_taints = None
                out_signs = None
                try:
                    outs = self.run(
                        inner.jaxpr,
                        inner.consts,
                        ins,
                        in_flags,
                        in_taints,
                        in_signs_cert,
                    )
                    if ieee:
                        out_flags = [
                            self.read_flag(o) for o in inner.jaxpr.outvars
                        ]
                        out_taints = [
                            self.read_taint(o) for o in inner.jaxpr.outvars
                        ]
                    if in_signs_cert is not None:
                        # read INSIDE the try, while the inner table is
                        # still installed — the `finally` below restores
                        # the outer one, and a read after it would be a
                        # read of the caller's table wearing the callee's
                        # name. Same shape, and the same reason, as
                        # `out_flags` above.
                        out_signs = [
                            self.read_strict_sign(o)
                            for o in inner.jaxpr.outvars
                        ]
                finally:
                    self.env = outer_env
                    self.exact = outer_exact
                    self.nan = outer_nan
                    self.taint = outer_taint
                    self._scope_path = outer_scope
                    self.strict_sign = outer_sign
                for j, (out, val, iout) in enumerate(
                    zip(eqn.outvars, outs, inner.jaxpr.outvars)
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
                        # THE OUT CARRY, in the arm that binds a real
                        # value and not in the one that binds a refused
                        # stand-in. A refused outvar's box is a ⊤
                        # stand-in that stands for no value at all, so a
                        # sign written onto it would be a certificate
                        # about nothing; bind and certify stay one
                        # operation here for the same reason bind and
                        # register do.
                        if out_signs is not None and out_signs[j]:
                            self.strict_sign[out.id] = out_signs[j]
                            self.boundary_crossings += 1
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
            outer_scope = self._scope_path
            # branch-scoped like the env, and for a second reason on top of
            # id uniqueness: an assume inside a possibly-untaken branch
            # constrains that branch's precondition only, so a certificate
            # it mints must not outlive the branch. The join below hands
            # the cond's outputs no certificate at all.
            #
            # **A `cond` CARRIES IN AND NEVER OUT, AND THE ASYMMETRY IS THE
            # POINT.** Under ``boundary="transparent"``
            # (:meth:`_carries_signs`) the operands' certificates are bound
            # onto each branch's invars below, and nothing is read back.
            #
            # IN is sound by the transparent-wrapper argument
            # (:meth:`eqn`'s call arm carries it in full), RESTRICTED: a
            # branch body is analysed over the subset of the assumed region
            # on which the index selects that branch, and a fact true at
            # EVERY point of the region is true at every point of a subset.
            # The value identity is the same positional one — the branch
            # invar IS the operand — and the operands are read once, above,
            # in the caller's scope.
            #
            # OUT IS NOT SOUND WITHOUT WORK THAT IS NOT DONE HERE, and this
            # is the false-VERIFIED trap the whole change has to avoid. A
            # certificate MINTED inside a branch — by an `assume` written in
            # the branch body — is conditional on that branch being taken.
            # Handing it to the cond's outputs would let a branch-local
            # precondition license a conclusion OUTSIDE the branch, which is
            # exactly what the branch-scoping machinery already in this
            # function exists to prevent (:data:`_BRANCH_SCOPED_REASON`, and
            # a ``branch_depth`` that is inherited rather than reset).
            #
            # And the obvious repair — "carry out only what came in" — needs
            # a fact the table cannot hold: its value is an ``int``, so
            # ``+1`` from the caller and ``+1`` minted by a branch-body
            # assume are the same entry. Distinguishing them is a PROVENANCE
            # question and provenance is not in this data structure. So the
            # OUT direction is refused here, as a decision with a reason
            # rather than as an omission, and a sound version of it is its
            # own scoped change with its own audit.
            outer_sign = self.strict_sign
            # positional over `operands` — `eqn.invars[1:]`, i.e. the same
            # slice `op_flags`/`op_taints` take, because the index is
            # invar 0 and is not a branch operand. Read ONCE here, in the
            # caller's scope, and re-bound for every possible branch.
            op_signs = (
                [self.read_strict_sign(a) for a in eqn.invars[1:]]
                if self._carries_signs()
                else None
            )
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
            #
            # `unforced_depth` is the OTHER counter, and it is not the same
            # question. A singleton `possible` means the index box admits
            # exactly one branch, so every point of the declared set that
            # reaches this cond takes it: that branch runs whenever the cond
            # runs, and a definite violation inside it refutes. A `possible`
            # with two or more members means the analysis ADMITS each of
            # them and has certified none — `x - x > 0` is admitted both
            # ways while the guard is false at every point of the box — so a
            # violation found inside is withheld from REFUTED unless a
            # witness certifies the branch (see _reachability_witnesses).
            forced = len(possible) == 1
            self.branch_depth += 1
            if not forced:
                self.unforced_depth += 1
            try:
                for i in sorted(possible):
                    b = branches[i]
                    self.env = {}
                    self.exact = exactness.ExactSet()
                    self.nan = {}
                    self.taint = {}
                    # a branch scope is TAGGED differently from a call scope,
                    # so a branch's ids can never be resolved through a call's
                    # rename even if the two sit at the same equation index.
                    # The slicer never descends a cond, so this step never
                    # appears in its map and any path containing one resolves
                    # to nothing — which is the right answer, not a near miss.
                    #
                    # NO FORWARDED ASSUME EVER CARRIES ONE OF THESE STEPS, and
                    # this comment used to read as though some might.
                    # `branch_depth` is incremented ABOVE, before this line
                    # runs, and the forwarding site refuses while it is
                    # nonzero — so the resolution described in the previous
                    # paragraph is a property of a path that
                    # `RelationalAssume` cannot hold. It is kept because the
                    # TAGGING is what makes that true by construction rather
                    # than by the guard alone: if a later batch forwards
                    # branch-scoped assumes as implications, the path it hands
                    # the slicer will already be un-confusable with a call's.
                    self._scope_path = outer_scope + (("cond", pos, i),)
                    self.strict_sign = {}
                    self._branch_path.append((id(eqn), i))
                    try:
                        results.append(
                            self.run(
                                b.jaxpr, list(b.consts), operands, op_flags,
                                op_taints, op_signs,
                            )
                        )
                    finally:
                        self._branch_path.pop()
                    if ieee:
                        branch_flags.append(
                            [self.read_flag(o) for o in b.jaxpr.outvars]
                        )
                        branch_taints.append(
                            [self.read_taint(o) for o in b.jaxpr.outvars]
                        )
            finally:
                self.branch_depth -= 1
                if not forced:
                    self.unforced_depth -= 1
                self.env = outer_env
                self.exact = outer_exact
                self.nan = outer_nan
                self.taint = outer_taint
                self._scope_path = outer_scope
                self.strict_sign = outer_sign
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
            if self.pin is not None and eqn.invars:
                # PROBE RUN: read the predicate's box at the pinned point,
                # BEFORE `_assume_constrain` can meet anything into it. A
                # box of [1, 1] on every element means the predicate is
                # true at this point — the box encloses the true value, so
                # a definite TRUE over the box is a TRUE at the value.
                #
                # Before, not after, and the ordering is not defensive
                # tidiness: `_classify_assumed_pred` writes narrowed
                # boxes straight back into the env, and a later conjunct
                # read against a narrowed sibling is read against a box
                # that is deliberately NOT an over-approximation of the
                # reachable set (see `_conjunct_certainly_true`). Reading
                # here keeps every witness answer a statement about the
                # PINNED POINT and nothing else. (In fact a
                # definitely-true predicate narrows nothing — the meet
                # with the closed half-space is a no-op exactly when the
                # comparison is definitely true — so on the runs that
                # certify, the two orders agree; the runs where they
                # differ are the runs that certify nothing.)
                key = id(eqn)
                true_here = self._conjunct_certainly_true(eqn.invars[0])
                self.assume_witness[key] = (
                    self.assume_witness.get(key, True) and true_here
                )
            if self.assume_mode == "constrain" and eqn.invars:
                self._assume_constrain(eqn, where)
            else:
                # inert (amendment 2): sound, counted, addressed — never
                # silent, never "known". Byte-identical to the MVP note, plus
                # the membership hint on the same gate every other face uses:
                # assume_mode="inert" is a SUPPORTED mode, and a reader who
                # wrote `jnp.all` there was the only one getting nothing while
                # the doc said all three paths print the rewrite.
                self.counter.record_inert(eqn.primitive)
                self.notes.append(
                    ASSUME_DROP_NOTE.format(where=where)
                    + (
                        self._membership_hint_for(eqn.invars[0])
                        if eqn.invars
                        else ""
                    )
                )
                # THE LEDGER MUST BE TOTAL OVER THE ASSUMES THE WALK SEES,
                # and this arm sees one and classifies nothing. An empty
                # ledger has to mean "no assume", never "an assume nobody
                # wrote down" — the second reading is what makes a release
                # rule default open. `assume_mode="inert"` never sets
                # `assume_dropped`, so today nothing downstream consults
                # this entry; recording it is what keeps that a fact about
                # the gate rather than about the ledger.
                self.assume_ledger.append(AssumeDisposition(
                    kind=ASSUME_DROPPED,
                    reason=(
                        "assume_mode='inert': the predicate was not "
                        "classified at all"
                    ),
                    where=where,
                    eqn_id=id(eqn),
                ))
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
            self.top_out(eqn, cause="no interval transfer is registered for it")
            return

        transfer, tier = entry
        ins = [self.read(a) for a in eqn.invars]
        in_flags = [self.read_flag(a) for a in eqn.invars] if ieee else None
        try:
            if ieee and tier == TIER_SOUND_LIBM:
                # THE FIFTH ARGUMENT, and which transfers get it is read
                # off the TIER rather than off a list of primitive names:
                # a transfer that rides a libm accuracy claim is exactly
                # one registered at `TIER_SOUND_LIBM`, and the next one
                # registered there becomes budget-gated by being
                # registered. `_assert_libm_transfers_take_a_budget`
                # refuses the import if the two conventions ever disagree.
                result = transfer(eqn, params, ins, in_flags, self.libm_budget)
            elif ieee:
                result = transfer(eqn, params, ins, in_flags)
            elif eqn.primitive in _REAL_TRANSFERS_READING_STRICT_SIGN:
                # the fourth argument is the real-mode counterpart of
                # `in_flags`: a per-operand certificate the transfer cannot
                # recover from its boxes (see `self.strict_sign`)
                result = transfer(
                    eqn,
                    params,
                    ins,
                    tuple(self.read_strict_sign(a) for a in eqn.invars),
                )
            else:
                result = transfer(eqn, params, ins)
        except iv.IndexOutOfBoundsError as e:
            # A FINDING, not a decline. Every other exception arriving here
            # means stelling has no rule for a legal form; this one means
            # the PROGRAM indexes outside its array for every input the
            # user declared, and stelling proved it over the whole set.
            # Reporting it through `_note_decline`'s wording — which is
            # what happened before this arm existed — told the reader "no
            # rule here" about a fact that has nothing to do with rules.
            #
            # The accounting is deliberately IDENTICAL to the decline
            # below: ⊤, unknown, unreached. A definite out-of-bounds index
            # does not make any asserted predicate false, so it must not
            # manufacture a REFUTED, and withholding the value is the only
            # safe direction. What changes is only what the reader is told
            # — and notes are the loudest channel a transfer has without a
            # new Stamp field (which would mean editing the solver
            # verdict's assembly site too).
            #
            # `self.notes` rather than `_note_decline`: findings keep their
            # multiplicity, one per equation site, the way the withhold and
            # assume notes do. Two out-of-bounds indexes at two places in a
            # program are two things to fix, and deduping byte-identical
            # text would hide the second whenever the two sites share a
            # message.
            where = (
                eqn.source_info[-1] if eqn.source_info else "unknown location"
            )
            self.notes.append(
                f"OUT-OF-BOUNDS INDEX (definite) in {eqn.primitive!r} at "
                f"{where}: {e}{self._operand_provenance(eqn)}; ⊤"
            )
            self.counter.record_unknown(eqn.primitive)
            self.mark_unreached(eqn)
            self.top_out(eqn, cause="its index is out of bounds for every declared input")
            return
        except iv.IntervalError as e:
            # a transfer whose domain doesn't cover this legal form (rank
            # broadcasting, batched selectors, …) DECLINES: sound ⊤
            # degradation with the reason quoted — the registered
            # degrade-don't-crash posture (second audit, FRAGILE 5; the
            # shape guards previously killed the whole analysis here).
            # Under ieee, top_out marks the outputs maybe-NaN.
            # The note names the SITE and the OPERANDS' provenance
            # (docs/proposed-decline-messages.md #2): a decline that quotes
            # a box says where the box came from — an upstream artifact ⊤
            # is named as stelling's own, never left to read as the user's
            # declaration. Message content only; same decline, same counts.
            where = (
                eqn.source_info[-1] if eqn.source_info else "unknown location"
            )
            self._note_decline(
                f"{eqn.primitive!r} declined this form at {where}: {e}"
                f"{self._operand_provenance(eqn)}; ⊤"
            )
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
            self.top_out(eqn, cause="its interval transfer declined this form")
            return
        if result is None:  # a known transfer declining this configuration
            where = (
                eqn.source_info[-1] if eqn.source_info else "unknown location"
            )
            self._note_decline(
                f"{eqn.primitive!r} has no sound rule for params "
                f"{ {k: v for k, v in params.items() if not isinstance(v, ir.ClosedJaxpr)} }"
                f" at {where}{self._operand_provenance(eqn)}; ⊤"
            )
            self.counter.record_unknown(eqn.primitive)
            # same accounting as the IntervalError decline above: inner
            # equations of a declined form count unreached, keeping the
            # denominator outcome-independent (third audit, F1)
            self.mark_unreached(eqn)
            self.top_out(
                eqn, cause="it has no sound rule for this configuration"
            )
            return
        outs, out_flags = result if ieee else (result, None)
        if self.pin is not None and eqn.primitive == "stelling_any":
            # PROBE RUN: the declaration's box collapses to one point OF
            # THAT BOX. Everything downstream is then a bracket of the
            # program's values at that point, which is what lets a cond
            # with a singleton index box certify its branch. Applied to
            # the transfer's output rather than by rewriting the query so
            # the content hash, the coverage counts and every other
            # instrument see the query the caller actually wrote.
            outs = self._pinned(eqn, outs)
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
        if (
            not ieee
            and self.pin is None
            and eqn.primitive in _SUBNORMAL_TELL_ROWS
        ):
            # THE REAL-MODE SUBNORMAL TELL. It reads `outs` and writes a
            # note; it touches no box, no counter, no tier and no flag, so
            # every verdict this run would have produced it still produces.
            # `self.pin is None` skips PROBE runs (the region and
            # reachability certificates), whose notes are discarded — a note
            # nobody reads is only cost. The predicate and its silence are
            # `_subnormal_flush_tell`'s.
            tell = _subnormal_flush_tell(eqn, params, ins, outs[0])
            if tell is not None:
                lost, fired = tell
                self.notes.append(_subnormal_tell_text(
                    eqn,
                    eqn.source_info[-1] if eqn.source_info
                    else "unknown location",
                    lost,
                    fired,
                    self._operand_provenance(eqn),
                ))
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
            if self.semantics == "ieee":
                # UNDER IEEE THE BINARY64 SENTENCE IS NOT THE CLAIM, and
                # stamping it alone is audit 0.2.0 S9/S11: it asserts a
                # property of the analysis host's libm and was read as one
                # of the target's. The run records which (op, format)
                # budgets it consumed and `propagate` renders ONE line
                # naming both halves — the host-libm bracket AND the
                # declared, unverified backend claim.
                d = _ieee_float_dtype_or_none(eqn)
                if d is not None:
                    self.libm_declared.add((eqn.primitive, d))
            else:
                self.assumptions.add(
                    _LIBM_ASSUMPTIONS.get(
                        eqn.primitive,
                        f"{eqn.primitive} endpoints assume a "
                        f"faithfully-rounded libm (error <= 1 ulp), bumped "
                        f"1 ulp outward",
                    )
                )
        if eqn.primitive == "stelling_assert":
            if self.unforced_depth == 0:
                # every cond between this assert and the top of the query
                # had a FORCED index, so reaching this equation by this
                # path means the program reaches it too. On a pinned probe
                # run that is a witness; on a real run it is unread.
                self.certain_reached.add(self._reach_key(eqn))
            # THE ONE ASSUME-STATE READ THAT IS STILL POSITIONAL, AND IT IS
            # MEANT TO BE. `any_constrained` is read HERE, at the
            # obligation, and it selects only between the detail's "over
            # the precondition-narrowed set" and "over the declared box".
            #
            # Why not run-scoped like the withholding. The two answer
            # different questions. The withholding asks whether a definite
            # violation may be called REFUTED at all — a fact about the
            # RUN's assume state, hence read once at the end
            # (`_withhold_uncertified_refutations`). This asks which SET
            # this obligation was judged over, and narrowing is
            # forward-only, so that really is positional: an obligation
            # traced above every assume was judged over the declared box,
            # and a run-scoped read would tell the reader it was judged
            # over the narrowed set — a weaker sentence than the truth,
            # for exactly the rows that differ. Making it run-scoped would
            # make it LESS accurate, not more consistent.
            #
            # The safe direction, and why the coarseness is tolerable.
            # `any_constrained` is one run-wide boolean applied per
            # obligation, so an obligation over a variable no assume
            # touched can still be described as judged "over the
            # precondition-narrowed set". That set is a SUBSET of the
            # declared box, so a predicate definitely false over the box
            # is definitely false over it: the sentence understates and
            # never overstates. The reverse — saying "the declared box"
            # of a narrowed judgment — WOULD overstate, and cannot
            # happen: `any_constrained` is set in the same branch that
            # narrows, and narrowing is forward-only, so no obligation is
            # judged over a narrowed box while this flag is still False.
            #
            # Measured: 6 of the 76 before/after obligation pairs in
            # `scratchpad/mechc` — a historical measurement, tracked and
            # not in the sdist — differ in this detail and 0 differ in
            # status; forcing the flag both ways moves 0 of 648
            # obligation statuses across two corpora and disagrees 0 times
            # in 7392 suite-wide calls (see `_bool_status`).
            status, detail = _bool_status(ins[0], constrained=self.any_constrained)
            if ins[0].size == 0:
                # THE DISCLOSURE, IN THE CHANNEL A CONSUMER READS (audit
                # 0.2.0 B8a, item 6 / M18). The detail already says it; a
                # note is what `Verdict.notes` carries and what the render
                # prints, and it is the difference between "an ordinary
                # VERIFIED" and "a VERIFIED that checked nothing".
                self.notes.append(EMPTY_UNIVERSAL_NOTE.format(
                    what="obligation",
                    index=len(self.obligations),
                    where=(
                        eqn.source_info[-1]
                        if eqn.source_info
                        else "unknown location"
                    ),
                ))
            if status == "unknown":
                # the undecided detail quotes the straddle it was judged
                # on (docs/proposed-decline-messages.md #1) — message
                # content only, appended before the withholding branches
                # below (which replace the detail with their own claims
                # and are untouched)
                detail += self._straddle_suffix(eqn)
                # ...and where the straddle quote has nothing to say because
                # the predicate is a `reduce_and` ⊤, the membership-idiom
                # hint does. The two are complements, never both: a ⊤ has no
                # producing comparison to quote. Note, not detail — see
                # _note_membership_idiom.
                self._note_membership_idiom(eqn, "obligation")
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
            # audit F7's withholding is NOT applied here. It is a fact about
            # the RUN's assume state, and this is a position in the walk: the
            # assume that puts the precondition's satisfiability in doubt may
            # be traced BELOW this obligation and would not be visible yet.
            # `_withhold_uncertified_refutations` applies it once, at the end,
            # over every obligation. See its docstring for the ruling.
            if status == "violated-over-set" and self.unforced_depth:
                # a definite violation inside a branch the analysis only
                # ADMITS. Recorded as a candidate rather than decided here:
                # the certificate is a witness search over the whole query
                # (propagate() runs it, once, and only if a candidate
                # exists), which cannot run from inside this walk.
                self.branch_violations.append(
                    (len(self.obligations), self._reach_key(eqn))
                )
            self.obligations.append(
                ObligationReport(
                    index=len(self.obligations),
                    status=status,
                    detail=detail,
                    source_info=eqn.source_info,
                    operand_var_ids=tuple(
                        a.id for a in eqn.invars if isinstance(a, ir.Var)
                    ),
                    # `_scope_path` is () in the query's own scope and
                    # nowhere else — every descent (transparent call, cond
                    # branch) extends it and restores it — so this is the
                    # exact test for "this assert is a top-level equation",
                    # and `pos` is its index in that scope's `eqns`. An
                    # obligation from any inner scope records None and is
                    # declined individually downstream (M17).
                    top_level_eqn_pos=(
                        pos if self._scope_path == () else None
                    ),
                )
            )
        if eqn.primitive == "stelling_nonvacuity":
            if self.unforced_depth == 0:
                # same witness recording as the assert above: this
                # membership condition is reached by a chain of forced
                # conds, so a pinned probe run that gets here has found a
                # point of the declared box that evaluates it.
                self.certain_reached.add(self._reach_key(eqn))
            # the same deliberately-positional detail read as the assert
            # above, on the nonvacuity face: which SET this condition was
            # judged over is a forward-only fact, while whether its FAILED
            # sentence may stand is the run-scoped one and is decided in
            # `_withhold_uncertified_refutations`. See the assert's note
            # for the argument and the measurements.
            status, detail = _bool_status(ins[0], constrained=self.any_constrained)
            if ins[0].size == 0:
                # the same disclosure on the nonvacuity face, where it is
                # if anything louder: a "checked — N membership condition(s)
                # definitely true (the declared set contains the stated
                # point)" summary derived from a ZERO-element check says
                # the declared set contains a point that was never tested.
                self.notes.append(EMPTY_UNIVERSAL_NOTE.format(
                    what="membership condition",
                    index=len(self.nonvacuity_checks),
                    where=(
                        eqn.source_info[-1]
                        if eqn.source_info
                        else "unknown location"
                    ),
                ))
            if status == "unknown":
                # the assert's hint, on the face that needs it MOST: an
                # undecided membership condition alongside discharged
                # obligations reaches neither a decline note nor
                # undecided_cause_note (which fires on an undecided
                # OBLIGATION), so before this the whole diagnosis was the
                # stamp's one word `undecided`.
                #
                # The status guard is LOAD-BEARING, not belt-and-braces: a
                # reduce_and whose output is SIZE-0 (a non-reduced axis of
                # length zero) is `discharged` — measured, vacuously
                # (:data:`EMPTY_UNIVERSAL_DETAIL`), matching jnp.all of an
                # empty array — while its box is still stelling's ⊤ and the
                # gate still says yes. Without the guard that decided face
                # would carry a note calling itself UNDECIDED.
                self._note_membership_idiom(eqn, "nonvacuity condition")
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
            # audit F9's withholding — the definite FAILED stamp sentence is
            # the same claim class the assert withholding guards — is
            # likewise applied once at the end, over the whole run, by
            # `_withhold_uncertified_refutations`. Not here, for the reason
            # given at the assert above: this is a position, and the rule is
            # about the run.
            if status == "violated-over-set" and self.unforced_depth:
                # the same class as the assert candidate above, on the
                # face that makes the strongest claim: FAILED says "the
                # stated point is NOT in the declared set (harness
                # defect)", and a membership condition inside a branch no
                # point of the box reaches supports no such sentence.
                self.branch_nonvacuity_violations.append(
                    (len(self.nonvacuity_checks), self._reach_key(eqn))
                )
            self.nonvacuity_checks.append(
                ObligationReport(
                    index=len(self.nonvacuity_checks),
                    status=status,
                    detail=detail,
                    source_info=eqn.source_info,
                )
            )
        out_sign = 0 if ieee else self._strict_sign_out(eqn, params, ins)
        for i, (out, val) in enumerate(zip(eqn.outvars, outs)):
            self.env[out.id] = val
            if out_sign:
                # single-output primitives only, in practice: every member
                # of _STRICT_SIGN_PRIMITIVES has one outvar, and a rule for
                # a multi-output primitive would have to say which output
                # it speaks about before it could be added here
                self.strict_sign[out.id] = out_sign
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
                # Format-parametric: check haze against the declaration's
                # own format band
                decl_dtype = str(params.get("dtype", ""))
                decl_fmt = _FLOAT_FORMATS.get(decl_dtype)
                if decl_fmt is not None:
                    haze_changed = iv.subnormal_haze_fmt(
                        raw, _ieee_format_min_normal(decl_fmt)
                    )[1]
                else:
                    haze_changed = iv.subnormal_haze(raw)[1]
                if haze_changed:
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

# THE BOUNDARY DIAL: does a sub-jaxpr boundary the walk crosses CARRY the
# strict-sign certificate? ``"opaque"`` is the default and is the behaviour
# of every release up to and including 0.2.1.
#
# **THE WORD IS ALREADY IN USE ONE MODULE OVER, AND IT MEANS SOMETHING
# ADJACENT AND DIFFERENT.** :data:`stelling.coverage.DEFAULT_TRANSPARENT`
# names the WRAPPERS THE WALK ENTERS — a "transparent" wrapper is one whose
# body the analysis looks inside instead of dropping to ⊤. This dial names
# whether the BOUNDARY the walk crosses CARRIES ANYTHING. The two are
# independent, and today's default is the combination that reads oddest:
# a wrapper is transparent (the walk goes in) and its boundary is opaque
# (nothing about the caller's values goes in with it). That is not a
# contradiction; it is the exact shape of the shipped limitation, which is
# why the two words sit beside each other here rather than one of them
# being renamed.
_BOUNDARY_MODES = ("opaque", "transparent")

# The stamped POSITION line, added to a run's assumptions whenever the dial
# is off its default. See :func:`propagate` for why the ``"opaque"``
# position adds no line at all.
BOUNDARY_TRANSPARENT_POSITION = (
    "boundary='transparent': the strict-sign certificate (\"every element "
    "of this value is certainly nonzero of this sign at every point of the "
    "assumed region\") was allowed to cross the sub-jaxpr boundaries this "
    "walk enters — IN and OUT for the unconditional wrappers "
    "(jit/remat2/custom_jvp_call/custom_vjp_call), IN ONLY for a cond "
    "branch, never out of one. Every verdict recorded before this dial "
    "existed is a boundary='opaque' verdict"
)

# The CONDITIONED disclosure: it reports an ACT, so it is written only when
# the act happened. A position line that also claimed a crossing would be
# claiming a measurement nobody made on the runs where nothing crossed —
# the same asymmetry the libm budget's stamp keeps (`p.libm_declared`
# gates its line, and the profile is never stamped for transfers that did
# not run).
BOUNDARY_CROSSED_DISCLOSURE = (
    "boundary='transparent' CARRIED {n} strict-sign certificate(s) across a "
    "sub-jaxpr boundary on this run (the count is of the carries performed, "
    "not of the verdicts that depended on one: a carried certificate that no "
    "division consulted is counted here, and one consulted many times is "
    "counted once)"
)

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


def _check_boundary(boundary: str) -> None:
    """Guard the boundary dial, eagerly and by name.

    The house idiom for a dial (:func:`_check_semantics`,
    :func:`_check_assume_mode`, :func:`_check_domain`): a module-level
    tuple of the legal values, a refusal that names what was passed, and
    a raise at the door rather than a decline three layers down. The
    dial only DOES anything on a query that both crosses a sub-jaxpr
    boundary and has a certificate to carry, so a typo'd value would
    otherwise ride green through every other query until the day one of
    those arrived.
    """
    if boundary not in _BOUNDARY_MODES:
        raise ValueError(
            f"boundary must be one of {_BOUNDARY_MODES}, got {boundary!r}"
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


# the label for a ⊤ box no top-level EQUATION produced: a closure const
# that had no bracket, or a value bound outside the equation list
_TOP_NOT_FROM_EQUATION = "<constant or closure const>"


def _is_top(box: iv.IntervalArray) -> bool:
    """Is this box ⊤ — [-inf, inf] on every element?

    A zero-size box is NOT ⊤: it has no element that could carry
    information, so "the analysis knows nothing about it" says nothing.
    An all-quantifier over no elements would report every empty array as
    a total loss.
    """
    return box.size > 0 and all(
        lo == -math.inf and hi == math.inf
        for lo, hi in zip(box.los, box.his)
    )


def _top_boxes(
    closed: ir.ClosedJaxpr, env: dict[int, iv.IntervalArray]
) -> tuple[tuple[str, int], ...]:
    """The ⊤ boxes in a finished top-level environment, attributed to the
    equation that produced each (see :attr:`Propagation.top_boxes`).

    Attribution is by outvar id against this jaxpr's own equations — the
    same map :meth:`_Propagator.run` builds for assume classification,
    rebuilt here because that one is scoped and restored. An id no
    equation produced is a constvar and is labelled as one rather than
    guessed at.
    """
    producers: dict[int, str] = {}
    for eqn in closed.jaxpr.eqns:
        for out in eqn.outvars:
            if isinstance(out, ir.Var):
                producers[out.id] = eqn.primitive
    seen: dict[str, int] = {}
    for vid, box in env.items():
        if isinstance(box, iv.IntervalArray) and _is_top(box):
            name = producers.get(vid, _TOP_NOT_FROM_EQUATION)
            seen[name] = seen.get(name, 0) + 1
    return tuple(sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])))


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


# The sentence the withheld obligation and its note both quote, so a reader
# who meets one meets the same claim as a reader who meets the other.
UNCERTIFIED_REACHABILITY_REFUSAL = (
    "a definite violation was found inside a cond/switch branch that the "
    "analysis only ADMITS — the index interval allows the branch, which is "
    "not evidence that any point of the declared set takes it (`x - x > 0` "
    "is admitted both ways while it is false everywhere) — and no point of "
    "the declared box was found that reaches this obligation. Refuting "
    "inside a branch presumes a reachability that nothing certifies, so "
    "the violation is WITHHELD from REFUTED. No claim is made in the other "
    "direction either: an unreachable obligation is vacuously true, and "
    "vacuous truth is not what VERIFIED means here"
)


# THE CONDITIONALITY PHRASE, and it is a constant because it is READ.
#
# A stamped assumption containing it means: this verdict was reached with the
# user's precondition GRANTED, so it claims something about the assumed
# region and not about the declared box. `stelling.verdict.Verdict.render`
# keys its conditional REFUTED wording on it, and
# `stelling.inductive.check_inductive_step` keys the M5 caveat on it — both
# by substring, on whatever mechanism granted the precondition.
#
# It was a bare literal in two files while only ONE mechanism (the interval
# narrowing) wrote it. The forwarded relational axiom is a second, and its
# conditionality was invisible to every one of those readers: audit 0.2.0 M5
# is what that invisibility looks like from the inductive-step API, where
# "the invariant is preserved by one step" was printed for a step preserved
# only inside an assumed sub-region the successor state need not re-enter.
CONDITIONAL_ON_PRECONDITION = "the verdict holds where the precondition holds"

# THE SCOPE OF A CONDITIONALITY LINE, and why reading it as whole-query was a
# defect (audit B3, FRAGILE).
#
# Both mechanisms write `CONDITIONAL_ON_PRECONDITION`, and they are scoped
# differently. An interval NARROWING changes the boxes every obligation of the
# run is judged over, so its line is a fact about the whole query and names no
# obligation. A FORWARDED relational axiom reaches exactly the obligations
# whose scripts stated it, and `stelling.solvers.relational_assume_assumption`
# says so — `… on obligation(s) #1, #3: <phrase> …`.
#
# Both consumers asked `any(CONDITIONAL_ON_PRECONDITION in a for a in
# assumptions)`, which is the whole-query question, and got wrong answers on
# runs where the two populations differ. MEASURED: one interval-refuted
# obligation plus one solver obligation with a forwarded axiom made
# `Verdict.render` call the refutation "conditional … judged over the
# propagated superset of the precondition-narrowed set, not over the full
# declared box" — while nothing had narrowed (a relational assume is inert in
# the interval domain) and that obligation's own detail line said "over the
# declared box". And body `{x,y} -> {0.5x, 0.5y}` on `[-1,1]²` with an assume
# carried only into a body-assert's slice made `check_inductive_step` print
# "inductive step CONDITIONAL — NOT the inductive step", while the four bound
# obligations have single-variable cones, were judged over the full box, and
# close the induction unconditionally (`|0.5·t| ≤ 0.5 ≤ 1`).
#
# So the read is SCOPED, and the failure direction of the parse is the safe
# one: a line whose scope cannot be read is treated as whole-query, which
# over-discloses rather than under-discloses.
_CONDITIONAL_SCOPE = re.compile(r" on obligation\(s\) ((?:#\d+, )*#\d+):")


def conditional_on_precondition(
    assumptions: "tuple[str, ...] | list[str]",
    indices: "frozenset[int] | set[int] | tuple[int, ...]",
) -> bool:
    """Does any stamped conditionality line bear on one of ``indices``?

    ``indices`` are the obligations the caller's sentence is ABOUT — the
    interval-refuted ones for :meth:`stelling.verdict.Verdict.render`, the
    state-bound ones for :func:`stelling.inductive.check_inductive_step`. A
    conditionality line that names obligations bears on those; one that names
    none is whole-query and bears on all of them.

    Asking with an EMPTY ``indices`` is the honest way to ask "is there a
    conditionality this sentence can be about", and the answer is False: a
    sentence about no obligation is conditional on nothing.
    """
    want = frozenset(indices)
    if not want:
        # a sentence about no obligation is conditional on nothing. Answered
        # here rather than falling out of an empty intersection, because a
        # whole-query line would otherwise return True for it and the caller
        # would get "conditional" for a claim it is not making.
        return False
    for a in assumptions:
        if CONDITIONAL_ON_PRECONDITION not in a:
            continue
        head = a.split(CONDITIONAL_ON_PRECONDITION, 1)[0]
        m = _CONDITIONAL_SCOPE.search(head)
        if m is None:
            # whole-query: no obligation named, so it qualifies every one
            return True
        named = frozenset(int(tok[1:]) for tok in m.group(1).split(", "))
        if named & want:
            return True
    return False


# The two STAMPED assumptions an uncertified assume state adds, and the
# one that SUPERSEDES them.
#
# They are constants rather than literals at their emission sites because
# `propagate` has to remove them again. Each says "the conditional claim
# may be vacuous", which is true when it is written — during the walk,
# before any witness exists — and FALSE on a run the non-emptiness
# certificate then settles. A stamped assumption is what a verdict claims
# to rest on; leaving a known-false one in the stamp is a disclosure
# defect whatever the verdict says, so the swap is done once, at the same
# place the certificate's answer is known.
# THE PREFIX IS THE JOIN, not a shared opening phrase. Several mechanisms
# can leave a run's precondition satisfiability unsettled — a narrowing on an
# over-approximated box, a dropped conjunct, and (from the solver layer) an
# undecided admitted-region check — and every consumer that must qualify a
# claim on "the precondition may be empty" has to see ALL of them or it
# qualifies some verdicts and not others for no reason a reader can state.
# A consumer therefore tests the PREFIX, never a member of a list it would
# have to be taught to extend. `stelling.solvers.UNCERTIFIED_REGION_ASSUMPTION`
# is built from it for exactly that reason.
UNCERTIFIED_PRECONDITION_PREFIX = "precondition satisfiability uncertified"
UNCERTIFIED_NARROWING_ASSUMPTION = (
    f"{UNCERTIFIED_PRECONDITION_PREFIX}: a constraining assume "
    "narrowed an over-approximated intermediate whose box may exceed its "
    "true image — the conditional claim may be vacuous; the inert-mode "
    "control is the visibility instrument"
)
UNCERTIFIED_DROP_ASSUMPTION = (
    f"{UNCERTIFIED_PRECONDITION_PREFIX}: a constraining assume "
    "dropped at least one conjunct, so the narrowed set is a superset of "
    "the assumed region and that region was not shown non-empty — the "
    "conditional claim may be vacuous; the inert-mode control is the "
    "visibility instrument"
)
UNDESCENDED_ASSUME_ASSUMPTION = (
    f"{UNCERTIFIED_PRECONDITION_PREFIX}: this query contains a "
    "stelling_assume the propagation NEVER CLASSIFIED — it sits inside a "
    "sub-jaxpr this walk does not enter — so it narrowed nothing, was not "
    "forwarded to the solver, and had no effect on the analysis; the judged "
    "set is a superset of the assumed region and that region was not shown "
    "non-empty. The per-assume note names the construct and the source line"
)
REGION_INHABITED_ASSUMPTION = (
    "precondition satisfiability CERTIFIED: a probed point of the declared "
    "set — a value of each declaration's own dtype inside its own declared "
    "box — satisfies every assume of this query, each predicate definitely "
    "true at that point in the arithmetic the obligations were judged in. "
    "The assumed region is therefore INHABITED and no obligation of this "
    "run is vacuous, which is what the 'precondition satisfiability "
    "uncertified' assumption would otherwise have said. What this rests "
    "on: the soundness of the interval transfers at a point (a definite "
    "TRUE over an enclosure is a TRUE at the value) and the membership of "
    "the probed point in the declared set"
)

# The scope sentence, quoted by every withholding this rule produces so a
# reader who meets one obligation's note meets the rule itself.
ASSUME_QUERY_SCOPE = (
    "an assume is a precondition on the WHOLE QUERY, not only on the "
    "obligations traced after it, so this withholding is applied to every "
    "obligation of the run — the refusal for a DETECTABLY empty assumed "
    "region already ends the run whole, obligations written above the "
    "assume included, because an empty assumed region makes EVERY "
    "obligation vacuously true; the possibly-empty case is the same fact "
    "known less precisely and takes the same scope"
)


def _uncertified_mechanism(p) -> str:
    """The mechanism(s) this run actually hit, named PER FLAG.

    Not one generic sentence for both. Quoting the
    over-approximated-intermediate mechanism at a run whose narrowing
    target was exact — or that narrowed nothing at all — would be a false
    statement of mechanism in the one sentence whose job is to explain the
    withholding, and a reader would have no way to tell it from the run
    where that mechanism really fired.
    """
    causes = []
    if p.narrowing_uncertified:
        causes.append(
            "a constraining assume narrowed an over-approximated "
            "intermediate whose box may exceed its true image"
        )
    if p.assume_dropped:
        causes.append(
            "an assume was DROPPED rather than applied, so the judged set "
            "is a SUPERSET of the assumed region and that region was never "
            "shown non-empty"
        )
    if not causes:
        # the shared decision declined with NEITHER flag set. Unreachable
        # from the flags themselves, and reachable from a caller that
        # overrides `exactness.certifies_set_refutation` — the routing pin
        # does exactly that. Naming no mechanism is the honest sentence
        # there; an empty parenthetical would read as a missing one.
        return (
            "the shared certification decision "
            "(stelling.exactness.certifies_set_refutation) declined for "
            "this run without naming a mechanism of its own"
        )
    return "; and ".join(causes)


def _withhold_uncertified_refutations(p) -> None:
    """Withhold every definite violation of a run whose assume state does
    not certify a set-level refutation — audit F7 (obligations) and F9
    (the nonvacuity FAILED face), applied ONCE, at the end, over the whole
    run.

    **The shared point.** The decision is
    :func:`stelling.exactness.certifies_set_refutation`, which
    :mod:`stelling.affine` consults for the same decision on its own leg.
    It is stated there rather than here because two legs answering it
    separately is an agreement that breaks silently: the refinement reads
    a whole-run quantity by architecture (it is a post-pass), this leg
    reads one because that is the rule, and a refinement restructured to
    run inline would drift with nothing catching it. The NON-EMPTINESS
    CERTIFICATE reaches that same point as its third argument
    (``region_inhabited``) rather than as a test of its own here — a leg
    that lifted the withholding locally would be a third channel around
    the shared decision, which is the arrangement it exists to end.

    **Why the end of the run and not the obligation.** Read at the
    obligation, this saw only the assumes traced ABOVE it, and the same
    claim under the same precondition returned UNKNOWN with the assume
    written first and REFUTED with it written second. An assume is a
    precondition on the query (:data:`ASSUME_QUERY_SCOPE`).

    **One-sided.** `discharged` is never touched: a discharge over a
    superset implies the discharge over the intended set. And the run is
    never declined wholly — `solvers.py` tried that on its own leg and
    reverted it.

    **The surface this widened, and the invariant that closes it.** The
    one-sidedness above is about what THIS function writes; it is not by
    itself an argument, because writing `unknown` at the end of the run
    rather than at the assert hands the obligation to two layers that
    never saw it. Both key on exactly this word:
    :func:`stelling.affine.refine_propagation` and
    :func:`stelling.solvers.escalate` each take
    ``[o for o in propagation.obligations if o.status == "unknown"]``.
    What keeps that from becoming a route to a wrong `discharged` is
    WHICH DOMAIN those layers judge over. Both decline wholly on a run
    with ``coverage.constrained`` — so every obligation this function
    newly offers them comes from a run in which nothing was narrowed,
    where the declared boxes they read ARE the boxes the interval leg
    judged. A sound layer cannot discharge a predicate that the interval
    leg found definitely FALSE at every point of that same domain, and a
    re-minted violation is caught by the shared point again on the affine
    leg. Measured across `scratchpad/mechc` and
    `scratchpad/claims/corpus_b3.py` — a historical measurement, tracked
    and not in the sdist, and the figures it produced are these —
    (100 harness-runs, jax 0.11.0):
    **24** obligations newly offered to each layer, **34** additional
    solver invocations, and **0** new affine discharges, **0** new affine
    violations, **0** solver-decided outcomes (every one `unknown`) and
    **0** new VERIFIEDs at the verdict layer (14 → 14 of 200 verdicts;
    48 verdicts move, all REFUTED → UNKNOWN).

    A run with no assumes, with only certified assumes, or in inert mode
    returns at the guard having done nothing: no note, no status, no
    detail changes.
    """
    if exactness.certifies_set_refutation(
        nonemptiness_certified=not p.narrowing_uncertified,
        assume_dropped=p.assume_dropped,
        region_inhabited=p.region_inhabited,
    ):
        return
    mechanism = _uncertified_mechanism(p)
    work = (
        (
            p.obligations,
            "violation WITHHELD from REFUTED",
            (
                "a definite violation was found over the judged set, but "
                "the precondition's satisfiability is uncertified "
                f"({mechanism}) — a possibly-vacuous refutation is not a "
                "refutation; REFUTED under constraining assumes requires "
                f"certified-satisfiable preconditions. {ASSUME_QUERY_SCOPE}"
            ),
            (
                "definite violation over the judged set WITHHELD from "
                f"REFUTED (precondition satisfiability uncertified: "
                f"{mechanism}; see notes)"
            ),
        ),
        (
            p.nonvacuity_checks,
            "nonvacuity FAILED face WITHHELD",
            (
                "the membership condition was judged definitely false over "
                "the judged set, but the precondition's satisfiability is "
                f"uncertified ({mechanism}) — the FAILED sentence is "
                "reserved for judgments not confounded by an uncertified "
                f"constraint. {ASSUME_QUERY_SCOPE}"
            ),
            (
                "membership condition definitely false over the judged set "
                "WITHHELD from FAILED (precondition satisfiability "
                f"uncertified: {mechanism}; see notes)"
            ),
        ),
    )
    for sink, headline, why, detail in work:
        for i, o in enumerate(sink):
            if o.status != "violated-over-set":
                continue
            where = o.source_info[-1] if o.source_info else "unknown location"
            p.notes.append(f"{headline} at {where}: {why}")
            sink[i] = dataclasses.replace(o, status="unknown", detail=detail)


# The NON-EMPTINESS CERTIFICATE's cap, in DECLARED ELEMENTS (Gate 2).
#
# The search costs up to `_PROBE_COUNT` extra propagations of the whole
# query, and a propagation's cost grows with the declared element count
# (every box is a pair of endpoint tuples that long). Unbounded, that is a
# per-query cost the caller never asked for and cannot predict. Bounded by
# the DECLARED SIZE — the one quantity the user wrote down and the one the
# cost actually tracks — it is predictable from the harness.
#
# TWO BOUNDS, because one was not enough and the measurement says which.
# `scratchpad/cert/RESULTS_cap.txt` is a historical measurement, tracked and
# not in the sdist, and its figures are stated here rather than linked to:
# jax 0.11.0, load 0.44, with the size
# cap alone at 4096, a search that FINDS nothing walks the full grid and
# costs **469 ms against a 23 ms propagation — 95% of the whole `check()`
# pipeline**. The successful search is cheap (3.7x) because it stops at
# the first witness; it is the failing one that had to be bounded.
#
# So the probe count itself scales with the declared size:
# `_CERT_PROBE_BUDGET` is a budget in ELEMENT-PROBES, and a declaration
# of n elements gets `budget // n` probes, floored at `_CERT_MIN_PROBES`
# and capped at `_PROBE_COUNT`. Worst case at the size cap falls from
# ~469 ms to ~70 ms, and small declarations still get the whole grid.
#
# WHAT THE FLOOR COSTS, measured rather than assumed
# (`scratchpad/cert/RESULTS_probe_index.txt` — a historical measurement,
# tracked and not in the sdist; this is the only place in the tree that
# names it, and the figures below are what it holds): across the 17 corpus
# rows
# that witness at all, the first witnessing probe index is 0, 1 or 2 in
# **17 of 17** — three probes recover 100% of them, one probe recovers
# 18%. Probes 0, 1 and 2 are the declared box's LOW corner, HIGH corner
# and MIDPOINT, which is why the floor is 3 and not some fitted number.
# The corpus's blind spot is stated with it: its assumed regions are
# half-space-shaped, and a region whose only members sit off the
# corner/midpoint grid would need a later probe and would be lost at the
# floor. That is a withholding, which is safe, and it is a real cost.
#
# WHICH SEARCH THESE BOUNDS BIND, because this module now holds TWO.
# `_region_witness` is bounded above; `_reachability_witnesses` is NOT —
# it still runs `for k in range(_PROBE_COUNT)`, 16 full propagations, at
# any declared size whatever. Measured on this tree (jax 0.11.0, load
# 1.18 before and 1.16 after, a violation inside a `lax.cond` branch;
# `scratchpad/pin/f6_repro.py time`, a historical measurement in an
# instrumented worktree, tracked and not in the sdist): `propagate` costs
# 1.6/9.7/126.6/549.9 ms at n = 16/256/4096/16384 against a bare walk of
# 0.1/0.5/6.2/25.7 ms — 21.4x at n = 16384, which is four times the size
# cap above, with the probe count unmoved at 16 throughout. That is the
# same shape as the 469 ms/23 ms number this cap was added to fix.
#
# THEY ARE MUTUALLY EXCLUSIVE, so no query pays for both. `_region_witness`
# gets past its own gate only when `narrowing_uncertified or
# assume_dropped`; `narrowing_uncertified` is set inside the same
# `if narrowed:` block that sets `any_constrained`, so that implies
# `any_constrained or assume_dropped`, which is exactly the condition on
# which `_reachability_witnesses` returns the empty set BEFORE probing.
# Contrapositive: a run that pays the 16 probes is a run whose certificate
# search declined at its gate for 0. MEASURED over
# `scratchpad/pin/corpus_pin.py` — a historical measurement, tracked and
# not in the sdist — plus a size grid built to reach both,
# **508 propagations** including 32 rows that put a branch-scoped
# violation beside a narrowing and a dropped assume: **0 pay for both**,
# and the worst combined probe count is **16**, not `16 +
# _certificate_probe_count(n)` — which, were they not exclusive, would
# peak at **32** at small n, where the certificate's budget is loosest,
# and not at the size cap. The two also cannot contradict each other
# for the same reason — the certificate can only fire on runs where the
# reachability search certifies nothing, so a branch-scoped violation is
# never restored by it (`test_the_certificate_can_never_restore_a_branch_
# scoped_refutation`).
#
# WHY THE OLDER SEARCH IS NOT CAPPED HERE. Applying
# `_certificate_probe_count` to it would move verdicts, which this branch
# may not do. Measured over 21 branch-violation rows at n = 4 … 16384,
# scored on the keys the branch pass ASKS about (`p.branch_violations`)
# and not on the ones it happens to find — a key asked and never
# certified within the budget is exactly a loss, and a found-key score
# would miss every one of them by construction: 15
# reachability keys asked, **3 lost** — the `x[0] > x[1]` shape at
# n >= 4096, whose first certifying probe is index 3 (the first
# per-element probe; the plain anchors put every element at the same
# value and cannot witness a relation between two of them) while the
# budget floor grants exactly 3. Each loss is a `violated-over-set` ->
# `unknown` move: the safe direction, and a real cost. Recorded here so
# the decision is a measured one rather than an omission.
_CERT_MAX_ELEMENTS = 4096
_CERT_PROBE_BUDGET = 4096
_CERT_MIN_PROBES = 3


def _certificate_probe_count(elements: int) -> int:
    """How many probes a declaration of ``elements`` elements earns.

    The whole grid for a small declaration, the corner/corner/midpoint
    floor for one at the size cap. Bounded by the DECLARED SIZE at both
    ends, so a caller can read the worst case off the harness.
    """
    return max(
        _CERT_MIN_PROBES,
        min(_PROBE_COUNT, _CERT_PROBE_BUDGET // max(elements, 1)),
    )


def _assume_equations(jaxpr) -> dict:
    """Every ``stelling_assume`` equation the query CONTAINS, keyed by
    ``id()``, with the chain of primitives whose sub-jaxprs enclose it.

    ONE TRAVERSAL, TWO READERS. :func:`_assume_equation_ids` is the
    identity half and this is the whole thing; they are not two walks that
    have to be kept agreeing, because the ids ARE this mapping's keys. That
    matters because the id set's TOTALITY is the property two soundness
    rules rest on and it was checked against an independent walk of the raw
    jax jaxpr (``test_the_static_assume_set_is_total_over_sub_jaxprs``); a
    second traversal added beside it would be a second thing to check, and
    the audit finding this fixes is precisely a claim scoped to one
    traversal while another one existed.

    The value is ``(equation, enclosing_primitives)``, outermost first, so
    ``()`` is a top-level assume and ``("scan", "jit")`` is one inside a
    ``jit`` inside a ``scan`` body. It is what lets a disposition for an
    assume nobody classified NAME the construct
    (:data:`UNDESCENDED_ASSUME_REASON`).
    """
    found: dict = {}
    stack = [(jaxpr, ())]
    while stack:
        j, path = stack.pop()
        for e in j.eqns:
            if e.primitive == "stelling_assume":
                found.setdefault(id(e), (e, path))
            inner = path + (e.primitive,)
            stack.extend((sub, inner) for sub in sub_jaxprs(e))
    return found


def _assume_equation_ids(jaxpr) -> frozenset:
    """The identity of every ``stelling_assume`` equation the query
    CONTAINS — statically, sub-jaxprs included, whether or not any walk
    reaches it.

    Static because :func:`stelling.exactness.certifies_point_witness`
    needs the requirement to cover assumes a probe could walk AROUND: a
    pinned probe forces conds and so walks one branch, and an assume in
    the other branch would otherwise be certified by not being looked at.
    Over-collecting is the safe direction here — an assume nothing ever
    evaluates simply makes the subset test fail and no certificate is
    issued.
    """
    return frozenset(_assume_equations(jaxpr))


def _declared_element_count(jaxpr) -> int:
    """Total elements across every ``stelling_any`` declaration in the
    query — the size the user declared, which is what the certificate
    search's cap is stated in.

    **THIS READS THE OUTVAR AVAL; THE EMISSION PATH READS THE ``shape``
    PARAM.** The two disagree on a declaration that describes itself
    twice — an absent ``shape`` param reads as ``()`` there and as the
    aval's own count here — so this function is the library's SECOND
    reader of a declaration's element count, and
    ``obligation._Slicer._declared_shape``'s docstring used to claim it
    was the only one (audit 0.2.0 B6 audit 3, F4). Reading the aval here
    is sound, and it is not an exception to that rule: the count gates
    only the CAP in :func:`_region_witness` — whether the non-emptiness
    search RUNS — whose direction is toward REFUTED, and the search
    re-derives its witness by re-running the honest propagator, so no
    verdict is derived from this number. Should that ever stop being
    true, this must move to the param.
    """
    total = 0
    stack = [jaxpr]
    while stack:
        j = stack.pop()
        for e in j.eqns:
            if e.primitive == "stelling_any":
                for out in e.outvars:
                    n = 1
                    for d in out.aval.shape:
                        n *= d
                    total += n
            stack.extend(sub_jaxprs(e))
    return total


def _region_witness(closed, p, *, assume_mode, semantics, boundary="opaque") -> bool:
    """Search for ONE point of the declared set at which EVERY assume of
    the query is definitely true — the non-emptiness certificate.

    **What it is for.** A run whose assume state does not certify a
    set-level refutation withholds every definite violation from REFUTED,
    query-wide. The withholding has exactly one ground: the assumed region
    *may be empty*, in which case every obligation is vacuously true. It
    does not say the judged set was wrong — a narrowing is a meet with a
    CLOSED half-space and a drop only widens, so the judged set is a
    superset of the assumed region either way, and a definite violation
    over a superset is a violation at every point of the region. Exhibit
    one point of that region and the ground is gone.

    **What a "point" has to be.** A member of the DECLARED SET, which for
    a narrow dtype is a value of that format and not merely a number
    inside the interval: `int32 (0.2, 2.8)` declares `{1, 2}` and
    `float32 (v, (v + nextafter(v))/2)` declares `{v}`. That problem is
    already solved — :func:`_member_bounds` and :func:`_probe_point` solve
    it for the reachability probe grid — and this reuses them rather than
    re-deriving them, through the same ``pin`` mechanism
    (:meth:`_Propagator._pinned`).

    **What arithmetic the check runs in, and why that one.** stelling's
    own, in ``semantics`` — the same propagation that judged the query.
    The endpoints of the basic operations are computed in ``Fraction``
    and correctly directed-rounded (:func:`stelling.interval._exact_down`
    / ``_exact_up``), so a predicate box is a sound enclosure of the
    predicate's value AT THE PINNED POINT; ``[1, 1]`` on every element
    therefore means true at that point. Two consequences, both wanted:

    * no second arithmetic. A witness checker with its own evaluator
      could disagree with the propagator about what a program computes,
      and the disagreement would be invisible. This one inherits the
      tool's soundness argument instead of running beside it.
    * the check is exact where the arithmetic is exact and INDETERMINATE
      where it is not, never wrong. Measured: at the point `(0.1, 0.2)`
      the predicate `x0 + x1 >= 0.30000000000000004` is TRUE in binary64
      and FALSE in ℝ; **under ``semantics="real"``, and only there,** the
      box is `[0x1.3333333333333p-2, 0x1.3333333333334p-2]`, which
      straddles the bound, so the predicate is INDETERMINATE and no
      witness is claimed. An exact-rational checker would answer FALSE
      there; this answers "not established", which withholds. Weaker,
      never unsound. **Under ``semantics="ieee"`` the SAME query
      certifies and REFUTES** — measured on this tree, `region_inhabited`
      False/`unknown` under `real` and True/`violated-over-set` under
      `ieee` — and that is also sound, because jax executes binary64 and
      the point genuinely satisfies the assume AS EXECUTED. The dial is
      part of the sentence, not a qualification 20 lines away: the check
      runs in the run's own semantics, and neither dial is the stronger
      one (see the two-semantics paragraph in ``SOUNDNESS.md``).

    **``sqrt``/``sin``/``exp``/``log`` are a boundary, not a gap.**
    Nothing confirms a point exactly through them, and this does not try:
    the enclosure at the pinned point has width, and a predicate whose
    bound falls inside that width reads INDETERMINATE and certifies
    nothing. What it does still do — soundly — is certify a predicate
    with SLACK: `assume(sqrt(x + 1) >= 1.2)` at the pinned `x = 0.5`
    encloses `sqrt(1.5)` in a box whose lower endpoint is above 1.2, and
    a definite TRUE over an enclosure is a TRUE at the value. So the
    boundary is the margin, not the primitive.

    **ONE-SIDED, and the code says so because the code is the claim.**
    Returning False means NO WITNESS WAS FOUND. It never means the region
    is empty: the grid is at most 16 points and fewer on a large
    declaration (:func:`_certificate_probe_count`), the arithmetic can be
    indeterminate, the cap can decline to search at all, and a probe that
    raises is caught and skipped. Every False path below therefore leaves the run
    EXACTLY as it found it — no note, no status, no detail, not even a
    disclosure that a search happened. That is not reticence; it is what
    makes the one-sidedness pinnable byte-for-byte
    (``test_a_failed_certificate_search_changes_nothing_at_all``) instead
    of argued. The withholding sentence the run already carries explains
    the withholding completely on its own, and it was true before this
    function existed.

    **What is NOT certified: an assume the probe walked AROUND.** The
    requirement is the STATIC set of assume equations
    (:func:`_assume_equation_ids`) and the witness is what one pinned walk
    evaluated, so an assume in a branch the probe did not take is required
    and not witnessed, and the certificate fails. Deliberate: a
    branch-scoped precondition that is empty in its branch is exactly the
    vacuity being guarded against, and a top-level point that walks around
    it certifies nothing about it.

    **NOT "branch-scoped assumes, always", which is what this paragraph
    used to say and is false.** Pinning a declaration FORCES a cond, and
    forcing it can force it either way; a probe whose point takes the
    branch evaluates the assume inside it and witnesses it like any other.
    Measured on this tree: a query whose ONLY assume sits inside a
    ``lax.cond`` branch is certified by probe 1 (the declared box's high
    corner), and the recovery is sound — at that point the program really
    does take the branch and really does satisfy the assume
    (``test_an_assume_the_probe_walks_INTO_is_witnessed_and_certified``).
    The guarantee that branch-scoped VIOLATIONS stay withheld is a
    DIFFERENT and independent mechanism — :func:`_reachability_witnesses`
    returns the empty set on any run with ``any_constrained or
    assume_dropped``, which is every run this function can fire on — and
    that one is true. Conflating the two overstated both.

    The static requirement does cost recoveries, in exactly the shape the
    overstated sentence claimed it prevented: a region inhabited only via
    the UNTAKEN branch (every admissible point walks the side WITHOUT the
    assume) is required-and-not-witnessed on every probe, and a sound
    refutation is withheld. Measured, 8 of 16 probes walking around it
    with an EMPTY witness map
    (``test_a_region_inhabited_only_via_the_UNTAKEN_branch_is_not_recovered``).
    """
    if exactness.certifies_set_refutation(
        nonemptiness_certified=not p.narrowing_uncertified,
        assume_dropped=p.assume_dropped,
        # explicitly False rather than omitted: this is the question
        # "would this run withhold ABSENT a certificate", asked before one
        # exists, and passing the answer we are about to compute would be
        # a circular read. Spelling it keeps the invariant that EVERY
        # reach of the shared point names all three inputs, which is what
        # `test_every_reach_of_the_shared_point_names_the_certificate`
        # observes.
        region_inhabited=False,
    ):
        # nothing is being withheld on this run, so there is nothing for a
        # certificate to lift and no reason to pay for one.
        return False
    statuses = {
        o.status for sink in (p.obligations, p.nonvacuity_checks) for o in sink
    }
    if "violated-over-set" not in statuses and not (
        "unknown" in statuses and not p.any_constrained
    ):
        # NOTHING FOR A CERTIFICATE TO LIFT ON THIS RUN, so it pays nothing.
        # Two ways there can be something, one per leg:
        #
        #  * a `violated-over-set` obligation is what THIS leg withholds;
        #  * an `unknown` one on a run that CONSTRAINED nothing is what the
        #    affine refinement may still mint a violation from — and that
        #    leg withholds through the same shared decision, reading this
        #    same answer off the propagation.
        #
        # The second clause is narrowed by `any_constrained` because
        # :func:`stelling.affine.refine_propagation` declines WHOLLY on
        # `coverage.constrained` (the same condition, one name over), so
        # searching on a run that narrowed something would buy that leg
        # nothing. Measured, jax 0.11.0: without the second clause,
        # `assume(x >= y)` (relational, dropped, region inhabited) with
        # `assert_(x - x >= 0.5)` — interval-undecided, affine-violated —
        # returns UNKNOWN from the refinement with the certificate never
        # computed; with it, REFUTED, which is the right answer. The cost
        # is a search on a run whose interval leg withheld nothing:
        # 1.4 ms -> 30 ms on a 256-element declaration, inside the bounds
        # below.
        return False
    required = _assume_equation_ids(closed.jaxpr)
    if not required:
        return False
    elements = _declared_element_count(closed.jaxpr)
    if elements > _CERT_MAX_ELEMENTS:
        # THE CAP (Gate 2). Silent, like every other failure path here:
        # declining to search is not a finding, and the run's own
        # withholding sentence is unchanged and still complete.
        return False
    for k in range(_certificate_probe_count(elements)):
        # the probe judges the SAME query in the SAME arithmetic, so it
        # must carry the same declaration — a probe that declined where
        # the run did not would certify less than the run earned
        # THE SAME DIAL THE WALK RAN UNDER, for the reason the comment
        # above gives about the declaration: a probe that declined where
        # the run did not would certify less than the run earned. The
        # certificate is real-mode only and the probes are the run's own
        # arithmetic, so this is the same argument one dial over.
        probe = _Propagator(assume_mode, semantics, p.libm_budget, boundary)
        probe.pin = k
        try:
            probe.run(closed.jaxpr, list(closed.consts), [])
        except Exception:  # noqa: BLE001 — a failed probe certifies nothing
            continue
        if exactness.certifies_point_witness(
            required_assumes=required,
            witnessed_assumes=frozenset(
                key for key, ok in probe.assume_witness.items() if ok
            ),
        ):
            # FIRST witness wins: one point is the whole claim, and
            # searching on after it is found buys nothing (Gate 2's other
            # half — a successful search costs one probe, not sixteen).
            p.notes.append(
                f"assumed region CERTIFIED NON-EMPTY: probe point {k} of "
                f"the declared set satisfies every assume of this query "
                f"(each assume's predicate is definitely true at that "
                f"point, in the same arithmetic the query was judged in), "
                f"so the assumed region is inhabited and a definite "
                f"violation over the judged set is not vacuous — definite "
                f"violations are NOT withheld from REFUTED on this run, "
                f"and this SUPERSEDES the 'precondition satisfiability "
                f"UNCERTIFIED' note(s) and stamped assumption(s) this run "
                f"wrote while walking, before any witness existed"
            )
            return True
    return False


def _reachability_witnesses(closed, p, *, assume_mode, semantics, boundary="opaque"):
    """Assert outvar ids the program provably reaches somewhere in the box.

    Each probe is one propagation of the SAME query with every declaration
    pinned to a single point of its own declared box. An assert reached on
    such a run under a chain of forced conds is an assert the program
    evaluates at that point — a witness, not an over-approximation.

    Returns the empty set (certifying nothing, so every candidate is
    withheld) when the run being certified is not judging the declared box:
    a constraining assume narrows the admitted set, and a point of the box
    outside the narrowed region is not a witness for it. A probe that
    raises certifies nothing either — the safe direction throughout is
    withholding.

    **THIS SEARCH IS NOT CAPPED**, and the bounds stated at
    :data:`_CERT_MAX_ELEMENTS` are the OTHER search's. It runs the full
    ``_PROBE_COUNT`` grid — 16 whole propagations — at any declared size:
    measured, 549.9 ms against a 25.7 ms bare walk at n = 16384. It never
    runs on the same query as :func:`_region_witness` (the guard above is
    the complement of that function's gate, so the worst combined cost is
    16 probes and not 16 plus a budget — measured 0 of 508 propagations
    paying for both), and the reasons it is left
    uncapped — a cap costs 3 of 15 measured reachability keys, i.e. moves
    verdicts — are recorded at :data:`_CERT_MAX_ELEMENTS`.
    """
    if p.any_constrained or p.assume_dropped:
        return frozenset()
    found: set[int] = set()
    for k in range(_PROBE_COUNT):
        # the walk's own dial, exactly as in :func:`_region_witness`
        probe = _Propagator(assume_mode, semantics, p.libm_budget, boundary)
        probe.pin = k
        try:
            probe.run(closed.jaxpr, list(closed.consts), [])
        except Exception:  # noqa: BLE001 — a failed probe certifies nothing
            continue
        found |= probe.certain_reached
    return frozenset(found)


def _withhold_uncertified_branch_refutations(
    closed, p, *, assume_mode, semantics, boundary="opaque"
):
    """Withhold REFUTED from violations in branches nothing certifies.

    Runs the witness search ONCE, and only when the walk actually recorded
    a branch-scoped violation: a query with none pays nothing.
    """
    work = [
        (p.obligations, p.branch_violations, "violation", "REFUTED"),
        (
            p.nonvacuity_checks,
            p.branch_nonvacuity_violations,
            "nonvacuity FAILED face",
            "FAILED",
        ),
    ]
    candidates = [
        (sink, i, key, what, face)
        for sink, recorded, what, face in work
        for i, key in recorded
        if sink[i].status == "violated-over-set"
    ]
    if not candidates:
        return
    certified = _reachability_witnesses(
        closed, p, assume_mode=assume_mode, semantics=semantics,
        boundary=boundary,
    )
    for sink, i, key, what, face in candidates:
        if key in certified:
            continue
        o = sink[i]
        where = o.source_info[-1] if o.source_info else "unknown location"
        p.notes.append(
            f"{what} WITHHELD from {face} at {where}: "
            f"{UNCERTIFIED_REACHABILITY_REFUSAL}"
        )
        sink[i] = dataclasses.replace(
            o,
            status="unknown",
            detail=(
                f"definite violation inside a branch whose reachability is "
                f"UNCERTIFIED, withheld from {face} (see notes); no "
                f"definite status is claimed in either direction"
            ),
        )


# The frame `stelling.harness.assume` binds the primitive from, as
# `_jax_compat.Decoder.source` spells a frame: "<file>:<line> (<function>)".
# Matched on BASENAME, not on the recorded absolute path, because a query can
# be decoded from a `from_dict` produced on another machine and the path there
# is that machine's. `harness.assume is _jax_compat.assume` (a re-export, not
# a wrapper), so this is the only stelling frame on the stack and there is no
# second spelling to keep in step.
_ASSUME_BIND_FRAME = "_jax_compat.py:"


def _assume_source(source_info: tuple) -> str:
    """WHICH OF THE USER'S ``assume`` CALLS THIS EQUATION IS — the frame that
    CALLED :func:`stelling.harness.assume`, not the outermost frame.

    **A DEPARTURE FROM THE ``source_info[-1]`` HOUSE CONVENTION, SCOPED TO
    THIS ONE READER, and measured rather than argued.** jax records the bind
    stack innermost-first, so ``[-1]`` is the OUTERMOST frame — which is the
    user's ``assume(`` line only when the user's harness function called it
    directly. Audit B9 measured ten carriers on jax 0.11.0; ``[-1]`` names
    something other than the ``assume(`` line in four of them, and twice that
    something is a line in jax:

      * ``lax.fori_loop`` — ``jax/_src/lax/control_flow/loops.py:2528
        (_fori_scan_body_fun.<locals>.scanned_fun)``;
      * ``lax.map`` — ``loops.py:2784 (map.<locals>.<lambda>)``;
      * an ``assume`` written inside any helper the harness calls — the
        helper's CALLER, not the ``assume(`` line.

    ``scan``, ``while_loop``, ``cond``, ``jit``, ``scan``-in-``cond`` and
    nested ``scan`` agree with ``[-1]`` on all six, so this changes nothing
    where the convention was already right.

    **WHY THIS READER AND NOT THE CONVENTION EVERYWHERE.** The convention is
    correct for its usual population — an equation jax's own machinery
    produced, where the outermost frame is the user's line — and
    :data:`UNDESCENDED_ASSUME_REASON` is the first string to make LOOP-BODY
    assumes visible, which is precisely where jax wraps the user's function
    in one of its own. Changing the convention globally would be changing
    every ``where`` in this module on the strength of one population's
    evidence. Every other site keeps ``[-1]``.

    **WHY THE CALLER OF ``assume`` AND NOT "THE LAST FRAME OUTSIDE jax".**
    The stated purpose of this string is "the reader's next question is which
    of their assumes this was", and the caller of ``assume`` answers exactly
    that, by construction, for any nesting: it is the line the ``assume(``
    token is on. "Last frame outside jax" answers a weaker question and still
    gets the helper case wrong (it returns the helper's caller), and it would
    need to recognise jax's installation directory from a string — which this
    module cannot ask jax for, since it must stay importable with no jax at
    all.

    Falls back to the house convention when the landmark is absent — an
    equation some other producer made, or a hand-built one — because a guess
    about which frame is the user's is worse than the convention.
    """
    if not source_info:
        return "unknown location"
    for i, frame in enumerate(source_info):
        if _ASSUME_BIND_FRAME in frame and frame.endswith("(assume)"):
            if i + 1 < len(source_info):
                return source_info[i + 1]
            break
    return source_info[-1]


def _record_undescended_assumes(closed, p) -> None:
    """Record, as a DROPPED disposition, every ``stelling_assume`` the query
    contains and the walk never classified — audit 0.2.0 S13.

    **THE ROOT CAUSE, NOT ITS THREE SYMPTOMS.** The propagator descends the
    :data:`stelling.coverage.DEFAULT_TRANSPARENT` wrappers and ``cond``; it
    does not descend ``scan`` or ``while_loop``. An assume in one of those
    bodies was therefore never classified — so it left NO ledger entry, and
    an absent entry is invisible to every rule keyed on the ledger. Three
    rules read it, and each failed in its own direction:

      * the WITHHOLDING rule — ``assume_dropped`` stayed ``False``, nothing
        withheld, and the solver returned a witness from OUTSIDE the assumed
        region as a counterexample. Measured on the released ``v0.1.0`` and
        on this tree: ``x, y in [-10, 10]``, ``assume(x <= y)`` inside a
        ``lax.scan`` body, ``assert_(x - y <= 0.0)`` — REFUTED at
        ``x = 0, y = -1``, which the user's own assume excludes. A FALSE
        REFUTED, the worse direction;
      * the ADMITTED-REGION gate — an empty region stamped entirely clean;
      * ``REGION_NOT_ASKED`` — the region question skipped outright.

    Fixing them one at a time would be fixing a missing record three times
    over. The record is what is missing, so the record is what this writes,
    and each rule then sees it through the machinery it already has.

    **WHY THE STATIC SET AND NOT THE POINT OF NON-DESCENT.** The obvious
    place is where the walk declines to enter — ``mark_unreached``, which
    already catches an unexamined ``stelling_assert`` there. That would be a
    fix scoped to A TRAVERSAL, which is the exact shape of the defect it is
    repairing: it records only what the walk reaches, and an assume the walk
    never reaches for some FOURTH reason is missed again in silence.
    :func:`_assume_equations` is static and total — every assume equation
    the query contains, whether or not any walk reaches it, checked against
    an independent walk of the raw jax jaxpr — so reconciling the ledger
    against it makes ``ledger_covers`` a POSTCONDITION of this function
    rather than a hope about the walk. It costs one traversal of the jaxpr
    and no probes, no solver calls.

    **THE DIRECTION IS CONSERVATIVE, and it is not free.** Every one of
    these entries makes ``unaccounted_assumes`` non-empty, which withholds
    definite violations from REFUTED and caveats discharges. That is the
    correct cost of an assume nobody honoured: a violation found over a
    superset of the assumed region may be at a point the assume excludes,
    and a discharge is sound but may be a claim about an empty region. It
    does NOT descend the loop — a loop body's assume is a per-iteration
    statement about a carry this analysis does not model, and inventing a
    reading of it would be a much larger feature than not ignoring it.

    Ordered by the static traversal, so the notes are reproducible.
    """
    recorded = {e.eqn_id for e in p.assume_ledger if e.eqn_id != -1}
    for eqn_id, (eqn, path) in _assume_equations(closed.jaxpr).items():
        if eqn_id in recorded:
            continue
        where = _assume_source(eqn.source_info)
        inside = (
            " -> ".join(repr(prim) for prim in path) if path
            else "a sub-jaxpr this walk did not enter"
        )
        p.assume_ledger.append(AssumeDisposition(
            kind=ASSUME_DROPPED,
            reason=UNDESCENDED_ASSUME_REASON.format(inside=inside),
            where=where,
            eqn_id=eqn_id,
        ))
        p.notes.append(
            UNDESCENDED_ASSUME_NOTE.format(where=where, inside=inside)
        )
        # THE SAME FLAG THE CLASSIFIER'S OWN DROP SETS, for the same reason
        # and with the same one-sidedness: the judged set is a superset of
        # the assumed region, so a discharge carries over and a definite
        # violation does not. Set here rather than at the walk because the
        # walk is what missed it.
        p.assume_dropped = True
        p.assumptions.add(UNDESCENDED_ASSUME_ASSUMPTION)


def _query_float_formats(closed: ir.ClosedJaxpr) -> tuple[str, ...]:
    """The float format names this query CONTAINS, sub-jaxprs included.

    A stamp is a disclosure, so this reads the PROGRAM rather than the
    walk: a format that appears only inside an untaken branch is still a
    format the verdict's assumption lines have to be true of, and
    over-naming is the safe direction for a sentence that says "these are
    the formats these claims are about".
    """
    found: set[str] = set()
    stack = [closed.jaxpr]
    seen: set[int] = set()
    while stack:
        j = stack.pop()
        if id(j) in seen:
            continue
        seen.add(id(j))
        for v in (*j.invars, *j.constvars, *j.outvars):
            d = getattr(getattr(v, "aval", None), "dtype", None)
            if d in _FLOAT_FORMATS:
                found.add(d)
        for e in j.eqns:
            for v in (*e.invars, *e.outvars):
                d = getattr(getattr(v, "aval", None), "dtype", None)
                if d in _FLOAT_FORMATS:
                    found.add(d)
            stack.extend(sub_jaxprs(e))
    return tuple(sorted(found))


def propagate(
    closed: ir.ClosedJaxpr,
    *,
    semantics: str = "real",
    assume_mode: str = "constrain",
    domain: str = "interval",
    libm_budget=None,
    boundary: str = "opaque",
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

    ``libm_budget`` — ``None`` (the default), a shipped profile NAME from
    :data:`LIBM_PROFILES`, or a :class:`LibmBudget`. Under
    ``semantics="ieee"`` the transfers that ride a libm accuracy claim
    (``exp``, ``pow``) **decline unless a budget names their exact
    (op, format) pair**: the bracket is built around this host's ``math``
    module and the program runs whatever the compiler emitted, and the
    measured gap reaches 5.5 float32 ulps (audit 0.2.0 S9, S11). Passing
    one under ``semantics="real"`` raises: it has no meaning there.

    ``boundary="opaque"`` (the default) is the behaviour of every release
    up to and including 0.2.1, with nothing added to it: the strict-sign
    certificate (``_Propagator.strict_sign``) does not cross a sub-jaxpr
    boundary in either direction, so ``assume(x > 0); 1/jit(f)(x)`` is
    UNKNOWN and so is the same query with the ``assume`` moved inside.
    ``boundary="transparent"`` lets that certificate cross the boundaries
    this walk already enters — **IN and OUT** for the unconditional
    wrappers of :data:`stelling.coverage.DEFAULT_TRANSPARENT`, **IN only**
    for a ``cond`` branch. Any other value raises :class:`ValueError`. The
    returned :class:`Propagation` records the position in
    :attr:`Propagation.boundary` and, when anything actually crossed, how
    many times in :attr:`Propagation.boundary_crossings`.

    **THE WORD IS OVERLOADED AND THAT IS DELIBERATE — SEE**
    :data:`_BOUNDARY_MODES`. ``coverage.DEFAULT_TRANSPARENT`` names the
    wrappers the walk ENTERS; this dial names whether the boundary it
    crosses CARRIES ANYTHING. Today's default is a transparent wrapper
    with an opaque boundary, which is not a contradiction but the exact
    shape of the limitation.

    **WHAT THIS DIAL DOES NOT REACH.** Only the strict-sign certificate
    crosses. The other value-annotation tables swapped at a descent are
    unchanged in both positions: the boxes, the maybe-NaN flags and the
    product-taints already crossed positionally and still do, and
    ``exact`` (the :class:`stelling.exactness.ExactSet`) still does not —
    that one governs WITHHOLDING, so carrying it would widen what may be
    REFUTED, which is the dangerous direction and a separate argument.
    ``scan``, ``while``, ``pjit`` and ``closed_call`` are untouched by
    either position: the walk does not enter them, so there is no
    boundary for anything to cross.

    **UNDER ``semantics="ieee"`` THE DIAL CHANGES NOTHING IN EITHER
    POSITION**, and that is enforced rather than left to the reader:
    :meth:`_Propagator._carries_signs` is the one gate and it tests the
    semantics as well as the dial, because the certificate is a claim
    about ℝ that is false on a flush-to-zero target.
    """
    _check_semantics(semantics)
    _check_assume_mode(assume_mode)
    _check_domain(domain, semantics)
    _check_boundary(boundary)
    # eagerly, so a typo'd profile name raises where it was written
    budget = resolve_libm_budget(libm_budget)
    if budget is not None and semantics != "ieee":
        raise ValueError(LIBM_BUDGET_REAL_MODE_REFUSAL)
    p = _Propagator(assume_mode, semantics, budget, boundary)
    if closed.jaxpr.invars:
        raise ir.TranscriptionError(
            "propagate expects a self-contained harness query (inputs declared "
            f"via any_array), got {len(closed.jaxpr.invars)} free invar(s)"
        )
    p.run(closed.jaxpr, list(closed.consts), [])
    # BEFORE anything reads this run's assume state: reconcile the ledger
    # against the assumes the query CONTAINS. The walk's ledger is total
    # over what the WALK SAW, and audit 0.2.0 S13 is that the walk does not
    # see a `scan` or `while_loop` body. It runs before the certificate and
    # before the withholding because both read `assume_dropped`, and a flag
    # set after its readers is a flag nobody read.
    _record_undescended_assumes(closed, p)
    # BEFORE the withholding: the non-emptiness certificate. It is an
    # INPUT to the withholding's shared decision, so it has to be computed
    # first; it is one-sided, so computing it can only ever leave the
    # withholding alone or lift it.
    p.region_inhabited = _region_witness(
        closed, p, assume_mode=assume_mode, semantics=semantics,
        boundary=boundary,
    )
    # FIRST: the run-scoped assume withholding. It runs before the
    # branch-reachability pass so that an obligation withheld here is not
    # also re-explained there — the branch pass skips anything no longer
    # `violated-over-set`, exactly as it did when this withholding was
    # applied inside the walk.
    _withhold_uncertified_refutations(p)
    _withhold_uncertified_branch_refutations(
        closed, p, assume_mode=assume_mode, semantics=semantics,
        boundary=boundary,
    )
    assumptions = set(p.assumptions)
    if p.region_inhabited:
        # THE STAMP SWAP. Both uncertified assumptions say "the
        # conditional claim may be vacuous" — true when written, during
        # the walk, before any witness existed, and FALSE on this run. A
        # stamped assumption is what a verdict claims to rest on, so a
        # known-false one is a disclosure defect whatever the verdict
        # says. Removed here rather than never written, because the walk
        # cannot know: the certificate is a whole-run answer computed
        # after it.
        assumptions.discard(UNCERTIFIED_NARROWING_ASSUMPTION)
        assumptions.discard(UNCERTIFIED_DROP_ASSUMPTION)
        assumptions.add(REGION_INHABITED_ASSUMPTION)
    if semantics == "ieee":
        # the mode-wide stamped assumptions: how ieee endpoints are
        # computed and what their soundness relies on, and the
        # subnormal-band indeterminacy (flush-vs-gradual is
        # target-dependent; band outcomes are never definite).
        # BOTH are format-parametric now. Their binary64 texts were
        # stamped verbatim on narrow-format verdicts, where both are
        # false — the endpoints WERE outward-rounded and the band applied
        # was the format's, not 2**-1022 (audit 0.2.0 M14). A binary64-only
        # run stamps the identical sentences it always did.
        fmts = _query_float_formats(closed)
        assumptions.add(iv.ieee_endpoint_assumption(fmts))
        assumptions.add(iv.subnormal_indeterminacy_assumption(fmts))
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
        if p.libm_declared:
            # A libm-riding transfer RAN, which means a budget covered it —
            # the transfer declines otherwise, so there is no path to a
            # populated set with no budget. Checked rather than assumed:
            # this line is the one that tells a reader the accuracy was
            # declared and not verified, and a verdict that quietly lost it
            # would be the S9/S11 defect again with better manners.
            if budget is None:  # pragma: no cover - unreachable by design
                raise RuntimeError(
                    f"libm-riding transfers {sorted(p.libm_declared)} ran "
                    f"with no declared budget: the decline gate has been "
                    f"bypassed and this verdict would carry no disclosure "
                    f"of the assumption it rests on"
                )
            assumptions.add(budget.render(sorted(p.libm_declared)))
    if boundary != "opaque":
        # THE DIAL'S POSITION, stamped whenever it is off the position
        # every recorded verdict in this project's history was taken in.
        # It is NOT conditioned on anything having crossed: a
        # boundary-transparent run on a query with no sub-jaxpr is still a
        # run under a different rule, and a reader comparing it against
        # the archive has to be told which rule.
        #
        # **AND THE `"opaque"` POSITION ADDS NO LINE AT ALL, WHICH IS A
        # DECISION AND NOT AN OVERSIGHT.** The acceptance criterion this
        # change was built to is that the default is byte-for-byte
        # today's behaviour, and a stamped line on every default run
        # moves the rendered stamp of every verdict in the archive. So
        # the position is recorded UNCONDITIONALLY on the structured
        # record (:attr:`Propagation.boundary`, which is what a
        # programmatic reader should consult) and stamped in prose only
        # where the prose would say something new. The cost is real and
        # is stated here rather than left to be discovered: a rendered
        # stamp with no boundary line means `"opaque"` BY CONVENTION, and
        # a reader who does not know the convention cannot tell it from a
        # stamp written before the dial existed — which is the same
        # thing, and is exactly why the convention is safe.
        assumptions.add(BOUNDARY_TRANSPARENT_POSITION)
        if p.boundary_crossings:
            # THE ACT, reported only where it happened. A disclosure that
            # rode on the position rather than on the crossing would say
            # "certificates crossed" about a run in which none did.
            assumptions.add(
                BOUNDARY_CROSSED_DISCLOSURE.format(n=p.boundary_crossings)
            )
    return Propagation(
        obligations=tuple(p.obligations),
        nonvacuity_checks=tuple(p.nonvacuity_checks),
        coverage=p.counter.freeze(),
        transfers_used=tuple(sorted(p.used.items())),
        assumptions=tuple(sorted(assumptions)),
        notes=tuple(p.notes),
        # WHICH QUERY THIS PROPAGATION IS ABOUT — the walk's own `closed`,
        # taken here and nowhere else. `query_identity` never raises, so a
        # query this walk could analyse but `content_hash()` cannot encode
        # still yields a `Propagation` (as it did before this field), and
        # the "" it carries is refused at every consumption site rather
        # than pairing with anything.
        query_sha256=query_identity(closed),
        semantics=semantics,
        boundary=boundary,
        boundary_crossings=p.boundary_crossings,
        assume_dropped=p.assume_dropped,
        narrowing_uncertified=p.narrowing_uncertified,
        region_inhabited=p.region_inhabited,
        top_boxes=_top_boxes(closed, p.env),
        relational_assumes=tuple(p.relational_assumes),
        assume_ledger=tuple(p.assume_ledger),
    )
