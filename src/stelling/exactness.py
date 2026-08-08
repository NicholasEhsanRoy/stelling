# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Exactness certification: which abstract boxes equal their true value set.

The principle this module exists to hold in one place: **a sound
over-approximation certifies emptiness (empty abstraction ⟹ empty truth)
but never nonemptiness — the abstraction may have added exactly the
points observed; nonemptiness claims require exact knowledge (declared
sets, exact points, concrete witnesses).**

Concretely, for interval propagation (audit F7): every transfer output is
an over-approximation (rounding pads, correlation-blind arithmetic), so a
nonempty meet of a transfer output's box with an assumed half-space does
NOT certify the true assumed region nonempty — the nonempty overlap may
consist entirely of points the abstraction invented. Only a box known to
EQUAL its variable's true value set — a declared input's closed box, an
exact-point constant — supports the nonemptiness claim, plus the one
box-independent channel: a predicate definitely true over the whole box
is true over the whole true set a fortiori (audit F8).

The last clause of that principle — *concrete witnesses* — is the one
this module gained a second function for. Exactness is the NEGATIVE
channel: it says which boxes may be read as their own truth. A point
witness is the POSITIVE channel: an actual member of the declared set,
at which every assume of the query is definitely true, settles
nonemptiness outright and needs no box to be exact.

:class:`ExactSet` is the per-scope bookkeeping (the per-var exact set and
its maintenance rules); :func:`certifies_nonemptiness` is the
per-variable certification decision; :func:`certifies_point_witness` is
the positive channel's decision — *did one probed point satisfy every
assume this query wrote?*; :func:`certifies_set_refutation` is the
RUN-level decision layered on both — *given this run's whole assume
state, is a set-level refutation certified?* All four are consumed by
:mod:`stelling.propagate`'s constraining-assume machinery and are
importable by any future layer that must certify a region inhabited, or
must decide whether a set-level refutation is owed.

**Why the run-level decision lives here rather than in either judging
leg.** Two legs can mint a set-level refutation — the interval
propagator and the affine refinement — and they must not answer this
question separately. The affine refinement is a post-pass over a
finished propagation, so it reads a whole-run quantity by ARCHITECTURE,
not by choice; the interval leg reads one because that is the rule. Two
legs agreeing for two different reasons is an agreement that breaks
silently the first time the refinement is restructured to run inline or
to interleave, with nothing in the tree catching it. So the rule is
stated ONCE, here, and both legs consult this function.

Maintenance rules (the complete list — anything not named here is
NON-exact by default, the conservative direction):

* a ``stelling_any`` output is exact: the declared closed box IS the
  declared value set (no rounding at declaration);
* a constant decoded to a per-element exact point is exact (a bracketed
  decode — e.g. an int above 2**53 — is NOT its value set and stays
  non-exact);
* meets of exact sets are exact, so a variable narrowed only by assumes
  keeps its exactness (no re-marking needed: narrowing does not remove a
  var from the set);
* every transfer output is non-exact;
* scope boundaries never leak exactness: each sub-jaxpr scope starts a
  fresh empty :class:`ExactSet`, and scope invars never inherit exactness
  from the operands that bound them (selector correlation would reopen
  the audit-F7 hole; pinned by
  ``test_branch_invars_never_inherit_exactness_selector_correlation``).

Zero-dep and structure-agnostic: the set holds opaque variable ids and
the point test takes raw endpoint sequences, so nothing here depends on
the interval representation.
"""

from __future__ import annotations

__all__ = [
    "ExactSet",
    "certifies_nonemptiness",
    "certifies_point_witness",
    "certifies_set_refutation",
]


class ExactSet:
    """The per-scope set of variable ids whose box is EXACT — equal to the
    variable's true value set, not an over-approximation of it."""

    __slots__ = ("_ids",)

    def __init__(self) -> None:
        self._ids: set[int] = set()

    def mark_declared(self, var_id: int) -> None:
        """Mark a declared-set variable (``stelling_any`` output): the
        declared closed box IS the declared value set."""
        self._ids.add(var_id)

    def mark_if_point(self, var_id: int, los, his) -> bool:
        """Mark a decoded constant exact iff every element is a point
        (``lo == hi``): a point decode is the value itself; a genuine
        bracket is not its value set. Returns whether it was marked."""
        if all(lo == hi for lo, hi in zip(los, his)):
            self._ids.add(var_id)
            return True
        return False

    def __contains__(self, var_id: int) -> bool:
        return var_id in self._ids


def certifies_nonemptiness(
    exact: ExactSet, var_id: int, *, definitely_true: bool
) -> bool:
    """The certification decision: does a nonempty meet of ``var_id``'s
    box with an assumed region certify the TRUE assumed region nonempty?

    Yes iff the box is exact (the overlap consists of real points) or the
    predicate was definitely true over the whole box (true on the box
    superset ⟹ true on the whole true set, so the assumed region equals
    the reachable set — audit F8). Everything else is an
    over-approximation and certifies nothing: the caller must treat the
    precondition's satisfiability as uncertified.
    """
    return definitely_true or var_id in exact


def certifies_point_witness(
    *, required_assumes: frozenset, witnessed_assumes: frozenset
) -> bool:
    """The POSITIVE channel: did one probed point of the declared set
    satisfy **every** assume this query wrote — so the assumed region is
    inhabited, whatever the boxes could or could not certify?

    ``required_assumes`` is the identity of every ``stelling_assume``
    equation the query CONTAINS, read statically off the IR, sub-jaxprs
    included. ``witnessed_assumes`` is the identity of every assume the
    probe run EVALUATED and found definitely true at the point it pinned.

    **Why the requirement is static and the witness dynamic — this is the
    whole safety of the function.** A probe pins each declaration to a
    point, which forces conds, so a probe walks ONE branch. An assume
    sitting in the branch it did not walk is never evaluated and never
    appears in ``witnessed_assumes``; the subset test then fails and no
    certificate is issued. Were the requirement gathered dynamically from
    the probe instead, a query whose only doubtful assume lives in an
    untaken branch would certify itself by walking around it — a
    branch-scoped empty precondition wearing a top-level witness's
    clothes.

    **What that does and does not say about branch-scoped assumes, since
    an earlier version of this paragraph said the stronger thing.** It
    declines every assume a probe walked AROUND. It does NOT decline every
    branch-scoped assume: forcing a cond forces it EITHER WAY, so a probe
    whose pinned point takes the branch evaluates the assume in it, and an
    assume that is definitely true at that point is witnessed and
    certified like any other. Measured, and pinned by
    ``test_an_assume_the_probe_walks_INTO_is_witnessed_and_certified``: a
    query whose ONLY assume sits inside a ``lax.cond`` branch certifies on
    probe 1, the declared box's high corner, and its withheld violation
    correctly comes back. That recovery is sound — at that point the
    program really does take the branch and really does satisfy the
    assume. What protects branch-scoped VIOLATIONS is a second,
    independent mechanism in :mod:`stelling.propagate`
    (``_reachability_witnesses`` returns the empty set on any run with
    ``any_constrained or assume_dropped``, which is every run this
    certificate can fire on), not this subset test.

    The asymmetry still costs recoveries, and in exactly the shape the
    overstated sentence claimed it prevented: a region inhabited only via
    the UNTAKEN branch — every admissible point walks the side WITHOUT the
    assume — is required-and-not-witnessed on every probe, so a sound
    refutation is withheld
    (``test_a_region_inhabited_only_via_the_UNTAKEN_branch_is_not_recovered``).

    **Empty ``required_assumes`` answers no.** A query with no assume has
    nothing whose satisfiability is in doubt, so it never reaches the
    withholding this lifts; answering yes there would be a certificate
    with no content, and a later caller reading a True would be reading a
    claim nobody established.

    **That guard is DEAD in production, and the docstring says so rather
    than presenting a constant as a live decision.** The one production
    caller (:func:`stelling.propagate._region_witness`) runs
    ``if not required: return False`` before its probe loop, so every call
    that reaches here from the propagator carries a non-empty set.
    MEASURED, not reasoned: instrumenting this function to record its
    caller and its argument size, and running the whole suite on
    jax 0.11.0, records **2203 calls from ``_region_witness``, non-empty
    at 2203 of 2203**, and **1544 of 1544** over
    ``scratchpad/pin/corpus_pin.py``. The only empty-set calls anywhere in
    the tree are the **2** from
    ``test_the_point_witness_decision_is_one_sided_and_static``, which
    exercises this guard directly and is the reason it stays: an importer
    that has not got the propagator's early return is not a hypothetical,
    it is the documented purpose of this module.

    **ONE-SIDED, and this is the load-bearing sentence.** A False answer
    means *no witness was found*. It is **not** a proof that the region is
    empty: the probe grid is finite, the arithmetic that evaluates a
    predicate at a point is an enclosure and may be indeterminate, and the
    search is capped by declared size. A caller may use a True to stop
    withholding; a caller may NEVER use a False to conclude anything —
    everything a False leaves in place was already in place for its own
    reasons.
    """
    return bool(required_assumes) and required_assumes <= witnessed_assumes


def certifies_set_refutation(
    *,
    nonemptiness_certified: bool,
    assume_dropped: bool,
    region_inhabited: bool = False,
) -> bool:
    """Given a run's WHOLE assume state, is a set-level refutation
    certified? The run-level layer over :func:`certifies_nonemptiness`
    and :func:`certifies_point_witness`.

    ``nonemptiness_certified`` is the conjunction, over the whole run, of
    every answer :func:`certifies_nonemptiness` gave where a constraining
    assume narrowed: False once ANY narrowing was applied to a region
    this module could not certify inhabited. ``assume_dropped`` says an
    assume was not applied at all, so the judged set is a SUPERSET of the
    assumed region and that region was never shown inhabited either.
    They are separate arguments and not one merged "uncertified" bit
    because the two name different mechanisms, and the sentence that
    explains a withholding has to name the one that actually happened:
    quoting the over-approximated-intermediate mechanism at a run whose
    narrowing target was exact is a false statement of mechanism in the
    one sentence whose job is to explain the withholding.

    **The scope is the WHOLE QUERY, not the obligations traced after the
    assume.** An assume is a precondition on the query. The refusal for a
    *detectably* empty assumed region
    (:class:`stelling.propagate.UnsatisfiableAssumptionError`) already
    ends the run whole — obligations written ABOVE the assume included —
    because if the assumed region is empty then *every* obligation is
    vacuously true, not only the ones below the line. The possibly-empty
    case is the same fact known less precisely, so it takes the same
    scope. A run-level state and a run-level answer are what that scope
    means in code: no argument here names an obligation, an equation or a
    position, so no caller can accidentally make the answer depend on
    where in the trace it was asked.

    **One-sided, and this is load-bearing.** A False answer withholds
    *violations only*. A discharge over a superset implies the discharge
    over the intended set, so suppressing discharges costs real verdicts
    for nothing — ``solvers.py`` tried the two-sided version on its own
    leg and reverted it. Nothing here may be used to decline a run
    wholly or to touch a ``discharged`` status.

    **``region_inhabited`` — the certificate, and why it is an INPUT
    here rather than a channel of its own.** Both of the other two
    arguments withhold for exactly one reason: the assumed region *may be
    empty*, in which case every obligation is vacuously true and no
    refutation is owed. Neither says the judged set is wrong — a
    narrowing is a meet with a CLOSED half-space and a drop only widens,
    so in both cases the judged set is a superset of the assumed region
    and a definite violation over it is a violation at every point of
    that region. What is missing in both is a point. Supply one — a
    member of the declared set at which every assume of the query holds
    (:func:`certifies_point_witness`) — and the region is inhabited, the
    violation is a violation of something, and REFUTED is owed.

    So it is a third argument to THIS function and not a fourth
    consultation somewhere else, because the question *"is a set-level
    refutation certified on this run?"* still has exactly one answer and
    two legs still have to get the same one. A leg that consulted the
    witness separately would be back to two legs agreeing by coincidence,
    which is the arrangement this function exists to end.

    It defaults to ``False`` — the pre-certificate answer — so a caller
    that does not know about witnesses gets the conservative behaviour
    rather than a signature error, and so the disjunction below can only
    ever turn withholding OFF, never on. The one-sidedness above is
    unchanged by it: a True here can restore a withheld
    ``violated-over-set`` and can do nothing else at all.
    """
    return region_inhabited or (nonemptiness_certified and not assume_dropped)
