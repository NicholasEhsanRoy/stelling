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

:class:`ExactSet` is the per-scope bookkeeping (the per-var exact set and
its maintenance rules); :func:`certifies_nonemptiness` is the
per-variable certification decision; :func:`certifies_set_refutation` is
the RUN-level decision layered on it — *given this run's whole assume
state, is a set-level refutation certified?* All three are consumed by
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

__all__ = ["ExactSet", "certifies_nonemptiness", "certifies_set_refutation"]


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


def certifies_set_refutation(
    *, nonemptiness_certified: bool, assume_dropped: bool
) -> bool:
    """Given a run's WHOLE assume state, is a set-level refutation
    certified? The run-level layer over :func:`certifies_nonemptiness`.

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
    """
    return nonemptiness_certified and not assume_dropped
