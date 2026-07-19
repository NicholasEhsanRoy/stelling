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
certification decision. Both are consumed by
:mod:`stelling.propagate`'s constraining-assume machinery and are
importable by any future layer that must certify a region inhabited.

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

__all__ = ["ExactSet", "certifies_nonemptiness"]


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
