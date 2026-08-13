# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Output-reachability analysis over :mod:`stelling.ir`.

The "reaches-output" conjunct: a backward walk from a jaxpr's output
variables through its equation graph, collecting every variable whose
value is transitively needed to produce an output.  A violated
obligation whose predicate operand is NOT in this set is a violation on
a dead variable -- one the caller never observes -- and is downgraded
from REFUTED to UNKNOWN with a note.

This is a standard fixed-point dataflow analysis (one pass, no
iteration needed, because jaxpr equations are in topological order and
outputs are at the end).
"""

from __future__ import annotations

from stelling import ir


def reaches_output(jaxpr: ir.Jaxpr) -> frozenset[int]:
    """Return the set of Var IDs that transitively reach an output of *jaxpr*.

    Algorithm: seed the live set with the jaxpr's output variables AND
    every ``stelling_assert`` equation's outvars, then walk equations in
    REVERSE order.  For each equation whose output vars intersect the
    live set, add its input vars to the live set.

    ``stelling_assert`` outvars are seeded unconditionally because an
    assert is a DECLARATION — the user stated an obligation about its
    operand, and that operand is live by intent regardless of whether
    the harness function happens to return the assert's output value.
    Without this, ``assert_(pred)`` without a ``return`` makes the
    predicate operand appear dead, downgrading real violations.
    """
    live: set[int] = set()
    # Seed: the jaxpr's output atoms (Vars only; Literals have no ID).
    for atom in jaxpr.outvars:
        if isinstance(atom, ir.Var):
            live.add(atom.id)
    # Seed: every stelling_assert outvar — asserts are always relevant.
    for eqn in jaxpr.eqns:
        if eqn.primitive == "stelling_assert":
            for out in eqn.outvars:
                live.add(out.id)

    # Walk in reverse topological order.
    for eqn in reversed(jaxpr.eqns):
        # If any of this equation's outvars is live, its invars become live.
        if any(out.id in live for out in eqn.outvars):
            for atom in eqn.invars:
                if isinstance(atom, ir.Var):
                    live.add(atom.id)
    return frozenset(live)


def defined_vars(jaxpr: ir.Jaxpr) -> frozenset[int]:
    """Return ALL Var IDs defined in *jaxpr*'s top-level scope.

    This includes constvars, invars, and all equation outvars.  Used to
    determine whether an obligation's operand_var_ids belong to THIS
    jaxpr's scope (and thus can be judged by :func:`reaches_output`)
    or to a sub-jaxpr's scope (where the top-level walk cannot decide).
    """
    ids: set[int] = set()
    for v in jaxpr.constvars:
        ids.add(v.id)
    for v in jaxpr.invars:
        ids.add(v.id)
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            ids.add(v.id)
    return frozenset(ids)
