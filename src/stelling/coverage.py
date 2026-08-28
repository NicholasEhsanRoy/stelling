# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""⊤-coverage instrument: how much of a query the analysis actually saw.

The pre-registered falsifier in ``design/value-model.md`` makes this a
scientific control, not UX: a null result at low coverage is not a null
result, so "found nothing in codebase X" must never travel without the
number that says how much of X was looked at. The instrument is a
**quantity** — equations interpreted vs. fallen to ⊤ vs. never reached —
with the offending primitive names riding along only to set registry
priority order.

Counting semantics, matched to what an interpreter would traverse:

* **transparent** wrapper primitives (``design/transparent-primitives.md``)
  are descended into; the wrapper itself is neither known nor ⊤.
* sub-jaxprs under **known** primitives are also counted as reached — a
  registered control-flow transfer analyzes its body, so the body's own
  coverage still matters.
* equations inside the sub-jaxprs of a *non-transparent unknown* primitive
  are **unreached**: the analysis never looked at them, and counting the
  wrapper as one ⊤ out of N would understate exactly the confound the
  falsifier guards against.
* **inert** equations have a known primitive whose *semantics were not
  honored* — a constraint dropped rather than applied (an
  ``stelling_assume`` whose shape admits no sound narrowing). The counter
  measures semantic fidelity, not just primitive coverage: a no-op
  transfer must never count as known, or a dropped constraint hides
  inside 100%.
* **constrained** equations are assumes whose constraint *was* honored by
  narrowing the propagated domain (a superset of the true assumed
  region). Counted alongside ``inert`` — its own category, deliberately
  outside ``known``: a constrained assume makes the verdict *conditional*
  on the precondition, and conditionality must not hide inside the
  fully-unconditional fraction any more than a drop may.

Two entry points: :func:`measure` walks a finished :class:`stelling.ir`
query (usable today, on transcribed code); :class:`CoverageCounter` is the
running counter a live interpreter threads through, so instrumentation
exists from the interpreter's first version instead of being retrofitted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from stelling import ir

# The wrapper primitives whose correct transfer is descend-into-sub-jaxpr,
# per design/transparent-primitives.md (membership verified on every tested
# jax series: 0.10.2 and 0.11.0). The membership is series-stable; the
# CONTAINER each member's body arrives in is not, and remat2's moves --
# reach a body through call_body, never through an isinstance test.
DEFAULT_TRANSPARENT = frozenset({"jit", "custom_jvp_call", "custom_vjp_call", "remat2"})


# THE PUBLISHED USER-FACING NAME OF A DECLARATION.
#
# `stelling.obligation.SliceInput` names the k-th declaration's SMT constant
# `x{k}` (and `x{k}_{i}` per element of an array one), `stelling.smt` emits
# under those names, `stelling.reproduce` reads witness values back by them,
# and a REFUTED verdict prints them at the user. That numbering is the
# DECLARATION ORDER of the query; it is not, and never was, an `ir.Var.id`.
#
# NOT THE ONE MINTER, AND THE CLAIM THAT IT WAS IS WITHDRAWN (audit 0.2.0
# B8a FIXUP). `declaration_name` was added by item 5 for `propagate`'s
# messages and described as the single place this name is spelled. It is
# one spelling of six. The other five, measured:
#
#     obligation._Slicer.slice   f"x{k}"      (the scalar SliceInput)
#     obligation._Slicer.slice   f"x{k}_{i}"  (the per-element SliceInput)
#     reproduce._envelope        f"x{k}"
#     reproduce._point           f"x{k}" if not shape else f"x{k}_{i}"
#     reproduce._flat_witness    f"x{k}" if not shape else f"x{k}_{i}"
#
# All six agree today, and the emission/witness/reproducer trio is pinned
# by the round-trip tests that read a witness back by name.
#
# CHANGING `DECLARATION_NAME_PREFIX` TODAY WOULD NOT RENAME ANYTHING — it
# would move `propagate`'s messages ALONE, off the names the emission,
# the witness and the reproducer still spell by hand, which is M3's
# two-namespace defect re-created in the other direction. Unifying the six
# is a real change with a real test surface (an emitted script, a witness
# document and a reproducer all read by name) and belongs in a batch that
# can drive it, not in a comment that asserts it.
#
# Audit 0.2.0 B8a, item 5 (M3): `propagate`'s assume messages spelled their
# subject `var {atom.id}` — the internal IR id — beside witnesses spelled
# `x{k}`. Two 0-based numeric namespaces printed at one reader with nothing
# relating them, and they genuinely disagree: measured on `aabb58d`, a
# two-declaration query numbers declaration 0 as IR var 1 and declaration 1
# as IR var 2, so an unsatisfiable-assume message about declaration 1 read
# "var 2" while the witness for it read "x1".
DECLARATION_NAME_PREFIX = "x"


def declaration_name(k: int) -> str:
    """The published SMT/witness name of the ``k``-th declaration — the
    WHOLE name for a scalar declaration and the PREFIX of one for an array,
    whose witness carries `x{k}_{i}` per flat element."""
    return f"{DECLARATION_NAME_PREFIX}{k}"


@dataclass(frozen=True)
class Coverage:
    total: int  # every equation in the query, at any depth
    known: int  # equations with a registered transfer function
    transparent: int  # wrapper equations descended through
    unknown: int  # equations reached but fallen to ⊤
    unreached: int  # equations inside sub-jaxprs of unknown primitives
    unknown_primitives: tuple[tuple[str, int], ...]  # (name, ⊤ count), most frequent first
    inert: int = 0  # known primitive, semantics NOT honored (dropped constraint)
    inert_primitives: tuple[tuple[str, int], ...] = ()
    constrained: int = 0  # assumes honored by narrowing (verdict conditional)
    constrained_primitives: tuple[tuple[str, int], ...] = ()
    # conjuncts NOT applied inside assumes counted constrained (a mixed
    # conjunction's relational/unsupported/branch-vacuous half). Counted
    # per conjunct, NOT part of `total` (the equation is already counted
    # once, as constrained) — but rendered in the summary, so a partial
    # drop cannot hide inside "CONSTRAINED" (audit F5).
    dropped_conjuncts: int = 0

    @property
    def fraction_known(self) -> float:
        """known / total; an empty query counts as fully covered."""
        return self.known / self.total if self.total else 1.0

    def summary(self) -> str:
        parts = [f"{self.total} eqns: {self.known} known ({self.fraction_known:.0%})"]
        if self.transparent:
            parts.append(f"{self.transparent} transparent")
        if self.unknown:
            names = ", ".join(f"{n} ×{c}" for n, c in self.unknown_primitives)
            parts.append(f"{self.unknown} ⊤ across {len(self.unknown_primitives)} primitives ({names})")
        if self.unreached:
            parts.append(f"{self.unreached} unreached")
        if self.inert:
            names = ", ".join(f"{n} ×{c}" for n, c in self.inert_primitives)
            parts.append(f"{self.inert} constraint(s) DROPPED ({names})")
        if self.constrained:
            names = ", ".join(f"{n} ×{c}" for n, c in self.constrained_primitives)
            parts.append(f"{self.constrained} assume(s) CONSTRAINED ({names})")
        if self.dropped_conjuncts:
            parts.append(
                f"{self.dropped_conjuncts} conjunct(s) DROPPED inside "
                f"constrained assume(s)"
            )
        return "; ".join(parts)


class CoverageCounter:
    """Mutable counter for threading through a live interpreter pass."""

    def __init__(self) -> None:
        self._known = 0
        self._transparent = 0
        self._unreached = 0
        self._unknown: Counter[str] = Counter()
        self._inert: Counter[str] = Counter()
        self._constrained: Counter[str] = Counter()
        self._dropped_conjuncts = 0

    def record_known(self, primitive: str) -> None:
        self._known += 1

    def record_transparent(self, primitive: str) -> None:
        self._transparent += 1

    def record_unknown(self, primitive: str) -> None:
        self._unknown[primitive] += 1

    def record_unreached(self, primitive: str) -> None:
        self._unreached += 1

    def record_inert(self, primitive: str) -> None:
        self._inert[primitive] += 1

    def record_constrained(self, primitive: str) -> None:
        self._constrained[primitive] += 1

    def record_dropped_conjunct(self) -> None:
        """One conjunct of a constrained assume that was NOT applied
        (relational/unsupported/branch-vacuous) — summary-visible, outside
        the equation total (audit F5)."""
        self._dropped_conjuncts += 1

    def freeze(self) -> Coverage:
        unknown_total = sum(self._unknown.values())
        inert_total = sum(self._inert.values())
        constrained_total = sum(self._constrained.values())
        return Coverage(
            total=self._known
            + self._transparent
            + unknown_total
            + self._unreached
            + inert_total
            + constrained_total,
            known=self._known,
            transparent=self._transparent,
            unknown=unknown_total,
            unreached=self._unreached,
            unknown_primitives=tuple(
                sorted(self._unknown.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            inert=inert_total,
            inert_primitives=tuple(
                sorted(self._inert.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            constrained=constrained_total,
            constrained_primitives=tuple(
                sorted(self._constrained.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            dropped_conjuncts=self._dropped_conjuncts,
        )


def call_body(eqn: ir.JaxprEqn) -> ir.ClosedJaxpr | None:
    """The body a transparent call wrapper carries, closed, or None.

    THE CANONICAL ACCESSOR for a wrapper's callee, for the same reason
    :func:`sub_jaxprs` is the canonical one for nesting: every descent that
    hand-rolled ``isinstance(v, ir.ClosedJaxpr)`` was reading a fact about
    the jax that produced the param, not about the callee.

    Measured across the two tested series, all four members of
    :data:`DEFAULT_TRANSPARENT`::

        primitive          param         jax 0.10.2    jax 0.11.0
        jit                jaxpr         ClosedJaxpr   ClosedJaxpr
        remat2             jaxpr         Jaxpr         ClosedJaxpr
        custom_jvp_call    call_jaxpr    ClosedJaxpr   ClosedJaxpr
        custom_vjp_call    call_jaxpr    ClosedJaxpr   ClosedJaxpr

    ``remat2`` is the single cell that moves, and it moves because jax 0.11
    merged ``Jaxpr`` and ``ClosedJaxpr`` into one class — so on 0.11 no
    transcribed param is ever a bare :class:`stelling.ir.Jaxpr` at all. A
    ``ClosedJaxpr``-only test therefore found nothing for ``remat2`` on
    0.10 and the wrapper was left opaque: measured end to end, a
    ``jax.checkpoint`` harness that is VERIFIED on 0.11 came back UNKNOWN
    on 0.10 with "transparent 'remat2' could not be inlined (no sub-jaxpr
    ...)". Safe in direction — the wrapper is refused, not misread — but a
    capability that silently depends on the jax series.

    A bare jaxpr is closed over an EMPTY const tuple, which is lossless:
    it has no consts of its own, and if it carries ``constvars`` the
    callers' ``len(constvars) == len(consts)`` guard then refuses the
    inline, which is the correct answer for a body whose consts are not
    available to bind.

    ``ClosedJaxpr`` is preferred over a bare ``Jaxpr`` when an equation
    somehow holds both, so that this cannot change which param is chosen
    on any input the previous code already handled.
    """
    for _, v in eqn.params:
        if isinstance(v, ir.ClosedJaxpr):
            return v
    for _, v in eqn.params:
        if isinstance(v, ir.Jaxpr):
            return ir.ClosedJaxpr(jaxpr=v, consts=())
    return None


def sub_jaxprs(eqn: ir.JaxprEqn) -> Iterable[ir.Jaxpr]:
    """Yield every sub-jaxpr held in this equation's params, however nested.

    **THE SEQUENCE TEST IS ``(tuple, list)``, AND THE ``list`` HALF IS NOT
    DEFENSIVE BREADTH.** This walk is what
    :func:`stelling.propagate._assume_equation_ids` collects the STATIC assume
    set with, so a sub-jaxpr it does not enter is a ``stelling_assume`` that
    exists in the query and is missing from the requirement — the exact
    direction that fails OPEN. Audit B9 measured it: a hand-built
    :class:`ir.JaxprEqn` holding a sub-jaxpr in a Python ``list`` was invisible
    here, and ``propagate`` accepted the equation and reported *static ids 0 /
    ledger 0 / assume_dropped False / covers True* — the pre-fix state
    presented as a satisfied postcondition.

    **NO PATH INSIDE STELLING PRODUCES ONE — AND THE ARGUMENT FOR THAT HAD
    TO BE REBUILT BEFORE IT COULD BE BELIEVED.** This paragraph read
    *"stelling.harness.trace and ir.JaxprEqn.from_dict both build tuples,
    so this is unreachable from a real query"*. **There is no
    ``ir.JaxprEqn.from_dict``**: the library's only ``from_dict`` is
    :meth:`ir.ClosedJaxpr.from_dict`, and equations are built by the
    module-level :func:`ir._decode`. So half of a two-item enumeration named
    a method that has never existed, which is another way of saying nobody
    had checked it — and the enumeration was the wrong shape besides.
    Measured by AST at `9b5b496` on 2026-08-28, ``src/`` constructs
    ``ir.JaxprEqn`` at **nine** sites, not two, and a list of the routes
    somebody remembered is exactly what a tenth route walks past.

    Re-derived over the routes as a class rather than by name, the
    conclusion SURVIVES, and on the document side it is stronger than the
    sentence it replaces:

    * **Seven of the nine originate nothing.** :mod:`stelling.vacuity`'s
      bound-widener, :mod:`stelling.obligation`'s renumberer, alias
      resolver and probe builder, and :mod:`stelling.propagate`'s three
      gauge equations either hand ``eqn.params`` through untouched or
      rebuild it as ``tuple(...)`` over a mapping they own outright. None
      of them puts a sub-jaxpr into a params value at all, so whatever
      container one holds is the container an originating route made.
    * **The tracing route NORMALISES.** Every param goes through one arm of
      :meth:`_jax_compat._Transcriber.param`, ``isinstance(v, (tuple,
      list))`` → ``tuple(...)``, so a jax primitive that stashes its
      branches in a list is transcribed to a tuple by construction and not
      by luck. :func:`stelling.harness.trace` is the door onto that route.
    * **The document route cannot SPELL a list.** :func:`ir._decode`
      accepts a scalar or a tagged mapping and refuses everything else; a
      sequence is written ``{"k": "tuple", "items": [...]}`` and read back
      through ``tuple(...)``. Driven on this tree at `9b5b496` on
      2026-08-28 — encode a ``cond`` equation whose ``branches`` param is a
      pair of :class:`ir.ClosedJaxpr`, round-trip it, then put the bare
      JSON array in place of the tagged mapping::

          round-trip of `branches`   tuple
          the bare array instead     ValueError: malformed IR
                                     serialization: unexpected list

      A route that REFUSES the list is a stronger fact than one that
      happens to build tuples, and it is the half the old sentence got
      closest to while pointing at the wrong door.

    It is hardened anyway because three rules now rest on this walk's
    totality (B3's non-emptiness gate, ``ledger_covers``, and the
    ``REGION_NOT_ASKED`` tightening), ``JaxprEqn`` is a plain dataclass any
    caller can construct, and a params mapping is untyped by construction:
    the tuple-ness of every sub-jaxpr container is a property of today's
    decoder, not of the type, so a future decoder that passed a list
    through would silently SHRINK the requirement rather than fail. Nothing
    else in this function distinguishes the two sequence types, and ``list``
    is the only other one a decoder plausibly yields.
    """
    pending = [v for _, v in eqn.params]
    while pending:
        item = pending.pop()
        if isinstance(item, ir.ClosedJaxpr):
            yield item.jaxpr
        elif isinstance(item, ir.Jaxpr):
            yield item
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        elif isinstance(item, ir.NamedTupleParam):
            pending.extend(v for _, v in item.fields)


def measure(
    closed_jaxpr: ir.ClosedJaxpr,
    known: frozenset[str] | set[str],
    *,
    transparent: frozenset[str] = DEFAULT_TRANSPARENT,
) -> Coverage:
    """Measure ⊤-coverage of a query against a set of known primitives."""
    counter = CoverageCounter()
    stack: list[tuple[ir.Jaxpr, bool]] = [(closed_jaxpr.jaxpr, True)]
    while stack:
        jaxpr, reached = stack.pop()
        for eqn in jaxpr.eqns:
            if not reached:
                counter.record_unreached(eqn.primitive)
                descend_reached = False
            elif eqn.primitive in transparent:
                counter.record_transparent(eqn.primitive)
                descend_reached = True
            elif eqn.primitive in known:
                counter.record_known(eqn.primitive)
                descend_reached = True
            else:
                counter.record_unknown(eqn.primitive)
                descend_reached = False
            for sub in sub_jaxprs(eqn):
                stack.append((sub, descend_reached))
    return counter.freeze()
