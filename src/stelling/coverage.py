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
  honored* — a constraint dropped rather than applied (the MVP's
  ``stelling_assume``). The counter measures semantic fidelity, not just
  primitive coverage: a no-op transfer must never count as known, or a
  dropped constraint hides inside 100%.

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
# per design/transparent-primitives.md (verified on jax 0.10.2).
DEFAULT_TRANSPARENT = frozenset({"jit", "custom_jvp_call", "custom_vjp_call", "remat2"})


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
        return "; ".join(parts)


class CoverageCounter:
    """Mutable counter for threading through a live interpreter pass."""

    def __init__(self) -> None:
        self._known = 0
        self._transparent = 0
        self._unreached = 0
        self._unknown: Counter[str] = Counter()
        self._inert: Counter[str] = Counter()

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

    def freeze(self) -> Coverage:
        unknown_total = sum(self._unknown.values())
        inert_total = sum(self._inert.values())
        return Coverage(
            total=self._known
            + self._transparent
            + unknown_total
            + self._unreached
            + inert_total,
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
        )


def sub_jaxprs(eqn: ir.JaxprEqn) -> Iterable[ir.Jaxpr]:
    """Yield every sub-jaxpr held in this equation's params, however nested."""
    pending = [v for _, v in eqn.params]
    while pending:
        item = pending.pop()
        if isinstance(item, ir.ClosedJaxpr):
            yield item.jaxpr
        elif isinstance(item, ir.Jaxpr):
            yield item
        elif isinstance(item, tuple):
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
