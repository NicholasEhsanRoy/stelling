# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Registry-independent primitive census over transcribed IR.

``coverage.measure`` answers *how much of this query could we analyse,
given the registry we have*. The census asks the prior question — *what
does real code contain* — and depends on no registry: every equation at
every depth is counted, transparent or not, known or not.

**A census is an inventory, not a value signal.** It makes no claim about
whether stelling finds bugs, so it needs no pre-registration
(``design/value-model.md``) — and the moment one of its numbers is cited
as evidence that stelling is useful, it has become the thing it isn't.
Its jobs are exactly two: set the transfer-registry priority order, and
date-stamp what the ecosystem's code is made of.

Context taxonomy, per equation: ``top`` — depth 0; ``transparent`` —
reachable from the top through transparent wrappers only
(:data:`stelling.coverage.DEFAULT_TRANSPARENT`); ``nested`` — beneath at
least one non-transparent structured primitive (``cond``, ``scan``,
``while``, …), which a default forward interpreter reaches only with a
dedicated transfer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from stelling import ir
from stelling.coverage import DEFAULT_TRANSPARENT, sub_jaxprs

CONTEXTS = ("top", "transparent", "nested")

# The Stage-1 wedge's target primitives: indexing operations whose silent
# clamp/drop behaviour under jit is the bug class (design/value-model.md).
def is_wedge_primitive(name: str) -> bool:
    return (
        name in {"gather", "dynamic_slice", "dynamic_update_slice"}
        or name.startswith("scatter")
    )


@dataclass(frozen=True)
class PrimitiveStats:
    name: str
    count: int  # total equations across the corpus
    breadth: int  # number of targets the primitive appears in
    contexts: tuple[tuple[str, int], ...]  # nonzero (context, count), in CONTEXTS order
    wedge: bool


@dataclass(frozen=True)
class Census:
    targets: tuple[str, ...]
    total: int
    primitives: tuple[PrimitiveStats, ...]  # sorted by (-breadth, -count, name)

    def top(self, n: int) -> tuple[str, ...]:
        return tuple(p.name for p in self.primitives[:n])

    def markdown_table(self) -> str:
        n = len(self.targets)
        lines = [
            "| primitive | breadth | count | top | transparent | nested | wedge |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in self.primitives:
            ctx = dict(p.contexts)
            lines.append(
                f"| `{p.name}` | {p.breadth}/{n} | {p.count} "
                f"| {ctx.get('top', 0)} | {ctx.get('transparent', 0)} "
                f"| {ctx.get('nested', 0)} | {'✓' if p.wedge else ''} |"
            )
        return "\n".join(lines)


class CensusAccumulator:
    def __init__(self) -> None:
        self._targets: list[str] = []
        self._contexts: dict[str, Counter] = {}  # primitive -> Counter(context)
        self._users: dict[str, set[str]] = {}  # primitive -> targets

    def add(
        self,
        target: str,
        closed_jaxpr: ir.ClosedJaxpr,
        *,
        transparent: frozenset[str] = DEFAULT_TRANSPARENT,
    ) -> None:
        if target not in self._targets:
            self._targets.append(target)
        stack: list[tuple[ir.Jaxpr, str]] = [(closed_jaxpr.jaxpr, "top")]
        while stack:
            jaxpr, context = stack.pop()
            for eqn in jaxpr.eqns:
                self._contexts.setdefault(eqn.primitive, Counter())[context] += 1
                self._users.setdefault(eqn.primitive, set()).add(target)
                child = (
                    "transparent"
                    if context != "nested" and eqn.primitive in transparent
                    else "nested"
                )
                for sub in sub_jaxprs(eqn):
                    stack.append((sub, child))

    def freeze(self) -> Census:
        stats = tuple(
            sorted(
                (
                    PrimitiveStats(
                        name=name,
                        count=sum(ctxs.values()),
                        breadth=len(self._users[name]),
                        contexts=tuple(
                            (c, ctxs[c]) for c in CONTEXTS if ctxs.get(c)
                        ),
                        wedge=is_wedge_primitive(name),
                    )
                    for name, ctxs in self._contexts.items()
                ),
                key=lambda p: (-p.breadth, -p.count, p.name),
            )
        )
        return Census(
            targets=tuple(self._targets),
            total=sum(p.count for p in stats),
            primitives=stats,
        )
