# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Running a generated harness through the tool, and counting what happened.

Two jobs, and the second one is the more important.

**Running.** :func:`run` calls ``stelling.preconditions.check`` on a generated
:class:`~_grammar.Spec` and normalises the result to a small immutable
:class:`Outcome`. A generated harness is very often one the tool is *entitled*
to refuse — an empty box, a bound with no value of its dtype, an unregistered
primitive, a declaration jax itself will not build. Those raise, and a refusal
is not a finding, so :func:`run` returns ``None`` for them and records the
exception type.

**Counting — the anti-vacuity floor.** A property that finds nothing and a
property whose strategy silently generated nothing look identical from the
outside: both print ``1 passed``. This project has shipped that shape
repeatedly. So every property in this suite carries a :class:`Census`, and
every property **asserts a floor on what it examined inside the same test**.
Not in a comment, not in a separate test that could be deselected or reordered
away — in the body, after the search, so that the test cannot pass without
having looked at something.

The floors are deliberately loose (a small fraction of the budget). They are
not a claim that the search was thorough; they are a tripwire for the search
having collapsed — a strategy that starts rejecting everything, a refusal list
that starts swallowing every example, a ``check()`` signature change that
turns every call into a ``TypeError``.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

# Exceptions that mean "the tool, or jax, declined this input" rather than
# "the tool is wrong". A refusal is always safe: it withholds a verdict.
#
# The list is broad on purpose, and the broadness is paid for by the census:
# `Census.require(examined=...)` fails if refusals swallow the run, so an
# over-broad entry cannot quietly turn a property into a no-op.
REFUSALS = frozenset(
    {
        "ValueError",
        "TypeError",
        "NotImplementedError",
        "OverflowError",
        "ZeroDivisionError",
        "XlaRuntimeError",
        "JaxRuntimeError",
        "UnexpectedTracerError",
        "ConcretizationTypeError",
        "KeyError",
        "AttributeError",
        "IndexError",
        "UnsatisfiableAssumptionError",
        "VacuityError",
        "EmptyDeclarationError",
        "OptionalDependencyError",
    }
)

DEFINITE = frozenset({"VERIFIED", "REFUTED"})


@dataclass(frozen=True)
class Outcome:
    status: str                 # VERIFIED | REFUTED | UNKNOWN | DECLINED
    obligations: tuple          # per-obligation status strings, in order


def run(spec, *, vacuity_mode="inputs-only", census=None, **kw):
    """Run ``spec`` through ``check``; ``None`` if the tool refused the input."""
    from stelling.preconditions import check

    from _grammar import build

    try:
        v = check(build(spec), vacuity_mode=vacuity_mode, **kw)
    except Exception as exc:  # noqa: BLE001 — the type is the classification
        name = type(exc).__name__
        if name in REFUSALS:
            if census is not None:
                census.refused(name)
            return None
        raise
    out = Outcome(v.status, tuple(o.status for o in v.obligations))
    if census is not None:
        census.verdict(out.status)
    return out


def contradiction(a: Outcome, b: Outcome) -> bool:
    """One leg definite and the other definite the other way."""
    return (
        a.status in DEFINITE
        and b.status in DEFINITE
        and a.status != b.status
    )


def obligation_contradiction(a: Outcome, b: Outcome):
    """The first index where one leg discharged and the other refuted it.

    Compared only when the two legs have the same number of obligations; a
    mutation that changes the obligation count has no per-index meaning.
    """
    if len(a.obligations) != len(b.obligations):
        return None
    bad = {"discharged", "violated-over-set"}
    for i, (oa, ob) in enumerate(zip(a.obligations, b.obligations)):
        if oa in bad and ob in bad and oa != ob:
            return i, oa, ob
    return None


@dataclass
class Census:
    """What a property actually examined. Asserted, not printed."""

    label: str
    drawn: int = 0
    skipped: collections.Counter = field(default_factory=collections.Counter)
    refusals: collections.Counter = field(default_factory=collections.Counter)
    verdicts: collections.Counter = field(default_factory=collections.Counter)
    compared: int = 0
    tags: collections.Counter = field(default_factory=collections.Counter)

    def draw(self) -> None:
        self.drawn += 1

    def skip(self, why: str) -> None:
        self.skipped[why] += 1

    def refused(self, exc_name: str) -> None:
        self.refusals[exc_name] += 1

    def verdict(self, status: str) -> None:
        self.verdicts[status] += 1

    def compare(self) -> None:
        """One completed metamorphic comparison: both legs produced a verdict."""
        self.compared += 1

    def tag(self, name: str) -> None:
        self.tags[name] += 1

    @property
    def examined(self) -> int:
        return sum(self.verdicts.values())

    def report(self) -> str:
        return (
            f"{self.label}: drawn={self.drawn} examined={self.examined} "
            f"compared={self.compared} verdicts={dict(self.verdicts)} "
            f"refusals={dict(self.refusals)} skipped={dict(self.skipped)} "
            f"tags={dict(self.tags)}"
        )

    def require(self, **floors: int) -> None:
        """Assert the floors. Called INSIDE the property, after the search.

        ``examined``/``compared``/``drawn`` are the census's own counters;
        anything else names a key of :attr:`tags`.
        """
        short = []
        for name, floor in sorted(floors.items()):
            got = (
                getattr(self, name)
                if name in ("drawn", "examined", "compared")
                else self.tags[name]
            )
            if got < floor:
                short.append(f"{name}={got} < {floor}")
        assert not short, (
            "GENERATOR FLOOR NOT MET — this property passed without examining "
            "what it claims to examine, which is indistinguishable from "
            "finding nothing:\n  "
            + "; ".join(short)
            + "\n  "
            + self.report()
        )
