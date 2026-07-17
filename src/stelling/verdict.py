# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Verdicts and their stamps, per SOUNDNESS.md's contract.

A verdict is ``VERIFIED``, ``REFUTED``, or ``UNKNOWN``. REFUTED
(`design/e2a-registration.md`, amendment 1) is a **set-level refutation
of the stated box**: at least one obligation is definitely false over
the propagated superset of the declared set, so the box is not invariant
*as stated*. It is not a witness — no concrete input is produced — and
not a counterexample to the program. Straddles stay UNKNOWN: *our*
imprecision, never their counterexample.

The stamp is the contract from SOUNDNESS.md ("What every verdict must
carry"), enforced structurally:

* every field is required — no defaults, so a missing field is a
  ``TypeError`` at construction, not a silently defaulted value;
* empty strings are rejected (:class:`StampError`) — "populated with
  nothing" is the same defect as missing;
* **solver absence is recorded, never implied**: :class:`SolverStamp` with
  ``invoked=False`` must carry the reason (e.g. "all obligations
  discharged by interval arithmetic; no solver invoked"), and a
  ``SolverStamp`` with ``invoked=True`` must carry name, version,
  transport, and the exact emitted option set — the never-on-defaults
  commitment stays banked until the first solver call and its can't-drift
  test lands with it.

This module is jax-free; versions are passed in by the caller (the
harness runner knows its jax; :mod:`stelling` knows its own version).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from stelling.propagate import ObligationReport, Propagation

__all__ = ["SolverStamp", "Stamp", "StampError", "Verdict", "make_verdict"]

ARITHMETIC_MODE_INTERVAL = "interval/f64/outward-1ulp (stelling.interval)"

# The representation names how brackets are computed; the SEMANTICS names
# which arithmetic the verdict is *about*. They are different fields because
# the gap between them is where false-VERIFIED lives: `t + dt > t` is
# trivially true in ℝ and was a 258-day float bug (diffrax#632). Today's
# only value — the founding doc's ℝ-with-margin/IEEE-exact dial, position
# recorded per verdict:
SEMANTICS_REAL = (
    "real (ℝ): obligations judged in exact real arithmetic over the declared "
    "sets; the traced program's IEEE float behaviour is NOT modeled — a "
    "predicate can hold in ℝ and fail in floats"
)

# A consequence of ℝ semantics, carried as an assumption because it is one:
# under IEEE, inf is a value and 0*inf is NaN; the closed-real-interval
# convention 0·∞ = 0 is sound only because half-infinite intervals here mean
# "unbounded above but finite", which is the ℝ reading.
REAL_CONVENTION_ASSUMPTION = (
    "closed-real-interval endpoint convention 0*inf = 0 — a consequence of "
    "'real' semantics; unsound under IEEE semantics, where inf is a value"
)


class StampError(ValueError):
    """A stamp field is present but unpopulated — forbidden by SOUNDNESS.md."""


@dataclass(frozen=True)
class SolverStamp:
    invoked: bool
    reason: str  # why not invoked / what role it played
    name: str | None
    version: str | None
    transport: str | None
    options: tuple[tuple[str, str], ...] | None  # the exact emitted set

    def __post_init__(self) -> None:
        if not self.reason:
            raise StampError("SolverStamp.reason must be populated")
        if self.invoked:
            missing = [
                f
                for f in ("name", "version", "transport", "options")
                if getattr(self, f) in (None, "", ())
            ]
            if missing:
                raise StampError(
                    f"solver invoked but stamp fields unpopulated: {missing} — "
                    f"never invoke a solver on defaults (SOUNDNESS.md)"
                )
        else:
            extra = [
                f
                for f in ("name", "version", "transport", "options")
                if getattr(self, f) is not None
            ]
            if extra:
                raise StampError(
                    f"solver not invoked but stamp carries {extra}; absence must "
                    f"be recorded as absence"
                )


def solver_absent(reason: str) -> SolverStamp:
    return SolverStamp(
        invoked=False, reason=reason, name=None, version=None, transport=None, options=None
    )


@dataclass(frozen=True)
class Stamp:
    stelling_version: str
    jax_version: str  # the jax that traced the harness
    query_content_hash: str  # ir.ClosedJaxpr.content_hash()
    arithmetic_mode: str  # endpoint representation (how brackets are computed)
    semantics: str  # which arithmetic the verdict is about (ℝ vs IEEE)
    precision_config: str  # e.g. "jax_enable_x64=True"
    device_class: str  # of any concrete execution the verdict relies on
    solver: SolverStamp
    nonvacuity: str  # is the declared set tied to the incident's own data?
    transfer_tiers: tuple[tuple[str, str], ...]  # (primitive, tier) actually used
    transfer_provenance: tuple[tuple[str, str], ...]  # (primitive, origin)
    assumptions: tuple[str, ...]
    coverage: str  # the ⊤-coverage summary line

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, str) and not v:
                raise StampError(f"stamp field {f.name!r} is empty; every field "
                                 f"in the contract must be populated")

    def render(self) -> str:
        solver = (
            f"{self.solver.name} {self.solver.version} ({self.solver.transport}) "
            f"options={dict(self.solver.options or ())}"
            if self.solver.invoked
            else f"none — {self.solver.reason}"
        )
        lines = [
            f"stelling {self.stelling_version} | jax {self.jax_version}",
            f"query {self.query_content_hash}",
            f"arithmetic: {self.arithmetic_mode}",
            f"semantics: {self.semantics}",
            f"precision: {self.precision_config} | device: {self.device_class}",
            f"solver: {solver}",
            f"nonvacuity: {self.nonvacuity}",
            "transfers: "
            + ", ".join(f"{p} [{t}]" for p, t in self.transfer_tiers),
            "provenance: "
            + ", ".join(f"{p}:{o}" for p, o in self.transfer_provenance),
        ]
        for a in self.assumptions:
            lines.append(f"assumes: {a}")
        lines.append(f"coverage: {self.coverage}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Verdict:
    status: str  # "VERIFIED" | "REFUTED" | "UNKNOWN"
    obligations: tuple[ObligationReport, ...]
    stamp: Stamp
    notes: tuple[str, ...]  # the addresses: where and why anything degraded

    def render(self) -> str:
        lines = [f"== {self.status}"]
        if self.status == "REFUTED":
            lines.append(
                "  (set-level: at least one obligation is definitely false over "
                "the declared set — the stated box is not invariant as stated. "
                "Not a witness; not a counterexample to the program.)"
            )
        for o in self.obligations:
            lines.append(f"  assert #{o.index}: {o.status} — {o.detail}")
            # location re-derived from the *current* query's source_info,
            # never from any cache (SOUNDNESS.md: cache the proof, not the
            # report).
            if o.source_info:
                lines.append(f"    at {o.source_info[-1]}")
        lines.append(self.stamp.render())
        for n in self.notes:  # under the coverage line: the actionable half
            lines.append(f"note: {n}")
        return "\n".join(lines)


def make_verdict(
    closed,
    propagation: Propagation,
    *,
    stelling_version: str,
    jax_version: str,
    precision_config: str,
    device_class: str = "none: no concrete execution in this verdict",
) -> Verdict:
    """Assemble the verdict for an interval-propagated harness query."""
    if propagation.any_violated:
        status = "REFUTED"
    elif propagation.all_discharged:
        status = "VERIFIED"
    else:
        status = "UNKNOWN"

    checks = propagation.nonvacuity_checks
    if not checks:
        nonvacuity = "UNCHECKED — no membership conditions declared"
    elif all(c.status == "discharged" for c in checks):
        nonvacuity = (
            f"checked — {len(checks)} membership condition(s) definitely true "
            f"(the declared set contains the stated point)"
        )
    elif any(c.status == "violated-over-set" for c in checks):
        nonvacuity = (
            "FAILED — a membership condition is definitely false: the stated "
            "point is NOT in the declared set (harness defect, not a box fact)"
        )
    else:
        nonvacuity = "undecided — a membership condition could not be decided"

    notes = propagation.notes
    if status == "VERIFIED" and not nonvacuity.startswith("checked"):
        notes = notes + (
            f"nonvacuity {nonvacuity.split(' — ')[0]}: this VERIFIED may be "
            f"vacuous — the declared set is not tied to the incident's data",
        )
    stamp = Stamp(
        stelling_version=stelling_version,
        jax_version=jax_version,
        query_content_hash=closed.content_hash(),
        arithmetic_mode=ARITHMETIC_MODE_INTERVAL,
        semantics=SEMANTICS_REAL,
        precision_config=precision_config,
        device_class=device_class,
        solver=solver_absent(
            "no solver invoked: every obligation was judged by outward-rounded "
            "interval arithmetic alone"
        ),
        nonvacuity=nonvacuity,
        transfer_tiers=propagation.transfers_used,
        transfer_provenance=tuple((p, "core") for p, _ in propagation.transfers_used),
        assumptions=tuple(
            sorted({*propagation.assumptions, REAL_CONVENTION_ASSUMPTION})
        ),
        coverage=propagation.coverage.summary(),
    )
    return Verdict(
        status=status,
        obligations=propagation.obligations,
        stamp=stamp,
        notes=notes,
    )
