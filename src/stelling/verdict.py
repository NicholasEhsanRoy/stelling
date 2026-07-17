# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Verdicts and their stamps, per SOUNDNESS.md's contract.

A verdict is ``VERIFIED`` or ``UNKNOWN`` — nothing else. This checker
never refutes: a definitely-false obligation is reported inside an
UNKNOWN verdict (see :mod:`stelling.propagate`).

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
    arithmetic_mode: str
    precision_config: str  # e.g. "jax_enable_x64=True"
    device_class: str  # of any concrete execution the verdict relies on
    solver: SolverStamp
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
            f"precision: {self.precision_config} | device: {self.device_class}",
            f"solver: {solver}",
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
    status: str  # "VERIFIED" | "UNKNOWN"
    obligations: tuple[ObligationReport, ...]
    stamp: Stamp

    def render(self) -> str:
        lines = [f"== {self.status}"]
        for o in self.obligations:
            lines.append(f"  assert #{o.index}: {o.status} — {o.detail}")
            # location re-derived from the *current* query's source_info,
            # never from any cache (SOUNDNESS.md: cache the proof, not the
            # report).
            if o.source_info:
                lines.append(f"    at {o.source_info[-1]}")
        lines.append(self.stamp.render())
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
    status = "VERIFIED" if propagation.all_discharged else "UNKNOWN"
    stamp = Stamp(
        stelling_version=stelling_version,
        jax_version=jax_version,
        query_content_hash=closed.content_hash(),
        arithmetic_mode=ARITHMETIC_MODE_INTERVAL,
        precision_config=precision_config,
        device_class=device_class,
        solver=solver_absent(
            "no solver invoked: every obligation was judged by outward-rounded "
            "interval arithmetic alone"
        ),
        transfer_tiers=propagation.transfers_used,
        transfer_provenance=tuple((p, "core") for p, _ in propagation.transfers_used),
        assumptions=propagation.assumptions,
        coverage=propagation.coverage.summary(),
    )
    return Verdict(status=status, obligations=propagation.obligations, stamp=stamp)
