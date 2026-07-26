# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Verdicts and their stamps, per SOUNDNESS.md's contract.

A verdict is ``VERIFIED``, ``REFUTED``, ``UNKNOWN`` — or ``DECLINED``,
which is not a fourth judgment but the absence of one: the query could not
be transcribed, so nothing was analysed and nothing is stamped. See
:class:`Verdict`. REFUTED
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

from stelling.interval import IEEE_ENDPOINT_ASSUMPTION
from stelling.propagate import ObligationReport, Propagation

__all__ = [
    "declined",
    "SEMANTICS_IEEE",
    "SEMANTICS_REAL",
    "SolverStamp",
    "Stamp",
    "StampError",
    "Verdict",
    "Witness",
    "make_verdict",
    "solver_absent",
]

ARITHMETIC_MODE_INTERVAL = "interval/f64/outward-1ulp (stelling.interval)"

# ieee-mode representation: endpoints are native binary64 results (no
# outward rounding — the float value itself is bracketed exactly; libm ops
# keep their 1-ulp outward brackets) plus a per-array maybe-NaN flag.
ARITHMETIC_MODE_INTERVAL_IEEE = (
    "interval/f64/native-endpoints+maybe-nan (stelling.interval)"
)

# The representation names how brackets are computed; the SEMANTICS names
# which arithmetic the verdict is *about*. They are different fields because
# the gap between them is where false-VERIFIED lives: `t + dt > t` is
# trivially true in ℝ and was a 258-day float bug
# (diffrax#632 — the census exhibit corpus/supply/exhibit_632.py). Today's
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

# The dial's second position. An ieee verdict claims facts about the float
# execution itself: overflow saturates to the value ±inf, NaN-producing
# corners are tracked, and 0·∞ = 0 does NOT ride (the
# REAL_CONVENTION_ASSUMPTION line must never appear in an ieee stamp).
SEMANTICS_IEEE = (
    "ieee (IEEE-754 binary64): obligations judged about the traced "
    "program's IEEE binary64 round-to-nearest float execution over the "
    "declared sets; the exact-real (ℝ) value is NOT what is claimed — a "
    "predicate can hold in floats and fail in ℝ, and vice versa"
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
class Witness:
    """A concrete violating input, independently confirmed by replay.

    Constructed only by the escalation layer, and only after the exact
    rational replay confirmed the violation — a witness that failed replay
    is an emission-infidelity bug and raises there instead of ever
    becoming one of these.

    **WHAT "confirmed by independent replay" DOES AND DOES NOT MEAN.**
    Replay is independent of the SOLVER, not of the PLAN. For primitives
    whose emission is driven by a shared routing plan, replay drives the
    SAME plan — so a defect in the plan is re-derived identically and the
    witness is stamped "confirmed". Measured, not hypothetical: an
    adversarial audit produced a witness on a trivially true property, and
    replay confirmed it, because both faces asked the same wrong question.

    So this field certifies "the solver did not fabricate this" and NOT
    "the translation of the program is faithful". The gap is exactly the
    class of defect a plan-level audit exists to find.

    The one independent leg is **executing the witness through the real
    program**: it shares no code with either the emission or the replay,
    so it is the only check that catches a plan defect. It carries more
    weight than its position in a three-way confirmation suggests.
    """

    obligation_index: int
    values: tuple[tuple[str, str], ...]  # (input name, exact rational)
    produced_by: str  # solver name/version/transport that returned the model
    replay: str  # how the confirmation was performed
    # for an ARRAY-shaped assert operand: the flat C-order indices of the
    # element predicates that are false at this point, per the same replay
    # that confirmed the violation. Empty for the scalar form, whose
    # rendering is unchanged — the violated predicate is the scalar itself.
    violating_elements: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.values:
            raise StampError("Witness.values must be populated")
        if not self.produced_by or not self.replay:
            raise StampError("Witness provenance fields must be populated")


def _render_one_solver(s: SolverStamp) -> str:
    return (
        f"{s.name} {s.version} ({s.transport}) options={dict(s.options or ())}"
        if s.invoked
        else f"none — {s.reason}"
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
    solver: SolverStamp | tuple[SolverStamp, ...]  # every invocation relied on
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
        # A portfolio verdict relied on several invocations: the solver field
        # then carries every one of them. The no-defaults/no-empty discipline
        # extends to the tuple — non-empty, each element a validated
        # SolverStamp with invoked=True (absence is only ever the bare
        # SolverStamp(invoked=False): an absence element rendered among the
        # invocations would misstate what happened).
        if isinstance(self.solver, tuple):
            if not self.solver:
                raise StampError(
                    "stamp field 'solver' is an empty tuple; absence must be "
                    "recorded as a SolverStamp(invoked=False), not as nothing"
                )
            for s in self.solver:
                if not isinstance(s, SolverStamp):
                    raise StampError(
                        f"stamp field 'solver' tuple contains "
                        f"{type(s).__name__}, expected SolverStamp"
                    )
                if not s.invoked:
                    raise StampError(
                        "stamp field 'solver' tuple contains a "
                        "SolverStamp(invoked=False); a non-invocation must "
                        "never be recorded among the invocations — absence "
                        "is a single bare SolverStamp"
                    )
        elif not isinstance(self.solver, SolverStamp):
            raise StampError(
                f"stamp field 'solver' is {type(self.solver).__name__}, "
                f"expected SolverStamp or tuple of SolverStamp"
            )

    def render(self) -> str:
        if isinstance(self.solver, tuple):
            solver = f"{len(self.solver)} invocation(s):" + "".join(
                f"\n  [{i}] {_render_one_solver(s)} — {s.reason}"
                for i, s in enumerate(self.solver)
            )
        else:
            solver = _render_one_solver(self.solver)
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


_STATUSES = frozenset({"VERIFIED", "REFUTED", "UNKNOWN", "DECLINED"})


def declined(reason: str) -> "Verdict":
    """The verdict for a query stelling could not transcribe.

    ``reason`` is quoted verbatim into the notes: a caller who points
    stelling at unsupported code learns what stelling could not read,
    which is the difference between "this tool is broken" and "this tool
    does not cover this yet".
    """
    return Verdict(
        status="DECLINED", obligations=(), stamp=None,
        notes=(f"declined: {reason}",), witnesses=(),
    )


@dataclass(frozen=True)
class Verdict:
    """A judged query, or a query that could not be read at all.

    ``DECLINED`` is not a fourth judgment — it is the ABSENCE of one. The
    query was never transcribed, so there is nothing to judge and nothing
    to stamp: a stamp identifies a query by ``query_content_hash``, and a
    query that could not be read has no hash. Minting a sentinel one would
    produce a stamp that LOOKS like an attestation of something, which is
    the failure this project exists to prevent, so ``stamp`` is ``None``
    exactly when the status is ``DECLINED`` and never otherwise.

    It is a status rather than an exception so that batch callers compose:
    a CI soak or a graph-wide pass over many nodes must be able to report
    "2 verified, 0 unknown, 3 declined" and keep going. An exception makes
    the first unsupported node kill the run.
    """

    status: str  # "VERIFIED" | "REFUTED" | "UNKNOWN" | "DECLINED"
    obligations: tuple[ObligationReport, ...]
    stamp: Stamp | None  # None IFF status == "DECLINED"
    notes: tuple[str, ...]  # the addresses: where and why anything degraded
    witnesses: tuple[Witness, ...] = ()  # replay-confirmed counterexamples

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise StampError(
                f"verdict status must be one of {sorted(_STATUSES)}, got "
                f"{self.status!r}"
            )
        if (self.stamp is None) != (self.status == "DECLINED"):
            raise StampError(
                "a DECLINED verdict carries no stamp and every other status "
                "must carry one: a stamp identifies a query by content hash, "
                "and a declined query was never read, so there is no hash to "
                f"identify it by (status={self.status!r}, "
                f"stamp={'None' if self.stamp is None else 'present'})"
            )
        if self.status == "DECLINED" and (self.obligations or self.witnesses):
            raise StampError(
                "a DECLINED verdict judged nothing, so it can carry neither "
                "obligations nor witnesses"
            )

    def render(self) -> str:
        lines = [f"== {self.status}"]
        if self.status == "DECLINED":
            lines.append(
                "  the query was not analysed — stelling could not read it. "
                "No claim is made about the program, in either direction."
            )
            lines.extend(f"  {n}" for n in self.notes)
            lines.append(
                "  no stamp: a stamp identifies a query by content hash, and "
                "this query was never transcribed."
            )
            return "\n".join(lines)
        # EQUATION COUNT, SECOND LINE. A wrong harness build or step boundary
        # does not produce a wrong answer — it silently changes WHICH PROGRAM
        # was verified, and the equation count is the only tell. Measured
        # instance: a projector built inside the harness traced 349 equations
        # instead of 5, yielding a wrong CAUSE rather than a wrong verdict.
        # The stamp has always carried this inside its coverage line, but at
        # the bottom, under transfers and provenance — not where an author
        # looks. Same reasoning as the w6 soak's `eqns` column.
        eqns = self.stamp.coverage.split(" eqns")[0]
        if eqns.isdigit():
            lines.append(f"  {eqns} equations verified")
        # THE OPT-IN, NAMED WHERE THE UNDECIDED RESULT IS READ.
        # `check()` does not escalate unless solver_timeout_ms is passed, and
        # its default is None -- a hard early return before the solver module
        # is even imported. So the documented front door produces
        # interval-only results, and a caller who does not know the opt-in
        # exists reads UNKNOWN as "stelling looked and could not decide".
        # The last sentence is the load-bearing one: without it the default
        # output invites exactly the misattribution the escalation-status
        # audit was built to detect, relocated from the record into the
        # reader's head. Surfaced only when something is actually undecided
        # -- on an interval-discharged VERIFIED the hint is noise.
        if self.status == "UNKNOWN" and "NOT ATTEMPTED" in (
            self.stamp.solver.reason
            if not isinstance(self.stamp.solver, tuple)
            else ""
        ):
            lines.append(
                "  interval-only: no SMT solver has seen this query. "
                "Undecided obligations here may still be decidable — pass "
                "solver_timeout_ms=<ms> to check(). This is NOT a finding "
                "that the property is undecidable."
            )
        if self.status == "REFUTED":
            # Rendering honesty: the set-level wording ("not a witness")
            # belongs ONLY to interval refutations. A witness-backed REFUTED
            # renders the witness instead — a concrete counterexample, a
            # strictly stronger refutation than the set-level one.
            if any(o.status == "violated-over-set" for o in self.obligations):
                # Wording honesty, second axis (audit F4): under a
                # constraining assume the judgment ran over the
                # precondition-narrowed set, and "the stated box is not
                # invariant as stated" is a claim about the declared box
                # this verdict did not check. Render-only derivation: the
                # stamped constrained-assume assumption line is the
                # semantic carrier of conditionality.
                if any(
                    "the verdict holds where the precondition holds" in a
                    for a in self.stamp.assumptions
                ):
                    lines.append(
                        "  (set-level, conditional: at least one obligation is "
                        "definitely false where the assumed precondition holds "
                        "— judged over the propagated superset of the "
                        "precondition-narrowed set, not over the full declared "
                        "box. Not a witness; not a counterexample to the "
                        "program.)"
                    )
                else:
                    lines.append(
                        "  (set-level: at least one obligation is definitely false over "
                        "the declared set — the stated box is not invariant as stated. "
                        "Not a witness; not a counterexample to the program.)"
                    )
            for w in self.witnesses:
                lines.append(
                    f"  witness for assert #{w.obligation_index} — a concrete "
                    f"violating input (strictly stronger than a set-level "
                    f"refutation):"
                )
                for name, value in w.values:
                    approx = _approx(value)
                    lines.append(f"    {name} = {value}{approx}")
                if w.violating_elements:
                    # array-scale honesty: the assert was a universal
                    # elementwise claim; name exactly which element(s) this
                    # point falsifies (flat C-order indices)
                    lines.append(
                        "    violating element(s) of the assert operand: "
                        + ", ".join(str(i) for i in w.violating_elements)
                    )
                lines.append(f"    produced by: {w.produced_by}")
                lines.append(f"    replay: {w.replay}")
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


def _approx(exact: str) -> str:
    """A human decimal approximation of an exact rational string, for the
    witness rendering only; the exact value is always shown first."""
    try:
        from fractions import Fraction

        fr = Fraction(exact)
    except (ValueError, ZeroDivisionError):
        return ""
    if fr.denominator == 1:
        return ""
    return f" (≈ {float(fr):.6g})"


# ── THE SCATTER VERIFIED BAR — ONE REMOVAL POINT ─────────────────────────────
#
# Obligations whose slice touches `scatter` are INELIGIBLE FOR VERIFIED until
# the emission row has been attacked by a distinct-context adversarial auditor.
#
# The reason is an asymmetry, not caution in general. A wrong scatter encoding
# that produces a SPURIOUS witness is caught downstream: every REFUTED witness
# is re-checked by independent exact-rational replay, and a witness that fails
# replay raises rather than becoming a verdict. A wrong encoding that MISSES a
# violation has no such downstream check — it mints a false VERIFIED, which is
# this project's own thesis defect.
#
# So the row may be built and exercised on the refutation path, where it is
# self-checking, while the direction that cannot be self-checked stays closed.
# Under this bar the worst case of a wrong row is a witness that fails replay.
# Without it the worst case is silent.
#
# TO LIFT: delete this set's contents (leave the empty frozenset and this
# comment). That is the whole mechanism — one identifiable place, by design,
# so lifting it is a deliberate act and not a diffuse relaxation. It is the
# principal's to lift, after the auditor reports.
VERIFIED_BARRED_PRIMITIVES = frozenset({"scatter"})

VERIFIED_BAR_REASON = (
    "the traced query contains {prims}, whose SMT emission has not yet been "
    "attacked by a distinct-context adversarial auditor. A wrong encoding that "
    "produced a spurious witness would be caught by exact-rational replay; one "
    "that MISSED a violation would mint a false VERIFIED with nothing "
    "downstream to catch it, so the un-self-checking direction stays closed "
    "until the audit lands. This is a bar on the CLAIM, not a finding about "
    "the program: nothing here says the obligation is false."
)


def _barred_primitives(closed) -> tuple[str, ...]:
    """Barred primitives present anywhere in the traced query, innermost
    scopes included. Deliberately whole-query rather than slice-scoped: a
    bar that under-fires is worse than one that over-fires, and the slice is
    not available at this layer.

    DESCENT GOES THROUGH :func:`stelling.coverage.sub_jaxprs`, THE CANONICAL
    ACCESSOR, and must not be hand-rolled here. The predecessor descended via
    ``getattr(v, "jaxpr", None)``, which finds a param that IS a ClosedJaxpr
    but not one holding a COLLECTION of them — and ``cond`` stores its
    branches as a tuple. Measured: a `scatter` inside a `cond` branch was not
    barred at all, while the same primitive at top level, inside `jit`, and
    inside `scan` all barred correctly. That is the UNDER-firing direction,
    which this function's own docstring calls the worse one.

    The lesson is narrower than "descend properly": **do not hand-roll a
    traversal when a canonical accessor exists.** A second implementation of
    a walk is a second thing to keep correct, and this one silently stopped
    matching. `tests/test_bar_walk_parity.py` asserts the two agree
    mechanically, in the spirit of the EMISSION == REPLAY census.
    """
    from stelling.coverage import sub_jaxprs

    if not VERIFIED_BARRED_PRIMITIVES or closed is None:
        return ()
    # Uses coverage.sub_jaxprs, the SAME nesting walk the coverage tool uses,
    # rather than a private one. The earlier version did
    # `getattr(v, "jaxpr", None)`, which finds a ClosedJaxpr and misses a bare
    # `ir.Jaxpr`, a jaxpr held inside a tuple, and a `NamedTupleParam` field —
    # so a transparent primitive storing its body in any of those forms would
    # have made the bar UNDER-fire. No such primitive exists on jax 0.11.0, so
    # this was latent; but a bar with weaker reachability than the coverage
    # tool is a hole in the one guard the unshipped rows rest on, and "latent"
    # is not a reason to keep two walks.
    from stelling.coverage import sub_jaxprs

    found, seen = set(), set()

    def walk(jaxpr):
        if jaxpr is None or id(jaxpr) in seen:
            return
        seen.add(id(jaxpr))
        for eqn in getattr(jaxpr, "eqns", ()):
            name = str(eqn.primitive)
            if name in VERIFIED_BARRED_PRIMITIVES:
                found.add(name)
            for inner in sub_jaxprs(eqn):
                if id(inner) not in seen:
                    seen.append(id(inner))
                    walk(inner)

    try:
        walk(getattr(closed, "jaxpr", None))
    except Exception:  # a bar must never be the thing that breaks a verdict
        return tuple(sorted(VERIFIED_BARRED_PRIMITIVES))
    return tuple(sorted(found))


def make_verdict(
    closed,
    propagation: Propagation,
    *,
    stelling_version: str,
    jax_version: str,
    precision_config: str,
    device_class: str = "none: no concrete execution in this verdict",
    refinement=None,
) -> Verdict:
    """Assemble the verdict for an interval-propagated harness query.

    ``refinement`` (default None — byte-identical assembly) is the
    :class:`stelling.affine.RefinementReport` of an affine refinement
    that ran on this propagation: the stamp then records the refinement
    ran (an appended assumptions line) and the solver-absence reason
    names the layers that actually judged, via the report's own
    derivation — the absence line must not claim "interval arithmetic
    alone" when the refinement decided anything.
    """
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
    # the stamp says which arithmetic the verdict is ABOUT, from the
    # propagation that ran — never from a guess. The 0·∞ = 0 convention is
    # a consequence of ℝ semantics and must NOT ride in an ieee stamp; the
    # ieee stamp carries the native-binary64-endpoint assumption instead.
    if propagation.semantics == "ieee":
        semantics = SEMANTICS_IEEE
        arithmetic_mode = ARITHMETIC_MODE_INTERVAL_IEEE
        convention = IEEE_ENDPOINT_ASSUMPTION
        solver_reason = (
            "no solver invoked: every obligation was judged by native-"
            "binary64 interval arithmetic alone (ieee semantics refuses "
            "solver escalation: the SMT backends emit over the reals)"
        )
    else:
        semantics = SEMANTICS_REAL
        arithmetic_mode = ARITHMETIC_MODE_INTERVAL
        convention = REAL_CONVENTION_ASSUMPTION
        solver_reason = (
            "no solver invoked: escalation was NOT ATTEMPTED "
            "(solver_timeout_ms not set); every obligation was judged by "
            "outward-rounded interval arithmetic alone"
        )
    assumptions = tuple(sorted({*propagation.assumptions, convention}))
    if refinement is not None:
        # the refinement's participation is derived from its record at
        # the single absence-derivation point, never narrated separately;
        # the arithmetic line names the deciding abstraction when the
        # affine domain decided anything
        solver_reason = refinement.reword_absence(solver_reason)
        arithmetic_mode = refinement.reword_arithmetic(arithmetic_mode)
        assumptions = assumptions + (refinement.stamp_line(),)
    stamp = Stamp(
        stelling_version=stelling_version,
        jax_version=jax_version,
        query_content_hash=closed.content_hash(),
        arithmetic_mode=arithmetic_mode,
        semantics=semantics,
        precision_config=precision_config,
        device_class=device_class,
        solver=solver_absent(solver_reason),
        nonvacuity=nonvacuity,
        transfer_tiers=propagation.transfers_used,
        transfer_provenance=tuple((p, "core") for p, _ in propagation.transfers_used),
        assumptions=assumptions,
        coverage=propagation.coverage.summary(),
    )
    return Verdict(
        status=status,
        obligations=propagation.obligations,
        stamp=stamp,
        notes=notes,
    )
