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
from stelling.reachability import defined_vars, reaches_output

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
    "top_despite_coverage_note",
]

ARITHMETIC_MODE_INTERVAL = "interval/f64/outward-1ulp (stelling.interval)"

# ieee-mode representation: endpoints are native binary64 results (no
# outward rounding — the float value itself is bracketed exactly; libm ops
# keep their 1-ulp outward brackets) plus a per-array maybe-NaN flag.
ARITHMETIC_MODE_INTERVAL_IEEE = (
    "interval/f64/native-endpoints+maybe-nan (stelling.interval)"
)

# Format-parametric ieee arithmetic mode strings. For queries using only
# float64, the legacy string above is used byte-identically. For queries
# involving narrower formats, the format is recorded in the stamp.
ARITHMETIC_MODE_INTERVAL_IEEE_FMT = (
    "interval/ieee-parametric/native-f64-endpoints+format-rounding+maybe-nan "
    "(stelling.interval)"
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

SEMANTICS_IEEE_FMT = (
    "ieee (IEEE-754 parametric): obligations judged about the traced "
    "program's IEEE round-to-nearest float execution in the declared "
    "format (float32/float16/bfloat16/float64) over the declared sets; "
    "interval endpoints computed in native float64 then rounded outward "
    "to the target format's ULP grid"
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
    # WHAT THE COVERAGE LINE ABOVE DID NOT ESTABLISH, or None when this
    # particular reading was not available to make.
    #
    # The coverage line is a CENSUS: it counts whether each primitive in
    # the query has a registered transfer. `unknown = 0` therefore means
    # "every primitive is registered" and reads as "the analysis saw
    # everything" — which it is not. A registered transfer can run and
    # return ⊤ on the values it is handed, and no count on that line
    # moves when it does. This field is populated exactly when the census
    # recorded no gap AND the propagation still produced a ⊤ box, and it
    # says only that: what the census did not rule out. It never asserts
    # that coverage is complete, and its ABSENCE asserts nothing at all —
    # absence means this reading did not fire, not that the query was
    # bounded (a ⊤ inside a discarded branch scope, for one, is not
    # visible to it).
    #
    # A separate field, not an addition to the coverage string, because
    # that string is trend data: `stamp.coverage.split(" eqns")[0]` is
    # parsed in reproduce.py, in Verdict.render, and across the sweep
    # scripts. Additive here costs those parsers nothing.
    #
    # REQUIRED like every other field, and deliberately NOT defaulted to
    # None even though None is a legitimate value. This landed with a
    # default and was the only defaulted field of the fourteen, which
    # made the module docstring's "no defaults" false and let an assembly
    # site omit it and publish a stamp that silently reads "this reading
    # did not fire" — the same defect as the implied solver absence two
    # doors up. Absence is recorded, never implied: a site with nothing
    # to disclose passes None and says so.
    top_despite_coverage: str | None

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
        # rendered immediately under the line it qualifies: a caveat a
        # reader reaches after the number it is about has already been
        # read is a caveat that arrives too late
        if self.top_despite_coverage:
            lines.append(f"coverage-not-established: {self.top_despite_coverage}")
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
    # THE REDUNDANCY THIS VERDICT ACTUALLY GOT, per solver-decided
    # obligation: ``(assert index, labels of the backends that ANSWERED)``.
    # Empty on an interval-only verdict, which no solver decided.
    #
    # A COUNTING SURFACE, and it exists because counting was the failure.
    # The stamp's solver tuple records who was ASKED — that is its contract
    # — so a two-backend stamp is compatible with a one-backend decision,
    # and a job tallying VERIFIEDs could not tell them apart. A tally that
    # cares about the cross-check reads this:
    #
    #     one_backend = [i for i, who in v.solver_redundancy if len(who) < 2]
    #
    # It is not a soundness gate and does not change any verdict: a
    # one-backend VERIFIED is still a VERIFIED, it just got half the
    # redundancy the portfolio is designed around, and this says so in a
    # form that survives being counted rather than read.
    solver_redundancy: tuple[tuple[int, tuple[str, ...]], ...] = ()

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
        # THE PORTFOLIO'S REDUNDANCY, THIRD LINE, FOR THE SAME REASON THE
        # EQUATION COUNT IS SECOND. A verdict decided by one backend under a
        # two-backend stamp does not read differently from a verdict decided
        # by both — the stamp records who was asked, the notes carry the
        # failure, and neither is where a reader looks. On a VERIFIED this
        # matters most: an `unsat` is a universal claim with no witness to
        # replay, so the second backend is the only independent check there
        # is. Surfaced only when the redundancy is actually short; a full
        # portfolio needs no line.
        short = [
            (i, who) for i, who in self.solver_redundancy if len(who) < 2
        ]
        for i, who in short[:3]:
            lines.append(
                f"  PORTFOLIO DEGRADED — assert #{i} was decided by ONE "
                f"solver backend ({', '.join(who) or 'none'}), not the two "
                f"the portfolio is designed around; the notes say which "
                f"backend was lost and why"
            )
        if len(short) > 3:
            lines.append(
                f"  PORTFOLIO DEGRADED — and {len(short) - 3} further "
                f"obligation(s): {', '.join(f'assert #{i}' for i, _ in short[3:])}"
            )
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
# Read "whose slice" literally: the scope is the emitted slice of the
# obligation the solver actually decided, derived by :func:`_bar_scope` from
# the traced query and the decided obligation's INDEX. It is NOT the whole
# traced query — a scatter elsewhere in the jaxpr, on an obligation intervals
# settled, withholds nothing. (Whole-query IS the fail-closed fallback,
# `_barred_primitives`, whenever that derivation cannot be completed.)
#
# THE SCOPE'S CONTENTS ARE DERIVED FROM `closed`, NEVER READ OFF THE
# ESCALATION, AND THAT IS THE SOUNDNESS PROPERTY. (Its DOMAIN — which
# obligations are asked about — does come from the escalation; that is
# `make_solver_verdict`'s stated precondition, spelled out in its docstring
# and in :func:`_bar_scope`, and it is not claimed as an immunity here.)
# The predecessor recorded the per-obligation
# barred set on `solvers.ObligationEscalation.barred_on_slice` at emission
# time and trusted it at the bar. Measured, both directions:
#
#   * `barred_on_slice=()` is a POSITIVE claim ("nothing barred on my slice")
#     and nothing validated it — a record carrying it earned VERIFIED on a
#     query whose genuine scope was `('scatter',)`;
#   * `make_solver_verdict` is public and gates mispairing on semantics, ieee,
#     constrained-assume and ledger provenance, but NOTHING binds the
#     escalation to `closed` — a scatter-free escalation stamped against a
#     scatter-bearing query returned VERIFIED where the whole-query bar it
#     replaced returned UNKNOWN.
#
# Both are closed by deriving instead of reading: `_bar_scope` re-slices the
# decided obligations out of `closed` itself, through the SAME
# `slice_obligation` call `escalate` sliced them with, so there is no field
# to forge.
#
# DERIVING FIXED THE CONTENTS AND NOT THE PAIRING, AND THE FIRST REPAIR SAID
# OTHERWISE. It claimed a mispaired `closed` "bars exactly as the whole-query
# bar did", on the reasoning that a mispaired index would not slice. Measured
# false, on two scatter-bearing queries of the SAME SHAPE — one with the
# scatter on the solver-decided obligation, one with it on an interval-decided
# one: the first's escalation stamped against the second re-sliced cleanly to
# `['ge','sub']`, found nothing, and returned VERIFIED where `8e42934`
# returned UNKNOWN. An index is not evidence about a query. So the narrowing
# is now earned per obligation by `_evidence_is_about`: the recorded
# invocation's `smt2_sha256` AND its `slice_sha256` must both re-emit from the
# slice re-derived out of THIS `closed`.
#
# THE SECOND HASH IS NOT BELT-AND-BRACES; THE FIRST ONE ALONE WAS MEASURED
# INSUFFICIENT. `eb1ff86` narrowed on the script hash alone, and a script hash
# cannot say which slice produced it: the static-index `scatter` SET row emits
# NO line (`smt.emit`, the `prim == "scatter"` branch), so an untouched element
# aliases the operand's term and `s[1] - x[1] <= 0` emits byte for byte what
# `x[1] - x[1] <= 0` emits — same sha, barred sets `('scatter',)` and `()`.
# On that pair `eb1ff86` returned VERIFIED where `8e42934` returns UNKNOWN.
# `smt.slice_fingerprint` is the quantity the text cannot carry, and it is
# derived from the slice's primitives and nesting rather than from its output.
#
# An earlier version of this paragraph ended "and every other outcome —
# including every stray index — widens to the whole query". The second clause
# was the false one: a stray index whose slice happens to emit the recorded
# script NARROWED, which is the hole above. What is true, and all that is
# claimed now, is that every outcome other than BOTH hashes matching widens.
# The two inputs stay anti-correlated — WHICH obligations the solver decided
# comes from the escalation (and is already load-bearing for VERIFIED itself:
# an index that does not match an unknown obligation leaves it undischarged
# and there is no VERIFIED to withhold), WHAT is on their slices comes from
# the query. The anti-correlation is only as good as ONE CONCEPT ON THE
# ESCALATION SIDE: "the solver decided it" and "the record discharges it" must
# be the same test, or the load-bearingness above is about a different set
# than the one the bar takes. They drifted once — `make_solver_verdict` read
# `outcome == OB_DISCHARGED and r.invocations` for the bar while discharging
# on `outcome == OB_DISCHARGED` alone, and a record stripped of `invocations`
# discharged its obligation while leaving the bar's domain. There is now one
# predicate, in one place. Re-slicing is not a second implementation of the emitted slice:
# `slice_obligation(closed, index, interval_env(closed))` is verbatim what
# `slice_unknown_obligations` calls, whose only other argument
# (`top_primitives`) is documented "message wording only, never admission",
# and `tests/test_bar_walk_parity.py` pins the two against each other.
#
# THE MEMBERSHIP IS EXACT-NAME, SO `scatter-add` IS NOT UNDER THIS BAR, AND
# THAT IS DELIBERATE. `scatter-add` is a separate primitive with separate
# rows, and `design/scatter-rows.md` records a COMPLETED fresh-adversarial-
# auditor pass over them (2026-07-22, zero UNSOUND). The bar is for the rows
# that have not had that pass — the static-index `scatter` SET row — not for
# scatter as a family. A reader who assumed otherwise would over-read every
# withheld VERIFIED in the ledger.
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

# THE FAIL-CLOSED FALLBACK'S ONE REASON THAT IS NOT AN f-STRING, named here
# rather than spelled inside :func:`_bar_scope`, for the same reason
# :data:`_EVIDENCE_BUDGET_KEYS` is named rather than spelled inside
# :func:`_evidence_budget`: the DECISION may no longer carry a string constant
# in a position that could drive it. Its two other fallback reasons interpolate
# the obligation index and are f-strings already; this one interpolates
# nothing, so it would be a bare `Constant` handed to a call — which is exactly
# `M9X`'s shape (``str(closed).count("stelling_backdoor")``, live at `faefc48`
# and 0 RED at `9fc44dd`). A module constant is read by the closure walk and
# ledgered to one reader; a literal inside a call was read by nothing.
# `tests/test_verified_bar.py::test_the_evidence_path_cannot_name_a_VALUE`
_BAR_UNDERIVABLE = ("the decided obligations' emitted slices could not be "
                    "re-derived")

# `{scope}` NAMES EACH OBLIGATION WITH ITS OWN BARRED SET, and the split from
# `{prims}` is why. The predecessor rendered `{where}` from ALL deciding
# obligations while `{prims}` was the UNION over them, with nothing
# intersecting the two: with two solver-decided obligations it rendered "the
# emitted slice of assert #0, assert #1 contains scatter" while #1's slice was
# `['add', 'le', 'sub']` — a message claiming a scope its mechanism does not
# have, which is this repo's own recurring defect. `{scope}` is now built
# per-obligation by :func:`_bar_scope_phrase` and names only obligations that
# carry something; `{prims}` is the union, and appears only in a statement
# about the emission ROWS, which is true of them wherever they were reached.
VERIFIED_BAR_REASON = (
    "{scope}. The SMT emission of {prims} has not yet been "
    "attacked by a distinct-context adversarial auditor. A wrong encoding that "
    "produced a spurious witness would be caught by exact-rational replay; one "
    "that MISSED a violation would mint a false VERIFIED with nothing "
    "downstream to catch it, so the un-self-checking direction stays closed "
    "until the audit lands. This is a bar on the CLAIM, not a finding about "
    "the program: nothing here says the obligation is false."
)


def _barred_in_eqns(eqns) -> tuple[str, ...]:
    """THE BAR'S ONE WALK, rooted at an arbitrary iterable of equations.

    Two roots use it and there is only ever one traversal:
    :func:`_barred_primitives` roots it at the whole query's top-level
    equations, and :func:`_bar_scope` roots it at ONE obligation's emitted
    slice (``ObligationSlice.eqns``) to derive that obligation's own barred
    set. Writing a second walk for the slice root would be the exact mistake
    the docstring below is about.

    DESCENT GOES THROUGH :func:`stelling.coverage.sub_jaxprs`, THE CANONICAL
    ACCESSOR, and must not be hand-rolled here. The predecessor descended via
    ``getattr(v, "jaxpr", None)``, which finds a param that IS a ClosedJaxpr
    but not one holding a COLLECTION of them — and ``cond`` stores its
    branches as a tuple. Measured: a `scatter` inside a `cond` branch was not
    barred at all, while the same primitive at top level, inside `jit`, and
    inside `scan` all barred correctly. That is the UNDER-firing direction,
    which is the worse one.

    THE SLICE ROOT NEEDS THE DESCENT JUST AS MUCH AS THE QUERY ROOT, and the
    reason is measured, not hypothetical: a slice equation can HOLD a barred
    primitive in a sub-jaxpr. With a synthetic barred set of ``{"add"}`` and a
    traced ``jax.ops.segment_sum``, the slice's `scatter-add` equation carries
    ``update_jaxpr = ['add']`` — so the naive flat
    ``{str(e.primitive) for e in sl.eqns}`` yields ``[]`` where this walk
    yields ``['add']``. `tests/test_bar_walk_parity.py` pins both roots
    against `sub_jaxprs`, with that disagreement as its anti-vacuity control.

    The lesson is narrower than "descend properly": **do not hand-roll a
    traversal when a canonical accessor exists.** A second implementation of
    a walk is a second thing to keep correct, and this one silently stopped
    matching.
    """
    # Uses coverage.sub_jaxprs, the SAME nesting walk the coverage tool uses,
    # rather than a private one. The earlier version did
    # `getattr(v, "jaxpr", None)`, which finds a ClosedJaxpr and misses a bare
    # `ir.Jaxpr`, a jaxpr held inside a tuple, and a `NamedTupleParam` field —
    # so a transparent primitive storing its body in any of those forms made
    # the bar UNDER-fire.
    #
    # THIS COMMENT USED TO SAY "no such primitive exists on jax 0.11.0, so
    # this was latent". That is true of 0.11 and FALSE of the other tested
    # series, which is the wrong series to scope a bar's reachability to.
    # Measured on 0.10.2, where `ClosedJaxpr is Jaxpr` is False: the
    # predecessor walk finds 0 sub-jaxprs where `sub_jaxprs` finds 1, for
    # BOTH `scatter-add` (its `update_jaxpr` combiner) and `remat2` (its
    # body). End to end, a `scatter` inside `jax.checkpoint`: this function
    # returns ('scatter',) on both series, the predecessor returned
    # ('scatter',) on 0.11.0 and () on 0.10.2 — the same primitive inside
    # `jax.jit` barred correctly on both, which is what makes it a container
    # bug and not a descent bug. So the hole was LIVE on 0.10 and latent only
    # on 0.11; the migration to the canonical accessor had already closed it
    # before the 0.10 lane existed to fail on it.
    #
    # The lesson survives the correction and is now load-bearing rather than
    # decorative: a bar with weaker reachability than the coverage tool is a
    # hole in the one guard the unshipped rows rest on, "latent" is not a
    # reason to keep two walks, and a latency claim scoped to one jax series
    # is not a latency claim at all.
    from stelling.coverage import sub_jaxprs

    if not VERIFIED_BARRED_PRIMITIVES or eqns is None:
        return ()
    found, seen = set(), set()

    def walk(items):
        for eqn in items:
            name = str(eqn.primitive)
            if name in VERIFIED_BARRED_PRIMITIVES:
                found.add(name)
            for inner in sub_jaxprs(eqn):
                if inner is None or id(inner) in seen:
                    continue
                seen.add(id(inner))
                walk(getattr(inner, "eqns", ()))

    walk(eqns)
    return tuple(sorted(found))


def _barred_primitives(closed) -> tuple[str, ...]:
    """Barred primitives present anywhere in the traced query, innermost
    scopes included — the WHOLE-QUERY root of :func:`_barred_in_eqns`.

    Two jobs, both in :func:`_bar_scope`. It is the FAIL-CLOSED FALLBACK, used
    whenever the per-obligation slices cannot be re-derived — a bar that
    under-fires is worse than one that over-fires, so the fallback is
    deliberately the wider set. And it is the CHEAP UPPER BOUND that decides
    whether the re-derivation is worth doing at all: every equation a slice
    can hold comes from this query at some depth, and this walk descends the
    same nesting, so an empty whole-query set means every slice's set is empty
    too and no query without a barred primitive pays for re-slicing.
    """
    if not VERIFIED_BARRED_PRIMITIVES or closed is None:
        return ()
    jaxpr = getattr(closed, "jaxpr", None)
    try:
        return _barred_in_eqns(getattr(jaxpr, "eqns", ()))
    except Exception:  # a bar must never be the thing that breaks a verdict
        return tuple(sorted(VERIFIED_BARRED_PRIMITIVES))


def _bar_scope_phrase(per_obligation) -> str:
    """The `{scope}` clause: one phrase per obligation, each naming only its
    OWN barred set, so no obligation is ever named beside a primitive its
    slice does not carry. See the comment on :data:`VERIFIED_BAR_REASON`."""
    return "; ".join(
        f"the emitted slice of assert #{index} contains "
        + ", ".join(per_obligation[index])
        for index in sorted(per_obligation)
    )


# THE ONLY KEYS THE NARROWING DECISION MAY READ OUT OF A STAMP'S `options`,
# and the frozen set is the mechanism rather than the documentation of one.
#
# `SolverStamp.options` is a record-carried `dict[str, str]` — the exact emitted
# option set — and the bar's narrowing decision reads it. Nothing constrained
# WHICH keys, so a conjunct keyed on a key no honest stamp carries
# (`options.get("audited_clean") == "yes"`) was a record certifying its own
# cleanliness with NO NEW FIELD ANYWHERE, invisible to every test that removes
# fields or probes their values. Measured on this branch: with that conjunct in
# `_evidence_is_about`, the mispaired assembly the bar withholds comes back
# VERIFIED, full suite green in both columns.
#
# The channel is closed by PROJECTION, not by inspection: `_evidence_options`
# returns a mapping whose keys are a subset of this set, so a key outside it
# cannot reach the decision at all, whatever it is called and whatever it
# holds. Adding a member here is therefore the whole cost of opening the
# channel again, and `tests/test_verified_bar.py` asserts this set EXACTLY.
#
# AND A WHITELIST ON KEYS CONSTRAINS NOTHING ABOUT VALUES. That half was
# measured open on `3e107cf` and is what the split below repairs;
# :func:`_evidence_reproduces` carries the three corruptions and the argument.
_EVIDENCE_OPTION_KEYS = frozenset({
    "smt2_sha256",  # the hash of the exact script that was sent
    "slice_sha256",  # the fingerprint of the slice it was emitted from
    ":timeout",  # z3's spelling of the budget, which is part of the text
    ":tlimit",  # cvc5's
})

# THE TWO WHOSE VALUES EMISSION *DERIVES*. Re-emitting this query's own slice
# recomputes both out of the slice, so a record cannot choose them — it can
# only match, or fail to match, what the re-derivation produces. They are also
# the two the narrowing has no meaning without, which is why an EMPTY one is
# refused rather than compared (see :func:`_reproduced_evidence`).
_EVIDENCE_DERIVED_KEYS = frozenset({"smt2_sha256", "slice_sha256"})

# ... AND THE TWO IT *ECHOES*: the caller's own `solver_timeout_ms`, under the
# spelling its flavour uses. Named here rather than spelled inside
# `_evidence_budget` so that the one function which turns a recorded value into
# an argument contains no string literal at all — see
# `tests/test_verified_bar.py::test_the_evidence_path_cannot_name_a_VALUE`.
_EVIDENCE_BUDGET_KEYS = (":timeout", ":tlimit")


def _whitelisted(raw) -> dict[str, str]:
    """:data:`_EVIDENCE_OPTION_KEYS` out of any mapping, and nothing else.

    Applied to BOTH sides of the narrowing comparison — the record's own option
    set and the one the re-derivation produces — so the two are projected by
    ONE function. Two projections that agree today are two things to keep
    agreeing; this is one thing.

    **AND IT SEES BOTH SIDES, WHICH IS WHY THE VALUE ZONE IS CLOSED UNDER
    CALL.** When this function was added, the zone was an ENUMERATION of four
    names and this was not one of them: not in the source pin's list, not in
    the read ledger's transparent frames (it is handed a plain ``dict`` and
    reads no attribute, so the ledger never sees it at all), and not among the
    functions the one-place scan reads. It could spell anything, and it is
    called once per side. A module-level ``list`` stashed here on the first
    call and returned on the second mints the narrowing at one chosen budget
    with NEITHER pinned signature touched — measured live, full suite
    byte-identical in both columns.

    The repair is not to add this name to a list. The zone is now DERIVED as
    the transitive closure of what :func:`_evidence_reproduces` can call, so a
    helper on this path is inside the pin the moment it is written; every
    module-level name the closure reads must be enumerated AND immutable; and
    the closure may import nothing but the one emission entry point it
    re-derives with. `tests/test_verified_bar.py::test_the_value_zone_is_CLOSED_UNDER_CALL`."""
    return {key: raw[key] for key in _EVIDENCE_OPTION_KEYS if key in raw}


def _evidence_options(stamp) -> dict[str, str]:
    """THE ONE READ of ``SolverStamp.options`` anywhere in the bar, and a
    WHITELIST PROJECTION rather than a copy.

    Returns only :data:`_EVIDENCE_OPTION_KEYS`, so the narrowing decision
    downstream is a function of four named quantities and cannot be a function
    of a fifth. That this is the ONLY reader is not left to convention: the
    bar's functions are scanned for other reads by
    `tests/test_verified_bar.py::test_the_narrowing_decision_reads_options_in_one_place`,
    because a second read site in `_bar_scope` would reopen the channel
    without touching this function or the key set.

    **IT CANNOT AIM, AND THAT IS WHY THE VALUE CHANNEL IS NOT CLOSED HERE.**
    This function is handed a stamp and nothing else — no slice, no ``closed``,
    no re-derivation. Whatever a conjunct written here did with the values it
    reads, it could not compute the mapping it would have to return to mint a
    false narrowing, because that mapping is a function of the QUERY and this
    function never sees the query. :func:`_reproduced_evidence` is safe from
    the other side by the mirror argument — it never sees the record — and
    between them that is why the comparison in :func:`_evidence_reproduces` can
    be one ``==`` over two mappings rather than four reads of two values.

    **AND IT HAS EXACTLY ONE PERMITTED CALLER,** :func:`_evidence_reproduces`.
    A projection is not a permission: whoever calls this reaches every
    whitelisted option value, and two of those values — ``:timeout`` and
    ``:tlimit`` — are the caller's own ``solver_timeout_ms`` carried verbatim
    into the stamp. A conjunct on a whitelisted key's VALUE therefore needs no
    forged record and no new field; it is driven by a public keyword argument.
    Measured on `e35de13`, with this function called from ``_bar_scope``:
    ``solver_timeout_ms=31337`` returned VERIFIED where 20000 returns UNKNOWN,
    with the full suite byte-identical in both columns. The same test file
    scans for the CALL as well as for the attribute, and the read ledger
    attributes an ``options`` read to the function that ASKED rather than to
    this one — a value the whitelist cannot constrain is constrained by
    keeping the number of places that can see it at one.

    Tolerates a mapping or a sequence of pairs (``invocations`` is whatever the
    caller put on the record) and never raises: an unreadable option set
    projects to nothing, which widens."""
    try:
        raw = dict(getattr(stamp, "options", None) or ())
    except Exception:  # noqa: BLE001 — an unreadable option set is no evidence
        return {}
    return _whitelisted(raw)


def _evidence_budget(recorded) -> int | None:
    """THE ONE RAW RECORDED VALUE THAT REACHES THE RE-DERIVATION, named rather
    than left implicit, with its influence bounded BY CONSTRUCTION.

    Emission is a pure function of ``(slice, flavour, timeout)``, so re-deriving
    what an honest record must carry needs the budget that record names. There
    is no caller-independent substitute — the bar cannot guess a number the
    caller chose — so the budget is not kept out of the bar. Its INFLUENCE is
    bounded instead, and the bound is structural rather than sampled.

    **THE BOUND HOLDS. THE ARGUMENT THAT USED TO BE WRITTEN HERE DID NOT, AND
    THE DIFFERENCE MATTERS.** That argument was: whatever this returns is used
    only as :func:`stelling.smt.emit`'s ``timeout_ms``, and the recorded budget
    is itself in the compared set, so a wrong budget puts a wrong
    ``:timeout``/``:tlimit`` in the reproduction, the equality fails, and the
    bar widens. Every step of that is true, and it is a statement about an
    HONEST record — it says a wrong budget disagrees with the budget THIS
    record names. The threat is a record that names whatever it likes: the
    question is not "does the reproduction still match the record it came
    from", it is "can a value returned here turn a reproduction of THIS slice
    into something equal to the projection of a record about a DIFFERENT
    query". Budget-versus-``:timeout`` self-consistency does not answer that.

    **WHAT ACTUALLY FORBIDS IT: the budget cannot reach ``slice_sha256`` at
    all.** ``slice_sha256`` is :func:`stelling.smt.slice_fingerprint` of the
    slice, and emission's ``timeout_ms`` is not one of its inputs — measured
    invariant over twelve budgets spanning 1..60000, including ``True``, which
    :func:`_reproduced_evidence`'s ``isinstance(budget, int)`` admits. It is
    also in :data:`_EVIDENCE_DERIVED_KEYS`, so an empty one is refused rather
    than compared, and it is in :data:`_EVIDENCE_OPTION_KEYS`, so it is part of
    the equality. Two different slices therefore differ in a component NO value
    this function can return is able to move — the bar's own neighbour pair has
    EQUAL ``smt2_sha256`` and DIFFERENT ``slice_sha256``, which is the case
    that separates the two arguments. The worst a conjunct written here can do
    is put a wrong budget in the reproduction, which breaks the ``:timeout``
    equality and WIDENS; it cannot make this slice's reproduction reproduce
    another slice's record, at any budget.
    `tests/test_verified_bar.py::test_the_budget_cannot_reach_the_SLICE_fingerprint`
    pins both halves.

    Contains no string literal: the two spellings live in
    :data:`_EVIDENCE_BUDGET_KEYS`, so the source pin in
    `tests/test_verified_bar.py::test_the_evidence_path_cannot_name_a_VALUE`
    needs no exemption for this function.

    Never raises: an absent, empty or unparseable budget returns None, which
    :func:`_reproduced_evidence` turns into no reproduction, which widens."""
    for key in _EVIDENCE_BUDGET_KEYS:
        text = recorded.get(key)
        if text:
            try:
                return int(text)
            except Exception:  # noqa: BLE001 — an unreadable budget is not a
                return None  # budget, and guessing one would re-emit a lie
    return None


def _reproduced_evidence(sliced, flavour, budget) -> dict[str, str]:
    """WHAT AN HONEST INVOCATION'S WHITELISTED OPTIONS MUST BE, re-derived from
    THIS query's own slice: the canonical, caller-independent projection the
    narrowing decision is a function of.

    **IT IS HANDED NO STAMP AND NO RECORD.** Its three arguments are the slice
    (derived from ``closed``), the flavour LABEL, and the budget already
    reduced to an ``int``. Nothing it can be keyed on tells it what any record
    carries, so a conjunct written here cannot aim: minting a false narrowing
    means returning a mapping EQUAL to the record's, and this function cannot
    see the record's. The signature is pinned for exactly that reason —
    `tests/test_verified_bar.py::test_the_reproduction_is_handed_no_record`.

    **BUILT BY THE SAME FUNCTION THE RECORD IS BUILT BY.**
    :meth:`stelling.smt.Script.stamp_options` is what
    :func:`stelling.solvers.escalate` stamps an invocation's options with, and
    it is what this calls. Not a bar-side re-statement of "what an honest stamp
    carries": there is ONE derivation, so the record and the check cannot
    drift.

    **THE SENTENCE THAT USED TO FINISH THAT PARAGRAPH WAS FALSE.** It said
    "corrupting it corrupts EMISSION — which the byte-level emission tests hold
    and which would change the scripts real solvers answer about". It does not:
    ``stamp_options`` appends ``set-logic``/``smt2_sha256``/``slice_sha256`` to
    an ALREADY EMITTED :class:`stelling.smt.Script` and contributes not one
    byte to ``Script.text``. Measured: a conjunct inside it leaves
    `tests/test_smt_emission.py` and `tests/test_verified_bar.py` both fully
    green, because
    `::test_the_reproduction_comes_from_the_stamps_own_derivation` checks only
    that SUBSTITUTING it MOVES the answer and never checks that the honest one
    is honest — "pinned by substitution rather than by two readings agreeing"
    constrains the substituted function's behaviour not at all. So its honest
    OUTPUT is pinned as well, structurally and by an independently derived
    expectation:
    `::test_the_stamps_own_derivation_is_the_HONEST_one`.

    Returns ``{}`` — which never equals a projection carrying either derived
    hash, so it never narrows — when the flavour is unusable, the budget is not
    a number, emission raises, or either :data:`_EVIDENCE_DERIVED_KEYS` value
    comes back empty. That last one is the gate `218f969` added: a hand-built
    :class:`stelling.smt.Script` defaults ``slice_sha256`` to ``""``, and two
    empty strings comparing equal is not a reproduction of anything."""
    from stelling.smt import emit

    if not flavour or not isinstance(budget, int):
        return {}
    try:
        reproduced = _whitelisted(dict(emit(sliced, flavour, budget).stamp_options()))
    except Exception:  # noqa: BLE001 — an unemittable re-derivation is not
        return {}  # evidence of anything, so it does not narrow
    if any(not reproduced.get(key) for key in _EVIDENCE_DERIVED_KEYS):
        return {}
    return reproduced


def _evidence_reproduces(sliced, stamp) -> bool:
    """THE ONE PLACE A RECORDED OPTION VALUE AND THE RE-DERIVATION MEET — and
    everything that leaves it is a BOOLEAN.

    THE DEFECT THIS SHAPE EXISTS FOR. :data:`_EVIDENCE_OPTION_KEYS` constrains
    WHICH keys reach the narrowing and says nothing about their VALUES, and the
    narrowing itself used to read those values. Three line-neutral corruptions
    were measured on `3e107cf`, each with the full suite byte-identical to
    unmutated in both columns (2055 passed / 2 skipped, 2051 / 6):

    * ``_evidence_is_about`` stashing its projected budget in a module global,
      and ``_bar_scope`` zeroing the barred set when that global is
      ``"30000"``. No new ``(function, attribute)`` pair, and ``_bar_scope``'s
      body carries no ``.options``, no ``"options"`` and no
      ``_evidence_options`` — so neither the read ledger nor either source scan
      sees it, and 30000 is not one of the budgets the sweep samples. Through
      the public API: ``solver_timeout_ms=30000`` returned VERIFIED with no
      withheld note where 20000, 29999, 30001 and 31337 all returned UNKNOWN.
      The isolating control is the IDENTICAL mutation keyed on ``"31337"``,
      which is 2 RED — **the only difference is the constant**;
    * a conjunct on the VALUE of ``smt2_sha256``, and
    * one on the value of ``slice_sha256``. Both live at the site the budget
      sweep is supposed to own, both key on whitelisted keys whose values are
      record-supplied and which the sweep never varies at all, and both turned
      the mispaired assembly's UNKNOWN into VERIFIED with no withheld note
      (measured with the pairing gate satisfied by hand, as the bar's own
      mispairing tests do).

    THE REPAIR IS NOT A WIDER SWEEP. A sample is a sample, and this project has
    watched that answer fail six times on other axes. The narrowing decision —
    :func:`_evidence_is_about` and :func:`_bar_scope` — now consumes a BOOLEAN
    and binds no recorded value at all, so no conjunct on any value is
    EXPRESSIBLE there, at any value, without a read the source scan and the
    read ledger both see. The two functions this one composes cannot aim, each
    for the reason its own docstring gives: :func:`_evidence_options` never
    sees the query, :func:`_reproduced_evidence` never sees the record.

    What is left is this function AND EVERYTHING IT CAN CALL — which is not
    the same thing, and the difference was the next defect. The pin is TOTAL
    over the source rather than a sample over values
    (`tests/test_verified_bar.py::test_the_evidence_path_cannot_name_a_VALUE`),
    and it now runs over the DERIVED closure of what this function reaches
    rather than over a written list of four names
    (`tests/test_verified_bar.py::test_the_value_zone_is_CLOSED_UNDER_CALL`).
    Between them: no string literal outside the attribute names the read ledger
    already permits, no comparison against a literal, no module-global
    smuggling, no default argument, no module-level MUTABLE anywhere in the
    closure, no helper outside it, no import that hides one.

    It does not claim to reach every predicate expressible in Python: a
    discriminator spelled as a method call on a value (``.startswith(...)``) is
    not a comparison and is not matched. **That residue is stated rather than
    hidden — and the sentence that used to say the budget sweep CORROBORATES it
    was wrong.** A sweep is a sample; a conjunct keyed outside the sample is
    invisible to it at every value it does not draw. Measured: two corruptions
    keyed on the recorded ``:timeout`` at 30000 — a default argument on this
    function, and a seventh helper called from it — are 0 RED across the whole
    suite INCLUDING the sweep, at exactly the value the sweep's own comment
    names as unsampled. The sweep is kept because a BEHAVIOURAL check
    anti-correlated with a source check is worth having, not because it
    corroborates this residue. Nothing corroborates this residue.

    WHAT THE COMPARISON IS. The record's whitelisted option set must equal, key
    for key and value for value, the whitelisted option set the re-derivation
    produces. That is strictly stronger than the two-hash equality it replaces:
    a record carrying BOTH ``:timeout`` and ``:tlimit`` used to narrow on
    whichever the reader looked at first, and now matches neither. Honest
    records are unaffected — :func:`stelling.smt._options` emits one spelling
    per flavour — and both sides go through :func:`_whitelisted`, so a key
    outside the whitelist cannot make the two differ either."""
    recorded = _evidence_options(stamp)
    reproduced = _reproduced_evidence(
        sliced, getattr(stamp, "name", None), _evidence_budget(recorded)
    )
    return bool(reproduced) and recorded == reproduced


def _evidence_is_about(sliced, invocations) -> bool:
    """Is this obligation's SOLVER EVIDENCE evidence about THIS slice?

    The narrowing's whole content is "the emission row was not consulted about
    the obligations the solver decided", and that is a statement about the
    RUN the escalation came from, not about ``closed``. Re-slicing ``closed``
    answers it only if the two are the same query, so the pairing has to be
    MEASURED rather than assumed. TWO recorded quantities are checked, and
    they are checked because ONE OF THEM IS NOT ENOUGH — the reason is the
    defect this function's predecessor shipped with, so it is stated first:

    **A SCRIPT HASH CANNOT WITNESS WHICH SLICE PRODUCED IT.** Emission IS a
    pure function of ``(slice, flavour, timeout)``; what the guard needs is
    the converse, *equal script implies equal slice*, and that is FALSE — false
    for exactly the primitive under the bar. The static-index ``scatter`` SET
    form emits no line at all (``smt.emit``'s ``prim == "scatter"`` branch
    routes terms and appends nothing), so an element the write did not touch
    aliases the operand's term and a scatter-bearing slice emits byte for byte
    what a scatter-free slice reading the same element emits. Measured, jax
    0.11.0, x64, ``s = x.at[0].set(0.5)``::

        assert s[1] - x[1] <= 0   slice barred ('scatter',)  sha 2896a0f2…
        assert x[1] - x[1] <= 0   slice barred ()            sha 2896a0f2…

    On that pair the predecessor of this function returned True for an
    escalation produced on the FIRST query and stamped against the SECOND, and
    ``_bar_scope`` narrowed to ``((), '')`` — VERIFIED where `8e42934`'s
    whole-query bar returns UNKNOWN.

    So the second quantity comes from THE SLICE rather than from its text:
    :func:`stelling.smt.slice_fingerprint`, the sha256 of the slice's primitive
    names with their nesting depth, recorded as ``slice_sha256`` beside
    ``smt2_sha256`` on every invocation. Both must reproduce from the slice
    re-derived out of ``closed``, at the flavour and timeout the stamp itself
    records.

    WHAT EACH ONE PROVES, EXACTLY:

    * ``smt2_sha256`` matching proves the TEXT the solver answered about is
      the text THIS slice emits. That is what makes the answer transferable at
      all — an ``unsat`` on that text is an ``unsat`` about this obligation's
      encoding.
    * ``slice_sha256`` matching proves the emission ran on a slice with the
      same primitive topology as this one, hence with the SAME BARRED SET.
      That is the bar's actual question, and it is the one the text cannot
      answer.

    WHAT NEITHER PROVES, and this is the part that must not be over-read:

    * **that the record is honest.** Both hashes are carried by the
      escalation, so a fabricated record can carry any pair of values it
      likes — as it can already carry a fabricated ``outcome``.
    * **that the REST of the verdict is about this query.** Even an honest
      pair speaks only for the obligation it stamps. The obligations INTERVALS
      decided come from the paired propagation, which nothing here reads.
      Measured: when the two queries' decided slices are genuinely the same
      expression, both hashes match, the bar narrows correctly (the barred row
      really was not involved), and the assembly reaches VERIFIED on a query
      whose honest verdict is REFUTED. No definition of "which slice did the
      solver answer about" can close that: the false claim is about a
      DIFFERENT obligation.

    So what this function is for is the bar's question — was the unaudited
    emission row involved in what the solver decided — and the pairing itself
    is not its job and never was. THE VERSION OF THIS PARAGRAPH THAT SAID A
    WHOLE-QUERY BAR "HAPPENED TO BACKSTOP" THE PAIRING AND A SCOPED ONE DOES
    NOT WAS MEASURING ONE CLASS OF QUERY AND GENERALISING. That backstop
    covered scatter-bearing queries only, because those are the only ones any
    version of the bar looks at. The identical mispaired VERIFIED on a REFUTED
    query is reachable with NO BARRED PRIMITIVE ANYWHERE, on every build
    including `8e42934` — so the true statement is not "scoping cost a
    backstop" but **scoping revealed that `make_solver_verdict` never bound
    its three arguments to one query.** It does now, by the query hash
    (:func:`stelling.solvers.make_solver_verdict`'s fourth
    :exc:`~stelling.solvers.MispairedEscalationError`), which is a different
    key from either of this function's and covers the whole query rather than
    one slice. See
    `tests/test_verified_bar.py::test_the_pairing_gate_closes_the_SCATTER_FREE_row`,
    which is the row that settles which of the two statements is true.

    THE POLARITY IS THE POINT, AND IT IS WHY THIS IS NOT THE DELETED
    `barred_on_slice` FIELD COMING BACK. A record cannot use either hash to
    CLEAR the bar — a missing stamp, a missing hash, a missing fingerprint, an
    unrecognised option profile, an emission that raises, and either hash
    failing to match all return False, and False WIDENS the bar to the whole
    query. Adding the second conjunct can only make narrowing RARER than the
    predecessor's, which is why the change cannot make a bar fire less than it
    did. `barred_on_slice` ran the other way: it was a positive claim about
    barred primitives that nothing recomputed. Nothing here names a barred
    primitive; the barred set is still `_barred_in_eqns` over ``closed``'s own
    re-derived slice.

    ``invocations`` is read here and NOT in the bar's domain, deliberately:
    the domain is `outcome == OB_DISCHARGED`, the same predicate that
    discharges (see :func:`stelling.solvers.make_solver_verdict`). A record
    that gives up its ``invocations`` keeps its discharge, stays in the domain,
    and loses its narrowing — the drift that made those two different concepts
    ran the other way.

    **AND IT HOLDS NO RECORDED VALUE.** This is the DECISION, and the whole of
    what it sees about a stamp is one boolean from
    :func:`_evidence_reproduces`. Reading the two hashes and the budget HERE is
    what made a conjunct on a whitelisted key's VALUE expressible at the
    narrowing itself, invisible to the read ledger, to both source scans and to
    the budget sweep at once — three such corruptions are recorded, with their
    measurements, on :func:`_evidence_reproduces`. Widening the sweep is not
    the repair; not binding the values is.
    """
    for stamp in invocations or ():
        # `not stamp.name` is UNCONSTRUCTIBLE for a real stamp and is kept as
        # a duck-type guard, not as a live branch: `SolverStamp.__post_init__`
        # refuses an `invoked=True` stamp with an empty name, so no
        # `SolverStamp` can reach it. It survives only because `invocations`
        # is whatever the caller put on the record, and an object without a
        # usable name must widen rather than reach `emit` with it.
        if not getattr(stamp, "invoked", False) or not stamp.name:
            continue
        # A BOOLEAN, and deliberately nothing else. `stamp.name` is the
        # emission FLAVOUR and the budget is part of the emitted text, so both
        # have to reach the re-derivation — but neither reaches THIS function,
        # which is the one that decides. See `_evidence_reproduces`.
        if _evidence_reproduces(sliced, stamp):
            return True
    return False


def _bar_scope(closed, decided) -> tuple[tuple[str, ...], str]:
    """THE BAR'S SCOPE FOR ONE VERDICT: ``(barred primitives in scope, the
    clause naming where they are)``, derived from ``closed`` alone plus the
    obligations the solver decided — ``decided`` maps each such obligation's
    INDEX to the invocation stamps its record carries.

    Never reads a barred primitive off the escalation, and the block comment
    above says what that bought — a recorded scope is a positive claim nothing
    validates, and at the time this was written `make_solver_verdict` did not
    bind its escalation to its query at all, so a read scope was forgeable in
    one direction and mispairable in the other. The pairing is bound now, by
    the query pairing gate one layer up; the CONTENTS are still derived here,
    because a gate on the whole query says nothing about which slice a
    recorded invocation answered. Re-slicing out of ``closed`` has neither
    exposure for the scope's CONTENTS, and keeps the precision: the slices are
    the ones this query's own obligations produce.

    ITS DOMAIN IS STILL SUPPLIED, AND THAT IS A PRECONDITION RATHER THAN AN
    IMMUNITY. The index set comes from the caller — in practice from
    `make_solver_verdict` reading `outcome == OB_DISCHARGED` off the
    escalation's records, the same test that discharges those obligations —
    and an obligation absent from it is simply not asked about. No field a
    record can carry names a barred primitive, so the CONTENTS cannot be
    forged; the DOMAIN rests on `make_solver_verdict`'s documented
    precondition that its escalation came from `escalate()` on the same query,
    and a caller who can violate that can hand-build a `Verdict` and skip this
    function entirely.

    NARROWING IS EARNED PER OBLIGATION, BY :func:`_evidence_is_about`. A
    re-derived slice narrows the bar only when the recorded invocation
    reproduces BOTH its ``smt2_sha256`` and its ``slice_sha256`` from it.
    Everything else — no stamps, no hash, no fingerprint, an unrecognised
    option profile, a slice that declines, an exception anywhere — returns the
    whole-query set. That is the repair for TWO measured regressions, not a
    hardening. The first predecessor narrowed on the INDEX alone, so an
    escalation produced on one scatter-bearing query and stamped against
    another of the same shape re-sliced cleanly, found nothing, and returned
    VERIFIED where the whole-query bar at `8e42934` returned UNKNOWN. The
    second narrowed on the SCRIPT HASH alone, which does not distinguish
    slices: the barred row emits no text, so `s[1] - x[1] <= 0` (barred
    `('scatter',)`) and `x[1] - x[1] <= 0` (barred `()`) hash the same script,
    and the same mispairing cleared the bar again. See
    :func:`_evidence_is_about` for what each hash does and does not prove.

    WHAT A "STRAY INDEX" ACTUALLY DOES — FOUR behaviours, not one, and an
    earlier wording here claimed the third for all of them ("a stray index
    does not slice, so the derivation drops to the whole-query set"). A later
    one listed three and presented them as the whole space, which is the same
    mistake one item shorter:

    * an index matching an obligation INTERVALS already decided SLICES
      perfectly well and contributes its own barred set (measured:
      `slice_obligation(closed, 1, env)` on the bar's own fixture returns
      `['broadcast_in_dim', 'ge', 'scatter']`). It is not in the domain
      because it was not solver-decided, not because it fails to slice;
    * a NEGATIVE index within range is Python indexing all the way down: `-1`
      slices the LAST obligation and would render "the emitted slice of assert
      #-1";
    * an index matching no top-level assert equation at all (`99`) declines —
      "obligation #99 has no matching top-level stelling_assert equation" —
      and that is the case the sentence was true of;
    * a negative index PAST the start (`-3` on the bar's two-obligation
      fixture, and anything below it) raises `IndexError` out of
      `slice_obligation` rather than declining, and is caught by this
      function's outer `except` — the same whole-query set by a different
      door. It is named because "three behaviours" was being read as a closed
      enumeration and it was not one; the list above is the four that have
      been MEASURED, and it is not claimed to be closed either.

    All four end at the whole-query set, and the first three by the evidence
    check rather than by the slicer: none of them carries an invocation whose
    script AND slice fingerprint both re-emit from the obligation it names. An
    EMPTY domain is the one case that silences the bar, and it is empty
    exactly when no record discharged anything — in which case there is no
    solver-decided VERIFIED to withhold.

    FAILS CLOSED, ALWAYS TOWARD THE WIDER BAR: every path that is not a
    slice-plus-both-matching-hashes returns the whole-query set rather than
    silence.
    """
    whole = _barred_primitives(closed)
    if not whole:
        # nothing barred anywhere in the query, so nothing on any slice of it
        return (), ""

    def fallback(why: str) -> tuple[tuple[str, ...], str]:
        return (
            whole,
            "the traced query contains "
            + ", ".join(whole)
            + f" ({why}, so the bar fell back to the whole query)",
        )

    try:
        from stelling.obligation import DeclinedObligation, slice_obligation
        from stelling.propagate import interval_env

        # a mapping is the contract; anything else lands in the except below
        # and widens, which is the direction a misread domain must fail in
        domain = dict(decided)
        env = interval_env(closed)
        per: dict[int, tuple[str, ...]] = {}
        for index in sorted(domain):
            sliced = slice_obligation(closed, index, env)
            if isinstance(sliced, DeclinedObligation):
                return fallback(
                    f"the decided obligation #{index} does not slice out of "
                    f"the query being stamped"
                )
            if not _evidence_is_about(sliced, domain[index]):
                # the slice re-derived fine; what is missing is any evidence
                # that the solver was ever asked THIS question. Narrowing on
                # it would be narrowing on an index, which is how a mispaired
                # `closed` cleared this bar.
                return fallback(
                    f"no recorded solver invocation for the decided "
                    f"obligation #{index} reproduces both this query's slice "
                    f"of it and the script that slice emits, so the "
                    f"escalation is not evidence about this query"
                )
            found = _barred_in_eqns(sliced.eqns)
            if found:
                per[index] = found
    except Exception:  # noqa: BLE001 — a bar must never break a verdict, and
        return fallback(_BAR_UNDERIVABLE)  # it must never go quiet either
    return (
        tuple(sorted({p for found in per.values() for p in found})),
        _bar_scope_phrase(per),
    )


def undecided_cause_note(coverage, obligations, unaccounted_runs=0
                         ) -> tuple[str, ...]:
    """One note classifying what the coverage instrument measured about a
    verdict that carries undecided obligation(s) — or () when none is
    undecided (docs/proposed-decline-messages.md #1: an UNKNOWN whose
    coverage is complete is a different situation from one downstream of
    ⊤ gaps, and the verdict says which it measured).

    Claims only measurements: the counts are the coverage instrument's
    own, and the complete-coverage branch asserts nothing beyond them —
    an interval straddle is compatible with BOTH a precision near-miss
    and a genuine violation, and the note says so rather than pick one.
    Shared by both assembly paths (interval-only and solver), so the two
    cannot drift.

    ``unaccounted_runs`` IS THE THIRD SITUATION, AND IT IS THE ONE THE
    COMPLETE-COVERAGE BRANCH USED TO MISATTRIBUTE. That branch reasons by
    elimination: coverage is complete, therefore what remains is the interval.
    The elimination is only valid if the escalation actually arrived. When the
    solver path can see that it did not — the ledger witnesses invoked runs
    that no supplied record accounts for — the obligation is undecided because
    its OUTCOME went missing, and "the propagated interval straddling the
    asserted bound" is a wrong explanation of a verdict the argument's shape
    caused. `stelling.solvers.make_solver_verdict` REFUSES the degenerate
    `records` it can prove degenerate (an empty one against a working ledger);
    what it cannot refuse without also refusing a deliberate probe of a
    different invariant is a non-empty STRICT SUBSET, and that residue lands
    here. It is soundness-harmless — a dropped record leaves its obligation
    ``unknown``, which can never mint VERIFIED — but silence about the cause is
    not the same as a wrong cause, and this branch is what keeps the two apart.
    The interval-only path never passes it: there is no ledger there.

    `tests/test_verified_bar.py::test_a_STRICT_SUBSET_records_does_not_blame_the_INTERVAL`."""
    if not any(o.status == "unknown" for o in obligations):
        return ()
    c = coverage
    parts = []
    if c.unknown:
        names = ", ".join(f"{n} ×{k}" for n, k in c.unknown_primitives)
        parts.append(f"{c.unknown} equation(s) fell to ⊤ ({names})")
    if c.unreached:
        parts.append(f"{c.unreached} equation(s) unreached")
    if c.inert:
        parts.append(f"{c.inert} constraint(s) dropped")
    if unaccounted_runs:
        return (
            f"undecided obligation(s), and the ESCALATION IS INCOMPLETE: the "
            f"ledger witnesses {unaccounted_runs} invoked solver run(s) that "
            f"no supplied record accounts for, so at least one obligation is "
            f"undecided because its outcome did not arrive — not because of "
            f"anything the propagation measured, and this verdict says "
            f"nothing about whether a solver could have settled it. `records` "
            f"is a container that does not yield the same thing on every "
            f"pass; pass it as a materialised sequence and re-run"
            + (f" (coverage also reports: {'; '.join(parts)})" if parts
               else ""),
        )
    if not parts:
        return (
            f"undecided obligation(s), and transfer coverage is not the "
            f"cause: {c.known}/{c.total} equations ran a registered "
            f"transfer, none fell to ⊤ and no constraint was dropped. "
            f"What remains is the propagated interval straddling the "
            f"asserted bound — a straddle is compatible with both a "
            f"precision near-miss and a genuine violation, and interval "
            f"arithmetic alone cannot tell which (the obligation detail "
            f"quotes the straddle where a top-level comparison produced "
            f"it)",
        )
    return (
        "undecided obligation(s) with coverage gaps in the query: "
        + "; ".join(parts)
        + " — the undecided status may be downstream of these; see the "
        "coverage line and the decline notes",
    )


def top_despite_coverage_note(propagation: Propagation) -> str | None:
    """The stamp's :attr:`Stamp.top_despite_coverage` field, derived from
    the propagation that ran — or ``None`` when this reading did not fire.

    THE INVERSION, and why the field is worded as one. The coverage line
    is a membership census: `unknown = 0` says every primitive in the
    query has a registered transfer, and it is read as "the analysis saw
    all of it". Measured in the external census of ``jax_md``'s
    ``space.distance`` (docs/state-0.1.0.md),
    :func:`stelling.coverage.measure` reports total=14, known=11,
    transparent=3, unknown=0 while the interval face for the same jaxpr
    is ⊤. The same split survives into the live counter for a different
    reason: a decline is counted there, but a transfer that SUCCEEDS and
    returns an unbounded box is not — ``exp(x) - exp(x)`` over
    x ∈ [-1000, 1000] counts 4/4 known, nothing fallen to ⊤, and
    propagates [-inf, inf].

    So this states what was NOT established — that the counts above
    bounded anything — and never that coverage is complete or correct.
    Its absence is not the complement of its presence: it fires on the
    top-level ⊤ boxes :attr:`Propagation.top_boxes` recorded, and a
    query with none of those has simply not had this particular reading
    made about it.

    Gated on the census reporting no gap of its own (``unknown``,
    ``unreached`` and ``inert`` all zero): when the census DOES report
    one, the coverage line already carries the disclosure and this field
    would only repeat it under a stronger-sounding name.

    Shared by both assembly paths (:func:`make_verdict` and
    :func:`stelling.solvers.make_solver_verdict`), so the two surfaces
    cannot disagree about whether the field exists.
    """
    c = propagation.coverage
    if c.unknown or c.unreached or c.inert:
        return None
    if not propagation.top_boxes:
        return None
    n = sum(k for _, k in propagation.top_boxes)
    names = ", ".join(f"{p} ×{k}" for p, k in propagation.top_boxes)
    return (
        f"NOT ESTABLISHED — that the coverage line bounded this query. "
        f"{n} propagated value(s) came out ⊤ (every element [-inf, inf], "
        f"the widest box there is), at {names}, while the census recorded "
        f"no equation fallen to ⊤ and none unreached. A registered "
        f"transfer can return ⊤ on the values it is "
        f"handed, and the census counts whether a primitive HAS a transfer "
        f"registered, never what the transfer returned — so the "
        f"{c.known}/{c.total} figure is not a statement that anything here "
        f"was bounded, and this verdict does not make one"
    )


def _query_has_non_f64_float(closed) -> bool:
    """Whether the query contains any non-float64 float dtype in its
    equations' operands or results. Used to select the parametric ieee
    stamp when narrower formats are present."""
    for eqn in closed.jaxpr.eqns:
        for v in (*eqn.invars, *eqn.outvars):
            dtype = getattr(getattr(v, "aval", None), "dtype", None) or ""
            if "float" in dtype and dtype != "float64":
                return True
    return False


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

    # -- reaches-output conjunct ------------------------------------------------
    #
    # A violated obligation whose predicate operand does NOT reach any output
    # of the harness function is a violation on a dead variable: the caller
    # never observes the bad value.  Downgrade from REFUTED to UNKNOWN with a
    # note explaining why.  Only REFUTED verdicts are affected; VERIFIED and
    # UNKNOWN are unchanged.
    #
    # Matching is by IDENTITY: each ObligationReport carries the Var IDs of
    # its assert equation's invars (operand_var_ids), populated by the
    # propagator at construction.  No positional indexing against the jaxpr's
    # equation list — obligations from sub-jaxprs (forced cond branches, jit
    # bodies) interleave with top-level ones and positional mapping misaligns.
    obligations = propagation.obligations
    reachability_notes: list[str] = []
    if status == "REFUTED":
        live = reaches_output(closed.jaxpr)
        scope = defined_vars(closed.jaxpr)
        downgraded: list[ObligationReport] = []
        for ob in obligations:
            if ob.status != "violated-over-set":
                downgraded.append(ob)
                continue
            # If the obligation carries no operand_var_ids (unexamined,
            # or from a path that did not record them), fail-safe: never
            # downgrade what we cannot prove dead.
            if not ob.operand_var_ids:
                downgraded.append(ob)
                continue
            # If the obligation's var IDs are not in the top-level
            # jaxpr's scope, the obligation came from a sub-jaxpr (cond
            # branch, etc.) and the top-level walk cannot judge it.
            # Fail-safe: keep as REFUTED.
            if not any(vid in scope for vid in ob.operand_var_ids):
                downgraded.append(ob)
            elif any(vid in live for vid in ob.operand_var_ids):
                # Reachable: the violation stays.
                downgraded.append(ob)
            else:
                # Unreachable: downgrade to UNKNOWN with a note.
                reachability_notes.append(
                    f"obligation #{ob.index} is violated but the violated "
                    f"variable does not reach any output of the harness "
                    f"function"
                )
                downgraded.append(
                    ObligationReport(
                        index=ob.index,
                        status="unknown",
                        detail=(
                            f"violated-over-set but unreachable: the "
                            f"predicate operand does not flow to any output "
                            f"of the traced function (dead variable)"
                        ),
                        source_info=ob.source_info,
                        operand_var_ids=ob.operand_var_ids,
                    )
                )
        obligations = tuple(downgraded)
        # Re-evaluate status: if all violations were unreachable, no
        # reachable violation remains to drive REFUTED.
        if not any(o.status == "violated-over-set" for o in obligations):
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

    notes = propagation.notes + undecided_cause_note(
        propagation.coverage, obligations
    ) + tuple(reachability_notes)
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
        # Detect whether the query uses non-float64 formats by checking
        # the declarations (stelling_any equations) for non-f64 float dtypes
        _has_non_f64 = _query_has_non_f64_float(closed)
        if _has_non_f64:
            semantics = SEMANTICS_IEEE_FMT
            arithmetic_mode = ARITHMETIC_MODE_INTERVAL_IEEE_FMT
        else:
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
        # both facts the reading needs are here — the census's counts and
        # the walk's ⊤ boxes — and nowhere else: the census itself holds
        # only the IR and a set of primitive names
        top_despite_coverage=top_despite_coverage_note(propagation),
    )
    return Verdict(
        status=status,
        obligations=obligations,
        stamp=stamp,
        notes=notes,
    )
