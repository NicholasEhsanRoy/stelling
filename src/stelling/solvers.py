# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Solver escalation: transports, portfolio dispatch, stamps, verdicts.

The layer above :mod:`stelling.propagate`: exactly the obligations
interval propagation left ``unknown`` are escalated to SMT solvers, over
the SMT-LIB2 text :mod:`stelling.smt` emits. Three transports deliver
that text — z3 wheel bindings (in-process, wall-guarded by a thread plus
``Context.interrupt``), cvc5 wheel bindings (fed through the wheel's own
SMT-LIB2 parser in a wall-guarded child process, because the wheel holds
the GIL for the whole check and its ``tlimit`` does not reliably preempt
the coverings solver — measured on cvc5 1.3.4), and an external cvc5
binary over stdin (discovered via ``stelling._optional.cvc5_binary``).

Dispatch is a portfolio: every fragment here (QF_LRA, QF_NRA) runs every
installed solver — cvc5 primary for nonlinear, z3 primary for linear.
Agreement on a definitive answer decides; a sat/unsat **disagreement
raises** :exc:`SolverDisagreement` (a bug oracle, never a tiebreak); a
``sat`` becomes REFUTED only through :func:`make_validated_witness`, the
dispatch path's only witness-construction site, whose single validator
(:func:`stelling.obligation.witness_is_valid`) checks box membership AND
the exact-rational violation as one conjunction — a failing conjunct
raises :exc:`EmissionInfidelityError`, and a witness the replay cannot
evaluate exactly (a non-rational model value, or a rational-``pow`` point
whose exact value is irrational — :exc:`stelling.obligation.ReplayDeclined`)
leaves the obligation UNKNOWN by policy. ``unknown``/timeout is UNKNOWN, never
VERIFIED. Every invocation is stamped **at the moment of invocation**:
the fully-populated :class:`SolverStamp` is appended to an append-only
ledger BEFORE the transport runs, so the record of the ask can never be
narrated from a result, and absence is derived from the empty ledger,
never written. A spawn counter incremented at the transport-entry
boundary is checked against the ledger before any escalated verdict
emits (:exc:`ProvenanceError` on divergence). Solvers are never invoked
on defaults, and the dispatch config's time limit is required.

**WHO ANSWERED IS NOT WHO WAS ASKED, and both are recorded.** A stamp is
the record of the ask; a backend that was invoked and stamped but
returned failed/unknown/timeout contributed nothing to the outcome, so a
two-backend stamp is compatible with a one-backend decision. That gap is
derived per obligation (:attr:`ObligationEscalation.answered_by`,
surfaced as :attr:`stelling.verdict.Verdict.solver_redundancy`) and, when
a decision rests on fewer than :data:`PORTFOLIO_SIZE` answers, said out
loud in the notes and on the obligation's own detail line. It matters
most on a DISCHARGE: a ``sat`` reaches REFUTED only through independent
exact-rational replay, while an ``unsat`` is a universal claim nothing
re-derives — there the second backend is the only cross-check there is.

Guard rule: every decline and every solver failure (crash, garbage
output, missing binary, unsupported fragment) degrades the obligation to
UNKNOWN with the reason quoted in the verdict notes. Raising is reserved
for stamp-contract violations, :exc:`SolverDisagreement`,
:exc:`EmissionInfidelityError`, and :exc:`ProvenanceError`.

Zero-dep at import time; solver wheels are reached only through
``stelling._optional`` inside call paths.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable

from stelling import _optional
from stelling import ir
from stelling import verdict as _verdict
from stelling.obligation import (
    DeclinedObligation,
    ObligationSlice,
    ReplayDeclined,
    fraction_text,
    slice_unknown_obligations,
    violating_elements,
    witness_is_valid,
)
from stelling.interval import IEEE_ENDPOINT_ASSUMPTION
from stelling.propagate import ObligationReport, Propagation, interval_env
from stelling.smt import Script, emit
from stelling.verdict import (
    ARITHMETIC_MODE_INTERVAL,
    ARITHMETIC_MODE_INTERVAL_IEEE,
    REAL_CONVENTION_ASSUMPTION,
    SEMANTICS_IEEE,
    SEMANTICS_REAL,
    SolverStamp,
    Stamp,
    Verdict,
    Witness,
    solver_absent,
)

__all__ = [
    "EmissionInfidelityError",
    "Escalation",
    "MispairedEscalationError",
    "ObligationEscalation",
    "ProvenanceError",
    "SolverConfig",
    "SolverDisagreement",
    "escalate",
    "make_solver_verdict",
    "make_validated_witness",
]

TRANSPORT_Z3_WHEEL = "wheel-bindings (smt2 text)"
TRANSPORT_CVC5_WHEEL = "wheel-bindings (smt2 text; wall-guarded child process)"
TRANSPORT_CVC5_BINARY = "external-binary subprocess"

INSTALL_HINT = (
    'no SMT solver is installed — pip install "stelling[solvers]" (or set '
    "STELLING_CVC5 / put cvc5 on PATH) to enable escalation"
)

# The v1 refusal for constrained-assume propagations, quoted verbatim in
# the escalation note and every declined obligation's detail. The emitted
# SMT problem carries the DECLARED box bounds and never the assume
# constraints, and the witness validator checks membership in the declared
# box — so while `unsat` over the wider box would be sound a-fortiori,
# `sat` is not: a model violating the assumed precondition would pass the
# membership∧violation conjunction and mint a REFUTED-with-witness that
# does not refute the conditional claim the propagation stamped. Full
# refusal is the sound v1; faithful narrowed-bounds emission is a later,
# separately-audited build.
DROPPED_ASSUME_REFUSAL = (
    "assume DROPPED present: the precondition had no decidable box, so the "
    "query ran over a SUPERSET of the intended set — a sat witness could lie "
    "outside the precondition entirely while the verdict reads REFUTED. "
    "VERIFIED over a superset still implies VERIFIED over the intended set "
    "and is unaffected; escalation declines rather than mint a witness that "
    "answers a weaker question than the one written"
)

# The next-step sentences are docs/proposed-decline-messages.md #5's
# intent against THIS tree (the proposal predates the constrain-mode
# refusal: its 'a single-input bound does not disable escalation' is
# false here — a constraining assume is exactly what fires this refusal,
# so the working form is the declaration-side bound). Every sentence is
# a measured mechanism: the refusal precedes backend discovery in
# escalate(), it keys on coverage.constrained alone, and the
# declaration-stated bound produces the identical narrowed box with
# escalation unaffected — each pinned in
# tests/test_constrained_refusal_message.py.
CONSTRAINED_ASSUME_REFUSAL = (
    "constrained assume present: solver escalation emits over the declared "
    "box, which does not respect the assumed precondition — a sat witness "
    "could violate the precondition while the verdict claims "
    "conditionality; escalation declines until constrained bounds can be "
    "emitted faithfully. This refusal keyed only on the constraining "
    "assume being present, and it fired before any solver was looked for: "
    "removing the assume removes it (escalation is then attempted, and "
    "any remaining decline names its own reason). WHAT WORKS TODAY: a "
    "bound on a single declared input can be stated in the declaration "
    "itself — the envelope passed to any_array — which narrows the same "
    "box without disabling escalation"
)

# The ieee-mode refusal, quoted verbatim in every declined obligation's
# detail and in the escalation notes. The SMT backends emit over the reals
# (QF_LRA/QF_NRA): an `unsat` there proves the ℝ obligation, not the
# binary64 one, and stamping it against an ieee-semantics claim would be
# exactly the ℝ-vs-float confusion the semantics dial exists to prevent.
IEEE_SEMANTICS_REFUSAL = (
    "ieee semantics: the SMT backends emit over the reals (QF_LRA/QF_NRA), "
    "which model neither binary64 rounding nor overflow-to-inf nor NaN — "
    "escalating would prove the ℝ obligation under an ieee-stamped claim; "
    "escalation declines until a float-semantics emission ships as its own "
    "audited build"
)

# The portfolio's designed redundancy: TWO backends, independently, on the
# same emitted text. It is a constant rather than `len(backends)` on
# purpose — the question a reader has is "did this verdict get the
# cross-check the design promises", and measuring the answer against
# whatever happened to be installed answers a different, easier question.
PORTFOLIO_SIZE = 2

OB_DISCHARGED = "discharged"
OB_VIOLATED_WITNESS = "violated-witness"
# a constants-only obligation (no declared inputs) whose predicate the
# exact-rational replay proved false: refuted with no witness values to
# render — the set-level "definitely false" reading is exact here.
OB_VIOLATED_CONSTANT = "violated-constant"
OB_UNKNOWN = "unknown"

# quoted verbatim when a sat answer arrives with no usable model (audit F2)
NO_USABLE_MODEL = "sat reported but no usable model returned"

_REPLAY_SENTENCE = (
    "confirmed by independent exact-rational replay (fractions.Fraction "
    "arithmetic, pure Python, no solver): the predicate is false at this point"
)


class SolverDisagreement(RuntimeError):
    """Two solvers returned conflicting definitive answers on one script.

    A loud bug oracle by design: never a silent pick, never a tiebreak.
    Carries both verdicts, both option sets, and the script text(s).
    """

    def __init__(
        self,
        obligation_index: int,
        verdicts: tuple[tuple[str, str], ...],
        options: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
        scripts: tuple[tuple[str, str], ...],
    ) -> None:
        self.obligation_index = obligation_index
        self.verdicts = verdicts
        self.options = options
        self.scripts = scripts
        summary = ", ".join(f"{name}: {answer}" for name, answer in verdicts)
        super().__init__(
            f"solver disagreement on obligation #{obligation_index} "
            f"({summary}) — one of the solvers, or this emission, is wrong; "
            f"refusing to pick. Scripts and option sets attached."
        )


class EmissionInfidelityError(RuntimeError):
    """A solver witness failed the independent exact-rational replay.

    The SMT problem did not mean exactly the obligation — the worst defect
    this layer can have. Loud by design; never degraded to a verdict.
    """

    def __init__(
        self,
        obligation_index: int,
        solver: str,
        values: tuple[tuple[str, str], ...],
        script_text: str,
        detail: str,
    ) -> None:
        self.obligation_index = obligation_index
        self.solver = solver
        self.values = values
        self.script_text = script_text
        witness = ", ".join(f"{n}={v}" for n, v in values)
        super().__init__(
            f"emission-infidelity bug on obligation #{obligation_index}: "
            f"{solver} reported sat with model ({witness}) but {detail}. "
            f"The emitted SMT problem does not mean the obligation."
        )


class MispairedEscalationError(RuntimeError):
    """A propagation whose escalation is refused outright was paired with
    an escalation that performed solver work.

    Two refusal classes share this shape. (1) A CONSTRAINED propagation:
    the escalation-refusal invariant ("a constrained propagation never
    reaches a solver") is enforced twice, anti-correlated
    (SOUNDNESS.md: one invariant, two mechanisms): once in
    :func:`escalate` against the propagation it receives, and once at
    verdict assembly against the propagation being STAMPED. The
    second mechanism catches the mode-mixed caller bypass (escalate on
    an inert propagation, assemble against a constrained one — audit
    F3), where a witness violating the stamped precondition would mint
    a REFUTED that does not refute the conditional claim. (2) An
    IEEE-SEMANTICS propagation: the SMT backends emit over the reals, so
    solver outcomes prove the ℝ obligation — stamping them against an
    ieee-semantics claim (escalate on the real-mode propagation,
    assemble against the ieee one) would mint an ℝ-proved verdict under
    a float-semantics stamp; same two-mechanism shape
    (:data:`IEEE_SEMANTICS_REFUSAL` in :func:`escalate`, this gate at
    assembly). In both classes the propagation may only ever pair with
    a refusal-shaped escalation: zero invocations, zero spawns, zero
    witnesses, every record UNKNOWN.
    """


class ProvenanceError(RuntimeError):
    """The spawn counter and the invocation ledger diverged.

    The counter increments at the transport-entry boundary; the stamp is
    appended at the dispatch site — two mechanically disjoint records of
    the same events, deliberately anti-correlated so one bug cannot
    silently satisfy both. On any escalated verdict they must agree; on
    mismatch the verdict does not emit, ever. Carries both counts and the
    stamps.
    """

    def __init__(
        self, spawns: int, stamped: int, stamps: tuple[SolverStamp, ...]
    ) -> None:
        self.spawns = spawns
        self.stamped = stamped
        self.stamps = stamps
        super().__init__(
            f"provenance divergence: {spawns} transport spawn(s) but "
            f"{stamped} invoked=True stamp(s) in the ledger; refusing to "
            f"emit a verdict over divergent provenance. Stamps: {stamps!r}"
        )


@dataclass(frozen=True)
class SolverConfig:
    """Escalation configuration. The time limit is required — this layer
    follows the same no-defaults discipline it imposes on solvers."""

    timeout_ms: int
    only: tuple[str, ...] | None = None  # restrict to a subset of {"z3","cvc5"}

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_ms, int) or self.timeout_ms <= 0:
            raise ValueError(
                f"SolverConfig.timeout_ms must be a positive integer number "
                f"of milliseconds, got {self.timeout_ms!r}"
            )
        if self.only is not None:
            if not self.only:
                # audit F7: an empty portfolio is a caller bug — escalating
                # to nobody is done by not calling escalate, and letting ()
                # through mislabels a configuration as a missing install.
                raise ValueError(
                    "SolverConfig.only=() names an empty portfolio; to run "
                    "without escalation, do not call escalate"
                )
            unknown = set(self.only) - {"z3", "cvc5"}
            if unknown:
                raise ValueError(
                    f"SolverConfig.only names unknown solvers {sorted(unknown)}; "
                    f"known: ['cvc5', 'z3']"
                )


@dataclass
class _Ledger:
    """The escalation-scoped, append-only record of solver invocations.

    ``stamps`` receives each invocation's fully-populated
    :class:`SolverStamp` at the dispatch site, immediately BEFORE the
    transport runs; ``spawns`` increments at the transport-entry boundary
    (:meth:`_Backend.run`). The two fields are updated from different code
    sites with no shared helper — deliberately anti-correlated, checked
    for agreement before any escalated verdict emits. Appends are
    irreversible: no code path pops, filters, rebuilds, or conditionally
    drops entries.
    """

    stamps: list[SolverStamp] = field(default_factory=list)
    spawns: int = 0


@dataclass(frozen=True)
class ObligationEscalation:
    """The escalation outcome for one obligation."""

    index: int
    outcome: str  # OB_DISCHARGED | OB_VIOLATED_WITNESS | OB_UNKNOWN
    detail: str
    invocations: tuple[SolverStamp, ...]
    witness: Witness | None
    notes: tuple[str, ...]
    # The labels of the backends whose DEFINITIVE answer (sat/unsat) this
    # outcome rests on — empty for an outcome no answer decided.
    # Deliberately NOT derivable from ``invocations``: a stamp records the
    # ASK, and a backend that was invoked and stamped but returned
    # failed/unknown/timeout contributed nothing to the outcome. The gap
    # between the two is exactly a degraded portfolio, and before this
    # field there was no quantity a consumer could read it from.
    answered_by: tuple[str, ...] = ()
    # NO `barred_on_slice` FIELD, DELIBERATELY, AND THIS IS WHERE IT WAS.
    # A predecessor recorded the scatter VERIFIED bar's per-obligation scope
    # here, computed in escalate()'s dispatch loop from the slice that was
    # actually emitted, and the bar trusted it. Two measured holes, both
    # absent from the whole-query bar that preceded it: an empty tuple is a
    # positive claim ("nothing barred on my slice") that nothing validated,
    # and `make_solver_verdict` did not bind its escalation to its `closed`
    # at all until the query pairing gate landed — so a scatter-free
    # escalation stamped against a scatter-bearing query went VERIFIED where
    # the whole-query bar went UNKNOWN. A record cannot
    # certify its own cleanliness; the scope's CONTENTS are derived at the bar
    # from the query instead — `stelling.verdict._bar_scope`, which re-slices
    # the decided obligations out of `closed` and so has neither exposure for
    # them. What a record still supplies is the bar's DOMAIN, through `index`
    # and `outcome` — the same two fields that discharge the obligation in the
    # first place, so a record cannot leave the bar's domain without also
    # giving up the discharge it needs for there to be a VERIFIED at all. See
    # `make_solver_verdict`'s docstring for the precondition that rests on.


@dataclass(frozen=True)
class Escalation:
    """All escalation outcomes for one propagated query, plus the ledger
    the invocations were recorded in as they happened.

    ``semantics`` records which semantics the propagation this escalation
    was produced from ran under — :func:`make_solver_verdict` refuses to
    stamp an escalation against a propagation of the other semantics in
    EITHER direction (the symmetric mispairing gate): obligation details
    and refusal reasons would be misattributed across the semantics
    boundary.

    ``query_sha256`` records WHICH QUERY :func:`escalate` was called on —
    :meth:`stelling.ir.ClosedJaxpr.content_hash` of the ``closed`` it
    received, taken at every one of this function's return sites.
    :func:`make_solver_verdict` refuses to stamp an escalation carrying
    work against a ``closed`` that does not reproduce it (the query
    pairing gate). It is the same trust model as ``semantics`` above and
    as the ``smt2_sha256``/``slice_sha256`` the invocation stamps carry:
    a hand-built record can hold any value, and this defends an honest
    caller against an accidentally mispaired assembly — the realistic
    mechanism being a CACHED escalation, which is one of the two uses
    :mod:`stelling.ir`'s own module docstring names ``content_hash`` for
    ("proof caching and the z3-vs-cvc5 *did both solvers see the same
    query* check"). Empty means "not recorded by this library", and the
    gate refuses that too when the escalation carries work: an absent
    hash is exactly the shape a stale cache entry from before this field
    existed has. **That sentence was false in one case until this
    build**, and the case is named because it is the one where the
    refusal matters most: BOTH legs run through :func:`_query_sha256`,
    which returns ``""`` when
    :meth:`stelling.ir.ClosedJaxpr.content_hash` RAISES, so an unhashable
    ``closed`` and an unrecorded escalation compared EQUAL and the gate
    passed. Measured on `e35de13`: the refusal then came from
    :class:`stelling.verdict.Stamp`'s own emptiness check one layer
    later, which is loud but is not this gate. The gate now refuses an
    empty hash on either leg. Note also that ``carries_work`` is a real
    exemption and not a formality: an escalation with no records, no
    notes, no spawns and no stamps bypasses this gate entirely, and that
    is harmless because it contributes nothing an assembly could
    misattribute — measured, such a pairing returns UNKNOWN off the
    propagation alone."""

    records: tuple[ObligationEscalation, ...]
    notes: tuple[str, ...] = ()
    ledger: _Ledger = field(default_factory=_Ledger)
    semantics: str = "real"
    query_sha256: str = ""

    @property
    def invocations(self) -> tuple[SolverStamp, ...]:
        return tuple(s for r in self.records for s in r.invocations)


# -- transports ---------------------------------------------------------------


@dataclass(frozen=True)
class _RawResult:
    answer: str  # "sat" | "unsat" | "unknown" | "timeout" | "failed" | "not-run"
    version: str = "unknown"
    values: tuple[tuple[str, str], ...] = ()  # (name, exact rational text)
    nonrational: bool = False
    detail: str = ""


@dataclass(frozen=True)
class _Backend:
    name: str  # stamp name: "z3" | "cvc5"
    flavor: str  # which script flavor to emit: "z3" | "cvc5"
    label: str  # human label for notes: "z3", "cvc5 (wheel)", "cvc5 (binary …)"
    transport: str
    transport_fn: Callable[[str, float], _RawResult] = field(compare=False)
    version_fn: Callable[[], str] = field(compare=False)
    _version_cache: list = field(default_factory=list, compare=False, repr=False)

    def version(self) -> str:
        """The backend's version, obtainable before any run and cached per
        backend, so the invocation stamp is fully populated at append time
        — nothing about the stamp waits for a result."""
        if not self._version_cache:
            try:
                v = str(self.version_fn())
            except Exception:  # noqa: BLE001 — a version probe never blocks a run
                v = "unknown"
            self._version_cache.append(v or "unknown")
        return self._version_cache[0]

    def run(self, ledger: _Ledger, script_text: str, wall_s: float) -> _RawResult:
        """The transport-entry boundary: counts the spawn as its first
        act, then calls the transport. Mechanically disjoint from the
        stamp-append site — no shared helper updates both records."""
        ledger.spawns += 1
        return self.transport_fn(script_text, wall_s)


def _wall_seconds(timeout_ms: int) -> float:
    """The wall-clock guard budget: the solver's own limit plus headroom so
    the emitted option can fire first, the guard second — but always
    bounded (a hung solver cannot hang the verdict)."""
    return timeout_ms / 1000.0 * 1.5 + 1.0


def _quote(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _z3_version() -> str:
    try:
        return str(_optional.require("z3").get_version_string()) or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_z3(script_text: str, wall_s: float) -> _RawResult:
    try:
        z3 = _optional.require("z3")
    except _optional.OptionalDependencyError as e:  # raced away since discovery
        return _RawResult(answer="not-run", detail=str(e))
    version = z3.get_version_string()
    ctx = z3.Context()  # fresh context: no symbol/option leakage across runs
    # TACTIC WORKAROUND for degree-80 factoring pathology. When the script
    # contains a rational-pow auxiliary variable (the `y^q = x^p` encoding),
    # the default z3 Solver() times out on the high-degree polynomial that
    # results (measured: d=80 from `x**(1/80)` on perfect-square bounds
    # timed out every run at 10s). The tactic chain below restores the z3
    # cross-check for these scripts (measured: 0.35-0.6s on d=80).
    if "(declare-const aux_" in script_text:
        tactic = z3.Then(
            z3.Tactic("simplify", ctx=ctx),
            z3.Tactic("solve-eqs", ctx=ctx),
            z3.With(z3.Tactic("factor", ctx=ctx), num_primes=4),
            z3.Tactic("purify-arith", ctx=ctx),
            z3.Tactic("tseitin-cnf", ctx=ctx),
            z3.Tactic("nlsat", ctx=ctx),
            ctx=ctx,
        )
        solver = tactic.solver()
    else:
        solver = z3.Solver(ctx=ctx)
    box: dict[str, object] = {}

    def work() -> None:
        try:
            solver.from_string(script_text)
            answer = str(solver.check())
            box["answer"] = answer
            if answer == "sat":
                model = solver.model()
                values: list[tuple[str, str]] = []
                nonrational = False
                for decl in model.decls():
                    if decl.arity() != 0:
                        continue
                    v = model[decl]
                    if v is None:
                        continue
                    if z3.is_rational_value(v):
                        values.append((decl.name(), str(v.as_fraction())))
                    elif z3.is_int_value(v):
                        values.append((decl.name(), str(v.as_long())))
                    elif z3.is_algebraic_value(v):
                        # an algebraic numeral (a root-obj like √2): a
                        # witness VALUE nothing can replay exactly — the
                        # one shape the nonrational policy exists for
                        nonrational = True
                    # everything else is a definition echo: z3 lists the
                    # script's define-funs (Real and Bool alike) among the
                    # model decls, VALUED AS EXPRESSIONS over the inputs
                    # (measured: `t8_0 -> x0_0 + x0_1`). Those are
                    # auxiliary definitions, not witness values, and must
                    # not poison replayability — declared inputs are
                    # always Real constants (rational or algebraic), and
                    # _screen_model ignores undeclared names anyway.
                    # Latent since the first z3 transport (every sat model
                    # with any define-fun was flagged); exposed by the
                    # first QF_LRA sat-with-witness path, where z3 is the
                    # portfolio primary (the array FACE case).
                box["values"] = tuple(sorted(values))
                box["nonrational"] = nonrational
            elif answer == "unknown":
                box["detail"] = str(solver.reason_unknown())
        except Exception as e:  # noqa: BLE001 — quoted, never raised (guard rule)
            box["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(wall_s)
    if thread.is_alive():
        ctx.interrupt()  # z3 releases the GIL and honors interrupts (measured)
        thread.join(5.0)
        return _RawResult(
            answer="timeout",
            version=version,
            detail=f"wall-clock guard fired at {wall_s:.1f}s; z3 interrupted",
        )
    if "error" in box:
        return _RawResult(answer="failed", version=version, detail=str(box["error"]))
    answer = str(box.get("answer", ""))
    detail = str(box.get("detail", ""))
    if answer == "unknown" and ("timeout" in detail or "canceled" in detail):
        return _RawResult(answer="timeout", version=version, detail=detail)
    if answer not in ("sat", "unsat", "unknown"):
        return _RawResult(
            answer="failed",
            version=version,
            detail=f"unrecognized z3 answer {answer!r}",
        )
    return _RawResult(
        answer=answer,
        version=version,
        values=tuple(box.get("values", ())),  # type: ignore[arg-type]
        nonrational=bool(box.get("nonrational", False)),
        detail=detail,
    )


def _cvc5_wheel_version() -> str:
    try:
        cvc5 = _optional.require("cvc5")
        version = cvc5.Solver(cvc5.TermManager()).getVersion()
        if isinstance(version, bytes):
            version = version.decode("utf-8", "replace")
        return str(version) or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _decode_child_stream(raw: bytes) -> str:
    """Decode one of the cvc5 driver child's streams: bytes in, records out.

    THE ONE TRANSLATION `text=True` WAS DOING FOR US, PUT BACK BY HAND, AND
    NOTHING ELSE. `capture_output=True, text=True` performs universal-newline
    decoding, which maps `\\r\\n` AND a bare `\\r` to `\\n`. The first is a real
    record boundary on a platform `README.md` names for both solver wheels and
    has to survive; the second is not a boundary anywhere `_cvc5_driver` runs,
    and promoting it to one is what let a stale driver forge this protocol's
    terminator (see `_run_cvc5_wheel` below).

    Undecodable bytes become U+FFFD rather than an exception, because the
    caller's alphabet check already refuses every character outside printable
    ASCII: a child that writes non-UTF-8 is a protocol violation this layer
    can state, not one it should raise out of the transport."""
    return raw.decode("utf-8", "replace").replace("\r\n", "\n")


def _run_cvc5_wheel(script_text: str, wall_s: float) -> _RawResult:
    version = _cvc5_wheel_version()
    argv = [sys.executable, "-m", "stelling._cvc5_driver"]
    try:
        # NO `text=`, NO `encoding=`, NO `universal_newlines=`: any of the
        # three switches universal-newline decoding back on and reopens the
        # `\r` hole described below without touching a line of the parser.
        # The child's stdin takes bytes for the same reason — there is no
        # text mode left to encode it.
        proc = subprocess.run(
            argv,
            input=script_text.encode("utf-8"),
            capture_output=True,
            timeout=wall_s,
        )
    except subprocess.TimeoutExpired:
        return _RawResult(
            answer="timeout",
            version=version,
            detail=f"wall-clock guard killed the cvc5 child at {wall_s:.1f}s "
            f"(cvc5's own tlimit did not preempt it)",
        )
    except OSError as e:
        return _RawResult(answer="not-run", version=version, detail=str(e))
    # ONE NOTION OF A LINE, and it is the driver's. This read used
    # `str.splitlines()` while the driver wrote records with `print` and
    # sanitised them with `replace("\n", " ")` — and `splitlines()` breaks on
    # TEN characters, not one (measured: U+000A U+000B U+000C U+000D U+001C
    # U+001D U+001E U+0085 U+2028 U+2029). A model value carrying any of the
    # other nine was ONE line to the writer and TWO to this reader, so the
    # payload could supply this parser's LAST line — a forged terminator —
    # while the child was truncated mid-model-walk. MEASURED end to end: real
    # cvc5, real driver, real SIGKILL, this parser reading the whole flushed
    # prefix (53311 bytes), last line `end 3800`, this function reporting
    # "terminator present" and, on exit 0, returning `sat` with 3800 values
    # harvested from a corpse. That defeats BOTH tells at once, which is the
    # one thing two tells exists to prevent.
    #
    # `split("\n")` is the writer's own boundary — `print`'s — so a record the
    # driver failed to sanitise stays ONE record here, and a payload holding
    # an `end <n>` is no longer this parser's last line: the terminator check
    # REFUSES it instead of reading it. That is why this narrows rather than
    # widening `splitlines()`'s set into a membership test on this side: this
    # side then has no second notion of a line to disagree with.
    #
    # The trailing empty element is `print`'s own newline on the last record,
    # not a record; exactly one is dropped, so a stdout that really does end
    # in a blank line still fails the terminator check.
    #
    # THE WRITER IS THE LOAD-BEARING HALF AND STILL IS; WHAT CHANGED IS THAT
    # THIS SIDE IS NO LONGER BLIND BESIDE IT. This read used to be
    # `capture_output=True, text=True`, so Python's universal-newline decoding
    # turned a `\r` into a real `\n` BEFORE this function saw the string
    # (MEASURED — the child wrote one record, both splitters saw two). No rule
    # on this side could see it, which is what settled the widen-the-writer /
    # narrow-the-reader question in the first place: `_cvc5_driver._token` and
    # `_tail` pass printable ASCII only, and that is where the boundary is
    # made or not made. THAT HALF HAS NOT MOVED and is not weakened by this.
    #
    # What moved is the fraction this side backstops when the WRITER is the
    # thing that is wrong — a driver out of step with this parser, i.e. a
    # PARTIAL UPGRADE, which is the case `_cvc5_driver`'s docstring promises
    # degrades to UNKNOWN. Three readers, same io layer, real children, real
    # bytes (`scratchpad/probe_cvc5_backstop.py`, parts B/C/D):
    #
    #   (a) `text=True`                             — what shipped before
    #   (b) `bytes.decode()`
    #   (c) `bytes.decode().replace("\r\n", "\n")`  — what is here now
    #
    #   case                             (a)      (b)      (c)
    #   healthy POSIX   `\n`             sat      sat      sat
    #   healthy Windows `\r\n`           sat      FAILED   sat
    #   stale `\r`, LF body              SAT (!)  failed   failed
    #   stale `\r`, CRLF body            SAT (!)  failed   failed
    #   stale `\x0b`                     failed   failed   failed
    #   separators refused (LF stale)    8 of 10  9 of 10  9 of 10
    #   child writes invalid UTF-8       RAISES   failed   failed
    #
    # (c) DOMINATES (a) on every case measured: identical answer AND identical
    # values on both healthy children, strictly stronger on both stale ones,
    # nine of the ten separators instead of eight, no platform coupling, and
    # `failed` where (a) raised an UNCAUGHT `UnicodeDecodeError` out of this
    # function. (b) buys the same ninth separator by refusing a healthy
    # Windows child OUTRIGHT, and `README.md` names Windows for both solver
    # wheels — that, and not behaviour on the stale children, is what rules
    # (b) out. (c)'s one measured cry-wolf case is a healthy child
    # reconfigured to BARE CR line endings, which no platform's `print`
    # default produces and which `_cvc5_driver` never sets.
    #
    # THE HOLE THIS CLOSES REACHED THE DISCHARGE DIRECTION. Under (a) a stale
    # child writing `opaque x1 j\rend 2\r`, with no terminator record of its
    # own anywhere, returned `sat` with a model — and returned `unsat` off the
    # identical corpse if that is what its `answer` line said (both MEASURED
    # at `9564728`, real child, real bytes). `sat` is the direction
    # `_require_valid_refutation` replays and can still refuse downstream;
    # `unsat` is the VERIFIED path and nothing downstream replays it.
    #
    # WHAT IS STILL NOT CLOSED HERE, said plainly rather than left to be
    # found. The nine is over SINGLE characters. `\n` is excluded by
    # construction — it is the protocol's own record boundary, so a writer
    # that leaves one inside a field has written two records and there is
    # nothing here to detect — and the two-character `\r\n` is that same fact
    # in a second spelling, since it is a genuine record boundary under this
    # reader exactly as it was under (a). Neither is a regression and neither
    # is an improvement; both are the writer's, and the writer escapes them.
    #
    # A byte outside the protocol's alphabet is a protocol violation, never
    # something to interpret.
    stdout = _decode_child_stream(proc.stdout)
    stderr = _decode_child_stream(proc.stderr)
    if any(not (" " <= c <= "~") for c in stdout if c != "\n"):
        return _RawResult(
            answer="failed",
            version=version,
            detail=f"cvc5 driver wrote outside the protocol's alphabet "
            f"(printable ASCII and newline); refusing to interpret it. "
            f"stdout: {_quote(stdout)!r}",
        )
    # A RECORD IS `text + "\n"`, because that is what `print` writes. A final
    # record with no newline is a record the child did not finish writing, and
    # this parser used to accept one: `…\nend 4` (the newline cut, nothing
    # else) read as a present terminator with a matching count, and on exit 0
    # became a definite answer. FOUND BY THE FUZZER ON THIS FIX, not reasoned
    # into it — 86 of 86 residual findings over 200k examples had exactly this
    # shape and no other. `splitlines()` accepted it identically, so this is
    # the same disagreement as above at the other end of the record: the writer
    # terminates, the reader did not require termination.
    lines = stdout.split("\n")
    terminated = len(lines) > 1 and lines[-1] == ""
    if terminated:
        lines.pop()  # `print`'s own newline on the last record, not a record
    answer = ""
    answers = 0
    values: list[tuple[str, str]] = []
    nonrational = False
    opaques = 0
    error = ""
    for line in lines:
        parts = line.split(maxsplit=2)
        if not parts:
            continue
        if parts[0] == "version" and len(parts) >= 2:
            version = parts[1]
        elif parts[0] == "answer" and len(parts) >= 2:
            answer = parts[1]
            answers += 1
        elif parts[0] == "value" and len(parts) == 3:
            values.append((parts[1], parts[2]))
        elif parts[0] == "opaque":
            nonrational = True
            opaques += 1
        elif parts[0] == "error":
            error = line[len("error "):]
    if error:
        return _RawResult(answer="failed", version=version, detail=_quote(error))
    # EXACTLY ONE `answer`. The driver emits one; a second one means the stream
    # is not the conversation this parser thinks it is. Without this, a stdout
    # carrying `answer sat`, a value, then `answer unsat` was read as unsat
    # while KEEPING the value harvested under the earlier sat — a model from
    # one answer attached to another (MEASURED at base and before this line:
    # `unsat` with 1 value). Input names are `x{k}` / `x{k}_{i}` by
    # construction (`obligation.py`), so no model line can begin with the token
    # `answer` and this cannot cry wolf on a healthy run.
    if answer not in ("sat", "unsat", "unknown") or answers != 1:
        return _RawResult(
            answer="failed",
            version=version,
            detail=f"cvc5 driver protocol violation; stdout: "
            f"{_quote(stdout)!r}, stderr: {_quote(stderr)!r}",
        )
    # THE CRASHED CHILD. The driver answers BEFORE it walks the model, so a
    # death inside cvc5's native `getValue` leaves `answer sat` on stdout with
    # no terminator. This is the wheel transport's form of the crashed run the
    # binary transport refuses at `_make_run_cvc5_binary` ("audit F4"); the
    # wheel never received it.
    #
    # WHY IT SURVIVED — and the boundary is the PIPE BUFFER, not the model
    # size. stdout to a pipe is block-buffered at `io.DEFAULT_BUFFER_SIZE`
    # (8192), so a killed child below that loses everything it wrote and the
    # parent correctly sees a protocol violation above. MEASURED, real child,
    # real SIGKILL, bisected: 493 value lines = 8186 bytes written = 0 bytes
    # through, caught as a protocol violation; 494 lines = 8203 written = 8202
    # through, `answer sat` present, and this function returned sat with 494
    # values from a dead process. Also measured at 4000 terms: 3572 lines
    # through, 3570 values harvested. A 2-value model leaves 0 lines through.
    #
    # But 8192 is the DEFAULT threshold, not a floor: it is ZERO when the child
    # is unbuffered. Under `PYTHONUNBUFFERED=1` (standard in Docker images and
    # CI) or any per-line flush, everything written before the death is
    # through — MEASURED, a 2-value model puts all 51 bytes of its stdout
    # through, `answer sat` among them. This function spawns with no `env=`,
    # so the child inherits whatever the ambient environment says. The earlier
    # reading that "small fixtures cannot reach it" was therefore an artifact
    # of one buffering regime, not a property of the defect.
    #
    # REACHABILITY is not exotic either way: `smt.py` emits one
    # `(declare-const … Real)` per input ELEMENT, so a single (32,32) array is
    # 1024 consts ≈ 17360 bytes of driver stdout, well over the buffer — and an
    # OOM kill is a SIGKILL that needs no cvc5 bug at all.
    #
    # WHAT THE HARM ACTUALLY IS (driven end-to-end, not reasoned): a truncated
    # model does NOT yield an unreproducible witness. `_require_valid_refutation`
    # routes every refutation through `witness_is_valid` (box membership AND
    # exact-rational violation), so a truncated model either raises
    # `EmissionInfidelityError` loudly (measured: the completed point satisfied
    # the predicate) or produces a witness that genuinely does reproduce
    # (measured: REFUTED, replay passed). The harm is that A CRASHED RUN
    # SILENTLY BECAME AN ACCEPTED ANSWER: the values the child never lived to
    # write are filled by `_complete_values` from the declared box, so the
    # reported witness is part fabrication and part corpse, and nothing in the
    # verdict says the solver died. (The fill itself is disclosed as a
    # don't-care note and does reach the render; the death was not disclosed
    # anywhere.)
    #
    # TWO TELLS, because each alone has a blind spot. The terminator is
    # positive evidence the driver ran to completion and is what catches a
    # death that somehow exits zero; `returncode` catches a death that somehow
    # emitted the terminator. Neither is derived from the other.
    #
    # THE TERMINATOR IS STRICT, because a token-prefix match let BOTH tells go
    # blind at once — precisely what "two tells" is supposed to prevent. Native
    # C++ output goes raw to fd 1 and bypasses Python's buffer, so a child that
    # writes `end of the resource limit` and exits 0 used to satisfy a
    # `parts[0] == "end"` test with no driver terminator at all (MEASURED: sat,
    # 1 value, at base AND under the first version of this guard). Requiring
    # the terminator to be the LAST line closes that, and with it two further
    # shapes that were accepted identically before and after that first version
    # (MEASURED at both): `end` followed by more values (sat, 3 values, two
    # written AFTER the terminator) and a truncated trailing line (sat, the
    # partial line silently dropped). Cry-wolf cost: none — `opaque x0 end` and
    # `value end 1/1` never false-triggered, because their first token is
    # `opaque`/`value`, and every healthy shape still passes.
    #
    # The count in `end <count>` is the driver's own tally of the model lines
    # it wrote, checked against what this parser actually parsed, so the
    # terminator asserts "the walk finished" rather than only "the driver
    # reached its last statement". A short walk is CONSTRUCTED, not observed
    # from cvc5 — stated as a constructed shape rather than a measured cvc5
    # bug — but it is the only reading of *complete* the word supports.
    #
    # DELIBERATE BEHAVIOUR CHANGE, beyond the crashed-sat case: a child with a
    # complete protocol that exits nonzero is now `failed` where it previously
    # returned its answer (MEASURED: a clean `answer sat` + terminator + exit 1
    # returned sat with its model at base, returns failed here). That direction
    # is deliberate and matches the binary transport's F4 policy — a nonzero
    # exit is not a transport this layer will discharge OR refute on, and the
    # cost is UNKNOWN, never a flipped verdict.
    complete = (
        terminated and bool(lines) and lines[-1] == f"end {len(values) + opaques}"
    )
    if not complete or proc.returncode != 0:
        why = "present" if complete else (
            "ABSENT (final record not newline-terminated)" if not terminated
            else "ABSENT"
        )
        return _RawResult(
            answer="failed",
            version=version,
            detail=f"cvc5 driver answered {answer!r} but the run is not "
            f"complete (exit {proc.returncode}; terminator "
            f"{why}); refusing to rely on a "
            f"crashed run. stderr: {_quote(stderr)!r}",
        )
    return _RawResult(
        answer=answer,
        version=version,
        values=tuple(sorted(values)),
        nonrational=nonrational,
        detail="",
    )


_BINARY_VERSION_RE = re.compile(r"^(?:This is )?cvc5(?: version)? (\S+)", re.MULTILINE)


def _cvc5_binary_version(path: str) -> str:
    try:
        proc = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=20
        )
        match = _BINARY_VERSION_RE.search(proc.stdout)
        return match.group(1) if match else "unknown (--version unparseable)"
    except Exception:  # noqa: BLE001
        return "unknown (--version unparseable)"


def _tokenize_sexpr(text: str) -> list[str]:
    return text.replace("(", " ( ").replace(")", " ) ").split()


def _parse_sexprs(tokens: list[str]) -> list[object]:
    forms: list[object] = []
    stack: list[list[object]] = []
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if not stack:
                raise ValueError("unbalanced ')'")
            done = stack.pop()
            (stack[-1] if stack else forms).append(done)
        else:
            (stack[-1] if stack else forms).append(tok)
    if stack:
        raise ValueError("unbalanced '('")
    return forms


def _sexpr_fraction(x: object) -> Fraction:
    if isinstance(x, str):
        return Fraction(x)  # integers, p/q, and exact SMT decimals
    if isinstance(x, list) and x:
        if x[0] == "-" and len(x) == 2:
            return -_sexpr_fraction(x[1])
        if x[0] == "/" and len(x) == 3:
            return _sexpr_fraction(x[1]) / _sexpr_fraction(x[2])
    raise ValueError(f"not a rational value term: {x!r}")


def _model_values_from_text(text: str) -> tuple[tuple[tuple[str, str], ...], bool]:
    """Parse ``(define-fun name () Sort value)`` forms out of solver stdout.
    Returns (values, saw_nonrational).

    Only arith-sorted (Real/Int) zero-arity entries are witness-value
    candidates; entries of other sorts (a solver echoing the script's Bool
    ``define-fun``\\ s, e.g.) are auxiliary definitions, not witness
    values, and must not poison replayability — declared inputs are always
    Real, and ``_screen_model`` ignores undeclared names. ``nonrational``
    is reserved for what it names: an arith value nothing can replay
    exactly (an algebraic number, an unparseable numeral)."""
    values: list[tuple[str, str]] = []
    nonrational = False
    try:
        forms = _parse_sexprs(_tokenize_sexpr(text))
    except ValueError:
        return (), True
    pending: list[object] = list(forms)
    while pending:
        form = pending.pop()
        if not isinstance(form, list) or not form:
            continue
        if form[0] == "define-fun" and len(form) >= 5 and form[2] == []:
            if form[3] in ("Real", "Int"):
                try:
                    values.append((str(form[1]), str(_sexpr_fraction(form[4]))))
                except (ValueError, ZeroDivisionError):
                    nonrational = True
        else:
            pending.extend(f for f in form if isinstance(f, list))
    return tuple(sorted(values)), nonrational


# The one tolerated post-answer shape on unsat/unknown: the script ends
# with (get-model), and a genuine cvc5 answers the model query after a
# non-sat check with an SMT-LIB error response (and exit status 1). The
# tolerance is exactly that shape — an error message mentioning the model
# query — and nothing wider; the accepted answer then carries a disclosure
# note quoting the noise verbatim (audit F4: an undisclosed crashed run
# must never become a VERIFIED).
_GET_MODEL_NOISE_RE = re.compile(
    r'^\s*\(error\s+"[^"]*model[^"]*"\.?\s*\)\s*$', re.IGNORECASE | re.DOTALL
)


def _make_run_cvc5_binary(path: str) -> Callable[[str, float], _RawResult]:
    def run(script_text: str, wall_s: float) -> _RawResult:
        version = _cvc5_binary_version(path)
        try:
            proc = subprocess.run(
                [path, "--lang", "smt2", "-"],
                input=script_text,
                capture_output=True,
                text=True,
                timeout=wall_s,
            )
        except subprocess.TimeoutExpired:
            return _RawResult(
                answer="timeout",
                version=version,
                detail=f"wall-clock guard killed {path} at {wall_s:.1f}s",
            )
        except OSError as e:
            return _RawResult(answer="not-run", version=version, detail=str(e))
        stdout_lines = proc.stdout.splitlines()
        lines = [ln.strip() for ln in stdout_lines if ln.strip()]
        if not lines or lines[0] not in ("sat", "unsat", "unknown"):
            return _RawResult(
                answer="failed",
                version=version,
                detail=f"unparseable solver output (exit {proc.returncode}); "
                f"stdout: {_quote(proc.stdout)!r}, "
                f"stderr: {_quote(proc.stderr)!r}",
            )
        answer = lines[0]
        first_idx = next(i for i, ln in enumerate(stdout_lines) if ln.strip())
        post_stdout = "\n".join(stdout_lines[first_idx + 1:])
        if answer == "sat":
            if proc.returncode != 0:
                return _RawResult(
                    answer="failed",
                    version=version,
                    detail=f"solver reported sat but exited {proc.returncode}; "
                    f"post-answer stdout: {_quote(post_stdout)!r}, "
                    f"stderr: {_quote(proc.stderr)!r}",
                )
            values, nonrational = _model_values_from_text(post_stdout)
            return _RawResult(
                answer=answer,
                version=version,
                values=values,
                nonrational=nonrational,
            )
        # unsat/unknown: the answer stands only on a clean exit with no
        # post-answer output, or on exactly the tolerated get-model error
        # shape (disclosed). Anything else — a nonzero exit, a segfault
        # banner, arbitrary trailing output — is a failed run: UNKNOWN with
        # the exit status and the noise quoted, never an undisclosed verdict.
        noise = "\n".join(part for part in (post_stdout, proc.stderr) if part.strip())
        if not noise.strip() and proc.returncode == 0:
            return _RawResult(answer=answer, version=version)
        if _GET_MODEL_NOISE_RE.match(noise.strip()):
            return _RawResult(
                answer=answer,
                version=version,
                detail=f"answer accepted with tolerated get-model noise "
                f"(exit {proc.returncode}): {_quote(noise)!r}",
            )
        return _RawResult(
            answer="failed",
            version=version,
            detail=f"solver answered {answer!r} but the run is not clean "
            f"(exit {proc.returncode}; post-answer output "
            f"{_quote(noise)!r}); refusing to rely on a crashed run",
        )

    return run


# -- backend discovery --------------------------------------------------------


def _backends_for(config: SolverConfig) -> tuple[tuple[_Backend, ...], tuple[str, ...]]:
    """The installed backends the config admits, plus the names it wanted
    but could not get (for degraded-portfolio notes).

    cvc5 precedence: an explicit ``STELLING_CVC5`` path outranks the wheel
    (that is what "point stelling at a different cvc5" means); otherwise
    the wheel outranks a PATH-discovered binary.
    """
    wanted = ("z3", "cvc5") if config.only is None else tuple(
        n for n in ("z3", "cvc5") if n in config.only
    )
    backends: list[_Backend] = []
    missing: list[str] = []
    for name in wanted:
        if name == "z3":
            if _optional.available("z3"):
                backends.append(
                    _Backend(
                        name="z3",
                        flavor="z3",
                        label="z3 (wheel)",
                        transport=TRANSPORT_Z3_WHEEL,
                        transport_fn=_run_z3,
                        version_fn=_z3_version,
                    )
                )
            else:
                missing.append("z3")
            continue
        env_path = os.environ.get("STELLING_CVC5")
        if env_path:
            backends.append(
                _Backend(
                    name="cvc5",
                    flavor="cvc5",
                    label=f"cvc5 (external binary {env_path})",
                    transport=TRANSPORT_CVC5_BINARY,
                    transport_fn=_make_run_cvc5_binary(env_path),
                    version_fn=lambda path=env_path: _cvc5_binary_version(path),
                )
            )
        elif _optional.available("cvc5"):
            backends.append(
                _Backend(
                    name="cvc5",
                    flavor="cvc5",
                    label="cvc5 (wheel)",
                    transport=TRANSPORT_CVC5_WHEEL,
                    transport_fn=_run_cvc5_wheel,
                    version_fn=_cvc5_wheel_version,
                )
            )
        else:
            path = _optional.cvc5_binary()
            if path:
                backends.append(
                    _Backend(
                        name="cvc5",
                        flavor="cvc5",
                        label=f"cvc5 (external binary {path})",
                        transport=TRANSPORT_CVC5_BINARY,
                        transport_fn=_make_run_cvc5_binary(path),
                        version_fn=lambda path=path: _cvc5_binary_version(path),
                    )
                )
            else:
                missing.append("cvc5")
    return tuple(backends), tuple(missing)


# -- portfolio dispatch for one obligation ------------------------------------


def _screen_model(
    sl: ObligationSlice,
    raw_values: tuple[tuple[str, str], ...],
) -> tuple[dict[str, Fraction] | None, str | None, tuple[str, ...]]:
    """Screen a transport's model against the slice's declarations.

    Returns ``(values, problem, disclosures)``: ``values`` maps DECLARED
    input names to exact Fractions (None when the model is unusable, with
    the reason in ``problem``); ``disclosures`` are facts an accepted model
    must still say out loud (undeclared names ignored, identical duplicate
    definitions collapsed). Conflicting duplicate definitions of a declared
    name make the model unusable — sort order must never pick a survivor
    (audit F3)."""
    declared = {inp.name for inp in sl.inputs}
    disclosures: list[str] = []
    undeclared = sorted({name for name, _ in raw_values if name not in declared})
    if undeclared:
        disclosures.append(
            f"model defined undeclared name(s) {', '.join(undeclared)}; "
            f"ignored (auxiliary definitions are not witness values)"
        )
    parsed: dict[str, list[Fraction]] = {}
    for name, text in raw_values:
        if name not in declared:
            continue
        try:
            fr = Fraction(text)
        except (ValueError, ZeroDivisionError):
            return (
                None,
                f"witness not independently replayable (model value for "
                f"{name} could not be parsed as an exact rational: "
                f"{_quote(text)!r})",
                tuple(disclosures),
            )
        parsed.setdefault(name, []).append(fr)
    conflicting = sorted(
        name for name, vs in parsed.items() if len(set(vs)) > 1
    )
    if conflicting:
        return (
            None,
            f"model unusable: conflicting duplicate definitions for "
            f"{', '.join(conflicting)} — refusing to let sort order pick a "
            f"survivor",
            tuple(disclosures),
        )
    for name, vs in parsed.items():
        if len(vs) > 1:
            disclosures.append(
                f"model defined {name} {len(vs)} times with the identical "
                f"value {vs[0]}; collapsed"
            )
    return ({name: vs[0] for name, vs in parsed.items()}, None, tuple(disclosures))


def _complete_values(
    sl: ObligationSlice,
    got: dict[str, Fraction],
    notes: list[str],
) -> dict[str, Fraction]:
    """Fill genuine per-element don't-cares from each input element's own
    declared box (disclosed per element; the replay still decides).
    Callers guarantee at least one declared input element was actually
    supplied (audit F2) — a model that supplies none is a transport
    failure, not a field of don't-cares."""
    values: dict[str, Fraction] = {}
    for inp in sl.inputs:
        if inp.name in got:
            values[inp.name] = got[inp.name]
            continue
        if inp.lo != float("-inf"):
            fill = Fraction(inp.lo)
        elif inp.hi != float("inf"):
            fill = Fraction(inp.hi)
        else:
            fill = Fraction(0)
        values[inp.name] = fill
        notes.append(
            f"assert #{sl.index}: the model omitted {inp.name} (don't-care); "
            f"completed with {fill} from its declared box before replay"
        )
    return values


def _require_valid_refutation(
    sl: ObligationSlice,
    values: dict[str, Fraction],
    *,
    solver_label: str,
    script_text: str,
) -> None:
    """The one gate deciding "this refutation is real", for both shapes
    (witnessed and constants-only): routes the whole claim through the
    single validator (:func:`stelling.obligation.witness_is_valid` — box
    membership AND exact-rational violation as one conjunction) and raises
    :exc:`EmissionInfidelityError` naming the failing conjunct (audit F1:
    an out-of-box or non-violating model must never mint a REFUTED)."""
    problem = witness_is_valid(sl, values)
    if problem is not None:
        # fraction_text, not str(): these are SOLVER MODEL values and
        # their terms are unbounded, so `str()` raises ValueError past
        # CPython's int -> str cap. `witness_is_valid` above already
        # renders the same values safely to build `problem` — leaving
        # bare `str()` here meant the alarm assembled its diagnosis and
        # then died one statement later trying to attach the values it
        # was about, which is the failure the safe rendering was added to
        # prevent, one line further on.
        raise EmissionInfidelityError(
            obligation_index=sl.index,
            solver=solver_label,
            values=tuple(
                (inp.name, fraction_text(values[inp.name]))
                for inp in sl.inputs
            ),
            script_text=script_text,
            detail=problem,
        )


def make_validated_witness(
    sl: ObligationSlice,
    values: dict[str, Fraction],
    produced_by: str,
    *,
    solver_label: str,
    script_text: str,
) -> Witness:
    """Validate then construct — the dispatch path's ONLY ``Witness``
    construction site. The REFUTED-with-witness outcome is built
    exclusively around this factory's return value, so no witness can
    exist that did not pass the single validator's conjunction."""
    _require_valid_refutation(
        sl, values, solver_label=solver_label, script_text=script_text
    )
    # `Witness.values` is DATA, not a message. Its contract is
    # `(input name, exact rational)` (`verdict.Witness`) and
    # `reproduce._point` parses it back with `Fraction()` to re-execute
    # the harness at the point. So the bit-length fallback that makes a
    # MESSAGE safe is the wrong instrument here: it would mint a REFUTED
    # whose witness is not the value the solver produced and cannot be
    # re-executed, moving a contained failure onto another module's public
    # surface as a broken contract. Applying a message renderer to a data
    # field is the category error.
    #
    # Fail closed instead — `smt._renderable`'s posture: detect by
    # attempting the conversion, and when it cannot be done exactly,
    # decline. `ReplayDeclined` is the channel that already means "no
    # usable witness here"; the caller's handler turns it into UNKNOWN
    # with the reason quoted and does not raise.
    #
    # NOT a remote hazard. Measured through `check()` on a traced harness
    # — `x ** 0.5` nested eight deep on `[1, 1e300]` with the threshold
    # under the box maximum — a REFUTED witness carries a model term of
    # **4091 digits against the 4300-digit cap: 95% of it, with no
    # decline yet observed**. An earlier note here claimed "16 decimal
    # digits", which measured the wrong quantity (a rendered `n/d` string
    # from harnesses that never nested `pow`) and made the margin look
    # like 99.6% when it is 5%. So this arm is close to live: it costs
    # nothing measurable today, and the reason to keep it is not that the
    # input is unreachable but that it is nearly reached.
    rendered = []
    for inp in sl.inputs:
        v = values[inp.name]
        try:
            rendered.append((inp.name, str(v)))
        except ValueError as e:
            raise ReplayDeclined(
                f"the model value for {inp.name} cannot be recorded "
                f"exactly: it is {fraction_text(v)} and CPython refuses to "
                f"render it ({e}). A witness carries the exact rational so "
                f"the reproducer can re-execute the harness at that point; "
                f"a summarised value would name a different point, so no "
                f"witness is issued"
            ) from e
    return Witness(
        obligation_index=sl.index,
        values=tuple(rendered),
        produced_by=produced_by,
        replay=_REPLAY_SENTENCE,
        # which element(s) of an ARRAY assert operand are false at the
        # point — the same replay computation the validator's violation
        # conjunct just ran, re-read for naming (a scalar operand names
        # nothing: the predicate itself is the violation, as before)
        violating_elements=violating_elements(sl, values),
    )


def _absences(
    config: SolverConfig,
    ordered: tuple[_Backend, ...],
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    """Why each portfolio member is not among the backends that ran, one
    phrase per absent member, in a fixed order.

    Three disjoint causes covering ``{z3, cvc5}``: it ran, it is wanted but
    not installed (``missing``), or the caller excluded it with
    ``SolverConfig.only``. The predecessor said "not installed" for every
    absence, which is FALSE for the ``only=`` case — a configured
    restriction rendered as a missing dependency sends a reader to install
    something they already have."""
    ran = {b.name for b in ordered}
    out = []
    for name in ("z3", "cvc5"):
        if name in ran:
            continue
        if name in missing:
            out.append(f"{name} is not installed")
        else:
            out.append(f"{name} was excluded by SolverConfig.only={config.only!r}")
    return tuple(out)


def _dispatch_obligation(
    sl: ObligationSlice,
    config: SolverConfig,
    backends: tuple[_Backend, ...],
    missing: tuple[str, ...],
    ledger: _Ledger,
    relational_assumes: tuple[ir.JaxprEqn, ...] = (),
) -> tuple[ObligationEscalation, int]:
    ordered = tuple(
        sorted(
            backends,
            key=lambda b: (b.name != "cvc5") if sl.fragment == "QF_NRA" else (b.name != "z3"),
        )
    )
    scripts: dict[str, Script] = {}
    for backend in ordered:
        if backend.flavor not in scripts:
            scripts[backend.flavor] = emit(
                sl, backend.flavor, config.timeout_ms,
                relational_assumes=relational_assumes,
            )
    # The count of relational assumes ACTUALLY emitted for this obligation's
    # script. Identical across flavors (the logical content is the same; only
    # the option block differs), so we take it from any script.
    n_emitted = next(iter(scripts.values())).relational_assumes_emitted if scripts else 0
    wall_s = _wall_seconds(config.timeout_ms)
    notes: list[str] = []
    if relational_assumes:
        notes.append(
            f"assert #{sl.index}: {len(relational_assumes)} relational "
            f"assume(s) forwarded to solver as axiom(s)"
        )
    absences = _absences(config, ordered, missing)
    if len(ordered) == 1:
        notes.append(
            f"assert #{sl.index}: portfolio degraded — only {ordered[0].label} "
            f"ran ({'; '.join(absences)})"
        )

    ledger_start = len(ledger.stamps)

    def invocations() -> tuple[SolverStamp, ...]:
        # this obligation's slice of the append-only ledger — a read, never
        # a rebuild; the ledger itself is untouched
        return tuple(ledger.stamps[ledger_start:])

    runs: list[tuple[_Backend, Script, _RawResult, SolverStamp]] = []
    for position, backend in enumerate(ordered):
        script = scripts[backend.flavor]
        role = "primary" if position == 0 else "secondary"
        if len(ordered) == 1:
            role = "alone (degraded portfolio)"
        # The stamp is the record of the ask, fully populated from the
        # invocation itself and appended BEFORE the transport runs: no
        # result exists yet, so no result can ever be narrated into it,
        # and a failure after this point leaves the stamp standing with
        # the failure disclosed in the notes (audit F5, structuralized).
        # The reason carries invocation context only — never the answer.
        stamp = SolverStamp(
            invoked=True,
            reason=(
                f"{sl.fragment} portfolio {role} on assert #{sl.index}"
            ),
            name=backend.name,
            version=backend.version(),
            transport=backend.transport,
            options=script.stamp_options(),
        )
        ledger.stamps.append(stamp)
        started = time.monotonic()
        raw = backend.run(ledger, script.text, wall_s)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        runs.append((backend, script, raw, stamp))
        if raw.answer == "not-run":
            notes.append(
                f"assert #{sl.index}: {backend.label} was invoked but its "
                f"transport failed before the solver could run "
                f"({_quote(raw.detail)})"
            )
            continue
        # outcome and latency are recorded here, additively, after the
        # run — never in the stamp, which predates the result
        notes.append(
            f"assert #{sl.index}: {backend.label} answered {raw.answer} "
            f"in {elapsed_ms}ms"
        )
        if raw.answer in ("unknown", "timeout", "failed"):
            notes.append(
                f"assert #{sl.index}: {backend.label} returned {raw.answer}"
                + (f" ({_quote(raw.detail)})" if raw.detail else "")
            )
        elif raw.detail:
            # a definitive answer accepted with tolerated noise (e.g. the
            # get-model error after unsat): the disclosure rides into the
            # notes — a verdict must never rest on it silently (audit F4)
            notes.append(f"assert #{sl.index}: {backend.label}: {raw.detail}")

    answers = {raw.answer for _, _, raw, _ in runs}
    # WHO ANSWERED, which is not who was ASKED. The stamps record the ask —
    # that is their whole contract, appended before the transport runs so no
    # result can ever be narrated into one — and until this line "how many
    # backends actually answered" was not a quantity any consumer could
    # read. The redundancy a verdict rests on is a property of the ANSWERS,
    # so it is derived here, once, from the runs.
    answered = tuple(
        b.label for b, _, raw, _ in runs if raw.answer in ("sat", "unsat")
    )

    def degraded_notes(*, universal: bool) -> tuple[str, ...]:
        """The note(s) a DECIDED obligation carries when its outcome rests
        on fewer than :data:`PORTFOLIO_SIZE` independent answers — ``()``
        when it rests on the full portfolio.

        The pre-invocation note above fires on an absent BACKEND; this one
        fires on an absent ANSWER, which is the wider condition and the one
        that was silent: a backend that is installed, invoked, and stamped,
        but whose transport failed or whose parser refused the script,
        leaves the stamp saying two and the decision resting on one.

        ``universal`` marks the direction with no downstream backstop. A
        ``sat`` becomes REFUTED only through an independent exact-rational
        replay of the model, so a lost backend there costs a cross-check
        that another mechanism still performs. An ``unsat`` is a universal
        claim over the whole declared box: there is no point to replay and
        nothing re-derives it, so the second backend IS the redundancy."""
        if len(answered) >= PORTFOLIO_SIZE:
            return ()
        gaps = [
            f"{b.label} was invoked and returned {raw.answer}"
            for b, _, raw, _ in runs
            if raw.answer not in ("sat", "unsat")
        ]
        gaps.extend(absences)
        out = [
            f"assert #{sl.index}: portfolio degraded — this outcome rests on "
            f"{len(answered)} of the {PORTFOLIO_SIZE} backends the portfolio "
            f"is designed around ({' and '.join(answered)} answered; "
            f"{'; '.join(gaps)})"
        ]
        if universal:
            out.append(
                f"assert #{sl.index}: and this is the direction with no "
                f"backstop: a discharge is a universal claim over the whole "
                f"declared box, so nothing downstream re-derives it the way "
                f"exact-rational replay re-derives a witness. The second "
                f"backend was the only independent check on this obligation "
                f"and it did not answer"
            )
        return tuple(out)

    def degraded_clause(*, universal: bool) -> str:
        """The same fact, on the obligation's own detail line — where a
        reader looks first, and where a REFUTED/VERIFIED is read one
        obligation at a time."""
        if len(answered) >= PORTFOLIO_SIZE:
            return ""
        return (
            f" [PORTFOLIO DEGRADED: {len(answered)} of {PORTFOLIO_SIZE} "
            f"backends answered" + ("; a discharge has no replay backstop]"
                                    if universal else "]")
        )

    if "sat" in answers and "unsat" in answers:
        raise SolverDisagreement(
            obligation_index=sl.index,
            verdicts=tuple((b.label, raw.answer) for b, _, raw, _ in runs),
            options=tuple(
                (b.label, script.stamp_options()) for b, script, _, _ in runs
            ),
            scripts=tuple(
                (flavor, script.text) for flavor, script in sorted(scripts.items())
            ),
        )

    if "unsat" in answers:
        agreed = [b.label for b, _, raw, _ in runs if raw.answer == "unsat"]
        return (ObligationEscalation(
            index=sl.index,
            outcome=OB_DISCHARGED,
            detail=(
                f"discharged by solver escalation ({sl.fragment}): the box "
                f"with the negated predicate is unsat per {' and '.join(agreed)}"
                + degraded_clause(universal=True)
            ),
            invocations=invocations(),
            witness=None,
            notes=tuple(notes) + degraded_notes(universal=True),
            answered_by=answered,
        ), n_emitted)

    if "sat" in answers:
        sat_problems: list[str] = []
        for backend, script, raw, stamp in runs:
            if raw.answer != "sat":
                continue
            got, problem, disclosures = _screen_model(sl, raw.values)
            for d in disclosures:
                notes.append(f"assert #{sl.index}: {backend.label}: {d}")
            if got is None:
                sat_problems.append(f"{backend.label}: {problem}")
                notes.append(
                    f"assert #{sl.index}: {backend.label} reported sat; "
                    f"{problem} — degraded to UNKNOWN"
                )
                continue
            if raw.nonrational:
                sat_problems.append(
                    f"{backend.label}: witness not independently replayable "
                    f"(model contains a non-rational value)"
                )
                notes.append(
                    f"assert #{sl.index}: {backend.label} reported sat; "
                    f"witness not independently replayable (model contains a "
                    f"non-rational value) — by policy this stays UNKNOWN"
                )
                continue
            if sl.inputs and not got:
                # audit F2: the slice has inputs and the model values NONE
                # of them — a transport/model-extraction failure, not a
                # field of per-variable don't-cares. Never completed, never
                # a crash: UNKNOWN with the reason quoted.
                sat_problems.append(f"{backend.label}: {NO_USABLE_MODEL}")
                notes.append(
                    f"assert #{sl.index}: {backend.label}: {NO_USABLE_MODEL} "
                    f"(transport/model-extraction failure) — degraded to "
                    f"UNKNOWN"
                )
                continue
            values = _complete_values(sl, got, notes)
            try:
                if not sl.inputs:
                    # audit F5: a constants-only obligation has no witness
                    # values to render; the SAME validator decides the
                    # refutation is real (membership vacuously true;
                    # violation via the empty-environment replay of the
                    # closed formula) — an honest REFUTED, no fabricated
                    # witness.
                    _require_valid_refutation(
                        sl, values, solver_label=backend.label,
                        script_text=script.text,
                    )
                    return (ObligationEscalation(
                        index=sl.index,
                        outcome=OB_VIOLATED_CONSTANT,
                        detail=(
                            f"constant refutation: the obligation has no "
                            f"declared inputs and its predicate is definitely "
                            f"false — {_REPLAY_SENTENCE}; {backend.label} "
                            f"({sl.fragment}) answered sat in agreement"
                            + degraded_clause(universal=False)
                        ),
                        invocations=invocations(),
                        witness=None,
                        notes=tuple(notes) + degraded_notes(universal=False),
                        answered_by=answered,
                    ), n_emitted)
                witness = make_validated_witness(
                    sl,
                    values,
                    produced_by=(
                        f"{backend.name} {stamp.version} ({backend.transport})"
                    ),
                    solver_label=backend.label,
                    script_text=script.text,
                )
            except ReplayDeclined as declined:
                # NOT an emission-infidelity finding, and the distinction
                # is the whole of audit 0.2.0 S3: the replay is refusing a
                # point whose exact value is not rational, which says
                # nothing about whether the script meant the obligation.
                # Same posture, same words, as a model carrying a
                # non-rational value a few lines above — the witness is
                # not independently replayable, so it does not become a
                # REFUTED and it does not raise.
                #
                # PREEMPTED BY THE TWO INSTALLED WHEELS ON THE `pow` ROW,
                # and reachable through any transport that is not them.
                # For a rational-`pow` slice the model carries the `aux`
                # constant, and under `aux >= 0 ∧ aux^q = x^p ∧ x >= 0` we
                # have `aux = x^(p/q)` exactly — so `aux` is an algebraic
                # numeral precisely when the exact value is irrational,
                # which is the one case `_exact_rational_power` refuses.
                # Both wheels flag such a value as `nonrational` (z3:
                # `is_algebraic_value`; cvc5: the driver's `opaque`
                # record), so the `raw.nonrational` branch above catches
                # it first. Measured on those two: every `nonrational=False`
                # model on this row had a rational `aux`, where the exact
                # replay succeeds.
                #
                # THAT IS A STATEMENT ABOUT THE TWO WHEELS ONLY. The
                # external cvc5-BINARY transport is not installed here and
                # its model parser (`_model_values_from_text`) has never
                # been driven by a real algebraic model; a transport that
                # reports a rational-looking model without flagging it
                # lands here, and the `pow` row's replay declines on the
                # exact value. `tests/test_pow_audit_findings.py` drives
                # exactly that shape (a fake binary answering `sat` with
                # `x0 = 2` on `x**0.5 <= 1.0`) and pins the outcome: an
                # UNKNOWN quoting the replay's own reason — never a
                # REFUTED, never a fabricated witness, never a raise.
                sat_problems.append(
                    f"{backend.label}: witness not independently replayable "
                    f"({declined})"
                )
                notes.append(
                    f"assert #{sl.index}: {backend.label} reported sat; "
                    f"witness not independently replayable ({declined}) — by "
                    f"policy this stays UNKNOWN"
                )
                continue
            elements = ""
            if witness.violating_elements:
                # the array assert is a universal elementwise claim; the
                # detail names which element(s) the witness falsifies
                # (flat C-order indices into the assert operand)
                elements = (
                    "; violating element(s) of the assert operand: "
                    + ", ".join(str(i) for i in witness.violating_elements)
                )
            return (ObligationEscalation(
                index=sl.index,
                outcome=OB_VIOLATED_WITNESS,
                detail=(
                    f"violated at a concrete witness found by {backend.label} "
                    f"({sl.fragment}); {_REPLAY_SENTENCE}{elements}"
                    + degraded_clause(universal=False)
                ),
                invocations=invocations(),
                witness=witness,
                notes=tuple(notes) + degraded_notes(universal=False),
                answered_by=answered,
            ), n_emitted)
        detail_tail = (
            "; ".join(sat_problems)
            if sat_problems
            else "witness not independently replayable"
        )
        return (ObligationEscalation(
            index=sl.index,
            outcome=OB_UNKNOWN,
            detail=f"solver reported sat; {detail_tail} — UNKNOWN by policy",
            invocations=invocations(),
            witness=None,
            notes=tuple(notes),
        ), n_emitted)

    reasons = (
        "; ".join(
            f"{b.label}: {raw.answer}" for b, _, raw, _ in runs if raw.answer != "not-run"
        )
        or "every invocation's transport failed before its solver could run"
    )
    return (ObligationEscalation(
        index=sl.index,
        outcome=OB_UNKNOWN,
        detail=f"solver escalation did not decide ({reasons}); a timeout is never a VERIFIED",
        invocations=invocations(),
        witness=None,
        notes=tuple(notes),
    ), n_emitted)


# -- escalation over a propagated query ---------------------------------------


def _query_sha256(closed) -> str:
    """:meth:`stelling.ir.ClosedJaxpr.content_hash` of ``closed``, or ``""``
    if it cannot be taken.

    ONE DERIVATION, TWO READERS, and that is the point: :func:`escalate`
    records it on the :class:`Escalation` and :func:`make_solver_verdict`
    recomputes it from the ``closed`` it is handed, so the two are compared
    rather than copied. The hash is the IR's own semantic content hash —
    stable across processes, insensitive to ``source_info``/``debug_info``,
    and named in :mod:`stelling.ir`'s module docstring as existing for
    caching and for the "did both solvers see the same query" check. This
    is that check, one layer up.

    NEVER RAISES, and the empty string is not a pass: it is the value the
    pairing gate REFUSES when the escalation carries work, so a ``closed``
    whose hash cannot be taken fails loudly at assembly rather than
    silently pairing with anything. **The gate has to refuse it
    EXPLICITLY, and until this build it did not.** Both legs come from
    here, so an unhashable ``closed`` and an escalation that recorded
    nothing both produce ``""`` and an equality test passes them:
    measured on `e35de13`, that assembly reached
    :class:`stelling.verdict.Stamp` and was refused there
    ("stamp field 'query_content_hash' is empty") rather than at the
    gate. Two absences are not a match, and the gate now says so."""
    try:
        return str(closed.content_hash())
    except Exception:  # noqa: BLE001 — an unhashable query is refused, not excused
        return ""


def escalate(
    closed: ir.ClosedJaxpr,
    propagation: Propagation,
    config: SolverConfig,
) -> Escalation:
    """Escalate exactly the obligations interval propagation left unknown.

    Returns per-obligation records with every invocation stamped. Declines
    and solver failures degrade to UNKNOWN with quoted reasons; only
    :exc:`SolverDisagreement` and :exc:`EmissionInfidelityError` raise.

    A propagation that ran under ``semantics="ieee"`` declines escalation
    wholly (:data:`IEEE_SEMANTICS_REFUSAL`): the SMT backends emit over
    the reals, and an answer there proves the ℝ obligation, not the
    binary64 one an ieee verdict would claim. Zero invocations; the
    empty ledger is the derived record of absence.

    A propagation that CONSTRAINED any assume declines escalation wholly
    (:data:`CONSTRAINED_ASSUME_REFUSAL` — the v1 refusal): the emitted
    problem would not respect the assumed precondition, and a sat witness
    outside the assumed region would falsely refute the conditional claim.
    Inert assumes (``coverage.constrained == 0``) escalate exactly as
    before — a drop over-approximates, so emission over the declared box
    remains faithful to the propagated semantics.
    """
    # WHICH QUERY THIS ESCALATION IS ABOUT, recorded once, at the top, and
    # attached to every return below — including the ones that do no work.
    # A return site that forgot it would be an escalation the pairing gate
    # cannot check, so the derivation is hoisted rather than repeated:
    # `tests/test_verified_bar.py::test_every_escalate_return_site_records_the_query`
    # asserts every path out of this function carries it.
    query_sha256 = _query_sha256(closed)
    unknown = [o for o in propagation.obligations if o.status == "unknown"]
    if not unknown:
        return Escalation(
            records=(), notes=(), semantics=propagation.semantics,
            query_sha256=query_sha256,
        )
    if propagation.semantics == "ieee":
        # guard 2, mechanism 1: an ieee-mode propagation refuses solver
        # escalation wholly — the refusal is semantic and holds whether
        # or not anything is installed, so it is checked before backend
        # discovery. No invocation occurs; the fresh (empty) ledger IS
        # the record of absence (derived, never written).
        return Escalation(
            records=tuple(
                ObligationEscalation(
                    index=o.index,
                    outcome=OB_UNKNOWN,
                    detail=f"escalation declined: {IEEE_SEMANTICS_REFUSAL}",
                    invocations=(),
                    witness=None,
                    notes=(),
                )
                for o in unknown
            ),
            notes=(IEEE_SEMANTICS_REFUSAL,),
            semantics=propagation.semantics,
            query_sha256=query_sha256,
        )
    if propagation.coverage.constrained:
        # checked before solver availability: the refusal is semantic and
        # holds whether or not anything is installed, so the disclosed
        # reason is the constrained one, never the install hint. No
        # invocation occurs; the fresh (empty) ledger IS the record of
        # absence, exactly as for the other pre-invocation declines.
        return Escalation(
            records=tuple(
                ObligationEscalation(
                    index=o.index,
                    outcome=OB_UNKNOWN,
                    detail=f"escalation declined: {CONSTRAINED_ASSUME_REFUSAL}",
                    invocations=(),
                    witness=None,
                    notes=(),
                )
                for o in unknown
            ),
            notes=(CONSTRAINED_ASSUME_REFUSAL,),
            semantics=propagation.semantics,
            query_sha256=query_sha256,
        )
    backends, missing = _backends_for(config)
    if not backends:
        # backends filtered out before any invocation never reach the
        # ledger's append site; the empty ledger IS the record of absence
        return Escalation(
            records=tuple(
                ObligationEscalation(
                    index=o.index,
                    outcome=OB_UNKNOWN,
                    detail=f"{o.detail}; escalation: no solver available",
                    invocations=(),
                    witness=None,
                    notes=(),
                )
                for o in unknown
            ),
            notes=(INSTALL_HINT,),
            semantics=propagation.semantics,
            query_sha256=query_sha256,
        )
    env = interval_env(closed)
    ledger = _Ledger()
    records: list[tuple[ObligationEscalation, int]] = []
    for item in slice_unknown_obligations(closed, propagation, env):
        if isinstance(item, DeclinedObligation):
            records.append((
                ObligationEscalation(
                    index=item.index,
                    outcome=OB_UNKNOWN,
                    detail=f"escalation declined: {item.reason}",
                    invocations=(),
                    witness=None,
                    notes=(f"assert #{item.index}: escalation declined — {item.reason}",),
                ),
                0,
            ))
            continue
        ledger_start = len(ledger.stamps)
        n_emitted = 0
        try:
            record, n_emitted = _dispatch_obligation(
                item, config, backends, missing, ledger,
                relational_assumes=propagation.relational_assumes,
            )
        except (SolverDisagreement, EmissionInfidelityError):
            raise  # loud by design
        except Exception as e:  # noqa: BLE001 — guard rule: degrade, quoted
            # defensive: a failure on a validated slice is a bug, but
            # mid-analysis the guard rule still applies — UNKNOWN, quoted.
            # Invocations recorded before the error are in the append-only
            # ledger and still ride into the record (audit F5,
            # structuralized: the stamps predate the error and cannot be
            # dropped by it).
            reason = f"escalation attempted; internal error: {type(e).__name__}: {e}"
            record = ObligationEscalation(
                index=item.index,
                outcome=OB_UNKNOWN,
                detail=reason,
                invocations=tuple(ledger.stamps[ledger_start:]),
                witness=None,
                notes=(f"assert #{item.index}: {reason}",),
            )
        records.append((record, n_emitted))
    if propagation.assume_dropped:
        # F7's no-op half, solver side, and it must be ONE-SIDED. Declining
        # escalation outright was the first attempt and it was wrong: it
        # suppressed unsat too, and a relational assume that stays inert in
        # constrain mode is DOCUMENTED as escalating normally precisely
        # because the drop over-approximates and unsat over a superset still
        # implies unsat over the subset. The suite caught it.
        #
        # So only VIOLATIONS are withheld. A discharge over a superset implies
        # a discharge over the intended set; a witness over a superset may lie
        # entirely outside the precondition, which is the measured defect.
        #
        # PER-OBLIGATION GRANULARITY: when ALL relational assumes were
        # ACTUALLY EMITTED for a specific obligation's script (n_emitted ==
        # len(relational_assumes)), the solver ran WITH the full constraint
        # set. Its witness satisfies the assume by construction — it's a
        # genuine violation, not an artifact of the wider domain. Only
        # un-withhold when ALL were emitted; if any were skipped (operands
        # not in the backward cone), the solver ran over a wider domain and
        # the witness might violate the missing assume.
        n_total = len(propagation.relational_assumes)
        records = [
            (r, ne) if (
                r.outcome not in (OB_VIOLATED_WITNESS, OB_VIOLATED_CONSTANT)
                or (n_total > 0 and ne == n_total)
            )
            else (ObligationEscalation(
                index=r.index,
                outcome=OB_UNKNOWN,
                detail=f"violation WITHHELD from REFUTED: {DROPPED_ASSUME_REFUSAL}",
                invocations=r.invocations,
                witness=None,
                notes=r.notes + (DROPPED_ASSUME_REFUSAL,),
            ), ne)
            for r, ne in records
        ]
    return Escalation(
        records=tuple(r for r, _ in records), notes=(), ledger=ledger,
        semantics=propagation.semantics, query_sha256=query_sha256,
    )


# -- solver-assisted verdict assembly -----------------------------------------

_DEVICE_CLASS_DEFAULT = (
    "none: no device execution; any witness replay is exact rational "
    "arithmetic in pure Python (device-independent)"
)


def _nonvacuity_summary(checks: tuple[ObligationReport, ...]) -> str:
    # mirrors make_verdict's wording exactly; the no-solver path must stay
    # byte-identical, so the logic is replicated rather than refactored.
    if not checks:
        return "UNCHECKED — no membership conditions declared"
    if all(c.status == "discharged" for c in checks):
        return (
            f"checked — {len(checks)} membership condition(s) definitely true "
            f"(the declared set contains the stated point)"
        )
    if any(c.status == "violated-over-set" for c in checks):
        return (
            "FAILED — a membership condition is definitely false: the stated "
            "point is NOT in the declared set (harness defect, not a box fact)"
        )
    return "undecided — a membership condition could not be decided"


class _UnreadableBarDomain:
    """The sentinel `_bar_domain` returns when it cannot read the escalation.

    Truthy, so the bar branch is ENTERED, and not a mapping, so
    `verdict._bar_scope`'s `dict(decided)` raises into its own `except` and
    falls back to the whole-query set. An empty dict would have been the
    silencing answer — the bar branch is guarded on `decided` being non-empty,
    because an empty domain legitimately means "no solver decided anything" —
    so an unreadable escalation must not be spelled the same way as an honest
    empty one.

    **THE TRUTHINESS IS THE MECHANISM, AND FOR A WHILE NOTHING DROVE IT.** The
    outer `except` this sentinel comes out of was reached by no test in the
    repo — the three `invocations`-shape probes exercise only the INNER
    `except TypeError` — so `__bool__` returning `False` instead of `True`
    turned an unreadable escalation's UNKNOWN into VERIFIED, with no withheld
    note and the full suite green. Both halves are now driven by
    `tests/test_verified_bar.py::test_an_UNREADABLE_domain_widens_the_bar_and_the_sentinel_is_why`.

    **AND IT DOES NOT COVER EVERY WAY THE DOMAIN CAN COME BACK EMPTY**, which
    the flat claim "widens rather than silencing" read as if it did. A
    `records` that can be iterated ONCE is READABLE; it is just readable once,
    and while the domain was built several passes into `make_solver_verdict` an
    earlier pass had already consumed it, so `_bar_domain` returned an
    honest-empty `{}` and the bar was skipped — a silencing path that never
    reached this sentinel. That is closed by ORDER (the domain is read on the
    first pass) rather than by anything here; see the comment at the read
    site."""

    __slots__ = ()

    def __bool__(self) -> bool:
        return True


_BAR_DOMAIN_UNREADABLE = _UnreadableBarDomain()


def _bar_domain(escalation) -> dict[int, tuple[SolverStamp, ...]] | object:
    """THE BAR'S DOMAIN: which obligations the solver decided, and the
    invocation stamps their records carry. ONE PLACE, and that is the point.

    Reads exactly TWO fields off a record to decide membership — ``outcome``
    and ``index`` — plus ``invocations``, which is carried across for
    `verdict._evidence_is_about` and can only ever fail to lift the bar.
    Nothing else about a record may enter this decision: a record that could
    certify its own cleanliness is the deleted `barred_on_slice` field under
    another name, and the defect is not confined to a field that NAMES a
    barred primitive. Measured, on `eb1ff86` (where this read was four lines
    in `make_solver_verdict`) and again here: adding `audit_token: str = ""`
    to `ObligationEscalation` and `and r.audit_token != "clean"` to this
    filter gives UNKNOWN at `''` and at `'a value no honest record carries'`
    and VERIFIED at `'clean'`, and at `eb1ff86` the full suite stayed green —
    the field-probe test that was meant to catch it moves each field to two
    values of its declared TYPE, which EXHAUSTS `bool` and merely SAMPLES
    `str`.

    So the channel is pinned by construction rather than by probing values:
    `tests/test_verified_bar.py` calls this with a record object that HAS no
    attribute but `index`, `outcome` and `invocations`, so a conjunct on any
    new field of any type raises `AttributeError` here instead of passing
    quietly at every value nobody thought to probe.

    AND IT MUST NOT RAISE. "A bar must never break a verdict" was true of
    `_bar_scope`'s body and false of the read that feeds it: at `eb1ff86` a
    record whose ``invocations`` is a `list` raised `TypeError` out of
    `make_solver_verdict` (`tuple + list`), from OUTSIDE `_bar_scope`'s
    protective `try`. It is tolerated here — a list of stamps is stamps —
    and anything genuinely unreadable returns :data:`_BAR_DOMAIN_UNREADABLE`,
    which widens rather than raising and rather than silencing.
    """
    if escalation is None:
        return {}
    try:
        decided: dict[int, tuple[SolverStamp, ...]] = {}
        for r in escalation.records:
            if r.outcome != OB_DISCHARGED:
                continue
            try:
                stamps = tuple(r.invocations)
            except TypeError:
                # not iterable at all: no stamps means no narrowing, which is
                # the widening direction, so this is a loss of precision and
                # never of the bar
                stamps = ()
            decided[r.index] = decided.get(r.index, ()) + stamps
        return decided
    except Exception:  # noqa: BLE001 — an unreadable escalation widens
        return _BAR_DOMAIN_UNREADABLE


def _unaccounted_solver_runs(escalation, ledger_stamps) -> int:
    """How many INVOKED solver runs the ledger witnesses that the supplied
    ``records`` do not account for. Zero for every escalation ``escalate()``
    builds, and zero whenever the records carry at least as much invocation as
    the ledger does.

    NOT A GATE, AND DELIBERATELY NOT ONE. The degenerate-`records` gate above
    refuses what it can prove came from nowhere — a ledger with work against an
    EMPTY `records`. It stops short of comparing the ledger's invoked stamps
    against the records' invocations because that comparison also refuses
    `tests/test_verified_bar.py::test_stripping_invocations_cannot_clear_the_bar`,
    which strips invocations on purpose to probe a different invariant. The
    same comparison is safe to CLASSIFY with: it decides which sentence the
    verdict writes about a still-undecided obligation, never whether the
    verdict is emitted and never what any obligation's status is.

    Counts INVOKED stamps on both sides, because an un-invoked stamp is a
    recorded non-run and witnesses nothing. Never raises and never returns a
    negative: an unreadable escalation yields 0, which is the quiet direction,
    and the only thing lost is one classifying sentence."""
    try:
        witnessed = sum(1 for s in ledger_stamps if s.invoked)
        accounted = sum(
            1
            for r in escalation.records
            for s in r.invocations
            if s.invoked
        )
        return max(witnessed - accounted, 0)
    except Exception:  # noqa: BLE001 — a note must never break a verdict
        return 0


def make_solver_verdict(
    closed: ir.ClosedJaxpr,
    propagation: Propagation,
    escalation: Escalation,
    *,
    stelling_version: str,
    jax_version: str,
    precision_config: str,
    device_class: str = _DEVICE_CLASS_DEFAULT,
    refinement=None,
) -> Verdict:
    """Assemble a verdict from interval propagation plus solver escalation.

    **PRECONDITION — the caller's, and it is now checked for two of the
    three arguments.** ``escalation`` must be the object :func:`escalate`
    returned for THIS ``closed`` and ``propagation``, unmodified. The
    gates below refuse several specific mispairings — divergent ledger
    provenance, a semantics mix in either direction, an ieee or
    constrained-assume propagation paired with an escalation carrying
    solver work, and **an escalation produced on a DIFFERENT query** —
    and they are the mispairings that arise from assembling a verdict out
    of the wrong RUN.

    **WHAT THE QUERY PAIRING GATE DOES AND DOES NOT BIND.** It binds
    ``closed`` to ``escalation``, by the query content hash
    :func:`escalate` recorded, and that is the leg the discharges travel
    on: an ``OB_DISCHARGED`` record from another run discharges an
    obligation here by INDEX alone, which is how a mispaired assembly
    minted VERIFIED on a query whose honest verdict is REFUTED. It does
    NOT bind ``propagation``, which carries no query hash. The residue is
    an assembly of (this query, ANOTHER query's propagation, this query's
    escalation): the obligations reported are the other query's, the
    stamp names this one, and it is measured rather than argued in
    `tests/test_verified_bar.py::test_the_pairing_gate_binds_the_ESCALATION_and_not_the_propagation`.

    None of the gates, and nothing else here, verifies that the
    records were produced by this library at all. A caller who
    hand-assembles an :class:`ObligationEscalation` is stating the
    outcome, and the assembly believes it: an ``OB_DISCHARGED`` record
    discharges its obligation, and an all-discharged verdict is
    ``VERIFIED``.

    That is a contract, not a hole to be plugged, and the reason is that
    plugging it would defend nothing. :class:`stelling.verdict.Verdict`
    is public, exported in ``verdict.__all__``, and a frozen dataclass
    whose ``__post_init__`` validates SHAPE and not PROVENANCE — it
    refuses an unknown status and a stamp/DECLINED mismatch, and asks
    nothing about where the numbers came from (so does
    :class:`stelling.verdict.Stamp`'s). A hand-built
    ``Verdict(status="VERIFIED", …)`` with a fabricated :class:`Stamp`
    therefore constructs and reports ``VERIFIED`` without passing through
    this function at all. Anyone able to forge a record can forge the
    verdict more cheaply, so hardening here buys no guarantee and would
    only make the weaker door look like the only one.
    **What the gates and the bar protect is an HONEST caller against an
    accidentally mispaired assembly, not this process against its own
    caller.** Consumers needing the stronger property should judge
    through :func:`stelling.preconditions.check`, which owns both sides.

    The scatter VERIFIED bar below rests on this precondition for its
    DOMAIN: WHICH obligations the solver decided is read off
    ``escalation.records`` (the same ``outcome == OB_DISCHARGED`` test
    that discharges them), while WHAT is barred on their slices is
    re-derived from ``closed``. **A wrong ``closed`` does NOT reliably
    widen the bar — that claim stood here bolded and is false, and the
    counterexample is one function over.** Measured on the
    identical-decided-slice pairing: ``_bar_scope(wrong_closed, decided)``
    returns ``((), '')`` — narrowed to NOTHING, empty reason — while
    ``_barred_primitives(wrong_closed)`` is ``('scatter',)``. The bar goes
    SILENT there, on a query that carries the barred primitive. It widens
    when the mispairing is one the evidence check can see, and the two
    earlier versions of this paragraph each named a mechanism ("the
    re-slice would decline", then "the evidence widens it") and then
    generalised it to every wrong ``closed``. What the evidence check
    actually does is narrower and is stated where it lives, in
    :func:`stelling.verdict._evidence_is_about`. **The pairing itself is
    not the bar's job: it is the query pairing gate above, which refuses
    the assembly outright.** What distinguishes the slices, when the
    assembly is reached at all, is the evidence: every recorded
    invocation carries
    ``smt2_sha256``, the hash of the exact script that was sent, AND
    ``slice_sha256``, the fingerprint of the slice that script was
    emitted from, and :func:`stelling.verdict._evidence_is_about` narrows
    the bar for an obligation only when re-emitting the slice re-derived
    from THIS ``closed`` reproduces BOTH. **The second is not a belt on
    the first.** A mispaired query does not necessarily re-emit a
    different script — the barred row emits no text, so a scatter-bearing
    slice and a scatter-free one reading the same untouched element emit
    byte for byte the same thing, which is exactly how `eb1ff86` cleared
    this bar. Only the slice fingerprint separates that pair. A fabricated
    record set is, as above, already a fabricated verdict.

    ``refinement`` (default None — byte-identical assembly) is the
    :class:`stelling.affine.RefinementReport` of an affine refinement
    that ran on ``propagation`` BEFORE the escalation (the escalation
    then saw only what the refinement left undecided): the stamp records
    the refinement ran, and the derived solver-absence reason names the
    layers that actually judged — when the refinement decided everything
    and a solver budget was offered, the absence line records that no
    solver was needed, truthfully.

    A separate assembly path: the public no-solver path
    (:func:`stelling.verdict.make_verdict`) is untouched. Status: REFUTED
    if any obligation is violated (set-level interval) or solver-refuted
    with a replayed witness; VERIFIED only if every obligation is
    discharged (by interval or by solver); else UNKNOWN.

    No escalated verdict emits over divergent provenance: the spawn
    counter (incremented at the transport-entry boundary) must equal the
    number of ``invoked=True`` stamps in the append-only ledger — checked
    unconditionally, on every escalated verdict; mismatch raises
    :exc:`ProvenanceError` instead of a verdict.

    Nor over a mispaired escalation: a propagation that constrained any
    assume may only pair with a refusal-shaped escalation (zero
    invocations, zero witnesses) — anything else raises
    :exc:`MispairedEscalationError` (the refusal invariant's second,
    anti-correlated mechanism; audit F3).
    """
    # -- ONE PASS OVER `records`, TAKEN HERE AND NOWHERE ELSE.
    #
    # This function walks `escalation.records` five times, and the field's
    # declared type is a tuple — but it is whatever the caller put there, and
    # a `records` that can be iterated ONCE (a generator, a `map`, a consumed
    # iterator) made those five passes see five different things. Ordering the
    # bar's domain first made that FAIL SAFE and did not make it CORRECT, and
    # two measurements on `e35de13` say so:
    #
    #   * on a SCATTER-FREE query — one the bar never touches — a one-shot
    #     `records` turned an honest VERIFIED into UNKNOWN, and the note it
    #     carried was the generic undecided-cause line, which attributes the
    #     UNKNOWN to "the propagated interval straddling the asserted bound".
    #     That is not silence, it is a WRONG EXPLANATION of a verdict the
    #     caller's own argument shape caused;
    #   * a TWO-FACED `records` — empty on the first pass, real on the rest —
    #     showed the bar nothing and the obligation loop everything, and
    #     returned VERIFIED with no withheld note on the bar's own fixture.
    #     Ordering cannot see that: the domain really was read first.
    #
    # Materialising once makes every later pass see ONE value, so nothing
    # downstream can disagree with anything else about what the records are.
    #
    # IT DOES NOT MAKE A DEGENERATE `records` BEHAVE LIKE "THE TUPLE IT
    # YIELDS", AND THE VERSION OF THIS SENTENCE THAT SAID SO — "rather than by
    # choosing which pass wins" — WAS FALSE. One pass at the top IS choosing
    # pass 1. For a one-shot `records` that happens to be the right choice,
    # because pass 1 is where the real records are. For a TWO-FACED one it is
    # the wrong one, and measured on `3e107cf` the misattribution the second
    # bullet above is about survives in exactly that shape:
    #
    #     scatter-free query, `records` empty on pass 1 and real after:
    #         VERIFIED -> UNKNOWN, obligation `unknown`, and the note is
    #         the generic undecided-cause line blaming an interval straddle
    #
    # which is the defect `SOUNDNESS.md` (2) recorded as closed. It was closed
    # for the ONE-SHOT shape only.
    #
    # SO THE SHAPE IS REFUSED RATHER THAN ABSORBED, one gate below: an
    # escalation whose LEDGER records solver work and whose `records` came back
    # empty is not a coherent escalation, and `escalate()` cannot produce one.
    # Ordering is kept below as a second, now-REDUNDANT mechanism — if this
    # line were ever removed, reading the domain first still costs the
    # discharges rather than the bar, which is the safer of the two failures.
    # That redundancy is stated because a mutation of the ordering ALONE is now
    # inert (measured: 0 RED), and an unpinned guard whose comment claims to be
    # load-bearing is this repo's own recurring defect.
    escalation = replace(escalation, records=tuple(escalation.records))

    # -- the provenance gate (runs before anything else, unconditionally)
    ledger_stamps = tuple(escalation.ledger.stamps)
    stamped = sum(1 for s in ledger_stamps if s.invoked)
    if escalation.ledger.spawns != stamped:
        raise ProvenanceError(escalation.ledger.spawns, stamped, ledger_stamps)

    # -- THE DEGENERATE-`records` GATE, and it is the ledger that catches it.
    # One pass fixes WHICH value the assembly sees; it cannot make a wrong
    # value right. The ledger is a separate field, carried whole, and it is an
    # independent witness of whether any solver ran: `spawns` is incremented at
    # the transport-entry boundary and the stamps are appended there. An
    # escalation that says solvers ran and hands over no record of what they
    # answered is internally inconsistent, and `escalate()` never builds one —
    # every spawn belongs to an obligation, and every obligation reaching a
    # backend gets a record.
    #
    # Refusing is the point. Absorbing it produced an UNKNOWN carrying a WRONG
    # EXPLANATION (the interval-straddle note) on a query whose honest verdict
    # is VERIFIED — worse than silence, because a reader believes it.
    #
    # WHAT IT DOES NOT REFUSE, stated rather than left to be found: a `records`
    # whose first pass yields a non-empty STRICT SUBSET. The ledger says work
    # happened and some record exists, so this gate passes, and the obligations
    # whose records were dropped come back `unknown`. Comparing the ledger's
    # invoked stamps against the invocations the records carry would REFUSE it
    # and would also refuse `test_stripping_invocations_cannot_clear_the_bar`'s
    # fixture, which is a deliberate probe of a DIFFERENT invariant; that trade
    # is not taken.
    #
    # BUT THE RESIDUE IS NOT LEFT CARRYING THE NOTE THIS GATE EXISTS TO STOP.
    # The justification above is "absorbing it produced an UNKNOWN carrying a
    # WRONG EXPLANATION", and measured on `faefc48` the strict-subset residue
    # emitted that explanation verbatim: scatter-free query, `records` yielding
    # one of two records on pass 1 — honest verdict VERIFIED, observed UNKNOWN
    # carrying "…the propagated interval straddling the asserted bound". A gate
    # justified by an argument its own residue violates is the argument being
    # wrong about its scope, so the same comparison that is too strong to
    # REFUSE with is used to CLASSIFY: `_unaccounted_solver_runs` below counts
    # the ledger's invoked stamps the records do not account for, and
    # `stelling.verdict.undecided_cause_note` says the outcome went missing
    # instead of blaming the propagation. Classifying is not refusing — the
    # stripped-invocations probe keeps its verdict and its bar, and gains one
    # true sentence about why its obligation is undecided.
    if (escalation.ledger.spawns or ledger_stamps) and not escalation.records:
        raise MispairedEscalationError(
            f"incoherent escalation: the ledger records "
            f"{escalation.ledger.spawns} spawn(s) and {len(ledger_stamps)} "
            f"stamp(s), so solvers ran, but the supplied `records` came back "
            f"EMPTY — no obligation outcome at all. `escalate()` cannot "
            f"produce that (every spawn belongs to an obligation, and every "
            f"obligation that reaches a backend gets a record), so `records` "
            f"is a container that does not yield what it holds: a generator, "
            f"a consumed iterator, or an iterable that answers differently on "
            f"different passes. Assembling anyway returns UNKNOWN with the "
            f"generic undecided-cause note, which attributes the verdict to "
            f"the propagation rather than to the argument that caused it; "
            f"refusing to emit. Pass the records as a materialised sequence."
        )

    # -- THE QUERY PAIRING GATE. Recomputed from `closed`, never copied, and
    # taken ONCE for the whole function: the stamp's `query_content_hash`
    # below is this same value, so the gate costs no additional hash.
    #
    # AN EMPTY HASH IS REFUSED ON ITS OWN, NOT ONLY WHEN IT DIFFERS. Both legs
    # go through `_query_sha256`, which returns "" when `content_hash()`
    # raises — so a `closed` that cannot be hashed and an escalation that
    # recorded nothing COMPARED EQUAL and the gate passed. Measured on
    # `e35de13`: the refusal then came from `Stamp.__post_init__`
    # ("stamp field 'query_content_hash' is empty") one layer later, which is
    # loud but is not this gate, and the field's own docstring claimed this
    # gate refused it. Equality is not the property wanted; "this escalation
    # is about this query, and both of them said so" is, and an empty string
    # says nothing.
    query_hash = _query_sha256(closed)
    carries_work = bool(
        escalation.records
        or escalation.notes
        or escalation.ledger.spawns
        or ledger_stamps
    )
    if carries_work and (
        not query_hash or escalation.query_sha256 != query_hash
    ):
        raise MispairedEscalationError(
            f"mispaired escalation: the supplied escalation was produced by "
            f"escalate() on the query "
            f"{escalation.query_sha256 or '<unrecorded>'}, but the query "
            f"being stamped hashes to {query_hash or '<unhashable>'} — the "
            f"two are not the same query (an UNHASHABLE query is refused "
            f"whatever the escalation recorded: two empty strings are not a "
            f"match, they are two absences), so this escalation's outcomes "
            f"are answers about a program other than the one the verdict "
            f"would claim; refusing to emit. An OB_DISCHARGED record from "
            f"another run discharges an obligation here by INDEX alone, so a "
            f"mispaired assembly can mint VERIFIED on a query whose honest "
            f"verdict is REFUTED. Assemble the verdict from the propagation "
            f"and escalation this query actually produced."
        )

    # THE BAR IS NOT THIS GATE, AND THE HISTORY IS WHY BOTH EXIST. Until this
    # gate, `make_solver_verdict` never bound its three arguments to one
    # query, and the whole-query scatter bar hid that on exactly one class of
    # query — scatter-bearing ones — by withholding every VERIFIED on them for
    # an unrelated reason. Scoping the bar to the decided obligation's slice
    # did not COST that backstop so much as REVEAL that it was a coincidence
    # of scope: the identical false VERIFIED was reachable on a SCATTER-FREE
    # query, where the bar never fires, on every build including `8e42934`.
    # Measured, both rows, on this branch's own fixtures. The bar answers "was
    # the unaudited emission row involved in what the solver decided"; this
    # gate answers "is this escalation about this query at all". Neither
    # substitutes for the other, and only one of them was ever load-bearing
    # for the pairing.

    # -- the symmetric semantics-pairing gate: an escalation may only be
    # stamped against a propagation of the semantics it was produced
    # from, in EITHER direction. Forward mix (ieee propagation +
    # real-produced escalation): solver outcomes over the reals must
    # never be stamped against a float-semantics claim. Reverse mix
    # (real propagation + ieee-produced refusal escalation): the
    # obligation details would quote the ieee refusal under a real stamp
    # — a misattribution, refused the same way. A completely EMPTY
    # escalation (no records, no notes, no spawns, no stamps — the
    # nothing-to-escalate shape) is exempt: it contributes nothing a
    # semantics mismatch could misattribute, and the absence reason is
    # derived from the propagation's own semantics below.
    if (
        escalation.records
        or escalation.notes
        or escalation.ledger.spawns
        or ledger_stamps
    ) and escalation.semantics != propagation.semantics:
        raise MispairedEscalationError(
            f"mispaired escalation: the propagation being stamped ran "
            f"under semantics='{propagation.semantics}', but the supplied "
            f"escalation was produced from a "
            f"semantics='{escalation.semantics}' propagation — obligation "
            f"details and refusal reasons would be misattributed across "
            f"the semantics boundary; refusing to emit. Assemble the "
            f"verdict from the propagation the escalation was actually "
            f"produced from."
        )

    # -- the ieee mispairing gate: the second, anti-correlated mechanism
    # of the ieee escalation refusal (guard 2). escalate() refuses against
    # the propagation IT receives; this gate refuses against the
    # propagation being STAMPED, catching the mode-mixed caller bypass —
    # kept alongside the semantics-pairing gate above (which relies on
    # the escalation's own semantics record; this one is derived from the
    # WORK the escalation carries, so a forged/buggy semantics field
    # cannot smuggle ℝ solver outcomes under an ieee stamp).
    if propagation.semantics == "ieee" and (
        escalation.ledger.spawns
        or ledger_stamps
        or any(
            r.invocations or r.witness is not None or r.outcome != OB_UNKNOWN
            for r in escalation.records
        )
    ):
        n_wit = sum(1 for r in escalation.records if r.witness is not None)
        raise MispairedEscalationError(
            f"mispaired escalation: the propagation being stamped ran "
            f"under semantics='ieee', but the supplied escalation carries "
            f"solver work ({escalation.ledger.spawns} spawn(s), "
            f"{len(ledger_stamps)} ledger stamp(s), {n_wit} witness(es)) — "
            f"it cannot have been produced from this propagation, whose "
            f"escalation is refused outright ({IEEE_SEMANTICS_REFUSAL}). "
            f"The SMT backends emit over the reals; their outcomes prove "
            f"the ℝ obligation and must not be stamped against an "
            f"ieee-semantics claim; refusing to emit. Assemble the verdict "
            f"from the propagation the escalation was actually produced "
            f"from."
        )

    # -- the mispairing gate: the second, anti-correlated mechanism of the
    # constrained-assume refusal (audit F3). escalate() refuses against
    # the propagation IT receives; this gate refuses against the
    # propagation being STAMPED. A constrained propagation may only pair
    # with a refusal-shaped escalation — zero spawns, zero stamps, zero
    # witnesses, every record UNKNOWN with no invocations.
    if propagation.coverage.constrained and (
        escalation.ledger.spawns
        or ledger_stamps
        or any(
            r.invocations or r.witness is not None or r.outcome != OB_UNKNOWN
            for r in escalation.records
        )
    ):
        n_wit = sum(1 for r in escalation.records if r.witness is not None)
        raise MispairedEscalationError(
            f"mispaired escalation: the propagation being stamped "
            f"constrained {propagation.coverage.constrained} assume(s), but "
            f"the supplied escalation carries solver work "
            f"({escalation.ledger.spawns} spawn(s), {len(ledger_stamps)} "
            f"ledger stamp(s), {n_wit} witness(es)) — it cannot have been "
            f"produced from this propagation, whose escalation is refused "
            f"outright. Solver outcomes over the un-assumed declared box "
            f"must not be stamped against the conditional claim (a sat "
            f"witness may violate the stamped precondition); refusing to "
            f"emit. Assemble the verdict from the propagation the "
            f"escalation was actually produced from."
        )

    # THE BAR'S DOMAIN IS READ BEFORE ANY OTHER PASS OVER `records`, AND THAT
    # ORDER IS NO LONGER LOAD-BEARING. It used to be read at the bar, several
    # passes later, and a `records` that can only be iterated ONCE — a
    # generator, a `map`, a consumed iterator — was therefore fully consumed by
    # `by_index` below before `_bar_domain` ever saw it. `_bar_domain` then
    # returned an HONEST-EMPTY `{}`, which is the one value that silences the
    # bar (empty means "no solver decided anything"), and the assembly returned
    # VERIFIED with no withheld note. Measured, and identical at `eb1ff86`: a
    # one-shot `records` cleared the scatter bar on the bar's own fixture. The
    # sentinel could not help — the read SUCCEEDED, it just ran second.
    #
    # THE SENTENCE THAT STOOD HERE — "AND THE ORDER IS LOAD-BEARING" — WAS
    # FALSE AS SOON AS `records` WAS MATERIALISED AT THE TOP, and it
    # contradicted the comment at that materialisation, which already called
    # the ordering "a second, now-REDUNDANT mechanism". Measured: moving this
    # line below `by_index` is 0 RED across the whole suite. With one pass
    # there is one value, so no ordering of the readers can show them different
    # things — which is exactly why the top comment also says that an unpinned
    # guard whose comment claims to be load-bearing is this repo's own
    # recurring defect. The order is kept as the safer arrangement if the
    # materialisation is ever removed, and it is described as what it is.
    # See `tests/test_verified_bar.py::test_a_ONE_SHOT_records_behaves_EXACTLY_LIKE_THE_TUPLE_it_yields`.
    decided = _bar_domain(escalation)
    by_index = {r.index: r for r in escalation.records}
    final: list[ObligationReport] = []
    for ob in propagation.obligations:
        record = by_index.get(ob.index) if ob.status == "unknown" else None
        if record is None:
            final.append(ob)
            continue
        if record.outcome == OB_DISCHARGED:
            status, detail = "discharged", record.detail
        elif record.outcome == OB_VIOLATED_WITNESS:
            status, detail = OB_VIOLATED_WITNESS, record.detail
        elif record.outcome == OB_VIOLATED_CONSTANT:
            # a constants-only refutation: no witness values exist, and the
            # set-level "definitely false over the declared set" reading is
            # exact for an inputless obligation — render it as such.
            status, detail = "violated-over-set", record.detail
        else:
            status, detail = "unknown", record.detail
        final.append(
            ObligationReport(
                index=ob.index,
                status=status,
                detail=detail,
                source_info=ob.source_info,
                operand_var_ids=ob.operand_var_ids,
            )
        )
    obligations = tuple(final)

    if any(o.status in ("violated-over-set", OB_VIOLATED_WITNESS) for o in obligations):
        status = "REFUTED"
    elif obligations and all(o.status == "discharged" for o in obligations):
        status = "VERIFIED"
    else:
        status = "UNKNOWN"

    # -- reaches-output conjunct ------------------------------------------------
    obligations, reachability_notes, status = (
        _verdict._apply_reachability_conjunct(closed, obligations, status)
    )

    nonvacuity = _nonvacuity_summary(propagation.nonvacuity_checks)
    notes = propagation.notes + escalation.notes
    for record in escalation.records:
        notes = notes + record.notes
    # the coverage-cause classification of any STILL-undecided obligation
    # (post-escalation — solver-decided ones need no cause), from the one
    # shared derivation in stelling.verdict so the two paths cannot drift
    notes = notes + _verdict.undecided_cause_note(
        propagation.coverage, obligations,
        _unaccounted_solver_runs(escalation, ledger_stamps),
    ) + reachability_notes
    # THE SCATTER VERIFIED BAR (stelling.verdict.VERIFIED_BARRED_PRIMITIVES).
    # Scoped to the SOLVER path deliberately, and this scoping is the whole
    # design decision: the bar exists because a new SMT EMISSION row that
    # missed a violation would mint a false VERIFIED with nothing downstream
    # to catch it. It is not a doubt about the long-standing interval
    # transfer, which is censused and unchanged. A whole-query bar would flip
    # every interval-only VERIFIED on a scatter-bearing slice — including the
    # Richardson/HeatNode flagship, whose Dirichlet writeback puts `scatter`
    # in its jaxpr — for a reason that has nothing to do with the row under
    # audit. Barring the claim the audit is about, and only that one, is the
    # narrowest bar that does the job.
    #
    # THE CONDITION IS "THE SOLVER DECIDED SOMETHING", NOT "THE CALLER ASKED
    # FOR A SOLVER". `escalation is not None` is true whenever the caller
    # passed solver_timeout_ms, INCLUDING when interval arithmetic had already
    # discharged every obligation and escalation therefore did nothing. Under
    # that condition the bar fired on verdicts no emission row had touched,
    # and the same query returned VERIFIED without the argument and UNKNOWN
    # with it -- measured on a downstream contact-mechanics contract, with
    # obl-solves = 0 and the obligation `discharged` in both runs.
    #
    # That violated a property much wider than the bar: THE VERDICT IS A
    # FUNCTION OF THE QUERY AND THE ENVELOPE, not of an escalation preference
    # that had no effect. Stamp comparison, the W6 soak's per-line records,
    # and the no-flip gate's whole premise assume it. See
    # tests/test_escalation_invariant.py, which asserts it directly.
    #
    # "Solver-decided" is read from the escalation RECORDS rather than from
    # the ledger: a ledger stamp says an invocation happened, while an
    # OB_DISCHARGED record says an invocation ANSWERED -- and a false VERIFIED
    # can only be minted by an answer.
    #
    # AND IT IS THE SAME PREDICATE THAT DISCHARGES, LITERALLY: `record.outcome
    # == OB_DISCHARGED`, the test the obligation loop above makes. A
    # predecessor wrote `... and r.invocations` here and nowhere else, which
    # made "discharging" and "decided" two different concepts over one record.
    # Measured on the two-obligation fixture: strip `invocations` from the
    # scatter obligation's discharging record and the obligation still
    # discharged -- so the VERIFIED still stood -- while its slice dropped out
    # of the bar's domain and the verdict went VERIFIED where `8e42934`
    # returned UNKNOWN. The extra conjunct was not a second check on the same
    # thing; it was a second definition of it, and the wider one won where it
    # mattered. It is gone rather than hardened: with one predicate there is no
    # second one to drift from. Nothing is lost -- exactly one of the eleven
    # `ObligationEscalation(` sites emits OB_DISCHARGED (the `unsat` branch of
    # `_dispatch_obligation`), it reaches that branch only because a backend
    # ANSWERED, and every answering backend was stamped into the ledger before
    # its transport ran, so an honest OB_DISCHARGED record always carries
    # invocations and the conjunct never excluded one.
    #
    # THE INDEX SET IS DELIBERATELY A SUPERSET OF WHAT WAS DISCHARGED HERE, and
    # that direction is the safe one. The loop above applies a record only to an
    # obligation that is still `unknown`; this set takes every OB_DISCHARGED
    # record's index, including one that matched nothing. A stray index widens
    # the bar and never narrows it -- but NOT, as a predecessor of this comment
    # claimed, because it "does not slice". Measured on the bar's own fixture:
    # index 1 names an obligation INTERVALS decided and slices to
    # `['broadcast_in_dim','ge','scatter']`; index -1 is Python indexing and
    # slices the LAST obligation; index 99 declines; and index -3 (or lower)
    # raises IndexError out of `slice_obligation`, which `_bar_scope`'s outer
    # `except` turns into the same whole-query set. That FOURTH behaviour is
    # named because the version of this comment that listed three read as a
    # closed enumeration and was not one -- the same shape of claim this file
    # keeps having to correct. What widens the bar for the first three is that
    # none of them carries a solver invocation whose recorded script hash AND
    # slice fingerprint both re-emit from the obligation it names -- see
    # `verdict._evidence_is_about`.
    #
    # AND THE SCOPE IS THE DECIDED OBLIGATION'S SLICE, NOT THE WHOLE QUERY.
    # Same argument one level finer. A query can carry `scatter` on an
    # obligation intervals settled while the obligation the SOLVER decided
    # never touches it -- the emission row the bar exists for was not
    # consulted, so there is nothing for it to be wrong about. Measured: the
    # bar's own regression fixture was exactly that shape (solver-decided
    # slice `['sub','ge']`, the scatter on a different, interval-decided
    # obligation) and returned UNKNOWN.
    #
    # NO BARRED PRIMITIVE COMES FROM HERE. `_bar_scope` re-derives what is on
    # those obligations' slices out of `closed` itself; this function hands it
    # the numbers of the obligations the solver decided -- already load-bearing
    # for the VERIFIED being withheld, since a record whose index matches no
    # unknown obligation leaves that obligation undischarged and there is no
    # VERIFIED to bar -- together with the INVOCATION STAMPS those records
    # carry, which is what lets the bar check that the escalation is evidence
    # about this query at all rather than about some other one of the same
    # shape. The stamps are handed over for the WIDENING decision only: no
    # stamps means no narrowing, so `invocations` cannot clear the bar, and the
    # domain itself is still `outcome == OB_DISCHARGED` alone -- the same
    # predicate that discharges, which is the drift this pass will not
    # reintroduce. See `verdict._bar_scope` and `verdict._evidence_is_about`,
    # the deleted `barred_on_slice` field for what reading the contents cost,
    # and this function's docstring for the precondition the domain rests on.
    #
    # THE READ ITSELF LIVES IN `_bar_domain`, ONE PLACE, and it is a function
    # rather than four lines here so that a test can hand it a record with no
    # other field and watch a conjunct on a new field RAISE. Enumerating the
    # values a field could hold does not close that channel; `str` has too
    # many. It also has to be a place that cannot raise: the `tuple + list`
    # this loop used to be raised `TypeError` out of this function from
    # OUTSIDE `_bar_scope`'s protective `try`, so "a bar must never break a
    # verdict" did not cover the whole path feeding the bar. `decided` is
    # computed at the TOP of the obligation loop, not here: see the comment
    # there for the one-shot `records` shape that ordering closes.
    if status == "VERIFIED" and decided:
        barred, scope = _verdict._bar_scope(closed, decided)
        if barred:
            status = "UNKNOWN"
            notes = notes + (
                "VERIFIED withheld — "
                + _verdict.VERIFIED_BAR_REASON.format(
                    scope=scope, prims=", ".join(barred)
                ),
            )

    if status == "VERIFIED" and not nonvacuity.startswith("checked"):
        notes = notes + (
            f"nonvacuity {nonvacuity.split(' — ')[0]}: this VERIFIED may be "
            f"vacuous — the declared set is not tied to the incident's data",
        )

    # -- the single derivation point for the stamp's solver field --------
    # Assembled from the ledger alone: a non-empty ledger IS the tuple of
    # appended stamps; an empty ledger IS absence, with the reason derived
    # (nothing was unknown / nothing installed or every unknown obligation
    # declined pre-invocation). Absence is never written by a degradation
    # branch — an appended stamp cannot be talked out of the record.
    solver: SolverStamp | tuple[SolverStamp, ...]
    if ledger_stamps:
        solver = ledger_stamps
    else:
        if not any(o.status == "unknown" for o in propagation.obligations):
            if propagation.semantics == "ieee":
                # the ieee wording names the arithmetic that actually
                # judged it (native binary64, no outward rounding) — the
                # real-mode sentence below stays byte-identical
                reason = (
                    "no solver invoked: every obligation was decided by "
                    "native-binary64 interval arithmetic alone; escalation "
                    "had nothing to do"
                )
            else:
                reason = (
                    "no solver invoked: every obligation was decided by outward-rounded "
                    "interval arithmetic alone; escalation had nothing to do"
                )
        else:
            reason = (
                "no solver invoked: escalation completed no invocation (solver "
                "unavailable, every unknown obligation declined, or a failure "
                "before any invocation; the notes carry the reasons)"
            )
        if refinement is not None:
            # the deciding layers, named truthfully: the report's own
            # derivation (shared with make_verdict, so the two cannot
            # drift), never separate narration
            reason = refinement.reword_absence(reason)
        solver = solver_absent(reason)

    # which arithmetic the verdict is about comes from the propagation
    # that ran (the honest ieee pairing — a refusal-shaped escalation —
    # still emits, and its stamp must say ieee, not ℝ; the 0·∞ = 0
    # convention line must not ride in it)
    if propagation.semantics == "ieee":
        semantics = SEMANTICS_IEEE
        arithmetic_mode = ARITHMETIC_MODE_INTERVAL_IEEE
        convention = IEEE_ENDPOINT_ASSUMPTION
    else:
        semantics = SEMANTICS_REAL
        arithmetic_mode = ARITHMETIC_MODE_INTERVAL
        convention = REAL_CONVENTION_ASSUMPTION
    assumptions = tuple(sorted({*propagation.assumptions, convention}))
    if refinement is not None:
        # a stamped line records the refinement was enabled (domain,
        # registry, ops actually used) — appended after the sorted set,
        # the same append-only mechanics the vacuity line uses; the
        # arithmetic line names the deciding abstraction when the affine
        # domain decided anything
        assumptions = assumptions + (refinement.stamp_line(),)
        arithmetic_mode = refinement.reword_arithmetic(arithmetic_mode)
    stamp = Stamp(
        stelling_version=stelling_version,
        jax_version=jax_version,
        # the SAME hash the pairing gate above compared — taken once per
        # assembly, so binding the escalation to the query costs nothing here
        query_content_hash=query_hash,
        arithmetic_mode=arithmetic_mode,
        semantics=semantics,
        precision_config=precision_config,
        device_class=device_class,
        solver=solver,
        nonvacuity=nonvacuity,
        transfer_tiers=propagation.transfers_used,
        transfer_provenance=tuple(
            (p, "core") for p, _ in propagation.transfers_used
        ),
        assumptions=assumptions,
        coverage=propagation.coverage.summary(),
        # the SAME derivation make_verdict uses (shared, so the two
        # surfaces cannot disagree about whether the field exists — a
        # field present on one published stamp and absent on the other
        # is a surface that contradicts itself)
        top_despite_coverage=_verdict.top_despite_coverage_note(propagation),
    )
    witnesses = tuple(
        r.witness for r in escalation.records if r.witness is not None
    )
    # the counting surface, derived from the records that actually DECIDED
    # something — an obligation nobody answered has no redundancy to report,
    # and listing it at zero would put every UNKNOWN in the degraded column
    redundancy = tuple(
        (r.index, r.answered_by)
        for r in escalation.records
        if r.outcome != OB_UNKNOWN
    )
    return Verdict(
        status=status,
        obligations=obligations,
        stamp=stamp,
        notes=notes,
        witnesses=witnesses,
        solver_redundancy=redundancy,
    )
