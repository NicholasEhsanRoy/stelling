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
raises :exc:`EmissionInfidelityError`, and a non-rational model leaves
the obligation UNKNOWN by policy. ``unknown``/timeout is UNKNOWN, never
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
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

from stelling import _optional
from stelling import ir
from stelling import verdict as _verdict
from stelling.obligation import (
    DeclinedObligation,
    ObligationSlice,
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
    # and `make_solver_verdict` never binds its escalation to its `closed`,
    # so a scatter-free escalation stamped against a scatter-bearing query
    # went VERIFIED where the whole-query bar went UNKNOWN. A record cannot
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
    boundary."""

    records: tuple[ObligationEscalation, ...]
    notes: tuple[str, ...] = ()
    ledger: _Ledger = field(default_factory=_Ledger)
    semantics: str = "real"

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


def _run_cvc5_wheel(script_text: str, wall_s: float) -> _RawResult:
    version = _cvc5_wheel_version()
    argv = [sys.executable, "-m", "stelling._cvc5_driver"]
    try:
        proc = subprocess.run(
            argv, input=script_text, capture_output=True, text=True, timeout=wall_s
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
    answer = ""
    values: list[tuple[str, str]] = []
    nonrational = False
    error = ""
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if not parts:
            continue
        if parts[0] == "version" and len(parts) >= 2:
            version = parts[1]
        elif parts[0] == "answer" and len(parts) >= 2:
            answer = parts[1]
        elif parts[0] == "value" and len(parts) == 3:
            values.append((parts[1], parts[2]))
        elif parts[0] == "opaque":
            nonrational = True
        elif parts[0] == "error":
            error = line[len("error "):]
    if error:
        return _RawResult(answer="failed", version=version, detail=_quote(error))
    if answer not in ("sat", "unsat", "unknown"):
        return _RawResult(
            answer="failed",
            version=version,
            detail=f"cvc5 driver protocol violation; stdout: "
            f"{_quote(proc.stdout)!r}, stderr: {_quote(proc.stderr)!r}",
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
        raise EmissionInfidelityError(
            obligation_index=sl.index,
            solver=solver_label,
            values=tuple((inp.name, str(values[inp.name])) for inp in sl.inputs),
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
    return Witness(
        obligation_index=sl.index,
        values=tuple((inp.name, str(values[inp.name])) for inp in sl.inputs),
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
) -> ObligationEscalation:
    ordered = tuple(
        sorted(
            backends,
            key=lambda b: (b.name != "cvc5") if sl.fragment == "QF_NRA" else (b.name != "z3"),
        )
    )
    scripts: dict[str, Script] = {}
    for backend in ordered:
        if backend.flavor not in scripts:
            scripts[backend.flavor] = emit(sl, backend.flavor, config.timeout_ms)
    wall_s = _wall_seconds(config.timeout_ms)
    notes: list[str] = []
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
        return ObligationEscalation(
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
        )

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
            if not sl.inputs:
                # audit F5: a constants-only obligation has no witness
                # values to render; the SAME validator decides the
                # refutation is real (membership vacuously true; violation
                # via the empty-environment replay of the closed formula)
                # — an honest REFUTED, no fabricated witness.
                _require_valid_refutation(
                    sl, values, solver_label=backend.label,
                    script_text=script.text,
                )
                return ObligationEscalation(
                    index=sl.index,
                    outcome=OB_VIOLATED_CONSTANT,
                    detail=(
                        f"constant refutation: the obligation has no declared "
                        f"inputs and its predicate is definitely false — "
                        f"{_REPLAY_SENTENCE}; {backend.label} ({sl.fragment}) "
                        f"answered sat in agreement"
                        + degraded_clause(universal=False)
                    ),
                    invocations=invocations(),
                    witness=None,
                    notes=tuple(notes) + degraded_notes(universal=False),
                    answered_by=answered,
                )
            witness = make_validated_witness(
                sl,
                values,
                produced_by=(
                    f"{backend.name} {stamp.version} ({backend.transport})"
                ),
                solver_label=backend.label,
                script_text=script.text,
            )
            elements = ""
            if witness.violating_elements:
                # the array assert is a universal elementwise claim; the
                # detail names which element(s) the witness falsifies
                # (flat C-order indices into the assert operand)
                elements = (
                    "; violating element(s) of the assert operand: "
                    + ", ".join(str(i) for i in witness.violating_elements)
                )
            return ObligationEscalation(
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
            )
        detail_tail = (
            "; ".join(sat_problems)
            if sat_problems
            else "witness not independently replayable"
        )
        return ObligationEscalation(
            index=sl.index,
            outcome=OB_UNKNOWN,
            detail=f"solver reported sat; {detail_tail} — UNKNOWN by policy",
            invocations=invocations(),
            witness=None,
            notes=tuple(notes),
        )

    reasons = (
        "; ".join(
            f"{b.label}: {raw.answer}" for b, _, raw, _ in runs if raw.answer != "not-run"
        )
        or "every invocation's transport failed before its solver could run"
    )
    return ObligationEscalation(
        index=sl.index,
        outcome=OB_UNKNOWN,
        detail=f"solver escalation did not decide ({reasons}); a timeout is never a VERIFIED",
        invocations=invocations(),
        witness=None,
        notes=tuple(notes),
    )


# -- escalation over a propagated query ---------------------------------------


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
    unknown = [o for o in propagation.obligations if o.status == "unknown"]
    if not unknown:
        return Escalation(
            records=(), notes=(), semantics=propagation.semantics
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
        )
    env = interval_env(closed)
    ledger = _Ledger()
    records: list[ObligationEscalation] = []
    for item in slice_unknown_obligations(closed, propagation, env):
        if isinstance(item, DeclinedObligation):
            records.append(
                ObligationEscalation(
                    index=item.index,
                    outcome=OB_UNKNOWN,
                    detail=f"escalation declined: {item.reason}",
                    invocations=(),
                    witness=None,
                    notes=(f"assert #{item.index}: escalation declined — {item.reason}",),
                )
            )
            continue
        ledger_start = len(ledger.stamps)
        try:
            record = _dispatch_obligation(item, config, backends, missing, ledger)
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
        records.append(record)
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
        records = [
            r if r.outcome not in (OB_VIOLATED_WITNESS, OB_VIOLATED_CONSTANT)
            else ObligationEscalation(
                index=r.index,
                outcome=OB_UNKNOWN,
                detail=f"violation WITHHELD from REFUTED: {DROPPED_ASSUME_REFUSAL}",
                invocations=r.invocations,
                witness=None,
                notes=r.notes + (DROPPED_ASSUME_REFUSAL,),
            )
            for r in records
        ]
    return Escalation(
        records=tuple(records), notes=(), ledger=ledger,
        semantics=propagation.semantics,
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
    empty one."""

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

    **PRECONDITION — the caller's, and it is not checked here.**
    ``escalation`` must be the object :func:`escalate` returned for THIS
    ``closed`` and ``propagation``, unmodified. The gates below refuse
    several specific mispairings — divergent ledger provenance, a
    semantics mix in either direction, an ieee or constrained-assume
    propagation paired with an escalation carrying solver work — and they
    are the mispairings that arise from assembling a verdict out of the
    wrong RUN. None of them, and nothing else here, verifies that the
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
    re-derived from ``closed``. **A wrong ``closed`` widens the bar, and
    the reason is measured rather than assumed.** An earlier version of
    this paragraph said it widened because the re-slice would decline;
    it does not decline — an obligation index that exists in the wrong
    query slices out of it perfectly well. What actually distinguishes
    the two is the evidence: every recorded invocation carries
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
    # -- the provenance gate (runs before anything else, unconditionally)
    ledger_stamps = tuple(escalation.ledger.stamps)
    stamped = sum(1 for s in ledger_stamps if s.invoked)
    if escalation.ledger.spawns != stamped:
        raise ProvenanceError(escalation.ledger.spawns, stamped, ledger_stamps)

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
            )
        )
    obligations = tuple(final)

    if any(o.status in ("violated-over-set", OB_VIOLATED_WITNESS) for o in obligations):
        status = "REFUTED"
    elif obligations and all(o.status == "discharged" for o in obligations):
        status = "VERIFIED"
    else:
        status = "UNKNOWN"

    nonvacuity = _nonvacuity_summary(propagation.nonvacuity_checks)
    notes = propagation.notes + escalation.notes
    for record in escalation.records:
        notes = notes + record.notes
    # the coverage-cause classification of any STILL-undecided obligation
    # (post-escalation — solver-decided ones need no cause), from the one
    # shared derivation in stelling.verdict so the two paths cannot drift
    notes = notes + _verdict.undecided_cause_note(
        propagation.coverage, obligations
    )
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
    # verdict" did not cover the whole path feeding the bar.
    decided = _bar_domain(escalation)
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
        query_content_hash=closed.content_hash(),
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
