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
``sat`` becomes REFUTED only after the model replays as an exact-rational
violation (:func:`stelling.obligation.evaluate_predicate`) — a replay
failure raises :exc:`EmissionInfidelityError`, and a non-rational model
leaves the obligation UNKNOWN by policy. ``unknown``/timeout is UNKNOWN,
never VERIFIED. Every invocation the verdict relied on is stamped, with
the exact emitted option set and the script hash; solvers are never
invoked on defaults, and the dispatch config's time limit is required.

Guard rule: every decline and every solver failure (crash, garbage
output, missing binary, unsupported fragment) degrades the obligation to
UNKNOWN with the reason quoted in the verdict notes. Raising is reserved
for stamp-contract violations, :exc:`SolverDisagreement`, and
:exc:`EmissionInfidelityError`.

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
from stelling.obligation import (
    DeclinedObligation,
    ObligationSlice,
    ReplayError,
    evaluate_predicate,
    slice_unknown_obligations,
)
from stelling.propagate import ObligationReport, Propagation, interval_env
from stelling.smt import Script, emit
from stelling.verdict import (
    ARITHMETIC_MODE_INTERVAL,
    REAL_CONVENTION_ASSUMPTION,
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
    "ObligationEscalation",
    "SolverConfig",
    "SolverDisagreement",
    "escalate",
    "make_solver_verdict",
]

TRANSPORT_Z3_WHEEL = "wheel-bindings (smt2 text)"
TRANSPORT_CVC5_WHEEL = "wheel-bindings (smt2 text; wall-guarded child process)"
TRANSPORT_CVC5_BINARY = "external-binary subprocess"

INSTALL_HINT = (
    'no SMT solver is installed — pip install "stelling[solvers]" (or set '
    "STELLING_CVC5 / put cvc5 on PATH) to enable escalation"
)

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


@dataclass(frozen=True)
class ObligationEscalation:
    """The escalation outcome for one obligation."""

    index: int
    outcome: str  # OB_DISCHARGED | OB_VIOLATED_WITNESS | OB_UNKNOWN
    detail: str
    invocations: tuple[SolverStamp, ...]
    witness: Witness | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Escalation:
    """All escalation outcomes for one propagated query."""

    records: tuple[ObligationEscalation, ...]
    notes: tuple[str, ...] = ()

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
    run: Callable[[str, float], _RawResult] = field(compare=False)


def _wall_seconds(timeout_ms: int) -> float:
    """The wall-clock guard budget: the solver's own limit plus headroom so
    the emitted option can fire first, the guard second — but always
    bounded (a hung solver cannot hang the verdict)."""
    return timeout_ms / 1000.0 * 1.5 + 1.0


def _quote(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


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
                    else:  # e.g. an algebraic root-obj: not a rational
                        nonrational = True
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
    Returns (values, saw_nonrational)."""
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
                        run=_run_z3,
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
                    run=_make_run_cvc5_binary(env_path),
                )
            )
        elif _optional.available("cvc5"):
            backends.append(
                _Backend(
                    name="cvc5",
                    flavor="cvc5",
                    label="cvc5 (wheel)",
                    transport=TRANSPORT_CVC5_WHEEL,
                    run=_run_cvc5_wheel,
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
                        run=_make_run_cvc5_binary(path),
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
    """Fill genuine per-variable don't-cares from each input's own declared
    box (disclosed; the replay still decides). Callers guarantee at least
    one declared input was actually supplied (audit F2) — a model that
    supplies none is a transport failure, not a field of don't-cares."""
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


def _box_escape(
    sl: ObligationSlice, values: dict[str, Fraction]
) -> str | None:
    """The first witness value outside its input's declared closed box, as
    an exact-rational comparison (finite sides only; a half-infinite bound
    checks its finite side; a (-inf, inf) input is unconstrained) — or None
    if every value is a member. The box constraints are part of the emitted
    problem, so an escaping model means the emitted problem does not mean
    the obligation (audit F1)."""
    for inp in sl.inputs:
        v = values[inp.name]
        if inp.lo != float("-inf") and v < Fraction(inp.lo):
            return (
                f"{inp.name} = {v} is below its declared lower bound "
                f"{Fraction(inp.lo)}"
            )
        if inp.hi != float("inf") and v > Fraction(inp.hi):
            return (
                f"{inp.name} = {v} is above its declared upper bound "
                f"{Fraction(inp.hi)}"
            )
    return None


def _dispatch_obligation(
    sl: ObligationSlice,
    config: SolverConfig,
    backends: tuple[_Backend, ...],
    missing: tuple[str, ...],
    stamp_sink: list[SolverStamp],
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
    if len(ordered) == 1:
        notes.append(
            f"assert #{sl.index}: portfolio degraded — only {ordered[0].label} "
            f"ran ({', '.join(missing) or 'the other solver'} not installed)"
        )

    runs: list[tuple[_Backend, Script, _RawResult, SolverStamp | None]] = []
    for position, backend in enumerate(ordered):
        script = scripts[backend.flavor]
        started = time.monotonic()
        raw = backend.run(script.text, wall_s)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if raw.answer == "not-run":
            notes.append(
                f"assert #{sl.index}: {backend.label} could not be invoked "
                f"({_quote(raw.detail)})"
            )
            runs.append((backend, script, raw, None))
            continue
        role = "primary" if position == 0 else "secondary"
        if len(ordered) == 1:
            role = "alone (degraded portfolio)"
        stamp = SolverStamp(
            invoked=True,
            reason=(
                f"{sl.fragment} portfolio {role} on assert #{sl.index}: "
                f"answered {raw.answer} in {elapsed_ms}ms"
            ),
            name=backend.name,
            version=raw.version or "unknown",
            transport=backend.transport,
            options=script.stamp_options(),
        )
        stamp_sink.append(stamp)  # survives even an internal error (audit F5)
        runs.append((backend, script, raw, stamp))
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

    invocations = tuple(stamp for _, _, _, stamp in runs if stamp is not None)
    answers = {raw.answer for _, _, raw, _ in runs}

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
            ),
            invocations=invocations,
            witness=None,
            notes=tuple(notes),
        )

    if "sat" in answers:
        sat_problems: list[str] = []
        for backend, script, raw, _stamp in runs:
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
            witness_values = tuple(
                (inp.name, str(values[inp.name])) for inp in sl.inputs
            )
            # audit F1: membership BEFORE replay — every value, solver-given
            # and completed alike, must lie inside its declared closed box.
            escape = _box_escape(sl, values)
            if escape is not None:
                raise EmissionInfidelityError(
                    obligation_index=sl.index,
                    solver=backend.label,
                    values=witness_values,
                    script_text=script.text,
                    detail=(
                        f"the model escapes the declared box ({escape}); the "
                        f"box constraints were part of the emitted problem"
                    ),
                )
            try:
                holds = evaluate_predicate(sl, values)
            except ReplayError as e:
                raise EmissionInfidelityError(
                    obligation_index=sl.index,
                    solver=backend.label,
                    values=witness_values,
                    script_text=script.text,
                    detail=f"the replay could not evaluate it ({e})",
                ) from e
            if holds:
                raise EmissionInfidelityError(
                    obligation_index=sl.index,
                    solver=backend.label,
                    values=witness_values,
                    script_text=script.text,
                    detail=(
                        "the exact-rational replay found the predicate TRUE "
                        "at that point"
                    ),
                )
            if not sl.inputs:
                # audit F5: a constants-only obligation has no witness
                # values to render; the replay of the closed formula proved
                # the violation outright — an honest REFUTED, no fabricated
                # witness.
                return ObligationEscalation(
                    index=sl.index,
                    outcome=OB_VIOLATED_CONSTANT,
                    detail=(
                        f"constant refutation: the obligation has no declared "
                        f"inputs and its predicate is definitely false — "
                        f"{_REPLAY_SENTENCE}; {backend.label} ({sl.fragment}) "
                        f"answered sat in agreement"
                    ),
                    invocations=invocations,
                    witness=None,
                    notes=tuple(notes),
                )
            witness = Witness(
                obligation_index=sl.index,
                values=witness_values,
                produced_by=(
                    f"{backend.name} {raw.version} ({backend.transport})"
                ),
                replay=_REPLAY_SENTENCE,
            )
            return ObligationEscalation(
                index=sl.index,
                outcome=OB_VIOLATED_WITNESS,
                detail=(
                    f"violated at a concrete witness found by {backend.label} "
                    f"({sl.fragment}); {_REPLAY_SENTENCE}"
                ),
                invocations=invocations,
                witness=witness,
                notes=tuple(notes),
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
            invocations=invocations,
            witness=None,
            notes=tuple(notes),
        )

    reasons = (
        "; ".join(
            f"{b.label}: {raw.answer}" for b, _, raw, _ in runs if raw.answer != "not-run"
        )
        or "no solver could be invoked"
    )
    return ObligationEscalation(
        index=sl.index,
        outcome=OB_UNKNOWN,
        detail=f"solver escalation did not decide ({reasons}); a timeout is never a VERIFIED",
        invocations=invocations,
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
    """
    unknown = [o for o in propagation.obligations if o.status == "unknown"]
    if not unknown:
        return Escalation(records=(), notes=())
    backends, missing = _backends_for(config)
    if not backends:
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
        )
    env = interval_env(closed)
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
        stamp_sink: list[SolverStamp] = []
        try:
            record = _dispatch_obligation(item, config, backends, missing, stamp_sink)
        except (SolverDisagreement, EmissionInfidelityError):
            raise  # loud by design
        except Exception as e:  # noqa: BLE001 — guard rule: degrade, quoted
            # defensive: a failure on a validated slice is a bug, but
            # mid-analysis the guard rule still applies — UNKNOWN, quoted.
            # Invocations that completed before the error still happened and
            # still ride into the record (audit F5: the verdict must never
            # claim "no solver invoked" over dropped stamps).
            reason = f"escalation attempted; internal error: {type(e).__name__}: {e}"
            record = ObligationEscalation(
                index=item.index,
                outcome=OB_UNKNOWN,
                detail=reason,
                invocations=tuple(stamp_sink),
                witness=None,
                notes=(f"assert #{item.index}: {reason}",),
            )
        records.append(record)
    return Escalation(records=tuple(records), notes=())


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


def make_solver_verdict(
    closed: ir.ClosedJaxpr,
    propagation: Propagation,
    escalation: Escalation,
    *,
    stelling_version: str,
    jax_version: str,
    precision_config: str,
    device_class: str = _DEVICE_CLASS_DEFAULT,
) -> Verdict:
    """Assemble a verdict from interval propagation plus solver escalation.

    A separate assembly path: the public no-solver path
    (:func:`stelling.verdict.make_verdict`) is untouched. Status: REFUTED
    if any obligation is violated (set-level interval) or solver-refuted
    with a replayed witness; VERIFIED only if every obligation is
    discharged (by interval or by solver); else UNKNOWN.
    """
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
    if status == "VERIFIED" and not nonvacuity.startswith("checked"):
        notes = notes + (
            f"nonvacuity {nonvacuity.split(' — ')[0]}: this VERIFIED may be "
            f"vacuous — the declared set is not tied to the incident's data",
        )

    invocations = escalation.invocations
    if invocations:
        solver: SolverStamp | tuple[SolverStamp, ...] = invocations
    elif not [o for o in propagation.obligations if o.status == "unknown"]:
        solver = solver_absent(
            "no solver invoked: every obligation was decided by outward-rounded "
            "interval arithmetic alone; escalation had nothing to do"
        )
    else:
        # reachable only when zero invocations happened anywhere — records
        # carrying stamps route through the tuple branch above, so this
        # wording can never mask an actual invocation (audit F5).
        solver = solver_absent(
            "no solver invoked: escalation completed no invocation (solver "
            "unavailable, every unknown obligation declined, or a failure "
            "before any invocation; the notes carry the reasons)"
        )

    stamp = Stamp(
        stelling_version=stelling_version,
        jax_version=jax_version,
        query_content_hash=closed.content_hash(),
        arithmetic_mode=ARITHMETIC_MODE_INTERVAL,
        semantics=SEMANTICS_REAL,
        precision_config=precision_config,
        device_class=device_class,
        solver=solver,
        nonvacuity=nonvacuity,
        transfer_tiers=propagation.transfers_used,
        transfer_provenance=tuple(
            (p, "core") for p, _ in propagation.transfers_used
        ),
        assumptions=tuple(
            sorted({*propagation.assumptions, REAL_CONVENTION_ASSUMPTION})
        ),
        coverage=propagation.coverage.summary(),
    )
    witnesses = tuple(
        r.witness for r in escalation.records if r.witness is not None
    )
    return Verdict(
        status=status,
        obligations=obligations,
        stamp=stamp,
        notes=notes,
        witnesses=witnesses,
    )
