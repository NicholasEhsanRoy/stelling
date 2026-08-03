# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Emit a runnable file that reproduces a witness through the real program.

**Why this exists.** Every other check stelling performs is a check of
stelling by stelling. The interval rows, the SMT emission and the
exact-rational replay all rest on the same transcription of the program,
so a defect in that transcription is re-derived identically by each of
them — :class:`stelling.verdict.Witness`'s own docstring records the
measured case where an adversarial audit produced a witness on a
trivially true property and replay confirmed it, "because both faces
asked the same wrong question". The ONE leg that shares no code with
either is **executing the witness through the real program**. This module
makes that leg a file, by construction, so it exists whether or not
anyone remembers to write it.

Hence the hard rule this module is built around: **the emitted file must
never import stelling.** A reproducer that reached back into the tool
would be the tool checking itself again, and the whole value would be
gone. The file imports the caller's own module, ``jax`` and ``numpy``,
and nothing of ours; :func:`reproducer_source` asserts that about its own
output text before returning it.

**The execution result is three-valued, and it is evidence, not a
verdict.**

* :data:`CONFIRMED` — the property is false at the witness under
  execution, in the program's own dtype, in at least one of the execution
  modes the file runs (eager and ``jax.jit``).
* :data:`DIVERGED` — it is false in ℝ (which the exact-rational replay
  established) and TRUE in the program's own dtype, in **every** mode
  that ran. A finding about the real/float gap. **Never a failed
  check**: nothing here is wrong, and the emitted file says so in those
  words.
* :data:`UNREACHABLE` — the witness lies at a point of the declared
  envelope that the caller's own precondition excludes. A
  caller-precondition result: the declaration was wider than the callers
  are, which is a statement about the declaration and not about the
  program.

None of the three is a verdict. The verdict is
:class:`stelling.verdict.Verdict`, it was decided elsewhere, and this
module never changes it — the reproducer is evidence *about* a verdict,
and merging the two would destroy exactly the plan-independent /
plan-dependent distinction that makes executing the witness worth
anything.

**WHY UNREACHABLE IS A DECLARED PREDICATE AND NEVER A MEASUREMENT.**
This project has answered the reachability question wrong twice, and both
times the same way: it MEASURED a span and reported it as the reachable
set.

* ``row7`` — the node's ``initial_state()`` was outside the declared
  ``T ∈ [10, 100]``, and that was read as "the node never occupies the
  envelope". The measurement was right and the inference was not:
  ``initial_state()`` is a starting point, the obligation is posed on the
  output of ``update()``, and a *driven* trajectory reaches ``[0, 100]``
  — entirely inside the box. The span looked degenerate only because the
  node was stepped with no boundary inputs, and this node echoes its own
  boundaries, so from zeros it stays zero forever.
* ``RigidBody`` — a velocity span of ``[-1.962, 0]`` was measured over
  200 steps and recorded as the reachable set. ``200 × 0.001 × 9.81 =
  1.962`` exactly: a free-fall artifact of the trajectory length that was
  chosen. At 100 000 steps the same node spans ``[-9805.9, 0]``. Velocity
  there is a pure accumulator — no drag, no clamp, no collision — so
  **there is no reachable span to compare against at all**, and any finite
  envelope on it is a modelling choice rather than a discovery.

The lesson both converge on is the one this module is built to: **a
measured span is "reachable under the trajectory I ran", never
"reachable in all operation".** So nothing here measures anything. The
author supplies a *predicate* over the declared inputs and, with it, the
*structural argument* that licenses it — the shape of argument that
settled the one genuinely unreachable case on record ("wall correction
factors are ~1 by construction, so a box four orders below their natural
value is not a caller's range"). :class:`Subject` requires exactly one of
``precondition`` / ``no_precondition_reason``, in the shape
:class:`stelling.contracts.Contract` requires exactly one of ``ensures``
/ ``no_ensures_reason``, and a declared precondition must carry its
reason. The emitted file prints that reason **beside every UNREACHABLE it
reports**, so the claim can never travel without the argument it rests
on. An author who has not thought about the question has to say so out
loud instead, and the file quotes that too.

The stakes are SOUNDNESS.md's, verbatim: a box correctly declared but
never occupied "manufactures a counterexample ... **A false
counterexample is the output shape a user trusts most.**"
"""

from __future__ import annotations

import functools
import json
import os
import string
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

__all__ = [
    "CONFIRMED",
    "DIVERGED",
    "EXECUTION_MODES",
    "EXECUTION_RESULTS",
    "Emission",
    "NOT_EXECUTED_EXIT",
    "RESULT_EXIT",
    "SCHEMA",
    "SIDECAR_KEYS",
    "ReproducerError",
    "Subject",
    "UNREACHABLE",
    "reproducer_source",
    "write_reproducer",
]

# ── the three-valued execution result ────────────────────────────────────────
#
# Tokens, not booleans, and deliberately not the verdict statuses: a reader
# or a tally that meets one of these must not be able to mistake it for
# VERIFIED/REFUTED/UNKNOWN — the same reasoning that gave the contracts
# layer's ensures face its own DECLARED token.
CONFIRMED = "CONFIRMED"
DIVERGED = "DIVERGED"
UNREACHABLE = "UNREACHABLE"
EXECUTION_RESULTS = (CONFIRMED, DIVERGED, UNREACHABLE)

# EXIT STATUS IS NOT A JUDGMENT. All three results exit 0, because a nonzero
# status is how CI renders "this check failed" and two of the three are not
# failures of anything. The one nonzero code marks "there is no execution
# result at all": the target could not be constructed, so nothing ran.
RESULT_EXIT = 0
NOT_EXECUTED_EXIT = 3

# ── the published sidecar schema ─────────────────────────────────────────────
#
# A CI coverage line and an external soak parse this, so it is a PUBLISHED
# SURFACE: adding a key is compatible, removing or retyping one is not, and
# either way the integer in SCHEMA moves. It is small on purpose. Every
# field is one an outside consumer cannot re-derive for itself, and nothing
# here is a rendering — the prose all lives in the emitted file's own
# output, where changing it breaks nobody's parser.
SCHEMA = "stelling.reproducer/1"

SIDECAR_KEYS = (
    "schema",       # str  — SCHEMA above
    "stelling",     # str  — the version that emitted this
    "jax",          # str  — the jax that traced the query
    "query",        # str  — ir.ClosedJaxpr.content_hash() of the judged query
    "contract",     # str  — Subject.name
    "verdict",      # str  — VERIFIED | REFUTED | UNKNOWN | DECLINED
    "obligation",   # int  — the assert index the witness is for
    "relation",     # str  — "<=" | ">=" | "<" | ">"
    "fragment",     # str|null — the SMT logic emitted, or null (no solver)
    "equations",    # int|null — equations in the traced query
    "envelope",     # list — one {name, shape, dtype, lo, hi} per declaration
    "witness",      # obj  — {input name: exact rational string}
    "execution",    # obj  — {result, detail, reachable, lhs, rhs, modes}
)

# ``execution.modes`` maps an execution mode to whether the ASSERTED
# COMPARISON HOLDS there (true/false, or null if that mode did not run).
# It is a schema field rather than prose because the modes measurably
# disagree: XLA's algebraic simplifier rewrites ``(1 + x) - 1`` to ``x``,
# so a violation binary64 absorbs eagerly is present in the compiled
# program — measured on jax 0.11.0, in this repository's own DIVERGED
# probe. A consumer that cannot see which mode showed what cannot tell
# that case from a clean one.
EXECUTION_MODES = ("eager", "jit")

_RELATIONS = ("<=", ">=", "<", ">")

# The declaration dtypes the obligation layer admits at all
# (stelling.obligation._FLOAT_INPUT_DTYPES). Named again here because the
# reproducer has its own reason to care: it must build a concrete array of
# this dtype from an exact rational, which is only meaningful for a float.
_DTYPES = ("float32", "float64")


class ReproducerError(ValueError):
    """A reproducer was asked for that could not be emitted honestly."""


# ── the subject ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Subject:
    """The user's own code, factored so a file that does not import
    stelling can call it.

    ``fn`` takes one argument per declaration, in declaration order, and
    returns ``(lhs, rhs)`` — the two sides of the asserted comparison.
    Returning the two sides rather than a bool is what lets the emitted
    file print BOTH of them: "it failed" is not evidence, "101.0 against
    100.0" is. ``rhs`` may be a plain Python number; a declared bound is
    the common case.

    ``fn`` must be importable by ``__module__``/``__qualname__`` and be
    the very object found there. A lambda, a nested function, a method
    bound to an instance built at run time — none can be named in a file,
    and :func:`write_reproducer` emits a file that says which one it met
    rather than one that fails obscurely at its import line.

    ``precondition``, when given, takes the same arguments and returns a
    truthy value iff a caller can produce that point. It is a **claim**,
    and ``precondition_reason`` is the argument for it — printed beside
    every UNREACHABLE the emitted file reports. Read this module's
    docstring before writing one: the two times this project got the
    reachability question wrong, it had measured a span and reported it as
    the reachable set, and a predicate derived that way carries the same
    defect with a stronger-looking output shape.

    Exactly one of ``precondition`` / ``no_precondition_reason`` must be
    populated. An author who declares no precondition states, in the
    record and in the emitted file, that they mean every point of the
    declared envelope to be producible.
    """

    name: str
    fn: Callable
    relation: str
    # one (shape, dtype, (lo, hi)) per declaration, in declaration order —
    # exactly any_array's argument triple, passed through unconverted so
    # the declaration layer judges the caller's own bound objects rather
    # than a copy this layer rounded (the measured hole any_array's own
    # comment records)
    declarations: tuple
    precondition: Callable | None = None
    precondition_reason: str = ""
    no_precondition_reason: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ReproducerError("Subject.name must be populated")
        if not callable(self.fn):
            raise ReproducerError(
                f"Subject.fn must be callable, got {type(self.fn).__name__}"
            )
        if self.relation not in _RELATIONS:
            raise ReproducerError(
                f"Subject.relation must be one of {sorted(_RELATIONS)}, got "
                f"{self.relation!r} — the emitted file compares two sides, so "
                f"the relation between them has to be one it can write down"
            )
        if not self.declarations:
            raise ReproducerError(
                "Subject.declarations is empty: a witness is a point of a "
                "declared envelope, and there is no envelope here"
            )
        for k, decl in enumerate(self.declarations):
            if not (isinstance(decl, (tuple, list)) and len(decl) == 3):
                raise ReproducerError(
                    f"Subject.declarations[{k}] must be a "
                    f"(shape, dtype, (lo, hi)) triple — any_array's own "
                    f"argument shape — got {decl!r}"
                )
            _, dtype, bounds = decl
            if dtype not in _DTYPES:
                raise ReproducerError(
                    f"Subject.declarations[{k}] declares dtype {dtype!r}; the "
                    f"reproducer builds a concrete array from an exact "
                    f"rational, which is only meaningful for {_DTYPES}"
                )
            if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
                raise ReproducerError(
                    f"Subject.declarations[{k}] bounds must be a (lo, hi) "
                    f"pair, got {bounds!r}"
                )
        if (self.precondition is None) == (not self.no_precondition_reason):
            raise ReproducerError(
                "exactly one of Subject.precondition / "
                "Subject.no_precondition_reason must be populated. The "
                "question 'can a caller actually produce this point?' has no "
                "safe default: a box correctly declared but never occupied "
                "manufactures a counterexample, and a false counterexample is "
                "the output shape a user trusts most (SOUNDNESS.md). Give a "
                "predicate over the declared inputs with the argument that "
                "licenses it, or state why every point of the declared "
                "envelope is producible."
            )
        if self.precondition is not None:
            if not callable(self.precondition):
                raise ReproducerError(
                    "Subject.precondition must be callable or None"
                )
            if not self.precondition_reason.strip():
                raise ReproducerError(
                    "Subject.precondition_reason must be populated whenever a "
                    "precondition is declared. UNREACHABLE is a claim that no "
                    "caller produces a point, and this project has twice "
                    "reached that claim from a measured trajectory span, "
                    "which licenses no such thing (see this module's "
                    "docstring: row7 and RigidBody). The reason travels with "
                    "every UNREACHABLE the emitted file prints, so the claim "
                    "cannot circulate without the argument it rests on."
                )
        elif not self.no_precondition_reason.strip():
            raise ReproducerError(
                "Subject.no_precondition_reason must be a populated statement"
            )
        for field_name in ("precondition_reason", "no_precondition_reason"):
            value = getattr(self, field_name)
            if "\n" in value or "\r" in value:
                # the same refusal EnsuresFace applies, for the same reason:
                # this text is printed at column 0 of the emitted file's
                # output, where an embedded newline could forge a
                # result-looking line ("== CONFIRMED")
                raise ReproducerError(
                    f"Subject.{field_name} must be a single physical line — "
                    f"it is printed at column 0 of the reproducer's output, "
                    f"where an embedded newline could forge a result line"
                )

    @property
    def harness(self):
        """The zero-arg harness to hand :func:`stelling.preconditions.check`.

        Generated here rather than written twice: the query stelling
        judges and the call the emitted file makes then come from ONE
        object, and there is no second place for them to drift apart.
        :func:`write_reproducer` re-traces this and compares content
        hashes with the verdict's stamp before emitting anything, so the
        agreement is checked rather than assumed.
        """
        subject = self

        def harness():
            from stelling.harness import any_array

            values = [
                any_array(shape, dtype, bounds)
                for shape, dtype, bounds in subject.declarations
            ]
            lhs, rhs = subject.fn(*values)
            return (_assert_relation(subject.relation, lhs, rhs),)

        return harness


def _assert_relation(relation: str, lhs, rhs):
    from stelling.harness import assert_

    if relation == "<=":
        return assert_(lhs <= rhs)
    if relation == ">=":
        return assert_(lhs >= rhs)
    if relation == "<":
        return assert_(lhs < rhs)
    return assert_(lhs > rhs)


# ── what could not be constructed ────────────────────────────────────────────


def _import_problem(fn, what: str) -> str | None:
    """Why ``fn`` cannot be named in a file that imports it, or None.

    Each measured shape gets its own sentence, because "ImportError" five
    frames deep is what this function exists to replace. The last branch
    is the important one: a module-level name that resolves to a
    DIFFERENT object — a decorator, a reassignment, a monkeypatch — would
    silently reproduce against something else, which is worse than not
    reproducing at all.
    """
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname:
        return (
            f"{what} {fn!r} carries no __module__/__qualname__, so there is "
            f"no name for an importing file to ask for"
        )
    if module == "__main__":
        # It resolves HERE and would resolve in the reproducer too — to the
        # reproducer's own module, which is a different object. Refused
        # rather than left to fail at run time, because the failure would
        # arrive in the emitted file and look like the reproducer's fault.
        return (
            f"{what} {qualname} is defined in the __main__ script, and "
            f"'__main__' names whatever module is being run — in the "
            f"reproducer that is the reproducer itself, not your script. "
            f"Move it into an importable module"
        )
    if getattr(fn, "__self__", None) is not None:
        return (
            f"{what} {module}.{qualname} is a method BOUND to a "
            f"{type(fn.__self__).__name__} instance built at run time; the "
            f"emitted file can import the name but not that instance. Wrap "
            f"the construction in a module-level function, so the file can "
            f"build the object the same way you did"
        )
    if qualname.endswith("<lambda>"):
        return (
            f"{what} is a lambda defined in {module}: a lambda has no "
            f"importable name. Give it a def at module level"
        )
    if "<locals>" in qualname:
        return (
            f"{what} {module}.{qualname} is defined inside another function, "
            f"so it has no importable name. Move it to module level — a "
            f"factory can still build its arguments"
        )
    try:
        import importlib

        obj = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001 — report, never break the emission
        return (
            f"{what}'s module {module!r} could not be imported here "
            f"({type(e).__name__}: {e}), so the emitted file could not "
            f"import it either"
        )
    sentinel = object()
    for part in qualname.split("."):
        obj = getattr(obj, part, sentinel)
        if obj is sentinel:
            return (
                f"{what} {module}.{qualname} does not resolve to anything: "
                f"the emitted file's import would fail at {part!r}"
            )
    if obj is not fn:
        return (
            f"{what} {module}.{qualname} resolves to a DIFFERENT object than "
            f"the one passed ({obj!r} is not {fn!r}) — a decorator, a "
            f"reassignment or a monkeypatch. Reproducing against whatever "
            f"that name holds would measure a different program"
        )
    return None


# ── emission ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Emission:
    """What :func:`write_reproducer` wrote, and what it could not build."""

    path: str
    sidecar_path: str
    source: str
    # None when the file can really run; otherwise the sentence the file
    # prints, at its top, in place of an execution result
    unconstructible: str | None = None

    @property
    def runnable(self) -> bool:
        return self.unconstructible is None


@functools.lru_cache(maxsize=1)
def _stelling_sha() -> str:
    """The stelling tree's git sha, or an honest statement that there is
    none. ``0.1.0`` identifies a release; a sha identifies the code that
    produced this witness, which is what a reader chasing it needs.

    Cached per process, for a reason and not only for the 3.9 ms: a batch
    that writes its reproducers INTO the repository would flip its own
    dirty flag partway through, and one provenance reading per run is the
    honest record of one run.
    """
    import stelling

    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(stelling.__file__)))
    )
    try:
        head = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return f"unknown (not a git checkout: {root})"
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance must never break emission
        return "unknown (git unavailable)"
    return head.stdout.strip() + (" (tree dirty)" if dirty else "")


def _fragment_of(stamp) -> str | None:
    """The SMT logic the escalation actually emitted, read off the stamp's
    own option set — the record of the ask, not a re-derivation."""
    solvers = (
        stamp.solver if isinstance(stamp.solver, tuple) else (stamp.solver,)
    )
    logics = {
        dict(s.options or ()).get("set-logic") for s in solvers if s.invoked
    }
    logics.discard(None)
    return "+".join(sorted(logics)) if logics else None


def _equation_count(stamp) -> int | None:
    head = stamp.coverage.split(" eqns")[0]
    return int(head) if head.isdigit() else None


def _witness_for(verdict, obligation_index: int | None):
    if verdict.status != "REFUTED":
        raise ReproducerError(
            f"a reproducer reproduces a witness, and a {verdict.status} "
            f"verdict carries none. Nothing is emitted, rather than a file "
            f"that would print an empty point"
        )
    if not verdict.witnesses:
        raise ReproducerError(
            "this REFUTED is set-level: an obligation is definitely false "
            "over the declared set, but no concrete point was produced — the "
            "verdict says so itself ('Not a witness; not a counterexample to "
            "the program'). There is nothing to execute"
        )
    if obligation_index is None:
        return verdict.witnesses[0]
    for w in verdict.witnesses:
        if w.obligation_index == obligation_index:
            return w
    raise ReproducerError(
        f"no witness for assert #{obligation_index}; this verdict carries "
        f"witnesses for "
        f"{sorted(w.obligation_index for w in verdict.witnesses)}"
    )


def _declaration_of(name: str, subject: Subject) -> tuple[int, int]:
    """``(declaration index, flat element index)`` for a witness value name.

    The naming is :class:`stelling.obligation.SliceInput`'s published
    contract — ``x{k}`` for a scalar declaration, ``x{k}_{i}`` for element
    ``i`` of an array one, ``k`` the declaration's order in the query —
    and this READS that contract rather than re-deriving the declaration
    order from the IR. A second walk over the query would be a second
    thing to keep correct, which is the lesson `_barred_primitives`
    already paid for.
    """
    body = name[1:] if name.startswith("x") else ""
    head, _, tail = body.partition("_")
    if not head.isdigit() or (tail and not tail.isdigit()):
        raise ReproducerError(
            f"witness value name {name!r} is not in the published "
            f"declaration naming (x{{k}} or x{{k}}_{{i}})"
        )
    k = int(head)
    if k >= len(subject.declarations):
        raise ReproducerError(
            f"witness value {name!r} names declaration #{k}, but the subject "
            f"declares {len(subject.declarations)}"
        )
    return k, int(tail) if tail else 0


def _envelope(subject: Subject) -> list[dict]:
    return [
        {
            "name": f"x{k}",
            "shape": [int(d) for d in shape],
            "dtype": dtype,
            "lo": float(lo),
            "hi": float(hi),
        }
        for k, (shape, dtype, (lo, hi)) in enumerate(subject.declarations)
    ]


def _point(subject: Subject, witness) -> tuple[list[list[str]], list[str]]:
    """The witness as one exact-rational string per element of every
    declaration, plus a disclosure line per element the model left free.

    A solver model names only what the obligation's slice reached, and an
    element the obligation does not touch is a genuine don't-care — the
    dispatch layer fills those from the declared box before replay
    (``stelling.solvers._complete_values``) and discloses each one. The
    emitted file has to build a whole array, so it fills the same way and
    says which elements it invented.
    """
    got = dict(witness.values)
    for name in got:
        _declaration_of(name, subject)  # refuse a name from another query
    values: list[list[str]] = []
    disclosures: list[str] = []
    for k, (shape, _dtype, (lo, hi)) in enumerate(subject.declarations):
        size = 1
        for d in shape:
            size *= int(d)
        elements = []
        for i in range(size):
            name = f"x{k}" if not tuple(shape) else f"x{k}_{i}"
            if name in got:
                elements.append(str(Fraction(got[name])))
                continue
            fill = lo if float(lo) != float("-inf") else (
                hi if float(hi) != float("inf") else 0
            )
            elements.append(str(Fraction(fill)))
            disclosures.append(
                f"{name} was not constrained by the model (the obligation "
                f"does not reach it); filled with {Fraction(fill)} from its "
                f"declared box, exactly as the replay did"
            )
        values.append(elements)
    return values, disclosures


def _tool_leak(fn) -> str | None:
    """Disclosure when the TARGET's own module imports stelling.

    The emitted file never imports stelling, and that is checked on its
    own text — but it does import the module the target lives in, and if
    THAT module reaches the tool then the tool is loaded in the
    reproducer's process after all. Nothing here uses it, so the executed
    computation is still independent; what is lost is the ability to run
    the file where stelling is absent, which is exactly how the
    independence gets demonstrated. Measured the hard way: the first
    version of this feature's own acceptance put its targets in the
    harness module and every reproducer stopped at the import line.

    A disclosure and not a refusal: the target module is the user's, the
    condition is easy to fix by moving the program out of the harness
    module, and refusing to emit would trade a working file for a lecture.
    """
    import inspect

    module = getattr(fn, "__module__", None)
    if not module:
        return None
    try:
        import importlib

        source = inspect.getsource(importlib.import_module(module))
    except Exception:  # noqa: BLE001 — a disclosure must never break emission
        return None
    for line in source.splitlines():
        if line.startswith(("import stelling", "from stelling")):
            return (
                f"the target's module {module} imports stelling at module "
                f"scope ({line.strip()!r}), so running this file loads the "
                f"tool even though this file never calls it. The executed "
                f"computation is still independent; what you lose is being "
                f"able to run this where stelling is not installed, which is "
                f"how that independence gets shown. Move the program out of "
                f"the harness module"
            )
    return None


def _flat_witness(subject: Subject, values: list[list[str]]) -> dict:
    out = {}
    for k, ((shape, _d, _b), elements) in enumerate(
        zip(subject.declarations, values)
    ):
        for i, text in enumerate(elements):
            out[f"x{k}" if not tuple(shape) else f"x{k}_{i}"] = text
    return out


def reproducer_source(
    verdict,
    subject: Subject,
    *,
    witness,
    query_hash: str,
    fragment: str | None,
    equations: int | None,
    x64: bool,
    unconstructible: str | None = None,
) -> str:
    """The emitted file's text. It never imports stelling — checked here,
    on the artefact, rather than trusted to the template."""
    envelope = _envelope(subject)
    values, disclosures = _point(subject, witness)
    leak = _tool_leak(subject.fn)
    if leak is not None:
        disclosures = list(disclosures) + [leak]
    sidecar = {
        "schema": SCHEMA,
        "stelling": _version(),
        "jax": _jax_version(),
        "query": query_hash,
        "contract": subject.name,
        "verdict": verdict.status,
        "obligation": witness.obligation_index,
        "relation": subject.relation,
        "fragment": fragment,
        "equations": equations,
        "envelope": envelope,
        "witness": _flat_witness(subject, values),
    }
    missing = set(SIDECAR_KEYS) - set(sidecar) - {"execution"}
    if missing:  # a schema key lost in a refactor must not ship silently
        raise ReproducerError(
            f"the sidecar payload is missing published schema key(s) "
            f"{sorted(missing)}; {SCHEMA} is a parsed surface"
        )
    payload = {
        "witness_elements": values,
        "x64": x64,
        "target_module": getattr(subject.fn, "__module__", None),
        "target_qualname": getattr(subject.fn, "__qualname__", None),
        "precondition_module": (
            getattr(subject.precondition, "__module__", None)
            if subject.precondition is not None
            else None
        ),
        "precondition_qualname": (
            getattr(subject.precondition, "__qualname__", None)
            if subject.precondition is not None
            else None
        ),
        "precondition_reason": subject.precondition_reason,
        "no_precondition_reason": subject.no_precondition_reason,
        "disclosures": disclosures,
        "unconstructible": unconstructible,
    }
    text = string.Template(_TEMPLATE).substitute(
        banner=_banner(sidecar, witness, disclosures, _stelling_sha(), x64),
        sidecar=json.dumps(sidecar, indent=2, sort_keys=True),
        payload=json.dumps(payload, indent=2, sort_keys=True),
        confirmed=CONFIRMED,
        diverged=DIVERGED,
        unreachable=UNREACHABLE,
        result_exit=RESULT_EXIT,
        not_executed_exit=NOT_EXECUTED_EXIT,
    )
    # THE ONE RULE, CHECKED ON THE OUTPUT rather than promised by the
    # template. A reproducer that reaches back into the tool checks the tool
    # with the tool and is worth nothing as independent evidence, so this is
    # a structural refusal at the point of emission, not a comment asking
    # the next author to be careful.
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith(("import stelling", "from stelling")):
            raise ReproducerError(
                f"the emitted reproducer imports stelling at line {lineno} "
                f"({line.strip()!r}) — a file that reaches back into the tool "
                f"checks the tool with the tool"
            )
    return text


def _version() -> str:
    import stelling

    return stelling.__version__


def _jax_version() -> str:
    from stelling._jax_compat import jax_version

    return jax_version()


def _banner(sidecar, witness, disclosures, sha, x64) -> str:
    """The header: query hash, declared envelope, solver and fragment,
    stelling sha, jax version. The provenance a reader needs to know WHICH
    claim this file is evidence about."""
    lines = [
        f"contract         : {sidecar['contract']}",
        f"verdict          : {sidecar['verdict']} "
        f"(assert #{sidecar['obligation']})",
        f"query hash       : {sidecar['query']}",
        f"stelling         : {sidecar['stelling']}  sha {sha}",
        f"jax              : {sidecar['jax']}  (jax_enable_x64={x64})",
        f"solver           : {witness.produced_by}",
        f"fragment         : {sidecar['fragment']}",
        f"equations        : {sidecar['equations']}",
        f"replay           : {witness.replay}",
        "declared envelope:",
    ]
    for d in sidecar["envelope"]:
        lines.append(
            f"  {d['name']}: shape {tuple(d['shape'])} {d['dtype']} in "
            f"[{d['lo']!r}, {d['hi']!r}]"
        )
    if witness.violating_elements:
        lines.append(
            "violating element(s) of the assert operand, per the replay: "
            + ", ".join(str(i) for i in witness.violating_elements)
        )
    for d in disclosures:
        lines.append(f"disclosure       : {d}")
    return "\n".join(lines)


def write_reproducer(
    verdict,
    subject: Subject,
    directory: str,
    *,
    obligation_index: int | None = None,
    closed=None,
    filename: str | None = None,
) -> Emission:
    """Write ``<directory>/<name>.py`` reproducing ``verdict``'s witness.

    ``closed`` is the traced query the verdict is about; when omitted the
    subject's own harness is re-traced. Either way its content hash must
    equal the verdict's stamped one, or nothing is written — a reproducer
    built against a different program than the one that was judged is a
    confident wrong answer, and this project has no use for one.

    The file IS written when the target cannot be constructed. That is
    deliberate (see :attr:`Emission.unconstructible`): a file that states
    plainly "I could not build this, here is exactly what was missing" is
    worth more than no file, and far more than one that dies at an import
    line five frames deep.
    """
    from stelling._jax_compat import trace, x64_enabled

    witness = _witness_for(verdict, obligation_index)
    if closed is None:
        closed = trace(subject.harness)
    stamped = verdict.stamp.query_content_hash
    traced = closed.content_hash()
    if traced != stamped:
        raise ReproducerError(
            f"this verdict is not about this subject's program: the verdict "
            f"stamps query {stamped}, and the subject's harness traces to "
            f"{traced}. Emitting anyway would produce a file that executes "
            f"one program and quotes another program's verdict"
        )
    unconstructible = _import_problem(subject.fn, "the target")
    if unconstructible is None and subject.precondition is not None:
        unconstructible = _import_problem(
            subject.precondition, "the caller precondition"
        )
    text = reproducer_source(
        verdict,
        subject,
        witness=witness,
        query_hash=stamped,
        fragment=_fragment_of(verdict.stamp),
        equations=_equation_count(verdict.stamp),
        x64=x64_enabled(),
        unconstructible=unconstructible,
    )
    stem = filename or (
        f"reproduce_{_slug(subject.name)}_assert{witness.obligation_index}"
    )
    if stem.endswith(".py"):
        stem = stem[:-3]
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, stem + ".py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return Emission(
        path=path,
        sidecar_path=os.path.join(directory, stem + ".json"),
        source=text,
        unconstructible=unconstructible,
    )


def _slug(name: str) -> str:
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    return out.strip("_") or "subject"


# ── the emitted file ─────────────────────────────────────────────────────────
#
# ONE template, substituted with JSON data and nothing executable: every
# value the file needs is a literal it loads, so there is no code
# generation to get subtly wrong and no user expression pasted into a
# program text. string.Template, not str.format, so the emitted file's own
# f-strings need no brace doubling and stay readable in this source.

_TEMPLATE = '''#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""stelling witness reproducer — GENERATED. Do not edit; regenerate.

${banner}

WHAT THIS FILE IS. It builds the witness point above, calls YOUR OWN
function with it, evaluates the asserted comparison, and prints both
sides. It does not import stelling, on purpose: every other check of this
witness shares stelling's transcription of your program, so executing the
witness is the only leg that can catch a defect in that transcription. A
reproducer that reached back into the tool would be the tool checking
itself with the tool.

WHAT IT REPORTS, AND WHAT THAT IS NOT. The result is one of three, and
none of them is a verdict — the verdict was decided elsewhere and this
file cannot change it:

  ${confirmed}    the comparison is false at the witness under
                 execution, in at least one of the two modes run below.
  ${diverged}     it is false in exact real arithmetic (the replay
                 named above established that) and TRUE in your
                 program's own dtype, in EVERY mode that ran. A finding
                 about the real/float gap. NOT A FAILED CHECK.
  ${unreachable}  the witness lies at a point your own caller
                 precondition excludes. A caller-precondition result:
                 the declared envelope is wider than your callers are.
                 NOT a bug in the program, and not a confirmation of one.

Exit status is ${result_exit} for all three, because a nonzero status is how CI
says "this check failed" and two of the three are not failures of
anything. Exit ${not_executed_exit} means there is no execution result at all: the
target could not be constructed. Read the JSON sidecar, not the status.
"""
import importlib
import json
import os
import sys
from fractions import Fraction

# the published sidecar surface, minus the execution result this run adds
SIDECAR = json.loads(r"""
${sidecar}
""")

# everything else this file needs to run; not part of the schema
PAYLOAD = json.loads(r"""
${payload}
""")


def _sidecar(execution):
    out = os.environ.get("STELLING_REPRODUCER_JSON") or (
        os.path.splitext(os.path.abspath(__file__))[0] + ".json"
    )
    record = dict(SIDECAR)
    record["execution"] = execution
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\\n")
    print(f"\\nsidecar: {out}")
    return record


def _stop(detail):
    """No execution result exists. Say exactly what was missing."""
    print("== NO EXECUTION RESULT")
    print("  This reproducer could not construct its target, so nothing was")
    print("  executed and nothing is claimed here in either direction.")
    print(f"  WHAT COULD NOT BE CONSTRUCTED: {detail}")
    print("  The verdict this file is evidence about is unchanged. It simply")
    print("  has no execution leg until the target can be called from a file.")
    _sidecar({"result": None, "detail": detail, "reachable": None,
              "lhs": None, "rhs": None, "modes": {}})
    return ${not_executed_exit}


def _load(module, qualname):
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _build(np, decl, elements):
    """One declared array, from exact rationals, in the declared dtype."""
    exact = [Fraction(t) for t in elements]
    flat = np.asarray([float(f) for f in exact], dtype=np.dtype(decl["dtype"]))
    rounded = [i for i, f in enumerate(exact) if Fraction(float(flat[i])) != f]
    return flat.reshape(tuple(decl["shape"])), rounded


def _flat(np, v):
    return [float(x) for x in np.asarray(v).reshape(-1)]


def main():
    if PAYLOAD["unconstructible"]:
        return _stop(PAYLOAD["unconstructible"])
    try:
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", PAYLOAD["x64"])
        np = importlib.import_module("numpy")
    except Exception as e:
        return _stop(f"jax/numpy unavailable ({type(e).__name__}: {e})")
    try:
        target = _load(PAYLOAD["target_module"], PAYLOAD["target_qualname"])
    except Exception as e:
        return _stop(
            f"the target {PAYLOAD['target_module']}."
            f"{PAYLOAD['target_qualname']} could not be imported "
            f"({type(e).__name__}: {e})"
        )
    precondition = None
    if PAYLOAD["precondition_qualname"]:
        try:
            precondition = _load(
                PAYLOAD["precondition_module"],
                PAYLOAD["precondition_qualname"],
            )
        except Exception as e:
            return _stop(
                f"the caller precondition "
                f"{PAYLOAD['precondition_module']}."
                f"{PAYLOAD['precondition_qualname']} could not be imported "
                f"({type(e).__name__}: {e})"
            )

    print("== the witness, exactly as the solver produced it")
    args, inexact = [], []
    for decl, elements in zip(SIDECAR["envelope"], PAYLOAD["witness_elements"]):
        try:
            arr, rounded = _build(np, decl, elements)
        except Exception as e:
            return _stop(
                f"the witness value(s) for {decl['name']} could not be built "
                f"as {decl['dtype']} ({type(e).__name__}: {e})"
            )
        args.append(arr)
        for i in rounded:
            inexact.append(f"{decl['name']}[{i}] = {elements[i]}")
        for i, t in enumerate(elements):
            print(f"  {decl['name']}[{i}] = {t}")
        print(f"  {decl['name']} as {decl['dtype']}: {arr!r}")
    if inexact:
        print("  NOTE: these exact values are NOT representable in the")
        print("  declared dtype and were rounded to build the array:")
        for line in inexact:
            print(f"    {line}")
    for d in PAYLOAD["disclosures"]:
        print(f"  disclosure: {d}")

    if precondition is not None:
        try:
            reachable = bool(precondition(*args))
        except Exception as e:
            return _stop(
                f"the caller precondition raised at the witness "
                f"({type(e).__name__}: {e})"
            )
        print(f"\\n== caller precondition holds at the witness: {reachable}")
        print(f"  declared because: {PAYLOAD['precondition_reason']}")
        if not reachable:
            print("\\n== ${unreachable}")
            print("  The witness lies at a point of the DECLARED envelope that")
            print("  your own caller precondition excludes, so no caller")
            print("  produces it. That is a fact about the declaration being")
            print("  wider than your callers are -- NOT a bug in the program,")
            print("  and NOT a confirmation of one. The program was not run.")
            print("  Narrow the declaration, or state the precondition to")
            print("  stelling with assume(), and re-run.")
            print("  This rests entirely on the reason printed above. A span")
            print("  measured from one trajectory is 'reachable under the")
            print("  trajectory I ran' and licenses no unreachability claim;")
            print("  twice in this project's history it was read as one.")
            _sidecar({"result": "${unreachable}", "reachable": False,
                      "lhs": None, "rhs": None, "modes": {},
                      "detail": "the declared caller precondition is false "
                                "at the witness; the program was not "
                                "executed. Licensed by: "
                                + PAYLOAD["precondition_reason"]})
            return ${result_exit}
    else:
        reachable = None
        print("\\n== caller precondition: NONE DECLARED")
        print(f"  {PAYLOAD['no_precondition_reason']}")
        print("  So ${unreachable} was not tested. This run assumes every")
        print("  point of the declared envelope is producible by some caller.")

    print("\\n== executing YOUR function")
    print(f"  {PAYLOAD['target_module']}.{PAYLOAD['target_qualname']}")
    rel = SIDECAR["relation"]

    def evaluate(fn, label):
        """Both sides, and whether the assertion holds, in one mode."""
        try:
            lhs, rhs = fn(*args)
        except Exception as e:
            print(f"\\n  [{label}] not run ({type(e).__name__}: {e})")
            return None, None, None
        lv, rv = np.asarray(lhs), np.asarray(rhs)
        ok = np.asarray(
            {"<=": lv <= rv, ">=": lv >= rv, "<": lv < rv, ">": lv > rv}[rel]
        )
        holds = bool(np.all(ok))
        print(f"\\n  [{label}] lhs = {lv!r}")
        print(f"  [{label}] rhs = {rv!r}")
        print(f"  [{label}] asserted: lhs {rel} rhs   ->  {holds}")
        flat_l, flat_r = _flat(np, lv), _flat(np, rv)
        bad = [int(i) for i, b in enumerate(ok.reshape(-1)) if not b]
        if bad:
            print(f"  [{label}] FALSE at flat element(s): {bad}")
            for i in bad[:8]:
                a = flat_l[i] if i < len(flat_l) else flat_l[0]
                b = flat_r[i] if i < len(flat_r) else flat_r[0]
                print(
                    f"    [{i}]  {a!r} {rel} {b!r}  is False"
                    f"   (margin {a - b:+})"
                )
        return holds, flat_l, flat_r

    # BOTH MODES ARE THE PROGRAM. The compiler is entitled to rewrite the
    # expression, and measurably does: XLA simplifies (1 + x) - 1 to x, so
    # a violation binary64 absorbs eagerly survives compilation. Running one
    # mode and calling it "the program" would report the other one's answer
    # as this one's.
    holds_eager, flat_l, flat_r = evaluate(target, "eager")
    if holds_eager is None:
        return _stop("the target raised at the witness under eager execution")
    holds_jit, _, _ = evaluate(jax.jit(target), "jit")
    modes = {"eager": holds_eager, "jit": holds_jit}
    ran = [m for m, h in modes.items() if h is not None]
    disagree = len({modes[m] for m in ran}) > 1

    if disagree:
        print("\\n  THE TWO MODES DISAGREE, and that is itself the finding:")
        for m in ran:
            print(f"    {m:5s}: the assertion {'HOLDS' if modes[m] else 'is FALSE'}")
        print("    The compiler is allowed to rewrite the expression, so one")
        print("    of these is the program you ship and the other is the")
        print("    program you debug. Compare the lowered HLO before reading")
        print("    either as the answer.")

    if not all(modes[m] for m in ran):
        where = ", ".join(m for m in ran if not modes[m])
        print("\\n== ${confirmed}")
        print(f"  The asserted comparison is FALSE at the witness under {where}")
        print("  execution of your own function. This is the one check of the")
        print("  refutation that shares no code with stelling's emission or")
        print("  its replay.")
        detail = (
            "the asserted comparison is false at the witness under execution ("
            + ", ".join(f"{m}: {'holds' if modes[m] else 'FALSE'}" for m in ran)
            + ")"
        )
        result = "${confirmed}"
    else:
        print("\\n== ${diverged}")
        print("  The comparison HOLDS at the witness in your program's own")
        print(f"  dtype, in every mode run ({', '.join(ran)}), and stelling's")
        print("  exact-rational replay established it is FALSE in exact real")
        print("  arithmetic at the same point.")
        print("  THIS IS NOT A FAILED CHECK. It is a finding about the gap")
        print("  between the reals the verdict is stated over and the floats")
        print("  the program runs in: the violation is real, and smaller than")
        print("  this dtype can represent. Nothing here is wrong, the verdict")
        print("  is unchanged, and its stamped semantics line already says")
        print("  which arithmetic it is a claim about.")
        detail = (
            "false in the reals per the exact-rational replay, true under "
            "execution in " + SIDECAR["envelope"][0]["dtype"] + " ("
            + ", ".join(ran) + ")"
        )
        result = "${diverged}"
    _sidecar({"result": result, "detail": detail, "reachable": reachable,
              "lhs": flat_l, "rhs": flat_r, "modes": modes})
    return ${result_exit}


if __name__ == "__main__":
    sys.exit(main())
'''
