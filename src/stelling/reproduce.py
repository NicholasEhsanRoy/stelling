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

**THE ONE DEFECT THIS MODULE CANNOT HAVE is a file that reports a result
the program does not produce — and a wrongly SILENT file is in the same
family.** A blinded audit of the first build found six, and what they
share is worth stating because the next one will share it too: every one
was a place where the file's answer came from something other than
running the program.

* It built ``numpy`` arrays and handed them to a jax target.
  ``x.at[i].set(v)``, the functional-write idiom every JAX program uses,
  does not exist on ``numpy.ndarray``, so eager raised and the file
  reported "no execution result" for a violation it could have confirmed
  — while ``jax.jit`` executed the same numpy input happily. The inputs
  are jax arrays now, and a mode that raises no longer cancels the mode
  that runs: silence requires BOTH to have failed.
* The published ``lhs``/``rhs`` came from eager even when jit was the
  mode that refuted, so a CONFIRMED published a counterexample that
  SATISFIED its own relation. They now come from a falsifying mode
  whenever one exists, and ``execution.sides_from`` says which.
* A ``closed=`` parameter let a caller supply the traced query the
  content hash was compared against, so the comparison never involved the
  subject: program A's query with subject B emitted a file carrying B's
  name and envelope, A's hash, and a witness outside B's own published
  envelope. The parameter is gone — a gate with a bypass is not a gate.
* Caller text reached the emitted file's *program text*, where a crafted
  contract name printed a result line and exited 0 without executing
  anything. :func:`one_line` is the funnel, :class:`Subject` refuses the
  same at its door, and the output must parse.
* Three importable target shapes — a classmethod, a module-level callable
  instance, a ``functools.partial`` — were reported uncallable, so the
  file said nothing about programs it could have run.
* The exact rational was rounded twice on the way to a narrow dtype,
  which moves the executed value by an ulp and can flip the result at a
  bound.
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
    "one_line",
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
    "witness_filled",  # list — witness names this layer invented, not solved
    "stelling_sha",    # str  — the tree that produced the verdict
    "x64",             # bool — jax_enable_x64 at trace and at execution
    "execution",    # obj  — {result, detail, reachable, lhs, rhs, modes,
                    #         sides_from}
)

# ``execution.modes`` maps an execution mode to whether the ASSERTED
# COMPARISON HOLDS there (true/false, or null if that mode did not run).
# It is a schema field rather than prose because the modes measurably
# disagree: XLA's algebraic simplifier rewrites ``(1 + x) - 1`` to ``x``,
# so a violation binary64 absorbs eagerly is present in the compiled
# program — measured on jax 0.11.0, in this repository's own DIVERGED
# probe. A consumer that cannot see which mode showed what cannot tell
# that case from a clean one. EVERY mode is always present as a key, and
# ``null`` means it did not run: a path that emitted ``{}`` instead handed
# a consumer following this comment a KeyError.
EXECUTION_MODES = ("eager", "jit")

# ``execution.sides_from`` names the mode ``lhs``/``rhs`` were read in, and
# it exists because publishing the wrong one is worse than publishing
# nothing. On a mode disagreement the predecessor published the EAGER
# sides beside a CONFIRMED, so a consumer checking ``lhs <= rhs`` against
# ``relation`` found the published counterexample SATISFYING the relation
# while the violating numbers lived only on stdout. The sides now come
# from a mode where the assertion is false whenever one exists.

_RELATIONS = ("<=", ">=", "<", ">")

# the docstring delimiter of the emitted file; caller text may not carry it
_TRIPLE = '"' * 3

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
        if any(c in self.name for c in "\n\r") or _TRIPLE in self.name:
            # the name reaches the emitted file's docstring: a crafted one
            # produced a file that printed a result and exited 0 without
            # executing anything, and a stray triple quote produced a file
            # that would not parse. :func:`one_line` neutralises both at
            # the funnel; this refuses them at the door as well.
            raise ReproducerError(
                "Subject.name must be a single physical line and must not "
                "contain a triple quote — it is written into the emitted "
                "file's docstring, where a newline can start a statement "
                "and a quote can end the docstring"
            )
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


# ── caller text, and the one funnel it passes through ────────────────────────


def one_line(text: str) -> str:
    """THE FUNNEL every piece of caller text passes through before it can
    reach the emitted file's source.

    A crafted ``Subject.name`` produced a file that printed a result line
    and exited 0 without executing anything; a contract name containing a
    triple quote produced a file that would not parse. Both are one defect
    — free text reaching a program text — and both are closed here rather
    than at each of the six slots that carry such text (the contract name,
    the solver's ``produced_by``, the replay sentence, the disclosures, the
    reasons, the unconstructible sentence).

    Newlines and control characters become spaces, so nothing a caller
    writes can start a line; the triple quote that would end the docstring
    is neutralised; a backslash that could escape the closing quote is
    replaced. :class:`Subject` also refuses newlines in its own text fields
    at construction — one invariant, two mechanisms, as elsewhere here.
    """
    out = "".join(
        " " if (ord(ch) < 32 or ord(ch) == 127) else ch for ch in str(text)
    )
    out = out.replace(_TRIPLE, "'''").replace("\\", "/")
    return out.strip() or "(empty)"


# ── what could not be constructed ────────────────────────────────────────────


def _resolve_target(fn, what: str) -> tuple[str | None, str | None, str | None]:
    """``(module, qualname, problem)`` — the name an importing file can ask
    for, or why there is none.

    THE TEST IS IDENTITY, NOT SHAPE. What matters is whether some
    module-level name resolves to *this exact object*, and the shapes that
    do are wider than "a plain def": a ``classmethod`` accessed off its
    class, a module-level callable INSTANCE (the shape a library gives a
    configured operator object, and the one this module's own docs
    recommend for fixture work), and a ``functools.partial`` bound at
    module level are all importable, and all three were reported
    uncallable by the predecessor, which asked about ``__self__`` and
    ``__qualname__`` instead. A reproducer that reports
    "no execution result" for a target it could have executed is a
    wrongly-silent file, which is a defect in the same family as a wrong
    one.

    So: try the object's own ``__module__``/``__qualname__`` path first,
    and if that does not land on ``fn``, SCAN the plausible module's own
    namespace for a module-level name bound to it. Only when neither finds
    it is there a problem, and each shape gets its own sentence.
    """
    import importlib

    sentinel = object()

    def same(a, b):
        """Identity, allowing for bound methods.

        ``C.method`` builds a NEW bound-method object on every attribute
        access, so a plain ``is`` test says a classmethod does not resolve
        to itself. What matters is whether the name reaches the same
        underlying function on the same owner.
        """
        if a is b:
            return True
        fa, fb = getattr(a, "__func__", None), getattr(b, "__func__", None)
        return (
            fa is not None
            and fa is fb
            and getattr(a, "__self__", None) is getattr(b, "__self__", None)
        )

    def resolve(module, qualname):
        try:
            obj = importlib.import_module(module)
        except Exception:  # noqa: BLE001 — a missing module is not a crash
            return sentinel
        for part in qualname.split("."):
            obj = getattr(obj, part, sentinel)
            if obj is sentinel:
                return sentinel
        return obj

    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    # __main__ FIRST, and independently of qualname: an instance does not
    # inherit its class's __qualname__ (it is not in the class __dict__), so
    # a check gated on both string attributes missed the callable-instance
    # shape entirely. It resolves HERE and resolves to something ELSE in the
    # reproducer — that file's own __main__ — so it is refused rather than
    # left to fail at run time, where it would look like the file's fault.
    if "__main__" in (module, getattr(type(fn), "__module__", None)):
        return None, None, (
            f"{what} {qualname or type(fn).__name__} is defined in the "
            f"__main__ script, and '__main__' names whatever module is being "
            f"run — in the reproducer that is the reproducer itself, not "
            f"your script. Move it into an importable module"
        )
    if isinstance(module, str) and isinstance(qualname, str):
        if "<locals>" not in qualname and same(resolve(module, qualname), fn):
            return module, qualname, None

    # Not reachable by its own name. Look for a module-level name bound to
    # this exact object, in the modules it could plausibly live in.
    candidates = []
    for cand in (
        module,
        getattr(type(fn), "__module__", None),
        getattr(getattr(fn, "func", None), "__module__", None),
    ):
        if isinstance(cand, str) and cand not in candidates and cand not in (
            "__main__", "builtins", "functools"
        ):
            candidates.append(cand)
    for cand in candidates:
        try:
            mod = importlib.import_module(cand)
        except Exception:  # noqa: BLE001
            continue
        names = sorted(
            n for n, v in vars(mod).items()
            if same(v, fn) and not n.startswith("__")
        )
        if names:
            # sorted, so the emitted file is byte-identical across runs
            return cand, names[0], None

    # Nothing names it. Say which shape it is — "ImportError five frames
    # deep" is what this function exists to replace.
    if isinstance(qualname, str) and qualname.endswith("<lambda>"):
        return None, None, (
            f"{what} is a lambda defined in {module}: a lambda has no "
            f"importable name. Give it a def at module level"
        )
    if isinstance(qualname, str) and "<locals>" in qualname:
        return None, None, (
            f"{what} {module}.{qualname} is defined inside another "
            f"function, so no module-level name holds it. Move it to module "
            f"level — a factory can still build its arguments"
        )
    owner = getattr(fn, "__self__", None)
    if owner is not None:
        return None, None, (
            f"{what} {module}.{qualname} is a method BOUND to a "
            f"{type(owner).__name__} instance built at run time, and no "
            f"module-level name holds that instance. Bind it to one, or "
            f"wrap the construction in a module-level function so the file "
            f"can build the object the same way you did"
        )
    if isinstance(module, str) and isinstance(qualname, str):
        return None, None, (
            f"{what} {module}.{qualname} resolves to a DIFFERENT object "
            f"than the one passed, and no module-level name in {module} "
            f"holds it — a "
            f"decorator, a reassignment or a monkeypatch. Reproducing "
            f"against whatever that name holds would measure a different "
            f"program"
        )
    return None, None, (
        f"{what} is a {type(fn).__name__} carrying no importable "
        f"__module__/__qualname__, and no module-level name in "
        f"{candidates or 'any plausible module'} holds it, so there is no "
        f"name for an importing file to ask for"
    )


def _import_problem(fn, what: str) -> str | None:
    """The problem alone — :func:`_resolve_target`'s third element."""
    return _resolve_target(fn, what)[2]


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
    declaration, plus THE NAMES of the elements this layer invented.

    A solver model names only what the obligation's slice reached. Two
    different absences hide behind that, and the predecessor's disclosure
    conflated them by saying the fills happened "exactly as the replay
    did":

    * a name that IS a slice input but the model left free — the dispatch
      layer completes those from the declared box before replay
      (``stelling.solvers._complete_values``) and they arrive here already
      in ``witness.values``;
    * a declaration element the obligation never reaches, which is not a
      slice input at all. ``_complete_values`` iterates ``sl.inputs``, so
      no solver and no replay ever assigned it anything.

    Only the second kind is invented here, and the sidecar names them
    under ``witness_filled`` so a consumer can tell an invented value from
    a model one — which it previously could not.
    """
    got = dict(witness.values)
    for name in got:
        _declaration_of(name, subject)  # refuse a name from another query
    values: list[list[str]] = []
    filled: list[str] = []
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
            filled.append(name)
        values.append(elements)
    return values, filled


def _tool_leak(fn, what: str = "the target") -> str | None:
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
        # .strip() because an INDENTED import loads the tool just as well
        # once the enclosing function runs; the predecessor anchored at
        # column 0 and missed every lazy one. This stays a best-effort
        # emission-time HINT — it cannot see a helper that imports stelling
        # two modules down — and the exact check is the one the emitted
        # file makes at run time, in the process where it matters.
        if line.strip().startswith(("import stelling", "from stelling")):
            return (
                f"{what}'s module {module} imports stelling "
                f"({line.strip()!r}), so running this file loads the "
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
    """The emitted file's text.

    Two structural gates on the OUTPUT, not on the template: it never
    imports stelling, and it parses. The second exists because a contract
    name carrying a triple quote produced a file that raised SyntaxError
    at its first line — a reproducer that cannot be run is not evidence
    of anything, and :func:`one_line` closing the injection route is a
    reason to check the result rather than a reason not to.
    """
    sha = _stelling_sha()
    envelope = _envelope(subject)
    values, filled = _point(subject, witness)
    disclosures = [
        f"{name} was not constrained by the model (the obligation does not "
        f"reach it, so no solver and no replay ever assigned it a value); "
        f"this file INVENTED it from the declared box and the sidecar lists "
        f"it under witness_filled"
        for name in filled
    ]
    for fn_, what_ in (
        (subject.fn, "the target"),
        (subject.precondition, "the caller precondition"),
    ):
        leak = _tool_leak(fn_, what_) if fn_ is not None else None
        if leak is not None:
            disclosures.append(leak)
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
        # WHICH witness values the solver never produced. They are in
        # `witness` because they are the point that was executed, and they
        # are named here because a consumer must be able to tell an
        # invented value from a model one. The wording used to say they
        # were filled "exactly as the replay did", which is false on this
        # path: solvers._complete_values only completes names that ARE
        # slice inputs, and an element the obligation does not reach is
        # not one — no replay ever saw it.
        "witness_filled": sorted(filled),
        # load-bearing for the result and not derivable from anything else
        # here: the same rational rounds differently under x64, and a sha
        # is what a reader chasing this needs (the version is a release).
        "stelling_sha": sha,
        "x64": x64,
    }
    missing = set(SIDECAR_KEYS) - set(sidecar) - {"execution"}
    if missing:  # a schema key lost in a refactor must not ship silently
        raise ReproducerError(
            f"the sidecar payload is missing published schema key(s) "
            f"{sorted(missing)}; {SCHEMA} is a parsed surface"
        )
    t_mod, t_qual, _ = _resolve_target(subject.fn, "the target")
    p_mod = p_qual = None
    if subject.precondition is not None:
        p_mod, p_qual, _ = _resolve_target(
            subject.precondition, "the caller precondition"
        )
    payload = {
        "witness_elements": values,
        "target_module": t_mod,
        "target_qualname": t_qual,
        "precondition_module": p_mod,
        "precondition_qualname": p_qual,
        "precondition_reason": one_line(subject.precondition_reason),
        "no_precondition_reason": one_line(subject.no_precondition_reason),
        "disclosures": [one_line(d) for d in disclosures],
        "unconstructible": (
            one_line(unconstructible) if unconstructible else None
        ),
    }
    text = string.Template(_TEMPLATE).substitute(
        banner=_banner(sidecar, witness, disclosures, sha, x64),
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
    try:
        compile(text, "<reproducer>", "exec")
    except SyntaxError as e:
        raise ReproducerError(
            f"the emitted reproducer does not parse ({e}); a file that "
            f"cannot be run is not evidence of anything. This is a defect "
            f"in the emitter, not in your contract — please report it with "
            f"the contract name"
        ) from e
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
        # BELT AND BRACES, and measured as such: `Subject.__post_init__`
        # already refuses the newline and the triple quote, and for every
        # other name a constructible Subject accepts — backslashes, tabs,
        # form feeds — the funnel makes no difference to what the file
        # does. So no test can kill this call without reaching past the
        # frozen dataclass, which is outside this codebase's threat model.
        # One invariant, two mechanisms, exactly as
        # `contracts._require_sealed_ensures` documents for its own
        # unreachable re-check. The slots below are the load-bearing ones:
        # they arrive on the WITNESS, which guards nothing.
        f"contract         : {one_line(sidecar['contract'])}",
        f"verdict          : {sidecar['verdict']} "
        f"(assert #{sidecar['obligation']})",
        f"query hash       : {sidecar['query']}",
        f"stelling         : {sidecar['stelling']}  sha {sha}",
        f"jax              : {sidecar['jax']}  (jax_enable_x64={x64})",
        f"solver           : {one_line(witness.produced_by)}",
        f"fragment         : {sidecar['fragment']}",
        f"equations        : {sidecar['equations']}",
        f"replay           : {one_line(witness.replay)}",
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
        lines.append(f"disclosure       : {one_line(d)}")
    return "\n".join(lines)


def write_reproducer(
    verdict,
    subject: Subject,
    directory: str,
    *,
    obligation_index: int | None = None,
    filename: str | None = None,
) -> Emission:
    """Write ``<directory>/<name>.py`` reproducing ``verdict``'s witness.

    THE SUBJECT'S OWN HARNESS IS ALWAYS RE-TRACED, and its content hash
    must equal the verdict's stamped one, or nothing is written — a
    reproducer built against a different program than the one that was
    judged is a confident wrong answer, and this project has no use for
    one. There used to be a ``closed=`` parameter that let a caller supply
    the traced query and skip the 2.8 ms re-trace. It skipped the gate
    with it: the hash was compared between the CALLER'S query and the
    stamp, never between the caller's query and the SUBJECT, so passing
    program A's query while emitting subject B produced a file whose
    sidecar carried B's name and envelope, A's query hash, a witness
    outside B's own published envelope, and a CONFIRMED from executing B.
    A gate with a documented bypass is not a gate; the parameter is gone.

    The file IS written when the target cannot be constructed. That is
    deliberate (see :attr:`Emission.unconstructible`): a file that states
    plainly "I could not build this, here is exactly what was missing" is
    worth more than no file, and far more than one that dies at an import
    line five frames deep.
    """
    from stelling._jax_compat import trace, x64_enabled

    witness = _witness_for(verdict, obligation_index)
    stamped = verdict.stamp.query_content_hash
    traced = trace(subject.harness).content_hash()
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


# every mode present, null meaning "did not run" — the documented shape
_NO_MODES = {"eager": None, "jit": None}


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
              "lhs": None, "rhs": None, "modes": _NO_MODES,
              "sides_from": None})
    return ${not_executed_exit}


def _load(module, qualname):
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _nearest(np, dtype, f):
    """The correctly-rounded ``dtype`` image of an exact rational.

    ``dtype(float(f))`` rounds TWICE for any type narrower than binary64 —
    exact to double, then double to dtype — and double rounding can land
    one ulp away from the correctly-rounded value. At a witness that sits
    near a bound, one ulp is the whole difference between the assertion
    holding and not, so the file would report the wrong result for the
    right reason. Corrected by search: take the double-rounded candidate
    and its two neighbours, and keep whichever is genuinely nearest to the
    rational, ties to even mantissa.
    """
    first = dtype.type(float(f))
    cands = {first}
    for direction in (np.inf, -np.inf):
        try:
            cands.add(dtype.type(np.nextafter(first, dtype.type(direction))))
        except Exception:
            pass
    best, best_err = None, None
    for c in cands:
        err = abs(Fraction(float(c)) - f)
        if best_err is None or err < best_err or (
            err == best_err and int(np.asarray(c).view(
                np.int32 if dtype.itemsize == 4 else np.int64)) % 2 == 0
        ):
            best, best_err = c, err
    return best


def _build(np, decl, elements):
    """One declared array, from exact rationals, in the declared dtype."""
    dtype = np.dtype(decl["dtype"])
    exact = [Fraction(t) for t in elements]
    flat = np.asarray([_nearest(np, dtype, f) for f in exact], dtype=dtype)
    rounded = [i for i, f in enumerate(exact) if Fraction(float(flat[i])) != f]
    return flat.reshape(tuple(decl["shape"])), rounded


def _flat(np, v):
    return [float(x) for x in np.asarray(v).reshape(-1)]


def main():
    if PAYLOAD["unconstructible"]:
        return _stop(PAYLOAD["unconstructible"])
    try:
        jax = importlib.import_module("jax")
        # before jnp, and before any array exists: the declared dtypes are
        # only meaningful under the same setting the query was traced with
        jax.config.update("jax_enable_x64", SIDECAR["x64"])
        jnp = importlib.import_module("jax.numpy")
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

    # THE EXACT VERSION OF THE INDEPENDENCE CHECK. The emitter scans the
    # target's module source for an `import stelling` and discloses one, but
    # it cannot see a helper that imports it two modules down. This can: the
    # tool is either in this process or it is not, and by here the target
    # has been imported.
    if "stelling" in sys.modules:
        print("  DISCLOSURE: importing the target loaded stelling into this")
        print("  process. Nothing in this file calls it and the executed")
        print("  computation is still independent — but running this where")
        print("  stelling is absent, which is how that independence gets")
        print("  shown, is not possible while the program module reaches it.")

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
        # A JAX ARRAY, because the target is a jax program. Handing it a
        # numpy array breaks the most idiomatic write pattern JAX has:
        # `x.at[i].set(v)` does not exist on numpy.ndarray, so eager raised
        # AttributeError and this file reported "no execution result" for a
        # violation it could have confirmed — measured, on
        # `(T*2.0).at[0].set(0.0) <= 100.0`, which is FALSE at its witness.
        # (jit tolerated the numpy input, so the two modes did not even
        # agree about whether the program could be called.)
        args.append(jnp.asarray(arr))
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
                      "lhs": None, "rhs": None, "modes": _NO_MODES,
                      "sides_from": None,
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
            why = f"{type(e).__name__}: {e}"
            print(f"\\n  [{label}] not run ({why})")
            return None, None, None, why
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
        return holds, flat_l, flat_r, ""

    # BOTH MODES ARE THE PROGRAM. The compiler is entitled to rewrite the
    # expression, and measurably does: XLA simplifies (1 + x) - 1 to x, so
    # a violation binary64 absorbs eagerly survives compilation. Running one
    # mode and calling it "the program" would report the other one's answer
    # as this one's.
    holds_eager, eager_l, eager_r, eager_why = evaluate(target, "eager")
    holds_jit, jit_l, jit_r, jit_why = evaluate(jax.jit(target), "jit")
    modes = {"eager": holds_eager, "jit": holds_jit}
    ran = [m for m, h in modes.items() if h is not None]
    if not ran:
        # NEITHER mode ran. Only now is there no execution result — the
        # predecessor stopped as soon as EAGER raised, which is how the
        # standard `.at[].set()` write produced no result at all while jit
        # would have executed it happily.
        return _stop(
            "the target raised at the witness in both execution modes "
            "(eager: " + eager_why + "; jit: " + jit_why + ")"
        )
    disagree = len({modes[m] for m in ran}) > 1

    if disagree:
        print("\\n  THE TWO MODES DISAGREE, and that is itself the finding:")
        for m in ran:
            print(f"    {m:5s}: the assertion {'HOLDS' if modes[m] else 'is FALSE'}")
        print("    The compiler is allowed to rewrite the expression, so one")
        print("    of these is the program you ship and the other is the")
        print("    program you debug. Compare the lowered HLO before reading")
        print("    either as the answer.")

    # WHICH MODE'S NUMBERS GET PUBLISHED. A counterexample that satisfies
    # its own relation is not a counterexample, so the sides come from a
    # mode where the assertion is FALSE whenever one exists; otherwise from
    # the first mode that ran. `sides_from` says which, always.
    sides = {"eager": (eager_l, eager_r), "jit": (jit_l, jit_r)}
    false_modes = [m for m in ran if not modes[m]]
    sides_from = (false_modes or ran)[0]
    flat_l, flat_r = sides[sides_from]

    if false_modes:
        where = ", ".join(false_modes)
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
        dt = "/".join(sorted({d["dtype"] for d in SIDECAR["envelope"]}))
        print(f"  The comparison HOLDS at the witness in {dt}, your program's")
        print(f"  own declared type, in every mode run ({', '.join(ran)}),")
        print("  and stelling's")
        print("  exact-rational replay established it is FALSE in exact real")
        print("  arithmetic at the same point.")
        print("  THIS IS NOT A FAILED CHECK. It is a finding about the gap")
        print("  between the reals the verdict is stated over and the floats")
        print("  the program runs in: the violation is real, and smaller than")
        print("  this dtype can represent. Nothing here is wrong, the verdict")
        print("  is unchanged, and its stamped semantics line already says")
        print("  which arithmetic it is a claim about.")
        dtypes = sorted({d["dtype"] for d in SIDECAR["envelope"]})
        detail = (
            "false in the reals per the exact-rational replay, true under "
            "execution in " + "/".join(dtypes) + " (" + ", ".join(ran) + ")"
        )
        result = "${diverged}"
    _sidecar({"result": result, "detail": detail, "reachable": reachable,
              "lhs": flat_l, "rhs": flat_r, "modes": modes,
              "sides_from": sides_from})
    return ${result_exit}


if __name__ == "__main__":
    sys.exit(main())
'''
