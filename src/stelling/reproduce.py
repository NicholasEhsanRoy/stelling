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
and nothing of ours; the one (private) producer asserts that about its
own output text before returning it.

**The execution result is three-valued, and it is evidence, not a
verdict.**

* :data:`CONFIRMED` — the property is false at the witness under
  execution, in the program's own dtype, in at least one of the execution
  modes the file runs (eager and ``jax.jit``).
* :data:`DIVERGED` — it is false in ℝ (which the exact-rational replay
  established) and TRUE in the program's own dtype, in **every** mode,
  all of which must have run. A finding about the real/float gap.
  **Never a failed check**: nothing here is wrong, and the emitted file
  says so in those words.
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
  200 steps and recorded as the reachable set. It is a free-fall artifact
  of the trajectory length that was chosen, and driving the node says so:
  stepped from its own ``initial_state()`` with no boundary inputs, 200
  steps at ``dt = 0.001`` span ``[-1.9619957208633423, 0]`` and 100 000
  steps at the same ``dt`` span ``[-981.4652709960938, 0]`` — the same
  law, five hundred times the trajectory.
  (This clause has now been wrong twice, in opposite ways. It read
  ``[-9805.9, 0]``, which is a ``dt = 0.01`` figure — 100 000 steps at
  ``dt = 0.01`` drive to ``[-9805.8662109375, 0]`` — so the larger number
  silently changed the ``dt`` the clause before it states. It was then
  replaced by ``[-981.0, 0]``, which is what ``100000 × 0.001 × 9.81``
  comes to and is NOT what the node reaches: the velocity accumulates in
  float32, and the driven span runs about half a unit past it. An
  arithmetic answer substituted for a measured one, in the paragraph whose
  whole subject is that substitution. Every span above is driven, not
  computed. The point — that the span is a property of the trajectory and
  not of the node — is unchanged, and is stronger when only one variable
  moves.) Velocity
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

A second audit of those repairs found six more, and TWO WERE
REPAIR-INTRODUCED — both in the execution path, both from adding a
special case rather than removing what made one necessary. The lesson is
recorded here because it is the one that generalises: **when a fix needs
another branch to hold, delete the thing that made the branch necessary.**

* Both execution modes were handed the SAME input buffers, and a jax
  buffer is destroyable: a target using ``donate_argnums`` deleted its
  argument during the eager call, the jit call then raised, and the
  "a mode that raises no longer cancels the mode that runs" rule from the
  first repair round reported the eager answer as the whole answer —
  ``DIVERGED ... Nothing here is wrong``, at a witness where the compiled
  form violates. Each mode builds its own inputs now, and DIVERGED
  additionally requires EVERY mode to have run: an unmeasured mode is not
  an agreeing one.
* ``_nearest``, added to fix double rounding, computed
  ``Fraction(float(inf))`` and raised for any witness above the declared
  dtype's finite range — turning a real CONFIRMED into no result at all.
  Overflow to ±inf IS the correctly rounded conversion; there is no
  neighbour to search for.
* Removing ``write_reproducer``'s ``closed=`` parameter closed one door
  and left five windows: ``reproducer_source`` was public and took the
  query hash, fragment, equation count and precision straight from its
  caller. The same cross-program file came out through it. There is now
  ONE producer, it is private, and it derives everything.
* Target resolution drew candidate modules only from the object's own,
  its type's and its ``func``'s ``__module__`` — never the module the
  object is actually bound in — so the fixture shape this module's own
  guidance recommends still reported uncallable.
* The sidecar emitted the bare ``Infinity`` token, which Python reads back
  and no other language's parser accepts.
* ``_tool_leak`` scanned the target module's source and was wrong in both
  directions: it accused a docstring that merely quoted the line, and
  missed a lazy import inside the function. It is DELETED; the emitted
  file asks ``sys.modules`` after running the target, which is exact.
"""

from __future__ import annotations

import ast
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
    "SCHEMA_STABILITY",
    "SIDECAR_KEYS",
    "ReproducerError",
    "one_line",
    "Subject",
    "UNREACHABLE",
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
# failures of anything. The one nonzero code marks "NO EXECUTION RESULT" —
# this file has nothing to report about the program, which is not the same
# as having nothing to report about its target. It covers both: the target
# could not be constructed or run at all, AND the case where it ran
# perfectly well and what ran was not enough to name a result (the
# assertion held in one mode and the other could not run, so DIVERGED —
# a claim of absence — is not available and nothing was false either).
RESULT_EXIT = 0
NOT_EXECUTED_EXIT = 3

# ── the sidecar schema, PROVISIONAL — on a CONDITION, not a version ──────────
#
# **The schema is PROVISIONAL / UNSTABLE.** Fields may be added, removed or
# renamed in any release without a deprecation cycle. It FREEZES ON A
# CONDITION, not on a version: once a consumer that did not write it has
# parsed real emissions and the fields have been exercised from outside.
#
# **AND `provisional` HERE IS A MARKING ON A DOCUMENT FORMAT, NOT
# `DOCUMENTATION_ARCHITECTURE.md` §8.5's STABILITY LEVEL OF THE SAME NAME.**
# The two words collide and they do not mean the same thing. §8.5's
# `provisional` is *"may change in minor with a deprecation cycle"*,
# guarantee *"one minor's notice"*. The sentence directly above gives
# strictly less than that — any release, no deprecation cycle — which is
# that table's `experimental`, *"may change without notice"*, guarantee
# *"none"*. A reader who knows the table would come away with a promise of
# notice that nothing here makes, and in 0.2.0 that reader is not
# hypothetical: `docs/harness-api.md` and `docs/preconditions.md` now teach
# §8.5's levels by name, for the `falsify` keyword — and those are the two
# pages documenting `check`, which is where the verdict this module
# reproduces comes from.
#
# THE COLLISION IS NAMED, AND THE IDENTIFIER IS NOT RENAMED. Three reasons,
# and the third is what decides it:
#
#   * **§8.5 does not reach this artifact.** Its *Applied to:* line names
#     the harness API, the verdict artifact schema, `stelling.ir` and the
#     evidence schemas. The verdict artifact and the evidence schemas are
#     the JSON under `evidence/`, which §3.2 of that same document records
#     as never having existed in this repository; this sidecar is neither
#     of them. Nor does that line reach `stelling.reproduce`: the Python
#     inventory in it is the harness API, and `SCHEMA` is not in that.
#     So no level was ever ASSIGNED here and then got wrong. What is here
#     is a word that reads like one.
#   * **The word is doing a different job, and it is a job a level cannot
#     do.** A §8.5 level is a standing promise about future changes to a
#     surface. `1-provisional` is a STATE of one document format: `1` is
#     the schema version and the suffix is its not-frozen-yet flag, which
#     stops being said the day the condition below is met. A promise a
#     surface can simply stop making is not the kind of thing that table
#     describes.
#   * **The identifier is WIRE.** `stelling.reproducer/1-provisional` is
#     stamped into every sidecar this module emits, into the `SIDECAR`
#     block of every reproducer it writes, and into that file's own header
#     line. Renaming the suffix is a change to what consumers see, and it
#     buys a consumer nothing: the whole design of the marking is that the
#     ordinary check `doc["schema"] == "stelling.reproducer/1"` FAILS
#     CLOSED, and it already does. It costs: any pin already written
#     against the emitted string breaks, `tests/test_reproduce.py` pins
#     the literal AND pins that the word `provisional` is in it, and
#     `docs/reproducing-a-witness.md` prints the literal twice and the
#     `stelling.reproducer/1` comparison a consumer would write once. A
#     wire change for a word's connotation is not a trade this file should
#     make.
#
# So the guarantee stands, the identifier stands, and the sentence that
# could mislead gets this disclaimer. Two things are deliberately NOT done.
# There is no version-numbered freeze promise — that defect was here once
# and was removed, and the paragraph below is why. And the emitted
# `stability` string and the reproducer's own banner are left byte for
# byte alone: they already state the guarantee in full words, and their
# reader is one who has the JSON and has never seen §8.5. The collision
# bites a reader of THIS repository, so the repair is here and on the page
# that reader also reads.
#
# THE CONDITION IS THE COMMITMENT, AND NAMING A RELEASE INSTEAD BROKE IT.
# This said "may be added, removed or renamed in 0.1.1 ... planned to FREEZE
# in 0.1.1". 0.1.1 came and went, and a second release arrived without it;
# the running version is 0.2.0, the schema is still `1-provisional`, and the
# sentence had become a promise about a release that is in the past — which
# reads to a consumer either as "this froze and nobody updated the string"
# or as an abandoned plan, and neither is true. The condition below has not
# been met, so nothing about the guarantee has changed; only the way it is
# stated, so that it cannot expire again.
#
# The withdrawal is deliberate, and the reason is that the argument for
# declaring it stable was never an argument for declaring it stable NOW. It
# ran: a CI coverage line and an external soak parse this, therefore it is a
# published surface. That is a reason for it to become stable EVENTUALLY —
# nothing has parsed a single one of these files in anger yet, so no field
# here has been tested by anyone but its author, and "small and designed to
# survive" is a prediction until a consumer has tried to live on it. A
# schema declared stable and then changed is a break for whatever parses
# it; the marking costs a version string.
#
# Everything else in this feature is repairable in a patch release. This is
# the only irreversible commitment in it, which is why it is the one thing
# held back.
#
# The marking is in the SCHEMA IDENTIFIER ITSELF, not only in a field, so a
# consumer doing the ordinary thing — comparing `schema` against the string
# it was written for — fails closed rather than succeeding against a
# guarantee that was never given. The `stability` field carries the sentence
# and the freeze condition for a reader who has only the JSON.
SCHEMA = "stelling.reproducer/1-provisional"

SCHEMA_STABILITY = (
    "PROVISIONAL / UNSTABLE: fields may be added, removed or renamed in any "
    "release without a deprecation cycle. This freezes on a CONDITION and "
    "not on a version number -- once a consumer that did not write it has "
    "parsed real emissions and these fields have been exercised from "
    "outside -- and until the identifier stops saying 'provisional' the "
    "condition has not been met. Do not build on this without pinning the "
    "stelling version."
)

# JSON HAS NO ENCODING FOR ±inf OR NaN, and Python's json module emits the
# bare tokens `Infinity`/`NaN`, which its own loader accepts and jq,
# JSON.parse, Go and serde all reject. A published surface that only its
# author's language can read is not published. So every numeric field here
# is a JSON number OR one of the strings below, and both writers set
# `allow_nan=False` so a path that forgets this raises instead of emitting
# something no consumer can parse.
NONFINITE = {float("inf"): "inf", float("-inf"): "-inf"}

SIDECAR_KEYS = (
    "schema",       # str  — SCHEMA above; the identifier says "provisional"
    "stability",    # str  — SCHEMA_STABILITY: what that means, and when it
                    #        stops being true
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

# The declaration dtypes the obligation layer admits at all. A COPY of
# stelling.obligation._FLOAT_INPUT_DTYPES, kept jax-free and pinned equal by
# tests/test_reproduce.py — the predecessor was a copy too, and had drifted:
# it omitted float16 and refused it with a reason ("only meaningful for a
# float") that is untrue of float16. A drifted copy that refuses a supported
# declaration is a wrongly-silent file, which is this module's own named
# defect family.
_DTYPES = ("float16", "float32", "float64")


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
                    f"Subject.declarations[{k}] declares dtype {dtype!r}; "
                    f"the declaration layer admits {_DTYPES} and nothing "
                    f"else, so no other dtype can reach a witness"
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
    import sys

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
    # this exact object.
    #
    # AN OBJECT IS NOT BOUND WHERE ITS CLASS IS DEFINED. The predecessor
    # drew candidates only from the object's own, its type's and its
    # ``func``'s ``__module__``, so a callable instance or a partial bound
    # in one module whose class or function comes from another — the exact
    # shape the fixture guidance recommends — was reported uncallable. The
    # cheap candidates are tried first because they usually hit; when they
    # do not, every loaded module is scanned, in sorted order so the answer
    # cannot depend on import order. Measured on 708 loaded modules: 0.6 ms
    # to a hit, 5.7 ms to exhaust, and only on the path where the cheap
    # answer already failed.
    def scan(names_to_try):
        for cand in names_to_try:
            if cand in ("__main__", "builtins"):
                continue  # __main__ names a different module over there
            try:
                mod = importlib.import_module(cand)
            except Exception:  # noqa: BLE001
                continue
            try:
                items = sorted(vars(mod).items())
            except Exception:  # noqa: BLE001
                continue
            for n, v in items:
                if same(v, fn) and not n.startswith("__"):
                    return cand, n
        return None

    cheap = []
    for cand in (
        module,
        getattr(type(fn), "__module__", None),
        getattr(getattr(fn, "func", None), "__module__", None),
    ):
        if isinstance(cand, str) and cand not in cheap:
            cheap.append(cand)
    hit = scan(cheap) or scan(sorted(sys.modules))
    if hit is not None:
        return hit[0], hit[1], None

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
        f"{what} is a {type(fn).__name__} from "
        f"{getattr(type(fn), '__module__', '?')}, and NO module-level name "
        f"anywhere in the loaded modules holds this object, so there is no "
        f"name for an importing file to ask for. Bind it to one"
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


def _require_same_program(verdict, subject: Subject, trace, x64: bool) -> None:
    """Refuse unless the subject's own harness traces to the query the
    verdict stamps. THE SUBJECT IS ALWAYS RE-TRACED — there is no
    parameter, public or private, that supplies the hash instead.

    THE FUNCTION'S JOB IS NOT TO DIAGNOSE, IT IS TO NOT MISDIAGNOSE. A hash
    mismatch has several possible causes and this check has one hash on each
    side; what it can do honestly is name every difference it can SEE in the
    stamp, and say plainly that it cannot tell whether that is all that
    differs. So it enumerates rather than branching: the stamp carries three
    facts about the run that produced the verdict — the query hash, the
    precision config, and the jax version — and every one of them that
    disagrees with the running environment goes into the refusal.

    THE jax VERSION CLAUSE IS THE ONE THAT WAS MISSING, and its absence was
    exactly the misdiagnosis this docstring exists to prevent. A query
    content hash is a function of the traced equations' PARAMS, and jax owns
    those, so one unchanged program traces to two hashes across a jax
    release: ``SOUNDNESS.md``'s 2026-08-18 entry records ``reduce_max`` and
    ``reduce_min`` gaining an ``out_sharding`` param in jax 0.11.1, and that
    entry names "a CI job that re-traces and diffs" as the shape that breaks.
    THIS FUNCTION IS THAT SHAPE. Driven end to end: one importable target,
    one ``Subject``, ``jax_enable_x64`` True on both sides, a verdict
    produced on real jax 0.11.0 and re-emitted on real jax 0.11.1 — the
    refusal read "this verdict is not about this subject's program", sending
    a reader to look for a program difference in a program that had not
    changed, while ``verdict.stamp.jax_version`` sat two attributes away
    carrying the answer.

    THE DIRECTION IS CONSERVATIVE AND STAYS THAT WAY. Every path here
    REFUSES; none of them emits. What was wrong was the reason given for a
    correct refusal, and what is added is a reason, not a permission.

    IT STILL DOES NOT CLAIM THE PROGRAMS ARE THE SAME. Its predecessor said
    "The program is the same" when the precision differed, which for an
    unrelated subject over an unrelated envelope is a misdiagnosis of exactly
    the kind the enumeration exists to prevent.
    """
    stamped = verdict.stamp.query_content_hash
    traced = trace(subject.harness).content_hash()
    if traced == stamped:
        return

    # Every difference the STAMP can witness, in the order a reader should
    # act on them: the cheap local knob first, then the environment.
    seen = []
    stamped_x64 = verdict.stamp.precision_config
    running_x64 = f"jax_enable_x64={x64}"
    if stamped_x64 != running_x64:
        seen.append(
            f"THE PRECISION SETTING MOVED — stamped {stamped_x64}, running "
            f"{running_x64}. A different jax_enable_x64 alone is enough to "
            f"make ONE program trace to two queries, because it changes the "
            f"declared dtypes, so the precision setting explains a hash "
            f"mismatch on its own. Set "
            f"jax.config.update('jax_enable_x64', "
            f"{stamped_x64.split('=')[-1]}) and emit again"
        )
    stamped_jax = verdict.stamp.jax_version
    running_jax = _jax_version()
    if stamped_jax != running_jax:
        seen.append(
            f"THE jax VERSION MOVED — stamped jax {stamped_jax}, running jax "
            f"{running_jax}. A query content hash is a function of the traced "
            f"equations' params and jax owns those, so one UNCHANGED program "
            f"traces to two hashes across a jax release; SOUNDNESS.md's "
            f"2026-08-18 entry records the measured case (reduce_max and "
            f"reduce_min gained out_sharding in 0.11.1). Emit on jax "
            f"{stamped_jax}, or re-run the check on jax {running_jax} and "
            f"emit from the verdict that produces"
        )

    if seen:
        raise ReproducerError(
            f"the verdict stamps query {stamped} and the subject's harness "
            f"traces to {traced}, and the stamp disagrees with this "
            f"environment in "
            f"{'1 way' if len(seen) == 1 else f'{len(seen)} ways'} that can "
            f"move a hash without any program changing: "
            + "; ".join(seen)
            + ". THIS CHECK CANNOT TELL WHETHER THAT IS ALL THAT DIFFERS: it "
            "has one hash on each side. If the hashes agree once the "
            "difference above is removed, that was the cause; if they still "
            "differ, this verdict is about a different program"
        )

    raise ReproducerError(
        f"this verdict is not about this subject's program: the verdict "
        f"stamps query {stamped}, and the subject's harness traces to "
        f"{traced}. The stamp's jax version and precision config both match "
        f"this environment, so neither of those explains it. Emitting anyway "
        f"would produce a file that executes one program and quotes another "
        f"program's verdict"
    )


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


def _declaration_size(subject: Subject, k: int) -> int:
    size = 1
    for d in subject.declarations[k][0]:
        size *= int(d)
    return size


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
    element = int(tail) if tail else 0
    # THE ELEMENT INDEX TOO. The predecessor checked only the declaration,
    # so a witness carrying `x1_77` for a scalar declaration was accepted
    # and its value then SILENTLY DROPPED — the element the model actually
    # constrained never reached the built array, and the file executed a
    # point the solver did not produce.
    size = _declaration_size(subject, k)
    if element >= size or (not tuple(subject.declarations[k][0]) and tail):
        raise ReproducerError(
            f"witness value {name!r} names element {element} of declaration "
            f"#{k}, which has {size} element(s) and shape "
            f"{tuple(subject.declarations[k][0])}; this witness is not from "
            f"this subject's query"
        )
    return k, element


def _json_number(value: float):
    """A JSON number, or the agreed string for a value JSON cannot hold."""
    if value != value:
        return "nan"
    return NONFINITE.get(value, value)


def _bound(raw) -> float:
    """A declared bound as its binary64 image, through the SAME classifier
    the declaration layer decides with.

    ``float(raw)`` and ``Fraction(raw)`` both refuse spellings
    ``stelling._bound_spelling.ACCEPTED_SPELLINGS`` explicitly admits —
    measured, ``Fraction(numpy.float32(2.5))`` raises a bare TypeError out
    of the emitter for a bound ``any_array`` accepts without comment. The
    canonical accessor already exists; a second conversion here would be a
    second thing to keep correct, which is the lesson `_barred_primitives`
    paid for.
    """
    from stelling._bound_spelling import binary64_image, declared_bound_value

    exact = declared_bound_value(raw)
    if exact is None:  # unreachable through any_array, refused loudly anyway
        raise ReproducerError(
            f"declared bound {raw!r} is not an accepted bound spelling; the "
            f"declaration layer would have refused it at trace time"
        )
    return binary64_image(exact)


def _envelope(subject: Subject) -> list[dict]:
    return [
        {
            "name": f"x{k}",
            "shape": [int(d) for d in shape],
            "dtype": dtype,
            "lo": _json_number(_bound(lo)),
            "hi": _json_number(_bound(hi)),
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
        size = _declaration_size(subject, k)
        elements = []
        for i in range(size):
            name = f"x{k}" if not tuple(shape) else f"x{k}_{i}"
            if name in got:
                elements.append(str(Fraction(got[name])))
                continue
            lo_f, hi_f = _bound(lo), _bound(hi)
            fill = lo_f if lo_f != float("-inf") else (
                hi_f if hi_f != float("inf") else 0.0
            )
            elements.append(str(Fraction(fill)))
            filled.append(name)
        values.append(elements)
    return values, filled


def _flat_witness(subject: Subject, values: list[list[str]]) -> dict:
    out = {}
    for k, ((shape, _d, _b), elements) in enumerate(
        zip(subject.declarations, values)
    ):
        for i, text in enumerate(elements):
            out[f"x{k}" if not tuple(shape) else f"x{k}_{i}"] = text
    return out


def _reproducer_source(verdict, subject: Subject, obligation_index) -> str:
    """The emitted file's text, and THE ONE PLACE A FILE IS PRODUCED.

    PRIVATE, and it takes no values from its caller beyond the verdict,
    the subject and which obligation. Its predecessor was public and
    accepted ``witness``, ``query_hash``, ``fragment``, ``equations`` and
    ``x64`` straight through — so removing ``write_reproducer``'s
    ``closed=`` parameter closed one door and left five windows open.
    Measured: the same cross-program file came out through this function,
    carrying subject B's name and envelope, program A's query hash, a
    witness outside B's own published envelope, and a CONFIRMED from
    executing B. Everything is derived here now, the gate runs here, and
    :func:`write_reproducer` only writes what this returns.

    Three structural gates on the OUTPUT rather than on the template: it
    is about the program it says it is, it never imports stelling, and it
    parses.
    """
    from stelling._jax_compat import trace, x64_enabled

    witness = _witness_for(verdict, obligation_index)
    query_hash = verdict.stamp.query_content_hash
    x64 = x64_enabled()
    _require_same_program(verdict, subject, trace, x64)
    fragment = _fragment_of(verdict.stamp)
    equations = _equation_count(verdict.stamp)
    unconstructible = _import_problem(subject.fn, "the target")
    if unconstructible is None and subject.precondition is not None:
        unconstructible = _import_problem(
            subject.precondition, "the caller precondition"
        )
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
    sidecar = {
        "schema": SCHEMA,
        "stability": SCHEMA_STABILITY,
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
        # allow_nan=False on BOTH, which is what the NONFINITE comment
        # above claims. It was true of the runtime writer only: these two
        # produce the blocks that land in the emitted FILE
        # (`SIDECAR = json.loads(r"""…""")`), and a non-finite reaching
        # here would have been written as a bare `Infinity`/`NaN` token
        # that jq, JSON.parse, Go and serde all reject. No escape was
        # demonstrated -- `_json_number` sanitises `envelope.lo/hi`
        # upstream -- so this closes a claim defect rather than a shown
        # bug, and it is the guard that makes the claim true.
        sidecar=json.dumps(sidecar, indent=2, sort_keys=True,
                           allow_nan=False),
        payload=json.dumps(payload, indent=2, sort_keys=True,
                           allow_nan=False),
        confirmed=CONFIRMED,
        diverged=DIVERGED,
        unreachable=UNREACHABLE,
        result_exit=RESULT_EXIT,
        not_executed_exit=NOT_EXECUTED_EXIT,
    )
    _refuse_tool_import(text)
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


def _refuse_tool_import(text: str) -> None:
    """THE ONE RULE, CHECKED ON THE OUTPUT rather than promised by the
    template. A reproducer that reaches back into the tool checks the tool
    with the tool and is worth nothing as independent evidence, so this is a
    structural refusal at the point of emission, not a comment asking the next
    author to be careful.

    TWO CHECKS, AND THE SECOND EXISTS BECAUSE THE FIRST HAS A HOLE THE
    ARGUMENT FOR IT DENIED. The line scan below was defended in `SOUNDNESS.md`
    on the ground that Python's statement-separator set is a strict SUBSET of
    ``str.splitlines()``' — so the scan would see more line-starts than the
    tokenizer and could only cry wolf. The two sets are not nested. Measured
    over the whole code-point range, `compile("x=1" + c + "y=2")` succeeds for
    ``;`` and ``#`` and `("a" + c + "b").splitlines()` does not split on
    either, while `splitlines()` splits on eight the compiler does not accept
    (U+000B U+000C U+001C U+001D U+001E U+0085 U+2028 U+2029). ``;`` is the one
    that bites, because it carries a real statement::

        "x = 1; import stelling\\ny = 2\\n"

    — a genuine ``import stelling`` that the line scan does not see and
    ``ast.parse`` does. Not reachable from caller text today (:func:`one_line`
    maps every character below U+0020 to a space and neutralises the triple
    quote, and the JSON blocks sit inside a raw triple-quoted string
    ``json.dumps`` cannot terminate), which is exactly why the scan's charter
    is the FUTURE edit of :data:`_TEMPLATE` rather than today's caller — and a
    refusal a semicolon walks past does not discharge that charter.

    The line scan is KEPT rather than replaced: it fires on an
    ``import stelling`` inside a string or a docstring, which the parse tree
    correctly does not report, and a false alarm at the point of emission is
    the cheap direction. Neither check implies the other.

    WHAT THE TREE WALK ACTUALLY COVERS. This said "the tree walk is exact and
    catches the rest wherever on the line it sits", and *the rest* claimed more
    than the code does. It is exact on ``ast.Import`` and ``ast.ImportFrom``
    and on nothing else. Measured against this function, all four of these
    PASS BOTH CHECKS and compile::

        __import__('stelling')
        importlib.import_module('stelling')
        exec("import stelling")
        from . import stelling

    The first two and the third genuinely reach the tool when run (measured:
    ``__import__('stelling')`` in a standalone script imports it). The fourth
    cannot — a reproducer is a standalone script, so a relative import raises
    ``ImportError: attempted relative import with no known parent package``
    (measured) — but it is still a shape this walk does not report.

    NOT WIDENED, and the reason is the charter rather than inertia. This
    refusal exists so that a future edit of :data:`_TEMPLATE` cannot
    reintroduce an import ACCIDENTALLY; it is not a sandbox, and it cannot
    become one, because no static check can see through ``exec`` or an
    ``__import__`` whose argument is computed. Widening to a name-based rule
    would buy a partial fence at a cry-wolf cost on a function that runs on
    every emission. What IS claimed is the whole of what is enforced: no
    ``Import`` or ``ImportFrom`` naming ``stelling`` or a ``stelling.*``
    submodule survives here, wherever on the line it sits.

    ONE THING ONLY THE TREE WALK CAN CATCH, which is why it is not merely a
    superset argument: ``import ﬅelling`` (U+FB05) is NFKC-normalised to
    ``stelling`` by the parser, so the walk reports it (measured) and no line
    scan over the source text ever could.

    The tree walk runs only when the text parses; when it does not, the
    caller's own :func:`compile` reports that with its own message, and this
    function stays silent so that message is unchanged.
    """
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith(("import stelling", "from stelling")):
            raise ReproducerError(
                f"the emitted reproducer imports stelling at line {lineno} "
                f"({line.strip()!r}) — a file that reaches back into the tool "
                f"checks the tool with the tool"
            )
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name == "stelling" or name.startswith("stelling."):
                raise ReproducerError(
                    f"the emitted reproducer imports {name} at line "
                    f"{node.lineno} — a file that reaches back into the tool "
                    f"checks the tool with the tool. Found on the PARSE TREE, "
                    f"not at a line start: a statement after a `;` is an "
                    f"import the line scan above cannot see"
                )


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
        f"sidecar schema   : {sidecar['schema']}",
        f"                   {one_line(sidecar['stability'])}",
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
    witness = _witness_for(verdict, obligation_index)
    text = _reproducer_source(verdict, subject, obligation_index)
    unconstructible = _unconstructible_of(subject)
    stem = filename or (
        f"reproduce_{_slug(subject.name)}_assert{witness.obligation_index}"
    )
    if stem.endswith(".py"):
        stem = stem[:-3]
    # A NAME, NOT A PATH. `filename="../x"` wrote outside `directory`, which
    # is not a thing a caller asking for a file in a directory can have meant.
    if os.sep in stem or (os.altsep and os.altsep in stem) or stem in (
        "", ".", ".."
    ):
        raise ReproducerError(
            f"filename {filename!r} must be a bare file name, not a path: "
            f"the reproducer and its sidecar are written inside {directory!r}"
        )
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


def _unconstructible_of(subject: Subject) -> str | None:
    problem = _import_problem(subject.fn, "the target")
    if problem is None and subject.precondition is not None:
        problem = _import_problem(
            subject.precondition, "the caller precondition"
        )
    return problem


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
anything. Exit ${not_executed_exit} means there is NO EXECUTION RESULT: this file has
nothing to report about the program. That covers a target it could not
construct or run, and also a target that ran fine but not in every mode —
the assertion held where it ran and the other mode raised, so DIVERGED,
which is a claim of absence, is not available and nothing was false
either. The sidecar's `detail` says which. Read it, not the status.

THE SIDECAR SCHEMA IS PROVISIONAL. Fields may be added, removed or
renamed in any release without a deprecation cycle. It freezes on a
CONDITION rather than on a version number — once a consumer that did not
write it has parsed real emissions and the fields have been exercised
from outside — and until the schema identifier stops saying
"provisional", that has not happened. The schema identifier says so, and every sidecar
this file writes carries the same sentence in its `stability` field. Pin
the stelling version if you build on it.
"""
import importlib
import json
import os
import sys
from fractions import Fraction

# THE TARGET IS IMPORTED BY NAME, AND THIS FILE IS NOT WHERE YOU RAN FROM.
# Python puts THIS SCRIPT'S directory on sys.path[0], not the working
# directory -- and a reproducer is emitted into a subdirectory, one level
# below the program it is evidence about. So the documented command, run
# from the directory holding that program,
#
#     $$ python reproducers/reproduce_<name>.py
#
# could not import a module sitting right there, and this file reported
# NO EXECUTION RESULT about a target it could have executed. Measured
# before this block existed, from exactly that directory:
# `ModuleNotFoundError: No module named 'myprogram'`, exit 3 -- a wrongly
# silent file, which is the same family of defect as a wrong one.
#
# APPENDED, never prepended. sys.path[0] stays this file's own directory
# and every other entry keeps its position, so a name that resolved to a
# MODULE or a REGULAR PACKAGE before this line existed resolves to the same
# one now, and the intended effect is that a name which previously resolved
# to NOTHING can resolve in the working directory. Run from somewhere the
# target is not and the file still stops with the same honest "could not be
# imported" detail it stopped with before.
#
# It is NOT true that position on the path settles every case, and this
# said it was. Under PEP 420 the finder scans the WHOLE path before
# settling for a namespace package: portions accumulated from earlier
# entries lose to a regular module or package found at any later entry.
# Measured in one process -- a directory `zzportion/` with no `__init__.py`
# on an earlier entry, a `zzportion.py` in the cwd -- `import zzportion`
# gives a namespace package (`__file__` None) before the append and the
# regular module after it. So a namespace package assembled from earlier
# entries CAN be displaced by a same-named module in the working directory.
# The impact here is very low: a reproducer runs one named target beside a
# program the author is debugging. It is narrowed rather than deleted
# because the sentence was an absolute in shipped source, and this file is
# the one that emits evidence.
_CWD = os.getcwd()
if _CWD not in sys.path:
    sys.path.append(_CWD)

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


# The FIRST point at which stelling was observed in this process, as
# (phase, who). Recorded rather than sampled once, because the answer
# depends on WHEN you look: the predecessor sampled a single flag after the
# caller precondition had already run and then attributed every case to
# "importing the target", which for a precondition that imports the tool
# lazily while running names the wrong callable AND the wrong phase.
_TOOL_PHASE = []


def _note_tool_phase(phase, who):
    if "stelling" in sys.modules and not _TOOL_PHASE:
        _TOOL_PHASE.append((phase, who))


# THIS GUARD FAILED TWICE, IN TWO SHAPES, AND THE SECOND IS WHAT THE
# OBVIOUS REPAIR FOR THE FIRST PRODUCES. Both are "the guard exists but
# does not fire where it is needed", which this project keeps meeting.
#
#   1. Reachable from one place only. It sat after both execution calls,
#      so four earlier returns left without asking — UNREACHABLE,
#      both-modes-raised, witness-build-failed, and target-import-failed,
#      the worst of them because it IS the stelling-absent environment
#      this disclosure is about.
#   2. Reachable everywhere and keyed on the wrong signal. Moved into
#      `_sidecar`, it asked `_TOOL_PHASE`, which is right there and looks
#      like the answer: a target that loads the tool lazily and then
#      raises in both modes returns before the last phase sample, so it
#      disclosed nothing with stelling demonstrably in the process.
#
# THE SEPARATION THAT RESOLVES BOTH: `sys.modules` decides WHETHER to
# disclose; `_TOOL_PHASE` supplies only WHO and WHEN. Moving this check
# back into the flow re-derives (1); re-keying it on the phase re-derives
# (2). Both were measured, the second before it shipped.


def _disclose_tool(detail):
    """The independence disclosure, on EVERY path that reports anything.

    Called from `_sidecar`, so it is reachable from every terminal path by
    construction rather than by placement. See the note above before
    moving it or changing what it asks.
    """
    if "stelling" in sys.modules:
        phase, who = (
            _TOOL_PHASE[0] if _TOOL_PHASE else ("running", "your program")
        )
        print(f"\\n  DISCLOSURE: {phase} {who} loaded stelling into this")
        print("  process. Nothing in this file calls it and the executed")
        print("  computation is still independent — but running this where")
        print("  stelling is absent, which is how that independence gets")
        print("  shown, is not possible while your program reaches it.")
    elif detail and "stelling" in str(detail):
        # the tool is NOT in this process and the reason we stopped names
        # it: the import failed because stelling is absent here, which is
        # the same fact seen from the other side
        print("\\n  DISCLOSURE: this run could not import your program"
              " BECAUSE")
        print("  stelling is unavailable here, so your program reaches the")
        print("  tool. The executed computation would still be independent;")
        print("  what you cannot do is run this where stelling is absent,")
        print("  which is how that independence gets shown.")


def _sidecar(execution):
    _disclose_tool(execution.get("detail"))
    out = os.environ.get("STELLING_REPRODUCER_JSON") or (
        os.path.splitext(os.path.abspath(__file__))[0] + ".json"
    )
    record = dict(SIDECAR)
    record["execution"] = execution
    with open(out, "w", encoding="utf-8") as fh:
        # allow_nan=False: Python emits the bare tokens Infinity/NaN, which
        # only Python reads back. A published surface that jq and JSON.parse
        # reject is not published, so a path that forgets _flat raises here.
        json.dump(record, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\\n")
    print(f"\\nsidecar: {out}")
    return record


def _stop(detail, reachable=None, modes=None):
    """No execution result exists. Say exactly what is missing.

    ``reachable`` and ``modes`` are parameters because they may already be
    KNOWN: a run whose precondition ran and held, and whose target then
    raised, published ``null`` for both — a measured value made
    indistinguishable from an unknowable one.
    """
    print("== NO EXECUTION RESULT")
    print("  This reproducer has no result to report, so nothing is claimed")
    print("  here in either direction.")
    print(f"  WHAT IS MISSING: {detail}")
    print("  The verdict this file is evidence about is unchanged. It simply")
    print("  has no execution leg until this is resolved.")
    _sidecar({"result": None, "detail": detail, "reachable": reachable,
              "lhs": None, "rhs": None,
              "modes": dict(_NO_MODES) if modes is None else modes,
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
    with np.errstate(over="ignore"):
        first = dtype.type(float(f))
    if not np.isfinite(first):
        # The value is outside this dtype's finite range, and overflow to
        # +-inf IS the correctly rounded IEEE result — there is no nearer
        # finite neighbour to search for. The predecessor went on to compute
        # Fraction(float(inf)), which raises OverflowError, and the file
        # turned that into "no execution result" for a violation it could
        # have confirmed.
        return first
    cands = {first}
    for direction in (np.inf, -np.inf):
        with np.errstate(over="ignore"):
            nxt = np.nextafter(first, dtype.type(direction))
        if np.isfinite(nxt):
            cands.add(dtype.type(nxt))
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
    rounded = [
        i for i, f in enumerate(exact)
        if not np.isfinite(flat[i]) or Fraction(float(flat[i])) != f
    ]
    return flat.reshape(tuple(decl["shape"])), rounded


def _flat(np, v):
    """Flat values, with the two JSON cannot hold spelled as strings."""
    out = []
    for x in np.asarray(v).reshape(-1):
        f = float(x)
        if f != f:
            out.append("nan")
        elif f == float("inf"):
            out.append("inf")
        elif f == float("-inf"):
            out.append("-inf")
        else:
            out.append(f)
    return out


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
    tgt = f"{PAYLOAD['target_module']}.{PAYLOAD['target_qualname']}"
    try:
        target = _load(PAYLOAD["target_module"], PAYLOAD["target_qualname"])
    except Exception as e:
        return _stop(
            f"the target {PAYLOAD['target_module']}."
            f"{PAYLOAD['target_qualname']} could not be imported "
            f"({type(e).__name__}: {e})"
        )
    _note_tool_phase("importing", tgt)
    precondition = None
    pre = None
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
        pre = (
            f"{PAYLOAD['precondition_module']}."
            f"{PAYLOAD['precondition_qualname']}"
        )
        _note_tool_phase("importing", pre)

    print("== the witness, exactly as the solver produced it")

    def build_args():
        """A FRESH input array per call, and one call per execution mode.

        The two modes shared one set of buffers, and a jax buffer is
        destroyable: a target using `jax.jit(..., donate_argnums=0)` — the
        standard step-function idiom — deletes its argument during the
        eager call, so the jit call then raises "Array has been deleted"
        and the file reported the eager answer as the whole answer.
        Measured: it printed DIVERGED and "Nothing here is wrong" for a
        witness at which the compiled form violates. Nothing is shared
        across modes now except the target itself, which is the user's
        program and whose own state is its own behaviour.
        """
        out = []
        for decl, elements in zip(SIDECAR["envelope"],
                                  PAYLOAD["witness_elements"]):
            arr, rounded = _build(np, decl, elements)
            # A JAX ARRAY, because the target is a jax program. A numpy
            # array breaks the most idiomatic write pattern JAX has:
            # `x.at[i].set(v)` does not exist on numpy.ndarray.
            out.append(jnp.asarray(arr))
        return out

    def rounded_note():
        """Which witness values the declared dtype could not hold.

        ITS OWN PASS, holding no state between calls. It used to be a
        closure list appended to from inside `build_args`, which is called
        once per execution mode — so it was guarded with `if not inexact`
        to stop it tripling, and that guard silently dropped every
        declaration after the first one that rounded. Measured on three
        float32 declarations that all round: the note named one. This note
        is the only thing distinguishing the point the file EXECUTED from
        the witness the verdict is about, so under-reporting it lets a
        reader take a rounded value for an exact one.
        """
        lines = []
        for decl, elements in zip(SIDECAR["envelope"],
                                  PAYLOAD["witness_elements"]):
            _, rounded = _build(np, decl, elements)
            lines.extend(
                f"{decl['name']}[{i}] = {elements[i]}" for i in rounded
            )
        return lines

    try:
        args = build_args()
        inexact = rounded_note()
    except Exception as e:
        return _stop(
            f"the witness value(s) could not be built in the declared "
            f"dtype(s) ({type(e).__name__}: {e})"
        )
    for decl, elements, arr in zip(SIDECAR["envelope"],
                                   PAYLOAD["witness_elements"], args):
        for i, t in enumerate(elements):
            print(f"  {decl['name']}[{i}] = {t}")
        print(f"  {decl['name']} as {decl['dtype']}: {np.asarray(arr)!r}")
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
        _note_tool_phase("running the caller precondition", pre)
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

    def evaluate(fn, label, make_args):
        """Both sides, and whether the assertion holds, in one mode."""
        try:
            lhs, rhs = fn(*make_args())
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
                # a side JSON cannot hold is carried as a string ("inf"),
                # and a margin is not defined against one
                margin = (
                    f"   (margin {a - b:+})"
                    if isinstance(a, float) and isinstance(b, float)
                    else ""
                )
                print(f"    [{i}]  {a!r} {rel} {b!r}  is False{margin}")
        return holds, flat_l, flat_r, ""

    # BOTH MODES ARE THE PROGRAM. The compiler is entitled to rewrite the
    # expression, and measurably does: XLA simplifies (1 + x) - 1 to x, so
    # a violation binary64 absorbs eagerly survives compilation. Running one
    # mode and calling it "the program" would report the other one's answer
    # as this one's.
    holds_eager, eager_l, eager_r, eager_why = evaluate(
        target, "eager", build_args)
    holds_jit, jit_l, jit_r, jit_why = evaluate(
        jax.jit(target), "jit", build_args)
    modes = {"eager": holds_eager, "jit": holds_jit}
    ran = [m for m, h in modes.items() if h is not None]
    missing = [m for m, h in modes.items() if h is None]
    if not ran:
        # NEITHER mode ran. Only now is there no execution result — the
        # predecessor stopped as soon as EAGER raised, which is how the
        # standard `.at[].set()` write produced no result at all while jit
        # would have executed it happily.
        return _stop(
            "the target raised at the witness in both execution modes "
            "(eager: " + eager_why + "; jit: " + jit_why + ")",
            reachable,
            modes,
        )
    disagree = len({modes[m] for m in ran}) > 1

    _note_tool_phase("running the target", tgt)

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
    elif missing:
        # EVERY MODE MUST HAVE RUN TO CLAIM DIVERGED. The token says the
        # property holds "in the program's own dtype"; a mode that raised
        # was not measured, and reporting an unmeasured mode as an agreeing
        # one is reporting a result the program did not produce. Nothing
        # was false either, so there is no CONFIRMED to give: this file has
        # no answer here, and says exactly that. Reporting less is the
        # trade this whole feature is built on.
        why = {"eager": eager_why, "jit": jit_why}
        return _stop(
            "the assertion HELD in " + ", ".join(ran) + " and "
            + ", ".join(missing) + " could not run ("
            + "; ".join(why[m] for m in missing)
            + "). DIVERGED needs every mode to have run and held — an "
            "unmeasured mode is not an agreeing one — so no execution "
            "result is claimed",
            reachable,
            modes,
        )
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
