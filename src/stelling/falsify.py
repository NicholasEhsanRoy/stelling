# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The falsification probe: try to break a VERIFIED by RUNNING the program.

**THE ASYMMETRY THIS EXISTS TO ATTACK.** ``stelling`` treats its two
answers differently, and its own source says so.
:class:`stelling.verdict.Witness` records that a REFUTED carries a
concrete violating input and that *"the one independent leg is executing
the witness through the real program: it shares no code with either the
emission or the replay, so it is the only check that catches a plan
defect."* A VERIFIED has no such leg. ``verdict.py``'s render comment
states the reason in one line — *an ``unsat`` is a universal claim with
no witness to replay*. So a wrong encoding that produced a SPURIOUS
witness is caught downstream by execution, while one that MISSED a
violation mints a false VERIFIED with nothing downstream to catch it.
``VERIFIED_BARRED_PRIMITIVES`` is the current mitigation for exactly one
primitive, and it is a hand-maintained policy, not a check.

This module is the missing check, in the only direction a check of this
kind can run: **after a discharge, execute the real program at concrete
points and try to find one that violates the obligation.**

--------------------------------------------------------------------------
WHAT THIS CAN AND CANNOT DO, SAID BEFORE ANYTHING ELSE
--------------------------------------------------------------------------

**THE PROBE CAN ONLY REFUTE.** Finding nothing after any number of
samples is not evidence that the VERIFIED is sound; it is evidence that
this sampler did not find a counterexample, which is a fact about the
sampler. Nothing in this module returns, stamps, or renders a value that
a reader could take as confidence gained. :meth:`ProbeReport.stamp_line`
is deliberately a sentence about WORK DONE — points built, points
executed, obligations left unprobed — and it carries its own disclaimer.
That is a design constraint, not a stylistic one: a verdict that grew a
"probe: clean" line would be a verdict that reads better than it is, and
this project exists to prevent exactly that.

**THE SKIP RATE IS PART OF THE RESULT.** The probe declines on shapes it
cannot sample — an unbounded declaration, a dtype it cannot construct, an
integer box containing no integer, a point at which the program raises.
Every one of those is counted by reason in :attr:`ProbeReport.skips` and
appears in the stamp line. A probe that silently declined most of what it
was pointed at would read as coverage while doing nothing, which is worse
than no probe at all.

**WHAT IT DID WHEN POINTED AT THE WHOLE SUITE.** Forced on for every
``check()`` call across this repository's own tests (jax 0.11.0, 3968
tests), the probe saw 332 VERIFIED verdicts. It executed points on 166 of
them and declined 5 outright — a probe-level decline rate near 3%, the
declines being unbounded or otherwise unsampleable declarations. It fired
**77 times, and every firing was inside a mutation**: the ``pow`` row's
own gauge battery installs deliberately wrong transfers and emissions, and
those are the runs that produced a discharge the program contradicts.
Driven WITHOUT a mutation, every one of those same fixtures returns
REFUTED or UNKNOWN and the probe never fires. So on this corpus the false
alarm count is zero and the instrument demonstrably fires — which is the
pair of facts a default-off flag has to establish before anyone argues
about releasing it, and neither of them is an argument that a VERIFIED it
did not break is any better for it.

--------------------------------------------------------------------------
THE INDEPENDENCE ARGUMENT: WHAT THIS MODULE IS ALLOWED TO IMPORT
--------------------------------------------------------------------------

A probe that reaches its answer through the machinery it is checking is a
second face asking the same wrong question. This repository has already
measured that failure: an adversarial audit produced a witness on a
trivially true property and exact-rational replay CONFIRMED it, because
both faces drove the same routing plan (:class:`stelling.verdict.Witness`).
So the import list below is a soundness argument and is enforced by a test
(``tests/test_falsify_independence.py``), not left to discipline.

**IMPORTED, and why each one is not the thing under test:**

* ``stelling._jax_compat`` — for ``jax``, ``numpy``, and the four
  declaration primitives (``stelling_any`` / ``stelling_assume`` /
  ``stelling_assert`` / ``stelling_nonvacuity``). Unavoidable twice over:
  a program containing those primitives cannot be executed without naming
  them, and ``_jax_compat`` is the one module in this package allowed to
  name jax at all (``tests/test_import_hygiene.py``). What is used from
  jax is its own reference evaluation of its own primitives —
  ``Primitive.bind`` — which is the same thing a user's program does when
  it runs.
* the standard library.

**NOT IMPORTED, and this is the whole point:**

``stelling.propagate``, ``stelling.interval``, ``stelling.affine``,
``stelling.smt``, ``stelling.solvers``, ``stelling.obligation``,
``stelling.exactness``, ``stelling.fidelity``, ``stelling.coverage``,
``stelling.reachability``, ``stelling.vacuity``, ``stelling.verdict``,
``stelling.ir``.

Three of those absences are load-bearing beyond the obvious:

* **``stelling.ir`` is absent, so the transcription is not trusted
  either.** The probe reads every declared box from the ``stelling_any``
  equation of **jax's own jaxpr**, which is where ``any_array`` put it —
  never from the transcribed :class:`stelling.ir.ClosedJaxpr` the analysis
  judged. A transcription that mangled a bound therefore cannot hide from
  the probe by handing it the mangled bound to sample.
* **The integer declared-set rule is re-derived here** (:func:`_window`)
  rather than imported from ``propagate._member_bounds``. An integer
  declaration ``(0.2, 2.8)`` at ``int32`` admits ``{1, 2}``; if the
  propagator's version of that rule is wrong, importing it would make the
  probe sample the same wrong set and agree. The cost of re-deriving is
  that the two can disagree, and the probe is the conservative side of any
  disagreement: :func:`_admissible` re-checks every sampled value against
  the raw declared endpoints before the point is allowed to falsify
  anything, so a probe-side error can lose a refutation but cannot invent
  one.
* **``stelling.verdict`` is absent.** The probe is handed the statuses it
  is trying to break as plain strings by its caller. Reading the CLAIM is
  unavoidable — you cannot falsify a claim without knowing what was
  claimed — but reading it does not require importing the module that
  produced it.

**AND ONE HONEST QUALIFICATION, BECAUSE IT WAS MEASURED RATHER THAN
ASSUMED.** "Does not import" is a statement about this module's own code,
not about what ends up in ``sys.modules``. Two of the banned modules are
loaded anyway, by machinery that runs before the probe does:
``_jax_compat`` imports ``stelling.ir`` at module scope, and ``any_array``
itself does ``from stelling.propagate import _INT_DTYPE_BOUNDS`` while
validating a declaration — so merely TRACING a harness loads the
propagator. Neither is the probe consulting the analysis, and the test
that guards this is therefore a DIFFERENCE: running the probe must load no
analysis module that tracing the same harness had not already loaded. That
is the claim that is both true and worth making.

**WHAT IS SHARED, STATED PLAINLY BECAUSE IT BOUNDS THE PROBE'S REACH.**
The probe and the analysis share exactly one step: **jax's tracer**. Both
start from ``jax.make_jaxpr(harness)``. Everything downstream of that
tracer is independent; nothing upstream of it is. That is not a gap in
the sampling, and no sample budget closes it — it is where the probe's
reach ends, and :ref:`the blind spot <blind-spot>` below names the live
defect that sits in it.

.. _blind-spot:

**THE MEASURED BLIND SPOT: THE INTEGER-LITERAL WRAP.** ``SOUNDNESS.md``
discloses an open false VERIFIED — ``jnp.full((), 256, jnp.int8)`` narrows
to ``0`` while the source says ``256``, so ``x + 256 <= 10.0`` is false at
all eleven declared points and stelling returns VERIFIED. **This probe
does not catch it, and cannot.** Measured, both of that entry's doors, jit
and eager, ``x64`` on and off: the executed program returns ``0.0`` and
the predicate HOLDS at every declared point. It holds because the wrap
happens in jax, at or before the trace, so the program stelling judged and
the program the probe executes are the same program — ``v + 0`` — and both
are faithful to it. Only the SOURCE TEXT says ``256``.

That defect is therefore not an analysis-versus-program disagreement,
which is the axis this probe measures; it is a source-versus-program
disagreement, one layer up. The instrument for that axis already exists
and is a different shape: ``stelling._tripwire``, the trace-time narrowing
gate armed by ``pytest -p stelling.overflow``, which watches the
constant being destroyed as it happens. The two instruments are
complementary and neither subsumes the other. Nothing here should be read
as covering what the tripwire covers.

--------------------------------------------------------------------------
WHAT HAPPENS WHEN IT FIRES, AND WHAT WAS REJECTED
--------------------------------------------------------------------------

**IT RAISES** :class:`VerifiedFalsified`, and returns no verdict at all.

The reasoning is about what kind of fact a firing is. VERIFIED, REFUTED
and UNKNOWN are all claims about the USER'S PROGRAM. A firing is not a
fact about the user's program — it is a fact about **stelling**: the
analysis discharged an obligation that the program violates at a point the
analysis admitted. There is no value of ``status`` that says "the verifier
is unsound here", and inventing a fourth one would put a case into every
consumer's dispatch for an event that must never occur in a released
build. An exception is the language's channel for "this call has no
answer"; it cannot be dropped by a consumer who does not read notes, and
it stops a CI run where it happened.

**REJECTED — return REFUTED with the probe's witness.** This has the best
evidence story of the three: the witness came from executing the real
program, which is a *stronger* standard than a REFUTED's ordinary witness
(a solver model, replayed). It has the worst accounting story, and the
accounting is what matters. It converts a soundness event *in the tool*
into a statement *about the user's code*: the user reads "your program is
wrong", investigates a correct program, and stelling's defect is never
recorded. ``SOUNDNESS.md``'s policy is that silent fixes are forbidden and
every verdict flip is a logged soundness event; a probe that quietly
repaired stelling's own false VERIFIED into a REFUTED would violate that
by construction, and would additionally make this module *look* like a
feature that improves verdicts — which is the "reads as coverage" failure
the whole design is trying to avoid.

**REJECTED — return UNKNOWN with a loud note.** This is the honest-sounding
option and it is the shape the trace-narrowing gate in
``preconditions._pipeline`` already uses. It does not transfer, for two
reasons. First, that gate's UNKNOWN is honest *because it genuinely does
not know* — it refused before propagation. Here we know more than
UNKNOWN: there is a concrete point, executed, that violates the
obligation. Second, and decisively, UNKNOWN is a routine outcome, and this
tree has already measured where notes on routine outcomes go.
``verdict.py``'s own comment on the degraded-portfolio line says it: *"the
stamp records who was asked, the notes carry the failure, and neither is
where a reader looks."* Routing a soundness event into a channel this
repository has measured as unread is building the instrument and throwing
away the signal.

The standing objection to raising is that it is useless in production.
That objection is answered by the dial rather than by the disposition:
:func:`stelling.preconditions.check`'s ``falsify`` keyword is ``None`` by
default and this module is not imported unless it is set. If, after the
audit the principal has reserved, the raise proves too sharp for a
released flag, softening it is a one-line change made deliberately and on
the record. The reverse — discovering months later that a laundered
REFUTED had been hiding a soundness event — is not recoverable.

--------------------------------------------------------------------------
SAMPLING, WHICH IS THE HARD PART
--------------------------------------------------------------------------

Uniform random sampling over a declared box essentially never finds the
interesting point, and the interesting point is row-shaped: for a ``pow``
obligation encoded as ``aux**q == x**p`` the discriminating inputs are
perfect q-th powers, the box endpoints, and the ulp-neighbourhood of the
place the predicate goes tight. The design rule here is that the
**strategy is row-agnostic while its instantiation is read off facts the
program itself declares** — never off a row table, which lives in
``propagate`` and is part of what is being checked.

Five strategies, each measured separately (:attr:`ProbeReport.strategy_hits`):

``endpoints``
    The corners. Every declaration's ``lo`` and ``hi`` as the declaration
    wrote them, plus the midpoint, plus zero when the box contains it, plus
    — for array declarations — single-element spikes, one coordinate at an
    extreme and the rest at the other. Read off ``(lo, hi, shape, dtype)``.
``exact``
    Values at which the program's own arithmetic is exact. Instantiated
    from the exponents the PROGRAM carries: every ``integer_pow[y=k]`` and
    every ``pow`` in the traced jaxpr contributes its k, and the candidate
    set gains the perfect k-th powers inside the box. Plus the integers and
    the powers of two inside the box, which are the exactness facts any
    float program has. This is the strategy the ``pow`` row needs and the
    one uniform sampling cannot supply.
``uniform``
    The control, and it is here to be beaten. Uniform random over the box.
    If the three shaped strategies do not out-hit this one on a corpus with
    known violations, that is a reportable result about them, not a reason
    to delete the control.
``tight``
    Boundary-tightness, by bisection on the executed MARGIN — the signed
    slack of the asserted comparison, read out of the program by
    evaluating the comparison's two operands rather than its boolean.
    Seeded from the admissible point with the least slack found so far and
    walked coordinate-wise toward the box end that reduces it. Uses no row
    facts whatsoever: the margin is a number the program produced.
``ulp``
    The last representable step. ``nextafter`` neighbours, in both
    directions and per declaration, of the least-slack points the other
    strategies reached. This is where a predicate that is tight but true
    over the reals becomes false in floats, and it is the reason the fire
    condition below is what it is.

--------------------------------------------------------------------------
THE FIRE CONDITION, AND WHY IT IS SEMANTICS-AWARE
--------------------------------------------------------------------------

A ``semantics="ieee"`` VERIFIED is a claim about the float the program
computes, so an executed violation refutes it outright.

A ``semantics="real"`` VERIFIED is a claim about the REALS, and the probe
executes in IEEE floats. An executed violation is then not automatically
an unsoundness: the analysis may be perfectly right about ℝ while the
float program lands a half-ulp the other side of a tight bound. Firing on
that would make the probe a machine for manufacturing false alarms about
correct analyses — and since it raises, a false alarm is expensive.

So under ``real`` semantics the probe fires only on a violation it has
some reason to believe is not a rounding artefact, and the test is
stability rather than a tolerance (a tolerance would be a fudge factor
with a number in it that nothing measured):

* if every declared value at the violating point is integral and the
  program's dtypes are integral, the arithmetic is exact and the violation
  is reported; otherwise
* the violation must survive perturbing the point to its ``nextafter``
  neighbours in every declaration — a violation that flips within one ulp
  of the input is a knife-edge and is DECLINED, counted under
  ``precision-ambiguous`` and reported in the stamp line rather than
  dropped.

This is a proxy and is named as one: ulp-stability in the INPUT is not a
proof about rounding in the COMPUTATION. It reduces false alarms; it does
not eliminate them. Under ``ieee`` semantics the filter is not applied,
because there the executed float IS the subject of the claim.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

from stelling._optional import require

# Ask for jax by name before touching the private module that imports it, so
# a caller who set `falsify=` in a jax-less environment is told which extra
# they need from the module they can see. Same posture as `harness.py`.
require("jax")

from stelling._jax_compat import (  # noqa: E402  (must follow the guard above)
    _any_p,
    _assert_p,
    _assume_p,
    _nonvacuity_p,
    jax,
    jex_core,
    np,
)

__all__ = [
    "Declaration",
    "Falsification",
    "FALSIFY_MODES",
    "ProbeReport",
    "STRATEGIES",
    "VerifiedFalsified",
    "probe",
]

# The dial's accepted values, exported so `preconditions` validates against
# ONE definition rather than a second copy of the tuple.  `None` is off and
# is the default everywhere.
FALSIFY_MODES = (None, "sample")

STRATEGIES = ("endpoints", "exact", "uniform", "tight", "ulp")

# Default point budget.  Deliberately modest: the probe runs on every
# VERIFIED when it is switched on, and a budget large enough to be
# interesting for one obligation is large enough to be intolerable for a
# suite.  The number is a cost decision and is not claimed to be a
# sufficiency threshold -- there is no such threshold for a refutation-only
# instrument, which is exactly why the stamp line reports the count.
DEFAULT_BUDGET = 256

# The comparison primitives whose two operands give a signed margin.  An
# obligation whose asserted value came from anything else is still probed
# through its boolean; it simply gets no `tight`/`ulp` phase, and the
# decline is counted.
_MARGIN_RELATIONS = {
    # name: (index of the side that must be SMALL, index of the large side)
    "le": (0, 1),
    "lt": (0, 1),
    "ge": (1, 0),
    "gt": (1, 0),
}

_DECL_PRIMS = (_any_p, _assume_p, _assert_p, _nonvacuity_p)


class VerifiedFalsified(AssertionError):
    """A VERIFIED obligation was violated by executing the real program.

    Raised, never returned as a status -- see this module's docstring for
    the argument and for the two dispositions that were rejected.  It is
    an :class:`AssertionError` because the failed thing is an assertion
    stelling made about a program, and because a bare ``except
    Exception`` in a batch caller should not quietly swallow a soundness
    event that ``except AssertionError`` would have to be written on
    purpose to catch.
    """

    def __init__(self, message: str, report: "ProbeReport"):
        super().__init__(message)
        self.report = report


# --------------------------------------------------------------------------
# what the probe reads off the program
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Declaration:
    """One ``any_array`` declaration, read off jax's own jaxpr.

    ``lo``/``hi`` are the params the ``stelling_any`` equation carries --
    that is, the endpoints ``any_array`` recorded from the caller's own
    bound objects.  They are read here and NOT from the transcribed IR, so
    that a transcription defect cannot steer the sampler onto the box it
    mis-transcribed.
    """

    position: int  # order of the stelling_any equation in the jaxpr
    shape: tuple[int, ...]
    dtype: str
    lo: float
    hi: float

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


@dataclass(frozen=True)
class Falsification:
    """A concrete point at which the real program violated a discharge."""

    strategy: str
    obligation_position: int  # index of the stelling_assert among asserts
    values: tuple[str, ...]  # one repr per declaration, as executed
    margin: float | None  # signed slack, when the obligation compared
    detail: str

    def render(self) -> str:
        margin = "" if self.margin is None else f", margin {self.margin!r}"
        return (
            f"assert #{self.obligation_position} was DISCHARGED by the "
            f"analysis and is FALSE when the program is executed at a "
            f"declared point found by the {self.strategy!r} strategy"
            f"{margin}: {self.detail}"
        )


@dataclass(frozen=True)
class ProbeReport:
    """What the probe did.  A record of WORK, never of confidence.

    There is deliberately no field here that summarises "how well the
    VERIFIED held up", because no such quantity exists: the probe can only
    refute, so its null result carries no information about soundness.
    Every count below is a count of effort spent or of work declined.
    """

    declarations: tuple[Declaration, ...] = ()
    obligations: int = 0
    points_built: int = 0
    points_executed: int = 0
    points_admissible: int = 0  # in-box AND every assume satisfied
    strategy_points: tuple[tuple[str, int], ...] = ()
    strategy_hits: tuple[tuple[str, int], ...] = ()
    skips: tuple[tuple[str, int], ...] = ()
    falsification: Falsification | None = None
    declined: str | None = None  # the whole probe declined; why

    @property
    def skip_rate(self) -> float:
        """Declined work as a fraction of work attempted.

        The denominator is points BUILT, because a point the sampler
        produced and then could not use is exactly the work that a
        silently-skipping probe would hide.  Returns ``1.0`` for a probe
        that declined outright, which is the honest reading: it sampled
        nothing.
        """
        if self.declined is not None:
            return 1.0
        if not self.points_built:
            return 1.0
        return 1.0 - (self.points_admissible / self.points_built)

    def stamp_line(self) -> str:
        """One sentence, about WORK DONE, carrying its own disclaimer.

        Read the wording as a constraint rather than as prose.  It says
        what was executed and what was declined; it does not say the
        verdict is better for having been probed, because it is not.  A
        reader who takes "0 violations found" as evidence of soundness has
        been misled, so the sentence refuses to stop there.
        """
        if self.declined is not None:
            return (
                f"falsification probe: DECLINED, nothing was executed "
                f"({self.declined}). This is not evidence about the "
                f"verdict."
            )
        skipped = ", ".join(f"{n} {why}" for why, n in self.skips)
        tail = f"; declined {skipped}" if skipped else ""
        return (
            f"falsification probe: {self.points_executed} point(s) "
            f"executed, {self.points_admissible} inside the declared set "
            f"and admitted by every assume, across {self.obligations} "
            f"obligation(s){tail}. NO VIOLATION WAS FOUND, WHICH IS NOT "
            f"EVIDENCE THAT THERE IS NONE: this probe can only refute, and "
            f"a null result is a fact about the sampler, not about the "
            f"verdict."
        )


class _Counter:
    """A tiny ordered counter, so the report's tuples are deterministic."""

    def __init__(self) -> None:
        self._d: dict[str, int] = {}

    def add(self, key: str, n: int = 1) -> None:
        self._d[key] = self._d.get(key, 0) + n

    def items(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._d.items()))


# --------------------------------------------------------------------------
# the declared set, re-derived
# --------------------------------------------------------------------------


def _window(decl: Declaration):
    """The closed range this declaration admits, or ``None`` with a reason.

    RE-DERIVED HERE ON PURPOSE, rather than imported from
    ``propagate._member_bounds``.  For an integer dtype the declared set is
    the INTEGERS of the interval intersected with the dtype's range: an
    ``int32`` declared ``(0.2, 2.8)`` admits ``{1, 2}`` and nothing else.
    Importing the propagator's version of that rule would make the probe
    agree with the propagator by construction on precisely the question a
    reachability defect in this repository already got wrong once.

    Returns ``(lo, hi)`` in the dtype's own domain, or ``(None, reason)``.
    """
    try:
        dt = np.dtype(decl.dtype)
    except TypeError:
        return None, "dtype-unconstructible"
    lo, hi = decl.lo, decl.hi
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
        return None, "bound-unreadable"
    if math.isnan(lo) or math.isnan(hi):
        return None, "bound-nan"
    if dt.kind == "b":
        return (0, 1), None
    if dt.kind in "iu":
        # infinite endpoints are legal declarations; they are simply not
        # sampleable, and saying so is better than clamping to the dtype
        # extreme and pretending the box was finite
        if math.isinf(lo) or math.isinf(hi):
            return None, "unbounded-declaration"
        info = np.iinfo(dt)
        a = max(int(math.ceil(lo)), int(info.min))
        b = min(int(math.floor(hi)), int(info.max))
        if a > b:
            return None, "empty-integer-box"
        return (a, b), None
    if dt.kind == "f":
        if math.isinf(lo) or math.isinf(hi):
            return None, "unbounded-declaration"
        if lo > hi:
            return None, "empty-box"
        return (float(lo), float(hi)), None
    return None, "dtype-not-sampleable"


def _admissible(decl: Declaration, arr) -> bool:
    """Is every element of ``arr`` inside the endpoints AS DECLARED?

    The conservative side of any disagreement between :func:`_window` and
    the propagator's own membership rule.  Because a firing RAISES, the
    probe must never be able to invent a refutation out of a point the
    declaration did not admit; a point this rejects is lost, which costs a
    refutation at worst, and that is the right direction to be wrong in.
    """
    a = np.asarray(arr)
    if not np.all(np.isfinite(a.astype("float64", copy=False))):
        return False
    return bool(np.all(a >= decl.lo) and np.all(a <= decl.hi))


# --------------------------------------------------------------------------
# the census: what the program says about itself
# --------------------------------------------------------------------------


@dataclass
class _Census:
    """Everything the probe needs, read off jax's own jaxpr, once."""

    closed: object
    declarations: tuple[Declaration, ...] = ()
    assert_positions: tuple[int, ...] = ()  # eqn index of each stelling_assert
    # for each assert, the (relation, small_atom, large_atom) of the
    # comparison that produced its operand, when there was one
    margins: dict = field(default_factory=dict)
    exponents: tuple[int, ...] = ()  # k of every integer_pow / pow in the program


def _census(harness) -> _Census:
    """Trace the harness with jax and read the program's own declarations.

    ``jax.make_jaxpr`` and NOT ``stelling.harness.trace``: the latter also
    transcribes to :mod:`stelling.ir`, and the transcription is part of
    what the probe is checking.  What comes back here is jax's object,
    which is also the object the probe will execute.
    """
    closed = jax.make_jaxpr(harness)()
    jaxpr = closed.jaxpr
    producer = {}
    for eqn in jaxpr.eqns:
        for ov in eqn.outvars:
            producer[ov] = eqn

    decls: list[Declaration] = []
    asserts: list[int] = []
    margins: dict = {}
    exponents: set[int] = set()

    for i, eqn in enumerate(jaxpr.eqns):
        name = eqn.primitive.name
        if name == "stelling_any":
            p = eqn.params
            decls.append(
                Declaration(
                    position=len(decls),
                    shape=tuple(int(d) for d in p["shape"]),
                    dtype=str(p["dtype"]),
                    lo=p["lo"],
                    hi=p["hi"],
                )
            )
        elif name == "stelling_assert":
            k = len(asserts)
            asserts.append(i)
            src = producer.get(eqn.invars[0])
            if src is not None and src.primitive.name in _MARGIN_RELATIONS:
                small, large = _MARGIN_RELATIONS[src.primitive.name]
                margins[k] = (
                    src.primitive.name,
                    src.invars[small],
                    src.invars[large],
                )

    # THE EXACTNESS FACTS COME FROM THE PROGRAM, NOT FROM A ROW TABLE.
    # `propagate`'s row registry knows that pow is encoded as `aux**q ==
    # x**p`; importing it to learn q would tie the sampler to the module
    # under test.  The exponent is also simply written in the program, so
    # it is read from there.  Sub-jaxprs are walked because a `jax.jit`
    # helper is where users put the arithmetic.
    def walk(jx):
        for eqn in jx.eqns:
            nm = eqn.primitive.name
            if nm == "integer_pow":
                y = eqn.params.get("y")
                if isinstance(y, int) and 2 <= abs(y) <= 64:
                    exponents.add(abs(y))
            elif nm == "pow":
                for atom in eqn.invars:
                    if isinstance(atom, jex_core.Literal):
                        try:
                            v = float(np.asarray(atom.val).reshape(-1)[0])
                        except Exception:  # noqa: BLE001 - a literal we cannot read
                            continue
                        if v == int(v) and 2 <= abs(int(v)) <= 64:
                            exponents.add(abs(int(v)))
            for sub in _sub_jaxprs(eqn):
                walk(sub)

    walk(jaxpr)
    return _Census(
        closed=closed,
        declarations=tuple(decls),
        assert_positions=tuple(asserts),
        margins=margins,
        exponents=tuple(sorted(exponents)),
    )


def _sub_jaxprs(eqn):
    """Every jaxpr nested in an equation's params, however it is wrapped."""
    out = []
    for v in eqn.params.values():
        for cand in (v if isinstance(v, (tuple, list)) else (v,)):
            inner = getattr(cand, "jaxpr", cand)
            if isinstance(inner, jex_core.Jaxpr):
                out.append(inner)
    return out


# --------------------------------------------------------------------------
# the executor: jax's own evaluation of jax's own jaxpr
# --------------------------------------------------------------------------


@dataclass
class _Run:
    """One execution of the program at one point."""

    assumes: list = field(default_factory=list)
    asserts: list = field(default_factory=list)
    margins: dict = field(default_factory=dict)
    raised: str | None = None


def _execute(census: _Census, point) -> _Run:
    """Run the traced program at ``point`` and report what it computed.

    A minimal jaxpr interpreter.  Every equation that is not one of
    stelling's four declaration primitives is evaluated by
    ``Primitive.bind`` -- jax's own implementation, the same one that runs
    when the user calls their function -- so nothing in this loop knows
    anything about intervals, encodings or obligations.  It knows how to
    read a jaxpr and it hands each equation back to jax.

    The four declaration primitives are handled here rather than bound:
    ``stelling_any`` has no concrete implementation (it is a tracing-time
    declaration and its ``def_impl`` says so), and this is where the
    sampled value is substituted for it.  Substituting into a local
    environment rather than monkeypatching the primitive's impl keeps the
    probe free of process-global state, which matters because it may run
    inside a caller's test session.
    """
    jaxpr = census.closed.jaxpr
    env: dict = {}
    run = _Run()

    def read(atom):
        if isinstance(atom, jex_core.Literal):
            return atom.val
        return env[atom]

    for v, c in zip(jaxpr.constvars, census.closed.consts):
        env[v] = c

    decl_i = 0
    assert_i = 0
    try:
        for eqn in jaxpr.eqns:
            prim = eqn.primitive
            name = prim.name
            if prim is _any_p or name == "stelling_any":
                env[eqn.outvars[0]] = point[decl_i]
                decl_i += 1
                continue
            invals = [read(a) for a in eqn.invars]
            if prim is _assume_p or name == "stelling_assume":
                run.assumes.append(np.asarray(invals[0]))
                env[eqn.outvars[0]] = invals[0]
                continue
            if prim is _nonvacuity_p or name == "stelling_nonvacuity":
                # NOT a precondition.  `nonvacuity` states a witness
                # condition about the query, not a constraint the asserted
                # obligation is relative to, so gating points on it would
                # throw away points the obligation genuinely covers.
                env[eqn.outvars[0]] = invals[0]
                continue
            if prim is _assert_p or name == "stelling_assert":
                run.asserts.append(np.asarray(invals[0]))
                spec = census.margins.get(assert_i)
                if spec is not None:
                    _, small, large = spec
                    try:
                        s = np.asarray(read(small), dtype="float64")
                        b = np.asarray(read(large), dtype="float64")
                        run.margins[assert_i] = float(np.min(b - s))
                    except Exception:  # noqa: BLE001 - margin is a nicety
                        pass
                assert_i += 1
                env[eqn.outvars[0]] = invals[0]
                continue
            out = prim.bind(*invals, **eqn.params)
            if prim.multiple_results:
                for var, o in zip(eqn.outvars, out):
                    env[var] = o
            else:
                env[eqn.outvars[0]] = out
    except Exception as exc:  # noqa: BLE001
        # The program raising at a sampled point is information about the
        # point, not about the verdict: a declared box may contain inputs
        # the program refuses.  Counted as a skip, never as a violation.
        run.raised = f"{type(exc).__name__}: {exc}"
    return run


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------


def _fills(decl: Declaration, window, strategy, exponents, rng):
    """Candidate SCALAR fill values for one declaration under one strategy.

    Every strategy is row-agnostic; what differs is which facts it reads.
    ``endpoints`` reads the box, ``exact`` reads the box and the exponents
    the program itself carries, ``uniform`` reads the box and the seed.
    """
    lo, hi = window
    dt = np.dtype(decl.dtype)
    integral = dt.kind in "iub"

    if strategy == "endpoints":
        mid = (lo + hi) / 2
        cands = [lo, hi, mid]
        if lo <= 0 <= hi:
            cands.append(0)
        if integral:
            cands += [lo + 1, hi - 1]
        else:
            cands += [math.nextafter(lo, hi), math.nextafter(hi, lo)]
        return _clean(cands, lo, hi, integral)

    if strategy == "exact":
        cands: list = []
        # the integers of the box: the exactness fact every float program
        # has, and the whole declared set of an integer one
        a, b = math.ceil(lo), math.floor(hi)
        if b - a <= 64:
            cands += list(range(int(a), int(b) + 1))
        else:
            cands += [a, b, 0 if lo <= 0 <= hi else a]
        # powers of two, exact in every binary format
        k = 0
        while k < 64:
            for v in (2.0**k, -(2.0**k), 2.0**-k):
                if lo <= v <= hi:
                    cands.append(v)
            k += 1
            if 2.0**k > max(abs(lo), abs(hi)):
                break
        # PERFECT k-TH POWERS, for each exponent k the PROGRAM carries.
        # This is the pow row's discriminating set, instantiated without
        # ever asking the row registry what pow's encoding is.
        for q in exponents:
            m = 0
            while m <= 4096:
                for v in (float(m) ** q, -(float(m) ** q)):
                    if lo <= v <= hi:
                        cands.append(v)
                if float(m) ** q > max(abs(lo), abs(hi)) and m > 1:
                    break
                m += 1
            # and the k-th roots of the endpoints: the place where a
            # `y**q == x**p` encoding stops being exactly representable
            for end in (lo, hi):
                if end > 0:
                    r = end ** (1.0 / q)
                    for v in (r, math.floor(r), math.ceil(r)):
                        if lo <= v <= hi:
                            cands.append(float(v))
        return _clean(cands, lo, hi, integral)

    if strategy == "uniform":
        n = 8
        if integral:
            return _clean(
                [rng.randint(int(lo), int(hi)) for _ in range(n)], lo, hi, integral
            )
        return _clean([rng.uniform(lo, hi) for _ in range(n)], lo, hi, integral)

    return ()


def _clean(cands, lo, hi, integral):
    """De-duplicate, clamp into the window, and keep the order stable."""
    out = []
    seen = set()
    for c in cands:
        try:
            v = int(round(c)) if integral else float(c)
        except (TypeError, ValueError, OverflowError):
            continue
        if not (lo <= v <= hi):
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return tuple(out)


def _arrays(decl: Declaration, fill, window):
    """Turn one scalar fill into the array(s) it stands for.

    A constant fill always; and for a declaration with more than one
    element, single-coordinate SPIKES -- one element at this fill and the
    rest at the opposite end of the box.  Spikes are what a reduction
    (``sum``, ``max``) and an indexed write (``scatter``) discriminate on,
    and a constant fill can never expose either.
    """
    dt = np.dtype(decl.dtype)
    out = [np.full(decl.shape, fill, dtype=dt)]
    n = decl.size
    if n > 1:
        lo, hi = window
        base = lo if abs(fill - hi) < abs(fill - lo) else hi
        for idx in _spike_indices(n):
            a = np.full(decl.shape, base, dtype=dt)
            a.reshape(-1)[idx] = fill
            out.append(a)
    return out


def _spike_indices(n: int):
    """First, middle and last -- a fixed, tiny, shape-agnostic set.

    Not every index: the point of a spike is that the reduction sees ONE
    outlier, and three positions cover "at the edge of the traversal" and
    "in the middle of it" without turning the budget into O(size).
    """
    return tuple(dict.fromkeys((0, n // 2, n - 1)))


def _points_for(strategy, decls, windows, exponents, rng, budget):
    """Assemble whole points (one array per declaration) for one strategy.

    The diagonal first -- every declaration at its k-th candidate -- then
    the capped cartesian product.  The diagonal matters because "all
    declarations at their upper bound" is the single most productive point
    in most boxes and a truncated product would not reach it until late.
    """
    per = []
    for d, w in zip(decls, windows):
        fills = _fills(d, w, strategy, exponents, rng)
        arrs = []
        for f in fills:
            arrs.extend(_arrays(d, f, w))
        per.append(arrs)
    if not per or any(not a for a in per):
        return []

    points = []
    seen = set()

    def push(combo):
        key = tuple(a.tobytes() for a in combo)
        if key in seen:
            return
        seen.add(key)
        points.append(tuple(combo))

    longest = max(len(a) for a in per)
    for k in range(longest):
        push([a[min(k, len(a) - 1)] for a in per])
        if len(points) >= budget:
            return points[:budget]
    for combo in itertools.product(*per):
        push(list(combo))
        if len(points) >= budget:
            break
    return points[:budget]


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------


def probe(
    harness,
    *,
    statuses,
    semantics="real",
    budget=DEFAULT_BUDGET,
    seed=0,
    strategies=STRATEGIES,
):
    """Try to falsify a discharge by executing the real program.

    ``harness`` is the same nullary callable :func:`stelling.harness.trace`
    was given.  ``statuses`` is the per-obligation status STRING sequence
    the analysis produced -- passed in rather than read off a
    :class:`stelling.verdict.Verdict`, so that this module never imports
    the module whose output it is attacking.  Only obligations whose
    status is ``"discharged"`` are attacked; everything else the analysis
    already declines to claim.

    ``strategies`` restricts the sampler to a subset of
    :data:`STRATEGIES`.  It exists so that each strategy's power can be
    MEASURED alone rather than asserted -- a strategy that never finds
    anything the others would not have found is a strategy that is not
    earning its budget, and there is no way to learn that from a run in
    which they all fire together.  The default is all of them and no
    caller in the library passes anything else.

    Returns a :class:`ProbeReport`.  Raises :class:`VerifiedFalsified` if
    the program violates a discharged obligation at a point inside every
    declaration and admitted by every assume -- see the module docstring
    for why that is a raise and not a status.
    """
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValueError(
            f"unknown falsification strateg(ies) {unknown!r}; known: "
            f"{list(STRATEGIES)}"
        )
    skips = _Counter()
    spoints = _Counter()
    shits = _Counter()
    rng = random.Random(seed)

    try:
        census = _census(harness)
    except Exception as exc:  # noqa: BLE001
        return ProbeReport(declined=f"the harness could not be traced: {exc}")

    targets = [i for i, s in enumerate(statuses) if s == "discharged"]
    if not targets:
        return ProbeReport(declined="no obligation was discharged")
    if len(census.assert_positions) != len(statuses):
        # The analysis records obligations in traversal order and may
        # descend into sub-jaxprs; this walk sees only top-level asserts.
        # When the two counts disagree the probe cannot say WHICH assert an
        # index means, so it declines rather than attacking the wrong one.
        return ProbeReport(
            declined=(
                f"the analysis reported {len(statuses)} obligation(s) and "
                f"the traced program has {len(census.assert_positions)} "
                f"top-level assert(s); the probe cannot pair them"
            )
        )
    if not census.declarations:
        return ProbeReport(declined="the harness declares no inputs to vary")

    windows = []
    for d in census.declarations:
        w, why = _window(d)
        if w is None:
            return ProbeReport(
                declarations=census.declarations,
                declined=f"declaration #{d.position} is not sampleable: {why}",
            )
        windows.append(w)

    report_kw = dict(
        declarations=census.declarations,
        obligations=len(targets),
    )

    built = 0
    executed = 0
    admissible = 0
    best: list = []  # (margin, point) of the least-slack admissible points

    def run_one(strategy, point):
        """Execute one point.  Returns ``(falsification, margin)``.

        ``margin`` is the least slack this point left across the attacked
        obligations, or ``None`` when the point was unusable or the
        obligations do not compare.  It is returned rather than only
        recorded because the ``tight`` search STEERS on it.
        """
        nonlocal built, executed, admissible
        built += 1
        spoints.add(strategy)
        if not all(
            _admissible(d, a) for d, a in zip(census.declarations, point)
        ):
            skips.add("point-outside-declaration")
            return None, None
        jpoint = [jax.numpy.asarray(a) for a in point]
        run = _execute(census, jpoint)
        executed += 1
        if run.raised is not None:
            skips.add("program-raised")
            return None, None
        if len(run.asserts) != len(statuses):
            skips.add("obligation-count-changed")
            return None, None
        if run.assumes and not all(bool(np.all(a)) for a in run.assumes):
            skips.add("assume-unsatisfied")
            return None, None
        admissible += 1
        for k in targets:
            if bool(np.all(run.asserts[k])):
                continue
            verdict = _confirm(census, statuses, point, k, semantics)
            if verdict is None:
                skips.add("precision-ambiguous")
                continue
            shits.add(strategy)
            return (
                Falsification(
                    strategy=strategy,
                    obligation_position=k,
                    values=tuple(repr(np.asarray(a).tolist()) for a in point),
                    margin=run.margins.get(k),
                    detail=verdict,
                ),
                None,
            )
        m = min(
            (run.margins[k] for k in targets if k in run.margins), default=None
        )
        if m is not None:
            best.append((m, point))
        return None, m

    def run_batch(strategy, points):
        for point in points:
            hit, _ = run_one(strategy, point)
            if hit is not None:
                return hit
        return None

    def finish(f=None):
        return ProbeReport(
            **report_kw,
            points_built=built,
            points_executed=executed,
            points_admissible=admissible,
            strategy_points=spoints.items(),
            strategy_hits=shits.items(),
            skips=skips.items(),
            falsification=f,
        )

    share = max(1, budget // max(1, len(strategies)))
    for strategy in ("endpoints", "exact", "uniform"):
        if strategy not in strategies:
            continue
        pts = _points_for(
            strategy, census.declarations, windows, census.exponents, rng, share
        )
        hit = run_batch(strategy, pts)
        if hit is not None:
            _fire(hit, finish(hit))

    # `tight` and `ulp` are SEEDED phases: they refine the least-slack
    # points the first three reached, so when they run alone they must
    # produce their own seeds first.  Those seed points are executed and
    # counted like any others -- a measurement that hid them would credit
    # a strategy with work it did not pay for.
    seeded = [s for s in ("tight", "ulp") if s in strategies]
    if seeded and not best:
        pts = _points_for(
            "endpoints", census.declarations, windows, census.exponents,
            rng, max(4, share // 4),
        )
        hit = run_batch("seed", pts)
        if hit is not None:
            _fire(hit, finish(hit))

    if seeded and not census.margins:
        skips.add("no-margin-no-boundary-search")
    elif seeded:
        tightest = None
        if "tight" in strategies:
            best.sort(key=lambda t: t[0])
            seed = best[0][1] if best else None
            if seed is not None:
                hit, tightest = _tight_search(
                    census, windows, seed, share, run_one
                )
                if hit is not None:
                    _fire(hit, finish(hit))
        if "ulp" in strategies:
            # SEEDED FROM WHERE THE PREDICATE GOES TIGHT, which is the
            # point `tight` converged on -- not from the global least-slack
            # list, which is dominated by corner points whose ulp
            # neighbours `endpoints` already sampled.  Seeding this from
            # the wrong place was measured: 0 hits on every fixture built
            # for it, because every neighbour it proposed had already been
            # tried.
            best.sort(key=lambda t: t[0])
            seeds = ([tightest] if tightest is not None else []) + [
                p for _, p in best[:2]
            ]
            hit = run_batch(
                "ulp", _ulp_points(census, windows, seeds, share)
            )
            if hit is not None:
                _fire(hit, finish(hit))

    return finish(None)


def _fire(hit: Falsification, report: ProbeReport):
    raise VerifiedFalsified(
        "FALSIFICATION PROBE FIRED — stelling is UNSOUND at this query.\n"
        + hit.render()
        + "\n\nNo verdict is returned. This is not a finding about the "
        "program under test: the program did what it does, and the "
        "ANALYSIS discharged an obligation the program violates at a point "
        "the analysis itself admitted. Returning REFUTED would report a "
        "defect in stelling as a defect in your code; returning UNKNOWN "
        "would file a soundness event in the notes. Both were rejected — "
        "see stelling/falsify.py.\n\n"
        + report.stamp_line(),
        report,
    )


def _tight_search(census, windows, seed, budget, run_one):
    """Drive the margin DOWN by coordinate descent, and probe where it lands.

    THE FIRST VERSION OF THIS WAS BISECTION AND IT FOUND NOTHING, which is
    worth recording because the reason is structural rather than a tuning
    miss.  Bisection needs a SIGN CHANGE to bracket, and on a VERIFIED
    there is by construction no point of opposite sign to bracket against
    -- the margin is positive everywhere the analysis looked.  Measured on
    a corpus of four interior violations it hit 0 of 4 while spending the
    largest budget of any strategy.  A boundary search that cannot bracket
    is not a boundary search.

    What the boundary problem actually is: MINIMISE the margin.  The
    obligation goes tight where the program's slack is least, and a
    minimiser needs no sign change to work.  So this sweeps each
    declaration's own box, keeps the least-margin value, and refines
    around it -- a coordinate descent, one coordinate at a time, re-using
    the improved point for the next coordinate.

    Row-agnostic by construction: the only quantity read is the margin the
    PROGRAM produced, and the only move made is to slide one declaration's
    fill along its own declared box.  It knows nothing about what the
    program computes, which is exactly why it transfers across rows.
    """
    spent = 0
    point = list(seed)
    for _sweep in range(2):
        for i, (decl, (lo, hi)) in enumerate(zip(census.declarations, windows)):
            dt = np.dtype(decl.dtype)
            if dt.kind not in "fiub":
                continue
            a, b = float(lo), float(hi)
            if not (b > a):
                continue
            for _refine in range(4):
                # a coarse sweep, then re-centre on the argmin and shrink
                # the window around it.  Nine samples is enough to bracket
                # a single interior minimum and cheap enough to afford
                # four times over.
                grid = [a + (b - a) * t / 8.0 for t in range(9)]
                scored = []
                for v in grid:
                    val = int(round(v)) if dt.kind in "iub" else v
                    if not (lo <= val <= hi):
                        continue
                    trial = list(point)
                    trial[i] = np.full(decl.shape, val, dtype=dt)
                    hit, margin = run_one("tight", tuple(trial))
                    spent += 1
                    if hit is not None:
                        return hit, None
                    if margin is not None:
                        scored.append((margin, val))
                    if spent >= budget:
                        return None, tuple(point)
                if not scored:
                    break
                scored.sort(key=lambda t: t[0])
                centre = float(scored[0][1])
                half = (b - a) / 8.0
                a, b = max(float(lo), centre - half), min(float(hi), centre + half)
                point[i] = np.full(
                    decl.shape,
                    int(round(centre)) if dt.kind in "iub" else centre,
                    dtype=dt,
                )
                if b - a <= 0:
                    break
    return None, tuple(point)


def _ulp_points(census, windows, seeds, budget):
    """The last representable step, in both directions, per declaration."""
    out = []
    for point in seeds:
        for i, (decl, (lo, hi)) in enumerate(zip(census.declarations, windows)):
            dt = np.dtype(decl.dtype)
            base = np.asarray(point[i])
            for direction in (-math.inf, math.inf):
                try:
                    if dt.kind in "iub":
                        step = -1 if direction < 0 else 1
                        moved = base.astype("int64") + step
                        if not np.all((moved >= lo) & (moved <= hi)):
                            continue
                        arr = moved.astype(dt)
                    else:
                        arr = np.nextafter(base, np.asarray(direction, dtype=dt))
                        if not np.all((arr >= lo) & (arr <= hi)):
                            continue
                except (TypeError, ValueError, OverflowError):
                    continue
                nxt = list(point)
                nxt[i] = arr
                out.append(tuple(nxt))
                if len(out) >= budget:
                    return out
    return out


def _confirm(census, statuses, point, k, semantics):
    """Decide whether an executed violation may be REPORTED, or is a tie.

    Returns the detail string when the violation stands, or ``None`` when
    it is a knife-edge that the probe declines to fire on.  See the module
    docstring's fire-condition section: under ``ieee`` the executed float
    IS the subject of the claim and the violation stands as it is; under
    ``real`` the claim is about ℝ and a violation that flips within one
    ulp of the input is not distinguishable, by this instrument, from the
    analysis being right about ℝ and the float landing the other side.
    """
    detail = (
        f"the obligation evaluated FALSE at this point; the declared box, "
        f"every assume, and the obligation itself were all evaluated by "
        f"executing the program"
    )
    if semantics == "ieee":
        return detail
    integral = all(
        np.dtype(d.dtype).kind in "iub" for d in census.declarations
    )
    if integral:
        return detail + " (exact integer arithmetic: no rounding involved)"

    # ulp-stability, as a PROXY and named as one
    neighbours: list = []
    for i, decl in enumerate(census.declarations):
        dt = np.dtype(decl.dtype)
        if dt.kind != "f":
            continue
        base = np.asarray(point[i])
        for direction in (-math.inf, math.inf):
            try:
                arr = np.nextafter(base, np.asarray(direction, dtype=dt))
            except (TypeError, ValueError):
                continue
            if not _admissible(decl, arr):
                continue
            nxt = list(point)
            nxt[i] = arr
            neighbours.append(tuple(nxt))
    if not neighbours:
        return detail + " (no ulp-neighbour inside the box to test against)"
    for nb in neighbours:
        run = _execute(census, [jax.numpy.asarray(a) for a in nb])
        if run.raised is not None or len(run.asserts) != len(statuses):
            return None
        if run.assumes and not all(bool(np.all(a)) for a in run.assumes):
            continue
        if bool(np.all(run.asserts[k])):
            return None
    return detail + " (and at every ulp-neighbour of it inside the box)"
