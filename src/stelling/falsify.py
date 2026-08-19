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
integer box containing no integer, a point at which the program raises —
and it declines on executed VIOLATIONS the fire condition will not stand
behind. Every one of those is counted by reason in
:attr:`ProbeReport.skips`, adjudicated in
:attr:`ProbeReport.adjudications`, and named in the stamp line. A probe
that silently declined most of what it was pointed at would read as
coverage while doing nothing, which is worse than no probe at all.

The declined violation is the half that used to be missing, and it was
the expensive half: on ``x + 1.0 > x`` over ``[0, 2**54]`` the report read
*120 points executed, skip rate 0.0000* while 32 of those points had
evaluated the obligation FALSE and been dropped. The rate now counts them
and the stamp line says the number in the same sentence as the counts.

**WHAT IT DID WHEN POINTED AT THE WHOLE SUITE.** Forced on for every
``check()`` call across this repository's own tests (jax 0.11.0, x64 on,
one process, ``-p no:randomly``, a plugin that sets ``falsify="sample"``
in the pipeline and swallows the raise so the run reaches the end), the
probe was handed **500 VERIFIED verdicts**. It declined **8** outright —
unbounded or otherwise unsampleable declarations — and on the rest it
built 31,340 points, executed 31,336, and found 24,914 of them inside
every declaration and admitted by every assume. It fired **169 times**,
across six files:

======================================  =======  ==============================
file                                    firings  what it is
======================================  =======  ==============================
``tests/test_falsify_probe.py``              81  the probe's own fixtures,
                                                 false by hand
``tests/test_pow_row_gauge_jax.py``          48  a mutation battery
``tests/test_square_row_gauge_jax.py``       24  a mutation battery
``tests/test_falsify_fire_condition.py``     11  the probe's own fixtures,
                                                 false by hand
``tests/test_scatter_gauge_jax.py``           3  a mutation battery
``tests/test_falsify_independence.py``        2  the lying propagator
======================================  =======  ==============================

**Every firing is inside a deliberate mutation or a by-construction-false
fixture**, and the attribution is spelled out rather than summarised
because the earlier version of this paragraph said *"every firing was
inside the ``pow`` row's own gauge battery"* and reported 77 firings in
four figures that no longer re-derive. Five files fire, not one; the
``pow`` gauge is 48 of 169. (The two ``falsify`` test files' own counts
are measured under an instrumentation that swallows the raise, which
changes those tests' control flow — the three GAUGE rows are not affected
by it, and they are identical before and after this batch's fire-condition
rework: 48, 24 and 3 both times.) Driven WITHOUT the mutation, every one
of those gauge fixtures returns REFUTED or UNKNOWN and the probe never
fires.

So on this corpus the false-alarm count is zero and the instrument
demonstrably fires — which is the pair of facts a default-off flag has to
establish before anyone argues about releasing it, and neither of them is
an argument that a VERIFIED it did not break is any better for it. **Zero
false alarms ON THIS CORPUS is also not zero false alarms**: the corpus
had none of the shape that reached one, and the fire condition this module
shipped with raised "stelling is UNSOUND" on a correct VERIFIED that four
lines of ordinary compensated summation produce. What that cost, and what
replaced the test that caused it, is the fire-condition section below.

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
  the declared endpoints before the point is allowed to falsify anything,
  so **a SAMPLER error can lose a refutation but cannot invent one.**

  That sentence used to end at "a probe-side error", and as a claim about
  the whole probe it is FALSE. :func:`_window` (which builds points) and
  :func:`_admissible` (which guards them) read the same
  ``Declaration.lo``/``hi`` off the same :func:`_census`: one reading, not
  two, so the guard is a check on the sampler and not on the census.
  Measured — mutating ``_census``'s ``hi=p["hi"]`` to ``hi=p["hi"] * 2 +
  1`` produced **four false alarms on correct VERIFIEDs**, because both
  halves were steered by the same wrong number. No second reading exists
  inside a module whose value depends on importing none of the analysis,
  which is the trade this design makes; what pins the census is a TEST,
  which is under no such constraint and compares the probe's reading
  against the transcription:
  ``tests/test_falsify_independence.py::test_the_census_reading_of_the_declared_box_is_pinned_against_the_IR``.
* **``stelling.verdict`` is absent.** The probe is handed the statuses it
  is trying to break as plain strings by its caller. Reading the CLAIM is
  unavoidable — you cannot falsify a claim without knowing what was
  claimed — but reading it does not require importing the module that
  produced it.

**AND ONE HONEST QUALIFICATION, BECAUSE IT WAS MEASURED RATHER THAN
ASSUMED.** "Does not import" is a statement about this module's own code,
not about what ends up in ``sys.modules``. **FIVE** of the banned modules
are loaded anyway, by machinery that runs before the probe does — counted,
because the sentence here used to say two: ``import stelling.harness``
brings in ``stelling.ir`` (``_jax_compat`` imports it at module scope),
and merely TRACING a harness adds ``stelling.propagate``,
``stelling.interval``, ``stelling.coverage`` and ``stelling.exactness``,
because ``any_array``'s own dtype validation does ``from
stelling.propagate import _INT_DTYPE_BOUNDS``.

None of that is the probe consulting the analysis, so the test that guards
it is a DIFFERENCE: running the probe must load no analysis module that
tracing the same harness had not already loaded. **That difference is
blind to exactly those five**, which is the set that matters, and it is
therefore not the whole test: a second measurement replaces each
already-loaded module with a recording proxy for the duration of the probe
call and refuses any attribute read whose immediate frame is this file.
Both mutations that defeated the difference alone — ``_window`` reading
``sys.modules["stelling.propagate"]._INT_DTYPE_BOUNDS``, and
``_admissible`` delegating to ``stelling.interval.from_bounds`` — fail the
proxy.

**WHAT IS SHARED, STATED PLAINLY BECAUSE IT BOUNDS THE PROBE'S REACH.**
The probe and the analysis share **jax's tracer and the declaration API**.
Both start from ``jax.make_jaxpr(harness)``, and both read a box that
``any_array`` — not either of them — decided to record.

The second half of that sentence is a correction: this paragraph used to
say "exactly one step: jax's tracer". ``any_array`` validates a
declaration through ``propagate._INT_DTYPE_BOUNDS``, so the propagator has
a vote on which declarations exist at all. Measured: with that table
replaced by ``{"int8": (-500, 500)}`` the declaration ``any_array((),
"int8", (200.0, 300.0))`` stops being refused and the probe reads and
samples a box that on the clean tree never reaches it. Everything
DOWNSTREAM of the declaration API is independent; nothing upstream of it
is. That is not a gap in the sampling, and no sample budget closes it —
it is where the probe's reach ends, and :ref:`the blind spot <blind-spot>`
below names the live defect that sits in it.

.. _blind-spot:

**THE MEASURED BLIND SPOT: THE INTEGER-LITERAL WRAP.** ``SOUNDNESS.md``
discloses an open false VERIFIED — ``jnp.full((), 256, jnp.int8)`` narrows
to ``0`` while the source says ``256``, so ``x + 256 <= 10.0`` is false at
all eleven declared points and stelling returns VERIFIED. **This probe
does not catch it, and cannot.** Measured, both of that entry's doors, jit
and eager, ``x64`` on and off: the executed program returns ``0.0`` and
the predicate HOLDS at every declared point.

The reason is sharper than "the wrap happens in the tracer", which is what
this paragraph used to say and is not accurate for the ``jnp.full`` door:
``jnp.full((), 256, jnp.int8)`` evaluates to ``0`` **EAGERLY, before any
trace exists**, and so does ``jnp.array(5, jnp.int8) + 256`` (it is ``5``);
numpy by contrast raises ``OverflowError`` on ``np.int8(256)``. **There is
no executable form of the program — traced or eager — in which 256
survives.** So an execution probe of any budget or design cannot see it;
only the source text, or a hook at the moment of the narrowing, can. The
accurate phrasing is the commit message's, *at or before the trace*.

That defect is therefore not an analysis-versus-program disagreement,
which is the axis this probe measures; it is a source-versus-program
disagreement, one layer up. The instrument for that axis already exists
and is a different shape: ``stelling._tripwire``, the trace-time narrowing
gate armed by ``pytest -p stelling.overflow``, which watches the
constant being destroyed as it happens. The two instruments are
complementary and neither subsumes the other. Nothing here should be read
as covering what the tripwire covers.

**AND THE BLIND SPOT IS SPECIFICALLY TRACE-TIME CONSTANT DESTRUCTION, NOT
INTEGER WRAPPING.** With ``propagate``'s integer guard out of the way the
probe DOES catch an ``int8`` runtime wrap: ``x + y >= 0`` over ``int8
[0, 100]**2`` is true over ℤ, the program computes ``-56`` at ``(100,
100)``, and the probe reports it. Pinned by
``tests/test_falsify_fire_condition.py::test_the_integer_branch_is_not_a_rational_replay``,
which also pins the reason that catch survives the fire-condition rework:
exact-rational arithmetic does not wrap, so the all-integral branch must
never be routed through the rational replay.

**A SECOND REACH GAP, NAMED HERE BECAUSE A DECLINE COUNT IS NOT A
DISCLOSURE: ``bfloat16`` AND THE ``float8`` FAMILY.** Every declaration in
one of those formats is declined outright with ``dtype-not-sampleable``,
so the probe has ZERO reach exactly where format-rounding defects are most
likely. The cause is mechanical rather than principled — numpy classifies
the ml_dtypes extension types as ``kind == "V"`` (and some of the
``float8`` family have no ``finfo``), while every phase of this sampler
steps with ``nextafter`` and measures with ``finfo`` in the declaration's
own format — but a reach gap a reader has to infer from a decline count is
a reach gap that reads as coverage. Closing it is a sampler that steps in
each of those grids, which is a feature and not a repair, and it is not in
this module today.

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

**WHERE IT CAN BE TURNED ON.** All three public doors that mint a
VERIFIED: :func:`stelling.preconditions.check`,
:func:`stelling.contracts.check_contract` and
:func:`stelling.inductive.check_inductive_step`. It reached only the first
when it landed, and the other two run the same ``_pipeline`` — so the
probe's reach was an accident of which function had been given the keyword
rather than a decision about what is worth checking. The decision is that
a VERIFIED is a VERIFIED whichever door minted it.

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

**WHICH OF THEM EARN THEIR PLACE, AND ON WHICH ARGUMENT.** Two of the five
have a fixture no other strategy reaches (``tight``, ``ulp``); the other
three do not, and are kept on COST, which is a weaker claim and is
asserted as the weaker claim. Measured over
``tests/test_falsify_probe.py``'s live corpus, per single-strategy
configuration — executions, then fixtures reached out of six:
``endpoints`` 20/4, ``ulp`` 26/4, ``uniform`` 28/3, ``exact`` 41/5,
``tight`` 49/6, all five together 55/6. So ``endpoints`` is the cheapest
configuration there is and ``exact`` reaches the ``pow`` perfect-power
shape in less than half of ``tight``'s executions. Both of those are now
pinned by a test; the ``endpoints`` half was claimed and unmeasured.

--------------------------------------------------------------------------
THE FIRE CONDITION, AND WHY IT IS SEMANTICS-AWARE
--------------------------------------------------------------------------

A ``semantics="ieee"`` VERIFIED is a claim about the float the program
computes, so an executed violation refutes it outright.

A ``semantics="real"`` VERIFIED is a claim about the REALS, and the probe
executes in IEEE floats. An executed violation is then not automatically
an unsoundness: the analysis may be perfectly right about ℝ while the
float program lands the other side of a tight bound. Firing on that would
make the probe a machine for manufacturing false alarms about correct
analyses — and since it raises, a false alarm is expensive.

**THE FIRST VERSION OF THIS TEST WAS ULP-STABILITY OF THE INPUT, AND IT
FALSE-ALARMED ON FOUR LINES OF ORDINARY NUMERICAL CODE.** The rule was
that a violation had to survive perturbing the point to its ``nextafter``
neighbours in every declaration. It is a good test for a ONE-ULP artefact
— ``(x/3)*3``, ``(x*x)/x``, ``x*0.1*10`` and ``sqrt(x)**2`` all have a
longest consecutive violating run of a single float, and all four are
correctly declined — and no test at all for COARSE QUANTISATION::

    y = any_array((), "float64", (0.0, 2.0))
    s = 1e16
    assert_((s + y) - s <= y)          # Kahan/Neumaier compensation

``1e16`` is exactly ``10**16``, float64's spacing there is ``2.0``, and
``(s + y) - s`` is ``2.0`` across a band about 4.5e15 ulps wide — every
point of it perfectly stable under a one-ulp input perturbation. Over ℝ
the expression IS ``y``, so the obligation is true, both solvers answered
unsat, and the verdict was RIGHT. The probe raised *"stelling is UNSOUND
at this query"* on it, and a soundness alarm that reports our defect as
the caller's, in the one message a reader acts on, is worse than no alarm.
It was reachable BECAUSE of this batch's own reach improvement: handing
the probe the VERDICT's statuses lets it attack solver-decided
discharges, and the solver reasons over ℝ, which is exactly where ℝ and
floats diverge most.

**WHAT DECIDES NOW.** In order:

* **``ieee`` semantics: nothing decides.** The executed float IS the
  subject of the claim and the violation stands as it is.
* **every declaration integral: exact integer arithmetic decides.** No
  rounding is involved and the violation stands. This branch MUST NOT
  become a rational replay — see the blind-spot note above: ℚ does not
  wrap, and replaying an ``int8`` program in ℚ would report values it
  never computed and would then declare a genuine runtime wrap an
  artefact.
* **otherwise: EXACT-RATIONAL REPLAY of the same traced jaxpr at the same
  point** (:func:`_replay`). Every finite float is a rational and
  ``Fraction(float)`` is exact, so the program can be re-evaluated over ℚ
  with each primitive carrying its REAL meaning. False over ℚ as well as
  in floats — the analysis discharged something false about ℝ, and the
  firing stands. True over ℚ — the violation was manufactured by rounding
  and is declined, counted under ``float-rounding-artefact``. This is the
  same standard :class:`stelling.verdict.Witness` already applies to a
  REFUTED, and it is ``fractions`` and nothing else: no analysis module is
  imported to do it, which is why it is an interpreter here rather than a
  call into ``stelling.exactness``.
* **where the replay ABSTAINS, the old ulp proxy decides.** ``exp``,
  ``log``, ``sin`` and a fractional ``pow`` are irrational at almost every
  rational argument; an integer intermediate that left its dtype wrapped
  and ℚ does not; a primitive with no rational reading is refused rather
  than guessed. On any of those the fire condition degrades to the weaker
  test rather than declining everything it cannot prove — and it COUNTS
  that it did, in :attr:`ProbeReport.adjudications` and in the firing
  message, because a fire condition that degraded silently would be the
  same defect one layer down.

The proxy is still named a proxy, because it still is one: stability of
the INPUT is not a proof about rounding in the COMPUTATION. What changed
is that it is now the fallback rather than the rule, and that every
firing says which of the two admitted it.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from fractions import Fraction

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
    "DECLINE_REASONS",
    "ProbeReport",
    "SEED_LABEL",
    "STRATEGIES",
    "VerifiedFalsified",
    "probe",
]

# The dial's accepted values.  `preconditions.check` spells the same pair
# out as a literal rather than importing this, because importing this
# module imports jax and the dial has to be validated in a jax-less
# environment too; the two spellings are pinned to each other by
# `tests/test_falsify_fire_condition.py::test_the_dial_has_ONE_definition_and_the_second_spelling_is_pinned`,
# which is what makes "one definition" a fact rather than a comment.
# `None` is off and is the default everywhere.
FALSIFY_MODES = (None, "sample")

STRATEGIES = ("endpoints", "exact", "uniform", "tight", "ulp")

# EVERY DECLINE REASON THIS MODULE CAN EMIT, in one place.  Not consumed
# by the code -- the emission sites keep their literals, because a reason
# read three lines from the branch that produces it is a reason a reviewer
# checks -- but pinned to those literals and to a test apiece by
# `tests/test_falsify_fire_condition.py`.  Ten of the thirteen this module
# shipped with appeared in NO test, which is how a decline reason drifts
# into meaning something other than what it says.
DECLINE_REASONS = (
    # `_window`: the declared set cannot be sampled at all
    "dtype-unconstructible",
    "bound-unreadable",
    "bound-nan",
    "unbounded-declaration",
    "empty-integer-box",
    "empty-box",
    "box-outside-the-dtype-range",
    "dtype-not-sampleable",
    # a built point that could not be used
    "point-outside-declaration",
    "program-raised",
    "obligation-count-changed",
    "assume-unsatisfied",
    # an executed VIOLATION the fire condition would not report
    "precision-ambiguous",
    "float-rounding-artefact",
    "assume-unsatisfied-over-the-rationals",
    # the boundary phases had nothing to steer on
    "no-margin-no-boundary-search",
)

# `tight` and `ulp` refine points the first three strategies reached, so
# when they are run without them they must generate their own starting
# points.  Those points are executed and counted like any others, under
# THIS label rather than under a strategy name -- because attributing them
# to `tight` would credit a margin search with reach its seeds supplied.
# It is a label and not a strategy: it is not in `STRATEGIES`, it cannot
# be requested, and it reaches the user-facing firing message, which is
# why it is named and exported rather than left as a bare string literal.
SEED_LABEL = "seed"

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
    # WHICH TEST LET THIS THROUGH.  A firing adjudicated by exact-rational
    # replay and one adjudicated by the ulp proxy are not the same claim,
    # and the message a reader acts on has to say which it is.
    adjudication: str = "unrecorded"

    def render(self) -> str:
        margin = "" if self.margin is None else f", margin {self.margin!r}"
        how = (
            f" [{self.strategy!r} is a sampling strategy; "
            f"{SEED_LABEL!r} labels the starting points a seeded strategy "
            f"generates for itself and is not one]"
            if self.strategy == SEED_LABEL
            else ""
        )
        return (
            f"assert #{self.obligation_position} was DISCHARGED by the "
            f"analysis and is FALSE when the program is executed at a "
            f"declared point found by the {self.strategy!r} strategy"
            f"{how}{margin}, and the violation was admitted by the "
            f"{self.adjudication!r} test: {self.detail}"
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
    points_declined: int = 0  # admissible, VIOLATED, and not reported
    violations_seen: int = 0  # executed points at which an attacked
    # obligation evaluated FALSE in floats, however they were then judged
    adjudications: tuple[tuple[str, int], ...] = ()
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

        **A DECLINED VIOLATION IS DECLINED WORK, AND IT USED NOT TO
        COUNT.** Measured on ``x + 1.0 > x`` over ``[0, 2**54]``: 120
        points executed, 32 of them points at which the obligation
        evaluated FALSE and the probe declined to report it, and the rate
        this property returned was ``0.0000`` -- a number that reads as
        "nothing was skipped" on a run whose most interesting 32 results
        were exactly the ones dropped.  The numerator is therefore the
        admissible points the probe actually drew a conclusion from, and
        ``points_declined`` is subtracted from it.
        """
        if self.declined is not None:
            return 1.0
        if not self.points_built:
            return 1.0
        used = self.points_admissible - self.points_declined
        return 1.0 - (used / self.points_built)

    def stamp_line(self) -> str:
        """One sentence, about WORK DONE, carrying its own disclaimer.

        Read the wording as a constraint rather than as prose.  It says
        what was executed and what was declined; it does not say the
        verdict is better for having been probed, because it is not.  A
        reader who takes "0 violations found" as evidence of soundness has
        been misled, so the sentence refuses to stop there.

        **THREE BRANCHES, AND THE FIRING ONE USED TO BE MISSING.**  This
        method is called from :func:`_fire` as well as from the VERIFIED
        path, and with only the two branches below it appended *"NO
        VIOLATION WAS FOUND"* to the message whose first line is "the
        probe fired" -- the module's single most important message
        contradicting itself in its own tail.

        The second correction is the declined violation.  A run that
        executed a violation and declined to report it must SAY the number,
        in the same sentence, and must not follow it with a phrase that
        reads as "nothing was there": those are precisely the points where
        the answer was not "no violation" but "a violation this instrument
        will not stand behind".
        """
        if self.declined is not None:
            return (
                f"falsification probe: DECLINED, nothing was executed "
                f"({self.declined}). This is not evidence about the "
                f"verdict."
            )
        skipped = ", ".join(f"{n} {why}" for why, n in self.skips)
        tail = f"; declined {skipped}" if skipped else ""
        head = (
            f"falsification probe: {self.points_executed} point(s) "
            f"executed, {self.points_admissible} inside the declared set "
            f"and admitted by every assume, across {self.obligations} "
            f"obligation(s){tail}"
        )
        if self.falsification is not None:
            return (
                f"{head}. A VIOLATION WAS FOUND AND IS REPORTED ABOVE; "
                f"this line is the work that reached it and nothing more."
            )
        if self.points_declined:
            how = ", ".join(f"{n} {why}" for why, n in self.adjudications)
            return (
                f"{head}. {self.points_declined} EXECUTED VIOLATION(S) WERE "
                f"DECLINED, NOT ABSENT: at those points the obligation "
                f"evaluated FALSE and the probe would not report it "
                f"({how}). Every other point left the obligation true, "
                f"WHICH IS NOT EVIDENCE THAT THERE IS NO VIOLATION: this "
                f"probe can only refute, and a null result is a fact about "
                f"the sampler, not about the verdict."
            )
        return (
            f"{head}. NO VIOLATION WAS FOUND, WHICH IS NOT "
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
        # INTERSECTED WITH THE FORMAT, exactly as the integer branch above
        # intersects with `np.iinfo`.  A declaration is a set of values of
        # its own dtype, and `float16` has no value at 1e5; sampling one
        # anyway meant `np.full(shape, 1e5, dtype="float16")`, whose
        # `RuntimeWarning: overflow encountered in cast` turned a green
        # VERIFIED into a crash under `-W error::RuntimeWarning` -- an
        # ordinary CI setting.  Both endpoints of an ordinary `float64`
        # declaration are inside `finfo(float64).max`, so this is a no-op
        # everywhere except the narrow formats, which is where it was
        # measured to matter.
        try:
            limit = float(np.finfo(dt).max)
        except (ValueError, TypeError):
            # a float-kinded dtype numpy has no `finfo` for -- some of the
            # `float8` family arrive here rather than at the `V` branch
            # below, and the sampler cannot step in a format it cannot ask
            # the extent of
            return None, "dtype-not-sampleable"
        a, b = max(float(lo), -limit), min(float(hi), limit)
        if a > b:
            return None, "box-outside-the-dtype-range"
        return (a, b), None
    # `bfloat16` and the `float8_*` family land here: numpy classifies the
    # ml_dtypes extension types as `kind == "V"`, they have no `nextafter`,
    # and every phase of this sampler is built on stepping and comparing in
    # the declaration's own format.  That is a REACH GAP and it is named in
    # the module docstring's blind-spot section rather than left to be
    # discovered from a decline count.
    return None, "dtype-not-sampleable"


def _admissible(decl: Declaration, arr) -> bool:
    """Is every element of ``arr`` inside the endpoints AS DECLARED?

    The conservative side of any disagreement between :func:`_window`'s
    re-derived declared set and the point the sampler actually built.
    Because a firing RAISES, the sampler must never be able to hand the
    fire condition a point the declaration did not admit; a point this
    rejects is lost, which costs a refutation at worst, and that is the
    right direction to be wrong in.

    **WHAT IT DOES NOT GUARD, SAID PLAINLY, BECAUSE THE DOCSTRING USED TO
    CLAIM MORE.**  This reads ``decl.lo``/``decl.hi`` -- the same two
    numbers :func:`_window` reads, from the same :class:`Declaration`
    :func:`_census` built.  That is ONE reading, not two, so it is not a
    check on the census: an audit mutation that doubled ``hi`` in
    ``_census`` produced FOUR false alarms on correct VERIFIEDs, because
    the sampler and this guard were both steered by the same wrong number.
    What pins the census reading is a test, which is allowed to import the
    analysis and compare the two:
    ``tests/test_falsify_independence.py::test_the_census_reading_of_the_declared_box_is_pinned_against_the_IR``.
    No second reading is available inside a module whose value depends on
    importing none of the analysis.

    **AND IT HAD NEVER REJECTED A POINT ON THE LIVE CORPUS** -- 0 of the
    30,194 points the corpus built before this batch, measured; the four
    rejections it shows now are the ones that test manufactures.  That is what a guard against a sampler
    defect looks like when the sampler has no defect, not evidence it is
    unnecessary, so
    ``tests/test_falsify_fire_condition.py::test_the_admissibility_guard_rejects_a_point_the_sampler_should_not_build``
    drives it with a sampler that leaves the box, and the rejection path
    has now been seen to fire.
    """
    a = np.asarray(arr)
    if a.size == 0:
        return True
    if a.dtype.kind == "f":
        # WIDEN THE ARRAY, NEVER NARROW THE ENDPOINT.  `np.all(a <=
        # decl.hi)` on a `float16` array against a `float64` endpoint casts
        # the ENDPOINT into float16 -- which both emits `RuntimeWarning:
        # overflow encountered in cast` (an exception under the ordinary CI
        # setting `-W error::RuntimeWarning`, and it escaped a green
        # VERIFIED) and gives the WRONG ANSWER, because a `hi` of 1e5
        # becomes `inf` and admits every float16 there is.  Widening is
        # exact for every binary format numpy classifies as `f`.
        w = a.astype("float64", copy=False)
        if not np.all(np.isfinite(w)):
            return False
        return bool(np.all(w >= decl.lo) and np.all(w <= decl.hi))
    if a.dtype.kind in "iub":
        # and for integers the comparison is done in PYTHON, whose
        # int-against-float comparison is exact: an int64 value above 2**53
        # does not survive a float64 cast, and a membership test that
        # rounded its own operand is not a membership test.
        return bool(
            int(np.min(a)) >= decl.lo and int(np.max(a)) <= decl.hi
        )
    return False


def _representable(v, dt) -> bool:
    """Can ``dt`` hold ``v`` as a finite value?

    A candidate fill outside the dtype's own finite range is not a value
    of that dtype, so it is not a point of the declared set and must be
    dropped BEFORE anything tries to build an array out of it.  Dropping
    it before the cast rather than after is not tidiness: ``np.full(shape,
    1e5, dtype="float16")`` emits ``RuntimeWarning: overflow encountered
    in cast``, and under ``-W error::RuntimeWarning`` -- an ordinary CI
    setting -- that warning became an exception that escaped a green
    VERIFIED and crashed the caller's run.  Measured on this tree, a
    ``float16`` declaration of ``(0.0, 1e5)`` crashed there and one of
    ``(-65504.0, 65504.0)`` crashed in ``np.nextafter``, while the probe's
    only interest in either value was to discard it.

    :func:`_window` now intersects the declared interval with this range
    before any fill is generated, so an in-window candidate is
    representable by construction; this stays as the second, cheap guard
    at the point of construction, because the failure it prevents is a
    warning in the CALLER's process rather than a wrong answer here.
    """
    if dt.kind == "f":
        try:
            limit = float(np.finfo(dt).max)
        except (ValueError, TypeError):
            return False
        return math.isfinite(v) and abs(v) <= limit
    if dt.kind in "iu":
        info = np.iinfo(dt)
        return int(info.min) <= v <= int(info.max)
    return dt.kind == "b"


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
    replay_cost: int = 0  # element-visits one exact-rational replay would do


def _census(harness) -> _Census:
    """Trace the harness with jax and read the program's own declarations.

    ``jax.make_jaxpr`` and NOT ``stelling.harness.trace``: the latter also
    transcribes to :mod:`stelling.ir`, and the transcription is part of
    what the probe is checking.  What comes back here is jax's object,
    which is also the object the probe will execute.
    """
    return _read(jax.make_jaxpr(harness)())


def _read(closed) -> _Census:
    """Read a traced program's declarations, asserts, margins and exponents.

    Split from :func:`_census` so that :func:`probe` can tell a TRACE
    failure from a READ failure.  It used to catch both under one
    ``except`` and report both as *"the harness could not be traced"*, and
    that mattered because the read had a live defect of its own (the
    literal-operand crash pinned just below) -- so a note reaching the user
    named the wrong stage, on a trace that had succeeded.
    """
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
            # THE OPERAND CAN BE A LITERAL, and a `dict.get` on one raises
            # `TypeError: unhashable type: 'Literal'`.  Hit twice by this
            # tree's own corpus -- `assert_` on a value the tracer folded
            # to a constant, e.g.
            # `tests/test_contracts.py::test_t2_constant_transform_returns_produce_verdicts`.
            # A literal has no producing equation, so
            # there is no margin to read; that is the whole answer, and the
            # crash was `_execute`'s `read()` handling literals while this
            # walk did not.
            operand = eqn.invars[0]
            src = (
                None
                if isinstance(operand, jex_core.Literal)
                else producer.get(operand)
            )
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
        replay_cost=_replay_cost(jaxpr),
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
        return _clean(cands, lo, hi, integral, dt)

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
        return _clean(cands, lo, hi, integral, dt)

    if strategy == "uniform":
        n = 8
        if integral:
            return _clean(
                [rng.randint(int(lo), int(hi)) for _ in range(n)],
                lo, hi, integral, dt,
            )
        return _clean(
            [rng.uniform(lo, hi) for _ in range(n)], lo, hi, integral, dt
        )

    return ()


def _clean(cands, lo, hi, integral, dt=None):
    """De-duplicate, drop what the window cannot hold, keep the order stable.

    ``dt`` is kept as a second, cheap guard: :func:`_window` already
    intersects the declared interval with the dtype's own finite range, so
    an in-window candidate is representable by construction, and this
    catches a future fill generator that stops respecting the window
    before `np.full` turns the mistake into a warning the caller sees.
    """
    out = []
    seen = set()
    for c in cands:
        try:
            v = int(round(c)) if integral else float(c)
        except (TypeError, ValueError, OverflowError):
            continue
        if not (lo <= v <= hi):
            continue
        if dt is not None and not _representable(v, dt):
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

    ``strategies`` never contains :data:`SEED_LABEL`: that is the label
    the seeded phases' own starting points are counted under, so a
    ``tight``-only run's reach is not credited to the margin search when
    its seeds supplied it.  It can appear in a report and in the firing
    message; it cannot be requested.

    Returns a :class:`ProbeReport`.  Raises :class:`VerifiedFalsified` if
    the program violates a discharged obligation at a point inside every
    declaration, admitted by every assume, AND admitted by the fire
    condition -- under ``semantics="real"`` that last one is exact
    rational replay of the same traced program, not the executed float
    alone; see the module docstring for what it costs to get that wrong
    and for why a firing is a raise and not a status.
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

    # TWO STAGES, TWO REASONS.  One `try` around both said "the harness
    # could not be traced" for a read defect on a trace that had worked --
    # a note that sent a reader to their own harness for a defect in this
    # file.
    try:
        closed = jax.make_jaxpr(harness)()
    except Exception as exc:  # noqa: BLE001
        return ProbeReport(declined=f"the harness could not be traced: {exc}")
    try:
        census = _read(closed)
    except Exception as exc:  # noqa: BLE001
        return ProbeReport(
            declined=(
                f"the harness traced, but the probe could not read the "
                f"traced program: {type(exc).__name__}: {exc}"
            )
        )

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
    declined_points = 0
    violations = 0
    adjudged = _Counter()
    best: list = []  # (margin, point) of the least-slack admissible points

    def run_one(strategy, point):
        """Execute one point.  Returns ``(falsification, margin)``.

        ``margin`` is the least slack this point left across the attacked
        obligations, or ``None`` when the point was unusable or the
        obligations do not compare.  It is returned rather than only
        recorded because the ``tight`` search STEERS on it.
        """
        nonlocal built, executed, admissible, declined_points, violations
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
        point_declined = False
        for k in targets:
            if bool(np.all(run.asserts[k])):
                continue
            violations += 1
            detail, why, how = _confirm(
                census, statuses, point, k, semantics
            )
            adjudged.add(how)
            if detail is None:
                skips.add(why)
                point_declined = True
                continue
            shits.add(strategy)
            return (
                Falsification(
                    strategy=strategy,
                    obligation_position=k,
                    values=tuple(repr(np.asarray(a).tolist()) for a in point),
                    margin=run.margins.get(k),
                    detail=detail,
                    adjudication=how,
                ),
                None,
            )
        if point_declined:
            # counted as DECLINED WORK, which is what it is: the probe
            # executed a violation here and would not stand behind it, and
            # a skip rate that ignored that read as "nothing was skipped"
            # on exactly the runs where the interesting result was dropped.
            declined_points += 1
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
            points_declined=declined_points,
            violations_seen=violations,
            adjudications=adjudged.items(),
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
        hit = run_batch(SEED_LABEL, pts)
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


def _step(base, direction, dt):
    """One representable step of ``base`` toward ``direction``, or ``None``.

    ``None`` when the step leaves the format -- which is not an error and
    not a point: past the last finite value of a format there is nothing
    to sample.  Handling it HERE rather than letting ``np.nextafter``
    return an infinity is what keeps ``RuntimeWarning: overflow
    encountered in nextafter`` from escaping the probe; under
    ``-W error::RuntimeWarning`` that warning turned a green VERIFIED on a
    plain ``float16`` declaration of ``(-65504.0, 65504.0)`` into a crash.
    """
    try:
        if dt.kind in "iub":
            moved = np.asarray(base).astype("int64") + (
                -1 if direction < 0 else 1
            )
            info = np.iinfo(dt)
            if not np.all((moved >= int(info.min)) & (moved <= int(info.max))):
                return None
            return moved.astype(dt)
        limit = float(np.finfo(dt).max)
        arr = np.asarray(base)
        target = math.copysign(limit, direction)
        if np.any(np.asarray(arr, dtype="float64") * math.copysign(1.0, direction)
                  >= limit):
            # already at (or past) the format's last finite value in this
            # direction: there is no next float, so there is no point
            return None
        return np.nextafter(arr, np.asarray(target, dtype=dt))
    except (TypeError, ValueError, OverflowError):
        return None


def _ulp_points(census, windows, seeds, budget):
    """The last representable step, in both directions, per declaration."""
    out = []
    for point in seeds:
        for i, (decl, (lo, hi)) in enumerate(zip(census.declarations, windows)):
            dt = np.dtype(decl.dtype)
            base = np.asarray(point[i])
            for direction in (-math.inf, math.inf):
                arr = _step(base, direction, dt)
                if arr is None:
                    continue
                if not np.all((arr >= lo) & (arr <= hi)):
                    continue
                nxt = list(point)
                nxt[i] = arr
                out.append(tuple(nxt))
                if len(out) >= budget:
                    return out
    return out


# --------------------------------------------------------------------------
# the exact-rational replay: the SAME traced program, judged over Q
# --------------------------------------------------------------------------
#
# WHY THIS EXISTS.  Under `semantics="real"` a VERIFIED is a claim about
# the REALS and the probe executes in IEEE floats, so an executed
# violation is not automatically an unsoundness -- the analysis may be
# right about R while the float program lands the other side of a tight
# bound.  The first version of the fire condition tested ULP-STABILITY OF
# THE INPUT as a proxy for that, and its own docstring named it a proxy.
# It is a good proxy for the shape it was built on (a 1-ulp artefact:
# `(x/3)*3`, `(x*x)/x`, `x*0.1*10`, `sqrt(x)**2` all have a longest
# consecutive violating run of one float and are declined) and NO proxy at
# all for COARSE QUANTISATION, where the violating set is thousands of
# ulps wide.  Four lines of ordinary numerical code reached it:
#
#     y = any_array((), "float64", (0.0, 2.0))
#     s = 1e16
#     assert_((s + y) - s <= y)          # the Kahan/Neumaier shape
#
# `1e16` is exactly 10**16 and the spacing of float64 there is 2.0, so
# `(s + y) - s` is 2.0 for every y strictly between 1 and 2 -- a violating
# band about 4.5e15 ulps wide, every point of it stable under a one-ulp
# input perturbation.  Over R the expression is y exactly, so the
# obligation is TRUE, both solvers answered unsat, and the verdict was
# RIGHT.  The probe raised "stelling is UNSOUND at this query" on it.
# Measured on this tree at 123ad75, x64=1, on jax 0.11.0 AND 0.10.2, with
# the same point out of each: `<=` fires at y = 1.6888437030500962 (the
# `uniform` strategy) and `>=` at y = 0.5 (`exact`).  A soundness alarm
# that reports our defect as the caller's is worse than no alarm.
#
# WHAT REPLACES IT.  The same standard `verdict.Witness` already applies
# to a REFUTED: replay the point through exact arithmetic.  Every finite
# IEEE float IS a rational -- `Fraction(float)` is exact and lossless --
# so the traced jaxpr can be re-evaluated over Q at the violating point,
# with `+ - * /` and the rest carrying their REAL meanings.  If the
# obligation is false over Q too, the analysis discharged something false
# about R and the firing stands.  If it is true over Q, the violation was
# manufactured by rounding and must not be reported.
#
# IT IS `fractions` AND NOTHING ELSE.  No analysis module is imported to
# do it, so the independence argument at the top of this file is untouched
# -- which is the reason this is a rational interpreter here rather than a
# call into `stelling.exactness`, which already knows how to do it.
#
# WHAT IT REFUSES TO DO, WHICH IS AS IMPORTANT.  Not every primitive has
# an exact rational reading: `exp`, `log`, `sin` and a fractional `pow`
# are irrational at almost every rational argument, and integer arithmetic
# that WRAPS is not R arithmetic at all.  On any of those this evaluator
# ABSTAINS -- and the fire condition then falls back to the ulp proxy,
# exactly as before, and COUNTS that it did (`ProbeReport.adjudications`),
# because a fire condition that silently degraded to the weaker test on
# the programs it matters most for would be the same defect one layer
# down.


class _Unreplayable(Exception):
    """This program has a step with no exact rational reading at this point."""


# HOW MUCH RATIONAL ARITHMETIC ONE REPLAY MAY DO, and the number is a cost
# decision with a measurement under it rather than a taste.  Measured on
# this tree (`sum(v*v) <= 0.5`, one float64 declaration, jax 0.11.0): the
# replay costs 2.8 - 10.5 microseconds per element visited, ~3.9 at the
# largest size driven (65,536 elements, 253 ms).  So a quarter of a million
# element-visits is on the order of one second, once, and only on a probe
# run that executed a violation.  Past that the replay ABSTAINS before it
# starts and the ulp proxy decides, which is the same degradation an
# irrational primitive produces and is counted the same way -- a fire
# condition that hung a caller's CI to prove a point would be traded for a
# fire condition nobody switches on.
REPLAY_ELEMENT_BUDGET = 250_000


def _replay_cost(jaxpr) -> int:
    """Element-visits one replay of this program will do, near enough.

    The sum of every equation's output size, which is what the
    element-at-a-time evaluators below actually walk.  Counted off the
    AVALS, so it is known before any arithmetic happens and costs nothing
    to ask.
    """
    total = 0
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            shape = getattr(getattr(var, "aval", None), "shape", ())
            n = 1
            for d in shape:
                n *= int(d)
            total += max(1, n)
        for sub in _sub_jaxprs(eqn):
            total += _replay_cost(sub)
    return total


# Call primitives whose body is a nested closed jaxpr to be replayed in
# place.  `jax.numpy` routes almost everything through `pjit`, so a replay
# that abstained here would abstain on most real programs.
_CALL_PRIMITIVES = ("pjit", "closed_call", "remat", "checkpoint")


def _call_jaxpr_of(eqn):
    """The nested closed jaxpr a call primitive carries, whatever it is called."""
    for key in ("jaxpr", "call_jaxpr"):
        sub = eqn.params.get(key)
        if sub is not None and hasattr(sub, "jaxpr") and hasattr(sub, "consts"):
            return sub
    raise _Unreplayable(f"{eqn.primitive.name!r} carries no closed jaxpr")


# The elementwise primitives whose meaning over R is a closed-form
# rational function of rational arguments.  Anything absent abstains, and
# that is the safe direction: abstaining loses a refutation, admitting a
# wrong reading invents one.
_EXACT_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": lambda a, b: a if a >= b else b,
    "min": lambda a, b: a if a <= b else b,
    "atan2": None,  # named to be explicit that it is refused
    "nextafter": None,  # a FLOAT operation; it has no meaning over R
}
_EXACT_UNARY = {
    "neg": lambda a: -a,
    "abs": lambda a: abs(a),
    "sign": lambda a: Fraction(0) if a == 0 else Fraction(1 if a > 0 else -1),
    "floor": lambda a: Fraction(math.floor(a)),
    "ceil": lambda a: Fraction(math.ceil(a)),
    "copy": lambda a: a,
    "stop_gradient": lambda a: a,
    "real": lambda a: a,
}
_COMPARISONS = {
    "le": lambda a, b: a <= b,
    "lt": lambda a, b: a < b,
    "ge": lambda a, b: a >= b,
    "gt": lambda a, b: a > b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}
_BOOLEAN = {
    "and": lambda a, b: bool(a) and bool(b),
    "or": lambda a, b: bool(a) or bool(b),
    "xor": lambda a, b: bool(a) != bool(b),
}

# Pure data movement: every output element is a copy of one input element,
# and WHICH one is a function of the shapes and the params alone.  Those
# are replayed by asking jax itself -- the primitive is bound to an INDEX
# array in place of the data, so the answer to "where did this element
# come from" comes from the same implementation the real run used, rather
# than from a second hand-written copy of jax's shape rules in this file.
# The value is the tuple of operand positions that carry DATA; every other
# operand (a gather's start indices, say) is passed through verbatim
# because its VALUE, not its position, is what the primitive reads.
_MOVEMENT = {
    "broadcast_in_dim": (0,),
    "reshape": (0,),
    "transpose": (0,),
    "squeeze": (0,),
    "expand_dims": (0,),
    "rev": (0,),
    "slice": (0,),
    "concatenate": None,  # every operand is data; filled in at use
    "pad": (0, 1),
}

_REDUCTIONS = {
    "reduce_sum": lambda vs: sum(vs, Fraction(0)),
    "reduce_prod": lambda vs: math.prod(vs, start=Fraction(1)),
    "reduce_max": max,
    "reduce_min": min,
    "reduce_and": lambda vs: all(bool(v) for v in vs),
    "reduce_or": lambda vs: any(bool(v) for v in vs),
}


def _exact(a):
    """Every element of ``a`` as an exact ``Fraction`` (or ``bool``).

    Lossless in both directions that matter: a finite binary float is a
    dyadic rational and ``Fraction(float)`` is its exact value, and a
    narrow float widens to ``float`` exactly.  A non-finite float has no
    rational value at all, and a dtype numpy does not classify as boolean,
    integer or float (``bfloat16`` and the ``float8`` family arrive here as
    ``kind == "V"``) has no exact reading this evaluator can produce.
    """
    arr = np.asarray(a)
    kind = arr.dtype.kind
    flat = arr.reshape(-1).tolist()
    out = np.empty(len(flat), dtype=object)
    for i, v in enumerate(flat):
        if kind == "b":
            out[i] = bool(v)
        elif kind in "iu":
            out[i] = Fraction(int(v))
        elif kind == "f":
            f = float(v)
            if not math.isfinite(f):
                raise _Unreplayable(
                    "a non-finite float has no rational value"
                )
            out[i] = Fraction(f)
        else:
            raise _Unreplayable(f"dtype {arr.dtype!r} has no rational reading")
    return out.reshape(arr.shape)


def _ew(fn, *args):
    """Apply ``fn`` elementwise over broadcast object arrays."""
    b = np.broadcast_arrays(*args)
    flat = [x.reshape(-1) for x in b]
    n = flat[0].size
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = fn(*(f[i] for f in flat))
    return out.reshape(b[0].shape)


def _rat_div(a, b):
    if b == 0:
        raise _Unreplayable("division by zero")
    return a / b


def _rat_pow(a, k):
    """``a ** k`` when the exponent is an integer; abstain otherwise.

    A rational raised to a non-integer power is irrational except on a
    measure-zero set, and the probe has no business guessing which side of
    that it is on.  ``_rat_sqrt`` handles the one exception worth taking.
    """
    if k.denominator != 1 or not (-64 <= k.numerator <= 64):
        raise _Unreplayable("a non-integer (or huge) exponent is not rational")
    n = int(k.numerator)
    if a == 0 and n < 0:
        raise _Unreplayable("zero raised to a negative power")
    return a ** n


def _rat_sqrt(a):
    """Exact only where the argument is a perfect rational square."""
    if a < 0:
        raise _Unreplayable("sqrt of a negative rational")
    rn, rd = math.isqrt(a.numerator), math.isqrt(a.denominator)
    if rn * rn == a.numerator and rd * rd == a.denominator:
        return Fraction(rn, rd)
    raise _Unreplayable("sqrt is irrational at this point")


def _rat_convert(a, src_kind, dst_dtype):
    """``convert_element_type`` over R.

    Float to float is the IDENTITY, of any width, because rounding onto a
    narrower grid is a float operation and R has no such operation -- and
    that reading is exactly what the ``semantics="real"`` claim under
    attack means.  Float to integer TRUNCATES toward zero, which is jax's
    rule and is exact on a rational.  Integer or boolean to float is the
    identity.  The integer target is range-checked by the caller.
    """
    dk = dst_dtype.kind
    if dk == "b":
        return bool(a != 0) if src_kind != "b" else bool(a)
    if src_kind == "b":
        return Fraction(1) if a else Fraction(0)
    if dk in "iu":
        return Fraction(math.trunc(a))
    if dk == "f":
        return a
    raise _Unreplayable(f"conversion to {dst_dtype!r} has no rational reading")


def _movement(prim, params, exact_ins, raw_ins, data_positions):
    """Replay a pure data-movement primitive by asking jax where each element
    came from.

    The operands that carry DATA are replaced by ``int32`` index arrays
    numbering their elements consecutively; every other operand is passed
    through with its real value.  Binding the primitive to those returns,
    for each output position, the index of the source element -- computed
    by jax's own implementation of its own primitive rather than by a
    second copy of its shape rules living here, which is the same posture
    the executor takes toward arithmetic.
    """
    idx_ins = []
    offset = 0
    pool = []
    for j, (ex, raw) in enumerate(zip(exact_ins, raw_ins)):
        if j in data_positions:
            n = int(np.asarray(raw).size)
            idx = np.arange(offset, offset + n, dtype="int32")
            idx_ins.append(idx.reshape(np.asarray(raw).shape))
            pool.extend(np.asarray(ex).reshape(-1).tolist())
            offset += n
        else:
            idx_ins.append(raw)
    out = prim.bind(*idx_ins, **params)
    if prim.multiple_results:
        raise _Unreplayable(f"{prim.name!r} has multiple results")
    mapped = np.asarray(out)
    if mapped.dtype.kind not in "iu":
        raise _Unreplayable(f"{prim.name!r} did not move indices")
    flat = mapped.reshape(-1).tolist()
    picked = np.empty(len(flat), dtype=object)
    for i, src in enumerate(flat):
        if not (0 <= int(src) < offset):
            raise _Unreplayable(f"{prim.name!r} produced an index it invented")
        picked[i] = pool[int(src)]
    return picked.reshape(mapped.shape)


def _reduce(fn, a, axes):
    axes = tuple(int(x) for x in axes)
    if not axes:
        return a
    moved = np.moveaxis(a, axes, tuple(range(-len(axes), 0)))
    kept = moved.shape[: moved.ndim - len(axes)]
    rows = moved.reshape(math.prod(kept) if kept else 1, -1)
    out = np.empty(rows.shape[0], dtype=object)
    for i in range(rows.shape[0]):
        vals = list(rows[i])
        if not vals:
            raise _Unreplayable("a reduction over an empty axis")
        out[i] = fn(vals)
    return out.reshape(kept)


def _int_ok(vals, dt):
    """Did an integer-typed result stay inside its own dtype?

    THIS IS WHAT KEEPS THE INTEGER WRAP VISIBLE.  Exact rational
    arithmetic does not wrap, so replaying a program whose integer
    arithmetic overflowed would report a value the program never computed
    -- and would then declare the executed violation a rounding artefact
    and decline it, suppressing exactly the runtime-wrap catch this probe
    was measured to have.  A result that left its dtype is therefore not
    replayable: the evaluator abstains and the ulp proxy decides, which is
    what it did before this replay existed.
    """
    info = np.iinfo(dt)
    for v in vals:
        if v.denominator != 1 or not (int(info.min) <= v <= int(info.max)):
            raise _Unreplayable(
                "integer arithmetic left its dtype's range: the program "
                "wraps there and rational arithmetic does not"
            )


def _replay(census, point):
    """Re-evaluate the traced program at ``point`` in exact rational arithmetic.

    Returns ``(assumes, asserts)`` as lists of Python bools, or raises
    :class:`_Unreplayable`.  The point is the same array tuple the float
    execution used, so the two runs differ in exactly one thing: the
    arithmetic.
    """
    cost = census.replay_cost
    if cost > REPLAY_ELEMENT_BUDGET:
        raise _Unreplayable(
            f"the replay would visit about {cost} elements and the budget "
            f"is {REPLAY_ELEMENT_BUDGET}"
        )
    assumes: list = []
    asserts: list = []

    def run(jaxpr, consts, args, decl):
        env: dict = {}

        def read(atom):
            if isinstance(atom, jex_core.Literal):
                return _exact(atom.val)
            return env[atom]

        for v, c in zip(jaxpr.constvars, consts):
            env[v] = _exact(c)
        for v, a in zip(jaxpr.invars, args):
            env[v] = a

        for eqn in jaxpr.eqns:
            prim = eqn.primitive
            name = prim.name
            if name == "stelling_any":
                env[eqn.outvars[0]] = _exact(point[decl[0]])
                decl[0] += 1
                continue
            ins = [read(a) for a in eqn.invars]
            raw = [
                a.val if isinstance(a, jex_core.Literal) else None
                for a in eqn.invars
            ]
            if name in _CALL_PRIMITIVES:
                # `jnp` puts most of its work behind `pjit`, so a replay
                # that abstained on it would abstain on most real programs.
                sub = _call_jaxpr_of(eqn)
                inner = run(sub.jaxpr, sub.consts, ins, decl)
                for var, o in zip(eqn.outvars, inner):
                    env[var] = o
                continue
            if name in ("stelling_assume", "stelling_nonvacuity"):
                if name == "stelling_assume":
                    assumes.append(all(bool(v) for v in ins[0].reshape(-1)))
                env[eqn.outvars[0]] = ins[0]
                continue
            if name == "stelling_assert":
                asserts.append(all(bool(v) for v in ins[0].reshape(-1)))
                env[eqn.outvars[0]] = ins[0]
                continue
            out = _apply(eqn, prim, name, ins, raw)
            env[eqn.outvars[0]] = out
        return [read(a) for a in jaxpr.outvars]

    def _apply(eqn, prim, name, ins, raw):
        params = dict(eqn.params)
        aval = eqn.outvars[0].aval
        out_dt = np.dtype(aval.dtype)

        if name in _EXACT_UNARY and _EXACT_UNARY[name] is not None:
            out = _ew(_EXACT_UNARY[name], ins[0])
        elif name in _EXACT_BINARY and _EXACT_BINARY[name] is not None:
            out = _ew(_EXACT_BINARY[name], *ins)
        elif name in _COMPARISONS:
            return _ew(_COMPARISONS[name], *ins)
        elif name in _BOOLEAN:
            return _ew(_BOOLEAN[name], *ins)
        elif name == "not":
            return _ew(lambda a: not bool(a), ins[0])
        elif name == "div":
            if out_dt.kind != "f":
                raise _Unreplayable(
                    "integer division truncates; that is not division over Q"
                )
            out = _ew(_rat_div, *ins)
        elif name == "integer_pow":
            y = params.get("y")
            if not isinstance(y, int):
                raise _Unreplayable("integer_pow with a non-integer exponent")
            out = _ew(lambda a, k=Fraction(y): _rat_pow(a, k), ins[0])
        elif name == "pow":
            out = _ew(_rat_pow, *ins)
        elif name == "sqrt":
            out = _ew(_rat_sqrt, ins[0])
        elif name == "square":
            out = _ew(lambda a: a * a, ins[0])
        elif name == "select_n":
            which = ins[0]
            cases = ins[1:]
            out = _ew(
                lambda w, *cs: cs[int(w)], which, *np.broadcast_arrays(*cases)
            )
        elif name == "convert_element_type":
            src_kind = np.dtype(eqn.invars[0].aval.dtype).kind
            out = _ew(
                lambda a: _rat_convert(a, src_kind, out_dt), ins[0]
            )
        elif name in _REDUCTIONS:
            out = _reduce(_REDUCTIONS[name], ins[0], params.get("axes", ()))
        elif name in _MOVEMENT:
            positions = _MOVEMENT[name]
            if positions is None:
                positions = tuple(range(len(ins)))
            # a non-literal operand contributes only its SHAPE here; the
            # values travel in `ins` and are picked out by the index map
            raw_ins = [
                r if r is not None else np.zeros(tuple(a.aval.shape))
                for r, a in zip(raw, eqn.invars)
            ]
            out = _movement(prim, params, ins, raw_ins, positions)
        else:
            raise _Unreplayable(f"{name!r} has no exact rational reading")

        if out_dt.kind in "iu":
            _int_ok(out.reshape(-1), out_dt)
        return out

    run(census.closed.jaxpr, census.closed.consts, (), [0])
    return assumes, asserts


def _confirm(census, statuses, point, k, semantics):
    """Decide whether an executed violation may be REPORTED, or is declined.

    Returns ``(detail, decline_reason, adjudication)``: exactly one of the
    first two is ``None``, and ``adjudication`` always names WHICH test
    decided, because the strength of a firing is the strength of the test
    that let it through.

    Under ``ieee`` the executed float IS the subject of the claim and the
    violation stands as it is.  Under ``real`` the claim is about ℝ and
    the float run is only evidence about floats, so the point is re-judged:

    * **all-integral declarations** short-circuit, exactly as before.  The
      arithmetic is exact, and this branch MUST NOT become a rational
      replay -- rational arithmetic does not wrap, so replaying an
      ``int8`` program would report values it never computed and would
      declare a genuine runtime wrap a rounding artefact.  Measured: with
      ``propagate._int_guarded`` removed the probe catches an ``int8``
      runtime wrap, and it catches it through this branch.
    * **otherwise, EXACT-RATIONAL REPLAY of the same traced jaxpr** at the
      same point (:func:`_replay`).  False over ℚ as well as in floats:
      the analysis discharged something false about ℝ and the firing
      stands.  True over ℚ: the violation was manufactured by rounding and
      is declined.
    * **and where the replay abstains** -- an irrational step, an integer
      result that left its dtype, a primitive with no rational reading --
      the ulp-stability PROXY below decides, as it did for everything
      before this replay existed.  It is named as a proxy because it is
      one: stability of the INPUT is not a proof about rounding in the
      COMPUTATION, and it is blind to coarse quantisation (see the
      commentary above :class:`_Unreplayable`).  Every fall-back is
      counted, so a report says how much of its firing rests on it.
    """
    detail = (
        f"the obligation evaluated FALSE at this point; the declared box, "
        f"every assume, and the obligation itself were all evaluated by "
        f"executing the program"
    )
    if semantics == "ieee":
        return detail, None, "ieee-executed-float"
    integral = all(
        np.dtype(d.dtype).kind in "iub" for d in census.declarations
    )
    if integral:
        return (
            detail + " (exact integer arithmetic: no rounding involved)",
            None,
            "exact-integer-arithmetic",
        )

    why = None
    try:
        assumes, asserts = _replay(census, point)
    except _Unreplayable as exc:
        why = str(exc)
    except Exception as exc:  # noqa: BLE001 - an abstention, never a firing
        why = f"{type(exc).__name__}: {exc}"
    else:
        if len(asserts) != len(statuses):
            why = "the rational replay saw a different number of obligations"
        elif assumes and not all(assumes):
            # The point satisfies the assume in floats but not over ℚ, so
            # over ℝ it is outside the assumed region and a violation there
            # refutes nothing.  Declining is the conservative side.
            return (
                None,
                "assume-unsatisfied-over-the-rationals",
                "exact-replay-outside-the-assumed-region",
            )
        elif asserts[k]:
            return (
                None,
                "float-rounding-artefact",
                "exact-replay-holds-over-the-rationals",
            )
        else:
            return (
                detail
                + " (and exact-rational replay of the same traced jaxpr at "
                "the same point makes it FALSE over ℚ, so this is not a "
                "rounding artefact)",
                None,
                "exact-replay-refutes-over-the-rationals",
            )

    # ulp-stability, as a PROXY and named as one
    neighbours: list = []
    for i, decl in enumerate(census.declarations):
        dt = np.dtype(decl.dtype)
        if dt.kind != "f":
            continue
        base = np.asarray(point[i])
        for direction in (-math.inf, math.inf):
            arr = _step(base, direction, dt)
            if arr is None or not _admissible(decl, arr):
                continue
            nxt = list(point)
            nxt[i] = arr
            neighbours.append(tuple(nxt))
    if not neighbours:
        return (
            detail
            + f" (no ulp-neighbour inside the box to test against; the "
            f"exact-rational replay abstained: {why})",
            None,
            "ulp-proxy-refutes-with-no-neighbour",
        )
    for nb in neighbours:
        run = _execute(census, [jax.numpy.asarray(a) for a in nb])
        if run.raised is not None or len(run.asserts) != len(statuses):
            return None, "precision-ambiguous", "ulp-proxy-ambiguous"
        if run.assumes and not all(bool(np.all(a)) for a in run.assumes):
            continue
        if bool(np.all(run.asserts[k])):
            return None, "precision-ambiguous", "ulp-proxy-ambiguous"
    return (
        detail
        + f" (and at every ulp-neighbour of it inside the box; the "
        f"exact-rational replay abstained: {why})",
        None,
        "ulp-proxy-refutes",
    )
