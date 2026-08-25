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
STABILITY: SHIPPED, DEFAULT-OFF, AND PROVISIONAL — AND THE WORD IT
REPLACES WAS ``UNAUDITED``
--------------------------------------------------------------------------

``falsify="sample"`` ships in 0.2.0. It is **default-off** — with
``falsify=None`` this module is never imported and every path is
byte-identical — and it is **provisional**: what it does when it fires is
settled, its SIGNATURE is not. ``probe()``'s first parameter changed name
and type inside this release cycle, and it is in ``__all__``.

**``UNAUDITED`` IS RETIRED, AND ON EVIDENCE THAT IS IN THIS TREE.** The
fire condition's exact reading is checked, on every run, against an
independent ``Fraction`` oracle that shares no code with this module:
``tests/test_probe_oracle.py``. Over ten fixtures and a 27-point grid
apiece it drives **297 gate readings and 270 obligation readings, and
asserts that all 270 point-comparisons agree** — and it asserts those
counts, so a figure that drifts fails there instead of going stale here.
Two of the ten fixtures state an assume whose float answer and whose ℚ
answer differ at **20 and 12 of the 27 points** — a separate test asserts
both counts, because an oracle that agrees on every route is measuring
nothing. (That sentence used to read *"over most of the grid"*, which is
true of the first fixture and FALSE of the second; the counts are quoted
because a count cannot round in the flattering direction.) The three
defects the blind audit behind this word was pointed at are fixed and
driven, each at its own site below.

**AND AN AGREEMENT COUNT IS NOT A COVERAGE FIGURE.** Measured by breaking
this module in 32 single-edit ways and re-running that file, **the oracle
catches 10 of the 32** — three of them added by two fixtures on this
branch, which is the whole delta from the 7 the eight-fixture version
caught. Three of the misses cannot be closed there at all: the oracle
calls :func:`_replay` directly, so :func:`_window` and :func:`_admissible`
are outside its reach and :func:`_body_runs_once` has no fixture to bite
on. The rest are named fixture gaps — no ``_REDUCTIONS``, no
``_MOVEMENT``, no boolean, no strict comparison, ``jit`` but never
``remat2``, and ``jax_enable_x64`` forced on. **The list is in that
file's own docstring**, by name and not by count, so that nothing here
reads as coverage it does not have.

**THE FIGURES THIS PARAGRAPH USED TO CARRY WERE NOT IN THE TREE, AND THAT
IS THE SAME DEFECT AS A MISSING CHECK.** It read *"363 gate readings, 363
agreements, over 266 driven base-versus-fix comparisons"*, and those three
numbers appeared in exactly one place in this repository — the sentence
that cited them, here and in ``preconditions.check``. No test, no
CHANGELOG entry, no doc. This module's whole subject is what a reader may
believe about a verdict, and a number nobody can re-derive is prose
asserting a check that does not exist. The audit that produced them ran
against two trees at once and cannot be a standing test of one; what could
be, and now is, is its ORACLE. The base-versus-fix comparisons that DO
re-derive in a single tree are the corpus re-runs recorded below, which
name the corpus, the files and the counts.

**WHAT "PROVISIONAL" NAMES IS A LIST, NOT A MOOD.** Every item is
disclosed in full at its own site; they are gathered here so that a reader
deciding whether to switch this on does not have to find them:

1. **THE ANALYSIS'S REGION IS NEVER READ.** The assume gate reads the
   program's assume TEXT; what the analysis discharged over is a
   CONSTRAINED BOX this module may not look at. Nothing checks that the
   second contains the first. It is the mirror of every defect this module
   has fixed and it is **open** — see the section of that name below,
   which also says why it is not fixable inside this module's import rule.
2. **REACH ON ORDINARY ``jnp`` CODE IS ROUGHLY HALF**, measured: 17 of 31
   one-line programs fire, all 17 on an exact refutation. ``dot_general``,
   ``sort``, ``cumsum``, ``stack``, ``rem``, ``scatter`` and
   ``scatter-add`` all HAVE exact rational readings and are simply not in
   this file's tables. That is a work list, not a limit of the approach.
3. **``semantics="ieee"`` IS THE WEAKEST SETTING AND KEEPS THE LOUDEST
   CAVEAT.** It is the one path on which the EXECUTED FLOAT admits a
   firing with no exact test behind it, guarded only by a second
   compilation route (:func:`_whole_program_route`); and an ``ieee``
   verdict is the one that can rest on a caller's ``libm_budget``, which
   stelling stamps as declared and unverified. :func:`_fire` splits the
   message for that, and **cannot** tell you whether the declaration bears
   on the obligation that fired — it says so in the message and prints
   what the obligation is computed from instead of guessing.
4. **THE ASSUME CONFIRMATION'S BUDGET SATURATES ON ORDINARY SHAPES.** A
   ``(60000,)`` float64 declaration with one assume leaves 90 of 91
   admissible points unconfirmed on one machine; the table is in
   :func:`probe` and **its admissible and unconfirmed columns are
   machine-dependent**, because an unread gate counts its point
   admissible. That costs evidence, never soundness, and the count is
   reported.
5. **THE CLOCK IS PART OF THE FIRE CONDITION.** ``REPLAY_SECONDS_BUDGET``
   is a wall-clock backstop, so whether this instrument fires on a given
   program can depend on the machine. It can only ABSTAIN, so a slow
   machine buys fewer firings and never wrong ones; the reasoning is at
   that constant.
6. **NEITHER WALKER ENTERS A LOOP OR BRANCH BODY**, so an assume or an
   obligation inside a ``scan``, ``while`` or ``cond`` is not read. The
   probe does not report one anyway: :func:`_execute`'s reading is
   checked against the every-depth census and declines
   (``assume-not-fully-executed``), and :func:`_replay` abstains on the
   primitive. That is a DECLINE STANDING IN FOR A GUARD, and descending
   those bodies is a plausible reach improvement, so the descent REFUSES
   a body that does not run exactly once per equation
   (:func:`_body_runs_once`) — by a name, an iteration count, and a
   signature, in that order. The version of this list that said *"rather
   than trusting a list of names to be complete"* had dropped the name
   check for the derivation alone, and that lost ``lax.fori_loop``: a
   static-bound loop traces to a ``scan`` whose body has the equation's
   own signature, so no structural fact fires. Both lists are back and
   the commentary at :func:`_body_runs_once` says which shape each of the
   three opinions covers and which none of them can. **It is guarded, not
   closed**, and closing it is reach work this release did not do.

Items 3 and 6 are the same shape — *a decline standing in for a guard* —
and neither is closed by this release. **ITEM 1 IS NOT ONE OF THEM**, and
this sentence used to say it was, which is an overclaim in the reassuring
direction inside a list of caveats. There is no decline there: the section
of item 1's own name below says the module *"does neither correctly nor
incorrectly: it does not do it at all"*. It is the MIRROR of the class
rather than an instance of it, it is the only one of the six that cannot
be closed inside this module's import rule at all, and nothing stands in
for the guard it does not have.

Nothing in that list is a reason to distrust a FIRING: only an exact test
may admit one, and the message names which test did. They are reasons a
run that found nothing says nothing, which is the first sentence of the
next section.

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
behind, including the ones where the violation is an artefact of how much
of the program this probe handed to jax at once. Every one of those is
counted by reason in :attr:`ProbeReport.skips`, adjudicated in
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
probe was handed **461 VERIFIED verdicts**. It declined **8** outright —
unbounded or otherwise unsampleable declarations — and on the rest it
built 34,697 points, executed 34,693, and found 28,262 of them inside
every declaration and admitted by every assume. It fired **92 times**,
across five files:

======================================  =======  ==============================
file                                    firings  what it is
======================================  =======  ==============================
``tests/test_falsify_probe.py``              35  the probe's own fixtures,
                                                 false by hand
``tests/test_square_row_gauge_jax.py``       24  a mutation battery
``tests/test_pow_row_gauge_jax.py``          21  a mutation battery
``tests/test_falsify_fire_condition.py``     10  the probe's own fixtures,
                                                 false by hand
``tests/test_falsify_independence.py``        2  the lying propagator
======================================  =======  ==============================

**Every firing is inside a deliberate mutation or a by-construction-false
fixture, and every one of the 92 was admitted by an EXACT test** — 90 by
exact-rational replay, one by exact integer arithmetic, one by the
executed float under ``ieee`` semantics, and none at all by anything
weaker. The attribution is spelled out rather than summarised because an
earlier version of this paragraph said *"every firing was inside the
``pow`` row's own gauge battery"* and reported 77 firings in four figures
that no longer re-derive. Five files fire, not one. Driven WITHOUT the
mutation, every one of those gauge fixtures returns REFUTED or UNKNOWN and
the probe never fires.

**RE-RUN AFTER THE TOTALITY GUARDS, ON THE SAME CORPUS: STILL 92.** The
batch that added the guards below (see THE CLASS) re-ran exactly this
measurement on both trees. Over the nodes both corpora share, **base 92
firings and fixed 92** — no firing is lost. Six probe calls change and
none of them fired in either tree: three now decline under
``assume-not-fully-executed`` (``test_undescended_assume.py`` ×2 and
``test_vacuous_precondition.py``, all three programs whose assume is
inside a ``scan``/``jit`` body, which is the false-alarm shape itself);
two replace 21 per-point ``program-raised`` skips apiece with one accurate
whole-probe decline (``test_vacuity_depth.py``, a ``stelling_any`` the
probe cannot vary); and one is the ``points_admissible`` correction. On 31
ordinary one-line ``jnp`` programs the reports are IDENTICAL field for
field, 18 firing in each tree.

**AND RE-RUN AGAIN AFTER THE PROBE STOPPED TRACING FOR ITSELF, WITH THE
POINT TOTALS ABOVE MARKED AS WHAT THEY ARE.** The batch that removed the
probe's own trace (see :func:`probe`), split the firing message on a
declared assumption (:func:`_fire`) and made the admissible count carry
exact evidence re-ran the forced-probe measurement over the three GAUGE
files — the only files in that corpus this project has twice used for
this comparison, because they are the ones no falsify batch edits. **The
two reports are IDENTICAL field for field**: 150 probe calls, 45 firings
(``square`` 24, ``pow`` 21, ``scatter`` 0), 6,446 points built, executed
and admissible, 791 declined violations, 836 violations seen, and the
same two adjudication counts. Nothing was gained and nothing was lost on
programs that state no assume, which those are.

**The 34,697 / 34,693 / 28,262 point totals in the paragraph above are
from BEFORE that batch and two of them no longer describe this tree**,
which is said here rather than left to be discovered: the admissible
count now excludes points an exact ℚ re-reading of the assumes rejects
(on one measured fixture 55 became 8, and 47 of those 55 were never in
the assumed region), and the built count drops on ``bool`` declarations
whose window used to ignore the declaration. The firing count and the
five-file table are unaffected — a point outside the assumed region over
ℚ could never fire, because ``_confirm`` already declined it.

**AND 169 BECAME 92, WHICH IS THE PRICE OF THE FIRE CONDITION AND IS
STATED IN THE ONE FORM THAT IS NOT MISLEADING.** The figures 169 and 92
are from DIFFERENT corpora — this batch rewrote the two ``falsify`` test
files themselves, so their own counts moved for reasons that have nothing
to do with the fire condition. The comparison that means something is over
the three GAUGE files, which this batch did not touch at all:

===============================  =====  =====  ==============================
mutation battery                  base  fixed  admitted by
===============================  =====  =====  ==============================
``test_square_row_gauge_jax``       24     24  base 16 exact + 8 proxy;
                                               fixed 24 exact
``test_pow_row_gauge_jax``          48     21  base 17 exact + 31 proxy;
                                               fixed 21 exact
``test_scatter_gauge_jax``           3      0  base 3 proxy; fixed none
===============================  =====  =====  ==============================

So on an identical corpus: **exact firings went 33 to 45** — naming the
call primitive (see ``_CALL_PRIMITIVES``) recovered 12 that the replay had
been abstaining on — **proxy firings went 42 to 0**, and 30 firings were
lost outright. Every one of the 30 was previously admitted by the
ulp-stability proxy, i.e. by a test that this repository has separately
measured producing a false alarm on a correct VERIFIED. The ``scatter``
row lost all three of its, and the reason is a missing table entry rather
than a limit of the approach — see the fire-condition section.

**ON ORDINARY ``jnp`` CODE, MEASURED SEPARATELY**, because a corpus of
this project's own fixtures is not a sample of what users write. 31
one-line programs of the kind that appear in real code — ``jnp.sum``,
``mean``, ``dot``, ``@``, ``linalg.norm``, ``where``, ``maximum``,
``clip``, ``mse``, ``softmax``, ``logsumexp``, ``tanh``, ``sqrt``,
``exp``, ``log1p``, ``power``, ``/``, ``cumsum``, ``sort``,
``reshape``/``transpose``, ``concatenate``, ``stack``, ``at[].set``,
``segment_sum``, a lerp, a running variance — each carrying an obligation
that is FALSE at a declared point:

* **base:** 31 of 31 fired, **17 of them adjudicated by the ulp proxy.**
* **fixed:** **17 of 31 fire, all 17 on an exact refutation**; the other
  14 decline with the primitive named (``dot_general`` ×3, ``exp``,
  ``log1p``, ``tanh``, ``sort``, ``cumsum``, ``stack``, ``scatter``,
  ``scatter-add``, a fractional ``pow``, and two non-finite intermediates
  inside ``softmax``/``logsumexp``).

Reach on ordinary code did not collapse; it roughly halved, and the half
that survives is the half that was ever a proof. The 14 are itemised
because they are the work list, not the verdict: ``dot_general``,
``sort``, ``cumsum``, ``stack``, ``rem``, ``scatter`` and ``scatter-add``
all HAVE exact rational readings and are simply not in this file's
tables. That is THE list, spelled the same way in the three places this
module carries it — here, under WHAT DECIDES NOW below, and in the
CHANGELOG entry for this feature. It used to be three different lists,
each missing something another had: ``stack`` was in one of them, and
``stack`` is a live measured abstention (``'stack' has no exact rational
reading``, on both supported jax series, through ``jnp.stack``, which
lowers to a ``stack`` PRIMITIVE rather than to ``concatenate``). ``rem``
is the one name here this corpus run happened not to reach; it abstains
under the same message when a program contains it.

So on this corpus the false-alarm count is zero and the instrument
demonstrably fires — which is the pair of facts a default-off flag has to
establish before anyone argues about releasing it, and neither of them is
an argument that a VERIFIED it did not break is any better for it. **Zero
false alarms ON THIS CORPUS is also not zero false alarms**: the corpus
had none of the shape that reached one, and the fire condition this module
shipped with raised "stelling is UNSOUND" on a correct VERIFIED that four
lines of ordinary compensated summation produce — and the version that was
supposed to have fixed that still raised it, one ``jnp.where`` away. What
that cost, and what replaced the test that caused it, is the fire-condition
section below.

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
``stelling.vacuity``, ``stelling.verdict``, ``stelling.ir``.

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
The probe and the analysis share **jax's tracer, the declaration API, and
— since this batch — ONE TRACE**. They read a box that ``any_array`` —
not either of them — decided to record, out of the same
``jax.make_jaxpr(harness)`` result.

**AND THE SHARED TRACE IS A CORRECTION AND A STRENGTHENING, NOT A
CONCESSION.** They used to trace SEPARATELY: ``_pipeline`` traced, and
``probe`` called ``jax.make_jaxpr(harness)()`` again for itself. That was
never independence — the second trace is the same tracer on the same
callable — and it introduced a failure mode a shared trace does not have.
Whether the second call re-ran the harness body was decided by jax's own
trace memo; the overflow tripwire's gate DEFEATS that memo on purpose
(``jax.clear_caches()`` and a fresh closure, so the trace happens under
the instrument), so an impure harness gave the probe a genuinely
DIFFERENT PROGRAM from the one the verdict is about, with nothing
comparing the two. Measured: harness-body invocations per ``check()``
were 1 everywhere and **2 with the tripwire armed and
``falsify="sample"``**; driven both directions through the public API. The
probe is therefore handed the analysis's jaxpr and cannot trace at all —
see :func:`probe`. **What that does NOT change is the rule that matters:
the probe still never reads the transcribed
:class:`stelling.ir.ClosedJaxpr`.** The rule is *"do not consume the
transcription, because the transcription is part of what is under test"*,
not *"trace it again"*, and jax's object is exactly what the probe's own
``make_jaxpr`` call produced.

The second half of that sentence is a correction: this paragraph used to
say "exactly one step: jax's tracer". ``any_array`` validates a
declaration through ``propagate._INT_DTYPE_BOUNDS``, so the propagator has
a vote on which declarations exist at all. Measured: with that table
replaced by ``{"int8": (-500, 500)}`` the declaration ``any_array((),
"int8", (200.0, 300.0))`` stops being refused and the probe READS a box
that on the clean tree never reaches it — and then declines it, 0 points
built, ``declaration #0 is not sampleable: empty-integer-box``, because
``_window`` still intersects with ``np.iinfo(int8)``. (This sentence said
"reads and samples" until it was driven. The vote the propagator has is
over which declarations EXIST, not over which points get sampled, and the
sampled-points half is the one this module could have been blamed for.)
Everything
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

**REJECTED — return REFUTED with the probe's witness.** This has a good
evidence story: the witness came from EXECUTING the real program, which
is the one leg :class:`stelling.verdict.Witness` calls independent of the
plan, and a REFUTED gets that leg as an emitted reproducer
(:mod:`stelling.reproduce`) for the caller to run rather than in-process.

**IT IS NOT A UNIFORMLY STRONGER STANDARD, AND THIS PARAGRAPH USED TO SAY
IT WAS.** It read *"a stronger standard than a REFUTED's ordinary witness
(a solver model, replayed)"*, and that is wrong twice. It understates the
REFUTED: a REFUTED's witness is a solver model that exact-rational replay
CONFIRMED before it could become a :class:`~stelling.verdict.Witness` at
all, and which ``stelling.reproduce`` then emits as a runnable file that
executes it through the real program. And it overstates this probe: of
the three tests that may admit a firing (see THE FIRE CONDITION below),
only ONE replays through ``Fraction``. The other two — ``ieee`` semantics
and an all-integral program — return before :func:`_replay` is ever
called. Why each of those is nonetheless exact is written out under the
fire condition; what is not true is that all three meet the REFUTED's
standard, and a comparison a reader cannot check is worse than no
comparison.

It has the worst accounting story, and the accounting is what matters.
It converts a soundness event *in the tool*
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
configuration — executions, then fixtures reached out of three:
``endpoints`` 12/1, ``ulp`` 18/1, ``uniform`` 24/0, ``exact`` 33/2,
``tight`` 41/3, all five together 47/3. So ``endpoints`` is the cheapest
configuration there is and ``exact`` reaches the ``pow`` perfect-power
shape in less than half of ``tight``'s executions. Both of those are now
pinned by a test; the ``endpoints`` half was claimed and unmeasured.
(The live corpus was SIX fixtures and these figures were 20/4, 26/4, 28/3,
41/5, 49/6 and 55/6. The three that left it are ``scatter`` fixtures the
fire condition now declines for want of an exact reading — see the fire
condition section; the ORDER of the cost column is what these two tests
are about and it is unchanged.)

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

**WHAT DECIDES NOW.** In order — and all of it downstream of the point
being ADMITTED BY EVERY ASSUME, which is a reading of the program that can
itself be partial and is guarded before any of this is consulted (see THE
CLASS below):

* **``ieee`` semantics: nothing decides.** The executed float IS the
  subject of the claim and the violation stands as it is — once WHICH
  float was executed has been settled, which is the fifth defect and is
  not settled by writing the walk one way rather than the other.
  ``_execute`` binds one equation at a time, so it computes the float the
  TRACE's granularity produces; ``jax.make_jaxpr`` inlines the ``jit``
  inside ``jnp.mean``, so that differs from what the caller's own call
  computes on programs that say ``jit`` nowhere. The same program is
  therefore also run at the same point as ONE compiled region
  (:func:`_whole_program_route`), and a violation whose truth value moves
  between the two routes declines under
  ``executed-float-depends-on-granularity``. That second route is
  consulted only after a violation and can only ever decline.
* **the PROGRAM integral throughout: exact integer arithmetic decides.**
  No rounding is involved and the violation stands. This branch MUST NOT
  become a rational replay — see the blind-spot note above: ℚ does not
  wrap, and replaying an ``int8`` program in ℚ would report values it
  never computed and would then declare a genuine runtime wrap an
  artefact.

  **THE WORD "PROGRAM" IS THIS BATCH'S REPAIR AND IT IS THE THIRD TIME
  THIS MODULE MADE ONE MISTAKE.** The predicate read the DECLARATIONS —
  a fact about the query's signature, standing in for a claim about its
  arithmetic — so an ``int16`` declaration cast to ``float32`` satisfied
  it, and four lines through the public API turned a correct ``VERIFIED``
  into *"stelling is UNSOUND at this query"* on both supported jax
  series, under a parenthetical reading *"exact integer arithmetic: no
  rounding involved"*::

      x = any_array((), "int16", (0, 3))
      y = x.astype("float32")        # int16 -> float32 is whitelisted
      b = jnp.float32(2**24)         # ulp(2**24) is 2 in float32
      assert_((b + y) - b <= 3.0)    # over ℝ this is ``y <= 3``: TRUE

  First ulp-stability of the input stood in for *"not a rounding
  artefact"*; then a fall-back to that same proxy stood in for an exact
  adjudication; then this. There have since been a fourth and a fifth, and
  the rule they share: **a predicate that licenses an exactness claim is
  computed from the object the claim is about, from all of it, and at the
  object's own granularity.** Here that is every operand and every result
  dtype in the jaxpr, at every depth, ``and``-ed with a declaration test
  that is redundant today and kept as the conjunct a reader checks by
  hand. See :func:`_integral_program` and the comment in
  :func:`_confirm`.
* **otherwise: EXACT-RATIONAL REPLAY of the same traced jaxpr at the same
  point** (:func:`_replay`). Every finite float is a rational and
  ``Fraction(float)`` is exact, so the program can be re-evaluated over ℚ
  with each primitive carrying its REAL meaning. False over ℚ as well as
  in floats — the analysis discharged something false about ℝ, and the
  firing stands. True over ℚ — the violation was manufactured by rounding
  and is declined, counted under ``float-rounding-artefact``. This is the
  same exact-rational standard :class:`stelling.verdict.Witness` applies
  before a solver model may become a witness at all, and it is
  ``fractions`` and nothing else: no analysis module is imported to do it,
  which is why it is an interpreter here rather than a call into
  ``stelling.exactness``. (It is the standard the THIRD bullet meets. The
  other two do not replay through ``Fraction`` at all — see immediately
  below.)

**TWO OF THE THREE DO NOT REPLAY THROUGH ℚ, AND HERE IS WHY EACH IS
NONETHELESS EXACT.** Said here because this file spent three revisions
claiming *"the same standard"* for the whole fire condition, which is
true of one bullet out of three; in this project prose asserting a check
that does not exist is the same defect as the missing check.

* ``ieee-executed-float`` **replays nothing because ℚ is the wrong
  arithmetic for it.** An ``ieee`` VERIFIED is a claim about the FLOAT
  the program computes, so the executed float is not evidence ABOUT the
  subject of the claim, it IS the subject. Replaying it over ℚ would
  answer a different question, and the answer would be no evidence either
  way about an ``ieee`` discharge. What that branch does need — and now
  has — is that the float it executed is the PROGRAM's float and not the
  trace-granularity's, which is the fifth instance of the class and is
  checked by :func:`_whole_program_route`, not by ℚ.
* ``exact-integer-arithmetic`` **replays nothing because the executed
  values are already exact.** Every operand and every result dtype in the
  program, at every depth, is integral (:func:`_integral_program`), so
  the machine did integer arithmetic and no rounding entered. Replaying
  it over ℚ would be actively WRONG: ℚ does not wrap, so a genuine
  ``int8`` runtime wrap would be re-evaluated as the unwrapped value and
  the firing declined as a "rounding artefact" it is not. This is the one
  branch whose correctness would be destroyed by meeting the REFUTED's
  literal standard.
* **AND WHAT THE TWO OF THEM SHARE, AS A LIMIT.** Neither re-reads the
  ASSUMES over ℚ for itself — but the admissibility gate in
  :func:`probe` now does, at every admissible point, under ``real``
  semantics (which is the semantics ``exact-integer-arithmetic`` is
  reachable under). Under ``ieee`` the assumes are read in the arithmetic
  the verdict is about, which is the executed float, and the stamp line
  says so. What remains uncovered in both is one level up and is not an
  arithmetic question at all: see **THE ANALYSIS'S REGION IS NEVER READ**
  under THE CLASS.
* **where the replay ABSTAINS, NOTHING decides and the violation is
  DECLINED.** ``exp``, ``log``, ``sin`` and a fractional ``pow`` are
  irrational at almost every rational argument; an integer intermediate
  that left its dtype wrapped and ℚ does not; a value wider than the
  replay's budget is refused before it is paid for; a primitive with no
  rational reading is refused rather than guessed. On any of those the
  probe declines under ``no-exact-reading-of-this-program``, counts the
  decline in the skip rate, and records WHY the exact reading was
  unavailable in :attr:`ProbeReport.abstentions` and in the stamp line.

**THE PROXY WAS THE FALLBACK FOR ONE BATCH AND THE FALSE ALARM SURVIVED
IT.** The first repair for the Kahan shape put the exact replay in FRONT
of the ulp proxy and left the proxy behind it. That is not a repair: it
narrows the false alarm to the programs the replay happens not to read,
and there are a lot of those. Measured at ``99abdb0``, through the public
API, with no mutation and in four lines::

    y = any_array((), "float64", (0.0, 2.0)); s = 1e16
    z = jnp.where(y >= 0.0, (s + y) - s, y)
    assert_(z <= y)          # true over ℝ; the VERIFIED is correct
    -> FALSIFICATION PROBE FIRED — stelling is UNSOUND at this query
       ... admitted by the 'ulp-proxy-refutes' test ...

and again with ``y + (y ** 0.5) * 0.0`` on the right-hand side, and it
would have again through ``exp``, ``sort``, ``cumsum``, ``stack``,
``rem``, a non-square ``sqrt``, ``scatter`` or ``dot_general`` — every
matmul.
**Every primitive the replay abstained on was a route back to the test the
replay had been added to replace**, and over this repository's own corpus
89 of 169 firings (53%) were still admitted by the proxy, 34 of them by
``ulp-proxy-refutes-with-no-neighbour``, which applies no rounding test at
all.

**SO ONLY AN EXACT TEST MAY ADMIT, AND THE PROXY IS GONE FROM THE FIRING
PATH ENTIRELY.** For an instrument whose message is the categorical
*"stelling is UNSOUND at this query"* and whose disposition is to RAISE,
the safe direction for an abstention is to decline. It is not kept as a
decline filter either: an abstention already declines, so the proxy could
only ever have turned a decline into a decline. ``_step`` survives because
the ``ulp`` SAMPLING strategy uses it to choose points, which is a
different job and cannot admit anything.

**WHAT THAT COSTS, MEASURED, BECAUSE IT IS NOT FREE.** A program the
replay cannot read THROUGH is a program this probe cannot fire on,
however false the obligation is. Three separate things produce that, and
they are not the same finding:

* **Steps with no exact rational reading at all** — ``exp``, ``log``,
  trigonometry, a fractional ``pow``, a non-square ``sqrt``. These are
  inherent. An exactness-only fire condition will never reach them, and
  the honest form of that is a decline that names the primitive.
* **Steps that HAVE an exact reading and are simply not in this file's
  tables** — ``dot_general``, ``sort``, ``cumsum``, ``stack``, ``rem``,
  ``scatter`` and ``scatter-add``. These are a table, not a limit, and
  the cost is concrete: three of the six live fixtures in
  ``tests/test_falsify_probe.py`` were ``scatter`` and now decline. They
  are listed there under ``DECLINED_FOR_WANT_OF_AN_EXACT_READING`` with
  the primitive that costs each one, and adding a reading turns the test
  red so the recovery announces itself.
* **Programs the replay COULD read but is not given the time to.** The
  wall-clock backstop is not the generous margin its constant's comment
  used to claim: the deterministic pair already permits about 4.75
  seconds of ``Fraction`` arithmetic against a 5.0-second clock, and a
  ``(35000,)`` float64 declaration squared six times replays in 1.55 s on
  the machine this was measured on. **So whether this probe fires on a
  given program can depend on the machine it runs on**, and that is
  stated here rather than left to be inferred from a constant. What
  varies is REACH and never soundness — the clock can only ABSTAIN and
  only an exact test may admit, so a slow machine declines what a fast
  one refutes and never the reverse — and a decline the clock produced
  names itself in ``ProbeReport.abstentions``. The deterministic bounds
  are pushed in front of it where they can be: the rational-width budget
  binds as each element is produced rather than after the equation that
  trips it is complete (``(30000,)`` squared seven times: 3.10 s then
  declined, now 1.37 s then declined).

The second list is where this instrument's reach is bought back, and
nothing in it is bought by relaxing the first paragraph of this section.

--------------------------------------------------------------------------
THE CLASS: A PREDICATE COMPUTED FROM SOMETHING ELSE, FIVE TIMES
--------------------------------------------------------------------------

Five audits of this module have found five defects and they are the same
defect. **Every one is a predicate that licenses a claim about the
PROGRAM while being computed from something else**, and every one let this
probe raise the categorical *"stelling is UNSOUND at this query"* about a
verdict that is CORRECT, in a handful of lines through the public API,
with no mutation and no solver:

1. **ulp-stability of the INPUT** standing in for *"this violation is not
   a rounding artefact"*. Kahan compensation, four lines.
2. **a fall-back to that same proxy** standing in for *"an exact test
   admitted this"* — with a jax primitive named that does not exist, so
   the exact test rarely ran and the proxy usually did.
3. **a predicate over the DECLARED DTYPES** standing in for *"this
   program does integer arithmetic"*. An ``int16`` declaration cast to
   ``float32`` rounds, and the parenthetical said no rounding was
   involved.
4. **the TOP-LEVEL ``stelling_assume`` walk** standing in for *"a point
   the analysis itself admitted"*. ``_execute`` iterates ``jaxpr.eqns`` at
   the top level, so an assume one ``jit`` deep executes and never reaches
   ``run.assumes``, and ``if run.assumes and not all(...)`` on an empty
   list admits every point::

       x = any_array((), "int32", (0, 10))
       y = jax.jit(lambda a: (assume(a >= 9), a)[1])(x)
       assert_(y >= 9)                        # VERIFIED, and correct

   ``propagate`` DOES narrow on that assume — it is the whole reason the
   VERIFIED exists — so the probe fired at ``x = 0``, a point outside the
   region the analysis had claimed anything about. Reached in all four
   shapes it has: through ``jit`` and through ``remat2``, under
   ``semantics="real"`` (via the all-integral branch) and under
   ``semantics="ieee"``, on both supported jax series and through all
   three public doors.

5. **the OP-BY-OP executed float** standing in for *"the executed float IS
   the subject of the claim"*, which is the sentence ``ieee`` semantics
   fires on. ``_execute`` hands jax one equation at a time, so XLA never
   sees two of them together; the user's own call compiles whole regions.
   The two are not the same float, and the difference is not confined to
   programs that say ``jit``: ``jax.make_jaxpr`` INLINES the ``jit``
   ``jnp.mean`` is built out of, so the trace carries a bare
   ``reduce_sum ; div`` where the caller's call compiled a region. Four
   lines, no ``jit`` written anywhere, on both series::

       X0    = 1.3102272059107631
       mean3 = lambda x: jnp.mean(jnp.stack([x, x * 2.0, x * 3.0]))
       C     = float(mean3(jnp.asarray(X0, "float64")))   # its OWN value
       x     = any_array((), "float64", (X0, X0))
       assert_(mean3(x) <= C)      # FIRED, margin -4.44e-16

   The obligation holds at the only point declared, eagerly and under
   ``jax.jit``. ``jnp.mean`` and ``jnp.average`` disagree between the two
   routes on 70 and 72 of 200 random points; every other wrapper surveyed
   agrees 200 of 200. **And this one was written INTO the sentence that
   justified instance 4's repair** — *"what this loop reproduces is what
   the user's own code does: their un-jitted top level op by op, their
   ``jit`` compiled whole"* — which is why it is worth saying that the
   class is a class and not a list: the fifth instance was sitting inside
   the argument for the fourth.

**PATCHING ONE INSTANCE IS NOT CLOSING THE CLASS**, so the repairs are a
rule rather than a patch. The rule has three clauses, one per audit that
added one.

*First:* **a predicate that licenses a claim about the program is computed
from the program** — the form recorded when instance 3 was fixed.

*Second, and this is what instance 4 added:* **a reading of the program
can be PARTIAL, and a partial reading licenses nothing.** Instance 4's
predicate WAS computed from the program; it was computed from part of it.
So every quantity this probe reads is now checked against a census taken
at every depth there is (``_declaration_totals``) before it may license
anything, and a reading that comes up short declines by name. Two
quantities already had such a guard and nobody had asked for the rest:
asserts had ``obligation-count-changed``, declarations had *"the harness
declares no inputs to vary"* (which catches zero, not partial), assumes
had none. The table is ``_READINGS``, it covers every field of ``_Census``
and of ``_Run``, and a test holds it to those two dataclasses
field-for-field — a new quantity cannot arrive without either a guard or a
written argument that it cannot license anything.

*Third, and this is what instance 5 added:* **a reading whose value
depends on HOW it was taken is not a reading of the program at all.**
Instance 5's predicate was computed from the program and from all of it;
it was computed at a granularity the program does not have. Completeness
was the wrong axis, which is why ``_READINGS`` — a table about
partial-versus-total — did not and could not catch it. So the granularity
is measured the way the depth is: the same program is run at the same
point by a second route that compiles it whole
(:func:`_whole_program_route`), and an executed violation whose truth
value moves between the two routes declines
(``executed-float-depends-on-granularity``). The second route can only
ever DECLINE — it is consulted after a violation and never admits one —
so it costs reach and cannot buy a firing.

**AND THE STRUCTURAL OBSERVATION UNDERNEATH ALL FIVE.** ``_execute`` and
``_replay`` are two walkers with different depth behaviour, and the gap
between them is where these defects live. The obvious reconciliation —
make ``_execute`` descend the way ``_replay`` does — was DRIVEN, and it is
NOT AVAILABLE: ``prim.bind`` on a call equation compiles the whole body
and XLA contracts across it, so walking the body op by op computes
different floats. Over 22 one-line ``jit`` bodies on both jax series the
two disagree on 5, including a SIGN disagreement on 3 of 3 attempts for
``a*b + c`` — and a sign is what an obligation reads. A descending
``_execute``, written out in full, does repair the first four false alarms
above — and then raises *"stelling is UNSOUND at this query"*, under
``ieee-executed-float``, on ``jit(lambda p,q,r: p*q+r)(a, b0, c0) != 0.0``
with ``c0 = -fl32(a0*b0)``, an obligation the real program satisfies at
both of its declared points (it computes ``-9.49e-08`` and ``+6.10e-08``
there). **That is instance 5 again — the same predicate, at a granularity
the program does not have — reached by widening the walk instead of by
leaving it narrow**, which is what settles that the axis is granularity
and not depth: BOTH depth policies produce it, so no depth policy is the
answer to it.

**AND REACH PRESERVATION HAS NOW BEEN REFUSED ON DATA TWICE, NOT ONCE.**
It is also why the reach-preserving option was refused on a measurement
rather than on a preference — and the refusal now rests on two
measurements. The second alternative keeps the call compiled and threads
the body's intermediates out of it as extra outputs, so that a descending
walk would not be needed to see them. It is **0 of 3 bitwise-identical**
to the plain compiled call on the same ``a*b + c`` fixture, on both
supported series, and it fails the same way the descending walk does: the
threaded call returns exactly ``0.0`` where the plain one returns the
rounding error of the product. Exposing the ``mul`` is itself a change to
the program,
because the value that has to be materialised is the one XLA was
contracting away. There is no third option of that shape — an
intermediate you can read is an intermediate XLA did not fuse.

The walkers are reconciled by MEASUREMENT instead: neither is trusted to
be complete because of how it is written, each one's reading is checked
against the census, and the executed reading is checked against a second
route that compiles the program whole. The argument in full is in
:func:`_execute`.

**A SIXTH INSTANCE, FOUND INSIDE THE PROBE'S OWN REPORT.**
``points_admissible`` is the count of points *"inside the declared set and
admitted by every assume"*, and the assume half was the EXECUTED FLOAT
reading of the assumes standing in for *"in the assumed region"* — the
same shape as instance 1, one field over. The exact re-reading existed
(:func:`_confirm`) and ran **only at points where a violation had already
been found**, so on a clean run — the common case, and the one whose
stamp line reads most like a result — the correction never happened and
the number stood unqualified. Measured on a clean VERIFIED,
``assume(y*0.1*10.0 <= y)`` over ``float64 [0, 2]``: *"71 point(s)
executed, 55 inside the declared set and admitted by every assume … NO
VIOLATION WAS FOUND"*, and **47 of those 55 are not in the assumed region
over ℚ**. That is this module's own *"reads as coverage"* failure,
occurring in this module's own report. The re-reading is now taken at the
gate on every admissible point under ``real`` semantics, on one shared
budget, and what could not be read is counted and said
(:attr:`ProbeReport.points_admissible_unconfirmed`) rather than folded
in. The same fixture now reports **8 admissible, 47 declined
``assume-unsatisfied-over-the-rationals``**.

**AND THE ANALYSIS'S REGION IS NEVER READ, WHICH IS NOT AN INSTANCE OF
THE CLASS BUT IS ITS MIRROR — AND IT IS OPEN.** Every predicate above is
about reading the PROGRAM correctly. This one is about reading the
ANALYSIS, and this module does neither correctly nor incorrectly: it does
not do it at all.

The assume gate reads **the program's assume TEXT**. It admits any point
at which every ``stelling_assume`` evaluates true — in floats at the
executed gate, and over ℚ at the exact one. What the analysis actually
discharged its obligations over is something else: ``propagate``
constrains a BOX from those assumes and reasons over that box. Soundness
of a firing needs the analysis's region to be a **superset** of the
pointwise-satisfying set the probe samples from, because a point in the
probe's set but outside the analysis's is a point the analysis never
claimed anything about — and a firing there is a false alarm of exactly
the kind this module keeps producing. **Nothing checks that**, and by
design nothing here can: checking it would mean reading the narrowed
region, which lives in the module this probe may not import, and the
whole value of the probe is that it does not.

Which direction the gap runs in matters and both are open. If a
``constrain`` rule OVER-approximates, the probe is the conservative side
and loses nothing. If one ever UNDER-approximates — narrows to less than
the assume admits — the probe fires at a point the analysis was never
asked about, and reports it as *"stelling is UNSOUND at this query"*. The
verdict already emits *"precondition satisfiability UNCERTIFIED … its box
may exceed its true image"* on some of these shapes, and **the probe does
not read that note either**; it now receives the stamped assumptions (for
the attribution split in :func:`_fire`) but it does not gate on them.

Fixing this is out of scope for a module built on importing no analysis,
and it may not be fixable inside that constraint at all — the honest
repair is a second, ANALYSIS-side check that the constrained box contains
the pointwise-satisfying set, which belongs where the constraining
happens. Concealing it is not out of scope, which is why it is here.
"""

from __future__ import annotations

import itertools
import math
import random
import time
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
    "DECLARED_NOT_VERIFIED",
    "DECLINE_REASONS",
    "ProbeInvariantViolated",
    "ProbeReport",
    "SEED_LABEL",
    "STRATEGIES",
    "VerifiedFalsified",
    "probe",
    "unverified_declarations",
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
#
# `assume-not-fully-executed` is the fourth defect's guard, and it is the
# one that says the quiet part: `_execute` walks the TOP LEVEL of the
# jaxpr, so a `stelling_assume` inside a `jit` or a `remat2` body executes
# but never reaches `run.assumes`, and the gate that reads *"admitted by
# every assume"* off that list was vacuous exactly there. See `_READINGS`.
#
# `executed-float-depends-on-granularity` is the fifth defect's guard and
# says its own quiet part: `_execute` hands jax ONE equation at a time, so
# the float it computes is the one the TRACE's granularity produces and
# not the one the caller's own call does -- and `jax.make_jaxpr` inlines
# the `jit` inside `jnp.mean`, so this is reached by programs that say
# `jit` nowhere.  The point declines when the same program, run at the
# same point as one compiled region (`_whole_program_route`), reads the
# obligation the other way.  `whole-program-route-unavailable` is the
# other side of the same guard and is NOT the same finding: nothing has
# been shown to move, the second reading simply could not be taken.
#
# `precision-ambiguous` is gone with the ulp proxy that emitted it, and
# `no-exact-reading-of-this-program` is new and is the one to watch: it is
# the reason every program this instrument cannot read exactly declines
# under, and its count IS the reach cost of requiring an exact refutation.
# The reasons the exact reading was unavailable are counted separately, in
# `ProbeReport.abstentions`, because "declined 400 times" and "declined 400
# times, all of them `'dot_general' has no exact rational reading`" are
# different facts and only the second one tells a reader what to do.
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
    # AND THE ONE THIS TUPLE SHIPPED WITHOUT, WHICH IS THE FAILURE THIS
    # TUPLE EXISTS TO PREVENT.  `dtype-narrowed-by-jax` is emitted three
    # lines after the `_window` loop, for a declaration whose dtype does
    # not survive `jax.numpy.asarray` under the live config -- a 64-bit
    # box under `jax_enable_x64=0`.  It is a live, user-visible decline and
    # it was NOT here, because the completeness test that guards this
    # tuple grepped for three EMISSION SHAPES (`skips.add("...")`,
    # `return None, "..."`, and `_confirm`'s tuple) and this reason is an
    # f-string inside a `ProbeReport(declined=...)` -- a fourth shape,
    # invisible to all three.  A completeness check a grep cannot reach is
    # the same defect as no check, which is this file's own rule; the scan
    # in `test_every_decline_reason_is_declared_and_accounted_for` no
    # longer reads emission shapes at all.  It reads the module's
    # hyphenated-code VOCABULARY -- every string literal that IS a
    # reason-shaped code, and every reason-shaped token inside any
    # `declined=` argument -- so a fifth emission shape cannot hide
    # either, and every such token has to be classified as a decline
    # reason or as something else by name.
    "dtype-narrowed-by-jax",
    # a built point that could not be used
    "point-outside-declaration",
    "program-raised",
    "obligation-count-changed",
    "assume-unsatisfied",
    "assume-not-fully-executed",
    # AND THE EXACT HALF OF THE SAME GATE, WHICH IS WHERE THIS REASON NOW
    # BELONGS.  It was filed below, under "an executed VIOLATION the fire
    # condition would not report", and that was true of it when `_confirm`
    # was the only place it could be emitted: back then the point had
    # already violated the obligation in floats and the exact re-reading
    # of the assumes was what took the firing back.  The re-reading now
    # runs at the GATE, on every admissible point, so the overwhelming
    # majority of these are points at which NO VIOLATION WAS EVER LOOKED
    # FOR -- measured on the `assume(y*0.1*10.0 <= y)` fixture, 47 of 47
    # come from the gate and none from `_confirm`.  The `_confirm` site
    # remains and is now the residue (see the `point_unassumed` branch in
    # `probe.run_one`), so the reason genuinely lives in both groups; it
    # is filed with the gate because that is where a reader will meet it.
    "assume-unsatisfied-over-the-rationals",
    # an executed VIOLATION the fire condition would not report
    "float-rounding-artefact",
    "no-exact-reading-of-this-program",
    "executed-float-depends-on-granularity",
    "whole-program-route-unavailable",
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

# THE MARK OF A VERDICT THAT IS NOT STELLING'S OWN CLAIM.
#
# A firing raises the categorical *"stelling is UNSOUND at this query"*.
# That sentence is only true when the discharge was stelling's to make.
# It is not, whenever the verdict rests on something the caller DECLARED
# and stelling says in the same breath that it cannot check: the
# `libm_budget` line stamps *"ieee libm accuracy DECLARED, NOT VERIFIED
# ... TWO claims compose to make this verdict and stelling checks
# NEITHER."*  Under-declare that budget and an `ieee` VERIFIED exists ONLY
# because of the bad declaration -- and the probe, which returns on
# `ieee-executed-float` before any exact test, then raised an
# `AssertionError` that stops the caller's CI and points them at
# `stelling/falsify.py`.  Driven, public API, no mutation: an `exp`/float32
# harness at X = 88.72167205810547 with a hand-written 0-ulp profile is
# VERIFIED and FIRES, while the honest shipped profile `xla-cpu-2026-08`
# (6 ulps) returns UNKNOWN on the same harness -- so the counterexample is
# REAL and the attribution was not.  That is the accounting failure this
# module rejected the REFUTED disposition to avoid, committed in the other
# direction: there it would have reported stelling's defect as the
# caller's, here it reported the caller's declaration as stelling's.
#
# THE FIRING IS KEPT AND THE MESSAGE IS SPLIT, rather than declining.
# Declining would throw away the one piece of evidence the caller most
# needs: the probe has just EXECUTED the program and shown the declared
# budget does not hold on this backend.  So the raise stands, and
# `_fire` names the declaration instead of accusing stelling.
#
# READ AS DATA, NEVER IMPORTED.  The line arrives as one of the
# `assumptions` strings the caller passes, exactly as `statuses` does;
# nothing here imports `propagate`, `verdict`, or anything that produced
# it.  The coupling is a PHRASE, so it is pinned by a test that is allowed
# to import the analysis:
# `tests/test_falsify_fire_condition.py::test_the_shipped_libm_budget_line_carries_the_phrase_the_probe_splits_on`.
DECLARED_NOT_VERIFIED = "DECLARED, NOT VERIFIED"

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

    **AND THAT IDIOM MUST NOT ALSO CATCH
    :class:`ProbeInvariantViolated`,** which is why that class is NOT an
    ``AssertionError`` and not an ``Exception`` at all.  A batch caller
    who writes ``except AssertionError`` around ``check()`` is asking for
    *"tell me when a verdict was falsified"*; a broken probe invariant is
    the opposite statement -- the probe has nothing to say about the
    verdict -- and answering the first question with the second is how a
    silent instrument gets read as a clean run.  The two classes are now
    disjoint in both directions: no ``except`` clause catches both without
    naming both.
    """

    def __init__(self, message: str, report: "ProbeReport"):
        super().__init__(message)
        self.report = report


class ProbeInvariantViolated(BaseException):
    """A fact this probe's readings rest on did not hold.

    **NOT A FIRING**, and deliberately a different class from
    :class:`VerifiedFalsified` so that no consumer can confuse the two:
    a firing is a statement about a VERDICT, and this is a statement that
    this module's own preconditions were not met and therefore that it has
    nothing to say about the verdict either way.

    It is raised rather than declined because the shapes that reach it are
    ones where declining would be a quieter form of the same wrong answer:
    the probe would report "nothing was executed" when what happened is
    that an invariant somewhere else in the package changed underneath it.

    **``BaseException``, ON THIS PACKAGE'S OWN PRECEDENT, AND IT SHIPPED
    AS AN ``AssertionError`` FOR ONE RELEASE CANDIDATE.**  The two other
    instruments this project builds on the same argument --
    :class:`stelling.EagerTruncationError` and
    :class:`stelling.NarrowingError` -- both inherit directly from
    ``BaseException`` and both carry the reason in their own docstrings:
    *an ordinary ``except Exception:`` must not be able to swallow an
    instrument's alarm.*  This class had exactly the property those two
    were written to avoid, in the module least able to afford it: **this
    file contains eight bare ``except Exception`` handlers**, and three of
    them sit on the paths an invariant check would be placed on.  Driven,
    with a raise injected at each site on the tree that shipped it:

    * inside :func:`_replay` during the assume gate -> caught by
      ``probe.assumes_over_the_rationals``'s ``except Exception: return
      None``, and the point is counted as an ordinary UNCONFIRMED;
    * inside :func:`_replay` during :func:`_confirm` -> caught by that
      function's ``except Exception``, and the stamp line prints
      *"8 x ProbeInvariantViolated: ..."* among the ordinary abstentions,
      as though the exact reading had merely been unavailable;
    * inside :func:`_read` -> caught by :func:`probe`'s ``except
      Exception``, and the whole probe reports the ordinary decline *"the
      probe could not read the traced program"*.

    It was **also** caught by ``except AssertionError``, which is the
    idiom :class:`VerifiedFalsified`'s own docstring tells a batch caller
    to write in order to catch a soundness event on purpose -- so the one
    caller who had followed this module's advice was the one guaranteed to
    swallow this.  Both properties are now impossible by construction:
    ``BaseException`` is outside ``Exception`` and outside
    ``AssertionError``.

    **WHAT THAT LETS ESCAPE IS THE POINT, AND THE PATHS WERE CHECKED ONE
    BY ONE.**  Every ``except`` in this module and in
    ``stelling.preconditions`` was read against this change.  Five of the
    eight bare handlers wrap work an invariant check does or could sit
    inside -- :func:`_execute`'s equation loop,
    :func:`_granularity_stable`'s second route, :func:`probe`'s read,
    ``probe.assumes_over_the_rationals``'s confirmation replay, and
    :func:`_confirm`'s replay -- and all five now let this class through
    instead of converting it into a skip, a decline, an unconfirmed point
    or an abstention; each says so at its own site.  Two of the other
    three wrap a single arithmetic conversion apiece (reading a ``pow``
    literal in :func:`_read`, reading a margin in :func:`_execute`) and
    can raise nothing but the numeric errors they are for; the eighth is
    :func:`_dtype_after_jax`, which asks jax what it would do with one
    dtype and answers *"the same dtype"* when jax refuses the question --
    a refusal :func:`_window` has already declined for by the time this
    runs.  The remaining handlers
    name concrete types (``TypeError``,
    ``ValueError``, ``OverflowError``, :class:`_Unreplayable`,
    :class:`_AssumesComplete`) and are unaffected.  ``preconditions``
    catches only ``ir.TranscriptionError`` and ``NestedDeclaration`` and
    wraps the :func:`probe` call in nothing at all, so a violated
    invariant reaches the caller's frame, which is where it belongs.

    A caller who genuinely wants to survive one must name it.  That is the
    same contract ``EagerTruncationError`` and ``NarrowingError`` offer,
    and it is why this class is in ``__all__``.
    """


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
    # WHICH TEST LET THIS THROUGH.  Three tests may admit -- exact-rational
    # replay, exact integer arithmetic, and the executed float under `ieee`
    # semantics -- and they are not the same claim, so the message a reader
    # acts on says which one it was.  This field was added when a fourth,
    # weaker test could also admit; it survives that test's removal because
    # naming the adjudicator is what makes a firing auditable, and because
    # the next weak test to be proposed will have to say its own name.
    adjudication: str = "unrecorded"
    # WHAT THIS OBLIGATION'S VALUE IS COMPUTED FROM: every primitive in
    # the backward slice of the violated `stelling_assert`, at every
    # depth, read off the traced program by
    # :func:`_obligation_operations`.  It is here because :func:`_fire`
    # needs a PER-OBLIGATION fact and the only per-obligation facts the
    # probe is handed are the status and the program; the stamped
    # assumptions arrive per VERDICT.  See :func:`_fire` for what the
    # message may and may not conclude from it.
    operations: tuple[str, ...] = ()

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
    # in-box AND every assume satisfied -- and BOTH halves are readings of
    # the program that can come up short, which is why this number is
    # taken back when a later, exact reading contradicts an earlier float
    # one (`assume-unsatisfied-over-the-rationals`) and is not given at
    # all when the executed run could not see every assume
    # (`assume-not-fully-executed`).  It shipped as the float reading
    # alone and produced a stamp line that contradicted itself in one
    # sentence.  Re-derived at `cefc4a9` on `assume(y*0.1*10.0 <= y)`
    # under the Kahan assert: *"74 point(s) executed, 65 inside the
    # declared set and admitted by every assume ... declined 9
    # assume-unsatisfied, 39 assume-unsatisfied-over-the-rationals"* --
    # 39 of those 65 were points the exact replay had just said NO assume
    # admitted.
    #
    # **AND THE TAKE-BACK USED TO RUN ONLY WHERE A VIOLATION HAD ALREADY
    # BEEN FOUND, WHICH IS THE UNCOMMON CASE.**  The exact re-reading lived
    # in `_confirm`, and `_confirm` is consulted only at a point where the
    # obligation evaluated FALSE.  So on a CLEAN run -- the common case,
    # and the one whose stamp line a reader is most likely to take at face
    # value -- the take-back never happened and this number stood as an
    # unqualified count of points "admitted by every assume" on the
    # strength of an EXECUTED FLOAT reading of the assumes alone.  That is
    # this module's own *"reads as coverage"* failure, occurring inside
    # the module.  Measured on a clean VERIFIED, `assume(y*0.1*10.0 <= y)`
    # over `float64 [0, 2]` with a trivially true assert: *"71 point(s)
    # executed, 55 inside the declared set and admitted by every assume
    # ... NO VIOLATION WAS FOUND"* -- and **47 of those 55 points are NOT
    # in the assumed region over ℚ** (`y*0.1*10 > y` for every y > 0 as
    # rationals, because the float `0.1` is above one tenth).
    #
    # The re-reading is therefore taken at the GATE, on every admissible
    # point and not only on violating ones, under `semantics="real"` --
    # the semantics whose assumes are a claim about ℝ.  What could not be
    # re-read is counted separately and reported, never folded in:
    # `points_admissible_unconfirmed` below.
    points_admissible: int = 0
    # ...of which THIS MANY rest on the executed float reading of the
    # assumes alone, because no exact reading of them was available: the
    # replay abstained (an irrational step, a primitive with no rational
    # reading, a value past the width budget) or the whole-probe
    # confirmation budget was spent.  Under `semantics="ieee"` it is ALL
    # of them, and that is not a gap: an `ieee` verdict is a claim about
    # the floats the program computes, so the executed reading of the
    # assumes is the reading the claim is about, and re-reading them over
    # ℚ would answer a question nobody asked.
    #
    # Reported rather than subtracted.  A point whose assumes could not be
    # re-read is not a point shown to be outside the assumed region; it is
    # a point about which the exact question was not answered, and this
    # module does not turn "not answered" into either answer.
    points_admissible_unconfirmed: int = 0
    points_declined: int = 0  # admissible, VIOLATED, and not reported
    violations_seen: int = 0  # executed points at which an attacked
    # obligation evaluated FALSE in floats, however they were then judged
    adjudications: tuple[tuple[str, int], ...] = ()
    # WHY the exact reading was unavailable, counted by reason, on the
    # violations that declined for want of one.  This is the reach cost of
    # requiring an exact refutation, said in the units a reader can act on:
    # "declined 400" is a number, "declined 400, all `'dot_general' has no
    # exact rational reading`" is a instruction to go and add `dot_general`.
    abstentions: tuple[tuple[str, int], ...] = ()
    strategy_points: tuple[tuple[str, int], ...] = ()
    strategy_hits: tuple[tuple[str, int], ...] = ()
    skips: tuple[tuple[str, int], ...] = ()
    falsification: Falsification | None = None
    declined: str | None = None  # the whole probe declined; why
    # HOW THE ASSUMES WERE READ, which decides what the admissible count
    # above is a count OF.  The semantics the verdict was made under, and
    # how many `stelling_assume` equations the program contains at every
    # depth: with none, "admitted by every assume" is vacuous and there is
    # no float-versus-exact question to have; with one or more there is,
    # and the stamp line says which reading was taken.
    semantics: str = "real"
    assumes_in_program: int = 0
    # The verdict's stamped assumption lines, as handed to :func:`probe`.
    # Carried on the report so a firing can name the DECLARATIONS a
    # conditioned verdict rests on instead of accusing stelling —
    # see :func:`_fire` and :data:`DECLARED_NOT_VERIFIED`.
    assumptions: tuple[str, ...] = ()

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

    def _unread_clause(self) -> str:
        """The reasons an exact reading was unavailable, or nothing.

        Only an EXACT test may admit a firing (:func:`_confirm`), so a
        program this module cannot read exactly is a program the probe
        cannot fire on however false it is.  A stamp line that reported
        the decline count and not the reason would hand a reader a number
        with no next step; this hands them the primitive to go and add.
        """
        if not self.abstentions:
            return ""
        why = "; ".join(f"{n} x {text}" for text, n in self.abstentions)
        return (
            f". THE EXACT READING WAS UNAVAILABLE, WHICH IS WHY THEY WERE "
            f"DECLINED AND NOT WHY THEY WERE ABSENT ({why})"
        )

    def _admissible_clause(self) -> str:
        """WHICH reading of the assumes stands behind the admissible count.

        The count is *"inside the declared set and admitted by every
        assume"*.  The first half is exact — :func:`_admissible` compares
        the built point against the declared endpoints themselves.  The
        second half is a reading that can be taken two ways, and saying
        which was taken is the difference between a number and a number
        that reads as coverage.  Five cases, and each one is a different
        fact:

        * **no assume in the program.**  There is nothing to read: the
          conjunction over an empty set is true under every arithmetic,
          and a qualifier here would manufacture a doubt that does not
          exist.  Said explicitly, because silence would be indistinguishable
          from the case below.
        * **``ieee`` semantics.**  The verdict is a claim about the floats
          the program computes, so the executed float reading of the
          assumes IS the reading the claim is about.  Named rather than
          left implicit — it is still an executed reading, taken at the
          trace's granularity, and a reader is entitled to know that.
        * **``real``, every point re-read over ℚ.**  The number has exact
          evidence and says so.
        * **``real``, some points not re-read.**  The number is a mixture
          and the sentence gives both halves.  Never folded into one
          number: a point whose exact reading could not be taken is not a
          point shown to be in the assumed region.
        * **``real``, NO point admissible.**  Zero points re-read is not
          "every one of them re-read"; it is no reading.  This case used
          to fall through the branch above and print *"0 ... every one of
          them re-read over ℚ and confirmed"*, a confirmation clause on a
          run that confirmed nothing.  It is reached by any program whose
          assume rejects every sampled point over ℚ, which is not exotic:
          ``assume(y*0.1*10.0 <= y)`` over ``float64 [0, 2]`` gets there
          whenever the sampler happens to draw only positive points.
        """
        if self.assumes_in_program <= 0:
            return (
                "inside the declared set, which this program states "
                "no assume to narrow"
            )
        head = "inside the declared set and admitted by every assume"
        if self.semantics == "ieee":
            return (
                f"{head} AS THE PROGRAM EXECUTED THEM IN FLOATS, which is "
                f"the arithmetic this ieee verdict is about"
            )
        unread = self.points_admissible_unconfirmed
        if self.points_admissible <= 0:
            # AND ZERO IS NOT "ALL OF THEM".  The branch below reads
            # `unread <= 0` and says *"every one of them re-read over
            # ℚ and confirmed"*, which at zero admissible points
            # rendered *"0 inside the declared set and admitted by every
            # assume, every one of them re-read over ℚ and confirmed"* --
            # a sentence claiming exact evidence for a run that took no
            # exact reading at all, in the report whose entire job is to
            # say which reading stands behind which number.  Vacuous
            # truth is not the standard this method is held to: the
            # clause exists to stop a number reading as coverage, and a
            # confirmation clause on a run that confirmed nothing is that
            # failure in miniature.
            return (
                f"{head} — none, so no assume was re-read over ℚ here "
                f"and this run carries no exact evidence about the "
                f"assumed region"
            )
        if unread <= 0:
            return f"{head}, every one of them re-read over ℚ and confirmed"
        confirmed = self.points_admissible - unread
        return (
            f"{head} — {confirmed} of them re-read over ℚ and confirmed, "
            f"and {unread} on the executed float reading of the assumes "
            f"alone, which is not evidence that those {unread} are in the "
            f"assumed region"
        )

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

        **THE THIRD IS HOW THE ASSUMES WERE READ**, and it is the same
        failure a third time: *"N inside the declared set and admitted by
        every assume"* was, on a clean run, a count taken on the EXECUTED
        FLOAT reading of the assumes with no exact evidence behind it, and
        it read as coverage of the assumed region.  Measured, 47 of 55.
        The reading is now taken exactly at the gate under ``real``
        semantics, and :meth:`_admissible_clause` says which reading
        stands behind the number, in the same sentence as the number.
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
            f"executed, {self.points_admissible} "
            f"{self._admissible_clause()}, across {self.obligations} "
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
                f"({how}){self._unread_clause()}. Every other point left "
                f"the obligation true, WHICH IS NOT EVIDENCE THAT THERE IS "
                f"NO VIOLATION: this probe can only refute, and a null "
                f"result is a fact about the sampler, not about the "
                f"verdict."
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


def _dtype_after_jax(name: str) -> str:
    """What ``jax.numpy.asarray`` makes of a numpy array of dtype ``name``.

    The identity under ``jax_enable_x64=1``.  Under ``jax_enable_x64=0``
    jax NARROWS the 64-bit formats on conversion -- ``float64`` becomes
    ``float32``, ``int64`` becomes ``int32`` -- silently, and that is the
    whole reason this function exists: :func:`probe` declines a
    declaration it cannot hand to jax unchanged, because the executed run
    and the exact rational reading would then be readings of two different
    programs.  See the decline in :func:`probe` for the driven evidence.

    A dtype jax refuses outright answers with its own name, so the caller
    declines for the reason it already has (:func:`_window` reaches the
    unsupported formats first) rather than for this one.
    """
    try:
        return str(
            np.dtype(jax.numpy.asarray(np.zeros((), dtype=name)).dtype).name
        )
    except Exception:  # noqa: BLE001 - jax cannot hold it; not this decline
        return name


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
        # INTERSECTED WITH THE DECLARATION, LIKE EVERY OTHER BRANCH.  This
        # one returned `(0, 1)` flat — the only branch that ignored the
        # two numbers it was handed.  `any_array((), "bool", (0.0, 0.0))`
        # declares FALSE and nothing else, and the sampler built `True`
        # anyway; only `_admissible` stopped it, at 4 wasted
        # `point-outside-declaration` skips per run on a two-value box.
        #
        # It also falsified two of this module's own written claims: that
        # an in-window candidate is inside the declared box BY
        # CONSTRUCTION (:func:`_representable`'s docstring), and that
        # `_admissible` had never rejected a live point — it had, four
        # times per run, on exactly this shape.  A guard whose rejections
        # are all manufactured by one unintersected branch is a guard
        # nobody can read a signal off.
        #
        # A bool is an integer set of `{0, 1}`, so the rule is the integer
        # rule: ceil the low end, floor the high end, clamp to the dtype,
        # and decline an empty box under the name the integer branch
        # already uses.
        # `(-inf, inf)` is a legal, and COMPLETE, bool declaration: both
        # values are inside it.  An infinite endpoint on a two-element set
        # never leaves the box unbounded, so it is replaced by a finite
        # value on the same side of `{0, 1}` and the intersection below
        # does the rest -- the integer branch's `unbounded-declaration`
        # has nothing to say about a set of two elements.
        if math.isinf(lo):
            lo = -1.0 if lo < 0 else 2.0
        if math.isinf(hi):
            hi = 2.0 if hi > 0 else -1.0
        a = max(int(math.ceil(lo)), 0)
        b = min(int(math.floor(hi)), 1)
        if a > b:
            return None, "empty-integer-box"
        return (a, b), None
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
        # AND SNAPPED INWARD ONTO THE DTYPE'S OWN GRID.  The declared set
        # is the values OF THIS DTYPE inside [lo, hi], and `0.3` is not a
        # value of `float32`: `np.full((), 0.3, "float32")` is
        # 0.30000001192092896, which is outside `(-0.3, 0.3)` over R.  So
        # the sampler built both corners of that declaration and
        # `_admissible` -- correctly -- threw all ten of them away
        # (`point-outside-declaration: 10`), at exactly the corners the
        # `endpoints` strategy exists to reach.  `_admissible` is the
        # guard and must stay strict; it is the WINDOW that was wrong.
        # Snapping is exact and costs nothing at float64, where the cast
        # is the identity and both branches below are no-ops.  A box that
        # falls BETWEEN two adjacent values of a narrow format snaps to
        # `a > b` and declines under the same reason as a box outside the
        # format's range, which is the same fact said twice: the declared
        # set holds no value of this dtype.
        a, b = _snap(a, +1, dt), _snap(b, -1, dt)
        if a is None or b is None or a > b:
            return None, "box-outside-the-dtype-range"
        return (a, b), None
    # `bfloat16` and the `float8_*` family land here: numpy classifies the
    # ml_dtypes extension types as `kind == "V"`, they have no `nextafter`,
    # and every phase of this sampler is built on stepping and comparing in
    # the declaration's own format.  That is a REACH GAP and it is named in
    # the module docstring's blind-spot section rather than left to be
    # discovered from a decline count.
    return None, "dtype-not-sampleable"


def _quantise(v, dt):
    """``v`` as the dtype will actually hold it, or ``None``.

    Every candidate this sampler produces is a Python float, and every
    point it builds goes through ``np.full(shape, v, dtype=dt)`` -- so the
    number the declaration is checked against must be the number AFTER
    that cast, not before it.  At ``float64`` this is the identity; at the
    narrow formats it is the difference between a point inside the
    declared box and a point just outside it.
    """
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            q = float(np.asarray(v, dtype=dt))
    except (TypeError, ValueError, OverflowError):
        return None
    return q if math.isfinite(q) else None


def _snap(v, direction, dt):
    """The nearest value of ``dt`` on the ``direction`` side of ``v``.

    ``direction`` is ``+1`` for "at or above" and ``-1`` for "at or
    below", both judged over ℝ in float64.  Returns ``None`` when the
    dtype has no such value.  The cast rounds to NEAREST and so can land
    on the wrong side; one step in the dtype's own format fixes that, and
    no more than one is ever needed.  The step has to be taken IN ``dt``:
    a float64 step off a float32 value rounds straight back onto it, which
    is the same class of mistake as comparing a float16 array against a
    float64 endpoint.
    """
    q = _quantise(v, dt)
    if q is None:
        return None
    if direction > 0 and q < v:
        q = _step_in(q, math.inf, dt)
    elif direction < 0 and q > v:
        q = _step_in(q, -math.inf, dt)
    if q is None or not math.isfinite(q):
        return None
    if (direction > 0 and q < v) or (direction < 0 and q > v):
        return None
    return q


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

    **AND IT USED TO SAY IT HAD NEVER REJECTED A POINT ON THE LIVE
    CORPUS.  THAT WAS FALSE, AND WHAT MADE IT FALSE WAS A DEFECT IN
    :func:`_window`.**  The sentence read *"0 of the 30,194 points the
    corpus built before this batch, measured"*, and it held only because
    that corpus has no ``bool`` declaration: the ``bool`` branch of
    :func:`_window` was the one branch that did NOT intersect with the
    declaration, so ``any_array((), "bool", (0.0, 0.0))`` -- which
    declares ``False`` and nothing else -- had ``True`` built for it and
    rejected here, four times per run.  A guard whose only live rejections
    come from one unintersected branch is a guard nobody can read a signal
    off, which is why the branch was fixed rather than the sentence.  With
    that done the claim is true again and is stated as what it is: a guard
    against a SAMPLER defect, live on a sampler that has none, and
    therefore driven deliberately by
    ``tests/test_falsify_fire_condition.py::test_the_admissibility_guard_rejects_a_point_the_sampler_should_not_build``
    rather than by an accident of one dtype.
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
    """Everything the probe needs, read off jax's own jaxpr, once.

    **EVERY FIELD HERE IS A READING OF THE PROGRAM, AND EVERY READING HAS
    A DEPTH.**  Some are taken at the top level of the jaxpr and some at
    every depth, and which one a field is decides what it may license.
    That is not a stylistic distinction: five separate defects in this
    module have been one predicate licensing a claim about the PROGRAM
    while being computed from something else, and the fourth was a
    top-level reading of ``stelling_assume`` licensing *"a point the
    analysis itself admitted"*.  (The fifth was not a DEPTH defect and no
    field of this dataclass would have caught it -- see the note under
    :data:`_READINGS`.)  The depths, the guards, and the reasons
    for the three fields that need no guard are in :data:`_READINGS`,
    which is checked field-by-field against this dataclass by
    ``tests/test_falsify_fire_condition.py::test_every_census_quantity_has_a_totality_guard``
    -- so a new field here cannot arrive without one.
    """

    closed: object
    declarations: tuple[Declaration, ...] = ()
    assert_positions: tuple[int, ...] = ()  # eqn index of each stelling_assert
    # for each assert, the (relation, small_atom, large_atom) of the
    # comparison that produced its operand, when there was one
    margins: dict = field(default_factory=dict)
    exponents: tuple[int, ...] = ()  # k of every integer_pow / pow in the program
    replay_cost: int = 0  # element-visits one exact-rational replay would do
    # every dtype in the PROGRAM -- operand and result, at every depth --
    # is an integer or a boolean one.  Defaulted to False because every
    # use of it licenses an exactness claim: a `_Census` built without
    # reading a jaxpr must not be able to license one by omission.
    integral: bool = False

    # HOW MANY OF EACH DECLARATION PRIMITIVE THE PROGRAM CONTAINS AT EVERY
    # DEPTH (:func:`_declaration_totals`), against which each partial
    # reading above -- and each partial reading `_execute` takes at run
    # time -- is checked before it may license anything.
    #
    # **DEFAULTED TO -1 AND NOT TO 0**, for the same reason `integral`
    # defaults to False: a `_Census` built without reading a jaxpr must
    # not be able to license "the reading was complete" by omission.  Zero
    # would agree with an empty reading and admit; -1 agrees with nothing
    # and declines.
    declarations_in_program: int = -1
    obligations_in_program: int = -1
    assumes_in_program: int = -1


@dataclass(frozen=True)
class _Reading:
    """One quantity this probe reads off the program, and its guard.

    ``subject`` is ``"census"`` or ``"run"`` and ``name`` is the field of
    :class:`_Census` or :class:`_Run` it describes; ``depth`` is where the
    reading is taken.  Exactly one of ``guard`` and ``why`` is set:
    ``guard`` names the decline a PARTIAL reading produces, and ``why`` is
    the argument that this particular reading cannot license anything and
    so needs none.

    ``site`` IS THE OTHER HALF OF ``guard`` AND IT WAS MISSING.  A guard
    name on its own says only that SOME line of this file emits that
    string, and a table that says only that is satisfied by a new field
    borrowing any decline reason already in the file -- driven: a new
    ``_Census`` field declared with ``guard="bound-nan"`` passed both
    tests that hold this table, and ``bound-nan`` is a ``_window``
    decline about a NaN endpoint that has nothing to do with any field.
    ``site`` is the SOURCE TEXT of the ``if`` that takes this field's
    guard, so the pair names a line rather than a file, and
    ``tests/test_falsify_fire_condition.py::test_the_guards_named_in_the_readings_table_are_LIVE_in_the_source``
    checks it against the parsed module: the ``if`` must exist and the
    guard must be emitted inside it.  Set exactly when ``guard`` is.
    """

    subject: str
    name: str
    depth: str
    guard: str | None = None
    site: str = ""
    why: str = ""


# THE RULE, IN A TABLE, BECAUSE FOUR AUDITS FOUND FOUR INSTANCES OF ONE
# DEFECT AND AN UNGUARDED NEW FIELD WOULD BE ANOTHER.
#
# Every one of the four was a predicate licensing a claim about the
# PROGRAM while being computed from something else, and the fourth --
# `stelling_assume` read at the TOP LEVEL, licensing *"a point the
# analysis itself admitted"* -- was possible because two of this module's
# quantities had a totality guard and the rest had never been asked for
# one.  Asserts had `obligation-count-changed`; declarations had *"the
# harness declares no inputs to vary"*, which catches zero and not
# partial; assumes had nothing at all.
#
# So the rule is stated once and applied to EVERY quantity: **a reading
# that may be partial is checked against what is in the program before it
# licenses anything, and declines when the two disagree.**  A reading
# needs no guard only when it cannot license a claim, and that argument is
# written in `why` rather than left to be re-derived.
#
# `tests/test_falsify_fire_condition.py::test_every_census_quantity_has_a_totality_guard`
# holds this table to the two dataclasses field-for-field, so a new
# quantity cannot arrive without either a guard or a written exemption,
# and `site` binds each guard to the `if` that takes it, so the guard
# cannot be a decline reason borrowed from somewhere else in this file.
#
# WHAT THIS TABLE DOES NOT CLOSE, SAID HERE RATHER THAN DISCOVERED AGAIN:
# it is a table about PARTIAL versus TOTAL, and the fifth instance of the
# class was not partial.  `_execute`'s reading of a float is total and is
# still not a reading of the program, because it is taken at the TRACE's
# granularity: `jax.make_jaxpr` inlines the `jit` inside `jnp.mean`, so
# the loop walks op by op what the caller compiles whole.  That guard is
# a second ROUTE rather than a second count (`_whole_program_route`), and
# it lives in `probe.run_one` with these.  Completeness and granularity
# are different axes and a table about one of them cannot close the other.
_READINGS = (
    _Reading(
        "census", "closed", "the-program-itself",
        why=(
            "not a reading -- it IS the traced program, the object every "
            "other reading is a reading OF"
        ),
    ),
    _Reading(
        "census", "declarations", "top-level",
        guard="the harness declares inputs the probe cannot see",
        site="len(census.declarations) != census.declarations_in_program",
    ),
    _Reading(
        "census", "assert_positions", "top-level",
        guard="the harness states obligations the probe cannot see",
        site="len(census.assert_positions) != census.obligations_in_program",
    ),
    _Reading(
        "census", "margins", "top-level",
        why=(
            "steering and display only: a margin chooses where `tight` and "
            "`ulp` look next and prints in the firing message. It admits "
            "nothing, and an assert whose margin is missing is still "
            "probed through its boolean -- `no-margin-no-boundary-search`"
        ),
    ),
    _Reading(
        "census", "exponents", "every-depth",
        why=(
            "sampling only: the exponents decide which perfect powers the "
            "`exact` strategy tries. A partial reading costs candidate "
            "points and can lose a refutation; it cannot invent one"
        ),
    ),
    _Reading(
        "census", "replay_cost", "every-depth",
        why=(
            "a budget, and it is compared with `>`: an UNDER-estimate lets "
            "a replay start that the two `_Guard` bounds then stop, and an "
            "over-estimate declines. Neither admits anything"
        ),
    ),
    _Reading(
        "census", "integral", "every-depth",
        why=(
            "guarded BY CONSTRUCTION rather than by a count: "
            "`_integral_program` answers False for any atom it cannot "
            "classify and for any sub-jaxpr that is not integral, so a "
            "partial reading of it is already a decline. This is the shape "
            "the other guards imitate"
        ),
    ),
    _Reading(
        "census", "declarations_in_program", "every-depth",
        why=(
            "a total, not a reading: it is what the guards are checked "
            "against"
        ),
    ),
    _Reading(
        "census", "obligations_in_program", "every-depth",
        why=(
            "a total, not a reading: it is what the guards are checked "
            "against"
        ),
    ),
    _Reading(
        "census", "assumes_in_program", "every-depth",
        why=(
            "a total, not a reading: it is what the guards are checked "
            "against"
        ),
    ),
    _Reading(
        "run", "assumes", "top-level",
        guard="assume-not-fully-executed",
        site="len(run.assumes) != census.assumes_in_program",
    ),
    _Reading(
        "run", "asserts", "top-level",
        guard="obligation-count-changed",
        site="len(run.asserts) != len(statuses)",
    ),
    _Reading(
        "run", "margins", "top-level",
        why=(
            "the executed side of `census.margins`, and the same argument: "
            "a margin steers the boundary phases and prints in the firing "
            "message, and admits nothing"
        ),
    ),
    _Reading(
        "run", "raised", "top-level",
        guard="program-raised",
        site="run.raised is not None",
    ),
)

# The guards above that stop the WHOLE probe with a `ProbeReport.declined`
# sentence rather than skipping one point with a `DECLINE_REASONS` code.
# They are whole-probe guards because what they find is a property of the
# program and not of a point: no other point would fare better.
_WHOLE_PROBE_GUARDS = (
    "the harness declares inputs the probe cannot see",
    "the harness states obligations the probe cannot see",
)


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
    totals = _declaration_totals(jaxpr)
    return _Census(
        closed=closed,
        declarations=tuple(decls),
        assert_positions=tuple(asserts),
        margins=margins,
        exponents=tuple(sorted(exponents)),
        replay_cost=_replay_cost(jaxpr),
        integral=_integral_program(jaxpr),
        declarations_in_program=totals["stelling_any"],
        obligations_in_program=totals["stelling_assert"],
        assumes_in_program=totals["stelling_assume"],
    )


def _obligation_operations(census, k) -> tuple[str, ...]:
    """Every primitive ONE obligation's value is computed from, at every depth.

    The backward slice of the ``k``-th top-level ``stelling_assert``: the
    equations its operand transitively depends on, plus every primitive
    inside any call body those equations reach.  The four ``stelling_*``
    declaration primitives are left out -- they are the query's own
    scaffolding, not arithmetic -- and the result is sorted and
    de-duplicated so it can be printed.

    **IT EXISTS BECAUSE :func:`_fire` NEEDS A PER-OBLIGATION FACT AND IS
    HANDED ALMOST NONE.**  ``statuses`` is per obligation; ``assumptions``
    is per VERDICT.  So when a verdict carries a declaration stelling does
    not check, nothing the probe is given says whether THIS obligation's
    discharge is one of the ones that declaration paid for.  This is the
    part of that question that is answerable exactly, off the program, and
    :func:`_fire` prints it and says plainly that the rest is not
    answerable here.

    **IT IS AN OVER-APPROXIMATION, DELIBERATELY, IN THE SAFE DIRECTION.**
    Where the slice reaches a call equation it takes EVERY primitive in
    that body rather than slicing the body too.  A cone that is too large
    can only make a declaration look more likely to bear on this
    obligation, which is the direction that keeps the message cautious; a
    cone that was too small could tell a reader a declaration is
    irrelevant when it is not, and that is the misdirection this whole
    function exists to remove.

    Returns ``()`` for an obligation whose operand is a literal or a bare
    declared input -- there is no arithmetic between the declaration and
    the assert -- and :func:`_fire` renders that case in words.
    """
    jaxpr = getattr(census.closed, "jaxpr", None)
    if jaxpr is None or k >= len(census.assert_positions):
        return ()

    producer = {}
    for eqn in jaxpr.eqns:
        for ov in eqn.outvars:
            producer[ov] = eqn

    def every_name(jx, out):
        for eqn in jx.eqns:
            nm = eqn.primitive.name
            if nm not in ("stelling_any", "stelling_assume",
                          "stelling_assert", "stelling_nonvacuity"):
                out.add(nm)
            for sub in _sub_jaxprs(eqn):
                every_name(sub, out)

    names: set[str] = set()
    seen: set[int] = set()
    frontier = list(jaxpr.eqns[census.assert_positions[k]].invars)
    while frontier:
        atom = frontier.pop()
        if isinstance(atom, jex_core.Literal):
            continue
        src = producer.get(atom)
        if src is None or id(src) in seen:
            continue
        seen.add(id(src))
        nm = src.primitive.name
        if nm not in ("stelling_any", "stelling_assume",
                      "stelling_assert", "stelling_nonvacuity"):
            names.add(nm)
        for sub in _sub_jaxprs(src):
            every_name(sub, names)
        frontier.extend(src.invars)
    return tuple(sorted(names))


def _declaration_totals(jaxpr) -> dict[str, int]:
    """How many of each stelling declaration primitive the program contains.

    **AT EVERY DEPTH**, through :func:`_sub_jaxprs`, which is the widest
    walk this module has: it follows a jaxpr wherever a parameter carries
    one, so a ``jit`` body, a ``remat2`` body, a ``cond`` arm, a ``scan``
    body and a ``custom_jvp`` rule all count.  That is deliberately wider
    than either walker: the number is not what any walker will see, it is
    what is THERE, and the difference between the two is exactly the
    quantity every guard in this module is checking.

    ``stelling_nonvacuity`` is counted with the others even though nothing
    gates on it, because the cost of counting is one dictionary entry and
    the cost of a quantity nobody counted is this batch.
    """
    totals = {
        "stelling_any": 0,
        "stelling_assert": 0,
        "stelling_assume": 0,
        "stelling_nonvacuity": 0,
    }

    def walk(jx):
        for eqn in jx.eqns:
            name = eqn.primitive.name
            if name in totals:
                totals[name] += 1
            for sub in _sub_jaxprs(eqn):
                walk(sub)

    walk(jaxpr)
    return totals


def _sub_jaxprs(eqn):
    """Every jaxpr nested in an equation's params, however it is wrapped."""
    out = []
    for v in eqn.params.values():
        for cand in (v if isinstance(v, (tuple, list)) else (v,)):
            inner = getattr(cand, "jaxpr", cand)
            if isinstance(inner, jex_core.Jaxpr):
                out.append(inner)
    return out


def _integral_atom(atom) -> bool:
    """Is this operand's or result's dtype an integer or a boolean one?

    A dtype that cannot be read or cannot be classified answers NO, which
    is the fail-safe direction here: the only caller uses this to license
    *"no rounding involved"*, and an unclassifiable dtype is not evidence
    for that sentence.  Saying no costs the integer branch and hands the
    point to the exact-rational replay, which declines what it cannot read.
    """
    dtype = getattr(getattr(atom, "aval", None), "dtype", None)
    if dtype is None:
        return False
    try:
        return np.dtype(dtype).kind in "iub"
    except (TypeError, ValueError):
        return False


def _integral_program(jaxpr) -> bool:
    """Does this PROGRAM compute in integers only, operand and result?

    **THIS PREDICATE IS READ OFF THE PROGRAM BECAUSE THE SENTENCE IT
    LICENSES IS ABOUT THE PROGRAM.**  :func:`_confirm` admits a firing on
    it under *"exact integer arithmetic: no rounding involved"*, and it
    used to be read off the DECLARATIONS instead -- which is a fact about
    the query's signature and no evidence at all about its arithmetic.  An
    int-declared program that converts to float and rounds satisfied it,
    and four lines through the public API then turned a CORRECT
    ``VERIFIED`` into *"stelling is UNSOUND at this query"*, with no
    mutation, no solver, on both supported jax series::

        x = any_array((), "int16", (0, 3))
        y = x.astype("float32")        # int16 -> float32 is whitelisted
        b = jnp.float32(2**24)         # ulp(2**24) is 2 in float32
        assert_((b + y) - b <= 3.0)    # over R this is `y <= 3`: TRUE

    Over ℚ the left-hand side IS ``y`` and ``y`` is at most 3, so the
    ``VERIFIED`` is right.  In float32 at ``x = 3`` the sum ties up to
    ``2**24 + 4`` and the program computes ``4.0``, so the probe fired --
    under a parenthetical the program beside it contradicted.

    **AND THAT WAS THE THIRD APPEARANCE OF ONE MISTAKE IN THIS MODULE:
    something cheap standing in for an exactness claim.**  First
    ulp-stability of the INPUT standing in for *"this violation is not a
    rounding artefact"*; then a fall-back to that same proxy standing in
    for an exact adjudication; then this.  Each time the cost was a
    correct ``VERIFIED`` called ``UNSOUND`` through the public API in a
    handful of lines.  The general form of the repair is the one applied
    here: **a predicate that licenses an exactness claim must be computed
    from the object the claim is about**, not from something correlated
    with it.

    Every operand and every result of every equation is inspected, and
    call bodies and structured-primitive branches are recursed into, so a
    float step is visible wherever a user put it -- inside a ``jit``
    helper, a ``jax.checkpoint``, a ``cond`` arm.  The jaxpr's own
    constvars and invars are read too, so a float constant that reaches
    nothing still answers no.
    """
    for atom in (*jaxpr.constvars, *jaxpr.invars):
        if not _integral_atom(atom):
            return False
    for eqn in jaxpr.eqns:
        for atom in (*eqn.invars, *eqn.outvars):
            if not _integral_atom(atom):
                return False
        for sub in _sub_jaxprs(eqn):
            if not _integral_program(sub):
                return False
    return True


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

    --------------------------------------------------------------------
    WHY THIS WALK DOES NOT DESCEND, AND :func:`_replay` DOES
    --------------------------------------------------------------------

    **THE TWO WALKERS HAVE DIFFERENT DEPTH BEHAVIOUR ON PURPOSE, AND
    FOUR OF THIS MODULE'S FIVE DEFECTS HAVE LIVED IN THAT GAP.  THE FIFTH
    LIVES IN A DIFFERENT ONE -- between this walk and the PROGRAM -- and
    is the section after this one.**  A call
    equation goes whole to ``prim.bind`` here, while :func:`_replay`
    descends into its body (``_CALL_PRIMITIVES``).  The fourth defect was
    exactly that gap: a ``stelling_assume`` one ``jit`` deep executes and
    never reaches ``run.assumes``, so the gate reading *"admitted by every
    assume"* off that list admitted everything, and the probe raised
    *"stelling is UNSOUND at this query"* about a correct VERIFIED.

    The obvious reconciliation is to make this walk descend too.  **It was
    measured, and it is not available: descending changes the program.**
    ``prim.bind`` on a call equation compiles the WHOLE body, and XLA is
    then free to contract across the equations inside it; walking those
    equations one at a time asks jax for each primitive separately and
    gets no contraction.  Over 22 one-line ``jit`` bodies on both
    supported jax series (``a*b+c``, ``sum(x*y)``, ``dot``, a matmul,
    ``exp``, ``log1p``, ``tanh``, Kahan, ``cumsum``, ``softmax``, in
    float32 and float64) the two disagree on **5**, and the disagreement
    is not confined to the last bits: for ``a*b + c`` with ``c`` set to
    ``-fl(a*b)``, the descended walk returns exactly ``0.0`` while the
    bound call returns the rounding error of the product -- **a SIGN
    disagreement, found on 3 of 3 attempts in each of float32 and
    float64**, and a sign is what an obligation ``>= 0`` reads.

    That is disqualifying rather than merely awkward, because of the
    sentence the ``ieee`` branch of :func:`_confirm` fires on: *"the
    executed float IS the subject of the claim."*  If this walk un-jitted
    the body, the executed float would be one the user's program never
    computes, and the branch would be licensing a claim about the program
    from something else -- **an instance of the class, manufactured by the
    repair for the fourth.**

    **AND THAT IS DRIVEN, NOT ARGUED.**  The descending version was
    written out in full and run.  It does repair the four false alarms
    this batch is about -- and on this program, whose obligation the real
    program SATISFIES at every declared point, it raises::

        a0 = float32(1.9669843); b0 = float32(1.3077438)
        c0 = float32(-fl32(a0 * b0))           # minus the ROUNDED product
        a  = any_array((), "float32", (nextafter_down(a0), a0))
        assert_(jax.jit(lambda p, q, r: p * q + r)(a, b0, c0) != 0.0)

    The body computes the rounding error of a product: nonzero when the
    call is compiled, and exactly zero when its body is walked op by op.
    Executed, the two declared points give ``-9.491779e-08`` and
    ``+6.097741e-08``, so the obligation holds.  With a descending
    ``_execute`` in place, on both supported jax series::

        FALSIFICATION PROBE FIRED — stelling is UNSOUND at this query.
        ... admitted by the 'ieee-executed-float' test ...
        at a = 1.9669841527938843

    Pinned by
    ``tests/test_falsify_fire_condition.py::test_the_EXECUTED_walk_does_not_descend_into_call_bodies``,
    which drives the fixture as well as the sign disagreement and refuses
    the source change.

    --------------------------------------------------------------------
    AND NOT DESCENDING DOES NOT MAKE THIS LOOP THE USER'S PROGRAM EITHER
    --------------------------------------------------------------------

    **THE SENTENCE THAT USED TO STAND HERE WAS FALSE, AND IT WAS THE
    SENTENCE THIS WHOLE DEPTH POLICY WAS JUSTIFIED BY.**  It read: *"what
    this loop reproduces today is what the user's own code does: their
    un-jitted top level op by op, their ``jit`` compiled whole."*  jax
    does not divide the two that way.  ``jnp.mean`` is a compiled REGION
    on the eager path and ``jax.make_jaxpr`` INLINES it, so the top level
    of the traced jaxpr carries a bare ``reduce_sum ; div`` with no
    ``jit`` equation anywhere -- and this loop then walks op by op exactly
    what the caller's own call compiles whole::

        jnp.mean(stack([x, 2x, 3x])) eager   2.620454411821526
        the same jaxpr, op by op             2.6204544118215263

    one ulp apart, bit-identical on both supported jax series.  Over 200
    float64 points drawn uniformly from ``[0.5, 4.0]``, each reduced over
    ``stack([t, 2t, 3t, t/2, 5t/4])`` with NO user-written ``jit``
    anywhere, the two routes disagree on **70 of 200** for ``jnp.mean``
    and **72 of 200** for ``jnp.average``; every other wrapper surveyed --
    ``sum``, ``prod``, ``max``, ``var``, ``std``, ``linalg.norm``,
    ``median``, ``cumsum``, ``dot``, a hand-rolled ``logsumexp``, a
    ``softmax`` -- agrees on 200 of 200, either because it emits a
    top-level ``jit`` equation this loop hands whole to ``bind`` or
    because XLA has nothing to contract across.

    Driven to a firing in four lines, on both series::

        X0    = 1.3102272059107631
        mean3 = lambda x: jnp.mean(jnp.stack([x, x * 2.0, x * 3.0]))
        C     = float(mean3(jnp.asarray(X0, "float64")))  # its OWN value
        x     = any_array((), "float64", (X0, X0))
        assert_(mean3(x) <= C)
        # BEFORE: FIRED, margin -4.44e-16, admitted by 'ieee-executed-float'

    The obligation holds at its one declared point, eagerly and under
    ``jax.jit``, because ``C`` is the value the program itself computes
    there.  The probe called it UNSOUND because it evaluated the program
    at the TRACE's granularity rather than at the program's, and
    ``ieee-executed-float`` licenses *"the executed float IS the subject
    of the claim"* -- which that float was not.  **That is the FIFTH
    instance of the class, it needed no repair to manufacture it, and it
    lived inside the sentence written to justify this function's central
    decision.**

    **SO THE GRANULARITY IS MEASURED TOO, ONE AXIS OVER FROM THE DEPTH.**
    :func:`_whole_program_route` runs the SAME traced program at the SAME
    point as ONE compiled region, and an executed violation whose truth
    value differs between the two routes declines
    (``executed-float-depends-on-granularity``, gated in
    ``probe.run_one`` beside the other readings of the executed run).
    Measured: the ``mean3`` fixture disagrees and now declines, and the
    ``a * b + c`` fixture above agrees at both of its declared points and
    is untouched.  **And the reach cost on ordinary code is zero**: over
    31 one-line ``jnp`` programs -- ``sum``, ``mean``, ``average``,
    ``var``, ``median``, ``dot``, a matmul, ``linalg.norm``, ``where``,
    ``maximum``, ``clip``, ``mse``, ``softmax``, a hand-rolled
    ``logsumexp``, ``tanh``, ``sqrt``, ``exp``, ``log1p``, ``power``,
    ``/``, ``cumsum``, ``sort``, ``reshape``/``transpose``,
    ``concatenate``, ``stack``, ``at[].set``, ``segment_sum``, a lerp, a
    running variance, the Kahan shape and one hand-written ``jit`` -- the
    firing counts are IDENTICAL with and without this guard: 31 of 31
    under ``ieee`` and 17 of 31 under ``real``, adjudicator for
    adjudicator.  What moves between the two routes is a HAIRLINE
    violation, and a violation whose sign the compiler decides is not a
    fact about the program.  A reading that changes when you change how
    you take it is not a reading of the program.

    **SO THE WALKERS ARE RECONCILED BY MEASUREMENT AND NOT BY MERGING.**
    Each one's reading is checked against a census taken at EVERY depth
    (:func:`_declaration_totals`) before it may license anything, and a
    reading that comes up short declines: ``run.assumes`` here against
    ``census.assumes_in_program`` in ``probe``, the replay's own assume
    list against the same number in :func:`_confirm`.  Neither walker is
    trusted to be complete because of how it is written.  See
    :data:`_READINGS` for the same rule applied to every other quantity.
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
        #
        # AND `ProbeInvariantViolated` IS NOT SUCH INFORMATION, which is
        # why it is a `BaseException` and passes this clause untouched.
        # A broken invariant is a statement about this MODULE, and
        # recording it in `run.raised` would file it under
        # `program-raised` -- a decline reason that names the USER's
        # program for a defect in this one.  See that class's docstring.
        run.raised = f"{type(exc).__name__}: {exc}"
    return run


def _whole_program_route(census):
    """The same traced program, at the same point, as ONE compiled region.

    **THE SECOND ROUTE, AND WHY A SECOND ROUTE IS NEEDED AT ALL.**
    :func:`_execute` hands every equation to ``Primitive.bind`` one at a
    time, so each one is compiled and run on its own and XLA never sees
    two of them together.  That is a granularity, and it is the TRACE's
    granularity rather than the program's: ``jax.make_jaxpr`` INLINES the
    ``jit`` that ``jnp.mean`` is built out of, so the top level of the
    traced jaxpr carries a bare ``reduce_sum ; div`` for a call the user's
    own program compiles whole, and the two give different floats.  The
    argument is in :func:`_execute`, with the measurement.

    This function is the other end of that axis: it interprets the SAME
    jaxpr under a single :func:`jax.jit`, so every equation is staged into
    one XLA module and contracts as far as XLA will contract it.  It
    returns ``(assumes, asserts)`` -- the two lists the fire condition
    reads -- and nothing else, because those are the quantities whose
    granularity-dependence would license a firing.

    **IT IS BUILT ONCE PER PROBE AND CALLED MANY TIMES**, so jax's own jit
    cache keys on one function object and compiles one module rather than
    one per point.

    **IT CANNOT ADMIT ANYTHING.**  It is consulted only where
    :func:`_execute` has already found a violation, and its only possible
    effect is to turn that violation into a decline.  That is what keeps
    it from being the ulp proxy in another spelling: the proxy re-executed
    at a NEIGHBOURING point and could ADMIT; this runs at the SAME point
    and can only DECLINE.
    """
    jaxpr = census.closed.jaxpr

    def staged(*vals):
        env: dict = {}

        def read(atom):
            if isinstance(atom, jex_core.Literal):
                return atom.val
            return env[atom]

        for v, c in zip(jaxpr.constvars, census.closed.consts):
            env[v] = c
        assumes: list = []
        asserts: list = []
        decl_i = 0
        for eqn in jaxpr.eqns:
            prim = eqn.primitive
            name = prim.name
            if prim is _any_p or name == "stelling_any":
                env[eqn.outvars[0]] = vals[decl_i]
                decl_i += 1
                continue
            invals = [read(a) for a in eqn.invars]
            if prim is _assume_p or name == "stelling_assume":
                assumes.append(invals[0])
                env[eqn.outvars[0]] = invals[0]
                continue
            if prim is _nonvacuity_p or name == "stelling_nonvacuity":
                env[eqn.outvars[0]] = invals[0]
                continue
            if prim is _assert_p or name == "stelling_assert":
                asserts.append(invals[0])
                env[eqn.outvars[0]] = invals[0]
                continue
            out = prim.bind(*invals, **eqn.params)
            if prim.multiple_results:
                for var, o in zip(eqn.outvars, out):
                    env[var] = o
            else:
                env[eqn.outvars[0]] = out
        return assumes, asserts

    return jax.jit(staged)


def _granularity_stable(route, point, run, k):
    """Did the two routes read this violation the same way?

    ``True`` when the whole-program route agrees with the executed run on
    the attacked obligation AND on every assume; ``False`` when they
    disagree; ``None`` when the second route could not be run at all, in
    which case nothing has been shown either way and the caller declines
    on THAT.

    The assumes are compared as well as the obligation because both
    license the firing: the gate in ``probe.run_one`` reads *"admitted by
    every assume"* off the executed run, and an assume whose truth moves
    with the compilation granularity is no more a reading of the program
    than an obligation whose truth does.
    """
    try:
        assumes, asserts = route(*point)
    except Exception:  # noqa: BLE001 - an unavailable route, never a firing
        # `ProbeInvariantViolated` is a `BaseException` and is NOT
        # absorbed here: "the second route could not be run" is a fact
        # about jax and this program, and a broken invariant of this
        # module is not that fact wearing a different name.
        return None
    if len(asserts) != len(run.asserts) or len(assumes) != len(run.assumes):
        # The two routes walk the same equation list, so this cannot
        # diverge today; a reading that came up short would be no reading,
        # and no reading is not agreement.
        return None
    if bool(np.all(np.asarray(asserts[k]))) != bool(np.all(run.asserts[k])):
        return False
    for a, b in zip(assumes, run.assumes):
        if bool(np.all(np.asarray(a))) != bool(np.all(b)):
            return False
    return True


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
            # ONE STEP IN THE DECLARATION'S OWN FORMAT.  A float64 step off
            # a `float16` endpoint quantises straight back onto the
            # endpoint and is deduplicated away, so "just inside the
            # corner" -- the point this pair exists to reach -- would be
            # sampled at float64 and at no other width.
            cands += [_step_in(lo, hi, dt), _step_in(hi, lo, dt)]
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


def _step_in(frm, towards, dt):
    """One representable step of ``dt`` from ``frm`` towards ``towards``."""
    a = np.asarray(frm, dtype=dt)
    b = np.asarray(towards, dtype=dt)
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.nextafter(a, b))


def _clean(cands, lo, hi, integral, dt=None):
    """De-duplicate, drop what the window cannot hold, keep the order stable.

    **THE RANGE TEST IS APPLIED TO THE VALUE THE DTYPE WILL HOLD**, not to
    the Python float the generator proposed, because those are different
    numbers at every format narrower than ``float64`` and it is the first
    one that reaches the program.  A ``uniform`` draw of 0.2999999 in a
    ``float32`` declaration of ``(-0.3, 0.3)`` becomes 0.30000001192...
    the moment ``np.full`` casts it, and a point outside the declared box
    is a point the fire condition must never see -- so it is quantised
    here and re-tested, rather than built and then thrown away by
    :func:`_admissible` further downstream.  Deduplication happens after
    the quantisation for the same reason: two candidates a float64 step
    apart are ONE ``float16`` value and running the second is running the
    first again.

    ``dt`` is kept as a second, cheap guard: :func:`_window` already
    intersects the declared interval with the dtype's own finite range, so
    an in-window candidate is representable by construction, and this
    catches a future fill generator that stops respecting the window
    before `np.full` turns the mistake into a warning the caller sees.

    **"IN-WINDOW IMPLIES IN-BOX BY CONSTRUCTION" IS A CLAIM ABOUT EVERY
    BRANCH OF** :func:`_window` **AND ONE BRANCH USED TO BREAK IT.**  The
    ``bool`` branch returned ``(0, 1)`` regardless of what was declared,
    so an in-window candidate for ``any_array((), "bool", (0.0, 0.0))``
    was OUT of the box and :func:`_admissible` -- not this guard -- was
    what caught it.  Fixed at the branch: the window is the declaration
    intersected with ``{0, 1}``, like every other branch.
    """
    out = []
    seen = set()
    for c in cands:
        try:
            v = int(round(c)) if integral else float(c)
        except (TypeError, ValueError, OverflowError):
            continue
        if dt is not None and not integral:
            q = _quantise(v, dt)
            if q is None:
                continue
            v = q
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


# THE MODULE THIS ONE MAY NOT READ, SPELLED AS A STRING BECAUSE THAT IS
# THE WHOLE POINT.  `probe`'s door has to be able to say *"you handed me
# the transcription"*, and the only object that can carry that name is a
# name: importing `stelling.ir` to write `isinstance` would break the
# independence rule the guard exists to protect.  Matched against
# `type(x).__module__` for every class in the MRO, so a subclass of
# `stelling.ir.ClosedJaxpr` is caught as well as the class itself.
#
# It is a MODULE and not a single qualified name deliberately: no object
# from `stelling.ir` is ever the right argument here, and `ClosedJaxpr` is
# merely the one that gets all the way into `_read` before failing.
_FORBIDDEN_MODULE = "stelling.ir"


def _is_transcription(obj):
    """``"stelling.ir.ClosedJaxpr"`` when ``obj`` is one, else ``None``.

    Reads ``__module__``/``__qualname__`` off the MRO rather than asking
    :func:`isinstance`, because asking would mean importing the module
    this one is forbidden to import.  A name comparison is weaker than an
    ``isinstance`` in general -- a second module of the same name would
    fool it -- and it is exactly strong enough here: the failure it is
    guarding against is a variable slip inside this package, between two
    objects produced by one line of ``preconditions._pipeline``.
    """
    for klass in type(obj).__mro__:
        if getattr(klass, "__module__", None) == _FORBIDDEN_MODULE:
            return f"{_FORBIDDEN_MODULE}.{klass.__qualname__}"
    return None


def probe(
    closed,
    *,
    statuses,
    semantics="real",
    budget=DEFAULT_BUDGET,
    seed=0,
    strategies=STRATEGIES,
    assumptions=(),
):
    """Try to falsify a discharge by executing the real program.

    ``closed`` is **jax's own** ``ClosedJaxpr`` for the program the
    analysis judged — the object ``jax.make_jaxpr(harness)()`` returned,
    handed over by :func:`stelling._jax_compat.trace_with_jaxpr`.  Not the
    harness, and not the transcribed :class:`stelling.ir.ClosedJaxpr`.

    **THE PROBE DOES NOT TRACE, AND THAT IS A SOUNDNESS PROPERTY RATHER
    THAN A COST SAVING.**  It used to be handed the harness and to call
    ``jax.make_jaxpr(harness)()`` for itself.  Whether that re-ran the
    user's body was decided by jax's trace memo — which is not a memo this
    project owns, and which ``preconditions._pipeline`` DELIBERATELY
    defeats when the overflow tripwire is armed (it evicts jax's caches
    and traces through a fresh closure, so that the trace happens under
    the instrument).  Measured on this tree, harness-body invocations per
    ``check()``: **1 everywhere, 2 with the tripwire armed and
    ``falsify="sample"``.**  Nothing then compared the probe's program
    with the analysis's — every totality guard in this module compares the
    probe's reading against the probe's OWN second trace, and the only
    cross-check was ``len(statuses) == len(census.assert_positions)``, a
    count.  Driven through the public API with no mutation, both
    directions: a harness returning ``assert_(x >= 0.0)`` on its first
    call and ``assert_(x <= 1.0)`` on its second was VERIFIED (correctly)
    under ``pytest`` and raised *"FALSIFICATION PROBE FIRED — stelling is
    UNSOUND at this query"* under ``pytest -p stelling.overflow``; and the
    mirror carried *"NO VIOLATION WAS FOUND"* about a declared box the
    verdict is not about.  ``check()``'s own docstring recommends both
    invocations.

    A guard comparing two programs was the fallback, and it is not what
    shipped: there is no second program to compare, because this function
    can no longer obtain one.  A caller who has only a harness traces it
    once and passes the result.

    ``statuses`` is the per-obligation status STRING sequence
    the analysis produced -- passed in rather than read off a
    :class:`stelling.verdict.Verdict`, so that this module never imports
    the module whose output it is attacking.  Only obligations whose
    status is ``"discharged"`` are attacked; everything else the analysis
    already declines to claim.

    ``assumptions`` is the verdict's stamped assumption lines, as plain
    strings and for the same reason ``statuses`` is.  A verdict that rests
    on a declaration stelling explicitly does not check — the caller's
    ``libm_budget``, stamped *"DECLARED, NOT VERIFIED"* — is not
    stelling's own claim, so a firing on one may not be reported as
    stelling's own defect.  See :func:`_fire` and
    :data:`DECLARED_NOT_VERIFIED`.

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
    transcription = _is_transcription(closed)
    if transcription is not None:
        # THE ONE WRONG OBJECT THE STRUCTURAL GUARD BELOW CANNOT SEE, AND
        # THE ONE WHOSE COST IS SILENCE RATHER THAN AN ERROR.
        # `stelling.ir.ClosedJaxpr` is not callable and DOES carry a
        # `.jaxpr`, so it satisfies every structural test this door can
        # ask -- its `.jaxpr` even carries `.eqns` -- and it then dies
        # inside `_read` with `AttributeError: 'str' object has no
        # attribute 'name'`, which `probe` converts into the ordinary
        # whole-probe DECLINE *"the probe could not read the traced
        # program"*.
        #
        # WHY THAT IS WORTH A NAMED GUARD RATHER THAN A BETTER MESSAGE.
        # `preconditions._pipeline` holds BOTH objects from one trace --
        # `cj, jaxpr = trace_with_jaxpr(...)` -- and passes `jaxpr`.  A
        # one-token slip to `cj` there would do two things at once: it
        # would feed this module the TRANSCRIPTION, laundering the
        # independence rule that says the probe may not consume the object
        # it is checking; and it would silently turn the probe OFF for
        # every VERIFIED in the corpus, because every call would decline
        # with *"DECLINED, nothing was executed"* and a stamp line that
        # reads like a probe that ran.  Same call, `FIRED` on jax's object
        # and silent on the transcription.  A probe that quietly stops
        # probing is this project's recurring defect in its purest form,
        # so the door says it.
        #
        # BY NAME, AND WITHOUT IMPORTING `stelling.ir`.  The independence
        # rule forbids the import (see this module's docstring), and an
        # `isinstance` check would need it.  The type is therefore
        # identified by its module and qualified name, walking the MRO so
        # a subclass is caught too -- which is also the only test that
        # separates the two objects at all, since structurally they agree.
        raise TypeError(
            f"probe() takes jax's own ClosedJaxpr for the program the "
            f"analysis judged, and was handed stelling's own "
            f"TRANSCRIPTION of it ({transcription}). Those are different "
            f"objects and this one is the object the probe exists to "
            f"check: it may not read `stelling.ir`, and it cannot execute "
            f"a transcription. Pass the SECOND member of "
            f"`stelling._jax_compat.trace_with_jaxpr(harness)` (or "
            f"`jax.make_jaxpr(harness)()` when there is no analysis to "
            f"agree with); the first member is what you passed"
        )
    if callable(closed) or not hasattr(closed, "jaxpr"):
        # SAID AT THE DOOR, because the thing that used to be passed here
        # is a harness and passing one now would fail deep inside `_read`
        # with a message about a traced program.  The whole point of the
        # parameter change is that the probe must be handed the program the
        # analysis judged; a caller who reaches for the old spelling is
        # told exactly how to produce it.
        raise TypeError(
            "probe() takes jax's own ClosedJaxpr for the program the "
            "analysis judged, not a harness: the probe must not trace a "
            "second program of its own. Pass "
            "`stelling._jax_compat.trace_with_jaxpr(harness)[1]` (or "
            "`jax.make_jaxpr(harness)()` when there is no analysis to "
            f"agree with), got {type(closed).__name__}"
        )
    skips = _Counter()
    spoints = _Counter()
    shits = _Counter()
    rng = random.Random(seed)

    # ONE STAGE, WHERE THERE USED TO BE TWO.  The trace stage is gone --
    # the program arrives already traced -- so the decline that named it
    # ("the harness could not be traced") is gone with it: a trace failure
    # now happens in the caller's own frame, where the caller's harness is,
    # instead of being reported by an instrument that ran afterwards.  The
    # READ stage keeps its own decline, which is the one that mattered:
    # the two used to share an `except` and a read defect was reported as
    # a trace failure, sending a reader to their harness for a defect in
    # this file.
    try:
        census = _read(closed)
    # `ProbeInvariantViolated` is deliberately outside `Exception` and so
    # is NOT converted into the decline below.  Driven on the tree that
    # shipped it as an `AssertionError`: a raise inside `_read` came back
    # as the ordinary whole-probe *"the probe could not read the traced
    # program: ProbeInvariantViolated: ..."*, i.e. a broken invariant
    # rendered in the same sentence as a malformed program.
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
    if len(census.assert_positions) != census.obligations_in_program:
        # AND THE PAIRING GUARD ABOVE IS NOT THIS GUARD, though today they
        # decline together on every program either has been shown.  That
        # one compares the probe's reading with the ANALYSIS's count and so
        # depends on the analysis descending into call bodies -- measured,
        # it does: an `assert_` one `jit` deep makes `check()` report two
        # obligations against one top-level assert.  This one compares the
        # probe's reading with the PROGRAM and holds whatever the analysis
        # does with a nested obligation later.  The distinction is the
        # whole subject of this batch: a guard on a claim about the program
        # is computed from the program.
        return ProbeReport(
            declined=(
                f"the traced program states {census.obligations_in_program} "
                f"obligation(s) and {len(census.assert_positions)} of them "
                f"are at the top level, where this probe reads them; it "
                f"cannot say which obligation an index means"
            )
        )
    if not census.declarations:
        return ProbeReport(declined="the harness declares no inputs to vary")
    if len(census.declarations) != census.declarations_in_program:
        # A `stelling_any` inside a call body is a declaration the probe
        # has no sampled value to substitute for, so `_execute` hands the
        # equation to `Primitive.bind` and `stelling_any` -- which has no
        # implementation, by design -- raises.  Every point then skipped
        # under `program-raised`, a reason that names the USER's program
        # for a limit of this walk.  Declining here says the true thing
        # instead, and says it once rather than once per point.
        return ProbeReport(
            declarations=census.declarations,
            declined=(
                f"the harness declares {census.declarations_in_program} "
                f"input(s) and {len(census.declarations)} of them are at "
                f"the top level, where this probe substitutes sampled "
                f"values; it cannot vary the rest"
            ),
        )

    windows = []
    for d in census.declarations:
        w, why = _window(d)
        if w is None:
            return ProbeReport(
                declarations=census.declarations,
                declined=f"declaration #{d.position} is not sampleable: {why}",
            )
        windows.append(w)

    # AND A DECLARATION WHOSE DTYPE JAX WILL NARROW IS NOT SAMPLEABLE
    # EITHER, WHICH IS THE SAME KIND OF FACT AND WAS NOT BEING SAID.
    #
    # `run_one` builds each point as a numpy array of the declaration's
    # dtype and then converts it with `jax.numpy.asarray` for `_execute`,
    # while `_confirm` and `_replay` are handed the UN-converted numpy
    # array.  Under `jax_enable_x64=0` that conversion is a NARROWING one
    # -- float64 arrives as float32, int64 as int32 -- so the executed run
    # and the exact test would be about different programs.  There is an
    # assertion at that conversion, and it is the last line of defence
    # rather than the policy: what it used to rest on was a comment
    # claiming *"nothing reaches it today ... an int64 box is admitted
    # only where it provably fits int32"*, which is INVERTED.  Fitting
    # int32 is the condition under which the box is admitted, the probe
    # runs and the assertion fires.  Driven through the public door at
    # `2bd7bc8` and at its parent, with `jax_enable_x64=0`::
    #
    #     check(h, vacuity_mode="inputs-only", falsify="sample")
    #       float64 declaration -> UNKNOWN, no probe note
    #       int64   declaration -> ProbeInvariantViolated out of `check()`
    #       float32 declaration -> VERIFIED, probe note present
    #
    # The reach is older than the class's move to `BaseException`; what
    # that move changed is the CONTAINMENT, from a batch caller's
    # `except Exception` recording one node and carrying on to the run
    # dying.  Neither disposition is right, because neither is what
    # happened: nothing about stelling's invariants broke.  The caller
    # turned x64 off and declared a 64-bit box, and THIS PROBE CANNOT
    # SAMPLE THAT -- which is a sentence this module already knows how to
    # say, once, in the report, for an unbounded declaration and for a
    # dtype it cannot construct.  It says it here too.  The verdict the
    # analysis reached is not touched: a probe that cannot run says
    # nothing, and saying nothing must not cost the caller their verdict.
    #
    # The assertion in `run_one` stays, and its premise is now a decline
    # three lines from it in THIS module rather than a refusal in another
    # one.
    #
    # **AND THE DECLINE IS ON THE DTYPE NAME RATHER THAN ON THE BOX, WHICH
    # WAS PROPOSED AND IS REFUSED.**  The proposal: decline only when the
    # declared WINDOW is not exactly representable in the narrowed dtype,
    # since an `int64` box of `(0, 9)` plainly is, and admit the rest.
    # The reach it would buy is real and it is measured: with this decline
    # and the `run_one` assertion both removed at `jax_enable_x64=0`, a
    # 51-program corpus fires on **52 of 95 probe calls** where the shipped
    # code fires on **9** -- and of those 52, **33 are adjudicated
    # `ieee-executed-float`**, which is a float32 EXECUTION admitting a
    # firing about a float64-DECLARED program with no exact test behind
    # it.  Declining those is right whatever the box says.  The other 19
    # (16 `exact-replay-refutes-over-the-rationals`, 3
    # `exact-integer-arithmetic`) are sound as claims about R, and they
    # are the reach on the table.  It is refused on two measurements,
    # both taken here:
    #
    # * **A REPRESENTABLE BOX DOES NOT CONTAIN ONLY REPRESENTABLE POINTS,
    #   FOR A FLOAT DTYPE.**  `float64 (0.0, 2.0)` has float32-exact
    #   endpoints, and of 1,000 uniform `float64` points drawn from it,
    #   **0 are float32-exact**; three `nextafter` steps from `1.0` --
    #   which is what the `ulp` phase does by construction -- reach
    #   `1.0000000000000007`, which is not.  So the box test admits
    #   exactly the case it was meant to exclude: a witness reported at a
    #   value jax never executed.  The condition that would actually mean
    #   what the proposal wants is per-POINT, not per-window, and it is a
    #   different and larger change.
    # * **AND FOR AN INTEGER DTYPE, WHERE THE BOX TEST *DOES* IMPLY THE
    #   POINT TEST, THE PROGRAM IS STILL NOT THE DECLARED ONE.**  Every
    #   integer in `(0, 9)` is int32-exact, but the executed program then
    #   computes in int32 throughout: `2**30 * 3` is `3221225472` in
    #   int64 and `-1073741824` in int32.  The probe's integer branch
    #   admits a firing under *"exact integer arithmetic: no rounding
    #   involved"*, and it would be admitting one about an overflow that
    #   exists only in the narrowed width -- reporting a violation of a
    #   program the caller did not declare, which is the false-alarm shape
    #   this whole module is built to avoid.
    #
    # The dtype name is a SUFFICIENT condition for *"the executed program
    # is the declared program"*, cheap and provable in one line.  The box
    # test is neither, and what it buys is reach.  Reach is the thing this
    # module is allowed to lose.
    for d in census.declarations:
        narrowed = _dtype_after_jax(d.dtype)
        if narrowed != d.dtype:
            return ProbeReport(
                declarations=census.declarations,
                declined=(
                    f"declaration #{d.position} is not sampleable: "
                    f"dtype-narrowed-by-jax ({d.dtype} becomes {narrowed} "
                    f"when a point is handed to jax, which jax does under "
                    f"jax_enable_x64=0). The executed run and the exact "
                    f"test would be about different programs, so this "
                    f"probe declines rather than reporting either"
                ),
            )

    report_kw = dict(
        declarations=census.declarations,
        obligations=len(targets),
        semantics=semantics,
        assumes_in_program=census.assumes_in_program,
        assumptions=tuple(assumptions),
    )

    # THE SECOND ROUTE, BUILT ONCE.  `jax.jit` here only wraps a closure;
    # nothing is traced or compiled until it is first called, which is
    # only where an executed violation has to be checked for granularity
    # dependence.  Building it once rather than per point is what lets
    # jax's own jit cache compile ONE module for the whole probe.
    route = _whole_program_route(census)

    # THE ASSUME CONFIRMATION'S BUDGET, AND WHY IT IS ONE FOR THE WHOLE
    # PROBE.  Re-reading every admissible point's assumes over ℚ is a
    # `_replay` per point, and a `_replay` costs a fresh
    # `REPLAY_SECONDS_BUDGET` when it is given one.  Per point that is
    # 256 x 5.0s = twenty-one minutes of worst case for a probe that finds
    # nothing, on a program whose values grow (the four-fold-per-squaring
    # cascade measured at `REPLAY_ELEMENT_BUDGET`), where today the same
    # run costs a tenth of a second.  A bound that the common case never
    # reaches and the pathological case cannot survive is not a bound.
    #
    # So all of the confirmations share ONE guard: the whole confirmation
    # pass costs at most `REPLAY_SECONDS_BUDGET`, and past it every
    # remaining admissible point is counted UNCONFIRMED and said to be
    # unconfirmed.
    #
    # **AND THE ORDINARY CASE IS NOT "NOWHERE NEAR IT", WHICH IS WHAT THIS
    # COMMENT USED TO SAY.** The per-point figures it quoted re-derive --
    # a scalar `assume(y*0.1*10.0 <= y)` confirms in 0.025 ms and a
    # `(1000,)` float64 one in 4.9 ms, measured on this tree, jax 0.11.0,
    # x64 on -- but the budget is not per point, and a probe confirms up
    # to `budget` of them. What one confirmation costs is linear in the
    # DECLARED ARRAY, and 5.0 seconds buys this many of them:
    #
    #     declaration     ms/confirmation   5.0s buys   admissible  unconfirmed
    #     -----------     ---------------   ---------   ----------  -----------
    #     ()                        0.025     196,437            8            0
    #     (1000,)                   4.935       1,013            8            0
    #     (10000,)                 42.971         116           11            3
    #     (20000,)                 94.977          52           57           55
    #     (60000,)                287.602          17           91           90
    #
    # The last two columns are a real `probe()` at the default budget of
    # 256 on `assume(all(y*0.1*10.0 <= y))` with a trivially-true assert.
    # **A `(60000,)` float64 declaration with one assume saturates the
    # backstop and reports 90 of its 91 admissible points unconfirmed**,
    # and there is nothing exotic about it: it is one array and one
    # elementwise assume.
    #
    # **EVERY COLUMN OF THAT TABLE IS MACHINE-DEPENDENT, INCLUDING THE
    # LAST TWO, AND IT USED TO READ AS FIVE MEASURED CONSTANTS.**  The
    # first two columns are obviously a clock.  The last two are the same
    # clock one step removed: an UNREAD gate counts its point ADMISSIBLE
    # and UNCONFIRMED (`exact_assumes is None` below), while a gate that
    # was read and said NO does not count the point at all -- so buying
    # fewer confirmations INFLATES both columns.  Driven on the `(1000,)`
    # row by shrinking the budget, which is what a slower machine does:
    #
    #     REPLAY_SECONDS_BUDGET   admissible   unconfirmed
    #     ---------------------   ----------   -----------
    #     5.0 (default)                    8             0
    #     0.5                              8             0
    #     0.05                            94            93
    #     0.005                          107           106
    #
    # and measured across machines on the `(10000,)` row, the same probe
    # at the default budget reads 14/6, 11/3 and 10/2.  The shape of the
    # table is the result -- saturation arrives on ordinary array sizes
    # and costs evidence rather than soundness -- and the individual cells
    # are one machine's.  This is the same fact as stability item 5 in the
    # module docstring, THE CLOCK IS PART OF THE FIRE CONDITION, seen from
    # the reporting side instead of the firing side.
    #
    # THAT IS THE BUDGET WORKING, NOT FAILING, and it is written down
    # because a reader who believed the old sentence would read
    # `points_admissible_unconfirmed = 90` as a defect rather than as the
    # bound doing its job. What the saturation costs is EVIDENCE, never
    # soundness: an unconfirmed point is counted, named in the stamp line
    # (`_admissible_clause`), and still attacked -- under `real`,
    # `_confirm` re-reads the assumes on its own fresh budget before
    # anything may fire; under `ieee` it returns on the executed float
    # before any replay, and the executed float reading of the assumes is
    # what an `ieee` claim is about, which is why this budget is zero
    # there. The
    # alternative, one full backstop per point, is 256 x 5.0s = twenty-one
    # minutes for a probe that finds nothing, which is why it is shared.
    #
    # IT IS A BUDGET OF CONFIRMATION WORK, NOT A DEADLINE FROM THE START
    # OF THE PROBE.  A single shared `_Guard` would have been simpler and
    # would have meant something else: its clock would run while `_execute`
    # and the sampler worked, so a long probe would report its later points
    # unconfirmed for time it never spent confirming anything.  What is
    # shared is the REMAINING SECONDS; each confirmation gets a guard whose
    # deadline is now plus that, and gives back what it did not use.
    #
    # `None` under `ieee`: there the executed float reading of the assumes
    # is the reading the verdict's own claim is about, so there is nothing
    # to confirm and nothing to charge.
    assume_seconds = [REPLAY_SECONDS_BUDGET if semantics != "ieee" else 0.0]

    built = 0
    executed = 0
    admissible = 0
    unconfirmed = 0
    declined_points = 0
    violations = 0
    adjudged = _Counter()
    abstained = _Counter()
    best: list = []  # (margin, point) of the least-slack admissible points

    def assumes_over_the_rationals(point):
        """Is this point inside the assumed region over ℚ?  ``None`` = unread.

        The exact half of the admissibility gate.  ``True`` when the
        rational replay reached every assume the program states and all of
        them hold; ``False`` when it reached every one and some do not;
        ``None`` when no exact reading could be taken at all -- the replay
        abstained, or the shared budget above is spent.

        A SHORT READING IS NOT A SATISFIED ASSUME AND NOT AN UNSATISFIED
        ONE, which is the rule the executed gate above already applies to
        its own list: `_replay` descends call bodies where `_execute` binds
        them, so its assume list is checked against
        `census.assumes_in_program` -- counted at every depth -- before it
        may say anything.
        """
        if census.assumes_in_program <= 0 or assume_seconds[0] <= 0.0:
            return None
        guard = _Guard(assume_seconds[0])
        started = time.monotonic()
        try:
            assumes, _ = _replay(
                census, point, guard=guard, assumes_only=True
            )
        # `ProbeInvariantViolated` passes: it is a `BaseException`.  An
        # unread assume and a broken invariant are different answers --
        # the first counts the point UNCONFIRMED and says so, the second
        # means no count taken here is a count of anything -- and on the
        # tree that shipped it as an `AssertionError` the second was
        # driven and came back as the first.
        except Exception:  # noqa: BLE001 - an unread assume, never a firing
            return None
        finally:
            assume_seconds[0] -= time.monotonic() - started
        if len(assumes) != census.assumes_in_program:
            return None
        return all(assumes)

    def run_one(strategy, point):
        """Execute one point.  Returns ``(falsification, margin)``.

        ``margin`` is the least slack this point left across the attacked
        obligations, or ``None`` when the point was unusable or the
        obligations do not compare.  It is returned rather than only
        recorded because the ``tight`` search STEERS on it.
        """
        nonlocal built, executed, admissible, unconfirmed
        nonlocal declined_points, violations
        built += 1
        spoints.add(strategy)
        if not all(
            _admissible(d, a) for d, a in zip(census.declarations, point)
        ):
            skips.add("point-outside-declaration")
            return None, None
        jpoint = [jax.numpy.asarray(a) for a in point]
        # THE POINT `_execute` RUNS MUST BE THE POINT `_confirm` JUDGES.
        # `jnp.asarray` is a CONVERSION and under `jax_enable_x64=0` it is
        # a NARROWING one: a float64 numpy array arrives as float32, an
        # int64 one as int32, silently -- while `_confirm` and `_replay`
        # are handed the un-narrowed numpy `point`.  The executed run
        # would then be a run of a different program from the one the
        # exact test admits, which is the class this module keeps finding.
        #
        # **THE JUSTIFICATION THAT SHIPPED HERE WAS INVERTED AND THE RAISE
        # WAS LIVE OUT OF THE PUBLIC DOOR.**  It read *"nothing reaches it
        # today ... an int64 box is admitted only where it provably fits
        # int32"* -- but fitting int32 is exactly the condition under which
        # the box IS admitted, the probe runs and this check fires.  Driven
        # at `2bd7bc8` and at its parent with `jax_enable_x64=0`, an
        # `int64` declaration raised `ProbeInvariantViolated` straight out
        # of `check(..., falsify="sample")`.
        #
        # What reaches it now is nothing, and the reason is a DECLINE IN
        # THIS FUNCTION: `probe` refuses any declaration whose dtype does
        # not survive `jax.numpy.asarray` under the live config
        # (`dtype-narrowed-by-jax`) before a point is built.  That is the
        # standard this comment set for itself and did not meet -- *"an
        # invariant that holds because of a decline somewhere else is an
        # invariant one line of that other module can remove"* -- and the
        # decline it used to lean on (`propagate` refusing the
        # `convert_element_type` a truncated float64 declaration produces,
        # so no VERIFIED exists to probe) is corroboration and no longer
        # the argument.  This stays ASSERTED because a probe that reported
        # anything at all about two different programs would be reporting
        # it as a fact about the caller's verdict.
        for a, j in zip(point, jpoint):
            if np.dtype(np.asarray(a).dtype) != np.dtype(j.dtype):
                raise ProbeInvariantViolated(
                    f"the probe's sampled point did not survive conversion "
                    f"to jax: {np.asarray(a).dtype} became {j.dtype}, which "
                    f"jax does under jax_enable_x64=0. The executed run and "
                    f"the exact test would then be about DIFFERENT PROGRAMS "
                    f"— `_execute` runs the narrowed point and `_confirm` "
                    f"and `_replay` are handed the un-narrowed one — so no "
                    f"reading taken here would be a reading of one program. "
                    f"This is not a falsification and not a verdict; it is "
                    f"this probe refusing to report either."
                )
        run = _execute(census, jpoint)
        executed += 1
        if run.raised is not None:
            skips.add("program-raised")
            return None, None
        if len(run.asserts) != len(statuses):
            skips.add("obligation-count-changed")
            return None, None
        if len(run.assumes) != census.assumes_in_program:
            # THE GATE BELOW READS *"admitted by every assume"* OFF A LIST
            # THAT CAN BE SHORT.  `_execute` walks the top level and hands
            # a call equation whole to `Primitive.bind`, so a
            # `stelling_assume` inside a `jit` or `remat2` body EXECUTES
            # and never reaches `run.assumes` -- and `if run.assumes and
            # ...` on an empty list admits everything.  `propagate` does
            # narrow on that assume, which is why the VERIFIED exists, so
            # the probe was attacking points the analysis never claimed
            # anything about and raising "stelling is UNSOUND" about a
            # correct verdict, in five lines through the public API:
            #
            #     x = any_array((), "int32", (0, 10))
            #     y = jax.jit(lambda a: (assume(a >= 9), a)[1])(x)
            #     assert_(y >= 9)          # VERIFIED, and correct
            #
            # A short reading is not a satisfied assume and it is not an
            # unsatisfied one; it is no reading, and the point declines.
            skips.add("assume-not-fully-executed")
            return None, None
        if run.assumes and not all(bool(np.all(a)) for a in run.assumes):
            skips.add("assume-unsatisfied")
            return None, None
        # AND THE EXACT HALF OF THE SAME GATE, TAKEN HERE RATHER THAN
        # ONLY WHERE A VIOLATION WAS FOUND.  Everything above is the
        # EXECUTED FLOAT reading of the assumes.  Under `real` semantics
        # an assume is a claim about ℝ, and the float reading of it is a
        # proxy that this repository has measured wrong on most of its
        # points: on a clean VERIFIED, `assume(y*0.1*10.0 <= y)` over
        # `float64 [0, 2]`, 47 of 55 float-admitted points are NOT in the
        # assumed region over ℚ. `_confirm` has always re-read the assumes
        # exactly -- but only at a point where the obligation had already
        # evaluated FALSE, so on a clean run the correction never ran and
        # `points_admissible` stood as a coverage figure with no exact
        # evidence behind it.
        #
        # Now it runs at the gate. Three answers and three different
        # things to do with them: outside over ℚ is a point the analysis
        # never claimed anything about, and it is not attacked and not
        # counted; inside over ℚ is counted with exact evidence; NOT READ
        # is counted and SAID (`points_admissible_unconfirmed`), never
        # silently folded into either.
        #
        # The unread point is still attacked, and that is deliberate: an
        # unreadable assume costs reach, not soundness, because under
        # `real` semantics `_confirm` re-checks the assumes on its own
        # fresh budget before anything fires.
        #
        # **THAT IS TRUE UNDER `real` AND NOT UNDER `ieee`, and the
        # sentence used to be written flat.**  Under `ieee` `_confirm`
        # returns on `ieee-executed-float` BEFORE any replay, so nothing
        # downstream re-reads the assumes -- and it does not need to: the
        # executed float reading of them IS the reading an `ieee` claim is
        # about, which is why `assume_seconds` is zero there and why every
        # `ieee` point with an assume is counted unconfirmed. What guards
        # the `ieee` return is the OTHER gate above (`run.assumes` short
        # -> `assume-not-fully-executed`), and `_confirm`'s docstring says
        # so at the return itself.
        exact_assumes = assumes_over_the_rationals(point)
        if exact_assumes is False:
            skips.add("assume-unsatisfied-over-the-rationals")
            return None, None
        admissible += 1
        if exact_assumes is None and census.assumes_in_program > 0:
            unconfirmed += 1
        point_declined = False
        point_unassumed = False
        for k in targets:
            if bool(np.all(run.asserts[k])):
                continue
            violations += 1
            if semantics == "ieee":
                # THE FIFTH INSTANCE OF THE CLASS, AND THE ONE GUARD IT
                # NEEDED.  Under `ieee` the executed float is what admits:
                # `_confirm` returns on *"the executed float IS the
                # subject of the claim"* before any exact test runs.  But
                # WHICH float `_execute` computes depends on how much of
                # the program it hands to jax at once, and it hands it one
                # equation at a time -- the TRACE's granularity, which is
                # not the program's, because `jax.make_jaxpr` inlines the
                # `jit` that `jnp.mean` is built out of.  Four lines with
                # no user-written `jit` anywhere then fire on a correct
                # VERIFIED; the reproduction and the measurement are in
                # `_execute`.
                #
                # So the granularity is MEASURED, exactly as the depth is:
                # the same program is run at the SAME point as one
                # compiled region, and a violation whose truth value moves
                # between the two routes is not a reading of the program
                # and licenses nothing.  This can only ever DECLINE.
                #
                # UNDER `real` THERE IS NOTHING HERE TO GUARD, and that is
                # an argument rather than an omission.  Neither admitting
                # test there reads an executed float: exact integer
                # arithmetic and the rational replay both re-evaluate the
                # jaxpr, and ℚ has no granularity to be at.  The executed
                # run is only the search that FINDS a candidate point, and
                # a wrong candidate costs a decline.  The assume gate is
                # the one executed reading `real` does lean on, and
                # `_confirm` already re-reads every assume over ℚ and
                # declines when the two disagree
                # (`assume-unsatisfied-over-the-rationals`).
                stable = _granularity_stable(route, jpoint, run, k)
                if stable is False:
                    skips.add("executed-float-depends-on-granularity")
                    adjudged.add("declined-executed-routes-disagree")
                    point_declined = True
                    continue
                if stable is None:
                    # NOT the same finding.  Nothing has been shown to
                    # move; the second reading could not be taken, and an
                    # unchecked granularity is not a checked one.
                    skips.add("whole-program-route-unavailable")
                    adjudged.add("declined-whole-program-route-unavailable")
                    point_declined = True
                    continue
            detail, why, how, unread = _confirm(
                census, statuses, point, k, semantics
            )
            adjudged.add(how)
            if unread is not None:
                abstained.add(unread)
            if detail is None:
                skips.add(why)
                if why == "assume-unsatisfied-over-the-rationals":
                    point_unassumed = True
                    break
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
                    operations=_obligation_operations(census, k),
                ),
                None,
            )
        if point_unassumed:
            # THE POINT WAS NEVER ADMISSIBLE AND `admissible` HAS ALREADY
            # COUNTED IT.  The gate above is the EXECUTED-FLOAT reading of
            # the assume; `_confirm`'s replay reads the same assume over ℚ,
            # which is the reading that decides under `real` semantics, and
            # it can say the point is outside the assumed region after the
            # float said it was inside.  Leaving the increment standing
            # produced a stamp line that contradicted itself in one
            # sentence.  Driven at `cefc4a9` on `assume(y*0.1*10.0 <= y)`
            # under the Kahan assert: *"74 point(s) executed, 65 inside
            # the declared set and admitted by every assume ... declined
            # 39 assume-unsatisfied-over-the-rationals"* -- i.e. a count
            # that READS AS COVERAGE for 39 of its 65, which is the
            # failure this module's docstring exists to prevent.  It
            # additionally called them *"39 EXECUTED VIOLATION(S) ...
            # DECLINED"*, which they were not.  Fixed, the same fixture
            # reports 71 built, 47 admissible, 16 + 8 declined, and no
            # declined violations at all.
            #
            # It is taken back rather than never given because the two
            # readings run at different times: only the exact one can say
            # this, and it only runs where the float found a violation.
            # Not counted in `points_declined` either: this is not an
            # admissible violation the probe would not stand behind, it is
            # a point that was never inside the assumed region, and the
            # skip rate counts it by leaving it out of the numerator.
            #
            # **AND THIS PATH IS NOW THE RESIDUE RATHER THAN THE RULE.**
            # The gate above re-reads the assumes over ℚ at EVERY
            # admissible point, so a point this branch can still reach is
            # one whose gate reading came back UNREAD -- the whole-probe
            # confirmation budget was spent -- and whose `_confirm`, on
            # its own fresh budget, then managed the reading after all. It
            # was counted unconfirmed there, so the take-back is of both
            # numbers.
            admissible -= 1
            if exact_assumes is None and census.assumes_in_program > 0:
                unconfirmed -= 1
            return None, None
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
            points_admissible_unconfirmed=unconfirmed,
            points_declined=declined_points,
            violations_seen=violations,
            adjudications=adjudged.items(),
            abstentions=abstained.items(),
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


def unverified_declarations(assumptions) -> tuple[str, ...]:
    """The stamped assumptions the verdict RESTS ON and stelling does not check.

    A verdict carrying one of these is not stelling's own claim: the
    caller declared something, stelling widened by it and said in the
    stamp that it verified NEITHER half.  See :data:`DECLARED_NOT_VERIFIED`
    for the measurement that made this necessary.

    Read out of plain strings the caller passed; nothing is imported to
    obtain them.
    """
    return tuple(a for a in assumptions if DECLARED_NOT_VERIFIED in a)


def _fire(hit: Falsification, report: ProbeReport):
    """Raise, with the attribution the evidence actually supports.

    **TWO HEADLINES, AND WHICH ONE IS TRUE IS NOT THE PROBE'S CHOICE.**
    The unconditional one — *"stelling is UNSOUND at this query"* — is a
    claim about this tool, and it is only correct when the discharge was
    stelling's to make.  When the verdict rests on a caller DECLARATION
    stelling stamps as unverified (:data:`DECLARED_NOT_VERIFIED`), the
    executed counterexample is exactly as real and its CAUSE is
    undetermined: either the declaration is false on this backend or
    stelling is unsound, and this probe cannot tell those apart.  Saying
    the first sentence there is the same accounting failure — a defect
    filed against the wrong party, in the one message a reader acts on —
    that this module rejected the REFUTED disposition to avoid.

    It still RAISES either way.  A VERIFIED that is false at a declared
    point must stop a CI run whichever half of the composition failed, and
    the caller has just been handed the point at which to check their
    declaration.

    **AND THE SPLIT IS PER-VERDICT WHILE THE FIRING IS PER-OBLIGATION,
    WHICH IS WHERE THE FIRST VERSION OF THIS MESSAGE MISDIRECTED.**
    ``report.assumptions`` is the verdict's stamped lines; the condition
    is *"does the phrase appear ANYWHERE in them"*.  A verdict has many
    obligations and one set of assumptions, so an `ieee` verdict whose
    obligation 0 goes through ``exp`` — the thing the ``libm_budget``
    declaration is about — softens the headline for obligation 1 as well,
    however unrelated obligation 1 is.  Driven through this module's own
    public entry point with the statuses supplied and nothing mutated,
    two obligations over ``float64 [0, 9]``::

        assert_(jnp.exp(x) >= 0.0)     # obligation 0, uses exp
        assert_(x * x <= 40.0)         # obligation 1, PURE MULTIPLY,
                                       #   false at x = 9

    With no assumptions, the firing on obligation 1 reads *"stelling is
    UNSOUND at this query"*.  With the real shipped ``xla-cpu-2026-08``
    line present — which is about ``exp``, and reaches this verdict
    through obligation 0 — **the same firing on the same pure-multiply
    obligation** got the softened headline and a tail telling the reader
    to go and check their libm profile.  The disjunction stayed true and
    the ACTIONABLE INSTRUCTION was wrong: there is no libm anywhere in
    ``x * x``.

    **THE HONEST KEY IS NOT REACHABLE, SO THE MESSAGE SAYS THAT INSTEAD
    OF GUESSING.**  Keying the split on *"does THIS obligation rest on a
    declared assumption"* needs a per-obligation attribution of the
    assumptions, and the probe is not given one: ``statuses`` is per
    obligation, ``assumptions`` is per verdict, and the module that knows
    the answer is the module this one may not import.  Nor can the
    attribution be reconstructed from the text: a heuristic that read
    ``exp@float32`` out of the declaration and compared it with the
    obligation's primitives would be RIGHT on today's one shipped line and
    would fail SILENTLY and in the unsafe direction on a declaration that
    does not name its operations — printing the categorical *"stelling is
    UNSOUND"* on a verdict a declaration paid for, which is the exact
    failure this split was added to remove.

    So the split stays conservative — any unverified declaration on the
    verdict softens the headline — and the two things that were wrong with
    the message are fixed in the message:

    * it no longer asserts that THIS obligation rests on the declaration.
      It says the VERDICT does, which is what the stamp establishes, and
      it says in plain words that the probe cannot tell which obligation a
      verdict-level declaration was carried for;
    * it prints, for the fired obligation, the exact list of primitives
      its value is computed from (:func:`_obligation_operations`), so the
      reader can settle the question in one glance instead of being sent
      to the wrong file.  On the driven case above that list is ``mul``,
      and no libm profile has anything to do with ``mul``.

    **AND THE ASYMMETRY THIS SPLIT DOES NOT COVER, RECORDED HERE BECAUSE
    IT IS AN ACCIDENT AND NOT A DESIGN.**  The split keys on the phrase
    :data:`DECLARED_NOT_VERIFIED`, and only the ``libm_budget`` line
    carries it.  A ``real`` verdict over ``exp`` carries a DIFFERENT
    stamped assumption — *"exp endpoints assume a faithfully-rounded libm
    (error <= 1 ulp), bumped 1 ulp outward"* — which is equally unchecked
    by stelling, equally a claim about somebody else's libm, and which
    this function does NOT see: a firing on such a verdict would print the
    categorical *"stelling is UNSOUND at this query"*.  Nothing reaches it
    today for one reason only: under ``real`` semantics only an exact test
    may admit, and :func:`_replay` has no rational reading of ``exp``, so
    every violation on such a program declines with ``'exp' has no exact
    rational reading``.  **That is a decline standing in for a guard.**
    Add ``exp`` to the replay's tables — a plausible reach improvement,
    and ``dot_general``/``sort``/``cumsum`` are already on that work list
    — and this message starts accusing stelling for an unverified host-libm
    assumption, with nothing here to notice.  The repair, when it is
    wanted, is on the emitting side: a stamped assumption that stelling
    does not check should carry the phrase that says so, and then this
    split sees it.
    """
    conditioned = unverified_declarations(report.assumptions)
    if hit.operations:
        computed = (
            f"  assert #{hit.obligation_position} is computed from: "
            f"{', '.join(hit.operations)}\n"
        )
    else:
        computed = (
            f"  assert #{hit.obligation_position} is computed from NO "
            f"operation at all: its value comes straight from a declared "
            f"input or a literal.\n"
        )
    if conditioned:
        head = (
            "FALSIFICATION PROBE FIRED — this VERIFIED is FALSE at a "
            "declared point, and THIS VERDICT rests on a DECLARATION "
            "stelling does not check.\n"
        )
        tail = (
            "\n\nNo verdict is returned. THIS IS NOT A REPORT THAT STELLING "
            "IS UNSOUND, and it is not a finding about the program under "
            "test either: the program did what it does, the analysis "
            "discharged an obligation the program violates at a point the "
            "analysis admitted, and THIS VERDICT was made UNDER THE "
            "FOLLOWING DECLARATION(S), which stelling accepted from you and "
            "states it cannot verify:\n\n  - "
            + "\n  - ".join(conditioned)
            + "\n\nWHETHER THE DECLARATION(S) ABOVE BEAR ON *THIS* "
            "OBLIGATION IS NOT SOMETHING THIS PROBE CAN TELL YOU. They are "
            "stamped on the VERDICT AS A WHOLE, and a verdict has many "
            "obligations: the probe is handed a status per obligation but "
            "the assumptions only per verdict, and it may not import the "
            "analysis to ask which discharge used which. What it CAN tell "
            "you exactly, read off the traced program itself, is what the "
            "obligation that just fired is computed from:\n\n"
            + computed
            + "\nSo, in this order. If a declaration above bears on those "
            "operations, CHECK IT FIRST, at the point reported above: a "
            "declaration that does not hold on this backend produces "
            "exactly this counterexample, and the point is where to look. "
            "If none of them does, then no declaration licensed THIS "
            "discharge and this IS a soundness event in stelling: see "
            "stelling/falsify.py.\n\n"
        )
    else:
        head = (
            "FALSIFICATION PROBE FIRED — stelling is UNSOUND at this query.\n"
        )
        tail = (
            "\n\nNo verdict is returned. This is not a finding about the "
            "program under test: the program did what it does, and the "
            "ANALYSIS discharged an obligation the program violates at a "
            "point the analysis itself admitted. Returning REFUTED would "
            "report a defect in stelling as a defect in your code; "
            "returning UNKNOWN would file a soundness event in the notes. "
            "Both were rejected — see stelling/falsify.py.\n\n"
        )
    raise VerifiedFalsified(
        head + hit.render() + tail + report.stamp_line(),
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
# WHAT REPLACES IT, ON THE ONE PATH THAT REACHES HERE.  The same
# exact-rational standard `verdict.Witness` applies before a solver model
# may become a witness at all: replay the point through exact arithmetic.
#
# AND THAT SENTENCE USED TO BE WRITTEN AS THOUGH IT DESCRIBED THE WHOLE
# FIRE CONDITION, WHICH IT DOES NOT.  `_confirm` has THREE tests that may
# admit and only this one calls `_replay`: `ieee-executed-float` and
# `exact-integer-arithmetic` both return before it.  Each of those is
# exact for its own reason -- the executed float IS the subject of an
# `ieee` claim, and an all-integral program's executed values are already
# exact while a rational replay of one would UNWRAP a genuine int8 wrap --
# and both reasons are written out in the module docstring under THE FIRE
# CONDITION.  What is not true, and was claimed here and in two other
# places, is that all three meet the standard this paragraph names.
#
# Every finite IEEE float IS a rational -- `Fraction(float)` is exact and
# lossless --
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
# ABSTAINS -- and an abstention is a DECLINE, counted by reason in
# `ProbeReport.abstentions` and never a firing.  It used to fall back to
# an ulp-stability proxy, and that fall-back is what made the Kahan false
# alarm reachable in four lines through `jnp.where` or a fractional `pow`
# long after the replay that was supposed to have fixed it shipped.  The
# argument for declining instead is in `_confirm`.


class _Unreplayable(Exception):
    """This program has a step with no exact rational reading at this point."""


class _AssumesComplete(Exception):
    """Every assume the program states has been read; stop walking.

    Not an error and never seen by a caller: :func:`_replay` catches it and
    returns what it has.  It exists because the assume-confirmation gate in
    :func:`probe` needs ONLY the assumes, and everything after the last one
    is work it will not look at -- work that can also ABSTAIN, which would
    cost the gate an exact reading of assumes it had already read
    successfully.  ``assume(x >= 0.0)`` in front of ``assert_(exp(x) <=
    C)`` is the shape: the assume is exactly readable over ℚ and ``exp``
    is not, and without the early stop every point of such a program would
    be reported unconfirmed.
    """


# HOW MUCH RATIONAL ARITHMETIC ONE REPLAY MAY DO.  THREE numbers, because
# the cost of a rational replay is the product of two independent things
# and the first version of this bounded only one of them.
#
# **WIDE IS NOT THE EXPENSIVE DIRECTION; DEEP IS.**  `REPLAY_ELEMENT_BUDGET`
# counts element-visits off the AVALS, which is a fact about the SHAPES and
# is known before any arithmetic happens.  That half still holds and was
# re-measured on this tree: `sum(v*v) <= 0.5` over one float64 declaration
# costs 267 ms at 65,536 elements, ~2 microseconds per element visited (254
# ms before the width check below, which walks the same elements once more
# and is 5% of the total).
#
# What it does not see at all is the SIZE OF THE RATIONALS.  A `Fraction`
# grows: `y = y*y` doubles the numerator and denominator bit-widths every
# step, so a program of nineteen equations -- nineteen element-visits, 0.008%
# of the element budget -- costs, measured here on jax 0.11.0 at one
# declared float64 point:
#
#     squarings    element-visits    replay wall time
#     ---------    --------------    ----------------
#            12                15             0.031 s
#            14                17             0.454 s
#            16                19             7.728 s
#            17                20            28.117 s
#
# i.e. 4x per step, forever, at a cost the element budget reads as free --
# and a probe run pays it once per VIOLATING POINT, so the default budget of
# 256 points multiplies it.  A budget that says "2.8 - 10.5 microseconds per
# element visited" and then charges 4.7 SECONDS for one is not a budget.
#
# So the missing term is bounded directly.  `REPLAY_BIT_BUDGET` caps the
# width of every value the replay produces, checked as each equation's
# output appears; past it the replay abstains, and (since abstention now
# declines) the probe declines rather than firing on a weaker test.  The two
# together bound the work: at most `REPLAY_ELEMENT_BUDGET` values, each at
# most `REPLAY_BIT_BUDGET` bits wide, and one `Fraction` multiply at 4,096
# bits is 19 microseconds here -- so the pair is a few seconds in the worst
# case rather than unbounded.
#
# 4,096 bits is chosen to sit well above ordinary float64 arithmetic and
# well below the cascade.  A float64 is a dyadic rational of at most 1,075
# denominator bits; a sum of any number of them shares that denominator, a
# product of k of them reaches 53k numerator bits, and `_rat_pow` already
# refuses an exponent past 64 (53 * 64 = 3,392).  Measured over this
# repository's own corpus the cap declines NOTHING that the unbounded
# version admitted -- not one abstention in the whole forced-probe run cites
# it.  It is not free everywhere, and the shape it costs is the shape it is
# for: `0.5` squared twelve times is `2**-4096`, one bit over, and a program
# that repeatedly squares more than about eleven times therefore declines
# where the unbounded version would have refuted.  That is a bound doing its
# job rather than a bound in the wrong place -- what makes it the right
# trade is that the same program at sixteen squarings cost 7.7 seconds per
# point.
#
# `REPLAY_SECONDS_BUDGET` is a BACKSTOP and not the bound: it exists for the
# shape the first two cannot see -- many equations, each individually cheap.
# It is checked inside the element loops as well as between equations, so a
# single very wide equation cannot run past it either.
#
# **IT IS THINNER THAN THE WORD "GENEROUS", SO WHETHER THIS INSTRUMENT
# FIRES ON A GIVEN PROGRAM CAN DEPEND ON THE MACHINE.  THAT IS WRITTEN
# DOWN HERE RATHER THAN LEFT TO BE INFERRED.**  The deterministic pair
# permits 250,000 values at 4,096 bits, and at ~19us a multiply that is
# 4.75 seconds against a 5.0-second clock -- no margin worth the word.
# And ordinary shapes land well inside it: a `(35000,)` float64
# declaration squared six times replays in 1.55 s on the machine this was
# written on, a third of the backstop, so a machine three times slower
# declines the same program.
#
# WHAT VARIES WITH THE MACHINE IS REACH, NEVER SOUNDNESS.  Only an exact
# test may admit a firing and the clock can only ABSTAIN, so a slow
# machine declines where a fast one fires and never the reverse: slow
# hardware buys fewer firings, not wrong ones.  A decline the clock
# produced says so in its own words (`ran past its 5.0s wall-clock
# backstop`) in `ProbeReport.abstentions` and in the stamp line, so a
# report that was decided by the non-deterministic bound shows it.
#
# The deterministic bounds are pushed IN FRONT of it wherever they can be,
# which is the repair actually available here: `guard.width` is checked as
# each element is produced rather than once the equation that trips it is
# complete, so a program the width budget will refuse no longer pays for
# the whole equation first.  Measured, `(30000,)` squared seven times:
# 3.10 s then declined, now 1.37 s then declined, same reason and same
# answer.  That does not make the clock deterministic -- nothing short of
# a deterministic work counter would, and that is a numeric-policy change
# rather than a repair -- it shrinks the window in which the clock, rather
# than the program, is what decided.
REPLAY_ELEMENT_BUDGET = 250_000
REPLAY_BIT_BUDGET = 4_096
REPLAY_SECONDS_BUDGET = 5.0


class _Guard:
    """The running cost of one replay, and the two bounds it is checked on.

    One instance per :func:`_replay` call, threaded through the evaluators
    rather than kept in a module global: the probe may run inside a
    caller's test session, and this module's independence argument extends
    to not holding process-wide state that a second probe could inherit.
    """

    __slots__ = ("_deadline",)

    def __init__(self, seconds=None) -> None:
        # `seconds` is how much wall clock THIS replay may have, and the
        # default is the whole backstop.  It is a parameter because the
        # assume-confirmation pass in `probe` spends ONE backstop across
        # every point it confirms rather than one per point, and hands each
        # replay what is left of it.
        if seconds is None:
            seconds = REPLAY_SECONDS_BUDGET
        self._deadline = time.monotonic() + seconds

    def tick(self) -> None:
        """Has this replay run past its wall-clock backstop?"""
        if time.monotonic() > self._deadline:
            raise _Unreplayable(
                f"the replay ran past its {REPLAY_SECONDS_BUDGET}s "
                f"wall-clock backstop"
            )

    def width(self, v) -> None:
        """Is this value still inside the rational-width budget?

        The larger of the numerator's and the denominator's bit-length,
        because that is the operand size a `Fraction` operation actually
        pays for.  Booleans have no width.
        """
        if type(v) is bool:
            return
        w = max(v.numerator.bit_length(), v.denominator.bit_length())
        if w > REPLAY_BIT_BUDGET:
            raise _Unreplayable(
                f"a value {w} bits wide, and the rational-width budget is "
                f"{REPLAY_BIT_BUDGET}: this program's exact values grow "
                f"faster than the replay may follow"
            )

    def check(self, arr) -> None:
        """Every element of one equation's output, plus the clock."""
        self.tick()
        for v in np.asarray(arr, dtype=object).reshape(-1):
            self.width(v)


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


# Call primitives whose body is a nested jaxpr to be replayed in place.
#
# **THE NAMES ARE MEASURED, AND THE FIRST VERSION OF THIS TUPLE NAMED FOUR
# PRIMITIVES THAT DO NOT EXIST.**  It read `("pjit", "closed_call",
# "remat", "checkpoint")` with a comment saying `jax.numpy` routes almost
# everything through `pjit` -- and on both jax series this package supports
# (0.10.2 and 0.11.x) the call primitive is spelled `jit` and the
# rematerialisation one `remat2`, so not one of the four ever matched.  The
# comment was an exact description of what the tuple then did: over 1,200
# fuzzed programs the replay decided 6%, and 1,062 of its 1,128 abstentions
# read `'jit' has no exact rational reading`.  Every one of those was a
# violation adjudicated by the weaker test instead.
#
# **THE FIVE SPELLINGS THAT MATCH NOTHING TODAY ARE KEPT, AND THAT IS A
# DECISION WITH A MEASUREMENT BEHIND IT RATHER THAN THE RHETORIC THIS FILE
# RETIRED.**  Traced across every jax in reach -- 0.5.1, 0.10.2, 0.11.0 --
# `pjit` IS the live call primitive on 0.5.1, and `jit` is on the two
# supported series; so `pjit` is a backward-compatibility claim with
# evidence behind it and not a guess.  `closed_call`, `core_call`,
# `xla_call`, `remat` and `checkpoint` match nothing on any of the three,
# and jax's own rematerialisation primitive object reports its name as
# `remat2` on all three, so neither historical spelling is live anywhere.
#
# What makes keeping them different from the deny-list this file condemned
# is the DIRECTION a dead name fails in, and the two are opposites:
#
# * a dead name in a DENY-list subtracts protection, silently.  It is only
#   consulted to say *"refuse this"*, so a name nothing matches refuses
#   nothing, and the guard reads as armed while it is inert.  Four of the
#   eight names in the first `_REPEATING_OR_CONDITIONAL_BODIES` were
#   exactly that, which is why the restored list below carries a test that
#   traces every one of its names off the live jax;
# * a dead name in an ALLOW-list subtracts nothing.  This mapping is only
#   ever consulted with a LIVE primitive name in hand (`if name in
#   _CALL_PRIMITIVES`), so a key no equation carries is never reached, can
#   admit nothing, and cannot make any claim true or false on this series.
#   And should one go live, membership is still not trusted: the descent
#   re-derives the property from the equation before it walks anything.
#
# So a dead ALLOW-list name costs one thing only -- an UNCHECKED claim
# about a jax nobody here runs -- and that is what this paragraph is: the
# claim, and the measurement of which names it covers.  What keeps the
# mapping honest for the names that ARE live is not the list, it is two
# tests in
# `tests/test_falsify_fire_condition.py` that trace the LIVE jax and
# assert the primitive they find is one of these:
# `test_the_call_primitive_the_live_jax_emits_is_replayed` for `jnp.where`
# and `test_the_rematerialisation_primitive_the_live_jax_emits_is_read`
# for `jax.checkpoint`.  A name list with no such test is how this got
# shipped.
#
# **AND THIS COMMENT CLAIMED THE `jax.checkpoint` HALF FOR A BATCH BEFORE
# IT EXISTED.**  It said one test traced both; that test traced
# `jnp.where` only, and no falsify test named `jax.checkpoint` or
# `remat2` anywhere outside a docstring -- so `remat2`, the one name
# `_call_jaxpr_of` below is load-bearing for, was exactly as unchecked as
# `pjit` had been.  Prose asserting that a live check exists is the same
# defect as a name list nothing checks, moved one level up, and it is
# recorded here because this is the level a reader audits from.
_CALL_PRIMITIVES = {
    "jit": "jaxpr",
    "pjit": "jaxpr",
    "closed_call": "call_jaxpr",
    "core_call": "call_jaxpr",
    "xla_call": "call_jaxpr",
    "remat2": "jaxpr",
    "remat": "jaxpr",
    "checkpoint": "jaxpr",
}

# **AND MEMBERSHIP OF THAT MAPPING IS A CLAIM: THE BODY AT THAT KEY RUNS
# EXACTLY ONCE PER EQUATION, WITH THE EQUATION'S OWN OPERANDS AS ITS
# ARGUMENTS.**  The claim is checked at the descent, by THREE opinions in
# order -- a name, an iteration count, and a signature -- and this is the
# third version of that arrangement.  The history is worth the space,
# because each version lost something the one before it caught.
#
# **VERSION ONE: A DENY-LIST OF NAMES, AND IT PROTECTED LESS THAN IT READ
# AS.**  `_REPEATING_OR_CONDITIONAL_BODIES` held eight names checked at
# the descent, and measured on this jax:
#
# * four of them (`cond_p`, `fori_loop`, `switch`, `while_loop`) are not
#   primitive names on either supported series at all, and a fifth
#   (`platform_index`) is a primitive that carries no jaxpr and so could
#   never be descended.  Only `scan`, `while` and `cond` were live.  That
#   is exactly the *"a name that matches nothing costs reach, silently"*
#   failure the commentary below condemns, sitting INSIDE the guard meant
#   to catch it;
# * `reduce` and `reduce_window` were NOT in it, are real primitives on
#   both series, and carry their bodies under the key `jaxpr` -- the first
#   key `_call_jaxpr_of` used to look for.  So the single-set edit the
#   guard existed to catch (adding a name to the descent set) descended a
#   `reduce` body with NO guard, driven: a `stelling_assume` inside a
#   four-element `lax.reduce` combiner read `[False]` where the body's
#   four real invocations all satisfy it, and the `assumes_only` early
#   stop then called that reading COMPLETE;
# * and a deny-list can only ever name what someone thought of.
#
# **VERSION TWO REPLACED IT WITH THE DERIVATION ALONE, AND THAT LOST
# `fori_loop`.**  The derivation (:func:`_body_runs_once`) is structural
# and cannot be behind a rename, which is the property the name list did
# not have -- but it is a NECESSARY condition and not a sufficient one,
# and the module's own docstring conceded as much with the words *"a
# hypothetical primitive with one signature-matching body could iterate
# it"*.  **IT IS NOT HYPOTHETICAL.  IT IS `lax.fori_loop`.**  A
# static-bound `fori_loop` -- and `lax.scan(f, init, None, length=N)` --
# traces to a `scan` equation carrying ONE nested jaxpr whose signature is
# the equation's, element for element, on jax 0.11.0 and on 0.10.2 alike:
# there is no `xs` to lose a leading axis and no stacked `ys` to grow one,
# so NEITHER structural fact fires and the body reads as a call.  Driven
# end to end on the one edit this guard exists to police, with `scan`
# added to the descent set and `assert_(fori_loop(0, 3, lambda i, c: c +
# 1.0, y) <= 3.0)` over `float64 [1, 2]` -- an obligation FALSE at every
# declared point::
#
#     version one   ProbeInvariantViolated, both `assumes_only` settings
#     version two   `_replay` reads `asserts == [True]`, no guard at all,
#                   and the probe DECLINES 31 real violations under
#                   `exact-replay-holds-over-the-rationals`
#
# That is a positively wrong adjudication label where the deny-list
# abstained, and `length=1` and `length=0` -- a body that runs ONCE with
# the wrong carry, and a body that runs NOT AT ALL -- pass the derivation
# too.  The executed-assume gate does not cover it either: this program
# states no assume, so `census.assumes_in_program` is 0.
#
# **VERSION THREE, HERE, IS BOTH, PLUS THE ONE STRUCTURAL FACT THAT SEES A
# TRIP COUNT.**  In the order :func:`_body_runs_once` asks them:
#
#     1. FIRST OPINION, A NAME.  `_REPEATING_OR_CONDITIONAL_BODIES`,
#        restored and corrected to the five names that are LIVE primitives
#        on both supported series -- and pinned by a test that traces each
#        one off the running jax, so no name in it can match nothing.
#     2. AN ITERATION COUNT.  A `scan` equation carries `length`; no call
#        primitive on any series in reach carries any parameter naming a
#        number of iterations.  A construct that says how many times it
#        runs its body is not a call, whatever it is called.
#     3. THE SIGNATURE, AND THE JAXPR COUNT.  A body entered once on the
#        equation's own operands must have the equation's own signature,
#        and a primitive carrying a second jaxpr is not a call.
#
# Traced on both supported series, this is what each opinion sees:
#
#     primitive       name  trip count  nested  body signature vs equation
#     ---------       ----  ----------  ------  --------------------------
#     jit / remat2      -       -          1    identical                 
#     scan, xs          Y    length=N      1    xs invar loses leading axis
#     scan, ys          Y    length=N      1    ys outvar GAINS one       
#     scan, neither     Y    length=N      1    IDENTICAL -- fori_loop    
#     reduce            Y       -          1    operand ARRAY, body scalar
#     reduce_window     Y       -          1    operand and result differ 
#     while             Y       -          2    (cond_jaxpr, body_jaxpr)  
#     cond              Y       -        N>=1   index operand has no invar
#
# **WHICH SHAPES THIS COVERS, AND WHICH IT DOES NOT.**  Covered: every row
# above, each by at least one opinion, and the `scan, neither` row -- the
# `fori_loop` row -- by opinions 1 and 2 and by nothing else.  NOT
# covered: a primitive that carries exactly one signature-matching body,
# no parameter naming a trip count, and a name in neither list, which
# iterates that body anyway.  Nothing here can see such a thing, and the
# honest statement of the guard is therefore that it makes the ONE EDIT a
# reader makes -- adding a name to `_CALL_PRIMITIVES` -- loud for every
# repeating construct jax ships today, on both series, rather than that it
# is closed.  Version one covered `{scan-with-xs, scan-without-xs, while,
# cond}` and not `{reduce, reduce_window}`; version two covered
# `{scan-with-xs, scan-with-ys, while, cond, reduce, reduce_window}` and
# not `{scan-without-xs}`; this covers the union, which is every one of
# them.
#
# All three opinions raise :class:`ProbeInvariantViolated` rather than
# reading one iteration's assume as the program's, and all three are
# asked by the SAME function, so `_replay` and `_assume_sites_reachable`
# cannot come to different answers about which bodies are walked.
#
# What the mapping's VALUE buys is the other half of the same lesson.  The
# key used to be probed -- `("jaxpr", "call_jaxpr")`, first hit wins -- for
# any name in the descent set, so a reader adding `reduce` got a body
# handed to them silently.  Now each name says where its own body lives,
# and a name whose key is wrong on some series abstains loudly
# (`_Unreplayable`) instead of finding somebody else's jaxpr.


# THE FIRST OPINION, RESTORED AND CORRECTED.
#
# Five names, and every one of them is a LIVE primitive on both supported
# jax series -- which is the property the eight-name version did not have
# and is pinned by
# `tests/test_falsify_fire_condition.py::test_every_name_in_the_repeating_deny_list_is_a_LIVE_primitive`,
# which traces a fixture per name off the running jax rather than taking
# this comment's word for it.  A name that matches nothing in a DENY-list
# reads as protection and provides none; see the commentary above for the
# four that did.
#
# Four of the five are ALSO refused structurally today, so on this jax the
# list is load-bearing for exactly one of them: **`scan`**, whose
# no-`xs`-no-`ys` shape -- `lax.fori_loop`, and `lax.scan(f, init, None,
# length=N)` -- has the equation's own signature and is invisible to every
# structural fact but the trip count.  It is not written as a one-name set
# because a deny-list whose entries are redundant TODAY is how a rename or
# a jax version bump gets caught, and because the entry a reader needs to
# find when they add a name to `_CALL_PRIMITIVES` is the name they typed.
_REPEATING_OR_CONDITIONAL_BODIES = frozenset({
    "scan",
    "while",
    "cond",
    "reduce",
    "reduce_window",
})

# AND THE ONE STRUCTURAL FACT THAT SEES A TRIP COUNT WITHOUT KNOWING A
# NAME.  `scan` carries `length` on both supported series (and on 0.5.1);
# `jit` and `remat2` carry no such parameter on any of the three, traced,
# so this clause costs no reach on anything this replay descends today --
# which is measured, not assumed, by the no-lost-reach control.
#
# A parameter name is still a name, and this one fails in the same safe
# direction as the deny-list: if jax ever spells a trip count differently
# the clause refuses nothing, and the equation still has to get past the
# deny-list and the signature check.  What it buys is the case those two
# cannot have: a repeating primitive nobody here has heard of that
# announces its own trip count.
_ITERATION_COUNT_PARAMS = ("length",)


def _body_runs_once(eqn, body):
    """Is ``body`` entered exactly once per ``eqn``, on ``eqn``'s operands?

    Returns ``(True, None)`` or ``(False, reason)``.  This is the property
    :data:`_CALL_PRIMITIVES` membership CLAIMS, and it is checked here
    rather than trusted -- by three opinions, asked in this order.  The
    commentary above carries the history and the traced table; this is
    what the code does.

    **FIRST, THE NAME.**  :data:`_REPEATING_OR_CONDITIONAL_BODIES` names
    the five constructs that are live primitives on both supported series
    and whose bodies are known to repeat or to branch.  It is the only
    opinion that sees a `scan` with no `xs` and no stacked `ys` -- which
    is what a static-bound `lax.fori_loop` traces to, whose body has the
    equation's own signature exactly, and which the derivation below
    passes as a call on both series.  That was a real regression and it
    is why this list is back.

    **SECOND, AN ITERATION COUNT.**  An equation carrying a parameter that
    says how many times its body runs (:data:`_ITERATION_COUNT_PARAMS`) is
    not a call, whatever it is named.  `scan` carries `length`; no call
    primitive in reach carries any of these.

    **THIRD, THE STRUCTURE.**  Two facts, neither of them a name:

    * **the equation carries exactly one nested jaxpr.**  `while` carries
      two (`cond_jaxpr`, `body_jaxpr`) and `cond` carries one per branch;
      a call carries one body.
    * **that jaxpr's signature is the equation's.**  `scan` strips the
      leading axis off every `xs` and grows one on every stacked `ys`,
      `reduce` and `reduce_window` hand their combiner two SCALARS where
      the equation takes an array, and `cond` consumes a branch index the
      body never sees.  A body entered once on the equation's own operands
      cannot do any of that.

    **WHAT NONE OF THE THREE CAN SEE** is a primitive with one
    signature-matching body, no parameter naming a trip count, and a name
    in neither list, which iterates that body anyway.  The structural
    facts are NECESSARY conditions for a call and not sufficient ones, and
    the first version of this function said so and then named the
    counterexample *hypothetical*; it was `fori_loop`.  The claim this
    docstring makes is therefore the narrow one: every repeating or
    conditional construct jax ships on either supported series fails at
    least one of the three, traced, and the one edit this file expects a
    reader to make is loud for all of them.

    A dtype or shape neither side can be read from compares EQUAL to the
    same unreadable thing on the other side: refusing an ordinary `jit`
    over a token operand would cost reach, and reach lost silently is the
    failure this file is built around.
    """
    name = eqn.primitive.name
    if name in _REPEATING_OR_CONDITIONAL_BODIES:
        return False, (
            f"{name!r} is named in `_REPEATING_OR_CONDITIONAL_BODIES` as a "
            f"construct whose body repeats or branches, and the structural "
            f"facts below cannot see every shape of it -- a `scan` with no "
            f"`xs` and no stacked `ys` has the equation's own signature"
        )
    for param in _ITERATION_COUNT_PARAMS:
        if param in eqn.params:
            return False, (
                f"it carries an iteration count ({param}="
                f"{eqn.params[param]!r}) and a call carries none"
            )
    nested = _sub_jaxprs(eqn)
    if len(nested) != 1:
        return False, (
            f"it carries {len(nested)} nested jaxprs and a call carries one"
        )
    if len(body.invars) != len(eqn.invars):
        return False, (
            f"its body takes {len(body.invars)} argument(s) for "
            f"{len(eqn.invars)} operand(s)"
        )
    if len(body.outvars) != len(eqn.outvars):
        return False, (
            f"its body returns {len(body.outvars)} value(s) for "
            f"{len(eqn.outvars)} result(s)"
        )
    for kind, ours, theirs in (
        ("operand", eqn.invars, body.invars),
        ("result", eqn.outvars, body.outvars),
    ):
        for i, (a, b) in enumerate(zip(ours, theirs)):
            if _aval_shape_dtype(a) != _aval_shape_dtype(b):
                return False, (
                    f"{kind} {i} is {_aval_shape_dtype(a)} at the equation "
                    f"and {_aval_shape_dtype(b)} in the body"
                )
    return True, None


def _aval_shape_dtype(atom):
    """``(shape, dtype)`` for an atom, with unreadable halves left as-is.

    Unreadable compares equal to unreadable, which is deliberate: see
    :func:`_body_runs_once`.
    """
    aval = getattr(atom, "aval", None)
    shape = getattr(aval, "shape", "unreadable-shape")
    dtype = getattr(aval, "dtype", None)
    try:
        dtype = np.dtype(dtype).name
    except (TypeError, ValueError):
        dtype = "unreadable-dtype"
    return (tuple(shape) if isinstance(shape, tuple) else shape, dtype)


def _assume_sites_reachable(jaxpr, path=()):
    """Every ``stelling_assume`` OCCURRENCE :func:`_replay`'s descent reaches.

    Returns ``(sites, alive)``: a set of ``(call path, equation)`` keys
    spelled as tuples of ``id``, and a list holding a reference to every
    object whose ``id`` is in a key so that none can be recycled while the
    caller is comparing.

    **THIS IS THE OTHER HALF OF THE GUARD, AND IT REPLACED A COUNT.**  The
    `assumes_only` early stop used to compare `len(assumes)` -- READINGS --
    with `census.assumes_in_program` -- EQUATIONS counted by the widest
    walk there is (:func:`_sub_jaxprs`, every depth, `scan` bodies
    included).  Two numbers taken by two different rules agree only by an
    argument, the argument was written in a comment, and a comment cannot
    fail.  Worse, the comparison was one-directional: one descent into a
    body that runs N times reads FEWER occurrences than the body has
    executions, and no count of readings can see that.

    So the stop compares SETS taken by the SAME rule: the occurrences this
    walk reached against the occurrences this walk can reach.  That is
    independent of the shape of the descent -- it says nothing about how
    many primitives are in :data:`_CALL_PRIMITIVES` -- and it catches a
    walk that read too few (the stop cannot fire) and a walk that read one
    twice or read one it cannot reach (both raise).

    It does NOT replace the whole-program check.  This set is what the
    descent rule REACHES; `census.assumes_in_program` is what the program
    STATES, and the gate in :func:`probe` still requires the reading to
    account for every one of the latter.  An assume behind a `scan` is in
    the second and not the first, and a reading that stopped without it is
    SHORT -- which is what that gate is for and why both survive.
    """
    sites: set = set()
    alive: list = []
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "stelling_assume":
            sites.add(path + (id(eqn),))
            alive.append(eqn)
            continue
        if name not in _CALL_PRIMITIVES:
            continue
        try:
            body, _ = _call_jaxpr_of(eqn)
        except _Unreplayable:
            # the replay will abstain here too, so nothing under it is
            # reachable and nothing under it will be read
            continue
        ok, _why = _body_runs_once(eqn, body)
        if not ok:
            # the replay RAISES when it reaches this equation; what is
            # under it is not reachable and must not be counted as such
            continue
        inner, inner_alive = _assume_sites_reachable(body, path + (id(eqn),))
        sites |= inner
        alive.append(eqn)
        alive.extend(inner_alive)
    return sites, alive



def _call_jaxpr_of(eqn):
    """The nested jaxpr a call primitive carries, and the consts to run it.

    TWO SHAPES REACH HERE AND BOTH ARE LIVE.  A `ClosedJaxpr` carries its
    own constants (`.jaxpr` and `.consts`); a BARE `Jaxpr` carries no
    constants, so its constvars -- if it had any -- have nothing here to
    bind them to, and one that does carry constvars is refused rather than
    guessed at.

    **WHICH SHAPE COMES FROM WHERE IS MEASURED, BECAUSE THE FIRST ACCOUNT
    OF THIS FUNCTION WAS WRONG.**  It said requiring the closed form had
    been "the second half of the same defect as the tuple above" and that
    "even spelled correctly, `jit` would have abstained".  Neither holds
    on either series this package supports.  Traced live:

    * jax 0.10.2 -- `jit` carries a real `ClosedJaxpr`; `remat2` carries a
      bare `Jaxpr` with no `.jaxpr`/`.consts` at all.
    * jax 0.11.x -- both carry a bare `Jaxpr`, but `Jaxpr` there has a
      `.jaxpr` property returning ITSELF and a `.consts` property
      returning `[]`, so the old `hasattr(sub, "jaxpr") and hasattr(sub,
      "consts")` test matched anyway.

    So the old test admitted `jit` on both series, and taking `99abdb0`
    and correcting ONLY `_CALL_PRIMITIVES` makes the `jnp.where` route
    read exactly and decline on 0.10.2 and 0.11.0 alike.  The widening
    here is real work and it is load-bearing for exactly one thing:
    **`remat2` on jax 0.10.2.**  Driven there too, with the Kahan shape
    behind a rematerialisation boundary::

        y = any_array((), "float64", (0.0, 2.0))
        z = jax.checkpoint(lambda a: (1e16 + a) - 1e16)(y)
        assert_(z <= y)        # true over R; the VERIFIED is correct

    At `99abdb0` with the tuple corrected and this function left
    unwidened, that program FIRES on jax 0.10.2 -- admitted by
    `ulp-proxy-refutes`, the false alarm one more route -- and reads
    exactly on 0.11.0.  Here it reads exactly on both, 46 violations
    declined `float-rounding-artefact` with no abstention.  The live check
    is `test_the_rematerialisation_primitive_the_live_jax_emits_is_read`.

    **AND THE KEY IS NAMED BY `_CALL_PRIMITIVES` RATHER THAN PROBED.**
    This function used to try `("jaxpr", "call_jaxpr")` in order for any
    name in the descent set and take the first hit.  `reduce` and
    `reduce_window` carry their combiner under `jaxpr`, so a reader adding
    either to the descent set -- the one edit this file expects a reader
    to make -- was handed a body immediately and silently.  Each name now
    says where its own body lives; a name whose key is wrong on some
    series abstains here, loudly and by name, instead of finding a jaxpr
    that belongs to a different construct.
    """
    name = eqn.primitive.name
    key = _CALL_PRIMITIVES.get(name)
    if key is None:
        raise _Unreplayable(
            f"{name!r} is not a call primitive this replay descends"
        )
    sub = eqn.params.get(key)
    if sub is not None:
        if hasattr(sub, "jaxpr") and hasattr(sub, "consts"):
            return sub.jaxpr, tuple(sub.consts)
        if isinstance(sub, jex_core.Jaxpr):
            if sub.constvars:
                raise _Unreplayable(
                    f"{name!r} carries a jaxpr with constvars "
                    f"and no constants to bind them to"
                )
            return sub, ()
    raise _Unreplayable(
        f"{name!r} carries no jaxpr to replay at its param {key!r}"
    )


# --------------------------------------------------------------------------
# THE NAME TABLES, AND WHICH DIRECTION EACH ONE FAILS IN
# --------------------------------------------------------------------------
#
# `_EXACT_BINARY`, `_EXACT_UNARY`, `_COMPARISONS`, `_BOOLEAN`, `_MOVEMENT`
# and `_REDUCTIONS` below, `_CALL_PRIMITIVES` above and `_MARGIN_RELATIONS`
# near the top of the file are all hardcoded jax primitive names, and a
# reader should know what each kind of mistake in one costs.
#
# **A NAME THAT MATCHES NOTHING COSTS REACH, SILENTLY.**  It never fires
# its branch, the replay reaches the final `raise`, and an abstention
# DECLINES -- so a jax rename, or a name that was never a primitive on any
# series, subtracts from what this instrument can prove and takes nothing
# from what it may claim.  Silently is the operative word: the only
# evidence is a decline count, which reads like a program the replay could
# not have read anyway.  That is why the tables are pinned against a LIVE
# trace rather than trusted -- `_CALL_PRIMITIVES` shipped with four names,
# not one of which was a primitive on either supported series, and the
# replay then decided 6% of violations while the weaker test decided the
# rest.  `_MOVEMENT` shipped with `expand_dims`, which is a `lax`
# FUNCTION and not a primitive: `jnp.expand_dims`, `lax.expand_dims` and
# `a[None]` all lower to `broadcast_in_dim` on 0.10.2 and 0.11.0 alike,
# so the entry had never matched anything and it is gone.
#
# **A NAME THAT MATCHES WITH THE WRONG READING INVENTS A REFUTATION**, and
# that is the direction with teeth, because this module's output is
# "stelling is UNSOUND at this query".  Two guards exist for exactly that
# and neither is optional: `_boolean_only`, for the four names jax spells
# both bitwise-integer and boolean arithmetic with, and `_movement`'s
# refusal to substitute zeros for an operand a primitive reads BY VALUE.
# Extending these tables is therefore not symmetric with trimming them.
#
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
# `and`/`or`/`xor`/`not` -- AND ONLY ON BOOLEANS.  jax spells BITWISE
# integer arithmetic with these same four primitives, and bitwise is not a
# boolean operation over Q: `5 & 2` is 0, while `bool(5) and bool(2)` is
# True.  Reading one as the other does not lose a refutation, it INVENTS
# one -- driven, before the guard below existed, to a `_confirm` returning
# `exact-replay-refutes-over-the-rationals` on an obligation TRUE over Q.
# It was unreachable only because `propagate._t_bool_logic` refuses
# bitwise-int, which is an invariant in a module this one is forbidden to
# import and therefore not this module's to rely on.  See `_boolean_only`.
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
    "rev": (0,),
    "slice": (0,),
    "concatenate": None,  # every operand is data; filled in at use
    "pad": (0, 1),
}

# `reduce_and` and `reduce_or` carry the same hazard as `_BOOLEAN` and go
# through the same guard: over an integer operand they are bitwise folds.
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


# How often the element loops ask the clock.  Frequent enough that one
# very wide equation cannot outrun the wall-clock backstop between
# equation boundaries; rare enough that `time.monotonic` is not the cost.
_TICK_STRIDE = 1024


def _ew(fn, *args, guard=None):
    """Apply ``fn`` elementwise over broadcast object arrays.

    THE WIDTH BOUND IS CHECKED AS EACH ELEMENT IS PRODUCED, not once the
    equation is complete.  Both give the same answer; only one of them
    gives it before paying for the answer.  A `(30000,)` declaration
    squared seven times is refused by the width budget, and used to
    compute all 30,000 elements of the squaring that trips it first.  That
    matters beyond the seconds: the wall-clock backstop is the one bound
    here that is not deterministic, so every second of work a
    deterministic bound could have refused earlier is a second in which
    the machine, rather than the program, might decide.
    """
    b = np.broadcast_arrays(*args)
    flat = [x.reshape(-1) for x in b]
    n = flat[0].size
    out = np.empty(n, dtype=object)
    for i in range(n):
        if guard is not None and i and i % _TICK_STRIDE == 0:
            guard.tick()
        v = fn(*(f[i] for f in flat))
        if guard is not None:
            guard.width(v)
        out[i] = v
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


def _reduce(fn, a, axes, guard=None):
    axes = tuple(int(x) for x in axes)
    if not axes:
        return a
    moved = np.moveaxis(a, axes, tuple(range(-len(axes), 0)))
    kept = moved.shape[: moved.ndim - len(axes)]
    rows = moved.reshape(math.prod(kept) if kept else 1, -1)
    out = np.empty(rows.shape[0], dtype=object)
    for i in range(rows.shape[0]):
        if guard is not None:
            # every row, not every `_TICK_STRIDE` rows: ONE row of a
            # `reduce_prod` over a long axis is itself unbounded work
            guard.tick()
        vals = list(rows[i])
        if not vals:
            raise _Unreplayable("a reduction over an empty axis")
        v = fn(vals)
        if guard is not None:
            guard.width(v)  # per row, for the reason `_ew` gives
        out[i] = v
    return out.reshape(kept)


def _boolean_only(name, eqn, out_dt):
    """Refuse ``and``/``or``/``xor``/``not`` on anything but booleans.

    THE ONE PLACE THIS EVALUATOR COULD INVENT A REFUTATION.  Everywhere
    else an unreadable step abstains, which loses a firing; here a wrong
    reading would produce one.  jax spells bitwise integer arithmetic with
    the same primitive names, and the two disagree at the first argument
    anyone would pick: `5 & 2` is 0, so a program asserting `(x & y) != 0`
    is FALSE at (5, 2) while a boolean replay of it says True -- and the
    fire condition would then report a correct VERIFIED as UNSOUND.

    Both directions are checked, output and operands, because either alone
    admits a case: an `and` of two int32s has an int32 OUTPUT, and a
    comparison of two int32s has a bool output over integer operands.
    """
    if out_dt.kind != "b" or any(
        np.dtype(v.aval.dtype).kind != "b" for v in eqn.invars
    ):
        raise _Unreplayable(
            f"{name!r} over a non-boolean operand is BITWISE arithmetic, "
            f"which has no reading as a boolean connective over ℚ"
        )


def _int_ok(vals, dt):
    """Did an integer-typed result stay inside its own dtype?

    THIS IS WHAT KEEPS THE INTEGER WRAP VISIBLE.  Exact rational
    arithmetic does not wrap, so replaying a program whose integer
    arithmetic overflowed would report a value the program never computed
    -- and would then declare the executed violation a rounding artefact
    and decline it, suppressing exactly the runtime-wrap catch this probe
    was measured to have.  A result that left its dtype is therefore not
    replayable: the evaluator abstains, and an abstention DECLINES.  (The
    all-integral case never reaches here at all -- :func:`_confirm`
    short-circuits it into exact integer arithmetic before the replay is
    consulted, which is what keeps the runtime-wrap catch.)
    """
    info = np.iinfo(dt)
    for v in vals:
        if v.denominator != 1 or not (int(info.min) <= v <= int(info.max)):
            raise _Unreplayable(
                "integer arithmetic left its dtype's range: the program "
                "wraps there and rational arithmetic does not"
            )


def _replay(census, point, guard=None, assumes_only=False):
    """Re-evaluate the traced program at ``point`` in exact rational arithmetic.

    Returns ``(assumes, asserts)`` as lists of Python bools, or raises
    :class:`_Unreplayable`.  The point is the same array tuple the float
    execution used, so under ``jax_enable_x64=1`` the two runs differ in
    exactly one thing: the arithmetic.

    ``guard`` is an optional :class:`_Guard` to charge this replay to.
    The default is a fresh one, i.e. this replay gets the whole
    :data:`REPLAY_SECONDS_BUDGET` to itself, which is right where the
    replay is adjudicating one violation.  The assume confirmation in
    :func:`probe` passes a SHARED guard instead, so that re-reading every
    admissible point's assumes costs the probe one budget in total rather
    than one per point -- see the comment at that call site for why the
    per-point spelling was not affordable.

    ``assumes_only`` stops the walk once every assume the program states
    has been read (:class:`_AssumesComplete`), leaving ``asserts``
    PARTIAL.  Only the assume-confirmation gate may pass it, and it reads
    nothing but the assumes; a caller that looked at ``asserts`` under
    this flag would be reading a short list.  It buys two things and both
    matter: the work after the last assume is not paid for, and -- more
    importantly -- an ABSTENTION after the last assume no longer costs the
    gate a reading it had already completed.

    **AND ITS STOP CONDITION COMPARES TWO SETS TAKEN BY THE SAME RULE,
    WHICH IS WHAT IT DID NOT USED TO DO.**  It used to compare
    ``len(assumes) >= census.assumes_in_program`` -- *readings taken* on
    the left, *equations written at every depth* on the right -- two
    numbers produced by two different walks, equal only by an argument
    that lived in a comment.  The stop now fires when the set of assume
    OCCURRENCES this walk has read equals the set
    :func:`_assume_sites_reachable` says this walk's own descent rule can
    reach; a reading that is short cannot fire it, a second reading of one
    occurrence raises, and a reading of an occurrence the rule cannot
    reach raises.  Both directions, and neither depends on the shape of
    the descent -- which is the point, because *"descend a ``scan`` body"*
    is a reach improvement someone will reasonably propose and the old
    comparison quietly assumed nobody had.

    That is what this walk read.  What the PROGRAM STATES is
    ``census.assumes_in_program``, counted by the widest walk there is,
    and the two are not the same number whenever an assume sits behind a
    primitive this replay does not descend.  The gate in :func:`probe`
    still requires the reading to account for every equation the program
    states before it may say anything, and that check is where a short
    reading turns into an UNCONFIRMED point rather than a satisfied
    assume.

    THAT IS NOT TRUE UNDER ``jax_enable_x64=0``, and the sentence used to
    claim it unconditionally.  With x64 off, jax truncates a ``float64``
    declaration to ``float32`` when the point is handed to it, so the
    executed run is a float32 run while this replay reads the float64
    array exactly -- two differences, not one, and the replay's answer is
    then about a program the float run did not execute.  Nothing reaches
    it, and the reason is now a decline in THIS module: :func:`probe`
    refuses a declaration whose dtype does not survive
    ``jax.numpy.asarray`` under the live config, before a single point is
    built (``dtype-narrowed-by-jax``).  The old argument was that
    ``propagate`` refuses the ``convert_element_type float64 -> float32``
    the truncation puts in the jaxpr, so no VERIFIED exists to probe --
    true, measured, and a fact about ANOTHER module, which is an invariant
    one line of that module can remove.  It is kept below as corroboration
    and no longer relied on.
    """
    cost = census.replay_cost
    if cost > REPLAY_ELEMENT_BUDGET:
        raise _Unreplayable(
            f"the replay would visit about {cost} elements and the budget "
            f"is {REPLAY_ELEMENT_BUDGET}"
        )
    guard = _Guard() if guard is None else guard
    assumes: list = []
    asserts: list = []
    # ONE ENTRY PER `stelling_assume` OCCURRENCE THIS WALK HAS READ, keyed
    # by the CALL PATH that reached it and not by the equation alone: a
    # `jit` body called twice is two occurrences of one equation object.
    # `_alive` holds a reference to every object whose `id` appears in a
    # key, so no id can be recycled underneath the set while the walk is
    # running.
    assume_sites: set = set()
    _alive: list = []
    # AND THE SET THIS ONE IS MEASURED AGAINST, taken by the SAME descent
    # rule off the same equation objects, so the two are comparable
    # key for key.  Built only when the program states an assume at all:
    # `assumes_in_program == 0` means the widest walk there is found none,
    # so there is nothing to reach and nothing to read, and the assume-free
    # programs this module is measured on pay nothing for this guard.
    if census.assumes_in_program > 0:
        reachable_sites, _reach_alive = _assume_sites_reachable(
            census.closed.jaxpr
        )
        _alive.append(_reach_alive)
    else:
        reachable_sites = set()

    def run(jaxpr, consts, args, decl, path=()):
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
                # `jnp` puts most of its work behind `jit` -- `jnp.where`,
                # `jnp.clip`, `jnp.power` and the rest of the wrappers all
                # trace to a `jit` equation on both supported jax series --
                # so a replay that abstained here would abstain on most
                # real programs.  It DID, until this batch: see the
                # commentary on `_CALL_PRIMITIVES`.
                body, body_consts = _call_jaxpr_of(eqn)
                # AND THE PROPERTY THAT MEMBERSHIP CLAIMS, CHECKED AGAINST
                # THE EQUATION.  `_CALL_PRIMITIVES` means *"this body runs
                # exactly once per equation, on the equation's own
                # operands"*, and everything downstream of this descent --
                # the assume reading, the assert reading, the early stop --
                # is only a reading of the PROGRAM while that holds.  It
                # is checked by THREE opinions -- a name, an iteration
                # count, and a signature -- because neither a name list
                # nor a derivation is complete on its own, and this file
                # has now shipped each of them alone and lost something
                # both times: a deny-list that named four non-primitives
                # and not `reduce`, and then a derivation that admitted
                # `fori_loop` as a call.  The three, their traced
                # coverage table, and the shape none of them can see are
                # at :func:`_body_runs_once`.
                once, why = _body_runs_once(eqn, body)
                if not once:
                    raise ProbeInvariantViolated(
                        f"the rational replay is about to descend {name!r}, "
                        f"which `_CALL_PRIMITIVES` names as a body that runs "
                        f"exactly once per equation on the equation's own "
                        f"operands -- and {why}, so it does not. Walking "
                        f"such a body once evaluates it at the wrong "
                        f"arguments, and the replay's answer would then be "
                        f"about a program that was never executed; an "
                        f"assume inside it would be read once for a body "
                        f"that runs many times, and the `assumes_only` "
                        f"early stop would call that reading complete. "
                        f"Re-derive both before naming {name!r} in "
                        f"`_CALL_PRIMITIVES`."
                    )
                _alive.append(eqn)
                inner = run(body, body_consts, ins, decl, path + (id(eqn),))
                if len(inner) != len(eqn.outvars):
                    raise _Unreplayable(
                        f"{name!r} returned {len(inner)} value(s) for "
                        f"{len(eqn.outvars)} output(s)"
                    )
                for var, o in zip(eqn.outvars, inner):
                    env[var] = o
                continue
            if name in ("stelling_assume", "stelling_nonvacuity"):
                if name == "stelling_assume":
                    # THE EARLY STOP BELOW COMPARES TWO SETS OF ASSUME
                    # OCCURRENCES TAKEN BY THE SAME DESCENT RULE, and both
                    # halves of that sentence are load-bearing.
                    #
                    # It used to compare `len(assumes)` -- READINGS -- with
                    # `census.assumes_in_program` -- EQUATIONS, counted by
                    # the widest walk in this file, `scan` and `cond`
                    # bodies included.  Two numbers taken by two different
                    # rules, equal only while every descent this walk makes
                    # enters its body exactly once, which was an argument
                    # in a comment.  And the comparison was one-directional:
                    # one descent into a body that runs N times reads FEWER
                    # occurrences than the body has executions, and no
                    # count of readings can see an under-reading.
                    #
                    # `_assume_sites_reachable` walks the same program by
                    # the same rule and returns the occurrences this walk
                    # CAN reach.  An occurrence is (call path, equation),
                    # so the two sets are comparable key for key, and:
                    #
                    #   * a reading that is SHORT leaves `reachable -
                    #     assume_sites` non-empty and cannot fire the stop;
                    #   * a SECOND reading of one occurrence can only come
                    #     from a walk that iterated, and raises -- ON THE
                    #     `assumes_only` PATH THAT IS TRUE FOR A REPEAT
                    #     THAT LANDS WHILE THE READING IS STILL SHORT, AND
                    #     ONLY THEN.  Measured on both series, with a
                    #     `jit` body listed twice: one assume in the
                    #     program and `assumes_only=True` returns
                    #     `([True], [])`, because the stop fires on the
                    #     first reading and the second never happens; the
                    #     same program under `assumes_only=False` raises.
                    #     Put a SECOND assume after the `jit` and the
                    #     repeat lands with `reachable - assume_sites`
                    #     non-empty, and it raises on both settings.  That
                    #     is not a hole -- the stop it would have fooled
                    #     fired on a COMPLETE reading, which is exactly
                    #     what it is allowed to do -- but the claim is the
                    #     narrow one and the test drives both sides
                    #     (`test_the_replay_refuses_to_read_one_assume_occurrence_twice`);
                    #   * a reading of an occurrence the rule cannot reach
                    #     means this walk and `_assume_sites_reachable`
                    #     disagree about the descent, and raises.
                    #
                    # `ProbeInvariantViolated` is a `BaseException` and
                    # therefore is NOT absorbed by the `except Exception`
                    # in `assumes_over_the_rationals` that this reading is
                    # taken under -- which is the whole reason that class
                    # stopped being an `AssertionError`.
                    #
                    # What the stop does NOT claim is that the reading
                    # covers the program: `reachable` is what this rule
                    # reaches and `census.assumes_in_program` is what the
                    # program states, and an assume behind a `scan` is in
                    # the second and not the first.  The gate in `probe`
                    # compares the reading against the second before it may
                    # say a point was admitted, and that is the check a
                    # short reading fails.
                    site = path + (id(eqn),)
                    if site in assume_sites:
                        raise ProbeInvariantViolated(
                            f"the rational replay read the same "
                            f"`stelling_assume` occurrence twice, so the "
                            f"occurrences it has READ are no longer a "
                            f"subset of the occurrences its descent rule "
                            f"can REACH ({len(reachable_sites)} of them). "
                            f"The `assumes_only` early stop compares those "
                            f"two sets and would stop on a PARTIAL "
                            f"reading -- one iteration's assume, reported "
                            f"as the program's. Some walker in this module "
                            f"now descends a repeating construct; the "
                            f"early stop must be re-derived before this "
                            f"probe reads an assume again."
                        )
                    if site not in reachable_sites:
                        raise ProbeInvariantViolated(
                            f"the rational replay read a "
                            f"`stelling_assume` occurrence that "
                            f"`_assume_sites_reachable` says this walk's "
                            f"own descent rule cannot reach, so the two "
                            f"disagree about which bodies this replay "
                            f"enters and neither the `assumes_only` early "
                            f"stop nor the reading it completes means what "
                            f"it says. The reachable set holds "
                            f"{len(reachable_sites)} occurrence(s) and "
                            f"this is not one of them."
                        )
                    assume_sites.add(site)
                    _alive.append(eqn)
                    assumes.append(all(bool(v) for v in ins[0].reshape(-1)))
                    if assumes_only and not (reachable_sites - assume_sites):
                        raise _AssumesComplete
                env[eqn.outvars[0]] = ins[0]
                continue
            if name == "stelling_assert":
                asserts.append(all(bool(v) for v in ins[0].reshape(-1)))
                env[eqn.outvars[0]] = ins[0]
                continue
            out = _apply(eqn, prim, name, ins, raw)
            guard.check(out)
            env[eqn.outvars[0]] = out
        return [read(a) for a in jaxpr.outvars]

    def _apply(eqn, prim, name, ins, raw):
        params = dict(eqn.params)
        aval = eqn.outvars[0].aval
        out_dt = np.dtype(aval.dtype)
        guard.tick()

        if name in _EXACT_UNARY and _EXACT_UNARY[name] is not None:
            out = _ew(_EXACT_UNARY[name], ins[0], guard=guard)
        elif name in _EXACT_BINARY and _EXACT_BINARY[name] is not None:
            out = _ew(_EXACT_BINARY[name], *ins, guard=guard)
        elif name in _COMPARISONS:
            return _ew(_COMPARISONS[name], *ins, guard=guard)
        elif name in _BOOLEAN:
            _boolean_only(name, eqn, out_dt)
            return _ew(_BOOLEAN[name], *ins, guard=guard)
        elif name == "not":
            _boolean_only(name, eqn, out_dt)
            return _ew(lambda a: not bool(a), ins[0], guard=guard)
        elif name == "div":
            if out_dt.kind != "f":
                raise _Unreplayable(
                    "integer division truncates; that is not division over Q"
                )
            out = _ew(_rat_div, *ins, guard=guard)
        elif name == "integer_pow":
            y = params.get("y")
            if not isinstance(y, int):
                raise _Unreplayable("integer_pow with a non-integer exponent")
            out = _ew(
                lambda a, k=Fraction(y): _rat_pow(a, k), ins[0], guard=guard
            )
        elif name == "pow":
            out = _ew(_rat_pow, *ins, guard=guard)
        elif name == "sqrt":
            out = _ew(_rat_sqrt, ins[0], guard=guard)
        elif name == "square":
            out = _ew(lambda a: a * a, ins[0], guard=guard)
        elif name == "select_n":
            which = ins[0]
            cases = ins[1:]
            out = _ew(
                lambda w, *cs: cs[int(w)], which,
                *np.broadcast_arrays(*cases), guard=guard,
            )
        elif name == "convert_element_type":
            src_kind = np.dtype(eqn.invars[0].aval.dtype).kind
            out = _ew(
                lambda a: _rat_convert(a, src_kind, out_dt), ins[0],
                guard=guard,
            )
        elif name in _REDUCTIONS:
            if name in ("reduce_and", "reduce_or"):
                _boolean_only(name, eqn, out_dt)
            out = _reduce(
                _REDUCTIONS[name], ins[0], params.get("axes", ()), guard=guard
            )
        elif name in _MOVEMENT:
            positions = _MOVEMENT[name]
            if positions is None:
                positions = tuple(range(len(ins)))
            # A DATA operand contributes only its SHAPE here -- the values
            # travel in `ins` and are picked out by the index map, so a
            # zero array of the right shape is exactly as good as the real
            # one.  A NON-data operand is the opposite: `_movement` passes
            # it through verbatim because its VALUE is what the primitive
            # reads (a gather's start indices, a dynamic_slice's offsets),
            # and substituting zeros there would replay a DIFFERENT
            # program and could invent a refutation.  Every entry in
            # today's `_MOVEMENT` table is all-data, so this branch is
            # unreachable now; it is written as a refusal rather than as a
            # zero because the day `gather` is added is the day the zero
            # becomes silently wrong.
            raw_ins = []
            for j, (r, a) in enumerate(zip(raw, eqn.invars)):
                if j in positions:
                    raw_ins.append(np.zeros(tuple(a.aval.shape)))
                elif r is None:
                    raise _Unreplayable(
                        f"{name!r} reads operand {j} by VALUE and it is not "
                        f"a literal, so the replay cannot see what it reads"
                    )
                else:
                    raw_ins.append(r)
            out = _movement(prim, params, ins, raw_ins, positions)
        else:
            raise _Unreplayable(f"{name!r} has no exact rational reading")

        if out_dt.kind in "iu":
            _int_ok(out.reshape(-1), out_dt)
        return out

    try:
        run(census.closed.jaxpr, census.closed.consts, (), [0])
    except _AssumesComplete:
        # `assumes_only`: every assume the program states has been read and
        # `asserts` is deliberately short.  Not an error, and it is caught
        # here rather than let out because no caller should have to know
        # this walk can stop early.
        pass
    return assumes, asserts


def _confirm(census, statuses, point, k, semantics):
    """Decide whether an executed violation may be REPORTED, or is declined.

    Returns ``(detail, decline_reason, adjudication, abstention)``: exactly
    one of the first two is ``None``, ``adjudication`` always names WHICH
    test decided, and ``abstention`` is the reason the exact reading was
    unavailable when that is what happened and ``None`` otherwise.

    **ONLY AN EXACT TEST MAY ADMIT A FIRING.  EVERYTHING ELSE DECLINES.**
    That is the whole disposition of this function and it is the one thing
    in this module that must not be traded for reach.  A firing RAISES,
    and the sentence it raises with is the categorical *"stelling is
    UNSOUND at this query"*; the cost of a wrong one is that this project
    reports its own defect as the caller's, in the message nobody can
    ignore.  So the abstention of an exact test is a DECLINE, named and
    counted, and never a fall-back to a weaker one.

    **THIS SHIPPED THE OTHER WAY ROUND ONCE, AND THE PRICE IS MEASURED.**
    The first fix for the Kahan false alarm put an exact-rational replay
    IN FRONT of the ulp-stability proxy and left the proxy as the
    fall-back.  The false alarm survived it, four lines through the public
    API and with no mutation, on every route the replay happened not to
    read::

        y = any_array((), "float64", (0.0, 2.0)); s = 1e16
        z = jnp.where(y >= 0.0, (s + y) - s, y)
        assert_(z <= y)        # true over ℝ; the VERIFIED is correct
        -> FALSIFICATION PROBE FIRED, admitted by 'ulp-proxy-refutes'

    and again with ``y + (y ** 0.5) * 0.0`` on the right, and it would
    have again through ``exp``, ``sort``, ``cumsum``, ``stack``, ``rem``,
    a non-square ``sqrt``, ``scatter`` or ``dot_general`` -- i.e. through
    every matmul -- because EVERY primitive the replay abstains on was a
    route back to the test the replay was added to replace.  An instrument
    is not made safe by putting a correct adjudicator in front of an
    unsafe one; it is made safe by removing the unsafe one from the
    firing path.

    The ulp proxy is therefore gone rather than demoted.  It is not kept
    as a decline filter either, because it has nothing left to do: an
    abstention already declines, so the proxy could only ever have
    converted a decline into a decline.  ``_step`` survives because the
    ``ulp`` SAMPLING strategy uses it to choose points, which is a
    different job and cannot admit anything.

    **AND ALL THREE ARE DOWNSTREAM OF ONE PRECONDITION THAT IS NOT STATED
    HERE.** Every branch below says "the violation stands", and a
    violation only means anything at a point the analysis CLAIMED
    something about — i.e. one admitted by every assume. That gate is in
    ``probe.run_one``, and for the first two branches it is the only
    reading of the assumes there is: they return before the replay is
    consulted, so nothing downstream re-checks. It was reading a list
    ``_execute`` fills at the TOP LEVEL only, so an assume one ``jit`` deep
    left it empty and the gate admitted everything — the fourth of this
    module's five defects, see the module docstring. The gate now declines
    (``assume-not-fully-executed``) unless the executed run saw every
    assume the program contains at every depth, which is what makes the
    two early returns below safe.

    **AND THE ``ieee`` RETURN HAS A SECOND SUCH PRECONDITION, FOR THE SAME
    REASON ONE LEVEL DOWN.** That gate is a reading of WHETHER the
    executed run saw everything; this one is a reading of WHICH float it
    saw, and it is in ``probe.run_one`` beside it. See the ``ieee`` bullet
    below.

    THE THREE TESTS THAT MAY ADMIT, in the order they are consulted:

    * **``ieee`` semantics.** The executed float IS the subject of the
      claim, so the violation stands exactly as executed.

      **AND WHICH FLOAT WAS EXECUTED IS ITSELF A CHOICE, WHICH IS THE
      FIFTH INSTANCE OF THE CLASS.** ``_execute`` binds one equation at a
      time, so XLA never sees two of them together, while the caller's own
      call compiles whole regions — and ``jax.make_jaxpr`` inlines the
      ``jit`` that ``jnp.mean`` is built out of, so this is reached with no
      ``jit`` written anywhere. The sentence above then licensed a firing
      about a float the program does not compute, on a correct VERIFIED,
      in four lines. The precondition is in ``probe.run_one`` with the
      other readings of the executed run: the same program is run at the
      same point as ONE compiled region, and a violation whose truth value
      moves between the two routes declines
      (``executed-float-depends-on-granularity``). See :func:`_execute`
      for the measurement and :func:`_whole_program_route` for the route.

      **AND AN ``ieee`` VERDICT IS NOT ALWAYS STELLING'S OWN CLAIM, WHICH
      IS WHERE THIS BRANCH'S REMAINING DEFECT WAS.** It returns before any
      exact test, and an ``ieee`` discharge can rest on a caller-declared
      ``libm_budget`` that the stamp itself marks *"DECLARED, NOT
      VERIFIED … TWO claims compose to make this verdict and stelling
      checks NEITHER."* Under-declare it and this branch produced a real
      counterexample under a false attribution: *"stelling is UNSOUND at
      this query"*, raised into the caller's CI, for the caller's own
      declaration. Driven, public API, no mutation, at ``exp``/float32
      and ``X = 88.72167205810547`` — the shipped profile
      ``"xla-cpu-2026-08"`` gives UNKNOWN on the same harness, so the
      VERIFIED existed only because of the declaration. The
      **counterexample is kept and the ATTRIBUTION is fixed**, in
      :func:`_fire`: a firing on an assumption-conditioned verdict names
      the declaration and says the probe cannot tell a false declaration
      from an unsound analysis. Nothing is declined — the caller has just
      been handed the point at which to check their own profile, which is
      the most useful thing this instrument could give them. See
      :data:`DECLARED_NOT_VERIFIED`.
    * **an all-integral PROGRAM.** The arithmetic is exact as executed
      and no rounding is involved.  This branch MUST NOT become a rational
      replay: rational arithmetic does not wrap, so replaying an ``int8``
      program would report values it never computed and would declare a
      genuine runtime wrap a rounding artefact.  Measured: with
      ``propagate._int_guarded`` removed the probe catches an ``int8``
      runtime wrap, and it catches it through this branch.

      **THE PREDICATE IS READ OFF THE PROGRAM AND IT USED TO BE READ OFF
      THE DECLARATIONS**, which is the same defect as the Kahan false
      alarm and the fall-back that followed it -- something cheap standing
      in for an exactness claim -- for the third time in this module.  An
      ``int16`` declaration cast to ``float32`` rounds, and four lines
      through the public API called a correct ``VERIFIED`` ``UNSOUND``
      under a parenthetical saying no rounding was involved.  The rule the
      three instances share: **a predicate that licenses an exactness
      claim is computed from the object the claim is about.**  See
      :func:`_integral_program`.

      **THERE WAS A FOURTH, AND IT ADDED THE OTHER HALF OF THE RULE.**
      Its predicate WAS computed from the program -- it was computed from
      part of it, a ``stelling_assume`` walk that stopped at the top
      level.  So: **and the computation must read all of it.**  Every
      quantity is now checked against a census taken at every depth
      (:data:`_READINGS`), and this branch is reachable only when that
      check passed.

      **AND WHAT THIS BRANCH RESTS ON THAT IS NOT IN THIS FILE.** Its
      sentence is *"no rounding is involved"*, and that is a statement
      about arithmetic, not about WRAPPING: an all-integral program that
      wraps computes a value ℤ does not contain, and this branch admits a
      firing on it — correctly, because the wrap is what the program does.
      What stops that from being a false alarm on a CORRECT VERIFIED is
      that the analysis refuses to discharge an integer obligation whose
      arithmetic could wrap, and that refusal is
      ``propagate._int_guarded`` — **a function in a module this file may
      not import, so nothing inside ``falsify`` would notice if it
      regressed.** Measured, in the direction that shows the coupling:
      with ``_int_guarded`` removed, ``x + y >= 0`` over ``int8 [0,
      100]²`` becomes a false VERIFIED and this branch catches it. The
      same removal in the other direction — a guard that stopped
      refusing — would make this branch fire on verdicts stelling should
      never have minted, and it would look identical from here. That
      dependency is named here because it cannot be checked here; what
      checks it is ``propagate``'s own tests.
    * **exact-rational replay** of the same traced jaxpr at the same point
      (:func:`_replay`), for everything else.  False over ℚ as well as in
      floats: the analysis discharged something false about ℝ and the
      firing stands.  True over ℚ: the violation was manufactured by
      rounding and is declined.

    And every other outcome -- an irrational step, a primitive with no
    rational reading, an integer result that left its dtype, a value wider
    than the replay's budget, a point outside the assumed region over ℚ --
    is a DECLINE with the reason recorded.  What that costs in reach is a
    real number and it is in the module docstring, not hidden here.
    """
    detail = (
        f"the obligation evaluated FALSE at this point; the declared box, "
        f"every assume, and the obligation itself were all evaluated by "
        f"executing the program"
    )
    if semantics == "ieee":
        return detail, None, "ieee-executed-float", None
    # BOTH CONJUNCTS, AND THE SECOND IS REDUNDANT TODAY.  `census.integral`
    # is read off the PROGRAM -- every operand and result dtype at every
    # depth (:func:`_integral_program`) -- and is what licenses the
    # sentence below.  The declaration test was written as a second,
    # different fact: the POINT handed to the executed run is built out of
    # `Declaration.dtype` by `_window`/`_clean`, so a declaration that is
    # not integral would be an integral value entering by a float door.
    # It cannot be: the `stelling_any` outvar's aval dtype IS
    # `Declaration.dtype` -- the same param the equation carries -- checked
    # across ten dtypes, so `census.integral` already IMPLIES this
    # conjunct.  Kept because it costs nothing and is the conjunct a reader
    # checks by hand, and DESCRIBED accurately because a comment claiming
    # two independent facts where there is one is how a guard gets deleted
    # by someone who finds the other one.
    integral = census.integral and all(
        np.dtype(d.dtype).kind in "iub" for d in census.declarations
    )
    if integral:
        return (
            detail + " (exact integer arithmetic: no rounding involved)",
            None,
            "exact-integer-arithmetic",
            None,
        )

    try:
        assumes, asserts = _replay(census, point)
    except _Unreplayable as exc:
        why = str(exc)
    # AND NOT `ProbeInvariantViolated`, which is a `BaseException` and
    # passes.  Driven on the tree that shipped it as an `AssertionError`:
    # the stamp line printed *"8 x ProbeInvariantViolated: ..."* among
    # `ProbeReport.abstentions`, where every other entry is a primitive
    # this replay cannot read -- a broken invariant filed as a reach cost.
    except Exception as exc:  # noqa: BLE001 - an abstention, never a firing
        why = f"{type(exc).__name__}: {exc}"
    else:
        if len(asserts) != len(statuses):
            why = "the rational replay saw a different number of obligations"
        elif len(assumes) != census.assumes_in_program:
            # DEFENCE IN DEPTH, and the same check as `run_one`'s applied
            # to the OTHER walker.  `_replay` descends `_CALL_PRIMITIVES`
            # where `_execute` binds them, so the two walkers read the
            # program at different depths -- which is the gap four of
            # this module's five defects have lived in.  Neither reading is
            # trusted to be complete because of how it is written; each is
            # checked against `census.assumes_in_program`, which is counted
            # at every depth there is.
            why = (
                f"the rational replay reached {len(assumes)} of the "
                f"program's {census.assumes_in_program} assume(s)"
            )
        elif assumes and not all(assumes):
            # The point satisfies the assume in floats but not over ℚ, so
            # over ℝ it is outside the assumed region and a violation there
            # refutes nothing.  Declining is the conservative side.
            return (
                None,
                "assume-unsatisfied-over-the-rationals",
                "exact-replay-outside-the-assumed-region",
                None,
            )
        elif asserts[k]:
            return (
                None,
                "float-rounding-artefact",
                "exact-replay-holds-over-the-rationals",
                None,
            )
        else:
            return (
                detail
                + " (and exact-rational replay of the same traced jaxpr at "
                "the same point makes it FALSE over ℚ, so this is not a "
                "rounding artefact)",
                None,
                "exact-replay-refutes-over-the-rationals",
                None,
            )

    # NO EXACT READING, SO NO FIRING.  The reason travels with the decline
    # (`ProbeReport.abstentions`) instead of being folded into a firing
    # message, because it is the number that says how much of this
    # instrument's reach the exactness requirement costs on this program.
    return (
        None,
        "no-exact-reading-of-this-program",
        "declined-no-exact-reading",
        why,
    )
