# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE DIFFERENTIAL FLOAT ORACLE: is the executed value inside the box?

``stelling`` judges obligations by interval arithmetic and stamps VERIFIED.
Every one of those stamps rests on one sentence:

    **the value the compiled program computes lies inside the box the
    propagator computed for it.**

Nothing in ``src/`` asserts that sentence and nothing in ``tests/`` asserted it
before this module. It is FALSE on this tree, in NINE measured places — the
seven this module was commissioned to pin (``_float_oracle.SEVEN``) plus two
it found itself. **Eight of them are false against ``v0.1.0``'s own ``src/``
too, and five of those falsify a discharge there**, which is re-derived at
:data:`_float_oracle.MEMBERS` rather than quoted. This module runs the
propagator and the compiled program over the same query and compares them
value by value.

**AND IT RUNS THE PROGRAM TWICE.** The op-by-op walk is a granularity — the
TRACE's, not the program's — so an equation whose operands are all
compile-time constants is executed at runtime, where the backend flushes a
subnormal, instead of being constant-folded in full precision the way the
compiled program folds it. Every candidate violation is therefore re-checked
against the same jaxpr staged into ONE ``jax.jit``, and one the compiled
program does not have is declined rather than reported: **9 of 572 candidates
in a 1500-example run** (``route_declined`` 9 against 563 reported; the figure
here read "76 of 639" and was quoted from the ``03b2dbe`` run, whose partition
the same file already called superseded — see :data:`FLOAT_ORACLE_MEASURED`).
That second route follows
``stelling.falsify._whole_program_route``, including the two properties that
make it safe — it is consulted only after a violation, and it can only
DECLINE.

**THE NEAREST INSTRUMENT THAT DOES EXIST, AND WHAT IT DOES WITH THESE EIGHT.**
``stelling.falsify.probe`` — ``check(..., falsify="sample")``, default-off and
``experimental`` — also executes the program at concrete points. It asks a
different question: *is the OBLIGATION false here*, not *is each value inside
its box*. Driven over all NINE members plus ``OVERFLOW_PROBE``: **it reports
none of the ten.** Two are UNKNOWN so it never runs (it fires only after a
VERIFIED); one is REFUTED; seven are VERIFIED and stay VERIFIED. For five of
those seven it SAW the executed violation and declined it — 28 points on
``ftz-subnormal-sum``, 12 on ``f32-underflow``, 2 on ``f32-single-multiply``
and 2 on ``assume-narrows-past-the-program`` as ``float-rounding-artefact``
(``exact-replay-holds-over-the-rationals``), and 58 on ``f32-exp`` as
``no-exact-reading-of-this-program`` — and its note says so in its own words:
*"28 EXECUTED VIOLATION(S) WERE DECLINED, NOT ABSENT"*. That decline rule is
the ℝ defence in instrument form, disclosed rather than hidden. The remaining
two produce no obligation violation at all, because the box violation is one
equation UPSTREAM of the obligation: ``1/Σ(…)`` executes ``inf > 0`` and the
float32 square executes ``inf > 0``, both TRUE. A box-containment oracle sees
a defect one equation before an obligation oracle can. (The ninth member was
driven separately when it was pinned: 72 points executed, 63 admitted, none
reported.)

**THE ARGUMENT THAT KEPT THIS OUT IS RETRACTED, AND IT IS RETRACTED IN
``tests/property/README.md`` WITH WHAT IT SAID.** That file scoped the
execution oracle to integers on the ground that *"an oracle pointed at floats
measures the documentation"* — a float harness may be correctly VERIFIED in ℝ
and violated by IEEE, which is the declared posture. The premise is true and
the conclusion does not follow: a VERIFIED that the compiled program
contradicts at a point of its own declared box is unacceptable whatever ℝ
says, and the nine programs pinned below are exactly that. What the ℝ posture
buys is that such a verdict is DISCLOSED, not that it is harmless.

────────────────────────────────────────────────────────────────────────────
THE TWO LEGS, AND WHY THERE ARE TWO
────────────────────────────────────────────────────────────────────────────

``test_the_executed_value_lies_inside_the_computed_box`` is the property in
full, and it **FAILS on ``main`` today**: the class is open, this generator
reaches it densely, and the marker is
``xfail(strict=True, raises=ExecutedValueOutsideBox)``. Three things about
that marker, each deliberate and each copied from ``test_oracle.py``'s wrap
leg, which is the same shape of amnesty for the same kind of reason:

* **strict** — a non-strict xfail passes silently the day the class is
  repaired, which is the silent-success shape this whole suite exists to
  prevent. Strict means the suite goes RED when the remedy lands and somebody
  has to come here and delete the marker;
* **raises=** — the amnesty covers a box violation and nothing else. A census
  floor failure, an import error and a break in the transcription coupling all
  arrive as some OTHER exception and are reported as real failures. Measured
  rather than argued, on a copy of this tree with ``_float_oracle.boxes_of``
  planted to ``return []`` so the instrument reads no boxes at all:

  =====================  =====================================================
  ``raises=`` deleted    ``1 xfailed`` — **green**, on a floor it did not meet
  ``raises=`` in place   ``1 failed`` — ``GENERATOR FLOOR NOT MET``,
                         ``finite_boxes=0 < 100``, ``boxes_read=0`` over 80
                         drawn programs
  =====================  =====================================================

  One deleted keyword between a floor that holds and a floor that cannot fail.
  ``test_suite_disclosure.py`` asserts the keyword is there, statically,
  because no log-reader can see it go;
* the reason names the open class, so a CI log says what is not being checked
  rather than leaving a reader to infer it from a green tick.

``test_every_violation_it_finds_has_a_known_ieee_cause`` is the residual, and
it **PASSES**. It is the leg that keeps the xfail honest: every violation the
same search finds is classified, and a violation with none of the five IEEE
explanations — an ``UNEXPLAINED`` one — is a defect in the interval domain
itself rather than a rediscovery of the float posture, and fails this leg. It
carries every pinned member as an ``@example`` and a per-member census
floor, so **"all of them are still found" is an assertion inside the property
body**, not a sentence in a docstring. A member that still violates for a
DIFFERENT reason fails it too — that is the drift check in the body, and it is
separate because a floor counts violations and cannot see a cause.

────────────────────────────────────────────────────────────────────────────
THE PARTITION — MEASURED, AND NOT THE SAME AS A COUNT
────────────────────────────────────────────────────────────────────────────

A violation is not automatically a defect. Under ``semantics="real"``'s
stamped disclaimer every one of these is permitted; under the ruling this
module is built on every one is a member of the class. Either way the useful
report is the partition by cause, not the total, and the partition is in the
residual leg's census on every run. The figures this module was landed with
are in :data:`FLOAT_ORACLE_MEASURED`, dated and anchored to a commit and to a
command, because a present-tense count rots and a count nobody can re-derive
is the same defect as a check that does not exist.

────────────────────────────────────────────────────────────────────────────
WHAT A GREEN RUN OF THE RESIDUAL LEG DOES NOT SAY — READ THIS FIRST
────────────────────────────────────────────────────────────────────────────

The three blindnesses are implemented in ``_float_oracle`` and stated in its
module docstring in full. In one line each, here, where a reader of a null
result will meet them:

1. **⊤ is unfalsifiable, and it is not the only bucket that is.** A declined
   equation's box admits every value its dtype can take, so this instrument is
   blind exactly where the propagator already declined. The pass rate is
   **not** a safety signal. The census counts boxes in FOUR buckets — ⊤,
   EMPTY (a size-0 declaration: no element to violate anything), INTEGER
   (never compared, because a binary64 box endpoint cannot represent an
   ``int64`` above 2**53) and compared — and both properties floor on the last
   alone, restricted to the unbiased generator. The percentages are in
   :data:`FLOAT_ORACLE_MEASURED` with the run that produced them; **most of
   the field of view is blind**, for three different reasons, and that figure
   has been wrong twice in the same direction. It read 24 % blind until the
   integer bucket was counted apart, then 39–41 % until an audit found that
   ``_float_oracle.is_top`` tested ``±inf`` and so missed the BOOL lattice's
   own top — a box of ``[0, 1]`` on a bool output, which every declined
   comparison in this grammar produces and which no executed bool can fall
   outside of.
2. **The sampler's grid.** Only dtype-representable points strictly inside the
   declared box are executed, snapped inward with ``nextafter``. A declaration
   whose box holds no value of its own dtype is reported ``unsampleable`` and
   contributes nothing — which is a refusal, not a null result.
3. **What the generator cannot reach.** The unbiased grammar cannot build the
   reassociation class at all (its shapes top out at four elements, and the
   split needs 33), and the deliberate strategy that can build it builds
   nothing else. Neither reaches ``scan``/``while``/``cond``, ``vmap``,
   ``grad``, the solver legs, the affine refinement, ``semantics="ieee"``, or
   any dtype the sampler cannot construct.

A FOURTH, and it is about this file rather than about the search:
**``UNEXPLAINED`` is a proof and it needs an exact reading to make.** Where
:func:`_float_oracle.exact_value` has no entry for a primitive — ``sqrt``,
``exp``, ``sign``, ``integer_pow``, every reduction but ``reduce_sum`` — no
rule can prove anything, and a green residual leg says only *no violation was
PROVED to be a box wrong about ℝ*.

**AND ``unclassified`` IS NOT THE MEASURE OF THAT, WHICH IS WHAT THIS
PARAGRAPH USED TO SAY.** It read *"the ``unclassified`` count is how much of
the field that proof did not cover. It is 0 in the shipped run."* Zero is
true and it measures the wrong thing: a violation an UNPROVED rule names is
not unclassified, it is classified on no evidence. The census carries
:data:`_float_oracle.BASES` — proved / sound-by-construction / heuristic, with
the run's own counts in :data:`FLOAT_ORACLE_MEASURED` — and the heuristic row
is the honest answer to "how much did the proof not cover".

**AND ``sound-by-construction`` WAS OVERSTATED, WHICH IS THE SAME MISTAKE ONE
COLUMN LEFT.** That basis means the rule needs no ℝ reading to be right, and
three rules claimed it while making a claim about ℝ: ``overflow-to-inf``
(*no finite box contains an infinity* is structural, but *an overflow of a
box that is RIGHT about ℝ produced it* is not), ``assume-narrowing`` (decided
by comparing two boxes, which cannot tell a narrowing that is right about ℝ
from one that is wrong), and the operand-NaN rule (which scanned every element
of every operand rather than the ones the element under judgement depends on).
All three are now proved where they can be and heuristic where they cannot;
the ``sound-by-construction`` row shrinks by exactly the amount the ``proved``
row grows.

────────────────────────────────────────────────────────────────────────────
POSITIVE CONTROLS
────────────────────────────────────────────────────────────────────────────

* containment leg — fails at ``main`` (``at="HEAD"``). Registered as
  ``float-oracle-box``.
* residual leg — fails under the two-corner ``interval.mul`` mutant, which
  makes a box that is simply WRONG about ℝ rather than about IEEE, so the
  violation classifies as ``UNEXPLAINED``. Registered as
  ``float-oracle-unexplained``.

Run them: ``python tools/property_check.py --controls``.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("hypothesis", reason="needs hypothesis")
jax = pytest.importorskip("jax")

from hypothesis import example, given  # noqa: E402

import _float_oracle as fo  # noqa: E402
import _grammar  # noqa: E402
import _profiles  # noqa: E402
import _runner  # noqa: E402

FLOAT_BOX_DEFECT = (
    "open class: the executed value falls outside the box the propagator "
    "computed, in nine pinned programs. Eight of them do it against v0.1.0's "
    "own src as well, and five of those falsify a discharge there. This "
    "marker is strict and narrowed by raises=: the day a remedy lands this "
    "test XPASSes and the suite goes red."
)

#: THE PARTITION, as a dated record rather than a present-tense claim, and
#: with the column that decides a repair beside the one that does not.
#:
#: Measured 2026-08-29 at ``4c8c835`` — jax 0.11.0, jaxlib 0.11.0, numpy
#: 2.5.2, CPython 3.12.3, hypothesis 6.165.10, pytest 9.1.1,
#: ``JAX_PLATFORMS=cpu``, x64 forced on by this module's fixture — by running
#: THIS FILE's residual leg at ``STELLING_PROPERTY_SCALE=12.5``, i.e. 1500
#: examples, and reading its own census. Re-derivable by anyone with that tree
#: and that command. ``4c8c835`` is the commit BEFORE this one, and this
#: commit changes only the sentence you are reading: a commit cannot contain
#: its own hash, and a sha typed at a guess is the defect the paragraph below
#: records. That is the second time this branch has split a commit for it, and
#: it is still cheaper than the defect.
#:
#: **AND THE TABLE IS UNCHANGED FROM THE ONE MEASURED AT ``eb86def``**, which
#: is asserted rather than assumed: the fix to ``_exact_apply``'s conversion
#: entry is a correction to the exact reading, and the way to know whether a
#: correction to the reading moves the partition is to run the command again
#: and diff the census. Done — **identical, every key**. The conversion defect
#: was PLAUSIBLE and not live: of 573 traced draws of
#: :func:`_float_oracle.uniform_float_programs`, 271 carry an ``assume``, 8
#: carry a float→int cast, 4 carry one inside an assume's dependency cone, and
#: in all 4 the walk refused for another reason.
#:
#: **THE ANCHOR ITSELF WAS WRONG AND THAT IS WHY THIS SENTENCE IS SO CAREFUL.**
#: It read *"Measured 2026-08-28 at ``03b2dbe``"* while the table it introduced
#: reproduced only at the commit AFTER it. Re-derived 2026-08-29 by
#: ``git archive 03b2dbe src tests`` into a scratch tree and running the same
#: command there: **13727 boxes, 8411 compared, 95 falsified discharges, 76
#: route declines, 537 obligation readings taken by both routes, 17 of them
#: disagreeing**, and 563 violations as 316 nan / 175 flush / 29 narrow /
#: 27 reassoc / 16 overflow. That is the run this file called superseded two
#: paragraphs below while pointing its own anchor at it.
#:
#: **AND TWO FIGURES FROM THAT RUN SURVIVED IN THE PROSE AROUND THE TABLE**,
#: 250 lines from a census that contradicts them: *"76 of 639 candidates"*
#: declined by the second route (76 + 563 = 639 — correct AT ``03b2dbe``, and
#: 9 of 572 on the tree that shipped it), and *"17 of the 537 obligation
#: readings … it is why this cannot be left on the eager reading"* (537 and 17
#: at ``03b2dbe``; 420 and **0** on the tree that shipped it, so the argument
#: was resting on a number that had gone to zero). Neither was wrong when it
#: was taken. Both were left behind by the tree, which is exactly what a
#: present-tense figure does and exactly what the anchor exists to prevent.
#: Both are corrected where they stood.
#:
#: **EVENTS ARE THE WRONG DENOMINATOR AND THE RANKING INVERTS ON THEM.** 563
#: events stand on 90 distinct programs and 110 distinct sites, and one pinned
#: ``@example`` re-drawn contributes as many events as it is re-drawn. By
#: events ``flush(232) > nan(162) > narrow(83) > reassoc(46)``; by distinct
#: programs ``nan(31) > reassoc(30) > flush(13) > narrow(10)``. **A repair
#: prioritised on 232-vs-46 would be prioritised on how often an example was
#: re-drawn** — reduction reassociation is the second largest class by
#: programs and the fourth by events. Both columns ship.
#:
#: **AND THE DISTINCT COLUMN IS STILL NOT A DENOMINATOR UNTIL IT IS CROSSED
#: WITH THE STRATEGY.** This is the correction this round exists for. The
#: three generators have nothing in common but this census: an unbiased
#: grammar, a strategy aimed at ONE known class whose own docstring says it
#: "finds that class and nothing else", and a fixed list of pinned
#: ``@example``s re-drawn as often as Hypothesis feels like re-drawing them.
#: Round 2 crossed the CAUSE partition with them and left the two figures a
#: repair is actually scoped on uncrossed. Crossed:
#:
#:   * **"116 violations are a discharge falsified, over 21 DISTINCT
#:     PROGRAMS"** — the sentence every consumer of this module quotes as the
#:     denominator — is **100 events over 5 programs from the pinned
#:     ``@example``s**, **16 over 16 from the aimed cancelling-sum strategy**,
#:     and **0 from the unbiased leg**. The 16 are **16 draws of ONE aimed
#:     construction**, which is what the strategy's own docstring says it
#:     builds, and that is the whole argument. A repair team reading "21
#:     independent situations" is reading **six**;
#:
#:     THIS READ "SHRINK NEIGHBOURS" AND THE DISTRIBUTION SAYS OTHERWISE. All
#:     16 constant vectors are DISTINCT; 15 of the 16 share a vector length
#:     and 13 a pinned envelope, and of the 69 pairs sharing both the
#:     differing-element histogram is ``{1: 1, 2: 6, 3: 1, 4: 2, 9: 2, 10: 1,
#:     11: 2, 12: 2, 13: 3, 15: 2, 30: 2, 31: 1, 32: 20, 33: 24}`` — so 10
#:     pairs differ in at most 4 of their 33 elements and **47 differ in 30 or
#:     more**. The four close numbers were right and the word over them was
#:     not: these are independent draws of one template, not shrunk copies of
#:     one program. The conclusion does not need the distribution — it follows
#:     from the strategy finding one class and nothing else — and the
#:     distribution ships as the secondary observation it is;
#:   * **"by distinct programs reassoc(30) is second"** rests on **29 draws of
#:     the aimed strategy**, against a leg this module PROVES cannot reach the
#:     class at all — the residual leg asserts that premise against
#:     ``_grammar.SHAPES`` on every run now. The only rows in the table that
#:     are a statement about the TOOL rather than about a strategy someone
#:     aimed are the unbiased column: nan 30, narrow 8, flush 7, overflow 7,
#:     reassoc 0.
#:
#: **AND SIX SITUATIONS IS NOT SIX PIECES OF WORK — THREE OF THEM ARE ONE
#: REPAIR.** "Six" is true of the PROGRAMS and false of the effort, which is
#: the next thing a repair team asks. Derived from
#: :data:`_float_oracle.MEMBERS`' own cause column rather than typed:
#:
#:     flush-or-subnormal       3   ftz-subnormal-sum, f32-underflow, f32-exp
#:     narrow-format-rounding   1   f32-single-multiply
#:     assume-narrowing         1   assume-narrows-past-the-program
#:     reduction-reassociation  1   the cancelling-sum construction
#:
#: **Four distinct causes over the six situations, and the largest of them is
#: three of the six.** That is the single most useful line for whoever scopes
#: the work, and it is one column away from the figure everybody was quoting.
#:
#: The unbiased leg's zero in the discharge column is NOT floorable — a floor
#: on zero is not a floor — so it is written into the table as an explicit
#: cell rather than left absent, because an absent cell reads as evidence of
#: absence. What IS floored is ``distinct/falsified/member``, at
#: ``len(_float_oracle.DISCHARGE_FALSIFYING)``: that figure is 5 at the ``ci``
#: profile and 5 here, where the uncrossed ``distinct/falsified`` is 5 and 21.
#: The crossed one is a fact about the tree; the uncrossed one is a fact about
#: the budget.
#:
#: **THE PARTITION THIS FILE SHIPPED AT ``03b2dbe`` IS SUPERSEDED.** It read
#: 563 as 316 nan / 175 flush / 29 narrow / 27 reassoc / 16 overflow. Four
#: things moved it: an unproved physical-band rule was preempting a provable
#: ``UNEXPLAINED``; the sampler executed at ±inf, a point that is not a member
#: of an unbounded declared set, which manufactured 51 NaN rows; the
#: assume-narrowing class was being counted as arithmetic; and the refutation
#: half of the obligation count was not element-aligned. The **ranking
#: survives** on the column that matters.
#:
#: **AND THE PARTITION AT ``100679f`` IS SUPERSEDED BY THIS ONE**, in three
#: places and by three fixes rather than by a re-run. The bucket table moves
#: because ``is_top`` now counts the bool lattice's own ⊤ as declined
#: (compared 7719 -> 5491, TOP 1787 -> 4015, and the disclosure everywhere
#: goes from "39 %/41 % blind" to 58 %; measured with that function
#: instrumented to report the split it was hiding, the 7719 were 2228 bool
#: ⊤, 1777 definite bool and 3714 float). The evidence column moves because
#: three rules that claimed ``sound-by-construction`` were making a claim
#: about ℝ (proved 303 -> 344, sound-by-construction 203 -> 162, heuristic
#: unchanged at 57). And ONE LIVE VIOLATION CHANGES CAUSE: a draw of
#: ``assume-narrows-past-the-program`` at ``x0 = 5e-324`` — a point that
#: satisfies its own ``assume`` EXACTLY over ℝ — was reported
#: ``assume-narrowing`` and is ``flush-or-subnormal`` now, which is why the
#: flush row is 232 and not 231 and its distinct column 13 and not 12.
FLOAT_ORACLE_MEASURED = """\
1511 programs drawn, 1129 read, 46.2 s (31 ms/example, load average 2.4)
 382 not read: 108 ValueError, 79 TypeError, 50 OverflowError and 1
     ZeroDivisionError at the declaration door; 95 whose assumes admit none of
     the sampled points; 29 unsampleable; 20 UnsatisfiableAssumptionError

13041 boxes read, in four buckets:
     5491  compared  (42 %)  where an executed value can be caught outside a box
     4015  TOP       (31 %)  the analysis declined. A ⊤ box admits every value
                             its dtype can take: [-inf, inf] for a float,
                             [0, 1] for a bool. Only a NaN escapes one
     1948  EMPTY     (15 %)  a size-0 declaration: no element to violate
     1587  INTEGER   (12 %)  never compared; a binary64 box endpoint cannot
                             represent an int64 above 2**53
                             --> 58 % of the field of view is blind, for three
                                 different reasons, and only the first of them
                                 is the analysis having declined

563 VIOLATIONS                      events   distinct programs
    flush-or-subnormal                 232                  13
    nan                                162                  31
    narrow-format-rounding              83                  10
    reduction-reassociation             46                  30
    overflow-to-inf                     25                   8
    assume-narrowing                    15                   1
    UNEXPLAINED                          0                   0
    unclassified                         0                   0
                                       ---                 ---
                                       563     90 programs, 110 sites

BY EVIDENCE, which "0 unclassified" was standing in for:
    344  proved                 an exact rational reading of the equation
                                backed the cause AND placed the box inside ℝ
    162  sound-by-construction  a NaN is in no box; an operand this element
                                depends on is not a real number. No reading
                                needed, and three rules claimed this and
                                needed one
     57  heuristic              a rule that names a plausible cause and proves
                                nothing, reachable only where no exact reading
                                of the primitive exists (exp, sqrt, sign)

BY STRATEGY, WHICH IS WHERE THE HEADLINE COMES APART
                            drawn   read  violations   FALSIFIED DISCHARGES
                                                        events  distinct progs
    uniform (unbiased)        773    391         255         0          0
    cancelling-sum (aimed)    607    607          29        16         16
    pinned members            129    129         276       100          5
    pinned probes               2      2           3         0          0
                                                          ----       ----
                                                           116         21
    The 16 cancelling programs are 16 draws of ONE aimed construction.
    "21 distinct programs" is six independent situations -- and those six
    carry FOUR causes, three of them flush-or-subnormal, so the six are
    four pieces of work and the largest is half of them.

DISTINCT PROGRAMS, BY CAUSE AND BY STRATEGY
                           uniform  cancelling  member  probe     all
    nan                         30           0       1      0      31
    reduction-reassociation      0          29       1      0      30
    flush-or-subnormal           7           0       6      0      13
    narrow-format-rounding       8           0       2      0      10
    overflow-to-inf              7           0       0      1       8
    assume-narrowing             0           0       1      0       1
    any cause                   51          29       9      1      90
    The uniform column is the only one that is a statement about the tool.
    Its reassociation cell is 0 BY PROOF, not by under-sampling: SHAPES tops
    out at 4 elements and jnp.sum splits into two windows at n >= 33.

2180 admitted points; 0 sampler artefacts (a violation on `stelling_any` in a
     harness whose assumes did not narrow that declaration)
   9 of 572 candidates DECLINED by the second (compiled) route; 0 times it
     was unavailable
 420 obligation readings taken by BOTH routes, 0 of them disagreeing
  10 violations are the other direction: a "violated-over-set" that executed
     TRUE
"""


class ExecutedValueOutsideBox(AssertionError):
    """The compiled program computed a value the propagator's box excludes.

    A distinct type so that ``xfail(raises=...)`` can excuse this class and
    nothing else — a census floor, a coupling break or an import error is a
    different exception and is reported as a real failure.
    """


@pytest.fixture(autouse=True)
def _x64():
    """Judge at x64-on, the posture the project's own docs use.

    README's Quickstart line 2 enables it and 13 of 20 ``corpus/supply``
    harnesses set it. It also decides what ``float64`` even means here: with
    x64 off, jax silently makes every ``float64`` declaration a float32 array
    and this instrument would report the demotion as a narrow-format
    violation of every f64 program in the suite. Saved and restored so this
    module cannot change what the rest of the session measures.
    """
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


#: A float32 square that overflows to ``inf`` through a FINITE binary64 box.
#:
#: NOT one of the pinned members, and pinned here for one reason:
#: ``overflow-to-inf`` is a cause :func:`_float_oracle.classify` can name and
#: the unbiased grammar reaches only by luck. Measured two ways and they do not
#: agree, which is the whole argument: one 1500-draw derandomised run of
#: :func:`_float_oracle.uniform_float_programs` produced **0** of them and
#: another produced **7**. Hypothesis's derandomised seed is a function of the
#: driving test's own source, so a cause floor that relied on the search
#: reaching this class would pass or fail on an edit to an unrelated line.
#:
#: It is rare because reaching it needs a NARROW-format declaration whose
#: square leaves that format's range while staying inside binary64's: the
#: float pool's large entries (``1e300``, ``1.797e308``) are outside float32
#: altogether, so the sampler refuses those declarations as ``unsampleable``
#: before any arithmetic happens, and only a bound near a format's own maximum
#: works.
#:
#: Measured 2026-08-28 at ``a90862b``: ``mul`` executes ``inf`` against the box
#: ``[9.999999999999999e+75, 9.000000000000001e+76]``.
OVERFLOW_PROBE = fo.Program(
    (_grammar.Decl("x0", (), "float32", 1e38, 3e38),),
    (_grammar.Stmt(
        "assert",
        ("cmp", ">", ("bin", "mul", ("var", "x0"), ("var", "x0")),
         ("const", 0.0)),
    ),),
    "overflow-probe",
    "probe",
)

#: ``x*x`` over ``float64 [-1, 1]``, pinned, with no violation on a clean tree.
#:
#: The residual leg's positive control in one harness: with interval
#: multiplication keeping only the two SAME-CORNER products — the registered
#: ``float-oracle-unexplained`` mutant — the box of ``x*x`` is ``[1, 1]``,
#: while the sampler executes ``x = 0`` and gets ``0.0``. That violation is at
#: float64, finite, not a reduction and nowhere near the subnormal band, so it
#: classifies ``UNEXPLAINED`` and the leg fails. On the clean tree the box is
#: ``[-1, 1]``, it contains ``0.0``, and there is no violation at all — which
#: is what lets this example ship as a passing pin.
#:
#: Pinned rather than searched for: the unbiased grammar draws ``x*x`` against
#: a straddling float64 box only occasionally, and a control that depends on
#: that is a control that fires on some pushes and not others.
MUL_CORNER_PROBE = fo.Program(
    (_grammar.Decl("x0", (), "float64", -1.0, 1.0),),
    (_grammar.Stmt(
        "assert",
        ("cmp", ">=", ("bin", "mul", ("var", "x0"), ("var", "x0")),
         ("const", 0.0)),
    ),),
    "mul-corner-probe",
    "probe",
)


def _record(reading, census, program, distinct=None):
    """Census one reading. Shared by both legs so they count the same things.

    ``distinct`` is a ``key -> set of rendered programs``: 563 events standing
    on 75 distinct programs is a different fact from 563 events, and the
    ranking INVERTS between them — by events ``flush(175) > reassoc(27)``, by
    distinct programs ``reassoc(17) > flush(8)``, because 170 of the 175
    flushes are one pinned example re-drawn. A repair prioritised on the first
    would be prioritised on how often an ``@example`` was re-drawn.

    ────────────────────────────────────────────────────────────────────────
    AND ``distinct`` ALONE IS STILL THE WRONG DENOMINATOR, WHICH IS THE SAME
    MISTAKE ONE COLUMN OVER
    ────────────────────────────────────────────────────────────────────────

    The two columns above were crossed with :attr:`_float_oracle.Program.source`
    for the CAUSE partition and not for the two figures a repair is actually
    scoped on — the falsified-discharge count, and the distinct-program column
    of the cause table. Both are crossed here now, and the crossing is not a
    refinement: it changes what the headline says.

    ``falsified_discharges`` is the count every consumer of this module quotes
    as *"the denominator a repair is scoped on"*. Uncrossed it is one number
    over one set of programs. Crossed, it separates three generators with
    nothing in common but this census — an unbiased grammar, a strategy aimed
    at one known class whose own docstring says it *"finds that class and
    nothing else"*, and a fixed list of pinned ``@example``s that are re-drawn
    as often as Hypothesis feels like re-drawing them. A number that adds
    those together is a statement about draw densities wearing the clothes of
    a statement about the tool. The dated composition is in
    :data:`FLOAT_ORACLE_MEASURED`.

    ``distinct/cause/<cause>/<source>`` is the same crossing on the cause
    table, and it is here for one specific reading it makes impossible:
    ``reduction-reassociation`` is second by distinct programs, and every one
    of those programs comes from ``cancelling_sum_programs``, against a leg
    that ``_float_oracle.uniform_float_programs`` PROVES cannot reach the
    class at all. Ranking a class found only by the strategy built to find it
    against classes found by an unbiased search is not a ranking.
    """
    source = program.source or "unknown"
    census.draw()
    census.tag(f"strategy/{source}")
    if reading.status != "read":
        census.skip(reading.status)
        return
    census.tag("read")
    census.tag(f"read/{source}")
    if source == "uniform":
        census.tags["compared_boxes/uniform"] += reading.compared_boxes
    if not program.label:
        # THE UNBIASED HALF, COUNTED SEPARATELY, because every other floor in
        # this file is satisfiable by the pinned `@example`s alone. Driven with
        # both unbiased generators replaced by one unsampleable program, BOTH
        # LEGS STAYED GREEN on `read=42 finite_boxes=191 admitted_points=112
        # falsified_discharges=31` — every number of them from the members.
        # Nothing required the search to have looked at anything.
        census.tag("read/unlabelled")
    census.tags["boxes_read"] += reading.boxes_read
    census.tags["top_boxes"] += reading.top_boxes
    census.tags["empty_boxes"] += reading.empty_boxes
    census.tags["integer_boxes"] += reading.integer_boxes
    census.tags["compared_boxes"] += reading.compared_boxes
    census.tags["admitted_points"] += reading.admitted_points
    census.tags["falsified_discharges"] += reading.falsified_discharges
    census.tags[f"falsified_discharges/{source}"] += (
        reading.falsified_discharges
    )
    census.tags["contradicted_refutations"] += (
        reading.contradicted_refutations
    )
    census.tags["route_declined"] += reading.route_declined
    census.tags["route_unavailable"] += reading.route_unavailable
    if distinct is not None and reading.falsified_discharges:
        distinct.setdefault("falsified", set()).add(program.render())
        distinct.setdefault(f"falsified/{source}", set()).add(program.render())
    census.tags["route_obligations_compared"] += (
        reading.route_obligations_compared
    )
    census.tags["route_obligations_disagreed"] += (
        reading.route_obligations_disagreed
    )
    census.tags["sampler_artefacts"] += reading.sampler_artefacts
    for v in reading.violations:
        census.tag(f"cause/{v.cause}")
        census.tag(f"basis/{v.basis}")
        census.tag(f"violations/{source}")
        if distinct is not None:
            key = program.render()
            distinct.setdefault(f"cause/{v.cause}", set()).add(key)
            distinct.setdefault(f"cause/{v.cause}/{source}", set()).add(key)
            distinct.setdefault("any", set()).add(key)
            distinct.setdefault(f"any/{source}", set()).add(key)
            # A SITE is a program AND the equation in it, because one program
            # violating at three equations is three things to repair and one
            # program re-drawn thirty times is one.
            distinct.setdefault("site", set()).add(
                (key, v.eqn, v.primitive)
            )
        if program.label:
            census.tag(f"pinned/{program.label}")


def _report(program, reading, violation) -> str:
    return (
        "EXECUTED VALUE OUTSIDE THE COMPUTED BOX — equation %d (%s, %s) "
        "element %d executed %r, box [%r, %r]; cause %s\n%s\n"
        "# sampled point (first element of each declaration): %r\n"
        "# boxes read %d = %d compared + %d ⊤ + %d empty + %d integer; the "
        "last three cannot be violated by a finite value at all\n"
        "# cause reached by: %s\n"
        "# violations the SECOND (compiled) route declined: %d\n"
        "# obligations whose box says 'definitely true' and whose predicate "
        "executed FALSE, over every sampled point of this program: %d"
        % (
            violation.eqn, violation.primitive, violation.dtype,
            violation.element, violation.executed, violation.lo, violation.hi,
            violation.cause, program.render(), violation.point,
            reading.boxes_read, reading.compared_boxes, reading.top_boxes,
            reading.empty_boxes, reading.integer_boxes, violation.basis,
            reading.route_declined, reading.falsified_discharges,
        )
    )


#: ``(value, source dtype, target dtype)`` for the conversion row of the exact
#: reading. Every entry is a case the grammar can build: it draws ``cast``
#: nodes over every dtype in ``_grammar.ALL_DTYPES``, and 8 of 573 traced
#: uniform draws carried a float->int one.
CONVERSIONS = (
    (0.5, "float64", "int32"),      # truncates toward zero — 0, not 0.5
    (-1.7, "float64", "int32"),     # toward zero, so -1, not -2
    (2.0, "float64", "int32"),      # already integral: the one case the
                                    # identity happened to get right
    (7, "int64", "int32"),          # in range: the identity
    (2 ** 40, "int64", "int32"),    # out of range: WRAPS. refused
    (2 ** 62, "int64", "int8"),     # ditto, far out
    (0.5, "float64", "bool"),       # `a != 0`, so 1, not 0.5
    (0.0, "float64", "bool"),
    (1e30, "float64", "int32"),     # clamps or wraps; refused
    (3, "int32", "int64"),          # widening: the identity
)

#: The three cases above this reader refuses BY DESIGN — outside the target's
#: range, where jax clamps or wraps and ``_float_oracle._exact_convert`` will
#: not guess which. Named so the check below can tell a designed refusal from
#: a reader that has quietly stopped answering.
CONVERSIONS_REFUSED_BY_DESIGN = frozenset({
    (2 ** 40, "int64", "int32"),
    (2 ** 62, "int64", "int8"),
    (1e30, "float64", "int32"),
})


def _assert_the_exact_reading_of_a_conversion_is_what_the_cast_computes():
    """A PREMISE OF THE LEG BELOW, ASSERTED WHERE IT IS SPENT.

    ``_float_oracle._exact_apply``'s table is a set of claims that this module
    computes the same real number each primitive is specified to compute, and
    the conversion's entry was ``lambda a: a``. That is false for a float→int
    cast, which truncates toward zero, and for a narrowing int→int cast, which
    wraps. The consequence is not cosmetic and it lands in THIS property: the
    exact walk feeds :func:`_float_oracle.assumes_hold_over_reals`, which is
    :data:`_float_oracle.NARROWING`'s third fact, and a wrong answer there
    either mints ``(assume-narrowing, proved)`` or suppresses the fall-through
    that would prove ``UNEXPLAINED`` — the one answer this leg forbids.

    IT IS A FUNCTION AND NOT A ``test_``, FOR THE REASON THE SUITE ITSELF
    ENFORCES. ``test_suite_disclosure.py`` refuses any property that ships
    without a registered positive control, and ``tools/property_check.py``
    materialises only ``src/`` from a revision — the properties always come
    from the checkout — so a mutant control CANNOT reach a file under
    ``tests/property/``. A new ``test_`` here would therefore have to be
    exempted from that rule by name, which is a hole in the one mechanism that
    keeps a green property from being indistinguishable from a search that
    generates nothing. Asserted inside a property that HAS a control instead,
    beside the ``_grammar.SHAPES`` premise below, which is there for the same
    reason.

    IT IS DECIDED BY EXECUTION, NOT BY A SECOND MODEL OF THE CAST, because a
    check that models a behaviour is one indirection behind it and this file's
    whole subject is the gap between a model and an execution. Where the
    reader answers at all, jax's own conversion must produce that number.
    Driven both ways with the entry restored to the identity: ``0.5`` as
    ``float64 -> int32`` reads ``1/2`` where jax executes ``0``, and this
    raises; with the fix it passes.

    WHAT IT DOES NOT REACH, and it is the larger half. Only conversions into
    an INTEGER or ``bool`` dtype are checked, because those outputs are exact
    integers and the reader's answer must equal one. Into a FLOAT format the
    reader answers the real value and the format rounds it — that difference
    is exactly ``narrow-format-rounding`` and asserting equality there would
    forbid the class this module exists to find. A conversion the reader
    REFUSES is skipped rather than failed: a refusal is never a finding.
    """
    import jax.numpy as jnp
    import numpy as np

    checked = []
    for value, src, dst in CONVERSIONS:
        operand = np.asarray(value, dtype=getattr(np, src))
        exact = fo.exact_value("convert_element_type", (operand,), 0, 1, dst)
        if exact is None or dst in fo.FLOAT_TYPES:
            continue
        executed = int(
            np.asarray(jnp.asarray(operand).astype(dst)).reshape(-1)[0]
        )
        if exact.denominator != 1 or int(exact) != executed:
            raise AssertionError(
                "THE EXACT READING OF A CONVERSION IS NOT WHAT THE CONVERSION "
                "COMPUTES. %r as %s -> %s: this module reads %s, jax executes "
                "%d. `_float_oracle._exact_apply`'s entry for "
                "`convert_element_type` is a claim that the two are the same "
                "number, and `exact_walk` spends that claim on "
                "`assumes_hold_over_reals`, which decides whether an "
                "`assume-narrowing` is proved or an `UNEXPLAINED` is "
                "suppressed — and `UNEXPLAINED` is the answer this leg exists "
                "to forbid." % (value, src, dst, exact, executed)
            )
        checked.append((value, src, dst))
    # THE ABSENCE HALF. Every entry the reader refuses is skipped above, so a
    # reader that refused EVERYTHING would satisfy the loop vacuously — the
    # shape this suite exists to prevent. The list of cases it is SUPPOSED to
    # answer is derived from the table rather than typed.
    want = [c for c in CONVERSIONS
            if c[2] not in fo.FLOAT_TYPES
            and c not in CONVERSIONS_REFUSED_BY_DESIGN]
    assert checked == want, (
        "the conversion reader refused a case it is supposed to answer, so "
        "the check above was vacuous for it: wanted %r, checked %r"
        % (want, checked)
    )


@pytest.mark.xfail(strict=True, raises=ExecutedValueOutsideBox,
                   reason=FLOAT_BOX_DEFECT)
def test_the_executed_value_lies_inside_the_computed_box():
    """The invariant every VERIFIED rests on, asserted for the first time."""
    census = _runner.Census("float-oracle/containment")

    @_profiles.current().settings(80)
    @given(fo.float_oracle_inputs())
    def search(drawn):
        program, interior = drawn
        reading = fo.read(program, interior=interior)
        _record(reading, census, program)
        if reading.first is not None:
            raise ExecutedValueOutsideBox(
                _report(program, reading, reading.first)
            )

    search()
    # Only reached once the class is repaired, i.e. on the XPASS that turns
    # this test red. The floor is here so that the day somebody deletes the
    # marker they inherit a property that cannot pass vacuously — and the
    # floor is on NON-⊤ boxes, because a run that read nothing but ⊤ has
    # looked at nothing a finite value could ever fall outside of.
    census.require(**{"read": 20, "read/uniform": 10,
                      "compared_boxes/uniform": 50, "admitted_points": 20})


def test_every_violation_it_finds_has_a_known_ieee_cause():
    """The residual, and the thing that keeps the xfail above honest.

    Five causes are IEEE differences, each of which is a way execution differs
    from exact real arithmetic and none of which is a defect in the interval
    domain itself: a NaN (which no box contains, ⊤ included), an overflow to
    infinity where the box is finite, a reduction the compiler reassociated
    while the box is provably right about ℝ, a rounding into a format narrower
    than the binary64 the box was computed in, and a flush or gradual
    underflow in the program's own subnormal band. A violation with none of
    those explanations would mean the box is wrong about the reals, and that
    is what this leg forbids.

    **THIS PARAGRAPH SAID "FIVE" AND THE LEG PERMITS SEVEN**, which is a
    docstring naming a smaller check than the one that runs.
    ``_float_oracle.CAUSES`` has eight entries: the five above, plus
    ``assume-narrowing`` — a violation that exists only against the box an
    ``assume`` tightened, which is a finding about the PRECONDITION rather
    than about the arithmetic — plus ``unclassified``, which is a refusal, and
    ``UNEXPLAINED``, which is the one this leg forbids. Seven answers pass
    here, and the two that are not IEEE differences are the two that most need
    a reader to know they exist: ``unclassified`` is where the proof did not
    reach, and ``assume-narrowing`` is proved only where the sampled point is
    shown to be outside the assumed set over ℝ and is ``heuristic``
    otherwise.

    EVERY PINNED MEMBER IS AN ``@example`` HERE, and the census floor
    requires each of them to have produced a violation, so this leg is also
    the regression pin for the class: the day a member stops violating, the
    floor names it, and the day one violates for a different reason the drift
    check in the body names it. Two probes ride along that are NOT members —
    ``OVERFLOW_PROBE``, because the unbiased grammar reaches that cause only
    by luck, and ``MUL_CORNER_PROBE``, which produces no violation at all on a
    clean tree and is here to be the mutant control's harness.
    """
    census = _runner.Census("float-oracle/residual")
    distinct: dict = {}
    # A PREMISE, BEFORE THE SEARCH SPENDS 120 EXAMPLES ON IT. The exact
    # reading of a conversion is what `assumes_hold_over_reals` — and so every
    # `assume-narrowing` answer below — rests on. See the function's own
    # docstring for why it is a function and not a `test_`.
    _assert_the_exact_reading_of_a_conversion_is_what_the_cast_computes()

    @_profiles.current().settings(120)
    @given(fo.float_oracle_inputs())
    @example((MUL_CORNER_PROBE, 0.5))
    @example((OVERFLOW_PROBE, 0.5))
    @example((fo.MEMBERS["underflow-reciprocal"][0], 0.5))
    @example((fo.MEMBERS["ftz-subnormal-sum"][0], 0.5))
    @example((fo.MEMBERS["f32-underflow"][0], 0.5))
    @example((fo.MEMBERS["nan-from-y-over-y"][0], 0.5))
    @example((fo.MEMBERS["reassociation-n33"][0], 0.5))
    @example((fo.MEMBERS["f32-single-multiply"][0], 0.5))
    @example((fo.MEMBERS["f32-exp"][0], 0.5))
    @example((fo.MEMBERS["subnormal-comparison"][0], 0.5))
    @example((fo.MEMBERS["assume-narrows-past-the-program"][0], 0.5))
    def search(drawn):
        program, interior = drawn
        reading = fo.read(program, interior=interior)
        _record(reading, census, program, distinct)
        registered = fo.MEMBERS.get(program.label)
        if registered is not None and reading.violations:
            # A PINNED MEMBER THAT STILL VIOLATES FOR A DIFFERENT REASON is
            # not the same evidence, and the per-member census floor cannot
            # see the difference: it counts violations, not causes. Asked of
            # the whole reading and not of each violation, because a member
            # may violate at several sampled points for several reasons —
            # `f32-exp` does, flushing to zero at x = -100 and landing one
            # float32 ulp above the box at x = -50 — and what is pinned is
            # that the REGISTERED cause is still among them.
            causes = {v.cause for v in reading.violations}
            if registered[1] not in causes:
                raise AssertionError(
                    "A PINNED MEMBER HAS CHANGED CAUSE — %r was registered as "
                    "%s (%s) and now reports only %s. Either the backend "
                    "changed under it or the classifier did; re-measure it "
                    "and re-register it, do not widen the classifier to make "
                    "this go away:\n%s"
                    % (program.label, registered[1], registered[2],
                       sorted(causes),
                       _report(program, reading, reading.violations[0]))
                )
        if reading.sampler_artefacts:
            # THE ELEVENTH CHECK, AND NOTHING ASSERTED IT. A box violation on
            # `stelling_any` in a harness with NO assume cannot be the
            # program's arithmetic — there is none — so it can only be this
            # module's own sampler landing outside the declared box, which is
            # what `snap_inward` exists to prevent. Driven with its two
            # `nextafter` steps deleted, 600 draws: 97 violations of which
            # **52 are on `stelling_any`**, offered as
            # `narrow-format-rounding` and `overflow-to-inf`. The containment
            # leg still XFAILed and this leg failed only incidentally, through
            # one member's cause-drift check.
            raise AssertionError(
                "SAMPLER ARTEFACT — %d violation(s) on `stelling_any` in a "
                "harness that states no assume. The declaration primitive "
                "computes nothing, so a value outside its own declared box is "
                "this instrument's sampler, not the program:\n%s"
                % (reading.sampler_artefacts,
                   _report(program, reading, reading.violations[0]))
            )
        for v in reading.violations:
            if v.cause == fo.UNEXPLAINED:
                raise AssertionError(
                    "UNEXPLAINED BOX VIOLATION — the executed value is "
                    "outside the computed box and none of the five IEEE "
                    "explanations applies, so the box is wrong about the "
                    "REALS rather than about the floats:\n"
                    + _report(program, reading, v)
                )

    search()
    # DISTINCT PROGRAMS ALONGSIDE EVENTS, because the ranking inverts between
    # them and a repair scoped on the event column would be scoped on how
    # often an `@example` was re-drawn.
    #
    # EVERY CELL OF THE STRATEGY CROSSING IS WRITTEN OUT, ZEROS INCLUDED, and
    # the zeros are the finding. A key that is simply absent reports absence of
    # evidence, which is read as evidence of absence — and the two cells that
    # matter most here are empty: the unbiased leg contributes no falsified
    # discharge and no reassociation at all. The source and cause lists are
    # read off the census rather than typed, so a fourth strategy or an eighth
    # cause appears in the table without anybody remembering to add it.
    sources = sorted(k.split("/", 1)[1] for k in census.tags
                     if k.startswith("strategy/"))
    causes = sorted(k.split("/", 1)[1] for k in census.tags
                    if k.startswith("cause/"))
    for s in sources:
        census.tags[f"falsified_discharges/{s}"] += 0
        distinct.setdefault(f"falsified/{s}", set())
        distinct.setdefault(f"any/{s}", set())
        for c in causes:
            distinct.setdefault(f"cause/{c}/{s}", set())
    for key, seen in distinct.items():
        census.tags[f"distinct/{key}"] = len(seen)
    # THE PREMISE OF THE ZERO IN THAT TABLE, ASSERTED WHERE IT IS STATED.
    # `distinct/cause/reduction-reassociation/uniform` is 0, and the crossing
    # is worth having only because that zero is a PROOF about the generator
    # rather than an under-sampled cell: `_grammar.SHAPES` tops out at four
    # elements and `jnp.sum` splits into two windows only at n >= 33, so no
    # draw of `uniform_float_programs` can cross the boundary. That is the
    # sentence `_float_oracle.uniform_float_programs` makes, and the whole
    # weight of "reassociation is second by distinct programs" rests on it —
    # the class is second on a strategy built to find it and nothing else,
    # against a leg that cannot reach it AT ALL rather than one that merely
    # did not. Nothing checked the premise, so it is checked here, at the
    # source rather than in the table: a shape big enough to reach the split
    # turns the zero from a proof into a null result, and a null result may
    # not be quoted the way this one is.
    biggest = max((math.prod(s) for s in _grammar.SHAPES), default=0)
    assert biggest < fo.REDUCTION_SPLIT_N, (
        "`_grammar.SHAPES` now admits an array of %d elements, which reaches "
        "`jnp.sum`'s n >= %d two-window split. The unbiased leg's ZERO in the "
        "reduction-reassociation row is a proof about the generator only "
        "while that is false; with this shape it becomes a null result, and "
        "every consumer that ranks the causes by distinct programs is quoting "
        "it as the former. Re-derive the ranking, or re-state the proof."
        % (biggest, fo.REDUCTION_SPLIT_N)
    )
    floors = {
        "read": 30,
        "read/uniform": 15,
        "read/cancelling": 10,
        "compared_boxes/uniform": 50,
        "admitted_points": 30,
        "falsified_discharges": 1,
        # THE FALSIFIED-DISCHARGE FIGURE, CROSSED, AND THE CROSSED ONE IS THE
        # ONE THAT MEANS ANYTHING. `falsified_discharges` and
        # `distinct/falsified` are both functions of the BUDGET — measured on
        # this tree, `distinct/falsified` is 5 at the `ci` profile and 21 at
        # `STELLING_PROPERTY_SCALE=12.5`, because the extra 16 are one aimed
        # strategy's shrink neighbours. `distinct/falsified/member` is 5 in
        # both: it is a fact about the tree, and it is the pin that says all
        # five discharge-falsifying members are still falsifying one.
        "distinct/falsified/member": len(fo.DISCHARGE_FALSIFYING),
        "distinct/any": 10,
    }
    floors.update({f"pinned/{name}": 1 for name in fo.MEMBER_NAMES})
    floors.update({
        f"cause/{c}": 1
        for c in (fo.NAN, fo.OVERFLOW, fo.NARROW, fo.FLUSH,
                  fo.REASSOCIATION)
    })
    census.require(**floors)
