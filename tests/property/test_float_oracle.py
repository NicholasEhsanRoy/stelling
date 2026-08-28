# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE DIFFERENTIAL FLOAT ORACLE: is the executed value inside the box?

``stelling`` judges obligations by interval arithmetic and stamps VERIFIED.
Every one of those stamps rests on one sentence:

    **the value the compiled program computes lies inside the box the
    propagator computed for it.**

Nothing in ``src/`` asserts that sentence and nothing in ``tests/`` asserted it
before this module. It is FALSE on this tree, in eight measured places — the
seven this module was commissioned to pin (``_float_oracle.SEVEN``) plus one
it found itself. **Seven of the eight are false against ``v0.1.0``'s own
``src/`` too, and four of those falsify a discharge there**, which is
re-derived at :data:`_float_oracle.MEMBERS` rather than quoted. This module
runs the propagator and the compiled program over the same query and compares
them value by value.

**THE NEAREST INSTRUMENT THAT DOES EXIST, AND WHAT IT DOES WITH THESE EIGHT.**
``stelling.falsify.probe`` — ``check(..., falsify="sample")``, default-off and
``experimental`` — also executes the program at concrete points. It asks a
different question: *is the OBLIGATION false here*, not *is each value inside
its box*. Driven over all eight members plus ``OVERFLOW_PROBE``, 2026-08-28 at
``a90862b``: **it reports none of the nine.** Two are UNKNOWN so it never runs
(it fires only after a VERIFIED); one is REFUTED; six are VERIFIED and stay
VERIFIED. For four of those six it SAW the executed violation and declined it
— 28 points on ``ftz-subnormal-sum``, 12 on ``f32-underflow``, 2 on
``f32-single-multiply`` as ``float-rounding-artefact``
(``exact-replay-holds-over-the-rationals``), and 58 on ``f32-exp`` as
``no-exact-reading-of-this-program`` — and its note says so in its own words:
*"28 EXECUTED VIOLATION(S) WERE DECLINED, NOT ABSENT"*. That decline rule is
the ℝ defence in instrument form, disclosed rather than hidden. The remaining
two produce no obligation violation at all, because the box violation is one
equation UPSTREAM of the obligation: ``1/Σ(…)`` executes ``inf > 0`` and the
float32 square executes ``inf > 0``, both TRUE. A box-containment oracle sees
a defect one equation before an obligation oracle can.

**THE ARGUMENT THAT KEPT THIS OUT IS RETRACTED, AND IT IS RETRACTED IN
``tests/property/README.md`` WITH WHAT IT SAID.** That file scoped the
execution oracle to integers on the ground that *"an oracle pointed at floats
measures the documentation"* — a float harness may be correctly VERIFIED in ℝ
and violated by IEEE, which is the declared posture. The premise is true and
the conclusion does not follow: a VERIFIED that the compiled program
contradicts at a point of its own declared box is unacceptable whatever ℝ
says, and the eight programs pinned below are exactly that. What the ℝ posture
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

1. **⊤ is unfalsifiable.** A declined equation's box is ``[-inf, inf]`` and
   contains every finite float, so this instrument is blind exactly where the
   propagator already declined. The pass rate is **not** a safety signal. The
   census counts boxes in three buckets — ⊤, EMPTY (a size-0 declaration, a
   value with no elements to violate anything) and the rest — and both
   properties floor on the third alone.
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
    "computed, in eight pinned programs. Seven of the eight do it against "
    "v0.1.0's own src as well, and four of those falsify a discharge there. "
    "This marker is strict and narrowed by raises=: the day a remedy lands "
    "this test XPASSes and the suite goes red."
)

#: THE PARTITION, as a dated record rather than a present-tense claim.
#:
#: Measured 2026-08-28 at ``a90862b`` — jax 0.11.0, jaxlib 0.11.0, numpy
#: 2.5.2, CPython 3.12.3, hypothesis 6.165.10, ``JAX_PLATFORMS=cpu``, x64
#: forced on by this module's fixture — by running THIS FILE's residual leg at
#: ``STELLING_PROPERTY_SCALE=12.5``, i.e. 1500 examples, and reading its own
#: census. Re-derivable by anyone with this tree and that command: the census
#: line is what ``Census.require`` prints when a floor is not met, and the
#: same object is what the floors below are asserted against.
#:
#: **THE SPEC THIS WAS BUILT FROM QUOTED A DIFFERENT PARTITION** — 120
#: violations in 1500 examples, as 61 float32 precision, 37 NaN, 19
#: overflow-to-inf, 3 box-touches-flush-region and 0 f64 in-band — and the
#: totals agree far better than the shape does. Re-measured over 1500 draws of
#: the UNBIASED leg alone, this instrument found violations in **146** of
#: them, which is the same order as 120. The partition differs for three
#: reasons, all of them properties of this instrument rather than
#: disagreements about the tree: it compares EVERY equation rather than the
#: obligation's own, so a NaN deep inside an expression counts; it executes up
#: to six points per program rather than one, so one program contributes
#: several violations; and it classifies a float32 underflow as
#: ``flush-or-subnormal`` rather than as "float32 precision", which moves the
#: largest bucket. The row that matters most is unchanged, and here it is a
#: PROOF rather than a count: **0 f64 in-band violations from the uniform
#: grammar**, because ``_grammar.SHAPES`` tops out at four elements and the
#: reassociation split needs 33.
FLOAT_ORACLE_MEASURED = """\
1510 programs drawn, 1171 read; 33.5 s and 40.5 s on two runs at load
     average 6 and 12 on a 24-core box, i.e. 22-27 ms/example, of which
     the great majority is op-by-op jax dispatch and not propagation.
     Both runs gave the census below byte for byte.
 339 not read: 125 ValueError, 52 OverflowError and 44 TypeError at the
     declaration door; 74 whose assumes admit none of the sampled points;
     22 UnsatisfiableAssumptionError; 22 unsampleable (no value of the
     declaration's own dtype lies inside its declared box)
11488 boxes read: 1603 ⊤ (14 %), 1177 EMPTY (10 %, a size-0 declaration),
     8708 (76 %) that a finite value could be caught outside of. The first two
     cannot be violated by a finite value at all, for different reasons

384 VIOLATIONS, by cause:
    178  flush-or-subnormal
    109  nan
     51  narrow-format-rounding
     40  reduction-reassociation
      6  overflow-to-inf
      0  UNEXPLAINED

110 of them are a DISCHARGE FALSIFIED: an obligation whose box says
    "definitely true for all elements", whose predicate executed FALSE at an
    admitted, dtype-representable point of its own declared box.
 18 are the other direction: a "violated-over-set" that executed TRUE.
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
)


def _record(reading, census, program):
    """Census one reading. Shared by both legs so they count the same things."""
    census.draw()
    if reading.status != "read":
        census.skip(reading.status)
        return
    census.tag("read")
    census.tags["boxes_read"] += reading.boxes_read
    census.tags["top_boxes"] += reading.top_boxes
    census.tags["empty_boxes"] += reading.empty_boxes
    census.tags["finite_boxes"] += reading.finite_boxes
    census.tags["admitted_points"] += reading.admitted_points
    census.tags["falsified_discharges"] += reading.falsified_discharges
    census.tags["contradicted_refutations"] += (
        reading.contradicted_refutations
    )
    for v in reading.violations:
        census.tag(f"cause/{v.cause}")
        if program.label:
            census.tag(f"pinned/{program.label}")


def _report(program, reading, violation) -> str:
    return (
        "EXECUTED VALUE OUTSIDE THE COMPUTED BOX — equation %d (%s, %s) "
        "element %d executed %r, box [%r, %r]; cause %s\n%s\n"
        "# sampled point (first element of each declaration): %r\n"
        "# boxes read %d, of which ⊤ (unfalsifiable for a finite value): %d\n"
        "# obligations whose box says 'definitely true' and whose predicate "
        "executed FALSE, over every sampled point of this program: %d"
        % (
            violation.eqn, violation.primitive, violation.dtype,
            violation.element, violation.executed, violation.lo, violation.hi,
            violation.cause, program.render(), violation.point,
            reading.boxes_read, reading.top_boxes,
            reading.falsified_discharges,
        )
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
    census.require(read=20, finite_boxes=100, admitted_points=20)


def test_every_violation_it_finds_has_a_known_ieee_cause():
    """The residual, and the thing that keeps the xfail above honest.

    Five causes are known, each of which is a way IEEE execution differs from
    exact real arithmetic and none of which is a defect in the interval domain
    itself: a NaN (which no box contains, ⊤ included), an overflow to infinity
    where the box is finite, a reduction the compiler reassociated while the
    box is provably right about ℝ, a rounding into a format narrower than the
    binary64 the box was computed in, and a flush or gradual underflow in the
    program's own subnormal band. A SIXTH kind — a violation with none of
    those explanations — would mean the box is wrong about the reals, and that
    is what this leg forbids.

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
    def search(drawn):
        program, interior = drawn
        reading = fo.read(program, interior=interior)
        _record(reading, census, program)
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
    floors = {
        "read": 30,
        "finite_boxes": 100,
        "admitted_points": 30,
        "falsified_discharges": 1,
    }
    floors.update({f"pinned/{name}": 1 for name in fo.MEMBER_NAMES})
    floors.update({
        f"cause/{c}": 1
        for c in (fo.NAN, fo.OVERFLOW, fo.NARROW, fo.FLUSH,
                  fo.REASSOCIATION)
    })
    census.require(**floors)
