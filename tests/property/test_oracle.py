# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE ORACLE PROPERTY: no VERIFIED has a violating admitted point.

**One-sided, and the asymmetry is the whole design.** Enumerating the declared
box can *refute* a VERIFIED — one admitted point at which the obligation, as
the user wrote it, is false, and the verdict is wrong. It can never *confirm*
one: a VERIFIED is a claim about every point, and this only ever looks at the
points it enumerated. So a green run here means "no counterexample was found in
what was searched", never "the tool is right". Everything this file asserts is
of the first kind.

**Why the ground truth is the source text and not a jax execution.** The first
formulation of this property in this project evaluated the predicate by running
it through jax, and it *could not have found* the defect it was aimed at — the
defect is that an out-of-dtype-range integer literal wraps mod ``2**bits``
before tracing, so an execution oracle wraps it too and cheerfully agrees with
the wrong answer. ``_grammar.eval_pred_exact`` evaluates the predicate in
unbounded Python integers and never wraps a constant. **For THAT defect** — one
that destroys the program before any stelling primitive binds it — a
source-text oracle is strictly stronger than an execution oracle.

NARROWED TO ITS SUBJECT. That sentence used to end *"Where a defect is in the
TRANSLATION to the jaxpr, a metamorphic or source-text oracle is strictly
stronger than an execution oracle"*, and it was read as an argument about
execution oracles in general. It is true of the wrap and of anything else that
rewrites the program before tracing. It says nothing about a defect in the
ANALYSIS of a faithfully traced program, where there is no second reading to be
stronger than — see ``test_float_oracle.py``, which executes the program and
pins eight harnesses, in five distinct IEEE causes, that this file cannot
express at all.

**Why integers only — AND WHAT THAT DOES NOT MEAN ANY MORE.**
``SOUNDNESS.md`` records that real mode judges floats in exact real arithmetic
while integers are judged execution-faithfully, so a float harness can be
correctly VERIFIED in ℝ and violated by IEEE execution. This paragraph used to
conclude from that *"an oracle pointed at floats would be reporting the
documentation rather than a defect"*, and **that conclusion is withdrawn**: a
VERIFIED the compiled program contradicts at a dtype-representable point of its
own declared box is unacceptable whatever ℝ says, and ``test_float_oracle.py``
pins eight of them. What survives is the narrow, mechanical reason THIS file is
integer-only: its ground truth is ``_grammar.eval_pred_exact``, which evaluates
in unbounded Python integers, and there is no float it could evaluate in. The
float question is asked next door, by executing the program.

────────────────────────────────────────────────────────────────────────────
WHAT IS AND IS NOT COVERED — read this before trusting a green run
────────────────────────────────────────────────────────────────────────────

Covered: one integer declaration, scalar or ``(2,)``, a box inside the dtype's
own range and small enough to enumerate exactly, an optional ``assume`` and one
``assert_``, over ``+ - *``, ``jnp.maximum``/``minimum``, unary ``-``,
``abs``, ``square``, ``** k`` and the six comparisons, with literals drawn from
in-range, just-out-of-range and far-out-of-range pools.

NOT covered: floats of any width (``test_float_oracle.py`` covers those, by a
different mechanism — executing the program and comparing against the
propagated box); ``bool``; more than one declaration; boxes too large to
enumerate; ``scan``/``while``/``fori_loop``; casts; reductions; ``where``;
scatter; the solver legs; ``nonvacuity``; anything reached only through the
public template helpers rather than ``any_array``. A green run says nothing
about any of those.

────────────────────────────────────────────────────────────────────────────
THE TWO LEGS, AND WHY THERE ARE TWO
────────────────────────────────────────────────────────────────────────────

``test_a_verified_is_true_at_every_admitted_point`` is the property in full.
**It FAILS on ``main`` today**, because the open integer-literal wrap defect is
exactly a family of wrong VERIFIEDs and this grammar produces them densely
(measured: 13 distinct constructions, each with an enumerated exact
counterexample, byte-identical on both jax series). It is marked
``xfail(strict=True, raises=WrongVerifiedFromWrap)``. Three things about that
marker, each deliberate:

* **strict** — a non-strict xfail passes silently the day the defect is fixed,
  which is the same silent-success shape this whole suite exists to prevent.
  Strict means the suite goes RED when the remedy lands, and somebody has to
  come here and delete the marker.
* **raises=** — the marker excuses *this* defect class and no other. **What it
  is load-bearing for is not what this docstring used to claim.** The old
  wording said a wrong VERIFIED carrying no out-of-dtype-range constant raises
  a plain ``AssertionError``, does not match, and is reported as a real
  failure. That is true of the code path below and **vacuous for this test**:
  the leg draws from ``wrap_biased_integer_specs``, which is
  ``integer_specs().filter(wrappable_constants)``, so every spec it examines
  carries a wrappable constant, ``WrongVerifiedFromWrap`` is always the branch
  taken, and ``raise AssertionError(msg)`` on the line below is unreachable
  from this test. The residual leg is what confines the class, not ``raises=``.

  ``raises=`` stays, because it is load-bearing for something that WAS
  measured: the **census floor**. With ``raises=`` deleted and
  ``propagate._bool_status`` mutated to violate everything, this leg reports
  **XFAIL — green — on ``verified_oracle_checked=0 < 5`` after drawing 150
  specs and examining 90**, because ``census.require``'s ``AssertionError`` is
  swallowed by the blanket amnesty. With ``raises=`` in place the same run is
  reported ``FAILED`` on ``GENERATOR FLOOR NOT MET``. One deleted keyword
  between a floor that holds and a floor that cannot fail.
  ``test_suite_disclosure.py`` now asserts the keyword is there, statically,
  because no log-reader can see it go.
* the reason names the open defect, so CI's own output says what is not being
  checked rather than leaving a reader to infer it from a green tick.

``test_a_verified_is_true_at_every_admitted_point_outside_the_wrap_class`` is
the same property with that one class masked out, and it **PASSES**. That is
what keeps the xfail honest: the residual is checked green, so the marker
covers the known defect rather than the whole property.

The mask folds maximal constant subexpressions before testing them against the
dtype range, because Python folds ``-abs(1)`` and ``-129`` before jax sees
anything. A mask written on literal tokens alone was measured to leave the
defect in, with the "masked" property still failing at every budget.

────────────────────────────────────────────────────────────────────────────
POSITIVE CONTROLS
────────────────────────────────────────────────────────────────────────────

* wrap leg — fails at ``main`` (``bf905b9``). Registered as ``oracle-wrap``.
* masked leg — fails at ``main`` with the mask disabled, and at the
  ``interval.mul`` cross-term mutant. Registered as ``oracle-masked``.

Run them: ``python tools/property_check.py --controls``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis", reason="needs hypothesis")
jax = pytest.importorskip("jax")

from hypothesis import example, given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import _grammar  # noqa: E402
import _profiles  # noqa: E402
import _runner  # noqa: E402

WRAP_DEFECT = (
    "open defect: an out-of-dtype-range integer literal wraps mod 2**bits "
    "before tracing, so stelling verifies a predicate that is false at every "
    "declared point (HANDOFF6 §71, §74 — release-blocking, remedy reserved to "
    "the principal). This marker is strict and narrowed by raises=: the day "
    "the remedy lands this test XPASSes and the suite goes red."
)


class WrongVerifiedFromWrap(AssertionError):
    """A wrong VERIFIED whose harness carries an out-of-dtype-range constant.

    A distinct type so that ``xfail(raises=...)`` can excuse this defect class
    and nothing else. **Inside the leg that raises it that narrowing is
    vacuous** — the leg's strategy is filtered to wrappable harnesses, so no
    other wrong VERIFIED can arrive there; see the module docstring. What the
    narrowing does confine is everything else the marked test can raise, and
    the one that matters is its own census floor.
    """


@pytest.fixture(autouse=True)
def _x64():
    """Judge at x64-on, which is the posture the project's own docs use.

    README's Quickstart line 2 enables it, 13 of 20 ``corpus/supply`` harnesses
    set it, and parts of the wrap class vanish without it (``int32`` genuinely
    wraps only at x64-on). Saved and restored so this module cannot change what
    the rest of the session measures.
    """
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _oracle_body(spec, census, *, mask_wrap):
    """Return a failure message, or ``None``. Shared by both legs."""
    census.draw()
    if mask_wrap and _grammar.wrappable_constants(spec):
        census.skip("masked: carries a wrappable constant")
        return None
    if not _grammar.exact_supported(spec):
        census.skip("no exact oracle for this shape")
        return None
    if not _grammar.any_obligation_is_admitted(spec):
        # `∀ x ∈ ∅` is vacuously true and the tool is entitled to VERIFIED.
        census.skip("preconditions admit nothing")
        return None
    outcome = _runner.run(spec, census=census)
    if outcome is None:
        return None
    if outcome.status != "VERIFIED":
        return None
    census.tag("verified_oracle_checked")
    ces = _grammar.counterexamples(spec)
    if not ces:
        return None
    index, point = ces[0]
    return (
        "WRONG VERIFIED — obligation %d is false at the admitted point %r\n%s\n"
        "# stelling: VERIFIED %r\n"
        "# constants outside the declared dtype's range: %r\n"
        "# further violating points: %r"
        % (
            index,
            point,
            spec.render(),
            outcome.obligations,
            _grammar.wrappable_constants(spec),
            [p for _, p in ces[1:4]],
        )
    )


@pytest.mark.xfail(strict=True, raises=WrongVerifiedFromWrap, reason=WRAP_DEFECT)
def test_a_verified_is_true_at_every_admitted_point():
    census = _runner.Census("oracle/wrap")

    @_profiles.current().settings(150)
    @given(_grammar.wrap_biased_integer_specs(shapes=((),), max_leaves=4))
    def search(spec):
        msg = _oracle_body(spec, census, mask_wrap=False)
        if msg is None:
            return
        if _grammar.wrappable_constants(spec):
            raise WrongVerifiedFromWrap(msg)
        # UNREACHABLE FROM THIS TEST, and left in deliberately. The strategy
        # above is `integer_specs().filter(wrappable_constants)`, so the branch
        # cannot be taken here — the narrowing this line represents is done by
        # the residual leg below, not by `raises=`. It stays because the day
        # somebody widens this leg's strategy it is the difference between a
        # non-wrap wrong VERIFIED being reported and being excused, and because
        # deleting it would leave a reader thinking `raises=` narrows something
        # inside this test. It does not; see the module docstring for what it
        # DOES do, which is hold the census floor.
        raise AssertionError(msg)

    search()
    # Only reached once the wrap defect is fixed, i.e. on the XPASS that turns
    # this test red. The floor is here so that the day someone deletes the
    # marker they inherit a property that cannot pass vacuously.
    census.require(examined=40, verified_oracle_checked=5)


MONOTONICITY_PROBE = _grammar.Spec(
    (_grammar.Decl("x0", (), "int8", -1, 1),),
    (
        _grammar.Stmt(
            "assert",
            ("cmp", ">=", ("bin", "mul", ("var", "x0"), ("var", "x0")),
             ("const", 1)),
        ),
    ),
)
"""``x*x >= 1`` over ``int8 [-1, 1]``, pinned, carrying no wrappable constant.

This is the residual leg's positive control in one harness: with interval
multiplication taking only the two same-corner products — the registered
``oracle-masked`` mutant — the image of ``x*x`` is ``[1, 1]`` and the
obligation is DISCHARGED, while ``x = 0`` is a declared point at which it is
false. Pinned so that the control does not depend on the search happening to
build ``x*x`` against a constant inside 500 examples.
"""


def test_a_verified_is_true_at_every_admitted_point_outside_the_wrap_class():
    """The residual, and the thing that keeps the xfail above honest."""
    census = _runner.Census("oracle/masked")

    @_profiles.current().settings(500)
    @given(_grammar.integer_specs())
    @example(MONOTONICITY_PROBE)
    def search(spec):
        msg = _oracle_body(spec, census, mask_wrap=True)
        if msg is not None:
            raise AssertionError(msg)

    search()
    census.require(examined=60, verified_oracle_checked=10)


# ── the other side of the same coin ──────────────────────────────────────────


VACUOUS_REFUTATION_EXAMPLE = _grammar.Spec(
    (_grammar.Decl("x0", (3,), "float64", -1.0, 1.0),),
    (
        _grammar.Stmt("assume", ("all", ("cmp", ">=", ("var", "x0"),
                                         ("const", 2.0)))),
        _grammar.Stmt("assert", ("cmp", ">=", ("var", "x0"), ("const", 5.0))),
    ),
)
"""The shape the affine refinement got wrong, pinned.

`jnp.all(...)` around the comparison is load-bearing and not decoration: the
plain form `assume(x0 >= 2.0)` is refused outright with
``UnsatisfiableAssumptionError``, so the defect is only reachable through the
form the assume is DROPPED on. Measured at 8106a55: ``refine=None`` UNKNOWN,
``refine="affine"`` REFUTED, over a region the harness's own precondition
leaves empty. Pinned as an ``@example`` because the search over this grammar
does not construct it — that is recorded rather than hidden, and it is the
honest reading of what a generated corpus reaches.
"""


def test_a_refuted_is_false_at_some_admitted_point():
    """A ``violated-over-set`` must not be minted over a region it is not about.

    ``violated-over-set`` says the obligation is definitely false **for every
    element of the declared set**. Two ways that is wrong:

    * the admitted set is EMPTY. ``∀ x ∈ ∅`` is vacuously true, so nothing is
      violated. This is a *ruling* rather than a theorem — over an empty set
      both a discharge and a violation are vacuously supportable — and it is
      this project's own: 463ee81, "the affine refinement may not restore what
      the interval leg withheld", fixed exactly this shape;
    * there is an admitted point at which the predicate, as written, is TRUE.
      One-sided: finding such a point refutes a REFUTED; finding none confirms
      nothing.

    Emptiness is decided EXACTLY, two ways, and never guessed: by enumerating
    the declared box in Python integers where the harness is integral, and by
    intersecting half-spaces with the declared box where it is float and the
    assumes are simple comparisons. Where neither applies the example is
    discarded — a refusal to answer, not a silent "no".

    **Both ``refine`` legs are checked**, and that is where the defect lived:
    at ``refine=None`` this harness is UNKNOWN, and only the refinement mints
    the violation.

    Harnesses carrying an out-of-dtype-range constant are masked, because under
    the open wrap defect the exact reading and the traced program are different
    programs, and a report here would be a second sighting of that defect
    rather than a finding about refutation.
    """
    census = _runner.Census("oracle/refuted")

    @_profiles.current().settings(250)
    @given(st.one_of(_grammar.integer_specs(), _grammar.general_specs()))
    @example(VACUOUS_REFUTATION_EXAMPLE)
    def search(spec):
        census.draw()
        if _grammar.wrappable_constants(spec):
            census.skip("masked: carries a wrappable constant")
            return
        exact = _grammar.exact_supported(spec)
        points = _grammar.declared_points(spec) if exact else None
        for leg in (None, "affine"):
            outcome = _runner.run(spec, census=census, refine=leg)
            if outcome is None:
                continue
            asserts = spec.asserts
            if len(outcome.obligations) != len(asserts):
                census.skip("obligation count does not match the assert count")
                continue
            for i, status in enumerate(outcome.obligations):
                if status != "violated-over-set":
                    continue
                admitted = None
                if points is not None:
                    admitted = _grammar.admitted_for_obligation(spec, i, points)
                    empty = not admitted
                else:
                    empty = _grammar.simple_admitted_region_is_empty(spec, i)
                    if empty is None:
                        census.skip("emptiness not exactly decidable here")
                        continue
                census.tag("refuted_oracle_checked")
                assert not empty, (
                    f"REFUTED OVER AN EMPTY ADMITTED REGION (refine={leg!r}): "
                    f"obligation {i} is 'violated-over-set', but the harness's "
                    f"own preconditions admit no point of the declared box, so "
                    f"the obligation is vacuously TRUE and nothing is violated"
                    f"\n{spec.render()}"
                )
                if admitted is None:
                    continue
                true_at = [
                    p
                    for p in admitted
                    if _grammar.eval_pred_exact(asserts[i].pred, p)
                ]
                assert not true_at, (
                    f"REFUTED BUT TRUE SOMEWHERE (refine={leg!r}): obligation "
                    f"{i} is 'violated-over-set', and the predicate as written "
                    f"is TRUE at admitted point(s) {true_at[:3]}\n"
                    f"{spec.render()}"
                )

    search()
    census.require(examined=60, refuted_oracle_checked=1)
