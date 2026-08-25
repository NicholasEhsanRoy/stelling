# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The probe's exact reading, against an oracle that shares no code with it.

**WHY THIS FILE EXISTS.** ``stelling/falsify.py`` and
``stelling/preconditions.py`` retired the word ``UNAUDITED`` from the
falsification probe on the strength of a sentence: *"the fire condition
was audited blind against an independent ``Fraction`` oracle that shares
no code with this package: 363 gate readings, 363 agreements, over 266
driven base-versus-fix comparisons."* Those figures appeared in exactly
one place in the repository — the sentence that cited them. No test, no
CHANGELOG entry, no doc.

This project's own rule is that **prose asserting a check that does not
exist is the same defect as the missing check**, and a number nobody can
re-derive is that shape, sitting in the two files whose subject is what a
reader may believe. So the audit's oracle is committed here. It makes the
claim verifiable AND standing: the property is guarded on every run
rather than asserted once by someone who ran it.

**WHAT "SHARES NO CODE" MEANS HERE, AND IT IS ENFORCED BY A TEST BELOW.**
The oracle is ordinary Python over :class:`fractions.Fraction`. It reads
the same numpy point the probe is handed, converts each element with
``Fraction(float)`` — exact for every float64, by construction — and
evaluates the SAME predicate the harness states, written a second time by
hand. It touches no function in ``stelling.falsify``: not ``_exact``, not
``_ew``, not ``_apply``, not the name tables. The only thing under test it
calls is ``_replay`` itself, which is unavoidable, and ``_read``, which
builds the census ``_replay`` takes.

A second independent reading is the point of the exercise. The probe's
own independence argument (``falsify.py``, THE INDEPENDENCE ARGUMENT) is
that a check reaching its answer through the machinery it is checking is a
second face asking the same wrong question — and this file applies that
argument one level further out, to the probe itself.

**WHAT IS COMPARED.** For every fixture and every point:

* the ASSUME reading, ``_replay(..., assumes_only=True)`` — the gate that
  decides whether a point is inside the region the analysis claimed
  anything about, and the reading the whole admissibility count rests on;
* the ASSERT reading, ``_replay(...)`` — the exact rational adjudication
  that is the only thing allowed to admit a firing.

The fixtures are chosen so that the float answer and the ℚ answer DISAGREE
on part of the grid. An oracle that agreed with the probe only where every
route agrees would be measuring nothing: ``y*0.1*10.0 <= y`` and
``(1e16 + y) - 1e16 <= y`` are both true for every real ``y >= 0`` and
both false in float64 at MANY of the points below — 20 of 27 for the
first, 12 of 27 for the second, counted by
``test_the_fixtures_would_CATCH_a_float_reading_of_the_assume_region`` —
and the probe must read them TRUE.

--------------------------------------------------------------------------
WHAT THIS ORACLE DOES **NOT** REACH, MEASURED
--------------------------------------------------------------------------

An agreement count says how much was compared, not how much is covered,
and this file's counts are quoted in two module docstrings. So the reach
is measured the only way a checker's reach can be: **by breaking the thing
it checks and seeing whether it notices.** Thirty-two single-edit defects
in ``stelling/falsify.py`` — one text substitution each, applied to a
private copy of the package, no monkeypatching — were run against this
file. **It catches 10 of the 32.** Against the eight-fixture version this
file shipped with it caught 7; the three the two new fixtures below add
are named in ``test_the_two_fixtures_that_close_a_STRUCTURAL_gap_really_do``.

The 22 it misses are not all the same kind of miss, and the difference is
what a reader needs:

* **THREE ARE STRUCTURAL AND NO FIXTURE CAN CLOSE THEM.** This file calls
  ``_replay`` directly, so ``_window`` and ``_admissible`` — the code that
  decides which points exist at all — are outside its reach entirely:
  ``_admissible`` always-True and ``_window`` one-wider-on-each-side both
  stay GREEN here. So does ``_body_runs_once`` always-yes, because every
  fixture is a program the replay can read and none contains a loop or a
  branch body. Closing those means testing ``probe`` rather than
  ``_replay``, which is a different file's job
  (``tests/test_falsify_fire_condition.py`` drives all three).
* **THE REST ARE FIXTURE GAPS, AND THEY ARE NAMED RATHER THAN COUNTED.**
  No fixture separates ``max`` from ``min`` (``mixed-comparison`` clamps
  at the box edge, where both readings agree); none separates a STRICT
  comparison from its non-strict partner (``lt``/``le``, ``gt``/``ge``,
  ``eq``/``ne``); none states a boolean ``and``/``or``/``xor``; none
  reaches ``_REDUCTIONS`` (no ``jnp.sum``, ``prod``, ``max`` or ``min``
  over an axis) or ``_MOVEMENT`` (no ``reshape``, ``transpose``,
  ``concatenate``, ``slice`` or ``pad``); and none reaches
  ``_EXACT_UNARY`` beyond what the arithmetic above emits, so ``neg``,
  ``abs``, ``sign``, ``floor``, ``ceil`` and ``copy`` are all uncovered.
  The descent is covered for ``jit`` only, never for ``remat2``.
* **``jax_enable_x64`` IS FORCED ON HERE**, so the ``dtype-narrowed-by-jax``
  decline — the whole x64-off path — is untested by this file.

**NONE OF THAT MAKES THE AGREEMENT COUNT WRONG; IT MAKES IT NARROW.** What
the count is evidence for is what it says: over these ten programs and
this grid, the probe's exact reading and an independent ``Fraction``
reading of the same predicates agree at every point. Anything a reader
wants to conclude beyond that is on the list above.

**WHY THIS IS NOT CALLED ``test_falsify_oracle.py``, WHICH IS THE NAME ITS
SUBJECT WOULD SUGGEST.** ``tests/test_narrowing_perimeter.py`` records a
measured COLLECTION RANK — *"this file sorts 72nd of the files ``pytest
--collect-only -q -p no:randomly`` names in this tree"* — because that rank
is what sets the size of the exposure its incident describes, and
``test_this_files_position_in_the_collection_is_the_measured_one`` demands
the phrase verbatim in that file AND in ``CHANGELOG.md``. A new file
sorting before ``test_n…`` moves the rank to 73rd and puts both artefacts
one measurement behind: driven, that test goes red under
``test_falsify_oracle.py`` and green under this name. Correcting the two
artefacts is the right repair and it is not this branch's to make — the
CHANGELOG is held elsewhere — so the file is placed where the measurement
it would otherwise falsify stays true, and the reason is written here
rather than left as a naming curiosity. ``probe`` is the module's own word
for what is under test.
"""

from __future__ import annotations

from fractions import Fraction as Q

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import stelling.falsify as F  # noqa: E402
from stelling.harness import any_array, assert_, assume  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def traced(harness):
    return jax.make_jaxpr(harness)()


# --------------------------------------------------------------------------
# the fixtures: a harness, its declared box, and the SAME predicates written
# a second time in exact rational arithmetic
# --------------------------------------------------------------------------
#
# Every oracle below takes `v`, the tuple of declared element values as
# `Fraction`s in declaration order, and returns `(assumes, asserts)` as
# lists of Python bools -- exactly the pair `_replay` returns.
#
# The float literals are written IDENTICALLY on both sides on purpose:
# `Fraction(0.1)` is the exact value of the float64 the jaxpr carries, not
# one tenth, and an oracle that used `Q(1, 10)` would be checking a
# different program.


def _scaled_roundtrip():
    """``y*0.1*10.0 <= y`` over ``float64 [0, 2]``.

    True for every real ``y >= 0`` (the product is ``y`` exactly over ℚ,
    since ``0.1 * 10.0`` in ℚ is ``0.1``'s float value times ten, which is
    ABOVE one -- so the left side is ``>= y``)... which is why the
    obligation is stated the way it is below. This is the shape the module
    measured 47 of 55 float-admitted points falling OUTSIDE the region on.
    """
    y = any_array((), "float64", (0.0, 2.0))
    assume(y * 0.1 * 10.0 <= y)
    return assert_(y >= 0.0)


def _scaled_roundtrip_oracle(v):
    (y,) = v
    return [y * Q(0.1) * Q(10.0) <= y], [y >= 0]


def _kahan_cancellation():
    """``(1e16 + y) - 1e16 <= y``: true over ℚ, and false in float64."""
    y = any_array((), "float64", (0.0, 2.0))
    assume((1e16 + y) - 1e16 <= y)
    return assert_(y + 0.0 >= 0.0)


def _kahan_cancellation_oracle(v):
    (y,) = v
    return [(Q(1e16) + y) - Q(1e16) <= y], [y + 0 >= 0]


def _two_assumes():
    """Two assume EQUATIONS, so the reading is a list of two in order."""
    y = any_array((), "float64", (-1.0, 3.0))
    assume(y >= 0.25)
    assume(y * 2.0 <= 5.0)
    return assert_(y - y <= 0.0)


def _two_assumes_oracle(v):
    (y,) = v
    return [y >= Q(0.25), y * Q(2.0) <= Q(5.0)], [y - y <= 0]


def _assume_behind_a_jit():
    """The descent: the assume is one ``jit`` deep and must still be read."""
    y = any_array((), "float64", (0.0, 4.0))
    j = jax.jit(lambda a: (assume(a * a >= a), a)[1])(y)
    return assert_(j + 1.0 >= 1.0)


def _assume_behind_a_jit_oracle(v):
    (y,) = v
    return [y * y >= y], [y + Q(1.0) >= Q(1.0)]


def _elementwise_over_an_array():
    """An array declaration: the assume is ANDed over every element."""
    y = any_array((4,), "float64", (-2.0, 2.0))
    assume(y * y >= 0.0)
    return assert_(y * 0.5 + y * 0.5 - y <= 0.0)


def _elementwise_over_an_array_oracle(v):
    return (
        [all(e * e >= 0 for e in v)],
        [all(e * Q(0.5) + e * Q(0.5) - e <= 0 for e in v)],
    )


def _division_and_subtraction():
    """``/`` over ℚ is exact division, which is where float and ℚ part."""
    y = any_array((), "float64", (1.0, 3.0))
    assume(y / 3.0 * 3.0 <= y)
    return assert_(y / 4.0 >= 0.25)


def _division_and_subtraction_oracle(v):
    (y,) = v
    return [y / Q(3.0) * Q(3.0) <= y], [y / Q(4.0) >= Q(0.25)]


def _integer_program():
    """An all-integer declaration, where ℚ and the machine agree exactly."""
    y = any_array((), "int32", (-8, 8))
    assume(y * 2 >= y)
    return assert_(y * 3 - y * 2 <= y)


def _integer_program_oracle(v):
    (y,) = v
    return [y * 2 >= y], [y * 3 - y * 2 <= y]


def _mixed_comparison():
    """``min``/``max`` and a strict comparison, on an array."""
    y = any_array((3,), "float64", (-1.0, 1.0))
    assume(jnp.maximum(y, -1.0) >= -1.0)
    return assert_(jnp.minimum(y, 1.0) <= 1.0)


def _mixed_comparison_oracle(v):
    return (
        [all(max(e, Q(-1.0)) >= Q(-1.0) for e in v)],
        [all(min(e, Q(1.0)) <= Q(1.0) for e in v)],
    )


def _two_declarations():
    """TWO ``stelling_any`` declarations, which no other fixture has.

    ``_replay`` substitutes ``point[decl[0]]`` for each ``stelling_any``
    and advances ``decl[0]``; with one declaration in every fixture, a
    replay that read ``point[0]`` for ALL of them would be indistinguishable
    from the correct one, and a mutation to exactly that stays green. So
    the predicates below are chosen to separate ``(a, b)`` from
    ``(a, a)``: the boxes do not overlap, ``b - a >= 2.0`` is true at some
    grid points and false at others while ``a - a >= 2.0`` is false at all
    of them, and ``a + b <= 6.0`` is false at some while ``a + a <= 6.0``
    is true at all of them.
    """
    a = any_array((), "float64", (0.0, 2.0))
    b = any_array((), "float64", (3.0, 5.0))
    assume(b - a >= 2.0)
    return assert_(a + b <= 6.0)


def _two_declarations_oracle(v):
    a, b = v
    return [b - a >= Q(2.0)], [a + b <= Q(6.0)]


def _array_gate_partly_true():
    """An array whose assume is true of SOME elements and not others.

    Every other array fixture states an assume that holds at every
    element of every grid point, so ``all`` and ``any`` agree everywhere
    and a replay that aggregated the gate with ``any`` — admitting a
    point where one element satisfies the assume and three do not — reads
    identically to the correct one. ``y >= 0.0`` over ``[-2, 2]`` with a
    per-element offset separates them; the count of separating points is
    asserted in
    ``test_the_two_fixtures_that_close_a_STRUCTURAL_gap_really_do``.

    The OBLIGATION is separated the same way and for the same reason: an
    assert reading that aggregated with ``any`` — reporting an obligation
    satisfied because ONE element of the array satisfies it — read
    identically to the correct one on every other fixture here, because
    every other array obligation holds at every element of every point.
    """
    y = any_array((4,), "float64", (-2.0, 2.0))
    assume(y >= 0.0)
    return assert_(y + y <= 0.0)


def _array_gate_partly_true_oracle(v):
    return (
        [all(e >= 0 for e in v)],
        [all(e + e <= 0 for e in v)],
    )


# label, harness, oracle, declarations as ((dtype, (lo, hi), shape), ...)
CASES = [
    ("scaled-roundtrip", _scaled_roundtrip, _scaled_roundtrip_oracle,
     (("float64", (0.0, 2.0), ()),)),
    ("kahan-cancellation", _kahan_cancellation, _kahan_cancellation_oracle,
     (("float64", (0.0, 2.0), ()),)),
    ("two-assumes", _two_assumes, _two_assumes_oracle,
     (("float64", (-1.0, 3.0), ()),)),
    ("assume-behind-a-jit", _assume_behind_a_jit, _assume_behind_a_jit_oracle,
     (("float64", (0.0, 4.0), ()),)),
    ("elementwise-over-an-array", _elementwise_over_an_array,
     _elementwise_over_an_array_oracle, (("float64", (-2.0, 2.0), (4,)),)),
    ("division-and-subtraction", _division_and_subtraction,
     _division_and_subtraction_oracle, (("float64", (1.0, 3.0), ()),)),
    ("integer-program", _integer_program, _integer_program_oracle,
     (("int32", (-8, 8), ()),)),
    ("mixed-comparison", _mixed_comparison, _mixed_comparison_oracle,
     (("float64", (-1.0, 1.0), (3,)),)),
    ("two-declarations", _two_declarations, _two_declarations_oracle,
     (("float64", (0.0, 2.0), ()), ("float64", (3.0, 5.0), ()))),
    ("array-gate-partly-true", _array_gate_partly_true,
     _array_gate_partly_true_oracle, (("float64", (-2.0, 2.0), (4,)),)),
]

GRID = 27  # points per fixture; see `test_the_oracle_agreement_COUNT`


def _one_declaration_points(dtype, box, shape, offset=0):
    """A deterministic grid over one declared box. No stelling code.

    Every element of an array declaration gets its own offset so that the
    fixtures with a shape are not eight readings of one scalar, and
    ``offset`` walks a SECOND declaration through the grid out of step
    with the first, so that no point of a two-declaration fixture has the
    two declarations at the same position in their boxes.
    """
    lo, hi = box
    size = 1
    for d in shape:
        size *= d
    out = []
    for i in range(GRID):
        arr = np.empty(size, dtype=dtype)
        for j in range(size):
            t = ((i * size + j + offset) % GRID) / (GRID - 1)
            value = lo + t * (hi - lo)
            arr[j] = np.dtype(dtype).type(
                round(value) if np.dtype(dtype).kind in "iu" else value
            )
        out.append(arr.reshape(shape))
    return out


def _points(decls):
    """One grid per declaration, zipped into the tuple ``_replay`` takes."""
    per = [
        _one_declaration_points(dtype, box, shape, offset=7 * k)
        for k, (dtype, box, shape) in enumerate(decls)
    ]
    return [tuple(arrays) for arrays in zip(*per)]


def _as_rationals(point):
    """The point as exact rationals, in declaration order.

    ``Fraction(float)`` is exact. The declarations are flattened into one
    tuple because that is what the oracles take: element values in
    declaration order, which is also the order ``_replay`` consumes them.
    """
    out = []
    for arr in point:
        flat = np.asarray(arr).reshape(-1)
        if np.asarray(arr).dtype.kind in "iub":
            out.extend(Q(int(x)) for x in flat)
        else:
            out.extend(Q(float(x)) for x in flat)
    return tuple(out)


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------


def _readings():
    """Every (label, point, probe reading, oracle reading) this file drives."""
    for label, harness, oracle, decls in CASES:
        census = F._read(traced(harness))
        for point in _points(decls):
            want_assumes, want_asserts = oracle(_as_rationals(point))
            got_assumes, _ = F._replay(census, point, assumes_only=True)
            _, got_asserts = F._replay(census, point)
            yield (
                label, point,
                (got_assumes, got_asserts),
                (want_assumes, want_asserts),
            )


@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_the_exact_reading_agrees_with_an_independent_Fraction_oracle(label):
    """Fixture by fixture, so a disagreement names the program it is in."""
    for got_label, point, got, want in _readings():
        if got_label != label:
            continue
        assert got == want, (
            f"{label} at "
            f"{[np.asarray(a).reshape(-1).tolist() for a in point]}: the "
            f"probe's exact reading is {got} and an independent Fraction "
            f"reading of the same predicates is {want}"
        )


def test_the_oracle_agreement_COUNT_is_the_figure_the_prose_may_cite():
    """The number, re-derivable, in the tree, on every run.

    ``falsify.py`` and ``preconditions.py`` may cite a gate-reading count
    for retiring ``UNAUDITED``; this is the test that produces one. Both
    files name this test, so a reader who wants to check the figure runs
    it, and a figure that drifts fails here rather than going stale in a
    docstring — which is what happened to the numbers this replaced.
    """
    gate_readings = 0
    obligation_readings = 0
    agreements = 0
    for _label, _point, got, want in _readings():
        gate_readings += len(got[0])
        obligation_readings += len(got[1])
        agreements += int(got == want)

    assert gate_readings == 297, gate_readings
    assert obligation_readings == 270, obligation_readings
    assert agreements == len(CASES) * GRID == 270, agreements


@pytest.mark.parametrize(
    "label,exactly", [("scaled-roundtrip", 20), ("kahan-cancellation", 12)]
)
def test_the_fixtures_would_CATCH_a_float_reading_of_the_assume_region(
    label, exactly
):
    """An oracle that agrees everywhere agrees about nothing.

    Two fixtures state an assume whose FLOAT answer and whose ℚ answer are
    different over part of the grid — which is the shape this module
    measured on a clean VERIFIED, where 47 of 55 float-admitted points
    were not in the assumed region over ℚ. If the gate ever read the
    region in float, the comparison above would fail at exactly the points
    counted here.

    It is a separate test because the comparison above cannot tell a
    correct agreement from a tautology, and a fixture set that drifted
    into agreeing on every route would leave the count in the docstrings
    still passing and still meaning nothing.

    **THE COUNTS ARE ASSERTED EXACTLY, AND THEY USED TO BE FLOORS.** With
    ``>= 15`` and ``>= 10`` in place, ``falsify.py`` described both
    fixtures as differing *"over most of the grid"* — true of the first at
    20 of 27 and FALSE of the second at 12 of 27, which is under half. A
    floor cannot catch a sentence that rounds in the flattering direction;
    an exact count can, and both are now quoted rather than characterised.
    """
    disagreements = 0
    for got_label, harness, _oracle, decls in CASES:
        if got_label != label:
            continue
        for point in _points(decls):
            y = float(np.asarray(point[0]).reshape(-1)[0])
            if label == "scaled-roundtrip":
                in_float = (y * 0.1 * 10.0) <= y
                over_q = Q(y) * Q(0.1) * Q(10.0) <= Q(y)
            else:
                in_float = ((1e16 + y) - 1e16) <= y
                over_q = (Q(1e16) + Q(y)) - Q(1e16) <= Q(y)
            disagreements += int(in_float != over_q)

    assert disagreements == exactly, (
        f"{label}: {disagreements} of {GRID} points separate the float "
        f"reading of this assume from the ℚ one, and this test — and the "
        f"sentence in `falsify.py` that cites it — say {exactly}"
    )


@pytest.mark.parametrize(
    "label,exactly",
    [("array-gate-partly-true", 7), ("two-declarations", 20)],
)
def test_the_two_fixtures_that_close_a_STRUCTURAL_gap_really_do(
    label, exactly
):
    """The two gaps that were one fixture each, and the count that closes them.

    Measured with the mutation battery described in this file's docstring,
    against the eight fixtures this file shipped with:

    * ``stelling_any`` reading ``point[0]`` instead of ``point[decl[0]]``
      stayed GREEN, because no fixture had two declarations;
    * the assume gate aggregating an array with ``any`` instead of ``all``
      stayed GREEN, and so did the same inversion on the obligation,
      because no fixture had an array where the two differ.

    Three defects, two fixtures, and 7 of 32 caught became 10 of 32.

    A fixture that closes a gap can drift back into not closing it without
    failing anything — the agreement count keeps passing either way — so
    what is asserted here is the SEPARATING POWER itself: how many of the
    27 grid points read differently under the defect than under the
    correct reading. If that number goes to zero the fixture is decoration
    and this test says so.

    **THE COUNTS WERE FLOORS AND ARE NOW EXACT, FOR THE REASON ITS
    SIBLING FORTY LINES ABOVE ALREADY GIVES.**
    ``test_the_fixtures_would_CATCH_a_float_reading_of_the_assume_region``
    gave up its own floors because *"a floor cannot catch a sentence that
    rounds in the flattering direction"*, and this test kept them —
    with ``>= 6`` standing under a measured 7, so the floor was not even
    a description of the fixture it guarded. Both numbers are measured on
    jax 0.11.0 and 0.10.2 alike: **7 of 27** for the ``any``/``all``
    inversion and **20 of 27** for the second declaration. Asserted
    exactly, a fixture that gains separating power fails here too — which
    is right, because that number is quoted, and a quoted number that has
    moved is the defect this file exists to prevent.
    """
    case = next(c for c in CASES if c[0] == label)
    _label, _harness, oracle, decls = case
    separating = 0
    for point in _points(decls):
        v = _as_rationals(point)
        right = oracle(v)
        if label == "two-declarations":
            # THE DEFECT: every declaration reads the first one's value
            first = _as_rationals((point[0],) * len(point))
            wrong = oracle(first)
        else:
            # THE DEFECT: the array is aggregated with `any`, not `all`
            wrong = (
                [any(e >= 0 for e in v)],
                [any(e + e <= 0 for e in v)],
            )
        separating += int(right != wrong)
    assert separating == exactly, (
        f"{label}: {separating} of {GRID} points read differently under "
        f"the defect this fixture exists to catch, and this test says "
        f"{exactly}. At zero the fixture is decoration and the agreement "
        f"count above would pass with the defect in place; at any other "
        f"number the fixture has changed and the figure is stale"
    )


def test_the_oracle_shares_no_code_with_the_module_it_checks():
    """The independence claim, read off this file's own source.

    ``falsify.py``'s independence argument is enforced by a test rather
    than left to discipline (``tests/test_falsify_independence.py``), and
    the same standard applies to an oracle written to check it. What this
    file may touch of the module under test is ``_read`` (which builds the
    census) and ``_replay`` (which is the subject); every other name in
    ``stelling.falsify`` would make the oracle a second reading through the
    machinery it is reading.

    **THIS IS A DISCIPLINE RAIL, NOT A PROOF, AND THE DIFFERENCE IS ONE
    BUILTIN.** The scan below reads ``ast.Attribute`` nodes, so it sees
    ``F._exact(point)`` and does not see ``getattr(F, "_exact")(point)``.
    Driven both ways: patching one oracle to call ``F._exact`` turns this
    test RED, and patching it to call ``getattr(F, "_exact")`` leaves it
    GREEN with the independence just as gone. Nothing here can close that
    — a dynamic attribute lookup is not visible to a static scan, and the
    module-level ban would have to be enforced at runtime by something
    this file also controls — so what the rail buys is that the ordinary
    spelling of the mistake fails loudly and on purpose, not that the
    property holds. A reader auditing the independence claim reads the
    fixtures, not this test.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "F"
    }
    assert used <= {"_read", "_replay"}, (
        f"the oracle reached into {sorted(used - {'_read', '_replay'})} of "
        f"the module it is checking; it must share no code with it"
    )

    # and nothing is imported FROM it either, which the attribute scan
    # above would not see
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(m.startswith("stelling.falsify") for m in imported), (
        f"the oracle imports from the module it is checking: {imported}"
    )
    assert "stelling.falsify" in {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }, "this file no longer drives `stelling.falsify` at all"
