# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""What can be gated about ``tools/solver_battery.py``, and what cannot.

``docs/choosing-a-solver-backend.md`` carries a ten-row solver comparison
table whose harnesses were never committed. The page said so itself, in as
many words — *a reader cannot re-derive it* — and that sentence was honest and
unfixable by editing. ``tools/solver_battery.py`` is the fix: a battery a
reader can run.

**WHAT THIS FILE GATES IS THE PART OF THAT TABLE THAT IS NOT A TIMING**, and
the split is not a matter of taste. Three tiers:

* **Machine-independent, no backend needed.** The ``fragment`` column
  (``QF_LRA`` / ``QF_NRA``) and the declared-input count are computed by
  :mod:`stelling.obligation` off the traced jaxpr, before any backend is
  discovered. Both are gated, and both run in the no-solver lane —
  :func:`test_the_fragment_column_is_what_the_page_publishes` and
  :func:`test_each_row_declares_the_variables_its_label_names`. The second is
  the sharper one: it re-derives *32 vars* and *64-element array* from the
  jaxpr rather than trusting the label.
* **Needs a backend, still a mechanism claim.** Whether the obligation is TRUE
  is a fact about the obligation, not about a solver's speed: row 3's
  predicate is false at ``x = (1, …, 1)`` and any backend that decides it must
  answer ``sat``. Gated for the six cheap rows, at a budget generous enough
  (:data:`_MECHANISM_TIMEOUT_MS`) that the gate is about the answer and not
  about the clock.
* **A timing.** Everything else. Rows 7-10's published cells are *"did this
  backend finish inside ten seconds"*, which is a millisecond wearing a hat,
  and nothing here asserts one. They are carried in
  ``tests/test_doc_examples.py``'s ``BLIND_SPOT`` reason for that page instead.

**AND THE HARNESSES ARE NOT THE PAGE'S HARNESSES — BUT NOT EQUALLY, ROW BY
ROW.** That was the load-bearing disclosure of this batch and it was applied
as one uniform sentence to all ten rows, which is two mistakes. **Rows 4 and 5
name a mathematical object**, so their labels DO pin a predicate; and *"does
the label pin a harness"* is the wrong question, because no label anywhere
pins one completely. The question that decides re-derivability is **whether
the freedom the label leaves reaches the published number**, and it is
measured. So four things are gated here rather than one:

* every row still says what its label left open
  (:func:`test_every_row_says_what_the_page_left_open`);
* every row's GRADE — reconstructed, direction only, unsupported — is the same
  on the page as in the tool, and so is every mark, so neither record can
  quietly promote a row the other still doubts
  (:func:`test_the_page_s_marks_are_the_tool_s_refusals`);
* the two named polynomials are the polynomials their labels name, coefficient
  by coefficient, against a reference written here from their published
  definitions (:func:`test_the_named_objects_are_the_objects_their_labels_name`)
  — the only claims in the tool that no measurement can check, because both
  mutations that break them leave every fragment, count, outcome and cell
  identical;
* and the three-readings table on the page still shows the reversal and the
  two non-splits its own prose claims
  (:func:`test_the_three_readings_table_still_shows_what_the_prose_says`).
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest

from _solver_gate import need_solver

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
PAGE = REPO / "docs" / "choosing-a-solver-backend.md"

#: The name ``tools/solver_battery.py`` is registered under while it executes,
#: and only while — see :func:`_battery`. Deliberately not ``solver_battery``:
#: a canonical entry left in ``sys.modules`` is importable by every module
#: after this one, which is the leak this indirection exists to close.
_PRIVATE_MODULE_NAME = "_stelling_tests_solver_battery"


def _battery():
    """Load ``tools/solver_battery.py`` WITHOUT putting ``tools/`` on the path.

    ``tools/`` is not a package and is not importable surface — it ships in the
    sdist as scripts. An earlier version of this function inserted it into
    ``sys.path`` and its docstring said that doing it here "rather than at
    ``sys.path`` scope" avoided leaking into the rest of the session. **It did
    not.** This function runs at module scope (see below), so the insertion
    happened at import and outlived the module: measured with a
    ``pytest_sessionfinish`` hook, ``/…/tools`` was still on ``sys.path`` when
    the session ended. And ``sys.path`` is one of the channels
    ``tests/_state_guard.py`` names in its own list of what it does NOT watch,
    so nothing would have said so.

    ``spec_from_file_location`` loads the file by path: no ``sys.path`` entry,
    and no ``sys.modules`` entry that outlives this call.

    THE ``sys.modules`` ENTRY DURING EXEC IS NOT OPTIONAL, which is worth
    saying because deleting it looks like the obvious simplification. The tool
    uses ``from __future__ import annotations``, so its ``@dataclass``
    annotations are strings, and ``dataclasses._process_class`` resolves them
    through ``sys.modules[cls.__module__].__dict__`` **while the file is
    executing**. Without the registration that lookup returns ``None`` and
    collection dies with ``AttributeError: 'NoneType' object has no attribute
    '__dict__'`` — measured. It is registered under a private name that no
    other module would import by accident, and removed in a ``finally``.

    **THIS MODULE'S COLLECTION DEPENDS ON THE TOOL BEING STDLIB-ONLY**, which
    is a contract :func:`test_the_module_imports_jax_nowhere_at_module_scope`
    holds in the lanes that have jax. In a lane that does NOT — the zero-dep
    job — breaking that contract errors here, at collection, and takes this
    whole module down rather than failing one test. That is loud and it is the
    safe direction, but a bare ``ModuleNotFoundError: jax`` from a file called
    ``test_solver_battery.py`` reads like a missing test dependency and is not
    one, so the cause is named on the way past."""
    spec = importlib.util.spec_from_file_location(
        _PRIVATE_MODULE_NAME, TOOLS / "solver_battery.py")
    assert spec is not None and spec.loader is not None, (
        f"tools/solver_battery.py is not loadable as a file: {TOOLS}"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PRIVATE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:  # pragma: no cover - only under the mutation
        raise ImportError(
            f"tools/solver_battery.py could not be imported: {e}. That tool is "
            f"stdlib-only BY CONTRACT — `--rows` has to work in an environment "
            f"with no jax and no solver — so an import error here is a defect "
            f"in the tool, not a missing dependency of this test module. See "
            f"test_the_module_imports_jax_nowhere_at_module_scope."
        ) from e
    finally:
        sys.modules.pop(_PRIVATE_MODULE_NAME, None)
    return module


battery = _battery()


#: Generous on purpose. The six cheap rows answer in tens of milliseconds; a
#: gate at the page's own 10 s budget would be a gate that goes red when the
#: box is loaded, which is how a mechanism claim turns into a timing claim
#: without anybody deciding to make it one.
_MECHANISM_TIMEOUT_MS = 120_000


@pytest.fixture
def x64():
    """Put jax in the configuration the page's table was taken under, and put
    it back afterwards.

    ``tools/solver_battery.py`` sets ``jax_enable_x64`` itself, because the
    page's provenance names it and a battery measuring at float32 would be
    declaring different boxes and emitting different rationals. Inside a
    pytest session that is PROCESS-GLOBAL state, and ``tests/_state_guard.py``
    is right to refuse it: a later test inheriting x64 does not go red, it
    goes GREEN for a reason nobody chose. Measured — without this fixture the
    guard reported
    ``jax:enable_x64: {'jax_enable_x64': False} -> {'jax_enable_x64': True}``
    against :func:`test_the_fragment_column_is_what_the_page_publishes`.
    """
    jax = pytest.importorskip("jax")
    before = bool(jax.config.read("jax_enable_x64"))
    battery._configure_jax()
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", before)


# ------------------------------------------------- the tool and the page agree


def _markdown_table(header: str, what: str) -> list[tuple[str, ...]]:
    """The rows under ``header``, as published, cell by cell.

    Located by its header rather than by line number, and asserted to be
    exactly one such table — a parser that silently found nothing would make
    every comparison built on it vacuous, which is the failure mode this whole
    module is written against."""
    text = PAGE.read_text(encoding="utf-8")
    assert text.count(header) == 1, (
        f"expected exactly one {what} header in {PAGE.name}; found "
        f"{text.count(header)}. Comparisons in this file are made against "
        f"that table, so a moved or duplicated header silently empties them."
    )
    lines = text.split(header, 1)[1].splitlines()
    rows = []
    for line in lines:
        if not line.startswith("|"):
            # the remainder of the header line itself, then the blank line
            # after the table -- stop at the second
            if rows:
                break
            continue
        if set(line) <= set("|- "):  # the |---|---| separator
            continue
        rows.append(tuple(c.strip() for c in line.strip().strip("|").split("|")))
    return rows


#: The page's comparison table header, INCLUDING the reconstruction column
#: added 2026-08-23. Spelled once; both the row-for-row gate and the grade
#: gate below locate the table with it.
_PAGE_TABLE_HEADER = (
    "| obligation | fragment | both | z3 alone | cvc5 alone | "
    "reconstruction (2026-08-23) |"
)

#: The header of the three-readings table — the one that carries the finding.
_READINGS_HEADER = "| reading of the label | z3 alone | cvc5 alone |"


def _page_table() -> list[tuple[str, ...]]:
    """The ten rows of the page's comparison table, as published.

    The fragment cell is written as a markdown code span and bare in the
    tool's plain-text table. Stripped here, in ONE place, so the comparison
    is about the fragment and not about the page's code-span convention — and
    stripped only from that cell, so a change to any other cell's spelling
    still goes red."""
    rows = _markdown_table(_PAGE_TABLE_HEADER, "comparison-table")
    return [r[:1] + (r[1].strip("`"),) + r[2:] for r in rows]


#: The marks the page's ``reconstruction`` column carries, and what each one
#: has to correspond to inside the tool. A mark on the page with nothing
#: behind it in the tool — or the reverse — is exactly how the two tables this
#: page used to carry came to mark the same row differently.
_MARK_MEANS = {
    "‡": "contested",        # the tool refuses to file a number against it
    "†": "published_notes",  # the tool records a defect in the page's own cells
}


def _grade_of(cell: str) -> str:
    """The bare grade word from a ``reconstruction`` cell, marks removed."""
    for mark in _MARK_MEANS:
        cell = cell.replace(mark, "")
    return cell.strip()


def test_the_battery_is_the_page_s_table_row_for_row():
    """The tool and the page cannot drift apart, in either direction.

    Row labels, order, fragment, all three published cells AND the
    reconstruction grade. This is what lets ``tools/solver_battery.py`` carry
    the page's numbers verbatim beside its own: if somebody edits a cell on the
    page, the tool's copy of it goes red here rather than quietly becoming a
    second, disagreeing record.
    """
    published = _page_table()
    assert len(published) == 10, (
        f"the page's comparison table has {len(published)} rows, not ten"
    )
    assert all(len(row) == 6 for row in published), (
        "every row of that table must carry six cells: the five published in "
        "2026-08 and the reconstruction grade added 2026-08-23.\n"
        + "\n".join(f"  row {i + 1}: {len(r)} cells {r!r}"
                     for i, r in enumerate(published) if len(r) != 6)
    )
    ours = [
        (r.name, r.fragment, r.page_both, r.page_z3, r.page_cvc5, r.grade)
        for r in battery.ROWS
    ]
    theirs = [tuple(row[:5]) + (_grade_of(row[5]),) for row in published]
    assert theirs == ours, (
        "tools/solver_battery.py's ROWS no longer reproduce the page's table.\n"
        + "\n".join(
            f"  row {i + 1}: page {p!r}\n           tool {o!r}"
            for i, (p, o) in enumerate(zip(theirs, ours)) if p != o
        )
    )


def test_every_grade_is_one_of_the_three_the_tool_defines():
    """A fourth grade is a fourth meaning, and nothing would say what it is.

    ``GRADES`` maps each grade to what it licenses a reader to do. A row (or a
    page cell) carrying a word that is not a key of it is claiming something
    undefined, which is how one uniform disclaimer became three grades in the
    first place — by nobody having to say what the disclaimer meant.
    """
    unknown = sorted({r.grade for r in battery.ROWS} - set(battery.GRADES))
    assert not unknown, (
        f"rows carry grades the tool does not define: {unknown}; known: "
        f"{sorted(battery.GRADES)}"
    )
    assert all(battery.GRADES.values()), (
        "a grade with an empty meaning is a word, not a grade"
    )
    unused = sorted(set(battery.GRADES) - {r.grade for r in battery.ROWS})
    assert not unused, (
        f"the tool defines grades no row uses: {unused}. A partition with an "
        f"empty class is not a partition of these rows."
    )


def test_the_page_s_marks_are_the_tool_s_refusals():
    """The marks on the page and the refusals in the tool are ONE fact.

    ``‡`` says *no number produced by any reading may be filed against this
    row*, which is exactly what ``Row.contested`` records; ``†`` says *this
    row's own published cells are internally inconsistent*, which is exactly
    what ``Row.published_notes`` records. Either side dropping a mark the other
    keeps is a silent disagreement between two records of the same thing —
    which this page has already had once, when it carried two ten-row tables
    that marked row 7 differently.

    Watched to fail: emptying row 7's ``contested`` string turns this red. It
    used to be green, because the only gate on ``contested`` asked whether SOME
    row was contested, and row 8 alone satisfied that.
    """
    published = _page_table()
    for mark, attr in _MARK_MEANS.items():
        marked = {i + 1 for i, row in enumerate(published) if mark in row[5]}
        in_tool = {r.n for r in battery.ROWS if getattr(r, attr)}
        assert marked == in_tool, (
            f"the page marks rows {sorted(marked)} with {mark!r} and the tool "
            f"records `{attr}` on rows {sorted(in_tool)}. They are the same "
            f"fact and must be the same set."
        )
    # ...and the ‡ mark must land on exactly the unsupported rows, so the mark,
    # the grade and the refusal cannot come apart from each other either.
    unsupported = {r.n for r in battery.ROWS
                   if r.grade == battery.GRADE_UNSUPPORTED}
    contested = {r.n for r in battery.ROWS if r.contested}
    assert unsupported == contested, (
        f"rows graded {battery.GRADE_UNSUPPORTED!r}: {sorted(unsupported)}; "
        f"rows carrying a `contested` refusal: {sorted(contested)}. A row is "
        f"unsupported BECAUSE its readings contest it; a grade with no refusal "
        f"behind it is a word."
    )


def test_every_row_says_what_the_page_left_open():
    """Every row must still say which parameters its label left open.

    ``chosen`` is the list of parameters the page's row LABEL does not fix and
    this battery therefore picked — the declared box, the predicate, the
    association of a monomial, what was timed. It is never empty, because no
    label anywhere fixes everything.

    **It is not a verdict on the row**, and reading it as one is what made row
    4 read as weakly as row 7. Whether a choice MATTERS is :attr:`Row.grade`,
    and that is measured. ``fixed`` is the other half and is equally required:
    a row that says nothing about what its label WAS good for cannot support a
    grade either.
    """
    naked = [r.n for r in battery.ROWS if not r.chosen]
    assert not naked, (
        f"rows {naked} declare nothing under `chosen`. Every row label on "
        f"that page leaves at least the declared box open."
    )
    silent = [r.n for r in battery.ROWS if not r.fixed]
    assert not silent, (
        f"rows {silent} declare nothing under `fixed`, so nothing says what "
        f"the label was good for"
    )


def test_a_contested_row_has_the_alternate_readings_that_contest_it():
    """A row is only "contested" if the readings that contest it are here.

    Otherwise ``contested`` is an assertion in a docstring — the thing this
    whole campaign keeps finding. Each contested row must carry at least one
    entry in ``VARIANTS``, which is a harness the tool actually drives; and so
    must each ``direction only`` row, whose grade rests on exactly the same
    kind of evidence — one reading that splits and one that does not.
    """
    contested = {r.n for r in battery.ROWS if r.contested}
    assert contested, "no row is marked contested; rows 7 and 8 were"
    varied = {v.row for v in battery.VARIANTS}
    needs_a_reading = contested | {
        r.n for r in battery.ROWS if r.grade == battery.GRADE_DIRECTION_ONLY}
    missing = sorted(needs_a_reading - varied)
    assert not missing, (
        f"rows {missing} are graded on how their readings compare and ship no "
        f"alternate reading. A contest with one contestant is a claim, not a "
        f"measurement."
    )
    stray = sorted(v.row for v in battery.VARIANTS
                   if v.row not in {r.n for r in battery.ROWS})
    assert not stray, f"VARIANTS name rows that do not exist: {stray}"


# ------------------------------------------- the two objects a label NAMES

#: A grid of exactly-representable points. Every value below is dyadic with a
#: small numerator, so both sides of every comparison in
#: :func:`test_the_named_objects_are_the_objects_their_labels_name` are exact
#: in binary64 and can be compared with ``==`` rather than a tolerance — which
#: matters, because a tolerance is exactly what would let a changed
#: coefficient through.
_GRID = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def _amgm_reference(x, y):
    """AM–GM's two-variable degree-2 form, written from the inequality.

    The arithmetic mean of ``x²`` and ``y²`` is at least their geometric mean,
    which at two variables and degree two is ``x² + y² ≥ 2xy``. Written here,
    independently of the tool, because a reference that imported the tool's
    own expression would agree with any mutation of it.
    """
    return x ** 2 + y ** 2, 2 * x * y


def _motzkin_reference(x, y):
    """The Motzkin polynomial as published: ``x⁴y² + x²y⁴ − 3x²y² + 1``.

    The first known nonnegative polynomial that is not a sum of squares. The
    ``−3`` is the whole of what makes it that one: at ``−2`` it is a different
    polynomial, still nonnegative, still degree 6, still in two variables —
    and still ``unsat``, which is why no measurement can tell them apart.
    """
    return x ** 4 * y ** 2 + x ** 2 * y ** 4 - 3 * x ** 2 * y ** 2 + 1


def test_the_named_objects_are_the_objects_their_labels_name():
    """The only two claims in this tool that no measurement can check.

    Rows 4 and 5 are the only rows whose LABEL pins a predicate, which makes
    their polynomials a claim about the page rather than a choice of this
    battery — and ``Row.fixed`` states them in as many words. Nothing else
    here can hold them: ``2xy → 1.5xy`` leaves a true inequality, and Motzkin's
    ``−3 → −2`` leaves a nonnegative degree-6 polynomial in two variables. Both
    mutations were applied to the harnesses and both left the whole module
    green, every fragment, every input count, every outcome and every table
    cell unchanged.

    So this compares the polynomials themselves, against references written
    above from their published definitions, on a grid of dyadic points where
    binary64 is exact — and then checks the two structural properties that
    make each one the object it is named for.
    """
    for x in _GRID:
        for y in _GRID:
            lhs, rhs = battery.amgm_sides(x, y)
            ref_lhs, ref_rhs = _amgm_reference(x, y)
            assert (lhs, rhs) == (ref_lhs, ref_rhs), (
                f"row 4's harness is no longer AM–GM's degree-2 form at "
                f"({x}, {y}): it compares {lhs} >= {rhs}, and "
                f"x² + y² >= 2xy is {ref_lhs} >= {ref_rhs}"
            )
            got = battery.motzkin_value(x, y)
            ref = _motzkin_reference(x, y)
            assert got == ref, (
                f"row 5's harness is no longer the Motzkin polynomial at "
                f"({x}, {y}): it evaluates to {got}, and "
                f"x⁴y² + x²y⁴ − 3x²y² + 1 is {ref}"
            )

    # AM–GM's equality case IS the inequality's content: the two sides meet
    # exactly on the diagonal, and nowhere else. A coefficient that is not 2
    # separates them there.
    for v in _GRID:
        lhs, rhs = battery.amgm_sides(v, v)
        assert lhs == rhs, (
            f"AM–GM is tight at x == y and this harness is not: at x = y = "
            f"{v} it compares {lhs} >= {rhs}"
        )

    # Motzkin's four zeros. x⁴y² + x²y⁴ − 3x²y² + 1 vanishes at (±1, ±1) —
    # that tightness is why it is the standard nonnegative-but-not-SOS
    # example. At −2 the same points give 1.
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            assert battery.motzkin_value(x, y) == 0.0, (
                f"the Motzkin polynomial vanishes at ({x}, {y}) and this "
                f"harness gives {battery.motzkin_value(x, y)}"
            )
    assert battery.motzkin_value(0.0, 0.0) == 1.0

    # ...and the rows that make the claim are the rows that carry a named
    # object, so a new named object cannot arrive without a gate.
    named = {r.n for r in battery.ROWS if r.named_object}
    assert named == {4, 5}, (
        f"rows carrying a named object: {sorted(named)}. This test knows how "
        f"to check rows 4 and 5; a row that starts naming an object without a "
        f"check here is the ungated claim this test exists to end."
    )


def test_every_builder_is_reachable_and_every_row_has_one():
    used = {r.build for r in battery.ROWS} | {v.build for v in battery.VARIANTS}
    assert used <= set(battery.BUILDERS), (
        f"rows/variants name builders that do not exist: "
        f"{sorted(used - set(battery.BUILDERS))}"
    )
    assert set(battery.BUILDERS) <= used, (
        f"BUILDERS carries entries nothing drives: "
        f"{sorted(set(battery.BUILDERS) - used)}. A harness no row runs is a "
        f"harness nobody measures."
    )


#: The page's instruction to its own reader, as one string rather than as two
#: words that happen to appear somewhere on it. The earlier version of the
#: test below asserted only that the substrings ``direction`` and
#: ``milliseconds`` occurred, on a 600-line page where they occur seven and
#: four times — so it would have passed a page that INVERTED the instruction.
_THE_INSTRUCTION = (
    "Read the ten rows for their *direction*, not their milliseconds"
)


def test_the_page_tells_the_reader_how_to_re_derive_the_table():
    """The page's provenance paragraph must name the tool and keep the caveat.

    Both halves. Naming the tool without the machine-dependence caveat would
    swap an honest "you cannot re-derive this" for a false "you can": the
    fragment column reproduces anywhere, the milliseconds do not.
    """
    text = PAGE.read_text(encoding="utf-8")
    assert "tools/solver_battery.py" in text, (
        "the page does not name the battery, so a reader still has no route "
        "to the table"
    )
    assert text.count(_THE_INSTRUCTION) == 1, (
        f"the page must carry its own instruction, once, in these words:\n"
        f"  {_THE_INSTRUCTION!r}\n"
        f"found {text.count(_THE_INSTRUCTION)} times. Asserting that the "
        f"WORDS `direction` and `milliseconds` appear somewhere would pass a "
        f"page that told the reader to do the opposite."
    )


# ------------------------------------ the table that carries the finding


def test_the_three_readings_table_still_shows_what_the_prose_says():
    """The page's headline nonlinear finding, held against its own table.

    That table is the finding: three readings of one row label, one of which
    runs the published direction BACKWARDS and two of which show no split at
    all. Every cell in it is a millisecond and none is gated — but *"z3
    finished and cvc5 did not"* is not a millisecond, it is the claim the
    prose beside it makes, and a table that stopped showing it would make the
    prose false without making any number wrong.

    Watched to fail: reversing the literal reading's two cells to the page's
    own direction — the headline finding inverted on the page — turns this
    red. It used to be green: nothing anywhere read this table.
    """
    rows = _markdown_table(_READINGS_HEADER, "three-readings-table")
    assert len(rows) == 4, (
        f"the three-readings table has {len(rows)} rows, not four (three "
        f"readings and the published row they are read against)"
    )
    assert all(len(r) == 3 for r in rows), rows

    def decided(cell: str) -> bool:
        """Did this backend finish? The two spellings the table uses."""
        assert ("unsat" in cell) != ("UNKNOWN" in cell), (
            f"a cell of that table says neither `unsat` nor `UNKNOWN`, so "
            f"nothing can be read off it: {cell!r}"
        )
        return "unsat" in cell

    *readings, published = rows
    row7 = next(r for r in battery.ROWS if r.n == 7)

    # THE LAST ROW IS THE PAGE'S OWN CELLS, and it is a hand-copy like any
    # other, so it is held to the tool's copy of them byte for byte.
    assert "as published" in published[0], (
        f"the last row of that table must be the published row it reads the "
        f"others against; it is {published[0]!r}"
    )
    assert (published[1], published[2]) == (row7.page_z3, row7.page_cvc5), (
        f"the `as published` row no longer carries row 7's published cells.\n"
        f"  table: {published[1]!r} / {published[2]!r}\n"
        f"  row 7: {row7.page_z3!r} / {row7.page_cvc5!r}"
    )

    literal, *no_split = readings
    assert "the literal one" in literal[0], (
        f"the first reading must be the literal one — it is the one the prose "
        f"says reverses the page; it is {literal[0]!r}"
    )

    # THE REVERSAL, as a reversal: literally the published row's pattern with
    # both backends swapped. Not "z3 was fast", which a re-timing could make
    # true or false; the DIRECTION, which is what the page asks to be read.
    pub = (decided(published[1]), decided(published[2]))
    lit = (decided(literal[1]), decided(literal[2]))
    assert lit == (not pub[0], not pub[1]), (
        f"the literal reading no longer reverses the published row. Published "
        f"z3/cvc5 finished: {pub}; the literal reading's: {lit}. The prose "
        f"beside this table says the first reading `reverses` this page's "
        f"direction; if the table stops showing that, the prose is false."
    )

    for reading in no_split:
        assert decided(reading[1]) and decided(reading[2]), (
            f"the prose says the other two readings show NO SPLIT — both "
            f"backends finishing. This one does not: {reading!r}"
        )

    text = PAGE.read_text(encoding="utf-8")
    assert "The first **reverses** this page's direction." in text, (
        "the sentence this table is checked against is gone from the page, so "
        "the check above is holding the table to nothing"
    )


def test_the_page_carries_no_second_copy_of_the_ten_row_table():
    """One record of those numbers, not two.

    The page used to print a full ten-row hand-copy of the battery's cells
    beside the published table. It was ungated (setting a cell to
    ``999–999 ms`` left this whole module green), it marked row 7 differently
    from the table above it, and it was a hand-copied snapshot on a page whose
    entire complaint is a hand-copied snapshot. The numbers live in the tool;
    what the page keeps is what the driving settled.
    """
    text = PAGE.read_text(encoding="utf-8")
    five_column = [
        line for line in text.splitlines()
        if line.startswith("|") and line.count("|") >= 6
        and "both" in line and "z3 alone" in line and "cvc5 alone" in line
    ]
    assert five_column == [_PAGE_TABLE_HEADER], (
        "exactly one table on that page may carry the both / z3 alone / cvc5 "
        "alone columns, and it is the published one. Found:\n"
        + "\n".join(f"  {line}" for line in five_column)
    )


# ----------------------------------------------- the machine-independent half


def test_the_fragment_column_is_what_the_page_publishes(x64):
    """The mechanism column, re-derived — no backend, no wall clock.

    ``stelling.obligation._Slicer._fragment`` decides ``QF_LRA`` vs ``QF_NRA``
    off the traced jaxpr, before ``escalate`` discovers a backend. So this is
    the one column of that ten-row table that a reader in the zero-solver lane
    can check for themselves, and it is machine-independent: the same tree
    gives the same answer on any box.
    """
    wrong = []
    for r in battery.ROWS:
        fragment, reason, _inputs, _terms = battery.classify(r.build, r.n)
        if fragment != r.fragment:
            wrong.append(f"  row {r.n} ({r.name}): page {r.fragment}, "
                         f"measured {fragment or '(none)'} {reason}")
    assert not wrong, (
        "the fragment column no longer reproduces the page's:\n"
        + "\n".join(wrong)
    )


#: What each row LABEL says about its declared inputs, re-derived from the
#: jaxpr rather than trusted. "32 vars", "64-element array", "2 vars" are
#: claims the slice can answer, and the page's row 8 says sixty-four.
_DECLARED_INPUTS = {1: 1, 2: 64, 3: 8, 4: 2, 5: 2, 6: 1,
                    7: 32, 8: 64, 9: 10, 10: 12}


def test_each_row_declares_the_variables_its_label_names(x64):
    """`32 vars` is checkable; `166–175 ms` is not. This checks the first.

    The count comes from ``ObligationSlice.inputs`` — the declaration-order
    tuple the emission actually built — so a harness that drifted away from
    its own label goes red here rather than being measured under it.
    """
    wrong = []
    for r in battery.ROWS:
        _f, reason, inputs, _t = battery.classify(r.build, r.n)
        if inputs != _DECLARED_INPUTS[r.n]:
            wrong.append(f"  row {r.n} ({r.name}): label says "
                         f"{_DECLARED_INPUTS[r.n]} declared inputs, the slice "
                         f"has {inputs} {reason}")
    assert not wrong, "\n".join(["a harness drifted from its own label:", *wrong])


def test_no_row_is_decided_before_a_backend_is_asked(x64):
    """A row interval propagation DECIDES is not a row of this table at all.

    Nothing escalates, no backend is invoked, and the cell would be blank for
    a reason that has nothing to do with either solver. The page never states
    this constraint — it is in every row's ``chosen`` list precisely because
    it had to be discovered rather than read — so it is pinned here: every
    harness must still be undecided when the slicer sees it.
    """
    decided = []
    for r in battery.ROWS:
        fragment, reason, _i, _t = battery.classify(r.build, r.n)
        if not fragment:
            decided.append(f"  row {r.n} ({r.name}): {reason}")
    assert not decided, (
        "these harnesses never reach a backend, so they measure nothing "
        "about one:\n" + "\n".join(decided)
    )


# ------------------------------------------------ mechanism, with a backend

#: The answer each cheap row's OBLIGATION forces, which is a fact about the
#: predicate and not about a solver. Rows 7-10 are deliberately absent: their
#: published cells are "finished inside ten seconds", and that is a timing.
_MECHANISM_OUTCOME = {1: "unsat", 2: "unsat", 3: "sat",
                      4: "unsat", 5: "unsat", 6: "sat"}


@need_solver
@pytest.mark.parametrize("n", sorted(_MECHANISM_OUTCOME))
def test_the_cheap_rows_answer_what_their_obligation_forces(n, x64):
    """Row 3 is false; a backend that decides it must say ``sat``.

    This is the strongest thing about the table that a gate can hold: not how
    fast, but WHAT. It is run at :data:`_MECHANISM_TIMEOUT_MS` rather than the
    page's 10 s so that a loaded box cannot turn a mechanism gate into a
    timing gate.
    """
    row = next(r for r in battery.ROWS if r.n == n)
    cell = battery.measure_cell(row.build, n, None, _MECHANISM_TIMEOUT_MS, 1)
    assert cell.outcome == _MECHANISM_OUTCOME[n], (
        f"row {n} ({row.name}) answered {cell.outcome!r}, and its obligation "
        f"forces {_MECHANISM_OUTCOME[n]!r}.\n"
        f"  verdict status: {cell.status}\n"
        f"  reason        : {cell.reason}\n"
        f"  invocations   : {cell.invocations}"
    )


# ------------------------------------------------------- the empty environment


def _hide(monkeypatch, *names):
    """Hide optional dependencies from ``stelling._optional.available``.

    This is the page's own method for its single-backend columns — *"nothing
    was uninstalled"* — and it reaches both consumers that matter: the tool's
    environment probe and ``stelling.solvers._backends_for``, which asks
    through the same module attribute.
    """
    from stelling import _optional

    real = _optional.available
    monkeypatch.setattr(
        _optional, "available",
        lambda name: False if name in names else real(name),
    )
    monkeypatch.setattr(_optional, "cvc5_binary", lambda: None)


def test_it_runs_with_no_backend_and_names_every_cell_it_could_not_measure(
    monkeypatch, capsys, x64
):
    """The zero-dep lane is a promise; a battery most readers cannot run is
    not much of a remedy.

    With neither backend reachable the tool must still exit 0, still report
    the whole machine-independent column, and account for every one of the
    thirty outcome cells rather than printing a table with holes in it.
    """
    pytest.importorskip("jax")
    _hide(monkeypatch, "z3", "cvc5")
    rc = battery.main(["--only-rows", "1,4"])
    out = capsys.readouterr().out
    assert rc == 0, "a solver-free environment is not an error"
    assert "QF_LRA" in out and "QF_NRA" in out, (
        "the fragment column needs no backend and must still be reported"
    )
    assert "ROWS AND CELLS THIS RUN COULD NOT MEASURE, AND WHY" in out
    assert 'pip install "stelling[solvers]"' in out, (
        "the reason must name what to install, not merely that something is "
        "missing"
    )
    for n in (1, 4):
        for label in ("both", "z3 alone", "cvc5 alone"):
            assert f"{n}[{label}]" in out, (
                f"row {n}'s {label} cell is not accounted for anywhere in the "
                f"output; a cell that is silently absent is the defect this "
                f"tool exists to end"
            )
    # ...and nothing may claim a finding it did not measure.
    assert "FINDING 1 (QF_LRA: both backends decide everything): NOT MEASURED" in out
    assert "DID NOT HOLD" not in out, (
        "an unmeasured finding was rendered as a negative result"
    )


def test_it_runs_with_no_jax_and_names_jax_as_what_is_missing(monkeypatch, capsys):
    """...and a fragment nothing traced is NOT a fragment that DISAGREES.

    The mechanism column compared its measured fragment with the page's and
    called everything else ``<- DISAGREES``. With no jax there is no measured
    fragment for any row, so the no-jax lane printed TEN of them — measured —
    while the section below it correctly said jax was missing. That is the same
    rule the no-backend test above asserts, broken one renderer over: an
    unmeasured thing must never render as a negative result.
    """
    _hide(monkeypatch, "jax")
    rc = battery.main(["--only-rows", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "jax             : NOT INSTALLED" in out
    assert 'pip install "stelling[jax]"' in out, (
        "with no jax nothing can be traced, and the tool must name the extra "
        "rather than the symptom"
    )
    assert "DISAGREES" not in out, (
        "a row whose fragment could not be measured was rendered as a row "
        "whose fragment disagrees with the page's"
    )
    assert "NOT MEASURED" in out, (
        "and it must say which it is, rather than printing a blank column"
    )
    assert "DID NOT HOLD" not in out


def test_a_repeat_that_raised_is_not_a_finding_that_did_not_hold(
    monkeypatch, capsys, x64
):
    """The same rule again, in the direction report, where it cost a verdict.

    ``Finding 1`` counted a cell as ATTEMPTED whenever its outcome was not the
    literal string ``not measured`` — and a repeat that RAISED leaves the
    string ``error``. So one linear harness blowing up printed ``FINDING 1 …
    DID NOT HOLD — 2 of 3 linear rows decided``, with the next line still
    saying ``on every linear row``. Measured, by making row 3's harness raise;
    reproduced here.

    Row 3's harness raises on the LAST of each cell's three repeats rather than
    on all of them, which is the case that actually distinguishes the fix: a
    cell whose every repeat raised has no milliseconds at all, so even the old
    predicate would have called it unattempted. The defect only bites when
    some repeats measured something and a later one did not.
    """
    calls = {"n": 0}
    real = battery.BUILDERS["array8_linear_false"]

    def flaky(row_n):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            def array8_linear_false():
                raise RuntimeError("simulated harness failure")
            return array8_linear_false
        return real(row_n)

    monkeypatch.setitem(battery.BUILDERS, "array8_linear_false", flaky)
    rc = battery.main(["--only-rows", "1,2,3", "--repeats", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    finding = [line for line in out.splitlines() if "FINDING 1" in line]
    assert "DID NOT HOLD" not in out, (
        "a row that raised on one repeat was counted as a row that failed to "
        "be decided:\n" + "\n".join(finding)
    )
    assert "2 of the 2 linear rows MEASURED" in out, (
        "the errored row must be OUTSIDE the count, not inside it as a "
        "failure:\n" + "\n".join(finding)
    )
    assert "NOT MEASURED on row(s) 3" in out, (
        "the errored row must be named as unmeasured rather than dropped"
    )
    assert "simulated harness failure" in out, (
        "and the reason it was not measured must survive to the reader"
    )


def test_the_environment_probe_asks_the_same_question_stelling_answers(
    monkeypatch, x64,
):
    """Which backends will RUN is stelling's question, not this tool's.

    The probe used to decide backend presence from ``_optional.version``, i.e.
    from the WHEEL — the exact proxy ``tests/_solver_gate.py`` documents at
    length as already-found-and-fixed. Measured with a ``cvc5`` shim on PATH
    and both wheels hidden: ``solvers._backends_for`` returned the binary while
    the battery printed *"no SMT backend is installed"* for all thirty cells,
    having already probed and printed the binary's path two lines above. It
    told a reader who had followed this page's own ``STELLING_CVC5``
    instructions to install what they already had.
    """
    from stelling import _optional, solvers

    def stelling_says() -> set[str]:
        backends, _missing = solvers._backends_for(
            solvers.SolverConfig(timeout_ms=1000))
        return {b.name for b in backends}

    env = battery.probe_environment(1000, 1)
    assert set(env.backends) == stelling_says(), (
        f"the tool thinks {sorted(env.backends)} will run and stelling will "
        f"run {sorted(stelling_says())}"
    )
    assert set(env.backends) | set(env.missing_backends) == {"z3", "cvc5"}
    assert not set(env.backends) & set(env.missing_backends)

    # ...and the third route, which is the one the wheel proxy cannot see.
    real = _optional.available
    monkeypatch.setattr(
        _optional, "available",
        lambda name: False if name in ("z3", "cvc5") else real(name))
    monkeypatch.setattr(_optional, "cvc5_binary", lambda: "/nonexistent/cvc5")
    monkeypatch.delenv("STELLING_CVC5", raising=False)

    shimmed = battery.probe_environment(1000, 1)
    assert set(shimmed.backends) == stelling_says() == {"cvc5"}, (
        "with both wheels hidden and an external cvc5 binary configured, "
        "stelling runs cvc5 and the tool must say so"
    )
    assert "external binary" in shimmed.cvc5_route, shimmed.cvc5_route
    assert shimmed.missing_backends == ("z3",)


def test_the_inventory_needs_neither_jax_nor_a_solver(capsys):
    """``--rows`` is data, and data must print in an empty environment.

    It is the section a reader has to read before any table — it is where the
    tool says which parameters it chose — so it is the last thing that may
    depend on an install.
    """
    rc = battery.main(["--rows"])
    out = capsys.readouterr().out
    assert rc == 0
    for r in battery.ROWS:
        assert r.name in out
    assert "chose here" in out


def test_the_module_imports_jax_nowhere_at_module_scope():
    """``--rows`` in a jax-free environment only works while this holds.

    Checked in a subprocess, because this session has already imported jax and
    an in-process assertion would measure nothing. Watched to fail: adding
    ``import jax`` at module scope of ``tools/solver_battery.py`` turns this
    red, and turns ``--rows`` in the zero-dep lane into an ImportError.
    """
    probe = (
        "import sys; sys.path.insert(0, %r); import solver_battery; "
        "assert 'jax' not in sys.modules, 'solver_battery imported jax'; "
        "assert len(solver_battery.ROWS) == 10; print('ok')" % str(TOOLS)
    )
    r = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=300,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0 and "ok" in r.stdout, (
        f"tools/solver_battery.py cannot be imported without pulling jax in.\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr[-2000:]}"
    )


# ------------------------------------------------------------------ interface


def test_the_row_selector_refuses_what_it_cannot_run():
    for bad in ("0", "11", "nine"):
        with pytest.raises(SystemExit) as e:
            battery.main(["--only-rows", bad])
        assert "row" in str(e.value)


@pytest.mark.parametrize("argv", [["--repeats", "0"], ["--timeout-ms", "0"]])
def test_the_tool_refuses_a_meaningless_budget(argv):
    with pytest.raises(SystemExit):
        battery.main(argv)


def test_the_help_text_says_it_is_not_a_re_derivation(capsys):
    """The disclaimer is the interface, not decoration.

    A reader who runs ``--help`` and nothing else must not come away thinking
    this reproduces the page's table.
    """
    with pytest.raises(SystemExit):
        battery.main(["--help"])
    # argparse re-wraps the description to the terminal width, so the phrase
    # can land across a line break. Compared on collapsed whitespace, which is
    # what the reader sees rather than what argparse emitted.
    # argparse re-wraps the description AND breaks on hyphens, so
    # "re-derivation" can arrive as "re- derivation". The phrase asserted here
    # is deliberately hyphen-free for that reason.
    out = " ".join(capsys.readouterr().out.split())
    assert "the harnesses behind it were never committed" in out
    assert "--rows" in out and "--variants" in out


def test_the_banner_never_claims_the_page_s_harnesses():
    flat = " ".join(battery._BANNER.split())
    assert "NOT A RE-DERIVATION OF THE PAGE'S TABLE" in flat
    assert "were never committed" in flat


# ----------------------------------------------------------------- rendering


@pytest.mark.parametrize(
    "lo,hi,want",
    [(7, 7, "7 ms"), (7, 9, "7–9 ms"), (1000, 1000, "1.0 s"),
     (8300, 8500, "8.3–8.5 s")],
)
def test_the_millisecond_renderer_is_the_page_s_own_spelling(lo, hi, want):
    """So a reader can put the two tables side by side and read them.

    ``8.3–8.5 s`` and ``71–84 ms`` are the page's own forms; a tool printing
    ``8300-8500ms`` beside them would make the comparison a conversion
    exercise."""
    assert battery._ms(lo, hi) == want


def test_the_tool_ships_in_the_sdist():
    """`/tools` is allowlisted, and a reader who installs the sdist gets it.

    A remedy that only exists in a git checkout is not a remedy for the reader
    the page is written for."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"/tools",' in text
    assert (TOOLS / "solver_battery.py").exists()


def test_the_tool_is_importable_as_a_file_not_a_package():
    """It is a script. If it ever becomes library surface it belongs in
    ``src/stelling`` with the import hygiene that implies."""
    assert not (TOOLS / "__init__.py").exists()
    assert importlib.util.spec_from_file_location(
        "solver_battery", TOOLS / "solver_battery.py") is not None


def test_loading_the_tool_leaves_nothing_behind_in_this_session():
    """The leak :func:`_battery` used to have, asserted rather than described.

    ``tools/`` on ``sys.path`` and ``solver_battery`` in ``sys.modules`` are
    both invisible to ``tests/_state_guard.py``, which lists those two channels
    among the ones it does not watch. So the only thing that can say this
    module cleaned up after itself is this module.
    """
    assert str(TOOLS) not in sys.path, (
        f"{TOOLS} is on sys.path; loading a script must not put it there, and "
        f"nothing else in the session will notice that it did"
    )
    for name in ("solver_battery", _PRIVATE_MODULE_NAME):
        assert name not in sys.modules, (
            f"{name!r} is still registered in sys.modules; the entry needed "
            f"while the file executes must be removed after it, or every "
            f"module collected after this one can import the tool by name"
        )


def test_no_row_label_carries_a_millisecond():
    """A label is what the battery reconstructs from; a timing is not part of
    one. This is the shape of the mistake: a row named for how long it took
    cannot be rebuilt by anybody."""
    for r in battery.ROWS:
        assert not re.search(r"\d+\s*(ms|s)\b", r.name), (
            f"row {r.n}'s label carries a timing: {r.name!r}"
        )
