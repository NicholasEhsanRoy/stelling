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

**AND THE HARNESSES ARE NOT THE PAGE'S HARNESSES.** That is the load-bearing
disclosure of this whole batch and it is gated too
(:func:`test_every_row_says_what_the_page_left_open`): a row of the battery
whose ``chosen`` tuple is empty would be claiming the page's label pinned its
harness, and no label on that page does. Measured, on the row the page's
headline nonlinear finding rests on: three defensible readings of ``32 vars,
16 elementwise products`` disagree with each other about which backend times
out. So the tool prints its numbers beside the page's and never into them,
and this file refuses a row that stops saying which is which.
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


def _battery():
    """Import ``tools/solver_battery.py`` as a module.

    ``tools/`` is not a package and is not on the path — it ships in the sdist
    as scripts, not as importable library surface — so it is put there here
    rather than at ``sys.path`` scope, which would leak into every other
    module in the session.

    **THIS MODULE'S COLLECTION DEPENDS ON THE TOOL BEING STDLIB-ONLY**, which
    is a contract :func:`test_the_module_imports_jax_nowhere_at_module_scope`
    holds in the lanes that have jax. In a lane that does NOT — the zero-dep
    job — breaking that contract errors here, at collection, and takes this
    whole module down rather than failing one test. That is loud and it is the
    safe direction, but a bare ``ModuleNotFoundError: jax`` from a file called
    ``test_solver_battery.py`` reads like a missing test dependency and is not
    one, so the cause is named on the way past."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    try:
        import solver_battery
    except ImportError as e:  # pragma: no cover - only under the mutation
        raise ImportError(
            f"tools/solver_battery.py could not be imported: {e}. That tool is "
            f"stdlib-only BY CONTRACT — `--rows` has to work in an environment "
            f"with no jax and no solver — so an import error here is a defect "
            f"in the tool, not a missing dependency of this test module. See "
            f"test_the_module_imports_jax_nowhere_at_module_scope."
        ) from e

    return solver_battery


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


def _page_table() -> list[tuple[str, ...]]:
    """The ten rows of the page's comparison table, as published.

    Located by its header rather than by line number, and asserted to be
    exactly one table with exactly ten rows — a parser that silently found
    nothing would make every comparison below vacuous."""
    text = PAGE.read_text(encoding="utf-8")
    header = "| obligation | fragment | both | z3 alone | cvc5 alone |"
    assert text.count(header) == 1, (
        f"expected exactly one comparison-table header in {PAGE.name}; "
        f"found {text.count(header)}. The battery's rows are compared against "
        f"this table, so a moved or duplicated header silently empties every "
        f"comparison in this file."
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
        cells = tuple(c.strip() for c in line.strip().strip("|").split("|"))
        # The fragment cell is written `QF_LRA` in markdown and QF_LRA in the
        # tool's plain-text table. Stripped here, in ONE place, so the
        # comparison is about the fragment and not about the page's code-span
        # convention -- and stripped only from that cell, so a change to any
        # other cell's spelling still goes red.
        cells = cells[:1] + (cells[1].strip("`"),) + cells[2:]
        rows.append(cells)
    return rows


def test_the_battery_is_the_page_s_table_row_for_row():
    """The tool and the page cannot drift apart, in either direction.

    Row labels, order, fragment and all three published cells. This is what
    lets ``tools/solver_battery.py`` carry the page's numbers verbatim beside
    its own: if somebody edits a cell on the page, the tool's copy of it goes
    red here rather than quietly becoming a second, disagreeing record.
    """
    published = _page_table()
    assert len(published) == 10, (
        f"the page's comparison table has {len(published)} rows, not ten"
    )
    ours = [
        (r.name, r.fragment, r.page_both, r.page_z3, r.page_cvc5)
        for r in battery.ROWS
    ]
    assert [tuple(p[:5]) for p in published] == ours, (
        "tools/solver_battery.py's ROWS no longer reproduce the page's table.\n"
        + "\n".join(
            f"  row {i + 1}: page {p[:5]!r}\n           tool {o!r}"
            for i, (p, o) in enumerate(zip(published, ours))
            if tuple(p[:5]) != o
        )
    )


def test_every_row_says_what_the_page_left_open():
    """No row may claim its harness was reconstructed.

    ``chosen`` is the list of parameters the page's row LABEL does not fix and
    this battery therefore picked — the declared box, the predicate, the
    association of a monomial, what was timed. A row with an empty ``chosen``
    would be asserting that ten words of English pinned a harness, which is
    the exact mistake that makes an invented harness worse than an
    unreproducible table: it looks reproducible.
    """
    naked = [r.n for r in battery.ROWS if not r.chosen]
    assert not naked, (
        f"rows {naked} declare nothing under `chosen`. Every row label on "
        f"that page leaves at least the declared box open; a row that says "
        f"otherwise is claiming a reconstruction nobody performed."
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
    entry in ``VARIANTS``, which is a harness the tool actually drives.
    """
    contested = {r.n for r in battery.ROWS if r.contested}
    assert contested, "no row is marked contested; rows 7 and 8 were"
    varied = {v.row for v in battery.VARIANTS}
    missing = sorted(contested - varied)
    assert not missing, (
        f"rows {missing} are marked contested and ship no alternate reading. "
        f"A contest with one contestant is a claim, not a measurement."
    )
    stray = sorted(v.row for v in battery.VARIANTS
                   if v.row not in {r.n for r in battery.ROWS})
    assert not stray, f"VARIANTS name rows that do not exist: {stray}"


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
    assert "direction" in text and "milliseconds" in text, (
        "the page has lost its own instruction to read the direction rather "
        "than the milliseconds — which is MORE true now that a tool exists, "
        "not less"
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
    _hide(monkeypatch, "jax")
    rc = battery.main(["--only-rows", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "jax             : NOT INSTALLED" in out
    assert 'pip install "stelling[jax]"' in out, (
        "with no jax nothing can be traced, and the tool must name the extra "
        "rather than the symptom"
    )


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
    assert importlib.util.find_spec("solver_battery") is not None


def test_no_row_label_carries_a_millisecond():
    """A label is what the battery reconstructs from; a timing is not part of
    one. This is the shape of the mistake: a row named for how long it took
    cannot be rebuilt by anybody."""
    for r in battery.ROWS:
        assert not re.search(r"\d+\s*(ms|s)\b", r.name), (
            f"row {r.n}'s label carries a timing: {r.name!r}"
        )
