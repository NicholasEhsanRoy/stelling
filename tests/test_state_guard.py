# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The state guard's own controls: every entry is shown to FIRE.

``tests/_state_guard.py`` is an instrument, and an instrument nobody has seen
move is a decoration. This file moves each of its four entries and asserts the
guard names the test that moved it.

**THE FIRING TESTS RUN A NESTED SESSION, and it is the real module they load.**
A planted test that mutates a global cannot live in this tree — it would
pollute the session it is planted in, which is the whole point of it — so it is
written into ``tmp_path`` and run by a fresh interpreter with ``-p
_state_guard``. That flag loads the SAME file ``tests/conftest.py`` registers;
a copy of the fixture written into the temporary tree would be a test of the
copy. What the nested route does not cover is the one line of wiring in
``conftest.py``, and :func:`test_the_conftest_registers_the_real_fixture`
covers that directly.

**THE ACCEPTANCE DRIVE, recorded because no log-reader can see it happen.**
The bar this guard was built to meet is the incident itself: delete
``adapter.reattach()`` from the ``disarmed`` fixture in
``tests/test_tripwire_arm.py`` — the restore on the disarm side — and run that
file. Measured on jax 0.11.0, at this commit::

    ERROR at teardown of test_the_anchor_removed_gives_no_entry_and_does_not_crash
      jax:const-fold-rule: ('_convert_elt_type_folding_rule', 'c808b3001114', 3, False)
                        -> (None, None, 2, False)
    ... 10 failed, 65 passed, 2 errors

The ten failures are the downstream victims, and they are what the incident
looked like from the outside: ten tests in three sections reporting that arming
does not arm, with nothing anywhere naming the test that removed jax's rule from
the registry. The guard names it, at its own teardown, before any victim runs —
and it names the second test too, the one whose ``armed`` fixture puts the rule
BACK, which is correct rather than noise: each test is measured against the
state it inherited, so the test that changes something back has also changed
something.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import _state_guard as G

TESTS = pathlib.Path(__file__).resolve().parent
REPO = TESTS.parent


# ── the inventory holds itself together ─────────────────────────────────────


def test_every_entry_has_a_name_a_reader_and_a_sentence():
    assert G.ENTRIES, "an empty inventory watches nothing"
    names = [e.name for e in G.ENTRIES]
    assert len(names) == len(set(names)), f"duplicate entry name in {names}"
    for entry in G.ENTRIES:
        assert entry.what.strip(), f"{entry.name} does not say what it watches"


def test_no_reader_is_unreadable_in_this_environment():
    """ANTI-VACUITY, and it is the failure this guard is most exposed to.

    A reader that has started raising returns a constant, and a constant
    fingerprint is a guard that cannot fire — the exact shape the whole batch
    this file belongs to is about. :class:`_state_guard.Unreadable` exists so
    that such a reader is visible rather than silent, and this is what looks.

    It asserts *readability*, not a value: with jax absent three of the four
    read :data:`_state_guard.ABSENT`, which is legitimate and is what the
    zero-dep lane sees all session long.
    """
    state = G.read_state()
    broken = {k: v for k, v in state.items() if isinstance(v, G.Unreadable)}
    assert not broken, (
        f"reader(s) raised: {broken}. A raising reader reports the same value "
        f"before and after every test, so the entry silently stops being "
        f"watched while the inventory still claims it."
    )
    assert set(state) == {e.name for e in G.ENTRIES}


def test_the_conftest_registers_the_real_fixture():
    """The one line the nested sessions below cannot cover.

    ``-p _state_guard`` proves the module works. It does not prove this tree
    is using it — that is the ``state_guard = ...`` line in
    ``tests/conftest.py``, and a suite whose guard is registered nowhere is
    exactly as quiet as one whose guard is broken.
    """
    conftest = sys.modules.get("conftest")
    if conftest is None:  # a session that did not import it under that name
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "conftest_probe", TESTS / "conftest.py"
        )
        conftest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conftest)
    assert getattr(conftest, "state_guard", None) is G.state_guard, (
        "tests/conftest.py does not re-export tests/_state_guard.py's autouse "
        "fixture, so nothing in this tree is guarded"
    )
    assert getattr(conftest, "module_state_guard", None) is G.module_state_guard, (
        "tests/conftest.py does not re-export the MODULE-scoped guard, so a "
        "module-scoped fixture that never restores is unwatched in this tree "
        "— which is the state the whole suite was in until it was measured"
    )


# ── the decision, driven directly ───────────────────────────────────────────


def test_changed_is_blind_to_the_exemption_list_and_offences_is_not():
    """Two questions, kept apart: *did this move* and *was that allowed*.

    ``changed`` answers the first and knows nothing about
    :data:`_state_guard.PINNED_EXEMPTIONS`; ``offences`` answers the second
    and is the one that fails a test. Only the second may consult a licence,
    because a licence is about a nodeid and movement is not.

    The module guard consults NEITHER — it reads where the state is between
    tests (see the comment above ``_state_guard._TRAJECTORY``). An earlier
    version did drive it from ``changed``, as a set of entry names to
    suppress, and that is precisely how the guard's own printed remedy came to
    switch the guard off; the two controls for that are further down this file.
    """
    entry = G.ENTRIES[0].name
    before = {e.name: 0 for e in G.ENTRIES}
    after = {**before, entry: 1}
    original = G.PINNED_EXEMPTIONS
    try:
        G.PINNED_EXEMPTIONS = (G.Exemption("tests/test_x.py::test_y", entry, "driven"),)
        assert [e.name for e in G.changed(before, after)] == [entry]
        assert G.offences("tests/test_x.py::test_y", before, after) == []
    finally:
        G.PINNED_EXEMPTIONS = original


def test_offences_names_exactly_the_entries_that_moved():
    before = {e.name: 0 for e in G.ENTRIES}
    after = dict(before)
    moved = G.ENTRIES[0].name
    after[moved] = 1
    found = G.offences("tests/test_x.py::test_y", before, after)
    assert len(found) == 1 and moved in found[0], found
    assert G.offences("tests/test_x.py::test_y", before, before) == []


def test_an_exemption_licenses_exactly_its_own_test_and_entry():
    """The exemption path, driven without planting an exempt test in the tree.

    Both halves matter: the licence must apply to the test it names, and it
    must NOT apply to a test that merely looks like it. A licence keyed on
    something coarser than the nodeid is a licence for whatever grows next to
    the test that earned it.
    """
    entry = G.ENTRIES[0].name
    exemption = G.Exemption("tests/test_x.py::test_y", entry, "driven")
    before = {e.name: 0 for e in G.ENTRIES}
    after = {**before, entry: 1}

    original = G.PINNED_EXEMPTIONS
    try:
        G.PINNED_EXEMPTIONS = (exemption,)
        assert G.offences("tests/test_x.py::test_y", before, after) == []
        # a parametrisation of the same test inherits it ...
        assert G.offences("tests/test_x.py::test_y[3]", before, after) == []
        # ... and nothing else does
        assert G.offences("tests/test_x.py::test_y_other", before, after)
        assert G.offences("tests/test_z.py::test_y", before, after)
        # nor does it license a DIFFERENT entry on the same test
        other = G.ENTRIES[1].name
        assert G.offences(
            "tests/test_x.py::test_y", before, {**before, other: 1}
        )
    finally:
        G.PINNED_EXEMPTIONS = original


def test_every_pinned_exemption_names_a_real_entry_and_a_real_test():
    """A dangling exemption is a licence nobody is using, pointed at nothing.

    Checked by reading the file rather than by collecting the suite, so it
    holds in a narrowed session too.
    """
    for e in G.PINNED_EXEMPTIONS:
        assert e.entry in G.ENTRY_NAMES, (
            f"{e.nodeid} is exempted for {e.entry!r}, which is not an entry: "
            f"{sorted(G.ENTRY_NAMES)}"
        )
        assert e.why.strip(), f"{e.nodeid} is exempted with no reason given"
        path, _, name = e.nodeid.partition("::")
        target = REPO / path
        assert target.is_file(), f"exemption names a file that is not here: {path}"
        stem = name.split("[", 1)[0]
        assert f"def {stem}(" in target.read_text(encoding="utf-8"), (
            f"exemption names {name!r}, which {path} does not define"
        )


# ── each entry is shown to fire, in a nested session ────────────────────────

_PLANT = '''
import os
import pytest


def test_planted_mutation():
    {body}
'''


def _run_nested(tmp_path, plant) -> subprocess.CompletedProcess:
    """Run one already-written plant under the REAL guard, in a fresh process.

    ``-p _state_guard`` loads the same file ``tests/conftest.py`` registers;
    ``tests/`` goes on ``PYTHONPATH`` so that it is importable under that name.
    Shared by every plant below so that they cannot drift into testing
    different things.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TESTS), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            # never shuffled: see `deterministic_order_args` in conftest.py
            "-p", "no:randomly", "-p", "_state_guard", str(plant),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )


def _nested(tmp_path, body: str) -> subprocess.CompletedProcess:
    """Run one planted, mutating test under the real guard, in a fresh process."""
    plant = tmp_path / "test_planted.py"
    plant.write_text(
        _PLANT.format(body=textwrap.indent(textwrap.dedent(body), "    ").strip()),
        encoding="utf-8",
    )
    return _run_nested(tmp_path, plant)


def _assert_named(proc, entry: str) -> None:
    assert proc.returncode != 0, (
        f"the planted mutation was not reported at all\n{proc.stdout}\n{proc.stderr}"
    )
    assert "test_planted.py::test_planted_mutation changed process-global " in proc.stdout, (
        f"the guard did not name the planted test\n{proc.stdout}\n{proc.stderr}"
    )
    assert entry in proc.stdout, (
        f"the guard fired but did not name the {entry!r} entry\n{proc.stdout}"
    )


def test_the_environment_entry_fires(tmp_path):
    _assert_named(
        _nested(tmp_path, 'os.environ["STELLING_PLANTED_BY_A_TEST"] = "1"'),
        "env:STELLING_*/JAX_*",
    )


def test_the_environment_entry_ignores_keys_outside_the_two_prefixes(tmp_path):
    """The negative half. Without it the entry could be watching `os.environ`
    entire, which would redden every test that sets a temporary variable."""
    proc = _nested(tmp_path, 'os.environ["PLANTED_WITH_NO_PREFIX"] = "1"')
    assert proc.returncode == 0, (
        f"a key outside STELLING_*/JAX_* was reported\n{proc.stdout}"
    )


def test_the_x64_entry_fires(tmp_path):
    pytest.importorskip("jax")
    _assert_named(
        _nested(
            tmp_path,
            """
            import jax
            jax.config.update("jax_enable_x64", not jax.config.jax_enable_x64)
            """,
        ),
        "jax:enable_x64",
    )


def test_the_const_fold_rule_entry_fires(tmp_path):
    """The incident's own entry: jax's rule taken out and not put back."""
    pytest.importorskip("jax")
    _assert_named(
        _nested(
            tmp_path,
            """
            from stelling._tripwire import _adapter_jax as adapter
            assert adapter.detach("entry") == "detached"
            """,
        ),
        "jax:const-fold-rule",
    )


def test_the_tripwire_installation_entry_fires(tmp_path):
    """An installation record left behind, with jax's rule still live.

    `detach("bypass")` puts the ORIGINAL back as the live entry, so the
    const-fold reading is unchanged and only the installation record moves.
    That separation is the point: two entries that always fire together would
    be one entry wearing two names.
    """
    pytest.importorskip("jax")
    proc = _nested(
        tmp_path,
        """
        from stelling._tripwire import _adapter_jax as adapter
        from stelling._tripwire import record
        assert adapter.install(record.Recorder()) == "installed"
        assert adapter.detach("bypass") == "detached"
        """,
    )
    _assert_named(proc, "tripwire:installed")


# ── the module-scoped altitude, driven the same way ─────────────────────────
#
# THE MEASUREMENT THESE THREE EXIST ON, taken before `module_state_guard`
# existed: the restore deleted from the module-scoped `_x64` fixture in
# `tests/test_0_2_0_regression.py`, jax 0.11.0 —
#
#     mutated : 21 passed   [X64PROBE] jax_enable_x64 at session finish = True
#     control : 21 passed   [X64PROBE] jax_enable_x64 at session finish = False
#
# Green either way. The same deletion in a function-scoped fixture is named
# immediately, so it was two near-identical defects with one instrument
# between them. After the guard, the same mutation gives `21 passed, 1 error`
# naming `tests/test_0_2_0_regression.py`.

_MODULE_PLANT = '''
import os
import pytest


@pytest.fixture(autouse=True, scope="module")
def _module_fixture():
    {setup}
    yield
    {teardown}


{one_mark}
def test_one():
    {one_body}


def test_two():
    {body}
'''

#: A ``conftest.py`` for the nested tree that installs one exemption — the
#: remedy the guard's own report prints — using the REAL
#: :data:`_state_guard.PINNED_EXEMPTIONS`, not a copy of it.
_EXEMPTION_CONFTEST = '''
import _state_guard as G

G.PINNED_EXEMPTIONS = (
    G.Exemption({nodeid!r}, {entry!r}, "driven: the report's own remedy"),
)
'''


def _nested_module(
    tmp_path,
    setup: str,
    teardown: str = "pass",
    body: str = "pass",
    one_mark: str = "",
    one_body: str = "assert True",
    exempt: tuple[str, str] | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    plant = tmp_path / "test_planted.py"
    plant.write_text(
        _MODULE_PLANT.format(
            setup=textwrap.indent(textwrap.dedent(setup), "    ").strip(),
            teardown=textwrap.indent(textwrap.dedent(teardown), "    ").strip(),
            body=textwrap.indent(textwrap.dedent(body), "    ").strip(),
            one_mark=one_mark,
            one_body=textwrap.indent(textwrap.dedent(one_body), "    ").strip(),
        ),
        encoding="utf-8",
    )
    if exempt is not None:
        (tmp_path / "conftest.py").write_text(
            _EXEMPTION_CONFTEST.format(nodeid=exempt[0], entry=exempt[1]),
            encoding="utf-8",
        )
    return _run_nested(tmp_path, plant)


def test_a_module_scoped_fixture_that_never_restores_is_named(tmp_path):
    """The blind spot itself. Outside every function-scoped window there is."""
    proc = _nested_module(
        tmp_path, setup='os.environ["STELLING_PLANTED_BY_A_MODULE"] = "1"'
    )
    assert proc.returncode != 0, (
        f"a module-scoped fixture left a watched global changed and nothing "
        f"said so\n{proc.stdout}\n{proc.stderr}"
    )
    assert "test_planted.py changed process-global state" in proc.stdout, (
        f"the guard did not name the MODULE\n{proc.stdout}"
    )
    assert "env:STELLING_*/JAX_*" in proc.stdout, proc.stdout
    assert "moved BETWEEN the tests of this module" in proc.stdout, (
        f"the report does not say where to look\n{proc.stdout}"
    )


def test_a_module_that_sets_a_global_FOR_ITS_OWN_DURATION_is_silent(tmp_path):
    """THE REFUTATION, AS A TEST, and it is why the outer guard can exist.

    The argument against a guard above function scope was that it *"would
    report every module that sets x64 for its own duration"*. pytest sets a
    conftest's module-scoped autouse fixture up BEFORE the test module's own
    and tears it down AFTER, so such a module is bracketed on both sides and
    reads identical. A guard that fired here would be reporting the normal
    case, and would be worth nothing.
    """
    proc = _nested_module(
        tmp_path,
        setup='os.environ["STELLING_PLANTED_FOR_THE_MODULE"] = "1"',
        teardown='del os.environ["STELLING_PLANTED_FOR_THE_MODULE"]',
    )
    assert proc.returncode == 0, (
        f"a module that put its own change back was reported anyway, which "
        f"would make the outer guard fire on the ordinary case\n{proc.stdout}"
    )


def test_a_TEST_that_pollutes_is_named_once_and_not_again_at_module_scope(tmp_path):
    """Two altitudes, one report. Otherwise every function-scoped offence
    would arrive twice and every PINNED_EXEMPTIONS entry would be undone at
    module teardown, since the module guard has no nodeid to license."""
    proc = _nested_module(
        tmp_path, setup="pass", body='os.environ["STELLING_PLANTED_BY_A_TEST"] = "1"'
    )
    assert proc.returncode != 0, proc.stdout
    named = proc.stdout.count("changed process-global state and did not put it back")
    assert named == 1, (
        f"the offence was reported {named} times, not once:\n{proc.stdout}"
    )
    assert "test_planted.py::test_two changed process-global state" in proc.stdout, (
        f"the report names something other than the test that did it\n{proc.stdout}"
    )


# ── the OVER-suppression direction, which had no control at all ─────────────
#
# THE VERSION THESE REFUSE. The module guard used to skip any entry a
# function-scoped guard had DECIDED about inside the module — a set of entry
# NAMES, updated from `changed()` unconditionally and before any report was
# computed. Every test above passed under it. What it bought was that a single
# test could blind the outer guard to a completely different offence, and the
# two cheapest ways to make a test "decide about" an entry are the two here:
# license it, or xfail it. `env:STELLING_*/JAX_*` is ONE entry covering every
# prefixed variable, so one licensed key blinded the module guard to all of
# them.
#
# Measured on the version that had it, and the baseline beside them — the same
# module with the test's leak neither licensed nor absorbed:
#
#     baseline                   2 passed, 1 error       EXIT 1
#       (the TEST named; the module's own leak silently not)
#     + the printed remedy       2 passed                EXIT 0
#     + an xfail instead         1 passed, 2 xfailed     EXIT 0
#
#   and in both green cases a probe at session finish still shows
#   ['STELLING_MODULE_LEAK', 'STELLING_TEST_LEAK'].
#
# Both plants have TWO leaks: one from the module fixture and one from a test.
# Only the test's is licensed or absorbed. The module's is a different act at a
# different altitude and has to survive.


def test_a_module_leak_survives_the_PINNED_EXEMPTION_of_a_test_in_it(tmp_path):
    """The guard's own printed remedy, applied — and the guard stays on.

    `render()` tells a reader whose test tripped the function guard to name
    that test in PINNED_EXEMPTIONS. Doing exactly that used to switch the
    MODULE guard off as well, for every ``STELLING_*``/``JAX_*`` key rather
    than the one exempted, with no output anywhere saying so. An instrument
    whose documented remedy disables it is worse than no instrument, because
    the green afterwards reads as a fix.
    """
    proc = _nested_module(
        tmp_path,
        setup='os.environ["STELLING_MODULE_LEAK"] = "1"',
        body='os.environ["STELLING_TEST_LEAK"] = "1"',
        exempt=("test_planted.py::test_two", "env:STELLING_*/JAX_*"),
    )
    assert proc.returncode != 0, (
        f"exempting the TEST silenced the MODULE's separate leak\n{proc.stdout}"
    )
    assert "test_planted.py changed process-global state" in proc.stdout, (
        f"the module's own leak was not named\n{proc.stdout}"
    )
    assert "STELLING_MODULE_LEAK" in proc.stdout, proc.stdout
    # and the exemption still does its job: the TEST is not named
    assert "test_planted.py::test_two changed process-global" not in proc.stdout, (
        f"the exemption did not license the test it names\n{proc.stdout}"
    )


def test_a_module_leak_survives_an_XFAILING_polluter_in_the_same_module(tmp_path):
    """The same blinding for free, with no exemption list involved.

    An ``xfail`` absorbs the function guard's report, so under the
    name-filtering version the entry was still recorded as decided-about and
    the module's own leak went unnamed — ``EXIT 0``. This needs no
    configuration and no licence: one xfail-marked test that happens to touch
    an environment key was enough to blind the outer guard to every other one.
    """
    proc = _nested_module(
        tmp_path,
        setup='os.environ["STELLING_MODULE_LEAK"] = "1"',
        one_mark='@pytest.mark.xfail(reason="driven: absorbs the guard\'s report")',
        one_body=(
            'os.environ["STELLING_TEST_LEAK"] = "1"\n'
            "assert False"
        ),
    )
    assert proc.returncode != 0, (
        f"an xfailing polluter silenced the MODULE's separate leak\n{proc.stdout}"
    )
    assert "test_planted.py changed process-global state" in proc.stdout, (
        f"the module's own leak was not named\n{proc.stdout}"
    )
    assert "STELLING_MODULE_LEAK" in proc.stdout, proc.stdout
    # AND THE LIMIT THIS PLANT ALSO CARRIES, PINNED HERE BECAUSE IT IS HERE.
    # The xfail absorbs the FUNCTION guard's report, and a test's own window is
    # outside the module trajectory by construction, so the xfailing test's own
    # leak is named at NEITHER altitude. That is the one exception to "a test's
    # offence is named at the altitude that can name a test"; it is in LIMITS
    # in tests/_state_guard.py, and this is what holds the LIMITS entry to
    # being a measurement. If this ever goes red the entry comes out.
    assert "STELLING_TEST_LEAK" not in proc.stdout, (
        f"the xfailing test's own leak IS named somewhere now, so the LIMITS "
        f"entry in tests/_state_guard.py is stale\n{proc.stdout}"
    )


def test_the_module_guard_cannot_see_an_IMPORT_TIME_statement_and_says_so(tmp_path):
    """A limit, pinned as a limit. Both halves are the point.

    pytest imports test modules during COLLECTION, before any fixture of any
    scope is set up, so a column-0 ``os.environ[...] = ...`` in a test module
    is already in force when the module guard reads ``before`` — it reads
    identical either side and the session ends polluted at ``EXIT 0``. That is
    not fixable at this altitude; it is the session guard's, and there is no
    session guard because there is nothing at session scope to watch yet.

    What WAS fixable is the report: it used to offer *"a module-level
    statement with no matching restore"* as something to go and look for,
    which is a cause this guard is incapable of having observed. A remedy that
    names a cause the instrument cannot see sends the reader to look for the
    wrong thing, and it makes the LIMITS list look shorter than it is.
    """
    plant = tmp_path / "test_planted.py"
    plant.write_text(
        'import os\n\nos.environ["STELLING_IMPORT_TIME_LEAK"] = "1"\n\n\n'
        "def test_one():\n    assert True\n\n\n"
        "def test_two():\n    assert True\n",
        encoding="utf-8",
    )
    proc = _run_nested(tmp_path, plant)
    assert proc.returncode == 0, (
        f"the import-time case is reported now, so this control is stale and "
        f"the LIMITS list needs the entry removed\n{proc.stdout}"
    )
    # the other half: the remedy must not send the reader after it
    reported = _nested_module(
        tmp_path / "reported",
        setup='os.environ["STELLING_PLANTED_BY_A_MODULE"] = "1"',
    )
    assert "or a module-level statement with no matching restore" not in reported.stdout, (
        f"the module guard's remedy still OFFERS a cause it cannot observe\n"
        f"{reported.stdout}"
    )
    assert "NOT a module-level statement at import" in reported.stdout, (
        f"the remedy no longer rules out the cause a reader would reach for "
        f"first, so the reader goes looking for it\n{reported.stdout}"
    )


# ── a SECOND SESSION IN THE SAME PROCESS, which is where this last broke ────
#
# THE REGRESSION THESE TWO REFUSE, and it was a strict one. The trajectory
# lived in three MODULE-LEVEL dicts and `module_state_guard` cleared all three
# at its teardown — so an inner session that loaded the same module cleared the
# OUTER session's bookkeeping halfway through the outer module. Driven at the
# commit before this one, plant = a module fixture leaking one key and
# `test_one` running `pytest.main([..., "-p", "_state_guard", inner])`:
#
#     module-level dicts, leaky module    EXIT 0   silent
#       ... and at session finish:        ['STELLING_LEAK_MODULE']
#     the SAME plant, no nested session   EXIT 1   names the module
#     the entry-NAME version it replaced  EXIT 1   names the module
#
# so the mechanism this file exists to argue for failed OPEN exactly where the
# one it replaced failed SAFE. The other direction reads worse: a WELL-BEHAVED
# module was reported for moving `()` to `()`, because `module_before.get(name)`
# had become `None` under it — and a report a maintainer cannot reproduce is
# how a guard gets weakened.
#
# Latent when it was found, and counted rather than assumed:
# `tests/test_tripwire_plugin.py` reaches `pytester.runpytest` — IN-PROCESS —
# through the `_run` helper at its line 83, from FOURTEEN call sites, and not
# one of them passes `-p _state_guard` (its args are
# `tripwire_plugin_args()` and `deterministic_order_args()`). The nested
# sessions in THIS file do pass it and are `subprocess.run`. What made it worth fixing rather than
# recording is that the reaching edit is the idiom this very file already uses,
# one argument away. The trajectory hangs off the session's `Config` now, so a
# nested session gets a fresh one and the separation is BY CONSTRUCTION.

#: ``test_one``'s body for the two controls below: an in-process pytest session
#: over a trivial, well-behaved tree, loading the SAME ``_state_guard`` module
#: object the outer session is guarded by. ``pytest.main`` rather than a
#: subprocess because a second PROCESS shares nothing and is not the shape at
#: issue, and rather than ``pytester`` because that is a plugin the plant would
#: have to request — this is the spelling that reaches the defect with the
#: least machinery between it and the guard.
_RUNS_A_NESTED_SESSION = """
import pathlib
import tempfile

inner = pathlib.Path(tempfile.mkdtemp()) / "test_inner.py"
inner.write_text("def test_inner():\\n    assert True\\n", encoding="utf-8")
assert pytest.main([
    "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
    "-p", "_state_guard", str(inner),
]) == 0
"""


def test_a_module_leak_is_still_named_when_a_TEST_RUNS_A_NESTED_SESSION(tmp_path):
    """The failing-open half: the leak is real and must still be named."""
    proc = _nested_module(
        tmp_path,
        setup='os.environ["STELLING_LEAK_MODULE"] = "1"',
        one_body=_RUNS_A_NESTED_SESSION,
    )
    assert proc.returncode != 0, (
        f"a module-scoped fixture leaked and the guard went silent because a "
        f"test in the module happened to run a nested pytest session\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "test_planted.py changed process-global state" in proc.stdout, (
        f"the guard did not name the MODULE\n{proc.stdout}"
    )
    assert "STELLING_LEAK_MODULE" in proc.stdout, proc.stdout


def test_a_WELL_BEHAVED_module_stays_silent_when_a_test_runs_a_nested_session(
    tmp_path,
):
    """The failing-noisy half, and it is the one that gets a guard weakened.

    Under the module-level dicts this module — which puts its own change back
    at teardown, the ordinary case — was reported for moving ``()`` to ``()``.
    A maintainer who meets that report and cannot reproduce it turns the guard
    off, so the silent direction is a control in its own right.
    """
    proc = _nested_module(
        tmp_path,
        setup='os.environ["STELLING_WELL_BEHAVED_MODULE"] = "1"',
        teardown='del os.environ["STELLING_WELL_BEHAVED_MODULE"]',
        one_body=_RUNS_A_NESTED_SESSION,
    )
    assert proc.returncode == 0, (
        f"a module that put its own change back was reported anyway, because "
        f"a test in it ran a nested pytest session\n{proc.stdout}"
    )


def test_the_module_report_does_not_claim_a_restore_a_TEST_performed_never_happened(
    tmp_path,
):
    """The module altitude's WORDING, held to what the module altitude measures.

    The shape: the module fixture sets ``K`` at setup, a TEST deletes ``K``,
    and the fixture's teardown restore is therefore a no-op. The session ends
    CLEAN — and the module is still reported, correctly, because its own move
    out was never undone by the module and the clean-up belongs to a test that
    can be skipped, deleted or reordered away.

    What was wrong was the sentence: *"changed process-global state and did
    not put it back"* is what the FUNCTION guard measures, and here it is
    false — it was put back. A report whose first line a reader can disprove
    by looking at the process is a report that gets the guard weakened rather
    than the fixture fixed, which is the same failure mode as the ``() -> ()``
    one above.
    """
    proc = _nested_module(
        tmp_path,
        setup='os.environ["STELLING_MODULE_SETS_IT"] = "1"',
        teardown='os.environ.pop("STELLING_MODULE_SETS_IT", None)',
        one_body='del os.environ["STELLING_MODULE_SETS_IT"]',
    )
    assert proc.returncode != 0, (
        f"the module moved a watched global outside every test and nothing "
        f"put it back THERE; that is the report this altitude exists for\n"
        f"{proc.stdout}"
    )
    assert "test_planted.py changed process-global state BETWEEN its tests" in proc.stdout, (
        f"the module was not named, or not in the terms it measured\n{proc.stdout}"
    )
    assert "STELLING_MODULE_SETS_IT" in proc.stdout, proc.stdout
    # the wording that is false HERE. `test_planted.py::test_one` legitimately
    # carries it — that IS a before/after reading of one test — so the check is
    # on the module's own line, which has no `::`.
    assert "test_planted.py changed process-global state and did not put it back" not in proc.stdout, (
        f"the MODULE is told it did not put something back that a test in it "
        f"did put back, and the process is clean at session finish\n{proc.stdout}"
    )
    assert "A TEST that happens to put it back leaves the process clean" in proc.stdout, (
        f"the module report no longer explains why it fires on a process that "
        f"looks clean, which is the half a reader needs\n{proc.stdout}"
    )


def test_a_test_that_restores_what_it_changed_is_silent(tmp_path):
    """The other direction, and the guard is worthless without it.

    A fixture that fired on everything would be indistinguishable from one
    that fired on nothing, since either way its verdict carries no information
    about the test.
    """
    pytest.importorskip("jax")
    proc = _nested(
        tmp_path,
        """
        import jax
        from stelling._tripwire import _adapter_jax as adapter
        from stelling._tripwire import record
        old = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", not old)
        assert adapter.install(record.Recorder()) == "installed"
        assert adapter.detach("bypass") == "detached"
        assert adapter.reattach() == "reattached"
        assert adapter.restore() == "restored"
        jax.config.update("jax_enable_x64", old)
        os.environ["STELLING_PLANTED_AND_REMOVED"] = "1"
        del os.environ["STELLING_PLANTED_AND_REMOVED"]
        """,
    )
    assert proc.returncode == 0, (
        f"a test that put everything back was reported anyway\n{proc.stdout}"
    )
