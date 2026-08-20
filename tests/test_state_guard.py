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
    """The split the two altitudes rest on.

    ``changed`` answers "did this move", which is what
    :data:`_state_guard._DECIDED_HERE` records so the module guard stays quiet
    about it; ``offences`` answers "was this allowed", which is what fails a
    test. If ``changed`` consulted the exemption list, a licensed change would
    be un-licensed one scope out — the module guard would report at teardown
    exactly what the exemption exists to permit.
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


def _nested(tmp_path, body: str) -> subprocess.CompletedProcess:
    """Run one planted, mutating test under the real guard, in a fresh process."""
    plant = tmp_path / "test_planted.py"
    plant.write_text(
        _PLANT.format(body=textwrap.indent(textwrap.dedent(body), "    ").strip()),
        encoding="utf-8",
    )
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


def test_one():
    assert True


def test_two():
    {body}
'''


def _nested_module(tmp_path, setup: str, teardown: str = "pass", body: str = "pass"):
    plant = tmp_path / "test_planted.py"
    plant.write_text(
        _MODULE_PLANT.format(
            setup=textwrap.indent(textwrap.dedent(setup), "    ").strip(),
            teardown=textwrap.indent(textwrap.dedent(teardown), "    ").strip(),
            body=textwrap.indent(textwrap.dedent(body), "    ").strip(),
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TESTS), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "-p", "no:randomly", "-p", "_state_guard", str(plant),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )


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
    assert "outside every test in it" in proc.stdout, (
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
