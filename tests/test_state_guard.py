# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The state guard's own controls: every entry is shown to FIRE.

``tests/_state_guard.py`` is an instrument, and an instrument nobody has seen
move is a decoration. This file moves each of its five entries and asserts the
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

import ast
import os
import pathlib
import re
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

    It asserts *readability*, not a value: with jax absent four of the five
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
        # an entry naming NO parametrisation licenses the whole family, which
        # is what its author wrote when they left the suffix off ...
        assert G.offences("tests/test_x.py::test_y[3]", before, after) == []
        # ... and nothing else does
        assert G.offences("tests/test_x.py::test_y_other", before, after)
        assert G.offences("tests/test_z.py::test_y", before, after)
        # nor does it license a DIFFERENT entry on the same test
        other = G.ENTRIES[1].name
        assert G.offences(
            "tests/test_x.py::test_y", before, {**before, other: 1}
        )
        # AND THE DIRECTION THE COMMENT ABOVE `PINNED_EXEMPTIONS` PROMISED AND
        # THE CODE INVERTED. An entry that DOES name a parametrisation covers
        # that case and no other: driven at `844ba48`, an exemption for
        # `test_x[a]` licensed `test_x[b]` — `EXIT 0`, both keys leaked —
        # under a comment saying the stripping existed so that *"a new
        # parameter does not silently inherit somebody else's licence"*. A
        # parameter added next to the one that earned the licence is a case
        # nobody reviewed, which is what the list is for.
        G.PINNED_EXEMPTIONS = (
            G.Exemption("tests/test_x.py::test_y[a]", entry, "driven"),
        )
        assert G.offences("tests/test_x.py::test_y[a]", before, after) == []
        assert G.offences("tests/test_x.py::test_y[b]", before, after), (
            "an exemption written for one parametrisation licenses its "
            "neighbours, so a new parameter inherits a licence nobody wrote "
            "for it"
        )
        assert G.offences("tests/test_x.py::test_y", before, after), (
            "an exemption for one parametrisation licenses the unparametrised "
            "nodeid too"
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


def test_the_perimeter_entry_fires(tmp_path):
    """39 dunder slots rebound on two foreign C-extension types and left there.

    This entry watches the state the 0.2.0 dunder perimeter installs, and it
    was added after that batch shipped rather than with it: for the whole of
    that batch, a test that armed the perimeter and did not release it moved
    the most order-sensitive state in the tree past an inventory that had no
    row for it. `ci.yml`'s `random-order` lane reads a shuffled failure the
    guard did not name as "state outside that inventory", so the omission did
    not just fail to report -- it made a true annotation say something false.
    """
    pytest.importorskip("jax")
    _assert_named(
        _nested(
            tmp_path,
            """
            from stelling._tripwire import perimeter
            status = perimeter.arm(("tracer",), owner="a test that never releases")
            assert status.armed, status.explanation
            """,
        ),
        "perimeter:installed",
    )


def test_the_perimeter_entry_is_silent_when_the_hold_is_RELEASED(tmp_path):
    """The negative half, and it is not decoration.

    The wrapper is a fresh closure on every `arm()`, so an entry that read the
    wrapper's identity would fire on any test that armed and disarmed --
    which is most of `tests/test_narrowing_perimeter.py` -- and an entry that
    fires on well-behaved tests is one that gets suppressed rather than read.
    """
    pytest.importorskip("jax")
    proc = _nested(
        tmp_path,
        """
        from stelling._tripwire import perimeter
        status = perimeter.arm(("tracer",), owner="A")
        assert status.armed, status.explanation
        assert perimeter.disarm("A") == "restored"
        status = perimeter.arm(("tracer",), owner="A")
        assert status.armed, status.explanation
        assert perimeter.disarm("A") == "restored"
        """,
    )
    assert proc.returncode == 0, (
        f"arming and releasing twice was reported as a leak\n{proc.stdout}"
    )


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
    assert "moved OUTSIDE every test of this module" in proc.stdout, (
        f"the report does not say where to look\n{proc.stdout}"
    )
    # THE HEADLINE SAYS "OUTSIDE" AND NOT "BETWEEN", and this plant is why:
    # the fixture leaks at SETUP, so its move is BEFORE the first test rather
    # than between two of them, and it is the commonest shape this altitude
    # reports.
    assert "changed process-global state OUTSIDE its tests" in proc.stdout, (
        f"the module headline names a narrower place than the one the "
        f"trajectory reads, and this leak is not in it\n{proc.stdout}"
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
    # COUNT THE REPORTS, NOT THE ECHOES. pytest prints each error twice: once
    # in the ERRORS block and once in `short test summary info`, where the
    # reason is truncated to the TERMINAL WIDTH. Counting the whole of stdout
    # therefore made this test pass at COLUMNS=80 and fail at 200 -- green on
    # the machine that wrote it, red in CI, for a reason that has nothing to
    # do with the property. The summary is an echo of the same report.
    body = proc.stdout.split("short test summary info", 1)[0]
    named = body.count("changed process-global state and did not put it back")
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


# ── the module's move, folded PART-WISE and not whole ───────────────────────
#
# WHAT THESE FOUR REFUSE, and it is the one shape in which the trajectory
# reported a test at the module's altitude. `_outside_a_test` used to write the
# WHOLE current reading into `shadow` whenever any part of it had moved outside
# a test — and an entry's reading covers many independently-movable parts
# (`env:STELLING_*/JAX_*` is a prefix rule, not a key). So a module that moved
# ANY prefixed variable at teardown re-folded the reading it found, a test's
# leak included, and the module's report then named a key no module had
# touched under a headline saying *"so no test of this module did it"*.
#
# Measured at `844ba48`, plant = a test leaking `STELLING_TEST_LEAK` and a
# module fixture setting `STELLING_MODULE_LEAK` at TEARDOWN, so that the
# module's move comes after the test's:
#
#     the module's report   () -> (('STELLING_MODULE_LEAK', '1'),
#                                  ('STELLING_TEST_LEAK', '1'))
#     the test's report     () -> (('STELLING_TEST_LEAK', '1'),)
#
# Three stated properties broke together, each driven and each one of these
# tests: "named exactly once" (the leak arrives twice), "keeps an exemption
# from being undone one scope out" (an exempted test's leak is still named at
# module scope), and the LIMITS entry's "named NOWHERE" for an xfailing
# polluter (named at module scope after all, whenever the module moves the same
# entry). Every direction was a strict OVER-report, so nothing was hidden;
# PINNED_EXEMPTIONS is empty, so nothing was licensed either. What was false
# was the docstring.


def test_a_module_move_AFTER_a_TESTS_leak_names_only_the_MODULES_leak(tmp_path):
    """The double report, and it is the property the two altitudes rest on.

    The module guard is supposed to be blind to what a test did, BY
    CONSTRUCTION rather than by suppression. It is blind to the test's own
    window; it was not blind to the test's CONTRIBUTION to a reading it folded
    for its own reasons.
    """
    proc = _nested_module(
        tmp_path,
        setup="pass",
        teardown='os.environ["STELLING_MODULE_LEAK"] = "1"',
        body='os.environ["STELLING_TEST_LEAK"] = "1"',
    )
    assert proc.returncode != 0, (
        f"neither altitude reported anything, so this plant has stopped "
        f"reaching the mechanism\n{proc.stdout}"
    )
    # COUNT THE REPORTS, NOT THE ECHOES -- see the note in
    # `test_a_TEST_that_pollutes_is_named_once_and_not_again_at_module_scope`.
    body = proc.stdout.split("short test summary info", 1)[0]
    module_report = body.split("OUTSIDE its tests", 1)
    assert len(module_report) == 2, f"the module was not named\n{proc.stdout}"
    module_report = module_report[1].split("Failed:", 1)[0]
    assert "STELLING_MODULE_LEAK" in module_report, proc.stdout
    assert "STELLING_TEST_LEAK" not in module_report, (
        f"the MODULE's report names a key a TEST set, under a headline that "
        f"says no test of this module did it. The offence is then reported "
        f"twice and an exemption for the test does not reach the second "
        f"report:\n{module_report}"
    )
    assert body.count("STELLING_TEST_LEAK") == 1, (
        f"the test's leak is named {body.count('STELLING_TEST_LEAK')} times, "
        f"not once:\n{proc.stdout}"
    )


def test_a_module_that_puts_ITS_OWN_key_back_is_silent_even_after_a_TEST_leak(
    tmp_path,
):
    """The silent direction, which is the one a whole-reading fold got wrong.

    The module sets `K` at setup and removes it at teardown — the ordinary
    case, already covered by
    `test_a_module_that_sets_a_global_FOR_ITS_OWN_DURATION_is_silent`. Here a
    test leaks a DIFFERENT key in between, which is what made the teardown
    reading differ from the last window's close and pulled the whole reading
    into `shadow`. The module still put its own change back, so the module
    altitude must still be silent about it.
    """
    proc = _nested_module(
        tmp_path,
        setup='os.environ["STELLING_MODULE_OWN"] = "1"',
        teardown='del os.environ["STELLING_MODULE_OWN"]',
        body='os.environ["STELLING_TEST_LEAK"] = "1"',
    )
    body = proc.stdout.split("short test summary info", 1)[0]
    assert "OUTSIDE its tests" not in body, (
        f"a module that put its own key back was reported at module scope "
        f"because a test in it leaked a different one\n{proc.stdout}"
    )
    # ... and the test's own leak is still named, so this is not silence
    # bought by turning the instrument off.
    assert "test_planted.py::test_two changed process-global state" in body, (
        f"the TEST's leak went unreported too, so this control is vacuous\n"
        f"{proc.stdout}"
    )


def test_an_EXEMPTED_tests_leak_is_not_reported_when_the_MODULE_also_moves(
    tmp_path,
):
    """The exemption property, in the shape that broke it.

    `test_a_module_leak_survives_the_PINNED_EXEMPTION_of_a_test_in_it` plants
    the module's leak at SETUP, before the test runs, so the module's only
    fold happens before the test's leak exists. Move the module's own act to
    TEARDOWN and the exempted key came back at an altitude with no exemption
    list — the licence undone one scope out, which is exactly what the
    trajectory replaced the entry-NAME filter to prevent.
    """
    proc = _nested_module(
        tmp_path,
        setup="pass",
        teardown='os.environ["STELLING_MODULE_LEAK"] = "1"',
        body='os.environ["STELLING_TEST_LEAK"] = "1"',
        exempt=("test_planted.py::test_two", "env:STELLING_*/JAX_*"),
    )
    assert proc.returncode != 0, (
        f"the module's own leak was not reported at all\n{proc.stdout}"
    )
    assert "STELLING_MODULE_LEAK" in proc.stdout, proc.stdout
    assert "STELLING_TEST_LEAK" not in proc.stdout, (
        f"an exempted TEST's key is named at MODULE scope, where "
        f"PINNED_EXEMPTIONS cannot reach it\n{proc.stdout}"
    )


def test_an_XFAILING_polluters_leak_is_named_NOWHERE_even_if_the_module_moves(
    tmp_path,
):
    """The LIMITS entry, held to being unconditional.

    `tests/_state_guard.py` states that an xfail-marked test's own leak is
    named at NEITHER altitude. That was conditional: it was named at module
    scope whenever the module moved the same entry outside a test. The
    sibling control
    `test_a_module_leak_survives_an_XFAILING_polluter_in_the_same_module`
    misses it for the same reason as the exemption one — its module acts at
    setup.

    If this ever goes red the LIMITS entry comes out; a limit that is wrong in
    the SAFE direction is still wrong.
    """
    proc = _nested_module(
        tmp_path,
        setup="pass",
        teardown='os.environ["STELLING_MODULE_LEAK"] = "1"',
        one_mark='@pytest.mark.xfail(reason="driven: absorbs the guard\'s report")',
        one_body=(
            'os.environ["STELLING_TEST_LEAK"] = "1"\n'
            "assert False"
        ),
    )
    assert proc.returncode != 0, (
        f"the module's own leak was not reported\n{proc.stdout}"
    )
    assert "STELLING_MODULE_LEAK" in proc.stdout, proc.stdout
    assert "STELLING_TEST_LEAK" not in proc.stdout, (
        f"the xfailing test's leak IS named somewhere now, so the LIMITS "
        f"entry in tests/_state_guard.py is stale\n{proc.stdout}"
    )


def test_the_carry_is_part_wise_for_a_mapping_and_whole_for_anything_else():
    """:func:`_state_guard._carry` directly, so the property is not only
    driven through three subprocesses.

    The four rows are the whole of it: a part the move did not touch keeps
    whoever put it there, a part the move set or removed is written through,
    and a reading that is not a mapping has one part and carries whole —
    which is what keeps `ABSENT` and `Unreadable` values like any other.
    """
    # the module added M; the test's T is untouched by that move and stays out
    assert G._carry({}, {"T": "1"}, {"T": "1", "M": "1"}) == {"M": "1"}
    # the module removed its own K; the test's T is untouched and stays out
    assert G._carry({"K": "1"}, {"K": "1", "T": "1"}, {"T": "1"}) == {}
    # a part the move overwrote is written through, whoever had it before
    assert G._carry({"K": "1"}, {"K": "1"}, {"K": "2"}) == {"K": "2"}
    # not a mapping: one opaque part, carried whole
    assert G._carry(G.ABSENT, G.ABSENT, False) is False
    assert G._carry({"K": "1"}, G.ABSENT, {"K": "2"}) == {"K": "2"}


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
# Latent when it was found, and now HELD rather than counted.
# `tests/test_tripwire_plugin.py` reaches `pytester.runpytest` — IN-PROCESS —
# through its `_run` helper, and no call site there hands a nested session
# `-p _state_guard`. The nested sessions in THIS file do pass it and are
# `subprocess.run`. What made it worth fixing rather than recording is that
# the reaching edit is the idiom this very file already uses, one argument
# away. The trajectory hangs off the session's `Config` now, so a nested
# session gets a fresh one and the separation is BY CONSTRUCTION.
#
# **THIS PARAGRAPH USED TO CARRY THE HELPER'S LINE NUMBER AND ITS CALL-SITE
# COUNT, AND BOTH WERE FALSE.** It said line 83 from FOURTEEN sites; measured
# at the commit that removed them, the helper was at line 98 with 24 sites,
# and a later repair moved it again. Neither figure was ever held by
# anything, which is why neither survived contact with a tree that moved —
# and a count a reader cannot re-derive is the same defect as a check that
# does not exist. The claim that carried the ARGUMENT was never the count; it
# was the absence, so the absence is what
# `test_no_nested_session_in_the_tripwire_file_loads_this_module` asserts and
# the numbers are gone rather than corrected.


#: The spellings that put this module's name into a nested session's
#: arguments. `-p` takes its value joined as well as separated, so the joined
#: forms are the same instruction written shorter — argparse accepts all three
#: and a scan that reads only the separated one is a scan of a style, not of a
#: behaviour.
_LOADS_THIS_MODULE = ("_state_guard", "-p_state_guard", "-p=_state_guard")

#: The `pytester` methods whose STRING ARGUMENT BECOMES A FILE the nested
#: session then parses — the one place an EMBEDDED occurrence of this module's
#: name is a load rather than a mention, because there the text stops being
#: prose and becomes source. `makeconftest('pytest_plugins = ["_state_guard"]')`
#: registers the plugin as surely as `-p` does, and that spelling is live in
#: the scanned file, every site writing the name through a constant — which is
#: why the file is clean, and for that reason rather than by luck.
#:
#: **`makefile` IS IN THIS SET BECAUSE IT IS THE PRIMITIVE THE OTHERS DELEGATE
#: TO**, and leaving it out was this guard's third round of one mistake:
#: enumerating spellings of the thing instead of the thing. The first version
#: read only direct `Call` arguments and missed `*ARGS`; the second matched
#: only whole strings and missed `makeconftest`; the third named four wrappers
#: and missed the base method all four call. Each gap was a spelling live in
#: the scanned file. A set of names is the wrong shape for "writes a file",
#: and this one is defensible only because `makefile` closes it from below.
_GENERATES_SOURCE = frozenset(
    {"makefile", "makeconftest", "makepyfile", "makeini", "makepyprojecttoml"}
)

#: The name as a WHOLE TOKEN, for generated text that is not Python. The
#: lookbehind excludes `.` as well as word characters so `tests._state_guard`
#: — a different module path — does not match, and the lookahead keeps
#: `_state_guardian` and `_state_guard_` out.
_EMBEDDED = re.compile(r"(?<![\w.])_state_guard(?![\w])")

#: How far to follow generated source into generated source. Bounded because
#: the recursion below is over attacker-shaped input in principle, and because
#: nothing in this tree nests a `make*` inside a `make*` even once.
_MAX_GENERATED_DEPTH = 3


def state_guard_references(
    source: str, _depth: int = 0
) -> list[tuple[int, str]]:
    """`(line, spelling)` for every literal in `source` that LOADS this module.

    **ONE RULE, APPLIED TWICE, BECAUSE A STRING MEANS DIFFERENT THINGS
    DEPENDING ON WHO PARSES IT.** Getting that wrong is what made three
    earlier versions of this scan miss three spellings the scanned file
    actually uses.

    1. **In THIS source.** A literal that IS one of
       :data:`_LOADS_THIS_MODULE`, wherever it sits — a call argument, a
       module-level tuple later spread as ``*ARGS``, a keyword's tuple, an
       inline list, a dict, a comprehension. Whole-string, because in an
       argument list the name is the whole argument. Prose is excluded by what
       it IS rather than where it is: a docstring and a bare string statement
       are the two ways Python holds text that is not a value, and comments
       never reach the AST at all.
    2. **In source this file GENERATES**, i.e. a literal anywhere beneath a
       call to one of :data:`_GENERATES_SOURCE`. If that literal parses as
       Python, **this function is applied to it** — so a generated conftest
       gets rule 1, and a comment or docstring written *into* a generated file
       is prose there for exactly the reason it is prose here. If it does not
       parse — an ini, a TOML fragment — the whole-token
       :data:`_EMBEDDED` match stands in.

    Recursing rather than pattern-matching the generated text is the point:
    it is the same rule, and a guard that fires on a comment it generated
    would be a guard that fires on its own explanation one level down. That
    direction is the one this suite says gets a check deleted.

    **The conservative edge, named.** Rule 2 catches ``makeini`` writing
    ``addopts = -p _state_guard``, which does NOT register the plugin today.
    That is pytest's current behaviour rather than a guarantee, and the
    failure mode here is silent state-sharing, so an attempt to load is worth
    naming even where the attempt would not currently work.

    **THIS SCAN IS A TRIPWIRE, NOT A PROOF, AND FIVE ROUNDS OF AUDIT ARE THE
    EVIDENCE FOR THAT SENTENCE.** Each round closed one syntactic form and the
    next found another, and the miss was always **one indirection** further
    out: direct call arguments missed a splat; whole-string matching missed a
    generated conftest; four wrapper names missed the base method they all
    delegate to; and naming the base method still misses a source constant
    reached through a variable. A syntactic single-file scan cannot close this
    class — there is always one more indirection — so the honest thing is to
    say what it does not reach and to say where the guarantee actually lives.

    **WHERE THE GUARANTEE ACTUALLY LIVES**, and it is not here: the trajectory
    hangs off the session's ``Config`` rather than off module-level dicts
    (``tests/_state_guard.py``, the comment above ``_TRAJECTORY``). A nested
    session builds a fresh ``Config``, **so the separation holds BY
    CONSTRUCTION whether or not this module is loaded there.** That is the
    braces. This scan is the belt: it catches a reaching edit at review time,
    when the cost of the conversation is low, and nothing rests on it being
    complete.

    **WHAT IT DOES NOT REACH**, declared rather than discovered — the previous
    four docstrings each claimed a completeness the code did not have, which is
    the defect this list exists to stop repeating:

    * a literal defined in **another module** and imported in — this is a
      single-file scan, and the test below says so in its own title;
    * a **direct write** into the nested tree
      (``(pytester.path / "conftest.py").write_text(...)``), which loads the
      plugin and goes through no ``make*`` call;
    * a **local helper** wrapping one of those methods, one indirection past a
      syntactic scan — worth naming because helper-wrapping is the scanned
      file's prevailing idiom, which is how several of these gaps arrived;
    * a **module-level source constant** handed to a ``make*`` call
      (``SG = 'pytest_plugins = ["_state_guard"]'``; ``makeconftest(SG)``).
      Measured to load the plugin. This is the same idiom the scanned file
      already uses for generated source — ``WRAPPING_TEST`` is defined once
      and handed to ``makepyfile`` at a dozen sites — so it is the likeliest
      of these four to be written by accident;
    * a **generated conftest that imports and registers**
      (``import _state_guard; config.pluginmanager.register(_state_guard)``).
      Measured to load the plugin. Rule 1 matches string literals, and here
      the name is an ``import`` alias and a ``Name``, never a literal. A bare
      ``import`` with no ``register`` does not load, and is correctly silent.

    **THE REPAIR THAT WOULD ACTUALLY CLOSE IT** is not a sixth form. It is to
    stop modelling the behaviour and measure it — assert on the nested
    session's plugin manager rather than on the spelling that reaches it,
    which is decidable, immune to every indirection above, and the same
    measure-don't-model cut this project applies everywhere else. That is
    recorded as work rather than done here, because it changes this from a
    static check into one that runs nested sessions, and the property it
    guards already holds by construction.
    """
    tree = ast.parse(source)
    prose: set[int] = set()
    for node in ast.walk(tree):
        # A bare string statement is text, not a value — and that covers every
        # docstring, which is only ever the first such statement in a body.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                prose.add(id(node.value))

    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in _LOADS_THIS_MODULE
            and id(node) not in prose
        ):
            found.add((node.lineno, node.value))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _GENERATES_SOURCE
        ):
            for inner in ast.walk(node):
                if not (
                    isinstance(inner, ast.Constant)
                    and isinstance(inner.value, str)
                ):
                    continue
                if _generated_text_loads(inner.value, _depth):
                    # The OUTER line, because that is the line a reader edits.
                    found.add((inner.lineno, " ".join(inner.value.split())))
    return sorted(found)


def _generated_text_loads(text: str, depth: int) -> bool:
    """Whether generated `text` would load this module in the nested session.

    Python gets the Python rule, recursively. Anything else gets the
    whole-token match, because there is no syntax to tell a value from a
    remark in an ini file.
    """
    if _EMBEDDED.search(text) is None:
        return False
    if depth >= _MAX_GENERATED_DEPTH:
        return True
    try:
        return bool(state_guard_references(text, depth + 1))
    except SyntaxError:
        return True


def test_no_nested_session_in_the_tripwire_file_loads_this_module():
    """No literal in `tests/test_tripwire_plugin.py` names this module.

    THE HALF OF THE PARAGRAPH ABOVE THAT IS A CLAIM RATHER THAN A STORY. That
    file's `_run` helper reaches `pytester.runpytest` IN-PROCESS, so a nested
    session there that loaded this module would share the outer session's
    bookkeeping — the exact shape the trajectory fix exists to make
    impossible. The fix makes it impossible BY CONSTRUCTION and this makes the
    argument re-derivable, which the line number and call count it replaced
    never were.

    The scan is :func:`state_guard_references`, and
    :func:`test_the_reference_scan_is_driven_both_ways` drives it — because a
    guard nobody can drive is the thing this suite refuses everywhere else.
    """
    path = pathlib.Path(__file__).resolve().parent / "test_tripwire_plugin.py"
    found = state_guard_references(path.read_text(encoding="utf-8"))
    assert not found, (
        "tests/test_tripwire_plugin.py names `_state_guard` at "
        f"{[line for line, _ in found]}. Its sessions reach "
        "`pytester.runpytest` IN-PROCESS, so loading this module there puts a "
        "nested session and the outer one on the same bookkeeping — which is "
        "the defect the trajectory-on-Config fix closed. If a nested session "
        "there genuinely needs the guard, it needs a SUBPROCESS, the way this "
        "file's own nested sessions get it."
    )


#: Reaching spellings and prose spellings, as SOURCE. Each reaching entry was
#: silent against the call-argument scan this replaced, except the first, which
#: is the only one that version caught.
_REACHING = (
    'pytester.runpytest("-p", "_state_guard")',
    'pytester.runpytest("-p_state_guard")',
    'pytester.runpytest("-p=_state_guard")',
    'pytester.runpytest(*("-p", "_state_guard"))',
    'pytester.runpytest(*["-p", "_state_guard"])',
    'pytester.runpytest(plugins=("_state_guard",))',
    'ARGS = ("-p", "_state_guard")\npytester.runpytest(*ARGS)',
    'def f():\n    args = ["-p", "_state_guard"]\n    return args',
    'pytester.makeconftest(\'pytest_plugins = ["_state_guard"]\')',
    'pytester.makepyfile(conftest=\'pytest_plugins = ["_state_guard"]\')',
    'pytester.makeini("[pytest]" + chr(10) + "addopts = -p _state_guard")',
    'pytester.makefile(".py", conftest=\'pytest_plugins = ["_state_guard"]\')',
    'pytester.makeconftest(dedent(\'pytest_plugins = ["_state_guard"]\'))',
)
_PROSE = (
    '# deliberately not `-p _state_guard`: see tests/test_state_guard.py',
    '"""Explains why `_state_guard` is not passed here."""',
    'def f():\n    """Not `-p _state_guard`, on purpose."""\n    return 1',
    '"_state_guard"\n',
    'x = "_state_guardian"',
    'x = "state_guard"',
    'x = "tests._state_guard"',
    'x = "__state_guard"',
    'x = "_state_guard_"',
    'NOTE = "we never pass -p _state_guard here, see test_state_guard.py"',
    'pytester.makeconftest("# nothing about _state_guardian here")',
    'pytester.makeconftest("# we never pass -p _state_guard")',
    'pytester.makepyfile(test_x=\'"""Not _state_guard."""\')',
    'pytester.makefile(".py", x="# _state_guard is not loaded here")',
)


#: THE TWO REACHING ROUTES THIS SCAN IS KNOWN NOT TO SEE, pinned so the gap is
#: a fixture rather than a memory. Both were measured to load the plugin in a
#: real nested session. If a later change makes one of them visible, this test
#: fails and the limit above should be struck — a declared limit that has
#: quietly stopped being a limit is its own defect, and the whole point of
#: writing them down is that they stay checked.
_KNOWN_MISSES = (
    'SG = \'pytest_plugins = ["_state_guard"]\'\npytester.makeconftest(SG)',
    'pytester.makeconftest("def pytest_configure(c):\\n"'
    ' "    import _state_guard\\n"'
    ' "    c.pluginmanager.register(_state_guard)\\n")',
)


@pytest.mark.parametrize("source", _KNOWN_MISSES, ids=("via-constant", "import-and-register"))
def test_the_declared_limits_are_still_limits(source):
    """The two routes the docstring declares unreachable are still unreached.

    **A DECLARED LIMIT THAT HAS STOPPED BEING ONE IS ITS OWN DEFECT.** Writing
    a limit down is worth nothing if nothing notices when it changes, and the
    direction that matters here is the happy one: if a later edit widens the
    scan so one of these becomes visible, this fails, and whoever did it
    should strike the bullet rather than leave the file claiming a blindness
    it no longer has.
    """
    assert not state_guard_references(source), (
        "a route the docstring declares unreachable is now caught:\n"
        f"  {source!r}\n"
        "That is good news, not a bug. Strike the matching bullet from "
        "`state_guard_references`'s WHAT IT DOES NOT REACH list and delete "
        "this fixture entry, so the file stops claiming a limit it has "
        "outgrown."
    )


@pytest.mark.parametrize("source", _REACHING, ids=range(len(_REACHING)))
def test_the_reference_scan_is_driven_both_ways(source):
    """Every reaching spelling is found and no prose spelling is.

    **THE ANTI-VACUITY SIBLING, AND IT IS NOT OPTIONAL HERE.** Every other
    gate this release added has one, and the guard above arrived without: an
    audit had to drive it by hand to learn that five reaching spellings were
    silent, and `.github/workflows/release.yml`'s own rule is that a refusal
    nothing observes is not known to be a refusal. Driving it by hand once is
    a measurement; this is the check.

    The prose half is the direction that gets a guard deleted rather than the
    one that gets a defect shipped, and it is the harder half. Four entries —
    `_state_guardian`, `tests._state_guard`, `__state_guard` and
    `_state_guard_` — are near-misses a naive `in` test fires on. The fifth,
    `state_guard`, is there for the opposite reason: it does NOT contain this
    module's name, and it is the shorter spelling rule 2's token boundary must
    not reach down to. **An earlier version of this sentence claimed
    `state_guard` for the first group**, which is the one case that does not
    hold — measured, `"_state_guard" in "state_guard"` is False.
    """
    assert state_guard_references(source), (
        f"a reaching spelling is invisible to the scan:\n  {source!r}\n"
        "It puts this module's name into a nested session's arguments, so the "
        "guard above would pass on a tree that has the defect."
    )


@pytest.mark.parametrize("source", _PROSE, ids=range(len(_PROSE)))
def test_the_reference_scan_is_silent_on_prose_and_near_misses(source):
    """Prose about the module, and names that merely resemble it, do not fire."""
    assert not state_guard_references(source), (
        f"the scan fires on text that loads nothing:\n  {source!r}\n"
        "A guard that reports its own explanation is a guard that gets deleted."
    )


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
    assert "test_planted.py changed process-global state OUTSIDE its tests" in proc.stdout, (
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
