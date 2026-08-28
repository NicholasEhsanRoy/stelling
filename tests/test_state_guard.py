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

import json
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
# through its `_run` helper, and no session it spawns registers this module.
# **THAT SENTENCE READ "no call site there hands a nested session `-p
# _state_guard`" UNTIL 2026-08-28, AND THAT IS A CLAIM ABOUT A SPELLING.** It
# was held by a scan of spellings for five rounds and it is held by a
# measurement of those sessions' plugin managers now; the section below is the
# record of why. The nested sessions in THIS file do pass `-p _state_guard`
# and are `subprocess.run`. What made it worth fixing rather than recording is
# that the reaching edit is the idiom this very file already uses, one
# argument away. The trajectory hangs off the session's `Config` now, so a
# nested session gets a fresh one and the separation is BY CONSTRUCTION.
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


# ── the guarantee, MEASURED rather than modelled ────────────────────────────
#
# **WHAT USED TO BE HERE WAS A SYNTACTIC SCAN, AND THE RECORD OF WHY IT IS
# GONE IS WORTH MORE THAN THE FUNCTION WAS.** Until 2026-08-28 the claim below
# rested on `state_guard_references`, a single-file AST scan over
# `tests/test_tripwire_plugin.py` that looked for the SPELLINGS which would
# make a nested session load `tests/_state_guard.py`. Five rounds of audit
# each closed one spelling, and each next round found another, always ONE
# INDIRECTION further out:
#
#   1. reading only direct `Call` arguments missed a tuple spread as `*ARGS`;
#   2. whole-string matching missed `makeconftest('pytest_plugins = [...]')`;
#   3. naming four `pytester` wrapper methods missed `makefile`, the base
#      method all four of them delegate to;
#   4. naming `makefile` still missed a MODULE-LEVEL SOURCE CONSTANT handed to
#      a `make*` call — which is the scanned file's own prevailing idiom, and
#      so the likeliest of the lot to be written by accident;
#   5. and the rule matched string LITERALS, so a generated conftest doing
#      `import _state_guard; config.pluginmanager.register(_state_guard)` —
#      where the name is an `import` alias and a `Name`, never a literal — was
#      invisible.
#
# Every one of those was a spelling live in the scanned file or one keystroke
# from it. Rounds 4 and 5 were then written down as DECLARED LIMITS and pinned
# by `test_the_declared_limits_are_still_limits`, whose own failure message
# said what to do when a limit is outgrown: strike the bullet and delete the
# fixture entry. This is that commit, for both at once.
#
# The class does not close syntactically, and the five rounds are the argument
# rather than the anecdote: every closed form admitted one more indirection,
# and "one more indirection" is not a finite set. The shape had a name in the
# struck prose already — ENUMERATING SPELLINGS OF THE THING INSTEAD OF THE
# THING, written into the `_GENERATES_SOURCE` comment at round 3 — and naming
# it did not stop it: `git show 519adcc -- tests/test_state_guard.py` is the
# same edit growing that set of names from four to five. There is one cut that
# does stop it. A check that models a behaviour is one indirection behind it,
# so where the question is what the program DOES, run the program. "Does a
# nested session load this module" is that question exactly.
#
# **AND ONE OF THE TWO PINNED LIMITS WAS PINNED AS SOURCE THAT COULD NOT HAVE
# REACHED.** The struck docstring said of both routes *"Measured to load the
# plugin"*. Driven now: the constant route does. The import-and-register route
# was pinned spelled `def pytest_configure(c)`, and pluggy validates a
# hookimpl's argument names against the hookspec's. Measured 2026-08-28 on
# pytest 9.1.1 / pluggy 1.6.0, that conftest never registers at all::
#
#     PluginValidationError: Plugin '.../conftest.py' for hook 'pytest_configure'
#     hookimpl definition: pytest_configure(c)
#     Argument(s) {'c'} are declared in the hookimpl but can not be found in
#     the hookspec
#
# The inner session dies during conftest loading at `ret == 3`, never reaches
# `pytest_configure`, and nobody registers anything. The ROUTE reaches —
# spelled `config`, and driven below. The SOURCE pinned for it did not, and
# nothing noticed, because that pin only ever asked whether the SCAN could see
# it: a guard that checks a claim is well-formed and never that it is true.
# The repair is not a better literal. It is that a route is now shown to reach
# by REACHING.
#
# **HOW THE INNER SESSION IS OBSERVED, AND WHY NOT THE OTHER WAYS**, because
# the choice is what the check can see. :data:`_OBSERVER` is a pytest plugin
# that reports through TWO halves whose union is what the reach is, and it
# matches on the resolved `__file__` of the registered object rather than on
# the name it was registered under, because a name is a spelling and spellings
# are the thing five rounds could not enumerate.
#
#   * **THE WRAP, which is the half that is not an enumeration.** On import it
#     wraps `pluggy.PluginManager.register`. That is the one function every
#     route into a plugin manager ends in — `-p`, `pytest_plugins`, a
#     `pytest11` entry point, `PYTEST_PLUGINS`, a conftest, a bare
#     `pluginmanager.register(mod)` — because `PytestPluginManager.register`
#     overrides it and calls `super().register`, and because nothing writes
#     `_name2plugin` directly (something that did would get no hooks wired and
#     no fixtures collected, so it would not be a loaded plugin at all). Once
#     the wrap is in, EVERY plugin manager in THAT PROCESS is covered, whether
#     or not this module is registered in it and whatever its environment
#     held. That is a closed class by construction, not a list of idioms.
#   * **THE HISTORIC HOOK, for what the wrap cannot have: registrations that
#     happened before it existed.** `pytest_plugin_registered` is declared
#     `historic=True`, so a manager this module is registered in reports every
#     plugin it already held. One real ordering needs it: `Config._preparse`
#     runs `consider_preparse` — where `-p _state_guard` is consumed — BEFORE
#     `consider_env`, which is what imports this module. In a child session
#     started with `-p _state_guard` the subject is already registered by the
#     time the wrap exists, and only the hook sees it.
#
# The module gets INTO a session through the `PYTEST_PLUGINS` environment
# variable, and the observation points that were NOT taken are named beside
# it, because a rejected alternative is half of what a check can see:
#
#   * `PYTEST_PLUGINS` rather than `-p`, because `-p` reaches the session this
#     file launches and nothing that session spawns. `Config._preparse` calls
#     `consider_env()` OUTSIDE the `PYTEST_DISABLE_PLUGIN_AUTOLOAD` branch
#     (pytest 9.1.1), so the observer arrives whether autoload is on or off —
#     measured both ways below — and `Pytester` deletes `PYTEST_ADDOPTS` from
#     the environment of a nested session but not this.
#   * NOT `runpytest_inprocess` and what it returns. `Pytester.inline_run`
#     does build a `HookRecorder` on the INNER `Config.pluginmanager`, and
#     `runpytest_inprocess` drops it; reaching for what it kept would in any
#     case only answer for the sessions the observed file happens to run
#     in-process, and that file runs sessions as subprocesses too.
#   * NOT a `pytest_configure` hook written into the inner tree, and not a
#     conftest that writes a file the outer test reads: the inner tree is
#     written by the code under observation, which calls `makeconftest` and
#     OVERWRITES it. A check that has to share a file with the thing it is
#     checking is a check that thing can switch off by accident.
#
# **WHAT IT DOES NOT REACH.** A behavioural check has limits like any other,
# and these are declared rather than discovered — **THE FIRST TWO BECAUSE AN
# AUDITOR DISCOVERED THEM, WHICH IS THE WHOLE OF WHY THIS LIST IS NOW WRITTEN
# AROUND THE MECHANISM RATHER THAN AROUND THE INTENTION.** This list used to
# open *"it sees the sessions IT RUNS — `sys.executable`, this venv's
# installed distributions, this pytest"*, naming only another interpreter,
# another pytest and another entry-point set as out of reach. A blinded audit
# of 0.2.1 drove a child spawned with an explicit `env=` — same interpreter,
# same pytest, same venv, so INSIDE the declared reach — which registered
# `tests/_state_guard.py` for real and produced no row, with all three
# assertions of the check below staying green over it. The sentence claimed a
# reach the wiring did not have; the wrap above is the answer for one class
# and these bullets are the answer for the rest:
#
#   * **the reach is per PROCESS, and it is not the same on both sides of that
#     line.** In the process the observer is loaded in, every REGISTRATION is
#     reported — that is the wrap, and it does not care how the session was
#     configured. **A CHILD process is reached only if THIS PROCESS'S OWN
#     ENTRY IN `PYTEST_PLUGINS` ARRIVES IN IT**, because that entry is the
#     whole of how the observer gets there. Joined by other names it still
#     arrives and the child is SEEN — measured 2026-08-28 both ways round,
#     `"_sg_observer,_state_guard"` and `"_state_guard,_sg_observer"`, and the
#     second of those is the historic hook earning its keep: pytest imports
#     the names in order, so there the subject is registered before the wrap
#     exists and only the replay reports it. DROPPED or REPLACED, the entry
#     does not arrive and the child is not seen.
#
#     **THIS BULLET SAID "only if it inherits this process's ENVIRONMENT"
#     UNTIL 2026-08-28, AND THAT WAS ONE SPELLING TOO WIDE.** A second
#     blinded audit spawned a child with `{**os.environ, "PYTEST_PLUGINS":
#     "_state_guard"}` — which inherits the environment, is the most
#     idiomatic way there is to put a pytest plugin into a child from an
#     environment, and is unseen here. The variable is the reach; the
#     environment is not. The correction is narrower than the sentence it
#     replaces, which is the direction a limit is allowed to move only when
#     something drives it.
#
#     ONE LIMIT, TWO FIXTURES, AND THEY ARE NOT TWO LIMITS. Dropping the
#     variable and replacing its value are the same act — the entry does not
#     arrive — so they are two drives of one predicate rather than a set of
#     idioms to keep adding to.
#     `test_a_CHILD_PROCESS_THAT_SCRUBS_THE_ENVIRONMENT_loads_it_UNSEEN`
#     drives the dropped half and
#     `test_a_CHILD_WHOSE_PYTEST_PLUGINS_IS_REPLACED_loads_it_UNSEEN` drives
#     the replaced half; each asserts the ROUTE first — the guard fires
#     inside that child and names the planted test — and only then that this
#     instrument is blind to it, and the second carries the POSITIVE CONTROL
#     that makes the boundary exact rather than a vague blindness: a second
#     child differing in nothing but that value, seen. **A LIMIT PINNED ONLY
#     BY "THE INSTRUMENT CANNOT SEE IT" IS THE MISTAKE THE
#     `pytest_configure(c)` FIXTURE MADE**, one layer out.
#     NOT LIVE UNDER TODAY'S SUBJECT, and that is a dated reading of another
#     file rather than anything held here: measured 2026-08-28 at `2e4b780`,
#     `tests/test_tripwire_plugin.py` has no `import subprocess`, no `env=`
#     and no mention of `PYTEST_PLUGINS`; its ten `subprocess` occurrences
#     are all `pytester.runpytest_subprocess` or prose, and every child it
#     spawns goes through `Pytester.popen`, which does
#     `env = os.environ.copy()` and therefore carries the entry. A file that
#     grew one would be outside this reach, and nothing here would say so —
#     which is why the limit is written down rather than reasoned away;
#   * **it watches REGISTRATION, not activation.** A conftest that does
#     `from _state_guard import state_guard` re-exports the fixtures into its
#     own namespace, where pytest collects them: they run, and no plugin was
#     ever registered, so there is nothing for this to report. Pinned the same
#     way by `test_a_conftest_that_IMPORTS_THE_FIXTURES_makes_them_live_UNSEEN`
#     — the guard is shown FIRING in the inner session, and the report is
#     shown empty;
#   * it sees the sessions IT RUNS — `sys.executable`, this venv's installed
#     distributions, this pytest. A session under another interpreter, another
#     pytest, or another set of `pytest11` entry points can have a different
#     plugin set and this says nothing about it;
#   * the developer's environment is REMOVED from the input rather than
#     inherited: :func:`_observe` drops `PYTEST_ADDOPTS` and
#     `PYTEST_DISABLE_PLUGIN_AUTOLOAD` from the child environment so that the
#     verdict is the same for everybody, which is the defect this repository
#     keeps re-finding in checks that read the ambient shell. The price is
#     that the verdict is about the autoload-ENABLED configuration, and the
#     disabled one is measured here rather than asserted;
#   * it identifies the module by the REAL PATH of one file. A copy of
#     `tests/_state_guard.py` loaded from somewhere else is a different module
#     object with its own everything, and is deliberately not a finding;
#   * a session that never reaches `pytest_configure` reports no session row —
#     the `pytest_configure(c)` measurement above is exactly such a session.
#     So a report with no `loaded` row in it is not by itself good news, and
#     the check below refuses to read it as good news: it asserts the observed
#     run exited 0, and that nested sessions of BOTH kinds were seen at all.
#
# **MEASURED 2026-08-28** — jax 0.11.0, pytest 9.1.1, pluggy 1.6.0, python
# 3.12.3 — running `tests/test_tripwire_plugin.py` under this observer, with
# `PYTEST_DISABLE_PLUGIN_AUTOLOAD` unset and set to `1`, identical both ways::
#
#     31 passed    37 sessions: 1 outer, 26 nested IN-PROCESS, 10 in child
#                  processes; 0 registrations of tests/_state_guard.py in any
#                  of them
#
# RE-DERIVED UNCHANGED after the wrap around `pluggy.PluginManager.register`
# was added, 2026-08-28 at `8dae8cb` plus this change: 37 rows, 26 in-process
# nested, 10 in child processes, 0 registrations. The wrap widens what CAN be
# seen; it found nothing new here, which is the reading a widened instrument
# is supposed to be able to give.
#
# The autoload-OFF half of that was driven by hand rather than through
# :func:`_observe`, because :func:`_observe` is what drops the variable: a
# measurement of a configuration the helper refuses to reproduce can only be
# taken beside it. Single runs of that file measured 10.82s unobserved
# against 10.58s observed, so the observer is not what this check costs. What
# it costs is running the file at all, and that is reported at the check.
#
# **THE GUARANTEE STILL DOES NOT LIVE HERE, RE-VERIFIED RATHER THAN
# REPEATED.** The trajectory hangs off the session's `Config` — `_TRAJECTORY`
# in `tests/_state_guard.py`, and the comment above it — so a nested session
# builds a fresh one and the separation holds whether or not the module loads
# there. Checked against the current code on 2026-08-28: every module-level
# binding in that file is immutable, so there is no module-level state for a
# second session to share — `ABSENT` a `str`, `ENV_PREFIXES`, `ENTRIES` and
# `PINNED_EXEMPTIONS` tuples, `ENTRY_NAMES` a `frozenset`, `_TRAJECTORY` a
# `pytest.StashKey`, `_GONE` a bare sentinel. The two controls that drive it
# are `test_a_module_leak_is_still_named_when_a_TEST_RUNS_A_NESTED_SESSION`
# and `test_a_WELL_BEHAVED_module_stays_silent_when_a_test_runs_a_nested_
# session`, both below, and both are green. The argument holds. This check is
# still the belt rather than the braces — but it is a belt that measures now.
#
# **THE SENTENCE THAT USED TO END THIS PARAGRAPH WAS "and it can no longer be
# defeated by a spelling nobody thought of", AND IT WAS FALSE WHEN IT WAS
# WRITTEN.** A blinded audit of 0.2.1 defeated it with a spelling nobody had
# thought of: a child process spawned with an explicit `env=`. The claim was
# also the wrong SHAPE — it announced a completeness instead of naming a
# mechanism, which is the same move the four docstrings before the syntactic
# scan's last one made, and which the scan's own history is a record of. What
# is true, and is all that is claimed now: within one process the enumeration
# is gone, because `pluggy.PluginManager.register` is one door rather than a
# set of names; across a process boundary the reach is exactly this process's
# own entry in `PYTEST_PLUGINS` arriving in the child — **NOT "inheritance of
# an environment", which is what this sentence said until a second audit
# spawned a child that inherited the environment and replaced that one value**
# — and the ways past it are declared above and pinned by fixtures that assert
# the route reaches before they assert this cannot see it, one of them with
# the positive control beside it. **AND THE AUDITOR'S DEEPER POINT IS
# CONCEDED AND ANSWERED RATHER THAN ARGUED WITH:** an enumeration of spawn
# idioms would have been the same defect one layer out, so there is no
# enumeration of spawn idioms here. There is one wrap, and beyond it, declared
# blindness — of one variable's entry, named and held at its boundary.


#: The observer, written into a temporary directory by :func:`_observe`.
#:
#: It reads its wiring with ``os.environ[...]`` and not ``.get``, on purpose:
#: an observer that quietly disables itself when its wiring is missing writes
#: an empty report, and an empty report from a broken instrument is
#: indistinguishable from a clean tree. Missing wiring must be a usage error
#: in the session that loads it, loudly, in the captured output below.
_OBSERVER = r'''"""Every registration of one FILE, in every session in this process.

Written by ``tests/test_state_guard.py``; the prose there says why the
question is asked by running the program rather than by reading it.
"""

import json
import os

import pluggy

_REPORT = os.environ["STATE_GUARD_OBSERVER_REPORT"]
_SUBJECT = os.path.realpath(os.environ["STATE_GUARD_OBSERVER_SUBJECT"])

#: ``id(manager)`` -> ordinal within this process, with the manager itself
#: held in ``_KEEP``. The strong reference is the point: a garbage-collected
#: nested session can have its ``id()`` handed to the next one, and two
#: sessions conflated into one is the exact misreading that would let a
#: nested load be attributed to the outer session.
_ORDINALS = {}
_KEEP = []

#: ``(session ordinal, id(plugin))`` already reported. The two halves below
#: overlap on every registration made after this module was imported -- the
#: wrap sees it going in, the historic hook sees it a moment later -- and one
#: act must be one row.
_SEEN = set()


def _session(manager):
    ordinal = _ORDINALS.get(id(manager))
    if ordinal is None:
        ordinal = _ORDINALS[id(manager)] = len(_ORDINALS)
        _KEEP.append(manager)
    return ordinal


def _emit(row):
    row["pid"] = os.getpid()
    # Opened for append and written one short line at a time, because the
    # observed tree runs sessions in child processes and they share this file.
    with open(_REPORT, "a", encoding="utf-8") as report:
        report.write(json.dumps(row, sort_keys=True) + "\n")


def _note(manager, plugin, name):
    """Report one registration of the subject, at most once.

    Matched on the resolved ``__file__`` of the registered object and never on
    ``name``: the name a plugin is registered under is a spelling, and
    spellings are the thing five rounds of a scan could not enumerate.
    """
    where = getattr(plugin, "__file__", None)
    if where is None or os.path.realpath(where) != _SUBJECT:
        return
    ordinal = _session(manager)
    if (ordinal, id(plugin)) in _SEEN:
        return
    _SEEN.add((ordinal, id(plugin)))
    _KEEP.append(plugin)  # so no later object can inherit this ``id()``
    _emit({"kind": "loaded", "session": ordinal, "name": name})


_REGISTER = pluggy.PluginManager.register


def _register(self, plugin, name=None):
    """``pluggy.PluginManager.register``, wrapped for THIS PROCESS.

    THE CHOKE POINT, AND IT IS WHY THIS HALF IS NOT AN ENUMERATION. Every
    route by which a plugin enters a plugin manager -- ``-p``,
    ``pytest_plugins``, a ``pytest11`` entry point, ``PYTEST_PLUGINS``, a
    conftest, a bare ``pluginmanager.register(mod)`` -- ends in this one
    function; ``PytestPluginManager.register`` overrides it and calls
    ``super().register``. Nothing writes ``_name2plugin`` directly, and
    something that did would not get its hooks wired or its fixtures
    collected, so it would not be a loaded plugin at all.

    So within this process the reach does not depend on how a session was
    configured, on what its environment held, or on whether this module is
    registered in it. It depends on pluggy having one door.
    """
    registered = _REGISTER(self, plugin, name)
    if registered is not None:
        _note(self, plugin, registered)
    return registered


if getattr(pluggy.PluginManager.register, "_state_guard_observer", None) is None:
    _register._state_guard_observer = True
    pluggy.PluginManager.register = _register


def pytest_plugin_registered(plugin, plugin_name, manager):
    """The half the wrap above cannot have: what was registered BEFORE it.

    A HISTORIC hook, so a manager this module is registered in reports every
    plugin it already held. That matters for one real ordering:
    ``Config._preparse`` runs ``consider_preparse`` -- which is where ``-p
    _state_guard`` is consumed -- BEFORE ``consider_env``, which is what
    imports this module. In a child session started with ``-p _state_guard``
    the subject is therefore already registered by the time the wrap exists,
    and only this hook sees it.
    """
    _note(manager, plugin, plugin_name)


def pytest_configure(config):
    """One row per session, so that "nothing loaded it" can be told apart
    from "nothing ran"."""
    _emit(
        {
            "kind": "session",
            "session": _session(config.pluginmanager),
            "args": list(config.invocation_params.args),
        }
    )
'''


class _Observed:
    """One observer report, staged into what the assertions below ask of it.

    A session is identified by ``(pid, ordinal)``: the ordinal alone repeats
    across processes. The OUTER session is the first one to configure, because
    nothing it spawns can configure before it does.

    **THE NESTED SESSIONS ARE COUNTED FROM EVERY ROW AND NOT FROM THE SESSION
    ROWS ALONE**, and that is not tidiness. A session the observer is not
    registered in emits no ``pytest_configure`` row, but the wrap around
    ``pluggy.PluginManager.register`` still reports what it loaded. That is
    the whole subject of
    ``test_an_IN_PROCESS_session_that_SCRUBS_THE_ENVIRONMENT_is_still_named``:
    a nested session with a registration and no session row. Counting only
    session rows would have called that session absent while holding the
    evidence that it ran.
    """

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        configured = [row for row in rows if row["kind"] == "session"]
        self.outer = (
            (configured[0]["pid"], configured[0]["session"]) if configured else None
        )
        seen: list[tuple[int, int]] = []
        for row in rows:
            key = (row["pid"], row["session"])
            if key != self.outer and key not in seen:
                seen.append(key)
        self.here = [key for key in seen if self.outer and key[0] == self.outer[0]]
        self.elsewhere = [key for key in seen if self.outer and key[0] != self.outer[0]]
        #: Registrations of the subject in a session that is NOT the outer
        #: one. The outer session is excluded because its plugin set is
        #: decided entirely by :func:`_observe` -- its own argv and its own
        #: ``PYTEST_PLUGINS`` -- and never by the file under observation, so
        #: nothing that file does can hide there.
        self.loaded = [
            row
            for row in rows
            if row["kind"] == "loaded" and (row["pid"], row["session"]) != self.outer
        ]

    def __str__(self) -> str:
        """The counts, then every registration — and NOT every row.

        The observed file spawns dozens of sessions, and printing all of them
        buries the one line a reader needs. The counts are here because they
        are what says the run happened at all; the registrations are here with
        the arguments of the session that made them, because that is what a
        reader has to go and find.
        """
        args = {
            (row["pid"], row["session"]): row["args"]
            for row in self.rows
            if row["kind"] == "session"
        }
        made = [
            f"  pid={row['pid']} session={row['session']} as {row['name']!r} "
            f"args={args.get((row['pid'], row['session']), '<never configured>')}"
            for row in self.loaded
        ]
        return "\n".join(
            [
                f"{len(self.rows)} rows: "
                f"{'an outer session' if self.outer else 'NO outer session'}, "
                f"{len(self.here)} nested in its process, "
                f"{len(self.elsewhere)} in child processes",
                "registrations of the subject outside the outer session:",
                *(made or ["  (none)"]),
            ]
        )


def _observe(tmp_path, *args, cwd) -> tuple[subprocess.CompletedProcess, _Observed]:
    """Run one pytest session over ``args``, watching every session it spawns.

    ``PYTHONPATH`` carries the observer's directory and ``tests/``: the second
    is what makes `_state_guard` importable BY NAME in the nested sessions, and
    a route that could not import it would come back silent for the wrong
    reason — an instrument reporting an ``ImportError`` as an absence of
    defect.

    ``PYTEST_ADDOPTS`` and ``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` are dropped from
    the child environment rather than inherited, so that this reports the same
    verdict to everybody. What that costs is stated in the limits above.
    """
    workshop = tmp_path / "observer"
    workshop.mkdir(exist_ok=True)
    (workshop / "_sg_observer.py").write_text(_OBSERVER, encoding="utf-8")
    report = tmp_path / "observed.jsonl"

    env = dict(os.environ)
    # The developer's shell is not an input to this measurement.
    for ambient in ("PYTEST_ADDOPTS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        env.pop(ambient, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(workshop),
            str(TESTS),
            *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []),
        ]
    )
    env["PYTEST_PLUGINS"] = "_sg_observer"
    env["STATE_GUARD_OBSERVER_REPORT"] = str(report)
    env["STATE_GUARD_OBSERVER_SUBJECT"] = str(TESTS / "_state_guard.py")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            # never shuffled: see `deterministic_order_args` in conftest.py
            "-p", "no:randomly", *args,
        ],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
    )
    lines = (
        report.read_text(encoding="utf-8").splitlines() if report.exists() else []
    )
    return proc, _Observed([json.loads(line) for line in lines if line])


def test_no_nested_session_in_the_tripwire_file_loads_this_module(tmp_path):
    """No session `tests/test_tripwire_plugin.py` spawns registers this module.

    THE HALF OF THE PARAGRAPH ABOVE THAT IS A CLAIM RATHER THAN A STORY, and
    it is decided by running that file rather than by reading it. Its `_run`
    helper reaches `pytester.runpytest` IN-PROCESS, so a nested session there
    that loaded this module would share the outer session's bookkeeping — the
    exact shape the trajectory-on-`Config` fix exists to make impossible. The
    fix makes it impossible BY CONSTRUCTION; this makes the argument
    re-derivable, which neither the line number and call count it first
    replaced nor the syntactic scan that replaced those ever were.

    THE THREE ASSERTIONS ARE NOT ONE ASSERTION AND TWO ORNAMENTS. A report
    with no registration in it is worth nothing until the run that produced it
    is known to have run: a nested session that dies before `pytest_configure`
    writes no row at all, so `exited 0` and `nested sessions were seen` are
    what separate "nothing loaded it" from "nothing happened". Both kinds are
    required because the observer's reach over both is a claim in the prose
    above, and a claim nothing drives is a story.
    """
    # The observed file gates its whole module on jax, and a session that
    # collects nothing spawns nothing.
    pytest.importorskip("jax")
    proc, seen = _observe(
        tmp_path, str(TESTS / "test_tripwire_plugin.py"), cwd=REPO
    )
    assert proc.returncode == 0, (
        "tests/test_tripwire_plugin.py does not pass under observation, so "
        "what was measured is not the file this asserts about. If the "
        "observer is what broke it, the instrument has changed its subject "
        f"and that is the bug.\n{proc.stdout}\n{proc.stderr}"
    )
    assert seen.here and seen.elsewhere, (
        "the observed run spawned no nested session of one or both kinds "
        f"({len(seen.here)} in-process, {len(seen.elsewhere)} in a child "
        "process), so this check measured nothing and must not read as "
        "green. Either the file stopped running nested sessions — in which "
        "case this test and the sentence it holds up should go — or the "
        f"observer stopped reaching them.\n{seen}\n{proc.stdout}"
    )
    assert not seen.loaded, (
        "a nested session of tests/test_tripwire_plugin.py registered "
        f"tests/_state_guard.py:\n{seen}\n"
        "Its sessions reach `pytester.runpytest` IN-PROCESS, so loading this "
        "module there puts a nested session and the outer one on the same "
        "bookkeeping — the defect the trajectory-on-`Config` fix closed. If a "
        "nested session there genuinely needs the guard, it needs a "
        "SUBPROCESS, the way this file's own nested sessions get it."
    )


#: THE TWO ROUTES THE SYNTACTIC SCAN COULD NOT SEE, now the drive for what
#: replaced it. Each is a whole test module for a MIDDLE session; each spawns
#: an inner session that really does register `tests/_state_guard.py`, and
#: :func:`_observe` is what has to notice. They are the anti-vacuity half of
#: the check above and they are not optional: a check driven only by its own
#: mutation is a check nobody has seen work.
#:
#: `assert_outcomes(passed=1)` inside each plant is the plant's own control.
#: A route that stopped reaching because the inner session broke would
#: otherwise come back looking like a route that never reached.
_VIA_A_CONSTANT = r'''pytest_plugins = ["pytester"]

# The scanned file's prevailing idiom for generated source: written once at
# module level, handed to a `make*` call wherever it is wanted. One name away
# from the literal a syntactic scan could see, and invisible to it.
SG = 'pytest_plugins = ["_state_guard"]'


def test_middle(pytester):
    pytester.makepyfile(test_inner="def test_inner():\n    assert True\n")
    pytester.makeconftest(SG)
    pytester.runpytest().assert_outcomes(passed=1)
'''

#: **SPELLED `config` AND NOT `c`, AND THE DIFFERENCE IS THE WHOLE FINDING.**
#: The struck `_KNOWN_MISSES` fixture pinned this route spelled
#: `pytest_configure(c)`, which pluggy refuses at conftest registration, so
#: the inner session died at `ret == 3` and never registered anything. See
#: the measurement in the prose above.
_IMPORT_AND_REGISTER = r'''pytest_plugins = ["pytester"]


def test_middle(pytester):
    pytester.makepyfile(test_inner="def test_inner():\n    assert True\n")
    pytester.makeconftest(
        "def pytest_configure(config):\n"
        "    import _state_guard\n"
        "    config.pluginmanager.register(_state_guard)\n"
    )
    pytester.runpytest().assert_outcomes(passed=1)
'''

#: The silent direction, and it is the one that gets a check deleted rather
#: than the one that ships a defect: a nested session that MENTIONS this
#: module in a comment it generates and loads nothing.
_ONLY_MENTIONS_IT = r'''pytest_plugins = ["pytester"]


def test_middle(pytester):
    pytester.makepyfile(test_inner="def test_inner():\n    assert True\n")
    pytester.makeconftest(
        "# deliberately not `-p _state_guard`: see tests/test_state_guard.py\n"
    )
    pytester.runpytest().assert_outcomes(passed=1)
'''


def _observe_plant(
    tmp_path, source: str
) -> tuple[subprocess.CompletedProcess, _Observed]:
    """Run one whole planted middle module under observation, in its own tree."""
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "test_middle.py").write_text(source, encoding="utf-8")
    return _observe(tmp_path, "test_middle.py", cwd=tree)


@pytest.mark.parametrize(
    "source",
    (_VIA_A_CONSTANT, _IMPORT_AND_REGISTER),
    ids=("via-constant", "import-and-register"),
)
def test_a_nested_session_that_reaches_this_module_IS_NAMED(tmp_path, source):
    """Both routes the scan declared unreachable are reached, and reported.

    These two are the evidence that the replacement is not the same instrument
    in different clothes. They were `_KNOWN_MISSES` — pinned as blind spots,
    with the docstring above them declaring the blindness — and the pin's own
    failure message said that a limit which stops being a limit gets struck
    and its fixture entry deleted. They are struck, and this is what they
    became.
    """
    proc, seen = _observe_plant(tmp_path, source)
    assert proc.returncode == 0, (
        f"the plant did not run, so it reached nothing\n{proc.stdout}\n{proc.stderr}"
    )
    assert seen.here, (
        f"the plant spawned no nested session in its own process\n{seen}\n{proc.stdout}"
    )
    assert seen.loaded, (
        "a route measured to register tests/_state_guard.py in a nested "
        f"session came back silent:\n{source}\n{seen}\n{proc.stdout}\n"
        "Either the route stopped reaching — in which case say so here with "
        "the measurement — or the observer stopped seeing it, and then the "
        "check above passes on a tree that has the defect."
    )


def test_a_nested_session_that_only_MENTIONS_this_module_is_not_named(tmp_path):
    """The silent direction, without which the two above prove nothing.

    An instrument that is stuck on reports every route as reaching, and would
    pass both firing tests above while being worthless. This plant spawns a
    real nested session, writes this module's name into the conftest that
    session loads, and loads nothing — which is the shape of every honest
    "deliberately not `-p _state_guard`" remark a reader might write.
    """
    proc, seen = _observe_plant(tmp_path, _ONLY_MENTIONS_IT)
    assert proc.returncode == 0, (
        f"the plant did not run\n{proc.stdout}\n{proc.stderr}"
    )
    assert seen.here, (
        "the plant spawned no nested session, so it shows nothing\n"
        f"{seen}\n{proc.stdout}"
    )
    assert not seen.loaded, (
        f"the observer fires on a nested session that loads nothing:\n{seen}\n"
        "A guard that reports its own explanation is a guard that gets deleted."
    )


# ── the reach, at the boundaries two audits found it at ─────────────────────
#
# FOUR PLANTS, AND EACH OF THEM ASSERTS THE ROUTE BEFORE ANYTHING ASSERTS THE
# INSTRUMENT. The first shows the wrap holding where the wiring does not: a
# nested session created after `PYTEST_PLUGINS` has been taken out of
# `os.environ` is still named, so environment-scrubbing is not a way past this
# in one process. The other three are declared LIMITS, and they are pinned in
# the shape the `pytest_configure(c)` fixture failed to be — the subject
# first, the blindness second. Two of them are one limit driven twice: a child
# that never receives this process's entry in `PYTEST_PLUGINS`, dropped in the
# one and replaced in the other. Each of the four drives the guard itself into
# the inner session and reads its report, because "the module is loaded there"
# is a claim about what the module DOES, and this file decides those by
# running it.
#
# **EVERY LIMIT FIXTURE HERE ALSO CARRIES A POSITIVE CONTROL, AND ONE OF THEM
# ONLY DOES BECAUSE WRITING THIS PARAGRAPH FOUND IT MISSING.** A limit fixture
# asserts that something is NOT reported, and an instrument that reports
# nothing at all satisfies that as well as a working one does — the same
# absence-read-as-evidence this file exists to refuse. So: the re-export
# fixture asserts `seen.here`, the nested session it plants having been seen;
# the env-scrubbed child fixture had NO such half until 2026-08-28 and now
# plants an ordinary `pytester` session beside the invisible child for it; and
# `_ONE_VARIABLE_TWO_CHILDREN` is the strongest of the three — two children
# differing in one environment value, both shown loading the guard, EXACTLY
# ONE row required, so it fails if the blindness widens and equally if it
# narrows. That last shape arrived only after a second audit found the
# sentence above it one spelling too wide.


#: An IN-PROCESS nested session created after the observer's own wiring has
#: been removed from the environment. The wrap around
#: `pluggy.PluginManager.register` is the only thing that can see this: the
#: observer is never registered in that session, so it gets no
#: `pytest_configure` row and the historic hook never fires for it.
_SCRUBBED_IN_PROCESS = r'''import os

import pytest


def test_middle(tmp_path):
    inner = tmp_path / "test_inner.py"
    inner.write_text("def test_inner():\n    assert True\n", encoding="utf-8")
    # The wiring the observer arrives by, taken out before the session exists.
    os.environ.pop("PYTEST_PLUGINS", None)
    assert pytest.main([
        "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
        "-p", "_state_guard", str(inner),
    ]) == 0
'''

#: A CHILD PROCESS spawned with an explicit `env=`. Same interpreter, same
#: pytest, same venv — and no `PYTEST_PLUGINS`, so the observer is not there
#: at all. This is the route a blinded audit of 0.2.1 drove; the plant asserts
#: it REACHES by making the guard fire inside the child and reading the
#: report, and `tests/` is derived from the path the observer was pointed at
#: rather than typed, so the child cannot resolve a different module than the
#: one being watched.
_SCRUBBED_CHILD = r'''pytest_plugins = ["pytester"]

import os
import subprocess
import sys


def test_the_observer_is_awake(pytester):
    """An ordinary nested session, which this instrument DOES see.

    The positive control. Without it, `not seen.loaded` below is satisfied
    just as well by an observer that saw nothing at all -- and an instrument
    reporting its own absence as an absence of defect is the shape this whole
    file is a record of refusing.
    """
    pytester.makepyfile(test_inner="def test_inner():\n    assert True\n")
    pytester.runpytest().assert_outcomes(passed=1)


def test_middle(tmp_path):
    child = tmp_path / "test_child.py"
    child.write_text(
        "import os\n\n\ndef test_leaks():\n"
        "    os.environ['STELLING_PLANTED_BY_THE_CHILD'] = '1'\n",
        encoding="utf-8",
    )
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.path.dirname(os.environ["STATE_GUARD_OBSERVER_SUBJECT"]),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "no:randomly", "-p", "_state_guard", str(child)],
        capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    # THE SUBJECT OF THE LIMIT, ASSERTED HERE AND NOT ASSUMED: the guard is
    # loaded in that child and doing its job, naming the planted test at its
    # own teardown. A limit pinned only by "the instrument cannot see it" can
    # be satisfied by source that never reached, which is exactly what the
    # struck `pytest_configure(c)` entry turned out to be.
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "test_child.py::test_leaks changed process-global state" in proc.stdout, (
        proc.stdout + proc.stderr
    )
'''

#: A conftest that IMPORTS the two fixtures by name instead of registering the
#: module. pytest collects fixtures out of a conftest's namespace, so they are
#: live in the inner session and nothing was ever registered — the boundary
#: between what this instrument watches and what makes the guard run.
_IMPORTS_THE_FIXTURES = r'''pytest_plugins = ["pytester"]


def test_middle(pytester):
    pytester.makepyfile(
        test_inner="import os\n\n\ndef test_leaks():\n"
        "    os.environ['STELLING_PLANTED_INNER'] = '1'\n"
    )
    pytester.makeconftest(
        "from _state_guard import state_guard, module_state_guard  # noqa: F401\n"
    )
    # THE SUBJECT OF THE LIMIT, ASSERTED: the fixtures are LIVE in there.
    pytester.runpytest().stdout.fnmatch_lines(["*changed process-global state*"])
'''


def _unconfigured(seen: _Observed) -> list[dict]:
    """Registrations in a session that never emitted a ``pytest_configure`` row.

    Which is to say: the ones only the wrap can have reported, because the
    observer was not registered in that session for the historic hook to reach
    it. Without this distinction
    ``test_an_IN_PROCESS_session_that_SCRUBS_THE_ENVIRONMENT_is_still_named``
    would pass just as well on a tree where the scrub silently did nothing.
    """
    configured = {
        (row["pid"], row["session"]) for row in seen.rows if row["kind"] == "session"
    }
    return [
        row for row in seen.loaded if (row["pid"], row["session"]) not in configured
    ]


def test_an_IN_PROCESS_session_that_SCRUBS_THE_ENVIRONMENT_is_still_named(tmp_path):
    """Taking the observer's wiring out of the environment does not hide a load.

    **THE ANSWER TO "THE ENUMERATION MERELY MOVED".** The audit's fair hit was
    that replacing *spellings that load a plugin* with *spawn idioms that
    inherit an environment* trades one open set for another. It would have,
    if the reach were the environment. In one process it is not: it is
    `pluggy.PluginManager.register`, which every route ends in, wrapped once.
    This plant removes `PYTEST_PLUGINS` from `os.environ` and only then builds
    the session, so the observer is registered nowhere in it and the historic
    hook cannot fire — and the registration is named anyway.

    The second assertion is what stops this passing for the wrong reason: the
    row must belong to a session that emitted NO `pytest_configure` row. If
    the scrub had quietly failed, the observer would have been registered
    there, the hook would have reported it, and a check that only asked
    "was it named" would have called that a pass.
    """
    proc, seen = _observe_plant(tmp_path, _SCRUBBED_IN_PROCESS)
    assert proc.returncode == 0, (
        f"the plant did not run, so it loaded nothing\n{proc.stdout}\n{proc.stderr}"
    )
    assert seen.loaded, (
        "an in-process nested session registered tests/_state_guard.py and "
        f"was not named:\n{seen}\n{proc.stdout}\n"
        "That is the defect this check exists for, arriving through the one "
        "door the wrap is on. If `pluggy.PluginManager.register` has stopped "
        "being that door, this file's central claim is gone and the prose "
        "above needs rewriting, not this test."
    )
    assert _unconfigured(seen), (
        "the load WAS named, but by a session that also configured the "
        f"observer — so the scrub did nothing and this proves nothing:\n{seen}"
    )


def test_a_CHILD_PROCESS_THAT_SCRUBS_THE_ENVIRONMENT_loads_it_UNSEEN(tmp_path):
    """A DECLARED LIMIT, pinned subject-first: the route reaches, and this is blind.

    A child spawned with an explicit `env=` does not inherit `PYTEST_PLUGINS`,
    so the observer is not in it and there is no wrap in that process either.
    The plant proves the route is real the only way that means anything — the
    guard fires in the child and names the planted test — and this asserts the
    report is empty of it.

    **WHY THIS IS DECLARED RATHER THAN CLOSED.** Closing it means injecting
    wiring into children the observed program deliberately scrubbed, which
    changes the program under observation; and a second process shares no
    bookkeeping, which is the whole of the harm this guard was built around.
    The honest form is a limit that is checked, and the check is here.
    """
    proc, seen = _observe_plant(tmp_path, _SCRUBBED_CHILD)
    assert proc.returncode == 0, (
        "the route no longer reaches: the guard did not fire in the "
        f"env-scrubbed child\n{proc.stdout}\n{proc.stderr}\n"
        "Say what changed, with the measurement. A limit whose route has "
        "quietly stopped reaching is a limit nobody is checking."
    )
    assert seen.here, (
        "the plant's ordinary nested session was not seen either, so this "
        f"report is an instrument saying nothing rather than a boundary:\n{seen}"
    )
    assert not seen.loaded, (
        f"the env-scrubbed child IS visible now:\n{seen}\n"
        "That is good news, not a bug. Strike the matching bullet from the "
        "WHAT IT DOES NOT REACH list above and delete this fixture, so the "
        "file stops declaring a blindness it has outgrown."
    )


def test_a_conftest_that_IMPORTS_THE_FIXTURES_makes_them_live_UNSEEN(tmp_path):
    """The other declared limit: this watches REGISTRATION, not activation.

    `from _state_guard import state_guard` puts the fixture in a conftest's
    namespace, where pytest collects it. It runs; no plugin is registered;
    there is nothing for a register-watcher to see. The plant asserts the
    first half by making the guard fire in the inner session, and this asserts
    the second.

    Harmless for the same reason everything else here is: the fixtures reach
    the trajectory through `request.config`, so an inner session gets its own.
    Named because a limits list that omits the route beside the one an audit
    found is a list that will be wrong again.
    """
    proc, seen = _observe_plant(tmp_path, _IMPORTS_THE_FIXTURES)
    assert proc.returncode == 0, (
        "the route no longer reaches: the re-exported fixtures did not fire "
        f"in the inner session\n{proc.stdout}\n{proc.stderr}"
    )
    assert seen.here, (
        f"the plant spawned no nested session, so it shows nothing\n{seen}"
    )
    assert not seen.loaded, (
        f"a conftest that only IMPORTS the fixtures is reported now:\n{seen}\n"
        "That is good news. Strike the matching bullet from the WHAT IT DOES "
        "NOT REACH list above and delete this fixture."
    )


#: TWO CHILDREN DIFFERING IN ONE VALUE, and that is the whole design. Both are
#: spawned with `dict(os.environ)` — the environment IS inherited — and both
#: really load `tests/_state_guard.py`; the only difference between them is
#: whether this process's own entry in `PYTEST_PLUGINS` survived into the
#: child. One is seen and one is not, so the report itself draws the boundary
#: rather than a sentence claiming where it is.
#:
#: The appended child is the POSITIVE CONTROL and it is not decoration: without
#: it, "the replaced child is unseen" is satisfied just as well by an
#: instrument that sees no child at all, which is exactly how the reach came to
#: be declared wider than it was.
_ONE_VARIABLE_TWO_CHILDREN = r'''import os
import subprocess
import sys

#: Leaks a watched key, so the guard — if it is loaded there — fails the test
#: at its own teardown and says so. That report is the only acceptable
#: evidence that the route reached: a limit pinned on the instrument's
#: silence alone can be satisfied by a route that never arrived.
LEAKS = "import os\n\n\ndef test_leaks():\n    os.environ['STELLING_X'] = '1'\n"


def _child(tmp_path, name, plugins):
    path = tmp_path / name
    path.write_text(LEAKS, encoding="utf-8")
    env = dict(os.environ)
    env["PYTEST_PLUGINS"] = plugins
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "no:randomly", str(path)],
        capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    assert proc.returncode == 1, name + "\n" + proc.stdout + proc.stderr
    assert "changed process-global state" in proc.stdout, name + "\n" + proc.stdout


def test_middle(tmp_path):
    mine = os.environ["PYTEST_PLUGINS"]
    # REPLACED: the idiomatic spelling, and the one the audit drove.
    _child(tmp_path, "test_child_replaced.py", "_state_guard")
    # APPENDED: the same act of loading a plugin through the same variable,
    # with this process's entry left in front of it.
    _child(tmp_path, "test_child_appended.py", mine + ",_state_guard")
'''


def test_a_CHILD_WHOSE_PYTEST_PLUGINS_IS_REPLACED_loads_it_UNSEEN(tmp_path):
    """The child-process limit at its exact boundary: one variable, two children.

    **WHAT THIS REPLACED A WIDER SENTENCE WITH.** The limits list above used to
    say a child is reached "only if it inherits this process's ENVIRONMENT". A
    second blinded audit spawned `{**os.environ, "PYTEST_PLUGINS":
    "_state_guard"}`, which inherits the environment and is unseen — and which
    is the most ordinary way there is to load a pytest plugin into a child from
    an environment, in a file whose subject is a pytest plugin. The reach is
    the entry, not the environment, and this is what holds that sentence to its
    width.

    Both children are spawned from `dict(os.environ)` and both are shown to
    load the guard by the guard's own report. Exactly one is seen. If the
    replaced child ever becomes visible the limit has been outgrown and the
    bullet should be struck; if the appended child ever stops being visible,
    the instrument has narrowed and the limit is being satisfied by blindness
    rather than by a boundary — which is the failure the positive control is
    here to refuse.
    """
    proc, seen = _observe_plant(tmp_path, _ONE_VARIABLE_TWO_CHILDREN)
    assert proc.returncode == 0, (
        "one of the two children did not load the guard, so this compares "
        f"nothing\n{proc.stdout}\n{proc.stderr}\n"
        "A limit whose route has quietly stopped reaching is a limit nobody "
        "is checking."
    )
    reported = {(row["pid"], row["session"]) for row in seen.loaded}
    named = [
        row
        for row in seen.rows
        if row["kind"] == "session" and (row["pid"], row["session"]) in reported
    ]
    assert len(seen.loaded) == 1, (
        "the two children differ in one environment value and should differ "
        f"in exactly one row of this report:\n{seen}\n"
        "Both seen means the replaced value no longer hides a child — strike "
        "the bullet above and this fixture. Neither seen means the positive "
        "control has stopped working, and the limit beside it is then held up "
        "by an instrument that sees nothing."
    )
    assert named and "test_child_appended.py" in " ".join(named[0]["args"]), (
        "the child that was seen is not the one whose PYTEST_PLUGINS kept "
        f"this process's entry:\n{seen}"
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
