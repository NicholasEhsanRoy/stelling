# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The pytest plugin, driven through real nested sessions.

Three of the four guardrails are untestable without one: "the suite still
exits 0", "the summary names the code" and "``-W error`` does not turn a
disabled tripwire into a crash" are all properties of a SESSION, and a unit
test on ``arm()`` cannot see any of them.

``pytester`` rather than the deprecated ``testdir``. Each of these is a real
pytest session and the budget is real, so they are few and deliberate.

**The opt-in is the thing being tested most often here**, because it is the
one property a user who never asked for this depends on: a plain
``pip install stelling`` must leave their suite exactly as it was.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

# The plugin itself needs no jax to be registered or to stay switched off, and
# that direction is measured below (test_the_default_is_off_and_imports_no_jax
# runs in the zero-dep lane). The tests that need a live hook gate here.
jax = pytest.importorskip("jax")

from conftest import TRIPWIRE_PLUGIN as PLUGIN
from conftest import deterministic_order_args, tripwire_plugin_args

# `("-p", PLUGIN)` where the distribution is not installed, `()` where it is
# and the `pytest11` entry point has already registered the module. Adding
# both is a hard error -- see `conftest.tripwire_plugin_args`.
PLUGIN_ARGS = tripwire_plugin_args()
#: Every nested session in this file plants tests whose ORDER is the
#: property. See `deterministic_order_args` in tests/conftest.py.
ORDER_ARGS = deterministic_order_args()
OPT_IN = "stelling.overflow"

WRAPPING_TEST = """
    import jax
    import jax.numpy as jnp

    def quantize(x):
        return x + 300

    def test_wrap():
        jax.make_jaxpr(quantize)(jnp.zeros((11,), jnp.int8))
"""


@pytest.fixture(autouse=True)
def _isolate(pytester, monkeypatch):
    """Nested sessions arm the hook in THIS process, because ``runpytest`` is
    in-process by default. Leaving one armed would silently instrument the
    rest of the outer suite.

    ``PYTHONPATH`` is rewritten to an ABSOLUTE path to the tree this suite
    imported. A subprocess session runs with its cwd inside pytester's tmpdir,
    so a developer running with a relative ``PYTHONPATH=src`` gets
    ``No module named 'stelling._tripwire'`` there and nowhere else —
    measured. CI installs with ``-e`` and would never have seen it.
    """
    import os
    import pathlib as _pathlib

    import stelling

    src = str(_pathlib.Path(stelling.__file__).resolve().parents[1])
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([src, existing]) if existing else src
    )
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    from stelling._tripwire import eager as _eager

    eager_was_armed = _eager.is_armed()
    yield
    from stelling import _tripwire

    _tripwire.disarm()
    # BOTH HOOKS. `runpytest` is in-process by default, so a nested session
    # that arms the EAGER detector arms it here -- and leaving it armed would
    # turn every later test in the outer suite into a test of it. Restored
    # rather than disarmed, for the same reason the tripwire's own fixtures
    # are: this suite is meant to be runnable with the detector armed for the
    # whole session, and unconditionally disarming would take that out.
    if _eager.is_armed() != eager_was_armed:
        if eager_was_armed:
            _tripwire.arm_eager()
        else:
            _tripwire.disarm_eager()
    _eager.reset_counters()


def _run(pytester, *args):
    return pytester.runpytest(*PLUGIN_ARGS, *ORDER_ARGS, "-p", "no:cacheprovider", *args)


# --- the opt-in -------------------------------------------------------------


def test_registered_and_switched_off_does_nothing_at_all(pytester):
    """The property a user who installed stelling for the verifier depends on.

    The plugin is loaded — ``--stelling-overflow`` is in ``--help``, which is
    the positive control that it really is registered — and the session is
    untouched: no report section, and the hook is not armed.
    """
    pytester.makepyfile(WRAPPING_TEST)
    result = _run(pytester)
    result.assert_outcomes(passed=1)
    assert "stelling overflow tripwire" not in result.stdout.str()

    from stelling import _tripwire

    assert not _tripwire.is_armed(), "a switched-off plugin armed the hook"

    helped = _run(pytester, "--help")
    assert "--stelling-overflow" in helped.stdout.str(), (
        "the plugin was not registered at all, so the silence above measured "
        "nothing."
    )


def test_one_line_in_conftest_switches_it_on(pytester):
    """The adoption story, driven: ``pytest_plugins = ["stelling.overflow"]``."""
    pytester.makepyfile(WRAPPING_TEST)
    pytester.makeconftest(f'pytest_plugins = ["{OPT_IN}"]')
    result = _run(pytester)
    result.assert_outcomes(passed=1)
    out = result.stdout.str()
    assert "stelling overflow tripwire" in out
    assert "armed" in out
    assert "the constant written there is 300" in out
    assert "int8 holds that as 44" in out


def test_dash_p_switches_it_on_too(pytester):
    pytester.makepyfile(WRAPPING_TEST)
    result = _run(pytester, "-p", OPT_IN)
    result.assert_outcomes(passed=1)
    assert "the constant written there is 300" in result.stdout.str()


def test_the_flag_switches_it_on_without_the_module(pytester):
    pytester.makepyfile(WRAPPING_TEST)
    result = _run(pytester, "--stelling-overflow=auto")
    result.assert_outcomes(passed=1)
    assert "the constant written there is 300" in result.stdout.str()


def test_the_flag_wins_over_the_module_so_off_means_off(pytester):
    """A user who loaded the plugin in conftest and wants one run without it."""
    pytester.makepyfile(WRAPPING_TEST)
    pytester.makeconftest(f'pytest_plugins = ["{OPT_IN}"]')
    result = _run(pytester, "--stelling-overflow=off")
    result.assert_outcomes(passed=1)
    assert "stelling overflow tripwire" not in result.stdout.str()


# --- the fail-closed contract, as a session ---------------------------------


def test_a_broken_anchor_leaves_the_suite_green_and_names_the_code(pytester):
    """§9 row 1 and §10 criterion 3, at session level.

    The registry entry is removed before the nested session configures, so
    that session's ``arm()`` meets a jax whose const-fold rule is not where
    the tripwire expects it. Exit code unaffected, code named in the summary,
    no crash.
    """
    from stelling._tripwire import _adapter_jax as adapter

    pytester.makepyfile(WRAPPING_TEST)
    assert adapter.detach("entry") == "detached"
    try:
        result = _run(pytester, "--stelling-overflow=auto")
    finally:
        adapter.reattach()
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    out = result.stdout.str()
    assert "NOT ARMED [no-entry]" in out
    assert "Static checking is unaffected" in out


def test_a_disabled_tripwire_does_not_crash_under_dash_W_error(pytester):
    """§9 row 2, and the whole reason the primary channel is the reporter.

    Under ``-W error::UserWarning`` a "safely disabled" warning becomes an
    exception and takes down the suite this guardrail exists to protect. The
    reporter cannot be escalated, so the session still exits 0.
    """
    from stelling._tripwire import _adapter_jax as adapter

    pytester.makepyfile(WRAPPING_TEST)
    assert adapter.detach("entry") == "detached"
    try:
        result = _run(pytester, "--stelling-overflow=auto", "-W", "error::UserWarning")
    finally:
        adapter.reattach()
    assert result.ret == 0, "a disabled tripwire crashed a -W error session"
    assert "NOT ARMED [no-entry]" in result.stdout.str()


def test_an_armed_tripwire_does_not_crash_under_dash_W_error_either(pytester):
    """The other half. A finding must not travel by a channel ``-W error``
    can turn into an exception, or the tool breaks suites when it WORKS."""
    pytester.makepyfile(WRAPPING_TEST)
    result = _run(pytester, "--stelling-overflow=auto", "-W", "error::UserWarning")
    assert result.ret == 0
    assert "the constant written there is 300" in result.stdout.str()


def test_require_plus_a_broken_anchor_fails_and_names_the_code(pytester):
    """§9 row 3. The escalation belongs to the user, and this is them taking it."""
    from stelling._tripwire import _adapter_jax as adapter

    pytester.makepyfile(WRAPPING_TEST)
    assert adapter.detach("entry") == "detached"
    try:
        result = _run(pytester, "--stelling-overflow=require")
    finally:
        adapter.reattach()
    assert result.ret != 0, "require did not fail on a tripwire that could not arm"
    assert "no-entry" in result.stdout.str() + result.stderr.str()


def test_require_with_a_working_anchor_is_green(pytester):
    """The positive control for the test above: ``require`` is not simply
    always red."""
    pytester.makepyfile(WRAPPING_TEST)
    result = _run(pytester, "--stelling-overflow=require")
    assert result.ret == 0
    assert "the constant written there is 300" in result.stdout.str()


# --- staying armed, which is a different property from arming ---------------

DETACHES_MIDWAY = """
    import jax
    import jax.numpy as jnp

    def test_1_before():
        jax.make_jaxpr(lambda a: a + 400)(jnp.zeros((41,), jnp.int8))

    def test_2_detach():
        from stelling import _tripwire
        assert _tripwire.disarm() == "restored"

    def test_3_after_and_invisible():
        jax.make_jaxpr(lambda a: a + 500)(jnp.zeros((42,), jnp.int8))
"""


def test_a_hook_that_LEFT_does_not_still_report_armed(pytester):
    """The status line and the denominator are claims about the whole session
    and were both fixed at ``pytest_configure``, with nothing re-checking
    them.

    So a session whose hook came out of the registry half way through printed
    ``armed`` over a denominator that had stopped growing, and
    ``--stelling-overflow=require`` — the mode a user picks precisely because
    they depend on the tripwire — exited **0** while never seeing the
    ``x + 500`` that ran after the detachment. `is_armed()` existed, cost
    nothing, and was consulted nowhere.

    This is the feature's own thesis inverted: *a zero with a dead instrument
    is the failure this project keeps finding*.
    """
    pytester.makepyfile(DETACHES_MIDWAY)
    result = _run(pytester, "--stelling-overflow=require")
    result.assert_outcomes(passed=3)
    out = result.stdout.str()

    assert "NOT ARMED [detached]" in out, out[-3000:]
    assert result.ret != 0, "require passed a session that did not stay armed"
    # what it DID see is still reported, and it says what it is
    assert "the constant written there is 400" in out
    assert "PARTIAL" in out and "not a total" in out
    # and what it did not see is not silently absent from a confident total
    assert "the constant written there is 500" not in out


def test_the_same_session_with_the_hook_LEFT_ALONE_is_green(pytester):
    """The control for the test above, and it is not decoration: a check that
    called every session detached would satisfy the assertions there.

    Same three tests, same ``require``, without the ``disarm()``."""
    pytester.makepyfile(DETACHES_MIDWAY.replace('_tripwire.disarm() == "restored"', "True"))
    result = _run(pytester, "--stelling-overflow=require")
    assert result.ret == 0
    out = result.stdout.str()
    assert "NOT ARMED" not in out and "PARTIAL" not in out
    assert "the constant written there is 400" in out and "the constant written there is 500" in out


def test_a_registry_rebind_surfaces_as_foreign_patch_IN_THE_REPORT(pytester):
    """``foreign-patch`` is advertised in ``docs/overflow-tripwire.md`` as a
    stable, greppable code, and it could not appear in any report: ``arm()``
    has no route that returns it, and its only surfacing was a note appended
    in ``pytest_unconfigure``, which runs AFTER the summary is written.

    It is reachable now because the same end-of-session check that catches a
    detachment distinguishes the two ways of not being armed — we hold no
    installation (``detached``) against we hold one and the live entry is not
    ours (``foreign-patch``). They are different things to tell a user.

    A SUBPROCESS SESSION, and that is forced rather than tidy. The contract
    for a foreign patch is that the tripwire does NOT clobber it, so an
    in-process run leaves somebody else's wrapper in this interpreter's
    registry for the rest of the outer suite — measured, it moves
    ``registry_size`` and ``rule_hash`` and fails two tests in
    ``test_tripwire_arm.py`` that pass in isolation.
    """
    pytester.makepyfile(
        """
        import jax
        import jax.numpy as jnp

        def test_1_fires():
            jax.make_jaxpr(lambda a: a + 400)(jnp.zeros((43,), jnp.int8))

        def test_2_someone_else_patches_over_us():
            from stelling._tripwire import _adapter_jax as adapter
            reg = adapter._installed["registry"]
            prim = adapter._installed["primitive"]
            original = adapter._installed["original"]
            def somebody_elses_wrapper(*a, **k):
                return original(*a, **k)
            reg[prim] = somebody_elses_wrapper
        """
    )
    result = pytester.runpytest_subprocess(
        *PLUGIN_ARGS, *ORDER_ARGS, "-p", "no:cacheprovider", "--stelling-overflow=require"
    )
    result.assert_outcomes(passed=2)
    out = result.stdout.str()
    assert "NOT ARMED [foreign-patch]" in out, out[-3000:]
    assert "left in place rather than clobbered" in out
    assert result.ret != 0
    assert "the constant written there is 400" in out and "PARTIAL" in out


# --- the report, as a session ----------------------------------------------


def test_a_clean_session_still_reports_its_denominator(pytester):
    """§9's last row, and §8's first absolute: a run with zero findings must
    still show a non-zero invocation count, or a dead hook and a clean suite
    print the same thing."""
    pytester.makepyfile(
        """
        import jax
        import jax.numpy as jnp

        def test_clean():
            jax.make_jaxpr(lambda a: a + 3)(jnp.zeros((13,), jnp.int8))
        """
    )
    result = _run(pytester, "--stelling-overflow=auto")
    result.assert_outcomes(passed=1)
    out = result.stdout.str()
    assert "no out-of-range integer narrowings outside jax" in out
    assert "ZERO invocations" not in out, (
        "the nested session reported a zero denominator: the hook was not "
        "live, so its clean report measured nothing."
    )
    denominator = [line for line in out.splitlines() if line.startswith("denominator:")]
    assert len(denominator) == 1, denominator
    assert " 0 integer const-folds" not in denominator[0], denominator


def test_the_report_does_not_depend_on_the_order_the_findings_fired(pytester):
    """§10a.7. Findings fire once per TRACE and trace order varies between
    runs; a report that changed with it would look unreliable however right it
    was.

    Driven by running the same three tests in the opposite order, which is a
    real reordering of emission and not a re-run. **Subprocess sessions**, and
    that is forced rather than chosen: jax's trace cache is process-wide, so a
    second in-process session tracing the same functions at the same avals
    would reach the const-fold site zero times and both reports would agree by
    being empty.
    """
    path = pytester.makepyfile(
        """
        import jax
        import jax.numpy as jnp

        def a(x): return x + 300
        def b(x): return x + 400
        def c(x): return x + 500

        def test_a(): jax.make_jaxpr(a)(jnp.zeros((21,), jnp.int8))
        def test_b(): jax.make_jaxpr(b)(jnp.zeros((22,), jnp.int8))
        def test_c(): jax.make_jaxpr(c)(jnp.zeros((23,), jnp.int8))
        """
    )
    name = path.name

    def section(result):
        """The tripwire's report and nothing after it.

        Cut at pytest's own summary line, which carries a WALL CLOCK — two
        runs differ at `3 passed in 0.07s` vs `0.09s` and the comparison
        below would be about the runner's speed rather than about the
        report. Measured, on the first run of this test.
        """
        lines = result.stdout.str().splitlines()
        start = next(i for i, line in enumerate(lines) if "overflow tripwire" in line)
        body = []
        for line in lines[start:]:
            if " passed in " in line or " warning" in line:
                break
            body.append(line)
        return body

    forward = section(
        pytester.runpytest_subprocess(
            *PLUGIN_ARGS, *ORDER_ARGS, "-p", "no:cacheprovider", "--stelling-overflow=auto",
            f"{name}::test_a", f"{name}::test_b", f"{name}::test_c",
        )
    )
    reverse = section(
        pytester.runpytest_subprocess(
            *PLUGIN_ARGS, *ORDER_ARGS, "-p", "no:cacheprovider", "--stelling-overflow=auto",
            f"{name}::test_c", f"{name}::test_b", f"{name}::test_a",
        )
    )
    joined = "\n".join(forward)
    assert "[3]" in joined and "the constant written there is 300" in joined, joined
    assert forward == reverse


def test_the_entry_point_declaration_is_what_makes_any_of_this_reachable():
    """Everything above runs with ``-p stelling._tripwire.plugin`` spelled out
    WHERE THE DISTRIBUTION IS NOT INSTALLED, and with nothing extra where it
    is — this said "a nested pytester session inherits no entry points" and
    ``tests/conftest.py::tripwire_plugin_args`` records the measured opposite,
    which is the reason that helper exists. Either way none of those runs can
    see the one declaration the whole adoption story rests on: a
    ``pytest11`` entry point that a rename, a typo or a deleted section would
    silently unregister, leaving `pytest_plugins = ["stelling.overflow"]` a
    line that does nothing at all.

    Read out of ``pyproject.toml`` and resolved against the module, rather
    than restated: the module named must import, and must define
    ``pytest_addoption``, or it is registered and inert.
    """
    import importlib
    import pathlib
    import re

    repo = pathlib.Path(__file__).resolve().parents[1]
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(
        r"^\[project\.entry-points\.pytest11\]\s*$(.*?)(?=^\[|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert section, (
        "pyproject.toml has no [project.entry-points.pytest11] section. "
        "Without it `pip install stelling` registers no plugin, "
        "`--stelling-overflow` does not exist, and the one-line conftest "
        "snippet in docs/overflow-tripwire.md is inert."
    )
    targets = dict(re.findall(r'^\s*(\w+)\s*=\s*"([^"]+)"', section.group(1), re.MULTILINE))
    assert targets, section.group(1)
    assert PLUGIN in targets.values(), (
        f"the pytest11 entry points are {sorted(targets.values())}; the plugin "
        f"module is {PLUGIN}"
    )
    for module_name in targets.values():
        module = importlib.import_module(module_name)
        assert hasattr(module, "pytest_addoption"), (
            f"{module_name} is registered as a pytest plugin and defines no "
            "hooks, so it is registered and inert"
        )
    # ...and the opt-in module the docs tell people to name really exists and
    # really is the one the plugin checks for
    optin = importlib.import_module(OPT_IN)
    assert optin.OPT_IN_PLUGIN == OPT_IN, (
        f"{OPT_IN} and the name the plugin looks for have drifted apart: "
        f"{optin.OPT_IN_PLUGIN!r}"
    )
    # ...and the NAME of the entry point, which the opt-in module registers
    # under by hand when autoload is off. It has to be this name and not the
    # dotted one, because `-p` is consumed before the entry point loader runs
    # and the loader's own guard is `get_plugin(ep.name)`.
    assert optin.ENTRY_POINT_NAME in targets, (
        f"the pytest11 entry point names are {sorted(targets)}; "
        f"stelling.overflow registers under {optin.ENTRY_POINT_NAME!r}, so "
        "with autoload ON and `-p stelling.overflow` the loader would "
        "register the same module a second time and raise ValueError before "
        "a single test collects"
    )


def test_the_documented_spellings_work_WITHOUT_the_entry_point(pytester, monkeypatch):
    """``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` is a common CI hygiene setting, and
    it turns off the ``pytest11`` entry point that is the only thing making the
    plugin reachable.

    Measured before this: **every documented switch-on spelling was inert,
    with no error and no warning.** ``pytest_plugins = ["stelling.overflow"]``
    was silently green, ``-p stelling.overflow`` was silently green, and only
    the undocumented ``-p stelling._tripwire.plugin`` worked. Nothing in
    ``README.md`` or ``docs/`` mentioned the dependency. A silent no-op is the
    worst available behaviour for a tool whose subject is instruments that are
    not running.

    Subprocess sessions: the environment variable is read once, in the child's
    own ``Config._preparse``, and an in-process nested session would inherit
    this process's already-loaded plugins and measure nothing.
    """
    pytester.makepyfile(WRAPPING_TEST)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    result = pytester.runpytest_subprocess(*ORDER_ARGS, "-p", "no:cacheprovider", "-p", OPT_IN)
    result.assert_outcomes(passed=1)
    assert "the constant written there is 300" in result.stdout.str(), (
        f"-p {OPT_IN} is a documented spelling and it did nothing at all "
        "with plugin autoload disabled"
    )

    pytester.makeconftest(f'pytest_plugins = ["{OPT_IN}"]')
    result = pytester.runpytest_subprocess(*ORDER_ARGS, "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    assert "the constant written there is 300" in result.stdout.str(), (
        "the one line in conftest.py that the docs open with did nothing at "
        "all with plugin autoload disabled"
    )

    # THE THIRD SPELLING CANNOT BE MADE TO WORK, and what matters is that it
    # fails LOUDLY. `--stelling-overflow` is registered BY the plugin, so with
    # no entry point and nothing naming a module there is no such flag; pytest
    # exits 4 with "unrecognized arguments". That is the acceptable end of
    # this — a silent green is not.
    pytester.makeconftest("")
    flag_only = pytester.runpytest_subprocess(
        *ORDER_ARGS, "-p", "no:cacheprovider", "--stelling-overflow=require"
    )
    assert flag_only.ret != 0
    assert "unrecognized arguments: --stelling-overflow" in (
        flag_only.stdout.str() + flag_only.stderr.str()
    )

    # THE CONTROL, and it is the half that a "just always register it" repair
    # would fail: with autoload disabled and no opt-in, the tripwire stays off
    # rather than arming because this module happened to get imported.
    quiet = pytester.runpytest_subprocess(*ORDER_ARGS, "-p", "no:cacheprovider")
    assert "stelling overflow tripwire" not in quiet.stdout.str()


@pytest.mark.parametrize("autoload", ["on", "off"])
def test_naming_the_opt_in_module_is_never_a_DOUBLE_registration(
    pytester, monkeypatch, autoload
):
    """The other direction, and it is not hypothetical: registering the same
    module object under two names raises ``ValueError: Plugin already
    registered under a different name``, which is an INTERNALERROR before a
    single test collects — strictly worse than the silent no-op it replaces.

    Two ways to hit it, and the repair needs a different guard for each:

    * the entry point got there first (autoload on, the module named in a
      ``conftest.py``) — ``pluginmanager.is_registered`` is the guard;
    * WE get there first (autoload on, ``-p stelling.overflow``), because
      ``-p`` is consumed in ``Config._preparse`` *before*
      ``load_setuptools_entrypoints`` — registering under the ENTRY POINT's
      name is the guard, since the loader's own check is
      ``get_plugin(ep.name)``. Measured: the dotted name fails here.
    """
    if autoload == "off":
        monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    else:
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)

    pytester.makepyfile(WRAPPING_TEST)
    pytester.makeconftest(f'pytest_plugins = ["{OPT_IN}"]')
    for extra in (("-p", OPT_IN), ()):
        result = pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", *extra, "--stelling-overflow=auto"
        )
        both = result.stdout.str() + result.stderr.str()
        assert "INTERNALERROR" not in both, both[-2500:]
        assert "already registered under a different name" not in both
        result.assert_outcomes(passed=1)
        assert "the constant written there is 300" in result.stdout.str()


# --- the eager construction-site detector's dial -----------------------------
#
# THESE MEASURE A PROCESS WITH NOTHING ARMED, and `pytester.runpytest` is
# in-process by default, so "nothing armed" has to be arranged rather than
# assumed: a session run with `--stelling-eager-truncation=error` arms the
# detector HERE, and a nested session's wrapping construction would then be
# refused by the outer process's hook rather than by the one under test.
# Measured, running this suite with that flag: three of the tests below failed
# for that reason and for no other.


@pytest.fixture
def outer_detector_off():
    """The detector off for this test, and back as it was afterwards."""
    from stelling import _tripwire
    from stelling._tripwire import eager as _eager

    was_armed = _eager.is_armed()
    if was_armed:
        _tripwire.disarm_eager()
    try:
        yield
    finally:
        if was_armed:
            _tripwire.arm_eager()


WRAPPING_CONSTRUCTION = """
    import jax.numpy as jnp

    def build():
        return jnp.full((4,), 300, jnp.int8)

    def test_construct():
        assert int(build().ravel()[0]) == 44
"""

DECLARED_CONSTRUCTION = """
    import jax.numpy as jnp
    from stelling import intentional_wrap

    def build():
        return jnp.full((4,), intentional_wrap(300, "int8"), jnp.int8)

    def test_construct():
        assert int(build().ravel()[0]) == 44
"""


def test_the_eager_dial_is_registered_always_and_off_always(
    pytester, outer_detector_off
):
    """The property a user who never asked for this depends on, for the
    SECOND instrument now as well as the first.

    The flag is in ``--help`` — the positive control that the plugin really
    is loaded — and a session that does not pass it constructs a wrapping
    array, passes, prints no eager section, and leaves the private jax
    attribute untouched.
    """
    from stelling._tripwire import eager

    pytester.makepyfile(WRAPPING_CONSTRUCTION)
    assert "--stelling-eager-truncation" in _run(pytester, "--help").stdout.str()
    result = _run(pytester)
    result.assert_outcomes(passed=1)
    assert "stelling eager truncation detector" not in result.stdout.str()
    assert not eager.is_armed()


def test_switching_the_tripwire_on_does_NOT_switch_the_eager_detector_on(
    pytester, outer_detector_off,
):
    """Two dials, and neither implies the other.

    The tripwire is a report and can be armed on a suite that contains
    undeclared truncations; the eager detector is a rule. Turning on the
    first must not turn on the second, or a user who wanted a report gets a
    suite that stops.
    """
    from stelling._tripwire import eager

    pytester.makepyfile(WRAPPING_CONSTRUCTION)
    result = _run(pytester, "--stelling-overflow=auto")
    result.assert_outcomes(passed=1)
    assert "stelling overflow tripwire" in result.stdout.str()
    assert "stelling eager truncation detector" not in result.stdout.str()
    assert not eager.is_armed()


def test_the_eager_dial_turns_a_silent_wrap_into_a_stopped_session(pytester):
    """The whole feature, through a real session.

    The same test file passes with the dial off and does not pass with it on,
    and the alarm arrives in the report as itself: the written value, the
    dtype, what it became, and the line that wrote it.

    IT IS A FAILURE AND NOT AN ERROR, measured rather than assumed. An
    earlier version of this test asserted ``errors=1`` on the reasoning that
    a ``BaseException`` from a test body is not a failed assertion; pytest
    does not agree, and files a ``BaseException`` raised in a body exactly
    where it files any other. What the ``BaseException`` choice buys is which
    handlers cannot swallow the alarm, not how pytest categorises it.
    """
    pytester.makepyfile(WRAPPING_CONSTRUCTION)
    result = _run(pytester, "--stelling-eager-truncation=error")
    assert result.ret != 0
    result.assert_outcomes(failed=1)
    out = result.stdout.str()
    assert "stelling eager truncation detector" in out
    assert "300 was TRUNCATED to 44" in out
    assert "intentional_wrap(300, 'int8')" in out, (
        "the alarm does not tell the reader what to write instead"
    )


def test_a_DECLARED_wrap_passes_the_same_session_and_is_printed(pytester):
    """The declaration, end to end, and the report that discloses it.

    A premise nobody can see is indistinguishable from the silence the
    detector exists to end, so the section names the site and the arithmetic.
    """
    pytester.makepyfile(DECLARED_CONSTRUCTION)
    result = _run(pytester, "--stelling-eager-truncation=error")
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    out = result.stdout.str()
    assert "1 wrap(s) DECLARED" in out
    assert "300 -> 44 (int8)" in out


def test_the_eager_section_prints_its_denominator_with_nothing_to_report(
    pytester,
):
    """"0 truncations" is what a hook that was never called reports too."""
    pytester.makepyfile(
        """
        import jax.numpy as jnp

        def test_in_range():
            assert int(jnp.full((4,), 44, jnp.int8).ravel()[0]) == 44
        """
    )
    result = _run(pytester, "--stelling-eager-truncation=error")
    result.assert_outcomes(passed=1)
    out = result.stdout.str()
    assert "scalar integer conversion(s) observed" in out
    assert "of them were out of range" in out
    assert "what this detector does NOT see" in out


def test_a_session_that_asked_for_the_rule_and_cannot_get_it_FAILS(
    pytester, outer_detector_off
):
    """There is no degraded mode, and this is what that means in practice.

    The tripwire has a ``require`` spelling because it has a degraded mode
    worth having — an unarmed tripwire still lets a suite run and still says
    why it is not watching. This has none: a detector that could not attach
    is not a quieter rule, it is no rule, and a session that asked for one and
    silently did not get it is the false assurance the instrument exists to
    remove. So arming failure is fatal with no flag to ask for it.
    """
    from stelling._tripwire import _adapter_jax as adapter

    pytester.makepyfile(WRAPPING_CONSTRUCTION)
    assert adapter.eager_detach("signature") == "detached"
    try:
        result = _run(pytester, "--stelling-eager-truncation=error")
    finally:
        adapter.eager_reattach()
    assert result.ret != 0, "a session that could not attach the rule passed"
    assert "signature-drift" in result.stderr.str() + result.stdout.str()

    # the control: the same session on the same jax passes
    control = _run(pytester, "--stelling-eager-truncation=error")
    assert control.ret != 0  # the wrapping construction still stops it
    assert "signature-drift" not in control.stdout.str()


EAGER_DETACHES_MIDWAY = """
    import jax.numpy as jnp

    def test_1_before():
        assert int(jnp.full((2,), 5, jnp.int8).sum()) == 10

    def test_2_displace_the_hook():
        from stelling._tripwire import _adapter_jax as adapter
        module = adapter._eager_module()
        setattr(module, adapter.EAGER_ATTR, adapter._eager_installed["original"])

    def test_3_after_and_unwatched():
        assert int(jnp.full((2,), 300, jnp.int8).sum()) == 88
"""


@pytest.mark.parametrize(
    "overflow",
    [(), ("--stelling-overflow=off",), ("--stelling-overflow=auto",)],
    ids=["eager-alone", "overflow-off", "overflow-auto"],
)
def test_the_eager_ESCALATION_does_not_depend_on_the_OTHER_dial(pytester, overflow):
    """A rule that could not stay attached must fail the session, in every
    spelling of "the eager detector on".

    THE ESCALATION USED TO SIT BELOW ``if state.recorder is None: return``,
    which is the tripwire's guard, and ``state.recorder`` is None exactly when
    ``--stelling-overflow=off`` -- the spelling ``docs/overflow-tripwire.md``
    recommends for running this detector alone. Measured, with the hook
    displaced mid-session: eager-alone exited 0, ``--stelling-overflow=off``
    exited 0, and only ``--stelling-overflow=auto`` exited 1. The two
    instruments are two dials and neither may need the other switched on.
    """
    pytester.makepyfile(EAGER_DETACHES_MIDWAY)
    result = _run(pytester, "--stelling-eager-truncation=error", *overflow)
    out = result.stdout.str()
    assert result.ret != 0, (
        "the eager detector was displaced mid-session and the session passed:"
        f"\n{out[-3000:]}"
    )
    assert "foreign-patch" in out or "detached" in out, out[-3000:]


def test_the_same_session_with_the_eager_hook_LEFT_ALONE_is_green(pytester):
    """The control. A check that failed every session would satisfy the test
    above, so the same three tests without the displacement must pass."""
    pytester.makepyfile(
        EAGER_DETACHES_MIDWAY.replace(
            'setattr(module, adapter.EAGER_ATTR, adapter._eager_installed["original"])',
            "assert module is not None",
        ).replace("jnp.full((2,), 300, jnp.int8).sum()) == 88",
                  "jnp.full((2,), 3, jnp.int8).sum()) == 6")
    )
    result = _run(
        pytester, "--stelling-eager-truncation=error", "--stelling-overflow=off"
    )
    assert result.ret == 0, result.stdout.str()[-3000:]
    assert "NOT ARMED" not in result.stdout.str()


# --- the eager detector's xdist aggregation, without xdist --------------------
#
# `tests/test_tripwire_xdist.py` drives a real worker split and SKIPS in every
# environment this repository routinely measures in (no venv here carries
# `pytest-xdist`; only the `test-jax` CI lane does). The two pieces of
# controller logic that do not need a worker are the merge arithmetic and the
# agreement rule, and those are driven here so that they are covered
# somewhere that runs. This is not a substitute for that file and does not
# claim to be: what is untested here is that a payload CROSSES.


def test_two_workers_eager_snapshots_are_summed_and_their_sites_kept():
    """A sum, and the per-site rows survive it.

    The totals alone cannot answer the question the section exists for --
    WHERE was a wrap declared -- and two workers that each declared a wrap at
    the same line are two declarations at one site, not one.
    """
    from stelling._tripwire.plugin import _merge_eager

    a = {
        "conversions": 10, "truncations": 2, "internal_errors": 1,
        "suppressed_jax": 1,
        "declared": {"f.py:1": [1, "300 -> 44 (int8)"]},
        "permitted": {"g.py:9": [2, "because"]},
        "suppressed": {"y.py:3": [1, "4294967295 -> -1 (int32)"]},
    }
    b = {
        "conversions": 5, "truncations": 1, "internal_errors": 0,
        "suppressed_jax": 2,
        "declared": {"f.py:1": [3, "300 -> 44 (int8)"],
                     "h.py:7": [1, "255 -> -1 (int8)"]},
        "permitted": {},
        "suppressed": {"y.py:3": [2, "4294967295 -> -1 (int32)"]},
    }
    merged = _merge_eager(_merge_eager(None, a), b)
    assert merged["conversions"] == 15
    assert merged["truncations"] == 3
    assert merged["internal_errors"] == 1
    assert merged["declared"]["f.py:1"][0] == 4, "the same site did not sum"
    assert merged["declared"]["h.py:7"][0] == 1
    assert merged["permitted"]["g.py:9"] == [2, "because"]
    # EVERY KEY THE SNAPSHOT CARRIES HAS TO BE IN THE MERGE, or a worker's
    # figure is silently dropped on the controller. Asserted against the
    # snapshot itself rather than against a list typed here.
    assert merged["suppressed_jax"] == 3
    assert merged["suppressed"]["y.py:3"][0] == 3
    from stelling._tripwire import eager as _eager

    dropped = set(_eager.snapshot()) - set(merged)
    assert not dropped, (
        f"the merge drops {sorted(dropped)}, so an xdist controller prints a "
        "total that is missing a worker's figure"
    )


def test_the_controller_reports_its_workers_agreement_and_says_so_when_there_is_none():
    """The controller runs no tests, so it never arms and has nothing of its
    own to report. Its status is what its workers agreed on -- and "no worker
    reported" is NOT a clean run, which is the case a sum over an empty dict
    would have rendered as a confident zero."""
    from stelling._tripwire.plugin import _State, _eager_controller_status

    state = _State()
    assert _eager_controller_status(state).code == "no-worker-reported"

    state.eager_worker_statuses = {"gw0": "armed", "gw1": "armed"}
    agreed = _eager_controller_status(state)
    assert agreed.armed and "2 worker(s) armed" in agreed.detail

    state.eager_worker_statuses = {"gw0": "armed", "gw1": "no-site"}
    mixed = _eager_controller_status(state)
    assert mixed.code == "mixed"
    assert "armed" in mixed.detail and "no-site" in mixed.detail

    state.eager_worker_statuses = {"gw0": "no-site", "gw1": "no-site"}
    assert _eager_controller_status(state).code == "no-site"
