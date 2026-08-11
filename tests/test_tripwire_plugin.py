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

# The always-registered plugin, spelled the way `-p` wants it. In an installed
# stelling this arrives through the `pytest11` entry point in pyproject.toml;
# a nested pytester session inherits neither the entry point nor the parent's
# plugins, so it is named explicitly.
PLUGIN = "stelling._tripwire.plugin"
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
    yield
    from stelling import _tripwire

    _tripwire.disarm()


def _run(pytester, *args):
    return pytester.runpytest("-p", PLUGIN, "-p", "no:cacheprovider", *args)


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
    assert "you wrote 300" in out
    assert "int8 holds that as 44" in out


def test_dash_p_switches_it_on_too(pytester):
    pytester.makepyfile(WRAPPING_TEST)
    result = _run(pytester, "-p", OPT_IN)
    result.assert_outcomes(passed=1)
    assert "you wrote 300" in result.stdout.str()


def test_the_flag_switches_it_on_without_the_module(pytester):
    pytester.makepyfile(WRAPPING_TEST)
    result = _run(pytester, "--stelling-overflow=auto")
    result.assert_outcomes(passed=1)
    assert "you wrote 300" in result.stdout.str()


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
    assert "you wrote 300" in result.stdout.str()


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
    assert "you wrote 300" in result.stdout.str()


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
    assert "no out-of-range integer narrowings in your own code" in out
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
            "-p", PLUGIN, "-p", "no:cacheprovider", "--stelling-overflow=auto",
            f"{name}::test_a", f"{name}::test_b", f"{name}::test_c",
        )
    )
    reverse = section(
        pytester.runpytest_subprocess(
            "-p", PLUGIN, "-p", "no:cacheprovider", "--stelling-overflow=auto",
            f"{name}::test_c", f"{name}::test_b", f"{name}::test_a",
        )
    )
    joined = "\n".join(forward)
    assert "[3]" in joined and "you wrote 300" in joined, joined
    assert forward == reverse
