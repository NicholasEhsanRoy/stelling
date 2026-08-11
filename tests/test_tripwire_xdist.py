# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""xdist aggregation, driven across a real process boundary.

``runpytest_subprocess`` throughout, and that is forced rather than tidy: the
thing under test *is* a process boundary, so an in-process controller would be
testing a configuration no user runs.

Two things ``PLAN-tripwire.md`` §2 lists as ASSUMED are established here rather
than cited:

* that ``@pytest.hookimpl(tryfirst=True)`` on ``pytest_sessionfinish``
  reliably beats xdist's own sender. Precedent exists (pytest-cov does this);
  the payload arriving is the proof, since a worker whose hook ran after the
  sender would send an empty ``workeroutput``;
* that ``execnet`` serialises the payload. Restricted to primitives, and the
  findings coming back through it is what verifies that.

The hazard §6 names is silent undercount, not aggregation: a crashed worker
never populates ``workeroutput`` and summing what arrives is a confident wrong
total. The lost-worker test is what makes that visible rather than assumed.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

jax = pytest.importorskip("jax")
xdist = pytest.importorskip(
    "xdist", reason="needs pytest-xdist to drive a real worker split"
)

from conftest import TRIPWIRE_PLUGIN as PLUGIN  # noqa: E402
from conftest import tripwire_plugin_args, xdist_plugin_args  # noqa: E402

# xdist reaches a nested session by the same `pytest11` entry point the
# tripwire does, so `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` takes `-n 2` away from
# these sessions as well -- and every one of them then measures a one-process
# run while claiming to measure two workers.
PLUGIN_ARGS = tripwire_plugin_args() + xdist_plugin_args()

TWO_FILES = {
    "test_alpha": """
        import jax
        import jax.numpy as jnp

        def alpha(x):
            return x + 300

        def test_alpha():
            jax.make_jaxpr(alpha)(jnp.zeros((31,), jnp.int8))
    """,
    "test_beta": """
        import jax
        import jax.numpy as jnp

        def beta(x):
            return x + 400

        def test_beta():
            jax.make_jaxpr(beta)(jnp.zeros((32,), jnp.int8))
    """,
}


def _run(pytester, *args):
    return pytester.runpytest_subprocess(
        *PLUGIN_ARGS, "-p", "no:cacheprovider", "--stelling-overflow=auto", *args
    )


def _findings(result) -> set[str]:
    """The written values the report named, read back out of the text."""
    return {
        line.split("written there is ")[1].split(";")[0]
        for line in result.stdout.str().splitlines()
        if "written there is " in line
    }


def test_two_workers_report_the_true_total(pytester):
    """§9's xdist row, with its control.

    The control is not decoration: a controller that aggregated nothing and a
    controller that aggregated correctly both print *a* report, and only
    comparing against the single-process run says which happened.
    """
    pytester.makepyfile(**TWO_FILES)

    split = _run(pytester, "-n", "2")
    split.assert_outcomes(passed=2)
    single = _run(pytester, "-p", "no:xdist")
    single.assert_outcomes(passed=2)

    assert _findings(single) == {"300", "400"}, single.stdout.str()
    assert _findings(split) == _findings(single), (
        "the two-worker run and the single-process run disagree about what "
        f"was found: {_findings(split)} vs {_findings(single)}"
    )
    out = split.stdout.str()
    assert "2 of 2 workers reported" in out
    assert "worker(s) armed" in out
    assert "The controller runs no tests" in out


def test_the_controller_is_not_instrumented(pytester):
    """§6: with ``-n auto`` the controller runs no tests, so arming there
    would import jax into a process that never traces — and would report a
    denominator built out of nothing."""
    pytester.makepyfile(**TWO_FILES)
    result = _run(pytester, "-n", "2")
    out = result.stdout.str()
    assert "The controller runs no tests and is deliberately not instrumented" in out
    denominator = [ln for ln in out.splitlines() if ln.startswith("denominator:")]
    assert len(denominator) == 1, denominator
    assert " 0 integer const-folds" not in denominator[0], (
        "the aggregated denominator is zero: nothing crossed the boundary, so "
        f"the findings above would be a coincidence. {denominator}"
    )


def test_a_lost_worker_is_visible_rather_than_a_quiet_total(pytester):
    """The hazard, driven. One worker dies without running its
    ``pytest_sessionfinish``; the controller must say it is short a worker
    instead of presenting a partial as a total."""
    pytester.makepyfile(
        test_alpha=TWO_FILES["test_alpha"],
        test_beta=TWO_FILES["test_beta"],
        test_gamma="""
            import os
            import signal

            def test_dies():
                os.kill(os.getpid(), signal.SIGKILL)
        """,
    )
    result = _run(pytester, "-n", "3")
    out = result.stdout.str()
    assert "LOST WORKERS" in out, out[-4000:]
    assert "PARTIAL, not a total" in out


def test_the_payload_is_primitives_and_survives_execnet(pytester):
    """§2's second assumption. A finding that crossed carries every field it
    started with, not just a count — the chain, the dtypes and the recomputed
    value all have to arrive or the report on the controller is a stub."""
    pytester.makepyfile(**TWO_FILES)
    result = _run(pytester, "-n", "2")
    out = result.stdout.str()
    assert "300 mod 2**8 = 44" in out  # the arithmetic, recomputed on the controller
    assert "CONFIRMED recomputed from (300, int8) without the hook: 44" in out
    assert "int8 holds that as 44" in out
    assert "test_alpha.py:" in out and "test_beta.py:" in out


def test_ONE_broken_worker_does_not_discard_what_the_OTHER_one_found(pytester):
    """The controller's status is its workers' agreement, so one broken worker
    of two makes it ``mixed`` — and a non-armed status used to return before
    the denominator, throwing away every finding the armed worker had
    serialised back, with no count and no mention of the loss.

    The existing xdist tests could not see it: they break the anchor in EVERY
    worker or in none, and both of those agree.

    The control is the same run with both workers healthy: a report that
    printed nothing either way would satisfy half of this, and one that lost
    nothing because nothing was ever found would satisfy the other.
    """
    pytester.makepyfile(**TWO_FILES)
    pytester.makeconftest(
        """
        def pytest_configure(config):
            # exactly ONE worker of the two, so they disagree
            if getattr(config, "workerinput", {}).get("workerid") == "gw0":
                from stelling._tripwire import _adapter_jax as adapter

                adapter.detach("entry")
        """
    )
    half = _run(pytester, "-n", "2")
    half.assert_outcomes(passed=2)
    out = half.stdout.str()

    assert "NOT ARMED [mixed]" in out
    assert "no-entry" in out, "the broken worker's code is not disclosed"
    assert _findings(half), (
        "the healthy worker's findings were discarded because the CONTROLLER "
        f"was not armed. {out[-3000:]}"
    )
    assert _findings(half) < {"300", "400"}, (
        "a broken worker found something, so the anchor was not broken"
    )
    denominator = [ln for ln in out.splitlines() if ln.startswith("denominator:")]
    assert len(denominator) == 1 and " 0 integer const-folds" not in denominator[0]
    assert "PARTIAL" in out and "not a total" in out, (
        "a partial was presented without saying it is one"
    )


def test_what_actually_delivers_the_payload_is_XDIST_S_HOOKWRAPPER():
    """§2's first assumed item, and it was assumed in the wrong place.

    ``plugin.py`` said ``tryfirst=True`` on ``pytest_sessionfinish`` was "the
    ordering claim, measured rather than cited", and pointed at the tests
    above. They do not measure it: they assert the payloads ARRIVED, and the
    payloads arrive under ``trylast`` as well — driven, the whole tripwire
    suite passes with the flag flipped.

    The real reason is upstream: xdist's own
    ``WorkerInteractor.pytest_sessionfinish`` is a **hookwrapper**, so it
    yields to every other implementation regardless of ordering and sends
    ``workeroutput`` afterwards. That is what this pins. If xdist ever ships a
    plain implementation instead, ``tryfirst`` becomes load-bearing, the
    ordering needs measuring for real, and this fails first.
    """
    from xdist.remote import WorkerInteractor

    opts = getattr(WorkerInteractor.pytest_sessionfinish, "pytest_impl", None)
    assert opts is not None, (
        "xdist's sessionfinish is no longer a declared hookimpl, so the "
        "ordering assumption this plugin rests on has to be re-measured"
    )
    assert opts.get("hookwrapper") or opts.get("wrapper"), (
        "xdist's WorkerInteractor.pytest_sessionfinish is no longer a wrapper "
        f"({opts}). Being one is what let every other implementation run "
        "before the payload was sent, whatever its ordering. `tryfirst=True` "
        "in stelling's plugin is now load-bearing and untested."
    )


def test_require_fails_the_session_when_a_worker_cannot_arm(pytester):
    """The controller cannot raise ``UsageError`` at configure time, because it
    is not the process that arms. So ``require`` under xdist has to escalate
    from what the workers reported, and this is the only place that path runs.

    The anchor is broken inside each WORKER via a conftest, which is the only
    way to reach a process the parent does not own. It goes through the
    adapter's API — rule 2 covers ``tests/`` with no exemption, and that
    includes a conftest this file writes.
    """
    pytester.makepyfile(**TWO_FILES)
    pytester.makeconftest(
        """
        def pytest_configure(config):
            if hasattr(config, "workerinput"):
                from stelling._tripwire import _adapter_jax as adapter

                adapter.detach("entry")
        """
    )
    broken = pytester.runpytest_subprocess(
        *PLUGIN_ARGS, "-p", "no:cacheprovider",
        "--stelling-overflow=require", "-n", "2",
    )
    assert broken.ret != 0, (
        "require under xdist did not fail although no worker could arm"
    )
    assert "no-entry" in broken.stdout.str()
