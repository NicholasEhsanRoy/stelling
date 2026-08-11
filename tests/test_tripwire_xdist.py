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
from conftest import tripwire_plugin_args  # noqa: E402

PLUGIN_ARGS = tripwire_plugin_args()

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
        line.split("you wrote ")[1].split(";")[0]
        for line in result.stdout.str().splitlines()
        if "you wrote " in line
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
