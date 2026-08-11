# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The pytest entry point.

**Registered always, active never — until you switch it on.** A user who
installed stelling for the verifier must not find their suite instrumented,
so this module adds one command-line option and, unless that option or the
opt-in plugin says otherwise, imports nothing further and touches nothing.
The two ways to switch it on are equivalent:

.. code-block:: python

    # conftest.py -- one line
    pytest_plugins = ["stelling.overflow"]

.. code-block:: console

    $ pytest -p stelling.overflow
    $ pytest --stelling-overflow=auto

THE PRIMARY CHANNEL IS THE TERMINAL REPORTER, NOT ``warnings.warn``. Under
``-W error::UserWarning`` — common in scientific repos — a "safely disabled"
warning becomes an exception and crashes the suite, which is exactly what this
guardrail exists to prevent. The reporter cannot be escalated.

``--stelling-overflow``:

``auto``
    the default once switched on. Arm if possible; if not, report it in the
    summary and leave the exit code alone.
``require``
    fail the session if it cannot arm. This is how a user who depends on the
    tripwire opts into hard failure. **The escalation decision belongs to
    them**, which is why it is not the default.
``off``
    do nothing, import nothing.
"""

from __future__ import annotations

import dataclasses

import pytest

#: The module a user names to switch this on. Named here rather than only in
#: prose so the check and the documentation cannot drift apart.
OPT_IN_PLUGIN = "stelling.overflow"

#: The ``pytest11`` entry point's NAME in ``pyproject.toml``, and therefore the
#: name this module is registered under when ``stelling/overflow.py`` has to
#: register it by hand because autoload is off.
#:
#: IT HAS TO BE THIS NAME AND NOT THE DOTTED ONE, measured: ``-p`` plugins are
#: consumed in ``Config._preparse`` BEFORE ``load_setuptools_entrypoints``, so
#: registering under ``stelling._tripwire.plugin`` first makes the entry point
#: loader register the same module object a second time and raise ``ValueError:
#: Plugin already registered under a different name`` — an INTERNALERROR before
#: a single test collects, on the documented spelling, in the default
#: environment. Under this name the loader's own ``get_plugin(ep.name)`` guard
#: sees it and skips.
#:
#: ``tests/test_tripwire_plugin.py`` pins it against ``pyproject.toml``.
ENTRY_POINT_NAME = "stelling_overflow"

#: Key under which a worker's payload travels back to the controller.
WORKER_KEY = "stelling_overflow"

#: Exit code when ``require`` is set and the tripwire could not arm under
#: xdist, where the controller cannot raise ``UsageError`` at configure time
#: because it is not the process that arms.
REQUIRE_EXITSTATUS = 1

_KEY: pytest.StashKey = pytest.StashKey()


class _State:
    """Per-process plugin state. Holds no jax object by construction: the
    recorder and the status are both primitives-only types."""

    def __init__(self) -> None:
        self.mode = "off"
        self.status = None
        self.recorder = None
        self.role = "single"
        self.workers_ready = 0
        self.workers_reported = 0
        self.worker_statuses: dict[str, str] = {}
        self.notes: list[str] = []


def pytest_addoption(parser):
    group = parser.getgroup("stelling")
    group.addoption(
        "--stelling-overflow",
        action="store",
        dest="stelling_overflow",
        default=None,
        choices=("auto", "require", "off"),
        help=(
            "Arm the stelling overflow tripwire, which reports out-of-range "
            "integer narrowings in traced code. 'auto' arms if it can and "
            "reports in the summary if it cannot; 'require' fails the session "
            "if it cannot arm; 'off' does nothing. Default: off, unless "
            f"'{OPT_IN_PLUGIN}' is loaded (conftest `pytest_plugins`, or -p), "
            "in which case 'auto'."
        ),
    )


def _resolve_mode(config) -> str:
    """Explicit opt-in, and the two spellings of it are equivalent.

    The flag wins when given, so ``-p stelling.overflow --stelling-overflow=off``
    means off. Otherwise loading the opt-in plugin means ``auto``, and
    anything else — including a plain ``pip install stelling`` — means
    ``off``.
    """
    explicit = config.getoption("stelling_overflow", None)
    if explicit is not None:
        return explicit
    if config.pluginmanager.hasplugin(OPT_IN_PLUGIN):
        return "auto"
    return "off"


def _is_worker(config) -> bool:
    """xdist's own discriminator: workers carry ``workerinput``."""
    return hasattr(config, "workerinput")


def _is_controller(config) -> bool:
    """A controller with workers runs no tests, so it must not arm: arming
    there would import jax into a process that never traces."""
    if _is_worker(config):
        return False
    return getattr(config.option, "dist", "no") not in ("no", None)


def _state(config) -> _State:
    state = config.stash.get(_KEY, None)
    if state is None:
        state = _State()
        config.stash[_KEY] = state
    return state


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    state = _state(config)
    state.mode = _resolve_mode(config)
    if state.mode == "off":
        return

    from stelling import _tripwire
    from stelling._tripwire import record

    if _is_controller(config):
        # The controller runs no tests. Arming here would import jax into a
        # process that never traces, and would report a denominator of zero.
        state.role = "controller"
        state.recorder = record.Recorder()
        return

    state.role = "worker" if _is_worker(config) else "single"
    status, recorder = _tripwire.arm()
    state.status = status
    state.recorder = recorder

    if state.mode == "require" and not status.armed and state.role == "single":
        raise pytest.UsageError(
            f"--stelling-overflow=require: the tripwire could not arm "
            f"[{status.code}] -- {status.explanation}"
        )


def pytest_unconfigure(config):
    state = config.stash.get(_KEY, None)
    if state is None or state.mode == "off" or state.role == "controller":
        return
    from stelling import _tripwire

    # Whatever this returns is NOT a disclosure route: ``pytest_unconfigure``
    # runs after the terminal summary has already been written, so a note
    # appended here is never printed. `foreign-patch` reaches the report
    # through :func:`_revalidate`, which runs while there is still a report to
    # put it in.
    _tripwire.disarm()


def _revalidate(state) -> None:
    """Ask the hook whether it is STILL live, at the end of the session.

    THE STATUS LINE AND THE DENOMINATOR ARE A CLAIM ABOUT THE WHOLE SESSION
    AND WERE BOTH FIXED AT ``pytest_configure``. Nothing re-checked them, so a
    hook that came out of the registry part-way through still printed
    ``armed`` over a denominator that stopped growing when it left. Driven
    three ways, and the sharpest is that under ``--stelling-overflow=require``
    the session exited 0, printed ``armed``, printed ``denominator: 1``, and
    had never seen the ``x + 500`` on ``int8`` that ran after the detachment:

    * a mid-session ``_tripwire.disarm()``;
    * a NESTED pytest session that also enables the tripwire — its
      ``pytest_unconfigure`` restores the original and disarms the outer one
      along with itself;
    * a registry rebind over the top of us.

    ``_adapter_jax.is_armed()`` existed, cost nothing, and was consulted
    nowhere. This is that check, plus the distinction between the two ways of
    not being armed, which are two different things to tell a user.

    Being honest about it is the other half, and it is §6's: what was counted
    up to the point of detachment is a PARTIAL, and
    :data:`report.PARTIAL_BANNER` says so in the same words a lost xdist
    worker gets.
    """
    if state.status is None or not state.status.armed:
        return
    from stelling import _tripwire

    live = _tripwire.live_check()
    if live == "armed":
        return
    state.status = dataclasses.replace(state.status, code=live)


# ---------------------------------------------------------------------------
# xdist: workers are isolated processes, so findings have to be serialised
# back. ``config.stash`` does not cross.
# ---------------------------------------------------------------------------


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Two jobs, in the one hook pytest gives for the end of a session.

    On a **worker**: populate ``workeroutput`` before xdist's own sender takes
    it. ``tryfirst=True`` is the ordering claim, and it is measured rather
    than cited — ``tests/test_tripwire_xdist.py`` runs a real two-worker
    session in a subprocess and asserts both payloads arrived. The payload is
    primitives only (dict / list / tuple / str / int / bool), which is what
    execnet serialises.

    On a **controller or single process** under ``require``: escalate. The
    controller cannot raise ``UsageError`` at configure time, because it is
    not the process that arms.

    And before either: RE-VALIDATE. This hook is the last point at which the
    session's own status can still reach both the exit code and the report,
    and :func:`_revalidate` is what makes the status a statement about the
    session rather than about its first millisecond. A worker re-validates
    too, so what crosses to the controller is what was true at the end of the
    worker's run.
    """
    config = session.config
    state = config.stash.get(_KEY, None)
    if state is None or state.mode == "off" or state.recorder is None:
        return

    _revalidate(state)

    if _is_worker(config):
        payload = state.recorder.as_payload()
        payload["status_code"] = state.status.code if state.status else "unexpected:none"
        payload["status_detail"] = state.status.explanation if state.status else ""
        config.workeroutput[WORKER_KEY] = payload
        return

    if state.mode == "require":
        status = _controller_status(state) if state.role == "controller" else state.status
        if status is not None and not status.armed and session.exitstatus == 0:
            session.exitstatus = REQUIRE_EXITSTATUS


# ``optionalhook=True`` on both: these are xdist's hooks, and without xdist
# installed pluggy refuses to register a plugin declaring hooks nothing has
# defined -- measured, `PluginValidationError: unknown hook
# 'pytest_testnodedown'`, an INTERNALERROR before a single test collects. The
# tripwire must be usable in an environment that has never heard of xdist.
@pytest.hookimpl(optionalhook=True)
def pytest_testnodeready(node):
    """Count workers EXPECTED.

    The hazard §6 names is silent undercount, not aggregation: a crashed
    worker never populates ``workeroutput``, and summing what arrives yields a
    confident wrong total. Expected-against-reported is what makes that
    visible, and is also what makes the JSONL fallback a measurable decision
    rather than a guess.
    """
    _state(node.config).workers_ready += 1


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """Absorb a worker's payload on the controller."""
    state = _state(node.config)
    if state.mode == "off" or state.recorder is None:
        return
    payload = (getattr(node, "workeroutput", None) or {}).get(WORKER_KEY)
    if payload is None:
        return
    state.workers_reported += 1
    gateway = getattr(node, "gateway", None)
    state.worker_statuses[getattr(gateway, "id", None) or f"w{state.workers_reported}"] = (
        payload.get("status_code", "?")
    )
    state.recorder.absorb(payload)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    state = config.stash.get(_KEY, None)
    if state is None or state.mode == "off":
        return
    from stelling._tripwire import report

    notes = tuple(state.notes)
    status = state.status
    if state.role == "controller":
        status = _controller_status(state)
        notes = notes + _worker_notes(state)
    if status is None:  # pragma: no cover - defensive
        return

    lines = report.render(status, state.recorder, notes)
    terminalreporter.write_sep("=", report.HEADER)
    for line in lines[2:]:
        terminalreporter.write_line(line)

    if state.mode == "require" and not status.armed:
        terminalreporter.write_line(
            "--stelling-overflow=require and the tripwire is not armed: "
            f"the session exits {REQUIRE_EXITSTATUS}."
        )


def _controller_status(state):
    """The controller does not arm, so its status is the workers' agreement."""
    from stelling import _tripwire

    codes = sorted(set(state.worker_statuses.values()))
    if not codes:
        return _tripwire.Status(
            code="no-worker-reported",
            detail=(
                "no worker reported a tripwire status, so nothing was "
                "measured. This is not a clean run."
            ),
        )
    if codes == ["armed"]:
        return _tripwire.Status(
            code="armed",
            detail=(
                f"{state.workers_reported} worker(s) armed. The controller "
                "runs no tests and is deliberately not instrumented."
            ),
        )
    return _tripwire.Status(
        code=codes[0] if len(codes) == 1 else "mixed",
        detail=f"worker statuses: {', '.join(codes)}",
    )


def _worker_notes(state) -> tuple[str, ...]:
    """§6's real hazard, disclosed: expected against reported."""
    if state.workers_ready == 0 and state.workers_reported == 0:
        return ()
    if state.workers_ready != state.workers_reported:
        return (
            f"    LOST WORKERS: {state.workers_reported} of "
            f"{state.workers_ready} workers reported. The figures below are a "
            "PARTIAL, not a total -- a worker that died never sent its "
            "findings, and summing what arrived would be a confident wrong "
            "answer.",
        )
    return (
        f"    {state.workers_reported} of {state.workers_ready} workers reported.",
    )
