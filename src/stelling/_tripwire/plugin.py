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

``--stelling-eager-truncation`` is a SECOND and INDEPENDENT dial, for the
opt-in eager construction-site detector (Mode 2). It is off by default, it
does not require ``--stelling-overflow``, and ``--stelling-overflow`` does not
turn it on. Two dials because they are two instruments answering two
questions: the tripwire REPORTS over a session and can be armed on a suite
that contains undeclared truncations, while the eager detector is a RULE — a
session it is armed on either contains no undeclared truncation or does not
finish.

``off``
    the default. Nothing is patched and no jax is imported for it.
``error``
    arm the detector: every out-of-range integer constant narrowed during
    array construction raises :class:`stelling.EagerTruncationError` at the
    line that wrote it, unless the author declared it with
    ``stelling.intentional_wrap``.

**IT FAILS THE SESSION IF IT CANNOT ARM, and there is no ``require`` spelling
of it.** The tripwire has one because it has a degraded mode worth having —
an unarmed tripwire still lets a suite run and still reports why it is not
watching. This has none: a detector that could not attach is not a quieter
rule, it is no rule, and a session that asked for one and silently did not get
it is the exact shape of false assurance the whole instrument exists to
remove.
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
        #: The eager detector's own dial, status and aggregated counters.
        #: Separate fields and not a second `_State`, because the two share a
        #: role and an xdist payload and nothing else.
        self.eager_mode = "off"
        self.eager_status = None
        self.eager_snapshot: dict | None = None
        self.eager_worker_statuses: dict[str, str] = {}


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
    group.addoption(
        "--stelling-eager-truncation",
        action="store",
        dest="stelling_eager_truncation",
        default="off",
        choices=("off", "error"),
        help=(
            "Arm the eager construction-site detector: an out-of-range "
            "integer constant narrowed during array construction "
            "(`jnp.full(shape, 256, jnp.int8)` and friends) raises "
            "stelling.EagerTruncationError at the line that wrote it. "
            "Declare a deliberate wrap with stelling.intentional_wrap(value, "
            "dtype). 'error' fails the session if the detector cannot attach. "
            "Default: off, and NOT turned on by --stelling-overflow."
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
    state.eager_mode = config.getoption("stelling_eager_truncation", "off") or "off"
    if state.mode == "off" and state.eager_mode == "off":
        return

    from stelling import _tripwire
    from stelling._tripwire import record

    if _is_controller(config):
        # The controller runs no tests. Arming here would import jax into a
        # process that never traces, and would report a denominator of zero.
        # TRUE OF BOTH INSTRUMENTS, and for the eager one it is sharper still:
        # a detector armed where nothing constructs an array cannot raise, so
        # a green controller would be a green with no program behind it.
        state.role = "controller"
        state.recorder = record.Recorder()
        return

    state.role = "worker" if _is_worker(config) else "single"
    if state.mode != "off":
        status, recorder = _tripwire.arm()
        state.status = status
        state.recorder = recorder

        if state.mode == "require" and not status.armed and state.role == "single":
            raise pytest.UsageError(
                f"--stelling-overflow=require: the tripwire could not arm "
                f"[{status.code}] -- {status.explanation}"
            )

    if state.eager_mode == "error":
        # ARMED SECOND, AND THE REASON IS THE DENOMINATOR AND NOT A CRASH.
        # This used to say that `_tripwire.arm()`'s self-check traces a
        # program the eager detector would RAISE on, so arming in the other
        # order would disable the tripwire on a healthy pair of hooks. Driven,
        # both ways: both report `armed` either way. What the reversed order
        # actually costs is measurable and smaller -- the tripwire's
        # self-check reaches the eager hook with 2 IN-RANGE conversions, so
        # arming the rule first starts every session's eager denominator at 2
        # instead of 0, and those two are stelling's own traffic rather than
        # the suite's. The order stands; the reason is this one.
        #
        # THE EAGER SIDE HAS THE MIRROR PROBLEM AND PAYS FOR IT THE SAME WAY.
        # Its route probes swap the observer out and never reach the policy
        # that raises -- which is why arming also runs `eager._origin_control`,
        # one narrowing of each origin through the LIVE policy. That control
        # moves the counters and writes rows, so it saves them and writes them
        # back: an instrument whose own self-check appeared in the denominator
        # would be reporting a rate about itself.
        state.eager_status = _tripwire.arm_eager()
        if not state.eager_status.armed and state.role == "single":
            raise pytest.UsageError(
                f"--stelling-eager-truncation=error: the eager "
                f"construction-site detector could not attach "
                f"[{state.eager_status.code}] -- {state.eager_status.explanation} "
                "There is no degraded mode for this instrument: a session that "
                "asked for the rule and did not get it would be a session "
                "whose green means nothing, so it fails here instead."
            )


def pytest_unconfigure(config):
    state = config.stash.get(_KEY, None)
    if state is None or state.role == "controller":
        return
    if state.mode == "off" and state.eager_mode == "off":
        return
    from stelling import _tripwire

    if state.eager_mode == "error":
        _tripwire.disarm_eager()
    if state.mode == "off":
        return

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


def _capture_eager(state) -> None:
    """Read the eager detector's counters, and re-validate it, before disarming.

    SAME ARGUMENT AS :func:`_revalidate`, one instrument over. An eager status
    fixed at ``pytest_configure`` is a claim about the first millisecond of the
    session; something can rebind ``_convert_element_type`` over stelling's
    wrapper at any point after that, and the detector then stops raising with
    no other symptom at all — there is no denominator to go flat, because the
    hook that maintains it is the hook that was displaced. The counters have
    to be read HERE, in the last hook that can still reach both the exit code
    and the report; ``pytest_unconfigure``, where the disarm happens, runs
    after the summary has been written.
    """
    if state.eager_mode == "off":
        return
    if state.role == "controller":
        # THE CONTROLLER NEVER ARMS, so its own counters are zeros -- and by
        # the time this runs, ``pytest_testnodedown`` has already merged every
        # worker's snapshot into ``state.eager_snapshot``. Reading the local
        # module here overwrote that merge with the zeros: measured, a `-n 2`
        # session whose workers reported `conversions: 5, truncations: 1` and
        # one declaration printed `0 scalar integer conversion(s)` and no
        # declarations at all. That is the beautiful zero this section's
        # docstring exists to prevent, produced by the section itself.
        return
    from stelling import _tripwire
    from stelling._tripwire import eager

    state.eager_snapshot = eager.snapshot()
    if state.eager_status is None or not state.eager_status.armed:
        return
    live = _tripwire.eager_live_check()
    if live != "armed":
        state.eager_status = dataclasses.replace(state.eager_status, code=live)


# ---------------------------------------------------------------------------
# xdist: workers are isolated processes, so findings have to be serialised
# back. ``config.stash`` does not cross.
# ---------------------------------------------------------------------------


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Two jobs, in the one hook pytest gives for the end of a session.

    On a **worker**: populate ``workeroutput`` before xdist's own sender takes
    it. The payload is primitives only (dict / list / tuple / str / int /
    bool), which is what execnet serialises.

    ``tryfirst=True`` IS A PRECAUTION AND NOT WHAT MAKES THAT WORK, and this
    used to say the ordering was "measured rather than cited". It was not:
    ``tests/test_tripwire_xdist.py`` asserts the payloads ARRIVED, which they
    do under ``trylast`` too — driven, the whole tripwire suite passes with
    ``tryfirst`` flipped to ``trylast``. The reason is upstream and is now
    pinned rather than assumed: xdist's own
    ``WorkerInteractor.pytest_sessionfinish`` is a **hookwrapper**, so it
    yields to every other implementation, whatever their ordering, and sends
    only afterwards. ``tryfirst`` is kept because it costs nothing and is what
    a plain sender would need; the test that would fail if xdist stopped being
    a wrapper is the one that names this.

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
    if state is None:
        return
    if state.mode == "off" and state.eager_mode == "off":
        return

    _revalidate(state)
    _capture_eager(state)

    if _is_worker(config):
        from stelling._tripwire import record

        recorder = state.recorder if state.recorder is not None else record.Recorder()
        payload = recorder.as_payload()
        payload["status_code"] = state.status.code if state.status else "unexpected:none"
        payload["status_detail"] = state.status.explanation if state.status else ""
        # The eager detector's own row travels in the SAME payload, under its
        # own keys. Primitives only, like everything else that crosses
        # execnet: `eager.snapshot()` returns ints, strings and dicts of them
        # by construction, for exactly this reason.
        payload["eager_status_code"] = (
            state.eager_status.code if state.eager_status else "off"
        )
        payload["eager_snapshot"] = state.eager_snapshot or {}
        config.workeroutput[WORKER_KEY] = payload
        return

    # THE EAGER DETECTOR'S ESCALATION COMES FIRST, AND ABOVE THE RECORDER
    # GUARD, because the two instruments are two dials and this one does not
    # depend on the other having been switched on. It used to sit below `if
    # state.recorder is None: return` -- and `state.recorder` is None exactly
    # when `--stelling-overflow=off`, which is the spelling the docs recommend
    # for running the eager detector alone. Measured with the hook displaced
    # mid-session: `--stelling-eager-truncation=error` alone exited 0, `...
    # --stelling-overflow=off` exited 0, and only `... --stelling-overflow=
    # auto` exited 1. A rule that fails to attach and exits 0 is the false
    # assurance the instrument exists to remove.
    #
    # It needs its own escalation, separate from `require`'s, because a
    # CONTROLLER cannot raise `UsageError`: it is not the process that arms,
    # so the `pytest_configure` refusal that covers a single process never
    # runs there.
    if state.eager_mode == "error":
        eager_status = (
            _eager_controller_status(state)
            if state.role == "controller"
            else state.eager_status
        )
        if (
            eager_status is not None
            and not eager_status.armed
            and session.exitstatus == 0
        ):
            session.exitstatus = REQUIRE_EXITSTATUS

    if state.recorder is None:
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
    """Absorb a worker's payload on the controller.

    THE TWO INSTRUMENTS ARE ABSORBED SEPARATELY, and the gate at the top used
    to be `state.mode == "off" or state.recorder is None` -- the tripwire's
    condition -- so a session running the eager detector ALONE dropped every
    worker's eager payload on the floor. Measured: `-n 2
    --stelling-overflow=off --stelling-eager-truncation=error` on a fully
    green suite reported `NOT ARMED [no-worker-reported]` and exited 1, having
    been told the answer by both workers.
    """
    state = _state(node.config)
    if state.mode == "off" and state.eager_mode == "off":
        return
    payload = (getattr(node, "workeroutput", None) or {}).get(WORKER_KEY)
    if payload is None:
        return
    state.workers_reported += 1
    gateway = getattr(node, "gateway", None)
    worker_id = getattr(gateway, "id", None) or f"w{state.workers_reported}"
    if state.mode != "off" and state.recorder is not None:
        state.worker_statuses[worker_id] = payload.get("status_code", "?")
        state.recorder.absorb(payload)
    eager_code = payload.get("eager_status_code")
    if eager_code is not None and eager_code != "off":
        state.eager_worker_statuses[worker_id] = eager_code
    state.eager_snapshot = _merge_eager(
        state.eager_snapshot, payload.get("eager_snapshot") or {}
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _merge_eager(into: dict | None, payload: dict) -> dict:
    """Sum two eager snapshots. Counts add; per-site rows add and keep a text.

    A SUM AND NOT A MAX, and the per-site rows are kept rather than only the
    totals, because the totals alone cannot answer the question the section
    exists for -- WHERE was a wrap declared -- and two workers that each
    declared a wrap at the same line are two declarations at one site.
    """
    merged = dict(into or {})
    merged["conversions"] = merged.get("conversions", 0) + payload.get("conversions", 0)
    merged["truncations"] = merged.get("truncations", 0) + payload.get("truncations", 0)
    merged["suppressed_jax"] = merged.get("suppressed_jax", 0) + payload.get(
        "suppressed_jax", 0
    )
    merged["internal_errors"] = merged.get("internal_errors", 0) + payload.get(
        "internal_errors", 0
    )
    merged["resets"] = merged.get("resets", 0) + payload.get("resets", 0)
    for key in ("declared", "permitted", "suppressed"):
        rows = dict(merged.get(key) or {})
        for site, row in (payload.get(key) or {}).items():
            existing = rows.get(site)
            rows[site] = (
                [existing[0] + row[0], row[1]] if existing else [row[0], row[1]]
            )
        merged[key] = rows
    return merged


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    state = config.stash.get(_KEY, None)
    if state is None:
        return
    if state.eager_mode != "off":
        _write_eager_summary(terminalreporter, state)
    if state.mode == "off":
        return
    from stelling._tripwire import report

    # NO GENERAL `notes` LIST. There was one, appended to in
    # ``pytest_unconfigure``, which runs after this function has written the
    # summary -- so it was a disclosure channel that could not disclose. The
    # notes that exist are the ones computed HERE, where there is still a
    # report to put them in.
    notes: tuple[str, ...] = ()
    status = state.status
    if state.role == "controller":
        status = _controller_status(state)
        notes = _worker_notes(state)
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


def _write_eager_summary(terminalreporter, state) -> None:
    """The eager detector's section, printed whether or not anything fired.

    THE DENOMINATOR IS ALWAYS PRINTED, which is this project's standing rule
    for every instrument and is load-bearing here for a reason particular to
    this one: the eager detector's success case is that NOTHING happened, and
    "0 undeclared truncations" is also exactly what a hook that was never
    called produces. The number of conversions it saw is what separates those
    two, and a section that printed only the zero would be the beautiful zero
    this project keeps finding.

    THE DECLARATIONS ARE PRINTED TOO, with their sites. A declaration is a
    premise the tool carried -- the same standing an `assume()` has in a
    verdict, which is stamped and disclosed rather than applied quietly -- and
    a premise nobody can see is indistinguishable from the silence the
    detector exists to end. The same goes double for `expected_truncation`
    regions, which permit a truncation outright: those are printed with their
    reasons.
    """
    from stelling._tripwire import report

    status = state.eager_status
    if state.role == "controller":
        status = _eager_controller_status(state)
    lines = report.render_eager(status, state.eager_snapshot)
    if not lines:
        return
    terminalreporter.write_sep("=", report.EAGER_HEADER)
    for line in lines:
        terminalreporter.write_line(line)


def _eager_controller_status(state):
    """The controller does not arm, so its status is its workers' agreement.

    The same shape, and the same three answers, as :func:`_controller_status`.
    """
    from stelling import _tripwire

    codes = sorted(set(state.eager_worker_statuses.values()))
    if not codes:
        return _tripwire.Status(
            code="no-worker-reported",
            detail=(
                "no worker reported an eager-detector status, so nothing "
                "about this run was measured by it."
            ),
        )
    if codes == ["armed"]:
        return _tripwire.Status(
            code="armed",
            detail=(
                f"{len(state.eager_worker_statuses)} worker(s) armed the "
                "eager detector. The controller runs no tests and is "
                "deliberately not instrumented."
            ),
        )
    return _tripwire.Status(
        code=codes[0] if len(codes) == 1 else "mixed",
        detail=f"worker eager statuses: {', '.join(codes)}",
    )
