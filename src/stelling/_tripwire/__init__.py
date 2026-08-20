# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The overflow tripwire: ``arm()``, ``disarm()``, ``Status``.

WHAT IT IS. An out-of-dtype-range Python integer constant is silently narrowed
on its way into a jax trace: ``x + 256`` on an ``int8`` array reaches the jaxpr
as ``add a 0:i8[]``, and the written ``256`` is destroyed with no error and no
warning. There is no supported mechanism that catches this — six were measured
and all leave it silent. This package attaches to the one site where the value
actually dies and reports it.

TWO INSTRUMENTS, TWO DOORS, AND NEITHER SUBSUMES THE OTHER. The one above is
the INLINE door and it REPORTS. ``eager.py`` is the second: ``jnp.full((), 256,
jnp.int8)`` is ``0`` before any primitive is bound, so nothing above ever sees
it, and that detector RAISES at the line that wrote the constant. It is
opt-in, arms separately, fails separately, and is not switched on by anything
that switches this one on. ``design/eager-truncation-detector.md`` is the
argument for its shape.

THE RULE THIS PACKAGE EXISTS INSIDE. Only ``_adapter_jax.py`` may name a
private jax module, and it is the only file in the repository that may.
``design/private-jax-boundary.md`` is why, and
``tests/test_import_hygiene.py::test_private_jax_modules_banned_everywhere``
is what holds it shut.

**This module imports nothing at module scope, and must stay that way.**
``import stelling`` pulls in no jax today; a package that armed a jax hook at
import would be the obvious way to lose that. The adapter is imported LAZILY,
inside functions, and no jax symbol is named here.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Codes that mean "not armed, and here is why". Stable and greppable; a
#: user's CI can match on them. ``unexpected:<ExcType>`` is the open one.
FAILURE_CODES = (
    "no-module",
    "no-registry",
    "no-entry",
    "not-invoked",
    "cries-wolf",
    "mis-attributed",
    "below-floor",
    "foreign-patch",
    "detached",
    "no-worker-reported",
    "mixed",
    # The eager construction-site detector's own codes (Mode 2). They are in
    # THIS tuple and not a second one because the tuple is advertised as the
    # complete list of "not armed, and here is why" and a user greps it; a
    # second list somewhere else is a second thing to find.
    "no-site-module",
    "no-site",
    "signature-drift",
    "route-blind",
)

_WHAT_STILL_WORKS = (
    "Static checking is unaffected: `stelling.preconditions.check` and every "
    "verdict path work exactly as before."
)

_EXPLAIN = {
    "no-module": (
        "jax is not installed in this environment, so there is no trace to "
        "instrument."
    ),
    "no-registry": (
        "jax is installed, but the const-fold registry is not where the "
        "tripwire expects it. A jax release moved it. Refusing to guess is "
        "the designed behaviour for a tool keyed on a private surface."
    ),
    "no-entry": (
        "the const-fold registry is there and no longer has a rule for "
        "convert_element_type. The narrowing this tool watches may now happen "
        "somewhere else entirely."
    ),
    "not-invoked": (
        "the hook is attached and did not fire on a program that must make it "
        "fire. Attached-but-blind is what a jax version bump actually "
        "produces, and it is the failure a presence check misses."
    ),
    "cries-wolf": (
        "the hook fired on an IN-RANGE value, which means every finding this "
        "run would be noise. Disabled rather than trusted."
    ),
    "mis-attributed": (
        "the hook fired the right number of times and reported the wrong "
        "content -- the wrong value, dtype, file or line, or a narrowing "
        "whose independent recomputation disagreed with what was observed. A "
        "finding that points at the wrong line costs more trust than a "
        "missing one, so this disables the tool."
    ),
    "below-floor": (
        "this jax is older than the version whose const-fold rule this tool "
        "was written against."
    ),
    # THESE TWO CODES ARE RAISED BY BOTH INSTRUMENTS, so they name both
    # attachment points. They used to say "in the registry", which is the
    # const-fold tripwire's; the eager detector attaches to a module
    # ATTRIBUTE, and a user reading `foreign-patch` off the eager section was
    # sent to look at a registry that has nothing to do with it.
    "foreign-patch": (
        "the instrument armed and something else replaced its wrapper -- the "
        "tripwire's entry in the const-fold registry, or the eager "
        "detector's module attribute -- before the session ended, so an "
        "unmeasured part of this run ran uninstrumented. Whatever replaced it "
        "is left in place rather than clobbered."
    ),
    "detached": (
        "the instrument armed and its wrapper was taken back out -- of the "
        "const-fold registry for the tripwire, or off the module attribute "
        "for the eager detector -- before the session ended: a mid-run "
        "`disarm()`, or a nested pytest session that enabled it and restored "
        "the original when it finished. An unmeasured part of this run "
        "therefore ran uninstrumented, and anything reported below covers "
        "only the part that did not."
    ),
    # The two codes an xdist CONTROLLER can carry. It never arms -- with `-n
    # auto` it runs no tests -- so its status is its workers' agreement, and
    # these two are the ways they can fail to agree. They live here, and not
    # only in ``plugin.py``, because this list is advertised as complete and a
    # user greps it.
    "no-worker-reported": (
        "this is an xdist CONTROLLER and not one worker sent back a tripwire "
        "status, so nothing about this run was measured. The controller does "
        "not arm -- it runs no tests -- so it has nothing of its own to "
        "report either. This is not a clean run."
    ),
    "mixed": (
        "this is an xdist CONTROLLER and its workers did not agree: some "
        "armed and some did not, so anything below covers only part of the "
        "run. The per-worker codes are on the status line."
    ),
    # --- the eager construction-site detector (Mode 2) ---
    "no-site-module": (
        "jax is installed and the module carrying the eager narrowing did "
        "not import. A jax release moved it, and a tool keyed on a private "
        "surface refuses rather than guessing."
    ),
    "no-site": (
        "the module is there and no longer carries the function that narrows "
        "an out-of-range integer during array construction. The narrowing "
        "this detector watches may now happen somewhere else entirely."
    ),
    "signature-drift": (
        "the eager narrowing site is there and its first two parameters are "
        "not the ones this hook reads. A hook that read the wrong argument "
        "would raise about a truncation that did not happen, at a line that "
        "did not write it, so it refuses to attach at all."
    ),
    "route-blind": (
        "the eager detector attached and one of the construction routes it "
        "claims to watch no longer reaches it. Attached-but-blind on ONE "
        "route is what a jax release actually produces; keeping the other "
        "routes and losing this one quietly is not a trade this tool makes "
        "on your behalf, so it disables itself and names the route."
    ),
}


@dataclass(frozen=True)
class Status:
    """What arming produced. Primitives only — this crosses no jax boundary."""

    code: str
    detail: str = ""
    jax_version: str | None = None
    rule_name: str | None = None
    rule_hash: str | None = None
    #: The hash recorded for :attr:`jax_version` in
    #: ``_adapter_jax._KNOWN_HASHES``, or ``None`` when that release has no
    #: row — i.e. when nobody has ever read the rule on this jax. ``None``
    #: here is a THIRD state, not a synonym for "changed"; see
    #: :attr:`hash_state`.
    known_hash: str | None = None
    registry_size: int | None = None

    @property
    def armed(self) -> bool:
        return self.code == "armed"

    @property
    def hash_state(self) -> str:
        """What the recorded rule hash says, in one word.

        ``unreadable``
            the installed rule's source could not be read, so there is
            nothing to compare. Nothing is claimed either way.
        ``never-read``
            :attr:`jax_version` has no row in the version -> hash map. Not
            "changed": nobody has ever read the rule on this release, so the
            tool has no opinion about it yet. A nightly build lands here by
            construction.
        ``as-tested``
            the running rule hashes to what this release's row records.
        ``changed``
            it does not. The SAME release is reporting a different rule
            source than the one written down for it.

        One definition, three readers — :func:`report.render_status`,
        ``.github/scripts/tripwire_canary.py`` and
        ``tests/test_tripwire_arm.py`` — because three copies of a
        four-way case is three chances to disagree about what ``None``
        means. **Nothing here gates arming**: `arm()` never calls it.
        """
        if not self.rule_hash:
            return "unreadable"
        if self.known_hash is None:
            return "never-read"
        return "as-tested" if self.rule_hash == self.known_hash else "changed"

    @property
    def meaning(self) -> str:
        """The middle third of §4's contract: what the code MEANS.

        Separated from :attr:`explanation` because the primary channel prints
        the three thirds on separate lines, and it used to render
        :attr:`detail` there — which ``arm()`` leaves EMPTY for every one of
        the failure codes, so the line came out as ``NOT ARMED [no-module] --``
        with a dangling dash and no middle third at all. This is the part that
        is never empty.

        The fallback does not say "while arming": a controller's status is its
        workers' agreement and it never arms, so a code it carries has nothing
        to do with arming.
        """
        if self.armed:
            return self.detail
        # SPLIT ON THE COLON FIRST. Two codes carry a payload --
        # ``unexpected:<ExcType>`` and ``route-blind:<route>`` -- and a lookup
        # on the whole string finds neither, so the one thing a reader most
        # needs (which route went blind) used to arrive with the generic "an
        # error it does not have a name for" beside it.
        return _EXPLAIN.get(
            self.code.split(":", 1)[0],
            "the tripwire hit an error it does not have a name for.",
        )

    @property
    def explanation(self) -> str:
        """What happened, what it means, and what still works. Every failure
        message carries the third part, so that "disabled" never reads as
        "you are unprotected"."""
        if self.armed:
            return self.detail
        return f"{self.meaning} {_WHAT_STILL_WORKS}".strip()

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{self.code}: {self.explanation}"


def arm(recorder=None):
    """Arm the tripwire. Returns ``(Status, Recorder)``; **never raises**.

    Fail-closed means fail *quietly and legibly*: every route out of here is a
    :class:`Status`, and the caller decides whether a non-armed status is
    fatal. That decision belongs to the user (``--stelling-overflow=require``),
    not to this function.

    The order is deliberate. Version bounding is a pre-filter, so it runs
    before anything is touched; then the registry is located; then the wrapper
    is installed; then it is probed **in both directions**, because a
    positive-only probe passes on a hook replaced by "record everything".
    """
    from stelling._tripwire import _adapter_jax as adapter
    from stelling._tripwire import record as _record

    rec = recorder if recorder is not None else _record.Recorder()

    def status(code: str, detail: str = "") -> Status:
        return Status(
            code=code,
            detail=detail,
            jax_version=_safe(adapter.jax_version),
            rule_name=_safe(adapter.rule_name),
            rule_hash=_safe(adapter.rule_hash),
            known_hash=_safe(adapter.known_hash),
            registry_size=_safe(adapter.registry_size),
        )

    try:
        located = adapter.locate()
        if located != "located":
            return status(located), rec

        version_code, disclosure = adapter.version_check()
        if version_code == "below-floor":
            return status("below-floor", disclosure), rec

        installed = adapter.install(rec)
        if installed not in ("installed", "already-armed"):
            return status(installed), rec
        if installed == "already-armed":
            # THE RECORDER THIS RETURNS MUST BE THE ONE THE LIVE WRAPPER
            # WRITES TO. Arming twice does not re-wrap -- correctly -- but the
            # fresh `Recorder()` above was then handed back to a caller who
            # had no way to know it was connected to nothing. Measured: under
            # `-p stelling.overflow`, which arms the tripwire for the whole
            # session, a test that called `arm()` again got a recorder whose
            # counters stayed at zero however much it traced, so an assertion
            # about what the tripwire saw was false BY CONSTRUCTION rather
            # than by measurement. A caller that passed its own recorder gets
            # the live one back instead: "here is what is actually recording"
            # is the only honest answer this function has.
            live = adapter.installed_recorder()
            if live is not None:
                rec = live

        probe = adapter.selfcheck()
        if probe != "armed":
            adapter.restore()
            return status(probe), rec
        return status("armed", disclosure), rec
    except Exception as exc:  # noqa: BLE001 - a guardrail may not raise
        try:
            adapter.restore()
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive
            pass
        return Status(code=f"unexpected:{type(exc).__name__}", detail=str(exc)[:200]), rec


def disarm() -> str:
    """Restore the original rule by identity. Returns a status code, never raises.

    ``foreign-patch`` if something else patched over the wrapper: say so
    rather than silently clobbering whatever replaced it.
    """
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return adapter.restore()
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"


def is_armed() -> bool:
    """Whether this process's wrapper is still the live registry entry."""
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return adapter.is_armed()
    except Exception:  # noqa: BLE001
        return False


def fires_count() -> int | None:
    """Total narrowing fires observed (including repeats), or None if not armed.

    Used by the gate to detect disarm-during-trace (before is int, after is
    None → unsafe). The per-gate fire count comes from ``_push_gate`` /
    ``_pop_gate`` instead.
    """
    try:
        from stelling._tripwire import _adapter_jax as adapter

        rec = adapter._installed.get("recorder")
        if rec is None:
            return None
        return rec.fires
    except Exception:  # noqa: BLE001
        return None


def _push_gate() -> None:
    """Begin a gated trace. The wrapper will increment this gate's counter."""
    from stelling._tripwire import _adapter_jax as adapter

    adapter._gate_fire_stack().append(0)


def _pop_gate() -> int:
    """End a gated trace. Returns narrowings observed during THIS trace only.

    Each gate invocation pushes a counter; the fold-rule wrapper increments
    only the top of the stack; this pops it. Nested ``check()`` calls each
    get their own counter and never contaminate the outer gate.
    """
    from stelling._tripwire import _adapter_jax as adapter

    stack = adapter._gate_fire_stack()
    if stack:
        return stack.pop()
    return 0


def evict_trace_caches() -> str:
    """Empty jax's trace caches. ``evicted``, or a code saying why not.

    The gate in :func:`stelling.preconditions.check` calls this immediately
    before the trace it watches. jax's trace cache is keyed on the jitted
    callable and its avals, not on the harness, so without this a
    ``@jax.jit`` helper some earlier trace already warmed is replayed from
    cache: the fold rule never runs over its body, and the gate's zero means
    "I saw nothing" rather than "there was nothing". Emptying the cache makes
    the observation complete WITH RESPECT TO JAX'S CACHES, IN A
    SINGLE-THREADED PROCESS. Both qualifiers are measured, not defensive:
    this empties jax's caches and no others, so a constant narrowed into a
    memo jax does not own (``jaxpr_as_fun`` over a saved jaxpr, a user
    ``functools.lru_cache``, ``jax.closure_convert``) is still unobserved;
    and jax's cache is process-global while the gate's counter is
    per-thread, so a competing thread can re-warm a body inside the
    eviction-to-trace window. ``report.UNCOVERED`` carries both with their
    numbers.

    A code other than ``evicted`` is not a warning to be logged and dropped:
    it means the next trace's observation may be PARTIAL, and the gate's
    third state exists to say exactly that. See
    :func:`_adapter_jax.evict_trace_caches` for the codes, for why this is
    eviction rather than detection, and for what it costs.
    """
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return adapter.evict_trace_caches()
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"


def live_check() -> str:
    """``armed``, ``detached`` or ``foreign-patch``. Never raises.

    What :func:`arm` established at the start of a session is not still true
    at the end of it, and the report is written at the end. See
    :func:`_adapter_jax.live_check`.
    """
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return adapter.live_check()
    except Exception as exc:  # noqa: BLE001
        return f"unexpected:{type(exc).__name__}"


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Mode 2: the eager construction-site detector.
#
# A SECOND, INDEPENDENT INSTRUMENT, and the independence is deliberate. The
# tripwire above watches a const-fold rule and REPORTS; this raises at the
# construction site and stops the program. They arm separately, fail
# separately and are switched on separately, because they answer different
# questions and a user may want either without the other:
#
#   * the tripwire is a REPORT over a session, and can be armed on a suite
#     that has undeclared truncations in it -- it says so and carries on;
#   * the eager detector is a RULE, and a session it is armed on either has no
#     undeclared truncation in it or does not finish.
#
# Everything about the rule, the exception, the two declarations, and why
# there is no value-based carve-out lives in ``eager.py``. These four are the
# same lazy-import façade the tripwire's own entry points are, for the same
# reason: importing this package must not import jax.
# ---------------------------------------------------------------------------


def arm_eager():
    """Arm the eager construction-site detector. Returns a :class:`Status`.

    Never raises -- the same fail-closed contract :func:`arm` has, and the same
    reason: the caller decides whether a non-armed status is fatal. For THIS
    instrument the caller almost always should, because there is no degraded
    mode: a detector that could not attach is not watching, and the whole
    value of Mode 2 is that a program either contains no undeclared truncation
    or does not run.
    """
    from stelling._tripwire import eager

    return eager.arm()


def disarm_eager() -> str:
    """Restore jax's own constructor by identity. A code, never an exception."""
    from stelling._tripwire import eager

    return eager.disarm()


def eager_live_check() -> str:
    """``armed``, ``detached`` or ``foreign-patch`` for the eager hook."""
    from stelling._tripwire import eager

    return eager.live_check()


def displaced() -> tuple[str, ...]:
    """The armed hooks that are no longer live. ``()`` when none are.

    ONE INSTRUMENT, BOTH HOOKS, and B15's audit is why it exists at all:
    ``live_check() == "foreign-patch"`` is a fourth way the trace gate's watch
    goes partial, and the gate consulted none of it. Rebinding the const-fold
    registry entry after arming leaves the recorder's identity unchanged and
    ``fires_count()`` unchanged -- so both of the gate's partiality tests pass,
    the fire counter stays at zero because our wrapper is no longer called, and
    ``check()`` returns VERIFIED on a route the inventory calls ``watched``.
    Pre-existing, and byte-identical on ``main`` before this batch.

    See :func:`_adapter_jax.displacement_check` for the per-hook detail and for
    why one instrument covers both.
    """
    try:
        from stelling._tripwire import _adapter_jax as adapter

        return adapter.displaced()
    except Exception:  # noqa: BLE001
        return ()
