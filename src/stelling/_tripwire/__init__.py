# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The overflow tripwire: ``arm()``, ``disarm()``, ``Status``.

WHAT IT IS. An out-of-dtype-range Python integer constant is silently narrowed
on its way into a jax trace: ``x + 256`` on an ``int8`` array reaches the jaxpr
as ``add a 0:i8[]``, and the written ``256`` is destroyed with no error and no
warning. There is no supported mechanism that catches this — six were measured
and all leave it silent. This package attaches to the one site where the value
actually dies and reports it.

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
    "foreign-patch": (
        "something else replaced the tripwire's wrapper while the session ran. "
        "Not restoring over it."
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
    known_hash: str | None = None
    registry_size: int | None = None

    @property
    def armed(self) -> bool:
        return self.code == "armed"

    @property
    def explanation(self) -> str:
        """What happened, what it means, and what still works. Every failure
        message carries the third part, so that "disabled" never reads as
        "you are unprotected"."""
        if self.armed:
            return self.detail
        base = _EXPLAIN.get(self.code)
        if base is None:
            base = (
                "the tripwire hit an error it does not have a name for while "
                "arming."
            )
        return f"{base} {_WHAT_STILL_WORKS}".strip()

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
            known_hash=adapter._KNOWN_HASH,
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


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None
