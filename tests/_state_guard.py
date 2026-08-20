# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A declared inventory of process-global state, fingerprinted around every test.

WHY THIS EXISTS, and it is an incident rather than a principle. A test in
``tests/test_tripwire_arm.py`` left stelling's own wrapper installed as jax's
live const-fold rule for the rest of the interpreter. The battery that noticed
was an *exit-code* battery two hundred lines further down the same file: every
one of its six cells went green off a background fatal branch (`arm()` reporting
a CONTRADICTED hash row) that no cell named, while the control branch each cell
claimed to measure was never entered. It survived two audit rounds. The comment
above ``_CANARY_PROCESS_TABLE`` in that file carries the full telling.

Nothing was wrong with the assertions. What was missing was an instrument: a
polluted process is indistinguishable from a clean one unless something looks,
and nothing looked. So this module looks — before and after every test — and
**fails the test that changed something**, which is the one fact a downstream
victim can never supply.

────────────────────────────────────────────────────────────────────────────
HOW IT DECIDES, AND WHY THERE IS NO CASCADE
────────────────────────────────────────────────────────────────────────────

Each test's *before* is read fresh at its own setup, not against a session
baseline. A polluter is therefore named exactly once: the next test inherits
the polluted value as its own baseline and passes unless it changes something
in its turn. That is deliberate, and it is what "name the culprit, not the
victim" means mechanically. It also means this guard **does not repair
anything** — it does not own the state it watches, and a guard that quietly put
jax's registry back would be hiding the defect it exists to report.

────────────────────────────────────────────────────────────────────────────
TWO ALTITUDES, BECAUSE ONE OF THEM COULD NOT SEE ITS OWN SHAPE OF DEFECT
────────────────────────────────────────────────────────────────────────────

:func:`state_guard` is function-scoped, so it brackets each test and the
fixtures that test owns. **A module- or class-scoped fixture that never
restores is outside every one of those windows** — it is set up before the
first test's guard reads ``before`` and torn down after the last one has read
``after`` — so it escaped entirely. Driven, with the restore deleted from the
module-scoped ``_x64`` fixture in ``tests/test_0_2_0_regression.py`` (jax
0.11.0)::

    mutated : 21 passed   [X64PROBE] jax_enable_x64 at session finish = True
    control : 21 passed   [X64PROBE] jax_enable_x64 at session finish = False

Green, silent, and every later module inherits ``x64 = True``. The SAME
deletion in a function-scoped fixture is named immediately. Two near-identical
defects, one caught and one invisible.

This module's own argument for having no outer guard was that it *"would
report every module that sets x64 for its own duration"*, and that is
**false**, measured: pytest sets a conftest's module-scoped autouse fixture up
*before* the test module's own module-scoped fixtures and tears it down
*after* them (lazily-requested module fixtures included), so a module that
sets x64 and puts it back is bracketed and silent. :func:`module_state_guard`
is therefore the same instrument one scope out.

**IT REPORTS ONLY WHAT THE FUNCTION GUARD COULD NOT SEE**, which is what keeps
"named exactly once" true and keeps :data:`PINNED_EXEMPTIONS` meaning what it
says. Every entry a function-scoped guard observed moving inside this module —
reported OR licensed by an exemption — is recorded in :data:`_DECIDED_HERE`
and skipped at module teardown. A test's offence is named at the altitude that
can name a test; what is left over happened outside every test, and only that
is the module's.

────────────────────────────────────────────────────────────────────────────
WHAT IT WATCHES — the inventory, and it is enumerated
────────────────────────────────────────────────────────────────────────────

:data:`ENTRIES`. Four, named by the incident: the identity of jax's registered
const-fold rule, the tripwire's installation record, ``jax_enable_x64``, and
the ``STELLING_*`` / ``JAX_*`` environment.

The environment entry is a **prefix rule and not a list**, on purpose. Every
other inventory in this repository that enumerated its own domain has had to
be widened after something outside it went unwatched; a key added to
``src/stelling`` tomorrow is watched here today without anybody remembering to
come back. The other three have no such spelling and are enumerated.

READERS TOLERATE ABSENT DEPENDENCIES. The zero-dep lane runs this too: with no
jax installed the three jax-shaped readers return :data:`ABSENT`, which is a
value like any other and compares equal to itself, so the guard is live there
and simply has less to watch. A reader that *raises* returns
:class:`Unreadable` carrying the exception text rather than a bare sentinel —
a reader that has silently started failing would otherwise report a constant,
and a constant fingerprint is a guard that cannot fire, which is the exact
shape this module exists to refuse.

────────────────────────────────────────────────────────────────────────────
WHAT IT DOES **NOT** WATCH — read this before trusting a green run
────────────────────────────────────────────────────────────────────────────

An enumerated inventory is incomplete by construction, and incompleteness
reading as coverage is the defect this file is a response to. So, measured
against what this suite actually mutates:

* **jax's trace and compilation caches.** Process-wide, unbounded, and
  routinely warmed by ordinary tests. ``tests/test_tripwire_arm.py`` documents
  a live hook reporting ``0 finding`` because shape ``(7,)`` had already been
  traced. Not watchable as a fingerprint, and eviction is a real operation
  with real cost, so this stays a subprocess-isolation problem.
* **jax configuration other than ``jax_enable_x64``** — ``jax_platforms``,
  ``jax_disable_jit``, ``jax_default_matmul_precision`` and the rest of the
  config object.
* **``sys.modules``, ``sys.path``, ``sys.meta_path``** and anything
  monkeypatched onto a module that ``monkeypatch`` did not do (an attribute
  replaced with a plain assignment in a test body is invisible here).
* **The tripwire recorder's CONTENTS.** The installation record's identity is
  watched; the counts inside a live recorder are not.
* **The warnings filter registry, logging configuration, the ``random`` and
  ``numpy.random`` global generators, the process CWD, open file descriptors,
  ``linecache``, and hypothesis's own database and profile registration.**
* **Environment keys outside the two prefixes** — ``PATH``, ``HOME``,
  ``PYTHONPATH``, ``CANARY_PARENT_STELLING``.
* **Anything a test changes and changes back within its own body.** This is a
  before/after fingerprint, not a trace: a test that arms the tripwire and
  disarms it is silent here, which is the intent.
* **A ``package``- or ``session``-scoped fixture that never restores, and
  anything a plugin or a conftest does at import.** The two guards below reach
  function and module scope; both of those are set up before the module guard
  reads ``before`` and torn down after it reads ``after``, exactly as a
  module-scoped fixture was outside the function guard. There is no such
  fixture in this tree today — measured, ``grep`` finds ``scope="module"`` and
  nothing wider — which is why the guard stops here rather than growing a
  third altitude nothing would exercise. **Add one and it is unwatched**, so
  the honest place to add the third guard is the commit that adds the first
  session-scoped fixture.

────────────────────────────────────────────────────────────────────────────
THE EXEMPTION LIST
────────────────────────────────────────────────────────────────────────────

:data:`PINNED_EXEMPTIONS`. A test that legitimately leaves a global changed
names itself here, with an entry and a reason. **The list is the mechanism, not
the loophole**: the incident's polluter was not on any list, so a list of any
length — including the empty one it has today — would have caught it. What the
list buys is that "this test pollutes" becomes a reviewable line in a diff
rather than an absence.

``tests/test_state_guard.py`` holds the exemption list to two conditions: every
entry it names must exist in :data:`ENTRIES`, and every test it names must
exist in the tree. A renamed test leaves a dangling exemption, and a dangling
exemption is a licence nobody is using pointed at nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import pytest

#: A reader whose dependency is not installed. A value, not an error: the
#: zero-dep lane reads this for every jax-shaped entry all session long, and
#: two ABSENTs compare equal, so the guard stays live and watches the rest.
ABSENT = "<dependency absent>"


@dataclass(frozen=True)
class Unreadable:
    """A reader raised. Compared like any other value, and printed like one.

    Carried rather than swallowed because a reader that has quietly started
    raising would otherwise report a constant, and a constant is a guard that
    cannot fire.
    """

    detail: str


@dataclass(frozen=True)
class Entry:
    """One watched piece of process-global state."""

    name: str
    read: Callable[[], object]
    what: str


# ── the readers ─────────────────────────────────────────────────────────────
#
# EVERY ROUTE TO jax's REGISTRY HERE GOES THROUGH THE ADAPTER'S SHIPPED API.
# `design/private-jax-boundary.md` rule 2 bans jax's private module tree
# everywhere, `tests/` included and with no exemption, so this module may not
# read the const-fold registry directly. `_adapter_jax.rule_name`, `rule_hash`,
# `registry_size` and `is_armed` are the shipped readings, they are what `report.render_status` and
# the nightly canary already read, and between them they separate the three
# states that matter:
#
#   clean     -> ('_convert_elt_type_folding_rule', <jax's hash>, 3, False)
#   armed     -> the same, with is_armed True (both hash readers prefer the
#                saved original, which is the reporting behaviour and is why
#                `is_armed` is in the tuple rather than inferred from them)
#   ORPHANED  -> ('stelling_const_fold_probe', <the wrapper's hash>, 3, False)
#
# The third is the incident. It is exactly the state in which `is_armed()` says
# "no" — nobody owns the wrapper — while the wrapper is live in jax's registry
# for the rest of the process, and it is why a bare `is_armed()` reading would
# have been an instrument that could not see the thing it was installed for.


def _adapter():
    from stelling._tripwire import _adapter_jax

    return _adapter_jax


def _read_const_fold_rule() -> object:
    from stelling import _optional

    if not _optional.available("jax"):
        return ABSENT
    adapter = _adapter()
    return (
        adapter.rule_name(),
        adapter.rule_hash(),
        adapter.registry_size(),
        adapter.is_armed(),
    )


def _read_tripwire_installation() -> object:
    from stelling import _optional

    if not _optional.available("jax"):
        return ABSENT
    adapter = _adapter()
    installed = adapter._installed
    # THE KEYS AND THE OBJECT IDENTITIES, never the recorder's contents: what
    # the incident left behind was a record that had been cleared while the
    # object it described stayed live, and both halves of that are here.
    #
    # `_detached` IS WATCHED BESIDE `_installed`, and it is not a bonus. It is
    # the record whose staleness produced the orphan: `detach("bypass")` saves
    # whatever the registry held — under an armed tripwire that is stelling's
    # own wrapper — and a `reattach()` after the wrapper has been retired puts
    # that wrapper back as jax's live rule with no `_installed` record owning
    # it. `restore()` fixes the saved entry up today, so a populated
    # `_detached` at the end of a test is no longer a live orphan; it is a
    # detachment nobody undid, and it is invisible in every other reading
    # here, `is_armed()` included.
    detached = adapter._detached
    return (
        tuple(sorted(installed)),
        id(installed.get("wrapper")),
        id(installed.get("original")),
        tuple(sorted(detached)),
        id(detached.get("entry")),
    )


def _read_jax_enable_x64() -> object:
    from stelling import _optional

    if not _optional.available("jax"):
        return ABSENT
    import jax

    return bool(jax.config.jax_enable_x64)


#: The two prefixes. See the docstring: a rule, not a list.
ENV_PREFIXES = ("STELLING_", "JAX_")


def _read_env() -> object:
    return tuple(
        sorted(
            (k, v) for k, v in os.environ.items() if k.startswith(ENV_PREFIXES)
        )
    )


def _guarded(read: Callable[[], object]) -> Callable[[], object]:
    def reader() -> object:
        try:
            return read()
        except Exception as exc:  # noqa: BLE001 - an instrument may not raise
            return Unreadable(f"{type(exc).__name__}: {exc}")

    return reader


ENTRIES: tuple[Entry, ...] = (
    Entry(
        "jax:const-fold-rule",
        _guarded(_read_const_fold_rule),
        "the identity of the rule object jax's const-fold registry holds for "
        "`convert_element_type`, and whether the tripwire owns it",
    ),
    Entry(
        "tripwire:installed",
        _guarded(_read_tripwire_installation),
        "`_adapter_jax._installed` and `._detached` — which keys each record "
        "holds and which rule objects they name",
    ),
    Entry(
        "jax:enable_x64",
        _guarded(_read_jax_enable_x64),
        "`jax.config.jax_enable_x64`, which decides the width every traced "
        "literal is judged at",
    ),
    Entry(
        "env:STELLING_*/JAX_*",
        _guarded(_read_env),
        "every environment key stelling or jax reads, by prefix",
    ),
)

ENTRY_NAMES = frozenset(e.name for e in ENTRIES)


# ── the exemption list ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Exemption:
    """One test licensed to leave one entry changed."""

    nodeid: str
    entry: str
    why: str


#: Tests permitted to leave a watched global changed, matched on the nodeid
#: with any ``[parametrisation]`` suffix removed so that a new parameter does
#: not silently inherit somebody else's licence.
#:
#: EMPTY, and measured empty rather than assumed so: with the guard installed,
#: the whole suite runs clean in all three supported environments (the two jax
#: cells and the zero-dep lane). Every test in this tree that mutates one of
#: these four restores it. The list is here because the incident's polluter
#: would have had to appear in it, and a diff adding a line here is a review
#: this repository can have — whereas a diff that merely omits a `finally:` is
#: not visible anywhere.
PINNED_EXEMPTIONS: tuple[Exemption, ...] = ()


def _bare(nodeid: str) -> str:
    return nodeid.split("[", 1)[0]


def exempt(nodeid: str, entry: str) -> Exemption | None:
    bare = _bare(nodeid)
    for e in PINNED_EXEMPTIONS:
        if _bare(e.nodeid) == bare and e.entry == entry:
            return e
    return None


# ── the decision, kept out of the fixture so it can be driven directly ──────


def read_state() -> dict[str, object]:
    """One fingerprint of every declared entry."""
    return {e.name: e.read() for e in ENTRIES}


def changed(before: dict[str, object], after: dict[str, object]) -> list[Entry]:
    """The entries whose reading moved, licence or no licence."""
    return [e for e in ENTRIES if before.get(e.name) != after.get(e.name)]


def offences(
    nodeid: str, before: dict[str, object], after: dict[str, object]
) -> list[str]:
    """Entries this test changed and was not licensed to change.

    A pure function of three values on purpose: the fixtures below are wrappers
    around it, and the exemption logic is testable without planting a polluting
    test in a nested session.
    """
    return [
        f"  {entry.name}: {before.get(entry.name)!r} -> {after.get(entry.name)!r}\n"
        f"      ({entry.what})"
        for entry in changed(before, after)
        if exempt(nodeid, entry.name) is None
    ]


def render(nodeid: str, found: list[str], subject: str = "test") -> str:
    """The report. ``subject`` is the altitude that measured it.

    Same first half either way — the entries, and why a green run is the
    dangerous outcome — and a different remedy, because a module has no nodeid
    for :data:`PINNED_EXEMPTIONS` to license.
    """
    remedy = (
        "Restore it (a `finally:`, a fixture, or `monkeypatch`), or, if "
        "leaving it changed is genuinely what the test is for, name the test "
        "and the entry in tests/_state_guard.py's PINNED_EXEMPTIONS with a "
        "reason."
        if subject == "test"
        else "Nothing INSIDE this module was named for these entries, so the "
        "change happened outside every test in it — a module- or class-scoped "
        "fixture that set something up and did not tear it down, or a "
        "module-level statement with no matching restore. Put it back at the "
        "same scope (`yield` then restore, in the fixture that changed it). "
        "There is no exemption list at module scope: PINNED_EXEMPTIONS "
        "licenses a TEST, and a module leaves its change to every test that "
        "follows the whole file."
    )
    return (
        f"{nodeid} changed process-global state and did not put it back.\n"
        + "\n".join(found)
        + "\n\n"
        f"This is measured before and after THIS {subject}, so this {subject} "
        f"is the one that changed it — not a later {subject} that inherits it. "
        "State left "
        "behind here is inherited by every test that follows, and the failure "
        "mode is not a red suite: it is a GREEN one. A battery in "
        "tests/test_tripwire_arm.py was satisfied for two audit rounds by a "
        "background branch a test in the same file had left armed.\n\n" + remedy
    )


#: Entry names a function-scoped guard has already DECIDED about within the
#: module currently running — reported as an offence or licensed by an
#: exemption. :func:`module_state_guard` skips them, so a test's offence is
#: named once, at the altitude that can name a test, and an exemption is not
#: undone one scope out. Cleared at every module's setup.
_DECIDED_HERE: set[str] = set()


@pytest.fixture(autouse=True)
def state_guard(request):
    """Fingerprint :data:`ENTRIES` around every test; fail the test that moved one.

    Autouse and function-scoped, and registered from ``tests/conftest.py`` so
    that it covers ``tests/`` and everything under it.

    ORDERING IS WHY THIS IS A CONFTEST FIXTURE RATHER THAN A HOOK. pytest sets
    autouse fixtures up outermost-first — session, then module, then function,
    and within a scope the conftest's before the test module's — so a
    module-scoped ``_x64`` fixture is already in force when this reads
    ``before``, and is torn down after this reads ``after``. That is what makes
    a module which sets x64 for its own duration silent HERE; it is also what
    made such a module invisible when it did NOT put it back, which is what
    :func:`module_state_guard` is for.
    """
    before = read_state()
    yield
    after = read_state()
    _DECIDED_HERE.update(e.name for e in changed(before, after))
    found = offences(request.node.nodeid, before, after)
    if found:
        pytest.fail(render(request.node.nodeid, found), pytrace=False)


@pytest.fixture(autouse=True, scope="module")
def module_state_guard(request):
    """The same instrument one scope out: fail the MODULE that moved one.

    See "TWO ALTITUDES" in the module docstring for the measurement this
    exists on, and for why it reports only entries no function-scoped guard
    inside this module already decided about.

    A module-scoped fixture cannot be exempted by nodeid, and that is
    deliberate rather than an omission: :data:`PINNED_EXEMPTIONS` licenses a
    TEST to leave a global changed, and a module that leaves one changed for
    every test that follows it is not the same act. If one is ever genuinely
    wanted, it needs its own list and its own argument.
    """
    _DECIDED_HERE.clear()
    before = read_state()
    yield
    after = read_state()
    found = [
        f"  {entry.name}: {before.get(entry.name)!r} -> {after.get(entry.name)!r}\n"
        f"      ({entry.what})"
        for entry in changed(before, after)
        if entry.name not in _DECIDED_HERE
    ]
    _DECIDED_HERE.clear()
    if found:
        pytest.fail(
            render(request.node.nodeid, found, subject="module"), pytrace=False
        )

