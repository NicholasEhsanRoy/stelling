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
says. It does that by tracking WHERE THE STATE IS WHENEVER NO TEST IS RUNNING —
see the comment above :data:`_TRAJECTORY` — so a test's own window simply never
enters the outer reading. A test's offence is named at the altitude that can
name a test; what moved between those windows happened outside every test, and
only that is the module's. Both altitudes fire when both offences are real:
a module fixture that leaks AND a test that leaks are two reports, because
they are two acts. **THAT SENTENCE HAS ONE EXCEPTION AND IT IS AN ``xfail``** —
pytest absorbs the guard's teardown failure along with every other failure of
an xfail-marked test, so such a test's own leak is named at neither altitude.
It is in LIMITS below, with the measurement.

**THAT SENTENCE WAS FALSE FOR ONE VERSION AND THE WAY IT WAS FALSE IS THE
LESSON.** The first implementation filtered by entry NAME: whatever a
function-scoped guard saw move was recorded and skipped one scope out. The set
was updated unconditionally and before any report was computed, and
``env:STELLING_*/JAX_*`` is a single entry covering every prefixed variable —
so **adding the** :data:`PINNED_EXEMPTIONS` **entry this guard's own report
prints as the remedy turned the outer guard off**, for a wider leak than the
one being licensed, silently, with the process still polluted at session
finish. Following an instrument's printed instructions must not be a way to
stop the instrument, and a suppression keyed on something coarser than the
offence always is. The controls for that direction are
``test_a_module_leak_survives_the_PINNED_EXEMPTION_of_a_test_in_it`` and
``test_a_module_leak_survives_an_XFAILING_polluter_in_the_same_module``.

────────────────────────────────────────────────────────────────────────────
WHAT IT WATCHES — the inventory, and it is enumerated
────────────────────────────────────────────────────────────────────────────

:data:`ENTRIES`. Five, named by the incident: the identity of jax's registered
const-fold rule, the tripwire's installation record, the dunder perimeter's
installation, ``jax_enable_x64``, and the ``STELLING_*`` / ``JAX_*``
environment.

The perimeter entry is the newest and it was added after the fact, which is
the lesson rather than the footnote: 0.2.0's third instrument rebinds 39
dunder slots on two foreign C-extension types and registers owners in a
module-global list, and none of that was watched by anything here for the
whole of the batch that built it.

The environment entry is a **prefix rule and not a list**, on purpose. Every
other inventory in this repository that enumerated its own domain has had to
be widened after something outside it went unwatched; a key added to
``src/stelling`` tomorrow is watched here today without anybody remembering to
come back. The other four have no such spelling and are enumerated.

READERS TOLERATE ABSENT DEPENDENCIES. The zero-dep lane runs this too: with no
jax installed the four jax-shaped readers return :data:`ABSENT`, which is a
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
* **The dunder perimeter's COUNTERS**, for the same reason one instrument
  over: ``perimeter.CHECKS``, ``FINDINGS``, ``INTERNAL_ERRORS`` and
  ``PERMITTED`` move on every guarded operation any test performs, so they are
  a running total rather than a fingerprint. What ``perimeter:installed``
  watches is the installation, the owners and the live slot bindings — the
  state whose staleness makes the instrument lie. ``prop_guard._TARGET_CACHE``
  is watched only for keys memoised to ``None``; its size grows legitimately.
* **The warnings filter registry, logging configuration, the ``random`` and
  ``numpy.random`` global generators, the process CWD, open file descriptors,
  ``linecache``, and hypothesis's own database and profile registration.**
* **Environment keys outside the two prefixes** — ``PATH``, ``HOME``,
  ``PYTHONPATH``, ``CANARY_PARENT_STELLING``.
* **Anything a test changes and changes back within its own body.** This is a
  before/after fingerprint, not a trace: a test that arms the tripwire and
  disarms it is silent here, which is the intent.
* **A leak by an** ``xfail``**-marked test**, and it is the one hole in "named
  at the altitude that can name a test". ``pytest.mark.xfail`` turns every
  failure of the marked test into an expected one, the guard's teardown
  ``pytest.fail`` included, so the FUNCTION altitude's report is absorbed; and
  the MODULE altitude cannot pick it up either, because a test's own window is
  outside the trajectory by construction — which is the same property that
  makes "named exactly once" true. So the offence is named NOWHERE, and
  ``test_a_module_leak_survives_an_XFAILING_polluter_in_the_same_module``
  plants exactly that shape and pins it: the module's separate leak is named,
  the xfailing test's is not. Not a regression — the entry-name version
  absorbed it too, and worse, since the absorbed report still blinded the
  outer guard. **UNCONDITIONALLY, and it was conditional until 2026-08-22**:
  that control's module fixture acts at SETUP, before the xfailing test runs,
  so it never exercised the case where the module's own out-of-test move comes
  AFTER the absorbed leak — where the whole-reading fold carried the leak into
  the outer report.
  ``test_an_XFAILING_polluters_leak_is_named_NOWHERE_even_if_the_module_moves``
  is the same plant with the module's act moved to teardown, and it is what
  holds this entry to being a measurement now.

  HARMLESS IN THIS TREE TODAY, AND MEASURED RATHER THAN HOPED. Grepped, one
  collected test carries the marker —
  ``tests/property/test_oracle.py::test_a_verified_is_true_at_every_admitted_point``;
  every other ``@pytest.mark.xfail`` and the one ``pytest.xfail()`` call are
  inside plant SOURCE STRINGS in ``tests/test_skip_inventory.py`` and
  ``tests/test_state_guard.py``, or in prose. A probe reading
  :func:`read_state` around every test of that module (a
  ``pytest_runtest_protocol`` wrapper; jax 0.11.0, hypothesis 6.165.10)
  reports ``(none)`` over ``2 passed, 1 xfailed``. The altitude that could see
  such a leak is a report-reading HOOK rather than a fixture raising into one,
  which is a different instrument from this file.
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
* **A statement in a TEST MODULE'S BODY, at import time.** Same reason, one
  step earlier: pytest imports every test module during COLLECTION, before any
  fixture of any scope is set up, so a column-0
  ``os.environ["STELLING_X"] = "1"`` is already in force when this module's
  ``before`` is read and the module reads identical either side. Driven: such
  a module gives ``2 passed``, exit 0, with the key live at session finish.
  The module guard's report used to offer *"a module-level statement with no
  matching restore"* as a thing to go and look for — an explanation it is
  incapable of having found — and it no longer does. There is no such
  statement in this tree (grepped); the altitude that would see one is the
  session guard above, whose ``before`` would have to be read at collection
  start.

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
import sys
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
    return {
        "rule_name": adapter.rule_name(),
        "rule_hash": adapter.rule_hash(),
        "registry_size": adapter.registry_size(),
        "is_armed": adapter.is_armed(),
    }


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
    return {
        "installed_keys": tuple(sorted(installed)),
        "wrapper_id": id(installed.get("wrapper")),
        "original_id": id(installed.get("original")),
        "detached_keys": tuple(sorted(detached)),
        "detached_entry_id": id(detached.get("entry")),
    }


def _binding(owner_type, slot, original, wrapper) -> str:
    """Which of the three known objects is live on ``slot`` right now."""
    live = owner_type.__dict__.get(slot)
    if live is wrapper:
        return "wrapper"
    if live is original:
        return "original"
    return "foreign"


def _read_perimeter() -> object:
    """The dunder perimeter's installation, its owners, and the LIVE bindings.

    THIS BATCH REBINDS DUNDER SLOTS ON TWO FOREIGN C-EXTENSION TYPES, which is
    exactly the class of process-global state this inventory exists to name,
    and it had no entry until this fixup: not `perimeter._installed`, not
    `_owners`, not the 39 live slot bindings, and not the predicate's lazy
    caches. `ci.yml`'s `random-order` lane annotates a shuffled failure the
    guard did not name as "state outside that inventory", so an uninventoried
    perimeter made that annotation say the wrong thing about the most
    order-sensitive thing in the tree.

    THE LIVE BINDING AND THE RECORD, both, and it is the same separation
    `_read_tripwire_installation` makes for the same reason: a record cleared
    while the object stays live, and an object replaced while the record still
    names it, are two different incidents and each is invisible in the other's
    reading.

    WHICH IDENTITY IS READ IS THE WHOLE DESIGN OF THIS ENTRY, and reading the
    wrong one makes it fire on every well-behaved test. The WRAPPER is a fresh
    closure on every `arm()`, so a test that disarms and re-arms -- or a
    fixture that hands a session's hold back the way
    `tests/test_narrowing_perimeter.py::_isolate` does -- produces a new
    wrapper object for an identical state, and `id(wrapper)` would report 58
    offences in that one file alone. What is load-bearing is the SAVED
    ORIGINAL: it is jax's own function, it is what `disarm()` puts back, and a
    re-arm that captured the wrong one is the defect
    `test_arm_disarm_arm_returns_to_the_original_object` exists for. So the
    original is read by identity and the live attribute is read as WHICH of
    the two it is -- `wrapper`, `original` or `foreign` -- which is exactly
    the three states `live_check()` separates and is stable under a
    restore-to-equivalent.

    THE TYPES ARE READ OFF THE INSTALLATION RECORD RATHER THAN LOCATED. This
    reader runs before and after every test in the session, and
    `adapter.perimeter_locate("tracer")` costs a fresh `make_jaxpr` -- roughly
    9,000 traces over a full suite, plus a trace-cache entry each. A face that
    is not installed reads `None`, which is a value like any other and is
    exactly what the transition into and out of armed has to look like.

    AND THE PREDICATE'S LAZY CACHES, because they are how this instrument goes
    blind without moving a single slot: a test that caches a STAND-IN into
    `_JNP`/`_ML`/`_JAX` -- measured, and disclosed in
    `tests/test_tripwire_record.py::_stub_jax` -- leaves every later
    `classify()` declining with an internal error and the perimeter dead for
    the rest of the process.

    WHAT IS READ THERE IS "IS ANYTHING FOREIGN IN IT", NOT THE IDENTITIES,
    and that is measured rather than tidy. These three are lazily bound, so
    `None -> the real module` is a legitimate one-time transition that
    whichever test first touches the guard performs; an entry reading `id()`
    reports that test as a polluter, which is how an instrument becomes
    something people suppress. Driven, before the reading was narrowed::

        perimeter:installed: (..., (10750112, 10750112, 10750112), ...)
                          -> (..., (129825609081344, ...), ...)

    -- three `id(None)`s becoming three modules, on a planted test that armed
    and released cleanly. So each cache is compared against what `sys.modules`
    holds for it and only a MISMATCH is named: unbound and correctly-bound are
    the same reading, and a `SimpleNamespace` in `_JNP` is not.

    `_TARGET_CACHE`'s contents grow legitimately as new dtypes are seen, so its
    size is NOT read either; what is read is the set of keys memoised to
    `None`, which is a permanent blindness on that key and can only arrive
    from a fault the module has since fixed (`prop_guard.py` edit 5).
    """
    from stelling import _optional

    if not _optional.available("jax"):
        return ABSENT
    from stelling._tripwire import perimeter, prop_guard

    faces = []
    for face in perimeter.FACES:
        entry = perimeter._installed.get(face)
        if entry is None:
            faces.append((face, None))
            continue
        owner_type = entry["type"]
        faces.append((
            face,
            f"{owner_type.__module__}.{owner_type.__qualname__}",
            tuple(
                (slot, id(original), _binding(owner_type, slot, original, wrapper))
                for slot, (original, wrapper) in sorted(entry["slots"].items())
            ),
        ))
    return {
        "faces": tuple(faces),
        "owners": tuple(id(held) for held in perimeter._owners),
        "foreign_module_caches": tuple(
            name
            for name, cached, module in (
                ("_JNP", prop_guard._JNP, "jax.numpy"),
                ("_ML", prop_guard._ML, "ml_dtypes"),
                ("_JAX", prop_guard._JAX, "jax"),
            )
            if cached is not None and cached is not sys.modules.get(module)
        ),
        "target_cache_memoised_None": tuple(
            sorted(k for k, v in prop_guard._TARGET_CACHE.items() if v is None)
        ),
        "unknown_slots": tuple(sorted(prop_guard.UNKNOWN_SLOTS)),
    }


def _read_jax_enable_x64() -> object:
    from stelling import _optional

    if not _optional.available("jax"):
        return ABSENT
    import jax

    return {"jax_enable_x64": bool(jax.config.jax_enable_x64)}


#: The two prefixes. See the docstring: a rule, not a list.
ENV_PREFIXES = ("STELLING_", "JAX_")


def _read_env() -> object:
    """Every prefixed key, AS A MAPPING — see :func:`_carry` for why the shape
    of a reading is load-bearing and not a presentation choice."""
    return {
        k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)
    }


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
        "perimeter:installed",
        _guarded(_read_perimeter),
        "the dunder perimeter's faces, the type and LIVE binding of each of "
        "its slots, its owner list, and the predicate's lazy module caches",
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


#: Tests permitted to leave a watched global changed. **THE LICENCE IS AS WIDE
#: AS THE NODEID SOMEBODY WROTE, and no wider** — an entry naming
#: ``test_x[a]`` licenses ``test_x[a]`` and nothing else, and an entry naming
#: the bare ``test_x`` licenses every parametrisation of it, which is a choice
#: its author makes by how the entry is spelled.
#:
#: THE STRIPPING USED TO BE UNCONDITIONAL AND ITS REASON WAS INVERTED. This
#: comment read *"matched on the nodeid with any ``[parametrisation]`` suffix
#: removed so that a new parameter does not silently inherit somebody else's
#: licence"* — and removing the suffix is precisely how a new parameter DID
#: inherit it: driven, an exemption written for ``test_x[a]`` licensed
#: ``test_x[b]``, ``EXIT 0`` with both keys leaked. It was a documented
#: behaviour with the opposite justification written over it, in a file whose
#: own argument against the version before this one is that *"a suppression
#: keyed on something coarser than the offence always is"* a way to switch the
#: instrument off. The code is what changed, because the sentence was the
#: right rule: a licence may not travel to a case nobody reviewed.
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


def _licenses(exemption: str, nodeid: str) -> bool:
    """Does an exemption written as ``exemption`` cover ``nodeid``?

    Exactly, unless the exemption names no parametrisation at all — in which
    case it covers the whole family, because that is what its author wrote.
    Both directions are driven in
    ``test_an_exemption_licenses_exactly_its_own_test_and_entry``.
    """
    if "[" in exemption:
        return exemption == nodeid
    return _bare(nodeid) == exemption


def exempt(nodeid: str, entry: str) -> Exemption | None:
    for e in PINNED_EXEMPTIONS:
        if e.entry == entry and _licenses(e.nodeid, nodeid):
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

    Same entries either way, and the same reason a green run is the dangerous
    outcome — and a different HEADLINE and remedy, because the two altitudes
    measured different things and a module has no nodeid for
    :data:`PINNED_EXEMPTIONS` to license.

    **OUTSIDE, NOT "BETWEEN", and the difference is the commonest case.** This
    headline said *"BETWEEN its tests"* until 2026-08-22. The trajectory is
    read at every window boundary the module has, and the first of those is
    module setup — so the ordinary offence, a module fixture that sets
    something up at SETUP and never tears it down, is a move BEFORE the first
    test rather than between two of them. The sentence under it was exact
    ("outside every test of this module"), so no reader was sent to the wrong
    place; the headline was simply narrower than the instrument.

    **THE MODULE MAY NOT BE TOLD IT "DID NOT PUT IT BACK", BECAUSE SOMETIMES
    IT DID.** The function guard reads before and after one test, so that
    sentence is literally what it measured. The module guard reads a
    TRAJECTORY of what moved outside every test, and the state can be back
    where it started when it fires: a module fixture that sets ``K`` at setup,
    a test that deletes ``K``, and a teardown restore that is therefore a
    no-op ends the session CLEAN and is still reported — correctly, because
    the module's own move was never undone by the module, and the clean-up
    belongs to a test that can be skipped, deleted or reordered away. Telling
    that module's author it did not put something back sends them to look at a
    process that looks fine, and a report a reader cannot reproduce is how a
    guard comes to be weakened.
    """
    headline = (
        f"{nodeid} changed process-global state and did not put it back."
        if subject == "test"
        else f"{nodeid} changed process-global state OUTSIDE its tests, and "
        f"nothing outside them put it back."
    )
    measured = (
        f"This is measured before and after THIS {subject}, so this {subject} "
        f"is the one that changed it — not a later {subject} that inherits it. "
        if subject == "test"
        else "This is the trajectory of the moves made OUTSIDE every test of "
        "this module, so no test of this module did it — and the reading on "
        "the right is where those moves left the state, which need not be "
        "where the process is now. A TEST that happens to put it back leaves "
        "the process clean and this still fires, correctly: the module's own "
        "change is unrestored, and the restore belongs to a test that can be "
        "skipped, deleted or reordered away. "
    )
    remedy = (
        "Restore it (a `finally:`, a fixture, or `monkeypatch`), or, if "
        "leaving it changed is genuinely what the test is for, name the test "
        "and the entry in tests/_state_guard.py's PINNED_EXEMPTIONS with a "
        "reason."
        if subject == "test"
        else "This moved OUTSIDE every test of this module, so no test did "
        "it — a module- or class-scoped fixture that set something up and did "
        "not tear it down. Put it back at the same scope (`yield` then "
        "restore, in the fixture that changed it). NOT a module-level "
        "statement at import: those run during collection, before this "
        "reading is taken, so this guard cannot see one and would not have "
        "reported it (see WHAT IT DOES NOT WATCH in tests/_state_guard.py). "
        "There is no exemption list at module scope: PINNED_EXEMPTIONS "
        "licenses a TEST, and a module leaves its change to every test that "
        "follows the whole file."
    )
    return (
        headline + "\n"
        + "\n".join(found)
        + "\n\n"
        + measured
        + "State left "
        "behind here is inherited by every test that follows, and the failure "
        "mode is not a red suite: it is a GREEN one. A battery in "
        "tests/test_tripwire_arm.py was satisfied for two audit rounds by a "
        "background branch a test in the same file had left armed.\n\n" + remedy
    )


# ── the out-of-test trajectory ──────────────────────────────────────────────
#
# WHAT THE MODULE GUARD MEASURES, AND WHY IT IS NOT A SUPPRESSION LIST. Its
# question is "did anything move OUTSIDE every test in this module", and the
# first answer to it was a set of entry NAMES a function-scoped guard had
# touched, skipped at module teardown. Three things were wrong with that, and
# the third is the one that mattered:
#
#   * it was updated from `changed()` unconditionally, BEFORE `offences()` —
#     so a name landed in it whether or not any report was ever delivered;
#   * `env:STELLING_*/JAX_*` is ONE entry covering every prefixed variable, so
#     "the same entry" is very much coarser than "the same offence";
#   * therefore, adding the PINNED_EXEMPTIONS entry that the guard's own
#     report PRINTS AS THE REMEDY silenced a DIFFERENT, WIDER leak. Driven,
#     with a module fixture leaking STELLING_MODULE_LEAK and one test leaking
#     STELLING_TEST_LEAK:
#
#       baseline                          2 passed, 1 error   exit 1
#       + the printed remedy for the TEST 2 passed            exit 0
#         ... and at session finish: ['STELLING_MODULE_LEAK', 'STELLING_TEST_LEAK']
#
#     Following the instrument's printed instructions switched the instrument
#     off. An xfail on the polluting test did the same thing for free.
#
# So the trajectory is tracked instead of names being filtered. `last` is the
# reading at the close of the last window some guard held; `shadow` is where
# the state has been carried by moves observed OUTSIDE any test. A test's own
# window advances `last` and never `shadow`, so what a test does is invisible
# here by construction rather than by suppression — which is what makes "named
# exactly once" true, and what keeps an exemption from being undone one scope
# out. The module is reported iff `shadow` ends somewhere other than where
# `module_before` started.
#
# Exact, and it needs nothing to be cleared conditionally: a module that sets
# a global at setup and puts it back at teardown moves the shadow out and then
# back, and reads silent.
#
# IT HANGS OFF THE SESSION'S `Config` AND NOT OFF THIS MODULE, and that is not
# tidiness. `module_before`, `last` and `shadow` were MODULE-LEVEL dicts,
# cleared at `module_state_guard`'s teardown — so a SECOND pytest session in
# the SAME PROCESS that loaded this module cleared the OUTER session's
# bookkeeping halfway through a module. Driven, with a module fixture leaking
# one key and `test_one` running
# `pytest.main([..., "-p", "_state_guard", inner])`:
#
#     module-level dicts   EXIT 0   silent, and the key live at session finish
#     on the Config        EXIT 1   naming the module
#
# and the other direction was worse to read: a module that DID put its change
# back was reported for moving `()` to `()`, because `module_before.get(name)`
# had become `None` under it. A report a maintainer cannot reproduce is how a
# guard gets weakened. It was also a STRICT REGRESSION from the entry-name
# filtering this replaced, which over-reported on the same plant and so failed
# SAFE where this failed OPEN.
#
# A nested session builds a fresh `Config`, so the separation is BY
# CONSTRUCTION — the same argument the trajectory itself is sold on, one level
# up — rather than by anybody remembering to key a module-level dict on the
# session. The two controls are
# `test_a_module_leak_is_still_named_when_a_TEST_RUNS_A_NESTED_SESSION` and
# `test_a_WELL_BEHAVED_module_stays_silent_when_a_test_runs_a_nested_session`.
_TRAJECTORY: pytest.StashKey[dict[str, dict[str, object]]] = pytest.StashKey()


def _trajectory(config) -> dict[str, dict[str, object]]:
    """This session's trajectory: ``module_before``, ``last`` and ``shadow``.

    Created on first use and never shared: two sessions in one process have
    two ``Config`` objects and therefore two of these.
    """
    return config.stash.setdefault(
        _TRAJECTORY, {"module_before": {}, "last": {}, "shadow": {}}
    )


#: A part of a reading that the move being folded did not have. Distinct from
#: ``None``, which a reader may legitimately produce for a part.
_GONE = object()


def _carry(previous: object, moved_from: object, moved_to: object) -> object:
    """``previous``, carried along the move ``moved_from -> moved_to``.

    PART-WISE WHEN THE READING IS A MAPPING, AND THAT IS THE WHOLE OF THE
    ATTRIBUTION. An entry's value is one reading of one piece of global state,
    but the pieces inside it move independently: ``env:STELLING_*/JAX_*`` is a
    prefix rule covering every prefixed variable, so a module setting ONE key
    and a test leaking ANOTHER are two different acts inside one entry. Carry
    the whole reading and the module's move drags the test's leak into
    ``shadow`` with it, and the module is then reported for a key a test set —
    which is the same coarseness that made the first, entry-NAME version of
    this guard licence a wider leak than the one being licensed. So the fold
    is per KEY: only the keys this move actually changed are written through,
    and a key the move left where it found it is left where ``previous`` had
    it, whoever put it there.

    Anything that is not a mapping is one opaque part and carries whole, which
    is the exact previous behaviour: :data:`ABSENT` and :class:`Unreadable`
    are values like any other and stay values like any other.

    IT DOES NOT MAKE THIS A TRACE. A part whose reading is the same on both
    sides of a gap records no move, exactly as a whole reading did — a module
    that writes a part the value a test had already written it is invisible
    here, for the same reason a test that changes something back within its own
    body is. Finer granularity, same rule.
    """
    if not (
        isinstance(previous, dict)
        and isinstance(moved_from, dict)
        and isinstance(moved_to, dict)
    ):
        return moved_to
    carried = dict(previous)
    for key in set(moved_from) | set(moved_to):
        was = moved_from.get(key, _GONE)
        now = moved_to.get(key, _GONE)
        if was == now:
            continue  # this move did not touch this part
        if now is _GONE:
            carried.pop(key, None)
        else:
            carried[key] = now
    return carried


def _outside_a_test(
    traj: dict[str, dict[str, object]], reading: dict[str, object]
) -> None:
    """Fold a reading taken outside any test into ``traj``, part by part.

    **THE `!=` IN FRONT OF THE FOLD IS A PREDECESSOR GUARD AND IS REDUNDANT
    IN EFFECT FOR EVERY READING IN THIS TREE**, said here because the
    previous version of this docstring did not say it and because
    ``_matrix_include`` in ``tests/_lanes.py`` sets the precedent for saying
    it. It was the whole of the fold before :func:`_carry` became part-wise:
    a reading that had not moved was skipped, and one that had was written
    through whole. Now that the fold is per KEY, an unmoved MAPPING carries
    to a copy of what ``shadow`` already held — ``_carry`` ``continue``s on
    every key whose two sides agree — so the guard decides nothing. All five
    :data:`ENTRIES` read as mappings today. Driven: replacing the condition
    with ``if True:`` leaves ``tests/test_state_guard.py`` at **29 passed**.

    IT STAYS, AND NOT ONLY AS A CHEAP GUARD IN FRONT OF A LOOP. A reading
    that is NOT a mapping — an :class:`Unreadable` from a reader that raised,
    or :data:`ABSENT` — is one opaque part and carries WHOLE, so for those
    ``_carry`` would return the unmoved value and overwrite the predecessor
    ``shadow`` is holding on the module's behalf. That is reachable only when
    a test moves such an entry and is exempted from the function-scope guard
    for it, and :data:`PINNED_EXEMPTIONS` is empty today, which is why no
    test distinguishes the two. Redundant in effect, not redundant in
    argument.
    """
    last, shadow = traj["last"], traj["shadow"]
    for name, value in reading.items():
        if last.get(name) != value:
            shadow[name] = _carry(shadow.get(name), last.get(name), value)
    last.clear()
    last.update(reading)


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

    It also maintains the outer guard's trajectory: this test's ``before`` is
    the first reading since the last window closed, so any difference happened
    outside every test, and this test's ``after`` is where the next window
    starts from.
    """
    traj = _trajectory(request.config)
    before = read_state()
    _outside_a_test(traj, before)
    yield
    after = read_state()
    traj["last"].clear()
    traj["last"].update(after)
    found = offences(request.node.nodeid, before, after)
    if found:
        pytest.fail(render(request.node.nodeid, found), pytrace=False)


@pytest.fixture(autouse=True, scope="module")
def module_state_guard(request):
    """The same instrument one scope out: fail the MODULE that moved one.

    See "TWO ALTITUDES" in the module docstring for the measurement this
    exists on, and the comment above :data:`_TRAJECTORY` for why it reports a
    TRAJECTORY rather than filtering by entry name.

    A module-scoped fixture cannot be exempted by nodeid, and that is
    deliberate rather than an omission: :data:`PINNED_EXEMPTIONS` licenses a
    TEST to leave a global changed, and a module that leaves one changed for
    every test that follows it is not the same act. If one is ever genuinely
    wanted, it needs its own list and its own argument.
    """
    traj = _trajectory(request.config)
    before = read_state()
    for shared in traj.values():
        shared.clear()
        shared.update(before)
    yield
    after = read_state()
    _outside_a_test(traj, after)  # module teardown is outside every test too
    shadow, module_before = traj["shadow"], traj["module_before"]
    found = [
        f"  {entry.name}: {before.get(entry.name)!r} -> {shadow.get(entry.name)!r}\n"
        f"      ({entry.what})"
        for entry in ENTRIES
        if shadow.get(entry.name) != module_before.get(entry.name)
    ]
    for shared in traj.values():
        shared.clear()
    if found:
        pytest.fail(
            render(request.node.nodeid, found, subject="module"), pytrace=False
        )

