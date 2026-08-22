# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""``N skipped`` is a number nothing asserts. This asserts it — by CONDITION.

A skipped test is a test that did not happen, reported in the same green line
as the ones that did. Two drifts hide in there and neither shows up:

* a **new** skip. The suite goes from 2 skipped to 3 and no eye catches it.
  This is not hypothetical: ``test_any_pytree.py``'s two ``any_pytree``
  acceptance tests — the ones holding the hash-equality bar for "faithful
  sugar" against a real third-party library — skipped in *every* CI run the
  project has ever had, because ``blackjax`` was declared nowhere.
* a skip that **stops skipping**, or a gate that fires when its dependency is
  right there. Equally silent, equally wrong.

**A count is the weak form and is exactly what let this drift.** Counting
cannot survive the fact that the skip set is ENVIRONMENT-DEPENDENT: the
zero-dep CI job skips whole modules on the jax gate, the jax job skips the
``maddening`` and ``jaxfluids`` gates, the blackjax acceptance job skips
neither of the two tests above, and a contributor with no solver wheels skips
a different set again. Any of those numbers hardcoded is wrong in the other
four environments, and a pin that has to be special-cased per environment gets
weakened until it means nothing.

So this pins the CONDITION, never the count. Every entry says *this test
skips iff <predicate>*, and the predicate is evaluated here, now, in whatever
environment this is:

* if the predicate holds, the test MUST have skipped;
* if it does not, the test MUST have run.

Both directions, and the same file is told which environment it is in by
nothing but the predicates it evaluates.

Four shapes of disclosure, in descending strength:

1. :data:`PINNED` — a named test, a named condition, both directions asserted.
2. :data:`RULES` — a skip *reason* that any test may legitimately carry, with
   the condition that makes it legitimate. Reason-keyed rather than
   test-keyed because "needs z3" governs a dozen tests and enumerating them
   would rot on the first rename. A rule excuses the EXACT reasons it names
   and there is no pattern language, because every bound on a pattern turned
   out to be a list of the ways somebody had already been broad; and the
   reasons it names must be written **in this file**, because ``reasons`` is
   an ordinary Python value and a rule that computed it out of another
   module's source excused everything that module skipped with while this
   file said nothing.

   What that guarantees, exactly: a skip is excused only by a string typed on
   the disclosure surface, next to a condition in prose, in the diff a
   reviewer reads. It does not guarantee the condition is true or that anyone
   checked — ``legitimate=lambda: True`` is a legal rule and this file will
   honour it. Disclosure is legibility, not justification.
3. :data:`DECLARED_OPTIONAL_DEPENDENCIES` — the libraries stelling's tests may
   gate on with ``importorskip``. A gate on anything else is undisclosed and
   fails here.
4. :data:`MEASURED` — the weakest, and the one the other three cannot express:
   a skip whose condition is a MEASUREMENT the test makes on its own subject
   ("interval propagation settled the obligation, so there is no solver
   verdict to check"). Not dependency-dependent, so no predicate here can
   evaluate it; test-keyed and exact-reason so that the excuse cannot travel
   to any other test or any other reason.

Anything a session skips that none of the four covers is a failure, with the
ways to fix it named in the message.

**Scope and ORDER, asserted rather than assumed.** The completeness half
claims something about *the suite*, and it can only claim it about a session
that collected the suite, ran it, and ran it before this pin. A session
narrowed by an explicit path, by ``--lf`` or by ``-k`` prints a green line that
reads exactly like a whole run's — and so does a whole run in a different
ORDER, which is the one that bites, because this pin reads outcomes and a pin
that runs first has no outcomes to read. ``pytest --nf``, an installed
pytest-randomly, a plugin that drops items from ``items[:]``, and
``--deselect tests/test_skip_inventory.py`` were each measured to hold an
undisclosed skip at exit 0 while the completeness half looked green.

An eighth route was found after those seven were closed, and this file's own
rule is what condemns it: ``@pytest.mark.xfail(run=False)`` reports through the
SKIP channel, the recorder dropped it, and the claim was MADE over a test that
never ran. pytest discloses it — ``1 xfailed`` — so the answer is the same one
``N deselected`` gets: WITHDRAWN. See :data:`conftest.XFAILED`.

**A ninth and a tenth are not that kind of thing at all, and the difference is
worth more than the two routes are.** The first eight are questions about a
REPORT — which channel does this outcome arrive down, and does the recorder
read that channel. These two record the skip perfectly, in
``conftest.SKIPPED``, under its right nodeid and its right reason, and then
nobody looks:

* ``pytest.exit(reason, returncode=0)`` from inside the run loop. ``Exit``
  propagates through the recorder's ``pytest_runtestloop`` wrapper, so the
  session-end decision never runs, and pytest assigns the process exit code
  from the returncode, which overrides the one the decision would have set.
  Measured at a80d60c, same tree, same plant, one ``-p`` apart: byte-identical
  summary lines, exit 1 with a banner and exit 0 with nothing.
* an undisclosed skip in THIS FILE, after this pin has already claimed. The
  pin runs last among FILES, not among TESTS; the guard that would catch it is
  disarmed by the pin's own claim. Measured at a80d60c on the whole tree:
  ``2022 passed, 3 skipped``, exit 0, no banner, the planted skip on screen.

Both are answered in ``tests/conftest.py`` — ``pytest_sessionfinish`` and
``_close_the_session`` respectively. The general lesson is about the taxonomy
and not about the two entries: a table of channels bounds what can be
MISREPORTED and says nothing about whether anybody read it.

So both are recorded in ``tests/conftest.py``, where the collection and the
run are, and the whole decision is made in
:func:`the_claim_this_session_can_make`, where the claim is. Four answers, one
per shortfall, listed above that function. Whatever the session did see is
checked either way — that half is the same size in every session, and it is
the half that catches an undisclosed skip in a narrowed one. The cases the pin
cannot report on at all (it was reordered early, filtered out, deselected, or
``--ignore``d) are answered by the conftest at the end of the session, which is
the one place an invocation cannot reach.

The session's outcomes come from ``tests/conftest.py``, which records them as
pytest reports them; see there for why this cannot be a static read of the
tree.

What "correct in every environment" has actually been measured to mean: this
file is exercised in the zero-dep lane, the jax lane, the solvers lane, a
no-solver lane (jax present, neither wheel installed) and the blackjax
acceptance job. The no-solver lane is the one that caught :data:`MEASURED`
missing: ``test_escalation_invariant``'s data-dependent skip fires there and
in no other lane, and until it was disclosed this file reported it as drift.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Callable

import pytest

from conftest import (
    CLAIM_MADE,
    RAN,
    SEEN_FILES,
    SKIPPED,
    USER_FILTERS,
    XFAILED,
    _reason,
    colliding_basenames,
    deselected_items,
    deterministic_order_args,
    pending_items,
    unseen_files,
)

from stelling import _optional

REPO = pathlib.Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

# This module, by the substring the recorder keys on. The pin's own module gets
# excluded from two questions — "is anything still owed?" and "was evidence
# deselected?" — because deselecting the pin removes the CLAIM (which
# tests/conftest.py then makes at the end of the session) while deselecting
# anything else removes the EVIDENCE, and nothing recovers a test that never
# ran.
_THIS_MODULE = "test_skip_inventory.py"

# Reasons this file's own pin skipped with, this session. The pin withdraws by
# skipping, and its withdrawal is a skip like any other to the recorder: the
# session-end check would otherwise read the pin's own "claim withdrawn" as an
# undisclosed skip and fail the session for it.
_OWN_WITHDRAWALS: set[str] = set()


def _importable(name: str) -> bool:
    """Exactly what the INSTALLED pytest calls absence — which moved in 9.1.

    ``pytest.importorskip``'s ``exc_type`` used to default to ``ImportError``;
    since pytest 9.1 it defaults to ``ModuleNotFoundError``. The difference is
    a package that is installed but BROKEN — a C extension that will not load,
    an ``__init__`` that raises: it used to skip, and now the error propagates
    and the test ERRORS instead.

    So "absent", for the gates this file judges, now means
    ``ModuleNotFoundError`` and nothing else, and this predicate has to say the
    same. A version of it that caught plain ``ImportError`` would be the LOOSER
    of the two and would demand a skip pytest will never produce: against a
    broken ``blackjax`` it reported the pinned test as "must skip", got an
    error instead, and blamed a missing dependency that was right there.

    Installed-but-broken therefore reads as PRESENT here, deliberately: the pin
    then demands a run, the run errors, and the error is what says the package
    is broken. That is the true cause, and it is the one the failure names.

    Pinned against the installed pytest by
    ``test__importable_agrees_with_the_installed_importorskip``.
    """
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        return False
    except ImportError:
        return True
    return True


def _wheel(name: str) -> bool:
    """z3 / cvc5 as a python wheel, by the SAME predicate the gates use.

    Every reason-carrying solver gate in this suite is a
    ``skipif(not _optional.available(...))``, i.e. ``find_spec``, so this is
    not an independent opinion about presence — it is the gates' own, which is
    what makes it safe to contradict them with. In particular an
    installed-but-broken wheel reads as present in both places at once, so the
    two cannot disagree about it and no false accusation is possible there.
    (Do not "fix" this into an import: that would make this predicate the
    stricter of the two and it would start calling legitimate skips wrong.)

    Deliberately NOT counting the cvc5 binary fallback, which two modules OR
    into their ``HAVE_CVC5``. This predicate is only ever used to call a skip
    WRONG, so the weaker form is the safe one: wheel-absent-binary-present
    makes those gates not fire, and a gate that did not fire is not accused.
    """
    return _optional.available(name)


def _git_cannot_read_the_routing_source() -> bool:
    """Whether git can read the history the routing manifest names.

    TWO CONDITIONS, AND THE SECOND IS WHY THIS IS NOT JUST A `git show`.
    `tests/test_soundness_routing.py` skips its git-gated legs when the source
    is unreadable AND the environment is one of the five that legitimately
    cannot answer — an unpacked sdist, an export, a shallow clone that does
    not reach the commit, a tree unpacked inside somebody else's repository,
    or a linked worktree whose parent has moved. When the source is
    unreadable and NONE of those holds, the manifest names a commit that is
    not there and that file FAILS instead: a defect, not an environment.

    So this asks both, and the second comes from
    `_why_the_history_is_out_of_reach` rather than from three probes copied
    here — a PREDICATE borrowed, which is not the thing `RULES` may not do.
    What a rule may not take from another module is its REASONS, because a
    rule that computed those excused everything that module skipped with;
    the reason below is typed here, in the diff a reviewer reads.

    ANY section, not every: a skip carrying that reason means at least one
    section could not be read. Answers False if the import or the probe
    cannot run at all, which is the safe direction — the skip is then
    reported as contradicted rather than quietly excused.
    """
    try:
        from _soundness_routing_manifest import SECTIONS
        from test_soundness_routing import _why_the_history_is_out_of_reach
    except Exception:  # noqa: BLE001 - a predicate may not raise
        return False
    unreadable = False
    for section in SECTIONS:
        try:
            probe = subprocess.run(
                ["git", "show", f"{section.source_commit}:CHANGELOG.md"],
                cwd=REPO, capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            unreadable = True
            break
        if probe.returncode != 0:
            unreadable = True
            break
    return unreadable and _why_the_history_is_out_of_reach() is not None


def _jax_x64_is_on() -> bool:
    """Whether this session runs with 64-bit dtypes enabled.

    Read from jax rather than from ``os.environ``, because a conftest or a
    test module can turn it on without the variable being set, and this
    predicate is used to call a skip WRONG. Answers False when jax is absent,
    which is the safe direction: the gate it governs cannot fire in a
    jax-less lane anyway, since the module gates on jax first.
    """
    if not _optional.available("jax"):
        return False
    try:
        import jax  # noqa: PLC0415 - deliberately lazy; the zero-dep lane has none

        return bool(jax.config.read("jax_enable_x64"))
    except Exception:  # noqa: BLE001
        return False


# --- what may be gated on ----------------------------------------------------

# The libraries a stelling test may reach through ``pytest.importorskip``.
# Every one of them is optional on purpose and named somewhere in the project's
# install story; the point of the list is that a gate on a library NOT here is
# a new dependency arriving quietly through a test, and fails below.
DECLARED_OPTIONAL_DEPENDENCIES = frozenset(
    {
        "jax",  # the [jax] extra; the zero-dep CI job has none
        "numpy",  # arrives with jax, absent in a bare solvers-only install
        "z3",  # the [z3] extra
        "cvc5",  # the [cvc5] extra
        "diffrax",  # corpus/supply probe library, test-only
        "blackjax",  # corpus/supply probe library, test-only (the CI job)
        "maddening",  # the pinned-jax reproducer library, test-only
        "jaxfluids",  # the WENO5 acceptance subject, test-only
        "hypothesis",  # the property suite's driver, dev-group, test-only
    }
)

# pytest 8's wording for a gate that fired. Pinned by
# ``test_the_import_gate_wording_is_the_one_pytest_actually_emits`` — if this
# regex went stale it would stop recognising the single most common skip in
# the suite, and (worse, if it were loose) would disclose everything.
_IMPORT_GATE = re.compile(r"^could not import '([A-Za-z_][\w.]*)'")


# --- the pins: a named test, a named condition, both directions -------------


@dataclasses.dataclass(frozen=True)
class Pinned:
    nodeid: str
    needs: tuple[str, ...]  # skips iff ANY of these is unimportable
    why: str


PINNED = (
    Pinned(
        "tests/test_any_pytree.py::test_h_clean_sugar_hash_equals_hand_declaration",
        ("jax", "diffrax", "blackjax"),
        "the acceptance bar for 'faithful sugar': the sugar declaration must "
        "trace to a content-hash-IDENTICAL query to the hand declaration, "
        "against a real library's pytrees. Gated on jax at module scope and on "
        "diffrax+blackjax in the `probe` fixture (the probe module imports "
        "both). Ran in no CI job until the blackjax job in ci.yml existed.",
    ),
    Pinned(
        "tests/test_any_pytree.py::test_h_hard_sugar_hash_equals_hand_declaration",
        ("jax", "diffrax", "blackjax"),
        "same bar, on the hard case: blackjax's MCLMC state, `wrap_key_data`, "
        "and PRNG key dtypes — the leaf kinds any_pytree refuses and redirects.",
    ),
    Pinned(
        "tests/test_dot_general_both_faces.py::"
        "test_the_oracle_NORMALISES_its_dims_and_does_not_merely_check_them",
        ("numpy",),
        "audit 0.2.0 B6 RE-AUDIT R4's property: `dot_general_geometry` must "
        "return plain `int` dims, because a guard that tests a predicate and "
        "discards the value leaves three protocols downstream — hashing, "
        "ordering, indexing — meeting an object nothing normalised. The "
        "exhibit has to be a 0-d `numpy` array: it satisfies `__index__` AND "
        "is unhashable, which is the pair that made `len(set(dims))` raise a "
        "raw TypeError out of the public `propagate()` while the emission "
        "declined. A hand-rolled unhashable `__index__` object would assert "
        "the same property against a synthetic instance rather than the real "
        "one the defect was found on. numpy is NOT a core dependency — the "
        "core is zero-dep — so it is absent in the solvers-only lane, and "
        "this module stays collectable there (48 of 52 rows still run) rather "
        "than taking `importorskip` at module scope for one transitive "
        "dependency of jax.",
    ),
)


# --- the rules: a skip reason any test may carry, and when it is legitimate --


@dataclasses.dataclass(frozen=True)
class Rule:
    when: str  # the condition, in prose — this is the disclosure
    reasons: frozenset[str]  # the EXACT reasons, and there is no other channel
    # Returns True when a skip carrying this reason is legitimate RIGHT NOW.
    # None means the condition is not computable from outside the test; such a
    # rule discloses the skip but cannot check its direction, and must say why.
    legitimate: Callable[[], bool] | None = None

    def matches(self, reason: str) -> bool:
        """Exact membership. A rule excuses what it NAMES and nothing else.

        This used to accept a regex, and every attempt to bound the regex was
        bounded in turn by a list of decoy strings — which is not a bound, it
        is a list. Three of them, in order, each defeated:

        * ``re.search`` excused anything that CONTAINED the disclosed reason,
          so ``"totally other cause; XLA did not contract this form on this
          build"`` was covered;
        * ``re.fullmatch`` fixed that and ``r".*"`` walked through it, so a
          decoy list was added;
        * the decoy list was passed unchanged by ``pattern=r"skip: .*"``,
          which excuses every reason beginning ``"skip: "`` and resembles
          none of the decoys. Measured: it passed all three halves.

        A pattern is a way to excuse text nobody wrote down. So there is no
        pattern. The one rule whose variable part is genuinely variable — the
        parametrised XLA form name — enumerates it by READING THE FORM NAMES
        out of the module that emits them, so its excuse set is still exactly
        the set of reasons that site can produce. See
        ``test_a_rule_can_only_excuse_a_reason_someone_wrote_down``.
        """
        return reason in self.reasons


# The one genuinely variable reason in the suite, and the enumeration of it.
#
# `tests/test_three_rows_acceptance.py` skips with
# `f"{name}: XLA did not contract this form on this build"`, where `name` is
# the parametrised contraction form ("nested jit", "broadcast+slice+squeeze",
# …). The names are READ OUT of that module rather than described by a regex,
# because a regex for "a form name" is a regex for "some prose", and the one
# that shipped — `r"[\w+ .\-]+: XLA did not contract this form on this build"`
# — admitted any punctuation-free prefix at all. Measured: a skip reading
# "the solver crashed and we gave up: XLA did not contract this form on this
# build" was excused, and the whole suite stayed green with it planted.
#
# Reading the source gets both edges at once: nothing but a real form name is
# excused, and every real form name IS excused — so a runner whose XLA folds
# differently does not read as inventory drift, and a form added with a comma
# in its name needs no pattern change at all.
_FORM_TABLE = TESTS / "test_three_rows_acceptance.py"
_XLA_SUFFIX = ": XLA did not contract this form on this build"


def _contraction_form_names() -> tuple[str, ...]:
    """The parametrised form names, from the module that emits the skip."""
    if not _FORM_TABLE.exists():
        return ()
    return tuple(
        re.findall(r'^\s*\("([^"]+)",\s*lambda u, w', _FORM_TABLE.read_text(), re.M)
    )


# THE ONE DERIVATION. Every other reason in this file is a literal typed here;
# these are read out of the module that emits them, because the form names are
# genuinely variable. It is hoisted out of the rule and given a name so that
# ``test_a_rule_excuses_only_reasons_written_down_in_this_file`` can tell "the
# declared derivation" from "some other module's source", which is the shape
# that walked through the bound this file claimed to have:
#
#     Rule(when="…read out of the module that emits them…",
#          reasons=frozenset(_reasons_declared_by(TESTS / "test_affine.py")),
#          legitimate=lambda: True)
#
# — the same idiom the XLA rule uses, pointed at a module full of skips nobody
# disclosed. Measured at bd1fa04 with `pytest.skip("we will get back to this
# one after the release")` planted in test_affine.py, a string appearing
# NOWHERE in this file: the planted skip on the screen, no rule here naming it,
# and the whole suite EXIT 0. `reasons` is a runtime value and
# nothing required it to be literals, so "the reason must be typed where a
# reviewer sees it" was prose, not a bound. It is a bound now.
_DERIVED_REASONS = frozenset(
    f"{name}{_XLA_SUFFIX}" for name in _contraction_form_names()
)


RULES = (
    Rule(
        when="neither solver wheel is installed",
        reasons=frozenset(
            {
                "needs an SMT solver",
                "needs a solver",
                "no SMT solver installed",
                "block opts into solver escalation; no backend installed",
            }
        ),
        legitimate=lambda: not (_wheel("z3") or _wheel("cvc5")),
    ),
    Rule(
        when="the two solver wheels are not both installed",
        reasons=frozenset({"needs both z3 and cvc5", "needs both real solver wheels"}),
        legitimate=lambda: not (_wheel("z3") and _wheel("cvc5")),
    ),
    Rule(
        when="the z3 wheel is not installed",
        reasons=frozenset(
            {"needs z3", "needs z3 to cross-check the emission", "needs the z3 wheel"}
        ),
        legitimate=lambda: not _wheel("z3"),
    ),
    Rule(
        when="the cvc5 wheel is not installed",
        reasons=frozenset({"needs cvc5"}),
        legitimate=lambda: not _wheel("cvc5"),
    ),
    Rule(
        when="`uv` is not on PATH, so no distribution can be built to inspect",
        reasons=frozenset({"needs `uv` to build", "needs `uv` to build an sdist"}),
        legitimate=lambda: shutil.which("uv") is None,
    ),
    Rule(
        when=(
            "hypothesis is not installed. It is a DEV-GROUP dependency, not an "
            "extra and not a runtime one, so the two shared jax venvs and the "
            "zero-dep CI job do not have it and every module under "
            "`tests/property/` gates at collection. What keeps that from "
            "reading as `the property suite found nothing` is "
            "`tests/property/test_suite_disclosure.py`, which needs neither "
            "hypothesis nor jax, runs in exactly these environments, and holds "
            "the registry of positive controls to the properties that exist"
        ),
        reasons=frozenset({"needs hypothesis"}),
        legitimate=lambda: not _importable("hypothesis"),
    ),
    Rule(
        when=(
            "no second interpreter with the other jax series was named, so "
            "the cross-series differential has nothing to differ against. The "
            "condition is an environment variable rather than an installed "
            "library, and it is checked in BOTH directions here: with "
            "`STELLING_PROPERTY_OTHER_PYTHON` set, that property MUST run"
        ),
        reasons=frozenset(
            {
                "needs a second interpreter with the other jax series "
                "(set STELLING_PROPERTY_OTHER_PYTHON)"
            }
        ),
        legitimate=lambda: not os.environ.get("STELLING_PROPERTY_OTHER_PYTHON"),
    ),
    Rule(
        when="`git` is not on PATH",
        reasons=frozenset({"needs git"}),
        legitimate=lambda: shutil.which("git") is None,
    ),
    Rule(
        when=(
            "`JAX_ENABLE_X64` is ON, where jax's own PRNG seed mask does not "
            "narrow. The tripwire's one honest fire across jax's whole test "
            "suite is `4294967295 -> -1 (int32)` from `threefry2x32.py:73`, "
            "and it happens at x64=0 ONLY: at x64=1 the mask fits its dtype "
            "and nothing narrows. A test asserting that fire is suppressed and "
            "named would pass at x64=1 having measured nothing, which is the "
            "beautiful zero this suite exists to refuse — so it skips, loudly, "
            "in the configuration where its subject does not occur. Computable "
            "from here, so BOTH directions are asserted"
        ),
        reasons=frozenset({"the threefry mask fires only at x64=0"}),
        legitimate=_jax_x64_is_on,
    ),
    Rule(
        when=(
            "`pytest-xdist` is not installed. It is a DEV dependency and never "
            "a runtime one — the tripwire's xdist aggregation is a guardrail "
            "for users who already run `-n auto`, not a dependency it imposes "
            "— so the zero-dep CI job, the jax-0.10 lane and the shared dev "
            "venvs do not have it and the aggregation tests gate at "
            "collection. The `tests (solvers + jax)` lane DOES install it, "
            "deliberately: those tests are the only executable evidence for "
            "acceptance criterion 4 and for both items PLAN-tripwire.md §2 "
            "lists as assumed, and a skip declared in EVERY lane is still a "
            "criterion nothing ever ran"
        ),
        reasons=frozenset({"needs pytest-xdist to drive a real worker split"}),
        legitimate=lambda: importlib.util.find_spec("xdist") is None,
    ),
    Rule(
        when="this tree is not a git checkout — an unpacked sdist, say",
        reasons=frozenset({"not a git checkout (an unpacked sdist, say)"}),
        legitimate=lambda: not (REPO / ".git").exists(),
    ),
    Rule(
        when=(
            "git cannot read THIS TREE'S OWN HISTORY at the commit the "
            "routing manifest names — an unpacked sdist, an export, a "
            "shallow CI clone that does not reach it, a tree unpacked "
            "inside somebody else's repository, or a linked worktree whose "
            "parent has moved. Five causes and one condition, and the "
            "condition IS computable here: `git show <commit>:CHANGELOG.md` "
            "at the commits the manifest names, AND the environment being one "
            "of those five rather than a manifest naming a commit that is "
            "not there -- which is a defect and FAILS over there rather than "
            "skipping, so this rule can never excuse one. "
            "`.git` absence is only ONE of the five causes, so the "
            "predicate asks git the same question the test does rather than "
            "testing for a directory. What the skip cannot say in "
            "its reason it says in a WARNING: which cause was measured, "
            "git's own words, and which manifest columns go unverified. "
            "UNTIL 2026-08-22 THE REASON WAS AN f-STRING CARRYING GIT'S "
            "STDERR, so no rule could name it and a git-less tree exited 1 "
            "on the completeness half no matter what — nine undisclosed "
            "skips, in the two files whose subject is that a checkout "
            "without git cannot verify the routing"
        ),
        reasons=frozenset({
            "git cannot read this tree's own history, so the routing "
            "manifest's source-side columns are unverified here",
        }),
        legitimate=_git_cannot_read_the_routing_source,
    ),
    Rule(
        when=(
            "XLA on THIS build did not contract the form, so eager and jitted "
            "agree and there is no contraction to be indeterminate about. Not "
            "computable from here: the condition is a measurement the test "
            "makes on the running XLA, and reproducing it would mean running "
            "the test. Disclosed so that a runner whose XLA folds differently "
            "does not read as inventory drift"
        ),
        # The variable part is the parametrised FORM NAME and nothing else,
        # which is now a statement about the code and not about a regex: the
        # names come from the site that emits them.
        reasons=_DERIVED_REASONS,
        legitimate=None,
    ),
)


# --- the measured: a named test whose skip condition is its own measurement --


@dataclasses.dataclass(frozen=True)
class Measured:
    """A skip the other three shapes cannot express, and the weakest of the four.

    ``PINNED`` and ``RULES`` both key off something this file can evaluate: is
    the library importable, is the wheel installed, is ``uv`` on PATH. These
    two skips key off something the test MEASURES on its own subject — whether
    interval propagation happened to settle an obligation before any solver
    saw it. That is not dependency-dependent, so no predicate here can decide
    it, and reproducing it would mean re-running the test.

    So the direction is not checked, and that has to be paid for. It is paid
    for by giving up every dimension of breadth at once:

    * **test-keyed**, not reason-keyed — the excuse cannot travel to another
      test, which is precisely what a ``RULES`` entry for the same reason
      would allow;
    * **exact reason**, no pattern, no anchors to get wrong;
    * **verified against the source** — the named test must exist and must
      really contain this exact ``pytest.skip`` literal, so an entry cannot
      outlive the skip it excuses (see
      ``test_every_measured_skip_is_written_at_the_site_it_names``).

    An entry here costs a real ``pytest.skip("<this exact string>")`` in a real
    test at a named nodeid. That is the bound: you cannot add one without the
    skip already existing, and it excuses nothing else.
    """

    nodeid: str
    reason: str  # matched in full, and checked against the test's source
    why: str  # the measurement, and why it cannot be made from here


MEASURED = (
    Measured(
        "tests/test_escalation_invariant.py::test_the_bar_now_fires_on_the_REAL_barred_set",
        "intervals settled it; this needs a solver-decided verdict",
        "the test asks whether the VERIFIED bar withholds a SOLVER-decided "
        "verdict on a scatter-bearing slice. If interval propagation settles "
        "every obligation first, no obligation reaches a solver and there is "
        "no solver-decided verdict for the bar to withhold — so the test has "
        "nothing to measure and says so. `_obl_solves(v) == 0` is the "
        "condition, and computing it means running `check` on the test's own "
        "query. Fires in the no-solver lane (nothing can reach a solver "
        "there) and can fire with solvers installed if the intervals get "
        "there first, which is why it is not a solver RULES entry: the "
        "solver rules assert `not _wheel(...)`, and that is FALSE in the "
        "environment where this skip is still legitimate.",
    ),
    Measured(
        "tests/test_square_row.py::test_a_boolean_square_declines",
        "interval propagation decided it; no slice to validate",
        "same shape, on the declined-obligation slice: if propagation returns "
        "a status other than 'unknown' there is no unknown obligation to "
        "slice, so `slice_unknown_obligations` has nothing to hand back. "
        "Disclosed although it has not been observed to fire in any of the "
        "five environments — a latent skip is exactly the kind that shows up "
        "as unexplained drift on someone else's machine.",
    ),
)


def _measured_for(nodeid: str, reason: str) -> Measured | None:
    for entry in MEASURED:
        if entry.reason == reason and (
            _same_node(nodeid, entry.nodeid) or _same_node(entry.nodeid, nodeid)
        ):
            return entry
    return None


# --- observing what this session actually did -------------------------------


def _same_node(observed: str, wanted: str) -> bool:
    """Node ids are relative to rootdir, which depends on how pytest was
    invoked; compare on the trailing path so `tests/x.py::t` and `x.py::t`
    are the same node."""
    return observed == wanted or observed.endswith("/" + wanted)


def _from_session(nodeid: str) -> tuple[str, str] | None:
    module = nodeid.split("::", 1)[0]
    for seen, reason in SKIPPED.items():
        if _same_node(seen, nodeid) or _same_node(seen, module):
            return "skipped", reason  # module gates skip at COLLECTION
    for seen in RAN:
        if _same_node(seen, nodeid):
            return "ran", ""
    return None


_SKIPPED_LINE = re.compile(r"^SKIPPED \[\d+\] \S+?:\d+: (.*)$", re.MULTILINE)


def _tail(proc: subprocess.CompletedProcess, lines: int = 40) -> str:
    out = (proc.stdout or "") + (proc.stderr or "")
    return "\n".join(out.splitlines()[-lines:])


def _from_a_run_of_its_own(nodeid: str) -> tuple[str, str]:
    """The pinned test was not part of this session — someone ran this file on
    its own. Run it on its own too, rather than assert nothing: a pin that goes
    vacuous the moment you narrow the invocation is not a pin.

    Costs nothing in a full run, which is where the session already has the
    answer.
    """
    # `PY_COLORS=0` for the same reason `_run_a_miniature_session` sets it, and
    # this is the site where it was actually costing something. `_SKIPPED_LINE`
    # is anchored with `^SKIPPED`, and pytest prefixes that line with an SGR
    # escape whenever it believes a human is watching — which it does whenever
    # `FORCE_COLOR` is set in the ambient environment. The regex then misses, so
    # a test that DID skip falls through to the returncode check below and is
    # reported as "neither SKIPPED nor passed".
    #
    # MEASURED, `test_pinned_skips_track_their_condition_in_this_environment`,
    # in a session narrowed to this file so that this path is taken at all:
    # with `FORCE_COLOR=3` it FAILS, accusing `tests/test_any_pytree.py::
    # test_h_clean_sugar_hash_equals_hand_declaration` of having errored — and
    # quoting, inside its own failure message, the `SKIPPED [1] … could not
    # import 'jax'` line it had just failed to match. With `PY_COLORS=0` it
    # passes. The child exits 4 either way (the module gate skips at
    # COLLECTION, so the narrowed nodeid has no collector); colour decides only
    # whether that is read correctly.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", nodeid],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "PY_COLORS": "0"},
    )
    hit = _SKIPPED_LINE.search(proc.stdout)
    if hit:
        return "skipped", hit.group(1)
    assert "no tests ran" not in proc.stdout, (
        f"{nodeid} collected nothing — the pin names a test that does not "
        f"exist:\n{proc.stdout}"
    )
    # The RETURNCODE, not merely the absence of a `SKIPPED` line. "Did not
    # skip" and "ran" are not the same claim, and every outcome pytest has —
    # a fixture ERROR, an outright FAILURE, an internal error, a usage error —
    # prints no `SKIPPED` line and does not say "no tests ran". Reading any of
    # them as "ran" is how an installed-but-BROKEN blackjax produced five
    # green inventory tests while both pinned tests errored inside this hidden
    # subprocess: the pin reported the strongest outcome for the worst one.
    assert proc.returncode == 0, (
        f"{nodeid} neither SKIPPED nor passed when run on its own: pytest "
        f"exited {proc.returncode}. The inventory pins whether this test "
        f"skips, and it cannot answer that from a session where the test "
        f"errored, failed, or never started — and it must not report the run "
        f"as 'ran', which is what reading only the absence of a `SKIPPED` "
        f"line would have done. An ERROR here usually means the dependency is "
        f"importable but broken; the tail below names the real cause.\n\n"
        f"{_tail(proc)}"
    )
    return "ran", ""


def _outcome(nodeid: str) -> tuple[str, str]:
    return _from_session(nodeid) or _from_a_run_of_its_own(nodeid)


# --- what THIS session is in a position to claim ------------------------------
#
# The completeness half claims that THE SUITE's skip set is disclosed, and a
# session can only support that about tests it collected AND ran AND ran before
# this pin did.
#
# What pytest's own report does and does not tell apart from a whole run is the
# thing this taxonomy turns on, and it was got wrong once. It DOES disclose a
# deselection — a line reading `<n> passed, <m> deselected` is not one reading
# `<n'> passed, <k> skipped`, and it never was, whatever the suite's size is on
# the day. It discloses an xfail too, in the same line. What it cannot
# disclose is a run in a different ORDER, or items removed by a plugin that
# never called `pytest_deselected`: those print a green that is byte-identical
# to a clean whole run's. So the shortfalls that pytest already reports are
# WITHDRAWN here, and the ones it cannot report are FAILED.
#
# THE ORDER THE QUESTIONS ARE ASKED IN IS PART OF THE ANSWER. The FAILURES are
# asked first and the WITHDRAWALS second, because a withdrawal is silenced by
# tests/conftest.py in a narrowed session and a failure never is. Asked the
# other way round — `unseen`, then `filtered_out`, then the undisclosed drop —
# ANY narrowing at all bought a silent pass on the one shortfall this whole
# mechanism exists for. Measured at b277083, one item removed from `items[:]`
# by a plugin that never calls `pytest_deselected`:
#
#     whole tree                          2008 passed, 3 skipped, EXIT 1
#     --ignore=tests/test_square_row.py   1991 passed, 3 skipped, EXIT 0, SILENT
#     pytest tests/test_affine.py         40 passed,              EXIT 0, SILENT
#     -k "not test_op_add"                2007 p, 3 s, 1 d,       EXIT 0, SILENT
#
# All four are EXIT 1 with a banner at a80d60c, and re-driven at a76ca51 with a
# plugin that removes the first non-pin item from `items[:]` and never calls
# `pytest_deselected`:
#
#     whole tree                          2033 passed, 3 skipped, EXIT 1, banner
#     --ignore=tests/test_square_row.py   2016 passed, 3 skipped, EXIT 1, banner
#     pytest tests/test_affine.py         40 passed,              EXIT 1, banner
#     -k "not test_op_add"                2032 p, 3 s, 1 desel,   EXIT 1, banner
#
# "here" was the label on that sentence until this pass, and a word that means
# whichever tree the file happens to be sitting in is a label that goes stale
# without anybody editing it — the same defect as an unlabelled count, one
# level down, in a sentence with no number in it at all. The EXIT and BANNER
# columns are what the block is FOR and they are size-independent; the counts
# are the tree's and are labelled as the tree's.
#
# The session is interrogated, and the answers differ because the mistakes
# differ:
#
#   * it never collected the whole tree (`pytest tests/test_x.py`, `--lf`,
#     `--ignore`, a shard). Ordinary development; nobody asked about the suite.
#     WITHDRAWN — the pin skips, saying which files it never saw. Failing here
#     would break `pytest tests/test_skip_inventory.py`, which is worse than
#     the hole it closes.
#
#   * it collected the whole tree and then DESELECTED part of it — `-k`, `-m`,
#     `--deselect`, `--sw`, `--lf`, a plugin. WITHDRAWN, naming the filter if
#     the developer passed one and saying it was not a filter if they did not.
#
#     This used to FAIL when the filter was the developer's own, on the stated
#     grounds that such a session "would print a green indistinguishable from a
#     whole run". That premise is false, and what shows it is a pair of summary
#     LINES rather than a pair of numbers:
#
#         pytest -q -k interval        `<n> passed, <m> deselected`
#         the silent-item-drop plugin  `<n'> passed, <k> skipped`
#
#     The first line NAMES the shortfall: the word `deselected`, and a count of
#     it, and there is no invocation that deselects without pytest saying so.
#     The second names nothing — two numbers moved and no word for what
#     happened. That is the entire argument, and it is a claim about which
#     WORDS pytest's summary line carries. No count is load-bearing in it, so
#     none is written here.
#
#     Twice now this passage has carried a figure belonging to another commit,
#     and the second time was the repair for the first:
#
#       * `98 passed, 1897 deselected` and `1992 passed, 3 skipped` once stood
#         here in a b277083 row; both were bd1fa04's. The FIGURE was stale.
#       * they were replaced by `98 passed, 1926 deselected` and `2020 passed,
#         3 skipped` under the words "both measured at b277083" — and those are
#         a80d60c's, this file's own commit. The LABEL was stale, which is the
#         same defect running the other way, and it is self-refuting on the
#         arithmetic: b277083 collects 2012, and 98 + 1926 = 2024.
#
#     Measured, one pristine worktree per commit, whole tree from the repo root:
#
#         -k interval    b277083: `98 passed, 1914 deselected`
#                        a80d60c: `98 passed, 1926 deselected`
#         silent drop    b277083: `2008 passed, 3 skipped`, exit 1
#                        a80d60c: `2020 passed, 3 skipped`, exit 1
#
#     A total is a fact about a commit wearing the shape of a fact about a
#     mechanism, which is why it keeps ending up under the wrong name. §53's
#     rule — record failures, not totals — is the fix, and obeying it here
#     means stating the argument in the words pytest prints rather than in the
#     numbers it prints them beside.
#
#     Failing on `-k` also cost more than it bought: at bd1fa04 no `-k`
#     expression that deselected this file could exit 0 from the repo root on a
#     CLEAN tree (`-k interval`: exit 1, where b1c69d1 and 8ef8f75 are exit 0),
#     which is the commonest invocation a developer has. The disclosure half
#     above is unaffected and still bites: `-k "not verdict"` with an
#     undisclosed skip inside the selected set FAILS, because that skip was
#     observed.
#
#     `USER_FILTERS` is still read off `config.option` rather than inferred
#     from `pytest_deselected` — `_pytest/stepwise.py` calls that hook exactly
#     as `-k` does, so the EFFECT cannot tell them apart, and the message says
#     which happened.
#
#   * it ran this pin before the rest of the session (`--nf`, a reordering
#     plugin). Nothing is missing; the pin is merely early. The claim is
#     DEFERRED to tests/conftest.py, which makes it when the loop is over.
#
#   * something dropped collected tests without reporting them deselected.
#     Then the session's own summary is wrong, not just this pin's, and the
#     end of the session FAILS: `N passed` while N+3 were collected is not a
#     narrowing, it is a session that lied about what it did. Asked FIRST, so
#     that no narrowing above can answer for it.
#
#   * it contains a test pytest counted as `xfailed`. That test handed back no
#     verdict this pin can read — with `run=False` it never started at all —
#     and pytest put `N xfailed` in the very same summary line it puts
#     `N deselected` in. Disclosed, therefore WITHDRAWN, by the same rule that
#     moved `-k` from FAILED to WITHDRAWN. An `xpassed` test RAN and passed and
#     is none of this. Measured at b277083, with the plant appended to
#     tests/test_affine.py:
#
#         @pytest.mark.xfail(run=False, reason="a planted reason nobody
#                            disclosed")
#         def test_planted_never_runs():
#             assert False
#
#         pytest -q -rs  ->  2010 passed, 2 skipped, 1 xfailed, EXIT 0, and no
#                            banner: the claim was MADE.
#
#     `pytest.skip()` with the same string in the same place is exit 1, and
#     `pytest.xfail()` in the body is the same exit 0. It was an eighth route,
#     and it was condemned by this file's own rule before it was found.
#
# Deliberately not a minimum-collection floor. A floor is a constant that has
# to be raised as the suite grows and quietly stops biting as it shrinks, and
# it cannot tell any of these apart — every one of them collects the tree.


# --- the assertions ----------------------------------------------------------


def test_every_pinned_test_exists_under_the_name_pinned_here():
    """A rename turns a pin into a comment. Static, so it holds even in an
    environment where the pinned test cannot be collected at all."""
    missing = []
    for pin in PINNED:
        path, _, name = pin.nodeid.partition("::")
        source = REPO / path
        if not source.exists() or f"def {name}(" not in source.read_text():
            missing.append(pin.nodeid)
    assert not missing, (
        "the skip inventory pins tests that no longer exist under these "
        "names:\n  " + "\n  ".join(missing) + "\n\nRe-point the pin, or drop it "
        "if the test is gone."
    )


def test_pinned_skips_track_their_condition_in_this_environment():
    """BOTH directions. In an environment missing the dependency the skip is
    the expected outcome; in one that has it, RUNNING is the expected outcome
    and a skip is the defect. Neither environment has to be identified: the
    condition is evaluated here."""
    for pin in PINNED:
        missing = [dep for dep in pin.needs if not _importable(dep)]
        outcome, reason = _outcome(pin.nodeid)
        if missing:
            assert outcome == "skipped", (
                f"{pin.nodeid} RAN, but {missing} cannot be imported here, so "
                f"the inventory says it must skip. Either its gate was removed "
                f"(restore it — an ungated test errors instead of skipping in "
                f"the environments that lack {missing[0]}), or this pin's "
                f"`needs` is stale.\n\nWhy this test is pinned: {pin.why}"
            )
            assert any(dep in reason for dep in missing), (
                f"{pin.nodeid} skipped for a reason unrelated to its pinned "
                f"condition {pin.needs}: {reason!r}. Disclose the new reason "
                f"here or make it not skip."
            )
        else:
            assert outcome == "ran", (
                f"{pin.nodeid} SKIPPED ({reason!r}) although all of "
                f"{list(pin.needs)} import fine here. A job that installs a "
                f"dependency and then silently skips the tests that need it is "
                f"the exact defect this inventory exists to close.\n\n"
                f"Why this test is pinned: {pin.why}"
            )


class _StringLiterals(ast.NodeVisitor):
    """Every string literal in a module's source, f-strings excluded.

    An f-string's fixed fragments are not reasons anybody wrote down — they are
    halves of one — so recursing into ``JoinedStr`` would let a rule excuse a
    string that appears in this file only as a piece of some other string.
    """

    def __init__(self) -> None:
        self.found: set[str] = set()

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.found.add(node.value)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        return


def _string_literals_in(path: pathlib.Path) -> set[str]:
    visitor = _StringLiterals()
    visitor.visit(ast.parse(path.read_text()))
    return visitor.found


def test_a_rule_excuses_only_reasons_written_down_in_this_file():
    """The bound the file CLAIMED and did not have: ``reasons`` is a runtime
    value, and nothing required it to be literals.

    ``Rule.matches`` is exact membership, so a rule's excuse set is finite and
    enumerable — but enumerable is not the same as WRITTEN DOWN, and the
    difference is a one-line rule using the same idiom the XLA rule already
    uses::

        Rule(when="…read out of the module that emits them…",
             reasons=frozenset(_reasons_declared_by(TESTS / "test_affine.py")),
             legitimate=lambda: True)

    That rule excuses every reason some other module happens to skip with,
    none of which appears in this file. Measured at bd1fa04 with a planted
    ``pytest.skip("we will get back to this one after the release")``: excused,
    the whole suite green, exit 0.

    So the bound is enforced instead of described. Every reason a rule excuses
    must be a string literal in THIS file — where the diff that adds it is a
    diff to the disclosure surface and a reviewer reads it — or one of the
    reasons produced by the single declared derivation,
    :data:`_DERIVED_REASONS`, whose channel is pinned separately by
    ``test_no_rule_is_broader_than_the_reason_it_discloses`` (the count of
    forms, the containment, and the exact ``pytest.skip`` literal at the site).

    What this does NOT bound, stated rather than implied: an author who types
    a reason here and writes ``legitimate=lambda: True`` next to it has
    disclosed it, and this file will excuse it. That is what disclosure means —
    the guarantee is that the excuse is legible in one file, not that it is
    justified. The two nets below narrow the laziest version of it.
    """
    literals = _string_literals_in(pathlib.Path(__file__))
    stray = [
        f"{reason!r} (rule for {rule.when[:50]!r})"
        for rule in RULES
        for reason in sorted(rule.reasons)
        if reason not in literals and reason not in _DERIVED_REASONS
    ]
    assert not stray, (
        "skip rule(s) excusing a reason that is not written in this file and "
        "did not come from the one declared derivation:\n  "
        + "\n  ".join(stray)
        + "\n\n`reasons` is an ordinary Python value, so it can be computed "
        "from anything — including another module's source, which excuses "
        "every skip that module happens to carry while this file, the "
        "disclosure surface, says nothing. Type the reason out here, or add "
        "the derivation to _DERIVED_REASONS where its channel gets pinned."
    )
    # Anti-vacuity, both halves. The derivation has to be real (an empty one
    # would make the clause above trivially satisfiable by anything claiming
    # to be derived), and it has to still be the XLA rule's.
    assert _DERIVED_REASONS, (
        "_DERIVED_REASONS is empty, so the exemption above admits nothing and "
        "would equally admit a rule that derived its reasons from anywhere"
    )
    assert _DERIVED_REASONS <= {r for rule in RULES for r in rule.reasons}, (
        "_DERIVED_REASONS is no longer the reason set of any rule, so the "
        "derivation named here and the derivation in use have come apart"
    )
    # and MEASURED, whose whole bound is enumeration, is held to the same thing
    off_file = [e.nodeid for e in MEASURED if e.reason not in literals]
    assert not off_file, (
        "MEASURED entries whose reason is not a literal in this file:\n  "
        + "\n  ".join(off_file)
    )


def _cannot_read_its_environment(predicate) -> bool:
    """A zero-argument callable that references no name at all.

    Such a callable cannot look at anything: no global, no closure cell, no
    attribute. Whatever it returns, it returns in every environment, so it is
    ``legitimate=None`` wearing a callable and belongs in the same budget.
    ``lambda: True`` and ``lambda: 1 == 1`` are both caught by it.

    A NET, and it says so: ``lambda: bool(os) or True`` reads a name and walks
    past. What bounds a rule is that its reason is written down here; this
    stops the laziest way of pretending to check a direction.
    """
    code = getattr(predicate, "__code__", None)
    if code is None:
        return False
    return code.co_argcount == 0 and not code.co_names and not code.co_freevars


def test_every_rule_states_the_condition_it_discloses():
    """A rule with no condition is a blanket permission slip.

    ``when`` used to be checked with ``.strip()``, which a single character
    passes — so ``Rule(when="x", reasons=frozenset({"<an exact reason>"}),
    legitimate=lambda: True)`` disclosed nothing and excused a planted skip.
    Both halves of that rule are now floored: a condition has to read like a
    condition, and a ``legitimate`` that cannot look at anything counts against
    the same budget as no ``legitimate`` at all.

    These are FLOORS, not bounds — a determined author writes four plausible
    words and a predicate that names something. The bound is
    ``test_a_rule_excuses_only_reasons_written_down_in_this_file``: whatever
    the rule says, the reason it excuses is in this file's diff.
    """
    silent = [
        rule
        for rule in RULES
        if len(rule.when.split()) < 4 or len(rule.when.strip()) < 15
    ]
    assert not silent, (
        "these skip rules disclose a reason without saying, in words, under "
        "what condition it is legitimate:\n  "
        + "\n  ".join(
            sorted(f"{r.when!r}: {', '.join(sorted(r.reasons))}" for r in silent)
        )
        + "\n\n`when` IS the disclosure for every rule whose direction cannot "
        "be checked, and it is the thing a reviewer reads for the ones that "
        "can. A placeholder there is a blanket permission slip."
    )
    unchecked = [
        rule
        for rule in RULES
        if rule.legitimate is None or _cannot_read_its_environment(rule.legitimate)
    ]
    assert len(unchecked) <= 1, (
        "more than one skip rule has given up on checking its own direction. "
        "Each of these discloses a skip without being able to contradict it, "
        "which is the weak form this file exists to avoid (a `legitimate` that "
        "reads no name is counted here: it returns the same answer in every "
        "environment, which is what `legitimate=None` means):\n  "
        + "\n  ".join(sorted(", ".join(sorted(r.reasons)) for r in unchecked))
    )


def test_the_import_gate_wording_is_the_one_pytest_actually_emits():
    """Anti-vacuity, and the load-bearing string in this file. If pytest's
    wording moved, every ``importorskip`` in the suite would read as
    undisclosed; if this regex were loose, everything would read as disclosed.
    Both failures are silent without this."""
    hit = _IMPORT_GATE.match("could not import 'blackjax': No module named 'blackjax'")
    assert hit and hit.group(1) == "blackjax"
    assert _IMPORT_GATE.match("could not import 'jax'")
    assert not _IMPORT_GATE.match("needs z3")
    assert not _IMPORT_GATE.match("could not import the thing")
    # and the recorder's own unwrapping of pytest's report tuple
    assert _reason(("t.py", 3, "Skipped: could not import 'x'")) == "could not import 'x'"


# Reasons no rule in this file has any business excusing. Some of them are
# real skip reasons from elsewhere in the suite (a rule that swallowed one
# would silently take over another shape's disclosure); the rest are the
# shapes a widened pattern produces. Fourteen of them (the commit that added
# this list said eleven and there were thirteen; there is now one more, below).
# The list is the SECOND net — the first is that a rule cannot express breadth
# at all, which
# is what ``test_a_rule_can_only_excuse_a_reason_someone_wrote_down`` pins. A
# decoy list on its own bounds nothing: `pattern=r"skip: .*"` passed every
# entry below unchanged.
_MUST_NOT_MATCH = (
    "a totally undisclosed reason nobody wrote down",
    "totally other cause; XLA did not contract this form on this build",
    "XLA did not contract this form on this build, and also something else",
    # The prose prefix. The shipped pattern was
    # `r"[\w+ .\-]+: XLA did not contract this form on this build"`, whose
    # comment claimed the variable part was "the parametrised FORM NAME and
    # nothing else"; every decoy above contains a `;`, which is the only thing
    # that character class excluded. Measured at b1c69d1 with this exact string
    # planted as a real skip: the XLA rule excused it and the suite was exit 0.
    "the solver crashed and we gave up: XLA did not contract this form on this build",
    # These two are what separates fullmatch from search. Each CONTAINS a
    # reason the XLA rule really does disclose, wrapped in text it does not:
    # under `re.search` both were excused, and an unverifiable rule that can
    # be reached by adding a prefix or a suffix is not bounded by anything.
    "a different failure happened first; reshape: XLA did not contract this form on this build",
    "nested jit: XLA did not contract this form on this build, and then the run gave up",
    "intervals settled it; this needs a solver-decided verdict",
    "interval propagation decided it; no slice to validate",
    "could not import 'blackjax': No module named 'blackjax'",
    "needs zebra",
    "flaky on this runner",
    "x",
    ".",
    "",
)

# One reason per rule that MUST match, so the test above cannot pass by every
# pattern having become inert.
_MUST_MATCH = (
    "needs an SMT solver",
    "needs both z3 and cvc5",
    "needs z3 to cross-check the emission",
    "needs cvc5",
    "needs `uv` to build an sdist",
    "needs git",
    "not a git checkout (an unpacked sdist, say)",
    "broadcast+slice+squeeze: XLA did not contract this form on this build",
    "nested jit: XLA did not contract this form on this build",
)


def test_a_rule_can_only_excuse_a_reason_someone_wrote_down():
    """The bound on breadth, as a property OF THE RULE rather than a list.

    Three bounds were tried here and each was defeated by the next rule
    somebody could have written: ``re.search`` → ``re.fullmatch`` → a list of
    decoy strings. The decoy list is the shape this project calls
    "enumerating current members rather than pinning the channel", and it was
    measured to fail exactly that way: ``Rule(when="…", pattern=r"skip: .*",
    legitimate=lambda: True)`` excuses every reason beginning ``"skip: "`` and
    passed all three halves of the old breadth test — decoys, must-match, form
    names — at ``7 passed, 1 skipped``.

    The channel is the matcher. A rule excuses a FINITE, ENUMERATED set of
    exact strings, so the most a bad rule can do is excuse a reason its author
    typed out in this file where a reviewer reads it. That is what disclosure
    means here, and a pattern is precisely the way to excuse text nobody typed.

    Finite and enumerated is only half of it, and this test only ever asserted
    that half: ``reasons`` is a runtime value, so a rule could ENUMERATE by
    computing — ``frozenset(_reasons_declared_by(TESTS / "test_affine.py"))``
    passes everything below and excuses reasons that appear nowhere here. The
    other half is
    ``test_a_rule_excuses_only_reasons_written_down_in_this_file``, and the
    sentence above is true because that test is there.

    So this asserts the shape, not a corpus:

    * a rule has no field a pattern could live in;
    * every rule names at least one reason, or it excuses nothing and is a
      comment;
    * matching is exact — every neighbour of every declared reason is
      REJECTED, which is what fails if ``matches`` is ever loosened back into
      a prefix, a substring or a regex.

    The one rule whose reasons are genuinely variable (the parametrised XLA
    form name) stays inside this by reading the names out of the module that
    emits them, which is the same enumeration with the author taken out of it.
    """
    fields = {f.name for f in dataclasses.fields(Rule)}
    assert fields == {"when", "reasons", "legitimate"}, (
        f"Rule's fields are now {sorted(fields)}. If one of them is a matcher "
        f"— a regex, a prefix, a callable — then a rule's excuse set stops "
        f"being enumerable and nothing below bounds it: that is how "
        f"`pattern=r'skip: .*'` passed the decoy list, the must-match list and "
        f"the form-name list at once. Disclose reasons by naming them."
    )
    for rule in RULES:
        assert rule.reasons, (
            f"the rule for {rule.when[:60]!r} names no reason, so it excuses "
            f"nothing and discloses nothing"
        )
        for reason in rule.reasons:
            assert isinstance(reason, str) and reason.strip(), rule.when
            assert rule.matches(reason)
            neighbours = (
                reason + " ",
                " " + reason,
                reason + ".",
                reason[:-1],
                reason + " and then the run gave up",
                "a different failure happened first; " + reason,
                reason.upper() if reason.upper() != reason else reason + "!",
            )
            for neighbour in neighbours:
                assert not rule.matches(neighbour), (
                    f"the rule for {rule.when[:60]!r} excuses {neighbour!r}, "
                    f"which is not a reason it names. Matching has been "
                    f"loosened past exact membership, and a rule that can be "
                    f"reached by adding a prefix or a suffix bounds nothing."
                )


def test_no_rule_is_broader_than_the_reason_it_discloses():
    """The second net: a corpus of reasons no rule may excuse.

    ``matches`` used ``re.search``, so a pattern excused anything that
    CONTAINED it: the one rule that cannot check its own direction covered
    ``"totally other cause; XLA did not contract this form on this build"``
    unmutated, and widening it to ``r"."`` disclosed the entire suite while
    the suite stayed green. Anchoring fixed the first, this fixed the second,
    and neither reaches a rule that is broad in a shape nobody listed — which
    is why the test above pins the matcher itself and this one now backs it up.
    """
    over_broad = [
        f"{decoy!r} is excused by the rule for: {rule.when[:60]}"
        for decoy in _MUST_NOT_MATCH
        for rule in RULES
        if rule.matches(decoy)
    ]
    assert not over_broad, (
        "skip rule(s) matching reasons they do not disclose — a rule that "
        "excuses text it was not written for is a blanket permission wearing "
        "a condition:\n  " + "\n  ".join(over_broad)
    )
    # Anti-vacuity for the above: patterns that match nothing would pass it.
    unmatched = [r for r in _MUST_MATCH if not any(x.matches(r) for x in RULES)]
    assert not unmatched, (
        "these reasons are disclosed by a rule in this file and no rule "
        "matches them any more, so the rules have gone inert and the check "
        "above proves nothing:\n  " + "\n  ".join(unmatched)
    )
    # The other edge of the same bound. Excusing only what it names must still
    # mean excusing every reason the site can actually emit, or a runner whose
    # XLA folds differently reads as drift — which is the exact thing that rule
    # exists to prevent. The names are read out of the source, so this half is
    # satisfied BY CONSTRUCTION and is here for the way the construction can
    # break: the extraction going inert. That is the assertion with the teeth
    # in it, and it is why the count is checked before the coverage.
    forms = _contraction_form_names()
    assert len(forms) >= 9, (
        f"only {len(forms)} contraction forms found in "
        f"{_FORM_TABLE.name} — the shape this reads has changed, so the XLA "
        f"rule's reasons are now derived from a shorter list (or an empty "
        f"one) and a build that does not fold those forms would read as "
        f"inventory drift"
    )
    assert set(forms) <= {
        reason[: -len(_XLA_SUFFIX)]
        for rule in RULES
        for reason in rule.reasons
        if reason.endswith(_XLA_SUFFIX)
    }, "a contraction form whose XLA skip no rule excuses"
    # and the derivation is of the REASON, not merely of a name: the string the
    # site formats has to be the string a rule holds.
    assert any(
        rule.matches(f"{forms[0]}{_XLA_SUFFIX}") for rule in RULES
    ), f"no rule excuses the reason test_three_rows_acceptance.py emits for {forms[0]!r}"
    assert f'pytest.skip(f"{{name}}{_XLA_SUFFIX}")' in _FORM_TABLE.read_text(), (
        f"{_FORM_TABLE.name} no longer emits its skip as "
        f'`pytest.skip(f"{{name}}{_XLA_SUFFIX}")`, so the reasons derived from '
        f"its form names are no longer the reasons it can produce"
    )


def _body_of(nodeid: str) -> str | None:
    """The source of one test function, from ``def name(`` to the next
    top-level ``def``. Text, not import: this must work in an environment
    where the module cannot even be collected."""
    path, _, name = nodeid.partition("::")
    source = REPO / path
    if not source.exists():
        return None
    text = source.read_text()
    start = text.find(f"\ndef {name}(")
    if start < 0:
        return None
    end = text.find("\ndef ", start + 1)
    return text[start : end if end > 0 else len(text)]


def test_every_measured_skip_is_written_at_the_site_it_names():
    """The whole bound on MEASURED: it may not outlive, or reach beyond, the
    ``pytest.skip`` it excuses.

    MEASURED is the one shape whose direction nothing here can check, so it
    buys that with enumeration — a named test and an exact reason. Both halves
    have to be real, or the enumeration is just prose: a renamed test or a
    reworded skip would leave an entry excusing a reason no test can emit,
    while the reason it CAN emit reads as undisclosed drift somewhere else.
    Static, so it holds in every environment, including the ones where the
    named module never collects.
    """
    broken = []
    for entry in MEASURED:
        body = _body_of(entry.nodeid)
        if body is None:
            broken.append(f"{entry.nodeid}: no test of that name in the tree")
        elif f'pytest.skip("{entry.reason}")' not in body:
            broken.append(
                f"{entry.nodeid}: does not contain "
                f'pytest.skip("{entry.reason}")'
            )
    assert not broken, (
        "MEASURED entries that no longer describe a real skip:\n  "
        + "\n  ".join(broken)
        + "\n\nRe-point the entry at the skip as it is actually written, or "
        "drop it. An entry that matches nothing excuses nothing and hides the "
        "fact that the real reason is now undisclosed."
    )
    assert len({e.nodeid for e in MEASURED}) == len(MEASURED), (
        "two MEASURED entries name the same test; fold them together so the "
        "disclosure stays one-to-one with the skip site"
    )
    # No MEASURED reason may ALSO be covered by a rule: the whole point of the
    # shape is that these reasons are test-keyed, and a rule matching one
    # would quietly restore the suite-wide permission.
    leaked = [
        e.nodeid for e in MEASURED if any(r.matches(e.reason) for r in RULES)
    ]
    assert not leaked, (
        "MEASURED reason(s) that a RULES entry also matches, which makes them "
        "excusable from ANY test:\n  " + "\n  ".join(leaked)
    )


def test__importable_agrees_with_the_installed_importorskip(tmp_path, monkeypatch):
    """The load-bearing relationship in this file, and it moved under us.

    Every PINNED entry asks ``_importable`` whether pytest would have skipped.
    pytest 9.1 changed ``importorskip``'s default ``exc_type`` from
    ``ImportError`` to ``ModuleNotFoundError``, which splits "installed but
    broken" off from "absent" — and this file's predicate was written against
    the old behaviour, so it demanded a skip pytest no longer produces.
    Measured here against the pytest that is actually installed, rather than
    described in a docstring.
    """
    (tmp_path / "_probe_absent_sibling.py").write_text(
        "import _no_such_module_anywhere\n"
    )
    (tmp_path / "_probe_broken.py").write_text(
        "raise ImportError('installed, and its extension will not load')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))  # invalidates the import caches

    # 1. genuinely absent -> pytest SKIPS, and so must this predicate.
    with pytest.raises(pytest.skip.Exception):
        pytest.importorskip("_no_such_module_at_all_")
    assert not _importable("_no_such_module_at_all_")

    # 2. present, but something INSIDE it is missing -> still a
    #    ModuleNotFoundError, so pytest still skips, and so must this.
    with pytest.raises(pytest.skip.Exception):
        pytest.importorskip("_probe_absent_sibling")
    assert not _importable("_probe_absent_sibling")

    # 3. installed and BROKEN -> pytest does NOT skip; the ImportError
    #    propagates and the test errors. Reporting this as absent is what made
    #    the pin demand a skip that could never happen and blame a dependency
    #    that was right there, so it must read as PRESENT.
    with pytest.raises(ImportError):
        pytest.importorskip("_probe_broken")
    assert _importable("_probe_broken")


def _skips_this_session_cannot_explain() -> tuple[list[str], list[str]]:
    """(undisclosed, contradicted) over everything the session skipped so far.

    Worth running on whatever the session saw, however narrow it was; the
    CLAIM is a separate and larger question, answered in
    :func:`the_claim_this_session_can_make`.
    """
    undisclosed: list[str] = []
    contradicted: list[str] = []

    for nodeid, reason in sorted(SKIPPED.items()):
        if reason in _OWN_WITHDRAWALS and _THIS_MODULE in nodeid:
            continue  # this pin's own withdrawal, emitted a moment ago
        gate = _IMPORT_GATE.match(reason)
        if gate:
            dep = gate.group(1)
            if dep not in DECLARED_OPTIONAL_DEPENDENCIES:
                undisclosed.append(
                    f"{nodeid}: gate on undeclared dependency {dep!r} ({reason})"
                )
            elif _importable(dep):
                contradicted.append(
                    f"{nodeid}: skipped for a missing {dep!r} that imports fine here"
                )
            continue
        if _measured_for(nodeid, reason) is not None:
            continue  # disclosed, test-keyed and exact; direction not checkable
        rule = next((r for r in RULES if r.matches(reason)), None)
        if rule is None:
            undisclosed.append(f"{nodeid}: {reason}")
        elif rule.legitimate is not None and not rule.legitimate():
            contradicted.append(
                f"{nodeid}: skipped as {reason!r}, but its disclosed condition "
                f"({rule.when}) does not hold here"
            )
    return undisclosed, contradicted


_UNDISCLOSED_ADVICE = (
    "\n\nFor each one: make it not skip, or disclose it here. A named test "
    "skipping on a named dependency is a PINNED entry (both directions get "
    "asserted). A reason any test may carry is a RULES entry, with the exact "
    "reason and the condition that makes it legitimate. A new `importorskip` "
    "gate needs its library in DECLARED_OPTIONAL_DEPENDENCIES, which is a "
    "decision about the project's dependencies and not a formality. A skip "
    "whose condition is a measurement the test makes on its own subject is a "
    "MEASURED entry, test-keyed and exact. A skip nothing asserts is how two "
    "acceptance tests ran in no CI job for the project's whole life."
)


def the_claim_this_session_can_make(
    at_session_end: bool = False, no_call_phase: bool = False
) -> tuple[str, str]:
    """What this session supports, and whether it holds. The whole decision.

    Returns ``(verdict, message)`` with verdict one of:

    * ``"made"`` — the claim was evaluated against a complete record and holds;
    * ``"failed"`` — evaluated and broken, or the session hid its own evidence;
    * ``"withdrawn"`` — the session cannot support the claim, and says what it
      did not see;
    * ``"deferred"`` — the pin ran early, so the END of the session will make
      the claim instead (never returned when ``at_session_end``).

    Called from exactly two places, deliberately: :func:`test_no_session_skip_is_undisclosed`,
    which is the ordinary surface and gets a nodeid and a proper failure, and
    ``tests/conftest.py``'s session-end guard, for the sessions where the pin
    was reordered, filtered, deselected or ``--ignore``d out of making it. One
    decision, two callers, so the two cannot drift apart.

    Note the ORDER of the two halves, which is load-bearing and is what makes
    the guard worth consulting from a narrowed session at all: the disclosure
    half runs FIRST and is the same size whatever the session's scope was. A
    session that saw one file can still say whether that file's skips were
    disclosed, and saying so is the whole of what
    ``--ignore=tests/test_skip_inventory.py`` was hiding.

    ``no_call_phase`` is ``--collect-only`` / ``--setup-only`` /
    ``--setup-plan``, read off the invocation by ``tests/conftest.py``. It cuts
    the claim off and it does NOT cut the disclosure half off, and the
    difference is a route: ``--setup-only`` executes fixture setup, so a
    ``pytest.skip()`` in a FIXTURE fires under it and is recorded. The guard
    used to return before consulting anything in these modes, so that skip went
    unjudged — `1 skipped, 2012 deselected`, exit 0, nothing printed, measured
    at b277083 against a fixture-level plant. A mode that runs nothing may not
    certify the suite; it is still in a position to say that a skip it watched
    happen is undisclosed.
    """
    undisclosed, contradicted = _skips_this_session_cannot_explain()
    if undisclosed:
        return "failed", (
            "undisclosed skip(s) — this suite's skip set is pinned, by "
            "condition, in tests/test_skip_inventory.py:\n  "
            + "\n  ".join(undisclosed)
            + _UNDISCLOSED_ADVICE
        )
    if contradicted:
        return "failed", (
            "skip(s) whose disclosed condition is FALSE here — the test "
            "skipped anyway:\n  "
            + "\n  ".join(contradicted)
            + "\n\nThis is the other direction of the same drift: a gate that "
            "fires when the thing it gates on is present tests nothing and "
            "says so in green."
        )
    # and only NOW the size of the claim. The checks above are worth running on
    # whatever the session saw; the CLAIM is not the same size as the checks,
    # and the difference gets stated rather than assumed.
    if no_call_phase:
        return "withdrawn", (
            "the completeness pin is WITHDRAWN, not passed: this session ran "
            "no call phase (--collect-only, --setup-only or --setup-plan), so "
            "a `pytest.skip()` in a test BODY could not fire in it and "
            "\"no undisclosed skip in this session\" is not a claim it can "
            "make. Every skip it DID see — `--setup-only` executes fixture "
            "setup, so a skip in a FIXTURE fires under it — has been checked "
            "and is disclosed. Run without the mode for the claim."
        )
    return _what_this_session_is_in_a_position_to_claim(at_session_end)


def _what_this_session_is_in_a_position_to_claim(
    at_session_end: bool = False,
) -> tuple[str, str]:
    """The scope-and-order half, on its own — nothing here reads a SKIP.

    Separate from the disclosure half above because the two answer different
    questions and are asserted from different places: this one has no opinion
    about whether the session's skips are covered, only about whether the
    session is one that could support the claim at all.

    **The order is the FAILURES first, then the WITHDRAWALS**, and that is not
    cosmetic. A withdrawal is silenced by ``tests/conftest.py`` in any narrowed
    session — deliberately, because a banner naming the 82 files
    ``pytest tests/test_affine.py`` did not run is noise — and a failure never
    is. So a shortfall answered by a withdrawal before it can be answered by a
    failure is a shortfall that any narrowing makes silent. It used to be
    ordered ``unseen`` → ``filtered_out`` → ``still_owed``, and ``still_owed``
    is the undisclosed drop: the one shortfall pytest cannot report and the one
    the commit that wrote this order said "keeps the failure". Measured at
    b277083 with a plugin that removes one item from ``items[:]`` and never
    calls ``pytest_deselected``::

        whole tree                          EXIT 1, banner
        --ignore=tests/test_square_row.py   EXIT 0, SILENT   (swallowed by unseen)
        pytest tests/test_affine.py         EXIT 0, SILENT   (swallowed by unseen)
        -k "not test_op_add"                EXIT 0, SILENT   (by filtered_out)

    The ergonomic case for the silence is untouched and still sound; what was
    understated was its cost, and the fix is to stop paying it: the drop is
    decided first, so no narrowing can answer for it.
    """
    collisions = colliding_basenames()
    if collisions:
        return "failed", (
            "two test files under tests/ share a basename "
            f"({', '.join(collisions)}), and a basename is the only key a "
            "nodeid reliably gives back — so the scope check would read one "
            "file as collected when the other was. Rename one."
        )
    still_owed = pending_items(_THIS_MODULE)
    if still_owed and not at_session_end:
        return "deferred", (
            "the completeness pin is WITHDRAWN here and made at the END of "
            f"this session instead: {len(still_owed)} collected test(s) had "
            f"not run when it did, so it is not looking at the session, it is "
            f"looking at a prefix of it. Either the run is in a different "
            f"ORDER (`--nf` and pytest-randomly both re-sort `items` from a "
            f"`wrapper=True, tryfirst=True` hookimpl, which lands after every "
            f"ordinary one) or items were removed from it without being "
            f"reported deselected. tests/conftest.py says which, in a summary "
            f"section, once the loop is over — unless this is a distributed "
            f"worker, which runs a share of the session and claims nothing."
        )
    if still_owed:
        return "failed", (
            f"{len(still_owed)} collected test(s) never ran and were never "
            f"reported as deselected, so this session's own summary line is "
            f"short by that many and nothing said so. `pytest_deselected` is "
            f"how a plugin discloses that it removed items; a plugin that "
            f"filters `items[:]` without calling it produces a summary "
            f"byte-identical to a clean whole run.\n\nFirst few: "
            + ", ".join(still_owed[:5])
            + (", …" if len(still_owed) > 5 else "")
        )
    # --- and now the withdrawals, which tests/conftest.py may silence --------
    unseen = unseen_files()
    if unseen:
        return "withdrawn", (
            "the completeness pin is WITHDRAWN, not passed: this session "
            f"never collected {len(unseen)} of the suite's test files "
            f"({', '.join(unseen[:4])}"
            + (", …" if len(unseen) > 4 else "")
            + "), so it cannot say what the suite's skip set is — only what "
            "these files' was. Everything it DID see is disclosed. Run the "
            "whole suite for the claim."
        )
    filtered_out = deselected_items(_THIS_MODULE)
    if filtered_out:
        blame = (
            f"through the filter you passed ({', '.join(USER_FILTERS)})"
            if USER_FILTERS
            else "and no -k, -m or --deselect was passed, so it was `--sw`, "
            "`--lf`, or a plugin"
        )
        return "withdrawn", (
            "the completeness pin is WITHDRAWN, not passed: this session "
            f"collected the whole tree and then deselected {len(filtered_out)} "
            f"test(s) {blame}. The skips inside those tests went unobserved "
            f"and the claim cannot be made from what is left. Everything this "
            f"session DID observe is disclosed.\n\nFirst few: "
            + ", ".join(filtered_out[:5])
            + (", …" if len(filtered_out) > 5 else "")
            + "\n\nRun the suite without the filter for the claim."
        )
    # Last of the withdrawals, deliberately: an xfail stops a session claiming
    # the suite, but "you narrowed the invocation" is the more useful sentence
    # when both are true, and this one is only news when it is the ONLY thing
    # between the session and the claim.
    if XFAILED:
        return "withdrawn", (
            "the completeness pin is WITHDRAWN, not passed: this session "
            f"reported {len(XFAILED)} test(s) as xfailed. An xfail is not a "
            "skip and nothing here asks you to disclose it as one — but it is "
            "a test that handed back no verdict this pin can read, and with "
            "`run=False` it never started. pytest puts `N xfailed` in the same "
            "summary line it puts `N deselected` in, so this is DISCLOSED and "
            "therefore withdrawn on rather than failed on, by the same rule "
            "that stopped `-k` from failing this pin.\n\nFirst few: "
            + ", ".join(f"{n} ({r})" for n, r in sorted(XFAILED.items())[:5])
            + (", …" if len(XFAILED) > 5 else "")
            + "\n\nRun without the xfail for the claim. (An xpassed test RAN "
            "and passed, and is not counted here.)"
        )
    return "made", ""


def test_the_scope_key_is_lossy_in_exactly_one_way_and_it_is_checked(
    tmp_path, monkeypatch
):
    """The comparison key is a BASENAME, because a nodeid gives back no more.

    Two consequences, and neither may be left to chance. A test file one
    directory down still has to be inside the scope check — the glob used to
    stop at the top level, so ``tests/sub/test_x.py`` would have been invisible
    to it and a session that never collected it would have claimed the suite
    anyway. And two files sharing a basename would collapse into one key, so
    the session could read one as collected when it was the other; that cannot
    be prevented by a key that has no directory in it, so it is DETECTED, and
    the completeness claim fails while it is true.

    Exercised against a tree of its own because the real ``tests/`` has neither
    a subdirectory nor a collision, which is precisely how a check for them
    goes quietly inert.
    """
    import conftest

    (tmp_path / "test_top.py").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "test_nested.py").write_text("")
    monkeypatch.setattr(conftest, "_TESTS", tmp_path)

    assert conftest.files_the_suite_has() == {"test_top.py", "test_nested.py"}, (
        "a test file one directory down is outside the scope check, so a "
        "session that never collected it would still claim the suite"
    )
    assert conftest.colliding_basenames() == []

    (tmp_path / "sub" / "test_top.py").write_text("")
    assert conftest.colliding_basenames() == ["test_top.py"]
    # the scope half on its own: what this session's skips were is a different
    # question and would otherwise answer this one, whichever way it went
    verdict, message = _what_this_session_is_in_a_position_to_claim()
    assert verdict == "failed" and "share a basename" in message, (
        "two files share a basename and the completeness claim did not fail "
        f"on it — the session cannot tell which of them it collected: "
        f"{verdict}, {message[:200]}"
    )


_PRUNED_DIRECTORIES = (".junk", "build", "dist", "node_modules", "venv", "sample.egg")


def test_the_scope_check_prunes_exactly_what_pytest_prunes(tmp_path, monkeypatch):
    """A file pytest will never open is not part of the suite.

    The scope key globbed recursively with nothing subtracted, so anything at
    all under ``tests/`` counted as a file the suite has — including the
    directories pytest itself refuses to recurse into (``norecursedirs``:
    every dot-directory, ``build``, ``dist``, ``node_modules``, ``venv``,
    ``*.egg``, …). A scratch directory, a stale build tree or a vendored
    checkout under ``tests/`` is enough, and both directions were measured on
    the real tree with a CLEAN whole run:

    * ``tests/build/test_zz_helper.py``, a UNIQUE basename: a file the suite
      "has" that no invocation can collect, so ``unseen_files()`` never
      emptied, the completeness claim was WITHDRAWN, and the pin silently
      stopped asserting anything — exit 0, the whole check disabled by a file
      pytest never looked at.
    * ``tests/.junk/test_affine.py``, a COLLIDING basename: a clean whole run
      FAILED, exit 1, on a collision between a real file and one that cannot be
      collected.

    Both were measured at bd1fa04; the counts are in ``tests/conftest.py``
    beside :data:`conftest._NORECURSEDIRS`, where they are labelled, and are
    not repeated here — one copy of a total is a fact about a commit, and two
    copies are two chances to relabel one of them wrongly.

    So the two sides are COMPARED rather than described: what a real pytest
    reports collecting from a probe tree, against what the scope check says
    that tree has. Not a list of directory names — those are pytest's to
    change, and the ini value this invocation carries is what gets used and
    what is handed to the probe.
    """
    import conftest

    body = "def test_x():\n    assert True\n"
    (tmp_path / "test_top.py").write_text(body)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "test_nested.py").write_text(body)
    for name in _PRUNED_DIRECTORIES:
        directory = tmp_path / name
        directory.mkdir()
        # one unique basename, which a broken pruning adds to the file set,
        # and one colliding, which a broken pruning turns into a collision.
        safe = re.sub(r"\W", "_", name)
        (directory / f"test_only_inside_{safe}.py").write_text(body)
        (directory / "test_top.py").write_text(body)

    # THE THIRD CHILD OF THIS FILE, and it gets `PY_COLORS=0` for the same
    # reason as the other two: what comes back is READ, and `FORCE_COLOR` is
    # set in the ambient environment on this box. This one was green WITHOUT
    # it, and only by luck about which lines pytest paints. Measured,
    # `--collect-only -q` under `FORCE_COLOR=3`:
    #
    #   sub/test_nested.py::test_x         <- NO escapes: nodeids are plain
    #   test_top.py::test_x
    #   ESC[32mESC[32m2 tests collected    <- the SUMMARY line IS coloured
    #
    # so `line.split("::", 1)[0]` happens to see a clean path today. That is a
    # fact about pytest's current painting, not about this test, and the parse
    # below has no defence if it changes: an escape at the head of a nodeid
    # goes into `PurePosixPath(...).name` and the set equality then fails
    # against output that visibly contains the right names — which is exactly
    # what the other two children did before they were fixed. A machine-read
    # subprocess should not be handed a rendering meant for a terminal.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-o",
            "norecursedirs=" + " ".join(conftest._NORECURSEDIRS),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PY_COLORS": "0"},
    )
    collected = {
        pathlib.PurePosixPath(line.split("::", 1)[0]).name
        for line in proc.stdout.splitlines()
        if "::" in line
    }
    assert collected == {"test_top.py", "test_nested.py"}, (
        "the probe tree did not collect what this test is about, so the "
        f"comparison below would prove nothing:\n\n{_tail(proc)}"
    )

    monkeypatch.setattr(conftest, "_TESTS", tmp_path)
    assert conftest.files_the_suite_has() == collected, (
        "the scope check and pytest disagree about which files the suite has. "
        f"pytest collected {sorted(collected)}; the scope check says "
        f"{sorted(conftest.files_the_suite_has())}. A file only the scope "
        "check can see is a file no session can ever collect, so the "
        "completeness claim can never be made again."
    )
    assert conftest.colliding_basenames() == [], (
        "a basename collision was reported against a file pytest will not "
        f"collect: {conftest.colliding_basenames()}. That fails a clean whole "
        "run for a directory pytest never enters."
    )


def test_no_session_skip_is_undisclosed():
    """The completeness half: everything this session skipped is covered by a
    pin, a rule, a declared optional-dependency gate, or a MEASURED entry —
    and every gate that fired had a dependency that really is absent.

    Scoped and ORDERED, and it says which. The claim is about the SUITE's skip
    set, so the session has to have collected the suite and to have run it
    before this pin; see the four cases above for what each shortfall gets.
    """
    # Anti-vacuity, and it has to be about the RECORDER, not about this module.
    # The old form asserted that some test in this file had run — which its own
    # module satisfies, so it fired only when this test was the first thing in
    # the session and then blamed a conftest that was demonstrably loaded. The
    # file set is populated by the collection hook for EVERY invocation shape,
    # including a single-nodeid one, so an empty one means the plugin is not
    # there.
    assert SEEN_FILES, (
        "the outcome recorder collected nothing — tests/conftest.py is not "
        "loaded, so there is no session to read"
    )

    verdict, message = the_claim_this_session_can_make()
    if verdict != "deferred":
        # Whatever happens next, the claim has been made HERE and the
        # session-end guard in tests/conftest.py must not make it again.
        CLAIM_MADE.append(verdict)
    if verdict in ("withdrawn", "deferred"):
        _OWN_WITHDRAWALS.add(message)
        pytest.skip(message)
    assert verdict == "made", message


# --- the mechanisms no ordinary session exercises ----------------------------
#
# Everything below runs pytest sessions of its own, because the paths being
# checked are the ones that only exist when this pin is NOT in the session, or
# is in it in the wrong place. There is no way to reach them from inside a
# session this file is part of.
#
# The previous version of this stood a STUB in for the decision function —
# `the_claim_this_session_can_make` returning whatever an environment variable
# said. That tested the wiring (does the guard consult the pin, does a failed
# verdict reach the exit code, does the reader see it) and NOTHING about the
# decision, so a survey of the mechanism's fourteen arms found ten of them
# invisible: the `unseen` withdrawal, both `filtered_out` answers, both
# `still_owed` answers, the USER_FILTERS discrimination, all four early
# returns in `_close_the_session`, `DESELECTED` never being filled, and the
# collection hook's sort. Every one of those could be deleted with the whole
# suite still green.
#
# So the sessions below load the REAL decision function out of this very file
# and drive it, once per shortfall, and assert on what a reader gets: the exit
# code and the terminal summary. An invariant checked only where it is
# produced is not checked at its surface, and the surface is what a developer
# sees.
#
# WHAT THIS NET IS AND IS NOT. The commit that built it said "twenty
# mutations, each caught by the case written for it and by no other". The
# second half of that is false and was measured to be false: mutations of this
# mechanism COLLIDE, and the collisions are named below.
#
# THERE IS NO TOTAL HERE, AND ITS ABSENCE IS THE POINT. Two versions of this
# comment carried one — "twenty", then "33 caught, 28 signatures" — and both
# were counts of work done in a working tree that is not in this repository. A
# reader cannot re-run a number; a reader who derived their own thirty-three
# mutations would be checking a different claim; and annotating the total with
# "not checkable from this tree" leaves it sitting in the tree as an assertion
# that has to be accepted on provenance. §53 says record the FAILURES and not
# the totals, and deleting them obeys that more exactly than labelling them
# did. What is left is what a reader can drive: named pairs, each two named
# edits to code that ships, each re-derivable in a few minutes.
#
# Six pairs are indistinguishable by which cases fail:
#
#   the wasxfail report dropped          | the XFAILED withdrawal removed
#   the `unseen` withdrawal removed      | `unseen_files()` forced empty
#   the `filtered_out` answer removed    | `DESELECTED` never filled
#   the no-call-phase withdrawal removed | the guard never silencing it
#   `USER_FILTERS` never filled          | the `-k` failure put back
#   `_OUR_FAILURES` ignored in the       | `pytest_unconfigure` never
#     already-red carve-out              |   re-assigning `session.exitstatus`
#
# The sixth was derived when the anchor moved to `pytest_unconfigure` and is
# the same shape as the first four: one invariant — "the anchor gives the exit
# code back" — at two sites, the carve-out that decides whether to act and the
# assignment that acts. Both fail exactly
# `exit-zero-from-a-plugins-own-sessionfinish`,
# `exit-zero-from-a-tryfirst-sessionfinish` and both halves of
# `…raises_nothing_still_takes_the_exit_code`.
#
# The first four pairs are one invariant at two sites — the recorder's end and
# the decision's end. The defence offered for them was that a signature naming
# the invariant rather than the line is the right resolution, and half of that
# is measured while the other half was restatement. Here is the measured half:
# the suite DOES separate two sites of one invariant as soon as they differ
# observably. Removing the `still_owed` FAILURE from the decision and forcing
# the recorder's `pending_items()` to return `[]` are both "the undisclosed
# drop stops being noticed", and they do NOT collide —
#
#   the `still_owed` failure removed  items-dropped-without-being-reported,
#                                     an-ignored-file-does-not-swallow-…,
#                                     a-filter-does-not-swallow-…,
#                                     exit-zero-part-way-through-…
#   `pending_items()` forced empty    those four, AND pin-reordered-to-the-front
#
# — because the second also destroys the DEFERRAL, and a case asserts on that.
# So the four collisions are places where no behavioural case could tell the
# two sites apart, not places where nothing tried to.
#
# What none of that establishes is that the lost localisation is free, and the
# sentence that used to stand here asserted it by restating the collision as a
# virtue. It is dropped rather than defended. A collision means a red suite
# names an invariant and leaves the reader to find which of two lines broke it.
# That is a real cost; it is paid because the alternative is a case that
# asserts on a line number, which is worse.
#
# The fifth pair is not of that kind at all: those two fail the SAME two cases
# on DIFFERENT assertions inside them (the wording assertion and the exit-code
# assertion), which the case id alone does not carry — so not even the
# invariant-naming defence applies to it.

_PIN_PROXY = '''
"""Stands in for this module in the miniature sessions: the guard's counterpart.

Loads the REAL decision function out of the real file, under another name, so
that the session-end guard consults the code that ships rather than a stub
that agrees with it. `import conftest` inside it binds to the MINIATURE
conftest, which pytest has already imported under that name, so the decision
is made against this session's record.

Carries a test of its own so that its FILE is collected — a module that yields
nothing is a module the session cannot prove it collected.
"""
import importlib.util
import os
import pathlib
import sys

_spec = importlib.util.spec_from_file_location(
    "_the_real_pin", pathlib.Path(os.environ["REAL_PIN"])
)
_real = importlib.util.module_from_spec(_spec)
sys.modules["_the_real_pin"] = _real
_spec.loader.exec_module(_real)

the_claim_this_session_can_make = _real.the_claim_this_session_can_make


def test_the_proxy_file_is_collected():
    assert True
'''

# The proxy PLUS the real completeness pin, re-exported so that pytest collects
# it here under this file's name. For the cases about a session the pin IS part
# of: whether it claims or defers, and whether the guard then keeps quiet.
_PIN_PROXY_WITH_THE_PIN = _PIN_PROXY + '''

test_no_session_skip_is_undisclosed = _real.test_no_session_skip_is_undisclosed
'''

# ROUTE 10, as a file. The pin is sorted last among FILES; inside a file pytest
# collects in line order, so a test written BELOW the pin runs after it — with
# `CLAIM_MADE` already appended and the session-end guard already disarmed. 29
# of the 41 tests in the real file are in that position at a80d60c, and the
# skip planted there was `2022 passed, 3 skipped`, exit 0, no banner.
#
# The pin is CALLED here rather than re-exported. A re-exported function keeps
# the real file's `reportinfo`, and pytest orders a module's tests by
# `(fspath, lineno)` — so with a re-export, whether the plant lands after the
# pin depends on how two absolute paths happen to sort. Calling it puts both
# functions in this file, where line order is the order.
_PIN_PROXY_WITH_A_SKIP_AFTER_THE_PIN = _PIN_PROXY + '''
import pytest as _pytest


def test_aaa_the_pin_claims_here():
    """The REAL pin, so that `CLAIM_MADE` is appended in this session."""
    _real.test_no_session_skip_is_undisclosed()


def test_zzz_skips_after_the_pin_has_claimed():
    """Written below the pin, in the pin's OWN file: it runs after the claim."""
    _pytest.skip("a planted reason nobody disclosed")
'''

_SUBJECT_CLEAN = '''
def test_that_passes():
    assert True
'''

_SUBJECT_UNDISCLOSED = '''
import pytest


def test_that_passes():
    assert True


def test_that_skips_undisclosed():
    pytest.skip("a planted reason nobody disclosed")
'''

# A module-level gate on something that cannot exist, so it fires in every
# environment. It skips at COLLECTION, which is what makes it the subject for
# the --collect-only case: that session records a skip and runs nothing.
_SUBJECT_GATED = '''
import pytest

pytest.importorskip("_no_such_module_anywhere_at_all")


def test_never_runs():
    assert True
'''

# A skip in a FIXTURE, which is the one that fires under `--setup-only`: that
# mode runs no call phase but DOES execute setup, which is the whole point of
# it. The guard's stated reason for keeping quiet in the no-call-phase modes
# was "a `pytest.skip()` in a test body cannot fire in them" — true, and one
# phase too narrow.
_SUBJECT_FIXTURE_GATE = '''
import pytest


@pytest.fixture
def a_planted_gate():
    pytest.skip("a planted reason nobody disclosed")


def test_uses_the_gate(a_planted_gate):
    assert True
'''

# The three xfail shapes that matter, one per subject.
#
# `run=False` raises XFailed in SETUP: the body never runs at all, and the
# report is `skipped` with `wasxfail` set. The recorder returned on `wasxfail`
# and kept nothing, so the pin certified a session containing a test that never
# happened. `pytest.xfail()` in a body is the same shape one phase later.
_SUBJECT_XFAIL_NEVER_RUNS = '''
import pytest


def test_that_passes():
    assert True


@pytest.mark.xfail(run=False, reason="a planted reason nobody disclosed")
def test_that_never_runs():
    assert False
'''

_SUBJECT_XFAIL_IN_THE_BODY = '''
import pytest


def test_that_passes():
    assert True


def test_that_xfails_itself():
    pytest.xfail("a planted reason nobody disclosed")
'''

# …and the one that is NOT a shortfall: an xpass RAN and passed. `strict` is
# written out rather than left to the ini, because a strict xpass is a failure
# and this case is about the non-strict one.
_SUBJECT_XPASSES = '''
import pytest


def test_that_passes():
    assert True


@pytest.mark.xfail(strict=False, reason="expected to fail here, and did not")
def test_that_xpasses():
    assert True
'''

# Two passing tests, so that a `-k` can deselect one while a silent-drop plugin
# removes the other: the shape in which a DISCLOSED shortfall used to answer
# for an UNDISCLOSED one.
_SUBJECT_TWO_PASSING = '''
def test_that_passes():
    assert True


def test_that_also_passes():
    assert True
'''

# A real failure, with a test after it. Under `-x` the loop aborts at the first
# one and the second never runs — the shape the session-end close deliberately
# does NOT speak about, because the session is already red.
_SUBJECT_THAT_FAILS = '''
def test_that_fails():
    assert False


def test_that_would_have_run_next():
    assert True
'''

# A test file with no tests in it. Its collect report is
# `CollectReport(nodeid, "passed", result=[])` — byte-identical to the one
# `LFPluginCollSkipfiles` produces for a file `--lf` has decided NOT TO OPEN,
# which is why the recorder cannot key on the nodeid alone and keys on
# `report.skipped or report.result` instead. This file is therefore UNSEEN and
# the claim is withdrawn on it, which is the corner tests/conftest.py documents
# and takes deliberately: the safe direction, and it names the file.
_SUBJECT_NO_TESTS_AT_ALL = '''
"""A module with no tests in it at all."""
'''

# The exact wording pytest's `importorskip` emits, on a DECLARED optional
# dependency. Paired with a supplied `blackjax.py`, this is the import-gate arm
# of the contradiction check: a gate that fired for a library that is right
# there.
_SUBJECT_GATE_ON_A_LIBRARY_THAT_IS_THERE = '''
import pytest


def test_that_passes():
    assert True


def test_skipped_for_a_library_that_is_importable():
    pytest.skip("could not import 'blackjax': No module named 'blackjax'")
'''

# The rule arm: a reason a RULES entry discloses, under a condition that the
# supplied `z3.py` makes false.
_SUBJECT_SKIPS_FOR_A_WHEEL_THAT_IS_THERE = '''
import pytest


def test_that_passes():
    assert True


def test_skipped_for_a_wheel_that_is_installed():
    pytest.skip("needs z3")
'''

# One-line stand-ins, supplied on the session's own PYTHONPATH. Not installs:
# `import blackjax` has to SUCCEED and `find_spec("z3")` has to return a spec,
# and that is all either of them is for.
_A_LIBRARY_THAT_IMPORTS = '"""A stand-in that exists and imports cleanly."""\n'

_DESELECTS_LIKE_STEPWISE = '''
"""Deselects through the hook with no -k/-m/--deselect on the command line.
`_pytest/stepwise.py` and `--lf` call `pytest_deselected` exactly like this,
which is why the EFFECT cannot tell them apart from a filter the developer
passed and the INVOCATION is read instead."""
def pytest_collection_modifyitems(config, items):
    doomed = [i for i in items if i.name == "test_that_passes"]
    if doomed:
        items[:] = [i for i in items if i not in doomed]
        config.hook.pytest_deselected(items=doomed)
'''

_DROPS_WITHOUT_SAYING_SO = '''
"""Drops items from `items[:]` and never calls `pytest_deselected`. The
hookspec says a plugin that removes items must report them; nothing enforces
it, and the summary line of a session that did this is byte-identical to a
clean whole run's."""
def pytest_collection_modifyitems(items):
    items[:] = [i for i in items if i.name != "test_that_passes"]
'''

_REORDERS_AFTER_EVERYONE = '''
"""`NFPlugin`'s shape, which is also pytest-randomly's: wrapper=True,
tryfirst=True, so it re-sorts `items` AFTER every non-wrapper
pytest_collection_modifyitems hookimpl, including the conftest's. Puts the pin
FIRST, which is what --nf does when the pin's file is the freshly-saved one."""
import pytest


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(items):
    result = yield
    items.sort(key=lambda item: "test_skip_inventory.py" not in item.nodeid)
    return result
'''

_LOOKS_LIKE_AN_XDIST_WORKER = '''
"""The documented worker marker: `xdist/remote.py` sets `config.workerinput`
to a dict before the session runs. xdist is not installed here, so this is a
stand-in of that shape and nothing more."""
def pytest_configure(config):
    config.workerinput = {"workerid": "gw0", "workercount": 2}
'''

# ROUTE 9. `pytest.exit(reason, returncode=0)` is public API, and inside the run
# loop it does two things at once: `Exit` propagates through the recorder's
# `pytest_runtestloop` wrapper so the session-end close never runs, and
# `wrap_session` assigns `session.exitstatus` from the returncode, which
# overrides the `session.testsfailed` the close carries its verdict in. So it is
# the `-x` carve-out with the exit code taken away — and the `-x` carve-out is
# safe only BECAUSE the exit code is non-zero.
#
# After the loop, this leaves a summary line byte-identical to the same session
# without the plugin. Measured at a80d60c on `tests/test_affine.py` with an
# undisclosed skip planted: `41 passed, 1 skipped` both times, exit 1 with a
# banner naming the skip, and exit 0 with nothing.
_EXITS_ZERO_AT_THE_END_OF_THE_LOOP = '''
"""Leaves the run loop by `pytest.exit(..., returncode=0)` once every item has
run: a complete record, a green exit code, and no session-end decision."""
import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtestloop(session):
    result = yield
    pytest.exit("stopping the session here, on purpose", returncode=0)
    return result
'''

# The same call one phase earlier, which is the version that also drops tests:
# out of the loop with items still to run and a returncode that says the session
# succeeded. On the whole tree at a80d60c this was `1 passed`, exit 0, no
# banner, with 2023 of the 2024 collected tests never run.
_EXITS_ZERO_PART_WAY_THROUGH_THE_LOOP = '''
"""Leaves the run loop after the first call phase, with returncode 0."""
import pytest

_ran = []


def pytest_runtest_logreport(report):
    if report.when == "call":
        _ran.append(report.nodeid)
        if len(_ran) >= 1:
            pytest.exit("stopping the session here, on purpose", returncode=0)
'''

# --- the same call ONE LEVEL OUT: from a plugin's own pytest_sessionfinish ---
#
# This was the disclosed-and-open hole under the two above. `wrap_session`
# catches the `Exit` in the very `finally` that called the hook and re-assigns
# `session.exitstatus` from the returncode; and the terminal reporter's own
# `pytest_sessionfinish` is a WRAPPER around every ordinary one, so an `Exit`
# raised inside the chain never reaches the code after its `yield` and
# `pytest_terminal_summary` is never called — whichever order the ordinary
# hookimpls ran in. Both halves went, one `-p` apart. Measured at 1b1c843 on
# `tests/test_affine.py` with an undisclosed skip planted: EXIT 1 with a banner
# without it, EXIT 0 with nothing with it; and the same again with `tryfirst`,
# which is the other hookimpl order.
#
# Answered by `tests/conftest.py`'s `pytest_unconfigure`: `wrap_session` calls
# `config._ensure_unconfigure()` AFTER `pytest_sessionfinish` and BEFORE it
# reads `session.exitstatus`.
_EXITS_ZERO_FROM_A_PLUGINS_SESSIONFINISH = '''
"""Raises `Exit` from a plugin's own pytest_sessionfinish: the hook the
session-end guard used to be anchored in, defeated from inside its own chain."""
import pytest


def pytest_sessionfinish(session, exitstatus):
    pytest.exit("stopping from a plugin sessionfinish", returncode=0)
'''

_EXITS_ZERO_FROM_A_TRYFIRST_SESSIONFINISH = '''
"""The same, ordered ahead of every ordinary hookimpl rather than behind."""
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    pytest.exit("stopping from a tryfirst sessionfinish", returncode=0)
'''

# --- and the defeat that does not raise at all -------------------------------
#
# The version of this file one commit ago carried a consolation: the
# sessionfinish defeat "does not produce the byte-identical green this whole
# mechanism exists to end, because pytest itself writes `Exit: <reason>` on the
# way out". True of the two above. False of this one, which is a single
# assignment and raises nothing — measured at 1b1c843, same plant: EXIT 0, the
# banner still printed, and NO `Exit:` line anywhere in the output.
_ASSIGNS_EXIT_ZERO_WITHOUT_RAISING = '''
"""Assigns session.exitstatus = 0 from a pytest_sessionfinish and raises
nothing, so pytest has no `Exit:` to report and nothing looks unusual."""


def pytest_sessionfinish(session, exitstatus):
    session.exitstatus = 0
'''

# …and the whole of it: the exit code AND the note list, which is exactly the
# byte-identical green. At 1b1c843 this was EXIT 0 with zero banners, and the
# `diff` against the same session without the plugin was exactly the banner
# lines and nothing else.
_ASSIGNS_EXIT_ZERO_AND_CLEARS_THE_NOTES = '''
"""The whole defeat in two lines: un-assign the exit code, and empty the note
list before anything prints it."""


def pytest_sessionfinish(session, exitstatus):
    session.exitstatus = 0
    import conftest

    conftest._NOTES.clear()
'''

# --- the two that beat the new anchor too, and are supposed to ---------------
#
# `session.exitstatus` is last-writer-wins. These are here so that the tree
# carries the DEMONSTRATION rather than the assertion: `pytest_cmdline_main` as
# a wrapper returns over `wrap_session` entirely, and a `trylast`
# `pytest_unconfigure` runs after the anchor. Neither is a bug to be fixed;
# both are what "every anchor has a later one" means, and the verdict FILE is
# the answer to them.
_RETURNS_98_FROM_A_CMDLINE_MAIN_WRAPPER = '''
"""Returns over wrap_session itself, which is later than every hook in it."""
import pytest


@pytest.hookimpl(wrapper=True)
def pytest_cmdline_main(config):
    yield
    return 98
'''

_ASSIGNS_EXIT_ZERO_FROM_A_TRYLAST_UNCONFIGURE = '''
"""Runs after the anchor, in the anchor's own hook."""
import pytest


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    session = config.pluginmanager.get_plugin("session")
    if session is not None:
        session.exitstatus = 0
'''

# --- the fifth zero-arriving abort, which is not a report shape or a decision -
#
# `initstate = 2` is assigned only AFTER pytest_sessionstart returns, so a
# session that leaves FROM that hook has `if initstate >= 2` False and
# pytest_sessionfinish is never called at all. Nothing is owed on this route —
# nothing was collected — which is exactly why the check has to be that the
# ANCHOR ARRIVED rather than that it said something.
_EXITS_ZERO_FROM_SESSIONSTART = '''
"""pytest.exit from inside pytest_sessionstart: the hook after it never runs."""
import pytest


def pytest_sessionstart(session):
    pytest.exit("stopping from sessionstart", returncode=0)
'''

_MARKS_WHICH_HOOKS_RAN = '''
"""Prints from both hooks, so that "never called at all" is observed rather
than inferred from a missing banner."""
import sys


def pytest_sessionfinish(session, exitstatus):
    sys.stderr.write("MARKER-SESSIONFINISH-RAN\\n")
    sys.stderr.flush()


def pytest_unconfigure(config):
    sys.stderr.write("MARKER-UNCONFIGURE-RAN\\n")
    sys.stderr.flush()
'''

# The stated limit of the verdict file, as a plugin rather than as a sentence.
_DELETES_THE_VERDICT_FILE = '''
"""Reads the same environment variable the conftest reads, and unlinks what it
finds there. The file defends against plugins that know nothing about this pin;
this one knows."""
import os
import pathlib

import pytest


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    destination = os.environ.get("STELLING_SKIP_INVENTORY_VERDICT")
    if destination:
        pathlib.Path(destination).unlink(missing_ok=True)
'''


def _run_a_miniature_session(
    tmp_path, files, argv=(), plugins=(), modules=(), env_extra=None
):
    """A whole pytest session over a tree of `files`, with the REAL conftest.

    `files` is {relative path under tests/: source}. Plugin modules are written
    at the root and loaded with `-p`, which is how an installed plugin arrives.

    `modules` is [(import name, source)] written at the same root and NOT
    loaded as plugins — the session's own ``PYTHONPATH`` is what makes them
    importable, and that is the whole point of them. The two directions this
    file can only check by CONTRADICTING a skip — a gate on a dependency that
    imports fine, and a rule whose condition does not hold — both need a
    library to be PRESENT, and every attempt to build them by READING the
    ambient environment goes vacuous in some lane: whichever library is chosen,
    some CI job has it and some has not. This harness does not have to read the
    environment. It SUPPLIES it. A one-line ``blackjax.py`` here makes
    ``import blackjax`` succeed in every lane there is, and a one-line
    ``z3.py`` makes ``find_spec("z3")`` succeed in every lane there is, so both
    cases are non-vacuous everywhere and neither consults a clock.

    ``env_extra`` is {name: value} added to the child's environment, and it
    exists for one variable: ``STELLING_SKIP_INVENTORY_VERDICT``, the channel
    that is NOT ``session.exitstatus``. It is passed rather than set on this
    process because the cases that need it are cases where the child's exit
    code and the child's screen are both under attack, and a variable this
    process exported would be inherited by every other miniature session too.

    Nothing is installed: these are files under ``tmp_path``, on the child
    process's ``PYTHONPATH``, gone when the test ends.
    """
    tests = tmp_path / "tests"
    tests.mkdir()
    # THE REAL CONFTEST, AND WHAT THE REAL CONFTEST NEEDS BESIDE IT.
    # `tests/conftest.py` loads `tests/_state_guard.py` by path — the
    # process-global state guard's inventory and autouse fixture — so a
    # miniature tree holding the conftest alone holds a conftest that
    # cannot load, and every session here dies at collection with an
    # ImportError instead of measuring what it was written to measure.
    # Driven: 40 of this file's tests failed that way, all of them on the
    # child's `ImportError while loading conftest`.
    for sibling in ("conftest.py", "_state_guard.py"):
        (tests / sibling).write_text((TESTS / sibling).read_text())
    for relative, source in files.items():
        path = tests / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    for name, source in plugins:
        (tmp_path / f"{name}.py").write_text(source)
    for name, source in modules:
        (tmp_path / f"{name}.py").write_text(source)

    # `PY_COLORS=0`: every assertion against a miniature session is a SUBSTRING
    # test over the child's stdout, and pytest colours its output when it
    # believes a human is watching. It believes that whenever `FORCE_COLOR` is
    # set in the ambient environment — which is not exotic, it is set on the box
    # this was found on. Nodeids in the `-rfsE` report then arrive split by SGR
    # escapes, so `"…::test_no_session_skip_is_undisclosed" in output` is False
    # against output that visibly contains it.
    #
    # MEASURED, `test_the_pin_makes_its_own_claim_when_it_is_ordered_last`,
    # nothing else changed: with `FORCE_COLOR=3` in the environment it FAILS —
    # "the pin did not fail at its own nodeid, so it did not run last: the
    # collection hook is no longer ordering it" — and with `PY_COLORS=0` it
    # passes. The mechanism that message accuses was working the whole time,
    # which is the worst shape a failure can take: a true-sounding accusation
    # against innocent machinery.
    #
    # Set here rather than per-assertion because it is a fact about how this
    # harness READS the child, not about any one case: a machine-read
    # subprocess should not be handed a rendering meant for a terminal.
    # `env_extra` is applied after, so a case that wants colour can ask.
    env = {
        **os.environ,
        "REAL_PIN": str(pathlib.Path(__file__).resolve()),
        "PY_COLORS": "0",
    }
    env.update(env_extra or {})
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), *([os.environ["PYTHONPATH"]] if "PYTHONPATH" in os.environ else [])]
    )
    # `-rfsE`: `s` because a WITHDRAWN or DEFERRED verdict from the pin is
    # delivered as a skip REASON and a bare `-q` prints the count without the
    # sentence, and `fE` because `-r` replaces the default report characters
    # rather than adding to them — asking for skips alone silently drops the
    # failure lines, which is how the nodeid a verdict lands on stops being
    # visible to the assertions below.
    argument_list = [
        sys.executable, "-m", "pytest", "-q", "-rfsE", "-p", "no:cacheprovider",
        *deterministic_order_args()
    ]
    for name, _ in plugins:
        argument_list += ["-p", name]
    argument_list += [*argv, "tests"]
    return subprocess.run(
        argument_list,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


_BANNER = "skip inventory:"

# (id, files, argv, plugins, exit-zero, must appear, must NOT appear)
#
# One row per shortfall the decision function can reach and per early return
# the guard has, plus the two ordering mechanisms. The `says` column is the
# sentence the reader gets, which is the half a green exit code does not carry.
_SESSIONS = (
    (
        # the guard's ordinary job: pin gone, record complete, nothing wrong
        "made",
        {"test_skip_inventory.py": _PIN_PROXY, "test_zzz_subject.py": _SUBJECT_CLEAN},
        (),
        (),
        True,
        ("pin absent from this session, claim made at the end",),
        ("WITHDRAWN", "FAILED"),
    ),
    (
        # …and the same session with a skip nobody disclosed in it
        "undisclosed",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        (),
        (),
        False,
        ("completeness claim FAILED", "a planted reason nobody disclosed"),
        (),
    ),
    (
        # THE --ignore ROUTE. The pin's own file is not collected, which reads
        # as an ordinary narrowing; the guard used to return before it could
        # say anything, so the whole suite was exit 0 with the planted skip
        # printed on the screen and no verdict anywhere. Measured at bd1fa04;
        # the counts are in tests/conftest.py's `_close_the_session`, labelled.
        "ignore-the-pins-own-file",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        ("--ignore=tests/test_skip_inventory.py",),
        (),
        False,
        ("completeness claim FAILED", "a planted reason nobody disclosed"),
        (),
    ),
    (
        # the same, by the glob spelling
        "ignore-glob",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        ("--ignore-glob=*skip_inventory*",),
        (),
        False,
        ("completeness claim FAILED",),
        (),
    ),
    (
        # …and the sibling underneath it: nothing reaches a call phase, so RAN
        # is empty and "nothing ran" was inferred from the effect.
        "nothing-ran-but-something-skipped",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        ("--ignore=tests/test_skip_inventory.py", "-k", "test_that_skips_undisclosed"),
        (),
        False,
        ("completeness claim FAILED", "a planted reason nobody disclosed"),
        (),
    ),
    (
        # a narrowed session with nothing else wrong says nothing at all, on
        # purpose: `pytest tests/test_x.py` is the commonest invocation there
        # is and a banner naming what it did not run is not news.
        "narrowed-and-quiet",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_CLEAN,
            "test_zzz_other.py": _SUBJECT_CLEAN,
        },
        ("--ignore=tests/test_zzz_other.py",),
        (),
        True,
        (),
        (_BANNER,),
    ),
    (
        # deselection through a filter the developer passed: WITHDRAWN, naming
        # the filter. It used to FAIL, which is why no -k expression could exit
        # 0 from the repo root on a clean tree.
        "deselected-by-a-user-filter",
        {"test_skip_inventory.py": _PIN_PROXY, "test_zzz_subject.py": _SUBJECT_CLEAN},
        ("-k", "not test_that_passes"),
        (),
        True,
        ("WITHDRAWN", "through the filter you passed", "-k 'not test_that_passes'"),
        ("FAILED",),
    ),
    (
        # deselection by something that is not a filter: same verdict, and the
        # message has to say so — this is the --sw false failure's channel.
        "deselected-by-something-that-is-not-a-filter",
        {"test_skip_inventory.py": _PIN_PROXY, "test_zzz_subject.py": _SUBJECT_CLEAN},
        (),
        (("deselects_like_stepwise", _DESELECTS_LIKE_STEPWISE),),
        True,
        ("WITHDRAWN", "no -k, -m or --deselect was passed"),
        ("through the filter you passed", "FAILED"),
    ),
    (
        # items removed and never reported. THIS one keeps the failure: pytest
        # discloses a deselection in its own summary line and cannot disclose
        # a drop nobody reported.
        "items-dropped-without-being-reported",
        {"test_skip_inventory.py": _PIN_PROXY, "test_zzz_subject.py": _SUBJECT_CLEAN},
        (),
        (("drops_without_saying_so", _DROPS_WITHOUT_SAYING_SO),),
        False,
        ("completeness claim FAILED", "never reported as deselected"),
        (),
    ),
    (
        # …and the same drop in a session that ALSO narrowed collection. The
        # undisclosed drop used to be the LAST question asked, so `unseen`
        # answered first, withdrew, and tests/conftest.py silenced the
        # withdrawal: `--ignore` of one unrelated file bought a silent pass on
        # the one shortfall pytest cannot report.
        "an-ignored-file-does-not-swallow-an-undisclosed-drop",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_CLEAN,
            "test_zzz_other.py": _SUBJECT_CLEAN,
        },
        ("--ignore=tests/test_zzz_other.py",),
        (("drops_without_saying_so", _DROPS_WITHOUT_SAYING_SO),),
        False,
        ("completeness claim FAILED", "never reported as deselected"),
        (),
    ),
    (
        # the same swallow through the other withdrawal: a `-k` that deselects
        # one test while the plugin silently removes another. `filtered_out`
        # answered first and the drop went unreported.
        "a-filter-does-not-swallow-an-undisclosed-drop",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_TWO_PASSING,
        },
        ("-k", "not test_that_also_passes"),
        (("drops_without_saying_so", _DROPS_WITHOUT_SAYING_SO),),
        False,
        ("completeness claim FAILED", "never reported as deselected"),
        (),
    ),
    (
        # the pin reordered to the front: it DEFERS rather than claiming, and
        # the end of the session answers.
        "pin-reordered-to-the-front",
        {
            "test_skip_inventory.py": _PIN_PROXY_WITH_THE_PIN,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        (),
        (("reorders_after_everyone", _REORDERS_AFTER_EVERYONE),),
        False,
        ("made at the END of this session instead", "completeness claim FAILED"),
        (),
    ),
    (
        # Two files with one basename, which is the key's single lossy axis.
        # The subdirectory is a PACKAGE on purpose: without an `__init__.py`
        # pytest refuses the second file outright ("import file mismatch",
        # exit 2) and the collision never reaches this check, so the shape
        # worth checking is the one pytest imports happily and the basename
        # key cannot tell apart.
        "colliding-basenames",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_CLEAN,
            "sub/__init__.py": "",
            "sub/test_zzz_subject.py": _SUBJECT_CLEAN,
        },
        (),
        (),
        False,
        ("completeness claim FAILED", "share a basename"),
        (),
    ),
    (
        # …and the same two basenames with the second one somewhere pytest will
        # not go, plus a unique basename in another such place. Neither is part
        # of the suite, because no invocation can collect either.
        "files-pytest-will-never-collect",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_CLEAN,
            ".junk/test_zzz_subject.py": _SUBJECT_CLEAN,
            "build/test_only_in_build.py": _SUBJECT_CLEAN,
        },
        (),
        (),
        True,
        ("pin absent from this session, claim made at the end",),
        ("WITHDRAWN", "FAILED"),
    ),
    (
        # a distributed worker claims nothing, and this is the KNOWN OPEN hole:
        # the skip is real, the worker is green, and only a controller could
        # say so. Asserted as it is, not as it should be.
        "an-xdist-worker-claims-nothing",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        (),
        (("looks_like_an_xdist_worker", _LOOKS_LIKE_AN_XDIST_WORKER),),
        True,
        (),
        (_BANNER,),
    ),
    (
        # --collect-only runs no call phase, so no `pytest.skip()` in a test
        # body can fire and "no undisclosed skip in this session" is not a
        # claim it is entitled to make. It does not claim, and — having nothing
        # else to say — it says nothing.
        "collect-only-claims-nothing",
        {"test_skip_inventory.py": _PIN_PROXY, "test_zzz_subject.py": _SUBJECT_CLEAN},
        ("--collect-only",),
        (),
        True,
        (),
        (_BANNER,),
    ),
    (
        # …and the half that the early return used to take with it. The subject
        # skips at COLLECTION, on a gate nobody declared, so this session has a
        # real undisclosed skip on record and no claim to make about the suite.
        # Not claiming is not the same as not answering.
        "collect-only-still-judges-the-skips-it-saw",
        {"test_skip_inventory.py": _PIN_PROXY, "test_zzz_subject.py": _SUBJECT_GATED},
        ("--collect-only",),
        (),
        False,
        ("completeness claim FAILED", "gate on undeclared dependency"),
        (),
    ),
    (
        # …and the mode underneath it that the reason above did NOT cover.
        # `--setup-only` runs no call phase but DOES execute fixture setup, so
        # a `pytest.skip()` in a FIXTURE fires, is recorded, and used to
        # disappear into the guard's early return. The claim is still off the
        # table for this mode; the DISCLOSURE half is not.
        "setup-only-runs-fixtures-so-their-skips-are-judged",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_FIXTURE_GATE,
        },
        ("--setup-only",),
        (),
        False,
        ("completeness claim FAILED", "a planted reason nobody disclosed"),
        (),
    ),
    (
        # and the other side of that fix: a no-call-phase session with nothing
        # to report still says nothing. Without this, closing the route above
        # would put a banner on every `--collect-only` and `--setup-only`.
        "setup-only-with-nothing-to-report-stays-quiet",
        {"test_skip_inventory.py": _PIN_PROXY, "test_zzz_subject.py": _SUBJECT_CLEAN},
        ("--setup-only",),
        (),
        True,
        (),
        (_BANNER,),
    ),
    (
        # a file with no tests in it reports exactly as an `--lf`-declined file
        # does, so it reads as UNSEEN and the claim is withdrawn naming it.
        # That is the documented corner of the `report.skipped or report.result`
        # guard in tests/conftest.py, and it is the surface at which that guard
        # is checked: relax it to `if True` and `--lf` goes back to claiming the
        # suite having opened one file of four.
        "a-file-with-no-tests-in-it-reads-as-unseen",
        {
            "test_skip_inventory.py": _PIN_PROXY_WITH_THE_PIN,
            "test_zzz_subject.py": _SUBJECT_CLEAN,
            "test_zzz_empty.py": _SUBJECT_NO_TESTS_AT_ALL,
        },
        (),
        (),
        True,
        ("never collected 1 of the suite's test files", "test_zzz_empty.py"),
        ("FAILED",),
    ),
    (
        # THE EIGHTH ROUTE. `xfail(run=False)` raises in SETUP and its report is
        # `skipped` with `wasxfail` set; the recorder returned on `wasxfail` and
        # kept nothing, so the claim was MADE over a test that never ran. pytest
        # discloses it (`1 xfailed`) exactly as it discloses `N deselected`, so
        # the answer is WITHDRAWN — the same cut this file makes everywhere.
        "an-xfail-that-never-ran",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_XFAIL_NEVER_RUNS,
        },
        (),
        (),
        True,
        ("WITHDRAWN", "as xfailed", "test_that_never_runs"),
        ("claim made at the end", "FAILED"),
    ),
    (
        # the same report shape one phase later, and the same answer.
        "an-xfail-taken-in-the-test-body",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_XFAIL_IN_THE_BODY,
        },
        (),
        (),
        True,
        ("WITHDRAWN", "as xfailed", "test_that_xfails_itself"),
        ("claim made at the end", "FAILED"),
    ),
    (
        # and the shape that is NOT a shortfall. An xpassed test ran its body
        # and passed; treating every `wasxfail` report as a withdrawal would
        # take the completeness claim away from it for nothing, which is the
        # over-correction this row exists to stop.
        "an-xpass-is-a-test-that-ran",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_XPASSES,
        },
        (),
        (),
        True,
        ("pin absent from this session, claim made at the end",),
        ("WITHDRAWN", "FAILED"),
    ),
    (
        # a filter that selects nothing at all. Nothing runs and nothing skips,
        # which is the state the old `if not RAN: return` was really aiming at
        # — and it withdraws through the ordinary scope answer like every other
        # deselection rather than through a branch of its own. pytest's own
        # "no tests ran" is what makes this non-zero; the banner is what says
        # the completeness claim did not happen.
        "a-filter-that-selects-nothing",
        {"test_skip_inventory.py": _PIN_PROXY, "test_zzz_subject.py": _SUBJECT_CLEAN},
        ("-k", "nothing_matches_this_expression"),
        (),
        False,
        ("WITHDRAWN", "through the filter you passed"),
        ("FAILED",),
    ),
    (
        # ROUTE 9, the pair. `pytest.exit(returncode=0)` after every item has
        # run: the record is COMPLETE and the undisclosed skip is in it, and
        # before tests/conftest.py had a `pytest_sessionfinish` the decision
        # simply never ran. The summary line is byte-identical to the same
        # session without the plugin, which is exit 1 with this banner.
        #
        # THE THIRD `says` ENTRY IS THE ONE THAT MAKES THIS ROW ITS OWN ID.
        # Without it the row passed with its plugin REMOVED — a trigger-strip
        # probe at 1b1c843 says so — because `EXIT != 0`, "completeness claim
        # FAILED" and the skip reason are all three satisfied by the
        # undisclosed skip ALONE. The row was still sensitive to the plugin
        # (put `SF_GONE` in the conftest with the plugin present and it fails),
        # so it was not a could-not-fail shape; but its id promised a
        # discrimination its assertions did not make. `Exit:` is the line
        # pytest writes only when a session left by `pytest.exit`, so asserting
        # it is asserting that the route this row is named for was taken.
        "exit-zero-from-inside-the-run-loop-buys-no-silence",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        (),
        (("exits_zero_at_the_end", _EXITS_ZERO_AT_THE_END_OF_THE_LOOP),),
        False,
        (
            "completeness claim FAILED",
            "a planted reason nobody disclosed",
            "Exit: stopping the session here, on purpose",
        ),
        (),
    ),
    (
        # …and the same call one phase earlier, which is the half `-x` never
        # reaches: out of the loop with items still owed and a returncode
        # saying the session succeeded. That is the `still_owed` failure, and
        # it is the one shortfall pytest cannot report on its own.
        "exit-zero-part-way-through-drops-the-rest-of-the-suite",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_TWO_PASSING,
        },
        (),
        (("exits_zero_part_way", _EXITS_ZERO_PART_WAY_THROUGH_THE_LOOP),),
        False,
        ("completeness claim FAILED", "never reported as deselected"),
        (),
    ),
    (
        # and the over-correction this one stops. A session that left the loop
        # early but left NOTHING owed and nothing undisclosed has no shortfall,
        # and the close has to run and say so rather than invent one: without
        # this row, "treat every abort as a drop" passes everything above.
        "exit-zero-with-a-complete-clean-record-stays-zero",
        {"test_skip_inventory.py": _PIN_PROXY, "test_zzz_subject.py": _SUBJECT_CLEAN},
        (),
        (("exits_zero_at_the_end", _EXITS_ZERO_AT_THE_END_OF_THE_LOOP),),
        True,
        ("pin absent from this session, claim made at the end",),
        ("WITHDRAWN", "FAILED"),
    ),
    (
        # ROUTE 11 — the hole the two above used to leave, one level out. The
        # `Exit` comes from a PLUGIN's own `pytest_sessionfinish`, which is the
        # hook the guard used to be anchored in: `wrap_session` re-assigns
        # `session.exitstatus` from the returncode in the very `finally` that
        # called the hook, and the terminal reporter's `pytest_sessionfinish`
        # is a wrapper around every ordinary one, so `pytest_terminal_summary`
        # never runs either. EXIT 0, no banner, at 1b1c843. The anchor is now
        # `pytest_unconfigure`, which `wrap_session` calls AFTER the hook and
        # BEFORE it reads `session.exitstatus`.
        "exit-zero-from-a-plugins-own-sessionfinish",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        (),
        (("exits_zero_at_finish", _EXITS_ZERO_FROM_A_PLUGINS_SESSIONFINISH),),
        False,
        (
            "completeness claim FAILED",
            "a planted reason nobody disclosed",
            "Exit: stopping from a plugin sessionfinish",
        ),
        (),
    ),
    (
        # the same, ahead of every ordinary hookimpl instead of behind them.
        # A separate row because the previous commit's reasoning about this
        # route turned on the hookimpl ORDER, and an ordering claim asserted at
        # one order only is an ordering claim nothing is holding.
        "exit-zero-from-a-tryfirst-sessionfinish",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        (),
        (("exits_zero_tryfirst", _EXITS_ZERO_FROM_A_TRYFIRST_SESSIONFINISH),),
        False,
        (
            "completeness claim FAILED",
            "a planted reason nobody disclosed",
            "Exit: stopping from a tryfirst sessionfinish",
        ),
        (),
    ),
    (
        # the other side of the same carve-out, and the reason it is safe: an
        # abort that is ALREADY red. `-x` stops at the first failure and leaves
        # a test owed, exactly as the exit-zero route does — and here the guard
        # must stay silent, because a session pytest has already failed is told
        # nothing by a second verdict. This is what makes the exit-code test in
        # `pytest_sessionfinish` load-bearing rather than decorative.
        "an-abort-that-is-already-red-gets-no-second-verdict",
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_THAT_FAILS,
        },
        ("-x",),
        (),
        False,
        (),
        (_BANNER,),
    ),
    (
        # ROUTE 10. The pin claims from inside the session it is claiming
        # about: it is sorted last among FILES, and its own file goes on. A
        # skip after it was recorded in SKIPPED, correctly, and read by
        # nobody — `2022 passed, 3 skipped`, exit 0, no banner, at a80d60c.
        "a-skip-after-the-pin-has-already-claimed",
        {
            "test_skip_inventory.py": _PIN_PROXY_WITH_A_SKIP_AFTER_THE_PIN,
            "test_zzz_subject.py": _SUBJECT_CLEAN,
        },
        (),
        (),
        False,
        ("completeness claim FAILED", "a planted reason nobody disclosed"),
        (),
    ),
    (
        # the pin present and in its right place: it claims, and the guard has
        # to stay out of the way rather than claim a second time.
        "pin-present-and-last",
        {
            "test_skip_inventory.py": _PIN_PROXY_WITH_THE_PIN,
            "test_zzz_subject.py": _SUBJECT_CLEAN,
        },
        (),
        (),
        True,
        (),
        (_BANNER,),
    ),
)


@pytest.mark.parametrize(
    "files,argv,plugins,expect_zero,says,does_not_say",
    [case[1:] for case in _SESSIONS],
    ids=[case[0] for case in _SESSIONS],
)
def test_the_session_end_guard_answers_every_shortfall(
    tmp_path, files, argv, plugins, expect_zero, says, does_not_say
):
    """One session per shortfall, driving the REAL decision function.

    The exit code and the terminal summary, because those are what a developer
    gets. A verdict that reaches neither is the defect this whole file exists
    to close: a session that dropped the completeness check used to print a
    summary line byte-identical to a clean whole run's.
    """
    proc = _run_a_miniature_session(tmp_path, files, argv, plugins)
    output = proc.stdout + proc.stderr

    assert (proc.returncode == 0) is expect_zero, (
        f"pytest exited {proc.returncode} where exit-zero was expected to be "
        f"{expect_zero}. A session that hid a skip has to be non-zero, and a "
        f"legitimate narrowing has to stay zero — the second half is why "
        f"`pytest -k …` and `pytest tests/test_x.py` have to keep "
        f"working.\n\n{_tail(proc)}"
    )
    for sentence in says:
        assert sentence in output, (
            f"the reader was not told {sentence!r}. The verdict has to reach "
            f"the terminal summary; an exit code alone does not say WHICH "
            f"shortfall this session had, and five of them share an exit "
            f"code.\n\n{_tail(proc)}"
        )
    for sentence in does_not_say:
        assert sentence not in output, (
            f"the reader was told {sentence!r}, which is not what this "
            f"session did.\n\n{_tail(proc)}"
        )


# --- the channel that is not the exit code -----------------------------------
#
# EVERY ANCHOR HAS A LATER ONE. That is not a concession, it is the shape of
# the thing: `session.exitstatus` is a LAST-WRITER-WINS channel, so any
# mechanism carrying its verdict there is beatable by whatever writes last —
# and `pytest_cmdline_main` as a wrapper returns over `wrap_session` entirely,
# beyond which lie `config._main`, `console_main`, `atexit` and `os._exit` —
# the first of those is driven below, the last two are read off pytest's own
# source and are not measured anywhere in this repository.
# Moving the anchor from `pytest_sessionfinish` to `pytest_unconfigure` buys
# the four routes that exist; it cannot buy the last word, and the previous
# commit's "there is no hook after the last hook" was that mistake in one
# sentence.
#
# So the two cases below are DEMONSTRATIONS RATHER THAN DEFECTS: they assert
# that the attacker gets the exit code it asked for, and that the verdict
# arrives anyway, by a channel the attacker did not write to.

_BEATS_THE_EXIT_CODE = (
    (
        # after the anchor, in the anchor's own hook
        "a-trylast-unconfigure-assigning-zero",
        ("beats_by_unconfigure", _ASSIGNS_EXIT_ZERO_FROM_A_TRYLAST_UNCONFIGURE),
        0,
    ),
    (
        # outside every hook `wrap_session` calls, by returning over it
        "a-cmdline-main-wrapper-returning-98",
        ("beats_by_cmdline_main", _RETURNS_98_FROM_A_CMDLINE_MAIN_WRAPPER),
        98,
    ),
)


@pytest.mark.parametrize(
    "plugin,exit_code_the_attacker_chose",
    [case[1:] for case in _BEATS_THE_EXIT_CODE],
    ids=[case[0] for case in _BEATS_THE_EXIT_CODE],
)
def test_the_verdict_leaves_by_a_channel_the_exit_code_cannot_be_taken_from(
    tmp_path, plugin, exit_code_the_attacker_chose
):
    """The exit code goes; the verdict does not.

    Both halves are asserted, and the first half is the point. If the attacker
    did NOT get the exit code it asked for, this case would be measuring a
    mechanism that happens to win rather than a channel that does not depend on
    winning — so the exit code is pinned to the attacker's number, and the file
    is what has to carry the verdict out.
    """
    verdict_file = tmp_path / "verdict.txt"
    proc = _run_a_miniature_session(
        tmp_path,
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        plugins=(plugin,),
        env_extra={"STELLING_SKIP_INVENTORY_VERDICT": str(verdict_file)},
    )
    assert proc.returncode == exit_code_the_attacker_chose, (
        f"the attacker did not get the exit code it asked for "
        f"({exit_code_the_attacker_chose}), so this case is no longer "
        f"demonstrating that the exit code is takeable and the assertion below "
        f"proves nothing.\n\n{_tail(proc)}"
    )
    assert verdict_file.exists(), (
        "no verdict file: the one channel that does not go through "
        f"`session.exitstatus` did not get written.\n\n{_tail(proc)}"
    )
    written = verdict_file.read_text()
    assert written.startswith("verdict=failed\n"), (
        "the verdict file did not carry the shortfall. It is the whole answer "
        "to a last-writer-wins exit code, and a channel that agrees with the "
        f"channel it exists to replace is not one.\n\n{written}\n\n{_tail(proc)}"
    )
    assert "a planted reason nobody disclosed" in written, (
        "the verdict file says FAILED and does not say WHY, which is the same "
        f"defect as an exit code with no banner.\n\n{written}"
    )


# --- the defeat that raises nothing, and why it is not a `_SESSIONS` row -----
#
# `def pytest_sessionfinish(session, exitstatus): session.exitstatus = 0` is
# the whole attack. At 1b1c843 it was EXIT 0 with the banner still on the
# screen and NO `Exit:` line anywhere; add `_NOTES.clear()` and it was EXIT 0
# with zero banners, whose `diff` against the same session unplugged was
# exactly the banner lines and nothing else. That falsifies the consolation
# both the docstring and the commit message carried — "it does not produce the
# byte-identical green this mechanism exists to end, because pytest itself
# writes `Exit: <reason>`". It does when the plugin RAISES. This one does not.
#
# NOT a `_SESSIONS` row, and the reason is the defect this pass was sent to fix
# one file up. A row can assert the exit code, the banner and the absence of
# `Exit:` — and ALL THREE are satisfied by the same session with the plugin
# REMOVED, so the row would carry an id promising a discrimination its
# assertions never make. What distinguishes the two sessions is the exit code
# AS IT STOOD when the anchor read it, and that is only visible in the verdict
# file, which records it before the anchor repairs it.

_RAISES_NOTHING = (
    (
        "assigns-zero-and-leaves-the-notes",
        ("assigns_exit_zero", _ASSIGNS_EXIT_ZERO_WITHOUT_RAISING),
        # the note list survives, so the ordinary route delivers the banner and
        # the out-of-band writer must NOT fire
        False,
    ),
    (
        "assigns-zero-and-clears-the-notes",
        ("assigns_and_clears", _ASSIGNS_EXIT_ZERO_AND_CLEARS_THE_NOTES),
        # `_NOTES` is emptied before anything prints it, so the only thing left
        # that can reach the reader is the anchor writing the banner itself
        True,
    ),
)


@pytest.mark.parametrize(
    "plugin,out_of_band",
    [case[1:] for case in _RAISES_NOTHING],
    ids=[case[0] for case in _RAISES_NOTHING],
)
def test_a_sessionfinish_that_raises_nothing_still_takes_the_exit_code(
    tmp_path, plugin, out_of_band
):
    """Both halves, and the first one is what makes the second mean anything.

    ``exitstatus=`` in the verdict file is the exit code as the anchor found
    it, recorded before the anchor repairs it. Pinning it to 0 is pinning that
    the attack LANDED; without that, this case passes on a session that never
    had a plugin at all and asserts nothing.
    """
    verdict_file = tmp_path / "verdict.txt"
    proc = _run_a_miniature_session(
        tmp_path,
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        plugins=(plugin,),
        env_extra={"STELLING_SKIP_INVENTORY_VERDICT": str(verdict_file)},
    )
    output = proc.stdout + proc.stderr
    written = verdict_file.read_text()

    assert "\nexitstatus=0\n" in written, (
        "the exit code was NOT zero when the anchor read it, so this plugin "
        "did not take it away and the rest of this case is measuring a session "
        f"with nothing wrong with it.\n\n{written}\n\n{_tail(proc)}"
    )
    assert "Exit:" not in output, (
        "pytest wrote an `Exit:` line, so this is the RAISING defeat and not "
        "the one-line one — and the consolation that used to stand in "
        "`tests/conftest.py` (`the defeat always leaves Exit: behind`) would be "
        f"true after all.\n\n{_tail(proc)}"
    )
    assert proc.returncode != 0, (
        "the exit code was taken and not given back: this is the byte-identical "
        f"green the whole mechanism exists to end.\n\n{_tail(proc)}"
    )
    assert "completeness claim FAILED" in output, _tail(proc)
    assert "a planted reason nobody disclosed" in output, _tail(proc)
    marker = "written from pytest_unconfigure"
    assert (marker in output) is out_of_band, (
        f"the banner came {'through' if out_of_band else 'around'} the ordinary "
        f"route (pytest_terminal_summary, called from inside "
        f"pytest_sessionfinish) when it should have come the other way. "
        f"`_NOTES` is a request and `_DELIVERED` is the receipt; this is what "
        f"tells them apart.\n\n{_tail(proc)}"
    )


def test_the_verdict_file_is_not_adversary_proof_and_this_is_the_limit(tmp_path):
    """The stated limit, driven rather than asserted in prose.

    The file defends against the class this mechanism actually meets: a plugin
    that re-assigns an exit code, clears a note list, or exits a session early
    while knowing nothing about this pin. It does NOT defend against a plugin
    that has read ``tests/conftest.py``, because that plugin can read the same
    environment variable. A limit written down and never driven is a limit
    nobody knows the size of, so here is its size: the file is gone.
    """
    verdict_file = tmp_path / "verdict.txt"
    proc = _run_a_miniature_session(
        tmp_path,
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        plugins=(("deletes_the_verdict", _DELETES_THE_VERDICT_FILE),),
        env_extra={"STELLING_SKIP_INVENTORY_VERDICT": str(verdict_file)},
    )
    assert not verdict_file.exists(), (
        "the verdict file survived a plugin written to delete it, which means "
        "the limit stated in `tests/conftest.py` understates what the channel "
        f"does — and an understated limit is the more dangerous kind.\n\n"
        f"{_tail(proc)}"
    )
    # …and ABSENCE is itself the signal, which is the reason this channel was
    # picked over the two alternatives. The exit code and the banner are both
    # still there, because deleting the file is not the same attack.
    assert proc.returncode != 0, _tail(proc)
    assert "completeness claim FAILED" in proc.stdout + proc.stderr, _tail(proc)


def test_the_anchor_arrives_where_pytest_sessionfinish_is_never_called(tmp_path):
    """The FIFTH zero-arriving abort, and it is outside the mechanism entirely.

    ``pytest.exit(reason, returncode=0)`` from inside ``pytest_sessionstart``.
    In ``wrap_session``, ``initstate = 2`` is assigned only AFTER that hook
    returns, so the ``finally``'s ``if initstate >= 2`` is False and
    ``pytest_sessionfinish`` is never called AT ALL — not called late, not
    called with the wrong exit status: not called. Measured at 1b1c843 with a
    marker plugin printing from both hooks::

        marker only               EXIT 1  SESSIONFINISH-RAN, UNCONFIGURE-RAN
        marker + exit0 at start   EXIT 0  UNCONFIGURE-RAN only

    **What is checked here is that the ANCHOR ARRIVED, and not that it said
    something**, because there is nothing for it to say: a session that leaves
    from ``pytest_sessionstart`` has collected nothing, so ``SKIPPED``, ``RAN``
    and ``DESELECTED`` are all empty and the honest verdict is a withdrawal.
    Nothing was hidden on this route. What was wrong was the sentence — the
    previous docstring scoped itself to "every way out of a session that got as
    far as ``pytest_sessionstart``", and a session that exits FROM that hook
    got that far and was not covered. Off by one.
    """
    verdict_file = tmp_path / "verdict.txt"
    proc = _run_a_miniature_session(
        tmp_path,
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
        plugins=(
            ("marks_which_hooks_ran", _MARKS_WHICH_HOOKS_RAN),
            ("exits_zero_from_sessionstart", _EXITS_ZERO_FROM_SESSIONSTART),
        ),
        env_extra={"STELLING_SKIP_INVENTORY_VERDICT": str(verdict_file)},
    )
    output = proc.stdout + proc.stderr
    assert "MARKER-SESSIONFINISH-RAN" not in output, (
        "pytest_sessionfinish RAN on a session that exited from "
        "pytest_sessionstart, so this case is no longer the route it is named "
        f"for and proves nothing about the anchor.\n\n{_tail(proc)}"
    )
    assert "MARKER-UNCONFIGURE-RAN" in output, _tail(proc)
    assert verdict_file.exists(), (
        "the anchor did not arrive on the one way out of a session that skips "
        f"pytest_sessionfinish entirely.\n\n{_tail(proc)}"
    )
    written = verdict_file.read_text()
    assert written.startswith("verdict=withdrawn\n"), (
        "a session that exited before collection has an EMPTY record — nothing "
        "ran, nothing skipped, nothing was deselected — so the only honest "
        "answer is a withdrawal. Anything else here would be the guard "
        f"inventing a shortfall out of an absence.\n\n{written}"
    )


def test_the_pin_makes_its_own_claim_when_it_is_ordered_last(tmp_path):
    """The collection hook's sort, at the surface: WHERE the verdict lands.

    Collection order is alphabetical by file, so without
    ``pytest_collection_modifyitems`` moving it, this module runs before
    ``test_zzz_subject.py`` and reads a session that has not happened yet.
    Nothing goes green either way — the session-end guard catches it — so the
    thing that distinguishes the sort from its absence is not the exit code
    but the NODEID the failure carries, and a mechanism whose only effect is
    invisible is a mechanism nothing is holding.
    """
    proc = _run_a_miniature_session(
        tmp_path,
        {
            "test_skip_inventory.py": _PIN_PROXY_WITH_THE_PIN,
            "test_zzz_subject.py": _SUBJECT_UNDISCLOSED,
        },
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, _tail(proc)
    assert "test_skip_inventory.py::test_no_session_skip_is_undisclosed" in output, (
        "the pin did not fail at its own nodeid, so it did not run last: the "
        f"collection hook is no longer ordering it.\n\n{_tail(proc)}"
    )
    assert "made at the END of this session instead" not in output, (
        "the pin DEFERRED in an ordinary session — it ran before the tests it "
        f"reads, which is what the collection hook's sort is for.\n\n{_tail(proc)}"
    )


# --- the OTHER direction: a skip this session can contradict ------------------
#
# `_skips_this_session_cannot_explain` returns two lists and only one of them
# had a surface case. `undisclosed` is easy to reach: plant a reason nobody
# wrote down. `contradicted` is the direction that catches a gate firing when
# the thing it gates on is right there — the drift the whole file was written
# for, running the other way — and it had no case at all, on the stated grounds
# that every way of building one goes vacuous in some CI lane.
#
# That reasoning rules out one kind of discriminator: one that READS the
# ambient environment. Pick a library and assert on its presence and you are
# asserting about the lane. But the harness above does not have to read the
# environment — it CONTROLS the child's `PYTHONPATH`, so it can SUPPLY the
# condition. A one-line `blackjax.py` makes `import blackjax` succeed in every
# lane there is; a one-line `z3.py` makes `find_spec("z3")` return a spec in
# every lane there is. Non-vacuous everywhere, no clock, and nothing installed.
#
# This is the second time this exact reasoning error has been made in this
# mechanism: an earlier note ruled out BEHAVIOURAL discrimination between two
# sessions and concluded that discrimination was impossible, when
# record-integrity discrimination needed no clock. Both times the fix was the
# same shape — supply the condition rather than read it.


def test_a_gate_that_fired_for_a_library_that_is_right_here_is_contradicted(tmp_path):
    """The import-gate arm of the contradiction check, at the surface.

    A skip carrying pytest's own ``importorskip`` wording for a DECLARED
    optional dependency, in a session where that dependency imports fine. The
    dependency is supplied rather than looked for, so this session is the same
    in the zero-dep lane, the jax lane and the blackjax acceptance job.

    "A job that installs a dependency and then silently skips the tests that
    need it" is the sentence at the top of this file. This is the session-end
    machinery being made to say it.
    """
    proc = _run_a_miniature_session(
        tmp_path,
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_GATE_ON_A_LIBRARY_THAT_IS_THERE,
        },
        modules=(("blackjax", _A_LIBRARY_THAT_IMPORTS),),
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "a gate skipped for a missing 'blackjax' in a session where blackjax "
        f"imports fine, and the session was green.\n\n{_tail(proc)}"
    )
    assert "skipped for a missing 'blackjax' that imports fine here" in output, (
        "the contradiction was not reported to the reader; an exit code does "
        f"not say which direction the drift ran in.\n\n{_tail(proc)}"
    )


def test_a_rule_whose_condition_is_false_here_contradicts_the_skip(tmp_path):
    """The rule arm of the same check, and the same trick.

    ``Rule(when="the z3 wheel is not installed", …,
    legitimate=lambda: not _wheel("z3"))`` is a disclosure with a direction, and
    the direction is worth nothing unless something drives the case where it is
    false. ``_wheel`` is ``_optional.available``, i.e. ``find_spec``, so a
    ``z3.py`` on the child's path is enough to make the condition false without
    installing anything and without asking what this lane has.
    """
    proc = _run_a_miniature_session(
        tmp_path,
        {
            "test_skip_inventory.py": _PIN_PROXY,
            "test_zzz_subject.py": _SUBJECT_SKIPS_FOR_A_WHEEL_THAT_IS_THERE,
        },
        modules=(("z3", _A_LIBRARY_THAT_IMPORTS),),
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "a test skipped as 'needs z3' in a session where z3 is importable, and "
        f"the session was green.\n\n{_tail(proc)}"
    )
    assert (
        "its disclosed condition (the z3 wheel is not installed) does not hold "
        "here"
    ) in output, (
        "the rule's own `when` is what makes a contradiction legible, and it "
        f"did not reach the reader.\n\n{_tail(proc)}"
    )


def test_a_measured_skip_is_excused_only_at_the_test_it_names(monkeypatch):
    """MEASURED is TEST-KEYED, and that is the whole of what it buys.

    The shape has no checkable direction, so it pays for itself by being
    unable to travel: the same reason at any other nodeid is undisclosed drift.
    Nothing exercised either half — no MEASURED skip fires in this environment,
    which is exactly how a shape goes quietly inert — so the recorder's own
    channel is used to put one there and take it away again.
    """
    entry = MEASURED[0]
    assert entry.reason not in {r for rule in RULES for r in rule.reasons}

    monkeypatch.setitem(SKIPPED, entry.nodeid, entry.reason)
    undisclosed, contradicted = _skips_this_session_cannot_explain()
    assert not [u for u in undisclosed if entry.nodeid in u], (
        f"the MEASURED entry for {entry.nodeid} did not excuse the very skip "
        f"it names, so the shape excuses nothing and the entry is a comment: "
        f"{undisclosed}"
    )
    assert not [c for c in contradicted if entry.nodeid in c]

    monkeypatch.delitem(SKIPPED, entry.nodeid)
    monkeypatch.setitem(SKIPPED, "tests/test_some_other_module.py::test_x", entry.reason)
    undisclosed, _ = _skips_this_session_cannot_explain()
    assert [u for u in undisclosed if "test_some_other_module" in u], (
        "a MEASURED reason was excused at a test the entry does not name, so "
        "the excuse travels and the shape has become a suite-wide permission "
        "with none of a RULES entry's disclosure"
    )
