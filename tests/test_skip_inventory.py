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
   would rot on the first rename.
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

**Scope, and it is asserted rather than assumed.** The completeness half
claims something about *the suite*, and it can only claim it about a session
that collected the suite. A session narrowed by an explicit path, by ``--lf``
or by ``-k`` prints a green line that reads exactly like a whole run's, so the
scope is recorded in ``tests/conftest.py``, where the collection is, and
checked in :func:`test_no_session_skip_is_undisclosed`, where the claim is:
a session that never collected the tree WITHDRAWS the claim (it skips, saying
what it did not see), and one that collected the tree and then deselected part
of it FAILS. Whatever the session did see is checked either way.

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

import dataclasses
import importlib
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Callable

import pytest

from conftest import DESELECTED, RAN, SEEN_FILES, SKIPPED, _reason

from stelling import _optional

REPO = pathlib.Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"


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
)


# --- the rules: a skip reason any test may carry, and when it is legitimate --


@dataclasses.dataclass(frozen=True)
class Rule:
    when: str  # the condition, in prose — this is the disclosure
    reasons: frozenset[str] = frozenset()
    pattern: str = ""
    # Returns True when a skip carrying this reason is legitimate RIGHT NOW.
    # None means the condition is not computable from outside the test; such a
    # rule discloses the skip but cannot check its direction, and must say why.
    legitimate: Callable[[], bool] | None = None

    def matches(self, reason: str) -> bool:
        """FULLMATCH, not search: a pattern describes the WHOLE reason.

        ``re.search`` let a pattern excuse any reason that merely CONTAINED
        what it disclosed — ``"totally other cause; XLA did not contract this
        form on this build"`` was covered by the XLA rule, and widening a
        pattern to ``r"."`` disclosed the entire suite. A rule that cannot
        check its own direction must at least be unable to reach beyond the
        reason it was written for; anchoring is how. Breadth is bounded
        further by ``test_no_rule_is_broader_than_the_reason_it_discloses``.
        """
        if reason in self.reasons:
            return True
        return bool(self.pattern) and re.fullmatch(self.pattern, reason) is not None


RULES = (
    Rule(
        when="neither solver wheel is installed",
        reasons=frozenset(
            {
                "needs an SMT solver",
                "needs a solver",
                "no SMT solver installed",
                "no SMT backend installed",
                "no solver installed",
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
        when="`git` is not on PATH",
        reasons=frozenset({"needs git"}),
        legitimate=lambda: shutil.which("git") is None,
    ),
    Rule(
        when="this tree is not a git checkout — an unpacked sdist, say",
        reasons=frozenset({"not a git checkout (an unpacked sdist, say)"}),
        legitimate=lambda: not (REPO / ".git").exists(),
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
        # The variable part is the parametrised FORM NAME and nothing else:
        # `f"{name}: XLA did not contract this form on this build"` at
        # tests/test_three_rows_acceptance.py, where the names are things like
        # "broadcast+slice+squeeze" and "nested jit". Spelling that prefix out
        # is the point — the permission to be unverifiable must not also be a
        # permission to match arbitrary text that happens to end this way.
        pattern=r"[\w+ .\-]+: XLA did not contract this form on this build",
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
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", nodeid],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1800,
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


# --- how much of the suite this session was allowed to look at ---------------
#
# The completeness half below claims that THE SUITE's skip set is disclosed.
# A session can only support that claim about tests it collected, and pytest's
# report does not distinguish `1986 passed, 2 skipped` from `1927 passed, 2
# skipped, 60 deselected` in any way the claim can read. So the scope is
# checked, and there are exactly two ways to fall short of it — which get
# different answers, because they are different mistakes:
#
#   * the session never collected the whole tree (`pytest tests/test_x.py`,
#     `--lf`, `--ignore`, a shard). Ordinary development, and the developer
#     never asked about the suite. The claim is WITHDRAWN — the test skips,
#     saying which files it never saw. Making this case fail would break
#     `pytest tests/test_skip_inventory.py`, which is worse than the hole it
#     would close.
#
#   * the session collected the whole tree and then DESELECTED part of it
#     (`-k`, `-m`, `--deselect`). This one is a full-suite invocation with a
#     filter: it looked at every file, ran this pin, and would otherwise print
#     a green indistinguishable from a whole run while the deselected tests
#     skipped unobserved. That FAILS.
#
# Deliberately not a minimum-collection floor. A floor is a constant that has
# to be raised as the suite grows and quietly stops biting as it shrinks, and
# it cannot tell the two cases above apart — the second one collects the whole
# tree, so any floor it clears is a floor `-k` clears too.


def _files_the_suite_has() -> set[str]:
    return {path.name for path in TESTS.glob("test_*.py")}


def _files_this_session_saw() -> set[str]:
    return set(SEEN_FILES)


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


def test_every_rule_states_the_condition_it_discloses():
    """A rule with no condition is a blanket permission slip."""
    silent = [rule for rule in RULES if not rule.when.strip()]
    assert not silent, (
        "these skip rules disclose a reason without saying under what "
        "condition it is legitimate:\n  "
        + "\n  ".join(sorted(r.pattern or ", ".join(sorted(r.reasons)) for r in silent))
    )
    unchecked = [rule for rule in RULES if rule.legitimate is None]
    assert len(unchecked) <= 1, (
        "more than one skip rule has given up on checking its own direction. "
        "Each of these discloses a skip without being able to contradict it, "
        "which is the weak form this file exists to avoid:\n  "
        + "\n  ".join(sorted(r.pattern or ", ".join(sorted(r.reasons)) for r in unchecked))
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


# Reasons no rule in this file has any business excusing. Half of them are
# real skip reasons from elsewhere in the suite (a rule that swallowed one
# would silently take over another shape's disclosure); the rest are the
# shapes a widened pattern produces.
_MUST_NOT_MATCH = (
    "a totally undisclosed reason nobody wrote down",
    "totally other cause; XLA did not contract this form on this build",
    "XLA did not contract this form on this build, and also something else",
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


def test_no_rule_is_broader_than_the_reason_it_discloses():
    """A permission to be unverifiable must not also be a permission to be
    unbounded.

    ``matches`` used ``re.search``, so a pattern excused anything that
    CONTAINED it: the one rule that cannot check its own direction covered
    ``"totally other cause; XLA did not contract this form on this build"``
    unmutated, and widening it to ``r"."`` disclosed the entire suite while
    the suite stayed green. Anchoring (``fullmatch``) fixes the first; this
    test is what fixes the second, because ``r".*"`` is still anchored.
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
    # And the shape of the bound itself: no rule may be satisfiable by a
    # reason it does not name, which for the unverifiable one means its
    # variable part is a form NAME and not arbitrary prose.
    unverifiable = [r for r in RULES if r.legitimate is None]
    for rule in unverifiable:
        assert rule.pattern, (
            "an unverifiable rule that disclosed only literal reasons would "
            "not need a pattern; one that has none cannot be bounded here"
        )
        assert not rule.matches("anything at all " * 5), rule.pattern

    # The other edge of the same bound. A pattern tight enough to exclude the
    # decoys must still cover every reason the site can actually emit, or a
    # runner whose XLA folds differently reads as drift — which is the exact
    # thing this rule exists to prevent. The form names come from the source,
    # so adding a form with a comma or a bracket in its name fails HERE, where
    # the pattern is, rather than on someone else's machine.
    forms = re.findall(
        r'^\s*\("([^"]+)",\s*lambda u, w',
        (TESTS / "test_three_rows_acceptance.py").read_text(),
        re.M,
    )
    assert len(forms) >= 9, (
        f"only {len(forms)} contraction forms found in "
        f"test_three_rows_acceptance.py — the shape this reads has changed "
        f"and the check below has gone vacuous"
    )
    uncovered = [
        name
        for name in forms
        if not any(
            r.matches(f"{name}: XLA did not contract this form on this build")
            for r in RULES
        )
    ]
    assert not uncovered, (
        "contraction form(s) whose XLA skip no rule excuses — on a build "
        "where XLA does not fold these, the suite would report inventory "
        "drift that is nothing of the kind:\n  " + "\n  ".join(uncovered)
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


def test_no_session_skip_is_undisclosed():
    """The completeness half: everything this session skipped is covered by a
    pin, a rule, a declared optional-dependency gate, or a MEASURED entry —
    and every gate that fired had a dependency that really is absent.

    Scoped, and it says so. The claim is about the SUITE's skip set, so the
    session has to have collected the suite; see the two failure modes above.
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

    undisclosed: list[str] = []
    contradicted: list[str] = []

    for nodeid, reason in sorted(SKIPPED.items()):
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

    assert not undisclosed, (
        "undisclosed skip(s) — this suite's skip set is pinned, by condition, "
        "in tests/test_skip_inventory.py:\n  "
        + "\n  ".join(undisclosed)
        + "\n\nFor each one: make it not skip, or disclose it here. A named "
        "test skipping on a named dependency is a PINNED entry (both "
        "directions get asserted). A reason any test may carry is a RULES "
        "entry, with the condition that makes it legitimate. A new "
        "`importorskip` gate needs its library in "
        "DECLARED_OPTIONAL_DEPENDENCIES, which is a decision about the "
        "project's dependencies and not a formality. A skip whose condition is "
        "a measurement the test makes on its own subject is a MEASURED entry, "
        "test-keyed and exact. A skip nothing asserts is how two acceptance "
        "tests ran in no CI job for the project's whole life."
    )
    assert not contradicted, (
        "skip(s) whose disclosed condition is FALSE here — the test skipped "
        "anyway:\n  "
        + "\n  ".join(contradicted)
        + "\n\nThis is the other direction of the same drift: a gate that "
        "fires when the thing it gates on is present tests nothing and says "
        "so in green."
    )

    # --- and only NOW, the scope. The checks above are worth running on
    # whatever the session saw; the CLAIM is not the same size as the checks,
    # and this is where the difference gets stated instead of assumed.
    deselected = list(DESELECTED)
    unseen = sorted(_files_the_suite_has() - _files_this_session_saw())
    if unseen:
        pytest.skip(
            "the completeness pin is WITHDRAWN, not passed: this session "
            f"never collected {len(unseen)} of the suite's test files "
            f"({', '.join(unseen[:4])}"
            + (", …" if len(unseen) > 4 else "")
            + "), so it cannot say what the suite's skip set is — only what "
            "these files' was. Everything it DID see is disclosed. Run the "
            "whole suite for the claim."
        )
    assert not deselected, (
        f"this session collected every test file and then DESELECTED "
        f"{len(deselected)} test(s), so the skips inside them went "
        f"unobserved — and the run would otherwise have printed a green "
        f"indistinguishable from a whole one. The completeness pin cannot be "
        f"made from a filtered session; that is exactly the invocation an "
        f"undisclosed skip survives.\n\nFirst few: "
        + ", ".join(deselected[:5])
        + (", …" if len(deselected) > 5 else "")
        + "\n\nDrop the -k/-m/--deselect filter to make the claim. If you are "
        "iterating on this file, run it by PATH "
        "(`pytest tests/test_skip_inventory.py`) — a session that never "
        "collected the tree withdraws the claim instead of failing."
    )
