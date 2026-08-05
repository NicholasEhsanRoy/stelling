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

Both directions, and the same file is correct in all five environments without
being told which one it is in.

Three shapes of disclosure, in descending strength:

1. :data:`PINNED` — a named test, a named condition, both directions asserted.
2. :data:`RULES` — a skip *reason* that any test may legitimately carry, with
   the condition that makes it legitimate. Reason-keyed rather than
   test-keyed because "needs z3" governs a dozen tests and enumerating them
   would rot on the first rename.
3. :data:`DECLARED_OPTIONAL_DEPENDENCIES` — the libraries stelling's tests may
   gate on with ``importorskip``. A gate on anything else is undisclosed and
   fails here.

Anything a session skips that none of the three covers is a failure, with the
two ways to fix it named in the message.

The session's outcomes come from ``tests/conftest.py``, which records them as
pytest reports them; see there for why this cannot be a static read of the
tree.
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

from conftest import RAN, SKIPPED, _reason

from stelling import _optional

REPO = pathlib.Path(__file__).resolve().parent.parent


def _importable(name: str) -> bool:
    """Exactly ``pytest.importorskip``'s predicate: it IMPORTS, and it treats
    only ``ImportError`` as absence. ``find_spec`` would disagree with it on a
    package that is installed but broken, and then this file would demand a
    test run that pytest had just skipped."""
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _wheel(name: str) -> bool:
    """z3 / cvc5 as a python wheel. Deliberately NOT counting the cvc5 binary
    fallback: this predicate is only ever used to call a skip WRONG, and some
    solver gates in this suite require the wheel while others accept the
    binary. The weaker predicate cannot produce a false accusation."""
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
        if reason in self.reasons:
            return True
        return bool(self.pattern) and re.search(self.pattern, reason) is not None


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
        pattern=r"XLA did not contract this form on this build$",
        legitimate=None,
    ),
)


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
    return "ran", ""


def _outcome(nodeid: str) -> tuple[str, str]:
    return _from_session(nodeid) or _from_a_run_of_its_own(nodeid)


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


def test_no_session_skip_is_undisclosed():
    """The completeness half: everything this session skipped is covered by a
    pin, a rule, or a declared optional-dependency gate — and every gate that
    fired had a dependency that really is absent."""
    # Anti-vacuity: this runs last (tests/conftest.py reorders it there), so
    # the recorder must at minimum have seen the other tests in this file.
    assert any(
        "test_skip_inventory.py" in nodeid for nodeid in RAN
    ), "the outcome recorder saw nothing — tests/conftest.py is not loaded"

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
        "project's dependencies and not a formality. A skip nothing asserts "
        "is how two acceptance tests ran in no CI job for the project's whole "
        "life."
    )
    assert not contradicted, (
        "skip(s) whose disclosed condition is FALSE here — the test skipped "
        "anyway:\n  "
        + "\n  ".join(contradicted)
        + "\n\nThis is the other direction of the same drift: a gate that "
        "fires when the thing it gates on is present tests nothing and says "
        "so in green."
    )
