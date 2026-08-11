# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE PROPERTY SUITE, CHECKED WHERE IT CANNOT RUN.

Everything else under ``tests/property/`` needs hypothesis, and hypothesis is a
dev-group dependency: the two shared jax venvs on this machine do not have it,
and neither does the zero-dep CI job. In those environments the property
modules gate at collection and contribute **nothing**, and the suite still
prints a green line.

That is the exact shape this suite exists to prevent, one level up: *the
property suite examined nothing* and *the property suite found nothing* look
identical. So this module imports nothing but the standard library and pytest,
runs everywhere, and asserts the things that remain checkable when the search
cannot run:

* every property in the tree is **registered with a positive control**, and
  every registered control names a property that **exists**. A property added
  without a control, or a control whose property was renamed away, fails here —
  in any environment, including the ones that cannot run either;
* every registered **mutation still applies exactly once** to the file it
  names. A mutant that stops matching is a control that silently always passes,
  and this catches it without running anything;
* **the one judgement ``tools/property_check.py`` makes** — whether a run
  demonstrated the defect — driven end to end over synthetic controls whose
  outcome is known, including a defect that crashes BEFORE the oracle is
  evaluated. That needs neither hypothesis nor jax, only a nested pytest, and
  until it was here the rule that separates a demonstration from a crash was
  guaranteed by prose in three docstrings and by nothing that runs;
* every ``xfail`` marker in the suite is **strict and narrowed by ``raises=``**,
  read out of the source. The CI step that greps the run's own log cannot see
  either of those weakened while the defect is still there — measured: with
  ``strict=True`` changed to ``strict=False``, and separately with ``raises=``
  deleted, the suite still reports ``27 passed, 1 skipped, 1 xfailed`` at exit
  0 and every log-reading guard stays green. A source read is the only
  instrument that fires on the push that weakens the marker rather than on the
  day the defect is fixed;
* every property module carries the **hypothesis gate with the registered skip
  reason**, so ``tests/test_skip_inventory.py`` can hold the skip to its
  condition rather than to a count;
* ``hypothesis`` is **pinned in the dev group** of ``pyproject.toml``, with an
  upper bound.

What it deliberately does NOT claim: that any property passed, that any
strategy drew anything, or that the controls fire. Those need the search, and
where the search cannot run this file says so by construction rather than by
being green.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import sys

import pytest

import positive_controls as pc

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

# Modules that hold properties. `_`-prefixed files are machinery, and this file
# and the floor tests are not properties.
_NOT_PROPERTIES = {
    pathlib.Path(__file__).name,
    "test_generator_floor.py",
}

GATE_REASON = "needs hypothesis"


def _property_modules():
    return sorted(
        p
        for p in HERE.glob("test_*.py")
        if p.name not in _NOT_PROPERTIES
    )


def _definitions_in(path: pathlib.Path) -> dict:
    """``nodeid -> the ast node that DEFINES it``, for the two shapes here.

    pytest collects two: a ``test_*`` function, and a ``TestCase`` bound off a
    ``RuleBasedStateMachine``. For the second the node returned is the state
    MACHINE CLASS where the binding names one in this file, because the
    one-line binding holds none of the property and a caller below reads the
    property's own source.
    """
    tree = ast.parse(path.read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                out[f"tests/property/{path.name}::{node.name}"] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                # `TestCvcTransport = CvcTransport.TestCase` — a unittest
                # TestCase pytest collects as `::runTest`.
                if not (isinstance(t, ast.Name) and t.id.startswith("Test")):
                    continue
                held = node.value
                owner = classes.get(held.value.id) if (
                    isinstance(held, ast.Attribute)
                    and isinstance(held.value, ast.Name)
                ) else None
                out[f"tests/property/{path.name}::{t.id}::runTest"] = owner or node
    return out


def _tests_defined_in(path: pathlib.Path):
    """``test_*`` functions and stateful ``TestCase`` bindings, parsed."""
    return list(_definitions_in(path))


def test_every_property_in_the_tree_has_a_registered_positive_control():
    """No property ships without a place it is known to fail."""
    defined = set()
    for path in _property_modules():
        defined.update(_tests_defined_in(path))
    # The floor tests and the census assertions inside a property are not
    # themselves properties, so they carry no control. Everything else must.
    exempt = {
        "tests/property/test_cvc5_protocol.py::"
        "test_the_state_machine_examined_the_protocol",
    }
    uncontrolled = sorted(defined - pc.property_nodeids() - exempt)
    assert not uncontrolled, (
        "these properties ship with no registered positive control, so a green "
        "run of them is indistinguishable from a strategy that generates "
        "nothing. Add an entry to tests/property/positive_controls.py naming a "
        "commit or a mutation at which the property FAILS, and demonstrate it "
        "with `python tools/property_check.py --control <name>`:\n  "
        + "\n  ".join(uncontrolled)
    )


def test_every_registered_control_names_a_property_that_exists():
    """A control whose property was renamed away is a control that never runs."""
    defined = set()
    for path in _property_modules():
        defined.update(_tests_defined_in(path))
    dangling = sorted(c.name for c in pc.CONTROLS if c.nodeid not in defined)
    assert not dangling, (
        "these positive controls name a property that no longer exists under "
        "that name:\n  "
        + "\n  ".join(
            f"{n}: {pc.by_name(n).nodeid}" for n in dangling
        )
    )


def test_every_registered_mutation_still_applies_exactly_once():
    """A mutant that stops matching is a control that silently always passes.

    Checked statically, against the working tree, so it fails on the commit
    that moves the line rather than on the next time somebody thinks to run the
    controls.
    """
    bad = []
    for c in pc.CONTROLS:
        if c.mutation is None:
            continue
        path = REPO / c.mutation.path
        if not path.exists():
            bad.append(f"{c.name}: {c.mutation.path} does not exist")
            continue
        n = path.read_text().count(c.mutation.old)
        if n != 1:
            bad.append(
                f"{c.name}: {c.mutation.path} contains {n} occurrences of the "
                f"target text, expected exactly 1"
            )
    assert not bad, (
        "these registered mutations no longer apply, so their controls cannot "
        "demonstrate anything:\n  " + "\n  ".join(bad)
    )


def _xfail_markers_in(path: pathlib.Path):
    """``(nodeid, keyword-name -> ast node)`` for every ``xfail`` decorator.

    Static, because that is the point: the run's own log cannot distinguish a
    strict marker from a weakened one while the defect the marker excuses is
    still present. Both report ``XFAIL``.
    """
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            target = call.func if call else dec
            name = target.attr if isinstance(target, ast.Attribute) else (
                target.id if isinstance(target, ast.Name) else None
            )
            if name != "xfail":
                continue
            kw = {k.arg: k.value for k in (call.keywords if call else [])}
            out.append((f"tests/property/{path.name}::{node.name}", kw))
    return out


def test_every_xfail_in_the_suite_is_strict_and_narrowed_by_raises():
    """An amnesty in this suite must be strict, narrow, and controlled.

    **This is the check the CI log-grep cannot be**, and the difference was
    measured rather than reasoned. The ``property`` job reads the suite's own
    ``-q -ra`` output and asserts one ``XFAIL`` and no ``XPASS``. Against a
    tree where ``strict=True`` had been changed to ``strict=False``, and again
    against one where ``raises=WrongVerifiedFromWrap`` had been deleted, that
    log is **byte-identical to a healthy one** — ``27 passed, 1 skipped,
    1 xfailed``, pytest exit 0, guard exit 0 — because the defect the marker
    excuses is still there, so the test still xfails either way. A log-reader
    can only notice on the day the defect is fixed, which is the one day it
    must not be the first to notice.

    Deleting ``raises=`` is not cosmetic. Measured on this tree: with
    ``raises=`` removed and ``_bool_status`` mutated to violate everything, the
    wrap leg reports **XFAIL — green — while its own generator floor was never
    met**. The floor's ``AssertionError`` is swallowed by the blanket amnesty.
    With ``raises=`` in place the same run is reported ``FAILED``. That is the
    silent-success shape this suite exists to prevent, reachable by deleting
    one keyword.

    So the rule is on the marker itself, and it applies to every xfail in the
    tree rather than to one named test: strict, narrowed to an exception type,
    carrying a reason, and on a test that is registered with a positive
    control. A second, casual ``@pytest.mark.xfail`` fails here.
    """
    bad = []
    for path in _property_modules():
        for nodeid, kw in _xfail_markers_in(path):
            if not (
                isinstance(kw.get("strict"), ast.Constant)
                and kw["strict"].value is True
            ):
                bad.append(
                    f"{nodeid}: xfail is not `strict=True`. A non-strict xfail "
                    f"passes silently the day the defect is fixed, and the CI "
                    f"log-grep cannot tell the two apart before then."
                )
            if "raises" not in kw:
                bad.append(
                    f"{nodeid}: xfail carries no `raises=`, so it is a blanket "
                    f"amnesty over the whole property — including its own "
                    f"generator floor, which then fails green."
                )
            if not kw.get("reason"):
                bad.append(
                    f"{nodeid}: xfail carries no `reason=`, so the run's output "
                    f"does not say what is not being checked."
                )
            if nodeid not in pc.property_nodeids():
                bad.append(
                    f"{nodeid}: xfail-marked and not registered in "
                    f"positive_controls.py, so nothing demonstrates that the "
                    f"property it excuses still finds anything."
                )
    assert not bad, "\n  ".join(["xfail markers in tests/property/:", *bad])


def test_every_property_module_carries_the_hypothesis_gate():
    """The gate, its exact reason, and its position — all three matter.

    The reason string is what ``tests/test_skip_inventory.py`` keys on, and the
    POSITION is what keeps an unguarded ``from hypothesis import ...`` from
    aborting collection in an environment that has no hypothesis. A collection
    error takes the whole run down, not just its own module.
    """
    bad = []
    for path in _property_modules():
        text = path.read_text()
        gate = re.search(
            r'pytest\.importorskip\(\s*"hypothesis"\s*,\s*reason="([^"]*)"', text
        )
        if gate is None:
            bad.append(f"{path.name}: no `pytest.importorskip(\"hypothesis\", reason=…)`")
            continue
        if gate.group(1) != GATE_REASON:
            bad.append(
                f"{path.name}: gate reason is {gate.group(1)!r}, and the skip "
                f"inventory registers {GATE_REASON!r}"
            )
        first_import = re.search(r"^(?:from|import)\s+hypothesis\b", text, re.M)
        if first_import and first_import.start() < gate.start():
            bad.append(
                f"{path.name}: imports hypothesis at line "
                f"{text[:first_import.start()].count(chr(10)) + 1}, BEFORE the "
                f"gate — this is a collection error where hypothesis is absent"
            )
    assert not bad, "\n  ".join(["property module gates:", *bad])


def test_hypothesis_is_pinned_in_the_dev_group_with_an_upper_bound():
    """The dependency is declared where it is used, and bounded.

    Read out of ``pyproject.toml`` rather than restated, because two places to
    type a version is one place for them to disagree — and ``tools/
    property_venv.sh`` reads the same line.
    """
    text = (REPO / "pyproject.toml").read_text()
    group = re.search(r"^dev\s*=\s*\[(.*?)\]", text, re.S | re.M)
    assert group, "pyproject.toml has no [dependency-groups] dev list"
    reqs = re.findall(r'"([^"]+)"', group.group(1))
    hyp = [r for r in reqs if r.startswith("hypothesis")]
    assert hyp, (
        "tests/property/ needs hypothesis and the dev group does not declare "
        "it; a contributor following CONTRIBUTING.md would get a suite that "
        "silently skips its property tests"
    )
    assert "<" in hyp[0], (
        f"hypothesis is declared as {hyp[0]!r} with no upper bound. A major "
        "bump may change strategy semantics and shrinker behaviour, and this "
        "suite's value rests on shrink quality over a recursive harness "
        "grammar."
    )
    assert "hypothesis" not in text.split("[dependency-groups]")[0], (
        "hypothesis appears above [dependency-groups] — it must not be a "
        "runtime dependency or an extra"
    )


def test_the_registry_still_covers_the_defect_classes_it_was_built_for():
    """A tripwire on deletion: the controls are the suite's own inventory.

    Not a count for its own sake. Each name below is a defect class this
    project has shipped or is shipping, and losing one silently is how a suite
    stops covering something without anybody deciding to.
    """
    have = {c.name for c in pc.CONTROLS}
    want = {
        "oracle-wrap",          # the open integer-literal wrap
        "oracle-masked",        # the residual, one-sided oracle
        "vacuous-refutation",   # the affine leg judging an empty region
        "conjunct",             # the size-0 conjunct narrowing to a subset
        "redundant-assume",     # a box-implied assume adding proving power
        "reorder",              # order-dependence among independent obligations
        "widen",                # non-monotonicity in the input box
        "cvc5-flat",            # the record-boundary disagreement
        "cvc5-stateful",
        "cvc5-exit-tell",       # a definite answer over a nonzero exit
        "cvc5-phantom-model",   # a harvested model the child never wrote
        "cross-series",         # jax 0.10.2 vs 0.11.0
    }
    assert want <= have, f"positive controls have been removed: {sorted(want - have)}"


@pytest.mark.parametrize("control", pc.CONTROLS, ids=lambda c: c.name)
def test_every_control_says_what_it_is_and_which_kind_of_evidence_it_is(control):
    """``kind`` is load-bearing: a mutant is weaker evidence than a commit.

    A defect somebody shipped is evidence that the class is real. A defect
    somebody invented is evidence only that the property can see something.
    Both are useful; conflating them is not.
    """
    assert control.kind in ("commit", "mutant"), control.kind
    assert len(control.why) > 60, f"{control.name}: `why` is too thin to act on"
    # `expect_message=""` is a substring of EVERY string, so a control carrying
    # it is checked on `returncode != 0` alone. Measured: with `""`, a planted
    # collection error in tests/property/test_cvc5_protocol.py — the module
    # would not even import — was reported "FIRED — the property failed where
    # it is supposed to", `1/1 controls fired`, exit 0. Two controls shipped
    # that way. A control that cannot tell a defect from a broken import is
    # worth nothing, and this is the cheapest place to say so.
    assert control.expect_message, (
        f"{control.name}: expect_message is empty, which is a substring of "
        f"every string — including a traceback, a collection error and a "
        f"typo'd nodeid. tools/property_check.py would report any of those as "
        f"FIRED. Name a substring the property's own failure message carries; "
        f"a leg tag such as '[flat]' covers every failure shape of one leg."
    )
    if control.kind == "mutant":
        assert control.mutation is not None, (
            f"{control.name} is registered as a mutant with no mutation"
        )
        assert "mutant" in control.why.lower() or "MUTANT" in control.why, (
            f"{control.name}: a mutant control must say so in `why`, so a "
            f"reader does not take it for a defect this tree shipped"
        )
    else:
        assert control.mutation is None, (
            f"{control.name} is registered as a commit control but carries a "
            f"mutation, which makes the revision it names misleading"
        )


# A guard that is exactly one bracketed word: the LEG TAG `_judge` stamps as
# `[{where}]`. Deliberately a `fullmatch` on `\w+` rather than "starts with `[`
# and ends with `]`" — the loose form exempted a guard such as
# `[flat] … [stateful]` from every check below, and a leg tag is one word.
_LEG_TAG = re.compile(r"\[(\w+)\]")


def _property_stamps(nodeid: str, tag: str) -> bool | None:
    """Does the property ``nodeid`` names write ``tag`` as a string literal?

    ``[flat]`` is legitimate for a control on the flat leg because that leg's
    own body says ``where="flat"``, and illegitimate for one on any other leg
    for the same reason. The whole module is the wrong scope for that question:
    ``stateful`` occurs in ``test_cvc5_protocol.py`` too, which is exactly how
    ``[stateful]`` on the flat property passed unnoticed.

    ``None`` when the definition cannot be found at all, which is its own
    report rather than a pass.
    """
    path = REPO / nodeid.split("::")[0]
    node = _definitions_in(path).get(nodeid)
    if node is None:
        return None
    return any(
        isinstance(n, ast.Constant) and n.value == tag for n in ast.walk(node)
    )


def test_two_controls_on_one_property_do_not_share_a_guard():
    """Clause-specificity, which is the only thing that makes them two controls.

    Several controls may name the SAME property — that is how a conjunctive
    oracle gets a control per conjunct. What makes them separate evidence is
    that each ``expect_message`` picks out its own conjunct's failure. If one
    control's guard contains another's, the narrower control fires on the
    broader one's failure and demonstrates nothing of its own.

    Nothing that EXECUTES protected this before: broadening
    ``cvc5-exit-tell``'s clause-(1) guard back to the leg tag ``[flat]`` — the
    guard ``cvc5-flat`` already carries — left every gate in the repository
    green, and ``tools/property_check.py`` still said ``1/1 controls fired``.

    AND NOTHING THAT EXECUTES PROTECTS IT AGAINST ``[flat]`` NOW EITHER — the
    ``[stateful]`` spelling below is the one a run does catch. Adding
    ``cvc5-exit-tell`` and ``cvc5-phantom-model`` to the per-push ``--control``
    list was written up
    as closing the other half — "it now fails two assertions in
    ``test_suite_disclosure.py`` AND breaks a job that runs the control".
    Re-measured with the guard broadened to ``[flat]`` and the two controls in
    the step::

        == 9/9 controls fired        exit 0

    The RUN of the control does not notice, and cannot: ``[flat]`` is stamped
    on all three of ``_judge``'s messages, so the clause-(2) failure
    ``cvc5-flat`` already finds satisfies it. What goes red is this file — two
    assertions, both static: THIS one, on the substring containment, and
    ``test_a_clause_specific_guard_is_written_once_in_its_own_property``, on
    the rule that at most one control on a property may carry a leg tag at all.
    Both already ran in that job before the controls were added to it. The job
    is red either way, so the sentence is literally satisfiable; the causal
    reading it invites, that executing the control is what catches the
    broadening, is false.

    THE SECOND OF THOSE TWO WENT MISSING FOR ONE COMMIT and this paragraph went
    on claiming it. As shipped at ``bc5c0e4`` the clause-specific test exempted
    every bracketed guard with a ``continue``, so the count above was wrong in
    the direction that flatters it. Measured, static failures in this file, one
    broadening at a time::

                                        at bc5c0e4   now
        cvc5-exit-tell -> [flat]             1         2
        cvc5-exit-tell -> [stateful]         0         1

    ``[stateful]`` — a bracketed tag the flat property never stamps, so THIS
    test cannot see it either, neither string containing the other — was caught
    by nothing static at all, and only by the control run reporting "wrong
    failure". Both rows are red in this file again, the second of them from the
    clause-specific test alone.
    """
    by_property: dict[str, list] = {}
    for c in pc.CONTROLS:
        by_property.setdefault(c.nodeid, []).append(c)
    shared = []
    for nodeid, controls in sorted(by_property.items()):
        for a in controls:
            for b in controls:
                if a.name < b.name and (
                    a.expect_message in b.expect_message
                    or b.expect_message in a.expect_message
                ):
                    shared.append(
                        f"{a.name} ({a.expect_message!r}) and {b.name} "
                        f"({b.expect_message!r}) both name {nodeid}"
                    )
    assert not shared, (
        "these controls name the same property with guards one of which "
        "contains the other, so the narrower one fires on the broader one's "
        "failure and demonstrates nothing of its own:\n  " + "\n  ".join(shared)
    )


def test_a_clause_specific_guard_is_written_once_in_its_own_property():
    """One count, two failure modes, and both have happened on this branch.

    A guard registered for one clause of a conjunctive oracle should be the
    sentence that clause's own ``return`` builds — so it occurs in the module
    that holds the property exactly once.

    DERIVED, NOT LISTED. This named ``cvc5-exit-tell`` and
    ``cvc5-phantom-model`` in a literal tuple, so a third clause control would
    have shipped unchecked. The subjects are now every control that shares its
    property with another control — which is what makes a guard clause-specific
    in the first place. ``checked`` and ``tagged`` are both asserted non-empty,
    because a registry edit that left no shared property, or none carrying a
    leg tag, would otherwise make one of the two branches below vacuous.

    * **Zero** means the guard is not that sentence.
    * **Two or more** means the sentence is ALSO written somewhere pytest
      echoes. A long traceback prints the entire source of every function on
      it, docstring included, so a clause sentence quoted in ``_judge``'s
      docstring lands in the captured output of any crash inside ``_judge``.
      Measured, with the two sentences quoted there as they were: one
      line-neutral defect in ``solvers.py`` that raises ``TypeError`` before
      the oracle is evaluated scored ``3/3 controls fired`` against three probe
      controls carrying the three shipped guard strings.
      ``tools/property_check.py`` now matches the failure pytest RECORDS rather
      than everything it echoes; this is the other half, and it fires on the
      push that re-adds the quote rather than on the day somebody plants a
      crash.

    A BRACKETED LEG TAG SUCH AS ``[flat]`` IS EXEMPT FROM THE COUNT AND NOT
    FROM THIS TEST, and that difference is what this paragraph exists for. The
    tag is built as ``[{where}]``, so it correctly occurs zero times as a
    literal and the count is simply the wrong instrument for it. Exempting it
    with a ``continue`` — which is what shipped — left the tag's SHAPE
    unconstrained instead: measured on this file's own suite, repointing
    ``cvc5-exit-tell``'s guard at ``[stateful]``, a bracketed tag the flat
    property never stamps, gave **zero** static failures here, and the only
    thing that noticed was the control RUN coming back "wrong failure". So a
    leg tag gets two rules of its own rather than a pass:

    * it must be a tag THIS property stamps — the tag has to appear as a string
      literal inside the definition the nodeid names, which is where
      ``where="flat"`` is written. ``[stateful]`` on the flat property fails
      here;
    * at most ONE control on a property may carry one. A leg tag matches every
      failure of that leg, so it cannot tell two clauses apart, and a second
      one is a broadening however it is spelt. ``[flat]`` on ``cvc5-exit-tell``
      fails here — alongside
      ``test_two_controls_on_one_property_do_not_share_a_guard``, which catches
      that spelling by substring.

    MEASURED, static failures in this file, with the broadening applied to
    ``cvc5-exit-tell``'s ``expect_message`` and nothing else changed::

                          before   after
        -> ``[flat]``       1        2
        -> ``[stateful]``   0        1

    Unmutated it is green with ``cvc5-flat``'s own ``[flat]`` going THROUGH
    both rules rather than round them, and that is driven rather than assumed:
    change the flat leg's ``where="flat"`` to anything else and this test fails
    on ``cvc5-flat``.
    """
    by_property: dict[str, list] = {}
    for c in pc.CONTROLS:
        by_property.setdefault(c.nodeid, []).append(c)
    bad, checked, tagged = [], [], []
    for nodeid, controls in sorted(by_property.items()):
        if len(controls) < 2:
            continue
        module = nodeid.split("::")[0]
        src = (REPO / module).read_text()
        legs = [c for c in controls if _LEG_TAG.fullmatch(c.expect_message)]
        if len(legs) > 1:
            bad.append(
                f"{nodeid}: {', '.join(sorted(c.name for c in legs))} all carry "
                f"a bracketed LEG TAG, which matches every failure of that leg "
                f"and so cannot pick out one clause. At most one control on a "
                f"property may carry one; the rest must name their own clause's "
                f"sentence"
            )
        for c in controls:
            guard = c.expect_message
            tag = _LEG_TAG.fullmatch(guard)
            if tag:
                # Built as `[{where}]`, so counting it as a literal is the
                # wrong instrument. What IS checkable is that this property is
                # the leg that stamps it.
                tagged.append(c.name)
                stamps = _property_stamps(nodeid, tag.group(1))
                if stamps is None:
                    bad.append(
                        f"{c.name}: {nodeid} has no definition this file can "
                        f"find, so its leg tag {guard!r} cannot be checked"
                    )
                elif not stamps:
                    bad.append(
                        f"{c.name}: {guard!r} is a leg tag {nodeid} never "
                        f"stamps — {tag.group(1)!r} is not written as a string "
                        f"literal anywhere in that property's own definition, "
                        f"so no failure of it can carry the guard"
                    )
                continue
            checked.append(c.name)
            n = src.count(guard)
            if n != 1:
                bad.append(f"{c.name}: {guard!r} occurs {n} times in {module}")
    assert checked, (
        "no control shares its property with another one, so this test just "
        "checked nothing. Either the registry lost its clause-specific "
        "controls or this test stopped being able to find them."
    )
    assert tagged, (
        "no control on a shared property carries a bracketed leg tag, so the "
        "leg-tag branch above checked nothing. That branch is the one a "
        "broadened guard escapes through; if the registry legitimately has no "
        "leg tag left on a shared property, say so here rather than leaving "
        "the branch silent."
    )
    assert not bad, (
        "a guard on a property that has more than one control has to pick out "
        "ONE clause's failure. A clause SENTENCE does that by occurring "
        "EXACTLY ONCE in the module that holds the property, in the `return` "
        "that builds it. A bracketed LEG TAG does not do it at all — it is "
        "stamped on every message of that leg — so at most one control per "
        "property may carry one, and it must be a tag that property "
        "stamps:\n  " + "\n  ".join(bad)
    )


# ── the decision procedure in tools/property_check.py ────────────────────────
#
# THE MECHANISM'S OWN ANTI-VACUITY GAP. Everything above reads the registry.
# The registry is inert: what turns it into evidence is one judgement in
# `tools/property_check.py` — "the run came back red AND the failure pytest
# RECORDED carried `expect_message`" — and until these tests landed, nothing in
# the repository imported that file or asserted anything about how it decides.
# It is referenced by prose in two dozen places and executed by one CI step,
# and the CI step only ever runs it against controls that pass.
#
# The gap is measurable, not theoretical: reverting the match to the run's
# ECHOED OUTPUT — one line — makes a defect that raises before the oracle is
# evaluated at all score FIRED, for controls that are in the per-push gate,
# with every gate in the repository still green.
#
# So the probes below drive the real `main()` over synthetic controls whose
# outcome is known, in a nested pytest. No hypothesis, no jax, no solver: the
# probe property is fifteen lines of plain Python, and the tree it is pointed
# at is an empty `src/`.

_PROBE_GUARD = "PROBE SAW A NONZERO-EXIT RUN"

# Shaped after `_judge` in test_cvc5_protocol.py: a conjunction that builds its
# message templates ABOVE the line that can crash, so a crash puts the guard
# into the echoed traceback (pytest prints a frame's whole source, from `def`
# down to the failing line) without any oracle having run.
_PROBE_ORACLE = '''\
def _oracle(res):
    """Three clauses, evaluated in order, returning on the first that fails."""
    if res["truncated"]:
        return "PROBE SAW A TRUNCATED RUN"
    if res["rc"]:
        return "PROBE SAW A NONZERO-EXIT RUN"
    if sorted(res["values"]) != res["present"]:
        return "PROBE SAW A MODEL THAT WAS NOT WRITTEN"
    return None


def test_the_probe_property():
    assert _oracle(%s) is None, _oracle(%s)
'''

# `case -> (the argument, the exit code main() must return, the outcome it must
# report)`. Read as a truth table: the first row is the one the revert flips,
# and the other two are what stop this test passing vacuously — a runner that
# had stopped running anything would report NOT DEMONSTRATED for all three.
_PROBE_CASES = {
    # `values=None` makes `sorted(...)` raise TypeError at the THIRD clause, so
    # the oracle never returns and nothing evaluated the property. The guard is
    # in the echoed traceback and in no recorded failure.
    "a crash before the oracle": (
        '{"truncated": False, "rc": 0, "values": None, "present": []}',
        1, "echoed, not raised",
    ),
    # the guard's own clause, genuinely reached and genuinely raised
    "the guard's own clause": (
        '{"truncated": False, "rc": 137, "values": [], "present": []}',
        0, None,
    ),
    # the property passes: the control demonstrates nothing
    "a green property": (
        '{"truncated": False, "rc": 0, "values": [], "present": []}',
        1, "passed where it must fail",
    ),
}


def _property_check():
    """``tools/property_check.py``, imported by path rather than installed."""
    path = REPO / "tools" / "property_check.py"
    assert path.is_file(), f"{path} is missing; the property job runs it"
    spec = importlib.util.spec_from_file_location("_property_check_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A NAIVE locator, written out here rather than imported. It is independent of
# how `tools/property_check.py` PRIMARILY reads the file (a real TOML parse)
# and NARROWER than the regex that file falls back to when no parser imports
# (which also reads quoted table keys and quoted entry-point names). It
# cross-checks the parser from below, and says nothing about the fallback —
# the fallback-vs-parser pin elsewhere in this file does that. Claiming
# more than that would be the thing it exists to catch.
_PYTEST11_SECTION = re.compile(
    r"^\[project\.entry-points\.pytest11\]\s*$(.*?)(?=^\[|\Z)", re.M | re.S
)


def _plant_entry_point(src: pathlib.Path, *pairs) -> None:
    """Declare ``pytest11`` entry points ON THE TREE the child will import.

    ``pairs`` are ``(entry point name, module target)``.

    A synthetic ``.dist-info`` beside the package, inside the ``src/`` that
    ``property_check.py`` puts on the child's ``PYTHONPATH``.
    ``importlib.metadata`` finds distributions by scanning ``sys.path`` for
    ``*.dist-info``, so this is a real entry point to the child in the sense
    that matters: pytest autoloads it at ``Config.parse``, before collection,
    with no exception handling around the import.

    NO INSTALL, and that is the point of doing it this way. The event being
    reproduced only happens where stelling is installed as a distribution —
    CI installs with ``-e``, the two shared dev venvs do not — so a test that
    depended on the real entry point would be a test that runs in CI and
    skips on the machine the fix is written on. This one runs everywhere and
    reproduces the same pytest code path.
    """
    info = src / "probe_dist-0.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: probe-dist\nVersion: 0.0.0\n"
    )
    (info / "entry_points.txt").write_text(
        "[pytest11]\n" + "".join(f"{n} = {t}\n" for n, t in pairs)
    )


def _score_one_probe(module, monkeypatch, work, argument, plant=None,
                     preamble="", nodeid=None):
    """Run the real ``main()`` over one synthetic control; return (exit, out).

    ``plant``, if given, is called with the probe repository's ``src/`` before
    the tree is materialised — the hook the environment probes below use to
    put a ``pytest11`` entry point where the child will autoload it.
    ``preamble`` is prepended to the probe module (an import that fails makes
    it uncollectable), and ``nodeid`` overrides what the control points at (a
    nodeid matching nothing is the registry's own way of running no test).
    """
    src = work / "repo" / "src"
    src.mkdir(parents=True)
    if plant is not None:
        plant(src)
    probe = work / "test_probe.py"
    probe.write_text(preamble + _PROBE_ORACLE % (argument, argument))
    control = pc.Control(
        name="probe",
        nodeid=nodeid or str(probe),
        kind="mutant",
        at="HEAD",
        why=(
            "synthetic, built by test_suite_disclosure.py to drive "
            "tools/property_check.py's decision procedure over an outcome "
            "that is known in advance"
        ),
        expect_message=_PROBE_GUARD,
    )

    class _Registry:
        CONTROLS = (control,)

        @staticmethod
        def by_name(name):
            assert name == "probe", name
            return control

    monkeypatch.setattr(module, "pc", _Registry)
    return module.main([
        "--control", "probe",
        "--repo", str(work / "repo"),
        "--python", sys.executable,
    ])


def test_the_control_runner_scores_a_crash_before_the_oracle_as_not_demonstrated(
    tmp_path, monkeypatch, capsys
):
    """The judgement that turns the registry into evidence, executed.

    A positive control asserts that TODAY'S PROPERTY, run against a tree that
    carries the defect, comes back RED. "Red" is not enough on its own: a
    typo'd nodeid, a collection error, an import failure and a crash inside the
    property's own helpers are all red, and each of them demonstrates nothing.
    ``expect_message`` is what separates them, and where it is matched is what
    makes the separation real — against the failure pytest RECORDS in
    ``--junitxml``, not against everything the run echoes.

    **Measured, and this is why the test exists.** With the match reverted to
    the echoed output — one line — a defect that raises ``TypeError`` before
    the oracle is evaluated at all scores ``FIRED`` and the tool exits 0.
    Driven against the real suite with the line-neutral ``solvers.py`` defect
    that makes ``sorted(res.values)`` raise, scored under both rules from one
    run: the OLD rule fires **2 of the 3** shipped cvc5 guard strings —
    including clause (1)'s, which is the live guard of a control in the
    per-push gate — where the shipped rule scores 0 of 3. Nothing else in the
    repository notices: no other test imports ``tools/property_check.py``.

    The three rows below are a truth table, and all three matter. The first is
    the one the revert flips. The other two are the anti-vacuity half — a
    runner that had stopped running anything, or one wired to refuse
    everything, would report NOT DEMONSTRATED for all three and would pass a
    test that only asserted the first.
    """
    module = _property_check()
    bad = []
    for case, (argument, want_exit, want_report) in _PROBE_CASES.items():
        work = tmp_path / case.replace(" ", "_").replace("'", "")
        work.mkdir()
        got_exit = _score_one_probe(module, monkeypatch, work, argument)
        out = capsys.readouterr().out
        if got_exit != want_exit:
            bad.append(f"{case}: main() returned {got_exit}, expected {want_exit}")
        if want_report is None:
            if "FIRED — the property failed where it is supposed to" not in out:
                bad.append(f"{case}: expected FIRED, got:\n{out}")
        else:
            if f"NOT DEMONSTRATED: probe — {want_report}" not in out:
                bad.append(
                    f"{case}: expected NOT DEMONSTRATED …— {want_report}, "
                    f"got:\n{out}"
                )
    assert not bad, (
        "tools/property_check.py decided a control's outcome differently from "
        "the way its own docstrings say it does. The first row is the whole "
        "anti-vacuity mechanism: a red run whose guard is only in the ECHOED "
        "output — a traceback prints the property's own source, docstring "
        "included — must be NOT DEMONSTRATED, never FIRED:\n  "
        + "\n  ".join(bad)
    )


def test_the_control_runner_says_a_session_that_never_ran_never_ran(
    tmp_path, monkeypatch, capsys
):
    """A control that reports on nothing must not be reported as a failure.

    THE SHAPE, MEASURED AT 260527b, on the nine per-push controls. The commit
    added ``stelling._tripwire`` and a ``pytest11`` entry point naming it. The
    child pytest that runs a control takes its ``stelling`` from
    ``PYTHONPATH=<tree>/src`` — the REVISION's — and its entry points from the
    installed distribution metadata — the CHECKOUT's. For every control pinned
    to a commit older than that package the two disagree, pytest died at
    ``Config.parse`` with ``ModuleNotFoundError``, no test was collected, no
    junit XML was written, and the runner printed:

        FIRED, but the failure did not carry 'REFUTED OVER AN EMPTY ADMITTED
        REGION'

    which is false twice. It did not fire, and there was no failure to carry
    anything. A clean partition — all five controls whose tree is ``HEAD``
    FIRED (``_materialise`` copies the working tree, which has the package),
    all four pinned to a commit reported as above — read in the log as four
    broken properties. (Of the nine, ``at="HEAD"`` and ``kind="commit"``
    happen to name the same two groups; ``at`` is the discriminant, and
    ``oracle-wrap``, outside the nine, is ``kind="commit"`` at ``at="HEAD"``.)

    The fault itself is closed in ``_run``, and
    ``test_the_control_runner_blocks_this_projects_own_pytest_plugin`` holds
    that shut. This test is about the SENTENCE, which is the general half: any
    session that dies before reaching a test produces this, and the runner's
    own rule — the echoed output is not evidence — is exactly what says it
    must not be scored against ``expect_message`` at all.

    FOUR RUNS AND TWO UNIT ROWS. The first run FIRES, so the three faults are
    measured against a probe that is known to work; the three faults are three
    distinct routes into ``_reported``, because a remedy for one of them
    passes a test that drives one of them. The unit rows hold the rule's shape
    where no run can see it: what it decides WITHOUT the discriminator, and
    that the discriminator is consulted before the echoed-output match.
    """
    module = _property_check()
    # The row that FIRES: same oracle, same argument, same probe, so the only
    # thing that differs from row to row below is the fault planted with it.
    argument = _PROBE_CASES["the guard's own clause"][0]

    alive = tmp_path / "alive"
    alive.mkdir()
    got_alive = _score_one_probe(module, monkeypatch, alive, argument)
    out_alive = capsys.readouterr().out

    # THREE SPELLINGS OF "NOTHING RAN", and each reaches `_reported` by a
    # different route. Measured on pytest 9.1.1, exit code and XML in
    # `_reported`'s docstring: no XML at all, XML with no `<testcase>`, and
    # XML whose one `<testcase>` is a collection failure. A remedy that
    # covered only the first passes a test that drives only the first — which
    # is what this test did until the second and third rows were added.
    faults = {
        "a plugin that cannot import": dict(
            plant=lambda src: _plant_entry_point(
                src, ("probe_dead_plugin", "probe_dead_plugin_module")
            ),
            expect="No module named 'probe_dead_plugin_module'",
        ),
        "a nodeid that matches nothing": dict(
            nodeid="does_not_exist.py::test_nothing_is_called_this",
            expect="does_not_exist.py",
        ),
        "a test module that cannot import": dict(
            preamble="import probe_module_that_is_not_there\n",
            expect="No module named 'probe_module_that_is_not_there'",
        ),
        "a module that skips at import": dict(
            preamble="import pytest\npytest.importorskip('nonexistent_module_probe')\n",
            expect="nonexistent_module_probe",
        ),
    }

    bad = []
    if got_alive != 0 or "FIRED — the property failed" not in out_alive:
        bad.append(
            f"the unplanted run did not FIRE (exit {got_alive}), so this test "
            f"proves nothing about what the faults changed:\n{out_alive}"
        )
    for label, spec in faults.items():
        expect = spec.pop("expect")
        work = tmp_path / label.replace(" ", "_")
        work.mkdir()
        got = _score_one_probe(module, monkeypatch, work, argument, **spec)
        out = capsys.readouterr().out
        if got != 1:
            bad.append(f"{label}: returned {got}, expected 1")
        if f"NOT DEMONSTRATED: probe — {module.NEVER_RAN}" not in out:
            bad.append(
                f"{label}: not reported as {module.NEVER_RAN!r}:\n{out}"
            )
        # THE SENTENCE ITSELF, ASSERTED POSITIVELY. Constraining it only by
        # what it must NOT say leaves every other wrong sentence admissible:
        # measured, replacing the whole `NEVER_RAN` branch's `print` with the
        # FIRED line — this change's own condemned verdict, in the spelling it
        # exists to abolish — left the suite at 27 passed, while the real tool
        # printed "FIRED — the property failed where it is supposed to" for a
        # session that had reported on nothing.
        if "THE PROPERTY NEVER RAN" not in out:
            bad.append(
                f"{label}: the run was summarised as {module.NEVER_RAN!r} but "
                f"the line a reader actually reads does not say so:\n{out}"
            )
        if "FIRED" in out:
            bad.append(
                f"{label}: a session that reported on no test was described "
                f"as having FIRED:\n{out}"
            )
        if expect not in out:
            bad.append(
                f"{label}: the run did not fail the way this row needs it to "
                f"fail — {expect!r} is not in what it printed:\n{out}"
            )
        # The sentence this exists to stop. It is the ONE thing a reader acts on.
        if "did not carry" in out:
            bad.append(
                f"{label}: a session that ran no test was still described in "
                f"terms of a failure that did not carry the guard:\n{out}"
            )
    # And the branch is load-bearing: those are the inputs the old rule saw —
    # red, nothing recorded, the guard nowhere — and its answer was WRONG.
    if module._verdict(_PROBE_GUARD, [], "", fired=True, reported=True) != module.WRONG:
        bad.append(
            "the rule WITHOUT the `reported` discriminator no longer scores "
            "these inputs as a wrong failure, so this test has stopped "
            "measuring the difference the discriminator makes"
        )
    # WHERE the branch sits, which the docstring calls out as deliberate and
    # which no run above can see: a session that died echoing its own guard
    # must still be NEVER_RAN, not ECHOED.
    if module._verdict(_PROBE_GUARD, [], _PROBE_GUARD,
                       fired=True, reported=False) != module.NEVER_RAN:
        bad.append(
            "`NEVER_RAN` no longer comes ahead of the echoed-output match, so "
            "a session that died before running anything can be scored on a "
            "string in its own traceback"
        )
    # A `pytest_internalerror` session writes classname="pytest", which is not
    # a report on any test. Synthetic XML because triggering rc 3 in a child
    # requires a conftest that breaks pytest itself.
    internal_xml = tmp_path / "internal.xml"
    internal_xml.write_text(
        '<?xml version="1.0"?><testsuites><testsuite>'
        '<testcase classname="pytest" name="internal">'
        '<error message="internal error"/></testcase>'
        '</testsuite></testsuites>'
    )
    if module._reported(internal_xml):
        bad.append(
            "a session with only classname='pytest' (internal error) was "
            "treated as having reported on a test — `_reported` must key "
            "positively on shapes that look like real tests"
        )
    assert not bad, (
        "tools/property_check.py described a run that reported on no test at "
        "all as though it had reported on the property. Every outcome here is "
        "NOT DEMONSTRATED either way — nothing goes from red to green — so "
        "what is at stake is only what the log SAYS, which is the whole "
        "product of an anti-vacuity mechanism:\n  " + "\n  ".join(bad)
    )


def test_the_control_runner_blocks_this_projects_own_pytest_plugin(
    tmp_path, monkeypatch, capsys
):
    """The fault above, reproduced with this project's own entry point.

    ``property_check.py`` joins the checkout's TESTS to a revision's SOURCE
    with ``PYTHONPATH`` and nothing else, which is the separation the whole
    file exists for. The cost of it is that the environment's distribution
    metadata still describes the checkout, and a ``pytest11`` entry point is
    loaded from metadata and resolved against ``sys.path``. Those two are
    permanently allowed to disagree, for every revision older than any module
    the entry point names, so the runner blocks the plugin by name.

    TWO RUNS ON ONE TREE, differing only in whether the block is in place.
    Both plant a ``stelling`` with no ``_tripwire`` in it and an entry point
    naming ``stelling._tripwire.plugin`` — the shape a control pinned to any
    revision before 260527b has in an installed environment.

    THE NAME IS NOT WRITTEN DOWN TWICE, and this test is what says so. A
    membership assertion — "the blocked name is one of the declared ones" —
    permits the declaration to GROW, and the second entry point reopens the
    fault for every commit-pinned control at once with every gate still green.
    Measured: with a second name added to ``pyproject.toml`` and nothing else
    changed, a probe whose tree carries both goes from FIRED to NEVER RAN.
    So the block reads the table, and the last row here drives a two-entry
    ``pyproject.toml`` rather than asserting anything about today's one.
    """
    module = _property_check()
    declared = module._entry_points_to_block()

    # THE FLOOR, DRIVEN. "Fails onto the known name, never below it" is a
    # claim about a path no other row here takes, and asserting the floor name
    # is in today's answer asserts nothing: the declaration contains it, so
    # the assertion holds even with the fallback deleted. Point the runner at
    # a file that is not there instead.
    monkeypatch.setattr(module, "PYPROJECT", tmp_path / "not-a-file.toml")
    assert module._entry_points_to_block() == (module.TRIPWIRE_ENTRY_POINT,), (
        "with no pyproject.toml to read, the runner blocks "
        f"{module._entry_points_to_block()} — it is supposed to fall onto "
        f"{module.TRIPWIRE_ENTRY_POINT!r}, never below it, because a checkout "
        "the file did not ship to still runs controls against revisions that "
        "predate the plugin."
    )
    monkeypatch.undo()

    # A NAIVE READ OF THE FILE, held against what the runner decided. This is
    # independent of the runner's PRIMARY path — a real TOML parse — and
    # NARROWER than the runner's FALLBACK (which also reads quoted table keys
    # and quoted names), so it cross-checks both from below. What it catches
    # is a parser that reads the file and comes back with less than is plainly
    # written in it.
    section = _PYTEST11_SECTION.search(
        (REPO / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert section is not None, (
        "pyproject.toml has no [project.entry-points.pytest11] section, so "
        "tools/property_check.py is blocking a name nothing declares"
    )
    # THE FALLBACK, HELD AGAINST THE PARSER on the file that actually ships.
    # It is narrower by construction and says so; what it must not be is
    # narrower HERE, because then the one environment that has no TOML parser
    # blocks less than every other environment and nothing anywhere says so.
    # `sys.modules[name] = None` is what makes `import name` raise.
    raw = (REPO / "pyproject.toml").read_bytes()
    parsed = module._declared_pytest11(raw)
    with monkeypatch.context() as no_toml:
        no_toml.setitem(sys.modules, "tomllib", None)
        no_toml.setitem(sys.modules, "tomli", None)
        fallback = module._declared_pytest11(raw)
    assert fallback == parsed and parsed, (
        f"on this project's own pyproject.toml the TOML parse returns "
        f"{sorted(parsed)} and the no-parser fallback returns "
        f"{sorted(fallback)}. They have to agree on the spelling that ships, "
        f"or a 3.10 environment without `tomli` silently blocks less."
    )

    in_file = set(re.findall(r"^\s*([\w.-]+)\s*=", section.group(1), re.M))
    assert in_file, (
        "the naive regex found no entry-point names in the "
        "[project.entry-points.pytest11] section — if all names are quoted "
        "the cross-check is vacuous and this test guarantees nothing"
    )
    assert module.TRIPWIRE_ENTRY_POINT in in_file, (
        f"TRIPWIRE_ENTRY_POINT is {module.TRIPWIRE_ENTRY_POINT!r} but "
        f"pyproject.toml declares {sorted(in_file)} — the floor name must be "
        f"one of the shipped names or the fallback blocks a plugin nothing "
        f"declares"
    )
    assert in_file <= set(declared), (
        f"pyproject.toml declares {sorted(in_file)}; the runner blocks "
        f"{sorted(declared)}. `-p no:` matches the entry point NAME, so any "
        f"declared name the runner does not follow restores the crash it was "
        f"written to stop, on every commit-pinned control at once."
    )

    name = module.TRIPWIRE_ENTRY_POINT

    def plant(src):
        (src / "stelling").mkdir()
        (src / "stelling" / "__init__.py").write_text("")
        _plant_entry_point(src, (name, "stelling._tripwire.plugin"))

    argument = _PROBE_CASES["the guard's own clause"][0]

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    got_blocked = _score_one_probe(module, monkeypatch, blocked, argument, plant=plant)
    out_blocked = capsys.readouterr().out

    # The same tree with the block aimed elsewhere. This is the state the
    # `property` job was in at 260527b. BOTH sources have to be moved: the
    # runner reads the declaration and falls back to the floor name, so
    # changing one alone leaves the other still blocking.
    monkeypatch.setattr(module, "PYPROJECT", tmp_path / "no-such-pyproject.toml")
    monkeypatch.setattr(module, "TRIPWIRE_ENTRY_POINT", "nothing-is-named-this")
    unblocked = tmp_path / "unblocked"
    unblocked.mkdir()
    got_unblocked = _score_one_probe(
        module, monkeypatch, unblocked, argument, plant=plant
    )
    out_unblocked = capsys.readouterr().out

    bad = []
    if got_blocked != 0 or "FIRED — the property failed" not in out_blocked:
        bad.append(
            f"with the block in place the probe still did not run "
            f"(exit {got_blocked}):\n{out_blocked}"
        )
    if "No module named 'stelling._tripwire'" not in out_unblocked:
        bad.append(
            "with the block removed the session did NOT die on the planted "
            "entry point, so the run above demonstrates nothing about the "
            f"block:\n{out_unblocked}"
        )
    if got_unblocked != 1:
        bad.append(f"the unblocked run returned {got_unblocked}, expected 1")

    # THE SECOND ENTRY POINT — the commit that adds one, which a membership
    # assertion would have let through. The TREE is the same in both runs
    # below and carries both plugins; only the DECLARATION the runner reads
    # differs, so the second run is what the first would be if the block were
    # a name written down here instead of a table read from the file.
    monkeypatch.undo()
    later = ("stelling_second", "stelling._later_thing.plugin")

    def plant_two(src):
        (src / "stelling").mkdir()
        (src / "stelling" / "__init__.py").write_text("")
        _plant_entry_point(src, (name, "stelling._tripwire.plugin"), later)

    outs = {}
    for label, table in (
        ("both declared", [(name, "stelling._tripwire.plugin"), later]),
        ("only the first declared", [(name, "stelling._tripwire.plugin")]),
    ):
        home = tmp_path / label.replace(" ", "_")
        home.mkdir()
        pyproject = home / "pyproject.toml"
        pyproject.write_text(
            "[project]\nname = 'probe'\n\n[project.entry-points.pytest11]\n"
            + "".join(f'{n} = "{t}"\n' for n, t in table)
            + "\n[tool.after-the-table]\nignored = true\n"
        )
        monkeypatch.setattr(module, "PYPROJECT", pyproject)
        run = home / "run"
        run.mkdir()
        outs[label] = (
            _score_one_probe(module, monkeypatch, run, argument, plant=plant_two),
            capsys.readouterr().out,
        )

    got, out = outs["both declared"]
    if got != 0 or "FIRED — the property failed" not in out:
        bad.append(
            f"with two entry points declared and two planted, the probe did "
            f"not run (exit {got}). The runner is not blocking everything "
            f"`pyproject.toml` declares:\n{out}"
        )
    got, out = outs["only the first declared"]
    if got != 1 or "No module named 'stelling._later_thing'" not in out:
        bad.append(
            f"with the second entry point planted but NOT declared, the "
            f"session did not die on it (exit {got}), so the run above "
            f"demonstrates nothing about reading the table:\n{out}"
        )
    assert not bad, (
        "the `-p no:` block in tools/property_check.py is not what keeps this "
        "project's own pytest plugins out of the children it starts. Without "
        "it every control pinned to a revision older than a module an entry "
        "point names dies before collection, and nine controls report as "
        "four broken properties:\n  " + "\n  ".join(bad)
    )


def test_two_controls_on_one_property_do_not_point_at_the_same_tree():
    """The other half of what makes them two controls: a different defect.

    ``test_two_controls_on_one_property_do_not_share_a_guard`` holds the
    GUARDS apart. This holds the TREES apart, and both are needed, because a
    conjunctive oracle gets a control per conjunct only if each control's tree
    is a place where that conjunct — and, ideally, only that conjunct — fails.
    Two controls on one property pointing at one tree are one control reported
    twice.

    **Nothing pinned the tree.** Measured: point ``cvc5-flat`` at ``8ef8f75``
    instead of ``0ad22bb`` and ``tools/property_check.py`` says "FIRED — the
    property failed where it is supposed to", because ``[flat]`` is stamped on
    all three of ``_judge``'s messages and the failure at ``8ef8f75`` is
    clause (1)'s. The registry's own disclosure — that ``[flat]``'s breadth is
    safe BECAUSE the tree it points at violates clause (2) and only clause
    (2) — becomes false, silently, and every gate stays green.

    A tree is the revision AND the mutation applied to it: ``at="HEAD"`` with
    two different mutations is two different trees.
    """
    by_property: dict[str, list] = {}
    for c in pc.CONTROLS:
        by_property.setdefault(c.nodeid, []).append(c)
    shared = []
    for nodeid, controls in sorted(by_property.items()):
        for a in controls:
            for b in controls:
                if a.name < b.name and (a.at, a.mutation) == (b.at, b.mutation):
                    shared.append(
                        f"{a.name} and {b.name} both name {nodeid} at "
                        f"{a.at!r}"
                        + ("" if a.mutation is None else " under one mutation")
                    )
    assert not shared, (
        "these controls name the same property AND point at the same tree, so "
        "they are one piece of evidence reported twice — whatever each one's "
        "`why` says it demonstrates, they both find whichever failure that "
        "tree produces first:\n  " + "\n  ".join(shared)
    )


def test_a_commit_control_names_the_revision_it_points_at_in_its_own_why():
    """``at`` is one field, and a field can be edited without editing the prose.

    Every ``why`` here is an argument about a specific tree — which defect it
    carries, which conjunct of the oracle that defect reaches, why some OTHER
    revision cannot supply the same evidence. Repointing ``at`` invalidates
    that argument. Nothing made the two move together, and the runner cannot
    help: it reports FIRED for any red run carrying the guard, whatever the
    revision under it.

    So the revision must appear in the sentence that argues for it.

    WHAT THAT IS WORTH, MEASURED — because "this makes a SILENT repoint
    impossible" is what it said, and that is stronger than the check. It is a
    substring test over the WHOLE ``why``, and a ``why`` that argues about a
    tree usually names a second revision: the commit that fixed the defect, or
    the one that added the guard whose absence IS the defect. FIVE of the seven
    commit controls name one, and repointing ``at`` to it passes this test
    unchanged. Driven, one repoint at a time, against every gate::

        vacuous-refutation -> 463ee81   passes; per-push 8/9, DID NOT FIRE
        conjunct           -> 717b9ca   passes; per-push 8/9, DID NOT FIRE
        cvc5-flat          -> 8d3051a   passes; per-push 8/9, DID NOT FIRE
        cvc5-exit-tell     -> 0ad22bb   passes; the same-tree pin above fires,
                                        and per-push 8/9, wrong failure
        cross-series       -> 76140c2   passes; NOTHING CI RUNS NOTICES

    ``cross-series`` is the one that gets through everything. It is the only
    control on its property, so the same-tree pin above has no sibling to hold
    it against; it is not in the per-push ``--control`` list; and demonstrating
    it needs ``--other-python``, which no job passes. Driven with one —
    ``--control cross-series --other-python`` — it IS caught, ``0/1 controls
    fired``, "passed where it must fail". That is an invocation no workflow
    makes: ``property_check.py`` appears once under ``.github/workflows/``, in
    the nine-name step.

    So the claim this test can carry is the narrow one: it refuses a repoint to
    any revision the control's own ``why`` has never mentioned. What it does NOT
    refuse is the revisions those ``why``s already name — the five above, and
    ``cvc5-exit-tell`` -> ``b6e4783``, which passes this test with nothing else
    static firing and is caught by the per-push step, 8/9, wrong failure — and,
    by the exemption below, a repoint to ``at="HEAD"``. For ``cross-series``,
    whose demonstration no job runs at all, that is a SECOND repoint nothing CI
    notices, not one.
    Closing them means giving ``why`` a canonical place to name its
    own tree rather than a substring test over the whole of it, and that is a
    registry change, not a test change.

    ``at="HEAD"`` is exempt: it names no revision, and "runs against the
    working tree" is what the surrounding ``why`` already says.
    """
    bad = []
    for c in pc.CONTROLS:
        if c.kind != "commit" or c.at == "HEAD":
            continue
        if c.at not in c.why:
            bad.append(f"{c.name}: points at {c.at!r}, unnamed in its own `why`")
    assert not bad, (
        "these commit controls point at a revision their own `why` never "
        "mentions, so the field and the argument for it can drift apart "
        "without anything noticing:\n  " + "\n  ".join(bad)
    )


# The scope block the property job prints, and the step that backs it. Kept
# here rather than in a workflow test because what it has to agree with is the
# registry, which lives next door.
_CI = REPO / ".github" / "workflows" / "ci.yml"
_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
}


def test_the_property_job_runs_the_controls_it_says_it_runs():
    """The per-push `--control` list, and every count the job prints about it.

    The list is read by nothing. A control could be dropped from the step —
    which is how ``cvc5-exit-tell`` and ``cvc5-phantom-model`` spent their
    first two commits, registered and run by nobody — or a name could be
    misspelt, and the step would keep exiting 0 over a shorter list.

    THE COUNTS ARE THE SAME PROBLEM AND HAVE ALREADY GONE WRONG. The scope
    block is one ``cat <<'EOF'`` heredoc; it said "seven controls still fire"
    twenty-four lines above "NINE of the twelve positive controls FIRE", both
    about this job, both on the same push. Nothing reads a heredoc, so nothing
    noticed. Every one of those numbers is derivable from the step itself and
    from the registry, so it is derived here instead of trusted:

    * how many FIRE            = the number of ``--control`` names in the step
    * how many there are       = ``len(pc.CONTROLS)``
    * how many are NOT run     = the difference

    ``tests/property/README.md`` carries the same sentence about the same job
    and is held to the same number.
    """
    ci = _CI.read_text()
    readme = (HERE / "README.md").read_text()
    ran = re.findall(r"^\s*--control\s+([\w-]+)", ci, re.M)
    registered = {c.name for c in pc.CONTROLS}

    assert ran, (
        "the property job's step has no `--control` names at all. Either the "
        "anti-vacuity step was deleted or this test can no longer find it, and "
        "both are the same kind of silence."
    )
    assert len(set(ran)) == len(ran), f"a control is listed twice: {sorted(ran)}"
    unknown = sorted(set(ran) - registered)
    assert not unknown, (
        "the property job runs `--control` names that are not in the registry, "
        "so the step exits non-zero on every push with a KeyError rather than "
        "on the push that breaks a property:\n  " + "\n  ".join(unknown)
    )

    bad = []
    for text, where, pattern, want, what in (
        (ci, "ci.yml", r"([A-Za-z]+) of the ([A-Za-z]+) positive controls FIRE",
         (len(ran), len(pc.CONTROLS)), "the RUN block's count"),
        (ci, "ci.yml", r"([A-Za-z]+) of the ([A-Za-z]+) positive controls\s+—",
         (len(pc.CONTROLS) - len(ran), len(pc.CONTROLS)),
         "the job header's NOT-RUN count"),
        (ci, "ci.yml", r"([A-Za-z]+) controls still fire",
         (len(ran),), "the rot-detector sentence"),
        (readme, "tests/property/README.md", r"([A-Za-z]+) controls still fire",
         (len(ran),), "the rot-detector sentence"),
    ):
        found = re.findall(pattern, text)
        if not found:
            bad.append(f"{where}: {what} is gone — {pattern!r} matches nothing")
            continue
        for match in found:
            words = match if isinstance(match, tuple) else (match,)
            got = tuple(_WORDS.get(w.lower()) for w in words)
            if got != want:
                bad.append(
                    f"{where}: {what} reads {' of '.join(words)!r}, and the "
                    f"step plus the registry say {want}"
                )
    assert not bad, (
        "the property job prints counts about itself that its own "
        "`--control` list and tests/property/positive_controls.py do not "
        "support. A green tick plus a wrong number in the log is worse than "
        "no number:\n  " + "\n  ".join(bad)
    )
