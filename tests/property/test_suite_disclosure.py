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
import pathlib
import re

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


def _tests_defined_in(path: pathlib.Path):
    """``test_*`` functions and stateful ``TestCase`` bindings, parsed."""
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                out.append(f"tests/property/{path.name}::{node.name}")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                # `TestCvcTransport = CvcTransport.TestCase` — a unittest
                # TestCase pytest collects as `::runTest`.
                if isinstance(t, ast.Name) and t.id.startswith("Test"):
                    out.append(f"tests/property/{path.name}::{t.id}::runTest")
    return out


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
        "reorder",              # order-dependence among independent obligations
        "widen",                # non-monotonicity in the input box
        "cvc5-flat",            # the record-boundary disagreement
        "cvc5-stateful",
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
