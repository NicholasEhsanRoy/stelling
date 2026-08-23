# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""``tests/_solver_gate.py`` is the only place that decides "a solver is here".

WHY THIS FILE IS A PARSER AND NOT A SENTENCE. The gate's docstring has now
stated the size of the problem it solves three times and been wrong three
times — *seven* spellings, then *eight*, then *nine*, and after the ninth was
folded in there were still three wheels-only decisions left in the tree. Every
correction was made by counting by hand, which is the same act that produced
the wrong number in the first place. So the claim is not a count any more. It
is a RULE with one named exception, and this measures it.

THE RULE. A test may ask whether the *z3 wheel* is installed, or whether the
*cvc5 wheel* is, or whether *both* are — those are backend-specific questions
with backend-specific answers. What it may not do is spell **"any SMT backend
at all"** for itself, because there is a third route to one: an external
``cvc5`` binary named by ``STELLING_CVC5`` or found on ``PATH``, which is the
route ``docs`` actively recommend and the only way to get a build the PyPI
wheel does not ship. ``available("z3") or available("cvc5")`` says "no solver"
in an environment that has one, and a test that skips there is reporting "not
applicable" about a configuration in which it applies perfectly well. That is
the false skip :data:`_solver_gate.HAVE_SOLVER` exists to make impossible.

THE ONE EXCEPTION, NAMED. ``tests/test_doc_examples.py`` carries the same
narrow spelling at its ``solver_timeout_ms`` gate. It is listed in
:data:`NARROW_BY_DECLARATION` rather than fixed here because that file is
being changed on another branch at the time of writing, and a fence that
silently tolerates what it cannot reach is worse than one that names it: the
entry is the reason it is still there, and removing the entry is what makes
this file go red until the site is folded in.
"""

from __future__ import annotations

import ast
import pathlib

# the runner's own conftest, for its `norecursedirs` pruning and nothing
# else — see `test_no_test_module_decides_ANY_SOLVER_AT_ALL_for_itself`
from conftest import _in_a_pruned_directory

import _solver_gate

TESTS = pathlib.Path(__file__).resolve().parent

#: Files allowed to spell the NARROW either-solver question themselves, each
#: with the reason.
#:
#: ``_solver_gate.py`` IS NOT IN HERE, and that is worth a sentence: its own
#: predicate carries the third route, so it is wide and the scan below never
#: reports it. An entry for it would be a licence for a defect it does not
#: have, which is the dangling-exemption shape this repository refuses
#: everywhere else.
NARROW_BY_DECLARATION = {
    "test_doc_examples.py": (
        "same narrow spelling at the `solver_timeout_ms` gate; owned by the "
        "documentation-routing branch, which is live in this file. Fold it "
        "into `_solver_gate.HAVE_SOLVER` there and delete this entry."
    ),
}

# WHAT THIS SCAN DOES NOT SEE — a floor, not a ceiling, and an AST matcher is
# incomplete by nature rather than by oversight. `either_solver_decisions`
# matches ONE syntactic shape: an `or` whose operands call `available()` with
# LITERAL "z3" and "cvc5", resolving module-level `NAME = …` bindings one level
# deep. Every other spelling of the same decision walks past it. Driven, each
# of these reads as clean:
#
#   importlib.util.find_spec("z3") or find_spec("cvc5")   <- AND THIS ONE HAS
#     ALREADY BITTEN THIS REPOSITORY. `_solver_gate.py`'s docstring records
#     `ci.yml`'s no-solver guard step asking exactly it: WHEEL absence proved
#     and read as SOLVER absence, with a `cvc5` on PATH making the real
#     predicate true and two tests failing behind it.
#   "z3" in sys.modules or "cvc5" in sys.modules
#   two `try: import …` / `except ImportError:` flags and then
#     `HAVE_Z3 or HAVE_CVC5` — the flags are bound inside the `try`, and
#     `_module_level_names` walks `tree.body` only
#   any(available(w) for w in ("z3", "cvc5")), and the same over a named tuple
#   `if available("z3"): … elif available("cvc5"): …` — no BoolOp at all
#   `available(name)` with a non-literal argument
#   the two halves bound inside a FUNCTION rather than at module level
#   pytest.importorskip("z3") or pytest.importorskip("cvc5")
#   flags read off another module: `_flags.HAVE_Z3 or _flags.HAVE_CVC5`
#
# NONE OF THEM IS IN THIS TREE TODAY, grepped, and each near miss checked
# rather than counted: no `find_spec` call under `tests/` names a solver wheel
# (the calls ask about `jax`; the other hits are meta-path finders' own
# `def find_spec` methods); the one `sys.modules` membership test that names a
# solver asserts `"z3" not in sys.modules` after `available("z3")`
# (`test_import_hygiene.py`), which is a no-import assertion and not a
# run/skip decision; and the `importorskip("z3")`/`("cvc5")` pairs are
# CONSECUTIVE calls — the two-backend `and`, a different question with a
# different answer, which the module docstring above already excludes. So the
# rule holds by measurement, and this list is what stops a green run here
# being read as "no such decision can exist".
#
# **EXCEPT THE LAST SPELLING, WHICH IS IN THIS TREE AND HAS BEEN ALL ALONG.**
# `tests/test_skip_inventory.py`'s solver rules write
# `legitimate=lambda: not (_wheel("z3") or _wheel("cvc5"))`, and `_wheel` is
# `_optional.available` with a docstring on it — exactly the shape above, and
# invisible here because `_calls` matches the NAME `available`. It is not a
# defect there and that file argues why: the predicate exists only to call a
# skip WRONG, so the narrow reading is the SAFE one — with an external cvc5
# and no wheels it answers "no wheel", the gate it is judging did fire, and
# no skip is accused. It is named here because *"NONE OF THEM IS IN THIS TREE
# TODAY"* reads as "no invisible disjunction exists", and one does. The
# residue is the class rather than the instance: a one-line wrapper of
# `available` is the cheapest way to write the narrow question by accident,
# and this scan will not report it.
#
# AND NOTHING ELSE WOULD CATCH ONE. The environment a narrow spelling
# misbehaves in is an external `cvc5` with NEITHER wheel, and no lane in
# `ci.yml` runs it: `test-jax-no-solvers` has no binary either, so the narrow
# and the wide predicate agree there and the false skip is invisible. That
# lane's guard step asserting `_solver_gate.HAVE_SOLVER` keeps the DEFINITION
# honest; it cannot see a copy of the question written somewhere else.


def _module_level_names(tree: ast.Module) -> dict[str, ast.expr]:
    """``NAME = <expr>`` at module level, so ``HAVE_Z3 or HAVE_CVC5`` resolves.

    One level deep and no more. The spelling this fence exists to catch is
    cheap to write, and the cheapest way to write it indirectly is to bind the
    two halves to names first — which is exactly what six modules in this tree
    do for the ``and`` case (``not (HAVE_Z3 and HAVE_CVC5)``).
    """
    bound: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                bound[target.id] = node.value
    return bound


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    """Every call to ``name`` or ``<anything>.name`` inside ``node``."""
    found = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called == name:
            found.append(sub)
    return found


def _wheels_asked_for(node: ast.AST) -> set[str]:
    asked = set()
    for call in _calls(node, "available"):
        if call.args and isinstance(call.args[0], ast.Constant):
            asked.add(call.args[0].value)
    return asked


def either_solver_decisions(source: str) -> list[int]:
    """The line of every ``or`` in ``source`` that means "any SMT backend".

    A decision qualifies when one ``or`` expression asks for BOTH wheels and
    asks about no binary. Asking for one wheel is a backend question; asking
    for both with ``and`` is a two-backend question; asking for either wheel
    **or** ``cvc5_binary()`` is this repository's actual predicate and is what
    ``_solver_gate`` spells.
    """
    tree = ast.parse(source)
    bound = _module_level_names(tree)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
            continue
        operands = [bound.get(v.id, v) if isinstance(v, ast.Name) else v
                    for v in node.values]
        wheels: set[str] = set()
        binary = False
        for operand in operands:
            wheels |= _wheels_asked_for(operand)
            binary = binary or bool(_calls(operand, "cvc5_binary"))
        if {"z3", "cvc5"} <= wheels and not binary:
            lines.append(node.lineno)
    return lines


def test_the_detector_can_see_both_spellings():
    """ANTI-VACUITY, and it is the whole risk this file carries.

    A parser that stopped matching would report nothing and read as "the tree
    is clean" — the exact shape of instrument failure the batch this file
    belongs to is about. So the narrow spelling, the indirect narrow spelling
    and the three shapes that are NOT this defect are all driven here.
    """
    narrow = 'x = available("z3") or available("cvc5")'
    assert either_solver_decisions(narrow) == [1]
    assert either_solver_decisions(
        'x = not (_optional.available("cvc5") or _optional.available("z3"))'
    ) == [1]
    # bound to names first, which is how five modules already spell the `and`
    assert either_solver_decisions(
        'HAVE_Z3 = available("z3")\n'
        'HAVE_CVC5 = available("cvc5")\n'
        "x = HAVE_Z3 or HAVE_CVC5\n"
    ) == [3]
    # and the shapes that are legitimate
    assert either_solver_decisions('x = available("z3")') == []
    assert either_solver_decisions(
        'x = available("z3") and available("cvc5")'
    ) == []
    assert either_solver_decisions(
        'x = available("z3") or available("cvc5") or cvc5_binary() is not None'
    ) == [], "the gate's OWN predicate must not be reported"
    assert either_solver_decisions(
        'HAVE_CVC5 = available("cvc5") or cvc5_binary() is not None\n'
        'x = available("z3") or HAVE_CVC5\n'
    ) == [], "a wide cvc5 half makes the whole `or` wide"


def test_no_test_module_decides_ANY_SOLVER_AT_ALL_for_itself():
    """The measurement that replaced the count.

    Every hit is a false skip waiting for an environment with an external
    cvc5 and no wheels: the test would pass there and says it does not apply.

    **THE SWEEP IS THE SUITE, NOT THE DIRECTORY — 0.2.0 D14.** It was
    `TESTS.rglob("*.py")` bare, so a stale copy in a directory pytest
    refuses to recurse into was reported as an offender. Driven at
    `a431646`::

        tests/build/test_offender_copy.py    1 failed
        tests/.junk/test_offender_copy2.py   (same run, both named)

    Neither file can be collected by any invocation — `pytest
    --collect-only` reaches **0** tests in either — so neither can produce
    the false skip this exists to prevent, and reddening on one is a verdict
    about the developer's directory rather than about the suite.

    **AND THE DIRECTION MATTERS, WHICH IS WHY THIS IS NOT THE SAME REPAIR AS
    `tests/test_prose_hygiene.py`'s.** This is an `assert not offenders`
    scan: an unpruned walk can only report MORE, never less, so it was never
    a false GREEN and pruning it is not a narrowing of what it can catch. It
    was a false RED, and it also let a pruned copy borrow a
    `NARROW_BY_DECLARATION` exemption, which is keyed by BASENAME.

    One predicate, `conftest._in_a_pruned_directory` — not
    `collectable_test_files()`, which yields `test_*.py` only, while this
    scan is over every `.py` under `tests/` including the helpers.
    """
    offenders = {}
    for path in sorted(TESTS.rglob("*.py")):
        if _in_a_pruned_directory(path):
            continue
        lines = either_solver_decisions(path.read_text(encoding="utf-8"))
        if lines and path.name not in NARROW_BY_DECLARATION:
            offenders[str(path.relative_to(TESTS))] = lines
    assert not offenders, (
        f"these decide `any SMT backend at all` from the two wheels alone: "
        f"{offenders}. Import `need_solver` or `HAVE_SOLVER` from "
        f"tests/_solver_gate.py instead — it also accepts an external cvc5 "
        f"binary (STELLING_CVC5, or `cvc5` on PATH), which is the route the "
        f"documentation recommends. A wheels-only spelling skips, in that "
        f"environment, a test that would have passed."
    )


def test_every_declared_exception_still_has_the_defect_it_declares():
    """A dangling exception is a licence pointed at nothing.

    Both directions: the file has to exist, and it has to still contain the
    spelling the entry excuses. When the documentation branch folds
    ``test_doc_examples.py`` in, this goes red and the entry comes out — which
    is the only way an exemption list stays a list rather than a habit.
    """
    for name, why in NARROW_BY_DECLARATION.items():
        target = TESTS / name
        assert target.is_file(), f"{name} is declared here and is not in tests/"
        assert why.strip(), f"{name} is excepted with no reason given"
        assert either_solver_decisions(target.read_text(encoding="utf-8")), (
            f"{name} no longer spells the either-solver question itself, so "
            f"its entry in NARROW_BY_DECLARATION licenses nothing. Delete it."
        )
    # and the definition itself needs no licence, because it has no defect
    gate = (TESTS / "_solver_gate.py").read_text(encoding="utf-8")
    assert not either_solver_decisions(gate), (
        "tests/_solver_gate.py's own predicate stopped carrying the external "
        "binary, so the one definition is now as narrow as the copies it "
        "replaced"
    )


def test_the_gate_itself_accepts_the_third_route():
    """The property every folded-in site is now relying on.

    ``HAVE_SOLVER`` is a value computed at import against this environment, so
    it cannot be asserted directly. What CAN be asserted is that it is built
    from all three routes and not two — which is what a site folding itself in
    is buying.
    """
    source = (TESTS / "_solver_gate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "HAVE_SOLVER" for t in node.targets)
        and not isinstance(node.value, ast.Constant)
    ]
    assert assigned, "HAVE_SOLVER is no longer computed from anything"
    predicate = assigned[0].value
    assert _wheels_asked_for(predicate) == {"z3", "cvc5"}, (
        "HAVE_SOLVER stopped asking about one of the two wheels"
    )
    assert _calls(predicate, "cvc5_binary"), (
        "HAVE_SOLVER stopped accepting an external cvc5 binary, which is the "
        "route that makes every wheels-only spelling in this tree a FALSE "
        "SKIP — the whole reason the definition is in one place"
    )
    assert _solver_gate.need_solver.kwargs["reason"] == "needs an SMT solver", (
        "the marker's reason is a key into tests/test_skip_inventory.py's "
        "RULES; a different spelling is an undisclosed skip"
    )
