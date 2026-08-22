# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Import hygiene: jax only in _jax_compat; private jax modules in exactly one
file; solvers never at module level — the zero-dep surface must import clean.

Two kinds of check live here and they measure different things.

*Token scans* (:func:`test_jax_imported_only_in_compat_module` and friends)
enforce exactly one thing: **no file outside ``_jax_compat.py`` spells the
jax token.** That is a proxy for the churn boundary, not a proof of it — a
file can reach jax without an ``import jax`` / ``from jax`` **statement**
(``from stelling._jax_compat import jnp, np, jax`` works today, and so does a
module-scope ``importlib.import_module("jax")`` with a plain literal name —
no splicing needed), and the scan stays green for both. It is worth pinning as the
cheap, statically checkable half — but it says nothing about what a user
without jax sees. It was mistaken for that guarantee once: ``harness.py``
contained no ``import jax`` token and still died at import in a bare
environment, with the token scan green, because it imported the module that
imports jax.

*Behavioural checks* (everything below :data:`_NO_JAX_PRELUDE`) measure the
thing the token scan cannot: **import the module with jax genuinely absent
and read the failure.** A user who runs ``pip install stelling`` and copies
the first line out of the README gets that failure, and it has to name the
extra that fixes it rather than a private module two levels down. Most of
them hide jax from a subprocess with a ``meta_path`` hook, which is a
stand-in for the import route and nothing else; one builds an interpreter
that really does not have jax
(:func:`test_a_real_jax_less_interpreter_says_the_same_thing`) and is what
the hook is checked against, since the two differ where a caller asks
whether jax is present rather than importing it.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import stelling

REPO = Path(__file__).resolve().parents[1]
# The tree the suite actually imported, which is what the subprocesses must
# exercise too. It is NOT derived from REPO: an editable install pointing at
# another checkout makes the two different, and then the token scans (which
# walk REPO) and the probes (which import SRC) measure different packages.
# test_the_probe_subprocess_imports_the_tree_under_test is the assertion that
# they are the same tree; nothing else in this file can see the difference.
SRC = Path(stelling.__file__).resolve().parents[1]
JAX_IMPORT = re.compile(r"^\s*(import jax\b|from jax\b)")
# built by concatenation so this test file cannot match itself
PRIVATE_JAX = "jax." + "_src"
SOLVER_MODULES = ("z3", "cvc5")

# THE TWO EXEMPTIONS, AND THEY ARE NOT THE SAME KIND OF THING.
#
# Rule 1's is a BASE NAME, matching `--exclude=_jax_compat.py` in the hook —
# grep's `--exclude` matches a base name when searching recursively, so any
# file under src/ called that is exempt in both controls. That is deliberate
# and is recorded as measured in the hook's own comment
# (`src/zzsub/_jax_compat.py` with a bare `import jax`: Passed in both).
#
# Rule 2's is a full repo-relative PATH, matching the hook's anchored
# `grep -v "^src/stelling/_tripwire/_adapter_jax\.py:"`. It has to be a path:
# `--exclude` cannot express one (spelled with slashes it is silently inert,
# measured on GNU grep 3.11), and a base name would exempt a decoy at any
# other path. `design/private-jax-boundary.md` carries that table, and the
# measured reason the exemption exists at all — the registry the tripwire
# attaches to is exported by no public or `jax.extend` module on either tested
# series.
JAX_IMPORT_EXEMPT_NAME = "_jax_compat.py"
PRIVATE_JAX_EXEMPT_PATH = Path("src/stelling/_tripwire/_adapter_jax.py")

# The install line inside what `require("jax")` promises the user. This is a
# hard-coded literal, so it pins the message against a silent reword but says
# nothing about the doc: docs/harness-api.md quotes the WHOLE sentence, and
# test_the_documented_failure_is_the_measured_failure compares the page
# against the message as measured rather than against anything written here.
JAX_EXTRA_HINT = 'pip install "stelling[jax]"'


def test_jax_imported_only_in_compat_module():
    """No file outside ``_jax_compat.py`` spells ``import jax`` / ``from jax``.

    That is the whole claim, and it is a proxy for the churn boundary rather
    than the boundary itself: a file that obtains jax without the token —
    ``from stelling._jax_compat import jax``, or a name assembled at runtime —
    leaves this green. What it does buy is cheap: a *grep-visible* boundary, so
    a jax release that moves a symbol has one file to read first, and the
    ``jax-import-hygiene`` pre-commit hook can enforce it without importing
    anything.

    NOT a claim about bare environments either — a module can require jax at
    import without ever spelling the token, which is exactly how ``harness.py``
    passed this while failing the property people read into it. That one is
    measured by :func:`test_importing_harness_without_jax_names_the_extra`.
    """
    offenders = []
    for path in (REPO / "src").rglob("*.py"):
        if path.name == JAX_IMPORT_EXEMPT_NAME:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if JAX_IMPORT.match(line):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "jax may only be imported in stelling/_jax_compat.py:\n" + "\n".join(offenders)


def test_private_jax_modules_banned_everywhere():
    """One file may name a private jax module, and it is named by PATH.

    The exemption is not a widening of the rule above and does not inherit
    from it: ``_tripwire/_adapter_jax.py`` is exempt from *this* check and is
    still subject to :func:`test_jax_imported_only_in_compat_module`, so an
    ``import jax`` added to it reddens the suite exactly as it would in any
    other module. That is the whole relaxation — one rule, one path — and
    ``design/private-jax-boundary.md`` records the measurement that closed the
    alternatives, including the seven public and ``jax.extend`` modules that do
    not export the registry the adapter is for.

    ``tests/`` is still covered with no exemption but this file's own (which
    is why :data:`PRIVATE_JAX` is built by concatenation as well): the
    tripwire's tests reach the registry through the adapter's API and never
    name it. The pre-commit hook scans ``src/`` only — a deliberate difference
    in SCOPE, not in the exempt set, and
    :func:`test_the_hook_and_this_file_exempt_the_same_set` is what holds the
    exempt set together.
    """
    offenders = []
    for directory in ("src", "tests"):
        for path in (REPO / directory).rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            if path.relative_to(REPO) == PRIVATE_JAX_EXEMPT_PATH:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if PRIVATE_JAX in line:
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"{PRIVATE_JAX} may only be named in {PRIVATE_JAX_EXEMPT_PATH}:\n"
        + "\n".join(offenders)
        + "\nThat file exists because the const-fold registry is exported by no "
        "public or `jax.extend` module on either tested series. If you need the "
        "same thing, call it — do not add a second bearer."
    )


def test_the_private_jax_exemption_is_load_bearing_and_is_exactly_one_file():
    """An exemption nothing uses is an exemption nobody watches biting.

    Both directions in one list, which is the acceptance criterion the plan
    states as "the private-module token appears in exactly one file under
    ``src/``" — spelled that way here, and everywhere else in this file,
    because :data:`PRIVATE_JAX` is built by concatenation precisely so that
    this file cannot match itself. It is also exempt by name in the check
    above, and relying on that instead would be one control where there are
    currently two.

    * the exempt path really does name the private module — so the check above
      is exercised rather than decorative, and a deletion of the tripwire has
      to take the exemption with it;
    * and nothing else under ``src/`` does.
    """
    bearers = sorted(
        str(path.relative_to(REPO))
        for path in (REPO / "src").rglob("*.py")
        if PRIVATE_JAX in path.read_text(encoding="utf-8")
    )
    assert bearers == [str(PRIVATE_JAX_EXEMPT_PATH)], (
        f"exactly one file under src/ may name {PRIVATE_JAX}, and it is "
        f"{PRIVATE_JAX_EXEMPT_PATH}. Found: {bearers}.\n"
        "An empty list means the exemption now points at a file that does not "
        "use it: remove the exemption from BOTH this file and the "
        "jax-import-hygiene hook, or the next file to want it inherits a hole "
        "nobody decided to leave."
    )


def _hook_entry(hook_id: str) -> str:
    """The ``entry:`` scalar of one local pre-commit hook, folded to one line.

    Read as text rather than parsed: this file is deliberately dependency-free
    (a YAML parser is not a core dependency, and the zero-dep lane runs these
    checks). The scalar is sliced by INDENTATION so that the hook's comment
    block — which quotes both exemption spellings, including the ones measured
    NOT to work — cannot be mistaken for the code that runs.
    """
    lines = (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == f"- id: {hook_id}"),
        None,
    )
    assert start is not None, f"no hook with id {hook_id!r} in .pre-commit-config.yaml"
    item_indent = len(lines[start]) - len(lines[start].lstrip())

    for i in range(start + 1, len(lines)):
        stripped = lines[i].lstrip()
        indent = len(lines[i]) - len(stripped)
        if stripped.startswith("- id: ") and indent == item_indent:
            break  # the next hook: this one has no entry
        if not stripped.startswith("entry:"):
            continue
        body = []
        for j in range(i + 1, len(lines)):
            nxt = lines[j]
            if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                break
            body.append(nxt.strip())
        return " ".join(body)
    raise AssertionError(f"hook {hook_id!r} has no entry: scalar")


def test_the_hook_and_this_file_exempt_the_same_set():
    """The two controls have a recorded history of disagreeing, so it is read.

    The hook and these checks are meant to be logically identical on WHAT is
    exempt; they differ on scope by design (the hook scans ``src/``, this file
    scans ``src/`` and ``tests/``). The first half of the hook was
    unenforceable for a period and its exemption was once by line CONTENT, both
    recorded in its own comment — a divergence here is not hypothetical.

    Derived from the shipped config, not restated: the exemption spellings are
    parsed out of the ``entry:`` that actually runs and compared with the
    constants these checks use. Adding a third exemption to either side, or
    moving one, fails here.
    """
    entry = _hook_entry("jax-import-hygiene")

    # rule 1: every `--exclude` in the entry, base names by construction
    excludes = set(re.findall(r"--exclude=([^\s;)]+)", entry))
    assert excludes == {JAX_IMPORT_EXEMPT_NAME}, (
        f"the hook's --exclude set is {sorted(excludes)}; this file exempts "
        f"{JAX_IMPORT_EXEMPT_NAME!r} by name and nothing else. A `--exclude` "
        "spelled with slashes is silently inert (measured, GNU grep 3.11), so "
        "a path added here would be an exemption in name only."
    )

    # rule 2: the anchored path filter(s), which is how a PATH is expressed
    anchored = re.findall(r'grep -v "\^([^"]+):"', entry)
    assert len(anchored) == 1, (
        f"expected exactly one anchored path exemption in the hook, found "
        f"{anchored}. Every one of them is a file allowed to name "
        f"{PRIVATE_JAX}, and this file exempts exactly one."
    )
    assert Path(anchored[0].replace("\\.", ".")) == PRIVATE_JAX_EXEMPT_PATH, (
        f"the hook exempts {anchored[0]!r} and this file exempts "
        f"{PRIVATE_JAX_EXEMPT_PATH}. They must be the same file."
    )


def test_importing_stelling_pulls_in_no_jax():
    """``import stelling`` imports no jax, read from ``sys.modules``.

    Load-bearing for the zero-dep posture and newly at risk: the tripwire is a
    package under ``src/stelling/`` whose whole purpose is to reach into jax,
    and a plugin that armed a hook at import time would take this down without
    touching any of the checks above. Measured in a fresh interpreter, because
    the suite's own process has jax loaded long before this runs.

    Carries its own positive control: where jax IS installed, the probe forces
    it in and confirms it can see it. Without that, an environment with no jax
    reports the same empty list for the wrong reason.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(
            """
            import importlib.util, json, sys

            import stelling

            out = {
                "file": stelling.__file__,
                "jax_modules": sorted(
                    m for m in sys.modules if m.split(".")[0] == "jax"
                ),
                "jax_installed": importlib.util.find_spec("jax") is not None,
            }
            if out["jax_installed"]:
                __import__("jax")
                out["visible_when_forced"] = "jax" in sys.modules
            print(json.dumps(out))
            """
        )],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": str(SRC), "JAX_PLATFORMS": "cpu"},
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stdout}\n{proc.stderr}"
    got = json.loads(proc.stdout.strip().splitlines()[-1])

    assert Path(got["file"]).resolve().parents[1] == SRC, got
    assert got["jax_modules"] == [], (
        "`import stelling` pulled jax in: " + ", ".join(got["jax_modules"])
        + ". The core is zero-dep and jax is an extra; whatever imported it "
        "must reach it inside a function through stelling._optional instead."
    )
    if got["jax_installed"]:
        assert got["visible_when_forced"] is True, got


def test_importing_the_conftest_pulls_in_no_jax_and_arms_nothing():
    """``tests/conftest.py`` is imported by every test file, zero-dep too.

    IT GREW A PERIMETER HELPER, and that is why this exists.
    :func:`conftest.lowered_perimeter` takes the dunder perimeter down for a
    block and hands it back; it is entered by three files that all gate on
    jax, but it lives in the one module every session imports **before
    collection**, in every lane. Three properties, and none of them is
    self-evident from reading it:

    * importing the conftest imports **no jax and no numpy** — the helper
      binds ``stelling._tripwire.perimeter`` inside the function body, and
      that module is itself numpy-free by construction;
    * importing it **arms nothing** — it is a plain function, not a fixture,
      so nothing runs unless a caller enters the block;
    * with nothing armed, entering the block is a **no-op** that yields
      ``()`` — which is what every dial-off and zero-dep session does if it
      reaches it at all.

    Measured in a fresh interpreter with the tree's ``tests/`` on the path,
    because this suite's own process has jax loaded long before this runs, and
    with a positive control for exactly :func:`test_importing_stelling_pulls_
    in_no_jax`'s reason.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(
            """
            import importlib.util, json, sys

            import conftest

            out = {
                "file": conftest.__file__,
                "heavy_modules": sorted(
                    m for m in sys.modules
                    if m.split(".")[0] in ("jax", "jaxlib", "numpy", "ml_dtypes")
                ),
                "jax_installed": importlib.util.find_spec("jax") is not None,
            }
            # entering the block with nothing armed must do nothing at all,
            # and must not import anything heavy on the way
            with conftest.lowered_perimeter() as lowered:
                out["yielded"] = list(lowered)
            out["heavy_after_entering"] = sorted(
                m for m in sys.modules
                if m.split(".")[0] in ("jax", "jaxlib", "numpy", "ml_dtypes")
            )
            from stelling._tripwire import perimeter
            out["armed_faces"] = list(perimeter.armed_faces())
            out["owners"] = perimeter.owners()
            if out["jax_installed"]:
                __import__("jax")
                out["visible_when_forced"] = "jax" in sys.modules
            print(json.dumps(out))
            """
        )],
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(REPO / "tests"), str(SRC)]),
            "JAX_PLATFORMS": "cpu",
        },
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stdout}\n{proc.stderr}"
    got = json.loads(proc.stdout.strip().splitlines()[-1])

    assert Path(got["file"]).resolve().parent == REPO / "tests", got
    assert got["heavy_modules"] == [], (
        "importing tests/conftest.py pulled in " + ", ".join(got["heavy_modules"])
        + ". Every test file in this tree imports it, the zero-dep lane "
        "included, so anything it reaches at module scope is a dependency of "
        "collection itself."
    )
    assert got["yielded"] == [], got
    assert got["heavy_after_entering"] == [], (
        "entering lowered_perimeter() with nothing armed imported "
        + ", ".join(got["heavy_after_entering"])
    )
    assert got["armed_faces"] == [] and got["owners"] == 0, (
        f"importing the conftest and entering the helper left state behind: {got}"
    )
    if got["jax_installed"]:
        assert got["visible_when_forced"] is True, got


def _module_level_statements(tree: ast.Module):
    """Module-level statements, descending through module-level compound
    statements (if/try/with/for/while) but never into function or class
    bodies — an import inside those is the allowed lazy pattern."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            for attr in ("body", "orelse", "finalbody"):
                stack.extend(getattr(node, attr, ()) or ())
            for handler in getattr(node, "handlers", ()) or ():
                stack.extend(handler.body)


def test_solver_imports_are_lazy_everywhere_in_src():
    """The solver mirror of the jax rule: no module-level ``import z3`` /
    ``import cvc5`` (or ``from z3/cvc5 import …``) anywhere under src/.
    Solvers are reached exclusively through ``stelling._optional`` inside
    functions; the zero-dep core must import with no solver installed."""
    offenders = []
    for path in (REPO / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in _module_level_statements(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root in SOLVER_MODULES:
                    offenders.append(
                        f"{path.relative_to(REPO)}:{node.lineno}: "
                        f"module-level solver import of {name!r}"
                    )
    assert not offenders, (
        "solver imports must be lazy (inside functions, via stelling._optional):\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# The behavioural half: what an environment WITHOUT jax actually sees.
# ---------------------------------------------------------------------------

# A meta_path finder that makes jax — and only jax — look uninstalled to an
# IMPORT: `import jax` raises the same exception a bare interpreter raises for
# it (a ModuleNotFoundError whose ``name`` is the missing top-level package,
# which is what ``stelling._optional.require`` keys on). Returning None for
# every other name means "no opinion, keep looking", so the normal finders
# still run: this hides one distribution, it does not sandbox the interpreter.
#
# It is a stand-in for the import route ONLY, and diverges from a real bare
# interpreter in one direction that matters: it raises from `find_spec` where
# a bare interpreter returns None. So under this hook
#
#     importlib.util.find_spec("jax")     raises   (bare: None)
#     stelling.available("jax")           raises   (bare: False)
#     stelling._optional.version("jax")   raises   (bare: None)
#     python -m stelling                  raises   (bare: prints, exits 0)
#
# — a probe that calls any of those is measuring the hook, not an environment
# any user has. None below does; `_run_without_jax` fails closed on a crashed
# probe, so one that started to would be loud rather than quiet. The answers
# this hook cannot give are measured in a real interpreter with an empty
# site-packages instead, in test_a_real_jax_less_interpreter_says_the_same_thing
# — including `python -m stelling`, which is the last step of CI's `test-no-jax`
# job (.github/workflows/ci.yml).
#
# The hook is load-bearing for every test below it, so its import behaviour is
# itself measured in both directions first — see
# test_the_no_jax_hook_blocks_jax_and_nothing_else.
_NO_JAX_PRELUDE = """
import importlib.util as _u, json, sys

_JAX_WAS_INSTALLED = _u.find_spec("jax") is not None


class _NoJax:
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top == "jax":
            raise ModuleNotFoundError(f"No module named {top!r}", name=top)
        return None


sys.meta_path.insert(0, _NoJax())
"""


def _run_without_jax(code: str, *, path_head: list[str] | None = None):
    """Run ``code`` in a fresh interpreter where importing jax fails."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([*(path_head or []), str(SRC)])
    proc = subprocess.run(
        [sys.executable, "-c", _NO_JAX_PRELUDE + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_probe_subprocess_imports_the_tree_under_test():
    """Anti-vacuity for both halves of this file, which read the tree two
    different ways.

    The token scans walk ``REPO / "src"`` — the checkout this file lives in.
    The probes import ``stelling`` and hand :data:`SRC` to a subprocess — the
    tree the suite actually imported. An editable install pointing at another
    checkout makes those two different directories, and then each half passes
    on a package the other never looked at: the scans on the edited source,
    the probes on the installed one. Measured, deleting the guards from a
    checkout and running it with ``PYTHONPATH`` at a copy that still has them
    left this file 9/9 green.

    Comparing the subprocess against ``SRC`` cannot see that — both sides
    derive from the same ``import stelling``, so it is true by construction.
    The assertion that can see it is ``SRC == REPO / "src"``.
    """
    assert SRC == REPO / "src", (
        f"the suite imported stelling from {SRC}, but this file scans "
        f"{REPO / 'src'} — two different trees, so half of this file is "
        "measuring source nobody is editing. Run the suite against the "
        "checkout under test: `pip install -e .` in it, or PYTHONPATH=src."
    )
    # ... and the probes really do put that tree on the subprocess's path
    got = _run_without_jax(
        """
        import stelling
        print(json.dumps({"file": stelling.__file__}))
        """
    )
    assert Path(got["file"]).resolve().parents[1] == SRC, got


def test_the_no_jax_hook_blocks_jax_and_nothing_else(tmp_path):
    """The hook, measured in both directions before anything relies on it.

    Blocks: ``jax`` and its submodules, with a real ModuleNotFoundError whose
    ``.name`` is ``jax`` — the shape ``require`` matches on, so a probe cannot
    pass by raising something merely similar.

    Does not block: the standard library, other installed distributions, the
    package under test, or names that merely *start with* ``jax`` (``jaxlib``
    is a separate distribution, and a hook that ate it would be measuring a
    different environment than the one users have). The two locally created
    modules make that direction checkable in any environment, installed jax or
    not.
    """
    (tmp_path / "jaxen.py").write_text("VALUE = 'not jax'\n")
    (tmp_path / "jax_shaped").mkdir()
    (tmp_path / "jax_shaped" / "__init__.py").write_text("VALUE = 'also not jax'\n")

    got = _run_without_jax(
        """
        out = {"jax_was_installed": _JAX_WAS_INSTALLED}

        for name in ("jax", "jax.numpy", "jax.extend.core"):
            try:
                __import__(name)
            except ModuleNotFoundError as e:
                out[name] = {"blocked": True, "name": e.name, "msg": str(e)}
            else:
                out[name] = {"blocked": False}

        allowed = {}
        for name in ("json", "pathlib", "pytest", "stelling", "stelling.ir",
                     "jaxen", "jax_shaped"):
            try:
                __import__(name)
            except BaseException as e:
                allowed[name] = f"{type(e).__name__}: {e}"
            else:
                allowed[name] = "ok"
        out["allowed"] = allowed
        print(json.dumps(out))
        """,
        path_head=[str(tmp_path)],
    )

    for name in ("jax", "jax.numpy", "jax.extend.core"):
        assert got[name]["blocked"], f"{name} was not blocked: {got[name]}"
        # `.name` must be the missing TOP-LEVEL package, exactly as a bare
        # interpreter reports it; require() refuses to translate anything else
        assert got[name]["name"] == "jax", got[name]
        assert got[name]["msg"] == "No module named 'jax'", got[name]

    assert got["allowed"] == {
        "json": "ok",
        "pathlib": "ok",
        "pytest": "ok",
        "stelling": "ok",
        "stelling.ir": "ok",
        "jaxen": "ok",
        "jax_shaped": "ok",
    }, got["allowed"]

    # and, where jax IS installed, the hook is what did the blocking rather
    # than the environment happening to lack it
    if got["jax_was_installed"]:
        probe = subprocess.run(
            [sys.executable, "-c", "import jax; print('jax imported')"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": str(SRC), "JAX_PLATFORMS": "cpu"},
        )
        assert "jax imported" in probe.stdout, probe.stderr


def test_a_real_jax_less_interpreter_says_the_same_thing(tmp_path):
    """Ground truth for the hook: an interpreter that simply does not have jax.

    ``python -m venv --without-pip`` gives one for free — the core is zero-dep,
    so it imports there with nothing but ``PYTHONPATH``. This is the only test
    in the file that measures an environment rather than a simulation of one,
    and it is here for the three answers the hook gets wrong by construction
    (it raises from ``find_spec`` where absence returns None):

    * ``available("jax")`` is False and ``version("jax")`` is None, rather than
      raising — these are what a caller asking *whether* jax is present gets;
    * ``python -m stelling`` runs, reports the missing extra and exits 0. That
      is the last step of CI's ``test-no-jax`` job, and under the hook it would
      die on the first ``available`` call;
    * ``import stelling.harness`` fails exactly as it does under the hook —
      same class, same sentence — which is what licenses the hook as a
      stand-in for the import route in the tests below.
    """
    venv = tmp_path / "bare"
    made = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        capture_output=True, text=True, timeout=300,
    )
    assert made.returncode == 0, f"could not build a bare venv:\n{made.stderr}"
    python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    env = {**os.environ, "PYTHONPATH": str(SRC)}

    probe = subprocess.run(
        [str(python), "-c", textwrap.dedent(
            """
            import importlib.util, json
            import stelling
            from stelling import _optional

            out = {
                "file": stelling.__file__,
                "find_spec": importlib.util.find_spec("jax") is not None,
                "available": _optional.available("jax"),
                "version": _optional.version("jax"),
            }
            try:
                import stelling.harness
            except BaseException as e:
                out["harness"] = {
                    "raised": True,
                    "type": type(e).__module__ + "." + type(e).__qualname__,
                    "is_import_error": isinstance(e, ImportError),
                    "msg": str(e),
                }
            else:
                out["harness"] = {"raised": False}
            print(json.dumps(out))
            """
        )],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert probe.returncode == 0, f"probe crashed:\n{probe.stdout}\n{probe.stderr}"
    got = json.loads(probe.stdout.strip().splitlines()[-1])

    # the venv is genuinely bare, and it is THIS tree it imported
    assert got["find_spec"] is False, f"the throwaway venv has jax in it: {got}"
    assert Path(got["file"]).resolve().parents[1] == SRC, got

    # the three answers the hook cannot give
    assert got["available"] is False, got
    assert got["version"] is None, got

    cli = subprocess.run(
        [str(python), "-m", "stelling"],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert cli.returncode == 0, f"python -m stelling failed:\n{cli.stdout}\n{cli.stderr}"
    assert JAX_EXTRA_HINT in cli.stdout, cli.stdout

    # and the one it does give, which is why the tests below may use it
    assert got["harness"]["raised"], got["harness"]
    assert got["harness"]["type"] == "stelling._optional.OptionalDependencyError", got
    assert got["harness"]["is_import_error"], got
    assert JAX_EXTRA_HINT in got["harness"]["msg"], got


def test_the_availability_probe_does_not_import_what_it_probes():
    """``available()`` says "without importing it" and nothing pinned that.

    Found by the survivor triage, the second time it has paid: an audit filed
    an import-and-catch reimplementation as an EQUIVALENT mutant because it
    still answers ``True``/``False`` correctly. It is not equivalent, and the
    triage rule — apply the mutant, then look at what the code produces —
    settles it: the answer is identical, the *effect* is not. Import-and-catch
    leaves the probed package in ``sys.modules``, and it makes the divergence
    table in this file's own prelude false, because under the no-jax hook it
    would return False where the shipped probe raises.

    A probe that imports is not a probe: it is the cost the extras exist to
    avoid, paid on every call.

    Break it: implement ``available`` as ``try: import; return True``.
    """
    import importlib.util
    import sys as _sys

    from stelling import _optional

    # z3 rather than jax: it is a real optional dependency, small, and not
    # already imported by the suite. Asserting that first, or the test is
    # measuring a module something else pulled in.
    _sys.modules.pop("z3", None)
    answer = _optional.available("z3")
    assert isinstance(answer, bool), answer
    assert "z3" not in _sys.modules, (
        "available('z3') imported z3. Its docstring says 'without importing "
        "it', and the extras exist so that a bare install pays nothing for a "
        "dependency it does not use."
    )
    # positive control: the probe is looking at something real, not answering
    # False for everything. `sys` is always importable and always a spec.
    assert importlib.util.find_spec("sys") is not None


def test_importing_harness_without_jax_names_the_extra():
    """The property the token scan could not see.

    ``stelling.harness`` is the one module a harness author cannot avoid —
    ``any_array``, ``assert_``, ``assume``, ``nonvacuity`` and ``trace`` all
    live there, and every documented example opens with ``from
    stelling.harness import ...``. With jax absent it must fail with the
    sentence that names the extra, not with a raw ``No module named 'jax'``
    from ``stelling/_jax_compat.py``.
    """
    got = _run_without_jax(
        """
        import os.path, traceback

        out = {}
        for label, stmt in (
            ("module", "import stelling.harness"),
            ("from", "from stelling.harness import any_array"),
            ("names", "from stelling.harness import (any_array, any_pytree, "
                      "assert_, assume, nonvacuity, trace)"),
        ):
            for mod in [m for m in sys.modules if m.startswith("stelling")]:
                del sys.modules[mod]
            try:
                exec(stmt, {})
            except BaseException as e:
                out[label] = {
                    "raised": True,
                    "type": type(e).__module__ + "." + type(e).__qualname__,
                    "is_import_error": isinstance(e, ImportError),
                    "msg": str(e),
                    "frames": [
                        os.path.basename(f.filename)
                        for f in traceback.extract_tb(e.__traceback__)
                    ],
                }
            else:
                out[label] = {"raised": False}
        print(json.dumps(out))
        """
    )

    for label in ("module", "from", "names"):
        detail = got[label]
        assert detail["raised"], f"{label}: stelling.harness imported without jax"
        # the sentence that names the fix
        assert JAX_EXTRA_HINT in detail["msg"], detail
        assert "jax is required for tracing harnesses to jaxprs" in detail["msg"], detail
        # ... and specifically NOT the raw one it used to be
        assert detail["msg"] != "No module named 'jax'", detail
        # still an ImportError, so `except ImportError` in user code and in
        # docs/harness-api.md keeps catching it
        assert detail["is_import_error"], detail
        assert detail["type"] == "stelling._optional.OptionalDependencyError", detail
        # the guard sits at the PUBLIC door, so the traceback stops at the
        # module the user typed: `_jax_compat.py` — the private jax bridge —
        # is never reached and never shown. (A guard only in `_jax_compat`
        # would print the same sentence but hand back a traceback through a
        # file the user has no business reading.)
        assert "harness.py" in detail["frames"], detail
        assert "_jax_compat.py" not in detail["frames"], detail

    # `import stelling.harness` alone fails (the "module" label above), so the
    # requirement is at IMPORT time, and the module docstring has to say so.
    # The whole clause is pinned rather than the substring "at import time",
    # which is also a substring of "does NOT require the extra at import time".
    #
    # WHAT THIS DOES NOT COVER, stated rather than claimed away: pinning a
    # clause cannot stop the docstring contradicting it ELSEWHERE. Measured, a
    # docstring keeping this clause and negating it in the next sentence -- the
    # ordinary shape of a correction record, "an earlier revision said X; that
    # was withdrawn" -- stays green. The bound on the damage is that it is a
    # documentation lie and never a masked regression: making the facade
    # genuinely lazy turns four behavioural tests red whatever the prose says.
    doc = " ".join((SRC / "stelling" / "harness.py").read_text().split())
    claim = "requires the ``[jax]`` extra **at import time**, not at first call"
    assert claim in doc, (
        f"harness.py's docstring must state the measured timing as {claim!r}; "
        "measured here, `import stelling.harness` raises with jax absent. If a "
        "future change makes the façade lazy, the assertions above fail first "
        "and this is the line that has to move with them."
    )


def test_no_public_module_fails_without_naming_the_jax_extra():
    """The generalisation: for EVERY public submodule, importing it with jax
    absent either works or produces the sentence naming the extra.

    ``harness`` is the module that failed this in practice, but nothing stops
    the next module from growing a module-scope ``stelling._jax_compat``
    import — the token scan would stay green for that too, because the token
    is in the module being imported, not the importer.
    """
    got = _run_without_jax(
        """
        import importlib, pkgutil
        import stelling

        names = sorted(
            m.name for m in pkgutil.iter_modules(stelling.__path__)
            if not m.name.startswith("_")
        )
        out = {"scanned": names, "failures": {}}
        for name in names:
            try:
                importlib.import_module("stelling." + name)
            except BaseException as e:
                out["failures"][name] = {
                    "type": type(e).__module__ + "." + type(e).__qualname__,
                    "msg": str(e),
                }
        print(json.dumps(out))
        """
    )

    # a broken enumeration would report zero failures just as loudly
    assert len(got["scanned"]) >= 17, got["scanned"]
    assert "harness" in got["scanned"] and "ir" in got["scanned"], got["scanned"]

    # ... and so would a broken hook, which the inventory cannot see: with jax
    # importable, every module imports, `failures` is empty, `nameless` is
    # empty and this passes having measured nothing. That is what the assertion
    # below is for. Before it existed, removing the hook turned the rest of the
    # suite red while leaving THIS test green; with it, removing the hook takes
    # this test down too.
    # `harness` is the module that cannot import without jax, so its presence
    # here is the evidence that jax really was absent in the subprocess.
    assert "harness" in got["failures"], (
        "no public module failed with jax supposedly blocked — the no-jax hook "
        "is not blocking, so the check below would pass vacuously. Failures "
        f"seen: {sorted(got['failures'])}"
    )

    nameless = {
        name: detail
        for name, detail in got["failures"].items()
        if JAX_EXTRA_HINT not in detail["msg"]
    }
    assert not nameless, (
        "these public modules die in a jax-less environment without naming the "
        f"extra that fixes it (expected {JAX_EXTRA_HINT!r} in the message):\n"
        + "\n".join(f"  stelling.{n}: {d['type']}: {d['msg']}" for n, d in nameless.items())
    )


def test_the_lazy_call_sites_also_name_the_jax_extra():
    """The other door into jax, which importing nothing can protect.

    :mod:`stelling.preconditions` imports jax-free on purpose and reaches
    ``stelling._jax_compat`` inside its functions — so it passes the
    import-time check above and still, before this fix, answered
    ``check(harness, ...)`` with a bare ``No module named 'jax'``. The guard
    therefore sits in ``_jax_compat`` as well as at the public façade, and
    both doors are measured.
    """
    got = _run_without_jax(
        """
        import stelling.preconditions as P

        out = {}
        for label, call in (
            ("check", lambda: P.check(lambda: (), vacuity_mode="inputs-only")),
            ("field_positive", lambda: P.field_positive((2,), "f", (0.0, 1.0))),
            ("private_module", lambda: __import__("stelling._jax_compat")),
        ):
            try:
                call()
            except BaseException as e:
                out[label] = {
                    "raised": True,
                    "type": type(e).__module__ + "." + type(e).__qualname__,
                    "msg": str(e),
                }
            else:
                out[label] = {"raised": False}
        print(json.dumps(out))
        """
    )

    for label in ("check", "field_positive", "private_module"):
        detail = got[label]
        assert detail["raised"], f"{label} succeeded with jax blocked: {detail}"
        assert JAX_EXTRA_HINT in detail["msg"], f"{label}: {detail}"
        assert detail["type"] == "stelling._optional.OptionalDependencyError", (
            label,
            detail,
        )


def test_the_documented_failure_is_the_measured_failure():
    """docs/harness-api.md makes three claims about a JAX-less environment —
    *when* the import fails, *what* it raises, and *what it says* — and this
    compares each against a measurement taken here.

    WHAT IT DOES NOT COVER, stated as a list rather than claimed away, because
    a sentence saying "the page is held to its claims" ranges over every way
    the page could be wrong and cannot be true:

    * **Only one paragraph.** The scan selects the paragraph containing
      ``Importing `stelling.harness```. Another paragraph elsewhere on the page
      may contradict all three claims; measured, that stays green.
    * **Two of the five comparisons are literals typed here**, not
      measurements: the timing clause and the two ``preconditions`` clauses.
      Reword the page past them and they go red loudly, but the page could be
      re-worded *and* falsified together in one edit.
    * **Nothing on this page outside that paragraph** — including the "every
      code block was executed verbatim" claim, which no test enforces.

    Concretely, all three of these have to go red, and none of them touches a
    token the page could keep by accident:

    * reverting the timing to "at first call, not at import time" — the exact
      claim this change corrected;
    * documenting a different exception (``RuntimeError``, say) — the page must
      name the class that was measured and nothing else that looks like a class;
    * rewording ``stelling._optional``'s message, which makes the sentence
      quoted on the page stale.
    """
    got = _run_without_jax(
        """
        out = {}
        try:
            import stelling.harness
        except BaseException as e:
            out["harness_import"] = {
                "raised": True,
                "type": type(e).__module__ + "." + type(e).__qualname__,
                "short_type": type(e).__qualname__,
                "bases": [b.__qualname__ for b in type(e).__mro__[1:]],
                "msg": str(e),
            }
        else:
            out["harness_import"] = {"raised": False}

        try:
            import stelling.preconditions as P
        except BaseException as e:
            out["preconditions_import"] = {"raised": True, "msg": str(e)}
        else:
            out["preconditions_import"] = {"raised": False}
            try:
                P.check(lambda: (), vacuity_mode="inputs-only")
            except BaseException as e:
                out["preconditions_check"] = {
                    "raised": True,
                    "short_type": type(e).__qualname__,
                    "msg": str(e),
                }
            else:
                out["preconditions_check"] = {"raised": False}
        print(json.dumps(out))
        """
    )

    # --- what the environment actually does ---------------------------------
    imported = got["harness_import"]
    assert imported["raised"], "stelling.harness imported with jax absent"
    assert "ImportError" in imported["bases"], imported
    assert got["preconditions_import"] == {"raised": False}, got
    assert got["preconditions_check"]["raised"], got["preconditions_check"]
    assert got["preconditions_check"]["short_type"] == imported["short_type"], got
    assert got["preconditions_check"]["msg"] == imported["msg"], got

    # --- and what the page says it does -------------------------------------
    paragraphs = (REPO / "docs" / "harness-api.md").read_text().split("\n\n")
    hits = [p for p in paragraphs if "Importing `stelling.harness`" in p]
    assert len(hits) == 1, f"expected one JAX-requirement paragraph, found {len(hits)}"
    para = " ".join(hits[0].split())  # the page wraps; the claims do not

    # WHEN. Ordered, so the reversal cannot satisfy it: "at import time" alone
    # is a substring of "at first call, not at import time".
    assert "it fails at import time, not at first call" in para, para

    # WHAT. Every Error-shaped name on the page is one that was measured — the
    # class itself, or a base it really has. Naming any other is the drift.
    named = set(re.findall(r"\b\w+Error\b", para))
    measured = {imported["short_type"], *imported["bases"]}
    assert named <= measured, (
        f"docs/harness-api.md names {sorted(named - measured)}, which is not "
        f"what a JAX-less environment raises ({imported['type']}, bases "
        f"{imported['bases']})"
    )
    assert imported["short_type"] in named, (
        f"the page must name {imported['short_type']}, the class measured here"
    )
    # ...and the RELATION, not just the tokens. Naming both classes while
    # DENYING the subclass relation kept every check above green: measured, a
    # page reading "which is NOT a subclass of `ImportError`, so `except
    # ImportError` will not catch it" passed, fifteen lines below the assertion
    # that ImportError IS in the measured bases. The phrase is derived from the
    # measurement rather than typed, so a base change moves it.
    relation = f"a subclass of `{imported['bases'][0]}`"
    assert relation in para, (
        f"docs/harness-api.md must state the measured relation verbatim: "
        f"{imported['short_type']} is a subclass of {imported['bases'][0]}. "
        f"user code and stelling's own callers rely on `except ImportError`"
    )
    # ...and not while negating it. The affirmative phrase is a SUBSTRING of
    # its own negation ("NOT a subclass of `ImportError`"), so the presence
    # check alone passed a page saying the opposite -- the same substring
    # defect this assertion was added to close, reproduced inside the fix for
    # it. Caught by mutating the fix rather than by reading it.
    negated = re.search(
        r"(?:\bnot\b|n't|\bnever\b)\s+" + re.escape(relation), para, re.I
    )
    assert not negated, (
        f"docs/harness-api.md names the measured relation and denies it: "
        f"{negated.group(0)!r}. Measured, {imported['short_type']} IS a "
        f"subclass of {imported['bases'][0]}."
    )

    # WHAT IT SAYS. The page quotes the sentence, not just the install line, so
    # rewording _optional's purpose or hint leaves the page stale — here.
    assert imported["msg"] in para, (
        f"docs/harness-api.md must quote the message as emitted:\n"
        f"  emitted: {imported['msg']}\n  page:    {para}"
    )
    assert JAX_EXTRA_HINT in imported["msg"], imported

    # And the second half of the paragraph, which is about the lazy door.
    assert "`stelling.preconditions` does not need JAX to import" in para, para
    assert "calling `check` in a JAX-less environment raises the same error" in para, para
