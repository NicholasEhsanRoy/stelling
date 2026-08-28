# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A name a `src/` docstring CROSS-REFERENCES, that `src/` cannot resolve.

`tests/test_documented_names_exist.py` is the sibling of this file and the
reason it exists. That gate's subject is *"a name a document presents as
EXISTING, that `src/` does not have"*, and it reads the project's `.md`
prose. It cannot see a stale reference written in a source docstring, and it
would not judge one the same way if it could — so **eleven references to
three names that do not exist** stood in this repository, and no gate could
reach any of them.

**THIS SAID "NINE", AND NINE WAS THE COUNT OF WHAT THIS FILE CAN SEE RATHER
THAN OF WHAT WAS THERE** — the 0.2.1 audit found the other two. They are in
`SOUNDNESS.md`, a tracked, shipped page and an sdist member, and they are
outside both gates by construction: this one reads `src/` and resolves
Sphinx roles, and `SOUNDNESS.md`'s two are plain backticks. **A census taken
with one instrument and reported as a census of the tree is the defect this
branch is about, one level up**, and it is worth more than the two sites it
undercounted.

**WHAT WAS THERE.** Measured at `9b5b496` on 2026-08-28 by counting every
occurrence of the three names in every tracked file, and staged by whether
the occurrence is a Sphinx role or a plain backtick:

    src/                     8   (6 roles, 2 plain)
    SOUNDNESS.md             2   (0 roles, 2 plain)
    tests/                   1   (0 roles, 1 plain)
                            ──
                            11   (6 roles, 5 plain)

Six of the eight in `src/` are in `src/stelling/propagate.py`, the file
whose subject is a soundness certificate. Per name: `_classify_cmp` eight,
`_Walker` two, `ir.JaxprEqn.from_dict` one.

* `_classify_cmp`, eight sites — four ``:meth:`` roles in
  `src/stelling/propagate.py`, a fifth occurrence in a plain backticked
  comment there, a sixth in `tests/test_strict_sign_algebra.py`, and TWO IN
  `SOUNDNESS.md`, both plain backticks and both present-tense claims about
  the propagator. **`git log --all --oneline -G '^[[:space:]]*def
  _classify_cmp' -- src/` returns nothing**: the name has never been defined
  in this repository. (That instruction read `-S "def _classify_cmp"` until
  the 0.2.1 re-audit, and the commit that wrote the sentence falsified it —
  :func:`definition_search` carries the mechanism and the repair.) The
  method those eight sentences describe is
  `_Propagator._classify_assumed_pred`, added in `8106a55` (2026-08-07);
  the first of the eight landed the same day, in `f116890`. The
  `SOUNDNESS.md` two were repaired one release later, in 0.2.1's fix round,
  and that page's entry carries the record.
* ``:meth:`_Walker._conjunct_certainly_true` `` in `propagate.py`, and
  `_Walker` again in a plain comment in `src/stelling/obligation.py`. **The
  same anchored search on `class _Walker` also returns nothing**, while on
  `class _Propagator` it returns `4f25390` — the control that makes the
  zero a reading. The class is `_Propagator` and has been since that MVP
  commit (2026-07-17); `_Walker` first appears in prose in `0874dd1`
  (2026-08-14).
* ``:meth:`ir.JaxprEqn.from_dict` `` in `src/stelling/coverage.py`, carrying
  an argument about what is reachable from a real query. The only `from_dict`
  in the library is `ir.ClosedJaxpr.from_dict`.

None of the three was ever a rename. A rename leaves a trail a `grep` can
follow and a reader can date; these were written down as if they existed,
which is exactly the defect `test_documented_names_exist.py` was built for —
one directory over, in a construction that file does not lint.

**WHY THIS IS A NEW MODULE AND NOT AN EXTENSION OF THAT ONE.** The two gates
share a subject and share nothing else, and merging them would mean holding
three contradictory rules in one file:

* That file lints ENGLISH. Its whole design problem is telling a plan from a
  claim, which is why it lints four indicative constructions and excludes
  fenced blocks and unticked checkboxes. A Sphinx role is not English: it is
  a machine-readable assertion with one meaning, so this file has no
  false-positive problem of that kind and needs none of those exclusions.
* That file's test of "present" is **a substring of the concatenated `src/`
  text**, deliberately the weak direction, because its subject is names that
  are NOWHERE. Every reference repaired here would pass that test: the string
  `_classify_cmp` was in `src/` — in the very sentences that were wrong.
  Resolution here is a symbol lookup in a parsed tree, which is the strong
  direction and the only one that can decide a cross-reference.
* That file **excludes dotted names by construction** (*"a qualified
  reference names somebody else's namespace, and prose does not reliably say
  whose"*). This file's hardest case is exactly a dotted one —
  `ir.JaxprEqn.from_dict` — because inside `src/` the namespace IS knowable:
  `ir` is a name the referring module imported and the tree says what it is.

## The construction

A Sphinx cross-reference role — ``:func:``, ``:meth:``, ``:class:``,
``:data:``, ``:attr:``, ``:exc:``, ``:mod:``, ``:obj:``, ``:const:`` — is an
assertion that a Python object of that name exists, addressed to a reader who
will go looking for it. :data:`PYTHON_ROLES` is those nine and not a
selection from them: they are Sphinx's python domain in full, so the tuple is
a definition rather than an enumeration this repository has to keep up to
date. Every role written anywhere in `src/**/*.py` is resolved against the
tree — docstring, comment or message string alike, because the role asserts
the same thing wherever it is written and a reader follows it the same way.

A target resolves if, read from the module it is written in:

1. it is a name that module BINDS — a `def`, a `class`, an `import`, a
   module- or class-level assignment, or a ``self.x`` store — or the final
   component of one of that module's attribute paths, which is how
   ``:meth:`slice` `` in `src/stelling/obligation.py` resolves to
   `_Slicer.slice`;
2. it names a module of this package, by its full dotted path
   (``:mod:`stelling.solvers` ``) or by its short name (``:mod:`eager` ``);
3. it is dotted and its head is one of: an import alias in the referring
   module, a module short name, or a class the module defines — in which case
   the REMAINDER must be an attribute path of whatever the head names.

## What this does NOT reach, so nobody reads a green as more than it is

* **A bare name that exists in a DIFFERENT module does not resolve here**,
  and that is a decision rather than an oversight. Sphinx resolves a bare
  target in the current module; a reader does the same; so a reference that
  needs another module's namespace does not resolve from where it stands even
  when the name it wants is real. Three were found and all three are now
  qualified. The cost is a class of red this gate will produce again, and the
  author's answer is to write the module in — which is what the neighbouring
  sentences in each of those three files already did.
* **A dotted reference into a namespace this tree does not own is out of
  scope**, and it cannot be otherwise without importing, which would make
  this check report a different truth on a lane with jax and a lane without —
  the environment-dependence class that put four of five consecutive reds on
  `main` in the week `tests/_repo_files.py` was rewritten. Two static
  discriminators stand in, both properties of the language rather than of
  this machine: :data:`sys.stdlib_module_names` for a stdlib head, and *"the
  referring module imported it"* for anything else. So ``:class:`
  fractions.Fraction` `` is out of scope, and so is ``:func:`jax.jit` `` in a
  module that imports jax — but a third-party name in a module that does NOT
  import it is a RED here, with nothing to tell it from a typo. The remedy
  then is the one the surrounding prose already uses for foreign APIs: write
  it as a ``literal``, not as a cross-reference.
* **A name that exists but is not the one the sentence means.** This decides
  existence, never aptness. ``:meth:`slice` `` would resolve just as happily
  if the sentence were about some other slicer.
* **The plain backticked identifier**, which is how five of the eleven
  stale references above were written (`` `_classify_cmp` `` in a comment in
  `propagate.py`, `` `_Walker._classify_assumed_pred` `` in one in
  `obligation.py`, `` `_classify_cmp` `` again in
  `tests/test_strict_sign_algebra.py`, and twice more in `SOUNDNESS.md`).
  A gate over every backticked word in
  `src/` would report parameter names, jax primitive names, SMT-LIB
  operators, shell fragments and English words in emphasis; that is the
  false-positive rate the sibling file's docstring says kills a lint. Those
  three were found by reading, and are repaired, and nothing here will catch
  the next one.
* **`:ref:` and `:doc:`** — see :data:`NON_PYTHON_ROLES`.
* **Builtins**, which are resolved as present without asking whether a reader
  could find them in this tree. That is right — Sphinx's python domain
  resolves them — and it is the one place a Python VERSION moves this gate: a
  name that becomes a builtin stops being checked here.

## The hole this branch OPENED in the sibling gate, and why it stays open

`tests/test_documented_names_exist.py` decides "present in `src/`" by
substring over the concatenated source text, comments and docstrings
included. **The correction records this branch wrote keep `_classify_cmp`
and `_Walker` in that text**, so a document that asserted either name would
now be told it exists — by the very sentences that say it does not. A gate
whose presence test is satisfied by a record of absence has stopped looking
for that name, and this branch is what put it there.

**It is not closed, and the obvious way to close it is refused on a
measurement rather than on a preference.** The strengthening would be to
resolve an asserted name as a SYMBOL — this file already builds that table,
and the 0.2.1 audit checked it against the imported runtime package. Run
over the sibling's own corpus on this tree with :func:`resolves_as_a_symbol`,
which is what :func:`resolve` grants a bare target:

    asserted-name rows            64,  2 of them PERMITTED,  62 checked
    findings after the swap       39   (28 pairs, 25 names, 14 documents)

and the names are `real`, `ieee`, `int8`, `float32`, `bool`, `UNCHECKED`,
`vacuity_mode`, `fill_value`, `precision`, `arithmetic`, `stability`:
verdict VALUES and record KEYS, which are strings in this project and
symbols in no tree. **Sixty-three per cent of the rows, and a lint that
reports two in three is a lint that is switched off** — which is exactly
the false-positive rate the sibling's docstring predicts, now re-derived
instead of believed. A derived rule — *"in the source text, not a symbol,
and asserted by a document"* — selects the same 39 rows and fails the same
way.

**THREE NUMBERS WERE CITED FOR THAT ONE MEASUREMENT AND THE 0.2.1 RE-AUDIT
WAS RIGHT TO SAY SO.** This paragraph read *"39 … 28 distinct pairs"* while
:func:`test_the_sibling_gates_presence_test_is_satisfied_by_a_record_of_absence`
read *"28 of 62 rows"* — the PAIRS figure quoted as a row count — and the
re-audit re-measured **40**. All three are arithmetic on the same corpus and
the spread had two separate causes, both worth more than the numeral:

* 39 against 40 is **one row**, `design/portability-pass.md`'s `_jax_compat`,
  and it is a disagreement about the QUESTION rather than about the tree.
  The narrow reading — bound in a module, or a member of a class in one —
  calls it absent; `_jax_compat` is a MODULE of this package, which
  :func:`resolve` grants a bare target and the narrow reading does not.
  This file was measuring with one notion and resolving with the other.
  There is now one predicate, :func:`resolves_as_a_symbol`, and both read
  it; the narrow reading still gives 40 and the difference is that row.
* 28 for 39 was a pairs figure read as a row count, and it survived
  because the two sentences were written in different places at different
  times with no derivation between them.

**Both are the shape this file exists to catch**, in the file itself: a
figure that moves with the tree, stated twice, in two spellings, with
nothing deriving either.

**So the limit is declared and DRIVEN instead**, by
:func:`test_the_sibling_gates_presence_test_is_satisfied_by_a_record_of_absence`,
which plants each name in the sibling's own constructions, runs its own
`missing_names`, and asserts it reports nothing — with a control that a name
absent from the text is still caught. Two facts that test also holds, and
either can go red: the names are still not symbols (if one becomes one, the
entry is dead), and **no document in the sibling's corpus asserts either of
them today**, so the hole is LATENT rather than live. That last assertion is
the alarm the sibling gate cannot raise for itself.

## Why `src/` and not `tests/`

`tests/test_strict_sign_algebra.py` carried one of the eleven stale
references, so the answer is not automatic. It is still `src/` only, and the
reason is about the construction rather than about the directory:

* This resolver is a static model of Python name resolution, and over `src/`
  the model is exact — one import-rooted package, every module reachable by a
  dotted path from `stelling`, so *"the name the role means"* and *"the name
  this file looks for"* are the same object. `tests/` has no import root:
  pytest puts each file's own directory on `sys.path` and imports by
  basename, `conftest.py` contributes names by protocol rather than by
  import, and a fixture is bound by name at call time with no definition site
  a parser can see. Resolving roles there would mean modelling pytest's name
  binding, and *a check that models a behaviour is one indirection behind
  it*.
* And it would not have caught the instance that raises the question anyway:
  `test_strict_sign_algebra.py`'s is a plain backticked comment, not a role.
  What would catch that one is the construction two bullets above, which this
  file declines for its false-positive rate. Widening the DIRECTORY would
  have answered a question nobody asked.

(For whoever merges: another branch is changing whether `tests/` sits inside
a different prose gate. Nothing here depends on that outcome — this file
reads `src/` and no other tree.)
"""

from __future__ import annotations

import ast
import builtins
import collections
import dataclasses
import functools
import importlib
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _repo_files import is_skipped  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src"

#: Sphinx's python domain, in full. Not a selection from it: these are the
#: nine roles `sphinx.domains.python.PythonDomain` defines, so this tuple is
#: a definition rather than a list this repository has to keep up to date.
PYTHON_ROLES = (
    "attr", "class", "const", "data", "exc", "func", "meth", "mod", "obj",
)

#: Roles whose targets are NOT Python names, with the reason each is out of
#: scope. `test_every_role_in_src_is_classified` holds the partition — a role
#: name in neither set goes red rather than being silently skipped, which is
#: how a lint stops covering a construction without anyone deciding to — and
#: holds the two sets DISJOINT, which is the anti-vacuity half: moving
#: `func` into this dict would silence 864 of the 1530 references in the tree
#: and nothing else would notice.
NON_PYTHON_ROLES = {
    "ref": "a label on a section of a document; `docs/` is not a namespace "
           "this file can resolve a name in, and the sibling gate "
           "`tests/test_documented_names_exist.py` is what reads that tree",
    "doc": "a path to a document, resolved against the documentation source "
           "root and never against any Python module",
}

@dataclasses.dataclass(frozen=True)
class Absent:
    """One name this repository records as never having been defined."""

    #: `def` or `class` — what the archaeology anchors on, and what makes
    #: the search a question about DEFINITIONS rather than about a string.
    declarator: str
    #: The name the sentences were reaching for. Its own search is the
    #: CONTROL: the same instrument, and it must find something.
    instead: str
    why: str


#: The names this repository has RECORDED AS NEVER DEFINED, and which its own
#: records therefore keep present in `src/` as text. Two, and the table is
#: prunable rather than a list nobody revisits —
#: :func:`test_the_sibling_gates_presence_test_is_satisfied_by_a_record_of_absence`
#: fails if a name here becomes a symbol, if the record that mentions it is
#: deleted, or if a document in the sibling gate's corpus starts asserting
#: it, and
#: :func:`test_the_archaeology_this_file_hands_a_reader_is_still_anchored`
#: fails if this tree ever grows a real definition of one. It is not an
#: allowlist: nothing here is excused, and the entries exist so that the
#: blindness they describe is measured on every run.
NEVER_DEFINED = {
    "_classify_cmp": Absent(
        "def", "_classify_assumed_pred",
        "the name eight sentences gave `_Propagator._classify_assumed_pred`; "
        "the correction record above `class _Propagator` is what keeps the "
        "string in `src/`, and is why the search has to be anchored",
    ),
    "_Walker": Absent(
        "class", "_Propagator",
        "the name two sentences gave `_Propagator`; the same correction "
        "record quotes it, and the same anchoring applies",
    ),
}


def definition_search(declaration: str) -> tuple[list[str], str]:
    """The archaeology — as argv, and as a reader would type it, from ONE source.

    **THIS FILE HANDED A READER `-S` AND THE COMMIT THAT WROTE THE SENTENCE
    FALSIFIED IT.** `git log --all -S "def _classify_cmp" -- src/` counts
    occurrences of a STRING, and 0.2.1's correction record quotes the name
    inside `src/stelling/propagate.py`, so from that commit onward the
    command returns `6985594` — the record itself — while four documents
    said it returned nothing. The finding was never in doubt (what `-S`
    reports is a record, not a definition); the INSTRUCTION was.

    That is this repository's recurring shape — *an instrument whose corpus
    includes the prose about the instrument* — arriving in the archaeology
    the whole never-a-rename argument rests on, one release after the same
    shape was disclosed here about the sibling gate's substring test.

    `-G` matches the regex against the diff's own lines, so anchoring at
    line start asks about a DEFINITION. Every mention this repository has is
    inside a comment or a quoted command and cannot match. **What it does
    not survive**: a fenced example that writes `def _classify_cmp(` at
    column zero. Nothing here would catch that in history — which is why
    :func:`test_the_archaeology_this_file_hands_a_reader_is_still_anchored`
    checks the tree it can see, at every revision the suite runs on.
    """
    argv = [
        "git", "log", "--all", "--oneline",
        "-G", f"^[[:space:]]*{declaration}", "--", "src/",
    ]
    shown = " ".join(f"'{part}'" if part.startswith("^") else part for part in argv)
    return argv, shown


#: The same anchor as :func:`definition_search`'s, for Python's engine rather
#: than git's. `[[:space:]]` and `[ \t]` are the same class at the start of a
#: Python source line, which is the only place either is applied.
_DECLARED_HERE = r"^[ \t]*{}\b"

#: A dotted path of Python identifiers, which is the only target shape this
#: file can decide. Anything else — a space, an angle bracket, a slash, a
#: hyphen — is a target of some other kind and is reported as out of scope
#: rather than guessed at.
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")

#: Any role, not a list of the roles this file happens to know: the role name
#: is captured so the partition above can be held over what the tree actually
#: writes.
#:
#: **THE TARGET MAY BE WRAPPED, AND A LINE-BY-LINE SCAN IS BLIND TO IT.**
#: Measured at `9b5b496`: of the 1530 roles in `src/`, TWELVE are split
#: across two lines mid-target (``:func:`stelling.propagate.<newline>
#: propagate` ``), and one of those twelve is split mid-IDENTIFIER
#: (``slice_unknown_<newline>    obligations``) — so a per-line scan reads
#: 1518 and says nothing at all about the twelve it cannot see. This scans
#: whole file text and collapses the target's internal whitespace instead.
_ROLE = re.compile(r":([a-z]+):`([^`]*)`")


@dataclasses.dataclass(frozen=True)
class Reference:
    """One cross-reference role, where it is written and what it names."""

    path: str
    line: int
    role: str
    target: str

    def __str__(self) -> str:  # pragma: no cover - failure text only
        return f"{self.path}:{self.line}  :{self.role}:`{self.target}`"


@dataclasses.dataclass
class Module:
    """One module's namespace, as much of it as a parser can see.

    `attributes` are the paths reachable by dotted access from the module
    object (`ClosedJaxpr.from_dict`, `_Slicer.slice`, `MIN_NORMAL`).

    `bound` is every name the module introduces anywhere, INCLUDING inside a
    function body: a nested `def` is not reachable as an attribute but it is
    a name a bare role can legitimately mean, and `stelling.solvers` writes
    one (``:func:`degraded_clause` ``). Ordinary local variables are not
    collected — a binding a reader cannot navigate to should not license a
    cross-reference to it.

    `aliases` maps each imported local name to the dotted path it was
    imported from, which is what makes `ir.JaxprEqn.from_dict` decidable at
    all.
    """

    attributes: set[str] = dataclasses.field(default_factory=set)
    bound: set[str] = dataclasses.field(default_factory=set)
    aliases: dict[str, str] = dataclasses.field(default_factory=dict)


def source_files() -> list[tuple[str, pathlib.Path]]:
    """`(relative posix path, path)` for every Python file under `src/`.

    Filtered through `_repo_files.is_skipped` for the reason that module
    carries: a checkout that grew a virtualenv with a copy of the package in
    it must not change what this check reads, or it reports a different truth
    to different people.
    """
    out = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if not is_skipped(rel):
            out.append((rel, path))
    return out


def module_name(path: pathlib.Path) -> str:
    """The dotted import path of a file under `src/`."""
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _read_module(path: pathlib.Path) -> Module:
    """Everything the parser can say about one file's namespace."""
    module = Module()
    dotted = module_name(path)
    package = dotted if path.name == "__init__.py" else dotted.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def note_import(node: ast.AST, prefix: str, reachable: bool) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.partition(".")[0]
                module.aliases[local] = (
                    alias.name if alias.asname else alias.name.partition(".")[0]
                )
                module.bound.add(local)
                if reachable:
                    module.attributes.add(prefix + local)
            return
        # `from . import x` and `from .report import y`: the dots are resolved
        # against this file's own package, so a relative import names the same
        # module a dotted role would.
        if node.level:
            base = package
            for _ in range(node.level - 1):
                base = base.rpartition(".")[0]
            base = f"{base}.{node.module}" if node.module else base
        else:
            base = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            module.aliases[local] = f"{base}.{alias.name}" if base else alias.name
            module.bound.add(local)
            if reachable:
                module.attributes.add(prefix + local)

    def walk(node: ast.AST, prefix: str, reachable: bool, cls: str) -> None:
        """`reachable` = still on a path of attribute access from the module.

        A `def` inside a `def` is `bound` but is not an attribute; a `class`
        inside a `class` is both. `cls` carries the enclosing class down so a
        ``self.x`` store inside a method becomes that class's attribute —
        `_Slicer.const_avals` and `_Slicer.producers` are referenced by
        `:attr:` and exist only that way.
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module.bound.add(child.name)
                if reachable:
                    module.attributes.add(prefix + child.name)
                walk(child, "", False, cls)
            elif isinstance(child, ast.ClassDef):
                module.bound.add(child.name)
                if reachable:
                    module.attributes.add(prefix + child.name)
                    walk(child, f"{prefix}{child.name}.", True, prefix + child.name)
                else:
                    walk(child, "", False, cls)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = (
                    child.targets if isinstance(child, ast.Assign) else [child.target]
                )
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name) and reachable:
                            module.bound.add(name.id)
                            module.attributes.add(prefix + name.id)
                        elif (
                            isinstance(name, ast.Attribute)
                            and isinstance(name.value, ast.Name)
                            and name.value.id == "self"
                        ):
                            module.bound.add(name.attr)
                            if cls:
                                module.attributes.add(f"{cls}.{name.attr}")
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                note_import(child, prefix, reachable)
            elif isinstance(
                child,
                (ast.If, ast.Try, ast.With, ast.For, ast.While, ast.ExceptHandler),
            ):
                walk(child, prefix, reachable, cls)
            else:
                walk(child, prefix, False, cls)

    walk(tree, "", True, "")
    return module


@functools.lru_cache(maxsize=1)
def namespace() -> dict[str, Module]:
    """The whole package's namespaces, keyed by dotted module name.

    Cached because six tests here ask for it and the parse is of every line
    of `src/`; the tree does not change inside a session.
    """
    return {module_name(path): _read_module(path) for _rel, path in source_files()}


def short_names_of(table: dict[str, Module]) -> dict[str, tuple[str, ...]]:
    """Final component -> the module(s) that spell it that way, for `table`.

    A tuple of modules and not one module: `test_module_short_names_are_unique`
    is what licenses route 2 of the construction, and it fails rather than
    picking one if this tree ever grows two modules with the same short name.

    Table-driven rather than reading :func:`namespace` itself, because
    :func:`resolve` is handed a table and a cached view of a DIFFERENT table
    is two answers to one question — which is the defect the three predicates
    below exist to prevent.
    """
    out: dict[str, list[str]] = collections.defaultdict(list)
    for dotted in table:
        out[dotted.rpartition(".")[-1]].append(dotted)
    return {short: tuple(sorted(mods)) for short, mods in out.items()}


@functools.lru_cache(maxsize=1)
def short_names() -> dict[str, tuple[str, ...]]:
    """:func:`short_names_of` over this tree, cached."""
    return short_names_of(namespace())


def binds(module: Module, name: str) -> bool:
    """Does this module introduce `name` — as a binding, or as a class member?

    One of THREE PREDICATES THAT ARE THE FILE'S ONLY SPELLINGS OF "resolves
    as a symbol", and they are separate functions because they were not.
    The 0.2.1 re-audit found this file measuring its own refused
    strengthening with `binds`-plus-attributes while :func:`resolve` also
    grants a bare name on module-hood — one row apart on this tree
    (`_jax_compat`, a module of this package), and two answers to one
    question in one file. :func:`resolve` and
    :func:`test_the_sibling_gates_presence_test_is_satisfied_by_a_record_of_absence`
    now read the same three.
    """
    return name in module.bound or any(
        path.rpartition(".")[-1] == name for path in module.attributes
    )


def names_a_module(name: str, table: dict[str, Module]) -> bool:
    """Is `name` a module of this package — full dotted path, or short name?"""
    return name in table or len(short_names_of(table).get(name, ())) == 1


def resolves_as_a_symbol(name: str, table: dict[str, Module]) -> bool:
    """Is `name` reachable in this package AT ALL, from anywhere?

    The union of the two above, and the WIDE reading deliberately: it is
    what :func:`resolve` grants a bare target, minus the builtin escape and
    minus the own-module-first rule that :func:`resolve` adds on top. A
    question about a name and the tree, never about a name and a place.
    """
    return any(binds(module, name) for module in table.values()) or names_a_module(
        name, table
    )


def references(text: str, path: str) -> list[Reference]:
    """Every role in one file, with its target's whitespace collapsed."""
    found = []
    for hit in _ROLE.finditer(text):
        found.append(
            Reference(
                path=path,
                line=text.count("\n", 0, hit.start()) + 1,
                role=hit.group(1),
                target=re.sub(r"\s+", "", hit.group(2)).lstrip("~"),
            )
        )
    return found


def resolve(ref: Reference, here: str, table: dict[str, Module]) -> str | None:
    """`None` if the reference resolves; otherwise WHY it does not.

    The verdict string is the failure message a reader acts on, so it names
    the namespace that was searched rather than saying "not found".
    """
    if ref.role not in PYTHON_ROLES:
        return None
    if not _DOTTED.match(ref.target):
        return None
    short = short_names_of(table)
    module = table[here]
    parts = ref.target.split(".")

    def attribute_of(dotted: str, rest: str) -> str | None:
        if rest in table[dotted].attributes:
            return None
        return f"`{dotted}` has no `{rest}`"

    if len(parts) == 1:
        name = parts[0]
        if binds(module, name):
            return None
        if names_a_module(name, table):
            return None
        # Before the search below, not after: a builtin is resolvable from
        # every module, so reporting one as "it is in `stelling.obligation`"
        # (which `slice` is) would send a reader after the wrong object.
        if name in dir(builtins):
            return None
        elsewhere = sorted(
            dotted for dotted, other in table.items() if binds(other, name)
        )
        if elsewhere:
            return (
                f"`{here}` does not bind `{name}`; it is in "
                f"{', '.join(f'`{m}`' for m in elsewhere)} — write the module "
                f"in, the way a reader following the reference has to"
            )
        return f"nothing in `src/` defines `{name}`"

    # Dotted. The head decides which namespace the remainder is read in, and
    # the order is most-specific-first: a full module path, then a name this
    # module imported, then a sibling module's short name, then a class this
    # module defines.
    for cut in range(len(parts), 0, -1):
        candidate = ".".join(parts[:cut])
        if candidate in table:
            rest = ".".join(parts[cut:])
            return None if not rest else attribute_of(candidate, rest)
    head, rest = parts[0], ".".join(parts[1:])
    aliased = module.aliases.get(head)
    if aliased is not None:
        if aliased in table:
            return attribute_of(aliased, rest)
        owner, _, leaf = aliased.rpartition(".")
        if owner in table:
            # An alias of an alias: `from stelling._jax_compat import jax`
            # re-exports something THAT module imported from outside `src/`,
            # so the chain leaves this tree here and the reference does too.
            chased = table[owner].aliases.get(leaf)
            if chased is not None and chased not in table and (
                chased.rpartition(".")[0] not in table
            ):
                return None
            return attribute_of(owner, f"{leaf}.{rest}")
        return None  # imported from outside `src/`: somebody else's namespace
    if len(short.get(head, ())) == 1:
        return attribute_of(short[head][0], rest)
    if head in module.attributes:
        return attribute_of(here, ref.target)
    if head in sys.stdlib_module_names:
        return None
    return (
        f"`{here}` neither defines nor imports `{head}`, no module of this "
        f"package is called that, and the standard library has no such "
        f"module — so `{ref.target}` names nothing a reader can reach"
    )


def unresolved() -> list[tuple[Reference, str]]:
    """The census: every role in `src/` that does not resolve, and why."""
    table = namespace()
    out = []
    for rel, path in source_files():
        here = module_name(path)
        for ref in references(path.read_text(encoding="utf-8"), rel):
            why = resolve(ref, here, table)
            if why is not None:
                out.append((ref, why))
    return out


def test_every_cross_reference_role_in_src_resolves():
    """The gate.

    **RED at `9b5b496`: ten references — seven naming something that is not
    there, and three bare names written from a module that does not bind
    them.** A figure a reader cannot re-derive is the same defect as a check
    that does not exist, so: copy this module into a worktree at `9b5b496`
    and run it, and the message below prints all ten with their files, lines
    and reasons. What they were:

    * four ``:meth:`_classify_cmp` `` in `src/stelling/propagate.py` and one
      ``:meth:`_Walker._conjunct_certainly_true` `` — two names that have
      never been defined in this repository, as the module docstring
      measures;
    * ``:meth:`ir.JaxprEqn.from_dict` `` in `src/stelling/coverage.py`,
      holding up an argument about what a real query can produce;
    * ``:meth:`is_registered` `` in `src/stelling/overflow.py` — pytest's
      `PluginManager` method, which this gate is right to refuse: it does not
      resolve from that module, and the paragraph around it writes every
      other foreign name as a ``literal``;
    * three bare names written from a module that does not bind them —
      ``:meth:`_declared_shape` `` in `ir.py`, ``:exc:`
      EmissionInfidelityError` `` in `obligation.py`, and ``:class:`
      IntervalError` `` in `propagate.py`, the last of which was the only
      bare one in a file carrying thirteen `interval.`-headed roles, two of
      them naming `IntervalError` itself.
    """
    census = unresolved()
    rendered = "\n  ".join(f"{ref} — {why}" for ref, why in census)
    assert not census, (
        f"{len(census)} cross-reference role(s) in `src/` do not resolve "
        f"against the tree:\n  {rendered}\n\n"
        f"A Sphinx role is an assertion that a Python object of that name "
        f"exists, addressed to a reader who will go looking for it. Three "
        f"things it can be: the name is WRONG, and the sentence needs the "
        f"real one; the name is RIGHT but lives in another module, and the "
        f"reference needs qualifying so that it resolves from where it is "
        f"written; or the name is somebody else's API that this module does "
        f"not import, in which case it is not a cross-reference at all and "
        f"belongs in ``double backticks``. There is no allowlist here on "
        f"purpose — each of those three has a repair that makes the sentence "
        f"more true, and an allowlist entry would make it permanently less."
    )


def test_each_resolution_route_is_driven():
    """Each route, observed to resolve and observed to refuse.

    *A guard never observed to fire is not known to be a guard*, and a
    resolver is where that bites hardest: a route that quietly started
    accepting everything — a set that filled up, a regex that stopped
    anchoring — would keep the gate above green while checking nothing. So
    every route is driven on a real name from this tree AND on a planted
    absent one beside it, which is the half that can go red.
    """
    table = namespace()
    absent = "a_name_that_is_definitely_not_in_this_package"
    routes = {
        # label: (role, a target that resolves, the module it is read from)
        "bare name bound here": ("func", "unpaired_propagation", "stelling.propagate"),
        "method of a class here": ("meth", "slice", "stelling.obligation"),
        "class attribute path": ("attr", "Propagation.top_boxes", "stelling.propagate"),
        "full dotted module": ("mod", "stelling.solvers", "stelling.verdict"),
        "name in a dotted module": (
            "func", "stelling.verdict.make_verdict", "stelling.propagate",
        ),
        "module short name": ("mod", "eager", "stelling._tripwire._adapter_jax"),
        "name via a short name": (
            "func", "report.render_status", "stelling._tripwire.record",
        ),
        "name via an import alias": ("class", "ir.ClosedJaxpr", "stelling.coverage"),
    }
    for label, (role, target, here) in routes.items():
        why = resolve(Reference("<plant>", 1, role, target), here, table)
        assert why is None, (
            f"the {label!r} route no longer resolves a real name: "
            f":{role}:`{target}` read from {here!r} was refused with {why!r}. "
            f"Either the name moved, or the route has gone inert."
        )
        # ... and the same route with its last component replaced by a name
        # that is nowhere. A route that cannot refuse is not a route.
        head = target.rpartition(".")[0]
        broken = f"{head}.{absent}" if head else absent
        refused = resolve(Reference("<plant>", 1, role, broken), here, table)
        assert refused is not None, (
            f"the {label!r} route ACCEPTED {broken!r} read from {here!r}. A "
            f"route with no absence half can never go red, and this gate's "
            f"green would then mean nothing."
        )


def test_each_out_of_scope_rule_is_driven():
    """The four escapes, each observed to fire — and observed to be narrow.

    An escape nobody drives is a hole nobody measured, and these four are
    where a stale reference would hide if one of them widened.
    """
    table = namespace()
    absent = "a_name_that_is_definitely_not_in_this_package"

    def verdict(role: str, target: str, module: str = "stelling.propagate"):
        return resolve(Reference("<plant>", 1, role, target), module, table)

    assert verdict("ref", absent) is None, (
        "a `:ref:` fired this gate. A document label is not a Python name, "
        "and resolving one here would report every section heading as missing."
    )
    assert verdict("func", "theblindspot<blind-spot>") is None, (
        "a target that is not a dotted identifier fired this gate. This file "
        "can decide names and nothing else; guessing at the rest is how a "
        "lint starts reporting things nobody can act on."
    )
    assert verdict("class", "ValueError") is None, (
        "a builtin did not resolve. Sphinx's python domain resolves builtins, "
        "so a red here would be this gate reporting the language as missing."
    )
    assert verdict("class", "fractions.Fraction") is None, (
        "a stdlib-headed reference fired this gate. Deciding it would mean "
        "importing, and a check that imports reports a different truth on a "
        "lane with the dependency and a lane without."
    )

    # THE ASYMMETRIES, which are what keep the four escapes from being a way
    # out of the gate.
    #
    # 1. The builtin excuse is consulted only for a BARE name, and only after
    #    the referring module has been searched, so `slice` is `_Slicer.slice`
    #    where that class lives and the builtin type everywhere else — never a
    #    report sending a reader into `stelling.obligation` for the builtin.
    assert verdict("meth", "slice", "stelling.obligation") is None
    assert verdict("meth", "slice", "stelling.smt") is None
    # 2. A head that is neither a module of this package, nor imported, nor a
    #    class defined here, nor stdlib is REFUSED and not waved through as
    #    somebody else's namespace. That is the exact shape
    #    `_Walker._conjunct_certainly_true` had, and it is why nothing could
    #    see it: an "unknown namespace" escape wide enough to cover a stdlib
    #    module is wide enough to cover a class that was never written.
    assert verdict("meth", f"{absent}.a_method") is not None, (
        "a dotted reference whose head this tree neither defines nor imports "
        "was let through. `_Walker._conjunct_certainly_true` is that shape."
    )


def test_module_short_names_are_unique():
    """Route 2 is only decidable while this holds.

    ``:mod:`eager` `` and ``:func:`report.render_status` `` are resolved by
    matching the LAST component of a module path, which is unambiguous only
    while no two modules share one. Two `ir.py` under `src/` would make every
    short-name reference in the tree silently pick whichever the walk found
    first — an instrument answering a question it can no longer decide. This
    fails instead, so that somebody chooses.
    """
    collisions = {
        short: mods for short, mods in short_names().items() if len(mods) > 1
    }
    assert not collisions, (
        f"two or more modules under `src/` share a short name: {collisions}. "
        f"Short-name cross-references cannot be resolved while that is true: "
        f"either rename one, or take route 2 out of `resolve` and make every "
        f"such reference fully dotted."
    )


def test_every_role_in_src_is_classified():
    """The role vocabulary is a PARTITION, not the roles this file knows.

    A role name in neither :data:`PYTHON_ROLES` nor :data:`NON_PYTHON_ROLES`
    would be skipped by :func:`resolve` and nothing would say so — *absence
    of evidence read as evidence of absence*, in the one place it is cheapest
    to close. So a new kind of role in `src/` fails here until somebody
    decides whether it names a Python object.

    The two sets must also be DISJOINT, which is the half that matters: with
    no such check, moving `func` into `NON_PYTHON_ROLES` would silence most
    of this gate and every other test here would still pass.

    :data:`NON_PYTHON_ROLES` is deliberately NOT held to "some role of this
    kind is still written in `src/`". It classifies Sphinx's vocabulary, not
    instances in this tree, so an entry stays true whether or not the tree
    currently uses it — unlike the sibling gate's `PERMITTED`, which excuses
    named instances and must be pruned.
    """
    overlap = sorted(set(PYTHON_ROLES) & set(NON_PYTHON_ROLES))
    assert not overlap, (
        f"{overlap} is classified both as a python-domain role and as out of "
        f"scope. The out-of-scope set wins in `resolve`, so an overlap "
        f"silences that role everywhere."
    )
    seen = collections.Counter(
        ref.role
        for rel, path in source_files()
        for ref in references(path.read_text(encoding="utf-8"), rel)
    )
    assert seen, "no cross-reference role was found anywhere in `src/`"
    undecided = sorted(set(seen) - set(PYTHON_ROLES) - set(NON_PYTHON_ROLES))
    assert not undecided, (
        f"role(s) written in `src/` that this gate neither resolves nor "
        f"declares out of scope: {undecided}. Add it to `PYTHON_ROLES` if it "
        f"names a Python object, or to `NON_PYTHON_ROLES` with the reason it "
        f"cannot be resolved here."
    )
    for role, why in NON_PYTHON_ROLES.items():
        assert len(why.split()) >= 10, (
            f"the entry for :{role}: carries no reason a reader can check"
        )


def test_the_archaeology_this_file_hands_a_reader_is_still_anchored():
    """The instruction four documents give, checked against the tree it reads.

    **THE PROSE USED TO HAND A READER `-S` AND THE COMMIT THAT WROTE THE
    PROSE FALSIFIED IT** — see :func:`definition_search` for the mechanism
    and for why the finding survived it. The repair is an anchored `-G`, and
    an anchor is only worth what the tree does not defeat, so this checks
    the tree rather than asserting the anchor.

    **NO GIT AND NO HISTORY**, deliberately, and that is the whole design.
    History is immutable and already measured; what moves is this tree's
    text, and this tree's text is what defeated the old instruction. A
    check that shelled out to `git log` would answer the settled half and
    would skip in the two places the suite most needs it to run — an
    unpacked sdist and a shallow clone — where a skip reads as a pass.
    Every assertion below is a scan of `src/` and holds in all three.

    Three things, and each can go red on its own:

    1. the anchored pattern matches NO line of `src/`, so the command still
       returns nothing for a reason this tree controls. A fenced example
       writing `def _classify_cmp(` at column zero is the one way the
       repaired instruction can be defeated, and it fails here;
    2. the CONTROL — the same pattern built from the name each was mistaken
       for — DOES match, so a zero above is a reading and not a pattern
       that matches nothing anywhere;
    3. the unanchored string IS in `src/`, which is what falsified `-S`.
       If that stops being true the correction record has been deleted, and
       both this test and the limit paragraph in the module docstring are
       describing a tree that no longer exists.
    """
    source = "\n".join(
        path.read_text(encoding="utf-8") for _rel, path in source_files()
    )
    for name, entry in NEVER_DEFINED.items():
        declaration = f"{entry.declarator} {name}"
        _argv, shown = definition_search(declaration)
        assert not re.search(
            _DECLARED_HERE.format(re.escape(declaration)), source, re.M
        ), (
            f"`src/` now declares `{declaration}` at the start of a line, so "
            f"`{shown}` — the command this file, `src/stelling/propagate.py` "
            f"and `SOUNDNESS.md` all hand a reader — returns a commit, and "
            f"every one of those documents says it returns none. Either the "
            f"name has been implemented (delete its `NEVER_DEFINED` entry and "
            f"re-derive the census), or prose has written a declaration at "
            f"column zero and the anchor no longer separates a definition "
            f"from a mention."
        )
        control = f"{entry.declarator} {entry.instead}"
        assert re.search(
            _DECLARED_HERE.format(re.escape(control)), source, re.M
        ), (
            f"the control failed: `src/` does not declare `{control}`, the "
            f"name `{name}` was mistaken for. The anchored search cannot be "
            f"read as evidence of absence while it finds nothing when the "
            f"thing IS there — re-derive the archaeology before trusting any "
            f"sentence that cites it."
        )
        # 3. and the reason the anchor was needed at all.
        assert name in source, (
            f"{name!r} is no longer anywhere in `src/`. The correction record "
            f"that quotes it is what made the unanchored `-S` search return "
            f"the record itself; with it gone, this test and the module "
            f"docstring's limit paragraph both describe a tree that is not "
            f"here any more."
        )


def test_the_sibling_gates_presence_test_is_satisfied_by_a_record_of_absence():
    """The limit this branch opened in `tests/test_documented_names_exist.py`.

    **A GUARD THAT IS GREEN BECAUSE IT IS BLIND IS THE THING THIS REPOSITORY
    KEEPS FINDING**, so the blindness is measured here rather than asserted
    in a docstring. The sibling gate decides "present in `src/`" by substring
    over the source text; the correction records this branch wrote put
    `_classify_cmp` and `_Walker` into that text; so a document asserting
    either name is now told it exists by the sentences that say it does not.

    **THIS TEST GOES RED WHEN THE HOLE IS CLOSED, AND THAT IS THE POINT.**
    If the sibling gate ever stops reporting the plant below as findable,
    delete this test and say so — the same discipline
    `test_documented_names_exist.py::test_every_permitted_name_is_still_needed`
    uses. Until then it is the only thing in the tree that re-derives the
    limit every run.

    The module docstring carries why the obvious repair — resolving an
    asserted name as a symbol — is refused: **39 of the 62 checked rows**
    become findings, on a measurement rather than on a preference. This
    said *"28 of 62 rows"*, which was that measurement's distinct-PAIRS
    figure quoted as a row count (0.2.1 re-audit).
    """
    sibling = importlib.import_module("test_documented_names_exist")
    table = namespace()
    source = sibling._in_source()
    absent = "a_name_that_is_definitely_not_in_this_package"

    for name, entry in NEVER_DEFINED.items():
        assert len(entry.why.split()) >= 12, (
            f"the entry for {name!r} carries no reason a reader can check"
        )
        # 1. still not a symbol, by the file's ONE predicate for that. This
        #    read its own inline `bound`/`attributes` test until the 0.2.1
        #    re-audit; see :func:`binds` for the row that separated them.
        assert not resolves_as_a_symbol(name, table), (
            f"{name!r} is a symbol of this package now, so nothing about it "
            f"is a record of absence any more. Delete its `NEVER_DEFINED` "
            f"entry and the paragraph that describes this limit."
        )
        # 2. still in the source TEXT, which is what makes the hole real.
        assert name in source, (
            f"{name!r} is no longer anywhere in `src/`, so the record that "
            f"kept it there is gone and the sibling gate can see it again. "
            f"Delete its `NEVER_DEFINED` entry."
        )
        # 3. THE HOLE, driven through the sibling's own machinery: its four
        #    constructions, its own `missing_names`, its own source text.
        planted = f"The `{name}` method is read on every assume."
        assert sibling.asserted_names(planted), (
            f"the plant for {name!r} does not fire the sibling's own "
            f"constructions, so this test would prove nothing about it"
        )
        assert not sibling.missing_names([("<plant>", planted)], source), (
            f"the sibling gate now REPORTS {name!r}. The hole is closed — "
            f"delete this test and the limit paragraph in the module "
            f"docstring, which both say it is open."
        )

    # THE CONTROL, without which the three assertions above are satisfied by
    # a gate that reports nothing at all.
    control = f"The `{absent}` method is read on every assume."
    assert sibling.missing_names([("<plant>", control)], source), (
        "the sibling gate did not report a name that is nowhere in `src/`. "
        "Its own tests own that failure, but this one cannot distinguish "
        "'blind to records of absence' from 'blind to everything' without it."
    )

    # ... and the hole is LATENT, not live: nothing the sibling gate actually
    # reads asserts either name. This is the alarm that gate cannot raise for
    # itself, and it is derived from its corpus rather than pinned.
    live = sorted(
        (rel, name)
        for rel, text in sibling.corpus()
        for _line, _label, name, _q in sibling.asserted_names(text)
        if name in NEVER_DEFINED
    )
    assert not live, (
        f"a document the sibling gate reads now asserts a name this tree "
        f"only records as never-defined: {live}. That gate will call it "
        f"PRESENT, because a correction record in `src/` spells it. Repair "
        f"the document — the name has never existed."
    )


@pytest.mark.parametrize("stage", ["files", "references", "resolutions"])
def test_the_scan_actually_reached_the_tree(stage: str):
    """Anti-vacuity, at the three places this gate can go quiet.

    Every figure below is DERIVED from the tree at run time rather than
    typed, because a numeral beside something that moves rots. An empty
    `src/`, a regex that stopped matching, or a resolver that answered `None`
    to everything would each show up as a collapse here and as a green above.
    """
    files = source_files()
    if stage == "files":
        assert len(files) > 1, f"`src/` walked to {len(files)} Python file(s)"
        return
    refs = [
        ref
        for rel, path in files
        for ref in references(path.read_text(encoding="utf-8"), rel)
    ]
    if stage == "references":
        assert len(refs) > len(files), (
            f"{len(refs)} cross-reference role(s) found across {len(files)} "
            f"files. This repository writes far more than one per module, and "
            f"a collapse to nearly none means the scan stopped matching."
        )
        return
    table = namespace()
    modules = {rel: module_name(path) for rel, path in files}
    decided = sum(
        1
        for ref in refs
        if ref.role in PYTHON_ROLES
        and _DOTTED.match(ref.target)
        and resolve(ref, modules[ref.path], table) is None
    )
    assert decided > len(refs) // 2, (
        f"only {decided} of {len(refs)} roles were positively resolved. This "
        f"gate's green means 'every reference was checked and found', and a "
        f"resolver that stopped finding things would report the same green "
        f"while checking nothing."
    )
