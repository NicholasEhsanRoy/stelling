# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The `falsify` keyword's stability level, in ONE place, and held there.

**THE DEFECT THIS ANSWERS IS A WORD, AND IT WAS THE WRONG WORD.**
`DOCUMENTATION_ARCHITECTURE.md` §8.5 carries a four-level table, and its
two middle levels are not softer and harder shades of one another —

    | `provisional`  | May change in minor with a deprecation cycle | One minor's notice |
    | `experimental` | May change without notice                    | None               |

— they differ by a PROMISE. `src/stelling/falsify.py`'s heading read
*"STABILITY: SHIPPED, DEFAULT-OFF, AND PROVISIONAL"* while the very next
sentence disclosed that `probe()`'s first parameter *"changed name and type
inside this release cycle, and it is in ``__all__``"*: a change made with no
deprecation cycle and no notice, which is the table's definition of the
OTHER word. `preconditions.check`'s docstring said *"SHIPPED AND
PROVISIONAL"* beside the same disclosure. So the shipped text spent a word
that promises one minor's notice on a keyword that had already broken that
notice once and could not promise it again — and a reader who knows the
table would have come away with a promise nobody made.

**AND THE USER-FACING PAGES ASSIGNED NO LEVEL AT ALL.** Before this file
there was not one level-word about `falsify` anywhere under `docs/`:
`docs/harness-api.md`'s argument table said *"the falsification pass; see
Reading a verdict"* and `docs/preconditions.md` said nothing about
stability. A reader deciding whether to switch on an instrument that can
RAISE had to open `src/` to find out what the keyword promised.

────────────────────────────────────────────────────────────────────────────
WHY THIS IS A PIN AND NOT A PROOFREAD
────────────────────────────────────────────────────────────────────────────

A level restated in five places is five copies of one fact, and this
repository's whole record is that copies drift: `falsify.py`'s own docstring
carries three separate paragraphs retracting figures that appeared in one
sentence and nowhere else. So the level is written down ONCE, as
:data:`stelling.falsify.STABILITY`, and every site that states it is held to
that string — read out of the source by :func:`_source_level`, never retyped
here. Flip the constant to any other level and every assertion below fails,
naming the site it failed at. That is the same discipline `tests/
_repo_files.py`'s `SKIP` gets from `test_sdist_contents.WITHHELD`, and the
same one `stelling.falsify.FALSIFY_MODES` gets from its own second-spelling
pin.

**THE POLICY TABLE IS READ TOO, AND NOT REMEMBERED.**
:func:`_policy_levels` parses §8.5 rather than hard-coding four words, so
this pin cannot certify a level the policy no longer defines, and the
MEANING each page quotes is the meaning that table gives today. Reword the
`experimental` row and the pages must be reworded with it — which is the
direction the defect ran the first time.

**IT READS THE TREE, NOT THE IMPORT, AND THAT IS DELIBERATE.**
`stelling.falsify` imports jax by design (`require("jax")` at its head), so
an importing pin would skip in the zero-dep lane — and the zero-dep lane is
exactly where a documentation claim is cheapest to leave rotting. Everything
here is `ast` and `str` over files, so it runs in every lane, including one
with no jax at all. Where jax IS importable, :func:`
test_the_constant_this_pin_reads_is_the_one_the_module_exports` closes the
gap between the text read and the object built.

**WHY THIS IS NOT CALLED `test_falsify_stability_level.py`, WHICH IS THE
NAME ITS SUBJECT WOULD SUGGEST.** The same reason
`tests/test_probe_oracle.py` is not called `test_falsify_oracle.py`, and
its docstring carries the long form. `tests/test_narrowing_perimeter.py` records a measured
COLLECTION RANK — *"this file sorts 72nd of the files `pytest
--collect-only -q -p no:randomly` names in this tree"* — because that rank
is what sets the size of the exposure its incident describes, and
`test_this_files_position_in_the_collection_is_the_measured_one` demands
the phrase verbatim in that file AND in `CHANGELOG.md`. Any new test file
sorting before `test_n…` moves the rank to 73rd and leaves both artefacts
one measurement behind; driven, that test failed under
`test_falsify_stability_level.py` and is green under this name. Correcting
the two artefacts is the right repair and it is not this branch's to make.
`probe` is this module's own word for what is under test, and the file
sits with `test_probe_oracle.py` and `test_probe_witness.py`, which are
its siblings by subject as well as by sort order.

────────────────────────────────────────────────────────────────────────────
WHAT IS DELIBERATELY *NOT* PINNED HERE
────────────────────────────────────────────────────────────────────────────

The six disclosures in `falsify.py`'s docstring are NOT a restatement of the
level and are not checked against it. A level is a promise about future
changes to a surface; the six are a statement about what the instrument does
not do today, and no level-word in §8.5 implies any of them. What IS pinned
about them is that the heading introducing them names no policy level — the
old heading, *"WHAT \"PROVISIONAL\" NAMES IS A LIST, NOT A MOOD"*, made a
policy word the label for a list the policy does not describe, and that is
the collision in miniature — and that item 1, the one open hole, is still
there in full.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FALSIFY = REPO / "src" / "stelling" / "falsify.py"
PRECONDITIONS = REPO / "src" / "stelling" / "preconditions.py"
POLICY = REPO / "DOCUMENTATION_ARCHITECTURE.md"
HARNESS_API = REPO / "docs" / "harness-api.md"
PRECONDITIONS_DOC = REPO / "docs" / "preconditions.md"


def _source_level() -> str:
    """`STABILITY` from `falsify.py`, by `ast` and without importing jax.

    Exactly one module-level binding of the name, or this raises: two
    bindings is the drift this file exists to stop, one commit earlier.
    """
    tree = ast.parse(FALSIFY.read_text(encoding="utf-8"), filename=str(FALSIFY))
    found = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "STABILITY":
                assert isinstance(node.value, ast.Constant), (
                    "src/stelling/falsify.py's STABILITY is not a literal; "
                    "this pin and every page below read it as one"
                )
                found.append(node.value.value)
    assert len(found) == 1, (
        f"src/stelling/falsify.py binds STABILITY {len(found)} times at "
        f"module level; the level must be written down exactly once"
    )
    assert isinstance(found[0], str) and found[0], (
        f"src/stelling/falsify.py's STABILITY is {found[0]!r}, not a "
        f"non-empty string"
    )
    return found[0]


def _flat(path: Path) -> str:
    """One file, with every run of whitespace collapsed to a single space.

    Every check below matches a PHRASE, and a phrase in this repository is
    routinely wrapped mid-way: `docs/harness-api.md` carries §8.5's meaning
    for the assigned level with a line break falling between *"may"* and
    *"change without notice"*, and the first draft of this file read that
    page as SILENT about the very sentence it prints. So the comparison is
    over flattened text: a rewrap is not a claim change and must not read
    as one, while every word change still does.
    """
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


_TABLE_ROW = re.compile(r"^\|\s*`([a-z]+)`\s*\|([^|]+)\|([^|]+)\|\s*$")


def _policy_levels() -> dict[str, tuple[str, str]]:
    """§8.5's table, parsed: level -> (meaning, guarantee).

    By text and not by a markdown parser, for the reason `tests/_lanes.py`
    gives about `ci.yml`: this suite has no such dependency and is not
    acquiring one for a fence. The section heading is required to exist, so
    a renumbering is a red test rather than an empty dict quietly agreeing
    with everything.
    """
    text = POLICY.read_text(encoding="utf-8")
    marker = "### 8.5 API stability levels"
    assert marker in text, (
        f"{POLICY.name} no longer carries {marker!r}. The `falsify` "
        f"keyword's level is quoted from that table in four places; if the "
        f"section moved, they must move with it"
    )
    section = text.split(marker, 1)[1]
    levels: dict[str, tuple[str, str]] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            if levels:  # the table has ended
                break
            continue
        match = _TABLE_ROW.match(line)
        if match:
            levels[match.group(1)] = (
                match.group(2).strip(), match.group(3).strip()
            )
    assert len(levels) >= 4, (
        f"§8.5's table parsed to {sorted(levels)}; this pin needs the four "
        f"levels and their meanings"
    )
    return levels


#: The four sites that STATE the level, and the template each states it in.
#: Every template is formatted with the ONE string read out of the source, so
#: a change to the constant moves all four together or fails naming the one it
#: did not move. The templates differ because the registers do: a module
#: docstring's ruled heading, a parameter paragraph in `check`'s own
#: docstring, a cell in an argument table, and the opening of a section on a
#: page. Nothing here is a style rule -- each template is the shortest string
#: that can only be a LEVEL ASSIGNMENT and not a mention of one, which is what
#: lets the negative check below run without tripping on the three sites that
#: deliberately name the neighbouring level in order to disclaim it.
_SITES: tuple[tuple[Path, str], ...] = (
    (FALSIFY, "STABILITY: SHIPPED, DEFAULT-OFF, AND ``{level}``"),
    (PRECONDITIONS, "**SHIPPED, AND ``{level}``**"),
    (HARNESS_API, "**Stability: `{level}`**"),
    (PRECONDITIONS_DOC, "**Stability: `{level}`.**"),
)


def test_the_source_names_a_level_the_policy_table_defines():
    """A word that is not in §8.5 is not a level, however confident it reads."""
    level = _source_level()
    levels = _policy_levels()
    assert level in levels, (
        f"src/stelling/falsify.py's STABILITY is {level!r}, which "
        f"{POLICY.name} §8.5 does not define. It defines {sorted(levels)}"
    )


def test_every_page_that_names_the_level_names_the_one_in_the_source():
    """The level is one string, and these four sentences are its copies.

    The failure this catches is the cheap one and the likely one: somebody
    settles the keyword down, moves the constant to `provisional`, and four
    pages go on saying `experimental` — or the reverse, which is how the
    wrong word got there in the first place.
    """
    level = _source_level()
    missing = []
    for path, template in _SITES:
        wanted = template.format(level=level)
        if wanted not in _flat(path):
            missing.append(f"{path.relative_to(REPO)}: expected {wanted!r}")
    assert not missing, (
        f"src/stelling/falsify.py says the `falsify` keyword's stability "
        f"level is {level!r}; these sites do not say it in the form this "
        f"pin reads:\n  " + "\n  ".join(missing)
    )


def test_no_site_assigns_this_keyword_a_DIFFERENT_policy_level():
    """The other three words, in the same templates, must appear nowhere.

    Naming the level in one place and contradicting it in another is worse
    than naming it nowhere, because the contradiction reads as a decision.
    The three pages that CONTRAST `experimental` with `provisional` are
    unaffected: they name the other level in a sentence, never in the
    assignment form this scans for.
    """
    level = _source_level()
    others = [name for name in _policy_levels() if name != level]
    assert others, "§8.5 defines only one level; there is nothing to confuse"
    wrong = []
    for path, template in _SITES:
        text = _flat(path)
        for other in others:
            claim = template.format(level=other)
            if claim in text:
                wrong.append(f"{path.relative_to(REPO)}: says {claim!r}")
    assert not wrong, (
        f"the level is {level!r} in src/stelling/falsify.py, and these "
        f"sites assign a different one:\n  " + "\n  ".join(wrong)
    )


def test_both_user_facing_pages_quote_the_MEANING_the_policy_gives():
    """A level-word alone is a shibboleth; the meaning is the content.

    `docs/` is the half of this that a caller reads, and a caller who has
    never opened `DOCUMENTATION_ARCHITECTURE.md` gets nothing from the word
    `experimental` on its own. Both pages quote §8.5's own meaning for the
    assigned level, VERBATIM and case-insensitively, so a reworded policy
    row takes the pages with it.
    """
    level = _source_level()
    meaning = _policy_levels()[level][0].lower().rstrip(".")
    silent = [
        str(path.relative_to(REPO))
        for path in (HARNESS_API, PRECONDITIONS_DOC)
        if meaning not in _flat(path).lower()
    ]
    assert not silent, (
        f"§8.5 defines {level!r} as {meaning!r}; these user-facing pages "
        f"name the level without quoting what it means: {silent}"
    )


def test_no_shipped_site_promises_this_keyword_a_deprecation_cycle():
    """The one promise the old word made and this keyword cannot keep.

    `provisional`'s meaning is the only row in §8.5 carrying the phrase, so
    the phrase is DERIVED from the table rather than typed here. Every
    occurrence of it across the four sites must be a denial or a contrast:
    it may not stand in a sentence without one of `no`, `not`, `never` or
    `without`, or without naming the level whose row it is quoting. The old
    text failed this the moment the level was assigned — it made the promise
    by NAMING `provisional`, with no sentence anywhere denying it.
    """
    level = _source_level()
    levels = _policy_levels()
    promising = [
        name for name, (meaning, _) in levels.items()
        if "deprecation cycle" in meaning.lower()
    ]
    assert promising == ["provisional"], (
        f"§8.5's deprecation-cycle promise now sits on {promising}; this "
        f"pin was written when it sat on 'provisional' alone"
    )
    assert level not in promising, (
        f"the keyword is assigned {level!r}, which §8.5 says comes WITH a "
        f"deprecation cycle. This pin exists because the shipped text made "
        f"exactly that promise and could not keep it"
    )
    bare = []
    for path, _template in _SITES:
        text = _flat(path)
        for sentence in re.split(r"(?<=[.!?]) ", text):
            low = sentence.lower()
            if "deprecation cycle" not in low:
                continue
            if re.search(r"\b(no|not|never|without)\b", low):
                continue
            if f"`{promising[0]}`" in sentence or promising[0] in low:
                continue  # quoting the OTHER level's row, not promising it
            bare.append(f"{path.relative_to(REPO)}: {sentence.strip()[:120]}")
    assert not bare, (
        "a deprecation cycle is promised, unqualified, at:\n  "
        + "\n  ".join(bare)
    )


def _exceptions_that_escape() -> dict[str, str]:
    """The classes `check(..., falsify=...)` can raise, read off the source.

    Public (in `__all__`) exception classes of `falsify.py`, mapped to the
    base they were given. The BASE is the load-bearing half: it is what
    decides whether `except Exception:` contains the class, and it is a fact
    about the source rather than about anybody's memory of it.
    """
    text = FALSIFY.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(FALSIFY))
    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            exported = {
                elt.value for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
    assert exported, "src/stelling/falsify.py has no readable `__all__`"
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in exported:
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id.endswith(
                ("Exception", "Error")
            ):
                out[node.name] = base.id
    assert len(out) == 2, (
        f"expected exactly two exported exception classes in "
        f"src/stelling/falsify.py, found {out}"
    )
    return out


def test_both_user_facing_pages_say_the_call_can_RAISE_and_name_both_classes():
    """A caller who reads only `docs/` must still learn this.

    The two class names and the base of each are read out of `falsify.py`,
    so renaming either class, or re-parenting the `BaseException` one under
    `Exception`, fails here until the pages follow.
    """
    escaping = _exceptions_that_escape()
    gaps = []
    for path in (HARNESS_API, PRECONDITIONS_DOC):
        text = _flat(path)
        for name in escaping:
            if name not in text:
                gaps.append(f"{path.relative_to(REPO)}: never names {name}")
    outside = [n for n, base in escaping.items() if base == "BaseException"]
    assert len(outside) == 1, (
        f"expected exactly one exported class outside `Exception`, got "
        f"{outside}"
    )
    for path in (HARNESS_API, PRECONDITIONS_DOC):
        text = _flat(path)
        if "BaseException" not in text:
            gaps.append(
                f"{path.relative_to(REPO)}: does not say that "
                f"{outside[0]} is a BaseException"
            )
        if "except Exception" not in text:
            gaps.append(
                f"{path.relative_to(REPO)}: does not say that "
                f"`except Exception:` will not catch {outside[0]}"
            )
    assert not gaps, (
        "`check(..., falsify=\"sample\")` may raise instead of returning, "
        "and one class is outside `Exception`. The pages a caller reads "
        "must say both:\n  " + "\n  ".join(gaps)
    )


def test_both_user_facing_pages_say_the_probe_can_only_REFUTE():
    """A firing is a counterexample; a silence is nothing.

    The asymmetry is the whole instrument, and it is the one thing a reader
    can get backwards in the expensive direction — reading a quiet probe as
    confidence. Both pages must say it in as many words.
    """
    gaps = [
        str(path.relative_to(REPO))
        for path in (HARNESS_API, PRECONDITIONS_DOC)
        if "only refute" not in _flat(path).lower()
    ]
    assert not gaps, (
        "these pages describe the falsification pass without saying it can "
        f"only refute, so a silent probe reads as evidence: {gaps}"
    )


def test_the_six_disclosures_are_not_introduced_by_a_policy_word():
    """The collision in miniature, and the half a level-word cannot fix.

    The list used to be headed *what "provisional" names*, which made a
    policy word the label for a list the policy does not describe. Whatever
    the heading says now, it may not be a level.
    """
    text = FALSIFY.read_text(encoding="utf-8")
    heading = "**THE SIX DISCLOSURES,"
    assert heading in text, (
        "src/stelling/falsify.py no longer introduces the six disclosures "
        "under a heading this pin can find"
    )
    line = text.split(heading, 1)[1].split("\n\n", 1)[0]
    intro = (heading + line).split("**", 2)[1].lower()
    named = sorted(n for n in _policy_levels() if n in intro)
    assert not named, (
        f"the six disclosures are introduced by a §8.5 level word {named}; "
        f"a level is a promise about future change and the six are a "
        f"statement about what the probe does not do today"
    )


def test_item_one_the_open_hole_is_still_stated_in_full():
    """The one disclosure that no fix in this module can close.

    It is the item most worth losing to an edit that tidies a list, so it is
    pinned by its own text rather than by the list's length.
    """
    text = FALSIFY.read_text(encoding="utf-8")
    for fragment in (
        "1. **THE ANALYSIS'S REGION IS NEVER READ.**",
        "Nothing checks that the",
        "AND THE ANALYSIS'S REGION IS NEVER READ, WHICH IS NOT AN INSTANCE OF",
    ):
        assert fragment in text, (
            f"src/stelling/falsify.py no longer states {fragment!r}; item 1 "
            f"of the six disclosures is the open hole and must stay in full"
        )
    for n in range(1, 7):
        assert f"\n{n}. **" in text, (
            f"src/stelling/falsify.py's disclosure list has lost item {n}"
        )


def test_the_constant_this_pin_reads_is_the_one_the_module_exports():
    """Text and object, closed where the object can be built.

    Everything above reads the tree, which is what lets it run with no jax.
    That leaves one gap a tree-reader cannot close on its own: an `ast`
    literal is not proof of what `import stelling.falsify` binds. Where jax
    is present, this closes it; where it is not, this test says so by
    skipping rather than by passing.
    """
    pytest.importorskip("jax")
    from stelling import falsify

    assert falsify.STABILITY == _source_level()
    assert "STABILITY" in falsify.__all__, (
        "the level is stated on four pages; a caller must be able to read it "
        "off the module too"
    )
