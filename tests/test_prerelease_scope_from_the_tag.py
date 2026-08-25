# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""*"0.1.0 pre-release builds only"* is checkable AGAINST THE TAG. Checked.

`tests/test_soundness_log_reach.py` requires every `## Log` bullet to answer
the reach question with exactly one of three `Versions:` phrases, and derives
the reached-release count from the answers. **It never asks whether an answer
is TRUE**, and that is the whole defect class this file exists to close: a
guard that checks a claim is WELL-FORMED but not that it holds. It was green
for the entire life of that vocabulary while **six** bullets said
*0.1.0 pre-release builds only* and were wrong.

**THE PHRASE HAS A MACHINE-CHECKABLE MEANING AND NOBODY WAS USING IT.**
`SOUNDNESS.md`'s own preamble defines it: *"means the thing this entry records
— a defect fixed, a disclosure gap closed, a claim narrowed — was over before
the `v0.1.0` tag"*. An event that was over before the tag was written up before
the tag, and what was written up before the tag is exactly what
`git show v0.1.0:SOUNDNESS.md` holds. So the scoped set and the tagged set are
the same set, and the tag is in the repository:

    the bullets scoped `0.1.0 pre-release builds only`
      ==  the `## Log` bullets present in `git show v0.1.0:SOUNDNESS.md`

An audit established that bijection by hand. Hand-established bijections are
what this campaign keeps re-establishing; this one is now derived on every run.

**WHY THIS AND NOT A DATE COMPARISON.** A date-versus-field consistency check
— *"a bullet dated after 2026-08-12 may not be scoped pre-release-only"* —
finds the same six on this tree, and it is the weaker instrument in three
ways. It reads a headline's typed date rather than the repository, so a wrong
date makes it agree with a wrong field. It cannot see the other direction at
all (a bullet that IS at the tag and is scoped as something else). And it has
nothing to say about a bullet whose date is right and whose scope is still
wrong — an entry dated 2026-08-11 that was written up in September is
date-consistent and absent from the tag. This check reads the tag, so it
answers all three.

**BOTH DIRECTIONS, BY NAME**, because they are different defects with
different fixes:

* *scoped pre-release, absent at the tag* — the field is wrong, or the entry
  is a later write-up of an earlier event and the log's definition is being
  stretched. Six of these, live on this tree.
* *present at the tag, not scoped pre-release* — the entry was in the shipped
  document and now says its event happened after the tag, which cannot be.

**IDENTITY IS `(date, headline)` AND NOT THE BYTES.** A `## Log` entry is
amended in place — a later paragraph, a corrected figure, a
cross-reference to the batch that superseded it — so byte equality would
report every amended entry as a stray in both directions at once and this
gate would be silenced in a week. The headline is the leading bold run with
its date-and-parenthetical prefix removed, whitespace-normalised: stable
under an amendment, and measured unique across both documents (33 at the tag,
54 here, no collisions) by
:func:`test_the_identity_is_unique_in_both_documents`.

**WHAT THIS DOES NOT COVER, AND IT IS TWO THIRDS OF THE VOCABULARY.** The
bijection exists because *0.1.0 pre-release builds only* is a claim about the
DOCUMENT at the tag. The other two phrases are not, and no comparison of
documents can reach them: *0.2.0 development builds only* and *`v0.1.0` and
0.2.0 development builds* both describe entries written up AFTER the tag, and
what separates them is whether the defect was **present in the released
code** — a question about `src/` at `v0.1.0`, which `SOUNDNESS.md` at
`v0.1.0` cannot answer, because a defect can be in the shipped code and in
nobody's log. A bullet that says *0.2.0 development builds only* about
something that shipped in `v0.1.0` is the same class of false field as the
six below and this file cannot see it. Settling one of those means
REPRODUCING the entry at the tag, which is a harness question and not a text
one; `tests/test_soundness_log_reach.py`'s docstring is where the reach
declarations that DO carry a reproduction at the tag are described.

**THIS NEEDS GIT AND THE TAG, SO IT SKIPS RATHER THAN DEGRADING.** There is
no weaker version of "compare against the tagged document" that is worth
running — a fallback to the date proxy would be this file quietly becoming the
instrument it was written to replace, in the environments where nobody looks.
The skip is disclosed through `tests/test_skip_inventory.py`'s `RULES`, whose
predicate asks git the same question this file does.
"""

from __future__ import annotations

import ast
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _release_record import VERSION_FIELDS  # noqa: E402
from test_soundness_log_reach import log_bullets  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
SOUNDNESS = REPO / "SOUNDNESS.md"

#: The tag the phrase is about. `SOUNDNESS.md`'s preamble names it in so many
#: words — *"was over before the `v0.1.0` tag"* — so this is the document's
#: own constant and not this file's choice of one.
TAG = "v0.1.0"

#: The scoping phrase, in the Log's italic spelling. Derived from
#: `_release_record.VERSION_FIELDS` for the reason
#: `test_soundness_log_reach.py` gives: the closed set is ONE set, written
#: once, and a second copy of the wording here would be a second thing to
#: keep in step.
PRE_RELEASE_ONLY = f"*{VERSION_FIELDS[0]}*"

#: The skip reason, typed here and typed again in
#: `tests/test_skip_inventory.py`'s `RULES`. It is a LITERAL on both sides on
#: purpose: that file's `test_a_rule_excuses_only_reasons_written_down_in_this
#: _file` requires the reason a rule excuses to be a string a reviewer reads
#: in the disclosure surface, so an f-string carrying git's stderr — which is
#: what made nine skips undisclosable in August — cannot be used here.
SKIP_REASON = (
    "git cannot read `v0.1.0:SOUNDNESS.md`, so the pre-release scope "
    "cannot be derived from the tag"
)

#: `log_bullets` ends the `## Log` at the next `## ` heading, and at the tag
#: the Log IS the last section — so the parser the whole campaign shares would
#: raise `StopIteration` on the tagged document and this file would be a
#: `## Log` parser of its own within a week. A sentinel heading is appended
#: instead, unconditionally so both documents go through one code path: on
#: `SOUNDNESS.md` today it is inert (`## 0.2.0 soundness-fix detail` already
#: closes the section) and on the tagged text it supplies the terminator the
#: end of the file otherwise is.
_END_SENTINEL = "\n## (end of document — appended by the reader, not in the file)\n"


def _log_bullets(text: str):
    """`log_bullets` over a document whose `## Log` may run to EOF."""
    return log_bullets(text + _END_SENTINEL)


_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_DATED_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*(?:\([^)]*\))?\s*:\s*")
_ANY_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def identity(body: str) -> tuple[str, str]:
    """`(date, headline)` for one `## Log` bullet.

    The headline is the entry's leading bold run — every bullet in both
    documents opens with one — with the `YYYY-MM-DD (parenthetical):` prefix
    stripped and whitespace collapsed. The parenthetical goes because it is
    the part that gets edited (*"(pre-release)"*, *"(pre-release, same day)"*,
    *"(B6, same batch)"*) while the sentence after the colon is what the entry
    IS.

    A bullet with no bold run at all falls back to its whole first paragraph,
    normalised, which is degraded but never silently wrong: it will simply
    fail to match its counterpart and be reported by name in whichever
    direction it stands. Nothing in either document takes that branch today.
    """
    hit = _BOLD.search(body)
    raw = " ".join((hit.group(1) if hit else body).split())
    prefix = _DATED_PREFIX.match(raw)
    if prefix:
        return prefix.group(1), raw[prefix.end():]
    date = _ANY_DATE.search(raw)
    return (date.group(0) if date else "?"), raw


def tagged_soundness() -> str | None:
    """`SOUNDNESS.md` as the tag has it, or `None` where git cannot say.

    `None` and never `""`: *"a command that could not run reports the same
    'found nothing' as a command that ran and found nothing"* is the rule
    `tests/_repo_files.py::tracked_paths` states, and an empty document here
    would read as *"the tag's log holds no bullets"*, which would make every
    scoped bullet a stray and this file a fountain of false accusations in
    exactly the environments that cannot check it.
    """
    if shutil.which("git") is None or not (REPO / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "show", f"{TAG}:SOUNDNESS.md"],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    return proc.stdout if proc.returncode == 0 and proc.stdout.strip() else None


def strays(here: str, at_the_tag: str) -> tuple[list, list]:
    """The two directions: `(scoped_but_absent, present_but_not_scoped)`.

    Takes both documents as TEXT so the rule this file ships is the rule its
    own drive test drives. A test that re-implements the comparison on
    synthetic input pins a copy, and a pin over a re-implementation goes green
    over the absence of the thing it pins — the shape
    `test_soundness_log_reach.py`'s preamble pin was measured failing at, at
    `9 passed` with the rule deleted.

    Entries are `(date, headline, line)`; the line is the one in whichever
    document the entry is missing from the other side of, so a failure names
    a place to go.
    """
    mine = {identity(body): line for line, body in _log_bullets(here)}
    scoped = {
        identity(body)
        for _line, body in _log_bullets(here)
        if PRE_RELEASE_ONLY in body
    }
    tagged = {identity(body): line for line, body in _log_bullets(at_the_tag)}
    absent = sorted(
        (date, head, mine[(date, head)]) for date, head in scoped - set(tagged)
    )
    unscoped = sorted(
        (date, head, tagged[(date, head)])
        for date, head in set(tagged) - scoped
    )
    return absent, unscoped


def _rewording_hint(absent: list, unscoped: list) -> str:
    """Name same-date pairs across the two directions.

    An entry whose HEADLINE was rewritten after the tag shows up once in each
    list, and reporting it as two unrelated strays would send a reader looking
    for a missing entry that is right there. Same date on both sides is the
    cheap signal for it, and it is offered as a hint rather than acted on:
    silently pairing them would be identity-by-date, which is not unique.
    """
    dates = {date for date, _h, _l in absent} & {date for date, _h, _l in unscoped}
    if not dates:
        return ""
    return (
        "\n\nNOTE: these dates appear on BOTH sides — "
        + ", ".join(sorted(dates))
        + ". An entry whose headline was rewritten after the tag is one entry "
        "reported twice, not two strays; compare the headlines before "
        "changing a field."
    )


@pytest.fixture(scope="module")
def documents() -> tuple[str, str]:
    at_the_tag = tagged_soundness()
    if at_the_tag is None:
        pytest.skip(SKIP_REASON)
    return SOUNDNESS.read_text(encoding="utf-8"), at_the_tag


def test_the_prerelease_scope_is_exactly_the_log_the_tag_shipped(documents):
    """The bijection, derived. Both directions, both reported by name.

    **THIS IS RED ON `main` AT `115d771` AND THE SIX IT NAMES ARE REAL.** They
    are the six false `Versions:` fields the campaign found by hand, dated
    2026-08-14 and 2026-08-15 against a tag dated 2026-08-12, and every one of
    them is in `SOUNDNESS.md`, which the agent that wrote this file does not
    own. The red IS the report. It is not allowlisted, because an allowlist
    entry for the defect a gate was written to catch is the gate arriving
    pre-silenced.
    """
    here, at_the_tag = documents
    absent, unscoped = strays(here, at_the_tag)

    def render(rows):
        return "\n  ".join(f"{d}  line {ln:>6}  {h[:96]}" for d, h, ln in rows)

    assert not (absent or unscoped), (
        f"`SOUNDNESS.md`'s `## Log` and `git show {TAG}:SOUNDNESS.md` "
        f"disagree about which entries are pre-release.\n\n"
        f"{len(absent)} bullet(s) scoped {PRE_RELEASE_ONLY!r} that the tagged "
        f"document does NOT contain (line numbers are in SOUNDNESS.md):\n  "
        f"{render(absent) or '(none)'}\n\n"
        f"{len(unscoped)} bullet(s) present at {TAG} that are NOT scoped "
        f"pre-release-only (line numbers are in the tagged document):\n  "
        f"{render(unscoped) or '(none)'}\n\n"
        f"The phrase means the entry's event was over before the {TAG} tag "
        f"(SOUNDNESS.md's own preamble), and an event that was over before "
        f"the tag was written up before the tag. Either the field is wrong, "
        f"or the entry records something the tagged document did not have "
        f"and the phrase does not fit it."
        + _rewording_hint(absent, unscoped)
    )


def test_the_identity_is_unique_in_both_documents(documents):
    """Anti-vacuity, and the assumption the comparison rests on.

    `strays` compares SETS. If two bullets collided on `(date, headline)` the
    sets would be short and a real stray could hide behind a duplicate — the
    comparison would go quiet, which is the direction that matters. Measured:
    54 distinct identities here, 33 at the tag, no collision on either side.
    """
    for label, text in (("SOUNDNESS.md", documents[0]), (f"{TAG}", documents[1])):
        bodies = [body for _line, body in _log_bullets(text)]
        ids = [identity(body) for body in bodies]
        seen: dict[tuple[str, str], int] = {}
        clashes = []
        for (line, _body), key in zip(_log_bullets(text), ids):
            if key in seen:
                clashes.append(f"{key[0]} {key[1][:70]!r} at lines {seen[key]}, {line}")
            seen[key] = line
        assert not clashes, (
            f"two `## Log` bullets in {label} share one (date, headline) "
            f"identity, so the set comparison in this file is short by one "
            f"and a stray can hide behind the duplicate:\n  "
            + "\n  ".join(clashes)
        )
        assert len(ids) == len(set(ids)) and ids, f"{label} parsed to {len(ids)} bullets"


def test_both_directions_of_the_rule_are_driven(documents):
    """The rule, observed to fire, in each direction, on scratch copies.

    *A guard never observed to fire is not known to be a guard* —
    `.github/workflows/release.yml`'s own header. So the two directions are
    planted here rather than argued for, and the plants go through
    :func:`strays`, not through a copy of it.

    Both plants are built by REWRITING a real entry in an in-memory copy of
    the documents, so the synthetic case has the same shape as the real one
    and no file on disk is touched.
    """
    here, at_the_tag = documents
    absent, unscoped = strays(here, at_the_tag)

    # THE CONTROL, and it is not "the tree is clean" — it cannot be, six
    # bullets are wrong. It is that each plant below adds EXACTLY ONE stray
    # to the direction it targets, which is a statement about the rule and
    # not about the tree.
    tagged_bodies = [body for _line, body in _log_bullets(at_the_tag)]
    assert len(tagged_bodies) >= 2, "the tagged log is too small to drive this"

    # DIRECTION 1 — scoped pre-release, absent at the tag. Delete one entry
    # from the tagged copy while `SOUNDNESS.md` goes on scoping it.
    victim = next(
        body for body in tagged_bodies
        if identity(body) not in {(d, h) for d, h, _ in absent}
    )
    # The stand-in carries no date-shaped string, because
    # `test_the_skip_is_the_only_degradation_offered` refuses a date literal
    # anywhere in this module's code and is right to: a date typed in here is
    # a date this check could start comparing against.
    thinned = at_the_tag.replace(
        victim, "- **a stand-in for the entry this plant removed.** body.\n", 1
    )
    a1, u1 = strays(here, thinned)
    added = {(d, h) for d, h, _ in a1} - {(d, h) for d, h, _ in absent}
    assert added == {identity(victim)}, (
        "removing one entry from the tagged document did not make exactly "
        f"that entry a scoped-but-absent stray; it added {added}"
    )

    # DIRECTION 2 — present at the tag, scoped as something else. Re-scope a
    # bullet that IS at the tag, in a copy of `SOUNDNESS.md`.
    at_tag_ids = {identity(body) for body in tagged_bodies}
    line, body = next(
        (line, body) for line, body in _log_bullets(here)
        if identity(body) in at_tag_ids and PRE_RELEASE_ONLY in body
    )
    rescoped = here.replace(
        body, body.replace(PRE_RELEASE_ONLY, f"*{VERSION_FIELDS[1]}*", 1), 1
    )
    a2, u2 = strays(rescoped, at_the_tag)
    assert [(d, h) for d, h, _ in u2] == [identity(body)], (
        f"re-scoping the bullet at line {line} — which IS in the tagged "
        f"document — to {VERSION_FIELDS[1]!r} was not reported as a "
        f"present-at-the-tag-but-not-scoped stray; got {u2}"
    )
    assert {(d, h) for d, h, _ in a2} == {(d, h) for d, h, _ in absent}, (
        "re-scoping a bullet moved something in the OTHER direction too, so "
        "the two lists are not independent"
    )

    # ... and an AMENDMENT is not a stray, which is the reason identity is
    # not byte equality. Append a paragraph to a matched entry and nothing
    # moves in either direction.
    amended = here.replace(body, body + "\n\n  A later amendment paragraph.\n", 1)
    a3, u3 = strays(amended, at_the_tag)
    # IDENTITIES, not the rows: the inserted paragraph moves every line
    # number below it, and a comparison including those would fail for a
    # reason that has nothing to do with what is being asserted.
    def ids(rows):
        return {(d, h) for d, h, _line in rows}

    assert (ids(a3), ids(u3)) == (ids(absent), ids(unscoped)), (
        "appending an amendment paragraph to a `## Log` entry changed the "
        "stray lists, so identity has become byte equality and every amended "
        "entry will be reported as a stray in both directions at once"
    )


def test_the_skip_is_the_only_degradation_offered():
    """The tag is not optional, and no weaker check stands in for it.

    Two things are asserted, because the failure mode is a later edit quietly
    adding a fallback rather than deleting the skip:

    * :func:`tagged_soundness` answers `None` — not `""` — when git cannot
      read the tag, so an unreadable tag can never read as an empty log and
      turn every scoped bullet into a stray;
    * **no date literal appears in this module's CODE.** A
      date-versus-field proxy is the instrument this file replaces; putting
      one back here as a fallback would leave the weaker check running under
      this file's name in exactly the environments that cannot run the
      stronger one. Prose may discuss the date — the module docstring does,
      at length — so the scan is over string constants that are not
      docstrings, via `ast`, and not over the file's bytes. Reading the bytes
      is what the first draft of this test did, and it failed on its own
      docstring.
    """
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    dated = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and _ANY_DATE.search(node.value)
    ]
    assert not dated, (
        f"this module carries a date literal in code: {dated}. The check is "
        f"against the tagged DOCUMENT. A date comparison is the proxy that "
        f"found six of these and is silent over an entry whose date is right "
        f"and whose scope is still wrong."
    )
    # ... and the same question about `tagged_soundness` alone, read off its
    # own returns rather than off the file's bytes: `_rewording_hint` returns
    # `""` legitimately, and a byte scan calls that a defect.
    reader = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "tagged_soundness"
    )
    returned = [
        node.value.value
        for node in ast.walk(reader)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
    ]
    assert None in returned and "" not in returned, (
        f"`tagged_soundness` returns {returned}. It must answer `None` and "
        f"never `\"\"` when git cannot say: an empty document reads as 'the "
        f"tagged log holds no bullets', which would make every scoped bullet "
        f"a stray in exactly the environments that cannot check one."
    )
