# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Absence of evidence, offered as evidence — as a lint, where it can be one.

Four spellings of *"we looked and found nothing, so it is sound"* were
catalogued across `design/`: the **streak** (*"fourth consecutive zero-UNSOUND
round"*); the **survival** family (*"survived every attack"*, *"held"*, *"no
route to Y was reachable"*); a **suite pass count under a verification
heading**; and an **existential negative drawn from a search**. None of them
is a lie. Each is a null result, and a null result bounds what an instrument
saw, not what is true.

**THE DESIGN PROBLEM IS THAT A DATED PASS RECORD LEGITIMATELY REPORTS A
NULL**, and rewriting one destroys the thing it exists to keep. *"On
2026-08-14 this run found nothing"* is a record and must survive; *"nothing
was found, therefore X is sound"* is a conclusion and must not stand. A lint
that cannot tell them apart deletes the project's evidence base, and a lint
that fires on records is a lint everybody silences — at which point the
conclusions come back too.

So this file gates **two** of the four, and says plainly what it leaves alone.

**GATED — the STREAK.** *"fourth consecutive zero-UNSOUND round"* is not a
record and cannot be made into one, because a streak has no date and no scope
of its own: it is an AGGREGATE over null results, and aggregating nulls is
precisely the inference this lint refuses. One round's report — *"~4,000
exact-rational fuzz queries, 0 violations, 2,643 decided obligations"* — says
what ran, on what, and how much; *"the fourth in a row was clean"* says only
that nobody has been caught yet, and it gets stronger-sounding the longer the
instrument stays blunt. The construction is `<count> consecutive` beside a
word for cleanliness, which is what separates it from *"two consecutive
scalar `pow`s"* (a fact about equations) and *"wrong on FOUR CONSECUTIVE
ATTEMPTS"* (a streak of failures, which is evidence of something).

**GATED — a SUITE PASS COUNT under a heading that asserts verification.** A
section headed *Verification* or *Gates, verified* is offering its contents
AS the verification, and `Suites: 803 passed` is a null: nothing went wrong.
The project's own rule — `.github/workflows/release.yml`'s header — is that a
guard never observed to fire is not known to be a guard. A verification
section in which nothing was ever observed to fire has measured that the
lights are on. So such a section must also record something that FAILED: a
REFUTED, a constructed divergence, a mutation caught, a raised error. All
four such sections in `design/` already do, which is why this is a gate and
not a rewrite.

**NOT GATED — the SURVIVAL family, and the reason is measured, not asserted.**
*"survived"* carries both polarities in this corpus, in the same words:

* `design/solver-integration-build.md` — *"the emission core survived every
  constructed attack"*: a conclusion.
* `docs/gauge-coverage.md` — *"five wrongnesses conditioned on it survived
  all 22 gates"*, eight times over: a **defect report**, and one of the most
  valuable pages here. The mutants surviving is the finding.

The polarity lives in the SUBJECT, not the construction, and a rule that
guessed at subjects would rewrite a mutation catalogue into nonsense.
:func:`test_the_survival_family_really_is_ambiguous_in_this_corpus` measures
both polarities on every run, so this paragraph is a fact about the tree
rather than an excuse — and the day the corpus stops using the word both ways,
that test reds and this family becomes gate-able.

**NOT GATED — the EXISTENTIAL NEGATIVE from a search.** *"no caller
anywhere"*, *"grep returns 0"*. The record and the conclusion are the same
sentence with different scope, and the scope is usually in a neighbouring
clause or a paragraph away. Nothing here can read that reliably, and a rule
that tried would fire on the many places this project correctly reports a
measured zero.

**Corpus**: every `.md` under `design/` and `docs/`. Not `SOUNDNESS.md` or
`CHANGELOG.md`, which are dated event records with gates of their own.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _repo_files import text_files  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

_CORPUS_DIRECTORIES = ("design/", "docs/")

# --- the streak -------------------------------------------------------------

_COUNT = (
    r"(?:second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"two|three|four|five|six|seven|eight|nine|ten|\d+)"
)
_STREAK = re.compile(rf"(?i)\b{_COUNT}\s+consecutive\b")

#: The cleanliness half. A streak is only this lint's business when what was
#: consecutive was a NULL — a clean round, a zero-finding audit. `zero[- ]\w+`
#: covers `zero-UNSOUND`, `zero findings`, `zero violations` in one shape
#: rather than three literals.
_CLEAN = re.compile(
    r"(?i)\b(?:clean|green|quiet|passing|zero[- ][A-Za-z]+|"
    r"no[- ](?:findings?|defects?|violations?|reds?)|UNSOUND-free|"
    r"nothing found)\b"
)

#: How far either side of the streak the cleanliness word may stand. Wide
#: enough for *"(zero-UNSOUND audit, fourth consecutive; …)"*, where the word
#: precedes the count by a clause; narrow enough that it is a neighbourhood
#: and not the paragraph.
_WINDOW = 90

#: A quotation. A page may QUOTE the streak form in order to criticise it —
#: `docs/state-0.1.0.md` does, correcting an earlier report that said *"eight
#: consecutive"* — and a lint that cannot be written about is a lint that
#: cannot be documented. Same discipline as
#: `test_soundness_log_reach.py::test_the_release_record_does_not_state_the_
#: old_count_unquoted`: quoted may, asserted may not.
_QUOTED = re.compile(r"[\"“][^\"”]{0,400}[\"”]")


def _flatten(text: str) -> str:
    """Whitespace-collapsed text.

    Markdown wraps at column ~72, so *"fourth\\nconsecutive"* is one phrase
    written across two lines and a line-by-line scan does not see it. The
    first draft of this rule was line-based and missed `design/roadmap.md`
    for exactly that reason — the second of the two live instances.
    """
    return " ".join(text.split())


def streaks(text: str) -> list[str]:
    """Every clean-streak claim in one document, as its own neighbourhood."""
    flat = _flatten(text)
    quoted = [m.span() for m in _QUOTED.finditer(flat)]
    found = []
    for hit in _STREAK.finditer(flat):
        if any(lo <= hit.start() and hit.end() <= hi for lo, hi in quoted):
            continue
        window = flat[max(0, hit.start() - _WINDOW):hit.end() + _WINDOW]
        if _CLEAN.search(window):
            found.append(window)
    return found


# --- the verification section ------------------------------------------------

_HEADING = re.compile(r"^(#{2,6})\s+(?P<title>.*)$")

#: A heading that offers what follows AS the verification. Not every heading
#: over a pass count — `design/ieee-semantics.md`'s *Verdict integrity*
#: section reports one and is not gated, which is a limit and is stated here
#: rather than left to be discovered.
_ASSERTS_VERIFICATION = re.compile(r"(?i)\b(?:verification|verified|acceptance)\b")

#: A suite pass count, in the shape every one of these sections writes it.
#: Matched anywhere in a line, not at its start: `design/ieee-semantics.md`
#: writes one mid-sentence.
_SUITE_COUNT = re.compile(r"(?i)\bsuites?\b\s*\**\s*:")

#: Something the section OBSERVED TO FAIL. The vocabulary is broad on
#: purpose: the claim is not *"this section is rigorous"*, it is the much
#: weaker *"at least one thing in here went wrong and was written down"*, and
#: a narrow vocabulary would fail sections that recorded a red in words this
#: file did not anticipate. Anti-vacuity for it is
#: `test_a_verification_section_of_pass_counts_alone_is_refused`.
_OBSERVED_TO_FAIL = re.compile(
    r"(?i)\b(?:REFUTED|UNSOUND|counterexample|mutation|mutant|raise[sd]|"
    r"caught|failed|fails|failure|violation|divergence|wrong|"
    r"false\s+VERIFIED|exit\s+1|red)\b"
)


@dataclasses.dataclass(frozen=True)
class Section:
    path: str
    line: int
    title: str
    body: str


def sections(rel: str, text: str) -> list[Section]:
    """Every `##`..`######` section of one document."""
    lines = text.split("\n")
    heads = [
        (number, hit.group("title"))
        for number, line in enumerate(lines, 1)
        if (hit := _HEADING.match(line))
    ]
    out = []
    for index, (number, title) in enumerate(heads):
        end = heads[index + 1][0] - 1 if index + 1 < len(heads) else len(lines)
        out.append(Section(rel, number, title, "\n".join(lines[number:end])))
    return out


def green_only_verification_sections(documents) -> list[Section]:
    """Verification sections whose evidence is a pass count and nothing else."""
    return [
        section
        for rel, text in documents
        for section in sections(rel, text)
        if _ASSERTS_VERIFICATION.search(section.title)
        and _SUITE_COUNT.search(section.body)
        and not _OBSERVED_TO_FAIL.search(section.body)
    ]


# --- the corpus --------------------------------------------------------------


def corpus() -> list[tuple[str, str]]:
    """`(relative path, text)` for `design/` and `docs/`.

    Through `_repo_files.text_files`, so a `.venv` holding a copy of the docs
    is not linted — the environment-dependence class that put four of five
    consecutive reds on `main`.
    """
    return [
        (rel, path.read_text(encoding="utf-8"))
        for rel, path in text_files()
        if rel.endswith(".md") and rel.startswith(_CORPUS_DIRECTORIES)
    ]


def test_no_clean_streak_stands_as_a_claim():
    """The streak, gated. **RED on `main` at `115d771`, two real instances.**

    * `design/affine-refinement.md` — *"## The audit (fourth consecutive
      zero-UNSOUND round)"*, the heading over the round's own report.
    * `design/roadmap.md` — *"Landed as the opt-in `refine="affine"`
      (zero-UNSOUND audit, fourth consecutive; …)"*, where the streak is
      offered as part of the reason the item is closed.

    Neither page is this agent's to edit and the red is the report. What the
    fix looks like, so it is routable: keep the round's own measurements —
    the query count, the violation count, the decided-obligation count, the
    date — and drop the ordinal. *"~4,000 exact-rational fuzz queries, 0
    violations"* is a record with a scope; *"the fourth in a row"* is a
    number that grows while the instrument stays the same.
    """
    found = [
        (rel, window) for rel, text in corpus() for window in streaks(text)
    ]
    rendered = "\n  ".join(f"{rel}: …{window}…" for rel, window in found)
    assert not found, (
        f"{len(found)} clean-streak claim(s) in `design/` and `docs/`:\n  "
        f"{rendered}\n\n"
        f"A streak has no date and no scope. It is an aggregate over null "
        f"results — N rounds looked and none caught anything — and it reads "
        f"stronger the longer an instrument stays blunt. Report the round: "
        f"what ran, over what, and what it decided. A streak of FAILURES is "
        f"not this (nothing clean about it) and neither is `two consecutive "
        f"scalar pow`s; both are outside this rule by construction."
    )


def test_a_verification_section_records_something_that_failed():
    """A pass count is a null. Under a verification heading it is the claim.

    GREEN on this tree: all four sections in `design/` that head themselves
    *Verification* / *Gates, verified* / *Fix round — verification* and
    report a suite count also record a red — a constructed divergence, a
    mutation REFUTED, an error raised. That is what makes this a gate rather
    than a rewrite, and it is why the rule is worth having: it holds the
    convention that is already in force.
    """
    bare = green_only_verification_sections(corpus())
    rendered = "\n  ".join(f"{s.path}:{s.line}  [{s.title[:60]}]" for s in bare)
    assert not bare, (
        f"{len(bare)} section(s) head themselves as verification, report a "
        f"suite pass count, and record nothing that was observed to "
        f"fail:\n  {rendered}\n\n"
        f"`N passed` is a null result — nothing went wrong — and a section "
        f"that offers it AS the verification has measured that the lights "
        f"are on. `.github/workflows/release.yml`'s own header: a guard "
        f"never observed to fire is not known to be a guard. Record the red "
        f"as well: the constructed divergence, the mutation that was caught, "
        f"the error the wrong input raised."
    )


def test_both_gated_rules_are_driven():
    """Both rules, observed to fire and observed to stay quiet.

    The quiet halves carry as much as the loud ones. Every false positive in
    this class is a page whose author is right and whose lint is wrong, and
    two of them are enough for the rule to be switched off.
    """
    # THE STREAK, firing.
    for claim in (
        "The audit (fourth consecutive zero-UNSOUND round)",
        "a zero-UNSOUND audit, fourth consecutive; built blind",
        "the eighth consecutive clean round",
        "three consecutive green sweeps of the corpus",
    ):
        assert streaks(claim), f"the streak rule does not see {claim!r}"
    # ... and the line wrap, which is how the second real instance is written.
    assert streaks("a zero-UNSOUND audit, fourth\nconsecutive; built blind"), (
        "the streak rule is line-based again, so a phrase wrapped across two "
        "lines walks past it — which is exactly how `design/roadmap.md` "
        "writes the instance this rule exists for"
    )
    # THE STREAK, quiet. Each of these is a real sentence from this corpus.
    for quiet in (
        "no gate here holds two consecutive scalar `pow`s",
        "the refusal-message code was wrong on FOUR CONSECUTIVE ATTEMPTS",
        'earlier reports said "eight consecutive", counting runs',
        "two consecutive scalar pows, present as elements 0 and 1",
    ):
        assert not streaks(quiet), (
            f"the streak rule fires on {quiet!r}. That is a fact about "
            f"equations, or a streak of FAILURES, or a quotation criticising "
            f"the form — none of them a null offered as evidence."
        )

    # THE VERIFICATION SECTION, firing and quiet, on synthetic documents.
    green_only = "## Verification\n\n- Suites: **803 passed** (venv-jax).\n"
    with_a_red = green_only + "- A constructed divergence raised `ProvenanceError`.\n"
    assert green_only_verification_sections([("x.md", green_only)]), (
        "a verification section whose entire evidence is a pass count is not "
        "refused, so this rule is inert"
    )
    assert not green_only_verification_sections([("x.md", with_a_red)]), (
        "a verification section that DOES record a red is refused, which "
        "would make this rule a rewrite of every legitimate pass record"
    )
    narrative = "## Ledger\n\n- Suites: **803 passed** (venv-jax).\n"
    assert not green_only_verification_sections([("x.md", narrative)]), (
        "a pass count in a section that does NOT head itself as verification "
        "is refused. A pass record is a record; this rule is about a section "
        "offering one as the verification."
    )


_DEFECT_SUBJECT = re.compile(
    r"(?i)\b(?:mutation|mutations|mutant|mutants|wrongness|wrongnesses|"
    r"violation|violations|decoy|decoys)\b"
)
_SURVIVED = re.compile(r"(?i)\bsurvived\b")


def test_the_survival_family_really_is_ambiguous_in_this_corpus():
    """The reason the survival family is NOT gated, measured every run.

    *"survived"* is a soundness conclusion when its subject is the thing
    being verified and a DEFECT REPORT when its subject is a planted mutant.
    Both are in this corpus, in the same words, and the module docstring says
    so — this is what makes that a fact rather than an excuse.

    It is a live measurement and not a comment because of what happens if it
    stops being true: if the defect-report polarity left the corpus, the
    family would become separable and the decision not to gate it should be
    revisited. A red here is that news.
    """
    documents = corpus()
    assert documents, "the corpus is empty"
    sentences = [
        (rel, part.strip())
        for rel, text in documents
        for part in re.split(r"(?<=[.!?|])\s+", _flatten(text))
        if _SURVIVED.search(part)
    ]
    as_defect = [(rel, s) for rel, s in sentences if _DEFECT_SUBJECT.search(s)]
    as_conclusion = [(rel, s) for rel, s in sentences if not _DEFECT_SUBJECT.search(s)]
    assert as_defect and as_conclusion, (
        f"`survived` no longer carries both polarities in `design/` and "
        f"`docs/`: {len(as_defect)} defect-report use(s), "
        f"{len(as_conclusion)} conclusion-shaped use(s). This module leaves "
        f"the survival family UNGATED because the polarity lives in the "
        f"subject and a rule would rewrite a mutation catalogue into "
        f"nonsense. If one polarity has gone, that reason is out of date."
    )
    # ... and the two live in DIFFERENT pages, which is what makes a
    # file-level exclusion no answer either.
    assert {rel for rel, _ in as_defect} & {rel for rel, _ in as_conclusion} or (
        len({rel for rel, _ in as_defect}) > 1
    ), "the ambiguity is confined to one page and could be excluded by file"


@pytest.mark.parametrize(
    "family", ["SURVIVAL family", "EXISTENTIAL NEGATIVE"]
)
def test_the_uncovered_families_are_named_in_this_module(family):
    """A gate must say what it does not cover, and be held to saying it.

    Two of the four catalogued spellings are outside this file. A green here
    means *"no clean streak and no green-only verification section"* and
    nothing more, and the way that gets forgotten is the docstring quietly
    losing the paragraph that says so.
    """
    doc = __doc__ or ""
    assert family in doc, (
        f"this module's docstring no longer names {family!r} as uncovered. "
        f"Two of the four catalogued spellings are not gated here; a reader "
        f"who takes a green for all four is reading a claim this file does "
        f"not make."
    )
    assert "NOT GATED" in doc
