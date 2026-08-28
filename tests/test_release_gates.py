# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""`release.yml`'s refusal points, pinned — because NOTHING read that file.

THE MEASUREMENT THAT PUT THIS HERE. Before it, exactly one test in the
repository opened a workflow file — `tests/test_reuse_pins.py`, and only
`ci.yml`. `release.yml` is the one workflow whose mistakes are immutable, and
no test, hook or lint read a byte of it. Six line-neutral mutations were
applied to it AT ONCE and the full suite was run on the mutated tree
(CPython 3.12.3, jax 0.11.0, `JAX_ENABLE_X64=1`): every one survived, with the
suite green and the skip-inventory verdict `made`. The six:

1. delete `rm -f "${RUNNER_TEMP}/verdict.txt"` — the line the file argues earns
   its place, and without which the step reads an EARLIER run's verdict;
2. `-ra` -> `-rs` — same exit code, and a red publish log that no longer names
   what failed;
3. the `verdict=made` condition made always-true — the only check that can see
   a NARROWED session;
4. `publish: needs: [test, build]` -> `[build]` — the suite stops gating the
   upload while still running, so a red suite and a green publish;
5. the header's `EIGHT` -> `TWELVE` — a count of refusals nobody can check
   (`release.yml` says FOURTEEN now and MEANS it: three refusals landed when
   the tag comparison was carried to the sdist, a fourth when the two counts
   stopped being mistaken for a statement about the DIRECTORY, and a fifth when
   the sdist step stopped dying on `tar`'s own exit code. That is
   not mutation 5 surviving —
   :func:`test_the_headers_refusal_COUNT_is_the_count_of_refusals` derives the
   number from the file's own `exit 1` sites, so the word and the arithmetic
   are checked against each other whatever the word is);
6. `comm -23` -> `comm -12` — the "every sdist member is committed" comparison
   inverted, so it reports the INTERSECTION, is empty exactly when the tree is
   healthy, and can never fire.

AND THE COUNT IS SPELLED THREE TIMES IN THAT FILE, NOT ONCE. The header states
it; two other sentences are DERIVED from it — "SEVEN OF THE <N> WERE DRIVEN",
where the SEVEN counts drives and the N is the header's count, and "rather than
adding a <Nth>", which is the header's count PLUS ONE. Both followed the header
in lockstep through EIGHT, NINE, TWELVE and THIRTEEN, and both went stale in
the one commit that took it to FOURTEEN: `release.yml` then stated its own
refusal count as two different numbers, and gave the name "THE FOURTEENTH" both
to the `tar` refusal that WAS added and to a refusal that deliberately was NOT.
Nothing here saw either — set to `SEVEN OF THE NINETY` and `adding a
NINETIETH`, this module was 21 passed.
:func:`test_the_counts_DERIVED_from_the_header_move_with_it` holds all three to
each other now: red on the commit before this one, naming BOTH sentences, and
green on the two before that, which are the last commits at which the file was
consistent.

WHAT THIS FILE IS. Two things, and the second was called IMPOSSIBLE here for
four commits.

The first is a TEXT PIN, in the same shape as `tests/test_reuse_pins.py`: it
reads the file and asserts the load-bearing literals are still there. That is
enough to kill all six mutations above, and it is measured against them below
rather than argued.

The second is a DRIVE. FOUR gate bodies — the tag step, the sdist step, the
manifest step and the changelog-heading step — are EXTRACTED from
`release.yml` and EXECUTED here against planted trees, and asserted on their
exit code, their annotation and, for the one that only prints, on the record
it writes. Extracted, not copied: a test that runs its own transcription of a
step body is a test of the transcription, and the rewrite it has to catch
lands in `release.yml`.

THIS PARAGRAPH SAID **THREE** AND NAMED THREE STEPS, AND IT WAS TRUE UNTIL
THE FIFTEENTH REFUSAL LANDED. The changelog-heading step is the fourth, and
what it plants is not a `dist/` — it is a git checkout carrying an ANNOTATED
TAG with a chosen tagger date, which is one more thing a test can plant and a
runner does not have to supply. Corrected here rather than left to be
inferred from the count of `_step_body` calls below.

AND THAT DESCRIPTION OF THE PLANT WAS ITSELF TOO SMALL, WHICH COST A RELEASE.
A checkout carrying a tag is not what the release job has: the job has a
checkout `actions/checkout@v4` BUILT, and for a release event that action
fetches every ref and then force-writes `+<github.sha>:refs/tags/<tag>`,
replacing the annotated tag ref with the release commit. `v0.2.1` was refused
by the changelog step on 2026-08-28 for being "a commit object and not an
annotated tag" when the tag on `origin` was a perfectly good annotated one,
and the recovery text told the maintainer to delete it. Nothing here could see
that, because every drive planted the tag with `git tag -a` and read it back —
which is the tag, and not what the checkout leaves. So the plant is bigger by
one thing: :func:`_checkout_the_way_actions_checkout_does` materialises a
workspace the way the action does, over a local path remote, with no network,
in each of the shapes the workflow's own `with:` block selects between. That
is the third thing a test can plant and a runner does not have to supply: not
a tag, but the shape a tag arrives in.

AND THIS PARAGRAPH SAID *"every planted tree has an `origin` (`_tagged_tree`)"*
AS IF THAT WERE HALF THE POINT. It is not: those trees' tag refs are intact,
so no resolution route decides anything in them, and the remote is there for
the REMEDY drive, which pushes. What decides the tag-reading rows is the
checkout fixture and the workflow input it models.

WHAT THAT FIXTURE DOES NOT MODEL, said here because it is the same sentence
that was too small before: it re-derives the action's refspecs, not the
action. No submodules, no LFS, no `persist-credentials` handling, no sparse
checkout, and its `origin` is a directory rather than an authenticated https
remote. It reproduces exactly one behaviour — which ref name ends up pointing
at which object — and the drive asserts that behaviour is present before
relying on it, so a git that stopped force-writing the ref is a red here
rather than a green for the wrong reason.

THE SENTENCE THAT USED TO STAND HERE WAS FALSE, and how it was false is worth
writing down, because it is what licensed the gap. It read: this is "NOT a
check that the gates WORK — nothing in this repository can be, because a
gate's behaviour is a property of a runner". The premise does not hold for
these four bodies. None of them contains one thing a runner supplies: they
are `tar`, `git ls-files`, `git for-each-ref`, `sort`, `comm`, `basename`,
`cut` and `mktemp`, over a `dist/` directory, a git index and a git TAG
OBJECT, all of which a test can plant. (The manifest body wants
`GITHUB_STEP_SUMMARY` and the changelog body wants `GITHUB_REF_NAME`, which
are two environment variables and a file this test writes.) THAT LIST READ
"these three bodies" AND STOPPED AT `mktemp` until the changelog step
arrived; a list of what a body needs is exactly the kind of sentence a fourth
body falsifies in silence. And `release.yml`'s own comments record driving
these same bodies by hand at
a61c01f — so the file already knew they were drivable, and the impossibility
claim was contradicted a few lines from where it was written.

WHAT IT COST, measured rather than asserted, because a false impossibility
claim is only worth correcting if something got through it. Two rewrites, one
line each, each leaving every text pin in this file green:

* `sort -u tracked.txt generated.txt > explained.txt` ->
  `sort -u members.txt generated.txt > explained.txt`, one line above the
  pinned `comm -23`. `explained.txt` then contains every member, so the
  comparison is empty BY CONSTRUCTION and the gate cannot fire at all. Driven,
  with an uncommitted file planted inside the tarball: unmutated refuses
  (rc=1, the member named); mutated reports "every one of N sdist members is
  committed to this tree" and exits 0, with the uncommitted file still in the
  tarball.
* `version="$(basename "${wheel}" | cut -d- -f2)"` ->
  `version="$(echo "${TAG}" | sed s/^v//)"`, which reads the version out of
  the tag it exists to check the tag AGAINST. Driven: `TAG=v9.9.9` against a
  `stelling-0.1.0-py3-none-any.whl` — unmutated rc=1, mutated rc=0. No pin of
  any kind stood on this line; the refusal point was unguarded outright.

A pin catches the mutation it names. A drive catches rewrites nobody thought to
name — but ONLY the ones its planted tree can express, and that bound is real
rather than theoretical. THIS SENTENCE USED TO END "the rewrite nobody thought
to name, which is the only kind that ships", and the sdist drive under it
planted one uncommitted file, under `src/`. Twelve characters added to the
step's own `sed` — `-e '/^docs\\//d'` — delete an entire allowlisted root from
`members.txt` before the comparison sees it, and at 461b2d5 that left the full
suite at 1430 / 0 / 0 / 94, every text pin here green, and
`test_the_drives_are_reading_the_real_step_bodies` green, while a hand-driven
plant at `docs/internal-release-checklist.md` went from rc=1 naming the file to
rc=0 reporting "every one of 5 sdist members is committed to this tree" with
the file still in the tarball.

So the sdist plant now covers one uncommitted file under EVERY directory root
of the allowlist, derived from `pyproject.toml`, with the suffix cycled across
them. That closes filters keyed on a ROOT, which is how an area gets exempted
in one line. It does not close a filter keyed on something else — a single
path, a size, a member count — and no plant closes the class outright: the
drive can only refute rewrites that change what happens to a file it thought
to put in the tarball.

STILL UNGUARDED AFTER THIS, said plainly rather than left to be assumed, and
now a shorter list than the argument that used to stand for it:

* the OTHER gate bodies. `-ra` being present still says nothing about what
  pytest prints. The two verdict refusals are pinned as literals and, below,
  as a PATH-COHERENCE check — but the recorder itself is not driven here.
  What is driven is the three `dist/`-reading bodies and the changelog step,
  and nothing else. (This bullet said "the three `dist/`-reading bodies and
  nothing else" while three was the whole list; the changelog step reads no
  `dist/` at all, so the phrase that used to be a complete description is now
  only half of one.)
* the sdist drive's plant is keyed on the allowlist's DIRECTORY roots, so a
  member filter keyed on anything else is outside it, as the paragraph above
  says.
* WHETHER `actions/checkout@v4` STILL DOES WHAT THE CHECKOUT FIXTURE SAYS IT
  DOES. `_checkout_the_way_actions_checkout_does` is a re-derivation of the
  action's refspecs from the behaviour they produce, taken 2026-08-28 against
  a mirror of this repository and against the tag that was refused; nothing
  here reads the action itself, and `@v4` is a MUTABLE ref — the workflow
  pins a major series, not a commit. If a future v4 stops force-writing the
  tag ref, this fixture
  keeps reproducing the old shape and the drives stay green over a workspace
  the job no longer gets. That is the safe direction — the step's re-fetch is
  correct either way, since it re-reads the ref from `origin` regardless of
  what the checkout left — but the fixture would be modelling history, and
  this bullet is where a reader finds that out.
* the `publish` job's two actions, which cannot be driven from here at all
  without cutting a tag and uploading. What IS now pinned is the ref one of
  them carries, and the environment the job runs in — see
  `test_the_attributes_that_decide_whether_a_refusal_refuses`, which exists
  because every other pin here reads a literal inside a `run:` block or
  `needs:`, and the three attributes BESIDE those blocks that decide whether a
  refusal refuses were read by nothing. Applied together they left the full
  suite at 1433 / 0 / 0 / 94. Pinned as literals, because a runner is what it
  would take to drive them.
* everything about the runner: which interpreter `uv venv` picks, whether the
  `pypi` environment exists, whether Trusted Publishing is configured. This is
  the part of the old sentence that was true, and it is true of the runner's
  own furniture rather than of gate behaviour in general.

KNOWN-OPEN, AND IT IS A PLACEMENT PROBLEM RATHER THAN A COVERAGE ONE.
`design/ci-readiness.md` is the document a release reviewer opens, and it says
nothing about the declared floor, nothing about the fact that no job runs the
floor interpreter, and nothing about these release gates being text pins over
a workflow. So the facts recorded here are recorded where somebody already
looking at `release.yml` will find them, and nowhere a reviewer deciding
whether to release would look. That document is owned elsewhere and is not
edited from here; the gap is recorded in this header so that it is at least
written down on the surface it concerns.

Read as TEXT and not with a YAML parser, deliberately: `yaml` is not a
dependency of this project, and the zero-dep CI job — the one whose whole
purpose is an environment with nothing in it — could not import one.
"""
from __future__ import annotations

import gzip
import io
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RELEASE = REPO / ".github" / "workflows" / "release.yml"

# The header's own count, spelled as a word. Only the words a refusal count
# could plausibly be; an unlisted word fails loudly rather than silently
# reading as zero.
_NUMBER_WORDS = {
    "ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
    "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10, "ELEVEN": 11,
    "TWELVE": 12, "THIRTEEN": 13, "FOURTEEN": 14, "FIFTEEN": 15,
    "SIXTEEN": 16, "SEVENTEEN": 17, "EIGHTEEN": 18, "NINETEEN": 19,
    "TWENTY": 20,
}

# The ordinal spellings, for the one sentence in `release.yml` that names the
# refusal it did NOT add. Same discipline as the cardinals above: an unlisted
# word fails LOUDLY rather than reading as zero, which is what an ordinal
# rewritten to a value nobody checked would otherwise do.
_ORDINAL_WORDS = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5,
    "SIXTH": 6, "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10,
    "ELEVENTH": 11, "TWELFTH": 12, "THIRTEENTH": 13, "FOURTEENTH": 14,
    "FIFTEENTH": 15, "SIXTEENTH": 16, "SEVENTEENTH": 17, "EIGHTEENTH": 18,
    "NINETEENTH": 19, "TWENTIETH": 20,
}

# The three sentences in `release.yml` that spell the refusal count. The first
# IS the count; the other two are DERIVED from it and had both gone stale by
# the time this was written. Named here so the line-ending test drives the same
# patterns the pins do, and so no pattern is written twice.
#
# FOUR PATTERNS OVER THREE SENTENCES, because the first sentence makes TWO
# claims on one line — "FOURTEEN REFUSAL POINTS … — twelve `exit 1` sites" —
# and a pattern `_one_word` reads may carry exactly one group. The opening of
# that sentence is therefore written ONCE, in `_HEADER_OPENING`, and both of
# its patterns are composed from it: a reworded header must not be a place
# where one of the two is updated and the other quietly stops matching.
_HEADER_OPENING = "# {} REFUSAL POINTS STAND BETWEEN A TAG AND PyPI"
_HEADER_COUNT_RE = "^" + _HEADER_OPENING.format("([A-Z]+)")
# THE SAME SENTENCE'S BREAKDOWN of that total into `exit 1` sites, and it is
# an anchored scan because for one commit it was not one. The count pin's
# second half was `"twelve `exit 1` sites" in text` — a bare substring over
# the whole file, where every other scan in this module is `^`-anchored and
# exactly-once — so ANY occurrence of that phrase anywhere in `release.yml`
# satisfied it. MEASURED on the committed blob before this was written: the
# phrase planted once in a comment paragraph below, the header's own breakdown
# mutated `twelve` -> `eleven`, and the pin PASSED; the same mutation without
# the plant is red. Anchored to the sentence it is a claim ABOUT, prose
# elsewhere in that file can neither satisfy it nor falsify it — which is what
# retired the "do not respell the phrase" rule `release.yml` used to carry in
# place of an instrument.
_HEADER_BREAKDOWN_RE = (
    "^" + _HEADER_OPENING.format("[A-Z]+") + r" — ([a-z]+) `exit 1` sites"
)
_DRIVEN_OF_RE = r"^# SEVEN OF THE ([A-Z]+) WERE DRIVEN IN BOTH DIRECTIONS"
# NARROW ON PURPOSE, AND THE REASON THAT STOOD HERE WAS HALF TRUE. It read:
# "No `.*` in this one, deliberately: `.` matches a carriage return, so a
# wildcard here would see the CR-only rendering that
# `test_the_release_gates_READ_ONE_FILE_however_its_lines_end` requires every
# scan in this module to be blind to." MEASURED on this tree's `release.yml`,
# CR-only rendering, `re.M`: this pattern with a TRAILING `.*` appended is
# still ZERO hits — the negative control is NOT broken — and so are the other
# two. In a CR-only rendering there is no `\n` at all, so `^` matches only at
# offset 0, and each pattern's own anchored literal is what refuses it there; a
# wildcard placed AFTER that literal never gets the chance to matter. What
# breaks the control is a `.*` BETWEEN `^#` AND THE DISTINGUISHING LITERAL:
# `^#.*adding a ([A-Z]+)` reads ONE hit where zero is required, because `.`
# matches `\r` and the whole file is then a single line beginning at offset 0.
# So the true claim is about the MID-PATTERN position and not about the
# position a wildcard would naturally occupy.
#
# AND THE STRONGER REASON FOR NARROWNESS IS UNIQUENESS, which is red on the
# NORMALISED file before the CR question is reached at all: `release.yml`
# QUOTES ITS OWN COUNTER-EXAMPLE. `^#.*SEVEN OF THE ([A-Z]+)` finds TWO hits
# today — `FOURTEEN` in the sentence that IS the claim, and `NINETY` in the
# sentence recording that `SEVEN OF THE NINETY` once went unnoticed — and
# `_one_word` demands exactly one, so a loose pattern fails on the file as it
# stands. That is the shape to expect from these three sentences in
# particular: their subject is the sentences themselves, so the file is full of
# retracted and hypothetical spellings of each. All FOUR patterns in this
# group are anchored to their own literal for that reason, and the CR argument
# is the second one.
#
# THE BREAKDOWN PATTERN IS THE SAME STORY, MEASURED ON THE FILE THIS COMMIT
# SHIPS. Without the anchor, "([a-z]+) `exit 1` sites" finds SIX hits —
# `twelve`, `the`, `of`, `twelve`, `own`, `five` — of which exactly one is the
# clause that IS the claim; the rest are paragraphs ABOUT the claim, the
# hazard note's own quotation of it, and a step comment counting that one
# step's refusals. The phrase is not rare in that file and never was, which is
# why a substring test over it was not a test of the header. And the
# mid-pattern wildcard is red on the CR control here too: "^#.*— ([a-z]+)
# `exit 1` sites" reads ONE hit in the CR-only rendering where zero is
# required, while the anchored form above reads none.
_DECLINED_ORDINAL_RE = (
    r"^# therefore refuse through refusal point 1 rather than adding a "
    r"([A-Z]+)"
)

# The two refusals that are not `exit 1` sites: a step that refuses by its own
# command's exit code. `release.yml` names them — the pytest step and
# `python -m stelling`. NOT derived, and said so: deciding mechanically which
# `- run:` can fail is deciding what every command in the file does.
_IMPLICIT_REFUSALS = 2


def _one_line_break(text: str) -> str:
    """``text`` with every line break spelled ``\\n`` -- done HERE.

    EVERY SCAN IN THIS MODULE IS `^`-ANCHORED UNDER `re.M`, and Python's
    `^` there matches after a newline and NOT after a carriage return.
    That is the defect `tests/test_tripwire_record.py` met three times --
    `^---`, `^\\s*needs:` and the shell reader's notion of a line -- and it
    is latent here for exactly the reason it was latent there: `read_text`
    happens to translate the file's line breaks on the way in, so these
    checks are correct because of how the file was OPENED and not because
    of anything they do.

    Measured on this tree, the same `release.yml` rendered with CR line
    endings and read without translation: the verdict recorder, the
    `rm -f` of the verdict path and the `verdict=` assignment each go from
    one hit to ZERO, and the refusal-count header from `EIGHT` to nothing.
    Those four fail LOUDLY. `tests/test_reuse_pins.py` carries two
    `re.sub`s of the same shape that fail quietly, leaving the pin lines
    standing in what that file then calls prose.

    So the file is opened with `newline=""` and normalised here, which
    makes the property this function's rather than the loader's --
    `_lines_of_this_grammar` in the tripwire record made the same move for
    the same reason.
    `test_the_release_gates_READ_ONE_FILE_however_its_lines_end` holds it,
    with the untranslated rendering as its negative control.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _release_text() -> str:
    assert RELEASE.is_file(), f"no release workflow at {RELEASE}"
    # `newline=""` so nothing is translated on the way in and the
    # normalisation is visibly this reader's. See `_one_line_break`.
    with RELEASE.open(encoding="utf-8", newline="") as handle:
        return _one_line_break(handle.read())


#: The `^`-anchored scans this module makes over `release.yml`, each of
#: which must find exactly one thing. Named here so the line-ending test
#: below drives the same patterns the tests do rather than a copy of them.
_ANCHORED_SCANS = (
    ("the verdict recorder",
     r"^\s*STELLING_SKIP_INVENTORY_VERDICT:\s*(.+?)\s*$"),
    ("the `rm -f` of the verdict path", r'^\s*rm -f "([^"]+)"\s*$'),
    ("the `verdict=` assignment", r'^\s*verdict="([^"]+)"\s*$'),
    ("the refusal-count header", _HEADER_COUNT_RE),
    ("the header's breakdown into `exit 1` sites", _HEADER_BREAKDOWN_RE),
    ("the count the drives are seven OF", _DRIVEN_OF_RE),
    ("the ordinal of the refusal NOT added", _DECLINED_ORDINAL_RE),
)


def test_the_release_gates_READ_ONE_FILE_however_its_lines_end():
    """A carriage return is a line break, and `re.M`'s `^` does not know it.

    Every pin in this module is `^`-anchored under `re.M`, and until
    2026-08-22 every one of them was correct only because `read_text()`
    translated the file's line breaks before they ran. That is a property
    of the loader and not of the check, and it is the accident
    `tests/test_tripwire_record.py::_lines_of_this_grammar` refused to
    leave standing for the workflow grammar after `^---` missed a document
    marker opened by a CR.

    Both directions are driven: the CRLF and CR-only renderings of this
    repository's own `release.yml` read back to the same {n} hits -- one per
    entry of :data:`_ANCHORED_SCANS` -- and the CR-only rendering with its
    breaks LEFT ALONE is invisible to all {n}, which is what the normalisation
    is worth.

    THAT NUMBER IS INTERPOLATED FROM `len(_ANCHORED_SCANS)` -- done just below
    this function -- AND IS NOT TYPED HERE, because it had already stopped
    tracking what it is derived from. It read "four" while the tuple held SIX:
    it was written when the tuple held four and was left behind by the commit
    that grew it to six, which is precisely the defect
    :func:`test_the_counts_DERIVED_from_the_header_move_with_it` exists to
    catch in `release.yml` -- committed in the docstring of a test in the
    module that catches it. A count derived from a collection is written down
    once, in the collection.
    """
    text = _release_text()
    assert "\r" not in text, (
        "`_release_text` handed back a carriage return, so the "
        "normalisation it exists to do did not happen"
    )
    wanted = {name: re.findall(pattern, text, re.M)
              for name, pattern in _ANCHORED_SCANS}
    # THE DICT IS KEYED ON THE NAME, so two entries sharing one would collapse
    # into a single key and a scan would go undriven while every assertion
    # below stayed green. Held to the tuple's own length, which is also where
    # this test's docstring gets its count.
    assert len(wanted) == len(_ANCHORED_SCANS), (
        f"two entries of `_ANCHORED_SCANS` share a name, so "
        f"{len(_ANCHORED_SCANS) - len(wanted)} of the "
        f"{len(_ANCHORED_SCANS)} scans this test claims to drive are not "
        f"being driven at all: {[name for name, _ in _ANCHORED_SCANS]}"
    )
    assert all(len(hits) == 1 for hits in wanted.values()), (
        f"one of this module's anchored scans no longer finds exactly one "
        f"thing in `release.yml`, so the test below would watch nothing: "
        f"{wanted}"
    )

    for label, rendered in (("CRLF", text.replace("\n", "\r\n")),
                            ("CR only", text.replace("\n", "\r"))):
        normalised = _one_line_break(rendered)
        assert normalised == text, (
            f"a {label} rendering of `release.yml` does not normalise back "
            f"to the text every check in this module reads"
        )
        got = {name: re.findall(pattern, normalised, re.M)
               for name, pattern in _ANCHORED_SCANS}
        assert got == wanted, f"{label}: {got} where {wanted} was read"

    # THE NEGATIVE CONTROL, and it is what makes `_one_line_break`
    # load-bearing rather than decorative: the same scans over the same
    # bytes with their line breaks left as the file spells them.
    untranslated = text.replace("\n", "\r")
    blind = {name: re.findall(pattern, untranslated, re.M)
             for name, pattern in _ANCHORED_SCANS}
    assert not any(blind.values()), (
        f"a CR-only `release.yml` was expected to be invisible to every "
        f"`^`-anchored scan in this module and some of them saw it: "
        f"{blind}. If `re.M` has changed, say so where `_one_line_break` "
        f"explains why it exists"
    )


# THE ONE COUNT IN THAT DOCSTRING, DERIVED. `str.replace` rather than
# `str.format` so a future brace in the prose is not a formatting error, and
# guarded because `python -OO` strips docstrings to `None`.
if test_the_release_gates_READ_ONE_FILE_however_its_lines_end.__doc__:
    test_the_release_gates_READ_ONE_FILE_however_its_lines_end.__doc__ = (
        test_the_release_gates_READ_ONE_FILE_however_its_lines_end.__doc__
        .replace("{n}", str(len(_ANCHORED_SCANS)))
    )


def _code_lines(text: str) -> list[str]:
    """Lines that are not comments — the mutations all landed in these."""
    return [
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ]


def test_the_three_verdict_paths_are_one_path():
    """THE PIN BELOW SURVIVES A REWRITE THAT DESTROYS WHAT IT PROTECTS, so the
    identity of the path is asserted here and not only the presence of the line.

    Three places in `release.yml` name the verdict file, and the signal the two
    refusals buy is ABSENCE — which is a signal only if the path cleared is the
    path written and the path read. Nothing tied them together:

      1. `STELLING_SKIP_INVENTORY_VERDICT:` — where `tests/conftest.py` WRITES;
      2. `rm -f "…"`                        — what is CLEARED before the suite;
      3. `verdict="…"`                       — what the two refusals READ.

    DRIVEN, and this is the rewrite that motivated the test rather than a shape
    imagined for it: point (1) and point (3) moved together to
    `/tmp/stelling-verdict.txt`, leaving `rm -f "${RUNNER_TEMP}/verdict.txt"`
    untouched and still ahead of pytest. Every assertion in
    :func:`test_the_verdict_path_is_cleared_before_the_suite_runs` stays green —
    the literal is there, the order is right — while the file the recorder
    writes and the refusals read is one nothing on the runner clears, so an
    earlier run's verdict satisfies the check the `rm` exists to make honest.

    `${{ runner.temp }}` and `${RUNNER_TEMP}` are the same directory spelled
    for the two languages in this file (workflow expression, shell), so they
    are normalised to one token before comparison. Nothing else is normalised:
    the point is that the three agree literally.
    """
    text = _release_text()
    # to end of line, NOT to whitespace: the workflow-expression spelling is
    # `${{ runner.temp }}/verdict.txt`, and spaces live inside it
    recorder = re.findall(
        r"^\s*STELLING_SKIP_INVENTORY_VERDICT:\s*(.+?)\s*$", text, re.M
    )
    cleared = re.findall(r'^\s*rm -f "([^"]+)"\s*$', text, re.M)
    read = re.findall(r'^\s*verdict="([^"]+)"\s*$', text, re.M)

    assert len(recorder) == 1, f"expected one recorder path, got {recorder}"
    assert len(cleared) == 1, f"expected one `rm -f` of a verdict path, got {cleared}"
    assert len(read) == 1, f"expected one `verdict=` assignment, got {read}"

    def _norm(p: str) -> str:
        return re.sub(r"\$\{\{\s*runner\.temp\s*\}\}", "${RUNNER_TEMP}", p)

    paths = {
        "the recorder writes (STELLING_SKIP_INVENTORY_VERDICT)": _norm(recorder[0]),
        "the step clears (rm -f)": _norm(cleared[0]),
        "the refusals read (verdict=)": _norm(read[0]),
    }
    assert len(set(paths.values())) == 1, (
        "the verdict file is named at three points in `release.yml` and they no "
        "longer agree, so the ABSENCE the two refusals treat as a signal is "
        "absence at a path something else may have written:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in paths.items())
    )


def test_the_verdict_path_is_cleared_before_the_suite_runs():
    """MUTANT 1. ABSENCE is the signal the verdict check buys, and it is only a
    signal if nothing else could have left a file at that path.

    The order matters as much as the line: an `rm` AFTER the suite clears the
    file the check is about. So both are asserted, and against the same step.
    """
    lines = _code_lines(_release_text())
    rm = [i for i, line in enumerate(lines) if line.strip() == 'rm -f "${RUNNER_TEMP}/verdict.txt"']
    pytest_run = [
        i for i, line in enumerate(lines)
        if line.strip().startswith(".venv/bin/python -m pytest")
    ]
    assert rm, (
        "`release.yml` no longer clears ${RUNNER_TEMP}/verdict.txt before the "
        "suite. Driven in that state: with a verdict file already at the path "
        "and a pytest that writes none, the step exits 0 and `cat`s an EARLIER "
        "run's verdict into the publish log."
    )
    assert pytest_run, "the release suite step no longer invokes pytest"
    assert min(rm) < min(pytest_run), (
        "the `rm` now comes AFTER the suite, which clears the very file the "
        "verdict checks are about"
    )


def test_the_suite_step_names_what_failed():
    """MUTANT 2. `-ra`, not `-rs`. Same exit code either way, so nothing about
    the job's colour changes — what changes is that the publish log of a red
    suite no longer prints `FAILED <nodeid>`."""
    lines = _code_lines(_release_text())
    invocations = [
        line.strip() for line in lines
        if line.strip().startswith(".venv/bin/python -m pytest")
    ]
    assert invocations == [".venv/bin/python -m pytest -q -ra"], (
        f"the release suite step's pytest invocation has changed: {invocations}. "
        "`-rs` prints the SKIPPED lines and omits `FAILED`, and both exit 1, so "
        "a red publish log stops naming what failed with nothing going red."
    )


def test_the_two_verdict_refusals_are_still_conditions_on_the_verdict():
    """MUTANT 3. The `verdict=made` check is the ONLY thing in the workflow
    that can see a NARROWED session — pytest exits 0 on `12 passed` and the
    recorder writes `verdict=withdrawn`. A condition made always-true leaves
    the step green, the `cat` still printing, and the refusal gone."""
    text = _release_text()
    assert 'if [ ! -f "${verdict}" ]; then' in text, (
        "the 'no verdict file was written' refusal is no longer a test of the "
        "file's existence"
    )
    assert """if ! head -1 "${verdict}" | grep -qx 'verdict=made'; then""" in text, (
        "the 'the completeness claim was not made' refusal no longer reads the "
        "verdict. `grep -qx` is exact-line-anchored on purpose: `grep -q` would "
        "match `verdict=made-not` and any line further down the file."
    )


def test_the_publish_job_still_needs_the_suite():
    """MUTANT 4, and the one with no local symptom at all. Dropping `test`
    from `needs` leaves the job running and reporting — a red suite and a green
    publish, in the same workflow run."""
    lines = _code_lines(_release_text())
    needs = [line.strip() for line in lines if line.strip().startswith("needs:")]
    assert "needs: [test, build]" in needs, (
        f"the publish job's `needs` has changed: {needs}. `needs: [build]` "
        "publishes a tree whose suite went red; the suite still RUNS, and "
        "still reports, so nothing about the run looks different."
    )
    assert "needs: test" in needs, "the build job no longer needs the suite either"


def test_the_attributes_that_decide_whether_a_refusal_refuses():
    """Not what a step RUNS — what the workflow does with the result.

    Every pin above reads a literal inside a `run:` block, or `needs:`. None of
    them reads the step and job ATTRIBUTES beside those blocks, and three of
    those decide whether a refusal refuses at all. Applied TOGETHER to
    `release.yml` at cc5ce89, one full suite run, CPython 3.11.15,
    `JAX_ENABLE_X64=1`, `-p no:randomly`, figures from `--junitxml`:

        `continue-on-error: true` on the pytest step
        `environment: pypi`   -> `environment: staging`
        `@release/v1`         -> `@main` on the publish action

        the full suite        tests=1433 failures=0 errors=0 skipped=94 — GREEN

    What each costs. `continue-on-error` on the step this file's own header
    calls a refusal point makes the job succeed on a red suite — the suite
    still runs and still reports, exactly the no-local-symptom shape MUTANT 4
    had. `environment:` is what binds Trusted Publishing: the publish job's
    OIDC subject includes the environment name, so a different one is a
    different identity and the upload this workflow is configured for is not
    the upload it performs. And a branch ref on the publisher is arbitrary
    future code holding an id-token on the path to PyPI — `release/v1` is
    already a branch and the file argues that where it stands is a deliberate
    choice; `@main` is not that choice.

    A LITERAL PIN, deliberately, and the same shape as the rest of this file:
    these are not drivable here (they need a runner, a tag and an upload), so
    what is available is that the attribute still says what it said. The
    comment block above the publish job records the SHA-pinning migration this
    would interact with; that is a decision left where it is written.
    """
    lines = _code_lines(_release_text())
    stripped = [line.strip() for line in lines]

    assert not [line for line in stripped if line.startswith("continue-on-error")], (
        "a step in `release.yml` now carries `continue-on-error`. On the "
        "pytest step it makes the job SUCCEED on a red suite while the suite "
        "still runs and still reports, so nothing about the run looks "
        "different — and `needs: [test, build]` then gates on a job that "
        "cannot fail."
    )
    assert "environment: pypi" in stripped, (
        "the publish job's `environment:` has changed. It is what binds "
        "Trusted Publishing — the OIDC subject carries the environment name, "
        "so a different name is a different identity to PyPI."
    )
    publishers = [line for line in stripped if "gh-action-pypi-publish@" in line]
    assert publishers == ["- uses: pypa/gh-action-pypi-publish@release/v1"], (
        f"the publish action's ref has changed: {publishers}. This is the one "
        "action in this file that holds an `id-token` on the path to PyPI, "
        "and its ref decides which code does. `@main` is arbitrary future code "
        "with that token."
    )


#: A `uses:` line naming the checkout action, with the optional sequence dash
#: and optional quoting around the value. This is YAML STRUCTURE — an action
#: can only enter a workflow through a `uses:` key — and not a spelling of how
#: a step happens to be written.
_USES_CHECKOUT = re.compile(r"""(?:-\s+)?uses:\s*["']?actions/checkout@""")


def _unquote(value: str) -> str:
    """`value` without one layer of surrounding quotes."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _checkout_steps() -> list[tuple[dict[str, str], dict[str, str]]]:
    """Every `actions/checkout@` step in `release.yml`, READ AS STRUCTURE.

    **THIS REPLACED TWO SCANS THAT KEYED ON A SPELLING, AND THE SPELLING WAS
    THE HOLE.** Both said `line.strip().startswith("- uses: actions/checkout@")`
    — so a step written in the ordinary alternative order, `- name:` first and
    `uses:` second, was invisible to them; and because each then asserted only
    that it had found *some* checkout, a file mixing the two orders was
    silently half-checked. Driven: the `build` checkout rewritten `- name:`
    first with BOTH `ref:` and `fetch-depth:` deleted left thirteen modules
    that read `release.yml` at `240 passed, 1 skipped`. A second shape was
    invisible the same way — `ref: ${{ github.sha }}` written at STEP level
    rather than inside `with:`, which satisfies a substring scan while the
    action never receives the input.

    So steps are located by their POSITION — block-sequence items under a
    `steps:` key — and their keys by INDENTATION, which is what makes key
    order irrelevant. Each result is `(step keys, the step's `with:` mapping)`,
    values unquoted once.

    **WHAT IT CANNOT SEE, and each of these is a LOUD failure rather than a
    quiet one**, which is the property the two scans it replaces did not have:

    * a step written as a FLOW mapping (`- {uses: actions/checkout@v4}`). The
      reader would not find it, and :func:`_the_checkout_reader_sees_every_checkout`
      asserts that the number of steps found equals the number of
      `uses:`-assignments naming the action, so the count disagrees and the
      suite goes red.
    * anchors, aliases and merge keys. `release.yml` carries none — re-derived
      with PyYAML 2026-08-28 — and one appearing would take the same count
      check red, because the alias would not carry a literal `uses:` line.
    * whether `${{ github.sha }}` is what the RUNNER substitutes. Nothing here
      can see that; what is held is what the file asks for.
    """
    lines = _code_lines(_release_text())
    steps: list[tuple[dict[str, str], dict[str, str]]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "steps:":
            i += 1
            continue
        seq_indent = len(lines[i]) - len(lines[i].lstrip())
        i += 1
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= seq_indent:
                break
            if not line.lstrip().startswith("- "):
                i += 1
                continue
            item_indent, key_indent = indent, indent + 2
            body = [" " * key_indent + line.lstrip()[2:]]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= item_indent:
                    break
                body.append(nxt)
                i += 1
            keys: dict[str, str] = {}
            with_map: dict[str, str] = {}
            in_with = False
            for entry in body:
                if not entry.strip():
                    continue
                depth = len(entry) - len(entry.lstrip())
                match = re.match(r"([A-Za-z0-9_.-]+):\s*(.*)$", entry.strip())
                if depth == key_indent and match:
                    in_with = match.group(1) == "with"
                    keys[match.group(1)] = _unquote(match.group(2).strip())
                elif in_with and depth == key_indent + 2 and match:
                    with_map[match.group(1)] = _unquote(match.group(2).strip())
            steps.append((keys, with_map))
    return [(k, w) for k, w in steps
            if k.get("uses", "").startswith("actions/checkout@")]


#: THE INPUTS BOTH CHECKOUTS TAKE, AS A CLOSED SET. Each is here for a reason
#: written beside it in `release.yml`, and the set is closed on purpose: the
#: header spends two paragraphs REJECTING `fetch-tags`, and nothing held that
#: rejection — driven, `fetch-tags: true` added to a checkout left this suite
#: at `169 passed` while three docstrings went on saying "three inputs … and
#: no `fetch-tags`". A set cannot be derived from the file it is a claim
#: about, so this is an enumeration; what makes it safe is that it is compared
#: for EQUALITY against a parsed mapping, so an addition is as red as a
#: deletion.
_CHECKOUT_INPUTS = {
    #: no token in the git config after the checkout returns
    "persist-credentials",
    #: the COMMIT and not the tag ref, so the action makes no second fetch and
    #: `refs/tags/<tag>` still names the annotated tag object
    "ref",
    #: full history: at depth 1 the refspec is `[commit]` and no tag ref is
    #: fetched at all, and the tagged-tree job's history checks SKIP
    "fetch-depth",
}


def test_the_checkouts_take_the_inputs_this_file_ARGUES_FOR_AND_NO_OTHERS():
    """The `with:` mapping of every checkout, compared for EQUALITY.

    The two pins below assert that one input each is PRESENT, which says
    nothing about an input added beside them — and `release.yml`'s header
    argues at length that `fetch-tags` in particular must not be here, because
    at depth 1 it makes `refs/tags/v0.1.0` resolve while the history stays one
    commit deep and turns a disclosed skip into a hard failure. Driven with
    `fetch-tags: true` added: `169 passed`, nothing red, while three
    docstrings in this repository said the checkouts carry three inputs and no
    `fetch-tags`.

    WHAT THIS DOES NOT REACH: whether each input's VALUE is right. `ref` and
    `fetch-depth` have pins of their own below; `persist-credentials` has
    none, and that is said here rather than left to be assumed.
    """
    for keys, with_map in _checkout_steps():
        assert set(with_map) == _CHECKOUT_INPUTS, (
            f"a checkout step's inputs are {sorted(with_map)}, and this file "
            f"argues for exactly {sorted(_CHECKOUT_INPUTS)}. An input added "
            f"here is a decision about how the tagged tree is materialised — "
            f"`fetch-tags` is the one the header rejects by name and measures "
            f"— and an input removed is one of the two repairs this workflow "
            f"carries. Decide it deliberately and move this set.\n{keys}"
        )


def test_the_checkout_reader_sees_every_checkout():
    """ANTI-VACUITY FOR THE READER THE TWO PINS BELOW REST ON.

    An action enters a workflow through a `uses:` key and no other way, so the
    number of `uses:`-assignments naming `actions/checkout@` is a count the
    reader must reproduce. When it does not — a flow-style step, an alias, a
    respelling nobody anticipated — the pins under it would be scanning a
    shorter list than the file has, and THAT is the failure mode both of their
    predecessors had silently.
    """
    lines = _code_lines(_release_text())
    declared = [line for line in lines if _USES_CHECKOUT.search(line.strip())]
    found = _checkout_steps()
    assert declared, (
        "no `uses: actions/checkout@` assignment on any code line of "
        "`release.yml`. Either the workflow stopped checking anything out, or "
        "this scan has stopped finding what it is meant to hold."
    )
    assert len(found) == len(declared), (
        f"{len(declared)} `uses:` assignments name `actions/checkout@` and "
        f"the structural reader found {len(found)} steps. A checkout the "
        f"reader cannot see is a checkout the two pins below do not hold, "
        f"which is exactly how a `- name:`-first step slipped past their "
        f"predecessors.\n{[line.strip() for line in declared]}\n{found}"
    )
    for keys, with_map in found:
        assert with_map, (
            f"a checkout step has no `with:` mapping this reader could read: "
            f"{keys}. Its inputs are then unheld, whatever they say."
        )


def test_every_checkout_in_this_file_takes_the_WHOLE_history():
    """The attribute that decides whether refusal point 1 CHECKS anything.

    `actions/checkout@v4` fetches the triggering ref at DEPTH 1 by default, so
    a tag build holds exactly one commit and every sha older than the tag is
    unreachable. Part of what the tagged-tree job runs reads git HISTORY:
    `tests/test_soundness_routing.py`'s four git-gated tests, and the
    status-paragraph ancestry check in `tests/test_proposed_page_headers.py`.
    At depth 1 they do not FAIL — they skip, or they refuse over the
    environment — so a green tagged-tree job is compatible with nothing having
    been decided about the tree being published.

    DRIVEN, in a sandbox built the way that job builds its tree (`git init`;
    one `git fetch` per the checkout config under test; `git checkout` of the
    tag), with `deadbee` planted in a `docs/proposed-*.md` status paragraph
    and the page restored byte-identically after each run:

        depth 1 (the default)            `deadbee` 1 skipped
        `fetch-depth: 0`                 `deadbee` 1 FAILED, named
        `fetch-tags`, depth 1            `deadbee` 1 skipped, and the suite
                                         red in a new place

    THE SCAN READS `actions/checkout@`, NOT `actions/`, AND THAT DISTINCTION
    IS THE DEFECT THAT KEPT THIS OPEN. `grep -c "uses: actions/"` reads 10 in
    `ci.yml` and 4 in `release.yml`; in `ci.yml` all ten of those ARE
    checkouts, so the grep is accidentally right there, while here two of the
    four are `upload-artifact` and `download-artifact`, which take no
    `fetch-depth` input at all. A comparison of the two workflows run on that
    grep reports "4 checkouts here, none of them with `fetch-depth`" — and a
    repair satisfying it on two of the four could have left a real checkout
    shallow, because two of the four it counted are not checkouts.
    Nothing is typed here: the steps are derived from the file, so a checkout
    ADDED later without the key is a red rather than a silent depth-1 job.

    NOT `fetch-tags: true`, and it is not a near-miss. Measured in the same
    sandbox: fetching the tag refs at depth 1 makes `v0.1.0` RESOLVE while the
    history stays one commit deep, so
    `tests/test_soundness_log_reach.py`'s tag guard stops skipping and the
    rule under it hard-fails — every cited commit unreachable, seven
    commit-cited paragraphs read as uncovered. It converts a lost check into a
    red release job. See `release.yml`'s header for the table.
    """
    steps = _checkout_steps()
    shallow = [(k, w) for k, w in steps if w.get("fetch-depth") != "0"]
    assert not shallow, (
        f"{len(shallow)} of the {len(steps)} `actions/checkout` steps in "
        f"`release.yml` no longer carry `fetch-depth: 0`: {shallow}. The "
        "default is depth 1, and at depth 1 the history-reading checks in the "
        "tagged-tree job SKIP rather than fail — a green refusal point 1 that "
        "decided nothing about which commits the shipped `docs/proposed-*.md` "
        "headers name. `ci.yml` carries this on all ten of its checkouts, "
        "under a comment giving the same reason. `fetch-tags` is not the "
        "substitute: see this test's docstring and `release.yml`'s header."
    )


def test_every_checkout_in_this_file_PINS_THE_COMMIT_AND_NOT_THE_TAG_REF():
    """The input that stops `actions/checkout@v4` overwriting the tag ref.

    WITHOUT IT THE ACTION UN-NAMES THE TAG, and that is not a quirk to work
    around but a branch the workflow selects. Read off the action's source at
    `11d5960`: `input-helper.ts` turns an input `ref` matching
    `^[0-9a-fA-F]{40}$` into `commit = <sha>` with `ref = ''`;
    `ref-helper.ts::testRef` answers `shaExists(commit)` when `ref` is empty
    and only otherwise reaches its `REFS/TAGS/` branch, which compares
    `github.sha` against `revParse(refs/tags/<tag>)` — the TAG OBJECT's sha
    for an annotated tag, so always different; and `git-source-provider.ts`
    re-fetches `+<github.sha>:refs/tags/<tag>` exactly when `testRef` is
    false. With `ref: ${{ github.sha }}` there is no second fetch and
    `refs/tags/<tag>` is still the annotated tag object.

    Driven against a mirror of this repository, 2026-08-28, `v0.2.1`: without
    the input, `refs/tags/v0.2.1` comes back `commit a90862ba` with an empty
    tagger date; with it, `tag 99418c06` dated `2026-08-28`. `HEAD` and
    `HEAD^{tree}` are byte-identical either way, which is why this is a safe
    input to add to a job whose subject is the tree.

    THIS PIN EXISTS BECAUSE THE REPAIR IT PROTECTS IS ONE LINE PER CHECKOUT
    AND ITS ABSENCE IS SILENT. Removing it from the `build` checkout left this
    module GREEN — the drives build their own workspaces and cannot see the
    workflow's checkout config, and the step's own re-fetch repairs the ref
    whenever `origin` can be reached, so CI would go on passing until the day
    it could not. Driven red here by deleting either occurrence.

    WHAT IT DOES NOT REACH. It cannot tell that `${{ github.sha }}` is what
    the runner will put there, only that the file asks for it; it types no
    count, so a checkout added later is covered without a re-derivation; and
    it is blind to the SHA-256 case, where a 64-hex `github.sha` fails the
    action's own 40-hex test. What follows from that is written beside the
    checkout and was written WRONG there first: `result.commit` is left
    undefined, so `testRef` returns true at `if (!commit) return true`, there
    is no second fetch and the tag ref stays intact — and the job then dies in
    `getCheckoutInfo`, which finds no branch or tag by that name and throws.
    Fail-closed, and nothing here can see which hash a future repository
    uses.
    """
    steps = _checkout_steps()
    unpinned = [(k, w) for k, w in steps
                if w.get("ref") != "${{ github.sha }}"]
    assert not unpinned, (
        f"{len(unpinned)} of the {len(steps)} `actions/checkout` steps in "
        f"`release.yml` no longer pin the COMMIT: {unpinned}. Without "
        "`ref: ${{ github.sha }}` the action fetches all refs and then "
        "re-fetches `+<github.sha>:refs/tags/<tag>`, force-writing the release "
        "commit over the tag ref — so the changelog gate reads a commit where "
        "an annotated tag object should be, which is how a correctly tagged "
        "`v0.2.1` was refused on 2026-08-28 and its maintainer told to delete "
        "the tag. The tree is identical either way; only which ref names what "
        "changes."
    )


def test_the_headers_refusal_COUNT_is_the_count_of_refusals():
    """MUTANT 5, and the reason it is worth a test rather than a proof-read: a
    number in a comment is exactly the thing nothing checks, and this one had
    already been wrong once — it said TWO, which was the count of REASONS
    listed under it, while the sdist checks landed later and did not move it.

    IT MOVED A SECOND TIME, and this test is what made that a decision rather
    than an oversight: the tag step's "dist/ does not hold exactly one wheel"
    refusal is a SEVENTH `exit 1` site, so the header read NINE.

    IT MOVED A THIRD TIME, by three, when the tag comparison was carried to the
    OTHER artefact: `dist/` must hold exactly one SDIST, that sdist's filename
    version must carry the tag, and the sdist step establishes the count for
    itself instead of trusting a step that ran earlier. `TWELVE` is also the
    wrong number mutation 5 planted when the right one was EIGHT, and that
    collision cost nothing: the word is compared to
    `len(exit_sites) + _IMPLICIT_REFUSALS` rather than to a constant, so a
    header that says TWELVE while the file holds nine `exit 1` sites is exactly
    as red as it was before.

    IT MOVED A FOURTH TIME, by one, when the two counts stopped being mistaken
    for a statement about the DIRECTORY: the tag step inventories `dist/` and
    refuses any entry that is neither of the two artefacts it just established.

    IT MOVED A FIFTH TIME, by one, when the sdist step stopped dying on `tar`'s
    own exit code: a tarball nothing in this workflow can read is that step's
    own refusal now, with an annotation, rather than an rc=2 out of `tar`.
    Twelve `exit 1` sites, FOURTEEN with the two implicit ones. NEITHER NUMBER
    IS TYPED HERE ANY MORE: both words are read off the header and compared to
    the file's own `exit 1` sites, so what is held is that the header agrees
    with the file rather than that the file still says what it said the day
    this was written. The typed `12` that stood in the second half was
    implied by the first half anyway — with the header at FOURTEEN,
    `_NUMBER_WORDS[word] == len(exit_sites) + 2` IS `len(exit_sites) == 12` —
    so all it added was a red on a commit that legitimately moves the count,
    and it was the one place in this module where a count derived from a
    collection was written down a second time by hand. THE OFF-BY-ONE A PERSON
    ACTUALLY MAKES is not a wild number, it is one step either side, so that is
    what this was driven against: `FOURTEEN` -> `THIRTEEN` red, `FOURTEEN` ->
    `FIFTEEN` red, and the breakdown `twelve` -> `eleven` red, each on its own,
    each naming this test. A well-formed `exit 1` added to the manifest step
    reddens it too, from the other direction. The `FOURTEEN` -> `THIRTEEN` row
    is not hypothetical here: THIRTEEN is what this file said before the
    twelfth site landed, so the pin was driven red by the fix itself before the
    header was moved to match it.

    THE BREAKDOWN HALF WAS A BARE SUBSTRING TEST FOR ONE COMMIT — the defect
    this module exists to catch, committed inside the module. While the other
    scans here are `^`-anchored and exactly-once, this one asked whether
    "twelve `exit 1` sites" appeared ANYWHERE in `release.yml`, so a second
    copy of the phrase satisfied it no matter what the header said. It was
    found by being triggered: a comment drafted for that file repeated the
    phrase, and the mutation that should have reddened this test did not.
    REPRODUCED on the committed blob before the fix — phrase planted once in a
    comment paragraph below the header, the header's own breakdown mutated
    `twelve` -> `eleven`, this test PASSED, and the same mutation without the
    plant is red.

    DRIVEN FOUR WAYS on the anchored form, each on its own, each naming this
    test: breakdown mutated with ONE occurrence of the phrase in the file RED
    (as before), breakdown mutated with TWO occurrences RED (green before, and
    this is the fix), the breakdown clause REWORDED OUT of the header RED on
    ZERO hits through `_one_word` rather than silently green, and the header
    SENTENCE — what this scan matches — duplicated RED naming the duplication.
    A fifth input is green BY DESIGN and is the property the anchor buys:
    the PHRASE alone repeated in prose elsewhere in `release.yml`, with the
    header correct, is not a failure. `release.yml` carries such a repeat
    today, in the paragraph recording this hazard, so the two-occurrence drive
    above is the state of the shipped file and not a planted one.

    WHAT THIS PIN STRUCTURALLY CANNOT DO, said here because `release.yml` was
    wrong about it in prose while this test was green: it checks the NUMBER of
    `exit 1` sites and never a POSITION among them. A header sentence naming a
    refusal as "the EIGHTH site in file order" goes stale the moment a refusal
    is inserted above it, and nothing here notices. `release.yml` names its
    refusals rather than numbering them by position for that reason.

    Derived from the file, not from a constant here: the `exit 1` sites are
    counted, and the two refusals that are a step's own exit code are added
    from :data:`_IMPLICIT_REFUSALS`, which is written down because deciding
    mechanically which `- run:` can fail is deciding what every command does.
    """
    text = _release_text()
    lines = _code_lines(text)
    exit_sites = [line for line in lines if line.strip() == "exit 1"]
    header = re.search(_HEADER_COUNT_RE, text, re.M)
    assert header, (
        "the header no longer states a refusal count in the form this reads"
    )
    word = header.group(1)
    assert word in _NUMBER_WORDS, f"unrecognised number word in the header: {word}"
    assert _NUMBER_WORDS[word] == len(exit_sites) + _IMPLICIT_REFUSALS, (
        f"the header says {word} ({_NUMBER_WORDS[word]}) refusal points and the "
        f"file has {len(exit_sites)} `exit 1` sites plus {_IMPLICIT_REFUSALS} "
        "steps that refuse by their own exit code. Move the header, or say why "
        "the arithmetic changed — this count was already wrong once."
    )
    # ...and the header's own breakdown must agree with the same arithmetic.
    # THROUGH `_one_word`, ANCHORED AND EXACTLY-ONCE, for the reason written
    # beside `_HEADER_BREAKDOWN_RE`: as a bare substring this half was
    # satisfied by any occurrence of the phrase anywhere in `release.yml`, and
    # the header's own breakdown could then be edited out from under it.
    breakdown = _one_word(
        text, _HEADER_BREAKDOWN_RE,
        "the header's breakdown of that count into `exit 1` sites",
    )
    # `_NUMBER_WORDS` is keyed on the SHOUTED spelling the header's total uses;
    # the breakdown is ordinary prose in the same sentence, so it is folded
    # here rather than listed twice in the table.
    assert breakdown.upper() in _NUMBER_WORDS, (
        f"the header's breakdown says {breakdown!r} `exit 1` sites, which is "
        f"not a number word this module knows, so nothing can compare it to "
        f"the {len(exit_sites)} sites in the file. Add it to `_NUMBER_WORDS` "
        f"or spell the count as a word."
    )
    assert _NUMBER_WORDS[breakdown.upper()] == len(exit_sites), (
        f"the header's breakdown says {breakdown} "
        f"({_NUMBER_WORDS[breakdown.upper()]}) `exit 1` sites and the file "
        f"has {len(exit_sites)}. The breakdown and the total in front of it "
        f"are one sentence and move together: the total is this number plus "
        f"the {_IMPLICIT_REFUSALS} steps that refuse by their own exit code."
    )


def _one_word(text: str, pattern: str, what: str) -> str:
    """The single capture of an `^`-anchored scan, or a loud failure."""
    hits = re.findall(pattern, text, re.M)
    assert len(hits) == 1, (
        f"expected exactly one sentence in `release.yml` stating {what}, "
        f"found {len(hits)}: {hits}. The sentence has been reworded or "
        f"duplicated, and a pin that finds nothing is green for the wrong "
        f"reason — see {pattern!r}"
    )
    return hits[0]


def test_the_counts_DERIVED_from_the_header_move_with_it():
    """TWO SENTENCES SPELL THE REFUSAL COUNT AGAIN AND NOTHING HELD THEM TO IT.

    The pin above holds the header's count to the file's own `exit 1` sites.
    Two other sentences RESTATE that count and are derived from it:

      * "SEVEN OF THE <N> WERE DRIVEN IN BOTH DIRECTIONS" — the SEVEN is a
        count of drives and does not move when a refusal lands; the N is the
        header's count and does;
      * "rather than adding a <Nth>" — the ordinal a refusal added there WOULD
        take, which is the header's count PLUS ONE.

    BOTH TRACKED THE HEADER IN LOCKSTEP THROUGH FOUR ROUNDS AND THEN STOPPED
    TOGETHER. Read off each commit's own header: EIGHT/EIGHT, NINE/NINE,
    TWELVE/TWELVE + THIRTEENTH, THIRTEEN/THIRTEEN + FOURTEENTH — and then
    FOURTEEN with THIRTEEN and FOURTEENTH left standing, so `release.yml`
    stated its refusal count as two different numbers in one file, and gave the
    name "THE FOURTEENTH" both to the `tar` refusal that WAS added and to the
    refusal that deliberately was NOT. A reader could take the first for the
    second.

    AND THE STALE SENTENCE STATED ITS OWN INVARIANT IN THE SAME BREATH: *"this
    sentence said 'a tenth' when the count was nine, and the count is what
    moves"* — and then did not move. That is why this exists as an instrument
    rather than as a note: the rule was written down, was correct, and was not
    enough.

    NOTHING SAW EITHER. Measured on the parent commit with both set to absurd
    values — `SEVEN OF THE NINETY` and `adding a NINETIETH` — this module was
    21 passed. Both words are unlisted in the tables above, so with this
    pin they now fail LOUDLY rather than reading as zero.

    DRIVEN, on the file this commit ships, against the off-by-ones a person
    actually makes and against the stale values themselves — each on its own,
    each naming this test: `FOURTEEN` -> `THIRTEEN` (the parent's own value)
    red, -> `FIFTEEN` red, -> `NINETY` red as an unrecognised word;
    `FIFTEENTH` -> `FOURTEENTH` (the parent's own value) red, -> `SIXTEENTH`
    red, -> `NINETIETH` red as an unrecognised word. And from the other side:
    moving the HEADER alone reddens this together with the count pin, which is
    the coupling this test exists to assert. GREEN, unchanged, on `eb26d482`
    and `400415fe` — the last two commits at which the file was consistent —
    so it is not a check that only passes on today's numbers.

    PLUS ONE IS WELL DEFINED ONLY UNDER AN INTRA-COMMIT ORDERING CONVENTION,
    AND THE CASE THAT NEEDS ONE HAS ALREADY HAPPENED. "The ordinal a refusal
    added there WOULD take" is the count plus one when ONE refusal is added.
    `400415f` added THREE in a single commit and took the count 9 -> 12, so
    "the next one" was 10, 11 or 12 depending on which of the three is meant,
    and nothing said which. The convention that commit actually used is FILE
    POSITION — the members of a group are numbered by where they STAND in
    `release.yml`, earliest lowest, and the group ends on the new total.
    Re-derived on it: the tag step's sdist-count refusal, the tag-versus-sdist
    comparison and the sdist step's own tarball count stand in that order in
    that file, and the header calls them the tenth, the eleventh and the
    twelfth. It was nowhere written down; it is written down now, in
    `release.yml`'s header beside the renumbering rule it completes. THIS PIN
    IS UNAFFECTED EITHER WAY — it compares the ordinal to the header's total
    plus one, and a group's last member is the one that lands on the total —
    but what the sentence MEANS while a group is being added was
    under-specified, and an ordinal that does not say which order it is in is
    the defect this whole pin was written for.

    WHAT IT CANNOT DO is the same thing the count pin cannot do: these are
    words in comments, and nothing in this repository derives an ORDINAL for a
    refusal. What is held is that the three sentences agree with each other and
    that the first of them agrees with the file's `exit 1` sites.
    """
    text = _release_text()
    total = _NUMBER_WORDS[_one_word(text, _HEADER_COUNT_RE, "the refusal count")]
    driven = _one_word(text, _DRIVEN_OF_RE, "the total the seven drives are OF")
    declined = _one_word(text, _DECLINED_ORDINAL_RE,
                         "the ordinal of the refusal deliberately NOT added")
    # BOTH SENTENCES ARE REPORTED, NOT THE FIRST ONE TO FAIL. They went stale
    # together in one commit and are repaired together; a red that names one
    # of them invites a fix that leaves the file inconsistent in the other
    # direction.
    wrong = []
    if driven not in _NUMBER_WORDS:
        wrong.append(
            f"'SEVEN OF THE {driven}' is not a number word this module knows, "
            f"so nothing can compare it to the header's {total}. Add it to "
            f"`_NUMBER_WORDS` or spell the count as a word."
        )
    elif _NUMBER_WORDS[driven] != total:
        wrong.append(
            f"'SEVEN OF THE {driven}' ({_NUMBER_WORDS[driven]}) restates the "
            f"refusal count and the header says {total}. The SEVEN is a count "
            f"of DRIVES and does not move when a refusal lands; the total it "
            f"is seven OF is the header's count and does. It followed the "
            f"header through EIGHT, NINE, TWELVE and THIRTEEN and then stopped."
        )
    if declined not in _ORDINAL_WORDS:
        wrong.append(
            f"'adding a {declined}' is not an ordinal this module knows, so "
            f"nothing can compare it to {total + 1}. Add it to "
            f"`_ORDINAL_WORDS` or spell the ordinal as a word."
        )
    elif _ORDINAL_WORDS[declined] != total + 1:
        wrong.append(
            f"'adding a {declined}' ({_ORDINAL_WORDS[declined]}) names the "
            f"refusal that was deliberately NOT added, and with {total} "
            f"refusal points the next one would be number {total + 1}. That "
            f"sentence says of ITSELF that it read 'a tenth' when the count "
            f"was nine and that the count is what moves. An ordinal one short "
            f"also COLLIDES: it hands a refusal that was never added the name "
            f"this file already gives to one that was, so a reader can "
            f"conclude the wrong refusal was the one declined."
        )
    assert not wrong, (
        f"`release.yml`'s header states {total} refusal points and the "
        f"sentences derived from that count no longer agree with it:\n  "
        + "\n  ".join(wrong)
    )


def test_the_uncommitted_member_comparison_is_the_difference_not_the_overlap():
    """MUTANT 6, and the worst of the six: it cannot fire.

    `comm -23 members.txt explained.txt` is "in members and not in explained" —
    the unexplained members, empty on a healthy tree and non-empty exactly when
    something uncommitted shipped. `comm -12` is the INTERSECTION: on a healthy
    tree it is every member of the sdist, so the step goes red on every release
    — but a reviewer reading a green run learns nothing, and the mutation that
    matters is any rewrite that leaves it empty when it should not be.

    THE LITERAL IS PINNED BECAUSE A LITERAL IS CHEAP TO PIN, and NOT — as this
    docstring said for four commits — because "the semantic cannot be tested
    without a runner". It can, it now is, and the sentence was doing real
    damage: it is the argument that made an unfirable rewrite one line above
    this one acceptable to leave uncovered. The semantic is driven in
    :func:`test_the_sdist_gate_refuses_a_member_that_is_not_committed`, which
    executes this very step body against a planted tree. This test and that one
    catch different things and both are wanted: `comm -12` is caught here
    (the drive would catch it too, but as a mysterious failure rather than as a
    named one), and the `explained.txt` rewrite is caught only there.
    """
    text = _release_text()
    assert (
        'unexplained="$(comm -23 "${work}/members.txt" "${work}/explained.txt")"'
        in text
    ), (
        "the sdist committed-members comparison is no longer `comm -23 "
        "members.txt explained.txt`. `-23` suppresses columns 2 and 3, leaving "
        "lines ONLY in members.txt — the unexplained ones. `-12` leaves the "
        "common lines, which is not a check at all. (The two operands moved "
        "into a `mktemp -d` scratch directory when the step stopped leaving "
        "four files in the checkout root on its refusal paths; the flags and "
        "the order of the operands are what this pins.)"
    )
    # CODE lines only, and the forbidden flags are assembled rather than
    # written: the header of `release.yml` names the mutation in prose, and so
    # does this file, so a whole-text scan for the string would fail on the
    # record of the defect rather than on the defect.
    code = "\n".join(_code_lines(text))
    for flags in ("-" + "12", "-" + "13"):
        assert f"comm {flags}" not in code, (
            f"a `comm {flags}` has appeared in the workflow's code; only `-23` "
            "answers the question this step asks"
        )
    assert "comm " in code, "the comparison no longer uses `comm` at all"


def test_this_pin_is_reading_the_real_file():
    """Anti-vacuity. Every assertion above is a substring test, and a substring
    test over an empty string, a missing file, or a file this stopped
    resolving is green in the most misleading possible way."""
    text = _release_text()
    assert len(text.splitlines()) > 200, (
        f"release.yml is {len(text.splitlines())} lines; this is not the file "
        "these pins were written against"
    )
    assert "name: release" in text
    assert "pypa/gh-action-pypi-publish" in text
    lines = _code_lines(text)
    assert len(lines) < len(text.splitlines()), (
        "the comment filter matched nothing, so `_code_lines` is not doing "
        "what the pins above assume"
    )
    # the filter must not eat the code the pins read
    assert any(line.strip() == "exit 1" for line in lines)


# --- THE GATE BODIES, EXTRACTED AND EXECUTED --------------------------------
#
# Everything above this line is a text pin. Everything below drives a real step
# body, in a real shell, against a planted tree.
#
# WHY THIS IS NOT A RUNNER PROBLEM, since this file argued for four commits
# that it was. Both bodies below read exactly two things: a `dist/` directory
# and a git index. A runner supplies neither — `uv build` and
# `actions/checkout` do, and a test can plant both. The one thing a runner
# does supply to these two steps is `${TAG}`, which is an ordinary environment
# variable. "A gate's behaviour is a property of a runner" is true of the
# `publish` job and false of these.
#
# THE BODIES ARE EXTRACTED, NEVER TRANSCRIBED. A copy of a step body here would
# be a second source that drifts — and worse, the rewrites these tests exist to
# catch land in `release.yml`, so a test running its own copy would stay green
# through every one of them. `_step_body` reads the YAML block scalar as text,
# for the reason the header gives: `yaml` is not a dependency and the zero-dep
# job could not import one.

#: THE COMMANDS THE FOUR EXTRACTED BODIES INVOKE. This tuple said it was
#: "derived by reading the extracted bodies" and it was not: it named seven
#: and the bodies invoke thirteen. `basename`, `cat`, `cut`, `grep`, `mktemp`
#: and `wc` were missing, and all six had been there since before the tag
#: repair — only `tr` arrived with it. The consequence was benign (a box
#: without `cut` would have failed a drive instead of skipping it, which is
#: the safe direction), but a comment claiming a method nobody carried out is
#: the defect this module exists to find one file over.
#:
#: MEASURED 2026-08-28 over the CODE lines of the four bodies, whole-word:
#: `tar` 22, `git` 18, `sort` 5, `cut` 4, `basename`/`cat`/`sed`/`tr`/`wc` 2
#: each, `comm`/`grep`/`mktemp` 1 each. `head`, `find` and `date` also match
#: as words and are NOT commands here — they are a shell variable and two
#: bits of English inside refusal messages, which is exactly why this is a
#: reading and not a derivation.
#:
#: AND A DERIVATION WAS TRIED AND REJECTED, because deciding which tokens in a
#: shell script are commands is parsing bash: a scan for words in command
#: position over these bodies returns 69 candidates, of which 13 are commands
#: and the rest are variable names and prose inside quoted strings. What IS
#: held mechanically is the other direction —
#: :func:`test_every_command_this_module_requires_is_one_the_bodies_USE`
#: fails if a name here stops appearing — so the list can go stale by omission
#: but not by fiction, and that asymmetry is stated rather than hidden.
_NEEDED = ("bash", "git", "tar", "comm", "sort", "sed", "tr",
           "basename", "cat", "cut", "grep", "mktemp", "wc")
_needs_a_shell = pytest.mark.skipif(
    any(shutil.which(t) is None for t in _NEEDED),
    reason="needs a POSIX shell and coreutils to drive a gate body",
)


def _table_labels(doc: str | None) -> set[str]:
    """The row labels of a docstring's drive table, parsed out of the table.

    **A TABLE HELD IN ONE DIRECTION IS HALF A CHECK, AND THE MISSING HALF IS
    THE ONE THAT ALREADY FAILED HERE.** The drives below assert that every row
    they run is named in their own table — rows -> table — which catches a row
    added without a line. Nothing caught a row DELETED with its table line left
    standing, and that is the direction the failure took: a `no-head` row was
    lost with the test that carried it, the table went on advertising it, and
    it came back only because a mutant was driven. Worse, a deleted row can be
    the only pin on a refusal: removing the `taggerless` block and leaving its
    table line was `38 passed`, and neutralising the empty-tagger-date refusal
    it was the sole pin on was then `43 passed` — where that same
    neutralisation with the row present is `1 failed`.

    So the labels are read back OUT of the table and compared for equality.
    The table block is delimited structurally — from its `───` rule to the
    first blank line — and a label is a backticked lowercase token that is the
    first thing on its row, so continuation lines and mid-row backticks are
    not labels. Under `python -OO` there is no docstring at all and this
    returns the empty set, which the callers treat as "nothing to hold".
    """
    if not doc:
        return set()
    found: list[tuple[int, str]] = []
    inside = False
    for line in doc.splitlines():
        if "\u2500" in line:
            inside = True
            continue
        if not inside:
            continue
        if not line.strip():
            inside = False
            continue
        match = re.match(r"(\s+)`([a-z][a-z0-9-]+)`", line)
        if match:
            found.append((len(match.group(1)), match.group(2)))
    if not found:
        return set()
    # THE LABEL COLUMN, NOT "THE FIRST BACKTICK ON THE ROW". A cell that wraps
    # can put a backticked word at the start of a CONTINUATION line — driven:
    # a row whose third column wrapped onto ``\u0060tag\u0060; rc=0`` was read as a
    # label called `tag`. Labels live in one column, so the leftmost
    # indentation any of them has is the column, and anything indented past it
    # is part of some row's prose.
    column = min(indent for indent, _ in found)
    return {label for indent, label in found if indent == column}


def test_every_command_this_module_requires_is_one_the_bodies_USE():
    """:data:`_NEEDED` holds nothing that the extracted bodies do not invoke.

    The skip guard this list feeds decides whether four drives RUN. A name
    that no body uses makes those drives skip on a box that could have run
    them — an instrument switching itself off over a command nobody needs —
    and a skip is what pytest exits 0 on. The other direction, a command the
    bodies use and this list omits, is not checked here and cannot be by a
    text reader: see the comment on :data:`_NEEDED` for the attempt and why it
    was dropped. That direction fails a drive loudly rather than skipping it.
    """
    code = "\n".join(
        "\n".join(line for line in _step_body(step).splitlines()
                  if not line.lstrip().startswith("#"))
        for step in (_SDIST_STEP, _TAG_STEP, _MANIFEST_STEP, _CHANGELOG_STEP)
    )
    unused = [
        name for name in _NEEDED
        if name != "bash" and not re.search(r"(?<![\w-])" + name + r"(?![\w-])", code)
    ]
    assert not unused, (
        f"{unused} are required by `_NEEDED` and appear in none of the four "
        f"extracted step bodies. Every name here gates whether the drives run "
        f"at all, so one that nothing needs turns a runnable box into a "
        f"skipped one. (`bash` is exempt: it is what RUNS the bodies rather "
        f"than something they name.)"
    )


def _step_lines(step_name: str) -> list[str]:
    """The block of `release.yml` belonging to the step called `step_name`."""
    lines = _release_text().splitlines()
    want = f"- name: {step_name}"
    starts = [i for i, line in enumerate(lines) if line.strip() == want]
    assert len(starts) == 1, (
        f"expected exactly one step named {step_name!r} in release.yml, found "
        f"{len(starts)}. These drives address a step BY NAME, so a rename or a "
        "duplicate must stop the suite rather than quietly measure the wrong "
        "step, or no step at all."
    )
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    out = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip():
            here = len(line) - len(line.lstrip())
            if here < indent or (here == indent and line.lstrip().startswith("- ")):
                break
        out.append(line)
    return out


def _block(lines: list[str], key: str) -> list[str]:
    """The lines indented under `key` within an already-extracted step."""
    at = [i for i, line in enumerate(lines) if line.strip() == key]
    assert len(at) == 1, f"expected exactly one {key!r} in this step, found {len(at)}"
    head = at[0]
    indent = len(lines[head]) - len(lines[head].lstrip())
    out = []
    for line in lines[head + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        out.append(line)
    return out


def _step_body(step_name: str) -> str:
    """The step's `run: |` script, dedented, verbatim."""
    body = _block(_step_lines(step_name), "run: |")
    real = [line for line in body if line.strip()]
    assert real, f"the step {step_name!r} has an empty `run:` block"
    cut = min(len(line) - len(line.lstrip()) for line in real)
    text = "\n".join(line[cut:] if line.strip() else "" for line in body)
    # ANTI-VACUITY FOR THE EXTRACTOR ITSELF. An extractor that silently returned
    # "" — a renamed key, a changed indent, a `run: >` — would make every drive
    # below pass against an empty script, which is the same shape of green as
    # the gate that cannot fire.
    assert text.strip().startswith("set -euo pipefail"), (
        f"the extracted body of {step_name!r} does not begin with its own "
        f"`set -euo pipefail`, so the block scalar is not being read as this "
        f"expects. Got: {text[:120]!r}"
    )
    return text + "\n"


def _step_env(step_name: str) -> dict[str, str]:
    """The step's literal `env:` entries. `${{ }}` expressions are the caller's."""
    out = {}
    for line in _block(_step_lines(step_name), "env:"):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        key, _, value = entry.partition(":")
        out[key.strip()] = value.strip()
    return out


def _drive(body: str, cwd: pathlib.Path, **env):
    """Run an extracted step body in `cwd` with a scrubbed environment."""
    clean = {
        k: v
        for k, v in os.environ.items()
        if k in ("PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT")
    }
    clean.update(env)
    return subprocess.run(
        ["bash", "-c", body],
        cwd=cwd,
        env=clean,
        capture_output=True,
        text=True,
        timeout=120,
    )


_SDIST_STEP = "every sdist member is a committed file of the tagged tree"
_TAG_STEP = "the tag and the artifacts must agree"
_MANIFEST_STEP = "the sdist manifest, for the record"

# Enough of a tree for the sdist body to agree to compare anything: it demands
# PKG-INFO, pyproject.toml, README.md and LICENSE by name before it looks at
# the difference, and a non-empty index.
_TRACKED = ("pyproject.toml", "README.md", "LICENSE", "src/stelling/contracts.py")

# The extensions the plants below wear, cycled across the roots so that a
# filter keyed on a SUFFIX rather than on a directory meets more than one.
_PLANT_SUFFIXES = (".py", ".md", ".yml", ".toml", "")


def _sdist_directory_roots() -> list[str]:
    """The DIRECTORY entries of the sdist allowlist, read from `pyproject.toml`.

    DERIVED AND NOT TYPED, for the reason the rest of this repository derives
    it: a typed list stops covering a root the moment the allowlist gains one,
    and silently. Read as text rather than with `tomllib`, which is 3.11+ while
    the declared floor is 3.10 — see `tests/test_zero_dep_import_discipline.py`.
    """
    repo = pathlib.Path(__file__).resolve().parent.parent
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    block = re.search(
        r"\[tool\.hatch\.build\.targets\.sdist\]\s*\ninclude\s*=\s*\[(.*?)^\]",
        text, re.S | re.M,
    )
    assert block, "the sdist allowlist is not where this expects it"
    roots = [m.group(1).lstrip("/") for m in re.finditer(r'"([^"]+)"', block.group(1))]
    return [r for r in roots if (repo / r).is_dir()]


def _plants() -> list[str]:
    """One uncommitted path under every allowlisted directory root."""
    roots = _sdist_directory_roots()
    assert len(roots) >= 5, roots
    return [
        f"{root}/_audit_probe{_PLANT_SUFFIXES[i % len(_PLANT_SUFFIXES)]}"
        for i, root in enumerate(sorted(roots))
    ]


def _plant_tree(tree: pathlib.Path, *, uncommitted: tuple[str, ...] = (),
                tarball: bytes | None = None, pax: bool = False) -> pathlib.Path:
    """A git checkout plus a `dist/` holding one sdist, as the build job leaves it.

    Each `uncommitted` path goes into the TARBALL and deliberately not into the
    index — the exact shape this gate exists to refuse: a file that ships and is
    not in the tagged tree.

    `dist/.gitignore` IS PART OF THE SHAPE and is planted here for the same
    reason :func:`_dist_of` plants it: `uv build` writes it (one byte, `*`), so
    a `dist/` without one is not the directory a release actually has, and a
    drive over a `dist/` that never held a dotfile could not notice a step that
    started refusing the real thing.
    """
    (tree / "dist").mkdir(parents=True)
    (tree / "dist" / ".gitignore").write_text("*", encoding="utf-8")
    for rel in _TRACKED:
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\n", encoding="utf-8")
    subprocess.run(
        ["git", "-c", "init.defaultBranch=main", "init", "-q"],
        cwd=tree, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "add", "--", *_TRACKED], cwd=tree, check=True, capture_output=True
    )

    members = list(_TRACKED) + ["PKG-INFO"]
    (tree / "PKG-INFO").write_text("Metadata-Version: 2.4\n", encoding="utf-8")
    for rel in uncommitted:
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# planted, and not in the index\n", encoding="utf-8")
        members.append(rel)

    at = tree / "dist" / "stelling-0.1.0.tar.gz"
    if tarball is not None:
        # `tarball` REPLACES the archive and not the tree: the checkout, the
        # index and the four demanded names are still what a healthy release
        # has, so anything the drive then sees is a property of the bytes.
        at.write_bytes(tarball)
        return tree
    fmt = tarfile.PAX_FORMAT if pax else tarfile.DEFAULT_FORMAT
    with tarfile.open(at, "w:gz", format=fmt) as tar:
        for rel in sorted(members):
            info = tar.gettarinfo(str(tree / rel), arcname=f"stelling-0.1.0/{rel}")
            if pax:
                # A keyword no `tar` knows. GNU tar 1.35 exits 0 and writes
                # `Ignoring unknown extended header keyword` to STDERR, once
                # per member — a diagnostic that is not a member, on the
                # success path. See
                # `test_the_sdist_gate_does_not_read_tars_warnings_as_members`.
                info.pax_headers = {"ANTHROPIC.weird": "1"}
            with open(tree / rel, "rb") as handle:
                tar.addfile(info, handle)
    return tree


@_needs_a_shell
def test_the_sdist_gate_refuses_a_member_that_is_not_committed(tmp_path):
    """THE SDIST GATE, DRIVEN — both directions, on the body `release.yml` ships.

    THE HOLE THIS CLOSES IS ONE LINE FROM A LINE THAT WAS PINNED.
    `explained.txt` is what the members get compared against::

        sort -u tracked.txt generated.txt > explained.txt    <- was unpinned
        unexplained="$(comm -23 members.txt explained.txt)"  <- was pinned

    Rewrite the first to `sort -u members.txt generated.txt` and `explained.txt`
    becomes a superset of `members.txt`, so `comm -23` is empty on every input
    and the gate cannot fire — while
    :func:`test_the_uncommitted_member_comparison_is_the_difference_not_the_overlap`
    stays green, because `comm -23 members.txt explained.txt` is still there
    letter for letter.

    MEASURED, this body, this test, CPython 3.11.15, `JAX_ENABLE_X64=1`:

        unmutated, healthy tree    rc=0, "every one of 5 sdist members is
                                          committed to this tree"
        unmutated, planted members rc=1, names every planted path
        mutated,   planted members rc=0, "every one of 14 sdist members is
                                          committed to this tree" — with the
                                          uncommitted files inside the tarball

    Both directions are asserted. The refusal is the point; the pass is here
    because a gate that refuses everything is not a gate either, and a harness
    that could never reach rc=0 would make the refusal meaningless.

    THE PLANT USED TO BE ONE FILE, UNDER `src/`, AND THAT WAS A HOLE THE SIZE
    OF A ROOT. The step's own `sed` normalises member paths, and one more
    expression on it — `-e '/^docs\\//d'`, twelve characters — deletes an
    entire allowlisted root from `members.txt` before the comparison sees it.
    Driven at 461b2d5 with an uncommitted `docs/internal-release-checklist.md`
    inside the tarball:

        unmutated  rc=1, names docs/internal-release-checklist.md
        mutated    rc=0, "every one of 5 sdist members is committed to this
                          tree" — with the uncommitted file in the tarball

    and the FULL SUITE on the mutated tree was 1430 / 0 / 0 / 94, every text
    pin here green and `test_the_drives_are_reading_the_real_step_bodies`
    green with it. So the plant is now one uncommitted file under EVERY
    directory root of the sdist allowlist, derived from `pyproject.toml` by
    :func:`_sdist_directory_roots` so that adding a root to the allowlist
    extends the drive rather than quietly leaving it behind, and every one of
    them must be named in the refusal.
    """
    body = _step_body(_SDIST_STEP)
    env = _step_env(_SDIST_STEP)
    assert env.get("GENERATED") == "PKG-INFO", (
        f"the sdist step's GENERATED is now {env.get('GENERATED')!r}; this "
        "drive plants exactly the members the body demands by name"
    )

    healthy_tree = _plant_tree(tmp_path / "healthy")
    healthy = _drive(body, healthy_tree, **env)
    assert healthy.returncode == 0, (
        "the sdist gate refuses a tree in which every member IS committed; a "
        f"gate that cannot pass is not a gate.\n{healthy.stdout}\n{healthy.stderr}"
    )
    assert "sdist members is committed to this tree" in healthy.stdout
    _no_workings_left_behind(healthy_tree, "the passing path")

    planted = tuple(_plants())
    planted_tree = _plant_tree(tmp_path / "planted", uncommitted=planted)
    bad = _drive(body, planted_tree, **env)
    assert bad.returncode != 0, (
        "THE SDIST GATE PASSED A TARBALL CONTAINING FILES THAT ARE NOT IN THE "
        f"INDEX ({', '.join(planted)}). It reported:\n{bad.stdout}\n"
        "An sdist on PyPI cannot be unpublished, only yanked. Check the line "
        "that builds `explained.txt`: comparing `members.txt` against a set "
        "BUILT FROM `members.txt` is empty by construction, and every text pin "
        "in this file stays green through it."
    )
    missing = [p for p in planted if p not in bad.stdout]
    assert not missing, (
        "the gate refused, but did not name every member responsible, so the "
        "publish log does not say what to fix — and a member it does not name "
        "is a member the comparison never examined, which is how a filter that "
        "exempts one ROOT hides inside a gate that still refuses on another:\n"
        f"  unnamed: {missing}\n{bad.stdout}"
    )
    assert "not committed to this tree" in bad.stdout
    # THE REFUSAL PATH LEAVES NOTHING EITHER, and it used to leave four files.
    # See :func:`_no_workings_left_behind`.
    _no_workings_left_behind(planted_tree, "a refusal path")


#: The four files the sdist step builds its comparison out of. They were
#: written into the CHECKOUT ROOT and removed by an `rm -f` on the success path
#: only, so every refusal that reached them left all four behind — re-derived
#: on the body this commit ships, in the historical shape (`work` pointed at
#: the checkout root, no `trap`): four files on each of the THREE paths that
#: get that far, and zero on each of the TWO that return above the writes (the
#: tarball count and `tar` failing), in a step whose own comment argues that
#: nothing is carried between steps. `release.yml` said "three … and zero on
#: the fourth" beside those writes from `eb26d48`, when the step had four
#: sites, until this commit. They live in a `mktemp -d` scratch directory now,
#: removed by a `trap` on every way out.
_WORKINGS = ("members.txt", "tracked.txt", "generated.txt", "explained.txt")


def _no_workings_left_behind(tree: pathlib.Path, path_taken: str) -> None:
    left = sorted(name for name in _WORKINGS if (tree / name).exists())
    assert not left, (
        f"the sdist step left {left} in the checkout root on {path_taken}. "
        "This step's own comment argues that nothing is carried between steps "
        "and that each `run:` block must be correct on the input it is given; "
        "a step that leaves its workings in the workspace is handing the next "
        "one an input nobody wrote down. They belong in the scratch directory "
        "the `trap` removes."
    )


def _dist_of(base: pathlib.Path, name: str, *, wheels=("0.1.0",),
             sdists=("0.1.0",)) -> pathlib.Path:
    """A checkout whose `dist/` holds the named wheels and sdists, and nothing.

    The tag step reads FILENAMES and never opens either file, so empty files
    are the whole artefact here. `sdists` defaults to one because that is what
    a single `uv build` leaves beside the wheel — a `dist/` with a wheel and no
    tarball is not the shape this release path produces, and after the sdist
    count landed it is not a shape the step accepts either.

    AND `dist/.gitignore`, WHICH IS NOT DECORATION. `uv build` writes it
    itself — one byte, `*`, verified on this tree — so every real release
    directory holds a third entry, and the step now inventories the WHOLE
    directory rather than two globs over it. Planting it here means every
    accepting drive in this module is a drive over the directory a release
    actually has: without it, a rewrite that refused `dist/.gitignore` would
    leave this module green and every release red.
    """
    tree = base / name
    (tree / "dist").mkdir(parents=True)
    (tree / "dist" / ".gitignore").write_text("*", encoding="utf-8")
    for version in wheels:
        (tree / "dist" / f"stelling-{version}-py3-none-any.whl").write_bytes(b"")
    for version in sdists:
        (tree / "dist" / f"stelling-{version}.tar.gz").write_bytes(b"")
    return tree


@_needs_a_shell
def test_the_sdist_gate_refuses_a_dist_it_cannot_read(tmp_path):
    """THE TWELFTH REFUSAL POINT, AND IT REPLACED A DEATH WITH NO ANNOTATION.

    ORDINALS IN THIS MODULE AND IN `release.yml`'s HEADER ARE ARRIVAL ORDER —
    the order the refusals were ADDED, which is what "the header said EIGHT
    until ..." tracks — and never a position in file order. This heading said
    "the ELEVENTH `exit 1` SITE" while the file held eleven `exit 1` sites in
    total and this one stood eighth among them. Nothing derives either number,
    so each refusal is NAMED as well as numbered; see the count pin's docstring
    for why a position claim cannot be pinned at all.

    AND IT THEN SAID "THE ELEVENTH REFUSAL POINT", WHICH WAS OFF BY ONE IN THE
    ORDER IT HAD JUST NAMED. This refusal — the sdist step establishing the
    tarball count for itself — arrived in the commit that took `release.yml`'s
    header from NINE to TWELVE, in a group of three, and it is the last of the
    three. Arrival order numbers a refusal by the count AFTER it arrives, so it
    is the TWELFTH and not the eleventh; `release.yml` carried the same
    off-by-one on the same four refusals and is corrected in the same commit as
    this line. The rule, stated there: the refusal called the Kth is the one
    that took the count TO K.

    `sdist="$(ls dist/*.tar.gz)"` is quoted, so two matches are ONE string with
    a newline in it and `tar tzf` is handed a filename that does not exist;
    with no match at all `ls` fails on the unexpanded glob. Driven on the body
    that shipped before this, a `dist/` holding `stelling-0.0.9.tar.gz` and
    `stelling-0.2.0.tar.gz`::

        rc=2   tar (child): dist/stelling-0.0.9.tar.gz<LF>dist/stelling-
               0.2.0.tar.gz: Cannot open: No such file or directory

    — red, and not by this gate: no `::error` annotation, nothing in the job
    summary, and a reviewer reading the log meets a `tar` error rather than a
    statement about `dist/`. That is the state the no-wheel case in the tag
    step was repaired out of, one step above and by the same argument.

    WHY THIS COUNT IS MADE TWICE. The tag step establishes it already and runs
    first, so this refusal cannot fire in the workflow as it stands. It is here
    because a `run:` block is a script that has to be correct on the input it
    is given: nothing is carried between steps, this drive extracts and runs
    THIS body alone, and "an earlier step would have caught it" is the same
    shape of argument as "same build, so it agrees by construction" — the
    sentence this whole change exists to retract.
    """
    body = _step_body(_SDIST_STEP)
    env = _step_env(_SDIST_STEP)

    healthy = _plant_tree(tmp_path / "one")
    ok = _drive(body, healthy, **env)
    assert ok.returncode == 0, (
        f"the sdist gate refuses a one-tarball `dist/`.\n{ok.stdout}\n{ok.stderr}"
    )

    two = _plant_tree(tmp_path / "two")
    shutil.copyfile(two / "dist" / "stelling-0.1.0.tar.gz",
                    two / "dist" / "stelling-0.0.9.tar.gz")
    pair = _drive(body, two, **env)
    assert pair.returncode != 0, (
        "THE SDIST GATE PASSED A `dist/` HOLDING TWO TARBALLS. It reads ONE "
        "and speaks only for that one, while the publish step uploads every "
        f"file in the directory. It reported:\n{pair.stdout}"
    )
    assert "cannot tell which tarball to read" in pair.stdout, (
        "the two-tarball `dist/` is not reaching this step's own refusal — it "
        "is dying inside `tar` on a filename with a newline in it, with no "
        f"annotation:\n{pair.stdout}\n{pair.stderr}"
    )
    for name in ("stelling-0.1.0.tar.gz", "stelling-0.0.9.tar.gz"):
        assert name in pair.stdout, (
            f"the refusal did not name {name}:\n{pair.stdout}"
        )

    none = _plant_tree(tmp_path / "none")
    (none / "dist" / "stelling-0.1.0.tar.gz").unlink()
    gone = _drive(body, none, **env)
    assert gone.returncode != 0, gone.stdout
    assert "cannot tell which tarball to read" in gone.stdout, (
        "an empty `dist/` is dying on `ls`'s own exit code again, with no "
        f"annotation:\n{gone.stdout}\n{gone.stderr}"
    )


def _valid_tarball_bytes(version: str = "0.2.0") -> bytes:
    """A real one-member `stelling-<version>.tar.gz`, as bytes."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        payload = (f"Metadata-Version: 2.4\nName: stelling\n"
                   f"Version: {version}\n").encode("utf-8")
        info = tarfile.TarInfo(f"stelling-{version}/PKG-INFO")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _corrupt_tarballs() -> dict[str, bytes]:
    """The FIVE ways a `dist/*.tar.gz` makes `tar tzf` exit non-zero, as bytes.

    The first four took the manifest body of two commits ago to rc=2 with no
    annotation, from a step `release.yml` calls a non-refusal. ALL FIVE took
    the SDIST step to rc=2 with no annotation — from a step that file calls a
    refusal point — until the twelfth `exit 1` site landed; measured on the
    parent body, one tarball in `dist/`. They are built here rather than named
    because "a corrupt tarball" is five different failures of `tar`, and one of
    them still prints a member list on stdout while exiting 2.

    THE EMPTY FILE IS THE FIFTH AND IT WAS MISSING FROM THAT RECORD, which is
    the shape a half-written upload leaves and the cheapest one to produce by
    accident. It is measured here rather than inherited: rc=2 in the sdist
    step before the refusal landed, rc=0 and "could not list this file" in the
    manifest step.

    FIVE IS THE COUNT OF SHAPES THAT MAKE `tar tzf` EXIT NON-ZERO, AND THAT IS
    NOT THE SAME AS THE COUNT OF UNREADABLE TARBALLS. This module holds TWO
    witnesses to that, and one paragraph used to name only the second.

    THE FIRST IS `_A_SHORT_GZIP_OF_A_NON_TAR`, DIRECTLY BELOW, and it is why
    two numbers in `release.yml` differ by one and are BOTH right. `tar tzf`
    exits **0** on it and lists nothing, so it is unreadable and never reaches
    the `tar` annotation at all — the empty-`members.txt` arm is what refuses
    it. So the sdist step's own comment says "all five corrupt shapes rc=1 with
    this annotation" (five shapes make `tar` exit non-zero — the five here) and
    the header's THE FOURTEENTH paragraph says the tag step "returns rc=0 on
    every corrupt shape measured — six of them" (six shapes are unreadable —
    these five plus the short gzip). Five and six count different sets, the
    sixth is the constant below, and the sentence that names one witness has to
    name the other or the two numbers read as a discrepancy.

    THE SECOND IS A FIFO named like a tarball, which makes `tar tzf` BLOCK — a
    FIFO with no writer never returns from `open` — so it produces no exit code
    at all and THE TWELFTH `exit 1` SITE, THE FOURTEENTH REFUSAL POINT, never
    runs.

    NOT "the twelfth refusal", which this paragraph said and which is a
    different and LIVE refusal. Under the arrival order this module and
    `release.yml` both number in, the TWELFTH refusal is the sdist step's own
    tarball count — the one
    :func:`test_the_sdist_gate_refuses_a_dist_it_cannot_read` heads with THE
    TWELFTH REFUSAL POINT, a hundred lines above — and a FIFO is ONE glob
    match, so on this shape it RUNS AND PASSES. TRACED under `bash -x` on the
    body this commit ships, a `dist/` whose only `*.tar.gz` is a FIFO: `[ 1 -ne
    1 ]` evaluates false, `sdist=` is assigned, the scratch directory is made
    and the `trap` set, and control stops inside `tar tzf` with no annotation
    printed. The refusal that never runs is the one BELOW that call. Naming it
    "the twelfth" handed a live refusal's name to a dead path in the same
    module that names the live one, which is the collision this file refused to
    leave standing for FOURTEENTH.

    Measured on the same bodies: the tag step rc=0 (it reads filenames), the
    sdist step and the manifest step both hung and were killed at 20 s. It
    fails CLOSED, since nothing is published, and it is unreachable from
    `checkout` + `uv build`; it is recorded beside both of those steps in
    `release.yml` and is not driven here, because a drive of it is a test that
    hangs.
    """
    valid = _valid_tarball_bytes()
    return {
        # not a gzip stream at all
        "non-gzip garbage": b"this is not a gzip stream\n" * 40,
        # a gzip stream whose payload is not a tar archive, LONG enough to
        # reach a full 512-byte tar block — see `_A_SHORT_GZIP_OF_A_NON_TAR`
        "gzip of a non-tar": gzip.compress(b"not a tar archive\n" * 200),
        # a gzip stream that stops half way
        "truncated gzip": valid[: len(valid) // 2],
        # decompresses, lists a member, and still exits non-zero
        "valid tar.gz + trailing garbage": valid + b"TRAILING GARBAGE" * 50,
        # zero bytes: `gzip: unexpected end of file`
        "an empty file": b"",
    }


#: THE SHAPE THAT PROVES `pipefail` WAS NEVER WHAT CAUGHT A CORRUPT TARBALL.
#: `release.yml` said beside the sdist step that a corrupt tarball "dies in
#: the pipeline above at tar's own rc 2" and that "`pipefail` is what catches
#: that one and not this loop". MEASURED with GNU tar 1.35, that is a property
#: of the corrupt file's LENGTH: `tar tzf` on a gzip whose decompressed
#: payload is under 512 bytes — one tar block — exits **0** and lists nothing
#: (511 bytes rc=0, 512 bytes rc=2, measured either side of the boundary). So
#: this shape walks past `pipefail` entirely and it is the `members.txt` arm of
#: the "examined nothing" loop that refuses it. Driven as its own row below,
#: because a fix that only annotated `tar`'s failures would leave it uncovered
#: and a reader would believe the annotation covers every corrupt tarball.
_A_SHORT_GZIP_OF_A_NON_TAR = gzip.compress(b"not a tar archive\n" * 10)


@_needs_a_shell
def test_the_sdist_gate_refuses_a_tarball_it_cannot_read(tmp_path):
    """THE TWELFTH `exit 1` SITE — the FOURTEENTH refusal point — AND IT
    REPLACED AN EXCUSE RATHER THAN AN OVERSIGHT.

    Arrival order, as everywhere in this module: it is the newest of the twelve
    and stands NINTH among them in file order, which is a number nothing
    derives and nothing pins. See the count pin's docstring.

    `release.yml` recorded this state and LEFT it, in as many words: a corrupt
    tarball "dies in the pipeline above at tar's own rc 2, with no annotation.
    Still red, so still safe" — tolerable because this step IS a refusal point,
    so failing closed is the answer it exists to give. Two things were wrong.

    THE THESIS IS NOT `FAIL CLOSED`, IT IS `A REFUSAL MUST SAY IT REFUSED`.
    rc=2 is TAR's exit code; this step's own refusals are rc=1. This same step
    has been repaired out of exactly that state twice already — the no-wheel
    case ("`ls`'s own rc=2 with no annotation — red, but not by this gate") and
    the two-tarball case — and a reviewer reading the log met a `tar` error
    rather than a statement about `dist/`.

    AND "AN EARLIER STEP WOULD HAVE CAUGHT IT" WAS NEVER AVAILABLE, which is
    the half that makes it live. MEASURED on the bodies the parent commit
    shipped, one tarball in `dist/`, `TAG=v0.2.0`::

        dist/stelling-*.tar.gz was…    tag step   sdist step  annotation
        non-gzip garbage               rc=0       rc=2        none
        a truncated gzip               rc=0       rc=2        none
        valid tar.gz + trailing junk   rc=0       rc=2        none
        an empty file                  rc=0       rc=2        none
        a gzip of a non-tar, >=512 B   rc=0       rc=2        none
        a gzip of a non-tar,  <512 B   rc=0       rc=1        examined nothing

    The tag step reads FILENAMES and never opens either file, so every corrupt
    tarball reaches this step THROUGH A GREEN GATE and nothing before it has
    read a byte of the archive. "An earlier step would have caught it" is the
    argument this whole file exists to retract.

    THE LAST TWO ROWS ARE ONE CLASS SPLIT BY LENGTH, and the split is the
    reason both are driven. "A gzip of something that is not tar" was one row
    in the record this test was written from; measured, `tar` exits 2 on it at
    512 bytes of payload and above, and 0 below — so the same corruption is
    caught by the annotation at one size and by the empty-`members.txt` arm at
    another. `_corrupt_tarballs()`'s own shape is 3600 bytes and therefore the
    rc=2 half; `_A_SHORT_GZIP_OF_A_NON_TAR` is the other, driven separately
    below so that neither half can be lost.

    BOTH FRAMES ARE DRIVEN HERE: the tag body must return 0 on each shape (so
    the gate really is green, and this test fails if that ever stops being
    true and the reason for this refusal quietly evaporates), and the sdist
    body must return 1 with its own annotation. The accepting direction is
    driven too — a readable tarball must still pass, or this is not a gate.
    """
    tag = _step_body(_TAG_STEP)
    body = _step_body(_SDIST_STEP)
    env = _step_env(_SDIST_STEP)

    for index, (label, payload) in enumerate(_corrupt_tarballs().items()):
        # THE GREEN GATE IN FRONT OF IT: filenames only, file never opened.
        upstream = _dist_of(tmp_path, f"upstream-{index}", wheels=("0.2.0",),
                            sdists=("0.2.0",))
        (upstream / "dist" / "stelling-0.2.0.tar.gz").write_bytes(payload)
        passed = _drive(tag, upstream, TAG="v0.2.0")
        assert passed.returncode == 0, (
            f"the tag step now refuses a tarball that is {label}. That is not "
            "a defect, but it is the premise this refusal was written on — "
            "re-derive the reasoning in `release.yml` beside the sdist step "
            f"before deleting anything:\n{passed.stdout}\n{passed.stderr}"
        )

        tree = _plant_tree(tmp_path / f"corrupt-{index}", tarball=payload)
        bad = _drive(body, tree, **env)
        assert bad.returncode == 1, (
            f"a tarball that is {label} does not reach this step's own "
            "refusal. rc=2 is `tar`'s exit code and this step's refusals are "
            "rc=1: red, but not by this gate, and no annotation in the "
            "publish log to say which file could not be read. Nothing earlier "
            "in the workflow opens the tarball, so this is the first step "
            f"that can say anything about it at all.\n{bad.stdout}\n{bad.stderr}"
        )
        assert "cannot read the tarball" in bad.stdout, (
            f"a tarball that is {label} is failing this step without its own "
            f"annotation:\n{bad.stdout}\n{bad.stderr}"
        )
        assert "stelling-0.1.0.tar.gz" in bad.stdout, (
            f"the refusal did not name the file it could not read:\n{bad.stdout}"
        )
        _no_workings_left_behind(tree, f"the corrupt-tarball path ({label})")

    # THE SHAPE `pipefail` NEVER CAUGHT, and it must still be refused — by the
    # OTHER arm. `tar` exits 0 here and lists nothing, so the annotation above
    # cannot fire and "the sdist check examined nothing" is what holds it.
    short = _plant_tree(tmp_path / "short-gzip",
                        tarball=_A_SHORT_GZIP_OF_A_NON_TAR)
    thin = _drive(body, short, **env)
    assert thin.returncode == 1, (
        "a gzip of a short non-tar passed the sdist gate. `tar tzf` exits 0 on "
        "one of those and lists NOTHING, so `pipefail` never saw it and the "
        "new annotation cannot fire either; what refuses it is the empty "
        f"`members.txt` arm.\n{thin.stdout}\n{thin.stderr}"
    )
    assert "examined nothing" in thin.stdout, (
        "the short-gzip shape is no longer refused by the `members.txt` arm. "
        "If it is now refused by the tar annotation, `tar`'s behaviour on a "
        "sub-512-byte payload has changed and the reasoning recorded beside "
        f"that arm needs re-measuring:\n{thin.stdout}"
    )

    # AND A GATE THAT REFUSES EVERYTHING IS NOT A GATE.
    healthy = _plant_tree(tmp_path / "readable")
    ok = _drive(body, healthy, **env)
    assert ok.returncode == 0, (
        f"the sdist gate refuses a readable tarball.\n{ok.stdout}\n{ok.stderr}"
    )


@_needs_a_shell
def test_the_sdist_gate_does_not_read_tars_warnings_as_members(tmp_path):
    """`tar` CAN EXIT 0 AND STILL SPEAK, and what it says is not a member.

    The refusal above needs `tar`'s exit code, which means capturing it —
    and the obvious capture, `2>&1`, folds every warning into the member list.
    That is not hypothetical: it is the defect measured on the SUCCESS path of
    the manifest step below, one step away, in the same commit.

    DRIVEN with a real `stelling-0.1.0.tar.gz` whose every member carries a pax
    extended header with a keyword no `tar` knows. GNU tar 1.35 exits 0 and
    writes `Ignoring unknown extended header keyword 'ANTHROPIC.weird'` to
    STDERR, once per member. On the body this commit ships: rc=0, "every one of
    5 sdist members is committed to this tree", and the warning in the step's
    LOG. With `2>&1` on that same capture: rc=1, and the release refused
    because `tar: Ignoring unknown extended header keyword …` is "in the sdist
    and not in `git ls-files`" — a healthy release turned away by a diagnostic
    dressed as a file.

    Every member of the tarball IS committed here, so the only way this test
    goes red is a diagnostic being counted as one.
    """
    body = _step_body(_SDIST_STEP)
    env = _step_env(_SDIST_STEP)
    tree = _plant_tree(tmp_path / "pax", pax=True)
    result = _drive(body, tree, **env)
    assert result.returncode == 0, (
        "the sdist gate refused a tarball whose every member is committed. If "
        "the refusal names `tar:` something, `tar`'s stderr is being `sed`ed "
        "and `sort`ed into `members.txt` as though it were a member — capture "
        f"stdout only.\n{result.stdout}\n{result.stderr}"
    )
    assert "sdist members is committed to this tree" in result.stdout
    assert "Ignoring unknown extended header" not in result.stdout, (
        "a `tar` diagnostic reached this step's stdout as part of its verdict:"
        f"\n{result.stdout}"
    )
    assert "Ignoring unknown extended header" in result.stderr, (
        "the drive did not reproduce the warning at all, so this test is "
        "vacuous: `tar` did not object to the pax header this plant writes, "
        f"and the shape needs re-deriving.\n{result.stderr}"
    )


@_needs_a_shell
def test_the_manifest_step_records_every_tarball_and_refuses_nothing(tmp_path):
    """`release.yml` calls this step NOT A REFUSAL, AND THE NAME USED TO
    OVERCLAIM: this drove 0, 1 and 2 tarballs and never a CORRUPT one.

    It carried the LAST `ls dist/` in the file — the second of the two
    `ls dist/*.tar.gz` sites — which made a step whose entire job is to PRINT
    capable of failing the release: two tarballs died inside `tar`, none died
    on the unexpanded glob. Adding a count refusal
    here would have contradicted the claim beside it, so the glob is a LOOP —
    every tarball the release would publish is recorded, and nothing is
    asserted about any of them.

    WHAT THE COUNTS DID NOT COVER IS WHAT IS INSIDE ONE FILE. `release.yml`
    said beside this step: *"The only way it goes red is `set -u` on an unset
    `GITHUB_STEP_SUMMARY`."* DRIVEN FALSE FOUR WAYS on the body that sentence
    described, ONE tarball in `dist/` and `GITHUB_STEP_SUMMARY` set::

        non-gzip garbage                     rc=2, no annotation
        a gzip of something that is not tar  rc=2, no annotation
        a truncated gzip                     rc=2, no annotation
        a valid tar.gz + trailing garbage    rc=2, no annotation

    `pipefail` and `set -e` turn `tar`'s own exit code into a failed release,
    from the one step whose entire purpose is to print, with nothing said about
    why. It was unreachable in the workflow ONLY because the sdist step
    `tar tzf`s the same file one step earlier — "an earlier step would have
    caught it", the argument this whole file exists to retract and the one the
    sdist step already refused by establishing its own tarball count.

    THE REPAIR IS NOT A COUNT, deliberately: a count would make this a refusal
    point. `tar` is allowed to fail and what it said goes into the RECORD. All
    five shapes are driven here, in both halves — the step exits 0, and the
    summary says the file could not be listed rather than silently omitting
    it. A readable tarball with no members is driven too, for the same reason
    the sdist step refuses an empty `members.txt`: zero examined must not read
    as zero problems, in a record any more than in a check.

    AND THE REPAIR ARRIVED WITH A DEFECT ON THE SUCCESS PATH, which is the path
    every release takes. `2>&1` was unconditional, so a `tar` that exits 0 and
    WARNS had its warning `sed`ed and `sort`ed into the record beside the
    members. DRIVEN on the body of one commit ago, a real one-member
    `stelling-0.2.0.tar.gz` carrying a pax extended header with an unknown
    keyword — rc=0, and the record::

        PKG-INFO
        tar: Ignoring unknown extended header keyword 'ANTHROPIC.weird'

    Two lines a reader cannot tell apart, in the one step whose entire product
    IS the record — the same family as "zero examined must not read as zero
    problems", which this step's own comment invokes. Driven here in both
    halves: the diagnostic must be OUT of the record and the member must be in
    it, and the diagnostic must still be somewhere a reader can find, which is
    the step's log.
    """
    body = _step_body(_MANIFEST_STEP)

    def _run(name: str, files: dict[str, bytes]):
        tree = tmp_path / name
        (tree / "dist").mkdir(parents=True)
        for filename, payload in files.items():
            (tree / "dist" / filename).write_bytes(payload)
        out = tree / "summary.md"
        out.write_text("", encoding="utf-8")
        result = _drive(body, tree, GITHUB_STEP_SUMMARY=str(out))
        return result, out.read_text(encoding="utf-8")

    def _summary(name: str, sdists: tuple[str, ...]):
        return _run(name, {f"stelling-{v}.tar.gz": _valid_tarball_bytes(v)
                           for v in sdists})

    for label, sdists in (("one", ("0.2.0",)),
                          ("two", ("0.2.0", "0.0.9")),
                          ("none", ())):
        result, written = _summary(label, sdists)
        assert result.returncode == 0, (
            f"the manifest step FAILED on a `dist/` with {len(sdists)} "
            "tarball(s). `release.yml` says beside it that it is not a refusal "
            "and asserts nothing about what it prints; a step that can fail "
            f"the release is a refusal nobody counted.\n{result.stdout}\n"
            f"{result.stderr}"
        )
        for version in sdists:
            assert f"stelling-{version}.tar.gz" in written, (
                f"the manifest for stelling-{version}.tar.gz is not in the "
                f"job summary, so the record of this release does not include "
                f"a file it would publish:\n{written}"
            )
        assert written.count("PKG-INFO") == len(sdists), (
            f"expected one manifest per tarball, got:\n{written}"
        )

    # THE FOUR CORRUPTIONS, one at a time, each a real file on disk.
    for index, (label, payload) in enumerate(_corrupt_tarballs().items()):
        result, written = _run(
            f"corrupt-{index}", {"stelling-0.2.0.tar.gz": payload}
        )
        assert result.returncode == 0, (
            f"THE MANIFEST STEP FAILED THE RELEASE on a tarball that is "
            f"{label}. It is not a refusal point — `release.yml` says so "
            "beside it and gives it no count — so a `dist/` it cannot read "
            "must reach the record, not the exit code. `pipefail` plus `set "
            f"-e` on a bare `tar tzf` is how it used to go red at rc=2 with no "
            f"annotation.\n{result.stdout}\n{result.stderr}"
        )
        assert "stelling-0.2.0.tar.gz" in written, (
            f"a tarball that is {label} is missing from the job summary "
            f"entirely, so the record of this release is silently short one "
            f"file it would publish:\n{written}"
        )
        assert "could not list this file" in written, (
            f"a tarball that is {label} was recorded as though it had been "
            f"read. A record that cannot distinguish 'no members' from 'could "
            f"not be opened' is the shape the sdist step refuses:\n{written}"
        )
        assert "::error" not in result.stdout, (
            "the manifest step emitted an annotation, which makes it a "
            f"refusal the header does not count:\n{result.stdout}"
        )

    # AND A READABLE TARBALL WITH NO MEMBERS, which is not a corruption and
    # must not be recorded as one — nor as nothing.
    empty = io.BytesIO()
    with tarfile.open(fileobj=empty, mode="w:gz"):
        pass
    result, written = _run("empty-archive",
                           {"stelling-0.2.0.tar.gz": empty.getvalue()})
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "holds no members" in written, (
        "a valid but empty tarball is recorded as an empty code fence, so the "
        f"record cannot tell it from a tarball that was never opened:\n{written}"
    )
    assert "could not list this file" not in written, (
        f"an empty but READABLE tarball was recorded as unreadable:\n{written}"
    )

    # AND A TARBALL `tar` READS SUCCESSFULLY WHILE COMPLAINING. The record
    # must hold the member and not the complaint, and the complaint must not
    # simply vanish: it belongs in the step's log.
    pax = io.BytesIO()
    payload = b"Metadata-Version: 2.4\nName: stelling\nVersion: 0.2.0\n"
    with tarfile.open(fileobj=pax, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo("stelling-0.2.0/PKG-INFO")
        info.size = len(payload)
        info.pax_headers = {"ANTHROPIC.weird": "1"}
        tar.addfile(info, io.BytesIO(payload))
    result, written = _run("pax-warning",
                           {"stelling-0.2.0.tar.gz": pax.getvalue()})
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "Ignoring unknown extended header" in result.stderr, (
        "the drive did not reproduce `tar`'s warning at all, so the two "
        "assertions below are vacuous — this `tar` does not object to the pax "
        f"keyword this plant writes:\n{result.stderr}"
    )
    assert "Ignoring unknown extended header" not in written, (
        "a `tar` DIAGNOSTIC is in the manifest, `sed`ed and `sort`ed in beside "
        "the members, and a reader cannot tell one from the other. This step's "
        "whole product is the record. Capture stdout only and ask for stderr "
        f"on the failure path:\n{written}"
    )
    assert "PKG-INFO" in written, (
        "the member is missing from the record of a tarball `tar` read "
        f"successfully:\n{written}"
    )
    assert "could not list this file" not in written, (
        f"a readable tarball was recorded as unreadable:\n{written}"
    )

    # THE ONE RED PATH THAT IS LEFT, driven rather than asserted: `set -u` on
    # an unset `GITHUB_STEP_SUMMARY`. `release.yml` names it as the only one,
    # and that sentence was false four ways until the body above landed.
    tree = tmp_path / "no-summary"
    (tree / "dist").mkdir(parents=True)
    (tree / "dist" / "stelling-0.2.0.tar.gz").write_bytes(_valid_tarball_bytes())
    unset = _drive(body, tree)
    assert unset.returncode != 0, (
        "the manifest step no longer fails on an unset GITHUB_STEP_SUMMARY, so "
        "it is writing this release's record somewhere nobody named: "
        f"{unset.stdout}\n{unset.stderr}"
    )
    assert "GITHUB_STEP_SUMMARY" in unset.stderr, unset.stderr


@_needs_a_shell
def test_the_tag_gate_refuses_a_tag_the_artifact_does_not_carry(tmp_path):
    """THE TAG GATE, DRIVEN. Nothing in the repository pinned this line at all.

    `release.yml`'s header calls this the refusal that stops `v0.2.0` from
    publishing `0.1.0` "silently, and permanently". The version has to come
    from the ARTEFACT — the filename is what gets uploaded::

        version="$(basename "${wheel}" | cut -d- -f2)"

    Read it from the tag instead — `version="$(echo "${TAG}" | sed s/^v//)"` —
    and the comparison compares the tag with itself, so it can never disagree.
    MEASURED: `TAG=v9.9.9` against a `stelling-0.1.0-py3-none-any.whl` gives
    rc=1 unmutated and rc=0 mutated, printing `tag=v9.9.9 artifact
    version=9.9.9` on a `dist/` that holds only 0.1.0.

    THE TWO-WHEEL WEAKNESS USED TO BE DECLARED OUT OF SCOPE HERE, in a
    paragraph that said it "is not re-driven here" because `ls` sorts and a
    stray sorting BEFORE the real wheel is read past. Both halves have moved:
    the step no longer picks, and the sorting explanation was false — see
    :func:`test_the_tag_gate_refuses_a_dist_it_cannot_reason_about`, which
    drives that refusal, and `release.yml`'s own comment, which corrects the
    record. This test keeps its original subject, which is the WHEEL's version
    compared to the tag on a `dist/` the step CAN reason about. The sdist's
    version is the same question asked of the other artefact and is driven in
    :func:`test_the_tag_gate_refuses_an_sdist_the_tag_does_not_name`.

    THE `dist/` HERE NOW CARRIES A TARBALL TOO, and that is not decoration: the
    step refuses a `dist/` without exactly one sdist, so a wheel-only plant
    would go red for a reason that has nothing to do with this test's subject
    and would say nothing about the tag comparison at all.
    """
    body = _step_body(_TAG_STEP)
    tree = _dist_of(tmp_path, "tree")

    agreeing = _drive(body, tree, TAG="v0.1.0")
    assert agreeing.returncode == 0, (
        "the tag gate refuses a tag that MATCHES the built artifacts, so it "
        f"would refuse every release.\n{agreeing.stdout}\n{agreeing.stderr}"
    )
    assert "wheel version=0.1.0" in agreeing.stdout
    assert "sdist version=0.1.0" in agreeing.stdout

    for tag in ("v9.9.9", "0.2.0", "", "V0.1.0"):
        r = _drive(body, tree, TAG=tag)
        assert r.returncode != 0, (
            f"THE TAG GATE PASSED tag {tag!r} against a 0.1.0 wheel. This is "
            "the refusal that stops a release tagged one version from putting "
            "another on PyPI permanently. It reported:\n" + r.stdout +
            "\nCheck that `version` is still read from the ARTEFACT's filename "
            "and not from ${TAG}, which would be the tag compared with itself."
        )
        assert "tag and wheel disagree" in r.stdout


@_needs_a_shell
def test_the_tag_gate_refuses_an_sdist_the_tag_does_not_name(tmp_path):
    """THE HOLE THIS CLOSES SHIPPED A TARBALL NOBODY TAGGED, AND EVERY GATE IN
    THE FILE WAS GREEN ON IT.

    `release.yml` used to end with: *"The sdist's own filename version is still
    never compared to the tag; same build, so it agrees by construction."*
    "Same build" is the premise the wheel COUNT refusal stopped accepting one
    step earlier, and the sdist inherited it. REPRODUCED on the bodies the
    parent branch shipped, `dist/` holding `stelling-0.2.0-py3-none-any.whl`
    beside a `stelling-0.0.9.tar.gz` whose every member is committed,
    `TAG=v0.2.0`::

        the tag step     rc=0   tag=v0.2.0 artifact version=0.2.0
        the sdist step   rc=0   every one of 5 sdist members is committed to
                                this tree

    Two green gates and `stelling-0.0.9.tar.gz` on PyPI under a release called
    `v0.2.0`, permanently. The sdist is not the lesser artefact here: it is
    what `pip install stelling` builds from wherever the wheel does not apply,
    and it is the artefact three steps of this workflow exist to check the
    CONTENTS of while nothing read its name.

    BOTH DIRECTIONS, and the accepting direction is half the point — a gate
    that refuses the shape one `uv build` produces is not a gate.
    """
    body = _step_body(_TAG_STEP)

    matched = _dist_of(tmp_path, "matched", wheels=("0.2.0",), sdists=("0.2.0",))
    ok = _drive(body, matched, TAG="v0.2.0")
    assert ok.returncode == 0, (
        "the tag gate refuses a `dist/` in which BOTH artefacts carry the "
        f"tag.\n{ok.stdout}\n{ok.stderr}"
    )
    assert "sdist version=0.2.0" in ok.stdout, (
        "the step no longer prints the sdist version it compared, so a "
        f"release log cannot show what was checked:\n{ok.stdout}"
    )

    # THE REPRODUCTION, exactly: the tag names the wheel and not the tarball.
    stale = _dist_of(tmp_path, "stale-sdist", wheels=("0.2.0",), sdists=("0.0.9",))
    bad = _drive(body, stale, TAG="v0.2.0")
    assert bad.returncode != 0, (
        "THE TAG GATE PASSED A `dist/` HOLDING A 0.2.0 WHEEL AND A 0.0.9 "
        "SDIST UNDER TAG v0.2.0. `upload-artifact` takes the whole directory "
        "and the publish step uploads every file in it, so the tarball this "
        "comparison never examined goes to PyPI permanently — and an sdist "
        "cannot be unpublished, only yanked. It reported:\n" + bad.stdout
    )
    assert "tag and sdist disagree" in bad.stdout
    assert "0.0.9" in bad.stdout, (
        "the refusal did not name the version it refused, so the release log "
        f"does not say what to fix:\n{bad.stdout}"
    )

    # AND THE MIRROR IMAGE, so the two comparisons are not one comparison
    # applied twice to the same filename: the tarball carries the tag and the
    # wheel does not.
    other = _dist_of(tmp_path, "stale-wheel", wheels=("0.0.9",), sdists=("0.2.0",))
    flipped = _drive(body, other, TAG="v0.2.0")
    assert flipped.returncode != 0, flipped.stdout
    assert "tag and wheel disagree" in flipped.stdout

    # A PEP 440 DEV SEGMENT, which is the shape `cut -d- -f2` on a `.tar.gz`
    # gets wrong if the `.tar.gz` is not stripped first: `basename` alone
    # yields `0.2.0.dev0.tar.gz`, which matches no tag and would refuse the
    # RIGHT tarball too. Driven in both directions.
    dev = _dist_of(tmp_path, "dev-sdist", wheels=("0.2.0",), sdists=("0.2.0.dev0",))
    r = _drive(body, dev, TAG="v0.2.0")
    assert r.returncode != 0, r.stdout
    assert "tag and sdist disagree" in r.stdout
    assert "0.2.0.dev0" in r.stdout, (
        "the sdist version was not extracted from the filename as "
        f"`0.2.0.dev0`:\n{r.stdout}"
    )
    dev_ok = _dist_of(tmp_path, "dev-both", wheels=("0.2.0.dev0",),
                      sdists=("0.2.0.dev0",))
    r = _drive(body, dev_ok, TAG="v0.2.0.dev0")
    assert r.returncode == 0, (
        "the tag gate refuses a pre-release whose two artefacts and tag all "
        f"agree, so `.tar.gz` is being stripped wrong:\n{r.stdout}\n{r.stderr}"
    )
    assert "sdist version=0.2.0.dev0" in r.stdout


def _locales_to_drive() -> list[str]:
    """``C`` plus whatever real language locales this box actually has.

    DISCOVERED AND NOT TYPED, for the reason the step below exists: a test
    that DEMANDED ``en_GB.UTF-8`` would itself be a check whose input is the
    developer's environment, red on a machine that ships only ``C`` and green
    on one that does not. ``C`` is always present, so the drive is never
    vacuous; every further locale found is one more reader whose verdict must
    agree.

    Which locales are worth asking is measured rather than guessed:
    ``C`` and ``C.UTF-8`` collate ``stelling-0.2.0-…`` and
    ``stelling-0.2.0.dev0-…`` by byte, and ``en_GB.UTF-8`` / ``en_US.UTF-8``
    do not, which is the whole of the divergence this step used to carry.
    """
    found = ["C"]
    if shutil.which("locale") is None:
        return found
    proc = subprocess.run(
        ["locale", "-a"], capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        return found
    have = {line.strip().lower().replace("-", "") for line in proc.stdout.splitlines()}
    for want in ("C.UTF-8", "en_GB.UTF-8", "en_US.UTF-8"):
        if want.lower().replace("-", "") in have:
            found.append(want)
    return found


@_needs_a_shell
def test_the_tag_gate_refuses_a_dist_it_cannot_reason_about(tmp_path):
    """THE SEVENTH `exit 1` SITE TO ARRIVE — the NINTH refusal point in
    `release.yml`'s own numbering — AND IT REPLACED A SILENT PICK.

    Arrival order, not file order: it stands third among the twelve `exit 1`
    sites the file now holds. See the count pin's docstring.

    IT SAID "the EIGHTH refusal point" UNTIL THIS COMMIT, and the two halves of
    that heading disagreed with each other. Seven `exit 1` SITES stood once
    this landed, because six stood before it — and the refusal count went from
    EIGHT to NINE for exactly the same reason, so this is the NINTH refusal and
    the SEVENTH site. Numbering a refusal off the count BEFORE it arrives is
    what left `release.yml` with no TWELFTH at all until the renumbering; see
    the header there. There IS one now — the sdist step's own tarball count —
    and it is named wherever it is numbered.

    `ls dist/*.whl` returns EVERY match and `basename` on a multi-line string
    returns whatever follows the LAST `/`. So with two wheels in `dist/` the
    step examined ONE and said nothing at all about the other, and WHICH one it
    examined was decided by the runner's `LC_ALL`. Driven on the OLD body, a
    `dist/` of two empty files, `TAG=v0.2.0`::

        0.2.0 + 0.2.0.dev0   LC_ALL=C            version=0.2.0.dev0  rc=1
        0.2.0 + 0.2.0.dev0   LC_ALL=en_GB.UTF-8  version=0.2.0       rc=0
        0.2.0 + 0.2.0.dev0   LC_ALL=en_US.UTF-8  version=0.2.0       rc=0

    — one tree, one tag, two verdicts, and the green one uploads the untagged
    `0.2.0.dev0` because `upload-artifact` takes the whole directory.

    `LC_ALL=C` ALONE IS NOT THE REPAIR, and this test exists because it is
    not. Driven with only the export added to the old body, the row above goes
    rc=1 under all three locales — and `0.0.9 + 0.1.0` tagged `v0.1.0` is still
    rc=0 in every one of them, so a wheel nobody tagged still ships. What makes
    the answer right is the COUNT: the step establishes there is exactly one
    wheel, and refuses a `dist/` it cannot reason about.

    THE LOCALE LOOP IS A REGRESSION GUARD RATHER THAN A LIVE DIVERGENCE, and
    saying so is the point: the shipped body pins its own `LC_ALL=C` and reads
    `wheels[0]` only after the count is 1, so glob collation cannot reach the
    verdict. The loop is what fails if either of those is undone.

    THE ZERO-WHEEL CASE IS DRIVEN HERE TOO. `release.yml` used to record it as
    a shortcoming — `ls` died on its own rc=2 with no annotation, "red, but not
    by this gate". It is this gate's refusal now, with a message.

    AND THE TARBALL COUNT, which did not exist until the tag comparison was
    carried to the sdist: two tarballs and none, both refused, under every
    locale as well. The construct was the same `ls dist/*.tar.gz` retired here
    for wheels, but the SYMPTOM was not a pick — the assignment is quoted, so
    two matches become ONE string with a newline in it and `tar` fails on a
    filename that does not exist. Driven on the shipped bodies, two tarballs in
    `dist/`: the sdist step exited 2 on tar's own error and printed no
    annotation at all. Same repair, and it belongs here because "will this
    release publish anything the tag does not name" is a question about the
    whole directory. The locale loop covers the tarball shapes for the same
    reason it covers the wheel ones: it is the regression guard on `LC_ALL=C`
    and on the count being read before any element is.
    """
    body = _step_body(_TAG_STEP)

    one = _dist_of(tmp_path, "one", wheels=("0.2.0",), sdists=("0.2.0",))
    stale = _dist_of(tmp_path, "stale", wheels=("0.2.0", "0.2.0.dev0"),
                     sdists=("0.2.0",))
    stray = _dist_of(tmp_path, "stray", wheels=("0.0.9", "0.1.0"),
                     sdists=("0.1.0",))
    empty = _dist_of(tmp_path, "empty", wheels=(), sdists=("0.2.0",))
    two_tarballs = _dist_of(tmp_path, "two-tarballs", wheels=("0.2.0",),
                            sdists=("0.2.0", "0.0.9"))
    no_tarball = _dist_of(tmp_path, "no-tarball", wheels=("0.2.0",), sdists=())

    locales = _locales_to_drive()
    assert "C" in locales, locales

    for locale in locales:
        # A GATE THAT REFUSES EVERYTHING IS NOT A GATE. The shape this release
        # path actually produces — one `uv build`, one wheel, one sdist — must
        # still pass.
        ok = _drive(body, one, TAG="v0.2.0", LC_ALL=locale)
        assert ok.returncode == 0, (
            f"the tag gate refuses the one-wheel one-sdist `dist/` a single "
            f"`uv build` leaves, under LC_ALL={locale}, so it would refuse "
            f"every release.\n{ok.stdout}\n{ok.stderr}"
        )
        assert "wheel version=0.2.0" in ok.stdout
        assert "sdist version=0.2.0" in ok.stdout

        # THE STALE PRE-RELEASE BESIDE ITS OWN RELEASE — the shape whose
        # verdict used to be the runner's locale.
        bad = _drive(body, stale, TAG="v0.2.0", LC_ALL=locale)
        assert bad.returncode != 0, (
            "THE TAG GATE PASSED A `dist/` HOLDING TWO WHEELS under "
            f"LC_ALL={locale}. `upload-artifact` takes the whole directory and "
            "the publish step uploads every file in it, so the wheel this "
            "comparison never examined goes to PyPI permanently. It "
            f"reported:\n{bad.stdout}"
        )
        assert "does not hold exactly one wheel" in bad.stdout
        # AND IT NAMES BOTH. "Examined one and said nothing about the other"
        # is the defect; a refusal that names only one is the same silence.
        for wheel in ("stelling-0.2.0-py3-none-any.whl",
                      "stelling-0.2.0.dev0-py3-none-any.whl"):
            assert wheel in bad.stdout, (
                f"the refusal did not name {wheel}, so the publish log does "
                f"not say what is in `dist/`:\n{bad.stdout}"
            )

        # THE STRAY THAT SORTS FIRST IN EVERY LOCALE — rc=0 on the old body
        # under C and under UTF-8 alike, which is why `LC_ALL=C` was not the
        # repair. The tag AGREES with a wheel here; the directory is still
        # unpublishable.
        low = _drive(body, stray, TAG="v0.1.0", LC_ALL=locale)
        assert low.returncode != 0, (
            "THE TAG GATE PASSED a `dist/` holding an untagged 0.0.9 beside "
            f"the tagged 0.1.0 under LC_ALL={locale}. The tag matches a wheel, "
            "which is not the question: both files are uploaded. It "
            f"reported:\n{low.stdout}"
        )
        assert "does not hold exactly one wheel" in low.stdout

        # NO WHEEL AT ALL is this gate's refusal now, not `ls`'s exit code.
        gone = _drive(body, empty, TAG="v0.2.0", LC_ALL=locale)
        assert gone.returncode != 0, (
            f"the tag gate passed a wheel-less `dist/` under LC_ALL={locale}: "
            f"{gone.stdout}\n{gone.stderr}"
        )
        assert "does not hold exactly one wheel" in gone.stdout, (
            "a wheel-less `dist/` no longer reaches this step's own refusal — "
            "it is dying on a command's exit code with no annotation, which is "
            f"the state this repaired:\n{gone.stdout}\n{gone.stderr}"
        )

        # TWO TARBALLS — the shape that used to reach the SDIST step and die
        # inside `tar` with nothing said. The tag step never opens either file,
        # so it is the count and only the count that refuses.
        pair = _drive(body, two_tarballs, TAG="v0.2.0", LC_ALL=locale)
        assert pair.returncode != 0, (
            "THE TAG GATE PASSED A `dist/` HOLDING TWO TARBALLS under "
            f"LC_ALL={locale}. One of them is a tarball nobody tagged and the "
            "publish step uploads every file in the directory. It "
            f"reported:\n{pair.stdout}"
        )
        assert "does not hold exactly one sdist" in pair.stdout
        for sdist in ("stelling-0.2.0.tar.gz", "stelling-0.0.9.tar.gz"):
            assert sdist in pair.stdout, (
                f"the refusal did not name {sdist}, so the publish log does "
                f"not say what is in `dist/`:\n{pair.stdout}"
            )

        # NO TARBALL AT ALL. `uv build` produces one; a `dist/` without one is
        # a `dist/` this job did not fill, and the release would publish a
        # wheel with no source distribution behind it.
        alone = _drive(body, no_tarball, TAG="v0.2.0", LC_ALL=locale)
        assert alone.returncode != 0, (
            f"the tag gate passed a `dist/` with no sdist under "
            f"LC_ALL={locale}: {alone.stdout}\n{alone.stderr}"
        )
        assert "does not hold exactly one sdist" in alone.stdout


def _a_real_zip_sdist(path: pathlib.Path, version: str = "0.0.9") -> pathlib.Path:
    """A real ZIP archive shaped like a source distribution, on disk.

    NOT an empty file, and that is the point. `.zip` is in twine's
    `DIST_EXTENSIONS` as a `sdist` — measured off twine 7.0.0 as
    `{'.whl': 'bdist_wheel', '.tar.gz': 'sdist', '.zip': 'sdist'}` — and PyPI
    accepts `.zip` source distributions, so the file this drive plants is a
    file the publish action would have recognised and uploaded. The coherent
    version of it (built from this tree's own `stelling-0.2.0.tar.gz` with
    `PKG-INFO`'s `Version:` rewritten to 0.0.9) passed `twine check` at rc=0;
    that check is not repeated here because `twine` is not a dependency of this
    project and the zero-dep job could not import one. What the gate reads is
    the DIRECTORY ENTRY, so a real archive with a real `PKG-INFO` is all this
    drive needs to be honest about what it is planting.
    """
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"stelling-{version}/PKG-INFO",
            f"Metadata-Version: 2.4\nName: stelling\nVersion: {version}\n",
        )
        archive.writestr(f"stelling-{version}/pyproject.toml", "[project]\n")
    return path


@_needs_a_shell
def test_the_tag_gate_refuses_a_dist_entry_it_did_not_build(tmp_path):
    """THE THIRTEENTH REFUSAL POINT: THE UNIT OF THE ANSWER IS THE DIRECTORY.

    This heading said "the THIRTEENTH `exit 1` SITE" when the file held ELEVEN
    of them. It means the thirteenth REFUSAL POINT in arrival order — the count
    `release.yml`'s header moved to when this landed — a distinction that file
    maintains and this one had dropped. It stands fifth among the twelve `exit
    1` sites in file order, and no test derives that.

    `release.yml` said, in its own header, that a `dist/` file which is neither
    a `.whl` nor a `.tar.gz` is invisible to every step — and then excused it:
    *"`pypa/gh-action-pypi-publish` rejects what it cannot recognise as a
    distribution."* MEASURED, false for the case that matters. twine's
    `DIST_EXTENSIONS` carries `.zip` as a SOURCE DISTRIBUTION; a coherent
    `stelling-0.0.9.zip` built from this tree's real `stelling-0.2.0.tar.gz`
    passed `twine check` at rc=0; and dropped into a real `dist/` beside the
    real wheel and tarball with `TAG=v0.2.0`, on the bodies the parent commit
    shipped::

        the tag step        rc=0   tag=v0.2.0 wheel version=0.2.0 sdist
                                   version=0.2.0
        the sdist step      rc=0   every one of 372 sdist members is committed
                                   to this tree
        the manifest step   rc=0   375 lines of manifest written

    Three green gates and a `0.0.9` source distribution on PyPI under a release
    called v0.2.0, permanently — the same defect the wheel COUNT and the sdist
    COUNT refusals were written for, one file extension over, resting on an
    escape hatch in third-party code behind a mutable branch ref.

    SO THE STEP INVENTORIES `dist/`, and this drives both directions of that.
    Every shape below is planted on disk; the `.zip` is a real archive, not an
    empty file, because the claim being retracted was about what a publisher
    RECOGNISES.

    THE DOTFILE DECISION IS DRIVEN AS TWO ROWS, not asserted. `uv build` writes
    `dist/.gitignore` itself, so the accepting shape HAS a dotfile in it and
    must still pass; a dotfile that is not that one must still be refused. That
    is why the exception is one entry in the step and not "ignore hidden
    files" — and the reason is NOT that `actions/upload-artifact@v4` excludes
    hidden files by default. THIS SUITE AND `release.yml` MAKE TWO CLAIMS ABOUT
    THAT ACTION AND RELY ON NEITHER: that it excludes hidden files by default,
    and that it takes the whole directory and uploads every file in it. They
    point opposite ways, both are third-party behaviour behind a mutable ref,
    and the conduct is the same under either — `dist/` is inventoried whole and
    every entry that is not one of the two established artefacts is refused by
    name. The whole-directory reading is the one written into the messages
    because it is the WORSE case. Ignoring the dotfile class would wave a
    `dist/.stelling-0.0.9.zip` straight through whichever reading is true.
    """
    body = _step_body(_TAG_STEP)

    # THE ACCEPTING SHAPE, with the dotfile a real `uv build` leaves.
    ok = _drive(body, _dist_of(tmp_path, "real", wheels=("0.2.0",),
                               sdists=("0.2.0",)), TAG="v0.2.0")
    assert ok.returncode == 0, (
        "the tag gate refuses the `dist/` a single `uv build` leaves — one "
        "wheel, one sdist and the `.gitignore` uv writes itself — so it would "
        f"refuse every release.\n{ok.stdout}\n{ok.stderr}"
    )

    # AND WITHOUT THE DOTFILE, so `dist/.gitignore` is an exception and not a
    # requirement: a `uv build --out-dir` elsewhere, or a cleaned directory,
    # must not be refused for the file it does not have.
    bare = _dist_of(tmp_path, "bare", wheels=("0.2.0",), sdists=("0.2.0",))
    (bare / "dist" / ".gitignore").unlink()
    still_ok = _drive(body, bare, TAG="v0.2.0")
    assert still_ok.returncode == 0, (
        "the tag gate now REQUIRES `dist/.gitignore`, which `uv build` happens "
        "to write and nothing guarantees. It is an allowed entry, not a "
        f"demanded one.\n{still_ok.stdout}\n{still_ok.stderr}"
    )

    def _extra(label, plant) -> tuple[pathlib.Path, str]:
        tree = _dist_of(tmp_path, label, wheels=("0.2.0",), sdists=("0.2.0",))
        return tree, plant(tree / "dist")

    def _zip(dist):
        _a_real_zip_sdist(dist / "stelling-0.0.9.zip")
        return "dist/stelling-0.0.9.zip"

    def _dotzip(dist):
        _a_real_zip_sdist(dist / ".stelling-0.0.9.zip")
        return "dist/.stelling-0.0.9.zip"

    def _notes(dist):
        (dist / "notes.txt").write_text("left here by a human\n", encoding="utf-8")
        return "dist/notes.txt"

    def _subdir(dist):
        (dist / "stray").mkdir()
        (dist / "stray" / "stelling-0.0.9.tar.gz").write_bytes(b"")
        return "dist/stray"

    for label, plant in (("zip", _zip), ("dotzip", _dotzip),
                         ("notes", _notes), ("subdir", _subdir)):
        tree, named = _extra(label, plant)
        bad = _drive(body, tree, TAG="v0.2.0")
        assert bad.returncode != 0, (
            f"THE TAG GATE PASSED A `dist/` HOLDING {named}. `upload-artifact` "
            "takes the whole directory and the publish step uploads what it "
            "globs, so an entry no gate examined is an entry that ships. For "
            "the `.zip` rows this is not hypothetical: twine recognises `.zip` "
            "as a source distribution and PyPI accepts one. It "
            f"reported:\n{bad.stdout}"
        )
        assert "holds files this job did not build" in bad.stdout, (
            f"the {label} shape is not reaching this step's own refusal:\n"
            f"{bad.stdout}\n{bad.stderr}"
        )
        assert named in bad.stdout, (
            f"the refusal did not name {named}, so the publish log does not "
            f"say what is in `dist/`:\n{bad.stdout}"
        )
        # and the two artefacts must NOT be named: a refusal that lists the
        # files it accepted is a refusal a reader cannot act on.
        for artefact in ("dist/stelling-0.2.0-py3-none-any.whl",
                         "dist/stelling-0.2.0.tar.gz"):
            assert f"  {artefact}\n" not in bad.stdout, (
                f"the refusal listed {artefact}, which is one of the two "
                f"artefacts it just established:\n{bad.stdout}"
            )

    # THREE AT ONCE, so the refusal is an inventory and not a first-hit.
    many = _dist_of(tmp_path, "many", wheels=("0.2.0",), sdists=("0.2.0",))
    _a_real_zip_sdist(many / "dist" / "stelling-0.0.9.zip")
    (many / "dist" / "notes.txt").write_text("x\n", encoding="utf-8")
    (many / "dist" / "stelling-0.2.0.tar.gz.asc").write_text("x\n", encoding="utf-8")
    all_of_them = _drive(body, many, TAG="v0.2.0")
    assert all_of_them.returncode != 0, all_of_them.stdout
    for name in ("dist/stelling-0.0.9.zip", "dist/notes.txt",
                 "dist/stelling-0.2.0.tar.gz.asc"):
        assert name in all_of_them.stdout, (
            "the refusal named some of the unaccounted entries and not all of "
            f"them, so it is reporting a first hit rather than an inventory — "
            f"{name} is missing:\n{all_of_them.stdout}"
        )


@_needs_a_shell
def test_the_tag_gate_takes_the_gitignore_exception_on_content(tmp_path):
    """THE EXCEPTION WAS BY NAME, AND A NAME IS NOT A FILE.

    `dist/.gitignore` is allowed because `uv build` writes it. It was allowed
    as a LITERAL PATH, so the one check in this workflow whose stated unit is
    the DIRECTORY exempted an entry of arbitrary type, size and content from
    itself, on its spelling. MEASURED on the body one commit ago, `TAG=v0.2.0`
    against an otherwise real `dist/`, only that entry changed::

        dist/.gitignore is                              verdict
        arbitrary text                                  rc=0
        0 bytes                                         rc=0
        a symlink to /etc/passwd                        rc=0
        a directory holding a .zip                      rc=0
        a coherent 4.4 MB stelling-0.0.9.zip renamed    rc=0

    Nothing named in any of them. The last row is the one that matters: that
    archive is the same one built from this tree's own `stelling-0.2.0.tar.gz`
    with `PKG-INFO`'s `Version:` rewritten, 4415389 bytes, `twine check` rc=0
    on twine 7.0.0 — the file the thirteenth refusal exists to stop. None of
    these is weaponisable to PyPI TODAY, because the publish step uploads by
    name and nothing calls a `.gitignore` a distribution; but "not
    weaponisable today, resting on a third-party table and a mutable ref" is
    precisely the argument `release.yml` spends sixty lines refusing to make,
    and its own header had ALREADY MEASURED the content (one byte, `*`) and
    not used it.

    SO THE EXCEPTION IS TAKEN ON THE CONTENT: a REGULAR FILE (not a symlink,
    not a directory) of exactly one byte, and that byte `*`. It adds no
    `exit 1` site and does not move the header's count — an entry this step
    will not vouch for is named by the refusal that was already there — and it
    stays ALLOW rather than REQUIRE, which the absent row below holds.

    NOTHING PINNED THIS BEHAVIOURALLY BEFORE. The literal was pinned as text,
    which is why a widening to `dist/.gitignore*` would have been caught and
    every row above would not.
    """
    body = _step_body(_TAG_STEP)

    def _verdict(label, plant):
        tree = _dist_of(tmp_path, label, wheels=("0.2.0",), sdists=("0.2.0",))
        entry = tree / "dist" / ".gitignore"
        entry.unlink()
        plant(entry)
        result = _drive(body, tree, TAG="v0.2.0")
        return result

    # THE ACCEPTING ROW: what `uv build` actually writes, one byte, no newline.
    ok = _verdict("gi-real", lambda at: at.write_text("*", encoding="utf-8"))
    assert ok.returncode == 0, (
        "the tag gate refuses the `dist/.gitignore` `uv build` writes — one "
        f"byte, `*` — so it would refuse every release.\n{ok.stdout}\n{ok.stderr}"
    )

    def _one_byte_x(at):
        at.write_text("x", encoding="utf-8")

    def _star_and_a_newline(at):
        at.write_text("*\n", encoding="utf-8")

    def _zip_renamed(at):
        _a_real_zip_sdist(at)

    def _symlink_to_a_star(at):
        # A symlink whose TARGET is a perfectly good `.gitignore`. `-f` follows
        # a symlink, so this row is what makes `! -L` load-bearing rather than
        # decorative — and a symlink is not what `uv build` writes.
        target = at.parent.parent / "a-real-gitignore"
        target.write_text("*", encoding="utf-8")
        at.symlink_to(target)

    def _directory(at):
        at.mkdir()
        _a_real_zip_sdist(at / "stelling-0.0.9.zip")

    rows = (
        ("arbitrary text", lambda at: at.write_text("left here by a human\n",
                                                    encoding="utf-8")),
        ("0 bytes", lambda at: at.write_text("", encoding="utf-8")),
        ("one byte that is not `*`", _one_byte_x),
        ("`*` and a newline", _star_and_a_newline),
        ("a real .zip renamed", _zip_renamed),
        ("a symlink to a file whose content is `*`", _symlink_to_a_star),
        ("a directory holding a .zip", _directory),
    )
    # Indexed, not hashed: `hash` of a str is salted per interpreter, so a
    # name derived from it is a different directory on every run and two
    # labels could collide into one.
    for index, (label, plant) in enumerate(rows):
        bad = _verdict(f"gi-{index}", plant)
        assert bad.returncode != 0, (
            f"THE TAG GATE PASSED a `dist/.gitignore` that is {label}. The "
            "exception exists because `uv build` writes ONE byte, `*`, and "
            "this entry is not that. The unit of this check is the DIRECTORY; "
            "exempting an entry of arbitrary type, size and content on its "
            f"spelling is not an inventory of one.\n{bad.stdout}"
        )
        assert "holds files this job did not build" in bad.stdout, (
            f"a `dist/.gitignore` that is {label} is not reaching this step's "
            f"own refusal:\n{bad.stdout}\n{bad.stderr}"
        )
        assert "dist/.gitignore" in bad.stdout, (
            f"the refusal did not name `dist/.gitignore` ({label}), so the "
            f"publish log does not say what is in `dist/`:\n{bad.stdout}"
        )

    # AND THE CHECK'S OWN FAILURE MODE, which must fail CLOSED. With the entry
    # unreadable, `wc -c` cannot read it, the `[` errors, the `&&` chain is
    # false and the entry is named like any other — an entry this step cannot
    # READ is not an entry it vouches for. Skipped as root, which can read it.
    if os.geteuid() != 0:
        unreadable = _verdict(
            "gi-unreadable",
            lambda at: (at.write_text("*", encoding="utf-8"),
                        os.chmod(at, 0o000)),
        )
        assert unreadable.returncode != 0 and "dist/.gitignore" in unreadable.stdout, (
            "a `dist/.gitignore` this step cannot READ was passed over. The "
            "exception is a statement about the file's content; a content "
            f"nobody could read is not one to vouch for.\n{unreadable.stdout}"
        )

    # AND STILL ALLOW, NOT REQUIRE — the same row the test above drives, kept
    # here beside the content rows because it is the half a stricter reading
    # would break: a `dist/` with no `.gitignore` at all is not a `dist/` this
    # step may refuse.
    absent = _dist_of(tmp_path, "gi-absent", wheels=("0.2.0",), sdists=("0.2.0",))
    (absent / "dist" / ".gitignore").unlink()
    gone = _drive(body, absent, TAG="v0.2.0")
    assert gone.returncode == 0, (
        "the tag gate now REQUIRES `dist/.gitignore`. It is an allowed entry, "
        f"not a demanded one.\n{gone.stdout}\n{gone.stderr}"
    )


# THE CHANGELOG DATE GATE — the fifteenth refusal point, and the one that was
# WRITTEN DOWN IN ANOTHER FILE BEFORE IT EXISTED HERE.
# `tests/test_changelog_names_the_version.py` routed it to `release.yml` by
# name and with a command, and `release.yml` did not have it for the 47
# commits between `0e79ede` and `9b5b496` (`git rev-list --count`). Prose
# asserting a check that does not exist is the same defect as the missing
# check; these drives are the half that makes the check a fact rather than a
# recommendation, and
# `tests/test_changelog_names_the_version.py::test_the_routed_date_check_is_in_this_workflow`
# is the half that makes the ROUTING a fact.
_CHANGELOG_STEP = "the tag and the changelog heading must agree"

#: The two releases this project has actually cut, with the TAGGER date of
#: each — `git for-each-ref --format='%(taggerdate:short)' refs/tags/vX`, read
#: 2026-08-28 at `9b5b496`, where `git log -1 --format=%cd --date=short` gives
#: the same answer for both. PLANTED here and not read out of the checkout
#: this test runs in: a drive that resolved them from its own tree would be
#: green in an unpacked sdist with no `.git` at all (`tests/` is in the sdist
#: allowlist, so this module runs there), and would move under anyone who
#: re-cut a tag. They are a dated record of two tags, which is what the green
#: rows need.
_REAL_RELEASES = (("v0.2.0", "2026-08-25"), ("v0.1.0", "2026-08-12"))


# THE IDENTITY AND THE THREE SETTINGS THAT WOULD MAKE THESE DRIVES' VERDICTS A
# PROPERTY OF WHOEVER RAN THEM. A global `commit.gpgsign` / `tag.gpgSign` turns
# every commit and annotated tag below into a signing prompt, and a global
# `core.hooksPath` runs somebody's hooks over a scratch commit — so commits are
# `--no-verify` and both signing settings are pinned off HERE rather than
# inherited. A check whose input includes the developer's environment reports a
# different truth to different people, which is the class the gate under test
# exists to keep OUT of the release path.
#
# WRITTEN ONCE, at module level, because two functions plant tags now: the
# tree-builder below and the remedy drive, which RE-CUTS a tag the way the
# refusal message tells a maintainer to. A second copy of these flags is a
# second place for the guard to be forgotten.
_GIT_IDENT = ("-c", "user.name=release gates",
              "-c", "user.email=gates@example.invalid",
              "-c", "commit.gpgsign=false",
              "-c", "tag.gpgSign=false")


def _git(tree: pathlib.Path, *args, **kwargs):
    """`git` in `tree`, failing loudly, with this module's identity guards."""
    return subprocess.run(
        ["git", *args], cwd=tree, check=True, capture_output=True, **kwargs
    )


def _annotate(tree: pathlib.Path, tag: str, tagger_date: str,
              *, message: str | None = None, at: str | None = None,
              force: bool = False) -> None:
    """Cut `tag` as an ANNOTATED tag whose tagger date is `tagger_date`.

    `GIT_COMMITTER_DATE` is what `git tag -a` writes into the tagger line —
    the date is planted the way git plants it rather than by writing a tag
    object by hand. `at` names the object, so the remedy drive can re-cut at
    the SAME commit the way the refusal message says to.

    `force=True` adds `-f`, and with `at` naming the TAG ITSELF that is how a
    NESTED tag is produced — git does not dereference, it warns *"You have
    created a nested tag"* and points the new object at the old one. That is
    an ordinary maintainer move (`git tag -a -f v0.2.1 v0.2.1`) and it leaves
    the superseded object reachable from the live ref, so it arrives in every
    checkout. THIS SENTENCE ENDED *"which is why the object-store fallback has
    to exclude objects that are some other tag object's target"*, and there is
    no such fallback any more — it was deleted when the workflow stopped
    letting the checkout un-name the tag. The shape still matters, for a
    simpler reason: it is what the `nested-reannotation` row drives, and a
    step that read anything other than the ref would take the SUPERSEDED
    object's date.
    """
    _git(tree, *_GIT_IDENT, "tag", "-a", *(("-f",) if force else ()), tag,
         *( (at,) if at else () ),
         "-m", message or f"release {tag}",
         env=dict(os.environ,
                  GIT_COMMITTER_DATE=f"{tagger_date}T12:00:00+00:00"))


def _tagged_tree(base: pathlib.Path, name: str, *, tag: str = "v0.2.0",
                 tagger_date: str = "2026-08-25", annotated: bool = True,
                 tagged: bool = True, origin: bool = True,
                 changelog: str | None = "## 0.2.0 — 2026-08-25\n") -> pathlib.Path:
    """A checkout carrying one tag and one `CHANGELOG.md`, as the tagged tree has.

    `annotated=False` gives the LIGHTWEIGHT shape, whose
    `%(taggerdate:short)` is empty and which this gate refuses by name;
    `tagged=False` gives a checkout with no such ref at all;
    `changelog=None` gives one with no `CHANGELOG.md`.

    Nothing else is planted. The step reads a ref, a tag object and one line
    of one file, so a `dist/`, an index or a second commit would be furniture
    the drive could not learn anything from.

    **AND IT HAS AN `origin`, WHICH IS NOT FURNITURE — THOUGH THE REASON
    GIVEN HERE WAS WRONG AND IS CORRECTED RATHER THAN RESTATED.** It read:
    *"Without one, every drive below would exercise the 'this step
    could not go and look' arm and none of them the arm they were written
    for."* That
    was never true of these trees and is plainly false now: the tag ref in a
    tree built here is INTACT — nothing clobbered it — so the step reads it
    directly and its re-fetch is a refresh rather than a repair. Measured:
    with `origin=False` the green annotated rows still exit 0. The real reason the remote is here is the REMEDY drive, which
    carries out the printed recipe including `git push origin
    :refs/tags/<tag>` and the push of the re-cut tag; with the tag re-fetched
    from `origin`, a recipe carried out only locally is undone by the step on
    the next run. The origin is a bare clone taken AFTER the tag is planted,
    so what it holds is what the row planted.

    `origin=False` is not a workspace shape and no test drives one: every one
    of its uses builds an UPSTREAM for
    :func:`_checkout_the_way_actions_checkout_does` to fetch from, which is a
    different job. The sentence claiming it was *"the third shape … driven in
    test_the_changelog_gate_reads_…"* is withdrawn; workspaces whose remote
    cannot be reached are built there by removing the remote after the
    checkout, not by this flag.
    """
    tree = base / name
    tree.mkdir(parents=True)
    _git(tree, "-c", "init.defaultBranch=main", "init", "-q")
    _git(tree, *_GIT_IDENT, "commit", "-q", "--no-verify", "--allow-empty",
         "-m", "the tagged tree")
    if tagged:
        if annotated:
            _annotate(tree, tag, tagger_date)
        else:
            _git(tree, "tag", tag)
    if origin:
        remote = base / f"{name}.origin.git"
        subprocess.run(["git", "clone", "-q", "--bare", str(tree), str(remote)],
                       check=True, capture_output=True)
        _git(tree, "remote", "add", "origin", str(remote))
    if changelog is not None:
        (tree / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return tree


@_needs_a_shell
def test_the_changelog_gate_refuses_a_heading_the_tag_does_not_carry(tmp_path):
    """THE FIFTEENTH REFUSAL POINT, DRIVEN IN BOTH DIRECTIONS on the body
    `release.yml` ships — greens included, because a gate driven only red is
    not known to let a real release through and this workflow's mistakes are
    immutable.

    THE GREEN ROWS ARE THE TWO RELEASES THIS PROJECT HAS ACTUALLY CUT, tag and
    tagger date and heading, plus the same tags spelled without the leading
    `v` (the step beside this one accepts both spellings and this one must
    agree with it, or a release that satisfies one gate is refused by the
    other). They are planted from :data:`_REAL_RELEASES` rather than read from
    the checkout — see its comment.

    AND THE GREENS REFUTE A CLOCK, which is the whole reason the check is here
    and not in the suite that routed it. Their dates are fixed in the past, so
    a gate that compared the heading against `date +%F` — the environment
    dependence
    `tests/test_changelog_names_the_version.py::test_this_gate_does_not_read_the_clock`
    exists to keep out of that module — is RED on every one of them for any
    reader whose clock is not exactly 2026-08-25. That is weaker than a pin
    and is why :func:`test_the_drives_are_reading_the_real_step_bodies` also
    asserts the construct absent; it is stronger than nothing, and it is the
    direction a text pin cannot reach.

    THE RED ROWS ARE THE SPEC'S, plus the ways the step cannot READ what it
    compares — each of which is a refusal here rather than a pass, because an
    instrument that cannot see is not an instrument that saw nothing:

        planted (row label)                          expected
        ───────────────────────────────────────────  ────────────────────────
        v0.2.0 / 2026-08-25 / `## 0.2.0 — …`         GREEN
        v0.1.0 / 2026-08-12 / `## 0.1.0 — …`         GREEN
        the same two, tag spelled without the `v`    GREEN
        the same, changelog written CRLF             GREEN
        `wrong-version`  `## 0.1.0 …` under v0.2.0   red, names both versions
        `unreleased`  `## 0.2.0 — unreleased`        red, on that branch's own
                                                     sentence and not on the
                                                     word, which the DATE
                                                     complaint also echoes
        `unreleased-cased`  `— Unreleased`           red, on the same branch
        `date-off-by-one`  a day before the tag      red, names both dates
        `no-changelog`  no `CHANGELOG.md` at all     red, names the file
        `no-heading`  no `## ` line in it            red, says so
        `unparseable-newest`  `## 0.2.0 = …` above
          a well-formed `## 0.1.0 — …`               red, and does NOT read
                                                     past it to the older one
        `lightweight-tag`  no tag object             red, names the shape
        `no-such-tag`  no such ref in the checkout   red, names the ref

    THE TABLE IS HELD TO THE ROWS BELOW rather than being prose about them —
    every label in `rows` must appear in this docstring, asserted at the end
    of the test. WHAT THAT DOES NOT REACH, said rather than implied: the three
    drives written out AFTER the loop — the empty and unset `GITHUB_REF_NAME`,
    the heading wrong in the version AND the date, and the unreadable tag
    beside a disagreeing heading — are not in `rows` and are described in
    their own comments beside themselves. And under `python -OO` there is no
    docstring at all, so the check has nothing to hold and says so instead of
    reporting every row missing.

    ONE REFUSAL, EVERY COMPLAINT — the drive after the loop plants a heading
    that is wrong in the version AND in the date and asserts BOTH are in the
    output. `release.yml` spends a paragraph on why that is one `exit 1` site
    and not eight, and this is what holds it: a step rewritten to `exit 1` on
    the first complaint passes every other row here.

    WHAT THE DRIVE BUYS OVER THE TEXT PIN BESIDE IT, measured on this branch
    rather than argued — two rewrites of `release.yml`, each ONE token, each
    leaving every needle in
    :func:`test_the_drives_are_reading_the_real_step_bodies` and every scan in
    this module green:

    * the date comparison's right-hand side, `"${tag_date}"` ->
      `"${heading_rest}"`, so the heading is compared with itself and the date
      half of this gate cannot fire. Driven: the `date-off-by-one` row goes
      from red to `rc=0` reporting *"tag=v0.2.0 version=0.2.0
      tagged=2026-08-25; CHANGELOG.md newest heading: 0.2.0 — 2026-08-24"*,
      and this test is the only thing in the repository that reddens.
    * the heading scan's condition, `=~ $heading_any` ->
      `=~ $heading_re` with a version equality on the capture, which is the
      obvious "simplification": look for A heading naming the tag's version
      instead of THE newest one. Driven: this test and
      :func:`test_the_changelog_gate_reads_the_NEWEST_heading_by_POSITION` go
      red together, the second one on a tree tagged `v0.1.0` whose changelog
      leads with `## 0.2.0` and which the mutant passes at rc=0.

    Neither is a spelling a pin was written for, and that is the point:
    a pin catches the mutation it names, a drive catches the one nobody did.
    """
    body = _step_body(_CHANGELOG_STEP)

    for tag, date in _REAL_RELEASES:
        version = tag.lstrip("v")
        for spelling in (tag, version):
            # THE TAG IS PLANTED UNDER THE SPELLING BEING DRIVEN, because
            # `GITHUB_REF_NAME` IS a ref name: a project that tags `0.2.0`
            # has `refs/tags/0.2.0` and no `refs/tags/v0.2.0`. Planting
            # `v0.2.0` and asking about `0.2.0` drives the missing-ref
            # refusal instead, which is a different row of this table.
            tree = _tagged_tree(
                tmp_path, f"green-{spelling}", tag=spelling, tagger_date=date,
                changelog=f"## {version} — {date}\n\n- a release note\n",
            )
            ok = _drive(body, tree, GITHUB_REF_NAME=spelling)
            assert ok.returncode == 0, (
                f"the changelog gate REFUSES the real {tag} shape spelled "
                f"{spelling!r} — tag dated {date}, heading `## {version} — "
                f"{date}` — so it would refuse a correct release.\n"
                f"{ok.stdout}\n{ok.stderr}"
            )
            assert f"tagged={date}" in ok.stdout, (
                f"the step no longer prints the tag date it compared, so a "
                f"release log cannot show what was checked:\n{ok.stdout}"
            )
            assert f"{version} — {date}" in ok.stdout, (
                f"the step no longer prints the heading it read:\n{ok.stdout}"
            )

    # A CRLF CHANGELOG IS THE SAME GREEN, and it is driven rather than
    # reasoned about. Every `^`-anchored reader in this repository has met a
    # carriage return eventually — `_one_line_break` above exists for it, and
    # `tests/test_tripwire_record.py` met it three times — so the one reader
    # that decides a release's date does not get to be correct by accident of
    # how the file was written. `read -r` keeps the `\r` on the line; the
    # heading pattern's trailing `[[:space:]]*` is what absorbs it. Without
    # that trailing class the date would read `2026-08-25\r`, match nothing,
    # and refuse a correct release on a Windows checkout.
    crlf = _tagged_tree(
        tmp_path, "green-crlf", tag="v0.2.0", tagger_date="2026-08-25",
        changelog="## 0.2.0 — 2026-08-25\r\n\r\n- a release note\r\n",
    )
    ok = _drive(body, crlf, GITHUB_REF_NAME="v0.2.0")
    assert ok.returncode == 0, (
        "the changelog gate refuses a CRLF `CHANGELOG.md` whose heading is "
        f"correct.\n{ok.stdout}\n{ok.stderr}"
    )
    assert "0.2.0 — 2026-08-25" in ok.stdout, (
        f"the heading read out of a CRLF file carries its carriage return "
        f"into the comparison:\n{ok.stdout!r}"
    )

    rows = (
        ("wrong-version", dict(changelog="## 0.1.0 — 2026-08-25\n"),
         ("names version 0.1.0", "names version 0.2.0"),
         "a heading naming the PREVIOUS release under this tag is the shape a "
         "forgotten changelog bump has, and it is the one the routed "
         "recommendation was written for"),
        # THE NEEDLE IS THE BRANCH'S OWN SENTENCE AND NOT THE WORD. `'${rest}'`
        # is echoed by the DATE complaint too, so a row asserting only
        # `unreleased` would be satisfied by the gate taking the wrong branch
        # and refusing for the wrong reason — a check that its claim is
        # well-formed rather than true.
        ("unreleased", dict(changelog="## 0.2.0 — unreleased\n"),
         ("a release that has not happened",),
         "a final tag over an `unreleased` heading is a release note "
         "describing a release that has not happened"),
        ("unreleased-cased", dict(changelog="## 0.2.0 — Unreleased\n"),
         ("a release that has not happened", "Unreleased"),
         "the `unreleased` test is case-sensitive, so a capitalised heading "
         "falls through to the date comparison and is refused for the wrong "
         "reason"),
        ("date-off-by-one", dict(changelog="## 0.2.0 — 2026-08-24\n"),
         ("is dated '2026-08-24'", "was made on 2026-08-25"),
         "a heading one day off the tag is the whole subject of this gate"),
        ("no-changelog", dict(changelog=None),
         ("no readable regular file called CHANGELOG.md",),
         "a tagged tree with no changelog is a release with no release note, "
         "and this step must refuse rather than find nothing to compare"),
        ("no-heading", dict(changelog="# Changelog\n\nnothing yet.\n"),
         ("no '## ' heading",),
         "a changelog whose headings stopped parsing is a changelog a "
         "version-scanning gate reads as satisfied"),
        ("unparseable-newest",
         dict(changelog="## 0.2.0 = 2026-08-25\n\n## 0.1.0 — 2026-08-12\n"),
         ("does not parse",),
         "a MALFORMED newest heading must be refused, not stepped over to an "
         "older one the gate can parse — reading past it is how the wrong "
         "heading gets compared"),
        ("lightweight-tag", dict(annotated=False),
         ("not an annotated tag", "git tag -a"),
         "a lightweight tag carries no tagger date at all, and the reading "
         "this gate chose is the tagger date. Falling back to the committer "
         "date here would be a can't-tell resolving to the permissive answer"),
        ("no-such-tag", dict(tagged=False),
         ("there is no refs/tags/v0.2.0 in this checkout",),
         "a checkout without the tag ref cannot answer what date the tag "
         "carries, and a step that cannot read its input does not vouch for "
         "it"),
    )
    for label, plant, needles, why in rows:
        tree = _tagged_tree(tmp_path, f"red-{label}", **plant)
        red = _drive(body, tree, GITHUB_REF_NAME="v0.2.0")
        assert red.returncode != 0, (
            f"THE CHANGELOG GATE PASSED the {label!r} shape: {why}. It "
            f"reported:\n{red.stdout}\n{red.stderr}"
        )
        assert "the tag and the changelog heading disagree" in red.stdout, (
            f"{label}: the step went red without its own annotation, which is "
            f"the state the sdist step's `tar` refusal was repaired out of — "
            f"red, but not by this gate.\n{red.stdout}\n{red.stderr}"
        )
        for needle in needles:
            assert needle in red.stdout, (
                f"{label}: the refusal does not say {needle!r}, so the "
                f"release log does not name what to fix:\n{red.stdout}"
            )

    # THE ENVIRONMENT ITSELF, in both of its absent spellings. `set -u` on a
    # bare `${GITHUB_REF_NAME}` is rc=1 with NO annotation, which is a red
    # this gate does not own; `${GITHUB_REF_NAME:-}` plus a named complaint is
    # what makes it one. EMPTY and UNSET are driven separately because they
    # are different shell conditions and only one of them is `set -u`'s.
    tree = _tagged_tree(tmp_path, "red-no-refname")
    for label, env in (("empty", {"GITHUB_REF_NAME": ""}), ("unset", {})):
        red = _drive(body, tree, **env)
        assert red.returncode != 0, f"{label} GITHUB_REF_NAME: {red.stdout}"
        assert "the tag and the changelog heading disagree" in red.stdout, (
            f"{label} GITHUB_REF_NAME: the step died without an annotation of "
            f"its own — `set -u`'s 'unbound variable' is not this gate "
            f"refusing.\n{red.stdout}\n{red.stderr}"
        )
        assert "GITHUB_REF_NAME is empty or unset" in red.stdout, red.stdout

    # ONE REFUSAL, EVERY COMPLAINT.
    both = _tagged_tree(tmp_path, "red-both",
                        changelog="## 0.1.0 — 2026-08-24\n")
    red = _drive(body, both, GITHUB_REF_NAME="v0.2.0")
    assert red.returncode != 0, red.stdout
    assert "names version 0.1.0" in red.stdout and "2026-08-24" in red.stdout, (
        "a heading wrong in BOTH the version and the date is reported with "
        "only one of them, so this step exits on the first complaint. It is "
        "one `exit 1` site and the header's arithmetic counts it as one "
        f"refusal precisely because it reports all of them:\n{red.stdout}"
    )
    assert "2 thing(s)" in red.stdout, (
        f"the annotation no longer counts the complaints it printed:\n"
        f"{red.stdout}"
    )

    # AND ONE UNREADABLE SIDE BESIDE ONE DISAGREEMENT, which is the shape that
    # exercises the fallback text in the `unreleased` message: it offers the
    # tag's date as the fix, and there ISN'T one on a lightweight tag. A bare
    # `${tag_date}` there would print an empty string and tell a human to put
    # nothing on the heading.
    lw = _tagged_tree(tmp_path, "red-lightweight-unreleased", annotated=False,
                      changelog="## 0.2.0 — unreleased\n")
    red = _drive(body, lw, GITHUB_REF_NAME="v0.2.0")
    assert red.returncode != 0, red.stdout
    assert "not an annotated tag" in red.stdout and "unreleased" in red.stdout, (
        f"an unreadable tag beside a disagreeing heading is reported as one "
        f"complaint, not two:\n{red.stdout}"
    )
    assert "no readable date" in red.stdout, (
        f"the `unreleased` refusal offers the tag's date as the remedy and "
        f"there is no date to offer, so it must say so rather than print an "
        f"empty string:\n{red.stdout}"
    )

    # THE DOCSTRING'S TABLE, HELD TO THE ROWS IT DESCRIBES. A table of drives
    # is prose about code and goes stale the moment a row is added — which is
    # the defect this module exists to catch one file over, so it is not left
    # to a proof-read here. Scoped to `rows`: the three drives above this line
    # are not in it and say so in the docstring.
    doc = test_the_changelog_gate_refuses_a_heading_the_tag_does_not_carry.__doc__
    described = _table_labels(doc)
    if doc:  # `python -OO` strips docstrings; there is then nothing to hold
        assert described == {label for label, *_ in rows}, (
            f"this test's table and its `rows` disagree.\n"
            f"  in `rows` and not in the table: "
            f"{sorted({label for label, *_ in rows} - described)}\n"
            f"  in the table and not in `rows`: "
            f"{sorted(described - {label for label, *_ in rows})}\n"
            f"This check used to run in one direction only — rows -> table — "
            f"so a row DELETED with its table line left standing was silent, "
            f"and that is the direction the failure actually took in the "
            f"sibling drive: a row was lost with a test that carried it and "
            f"the table went on advertising it. The four GREEN rows above the "
            f"rule carry no label and are described in prose; only the "
            f"labelled rows are held here."
        )


#: The instructions the lightweight refusal PRINTS, as literals. The drive
#: below executes each of them, so this tuple is what keeps the printed recipe
#: and the executed one from drifting apart — a remedy the suite plays out but
#: does not read would be a remedy that could be reworded into nonsense while
#: the drive stayed green, which is the shape of the defect this whole test
#: exists to close.
#:
#: IT IS A SPELLING PIN AND SAYS SO. It cannot tell a correct instruction from
#: an incorrect one; what says the instruction WORKS is the three-row drive.
_LIGHTWEIGHT_REMEDY = (
    "git tag -d",
    "git push origin :refs/tags/",
    "git tag -a",
    "onto the newest CHANGELOG.md heading",
    "TODAY the tag's date",
)


@_needs_a_shell
def test_the_lightweight_remedy_this_step_prints_is_the_one_that_works(tmp_path):
    """A REFUSAL MESSAGE IS AN INSTRUCTION, AND THIS ONE IS EXECUTED.

    THE DEFECT THIS CLOSES SHIPPED IN THE MESSAGE ITSELF. It read *"Cut the
    release with 'git tag -a' — a GitHub release created from the web UI
    writes an annotated tag — and re-run."* The clause after the dash was a
    present-tense claim about a third party's UI, nothing here could check it,
    and two dated reports say the release form creates a LIGHTWEIGHT tag — the
    exact shape this refusal exists to stop. So the message was pointing a
    maintainer whose release had just been refused back at the flow that
    produced the refusal. `release.yml` carries the evidence, what could not be
    determined, and why the refusal is kept anyway.

    THE REPAIR IS NOT A BETTER SENTENCE ABOUT GITHUB, IT IS A SENTENCE ABOUT
    `git` THAT THIS SUITE RUNS. The message now prints a recipe, and this test
    plays that recipe out on the body `release.yml` ships, one instruction at
    a time, asserting the verdict after each:

        the tree                                       verdict
        ─────────────────────────────────────────────  ──────────────────────
        LIGHTWEIGHT tag, heading dated D0              rc=1, "not an
                                                       annotated tag"
        re-cut ANNOTATED at the SAME commit, tagger    rc=1, and now on the
        date D1, heading still D0                      DATE arm: names D0
                                                       and D1
        heading moved to D1, tag re-cut at that        rc=0
        commit with tagger date D1

    THE MIDDLE ROW IS THE POINT AND A ONE-LINE REMEDY GETS IT WRONG.
    Re-cutting an annotated tag gives it TODAY's tagger date, so a heading
    written for the first attempt is then the wrong date and the step refuses
    a SECOND time. A message that says only "cut it with `git tag -a`" sends
    the maintainer round twice and looks, on the second round, like a gate
    that cannot be satisfied — which is how a gate gets silenced. The message
    predicts that round, and this row is what holds the prediction.

    D0 AND D1 ARE FIXED PAST DATES AND NOT `today`. A drive that re-cut "now"
    would compare against the clock of whoever runs the suite, which is the
    environment-dependence this whole gate was moved out of
    `tests/test_changelog_names_the_version.py` to avoid — arriving inside the
    test that checks it. `GIT_COMMITTER_DATE` is what plants them.

    WHAT THIS DOES NOT TOUCH is the other half of the message — *"push an
    annotated tag FIRST and create the release on the tag that already
    exists"*. That path is the one every green row in
    :func:`test_the_changelog_gate_refuses_a_heading_the_tag_does_not_carry`
    already drives, and it is what this repository's own two releases actually
    did (measured in `release.yml` beside the step: both tags annotated, both
    taggers the maintainer's git identity, both releases' `created_at` equal
    to the tag object's tagger timestamp with `published_at` later). It is
    cited rather than driven twice.
    """
    body = _step_body(_CHANGELOG_STEP)
    d0, d1 = "2026-08-25", "2026-08-28"
    tag = "v0.2.1"

    # THE WEB-FORM SHAPE, whatever made it: a ref pointing straight at a
    # commit, with no tag object and so no date of its own.
    tree = _tagged_tree(tmp_path, "remedy", tag=tag, annotated=False,
                        changelog=f"## 0.2.1 — {d0}\n\n- a release note\n")
    first = _drive(body, tree, GITHUB_REF_NAME=tag)
    assert first.returncode != 0, (
        "the changelog gate accepted a LIGHTWEIGHT tag. It has no tagger date "
        "at all, so the only date left is the commit's — the reading this "
        f"step declines, four paragraphs of `release.yml` over.\n{first.stdout}"
    )
    assert "not an annotated tag" in first.stdout, first.stdout

    # AND THE INSTRUCTIONS IT PRINTS ARE THE ONES EXECUTED BELOW.
    missing = [want for want in _LIGHTWEIGHT_REMEDY if want not in first.stdout]
    assert not missing, (
        f"the lightweight refusal no longer prints {missing}, and those are "
        f"the instructions this test then carries out. A recipe the suite "
        f"plays but does not read can be reworded into nonsense with every "
        f"drive still green.\n{first.stdout}"
    )
    # ...AND THE RETRACTED CLAIM IS NOT BACK. Asserted absent rather than the
    # repair pinned by its spelling: what was wrong was a message resting a
    # maintainer's next action on third-party behaviour nobody here can check.
    # This sees one wording of one such claim and no other — the general form
    # is not decidable by a text reader, and `release.yml` states the rule.
    assert "web UI" not in first.stdout, (
        "the refusal message is making a claim about a release UI again. The "
        "one it used to make — that the web form writes an ANNOTATED tag — is "
        "contradicted by two dated reports and would send a maintainer back "
        "to the flow that produced this refusal. Nothing here can check what "
        "any UI does; the remedy is about `git`, which this suite can run."
    )

    # STEP ONE OF THE RECIPE, EXACTLY AS PRINTED — ALL FOUR INSTRUCTIONS NOW,
    # INCLUDING THE TWO THAT TOUCH THE REMOTE. This comment read: *"(`git push
    # origin :refs/tags/...` is the remote half of the same instruction and
    # has no meaning in a repository with no remote; the local half is the one
    # that changes what this step reads, and it is the one driven.)"* Both
    # clauses are now false, and the second was the load-bearing one: since
    # the step re-fetches the tag from `origin` before reading it, a recipe
    # carried out LOCALLY is undone by the step itself on the next run — the
    # re-fetch pulls the old lightweight tag straight back over the re-cut
    # one. So the remote half is not decoration, it is the half that decides,
    # and a drive that skipped it was measuring a recipe no maintainer would
    # be following. `_tagged_tree` gives every tree an `origin` for this.
    at = _git(tree, "rev-parse", "HEAD").stdout.decode().strip()

    def _recut(date: str) -> None:
        """The printed recipe, executed: drop it here AND on the remote,
        re-cut it ANNOTATED at the same commit, push it."""
        _git(tree, "tag", "-d", tag)
        _git(tree, "push", "-q", "origin", f":refs/tags/{tag}")
        _annotate(tree, tag, date, at=at, message="stelling 0.2.1")
        _git(tree, "push", "-q", "origin", f"refs/tags/{tag}")

    _recut(d1)
    second = _drive(body, tree, GITHUB_REF_NAME=tag)
    assert second.returncode != 0, (
        "re-cutting the tag annotated was enough on its own, so the message's "
        "warning that this takes TWO edits is now false and should be "
        f"withdrawn with the reason.\n{second.stdout}"
    )
    assert "not an annotated tag" not in second.stdout, (
        "re-cutting the tag with `git tag -a` did not satisfy the arm the "
        f"message says it satisfies:\n{second.stdout}"
    )
    assert f"is dated '{d0}'" in second.stdout and f"was made on {d1}" in second.stdout, (
        "the second round is not the DATE arm naming both dates, so the "
        f"maintainer cannot see what the second edit has to be:\n"
        f"{second.stdout}"
    )

    # STEP TWO: the heading carries the date the NEW tag object got.
    (tree / "CHANGELOG.md").write_text(
        f"## 0.2.1 — {d1}\n\n- a release note\n", encoding="utf-8"
    )
    _recut(d1)
    third = _drive(body, tree, GITHUB_REF_NAME=tag)
    assert third.returncode == 0, (
        "the recipe this step PRINTS does not end in a release this step "
        "ACCEPTS. A refusal whose remedy does not work is a refusal a "
        f"maintainer routes around.\n{third.stdout}\n{third.stderr}"
    )
    assert f"tagged={d1}" in third.stdout and f"0.2.1 — {d1}" in third.stdout

#: The refusal text that DESTROYS A TAG, as literals. Asserted PRESENT on the
#: row where a lightweight tag really was pushed, and asserted ABSENT on every
#: row where the step could not find out or is refusing for another reason —
#: which is the whole of what this item repaired, and it is not a spelling pin
#: dressed up as one: what shipped in `v0.2.1` printed exactly these
#: instructions about a tag that was correctly annotated, and following them
#: would have failed again identically.
_DESTRUCTIVE_INSTRUCTIONS = ("git tag -d", "git push origin :refs/tags/",
                             "re-cut it ANNOTATED")

#: The three workspace shapes this module builds, named rather than spelled as
#: booleans because each one is an argument about what the release job IS.
_SHAPES = ("pinned-sha", "release", "depth-one")


def _checkout_the_way_actions_checkout_does(
    origin: pathlib.Path, dest: pathlib.Path, tag: str, commit: str,
    *, shape: str = "pinned-sha", unreachable: bool = False,
) -> pathlib.Path:
    """A workspace built the way `actions/checkout@v4` builds one, in one of
    three shapes — with the refspecs the action COMPUTES rather than the one a
    reader would write by hand.

    THE ACTION'S ALGORITHM, read off its source at `11d5960`. With
    `fetch-depth: 0` it fetches all history — `+refs/heads/*:refs/remotes/origin/*`
    and `+refs/tags/*:refs/tags/*` — and then re-fetches only if `testRef` is
    false. `testRef` is `if (!commit) return true`, `else if (!ref) return
    shaExists(commit)`, and only then the `REFS/TAGS/` branch
    `tagExists(t) && commit === revParse(ref)`. And `input-helper` turns an
    input `ref` matching `^[0-9a-fA-F]{40}$` into `commit = <sha>`,
    `ref = ''`. So WHICH BRANCH IS TAKEN IS A PROPERTY OF THE WORKFLOW:

        shape         what it models
        ────────────  ──────────────────────────────────────────────────────
        pinned-sha    `ref: ${{ github.sha }}` — `testRef` takes the `!ref`
                      branch, `shaExists` is true after the all-history
                      fetch, there is NO second fetch, and `refs/tags/<tag>`
                      is still the annotated tag object. This is what
                      `release.yml` does now
        release       no `ref:` — the action takes `github.ref` =
                      `refs/tags/<tag>`, `testRef` compares the TAG OBJECT's
                      sha against `github.sha`, finds them different, and
                      re-fetches `+<github.sha>:refs/tags/<tag>`, force-
                      writing the commit over the tag ref. This is what the
                      workflow did when it refused `v0.2.1`, and it is kept
                      because the step must still cope with a workspace built
                      that way
        depth-one     `--depth=1` with a pinned sha: the refspec is
                      `[commit]` and NO tag ref is fetched at all

    HEAD AND `HEAD^{tree}` ARE IDENTICAL IN `pinned-sha` AND `release`, which
    is asserted in the drive rather than trusted: the input changes which ref
    names what, and nothing about which tree is built.

    WHAT THIS DOES NOT MODEL. It is a re-derivation of the action's refspecs,
    not the action: no submodules, no LFS, no `persist-credentials` handling,
    no sparse checkout, and `origin` is a path on this disk rather than an
    authenticated https remote. `unreachable=True` points `origin` at a path
    that is not a repository, which is what a PRIVATE repository looks like to
    this step — the fetch is anonymous, because both checkouts say
    `persist-credentials: false`.
    """
    assert shape in _SHAPES, f"unknown workspace shape {shape!r}"
    dest.mkdir(parents=True)
    _git(dest, "-c", "init.defaultBranch=main", "init", "-q")
    _git(dest, "remote", "add", "origin", str(origin))
    if shape == "depth-one":
        _git(dest, "fetch", "-q", "--no-tags", "--prune", "--depth=1",
             "--no-recurse-submodules", "origin", commit)
        _git(dest, "checkout", "-q", "--force", commit)
    else:
        _git(dest, "fetch", "-q", "--prune", "--no-recurse-submodules", "origin",
             "+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*")
        if shape == "release":
            _git(dest, "fetch", "-q", "--no-tags", "--prune",
                 "--no-recurse-submodules", "origin",
                 f"+{commit}:refs/tags/{tag}")
            _git(dest, "checkout", "-q", "--force", f"refs/tags/{tag}")
        else:
            _git(dest, "checkout", "-q", "--force", commit)
    if unreachable:
        _git(dest, "remote", "set-url", "origin", str(dest / "no-such.git"))
    return dest


def _ref_object_type(tree: pathlib.Path, tag: str) -> str:
    """`%(objecttype)` for `refs/tags/<tag>` in `tree`, or `""` for no ref."""
    got = _git(tree, "for-each-ref", "--format=%(objecttype)",
               f"refs/tags/{tag}")
    return got.stdout.decode().strip()


def _tag_refs(tree: pathlib.Path) -> set[str]:
    """Every `refs/tags/*` name in `tree`. Used to catch a fetch that took
    more than the one ref it was asked for."""
    got = _git(tree, "for-each-ref", "--format=%(refname)", "refs/tags/")
    return {line for line in got.stdout.decode().split() if line}


@_needs_a_shell
def test_the_changelog_gate_reads_the_tag_THROUGH_the_checkout_that_rewrites_its_ref(tmp_path):
    """THE DEFECT THAT REFUSED `v0.2.1`, REPRODUCED, REPAIRED AT ITS CAUSE.

    `v0.2.1` was tagged ANNOTATED, pushed, and its release created on the tag
    that already existed — the exact procedure this step's own *"TO AVOID IT
    NEXT TIME"* sentence prescribes. The suite passed on the tagged tree and
    this step refused it, saying *"refs/tags/v0.2.1 is a commit object and not
    an annotated tag"* and instructing the maintainer to delete the tag, re-cut
    it, and move `CHANGELOG.md`'s date. Every clause was false and the
    instructions would have failed again identically, because what made the
    ref a commit was the CHECKOUT and not the tag.

    NOTHING IN THIS REPOSITORY COULD SEE THAT, and the reason is worth stating
    where the repair is: the two places in `release.yml` that had measured the
    behaviour — beside the `build` job's `fetch-depth: 0` and beside the step
    itself — both built a sandbox with ONE `git fetch` of a refspec they chose
    themselves, and `actions/checkout@v4` made TWO, of refspecs it computed.
    A correct measurement of the wrong process is this project's L28.

    AND THE FIX IS AN INPUT, NOT AN ALGORITHM, WHICH IS WHY THIS TEST IS
    SHORTER THAN IT WAS. Two earlier repairs reconstructed the tag object the
    checkout had un-named — first from `origin`, then from the workspace's own
    object store with three filters and a refuse-rather-than-pick rule. Both
    were sound only up to what they could infer, and the second was defeated
    twice by ordinary git (`git tag -a -f <t> <t>` nests; `git tag <new>
    <annotated>` does not dereference). `ref: ${{ github.sha }}` makes the
    action take its `!ref` branch, so there is no second fetch and the ref is
    never un-named: about forty-five lines of reconstruction deleted, and the
    two shapes that defeated it now come out right by reading the ref.

    THE ROWS::

        label                    the workspace / the input     this step
        ───────────────────────  ────────────────────────────  ─────────────
        `pinned-tag-intact`      `pinned-sha`, ANNOTATED       ref reads
                                                               `tag`; rc=0
        `pinned-offline`         the same, `origin`            rc=0 — the
                                 unreachable                   private-repo
                                                               case, free
        `clobbered-refetched`    `release` shape — the input   ref reads
                                 removed — origin reachable    `commit`; the
                                                               fetch repairs
                                                               it; rc=0
        `clobbered-offline`      the same, unreachable         rc=1, "DOES
                                                               NOT KNOW", no
                                                               recipe
        `lightweight`            LIGHTWEIGHT at origin — THE   rc=1, recipe
                                 NEGATIVE CONTROL              IS printed
        `nested-reannotation`    `git tag -a -f <t> <t>`,      rc=0 with the
                                 offline                       OUTER date
        `decoy-backup`           LIGHTWEIGHT + a `-backup`     rc=1
                                 ref on the old object
        `moved-rerun`            tag re-cut elsewhere before   rc=1, names
                                 the checkout                  both shas
        `moved-mid-job`          re-cut after it, origin       rc=1, same arm
                                 reachable
        `not-a-repo`             not a git repository at all   rc=1, naming
                                                               git's own
                                                               failure
        `no-head`                a repository with no commit   rc=1, naming
                                                               HEAD
        `taggerless`             a tag object with NO tagger   rc=1, and the
                                 line                          date check is
                                                               NOT skipped
        `versionless-tag`        a tag named exactly `v`       rc=1, and the
                                                               version half
                                                               does not skip
        `prefix-match`           `GITHUB_REF_NAME=v0.2` with   rc=1, refname
                                 `refs/tags/v0.2/1` present    mismatch
        `pattern-name`           GITHUB_REF_NAME is a PATTERN  rc=1, no other
                                                               tag ref
        `no-tags-guard`          `release` shape, so the       rc=0, no other
                                 fetch runs                    tag ref

    The labels are asserted against this docstring at the end, for the reason
    the sibling drive gives: a table of drives is prose about code and goes
    stale the moment a row is added.

    THE NEGATIVE CONTROL IS WHY THIS IS NOT JUST A GREEN. A repair that made
    the gate permissive would be worse than the false positive it fixes: a
    lightweight tag has no tagger date, so the date half of this gate cannot
    fire on one, and accepting it would mean the gate silently means something
    different for the two kinds of tag.

    `taggerless` AND `prefix-match` ARE REFUSALS THAT THIS FILE ALREADY
    SHIPPED AND NOTHING HELD. Neutralising each `problems+=` site one at a
    time, four of seventeen did not redden; these are the two reachable ones.
    A tag object with no tagger line is written by `git hash-object -t tag
    --literally`, reads `%(objecttype)=tag` with an empty `%(taggerdate:short)`,
    and passes a fetch — neither `fetch.fsckObjects` nor `transfer.fsckObjects`
    is on by default, measured, and it survives a clone. Remove the refusal
    that names it and `tag_date` stays empty, so the `[ -n "${tag_date}" ]`
    guard at the bottom SKIPS the date comparison entirely: the gate's headline
    check evaporates rather than refusing. And `for-each-ref` matches a
    glob-free pattern as a PATH PREFIX, so `refs/tags/v0.2` answers about
    `refs/tags/v0.2/1` — a name `git check-ref-format` accepts.

    THE FRESHNESS RESIDUAL, DECLARED. `refs/tags/<tag>` is origin as of the
    checkout's fetch, refreshed by this step's own fetch when `origin` can be
    reached. A tag moved between the two is therefore read as moved when the
    step can refresh and as the pinned object when it cannot, so `moved-mid-job`
    is driven only with a reachable origin. Both answers are about a tag that
    names a commit this job is not building, and the peel-equals-HEAD refusal
    is what makes either of them a refusal rather than a date.
    """
    body = _step_body(_CHANGELOG_STEP)
    tag, date = "v0.2.1", "2026-08-28"
    changelog = f"## 0.2.1 — {date}\n\n- a release note\n"
    driven: list[str] = []

    upstream = _tagged_tree(tmp_path, "upstream-annotated", tag=tag,
                            tagger_date=date, changelog=None, origin=False)
    commit = _git(upstream, "rev-parse", f"refs/tags/{tag}^{{commit}}"
                  ).stdout.decode().strip()
    # A SECOND ANNOTATED TAG OVER THE SAME COMMIT — a release candidate tagged
    # on the commit that becomes the release, which is an ordinary shape. It
    # is what makes `pattern-name` reach the "matched N refs" refusal: with
    # one tag ref in the workspace a pattern-shaped `GITHUB_REF_NAME` resolves
    # to a single row and is caught one branch earlier, by the refname
    # comparison, so the multiple-match refusal went unexercised.
    _annotate(upstream, f"{tag}-rc1", "2026-08-20", at=commit)

    def workspace(name, *, origin=upstream, shape="pinned-sha", at=commit,
                  unreachable=False, heading=changelog):
        ws = _checkout_the_way_actions_checkout_does(
            origin, tmp_path / name, tag, at, shape=shape,
            unreachable=unreachable)
        (ws / "CHANGELOG.md").write_text(heading, encoding="utf-8")
        return ws

    driven.append("pinned-tag-intact")
    # THE SHAPE THE WORKFLOW NOW BUILDS. The precondition is asserted BEFORE
    # the drive: a green here with the ref reading `commit` would be a green
    # for a different reason than the one claimed.
    pinned = workspace("ws-pinned")
    assert _ref_object_type(pinned, tag) == "tag", (
        "with `ref: ${{ github.sha }}` the action takes its `!ref` branch and "
        "makes no second fetch, so `refs/tags/<tag>` must still be the "
        "annotated tag object. It is not, so either the action's algorithm "
        "has been mis-derived here or this git no longer behaves that way."
    )
    ok = _drive(body, pinned, GITHUB_REF_NAME=tag)
    assert ok.returncode == 0, (
        "THE GATE REFUSES A CORRECTLY ANNOTATED TAG in the workspace shape "
        f"`release.yml` now builds.\n{ok.stdout}\n{ok.stderr}"
    )
    assert f"tagged={date}" in ok.stdout, ok.stdout
    for banned in _DESTRUCTIVE_INSTRUCTIONS:
        assert banned not in ok.stdout, (
            f"the step tells a maintainer to {banned!r} on a release it is "
            f"ACCEPTING:\n{ok.stdout}"
        )

    driven.append("pinned-offline")
    # THE PRIVATE-REPOSITORY CASE, AND IT IS FREE NOW. `persist-credentials:
    # false` makes this step's fetch anonymous, so on a private repository it
    # can never succeed — and it does not need to, because the ref the first
    # fetch brought was never overwritten. Two earlier repairs spent a network
    # round trip and then forty-five lines of inference on exactly this.
    offline = workspace("ws-pinned-offline", unreachable=True)
    free = _drive(body, offline, GITHUB_REF_NAME=tag)
    assert free.returncode == 0, (
        "the step refused a workspace whose tag ref is a perfectly good "
        "annotated tag object, because it could not reach `origin`. Nothing "
        f"had rewritten that ref; there was nothing to repair.\n{free.stdout}"
    )
    assert f"tagged={date}" in free.stdout, free.stdout

    driven.append("clobbered-refetched")
    # THE INPUT REMOVED — DEFENCE IN DEPTH. This step cannot verify what its
    # own job's checkout was configured with, so it keeps a fetch that repairs
    # the ref when something has overwritten it.
    clob = workspace("ws-clobbered", shape="release")
    assert _ref_object_type(clob, tag) == "commit", clob
    assert (_git(clob, "rev-parse", "HEAD").stdout
            == _git(pinned, "rev-parse", "HEAD").stdout), (
        "the two checkout shapes do not produce the same HEAD, so the claim "
        "that `ref: ${{ github.sha }}` changes which ref names what and "
        "nothing about which tree is built is false."
    )
    repaired = _drive(body, clob, GITHUB_REF_NAME=tag)
    assert repaired.returncode == 0, (
        "a workspace whose tag ref was overwritten by the checkout was not "
        f"repaired by this step's own fetch.\n{repaired.stdout}\n{repaired.stderr}"
    )
    assert "re-fetched" in repaired.stdout and f"tagged={date}" in repaired.stdout

    driven.append("clobbered-offline")
    # AND WHEN NEITHER THE INPUT NOR THE FETCH IS THERE, IT SAYS SO.
    blind = workspace("ws-clobbered-offline", shape="release", unreachable=True)
    cant = _drive(body, blind, GITHUB_REF_NAME=tag)
    assert cant.returncode != 0, (
        "the step passed on a workspace whose tag ref is a commit and which "
        f"it could not check against origin.\n{cant.stdout}"
    )
    assert "DOES NOT KNOW" in cant.stdout, (
        "the step refused without saying that it cannot tell which of the two "
        f"causes it is looking at.\n{cant.stdout}"
    )
    # THE CHECK IT HANDS OVER IS THE SHELL-SAFE SPELLING. `git ls-remote
    # origin refs/tags/<tag>*` is shorter and is a TRAP: unquoted, a shell
    # standing in a directory that holds a path of that name expands the `*`
    # away, and the command then prints ONE line for an ANNOTATED tag — the
    # reading that says "lightweight", which is the wrong answer in the exact
    # direction this whole repair is about.
    assert "git ls-remote --tags origin" in cant.stdout, cant.stdout
    for banned in _DESTRUCTIVE_INSTRUCTIONS:
        assert banned not in cant.stdout, (
            f"THE STEP IS TELLING A MAINTAINER TO {banned!r} ON EVIDENCE IT "
            f"DOES NOT HAVE.\n{cant.stdout}"
        )

    driven.append("lightweight")
    light = _tagged_tree(tmp_path, "upstream-lightweight", tag=tag,
                         annotated=False, changelog=None, origin=False)
    light_at = _git(light, "rev-parse", f"refs/tags/{tag}"
                    ).stdout.decode().strip()
    lw = workspace("ws-lightweight", origin=light, at=light_at)
    red = _drive(body, lw, GITHUB_REF_NAME=tag)
    assert red.returncode != 0, (
        "THE GATE ACCEPTED A GENUINELY LIGHTWEIGHT TAG. A lightweight tag has "
        "no tagger date, so the date half of this gate cannot fire on one, "
        "and a permissive answer here is a worse outcome than the false "
        f"positive it replaced.\n{red.stdout}"
    )
    assert "not an annotated tag" in red.stdout, red.stdout
    missing = [w for w in _DESTRUCTIVE_INSTRUCTIONS if w not in red.stdout]
    assert not missing, (
        f"the lightweight refusal no longer prints {missing}.\n{red.stdout}"
    )

    driven.append("nested-reannotation")
    # `git tag -a -f <t> <t>` NESTS — git says so itself — so the superseded
    # object stays reachable from the live ref and arrives in every checkout.
    # A resolver that hunted the object store for "a tag object named this,
    # over this commit" picked the SUPERSEDED one and read its date; reading
    # the REF cannot, because the ref names the outer object.
    inner, outer = "2026-08-20", "2026-08-28"
    nest_up = _tagged_tree(tmp_path, "up-nested", tag=tag, tagger_date=inner,
                           changelog=None, origin=False)
    nest_at = _git(nest_up, "rev-parse", f"refs/tags/{tag}^{{commit}}"
                   ).stdout.decode().strip()
    superseded = _git(nest_up, "rev-parse", tag).stdout.decode().strip()
    _annotate(nest_up, tag, outer, at=tag, force=True)
    nested = workspace("ws-nested", origin=nest_up, at=nest_at,
                       unreachable=True,
                       heading=f"## 0.2.1 — {outer}\n")
    assert _git(nested, "cat-file", "-t", superseded
                ).stdout.decode().strip() == "tag", (
        "the superseded object did not arrive, so this row is not the shape "
        "it says it is."
    )
    deep = _drive(body, nested, GITHUB_REF_NAME=tag)
    assert deep.returncode == 0, f"{deep.stdout}\n{deep.stderr}"
    assert f"tagged={outer}" in deep.stdout and f"tagged={inner}" not in deep.stdout, (
        "the step read the SUPERSEDED tagger date rather than the one the ref "
        f"actually names.\n{deep.stdout}"
    )

    driven.append("decoy-backup")
    # `git tag <new> <annotated>` does NOT dereference, so backing the tag up
    # before following this step's own delete-and-re-cut recipe leaves an
    # object carrying the header `tag v0.2.1` behind. It is not what the ref
    # names, and reading the ref cannot be fooled by it.
    old = "2019-01-01"
    decoy_up = _tagged_tree(tmp_path, "up-decoy", tag=tag, tagger_date=old,
                            changelog=None, origin=False)
    decoy_at = _git(decoy_up, "rev-parse", f"refs/tags/{tag}^{{commit}}"
                    ).stdout.decode().strip()
    _git(decoy_up, "tag", f"{tag}-backup", tag)
    assert _ref_object_type(decoy_up, f"{tag}-backup") == "tag", (
        "`git tag <new> <annotated>` dereferenced, so this row is not the "
        "shape it says it is."
    )
    _git(decoy_up, "tag", "-d", tag)
    _git(decoy_up, "tag", tag, decoy_at)
    decoy = workspace("ws-decoy", origin=decoy_up, at=decoy_at,
                      unreachable=True, heading=f"## 0.2.1 — {old}\n")
    fooled = _drive(body, decoy, GITHUB_REF_NAME=tag)
    assert fooled.returncode != 0, (
        "THE STEP READ A DECOY AND ACCEPTED A LIGHTWEIGHT TAG.\n{}".format(
            fooled.stdout)
    )
    assert f"tagged={old}" not in fooled.stdout, fooled.stdout

    driven.append("moved-rerun")
    driven.append("moved-mid-job")
    for label, recut_first in (("moved-rerun", True), ("moved-mid-job", False)):
        up = _tagged_tree(tmp_path, f"up-{label}", tag=tag, tagger_date=date,
                          changelog=None, origin=False)
        pinned_at = _git(up, "rev-parse", f"refs/tags/{tag}^{{commit}}"
                         ).stdout.decode().strip()

        def recut():
            _git(up, *_GIT_IDENT, "commit", "-q", "--no-verify",
                 "--allow-empty", "-m", "a commit the release event never saw")
            other = _git(up, "rev-parse", "HEAD").stdout.decode().strip()
            _git(up, "tag", "-d", tag)
            _annotate(up, tag, "2026-08-29", at=other)
            return other

        if recut_first:
            elsewhere = recut()
            moved = workspace(f"ws-{label}", origin=up, at=pinned_at)
        else:
            moved = workspace(f"ws-{label}", origin=up, at=pinned_at)
            elsewhere = recut()
        drift = _drive(body, moved, GITHUB_REF_NAME=tag)
        assert drift.returncode != 0, (
            f"[{label}] THE GATE VOUCHED FOR A TAG THAT DOES NOT NAME THE "
            f"TREE BEING BUILT. HEAD is {pinned_at} and the tag peels to "
            f"{elsewhere}.\n{drift.stdout}"
        )
        assert pinned_at in drift.stdout and elsewhere in drift.stdout, (
            f"[{label}] the correspondence refusal does not name BOTH "
            f"commits.\n{drift.stdout}"
        )
        for banned in _DESTRUCTIVE_INSTRUCTIONS:
            assert banned not in drift.stdout, (
                f"[{label}] the step tells a maintainer to {banned!r} because "
                f"a tag MOVED.\n{drift.stdout}"
            )

    driven.append("taggerless")
    # A TAG OBJECT WITH NO TAGGER LINE. `git hash-object -t tag --literally`
    # writes one, `%(objecttype)` reads `tag`, `%(taggerdate:short)` is empty,
    # and it survives a fetch and a clone because neither `fetch.fsckObjects`
    # nor `transfer.fsckObjects` is on by default (measured). The refusal that
    # names it is load-bearing in a way a reader would not guess: without it
    # `tag_date` stays empty, and the `[ -n "${tag_date}" ]` guard at the
    # bottom of the step SKIPS the date comparison — so the gate's headline
    # check would evaporate rather than refuse. The heading here disagrees
    # with everything, which is what makes that visible.
    bare_up = _tagged_tree(tmp_path, "up-taggerless", tag=tag, tagged=False,
                           changelog=None, origin=False)
    bare_at = _git(bare_up, "rev-parse", "HEAD").stdout.decode().strip()
    raw = tmp_path / "taggerless.raw"
    raw.write_text(f"object {bare_at}\ntype commit\ntag {tag}\n\n"
                   "no tagger line at all\n", encoding="utf-8")
    obj = _git(bare_up, "hash-object", "-t", "tag", "-w", "--literally",
               str(raw)).stdout.decode().strip()
    _git(bare_up, "update-ref", f"refs/tags/{tag}", obj)
    bare = workspace("ws-taggerless", origin=bare_up, at=bare_at,
                     heading="## 0.9.9 — 1999-01-01\n")
    assert _ref_object_type(bare, tag) == "tag", bare
    none_dated = _drive(body, bare, GITHUB_REF_NAME=tag)
    assert none_dated.returncode != 0, (
        "a tag object with an EMPTY tagger date was accepted, and with it a "
        "heading naming another version on another date. The empty-date "
        "refusal is what stops `tag_date` staying empty and the date "
        f"comparison being skipped altogether.\n{none_dated.stdout}"
    )
    assert "tagger date is empty" in none_dated.stdout, none_dated.stdout

    driven.append("not-a-repo")
    # NOT A GIT REPOSITORY AT ALL. `git for-each-ref` returns rc=0 for every
    # pattern this step can hand it — measured on `refs/tags/[`,
    # `refs/tags/\\`, `refs/tags/**/*` and `refs/tags/*`, all rc=0 — so the
    # ONLY way its exit code is non-zero is that there is no repository to
    # ask, and that is the row. The step's own "WHAT THIS STEP CANNOT SEE"
    # list already says the path is relative to whatever directory the step
    # runs in; this is what that assumption failing looks like.
    outside = tmp_path / "ws-not-a-repo"
    outside.mkdir()
    (outside / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    nowhere = _drive(body, outside, GITHUB_REF_NAME=tag)
    assert nowhere.returncode != 0, nowhere.stdout
    assert "git for-each-ref exited" in nowhere.stdout, (
        "a working directory that is not a git repository did not produce the "
        "refusal that names git's own failure. Without it the step reports "
        "'there is no refs/tags/<tag> in this checkout', which is a claim "
        f"about a checkout that is not there at all.\n{nowhere.stdout}"
    )

    driven.append("no-head")
    # NO COMMIT AT ALL, so there is no build subject to tie a tag to. This row
    # came back from a test this commit deleted, and its absence was found by
    # driving rather than by reading: with the unreadable-HEAD refusal removed
    # the module stayed GREEN.
    headless = tmp_path / "ws-headless"
    headless.mkdir()
    _git(headless, "-c", "init.defaultBranch=main", "init", "-q")
    (headless / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    none = _drive(body, headless, GITHUB_REF_NAME=tag)
    assert none.returncode != 0, none.stdout
    assert "cannot tell which commit this job is building" in none.stdout, (
        "a workspace with no HEAD did not produce the refusal that names it. "
        "Everything this step compares is a comparison against the commit "
        "being built, so a step that cannot read one must say so rather than "
        f"compare against an empty string.\n{none.stdout}"
    )

    driven.append("versionless-tag")
    # A TAG NAMED EXACTLY `v`. `${tag#v}` strips to the empty string, and an
    # empty `tag_version` makes the version comparison at the bottom of the
    # step SKIP — so the heading here names 9.9.9 and carries a date that
    # MATCHES, leaving the version half as the only thing that could refuse.
    # Pre-existing; found by neutralising each refusal in turn.
    v_up = _tagged_tree(tmp_path, "up-versionless", tag="v",
                        tagger_date=date, changelog=None, origin=False)
    v_at = _git(v_up, "rev-parse", "refs/tags/v^{commit}").stdout.decode().strip()
    v_ws = _checkout_the_way_actions_checkout_does(
        v_up, tmp_path / "ws-versionless", "v", v_at)
    (v_ws / "CHANGELOG.md").write_text(f"## 9.9.9 — {date}\n", encoding="utf-8")
    versionless = _drive(body, v_ws, GITHUB_REF_NAME="v")
    assert versionless.returncode != 0, (
        "a tag named `v` strips to an empty version, and the version half of "
        "this gate silently skipped: the heading names 9.9.9 and the step "
        f"accepted it.\n{versionless.stdout}"
    )
    assert "leaves no version at all" in versionless.stdout, versionless.stdout

    driven.append("prefix-match")
    # `for-each-ref` MATCHES A GLOB-FREE PATTERN AS A PATH PREFIX, so
    # `refs/tags/v0.2` answers about `refs/tags/v0.2/1` — and `v0.2/1` is a
    # name `git check-ref-format` accepts, so nothing upstream refuses it.
    # The step compares the refname it got against the one it asked for.
    slash_up = _tagged_tree(tmp_path, "up-prefix", tag="v0.2/1",
                            tagger_date=date, changelog=None, origin=False)
    slash_at = _git(slash_up, "rev-parse", "refs/tags/v0.2/1^{commit}"
                    ).stdout.decode().strip()
    slash = _checkout_the_way_actions_checkout_does(
        slash_up, tmp_path / "ws-prefix", "v0.2/1", slash_at)
    (slash / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    prefix = _drive(body, slash, GITHUB_REF_NAME="v0.2")
    assert prefix.returncode != 0, (
        "`git for-each-ref refs/tags/v0.2` answered about `refs/tags/v0.2/1` "
        "and the step took it for the tag it asked about. The name comes from "
        f"the environment.\n{prefix.stdout}"
    )
    assert "and it answered about" in prefix.stdout, prefix.stdout

    driven.append("pattern-name")
    # BOTH WORKSPACES ARE BUILT BEFORE THE PLANT, which is the whole design of
    # these two rows: the extra tag must not be reachable through the
    # checkout's own fetch, or its presence afterwards says nothing about what
    # the STEP fetched. Driven the wrong way round first — the second
    # workspace built after the plant came back holding it, from its own
    # fetch 1, with the step's fetch entirely innocent.
    pattern_ws = workspace("ws-pattern")
    narrow_ws = workspace("ws-no-tags", shape="release")
    _git(upstream, "tag", "v9.9.9-planted", commit)
    # THE POSITIVE CONTROL FOR AN ABSENCE ASSERTION: with the plant deleted
    # this row and the next were still green, so their own liveness rested on
    # a line nothing checked.
    assert "refs/tags/v9.9.9-planted" in _tag_refs(upstream), (
        "the extra tag was not planted at origin, so the two absence "
        "assertions below cannot fail and are measuring nothing."
    )
    before = _tag_refs(pattern_ws)
    wide = _drive(body, pattern_ws, GITHUB_REF_NAME="*")
    assert wide.returncode != 0, wide.stdout
    after = _tag_refs(pattern_ws)
    assert "matched 2 refs" in wide.stdout, (
        "the step refused a pattern-shaped `GITHUB_REF_NAME` for some other "
        "reason than the one that matters. `git for-each-ref refs/tags/*` "
        "answers about EVERY tag, and a step that read the first row would be "
        "reporting a date for whichever tag `for-each-ref` happened to list "
        f"first.\n{wide.stdout}"
    )
    assert "refs/tags/v9.9.9-planted" not in after, (
        "a `GITHUB_REF_NAME` that git reads as a PATTERN made this step fetch "
        "`refs/tags/*:refs/tags/*` and drag in every tag on the remote. "
        "`git check-ref-format` is what refuses it BEFORE anything is "
        "written, and a text needle cannot see that guard go away because the "
        f"literal survives in the step's own comments.\n{sorted(after - before)}"
    )

    driven.append("no-tags-guard")
    # `--no-tags`, DRIVEN, on a shape where the fetch actually runs.
    narrow = _drive(body, narrow_ws, GITHUB_REF_NAME=tag)
    assert narrow.returncode == 0, f"{narrow.stdout}\n{narrow.stderr}"
    assert "re-fetched" in narrow.stdout, (
        "this row is supposed to exercise the fetch — if something else "
        f"decided it, the assertion below measures a fetch that never ran.\n"
        f"{narrow.stdout}"
    )
    assert "refs/tags/v9.9.9-planted" not in _tag_refs(narrow_ws), (
        "the fetch lost `--no-tags` and auto-followed the remote's other "
        "tags, so this step wrote ref names it was never asked about.\n"
        f"{narrow.stdout}"
    )

    # THE TABLE, HELD TO THE ROWS IT DESCRIBES. Under `python -OO` there is no
    # docstring at all, so the check has nothing to hold and says so instead
    # of reporting every row missing.
    doc = test_the_changelog_gate_reads_the_tag_THROUGH_the_checkout_that_rewrites_its_ref.__doc__
    described = _table_labels(doc)
    if doc:  # `python -OO` strips docstrings; there is then nothing to hold
        assert described == set(driven), (
            f"this test's table and its drives disagree.\n"
            f"  driven and not in the table: {sorted(set(driven) - described)}\n"
            f"  in the table and not driven: {sorted(described - set(driven))}\n"
            f"A row added without a table line leaves the docstring behind; a "
            f"row DELETED with its table line left standing leaves the table "
            f"advertising a drive that no longer runs — and some of these rows "
            f"are the only pin on a refusal, so that direction loses coverage "
            f"silently. Both are held now."
        )


#: SHAPES BOTH READERS MUST AGREE ABOUT, and the reason this table exists is
#: that they did not. `.github/workflows/release.yml` decides "the newest
#: heading" in bash and `tests/test_changelog_names_the_version.py` decides it
#: in Python, and until an auditor drove them side by side the two were
#: `^##[[:space:]]` and `^##\s` over a whole document — which agreed on the
#: shapes anybody had thought to try and disagreed on five, three of them in
#: the direction that returns rc=0 over a real tag/heading disagreement.
#:
#: Each row is `(label, changelog text, expected reading)`, where the reading
#: is the `(version, rest)` pair or `None` for "refuses". The two readers are
#: run over the SAME rows, so the table cannot describe one of them.
_HEADING_SHAPES = (
    ("plain",
     "## 0.2.0 — 2026-08-25\n", ("0.2.0", "2026-08-25")),
    ("older below the newest",
     "## 0.2.0 — 2026-08-25\n\nbody\n\n## 0.1.0 — 2026-08-12\n",
     ("0.2.0", "2026-08-25")),
    # A CommonMark <h2> may carry one to three leading spaces. Both readers
    # missed this, and a NEWER heading written this way was stepped past.
    ("indented one space",
     "  ## 0.2.0 — 2026-08-25\n\n## 0.1.0 — 2026-08-12\n",
     ("0.2.0", "2026-08-25")),
    ("indented three spaces",
     "   ## 0.2.0 — 2026-08-25\n\n## 0.1.0 — 2026-08-12\n",
     ("0.2.0", "2026-08-25")),
    # Four is an indented code block, not a heading -- so the real heading
    # below it is the newest, and that is a reading rather than a miss.
    ("indented four spaces is a code block",
     "    ## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n",
     ("0.2.0", "2026-08-25")),
    # `CHANGELOG.md` opens with an HTML comment, so this route is always one
    # edit away.
    ("inside an HTML comment",
     "<!--\n## 9.9.9 — 2000-01-01\n-->\n\n## 0.2.0 — 2026-08-25\n",
     ("0.2.0", "2026-08-25")),
    ("inside a fenced block",
     "```\n## 9.9.9 — 2000-01-01\n```\n\n## 0.2.0 — 2026-08-25\n",
     ("0.2.0", "2026-08-25")),
    # `##` alone IS a heading line and does not parse: refusing beats reading
    # past to an older one. In Python `\s` used to match the newline and fuse
    # the two lines into one heading.
    ("bare ## above a real heading",
     "##\n0.2.0 — 2026-08-25\n\n## 0.1.0 — 2026-08-12\n", None),
    ("no em dash", "## 0.2.0 - 2026-08-25\n", None),
    ("no heading at all", "nothing here\n", None),
)


@_needs_a_shell
@pytest.mark.parametrize(
    "label,changelog,expected", _HEADING_SHAPES,
    ids=[row[0].replace(" ", "-") for row in _HEADING_SHAPES],
)
def test_BOTH_readers_of_the_newest_heading_agree(label, changelog, expected, tmp_path):
    """The bash gate and the Python gate, over one table of shapes.

    **TWO READERS OF ONE RULE, WRITTEN IN TWO LANGUAGES, HELD TO EACH OTHER BY
    NOTHING** — until an auditor ran them side by side and found five shapes
    they disagree about. Three of those five are the direction that matters:
    the bash gate returned `rc=0` on a tree whose newest heading names a
    DIFFERENT version from the tag, because the heading was indented one
    space and neither reader called it a heading.

    So the rule is one table now and both readers are driven over it. The
    Python side is compared by its return value; the bash side by whether it
    refuses, and — where it does not — by the version and date it read, which
    are recovered from its own complaint when it is made to disagree with a
    tag it should agree with.

    What this does NOT hold: that either reader implements CommonMark. Both
    are line grammars, both say so, and shapes neither handles (setext
    headings, an unbalanced fence) are outside the table rather than pinned
    to a behaviour.
    """
    from test_changelog_names_the_version import (
        MalformedNewestHeading, newest_heading,
    )

    # THE PYTHON READER.
    try:
        got = newest_heading(changelog)
    except MalformedNewestHeading:
        got = None
    assert got == expected, f"{label}: python read {got!r}, table says {expected!r}"

    # THE BASH GATE, over the same text. It is driven against a real annotated
    # tag whose version and date are the table's expected reading, so it must
    # be GREEN exactly where the table says a reading exists.
    body = _step_body(_CHANGELOG_STEP)
    if expected is None:
        tree = _tagged_tree(tmp_path, f"{label}-none", tag="v0.2.0",
                            tagger_date="2026-08-25", changelog=changelog)
        done = _drive(body, tree, GITHUB_REF_NAME="v0.2.0")
        assert done.returncode == 1, (
            f"{label}: the python reader refuses this and the bash gate "
            f"accepted it:\n{done.stdout}"
        )
        return

    version, rest = expected
    tree = _tagged_tree(tmp_path, f"{label}-ok", tag=f"v{version}",
                        tagger_date=rest, changelog=changelog)
    done = _drive(body, tree, GITHUB_REF_NAME=f"v{version}")
    assert done.returncode == 0, (
        f"{label}: python read {expected!r} and the bash gate refused a tag "
        f"carrying exactly that:\n{done.stdout}"
    )

    # ... and it must REFUSE the same text against a tag that disagrees, or
    # the green above is a gate that accepts everything rather than one that
    # read the same heading.
    other = _tagged_tree(tmp_path, f"{label}-bad", tag="v9.9.9",
                         tagger_date="2000-01-01", changelog=changelog)
    done = _drive(body, other, GITHUB_REF_NAME="v9.9.9")
    assert done.returncode == 1, (
        f"{label}: the bash gate accepted a tag naming neither the version "
        f"nor the date the python reader found:\n{done.stdout}"
    )


@_needs_a_shell
def test_the_changelog_gate_reads_the_NEWEST_heading_by_POSITION(tmp_path):
    """Newest is a POSITION in the document, and this is what says so.

    A gate that scanned `CHANGELOG.md` for *a* heading naming the tag's
    version would be green on a file that still leads with the PREVIOUS
    release — the exact shape a forgotten changelog bump leaves, and the shape
    the routed recommendation named. So the file here holds BOTH headings,
    newest-first, and the verdict has to depend on which one is first:

        `## 0.2.0 — …` then `## 0.1.0 — …`, tag v0.2.0   GREEN
        the same file,                       tag v0.1.0  RED, and the refusal
                                                         names 0.2.0 as what
                                                         it read

    The second row is the one a whole-file scan cannot fail: `## 0.1.0 —
    2026-08-12` is in that file, verbatim, and the tag is `v0.1.0`.

    `headings()` in `tests/test_changelog_names_the_version.py` takes the same
    definition for the same reason, and says so in the same words. The two
    readers are written twice in two languages with nothing holding them to
    each other — declared here and beside the step, rather than left for a
    release to discover.
    """
    body = _step_body(_CHANGELOG_STEP)
    both = "## 0.2.0 — 2026-08-25\n\n- newer\n\n## 0.1.0 — 2026-08-12\n\n- older\n"

    newest = _tagged_tree(tmp_path, "position-newest", tag="v0.2.0",
                          tagger_date="2026-08-25", changelog=both)
    ok = _drive(body, newest, GITHUB_REF_NAME="v0.2.0")
    assert ok.returncode == 0, (
        f"the gate refuses a changelog whose NEWEST heading is this "
        f"release's, with an older release below it — which is every "
        f"changelog this project has ever had.\n{ok.stdout}\n{ok.stderr}"
    )

    older = _tagged_tree(tmp_path, "position-older", tag="v0.1.0",
                         tagger_date="2026-08-12", changelog=both)
    red = _drive(body, older, GITHUB_REF_NAME="v0.1.0")
    assert red.returncode != 0, (
        "THE CHANGELOG GATE PASSED a tree tagged v0.1.0 whose changelog leads "
        "with `## 0.2.0`. It found `## 0.1.0 — 2026-08-12` somewhere in the "
        "file and was satisfied, which is a scan and not a reading of the "
        f"NEWEST heading.\n{red.stdout}"
    )
    assert "names version 0.2.0" in red.stdout, (
        f"the refusal does not name the heading it actually read:\n"
        f"{red.stdout}"
    )


@_needs_a_shell
def test_the_drives_are_reading_the_real_step_bodies():
    """Anti-vacuity for the four drives, in the shape this file already uses.

    THIS SAID "the three drives" AND WAS RIGHT UNTIL THE CHANGELOG STEP
    LANDED. A count of the drives, written in the docstring of the test whose
    job is to notice when a drive stops driving anything, is the same defect
    this module holds `release.yml` to; it is corrected rather than deleted so
    the next reader can see it happened here too.

    The drives assert on exit codes of a script this file did not write. Three
    ways that goes quietly wrong — a step name that no longer resolves, a
    `run:` block read as empty, an `env:` that stopped carrying `GENERATED` —
    and the first two are fatal in `_step_lines`/`_step_body` already. This
    pins what the extracted text must CONTAIN, so that a body reduced to its
    `set -euo pipefail` cannot satisfy the drives above by exiting 0.

    AND TWO CONSTRUCTS ARE PINNED AS ABSENT rather than as present, which is
    the shape this file already uses for `ls dist/`: a construct that is the
    defect is asserted GONE, because pinning the repair by its spelling pins
    the spelling. The second is an `exit 1` in the manifest step — `release.yml`
    calls that step a non-refusal and gives it no count, so an `exit 1` there
    is a refusal point the header's arithmetic never saw.
    """
    # CODE LINES, FOR ALL FOUR BODIES, AND THE REASON IS A DEFECT THIS
    # FUNCTION HAD IN THREE OF ITS FOUR LOOPS. A needle matched against an
    # extracted `run:` block INCLUDING ITS COMMENTS is satisfied by prose, and
    # `release.yml` is a file whose step comments quote the constructs beside
    # them — usually because a retired one is being recorded, which is exactly
    # what this module asks that file to do. Measured on the bodies this
    # commit ships: `tar tzf` is 1 code line and 2 comment lines in the sdist
    # step, `mktemp -d` the same, and replacing the sole code occurrence of
    # `mktemp -d` with a behaviour-identical spelling left this module GREEN
    # with three comment hits standing. The changelog loops were repaired one
    # commit earlier and the other three were left; they are all one
    # projection now, so a body cannot be pinned by its own footnotes.
    def _code_of(step_name: str) -> str:
        return "\n".join(
            line for line in _step_body(step_name).splitlines()
            if not line.lstrip().startswith("#")
        )

    sdist = _code_of(_SDIST_STEP)
    tag = _code_of(_TAG_STEP)
    manifest = _code_of(_MANIFEST_STEP)
    changelog = _code_of(_CHANGELOG_STEP)
    for needle in ("tar tzf", "git ls-files", "comm -23", "explained.txt",
                   "dist/*.tar.gz", "${#sdists[@]}",
                   "shopt -s nullglob dotglob", "mktemp -d", "trap "):
        assert needle in sdist, f"{needle!r} is gone from the sdist step body"
    # THE TARBALL READ IS CAPTURED, AND ITS STDERR IS NOT PART OF THE CAPTURE.
    # Both halves matter and each is driven: without the capture the step dies
    # on `tar`'s rc=2 with no annotation
    # (`test_the_sdist_gate_refuses_a_tarball_it_cannot_read`), and with
    # `2>&1` on it a warning becomes a member and a healthy release is refused
    # (`test_the_sdist_gate_does_not_read_tars_warnings_as_members`).
    assert 'listing="$(tar tzf "${sdist}" 2>"${work}/tar.err")"' in sdist, (
        "the sdist step's `tar tzf` is no longer captured with its stderr "
        "kept OUT of the capture. `2>&1` here `sed`s and `sort`s a diagnostic "
        "into `members.txt` as though it were a member; no redirection at all "
        "puts the step back to dying on `tar`'s own exit code with nothing "
        "said."
    )
    assert "cannot read the tarball" in sdist, (
        "the sdist step no longer annotates a tarball it cannot read. rc=2 is "
        "`tar`'s exit code and this step's refusals are rc=1 — red, but not "
        "by this gate, which is the state the no-wheel and two-tarball cases "
        "in this same file were repaired out of."
    )
    # `ls dist/*.whl` USED TO BE ON THIS LIST and is deliberately not: reading
    # the LAST line of `ls` is the defect that step was repaired for, so a pin
    # demanding it back would pin the defect. What replaces it is the glob
    # itself (the step must still be looking in `dist/`), the array count that
    # makes "the wheel" a well-defined noun, and the `LC_ALL` the file argues
    # for — each of which a rewrite back to `ls` would drop. The tarball
    # spellings below are the same decision made a second time.
    for needle in ("basename", "cut -d- -f2", "dist/*.whl", "dist/*.tar.gz",
                   "export LC_ALL=C", "shopt -s nullglob dotglob",
                   "${#wheels[@]}", "${#sdists[@]}", ".tar.gz | cut -d- -f2",
                   # THE DIRECTORY, which the two globs above are not. A
                   # rewrite that drops the inventory loop leaves both counts
                   # standing and re-opens the `.zip` shape driven in
                   # `test_the_tag_gate_refuses_a_dist_entry_it_did_not_build`.
                   "for entry in dist/*", "${#extras[@]}"):
        assert needle in tag, f"{needle!r} is gone from the tag step body"
    # `dotglob` IS LOAD-BEARING AND NOT TIDINESS: without it the inventory loop
    # cannot see a `dist/.stelling-0.0.9.zip` at all, and the one exception
    # this step makes is a NAME rather than a class, so the loop has to be able
    # to reach every dotfile in order to refuse the ones that are not it.
    for needle in ("dist/.gitignore)", '[ ! -L "${entry}" ]',
                   '[ -f "${entry}" ]', '[ "$(wc -c < "${entry}")" -eq 1 ]',
                   '[ "$(cat "${entry}")" = "*" ]'):
        assert needle in tag, (
            f"{needle!r} is gone from the tag step's `dist/.gitignore` "
            "exception. The exception is taken on the CONTENT — a regular "
            "file of exactly one byte, and that byte `*` — and every one of "
            "these four conditions is a shape that was rc=0 without it: a "
            "symlink (`-f` follows one, so `! -L` has to come first), a "
            "directory, a 4.4 MB `.zip` renamed, and a one-byte file that is "
            "not `*`. If it has become a pattern over hidden files instead, "
            "the answer rests on `actions/upload-artifact@v4` excluding them "
            "by default — third-party behaviour behind a mutable ref, which "
            "this file relies on in neither direction. Driven in "
            "`test_the_tag_gate_takes_the_gitignore_exception_on_content`."
        )
    for needle in ("dist/*.tar.gz", "shopt -s nullglob dotglob", "tar tzf",
                   "|| status=$?"):
        assert needle in manifest, f"{needle!r} is gone from the manifest step"
    # STDOUT ONLY, and this is the line the record's integrity rests on: with
    # `2>&1` here a `tar` that exits 0 and WARNS puts its warning in the
    # manifest beside the members, where a reader cannot tell them apart.
    # Driven in the manifest test's pax row.
    assert 'listing="$(tar tzf "${sdist}")"' in manifest, (
        "the manifest step's capture is no longer stdout-only. `2>&1` on it "
        "folds `tar`'s warnings into the record as though they were members, "
        "on the SUCCESS path, which is the path every release takes — and "
        "this step's whole product is the record."
    )
    # THE ABSENT CONSTRUCT, asserted absent rather than the fix pinned by
    # spelling: `release.yml` calls this step NOT a refusal and gives it no
    # count, so an `exit 1` in it is a refusal point nobody counted — and the
    # header's arithmetic would move without anyone deciding to move it.
    assert "exit 1" not in manifest, (
        "an `exit 1` has appeared in the manifest step. `release.yml` says "
        "beside it that it is not a refusal and asserts nothing about what it "
        "prints; the header counts refusal points and would now be wrong. If "
        "this step is meant to refuse, say so there and move the count."
    )
    assert "for sdist in dist/*.tar.gz" in manifest, (
        "the manifest step is back to reading ONE tarball. It is not a "
        "refusal — `release.yml` says so beside it — so with more than one "
        "tarball it must record all of them rather than pick, and with none it "
        "must record nothing rather than fail the release."
    )
    # `ls dist/` IS PINNED AS ABSENT, in the whole file rather than in one
    # body, for the reason the wheel needle above was dropped: the construct is
    # the defect. It stood at three sites and was retired in two passes —
    # `ls dist/*.whl` in the tag step, by the commit that made "the wheel" a
    # well-defined noun, and `ls dist/*.tar.gz` in the sdist step and in the
    # manifest step, here — and `ls` returns EVERY match, so a caller that
    # reads its output as one filename is picking (`basename` on the last
    # line) or crashing (`tar` on a filename with a newline in it). This pin
    # is over the whole file so that the manifest step, which no drive reads,
    # is covered by it too. CODE lines only:
    # `release.yml` and this module both describe the retired construct in
    # prose, and a whole-text scan would fail on the record of the defect.
    code = "\n".join(_code_lines(_release_text()))
    assert "ls dist/" not in code, (
        "`ls dist/` is back in `release.yml`. Every match, collated by the "
        "runner's locale, read by a caller that wants one filename — the shape "
        "driven in `test_the_tag_gate_refuses_a_dist_it_cannot_reason_about` "
        "and `test_the_sdist_gate_refuses_a_dist_it_cannot_read`."
    )
    # THE CHANGELOG STEP. `taggerdate:short` is the DECISION this file argues
    # beside that step — the tag OBJECT's date and not the committer date the
    # routing paragraph named — and `%(objecttype)` is what makes the
    # lightweight case a named refusal instead of an empty string compared to
    # a heading. `GITHUB_REF_NAME` rather than an `env:` block is the other
    # half: nothing is interpolated into that shell at all.
    # THE TAG RESOLUTION, PINNED ON CODE LINES ONLY — AND THIS COMMENT USED
    # TO CLAIM SOMETHING FALSE ABOUT ITSELF. It read: *"The BEHAVIOUR of all
    # three is driven in
    # test_the_changelog_gate_reads_the_tag_THROUGH_the_checkout_that_rewrites_its_ref;
    # these are here because a pin catches the rewrite a drive is blind to."*
    # That test drove `restore_ok` and drove NEITHER of the other two: every
    # row passed a well-formed `v0.2.1` and no row ever looked at a second tag
    # ref. Worse, the needles were scanned over the WHOLE extracted body,
    # comments included, and `release.yml` names both constructs in its own
    # prose beside them — so a mutant that disabled either guard while leaving
    # the sentence about it standing was green. Driven, full module each time:
    # `--no-tags` -> `--tags` with the literal kept in a comment, 37 passed;
    # the `check-ref-format` guard replaced by `if false` with the literal
    # kept, 37 passed. Both were behaviourally live — with a pattern-shaped
    # `GITHUB_REF_NAME` the shipped body rewrote no tag ref and the mutant
    # rewrote every one.
    #
    # BOTH HALVES ARE FIXED, because either alone leaves the other hole. The
    # scan is over CODE LINES now, so prose recording a construct cannot stand
    # in for the construct — the same decision `_code_lines` exists for one
    # file over. And the two guards are DRIVEN, in the last two rows of the
    # test named above: each plants an extra tag at `origin` after the
    # workspace is built and asserts it does not appear, which is exactly what
    # a glob refspec or a fetch that lost `--no-tags` drags in.
    for needle in ("git fetch --no-tags --force origin", "check-ref-format",
                   "restore_ok", "rev-parse --verify --quiet", "^{commit}",
                   "HEAD^{commit}"):
        assert needle in changelog, (
            f"{needle!r} is gone from the CODE of the changelog step body. "
            "That step reads `%(objecttype)` and `%(taggerdate:short)` off "
            "`refs/tags/<tag>`, and `actions/checkout@v4` force-writes the "
            "release COMMIT over that ref UNLESS the checkout is given "
            "`ref: ${{ github.sha }}` — so without the re-fetch that repairs "
            "it anyway, a workspace built without that input is read as a "
            "lightweight tag; and without the `^{commit}`/`HEAD` comparison "
            "the step will vouch for a tag that names a different commit than "
            "the one being built. It did the first to `v0.2.1` on 2026-08-28 "
            "and refused a correct tag; the second was driven passing a "
            "re-run whose tag had moved."
        )
    # AND THE SECOND LOOP READS CODE LINES TOO, WHICH IT DID NOT, AND THE
    # COMMIT THAT FIXED THE FIRST ONE IS WHAT BROKE IT. This loop scanned the
    # whole extracted body; at `a90862b` `%(taggerdate:short)` appeared on one
    # CODE line and nowhere else, so the needle was live. The commit that
    # added the tag re-fetch wrote a comment INSIDE the `run:` block spelling
    # the same literal, and the needle went inert — measured on both blobs:
    # `a90862b` 0 comment-line hits and 1 code-line hit, `fd03c02` 1 and 1.
    # Driven at `fd03c02`: `%(taggerdate:short)` -> `%(creatordate:short)` on
    # the CODE line alone left the module at 42 passed, and the same mutation
    # with the new comment's copy of the literal deleted was 2 failed. So the
    # tagger-date-versus-committer-date decision — the argument `release.yml`
    # spends about forty lines on — was pinned by nothing in this module or in
    # `tests/test_changelog_names_the_version.py`, which carries the same
    # needle over the same whole body and is repaired the same way.
    for needle in ("CHANGELOG.md", "git for-each-ref", "%(taggerdate:short)",
                   "%(objecttype)", "GITHUB_REF_NAME", "problems+=",
                   "refs/tags/"):
        assert needle in changelog, (
            f"{needle!r} is gone from the changelog step body. The date this "
            "gate compares against is the TAG OBJECT's tagger date, chosen "
            "over the commit's committer date on the argument beside the "
            "step; a rewrite to `git log -1 --format=%cd` refuses a release "
            "cut on any day but its last commit's, and the only way past THAT "
            "is to write a false date into CHANGELOG.md."
        )
    # AND THIS IS THE PIN CATCHING WHAT THE DRIVE CANNOT SEE, which is the
    # half of the argument the paragraph above usually gets to skip. THE
    # SENTENCE HERE SAID THE MUTATION WAS "3 failed" AND THAT WAS WRONG, and
    # the sentence three lines down is what gives it away: `%(creatordate)` IS
    # the tagger date for an annotated tag, so no row whose tag is annotated
    # can see the difference, and every drive row's tag is annotated or is
    # refused before a date is read. RE-MEASURED, all eight occurrences of the
    # literal in `release.yml` rewritten at once: 1 failed, which is this
    # needle and nothing else. The earlier figure came from a COMBINED mutant
    # that also forced `%(objecttype)` to `tag`, and it was the forcing that
    # reddened the extra rows. It is not a cosmetic
    # difference: `%(creatordate)` falls back to the COMMITTER date for a
    # lightweight tag (measured: a lightweight tag reads `creator=2026-08-28`
    # and `tagger=` empty), which is precisely the hybrid `release.yml`
    # rejects beside the step as a can't-tell resolving to the permissive
    # answer. A drive catches the rewrite nobody named; a pin catches the one
    # a drive is blind to, and this is the instance.
    # THE ABSENT CONSTRUCT, and it is the one the whole routing argument is
    # about: a clock. `tests/test_changelog_names_the_version.py` moved this
    # check out of the suite precisely because the only date available there
    # is `date.today()`, so a `$(date …)` arriving HERE would be the same
    # defect one file over — and it would be invisible to every text pin that
    # only asserts what IS present. Read on CODE lines: the paragraphs beside
    # the step have to be able to say what they refuse.
    changelog_code = "\n".join(
        line for line in changelog.splitlines()
        if not line.lstrip().startswith("#")
    )
    for construct in ("$(date", "`date ", "date +%"):
        assert construct not in changelog_code, (
            f"{construct!r} has appeared in the changelog step. The date this "
            "step compares must come from the TAG, never from the runner: a "
            "clock is the environment-dependence class that produced four of "
            "five recent CI reds, and moving this check out of the suite was "
            "the repair. The drives are the other half — the green rows plant "
            "dates fixed in the past, so a clock reading is red on them for "
            "any reader whose clock is not exactly that day."
        )
    # the bodies are scripts, not one-liners: a body that shrank to nothing
    # would still start with `set -euo pipefail` and pass the extractor's check
    assert len(sdist.splitlines()) > 20, sdist
    assert len(tag.splitlines()) > 5, tag
    assert len(manifest.splitlines()) > 5, manifest
    assert len(changelog.splitlines()) > 20, changelog
