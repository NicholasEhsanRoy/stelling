# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""`CHANGELOG.md`'s newest heading and `stelling.__version__`, tied.

Nothing tied them. They agreed when this file was written — `0.2.0.dev0` in
`src/stelling/__init__.py`, `## 0.2.0 — unreleased` at the top of the
changelog — so nothing shipped wrong, and that was the whole reason to write
it then rather than after it did.

**BOTH OF THOSE READINGS HAVE SINCE MOVED, AND THEY WERE WRITTEN IN THE
PRESENT TENSE.** The release bump made `__version__` `0.2.0` and put a date
on the heading — the OTHER of the two states this file checks — so two
sentences that were true on the branch that added them were false on the
merge that carried them, and no branch could have seen it: this file exists
on one side of that merge only, so the merge was textually clean. Nothing in
the tree could catch it either, because `tests/test_prose_hygiene.py` scopes
itself to shipped prose and `tests/` is outside it — while `tests/` is 181 of
the sdist's 379 members. They are dated records now rather than live claims,
which is the only form of illustration that cannot rot: it is the CHECK below
that carries the argument, and the check exercises both states.

It is the same unguarded coupling the tag gate in
`.github/workflows/release.yml` exists to close, one file over: **a value
maintained by hand against a value read from source**, with no check between
them. `pyproject.toml` already reads the version from
`src/stelling/__init__.py` (`[tool.hatch.version]`), so the wheel, the sdist
and the tag cannot disagree with the module. The changelog can, and a release
whose changelog heading names the previous version is a release note that
describes the wrong release.

**THE VERSIONS NEED NOT BE SPELLED THE SAME AND THE COUPLING IS STILL
EXACT.** At that commit `__version__` was `0.2.0.dev0` and the heading said
`0.2.0`, because a heading names the RELEASE and `__version__` names the
BUILD. PEP 440 makes that
relation total rather than a matter of taste: `0.2.0.dev0` is a development
build *of* `0.2.0`, its release segment is `(0, 2, 0)`, and `is_prerelease`
answers whether the build has shipped. So the assertion is on the release
segment, in both directions:

* the newest heading's version is the release `__version__` belongs to;
* and the two agree about whether it has HAPPENED — a `.devN` / `aN` / `bN` /
  `rcN` build's heading must say *unreleased*, and a final version's heading
  must carry a date. Without that second half the check passes forever on a
  changelog frozen at *"0.2.0 — unreleased"* through the 0.2.0 release
  itself, which is the same defect one word to the left.

**THIS DOES NOT GATE THE DATE, AND THAT IS DELIBERATE.** At release the
heading's date should equal the tag's date. The tag does not exist while this
suite runs — it is created by the release workflow, after this suite has to
have passed — so the only date a check here could compare against is
`datetime.date.today()`, i.e. **the developer's clock**. That is the
environment-dependence class that produced four of five recent CI reds on
`main`, arriving in a new place, and it would red every checkout whose author
wrote the heading yesterday. There is no version of it that is right.

**THE DATE CHECK IS IN `release.yml`, WHERE THE TAG EXISTS.** It is the step
*"the tag and the changelog heading must agree"*, in the `build` job, before
anything is uploaded — the fifteenth refusal point between a tag and PyPI, and
that file's header numbers and argues it. This check does not have the tag;
that one does.

**AND UNTIL THE COMMIT THAT ADDED IT, THIS PARAGRAPH WAS A RECOMMENDATION
STANDING AGAINST A FILE THAT DID NOT CARRY IT.** What stood here, verbatim:
*"THE DATE CHECK BELONGS IN `release.yml`, WHERE THE TAG EXISTS, and this file
does not own that workflow. The recommendation, written down so it is routable
rather than remembered: in the release job, after the tag is resolved, assert
that the newest `## <version> — <date>` heading in `CHANGELOG.md` carries the
tag's own committer date — `git log -1 --format=%cd --date=short
"$GITHUB_REF_NAME"` — and that its version equals the tag minus the leading
`v`. That check has the tag; this one cannot."* Every word of that was a
present-tense claim about another file, `.github/workflows/release.yml` did
not have the check, and **nothing in this repository read the sentence** — so
it could not go red, only stale. Measured: this module arrived at `0e79ede`
(committed 2026-08-25T00:32:48+02:00) and `git rev-list --count
0e79ede..9b5b496` is 47, so the recommendation stood over 47 commits
describing a gate that was never written. Prose asserting a check that does
not exist is the same defect as the missing check.

**WHAT HOLDS IT NOW, so it cannot go stale the way it just did.**
:func:`test_the_routed_date_check_is_in_this_workflow` below reads
`.github/workflows/release.yml` and refuses a tree in which the routed step is
gone, renamed, moved into `publish`, moved into a job the upload does not wait
for, or no longer reading the DOCUMENT, the TAG and the DATE this paragraph
names — and refuses one in which it has stopped carrying an `exit 1` at all,
because a step that reads all three and never refuses is a routing target in
name only. What it can and cannot see is in its own docstring, because a
text reader over a workflow file that does not declare its blindness is the
defect `tests/_lanes.py` spends a section on. The step's BEHAVIOUR is a
separate question and a separate instrument:
`tests/test_release_gates.py::test_the_changelog_gate_refuses_a_heading_the_tag_does_not_carry`
extracts that step's own body and drives it against planted trees carrying
real annotated tags, red and green.

**AND THE GATE READS A DIFFERENT DATE FROM THE ONE THIS PARAGRAPH ASKED FOR**,
recorded here and not only there. The recommendation named the tag's COMMITTER
date; the gate reads the TAG OBJECT's tagger date —
`git for-each-ref --format='%(taggerdate:short)' refs/tags/"$GITHUB_REF_NAME"`.
Measured 2026-08-28 at `9b5b496`, the two readings AGREE on both of this
repository's real tags — `v0.1.0` 2026-08-12 and `v0.2.0` 2026-08-25, both
ways — so the data cannot discriminate between them and the choice was made on
the argument, which is written beside the step: a heading's date is a claim
about when the RELEASE happened, and a release cut later than its last commit
has a committer date that is not that date, so the committer reading refuses a
correct heading and the only way past it is to write a false one. The
lightweight-tag case, which carries no tagger date at all, is a named refusal
there rather than a fallback.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import pytest

import stelling

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# THE WORKFLOW EXTRACTOR, BORROWED AND NOT RE-WRITTEN. `_step_lines` /
# `_step_body` / `_release_text` are the readers `tests/test_release_gates.py`
# already applies to `release.yml`, with the carriage-return normalisation
# that module had to learn the hard way and an anti-vacuity assertion on the
# extracted block. A second implementation of one file-reading rule is the
# defect `tests/test_workflows_make_no_silent_pick.py` refuses one directory
# over, and it would be a worse one here: two readers could disagree about
# what "the step" is, and this file's whole claim is that the step the OTHER
# module drives is the step this module routes to.
from test_release_gates import _code_lines, _release_text, _step_body  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = REPO / "CHANGELOG.md"

#: The step `release.yml` carries the routed check in. Named ONCE, here, and
#: used by the check below: the name is what `_step_body` addresses the step
#: by, so a rename must move one literal and not two.
ROUTED_STEP = "the tag and the changelog heading must agree"

#: A release heading. `## <version> — <rest>`, em dash, which is the shape
#: every heading in the file uses. `rest` is either `unreleased` or a date,
#: and which one it is carries half the claim.
_HEADING = re.compile(r"^##\s+(?P<version>\d[\w.+!-]*)\s+—\s+(?P<rest>.+?)\s*$", re.M)

#: PEP 440's pre-release and development spellings, which are what separates
#: "this build is of 0.2.0" from "0.2.0 has happened".
_UNSHIPPED = re.compile(r"(?:\.dev\d+|[ab]\d+|rc\d+)$")

#: The word a heading uses for a release that has not happened.
_UNRELEASED = "unreleased"

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def release_segment(version: str) -> str:
    """`0.2.0.dev0` -> `0.2.0`; `0.2.0` -> `0.2.0`.

    The release a build belongs to, by string surgery on PEP 440's own
    suffixes rather than by importing `packaging`: this file must run in the
    zero-dep lane, whose whole point is that the core imports nothing, and a
    version comparison that needs a dependency is a version comparison the
    zero-dep job cannot make.
    """
    return _UNSHIPPED.sub("", version)


def is_unshipped(version: str) -> bool:
    """Whether this version names a build that has not been released."""
    return bool(_UNSHIPPED.search(version))


#: A release heading's POSITION, independent of whether it parses. `## ` and
#: nothing else, because that is what makes a line a heading in this file;
#: whether it is a WELL-FORMED heading is the next question and not this one.
_ANY_HEADING_LINE = re.compile(r"^##\s.*$", re.M)


def headings(text: str) -> list[tuple[str, str]]:
    """`(version, rest)` for every release heading that PARSES, newest first.

    File order, not sorted: the changelog is written newest-first and the
    *newest heading* is a position in the document, not the largest version.
    Sorting here would quietly accept a file whose newest entry had been
    inserted in the wrong place.

    **This is not the function to ask for the newest heading** — see
    :func:`newest_heading`, and the paragraph below for why the difference is
    a defect rather than a nicety.
    """
    return [(m.group("version"), m.group("rest")) for m in _HEADING.finditer(text)]


def newest_heading(text: str) -> tuple[str, str] | None:
    """The newest heading, FOUND by position and PARSED second, or `None`.

    **`headings(text)[0]` WAS THE NEWEST HEADING AND IT WAS NOT**, and the
    difference is a false GREEN rather than a confusing red.
    :data:`_HEADING` is applied with `finditer`, which **skips** what it
    cannot match — so a newest heading that does not parse is stepped over and
    `[0]` becomes an OLDER one. Today that usually reds for the wrong reason,
    naming the wrong version. On the day `stelling.__version__` happens to
    equal that older heading's version — which is exactly the state a release
    bump passes through — **it goes green with a malformed heading standing
    above it**, and the one coupling this module exists to hold is not held.

    Found by the branch that built the same gate in
    `.github/workflows/release.yml`, which refuses that shape deliberately:
    its step locates the first `## ` line and *then* parses it, so a malformed
    newest heading is a refusal and never a step past. The two readers now
    agree about what "newest" means, which they did not.

    `None` means **no `## ` line at all**; a line that is present and does not
    parse raises :exc:`MalformedNewestHeading`, because a can't-read resolving
    to the same answer as a nothing-to-read is how the two stop being
    distinguishable — and they are the two cases the caller's message has to
    tell apart.
    """
    line = _ANY_HEADING_LINE.search(text)
    if line is None:
        return None
    parsed = _HEADING.match(line.group(0))
    if parsed is None:
        raise MalformedNewestHeading(line.group(0))
    return parsed.group("version"), parsed.group("rest")


class MalformedNewestHeading(ValueError):
    """The first `## ` line is not a `## <version> — <rest>` heading.

    Its own exception type rather than a `None`, so that a caller cannot
    handle it by accident with the same branch that handles an empty file.
    """


def test_the_newest_changelog_heading_names_this_version():
    """The coupling. Two halves, and the second is what keeps it live.

    Half one: the heading's version is the release `stelling.__version__`
    belongs to. Half two: the heading and the version agree about whether
    that release has happened.
    """
    text = CHANGELOG.read_text(encoding="utf-8")
    try:
        found = newest_heading(text)
    except MalformedNewestHeading as bad:
        raise AssertionError(
            f"{CHANGELOG.name}'s newest `## ` line does not parse as a "
            f"release heading: {str(bad)!r}. It is NOT stepped over to reach "
            f"an older one that does — that step is how this check went green "
            f"over a malformed heading whenever the older one happened to "
            f"name the current version."
        ) from None
    assert found is not None, (
        f"{CHANGELOG.name} has no `## <version> — <something>` heading, so "
        f"there is nothing to tie `stelling.__version__` to. A changelog "
        f"whose headings stopped parsing is a changelog this check reads as "
        f"satisfied, which is why the shape is asserted before the content."
    )
    version = stelling.__version__
    newest, rest = found

    assert newest == release_segment(version), (
        f"`CHANGELOG.md`'s newest heading names {newest!r} and "
        f"`stelling.__version__` is {version!r}, which is a build of "
        f"{release_segment(version)!r}. The changelog is the release note for "
        f"the version in `src/stelling/__init__.py` — `pyproject.toml` reads "
        f"the distribution's version from that same line — so a heading "
        f"naming anything else describes a different release."
    )

    if is_unshipped(version):
        assert rest.strip().lower() == _UNRELEASED, (
            f"`stelling.__version__` is {version!r}, a build of "
            f"{newest!r} that has not shipped, and the heading says "
            f"{rest!r}. It must say {_UNRELEASED!r}: a dated heading over an "
            f"unshipped version is a release note claiming a release date "
            f"the release has not got."
        )
    else:
        assert _ISO_DATE.match(rest.strip()), (
            f"`stelling.__version__` is {version!r}, a released version, and "
            f"the heading says {rest!r}. A released version's heading carries "
            f"the date it was released. (The date's VALUE is not checked "
            f"here, on purpose — see this module's docstring: the only date "
            f"available to this suite is the developer's clock.)"
        )


def test_this_gate_does_not_read_the_clock():
    """The trap, closed by assertion rather than by intention.

    A date check here would have to call `date.today()`, because the tag that
    carries the real date is created after this suite runs. That is the
    environment-dependence class four of five recent reds came from, and the
    way it arrives is somebody adding it later in good faith. So the module is
    checked for the imports that would make it possible.

    Not a substitute for reading the file — it is a tripwire on the one
    mechanism, and it says so. The date check belongs in `release.yml`; the
    recommendation is in this module's docstring, and this agent does not own
    that workflow.
    """
    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    clocks = {"datetime", "time", "calendar", "today", "now", "utcnow", "monotonic"}
    reached = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in clocks
        }
        | {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in clocks
        }
        | {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
            if alias.name.split(".")[0] in clocks
        }
        | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            and node.module.split(".")[0] in clocks
        }
    )
    assert not reached, (
        f"this module reaches for a clock: {reached}. The tag that carries a "
        f"release's real date does not exist when this suite runs, so a date "
        f"compared here is compared against whoever is running the tests. "
        f"Put the date check in `.github/workflows/release.yml`, after the "
        f"tag is resolved."
    )
    # Read off the SYNTAX and not the bytes, because the prose above has to be
    # able to say `date.today()` in order to explain why it is refused — the
    # first draft of this test scanned the file's lines and failed on its own
    # docstring, which is the same defect one layer down.


def test_the_routed_date_check_is_in_this_workflow():
    """The paragraph above ROUTES a check to another file. This reads that file.

    THE DEFECT IT CLOSES IS ITS OWN HISTORY. That paragraph was a
    recommendation for 47 commits — a present-tense claim about
    `.github/workflows/release.yml`, naming a check that file did not carry,
    with nothing in the tree reading it. It could not go red; it could only go
    stale, and it did. So the claim is now an assertion: the routed step is
    HERE, in a job the upload waits for and is not, reading the three things
    the routing names, and still carrying a refusal.

    WHAT "ACTUALLY THERE" MEANS TO A TEXT READER, and it is four things and not
    one:

    * the STEP resolves, exactly once, by the name this module routes to.
      `_step_lines` refuses zero matches and two matches alike, so a rename is
      a red rather than a scan that finds nothing and is satisfied.
    * its BODY is a real script — `_step_body` refuses a block that does not
      open with the step's own `set -euo pipefail`, so a `run:` read as empty
      cannot satisfy the needles below by containing nothing.
    * that body names `CHANGELOG.md`, `GITHUB_REF_NAME` and
      `%(taggerdate:short)` — the document, the tag and the date, which are
      the three nouns the routing paragraph is made of — and still carries an
      `exit 1`. A step that stopped reading any one of the three is not the
      routed check under its own name, and one that reads all three and
      refuses nothing is a routing target in name only.
    * the step is in a job that `publish` WAITS FOR and is not `publish`
      itself. A refusal that runs beside the upload is not a refusal, and one
      in a job nothing waits for is not in the release path at all.

    **WHAT THIS CHECK CANNOT SEE.** `tests/_lanes.py`'s docstring is the
    reason this paragraph exists: a line-anchored text reader over
    `.github/workflows/nightly-jax-canary.yml` went past NINE legal spellings
    of the thing it was looking for — a quoted key, a flow mapping, an alias,
    `matrix.exclude`, a job-level `if:`, a `defaults:` block, `set -a` with a
    sourced file, `continue-on-error: true`, and `runs-on:` written before
    `name:` — and reported "no setting" for each. This reader is the same kind
    of instrument over a different file, and:

    * IT IS BLIND TO EVERY RESPELLING OF THE STEP — `- {name: …, run: …}`,
      `"name": …`, an anchor and alias, `run: >` instead of `run: |` — but it
      fails LOUDLY on all of them rather than quietly: each one takes
      `_step_lines` or `_step_body` to zero matches, which is an assertion
      error naming the step. That is the difference between this and the
      nightly reader, and it is a property of `_step_lines`, not of this test.
      Measured with PyYAML on the `release.yml` this commit ships,
      2026-08-28: it carries 0 anchors, 0 aliases and 0 merge keys, its jobs
      are `build`, `publish` and `test`, and the routed step parses as
      `jobs.build.steps[1]` with the checkout still `steps[0]`. That is a
      dated reading of one file and not a guarantee about the next edit of
      it — and PyYAML is not a dependency of this project, so the reading
      could be taken but not kept: `tests/_lanes.py` explains why neither
      module parses.
    * IT CANNOT SEE WHETHER THE STEP RUNS. A step-level or job-level `if:`
      switches it off and nothing in this repository reads `if:` at all —
      said plainly because the sibling attribute IS covered:
      `tests/test_release_gates.py::test_the_attributes_that_decide_whether_a_refusal_refuses`
      holds `continue-on-error` absent for the whole file, and
      `::test_the_publish_job_still_needs_the_suite` holds `needs`. `if:` is
      the hole, and it is a hole in `release.yml`'s coverage generally rather
      than one this check introduces.
    * IT CANNOT SEE WHETHER THE CHECK WORKS. Four needles in a script are not
      a verdict about what the script does: a body that names all four and
      never reaches its `exit 1` satisfies every assertion here. What decides
      that is
      `tests/test_release_gates.py::test_the_changelog_gate_refuses_a_heading_the_tag_does_not_carry`,
      which runs this same extracted body against planted trees, and
      `::test_the_changelog_gate_reads_the_NEWEST_heading_by_POSITION`.
    * IT CANNOT SEE WHETHER THE WORKFLOW RUNS AT ALL — the `on:` trigger, the
      repository's own branch protection, whether the `pypi` environment
      exists. That is the runner's furniture and no test here reaches it.
    """
    body = _step_body(ROUTED_STEP)

    missing = [
        needle for needle in ("CHANGELOG.md", "GITHUB_REF_NAME",
                              "%(taggerdate:short)", "exit 1")
        if needle not in body
    ]
    assert not missing, (
        f"the step {ROUTED_STEP!r} in `.github/workflows/release.yml` no "
        f"longer reads {missing}. This module routes the changelog DATE check "
        f"there — it cannot make it here, because the tag that carries the "
        f"real date does not exist while this suite runs — and a routing "
        f"paragraph whose target has stopped carrying the check is the exact "
        f"defect this test was written for: it stood as a recommendation for "
        f"47 commits and nothing could see it. Either restore the check or "
        f"rewrite the paragraph above to say where it went."
    )

    # THE JOB IT STANDS IN, because a refusal that runs beside the upload is
    # not a refusal. Read off the code lines: job headers are the only thing
    # in this file at indent 2 that ends in a colon.
    lines = _code_lines(_release_text())
    at = [i for i, line in enumerate(lines)
          if line.strip() == f"- name: {ROUTED_STEP}"]
    assert len(at) == 1, (
        f"expected exactly one step named {ROUTED_STEP!r}; found {len(at)}. "
        f"(`_step_body` above addresses the step by that name, so this cannot "
        f"normally be reached — it is here so that a divergence between the "
        f"two readers is a red rather than a silent disagreement about which "
        f"step is being spoken of.)"
    )
    header = re.compile(r"^  ([a-z][a-z0-9-]*):\s*$")
    job = next((m.group(1) for m in
                (header.match(line) for line in reversed(lines[:at[0]])) if m),
               None)
    assert job is not None, (
        "could not tell which job the routed step is in, so this check cannot "
        "say whether it runs before the upload. The job-header shape this "
        "reads — a name at indent 2 — has changed."
    )
    assert job != "publish", (
        f"the routed changelog check is in the `{job}` job, which is the job "
        f"that uploads to PyPI. A refusal that runs beside the upload it is "
        f"meant to prevent is not a refusal."
    )
    publish = [i for i, line in enumerate(lines) if line.strip() == "publish:"]
    assert len(publish) == 1, (
        f"expected exactly one `publish:` job in `release.yml`, found "
        f"{len(publish)}; this check cannot say what waits for what."
    )
    # BOUNDED TO THAT JOB, and PARSED rather than substring-tested. Scanning
    # to the end of the file would read some LATER job's `needs:` if the job
    # order ever changed, and `job in needs` over the whole line is true for
    # any job whose name is a substring of anything else on it — a check that
    # a claim is well-formed rather than that it is true, which is the defect
    # class this whole branch is about.
    tail = lines[publish[0] + 1:]
    for stop, line in enumerate(tail):
        if header.match(line):
            after = tail[:stop]
            break
    else:
        after = tail
    needs = next((line for line in after
                  if line.strip().startswith("needs:")), None)
    waits_for = re.findall(r"[A-Za-z0-9_-]+",
                           "" if needs is None else needs.split(":", 1)[1])
    assert job in waits_for, (
        f"`publish` waits for {waits_for} and the routed changelog check "
        f"stands in the `{job}` job (`needs:` line read as {needs!r}). A gate "
        f"in a job the upload does not depend on is a gate the upload does "
        f"not pass through: it goes red beside a green publish."
    )

def test_both_halves_of_the_coupling_are_driven():
    """The rule, observed to fire, on planted changelogs.

    Green on this tree, so the only evidence it works is a plant. Four of
    them: the version wrong, the *unreleased*/dated agreement wrong in each
    direction, and a control that the real pairing is accepted — because a
    check that refused everything would pass all three reds and be useless.
    """
    def verdict(version: str, heading: str) -> str | None:
        """The complaint this rule makes about a `(version, heading)` pair."""
        try:
            found = newest_heading(heading)
        except MalformedNewestHeading:
            # A `## ` line that is not a heading is its OWN complaint, and the
            # drive below plants one standing above a well-formed older
            # heading — the shape `headings()[0]` used to step past.
            return "malformed-newest"
        if found is None:
            return "unparseable"
        newest, rest = found
        if newest != release_segment(version):
            return "version"
        if is_unshipped(version):
            return None if rest.strip().lower() == _UNRELEASED else "should-be-unreleased"
        return None if _ISO_DATE.match(rest.strip()) else "should-be-dated"

    # THE CONTROL, and it is the real pairing rather than an invented one.
    assert verdict(
        stelling.__version__, CHANGELOG.read_text(encoding="utf-8")
    ) is None, "the rule refuses this tree's own version/heading pair"

    assert verdict("0.3.0.dev0", "## 0.2.0 — unreleased\n") == "version", (
        "a heading naming the PREVIOUS release while `__version__` has moved "
        "on is not caught — that is the shape a forgotten changelog bump has"
    )
    assert verdict("0.2.0.dev0", "## 0.2.0 — 2026-08-12\n") == "should-be-unreleased", (
        "a DATED heading over an unshipped `.devN` version is not caught"
    )
    assert verdict("0.2.0", "## 0.2.0 — unreleased\n") == "should-be-dated", (
        "an `unreleased` heading over a RELEASED version is not caught, so "
        "this check would pass forever on a changelog frozen at the moment "
        "before the release it documents"
    )
    # ... and the release-segment surgery is the thing both halves rest on.
    for build, release in (
        ("0.2.0.dev0", "0.2.0"), ("1.0.0rc1", "1.0.0"), ("0.9.0b2", "0.9.0"),
        ("0.9.0a1", "0.9.0"), ("0.2.0", "0.2.0"), ("1.2.3.dev17", "1.2.3"),
    ):
        assert release_segment(build) == release, build
        assert is_unshipped(build) == (build != release), build


def test_a_malformed_newest_heading_is_REFUSED_and_never_stepped_past():
    """The false GREEN `headings(text)[0]` had, driven on the shape that has it.

    **THIS IS THE ONE THE OLD READER GOT WRONG, AND IT IS A GREEN RATHER THAN
    A RED.** `_HEADING.finditer` skips what it cannot match, so a newest
    heading that does not parse was stepped over and `[0]` became an OLDER
    one. Usually that reds for the wrong reason — it names the wrong version.
    But when the older heading happens to name the current release segment,
    which is exactly the state a version bump passes through, every assertion
    downstream is satisfied by a heading that is not the newest and the file
    is accepted with a malformed heading standing above it.

    The plant below is that state, minimally: a `## ` line with no em dash at
    all, above a well-formed `## 0.2.0 — 2026-08-25`, read against `0.2.0`.
    Both readings are taken here rather than described, so the difference is a
    measurement:

    * the OLD reading — `headings(plant)[0]` — is `("0.2.0", "2026-08-25")`,
      i.e. the second heading, and every check downstream passes;
    * the NEW reading refuses, because it locates the first `## ` line and
      *then* parses it.

    That is the same order `.github/workflows/release.yml`'s changelog step
    uses, deliberately, and the two readers now agree about what "newest"
    means.
    """
    plant = "## 0.2.0 no em dash here\n\n## 0.2.0 — 2026-08-25\n"

    # THE OLD READING, taken rather than remembered. `headings` still exists
    # and still skips; keeping it is what lets this test be a measurement of
    # the difference instead of a story about it.
    stepped_past = headings(plant)
    assert stepped_past and stepped_past[0] == ("0.2.0", "2026-08-25"), (
        "the premise of this test is that `headings()[0]` steps past a "
        "malformed newest heading and returns an older one; it did not, so "
        "either `_HEADING` or the plant has changed and this test is no "
        "longer measuring what it says"
    )

    # ... and every downstream assertion is satisfied by that older heading,
    # which is what makes the old behaviour a GREEN and not a RED.
    older_version, older_rest = stepped_past[0]
    assert older_version == release_segment("0.2.0")
    assert _ISO_DATE.match(older_rest.strip())

    # THE NEW READING refuses it, by its own exception type.
    with pytest.raises(MalformedNewestHeading) as caught:
        newest_heading(plant)
    assert "no em dash here" in str(caught.value), caught.value

    # An empty file and a headingless file are the OTHER case, and they stay
    # distinguishable: `None`, not an exception.
    assert newest_heading("") is None
    assert newest_heading("nothing here\n\nnot a heading either\n") is None
