# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""`CHANGELOG.md`'s newest heading and `stelling.__version__`, tied.

Nothing tied them. They agree today — `0.2.0.dev0` in
`src/stelling/__init__.py`, `## 0.2.0 — unreleased` at the top of the
changelog — so nothing ships wrong, and that is the whole reason to write this
now rather than after it does. It is the same unguarded coupling the tag gate
in `.github/workflows/release.yml` exists to close, one file over: **a value
maintained by hand against a value read from source**, with no check between
them. `pyproject.toml` already reads the version from
`src/stelling/__init__.py` (`[tool.hatch.version]`), so the wheel, the sdist
and the tag cannot disagree with the module. The changelog can, and a release
whose changelog heading names the previous version is a release note that
describes the wrong release.

**THE VERSIONS ARE NOT SPELLED THE SAME AND THE COUPLING IS STILL EXACT.**
`__version__` is `0.2.0.dev0` and the heading says `0.2.0`, because a heading
names the RELEASE and `__version__` names the BUILD. PEP 440 makes that
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

**THE DATE CHECK BELONGS IN `release.yml`, WHERE THE TAG EXISTS**, and this
file does not own that workflow. The recommendation, written down so it is
routable rather than remembered: in the release job, after the tag is
resolved, assert that the newest `## <version> — <date>` heading in
`CHANGELOG.md` carries the tag's own committer date — `git log -1
--format=%cd --date=short "$GITHUB_REF_NAME"` — and that its version equals
the tag minus the leading `v`. That check has the tag; this one cannot.
"""

from __future__ import annotations

import ast
import pathlib
import re

import stelling

REPO = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = REPO / "CHANGELOG.md"

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


def headings(text: str) -> list[tuple[str, str]]:
    """`(version, rest)` for every release heading, newest first.

    File order, not sorted: the changelog is written newest-first and the
    *newest heading* is a position in the document, not the largest version.
    Sorting here would quietly accept a file whose newest entry had been
    inserted in the wrong place.
    """
    return [(m.group("version"), m.group("rest")) for m in _HEADING.finditer(text)]


def test_the_newest_changelog_heading_names_this_version():
    """The coupling. Two halves, and the second is what keeps it live.

    Half one: the heading's version is the release `stelling.__version__`
    belongs to. Half two: the heading and the version agree about whether
    that release has happened.
    """
    text = CHANGELOG.read_text(encoding="utf-8")
    found = headings(text)
    assert found, (
        f"{CHANGELOG.name} has no `## <version> — <something>` heading, so "
        f"there is nothing to tie `stelling.__version__` to. A changelog "
        f"whose headings stopped parsing is a changelog this check reads as "
        f"satisfied, which is why the shape is asserted before the content."
    )
    version = stelling.__version__
    newest, rest = found[0]

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


def test_both_halves_of_the_coupling_are_driven():
    """The rule, observed to fire, on planted changelogs.

    Green on this tree, so the only evidence it works is a plant. Four of
    them: the version wrong, the *unreleased*/dated agreement wrong in each
    direction, and a control that the real pairing is accepted — because a
    check that refused everything would pass all three reds and be useless.
    """
    def verdict(version: str, heading: str) -> str | None:
        """The complaint this rule makes about a `(version, heading)` pair."""
        found = headings(heading)
        if not found:
            return "unparseable"
        newest, rest = found[0]
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
