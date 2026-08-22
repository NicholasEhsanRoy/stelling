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
5. the header's `EIGHT` -> `TWELVE` — a count of refusals nobody can check;
6. `comm -23` -> `comm -12` — the "every sdist member is committed" comparison
   inverted, so it reports the INTERSECTION, is empty exactly when the tree is
   healthy, and can never fire.

WHAT THIS FILE IS. Two things, and the second was called IMPOSSIBLE here for
four commits.

The first is a TEXT PIN, in the same shape as `tests/test_reuse_pins.py`: it
reads the file and asserts the load-bearing literals are still there. That is
enough to kill all six mutations above, and it is measured against them below
rather than argued.

The second is a DRIVE. Two gate bodies are EXTRACTED from `release.yml` and
EXECUTED here against planted trees, and asserted on their exit code and their
annotation. Extracted, not copied: a test that runs its own transcription of a
step body is a test of the transcription, and the rewrite it has to catch
lands in `release.yml`.

THE SENTENCE THAT USED TO STAND HERE WAS FALSE, and how it was false is worth
writing down, because it is what licensed the gap. It read: this is "NOT a
check that the gates WORK — nothing in this repository can be, because a
gate's behaviour is a property of a runner". The premise does not hold for
these two bodies. Neither contains one thing a runner supplies: they are
`tar`, `git ls-files`, `sort`, `comm`, `ls`, `basename` and `cut`, over a
`dist/` directory and a git index, both of which a test can plant. And
`release.yml`'s own comments record driving these same bodies by hand at
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
  What is driven is the two bodies above and nothing else.
* the sdist drive's plant is keyed on the allowlist's DIRECTORY roots, so a
  member filter keyed on anything else is outside it, as the paragraph above
  says.
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

import os
import pathlib
import re
import shutil
import subprocess
import tarfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RELEASE = REPO / ".github" / "workflows" / "release.yml"

# The header's own count, spelled as a word. Only the words a refusal count
# could plausibly be; an unlisted word fails loudly rather than silently
# reading as zero.
_NUMBER_WORDS = {
    "ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
    "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10, "ELEVEN": 11,
    "TWELVE": 12,
}

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
    ("the refusal-count header",
     r"^# ([A-Z]+) REFUSAL POINTS STAND BETWEEN A TAG AND PyPI"),
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
    repository's own `release.yml` read back to the same four hits, and the
    CR-only rendering with its breaks LEFT ALONE is invisible to all four
    -- which is what the normalisation is worth.
    """
    text = _release_text()
    assert "\r" not in text, (
        "`_release_text` handed back a carriage return, so the "
        "normalisation it exists to do did not happen"
    )
    wanted = {name: re.findall(pattern, text, re.M)
              for name, pattern in _ANCHORED_SCANS}
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


def test_the_headers_refusal_COUNT_is_the_count_of_refusals():
    """MUTANT 5, and the reason it is worth a test rather than a proof-read: a
    number in a comment is exactly the thing nothing checks, and this one had
    already been wrong once — it said TWO, which was the count of REASONS
    listed under it, while the sdist checks landed later and did not move it.

    Derived from the file, not from a constant here: the `exit 1` sites are
    counted, and the two refusals that are a step's own exit code are added
    from :data:`_IMPLICIT_REFUSALS`, which is written down because deciding
    mechanically which `- run:` can fail is deciding what every command does.
    """
    text = _release_text()
    lines = _code_lines(text)
    exit_sites = [line for line in lines if line.strip() == "exit 1"]
    header = re.search(
        r"^# ([A-Z]+) REFUSAL POINTS STAND BETWEEN A TAG AND PyPI", text, re.M
    )
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
    # ...and the header's own breakdown must agree with the same arithmetic
    assert "six `exit 1` sites" in text and len(exit_sites) == 6, (
        f"the header's breakdown says six `exit 1` sites and there are "
        f"{len(exit_sites)}"
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
    assert 'unexplained="$(comm -23 members.txt explained.txt)"' in text, (
        "the sdist committed-members comparison is no longer `comm -23 "
        "members.txt explained.txt`. `-23` suppresses columns 2 and 3, leaving "
        "lines ONLY in members.txt — the unexplained ones. `-12` leaves the "
        "common lines, which is not a check at all."
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

_NEEDED = ("bash", "git", "tar", "comm", "sort", "sed")
_needs_a_shell = pytest.mark.skipif(
    any(shutil.which(t) is None for t in _NEEDED),
    reason="needs a POSIX shell and coreutils to drive a gate body",
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
_TAG_STEP = "the tag and the artifact must agree"

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


def _plant_tree(tree: pathlib.Path, *, uncommitted: tuple[str, ...] = ()) -> pathlib.Path:
    """A git checkout plus a `dist/` holding one sdist, as the build job leaves it.

    Each `uncommitted` path goes into the TARBALL and deliberately not into the
    index — the exact shape this gate exists to refuse: a file that ships and is
    not in the tagged tree.
    """
    (tree / "dist").mkdir(parents=True)
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

    with tarfile.open(tree / "dist" / "stelling-0.1.0.tar.gz", "w:gz") as tar:
        for rel in sorted(members):
            tar.add(tree / rel, arcname=f"stelling-0.1.0/{rel}")
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

    healthy = _drive(body, _plant_tree(tmp_path / "healthy"), **env)
    assert healthy.returncode == 0, (
        "the sdist gate refuses a tree in which every member IS committed; a "
        f"gate that cannot pass is not a gate.\n{healthy.stdout}\n{healthy.stderr}"
    )
    assert "sdist members is committed to this tree" in healthy.stdout

    planted = tuple(_plants())
    bad = _drive(body, _plant_tree(tmp_path / "planted", uncommitted=planted), **env)
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

    THE TWO-WHEEL WEAKNESS IS NOT RE-DRIVEN HERE and is not this test's
    subject: `ls` sorts, so a stray that sorts BEFORE the real wheel is read
    past and uploaded anyway. That is recorded in `release.yml` beside the
    step, with the runs that established it, and is unreachable from that
    workflow because `dist/` is filled by one `uv build` into a fresh clone.
    """
    body = _step_body(_TAG_STEP)
    dist = tmp_path / "tree" / "dist"
    dist.mkdir(parents=True)
    (dist / "stelling-0.1.0-py3-none-any.whl").write_bytes(b"")
    tree = tmp_path / "tree"

    agreeing = _drive(body, tree, TAG="v0.1.0")
    assert agreeing.returncode == 0, (
        "the tag gate refuses a tag that MATCHES the built artifact, so it "
        f"would refuse every release.\n{agreeing.stdout}\n{agreeing.stderr}"
    )
    assert "artifact version=0.1.0" in agreeing.stdout

    for tag in ("v9.9.9", "0.2.0", "", "V0.1.0"):
        r = _drive(body, tree, TAG=tag)
        assert r.returncode != 0, (
            f"THE TAG GATE PASSED tag {tag!r} against a 0.1.0 wheel. This is "
            "the refusal that stops a release tagged one version from putting "
            "another on PyPI permanently. It reported:\n" + r.stdout +
            "\nCheck that `version` is still read from the ARTEFACT's filename "
            "and not from ${TAG}, which would be the tag compared with itself."
        )
        assert "tag and artifact disagree" in r.stdout


@_needs_a_shell
def test_the_drives_are_reading_the_real_step_bodies():
    """Anti-vacuity for the two drives, in the shape this file already uses.

    The drives assert on exit codes of a script this file did not write. Three
    ways that goes quietly wrong — a step name that no longer resolves, a
    `run:` block read as empty, an `env:` that stopped carrying `GENERATED` —
    and the first two are fatal in `_step_lines`/`_step_body` already. This
    pins what the extracted text must CONTAIN, so that a body reduced to its
    `set -euo pipefail` cannot satisfy the drives above by exiting 0.
    """
    sdist = _step_body(_SDIST_STEP)
    tag = _step_body(_TAG_STEP)
    for needle in ("tar tzf", "git ls-files", "comm -23", "explained.txt"):
        assert needle in sdist, f"{needle!r} is gone from the sdist step body"
    for needle in ("basename", "cut -d- -f2", "ls dist/*.whl"):
        assert needle in tag, f"{needle!r} is gone from the tag step body"
    # the bodies are scripts, not one-liners: a body that shrank to nothing
    # would still start with `set -euo pipefail` and pass the extractor's check
    assert len(sdist.splitlines()) > 20, sdist
    assert len(tag.splitlines()) > 5, tag
