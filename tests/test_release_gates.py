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

WHAT THIS FILE IS AND IS NOT. It is a TEXT PIN, in the same shape as
`tests/test_reuse_pins.py`: it reads the file and asserts the load-bearing
literals are still there. That is enough to kill all six, and it is measured
against them below rather than argued. It is NOT a check that the gates WORK —
nothing in this repository can be, because a gate's behaviour is a property of
a runner and the honest local substitute is driving each step body by hand,
which is what `release.yml`'s own comments record having done. A pin catches
the mutation; only a drive catches the misconception.

STILL UNGUARDED AFTER THIS, said plainly rather than left to be assumed:

* every gate BODY's behaviour. `comm -23` being present says nothing about
  whether the comparison is right; `-ra` being present says nothing about what
  pytest prints. Both are recorded in the file with the run that established
  them, and this pin is what keeps that record attached to the code it
  describes.
* the `publish` job's two actions, which cannot be driven from here at all
  without cutting a tag and uploading.
* everything about the runner: which interpreter `uv venv` picks, whether the
  `pypi` environment exists, whether Trusted Publishing is configured.

Read as TEXT and not with a YAML parser, deliberately: `yaml` is not a
dependency of this project, and the zero-dep CI job — the one whose whole
purpose is an environment with nothing in it — could not import one.
"""
from __future__ import annotations

import pathlib
import re

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


def _release_text() -> str:
    assert RELEASE.is_file(), f"no release workflow at {RELEASE}"
    return RELEASE.read_text(encoding="utf-8")


def _code_lines(text: str) -> list[str]:
    """Lines that are not comments — the mutations all landed in these."""
    return [
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ]


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
    matters is any rewrite that leaves it empty when it should not be. The
    literal is pinned because the semantic cannot be tested without a runner.
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
