# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""No command-substituted `ls` anywhere in `.github/workflows/`.

`wheel="$(ls dist/*.whl)"` was in the release path. With one wheel in `dist/`
it reads as the wheel; with two it is a **locale-collated pick of one of
them**, silently, and every gate downstream then inspects a file nobody chose.
It is the failure shape this project keeps naming in other places: an
instrument that reports the same thing whether there was one answer or three.

**THE RULE IS ABSOLUTE AND THAT IS THE POINT.** Not *"quote it"*, not
*"check the count first"* — no command-substituted `ls`, at all, in any
workflow. Three reasons it is stated that way rather than as a heuristic:

* `ls` output is **not a filename**. It is a rendering of filenames, collated
  by locale, and its escaping and column behaviour depend on whether stdout is
  a terminal. Nothing that must be a path should come from it.
* the correct shape already exists and costs one line —
  ``mapfile -t wheels < <(printf '%s\\n' dist/*.whl)`` or
  ``wheels=(dist/*.whl)`` — and an array makes *"how many were there"* a
  question the script can ask, which is the question `$(ls …)` erases.
* a rule with an exception is a rule whose exception is where the next one
  lands. *Zero* is checkable by grep; *"zero, unless the author was sure"* is
  not checkable at all.

**THIS AGENT DOES NOT OWN `.github/`.** The three instances below are live on
`main` at `115d771` and a sibling is retiring them. The red is the report;
this file does not edit the workflow and does not allowlist it, because an
allowlist entry for the exact construction a gate was written to refuse is the
gate arriving switched off.

**WHAT THIS DOES NOT COVER, said rather than implied.** It reads the workflow
files as TEXT. A `$(ls …)` inside a script the workflow *calls* — a file under
`tools/`, an action from the marketplace — is out of scope, as is
`find`/`echo` piping into `head -1`, which is the same silent pick in a
different spelling. This gate is one construction in one directory, and its
value is that the construction is unambiguous.
"""

from __future__ import annotations

import pathlib
import re

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"

#: `$(ls …)` and its backtick twin. `\bls\b` after the opener so that
#: `$(lsof …)` and `$(tools/ls-something)` are not swept in — the rule is
#: about the `ls` COMMAND, and a gate that fired on a substring would be
#: turned off by the first false accusation.
_COMMAND_SUBSTITUTED_LS = re.compile(r"(?:\$\(\s*|`\s*)ls(?=\s|\))")


def offenders() -> list[tuple[str, int, str]]:
    """`(file, line, text)` for every command-substituted `ls` in a workflow."""
    if not WORKFLOWS.is_dir():
        return []
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if _COMMAND_SUBSTITUTED_LS.search(line):
                found.append((path.name, number, line.strip()))
    return found


def test_no_workflow_command_substitutes_ls():
    """The gate. **RED on `main` at `115d771`, three real instances.**

    All three are in `release.yml`: the wheel pick at the wheel-inspection
    step, and the sdist twin at each of the two steps that unpack a tarball.
    Every one of them puts a locale-collated choice between `dist/` and the
    gate that is supposed to be inspecting what will ship.
    """
    found = offenders()
    rendered = "\n  ".join(f"{name}:{line}  {text}" for name, line, text in found)
    assert not found, (
        f"{len(found)} command-substituted `ls` in `.github/workflows/`:\n  "
        f"{rendered}\n\n"
        f"`ls` renders filenames; it does not produce them, and with more "
        f"than one match it picks one by locale collation and says nothing. "
        f"A glob into an array is the correct shape — `wheels=(dist/*.whl)` "
        f"— because it makes the count a thing the script can check, which "
        f"is exactly what this construction throws away."
    )


def test_the_workflow_directory_is_actually_there():
    """Anti-vacuity, and it is the whole of this file's exposure.

    :func:`offenders` answers `[]` for a missing directory, so in a tree with
    no `.github/` the gate above is satisfied by having read nothing. That is
    the right behaviour for an unpacked sdist — the workflows are not shipped
    — and the wrong thing to be silent about in a checkout, so the two are
    separated: this test asserts the directory is present and non-empty
    HERE, and fails rather than skipping, because a checkout of this
    repository that has lost `.github/workflows/` has lost the release path.
    """
    assert WORKFLOWS.is_dir(), (
        f"{WORKFLOWS} is not a directory, so the gate beside this one read "
        f"no files and passed on an empty set"
    )
    workflows = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert workflows, f"{WORKFLOWS} holds no workflow files"


def test_the_pattern_is_driven_both_ways():
    """The construction, observed to fire — and observed NOT to.

    The near-misses matter as much as the hits: a pattern that also caught
    `lsof`, or a path with `ls` in its name, would be a pattern somebody
    disables. And a pattern that stopped matching backticks would leave the
    older spelling of the same defect uncovered while this file went on
    reporting green.
    """
    fires = (
        'wheel="$(ls dist/*.whl)"',
        "wheel=`ls dist/*.whl`",
        'sdist="$( ls dist/*.tar.gz )"',
        'name="$(ls)"',
    )
    for line in fires:
        assert _COMMAND_SUBSTITUTED_LS.search(line), (
            f"the pattern does not see {line!r}, which is the defect itself"
        )
    quiet = (
        "        run: ls dist/",          # a diagnostic, not a value
        'pids="$(lsof -t -i:8080)"',      # a different command
        'x="$(tools/ls-report.sh)"',      # a path that merely contains `ls`
        'echo "$(cat dist/SHA256SUMS)"',  # a substitution that is not `ls`
        "wheels=(dist/*.whl)",            # the correct shape
    )
    for line in quiet:
        assert not _COMMAND_SUBSTITUTED_LS.search(line), (
            f"the pattern fires on {line!r}, which is not a silent pick. A "
            f"gate that makes a false accusation is a gate somebody switches "
            f"off, and then the real ones come back."
        )
