# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""No command-substituted `ls` on a CODE LINE anywhere in `.github/workflows/`.

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

**BUT A COMMENT IS NOT A CODE LINE, AND THE FIRST DRAFT DID NOT KNOW THAT.**
It read the workflow files as plain text, and on the tree where
`release/0.2.0-dist-unit` had *retired* every one of the three live
constructions it reported **four fresh offences — all four of them comments**:
the paragraphs `release.yml` grew to record what was removed and why, one of
them a deliberately-worded anti-vacuity note.

**A gate that forbids writing about the defect you just fixed is worse than
no gate**, because it pushes the record out of the file that carries it, and
this repository's practice is that the record lives beside the code it is
about. So the scan reads **code lines**: everything on a line up to a `#`
that opens a comment.

**THE COMMENT RULE IS NOT A SECOND IMPLEMENTATION.** It is
``_lanes._strip_comment`` — *"THE ONE STRIP"* — which this repository already
uses to read `ci.yml` and `release.yml`, and whose own docstring states the
rule: **`#` opens a comment when it BEGINS A WORD** (at the start of the
line, or after whitespace) **and never inside a quoted string**. So `foo#bar`
and a URL fragment survive; `value  # note` becomes `value`; and a `#` inside
a quoted scalar is DATA, so ``run: echo "a # b $(ls dist)"`` is still an
offence. Writing a second comment-stripper here is the defect this file is
about, one directory over.

**WHAT THE COMMENT RULE DOES NOT COVER, said rather than implied.**

* **Block scalars are not modelled, and the rule is applied to their bodies
  anyway.** Inside `run: |` YAML has no comments at all — every `#` is
  data — but that data is a *shell script*, where `#`-begins-a-word is the
  same rule. That coincidence is why one function can serve both layers, and
  it is a coincidence: a `#` inside a block scalar that is neither a YAML
  comment nor a shell comment — **heredoc body text**, an unquoted `printf`
  argument — is stripped as though it were one, and a `$(ls …)` behind it
  goes unreported.
* **A quoted scalar that spans lines is read as several lines.** Quote state
  is per line, so the continuation lines of a multi-line flow scalar are read
  with a fresh state, and a `#` in one of them is taken for a comment.
* **An unbalanced quote earlier in the line suppresses the strip.** ``the
  wheel's`` opens a quote that never closes, so a trailing `#` after it is
  not seen and the whole line is read as code. That direction is the safe
  one — the gate over-reports rather than going quiet — and the four comments
  that prompted this are whole-line comments, where the `#` is the first
  non-blank character and nothing precedes it at all.
* **It is a line rule, not a parse.** Nothing here knows which lines are a
  `run:` body, so nothing here can tell a YAML comment between two steps from
  a shell comment inside one. Both are prose about the file, both are
  stripped, and a `$(ls …)` that a shell comment has commented **out** is
  therefore not reported — the same judgement, made twice.

**AND WHAT THE GATE AS A WHOLE DOES NOT COVER, unchanged.** A `$(ls …)`
inside a script the workflow *calls* — a file under `tools/`, an action from
the marketplace — is out of scope, as is `find`/`echo` piping into `head -1`,
which is the same silent pick in a different spelling. This gate is one
construction in one directory, and its value is that the construction is
unambiguous.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _lanes  # noqa: E402

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"

#: THE ONE STRIP, borrowed rather than re-written. See the module docstring:
#: a second implementation of a rule this repository already states in one
#: place is the shape of defect this file exists to refuse.
_code_lines = _lanes._code_lines

#: `$(ls …)` and its backtick twin. `\bls\b` after the opener so that
#: `$(lsof …)` and `$(tools/ls-something)` are not swept in — the rule is
#: about the `ls` COMMAND, and a gate that fired on a substring would be
#: turned off by the first false accusation.
_COMMAND_SUBSTITUTED_LS = re.compile(r"(?:\$\(\s*|`\s*)ls(?=\s|\))")


def workflow_files() -> list[pathlib.Path]:
    """Every workflow file this gate reads, in a stable order."""
    if not WORKFLOWS.is_dir():
        return []
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def offenders() -> list[tuple[str, int, str]]:
    """`(file, line, text)` for every command-substituted `ls` on a code line.

    The line REPORTED is the raw one — a reader has to see what is in the
    file — and the line MATCHED is the code line.
    """
    found = []
    for path in workflow_files():
        text = path.read_text(encoding="utf-8")
        raw = text.splitlines()
        for number, code in enumerate(_code_lines(text), 1):
            if _COMMAND_SUBSTITUTED_LS.search(code):
                found.append((path.name, number, raw[number - 1].strip()))
    return found


def test_no_workflow_command_substitutes_ls():
    """The gate.

    **RED on `main` at `115d771` and on this branch, three real instances**,
    all in `release.yml`: the wheel pick at the wheel-inspection step, and the
    sdist twin at each of the two steps that unpack a tarball. Every one of
    them puts a locale-collated choice between `dist/` and the gate that is
    supposed to be inspecting what will ship. **GREEN once
    `release/0.2.0-dist-unit` is in the tree**: it retires all three and
    writes up each removal in place — and those write-ups are the reason this
    scan reads code lines rather than text.

    This file does not edit the workflow and does not allowlist it: an
    allowlist entry for the exact construction a gate was written to refuse is
    the gate arriving switched off.
    """
    found = offenders()
    rendered = "\n  ".join(f"{name}:{line}  {text}" for name, line, text in found)
    assert not found, (
        f"{len(found)} command-substituted `ls` on code lines in "
        f"`.github/workflows/`:\n  "
        f"{rendered}\n\n"
        f"`ls` renders filenames; it does not produce them, and with more "
        f"than one match it picks one by locale collation and says nothing. "
        f"A glob into an array is the correct shape — `wheels=(dist/*.whl)` "
        f"— because it makes the count a thing the script can check, which "
        f"is exactly what this construction throws away. A COMMENT IS NOT A "
        f"CODE LINE and is not reported here, so recording a retired "
        f"construction beside the code that used to carry it is not an "
        f"offence: if a line above is a comment, the strip is broken and not "
        f"the workflow."
    )


def test_the_workflow_directory_is_actually_there():
    """Anti-vacuity, and half of this file's exposure.

    :func:`offenders` answers `[]` for a missing directory, so in a tree with
    no `.github/` the gate above is satisfied by having read nothing. **An
    unpacked sdist is not such a tree, and the sentence here used to say it
    was**: `pyproject.toml`'s sdist allowlist names `/.github` beside
    `/tests`, so the workflows ship with the suite that reads them and this
    gate runs over the real files wherever the suite runs. That leaves one
    honest reason for the empty answer — a tree that has LOST
    `.github/workflows/` — and it is not a reason to be quiet, because such a
    checkout has lost the release path. So the two are separated: this test
    asserts the directory is present and non-empty HERE, and fails rather
    than skipping.
    """
    assert WORKFLOWS.is_dir(), (
        f"{WORKFLOWS} is not a directory, so the gate beside this one read "
        f"no files and passed on an empty set"
    )
    assert workflow_files(), f"{WORKFLOWS} holds no workflow files"


def test_the_strip_did_not_empty_the_corpus():
    """The other half, and it is NEW WITH THE COMMENT RULE.

    The test above proves there are files. It does not prove there is
    anything left of them once the comments are gone, and that became a real
    exposure the moment this gate stopped reading raw text: a strip that
    returned `""` for every line — a quote rule that never closes, a `#` test
    that matched from column zero — would leave the gate green over a corpus
    of blanks, and every word of the docstring above would still be true of a
    scan that reads nothing. `release.yml` is a heavily commented file and
    that is the point: the surviving code has to still be most of a workflow.
    """
    for path in workflow_files():
        text = path.read_text(encoding="utf-8")
        code = _code_lines(text)
        assert len(code) == len(text.splitlines()), (
            f"{path.name}: the strip returned {len(code)} lines for a file of "
            f"{len(text.splitlines())} — the code lines must index like the "
            f"file, or every line number this gate reports is someone else's"
        )
        alive = sum(1 for line in code if line.strip())
        assert alive > len(code) // 10, (
            f"{path.name}: only {alive} of {len(code)} lines survived the "
            f"comment strip. The gate beside this one is then reading almost "
            f"nothing and passing for it."
        )


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


def test_a_comment_is_prose_and_a_quoted_hash_is_data():
    """The code-line rule, driven in both directions on the real shapes.

    The four comment lines are the ones `release.yml` actually grew when the
    construction was retired, copied here verbatim, so that a strip which
    stopped seeing them fails HERE with their text in front of the reader
    rather than by reporting four offences nobody can act on. Each is
    asserted to match the pattern FIRST: a line the pattern never saw drives
    nothing, and would leave this test passing on a typo.

    The other direction is the half a comment rule usually gets wrong. `#`
    inside a quoted scalar is DATA, so the shell after it is still shell and
    a `$(ls …)` there is still a silent pick; and a comment written AFTER a
    substitution does not unmake the substitution.
    """
    comments = (
        "      # it replaced a silent PICK. `ls dist/*.whl` returns EVERY match and",
        '      # be easy to copy that reading across. `sdist="$(ls dist/*.tar.gz)"` is',
        "      # non-refusal. It carried the LAST `ls dist/` in this file — the second",
        "      # of the two `ls dist/*.tar.gz` sites, the wheel's having been retired in",
        '          # wheel="$(ls dist/*.whl)"   # retired; see the step header',
    )
    for line in comments:
        assert _COMMAND_SUBSTITUTED_LS.search(line), (
            f"{line!r} does not match the pattern even raw, so it drives "
            f"nothing about the strip"
        )
        assert not _COMMAND_SUBSTITUTED_LS.search(_lanes._strip_comment(line)), (
            f"the strip left {line!r} matching. A gate that reports a comment "
            f"forbids recording the defect it just caught."
        )

    code = (
        '          wheel="$(ls dist/*.whl)"',           # the defect itself
        '          run: echo "a # b $(ls dist)"',       # `#` inside "…" is data
        "          run: echo 'a # b $(ls dist)'",       # and inside '…' too
        '          x="$(ls dist)"  # a note about it',  # code, then a comment
    )
    for line in code:
        assert _COMMAND_SUBSTITUTED_LS.search(_lanes._strip_comment(line)), (
            f"the strip removed the code in {line!r}. A `#` inside a quoted "
            f"scalar is data, and a comment after a silent pick does not "
            f"unmake the pick."
        )


def test_the_strip_is_the_one_this_repository_already_has():
    """No second comment rule, and this is what says so.

    `_code_lines` above is a BINDING, so a rename or a signature change in
    `tests/_lanes.py` fails at import and this file cannot silently grow its
    own copy. What a binding does not catch is somebody replacing the binding
    with a local function that looks the same and drifts — so the identity is
    asserted, not assumed, and the rule the borrowed function implements is
    driven here on the two shapes `_lanes`'s own docstring names.
    """
    assert _code_lines is _lanes._code_lines, (
        "this file no longer reads its code lines with `_lanes._code_lines`. "
        "Two implementations of one comment rule is the defect this gate is "
        "about, one directory over."
    )
    assert _lanes._strip_comment("value  # note") == "value"
    assert _lanes._strip_comment("foo#bar") == "foo#bar"
    assert _lanes._strip_comment('a: "x # y"') == 'a: "x # y"'
