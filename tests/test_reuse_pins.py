# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The two `reuse` pins guard one property, and nothing made them move together.

CI lints with `fsfe/reuse-action@v5`; `.pre-commit-config.yaml` pins
`reuse-tool` `v6.2.0`. Two versions, one property. That was a curiosity until
`insert-license` was narrowed to exclude `scratchpad/`, at which point what
makes those files compliant is `REUSE.toml`'s `scratchpad/**` `[[annotations]]`
block — and an annotation is only as good as the version that reads it. Thirty
or so tracked files under `scratchpad/` carry no inline header at all and rest
entirely on that block (counted below, so the number cannot be wrong here).

WHAT THIS FILE IS AND IS NOT. It is a DRIFT DETECTOR, and that is all it was
ever able to be — but the sentence that used to say so here overreached and is
WITHDRAWN. It read: "it cannot install `reuse 5.1.1`, cannot reach the
network, and therefore cannot measure whether 5.1.1 and 6.2.0 read
`REUSE.toml` the same way. That question is bounded, not closed." The
*therefore* was wrong. reuse 5.1.1 was on the box the whole time, in pip's own
wheel cache; assembled from there offline, it lints THIS tree IDENTICALLY to
the pinned 6.2.0 — the same rc and the same count from both, the same file
PATHS from `lint --json` and not merely the same total, and deleting the
`scratchpad/**` annotation (the positive control) moves both by the same files
in the same direction. THE CLAIM IS THAT EQUALITY, not any of the numbers: the
absolutes move with the next file added, so each is recorded with the commit it
was read on. Re-measured 2026-08-09 at `53f9f84`: rc=0 350/350 from both, rc=1
321/350 from both under the control, 350 paths from each with an empty
symmetric difference. An earlier reading, at an unnamed tip, gave 331/331 and
302/331 — the same equality at different values, which is the point. The
question is CLOSED on this tree, by measurement, and the commands are at the
`reuse` job in `.github/workflows/ci.yml`.

None of which retires this file. A measurement is about the two versions it
ran; a bump on either side makes it stale, and nothing but a test notices a
bump. **What is still SUSPECTED, and it is now exactly one link:** that
`fsfe/reuse-action@v5` carries exactly 5.1.1, which is read off the action's
own pinning and needs `git ls-remote` to confirm. The changelog bound (newest
release documenting ANY change to `REUSE.toml` semantics is v5.0.0, below both
pins; 52 releases vendored, v0.0.2 to v6.2.0, so the window is not truncated;
the 5.1.1→6.2.0 window yields exactly one hit, about the sort order of `reuse
lint --lines` — which is also the ONLY difference the measurement found) is
still recorded there and is no longer the only evidence.

What this file adds is the thing the write-up asked for and did not have: a
control that FAILS if either pin moves without the other being reconsidered. A
prose paragraph describing a version skew goes stale the moment someone bumps
one side; a test does not.
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CI = os.path.join(ROOT, ".github", "workflows", "ci.yml")
PRECOMMIT = os.path.join(ROOT, ".pre-commit-config.yaml")
REUSE_TOML = os.path.join(ROOT, "REUSE.toml")

# THE RECORDED PAIR. Change these two lines only after doing the four steps
# written at the `reuse` job in .github/workflows/ci.yml — in particular step
# 4, `pip install reuse==<the action's version> && reuse lint` on this tree,
# which is the measurement no reading of a changelog can replace.
CI_ACTION_REF = "fsfe/reuse-action@v5"
PRECOMMIT_REV = "v6.2.0"

# The reuse release `fsfe/reuse-action@v5` carries. NOT verifiable from this
# repository and not verifiable offline: it is read off the action's own
# pinning. Recorded so that a future reader knows which version the skew is
# against, and labelled for what it is.
CI_ACTION_REUSE_VERSION_SUSPECTED = "5.1.1"

_DRIFT = (
    "The two `reuse` pins have drifted apart from what this file records.\n"
    "  ci.yml           : {ci}\n"
    "  pre-commit       : {pc}\n"
    "  recorded here    : {want_ci} / {want_pc}\n"
    "This is not a formatting nit. CI's `reuse` job and the local hook guard "
    "ONE property — that every tracked file carries copyright and licence "
    "information — and since `insert-license` stopped touching `scratchpad/`, "
    "what covers those files is `REUSE.toml`'s `scratchpad/**` annotation, "
    "read by whichever version runs. Two versions reading one annotation is a "
    "skew; a skew nobody notices is the defect.\n"
    "Do the four steps at the `reuse` job in .github/workflows/ci.yml before "
    "editing the constants above, and record what you measured."
)


def _one_line_break(text: str) -> str:
    """``text`` with every line break spelled ``\\n`` -- done HERE.

    THE THREE SCANS BELOW ARE `^`-ANCHORED UNDER `re.M`, and Python's `^`
    there matches after a newline and NOT after a carriage return. Until
    2026-08-22 they were correct only because `open()` in text mode
    translated the file's line breaks before they ran -- a property of how
    the file was OPENED and not of the check, which is the accident
    `tests/test_tripwire_record.py::_lines_of_this_grammar` refused to
    leave standing for the workflow grammar after `^---` missed a document
    marker opened by a CR.

    Measured on this tree, the same `ci.yml` and `.pre-commit-config.yaml`
    rendered with CR line endings and read without translation: the
    `fsfe/reuse-action@` pin goes from one hit to NONE -- loud -- and the
    two `re.sub`s that strip the pin lines out of what this file then calls
    prose strip NOTHING, which is quiet and is the direction that matters.
    A pin left standing in "prose" satisfies the write-up assertion without
    a word of write-up existing.

    So the files are opened with `newline=""` and normalised here.
    `test_the_reuse_pins_are_READ_however_the_files_lines_end` holds it,
    with the untranslated rendering as its negative control.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read(path: str) -> str:
    # `newline=""` so nothing is translated on the way in and the
    # normalisation is visibly this reader's. See `_one_line_break`.
    with open(path, encoding="utf-8", newline="") as fh:
        return _one_line_break(fh.read())


def test_the_reuse_pins_are_READ_however_the_files_lines_end():
    """A carriage return is a line break, and `re.M`'s `^` does not know it.

    Driven both ways on this repository's own two files: the CRLF and
    CR-only renderings read back to the same pin and the same stripped
    prose, and the CR-only rendering with its breaks LEFT ALONE finds no
    pin at all and strips no line -- which is what `_one_line_break` is
    worth. The quiet half is the second: an unstripped `uses:` line leaves
    the version named in what this file calls the write-up, so the
    assertion that a human was told how to close the skew passes on a file
    with no write-up in it.
    """
    ci = _read(CI)
    precommit = _read(PRECOMMIT)
    assert "\r" not in ci and "\r" not in precommit, (
        "`_read` handed back a carriage return, so the normalisation it "
        "exists to do did not happen"
    )

    def scans(one_ci: str, one_pc: str):
        return (
            re.findall(r"^\s*-?\s*uses:\s*(fsfe/reuse-action@\S+)\s*$",
                       one_ci, re.MULTILINE),
            len(re.findall(r"^\s*-?\s*uses:\s*\S+\s*$", one_ci,
                           re.MULTILINE)),
            len(re.findall(r"^\s*rev:\s*\S+\s*$", one_pc, re.MULTILINE)),
        )

    wanted = scans(ci, precommit)
    assert wanted[0] and wanted[1] and wanted[2], (
        f"one of this module's anchored scans no longer finds anything in "
        f"the clean files, so the test below would watch nothing: {wanted}"
    )

    for label, breaks in (("CRLF", "\r\n"), ("CR only", "\r")):
        rendered = (ci.replace("\n", breaks), precommit.replace("\n", breaks))
        assert scans(*[_one_line_break(text) for text in rendered]) == wanted, (
            f"a {label} rendering of the two files does not read back to "
            f"the pin and the stripped lines this module reads"
        )

    # THE NEGATIVE CONTROL. Without the normalisation above, a CR-only
    # checkout finds no pin (loud) and strips no pin line out of the prose
    # (quiet, and permissive).
    blind = scans(ci.replace("\n", "\r"), precommit.replace("\n", "\r"))
    assert blind == ([], 0, 0), (
        f"CR-only renderings of `ci.yml` and `.pre-commit-config.yaml` were "
        f"expected to be invisible to all three anchored scans and were not: "
        f"{blind}. If `re.M` has changed, say so where `_one_line_break` "
        f"explains why it exists"
    )


def test_the_two_reuse_pins_are_the_pair_this_repository_recorded():
    """DRIFT DETECTOR. Either pin moving on its own is the failure."""
    ci = _read(CI)
    precommit = _read(PRECOMMIT)

    # the STEP, not the prose: the write-up around it names `@v6` too, as the
    # thing a human would move to, and a comment is not a pin.
    ci_refs = re.findall(
        r"^\s*-?\s*uses:\s*(fsfe/reuse-action@\S+)\s*$", ci, re.MULTILINE
    )
    assert ci_refs, "no `fsfe/reuse-action` step found in ci.yml at all"
    pc_rev = re.search(
        r"repo:\s*https://github\.com/fsfe/reuse-tool\s*\n\s*rev:\s*(\S+)",
        precommit,
    )
    assert pc_rev, "no `fsfe/reuse-tool` repo/rev pair found in .pre-commit-config.yaml"

    assert set(ci_refs) == {CI_ACTION_REF} and pc_rev.group(1) == PRECOMMIT_REV, (
        _DRIFT.format(
            ci=sorted(set(ci_refs)),
            pc=pc_rev.group(1),
            want_ci=CI_ACTION_REF,
            want_pc=PRECOMMIT_REV,
        )
    )


def test_the_skew_is_still_written_up_where_the_pins_are():
    """A pin without its write-up is a pin nobody can act on. Both files have
    to keep naming BOTH versions, so a bump that edits one number and leaves
    the prose describing the other cannot pass quietly.

    ONE CONJUNCT HERE WAS IMPLIED BY ANOTHER TEST AND COULD NOT FAIL. It read
    `local in precommit`, where `local` is `6.2.0` — but
    :func:`test_the_two_reuse_pins_are_the_pair_this_repository_recorded`
    already requires the literal `rev: v6.2.0` in that same file, so the
    substring was guaranteed by the pin itself. MEASURED: rewriting all SIX
    prose mentions of `6.2.0` in `.pre-commit-config.yaml` to a nonsense token
    and leaving only `rev: v6.2.0` still gave `3 passed`. The write-up could
    be deleted wholesale and this test would not notice, which is the one
    thing it exists to notice. It now reads the file with the pin LINE
    REMOVED, so the prose has to carry the version on its own."""
    ci = _read(CI)
    precommit = _read(PRECOMMIT)
    suspected = CI_ACTION_REUSE_VERSION_SUSPECTED
    local = PRECOMMIT_REV.lstrip("v")

    # the write-up is everything that is NOT the pin the other test asserts;
    # `6.2.0` inside `rev: v6.2.0` is that pin, not prose about it.
    precommit_prose = re.sub(
        r"^\s*rev:\s*\S+\s*$", "", precommit, flags=re.MULTILINE
    )
    # likewise for ci.yml: the `uses:` step is the pin, the comments are prose
    ci_prose = re.sub(r"^\s*-?\s*uses:\s*\S+\s*$", "", ci, flags=re.MULTILINE)

    assert suspected in ci_prose and local in ci_prose, (
        f"ci.yml no longer names both reuse versions ({suspected} and {local}) "
        f"ANYWHERE BUT THE `uses:` STEP; the `reuse` job's write-up is what "
        f"tells a human how to close this, and a step is not a write-up."
    )
    assert suspected in precommit_prose and local in precommit_prose, (
        f".pre-commit-config.yaml no longer names both reuse versions "
        f"({suspected} and {local}) ANYWHERE BUT THE `rev:` PIN; the "
        f"`insert-license` comment's argument for excluding scratchpad/ "
        f"depends on saying which version reads the annotation that replaced "
        f"it, and the pin line does not say it."
    )
    # and the write-up must still point at where the bound was taken
    assert "v5.0.0" in ci, (
        "ci.yml no longer records WHICH release the REUSE.toml bound comes "
        "from; the bound is the only thing standing in for a measurement here."
    )


def test_the_scratchpad_annotation_is_load_bearing_and_present():
    """WHAT THE SKEW IS ABOUT, pinned so it cannot be quietly removed.

    `insert-license` excludes `scratchpad/`, so the `[[annotations]]` block for
    `scratchpad/**` is the only thing making the header-less files there
    compliant. The census below is the non-vacuity floor: if it ever reaches 0,
    this test would be guarding nothing and says so instead of passing."""
    toml = _read(REUSE_TOML)
    block = re.search(
        r'\[\[annotations\]\]\s*\npath\s*=\s*\["scratchpad/\*\*"\]\s*\n'
        r'precedence\s*=\s*"aggregate"\s*\n'
        r'SPDX-FileCopyrightText\s*=\s*"[^"]+"\s*\n'
        r'SPDX-License-Identifier\s*=\s*"[^"]+"',
        toml,
    )
    assert block, (
        "REUSE.toml no longer carries the `scratchpad/**` annotation with "
        "`precedence = \"aggregate\"` and both SPDX fields. `insert-license` "
        "excludes `scratchpad/` (.pre-commit-config.yaml), so removing this "
        "block leaves the header-less evidence files there uncovered — and "
        "`reuse lint` in CI runs a DIFFERENT reuse version from the pinned "
        "local one, so a local green is not the same evidence as a CI green."
    )

    tracked = subprocess.run(
        ["git", "ls-files", "scratchpad"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    # `return` HERE WAS A PASS, and the comment on it said the opposite. It
    # read `# not a git checkout: say so, do not pass` and then returned,
    # which is exactly a pass — the non-vacuity floor below never ran and
    # nothing said so. MEASURED on the unfixed tree: a non-git copy of this
    # repository with `scratchpad/` absent ENTIRELY still gave `3 passed`,
    # the floor silently skipped. A guard that cannot examine its subject
    # must not report the same thing as a guard that examined it and was
    # satisfied, so this skips (visible in `-ra`) instead of returning.
    if tracked.returncode != 0:
        # THE REASON IS A CONSTANT AND THE DETAIL IS A WARNING, from
        # 2026-08-22. `tests/test_skip_inventory.py` excuses a skip only by an
        # EXACT string typed on its own disclosure surface, and this reason
        # carried `rc=` and git's stderr — so no rule could ever have named
        # it, and this skip was one of the NINE that made a git-less tree exit
        # 1 no matter what (`1 failed, 2106 passed, 159 skipped` in the
        # zero-dep lane with `.git` removed, at `844ba48`). Nothing is lost:
        # the floor is still not run, the skip still says so, and the
        # `rc`/`stderr` a reader would want are in a warning rather than in a
        # string `pytest -rs` truncates to the terminal width anyway.
        #
        # NO `stacklevel=2` HERE, AND THAT IS NOT AN OVERSIGHT. This `warn`
        # is in the TEST BODY, so one frame out is pytest's own caller:
        # `stacklevel=2` attributed it to `_pytest/python.py`, the line where
        # the runner calls the test function, and pointed the reader at the
        # runner instead of at the check that could not run. The two in
        # `tests/test_soundness_routing.py` sit in a HELPER the test calls,
        # so their `stacklevel=2` lands on the test file, which is what all
        # three want; the number differs because the depth does. Measured in
        # a `.git`-less tree: each routing warning is attributed to the line
        # in that file where its TEST calls the helper -- which is what
        # `stacklevel=2` means and is a line a reader can act on -- and this
        # one to the `warnings.warn` below. NO LINE NUMBER IS QUOTED FOR
        # THEM: the first draft of this comment named one, and it had
        # already moved by eleven before the commit, because the same fixup
        # added a paragraph to that file's docstring.
        #
        # LATENT UNDER `-W error`, and inert today: `pyproject.toml` sets no
        # `filterwarnings` and no workflow passes `-W`. If warnings are ever
        # promoted, these disclosed skips become errors rather than skips —
        # driven in a `.git`-less zero-dep tree, `-W error::UserWarning` turns
        # `23 passed, 9 skipped` into `9 failed, 23 passed` across this file
        # and the routing one, so the sdist lane would go red for a reason
        # that is a disclosure working correctly. The fix if that day comes is
        # a `filterwarnings` entry naming these, not deleting the detail.
        warnings.warn(
            f"`git ls-files scratchpad` failed (rc={tracked.returncode}, "
            f"stderr={tracked.stderr.strip()!r}), so this test's non-vacuity "
            f"floor did not run. The annotation assertion above DID."
        )
        pytest.skip("not a git checkout (an unpacked sdist, say)")
    headerless = []
    for rel in tracked.stdout.split("\n"):
        if not rel:
            continue
        path = os.path.join(ROOT, rel)
        try:
            with open(path, "rb") as fh:
                head = fh.read(4096).decode("utf-8", "replace")
        except OSError:
            continue
        if "SPDX-License-Identifier" not in "\n".join(head.split("\n")[:5]):
            headerless.append(rel)
    assert len(headerless) > 0, (
        "every tracked file under scratchpad/ now carries an inline SPDX "
        "header, so the annotation this test guards covers nothing and the "
        "test above would pass vacuously. Either the exclusion in "
        "`insert-license` was reverted, or the census is broken — check which."
    )
