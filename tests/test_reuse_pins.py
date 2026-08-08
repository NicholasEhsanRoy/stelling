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

WHAT THIS FILE IS AND IS NOT. It is a DRIFT DETECTOR, not a resolution: it
cannot install `reuse 5.1.1`, cannot reach the network, and therefore cannot
measure whether 5.1.1 and 6.2.0 read `REUSE.toml` the same way. That question
is bounded, not closed, and the bound is written up at the `reuse` job in
`.github/workflows/ci.yml` — the newest release documenting ANY change to
`REUSE.toml` semantics is v5.0.0, below both pins. That reading was re-verified
against the vendored 6.2.0 `CHANGELOG.md` (52 releases, v0.0.2 to v6.2.0, so
the window is not truncated; grepping the 5.1.1→6.2.0 window for `REUSE.toml`,
`annotation`, `precedence`, `glob` and `path` returns exactly one line, about
the sort order of `reuse lint --lines`). It is a bound on what the changelog
DOCUMENTS. **SUSPECTED, not measured.**

What this file adds is the thing the write-up asked for and did not have: a
control that FAILS if either pin moves without the other being reconsidered. A
prose paragraph describing a version skew goes stale the moment someone bumps
one side; a test does not.
"""

from __future__ import annotations

import os
import re
import subprocess

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


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


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
    the prose describing the other cannot pass quietly."""
    ci = _read(CI)
    precommit = _read(PRECOMMIT)
    suspected = CI_ACTION_REUSE_VERSION_SUSPECTED
    local = PRECOMMIT_REV.lstrip("v")

    assert suspected in ci and local in ci, (
        f"ci.yml no longer names both reuse versions ({suspected} and {local}); "
        f"the `reuse` job's write-up is what tells a human how to close this."
    )
    assert suspected in precommit and local in precommit, (
        f".pre-commit-config.yaml no longer names both reuse versions "
        f"({suspected} and {local}); the `insert-license` comment's argument "
        f"for excluding scratchpad/ depends on saying which version reads the "
        f"annotation that replaced it."
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
    if tracked.returncode != 0:  # not a git checkout: say so, do not pass
        return
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
