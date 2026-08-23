# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""No shipped file may point at a path the sdist does not carry.

**THE RULE IS NOT THIS FILE'S. IT IS `pyproject.toml`'S, WRITTEN ABOUT A
DIFFERENT DIRECTORY AND ENFORCED BY NOTHING.** The sdist allowlist ships
`/tools` for one stated reason — `tests/property/README.md` ships too, and
tells a reader to run `tools/property_check.py` — and states the general
rule beside it:

    A shipped instruction pointing at a path the artefact does not contain
    is a claim defect waiting to be found by somebody who trusted it.

**MEASURED AT `c7cf164`, ONE DIRECTORY OVER: `SOUNDNESS.md` ships and
`scratchpad/` does not, and the page carried 48 CITATIONS of 32 DISTINCT
PATHS under it** — 51 occurrences of the string `scratchpad/` once the
three bare mentions of the directory name are counted in, which is the
figure the batch was opened on and is a different unit. All 32 exist in the
checkout, so nothing was broken for a reader of the repository and every one
of the 48 dangled for a reader of the tarball. `docs/harness-api.md` carried
two more, and those two already said *"Both are in the git checkout; neither
is in the sdist"* — which is the honest form, and the form nothing was
holding anyone to.

**THE GATE'S FIRST LIVE CATCH WAS ONE COMMIT LATER, AND IT IS THE REASON
THIS FILE IS SHAPED AS A PARTITION RATHER THAN A SWEEP.** D7 merged at
`da91658`, and `docs/choosing-a-solver-backend.md` and
`tools/solver_battery.py` arrived carrying six citations of
`scratchpad/D7-solver-battery/` — the same shape as the 32, in two files
that ship, pointing at a directory that does not. Rebased onto it, before
anything was repaired::

    docs/choosing-a-solver-backend.md: scratchpad/D7-solver-battery/
    docs/choosing-a-solver-backend.md: scratchpad/D7-solver-battery/probe-does-the-freedom-reach-the-number.py
    docs/choosing-a-solver-backend.md: scratchpad/D7-solver-battery/probe-where-does-the-box-stop.py
    tools/solver_battery.py:           scratchpad/D7-solver-

That fourth line is the useful one: `tools/solver_battery.py` had the path
SPLIT ACROSS TWO STRING LITERALS, so the scan saw a truncated token where a
reader saw a whole path. Both are repaired at the sites — the disposal, not
a declaration — and the split literal is one literal now.

Shipping `scratchpad/` is not the fix and the same comment says why: the
allowlist exists BECAUSE shipping working files leaked an internal release
checklist, untracked and deliberately uncommitted, byte-identical inside
`stelling-0.1.0.tar.gz`, and **an sdist on PyPI cannot be unpublished**. So
the shipped files say what their citations are, and this file holds them to
it.

────────────────────────────────────────────────────────────────────────────
THE ALLOWLIST IS DERIVED, BECAUSE A HAND-COPIED ONE IS THE DEFECT ONE LEVEL
UP
────────────────────────────────────────────────────────────────────────────

:func:`_shipped_roots` calls `tests/test_sdist_contents.py`'s own
:func:`~test_sdist_contents._allowlist`, which parses
`[tool.hatch.build.targets.sdist].include`. There is no second parser and no
second copy of the list: an allowlist entry added or removed moves this gate
on the same commit. :func:`~test_sdist_contents._assert_allowlist_is_plain`
is called for the same reason it exists there — the first-path-component
comparison below is faithful to hatchling's `GitIgnoreSpec` only while every
entry is a plain rooted name, and a glob must fail here rather than quietly
widen what this gate accepts.

**AND THE ALLOWLIST IS NOT THE WHOLE ANSWER, WHICH THAT COMMENT ALSO
DOCUMENTS.** Hatchling FORCE-INCLUDES a handful of paths past the allowlist
entirely: `pyproject.toml`, `hatch.toml`, `hatch_build.py`, the readme, the
`license-files` globs, and — located by walking UP from the build root —
`.gitignore` and `.hgignore`, the second of which can therefore be a file
from a PARENT of the checkout arriving as `stelling-0.1.0/.hgignore`. A
static `[tool.hatch.build] force-include` table is a third route.
:func:`~test_sdist_contents._force_included` models all of them and is
unioned in here rather than ignored.

**THE DIRECTION OF THE RESIDUAL ERROR MATTERS AND IS STATED IN BOTH
DIRECTIONS.**

* *Force-include only ADDS.* A route the model misses makes this gate call
  a path unshipped that in fact ships — a false RED, which is noisy and
  safe. The one route the model cannot see at all, a `hatch_build.py` hook
  adding entries at build time, is refused rather than modelled, by
  `test_sdist_contents.py::test_no_build_hook_can_force_include_behind_this
  _module`.
* *The allowlist ADMITTING a path does not mean the path SHIPS*, and this
  is the false GREEN, so it is the real residue. A gitignored file inside
  an allowlisted directory does not ship; hatchling DISCARDS its whole
  exclusion set when the checkout's own absolute path matches one of its
  patterns, which moves the answer by two members either way. A reference
  to such a path passes this gate and still dangles in the tarball.
  `tests/test_sdist_contents.py` holds the TREE side of that — what would
  ship, and what leaks — by intervention; nothing holds the REFERENCE side,
  and closing it needs a built sdist rather than a parse.

────────────────────────────────────────────────────────────────────────────
WHAT COUNTS AS A REFERENCE, AND WHAT THIS SCAN CANNOT SEE
────────────────────────────────────────────────────────────────────────────

A reference is a token of the shape ``root/component[/component…]``, with an
optional trailing slash, whose ROOT COMPONENT names a real entry at the root
of this checkout. Everything else is prose that happens to contain a slash:
``and/or``, ``jax/lax.py``, ``stelling-0.1.0/scratchpad/PREREG_SDIST.md``
(a tarball member, not a tree path — the lookbehind refuses a token that
begins after a slash).

**A BARE ``scratchpad/`` IS NOT A REFERENCE.** It is a name for the
directory, and the directory is what four shipped files exist to talk
ABOUT — `pyproject.toml`'s allowlist, `REUSE.toml`'s `scratchpad/**`
annotation, `.pre-commit-config.yaml`'s `insert-license` exclusion and
`.github/workflows/ci.yml`'s reuse job. Requiring one component after the
root is what separates the machinery that documents an exclusion from a
pointer into what was excluded; measured on this tree, that one rule keeps
`REUSE.toml`, `ci.yml`, `tests/_repo_files.py`, `tests/test_prose_hygiene.py`
and `tests/test_zero_dep_import_discipline.py` out with no exemption of
their own.

Three limits, stated because they are real:

* **A reference to a root this project has never had is invisible here.**
  The root component has to be a name :func:`_root_entries` knows — on
  disk, in the allowlist, in `WITHHELD` or in
  `GENERATED_IN_DISTRIBUTION` — so ``notes/plan.md`` is not seen at all
  when there is no ``notes/`` and never was. This gate answers *"is the
  path shipped"*, not *"does the path exist"*;
  `test_sdist_contents.py::test_every_readme_repo_link_resolves_to_a_real
  _path` answers the second question, for `README.md` only.
* **The sweep is `tests/_repo_files.text_files()`**, one walker shared with
  the rest of the suite rather than a second one to keep in step — so it
  covers `.md`, `.py`, `.toml`, `.cff`, `.yml`, `.yaml` and `.txt`, and does
  NOT cover the extensionless shipped files (`DCO`, `LICENSE`) or
  `.gitignore`.
* **A path hard-wrapped across a line is rejoined in Markdown and nowhere
  else.** `SOUNDNESS.md` wraps one (``scratchpad/`` then
  ``fuzz_transport.py``) and prose is the only place that happens; joining
  in YAML instead invented ``scratchpad/args`` out of
  `.pre-commit-config.yaml`'s ``exclude: ^scratchpad/`` and the ``args:``
  key on the next line, which is how the restriction was found.

────────────────────────────────────────────────────────────────────────────
WHERE THE DECLARATIONS LIVE, AND WHY NOT ALL IN ONE PLACE
────────────────────────────────────────────────────────────────────────────

`SOUNDNESS.md` declares its own, in the shipped file, in the section a
reader meets before the log — one row per path, saying what the instrument
is and what the distribution carries instead. That is where the declaration
belongs, because the reader who needs it is the reader of the artefact and
not the reader of this test. :func:`soundness_inventory` parses it and
:func:`partition` holds it to the page in BOTH directions, so it can neither
miss a citation nor keep a row for one the page has dropped.

Every other shipped file declares here, in :data:`DECLARED`, because there
is nowhere reader-facing to put it. Three kinds, and the difference between
them is the whole point of separating them:

``DISCLOSED``
    the file cites the path AND, in the same paragraph, does both halves of
    what `SOUNDNESS.md`'s inventory does per row: says the path is not
    distributed, and disposes of it — either by naming something the sdist
    DOES carry that re-derives the behaviour, or by calling the figure a
    *historical measurement* in those words. A denial on its own is not
    enough and was not the standard applied to the 32.
    `docs/harness-api.md` was already halfway there — *"Both are in the git
    checkout; neither is in the sdist"* — and now names `s13_scan` as well.
    **Neither half is taken on trust:**
    `test_a_DISCLOSED_citation_really_carries_its_disclosure` re-reads the
    file and requires both beside the citation, which is the same predicate
    `test_every_inventory_row_disposes_of_its_path` applies to a row.

``MACHINERY``
    the path is the SUBJECT of the sentence — the exclusion's own
    mechanism, or a plant in a build experiment. Nobody is invited to open
    it, and there is nothing to repair.

``OUTSTANDING``
    the file CITES it as evidence and does not dispose of it. This is the
    D4 defect, live. **EMPTY TODAY**, and empty is the target state rather
    than a dead kind: the eighteen references in ten files this list held
    when the gate was built have each been given the sentence at the site
    and moved to ``DISCLOSED``, where they are checked. The kind stays
    because the next citation to land without a disposal has to have
    somewhere honest to sit — better an ``OUTSTANDING`` entry with a reason
    than a hurried ``DISCLOSED`` the check would then be measuring against
    prose written to satisfy it.

`test_sdist_contents.py`'s `WITHHELD` comment is the precedent and the
warning: *"MEMBERSHIP HERE IS A RECORD, NOT AN ENFORCEMENT"*. A
``MACHINERY`` or ``OUTSTANDING`` declaration makes a reference visible and
accounted for; it does not make it right. ``DISCLOSED`` is the only kind
that is checked against the file's own text.

**AND THIS MODULE IS EXEMPT FROM ITS OWN SCAN, WHICH IS A HOLE, SO IT IS A
HOLE WITH A FLOOR IN IT.** Every path in :data:`DECLARED` appears in this
file by definition, so a registry that had to declare itself would be a
fixed point nobody could read. :data:`SELF` names the one exemption, and
`test_this_modules_own_references_are_only_what_it_declares` requires every
unshipped path named here to be one it declares somewhere or one of the
:data:`_PLANTS` the controls below drive — so a citation cannot be parked in
the exemption.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import NamedTuple

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _repo_files import read_text_files  # noqa: E402
from test_sdist_contents import (  # noqa: E402
    GENERATED_IN_DISTRIBUTION,
    WITHHELD,
    _allowlist,
    _assert_allowlist_is_plain,
    _force_included,
)

REPO = pathlib.Path(__file__).resolve().parent.parent

#: The heading of `SOUNDNESS.md`'s own inventory. Matched exactly: a renamed
#: heading is a broken anchor as well as a lost declaration, and the page
#: links to it by the slug this heading mints.
INVENTORY_HEADING = (
    "## The evidence this page cites from `scratchpad/`, which the "
    "distribution does not carry"
)

#: ``root/component[/component…]`` with an optional trailing slash. The
#: lookbehind refuses a token that begins immediately after a slash, a word
#: character or a dot, so a path inside a longer path is not re-read as a
#: repo-relative one.
_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.\-/])([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+/?)"
)


class Declaration(NamedTuple):
    """One shipped file's account of the unshipped paths it names.

    ``paths`` is EXACT, not a ceiling: a path named by the file and missing
    here reddens, and a path here the file no longer names reddens too. A
    ceiling would let the next citation in an already-declared file land
    unread, which is the whole shape this file exists against.
    """

    kind: str
    why: str
    paths: tuple[str, ...]


DISCLOSED = "DISCLOSED"
MACHINERY = "MACHINERY"
OUTSTANDING = "OUTSTANDING"

#: The sentence a ``DISCLOSED`` citation has to carry, in the paragraph the
#: citation sits in. Narrow on purpose: a pattern wide enough to match any
#: sentence mentioning the sdist would license the paragraph rather than the
#: disclosure. The paragraph is whitespace-collapsed before matching,
#: because hard-wrapped prose splits this sentence across two lines as often
#: as not — `docs/harness-api.md` wraps it between *"neither"* and *"is in
#: the sdist"*, and an uncollapsed match missed it.
_DISCLOSURE = re.compile(
    r"not (?:in the sdist|distributed)|neither is in the sdist",
    re.IGNORECASE,
)

#: The root entries that hold this repository's own CONTENT and do not
#: ship. Every other `WITHHELD` key names a cache (`.pytest_cache`,
#: `.mypy_cache`, `__pycache__`), a build output (`dist`, `build`,
#: `.pdm-build`), a local environment (`.venv`, `venv`, `.venv-prop`,
#: `.hypothesis`), local tool configuration (`.claude`, `.vscode`) or a
#: single file — and a path under one of THOSE is created by running
#: something rather than carried by this repository, so a shipped file
#: naming `.venv/bin/python` or `dist/stelling-0.1.0.tar.gz` is describing a
#: command's working directory and not pointing at a missing artefact
#: member. Measured: unioning all of `WITHHELD` in flags nine such paths
#: across `ci.yml`, `release.yml`, `nightly-jax-canary.yml`,
#: `tests/test_lanes.py`, `tests/test_release_gates.py`,
#: `tests/test_skip_inventory.py` and `tests/test_tripwire_record.py`, none
#: of which is this defect.
#:
#: `.git` is not here either, and that is the same rule rather than an
#: exception: `.git/info/exclude` names one of the four exclusion sources
#: git itself reads, in the two files whose subject is that hatchling reads
#: a different set. It is a git mechanism, not a path in this project's
#: tree, and both mentions are declared as MACHINERY below.
#:
#: Held to `WITHHELD` by
#: `test_the_content_roots_are_withheld_roots_and_not_a_second_list`, so
#: this cannot drift into naming something the allowlist decides.
_CONTENT_ROOTS = frozenset({"scratchpad"})

#: This module, exempt from its own scan. See the module docstring.
SELF = "tests/test_sdist_reference_hygiene.py"

#: The unshipped paths this module names that are NOT citations of
#: evidence: the plants the controls below drive, and the worked examples
#: the docstrings above quote. None of them exists in the tree, and none of
#: them sends a reader anywhere.
_PLANTS = (
    "scratchpad/probe/newthing.py",
    "scratchpad/nothing/cites/this.py",
    "scratchpad/anything.py",
    "scratchpad/args",
    # the truncated token `tools/solver_battery.py` used to hand this scan,
    # quoted in its declaration as the defect it was
    "scratchpad/D7-solver-",
)

DECLARED: dict[str, Declaration] = {
    "docs/harness-api.md": Declaration(
        DISCLOSED,
        "the 240-harness loop-carrier sweep and its results, cited for the "
        "80-of-200 figure and already carrying the disclosure at the site: "
        '"Both are in the git checkout; neither is in the sdist."',
        (
            "scratchpad/s13/RESULTS_loop_wide.txt",
            "scratchpad/s13/sweep_loop_assume_wide.py",
        ),
    ),
    "pyproject.toml": Declaration(
        MACHINERY,
        "the allowlist comment names `.git/info/exclude` to say what "
        "hatchling does NOT read — the sentence is about git's exclusion "
        "sources, and no reader is sent to open the file",
        (".git/info/exclude",),
    ),
    ".pre-commit-config.yaml": Declaration(
        MACHINERY,
        "`scratchpad/zz.py` is one row of a two-row lint control measured "
        "with reuse 6.2.0 (`scratchpad/zz.py` rc=0 against `src/zz.py` "
        "rc=1). Neither file exists in the tree or is meant to: they are "
        "the intervention that shows the `insert-license` exclusion drops "
        "exactly the paths `REUSE.toml` already covers",
        ("scratchpad/zz.py",),
    ),
    "tests/test_sdist_contents.py": Declaration(
        MACHINERY,
        "the module whose subject IS the exclusion. `.git/info/exclude` is "
        "one of the four sources `git status` reads and hatchling does not; "
        "`scratchpad/reach/.gitignore` is a real nested ignore file the "
        "subdirectory guard is measured against; and "
        "`scratchpad/PREREG_SDIST.md`, `scratchpad/PREREG*.md` and "
        "`scratchpad/zz_parity_note.md` are PLANTS in force-include and "
        "parity experiments, quoted with the tarball listing they produced",
        (
            ".git/info/exclude",
            "scratchpad/PREREG",
            "scratchpad/PREREG_SDIST.md",
            "scratchpad/reach/.gitignore",
            "scratchpad/zz_parity_note.md",
        ),
    ),
    # ------------------------------------------------------- disclosed
    #
    # THESE EIGHTEEN WERE `OUTSTANDING` WHEN THIS GATE LANDED, in ten files
    # D4's scope did not cover, and each is now a sentence at the site: the
    # path is not distributed, and the paragraph either names what ships in
    # its place or calls the figure a historical measurement. Every one of
    # them already stated its figures beside the citation — which is why a
    # sentence was enough and why none of them needed a judgement about
    # what the evidence shows. Where that had not been true the honest
    # answer would have been to leave it `OUTSTANDING` and say so.
    "design/ieee-reexamination.md": Declaration(
        DISCLOSED,
        "a build record citing the four probes behind its classification; "
        "the classification itself is the four numbered findings beside the "
        "citation, so the probes are a historical measurement",
        ("scratchpad/taskA_diagnose.py",),
    ),
    "design/solver-integration-build.md": Declaration(
        DISCLOSED,
        "a build record citing the main-agent acceptance script; what it "
        "checked is the 22/22 list beside the citation",
        ("scratchpad/acceptance_verify.py",),
    ),
    "design/transparent-primitives.md": Declaration(
        DISCLOSED,
        "a build record citing the sweep behind its per-series container "
        "table; the table beside the citation is that sweep's result",
        ("scratchpad/SERIES_CLAIM_SWEEP.md",),
    ),
    "src/stelling/_cvc5_driver.py": Declaration(
        DISCLOSED,
        "the module docstring cites the backstop probe for its "
        "nine-of-ten separator figure, and already named the shipped test "
        "that re-drives the property in the same breath",
        ("scratchpad/probe_cvc5_backstop.py",),
    ),
    "src/stelling/affine.py": Declaration(
        DISCLOSED,
        "a comment citing the per-obligation corpus for one of two halves "
        "of a measurement whose other half re-drives from the shipped tree",
        ("scratchpad/pin/corpus_pin.py",),
    ),
    "src/stelling/exactness.py": Declaration(
        DISCLOSED,
        "a docstring citing the per-obligation corpus for the second of two "
        "call-site censuses; the whole-suite one re-drives here",
        ("scratchpad/pin/corpus_pin.py",),
    ),
    "src/stelling/propagate.py": Declaration(
        DISCLOSED,
        "six citations of corpora and cost logs, in docstrings and comments "
        "beside the figures they produced — including the tree's ONLY "
        "mention of `scratchpad/cert/RESULTS_probe_index.txt`, which "
        "`SOUNDNESS.md`'s inventory therefore does not carry",
        (
            "scratchpad/cert/RESULTS_cap.txt",
            "scratchpad/cert/RESULTS_probe_index.txt",
            "scratchpad/claims/corpus_b3.py",
            "scratchpad/mechc",
            "scratchpad/pin/corpus_pin.py",
            "scratchpad/pin/f6_repro.py",
        ),
    ),
    "src/stelling/solvers.py": Declaration(
        DISCLOSED,
        "a comment citing the backstop probe's parts B/C/D; the arm that "
        "landed is this file, and the shipped audit suite holds it",
        ("scratchpad/probe_cvc5_backstop.py",),
    ),
    "tests/test_nonempty_certificate.py": Declaration(
        DISCLOSED,
        "three citations of the certificate build's oracle and result logs, "
        "in docstrings beside the figures they produced; this file is "
        "itself the shipped half of two of them",
        (
            "scratchpad/cert/RESULTS_cap.txt",
            "scratchpad/cert/RESULTS_invariant.txt",
            "scratchpad/cert/oracle.py",
        ),
    ),
    "tests/test_vacuous_refutation.py": Declaration(
        DISCLOSED,
        "two citations of the pre-registrations these tests were written "
        "against; the harnesses they scored are in this file",
        ("scratchpad/PREREG_MECHC.md", "scratchpad/PREREG_REF1.md"),
    ),
    # ------------------------------------------------------------ from D7
    #
    # D7 landed after this gate was written and its citations are the same
    # shape as the 32: evidence for a claim, in a page and a tool that both
    # ship, pointing into a directory that does not. They are disposed of
    # at the sites rather than exempted here — which is what the gate going
    # RED on the merge was for.
    "docs/choosing-a-solver-backend.md": Declaration(
        DISCLOSED,
        "the solver battery's transcript directory and its two "
        "reconstruction sweeps. What ships in their place is the reason "
        "`/tools` is in the allowlist at all: `tools/solver_battery.py` "
        "re-drives the battery on the reader's own machine, and "
        "`tests/test_solver_battery.py` holds every figure on the page to "
        "the transcript it is attributed to. The wall times are a "
        "historical measurement of two machines at stated loads, which is "
        "the page's own position about wall times",
        (
            "scratchpad/D7-solver-battery/",
            "scratchpad/D7-solver-battery/probe-does-the-freedom-reach-the-number.py",
            "scratchpad/D7-solver-battery/probe-where-does-the-box-stop.py",
        ),
    ),
    "tools/solver_battery.py": Declaration(
        DISCLOSED,
        "the `--rows` prose names the two sweeps behind its grades. The "
        "path used to be SPLIT ACROSS TWO STRING LITERALS, so the scan saw "
        "`scratchpad/D7-solver-` and a reader saw a whole path — it is one "
        "literal now, which is the only way the two agree",
        ("scratchpad/D7-solver-battery/",),
    ),
}

#: How many OUTSTANDING references stand, and in how many files. **Both are
#: ZERO**, and zero is a claim this file makes rather than a place where a
#: list happens to be empty: `DROPPED` in `tests/_soundness_routing_manifest.py`
#: is the precedent — an empty tuple whose emptiness means something because
#: a partition is what makes it checkable.
#:
#: They were 18 and 10 when this gate landed, and every one of those was a
#: real instance of the defect, left standing because D4's scope was
#: `SOUNDNESS.md`, `docs/harness-api.md` and this gate. They are disposed of
#: at their sites now. The numbers are still pinned, and going UP means a
#: shipped file has gained a dangling pointer that somebody chose to record
#: rather than repair — legitimate, but never silent.
OUTSTANDING_REFERENCES = 0
OUTSTANDING_FILES = 0

#: The SIXTH numeral the inventory states, and the only one that is a fact
#: about the repository rather than about the page. Held by a git-gated test
#: of its own below, because a skip raised inside a test skips the whole
#: test and the other five must be measured wherever this file runs.
_TRACKED_CLAIM = r"`git ls-files scratchpad` is \*\*(\d+)\*\* files"

#: The other five, each recomputed from the page by
#: `test_the_inventorys_own_numerals_are_derived_from_the_page`. A numeral
#: standing beside the data that derives it is the shape `docs/norms.md`
#: legislates against, so none of them is trusted here.
#:
#: All six are matched against a WHITESPACE-COLLAPSED section, never against
#: the raw lines. The page is hard-wrapped at 76 columns, so a numeral sits
#: wherever the wrap puts it — and a `\n?` sprinkled through each pattern
#: only makes the brittleness harder to see. Driven: re-flowing one
#: paragraph, changing not a word, moved `**20** of the **32** have a
#: shipped` across a line break and gave `the inventory no longer states its
#: 'shipped' numeral`, which is a check reporting a deletion that had not
#: happened.
_CLAIMS = (
    (r"\*\*(\d+)\*\* citations,", "citations"),
    (r"naming \*\*(\d+)\*\* distinct paths", "distinct"),
    (r"which is \*\*(\d+)\*\* occurrences of the string", "occurrences"),
    (r"\*\*(\d+)\*\* of the \*\*(\d+)\*\* have a shipped", "shipped"),
    (r"the other \*\*(\d+)\*\* are historical", "historical"),
)


# ------------------------------------------------------------------ reading


def _shipped_roots() -> set[str]:
    """Every first path component the sdist carries.

    The allowlist, parsed by `test_sdist_contents.py` and not re-parsed
    here, unioned with the force-include model. See this module's docstring
    for which direction each source's error runs in.
    """
    allow = _allowlist()
    _assert_allowlist_is_plain(allow)
    forced = {name.split("/", 1)[0] for name in _force_included(REPO)}
    return allow | forced


def _root_entries() -> set[str]:
    """Every name that counts as a root entry of this project, present or not.

    **NOT `REPO.iterdir()` ALONE, AND THE REASON IS AN ENVIRONMENT THIS
    SUITE IS MEANT TO RUN IN.** An unpacked sdist has no `scratchpad/`, so
    a scan keyed on what is on disk stops recognising
    `scratchpad/pin/corpus_pin.py` as a repo path in exactly the artefact
    where it dangles. Driven, in a `.git`-less copy of this tree:
    `1 failed, 13 passed, 1 skipped`. So the allowlist and
    :data:`_CONTENT_ROOTS` are unioned in, and the answer no longer depends
    on which directories happen to exist.

    `GENERATED_IN_DISTRIBUTION` is deliberately NOT unioned in: `PKG-INFO`
    is a file, exists only in a tarball, and nothing can sit under it.
    """
    return {path.name for path in REPO.iterdir()} | _allowlist() | _CONTENT_ROOTS


def references(text: str, *, markdown: bool) -> list[str]:
    """Every repo-relative path token in `text`, in order, duplicates kept.

    `markdown` rejoins a path hard-wrapped across a line. See the docstring
    for why that is Markdown-only.
    """
    if markdown:
        text = re.sub(r"/\n\s*", "/", text)
    roots = _root_entries()
    return [
        token
        for token in _TOKEN.findall(text)
        if token.split("/", 1)[0] in roots
    ]


def dangling(text: str, *, markdown: bool, shipped: set[str]) -> list[str]:
    """The references in `text` naming a path the sdist does not carry."""
    return [
        token
        for token in references(text, markdown=markdown)
        if token.split("/", 1)[0] not in shipped
    ]


def _shipped_text_files() -> list[tuple[str, str]]:
    """`(relative path, text)` for every shipped file the walker reaches."""
    shipped = _shipped_roots()
    return [
        (rel, text)
        for rel, text in read_text_files()
        if rel.split("/", 1)[0] in shipped
    ]


def _dangling_by_file(*, include_self: bool = False) -> dict[str, set[str]]:
    shipped = _shipped_roots()
    out: dict[str, set[str]] = {}
    for rel, text in _shipped_text_files():
        if rel == SELF and not include_self:
            continue
        found = set(dangling(text, markdown=rel.endswith(".md"), shipped=shipped))
        if found:
            out[rel] = found
    return out


def _paragraphs(text: str) -> list[str]:
    """Maximal runs of non-blank lines, which is what "beside it" means for
    a check that can only read position."""
    out, run = [], []
    for line in text.split("\n"):
        if line.strip():
            run.append(line)
        elif run:
            out.append("\n".join(run))
            run = []
    if run:
        out.append("\n".join(run))
    return out


def _flow(para: str) -> str:
    """One paragraph as a single line, the way a reader reads it.

    **THE COMMENT MARKER HAS TO GO FIRST.** Prose is hard-wrapped, so the
    sentence a citation carries is regularly split across two lines — and in
    a `.py` comment the second line begins `# `, so a plain whitespace
    collapse yields *"is not in the # sdist"* and the disclosure matches
    nothing. Measured: `src/stelling/affine.py` and
    `tests/test_vacuous_refutation.py` both carried the sentence and both
    were reported as carrying none.
    """
    stripped = [re.sub(r"^\s*#\s?", "", line) for line in para.split("\n")]
    return re.sub(r"\s+", " ", " ".join(stripped)).strip()


def _collapsed(section: list[str]) -> str:
    """The section as one whitespace-collapsed string, for numeral matching.

    See :data:`_CLAIMS` for why nothing here is matched against raw lines.
    """
    return re.sub(r"\s+", " ", "\n".join(section))


def _soundness_section() -> list[str]:
    lines = (REPO / "SOUNDNESS.md").read_text(encoding="utf-8").split("\n")
    assert INVENTORY_HEADING in lines, (
        "SOUNDNESS.md no longer carries the inventory heading this gate "
        f"reads, and the page links to its anchor:\n  {INVENTORY_HEADING}"
    )
    start = lines.index(INVENTORY_HEADING)
    stop = next(
        i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")
    )
    return lines[start:stop]


def soundness_inventory(section: list[str] | None = None) -> list[list[str]]:
    """The inventory's rows as `[path, what it is, what ships]`.

    `section` is a parameter so the partition check below can be driven
    against a mutated inventory without editing the page — a test that
    rewrites the file it reads can leave the tree changed.
    """
    if section is None:
        section = _soundness_section()
    rows = []
    for line in section:
        if not line.startswith("| `scratchpad"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == 3, f"inventory row is not three cells: {line}"
        rows.append([cells[0].strip("`"), cells[1], cells[2]])
    return rows


def soundness_citations(section: list[str] | None = None) -> list[str]:
    """Every `scratchpad/…` reference on the page OUTSIDE the inventory."""
    lines = (REPO / "SOUNDNESS.md").read_text(encoding="utf-8").split("\n")
    start = lines.index(INVENTORY_HEADING)
    stop = next(
        i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")
    )
    outside = "\n".join(lines[:start] + lines[stop:])
    return [
        token
        for token in references(outside, markdown=True)
        if token.split("/", 1)[0] == "scratchpad"
    ]


def partition(cited: set[str], declared: set[str]) -> tuple[set[str], set[str]]:
    """`(cited and not declared, declared and not cited)`."""
    return cited - declared, declared - cited


#: A backticked path, with NO closing backtick required. `` ``a/b.py::c`` ``
#: is how this tree cites a test in RST, and a pattern demanding the closing
#: mark reads the `::name` as part of the token and matches nothing —
#: measured, on `src/stelling/_cvc5_driver.py`, which names a shipped test
#: right beside the citation and was reported as naming none.
_BACKTICKED_PATH = re.compile(r"`+([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+)")


def _names_a_shipped_path(text: str) -> bool:
    """Does `text` name, in backticks, a path the sdist carries?"""
    shipped = _shipped_roots()
    return any(
        token.split("/", 1)[0] in shipped
        for token in _BACKTICKED_PATH.findall(text)
    )


def _disposes(text: str) -> bool:
    """The second half of a disposal, shared by the inventory rows and the
    ``DISCLOSED`` declarations so the two cannot drift into two standards:
    name something that ships, or say *historical measurement*."""
    return _names_a_shipped_path(text) or "historical measurement" in text


def _row_has_a_shipped_answer(row: list[str]) -> bool:
    """Does the row's third cell name a path the sdist carries?"""
    return _names_a_shipped_path(row[2])


# -------------------------------------------------------------------- gates


def test_the_allowlist_is_derived_and_the_force_include_hole_is_covered():
    """The shipped set comes from `pyproject.toml`, through the parser that
    was already there. Break it: add a glob to the allowlist.

    Four ways this could be vacuous or wrong, and one of each: an allowlist
    that parsed to nothing, a force-include model that has gone empty and
    stopped covering the hole the allowlist cannot reach, a shipped set that
    has swallowed every root entry so the gate can never fire, and
    `/scratchpad` having been allowlisted — which is the one repair this
    defect must not be given.
    """
    allow = _allowlist()
    assert len(allow) >= 20, f"the allowlist parsed to {len(allow)} entries"
    forced = {name.split("/", 1)[0] for name in _force_included(REPO)}
    assert forced, (
        "the force-include model is empty, so this gate is testing the "
        "allowlist as though it were exhaustive — which the allowlist's own "
        "comment says it is not"
    )
    shipped = _shipped_roots()
    assert forced <= shipped, (
        "the force-include model names a root the shipped set does not "
        "carry, so the union below is not the union it claims to be"
    )
    missing = _root_entries() - shipped
    assert missing, (
        "every root entry is shipped, so this gate can never fire. That is "
        "either a real change to the allowlist or a broken derivation"
    )
    assert "scratchpad" in missing, (
        "`/scratchpad` is in the sdist allowlist. It must not be: the "
        "allowlist exists because shipping working files leaked an internal "
        "release checklist inside `stelling-0.1.0.tar.gz`, and an sdist on "
        "PyPI cannot be unpublished"
    )
    # `SOUNDNESS.md`'s inventory says `/scratchpad` appears ZERO times in
    # this file, which is a stronger claim than "not allowlisted" and is
    # the one a reader checks first.
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "scratchpad" not in pyproject, (
        "`pyproject.toml` now names `scratchpad`, and SOUNDNESS.md's "
        "inventory says it appears there zero times"
    )


def test_the_content_roots_are_withheld_roots_and_not_a_second_list():
    """:data:`_CONTENT_ROOTS` names root entries this repository HAS and does
    not ship, so every one of them must be a `WITHHELD` key — that dict is
    where "exists here, is not distributed" is decided, and a name here that
    is not there would be a second, unheld list.

    Break it: add a name to `_CONTENT_ROOTS` that `WITHHELD` does not carry.
    """
    assert _CONTENT_ROOTS, "no content root — the scan cannot see any"
    stray = sorted(_CONTENT_ROOTS - set(WITHHELD))
    assert not stray, (
        "_CONTENT_ROOTS names entries `test_sdist_contents.WITHHELD` does "
        "not:\n  " + "\n  ".join(stray)
    )
    assert not (_CONTENT_ROOTS & _allowlist()), (
        "_CONTENT_ROOTS names something the allowlist ships, which makes it "
        "shipped and not a content root at all"
    )
    assert "PKG-INFO" in GENERATED_IN_DISTRIBUTION, (
        "the exemption this module states for GENERATED_IN_DISTRIBUTION — "
        "that its one member is a file nothing can sit under — no longer "
        "describes that dict"
    )


def test_no_shipped_file_names_a_path_the_sdist_does_not_carry():
    """The gate. Every reference in a shipped file to a path outside the
    sdist must be declared — by `SOUNDNESS.md`'s own inventory, or in
    :data:`DECLARED` with a reason and a kind.

    Break it: write `scratchpad/anything.py` into any shipped file.
    """
    found = _dangling_by_file()
    assert found, "no shipped file names an unshipped path — check is vacuous"

    undeclared: list[str] = []
    for rel, tokens in sorted(found.items()):
        if rel == "SOUNDNESS.md":
            declared = {row[0] for row in soundness_inventory()}
        else:
            entry = DECLARED.get(rel)
            declared = set(entry.paths) if entry else set()
        for token in sorted(tokens - declared):
            undeclared.append(f"{rel}: {token}")
    assert not undeclared, (
        "shipped file(s) point at paths the sdist does not carry, with "
        "nothing saying so:\n  " + "\n  ".join(undeclared) + "\n\n"
        "A reader of the tarball follows one of these and finds nothing. "
        "Either say at the site that the path is not distributed (the form "
        "`docs/harness-api.md` already uses), or re-derive the evidence "
        "from something that ships. Adding `/scratchpad` to the allowlist "
        "is NOT the fix — see this module's docstring."
    )


def test_every_declaration_is_still_a_reference_its_file_makes():
    """The other half of the partition: a declaration for a reference that
    is gone is a stale record, and a stale record is how a list stops
    meaning anything.

    Read off the FILE'S OWN TEXT and not off the scan, so the answer is the
    same in a checkout and in an unpacked sdist. The scan cannot see a
    `.git/…` reference where there is no `.git`, and a staleness check that
    inherited that would call two true declarations stale in the artefact —
    which is what it did, driven: `1 failed, 13 passed, 1 skipped` in a
    `.git`-less copy of this tree.

    Break it: delete a `scratchpad/` citation from a declared file without
    deleting its row here.
    """
    stale = []
    for rel, entry in sorted(DECLARED.items()):
        path = REPO / rel
        assert path.is_file(), f"DECLARED names a file that is not here: {rel}"
        text = path.read_text(encoding="utf-8")
        if rel.endswith(".md"):
            text = re.sub(r"/\n\s*", "/", text)
        stale += [f"{rel}: {tok}" for tok in entry.paths if tok not in text]
    assert not stale, (
        "DECLARED names references these files no longer make:\n  "
        + "\n  ".join(stale)
    )


def test_a_DISCLOSED_citation_really_carries_its_disclosure():
    """The one kind that is a claim about the file's own text, checked
    against the file's own text.

    Break it: delete *"neither is in the sdist"* from `docs/harness-api.md`
    and leave the declaration standing.
    """
    live = {rel: e for rel, e in DECLARED.items() if e.kind == DISCLOSED}
    assert live, "no DISCLOSED declaration — the kind is dead text"
    bare = []
    for rel, entry in sorted(live.items()):
        text = (REPO / rel).read_text(encoding="utf-8")
        paragraphs = [
            _flow(para)
            for para in _paragraphs(
                re.sub(r"/\n\s*", "/", text) if rel.endswith(".md") else text
            )
        ]
        for path in entry.paths:
            carrying = [p for p in paragraphs if path in p]
            assert carrying, f"{rel} no longer names {path}"
            for para in carrying:
                if not _DISCLOSURE.search(para):
                    bare.append(f"{rel}: {path} — says nothing about the "
                                f"path being undistributed")
                elif not _disposes(para):
                    bare.append(f"{rel}: {path} — denies but does not "
                                f"dispose: names nothing the sdist carries "
                                f"and is not called a historical measurement")
    assert not bare, (
        "citation(s) declared DISCLOSED that do not meet the standard the 32 "
        "were held to:\n  " + "\n  ".join(bare) + "\n\n"
        "A DISCLOSED paragraph does both halves — says the path is not "
        "distributed, AND either names a shipped path that re-derives the "
        "behaviour or calls the figure a historical measurement in those "
        "words. Write the second half beside the citation, or move the file "
        "to OUTSTANDING, which is what it actually is."
    )


def test_this_modules_own_references_are_only_what_it_declares():
    """The floor under this module's exemption from its own scan.

    Break it: cite a `scratchpad/` path here that nothing declares.
    """
    found = _dangling_by_file(include_self=True).get(SELF, set())
    assert found, f"{SELF} names no unshipped path — the floor is vacuous"
    accounted = (
        {path for entry in DECLARED.values() for path in entry.paths}
        | {row[0] for row in soundness_inventory()}
        | set(_PLANTS)
    )
    stray = sorted(found - accounted)
    assert not stray, (
        f"{SELF} is exempt from the scan and names paths it declares "
        "nowhere:\n  " + "\n  ".join(stray)
    )


@pytest.mark.parametrize("kind", (DISCLOSED, MACHINERY, OUTSTANDING))
def test_every_declaration_carries_a_kind_and_a_reason(kind):
    """A declaration with no reason is a whitelist entry.

    ``DISCLOSED`` and ``MACHINERY`` must be non-empty or the kind has gone
    dead; ``OUTSTANDING`` is EMPTY by design and is held to that by
    `test_the_outstanding_list_is_a_record_that_has_to_shrink` instead —
    asserting it is non-empty here would make emptying it a failure, which
    is backwards.
    """
    entries = [e for e in DECLARED.values() if e.kind == kind]
    if kind != OUTSTANDING:
        assert entries, f"no {kind} declaration at all — the kind is dead text"
    for entry in entries:
        assert len(entry.why) > 40, f"{kind} declaration with a stub reason"
        assert entry.paths, f"{kind} declaration with no paths"


def test_the_outstanding_list_is_a_record_that_has_to_shrink():
    """What is declared and NOT repaired, counted.

    These are live instances of the same defect `SOUNDNESS.md` was repaired
    for, standing in files outside the batch that built this gate. The two
    numerals are pinned so the list cannot grow unnoticed; they are not a
    budget, and repairing a reference should move them DOWN and require an
    edit here saying so.
    """
    live = {rel: e for rel, e in DECLARED.items() if e.kind == OUTSTANDING}
    references_ = sum(len(e.paths) for e in live.values())
    assert (references_, len(live)) == (
        OUTSTANDING_REFERENCES,
        OUTSTANDING_FILES,
    ), (
        f"{references_} outstanding references in {len(live)} files, against "
        f"a declared {OUTSTANDING_REFERENCES} in {OUTSTANDING_FILES}. If the "
        f"list shrank, lower the numerals. If it GREW, a shipped file has "
        f"gained a dangling pointer and the fix is not to raise them."
    )
    for rel, entry in sorted(live.items()):
        assert "not repaired" in entry.why or "judgement" in entry.why, (
            f"{rel} is declared OUTSTANDING without saying WHY it was not "
            f"repaired. An honest OUTSTANDING names what the repair would "
            f"need — a judgement about what the evidence shows, a file "
            f"another batch owns — and a hurried DISCLOSED is the only "
            f"thing worse than it."
        )


def test_soundness_inventories_every_scratchpad_path_it_cites():
    """`SOUNDNESS.md`'s inventory against `SOUNDNESS.md`'s citations, both
    ways. Break it either way: cite a new path, or delete a row."""
    rows = soundness_inventory()
    assert len(rows) >= 30, f"the inventory parsed to {len(rows)} rows"
    cited = set(soundness_citations())
    assert cited, "no `scratchpad/` citation found — the check is vacuous"
    missing, stale = partition(cited, {row[0] for row in rows})
    assert not missing, (
        "SOUNDNESS.md cites paths its own inventory does not carry:\n  "
        + "\n  ".join(sorted(missing))
    )
    assert not stale, (
        "SOUNDNESS.md's inventory carries rows for paths the page no longer "
        "cites:\n  " + "\n  ".join(sorted(stale))
    )


def test_the_page_links_to_the_inventory_by_the_anchor_its_heading_mints():
    """The heading is pinned by :data:`INVENTORY_HEADING`, so a rename
    reddens — but renaming BOTH leaves the in-page link dangling and every
    check green, which is a hole exactly the width of a rename.

    The anchor is computed with `test_soundness_routing.py`'s `slug`, the
    rule `test_the_anchor_rule_is_the_one_this_repo_already_links_by` pins
    against three anchors that predate it, so this is not a second guess at
    what GitHub does.

    Break it: rename the heading and the constant together.
    """
    from test_soundness_routing import slug

    anchor = slug(INVENTORY_HEADING[3:])
    text = (REPO / "SOUNDNESS.md").read_text(encoding="utf-8")
    assert f"](#{anchor})" in text, (
        f"SOUNDNESS.md carries no link to its own inventory: expected "
        f"`](#{anchor})`, from the heading\n  {INVENTORY_HEADING}"
    )


def test_every_inventory_row_disposes_of_its_path():
    """A row is a disposal, not a listing. Each says what the instrument is
    and either names a shipped thing that re-derives the behaviour or calls
    itself a historical measurement in those words."""
    bad = []
    for row in soundness_inventory():
        if len(row[1]) < 20:
            bad.append(f"{row[0]}: no account of what it is")
        elif not (
            _row_has_a_shipped_answer(row)
            or "historical measurement" in row[2]
        ):
            bad.append(f"{row[0]}: names nothing shipped and is not marked "
                       f"a historical measurement")
    assert not bad, "inventory row(s) that dispose of nothing:\n  " + "\n  ".join(bad)


def test_the_inventorys_own_numerals_are_derived_from_the_page():
    """Five of the six numerals in the inventory's prose, recomputed here.

    The sixth needs git and has a test of its own; see :data:`_TRACKED_CLAIM`.

    Break it: add a citation, or a row, without touching the prose.
    """
    section = _collapsed(_soundness_section())
    rows = soundness_inventory()
    cited = soundness_citations()
    lines = (REPO / "SOUNDNESS.md").read_text(encoding="utf-8").split("\n")
    start = lines.index(INVENTORY_HEADING)
    stop = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    outside = "\n".join(lines[:start] + lines[stop:])

    stated = {}
    for pattern, name in _CLAIMS:
        match = re.search(pattern, section)
        assert match, f"the inventory no longer states its {name!r} numeral"
        stated[name] = [int(g) for g in match.groups()]

    shipped_rows = sum(1 for row in rows if _row_has_a_shipped_answer(row))
    expected = {
        "citations": [len(cited)],
        "distinct": [len(set(cited))],
        "occurrences": [len(re.findall(r"scratchpad/", outside))],
        "shipped": [shipped_rows, len(rows)],
        "historical": [len(rows) - shipped_rows],
    }
    assert stated == expected, (
        "SOUNDNESS.md's inventory states numerals the page no longer "
        f"supports.\n  stated:   {stated}\n  measured: {expected}"
    )


def test_the_inventorys_tracked_file_count_is_derived_from_git():
    """The sixth numeral, split out because it is the ONLY one that needs
    git and a skip raised inside a test skips the whole test.

    `tests/test_soundness_routing.py` paid for that lesson — one git-gated
    leg took seven git-less mutation controls into the skip with it — so the
    five numerals above are measured wherever this file runs and this one is
    on its own. An unpacked sdist has no `.git` AND no `scratchpad/`, so
    there is nothing there to count either way.

    Break it: change the numeral, or add a file under `scratchpad/`.
    """
    section = _collapsed(_soundness_section())
    match = re.search(_TRACKED_CLAIM, section)
    assert match, "the inventory no longer states its tracked-file numeral"
    # the numeral's PRESENCE is checked above, without git, so a deleted
    # claim reddens in an sdist too; only the comparison needs the skip
    proc = _git_ls_files_scratchpad()
    files = [line for line in proc.stdout.split("\n") if line.strip()]
    tracked = len(files)
    assert int(match.group(1)) == tracked, (
        f"SOUNDNESS.md says `git ls-files scratchpad` is {match.group(1)} "
        f"files; it is {tracked}. This numeral has rotted once already — it "
        f"read 80 against a tree holding 135."
    )

    # and every DIRECTORY row that states a file count, by the same rule:
    # `— N files` in the middle cell is a claim about the tree, and a row
    # is the only place this page is allowed to make one.
    wrong = []
    for row in soundness_inventory():
        if not row[0].endswith("/"):
            continue
        stated = re.search(r"— (\d+) files", row[1])
        if stated is None:
            continue
        under = len([f for f in files if f.startswith(row[0])])
        if int(stated.group(1)) != under:
            wrong.append(f"{row[0]}: says {stated.group(1)}, git says {under}")
    assert not wrong, (
        "inventory row(s) stating a file count the tree does not have:\n  "
        + "\n  ".join(wrong)
    )
    assert any(
        re.search(r"— \d+ files", row[1])
        for row in soundness_inventory() if row[0].endswith("/")
    ), "no directory row states a file count — that leg is vacuous"

    # and the inventory's "all 32 are present in this checkout": every row
    # names something `git ls-files scratchpad` actually lists, as a file
    # or as the prefix of one.
    absent = [
        row[0] for row in soundness_inventory()
        if not any(f == row[0] or f.startswith(row[0]) for f in files)
    ]
    assert not absent, (
        "SOUNDNESS.md's inventory says every cited path is present in this "
        "checkout; git lists none of these:\n  " + "\n  ".join(absent)
    )


def _git_can_read_this_tree() -> bool:
    """`.git` is a directory in a plain checkout and a FILE in a linked
    worktree, which is what this batch was built in — `.exists()` and not
    `.is_dir()`, and the same predicate `tests/test_skip_inventory.py`'s
    rule for this reason uses."""
    return (REPO / ".git").exists()


def _git_ls_files_scratchpad():
    """`git ls-files scratchpad`, skipping only where `.git` is absent.

    The same shape and the same reason string as
    `tests/test_reuse_pins.py`, so `tests/test_skip_inventory.py`'s existing
    rule discloses this skip and no second disclosure has to be kept in
    step. A `.git` that is present and a git that still fails is a DEFECT
    here and not an environment, so it raises rather than skipping.
    """
    import subprocess

    if not _git_can_read_this_tree():
        pytest.skip("not a git checkout (an unpacked sdist, say)")
    proc = subprocess.run(
        ["git", "ls-files", "scratchpad"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0 and proc.stdout.strip(), (
        f"`.git` is present and `git ls-files scratchpad` failed "
        f"(rc={proc.returncode}, stderr={proc.stderr.strip()!r})"
    )
    return proc


# ------------------------------------------------------------------ controls


def test_the_gate_bites():
    """The two plants, driven through the same functions the gate uses.

    A pin nobody has watched fail is not a pin, and the shape this one has
    to refuse is precise: an unshipped path reddens, a SHIPPED one must not,
    or the gate is just a scan for slashes.
    """
    shipped = _shipped_roots()
    prose = "the figures come from {path}, driven for each.\n"

    unshipped = prose.format(path="`scratchpad/probe/newthing.py`")
    assert dangling(unshipped, markdown=True, shipped=shipped) == [
        "scratchpad/probe/newthing.py"
    ], "a reference to an unshipped path did not redden"

    ok = prose.format(path="`tests/test_sdist_contents.py`")
    assert dangling(ok, markdown=True, shipped=shipped) == [], (
        "a reference to a SHIPPED path reddened"
    )

    # and the two shapes that must stay out of the scan entirely
    assert dangling("`scratchpad/`", markdown=True, shipped=shipped) == [], (
        "a bare `scratchpad/` was read as a pointer into the directory"
    )
    assert dangling(
        "stelling-0.1.0/scratchpad/PREREG_SDIST.md", markdown=True,
        shipped=shipped,
    ) == [], "a tarball member name was read as a repo-relative path"


def test_the_soundness_partition_bites():
    """Both directions of the inventory partition, driven on a mutated copy
    of the section rather than on the file."""
    section = _soundness_section()
    rows = soundness_inventory(section)
    assert len(rows) >= 30

    without = [
        line for line in section
        if not line.startswith("| `" + rows[0][0] + "`")
    ]
    assert len(soundness_inventory(without)) == len(rows) - 1
    missing, stale = partition(
        set(soundness_citations()), {r[0] for r in soundness_inventory(without)}
    )
    assert missing == {rows[0][0]}, (
        "deleting an inventory row left the partition green"
    )
    assert not stale

    invented = section + ["| `scratchpad/nothing/cites/this.py` | x | y |"]
    missing, stale = partition(
        set(soundness_citations()), {r[0] for r in soundness_inventory(invented)}
    )
    assert stale == {"scratchpad/nothing/cites/this.py"}, (
        "a row for a path the page does not cite left the partition green"
    )
    assert not missing
