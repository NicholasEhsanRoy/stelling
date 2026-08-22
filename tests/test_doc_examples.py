# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Documented examples run, and their recorded output is compared.

The can't-drift rule applied to the docs' code: an example that no longer
runs, or whose printed output has silently moved, is worse than no
example — it is a claim about the tool that the tool contradicts.

Its own history is the argument for it. The user-facing docs were written
and hand-verified against one commit; three merges later, two recorded
outputs were stale (the decline/undecided message passes had lengthened
an obligation detail) and nothing in the suite noticed, because nothing
in the suite read a fenced block.

**EXACTLY WHAT IS AND IS NOT CHECKED.** Stated with numbers because the
first version of this file claimed "every documented example runs, and
prints what the doc says it prints", which was wider than the mechanism.
Measured over ``README.md`` + ``docs/*.md``, and pinned by
:func:`test_inventory_is_what_the_docstring_says`::

    ```python blocks                                55
      marked illustrative — not run                 14
      EXECUTED (exit 0 required)                    41
        marked run-only — output not compared        3
        OUTPUT COMPARED against a fence             38
    plain ``` fences                                81
      consumed as an example's claimed output       38
      HAND-WRITTEN, compared to nothing             43
So the claim this file earns is: *every runnable example runs, and 38 of
the 41 have their stdout compared byte for byte after a narrow
normalisation.* The 43 unattached fences — a render pasted into prose, a
quoted stamp line, an excerpt from another page's table — are **not**
verified here. Writing one of those is a hand-check and stays one.

**Every figure in that table and in the sentence after it is read back
OUT of this docstring** by :func:`_docstring_inventory` and compared to
``EXPECTED_INVENTORY``. Until it was, the table was the honour system:
the dict was checked against the docs and the docstring against nothing,
so the table could be set to any digits at all with the suite green — in
the file that two paragraphs below declares a documented count should be
written by the tree and not by an author.

**A second job this file now does, and it is not an example.** One of the
compared blocks is a REGISTRY CENSUS in ``docs/state-0.1.0.md`` — it
teaches nobody how to use the tool; it exists so that a *number in prose*
about what is in ``TRANSFERS``/``_SUPPORTED`` cannot drift from the
registries it describes. That page's figure had already been wrong twice,
once against the very sha it was stamped with, and stamping a sha does not
help when a human still types the digit. So the rule this file makes
enforceable, and enforces wherever a count is written as a block, is:
**a documented count that is computable from this tree should be written by
the tree, not by an author.** Counts over populations that are NOT in this
tree (the two blinded external contracts, the ``jax_md`` census) cannot be
gated here and carry an as-of-sha instead.

**Markers.** Default-deny: an unmarked ```python block is executed and
must be followed by a fence carrying its output. A block that is not
that says which it is, on the line before its fence::

    <!-- doc-example: illustrative -->   a record, not an instruction: never run
    <!-- doc-example: run-only -->       runs, but has no fenced output

Requiring the second marker closes the hole where deleting an example's
output fence silently downgraded it to run-only with the suite green: it
now fails until someone says, in the file, that the downgrade was meant.

**Normalisation, and its limits.** Three things in a real verdict are
environment-dependent and are neutralised on BOTH sides before
comparison:

* the *directory and file name* of an ``at <path>:<line> (<fn>)``
  source-info line — the runner's temp file is not the doc's file name.
  **The line number and the function name are compared**, because the
  block that produced them is byte-identical to the one in the doc;
* solver wall times (``answered sat in 8ms``);
* solver versions (``z3 5.0.0`` -> ``z3 <version>``) — a real limit: the
  version *digits* a doc prints are not verified, only their shape.

Everything else — statuses, stamp fields, coverage lines, query content
hashes, refusal text — is compared byte for byte, and
:func:`test_normalisation_is_narrow` pins that the normaliser touches
nothing else.

**Two further neutralisations, and they are NOT in the normaliser and NOT
one condition.** A verdict stamps the jax version that produced it, and a
query content hash is a function of that version (measured: jax 0.11
replaced ``jit``'s boolean ``inline`` param with an ``Inline`` enum, and
that one param is the entire difference between the 0.10.2 and 0.11.0
renders of ``docs/quickstart.md``'s block). A doc therefore cannot be a
byte-exact transcript of two jax series at once. So
:func:`test_doc_example` neutralises

* **the stamp line** whenever the running jax differs from the doc's AT
  ALL, exact version to exact version, because that line prints the
  running version and there is nothing there to compare; and
* **the** ``query <hash>`` **line** only when the running jax's SERIES
  differs from the doc's — so within the series a doc names, its hash is
  compared byte for byte, including across a patch release.

Every other line stays compared, so an off-series lane still measures
that the verdict is otherwise identical. The limit that buys: **on a
series other than the one a doc names, that doc's query hash is not
verified here** — and that limit is now DECLARED rather than inferred, in
``EXPECTED_HASH_COVERAGE``, whose right-hand side
:func:`test_every_documented_query_hash_is_compared_on_some_lane`
recomputes from the same predicate the comparison uses and which fails on
any hash checked by no lane at all.

**Those were one condition until 3482822, and the conflation cost the
whole coverage of one hash.** The single condition was the exact-version
one, so the day ``test-jax`` floated past the version the doc stamps,
nothing compared that hash anywhere. See the block comment above
``_STAMP_JAX`` for the mutation that measures it.

Blocks that opt into a solver (they pass ``solver_timeout_ms``) are
skipped when no backend is installed, since their output is about an
escalation that cannot happen. Those examples are unverified in a
solver-free environment.

**And EVERY example here is skipped when jax is absent** —
:func:`test_doc_example` opens with ``pytest.importorskip("jax")``, so in
the zero-dep configuration this module verifies nothing about any
documented output. That matters most for the one block that needs no jax:
the registry census above counts a ZERO-DEP-CORE registry, and it was the
block skipped in the zero-dep run. Its jax-free gate is
``tests/test_release_doc_claims.py``, which also re-derives the prose
around it; this module's byte-for-byte comparison is the with-jax half.

The subprocess runs with ``PYTHONPATH`` pointing at **this repo's**
``src/``. Without that a developer with stelling installed from
elsewhere would measure a different tree than the one they are editing,
which is the exact failure this file exists to prevent.
"""

from __future__ import annotations

import difflib
import os
import pathlib
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest

import _lanes
from stelling._optional import TESTED_JAX_SERIES, available, jax_series_tested

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"

# The inventory the docstring states. A change here is a change to what
# this file promises, so it must be made deliberately and in both places.
EXPECTED_INVENTORY = {
    # B16 added two, both illustrative: `docs/overflow-tripwire.md`'s eager
    # section shows the defect (`jnp.full((), 256, jnp.int8)` is 0) and the
    # declaration that answers it. The first CANNOT be executed here — with
    # the eager detector armed it raises by design, and with it off it prints
    # nothing, so there is no output to compare either way; it is a record of
    # what jax does, which is what `illustrative` means.
    # B19 on `docs/reading-a-verdict.md`: ROW 1 of the status table -- the
    # first thing a reader of that page reads -- said VERIFIED means "true at
    # every point of the declared box", and under a narrowing `assume` it does
    # not. The page said the qualified thing correctly eight hundred lines
    # lower down. The new block DEMONSTRATES the counter-example rather than
    # asserting the qualifier: VERIFIED, with `0.2` in the declared box and
    # `0.2 > 0.5` False, and the stamp's `constrained assume` line naming the
    # narrowing.
    #
    # B19 on `docs/harness-api.md`: the page opened "Every code block on this
    # page was executed verbatim ... and the outputs are what it printed",
    # which one block on the page contradicted -- it was marked illustrative,
    # and it could not have been executed verbatim because it uses `lax` and
    # nothing on the page imports it. It also carried `# VERIFIED` on a
    # harness that is UNKNOWN under the call the page documents. Now executed
    # and compared, with a third harness added so the withholding it is about
    # is visible: `no_assume` REFUTED against `not_honoured` UNKNOWN is what
    # makes the first row evidence rather than a comment. And a second block
    # PRINTS `check`'s signature from the object, because the typed one had
    # published four of its nine parameters for six weeks.
    #
    # B19: `docs/inductive-step.md` went from FOUR illustrative blocks and no
    # gate at all to four executed-and-compared ones. It was the page whose
    # running example was REFUTED by the solver the page recommends, with the
    # 0.0495 escape (`10 + 5 x 0.99 x 0.01`) diagnosed in prose as 1-ULP
    # outward rounding -- so the page taught a reader to dismiss a real escape
    # as a rounding artefact, and nothing in the suite ran a line of it. The
    # four now are: a body that genuinely preserves its invariant (so the
    # page's "If VERIFIED" is demonstrable), the array-valued state shape, the
    # ONE-ULP PAIR (`x/3*3` VERIFIED against `x*0.1*10` REFUTED, identical
    # 2.2e-16 miss, opposite answers), and the escaping body kept as an
    # explicit refutation.
    #
    # B19 also added four elsewhere. Two are on
    # `docs/choosing-a-solver-backend.md`, which
    # had NO gate at all -- `grep -rn choosing-a-solver-backend tests/
    # .github/` had no hits -- and which had a whole section arguing from a
    # case the emission cannot produce, retracted in `solvers.py` on
    # 2026-08-20 and not on the page. Its two new blocks are the two claims
    # on it that are about MECHANISM rather than wall-clock: the `x**(1/80)`
    # degree-cap decline (which prints that ZERO backends were invoked, the
    # whole of the retraction) and the `semantics="ieee"` + solver_timeout_ms
    # caller error. The page's speed table stays hand-checked and says so.
    #
    # And two are a corrected
    # `proposed-*.md` header paying for itself. `proposed-int-literal-convert.md`
    # and `proposed-div-straddle-decline.md` each argued from a hand-written
    # fence that had gone false under the commit that BUILT the thing they
    # proposed (`cbb1d60`, `32c6c56`) — so each keeps its pre-change fence,
    # marked as pre-change, and gains a block that prints the current reading
    # here. Both are chosen to be path-free: they print interval endpoints and
    # booleans, not a render, because the normaliser only rewrites a source-info
    # line that begins the line and a fence carrying an inline temp path could
    # not be byte-compared.
    "python_blocks": 55,
    "illustrative": 14,
    "executed": 41,
    "run_only": 3,
    "compared": 38,
    # B15 added two: `docs/overflow-tripwire.md` now quotes the trace gate's
    # narrowed refusal and its NOT-FULLY-OBSERVED refusal side by side, which
    # is the whole point of that section — the two sentences have to be
    # readable against each other. Both are hand-written excerpts of a note
    # `preconditions._pipeline` composes, not example output, so they are
    # unattached by construction; `tests/test_tripwire_gate.py` is what holds
    # the real sentences down.
    # B16 added one: the eager detector's alarm, quoted so a reader can see
    # what it says before switching it on. Hand-written and attached to
    # nothing, like every other rendered excerpt on that page.
    # B16 fixup 3 added one more, for the same reason and in the same
    # section: the three measured `PRNGKey(N) == PRNGKey(M)` equalities that
    # say a seed wider than int32 does not survive jax's numpy-level cast.
    # It is a MEASUREMENT quoted in prose, not an example's output --
    # `tests/test_tripwire_eager.py` re-drives the equalities against the
    # real jax, which is what holds them down.
    # B16 fixup 4 added one more, in the same section and for the same
    # reason: the two instruments' readings of
    # `threefry_prng_impl.seed(np.int64(2**32 - 1))` with `jit` on and off,
    # which is the gap that program has in jax's DEFAULT configuration.
    # Measured, hand-written, attached to nothing;
    # `tests/test_tripwire_eager.py` re-drives every line of it.
    "plain_fences": 81,
    "plain_unattached": 43,
}

_MARKER = re.compile(r"<!--\s*doc-example:\s*(illustrative|run-only)\s*-->")
_FENCE = re.compile(r"^(\s*)```(\w*)\s*$")

# --- the docstring's own numbers, read back out of it ------------------
#
# ``consumed`` has no entry in EXPECTED_INVENTORY: it is the same
# quantity as ``compared`` seen from the fence side, and the table states
# both, so the reader keeps it and the test asserts they agree.
_TABLE_ANCHOR = ":func:`test_inventory_is_what_the_docstring_says`::"
_TABLE_ROW = re.compile(r"^\s{4,}(?P<label>\S.*?)\s{2,}(?P<n>\d+)\s*$")
_DOCSTRING_ROWS = {
    "```python blocks": "python_blocks",
    "marked illustrative — not run": "illustrative",
    "EXECUTED (exit 0 required)": "executed",
    "marked run-only — output not compared": "run_only",
    "OUTPUT COMPARED against a fence": "compared",
    "plain ``` fences": "plain_fences",
    "consumed as an example's claimed output": "consumed",
    "HAND-WRITTEN, compared to nothing": "plain_unattached",
}
# figures the prose derives from the table, in the order they are stated
_DERIVED_FIGURES = [
    (re.compile(r"and (\d+) of the (\d+) have their stdout compared"),
     ("compared", "executed")),
    (re.compile(r"The (\d+) unattached fences"), ("plain_unattached",)),
]


def _docstring_inventory() -> dict[str, int]:
    """The inventory as the DOCSTRING states it, parsed out of it.

    The dict was checked against the docs and the docstring against
    nothing, so the table beside it was an honour-system copy — the
    failure message below asks a contributor to update both, and only one
    of them was load-bearing."""
    lines = (__doc__ or "").split("\n")
    starts = [n for n, ln in enumerate(lines) if ln.rstrip().endswith(_TABLE_ANCHOR)]
    assert len(starts) == 1, (
        f"the inventory table is located by the line ending {_TABLE_ANCHOR!r}; "
        f"found {len(starts)} such lines in this docstring"
    )
    got: dict[str, int] = {}
    for ln in lines[starts[0] + 1:]:
        if not ln.strip():
            continue
        if not ln.startswith("    "):
            break
        m = _TABLE_ROW.match(ln)
        assert m, f"inventory table row this reader cannot parse: {ln!r}"
        key = _DOCSTRING_ROWS.get(m.group("label").strip())
        assert key is not None, (
            f"unrecognised inventory row label {m.group('label').strip()!r} — "
            "add it to _DOCSTRING_ROWS rather than leaving it unread"
        )
        got[key] = int(m.group("n"))
    missing = sorted(set(_DOCSTRING_ROWS.values()) - set(got))
    assert not missing, f"the docstring inventory table lost rows: {missing}"
    return got


def _doc_files() -> list[pathlib.Path]:
    return [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]


def _fences(text: str):
    """Yield (lang, body, line_no, marker_or_None) for every fenced block.

    Indented fences (a block inside a list item) are dedented, because
    that is how a reader copies them.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        body: list[str] = []
        while j < len(lines) and not _FENCE.match(lines[j]):
            body.append(lines[j])
            j += 1
        marker = None
        for ln in lines[max(0, i - 2): i]:
            found = _MARKER.search(ln)
            if found:
                marker = found.group(1)
        yield m.group(2), textwrap.dedent("\n".join(body)), i + 1, marker
        i = j + 1


@dataclass(frozen=True)
class Block:
    line: int
    source: str
    claimed: str | None  # None == nothing to compare
    run_only: bool


def collect(text: str) -> tuple[list[Block], list[str]]:
    """THE PAIRING. Returns (runnable blocks, complaints).

    A complaint is a block that is executed, carries no output fence, and
    has not been marked ``run-only`` — the state that used to pass
    silently.
    """
    blocks = list(_fences(text))
    out: list[Block] = []
    complaints: list[str] = []
    for n, (lang, body, line, marker) in enumerate(blocks):
        if lang != "python" or marker == "illustrative":
            continue
        claimed = None
        if marker != "run-only" and n + 1 < len(blocks) and blocks[n + 1][0] == "":
            claimed = blocks[n + 1][1]
        if claimed is None and marker != "run-only":
            complaints.append(
                f"line {line}: executed with no output fence and no "
                f"`<!-- doc-example: run-only -->` marker"
            )
        out.append(Block(line=line, source=body, claimed=claimed,
                         run_only=marker == "run-only"))
    return out, complaints


def _cases():
    return [
        pytest.param(path, b, id=f"{path.name}:{b.line}")
        for path in _doc_files()
        for b in collect(path.read_text(encoding="utf-8"))[0]
    ]


CASES = _cases()

_AT_LINE = re.compile(r"^at\s+.*\.py:(\d+)")
_SOLVER_TIME = re.compile(r"answered \w+ in \d+ms")
_SOLVER_VERSION = re.compile(r"\b(z3|cvc5) [0-9][^\s(]*")


def normalise(text: str) -> list[str]:
    keep = []
    for raw in text.rstrip("\n").split("\n"):
        stripped = raw.rstrip().strip()
        m = _AT_LINE.match(stripped)
        if m:
            # keep the LINE NUMBER and everything after it: the block that
            # produced them is byte-identical to the one in the doc
            line = f"at <path>:{m.group(1)}{stripped[m.end(1):]}"
        else:
            line = raw.rstrip()
        line = _SOLVER_TIME.sub("answered <status> in <time>", line)
        line = _SOLVER_VERSION.sub(r"\1 <version>", line)
        keep.append(line)
    return keep


# --- a recorded transcript belongs to the jax series that produced it ---
#
# A verdict stamp carries the jax version that produced it, and the query
# content hash is a FUNCTION of that version. Measured on
# ``docs/quickstart.md``'s block, jax 0.10.2 vs 0.11.0: jax 0.11 replaced
# ``jit``'s boolean ``inline`` param with an ``Inline`` enum, so the same
# harness traces to ``inline=False`` on 0.10.2 and ``inline=Inline.AUTO``
# on 0.11.0. Those two lines — the stamp and the hash — are the ONLY
# divergence in the entire verdict between the two series; the transcribed
# IR is otherwise byte-identical.
#
# stelling deliberately does NOT normalise that away in the transcriber.
# The param genuinely changed, ``Inline`` carries four further states that
# 0.10 cannot express, and rewriting AUTO to False would both falsify the
# record and move every 0.11 query hash. The hash is honestly
# series-dependent, so a doc cannot be a byte-exact transcript of more
# than one series at once.
#
# Hence: a recorded transcript is compared byte for byte — INCLUDING its
# hash — whenever the running jax is the jax the doc names. On any other
# series those two lines, and only those two, are neutralised on both
# sides; everything else stays byte-for-byte, so this still measures that
# the verdict is otherwise identical across the series.
#
# Deliberately NOT part of normalise(): that function's promise, pinned by
# test_normalisation_is_narrow, is that it never eats a hash, and the
# promise stays literally true.
#
# THE LIMIT, STATED: on a jax series other than the one a doc names, this
# file does not verify that doc's query hash. A stelling change that moved
# a hash while moving no other line of the verdict would pass here on that
# lane — and fail on the lane whose series the doc names, which is the
# reason a doc names one.
#
# WHICH MAKES "THE LANE WHOSE SERIES THE DOC NAMES" LOAD-BEARING, AND IT
# WAS NOT ENFORCED. The escape above is keyed on the doc's own stamp, so a
# doc naming a series NO lane runs escaped on every lane at once and its
# hash was then verified nowhere. Measured, before the guard below existed:
# stamping `docs/quickstart.md` with `jax 0.7.3` and overwriting every
# `query <sha>` line with `d`×64 passed 37/37 on BOTH lanes. Benign as
# found — the file named 0.11.0, which has a lane — but it is the same
# shape as the defect this whole pass is about: a claim with no lane
# behind it. `test_every_documented_stamp_names_a_tested_series` is the
# fence, and it is deliberately NOT inside the off-series branch: a block
# that skips (no solver installed) must not take its stamp with it.
# TWO FACTS, TWO CONDITIONS, AND THEY WERE ONE — which is how a documented
# hash's coverage went to zero without anybody deciding it should.
#
# `off_series` was computed as `doc_jax != run_jax`, an EXACT-VERSION test,
# and it gated BOTH neutralisations. But the two lines are different facts
# with different conditions:
#
#   * THE STAMP LINE prints the RUNNING jax version. It differs the moment
#     the exact version differs, for any reason, and there is nothing to
#     compare — `jax 0.11.0` against `jax 0.11.1` is the example literally
#     doing its job. Exact version is the right condition for it.
#   * THE QUERY HASH is a function of the traced equations' params. It moves
#     when jax moves those params, which is a per-release fact but NOT the
#     same fact as "the version string differs".
#
# Conflating them means: the day `test-jax` floats past the version a doc
# stamps, the hash stops being compared on every lane at once. That is not
# hypothetical. Measured at 3482822, `docs/quickstart.md`'s `query <sha>`
# overwritten with `d`x64 in a scratch tree, running the whole of
# `tests/test_doc_examples.py`:
#
#     jax 0.10.2  40 passed          jax 0.11.0  1 failed, 39 passed
#     jax 0.11.1  40 passed
#
# and FIVE ci.yml jobs run a bare whole-tree `pytest` and therefore run this
# module -- counted rather than assumed, because an earlier draft of this
# comment named three: `test-no-jax` (this module skips: no jax),
# `test-jax` (floats -> jax 0.11.1 today), `test-jax-0-10` (pins the SERIES
# -> 0.10.2), `acceptance-any-pytree` (`.[solvers,jax]`, floats -> 0.11.1)
# and `acceptance-reproducer` (a matrix over the two SERIES). Not one of
# them resolves 0.11.0, so the hash was checked on NO lane. The same mutation to `docs/harness-api.md`'s
# `query hash:` line — whose fence carries no stamp, so nothing was ever
# neutralised for it — failed on all three.
#
# THE HASH'S CONDITION IS THE SERIES, and that is a choice with an argument
# rather than a convenience. A doc can only be a byte-exact transcript of one
# jax, so SOME escape is needed or the 0.10 lane is permanently red on a page
# that names 0.11. Keying the escape on the series makes the escape as narrow
# as the doc's own claim: within the series the doc names, the hash is
# compared byte for byte, so a patch release that moves it — which happens;
# `SOUNDNESS.md`'s 2026-08-18 entry is a patch release moving a query hash —
# turns that lane red and the remedy is to re-record the block. Across series
# it is neutralised, and THAT gap is not left as an absence: it is declared
# in `EXPECTED_HASH_COVERAGE` and measured by
# `test_every_documented_query_hash_is_compared_on_some_lane`.
#
# Measured, and it is why the tight condition is green today rather than red:
# `docs/quickstart.md`'s block traces to the SAME hash on jax 0.11.0 and jax
# 0.11.1 (it differs on 0.10.2, which is the `Inline` param above), and
# `docs/harness-api.md`'s `reduce_sum` hash is identical on all three.
_STAMP_JAX = re.compile(r"^(stelling \S+ \| jax )(\S+)$")
_QUERY_HASH = re.compile(r"^query [0-9a-f]{64}$")

#: Every shape in which a doc PUBLISHES a query content hash: the verdict
#: render's `query <sha>` line and `ClosedJaxpr.content_hash()` printed by
#: hand as `query hash: <sha>`. Wider than :data:`_QUERY_HASH` on purpose —
#: that one is the neutralisation TARGET (only a verdict render carries a
#: stamp, so only it can ever be off-series), this one is the INVENTORY
#: DOMAIN, and an inventory that could not see the second shape would have
#: declared coverage for half the hashes in the docs.
_DOCUMENTED_QUERY_HASH = re.compile(r"^query(?: hash:)? [0-9a-f]{64}$")


def claimed_jax_version(lines: list[str]) -> str | None:
    """The jax version a normalised verdict stamps itself with, if any."""
    for line in lines:
        m = _STAMP_JAX.match(line)
        if m:
            return m.group(2)
    return None


def _series_of(version: str) -> str:
    """``0.11.1`` -> ``0.11``. The unit `TESTED_JAX_SERIES` is keyed on."""
    return ".".join(version.split(".")[:2])


def hash_is_compared(doc_jax: str | None, running_jax: str) -> bool:
    """Whether a doc's query hash is compared byte for byte on ``running_jax``.

    ONE FUNCTION, TWO CALLERS, deliberately: :func:`test_doc_example` uses it
    to decide, and
    :func:`test_every_documented_query_hash_is_compared_on_some_lane` uses it
    to MEASURE the declared inventory. A second copy of this rule is a second
    place for the inventory to be a claim about something other than what
    runs.

    A block with no stamp (``doc_jax is None``) names no series, so nothing
    licenses an escape and it is compared everywhere.
    """
    if doc_jax is None:
        return True
    return _series_of(doc_jax) == _series_of(running_jax)


def neutralise_stamp_line(lines: list[str]) -> list[str]:
    """Blank the jax version in the stamp, and nothing else."""
    return [_STAMP_JAX.sub(r"\1<version>", line) for line in lines]


def neutralise_query_hash(lines: list[str]) -> list[str]:
    """Blank a verdict's ``query <sha>`` line, and nothing else."""
    return [
        "query <hash: series-dependent>" if _QUERY_HASH.match(line) else line
        for line in lines
    ]


def neutralise_series_stamp(lines: list[str]) -> list[str]:
    """Both of the above. Kept as one name because "the two lines a jax
    series change is allowed to move" is still a meaningful set — it is the
    UNION the old single condition applied, and the tests below pin that the
    union touches exactly those two lines and no others. What changed is
    that :func:`test_doc_example` now applies the halves under their own
    conditions rather than this composition under one."""
    return neutralise_query_hash(neutralise_stamp_line(lines))


def run_block(source: str, tmp_path: pathlib.Path) -> subprocess.CompletedProcess:
    script = tmp_path / "block.py"
    script.write_text(source, encoding="utf-8")
    env = dict(os.environ)
    # measure THIS tree, never whatever stelling the venv has installed
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    env.setdefault("JAX_PLATFORMS", "cpu")
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=600,
    )


def test_every_executed_block_declares_whether_it_is_compared():
    """The hole this closes: deleting an example's output fence used to
    downgrade it to run-only with every test still green."""
    complaints = [
        f"{path.name} {c}"
        for path in _doc_files()
        for c in collect(path.read_text(encoding="utf-8"))[1]
    ]
    assert not complaints, "\n".join(complaints)


def test_inventory_is_what_the_docstring_says():
    """The docstring states numbers; this is what makes them true.

    It goes red when the docs gain or lose an example, or when a checked
    block is quietly marked illustrative — decisions that should be made
    in the open rather than absorbed.

    Two halves, and for a long time only the first existed: the counts
    are recomputed from the docs and compared to ``EXPECTED_INVENTORY``,
    AND ``EXPECTED_INVENTORY`` is compared to the numbers read back out
    of the docstring table and the sentences derived from it. Without the
    second half this test did not read the docstring it is named for."""
    got = dict.fromkeys(EXPECTED_INVENTORY, 0)
    for path in _doc_files():
        blocks = list(_fences(path.read_text(encoding="utf-8")))
        for n, (lang, _body, _line, marker) in enumerate(blocks):
            if lang == "":
                got["plain_fences"] += 1
                prev = blocks[n - 1] if n else None
                consumed = (
                    prev is not None
                    and prev[0] == "python"
                    and prev[3] is None
                )
                if not consumed:
                    got["plain_unattached"] += 1
                continue
            if lang != "python":
                continue
            got["python_blocks"] += 1
            if marker == "illustrative":
                got["illustrative"] += 1
                continue
            got["executed"] += 1
            if marker == "run-only":
                got["run_only"] += 1
            else:
                got["compared"] += 1
    assert got == EXPECTED_INVENTORY, (
        f"doc-example inventory moved.\n  expected {EXPECTED_INVENTORY}\n"
        f"  actual   {got}\n"
        "Update EXPECTED_INVENTORY *and* the table in this module's "
        "docstring — together they are this file's claim about itself."
    )

    stated = _docstring_inventory()
    assert stated.pop("consumed") == stated["compared"], (
        "the docstring says a different number of plain fences are consumed "
        "as claimed output than are compared; they are one quantity"
    )
    assert stated == EXPECTED_INVENTORY, (
        f"the docstring's inventory table is not EXPECTED_INVENTORY.\n"
        f"  docstring {stated}\n  dict      {EXPECTED_INVENTORY}\n"
        "Both are this file's claim about itself and neither is decorative."
    )

    flat = re.sub(r"\s+", " ", __doc__ or "")
    for pattern, keys in _DERIVED_FIGURES:
        m = pattern.search(flat)
        assert m is not None, (
            f"the docstring sentence matching {pattern.pattern!r} is gone; it "
            "carried figures derived from the inventory table"
        )
        stated_figures = tuple(int(g) for g in m.groups())
        wanted = tuple(EXPECTED_INVENTORY[k] for k in keys)
        assert stated_figures == wanted, (
            f"the docstring prose says {stated_figures} where the inventory "
            f"says {wanted} for {keys}"
        )


def test_collected_cases_match_the_inventory():
    assert len(CASES) == EXPECTED_INVENTORY["executed"]


@pytest.mark.parametrize("path,block", CASES)
def test_doc_example(path, block, tmp_path):
    pytest.importorskip("jax")
    if "solver_timeout_ms" in block.source and not (
        available("z3") or available("cvc5")
    ):
        pytest.skip("block opts into solver escalation; no backend installed")

    r = run_block(block.source, tmp_path)
    assert r.returncode == 0, (
        f"{path.name}:{block.line} — example does not run\n"
        f"--- stderr ---\n{r.stderr[-2000:]}"
    )
    if block.claimed is None:
        return
    want, got = normalise(block.claimed), normalise(r.stdout)
    doc_jax, run_jax = claimed_jax_version(want), claimed_jax_version(got)
    # TWO FACTS, TWO CONDITIONS — see the block comment above _STAMP_JAX.
    known = doc_jax is not None and run_jax is not None
    off_version = known and doc_jax != run_jax
    off_series = off_version and not hash_is_compared(doc_jax, run_jax)
    if off_version:
        # the stamp prints the RUNNING jax; a different jax at all makes
        # this line differ with nothing to compare
        want, got = neutralise_stamp_line(want), neutralise_stamp_line(got)
    if off_series:
        # and only a different SERIES licenses dropping the hash
        want, got = neutralise_query_hash(want), neutralise_query_hash(got)
    assert want == got, (
        f"{path.name}:{block.line} — printed output does not match the doc"
        + (
            f"\n(running jax {run_jax}, doc recorded on jax {doc_jax}: the "
            f"stamp line was neutralised"
            + (
                " and so was the query hash, since the SERIES differs, so "
                "this failure is in the series-independent part of the "
                "verdict"
                if off_series
                else ", but the QUERY HASH WAS COMPARED — this is the same "
                "series as the doc, so a hash difference here is a real "
                "one: either the tree moved it or the doc needs re-recording"
            )
            + ")"
            if off_version
            else ""
        )
        + "\n"
        + "\n".join(
            difflib.unified_diff(want, got, "doc claims", "actual", lineterm="")
        )
    )


# ---------------------------------------------------------------- controls

SYNTHETIC = """\
# synthetic

```python
print('two')
```

```
one
```

<!-- doc-example: illustrative -->
```python
this is not python
```

<!-- doc-example: run-only -->
```python
print('unchecked')
```

```python
print('no fence follows me')
```

```js
not python either
```
"""


def test_pairing_control():
    """Drives the REAL pairing function, which the other controls do not.

    Without this, a regression in ``collect`` could silently degrade
    every case to run-only and leave the whole file green."""
    blocks, complaints = collect(SYNTHETIC)

    # the illustrative block is gone, the js block is gone, three remain
    assert [b.line for b in blocks] == [3, 17, 21]

    # 1. a block followed by a plain fence is paired with it
    assert blocks[0].source.strip() == "print('two')"
    assert blocks[0].claimed.strip() == "one"
    assert not blocks[0].run_only

    # 2. run-only is executed but carries no claim
    assert blocks[1].run_only and blocks[1].claimed is None

    # 3. an unmarked block with no fence is COMPLAINED about rather than
    #    silently downgraded — the hole that used to pass
    assert blocks[2].claimed is None and not blocks[2].run_only
    assert len(complaints) == 1 and "line 21" in complaints[0]

    # 4. deleting an existing block's output fence starts complaining
    without_fence = SYNTHETIC.replace("```\none\n```\n", "")
    assert any("line 3" in c for c in collect(without_fence)[1])


def test_the_checker_bites(tmp_path):
    """Positive control on the comparison and on the runner."""
    blocks, _ = collect(SYNTHETIC)

    # a wrong claimed output is detected ...
    r = run_block(blocks[0].source, tmp_path)
    assert r.returncode == 0
    assert normalise(blocks[0].claimed) != normalise(r.stdout), (
        "the comparison accepted 'one' as the output of print('two')"
    )
    assert normalise("two\n") == normalise(r.stdout)

    # ... and so is a block that fails
    assert run_block("raise SystemExit(3)\n", tmp_path).returncode != 0


def test_normalisation_is_narrow():
    """The normaliser must neutralise the environment-dependent parts and
    NOTHING else — one that ate a status or a hash would make the
    comparison meaningless."""
    # the directory and file name go; the line number and function stay
    assert normalise("at /tmp/pytest-1/block.py:12 (harness)") == [
        "at <path>:12 (harness)"
    ]
    assert normalise("  at <your dir>/quickstart.py:12 (harness)") == [
        "at <path>:12 (harness)"
    ]
    # so a doc that misreports the line or the function is still caught
    assert normalise("at a/x.py:12 (harness)") != normalise("at b/y.py:13 (harness)")
    assert normalise("at a/x.py:12 (harness)") != normalise("at a/x.py:12 (other)")

    assert normalise("note: assert #0: z3 (wheel) answered sat in 8ms") == [
        "note: assert #0: z3 (wheel) answered <status> in <time>"
    ]
    assert normalise("produced by: z3 5.0.0 (wheel-bindings)") == [
        "produced by: z3 <version> (wheel-bindings)"
    ]
    # everything that carries a claim survives untouched.
    #
    # THE TWO HASHES BELOW ARE SAMPLE TEXT, NOT MEASUREMENTS — which is
    # worth writing down now that a jax release has been shown to move a
    # query hash (jax 0.11.1, `reduce_max`/`reduce_min`; see SOUNDNESS.md).
    # What is asserted here is that `normalise` returns them UNTOUCHED, and
    # that holds for any 64 hex digits. They happen to be `quickstart.md`'s,
    # so a reader will think they are pinned to it: they are not, and
    # re-recording them for a new jax would measure nothing. The real pins
    # live in `docs/quickstart.md` and `docs/harness-api.md`.
    for line in (
        "== VERIFIED",
        "nonvacuity: UNCHECKED — no membership conditions declared",
        "coverage: 9 eqns: 8 known (89%); 1 transparent",
        "query 628a25efd4417f44966443e7275a31b7c437cc45ddb6b42efcadb59308171765",
        "assert #0: discharged — definitely true for all 8 element(s)",
        "hand declaration       5 eqns  hash 93bfe936574a4195",
        # the stamp and the hash are series-dependent, but the NORMALISER
        # is not what neutralises them — that gate lives in
        # test_doc_example and fires only off-series. Pinned here so a
        # future edit cannot quietly move it into the always-on path.
        "stelling 0.1.0 | jax 0.11.0",
    ):
        assert normalise(line) == [line]


def test_the_series_gate_neutralises_two_lines_and_no_others():
    """The off-series gate must blank the stamp and the query hash — and
    leave every other line of a verdict alone. A gate that ate a status or
    a coverage line would make an off-series lane measure nothing."""
    verdict = [
        "== VERIFIED",
        "  9 equations verified",
        "stelling 0.1.0 | jax 0.11.0",
        "query 628a25efd4417f44966443e7275a31b7c437cc45ddb6b42efcadb59308171765",
        "coverage: 9 eqns: 8 known (89%); 1 transparent",
    ]
    assert neutralise_series_stamp(verdict) == [
        "== VERIFIED",
        "  9 equations verified",
        "stelling 0.1.0 | jax <version>",
        "query <hash: series-dependent>",
        "coverage: 9 eqns: 8 known (89%); 1 transparent",
    ]
    # the stelling version is NOT neutralised — only jax's
    assert neutralise_series_stamp(["stelling 9.9.9 | jax 0.11.0"]) == [
        "stelling 9.9.9 | jax <version>"
    ]
    # a hash that is not the query stamp is untouched
    assert neutralise_series_stamp(["hand declaration  5 eqns  hash 93bfe936"]) == [
        "hand declaration  5 eqns  hash 93bfe936"
    ]


def test_the_series_gate_reads_the_version_out_of_the_transcript():
    """No jax version is hardcoded here: the doc names its own series, and
    a transcript with no stamp gets no gate at all (so a stamp-less block
    is always compared at full strength)."""
    assert claimed_jax_version(["== VERIFIED", "stelling 0.1.0 | jax 0.10.2"]) == "0.10.2"
    assert claimed_jax_version(["== VERIFIED", "coverage: 9 eqns"]) is None


def test_every_documented_stamp_names_a_tested_series():
    """A doc may only stamp a jax series that has a CI lane.

    THE HOLE THIS CLOSES. ``test_doc_example`` neutralises the stamp and
    the query hash whenever the running jax is not the jax the doc names.
    That escape is keyed on the doc's own text, so a doc naming a series
    NOBODY runs took the escape on every lane simultaneously and its hash
    was verified nowhere. Measured before this existed: ``quickstart.md``
    stamped ``jax 0.7.3`` with every ``query <sha>`` line overwritten by
    ``d``×64 passed 37/37 on the 0.10.2 lane AND the 0.11.0 lane.

    With this test, a stamp is only escapable if some lane is obliged to
    check it — which is what makes "fails on the lane whose series the doc
    names" a true sentence rather than a hopeful one.

    Scans the RAW text of every doc, not just executed blocks: a block
    that skips (no solver installed) must not carry its stamp out of
    range, and an illustrative fence still makes a claim to a reader.
    """
    offenders = []
    for path in _doc_files():
        for n, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            m = _STAMP_JAX.match(line.strip())
            if m and not jax_series_tested(m.group(2)):
                offenders.append(f"{path.name}:{n} stamps jax {m.group(2)}")
    assert not offenders, (
        "a documented verdict stamps a jax series with no CI lane, so its "
        "query hash is compared on no lane at all — re-record the block on "
        f"a tested series (TESTED_JAX_SERIES = {TESTED_JAX_SERIES}):\n  "
        + "\n  ".join(offenders)
    )


#: WHICH LANE CHECKS WHICH DOCUMENTED QUERY HASH. Keys are
#: ``<doc file>#<n>``, ``n`` counting the hash lines in that file in document
#: order; values are the jax series THE MERGE-BEARING LANES ACTUALLY RESOLVE
#: (``_lanes.lane_series()``) on which :func:`test_doc_example` compares that
#: hash BYTE FOR BYTE. An empty tuple means the hash is checked nowhere, and
#: is a failure.
#:
#: KEYED ON THE LANES AND NOT ON ``TESTED_JAX_SERIES``, and the difference is
#: the hole B13 closed reappearing one level up. This dict is a claim about
#: which LANE is obliged to check which hash — the failure text below has said
#: "compared on NO tested jax lane" since it was written — while the
#: recomputation read a tuple of series, and nothing forced a series entry to
#: have a lane. Today ``"0.11"`` is delivered by the FLOATING lane alone, so
#: the day jax 0.12 ships and ``TESTED_JAX_SERIES`` is bumped to
#: ``("0.10", "0.11", "0.12")`` — which is what a maintainer does after the two
#: fences that go red are cleared — ``measured`` for ``quickstart.md#0`` stays
#: ``("0.11",)``, this dict still says ``("0.11",)``, the test passes, and no
#: lane runs 0.11 at all. Driven end to end; ``tests/test_lanes.py``'s
#: ``test_every_tested_series_has_a_lane`` carries the transcript.
#:
#: THE RIGHT-HAND SIDE IS MEASURED, NEVER TYPED FROM HOPE — recomputed by
#: :func:`test_every_documented_query_hash_is_compared_on_some_lane` from the
#: same :func:`hash_is_compared` that decides the real comparison. This is
#: `tests/test_skip_inventory.py`'s idiom and it is here for the same reason:
#: this coverage went silently to zero once already (see the block comment
#: above ``_STAMP_JAX``) and an absence cannot be reviewed. A declaration can.
#:
#: WHY THE TWO ENTRIES DIFFER, which is the thing to read before changing one:
#:
#: * ``quickstart.md#0`` sits in a verdict render that stamps ``jax 0.11.0``,
#:   so its hash is compared on the 0.11 lane and neutralised on the 0.10
#:   one — a doc cannot be a byte-exact transcript of two series at once.
#: * ``harness-api.md#0`` is a hand-printed ``content_hash()`` in a fence
#:   that carries NO stamp, so no series escape applies to it anywhere and it
#:   is compared on every lane.
EXPECTED_HASH_COVERAGE = {
    "harness-api.md#0": ("0.10", "0.11"),
    "quickstart.md#0": ("0.11",),
}


def _documented_query_hashes():
    """Every query content hash the docs publish, and whether it is compared.

    Scans the RAW text — an illustrative fence, a run-only block or a fence
    attached to nothing still shows a reader a hash — and then asks whether
    that exact line is part of some COMPARED block's claimed output. A 64-hex
    line is unique enough for that to be a sound membership test, and it
    keeps this function from re-implementing :func:`collect`'s pairing, which
    would be a second place for the two to disagree.

    Yields ``(key, file name, line number, doc's jax version or None,
    compared)``.
    """
    for path in _doc_files():
        text = path.read_text(encoding="utf-8")
        compared: dict[str, str | None] = {}
        for block in collect(text)[0]:
            if block.claimed is None:
                continue
            lines = normalise(block.claimed)
            stamp = claimed_jax_version(lines)
            for line in lines:
                if _DOCUMENTED_QUERY_HASH.match(line.strip()):
                    compared[line.strip()] = stamp
        n = 0
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not _DOCUMENTED_QUERY_HASH.match(line):
                continue
            yield (
                f"{path.name}#{n}",
                path.name,
                lineno,
                compared.get(line),
                line in compared,
            )
            n += 1


def test_every_documented_query_hash_is_compared_on_some_lane():
    """A documented hash checked on no lane is a named fact, not an absence.

    THE HOLE THIS CLOSES, and it opened silently. The off-series escape used
    to be keyed on the EXACT jax version, so the day `test-jax` floated past
    the version `docs/quickstart.md` stamps, that page's query hash was
    neutralised on every lane at once. Driven at 3482822 by overwriting it
    with `d`x64 in a scratch tree: 40 passed on jax 0.10.2, 40 passed on jax
    0.11.1, 1 failed on jax 0.11.0 — and no CI lane runs 0.11.0. The hash was
    verified nowhere and nothing said so.

    `test_every_documented_stamp_names_a_tested_series` is the fence for a
    doc naming a series NO lane runs. This is the fence for the other half:
    the escape's condition widening until it covers every lane there is. That
    one reads the doc's stamp; this one reads what the comparison actually
    does, through the same predicate the comparison uses.
    """
    measured = {}
    where = {}
    for key, name, lineno, doc_jax, compared in _documented_query_hashes():
        where[key] = f"{name}:{lineno}"
        measured[key] = (
            tuple(
                s
                for s in _lanes.lane_series()
                if hash_is_compared(doc_jax, f"{s}.0")
            )
            if compared
            else ()
        )

    unchecked = {k: where[k] for k, v in measured.items() if not v}
    assert not unchecked, (
        "a documented query content hash is compared on NO tested jax lane, "
        "so nothing in CI would notice it going stale — either re-record the "
        "block on a series that has a lane, or put the hash in a fence that "
        f"is compared: {unchecked}\n"
        f"The lanes resolve {_lanes.lane_series()}; TESTED_JAX_SERIES claims "
        f"{TESTED_JAX_SERIES}. Where those differ, the claim is the thing "
        "that is wrong."
    )
    assert measured == EXPECTED_HASH_COVERAGE, (
        "the documented-hash coverage moved.\n"
        f"  declared {EXPECTED_HASH_COVERAGE}\n"
        f"  measured {measured}\n"
        f"  where    {where}\n"
        "Update EXPECTED_HASH_COVERAGE *and* say in its comment why the new "
        "coverage is the coverage that block should have. This dict is a "
        "claim about which lane is obliged to check which hash; it is not a "
        "number to be bumped until the suite goes green."
    )


def test_the_hash_coverage_predicate_is_not_vacuous():
    """The inventory above is only worth having if its predicate discriminates.

    Without this, `hash_is_compared` returning True for everything would
    declare full coverage for every hash and this file would once again be
    measuring nothing.
    """
    # the domain is non-empty: an empty inventory proves nothing
    assert EXPECTED_HASH_COVERAGE
    # same series: compared, including across a PATCH release, which is the
    # narrowing that makes a moved hash inside a series go red
    assert hash_is_compared("0.11.0", "0.11.0")
    assert hash_is_compared("0.11.0", "0.11.1")
    # different series: not compared
    assert not hash_is_compared("0.11.0", "0.10.2")
    assert not hash_is_compared("0.10.2", "0.11.1")
    # no stamp at all: nothing licenses an escape
    assert hash_is_compared(None, "0.10.2")
    # and the wider inventory regex really is wider than the neutraliser's
    assert _DOCUMENTED_QUERY_HASH.match("query hash: " + "0" * 64)
    assert not _QUERY_HASH.match("query hash: " + "0" * 64)
    assert _DOCUMENTED_QUERY_HASH.match("query " + "0" * 64)
    assert _QUERY_HASH.match("query " + "0" * 64)


def test_the_stamp_fence_rejects_an_untested_series():
    """The fence's own anti-vacuity control: it must actually reject."""
    assert not jax_series_tested("0.7.3")
    assert all(jax_series_tested(f"{s}.0") for s in TESTED_JAX_SERIES)
    # and the regex it leans on finds the version it is given
    assert _STAMP_JAX.match("stelling 0.1.0 | jax 0.7.3").group(2) == "0.7.3"
