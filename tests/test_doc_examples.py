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

    ```python blocks                                40
      marked illustrative — not run                 11
      EXECUTED (exit 0 required)                    29
        marked run-only — output not compared        3
        OUTPUT COMPARED against a fence             26
    plain ``` fences                                61
      consumed as an example's claimed output       26
      HAND-WRITTEN, compared to nothing             35

So the claim this file earns is: *every runnable example runs, and 26 of
the 29 have their stdout compared byte for byte after a narrow
normalisation.* The 35 unattached fences — a render pasted into prose, a
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

**One further neutralisation, and it is NOT in the normaliser.** A verdict
stamps the jax version that produced it, and a query content hash is a
function of that version (measured: jax 0.11 replaced ``jit``'s boolean
``inline`` param with an ``Inline`` enum, and that one param is the entire
difference between the 0.10.2 and 0.11.0 renders of
``docs/quickstart.md``'s block). A doc therefore cannot be a byte-exact
transcript of two jax series at once. So :func:`test_doc_example` compares
a transcript byte for byte — hash included — when the running jax is the
jax the doc names, and neutralises exactly two lines, the stamp and the
``query <hash>``, when it is not. Every other line stays compared, so an
off-series lane still measures that the verdict is otherwise identical.
The limit that buys: **on a series other than the one a doc names, that
doc's query hash is not verified here.** It is verified on the lane whose
series the doc names, which is the reason a doc names one.

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

from stelling._optional import TESTED_JAX_SERIES, available, jax_series_tested

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"

# The inventory the docstring states. A change here is a change to what
# this file promises, so it must be made deliberately and in both places.
EXPECTED_INVENTORY = {
    "python_blocks": 40,
    "illustrative": 11,
    "executed": 29,
    "run_only": 3,
    "compared": 26,
    "plain_fences": 61,
    "plain_unattached": 35,
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
_STAMP_JAX = re.compile(r"^(stelling \S+ \| jax )(\S+)$")
_QUERY_HASH = re.compile(r"^query [0-9a-f]{64}$")


def claimed_jax_version(lines: list[str]) -> str | None:
    """The jax version a normalised verdict stamps itself with, if any."""
    for line in lines:
        m = _STAMP_JAX.match(line)
        if m:
            return m.group(2)
    return None


def neutralise_series_stamp(lines: list[str]) -> list[str]:
    """Blank the two lines a jax series change is allowed to move."""
    out = []
    for line in lines:
        line = _STAMP_JAX.sub(r"\1<version>", line)
        if _QUERY_HASH.match(line):
            line = "query <hash: series-dependent>"
        out.append(line)
    return out


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
    off_series = (
        want != got
        and doc_jax is not None
        and run_jax is not None
        and doc_jax != run_jax
    )
    if off_series:
        # not this doc's series: its stamp and its query hash are the two
        # lines the series is allowed to move (see the note above). Every
        # other line is still compared byte for byte.
        want, got = neutralise_series_stamp(want), neutralise_series_stamp(got)
    assert want == got, (
        f"{path.name}:{block.line} — printed output does not match the doc"
        + (
            f"\n(running jax {run_jax}, doc recorded on jax {doc_jax}: the "
            f"stamp and query-hash lines were neutralised, so this failure "
            f"is in the SERIES-INDEPENDENT part of the verdict)"
            if off_series
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
    # everything that carries a claim survives untouched
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


def test_the_stamp_fence_rejects_an_untested_series():
    """The fence's own anti-vacuity control: it must actually reject."""
    assert not jax_series_tested("0.7.3")
    assert all(jax_series_tested(f"{s}.0") for s in TESTED_JAX_SERIES)
    # and the regex it leans on finds the version it is given
    assert _STAMP_JAX.match("stelling 0.1.0 | jax 0.7.3").group(2) == "0.7.3"
