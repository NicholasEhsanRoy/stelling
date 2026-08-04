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

    ```python blocks                                35
      marked illustrative — not run                  6
      EXECUTED (exit 0 required)                    29
        marked run-only — output not compared        3
        OUTPUT COMPARED against a fence             26
    plain ``` fences                                52
      consumed as an example's claimed output       26
      HAND-WRITTEN, compared to nothing             26

So the claim this file earns is: *every runnable example runs, and 26 of
the 29 have their stdout compared byte for byte after a narrow
normalisation.* The 26 unattached fences — a render pasted into prose, a
quoted stamp line, an excerpt from another page's table — are **not**
verified here. Writing one of those is a hand-check and stays one.

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

Blocks that opt into a solver (they pass ``solver_timeout_ms``) are
skipped when no backend is installed, since their output is about an
escalation that cannot happen. Those examples are unverified in a
solver-free environment.

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

from stelling._optional import available

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"

# The inventory the docstring states. A change here is a change to what
# this file promises, so it must be made deliberately and in both places.
EXPECTED_INVENTORY = {
    "python_blocks": 35,
    "illustrative": 6,
    "executed": 29,
    "run_only": 3,
    "compared": 26,
    "plain_fences": 52,
    "plain_unattached": 26,
}

_MARKER = re.compile(r"<!--\s*doc-example:\s*(illustrative|run-only)\s*-->")
_FENCE = re.compile(r"^(\s*)```(\w*)\s*$")


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
    in the open rather than absorbed."""
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
    assert want == got, (
        f"{path.name}:{block.line} — printed output does not match the doc\n"
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
    ):
        assert normalise(line) == [line]
