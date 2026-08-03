# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Every documented example runs, and prints what the doc says it prints.

The can't-drift rule applied to the docs' code: an example that no longer
runs, or whose printed output has silently moved, is worse than no
example — it is a claim about the tool that the tool contradicts. This is
the test that catches it.

Its own history is the argument for it. The user-facing docs were written
and hand-verified against one commit; three merges later, two of the
recorded outputs were stale — the decline/undecided message passes had
lengthened an obligation detail line — and nothing in the suite noticed,
because nothing in the suite read a fenced block.

**Mechanism.** For every ```python block in ``README.md`` and
``docs/*.md``: write it to a temp file, run it in a subprocess, and
require exit 0. If the block is followed by a plain ``` fence, that fence
is the block's *claimed output* and must match the real stdout after
normalisation.

**A block that is not meant to run must say so**, with an HTML comment on
the line before its fence::

    <!-- doc-example: illustrative -->

Default-deny, deliberately: an unmarked block in a new doc is executed,
so the failure mode is a loud red test rather than a quietly unchecked
example. The marker is for fragments that are records rather than
instructions — an excerpt of upstream source, a one-line sketch.

**Normalisation, and its limits.** Three things in a real verdict are
environment-dependent and are neutralised on BOTH sides before
comparison:

* the ``at <path>:<line> (<fn>)`` source-info line — it names the temp
  file, and in the docs a ``<your dir>`` placeholder;
* solver wall times (``answered sat in 8ms``);
* solver versions (``z3 5.0.0`` -> ``z3 <version>``).

The third is a real limit worth stating: the *version digits* a doc
prints are not verified by this test, only their shape. Everything else —
statuses, stamp fields, coverage lines, query content hashes, refusal
text — is compared byte for byte.

Blocks that opt into a solver (they pass ``solver_timeout_ms``) are
skipped when no solver backend is installed, since their output is about
an escalation that cannot happen.

The subprocess runs with ``PYTHONPATH`` pointing at **this repo's**
``src/``. Without that a developer with stelling installed from
elsewhere would measure a different tree than the one they are editing,
which is the exact failure this file exists to prevent.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

from stelling._optional import available

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"

ILLUSTRATIVE = re.compile(r"<!--\s*doc-example:\s*illustrative\s*-->")
_FENCE = re.compile(r"^(\s*)```(\w*)\s*$")


def _doc_files() -> list[pathlib.Path]:
    return [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]


def _fences(text: str):
    """Yield (lang, body, line_no, preceded_by_marker) for every fenced block.

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
        marked = any(ILLUSTRATIVE.search(ln) for ln in lines[max(0, i - 2): i])
        yield m.group(2), textwrap.dedent("\n".join(body)), i + 1, marked
        i = j + 1


def _cases():
    """(doc, source, line_no, claimed_output_or_None) per runnable block."""
    out = []
    for path in _doc_files():
        blocks = list(_fences(path.read_text(encoding="utf-8")))
        for n, (lang, body, line, marked) in enumerate(blocks):
            if lang != "python" or marked:
                continue
            claimed = None
            if n + 1 < len(blocks) and blocks[n + 1][0] == "":
                claimed = blocks[n + 1][1]
            out.append(
                pytest.param(
                    path, body, line, claimed,
                    id=f"{path.name}:{line}",
                )
            )
    return out


CASES = _cases()

_AT_LINE = re.compile(r"^at .*\.py:\d+")
_SOLVER_TIME = re.compile(r"answered \w+ in \d+ms")
_SOLVER_VERSION = re.compile(r"\b(z3|cvc5) [0-9][^\s(]*")


def normalise(text: str) -> list[str]:
    keep = []
    for raw in text.rstrip("\n").split("\n"):
        line = raw.rstrip()
        if _AT_LINE.match(line.strip()):
            continue  # names the temp file / the doc's <your dir> placeholder
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


def test_there_are_examples_to_check():
    """A globbing bug that silently found nothing would make every other
    test in this file pass vacuously."""
    assert len(CASES) >= 20, f"only {len(CASES)} runnable doc example(s) found"


@pytest.mark.parametrize("path,source,line,claimed", CASES)
def test_doc_example(path, source, line, claimed, tmp_path):
    pytest.importorskip("jax")
    if "solver_timeout_ms" in source and not (
        available("z3") or available("cvc5")
    ):
        pytest.skip("block opts into solver escalation; no backend installed")

    r = run_block(source, tmp_path)
    assert r.returncode == 0, (
        f"{path.name}:{line} — example does not run\n"
        f"--- stderr ---\n{r.stderr[-2000:]}"
    )
    if claimed is None:
        return
    want, got = normalise(claimed), normalise(r.stdout)
    assert want == got, (
        f"{path.name}:{line} — printed output does not match the doc\n"
        + "\n".join(
            __import__("difflib").unified_diff(
                want, got, "doc claims", "actual", lineterm=""
            )
        )
    )


def test_the_checker_bites(tmp_path):
    """Positive control: the comparison must go red on a wrong fence, and
    on a block that raises. Without this, a normalisation bug that ate
    every difference would leave the parametrized tests green forever."""
    md = tmp_path / "synthetic.md"
    md.write_text(
        "# synthetic\n\n"
        "```python\n"
        "print('two')\n"
        "```\n\n"
        "```\n"
        "one\n"
        "```\n\n"
        "```python\n"
        "raise SystemExit(3)\n"
        "```\n",
        encoding="utf-8",
    )
    blocks = list(_fences(md.read_text(encoding="utf-8")))
    py = [b for b in blocks if b[0] == "python"]
    assert len(py) == 2

    # 1. wrong claimed output is detected
    r = run_block(py[0][1], tmp_path)
    assert r.returncode == 0
    assert normalise("one\n") != normalise(r.stdout), (
        "the comparison accepted 'one' as the output of print('two')"
    )
    assert normalise("two\n") == normalise(r.stdout)

    # 2. a block that fails is detected
    assert run_block(py[1][1], tmp_path).returncode != 0

    # 3. the illustrative marker is honoured, and only when present
    marked = tmp_path / "marked.md"
    marked.write_text(
        "<!-- doc-example: illustrative -->\n```python\nthis is not python\n```\n",
        encoding="utf-8",
    )
    assert all(b[3] for b in _fences(marked.read_text()) if b[0] == "python")
    unmarked = tmp_path / "unmarked.md"
    unmarked.write_text("```python\nthis is not python\n```\n", encoding="utf-8")
    assert not any(b[3] for b in _fences(unmarked.read_text()) if b[0] == "python")


def test_normalisation_is_narrow():
    """The normaliser must neutralise the three environment-dependent
    parts and NOTHING else — a normaliser that ate a status or a hash
    would make the comparison meaningless."""
    assert normalise("at /tmp/x/block.py:12 (harness)") == []
    assert normalise("  at /tmp/x/block.py:12 (harness)") == []
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
    ):
        assert normalise(line) == [line]
