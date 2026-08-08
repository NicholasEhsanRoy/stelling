# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE TWO TESTED jax SERIES MUST AGREE.

``stelling._optional.TESTED_JAX_SERIES`` claims ``("0.10", "0.11")``. A verdict
is a statement about the USER'S program; the jax series is an implementation
detail of the tool. So the same tree, given the same harness, must return the
same verdict on both — and if it does not, either a verdict is wrong or the
claim is.

**That claim has been false on this tree, and expensively.** At ``8ef8f75``,
``propagate._is_add_combiner`` recognised a scatter's combiner by testing
``isinstance(v, ir.ClosedJaxpr)``. jax 0.11 merged ``ClosedJaxpr`` into
``Jaxpr``; 0.10.2 did not. So ``x.at[0].add(5.0)`` came back VERIFIED at 6/6
equations known on 0.11.0 and UNKNOWN at 4/6 with a ⊤ scatter-add on 0.10.2 —
while ``TESTED_JAX_SERIES`` already named both series. Nine of the ten tests
that failed on 0.10.2 at that tree had that one cause.

────────────────────────────────────────────────────────────────────────────
HOW IT RUNS, AND THE THREE THINGS THAT MAKE IT NOT VACUOUS
────────────────────────────────────────────────────────────────────────────

Two jax series cannot coexist in one process, so this test spawns a second
interpreter named by ``STELLING_PROPERTY_OTHER_PYTHON``, runs the same corpus
against the same ``PYTHONPATH``, and diffs the two ledgers. Without that
variable the test SKIPS, with a registered reason — and three guards keep the
skip from becoming the finding:

1. **The two interpreters must actually differ.** If the child reports the same
   ``jax.__version__`` as the parent, the comparison is a tautology and the
   test FAILS rather than passing. A cross-version property run against one
   version is the purest form of the failure mode this suite exists to prevent.
2. **Both versions must be series this tree CLAIMS to support.** Comparing two
   untested series would be measuring something nobody promised.
3. **Both interpreters must be looking at the same ``stelling``.** The child
   inherits ``PYTHONPATH``, and its ``stelling.__file__`` is compared to the
   parent's. A child that silently imported an installed copy would be
   comparing two trees, not two series.

The corpus is built by ``_corpus.py`` from ``random.Random(seed)`` — NOT from a
Hypothesis search, because two processes cannot share one — and the diff is
keyed on the RENDERED HARNESS TEXT, so a corpus that did diverge shows up as
"present on one side only" instead of silently comparing two different programs.

────────────────────────────────────────────────────────────────────────────
WHAT IS AND IS NOT COVERED
────────────────────────────────────────────────────────────────────────────

Covered: whatever ``_corpus.corpus()`` contains — 220 pseudo-random harnesses
over the general grammar, plus five hand-built ones including the scatter-add
form that the ``8ef8f75`` control needs. Verdict, per-obligation statuses, and
refusal EXCEPTION TYPE are all compared.

NOT covered: anything the corpus does not build. That is the honest limit of a
corpus-driven differential and it is why the fixed entries exist at all: a
purely random arithmetic grammar would never have written ``x.at[0].add(5.0)``,
and this property would have been green through the entire defect. Not covered
either: the notes and stamps (compared nowhere), the solver legs, or any
difference that both series get equally wrong.

POSITIVE CONTROL: ``8ef8f75``. Run it with

    python tools/property_check.py --control cross-series \\
        --python <venv with jax 0.11> --other-python <venv with jax 0.10>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("hypothesis", reason="needs hypothesis")
jax = pytest.importorskip("jax", reason="needs jax")

import _corpus  # noqa: E402

_OTHER = "STELLING_PROPERTY_OTHER_PYTHON"


def _series(version: str) -> str:
    return ".".join(version.split(".")[:2])


def test_the_two_tested_jax_series_agree():
    other = os.environ.get(_OTHER)
    if not other:
        pytest.skip(
            "needs a second interpreter with the other jax series "
            "(set STELLING_PROPERTY_OTHER_PYTHON)"
        )
    import stelling
    from stelling._optional import TESTED_JAX_SERIES

    proc = subprocess.run(
        [other, _corpus.__file__],
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert proc.returncode == 0, (
        f"the other interpreter failed to produce a ledger:\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    )
    there = json.loads(proc.stdout)

    # (1) the two interpreters must actually differ.
    assert there["jax"] != jax.__version__, (
        f"BOTH INTERPRETERS ARE jax {jax.__version__}. A cross-series property "
        f"run against one series compares a thing to itself and can only pass; "
        f"point {_OTHER} at the other series."
    )
    # (2) both must be series this tree claims.
    for v in (jax.__version__, there["jax"]):
        assert _series(v) in TESTED_JAX_SERIES, (
            f"jax {v} is series {_series(v)}, which TESTED_JAX_SERIES "
            f"({TESTED_JAX_SERIES}) does not claim; this comparison would be "
            f"measuring something the tree never promised"
        )
    # (3) both must be reading the same stelling.
    assert there["stelling_file"] == stelling.__file__, (
        f"the two interpreters imported DIFFERENT stelling trees:\n"
        f"  this : {stelling.__file__}\n  other: {there['stelling_file']}\n"
        f"that is a comparison of two trees, not of two jax series"
    )

    here = _corpus.ledger()
    other_ledger = there["ledger"]

    only_here = sorted(set(here) - set(other_ledger))
    only_there = sorted(set(other_ledger) - set(here))
    assert not (only_here or only_there), (
        "the two interpreters built DIFFERENT corpora, so nothing below can be "
        f"compared:\n  only here ({len(only_here)}): {only_here[:2]}\n"
        f"  only there ({len(only_there)}): {only_there[:2]}"
    )

    disagreements = [
        (key, here[key], other_ledger[key])
        for key in sorted(here)
        if here[key] != other_ledger[key]
    ]
    # The floor: a corpus that stopped producing verdicts would agree perfectly.
    decided = sum(1 for v in here.values() if v["status"] is not None)
    assert decided >= 40, (
        f"GENERATOR FLOOR NOT MET — only {decided} of {len(here)} corpus "
        f"entries produced a verdict at all, so an agreement between the two "
        f"series would be an agreement about nothing"
    )

    assert not disagreements, (
        f"THE TWO TESTED jax SERIES DISAGREE on {len(disagreements)} of "
        f"{len(here)} harnesses (this = jax {jax.__version__}, other = jax "
        f"{there['jax']}), while TESTED_JAX_SERIES claims both:\n"
        + "\n".join(
            f"--- {k}\n    jax {jax.__version__}: {a}\n    jax {there['jax']}: {b}"
            for k, a, b in disagreements[:3]
        )
    )


if __name__ == "__main__":  # pragma: no cover - convenience
    sys.exit(pytest.main([__file__, "-ra"]))
