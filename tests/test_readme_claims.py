# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The README can't-drift test: capability tokens must have witnesses.

The README once claimed "lowers jaxpr to SMT queries to mathematically
prove…" for as long as it stood, and nothing caught it — no verdict has
ever touched an SMT query. The artifact that makes claims about the one
thing we fully control had a convention ("someone will notice") where the
census has a test. This is the test, per the can't-drift rule: *invariants
that must not drift get a test that they can't, not a convention that they
shouldn't.*

Mechanism: each capability token maps to a **witness** that must exist in
the code/tests for the token to be claimable. A token inside an
explicitly-marked exempt region (`<!-- capability-exempt: ... -->` …
`<!-- /capability-exempt -->`, used for the roadmap and the measured
"doesn't" list) is exempt — that is the whole point of marking it.

Two honest limits, by construction:

* **It catches the tokens listed here, not all overclaiming.** Partial.
* **It catches overclaiming, not underclaiming** — which is correct:
  overclaiming is the direction that hurts, and it is the direction that
  already bit. Absence/negation phrasings ("solver-free", "recorded
  absence") are therefore let through by a small negation window.

Scope: the capability-description region — everything before the
`## Installation` heading. Installation / Development / License are
packaging and legal prose; "pip install stelling[solvers]" is not a claim
that stelling verifies via a solver. Drift into capability claims happens
in the capability region, which is what this guards.
"""

from __future__ import annotations

import pathlib
import re

from stelling.propagate import TRANSFERS

REPO = pathlib.Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")

EXEMPT = re.compile(
    r"<!--\s*capability-exempt.*?-->.*?<!--\s*/capability-exempt\s*-->",
    re.S,
)
# Negation/absence cues that turn a token mention into a non-claim; kept
# tight because the capability region has few token occurrences.
_CUES = ("-free", "free of", "no solver", "without", "absence", "recorded absence")


def capability_region() -> str:
    """The README above `## Installation`, with exempt regions removed."""
    head = README.split("\n## Installation", 1)[0]
    return EXEMPT.sub(" ", head)


def _src_text() -> str:
    return " ".join(
        p.read_text(encoding="utf-8") for p in (REPO / "src" / "stelling").glob("*.py")
    )


def _witness_solver() -> bool:
    """The tool has a producing path for a solver-backed verdict — i.e.
    src constructs ``SolverStamp(... invoked=True ...)``.

    Read of the rule "a test produces SolverStamp(invoked=True)" that
    resists two false positives a literal-string scan of tests would hit:
    a unit test that constructs ``invoked=True`` only to assert it *raises*
    (validation, not a real invocation), and this file's own mention of
    the token. A construction in src is the capability actually existing;
    today there is none (`make_verdict` always emits `solver_absent`)."""
    return bool(re.search(r"SolverStamp\([^)]*invoked=True", _src_text()))


def _witness_derive() -> bool:
    """E2b machinery exists (a widening / fixpoint / derive path). Proxy:
    a defining occurrence in src; there is none today."""
    return bool(re.search(r"\b(widen\w*|fixpoint|derive_invariant)\b", _src_text()))


def _witness_discrete() -> bool:
    """A verdict exists that does not carry the continuous-flow demotion —
    proxy: a discrete-step capability marker in src; none today."""
    return "discrete_step" in _src_text()


def _negated(region: str, start: int, end: int) -> bool:
    window = region[max(0, start - 24) : end + 40].lower()
    return any(cue in window for cue in _CUES)


# token -> (compiled pattern, witness predicate, human name)
CLAIMS = [
    (re.compile(r"\b(SMT|Z3|cvc5|solver)\b", re.I), _witness_solver, "solver/SMT"),
    (re.compile(r"\b(deriv\w*|infer\w*)\b", re.I), _witness_derive, "derive/infer"),
    (re.compile(r"`(cond|scan|while)`"), lambda t=None: False, "control-flow"),
    (re.compile(r"\bdiscrete[ -]step"), _witness_discrete, "discrete-step"),
]


def test_readme_makes_no_unwitnessed_capability_claim():
    region = capability_region()
    violations = []
    for pattern, witness, name in CLAIMS:
        for m in pattern.finditer(region):
            if _negated(region, m.start(), m.end()):
                continue
            # control-flow tokens carry their witness per-token (in TRANSFERS)
            if name == "control-flow":
                if m.group(1) in TRANSFERS:
                    continue
                violations.append(f"`{m.group(1)}` claimed but not in propagate.TRANSFERS")
                continue
            if not witness():
                violations.append(
                    f"{name!r} token {m.group(0)!r} appears as a capability claim, "
                    f"but its witness does not exist"
                )
    assert not violations, (
        "README capability claim(s) without a witness (fence roadmap/planned "
        "claims with <!-- capability-exempt -->):\n  " + "\n  ".join(violations)
    )


def test_the_test_actually_bites(monkeypatch):
    """The positive control for this control: the exact drift that shipped
    must fail against a src corpus with no solver witness — checked against
    a synthetic corpus, so the control keeps biting now that the real src
    HAS earned the witness (the escalation layer constructs
    SolverStamp(invoked=True, …))."""
    old_drift = "Stelling lowers jaxpr to SMT queries to mathematically prove invariants."
    m = re.search(r"\b(SMT|Z3|cvc5|solver)\b", old_drift, re.I)
    assert m is not None
    assert not _negated(old_drift, m.start(), m.end())  # no negation cue
    # the real src now witnesses the capability (the drift line would pass
    # today, correctly) …
    assert _witness_solver()
    # … so the control replays history: against a corpus WITHOUT the
    # construction, the witness is absent and the drift line would fail.
    synthetic_src = (
        "def make_verdict():\n"
        "    return solver_absent('no solver invoked')\n"
    )
    import sys

    monkeypatch.setattr(sys.modules[__name__], "_src_text", lambda: synthetic_src)
    assert not _witness_solver()


def test_control_flow_witness_tracks_transfers():
    # if any of cond/scan/while were added to TRANSFERS, the README could
    # claim it; today none are, so a claim would be caught
    assert not ({"cond", "scan", "while"} & set(TRANSFERS))
