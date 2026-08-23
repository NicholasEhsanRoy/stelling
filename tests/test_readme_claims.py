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

Scope OF THE CAPABILITY SCAN: the capability-description region —
everything before the `## Installation` heading. Installation /
Development / License are packaging and legal prose; "pip install
stelling[solvers]" is not a claim that stelling verifies via a solver.
Drift into capability claims happens in the capability region, which is
what that scan guards.

**This file holds four claim families now, and only the first has that
scope.** Below the capability scan are (2) the README's INTERPRETER claims
— the python badge and `### Which Python`, held to `pyproject.toml` and to
`.github/workflows/`, which live under `## Installation` deliberately —
(3) the `--stelling-*` DIAL spellings the coverage paragraph names,
resolved against the plugin's own registration, and (4) the two spellings
that paragraph says still reach VERIFIED, held to the uncovered list the
tool prints. Each block states its own defect and its own scope; nothing
here reads another block's region.
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


# ── the interpreter claim: a badge, and the two facts it used to conflate ───
#
# THE DEFECT. `README.md:4` read
#
#     ![python: 3.12, the version CI measures](…/badge/python-3.12%20tested…)
#
# and nothing held it. Measured: `requires-python = ">=3.10"`; exactly one job
# in `.github/workflows/` names an interpreter, and `ci.yml`'s own comment says
# why — *"Python is PINNED here where every other lane takes the runner
# default"* — for a dependency reason rather than a coverage decision; and
# nothing runs the declared floor. So the badge asserted a TESTED version while
# the version it named was whatever `ubuntu-latest` happened to ship: true on
# the day it was written, held by nothing afterwards. That is the same
# environment-dependence that reddened `main` three times in one week, sitting
# in the first line a reader sees.
#
# The badge now names the FLOOR, which `pyproject.toml` holds and pip enforces,
# and the README says separately what CI actually provisions. This is the gate
# on both, because replacing an unheld string with a different unheld string is
# the same defect wearing new words.
#
# WHY IT LIVES IN THIS FILE rather than with `tests/test_lanes.py`'s fences:
# this is a README can't-drift check, the same shape as the capability scan
# above — a claim in the artefact that makes claims, held to a witness. The
# WITNESSES are read by `_lanes.python_provisioning()` (the workflows) and by
# `test_zero_dep_import_discipline._declared_floor()` (`requires-python`), each
# already the single reader of its file.
#
# THE DOCSTRING'S SCOPE PARAGRAPH IS THE CAPABILITY SCAN'S AND NOT THIS
# BLOCK'S. That scan stops at `## Installation`; `### Which Python` sits under
# it, which is where a reader deciding whether to install asks the question and
# is not a capability claim. Nothing here reads the capability region and
# nothing there reads this section.

import _lanes
from test_zero_dep_import_discipline import _declared_floor

#: The badge's message may not carry a coverage word. This is the defect
#: itself, not a style rule: `3.12 tested` was false-by-drift the moment the
#: runner image moved, and no job asserts the interpreter it got.
_COVERAGE_WORDS = ("tested", "measured", "verified", "supported", "passing")

#: A version this README may name that is neither the floor nor a pin, with
#: the source that makes it a fact. Held below against that source, so a
#: number cannot enter this section by being written in it.
_FOREIGN_VERSIONS = {
    "3.11": ('python_requires=">=3.11"', "jaxfluids' own floor, quoted from ci.yml"),
}

_NUMBER_WORDS = {"no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _floor() -> str:
    major, minor = _declared_floor()
    return f"{major}.{minor}"


def _python_badge() -> str:
    """The one badge line naming an interpreter version."""
    badges = [
        line
        for line in README.splitlines()
        if "img.shields.io/badge/python-" in line
    ]
    assert len(badges) == 1, f"expected one python badge, found {len(badges)}: {badges}"
    return badges[0]


def _which_python() -> str:
    """The `### Which Python` section, heading excluded."""
    _, _, rest = README.partition("\n### Which Python\n")
    assert rest, (
        "README.md has no `### Which Python` section. It is where the badge's "
        "claim is separated from the version CI runs; deleting it puts the "
        "conflation back."
    )
    # ends at the next heading of ANY level, not at the next `### `: a section
    # that ran on into the following `## ` would let a version named in
    # unrelated prose count as one of this section's claims.
    return re.split(r"\n#{1,6} ", rest, maxsplit=1)[0]


def _interpreter_violations(badge: str, section: str, floor: str, pins: dict) -> list[str]:
    """Every way the README's two interpreter claims can disagree with the
    files that hold them. Pure, so the control below can drive it on prose
    this repository does not ship."""
    bad = []
    versions_in_badge = set(re.findall(r"\d+\.\d+", badge))
    if versions_in_badge != {floor}:
        bad.append(
            f"the python badge names {sorted(versions_in_badge)} where the only "
            f"interpreter version this repository holds is the declared floor "
            f"{floor!r} (`requires-python`). Any other version in a badge is a "
            f"claim about the runner image."
        )
    for word in _COVERAGE_WORDS:
        if word in badge.lower():
            bad.append(
                f"the python badge says {word!r}. Nothing tests the floor and no "
                f"job asserts the interpreter it was given, so a coverage word "
                f"here is the defect this gate exists for."
            )
    if floor in pins.values():
        bad.append(
            f"a job now pins the floor {floor!r} — the README says none does. "
            f"That is good news for the floor and this section has to say so."
        )
    if floor not in section:
        bad.append(f"`### Which Python` never names the floor {floor!r}")
    for job, version in pins.items():
        name = job.split(":", 1)[1]
        if name not in section:
            bad.append(f"{job} pins python {version} and the README does not name it")
        if version not in section:
            bad.append(f"{job} pins python {version} and the README does not say so")
    counted = re.search(r"exactly (\w+) job", section, re.I)
    if counted is None:
        bad.append(
            "`### Which Python` states no count of pinning jobs. The count is "
            "the half a second pin would falsify."
        )
    elif _NUMBER_WORDS.get(counted.group(1).lower()) != len(pins):
        bad.append(
            f"the README says exactly {counted.group(1)} job(s) pin an "
            f"interpreter; the workflows say {len(pins)}: {sorted(pins)}"
        )
    allowed = {floor, *pins.values(), *_FOREIGN_VERSIONS}
    for version in sorted(set(re.findall(r"\d+\.\d+", section)) - allowed):
        bad.append(
            f"`### Which Python` names python {version}, which is neither the "
            f"declared floor, a pin any job carries, nor a recorded foreign "
            f"floor. A version in this section has to come from a file."
        )
    return bad


def test_the_interpreter_claims_are_what_pyproject_and_the_workflows_say():
    """The badge names the floor; the section names what CI provisions."""
    bad = _interpreter_violations(
        _python_badge(), _which_python(), _floor(), _lanes.python_pins()
    )
    assert not bad, "\n  ".join(
        ["the README's python claims and the files that hold them disagree:"] + bad
    )


def test_the_workflows_provision_the_interpreters_they_are_declared_to():
    """The measured/declared pin under the claim above.

    `python_pins()` answers *how many* jobs pin; this answers *which jobs
    provision an interpreter at all and how*, so a lane that stops creating an
    environment — or one whose spelling this module cannot read — is a line in
    a diff rather than a silently smaller denominator."""
    measured = _lanes.python_provisioning()
    assert measured == _lanes.EXPECTED_PYTHON, (
        "how CI provisions its interpreters moved.\n"
        f"  declared {_lanes.EXPECTED_PYTHON}\n"
        f"  measured {measured}\n"
        "An `unreadable:` reading is a spelling `_lanes.python_provisioning` "
        "does not know — read it by hand before declaring it, because the "
        "README's count of pinning jobs rests on this."
    )


def test_every_foreign_version_named_in_the_readme_comes_from_a_file():
    """`3.11` is in that section because jaxfluids declares it, and the
    section says so. Held to `ci.yml`'s text: a foreign floor that moves
    upstream must not keep standing here because it was once typed."""
    ci = (_lanes.CI).read_text(encoding="utf-8")
    for version, (evidence, why) in _FOREIGN_VERSIONS.items():
        assert evidence in ci, (
            f"{version} is allowed in `### Which Python` as {why}, on the "
            f"strength of {evidence!r} appearing in ci.yml. It does not."
        )


def test_the_interpreter_gate_actually_bites():
    """The positive control: the badge that shipped, and the four ways this
    pair can come apart.

    `floor` and `pins` are HISTORY — the values measured at this commit,
    written down rather than read, so that the replay keeps replaying the
    defect after the files move. The last assertion drives the LIVE badge and
    section against them, so a floor or a pin that moves reddens here as well
    as in the measured/declared tests above. That is deliberate: this control
    is a claim about a specific pair of facts, and it should stop passing when
    it stops describing them."""
    floor, pins = "3.10", {"ci.yml:acceptance-reproducer": "3.12"}
    section = _which_python()
    shipped = (
        "![python: 3.12, the version CI measures]"
        "(https://img.shields.io/badge/python-3.12%20tested-blue.svg)"
    )
    # the badge as it stood: wrong version, and a coverage word
    assert len(_interpreter_violations(shipped, section, floor, pins)) >= 2
    # a badge naming the floor with a coverage word is still the defect
    assert _interpreter_violations(
        "![python](…/badge/python-%3E%3D3.10%20tested-blue.svg)", section, floor, pins
    )
    # a second job starts pinning and the README still says one
    assert _interpreter_violations(
        _python_badge(), section, floor, {**pins, "ci.yml:test-jax": "3.13"}
    )
    # the pin moves under the sentence
    assert _interpreter_violations(
        _python_badge(), section, floor, {"ci.yml:acceptance-reproducer": "3.13"}
    )
    # and the shipped pair is clean, which is what makes the four above mean
    # something
    assert not _interpreter_violations(
        _python_badge(), section, floor, pins
    )


# ── the dials the README names have to be dials ─────────────────────────────
#
# The coverage paragraph names three instruments and the flag that arms each,
# where it named one before. A flag spelled in the first artefact a reader
# meets is a promise about a command line: if it is renamed, or its value
# spelling changes, the README sends a reader to a `pytest` that exits 4 with
# `unrecognized arguments`. So the spellings are resolved against the
# registration rather than restated — the same argument as the `pytest11`
# entry-point check in `tests/test_tripwire_plugin.py`, one artefact along.
#
# It also holds the OFF-BY-DEFAULT half, which is the load-bearing part of
# "each is a separate opt-in dial and none turns on either of the others": a
# default that quietly became `error` would make the paragraph false in the
# direction that costs a reader a broken suite.


def _registered_options() -> dict[str, dict]:
    """Every stelling command-line option, as the plugin registers it."""
    import stelling._tripwire.plugin as plugin

    recorded: dict[str, dict] = {}

    class _Group:
        def addoption(self, *names, **kwargs):
            for name in names:
                recorded[name] = kwargs

    class _Parser:
        def getgroup(self, _name):
            return _Group()

    plugin.pytest_addoption(_Parser())
    return recorded


def test_every_dial_the_readme_spells_is_a_dial_that_exists():
    options = _registered_options()
    spelled = set(re.findall(r"--stelling-[a-z-]+(?:=[a-z]+)?", README))
    assert spelled, "the README names no stelling flag — this check is vacuous"
    bad = []
    for spelling in sorted(spelled):
        flag, _, value = spelling.partition("=")
        if flag not in options:
            bad.append(f"{flag} is in README.md and is not a registered option")
            continue
        choices = options[flag].get("choices") or ()
        if value and value not in choices:
            bad.append(
                f"README.md writes `{spelling}`; {flag} takes {list(choices)}"
            )
    assert not bad, (
        "the README sends a reader to a command line that does not exist "
        "(`pytest` exits 4 on an unrecognized argument):\n  " + "\n  ".join(bad)
    )


def test_the_two_further_dials_are_off_until_asked_for():
    """The README says each instrument is separately opted into. Both of the
    two it added are `default="off"`, and neither is reachable by arming the
    other — which is what makes "three instruments" a description of what a
    reader gets rather than of what is installed."""
    options = _registered_options()
    for flag in ("--stelling-eager-truncation", "--stelling-narrowing-perimeter"):
        assert flag in options, (
            f"{flag} is no longer registered at all, and README.md names it as "
            f"one of the three instruments a reader can arm"
        )
        assert options[flag].get("default") == "off", (
            f"{flag} no longer defaults to off, and README.md tells a reader "
            f"it does"
        )
        assert "error" in (options[flag].get("choices") or ()), (
            f"{flag} no longer takes `error`, which is the value README.md "
            f"spells"
        )


# ── the residual the README keeps, held to the list the tool prints ─────────
#
# The coverage paragraph names two spellings that reach VERIFIED with all
# three instruments armed. That is the honest half of a paragraph that would
# otherwise read as a coverage claim, and it is the half most likely to rot:
# it is a statement about what is NOT watched, so nothing fails when it
# becomes obsolete. `report.PERIMETER_UNCOVERED` is the same statement in the
# form the tool prints at the end of every armed run — the artefact a reader
# is sent to check — so the two are held together. A door that closes has to
# be closed in both places, and a README that goes on advertising a hole the
# release no longer has is caught here.
#
# Driven at this commit before it was written down, jax 0.11.0 and 0.10.2, x64
# on and off, all three dials armed: `assert_(x <= 2**31 - 1)` raises
# `NarrowingError`; `assert_(jnp.less_equal(x, 2**31 - 1))` and
# `assert_((x - (2**31 - 1)) <= 0.0)` are both VERIFIED.

#: (token in the printed bullet, the README's spelling of the same door)
_README_RESIDUALS = (
    ("jnp.less_equal(x, N)", "jnp.less_equal(x, 2**31 - 1)"),
    ("(x - N) <= 0.0", "(x - (2**31 - 1)) <= 0.0"),
)


def test_the_residual_the_readme_names_is_the_one_the_run_prints():
    from stelling._tripwire.report import PERIMETER_UNCOVERED

    bad = []
    for printed_token, readme_spelling in _README_RESIDUALS:
        bullets = [b for b in PERIMETER_UNCOVERED if printed_token in b]
        if not bullets:
            bad.append(
                f"the printed uncovered list no longer names {printed_token!r}, "
                f"and README.md still says {readme_spelling!r} reaches VERIFIED"
            )
        elif not any("VERIFIED" in b for b in bullets):
            bad.append(
                f"the bullet naming {printed_token!r} no longer says VERIFIED, "
                f"which is what README.md claims of {readme_spelling!r}"
            )
        if readme_spelling not in README:
            bad.append(
                f"README.md no longer names {readme_spelling!r}. The paragraph "
                f"claims the watched set is still finite with all three armed; "
                f"the two spellings are what makes that concrete rather than a "
                f"disclaimer"
            )
    assert not bad, "\n  ".join(["README.md and the printed residue disagree:"] + bad)
