# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""WHY `VERIFIED_BARRED_PRIMITIVES` CONTAINS WHAT IT CONTAINS — the decision,
not its current value.

`tests/test_verified_bar.py` pins that the bar FIRES and that it fires only
where intended. Nothing pinned the MEMBERSHIP, and a set whose contents nobody
has to justify drifts in whichever direction is convenient: adding a row is
free to write and expensive to everyone downstream, removing one is free to
write and unrecoverable if wrong. This file is the justification, made
checkable.

WHAT THE STANDING RULE ACTUALLY CLAIMS, read at its definition site. The
lifting condition `stelling.verdict` states — three times, in three places, and
identically each time — is **"until the emission row has been attacked by a
distinct-context adversarial auditor"**: in the block comment's first
paragraph, in the paragraph that excludes `scatter-add`, and in
:data:`stelling.verdict.VERIFIED_BAR_REASON`, which every withheld verdict
quotes to its reader. It is NOT "until the row has a fidelity gauge" — no
gauge is named as the condition anywhere — and it is NOT specific to
`scatter`: the asymmetry it rests on ("a spurious witness is caught by
exact-rational replay; a MISSED violation has no downstream check") is a
property of every discharge-only emission row.

`scatter-add` is the worked example and it settles the reading. It is a
discharge-only row with the same asymmetry, it is NOT in the set, and the
reason the comment gives is that `design/scatter-rows.md` records a COMPLETED
fresh-adversarial-auditor pass over it. So the rule is **barred while
UNATTACKED**, and what retires the bar is the auditor's report.

HOW THAT APPLIES TO THE TWO ROWS 0.2.0 ADDED. `pow` and `is_finite` are both
discharge-only and neither is in the set. Both have now had exactly the pass
the rule names — the 0.2.0 pre-release audit, a distinct-context adversarial
audit with a dedicated `pow` lens. Its `pow` findings are in `SOUNDNESS.md`
(the rational-`pow` exponent substitution, the unary `(* aux)`, the float
replay) and every one is fixed with a permanent regression in
`tests/test_pow_audit_findings.py`; its `is_finite` probe measured the transfer
and the emission agreeing in both directions and found nothing unsound.

THE COUNTER-ARGUMENT, STATED BECAUSE IT IS THE STRONGEST ONE AGAINST THIS
DECISION. `scatter-add`'s pass cleared it with ZERO UNSOUND. `pow`'s pass found
three false-VERIFIED-class defects, and the REPAIRS have not themselves had a
distinct-context pass. "Attacked" is satisfied for both rows; "attacked and
found clean" is satisfied only for one, and it is the second that the rule's
purpose is about.

WHY THE DECISION IS NEVERTHELESS NOT TO BAR, and it turns on what a bar buys
against what a gauge buys. The bar's own text says what it buys: *"Under this
bar the worst case of a wrong row is a witness that fails replay. Without it
the worst case is silent."* A bar makes a wrong row's discharges UNUSABLE; it
does not make them VISIBLE. `tests/test_pow_row_gauge_jax.py` makes them
visible: a battery of deliberate wrongnesses of this row's encoding whose floor
is `assert len(muts) >= 20` — the gauge's own anti-vacuity assertion, and the
exact string this file checks for below, so the figure written here cannot
drift from the one that runs — with zero survivors, and one of them, a single
auxiliary constant shared between two elements of a vectorised `pow`, silently
DISCHARGES an obligation that is false at `x = [4, 1]`, which is exactly the
missed violation the bar exists to contain and which nothing in this tree
caught before. For a row that has been attacked and repaired, the gauge is the
stronger instrument, and it is the one that would have caught the repairs going
wrong. A bar is not a gauge.

THAT LAST SENTENCE WAS OVER-CLAIMED, AND THE CORRECTION IS WHY IT CAN BE
TRUSTED NOW. "Would have caught the repairs going wrong" is a claim about the
repairs AT THE EXPONENTS THE BATTERY DRIVES, and a blinded audit measured
those: the shipped battery reached integer exponents `{-2, 3}` and the single
pair `(1, 2)`. It then wrote three wrongnesses conditioned outside that set —
wrong only above degree 3, wrong only for `q >= 4`, wrong only for `p != 1` —
and all three passed every gate, each turning a genuinely REFUTED query into
VERIFIED on an exponent the admission guard admits. A CONDITIONAL repair is
exactly the shape a repaired row regresses in, so the sentence was claiming the
property it was least able to support: "both exponent branches" is true of the
branches and says nothing about the exponents.

The battery now drives integer exponents `[-4, -2, 3, 5]` and the pairs `1/2`,
`1/4`, `3/2`; all three of those mutations are IN it, so the widening is
pinned rather than done once; and the arity is measured at the seams rather
than asserted in prose
(`test_the_driven_arity_is_MEASURED_at_the_seams_not_asserted_in_prose`). The
claim above therefore stands at four integer exponents of both signs and three
of the 448 admitted `(p, q)` pairs. That bound is stated instead of "both
branches" because "both branches" is what was true while three mutations were
getting through.

`is_finite` is a weaker case still, and for a reason that is about the row
rather than about process: under the semantics the verdict CLAIMS, emitting
`true` is not merely un-self-checking, it is RIGHT. `isfinite(v)` is a
tautology over ℝ, so there is no free parameter in the discharge direction for
a wrong encoding to move. The guard on the propagated interval is
belt-and-braces for a reader's IEEE intuition, and the ℝ/float gap it is about
is the one the stamp already discloses. Measured below, both directions.

THE COST, MEASURED. Environment: jax 0.11.0, python 3.12.3,
`/home/nick/venvs/stelling-jax`, z3 wheel + cvc5 wheel, x64 enabled,
2026-08-16, re-measured on the tree that ships this file.

* **The repository's own suite** — the population of queries this project
  actually verifies, 3798 passed / 10 skipped with x64 and 3799 / 9 without.
  With `pow` added to the set: **21 RED, of which 9 are pre-existing tests**,
  every one of those nine a `pow`-bearing VERIFIED becoming UNKNOWN (8 in
  `tests/test_0_2_0_regression.py`, 1 in `tests/test_pow_audit_findings.py`).
  Two of the remaining twelve are this file's own detectors below, which is
  what they are for; the other **10 are demonstration assertions inside
  `tests/test_pow_row_gauge_jax.py`** whose per-item lines read
  `check().status`, so a withheld VERIFIED reddens them by construction — the
  BATTERY itself reads identically barred and unbarred (32 mutations, 0
  survivors either way, every catch set unchanged). **This paragraph counted
  only the nine for two rounds**, which is why the breakdown is now given in
  full: 9 + 2 + 10. With `is_finite` added instead: **2 RED, both of them this
  file's detectors, and 0 pre-existing tests** — the suite is otherwise fully
  green.

  *THE BASELINE IS THE MERGED TREE'S AND THE COST IS NOT. This bullet read
  "3501 passed / 10 skipped with x64 and 3502 / 9 without" — B7's tree,
  before B6 merged — and the baseline is repinned on the merge (parents
  `198a2b5` and `dd95333`). The COST
  figures were RE-MEASURED there, not carried: `+pow` is still 21 red and
  still 9 + 2 + 10, the same named tests, and `+is_finite` is still 2 red
  and 0 pre-existing. B6 added 297 tests and not one of them puts `pow` or
  `is_finite` on a solver-decided slice, so the cost is unmoved while the
  population it is a cost against grew by 8.5%. Both flips were driven on
  clean clones of the merge with only `VERIFIED_BARRED_PRIMITIVES` edited.*
* **Two purpose-built harnesses in this file** (`_pow_harness`,
  `_is_finite_harness`) that put the row on a solver-decided slice. A wider
  dated measurement, taken off-tree and NOT reproducible from this checkout —
  recorded as such, the way the corpus bullet below carries its own caveat — had `+pow` withhold
  4 of 4 `pow`-bearing VERIFIEDs and `+is_finite` withhold 1 of the 2
  `is_finite` harnesses (the second never escalates).
* **The fourteen MADDENING/MIME contracts**: 0 of 7 VERIFIEDs lost under every
  candidate set — AND THAT NUMBER MEANS NOTHING ABOUT THE BAR. Not one of the
  fourteen has a solver-decided obligation at all (the only invocations
  anywhere are `MADD projection`'s vacuity widen re-check), and not one
  contains `pow`. A corpus that never reaches the mechanism cannot price it,
  and reporting its zero without that sentence would be the misuse this file
  exists to prevent.

So `+pow` costs nine known-good verdicts in this tree and every `pow`-bearing
VERIFIED downstream; `+is_finite` costs nothing measurable here and is not free
in general. The two live measurements below are the ones that cannot rot: they
flip the set and watch a verdict move.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import stelling.obligation as OB  # noqa: E402
import stelling.smt as SM  # noqa: E402
import stelling.verdict as V  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import interval_env, propagate  # noqa: E402
from stelling.solvers import SolverConfig, escalate  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
TIMEOUT_MS = 20_000


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# THE DECISION, as data. Every discharge-affecting emission row that has ever
# been considered for this set, with its verdict and the evidence that verdict
# rests on. A row added to `_SUPPORTED` and absent from here fails
# `test_every_discharge_affecting_row_has_a_recorded_decision`, which is the
# point: the next row cannot ship without someone writing the sentence.
BARRED = "IN — no completed distinct-context adversarial pass over this row"
CLEARED = "OUT — the pass the rule names has completed"

DECISIONS: dict[str, str] = {
    "scatter": BARRED,
    "scatter-add": CLEARED,
    "pow": CLEARED,
    "is_finite": CLEARED,
}


def _pow_harness():
    """A `pow`-bearing query that VERIFIES today: `sqrt(x) <= 2` over [1, 4]
    holds at the endpoint, and the correlation `+ (x - x)` keeps the interval
    domain from settling it, so a solver is what discharges it."""

    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return (assert_(jnp.power(x, 0.5) + (x - x) <= 2.0),)

    return h


def _is_finite_harness():
    """An `is_finite`-bearing query that VERIFIES today. `x - x <= 0` is
    invisible to the interval domain, so the conjunction escalates and the
    `is_finite` row is on the emitted slice."""

    def h():
        x = any_array((), "float64", (0.0, 1.0))
        return (assert_(jnp.logical_and(jnp.isfinite(x * x), x - x <= 0.0)),)

    return h


def _verdict_under(barred, harness, monkeypatch):
    monkeypatch.setattr(V, "VERIFIED_BARRED_PRIMITIVES", frozenset(barred))
    return check(harness, vacuity_mode="inputs-only",
                 solver_timeout_ms=TIMEOUT_MS)


# --- the rule, read at its definition site -----------------------------------


def test_the_lifting_condition_the_SOURCE_states_is_an_ADVERSARIAL_PASS():
    """PIN THE RULE, not the membership. Every test below applies a condition;
    this one asserts what the condition IS, out of the source that states it,
    so that changing the rule is a visible act rather than a re-reading.

    Both halves matter. The phrase must be there — it is what a reader of a
    withheld verdict is told — and the alternative reading this batch
    considered and rejected ("barred until GAUGED") must NOT be there, because
    a set justified by two different rules is justified by neither."""
    reason = V.VERIFIED_BAR_REASON
    assert "attacked by a distinct-context adversarial auditor" in reason
    assert "gauge" not in reason.lower(), (
        "VERIFIED_BAR_REASON now names a gauge as the lifting condition. That "
        "is a different rule from the one this file's decisions were derived "
        "under — re-derive them rather than editing this assertion"
    )
    source = inspect.getsource(V)
    block = source[source.index("THE SCATTER VERIFIED BAR"):
                   source.index("VERIFIED_BARRED_PRIMITIVES = ")]
    assert "distinct-context adversarial auditor" in block
    assert "TO LIFT:" in block
    # the block is a comment, so the sentence is wrapped and `# `-prefixed;
    # compare on the prose rather than on the layout
    flat = " ".join(block.replace("#", " ").split())
    assert "It is the principal's to lift, after the auditor reports." in flat, (
        "the block no longer says WHO lifts it and on WHAT report; the "
        "membership decisions in this file rest on that sentence"
    )


def test_the_barred_set_is_exactly_the_rows_with_no_completed_pass():
    """The membership, DERIVED from :data:`DECISIONS` rather than restated.
    A row whose decision changes must have its decision changed here, where
    the sentence justifying it lives."""
    expected = frozenset(k for k, v in DECISIONS.items() if v is BARRED)
    assert V.VERIFIED_BARRED_PRIMITIVES == expected, (
        f"the bar's membership {sorted(V.VERIFIED_BARRED_PRIMITIVES)} no "
        f"longer matches the decisions recorded in this file "
        f"({sorted(expected)}). Neither is authoritative over the other: the "
        f"set is the mechanism and this file is the argument, and they are "
        f"required to agree so that a change to one is a change to both"
    )
    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"


def test_every_discharge_affecting_row_has_a_recorded_decision():
    """ANTI-DRIFT, and the reason this file is a policy file rather than a
    constant. Any emission row can mint a discharge, so any emission row is a
    candidate for this set; a new one arriving with no recorded decision is the
    silence the 0.2.0 audit's S4 was about — two rows shipped, neither barred
    and neither argued.

    Scoped to the rows this file has actually reasoned about: the four in
    :data:`DECISIONS` must all still be emission rows, and any row added to
    the barred set without a decision here fails. It does NOT demand a
    sentence for all 36 emission rows — that would be a list nobody maintains
    and the audit's finding was about the ones a ROUND ADDS."""
    for name in DECISIONS:
        assert name in OB._SUPPORTED, (
            f"{name!r} is no longer an emission row, so the decision recorded "
            f"for it is about nothing"
        )
    undecided = set(V.VERIFIED_BARRED_PRIMITIVES) - set(DECISIONS)
    assert not undecided, (
        f"{sorted(undecided)} are barred with no recorded reason — write the "
        f"sentence in DECISIONS, or the next reader cannot tell a deliberate "
        f"bar from a leftover one"
    )


def test_scatter_add_is_out_because_its_pass_is_RECORDED_in_the_tree():
    """The worked example the reading rests on, checked against the record
    rather than against the comment that cites it."""
    record = REPO / "design" / "scatter-rows.md"
    text = record.read_text(encoding="utf-8")
    assert "PASS RECORD" in text
    assert "adversarial auditor" in text
    assert "zero UNSOUND" in text, (
        f"{record} no longer records a completed clean adversarial pass, so "
        f"`scatter-add`'s exclusion from the bar has lost its evidence"
    )
    assert "scatter-add" not in V.VERIFIED_BARRED_PRIMITIVES


def test_pow_is_out_because_its_pass_completed_AND_the_row_is_now_GAUGED():
    """`pow`'s two pieces of evidence, both in this tree.

    The PASS: the 0.2.0 audit's `pow` findings are logged in SOUNDNESS.md and
    each carries a permanent regression. The GAUGE, which is what answers the
    counter-argument in the module docstring — the repairs themselves are now
    measured by mutation rather than trusted."""
    soundness = (REPO / "SOUNDNESS.md").read_text(encoding="utf-8")
    assert "the rational-`pow` row emitted about a" in soundness, (
        "SOUNDNESS.md no longer logs the audit's rational-`pow` finding, "
        "which is the in-tree record that the pass happened"
    )
    regressions = (REPO / "tests" / "test_pow_audit_findings.py").read_text(
        encoding="utf-8"
    )
    for finding in ("S1", "S2", "S3"):
        assert f"--- {finding}" in regressions or f"{finding}:" in regressions, (
            f"the permanent regression for audit finding {finding} is gone"
        )

    gauge_path = REPO / "tests" / "test_pow_row_gauge_jax.py"
    assert gauge_path.exists(), (
        "the `pow` row's fidelity gauge is gone, and it is half the argument "
        "for leaving `pow` out of the bar"
    )
    # READ AS SOURCE, NOT IMPORTED. Importing it would run its module-level
    # `importorskip("z3")`/`("cvc5")`, so in an environment with jax and no
    # solver wheel this decision test would become a SKIP — a decision nobody
    # is told stopped being checked, which is what
    # `tests/test_skip_inventory.py` exists to prevent. The gauge's own
    # anti-vacuity assertion is what pins the battery SIZE; this pins that the
    # battery still contains the mutations the decision was argued from.
    battery = gauge_path.read_text(encoding="utf-8")
    assert "assert len(muts) >= 20" in battery, (
        "the gauge no longer asserts its own battery size, so this file "
        "cannot delegate that pin to it — and the floor quoted in this "
        "module's docstring is that literal string, so the two move together"
    )
    for named in ('"emit-rational-sides-swapped"',
                  '"emit-rational-root-guard-dropped"',
                  '"emit-rational-constraint-never-asserted"',
                  '"emit-rational-denominator-off-by-one"',
                  '"emit-integer-off-by-one"',
                  '"emit-integer-loses-the-reciprocal"',
                  '"replay-exponent-inverted"'):
        assert named in battery, f"{named} is gone from the battery"
    assert '"emit-rational-one-aux-for-two-elements"' in battery, (
        "the missed-violation mutation is gone from the battery — it is the "
        "one that shows the gauge reaches the direction the bar protects, "
        "and it is the strongest single sentence in the argument for not "
        "barring this row"
    )
    # The four CONDITIONAL mutations, which are the ones that answer the
    # counter-argument above rather than restating it. A repaired row
    # regresses by being fixed at the exponent someone tested and left wrong
    # at the general one; all four were written by a blinded auditor, all four
    # survived the battery this file used to argue from at the time they were
    # written, and each minted a real false VERIFIED. Without them in the
    # battery the docstring's claim is back to the one that was measured
    # false. THE FOURTH IS ON THE SHAPE AXIS, not the exponent one — correct
    # at elements 0 and 1 and wrong from element 2 on — and it is here because
    # the first three being caught is what made the shape gap the next thing
    # to survive: this list is a record of where the reach has been shown to
    # end, not a claim that it now ends nowhere.
    for conditional in ('"emit-integer-wrong-only-above-degree-three"',
                        '"emit-rational-wrong-only-at-a-larger-denominator"',
                        '"emit-rational-wrong-only-at-a-numerator-past-one"',
                        '"emit-rational-wrong-only-past-the-second-element"'):
        assert conditional in battery, (
            f"{conditional} is gone from the battery — that is the "
            f"conditional-wrongness shape the audit got past every gate, and "
            f"this file's decision not to bar `pow` rests on it being gauged"
        )


def test_is_finite_is_out_because_the_row_cannot_be_wrong_in_that_DIRECTION():
    """MEASURED, not argued: the `is_finite` row emits the constant `true`,
    its replay returns `True`, and under the ℝ semantics the verdict claims
    both are the right answer for every real. There is no value the row could
    emit instead that would MISS a violation — the only other emission is
    `false`, which can only produce a witness, and every witness is re-checked
    by exact-rational replay.

    The guard on the propagated interval is exercised in the other direction
    as the audit measured it: a box whose square saturates to infinity makes
    the real-mode escalation DECLINE with the operand named."""
    closed = trace(_is_finite_harness())
    p = propagate(closed)
    (item,) = OB.slice_unknown_obligations(closed, p, interval_env(closed))
    assert not isinstance(item, OB.DeclinedObligation), item.reason
    text = SM.emit(item, "z3", TIMEOUT_MS).text
    finite_lines = [ln for ln in text.splitlines()
                    if "define-fun" in ln and "Bool true" in ln]
    assert finite_lines, (
        f"the is_finite row no longer emits the constant `true`, so the "
        f"argument for leaving it out of the bar is about a different row:\n"
        f"{text}"
    )

    # the guard's own direction, quoted from the run rather than paraphrased
    def saturating():
        x = any_array((), "float64", (1e200, 1e300))
        return (assert_(jnp.isfinite(x * x)),)

    sat_closed = trace(saturating)
    sat_p = propagate(sat_closed)
    esc = escalate(sat_closed, sat_p, SolverConfig(timeout_ms=TIMEOUT_MS))
    detail = " ".join(r.detail or "" for r in esc.records)
    assert "'is_finite' operand interval has non-finite endpoints" in detail, (
        f"the real-mode guard no longer declines a saturating operand: "
        f"{detail}"
    )


# --- the cost, MEASURED here rather than estimated ---------------------------


def test_the_cost_of_barring_pow_is_MEASURED_here(monkeypatch):
    """A currently-VERIFIED `pow` query, and the same query with `pow` in the
    set. The point is that the cost is a live measurement in the same file as
    the decision, so nobody can quote the decision without the price beside
    it.

    ANTI-VACUITY FIRST: the baseline must be VERIFIED and must have been
    decided BY A SOLVER, or the flip below measures nothing."""
    shipped = _verdict_under(V.VERIFIED_BARRED_PRIMITIVES, _pow_harness(),
                             monkeypatch)
    assert shipped.status == "VERIFIED", (
        f"the fixture no longer verifies ({shipped.status}), so it cannot "
        f"price the bar: {shipped.notes}"
    )
    solver = shipped.stamp.solver
    assert isinstance(solver, tuple) and any(s.invoked for s in solver), (
        "the fixture was settled without a solver, so no bar applies to it "
        "and this test measures nothing"
    )

    barred = _verdict_under(V.VERIFIED_BARRED_PRIMITIVES | {"pow"},
                            _pow_harness(), monkeypatch)
    assert barred.status == "UNKNOWN", barred.status
    withheld = [n for n in barred.notes if "VERIFIED withheld" in n]
    assert withheld and "contains pow" in withheld[0], withheld
    assert all(o.status == "discharged" for o in barred.obligations), (
        "the obligation stopped discharging, so the UNKNOWN above is not the "
        "bar's doing and the price measured is the wrong one"
    )


def test_the_cost_of_barring_is_finite_is_MEASURED_here_too(monkeypatch):
    """The same measurement for `is_finite`. It costs zero tests in this
    suite, which is a fact about this suite and not about the mechanism —
    so the mechanism is exhibited on a query that does reach it."""
    shipped = _verdict_under(V.VERIFIED_BARRED_PRIMITIVES, _is_finite_harness(),
                             monkeypatch)
    assert shipped.status == "VERIFIED", (
        f"the fixture no longer verifies ({shipped.status}): {shipped.notes}"
    )
    solver = shipped.stamp.solver
    assert isinstance(solver, tuple) and any(s.invoked for s in solver)

    barred = _verdict_under(V.VERIFIED_BARRED_PRIMITIVES | {"is_finite"},
                            _is_finite_harness(), monkeypatch)
    assert barred.status == "UNKNOWN", barred.status
    assert any("contains is_finite" in n for n in barred.notes), barred.notes


def test_the_two_candidate_rows_really_are_discharge_affecting(monkeypatch):
    """ANTI-VACUITY for the whole file. If neither row could put itself on a
    solver-decided slice, the decision would be about nothing and both
    measurements above would be measuring a bar that never fires.

    Asserted through the DERIVATION the bar itself uses — `_bar_scope`'s walk
    over the re-derived slice — rather than by reading the jaxpr, so a row
    that stopped reaching the emitted slice fails here."""
    for name, harness in (("pow", _pow_harness()),
                          ("is_finite", _is_finite_harness())):
        closed = trace(harness)
        p = propagate(closed)
        sliced = [
            s for s in OB.slice_unknown_obligations(
                closed, p, interval_env(closed))
            if not isinstance(s, OB.DeclinedObligation)
        ]
        assert sliced, f"{name}: nothing sliced; the row never reaches a solver"
        monkeypatch.setattr(V, "VERIFIED_BARRED_PRIMITIVES", frozenset({name}))
        found = {q for s in sliced for q in V._barred_in_eqns(s.eqns)}
        assert found == {name}, (
            f"{name} is not on any emitted slice of this fixture ({found}), "
            f"so barring it would cost this query nothing and the measurement "
            f"above is not about the row"
        )
