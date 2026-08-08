# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""WHERE EVERY PROPERTY IS KNOWN TO FAIL. The registry, and why it exists.

**A property that finds nothing and a property whose strategy silently
generates nothing look identical from the outside.** Both print a green line.
This project has shipped that shape repeatedly — most recently a pin test that
returned early on a non-git checkout and reported ``3 passed`` having examined
nothing at all.

The census floors inside each property are one answer to that (a property
cannot pass without counting what it examined). This file is the other, and it
is the stronger one: **every property ships with a place where it is known to
FAIL, and a way to run it there.** Not as a comment saying "this would have
caught X" — as a machine-checkable entry that ``tools/property_check.py
--controls`` executes, with the property's own current source, against a tree
that carries the defect, asserting that the run comes back RED.

A property whose control cannot be demonstrated does not ship. That is the
rule, and the cost of it is visible here: two of the eight controls are source
MUTANTS rather than historical commits, because the defect they describe has
never been in this tree. That is recorded as such rather than papered over —
``kind`` says which, and a mutant is honestly weaker evidence than a commit,
because a mutant is a defect somebody invented while a commit is one somebody
shipped.

**This module imports nothing.** It is read by ``test_suite_disclosure.py``,
which runs in environments with neither hypothesis nor jax — so that "the
property suite examined nothing here" can never be silent.
"""

from __future__ import annotations

from dataclasses import dataclass

PROPERTY_DIR = "tests/property"


@dataclass(frozen=True)
class Mutation:
    """A one-place textual change to a source file, applied to a scratch copy."""

    path: str
    old: str
    new: str


@dataclass(frozen=True)
class Control:
    name: str
    nodeid: str          # the property, as pytest names it
    kind: str            # "commit" | "mutant"
    at: str              # the revision to run against
    why: str             # the defect, in one sentence
    mutation: Mutation | None = None
    series: str = "any"  # "any" | "both" — "both" needs two interpreters
    expect_message: str = ""  # a substring the failure must carry


_ORACLE = f"{PROPERTY_DIR}/test_oracle.py"
_META = f"{PROPERTY_DIR}/test_metamorphic.py"
_CVC5 = f"{PROPERTY_DIR}/test_cvc5_protocol.py"
_CROSS = f"{PROPERTY_DIR}/test_cross_series.py"


CONTROLS = (
    # ── the open defect, live on main today ─────────────────────────────────
    Control(
        name="oracle-wrap",
        nodeid=f"{_ORACLE}::test_a_verified_is_true_at_every_admitted_point",
        kind="commit",
        at="HEAD",
        why=(
            "an out-of-dtype-range integer literal wraps mod 2**bits before "
            "tracing, so stelling returns VERIFIED for a predicate that is "
            "false at every declared point. OPEN — this control passes for as "
            "long as the defect does, and the property is xfail(strict) so "
            "that the day it is fixed the suite goes red instead of quiet."
        ),
        expect_message="WRONG VERIFIED",
    ),
    # ── the residual leg, which must be green on main ────────────────────────
    Control(
        name="oracle-masked",
        nodeid=(
            f"{_ORACLE}::test_a_verified_is_true_at_every_admitted_point"
            "_outside_the_wrap_class"
        ),
        kind="mutant",
        at="HEAD",
        why=(
            "interval multiplication that considers only the two SAME-CORNER "
            "products instead of all four is unsound in both directions and "
            "mints VERIFIEDs that are false at declared points. No historical "
            "commit carries this; it is a mutant, and is labelled as one."
        ),
        mutation=Mutation(
            path="src/stelling/interval.py",
            old=(
                "        products = (_prod(alo, blo), _prod(alo, bhi), "
                "_prod(ahi, blo), _prod(ahi, bhi))"
            ),
            new="        products = (_prod(alo, blo), _prod(ahi, bhi))",
        ),
        expect_message="WRONG VERIFIED",
    ),
    # ── the size-0 conjunct ─────────────────────────────────────────────────
    Control(
        name="vacuous-conjunct",
        nodeid=(
            f"{_META}::test_a_conjunct_that_empties_the_predicate"
            "_says_what_it_says_alone"
        ),
        kind="commit",
        at="fb34e0d",
        why=(
            "`_apply_assumed_pred` classified each conjunct of an `&` "
            "standalone, so a `bool[0]` conjunct — vacuously true over the "
            "whole declared box — still narrowed its rank-0 sibling to a "
            "strict SUBSET, minting VERIFIED over less than the declared set. "
            "Fixed at 717b9ca."
        ),
        expect_message="toward-VERIFIED",
    ),
    Control(
        name="redundant-conjunct",
        nodeid=f"{_META}::test_a_box_implied_conjunct_does_not_add_proving_power",
        kind="commit",
        at="fb34e0d",
        why=(
            "the same size-0 narrowing, reached through the shape-preserving "
            "conjunct rather than the emptying one. Weaker evidence than "
            "`vacuous-conjunct` and recorded separately so that a green run "
            "here is not read as covering that defect."
        ),
        expect_message="",
    ),
    # ── reordering ──────────────────────────────────────────────────────────
    Control(
        name="reorder",
        nodeid=(
            f"{_META}::test_reordering_statements_moves_only_what_narrowing"
            "_entitles_it_to"
        ),
        kind="commit",
        at="e8b9377",
        why=(
            "the interval leg's withhold of a definite violation was read AT "
            "THE ASSERT, so it saw only the assumes traced above that "
            "obligation. Transposing two independent `assert_`s therefore "
            "moved each one between REFUTED and UNKNOWN. Fixed at d081d5f — "
            "'an assume is a precondition on the WHOLE QUERY'."
        ),
        expect_message="transposed",
    ),
    # ── refine ──────────────────────────────────────────────────────────────
    Control(
        name="refine",
        nodeid=f"{_META}::test_the_affine_refinement_never_contradicts_the_interval_leg",
        kind="mutant",
        at="HEAD",
        why=(
            "the affine leg deciding `discharged` off the UPPER end of the "
            "concretised slack instead of the lower end — a bound confusion, "
            "which discharges obligations the interval leg refutes and so "
            "produces the both-definite disagreement this property forbids. "
            "A mutant: the affine defect this tree actually shipped (8106a55) "
            "was UNKNOWN-vs-REFUTED, which this property permits, and is "
            "covered by `vacuous-refutation` instead."
        ),
        mutation=Mutation(
            path="src/stelling/affine.py",
            old="    n_true = sum(1 for lo, _ in ranges if _element_true(lo))",
            new="    n_true = sum(1 for _, hi in ranges if _element_true(hi))",
        ),
        expect_message="contradict",
    ),
    Control(
        name="vacuous-refutation",
        nodeid=f"{_ORACLE}::test_a_refuted_is_false_at_some_admitted_point",
        kind="commit",
        at="8106a55",
        why=(
            "`affine.refine_propagation` declined only on `coverage."
            "constrained`, which a DROPPED assume never raises, so the "
            "refinement judged over declared boxes that are a superset of the "
            "assumed region and re-minted a violation the interval leg had "
            "withheld — REFUTED over an empty assumed region. Fixed at "
            "463ee81."
        ),
        expect_message="REFUTED",
    ),
    # ── widening ────────────────────────────────────────────────────────────
    Control(
        name="widen",
        nodeid=(
            f"{_META}::test_widening_a_declared_bound_cannot_turn_unknown"
            "_into_verified"
        ),
        kind="mutant",
        at="HEAD",
        why=(
            "interval multiplication over two corners instead of four is "
            "non-monotone in the input box: x ∈ [-0.4, 0.5] gives [0.16, "
            "0.25] for x*x and leaves `x*x >= 0.25` UNKNOWN, while WIDENING "
            "to x ∈ [-1.0, 0.5] gives [0.25, 1.0] and DISCHARGES it. No "
            "historical commit carries a non-monotone domain — the first "
            "spike ran this property over 6000 examples and it passed — so "
            "the control is a mutant and is labelled as one."
        ),
        mutation=Mutation(
            path="src/stelling/interval.py",
            old=(
                "        products = (_prod(alo, blo), _prod(alo, bhi), "
                "_prod(ahi, blo), _prod(ahi, bhi))"
            ),
            new="        products = (_prod(alo, blo), _prod(ahi, bhi))",
        ),
        expect_message="toward-VERIFIED under widening",
    ),
    # ── the cvc5 line protocol ──────────────────────────────────────────────
    Control(
        name="cvc5-flat",
        nodeid=f"{_CVC5}::test_the_parent_never_trusts_an_unspoken_transcript_flat",
        kind="commit",
        at="0ad22bb",
        why=(
            "`_run_cvc5_wheel` read the child's stdout with `str.splitlines()` "
            "while the child wrote records with `print`. splitlines breaks on "
            "ten characters, not one, so a payload could forge the parser's "
            "last line — the terminator — while the child was truncated "
            "mid-model-walk; and `…\\nend 4`, the final newline cut and "
            "nothing else, read as a present terminator. Fixed at 8d3051a."
        ),
        expect_message="",
    ),
    Control(
        name="cvc5-stateful",
        nodeid=f"{_CVC5}::TestCvcTransport::runTest",
        kind="commit",
        at="0ad22bb",
        why="the same two defects, reached by the rule-based state machine.",
        expect_message="",
    ),
    # ── cross-series ────────────────────────────────────────────────────────
    Control(
        name="cross-series",
        nodeid=f"{_CROSS}::test_the_two_tested_jax_series_agree",
        kind="commit",
        at="8ef8f75",
        why=(
            "`_is_add_combiner` tested `isinstance(v, ir.ClosedJaxpr)`, and "
            "0.11 merged ClosedJaxpr into Jaxpr while 0.10.2 did not — so a "
            "scatter-add harness was VERIFIED at 6/6 equations known on "
            "0.11.0 and UNKNOWN at 4/6 with a ⊤ scatter-add on 0.10.2, with "
            "TESTED_JAX_SERIES already claiming both. Fixed at 76140c2."
        ),
        series="both",
        expect_message="disagree",
    ),
)


def by_name(name: str) -> Control:
    for c in CONTROLS:
        if c.name == name:
            return c
    raise KeyError(f"no positive control named {name!r}; have "
                   f"{[c.name for c in CONTROLS]}")


def property_nodeids() -> frozenset[str]:
    """Every property this registry claims to control."""
    return frozenset(c.nodeid for c in CONTROLS)
