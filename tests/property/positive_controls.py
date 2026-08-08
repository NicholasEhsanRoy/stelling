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
rule, it cost one property (see ``test_metamorphic.py``'s module docstring),
and the rest of its cost is visible here: **four of the ten** controls are
source MUTANTS rather than historical commits, because the defect they describe
has never been in this tree.

**The rule cuts the other way too, and it has.** A second property was dropped
under it on a premise — "no one-line mutation makes it fail" — that was written
down without being run. The mutation exists; it is registered below as
``redundant-assume``; the property is back. A control that was not looked for
is not a control that does not exist, and this file is the place that
distinction has to be made honestly. That is recorded as such rather than papered over
— ``kind`` says which, and a mutant is honestly weaker evidence than a commit,
because a mutant is a defect somebody invented while a commit is one somebody
shipped. ``test_suite_disclosure.py`` asserts the split is written down
(a mutant control must say so in its own ``why``) rather than left to be
counted.

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
    # A substring the failure must carry. NOT optional in practice, and
    # ``test_suite_disclosure`` asserts it: ``""`` is a substring of every
    # string, so a control carrying it checks only ``returncode != 0`` and
    # reports a collection error, an import failure or a typo'd nodeid as
    # FIRED. The default stays for the dataclass's sake and is refused.
    expect_message: str = ""
    scale: float = 1.0  # budget multiplier, where the search needs more room


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
            "false at every declared point. OPEN — this control fires for as "
            "long as the defect lives, and the property is xfail(strict) so "
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
            "products instead of all four discharges `x*x >= 1` over "
            "int8 [-1, 1], which is false at x = 0. No historical commit "
            "carries a non-monotone interval domain — the first spike ran the "
            "monotonicity property over 6000 examples and it passed — so this "
            "is a MUTANT, which is weaker evidence than a commit and is "
            "labelled as one."
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
    # ── the affine leg deciding over a region it is not about ───────────────
    Control(
        name="vacuous-refutation",
        nodeid=f"{_ORACLE}::test_a_refuted_is_false_at_some_admitted_point",
        kind="commit",
        at="8106a55",
        why=(
            "`affine.refine_propagation` declined only on `coverage."
            "constrained`, which a DROPPED assume never raises, so the "
            "refinement judged over the DECLARED boxes — a superset of the "
            "assumed region — and re-minted a violation the interval leg had "
            "withheld. Measured at 8106a55: `assume(jnp.all(x >= 2.0))` over "
            "(-1.0, 1.0) is UNKNOWN at refine=None and REFUTED at "
            "refine='affine'. Fixed at 463ee81."
        ),
        expect_message="REFUTED OVER AN EMPTY ADMITTED REGION",
    ),
    # ── the size-0 conjunct ─────────────────────────────────────────────────
    Control(
        name="conjunct",
        nodeid=(
            f"{_META}::test_a_conjunct_that_adds_no_information"
            "_adds_no_proving_power"
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
    # ── the box-implied assume ──────────────────────────────────────────────
    Control(
        name="redundant-assume",
        nodeid=(
            f"{_META}::test_inserting_a_box_implied_assume"
            "_adds_no_proving_power"
        ),
        kind="mutant",
        at="HEAD",
        why=(
            "`_classify_assumed_pred` builds a half-space for `ge`/`gt` and "
            "`le`/`lt` and a POINT only for `eq`. Using the point for every "
            "comparison makes an assumed `x >= k` pin x to k, so restating a "
            "declaration's own lower bound narrows it to a single value: "
            "x in float64 [-1,1] |- x <= -0.5 is UNKNOWN, and VERIFIED once "
            "assume(x >= -1.0) is inserted in front of it. A MUTANT, not a "
            "commit — no revision of this tree carried it — and weaker "
            "evidence for that reason. This property shipped DROPPED, on the "
            "written-but-unrun claim that no one-line mutation makes it fail; "
            "this is that mutation."
        ),
        mutation=Mutation(
            path="src/stelling/propagate.py",
            old=(
                '        if cmp in ("ge", "gt"):\n'
                "            half = iv.IntervalArray(\n"
                "                shape=target_box.shape, los=ks, "
                "his=(math.inf,) * n\n"
                "            )\n"
                '        elif cmp in ("le", "lt"):\n'
                "            half = iv.IntervalArray(\n"
                "                shape=target_box.shape, los=(-math.inf,) * n, "
                "his=ks\n"
                "            )\n"
                "        else:  # eq\n"
                "            half = iv.IntervalArray(shape=target_box.shape, "
                "los=ks, his=ks)"
            ),
            new=(
                "        half = iv.IntervalArray(shape=target_box.shape, "
                "los=ks, his=ks)"
            ),
        ),
        expect_message="toward-VERIFIED",
    ),
    # ── reordering ──────────────────────────────────────────────────────────
    Control(
        name="reorder",
        nodeid=(
            f"{_META}::test_reordering_statements_moves_only_what_narrowing"
            "_entitles_it_to"
        ),
        kind="mutant",
        at="HEAD",
        why=(
            "an `assert_` narrowing the environment the way an `assume` does "
            "— the assume/assert confusion — makes two adjacent, independent "
            "obligations order-dependent: `assert_(x >= 0.5); assert_(x >= "
            "0.0)` gives ('unknown', 'discharged') in one order and "
            "('unknown', 'unknown') in the other. A MUTANT, and deliberately "
            "so: the reordering defect this tree shipped (e8b9377, fixed at "
            "d081d5f) is only visible when two asserts are permuted ACROSS an "
            "intervening assume, and asserting invariance under THAT "
            "permutation is unsound today, because narrowing is now "
            "deliberately forward-scoped. This property would not have caught "
            "that defect, and says so."
        ),
        mutation=Mutation(
            path="src/stelling/propagate.py",
            old=(
                '        if eqn.primitive == "stelling_assert":\n'
                "            if self.unforced_depth == 0:"
            ),
            new=(
                '        if eqn.primitive == "stelling_assert":\n'
                '            if self.assume_mode == "constrain" and eqn.invars:\n'
                '                self._assume_constrain(eqn, "assert")\n'
                "            if self.unforced_depth == 0:"
            ),
        ),
        expect_message="transposed",
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
            "non-monotone in the input box: x*x >= 0.5 over x in [-1.0, 0.5] "
            "gives [0.25, 1.0] and stays UNKNOWN, while WIDENING to "
            "[-4.0, 3.5] gives [4.0, 12.25] and DISCHARGES it. A mutant, for "
            "the same reason as `oracle-masked`, and labelled as one."
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
            "mid-model-walk; and `...\\nend 4`, the final newline cut and "
            "nothing else, read as a present terminator. Fixed at 8d3051a."
        ),
        # `[flat]` is the leg tag `_judge` stamps on ALL THREE of its failure
        # messages — truncated run, nonzero-exit run, model that was not
        # written — so it matches any genuine failure of this property and
        # nothing else. It was `""`, which is a substring of every string
        # including a traceback: measured, a planted COLLECTION ERROR in
        # test_cvc5_protocol.py was reported "FIRED — the property failed where
        # it is supposed to", `1/1 controls fired`, exit 0. A control that
        # cannot tell a defect from a broken import is not a control.
        expect_message="[flat]",
    ),
    Control(
        name="cvc5-stateful",
        nodeid=f"{_CVC5}::TestCvcTransport::runTest",
        kind="commit",
        at="0ad22bb",
        why=(
            "the same two defects, reached by the rule-based state machine "
            "without being told the record layout. It needs ~20x the flat "
            "leg's budget to get there, which is the honest cost of stateful "
            "search on a protocol whose record ORDER is fixed."
        ),
        scale=20.0,
        expect_message="[stateful]",  # see cvc5-flat for why this is not ""
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
            "0.11.0 and UNKNOWN at 4/6 with a top scatter-add on 0.10.2, with "
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
