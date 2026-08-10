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
and the rest of its cost is visible here: **five of the twelve** controls are
source MUTANTS rather than historical commits, because the defect they describe
has never been in this tree.

That count has been wrong in the cheap direction, once, on this branch.
``cvc5-exit-tell`` shipped as a mutant whose ``why`` said "no revision of this
tree has carried it", and one line of git says otherwise:
``git log -S "or proc.returncode != 0" -- src/stelling/solvers.py`` shows the
guard was ADDED at ``b6e4783``, so its parent ``8ef8f75`` carries the defect
and the entry is a commit control. **Ask git before registering a mutant.**

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
    #
    # MATCHED AGAINST THE FAILURE pytest RECORDS, not against everything the
    # run echoes: ``property_check.py`` reads the ``message`` attributes out of
    # ``--junitxml``. A traceback prints the whole source of every function on
    # it, docstring included, so a guard string is present in the OUTPUT of any
    # crash inside the property's own helpers even when the oracle never ran.
    # See ``tools/property_check.py``'s module docstring for the measurement.
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
            "nothing else, read as a present terminator. Fixed at 8d3051a. "
            "WHAT A GREEN RUN OF THIS CONTROL DOES NOT SAY: the property's "
            "oracle is a conjunction of THREE clauses and this tree exercises "
            "ONE of them. Measured over the flat leg's own 1500 ci examples at "
            "0ad22bb, all three evaluated independently: 5 violations, every "
            "one of them clause (2) `nothing was truncated`, none of clause "
            "(1) or clause (3). The table is in `_judge`'s docstring in "
            "test_cvc5_protocol.py, and clauses (1) and (3) are demonstrated "
            "by `cvc5-exit-tell` and `cvc5-phantom-model` below instead."
        ),
        # `[flat]` is the leg tag `_judge` stamps on ALL THREE of its failure
        # messages — truncated run, nonzero-exit run, model that was not
        # written — so it matches any genuine failure of this property and
        # nothing else. It was `""`, which is a substring of every string
        # including a traceback: measured, a planted COLLECTION ERROR in
        # test_cvc5_protocol.py was reported "FIRED — the property failed where
        # it is supposed to", `1/1 controls fired`, exit 0. A control that
        # cannot tell a defect from a broken import is not a control.
        #
        # THE SAME BREADTH IS WHY THIS GUARD CANNOT SAY WHICH CLAUSE FIRED, and
        # it is not doing so here: at 0ad22bb what it matches is always the
        # clause-(2) message. `cvc5-exit-tell` and `cvc5-phantom-model` carry
        # clause-specific guards for exactly that reason.
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
            "search on a protocol whose record ORDER is fixed. It demonstrates "
            "the SAME ONE of `_judge`'s three clauses that `cvc5-flat` does — "
            "(2), nothing was truncated — and for the same reason: the tree, "
            "not the search. See `cvc5-flat`'s `why` and `_judge`'s docstring."
        ),
        scale=20.0,
        expect_message="[stateful]",  # see cvc5-flat for why this is not ""
    ),
    # ── the other two clauses of the same oracle ────────────────────────────
    #
    # `_judge` states its invariant as THREE clauses and the two controls above
    # demonstrate exactly one of them — clause (2), measured, tabulated in
    # `_judge`'s own docstring. These two demonstrate the other two. Neither
    # touches `_PAYLOADS`, so neither costs the 673/8165 example-efficiency
    # figures in `test_cvc5_protocol.py`'s module docstring anything.
    #
    # ONE COMMIT AND ONE MUTANT, and the split was got wrong here first. Both
    # shipped as mutants "because the defect they describe has never been in
    # this tree". For clause (1) that was false and git said so in one line
    # (`git log -S "or proc.returncode != 0"`): `8ef8f75` is the parent of the
    # commit that added the guard, and it carries the defect. For clause (3)
    # it holds — `git log -S "sorted(set(values))" -- src/stelling/solvers.py`
    # is empty, no revision has ever deduped the harvested model — so
    # `cvc5-phantom-model` stays a mutant and says so.
    #
    # Their `expect_message` is the clause's own sentence and NOT the `[flat]`
    # leg tag the two controls above carry. `[flat]` is stamped on all three of
    # `_judge`'s messages, so a control carrying it would be satisfied by the
    # clause-(2) failure `cvc5-flat` already finds — i.e. it would fire without
    # demonstrating the clause it is registered for, which is the exact
    # distinction these two entries exist to make.
    # `test_suite_disclosure.py` now asserts that distinction two ways, because
    # broadening either guard back to `[flat]` left every gate green.
    Control(
        name="cvc5-exit-tell",
        nodeid=f"{_CVC5}::test_the_parent_never_trusts_an_unspoken_transcript_flat",
        kind="commit",
        at="8ef8f75",
        why=(
            "`_run_cvc5_wheel` had NEITHER of the two tells it has today — no "
            "terminator check and no `proc.returncode` check — so a child that "
            "wrote a complete, well-formed transcript and then died with a "
            "nonzero exit was answered on. Both tells were added together at "
            "b6e4783 ('The cvc5 wheel transport refuses a crashed child'), of "
            "which this revision is the parent. MEASURED, hand-driven through "
            "test_cvc5_protocol.py's own fixture at 8ef8f75: 'version "
            "1.3.4\\nanswer sat\\nend 0\\n' at exit 1 is a DEFINITE sat, and "
            "'...\\nvalue x0 0/1\\nend 1\\n' at exit 137 is a DEFINITE sat "
            "harvesting ('x0', '0/1'); the tip and 0ad22bb both refuse both. "
            "Today's flat property fails there in 0.4 s at the ci profile, on "
            "an INTACT transcript at exit 1 — nothing was truncated, so clause "
            "(2) holds on that example and the failure is clause (1)'s own. "
            "8ef8f75 predates the splitlines fix as well, so its parser "
            "carries clause (2)'s defect too; `expect_message` is what makes "
            "this a control for clause (1) rather than a second `cvc5-flat`. "
            "`0ad22bb` cannot supply one: by then the parser refused a nonzero "
            "exit, so that tree carries the other defect and not this one."
        ),
        expect_message="ACCEPTED A NONZERO-EXIT RUN",
    ),
    Control(
        name="cvc5-phantom-model",
        nodeid=f"{_CVC5}::test_the_parent_never_trusts_an_unspoken_transcript_flat",
        kind="mutant",
        at="HEAD",
        why=(
            "the harvested model stops being the value records that were "
            "written: `values=tuple(sorted(values))` becomes "
            "`tuple(sorted(set(values)))`, so a child that wrote `value x0 0/1` "
            "twice is reported as having written it once. The dedupe sits "
            "AFTER the terminator check, so `end <n>`'s count still matches "
            "and clauses (1) and (2) both hold — only clause (3) can fail. "
            "MEASURED under the mutation at the ci profile: `HARVESTED A MODEL "
            "THAT WAS NOT WRITTEN [flat]`, parent harvested [('x0', '0/1')] "
            "against actually present [('x0', '0/1'), ('x0', '0/1')], found in "
            "2.0 s. A MUTANT, weaker evidence than a commit, and the narrow "
            "answer to `_judge`'s standing finding that no draw of either "
            "generator reaches clause (3): a FORGED value record is still out "
            "of the alphabet's reach and that finding stands, but a DUPLICATE "
            "one is in it and needs no new payload."
        ),
        mutation=Mutation(
            path="src/stelling/solvers.py",
            old="        values=tuple(sorted(values)),",
            new="        values=tuple(sorted(set(values))),",
        ),
        expect_message="HARVESTED A MODEL THAT WAS NOT WRITTEN",
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
        # WAS `disagree`, WHICH THIS PROPERTY'S FAILURE NEVER CARRIED. The
        # assertion says "THE TWO TESTED jax SERIES DISAGREE on N of M
        # harnesses"; the lower-case word occurs only in the local
        # `disagreements` list four lines above it. Matching against captured
        # output hid that — a traceback prints the whole function source down
        # to the failing line, so `disagree` was there for the corpus-floor
        # assertion below it too, and a corpus that had stopped producing
        # verdicts would have been reported as a demonstration of series
        # disagreement. Measured, against 8ef8f75 with two interpreters: the
        # recorded failure is `AssertionError: THE TWO TESTED jax SERIES
        # DISAGREE on 4 of 225 harnesses`, and `disagree` is not in it.
        expect_message="SERIES DISAGREE",
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
