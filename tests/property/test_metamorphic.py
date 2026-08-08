# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""METAMORPHIC PROPERTIES: two runs of the tool, related to each other.

A metamorphic property needs no ground truth. It takes a harness, changes it in
a way whose effect on the *answer* is known, and checks that the tool's two
answers stand in the relation the change implies. That is why these reach
defects an oracle cannot: where a defect is in the TRANSLATION of the user's
program to the verification IR, an oracle that executes the program executes
the same broken translation and agrees with the wrong answer, while two runs
that should agree still disagree.

**DIRECTION VOCABULARY, used throughout and load-bearing.**

*toward VERIFIED*
    the changed run gained proving power it should not have. This is the
    catastrophic direction: a wrong VERIFIED is a claim the user acts on.
*toward REFUTED*
    the changed run gained refuting power it should not have. Bad, recoverable.
*neither*
    a precision difference in the safe direction — the tool withheld more. A
    power gap, not a defect, and **not asserted against**. Every property below
    permits it, because a property that forbade it would report the tool being
    careful.

**Each property below states, in its own docstring, the version that HOLDS —
which is in three cases not the version it was originally posed as.** A
property that is unsound as posed does not become sound by being written down
in a test; it becomes a generator of false reports. Where a clause was refused,
the refusal and its counterexample are recorded next to the clause that
replaced it.

────────────────────────────────────────────────────────────────────────────
POSITIVE CONTROLS
────────────────────────────────────────────────────────────────────────────

Every property here has a commit or a source mutation at which it FAILS; they
are registered in ``tests/property/positive_controls.py`` and demonstrated by
``python tools/property_check.py --controls``. A property whose control cannot
be demonstrated does not belong in this file.

────────────────────────────────────────────────────────────────────────────
WHAT IS AND IS NOT COVERED
────────────────────────────────────────────────────────────────────────────

Covered: one to three ``any_array`` declarations of mixed dtype and mixed rank
(including size-0), one to four ``assume``/``assert_`` statements over
arithmetic, ``jnp.maximum``/``minimum``, ``abs``/``sign``/``square``,
``sqrt``/``exp`` at float dtypes, ``x - x`` cancellation, ``jnp.sum``, casts
between the harness's own dtypes, and ``&``/``|``/``~`` over the six
comparisons. Interval-only: no property here passes ``solver_timeout_ms``, so
solver timeout noise cannot masquerade as non-monotonicity.

NOT covered: the solver legs, ``nonvacuity``, control flow (``scan``/``while``/
``fori_loop``), scatter beyond the single ``.at[].add`` node, ``vmap``/``grad``,
the public template helpers, ``strict=True``, and ``vacuity_mode="all"``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis", reason="needs hypothesis")
jax = pytest.importorskip("jax", reason="needs jax")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import _grammar  # noqa: E402
import _profiles  # noqa: E402
import _runner  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _both(spec_a, spec_b, census, **kw):
    """Run two legs; ``None`` unless both produced a verdict."""
    a = _runner.run(spec_a, census=census, **kw)
    if a is None:
        return None
    b = _runner.run(spec_b, census=census, **kw)
    if b is None:
        return None
    census.compare()
    return a, b


# ── P1: a redundant conjunct ─────────────────────────────────────────────────


def test_a_box_implied_conjunct_does_not_add_proving_power():
    """Conjoining ``x >= lo`` — true of every member of the declared box.

    **The property as posed — "adding a redundant conjunct must not change the
    verdict" — is REFUSED, and this is the version that holds.** A bigger
    formula is a harder formula for a non-relational abstract domain, so
    ``VERIFIED -> UNKNOWN`` under a redundant conjunct is an ordinary precision
    loss in the safe direction. Asserting equality would report the tool
    withholding, which is the one thing it is always allowed to do.

    What holds, and is asserted:

    * **no contradiction.** ``p`` and ``p & c`` are the same claim over the
      declared set when ``c`` holds everywhere on it, so one leg VERIFIED and
      the other REFUTED is a flat inconsistency.
    * **no gain toward VERIFIED.** Conjunction makes an obligation *stronger*,
      so it can never become easier to prove. A gain here means the analysis
      narrowed the set it was reasoning over — which is precisely the shape of
      the size-0 conjunct defect.

    Three ways the conjunct would not actually be redundant are excluded before
    running anything, in ``_grammar.redundant_conjunct_is_sound_for``: a NaN
    bound, a ``bool`` declaration, and — the one that matters — an integer
    bound outside its own dtype's range, which jax reduces mod ``2**bits`` so
    that the conjunct the tool sees is not the conjunct that was written. That
    last exclusion is the open wrap defect, and conflating it with a redundancy
    violation would file the wrong report.

    Also excluded: a conjunct that changes the predicate's ELEMENT COUNT. A
    ``bool[0]`` conjunct collapses the conjunction to zero elements, which
    makes it vacuously true rather than redundant — a different property, and
    the next one.
    """
    census = _runner.Census("metamorphic/redundant-conjunct")

    @_profiles.current().settings(250)
    @given(_grammar.general_specs(), st.data())
    def search(spec, data):
        census.draw()
        usable = [d for d in spec.decls if _grammar.redundant_conjunct_is_sound_for(d)]
        if not usable:
            census.skip("no declaration with a sound box-implied conjunct")
            return
        decl = data.draw(st.sampled_from(usable), label="decl")
        i = data.draw(st.integers(0, len(spec.stmts) - 1), label="stmt")
        before = _grammar.static_shape_pred(spec.stmts[i].pred, spec.decls)
        after = _grammar._broadcast(before, decl.shape)
        if before is None or after is None or after != before:
            census.skip("conjunct changes the predicate's shape")
            return
        mutated = _grammar.with_redundant_conjunct(spec, i, decl)
        pair = _both(spec, mutated, census)
        if pair is None:
            return
        a, b = pair
        census.tag("conjunct_compared")
        assert not _runner.contradiction(a, b), (
            f"CONTRADICTION under a box-implied conjunct: {a.status} -> "
            f"{b.status}\n{spec.render()}\n--- with the conjunct ---\n"
            f"{mutated.render()}"
        )
        assert not (a.status != "VERIFIED" and b.status == "VERIFIED"), (
            f"toward-VERIFIED: {a.status} -> VERIFIED under a conjunct that "
            f"cannot add information\n{spec.render()}\n"
            f"--- with the conjunct ---\n{mutated.render()}"
        )

    search()
    census.require(compared=40, conjunct_compared=40)


def test_a_conjunct_that_empties_the_predicate_says_what_it_says_alone():
    """``p & c`` where ``c`` is ``bool[0]`` must equal ``c`` alone.

    ``assume(q)`` means "for every element of ``q``". When ``c`` has zero
    elements, ``p & c`` has zero elements too, so both ``p & c`` and ``c``
    alone are ``∀ ∈ ∅`` — vacuously true, constraining nothing. The two
    harnesses are therefore the same program and must get the same verdict.

    **This is the size-0 conjunct defect stated as a metamorphic property.**
    The defect narrowed a rank-0 sibling variable to a strict SUBSET on the
    strength of a conjunction that was vacuously true, minting VERIFIED over
    less than the declared set. Against the ``c``-alone leg, which narrows
    nothing, that shows up as ``UNKNOWN -> VERIFIED``: the catastrophic
    direction, and the reason this clause is stated in terms of a *gain* rather
    than a difference.

    Note what is NOT asserted: that ``p & c`` equals ``p``. It does not — the
    conjunct genuinely destroys the constraint, and that is correct behaviour.
    """
    census = _runner.Census("metamorphic/vacuous-conjunct")

    @_profiles.current().settings(250)
    @given(_grammar.general_specs(), st.data())
    def search(spec, data):
        census.draw()
        empties = [
            d
            for d in spec.decls
            if _grammar.n_elements(d.shape) == 0
            and _grammar.redundant_conjunct_is_sound_for(d)
        ]
        if not empties:
            census.skip("no size-0 declaration to conjoin")
            return
        decl = data.draw(st.sampled_from(empties), label="decl")
        i = data.draw(st.integers(0, len(spec.stmts) - 1), label="stmt")
        before = _grammar.static_shape_pred(spec.stmts[i].pred, spec.decls)
        after = _grammar._broadcast(before, decl.shape)
        if after is None or _grammar.n_elements(after) != 0:
            census.skip("conjunction is not zero-element")
            return
        conjoined = _grammar.with_redundant_conjunct(spec, i, decl)
        alone = _grammar.Spec(
            spec.decls,
            tuple(
                _grammar.Stmt(s.kind, _grammar.box_implied_pred(decl))
                if k == i
                else s
                for k, s in enumerate(spec.stmts)
            ),
        )
        pair = _both(conjoined, alone, census)
        if pair is None:
            return
        both_ways, alone_only = pair
        census.tag("vacuous_compared")
        assert not (
            alone_only.status != "VERIFIED" and both_ways.status == "VERIFIED"
        ), (
            "toward-VERIFIED: a zero-element conjunction proved more than the "
            f"same zero-element predicate alone ({alone_only.status} -> "
            f"{both_ways.status}); a vacuously-true assume narrowed something"
            f"\n--- p & c ---\n{conjoined.render()}"
            f"\n--- c alone ---\n{alone.render()}"
        )

    search()
    census.require(compared=15, vacuous_compared=15)


def test_a_box_implied_assume_does_not_add_proving_power():
    """Inserting ``assume(x >= lo)`` intersects the box with a superset of itself.

    In the interval domain that is literally a no-op, so it cannot make
    anything easier to prove. Same two clauses as the conjunct property, same
    exclusions, and the same refusal of the "must not change the verdict"
    phrasing: an extra statement is an extra traced equation, and losing
    precision to it is safe.
    """
    census = _runner.Census("metamorphic/redundant-assume")

    @_profiles.current().settings(250)
    @given(_grammar.general_specs(), st.data())
    def search(spec, data):
        census.draw()
        usable = [
            d
            for d in spec.decls
            if _grammar.redundant_conjunct_is_sound_for(d)
            and _grammar.n_elements(d.shape) > 0
        ]
        if not usable:
            census.skip("no declaration with a sound box-implied assume")
            return
        decl = data.draw(st.sampled_from(usable), label="decl")
        at = data.draw(st.integers(0, len(spec.stmts)), label="at")
        mutated = _grammar.with_redundant_assume(spec, at, decl)
        pair = _both(spec, mutated, census)
        if pair is None:
            return
        a, b = pair
        census.tag("assume_compared")
        assert not _runner.contradiction(a, b), (
            f"CONTRADICTION under a box-implied assume: {a.status} -> "
            f"{b.status}\n{spec.render()}\n--- with the assume ---\n"
            f"{mutated.render()}"
        )
        assert not (a.status != "VERIFIED" and b.status == "VERIFIED"), (
            f"toward-VERIFIED: {a.status} -> VERIFIED under a box-implied "
            f"assume\n{spec.render()}\n--- with the assume ---\n"
            f"{mutated.render()}"
        )

    search()
    census.require(compared=40, assume_compared=40)


# ── P2: reordering independent statements ────────────────────────────────────


def test_reordering_statements_moves_only_what_narrowing_entitles_it_to():
    """Swapping two adjacent statements.

    **The property as posed — "reordering independent statements must not
    change the verdict" — is REFUSED for one of the two cases**, because this
    tree is now *deliberately* forward-scoped on narrowing: an ``assume``
    constrains the obligations written after it and not the ones written
    before. Moving an ``assume`` across an ``assert_`` therefore moves a real
    scoping boundary and is entitled to cost a VERIFIED, or a REFUTED, in
    either direction. Asserting equality there would report a design decision
    as a defect.

    So the property splits by what was swapped.

    **Two adjacent ``assert_``s — equality, both directions.** An ``assert_``
    narrows nothing, so transposing two of them cannot move any scoping
    boundary. The whole verdict must be identical and the *multiset* of
    per-obligation statuses must be identical (a multiset, because the
    obligations themselves are renumbered by the swap). This is the clause with
    a real historical failure behind it: an interval-leg withhold that was read
    at the assert rather than over the whole query made the verdict of each
    obligation depend on which assumes happened to be traced above it, so
    transposing two independent asserts moved each one between REFUTED and
    UNKNOWN.

    **Anything else — no contradiction, and only where the harness is not
    vacuous.** Whatever the order, both runs over-approximate the same set of
    admitted points, so one leg VERIFIED and the other REFUTED cannot both be
    right — *unless* the admitted set is empty, where ``∀ ∈ ∅`` is true and a
    REFUTED on the other side of the swap is the tool judging a different
    (non-empty) region. That exception is not hand-waved: it is decided exactly,
    by enumerating the declared box in Python integers, and the clause is
    simply not applied when the exact admitted set is empty.

    **Two adjacent ``assume``s are in the second bucket, not the first, and
    that was a correction.** An earlier draft asserted equality for them on the
    grounds that intersection commutes. It does not commute *in a single
    narrowing pass*: with ``x, y ∈ [0, 10]``, ``assume(x >= y); assume(y >= 5)``
    leaves ``x ∈ [0, 10]``, while the other order narrows ``y`` first and then
    ``x`` to ``[5, 10]``. Both are sound over-approximations; one is sharper.
    Asserting equality would have reported that as a defect.
    """
    census = _runner.Census("metamorphic/reorder")

    @_profiles.current().settings(250)
    @given(
        st.one_of(_grammar.general_specs(), _grammar.integer_program_specs()),
        st.data(),
    )
    def search(spec, data):
        census.draw()
        if len(spec.stmts) < 2:
            census.skip("fewer than two statements")
            return
        i = data.draw(st.integers(0, len(spec.stmts) - 2), label="i")
        s1, s2 = spec.stmts[i], spec.stmts[i + 1]
        swapped = _grammar.swapped(spec, i)
        pair = _both(spec, swapped, census)
        if pair is None:
            return
        a, b = pair
        if s1.kind == "assert" and s2.kind == "assert":
            census.tag("assert_swap_compared")
            assert a.status == b.status, (
                f"two adjacent assert_s were transposed and the verdict moved: "
                f"{a.status} -> {b.status}\n{spec.render()}"
            )
            assert sorted(a.obligations) == sorted(b.obligations), (
                f"two adjacent assert_s were transposed and the per-obligation "
                f"statuses moved: {a.obligations} -> {b.obligations}\n"
                f"{spec.render()}"
            )
            return
        if not _grammar.exact_supported(spec):
            census.skip("cross-kind swap, no exact vacuity decision available")
            return
        if not _grammar.any_obligation_is_admitted(spec):
            census.skip("preconditions admit nothing")
            return
        census.tag("crosskind_swap_compared")
        assert not _runner.contradiction(a, b), (
            f"CONTRADICTION under a {s1.kind}/{s2.kind} transposition over a "
            f"NON-EMPTY admitted set: {a.status} -> {b.status}\n{spec.render()}"
        )

    search()
    census.require(compared=40, assert_swap_compared=10, crosskind_swap_compared=5)


# ── P4: refine=None vs refine="affine" ───────────────────────────────────────


def test_the_affine_refinement_never_contradicts_the_interval_leg():
    """``refine=None`` and ``refine="affine"`` may not disagree on a definite verdict.

    The refinement exists to close obligations intervals left undecided, so
    either leg may be UNKNOWN where the other is definite and **that is not
    asserted against**. What cannot happen is both legs being definite and
    definite differently: they are two analyses of one query, and the query has
    one answer.

    Asserted at two granularities, because a whole-verdict comparison alone
    would miss a swap inside a multi-obligation harness: the ``VERIFIED``/
    ``REFUTED`` contradiction, and any obligation ``discharged`` in one leg and
    ``violated-over-set`` in the other.

    **Stated honestly: this does NOT cover the affine defect the project
    actually shipped.** That one was ``UNKNOWN`` at ``refine=None`` and
    ``REFUTED`` at ``refine="affine"`` — the refinement re-minting a violation
    the interval leg had deliberately withheld over an empty assumed region.
    Only one leg was definite, so the clause above permits it, correctly:
    deciding what intervals could not is what a refinement is *for*. The
    vacuous half of that defect is caught by
    ``test_oracle.py::test_a_refuted_is_false_at_some_admitted_point``, which
    is an oracle property rather than this one, and its positive control is
    that commit.
    """
    census = _runner.Census("metamorphic/refine")

    @_profiles.current().settings(200)
    @given(_grammar.general_specs())
    def search(spec):
        census.draw()
        pair = _both(spec, spec, census, refine=None)
        if pair is None:
            return
        a = pair[0]
        b = _runner.run(spec, census=census, refine="affine")
        if b is None:
            return
        census.tag("refine_compared")
        assert not _runner.contradiction(a, b), (
            f"the refine legs contradict: refine=None {a.status} vs "
            f"refine='affine' {b.status}\n{spec.render()}"
        )
        clash = _runner.obligation_contradiction(a, b)
        assert clash is None, (
            f"obligation {clash[0] if clash else '?'} contradicts across "
            f"refine: refine=None {clash[1] if clash else '?'} vs "
            f"refine='affine' {clash[2] if clash else '?'}\n{spec.render()}"
        )

    search()
    census.require(compared=30, refine_compared=30)


# ── P5: widening a declared bound ────────────────────────────────────────────


def test_widening_a_declared_bound_cannot_turn_unknown_into_verified():
    """Widening the declared box must not make an obligation *easier*.

    **The clause this property was originally given — "widening must not turn
    VERIFIED into REFUTED" — is REFUSED, and it is worth being explicit about
    why, because it is the more intuitive of the two.** Widening the declared
    set STRENGTHENS the claim: the obligation now has to hold over more points.
    An obligation genuinely true on the narrow box can be genuinely false on
    the wide one. ``x ∈ (0, 1) ⊢ x <= 1`` is VERIFIED and ``x ∈ (-3, 4) ⊢
    x <= 1`` is correctly REFUTED. Asserting that clause would report the tool
    being right, on almost every example it fired on.

    What is sound, and is asserted: an interval extension is **monotone in its
    input box**, so ``F(I) ⊆ F(J)`` whenever ``I ⊆ J``, so anything provable
    over the wide box is provable over the narrow one. ``UNKNOWN -> VERIFIED``
    under widening therefore means the abstract domain is non-monotone — the
    same shape as reasoning over a subset of the declared set, which is how
    every wrong VERIFIED in this project's catalogue that was not a translation
    defect came about.

    Run interval-only, so that solver timeout noise cannot masquerade as
    non-monotonicity. Integer boxes are widened **within the dtype's own
    range**, so that widening cannot introduce the literal wrap and be mistaken
    for it.
    """
    census = _runner.Census("metamorphic/widen")

    @_profiles.current().settings(250)
    @given(_grammar.general_specs(), st.data())
    def search(spec, data):
        census.draw()
        i = data.draw(st.integers(0, len(spec.decls) - 1), label="decl")
        wide = _grammar.widened(spec, i)
        if wide is None:
            census.skip("declaration cannot be widened")
            return
        pair = _both(spec, wide, census)
        if pair is None:
            return
        narrow, widened = pair
        census.tag("widen_compared")
        assert not (narrow.status != "VERIFIED" and widened.status == "VERIFIED"), (
            f"toward-VERIFIED under widening: {narrow.status} -> VERIFIED. "
            f"Declaration {spec.decls[i].name} widened from "
            f"({spec.decls[i].lo!r}, {spec.decls[i].hi!r}) to "
            f"({wide.decls[i].lo!r}, {wide.decls[i].hi!r}); interval "
            f"propagation is monotone in the input box, so this cannot happen."
            f"\n{spec.render()}"
        )

    search()
    census.require(compared=40, widen_compared=40)
