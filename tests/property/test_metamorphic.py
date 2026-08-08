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

────────────────────────────────────────────────────────────────────────────
TWO PROPERTIES THAT DO NOT SHIP, AND WHY
────────────────────────────────────────────────────────────────────────────

**"``refine=None`` and ``refine="affine"`` must not disagree on a definite
verdict" — REFUSED AS UNFALSIFIABLE ON THIS TREE.** The clause is sound. It is
also incapable of failing, which makes it exactly the kind of green line this
whole suite exists to prevent. The refinement only ever visits obligations the
interval leg left UNDECIDED, so an obligation the interval leg called
``discharged`` or ``violated-over-set`` keeps that answer whatever the
refinement thinks. Measured, not argued: with ``affine.py`` mutated so that the
refinement discharges EVERY obligation it evaluates, ``x0 ∈ [0.0, 1.0] ⊢
x0 >= 2.0`` is still ``REFUTED`` at ``refine="affine"``, byte-identically to
``refine=None``, while an interval-undecided obligation flips to VERIFIED as
expected. A property that cannot be made to fail by breaking the thing it
watches is not watching it.

What ships instead is the falsifiable half, and it is where the affine defect
this tree actually shipped lived:
``test_oracle.py::test_a_refuted_is_false_at_some_admitted_point`` runs BOTH
refine legs and forbids a ``violated-over-set`` over an admitted region that is
exactly empty. Its positive control is ``8106a55``.

**"Inserting a box-implied ``assume`` must not add proving power" — DROPPED FOR
WANT OF A CONTROL.** The clause is sound and it is what the first spike ran.
No commit in this tree's history and no one-line mutation makes it fail: the
size-0 defect it would plausibly have caught needs the vacuous predicate to be
CONJOINED onto an existing narrowing assume, not inserted beside one, and the
conjunct property above covers that. Under this suite's own rule — a property
whose positive control cannot be demonstrated does not ship — it is not here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis", reason="needs hypothesis")
jax = pytest.importorskip("jax", reason="needs jax")

from hypothesis import example, given  # noqa: E402
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


# ── P1: a conjunct that adds no information ──────────────────────────────────

SIZE0_CONJUNCT_EXAMPLE = (
    _grammar.Spec(
        (
            _grammar.Decl("x0", (), "float64", -1.0, 1.0),
            _grammar.Decl("x1", (0,), "float64", -1.0, 1.0),
        ),
        (
            _grammar.Stmt("assume", ("cmp", ">=", ("var", "x0"), ("const", 0.5))),
            _grammar.Stmt("assert", ("cmp", ">", ("var", "x0"), ("const", 0.0))),
        ),
    ),
    1,  # conjoin the box-implied predicate of x1 — the size-0 declaration
    0,  # onto the assume
)
"""The size-0 conjunct shape, pinned.

Measured at ``fb34e0d``: ``assume((x0 >= 0.5) & (x1 >= -1.0))`` with ``x1``
of shape ``(0,)`` is VERIFIED, while the same zero-element predicate ALONE is
UNKNOWN — the tool narrowed ``x0`` on the strength of a conjunction that has no
elements at all. Pinned as an ``@example`` because the unbiased search over
this grammar did not construct it within 2500 examples: it needs a size-0
declaration beside a rank-0 one, an ``assume`` statement, and a base narrowing
that is load-bearing, all at once. That is recorded rather than hidden.
"""


def _conjunct_targets():
    """``(spec, declaration index, statement index)``.

    The statement index is drawn with a bias toward ``assume`` statements when
    the harness has any, because that is where a conjunct can do damage:
    conjoining onto an ``assert_`` makes both legs of the emptying clause
    vacuously discharged, and spends budget on a comparison that cannot fail.
    """

    def with_indices(spec):
        assumes = [i for i, s in enumerate(spec.stmts) if s.kind == "assume"]
        stmt = st.integers(0, len(spec.stmts) - 1)
        if assumes:
            stmt = st.one_of(st.sampled_from(assumes), stmt)
        return st.tuples(
            st.just(spec), st.integers(0, len(spec.decls) - 1), stmt
        )

    return _grammar.general_specs().flatmap(with_indices)


def test_a_conjunct_that_adds_no_information_adds_no_proving_power():
    """Conjoining ``x >= lo`` — true of every member of ``x``'s declared box.

    **The property as posed — "adding a redundant conjunct must not change the
    verdict" — is REFUSED, and this is the version that holds.** A bigger
    formula is a harder formula for a non-relational abstract domain, so
    ``VERIFIED -> UNKNOWN`` under a redundant conjunct is an ordinary precision
    loss in the safe direction. Asserting equality would report the tool
    withholding, which is the one thing it is always allowed to do.

    Two clauses, because a conjunct can be uninformative in two different ways.

    **(a) The conjunct preserves the predicate's element count.** Then ``p`` and
    ``p & c`` are the same claim over the declared set. Forbidden: a
    contradiction (one leg VERIFIED, the other REFUTED), and any gain toward
    VERIFIED — conjunction makes an obligation *stronger*, so it can never
    become easier to prove, and a gain means the analysis narrowed the set it
    was reasoning over.

    **(b) The conjunct drives the element count to ZERO.** ``assume(q)`` means
    "for every element of ``q``", so a ``bool[0]`` conjunct makes ``p & c`` and
    ``c`` alone the same program — both ``∀ ∈ ∅``, both constraining nothing.
    Forbidden: ``p & c`` proving more than ``c`` alone. **This is the size-0
    conjunct defect stated metamorphically**, and note what is deliberately NOT
    asserted: that ``p & c`` equals ``p``. It does not, and should not — the
    conjunct genuinely destroys the constraint.

    Three ways the conjunct would not actually be uninformative are excluded
    before running anything, in ``_grammar.redundant_conjunct_is_sound_for``: a
    NaN bound, a ``bool`` declaration, and — the one that matters — an integer
    bound outside its own dtype's range, which jax reduces mod ``2**bits`` so
    that the conjunct the tool sees is not the conjunct that was written. That
    last exclusion is the open wrap defect; conflating it with a redundancy
    violation would file the wrong report.
    """
    census = _runner.Census("metamorphic/conjunct")

    @_profiles.current().settings(250)
    @given(_conjunct_targets())
    @example(SIZE0_CONJUNCT_EXAMPLE)
    def search(item):
        spec, di, i = item
        census.draw()
        decl = spec.decls[di]
        if not _grammar.redundant_conjunct_is_sound_for(decl):
            census.skip("conjunct would not be redundant for this declaration")
            return
        before = _grammar.static_shape_pred(spec.stmts[i].pred, spec.decls)
        after = _grammar._broadcast(before, decl.shape)
        if before is None or after is None:
            census.skip("conjunction does not broadcast")
            return
        conjoined = _grammar.with_redundant_conjunct(spec, i, decl)
        if _grammar.n_elements(after) == 0 and _grammar.n_elements(before) != 0:
            # (b) the emptying clause: compare against the conjunct ALONE.
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
            census.tag("emptying_compared")
            assert not (
                alone_only.status != "VERIFIED" and both_ways.status == "VERIFIED"
            ), (
                "toward-VERIFIED: a zero-element conjunction proved more than "
                f"the same zero-element predicate alone ({alone_only.status} "
                f"-> {both_ways.status}); a vacuously-true assume narrowed "
                f"something\n--- p & c ---\n{conjoined.render()}"
                f"\n--- c alone ---\n{alone.render()}"
            )
            return
        if after != before:
            census.skip("conjunct changes the predicate's shape")
            return
        # (a) the shape-preserving clause.
        pair = _both(spec, conjoined, census)
        if pair is None:
            return
        a, b = pair
        census.tag("preserving_compared")
        assert not _runner.contradiction(a, b), (
            f"CONTRADICTION under a box-implied conjunct: {a.status} -> "
            f"{b.status}\n{spec.render()}\n--- with the conjunct ---\n"
            f"{conjoined.render()}"
        )
        assert not (a.status != "VERIFIED" and b.status == "VERIFIED"), (
            f"toward-VERIFIED: {a.status} -> VERIFIED under a conjunct that "
            f"cannot add information\n{spec.render()}\n"
            f"--- with the conjunct ---\n{conjoined.render()}"
        )

    search()
    census.require(compared=40, preserving_compared=30, emptying_compared=3)


# ── P2: reordering independent statements ────────────────────────────────────

ASSERT_SWAP_EXAMPLE = (
    _grammar.Spec(
        (_grammar.Decl("x0", (), "float64", -1.0, 1.0),),
        (
            _grammar.Stmt("assert", ("cmp", ">=", ("var", "x0"), ("const", 0.5))),
            _grammar.Stmt("assert", ("cmp", ">=", ("var", "x0"), ("const", 0.0))),
        ),
    ),
    0,
)
"""Two adjacent, independent ``assert_``s, pinned.

Under the ``assert-narrows-like-an-assume`` mutant this harness gives
``('unknown', 'discharged')`` in one order and ``('unknown', 'unknown')`` in
the other — the multiset clause's exact signature. Pinned because a random
draw needs two asserts over the same variable at two thresholds that straddle
the box, which the wide grammar produces only rarely.
"""


def _reorder_targets():
    return st.one_of(
        _grammar.general_specs(), _grammar.integer_program_specs()
    ).flatmap(
        lambda s: st.tuples(st.just(s), st.integers(0, max(0, len(s.stmts) - 2)))
    )


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
    per-obligation statuses must be identical — a multiset, because the
    obligations are renumbered by the swap.

    **Anything else — no contradiction, and only where the harness is not
    vacuous.** Whatever the order, both runs over-approximate the same set of
    admitted points, so one leg VERIFIED and the other REFUTED cannot both be
    right — *unless* the admitted set is empty, where ``∀ ∈ ∅`` is true and a
    REFUTED on the other side of the swap is the tool judging a different,
    non-empty region. That exception is decided exactly, by enumerating the
    declared box in Python integers, and the clause is simply not applied where
    the exact admitted set is empty.

    **Two adjacent ``assume``s are in the second bucket, not the first, and
    that was a correction.** An earlier draft asserted equality for them on the
    grounds that intersection commutes. It does not commute *in a single
    narrowing pass*. The counterexample this docstring used to give —
    ``x, y ∈ [0, 10]``, ``assume(x >= y); assume(y >= 5)`` against the other
    order — **does not reproduce on this tree and never did**: measured with
    ``assert_(x >= 5.0)`` appended as the probe, both orders give UNKNOWN.
    ``assume(x >= y)`` is *relational*, neither side is a point interval, and
    it is DROPPED in both orders, so there is nothing left to commute.

    The one that does reproduce, measured at 9cefc6d, jax 0.11.0, x64 on::

        x, y ∈ float64 [0, 10]
        assume(x == 5.0); assume(y <= x)  ⊢  y <= 5.0   ->  VERIFIED
        assume(y <= x); assume(x == 5.0)  ⊢  y <= 5.0   ->  UNKNOWN

    Same three statements, one transposition, and one order proves what the
    other cannot. The mechanism is the relational-drop rule keying on
    *point-ness*: ``eq`` collapses ``x`` to the point ``[5, 5]``, after which
    ``y <= x`` is no longer relational and narrows ``y``; run the other way
    round, ``y <= x`` is dropped before ``x`` becomes a point. Both are sound
    over-approximations; one is sharper. Asserting equality would have reported
    that as a defect.

    **The reach this does NOT have, stated because a green run would otherwise
    over-claim it.** The historical reordering defect on this tree (``e8b9377``,
    fixed at ``d081d5f``) is only visible when two asserts are permuted ACROSS
    an intervening ``assume``, and asserting invariance under that permutation
    is unsound *today* — forward-scoped narrowing means the two asserts see
    different preconditions after the permutation. So this property's control
    is a mutant, not that commit, and this property would not have caught that
    defect.
    """
    census = _runner.Census("metamorphic/reorder")

    @_profiles.current().settings(250)
    @given(_reorder_targets())
    @example(ASSERT_SWAP_EXAMPLE)
    def search(item):
        spec, i = item
        census.draw()
        if len(spec.stmts) < 2:
            census.skip("fewer than two statements")
            return
        i = min(i, len(spec.stmts) - 2)
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


# ── P5: widening a declared bound ────────────────────────────────────────────

WIDEN_EXAMPLE = (
    _grammar.Spec(
        (_grammar.Decl("x0", (), "float64", -1.0, 0.5),),
        (
            _grammar.Stmt(
                "assert",
                ("cmp", ">=", ("bin", "mul", ("var", "x0"), ("var", "x0")),
                 ("const", 0.5)),
            ),
        ),
    ),
    0,
)
"""The monotonicity probe, pinned.

``x*x >= 0.5`` over ``x ∈ [-1.0, 0.5]``: with all four corner products the
image is ``[0.0, 1.0]`` and the obligation is UNKNOWN in both the narrow and
the widened box. With only the two SAME-CORNER products — the registered
mutant — the narrow box gives ``[0.25, 1.0]`` (still UNKNOWN) and the widened
box ``[-4.0, 3.5]`` gives ``[4.0, 12.25]``, which DISCHARGES. Widening made an
obligation provable; that is non-monotone, and it is the whole property.

Pinned because the shape needs one variable squared against a constant with
bounds that straddle zero asymmetrically, and 2500 examples of the wide grammar
did not produce one.
"""


def _widen_targets():
    return _grammar.general_specs().flatmap(
        lambda s: st.tuples(st.just(s), st.integers(0, len(s.decls) - 1))
    )


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
    @given(_widen_targets())
    @example(WIDEN_EXAMPLE)
    def search(item):
        spec, i = item
        census.draw()
        decl = spec.decls[i]
        if decl.dtype in _grammar.INT_DTYPES and not (
            _grammar.in_dtype_range(decl.dtype, decl.lo)
            and _grammar.in_dtype_range(decl.dtype, decl.hi)
        ):
            # A declared bound outside its own dtype's range is the open wrap
            # class, and there the DECLARED box and the box the tool was handed
            # are different boxes. Anything reported here would be a second
            # sighting of that defect wearing a monotonicity label.
            census.skip("declared bound outside the dtype range (the wrap class)")
            return
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
