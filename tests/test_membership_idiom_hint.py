# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""One cause, five emission points, one gate — and the hint's own text is the
test input.

``MEMBERSHIP_IDIOM_HINT`` was emitted at ONE call site, the dropped-assume
note, while the SAME ⊤ silently weakens every other way a property is stated.
Measured on this tree before the fix:

* ``assume(jnp.all(x >= 0))``      DROPPED, hint printed
* ``assert_(jnp.all(x >= 0))``     obligation ``unknown``, detail "undecided
                                   for 1/1 element(s)", **no note at all**
* ``nonvacuity(jnp.all(...))``     check ``unknown``, stamp "undecided — a
                                   membership condition could not be decided",
                                   **no note at all** — and not even
                                   :func:`verdict.undecided_cause_note`, which
                                   fires only on an undecided OBLIGATION
* ``assume_mode="inert"``          no hint, in a supported mode
* mixed conjunction                ``assume((x >= lo) & jnp.all(x <= hi))``
                                   narrows on the first conjunct and dropped
                                   the second in silence

The cause is a REGISTRY ASYMMETRY, pinned below: ``reduce_or`` has an interval
transfer in both registries and ``reduce_and`` has one in neither, so
``jnp.any`` decides and ``jnp.all`` does not.

**THE REWRITES ARE READ OUT OF THE SHIPPED STRING.** The first version of this
file hand-copied them into a table, and the table had already drifted from the
string it claimed to test (a one-sided hinge here, a two-sided one shipped —
different programs, narrowing different variables). Worse, replacing the whole
hint body with ``"Rewrite it as jnp.all(jnp.all(k >= lo)), or as jnp.min(k) >=
lo; both decide."`` — two dead ends — left the file green. A hand-copied
predicate is the same defect class as a hand-copied bound.

So :func:`_rewrites` extracts every backticked span of the LIVE
``MEMBERSHIP_IDIOM_HINT`` that evaluates to a boolean predicate over a declared
array, and every test below executes those. Editing the text edits what is
tested; naming a form that does not decide fails here before it reaches a user
who is already stuck.

And the ATTRIBUTION is pinned, not just the forms:
``propagate._REAL_ONLY_MARKER`` is a sentinel the text carries, every rewrite
named after it is claimed to work under ``semantics="real"`` only, and
:func:`test_the_texts_ieee_attribution_matches_the_measured_partition` requires
that split to equal the measured one. Inverting the attribution fails there.
It did not before: the test asserted only that the words "ieee" and
"reduce_sum" appeared *somewhere*, so a text blaming the wrong form passed.

**WHAT THIS MECHANISM DOES NOT COVER**, stated because the previous version of
this docstring claimed "every sentence in the text is measured here" and that
was false:

* Only **backticked** spans are seen. Advice written as bare prose
  ("just use jnp.min") is invisible to every test in this file.
* Only spans spelled with ``k`` / ``lo`` / ``hi`` are executed. A rewrite
  spelled with any other variable raises ``NameError`` in the classifier and
  is silently skipped — it is neither tested nor reported.
* Only the real-only/both-modes SPLIT is pinned. The finer prose within it
  (which primitive declines, at what size, why) is checked by two coarse
  consistency assertions and no more.
* The assume-differentiation clause is pinned as a partition (exactly one
  named form stays certified) — not sentence by sentence.

Both semantics, everywhere. The rewrite tests were the only ones in this file
not parametrized over ``semantics``, and that is exactly where a false claim
hid: two of the three shipped forms fall to ⊤ at ``reduce_sum`` under
``semantics="ieee"`` — the mode ``docs/preconditions.md`` tells users to pose
float-boundary facts in.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from _solver_gate import need_solver  # noqa: E402
from stelling import propagate as P  # noqa: E402
from stelling import solvers as S  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.harness import any_array, assert_, assume, nonvacuity  # noqa: E402
from stelling.preconditions import check  # noqa: E402

LO, HI = 0.0, 10.0
SEMANTICS = ["real", "ieee"]
HINT = P.MEMBERSHIP_IDIOM_HINT
POINTER = P.MEMBERSHIP_IDIOM_POINTER


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# -- reading the rewrites out of the shipped text -----------------------------

_BACKTICK = re.compile(r"`([^`]+)`")


def _rewrites() -> list[str]:
    """Every backticked span of the live hint that IS a membership predicate,
    in the order the text names them.

    The classifier is EXECUTION, not pattern matching: a span is a rewrite iff
    evaluating it against an array ``k`` with bounds ``lo``/``hi`` yields a
    boolean array. That drops ``jnp.all(...)`` (raises on the Ellipsis), the
    bare primitive names, and the API names — with no exclusion list to keep in
    sync with the prose."""
    found: list[str] = []
    # no explicit dtype: this runs at COLLECTION time, before the x64 fixture,
    # and the classifier only asks "does this yield a bool array"
    probe = jnp.ones((3,))
    for span in _BACKTICK.findall(HINT):
        if span in found:
            continue
        try:
            value = eval(  # noqa: S307 - the input is this repo's own constant
                span,
                {"__builtins__": {}},
                {"jnp": jnp, "k": probe, "lo": LO, "hi": HI},
            )
        except Exception:
            continue
        if getattr(value, "shape", None) is None:
            continue
        if str(getattr(value, "dtype", "")) != "bool":
            continue
        found.append(span)
    return found


def _pred(span: str, k):
    return eval(  # noqa: S307 - the input is this repo's own constant
        span, {"__builtins__": {}}, {"jnp": jnp, "k": k, "lo": LO, "hi": HI}
    )


def _run(h, semantics="real", assume_mode="constrain"):
    return P.propagate(
        transcribe(jax.make_jaxpr(h)()),
        assume_mode=assume_mode,
        semantics=semantics,
    )


def _hinted(p):
    return [n for n in p.notes if HINT in n or POINTER in n]


def test_the_text_names_rewrites_at_all():
    """The extractor is the instrument for every claim below; a text that
    yields nothing would make those tests vacuously green. Three today: the
    elementwise form and the two arithmetic ones."""
    assert len(_rewrites()) >= 3, (
        f"MEMBERSHIP_IDIOM_HINT yields {_rewrites()} executable rewrite(s); "
        f"the tests below measure exactly what this extracts, so a text that "
        f"names none is a text nothing checks"
    )


# -- the cause ----------------------------------------------------------------


def test_the_registry_asymmetry_the_hint_asserts():
    """The hint's first sentence is a MEMBERSHIP fact about two registries,
    and a census can see it. Register a `reduce_and` row and the sentence
    "which has no interval transfer" becomes a lie printed at a user — so the
    row and the text move together or this fails."""
    assert "reduce_or" in P.TRANSFERS
    assert "reduce_or" in P.IEEE_TRANSFERS
    assert "reduce_and" not in P.TRANSFERS, (
        "a `reduce_and` interval transfer is registered, so "
        "MEMBERSHIP_IDIOM_HINT's 'which has no interval transfer' is false — "
        "rewrite the hint (and re-measure every rewrite it names) before "
        "landing the row"
    )
    assert "reduce_and" not in P.IEEE_TRANSFERS, (
        "same, for the ieee registry: the hint is emitted under both "
        "semantics and claims the gap unconditionally"
    )
    assert "no interval transfer" in HINT


# -- every rewrite the text names, executed -----------------------------------


def _assert_status(span, n, semantics):
    def h():
        x = any_array((n,), "float64", (1.0, 9.0))
        return (assert_(_pred(span, x)),)

    p = _run(h, semantics)
    return p.obligations[0].status, p.coverage.unknown_primitives


def _nonvacuity_status(span, n, semantics):
    def h():
        x = any_array((2,), "float64", (LO, HI))
        pt = jnp.ones((n,), jnp.float64)
        return (assert_(jnp.sum(x) >= 0.0), nonvacuity(_pred(span, pt)))

    p = _run(h, semantics)
    return p.nonvacuity_checks[0].status, p.coverage.unknown_primitives


def _assume_state(span, n, semantics):
    def h():
        x = any_array((n,), "float64", (-10.0, 10.0))
        assume(_pred(span, x))
        return (assert_(x >= -100.0),)

    p = _run(h, semantics)
    return ("CONSTRAIN" if p.coverage.constrained else
            "DROP" if p.assume_dropped else "neither")


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("span", _rewrites())
def test_every_shipped_rewrite_decides_under_real(span, n):
    """The text's unqualified claim, on the assert and nonvacuity faces.
    `semantics="real"` is where it holds without a caveat."""
    assert _assert_status(span, n, "real")[0] == "discharged", span
    assert _nonvacuity_status(span, n, "real")[0] == "discharged", span


@pytest.mark.parametrize("span", _rewrites())
@pytest.mark.parametrize("semantics", SEMANTICS)
def test_every_shipped_rewrite_makes_an_assume_constrain(span, semantics):
    """The claim that survived the audit unchanged, now measured out of the
    string and under BOTH semantics rather than one."""
    assert _assume_state(span, 3, semantics) == "CONSTRAIN", span


def _ieee_partition():
    """(forms that decide under ieee at every size, {form: ⊤ primitives})."""
    survives, stuck = [], {}
    for span in _rewrites():
        tops: set[str] = set()
        for n in (1, 2, 3, 4):
            status, prims = _assert_status(span, n, "ieee")
            if status != "discharged":
                tops.update(p for p, _ in prims)
        if tops:
            stuck[span] = tops
        else:
            survives.append(span)
    return survives, stuck


def test_the_texts_ieee_attribution_matches_the_measured_partition():
    """"All three decide" was FALSE under `semantics="ieee"`, and the hint
    prints identically in both modes, so a stuck reader in the mode
    docs/preconditions.md recommends for float-boundary facts was handed a
    second dead end.

    The first repair measured the partition but asked the text only to CONTAIN
    the words "ieee" and "reduce_sum" — so a text that blamed the wrong form
    passed. The attribution is now STRUCTURE: `_REAL_ONLY_MARKER` splits the
    string, and the forms named after it must be exactly the forms measured to
    fail. Inverting the attribution fails here."""
    survives, stuck = _ieee_partition()
    assert survives, (
        "no form the hint names decides under semantics='ieee' at every "
        "size; the text must stop recommending them there"
    )
    assert P._REAL_ONLY_MARKER in HINT, (
        "the hint no longer carries the real-only marker, so its ieee "
        "attribution is unpinned prose again"
    )
    cut = HINT.index(P._REAL_ONLY_MARKER)
    claimed_real_only = {s for s in _rewrites() if HINT.index(f"`{s}`") > cut}

    assert claimed_real_only == set(stuck), (
        f"the text claims {sorted(claimed_real_only)} are real-only; measured "
        f"real-only is {sorted(stuck)}. Inverting or mis-attributing which "
        f"form survives ieee sends the reader to the one that does not."
    )
    assert set(survives) & set(_rewrites()) - claimed_real_only == set(survives), (
        "a form measured to survive ieee is named after the real-only marker"
    )
    for span, tops in stuck.items():
        assert tops == {"reduce_sum"}, (
            f"{span!r} fails under ieee at {sorted(tops)}, not at "
            f"reduce_sum as the text says"
        )
    # ...and the two size qualifiers the text draws inside that group: one
    # form decides at small n and one never does, so both phrases must be
    # earned rather than inherited from a previous edit
    small_n_ok = {
        s for s in stuck if _assert_status(s, 2, "ieee")[0] == "discharged"
    }
    assert bool(small_n_ok) == ("three or more" in HINT), (
        f"{sorted(small_n_ok)} decide under ieee at n=2, so the text must "
        f"qualify by size (or must stop, if none do)"
    )
    assert bool(set(stuck) - small_n_ok) == ("at every size" in HINT), (
        f"{sorted(set(stuck) - small_n_ok)} fail under ieee at every size, "
        f"so the text must say so (or must stop, if none do)"
    )


def test_the_form_the_text_names_first_is_one_that_survives_ieee():
    """The ordering is a claim ("named FIRST"), so it is measured. Naming a
    conditionally-working form first is how a reader picks the wrong one."""
    first = _rewrites()[0]
    for n in (1, 2, 3, 4):
        assert _assert_status(first, n, "ieee")[0] == "discharged", (
            f"the hint names {first!r} first, but it does not decide under "
            f"ieee at n={n}"
        )


def test_the_texts_assume_clause_partitions_the_shipped_rewrites(monkeypatch):
    """The differentiation clause, measured as a PARTITION of whatever the
    text names — no form spelled out here.

    Exactly one named form narrows the declared input, stays certified and
    leaves the REFUTED face reachable; every other named form narrows the
    reduction's own intermediate, raises satisfiability-UNCERTIFIED, and has
    its definite violations withheld. That is what the clause claims.

    **The withholding half is measured with the NON-EMPTINESS CERTIFICATE
    closed, and the clause says "unless" for the same reason.** The
    declared box here is `[-10, 10]^3` and the membership condition the
    rewrites express is satisfied at plenty of its points, so a probed
    point witnesses the assume and the withheld refutation correctly comes
    back — on the arithmetic forms, which evaluate to a definite value at
    a point, though NOT on a `jnp.all` whose `reduce_and` is ⊤ at a point
    as much as over a box. That is a fact about the certificate, not about
    the partition this test measures, so the certificate's route is closed
    and the partition is read underneath it.
    """
    from stelling import exactness

    monkeypatch.setattr(
        exactness, "certifies_point_witness", lambda **k: False
    )

    def violation_under(span):
        def h():
            x = any_array((3,), "float64", (-10.0, 10.0))
            assume(_pred(span, x))
            return (assert_(jnp.sum(x) <= -100.0),)

        return _run(h)

    certified, uncertified = [], []
    for span in _rewrites():
        p = violation_under(span)
        target = uncertified if any("UNCERTIFIED" in n for n in p.notes) else certified
        target.append((span, p))

    assert len(certified) == 1, (
        f"the clause says exactly one named form stays certified; measured "
        f"{[s for s, _ in certified]}"
    )
    assert certified[0][0] == _rewrites()[0], (
        "the certified form must be the one the text names first"
    )
    assert certified[0][1].obligations[0].status == "violated-over-set"
    assert uncertified, "the clause contrasts against something; nothing did"
    for span, p in uncertified:
        assert p.obligations[0].status == "unknown", span
        assert "WITHHELD from REFUTED" in p.obligations[0].detail, span


def test_only_the_first_named_form_discharges_the_downstream_obligation():
    """The same difference from the VERIFIED side: a narrowing that lands on
    a dead intermediate buys the query nothing."""
    def under(span):
        def h():
            x = any_array((3,), "float64", (-10.0, 10.0))
            assume(_pred(span, x))
            return (assert_(jnp.sum(x) >= 0.0),)

        return _run(h).obligations[0].status

    statuses = {span: under(span) for span in _rewrites()}
    first = _rewrites()[0]
    assert statuses[first] == "discharged", statuses
    assert all(s == "unknown" for k, s in statuses.items() if k != first), statuses


# -- the five emission points -------------------------------------------------


def _assume_all():
    x = any_array((3,), "float64", (-10.0, 10.0))
    assume(jnp.all(x >= LO))
    return (assert_(jnp.sum(x) >= 0.0),)


def _assert_all():
    x = any_array((3,), "float64", (1.0, 9.0))
    return (assert_(jnp.all(x >= LO)),)


def _nonvacuity_all():
    x = any_array((3,), "float64", (LO, HI))
    pt = jnp.array([1.0, 2.0, 3.0])
    return (assert_(jnp.sum(x) >= 0.0), nonvacuity(jnp.all(pt >= LO)))


def _mixed_conjunction():
    x = any_array((3,), "float64", (-10.0, 10.0))
    assume((x >= LO) & jnp.all(x <= HI))
    return (assert_(jnp.sum(x) >= 0.0),)


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_assume_path_hints(semantics):
    p = _run(_assume_all, semantics)
    assert p.assume_dropped
    assert len(_hinted(p)) == 1
    assert "DROPPED" in _hinted(p)[0]


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_inert_mode_assume_path_hints(semantics):
    """`assume_mode="inert"` is a SUPPORTED mode and was the one path still
    printing nothing after the first pass, while docs/harness-api.md said all
    three paths print the rewrite."""
    p = _run(_assume_all, semantics, assume_mode="inert")
    assert p.coverage.inert == 1
    assert len(_hinted(p)) == 1


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_mixed_conjunction_assume_path_hints(semantics):
    """One constrainable conjunct and one `jnp.all` conjunct: the assume
    CONSTRAINS, so the dropped half took the conjunct-DROPPED branch, which
    reached no hint at all."""
    p = _run(_mixed_conjunction, semantics)
    assert p.coverage.constrained == 1
    assert len(_hinted(p)) == 1
    assert "conjunct DROPPED" in _hinted(p)[0]


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_assert_path_hints(semantics):
    p = _run(_assert_all, semantics)
    assert p.obligations[0].status == "unknown"
    notes = _hinted(p)
    assert len(notes) == 1, p.notes
    assert notes[0].startswith("obligation UNDECIDED at ")
    assert "'reduce_and'" in notes[0]


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_nonvacuity_path_hints(semantics):
    p = _run(_nonvacuity_all, semantics)
    assert p.nonvacuity_checks[0].status == "unknown"
    notes = _hinted(p)
    assert len(notes) == 1, p.notes
    assert notes[0].startswith("nonvacuity condition UNDECIDED at ")


# -- one gate, not three ------------------------------------------------------

TWO_SIDED = {
    "all(a) & all(b)": lambda v: jnp.all(v >= LO) & jnp.all(v <= HI),
    "logical_and(all, all)": lambda v: jnp.logical_and(
        jnp.all(v >= LO), jnp.all(v <= HI)
    ),
    "keepdims + squeeze": lambda v: jnp.squeeze(jnp.all(v >= LO, keepdims=True)),
    "keepdims + reshape": lambda v: jnp.reshape(
        jnp.all(v >= LO, keepdims=True), ()
    ),
    "all(a & b)": lambda v: jnp.all((v >= LO) & (v <= HI)),
}


@pytest.mark.parametrize("name", sorted(TWO_SIDED))
def test_every_spelling_of_the_canonical_idiom_hints_on_every_face(name):
    """`jnp.all(k >= lo) & jnp.all(k <= hi)` is the exact two-sided condition
    the arithmetic rewrite is written for, and it hinted as an `assume` (whose
    gate substring-matched its own dropped reasons) and NOT as an `assert_` or
    a `nonvacuity` (whose gate read the predicate itself, and the predicate
    there is the `and` output). Three gates, one documented instrument. Now
    one gate."""
    form = TWO_SIDED[name]

    def h_assert():
        x = any_array((3,), "float64", (1.0, 9.0))
        return (assert_(form(x)),)

    def h_nonvac():
        x = any_array((2,), "float64", (LO, HI))
        pt = jnp.ones((3,), jnp.float64)
        return (assert_(jnp.sum(x) >= 0.0), nonvacuity(form(pt)))

    def h_assume():
        x = any_array((3,), "float64", (-10.0, 10.0))
        assume(form(x))
        return (assert_(x >= -100.0),)

    assert len(_hinted(_run(h_assert))) == 1, name
    assert len(_hinted(_run(h_nonvac))) == 1, name
    assert len(_hinted(_run(h_assume))) == 1, name


NESTED = {
    "nested all": lambda M: jnp.all(jnp.all(M >= LO, axis=1)),
    "nested all in a conjunction": lambda M: (
        jnp.all(M >= LO) & jnp.all(jnp.all(M >= LO, axis=1))
    ),
}


@pytest.mark.parametrize("name", sorted(NESTED))
@pytest.mark.parametrize("semantics", SEMANTICS)
def test_a_NESTED_reduction_still_hints(name, semantics):
    """The dead-end guard must not count a `reduce_and` ⊤ against itself.

    Scoped to one reduction's own operands it did: `jnp.all(jnp.all(M >= 0,
    axis=1))` has a `reduce_and` ⊤ under a `reduce_and` ⊤ — the ONE thing the
    rewrite removes — and got nothing, on a program where deleting both
    reductions reaches `discharged` at 100% coverage in both semantics. A
    conjunction holding a nested `all` lost the hint for its fixable half
    too."""
    form = NESTED[name]

    def h():
        m = any_array((3, 4), "float64", (1.0, 2.0))
        return (assert_(form(m)),)

    assert len(_hinted(_run(h, semantics))) == 1, name


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_advice_actually_lands_on_a_nested_reduction(semantics):
    """...and the premise: deleting BOTH reductions decides, at full
    coverage, which is why withdrawing the hint there was withdrawing correct
    guidance rather than avoiding a dead end."""
    def h():
        m = any_array((3, 4), "float64", (1.0, 2.0))
        return (assert_(m >= LO),)

    p = _run(h, semantics)
    assert p.obligations[0].status == "discharged"
    assert p.coverage.unknown == 0
    assert p.coverage.known == p.coverage.total


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_separate_calls_are_the_conjunction_and_carry_no_shape_condition(
    semantics,
):
    """The `&` caveat the text now carries. `jnp.all(w >= 0) & jnp.all(b >= 0)`
    over differently-shaped arrays traces with the reductions in place and
    raises once they are gone; two calls do not, and decide.

    The SENTENCE is pinned too, not only the behaviour: measuring the
    TypeError proves the hazard is real but leaves the caveat droppable from
    the text, which is exactly the reader-facing half."""
    assert P._SEPARATE_CALLS_MARKER in HINT, (
        "the hint no longer tells the reader to split a conjunction into "
        "separate calls, and 'delete the reduction' applied verbatim to "
        "differently-shaped conjuncts does not trace"
    )
    w_shape, b_shape = (3,), (5,)

    def deleted_into_one_and():
        w = any_array(w_shape, "float64", (1.0, 2.0))
        b = any_array(b_shape, "float64", (1.0, 2.0))
        return (assert_((w >= LO) & (b >= LO)),)

    with pytest.raises(TypeError, match="incompatible shapes"):
        jax.make_jaxpr(deleted_into_one_and)()

    def separate_calls():
        w = any_array(w_shape, "float64", (1.0, 2.0))
        b = any_array(b_shape, "float64", (1.0, 2.0))
        return (assert_(w >= LO), assert_(b >= LO))

    p = _run(separate_calls, semantics)
    assert [o.status for o in p.obligations] == ["discharged", "discharged"]

    def separate_assumes():
        w = any_array(w_shape, "float64", (-1.0, 1.0))
        b = any_array(b_shape, "float64", (-1.0, 1.0))
        assume(w >= LO)
        assume(b >= LO)
        return (assert_(w >= LO), assert_(b >= LO))

    q = _run(separate_assumes, semantics)
    assert q.coverage.constrained == 2
    assert [o.status for o in q.obligations] == ["discharged", "discharged"]


def test_a_reduction_over_another_top_gets_no_hint():
    """The reason the gate has a dead-end guard.

    `assert_(jnp.all(jnp.max(x) >= 0.0))` satisfies every structural test —
    the judged predicate genuinely IS a `reduce_and` ⊤ — and deleting the
    reduction fixes nothing, because `reduce_max` has no transfer either. The
    premise is measured below too: no rewrite the hint names rescues it."""
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        return (assert_(jnp.all(jnp.max(x) >= LO)),)

    p = _run(h)
    assert p.obligations[0].status == "unknown"
    assert dict(p.coverage.unknown_primitives) == {"reduce_and": 1, "reduce_max": 1}
    assert _hinted(p) == []

    for span in _rewrites():
        def hr(span=span):
            x = any_array((3,), "float64", (1.0, 2.0))
            return (assert_(_pred(span, jnp.max(x))),)

        assert _run(hr).obligations[0].status == "unknown", span


# -- negative controls --------------------------------------------------------


def test_a_dead_end_in_a_SIBLING_conjunct_gets_no_hint():
    """The dead-end guard's other half, and the reason its scope is the whole
    predicate. `assert_(jnp.all(x >= 0) & (jnp.max(y) >= 0))` and
    `assert_(jnp.all(jnp.max(x) >= 0))` are the same two primitives and the
    same dead end; scoped to one reduction's operands the gate said no to the
    second and yes to the first — the decision turned on spelling."""
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        y = any_array((3,), "float64", (1.0, 2.0))
        return (assert_(jnp.all(x >= LO) & (jnp.max(y) >= LO)),)

    p = _run(h)
    assert dict(p.coverage.unknown_primitives) == {"reduce_and": 1, "reduce_max": 1}
    assert _hinted(p) == []


def test_the_walk_cap_goes_silent_rather_than_guess():
    """A bound on the PREDICATE'S CONE, not on the program, and documented in
    docs/harness-api.md as a numbered gate condition. Degrading to no hint is
    right; an undocumented, unpinned boundary is what this pins."""
    def chained(m):
        def h():
            x = any_array((2,), "float64", (1.0, 2.0))
            acc = jnp.all(x >= LO)
            for _ in range(m - 1):
                acc = acc & jnp.all(x >= LO)
            return (assert_(acc),)

        return h

    assert len(_hinted(_run(chained(170)))) == 1
    assert _hinted(_run(chained(171))) == []

    # ...and a LARGE program with a small predicate is unaffected: the cap is
    # not a program-size condition, which the doc would otherwise imply
    def big_program_small_predicate():
        x = any_array((3,), "float64", (1.0, 2.0))
        acc = x
        for _ in range(400):
            acc = acc + x
        return (assert_(jnp.all(x >= LO)), assert_(jnp.sum(acc) >= 0.0))

    p = _run(big_program_small_predicate)
    assert p.coverage.total > 400
    assert len(_hinted(p)) == 1


def test_the_documented_transparent_set_is_the_implemented_one():
    """docs/harness-api.md enumerates the ops the gate walks through, and said
    "`&`/reshape/squeeze only" while the code also walked broadcast_in_dim and
    convert_element_type — which the doc's own keepdims example needs. Prose
    that lists a set drifts from the set unless something reads both."""
    doc = (
        Path(__file__).resolve().parent.parent / "docs" / "harness-api.md"
    ).read_text(encoding="utf-8")
    missing = [
        p for p in P._MEMBERSHIP_HINT_TRANSPARENT if f"`{p}`" not in doc
    ]
    assert not missing, (
        f"docs/harness-api.md does not name {sorted(missing)}, which the gate "
        f"walks through"
    )


def test_a_plain_straddle_gets_no_hint():
    """The commonest UNKNOWN of all. Coverage is complete; there is no
    reduction to delete."""
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return (assert_(jnp.sum(x) >= 0.0),)

    p = _run(h)
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 0
    assert _hinted(p) == []


def test_a_top_upstream_of_the_predicate_gets_no_hint():
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        return (assert_(jnp.max(x) >= LO),)

    p = _run(h)
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown_primitives == (("reduce_max", 1),)
    assert _hinted(p) == []


def test_a_predicate_that_IS_another_primitives_top_gets_no_hint():
    """The gate's own mutant: here the predicate is a correctly-evaluated
    NOT(definitely-true) = definitely-false, so the obligation is
    violated-over-set. No hint should fire (the hint is about membership
    idioms on UNKNOWN obligations, not violated ones)."""
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        return (assert_(jnp.logical_not(x >= LO)),)

    p = _run(h)
    assert p.obligations[0].status == "violated-over-set"
    assert p.coverage.unknown_primitives == ()
    assert _hinted(p) == []


@pytest.mark.parametrize(
    "name,form",
    [
        ("or", lambda v: jnp.all(v >= LO) | jnp.all(v <= HI)),
        ("not", lambda v: jnp.logical_not(jnp.all(v >= LO))),
    ],
)
def test_connectives_that_do_not_survive_deletion_get_no_hint(name, form):
    """`or` and `not` are boolean and are deliberately NOT walked through:
    `all(a) | all(b)` is not elementwise `a | b`, and `~all(a)` is not `~a`,
    so "delete the reduction" would change the stated property."""
    def h():
        x = any_array((3,), "float64", (1.0, 9.0))
        return (assert_(form(x)),)

    p = _run(h)
    assert dict(p.coverage.unknown_primitives)["reduce_and"] >= 1
    assert _hinted(p) == []


def test_a_reduction_used_as_a_SELECTOR_gets_no_hint():
    """`jnp.where(jnp.all(...), a, b)`: the reduction is control flow, not the
    judged property, and the obligation is undecided for its own reason."""
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        y = jnp.where(jnp.all(x >= LO), x, x)
        return (assert_(jnp.sum(y) >= 0.0),)

    p = _run(h)
    assert p.obligations[0].status == "unknown"
    assert ("reduce_and", 1) in p.coverage.unknown_primitives
    assert _hinted(p) == []


def test_a_SIZE_ZERO_reduction_is_decided_and_gets_no_hint_NONVACUITY():
    """The status guard is load-bearing, not belt-and-braces. A `reduce_and`
    with a non-reduced zero-length axis is `discharged` — vacuously, matching
    `jnp.all` of an empty array — while its box is still stelling's ⊤ and the
    gate still says yes. Drop the guard and a DECIDED face carries a note
    calling itself UNDECIDED."""
    def h():
        x = any_array((2,), "float64", (LO, HI))
        pt = jnp.zeros((0, 2), jnp.float64)
        return (assert_(jnp.sum(x) >= 0.0), nonvacuity(jnp.all(pt >= LO, axis=1)))

    p = _run(h)
    assert p.nonvacuity_checks[0].status == "discharged"
    assert p.nonvacuity_checks[0].detail == P.EMPTY_UNIVERSAL_DETAIL
    # ... and the emptiness reaches the NOTES, not only the detail (audit
    # 0.2.0 B8a, item 6 / M18)
    assert any("VACUOUSLY discharged" in n for n in p.notes), p.notes
    assert ("reduce_and", 1) in p.coverage.unknown_primitives
    assert _hinted(p) == []


def test_a_SIZE_ZERO_reduction_is_decided_and_gets_no_hint_ASSERT():
    """The ASSERT face's twin of the guard above, which had no test of its
    own: the same size-0 `reduce_and` reaches `discharged` on an obligation,
    and without the status guard that decided obligation gains a note reading
    UNDECIDED."""
    def h():
        x = any_array((0, 2), "float64", (1.0, 2.0))
        return (assert_(jnp.all(x >= LO, axis=1)),)

    p = _run(h)
    assert p.obligations[0].status == "discharged"
    assert p.obligations[0].detail == P.EMPTY_UNIVERSAL_DETAIL
    assert any("VACUOUSLY discharged" in n for n in p.notes), p.notes
    assert ("reduce_and", 1) in p.coverage.unknown_primitives
    assert _hinted(p) == []


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_a_MANY_conjunct_assume_hints_once_not_once_per_conjunct(semantics):
    """The `i == 0` restriction on the conjunct-DROPPED loop. Without it a
    three-conjunct assume emits the locator twice for one predicate — the
    dedupe would keep the BODY single, but two notes about one fact is the
    repetition the pointer exists to prevent."""
    def h():
        x = any_array((3,), "float64", (-10.0, 10.0))
        assume((x >= LO) & jnp.all(x <= HI) & jnp.all(x <= 100.0))
        return (assert_(jnp.sum(x) >= 0.0),)

    p = _run(h, semantics)
    assert p.coverage.constrained == 1
    assert sum("conjunct DROPPED" in n for n in p.notes) == 2
    assert len(_hinted(p)) == 1


def test_the_decided_faces_get_no_hint():
    """`jnp.any` lowers to `reduce_or`, which IS registered — the asymmetry,
    from the deciding side."""
    def h():
        x = any_array((3,), "float64", (1.0, 9.0))
        pt = jnp.array([1.0, 2.0, 3.0])
        return (assert_(jnp.any(x >= LO)), nonvacuity(pt >= LO))

    p = _run(h)
    assert p.obligations[0].status == "discharged"
    assert p.nonvacuity_checks[0].status == "discharged"
    assert _hinted(p) == []


# -- the body is printed once -------------------------------------------------


def test_one_query_stating_all_three_prints_the_body_once():
    """The hint is ~1.2k characters and one run can state a property on three
    faces; three copies is the complaint `_note_decline`'s dedupe exists for,
    and verbatim dedupe cannot fire because each face names its own site."""
    def h():
        x = any_array((3,), "float64", (-10.0, 10.0))
        pt = jnp.array([1.0, 2.0, 3.0])
        assume(jnp.all(x >= LO))
        return (assert_(jnp.all(x <= HI)), nonvacuity(jnp.all(pt >= LO)))

    p = _run(h)
    hinted = _hinted(p)
    assert len(hinted) == 3
    assert sum(HINT in n for n in p.notes) == 1
    assert sum(POINTER in n for n in p.notes) == 2
    assert sum(n.startswith("assume constraint DROPPED") for n in hinted) == 1
    assert sum(n.startswith("obligation UNDECIDED") for n in hinted) == 1
    assert sum(n.startswith("nonvacuity condition UNDECIDED") for n in hinted) == 1


def test_two_faces_on_one_source_line_say_it_once():
    """Same subject, same site, same origin is one fact. Before the dedupe two
    asserts on one line produced two byte-identical pointer notes."""
    def h():
        x = any_array((3,), "float64", (1.0, 9.0))
        assume(jnp.all(x >= LO))
        return (assert_(jnp.all(x >= LO)), assert_(jnp.all(x >= LO)))

    p = _run(h)
    assert len(p.notes) == len(set(p.notes))
    assert len(_hinted(p)) == 2


# -- the hint reaches the reader ----------------------------------------------


def test_the_hint_survives_the_front_door_without_a_solver():
    v = check(_assert_all, vacuity_mode="inputs-only")
    assert v.status == "UNKNOWN"
    assert any(HINT in n for n in v.notes)


@need_solver
def test_the_hint_survives_escalation_which_replaces_the_detail():
    """Why the hint is a NOTE and not the obligation detail. Measured: the
    escalation record's detail REPLACES the propagation's, so a detail-only
    hint would vanish for exactly the reader who paid for a solver."""
    cj = transcribe(jax.make_jaxpr(_assert_all)())
    p = P.propagate(cj)
    esc = S.escalate(cj, p, S.SolverConfig(timeout_ms=60_000))
    assert HINT not in esc.records[0].detail
    v = check(_assert_all, vacuity_mode="inputs-only", solver_timeout_ms=60_000)
    assert any(HINT in n for n in v.notes)


# -- diagnostics only ---------------------------------------------------------


@pytest.mark.parametrize(
    "harness,semantics,statuses,nonvac",
    [
        (_assume_all, "real", ["unknown"], []),
        (_assume_all, "ieee", ["unknown"], []),
        (_assert_all, "real", ["unknown"], []),
        (_assert_all, "ieee", ["unknown"], []),
        (_mixed_conjunction, "real", ["discharged"], []),
        (_mixed_conjunction, "ieee", ["unknown"], []),
        (_nonvacuity_all, "real", ["discharged"], ["unknown"]),
        # ieee's reduce_sum declines above two elements, so THIS row's
        # obligation was already unknown for a reason of its own — pinned as
        # measured, not normalized away
        (_nonvacuity_all, "ieee", ["unknown"], ["unknown"]),
    ],
)
def test_the_hint_moves_no_verdict(harness, semantics, statuses, nonvac):
    """The statuses on every emitting path, pinned at what they were before
    the hint reached them."""
    p = _run(harness, semantics)
    assert [o.status for o in p.obligations] == statuses
    assert [c.status for c in p.nonvacuity_checks] == nonvac
