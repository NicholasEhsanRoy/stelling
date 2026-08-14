# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE ASSUME DISPOSITION LEDGER, and the release rule built on it.

A definite violation found by a solver may be released from withholding
only when EVERY ``assume`` the user wrote is accounted for on that
obligation's query: applied in the interval domain, certainly true over the
boxes in force, or emitted to that obligation's script about the terms its
operands denote. Anything else means the solver ran over a SUPERSET of the
admitted region, and its witness may lie outside it.

**Why this file is about the invariant and not about a harness.** The rule
it replaces compared two integers — how many relational assumes the
propagation forwarded, against how many a script emitted — and that shape
produced a false REFUTED twice:

* **audit 0.2.0 S6.** The denominator counted only the RELATIONAL assumes
  while the flag gating the whole rule (``assume_dropped``) is set by ANY
  drop reason: a predicate from a primitive outside ``{ge,gt,le,lt,eq}``,
  ``and`` on non-bool, wrong operand count, a non-finite or subnormal
  constant bound, an out-of-scope producer. ``assume(x < y)`` beside
  ``assume(jnp.logical_or(x > 5, y > 5))`` therefore satisfied ``1 == 1``
  and released a witness violating the ``or``.
* **and then the branch-scoping repair.** Once a relational assume traced
  inside a ``lax.cond`` branch stopped being forwarded, it stopped being in
  ``relational_assumes`` — so the DENOMINATOR moved. One ordinary assume
  emitted plus one branch-scoped assume dropped satisfied ``1 == 1`` again,
  and a REFUTED came back whose witness runs the branch and falsifies its
  precondition.

Both are one defect: an equality between two populations that nothing
forces to be the same population, maintained in two files. Fixing one and
leaving the other is exactly the mistake, so the tests below are written
against the RULE — including one that invents a disposition nobody has
taught the rule about and requires it to refuse.
"""

from __future__ import annotations

import dataclasses

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from stelling import propagate as P  # noqa: E402
from stelling import solvers as S  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.harness import any_array, assert_, assume  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import (  # noqa: E402
    ASSUME_APPLIED,
    ASSUME_DROPPED,
    ASSUME_FORWARDED,
    ASSUME_NOOP,
    AssumeDisposition,
    unaccounted_assumes,
)


def _prop(h, **kw):
    return P.propagate(transcribe(jax.make_jaxpr(h)()), **kw)


def _run(h, **kw):
    return check(h, vacuity_mode="inputs-only", solver_timeout_ms=8000, **kw)


def _kinds(h):
    return [e.kind for e in _prop(h).assume_ledger]


# ---------------------------------------------------------------------------
# the rule itself, with no jaxpr in sight
# ---------------------------------------------------------------------------

def test_applied_and_noop_are_accounted_without_any_solver():
    """The judged set is already inside those conjuncts, so nothing has to be
    told to a solver and an empty emitted set accounts for them."""
    ledger = (
        AssumeDisposition(kind=ASSUME_APPLIED, reason="narrowed var 3"),
        AssumeDisposition(kind=ASSUME_NOOP, reason="definitely true"),
    )
    assert unaccounted_assumes(ledger, ()) == ()


def test_a_forwarded_assume_is_accounted_only_when_ITS_index_was_emitted():
    """Not "one was emitted" — THAT one. A count cannot tell the difference,
    which is the whole reason the join is on identity."""
    ledger = (
        AssumeDisposition(kind=ASSUME_FORWARDED, reason="a", forwarded_index=0),
        AssumeDisposition(kind=ASSUME_FORWARDED, reason="b", forwarded_index=1),
    )
    assert unaccounted_assumes(ledger, (0, 1)) == ()
    # the same CARDINALITY as the ledger, and the wrong assume
    (missing,) = unaccounted_assumes(ledger, (0, 0))
    assert missing.reason == "b"
    (missing,) = unaccounted_assumes(ledger, (1, 1))
    assert missing.reason == "a"


def test_a_drop_is_never_accounted_however_many_axioms_were_emitted():
    """S6's shape, stated as the rule: an emitted relational assume cannot
    stand in for a differently-dropped one."""
    ledger = (
        AssumeDisposition(kind=ASSUME_FORWARDED, reason="x < y",
                          forwarded_index=0),
        AssumeDisposition(kind=ASSUME_DROPPED, reason="the 'or' assume"),
    )
    (missing,) = unaccounted_assumes(ledger, (0,))
    assert missing.reason == "the 'or' assume"


def test_a_disposition_the_rule_was_never_taught_is_UNACCOUNTED():
    """THE REGRESSION THAT MATTERS. `unaccounted_assumes` whitelists the
    dispositions it can show the judged set to be inside of; a kind added
    later that nobody taught it must fail CLOSED.

    A blacklist ("everything except `dropped` releases") passes every other
    test in this file and fails this one — and a blacklist is precisely how a
    new drop reason silently shrank a denominator twice.
    """
    invented = AssumeDisposition(
        kind="a-reason-invented-after-this-code-was-written",
        reason="whatever a later batch decides this means",
    )
    assert unaccounted_assumes((invented,), ()) == (invented,)
    assert unaccounted_assumes((invented,), (0, 1, 2)) == (invented,)


def test_the_forwarded_index_sentinel_matches_no_emitted_origin():
    """A `SliceAssume` built by hand carries `origin == -1`, so it lands in a
    script's `emitted_origins`; a ledger entry never does. The two can
    therefore never meet, and a slice assembled outside the slicer cannot
    release a withheld violation."""
    ledger = (AssumeDisposition(kind=ASSUME_FORWARDED, reason="hand-built"),)
    assert ledger[0].forwarded_index == -1
    assert unaccounted_assumes(ledger, (-1,)) == ()  # an emitted -1 WOULD match
    # but no ledger entry the propagator writes carries -1 with kind forwarded
    p = _prop(_top_level_relational)
    assert all(
        e.forwarded_index >= 0
        for e in p.assume_ledger if e.kind == ASSUME_FORWARDED
    )


# ---------------------------------------------------------------------------
# the ledger is TOTAL over the assumes the propagator sees
# ---------------------------------------------------------------------------

def _top_level_relational():
    """One relational assume at top level: forwarded, and emitted for this
    obligation's slice. Its witness must satisfy it."""
    x = any_array((), "float64", (0.0, 10.0))
    y = any_array((), "float64", (0.0, 10.0))
    assume(x <= y)
    return (assert_(x + y <= 5.0),)


def _applied_only():
    """Every assume APPLIED: a point bound narrows the declared box."""
    x = any_array((), "float64", (-10.0, 10.0))
    assume(x >= 2.0)
    return (assert_(x >= 0.0),)


def _s6_mixed_drop_reasons():
    """Audit S6: one relational assume forwarded and emitted, one assume
    dropped for a DIFFERENT reason (`or` has no narrowing rule).

    Admitted domain `{x < y} ∩ {x > 5 ∨ y > 5}` = `{x < y, y > 5}`, on which
    `x + y > -5` everywhere — so any REFUTED here is false.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    assume(x < y)
    assume(jnp.logical_or(x > 5.0, y > 5.0))
    return (assert_(x + y >= -5.0),)


def _f1_mixed_branch_scoped():
    """The regression the branch-scoping repair opened: one ordinary
    relational assume (forwarded, emitted) plus one BRANCH-SCOPED relational
    assume, which is no longer forwarded AND no longer counted.

    `s ∈ [1, 10]`, so the `yes` branch is the one that runs and its
    precondition `v[0] >= v[1]` holds at every admitted point; together with
    the top-level `x <= y` that forces `x == y`, under which `y <= x + 1`.
    A witness with `x < y` is therefore not an admitted point.
    """
    x = any_array((), "float64", (0.0, 10.0))
    y = any_array((), "float64", (0.0, 10.0))
    s = any_array((), "float64", (1.0, 10.0))
    assume(x <= y)

    def yes(v):
        assume(v[0] >= v[1])
        return v[0] * 0.0

    def no(v):
        return v[1] * 0.0

    jax.lax.cond(s > 0.0, yes, no, jnp.stack([x, y]))
    return (assert_(y <= x + 1.0),)


def _only_branch_scoped():
    """The same query with the TOP-LEVEL assume deleted: the only assume is
    branch-scoped, so nothing is forwarded at all and the ledger's single
    entry is a drop."""
    x = any_array((), "float64", (0.0, 10.0))
    y = any_array((), "float64", (0.0, 10.0))
    s = any_array((), "float64", (1.0, 10.0))

    def yes(v):
        assume(v[0] >= v[1])
        return v[0] * 0.0

    def no(v):
        return v[1] * 0.0

    jax.lax.cond(s > 0.0, yes, no, jnp.stack([x, y]))
    return (assert_(y <= x + 1.0),)


def test_every_assume_applied_leaves_only_applied_entries():
    p = _prop(_applied_only)
    assert [e.kind for e in p.assume_ledger] == [ASSUME_APPLIED]
    assert unaccounted_assumes(p.assume_ledger, ()) == ()


def test_every_assume_emitted_leaves_only_forwarded_entries():
    p = _prop(_top_level_relational)
    assert [e.kind for e in p.assume_ledger] == [ASSUME_FORWARDED]
    assert len(p.relational_assumes) == 1
    assert unaccounted_assumes(p.assume_ledger, (0,)) == ()
    assert len(unaccounted_assumes(p.assume_ledger, ())) == 1


def test_a_differently_dropped_assume_is_in_the_ledger_and_not_in_the_tuple():
    """The gap S6 fell through, measured directly: `relational_assumes` does
    not know the `or` assume exists, and the ledger does."""
    p = _prop(_s6_mixed_drop_reasons)
    assert len(p.relational_assumes) == 1
    assert sorted(_kinds(_s6_mixed_drop_reasons)) == sorted(
        [ASSUME_FORWARDED, ASSUME_DROPPED]
    )
    (missing,) = unaccounted_assumes(p.assume_ledger, (0,))
    assert "'or'" in missing.reason


def test_a_branch_scoped_assume_is_in_the_ledger_and_not_in_the_tuple():
    """The same gap, opened by the branch-scoping repair rather than by a
    drop reason: nothing was forwarded, so nothing moved the count."""
    p = _prop(_f1_mixed_branch_scoped)
    assert len(p.relational_assumes) == 1, (
        "the branch-scoped assume must NOT be forwarded — re-forwarding it "
        "reopens the defect the branch closed"
    )
    kinds = [e.kind for e in p.assume_ledger]
    assert kinds.count(ASSUME_FORWARDED) == 1
    assert kinds.count(ASSUME_DROPPED) == 1
    (missing,) = unaccounted_assumes(p.assume_ledger, (0,))
    assert "branch-scoped" in missing.reason


def test_a_query_whose_only_assume_is_branch_scoped_forwards_nothing():
    p = _prop(_only_branch_scoped)
    assert p.relational_assumes == ()
    assert [e.kind for e in p.assume_ledger] == [ASSUME_DROPPED]
    assert p.assume_dropped is True
    assert len(unaccounted_assumes(p.assume_ledger, ())) == 1


def test_an_inert_mode_assume_is_still_recorded():
    """An empty ledger has to mean "no assume", never "an assume nobody wrote
    down" — the second reading is what makes a release rule default open."""
    p = _prop(_top_level_relational, assume_mode="inert")
    assert [e.kind for e in p.assume_ledger] == [ASSUME_DROPPED]
    assert "inert" in p.assume_ledger[0].reason


# ---------------------------------------------------------------------------
# end to end: the verdicts the rule decides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "harness",
    [_s6_mixed_drop_reasons, _f1_mixed_branch_scoped, _only_branch_scoped],
    ids=["s6_mixed_drops", "f1_branch_scoped", "only_branch_scoped"],
)
def test_an_unaccounted_assume_withholds_the_violation(harness):
    v = _run(harness)
    assert v.status == "UNKNOWN", (
        f"{harness.__name__} returned {v.status}: the solver ran over a "
        f"superset of the admitted region and its witness was released"
    )
    assert "WITHHELD from REFUTED" in v.render()


def test_the_withholding_NAMES_the_conjunct_that_caused_it():
    """A refusal that only restates the rule leaves the reader unable to tell
    WHICH of their assumes the solver never saw."""
    rendered = _run(_f1_mixed_branch_scoped).render()
    assert "WITHHELD from REFUTED" in rendered
    assert "unaccounted for on this obligation" in rendered
    assert "branch-scoped" in rendered


def test_a_fully_emitted_query_still_refutes():
    """The one-sidedness has a cost and the cost is bounded: when every
    assume IS accounted for, the violation is released as before. Without
    this the rule could pass every test above by never releasing anything."""
    v = _run(_top_level_relational)
    assert v.status == "REFUTED"


def _noop_only():
    """An assume whose WHOLE content excludes nothing: `x >= -1 or x >= -2`
    is definitely true at every point of `[0, 10]`.

    It takes the propagator's whole-drop path, which sets `assume_dropped`
    unconditionally, so the withholding gate fires — and the OLD rule then
    had `len(relational_assumes) == 0` and could never release. The judged
    set is the assumed region here, so withholding was a pure loss.
    """
    x = any_array((), "float64", (0.0, 10.0))
    assume(jnp.logical_or(x >= -1.0, x >= -2.0))
    return (assert_(x <= 5.0),)


def test_an_assume_that_excludes_nothing_no_longer_withholds_forever():
    """The one direction in which this rule releases MORE than the count it
    replaced, disclosed in SOUNDNESS.md rather than left to be found.

    Sound, and it is the rule the mixed-conjunction path already applied to
    the same class of conjunct: a conjunct definitely true over the boxes in
    force excluded nothing, so a witness in those boxes is admitted.
    """
    from fractions import Fraction as F
    p = _prop(_noop_only)
    assert p.assume_dropped is True, "the withholding gate must still fire"
    assert [e.kind for e in p.assume_ledger] == [ASSUME_NOOP]
    assert p.relational_assumes == ()   # denominator zero for the old rule
    v = _run(_noop_only)
    assert v.status == "REFUTED"
    for line in v.render().splitlines():
        if line.strip().startswith("x0 = "):
            w = F(line.strip().split(" = ")[1].split(" ")[0])
            assert 0 <= w <= 10, "the witness must be in the declared box"
            assert (w >= -1) or (w >= -2), "and must satisfy the assume"
            assert not (w <= 5), "and must falsify the assert"
            break
    else:
        raise AssertionError("no witness rendered")


def test_the_released_witness_satisfies_the_assume_it_was_released_under():
    """The release rule's own claim, checked rather than assumed."""
    from fractions import Fraction as F
    v = _run(_top_level_relational)
    assert v.status == "REFUTED"
    vals = {}
    for line in v.render().splitlines():
        line = line.strip()
        if line.startswith("x") and " = " in line:
            name, _, rest = line.partition(" = ")
            vals[name.strip()] = F(rest.split(" ")[0])
    assert vals["x0"] <= vals["x1"], "the released witness violates x <= y"
    assert vals["x0"] + vals["x1"] > 5, "the released witness is not violating"


def test_an_invented_disposition_withholds_a_query_that_otherwise_refutes():
    """THE END-TO-END FORM OF THE WHITELIST. Same query, same solver, same
    emitted axiom — one extra ledger entry naming a disposition this build
    has never heard of, and the violation must be withheld.

    A rule that counts cannot see this entry at all; a rule that blacklists
    releases on it. Only a rule that requires every entry to name an
    accounted-for disposition refuses.
    """
    closed = transcribe(jax.make_jaxpr(_top_level_relational)())
    p = P.propagate(closed)
    config = S.SolverConfig(timeout_ms=8000)

    baseline = S.escalate(closed, p, config)
    assert [r.outcome for r in baseline.records] == [S.OB_VIOLATED_WITNESS], (
        "the control must actually refute, or this test proves nothing"
    )

    future = dataclasses.replace(
        p,
        assume_ledger=p.assume_ledger + (
            AssumeDisposition(
                kind="a-drop-reason-added-by-a-later-batch",
                reason="a conjunct this build does not know how to classify",
                where="somewhere",
            ),
        ),
    )
    esc = S.escalate(closed, future, config)
    assert [r.outcome for r in esc.records] == [S.OB_UNKNOWN]
    assert "WITHHELD from REFUTED" in esc.records[0].detail
    assert "a-drop-reason-added-by-a-later-batch" in esc.records[0].detail


def test_the_release_rule_reads_no_count_at_all():
    """The specific failure both rounds shared: an equality between two
    populations. Adding a forwarded-but-unemitted assume to the ledger must
    withhold; adding one that IS emitted must not — and neither answer may
    depend on how many entries there are.
    """
    closed = transcribe(jax.make_jaxpr(_top_level_relational)())
    p = P.propagate(closed)
    config = S.SolverConfig(timeout_ms=8000)

    # padding the ledger with accounted entries changes nothing
    padded = dataclasses.replace(
        p,
        assume_ledger=p.assume_ledger + tuple(
            AssumeDisposition(kind=ASSUME_NOOP, reason=f"pad {i}")
            for i in range(5)
        ),
    )
    assert [r.outcome for r in S.escalate(closed, padded, config).records] == [
        S.OB_VIOLATED_WITNESS
    ]

    # one unemitted forwarded entry, whatever the totals, withholds
    unemitted = dataclasses.replace(
        p,
        assume_ledger=p.assume_ledger + (
            AssumeDisposition(
                kind=ASSUME_FORWARDED, reason="never emitted",
                forwarded_index=99,
            ),
        ),
    )
    assert [r.outcome for r in S.escalate(closed, unemitted, config).records] == [
        S.OB_UNKNOWN
    ]


# ---------------------------------------------------------------------------
# the join is on identity all the way down
# ---------------------------------------------------------------------------

def test_the_script_records_WHICH_assumes_it_stated():
    from stelling.obligation import slice_unknown_obligations
    from stelling.smt import emit

    closed = transcribe(jax.make_jaxpr(_top_level_relational)())
    p = P.propagate(closed)
    env = P.interval_env(closed)
    (sl,) = list(slice_unknown_obligations(closed, p, env))
    assert [sa.origin for sa in sl.assumes] == [0]
    script = emit(sl, "z3", 5000)
    assert script.emitted_origins == (0,)
    assert script.relational_assumes_emitted == len(script.emitted_origins)


def test_a_skipped_assume_is_absent_from_the_origins_not_merely_uncounted():
    """An assume whose operands are outside this obligation's backward cone
    is skipped with a reason; its index must not appear among the origins,
    or the join would account for an axiom that was never written."""
    from stelling.obligation import slice_unknown_obligations
    from stelling.smt import emit

    def h():
        x = any_array((), "float64", (0.0, 10.0))
        y = any_array((), "float64", (0.0, 10.0))
        z = any_array((), "float64", (0.0, 10.0))
        w = any_array((), "float64", (0.0, 10.0))
        assume(x <= y)
        assume(z <= w)          # nothing downstream reads z or w
        return (assert_(x + y <= 5.0),)

    closed = transcribe(jax.make_jaxpr(h)())
    p = P.propagate(closed)
    env = P.interval_env(closed)
    (sl,) = list(slice_unknown_obligations(closed, p, env))
    assert len(p.relational_assumes) == 2
    assert [sa.origin for sa in sl.assumes] == [0]
    assert len(sl.assumes_skipped) == 1
    assert emit(sl, "z3", 5000).emitted_origins == (0,)
    # and so the violation stays withheld: assume #1 never reached the solver
    assert _run(h).status == "UNKNOWN"
