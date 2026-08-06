# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The scatter VERIFIED bar: it fires, and it fires only where intended.

The bar closes the one direction a wrong SMT emission row could not
self-check. A spurious witness is caught by exact-rational replay; a MISSED
violation would mint a false VERIFIED with nothing downstream to catch it.
So solver-path VERIFIED is withheld on obligations whose EMITTED SLICE
carries a barred primitive, and the refutation path — which is
self-checking — stays open.

Three halves are pinned here, because there turned out to be three. A bar
nobody has seen fire is not a mechanism; a bar that fires on interval-only
verdicts would silently withhold verdicts it was never meant to touch; and a
bar scoped to the whole traced query fires on obligations the emission row
never touched, which is the same over-firing one level finer. That third
case is the reconstruction: this module's original fixture WAS such a case
(solver-decided slice ``['sub','ge']``, the scatter on a different,
interval-decided obligation) and its docstring anticipated needing rebuild.
It is kept, inverted, as the regression — see
``test_a_scatter_OFF_the_decided_slice_withholds_nothing``.

**AND A FOURTH, WHICH IS WHERE THE SCOPE COMES FROM.** Narrowing the bar to
one obligation means answering "which obligation, and what is on it", and the
answer must not be a claim the verdict's own inputs can make about
themselves. A predecessor recorded the per-obligation barred set on
``solvers.ObligationEscalation.barred_on_slice`` and the bar read it;
measured, that lost BOTH immunities the whole-query bar had — an empty tuple
is a positive claim nothing validated, and ``make_solver_verdict`` binds its
escalation to its ``closed`` nowhere, so a scatter-free escalation stamped
against a scatter-bearing query returned VERIFIED where the whole-query bar
returned UNKNOWN. The scope is now DERIVED from ``closed`` (re-slicing the
decided obligations), and both directions are pinned below.

**AND A FIFTH: DERIVING THE CONTENTS DID NOT BIND THE PAIRING.** The repair
above claimed it did, on the reasoning that a mispaired index would not slice
— and `test_a_scatter_free_escalation_cannot_clear_a_scatter_BEARING_query`
agreed, because its mispaired index reaches no real obligation of the wrong
query. Its NEIGHBOUR does: two scatter-bearing queries of the same shape, the
escalation of the one whose scatter is on the decided obligation stamped
against the one whose scatter is elsewhere. That index slices, finds no
barred primitive, and cleared the bar (measured VERIFIED on `caac1ee` and
`45cf526`, UNKNOWN on `8e42934`). An index is not evidence about a query.

**AND A SIXTH: THE SCRIPT HASH IS NOT EVIDENCE ABOUT A SLICE.** The fifth
repair re-emitted the re-derived slice and matched the `smt2_sha256` the
invocation records. Emission IS a pure function of (slice, flavour, timeout);
the converse is what the guard needed, and it is FALSE for the one primitive
under the bar. The static-index `scatter` SET row emits NO line, so an
untouched element aliases the operand's term and `s[1] - x[1] <= 0` emits byte
for byte what `x[1] - x[1] <= 0` emits — same sha, barred sets `('scatter',)`
and `()`. The fixture that was supposed to catch this **built away its own
trigger**: it said "the one difference is WHERE" while also introducing a
fresh input and a different predicate, and it was THAT which made the hashes
differ. Rebuilt to differ only in where the scatter sits, it returns VERIFIED
on `eb1ff86`. The narrowing now also requires `slice_sha256`, the fingerprint
of the slice's own primitives and nesting, which the emitted text cannot
carry — see ``test_the_script_hash_alone_cannot_separate_these_two_slices``,
``test_a_mispaired_query_that_still_SLICES_cannot_clear_the_bar`` in both
parametrisations, ``test_the_bar_scope_itself_widens_on_the_colliding_pair``,
and the other direction,
``test_the_correct_pairing_still_narrows_and_the_hash_is_why``.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp

import stelling.verdict as V
from stelling.harness import any_array, assert_, trace
from stelling.preconditions import check


@pytest.fixture(autouse=True)
def _x64():
    """Scope x64 to this module. Setting it at import leaks float64 into every
    later-run module in the same process — measured here the same way the
    suite already documents: it flipped test_transcribe's cross-process
    hash-stability test, which passes alone and fails in-suite."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _obl_solves(v) -> int:
    sol = v.stamp.solver if v.stamp else None
    sols = sol if isinstance(sol, tuple) else (sol,)
    return len([s for s in sols if s and s.invoked and "widen" not in s.reason])


def _scatter_ON_the_decided_slice():
    """A query whose SOLVER-DECIDED obligation carries `scatter` on its
    emitted slice — the shape the bar exists for.

    Obligation 0 is `s[1] - x[1] <= 0` where `s = x.at[0].set(0.5)`. It is
    exactly true (element 1 is untouched by the write), but intervals are
    correlation-blind and propagate `[-1, 1]`, so it escalates; the emitted
    slice is measured as `['broadcast_in_dim', 'scatter', 'slice', 'squeeze',
    'slice', 'squeeze', 'sub', 'le']` — the `scatter` really is on the slice
    the solver was asked about. Obligation 1 is settled by intervals. Both
    discharge, so the verdict would be VERIFIED but for the bar.

    THIS SCENARIO DEPENDS ON AN IMPRECISION, and says so deliberately: it
    needs the correlation between `s[1]` and `x[1]` to stay invisible to the
    abstraction so that escalation runs and the bar is reached. A refinement
    that recovers that correlation (affine arithmetic across the scatter row
    is the obvious one) will decide obligation 0 without a solver, and this
    test will fail. The repair is a DIFFERENT construction that still forces
    a solver-decided obligation WITH `scatter` on its slice — not deleting
    the assertion, which would leave the bar untested while looking green.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    return (assert_(s[1] - x[1] <= 0.0), assert_(s >= 0.0))


def _scatter_OFF_the_decided_slice():
    """A query that CONTAINS scatter whose solver-decided obligation does not
    touch it — the false bar this module's predecessor asserted.

    Obligation 0 is `y - y >= 0`: undecidable by intervals (correlation-blind
    again) and trivial for SMT, and its emitted slice is measured as
    `['sub', 'ge']` — no scatter anywhere on it. Obligation 1 holds the
    scatter and is settled by intervals, so no emission row was consulted
    about it either. Nothing in this verdict can be wrong because the scatter
    emission is wrong.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(y - y >= 0.0), assert_(s >= 0.0))


def test_the_bar_withholds_a_solver_path_verified_on_a_scatter_slice():
    # Deliberately NOT skipif-guarded on the bar being non-empty. A skip is
    # not a failure, so a guarded test cannot be mutation-proved: emptying
    # VERIFIED_BARRED_PRIMITIVES would silently skip this rather than fail it,
    # which is exactly the false negative the mutation norm exists to catch.
    # When the principal lifts the bar this test fails loudly and is updated
    # deliberately — the same one-identifiable-place discipline as the bar.
    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    v = check(_scatter_ON_the_decided_slice,
              vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert _obl_solves(v) > 0, (
        "intervals settled everything, so the bar never applies and this "
        "test does not exercise the firing direction"
    )
    assert all(o.status == "discharged" for o in v.obligations), (
        "the scenario must reach the bar with every obligation discharged — "
        "otherwise UNKNOWN would prove nothing about the bar"
    )
    assert v.status == "UNKNOWN"
    withheld = [n for n in v.notes if "VERIFIED withheld" in n]
    assert withheld
    assert all("assert #0" in n for n in withheld), (
        f"the note must NAME the obligation whose slice carries the barred "
        f"primitive rather than the query as a whole: {withheld}"
    )


def test_the_scatter_really_is_on_the_decided_slice():
    """ANTI-VACUITY for the test above (Norm C). If the fixture's scatter
    drifted OFF the escalated obligation's slice, the test above would stop
    measuring the bar's scope while still looking green under a fallback.
    Assert the slice itself."""
    from stelling.obligation import (
        DeclinedObligation,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    closed = trace(_scatter_ON_the_decided_slice)
    p = propagate(closed)
    env = interval_env(closed)
    sliced = [
        s for s in slice_unknown_obligations(closed, p, env)
        if not isinstance(s, DeclinedObligation)
    ]
    assert sliced, "no obligation was sliced; nothing reaches the solver"
    prims = {str(e.primitive) for s in sliced for e in s.eqns}
    assert prims & V.VERIFIED_BARRED_PRIMITIVES, (
        f"no barred primitive on any escalated slice ({sorted(prims)}) — the "
        f"bar test above is not measuring what it claims"
    )


def test_a_scatter_OFF_the_decided_slice_withholds_nothing():
    """THE DIRECTION THE SLICE-SCOPING CHANGED, and the reconstruction of
    this module's original fixture.

    The bar used to read `_barred_primitives(closed)` — the WHOLE traced
    query — so a scatter anywhere in the jaxpr withheld a VERIFIED resting
    entirely on obligations the scatter emission row never touched. Measured
    on exactly this query: solver-decided slice `['sub','ge']`, verdict
    UNKNOWN. The emission row cannot be wrong about an obligation it was not
    asked, so there was nothing to withhold.
    """
    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    closed = trace(_scatter_OFF_the_decided_slice)
    assert any(str(e.primitive) == "scatter" for e in closed.jaxpr.eqns), (
        "this test is vacuous unless the query really does contain scatter"
    )
    assert V._barred_primitives(closed), (
        "the WHOLE-QUERY barred set is empty on this fixture, so it cannot "
        "distinguish a slice-scoped bar from a whole-query one"
    )
    v = check(_scatter_OFF_the_decided_slice,
              vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert _obl_solves(v) > 0, (
        "nothing was solver-decided, so the bar was never consulted and this "
        "test does not exercise the scoping"
    )
    assert v.status == "VERIFIED", (
        f"{v.status}: a VERIFIED resting on a scatter-free slice was "
        f"withheld — the bar is scoped to the query again, not to the slice"
    )
    assert not any("VERIFIED withheld" in n for n in v.notes)


# ── the scope's PROVENANCE: its CONTENTS derived, its DOMAIN a precondition ──
#
# Everything below stamps a verdict through `make_solver_verdict` directly
# rather than through `check`, because that is the surface the defects were
# on: it is public, it takes `closed`, `propagation` and `escalation` as three
# independent arguments, and it gates their pairing on semantics, ieee,
# constrained-assume and ledger provenance — but on nothing that ties the
# escalation to the query.
#
# WHAT IS PINNED HERE IS NOT IMMUNITY TO A FORGED ESCALATION, and the section
# heading used to read as if it were ("never read off a record", full stop).
# No barred PRIMITIVE is read off a record — that is real, and it is what the
# deleted `barred_on_slice` field cost. But WHICH obligations the bar is asked
# about does come from the records, so the bar inherits
# `make_solver_verdict`'s documented precondition that its escalation came
# from `escalate()` on this query. Hardening against a violation of it would
# defend nothing: `stelling.verdict.Verdict` is public and a plain frozen
# dataclass, so a caller who can hand-build an `ObligationEscalation` can
# hand-build the VERIFIED and never reach this function. These tests measure
# what a MISPAIRED honest assembly does, and that the domain cannot disagree
# with the discharge about one record — not what an adversary cannot do.

VERSIONS = dict(
    stelling_version="0.0.0", jax_version="0.0.0", precision_config="x64"
)


def _scatter_free():
    """No scatter anywhere, one solver-decided obligation, VERIFIED."""
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(y - y >= 0.0),)


def _two_solver_decided_obligations():
    """BOTH obligations escalate and BOTH discharge, and only #0's slice
    carries scatter. Measured slices: #0 is
    ``['broadcast_in_dim','le','scatter','slice','slice','squeeze','squeeze',
    'sub']`` and #1 is ``['add','le','sub']``.

    #1 is `y + 1.0 - y <= 1.0`: exactly 1.0 in ℝ, but the correlation is
    invisible to intervals ([2,3] − [1,2] = [0,2]), so it escalates instead of
    being settled — which is what makes it a SECOND deciding obligation and
    not just a second obligation."""
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(s[1] - x[1] <= 0.0), assert_(y + 1.0 - y <= 1.0))


def _stamped(build):
    """(closed, propagation, escalation) for a build, un-assembled."""
    from stelling.propagate import propagate
    from stelling.solvers import SolverConfig, escalate

    closed = trace(build)
    prop = propagate(closed)
    return closed, prop, escalate(closed, prop,
                                  SolverConfig(timeout_ms=20000))


def _verdict_status(closed, prop, esc):
    from stelling.solvers import make_solver_verdict

    return make_solver_verdict(closed, prop, esc, **VERSIONS).status


def test_a_scatter_free_escalation_cannot_clear_a_scatter_BEARING_query():
    """THE MISPAIRING THE WHOLE-QUERY BAR WAS IMMUNE TO, and the reason the
    scope is derived rather than recorded.

    Nothing in `make_solver_verdict` binds `escalation` to `closed`. Stamp a
    genuine scatter-free escalation against a scatter-bearing query and the
    bar must still fire: the whole-query bar it replaced read `closed`
    directly and returned UNKNOWN here (measured on 8e42934), and a
    slice-scoped bar that reads its scope off the escalation returned
    VERIFIED. Narrowing the scope must not cost this.

    THIS FIXTURE IS THE MISPAIRING THAT FAILS SAFE FOR THE WEAKER REASON, and
    for a while it was the only one, which is how a live hole stayed green.
    Its `clean` query has ONE obligation, so the decided index 0 is the only
    index there is and the `dirty` query's obligation #0 carries the scatter
    itself — the bar fires whatever mechanism is asked. The arrangement that
    separates "the index exists here" from "the escalation is about here" is
    `test_a_mispaired_query_that_still_SLICES_cannot_clear_the_bar`, and it is
    where the mechanism this test cannot see is measured.
    """
    from stelling.solvers import make_solver_verdict

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    dirty, _, _ = _stamped(_scatter_ON_the_decided_slice)
    clean, prop, esc = _stamped(_scatter_free)
    assert V._barred_primitives(dirty) and not V._barred_primitives(clean), (
        "the two queries do not differ in the whole-query barred set, so this "
        "test cannot distinguish a query-derived scope from a recorded one"
    )
    assert make_solver_verdict(clean, prop, esc, **VERSIONS).status == (
        "VERIFIED"
    ), "the correctly-paired assembly does not VERIFY; the fixture is wrong"

    v = make_solver_verdict(dirty, prop, esc, **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: an escalation carrying no scatter cleared the bar on a "
        f"query that does — the bar's scope is being read off the escalation "
        f"instead of derived from the query it is stamped against"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


def _scatter_ELSEWHERE_same_shape():
    """THE NEAREST NEIGHBOUR of `_scatter_ON_the_decided_slice`, and the
    difference is WHERE THE SCATTER SITS AND NOTHING ELSE.

    Same declared inputs, same two obligations, same predicate on #0, same
    `s >= 0` on #1. The single edit is that #0 reads `x[1]` where
    `_scatter_ON_the_decided_slice` reads `s[1]` — so the scatter is on #1
    alone, and #0's emitted slice carries none.

    THE PREDECESSOR OF THIS FIXTURE BUILT AWAY ITS OWN TRIGGER. It said "the
    one difference is WHERE" while ALSO introducing a fresh scalar input `y`
    and a different predicate (`y - y >= 0`), and it was that difference — not
    the scatter's location — that made the two queries emit different scripts.
    So the test below passed on `eb1ff86`, whose narrowing keyed on the script
    hash alone, while the defect it was written for was live: with the fixture
    differing only in WHERE, `eb1ff86` returns VERIFIED. See
    `test_the_script_hash_alone_cannot_separate_these_two_slices`, which
    measures the byte-equality that makes this the hard case.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    return (assert_(x[1] - x[1] <= 0.0), assert_(s >= 0.0))


def _scatter_ELSEWHERE_different_predicate():
    """The WEAKER neighbour, kept because it separates the two keys.

    Same arrangement as `_scatter_ELSEWHERE_same_shape` — scatter on #1, #0
    solver-decided and scatter-free — but #0 is a different claim over a fresh
    input, so its emitted slice (`['ge','sub']`) emits a DIFFERENT script.
    That difference is enough for the script hash alone, which is why this
    shape was green on `eb1ff86` while its sibling above was not.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(y - y >= 0.0), assert_(s >= 0.0))


def test_the_script_hash_alone_cannot_separate_these_two_slices():
    """THE COLLISION, AS A MEASUREMENT — the premise the bar's key rests on.

    `_evidence_is_about` re-emits the re-derived slice and compares hashes.
    That can only witness WHICH SLICE was emitted from if emission is
    injective, and it is not: the static-index `scatter` SET row appends no
    line, so an element the write did not touch aliases the operand's term.
    Here the scatter-BEARING slice of `_scatter_ON_the_decided_slice`#0 and
    the scatter-FREE slice of `_scatter_ELSEWHERE_same_shape`#0 emit the same
    bytes, and their barred sets differ.

    If this test ever goes red because the two scripts stopped colliding, the
    bar has not been repaired — the fixture has drifted back to the shape that
    could not fail, and `test_a_mispaired_query_that_still_SLICES...` is again
    passing for the weaker reason.
    """
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    from stelling.smt import emit, slice_fingerprint

    on_closed = trace(_scatter_ON_the_decided_slice)
    el_closed = trace(_scatter_ELSEWHERE_same_shape)
    on_sl = slice_obligation(on_closed, 0, interval_env(on_closed))
    el_sl = slice_obligation(el_closed, 0, interval_env(el_closed))
    assert not isinstance(on_sl, DeclinedObligation)
    assert not isinstance(el_sl, DeclinedObligation)

    assert V._barred_in_eqns(on_sl.eqns) == ("scatter",)
    assert V._barred_in_eqns(el_sl.eqns) == ()
    assert emit(on_sl, "z3", 20000).sha256 == emit(el_sl, "z3", 20000).sha256, (
        "the two slices no longer emit byte-identical scripts, so the script "
        "hash WOULD separate them and the mispairing test below is measuring "
        "the weaker key"
    )
    assert slice_fingerprint(on_sl) != slice_fingerprint(el_sl), (
        "the slice fingerprint does not separate a scatter-bearing slice from "
        "a scatter-free one, so the bar's key distinguishes nothing that the "
        "script hash did not already"
    )


@pytest.mark.parametrize("elsewhere,key", [
    (_scatter_ELSEWHERE_same_shape, "the slice fingerprint"),
    (_scatter_ELSEWHERE_different_predicate, "the script hash"),
])
def test_a_mispaired_query_that_still_SLICES_cannot_clear_the_bar(
    elsewhere, key
):
    """THE MISPAIRING THE INDEX-ONLY DOMAIN DID NOT CATCH — and, in its first
    parametrisation, the one the SCRIPT-HASH-ONLY narrowing did not catch
    either.

    `test_a_scatter_free_escalation_cannot_clear_a_scatter_BEARING_query`
    passes on the arrangement where the mispaired index does not reach a real
    obligation of the wrong query. These are its neighbours: BOTH queries
    carry `scatter`, both have two obligations, and the decided index 0 exists
    in both. So the re-slice does not decline — it succeeds, on the wrong
    query, and finds no barred primitive.

    THE TWO PARAMETRISATIONS NEED DIFFERENT KEYS, which is why both are here.

        elsewhere = same shape (scatter's LOCATION is the only difference)
            8e42934 UNKNOWN | caac1ee VERIFIED | 45cf526 VERIFIED
            eb1ff86 VERIFIED | here UNKNOWN
            separated only by `slice_sha256`; the two scripts are byte-equal

        elsewhere = different predicate (a fresh input and another claim)
            8e42934 UNKNOWN | caac1ee VERIFIED | 45cf526 VERIFIED
            eb1ff86 UNKNOWN | here UNKNOWN
            separated by `smt2_sha256` already

    The repair is `verdict._evidence_is_about`: an obligation narrows the bar
    only when a recorded invocation reproduces BOTH the script hash and the
    slice fingerprint from the slice re-derived out of the query being
    stamped.
    """
    from stelling.solvers import make_solver_verdict

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    on_closed, on_prop, on_esc = _stamped(_scatter_ON_the_decided_slice)
    el_closed, _, _ = _stamped(elsewhere)

    # the pair must be indistinguishable to every WEAKER test than the one
    # this pins, or it does not measure the mechanism it claims to
    assert V._barred_primitives(on_closed) == V._barred_primitives(el_closed), (
        "the two queries differ in their whole-query barred set, so a "
        "whole-query bar would already separate them and this test would not "
        "be measuring the slice-scoped one"
    )
    decided = {r.index: r.invocations
               for r in on_esc.records if r.outcome == "discharged"}
    assert set(decided) == {0}, decided
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    again = slice_obligation(el_closed, 0, interval_env(el_closed))
    assert not isinstance(again, DeclinedObligation), (
        "the mispaired index DECLINES to slice out of the wrong query, so "
        "this test is the case the fallback already caught and not the one "
        "it missed"
    )
    assert not (V._barred_in_eqns(again.eqns)), (
        "the mispaired query's obligation #0 carries a barred primitive on "
        "its own slice, so the bar would fire for a reason having nothing to "
        "do with the mispairing"
    )

    assert make_solver_verdict(on_closed, on_prop, on_esc, **VERSIONS).status == (
        "UNKNOWN"
    ), "the correctly-paired assembly does not bar; the fixture is wrong"

    v = make_solver_verdict(el_closed, on_prop, on_esc, **VERSIONS)
    assert [o.status for o in v.obligations] == ["discharged"] * len(
        v.obligations
    ), (
        "the mispaired assembly did not reach a would-be VERIFIED, so there "
        "is nothing for the bar to withhold and this test proves nothing"
    )
    assert v.status == "UNKNOWN", (
        f"{v.status}: an escalation produced on a DIFFERENT scatter-bearing "
        f"query cleared the bar. This arrangement is separated by {key}, so "
        f"that is the part of `_evidence_is_about` that has stopped working"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


def test_the_bar_scope_itself_widens_on_the_colliding_pair():
    """The same defect one layer down, at `_bar_scope` rather than at the
    assembled verdict — so that deleting the `_evidence_is_about` call is not
    a one-test edit.

    Not a duplicate of the parametrised test above: that one can only see the
    bar through a VERIFIED that may be withheld for some other reason, and
    this one reads the scope directly. Both must move together, because both
    are downstream of the same call.
    """
    on_closed, _on_prop, on_esc = _stamped(_scatter_ON_the_decided_slice)
    el_closed = trace(_scatter_ELSEWHERE_same_shape)
    decided = {r.index: r.invocations
               for r in on_esc.records if r.outcome == "discharged"}
    assert set(decided) == {0}, decided

    honest, honest_why = V._bar_scope(on_closed, decided)
    assert honest == ("scatter",) and "assert #0" in honest_why, (
        f"the honestly-paired scope is {honest!r}; the fixture is wrong"
    )
    barred, why = V._bar_scope(el_closed, decided)
    assert barred == ("scatter",), (
        f"_bar_scope narrowed to {barred!r} on a query the escalation is not "
        f"about, whose obligation #0 emits the SAME script as the one the "
        f"solver answered. Narrowing is keyed on the emitted text again"
    )
    assert "fell back to the whole query" in why, why


def test_the_correct_pairing_still_narrows_and_the_hash_is_why():
    """THE OTHER DIRECTION of the test above, and the case that distinguishes
    the repair from a silent revert to the whole-query bar.

    Withholding everything would pass the mispairing test trivially. What must
    also hold is that an HONEST assembly still narrows — and that the things
    permitting it are the measured ones: BOTH the script the solver was
    actually sent and the fingerprint of the slice it came out of, re-derived
    from this query, reproduce what the invocation recorded.
    """
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    from stelling.smt import emit, slice_fingerprint

    closed, _prop, esc = _stamped(_scatter_OFF_the_decided_slice)
    (record,) = [r for r in esc.records if r.outcome == "discharged"]
    sl = slice_obligation(closed, record.index, interval_env(closed))
    assert not isinstance(sl, DeclinedObligation)
    assert record.invocations, "no invocation to check the pairing against"
    matched, fingerprinted = [], []
    for stamp in record.invocations:
        opts = dict(stamp.options or ())
        timeout = opts.get(":timeout") or opts.get(":tlimit")
        assert timeout, (
            f"the {stamp.name} stamp records no timeout under a key "
            f"`_evidence_is_about` reads, so the honest path cannot bind and "
            f"the bar would widen on every correctly-paired verdict"
        )
        matched.append(emit(sl, stamp.name, int(timeout)).sha256
                       == opts.get("smt2_sha256"))
        fingerprinted.append(slice_fingerprint(sl) == opts.get("slice_sha256"))
    assert all(matched), (
        f"re-emitting this query's own slice does not reproduce the hash the "
        f"invocation recorded ({matched}) — emission is no longer a function "
        f"of (slice, solver, timeout), so the pairing check can never say yes "
        f"and the bar has silently become the whole-query one again"
    )
    assert all(fingerprinted), (
        f"the stamp does not carry this slice's fingerprint ({fingerprinted}) "
        f"— the second conjunct of `_evidence_is_about` can never say yes, so "
        f"every correctly-paired verdict on a scatter-bearing query now gets "
        f"the whole-query bar"
    )
    assert V._evidence_is_about(sl, record.invocations)
    assert not V._evidence_is_about(sl, ()), (
        "an obligation with NO recorded invocation binds, so the check is "
        "vacuous"
    )


def test_a_mispaired_PROPAGATION_cannot_empty_the_scope_either():
    """The other mispairing, and the one the recording design was built
    against: a propagation whose obligations are already `discharged` slices
    to NOTHING (the slicer only slices `unknown` ones), so a scope recomputed
    from `(closed, propagation)` would come back empty while every existing
    gate passed.

    The derivation does not take the propagation. It re-slices by INDEX out of
    `closed`, so there is no propagation to mispair — asserted here rather
    than argued, because it was the stated reason for recording.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    settled = dataclasses.replace(prop, obligations=tuple(
        dataclasses.replace(o, status="discharged") for o in prop.obligations
    ))
    v = make_solver_verdict(closed, settled, esc, **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: a propagation with nothing left to slice emptied the "
        f"bar's scope — the scope is being recomputed from the propagation "
        f"rather than from the query"
    )


def test_the_fallback_is_the_WHOLE_QUERY_SET_and_never_silence():
    """THE FAIL-CLOSED FALLBACK ITSELF, which nothing measured.

    Every "derive, don't record" argument in this module ends at the same
    sentence: when the derivation cannot be completed the bar drops to the
    whole-query set, which is WIDER, rather than to an empty scope, which is
    silence. Measured before this test existed: replacing both `return
    fallback` statements in `_bar_scope` with `return (), ""` left the full
    suite at 2008 passed, 2 skipped — the fallback the design rests on was
    load-bearing for nothing.

    Both directions are asserted, because the two are different code paths and
    a test that only calls the function cannot see the verdict move:

    * the derivation itself returns the whole-query set for an index that does
      not slice;
    * and a verdict assembled with such a record goes UNKNOWN rather than
      VERIFIED — a stray `discharged` record is the cheapest way for an
      honest-but-mispaired assembly to reach this path.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    closed, prop, esc = _stamped(_scatter_OFF_the_decided_slice)
    whole = V._barred_primitives(closed)
    assert whole, "the fixture carries nothing; the fallback is unreachable"

    barred, why = V._bar_scope(closed, {99: ()})
    assert barred == whole, (
        f"an index that does not slice returned {barred!r} where the "
        f"whole-query set is {whole!r} — the derivation fell to SILENCE, not "
        f"to the wider bar, and every 'fails closed' sentence about this "
        f"function is false"
    )
    assert "fell back to the whole query" in why

    # ... and it moves a verdict, which the call above cannot show
    assert make_solver_verdict(closed, prop, esc, **VERSIONS).status == (
        "VERIFIED"
    ), "the honest assembly does not VERIFY, so nothing here can be withheld"
    (real,) = [r for r in esc.records if r.outcome == "discharged"]
    strayed = dataclasses.replace(esc, records=esc.records + (
        dataclasses.replace(real, index=99),
    ))
    v = make_solver_verdict(closed, prop, strayed, **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: a discharged record naming an obligation this query "
        f"does not have left the bar's scope EMPTY instead of falling back "
        f"to the whole query"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


def test_what_a_stray_index_ACTUALLY_DOES_all_four_of_them():
    """The stray-index enumeration, MEASURED rather than asserted.

    `_bar_scope`'s docstring and `make_solver_verdict`'s block comment both
    enumerate what a stray index does. The enumeration has been wrong twice —
    first "a stray index does not slice" (one behaviour, and the wrong one),
    then a list of three presented as the whole space. There is a fourth: an
    index past the START of the list raises `IndexError` out of
    `slice_obligation` rather than declining, and reaches the whole-query set
    through `_bar_scope`'s outer `except` instead of through a `fallback`
    call. No soundness difference — it is named because "three behaviours"
    was being read as closed and was not, and this test is what makes the
    enumeration a measurement.

    NOT PRESENTED AS EXHAUSTIVE EITHER. What is asserted is that each listed
    behaviour is the one claimed, and that every one of them ends at the
    whole-query set.
    """
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env

    closed = trace(_scatter_ON_the_decided_slice)
    env = interval_env(closed)
    whole = V._barred_primitives(closed)
    assert whole, "the fixture carries nothing barred; this measures nothing"

    # 1. an index intervals decided slices perfectly well
    sl = slice_obligation(closed, 1, env)
    assert not isinstance(sl, DeclinedObligation)
    assert V._barred_in_eqns(sl.eqns) == ("scatter",)
    # 2. a negative index within range is Python indexing
    assert not isinstance(slice_obligation(closed, -1, env), DeclinedObligation)
    # 3. an index matching no assert equation DECLINES
    assert isinstance(slice_obligation(closed, 99, env), DeclinedObligation)
    # 4. and one past the start RAISES rather than declining
    with pytest.raises(IndexError):
        slice_obligation(closed, -3, env)

    for index in (1, -1, 99, -3):
        barred, why = V._bar_scope(closed, {index: ()})
        assert barred == whole, (
            f"index {index} narrowed the bar to {barred!r} instead of "
            f"falling back to the whole-query set {whole!r}"
        )
        assert "fell back to the whole query" in why


def test_the_fallback_also_holds_when_the_derivation_RAISES(monkeypatch):
    """The other `return fallback`, and it is a different statement.

    `_bar_scope` wraps the whole re-derivation in `except Exception` with the
    comment "a bar must never break a verdict, and it must never go quiet
    either". The first half is pinned by construction — an exception cannot
    escape. The second half was pinned by nothing: silencing THIS branch alone
    also left the suite fully green.

    Driven by making the slicer raise, which is the only way to reach a branch
    whose whole purpose is a failure no fixture produces.
    """
    import stelling.obligation as _ob

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    closed, prop, esc = _stamped(_scatter_OFF_the_decided_slice)
    whole = V._barred_primitives(closed)
    assert whole, "the fixture carries nothing; the fallback is unreachable"
    assert _verdict_status(closed, prop, esc) == "VERIFIED", (
        "the honest assembly does not VERIFY, so nothing here can be withheld"
    )

    def boom(*_a, **_k):
        raise RuntimeError("slicer exploded")

    monkeypatch.setattr(_ob, "slice_obligation", boom)
    barred, why = V._bar_scope(closed, {0: ()})
    assert barred == whole, (
        f"a raising re-derivation returned {barred!r} where the whole-query "
        f"set is {whole!r} — the except branch goes QUIET, so a bar that "
        f"cannot compute its scope silently stops being a bar"
    )
    assert "could not be re-derived" in why
    assert _verdict_status(closed, prop, esc) == "UNKNOWN", (
        "a verdict assembled while the slicer raises came back VERIFIED — "
        "the except branch is not withholding"
    )


# The values a record field of each declared type can be moved to. Keyed on
# the annotation text rather than on the field name, so a field added tomorrow
# is probed by this test without anybody remembering to list it — and an
# annotation not in this map is a LOUD failure below, never a silent skip.
#
# WHAT TWO VALUES BUY DEPENDS ENTIRELY ON THE TYPE, and an earlier version of
# this comment said they were "chosen so that a field whose NON-default value
# would clear the bar is probed with it" as though that were true of all of
# them. For `bool` it is: the two probes EXHAUST the type, so a conjunct on a
# new bool field is caught whichever way it points — that is how the
# `audited_clean` mutant this test was rewritten for is caught. For `str`,
# `int` and the tuples they are a SAMPLE of an unbounded space, and a sample
# cannot pin a channel. Measured on `eb1ff86`: `audit_token: str = ""` plus
# `and r.audit_token != "clean"` in the bar's domain is UNKNOWN at both probe
# values and VERIFIED at `'clean'`, full suite green.
#
# So the channel is pinned by `test_the_bars_domain_cannot_read_a_new_field`,
# which hands the domain a record that HAS no other field; this sweep stays
# for what it does do — it exercises the whole assembly at each probe value,
# which the construction test cannot, and it is the loud failure for a field
# of a type nobody has thought about.
def _field_probes():
    from stelling.verdict import solver_absent

    return {
        "int": (0, 7),
        "bool": (False, True),
        "str": ("", "a value no honest record carries"),
        "tuple[str, ...]": ((), ("a value no honest record carries",)),
        "tuple[SolverStamp, ...]": ((), (solver_absent("probe"),)),
        # a NON-NONE witness is what this probes; the bar reads no field of
        # one, and if that ever changes the sentinel raises here rather than
        # passing quietly
        "Witness | None": (None, object()),
    }


def test_no_record_field_can_narrow_the_bars_domain():
    """A RECORD MUST NOT BE ABLE TO CERTIFY ITS OWN CLEANLINESS — pinned as a
    CHANNEL, not as a list of today's fields.

    The predecessor's `barred_on_slice=()` was a positive claim ("nothing
    barred on my slice") that nothing validated: measured on this fixture, a
    record carrying it earned VERIFIED where the genuine scope was
    `('scatter',)`. The repair deletes the field rather than validating it, so
    what is pinned is the ABSENCE of any such channel.

    THE VERSION OF THIS TEST THAT PINNED A MEMBER LIST COULD NOT FAIL FOR ITS
    OWN DEFECT, and that is why it was rewritten. It reset every
    non-load-bearing field **to its declared default** and accounted for the
    field names — so a new field whose default is inert and whose OTHER value
    clears the bar was invisible to both halves. Measured: adding
    `audited_clean: bool = False` to `ObligationEscalation` and `and not
    r.audited_clean` to the bar's domain, then handing in records with
    `audited_clean=True`, produced VERIFIED where the honest assembly gives
    UNKNOWN — with the full suite green, the field-accounting assertion
    included, because `dataclasses.replace(r, audited_clean=False)` is exactly
    what the test did.

    So each field is now moved AWAY from what it holds, with the values taken
    from its declared TYPE. A field of a type `_field_probes` does not know
    fails this test loudly, which is the only way an enumeration can stay
    honest about the thing it has not thought of.

    THAT REWRITE MOVED THE ENUMERATION FROM NAMES TO TYPES; IT DID NOT PIN THE
    CHANNEL, and the difference is measurable. Two probe values exhaust `bool`
    and merely sample `str`: `audit_token: str = ""` on the record plus
    `and r.audit_token != "clean"` in the bar's domain is UNKNOWN at both
    probes and VERIFIED at `'clean'`, with the full suite green. This test
    still earns its place — it drives the WHOLE assembly at each value, which
    a construction test cannot — but the channel itself is pinned next door,
    by `test_the_bars_domain_cannot_read_a_new_field`, which removes the
    fields rather than guessing their values.

    LOAD-BEARING IS EXACTLY `index` AND `outcome`. `invocations` is read by
    `_evidence_is_about`, but only to PERMIT narrowing — every value of it
    other than a matching script hash widens the bar — so it is probed here
    like any other field and the bar must not move. An earlier version
    exempted it, which was false in the direction that matters; see
    `test_stripping_invocations_cannot_clear_the_bar` for that drift.
    """
    import dataclasses

    from stelling.solvers import ObligationEscalation, make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    assert make_solver_verdict(closed, prop, esc, **VERSIONS).status == (
        "UNKNOWN"
    ), "the genuine assembly does not bar; the fixture is wrong"

    assert "barred_on_slice" not in {
        f.name for f in dataclasses.fields(ObligationEscalation)
    }, (
        "a record field named `barred_on_slice` is back. It was deleted "
        "because a recorded scope is a claim the bar cannot check; if it "
        "returns, the bar must not read it"
    )

    probes = _field_probes()
    load_bearing = {"index", "outcome"}
    fields = [f for f in dataclasses.fields(ObligationEscalation)
              if f.name not in load_bearing]
    assert fields, "every field is load-bearing; this test measures nothing"
    unknown = [f.name for f in fields if str(f.type) not in probes]
    assert not unknown, (
        f"record field(s) {unknown} have a type this test does not know how "
        f"to move ({[str(f.type) for f in fields if f.name in unknown]}). It "
        f"CANNOT be skipped: the defect this test exists for is a field whose "
        f"non-default value clears the bar, and an unprobed field is exactly "
        f"that field. Add the type to `_field_probes` with a value no honest "
        f"record carries"
    )

    def stamp(**overrides):
        forged = dataclasses.replace(esc, records=tuple(
            dataclasses.replace(r, **overrides) for r in esc.records
        ))
        return make_solver_verdict(closed, prop, forged, **VERSIONS)

    # one field at a time, at every probe value ...
    for f in fields:
        for value in probes[str(f.type)]:
            v = stamp(**{f.name: value})
            assert [o.status for o in v.obligations] == (
                ["discharged"] * len(v.obligations)
            ), (
                f"setting {f.name}={value!r} un-discharged an obligation, so "
                f"there is no VERIFIED left for the bar to withhold and this "
                f"probe passes for the wrong reason"
            )
            assert v.status == "UNKNOWN", (
                f"{v.status}: a record carrying {f.name}={value!r} cleared "
                f"the bar. The bar's domain is `outcome == OB_DISCHARGED` and "
                f"nothing else may narrow it — a field that does is the "
                f"deleted `barred_on_slice` under another name"
            )
    # ... and all of them at once, in both directions
    for which in (0, 1):
        v = stamp(**{f.name: probes[str(f.type)][which] for f in fields})
        assert v.status == "UNKNOWN", (
            f"{v.status}: moving every non-load-bearing field at once "
            f"cleared the bar"
        )


class _OnlyThreeFields:
    """A record with `index`, `outcome`, `invocations` AND NOTHING ELSE.

    `__slots__` is the whole mechanism: reading any other attribute raises
    `AttributeError`, so a conjunct on a field of ANY type — `str`, `int`, a
    tuple, something not thought of — cannot silently evaluate. This is what
    an enumeration of probe VALUES cannot do, because `str` has more values
    than a test can list."""

    __slots__ = ("index", "outcome", "invocations")

    def __init__(self, index, outcome, invocations=()):
        self.index = index
        self.outcome = outcome
        self.invocations = invocations


class _JustRecords:
    __slots__ = ("records",)

    def __init__(self, records):
        self.records = records


def test_the_bars_domain_cannot_read_a_new_field():
    """PIN THE CHANNEL BY REMOVING IT, not by guessing what would flow down
    it. The bar's domain is `outcome == OB_DISCHARGED`, keyed by `index`, and
    carries `invocations` across for the widening check — three fields, and
    the record it is handed here HAS no fourth.

    THE DEFECT THIS IS FOR IS MEASURED, not imagined. On `eb1ff86`, adding
    `audit_token: str = ""` to `ObligationEscalation` and
    `and r.audit_token != "clean"` to the domain filter produced VERIFIED for
    a record carrying `audit_token='clean'` on the bar's own fixture, with the
    full suite green — including
    `test_no_record_field_can_narrow_the_bars_domain`, whose two `str` probes
    (`''` and `'a value no honest record carries'`) both left the bar firing.
    Two values exhaust `bool`; they sample `str`.

    Under this test the same conjunct raises `AttributeError` inside
    `_bar_domain`, which widens to the sentinel rather than returning a
    domain, and the equality below fails.
    """
    from stelling.solvers import OB_DISCHARGED, _bar_domain

    stamps = (V.solver_absent("probe"),)
    records = (
        _OnlyThreeFields(0, OB_DISCHARGED, stamps),
        _OnlyThreeFields(1, "unknown", ()),
        _OnlyThreeFields(2, OB_DISCHARGED, ()),
    )
    domain = _bar_domain(_JustRecords(records))
    assert domain == {0: stamps, 2: ()}, (
        f"the bar's domain came back {domain!r} from a record carrying "
        f"nothing but `index`, `outcome` and `invocations`. Either the "
        f"domain now reads a fourth field — which is the deleted "
        f"`barred_on_slice` under another name, whatever its type — or its "
        f"membership test is no longer `outcome == OB_DISCHARGED`"
    )
    # and it is not vacuous: the outcome really is what selects
    assert _bar_domain(_JustRecords((_OnlyThreeFields(0, "unknown", stamps),))) == {}


def test_the_bar_is_consulted_with_exactly_that_domain(monkeypatch):
    """The surface half: `make_solver_verdict` must hand `_bar_scope` what
    `_bar_domain` returns and nothing else.

    Without this, the test above pins an invariant at its producer and never
    at the place it matters — a conjunct written at the CALL SITE rather than
    inside `_bar_domain` would leave that test green. Checked on the honest
    assembly and on every `_field_probes` assignment, so a call-site conjunct
    keyed on any probed value moves the captured domain away from the
    computed one.
    """
    import dataclasses

    from stelling.solvers import _bar_domain, make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    seen: list = []
    real = V._bar_scope

    def spy(c, decided):
        seen.append(decided)
        return real(c, decided)

    monkeypatch.setattr(V, "_bar_scope", spy)

    probes = _field_probes()
    load_bearing = {"index", "outcome"}
    from stelling.solvers import ObligationEscalation
    fields = [f for f in dataclasses.fields(ObligationEscalation)
              if f.name not in load_bearing]
    cases = [esc] + [
        dataclasses.replace(esc, records=tuple(
            dataclasses.replace(r, **{f.name: probes[str(f.type)][which]})
            for r in esc.records
        ))
        for f in fields for which in (0, 1)
    ]
    for case in cases:
        seen.clear()
        make_solver_verdict(closed, prop, case, **VERSIONS)
        assert len(seen) == 1, (
            f"the bar was consulted {len(seen)} time(s); a domain that came "
            f"back empty means some conjunct removed every discharged record "
            f"from it between `_bar_domain` and the call"
        )
        assert seen[0] == _bar_domain(case), (
            f"`make_solver_verdict` handed the bar {seen[0]!r} where "
            f"`_bar_domain` returns {_bar_domain(case)!r} — the domain is "
            f"being filtered at the call site, outside the one place the "
            f"channel is pinned"
        )


@pytest.mark.parametrize("build,strip,label", [
    (_scatter_ON_the_decided_slice, 0, "the scatter obligation, alone"),
    (_two_solver_decided_obligations, 0, "the scatter obligation, of two"),
])
def test_stripping_invocations_cannot_clear_the_bar(build, strip, label):
    """ONE CONCEPT, ONE PREDICATE — the regression for the drift that made
    `discharging` and `decided` two different tests over one record.

    `make_solver_verdict` discharged an obligation on
    `record.outcome == OB_DISCHARGED` and took the bar's domain from
    `outcome == OB_DISCHARGED and r.invocations`. The conjunct looks like a
    hardening and behaves like a second definition: empty `invocations` on a
    discharging record and the obligation still discharges — so the VERIFIED
    still stands — while its slice silently leaves the bar's domain.

    Measured on both shapes, against `8e42934` (whole-query bar) as control:

        fixture                         base      drifted   here
        one decided obligation          UNKNOWN   VERIFIED  UNKNOWN
        two, strip the scatter one's    UNKNOWN   VERIFIED  UNKNOWN

    The one-obligation row is the sharper of the two, because it ALSO closes a
    hole the base had: with every record's `invocations` gone the old
    `solver_decided` was False and the bar branch was never entered at all, so
    `8e42934` returned VERIFIED there too. Both rows are UNKNOWN now.

    This is a regression, not a defence. `make_solver_verdict`'s docstring
    states the precondition — the escalation came from `escalate()` on this
    query — and a caller who can hand-edit a record can hand-build a
    `Verdict` and skip this function entirely. What is pinned is that the
    bar's domain and the discharge cannot disagree about the SAME record.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    closed, prop, esc = _stamped(build)
    assert make_solver_verdict(closed, prop, esc, **VERSIONS).status == (
        "UNKNOWN"
    ), f"{label}: the genuine assembly does not bar; the fixture is wrong"
    assert any(r.index == strip and r.invocations for r in esc.records), (
        f"{label}: record #{strip} carries no invocations to strip, so this "
        f"test cannot see the drift"
    )

    forged = dataclasses.replace(esc, records=tuple(
        dataclasses.replace(r, invocations=()) if r.index == strip else r
        for r in esc.records
    ))
    v = make_solver_verdict(closed, prop, forged, **VERSIONS)
    assert [o.status for o in v.obligations] == (
        ["discharged"] * len(v.obligations)
    ), (
        f"{label}: emptying `invocations` un-discharged an obligation, so "
        f"there is no VERIFIED left for the bar to withhold and this test "
        f"passes for the wrong reason"
    )
    assert v.status == "UNKNOWN", (
        f"{label}: {v.status} — a record kept its discharge and left the "
        f"bar's domain, so `decided` and `discharged` are two predicates "
        f"again"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


@pytest.mark.parametrize("invocations,label", [
    ([], "an empty list"),
    (None, "None"),
    (7, "not iterable at all"),
])
def test_a_bar_must_never_BREAK_a_verdict_either(invocations, label):
    """"A bar must never break a verdict" has to cover the whole path FEEDING
    the bar, and at `eb1ff86` it did not.

    `_bar_scope` wraps its own body in `except Exception`. The read that
    builds its domain sat OUTSIDE that `try`, in `make_solver_verdict`, and
    was spelled `decided.get(r.index, ()) + r.invocations` — so a record whose
    `invocations` is a plain `list` raised `TypeError` out of the public
    assembly function. `45cf526` tolerated the same record.

    Both directions are required here. A tolerated shape must produce a
    verdict, and an untolerated one must WIDEN — never raise, and never go
    quiet, because an empty domain is how "no solver decided anything" is
    spelled and that silences the bar.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    forged = dataclasses.replace(esc, records=tuple(
        dataclasses.replace(r, invocations=(
            list(r.invocations) if invocations == [] else invocations
        ))
        for r in esc.records
    ))
    v = make_solver_verdict(closed, prop, forged, **VERSIONS)
    assert [o.status for o in v.obligations] == (
        ["discharged"] * len(v.obligations)
    ), f"{label}: nothing left to withhold, so this proves nothing"
    assert v.status == "UNKNOWN", (
        f"{label}: {v.status} — a record shape the bar cannot read cleared it"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


def test_the_containment_guard_short_circuits_before_any_re_slicing(monkeypatch):
    """SOUNDNESS.md's CONTAINMENT ARGUMENT, pinned at the mechanism it cites.

    The entry says every verdict the slice-scoped bar fires on is one the
    whole-query bar fired on too, and the reason it gives is structural:
    `_bar_scope` evaluates `whole = _barred_primitives(closed)` — the old
    bar's own set — FIRST, and returns an empty scope without re-slicing when
    it is empty. So "the new bar fired" implies "the whole-query set was
    non-empty" implies "the old bar fired", with no dependence on the two
    walks agreeing about anything.

    The weaker argument (a slice's equations come from the query, and both
    roots use the same walk) is also true and is pinned in
    `tests/test_bar_walk_parity.py`. It is not what the containment rests on,
    and the difference is the whole point: it would stop holding the moment
    the roots disagreed, which is the failure mode this repo keeps finding.

    Asserted as ORDER, not as outcome, and deliberately. Deleting the guard
    leaves every verdict in this suite unchanged — the re-derivation on a
    scatter-free query finds nothing on any slice and returns the same empty
    scope — so an outcome assertion cannot see it and the argument would be
    unpinned prose. What is observable is that the slicer is never consulted.
    """
    import stelling.obligation as _ob

    closed = trace(_scatter_free)
    assert not V._barred_primitives(closed), (
        "the fixture's whole-query barred set is non-empty, so the guard "
        "would not short-circuit and this test measures nothing"
    )

    calls = []
    real = _ob.slice_obligation

    def spy(*args, **kwargs):
        calls.append(args[1] if len(args) > 1 else None)
        return real(*args, **kwargs)

    monkeypatch.setattr(_ob, "slice_obligation", spy)
    assert V._bar_scope(closed, {0: ()}) == ((), "")
    assert calls == [], (
        f"`_bar_scope` re-sliced obligation(s) {calls} on a query whose "
        f"whole-query barred set is EMPTY. The early return is gone, so the "
        f"new bar's fired-set is no longer contained in the old one by "
        f"construction — it now depends on the slice walk finding nothing the "
        f"query walk missed. SOUNDNESS.md cites the guard; restore it or "
        f"re-derive the containment."
    )

    # ANTI-VACUITY: the spy must be reachable at all, or `calls == []` holds
    # for the trivial reason that nothing ever calls `slice_obligation`.
    dirty = trace(_scatter_ON_the_decided_slice)
    assert V._barred_primitives(dirty), "the control query carries nothing"
    V._bar_scope(dirty, {0: ()})
    assert calls, (
        "`_bar_scope` did not reach `slice_obligation` even on a "
        "scatter-bearing query, so the assertion above is vacuous"
    )


def test_the_note_names_only_the_obligations_whose_slice_carries_it():
    """FINDING 3: the message must not claim a scope the mechanism lacks.

    With two solver-decided obligations the predecessor built `{where}` from
    ALL of them while `{prims}` was the UNION, and nothing intersected the
    two: it rendered "the emitted slice of assert #0, assert #1 contains
    scatter" while #1's slice was `['add','le','sub']`. The single-deciding-
    obligation fixture above cannot see that, which is why this one exists.
    """
    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_two_solver_decided_obligations)
    # the bar's own predicate, verbatim — not a paraphrase of it
    deciding = [r.index for r in esc.records if r.outcome == "discharged"]
    assert sorted(deciding) == [0, 1], (
        f"the fixture no longer has TWO solver-decided obligations "
        f"({deciding}), so it cannot see the misattribution"
    )
    v = make_solver_verdict(closed, prop, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    (withheld,) = [n for n in v.notes if "VERIFIED withheld" in n]
    assert "assert #0" in withheld
    assert "assert #1" not in withheld, (
        f"the note names assert #1, whose emitted slice carries no barred "
        f"primitive — `where` is built from every deciding obligation while "
        f"`prims` is the union over them:\n  {withheld}"
    )


def test_the_bar_does_not_touch_interval_only_verdicts():
    """The negative half, and the reason the bar is scoped to the solver path.

    HeatNode's Dirichlet writeback puts `scatter` in the jaxpr, and this
    obligation is discharged by intervals alone. A whole-verdict bar would
    withhold the Richardson flagship for a reason having nothing to do with
    the emission row under audit — the interval transfer is long-standing and
    unchanged by that work.
    """
    pytest.importorskip("maddening")  # jax CI job does not install it
    from maddening.nodes.heat import HeatNode

    node = HeatNode("h", timestep=0.001, n_cells=5, length=1.0,
                    thermal_diffusivity=0.01)

    def row7():
        T = any_array((5,), "float64", (10.0, 100.0))
        st = {"temperature": T}
        full = node.update(st, {}, 0.01)
        half = node.update(node.update(st, {}, 0.005), {}, 0.005)
        scale = 0.0 + 1e-6 * jnp.maximum(jnp.abs(half["temperature"]),
                                         jnp.abs(full["temperature"]))
        return (assert_(scale > 0.0),)

    cj = jax.make_jaxpr(row7)()
    assert any(str(e.primitive) == "scatter" for e in cj.jaxpr.eqns), (
        "this test is vacuous unless the query really does contain scatter"
    )
    v = check(row7, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"
    assert not any("VERIFIED withheld" in n for n in v.notes)
