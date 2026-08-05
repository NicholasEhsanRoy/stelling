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


# ── the scope's PROVENANCE: derived from the query, never read off a record ──
#
# Everything below stamps a verdict through `make_solver_verdict` directly
# rather than through `check`, because that is the surface the defects were
# on: it is public, it takes `closed`, `propagation` and `escalation` as three
# independent arguments, and it gates their pairing on semantics, ieee,
# constrained-assume and ledger provenance — but on nothing that ties the
# escalation to the query.

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


def test_a_scatter_free_escalation_cannot_clear_a_scatter_BEARING_query():
    """THE MISPAIRING THE WHOLE-QUERY BAR WAS IMMUNE TO, and the reason the
    scope is derived rather than recorded.

    Nothing in `make_solver_verdict` binds `escalation` to `closed`. Stamp a
    genuine scatter-free escalation against a scatter-bearing query and the
    bar must still fire: the whole-query bar it replaced read `closed`
    directly and returned UNKNOWN here (measured on 8e42934), and a
    slice-scoped bar that reads its scope off the escalation returned
    VERIFIED. Narrowing the scope must not cost this.
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


def test_no_escalation_field_carries_the_bars_scope():
    """A RECORD MUST NOT BE ABLE TO CERTIFY ITS OWN CLEANLINESS.

    The predecessor's `barred_on_slice=()` was a positive claim ("nothing
    barred on my slice") that nothing validated: measured on this fixture, a
    record carrying it earned VERIFIED where the genuine scope was
    `('scatter',)`. The repair deletes the field rather than validating it, so
    what is pinned here is the ABSENCE of any such channel: every record field
    a caller can choose is reset to its default, and the bar must be
    unmoved — only `index`, `outcome` and `invocations` may reach it, and
    those are already load-bearing for the VERIFIED being withheld.
    """
    import dataclasses

    from stelling.solvers import ObligationEscalation, make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    assert make_solver_verdict(closed, prop, esc, **VERSIONS).status == (
        "UNKNOWN"
    ), "the genuine assembly does not bar; the fixture is wrong"

    load_bearing = {"index", "outcome", "invocations"}
    defaulted = {
        f.name: f.default
        for f in dataclasses.fields(ObligationEscalation)
        if f.name not in load_bearing and f.default is not dataclasses.MISSING
    }
    assert "barred_on_slice" not in {
        f.name for f in dataclasses.fields(ObligationEscalation)
    }, (
        "a record field named `barred_on_slice` is back. It was deleted "
        "because a recorded scope is a claim the bar cannot check; if it "
        "returns, the bar must not read it"
    )
    stripped = dataclasses.replace(esc, records=tuple(
        dataclasses.replace(r, detail="", witness=None, notes=(), **defaulted)
        for r in esc.records
    ))
    v = make_solver_verdict(closed, prop, stripped, **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: emptying every non-load-bearing record field cleared "
        f"the bar, so one of them is carrying the scope"
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
    deciding = [r.index for r in esc.records
                if r.outcome == "discharged" and r.invocations]
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
