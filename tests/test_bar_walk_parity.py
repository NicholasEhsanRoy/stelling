# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The VERIFIED bar's traversal, and its firing direction.

TWO THINGS ARE TESTED HERE THAT COULD NOT BE TESTED BEFORE.

**Parity.** `_barred_primitives` used to descend via
`getattr(v, "jaxpr", None)`, which finds a param that IS a ClosedJaxpr but not
one holding a COLLECTION of them — and `cond` stores its branches as a tuple.
Measured: a `scatter` inside a `cond` branch was not barred at all, while the
same primitive at top level, inside `jit`, and inside `scan` all barred
correctly. That is the UNDER-firing direction, which the function's own
docstring calls the worse one. The fix routes descent through
`coverage.sub_jaxprs`; this module asserts mechanically that the two agree, in
the spirit of the EMISSION == REPLAY census — a review question nobody has to
remember to ask.

**The firing direction, via a SYNTHETIC barred primitive.** The bar withholds a
SOLVER-DECIDED VERIFIED, and a fixture reaching that branch needs a barred
primitive that can be EMITTED. `scatter` is one: measured on this build,
`"scatter" in obligation._SUPPORTED` is True, seven census idioms reach the row
(`tests/test_scatter_emission_reach.py`) and `tests/test_verified_bar.py`
drives a solver-decided scatter obligation end to end. This module said the
opposite for a while — that `scatter` was absent from the emission set, so the
protective branch was unreachable and the bar's whole behavioural history was
over-firing — and that was false when it was written; the correction is worth
keeping visible, because it is the same defect this file exists to catch, one
level up: a claim of non-occurrence that nothing checked.

Monkeypatching the barred set to `{"add"}` still earns its place, for two
reasons that survive the correction. `add` appears INSIDE a sub-jaxpr on a
real slice (`scatter-add` carries `update_jaxpr`), which is what makes the
descent observable at all; and it is emittable under enclosing constructs
where the real barred set cannot be driven — see
`test_the_bar_finds_a_barred_primitive_nested_in_cond` for the one that is
genuinely unreachable end-to-end, and why.

**A THIRD, since the bar became slice-scoped: THE SLICE ROOT.** The bar now
derives the barred set of the obligation the solver actually decided, by the
SAME walk rooted at that obligation's emitted slice equations rather than at
the query. Rooting a walk somewhere new is where a hand-rolled flat version
looks adequate and is not: measured, a `scatter-add` slice equation carries
`update_jaxpr = ['add']`, so under a synthetic barred set of `{"add"}` the flat
`{str(e.primitive) for e in sl.eqns}` finds NOTHING where the canonical walk
finds `add`. That measured disagreement is this module's anti-vacuity control
for the slice-root parity test — a parity test that cannot fail proves nothing.

**A FOURTH, since the bar's narrowing became slice-keyed: THE FINGERPRINT
WALK.** `smt.slice_primitive_walk` records every primitive name on a slice with
its nesting depth, and `smt.slice_fingerprint` hashes that; the bar narrows only
when a recorded invocation reproduces it. That is only a legitimate key if
**equal fingerprint implies equal barred set**, which holds exactly when the
fingerprint walk visits everything the bar's walk visits. It is a third
traversal of the same shape, so it is a third thing that can silently stop
matching — the lesson `_barred_in_eqns`'s own docstring draws. See
`test_the_fingerprint_walk_sees_everything_the_bar_walk_sees`, whose
anti-vacuity control is the same `cond` nesting that broke the first walk.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling import verdict as _verdict  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.coverage import sub_jaxprs  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from _solver_gate import need_solver  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ---- the query shapes, one per nesting construct --------------------------

@jax.jit
def _jitted(x):
    return x.at[0].set(0.5)


def _top():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[0].set(0.5) <= 2.0),)


def _in_jit():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(_jitted(x) <= 2.0),)


def _in_cond():
    x = any_array((3,), "float64", (0.0, 1.0))
    y = jax.lax.cond(x[0] > 0.5, lambda a: a.at[0].set(0.5), lambda a: a, x)
    return (assert_(y <= 2.0),)


def _in_scan():
    x = any_array((3,), "float64", (0.0, 1.0))

    def step(c, _):
        return c.at[0].set(0.5), 0.0

    y, _ = jax.lax.scan(step, x, jnp.arange(2.0))
    return (assert_(y <= 2.0),)


def _in_cond_in_jit():
    x = any_array((3,), "float64", (0.0, 1.0))

    @jax.jit
    def outer(a):
        return jax.lax.cond(a[0] > 0.5, lambda b: b.at[0].set(0.5),
                            lambda b: b, a)

    return (assert_(outer(x) <= 2.0),)


SHAPES = [
    (_top, "top level"),
    (_in_jit, "inside jit"),
    (_in_cond, "inside cond"),
    (_in_scan, "inside scan"),
    (_in_cond_in_jit, "cond inside jit"),
]


def _all_primitives_via_canonical(closed):
    found, seen = set(), []

    def walk(j):
        for e in j.eqns:
            found.add(str(e.primitive))
            for sub in sub_jaxprs(e):
                if id(sub) not in seen:
                    seen.append(id(sub))
                    walk(sub)

    walk(closed.jaxpr)
    return found


@pytest.mark.parametrize("build,label", SHAPES, ids=[s[1] for s in SHAPES])
def test_bar_walk_has_parity_with_the_canonical_accessor(build, label):
    """The bar must see every primitive `sub_jaxprs` sees. Checked over the
    barred set's own membership, which is what the bar acts on."""
    closed = transcribe(jax.make_jaxpr(build)())
    canonical = _all_primitives_via_canonical(closed)
    barred = set(_verdict._barred_primitives(closed))
    expected = canonical & _verdict.VERIFIED_BARRED_PRIMITIVES
    assert barred == expected, (
        f"{label}: the bar found {sorted(barred)} where the canonical walk "
        f"implies {sorted(expected)} — the two traversals have diverged"
    )


@pytest.mark.parametrize("build,label", SHAPES, ids=[s[1] for s in SHAPES])
def test_the_fingerprint_walk_sees_everything_the_bar_walk_sees(build, label):
    """THE KEY'S PREMISE: equal fingerprint implies equal barred set.

    The bar narrows an obligation only when the recorded invocation
    reproduces `smt.slice_fingerprint` of the re-derived slice. That is a
    legitimate key for a question about BARRED PRIMITIVES only if the walk
    the fingerprint hashes records every primitive the bar's walk can find —
    otherwise two slices could hash equal with different barred sets, which is
    precisely the failure the script hash has (`scatter` emits no text, so the
    SCRIPT cannot tell those two slices apart).

    Checked as an identity rather than a containment, because the containment
    direction alone would pass for a walk that records extra names it never
    reaches.
    """
    from stelling import smt as _smt

    closed = transcribe(jax.make_jaxpr(build)())
    eqns = closed.jaxpr.eqns
    names = {s.split(":", 1)[1] for s in _smt.slice_primitive_walk(eqns)}
    assert set(_verdict._barred_in_eqns(eqns)) == (
        names & _verdict.VERIFIED_BARRED_PRIMITIVES
    ), (
        f"{label}: the bar walks to {sorted(_verdict._barred_in_eqns(eqns))} "
        f"while the fingerprint walk records {sorted(names)} — the two "
        f"traversals have diverged, so a fingerprint match no longer implies "
        f"an equal barred set and the bar's key proves nothing about it"
    )
    # and it is not vacuous on this shape: the canonical accessor agrees, and
    # for the nested shapes a flat walk would not reach the primitive at all
    assert names == _all_primitives_via_canonical(closed), (
        f"{label}: the fingerprint walk and the canonical accessor disagree"
    )


def test_the_fingerprint_walk_would_catch_the_old_accessor():
    """ANTI-VACUITY for the test above, in the same shape that broke the first
    walk: a barred primitive inside a `cond` branch.

    A fingerprint walk descending the old way misses it, so the two walks
    would agree on a scatter-free NAME SET while the bar's walk found
    `scatter` — the identity above would fail. If this control ever stops
    failing, the fixture no longer nests anything and the parity test above
    is measuring a flat query."""
    from stelling import smt as _smt

    closed = transcribe(jax.make_jaxpr(_in_cond)())
    eqns = closed.jaxpr.eqns

    def old_walk(items):
        found = set()
        for eqn in items:
            found.add(str(eqn.primitive))
            for v in eqn.params_dict().values():
                inner = getattr(v, "jaxpr", None)  # the defect
                if inner is not None:
                    found |= old_walk(getattr(inner, "eqns", ()))
        return found

    canonical = {s.split(":", 1)[1] for s in _smt.slice_primitive_walk(eqns)}
    assert old_walk(eqns) != canonical, (
        "the OLD descent records the same names as the fingerprint walk on "
        "this query, so the parity test above cannot distinguish them"
    )
    assert canonical & _verdict.VERIFIED_BARRED_PRIMITIVES, (
        "the fixture carries no barred primitive; the control is vacuous"
    )


def test_the_parity_test_catches_the_old_accessor():
    """ANTI-VACUITY (docs/norms.md, "A measurement whose result is an ABSENCE needs a positive control"). A parity test that passes under the BROKEN walk
    would prove nothing. Re-implement the old descent and confirm it fails
    exactly where it did in the field: a barred primitive inside `cond`."""
    closed = transcribe(jax.make_jaxpr(_in_cond)())

    def old_walk(closed):
        found, seen = set(), []

        def walk(jaxpr):
            for eqn in getattr(jaxpr, "eqns", ()):
                if str(eqn.primitive) in _verdict.VERIFIED_BARRED_PRIMITIVES:
                    found.add(str(eqn.primitive))
                for v in eqn.params_dict().values():
                    inner = getattr(v, "jaxpr", None)   # the defect
                    if inner is not None and id(inner) not in seen:
                        seen.append(id(inner))
                        walk(inner)

        walk(closed.jaxpr)
        return found

    canonical = _all_primitives_via_canonical(closed)
    expected = canonical & _verdict.VERIFIED_BARRED_PRIMITIVES
    assert expected, "the fixture carries no barred primitive; test is vacuous"
    assert old_walk(closed) != expected, (
        "the OLD accessor agrees with the canonical one on this query, so the "
        "parity test above cannot distinguish them and proves nothing"
    )
    assert set(_verdict._barred_primitives(closed)) == expected


# ---- the firing direction, via a SYNTHETIC barred primitive ---------------

# The number words this module's docstring is allowed to spell a count with.
# A count that drifts out of this map fails loudly rather than silently
# comparing unequal against a string.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def test_the_docstring_premises_this_module_argues_from():
    """THE DOCSTRING'S PREMISES — the ones an assertion can actually be the
    signal for.

    This module's docstring used to say `scatter` was absent from the emission
    set, so the bar's protective branch was unreachable. That was false when it
    was written and nothing checked it — a claim of non-occurrence resting on a
    registry membership, in the file whose whole subject is claims that stopped
    matching their mechanism.

    THE FIRST REPAIR ASSERTED FOUR REGISTRY MEMBERSHIPS AND THAT WAS
    DECORATIVE. Measured, one mutation per assertion:

      * `"scatter" in propagate.TRANSFERS` and `"cond" not in
        propagate.TRANSFERS` — the falsifying edits raise
        `RuntimeError: the integer-semantics census must stay total over
        TRANSFERS` at import. `stelling.verdict` imports `stelling.propagate`,
        so THIS FILE fails at COLLECTION and the assertions never run; the
        full suite stops with 30 collection errors and runs nothing at all.
      * `"scatter" in obligation._SUPPORTED` and `"cond" not in
        obligation._SUPPORTED` — the same census, emission side, raised from
        the test's own import statement. Withdrawing `scatter` from
        `_SUPPORTED`, `_INT_SAFE_EMITTED` and `_REPLAY_SUPPORTED` together
        does NOT get as far as running either, and an earlier version of this
        paragraph said it was "the only version that does": those three keep
        the int-semantics and replay censuses total, and a FOURTH census —
        `obligation._assert_emission_classification_censused`, called at
        import — then refuses the `_INT_SAFE_EMITTED_REASONS` entry for
        `scatter` as a stale claim ("a stale reason is a soundness claim
        about nothing"). Withdrawing that fourth registry too does import,
        and leaves 3 failures in this file, every one of them naming the
        mechanism (the slice-root parity pair and the emitted-vs-re-derived
        agreement, on the `scatter set slice` shape). Measured on this build.

    An assertion whose falsification is already a louder failure somewhere
    else is not a check; it is a restatement. So the memberships are gone and
    what is asserted here is the premise NOTHING else measures — the docstring
    quotes a COUNT, and a count is exactly the kind of claim that goes stale
    silently. `caac1ee` moved `REACHES_THE_EMISSION_ROW` from six entries to
    seven, in the same commit that wrote "six census idioms reach the row"
    into the text above, and nothing noticed.

    The `cond` premise survives, restated as the thing it is actually about:
    `test_the_bar_finds_a_barred_primitive_nested_in_cond` says the
    end-to-end version "cannot be written" because a cond-bearing query never
    reaches the solver. That is a behaviour, not a membership, so it is
    measured as one.
    """
    import re

    from test_scatter_emission_reach import REACHES_THE_EMISSION_ROW

    module_doc = globals()["__doc__"]
    found = re.findall(r"(\w+) census idioms reach the row", module_doc)
    assert len(found) == 1, (
        f"the module docstring no longer states the reach count exactly once "
        f"({found}); this test reads it from there"
    )
    stated = _NUMBER_WORDS.get(found[0].lower())
    assert stated is not None, (
        f"the module docstring spells the reach count {found[0]!r}, which is "
        f"not a number word this test can read — add it to _NUMBER_WORDS or "
        f"spell the count with one"
    )
    assert stated == len(REACHES_THE_EMISSION_ROW), (
        f"the docstring above says {found[0]} census idioms reach the scatter "
        f"emission row; tests/test_scatter_emission_reach.py pins "
        f"{len(REACHES_THE_EMISSION_ROW)}. One of the two moved — re-derive "
        f"which, and note that the pin is the measurement and the docstring "
        f"is the claim about it"
    )


def test_a_cond_bearing_query_never_reaches_the_solver():
    """The `cond` premise, as a behaviour rather than a registry membership.

    `test_the_bar_finds_a_barred_primitive_nested_in_cond` argues that the
    end-to-end version of the cond under-fire "cannot be written" because a
    query containing `cond` can never carry a solver-decided obligation. The
    registry form of that argument (`"cond" not in obligation._SUPPORTED and
    not in propagate.TRANSFERS`) cannot be falsified without an import-time
    census raise, so it could never be the signal. This can: it asks the
    slicer, and any consistent change that made `cond` emittable or
    transparent would let the slice through and fail here.
    """
    from stelling.obligation import DeclinedObligation, slice_unknown_obligations
    from stelling.propagate import interval_env, propagate

    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = jax.lax.cond(x[0] > 0.5, lambda a: (a + 1.0) - a,
                         lambda a: (a + 1.0) - a, x)
        return (assert_(y <= 1.5),)

    closed = transcribe(jax.make_jaxpr(q)())
    p = propagate(closed)
    assert any(o.status == "unknown" for o in p.obligations), (
        "intervals settled the cond query, so nothing would have escalated "
        "anyway and this test is not measuring the emission set"
    )
    sliced = list(slice_unknown_obligations(closed, p, interval_env(closed)))
    assert sliced and all(isinstance(s, DeclinedObligation) for s in sliced), (
        "a cond-bearing obligation now slices for escalation, so the "
        "cond-nested under-fire may be reachable end-to-end — see "
        "test_the_bar_finds_a_barred_primitive_nested_in_cond, whose "
        "'cannot be written' rests on exactly this"
    )
    assert all("'cond'" in s.reason for s in sliced), (
        f"the cond obligation stops short for a reason other than `cond` "
        f"itself: {[s.reason[:90] for s in sliced]}"
    )


@pytest.fixture
def _bar_add(monkeypatch):
    """Bar `add` — which IS emittable, unlike `scatter`. That is the whole
    point: it makes the bar's protective branch reachable."""
    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES",
                        frozenset({"add"}))
    yield


def _solver_decided_query():
    """Correlation-sensitive, so intervals cannot settle it and the solver
    must — which is the precondition for the bar to apply at all."""
    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        return (assert_((x + 1.0) - x <= 1.5),)
    return q


def _obl_solves(v):
    sol = v.stamp.solver if v.stamp else None
    sols = sol if isinstance(sol, tuple) else (sol,)
    return len([s for s in sols if s and s.invoked and "widen" not in s.reason])


@need_solver
def test_the_bar_withholds_a_solver_decided_verified(_bar_add):
    """THE POSITIVE DIRECTION, exercised for the first time."""
    build = _solver_decided_query()
    v = check(build, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert _obl_solves(v) > 0, (
        "intervals settled it, so the bar never applies and this test does "
        "not exercise the firing direction"
    )
    assert v.status != "VERIFIED", "a solver-decided VERIFIED was not withheld"
    assert any("VERIFIED withheld" in n for n in v.notes)


def test_the_bar_finds_a_barred_primitive_nested_in_cond(_bar_add):
    """The traversal fix, at the level where it is decidable.

    The end-to-end version cannot be written: reaching the bar requires a
    SOLVER-DECIDED verdict, and a query containing `cond` can never have one,
    because `cond` is in NEITHER `TRANSFERS` NOR the emission set (measured on
    this build; `scatter` is in both). The enclosing construct blocks
    escalation whatever the barred set contains, so this under-fire was
    latent even though the barred primitive itself is emittable.

    That is worth stating precisely rather than overclaiming: the defect was
    latent, not live, and its reachability needed a collection-valued jaxpr
    param on a primitive that is emittable or transparent. `cond`/`scan`/
    `while` are collection-valued but not emittable; `jit`/`remat` are
    transparent and ARE reached, and the walk handled those correctly all
    along.

    What remains testable, and is what the fix is actually about, is that
    `_barred_primitives` SEES the primitive. Asserted directly.
    """
    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = jax.lax.cond(x[0] > 0.5, lambda a: (a + 1.0) - a,
                         lambda a: (a + 1.0) - a, x)
        return (assert_(y <= 1.5),)

    closed = transcribe(jax.make_jaxpr(q)())
    assert "add" in _verdict._barred_primitives(closed), (
        "a barred primitive inside a cond branch was not found — the "
        "traversal is not reaching cond's branches"
    )


def test_the_synthetic_bar_does_not_leak(_bar_add):
    """The fixture must not make everything UNKNOWN. A query with no `add`
    on it is unaffected, or the two tests above pass for the wrong reason."""
    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        return (assert_(x * 1.0 <= 2.0),)

    v = check(q, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert not any("VERIFIED withheld" in n for n in v.notes)


# ---- the SLICE root: the same walk, rooted at an obligation's slice -------

def _all_primitives_in_eqns(eqns):
    """The canonical walk, rooted at an arbitrary equation iterable rather
    than at a jaxpr — `sub_jaxprs` and nothing hand-rolled."""
    found, seen = set(), []

    def walk(items):
        for e in items:
            found.add(str(e.primitive))
            for sub in sub_jaxprs(e):
                if id(sub) not in seen:
                    seen.append(id(sub))
                    walk(sub.eqns)

    walk(eqns)
    return found


def _flat_primitives_in_eqns(eqns):
    """THE NAIVE VERSION, re-implemented here so the parity test below has
    something it can actually fail against: the one-line comprehension a
    slice-rooted bar invites, which never opens a sub-jaxpr."""
    return {str(e.primitive) for e in eqns}


def _slices_of(build):
    from stelling.obligation import (
        DeclinedObligation,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    closed = transcribe(jax.make_jaxpr(build)())
    p = propagate(closed)
    return [
        s for s in slice_unknown_obligations(closed, p, interval_env(closed))
        if not isinstance(s, DeclinedObligation)
    ]


def _segment_sum_relational():
    """Escalates, and its slice's `scatter-add` equation carries the
    recorded combiner sub-jaxpr — the case the flat version misses."""
    import numpy as np

    d = any_array((2,), "float64", (0.0, 1.0))
    s = jax.ops.segment_sum(
        d, jnp.asarray(np.array([0, 0], dtype=np.int32)), num_segments=1
    )
    return (assert_(s[0] >= d[0]),)


def _set_relational():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[0].set(0.5)[1] - x[1] <= 0.0),)


def _plain_relational():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_((x + 1.0) - x <= 1.5),)


SLICE_SHAPES = [
    (_segment_sum_relational, "scatter-add slice (carries a sub-jaxpr)"),
    (_set_relational, "scatter set slice"),
    (_plain_relational, "flat arithmetic slice"),
]


@pytest.mark.parametrize("build,label", SLICE_SHAPES,
                         ids=[s[1] for s in SLICE_SHAPES])
@pytest.mark.parametrize("barred", [frozenset({"add"}), frozenset({"scatter"})],
                         ids=["synthetic add", "real scatter"])
def test_slice_root_walk_has_parity_with_the_canonical_accessor(
    monkeypatch, build, label, barred
):
    """The slice-rooted root of the bar's walk must see every primitive
    `sub_jaxprs` sees from the SAME equations — checked over the barred set's
    own membership, which is what the bar acts on, and under both the real
    barred set and a synthetic one that reaches inside a sub-jaxpr."""
    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES", barred)
    slices = _slices_of(build)
    assert slices, f"{label}: nothing escalated, so no slice exists to root at"
    for sl in slices:
        canonical = _all_primitives_in_eqns(sl.eqns)
        got = set(_verdict._barred_in_eqns(sl.eqns))
        assert got == canonical & barred, (
            f"{label}: the slice-rooted walk found {sorted(got)} where the "
            f"canonical walk implies {sorted(canonical & barred)} — the two "
            f"traversals have diverged"
        )


def test_the_slice_parity_test_catches_the_naive_flat_version(monkeypatch):
    """ANTI-VACUITY (docs/norms.md, "A measurement whose result is an ABSENCE needs a positive control") for the slice root, on the MEASURED case.

    The flat comprehension is the implementation a slice-scoped bar invites,
    and on a `scatter-add` slice under a barred set of `{"add"}` it is wrong:
    the combiner lives in the equation's `update_jaxpr`, not in the slice's
    own equation list. If it AGREED here, the parity test above could not
    distinguish the two and would prove nothing.
    """
    barred = frozenset({"add"})
    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES", barred)
    slices = _slices_of(_segment_sum_relational)
    assert slices, "nothing escalated; the control has no slice to run on"
    sl = slices[0]

    canonical = _all_primitives_in_eqns(sl.eqns) & barred
    assert canonical, (
        "the fixture's slice carries no barred primitive at any depth; the "
        "control is vacuous"
    )
    flat = _flat_primitives_in_eqns(sl.eqns) & barred
    assert flat != canonical, (
        f"the NAIVE flat version agrees with the canonical walk on this "
        f"slice ({sorted(flat)}), so the parity test above cannot "
        f"distinguish them and proves nothing"
    )
    assert set(_verdict._barred_in_eqns(sl.eqns)) == canonical


@pytest.mark.parametrize("build,label", SLICE_SHAPES,
                         ids=[s[1] for s in SLICE_SHAPES])
def test_the_bar_re_derives_the_slice_that_was_actually_emitted(
    monkeypatch, build, label
):
    """THE SEAM THE DERIVED SCOPE RESTS ON, asserted rather than argued.

    `verdict._bar_scope` does not read a scope off the escalation — a record
    cannot certify its own cleanliness, and nothing binds an escalation to the
    query it is stamped against. It re-slices the decided obligations out of
    `closed` instead. That is only sound if the re-derivation is the SAME
    slice `escalate` emitted, and it is by construction:
    `slice_unknown_obligations` calls `slice_obligation(closed, index,
    interval_env(closed))`, and its two further arguments are
    `top_primitives`, documented "message wording only, never admission", and
    `relational_assumes`, which is NOT wording — it is the script's axiom
    lines. This test pins the EQUATION walk only: it passes no
    `relational_assumes` and holds no assume-carrying fixture, so that axis is
    unpinned here. What makes the omission safe is measured, not assumed —
    `relational_assumes` moves `sl.assumes`/`sl.assumes_skipped` and never `sl.eqns`.

    **"BY CONSTRUCTION" STOPPED BEING TRUE QUIETLY AT THE B6/B7 MERGE, WHICH
    IS THE THING THE PARAGRAPH ABOVE SAYS ABOUT ITSELF.** There are THREE
    further arguments now, not two: B6's M17′ added `assert_position`, and
    `slice_unknown_obligations` passes it — the ordinal among top-level
    asserts, read off `propagate.ObligationReport.top_level_eqn_pos` —
    while `verdict._bar_scope` cannot, because it holds no propagation. So
    the bar's re-derivation is the same slice `escalate` emitted only for a
    query whose asserts are ALL top-level; from the first `assert_` written
    inside a sub-jaxpr onward the two select different asserts. The
    direction is safe (neither recorded hash reproduces, so the bar widens
    to the whole query) and it costs precision, not soundness. This module
    holds no nested-assert fixture, so that axis is unpinned here too; the
    measurement is in `verdict._bar_scope`'s block comment and in
    SOUNDNESS.md's B7 M10 entry.

    "By construction" is exactly the kind of claim that stops being true
    quietly, so it is measured here on every slice shape, under a synthetic
    barred set that reaches into a sub-jaxpr as well as the real one.
    """
    from stelling.obligation import (
        DeclinedObligation,
        slice_obligation,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES",
                        frozenset({"add", "scatter"}))
    closed = transcribe(jax.make_jaxpr(build)())
    p = propagate(closed)
    env = interval_env(closed)
    emitted = [s for s in slice_unknown_obligations(closed, p, env)
               if not isinstance(s, DeclinedObligation)]
    assert emitted, f"{label}: nothing escalated, so nothing is being pinned"
    for sl in emitted:
        again = slice_obligation(closed, sl.index, interval_env(closed))
        assert not isinstance(again, DeclinedObligation), (
            f"{label}: assert #{sl.index} sliced for escalation but DECLINES "
            f"on re-derivation ({again.reason}) — the bar would fall back to "
            f"the whole query on a verdict that has a slice"
        )
        assert _verdict._barred_in_eqns(again.eqns) == (
            _verdict._barred_in_eqns(sl.eqns)
        ), (
            f"{label}: the bar's re-derived slice for assert #{sl.index} "
            f"carries a different barred set than the one that was emitted — "
            f"the scope is no longer measuring the slice the solver saw"
        )


def test_the_re_derivation_is_not_vacuous(monkeypatch):
    """ANTI-VACUITY (docs/norms.md, "A measurement whose result is an ABSENCE needs a positive control") for the agreement above: at least one of those
    slices must actually CARRY a barred primitive, or the parity is between
    two empty sets and would hold under any re-derivation at all."""
    from stelling.obligation import (
        DeclinedObligation,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES",
                        frozenset({"add", "scatter"}))
    found = set()
    for build, _label in SLICE_SHAPES:
        closed = transcribe(jax.make_jaxpr(build)())
        p = propagate(closed)
        for sl in slice_unknown_obligations(closed, p, interval_env(closed)):
            if not isinstance(sl, DeclinedObligation):
                found.update(_verdict._barred_in_eqns(sl.eqns))
    assert found, (
        "no slice in SLICE_SHAPES carries a barred primitive under the "
        "synthetic set, so the agreement test compares empty against empty"
    )
