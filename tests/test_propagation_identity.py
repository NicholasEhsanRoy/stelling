# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A propagation is about ONE query, and every site that consumes one
against a query checks that it is this one.

**THE DEFECT THIS MODULE CLOSES** — audit 0.2.0 B6 re-audit UNSOUND-3, live
on `main` and on the released `v0.1.0` until B11. `stelling.propagate.
Propagation` carried no query identity, so nothing anywhere compared it with
the query being judged. Two queries traced from ONE factory carry identical
`source_info` at identical top-level positions, which is exactly what the
per-obligation association check in `stelling.obligation.
slice_unknown_obligations` verifies — so the structural checks pass and the
assembly reports query B's obligations with query A's statuses.

`stelling.solvers.MispairedEscalationError` did not stop it and could not:
it binds `closed` to the ESCALATION, and `escalate` hashes the `closed` IT
was handed, so the pair that gate checks genuinely matches. On an obligation
the interval or affine leg decides outright the escalation carries no records,
no notes, no spawns and no stamps at all, `carries_work` is False, and the
gate is not consulted. Both arms reached VERIFIED on a query whose honest
verdict is REFUTED, and the mirror reached REFUTED on one whose honest
verdict is VERIFIED.

**WHAT IS PINNED HERE.** One red-without-the-fix row per site that consumes
a propagation against a query, driven through the public entry point:

* `stelling.verdict.make_verdict` — the no-solver assembler
* `stelling.solvers.make_solver_verdict` — the solver assembler, both
  `carries_work` arms
* `stelling.solvers.escalate` — the producer of solver outcomes
* `stelling.affine.refine_propagation` — public, below every gate, and it
  WRITES decided statuses into the argument nothing bound
* `stelling.obligation.slice_unknown_obligations` — public, and the place
  the structural association check lives

plus the identity's own producer (`propagate` stamps it, and the field has no
default so a hand-built propagation must state it), the refusal rule
(`unpaired_propagation` refuses an absent identity on EITHER leg — two
absences are not a match), and a STRUCTURAL row that re-derives the site list
from the source and fails when a new consumption site appears without the
check.

**AND WHAT IS NOT CLOSED, measured rather than asserted:** the `env`
argument of `slice_obligation`/`slice_unknown_obligations`. See
`test_the_env_channel_is_NOT_bound_and_here_is_what_that_costs`.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp

from stelling.harness import any_array, assert_, trace


@pytest.fixture(autouse=True)
def _x64():
    """Scope x64 to this module — the same discipline the rest of the suite
    uses, so a float64 declaration here cannot leak into a later module."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


VERSIONS = dict(
    stelling_version="0.0.0", jax_version="0.0.0", precision_config="x64"
)
CFG_KW = dict(timeout_ms=20_000)


# -- the fixtures: ONE factory, two boxes -------------------------------------
#
# One factory is the whole point. It gives both queries the same file, the
# same line, the same top-level assert positions and the same `source_info`,
# so every STRUCTURAL check downstream passes and the only thing that can
# separate them is the declared box — which is exactly what the content hash
# encodes and what nothing compared until B11.


def _two_asserts(lo, hi):
    """Two asserts. On [100, 101] both hold; on [1e9, 2e9] the first is false
    everywhere, so the honest verdicts are VERIFIED and REFUTED. The second
    assert is left `unknown` by the interval leg, so `escalate` does real
    solver work and `carries_work` is True."""

    def h():
        c = any_array((2,), jnp.float64, (lo, hi))
        return (assert_(c + c <= 1e9), assert_(c * c - c >= 9900.0))

    return h


def _interval_decided(lo, hi):
    """One assert the interval leg decides outright, so `escalate` returns an
    escalation carrying nothing at all and the escalation pairing gate is
    exempted entirely."""

    def h():
        c = any_array((2,), jnp.float64, (lo, hi))
        return assert_(c + c <= 1e9)

    return h


def _affine_decided(shift):
    """`c - c` is correlation-blind to intervals and EXACT in the affine
    domain, so the interval leg leaves this `unknown` and the affine
    refinement DECIDES it — discharged at shift 1.0, violated at shift 0.0.
    This is the fixture for `refine_propagation`, the site that writes
    decided statuses into the unbound argument."""

    def h():
        c = any_array((2,), jnp.float64, (0.0, 1.0))
        return assert_(c - c + shift >= 0.5)

    return h


def test_the_fixtures_ground_truth_is_exact_and_concrete():
    """The oracle, before any of this module's machinery is trusted: the
    claims are decided in exact rationals AND executed in concrete jax, so a
    row below that says "honest verdict REFUTED" is checkable without
    re-deriving it from the tool under test."""
    lo, hi = Fraction(10) ** 9, 2 * Fraction(10) ** 9
    assert not any(c + c <= Fraction(10) ** 9 for c in (lo, hi))
    assert all(c * c - c >= 9900 for c in (lo, hi))
    assert all(
        Fraction(100) + Fraction(100) <= Fraction(10) ** 9 for _ in (0,)
    )
    assert all(c * c - c >= 9900 for c in (Fraction(100), Fraction(101)))
    # concrete jax agrees about the false one
    x = jnp.full((2,), 1e9, jnp.float64)
    assert not bool(jnp.all(x + x <= 1e9))
    assert float((x + x)[0]) == 2e9
    # and about the affine pair: c - c is identically zero
    c = jnp.array([0.3, 0.7], jnp.float64)
    assert [float(v) for v in (c - c + 0.0)] == [0.0, 0.0]
    assert not bool(jnp.all(c - c + 0.0 >= 0.5))
    assert bool(jnp.all(c - c + 1.0 >= 0.5))


def _pair(factory, a_box, b_box):
    """(A, p_A, B, p_B) — two queries from one factory, with their own
    propagations, asserted distinct as queries and identical in structure."""
    from stelling.propagate import propagate

    a, b = trace(factory(*a_box)), trace(factory(*b_box))
    assert a.content_hash() != b.content_hash(), "the fixture is not two queries"
    pos = lambda q: [  # noqa: E731
        i for i, e in enumerate(q.jaxpr.eqns) if e.primitive == "stelling_assert"
    ]
    info = lambda q: [  # noqa: E731
        tuple(e.source_info)
        for e in q.jaxpr.eqns
        if e.primitive == "stelling_assert"
    ]
    assert pos(a) == pos(b) and info(a) == info(b), (
        "the two queries do not share assert positions and source_info, so "
        "the structural association check would separate them and these rows "
        "would not be measuring the query-identity check at all"
    )
    return a, propagate(a), b, propagate(b)


# -- the identity itself ------------------------------------------------------


def test_propagate_stamps_the_query_it_walked():
    from stelling.propagate import propagate, query_identity

    q = trace(_two_asserts(100.0, 101.0))
    assert propagate(q).query_sha256 == q.content_hash() == query_identity(q)


def test_the_identity_field_has_NO_default():
    """A `Propagation` cannot be built without saying which query it is
    about. The sibling `Escalation.query_sha256` DOES default to `""`, which
    is why a whole test exists to pin that every `escalate` return site
    remembers to pass it; a `Propagation` has one construction site in the
    library, so the hole is made unconstructible instead of tested for."""
    import inspect

    from stelling.propagate import Propagation

    f = Propagation.__dataclass_fields__["query_sha256"]
    assert f.default is dataclasses.MISSING, (
        "query_sha256 has acquired a default; a construction site that "
        "forgets it is now silent rather than a TypeError"
    )
    assert f.default_factory is dataclasses.MISSING
    with pytest.raises(TypeError):
        Propagation(
            obligations=(),
            nonvacuity_checks=(),
            coverage=None,
            transfers_used=(),
            assumptions=(),
            notes=(),
        )
    # and exactly one construction site in the library, which is what makes
    # the no-default argument above true rather than lucky
    from stelling import propagate as P

    assert inspect.getsource(P).count("return Propagation(") == 1


@pytest.mark.parametrize(
    "recorded,query,why",
    [
        ("", "abc", "an absent identity on the propagation"),
        (None, "abc", "a None identity on the propagation"),
        (12345, "abc", "a non-string identity on the propagation"),
        (b"abc", "abc", "a bytes identity that would never equal a str"),
        ("abc", "", "an unhashable query on the other leg"),
        ("", "", "TWO absences, which are not a match"),
        ("abc", "def", "two different queries"),
    ],
)
def test_unpaired_propagation_refuses_every_way_of_not_saying_so(
    recorded, query, why
):
    """The property is "this propagation is about this query, and BOTH of
    them said so" — not equality. Two empty strings compared equal is the
    exact hole the escalation leg carried until `e35de13`, and it is not
    being rebuilt here: `("", "")` is refused."""
    from stelling.propagate import unpaired_propagation

    stub = dataclasses.make_dataclass("Stub", ["query_sha256"])(recorded)
    reason = unpaired_propagation(stub, query)
    assert reason is not None, why
    assert reason.startswith("unpaired propagation: "), reason


def test_the_refusal_itself_cannot_raise_on_a_hostile_propagation():
    """A REFUSAL THAT RAISES IS NOT A REFUSAL — audit 0.2.0 B6 re-audit R7,
    applied to this repair's own check.

    `obligation.slice_unknown_obligations` MAY NOT raise: both library
    callers iterate it in a `for` header outside their own per-obligation
    nets, so an escape costs every obligation's verdict rather than one. The
    pairing check runs before that function's net, on whatever the caller
    passed as a propagation, so its own two reads — the attribute, and then
    a description of what was read — are each a place a hostile object can
    make it raise. All three shapes are driven, and each must produce a
    STRING (a refusal) rather than an exception.
    """
    from stelling.propagate import (
        UNREADABLE_PROPAGATION_IDENTITY,
        unpaired_propagation,
    )

    class RaisingAttribute:
        @property
        def query_sha256(self):
            raise RuntimeError("no")

    class RaisingRepr(int):
        def __repr__(self):
            raise RuntimeError("no")

    class RaisingName(type):
        @property
        def __name__(cls):
            raise RuntimeError("no")

    class Undescribable(metaclass=RaisingName):
        pass

    stub = dataclasses.make_dataclass("Stub", ["query_sha256"])

    r1 = unpaired_propagation(RaisingAttribute(), "abc")
    assert isinstance(r1, str) and "raised" in r1 and "RuntimeError" in r1

    r2 = unpaired_propagation(stub(RaisingRepr(3)), "abc")
    assert isinstance(r2, str) and "<RaisingRepr>" in r2, r2

    r3 = unpaired_propagation(stub(Undescribable()), "abc")
    assert r3 == UNREADABLE_PROPAGATION_IDENTITY, r3

    # ... and end to end at the site where a raise is most expensive
    from stelling.obligation import DeclinedObligation, slice_unknown_obligations
    from stelling.propagate import interval_env, propagate

    q = trace(_two_asserts(100.0, 101.0))
    honest = propagate(q)
    hostile = dataclasses.replace(honest, query_sha256=RaisingRepr(3))
    got = slice_unknown_obligations(q, hostile, interval_env(q))
    assert got and all(isinstance(i, DeclinedObligation) for i in got)
    assert all("<RaisingRepr>" in i.reason for i in got), [i.reason for i in got]


def test_unpaired_propagation_accepts_the_honest_pairing():
    from stelling.propagate import propagate, unpaired_propagation

    q = trace(_two_asserts(100.0, 101.0))
    assert unpaired_propagation(propagate(q), q.content_hash()) is None


# -- one row per consumption site ---------------------------------------------


@pytest.mark.parametrize(
    "factory,a_box,b_box,carries_work",
    [
        (_two_asserts, (100.0, 101.0), (1e9, 2e9), True),
        (_interval_decided, (0.0, 1.0), (1e9, 2e9), False),
    ],
    ids=["carries-work", "exempt"],
)
def test_SITE_make_solver_verdict_refuses_a_stranger_propagation(
    factory, a_box, b_box, carries_work
):
    """`solvers.make_solver_verdict` — the row that was the false VERIFIED.

    RED WITHOUT THE FIX: both arms returned `VERIFIED` on B, whose honest
    verdict is `REFUTED`, on `main` (`207faca`) and on `v0.1.0`."""
    from stelling.solvers import SolverConfig, escalate, make_solver_verdict

    cfg = SolverConfig(**CFG_KW)
    a, p_a, b, p_b = _pair(factory, a_box, b_box)
    honest = make_solver_verdict(b, p_b, escalate(b, p_b, cfg), **VERSIONS)
    assert honest.status == "REFUTED", (
        f"B's honest verdict is {honest.status}; the fixture no longer "
        f"measures a FALSE verified"
    )

    # the arm this row names, pinned on the HONEST escalation of the query
    # whose propagation is about to be mispaired forward: `carries_work` is
    # what decides whether the ESCALATION gate is consulted at all, and the
    # `exempt` arm is the one where it is not.
    own = escalate(a, p_a, cfg)
    assert bool(
        own.records or own.notes or own.ledger.spawns or own.ledger.stamps
    ) is carries_work, (
        "this fixture no longer exercises the carries_work arm it names"
    )

    esc = escalate(b, p_a, cfg)
    assert esc.query_sha256 == b.content_hash(), (
        "the escalation does not name B, so the ESCALATION gate would refuse "
        "it and this row would measure the covered leg instead"
    )
    v = make_solver_verdict(b, p_a, esc, **VERSIONS)
    assert v.status == "UNKNOWN", f"{v.status}: a stranger propagation minted"
    assert v.obligations == (), (
        "the mispaired propagation's obligations are still being reported "
        "under B's name, which is the misattribution itself"
    )
    assert v.stamp.query_content_hash == b.content_hash()
    assert len(v.notes) == 1 and v.notes[0].startswith("unpaired propagation:")
    assert a.content_hash() in v.notes[0] and b.content_hash() in v.notes[0], (
        "the note does not name BOTH queries; a caller who has just lost a "
        "verdict cannot tell which propagation they passed"
    )
    # the mirror: a stranger cannot mint a REFUTED either
    mirror = make_solver_verdict(a, p_b, escalate(a, p_b, cfg), **VERSIONS)
    assert mirror.status == "UNKNOWN", (
        f"{mirror.status}: the mirror direction mints — A's honest verdict is "
        f"VERIFIED and B's propagation refutes"
    )


def test_SITE_make_verdict_refuses_a_stranger_propagation():
    """`verdict.make_verdict` — the no-solver assembler, and the one with no
    refusal vocabulary of its own before B11.

    RED WITHOUT THE FIX: `VERIFIED` on B (honest `REFUTED`) and `REFUTED` on
    A (honest `VERIFIED`), with no solver anywhere in the chain."""
    from stelling.affine import refine_propagation
    from stelling.verdict import make_verdict

    a, p_a, b, p_b = _pair(_affine_decided, (1.0,), (0.0,))
    ra, rep_a = refine_propagation(a, p_a)
    rb, rep_b = refine_propagation(b, p_b)
    assert make_verdict(a, ra, refinement=rep_a, **VERSIONS).status == "VERIFIED"
    assert make_verdict(b, rb, refinement=rep_b, **VERSIONS).status == "REFUTED"

    for closed, stranger, honest in ((b, ra, "REFUTED"), (a, rb, "VERIFIED")):
        v = make_verdict(closed, stranger, **VERSIONS)
        assert v.status == "UNKNOWN", (
            f"{v.status}: a propagation of another query reached a definite "
            f"verdict through make_verdict (honest here: {honest})"
        )
        assert v.obligations == ()
        assert v.stamp.query_content_hash == closed.content_hash()
        assert v.stamp.semantics.startswith("not reached"), (
            "the stamp is quoting the stranger propagation's derived fields, "
            "which is the misattribution in a smaller costume"
        )
        assert v.stamp.transfer_tiers == () and v.stamp.assumptions == ()
        assert v.notes and v.notes[0].startswith("unpaired propagation:")


def test_SITE_escalate_refuses_a_stranger_propagation():
    """`solvers.escalate` — it selects obligations off the propagation and
    slices them out of `closed`, so a stranger's obligation NUMBERING would
    be filed against this query's slices.

    RED WITHOUT THE FIX: `escalate(B, p_A)` returned a real discharge record
    for obligation 1, produced by two solver spawns on B's slice."""
    from stelling.solvers import SolverConfig, escalate

    cfg = SolverConfig(**CFG_KW)
    a, p_a, b, p_b = _pair(_two_asserts, (100.0, 101.0), (1e9, 2e9))
    # the honest control is A's own escalation: A is the query whose
    # propagation has an obligation left `unknown`, so it is the one that
    # reaches a backend at all. B's interval leg decides both of its own.
    honest = escalate(a, p_a, cfg)
    stranger = escalate(b, p_a, cfg)

    assert stranger.records == (), (
        f"{[(r.index, r.outcome) for r in stranger.records]}: a stranger "
        f"propagation still produced per-obligation outcomes"
    )
    assert stranger.ledger.spawns == 0, "a solver ran on a stranger pairing"
    assert len(stranger.notes) == 1
    assert stranger.notes[0].startswith("escalation declined: unpaired ")
    assert stranger.query_sha256 == b.content_hash(), (
        "the refusal must still name the query it was asked about, or the "
        "ESCALATION gate one layer up refuses it for the wrong reason"
    )
    assert stranger.semantics == p_a.semantics
    # the honest control, so the row above is not passing because escalation
    # is broken for everything: the SAME propagation, against its OWN query,
    # reaches a backend and returns a real outcome
    assert [(r.index, r.outcome) for r in honest.records] == [(1, "discharged")]
    assert honest.ledger.spawns > 0


def test_SITE_refine_propagation_refuses_a_stranger_propagation():
    """`affine.refine_propagation` — public, below every gate, and it WRITES
    decided statuses into the propagation it returns.

    RED WITHOUT THE FIX: `refine_propagation(A, p_B)` returned
    `['discharged']` for B's obligation, and `refine_propagation(B, p_A)`
    returned `['violated-over-set']` for A's."""
    from stelling.affine import refine_propagation

    a, p_a, b, p_b = _pair(_affine_decided, (1.0,), (0.0,))
    assert [o.status for o in refine_propagation(a, p_a)[0].obligations] == [
        "discharged"
    ]
    assert [o.status for o in refine_propagation(b, p_b)[0].obligations] == [
        "violated-over-set"
    ]

    for closed, stranger in ((a, p_b), (b, p_a)):
        refined, report = refine_propagation(closed, stranger)
        assert [o.status for o in refined.obligations] == ["unknown"], (
            "the affine domain decided a stranger's obligation with this "
            "query's arithmetic"
        )
        assert report.discharged == () and report.violated == ()
        assert [i for i, _ in report.declined] == [0]
        assert any(
            n.startswith("affine refinement declined wholly: unpaired ")
            for n in refined.notes
        ), refined.notes
        assert refined.query_sha256 == stranger.query_sha256, (
            "the refusal RE-STAMPED the propagation with this query's "
            "identity, which launders the stranger past every later gate"
        )


def test_SITE_refine_propagation_refuses_a_stranger_with_NOTHING_unknown():
    """The arm the `not unknown` early return used to hand back silently. A
    propagation whose obligations are all decided is returned UNCHANGED by
    that return, which reads exactly like a refinement that had nothing to
    do — and it is the arm the measured false VERIFIED went through, because
    the mispairing happened AFTER the refinement decided."""
    from stelling.affine import refine_propagation
    from stelling.propagate import propagate

    a = trace(_interval_decided(0.0, 1.0))
    b = trace(_interval_decided(1e9, 2e9))
    p_a = propagate(a)
    assert all(o.status != "unknown" for o in p_a.obligations)

    refined, report = refine_propagation(b, p_a)
    assert report.attempted == () and report.declined == ()
    assert any(
        n.startswith("affine refinement declined wholly: unpaired ")
        for n in refined.notes
    ), (
        f"{refined.notes}: a stranger with nothing left `unknown` was handed "
        f"back with no record that anything was refused"
    )


def test_SITE_slice_unknown_obligations_refuses_a_stranger_propagation():
    """`obligation.slice_unknown_obligations` — public, in `__all__`, and the
    home of the STRUCTURAL association check that two queries from one factory
    satisfy by construction.

    RED WITHOUT THE FIX: it returned a real `ObligationSlice` for obligation
    1, bounded by B's declarations, under A's propagation."""
    from stelling.obligation import (
        DeclinedObligation,
        ObligationSlice,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env

    a, p_a, b, p_b = _pair(_two_asserts, (100.0, 101.0), (1e9, 2e9))
    # the honest control is A's own pairing: A is the query whose propagation
    # has an obligation left `unknown`, so it is the one there is anything to
    # slice for
    honest = slice_unknown_obligations(a, p_a, interval_env(a))
    assert any(isinstance(i, ObligationSlice) for i in honest), (
        "the honest pairing does not slice, so a decline below proves nothing"
    )

    got = slice_unknown_obligations(b, p_a, interval_env(b))
    assert got and all(isinstance(i, DeclinedObligation) for i in got), (
        f"{[type(i).__name__ for i in got]}: a stranger propagation still "
        f"slices out of this query"
    )
    assert all(i.reason.startswith("unpaired propagation:") for i in got)
    assert [i.index for i in got] == [
        o.index for o in p_a.obligations if o.status == "unknown"
    ], "the declines do not account for every obligation that was asked about"


# -- the check cannot be forgotten by a new caller ----------------------------


def _consumption_sites():
    """Every module-level function in the library whose signature takes BOTH
    a query and a propagation — derived from the source, not listed.

    This IS the enumeration the fix rests on, so it is re-derived on every
    run rather than trusted: an earlier attempt at this repair inherited a
    six-name list of which three names (`inductive`, `vacuity`,
    `reachability`) consume no propagation at all and two real sites
    (`slice_unknown_obligations`, `make_solver_verdict`) were missing.
    """
    import ast
    import pathlib

    import stelling

    root = pathlib.Path(stelling.__file__).parent
    out = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            args = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            ann = {
                a.arg: ast.unparse(a.annotation) if a.annotation else ""
                for a in args
            }
            takes_prop = any("Propagation" in t for t in ann.values())
            # a query argument is named `closed` or annotated ClosedJaxpr:
            # `verdict.make_verdict`'s is UNANNOTATED, which is precisely the
            # site a type-only derivation would miss
            takes_query = any(
                "ClosedJaxpr" in t or n == "closed" for n, t in ann.items()
            )
            if takes_prop and takes_query:
                out.append((path.name, node.name, node.lineno, ast.unparse(node)))
    return out


def test_every_consumption_site_checks_the_pairing():
    """THE ANSWER TO "a check at the site can be forgotten by a new caller".

    The site list is re-derived from the source on every run, and each site
    must name `unpaired_propagation`. A new function that takes a query and a
    propagation and does not check them fails HERE, at the moment it is
    written, rather than in the audit that finds the verdict it minted.
    """
    sites = _consumption_sites()
    names = {(f, n) for f, n, _, _ in sites}
    assert names == {
        ("affine.py", "refine_propagation"),
        ("obligation.py", "slice_unknown_obligations"),
        ("solvers.py", "escalate"),
        ("solvers.py", "make_solver_verdict"),
        ("verdict.py", "make_verdict"),
    }, (
        f"the set of (query, propagation) consumption sites has changed: "
        f"{sorted(names)} — every new one needs a pairing check and a row in "
        f"this module, and every removed one needs its row struck"
    )
    missing = [
        f"{f}:{ln} {n}"
        for f, n, ln, src in sites
        if "unpaired_propagation" not in src
    ]
    assert not missing, (
        f"these sites consume a propagation against a query without checking "
        f"that the two belong together: {missing}"
    )


def test_the_library_driver_pairs_by_construction():
    """`preconditions._pipeline` is the one internal driver — `check()` and
    `contracts.check_contract()` are both it — and it is correct by
    construction rather than by the gates: it propagates the query it is
    about to judge and hands both to the same `_finish`, and the vacuity
    re-check propagates the WIDENED query before judging it. Driven end to
    end so that the five gates above are demonstrably not firing on ordinary
    work — a repair whose gates fired on the honest path would be a much
    louder defect than the one it closes.

    Driven at three DEPTHS, because `_finish` is reached at three and each
    reaches a different subset of the gated sites: interval only
    (`make_verdict`), with the affine refinement (`refine_propagation` then
    `make_verdict`), and with solver escalation (`escalate` and
    `slice_unknown_obligations` then `make_solver_verdict`)."""
    from stelling.preconditions import check

    def h():
        c = any_array((2,), jnp.float64, (100.0, 101.0))
        return assert_(c + c <= 1e9)

    def unknown_to_intervals():
        c = any_array((2,), jnp.float64, (0.0, 1.0))
        return assert_(c - c >= 0.0)

    for label, harness, kw in (
        ("interval only", h, {}),
        ("affine refinement", unknown_to_intervals, dict(refine="affine")),
        ("solver escalation", unknown_to_intervals,
         dict(solver_timeout_ms=20000)),
    ):
        v = check(harness, vacuity_mode="inputs-only", **kw)
        assert v.status == "VERIFIED", f"{label}: {v.status}"
        assert not any("unpaired propagation" in n for n in v.notes), (
            f"{label}: a pairing gate fired on the documented front door — "
            f"{v.notes}"
        )
        assert v.stamp.query_content_hash and not v.stamp.semantics.startswith(
            "not reached"
        ), f"{label}: the stamp is the unpaired-propagation one"


# -- the channel that is NOT closed -------------------------------------------


def test_the_env_channel_is_NOT_bound_and_here_is_what_that_costs():
    """DISCLOSURE, MEASURED — the third channel, named rather than implied.

    `slice_obligation` and `slice_unknown_obligations` take the interval
    environment as an argument. It is a plain mapping with no identity, and
    B11 did not give it one. What that costs, measured here: the div-by-zero
    straddle guard reads the DIVISOR'S BOX OUT OF `env`, not out of the
    query's declarations, so an `env` from a query whose divisor excludes
    zero lets a slice be emitted for a query whose divisor straddles it —
    where the honest pairing declines.

    **WHY IT IS A DISCLOSURE AND NOT A LIVE FALSE VERDICT.** The exposure
    stops at the SLICE. Both library consumers derive the environment from
    their own `closed` (`solvers.escalate` and `affine.refine_propagation`
    each call `interval_env(closed)`), so no library path can supply a
    foreign one, and both facts are asserted below off the live source. A
    caller who drives the slicer directly gets a slice, which is not a
    verdict: reaching one from here means hand-emitting, hand-solving and
    hand-building an `Escalation`, and a hand-built record is outside the
    trust model this library states (`solvers.Escalation`'s docstring: "a
    hand-built record can hold any value, and this defends an honest caller
    against an accidentally mispaired assembly").

    If a library path is ever added that takes an `env` from its caller, THIS
    test is the one that should go red, and the answer then is an identity on
    the environment rather than a wider disclosure.
    """
    import inspect

    from stelling.obligation import (
        DeclinedObligation,
        ObligationSlice,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    def div(lo, hi):
        def h():
            d = any_array((2,), jnp.float64, (lo, hi))
            n = any_array((2,), jnp.float64, (1.0, 2.0))
            q = n / d
            return assert_(q - q <= 0.0)

        return h

    safe, hazard = trace(div(1.0, 2.0)), trace(div(-1.0, 1.0))
    p_safe, p_hazard = propagate(safe), propagate(hazard)

    # the honest pairing declines: the divisor straddles zero
    (honest,) = slice_unknown_obligations(hazard, p_hazard, interval_env(hazard))
    assert isinstance(honest, DeclinedObligation)
    assert "divisor may be zero" in honest.reason

    # the same query and the same propagation, with the SAFE query's env:
    # the guard's premise is a lie and the slice is emitted
    (leaked,) = slice_unknown_obligations(
        hazard, p_hazard, interval_env(safe)
    )
    assert isinstance(leaked, ObligationSlice), (
        "the env channel has been bound after all — if that is deliberate, "
        "this test's docstring is now the stale disclosure and must be "
        "rewritten around the mechanism that closed it"
    )

    # ... and no library path can supply one, which is the whole of why the
    # row above is a disclosure. Read off the live source, not asserted.
    from stelling import affine, solvers

    for fn in (solvers.escalate, affine.refine_propagation):
        src = inspect.getsource(fn)
        assert "interval_env(closed)" in src, (
            f"{fn.__qualname__} no longer derives its environment from its "
            f"own query, so the env channel now has a library path to a "
            f"verdict and the disclosure above is no longer true"
        )
