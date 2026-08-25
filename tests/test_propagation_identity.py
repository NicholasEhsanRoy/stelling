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

**THE SITE DERIVATION IS DRIVEN BACKWARDS, because an oracle nobody has
broken on purpose is an oracle nobody has tested.** Each of the five gates is
deleted from the source in turn and the derivation must name it; a new
assembler whose query parameter carries no annotation and is not called
`closed` is injected and the derivation must see it. Driven against the
version of this file that shipped at `4bc502b`, THREE of the five deletions
left it GREEN and both injections were invisible to it entirely — the comment
above `_own_body_nodes` says which two weaknesses those exploited and which
recognition rule replaced them.

**AND WHAT IS NOT CLOSED, measured rather than asserted:**
`obligation.slice_obligation` takes FOUR caller-supplied arguments that carry
facts about the query and have no identity — `env`, `assert_position`,
`top_primitives` and `relational_assumes`. `env` relaxes a guard;
`relational_assumes` injects a false premise into the emitted script, which
is worse. Both are driven, and the boundary — that no library path forwards
one — is derived from the source rather than asserted. See
`test_the_slicer_takes_FOUR_unbound_arguments_and_TWO_of_them_are_measured`
and `test_NO_library_path_FORWARDS_a_slicer_argument_it_did_not_derive`.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp

from stelling.harness import any_array, assert_, trace
from _solver_gate import need_solver  # noqa: E402


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


def test_an_ALWAYS_EQUAL_str_SUBCLASS_PAIRS_and_that_is_the_eighth_shape():
    """THE EIGHTH IDENTITY SHAPE, DRIVEN BECAUSE IT DOES **NOT** REFUSE —
    audit 0.2.0 B11 audit, fix 6.

    The seven rows above are refusals. This one is a pairing, and it is
    pinned as a pairing so that the comparison's limit is a measured fact
    rather than a gap between what the docstring says and what the code does.

    `unpaired_propagation` establishes that the recorded identity IS a string
    (`isinstance`) and is non-empty, and then compares it with `!=`. A `str`
    SUBCLASS satisfies both tests honestly and then decides the comparison
    itself: `!=` dispatches to the subclass's `__ne__` (or to the `__ne__`
    Python derives from its `__eq__`), so an always-equal one pairs with any
    query at all. The `isinstance` test is what keeps this to `str`
    subclasses — an always-equal object of any other type is refused as "no
    query identity" — and the emptiness test is honest for the same reason
    `__eq__` is not: a plain subclass inherits `str.__len__`.

    **IT IS NOT A PRIVILEGE ESCALATION, and that is why it is disclosed
    rather than closed.** Constructing one is attacker Python in the caller's
    own process, and anyone who can define a class there can equally pass the
    query's true hash, which pairs by the honest rule. What it falsifies is
    only the unqualified reading of "two different strings are refused";
    `unpaired_propagation`'s docstring carries the qualification.

    **AND IT IS PRE-EXISTING, MEASURED RATHER THAN ASSUMED** (audit 0.2.0 B11
    re-audit, fix 5). This row is a DISCLOSURE, so it must not be readable as
    a defect the batch that wrote it introduced: run against `4bc502b`'s
    source — the commit that added the identity field, before this batch — it
    PASSES. Nothing about the comparison changed here; only its description
    did.
    """
    from stelling.propagate import unpaired_propagation

    class AlwaysEqualStr(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

        def __hash__(self):
            return 0

    class AlwaysEqualObject:
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    stub = dataclasses.make_dataclass("Stub", ["query_sha256"])
    real = "a" * 64

    assert unpaired_propagation(stub(AlwaysEqualStr("zzz")), real) is None, (
        "the comparison no longer dispatches to the recorded value's `__ne__` "
        "— if that is deliberate, this row is now a refusal and both this "
        "test and `unpaired_propagation`'s docstring must say so"
    )
    # ... and the two neighbours that bound it, so the disclosure is exact
    for value, why in (
        (AlwaysEqualObject(), "a non-`str` always-equal object: `isinstance`"),
        (AlwaysEqualStr(""), "an EMPTY always-equal subclass: `not recorded`"),
    ):
        reason = unpaired_propagation(stub(value), real)
        assert reason is not None and reason.startswith(
            "unpaired propagation: the supplied propagation records no "
        ), f"{why} must still refuse: {reason!r}"


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


@need_solver
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


# -- every site, every hostile shape, no raw escape ---------------------------


def test_EVERY_site_fails_CLOSED_on_EVERY_hostile_propagation_shape():
    """SEVEN HOSTILE SHAPES × FIVE SITES = 35, AND ALL 35 MUST FAIL CLOSED —
    audit 0.2.0 B11 audit, fixes 1 and 2 and the read above
    `make_solver_verdict`'s gate.

    The rows above drive ONE hostile propagation (a stranger of the same
    shape) at each site. This one drives the shapes a caller can hand in that
    are not propagations in any useful sense, because a gate is only worth
    what it does on the object it was built to refuse — and on `4bc502b`
    **four of these thirty-five raised a raw exception**:

        attribute read raises  refine_propagation   TypeError: replace() should
                                                    be called on dataclass
                                                    instances
        __new__ (no fields)    make_solver_verdict  AttributeError: 'Propagation'
                                                    object has no attribute
                                                    'coverage'
        __new__ (no fields)    refine_propagation   AttributeError: ... no
                                                    attribute 'obligations'
        __new__ (no fields)    slice_unknown_obl    AttributeError: ... no
                                                    attribute 'obligations'

    FAILING CLOSED IS NOT THE SAME AS DEGRADING, and the difference is the
    ordering this batch measured. `make_solver_verdict`'s escalation gates sit
    ABOVE its propagation gate on purpose, and they RAISE
    `MispairedEscalationError` — so a shape that cannot be asked whether it
    constrained an assume is refused loudly there rather than degraded quietly
    below. What is forbidden is a RAW escape: an `AttributeError` or a
    `TypeError` out of a refusal, which is the caller's crash rather than the
    library's answer, and which at
    `obligation.slice_unknown_obligations` costs every obligation's verdict
    rather than one.
    """
    from stelling.affine import refine_propagation
    from stelling.obligation import (
        DeclinedObligation,
        ObligationSlice,
        slice_unknown_obligations,
    )
    from stelling.propagate import Propagation, interval_env, propagate
    from stelling.solvers import (
        Escalation,
        MispairedEscalationError,
        SolverConfig,
        escalate,
        make_solver_verdict,
    )
    from stelling.verdict import make_verdict

    class AlwaysEqualStr(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

        def __hash__(self):
            return 0

    class RaisingRepr(str):
        def __repr__(self):
            raise RuntimeError("repr boom")

    class RaisingName(type):
        @property
        def __name__(cls):  # noqa: N805
            raise RuntimeError("name boom")

    class Undescribable(metaclass=RaisingName):
        def __repr__(self):
            raise RuntimeError("repr boom")

    class AttrBoom:
        """Everything a real `Propagation` has, except that the one field the
        gate reads raises — so `dataclasses.replace` sees a non-dataclass."""

        def __init__(self, base):
            object.__setattr__(self, "_b", base)

        def __getattr__(self, name):
            if name == "query_sha256":
                raise RuntimeError("attr boom")
            return getattr(object.__getattribute__(self, "_b"), name)

    q = trace(_two_asserts(100.0, 101.0))
    honest = propagate(q)
    cfg = SolverConfig(**CFG_KW)
    esc = escalate(q, honest, cfg)
    assert bool(
        esc.records or esc.notes or esc.ledger.spawns or esc.ledger.stamps
    ), (
        "the escalation carries no work, so `make_solver_verdict`'s "
        "escalation gates are exempt and this row would not reach the read "
        "that used to raise"
    )

    shapes = {
        "wrong hash": lambda: dataclasses.replace(
            honest, query_sha256="f" * 64
        ),
        "empty hash": lambda: dataclasses.replace(honest, query_sha256=""),
        "None hash": lambda: dataclasses.replace(honest, query_sha256=None),
        "repr raises": lambda: dataclasses.replace(
            honest, query_sha256=RaisingRepr("x")
        ),
        "type().__name__ raises": lambda: dataclasses.replace(
            honest, query_sha256=Undescribable()
        ),
        "attribute read raises": lambda: AttrBoom(honest),
        "__new__ (no fields)": lambda: Propagation.__new__(Propagation),
    }
    sites = {
        "make_verdict": lambda p: make_verdict(q, p, **VERSIONS),
        "make_solver_verdict": lambda p: make_solver_verdict(
            q, p, esc, **VERSIONS
        ),
        "escalate": lambda p: escalate(q, p, cfg),
        "refine_propagation": lambda p: refine_propagation(q, p),
        "slice_unknown_obligations": lambda p: slice_unknown_obligations(
            q, p, interval_env(q)
        ),
    }

    def safely(read, fallback):
        """The CHECKER may not raise either: the audit's own driver computed
        `any(... for n in prop.notes)` on the returned object and went down on
        the `__new__` shape before it could print the row."""
        try:
            return read()
        except BaseException:  # noqa: BLE001
            return fallback

    raw, open_ = [], []
    for label, build in shapes.items():
        for site, call in sites.items():
            try:
                got = call(build())
            except MispairedEscalationError:
                continue  # the documented, designed refusal
            except Exception as e:  # noqa: BLE001
                raw.append(f"{label} x {site}: {type(e).__name__}: {e}")
                continue
            if site in ("make_verdict", "make_solver_verdict"):
                ok = (
                    safely(lambda: got.status, None) == "UNKNOWN"
                    and safely(lambda: got.obligations, None) == ()
                    and safely(lambda: got.notes[0], "").startswith(
                        "unpaired propagation:"
                    )
                )
            elif site == "escalate":
                ok = safely(lambda: got.records, None) == () and safely(
                    lambda: got.ledger.spawns, None
                ) == 0
            elif site == "refine_propagation":
                _prop, rep = got
                ok = (
                    "unpaired propagation:" in safely(
                        lambda: rep.declined_wholly, ""
                    )
                    and rep.discharged == ()
                    and rep.violated == ()
                )
            else:
                ok = bool(got) and all(
                    isinstance(i, DeclinedObligation) for i in got
                ) and not any(isinstance(i, ObligationSlice) for i in got)
            if not ok:
                open_.append(f"{label} x {site}: {got!r}"[:200])

    assert not raw, (
        "a refusal raised a RAW exception instead of failing closed:\n  "
        + "\n  ".join(raw)
    )
    assert not open_, (
        "a hostile propagation was not refused:\n  " + "\n  ".join(open_)
    )
    assert len(shapes) * len(sites) == 35

    # -- AND EIGHT SHAPES AGAINST A HOSTILE ESCALATION (audit 0.2.0 B11
    # re-audit, fix 2). `make_solver_verdict` takes THREE arguments and the
    # thirty-five above vary one. Its escalation gates decide by COMPARING
    # the two records, so the escalation is the other operand — and the
    # operand on the LEFT of `escalation.semantics != prop_semantics`, which
    # is the side Python asks first. Two escalations get past all three
    # gates on their own account, whatever the propagation answered:
    #
    #   * an ALWAYS-EQUAL `str` semantics, which answers the pairing gate
    #     `False` whatever the propagation said, and needs no private import;
    #   * a completely EMPTY escalation, which that gate EXEMPTS by design.
    #
    # Eight propagation shapes (the seven above plus `attribute read raises`
    # aimed at `semantics` rather than at `query_sha256`) x two escalations =
    # SIXTEEN more cells, same rule: `MispairedEscalationError` or a degraded
    # verdict, never a raw escape. Measured on `bd50171`, exactly two of the
    # sixteen raised `RuntimeError` out of `make_solver_verdict`, both on that
    # eighth shape — which is why it is driven here rather than only above.
    esc_always_equal = dataclasses.replace(esc, semantics=AlwaysEqualStr("real"))
    esc_empty = Escalation(records=(), notes=(), semantics="real")
    assert (esc_always_equal.semantics != honest.semantics) is False
    assert not (
        esc_empty.records
        or esc_empty.notes
        or esc_empty.ledger.spawns
        or esc_empty.ledger.stamps
    )

    class SemBoom:
        """`attribute read raises`, aimed at the field the gates COMPARE."""

        def __init__(self, base):
            object.__setattr__(self, "_b", base)

        def __getattr__(self, name):
            if name == "semantics":
                raise RuntimeError("attr boom")
            return getattr(object.__getattribute__(self, "_b"), name)

    esc_shapes = {
        "always-equal str semantics": esc_always_equal,
        "EMPTY escalation": esc_empty,
    }
    prop_shapes = dict(shapes, **{"semantics read raises": lambda: SemBoom(honest)})
    raw2, open2 = [], []
    for elabel, escalation in esc_shapes.items():
        for label, build in prop_shapes.items():
            try:
                got = make_solver_verdict(q, build(), escalation, **VERSIONS)
            except MispairedEscalationError:
                continue
            except Exception as e:  # noqa: BLE001
                raw2.append(f"{label} x {elabel}: {type(e).__name__}: {e}")
                continue
            ok = safely(lambda: got.status, None) == "UNKNOWN" and safely(
                lambda: got.notes[0], ""
            ).startswith("unpaired propagation:")
            if not ok:
                open2.append(f"{label} x {elabel}: {got!r}"[:200])

    assert not raw2, (
        "a hostile ESCALATION let a refusal raise a RAW exception:\n  "
        + "\n  ".join(raw2)
    )
    assert not open2, (
        "a hostile (propagation, escalation) pair was not refused:\n  "
        + "\n  ".join(open2)
    )
    assert len(prop_shapes) * len(esc_shapes) == 16


@need_solver
def test_the_reads_ABOVE_make_solver_verdicts_gate_refuse_rather_than_raise():
    """A GATE CANNOT GUARD A READ THAT HAPPENS ABOVE IT.

    `make_solver_verdict` asks the propagation two things before the pairing
    gate has established anything about it — its `semantics` and its
    `coverage.constrained` — and both are inside REFUSAL conditions of the
    escalation gates, which sit above the propagation gate by a measured
    decision. So neither read may move, and both must fail CLOSED: a
    propagation that cannot be asked how many assumes it constrained cannot
    be shown to have constrained none, and stamping solver work over the
    un-assumed box against a possibly conditional claim is the exact
    unsoundness that gate exists for.

    THE SENTINEL IS AN OBJECT AND NOT THE STRING `"<unreadable>"`, and the
    last row here is why: the gates decide by COMPARING the two records, so a
    string placeholder is a value a hand-built pair can carry on both sides
    and thereby agree on.
    """
    from stelling.propagate import propagate
    from stelling.solvers import (
        MispairedEscalationError,
        SolverConfig,
        escalate,
        make_solver_verdict,
        _UNREADABLE_PROPAGATION_FIELD,
    )

    q = trace(_two_asserts(100.0, 101.0))
    honest = propagate(q)
    esc = escalate(q, honest, SolverConfig(**CFG_KW))
    assert esc.ledger.spawns > 0, "the escalation carries no solver work"
    # the honest control: the same query, the same propagation, no refusal
    assert make_solver_verdict(q, honest, esc, **VERSIONS).status in (
        "VERIFIED",
        "REFUTED",
        "UNKNOWN",
    )

    class CoverageRaises:
        def __init__(self, base):
            object.__setattr__(self, "_b", base)

        def __getattr__(self, name):
            if name == "coverage":
                raise RuntimeError("coverage boom")
            return getattr(object.__getattribute__(self, "_b"), name)

    class SemanticsRaises:
        def __init__(self, base):
            object.__setattr__(self, "_b", base)

        def __getattr__(self, name):
            if name == "semantics":
                raise RuntimeError("semantics boom")
            return getattr(object.__getattribute__(self, "_b"), name)

    with pytest.raises(MispairedEscalationError) as exc:
        make_solver_verdict(q, CoverageRaises(honest), esc, **VERSIONS)
    assert "cannot be asked how many assume(s) it constrained" in str(exc.value)
    assert "<unreadable>" not in str(exc.value), (
        "the message quotes the sentinel instead of saying what happened"
    )

    with pytest.raises(MispairedEscalationError) as exc2:
        make_solver_verdict(q, SemanticsRaises(honest), esc, **VERSIONS)
    assert "semantics='<unreadable>'" in str(exc2.value), str(exc2.value)

    # ... and what the sentinel's own design does and does not buy. The
    # comment that stood here said it "cannot be spelled from outside" —
    # three lines under an `import` of it. What is true is narrower: it is
    # not a value any SUCCESSFUL read produces, it renders as a sentence,
    # and it is TRUTHY, which is the one place its own object identity
    # carries a gate. Every other consult of it tests `is`.
    assert _UNREADABLE_PROPAGATION_FIELD != "<unreadable>"
    assert "<unreadable>" != _UNREADABLE_PROPAGATION_FIELD
    assert str(_UNREADABLE_PROPAGATION_FIELD) == "<unreadable>"
    assert bool(_UNREADABLE_PROPAGATION_FIELD) is True, (
        "an unreadable count must be TRUTHY, or the constrained gate's "
        "condition reads it as `constrained nothing` and proceeds"
    )

    class AlwaysEqualStr(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

        def __hash__(self):
            return 0

    # the comparison the sentinel is on the RIGHT of is not its to decide
    assert (AlwaysEqualStr("real") != _UNREADABLE_PROPAGATION_FIELD) is False, (
        "`_Unreadable` now defines an `__ne__`/`__eq__` that wins against a "
        "left operand with a total one — which Python does not allow, so if "
        "this passes the docstring's account of the dispatch is wrong"
    )
    # ... and handing the sentinel in from outside buys a REFUSAL, which is
    # the direction an unreadable field already goes
    with pytest.raises(MispairedEscalationError) as exc3:
        make_solver_verdict(
            q,
            dataclasses.replace(
                honest, semantics=_UNREADABLE_PROPAGATION_FIELD
            ),
            esc,
            **VERSIONS,
        )
    # (the pairing gate gets there first, because the escalation's honest
    # `"real"` is genuinely `!=` the sentinel — which is the same refusal an
    # unreadable read produces, and is the point: the spelling buys nothing)
    assert "semantics='<unreadable>'" in str(exc3.value), str(exc3.value)


@need_solver
def test_a_SENTINEL_is_recognised_by_is_and_never_by_a_COMPARISON():
    """THE THREE GATES DECIDE BY COMPARING, AND A COMPARISON IS THE OTHER
    OPERAND'S TO ANSWER — audit 0.2.0 B11 re-audit, fix 2.

    `_Unreadable` defines no `__eq__` so that every comparison falls through
    to identity. That protects its OWN side of a comparison and nothing else:
    `a != b` asks `type(a).__ne__` first, and a left operand with a total
    `__ne__` never consults the right one at all. Two pairs therefore walked
    past all three of `make_solver_verdict`'s escalation gates on `bd50171`
    and left the function through a RAW `RuntimeError` — the one shape
    `test_EVERY_site_fails_CLOSED_on_EVERY_hostile_propagation_shape`
    forbids:

    * an ALWAYS-EQUAL `str` on the ESCALATION side, needing no private
      import and no unreadable-field knowledge, against a propagation whose
      `semantics` raises: the pairing gate is answered by the escalation, the
      ieee gate's `== "ieee"` reads a sentinel as NOT ieee (the non-refusing
      direction), and the constrained gate is about a different field;
    * a completely EMPTY escalation — the nothing-to-escalate shape, which
      the pairing gate exempts BY DESIGN — against a propagation whose
      `semantics` or whose `coverage` raises.

    Both are closed by an `is` test below all three gates, so each of them
    keeps its own narrower message: this test asserts the two specific
    messages are still the ones a work-carrying escalation gets.
    """
    from stelling.propagate import propagate
    from stelling.solvers import (
        Escalation,
        MispairedEscalationError,
        SolverConfig,
        escalate,
        make_solver_verdict,
    )

    class AlwaysEqualStr(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

        def __hash__(self):
            return 0

    def raising(base, field):
        class Raises:
            def __init__(self):
                object.__setattr__(self, "_b", base)

            def __getattr__(self, name):
                if name == field:
                    raise RuntimeError(f"{field} boom")
                return getattr(object.__getattribute__(self, "_b"), name)

        return Raises()

    q = trace(_two_asserts(100.0, 101.0))
    honest = propagate(q)
    esc = escalate(q, honest, SolverConfig(**CFG_KW))
    assert esc.ledger.spawns > 0, "the escalation carries no solver work"

    # the escalation-side shape: no private import, real solver work
    esc_always_equal = dataclasses.replace(
        esc, semantics=AlwaysEqualStr("real")
    )
    assert (esc_always_equal.semantics != honest.semantics) is False, (
        "the escalation's own `__ne__` no longer decides the pairing gate's "
        "comparison, so this row measures nothing"
    )
    # the honest control: an always-equal escalation semantics alone is a
    # pairing, not a refusal, exactly as on the propagation side
    assert make_solver_verdict(
        q, honest, esc_always_equal, **VERSIONS
    ).status in ("VERIFIED", "REFUTED", "UNKNOWN")

    empty = Escalation(records=(), notes=(), semantics="real")
    assert not (
        empty.records or empty.notes or empty.ledger.spawns or empty.ledger.stamps
    ), "the empty escalation is no longer the shape the pairing gate exempts"

    for label, prop, escalation in (
        ("semantics raises x always-equal escalation",
         raising(honest, "semantics"), esc_always_equal),
        ("semantics raises x EMPTY escalation",
         raising(honest, "semantics"), empty),
        ("coverage raises x EMPTY escalation",
         raising(honest, "coverage"), empty),
    ):
        with pytest.raises(MispairedEscalationError) as exc:
            make_solver_verdict(q, prop, escalation, **VERSIONS)
        assert "cannot be asked about itself" in str(exc.value), (
            f"{label}: {exc.value}"
        )

    # and the two narrower gates keep their own words, which is why the `is`
    # test sits BELOW them rather than above
    with pytest.raises(MispairedEscalationError) as exc_c:
        make_solver_verdict(q, raising(honest, "coverage"), esc, **VERSIONS)
    assert "cannot be asked how many assume(s) it constrained" in str(exc_c.value)
    with pytest.raises(MispairedEscalationError) as exc_s:
        make_solver_verdict(q, raising(honest, "semantics"), esc, **VERSIONS)
    assert "semantics='<unreadable>'" in str(exc_s.value), str(exc_s.value)


@need_solver
def test_the_ieee_GATE_and_the_STAMP_read_ONE_ANSWER_and_not_two():
    """ONE DECISION PER VALUE — audit 0.2.0 B11 re-audit, fix 1.

    `make_solver_verdict` asks its propagation `semantics == "ieee"` in two
    anti-correlated places: the ieee mispairing gate, which REFUSES an
    escalation carrying solver work under that answer, and the stamp, which
    WRITES `ieee` under it. On `bd50171` the second was a fresh, unbound
    `propagation.semantics`, below the pairing gate and below every refusal.

    **BINDING THE ATTRIBUTE IS NOT ENOUGH AND THAT IS WHY THIS TEST DRIVES A
    COMPARISON.** The two-facedness that matters here does not live in the
    attribute — `dataclasses.replace` stores one object and every read
    returns it — it lives in that object's own `__eq__`. A `str` subclass
    whose `__eq__` answers `"ieee"` differently on successive asks defeats a
    bind and is stopped only by asking ONCE and sharing the answer. Measured
    on `bd50171` **with the fixture below and no other** — obligation 1 of
    `_two_asserts(100.0, 101.0)`, discharged "unsat per cvc5 (wheel) and z3
    (wheel)" over QF_NRA: **VERIFIED, `stamp.semantics = "ieee (IEEE-754
    binary64)"`, on a genuine cvc5+z3 unsat over ℝ**, with the comparison
    asked exactly twice. Both assertions below fail there, and the first one
    names the mechanism rather than the symptom.

    **THE ATTACK CLASS IS OUT OF SCOPE AND THIS TEST DOES NOT REOPEN IT.**
    Reaching it needs actively malicious Python in the caller's own process,
    ruled out of scope by the principal on 2026-08-18 by exactly that test.
    What is pinned is the narrower thing the code can honestly claim: the
    gate and the stamp read ONE answer, so they cannot disagree about it.
    """
    from stelling.propagate import propagate
    from stelling.solvers import SolverConfig, escalate, make_solver_verdict

    asks: list[object] = []

    class Flip(str):
        """Answers `"ieee"` False on the first ask and True on every later
        one — so a gate that asks before the stamp does gets `real` and the
        stamp gets `ieee`."""

        def __eq__(self, other):
            asks.append(other)
            if other == "ieee":
                return sum(1 for a in asks if a == "ieee") >= 2
            return str.__eq__(self, other)

        def __ne__(self, other):
            got = self.__eq__(other)
            return got if got is NotImplemented else not got

        def __hash__(self):
            return str.__hash__(self)

    q = trace(_two_asserts(100.0, 101.0))
    honest = propagate(q)
    assert honest.semantics == "real"
    esc = escalate(q, honest, SolverConfig(**CFG_KW))
    assert esc.ledger.spawns > 0, "the escalation carries no solver work"

    truth = make_solver_verdict(q, honest, esc, **VERSIONS)
    assert truth.stamp.semantics.startswith("real"), truth.stamp.semantics

    asks.clear()
    got = make_solver_verdict(
        q, dataclasses.replace(honest, semantics=Flip("real")), esc, **VERSIONS
    )
    ieee_asks = [a for a in asks if a == "ieee"]
    assert len(ieee_asks) == 1, (
        f"the propagation's `semantics` was asked `== 'ieee'` "
        f"{len(ieee_asks)} times ({asks!r}). Two anti-correlated consults of "
        f"one quantity is the defect: the ieee gate refuses under that "
        f"answer and the stamp writes it, so a value free to answer them "
        f"differently mints an ieee stamp over a real solver result"
    )
    assert got.stamp.semantics == truth.stamp.semantics, (
        f"the stamp says {got.stamp.semantics[:40]!r} where the gate that "
        f"anti-correlates with it was told {'ieee' if ieee_asks else 'real'}"
    )


def test_the_reads_ABOVE_make_solver_verdicts_gate_are_the_ONLY_reads():
    """THE SENTENCE AT THE PAIRING GATE, COUNTED OFF THE SOURCE.

    That gate's own comment justifies sitting BELOW the escalation gates by
    naming every executable read of `propagation` above it and showing each
    is inside a refusal condition. It said "`propagation.semantics` (twice)"
    **and was already wrong when it was written**: `bd50171` is the commit
    that introduced `_propagation_read` and bound `semantics` ONCE above the
    gate, and it left the count describing the code it had just replaced.
    Measured on `bd50171` by this test's own rule, the reads above the gate
    were `['coverage', 'semantics']` — one each — while `semantics` was read
    TWICE BELOW it, which that sentence did not mention at all. A count in
    prose beside code that moves is a count that has to be computed, so this
    computes it.

    The rule: inside `make_solver_verdict`, every `propagation.<member>`
    ABOVE the `unpaired_propagation(...)` call is one of the two the
    `_propagation_read` docstring accounts for, and each appears exactly
    once. Below the gate the propagation is the checked object and may be
    read freely — except for `semantics`, whose two consults are the stamp
    and the gate it anti-correlates with, and which must not reappear.
    """
    import ast
    import inspect

    from stelling import solvers

    src = inspect.getsource(solvers.make_solver_verdict)
    tree = ast.parse(src)
    (fn,) = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    gate = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and _callee_name(n.func) == "unpaired_propagation"
    ]
    assert len(gate) == 1, f"{len(gate)} pairing-gate calls, expected 1"
    gate_line = gate[0].lineno

    above, below = [], []
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "propagation"
        ):
            (above if n.lineno < gate_line else below).append(n.attr)

    assert sorted(above) == ["coverage", "semantics"], (
        f"the reads of `propagation` above the pairing gate are {sorted(above)}"
        f" — the gate's own comment names exactly `semantics` (once) and "
        f"`coverage.constrained` (once) and rests its zero-cost argument on "
        f"that list. A new read needs the comment updated and needs to be "
        f"shown to fail closed"
    )
    assert "semantics" not in below, (
        "`propagation.semantics` is read again BELOW the pairing gate. The "
        "stamp derives from that value and the ieee gate refuses on it, so "
        "the two must read one answer — see `prop_is_ieee`"
    )


# -- the check cannot be forgotten by a new caller ----------------------------
#
# THE RECOGNITION RULE, AND WHY IT IS THIS ONE (audit 0.2.0 B11 audit, fix 3).
#
# The rule that stood here was "the signature takes a parameter annotated
# `Propagation` AND a parameter annotated `ClosedJaxpr` **or named `closed`**",
# and it was defeatable twice over, and both defeats are re-driven below as
# tests rather than recorded as sentences.
#
# The name `closed` is a convention, not a fact about the argument: a public
# `judge_with(query, propagation: Propagation, ...)` that assembles a verdict
# with no pairing check at all is INVISIBLE to that rule. And the CHECK was a
# substring test over `ast.unparse(node)`, which keeps docstrings — three of the
# five sites name `unpaired_propagation` in their own docstring, so deleting
# their gate outright left the old rule reporting nothing missing. Driven here
# on this build: of the five gate deletions, the `4bc502b` rule stayed GREEN on
# `solvers.escalate`, `solvers.make_solver_verdict` and `verdict.make_verdict`,
# and it did not see either injection at all.
#
# WHAT REPLACED THEM.
#
# * The gate is recognised as an `ast.Call` node, never as text. A docstring, a
#   comment and a string literal are all `ast.Constant`s and none of them is a
#   call, so the whole class of "the name appears somewhere in the function" is
#   gone rather than narrowed.
#
# * The QUERY half is the hard half, and it is not keyed on a name. It is
#   seeded structurally — a parameter annotated `ClosedJaxpr`, or one off which
#   a member only `ir.ClosedJaxpr` has is read — and then closed under calls IN
#   BOTH DIRECTIONS: if a call fills a callee parameter that carries a query,
#   the caller's argument carries one, and vice versa. That is what sees
#   `make_verdict`'s `closed`, which is UNANNOTATED and never has an attribute
#   read off it at all — it reaches `query_identity`, whose own parameter is
#   seeded by `closed.content_hash()`. It is also what sees the injected
#   `judge_with(query, ...)`, which
#   `test_a_NEW_site_with_an_unannotated_query_is_SEEN` drives.
#
# * THE STRUCTURAL RULE WAS CHOSEN OVER THE BEHAVIOURAL ONE, and the reason is
#   that the behavioural one does not cover this batch's own sites. "Every
#   function that can return a `Verdict` or an `ObligationSlice`", read off
#   return ANNOTATIONS — which is the only way a static rule can read it —
#   misses THREE of the five, and the three are counted here rather than
#   estimated:
#
#       verdict.make_verdict                -> Verdict                    SEEN
#       solvers.make_solver_verdict         -> Verdict                    SEEN
#       solvers.escalate                    -> Escalation                 MISSED
#       affine.refine_propagation           -> tuple[Propagation,
#                                                RefinementReport]        MISSED
#       obligation.slice_unknown_obligations-> tuple[ObligationSlice |
#                                                DeclinedObligation, ...] MISSED
#
#   — including the one that WRITES decided statuses, and including the one
#   whose slices the rule is NAMED after: `slice_unknown_obligations` returns
#   a TUPLE of them, so the annotation the rule would read does not say
#   `ObligationSlice` at its head at all. Widening it to "returns a `Verdict`,
#   an `ObligationSlice`, an `Escalation` or a `Propagation`" is an
#   enumeration of return types, which a new site dodges by returning a new
#   type, and it still has to read those types off return ANNOTATIONS, which
#   are optional — the same optional-declaration weakness that made
#   name-keying necessary in the first place. Reading the propagation's fields
#   is what the misattribution IS: a
#   mispaired assembly reports one query's obligation statuses under another
#   query's name, and it has to read them to do that. So the rule keys on the
#   read.
#
# * AND THE CLAIM IS NARROWED TO WHAT IT CHECKS. The docstring used to say "a
#   new function that takes a query and a propagation and does not check them
#   fails HERE". `preconditions._finish(closed, prop)` takes both and needs no
#   check, because it reads neither — it hands both to the gated assemblers.
#   The rule now says READS, and `_finish` is accounted for as a pass-through
#   whose propagation is shown to reach nothing but a checked site
#   (`test_a_PASS_THROUGH_holds_no_judgement_and_its_propagation_is_CLOSED`).
#
# * THE RESIDUAL, NAMED — AND THE LIST THAT STOOD HERE WAS WRONG IN BOTH
#   DIRECTIONS (audit 0.2.0 B11 re-audit, fix 3). It said the derivation "does
#   not see a propagation reached through a local alias, a `getattr` with a
#   computed name, `*args`/`**kwargs`, or a parameter that a caller OUTSIDE the
#   library supplies with no annotation". Each of the first three is DRIVEN by
#   `test_the_QUERY_half_is_derived_over_BARE_PARAMETERS_only`, and the first
#   two are in fact CAUGHT: a function that holds an annotated propagation and
#   a query and reads neither in the shape the derivation follows is classified
#   a PASS-THROUGH, and `test_a_PASS_THROUGH_holds_no_judgement_and_its_
#   propagation_is_CLOSED` pins the pass-through set at exactly
#   `preconditions._finish` and requires its propagation to reach a carrying
#   parameter. An alias or a computed `getattr` reddens there, one line later
#   than a site would, but it reddens. So does a private helper reached from an
#   ungated public function: the helper itself becomes the unchecked site.
#   `*args` genuinely is invisible, and belongs with the residue below rather
#   than with the two shapes it was listed beside.
#
#   The LAST of the four is real and is closed a second way rather than
#   disclosed: `test_a_PUBLIC_function_that_reads_a_propagation_must_ANNOTATE_
#   it` fails on any public library function that reads a `Propagation` member
#   name off an unannotated parameter, so the shape can be written but not
#   shipped.
#
# * AND THE RESIDUE THAT IS NOT ON THE PROPAGATION HALF AT ALL. A function is a
#   SITE only when it holds a query, and a query is derived over BARE PARAMETER
#   NAMES: the seed is a parameter, the closure moves along `ast.Name` call
#   arguments, and a member read is credited only when it is `param.member`. So
#   a function that reads an annotated `Propagation` while its query arrives as
#   anything but a bare parameter of its own is not a site, is not a
#   pass-through, and is not an unannotated public read — it is invisible to
#   all FIVE oracle tests at once, with the propagation correctly annotated
#   throughout. `test_the_QUERY_half_is_derived_over_BARE_PARAMETERS_only`
#   drives the shapes; NONE EXISTS IN THE LIBRARY TODAY, and the census that
#   says so is `test_EVERY_Propagation_annotated_parameter_is_ACCOUNTED_for`.
#   It is named rather than closed: closing it statically means following
#   values through containers, attributes and module state, which is a
#   points-to analysis and not a test.
#
# * AND THE ONE PLACE IT IS DELIBERATELY IMPRECISE: a call is resolved by the
#   callee's SIMPLE NAME, so two library functions sharing a name are treated
#   as one. That can only ADD carriers, never remove them, which is the safe
#   direction FOR THIS RULE, whose failure mode is missing a site. **That
#   disclosure is about THIS rule and does not transfer** — see
#   `test_NO_library_path_FORWARDS_a_slicer_argument_it_did_not_derive`, where
#   the same simple-name resolution runs in the UNSAFE direction.


def _own_body_nodes(node):
    """Every node of ``node``'s own body — NOT descending into a nested
    ``def``/``class``, so each function is judged by its own statements and a
    check inside a nested helper is credited to that helper, which the walk
    visits separately."""
    import ast

    out, stack = [], list(node.body)
    while stack:
        n = stack.pop()
        out.append(n)
        for child in ast.iter_child_nodes(n):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            stack.append(child)
    return out


def _callee_name(func):
    """The simple name a call is made through: ``f(...)`` -> ``f``,
    ``mod.f(...)`` -> ``f``, anything else -> ``None``."""
    import ast

    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _member_names(sources):
    """``(every name `Propagation` answers to, the names only
    `ir.ClosedJaxpr` answers to)`` — derived, never listed, so a renamed or
    added field cannot leave a hard-coded list behind.

    THE TWO HALVES ARE DELIBERATELY DIFFERENT and each is the safe direction
    for the job it does. A query SEED must not fire on a parameter that is
    not a query, because the seed is closed under calls in both directions
    and a wrong one propagates — so it uses the names `ClosedJaxpr` answers
    to ALONE among the library's own classes. The public-annotation rule
    must not MISS a propagation read, and a miss there is a hole rather than
    noise — so it uses every name `Propagation` answers to, including the
    ones it shares (`obligations` and `notes` are also `Verdict`'s), which
    at worst asks for an annotation that was worth writing anyway.

    **THE "EVERY OTHER CLASS" HALF IS PARSED, NOT IMPORTED, AND THAT IS A
    BUG THIS TEST CAUSED.** It used to `importlib.import_module` every
    non-underscore module under `stelling/` to collect their attribute
    names. That pulls in `stelling.overflow`, which imports
    `stelling._tripwire.plugin` at module scope — and pytest cannot
    assertion-rewrite a plugin module that is already imported, so it issues
    a `PytestAssertRewriteWarning`. **Measured at the commit that removed the
    sweep: 2 failed / 3828 passed with it, 0 failed without.** The two that
    failed were `tests/test_tripwire_plugin.py`'s `-W error::UserWarning`
    rows, which were then the only ones passing that flag to a nested
    session. **An instrument that changes the process it measures is not an
    instrument**, and the two classes this needs are the two it can read off
    objects it already holds.

    **THAT SENTENCE WAS IN THE PRESENT TENSE UNTIL THE CHAIN WAS CLOSED AT ITS
    OTHER END, AND THE TENSE WAS THE WHOLE DEFECT.** It read that those two
    rows *turn* the warning into an `INTERNAL_ERROR` — a live claim about a
    file this one does not own, standing on a measurement taken once. Those
    rows no longer reach the warning at all: they hand the plugin to the
    nested session as an OBJECT, so the import that would need rewriting never
    happens. Deleting the sweep fixed one importer and left behind an
    invariant nobody could hold — *no test in this tree may import that module
    in the outer process* — which a second importer went on to break, in a
    third file, and which is why that repair is now at the nested session
    rather than here. What is recorded above is a measurement and stays true;
    what was wrong was writing it as a standing property of somebody else's
    file.
    """
    import ast
    import dataclasses as dc

    from stelling import ir
    from stelling.propagate import Propagation

    def attrs(cls):
        out = {f.name for f in dc.fields(cls)} if dc.is_dataclass(cls) else set()
        return out | {n for n in vars(cls) if not n.startswith("_")}

    others: set[str] = set()
    for src in sources.values():
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.ClassDef) or node.name in (
                "Propagation",
                "ClosedJaxpr",
            ):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    others.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    others |= {
                        t.id for t in stmt.targets if isinstance(t, ast.Name)
                    }
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    others.add(stmt.name)
    return attrs(Propagation), attrs(ir.ClosedJaxpr) - others


class _Fn:
    """One library function reduced to exactly what the recognition rule
    reads: its parameters and their annotations, the members read off each
    parameter, the calls it makes with a parameter as an argument, and whether
    it calls the pairing gate."""

    def __init__(self, file, node, public_module):
        import ast

        self.file, self.name, self.lineno = file, node.name, node.lineno
        args = node.args
        declared = (*args.posonlyargs, *args.args, *args.kwonlyargs)
        self.params = [p.arg for p in declared]
        # only these can be filled BY POSITION at a call site
        self.positional = [p.arg for p in (*args.posonlyargs, *args.args)]
        self.ann = {
            p.arg: (ast.unparse(p.annotation) if p.annotation else None)
            for p in declared
        }
        self.public = public_module and not node.name.startswith("_")
        body = _own_body_nodes(node)
        self.reads: dict[str, set[str]] = {}
        for n in body:
            if (
                isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id in self.params
            ):
                self.reads.setdefault(n.value.id, set()).add(n.attr)
        self.calls: list[tuple[str, dict]] = []
        for n in body:
            if not isinstance(n, ast.Call):
                continue
            callee = _callee_name(n.func)
            if callee is None:
                continue
            slots: dict = {}
            for i, a in enumerate(n.args):
                if isinstance(a, ast.Name) and a.id in self.params:
                    slots[i] = a.id
            for kw in n.keywords:
                if (
                    kw.arg
                    and isinstance(kw.value, ast.Name)
                    and kw.value.id in self.params
                ):
                    slots[kw.arg] = kw.value.id
            self.calls.append((callee, slots))
        # AN `ast.Call`, NOT A SUBSTRING: this is the half the old oracle lost
        self.calls_the_gate = any(
            callee == "unpaired_propagation" for callee, _ in self.calls
        )
        # a whole-object read: `dataclasses.replace` reads every field, so a
        # function that only rebuilds a propagation is still consuming one
        self.rebuilds = {
            slots[0]
            for callee, slots in self.calls
            if callee == "replace" and 0 in slots
        }
        self.key = (self.file, self.name, self.lineno)

    def __repr__(self):
        return f"{self.file}:{self.lineno} {self.name}"


class _Derivation:
    """The answer: which functions consume a propagation against a query,
    which merely pass one through, and which public functions read one without
    saying so."""

    def __init__(self, sources):
        import ast

        prop_all, query_only = _member_names(sources)
        self.functions: list[_Fn] = []
        for name, src in sources.items():
            public_module = not name.startswith("_")
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.functions.append(_Fn(name, node, public_module))
        self.by_name: dict[str, list[_Fn]] = {}
        for f in self.functions:
            self.by_name.setdefault(f.name, []).append(f)

        self.queries = self._close(
            lambda f: {
                p
                for p in f.params
                if (f.ann.get(p) and "ClosedJaxpr" in f.ann[p])
                or (f.reads.get(p, set()) & query_only)
            }
        )
        self.propagations = self._close(
            lambda f: {
                p
                for p in f.params
                if f.ann.get(p) and "Propagation" in f.ann[p]
            }
        )

        self.sites: list[_Fn] = []
        self.pass_throughs: list[_Fn] = []
        for f in self.functions:
            held = self.propagations[f.key]
            read = {p for p in held if f.reads.get(p) or p in f.rebuilds}
            if not (held and self.queries[f.key]):
                continue
            (self.sites if read else self.pass_throughs).append(f)
        self.unannotated_public = [
            (f, p, sorted(f.reads[p] & prop_all))
            for f in self.functions
            if f.public
            for p in f.params
            if p not in ("self", "cls")
            and p not in self.propagations[f.key]
            and (f.reads.get(p, set()) & prop_all)
            and f.ann.get(p) is None
        ]

    def _slot_param(self, g, slot):
        if isinstance(slot, int):
            return g.positional[slot] if slot < len(g.positional) else None
        return slot if slot in g.params else None

    def _close(self, seed):
        """``seed`` closed under calls in BOTH directions — a parameter that
        fills a carrying parameter carries, and a parameter filled BY a
        carrying argument carries. Both directions are needed and each catches
        a different shape: backwards sees `preconditions._finish`'s `prop`
        (unannotated, handed to `refine_propagation`), forwards sees a helper
        that a checked site hands its own propagation to."""
        table = {f.key: seed(f) for f in self.functions}
        changed = True
        while changed:
            changed = False
            for f in self.functions:
                for callee, slots in f.calls:
                    for g in self.by_name.get(callee, ()):
                        for slot, mine in slots.items():
                            theirs = self._slot_param(g, slot)
                            if theirs is None:
                                continue
                            if theirs in table[g.key] and mine not in table[f.key]:
                                table[f.key].add(mine)
                                changed = True
                            if mine in table[f.key] and theirs not in table[g.key]:
                                table[g.key].add(theirs)
                                changed = True
        return table

    def flows_of(self, f, param):
        """Every ``(callee, slot, lands_on_a_carrying_parameter)`` a
        pass-through's propagation reaches."""
        out = []
        for callee, slots in f.calls:
            for slot, mine in slots.items():
                if mine != param:
                    continue
                landings = [
                    self._slot_param(g, slot) in self.propagations[g.key]
                    for g in self.by_name.get(callee, ())
                ]
                out.append((callee, slot, bool(landings) and all(landings)))
        return out


def _library_sources():
    import pathlib

    import stelling

    root = pathlib.Path(stelling.__file__).parent
    return {p.name: p.read_text() for p in sorted(root.glob("*.py"))}


def _without_the_gate(sources, file, function):
    """``sources`` with the ``unpaired_propagation(...)`` CALL inside
    ``file``'s ``function`` replaced by the literal ``None``.

    THE MUTATION THE ORACLE IS DRIVEN BACKWARDS ON. Splicing is done through
    the line's UTF-8 bytes because `ast` column offsets are byte offsets and
    this library's sources are not ASCII."""
    import ast

    src = sources[file]
    target = None
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function
        ):
            for n in _own_body_nodes(node):
                if isinstance(n, ast.Call) and _callee_name(n.func) == (
                    "unpaired_propagation"
                ):
                    target = n
    assert target is not None, f"{file}:{function} does not call the gate"
    lines = src.splitlines(keepends=True)
    head = lines[target.lineno - 1].encode()[: target.col_offset].decode()
    tail = lines[target.end_lineno - 1].encode()[target.end_col_offset :].decode()
    out = dict(sources)
    out[file] = "".join(
        lines[: target.lineno - 1] + [head + "None" + tail] + lines[target.end_lineno :]
    )
    return out


THE_FIVE_SITES = {
    ("affine.py", "refine_propagation"),
    ("obligation.py", "slice_unknown_obligations"),
    ("solvers.py", "escalate"),
    ("solvers.py", "make_solver_verdict"),
    ("verdict.py", "make_verdict"),
}


def test_every_consumption_site_checks_the_pairing():
    """THE ANSWER TO "a check at the site can be forgotten by a new caller".

    The site list is re-derived from the source on every run: every library
    function that holds a query, holds a propagation, and READS the
    propagation must call `unpaired_propagation`. A new function of that shape
    that does not check them fails HERE, at the moment it is written, rather
    than in the audit that finds the verdict it minted — and the two ways the
    previous version of this test could be defeated are driven, below, as
    mutations it must redden on.
    """
    d = _Derivation(_library_sources())
    names = {(f.file, f.name) for f in d.sites}
    assert names == THE_FIVE_SITES, (
        f"the set of (query, propagation) consumption sites has changed: "
        f"{sorted(names)} — every new one needs a pairing check and a row in "
        f"this module, and every removed one needs its row struck"
    )
    missing = [
        f"{f.file}:{f.lineno} {f.name}" for f in d.sites if not f.calls_the_gate
    ]
    assert not missing, (
        f"these sites read a propagation against a query without checking "
        f"that the two belong together: {missing}"
    )


def test_the_oracle_REDDENS_when_a_gate_is_deleted():
    """THE ORACLE, DRIVEN BACKWARDS — audit 0.2.0 B11 audit, fix 3, and the
    campaign's own signature defect.

    An enumeration nobody has broken on purpose is an enumeration nobody has
    tested. Each of the five gates is deleted from the source in turn — the
    `unpaired_propagation(...)` call replaced by `None`, which is exactly the
    mutation a caller who "simplified" the gate away would make — and the
    derivation must name that site and only that site.

    IT ALSO PINS WHY THE OLD SUBSTRING ORACLE COULD NOT DO THIS. Three of the
    five sites name `unpaired_propagation` in their own DOCSTRING, so after
    the deletion the function's source still contains the name and a substring
    test over `ast.unparse(node)` — which keeps docstrings — sees nothing
    missing. Driven: the `4bc502b` oracle stayed GREEN on exactly those three
    deletions (`solvers.escalate`, `solvers.make_solver_verdict`,
    `verdict.make_verdict`) and went red on the other two. The count is
    asserted rather than described, so if those docstrings are ever rewritten
    this test says so instead of quietly becoming a weaker claim.
    """
    import ast

    sources = _library_sources()
    still_named = []
    for file, function in sorted(THE_FIVE_SITES):
        mutated = _without_the_gate(sources, file, function)
        d = _Derivation(mutated)
        unchecked = {
            (f.file, f.name) for f in d.sites if not f.calls_the_gate
        }
        assert unchecked == {(file, function)}, (
            f"deleting {file}:{function}'s gate left the derivation reporting "
            f"{sorted(unchecked)} — the oracle does not see the deletion it "
            f"exists to see"
        )
        for node in ast.walk(ast.parse(mutated[file])):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == function
                and "unpaired_propagation"
                in (ast.get_source_segment(mutated[file], node) or "")
            ):
                still_named.append(f"{file}:{function}")
    assert len(still_named) >= 3, (
        f"only {still_named} still name `unpaired_propagation` after their "
        f"gate is deleted; the substring oracle this replaced was defeatable "
        f"at three sites and that fact is what this test's docstring rests on"
    )


_INJECTED_SITE = '''
"""A new assembler, injected to drive the derivation forwards."""

from stelling.propagate import Propagation, query_identity


def judge_with(query, propagation: Propagation, *, versions):
    """Assembles a verdict from a propagation and a query.

    Its query parameter is neither annotated `ClosedJaxpr` nor named
    `closed`, which is the exact shape the name-keyed derivation could not
    see. It names `unpaired_propagation` here in its docstring and never
    calls it, which is the exact shape the substring check could not see.
    """
    # unpaired_propagation(propagation, query_identity(query))
    status = "VERIFIED" if propagation.all_discharged else "UNKNOWN"
    return (status, propagation.obligations, query_identity(query))
'''


def test_a_NEW_site_with_an_unannotated_query_is_SEEN():
    """The audit's injection, driven. A public function taking a propagation
    and a query whose parameter is called `query` and carries no annotation
    assembled a verdict with no pairing check — and the derivation that
    shipped at `4bc502b` did not see it AT ALL, because its query half was
    "annotated `ClosedJaxpr` or NAMED `closed`" and this parameter is neither.
    Driven both ways: that derivation reports the same five sites and no
    missing check on this source; this one reports six and names the new one.
    """
    sources = dict(_library_sources())
    sources["injected_assembler.py"] = _INJECTED_SITE
    d = _Derivation(sources)
    names = {(f.file, f.name) for f in d.sites}
    assert ("injected_assembler.py", "judge_with") in names, (
        f"the derivation does not see the injected site: {sorted(names)}"
    )
    unchecked = {(f.file, f.name) for f in d.sites if not f.calls_the_gate}
    assert unchecked == {("injected_assembler.py", "judge_with")}, unchecked


def test_a_PASS_THROUGH_holds_no_judgement_and_its_propagation_is_CLOSED():
    """WHY THE CLAIM IS "READS", NOT "TAKES" — and why narrowing it is safe.

    `preconditions._finish(closed, prop)` holds a query and a propagation and
    needs no pairing check: it never reads a field off either, it hands both to
    the assemblers that do check. That is the one function in the library of
    that shape, and the narrowing is not taken on trust — every call it makes
    with the propagation is shown to land on a parameter the derivation also
    carries as a propagation, so the object cannot reach a read the derivation
    has not looked at.
    """
    d = _Derivation(_library_sources())
    holders = {(f.file, f.name) for f in d.pass_throughs}
    assert holders == {("preconditions.py", "_finish")}, (
        f"the set of functions holding a query and a propagation without "
        f"reading either has changed: {sorted(holders)} — each needs its "
        f"propagation shown to reach nothing but a checked site"
    )
    (f,) = d.pass_throughs
    (param,) = d.propagations[f.key]
    flows = d.flows_of(f, param)
    assert flows, f"{f}: the propagation is held and never passed anywhere"
    leaks = [(c, s) for c, s, ok in flows if not ok]
    assert not leaks, (
        f"{f}: the propagation reaches {leaks}, which the derivation does not "
        f"carry as a propagation — so it may be read somewhere unchecked"
    )
    assert {c for c, _, _ in flows} == {
        "make_verdict",
        "make_solver_verdict",
        "escalate",
        "refine_propagation",
    }, sorted({c for c, _, _ in flows})


_INJECTED_UNANNOTATED = '''
def summarise(closed, propagation):
    """Public, holds a query, reads the propagation's judgements, and says
    nothing in its signature about what the second argument is."""
    return (closed.content_hash(), len(propagation.obligations))
'''


def test_a_PUBLIC_function_that_reads_a_propagation_must_ANNOTATE_it():
    """The residual the static derivation cannot see, closed rather than
    disclosed.

    A propagation reaches a read either from somewhere the derivation can
    follow — an annotation, or a call it can trace in either direction — or
    from a caller outside the library. The second is invisible to any static
    rule, so it is forbidden instead: a PUBLIC library function that reads a
    name `Propagation` answers to, off a parameter it has not annotated, fails
    here. With the annotation the function becomes a site and must gate; the
    shape can be written but not shipped.
    """
    d = _Derivation(_library_sources())
    assert not d.unannotated_public, [
        f"{f.file}:{f.lineno} {f.name}({p}) reads {m}"
        for f, p, m in d.unannotated_public
    ]
    # driven backwards: the rule catches the shape it forbids
    sources = dict(_library_sources())
    sources["injected_summary.py"] = _INJECTED_UNANNOTATED
    hits = {
        (f.file, f.name, p) for f, p, _ in _Derivation(sources).unannotated_public
    }
    assert hits == {("injected_summary.py", "summarise", "propagation")}, hits


# Each shape holds a correctly annotated `Propagation`, reads its judgements,
# and pairs nothing — and differs from the injections above only in HOW ITS
# QUERY ARRIVES. `# unpaired_propagation(...)` appears in none of them.
_QUERY_HIDDEN = {
    "inside a container": '''
from stelling.propagate import Propagation, query_identity
def judge_bundle(bundle, propagation: Propagation):
    """The query is element 0 of a tuple the caller passes."""
    return (query_identity(bundle[0]), propagation.obligations)
''',
    "on a context object": '''
from stelling.propagate import Propagation, query_identity
def judge_ctx(ctx, propagation: Propagation):
    """The query is an attribute of a context object."""
    return (query_identity(ctx.closed), propagation.obligations)
''',
    "as a dict value": '''
from stelling.propagate import Propagation, query_identity
def judge_dict(request, propagation: Propagation):
    """The query is a dict value."""
    return (query_identity(request["closed"]), propagation.obligations)
''',
    "from module state": '''
from stelling.propagate import Propagation, query_identity
REGISTRY = {}
def judge_registered(key, propagation: Propagation):
    """The query comes from module state; no parameter carries it."""
    q = REGISTRY[key]
    return (query_identity(q), propagation.obligations)
''',
    "through *args": '''
from stelling.propagate import Propagation, query_identity
def judge_star(*args, propagation: Propagation):
    """The query arrives packed into `*args`."""
    return (query_identity(args[0]), propagation.obligations)
''',
    "on `self`": '''
from stelling.propagate import Propagation, query_identity
class Judge:
    """The query and the propagation are fields; the read is on `self`."""
    def __init__(self, closed, propagation: Propagation):
        self.closed = closed
        self.propagation = propagation
    def judge(self):
        return (query_identity(self.closed), self.propagation.obligations)
''',
    "annotated under an alias": '''
from stelling.propagate import Propagation as P, query_identity
def judge_alias(closed, propagation: P):
    """The query is a bare parameter; the PROPAGATION's annotation is an
    import alias, which the substring rule does not spell."""
    return (query_identity(closed), propagation.obligations)
''',
}

# ... and the control: the SAME body with the query taken from a parameter the
# closure can follow. `decode` hands `spec` to `ClosedJaxpr.from_dict`, whose
# parameter the derivation carries, so the backwards closure reaches `spec`
# and the site is SEEN. "The query comes from a call" is therefore not the
# residue by itself — what hides a site is that NO parameter carries the query.
_QUERY_VIA_A_FOLLOWED_CALL = '''
from stelling.propagate import Propagation, query_identity
def judge_decoded(spec, propagation: Propagation):
    q = decode(spec)
    return (query_identity(q), propagation.obligations)
def decode(spec):
    from stelling import ir
    return ir.ClosedJaxpr.from_dict(spec)
'''


def test_the_QUERY_half_is_derived_over_BARE_PARAMETERS_only():
    """THE SECOND RESIDUE, AND IT IS ON THE QUERY HALF — audit 0.2.0 B11
    re-audit, fix 3.

    `SOUNDNESS.md` said "**the one** residue a static rule cannot see". There
    are two. The first is the disclosed one: a propagation supplied by a
    caller outside the library with no annotation, which
    `test_a_PUBLIC_function_that_reads_a_propagation_must_ANNOTATE_it` forbids
    rather than discloses. The second is the mirror of it on the other
    argument, and nothing forbids it.

    A function is a SITE only when the derivation carries a QUERY for it, and
    the query half is seeded and closed over BARE PARAMETER NAMES: the seed is
    a parameter annotated `ClosedJaxpr` or one off which a query-only member
    is read as `param.member`, and the closure walks call arguments that are
    `ast.Name`s naming parameters. So a function whose query arrives as
    anything else holds no query as far as the derivation is concerned — and
    a function with no query is neither a site, nor a pass-through, nor an
    unannotated public read. **It is invisible to all FIVE oracle tests at
    once, with its propagation correctly annotated the whole way** —
    `test_every_consumption_site_checks_the_pairing` (not in `sites`),
    `test_the_oracle_REDDENS_when_a_gate_is_deleted` and
    `test_a_NEW_site_with_an_unannotated_query_is_SEEN` (not in `unchecked`),
    `test_a_PASS_THROUGH_holds_no_judgement_and_its_propagation_is_CLOSED`
    (not a pass-through either, for want of a query) and
    `test_a_PUBLIC_function_that_reads_a_propagation_must_ANNOTATE_it` (its
    propagation IS annotated). The three derivation sets those five read are
    the three this test checks.

    Seven shapes are driven. **NONE OF THEM EXISTS IN THE LIBRARY** — see
    `test_EVERY_Propagation_annotated_parameter_is_ACCOUNTED_for` — which is
    why this is a disclosure and not a repair. Closing it statically means
    following values through containers, attributes and module state, which
    is a points-to analysis and not a test; the honest move is to name it.

    THE CONTROL MATTERS AS MUCH AS THE SHAPES. "The query comes from a call"
    is NOT by itself a residue: when the call's arguments trace back to a
    parameter, the closure runs BACKWARDS through the callee and the site is
    seen. The last row drives exactly that and asserts it is CAUGHT, so this
    test cannot be read as claiming more than it measures.
    """
    base = _library_sources()
    invisible, seen = [], []
    for label, src in _QUERY_HIDDEN.items():
        sources = dict(base)
        sources["injected_hidden.py"] = src
        d = _Derivation(sources)
        touched = (
            {(f.file, f.name) for f in d.sites}
            | {(f.file, f.name) for f in d.pass_throughs}
            | {(f.file, f.name) for f, _, _ in d.unannotated_public}
        )
        (seen if any(n[0] == "injected_hidden.py" for n in touched) else invisible).append(
            label
        )

    assert sorted(invisible) == sorted(_QUERY_HIDDEN), (
        f"these shapes are now VISIBLE to the derivation: {sorted(seen)}. That "
        f"is a strictly better oracle and the disclosure above must be "
        f"narrowed to match — this test is the disclosure, so it says so "
        f"rather than passing quietly"
    )

    # ... and every one of them would be an unchecked SITE if it were seen,
    # which is what makes the residue worth naming rather than a curiosity
    sources = dict(base)
    sources["injected_hidden.py"] = _QUERY_VIA_A_FOLLOWED_CALL
    d = _Derivation(sources)
    unchecked = {(f.file, f.name) for f in d.sites if not f.calls_the_gate}
    assert unchecked == {("injected_hidden.py", "judge_decoded")}, (
        f"the control shape is no longer caught: {sorted(unchecked)}. A query "
        f"reached through a call whose ARGUMENT traces back to a parameter is "
        f"inside the closure, and the residue above is about the query "
        f"reaching a body with no parameter carrying it at all"
    )


def test_EVERY_Propagation_annotated_parameter_is_ACCOUNTED_for():
    """THE CENSUS THAT MAKES THE RESIDUE ABOVE A DISCLOSURE AND NOT A HOLE.

    The residue is invisible to the derivation, so the derivation cannot be
    the thing that says the library is free of it. This counts the library's
    `Propagation`-annotated parameters directly off the source — a rule the
    residue does NOT defeat, because hiding a query does not hide the
    propagation's own annotation — and requires every function holding one to
    be one of the accounted-for shapes.

    Two functions hold an annotated propagation and are NOT sites, and each
    is accounted for by what it does with it:

    * `affine._decline_all` — private, builds a wholly-declined report from a
      propagation it never pairs, and is reached only from
      `refine_propagation` AFTER that function's own gate;
    * `verdict.top_despite_coverage_note` — takes a propagation and no query
      at all, so there is no pairing to check.

    `preconditions._finish` holds one without annotating it and is the
    pass-through the derivation accounts for separately.

    IT RESOLVES IMPORT ALIASES, WHERE THE DERIVATION DOES NOT. The site rule
    recognises the annotation by the substring `Propagation`, so
    `from stelling.propagate import Propagation as P` followed by `p: P` is
    invisible to it — one of the seven shapes driven next door. A census that
    shared that blindness could not be the thing that says the shape is
    absent, so this one reads each module's `ImportFrom` and counts the local
    names `Propagation` was bound to as well.
    """
    import ast
    import pathlib

    import stelling

    root = pathlib.Path(stelling.__file__).parent
    holders = set()
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"Propagation"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names |= {
                    a.asname or a.name
                    for a in node.names
                    if a.name == "Propagation"
                }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if a.annotation is None:
                    continue
                mentioned = {
                    x.id for x in ast.walk(a.annotation) if isinstance(x, ast.Name)
                } | {
                    x.value
                    for x in ast.walk(a.annotation)
                    if isinstance(x, ast.Constant) and isinstance(x.value, str)
                }
                if mentioned & names:
                    holders.add((path.name, node.name))

    accounted = THE_FIVE_SITES | {
        ("affine.py", "_decline_all"),
        ("verdict.py", "top_despite_coverage_note"),
    }
    assert holders == accounted, (
        f"the set of library functions taking a `Propagation`-ANNOTATED "
        f"parameter has changed: {sorted(holders)}. A new one is either a "
        f"site (and needs a gate and a row in `THE_FIVE_SITES`) or needs "
        f"saying here why it holds a propagation and pairs nothing — the "
        f"derivation cannot answer that question for a function whose query "
        f"does not arrive as a bare parameter, which is the residue "
        f"`test_the_QUERY_half_is_derived_over_BARE_PARAMETERS_only` drives"
    )


@need_solver
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
#
# FOUR ARGUMENTS, NOT ONE (audit 0.2.0 B11 audit, fix 4). The disclosure that
# stood here named `env` and only `env`. `obligation.slice_obligation` is
# public, is in `__all__`, and takes FOUR caller-supplied arguments that the
# propagation would otherwise have derived — `env`, `assert_position`,
# `top_primitives` and `relational_assumes`. None of them is bound to the
# query, and none of them is visible to either derivation above, because they
# arrive UNPACKED INTO SCALARS rather than as a `Propagation`: there is no
# object left whose identity anything could compare.
#
# `env` RELAXES A GUARD. `relational_assumes` INJECTS A FALSE PREMISE, and is
# the worse of the two: it puts an axiom into the emitted script, so the solver
# proves a different theorem. Both are measured below rather than described.
#
# It is still a DISCLOSURE and not a live false verdict, and the boundary is
# derived rather than asserted: every library call into either slicer supplies
# all four from the query it is judging, and the ONE argument any library path
# forwards from its own caller is `slice_unknown_obligations`' `env`, passed
# through to `slice_obligation` — which is the declared channel itself. That is
# computed off the source in
# `test_NO_library_path_FORWARDS_a_slicer_argument_it_did_not_derive`, so a new
# path that takes any of the four from its caller reddens it.


def _slicer_channels():
    """The caller-supplied arguments of `slice_obligation` that the
    propagation would otherwise have derived — read off the SIGNATURE, so a
    fifth one added later appears here instead of being missed by a list."""
    import inspect

    from stelling.obligation import slice_obligation

    params = list(inspect.signature(slice_obligation).parameters)
    # `closed` is the query itself and `index` selects within it; everything
    # after them is a fact about the query that the caller states instead
    return tuple(p for p in params if p not in ("closed", "index"))


def test_the_slicer_takes_FOUR_unbound_arguments_and_TWO_of_them_are_measured():
    """DISCLOSURE, MEASURED — the channel that is not closed, named in full.

    `slice_obligation` takes four caller-supplied arguments that carry facts
    about the query, and not one of them has an identity. Two are driven here.

    **`env` RELAXES A GUARD.** The div-by-zero straddle guard reads the
    DIVISOR'S BOX OUT OF `env`, not out of the query's declarations, so an
    `env` from a query whose divisor excludes zero lets a slice be emitted for
    a query whose divisor straddles it — where the honest pairing declines.

    **`relational_assumes` INJECTS A FALSE PREMISE, and is worse.** `smt.emit`
    reads its axioms off `sl.assumes` and from nowhere else, and `_Slicer`
    fills `sl.assumes` from the `relational_assumes` it was constructed with.
    A tuple taken from another query's propagation therefore becomes an
    `(assert ...)` line in the emitted script: the solver is asked a different
    question, and answers it correctly. Measured below on a query whose honest
    verdict is REFUTED, with the ground truth in exact rationals and in
    concrete jax beside it.

    **WHY IT IS A DISCLOSURE AND NOT A LIVE FALSE VERDICT.** The exposure stops
    at the SLICE, and that boundary is derived from the source in the test
    below rather than asserted here. A caller who drives the slicer directly
    gets a slice, which is not a verdict: reaching one from here means
    hand-emitting, hand-solving and hand-building an `Escalation`, and a
    hand-built record is outside the trust model this library states
    (`solvers.Escalation`'s docstring: "a hand-built record can hold any value,
    and this defends an honest caller against an accidentally mispaired
    assembly").

    If a library path is ever added that takes one of these from its caller,
    the test below is the one that goes red, and the answer then is an identity
    on the argument rather than a wider disclosure.
    """
    from fractions import Fraction

    from stelling import smt
    from stelling.harness import assume
    from stelling.obligation import (
        DeclinedObligation,
        ObligationSlice,
        slice_obligation,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    assert _slicer_channels() == (
        "env",
        "assert_position",
        "top_primitives",
        "relational_assumes",
    ), (
        f"`slice_obligation` no longer takes exactly the four caller-supplied "
        f"arguments this disclosure names: {_slicer_channels()} — a new one "
        f"needs its cost measured here, or an identity"
    )

    # -- (1) `env`: the guard's premise made false --------------------------
    def div(lo, hi):
        def h():
            d = any_array((2,), jnp.float64, (lo, hi))
            n = any_array((2,), jnp.float64, (1.0, 2.0))
            q = n / d
            return assert_(q - q <= 0.0)

        return h

    safe, hazard = trace(div(1.0, 2.0)), trace(div(-1.0, 1.0))
    p_hazard = propagate(hazard)

    (honest,) = slice_unknown_obligations(hazard, p_hazard, interval_env(hazard))
    assert isinstance(honest, DeclinedObligation)
    assert "divisor may be zero" in honest.reason

    (leaked,) = slice_unknown_obligations(hazard, p_hazard, interval_env(safe))
    assert isinstance(leaked, ObligationSlice), (
        "the env channel has been bound after all — if that is deliberate, "
        "this test's docstring is now the stale disclosure and must be "
        "rewritten around the mechanism that closed it"
    )

    # -- (2) `relational_assumes`: a false axiom in the emitted script ------
    def cmp_query(with_assume):
        def h():
            x = any_array((2,), jnp.float64, (-5.0, 5.0))
            y = any_array((2,), jnp.float64, (-5.0, 5.0))
            if with_assume:
                assume(x <= y)
            return assert_(x - y <= 0.0)

        return h

    a, b = trace(cmp_query(True)), trace(cmp_query(False))
    p_a, p_b = propagate(a), propagate(b)
    assert len(p_a.relational_assumes) == 1 and p_b.relational_assumes == ()

    # the ground truth about B, before the tool is consulted: x=1, y=0 lie in
    # the declared box, and x - y = 1 is not <= 0, so B's honest verdict is
    # REFUTED and the borrowed axiom `x <= y` is FALSE of B
    assert Fraction(1) - Fraction(0) > 0
    xv = jnp.full((2,), 1.0, jnp.float64)
    yv = jnp.zeros((2,), jnp.float64)
    assert not bool(jnp.all(xv - yv <= 0.0))

    def script_for(relational_assumes):
        sl = slice_obligation(
            b, 0, interval_env(b), relational_assumes=relational_assumes
        )
        assert isinstance(sl, ObligationSlice), sl
        return sl, smt.emit(sl, "z3", 20_000)

    sl_honest, honest_script = script_for(())
    sl_foreign, foreign_script = script_for(p_a.relational_assumes)
    assert len(sl_honest.assumes) == 0 and len(sl_foreign.assumes) == 1

    def axioms(script):
        return [
            ln for ln in script.text.splitlines() if ln.startswith("(assert ")
        ]

    extra = [ln for ln in axioms(foreign_script) if ln not in axioms(honest_script)]
    assert extra == ["(assert (<= x0_0 x1_0))", "(assert (<= x0_1 x1_1))"], extra

    from stelling import _optional
    from stelling.solvers import SolverConfig, _backends_for

    backends, _missing = _backends_for(
        SolverConfig(timeout_ms=20_000, only=("z3",))
    )
    if _optional.available("z3") and backends:
        answers = tuple(
            backends[0].transport_fn(s.text, 20.0).answer
            for s in (honest_script, foreign_script)
        )
        assert answers == ("sat", "unsat"), (
            f"{answers}: the honest script must be sat (a witness, hence "
            f"REFUTED) and the one carrying the foreign axiom unsat (hence "
            f"DISCHARGED) — if that has changed, this measurement is stale"
        )


def test_NO_library_path_FORWARDS_a_slicer_argument_it_did_not_derive():
    """THE BOUNDARY OF THE DISCLOSURE ABOVE, DERIVED FROM THE SOURCE.

    The version of this that stood here read `inspect.getsource` of TWO
    functions and substring-matched `"interval_env(closed)"`. It was short by
    one supplier (`verdict._bar_scope`, which also calls `interval_env(closed)`
    and then `slice_obligation`), it said nothing about the other three
    channels, and — the part that matters — a NEW path taking an `env` from its
    caller would not have reddened it at all, because it only ever looked at
    the two functions it already knew about.

    This looks at every call into either slicer instead. For each, each of the
    four channels is resolved back through the enclosing function's local
    assignments; an argument that resolves to a PARAMETER of that function
    which the pairing gate has not bound is a FORWARD, and every other one is
    derived from the query being judged. The library is allowed exactly one
    forward — `slice_unknown_obligations` handing its own `env` parameter
    through to `slice_obligation`, which is the declared channel itself.

    **AND THE SIMPLE-NAME RESOLUTION RUNS THE UNSAFE DIRECTION HERE** — audit
    0.2.0 B11 re-audit, fix 5. A call is recognised by `_callee_name`, the
    callee's SIMPLE NAME, exactly as in the site derivation above. There the
    imprecision "can only ADD carriers, never remove them, which is the safe
    direction"; **that disclosure is about that rule and does not transfer to
    this one.** This rule ENUMERATES calls into the slicers, so a call it
    cannot name is a call it does not see — `fn = slice_obligation` followed
    by `fn(closed, index, env)` has `_callee_name` `"fn"`, contributes no row
    to `calls`, and forwards nothing this test can find. Missing one here
    loses a forward instead of gaining a spurious carrier, and the count
    below stays green while the boundary has moved. The last block drives
    exactly that pair — the direct call seen, the aliased one not — so the
    limit is measured rather than argued. It is disclosed rather than closed
    for the reason the whole channel is: the answer to an unbound argument is
    an identity on the argument, not a wider static rule.
    """
    import ast

    sources = _library_sources()
    d = _Derivation(sources)
    channels = set(_slicer_channels())
    slicers = ("slice_obligation", "slice_unknown_obligations")
    # `env` is the third POSITIONAL parameter of both slicers
    positional = {2: "env"}

    def chain_of(tree, node):
        out = [
            f
            for f in ast.walk(tree)
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
            and f.lineno <= node.lineno <= (f.end_lineno or f.lineno)
        ]
        return sorted(out, key=lambda f: f.lineno)

    def assigned_in(fn):
        out: dict[str, list] = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out.setdefault(t.id, []).append(n.value)
            elif (
                isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and n.value is not None
            ):
                out.setdefault(n.target.id, []).append(n.value)
        return out

    def roots(expr, assigns, depth=0):
        names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
        if depth > 4:
            return names
        out = set()
        for name in names:
            rhs = assigns.get(name)
            if rhs is not None and len(rhs) == 1:
                out |= roots(rhs[0], assigns, depth + 1)
            else:
                out.add(name)
        return out

    calls, forwards = [], []
    for fname, src in sources.items():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _callee_name(node.func) not in slicers:
                continue
            chain = chain_of(tree, node)
            outer = chain[0]
            key = (fname, outer.name, outer.lineno)
            params = set(_Fn(fname, outer, True).params)
            bound = d.queries[key] | d.propagations[key]
            assigns: dict[str, list] = {}
            for f in chain:
                for k, v in assigned_in(f).items():
                    assigns.setdefault(k, []).extend(v)
            supplied = {}
            for i, arg in enumerate(node.args):
                if i in positional:
                    supplied[positional[i]] = arg
            for kw in node.keywords:
                if kw.arg in channels:
                    supplied[kw.arg] = kw.value
            calls.append((fname, node.lineno, outer.name, sorted(supplied)))
            for channel, expr in supplied.items():
                if (roots(expr, assigns) & params) - bound:
                    forwards.append((fname, outer.name, channel))

    # the load-bearing assertion first, so a new forwarding path names itself
    # rather than being reported as a changed count
    assert sorted(forwards) == [
        ("obligation.py", "slice_unknown_obligations", "env")
    ], (
        f"{sorted(forwards)}: a library path now supplies a slicer argument it "
        f"took from ITS OWN caller. The exposure no longer stops at the slice, "
        f"the disclosure above is no longer true, and the answer is an "
        f"identity on that argument rather than a wider disclosure"
    )
    assert len(calls) == 4, (
        f"the set of library calls into the slicers has changed: {calls} — "
        f"each new one needs its four channels accounted for"
    )

    # -- the limit of the enumeration, driven (audit 0.2.0 B11 re-audit,
    # fix 5). Two calls with identical semantics; only one has a name this
    # rule can resolve, and the count above is what a missed one would not
    # move.
    probe = ast.parse(
        "def direct(closed, index, env):\n"
        "    return slice_obligation(closed, index, env)\n"
        "def aliased(closed, index, env):\n"
        "    fn = slice_obligation\n"
        "    return fn(closed, index, env)\n"
    )
    resolved = [
        n.func.id if isinstance(n.func, ast.Name) else _callee_name(n.func)
        for n in ast.walk(probe)
        if isinstance(n, ast.Call)
    ]
    assert [r for r in resolved if r in slicers] == ["slice_obligation"], (
        f"both probe calls now resolve: {resolved}. If the resolution has "
        f"been widened beyond the callee's simple name, the disclosure above "
        f"is no longer this rule's limit and must be rewritten"
    )
    assert "fn" in resolved, (
        "the aliased call is no longer resolved to `fn`, so this probe is no "
        "longer measuring the shape the disclosure names"
    )
