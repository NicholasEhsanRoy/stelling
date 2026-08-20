# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE ROUTES PAST THE CANONICALIZATION DOOR, DRIVEN RATHER THAN ENUMERATED.

`ir`'s door decides what a document may store. `SOUNDNESS.md` and
`tests/test_aval_lie_both_faces.py` both disclosed one residue behind it
and both said the same thing about how it is reached: *"only an
`object.__setattr__` past the frozen dataclass reaches it."*

**THAT SENTENCE IS FALSE, AND THIS FILE IS WHY IT IS NOT REPLACED BY
ANOTHER SENTENCE** — audit 0.2.0 B6 audit 8. Three routes reach a stored
or carried value without the caller calling `object.__setattr__` at all,
and each of them is DRIVEN below rather than described:

* **the root is never canonicalized.** `ir._canonicalise` runs on an
  object's FIELDS. Nothing calls `ir._canonical` on the object a caller
  hands `propagate`, `escalate` or `make_solver_verdict`, so a
  `ClosedJaxpr` SUBCLASS whose `jaxpr` is a `property` is read once by its
  own `__post_init__` and is free to answer differently every time after.
  **This one is OUT OF SCOPE BY DECISION as of 2026-08-18, not an open
  item** — see `test_the_ROOT_of_a_query_is_never_canonicalized` for the
  principal's words and what the decision does and does not cover. The test
  below stays exactly as it is: the route is still true of the tree, and a
  decision to leave something open is worth nothing if the thing left open
  stops being measured.
* **the install is not guaranteed to install.** `_canonicalise` installs
  the canonical twin with `object.__setattr__`, which resolves the field
  NAME — and a name that resolves to a class-level DATA DESCRIPTOR goes to
  that descriptor's setter, never to the instance `__dict__`. The door
  reads, decides, installs, and the install is swallowed.
* **`__class__` assignment.** `s.__class__ = ir.Var` makes `type(s)` be
  `ir.Var` genuinely — no metaclass, no property, nothing for the door to
  detect — so the `id()`-keyed index arm returns the object unchanged,
  with `Var.__post_init__` never having run on it.

**ALL THREE ARE PRE-EXISTING**, not regressions of this batch: measured on
`dff95fc` and on `main` at `198a2b5` as well as here. **All three need
attacker Python** — a subclass definition or a `__class__` assignment in
the caller's own process — and none is reachable from a DOCUMENT:
`ClosedJaxpr.from_dict` returns an exact `ClosedJaxpr` (pinned below).
That is what makes them a disclosure rather than a merge block, and it is
also why they are pinned: a residue that only a sentence records is a
residue that rots.

Two further claims about the door are computed here for the same reason,
both from audit 0.2.0 B6 audit 8:

* `ir._register_stored_type` says what its frozen-dataclass check does and
  does not establish. It does not establish single-valuedness, and the
  class it does not catch is the one its own prose used to name.
* Which stored types a metaclass can forge an MRO entry for, and why
  forging one buys nothing at this door.

`tests/test_ir_canonicalization.py` holds the door's positive behaviour;
this file holds the edges the door does not reach and says so in the one
form that cannot go stale.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from stelling import interval, ir
from stelling.propagate import propagate


_REPO = Path(__file__).resolve().parent.parent


def av(shape=(), dtype="float64"):
    return ir.Aval(kind="ShapedArray", shape=tuple(shape), dtype=dtype)


# The routes, as data. `test_the_DISCLOSURES_name_exactly_these_routes`
# reads the shipped prose against this list, so an enumeration in prose is
# checkable against the enumeration that is driven — which is the whole
# repair. A route added here without its disclosure fails; a disclosure
# that names a route nobody drives has nothing to hold it.
ROUTES: tuple[tuple[str, str], ...] = (
    ("root", "the root is never canonicalized"),
    ("install", "the install is not guaranteed to install"),
    ("class_assignment", "`__class__` assignment"),
)


# ---------------------------------------------------------------------------
# route 1 — the ROOT is never canonicalized
# ---------------------------------------------------------------------------

def test_the_ROOT_of_a_query_is_never_canonicalized():
    """A `ClosedJaxpr` SUBCLASS with a `jaxpr` PROPERTY answers `propagate`
    and `content_hash` with DIFFERENT DOCUMENTS — audit 0.2.0 B6 audit 8.

    No `object.__setattr__`, no `from_dict`, no metaclass: every object
    here is built through a public `stelling.ir` constructor and the only
    unusual thing about the query is that one of its fields is a
    descriptor. `_canonicalise` cannot repair it even in principle —
    it runs on the FIELDS of an object and nothing runs it on the object
    itself — and `propagate`, `escalate` and `make_solver_verdict` all
    take the root as given.

    The consequence, driven end to end in the audit's own reproducer
    (`b6audit8/f1_false_verified_property.py`), is a **VERIFIED** verdict
    stamped with the content hash of a document that a concrete `jnp.sum`
    falsifies. This test holds only the MECHANISM — that the root answers
    its constructor and every later reader differently — so that it needs
    no solver and no jax. What that mechanism costs is the test below,
    where `propagate` and `content_hash` are shown reading two different
    documents out of one object.

    **STATUS: OUT OF SCOPE BY DECISION, 2026-08-18 — not an open item.**
    Ruled out of scope by the principal, in these words: *"It can be
    addressed with proper CI security on projects that need it. It requires
    actively malicious python to actually happen."* B6's door covers an
    object's FIELDS and was never scoped to its root, so this is a boundary
    of that repair rather than a gap in it.

    NOTHING ABOUT THIS TEST CHANGES, and that is the point of recording the
    decision here rather than deleting the row. The route is still true of
    the tree; the reproducer still reaches VERIFIED on a claim a concrete
    `jnp.sum` falsifies; this test still drives the mechanism. What changes
    is only its STATUS, so that a reader meeting it later records a decision
    that was taken instead of reopening an oversight that was missed.

    WHAT THE DECISION DOES NOT COVER: reachability from a DOCUMENT.
    `ClosedJaxpr.from_dict` returns an exact `ClosedJaxpr`, pinned by
    `test_NONE_of_the_routes_is_reachable_from_a_DOCUMENT`, and the decision
    rests on that. A route to this behaviour that needs no attacker Python
    would be a NEW finding, not an instance of a closed decision.
    """
    one = ir.Jaxpr(constvars=(), invars=(ir.Var(1, av((1,))),), outvars=(),
                   eqns=(), effects=())
    two = ir.Jaxpr(constvars=(), invars=(ir.Var(1, av((2,))),), outvars=(),
                   eqns=(), effects=())

    class RootProperty(ir.ClosedJaxpr):
        reads = 0

        @property
        def jaxpr(self):
            RootProperty.reads += 1
            # `__post_init__` gets `one`; everyone after it gets `two`
            return one if RootProperty.reads <= post_init_reads else two

        @jaxpr.setter
        def jaxpr(self, value):
            pass

    # how many reads the constructor itself makes is a fact about the
    # tree, not a constant, so it is MEASURED and then used
    post_init_reads = 10 ** 9
    probe = RootProperty(jaxpr=one, consts=())
    post_init_reads = RootProperty.reads
    assert post_init_reads >= 1, "the constructor read `jaxpr` not at all"
    del probe

    RootProperty.reads = 0
    q = RootProperty(jaxpr=one, consts=())
    assert isinstance(q, ir.ClosedJaxpr)
    assert q.jaxpr is two, (
        "the root's `jaxpr` property answered the constructor and every "
        "later reader identically; this route has closed and the "
        "disclosures that name it must be revisited"
    )
    assert q.jaxpr is two and q.jaxpr is two   # and it stays two

    # the door never saw the root at all
    assert type(q) is not ir.ClosedJaxpr
    with pytest.raises(ir._NotCanonical):
        # and it would have refused it: a SUBCLASS of a stored type is not
        # a stored type, which is why the root being skipped is the whole
        # of the route
        ir._canonical(q)


def test_the_root_route_gives_two_readers_two_DIFFERENT_documents():
    """The falsification, without a solver: `propagate` and `content_hash`
    disagree about which document they were given.

    `propagate` walks the jaxpr it reads; `content_hash` hashes the jaxpr
    IT reads. On an honest `ClosedJaxpr` those are the same object. Here
    they are not, and nothing between them compares notes — which is the
    property the query-pairing gate is supposed to have.
    """
    lo, hi = 1.0, 2.0

    def decl(n):
        x = ir.Var(1, av((n,)))
        s = ir.Var(2, av())
        pred = ir.Var(3, av((), "bool"))
        out = ir.Var(4, av((), "bool"))
        return ir.Jaxpr(
            constvars=(), invars=(), outvars=(out,),
            eqns=(
                ir.JaxprEqn("stelling_any", (), (x,),
                            (("dtype", "float64"), ("hi", hi), ("lo", lo),
                             ("shape", (n,)))),
                ir.JaxprEqn("reduce_sum", (x,), (s,),
                            (("axes", (0,)), ("out_sharding", None))),
                ir.JaxprEqn("le", (s, ir.Literal(3.9, av())), (pred,)),
                ir.JaxprEqn("stelling_assert", (pred,), (out,)),
            ), effects=())

    one, two = decl(1), decl(2)
    honest_two = ir.ClosedJaxpr(jaxpr=two, consts=())

    class RootProperty(ir.ClosedJaxpr):
        reads = 0

        @property
        def jaxpr(self):
            RootProperty.reads += 1
            return one if RootProperty.reads <= switch else two

        @jaxpr.setter
        def jaxpr(self, value):
            pass

    # measure the constructor's reads, then let `propagate` have `one` and
    # `content_hash` have `two`
    switch = 10 ** 9
    RootProperty(jaxpr=one, consts=())
    ctor = RootProperty.reads

    RootProperty.reads = 0
    switch = 10 ** 9
    q = RootProperty(jaxpr=one, consts=())
    result = propagate(q)
    assert [ob.status for ob in result.obligations] == ["discharged"], (
        "the ONE-element document should discharge; if it no longer does, "
        "this pin is measuring something else"
    )
    switch = RootProperty.reads          # everything from here on gets `two`
    assert q.content_hash() == honest_two.content_hash(), (
        "the hash the verdict would be stamped with is no longer the "
        "two-element document's; the route may have changed shape"
    )
    assert ctor >= 1


# ---------------------------------------------------------------------------
# route 2 — the INSTALL is not guaranteed to install
# ---------------------------------------------------------------------------

def test_the_INSTALL_is_swallowed_by_a_class_level_DATA_DESCRIPTOR():
    """`object.__setattr__` resolves the NAME, and a name that resolves to
    a data descriptor goes to its setter — audit 0.2.0 B6 audit 8.

    This is the mechanism half, on a bare class, so that what is being
    claimed about `_canonicalise`'s install is visible without any `ir` in
    the way.
    """
    class Descriptored:
        @property
        def field(self):
            return "WHAT THE READER GETS"

        @field.setter
        def field(self, value):
            pass

    obj = Descriptored()
    object.__setattr__(obj, "field", "WHAT THE DOOR INSTALLED")
    assert obj.field == "WHAT THE READER GETS"
    assert obj.__dict__ == {}, (
        "the install reached the instance dict after all; if CPython has "
        "changed here, `_canonicalise` is stronger than this file records"
    )


def test_a_field_that_is_a_PROPERTY_keeps_a_value_the_door_rewrote():
    """The same thing, at the door: a `JaxprEqn` SUBCLASS whose `params` is
    a property keeps a `list` in a param value.

    A `list` is the sharpest available witness because the door has an
    explicit rule about it — `_canonical` stores a `list` AS A TUPLE,
    since a list is mutable and `_encode` has no `list` arm — so a `list`
    surviving in a stored `params` is the door having decided and its
    decision having been discarded.
    """
    payload = (("shape", [2]),)

    # THE CONTROL, FIRST: the same params on an HONEST `JaxprEqn`. The
    # door rewrites the `list` to a `tuple` and installs it, so the
    # contrast below is the install being swallowed and nothing else.
    honest = ir.JaxprEqn(primitive="add", invars=(), outvars=(),
                         params=(("shape", [2]),))
    assert type(dict(honest.params)["shape"]) is tuple

    class Swallow(ir.JaxprEqn):
        @property
        def params(self):
            return payload

        @params.setter
        def params(self, value):
            pass

    eqn = Swallow(primitive="add", invars=(), outvars=(), params=payload)
    stored = eqn.params[0][1]
    assert type(stored) is list, (
        "the door's install now reaches a property-backed field; this "
        "route has closed and the disclosures that name it must move"
    )


# ---------------------------------------------------------------------------
# route 3 — `__class__` assignment
# ---------------------------------------------------------------------------

def test___class___ASSIGNMENT_makes_the_type_GENUINELY_a_stored_type():
    """`s.__class__ = ir.Var` is not a lie the door could detect: `type(s)`
    IS `ir.Var` afterwards, by the object header — audit 0.2.0 B6 audit 8.

    Every other bypass audit 7 drove was an object CLAIMING a type it did
    not have, and the repair was to stop asking the object. This one
    changes the answer `type()` itself gives, so there is nothing left to
    ask: the `id()`-keyed index finds a real hit and carries the object
    unchanged, with `Var.__post_init__` never having run on it.
    """
    class Shell:
        pass

    s = Shell()
    s.__class__ = ir.Var

    assert type(s) is ir.Var
    assert id(type(s)) in ir._STORED_AS_IS
    assert s.__dict__ == {}, "`Var.__post_init__` was not supposed to run"
    assert ir._canonical(s) is s, (
        "the door no longer carries a `__class__`-assigned object; this "
        "route has closed"
    )
    # and it lands in a real document field, carried, unvalidated
    eqn = ir.JaxprEqn(primitive="add", invars=(s,), outvars=())
    assert eqn.invars[0] is s


# ---------------------------------------------------------------------------
# what the routes do NOT reach
# ---------------------------------------------------------------------------

def test_NONE_of_the_routes_is_reachable_from_a_DOCUMENT():
    """`from_dict` returns an EXACT `ClosedJaxpr`, so none of the three
    routes above is a document's to take.

    This is the sentence that makes the disclosure a disclosure. It is
    asserted here rather than reasoned about in prose, because it is the
    load-bearing half: all three routes need a class definition or a
    `__class__` assignment in the caller's own process.
    """
    doc = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(ir.Var(1, av((2,))),), invars=(),
                       outvars=(), eqns=(), effects=()),
        consts=()).to_dict()
    back = ir.ClosedJaxpr.from_dict(doc)
    assert type(back) is ir.ClosedJaxpr
    assert type(back.jaxpr) is ir.Jaxpr
    assert type(back.jaxpr.constvars[0]) is ir.Var
    assert type(back.jaxpr.constvars[0].aval) is ir.Aval
    assert type(back.jaxpr.constvars[0].aval.shape) is tuple


# ---------------------------------------------------------------------------
# `_register_stored_type`: what the frozen check establishes
# ---------------------------------------------------------------------------

def test_the_FROZEN_check_does_not_establish_SINGLE_VALUEDNESS():
    """`_register_stored_type` accepts a frozen dataclass with a PROPERTY
    for a field — audit 0.2.0 B6 audit 8.

    Its docstring used to say single-valuedness was *"true because it is a
    FROZEN dataclass"* and that *"a plain class with a property for a
    field ... would be exactly the hole the door exists to close"*, which
    is the class this test registers and the door then carries. Frozen-ness
    stops REBINDING; it says nothing about a field name that resolves to a
    descriptor and was never an instance attribute.

    `payload` below is a REAL dataclass field — `dataclasses.fields` lists
    it — whose name resolves to a `property` in the same class body. That
    is the construction the old sentence named, not a near neighbour of
    it.

    Registration is restored afterwards, because it is module state.
    """
    saved = ir._LIBRARY_STORED_TYPES
    saved_index = ir._STORED_AS_IS

    @dataclasses.dataclass(frozen=True)
    class TwoFacedButFrozen:
        payload: tuple = ()
        _reads = 0

        @property
        def payload(self):                            # noqa: F811
            TwoFacedButFrozen._reads += 1
            return (1.0, 1.0) if TwoFacedButFrozen._reads <= 1 else (99.0, 99.0)

        @payload.setter
        def payload(self, value):
            pass

    assert [f.name for f in dataclasses.fields(TwoFacedButFrozen)] == ["payload"]

    try:
        ir._register_stored_type(TwoFacedButFrozen)
        obj = TwoFacedButFrozen(payload=(1.0, 1.0))
        assert obj.payload == (1.0, 1.0)
        assert obj.payload == (99.0, 99.0)
        assert ir._canonical(obj) is obj, (
            "the door no longer carries a registered type; the "
            "`_register_stored_type` docstring must move with this"
        )
        # FROZEN-NESS IS GENUINELY ESTABLISHED, and is genuinely a
        # different property: `setattr` still raises, and the object is
        # still two-faced.
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.payload = (0.0, 0.0)
        assert obj.payload == (99.0, 99.0)
    finally:
        ir._LIBRARY_STORED_TYPES = saved
        ir._STORED_AS_IS = saved_index
    assert ir._LIBRARY_STORED_TYPES is saved
    assert ir._STORED_AS_IS is saved_index


def test_a_SUBCLASS_of_a_registered_type_is_refused_not_carried():
    """What bounds the gap above: the door carries the EXACT registered
    type, so a property-backed variant must be the type someone
    REGISTERED — it cannot be smuggled in under `IntervalArray`'s
    registration.
    """
    class SneakyBox(interval.IntervalArray):
        reads = 0

        @property
        def los(self):
            SneakyBox.reads += 1
            return (1.0, 1.0) if SneakyBox.reads <= 2 else (99.0, 99.0)

        @los.setter
        def los(self, value):
            pass

    box = SneakyBox(shape=(2,), los=(1.0, 1.0), his=(2.0, 2.0))
    assert box.los == (99.0, 99.0)          # it really is two-faced
    with pytest.raises(ir._NotCanonical):
        ir._canonical(box)


def test_NO_DOCUMENT_reaches_the_registered_arm():
    """The other bound: a registered value cannot be serialized or
    deserialized, so it can only come from a caller who built it.
    """
    box = interval.IntervalArray(shape=(1,), los=(1.0,), his=(2.0,))
    q = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=(), eqns=(),
                       effects=()),
        consts=(box,))
    for call in (q.to_dict, q.content_hash):
        with pytest.raises(TypeError, match="cannot encode IntervalArray"):
            call()
    with pytest.raises(ValueError, match="unknown tag"):
        ir.ClosedJaxpr.from_dict({
            "k": "closed",
            "jaxpr": {"k": "jaxpr", "constvars": [], "invars": [],
                      "outvars": [], "eqns": [], "effects": []},
            "consts": [{"k": "interval"}]})


# ---------------------------------------------------------------------------
# the MRO-forgery claim, in the scope it holds in
# ---------------------------------------------------------------------------

def _forgeable(base, derive_from=()):
    """Can a metaclass hand the interpreter an MRO containing ``base``?"""
    class Meta(type):
        def mro(cls):
            extra = [b for b in derive_from if b is not object]
            tail = [] if base is object else [object]
            return [cls] + extra + [base] + tail

    try:
        return Meta("Forged", derive_from, {})
    except TypeError:
        return None


def test_MRO_FORGERY_is_possible_for_most_stored_types_and_buys_NOTHING():
    """*"`bool` is the only type this module stores for which it is
    possible"* was false — audit 0.2.0 B6 audit 8.

    What CPython's layout check refuses is a base that carries its own
    instance layout. Every type whose solid base is `object` — all of
    `ir`'s own dataclasses, `NoneType`, and the registered
    `IntervalArray` — adds no layout and can be claimed by a plain heap
    class. The claim that IS true is the one `_read_or_refuse`'s docstring
    makes, scoped to the arms that actually read `issubclass`.

    Both halves are computed from the door's own declaration of what it
    stores, so a type added later joins this measurement by being added.
    """
    stored = (tuple(ir._CANONICAL_EXACT_TYPES)
              + tuple(ir._CANONICAL_IR_TYPES)
              + tuple(ir._LIBRARY_STORED_TYPES))
    read_bases = tuple(b for b, _ in ir._CANONICAL_READS)
    containers = tuple(ir._SHAPE_PARAM_CONTAINERS)

    forgeable, refused = [], []
    for t in stored:
        cls = _forgeable(t)
        if cls is None and t is bool:
            # the one direction CPython permits: from a real `int`
            # subclass, which shares `int`'s layout
            cls = _forgeable(bool, derive_from=(int,))
        (forgeable if cls is not None else refused).append(t)

    # THE ARMS THAT READ `issubclass` ARE EXACTLY THE ONES CPYTHON REFUSES.
    # That, and not "only `bool`", is what makes the forgery inert here.
    for base in read_bases + containers:
        assert _forgeable(base) is None, (
            f"an MRO entry for {base.__name__} can now be forged, and it "
            f"is a base this door dispatches `issubclass` against"
        )
    # what CPython refuses, among the types this module STORES, is exactly
    # the read bases: `tuple` and `list` are refused too but are not
    # themselves stored types, which is why they are checked in the loop
    # above and not in this set
    assert set(refused) == set(read_bases), (
        f"refused={sorted(t.__name__ for t in refused)} "
        f"read_bases={sorted(t.__name__ for t in read_bases)}"
    )
    assert len(forgeable) == len(stored) - len(refused)
    assert len(forgeable) > 1, (
        "the record says most stored types are forgeable; measured "
        f"{len(forgeable)} of {len(stored)}"
    )

    # AND FORGING ONE BUYS NOTHING: membership is `id(type(obj))` against
    # the real type object, which no MRO can move.
    checked = 0
    for t in stored:
        if t in read_bases or t in containers:
            continue
        cls = _forgeable(t)
        if cls is None:
            continue
        obj = object.__new__(cls)         # the base's __post_init__ never runs
        assert issubclass(type(obj), t)
        with pytest.raises(ir._NotCanonical):
            ir._canonical(obj)
        checked += 1
    assert checked >= 13, f"only {checked} forged claimers were driven"


def test_the_one_forgery_CPython_permits_is_stored_as_the_int_it_carries():
    """`bool` from a real `int` subclass shares `int`'s layout, so it is
    not a lie about the payload — and `int.__index__` stores that payload.
    """
    cls = _forgeable(bool, derive_from=(int,))
    assert cls is not None, "CPython no longer permits this direction"
    obj = cls(7)
    assert issubclass(type(obj), bool)
    out = ir._canonical(obj)
    assert type(out) is int and out == 7


# ---------------------------------------------------------------------------
# and the disclosures are held to the routes this file drives
# ---------------------------------------------------------------------------

# the sentence audit 8 falsified. It must not come back in any shipped
# page, in either of the two files that carried it.
_FALSE_SENTENCE = re.compile(
    r"only an\s+`?object\.__setattr__`?\s+past the frozen dataclass\s+reaches it",
    re.S,
)

_DISCLOSURE_PAGES = (
    "SOUNDNESS.md",
    "CHANGELOG.md",
    "tests/test_aval_lie_both_faces.py",
)

# The sentence is allowed in ONE shape: quoted, inside `*"..."*`, in a
# paragraph that also says how many routes there really are. That is how
# this project records a claim it has retracted, and a check that forbade
# the words outright would forbid saying that they were wrong.
#
# `CHANGELOG.md` JOINED THE LIST ABOVE FOR A REASON — batch B8c. The
# falsified sentence was quoted in `CHANGELOG.md`, which this check did
# not read, and the 0.2.0 documentation routing moved that quotation into
# `SOUNDNESS.md`, which it did. The check went red at a sentence that had
# been sitting unexamined in a shipped page all along: a page-list guard
# is only as wide as its list. Both pages are read now, and the shape is
# checked rather than the words banned.
_RETRACTED = re.compile(
    r"\*\"[^\"]*only an\s+`?object\.__setattr__`?[^\"]*\"\*"
    r"(?:.{0,400}?)(?:three|3)\s+routes?|"
    r"named one route where there are three",
    re.S,
)


def _enclosing_block(text: str, index: int) -> str:
    """The blank-line-delimited paragraph containing `index`."""
    start = text.rfind("\n\n", 0, index)
    start = 0 if start < 0 else start + 2
    end = text.find("\n\n", index)
    return text[start:len(text) if end < 0 else end]


def test_the_DISCLOSURES_name_exactly_these_routes():
    """AN ENUMERATION A READER CAN CHECK — audit 0.2.0 B6 audit 8.

    The repair for *"only an `object.__setattr__` ... reaches it"* is not
    a longer sentence; it is that the sentence and the tests move
    together. Every route this file drives must be named in the prose that
    discloses the residue, and the falsified sentence must be gone from
    both places that carried it.

    IF THIS TEST FAILS because a route name changed, change it in
    :data:`ROUTES` and in the disclosure in the same commit — that is the
    coupling it exists to impose.
    """
    for rel in _DISCLOSURE_PAGES:
        path = _REPO / rel
        assert path.is_file(), rel
        text = path.read_text(encoding="utf-8")
        for match in _FALSE_SENTENCE.finditer(text):
            block = _enclosing_block(text, match.start())
            assert _RETRACTED.search(block), (
                f"{rel} states that only an `object.__setattr__` reaches "
                f"the residue, and this file drives {len(ROUTES)} routes "
                f"that do not. The sentence is permitted ONLY as a QUOTED "
                f"RETRACTION — inside `*\"...\"*` and in a paragraph that "
                f"says how many routes there really are. This occurrence "
                f"does neither:\n{block[:400]}"
            )

    soundness = (_REPO / "SOUNDNESS.md").read_text(encoding="utf-8")
    missing = [phrase for _key, phrase in ROUTES if phrase not in soundness]
    assert not missing, (
        f"SOUNDNESS.md does not name these driven routes: {missing}. "
        f"Every route in `ROUTES` is exhibited by a test in this file and "
        f"must be named where the residue is disclosed."
    )


# the marker that says a route's status is a DECISION rather than an
# oversight. The same string in both places that carry the status —
# `SOUNDNESS.md` and the driven test's own docstring — so neither can drift.
# It is a constant HERE and read from THERE: a readback that reads the file
# it is written in is answered by its own definition, which is what this
# constant's readback used to do (see that test).
_DECIDED = "OUT OF SCOPE BY DECISION, 2026-08-18"


def test_the_ROOT_route_is_recorded_as_a_DECISION_not_as_an_OPEN_item():
    """A ROUTE'S STATUS IS PROSE, AND PROSE ROTS UNLESS SOMETHING READS IT.

    The root route is out of scope by an explicit decision of the principal
    (2026-08-18): *"It can be addressed with proper CI security on projects
    that need it. It requires actively malicious python to actually happen."*
    B6's door covers an object's FIELDS and was never scoped to its root.

    That is a decision, not a discovery, and the failure mode it invites is
    the opposite of the usual one: not a residue that stops being measured,
    but a decision that stops being recorded — after which a later reader
    meets the row below, reads it as an oversight, and reopens work that was
    deliberately declined. So both places that carry the status are read here
    against the same string, and the route's own driven test is required to
    still MEASURE something. **Removing the test is not how this route gets
    closed**; it is still true of the tree, and only its status changed.

    **THE READBACK USED TO BE SATISFIED BY ITSELF, TWICE OVER** — audit 0.2.0
    B11 re-audit, fix 4, and it is the readback that keeps the principal's
    ruling honest, so a self-satisfying one is worse than none. It read the
    marker out of `Path(__file__)` — the file three lines above defines
    `_DECIDED` as a module constant, so `_DECIDED in here` was answered by
    its own definition, and `"actively malicious python" in here` by THIS
    docstring. And it looked for the driven test by grepping its `def` line,
    which a body of `pass` still satisfies. Measured on `bd50171`: deleting
    the whole nineteen-line STATUS block from that test's docstring left this
    file `14 passed`, and replacing that test's entire body with `pass` left
    it `14 passed` — the route stops being measured and nothing reddens,
    which is the exact failure this test says it prevents.

    So both legs now read the DRIVEN TEST'S OWN DOCSTRING, parsed out of the
    file rather than grepped from it. That region defines no constant this
    test uses and contains none of this test's own words, so neither leg can
    answer itself; and the third assertion counts the driven test's
    ASSERTIONS instead of its name.
    """
    import ast

    soundness = (_REPO / "SOUNDNESS.md").read_text(encoding="utf-8")
    source = Path(__file__).read_text(encoding="utf-8")
    driven = "test_the_ROOT_of_a_query_is_never_canonicalized"
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == driven
    ]
    assert nodes, (
        f"the root route's driven test `{driven}` is gone. Out of scope is a "
        f"statement about what will be FIXED, never about what will be "
        f"MEASURED"
    )
    (node,) = nodes
    status = ast.get_docstring(node) or ""

    for label, text in (("SOUNDNESS.md", soundness), (f"{driven}'s docstring", status)):
        assert _DECIDED in text, (
            f"{label} no longer records the root route's status as a "
            f"decision taken on a date. If the decision has been REVERSED, "
            f"say so here and reopen it; if it has merely been edited away, "
            f"put it back — an undated 'we are not doing this' is what a "
            f"later reader reopens as an oversight."
        )
        assert "actively malicious python" in text, (
            f"{label} no longer records the reasoning the decision rests on "
            f"beside the status it justifies"
        )

    # ... and the route is still MEASURED, not merely still named. A `def`
    # line survives a body of `pass`; an assertion count does not.
    asserts = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Assert)
        or (
            isinstance(n, ast.withitem)
            and isinstance(n.context_expr, ast.Call)
            and getattr(n.context_expr.func, "attr", None) == "raises"
        )
    ]
    assert len(asserts) >= 2, (
        f"`{driven}` makes {len(asserts)} assertion(s). The decision to leave "
        f"this route open is worth nothing if the route stops being measured, "
        f"and a test that asserts nothing has stopped measuring it while "
        f"still satisfying every check that looks for its name"
    )


def test_every_route_in_the_list_has_a_test_that_drives_it():
    """`ROUTES` is the enumeration the disclosure is checked against, so
    it may not grow an entry that nothing exercises.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    names = set(re.findall(r"^def (test_\w+)", source, re.M))
    drivers = {
        "root": "test_the_ROOT_of_a_query_is_never_canonicalized",
        "install": "test_the_INSTALL_is_swallowed_by_a_class_level_DATA_DESCRIPTOR",
        "class_assignment":
            "test___class___ASSIGNMENT_makes_the_type_GENUINELY_a_stored_type",
    }
    assert set(drivers) == {key for key, _ in ROUTES}
    for key, name in drivers.items():
        assert name in names, f"route {key!r} names a test that does not exist"
