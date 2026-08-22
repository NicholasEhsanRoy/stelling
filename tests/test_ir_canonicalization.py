# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A DOCUMENT-SUPPLIED VALUE IS STORED AS AN EXACT BUILT-IN, OR REFUSED.

AUDIT 0.2.0 B6 AUDIT 6. Five repairs before this one each closed one PAIR
of reads of one document-supplied value:

    1  the `shape` param      validated by `tuple()`   consumed by `tuple()`
    2  the param KEY          validated by `==` x2     read by `==` a third time
    3  the same KEY           counted by `hash`        counted by `eq`
    4  the `dtype` param      validated by `==`        consumed by `str()`
    5  `axes`, `new_sizes`,   validated by NOTHING     read by various
       `slice_sizes`,
       `dimension_numbers`

Row 2 was found inside the fix for row 1, in the same line: the install
`(k, dims) if k == "shape" else (k, v)` is itself a read of a
document-supplied key. Three consecutive rounds ended with a new row, and
a sixth (`lo`/`hi`, read once by `_ieee_any` and again by the
assume-certification gate) fell out of writing this file. Enumerating the
rows is what produced the last three rounds.

**THE FACT UNDERNEATH ALL OF THEM.** Python's protocols are overridable,
so an object a document supplies can answer `==`, `hash`, `iter`, `str`,
`index`, `len` or `getitem` differently on two reads, and a guard that
checked one answer has not checked the other. The repair that does not
need the rows enumerated is to leave nothing overridable in what is
STORED: `ir`'s canonicalization door replaces every value with an exact
instance of a type the module is closed over, reading a subclass once
through its base type's own accessor, and refuses a type it has no exact
form for. A later read cannot differ because there is no subclass left to
answer it — a property of the stored object, so it holds for readers
nobody has written yet.

**WHY A SUBCLASS IS READ AND NOT REFUSED**, measured below: the trace path
produces them. `np.float64` IS a `float` subclass and `np.str_` IS a `str`
subclass, and `_jax_compat.Transcriber.param` returns both unchanged.

**WHAT THIS DOES NOT DO**, said plainly, because the distinction is the
whole of row 5: it makes each param SINGLE-VALUED, not CORRECT. `axes`
still has no schema — giving it one would be per-primitive shape
inference, which `ir.py` scopes out in writing — so the transfer and the
emission now read the same extents, and whether those are the right
extents is a different claim this file does not make.
"""
from __future__ import annotations

import array as _arraymod
import base64
import binascii
import dataclasses
import collections
import hashlib
import pathlib
import re
import struct
import typing
from fractions import Fraction

import pytest

from stelling import ir
from stelling.obligation import DeclinedObligation, slice_unknown_obligations
from stelling.propagate import interval_env, propagate


def av(shape=(), dtype="float64"):
    return ir.Aval(kind="ShapedArray", shape=tuple(shape), dtype=dtype)


def _canonical_document() -> ir.ClosedJaxpr:
    """One well-formed document holding every value KIND the door sees."""
    a4, a0 = av((4,)), av()
    ab = av((), "bool")
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(ir.Var(id=10, aval=a0),),
                       outvars=(ir.Var(id=10, aval=a0),), eqns=())
    )
    x, s = ir.Var(id=2, aval=a4), ir.Var(id=3, aval=a0)
    pred, out = ir.Var(id=4, aval=ab), ir.Var(id=5, aval=ab)
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(ir.Var(id=0, aval=av((2,))),),
            invars=(),
            outvars=(out,),
            eqns=(
                ir.JaxprEqn(
                    primitive="stelling_any", invars=(), outvars=(x,),
                    params=(("dtype", "float64"), ("hi", 2.0), ("lo", 1.0),
                            ("shape", (4,))),
                    source_info=("harness.py:1 (h)",)),
                ir.JaxprEqn(
                    primitive="reduce_sum", invars=(x,), outvars=(s,),
                    params=(("axes", (0,)), ("out_sharding", None))),
                ir.JaxprEqn(
                    primitive="closed_call", invars=(s,), outvars=(s,),
                    params=(("call_jaxpr", inner),
                            ("ntuple", ir.NamedTupleParam(
                                cls="Dims", fields=(("lhs", (0,)),))),
                            ("enum", ir.EnumParam(cls="Mode", member="CLIP")),
                            ("treedef", ir.TreeDefParam(text="PyTreeDef(*)")),
                            ("sentinel", ir.SentinelParam(cls="object")),
                            ("opaque", ir.OpaqueParam(cls="Thunk")),
                            ("note", "a str param")),
                    effects=("io",)),
                ir.JaxprEqn(primitive="le",
                            invars=(s, ir.Literal(val=4.5, aval=a0)),
                            outvars=(pred,)),
                ir.JaxprEqn(primitive="stelling_assert",
                            invars=(pred,), outvars=(out,)),
            ),
            debug_info=ir.DebugInfo(func="h", arg_names=("a",),
                                    result_paths=("",)),
        ),
        consts=(ir.Array(dtype="<f8", shape=(2,),
                         data=struct.pack("<2d", 1.0, 2.0)),),
    )


def _stored_values(obj, path="query"):
    """(path, value) for every value reachable from a document, walked
    through `dataclasses.fields` so a field added later joins with no edit
    here."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _stored_values(getattr(obj, f.name), f"{path}.{f.name}")
    elif isinstance(obj, (tuple, list)):
        yield path, obj
        for i, item in enumerate(obj):
            yield from _stored_values(item, f"{path}[{i}]")
    else:
        yield path, obj


# ---------------------------------------------------------------------------
# the class, one row at a time
# ---------------------------------------------------------------------------

def test_row_4_the_dtype_validated_by_eq_is_the_one_consumed_by_str():
    """`_validate_decl_eqn` compared ``dtype == aval.dtype``, which a `str`
    SUBCLASS satisfies (and it satisfies the `isinstance` above the
    comparison too), while `propagate._ieee_any` consumes
    ``str(_req(params, "dtype", ...))`` and picks the SUBNORMAL BAND from
    that. Overriding `__str__` alone showed the door one dtype and the
    ieee declaration transfer another.

    Measured at `f729d70`: the door accepted a param whose `repr` is
    ``'float64'`` and whose `str()` is ``'int64'`` — the arm `_ieee_any`
    takes for a declaration with no float format, so no subnormal haze at
    all. The auditor did not drive it to a moved verdict and neither did
    this repair (the comparison transfers haze their operands from the
    value's own aval as well, which caught every direction tried), so what
    is asserted here is the READ and not a verdict.

    TWO HALVES, because the repair has two and only one of them is where
    the class lives. What is STORED is settled by
    `_canonical_param_values`, which runs over every param; what is
    COMPARED is settled by `_validate_decl_eqn` reading the same value
    through `str.__str__` before it compares. Without the second, a param
    whose `__eq__` answers True to anything passes the door as
    ``'float64'`` and is then stored as the ``'float32'`` it actually
    carries — the door would have validated a dtype the equation does not
    hold, which is the same defect one protocol over."""
    class LyingStr(str):
        def __str__(self):
            return "int64"

    eqn = ir.JaxprEqn(
        primitive="stelling_any", invars=(), outvars=(ir.Var(1, av((2,))),),
        params=(("dtype", LyingStr("float64")), ("hi", 2.0), ("lo", 1.0),
                ("shape", (2,))),
    )
    stored = eqn.params_dict()["dtype"]
    assert type(stored) is str, type(stored)
    # the two reads the door and the transfer make are now one value
    assert stored == "float64"
    assert str(stored) == "float64", (
        "the dtype the ieee declaration transfer consumes is not the dtype "
        "the door compared against the outvar aval"
    )

    class LyingEq(str):
        def __eq__(self, other):
            return True

        def __hash__(self):
            return str.__hash__(self)

    with pytest.raises(ir.TranscriptionError, match="contradicts"):
        ir.JaxprEqn(
            primitive="stelling_any", invars=(),
            outvars=(ir.Var(1, av((2,), "float64")),),
            params=(("dtype", LyingEq("float32")), ("hi", 2.0), ("lo", 1.0),
                    ("shape", (2,))))


def test_row_5_a_param_with_no_door_is_still_single_valued():
    """``axes`` has no validation between the document and its readers, and
    this repair does not give it one — that would be per-primitive shape
    inference, which `ir.py` scopes out in writing. What it gives it is
    single-valuedness.

    Measured at `f729d70` on this document: the param was read TWICE with
    two different answers, the propagation built a box of shape ``(2,)``
    for a value whose aval says scalar, and `_one_shape_per_value` caught
    the divergence and DECLINED. So the row was real, and what stood
    between it and a verdict was an invariant somewhere else.

    Here the lying `__iter__` is not consulted at all — the door reads the
    tuple's own payload — so the two legs agree and the obligation is
    SLICED. The change visible in this row is liveness; the soundness
    claim is the single-valuedness itself, which is what removes the
    divergences that invariant does not happen to catch."""
    reads = []

    class LyingAxes(tuple):
        def __iter__(self):
            reads.append(1)
            return iter((0, 1) if len(reads) <= 1 else (0,))

    x, s = ir.Var(1, av((2, 2))), ir.Var(2, av())
    pred, out = ir.Var(3, av((), "bool")), ir.Var(4, av((), "bool"))
    q = ir.ClosedJaxpr(jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=(out,), eqns=(
        ir.JaxprEqn(primitive="stelling_any", invars=(), outvars=(x,),
                    params=(("dtype", "float64"), ("hi", 2.0), ("lo", 1.0),
                            ("shape", (2, 2)))),
        ir.JaxprEqn(primitive="reduce_sum", invars=(x,), outvars=(s,),
                    params=(("axes", LyingAxes((0, 1))),
                            ("out_sharding", None))),
        ir.JaxprEqn(primitive="le", invars=(s, ir.Literal(val=7.9, aval=av())),
                    outvars=(pred,)),
        ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
    )))

    stored = q.jaxpr.eqns[1].params_dict()["axes"]
    assert type(stored) is tuple and stored == (0, 1), stored
    assert reads == [], (
        f"the lying `__iter__` was consulted {len(reads)} time(s); the door "
        f"is supposed to read the tuple's own payload"
    )

    # the exact oracle: four elements each at most 2 sum to 8 > 79/10
    assert Fraction(8) > Fraction(79, 10)
    p = propagate(q)
    assert p.obligations[0].status == "unknown", p.obligations[0].status
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert not isinstance(item, DeclinedObligation), item.reason
    assert len(item.inputs) == 4, item.inputs


def test_lo_and_hi_are_single_valued_too():
    """The row that fell out of writing this file, and the reason the rule
    is uniform rather than a list. `lo` and `hi` are read by
    `propagate._ieee_any` to build the declared box and AGAIN by the
    assume-certification gate at the foot of the same module, and nothing
    ever validated them. They are `float` params, so the same door covers
    them without anyone having noticed they were a row."""
    reads = []

    class Drifting(float):
        def __float__(self):
            reads.append(1)
            return 2.0 if len(reads) <= 1 else 99.0

    eqn = ir.JaxprEqn(
        primitive="stelling_any", invars=(), outvars=(ir.Var(1, av((2,))),),
        params=(("dtype", "float64"), ("hi", Drifting(2.0)), ("lo", 1.0),
                ("shape", (2,))))
    hi = eqn.params_dict()["hi"]
    assert type(hi) is float and hi == 2.0, hi
    assert reads == [], (
        "the door asked the object what it would like to be read as, "
        "instead of reading the float it carries"
    )


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------

def _sub(base):
    """A SUBCLASS of `base` that is a perfectly good value of its type.

    Built with `type()` rather than named, so `bool` and `NoneType` — which
    cannot be subclassed — are simply not asked for anywhere below."""
    return type(f"Sub{base.__name__}", (base,), {})


# ---------------------------------------------------------------------------
# the oracle, which may not be the door's own test
# ---------------------------------------------------------------------------

def _allowed_stored_types() -> tuple[type, ...]:
    """Every type a stored value is permitted to be, from `ir`'s own
    DECLARATIONS — not from `ir._STORED_AS_IS`, which is the door's own
    index of them and would make this test agree with the door by
    construction."""
    return (tuple(ir._CANONICAL_EXACT_TYPES) + tuple(ir._CANONICAL_IR_TYPES)
            + tuple(ir._LIBRARY_STORED_TYPES) + (tuple,))


def _is_an_allowed_stored_type(v) -> bool:
    """Is ``type(v)`` one of the permitted types — BY IDENTITY?

    **THE ORACLE MAY NOT SHARE THE DOOR'S OWN PRIMITIVE** — audit 0.2.0 B6
    audit 7, and this file is where that was found. The two assertions
    below used to read ``type(v) not in allowed`` over a `set` of types,
    which is the same `frozenset` membership `ir._canonical` used to
    decide whether to canonicalize at all — so the METACLASS of a
    document-supplied object answered both, and this test reported NO
    DEFECT for exactly the object the door had waved through:

        the TEST oracle: type(v) not in allowed  -> False   (no defect)
        the DOOR:        type(v) in _CANONICAL_EXACT -> True (store as is)

    An instrument that shares a primitive with the thing it measures
    cannot measure that primitive. Identity is not a protocol and nothing
    can answer it, so this loop reports on the object rather than asking
    it — and the door's own membership test is spelled differently again
    (``id(t)`` into a dict), so the two are independent implementations of
    the same meaning rather than one calling the other."""
    t = type(v)
    for k in _allowed_stored_types():
        if t is k:
            return True
    return False


def _meta_lie(face):
    """A metaclass answering ``==``/``hash`` as ``face`` — which is all a
    `frozenset` membership test on a type ever asks."""

    class M(type):
        def __hash__(cls):
            return hash(face)

        def __eq__(cls, other):
            return other is face

    return M


def _liar(face, *, metaclass: bool, class_property: bool):
    """An object that CLAIMS to be ``face`` and carries none of its payload.

    The two bypasses audit 7 drove, separately and together: a metaclass
    that answers the door's `frozenset` membership, and a ``__class__``
    property that answers its ``isinstance``. The protocols this module's
    readers ask of a stored value are all implemented, so the object is
    refused for what it IS and not for failing to behave."""
    ns = {
        "__float__": lambda self: 3.9,
        "__index__": lambda self: 3,
        "__str__": lambda self: "float64",
        "__repr__": lambda self: "<liar>",
        "__len__": lambda self: 1,
        "__iter__": lambda self: iter((1,)),
        "__getitem__": lambda self, k: 1,
        "__eq__": lambda self, other: True,
        "__hash__": lambda self: 0,
    }
    if class_property:
        ns["__class__"] = property(lambda self: face)
    mcls = _meta_lie(face) if metaclass else type
    return mcls(f"Liar_{face.__name__}", (), ns)()


_S, _I, _F, _B, _T = (_sub(str), _sub(int), _sub(float), _sub(bytes),
                      _sub(tuple))


def _subclass_document() -> ir.ClosedJaxpr:
    """The same shape of document, written entirely in SUBCLASS spellings.

    Every field of every dataclass in `stelling.ir` that can hold one holds
    one — which is what makes this the document that measures the door
    rather than the document that happens to be canonical already."""
    a0 = ir.Aval(kind=_S("ShapedArray"), shape=(), dtype=_S("float64"))
    a4 = ir.Aval(kind=_S("ShapedArray"), shape=_T((_I(4),)), dtype=_S("float64"))
    ab = ir.Aval(kind=_S("ShapedArray"), shape=(), dtype=_S("bool"))
    arr = ir.Array(dtype=_S("<f8"), shape=_T((_I(2),)),
                   data=_B(struct.pack("<2d", 1.0, 2.0)))
    inner = ir.ClosedJaxpr(jaxpr=ir.Jaxpr(
        constvars=(), invars=(ir.Var(id=_I(10), aval=a0),),
        outvars=(ir.Var(id=_I(10), aval=a0),), eqns=()))
    x, s = ir.Var(id=_I(2), aval=a4), ir.Var(id=_I(3), aval=a0)
    pred, out = ir.Var(id=_I(4), aval=ab), ir.Var(id=_I(5), aval=ab)
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(ir.Var(id=_I(0), aval=ir.Aval(
                kind=_S("ShapedArray"), shape=_T((_I(2),)),
                dtype=_S("float64"))),),
            invars=(),
            outvars=(out,),
            eqns=_T((
                ir.JaxprEqn(
                    primitive=_S("stelling_any"), invars=(), outvars=(x,),
                    params=_T(((_S("dtype"), _S("float64")),
                               (_S("hi"), _F(2.0)), (_S("lo"), _F(1.0)),
                               (_S("shape"), _T((_I(4),))))),
                    source_info=_T((_S("harness.py:1 (h)"),))),
                ir.JaxprEqn(
                    primitive=_S("reduce_sum"), invars=(x,), outvars=(s,),
                    params=((_S("axes"), [_I(0)]),
                            (_S("out_sharding"), None))),
                ir.JaxprEqn(
                    primitive=_S("closed_call"), invars=(s,), outvars=(s,),
                    params=((_S("call_jaxpr"), inner),
                            (_S("ntuple"), ir.NamedTupleParam(
                                cls=_S("Dims"),
                                fields=_T(((_S("lhs"), _T((_I(0),))),)))),
                            (_S("enum"), ir.EnumParam(cls=_S("Mode"),
                                                      member=_S("CLIP"))),
                            (_S("treedef"), ir.TreeDefParam(
                                text=_S("PyTreeDef(*)"))),
                            (_S("sentinel"), ir.SentinelParam(
                                cls=_S("object"))),
                            (_S("opaque"), ir.OpaqueParam(cls=_S("Thunk"))),
                            (_S("cplx"), _sub(complex)(1, 2))),
                    effects=_T((_S("io"),))),
                ir.JaxprEqn(primitive=_S("le"),
                            invars=(s, ir.Literal(val=_F(4.5), aval=a0)),
                            outvars=(pred,)),
                ir.JaxprEqn(primitive=_S("stelling_assert"),
                            invars=(pred,), outvars=(out,)),
            )),
            effects=_T((_S("io"),)),
            debug_info=ir.DebugInfo(func=_S("h"), arg_names=_T((_S("a"),)),
                                    result_paths=_T((_S(""),)))),
        consts=_T((arr,)))


_LIAR_SPELLINGS = [
    ("metaclass only", dict(metaclass=True, class_property=False)),
    ("__class__ only", dict(metaclass=False, class_property=True)),
    ("both", dict(metaclass=True, class_property=True)),
]


def _liar_document(face, **how):
    """A document of the false-VERIFIED reproducer's shape, holding ONE
    object that lies about its type in the position
    `a1_false_verified_metaclass.py` attacked — the ceiling of the
    asserted predicate."""
    def build():
        a0 = av()
        x, s = ir.Var(id=2, aval=av((4,))), ir.Var(id=3, aval=a0)
        pred, out = ir.Var(id=4, aval=av((), "bool")), ir.Var(id=5, aval=av((), "bool"))
        return ir.ClosedJaxpr(jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=(out,), eqns=(
                ir.JaxprEqn(primitive="stelling_any", invars=(), outvars=(x,),
                            params=(("dtype", "float64"), ("hi", 2.0),
                                    ("lo", 1.0), ("shape", (4,)))),
                ir.JaxprEqn(primitive="reduce_sum", invars=(x,), outvars=(s,),
                            params=(("axes", (0,)), ("out_sharding", None))),
                ir.JaxprEqn(primitive="le",
                            invars=(s, ir.Literal(val=_liar(face, **how),
                                                  aval=a0)),
                            outvars=(pred,)),
                ir.JaxprEqn(primitive="stelling_assert", invars=(pred,),
                            outvars=(out,)),
            )))
    return build


# COMPUTED from the door's own declaration of what it stores, plus the
# containers it names separately, so a type added to `ir` later is lied
# about here without anyone having to remember this list.
_LIAR_FACES = (tuple(ir._CANONICAL_EXACT_TYPES)
               + tuple(ir._SHAPE_PARAM_CONTAINERS))

_POPULATION = (
    [(_canonical_document, False, "exact spellings"),
     (_subclass_document, False, "subclass spellings")]
    + [(_liar_document(face, **how), True, f"LIAR {face.__name__} ({name})")
       for face in _LIAR_FACES
       for name, how in _LIAR_SPELLINGS]
)


@pytest.mark.parametrize(
    "build,must_refuse",
    [(b, r) for b, r, _id in _POPULATION],
    ids=[i for _b, _r, i in _POPULATION],
)
def test_every_value_a_document_stores_is_of_an_EXACT_type(build, must_refuse):
    """THE PROPERTY, over the whole document and computed from the
    dataclasses' own fields rather than from a list of the ones anyone
    remembered.

    Driven over three populations: one document in EXACT spellings, which
    says the door does not damage a well-formed query; the same document
    in SUBCLASS spellings, which holds a subclass in every field of every
    dataclass that can carry one; and a family of LIAR documents — added
    at audit 0.2.0 B6 audit 7 — each holding one object that merely
    CLAIMS to be a stored type, which is what this test was blind to for
    the whole of its existence.

    **WHY IT WAS BLIND, because it is the batch's signature pattern
    arriving inside the fix for the batch's signature pattern.** The
    assertion read ``type(v) not in allowed`` over a set of types — the
    same `frozenset` membership `ir._canonical` used to decide whether to
    canonicalize at all. The metaclass answered both, so the door stored
    the liar and this test agreed that it had not. The oracle is
    :func:`_is_an_allowed_stored_type` now, which asks identity and asks
    nothing of the object; the door asks ``id()``; neither is a call into
    the other.

    A liar document is REFUSED rather than stored, so the property holds
    over it vacuously — which is why ``must_refuse`` is checked rather
    than left to the walk. A row that stopped being refused would
    otherwise pass this test by having nothing to walk."""
    try:
        doc = build()
    except ir.TranscriptionError as exc:
        assert must_refuse, f"a well-formed document was refused: {exc}"
        assert "has no exact form to store" in str(exc), str(exc)
        return
    wrong = [(p, type(v).__name__) for p, v in _stored_values(doc)
             if not _is_an_allowed_stored_type(v)]
    assert not must_refuse, (
        "a document holding a value that LIES about its type was BUILT "
        "rather than refused; what it now carries, as the oracle sees it: "
        + (repr(wrong) if wrong else
           "nothing the oracle objects to — which means the oracle shares "
           "the hole with the door, and that is the defect twice")
    )
    assert not wrong, (
        f"{len(wrong)} stored value(s) are not of an exact stored type — a "
        f"later reader can be handed a different answer at each of them:\n  "
        + "\n  ".join(f"{p}: {t}" for p, t in wrong)
    )
    # ... and the walk is worth running
    assert len(list(_stored_values(doc))) >= 60


def test_the_two_spellings_of_the_same_document_are_ONE_document():
    """And the point of storing the exact twin rather than merely checking
    it: a query written in subclass spellings and the same query written
    in exact ones are equal, serialize alike and hash alike. A door that
    only VALIDATED would leave two documents here."""
    a, b = _subclass_document(), _canonical_document()
    # the declaration is written in subclass spellings on one side and in
    # exact ones on the other, and they are the same equation
    assert a.jaxpr.eqns[0] == b.jaxpr.eqns[0]
    # `axes` was a `list` of an `int` subclass on the left and a `tuple` of
    # plain `int` on the right
    assert a.jaxpr.eqns[1].params_dict()["axes"] == (0,)
    assert type(a.jaxpr.eqns[1].params_dict()["axes"]) is tuple
    assert a.jaxpr.effects == ("io",)
    assert type(a.jaxpr.debug_info.func) is str
    # and the whole document round-trips and hashes, which the subclass
    # spelling could not do before — `_encode` has no `list` arm at all and
    # this one's `axes` param is one. (A `str` subclass was never the
    # problem for `_encode`: it takes the `isinstance(obj, str)` arm and
    # `json.dumps` writes it. The problem it was is that two reads of it
    # can differ, which is what the rest of this file is about.)
    assert ir.ClosedJaxpr.from_dict(a.to_dict()) == a
    assert len(a.content_hash()) == 64


@pytest.mark.parametrize(
    "make,base",
    [
        (lambda: type("S", (str,), {})("float64"), str),
        (lambda: type("I", (int,), {})(4), int),
        (lambda: type("F", (float,), {})(1.5), float),
        (lambda: type("C", (complex,), {})(1, 2), complex),
        (lambda: type("B", (bytes,), {})(b"ab"), bytes),
        (lambda: type("T", (tuple,), {})((1, 2)), tuple),
    ],
    ids=["str", "int", "float", "complex", "bytes", "tuple"],
)
def test_a_subclass_of_a_stored_type_is_replaced_by_its_exact_twin(make, base):
    """A subclass is CANONICALIZED, not refused, because the trace path
    produces them (see the two-route test below)."""
    v = make()
    assert type(v) is not base
    out = ir._canonical(v)
    assert type(out) is base, type(out)
    assert out == base(v) if base is not tuple else out == tuple(v)


@pytest.mark.parametrize(
    "value,name",
    [
        (memoryview(b"\x02\x02"), "memoryview"),
        (_arraymod.array("b", b"\x02\x02"), "array"),
        (range(2), "range"),
        (object(), "object"),
        ({1: 2}, "dict"),
        ({1, 2}, "set"),
        (bytearray(b"ab"), "bytearray"),
    ],
    ids=["memoryview", "array.array", "range", "object", "dict", "set",
         "bytearray"],
)
def test_a_type_with_no_exact_form_to_store_is_REFUSED(value, name):
    """Not carried, and not coerced. A value the module cannot store as an
    exact instance of a type it is closed over is one whose second read it
    cannot vouch for, and the refusal names the type it refused."""
    with pytest.raises(ir.TranscriptionError) as exc:
        ir.JaxprEqn(primitive="add", invars=(), outvars=(),
                    params=(("thing", value),))
    assert name in str(exc.value), str(exc.value)
    assert ir._CANONICAL_RULE in str(exc.value), str(exc.value)


def test_a_subclass_of_an_ir_DATACLASS_is_refused_rather_than_rebuilt():
    """The one place a subclass is refused instead of read, and the reason
    is that there is no read that would settle it: a dataclass subclass can
    make ANY field a property, so `x.aval` is a fresh call every time and
    no single read of the object exists. No route produces one —
    `ir._decode` builds exact instances and so does
    `_jax_compat.Transcriber` — so refusing costs nothing."""
    class SneakyAval(ir.Aval):
        pass

    bad = SneakyAval(kind="ShapedArray", shape=(), dtype="float64")
    with pytest.raises(ir.TranscriptionError, match="SneakyAval"):
        ir.Var(id=1, aval=bad)


@pytest.mark.parametrize("base", sorted(ir._CANONICAL_IR_TYPES,
                                        key=lambda c: c.__name__),
                         ids=lambda c: c.__name__)
def test_an_ir_DATACLASS_subclass_is_refused_even_with_a_LYING_METACLASS(base):
    """AUDIT 0.2.0 B6 AUDIT 7. The refusal above rests on
    ``type(obj) in _CANONICAL_IR_TYPES`` being FALSE for a subclass, and
    that is a `frozenset` membership: three lines of metaclass answering as
    the base made it True, and the subclass was CARRIED by the arm whose
    whole justification is that the object canonicalized its own fields.

    Measured on `dff95fc`: an `ir.Var` subclass whose ``__getattribute__``
    counts reads of ``id`` was accepted by ``ir.JaxprEqn`` and handed every
    reader ``[1, 99, 99, 99]`` — which is precisely the hazard
    `Var.__post_init__`'s own comment names. Driven over EVERY frozen
    dataclass in `ir`, computed from the module, because the arm is
    computed from the module.
    """
    class M(type):
        def __hash__(cls):
            return hash(base)

        def __eq__(cls, other):
            return other is base

    liar = M(f"Evil{base.__name__}", (base,), {})
    assert liar in ir._CANONICAL_IR_TYPES, (
        "the metaclass no longer answers the frozenset membership, so this "
        "test is not driving the bypass it is named for"
    )
    assert liar is not base, "the bypass must be a DIFFERENT class"
    # the door decides by IDENTITY, so the lie buys nothing
    with pytest.raises(ir._NotCanonical):
        ir._canonical(object.__new__(liar))


@pytest.mark.parametrize("name,how", _LIAR_SPELLINGS,
                         ids=[n for n, _ in _LIAR_SPELLINGS])
def test_the_ORACLE_this_file_uses_is_NOT_the_doors_own_primitive(name, how):
    """THE INSTRUMENT, MEASURED — audit 0.2.0 B6 audit 7, and the reason
    it has a test of its own.

    Every other assertion in this file is about the door. This one is
    about the ORACLE those assertions use, because an instrument that
    shares a primitive with what it measures reports no defect for
    exactly the defect it exists to find.

    THE OLD ORACLE WAS BLIND TO ONE OF THE TWO BYPASSES AND NOT THE
    OTHER, and the rows say which. ``type(v) in <a set of types>`` reads
    ``type()`` honestly and then asks the METACLASS, so the metaclass
    liar walked past it; a ``__class__`` property lies to ``isinstance``
    and has nothing to say to a `set` membership on ``type(v)``, so that
    form was always visible to the oracle even while the door was storing
    it. Two bypasses, two repairs, and an instrument that catches one of
    them is not an instrument that catches the class.

    Without this row the repair is unattributable, because a liar
    document is refused before any oracle is consulted. That is a fine
    state of the world and a poor experiment."""
    liar = _liar(float, **how)
    if how["metaclass"]:
        assert type(liar) in {float}, (
            "the metaclass no longer satisfies a `set` membership on the "
            "type, so this row is not driving the primitive it names"
        )
    assert not _is_an_allowed_stored_type(liar), (
        "the oracle this file measures the door with accepts an object "
        "that merely CLAIMS to be a stored type — it is the door's own "
        "membership test again, and it cannot report on it"
    )


def test_the_ir_dataclass_arm_is_reached_by_IDENTITY_not_by_membership():
    """The same fact stated about the door's index rather than about one
    class: `ir._STORED_AS_IS` is keyed on ``id()``, which nothing can
    answer, and it holds each type as its VALUE so the address cannot be
    reused under it."""
    assert all(isinstance(k, int) for k in ir._STORED_AS_IS)
    assert all(id(t) == k for k, t in ir._STORED_AS_IS.items())


def test_the_doors_index_is_the_three_sets_it_MERGES():
    """The index is a cache, and a cache is a second copy — so it is
    re-derived here rather than trusted to have been rebuilt. It is also
    what makes merging the three arms into one lookup safe: the sets must
    be DISJOINT, or merging them would move which arm decides a type."""
    groups = (tuple(ir._CANONICAL_EXACT_TYPES), tuple(ir._CANONICAL_IR_TYPES),
              tuple(ir._LIBRARY_STORED_TYPES))
    declared = [t for g in groups for t in g]
    assert {id(t) for t in declared} == set(ir._STORED_AS_IS), (
        "`ir._STORED_AS_IS` is stale: something changed one of the three "
        "sets without calling `ir._rebuild_stored_index()`"
    )
    assert len({id(t) for t in declared}) == len(declared), (
        "the three sets the door merges overlap, so one lookup can no "
        "longer stand for three arms in a fixed order"
    )
    # and neither container arm is in it — those are decided AFTER it
    for c in ir._SHAPE_PARAM_CONTAINERS:
        assert id(c) not in ir._STORED_AS_IS


def test_the_types_with_NO_read_are_the_ones_that_cannot_be_subclassed():
    """AUDIT 0.2.0 B6 AUDIT 7. ``_CANONICAL_UNSUBCLASSABLE`` is the reason
    `bool` needs no read, and it is a claim about CPython — so it is
    measured rather than asserted. A type listed there that CAN be
    subclassed would silently make every subclass of it REFUSED rather
    than read, which is the narrowing `np.float64` and `np.str_` show the
    door cannot afford; and a type in both tables would be settled by
    identity before its read ever ran."""
    for t in ir._CANONICAL_UNSUBCLASSABLE:
        with pytest.raises(TypeError, match="not an acceptable base type"):
            type(f"Sub{t.__name__}", (t,), {})
    assert set(ir._CANONICAL_UNSUBCLASSABLE) <= set(ir._CANONICAL_EXACT)
    assert not (set(ir._CANONICAL_UNSUBCLASSABLE)
                & {b for b, _ in ir._CANONICAL_READS})


def test_NO_read_in_the_table_is_the_IDENTITY():
    """AUDIT 0.2.0 B6 AUDIT 7, and the whole of what the `bool` row was.

    An identity read is the one read that cannot detect having been
    applied to the wrong object, so a table entry that is the identity
    turns whatever reaches its arm into a stored value. Driven rather than
    read: each read is applied to a fresh instance of its base and must
    return something that is not that instance."""
    args = {int: (3,), float: (1.5,), complex: (1, 2), str: ("x",),
            bytes: (b"x",)}
    for base, _read in ir._CANONICAL_READS:
        assert not any(base is u for u in ir._CANONICAL_UNSUBCLASSABLE), (
            f"{base.__name__} cannot be subclassed, so its only possible "
            f"read is the identity and it may not be in this table at all; "
            f"it belongs in `_CANONICAL_UNSUBCLASSABLE`, which reads nothing"
        )
        assert base in args, f"no probe for a new read base {base.__name__}"
    for base, read in ir._CANONICAL_READS:
        probe = type(f"P{base.__name__}", (base,), {})(*args[base])
        out = read(probe)
        assert out is not probe, (
            f"the read for {base.__name__} returned the object it was "
            f"handed; an identity read stores whatever reaches its arm"
        )
        assert type(out) is base, (base, type(out))


def test_the_base_types_the_door_asks_have_no_ABCMeta_in_play():
    """``issubclass(type(obj), base)`` dispatches
    ``type(base).__subclasscheck__``, so the door's safety is a fact about
    the BASES' metaclasses. `ABCMeta` would consult ``__subclasshook__``
    and hand the derived class a say again."""
    for base in (tuple(ir._CANONICAL_EXACT_TYPES)
                 + tuple(b for b, _ in ir._CANONICAL_READS)
                 + tuple(ir._SHAPE_PARAM_CONTAINERS)):
        assert type(base) is type, (base, type(base))


def test_a_metaclass_that_FORGES_the_mro_cannot_forge_a_payload():
    """The one thing a derived class can still do to ``issubclass``, and
    why the accessor is what settles it — audit 0.2.0 B6 audit 7.

    A metaclass may override ``mro()``. CPython's own layout check refuses
    the result for every base whose instances have their own layout, which
    is every base this door asks ``issubclass`` against; the single
    exception among the types it stores is ``bool``, which shares
    ``int``'s layout, so a REAL ``int`` subclass can forge
    ``issubclass(cls, bool)``. Under the identity read this file removed,
    that object was STORED; under ``int.__index__`` it is stored as the
    exact ``int`` it actually carries, which is the truth about it."""
    for base in (tuple(b for b, _ in ir._CANONICAL_READS)
                 + tuple(ir._SHAPE_PARAM_CONTAINERS)):
        class M(type):
            def mro(cls, _b=base):
                return [cls, _b, object]

        with pytest.raises(TypeError, match="unsuitable layout"):
            M("Forged", (), {})

    class MBool(type):
        def mro(cls):
            return [cls, bool, int, object]

    forged = MBool("ForgedBool", (int,), {})
    assert issubclass(forged, bool), (
        "CPython no longer permits this forgery, so the identity read this "
        "test justifies removing has one fewer reason to be gone — the "
        "other reason (`isinstance` reads `__class__`) still stands"
    )
    stored = ir._canonical(forged(7))
    assert type(stored) is int and stored == 7, stored


def test_a_param_key_must_be_an_exact_str_and_an_entry_must_be_a_pair():
    """Two structural facts about `params` that every reader assumes and
    nothing checked. The pair one is not hypothetical: a 2-character `str`
    unpacks into a key and a value perfectly well, and `dict()` accepts a
    sequence of them."""
    with pytest.raises(ir.TranscriptionError, match="a param key is a NAME"):
        ir.JaxprEqn(primitive="add", invars=(), outvars=(), params=((7, 1),))
    for entry in (("a", 1, 2), ("a",), "xy"):
        with pytest.raises(ir.TranscriptionError,
                           match="is not a .key, value. pair"):
            ir.JaxprEqn(primitive="add", invars=(), outvars=(),
                        params=(entry,))


def test_the_reads_are_the_base_types_OWN_and_cannot_be_redirected():
    """THE MECHANISM, driven. Each canonicalizing read is the base type's
    own accessor, so an override cannot send it somewhere else — which is
    what makes "one read" also "the right read"."""
    class Hostile(str):
        def __str__(self):
            return "NOT THE PAYLOAD"

    class HostileTuple(tuple):
        def __iter__(self):
            return iter(("NOT THE PAYLOAD",))

        def __len__(self):
            return 99

        def __getitem__(self, k):
            return "NOT THE PAYLOAD"

    assert ir._canonical(Hostile("payload")) == "payload"
    assert ir._canonical(HostileTuple((1, 2))) == (1, 2)


@pytest.mark.parametrize("face", list(_LIAR_FACES),
                         ids=lambda t: t.__name__)
@pytest.mark.parametrize("name,how", _LIAR_SPELLINGS, ids=[n for n, _ in _LIAR_SPELLINGS])
def test_a_liar_is_refused_as_a_TranscriptionError_and_never_a_raw_TypeError(
        face, name, how):
    """AUDIT 0.2.0 B6 AUDIT 7, the FRAGILE half of the same finding.

    The door's reads were applied to objects that only CLAIMED the base
    type, so ``descriptor '__getitem__' requires a 'tuple' object but
    received a 'Liar'`` came out of `ir.JaxprEqn(...)` raw. That is not a
    tidiness point: `ir.TranscriptionError` SUBCLASSES `TypeError`, so
    ``except ir.TranscriptionError`` does not catch a raw `TypeError`, and
    the contract `tests/test_ir_message_totality.py` enforces — a public
    constructor raises this module's error or nothing — was broken at
    three sites that sweep cannot reach, because it injects SUBCLASSES and
    these objects are not subclasses of anything.

    Driven through the public constructor and caught only as
    `TranscriptionError`, which is the assertion: anything else propagates
    out of `pytest.raises` and reds."""
    liar = _liar(face, **how)
    with pytest.raises(ir.TranscriptionError):
        ir.JaxprEqn(primitive="add", invars=(), outvars=(),
                    params=(("thing", liar),))
    # the two positions `_validate_decl_eqn` owns, which are read BEFORE
    # the generic pass and had their own `isinstance` each
    with pytest.raises(ir.TranscriptionError):
        ir.JaxprEqn(primitive="stelling_any", invars=(),
                    outvars=(ir.Var(id=1, aval=av((2,))),),
                    params=(("dtype", "float64"), ("hi", 2.0), ("lo", 1.0),
                            ("shape", liar)))
    with pytest.raises(ir.TranscriptionError):
        ir.JaxprEqn(primitive="stelling_any", invars=(),
                    outvars=(ir.Var(id=1, aval=av((2,))),),
                    params=(("dtype", liar), ("hi", 2.0), ("lo", 1.0),
                            ("shape", (2,))))
    # and as a leaf under a container, where the recursion must keep its
    # own path back to the field rather than be swallowed by the read
    with pytest.raises(ir.TranscriptionError, match=r"\[0\]"):
        ir.JaxprEqn(primitive="add", invars=(), outvars=(),
                    params=(("thing", (liar,)),))


def test_the_dtype_param_must_BE_a_str_and_not_merely_pass_an_isinstance():
    """AUDIT 0.2.0 B6 AUDIT 7, a seventh member of the read-pair class in
    the guard that closed the fourth.

    `_validate_decl_eqn` gated the agreement check with
    ``isinstance(raw_dtype, str)``, so a `dtype` param that is any OTHER
    exact built-in did not fail the comparison — it SKIPPED it. Measured
    on `dff95fc`: ``b'float64'``, ``0``, ``64.0`` and ``('float64',)``
    were all ACCEPTED under a `float64` aval, while a `str` ``'int64'``
    that contradicts the aval was correctly refused. So *"a declaration's
    two self-descriptions of one declared set must agree"* held only when
    the param happened to be a `str`.

    The verdict does not move on any of the three spellings the audit
    drove, because `propagate._ieee_any` re-derives most of the haze from
    the AVAL — and that is a fact about how much of the model the param
    reaches, not a reason to leave a self-description unchecked. What it
    does reach is the subnormal band: `_ieee_any` selects it from
    ``str()`` of this param, and ``str(b'float64')`` is ``"b'float64'"``,
    which names no ieee format at all."""
    def decl(dtype_param):
        return ir.JaxprEqn(
            primitive="stelling_any", invars=(),
            outvars=(ir.Var(id=1, aval=av((2,), "float64")),),
            params=(("dtype", dtype_param), ("hi", 2.0), ("lo", 1.0),
                    ("shape", (2,))))

    # the honest spelling, and the honest subclass the trace path produces
    assert decl("float64").params_dict()["dtype"] == "float64"
    assert type(decl(_S("float64")).params_dict()["dtype"]) is str
    # the disagreement that was always caught
    with pytest.raises(ir.TranscriptionError, match="contradicts"):
        decl("int64")
    # and every non-`str` exact built-in the door happily STORES, which
    # used to skip the comparison entirely
    for bad in (b"float64", 0, 64.0, ("float64",), None, True, 1 + 2j):
        with pytest.raises(ir.TranscriptionError, match="a dtype is a NAME"):
            decl(bad)
    # the claim's own boundary, stated so it is not read as wider: a
    # declaration whose AVAL has no dtype has only one self-description,
    # and this function makes no claim about the param then
    ok = ir.JaxprEqn(
        primitive="stelling_any", invars=(),
        outvars=(ir.Var(id=1, aval=ir.Aval(kind="ShapedArray", shape=(2,),
                                           dtype=None)),),
        params=(("dtype", b"float64"), ("hi", 2.0), ("lo", 1.0),
                ("shape", (2,))))
    assert ok.params_dict()["dtype"] == b"float64"


def test_registering_a_stored_type_is_gated_on_the_property_it_delegates():
    """AUDIT 0.2.0 B6 AUDIT 7. `ir._register_stored_type` accepted any
    object and checked nothing about it, so ``_canonical(Wild()) is it``
    was two lines away for anything that can `import stelling.ir`.

    What registration delegates is single-valuedness, and the property
    that gives `interval.IntervalArray` it is that the class is a FROZEN
    dataclass — its fields are settled in its own ``__post_init__`` and
    cannot be reassigned. That is checkable, it is the same property
    `ir._CANONICAL_IR_TYPES` is computed with, and it is checked now.

    IT IS STILL NOT A SECURITY BOUNDARY and `ir.py` says so where the
    function is: code that can call this can equally rebind
    `ir._canonical`. The boundary the door defends is a DOCUMENT, and no
    document reaches this arm — `_decode` has no tag for a registered type
    and `_encode` refuses to encode one."""
    saved = ir._LIBRARY_STORED_TYPES
    try:
        class Wild:
            pass

        with pytest.raises(TypeError, match="not a frozen dataclass"):
            ir._register_stored_type(Wild)

        @dataclasses.dataclass
        class Mutable:
            a: int = 0

        with pytest.raises(TypeError, match="not a frozen dataclass"):
            ir._register_stored_type(Mutable)

        with pytest.raises(TypeError, match="takes a class"):
            ir._register_stored_type(Wild())

        for known in (str, ir.Var):
            with pytest.raises(TypeError, match="already a type this module"):
                ir._register_stored_type(known)

        # and the one shape it does accept, with the index rebuilt for it
        @dataclasses.dataclass(frozen=True)
        class Boxed:
            a: int = 0

        ir._register_stored_type(Boxed)
        assert id(Boxed) in ir._STORED_AS_IS
        b = Boxed()
        assert ir._canonical(b) is b
    finally:
        ir._LIBRARY_STORED_TYPES = saved
        ir._rebuild_stored_index()
    assert ir._LIBRARY_STORED_TYPES is saved
    assert set(ir._STORED_AS_IS) == {
        id(t) for t in (tuple(ir._CANONICAL_EXACT_TYPES)
                        + tuple(ir._CANONICAL_IR_TYPES)
                        + tuple(saved))
    }


def test_a_list_is_stored_as_a_tuple_and_the_document_now_hashes():
    """An exact `list` is an exact built-in and is still two answers
    waiting to happen, because it is MUTABLE. It is also a value `_encode`
    has no arm for, so a `list` param that survived the door produced IR
    `content_hash()` could not serialize at all. Storing it as a `tuple`
    closes both, and the trace path already collapses list to tuple."""
    holder = [0, 1]
    eqn = ir.JaxprEqn(primitive="reduce_sum", invars=(), outvars=(),
                      params=(("axes", holder), ("out_sharding", None)))
    holder.append(2)  # the document's own reference, mutated after the door
    assert eqn.params_dict()["axes"] == (0, 1)
    assert type(eqn.params_dict()["axes"]) is tuple
    # and it serializes, which the `list` spelling did not
    q = ir.ClosedJaxpr(jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=(),
                                      eqns=(eqn,)))
    assert ir.ClosedJaxpr.from_dict(q.to_dict()) == q
    assert len(q.content_hash()) == 64


# ---------------------------------------------------------------------------
# what it costs a legitimate document
# ---------------------------------------------------------------------------

def test_the_from_dict_and_hand_built_routes_are_unchanged():
    """Measured rather than assumed. `from_dict` keys come from JSON object
    entries and its values from `_decode`, all exact already, so the door
    is a type check per value and nothing moves."""
    doc = _canonical_document()
    d = doc.to_dict()
    loaded = ir.ClosedJaxpr.from_dict(d)
    assert loaded == doc
    assert loaded.to_dict() == d
    assert loaded.content_hash() == doc.content_hash()


def test_the_traced_route_pays_nothing_and_needs_the_subclass_arm():
    """The route the SUBCLASS arm exists for. `_jax_compat.Transcriber.param`
    returns a `np.float64` unchanged from its
    ``isinstance(v, (int, float, complex))`` arm and a `np.str_` unchanged
    from its ``isinstance(v, (bool, str))`` arm, and both ARE subclasses of
    a stored type — so a door that refused subclasses outright would refuse
    traced queries. It canonicalizes them instead, and the traced document
    round-trips to the same `to_dict` and the same `content_hash`."""
    np = pytest.importorskip("numpy")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_, trace

    assert issubclass(np.float64, float)
    assert issubclass(np.str_, str)

    def h():
        x = any_array((2,), "float64", (1.0, 2.0))
        assert_(jnp.sum(x) <= 4.5)

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        q = trace(h)
    finally:
        jax.config.update("jax_enable_x64", old)

    wrong = [(p, type(v).__name__) for p, v in _stored_values(q)
             if not _is_an_allowed_stored_type(v)]
    assert not wrong, wrong
    assert ir.ClosedJaxpr.from_dict(q.to_dict()) == q
    assert ir.ClosedJaxpr.from_dict(q.to_dict()).to_dict() == q.to_dict()
    assert ir.ClosedJaxpr.from_dict(q.to_dict()).content_hash() == \
        q.content_hash()

    # ... and the arm really is exercised by values of those two types
    assert ir._canonical(np.float64(1.5)) == 1.5
    assert type(ir._canonical(np.float64(1.5))) is float
    assert type(ir._canonical(np.str_("f8"))) is str


# ---------------------------------------------------------------------------
# the commitment list, written once and witnessed
# ---------------------------------------------------------------------------

def _witness_order():
    eqn = ir.JaxprEqn(primitive="add", invars=(), outvars=(),
                      params=(("z", 1), ("a", 2)), effects=("b", "a"))
    return (eqn.params, eqn.effects), ((("a", 2), ("z", 1)), ("a", "b"))


def _witness_hash_scope():
    def build(src):
        return ir.ClosedJaxpr(jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=(),
            eqns=(ir.JaxprEqn(primitive="add", invars=(), outvars=(),
                              source_info=src),)))
    a, b = build(("one.py:1",)), build(("other.py:99",))
    assert a.to_dict() != b.to_dict()
    return a.content_hash(), b.content_hash()


def _witness_extents():
    def build(shape):
        return ir.ClosedJaxpr(jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=(),
            eqns=(ir.JaxprEqn(
                primitive="stelling_any", invars=(),
                outvars=(ir.Var(1, av((1,))),),
                params=(("dtype", "float64"), ("hi", 2.0), ("lo", 1.0),
                        ("shape", shape))),)))
    return build([True]).content_hash(), build((1,)).content_hash()


def _witness_value_types():
    class Key(str):
        def __eq__(self, other):
            return False           # answers nothing the way it reads

        def __hash__(self):
            return str.__hash__(self)

    eqn = ir.JaxprEqn(primitive="add", invars=(), outvars=(),
                      params=((Key("axes"), [1, 2]),))
    return eqn.params, (("axes", (1, 2)),)


def _witness_alpha_renaming():
    # this module transcribes the ids it is handed; the renaming that makes
    # alpha-variants equal happens at transcription, so the witness here is
    # the transcription of one structure twice
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_, trace

    def h():
        x = any_array((2,), "float64", (0.0, 1.0))
        assert_(jnp.sum(x) <= 3.0)

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return trace(h).content_hash(), trace(h).content_hash()
    finally:
        jax.config.update("jax_enable_x64", old)


def _one_param_document(key, value):
    """The smallest document that carries one param value, for the two
    witnesses below whose collapse happens in the READER and not in a
    constructor."""
    return {
        "k": "closed",
        "jaxpr": {
            "k": "jaxpr", "constvars": [], "invars": [], "outvars": [],
            "eqns": [{"k": "eqn", "primitive": "add", "invars": [],
                      "outvars": [], "params": [[key, value]],
                      "effects": []}],
            "effects": [],
        },
        "consts": [],
    }


def _witness_complex_parts():
    """AUDIT 0.2.0 B12. `<complex>.re`/`.im` are read as ONE number, so a
    part's integer spelling and its `float` spelling are one document.

    Driven through `from_dict`, because the collapse is the READER's: a
    python `complex` has `float` parts already, and it is the two JSON
    spellings that differ."""
    def build(re, im):
        return ir.ClosedJaxpr.from_dict(
            _one_param_document("c", {"k": "complex", "re": re, "im": im})
        )
    a, b = build(1, 0), build(1.0, 0.0)
    assert a.to_dict() == b.to_dict(), (a.to_dict(), b.to_dict())
    return a.content_hash(), b.content_hash()


def _witness_array_payload_spelling():
    """AUDIT 0.2.0 B12. An `Array`'s `data` is base64 TEXT stored as the
    BYTES it denotes, so two spellings of one byte string are one document:
    base64's trailing bits are not part of the value it encodes, and neither
    `validate=True` nor `binascii`'s `strict_mode=True` treats them as part
    of it."""
    assert base64.b64decode("AB==", validate=True) == b"\x00"
    assert base64.b64decode("AC==", validate=True) == b"\x00"
    assert binascii.a2b_base64("AC==", strict_mode=True) == b"\x00"

    def build(text):
        return ir.ClosedJaxpr.from_dict(
            _one_param_document(
                "a", {"k": "array", "dtype": "|u1", "shape": [1],
                      "data": text}
            )
        )
    a, b = build("AB=="), build("AC==")
    assert a.to_dict() == b.to_dict(), (a.to_dict(), b.to_dict())
    return a.content_hash(), b.content_hash()


_WITNESSES = {
    "alpha-renaming": _witness_alpha_renaming,
    "param and effect order": _witness_order,
    "hash scope": _witness_hash_scope,
    "shape extents": _witness_extents,
    "stored value types": _witness_value_types,
    "complex parts": _witness_complex_parts,
    "array payload spelling": _witness_array_payload_spelling,
}


def test_the_canonicalization_list_is_written_once_and_the_prose_points_at_it():
    """AUDIT 0.2.0 B6 AUDIT 6, F4. The module docstring used to BE the
    list, introduced as "two commitments ... plus one hash-scope decision"
    and closed with "Nothing here licenses more"; by the time it was read
    the sentence had three entries and the code had five, because two
    commits had added one each and no test held either sentence.

    The list is `ir.CANONICALIZATIONS` now, and the docstring cites it. A
    docstring that restates an entry is the defect this test exists for,
    so it may not contain one."""
    doc = ir.__doc__
    assert "CANONICALIZATIONS" in doc, doc[:400]
    for name, what in ir.CANONICALIZATIONS:
        assert what not in doc, (
            f"the module docstring restates the {name!r} entry; two copies "
            f"of a commitment are two commitments"
        )


# The prose this rule is about, and until 2026-08-22 it was five markdown
# files and nothing else. `src/` SHIPS PROSE TOO — this repository's modules
# are more comment than code by design — and a 16-hex literal planted in
# `src/stelling/ir.py`, in the very paragraph explaining an earlier literal's
# removal, left the whole suite green while the same literal planted in
# `SOUNDNESS.md` reddened. That is how a pair of `content_hash` literals came
# to stand in `ir.py` for a day inside the paragraph whose entire subject is
# that a figure nobody can re-derive is not a measurement. A rule applied
# where somebody was told and nowhere else is a rule with a hole in the shape
# of wherever nobody looked.
_HASH_LITERAL_RECORDS = ("CHANGELOG.md", "SOUNDNESS.md", "README.md",
                         "ARCHITECTURE.md", "DOCUMENTATION_ARCHITECTURE.md")

# THE TRUNCATED LITERALS THIS TREE CARRIES, FROZEN — AND FROZEN WITH THE
# SENTENCE EACH ONE STANDS IN, which is the half that was missing. The
# exact-length rule below (16 or 64 hex) matches NOTHING anywhere in this
# repository — measured — and the form that actually lands is eight hex
# digits and a cut mark, which is this tree's own spelling for a hash prefix
# and is unrecomputable BY CONSTRUCTION: a reader cannot check digits that
# have been cut off. So the truncated form is scanned too, and because these
# tokens predate the rule it is a RATCHET rather than a ban: the inventory
# found must equal this one exactly, so a new literal anywhere in scope reds,
# and removing one reds until it is struck from here.
#
# **THE KEY IS `(file, digits, SENTENCE)` AND IT USED TO BE `(file, digits)`,
# WHICH IS BLIND TO REPURPOSE-IN-PLACE.** Nothing keyed on WHERE a literal
# stood or on what the sentence around it said, so the same digits could be
# turned from a retraction into an assertion underneath the ratchet without
# moving it. Two measured, both at `107 passed`:
#
#   * `src/stelling/ir.py`'s ``and it is not ``64a0ce8d…``` reworded to
#     ``and it IS ``64a0ce8d…``` — the sentence that exists to withdraw the
#     figure now pins it, and the inventory still read `("ir.py",
#     "64a0ce8d"): 3`;
#   * one of `SOUNDNESS.md`'s two occurrences relocated out of its retraction
#     into a fresh claim elsewhere in the file. Count unchanged, meaning
#     reversed.
#
# A retraction that can become an unreproducible assertion without the guard
# moving is the guard measuring the wrong thing, and the fix is the same one
# `tests/_soundness_routing_manifest.py` makes with `not_carried`: write the
# lines out, in this file, where a reviewer meets them.
#
# Each entry, and why it is here rather than fixed:
#
#   `64a0ce8d…` x2 in `SOUNDNESS.md`, x3 in `src/stelling/ir.py` — the
#     RETRACTION of that literal, in both places that retract it. A
#     retraction has to be able to name what it retracts, and every one of
#     those sentences exists to say the figure was unreproducible. (One of
#     the three is `` ``content_hash 64a0ce8d…`` ``, and a pattern anchored
#     on the backtick could not see it — see `_BACKTICKED`.)
#   `3c15c4f5…`, `4dc49e99…` (`SOUNDNESS.md`) — query hashes pinned as
#     "byte-identical at base and tip on both series" in the 0.11-bump entry.
#     These ARE the defect this rule names: no document is published with
#     them. Owed, not exempted.
#   `628a25ef…`, `f32f4860…` (`SOUNDNESS.md`) — the quickstart's query hash
#     on the two series. `tests/test_doc_examples.py` runs that block, so
#     unlike the two above there IS a document; the prose does not say so.
#   `2896a0f2…` x2 in `SOUNDNESS.md`, x2 in `src/stelling/smt.py`, x2 in
#     `src/stelling/verdict.py`, with `2de5e041…` and `2f2e0ed8…` once each
#     in `SOUNDNESS.md` — EIGHT that were inside this scan's own scope and
#     that its grammar could not see, four of them under `src/`. They are
#     the emitted-script shas of the scatter SET collision table, and they
#     are the same class as the `3c15c4f5…` / `4dc49e99…` pair above:
#     measured figures with a verdict attached and no published script to
#     check them against — owed rather than merely uncited, which is what
#     separates them from the quickstart pair. What
#     the table CLAIMS a reader can re-derive — that a scatter-bearing slice
#     and a scatter-free slice emit byte for byte the same script — is the
#     equality of two of them and not any one digit, which is why the rows
#     read `collides` / `no collision` rather than reading as four values.
#     Owed, not exempted, and now visible.
#
# THE INVENTORY IS ALSO THE REACH CONTROL. SEVEN of its seventeen entries
# come from `src/`, so a scan that stopped seeing `src/` could not match it —
# the widening cannot silently un-widen, which is the failure the five
# hand-listed files had no guard against.
#
# **AND THE THIRD FIELD WAS A LINE, WHICH IS SMALLER THAN A CLAIM.** It was
# `" ".join(line.split())` — the ONE line the digits stand on — and a
# sentence does not end where a line does. THIRD DEFEAT, measured at
# `4a13824`: the line of `src/stelling/ir.py` that reads *"return the
# identical hash, and it is not ``64a0ce8d…``. What a"* left BYTE-IDENTICAL
# and the three lines after it
# rewritten from *"what a reader can re-derive is the PROPERTY … the hash is
# not needed"* into *"what a reader CAN re-derive is that very hash … it is a
# measurement and is retained deliberately"*. Whole zero-dep suite
# `2175 passed, 149 skipped`, identical to clean; ratchet unmoved. The
# retraction had become an assertion with the literal's own line untouched.
#
# The same defeat inside `SOUNDNESS.md` IS caught — by
# `tests/test_soundness_routing.py`, which hashes whole ROUTED BLOCKS — so
# the exposure is exactly where a literal's surrounding lines are outside a
# routed block, and shipped `src/` comments are the main such place. So the
# fix is the routed-block one, applied here: the entry now carries the
# sha256 of the whole PROSE BLOCK the literal stands in — the paragraph, in
# markdown; the run of comment lines between two `#`-only separators, in
# python — and a rewording anywhere in it moves the entry. The readable
# `sentence` stays, because a sha nobody can read is a diff nobody can
# review, and `_prose_block_text` prints the block beside any entry that
# moved.
#
# **AND THE BLOCK'S BOUNDARY HAS A CONSEQUENCE, WRITTEN DOWN HERE RATHER
# THAN WIDENED AWAY.** A block ends at a `#`-only separator on purpose:
# `test_the_ratchets_MEMORY_is_the_BLOCK_and_not_the_LINE` asserts that a
# rewrite on the far side of one does NOT move an entry, because otherwise
# every entry would be a claim about its whole file. So a FOURTH repurpose
# exists, is permitted by construction, and is not a hole to be closed by
# enlarging the unit. Two variants, both driven, both green at
# `2190 passed, 149 skipped` with the inventory unmoved:
#
#   * a NEW comment paragraph one separator BELOW the tracked block --
#     naming no digits, so nothing new enters the scan -- saying the
#     withdrawn figure *"is reproducible after all and is retained as a
#     measurement"*. The retraction stands untouched and is overturned by
#     prose the ratchet's memory does not reach.
#   * editing the harness the retraction POINTS AT, so its own
#     justification -- *"the document is above, so the hash is not
#     needed"* -- no longer names the document it claims. The document is
#     in a different block; the sentence that leans on it is in this one.
#
# **THE CONSEQUENCE, STATED: a tracked retraction can be overturned by
# prose one separator away, and EVERY CROSS-BLOCK REFERENCE A TRACKED
# BLOCK MAKES IS OUTSIDE ITS OWN MEMORY.** A block sha is a claim about one
# paragraph and about nothing that paragraph refers to. What this ratchet
# delivers, exactly: a literal's own sentence and its own paragraph cannot
# be repurposed without the entry moving. What it does not and by
# construction cannot deliver: that the paragraph is still true. A
# retraction that has to stand alone should carry inside its own block
# everything it rests on, and one that points elsewhere is pointing at
# unwatched ground -- which is a fact about this instrument, said here,
# and not a defect to be fixed by making it measure a whole file.
class _Truncated(typing.NamedTuple):
    """One truncated hash literal: where it is, what it is, what the line it
    stands in SAYS about it, and the sha256 of the whole prose BLOCK that
    line belongs to. The last two fields are the ratchet's memory — see the
    block above for the three repurposes that moved nothing without them,
    the third of which the `sentence` field alone could not see."""

    path: str
    digits: str
    sentence: str
    block: str


_TRUNCATED_HASH_LITERALS = (
    _Truncated('SOUNDNESS.md', '2896a0f2',
               "slice of `s[1] - x[1] <= 0` barred ('scatter',) sha 2896a0f2…",
               'fc065777de332d4aa90406dfff86d7929541518c5a6975f7f5937a4a4f478cbd'),
    _Truncated('SOUNDNESS.md', '2896a0f2',
               'slice of `x[1] - x[1] <= 0` barred () sha 2896a0f2… collides',
               'fc065777de332d4aa90406dfff86d7929541518c5a6975f7f5937a4a4f478cbd'),
    _Truncated('SOUNDNESS.md', '2de5e041',
               "slice of `s[0] - x[0] >= 0` barred ('scatter',) sha 2de5e041…",
               'fc065777de332d4aa90406dfff86d7929541518c5a6975f7f5937a4a4f478cbd'),
    _Truncated('SOUNDNESS.md', '2f2e0ed8',
               'slice of `x[0] - x[0] >= 0` barred () sha 2f2e0ed8… no collision',
               'fc065777de332d4aa90406dfff86d7929541518c5a6975f7f5937a4a4f478cbd'),
    _Truncated('SOUNDNESS.md', '3c15c4f5',
               '(`3c15c4f5…` / `4dc49e99…`), the 0.11 arm behaviourally the old code over',
               'b82ce3569d1174a56257835c0896b985d30bedc711e302e9f4d6a54bf5a927ac'),
    _Truncated('SOUNDNESS.md', '4dc49e99',
               '(`3c15c4f5…` / `4dc49e99…`), the 0.11 arm behaviourally the old code over',
               'b82ce3569d1174a56257835c0896b985d30bedc711e302e9f4d6a54bf5a927ac'),
    _Truncated('SOUNDNESS.md', '628a25ef',
               "0.11 query hash (proved — the rewrite turns the quickstart's `628a25ef…`",
               'b82ce3569d1174a56257835c0896b985d30bedc711e302e9f4d6a54bf5a927ac'),
    _Truncated('SOUNDNESS.md', 'f32f4860',
               'into exactly the 0.10 `f32f4860…`), and `Inline` has 5 members of which 4',
               'b82ce3569d1174a56257835c0896b985d30bedc711e302e9f4d6a54bf5a927ac'),
    _Truncated('SOUNDNESS.md', '64a0ce8d',
               'pinned a hash literal (`64a0ce8d…`) instead, and a hash literal is',
               'd5e6502f637daeb2132a4aedc71afc98ed240813cf876c9345808b556c287a46'),
    _Truncated('SOUNDNESS.md', '64a0ce8d',
               'driven at `198a2b5` and none reproduces `64a0ce8d…`. The refusal is right —',
               'd5e6502f637daeb2132a4aedc71afc98ed240813cf876c9345808b556c287a46'),
    _Truncated('src/stelling/ir.py', '64a0ce8d',
               '# ``64a0ce8d…``: the document is above, and the hash is not needed.',
               'f8f92f5e1bcedfba11cdb76fd1dc3396892ddb23e422e35e59d281e33abf1eb6'),
    _Truncated('src/stelling/ir.py', '64a0ce8d',
               '# comment read ``content_hash 64a0ce8d…`` until 2026-08-21, and',
               '18e2edea332c631147fbce621dfd91aeb9c4e7663b9425b4d4d740aa713025b1'),
    _Truncated('src/stelling/ir.py', '64a0ce8d',
               '# return the identical hash, and it is not ``64a0ce8d…``. What a',
               '18e2edea332c631147fbce621dfd91aeb9c4e7663b9425b4d4d740aa713025b1'),
    _Truncated('src/stelling/smt.py', '2896a0f2',
               "assert s[1] - x[1] <= 0 slice barred ('scatter',) sha 2896a0f2…",
               'f4dbb4fb877043dcd58ec2486197435db638bab37ebfd69934269e54b1b24bee'),
    _Truncated('src/stelling/smt.py', '2896a0f2',
               'assert x[1] - x[1] <= 0 slice barred () sha 2896a0f2…',
               'f4dbb4fb877043dcd58ec2486197435db638bab37ebfd69934269e54b1b24bee'),
    _Truncated('src/stelling/verdict.py', '2896a0f2',
               "assert s[1] - x[1] <= 0 slice barred ('scatter',) sha 2896a0f2…",
               'f375a95b358f3a6a7fda74cd6240afbee94d8b1319ff391bff4bc109094abe22'),
    _Truncated('src/stelling/verdict.py', '2896a0f2',
               'assert x[1] - x[1] <= 0 slice barred () sha 2896a0f2…',
               'f375a95b358f3a6a7fda74cd6240afbee94d8b1319ff391bff4bc109094abe22'),
)

#: A backticked span, RST's double form first so ``x`` is one span and not
#: two empty ones. THE HEX NEED NOT TOUCH THE BACKTICK, and requiring it to
#: was a hole the plant test below found in this very commit: the literals
#: that landed in `src/stelling/ir.py` were written ``content_hash a037be6d…``
#: — the digits are a word inside the span, not the whole of it, and a
#: pattern anchored on the delimiter walks straight past the only spelling
#: the defect has ever actually taken.
_BACKTICKED = re.compile(r"``([^`]+)``|`([^`\n]+)`")

#: Exactly 16 or 64 hex characters, as a word. A git SHA is 7 or 40 and does
#: not match; a longer hex run does not match a 16-character window inside
#: itself; the decimal digit runs these files are full of
#: (`4028234663852886e38`) are 19 and do not match either.
#:
#: **THE `.` IN THE LOOKBEHIND IS A DOUBLE'S FRACTION, AND IT WAS A LATENT
#: FALSE POSITIVE.** `2.220446049250313e-16` carries a 16-character run of
#: `[0-9a-f]` with a letter in it — `220446049250313e` — and the rule fired
#: on it, so any backticked 17-significant-digit double would have been
#: reported as *"the record pins a content-hash literal in prose"*. The tree
#: was safe only because those floats happen not to be backticked. Measured:
#: with the guard, that float yields nothing; without it, `['220446049250313e']`.
_HASH_LITERAL_EXACT = re.compile(
    r"(?<![0-9A-Za-z.])([0-9a-fA-F]{16}|[0-9a-fA-F]{64})(?![0-9A-Za-z])"
)

#: ...and a hash PREFIX: eight or more hex digits WITH THE CUT MARKED. The
#: cut mark is what makes this unrecomputable by construction and it is also
#: what keeps a git short sha out without asking git: `dff95fc` and
#: `a1521744` are written whole because they ARE whole. A hash prefix with no
#: cut mark is outside this rule and is said so rather than silently missed.
#:
#: TWO MORE SPELLINGS OF THE CUT THAT THE `…`/`...` ALTERNATION COULD NOT
#: SEE, and one of them is in this repository: a PYTHON SOURCE ESCAPE. The
#: FOURTH `64a0ce8d` prefix in the tree stands in
#: `tests/_soundness_routing_manifest.py`, and its cut is spelled as the
#: six characters `\` `u` `2` `0` `2` `6`, because that file quotes a routed
#: line as a Python string literal. A scan reading the SOURCE sees the escape
#: and not the character. (That file is outside this scan's scope, which is
#: the shipped record and the shipped modules — but the escape is not: a
#: docstring under `src/` may be written the same way.) An UPPERCASE prefix
#: and a space before the cut were invisible for the same reason: the
#: grammar named one spelling of a thing that has several.
_HASH_LITERAL_TRUNCATED = re.compile(
    r"(?<![0-9A-Za-z.])([0-9a-fA-F]{8,})[ \t]?"
    r"(?:\.\.\.|…|\\u2026|\\N\{HORIZONTAL ELLIPSIS\})"
)

#: `(pattern, must it stand inside a backticked span)`. THE SPAN FILTER IS
#: NOT THE SAME BARGAIN FOR THE TWO RULES, and applying it to both is what
#: hid eight literals inside this scan's own scope. For the EXACT rule the
#: span is the only thing separating a digest from any hex-looking word in a
#: sentence, so it stays. For the TRUNCATED rule the CUT MARK already does
#: that work — nobody writes `2896a0f2…` about anything but a digest — and
#: requiring backticks as well meant the scan could not see a measured
#: figure sitting in a plain-text table, which is exactly where measured
#: figures sit. Driven: the eight in `SOUNDNESS.md`, `src/stelling/smt.py`
#: and `src/stelling/verdict.py` are invisible with the filter and inventoried
#: without it.
_HASH_LITERAL_RULES = {
    "exact": (_HASH_LITERAL_EXACT, True),
    "truncated": (_HASH_LITERAL_TRUNCATED, False),
}


def _hash_literals(rule, text):
    """Every match of ``rule``'s pattern in ``text``.

    Two filters, and each has been the difference between a hit and a miss.
    The span filter — for the exact rule only, see `_HASH_LITERAL_RULES` —
    is what makes that rule about typeset figures rather than about any
    hex-looking word in a sentence. The letter test is what keeps a decimal
    run out: a token of pure digits is a number somebody wrote, not a
    digest.
    """
    pattern, inside_a_span = _HASH_LITERAL_RULES[rule]
    regions = []
    if inside_a_span:
        for span in _BACKTICKED.finditer(text):
            regions.append(span.group(1) or span.group(2))
    else:
        regions.append(text)
    found = []
    for region in regions:
        found += [
            m.group(1) for m in pattern.finditer(region)
            if any(c in "abcdefABCDEF" for c in m.group(1))
        ]
    return found


def _hash_literal_scan_paths(root):
    """The five record files and every shipped module, in that order."""
    return [root / name for name in _HASH_LITERAL_RECORDS] + sorted(
        root.glob("src/**/*.py")
    )


def _prose_line(line: str, python: bool) -> str:
    """One line, stripped of its indentation and of a python comment marker.

    A ``#`` is a comment marker in a `.py` file and a heading in markdown, so
    which files it is stripped from is decided by the caller and not guessed
    at from the character.
    """
    body = line.strip()
    if python and body.startswith("#"):
        body = body[1:].strip()
    return " ".join(body.split())


def _prose_block_text(lines, index: int, python: bool) -> str:
    """The BLOCK of prose the line at ``index`` belongs to, normalised.

    THE UNIT IS THE PARAGRAPH, and it is the paragraph because a SENTENCE is
    not big enough. The third defeat of this ratchet left the literal's own
    line byte-identical and rewrote the three lines AFTER it — which are the
    next sentences of the same paragraph — turning a retraction into an
    assertion. A grammatical sentence would not have caught that; the block
    it sits in does.

    One rule serves markdown and python because after `_prose_line` they are
    the same shape: a run of non-empty lines, bounded by a blank line, by a
    `#`-only comment separator, or by the code around the comment. That is
    the paragraph in a `.md` file and the comment paragraph in a `.py` one,
    which is the unit each is actually written and reviewed in.
    """
    lo = hi = index
    while lo > 0 and _prose_line(lines[lo - 1], python):
        lo -= 1
    while hi + 1 < len(lines) and _prose_line(lines[hi + 1], python):
        hi += 1
    return " ".join(
        text for text in (_prose_line(line, python) for line in lines[lo:hi + 1])
        if text
    )


def _prose_block(lines, index: int, python: bool) -> str:
    """:func:`_prose_block_text`, as the sha256 the inventory freezes."""
    return hashlib.sha256(
        _prose_block_text(lines, index, python).encode("utf-8")
    ).hexdigest()


def test_the_record_does_not_pin_a_hash_LITERAL_in_prose():
    """AUDIT 0.2.0 B6 AUDIT 6, F5. `CHANGELOG.md` pinned a 16-character
    `content_hash()` literal for "the canonical `shape=(2,)` declaration",
    and the literal matched neither of the two documents that phrase can
    name — so the sentence's PROPERTY (the fix moves no hash) verified
    while its figure was wrong from the day it shipped, and nothing could
    have told a reader.

    A hash in PROSE is a figure no reader can recompute and no test holds.
    A hash inside an executed doc example is a different thing entirely —
    `tests/test_doc_examples.py` runs the block and compares the output —
    so this scan is over the shipped record and the shipped modules, and it
    is exact about what it looks for. See `_HASH_LITERAL_RECORDS` for the
    scope this had until 2026-08-22 and what a plant in `src/` measured, and
    `_TRUNCATED_HASH_LITERALS` for the eight-hex-and-a-cut form the
    exact-length rule alone never could have caught — and for the SENTENCE
    each one is frozen with, which is what stops a retraction being reworded
    into an assertion underneath the count."""
    root = pathlib.Path(__file__).resolve().parent.parent
    # ABSENCE IS NOT COMPLIANCE, and this guard used to `continue` past a
    # missing file. Driven: move `CHANGELOG.md`, `SOUNDNESS.md` and
    # `README.md` out of the tree and it reported `1 passed` — a scan over
    # nothing, reading as a clean record. It polices the very edit batch
    # B8c declared (`SF-0.2.0-59`: an unreproducible hash literal removed
    # from the record), so "the files are not here" has to be the loudest
    # thing it can say and not the quietest.
    missing = [
        name for name in _HASH_LITERAL_RECORDS if not (root / name).is_file()
    ]
    assert not missing, (
        f"the record files {missing} are not in this tree, so this scan "
        f"read no prose at all. A hash-literal check that passes because "
        f"there is nothing to check is the shape this campaign keeps "
        f"closing: it would go green on a tree that had deleted the record "
        f"the literal was removed from."
    )
    paths = _hash_literal_scan_paths(root)
    modules = [p for p in paths if p.suffix == ".py"]
    assert modules, (
        "this scan found no module under `src/` to read, so its widening to "
        "shipped code is doing nothing — which is the state it was in when a "
        "pair of `content_hash` literals stood in `src/stelling/ir.py` for a "
        "day with the whole suite green"
    )

    found = []
    inventory = collections.Counter()
    blocks: dict = {}
    for path in paths:
        name = path.relative_to(root).as_posix()
        python = path.suffix == ".py"
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            for digits in _hash_literals("exact", line):
                found.append(f"{name}:{i}: `{digits}`")
            for digits in _hash_literals("truncated", line):
                # THE BLOCK, NOT THE LINE. See `_prose_block_text` and the
                # third defeat above `_Truncated`: the literal's own line
                # stayed byte-identical while the sentences after it turned
                # its retraction into an assertion.
                entry = _Truncated(name, digits, " ".join(line.split()),
                                   _prose_block(lines, i - 1, python))
                inventory[entry] += 1
                blocks[entry] = _prose_block_text(lines, i - 1, python)
    assert not found, (
        "the record pins a content-hash literal in prose, which no reader "
        "can recompute and no test holds:\n  " + "\n  ".join(found)
    )
    frozen = collections.Counter(_TRUNCATED_HASH_LITERALS)
    added = sorted(set(inventory) - set(frozen))
    gone = sorted(set(frozen) - set(inventory))
    moved = sorted(
        k for k in set(inventory) & set(frozen) if inventory[k] != frozen[k]
    )
    assert inventory == frozen, (
        f"the truncated hash literals in the shipped record and modules are "
        f"no longer the frozen inventory. Added: {added}; gone: {gone}; "
        f"counts that moved: {moved}. The prose block each ADDED entry now "
        f"stands in, which is what its `block` sha is over: "
        + "; ".join(f"{e.path}: {blocks[e]!r}" for e in added) + ". "
        f"A hash prefix is a figure a reader cannot even partly check; if a "
        f"new one is right, publish the document beside it and strike the "
        f"digits, and if an old one has gone, take it out of the inventory "
        f"in the same commit. **AND THE SENTENCE IS PART OF THE ENTRY**: a "
        f"literal whose line has been reworded shows up as one gone and one "
        f"added, deliberately, because that is how a retraction turns into "
        f"an assertion without the count moving. **AND SO IS THE WHOLE PROSE "
        f"BLOCK**, by its sha: the third defeat left the literal's own line "
        f"byte-identical and reworded the three lines after it, which is the "
        f"same repurpose one sentence further out. Re-word the entry here in "
        f"the same commit and say which of the two it now is."
    )


def test_the_hash_literal_scan_SEES_A_PLANT_IN_EVERY_FORM_IT_CLAIMS():
    """The scan above is an absence claim, and an absence claim needs a plant.

    Every form is driven against the finder rather than asserted about it:
    the two lengths the exact rule names, the truncated form that landed in
    `src/stelling/ir.py`, the two escape spellings of the cut, the space
    before it and the UPPERCASE prefix — all invisible until 2026-08-23 —
    the plain-text table row
    that no backtick reaches — and the four things that must NOT match: a
    git short sha, a git long sha, a decimal run, and a 17-significant-digit
    double, which the exact rule really did fire on.
    """
    for text, exact, trunc in [
        ("`a037be6dfc050ff5`", ["a037be6dfc050ff5"], []),
        ("`" + "a" * 64 + "`", ["a" * 64], []),
        ("`a037be6dfc050ff5…`", ["a037be6dfc050ff5"], ["a037be6dfc050ff5"]),
        ("`a037be6d…`", [], ["a037be6d"]),
        ("`a037be6d...`", [], ["a037be6d"]),
        ("``content_hash a037be6d…``", [], ["a037be6d"]),
        # NO BACKTICK AT ALL. This is the shape of the eight the scan could
        # not see: a measured figure in a plain-text table.
        ("    slice of s[1] barred ('scatter',) sha 2896a0f2…", [], ["2896a0f2"]),
        # ...THE CUT, SPELLED TWO MORE WAYS AND SET OFF BY A SPACE.
        # `\\u2026` is how a Python source file that quotes a routed line
        # writes it, and there is one in this tree.
        ("`64a0ce8d\\u2026`", [], ["64a0ce8d"]),
        ("`64a0ce8d\\N{HORIZONTAL ELLIPSIS}`", [], ["64a0ce8d"]),
        ("`a037be6d …`", [], ["a037be6d"]),
        # ...AND THE DIGITS IN UPPERCASE.
        ("`A037BE6D…`", [], ["A037BE6D"]),
        ("`A037BE6DFC050FF5`", ["A037BE6DFC050FF5"], []),
        ("`dff95fc`", [], []),                      # a git short sha
        ("`a1521744c6dc074443fe549f19f48d7197abf759`", [], []),   # 40, a sha
        ("`4028234663852886e38`", [], []),          # a decimal run
        ("`0123456789012345`", [], []),             # 16, but no letter
        # A DOUBLE'S FRACTION IS NOT A DIGEST. `220446049250313e` is 16
        # characters of `[0-9a-f]` with a letter in it and this rule fired
        # on it until 2026-08-23.
        ("`2.220446049250313e-16`", [], []),
        ("`1.1102230246251565e-16`", [], []),
    ]:
        assert _hash_literals("exact", text) == exact, text
        assert _hash_literals("truncated", text) == trunc, text

def test_the_ratchets_MEMORY_is_the_BLOCK_and_not_the_LINE():
    """Third defeat of this ratchet, driven against the finder that fixes it.

    The measured defeat, at `4a13824`: the line of `src/stelling/ir.py` that
    reads *"return the identical hash, and it is not ``64a0ce8d…``. What a"*
    left BYTE-IDENTICAL while
    the three lines after it were rewritten from a withdrawal — *"what a
    reader can re-derive is the PROPERTY … the hash is not needed"* — into an
    assertion — *"a reader CAN re-derive that very hash … it is a measurement
    and is retained deliberately"*. Whole zero-dep suite `2175 passed, 149
    skipped`, identical to clean, ratchet unmoved. The `sentence` field could
    not see it because a sentence does not end where a line does.

    So the unit is the prose BLOCK, and both halves of that are driven here:
    the sha moves when a NEIGHBOURING line of the same block is reworded, and
    it does NOT move for a rewrite on the far side of a block boundary —
    which is what keeps the entry a claim about one paragraph rather than
    about a whole file.

    THE SECOND HALF IS A BOUNDARY AND A BOUNDARY HAS A FAR SIDE. A fourth
    repurpose lives there, permitted by construction and measured: see the
    paragraph above `_Truncated` for the two variants and for the
    consequence they carry — a tracked retraction can be overturned by
    prose one separator away, and every cross-block reference a tracked
    block makes is outside its own memory. Do not widen this test to catch
    them; the assertion below is what stops every entry from becoming a
    claim about its whole file.
    """
    withdrawn = [
        "        # return the identical hash, and it is not ``64a0ce8d…``. What a",
        "        # reader can re-derive is the PROPERTY — accepted on both, to one",
        "        # hash, refused here — and that is what this says now.",
        "        #",
        "        # THE REFUSAL IS RIGHT: an absent `dtype` param is a declaration",
    ]
    asserted = list(withdrawn)
    asserted[1] = "        # reader CAN re-derive is that very hash — both trees produce it,"
    asserted[2] = "        # it is a measurement and is retained deliberately."
    assert asserted[0] == withdrawn[0], "the literal's own line must not move"

    before = _prose_block(withdrawn, 0, python=True)
    assert _prose_block(asserted, 0, python=True) != before, (
        "the three lines after the literal turned its retraction into an "
        "assertion and the ratchet's memory did not move. That is the third "
        "defeat this field exists to catch"
    )

    # ...AND THE BLOCK ENDS WHERE THE PARAGRAPH DOES. The `#`-only line is
    # the boundary, so the sentence beyond it is somebody else's claim and
    # rewording it must NOT move this entry.
    beyond = list(withdrawn)
    beyond[4] = "        # THE REFUSAL IS WRONG, and this is a different paragraph."
    assert _prose_block(beyond, 0, python=True) == before, (
        "a rewrite on the far side of a `#`-only separator moved this entry's "
        "block sha, which would make every entry a claim about its whole file"
    )

    # THE LINE ITSELF STILL COUNTS, which is the first two defeats: the
    # `sentence` field is not replaced by the block, it is joined by it.
    reworded = list(withdrawn)
    reworded[0] = reworded[0].replace("it is not", "it IS")
    assert " ".join(reworded[0].split()) != " ".join(withdrawn[0].split())

    # MARKDOWN GETS THE SAME RULE WITHOUT THE COMMENT STRIP: a `#` there is a
    # heading and not a marker, which is why `python` is a parameter and not
    # sniffed from the character.
    assert _prose_line("# Heading", python=False) == "# Heading"
    assert _prose_line("# Heading", python=True) == "Heading"
    markdown = ["a claim about `64a0ce8d…`,", "continued on the next line.",
                "", "a different paragraph."]
    assert _prose_block_text(markdown, 0, python=False) == (
        "a claim about `64a0ce8d…`, continued on the next line."
    )


def test_every_canonicalization_the_record_names_is_DEMONSTRATED():
    """Each entry collapses two spellings of one document, driven here.

    An entry with no witness reds, a witness with no entry reds, and an
    entry whose canonicalization stops happening reds — which is the whole
    of what "computed so it cannot go stale" can mean for a commitment."""
    named = {name for name, _ in ir.CANONICALIZATIONS}
    assert named == set(_WITNESSES), (
        f"entries without a witness: {sorted(named - set(_WITNESSES))}; "
        f"witnesses without an entry: {sorted(set(_WITNESSES) - named)}"
    )
    assert len(ir.CANONICALIZATIONS) == len(named), "duplicate entry name"
    for name, witness in _WITNESSES.items():
        left, right = witness()
        assert left == right, (
            f"the {name!r} canonicalization no longer collapses the two "
            f"spellings it is recorded as collapsing: {left!r} != {right!r}"
        )
