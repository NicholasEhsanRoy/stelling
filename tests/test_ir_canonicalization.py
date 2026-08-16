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
import dataclasses
import pathlib
import re
import struct
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


@pytest.mark.parametrize("build", [_canonical_document, _subclass_document],
                         ids=["exact spellings", "subclass spellings"])
def test_every_value_a_document_stores_is_of_an_EXACT_type(build):
    """THE PROPERTY, over the whole document and computed from the
    dataclasses' own fields rather than from a list of the ones anyone
    remembered.

    Driven over TWO spellings of the same document. The exact one says the
    door does not damage a well-formed query; the subclass one is what
    measures the door at all, and it holds a subclass in every field of
    every dataclass that can carry one."""
    doc = build()
    allowed = (ir._CANONICAL_EXACT | ir._CANONICAL_IR_TYPES
               | set(ir._LIBRARY_STORED_TYPES) | {tuple})
    wrong = [(p, type(v).__name__) for p, v in _stored_values(doc)
             if type(v) not in allowed]
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

    allowed = (ir._CANONICAL_EXACT | ir._CANONICAL_IR_TYPES
               | set(ir._LIBRARY_STORED_TYPES) | {tuple})
    wrong = [(p, type(v).__name__) for p, v in _stored_values(q)
             if type(v) not in allowed]
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


_WITNESSES = {
    "alpha-renaming": _witness_alpha_renaming,
    "param and effect order": _witness_order,
    "hash scope": _witness_hash_scope,
    "shape extents": _witness_extents,
    "stored value types": _witness_value_types,
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
    so this scan is over the record files only, and it is exact about what
    it looks for: a backticked token that is 16 or 64 lowercase-hex
    characters with at least one letter in it. A git SHA is 7 or 40 and
    does not match; the decimal digit runs these files are full of
    (`4028234663852886e38`) do not either."""
    root = pathlib.Path(__file__).resolve().parent.parent
    tok = re.compile(r"`([0-9a-f]{16}|[0-9a-f]{64})[.…]*`")
    found = []
    for name in ("CHANGELOG.md", "SOUNDNESS.md", "README.md",
                 "ARCHITECTURE.md", "DOCUMENTATION_ARCHITECTURE.md"):
        path = root / name
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            for m in tok.finditer(line):
                if any(c in "abcdef" for c in m.group(1)):
                    found.append(f"{name}:{i}: {m.group(0)}")
    assert not found, (
        "the record pins a content-hash literal in prose, which no reader "
        "can recompute and no test holds:\n  " + "\n  ".join(found)
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
