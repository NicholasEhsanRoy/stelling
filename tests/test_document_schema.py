# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE TYPE THE CODE DECLARES AT EVERY POSITION A DOCUMENT SUPPLIES.

Audit 0.2.0 B12, S15 and S16. `ClosedJaxpr.from_dict` judged almost none
of the document it was handed. `_encode` is a total function from IR to
JSON with one arm per stored type, so the JSON type at every position is
not a convention anybody has to remember — it is what that function writes
there — and nothing asked. Measured on `main` at `a4e4056`, from pure
JSON, with no attacker Python anywhere:

* ``<eqn>.primitive`` as ``null``/``true``/``0``/``-1``/``1.5``/``[]``/
  ``[0]`` was ACCEPTED, silently reclassified as an unknown primitive, and
  the ``stelling_assert`` the equation carried DISAPPEARED — a REFUTED
  two-obligation query returned VERIFIED with one obligation (S15);
* ``stelling_any``'s ``lo``/``hi`` had no type rule and no emptiness rule.
  ``"0.5"``, ``true`` and ``1`` loaded and MOVED the declared box; ``""``,
  ``"xx"``, ``null`` and ``()`` loaded and then raw-crashed out of the
  public ``propagate()``; and a declaration of ``(inf, inf)`` — the empty
  real set the trace face refuses at declaration time — returned VERIFIED
  with 100% coverage (S16);
* ``<tuple>.items`` read ``{}`` as the empty geometry, ``"xx"`` as
  ``('x','x')`` and a tagged object as its dict KEYS;
* ``<var>.id: null``, ``<aval>.weak_type: "xx"``, ``<aval>.dtype: 1.5``
  and ``<dbg>.func: 0`` all loaded and round-tripped faithfully.

**THIS FILE DRIVES THE RULE AND NOT ITS INSTANCES.** Every population it
judges is COMPUTED — the tags from `ir`'s own dataclasses, the key sets
from `_encode`'s own output, the sequence positions from a real document —
so a position added later is covered without anyone editing a list here.
That is the same discipline `tests/test_shape_param_rule.py` applies to
the one rule this one generalises.
"""

from __future__ import annotations

import array as _arraymod
import base64
import copy
import dataclasses
import json

import pytest

from stelling import ir

# --------------------------------------------------------------- documents


def _doc(**over):
    """A minimal well-formed document: one declaration, one comparison, one
    assertion over ``x in [1.0, 2.0]``. Deep-copied per call so a mutation
    cannot leak between tests."""
    f64 = {"k": "aval", "kind": "ShapedArray", "shape": [2],
           "dtype": "float64", "weak_type": False}
    b = {"k": "aval", "kind": "ShapedArray", "shape": [2], "dtype": "bool",
         "weak_type": False}
    d = {
        "k": "closed",
        "jaxpr": {
            "k": "jaxpr",
            "constvars": [],
            "invars": [],
            "outvars": [{"k": "var", "id": 0, "aval": b}],
            "eqns": [
                {"k": "eqn", "primitive": "stelling_any", "invars": [],
                 "outvars": [{"k": "var", "id": 1, "aval": f64}],
                 "params": [["dtype", "float64"], ["hi", 2.0], ["lo", 1.0],
                            ["shape", {"k": "tuple", "items": [2]}]],
                 "effects": [], "source_info": ["h.py:1 (stelling_any)"]},
                {"k": "eqn", "primitive": "gt",
                 "invars": [{"k": "var", "id": 1, "aval": f64},
                            {"k": "lit", "val": 0.0,
                             "aval": {"k": "aval", "kind": "ShapedArray",
                                      "shape": [], "dtype": "float64",
                                      "weak_type": False}}],
                 "outvars": [{"k": "var", "id": 2, "aval": b}],
                 "params": [], "effects": [],
                 "source_info": ["h.py:2 (gt)"]},
                {"k": "eqn", "primitive": "stelling_assert",
                 "invars": [{"k": "var", "id": 2, "aval": b}],
                 "outvars": [{"k": "var", "id": 0, "aval": b}],
                 "params": [], "effects": [],
                 "source_info": ["h.py:3 (stelling_assert)"]},
            ],
            "effects": [],
            "debug_info": {"k": "dbg", "func": "h at h.py:1",
                           "arg_names": [], "result_paths": ["result"]},
        },
        "consts": [],
    }
    d = copy.deepcopy(d)
    for path, value in over.items():
        _set(d, path.split("__"), value)
    return d


_ABSENT = object()


def _set(doc, path, value):
    cur = doc
    for c in path[:-1]:
        cur = cur[int(c)] if isinstance(cur, list) else cur[c]
    last = path[-1]
    key = int(last) if isinstance(cur, list) else last
    if value is _ABSENT:
        del cur[key]
    else:
        cur[key] = value


def _decl_param(doc, name):
    """The path components of a ``stelling_any`` param's VALUE slot."""
    eqn = doc["jaxpr"]["eqns"][0]
    for i, (k, _v) in enumerate(eqn["params"]):
        if k == name:
            return "jaxpr__eqns__0__params__%d__1" % i
    raise AssertionError(name)


def test_the_base_document_this_file_mutates_is_well_formed():
    """Every refusal below is worth nothing if the unmutated document is
    already refused — the first thing to check, and the thing a mutation
    suite most often gets wrong."""
    q = ir.ClosedJaxpr.from_dict(_doc())
    assert q.to_dict() == _doc()
    assert ir.ClosedJaxpr.from_dict(q.to_dict()).content_hash() \
        == q.content_hash()


# ------------------------------------------------------- S15: the primitive

# Every value the B12 census sweeps that cannot be a primitive NAME. Seven
# of the nine were ACCEPTED on `a4e4056` and are S15; the two `dict`s were
# already refused there, by the canonicalization door, for having no exact
# form to store — they are here so the population is the schema's and not
# the defect's. The census's other two values, `""` and `"xx"`, ARE names
# and must still be accepted: an unknown primitive going to ⊤ is coverage,
# not a malformed document, and the test below pins that.
_NOT_A_NAME = [None, True, 0, -1, 1.5, [], [0], {}, {"k": "tuple", "items": []}]


@pytest.mark.parametrize("value", _NOT_A_NAME, ids=repr)
def test_S15_a_primitive_that_is_not_a_name_is_REFUSED(value):
    """AUDIT 0.2.0 B12, S15. `<eqn>.primitive` is declared `str` and had no
    type rule at all, so each of these loaded as an unknown primitive — and
    an equation whose primitive is unknown goes to ⊤, which for a
    `stelling_assert` means the OBLIGATION DISAPPEARS."""
    with pytest.raises(ir.TranscriptionError):
        ir.ClosedJaxpr.from_dict(_doc(jaxpr__eqns__2__primitive=value))


@pytest.mark.parametrize("name", ["", "xx", "some_unknown_primitive"])
def test_an_unknown_primitive_NAME_is_still_accepted(name):
    """The other half, and the one a refusal must not take away: a `str`
    this build does not know is a coverage fact, not a malformed document.
    It loads, and the ⊤ it produces is what `stelling.coverage` reports."""
    q = ir.ClosedJaxpr.from_dict(_doc(jaxpr__eqns__1__primitive=name))
    assert q.jaxpr.eqns[1].primitive == name


def test_S15_the_obligation_a_non_name_used_to_delete_is_still_there():
    """The consequence, stated as the thing that matters rather than as the
    refusal. The document carries one `stelling_assert`; every value in the
    S15 population is refused, so there is no document in that population
    that loads carrying zero."""
    base = ir.ClosedJaxpr.from_dict(_doc())
    assert sum(1 for e in base.jaxpr.eqns
               if e.primitive == "stelling_assert") == 1
    for value in _NOT_A_NAME:
        with pytest.raises(ir.TranscriptionError):
            ir.ClosedJaxpr.from_dict(_doc(jaxpr__eqns__2__primitive=value))


# ------------------------------------------------------- S16: the bounds

_NOT_A_BOUND = [None, True, False, 0, 1, -1, "0.5", "1", "", "xx",
                {"k": "tuple", "items": []}]


@pytest.mark.parametrize("name", ["lo", "hi"])
@pytest.mark.parametrize("value", _NOT_A_BOUND, ids=repr)
def test_S16_a_bound_that_is_not_binary64_is_REFUSED(name, value):
    """AUDIT 0.2.0 B12, S16. Ten reader sites do `float(params["lo"])` with
    no gate in front of them, and `vacuity.widen` compares the two RAW —
    so `"1.0"` was a point by the reading that decides and not a point by
    the reading that widens."""
    doc = _doc()
    _set(doc, _decl_param(doc, name).split("__"), value)
    with pytest.raises(ir.TranscriptionError):
        ir.ClosedJaxpr.from_dict(doc)


def test_S16_the_two_readers_of_a_bound_now_read_ONE_value():
    """The install, driven as the property it exists for: `vacuity.widen`
    decides "is this a POINT declaration?" with a raw `!=` while every
    analysis reads `float(...)`, and the guard's own value being stored is
    what makes those the same read."""
    q = ir.ClosedJaxpr.from_dict(_doc())
    params = dict(q.jaxpr.eqns[0].params)
    assert type(params["lo"]) is float and type(params["hi"]) is float
    # the two protocols, on the stored value
    assert (params["lo"] != params["hi"]) == (
        float(params["lo"]) != float(params["hi"]))


@pytest.mark.parametrize("lo,hi", [
    (1.5, 1.0),                       # inverted
    (float("nan"), 1.0),              # NaN: `nan <= x` is False
    (1.0, float("nan")),
    (float("inf"), float("inf")),     # the infinite point
    (float("-inf"), float("-inf")),
    (float("inf"), 1.0),              # +inf below a finite hi
])
def test_S16_an_EMPTY_declared_set_is_REFUSED_at_the_door(lo, hi):
    """The refusal `harness.any_array` already makes, made at the
    deserialization door too. An empty declared set verifies every
    universal claim over it vacuously: measured on `main` at `a4e4056`,
    `(inf, inf)` on this document returned VERIFIED with 100% coverage."""
    doc = _doc()
    _set(doc, _decl_param(doc, "lo").split("__"), lo)
    _set(doc, _decl_param(doc, "hi").split("__"), hi)
    with pytest.raises(ir.TranscriptionError):
        ir.ClosedJaxpr.from_dict(json.loads(json.dumps(doc)))


@pytest.mark.parametrize("lo,hi", [
    (1.0, 2.0), (1.0, 1.0), (float("-inf"), float("inf")),
    (float("-inf"), 0.0), (0.0, float("inf")), (-0.0, 0.0),
])
def test_a_NON_empty_declared_set_still_loads(lo, hi):
    """A rule that refuses a legitimate document is a coverage defect. A
    point, an unbounded box and a half-line are all non-empty and all
    things `harness.any_array` admits."""
    doc = _doc()
    _set(doc, _decl_param(doc, "lo").split("__"), lo)
    _set(doc, _decl_param(doc, "hi").split("__"), hi)
    q = ir.ClosedJaxpr.from_dict(json.loads(json.dumps(doc)))
    assert dict(q.jaxpr.eqns[0].params)["lo"] == lo or lo != lo


def test_the_emptiness_rule_is_the_LOAD_doors_and_not_the_constructors():
    """SCOPE, DRIVEN. `ir.JaxprEqn` is the constructor underneath both
    faces and `tests/test_ieee_semantics.py` builds `(inf, inf)` and
    `(nan, hi)` declarations through it on purpose, to drive the ieee
    transfers over an operand no `any_array` will produce. The document
    surface is where the vacuous verdict was reachable, so that is where
    the refusal is — the same split `_validate_required_params` makes, for
    a different reason, which its own docstring states."""
    eqn = ir.JaxprEqn(
        primitive="stelling_any", invars=(),
        outvars=(ir.Var(id=0, aval=ir.Aval(kind="ShapedArray", shape=(),
                                           dtype="float64")),),
        params=(("dtype", "float64"), ("hi", float("inf")),
                ("lo", float("inf")), ("shape", ())),
    )
    assert dict(eqn.params)["lo"] == float("inf")
    # ... and the TYPE rule is the constructor's, on every path
    with pytest.raises(ir.TranscriptionError):
        ir.JaxprEqn(
            primitive="stelling_any", invars=(),
            outvars=(ir.Var(id=0, aval=ir.Aval(kind="ShapedArray", shape=(),
                                               dtype="float64")),),
            params=(("dtype", "float64"), ("hi", 2.0), ("lo", 1),
                    ("shape", ())),
        )


# ------------------------------------------- the container rule, everywhere


def _sequence_paths(doc, path=()):
    """Every position of ``doc`` at which `_encode` writes a JSON list.

    COMPUTED from a document rather than listed, so a sequence position
    added to the encoding later is swept here without anyone remembering
    this file."""
    if isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, list):
                yield path + (k,)
            yield from _sequence_paths(v, path + (k,))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            yield from _sequence_paths(v, path + (i,))


_NOT_A_SEQUENCE = ["", "xx", {}, {"k": "tuple", "items": []}, 0, None, True]


def test_every_sequence_position_of_a_document_refuses_a_non_sequence():
    """AUDIT 0.2.0 B12, the container rule. `_decode`'s tuple arm was
    VERBATIM the reader `_load_extents` replaced in B6 audit 8, and it was
    one level short: `_load_extents` judges the container of a SHAPE, and
    every geometry param is spelled ``{"k":"tuple","items":[…]}``.
    ``items: {}`` came out as ``()``, ``items: "xx"`` as ``('x','x')``, and
    ``items: {"k":…}`` as the dict's KEYS.

    Swept over EVERY list position of the document, not the one the finding
    was noticed at."""
    doc = _doc()
    paths = list(_sequence_paths(doc))
    assert len(paths) >= 12, paths
    escaped = []
    for path in paths:
        for value in _NOT_A_SEQUENCE:
            mut = _doc()
            _set(mut, [str(c) for c in path], value)
            try:
                ir.ClosedJaxpr.from_dict(mut)
            except (ir.TranscriptionError, ValueError):
                continue
            except BaseException as exc:  # noqa: BLE001 — a RAW escape
                escaped.append((path, value, type(exc).__name__))
            else:
                escaped.append((path, value, "ACCEPTED"))
    assert not escaped, escaped


def test_the_sequence_containers_the_rule_NAMES_are_the_ones_it_ACCEPTS():
    """:data:`ir._DOC_SEQUENCE_CONTAINERS` is a name for the pair
    :func:`ir._canonical_shell` accepts, and a name is not an
    implementation. The accepted set is COMPUTED here, over a population of
    container types built rather than listed, exactly as
    `tests/test_shape_param_rule.py` does for the shape rule."""
    population = [
        (), [], (1,), [1], b"ab", bytearray(b"ab"), "ab", memoryview(b"ab"),
        _arraymod.array("i", [1, 2]), range(2), {}, {"a": 1}, {1, 2},
        frozenset({1}), iter([1]), (x for x in (1,)), 1, 1.5, None, True,
        object(),
    ]
    accepted = set()
    for obj in population:
        try:
            ir._canonical_shell(obj)
        except ir._NotCanonical:
            continue
        accepted.add(type(obj))
    assert accepted == set(ir._DOC_SEQUENCE_CONTAINERS), (
        sorted(t.__name__ for t in accepted),
        sorted(t.__name__ for t in ir._DOC_SEQUENCE_CONTAINERS),
    )
    # and the sentence a refusal quotes is DERIVED from that tuple
    for t in ir._DOC_SEQUENCE_CONTAINERS:
        assert t.__name__ in ir._DOC_SEQUENCE_RULE


# ------------------------------------------------- the field-annotation rule


def _ir_dataclasses():
    return [obj for obj in vars(ir).values()
            if isinstance(obj, type) and dataclasses.is_dataclass(obj)
            and obj.__module__ == ir.__name__]


def test_every_ir_dataclass_field_has_a_spec_this_module_can_read():
    """A field whose annotation :func:`ir._spec_of` cannot read would be a
    field nothing is checking, which is the state this whole rule replaces.
    It raises rather than defaulting to "unconstrained", and this drives
    every field of every dataclass through it."""
    for cls in _ir_dataclasses():
        specs = {n: ir._spec_of(ir._typing.get_type_hints(cls)[n])
                 for n in ir._field_names(cls)}
        assert specs, cls
        for name, spec in specs.items():
            assert spec[0] in ("any", "exact", "seq", "pair"), (cls, name)
            assert ir._spec_rule(spec), (cls, name)


def test_a_document_value_of_the_wrong_declared_type_is_REFUSED():
    """One row per scalar position the census found accepting out of
    schema, driven through `from_dict`."""
    cases = [
        ("jaxpr__outvars__0__id", None),
        ("jaxpr__outvars__0__id", [0]),
        ("jaxpr__outvars__0__id", "0"),
        ("jaxpr__outvars__0__aval__kind", 0),
        ("jaxpr__outvars__0__aval__dtype", 1.5),
        ("jaxpr__outvars__0__aval__weak_type", "xx"),
        ("jaxpr__outvars__0__aval__weak_type", 1),
        ("jaxpr__debug_info__func", 0),
        ("jaxpr__debug_info__arg_names", "xx"),
        ("jaxpr__eqns__0__effects", "ab"),
        ("jaxpr__eqns__0__source_info", 7),
        ("jaxpr__debug_info", 5),
    ]
    for path, value in cases:
        with pytest.raises((ir.TranscriptionError, ValueError)):
            ir.ClosedJaxpr.from_dict(_doc(**{path: value}))


def test_a_falsy_debug_info_is_no_longer_read_as_ABSENT():
    """`_encode` writes `null` when and only when there is no `DebugInfo`,
    so `if dbg` asked the document a question the encoding does not
    answer: `0`, `""`, `[]`, `false` and `{}` were each silently read as
    "no debug info"."""
    for value in (0, "", [], False, {}):
        with pytest.raises((ir.TranscriptionError, ValueError)):
            ir.ClosedJaxpr.from_dict(_doc(jaxpr__debug_info=value))
    # and the two spellings that really do mean "none" still do
    assert ir.ClosedJaxpr.from_dict(
        _doc(jaxpr__debug_info=None)).jaxpr.debug_info is None
    assert ir.ClosedJaxpr.from_dict(
        _doc(jaxpr__debug_info=_ABSENT)).jaxpr.debug_info is None


def test_every_field_of_a_REAL_query_matches_its_own_declared_spec():
    """INCLUDING THE FIELDS :func:`ir._canonicalise` SKIPS, which is the
    half that is a claim rather than a mechanism. An `Aval`'s and an
    `Array`'s ``shape`` and a `JaxprEqn`'s ``params`` are skipped by the
    generic rule because each has a stronger rule of its own; that those
    rules PRODUCE what the annotation states is asserted here rather than
    in a comment."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_, trace

    def h():
        x = any_array((2, 3), "float64", (0.0, 1.0))
        assert_(jnp.sum(jnp.reshape(x, (6,))) <= 6.0)

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        q = trace(h)
    finally:
        jax.config.update("jax_enable_x64", old)

    seen = set()
    checked = 0

    def walk(o):
        nonlocal checked
        if id(o) in seen:
            return
        seen.add(id(o))
        if type(o) in set(_ir_dataclasses()):
            cls = type(o)
            hints = ir._typing.get_type_hints(cls)
            for name in ir._field_names(cls):
                v = getattr(o, name)
                spec = ir._spec_of(hints[name])
                assert ir._matches_spec(spec, v), (
                    cls.__name__, name, type(v).__name__, ir._spec_rule(spec))
                checked += 1
                walk(v)
        elif isinstance(o, tuple):
            for x in o:
                walk(x)

    walk(q)
    assert checked > 100, checked


# ------------------------------------------------------------ the key rule


def test_the_document_KEYS_are_the_dataclass_FIELDS_encode_writes():
    """THE DERIVATION :func:`ir._doc_keys` RESTS ON, computed rather than
    trusted. `_encode` writes ``"k"`` plus one key per declared field for
    every tagged dataclass; if that ever stopped being true, the required
    key set would be a different set from the one the encoder produces and
    a legitimate document would be refused."""
    q = ir.ClosedJaxpr.from_dict(_doc())
    tags = {}

    def walk(o):
        if isinstance(o, dict) and "k" in o:
            tags.setdefault(o["k"], set(o))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(q.to_dict())
    assert {"closed", "jaxpr", "eqn", "var", "aval", "lit", "tuple", "dbg"} \
        <= set(tags), sorted(tags)
    by_tag = {
        "closed": (ir.ClosedJaxpr, ()), "jaxpr": (ir.Jaxpr, ("debug_info",)),
        "eqn": (ir.JaxprEqn, ("source_info",)), "var": (ir.Var, ()),
        "aval": (ir.Aval, ()), "lit": (ir.Literal, ()), "dbg": (ir.DebugInfo, ()),
    }
    for tag, keys in tags.items():
        if tag not in by_tag:
            continue
        cls, _optional = by_tag[tag]
        assert keys == {"k"} | set(ir._field_names(cls)), (tag, sorted(keys))


def test_a_MISSING_key_is_a_refusal_and_not_a_raw_KeyError():
    """The last of the reader's raw escapes: every ``obj["…"]`` was
    unguarded, so DELETING a required key raised `KeyError` out of
    `from_dict` — 880 of the B12 census's 20,424 cells, and
    ``except TranscriptionError`` catches none of it."""
    for path in ("jaxpr__eqns__0__primitive", "jaxpr__outvars__0__id",
                 "jaxpr__eqns", "consts", "jaxpr__eqns__0__params",
                 "jaxpr__outvars__0__aval__weak_type"):
        with pytest.raises((ir.TranscriptionError, ValueError)):
            ir.ClosedJaxpr.from_dict(_doc(**{path: _ABSENT}))


def test_an_UNKNOWN_key_is_refused_rather_than_silently_dropped():
    """The direction the census's single-position sweep could not reach,
    because it only ever REPLACED a value that was already there. An extra
    key loaded, was dropped by the reader, and hashed as though the
    document had never said it."""
    for path, value in (("jaxpr__eqns__0__junk", 1),
                        ("jaxpr__outvars__0__primitive", "stelling_assert"),
                        ("junk", 1)):
        with pytest.raises((ir.TranscriptionError, ValueError)):
            ir.ClosedJaxpr.from_dict(_doc(**{path: value}))


def test_the_two_METADATA_keys_stay_optional():
    """`to_dict(include_metadata=False)` omits them — the hash-scope
    commitment — so a document that came from it must still load."""
    q = ir.ClosedJaxpr.from_dict(_doc())
    bare = q.to_dict(include_metadata=False)
    back = ir.ClosedJaxpr.from_dict(bare)
    assert back.content_hash() == q.content_hash()
    assert back.jaxpr.debug_info is None
    assert all(e.source_info == () for e in back.jaxpr.eqns)


# ------------------------------------------------ the leaves the reader eats


def test_an_array_payload_that_is_not_base64_TEXT_is_refused():
    """`<array>.data` is consumed by this reader (base64 in, `bytes` in the
    field), so it has no field to be judged at — and the default
    `b64decode` SILENTLY DISCARDS every character outside the alphabet.

    The decisive row is ``"AAAA!!AAAAAAA="``: the default strips the
    ``!!``, decodes the eight bytes the shape and dtype call for, and the
    byte-length check in `_validate_array_value` then PASSES on bytes the
    document did not write. `validate=True` is what refuses it, and the
    row is chosen so that nothing else could have."""
    def build(data):
        return _doc(jaxpr__eqns__1__params=[
            ["a", {"k": "array", "dtype": "<f8", "shape": [1],
                   "data": data}]])
    ok = ir.ClosedJaxpr.from_dict(build("AAAAAAAAAAA="))
    assert dict(ok.jaxpr.eqns[1].params)["a"].data == b"\x00" * 8
    # the default decoder accepts this and gets exactly the right LENGTH
    assert base64.b64decode("AAAA!!AAAAAAA=") == b"\x00" * 8
    for bad in (0, None, [], {}, "AAAA!!AAAAAAA=", "AA!!AA==", "A"):
        with pytest.raises((ir.TranscriptionError, ValueError)):
            ir.ClosedJaxpr.from_dict(build(bad))


def test_a_complex_part_that_is_not_a_number_is_refused():
    """`<complex>.re`/`.im` are multiplied into one `complex` by this
    reader. An `int` or `bool` there IS accepted and is a declared
    canonicalization (`ir.CANONICALIZATIONS`, "complex parts"); everything
    else has no number to denote, and ``complex(10**400, 0.0)`` raises
    `OverflowError` for two parts of an accepted type."""
    def build(re, im):
        return _doc(jaxpr__eqns__1__params=[
            ["c", {"k": "complex", "re": re, "im": im}]])
    assert dict(ir.ClosedJaxpr.from_dict(build(1, 0)).jaxpr.eqns[1].params
                )["c"] == complex(1.0, 0.0)
    for re, im in ((None, 0.0), ("1", 0.0), (0.0, "1"), ([], 0.0),
                   ({}, 0.0), (10 ** 400, 0.0), (0.0, 10 ** 400)):
        with pytest.raises((ir.TranscriptionError, ValueError)):
            ir.ClosedJaxpr.from_dict(build(re, im))


# --------------------------------------------------------------- no escapes


def test_from_dict_raises_only_its_two_DECLARED_shapes():
    """The catchability half of the residual class, driven. A `from_dict`
    refusal is a `TranscriptionError` or one of the reader's three
    documented `ValueError` arms (not a `ClosedJaxpr`, a non-dict where a
    tagged object belongs, an unknown tag) — never a raw `TypeError`,
    `AttributeError` or `KeyError`. Measured on `main` at `a4e4056`, the
    B12 census sweep produced 4,879 raw escapes over 20,424 cells; on this
    tree the same sweep produces none, and this is that sweep's shape over
    the document above."""
    doc = _doc()
    values = [None, True, 0, -1, 1.5, "", "xx", [], [0], {},
              {"k": "tuple", "items": []}, _ABSENT]

    def positions(o, path=()):
        yield path
        if isinstance(o, dict):
            for k in list(o):
                yield from positions(o[k], path + (k,))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from positions(v, path + (i,))

    raw = []
    n = 0
    for path in positions(doc):
        if not path:
            continue
        for value in values:
            mut = _doc()
            _set(mut, [str(c) for c in path], value)
            n += 1
            try:
                ir.ClosedJaxpr.from_dict(mut)
            except (ir.TranscriptionError, ValueError):
                continue
            except BaseException as exc:  # noqa: BLE001 — this IS the finding
                raw.append((path, value, type(exc).__name__, str(exc)[:60]))
    assert n > 1000, n
    assert not raw, raw[:10]
