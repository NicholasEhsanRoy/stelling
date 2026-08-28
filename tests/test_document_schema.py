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
import ast
import base64
import collections
import copy
import dataclasses
import json
import pathlib
import re
import sys

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
    """SCOPE, DRIVEN, AND THE CAPABILITY PINNED IN ONE PLACE.

    `ir.JaxprEqn` is the constructor underneath both faces, and the suite
    builds empty and NaN declarations through it on purpose, over operands
    no `any_array` will produce. The document surface is where the vacuous
    verdict was reachable, so that is where the refusal is — the same
    split `_validate_required_params` makes, for a different reason, which
    its own docstring states.

    **A WITNESS FOR EACH OF THE TWO REFUSALS, IN EACH DIRECTION, IS
    CONSTRUCTED HERE**, so the argument for the split does not rest on a
    list of other files' names. It did: `ir.py` and both logs credited the
    whole capability to `tests/test_ieee_semantics.py`, and moving this
    rule to `JaxprEqn.__post_init__` in fact turns 11 pre-existing tests
    red across FOUR files — the `(nan, hi)` form among them living in
    `tests/test_undecided_detail.py`, not in the file named. B12's own
    review."""
    for lo, hi in ((float("inf"), float("inf")),
                   (float("-inf"), float("-inf")),
                   (float("nan"), 4.0),
                   (2.0, 1.0)):
        eqn = ir.JaxprEqn(
            primitive="stelling_any", invars=(),
            outvars=(ir.Var(id=0, aval=ir.Aval(kind="ShapedArray", shape=(),
                                               dtype="float64")),),
            params=(("dtype", "float64"), ("hi", hi),
                    ("lo", lo), ("shape", ())),
        )
        got = dict(eqn.params)["lo"]
        assert got == lo or (got != got and lo != lo)
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


# ------------------------------------- what "round-trips" actually means
#
# B12's OWN REVIEW, on B12's tree. The serialization comment above
# `ir._encode` said ``to_dict`` / ``from_dict`` *"must round-trip
# losslessly"*, full stop. That was already false at `a4e4056` for a
# params-less equation, and this batch widened it by two classes — the
# `(inf, inf)` and `(nan, hi)` declarations `_validate_decl_nonempty`'s own
# docstring says must stay CONSTRUCTIBLE, which it made non-RELOADABLE
# without saying so. The comment now states the bound; these tests are what
# stop it going stale, which is the failure mode this whole campaign is
# about.

_IR_PATH = pathlib.Path(ir.__file__)
# whitespace-normalised, so a phrase this file pins is matched however
# `ir.py` happens to have wrapped it
_IR_SRC = " ".join(_IR_PATH.read_text(encoding="utf-8").split())


def _load_only_rules():
    """The refusals reachable from the LOAD PATH and from NO
    ``__post_init__``, read off `ir.py`'s own call graph, partitioned by
    which half of the load path reaches them; plus the refusals reached by
    NEITHER, which is this instrument telling on itself.

    COMPUTED, not listed — the whole point. A third load-path-only rule
    added later lands in one of these buckets without anyone editing this
    file: in either of the first two the assertion below names the
    paragraphs that have to be rewritten before it can ship; in the third
    it says only that no closure reaches the rule, which is a different
    thing to fix and is said differently.

    **THE SEED IS THE WHOLE LOAD PATH, AND IT USED TO BE HALF OF IT.**
    `ClosedJaxpr.from_dict` is `_decode` and then `_validate_loaded`; this
    closure seeded from ``["_validate_loaded"]`` alone, so THE DECODER WAS
    NEVER IN ITS DOMAIN — and five of `ir.py`'s own refusals live there
    (`_doc_complex`, `_doc_keys`, `_doc_leaf`, `_doc_payload`,
    `_doc_sequence`). Driven on this tree, a synthetic third rule added
    seven ways:

        A_direct           called from `_validate_loaded`        RED
        B_helper           called from a helper it reaches       RED
        E_decoder_side     called from `_decode`                 RED
        C_alias            reached through a module-level alias  RED*
        D_dispatch_table   reached through a dict of functions   RED*
        G_lambda_indirect  reached through a module-level lambda RED*
        F_both_paths       on the load path AND a constructor    green

    ``*`` = red on the THIRD return value and not on the enumeration,
    which is what that return value is for. F is green because it is
    CORRECT: a rule a constructor also runs creates no
    constructible-and-not-reloadable subject. E was GREEN before the seed
    was widened, and is why it was widened.

    **WHAT THE CALL GRAPH CANNOT SEE, AND HOW THAT IS CAUGHT ANYWAY.**
    The edges here are `ast.Call` targets that are a `Name` or an
    `Attribute`, matched by SHORT name. A function reached through a
    module-level alias, through a dispatch table, or through a lambda is
    therefore reached by no edge at all — C, D and G above. Widening the
    edge relation would over-approximate ``from_ctor`` as well, which is
    the direction that makes this instrument MISS things, so it is not
    widened. Instead the third return value is every refusal in `ir.py`
    reached by NEITHER the load path NOR a constructor: a rule that is in
    no closure is either dead code or an edge this reader cannot follow,
    and both have to be accounted for by hand before they can ship. It is
    empty on this tree, and each of C, D and G puts the synthetic rule
    into it."""
    tree = ast.parse(_IR_PATH.read_text(encoding="utf-8"))
    calls = collections.defaultdict(set)
    by_short = collections.defaultdict(set)

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                q = prefix + child.name
                by_short[child.name].add(q)
                for n in ast.walk(child):
                    if isinstance(n, ast.Call):
                        f = n.func
                        if isinstance(f, ast.Name):
                            calls[q].add(f.id)
                        elif isinstance(f, ast.Attribute):
                            calls[q].add(f.attr)
                visit(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, child.name + ".")
            else:
                visit(child, prefix)

    visit(tree)

    def closure(seeds):
        seen, stack = set(), list(seeds)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            for q in by_short.get(n, ()):
                stack.extend(calls[q])
        return seen

    post_inits = [q for q in calls if q.endswith("__post_init__")]
    assert post_inits, "no __post_init__ found; this instrument is broken"
    # seeded with `__post_init__` ITSELF and not only its callees, so a
    # constructor that refuses in its own body is a constructor path
    from_ctor = closure(["__post_init__"])
    from_decode = closure(["_decode"])
    from_validate = closure(["_validate_loaded"])
    refusing = {
        n for n, qs in by_short.items()
        if any("_load_check" in calls[q] or "_doc_refuse" in calls[q]
               for q in qs)
    }
    assert refusing, "no refusal found; this instrument is broken"
    post_decode = (from_validate - from_ctor) & refusing
    decoder_side = (from_decode - from_validate - from_ctor) & refusing
    unreached = refusing - (from_decode | from_validate | from_ctor)
    return decoder_side, post_decode, unreached


def test_the_LOAD_ONLY_rules_are_exactly_the_two_the_module_NAMES():
    """The bound on the round-trip claim, enumerated from the call graph.

    A POST-DECODE rule that runs on the load path and on no construction
    path has a subject that is CONSTRUCTIBLE AND NOT RELOADABLE —
    `to_dict` writes it and `from_dict` will not take it back. There are
    exactly two, and the paragraph that states the bound has to name both.

    THE DECODER-SIDE REFUSALS ARE ASSERTED APART RATHER THAN SUMMED IN,
    and the reason is what their subject is. `_doc_*` judges whether a
    value IS A DOCUMENT — the shape `_encode` writes — before any IR
    object exists, so it can only fire on something handed to `from_dict`
    and never on an object a constructor built. It therefore does not
    widen the round-trip bound, which is a bound on OBJECTS. The widest
    document `_encode` will write is one with a registered value sitting
    in a straight-through slot, and `from_dict` takes that back — driven
    below, in
    `test_a_registered_type_is_kept_out_by_DECODE_and_not_by_encode`.
    They are enumerated all the same, so that a sixth decoder refusal
    cannot arrive without this test naming it."""
    decoder_side, post_decode, unreached = _load_only_rules()
    assert post_decode == {
        "_validate_required_params", "_validate_decl_nonempty"
    }, (
        "the set of post-decode load-path-only refusals changed, so the "
        "round-trip claim's bound changed: rewrite the serialization "
        "comment above `ir._encode` and the LOAD PATH ONLY paragraph in "
        "`ir._validate_decl_nonempty`, then update this test"
    )
    assert decoder_side == {
        "_doc_complex", "_doc_keys", "_doc_leaf", "_doc_payload",
        "_doc_sequence",
    }, (
        "the set of decoder-side load-only refusals changed. These judge "
        "the DOCUMENT and not a constructed object, so they do not widen "
        "the round-trip bound — but say so deliberately for the new one "
        "rather than letting it land in a set nobody reads"
    )
    assert unreached == set(), (
        f"`ir.py` defines refusal(s) {sorted(unreached)} that this "
        f"module's call graph reaches from NEITHER the load path nor any "
        f"constructor. Either they are dead, or they are reached by an "
        f"edge an `ast.Call` walk cannot follow — a module-level alias, a "
        f"dispatch table, a lambda — and a rule reached by such an edge "
        f"is invisible to the enumeration above. Name the edge, or seed it"
    )
    head = "TWO REFUSALS RUN ON THE LOAD PATH ONLY"
    tail = "pins both halves and ENUMERATES"
    assert head in _IR_SRC and tail in _IR_SRC
    paragraph = _IR_SRC[_IR_SRC.index(head):_IR_SRC.index(tail)]
    for name in post_decode:
        assert name in paragraph, (
            f"the serialization comment states the round-trip bound "
            f"without naming {name}, which is one of the two rules that "
            f"creates it"
        )


def test_the_serialization_comment_states_the_BOUND_it_used_to_deny():
    """The sentence, pinned against the code beside it.

    `tests/test_bar_membership_policy.py` pins claims this way for the
    same reason: a prose statement nothing reads is a statement that goes
    stale, and this one did."""
    assert "must round-trip losslessly" not in _IR_SRC, (
        "the unconditional round-trip claim is back in ir.py; it is false "
        "for a params-less equation and for an empty declared set"
    )
    assert _IR_SRC.count("CONSTRUCTIBLE AND NOT RELOADABLE") == 2, (
        "the bound is stated at the serialization comment and again in "
        "`_validate_decl_nonempty`'s LOAD PATH ONLY paragraph, which is "
        "where a reader looks for it"
    )


def test_the_doc_keys_heading_is_SCOPED_to_the_sweep_it_measured():
    """`_doc_keys` closed the last raw escape OVER THE B12 CENSUS SWEEP,
    which is single-position and finite-valued. It is not the last one
    `from_dict` has: `_decode` recurses, and a deep enough
    ``{"k":"tuple","items":[…]}`` chain — reachable from pure JSON, since
    ``json.loads``/``json.dumps`` both accept it — raises a bare
    `RecursionError` on this tree and on `a4e4056` alike. Driven here so
    the heading's scope is a measurement and not a hedge."""
    assert "THE LAST OF THE READER'S RAW ESCAPES OVER THE B12 CENSUS SWEEP" \
        in _IR_SRC, "`ir._doc_keys`' heading dropped its scope"

    def nest(depth):
        d = {"k": "tuple", "items": []}
        for _ in range(depth):
            d = {"k": "tuple", "items": [d]}
        return d

    # TWO CLAIMS, AND THEY WERE FUSED INTO ONE LINE THAT ONLY ONE
    # INTERPRETER COULD RUN. The deep case read
    # `json.loads(json.dumps(nest(4 * sys.getrecursionlimit())))`, so the
    # fixture was built by round-tripping a structure thousands deep through
    # `json`. Measured 2026-08-28: CPython 3.12.3 manages it; **3.10.20 and
    # 3.11.15 raise `RecursionError` inside `json/encoder.py` itself**, before
    # `ir._decode` is reached at all — so this test failed on two of the three
    # interpreters `requires-python` admits, in a suite that ships inside the
    # sdist, and it failed in the SETUP rather than in its subject.
    #
    # The round-trip carries the *reachable from pure JSON* half and the depth
    # carries the *`_decode` recurses* half, and neither needs the other. They
    # are separated now: the reachability is shown at a depth every
    # interpreter round-trips, and the recursion is shown on a structure built
    # in Python, which is what `_decode` would receive either way.
    shallow = json.loads(json.dumps(nest(200)))
    assert isinstance(ir._decode(shallow), tuple)
    deep = nest(4 * sys.getrecursionlimit())
    with pytest.raises(RecursionError):
        ir._decode(deep)


def test_every_document_from_dict_ACCEPTS_reloads_with_its_hash():
    """The positive half, which is the half that carries the hash.

    Not a corpus result: the load rules are functions of the LOADED
    object, so accepting a document is accepting its re-encoding. Driven
    over the base and over every accepted single-position mutation of it,
    so the population is the schema's rather than one hand-picked
    document."""
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

    n = 0
    for path in positions(_doc()):
        for value in ([None] if not path else values):
            mut = _doc()
            if path:
                _set(mut, [str(c) for c in path], value)
            try:
                q = ir.ClosedJaxpr.from_dict(mut)
            except (ir.TranscriptionError, ValueError):
                continue
            again = ir.ClosedJaxpr.from_dict(q.to_dict())
            assert again.to_dict() == q.to_dict(), path
            assert again.content_hash() == q.content_hash(), path
            n += 1
    assert n > 100, n


def test_the_two_LOAD_ONLY_rules_write_documents_from_dict_REFUSES():
    """The negative half, one witness per rule, each ENCODED and then
    handed back.

    `_validate_required_params`' subject was non-reloadable at `a4e4056`
    already; `_validate_decl_nonempty`'s became non-reloadable HERE, and
    it is exactly the object that function's docstring promises stays
    constructible. Both refusals are `TranscriptionError` — loud, minting
    no verdict — and `content_hash` is a function of `to_dict` alone, so
    it still answers on both."""
    f64 = ir.Aval(kind="ShapedArray", shape=(2,), dtype="float64",
                  weak_type=False)

    def decl(params):
        v = ir.Var(id=1, aval=f64)
        eqn = ir.JaxprEqn(primitive="stelling_any", invars=(),
                          outvars=(v,), params=params)
        return ir.ClosedJaxpr(
            jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=(v,),
                           eqns=(eqn,)),
            consts=(),
        )

    full = (("dtype", "float64"), ("shape", (2,)))
    witnesses = {
        # _validate_required_params: hand-built IR legitimately omits params
        "_validate_required_params": decl(()),
        # _validate_decl_nonempty: one witness per refusal, both of them
        # forms the suite builds through the constructor on purpose
        "_validate_decl_nonempty (inf, inf)":
            decl(full + (("hi", float("inf")), ("lo", float("inf")))),
        "_validate_decl_nonempty (nan, hi)":
            decl(full + (("hi", 1.0), ("lo", float("nan")))),
    }
    for label, q in witnesses.items():
        d = q.to_dict()                      # encodes without complaint
        assert isinstance(d, dict), label
        assert q.content_hash(), label       # and the hash is unaffected
        with pytest.raises(ir.TranscriptionError):
            ir.ClosedJaxpr.from_dict(d)

    # and the honest declaration beside them reloads exactly
    ok = decl(full + (("hi", 2.0), ("lo", 1.0)))
    back = ir.ClosedJaxpr.from_dict(ok.to_dict())
    assert back.to_dict() == ok.to_dict()
    assert back.content_hash() == ok.content_hash()


# ------------------------------------ the registered type, and which door
#
# The field rule accepts a `_LIBRARY_STORED_TYPES` registration at ANY
# field, which is the door's widest exception. `ir.py` licensed it with
# TWO reasons — *"`_decode` has no tag for it and `_encode` refuses to
# encode one"* — and only the first is true. B12's own review, at THREE
# sites: two found by that review and the third — the comment above
# `_LIBRARY_STORED_TYPES`, above both of them and worded differently —
# only by the re-review of its own fix.


# The claim, as a SHAPE rather than as two spellings. The first version of
# the pin below asserted the absence of two exact strings — and the third
# copy of the same claim, in the comment above `_LIBRARY_STORED_TYPES` and
# earlier in the file than either of them, said "such a type" where those
# said "it" and sailed straight through. A pin that lists literal strings is the
# defect it is pinning, one level up.
_REGISTERED_VALUE = re.compile(
    r"registered|registration|IntervalArray|_LIBRARY_STORED_TYPES"
    r"|_STORED_AS_IS"
)
# the WRITING side, all of it — the claim was false about `_encode` and
# false again about `to_dict` and `content_hash`, so keying the rule on
# one function name would leave the other two spellings open
_WRITING_SIDE = re.compile(
    r"`_encode`|`to_dict`|to_dict\(\)|`content_hash`|content_hash\(\)"
    r"|encoder",
    re.I,
)
_EXCLUDES = re.compile(
    r"refus\w*|declin\w*|cannot encode|will not encode|does not encode"
    r"|never encodes|never reaches|outside|both raise",
    re.I,
)
# The qualification every TRUE form of the claim carries, because the truth
# is quantitative: `_encode` refuses only where it RECURSES, it writes
# through at EIGHTEEN slots, `content_hash` raises at FOURTEEN of them, and
# the other four are what `include_metadata=False` drops. An unqualified
# form is the false one however it is worded.
_QUALIFIED = re.compile(
    r"recurs\w*|eighteen|fourteen|straight through|include_metadata", re.I
)
# The three tokens have to land in the same breath, not merely in the same
# paragraph: `ir.py` has paragraphs that mention a registered type and,
# four sentences later, a document no encoder writes — a true and unrelated
# sentence that a paragraph-wide conjunction flags.
_CLAIM_WINDOW = 180


def _ir_paragraphs():
    """`ir.py` as (first line number, one-line text) paragraphs.

    A paragraph ends at a blank line, at a bare ``#``, or at a docstring
    delimiter — which is how this file already separates one argument from
    the next, so the unit the rule below judges is the unit a reader
    reads."""
    out, cur, start = [], [], 1
    for i, line in enumerate(
        _IR_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        s = line.strip()
        if s in ("", "#", '"""'):
            if cur:
                out.append((start, " ".join(cur)))
            cur, start = [], i + 1
        else:
            if not cur:
                start = i
            cur.append(s)
    if cur:
        out.append((start, " ".join(cur)))
    return out


def test_the_registered_type_exception_rests_on_DECODE_in_writing_too():
    """Every paragraph that licenses the exception, judged by SHAPE.

    The rule, and it is a rule and not a list: where `ir.py` names a
    REGISTERED VALUE within `_CLAIM_WINDOW` characters of the WRITING side
    (`_encode`, `to_dict`, `content_hash`, "the encoder") and an exclusion
    verb, it is claiming the writing side keeps a registered value out —
    and every TRUE form of that claim is scoped, because the truth is
    quantitative: only where `_encode` RECURSES, eighteen straight-through
    slots, fourteen of eighteen for `content_hash`, and the four that
    `include_metadata=False` drops. An unscoped form is the false one, in
    whatever words it is written.

    Three copies of it shipped. Two were corrected by the pass that found
    it; the third — the comment introducing `_LIBRARY_STORED_TYPES`, which
    is the first thing a would-be registrant reads — was found by B12's
    own re-review, and it carried an extra clause (*"outside
    `content_hash` and `to_dict` entirely"*) that is more strongly false
    than the one the pass was looking for."""
    claiming = []
    for n, para in _ir_paragraphs():
        for m in _REGISTERED_VALUE.finditer(para):
            window = para[max(0, m.start() - _CLAIM_WINDOW):
                          m.end() + _CLAIM_WINDOW]
            if _WRITING_SIDE.search(window) and _EXCLUDES.search(window):
                claiming.append((n, para))
                break
    # ...and the rule looked at something: a regex that stopped matching
    # and a file with no such paragraph give the same empty list
    assert len(claiming) >= 3, (
        f"only {len(claiming)} paragraph(s) of ir.py argue that the writing "
        f"side excludes a registered value; this rule has stopped matching "
        f"how they are written"
    )
    unqualified = [n for n, p in claiming if not _QUALIFIED.search(p)]
    assert not unqualified, (
        "ir.py line(s) "
        + ", ".join(str(n) for n in unqualified)
        + " license the registered-type exception off the WRITING side "
        "without scoping the claim. `_encode` refuses a registered value "
        "only in the arms where it RECURSES; at the EIGHTEEN slots it "
        "writes straight through, `to_dict()` returns a dict with the "
        "object in it and raises nothing, and `content_hash` raises at "
        "FOURTEEN of those eighteen and answers at the four that "
        "`include_metadata=False` drops. The argument rests on `_decode` "
        "alone"
    )


def test_a_registered_type_is_kept_out_by_DECODE_and_not_by_encode():
    """The surviving half, and the half that does not survive.

    `_decode` has no tag for a registered type, so no JSON document
    produces one — that is the whole argument and it holds. `_encode`
    refuses one only where it RECURSES: at a slot it writes straight
    through, `to_dict()` returns a dict with the object sitting in it and
    raises nothing. Pinned as BEHAVIOUR so the corrected comment cannot
    drift back."""
    from stelling.interval import IntervalArray

    assert any(t is IntervalArray for t in ir._LIBRARY_STORED_TYPES)

    # (a) the surviving half: no tag, at any spelling a document might use
    for tag in ("interval", "IntervalArray", "intervalarray"):
        with pytest.raises(ValueError, match="unknown tag"):
            ir._decode({"k": tag})

    iv = IntervalArray(shape=(1,), los=(0.0,), his=(1.0,))
    f64 = ir.Aval(kind="ShapedArray", shape=(2,), dtype="float64",
                  weak_type=False)

    # (b) where `_encode` RECURSES it does refuse — the arm the comment
    #     was generalising from
    cv = ir.Var(id=2, aval=f64)
    v = ir.Var(id=1, aval=f64)
    eqn = ir.JaxprEqn(primitive="gt", invars=(v, v), outvars=(v,), params=())
    with pytest.raises(TypeError, match="cannot encode IntervalArray"):
        ir.ClosedJaxpr(
            jaxpr=ir.Jaxpr(constvars=(cv,), invars=(), outvars=(v,),
                           eqns=(eqn,)),
            consts=(iv,),
        ).to_dict()

    # (c) and where it does NOT recurse it writes the object through, at
    #     every one of the eighteen positions ir.py enumerates
    through = ir._encode(
        ir.Aval(kind=iv, shape=(2,), dtype="float64", weak_type=False), True
    )
    assert through["kind"] is iv, (
        "`_encode` now judges a non-recursing slot; if that is deliberate "
        "the registered-type paragraphs in ir.py may claim it again"
    )
    prim = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=(v,),
            eqns=(ir.JaxprEqn(primitive=iv, invars=(v,), outvars=(v,),
                              params=()),),
        ),
        consts=(),
    ).to_dict()
    assert prim["jaxpr"]["eqns"][0]["primitive"] is iv
    # ...and NO decoder-side rule refuses that document: `from_dict` takes
    # the widest thing `_encode` writes straight back, which is why the
    # `_doc_*` refusals do not widen the round-trip bound
    assert ir.ClosedJaxpr.from_dict(prim).to_dict() == prim

    built = {}
    for label, mk in _REGISTERED_AT.items():
        try:
            obj = mk(iv, f64, v)
        except (ir.TranscriptionError, TypeError):
            continue
        try:
            ir._encode(obj, True)
        except TypeError:
            continue
        built[label] = True
    assert set(built) == set(_WRITES_THROUGH), (
        "the set of positions `_encode` writes a registered value through "
        "changed; ir.py's registered-type paragraphs enumerate it"
    )


# The eighteen, as `ir.py` states them. Kept as data beside the driver
# above rather than inline, so the comparison the test makes is the one a
# reader of `ir.py` can do by eye.
_WRITES_THROUGH = (
    "<aval>.kind", "<aval>.dtype", "<aval>.weak_type", "<var>.id",
    "<enum>.cls", "<enum>.member", "<sentinel>.cls", "<opaque>.cls",
    "<treedef>.text", "<ntuple>.cls", "<ntuple>.fields KEY",
    "<eqn>.primitive", "<eqn>.effects[*]", "<eqn>.source_info[*]",
    "<dbg>.func", "<dbg>.arg_names[*]", "<dbg>.result_paths[*]",
    "<jaxpr>.effects[*]",
)

# Every position `_encode` writes WITHOUT a recursive call, including the
# five another rule already owns and refuses at construction — they are
# driven too, so "refused there" and "written through" are both
# measurements and the list above cannot quietly grow.
_REGISTERED_AT = {
    "<aval>.kind": lambda iv, f64, v: ir.Aval(kind=iv, shape=(2,),
                                              dtype="f8", weak_type=False),
    "<aval>.dtype": lambda iv, f64, v: ir.Aval(kind="S", shape=(2,),
                                               dtype=iv, weak_type=False),
    "<aval>.shape[*]": lambda iv, f64, v: ir.Aval(kind="S", shape=(iv,),
                                                  dtype="f8", weak_type=False),
    "<aval>.weak_type": lambda iv, f64, v: ir.Aval(kind="S", shape=(2,),
                                                   dtype="f8", weak_type=iv),
    "<array>.dtype": lambda iv, f64, v: ir.Array(dtype=iv, shape=(1,),
                                                 data=b"\x00" * 8),
    "<array>.shape[*]": lambda iv, f64, v: ir.Array(dtype="<f8", shape=(iv,),
                                                    data=b"\x00" * 8),
    "<array>.data": lambda iv, f64, v: ir.Array(dtype="<f8", shape=(1,),
                                                data=iv),
    "<var>.id": lambda iv, f64, v: ir.Var(id=iv, aval=f64),
    "<enum>.cls": lambda iv, f64, v: ir.EnumParam(cls=iv, member="m"),
    "<enum>.member": lambda iv, f64, v: ir.EnumParam(cls="C", member=iv),
    "<sentinel>.cls": lambda iv, f64, v: ir.SentinelParam(cls=iv),
    "<opaque>.cls": lambda iv, f64, v: ir.OpaqueParam(cls=iv),
    "<treedef>.text": lambda iv, f64, v: ir.TreeDefParam(text=iv),
    "<ntuple>.cls": lambda iv, f64, v: ir.NamedTupleParam(cls=iv, fields=()),
    "<ntuple>.fields KEY": lambda iv, f64, v: ir.NamedTupleParam(
        cls="C", fields=((iv, 1),)),
    "<eqn>.primitive": lambda iv, f64, v: ir.JaxprEqn(
        primitive=iv, invars=(v,), outvars=(v,)),
    "<eqn>.params KEY": lambda iv, f64, v: ir.JaxprEqn(
        primitive="gt", invars=(v, v), outvars=(v,), params=((iv, 1),)),
    "<eqn>.effects[*]": lambda iv, f64, v: ir.JaxprEqn(
        primitive="gt", invars=(v, v), outvars=(v,), effects=(iv,)),
    "<eqn>.source_info[*]": lambda iv, f64, v: ir.JaxprEqn(
        primitive="gt", invars=(v, v), outvars=(v,), source_info=(iv,)),
    "<dbg>.func": lambda iv, f64, v: ir.DebugInfo(func=iv, arg_names=(),
                                                  result_paths=()),
    "<dbg>.arg_names[*]": lambda iv, f64, v: ir.DebugInfo(
        func="f", arg_names=(iv,), result_paths=()),
    "<dbg>.result_paths[*]": lambda iv, f64, v: ir.DebugInfo(
        func="f", arg_names=(), result_paths=(iv,)),
    "<jaxpr>.effects[*]": lambda iv, f64, v: ir.Jaxpr(
        constvars=(), invars=(), outvars=(v,), eqns=(), effects=(iv,)),
}


def _slot_writes_through(node):
    """Does any part of this expression reach the document un-`_encode`d?

    Structural, so that a slot `_encode` writes PARTLY through is seen.
    ``[[key, _encode(v, meta)] for key, v in obj.params]`` recurses on the
    value and writes the KEY straight through; a substring test for
    ``"_encode("`` over the whole expression calls that a recursion and
    drops it, which is how `<eqn>.params` and `<ntuple>.fields` — the two
    KEY positions — used to be hand-listed instead of read.

    Conservative by construction: anything this does not recognise counts
    as written through, so a new encoding shape lands in the population
    rather than out of it."""
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, ast.Call):
        f = node.func
        return not (isinstance(f, ast.Name) and f.id == "_encode")
    if isinstance(node, ast.IfExp):          # the TEST is not written
        return (_slot_writes_through(node.body)
                or _slot_writes_through(node.orelse))
    if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
        return _slot_writes_through(node.elt)
    if isinstance(node, ast.DictComp):
        return (_slot_writes_through(node.key)
                or _slot_writes_through(node.value))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_slot_writes_through(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return any(_slot_writes_through(e)
                   for e in [*node.keys, *node.values] if e is not None)
    return True                               # Name, Attribute, anything else


def _encode_write_through_positions():
    """Every ``<tag>.key`` `_encode` writes without recursing, from its AST.

    The TAG is carried, and that is the fix. The previous comparison
    reduced a position to its bare key name (``w.split(".")[-1]``), so a
    slot whose key name was already driven under a DIFFERENT tag was
    invisible: driven at `e3fe0fb`, a new ``<aval>.cls`` slot passed
    (``cls`` is driven at ``<enum>``/``<sentinel>``/``<opaque>``/
    ``<ntuple>``) while a new ``<aval>.zzz`` failed. Same defect one level
    up as the pin above it: a check that compares names rather than the
    thing the name is part of.

    The tag of a ``d[key] = …`` write is the tag of the dict literal ``d``
    was bound to, tracked in source order — ``<eqn>.source_info`` reached
    this set as a bare ``"source_info"`` with no tag before that."""
    enc = next(
        n for n in ast.walk(ast.parse(_IR_PATH.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "_encode"
    )
    out, tag_of_var = set(), {}

    def tag_of(dct):
        return next(
            (v.value for k, v in zip(dct.keys, dct.values)
             if isinstance(k, ast.Constant) and k.value == "k"
             and isinstance(v, ast.Constant)),
            None,
        )

    def take(dct):
        for k, v in zip(dct.keys, dct.values):
            if isinstance(k, ast.Constant) and k.value != "k" \
                    and _slot_writes_through(v):
                out.add("<%s>.%s" % (tag_of(dct), k.value))

    def walk(stmts):
        for st in stmts:
            if isinstance(st, ast.Return) and isinstance(st.value, ast.Dict):
                take(st.value)
            elif isinstance(st, ast.Assign) \
                    and isinstance(st.value, ast.Dict) \
                    and isinstance(st.targets[0], ast.Name):
                tag_of_var[st.targets[0].id] = tag_of(st.value)
                take(st.value)
            elif isinstance(st, ast.Assign) \
                    and isinstance(st.targets[0], ast.Subscript) \
                    and isinstance(st.targets[0].value, ast.Name) \
                    and _slot_writes_through(st.value):
                out.add("<%s>.%s" % (
                    tag_of_var.get(st.targets[0].value.id),
                    ast.literal_eval(st.targets[0].slice)))
            for field in ("body", "orelse", "finalbody"):
                walk(getattr(st, field, None) or [])

    walk(enc.body)
    return out


def test_the_non_recursing_slots_are_read_off_encodes_own_AST():
    """The population the test above judges is COMPUTED, like every other
    population in this file.

    `_REGISTERED_AT` has to be exactly the set of positions `_encode`
    writes without a recursive call — otherwise a slot added to the
    encoding later would be driven nowhere and `ir.py`'s enumeration
    would be a list nothing checks.

    COMPARED BY FULL ``<tag>.key`` AND IN BOTH DIRECTIONS. By bare key
    name — as this read until B12's own re-review — a new ``<aval>.cls``
    slot was undriven and green; it is now red, as is a new ``<aval>.zzz``,
    and as is a position DROPPED from `_REGISTERED_AT`. A new slot that
    RECURSES stays green, which is correct: it is not a straight-through
    position."""
    written = _encode_write_through_positions()

    def position(label):
        return label.split(" ")[0].replace("[*]", "")

    driven = {position(k) for k in _REGISTERED_AT}
    # `<complex>.re`/`.im` are read off a `complex`, so no other object
    # can occupy them; everything else `_encode` writes through is driven.
    assert written - driven <= {"<complex>.re", "<complex>.im"}, (
        f"`_encode` writes {sorted(written - driven)} straight through and "
        f"`_REGISTERED_AT` drives it nowhere"
    )
    assert not driven - written, (
        f"`_REGISTERED_AT` drives {sorted(driven - written)}, which "
        f"`_encode` no longer writes through: the enumeration in `ir.py` "
        f"names a position that is gone"
    )
    # the two KEY positions this set used to miss entirely, because their
    # values recurse and their keys do not
    assert {"<eqn>.params", "<ntuple>.fields"} <= written
