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
    """The refusals reachable from `ir._validate_loaded` and from NO
    ``__post_init__``, read off `ir.py`'s own call graph.

    COMPUTED, not listed — the whole point. A third load-path-only rule
    added later lands here without anyone editing this file, and the
    assertion below then names the two paragraphs that have to be
    rewritten before it can ship."""
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
    from_ctor = closure([c for q in post_inits for c in calls[q]])
    from_load = closure(["_validate_loaded"])
    refusing = {
        n for n, qs in by_short.items()
        if any("_load_check" in calls[q] or "_doc_refuse" in calls[q]
               for q in qs)
    }
    return (from_load - from_ctor) & refusing


def test_the_LOAD_ONLY_rules_are_exactly_the_two_the_module_NAMES():
    """The bound on the round-trip claim, enumerated from the call graph.

    A rule that runs on the load path and on no construction path has a
    subject that is CONSTRUCTIBLE AND NOT RELOADABLE — `to_dict` writes it
    and `from_dict` will not take it back. There are exactly two, and the
    paragraph that states the bound has to name both."""
    rules = _load_only_rules()
    assert rules == {
        "_validate_required_params", "_validate_decl_nonempty"
    }, (
        "the set of load-path-only refusals changed, so the round-trip "
        "claim's bound changed: rewrite the serialization comment above "
        "`ir._encode` and the LOAD PATH ONLY paragraph in "
        "`ir._validate_decl_nonempty`, then update this test"
    )
    head = "TWO REFUSALS RUN ON THE LOAD PATH ONLY"
    tail = "pins both halves and ENUMERATES"
    assert head in _IR_SRC and tail in _IR_SRC
    paragraph = _IR_SRC[_IR_SRC.index(head):_IR_SRC.index(tail)]
    for name in rules:
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

    shallow = json.loads(json.dumps(nest(200)))
    assert isinstance(ir._decode(shallow), tuple)
    deep = json.loads(json.dumps(nest(4 * sys.getrecursionlimit())))
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
# encode one"* — and only the first is true. B12's own review.


def test_the_registered_type_exception_rests_on_DECODE_in_writing_too():
    """Both paragraphs that license the exception, pinned as text.

    The half that is false has to be gone from both — the door narrative
    below `_encode` and `_register_stored_type`'s own docstring, which is
    where a would-be registrant reads what registration costs."""
    for false_half in (
        "`_decode` has no tag for it and `_encode` refuses to encode one",
        "so `to_dict` and `content_hash` both raise",
    ):
        assert false_half not in _IR_SRC, (
            "ir.py licenses the registered-type exception with `_encode` "
            "again; `_encode` refuses one only where it RECURSES"
        )
    assert "THE ARGUMENT RESTS ON `_decode` ALONE" in _IR_SRC
    assert "NOT `_encode`, which this paragraph also named" in _IR_SRC


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


def test_the_non_recursing_slots_are_read_off_encodes_own_AST():
    """The population the test above judges is COMPUTED, like every other
    population in this file.

    `_REGISTERED_AT` has to be exactly the set of positions `_encode`
    writes without a recursive call — otherwise a slot added to the
    encoding later would be driven nowhere and `ir.py`'s enumeration
    would be a list nothing checks."""
    enc = next(
        n for n in ast.walk(ast.parse(
            _IR_PATH.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "_encode"
    )
    written = set()
    for node in ast.walk(enc):
        if isinstance(node, ast.Dict):
            tag = next(
                (v.value for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and k.value == "k"
                 and isinstance(v, ast.Constant)),
                None,
            )
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value != "k" \
                        and "_encode(" not in ast.unparse(v):
                    written.add("<%s>.%s" % (tag, k.value))
        elif isinstance(node, ast.Assign) \
                and isinstance(node.targets[0], ast.Subscript) \
                and "_encode(" not in ast.unparse(node.value):
            written.add(ast.literal_eval(node.targets[0].slice))

    def base(label):
        return label.split(" ")[0].replace("[*]", "").split(".")[-1]

    driven = {base(k) for k in _REGISTERED_AT}
    # `<complex>.re`/`.im` are read off a `complex`, so no other object
    # can occupy them; everything else `_encode` writes through is driven.
    missing = {w for w in written if w.split(".")[-1] not in driven}
    assert missing <= {"<complex>.re", "<complex>.im"}, missing
