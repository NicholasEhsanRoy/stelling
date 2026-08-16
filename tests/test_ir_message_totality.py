# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A REFUSAL MAY NOT ITSELF RAISE, SWEPT RATHER THAN ASSERTED.

`ir`'s validation runs inside the public dataclasses' own `__post_init__`,
so a message composed there that raises does not become a bad error
message — it becomes a raw exception out of `ir.Aval(...)`,
`ir.Array(...)`, `ir.Literal(...)`, `ir.JaxprEqn(...)` or
`ir.ClosedJaxpr(...)`, which is the exact class the validation exists to
close. And a `_load_check` message is an ARGUMENT: it is composed on the
PASSING path too, so a well-formed document whose extent merely has a
`__repr__` that refuses crashes the constructor.

**WHY THIS IS A SWEEP AND NOT A LIST.** The previous pass fixed two such
sites and wrote, in a comment, *"EVERY QUOTE HERE IS GUARDED"*. That
sentence was false 44 lines below itself, and the audit that found it
named four more sites. Driving the class rather than reading it: this
sweep produces **28 raw escapes on `30d4b04` and 0 on this tree, from 9
distinct quote sites**, of which that audit had named three. Two of the
six it had not fire on the PASSING path, on documents with nothing wrong
with them:

    ir.Array(dtype=<a str subclass whose __repr__ raises>, ...)
    ir.Literal(val=<an int subclass whose __repr__ raises>, aval=Aval(()))

Three further sites this document MASKS rather than clears —
`_validate_jaxpr` composes its `where` string before
`_validate_required_params` runs — are driven one line each below:
`ir.JaxprEqn`'s duplicate-key refusal and its `{dups}` list, and
`_validate_required_params`' own primitive quote. So the enumeration was
not the answer; the enumeration was the defect. What is held here instead
is a property over the document's own structure.

**METHOD.** One canonical, well-formed `ClosedJaxpr`. Its leaves are found
by walking `dataclasses.fields()` — so a field added to any IR dataclass
joins the sweep without anyone remembering to add it. Each leaf is
replaced, one at a time, by a SUBCLASS OF ITS OWN TYPE that refuses
`__repr__`, `__str__` and `__format__` and behaves normally in every other
respect; the document is rebuilt from the root, so every `__post_init__`
on the path re-runs, and `ir._validate_loaded` — the pass `from_dict`
runs — is driven over the result as well. The required outcome is
`TranscriptionError` or success. Never anything else.

**WHAT THIS SWEEP COVERS**

* Every `str`, `int`, `float`, `complex` and `bytes` leaf reachable from
  the canonical document, at both the construction door and the load door.
* Both the failing and the passing path at each: substituting a hostile
  leaf sometimes makes the document invalid and sometimes does not, and
  both outcomes are swept.

**WHAT IT DOES NOT COVER, said plainly**

* Leaves whose type cannot be subclassed — `bool` and `None`. They are
  counted and reported by the sweep rather than silently skipped.
* Objects that misbehave in ways other than refusing to be printed: an
  `__index__` that answers differently between calls, an `__eq__` that
  lies. Those are different findings with their own tests
  (`test_aval_lie_both_faces.py`).
* Any structure not reachable from the canonical document. It carries a
  const array, a scalar const, a declaration, a nested sub-jaxpr param, a
  list param, a namedtuple param, a literal operand and debug info; a
  param SHAPE the canonical document does not contain is not swept, and
  the census of what it holds is asserted below so that stays visible.
* `obligation.py`, `interval.py` and `propagate.py`, which compose their
  own refusals. Two message sites there are disclosed as reported-not-
  fixed in `CHANGELOG.md`.

Audit 0.2.0 B6 audit 4, F2.
"""
from __future__ import annotations

import dataclasses
import struct

import pytest

from stelling import ir


# ---------------------------------------------------------------------------
# the hostile leaves
# ---------------------------------------------------------------------------

def _refusing(base: type) -> type:
    """A subclass of `base` that is a perfectly good value of its type and
    only refuses to be PRINTED. `__format__` as well as `__repr__`,
    because an f-string's `{x}` reaches neither of the other two."""

    def _no(self, *a):
        raise RuntimeError(f"{base.__name__} refuses to be printed")

    return type(
        f"Refusing{base.__name__.capitalize()}", (base,),
        {"__repr__": _no, "__str__": _no, "__format__": _no},
    )


_HOSTILE: dict[type, type] = {
    b: _refusing(b) for b in (str, int, float, complex, bytes)
}

# leaf types that cannot carry a hostile subclass at all
_UNSUBCLASSABLE = (bool, type(None))


# ---------------------------------------------------------------------------
# the canonical document, and a structural walk of it
# ---------------------------------------------------------------------------

def _canonical() -> ir.ClosedJaxpr:
    """One well-formed document holding every leaf KIND the pass quotes."""
    f64 = "float64"
    a4 = ir.Aval(kind="ShapedArray", shape=(4,), dtype=f64)
    a0 = ir.Aval(kind="ShapedArray", shape=(), dtype=f64)
    ab = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")

    const_arr = ir.Array(dtype="<f8", shape=(2,), data=struct.pack("<2d", 1.0, 2.0))
    cv_arr = ir.Var(id=0, aval=ir.Aval(kind="ShapedArray", shape=(2,), dtype=f64))
    cv_scalar = ir.Var(id=1, aval=a0)

    x = ir.Var(id=2, aval=a4)
    s = ir.Var(id=3, aval=a0)
    pred = ir.Var(id=4, aval=ab)
    out = ir.Var(id=5, aval=ab)

    inner_v = ir.Var(id=10, aval=a0)
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(inner_v,), outvars=(inner_v,),
                       eqns=(), effects=()),
        consts=(),
    )

    decl = ir.JaxprEqn(
        primitive="stelling_any", invars=(), outvars=(x,),
        params=(("dtype", f64), ("hi", 2.0), ("lo", 1.0), ("shape", (4,))),
        source_info=("harness.py:1 (h)",),
    )
    red = ir.JaxprEqn(
        primitive="reduce_sum", invars=(x,), outvars=(s,),
        params=(("axes", (0,)), ("out_sharding", None)),
    )
    # a primitive with no required-param row, carrying the container
    # shapes `_validate_param_value` recurses through
    call = ir.JaxprEqn(
        primitive="closed_call", invars=(s,), outvars=(s,),
        params=(
            ("call_jaxpr", inner),
            ("nested_list", [inner, const_arr]),
            ("ntuple", ir.NamedTupleParam(cls="Dims", fields=(("lhs", (0,)),))),
            ("enum", ir.EnumParam(cls="Mode", member="CLIP")),
            ("treedef", ir.TreeDefParam(text="PyTreeDef(*)")),
            ("note", "a str param"),
        ),
        effects=("io",),
    )
    le = ir.JaxprEqn(
        primitive="le",
        invars=(s, ir.Literal(val=4.5, aval=a0)),
        outvars=(pred,),
    )
    asrt = ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,))

    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(cv_arr, cv_scalar), invars=(), outvars=(out,),
            eqns=(decl, red, call, le, asrt), effects=(),
            debug_info=ir.DebugInfo(func="h", arg_names=("a",),
                                    result_paths=("",)),
        ),
        consts=(const_arr, 3.0),
    )


def _leaf_paths(obj, path=()):
    """(path, value) for every scalar leaf, derived from the dataclasses'
    OWN fields — a new field joins the sweep with no edit here."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _leaf_paths(getattr(obj, f.name), path + (("f", f.name),))
    elif type(obj) in (tuple, list):
        for i, item in enumerate(obj):
            yield from _leaf_paths(item, path + (("i", i),))
    else:
        yield path, obj


def _rebuild(obj, path, new):
    """A copy of `obj` with the leaf at `path` replaced. Every dataclass on
    the path is CONSTRUCTED again, so every `__post_init__` re-runs — the
    construction door is measured, not bypassed."""
    if not path:
        return new
    (kind, key), rest = path[0], path[1:]
    if kind == "f":
        return dataclasses.replace(
            obj, **{key: _rebuild(getattr(obj, key), rest, new)}
        )
    items = list(obj)
    items[key] = _rebuild(items[key], rest, new)
    return items if type(obj) is list else tuple(items)


def _path_str(path) -> str:
    return "".join(f".{k}" if kind == "f" else f"[{k}]" for kind, k in path)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def _sweep() -> tuple[list[str], list[str], int]:
    """(raw escapes, sites swept, leaves skipped)."""
    doc = _canonical()
    escapes: list[str] = []
    swept: list[str] = []
    skipped = 0
    for path, value in _leaf_paths(doc):
        base = type(value)
        if base in _UNSUBCLASSABLE or base not in _HOSTILE:
            skipped += 1
            continue
        hostile = _HOSTILE[base](value)
        where = f"{_path_str(path)} ({base.__name__})"
        swept.append(where)
        # the CONSTRUCTION door
        try:
            mutated = _rebuild(doc, path, hostile)
        except ir.TranscriptionError:
            continue
        except Exception as exc:  # noqa: BLE001 — this IS the finding
            escapes.append(f"construct {where}: {type(exc).__name__}: {exc!s:.60}")
            continue
        # the LOAD door — the same pass `ClosedJaxpr.from_dict` runs
        try:
            ir._validate_loaded(mutated)
        except ir.TranscriptionError:
            pass
        except Exception as exc:  # noqa: BLE001
            escapes.append(f"load {where}: {type(exc).__name__}: {exc!s:.60}")
    return escapes, swept, skipped


def test_no_message_in_the_ir_validation_pass_can_raise():
    """THE SWEEP. Every scalar leaf of a canonical document replaced, one
    at a time, by a value of its own type that refuses to be printed."""
    escapes, swept, skipped = _sweep()
    assert not escapes, (
        f"{len(escapes)} of {len(swept)} leaf substitution(s) let a RAW "
        f"exception out of a public `ir` constructor or out of the load "
        f"validation. A refusal whose own message raises is not a "
        f"refusal, and a message composed on the passing path turns a "
        f"well-formed document into a crash:\n    "
        + "\n    ".join(escapes)
    )
    # the sweep must be worth running
    assert len(swept) >= 60, (
        f"only {len(swept)} leaf site(s) swept; the canonical document has "
        f"shrunk and this test is measuring less than it claims"
    )
    kinds = {s.rsplit("(", 1)[1] for s in swept}
    assert {"str)", "int)", "float)", "bytes)"} <= kinds, sorted(kinds)
    # and what it cannot reach is COUNTED, not silent
    assert skipped <= 40, (
        f"{skipped} leaves were skipped as unsubclassable ({_UNSUBCLASSABLE} "
        f"and unmapped types); if that number has grown, leaves the sweep "
        f"used to reach are now outside it"
    )


def test_the_sweep_catches_an_injected_defect_of_exactly_this_class():
    """THE POSITIVE CONTROL, and it is what makes the green above mean
    something. With `_safe_repr` and `_safe_type_name` neutered to the
    unguarded reads they replaced, the same sweep must find escapes."""
    saved = (ir._safe_repr, ir._safe_type_name, ir._safe_str)
    try:
        ir._safe_repr = repr
        ir._safe_type_name = lambda o: type(o).__name__
        ir._safe_str = str
        escapes, swept, _ = _sweep()
    finally:
        ir._safe_repr, ir._safe_type_name, ir._safe_str = saved
    assert escapes, (
        "neutering the guarded reads produced NO raw escape, so the sweep "
        "is not measuring the class it claims to: either the quotes moved "
        "or the canonical document no longer reaches them"
    )
    assert len(escapes) >= 10, (
        f"only {len(escapes)} escape(s) with the guards neutered, over "
        f"{len(swept)} sites; measured on this tree there are 27, across "
        f"eight distinct quote sites"
    )
    # and the tree is genuinely restored
    assert (ir._safe_repr, ir._safe_type_name, ir._safe_str) == saved


@pytest.mark.parametrize(
    "label,build",
    [
        # the three sites the sweep found that no audit had named — each
        # a PUBLIC constructor call on a document with nothing wrong with
        # it, or a refusal whose own message raised
        ("Array dtype, PASSING path",
         lambda h: ir.Array(dtype=h["str"]("<f8"), shape=(1,),
                            data=b"\x00" * 8)),
        ("Literal scalar val, PASSING path",
         lambda h: ir.Literal(
             val=h["int"](3),
             aval=ir.Aval(kind="ShapedArray", shape=(), dtype="float64"))),
        # the three the canonical document MASKS: `_validate_jaxpr`
        # composes its own `where` first, and no traced document carries
        # a duplicate param key
        ("JaxprEqn primitive, in the duplicate-key refusal",
         lambda h: ir.JaxprEqn(primitive=h["str"]("add"), invars=(),
                               outvars=(), params=(("a", 1), ("a", 2)))),
        ("the duplicate-key LIST itself",
         lambda h: ir.JaxprEqn(
             primitive="add", invars=(), outvars=(),
             params=((h["str"]("a"), 1), (h["str"]("a"), 2)))),
        ("_validate_required_params' own primitive quote",
         lambda h: ir._validate_required_params(
             ir.JaxprEqn(primitive=h["str"]("stelling_any"), invars=(),
                         outvars=()),
             "query")),
        # and the four the audit did name
        ("Literal aval extent",
         lambda h: ir.Literal(
             val=1.0,
             aval=ir.Aval(kind="ShapedArray", shape=(h["int"](2),),
                          dtype="float64"))),
        ("ClosedJaxpr const pairing",
         lambda h: ir.ClosedJaxpr(
             jaxpr=ir.Jaxpr(
                 constvars=(ir.Var(id=0, aval=ir.Aval(
                     kind="ShapedArray", shape=(h["int"](2),),
                     dtype="float64")),),
                 invars=(),
                 outvars=(ir.Var(id=1, aval=ir.Aval(
                     kind="ShapedArray", shape=(), dtype="bool")),),
                 eqns=()),
             consts=(1.0,))),
        ("stelling_any dtype param vs aval dtype",
         lambda h: ir.JaxprEqn(
             primitive="stelling_any", invars=(),
             outvars=(ir.Var(id=0, aval=ir.Aval(
                 kind="ShapedArray", shape=(2,), dtype=h["str"]("float64"))),),
             params=(("dtype", "float64"), ("lo", 1.0), ("hi", 2.0),
                     ("shape", (2,))))),
        ("stelling_any shape param extent",
         lambda h: ir.JaxprEqn(
             primitive="stelling_any", invars=(),
             outvars=(ir.Var(id=0, aval=ir.Aval(
                 kind="ShapedArray", shape=(2,), dtype="float64")),),
             params=(("dtype", "float64"), ("lo", 1.0), ("hi", 2.0),
                     ("shape", (h["int"](2),))))),
    ],
)
def test_the_named_sites_one_line_each(label, build):
    """The seven sites the record names, held individually so a reader can
    check the entry against one line each rather than against the sweep."""
    h = {"str": _HOSTILE[str], "int": _HOSTILE[int]}
    try:
        build(h)
    except ir.TranscriptionError as exc:
        # a refusal is a fine outcome; what may not happen is a raw
        # escape. Where the message quotes the hostile OBJECT it shows a
        # placeholder; where it quotes the extents the guard NORMALISED
        # it shows plain ints, and that is the better answer — the quote
        # is then of what was validated rather than of what was handed in.
        text = str(exc)
        assert "<unreadable>" in text or "shape (2,)" in text, text
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"{label}: RAW {type(exc).__name__} out of a public constructor: "
            f"{exc}"
        ) from None
