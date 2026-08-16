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
named four more sites. Driving the class rather than reading it, the
record's figure is **10 distinct quote sites**: four the audit named and
six it had not.

**THE FIGURES, AND WHICH MEASUREMENT EACH BELONGS TO** — audit 0.2.0 B6
audit 5, F2, where this file carried four numbers for two measurements.
They are computed by
`test_the_recorded_FIGURES_are_the_ones_the_sweep_MEASURES` and
`test_the_QUOTE_SITE_COUNT_the_record_quotes_is_the_union_it_measures`,
and the unit is the MESSAGE EXPRESSION — one `_load_check(...)` spans
several source lines and can interpolate on more than one, so a per-LINE
count is a different and larger number, and it is the one that got
written down as if it were this one:

    this tree, as shipped        95 swept /  0 escapes / 20 skipped
    this tree, guards neutered   27 escapes /  9 lines /  8 messages
    `30d4b04`, guards absent     28 escapes / 10 lines /  8 messages
    message-expression union     10 = those 8 + the 2 the sweep masks

**AND "TEN" IS SAID TWICE IN TWO UNITS.** `CHANGELOG.md` also decomposes
the finding as four sites the audit named and six it had not; that counts
QUOTES — individual interpolations — where the figure above counts MESSAGE
EXPRESSIONS. They agree at ten by different routes (the duplicate-key
refusal is two quotes in one message; two audit-4 quotes no longer escape
at all), and neither is derived from the other, so the test asserts only
the computed one.

Two of the six the audit had not named fire on the PASSING path, on
documents with nothing wrong with them:

    ir.Array(dtype=<a str subclass whose __repr__ raises>, ...)
    ir.Literal(val=<an int subclass whose __repr__ raises>, aval=Aval(()))

Three further sites this document MASKS rather than clears —
`_validate_jaxpr` composes its `where` string before
`_validate_required_params` runs — are driven one row each below:
`ir.JaxprEqn`'s duplicate-key refusal and its `{dups}` list, and
`_validate_required_params`' own primitive quote (of which the first two
are one message expression, which is why the row count and the site
count differ). The sixth is the `NamedTupleParam` field name in a
`where` path, which the sweep reaches and no row drives. So the
enumeration was not the answer; the enumeration was the defect. What is
held here instead is a property over the document's own structure.

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

import ast
import dataclasses
import struct
from pathlib import Path
from typing import NamedTuple

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

class _Escape(NamedTuple):
    """One raw escape, with the QUOTE SITE that produced it derived rather
    than described — audit 0.2.0 B6 audit 5, F2.

    ``line`` is the DEEPEST frame in `ir.py` on the traceback, i.e. the
    physical line whose interpolation raised. ``message`` is the first
    line of the smallest `ast` statement containing it, i.e. the
    `_load_check(...)` or `raise` whose ARGUMENT that interpolation
    belongs to. The two counts differ because one message expression spans
    several lines and can carry a quote on more than one of them, which is
    exactly how a single measurement came to be recorded as two different
    numbers.
    """

    phase: str
    where: str
    line: int | None
    message: int | None
    text: str


def _ir_statement_starts() -> list[tuple[int, int]]:
    """(first line, last line) for every statement in `ir.py`, parsed from
    the shipped source. Derived, so a moved message moves with it."""
    tree = ast.parse(Path(ir.__file__).read_text(encoding="utf-8"))
    return [
        (n.lineno, getattr(n, "end_lineno", n.lineno))
        for n in ast.walk(tree)
        if isinstance(n, ast.stmt)
    ]


def _message_of(line: int | None, stmts) -> int | None:
    """The innermost statement containing `line` — the message expression
    the escaping interpolation is an argument to."""
    if line is None:
        return None
    best = None
    for lo, hi in stmts:
        if lo <= line <= hi and (best is None or (lo, -hi) > (best[0], -best[1])):
            best = (lo, hi)
    return None if best is None else best[0]


def _escaping_line(exc: BaseException) -> int | None:
    """The deepest frame in `ir.py` on the traceback: the interpolation
    that raised, not the constructor the caller sees."""
    tb, line = exc.__traceback__, None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == ir.__file__:
            line = tb.tb_lineno
        tb = tb.tb_next
    return line


def _sweep() -> tuple[list[_Escape], list[str], int]:
    """(raw escapes, sites swept, leaves skipped)."""
    doc = _canonical()
    stmts = _ir_statement_starts()
    escapes: list[_Escape] = []
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
            line = _escaping_line(exc)
            escapes.append(_Escape(
                "construct", where, line, _message_of(line, stmts),
                f"construct {where}: {type(exc).__name__}: {exc!s:.60}"))
            continue
        # the LOAD door — the same pass `ClosedJaxpr.from_dict` runs
        try:
            ir._validate_loaded(mutated)
        except ir.TranscriptionError:
            pass
        except Exception as exc:  # noqa: BLE001
            line = _escaping_line(exc)
            escapes.append(_Escape(
                "load", where, line, _message_of(line, stmts),
                f"load {where}: {type(exc).__name__}: {exc!s:.60}"))
    return escapes, swept, skipped


def _neutered_sweep():
    """The same sweep with the three guarded reads replaced by the
    unguarded ones they stand in for. Restores the module on the way out
    whatever happens."""
    saved = (ir._safe_repr, ir._safe_type_name, ir._safe_str)
    try:
        ir._safe_repr = repr
        ir._safe_type_name = lambda o: type(o).__name__
        ir._safe_str = str
        return _sweep()
    finally:
        ir._safe_repr, ir._safe_type_name, ir._safe_str = saved


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
        + "\n    ".join(e.text for e in escapes)
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
    something. With `_safe_repr`, `_safe_type_name` and `_safe_str`
    neutered to the unguarded reads they replaced, the same sweep must
    find escapes."""
    before = (ir._safe_repr, ir._safe_type_name, ir._safe_str)
    escapes, swept, _ = _neutered_sweep()
    assert escapes, (
        "neutering the guarded reads produced NO raw escape, so the sweep "
        "is not measuring the class it claims to: either the quotes moved "
        "or the canonical document no longer reaches them"
    )
    assert len(escapes) >= 10, (
        f"only {len(escapes)} escape(s) with the guards neutered, over "
        f"{len(swept)} site(s); the exact figures and the unit they are "
        f"counted in are pinned by "
        f"test_the_recorded_FIGURES_are_the_ones_the_sweep_MEASURES"
    )
    # and the tree is genuinely restored
    assert (ir._safe_repr, ir._safe_type_name, ir._safe_str) == before


# The figures the record quotes, in ONE place, in the unit they are counted
# in — audit 0.2.0 B6 audit 5, F2. `CHANGELOG.md`, this module's docstring
# and `ir._safe_repr`'s docstring had four numbers for two measurements
# ("26"/"27" escapes, "9"/"eight" quote sites) because each was typed where
# it was needed. They are computed here and the prose cites this test.
_SHIPPED = {"swept": 95, "escapes": 0, "skipped": 20}
_NEUTERED = {"escapes": 27, "lines": 9, "messages": 8}


def test_the_recorded_FIGURES_are_the_ones_the_sweep_MEASURES():
    """THE ARITHMETIC, COMPUTED — audit 0.2.0 B6 audit 5, F2.

    THE UNIT IS THE MESSAGE EXPRESSION, not the physical line. One
    `_load_check(...)` message spans several source lines and may
    interpolate on more than one of them, so a per-line count is larger
    than a per-message count and is not the number a reader means by "a
    quote site". Both are computed below; the per-message figure is the
    one the record quotes, and it is the one that is stable across the two
    measurements the record was conflating:

        `30d4b04`, guards ABSENT   28 escapes / 10 lines / 8 messages
        this tree, guards NEUTERED 27 escapes /  9 lines / 8 messages
        this tree, as shipped       0 escapes

    Those are two different documents' worth of arithmetic and the record
    quoted them as one. The `30d4b04` row is a historical measurement of a
    tree this suite does not carry; it is stated in the prose and is not
    asserted here, and the two figures that ARE about this tree are.
    """
    escapes, swept, skipped = _sweep()
    assert (len(swept), len(escapes), skipped) == (
        _SHIPPED["swept"], _SHIPPED["escapes"], _SHIPPED["skipped"]
    ), (
        f"the SHIPPED sweep now measures swept={len(swept)} "
        f"escapes={len(escapes)} skipped={skipped}, and the record says "
        f"{_SHIPPED}. Update this module's docstring and the "
        f"`ir.py` message-totality entry in `CHANGELOG.md` together."
    )

    escapes, _swept, _skipped = _neutered_sweep()
    measured = {
        "escapes": len(escapes),
        "lines": len({e.line for e in escapes}),
        "messages": len({e.message for e in escapes}),
    }
    assert measured == _NEUTERED, (
        f"the positive control now measures {measured} and the record says "
        f"{_NEUTERED}. Three places state it and they must move together: "
        f"this module's docstring, `ir._safe_repr`'s docstring, and the "
        f"`ir.py` message-totality entry in `CHANGELOG.md`."
    )
    assert None not in {e.message for e in escapes}, (
        "an escape could not be attributed to a statement in `ir.py`, so "
        "the per-message count is not measuring what it says"
    )
    # per-line >= per-message by construction; equality would mean no
    # message expression interpolates on two lines, which is not this file
    assert measured["lines"] > measured["messages"]


# The sites the RECORD names, one row each, tagged with the group the
# record puts them in — audit 0.2.0 B6 audit 5, F2. The groups were three
# sentences carrying three numbers ("the three sites the sweep found",
# "the four the audit did name", "the seven sites the record names") over
# a list of nine rows. They are now one structure, and the counts are
# `len` of it.
GROUP_UNNAMED = "found by the sweep, named by no audit"
GROUP_MASKED = "reachable only off the canonical document"
GROUP_AUDIT4 = "named by audit 4"

_NAMED_SITES: list[tuple[str, str, object]] = [
    # PUBLIC constructor calls on documents with nothing wrong with them:
    # a `_load_check` message is an argument, so it composes on the
    # passing path
    (GROUP_UNNAMED, "Array dtype, PASSING path",
     lambda h: ir.Array(dtype=h["str"]("<f8"), shape=(1,),
                        data=b"\x00" * 8)),
    (GROUP_UNNAMED, "Literal scalar val, PASSING path",
     lambda h: ir.Literal(
         val=h["int"](3),
         aval=ir.Aval(kind="ShapedArray", shape=(), dtype="float64"))),
    # sites the canonical document MASKS: `_validate_jaxpr` composes its
    # own `where` first, and no traced document carries a duplicate param
    # key
    (GROUP_MASKED, "JaxprEqn primitive, in the duplicate-key refusal",
     lambda h: ir.JaxprEqn(primitive=h["str"]("add"), invars=(),
                           outvars=(), params=(("a", 1), ("a", 2)))),
    (GROUP_MASKED, "the duplicate-key LIST itself",
     lambda h: ir.JaxprEqn(
         primitive="add", invars=(), outvars=(),
         params=((h["str"]("a"), 1), (h["str"]("a"), 2)))),
    (GROUP_MASKED, "_validate_required_params' own primitive quote",
     lambda h: ir._validate_required_params(
         ir.JaxprEqn(primitive=h["str"]("stelling_any"), invars=(),
                     outvars=()),
         "query")),
    (GROUP_AUDIT4, "Literal aval extent",
     lambda h: ir.Literal(
         val=1.0,
         aval=ir.Aval(kind="ShapedArray", shape=(h["int"](2),),
                      dtype="float64"))),
    (GROUP_AUDIT4, "ClosedJaxpr const pairing",
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
    (GROUP_AUDIT4, "stelling_any dtype param vs aval dtype",
     lambda h: ir.JaxprEqn(
         primitive="stelling_any", invars=(),
         outvars=(ir.Var(id=0, aval=ir.Aval(
             kind="ShapedArray", shape=(2,), dtype=h["str"]("float64"))),),
         params=(("dtype", "float64"), ("lo", 1.0), ("hi", 2.0),
                 ("shape", (2,))))),
    (GROUP_AUDIT4, "stelling_any shape param extent",
     lambda h: ir.JaxprEqn(
         primitive="stelling_any", invars=(),
         outvars=(ir.Var(id=0, aval=ir.Aval(
             kind="ShapedArray", shape=(2,), dtype="float64")),),
         params=(("dtype", "float64"), ("lo", 1.0), ("hi", 2.0),
                 ("shape", (h["int"](2),))))),
]

# The one place the fourth group lives: a site the sweep reaches and no
# row drives, because the sweep reaching it IS the finding.
_SWEEP_ONLY_SITES = 1   # the NamedTupleParam field name in a `where` path


def _hostiles() -> dict[str, type]:
    return {"str": _HOSTILE[str], "int": _HOSTILE[int]}


@pytest.mark.parametrize(
    "label,build",
    [(f"{group}: {label}", build) for group, label, build in _NAMED_SITES],
    ids=[label for _group, label, _build in _NAMED_SITES],
)
def test_the_named_sites_one_line_each(label, build):
    """Every site the record names, held individually so a reader can
    check the entry against one line each rather than against the sweep.

    HOW MANY THERE ARE IS `len(_NAMED_SITES)` AND IS NOT TYPED HERE —
    audit 0.2.0 B6 audit 5, F2, where this docstring said "the seven
    sites" over nine rows and the comment above them said "the three
    sites" over two."""
    h = _hostiles()
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


def test_the_QUOTE_SITE_COUNT_the_record_quotes_is_the_union_it_measures():
    """TEN, AND IT IS COMPUTED — audit 0.2.0 B6 audit 5, F2.

    The record's headline figure is the number of DISTINCT MESSAGE
    EXPRESSIONS in `ir.py` at which a hostile leaf produces a RAW escape
    once the guarded reads are neutered. It is a union of two drivers,
    and stating it as one number without saying so is how it came to be
    written as "9 ... of which the audit had named three" three lines
    under a heading that said the audit had named FOUR:

        the canonical sweep         8 message expressions
        the driven rows above     + 2 the sweep cannot reach
                                  = 10

    A `TranscriptionError` is NOT an escape and is not counted: its
    traceback always ends at `_load_check`'s own `raise`, which is not a
    quote site, and two of the audit-4 rows now refuse cleanly because the
    message they compose quotes extents the guard normalised.

    **AND THE RECORD'S OTHER TEN IS NOT THIS ONE.** `CHANGELOG.md` also
    decomposes the finding as *four sites the audit named, six it had
    not* — two on the passing path, three the canonical document masks,
    and the `NamedTupleParam` field name the sweep found. That is a count
    of QUOTES: individual interpolations of an object into a message. It
    is a different unit, and it reaches ten by a different route (the
    duplicate-key refusal is two quotes in ONE message expression, and
    two of the audit-4 quotes no longer escape at all). The two agree at
    ten and neither is derived from the other, so only the computed one is
    asserted here — writing an identity across two units is the defect
    this file exists for, one step subtler.
    """
    stmts = _ir_statement_starts()

    sweep_sites = {e.message for e in _neutered_sweep()[0]}
    assert len(sweep_sites) == _NEUTERED["messages"], sorted(sweep_sites)

    row_sites: set[int] = set()
    refused_cleanly = 0
    saved = (ir._safe_repr, ir._safe_type_name, ir._safe_str)
    try:
        ir._safe_repr = repr
        ir._safe_type_name = lambda o: type(o).__name__
        ir._safe_str = str
        for _group, label, build in _NAMED_SITES:
            try:
                build(_hostiles())
            except ir.TranscriptionError:
                refused_cleanly += 1
            except Exception as exc:  # noqa: BLE001 — this is the measurement
                site = _message_of(_escaping_line(exc), stmts)
                assert site is not None, label
                row_sites.add(site)
    finally:
        ir._safe_repr, ir._safe_type_name, ir._safe_str = saved
    assert (ir._safe_repr, ir._safe_type_name, ir._safe_str) == saved

    assert len(sweep_sites | row_sites) == 10, (
        f"the record says 10 distinct quote sites and the union measures "
        f"{len(sweep_sites | row_sites)}: sweep={sorted(sweep_sites)} "
        f"rows={sorted(row_sites)}. `CHANGELOG.md`, this module's "
        f"docstring and `ir._safe_repr`'s docstring all state it."
    )

    # the grouping the record uses, counted from the rows themselves
    groups = {g: sum(1 for x, _l, _b in _NAMED_SITES if x == g)
              for g in (GROUP_UNNAMED, GROUP_MASKED, GROUP_AUDIT4)}
    assert groups == {GROUP_UNNAMED: 2, GROUP_MASKED: 3, GROUP_AUDIT4: 4}, groups
    assert sum(groups.values()) == len(_NAMED_SITES) == 9
    # the tenth QUOTE has no row: `NamedTupleParam`'s field name, which
    # the sweep reaches. Counted here, NOT added to the union above —
    # rows and message expressions are different units (see the docstring).
    assert _SWEEP_ONLY_SITES == 1
