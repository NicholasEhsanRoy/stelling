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

    this tree, as shipped              95 swept / 0 escapes / 20 skipped
    guards neutered                     1 escape  /  1 line  / 1 message
    guards neutered, and the door's
      LEAF READS neutered too          26 escapes /  8 lines / 8 messages
    `30d4b04`, guards absent           28 escapes / 10 lines / 8 messages
    message-expression union           10 = those 8 + the 2 the sweep masks

**THERE ARE NOW TWO DEFENCES AND EACH IS MEASURED WITH THE OTHER
REMOVED** — audit 0.2.0 B6 audit 6. `ir`'s canonicalization door replaces
a leaf that is a SUBCLASS of a stored type with an exact twin, read
through the base type's own accessor — which is exactly what the hostile
leaves this sweep injects are, so their `__repr__` no longer survives into
the document. 25 of the 26 ESCAPES therefore do not happen, and they are
not GUARDED — they are not reached: 7 of the 8 message expressions are
never composed with a hostile object at all. (Escapes and message
expressions are different units and this sentence states both, which is
the discipline the rest of this file is about.) A positive control that
only neutered the guards would therefore be pushing on a door already
shut, and would go on reporting green if every `_safe_repr` in the module
were deleted. So
`_neutered_sweep(canonicalization=False)` removes the second defence too,
and that is the measurement the guard figures belong to.

The one escape that survives canonicalization is the hostile `int` inside
a declaration's `shape` PARAM tuple: the declaration door owns that param
(it has a container rule and normalises the extents itself), so the
generic door is held back from it and `_validate_decl_eqn`'s quotes see
the object as handed in. That is one site, it is guarded, and it is why
the guards are still load-bearing.

The guards-neutered figures also moved by exactly one escape and one
LINE against `f729d70` (27/9/8 -> 26/8/8), and the difference is
attributable: `_validate_decl_eqn`'s `dtype` message interpolated the
param on one line and the outvar aval's dtype on the next, and the param
half is now canonicalized inside that function (audit 6's `dtype`
finding), so one of that message expression's two lines no longer escapes.
The MESSAGE count is unchanged at 8, which is the unit the record quotes.

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
import re
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


def _neutered_sweep(*, canonicalization: bool = True):
    """The same sweep with the three guarded reads replaced by the
    unguarded ones they stand in for. Restores the module on the way out
    whatever happens.

    ``canonicalization=False`` ALSO removes the second defence — audit
    0.2.0 B6 audit 6. `ir`'s canonicalization door replaces a leaf that is
    a SUBCLASS of a stored type with an exact twin, read through the base
    type's own accessor, and a hostile `__repr__` does not survive that:
    which means the hostile leaves this sweep injects no longer REACH most
    of the quotes it exists to measure, and a positive control that only
    neutered the guards would be measuring a door that is already shut.

    Both defences are therefore measured with the OTHER out of the way.
    Setting every entry of `ir._CANONICAL_READS` to the identity leaves a
    subclass LEAF in the document exactly as it arrived, which is the tree
    the guards alone stood on for the values this sweep injects.

    IT DOES NOT REMOVE THE WHOLE DOOR, and saying so matters because the
    figure is quoted: `_canonical`'s `list`-to-`tuple` arm,
    `_canonical_shell`, and `_validate_decl_eqn`'s own `str.__str__` on
    the `dtype` param all still run. The last of those is the whole of the
    residue against `f729d70` — one escape and one line — and it is
    attributed in this module's docstring rather than left as drift.
    """
    saved = (ir._safe_repr, ir._safe_type_name, ir._safe_str)
    saved_reads = ir._CANONICAL_READS
    try:
        ir._safe_repr = repr
        ir._safe_type_name = lambda o: type(o).__name__
        ir._safe_str = str
        if not canonicalization:
            ir._CANONICAL_READS = tuple(
                (base, lambda v: v) for base, _ in saved_reads
            )
        return _sweep()
    finally:
        ir._safe_repr, ir._safe_type_name, ir._safe_str = saved
        ir._CANONICAL_READS = saved_reads


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
    before = (ir._safe_repr, ir._safe_type_name, ir._safe_str, ir._CANONICAL_READS)

    # 1. GUARDS ALONE. Canonicalization is removed as well, because it now
    #    stands in front of almost every quote this sweep aims at: with it
    #    in place a hostile leaf never reaches the message, so a control
    #    that only neutered the guards would report green with every
    #    `_safe_repr` in the module deleted (audit 0.2.0 B6 audit 6).
    escapes, swept, _ = _neutered_sweep(canonicalization=False)
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

    # 2. CANONICALIZATION ALONE, which is the same control from the other
    #    side: with the guards neutered and the door in place, all but one
    #    of those escapes is gone — not caught, NOT REACHED.
    still, _swept, _ = _neutered_sweep()
    assert len(still) < len(escapes), (
        f"the canonicalization door stopped no escape at all "
        f"({len(still)} with it, {len(escapes)} without); either the door "
        f"no longer replaces subclass leaves or the sweep no longer "
        f"injects them"
    )

    # and the tree is genuinely restored
    assert (ir._safe_repr, ir._safe_type_name, ir._safe_str,
            ir._CANONICAL_READS) == before


# The figures the record quotes, in ONE place, in the unit they are counted
# in — audit 0.2.0 B6 audit 5, F2. `CHANGELOG.md`, this module's docstring
# and `ir._safe_repr`'s docstring had four numbers for two measurements
# ("26"/"27" escapes, "9"/"eight" quote sites) because each was typed where
# it was needed. They are computed here and the prose cites this test.
_SHIPPED = {"swept": 95, "escapes": 0, "skipped": 20}
# guards neutered, canonicalization SHIPPED: what the guards still catch
# ON THE LEAF THIS SWEEP INJECTS, and that qualifier is the whole of the
# figure's meaning — audit 0.2.0 B6 audit 7. "1" was read as *"the door
# makes 25 of the 26 escapes unreachable rather than guarded"*, i.e. as a
# statement that `_safe_repr` is load-bearing at one site. It is not one.
# Every leaf this sweep injects is a SUBCLASS of a stored type, so the
# door replaces it with an exact twin and it never reaches the quote. A
# leaf that BYPASSES the door is stored UNCHANGED — which is precisely
# what neutering the leaf reads to the identity simulates — so it reaches
# every site the row below counts, and there the guards are the only
# thing between it and a raw escape. What this row measures is therefore
# the reach of one sweep against one defence, not the reach of the
# guards; the row below is the one that measures the guards.
_NEUTERED = {"escapes": 1, "lines": 1, "messages": 1}
# guards neutered AND canonicalization removed: the tree the guards alone
# stood on, and the measurement the record's per-site figures belong to
_NEUTERED_NO_CANON = {"escapes": 27, "lines": 9, "messages": 9}
# the union of the two drivers, in MESSAGE EXPRESSIONS — the record's
# headline quote-site figure, decomposed and asserted by
# `test_the_QUOTE_SITE_COUNT_the_record_quotes_is_the_union_it_measures`
_QUOTE_SITE_UNION = 11


_FIGURE_ROW = re.compile(
    r"^\s{4,}(?P<label>this tree[^0-9]*?)\s{2,}"
    r"(?P<escapes>\d+)\s*/\s*(?P<lines>\d+)\s*/\s*(?P<messages>\d+)\s*$"
)


def _docstring_figures() -> dict[str, dict[str, int]]:
    """The rows of the table in the docstring below, parsed out of it.

    Only the rows about THIS tree: the historical row is a measurement of
    a commit this suite does not carry and there is nothing here that
    could hold it, which is why it is labelled as historical rather than
    quietly listed beside two rows that are checked.
    """
    doc = test_the_recorded_FIGURES_are_the_ones_the_sweep_MEASURES.__doc__ or ""
    out: dict[str, dict[str, int]] = {}
    for line in doc.split("\n"):
        m = _FIGURE_ROW.match(line)
        if m:
            out[m.group("label").strip()] = {
                k: int(m.group(k)) for k in ("escapes", "lines", "messages")
            }
    assert len(out) == 3, f"the figure table has {len(out)} rows about this tree"
    return out


def test_the_recorded_FIGURES_are_the_ones_the_sweep_MEASURES():
    """THE ARITHMETIC, COMPUTED — audit 0.2.0 B6 audit 5, F2.

    THE UNIT IS THE MESSAGE EXPRESSION, not the physical line. One
    `_load_check(...)` message spans several source lines and may
    interpolate on more than one of them, so a per-line count is larger
    than a per-message count and is not the number a reader means by "a
    quote site". Both are computed below; the per-message figure is the
    one the record quotes, and it is the one that is stable across the two
    measurements the record was conflating:

        historical: 30d4b04, guards absent, no door  28 / 10 / 8
        this tree, guards neutered, door removed     27 /  9 / 9
        this tree, guards neutered, door shipped      1 /  1 / 1
        this tree, as shipped                         0 /  0 / 0

    Those are different documents' worth of arithmetic and the record
    quoted them as one. The `30d4b04` row is a historical measurement of a
    tree this suite does not carry, and is not asserted; every row whose
    label begins "this tree" is READ BACK OUT of this docstring below and
    compared to the measurement, which is the only thing that makes a
    table in prose worth writing.

    **AND THIS TABLE DID NOT MATCH THE DICTS ABOVE IT** — audit 0.2.0 B6
    audit 7. It read *"this tree, guards NEUTERED 27 escapes / 9 lines /
    8 messages"* over `_NEUTERED_NO_CANON = 26/8/8` and
    `_NEUTERED = 1/1/1`: three numbers no configuration in this file
    produces, in the docstring of the test that exists to stop exactly
    that, two commits after it was written to stop it. Nothing connected
    the row to a control, because the row did not say which control it
    was — so the repair is not to retype the digits. It is the labels,
    and the parser below that now holds them.

    The door-removed row gained one escape, one line and one message at
    audit 0.2.0 B6 audit 7: `_validate_decl_eqn` refuses a `dtype` param
    whose TYPE is not `str` in its own sentence rather than skipping the
    agreement check, and that sentence quotes the raw param.
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

    def _measure(**kw):
        escapes, _swept, _skipped = _neutered_sweep(**kw)
        assert None not in {e.message for e in escapes}, (
            "an escape could not be attributed to a statement in `ir.py`, "
            "so the per-message count is not measuring what it says"
        )
        return {
            "escapes": len(escapes),
            "lines": len({e.line for e in escapes}),
            "messages": len({e.message for e in escapes}),
        }

    for measured, record, which in (
        (_measure(), _NEUTERED, "guards neutered"),
        (_measure(canonicalization=False), _NEUTERED_NO_CANON,
         "guards neutered AND canonicalization removed"),
    ):
        assert measured == record, (
            f"the positive control ({which}) now measures {measured} and "
            f"the record says {record}. Three places state it and they must "
            f"move together: this module's docstring, `ir._safe_repr`'s "
            f"docstring, and the `ir.py` message-totality entry in "
            f"`CHANGELOG.md`."
        )
    # per-line >= per-message by construction. It used to be STRICTLY
    # greater, and the message expression that made it so was
    # `_validate_decl_eqn`'s `dtype` refusal, which interpolated the param
    # on one line and the outvar aval's dtype on the next; the param half
    # is canonicalized inside that function now (audit 0.2.0 B6 audit 6),
    # so the two counts have met. Equality is a fact about this tree, not
    # an invariant, which is why it is stated as the bound it really is.
    assert _NEUTERED_NO_CANON["lines"] >= _NEUTERED_NO_CANON["messages"]

    # AND THE TABLE IN THE DOCSTRING ABOVE IS READ BACK OUT AND COMPARED —
    # audit 0.2.0 B6 audit 7, where it stated three numbers no control in
    # this file produces. A table beside a dict is an honour-system copy
    # of it, which is the finding `tests/test_doc_examples.py` closes for
    # its own inventory and the same repair applies here.
    stated = _docstring_figures()
    assert stated == {
        "this tree, guards neutered, door removed": _NEUTERED_NO_CANON,
        "this tree, guards neutered, door shipped": _NEUTERED,
        "this tree, as shipped": {"escapes": 0, "lines": 0, "messages": 0},
    }, (
        f"the table in this test's docstring states {stated} and the dicts "
        f"above measure something else; the row labels are the dicts' own "
        f"names and both must move together"
    )


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
    sites" over two.

    **TWO DRIVES PER ROW SINCE AUDIT 6.** As shipped, the required outcome
    is a refusal or a success and never a raw escape — and the message a
    reader gets is now usually the ORDINARY one, because `ir`'s
    canonicalization door replaced the hostile leaf with an exact twin
    before any quote reached it, so there is nothing left to print a
    placeholder for. That is a better outcome and it is also one that
    would hold with every guard deleted, so each row is driven a second
    time with the door removed, where the GUARD is what must hold and the
    message must show the placeholder or the normalised extents. Both
    defences, each measured with the other out of the way.
    """
    def _drive(what):
        try:
            build(_hostiles())
        except ir.TranscriptionError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"{label} ({what}): RAW {type(exc).__name__} out of a "
                f"public constructor: {exc}"
            ) from None
        return None

    # 1. as shipped: a refusal or a success, never a raw escape
    _drive("as shipped")

    # 2. with the canonicalization door removed, the guard alone. Where
    #    the message quotes the hostile OBJECT it shows a placeholder;
    #    where it quotes the extents the guard NORMALISED it shows plain
    #    ints, and that is the better answer — the quote is then of what
    #    was validated rather than of what was handed in.
    saved = ir._CANONICAL_READS
    try:
        ir._CANONICAL_READS = tuple((b, lambda v: v) for b, _ in saved)
        text = _drive("canonicalization removed")
    finally:
        ir._CANONICAL_READS = saved
    if text is not None:
        assert "<unreadable>" in text or "shape (2,)" in text, text


def test_the_QUOTE_SITE_COUNT_the_record_quotes_is_the_union_it_measures():
    """ELEVEN, AND IT IS COMPUTED — audit 0.2.0 B6 audit 5, F2.

    The record's headline figure is the number of DISTINCT MESSAGE
    EXPRESSIONS in `ir.py` at which a hostile leaf produces a RAW escape
    once the guarded reads are neutered. It is a union of two drivers,
    and stating it as one number without saying so is how it came to be
    written as "9 ... of which the audit had named three" three lines
    under a heading that said the audit had named FOUR:

        the canonical sweep         9 message expressions
        the driven rows above     + 2 the sweep cannot reach
                                  = 11

    The figure is `_QUOTE_SITE_UNION` and the two addends are computed
    here; it was 10 until audit 0.2.0 B6 audit 7 gave
    `_validate_decl_eqn`'s `dtype` param a refusal for its TYPE, which is
    a ninth message expression the sweep reaches.

    Both halves are driven with `ir`'s canonicalization door removed, for
    the reason the module docstring gives: this is a count of the sites
    the GUARDS cover, and with the door in place a hostile leaf does not
    reach most of them (audit 0.2.0 B6 audit 6).

    A `TranscriptionError` is NOT an escape and is not counted: its
    traceback always ends at `_load_check`'s own `raise`, which is not a
    quote site, and two of the audit-4 rows now refuse cleanly because the
    message they compose quotes extents the guard normalised.

    **AND THE RECORD'S OTHER TEN WAS NEVER THIS ONE, AND NO LONGER AGREES
    WITH IT.** `CHANGELOG.md` also decomposes the finding as *four sites
    the audit named, six it had not* — two on the passing path, three the
    canonical document masks, and the `NamedTupleParam` field name the
    sweep found. That is a count of QUOTES: individual interpolations of
    an object into a message, a different unit reaching ten by a different
    route (the duplicate-key refusal is two quotes in ONE message
    expression, and two of the audit-4 quotes no longer escape at all).
    The two agreeing at ten was a coincidence of one tree and it has now
    ended: this union is eleven and the quote decomposition is a fixed
    historical statement about what audit 4 named. Only the computed one
    is asserted here, and the other is left labelled as the history it
    is — writing an identity across two units is the defect this file
    exists for, one step subtler, and keeping the identity alive by
    retyping the second number would be the same defect again.
    """
    stmts = _ir_statement_starts()

    # BOTH HALVES ARE DRIVEN WITH THE CANONICALIZATION DOOR REMOVED —
    # audit 0.2.0 B6 audit 6. This figure counts the quote sites the
    # GUARDS cover, and the door now stops a hostile leaf reaching most of
    # them; measuring the union with the door in place would count the
    # sites the guards happen to be in front of today, which is a
    # different and much smaller thing, and would shrink silently the next
    # time the door widened.
    sweep_sites = {e.message
                   for e in _neutered_sweep(canonicalization=False)[0]}
    assert len(sweep_sites) == _NEUTERED_NO_CANON["messages"], sorted(sweep_sites)

    row_sites: set[int] = set()
    refused_cleanly = 0
    saved = (ir._safe_repr, ir._safe_type_name, ir._safe_str)
    saved_reads = ir._CANONICAL_READS
    try:
        ir._safe_repr = repr
        ir._safe_type_name = lambda o: type(o).__name__
        ir._safe_str = str
        ir._CANONICAL_READS = tuple((b, lambda v: v) for b, _ in saved_reads)
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
        ir._CANONICAL_READS = saved_reads
    assert (ir._safe_repr, ir._safe_type_name, ir._safe_str) == saved
    assert ir._CANONICAL_READS is saved_reads

    assert len(sweep_sites | row_sites) == _QUOTE_SITE_UNION, (
        f"the record says {_QUOTE_SITE_UNION} distinct quote sites and the "
        f"union measures {len(sweep_sites | row_sites)}: "
        f"sweep={sorted(sweep_sites)} rows={sorted(row_sites)}. "
        f"`CHANGELOG.md`, this module's docstring and `ir._safe_repr`'s "
        f"docstring all state it."
    )
    # and the addends the docstring decomposes it into, so the arithmetic
    # is the tree's and not an author's
    assert len(sweep_sites) == _NEUTERED_NO_CANON["messages"]
    assert len(row_sites - sweep_sites) == 2, sorted(row_sites - sweep_sites)

    # the grouping the record uses, counted from the rows themselves
    groups = {g: sum(1 for x, _l, _b in _NAMED_SITES if x == g)
              for g in (GROUP_UNNAMED, GROUP_MASKED, GROUP_AUDIT4)}
    assert groups == {GROUP_UNNAMED: 2, GROUP_MASKED: 3, GROUP_AUDIT4: 4}, groups
    assert sum(groups.values()) == len(_NAMED_SITES) == 9
    # the tenth QUOTE has no row: `NamedTupleParam`'s field name, which
    # the sweep reaches. Counted here, NOT added to the union above —
    # rows and message expressions are different units (see the docstring).
    assert _SWEEP_ONLY_SITES == 1
