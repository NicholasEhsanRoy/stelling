# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""``SOUNDNESS.md``'s retrospective IR SCREEN, EXECUTED.

The S12′ entry publishes a four-clause instruction a reader is supposed to
run over an old query to decide whether a past verdict is invalidated by
the declaration-shape defect. It is a **remediation instruction**, and a
remediation instruction that says ALL CLEAR on an actually-false VERIFIED
is worse than no instruction at all — which is what the screen it replaced
did, on the very document its own entry was written about.

So the clauses are implemented here, from the published text and NOT by
calling the library, and driven against documents whose status is known
independently. A screen that only ever runs in a reader's head is a claim
nobody checks.

**THREE THINGS THIS FILE PINS, all from audit 0.2.0 B6 audit 3.**

*Clause 4 had no convention for a binding it cannot read* (F6). It said
only "compare … if every reference agrees", which is silent about an
operand with no binding at all and about a `stelling_any` whose `shape`
param cannot be read. The same population supports three different
AFFECTED counts depending on which silence a reader fills in, and the
published figure sat in the middle of them — so the screen shipped more
PERMISSIVE than the code it screens for. The convention is now stated, and
it is the code's own: `_Slicer._one_shape_per_value` DECLINES an operand it
cannot bind and `_declared_shape` DECLINES a param it cannot read, so an
unreadable binding is AFFECTED, not skipped.

*The screen is WITNESS 1 ALONE* (F7). It reads no propagated boxes and no
const payloads, so it inherits the binding witness's documented blindness
exactly. Three classes read UNTOUCHED, and each is named below with the
tree machinery that actually catches it. This matters more than its
severity suggests **because the screen is retrospective**: it is pointed
at verdicts produced by OLDER trees with fewer checks, so "the current
tree catches these by other means" is not the reassurance it looks like.

*Clause 4 never reads a jaxpr's own outvars.* It walks equation operands,
which is what the emission's cross-check walks; a lie on the outvar list
itself is outside both.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from stelling import ir

REPO = pathlib.Path(__file__).resolve().parent.parent
SOUNDNESS = REPO / "SOUNDNESS.md"


# -- the published screen, implemented from its own text --------------------
#
# Deliberately NOT `stelling.coverage.sub_jaxprs` / `stelling.obligation`:
# the subject is the INSTRUCTION, and an instruction implemented by calling
# the thing it is screening cannot disagree with it.

UNREADABLE = "<unreadable>"


def _scopes_in(value):
    """Clause 1: every sub-jaxpr a param holds, however nested."""
    if isinstance(value, ir.ClosedJaxpr):
        yield value.jaxpr
    elif isinstance(value, ir.Jaxpr):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _scopes_in(item)
    elif isinstance(value, ir.NamedTupleParam):
        for _name, item in value.fields:
            yield from _scopes_in(item)


def _sub_scopes(eqn):
    for _k, v in eqn.params:
        yield from _scopes_in(v)


def _declared_param_shape(eqn):
    """Clause 2, first bullet: the ``shape`` PARAM, or UNREADABLE."""
    raw = dict(eqn.params).get("shape", ())
    if not isinstance(raw, (tuple, list)):
        return UNREADABLE
    try:
        dims = tuple(raw)
    except Exception:  # noqa: BLE001
        return UNREADABLE
    out = []
    for d in dims:
        try:
            k = int.__index__(int(d)) if isinstance(d, bool) else d.__index__()
        except Exception:  # noqa: BLE001
            return UNREADABLE
        if k < 0:
            return UNREADABLE
        out.append(k)
    return tuple(out)


def _bindings(jaxpr):
    """Clause 2: this scope's own bindings, by var id."""
    b = {}
    for v in jaxpr.constvars:
        b[v.id] = tuple(v.aval.shape)
    for v in jaxpr.invars:
        b[v.id] = tuple(v.aval.shape)
    for e in jaxpr.eqns:
        for ov in e.outvars:
            if not isinstance(ov, ir.Var):
                continue
            if e.primitive == "stelling_any":
                b[ov.id] = _declared_param_shape(e)
            else:
                b[ov.id] = tuple(ov.aval.shape)
    return b


def screen(closed, *, convention="fail-closed"):
    """Clauses 3 and 4. Returns (verdict, findings).

    ``convention`` fills in clause 4's silence about a binding that cannot
    be read:

    ``"strict-literal"``   only a READABLE binding that DISAGREES counts.
    ``"fail-closed"``      an unbindable operand or an unreadable
                           declaration param counts too — what
                           ``_one_shape_per_value`` and ``_declared_shape``
                           actually do.
    """
    findings = []

    def walk(jaxpr, enclosing):
        here = _bindings(jaxpr)
        chain = (here, *enclosing)  # clause 3: innermost first
        for e in jaxpr.eqns:
            for atom in e.invars:
                if not isinstance(atom, ir.Var):
                    continue
                for scope in chain:
                    if atom.id in scope:
                        bound = scope[atom.id]
                        break
                else:
                    findings.append(
                        (e.primitive, atom.id, tuple(atom.aval.shape),
                         "NO BINDING"))
                    continue
                if bound is UNREADABLE:
                    findings.append(
                        (e.primitive, atom.id, tuple(atom.aval.shape),
                         "UNREADABLE DECLARATION"))
                elif bound != tuple(atom.aval.shape):
                    findings.append(
                        (e.primitive, atom.id, tuple(atom.aval.shape), bound))
            for sub in _sub_scopes(e):
                walk(sub, chain)

    walk(closed.jaxpr, ())
    if convention == "strict-literal":
        hard = [f for f in findings if f[3] not in ("NO BINDING",
                                                    "UNREADABLE DECLARATION")]
    elif convention == "fail-closed":
        hard = findings
    else:  # pragma: no cover - guard against a typo in a caller
        raise AssertionError(f"unknown convention {convention!r}")
    return ("AFFECTED" if hard else "UNTOUCHED"), findings


# -- fixtures ---------------------------------------------------------------

LO, HI, CEIL = 1.0, 2.0, 4.5


def av(shape=(), dtype="float64"):
    return ir.Aval(kind="ShapedArray", shape=tuple(shape), dtype=dtype)


def V(i, a):
    return ir.Var(id=i, aval=a)


def decl(out, shape_param, *, install=False):
    """A `stelling_any`; `install` puts `shape_param` past `__post_init__`."""
    honest = tuple(out.aval.shape)
    e = ir.JaxprEqn(
        primitive="stelling_any", invars=(), outvars=(out,),
        params=(("dtype", "float64"), ("hi", HI), ("lo", LO),
                ("shape", honest if install else shape_param)))
    if install:
        object.__setattr__(e, "params", tuple(sorted(
            (("dtype", "float64"), ("hi", HI), ("lo", LO),
             ("shape", shape_param)), key=lambda kv: kv[0])))
    return e


def sum_le(x, n):
    """`reduce_sum(x) <= CEIL`, asserted — the S12′ shape."""
    s, pr, o = V(90, av()), V(91, av((), "bool")), V(92, av((), "bool"))
    return (
        ir.JaxprEqn(primitive="reduce_sum", invars=(x,), outvars=(s,),
                    params=(("axes", tuple(range(n))),
                            ("out_sharding", None))),
        ir.JaxprEqn(primitive="le",
                    invars=(s, ir.Literal(val=CEIL, aval=av())),
                    outvars=(pr,)),
        ir.JaxprEqn(primitive="stelling_assert", invars=(pr,), outvars=(o,)),
    ), o


def close(eqns, out, consts=(), constvars=()):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=tuple(constvars), invars=(),
                       outvars=(out,), eqns=tuple(eqns)),
        consts=tuple(consts))


# -- the screen does its job ------------------------------------------------


def test_the_screen_flags_the_declaration_lie_and_clears_its_control():
    """The document this whole entry is about: a `stelling_any` whose
    `shape` param says four elements and whose consumer reads two."""
    x = V(0, av((4,)))
    tail, out = sum_le(V(0, av((2,))), 1)
    q = close((decl(x, (4,)),) + tail, out)
    verdict, findings = screen(q)
    assert verdict == "AFFECTED", findings
    assert ("reduce_sum", 0, (2,), (4,)) in findings, findings

    # the unedited control: the same document with the reference honest
    tail_ok, out_ok = sum_le(x, 1)
    assert screen(close((decl(x, (4,)),) + tail_ok, out_ok))[0] == "UNTOUCHED"


def test_the_screen_reads_the_PARAM_and_not_the_outvar_aval():
    """Clause 2's first bullet, which is the clause that cleared the
    affected document before it existed. Here the aval and every reference
    agree at `(2,)` and only the PARAM says four: a screen that named the
    producing outvar's aval for every producer returns ALL CLEAR."""
    x2 = V(0, av((2,)))
    lying = decl(x2, (4,), install=True)
    tail, out = sum_le(x2, 1)
    q = close((lying,) + tail, out)
    assert screen(q)[0] == "AFFECTED", screen(q)[1]
    # ... and the reading that names the outvar aval instead does not see it
    b = {0: tuple(lying.outvars[0].aval.shape)}
    assert b[0] == (2,) == tuple(x2.aval.shape)


def test_an_ABSENT_shape_param_binds_at_the_scalar_it_emits():
    """Clause 2's parenthesis. The door blesses a declaration with no
    `shape` param — hand-built IR legitimately omits params — and the
    emission then mints ONE symbol for it, so the screen must read `()`."""
    x = V(0, av((4,)))
    e = ir.JaxprEqn(
        primitive="stelling_any", invars=(), outvars=(x,),
        params=(("dtype", "float64"), ("hi", HI), ("lo", LO)))
    tail, out = sum_le(x, 1)
    verdict, findings = screen(close((e,) + tail, out))
    assert verdict == "AFFECTED", findings
    assert ("reduce_sum", 0, (4,), ()) in findings, findings


# -- F6: clause 4's convention, and the three answers it used to allow ------


def _unreadable_param_query():
    """Door-legal on its face; the `shape` param is installed past
    `__post_init__` and cannot be read as extents."""
    x = V(0, av((4,)))
    tail, out = sum_le(x, 1)
    return close((decl(x, object(), install=True),) + tail, out)


def _unbindable_operand_query():
    """A reference to a variable nothing in any scope binds."""
    x = V(0, av((4,)))
    tail, out = sum_le(V(7, av((4,))), 1)   # 7 is bound nowhere
    return close((decl(x, (4,)),) + tail, out)


@pytest.mark.parametrize(
    "make,why",
    [(_unreadable_param_query, "UNREADABLE DECLARATION"),
     (_unbindable_operand_query, "NO BINDING")],
    ids=["unreadable-shape-param", "unbindable-operand"],
)
def test_clause_4s_convention_is_the_CODES_and_it_is_fail_closed(make, why):
    """AUDIT 0.2.0 B6 AUDIT 3, F6 — the published clause 4 was SILENT here.

    It said only "compare … if every reference agrees with its binding",
    which answers nothing about an operand with no binding and nothing
    about a declaration whose `shape` param cannot be read as extents. Two
    readings are defensible from the text and they disagree, so the screen
    as published was more PERMISSIVE than the code it screens for — a
    reader following it could clear a query the tool itself refuses.

    The convention is now stated in the entry, and it is the one the code
    already implements: `_Slicer._one_shape_per_value` declines an operand
    it cannot bind, and `_declared_shape` declines a param it cannot read.
    A screen may be conservative where the code is conservative; it may not
    be lenient where the code is strict."""
    q = make()
    strict, findings = screen(q, convention="strict-literal")
    closed_v, _ = screen(q, convention="fail-closed")
    assert any(f[3] == why for f in findings), findings
    assert strict == "UNTOUCHED", (
        "this document is exactly where the two readings part; if the "
        "strict-literal reading now flags it, the fixture has drifted"
    )
    assert closed_v == "AFFECTED"

    # AND THE CODE AGREES WITH THE FAIL-CLOSED READING, which is the whole
    # argument for preferring it
    from stelling.obligation import DeclinedObligation, slice_obligation

    item = slice_obligation(q, 0, {})
    assert isinstance(item, DeclinedObligation), item
    assert "internal error" not in item.reason, item.reason


def test_the_entrys_clause_4_states_the_convention():
    """The prose half: an instruction whose silence a reader has to fill
    in is not an instruction. Goes red if clause 4 loses the sentence."""
    text = SOUNDNESS.read_text(encoding="utf-8")
    anchor = "> **4. Then**:"
    assert anchor in text, "clause 4 has moved; this pin needs re-siting"
    clause = text[text.index(anchor):text.index(anchor) + 1400].lower()
    for phrase in ("cannot be read", "affected", "may never be looser"):
        assert phrase in clause, (
            f"clause 4 no longer states what to do with a binding that "
            f"{phrase!r}; the published screen is then more permissive "
            f"than the code it screens for"
        )


# -- F7: the screen is witness 1 alone --------------------------------------


def test_a_CONSISTENTLY_relabelled_computed_value_reads_UNTOUCHED():
    """BLIND CLASS 1. The screen is the binding witness and nothing else,
    so it inherits its documented blindness exactly: a lie applied at the
    binding AND at every reference leaves no disagreement in the IR to
    find. Here `mul`'s outvar and every reference to it say `(2,)` while
    the operands say `(4,)`.

    WHAT CATCHES IT, MEASURED — and it is not one thing. The propagated
    box disagrees (`interval propagation computed a box of shape (4,)`),
    and it is the witness a consistent lie cannot forge in general. But on
    THIS document the emission's own elementwise pairing rule catches it
    too, with the box withheld entirely (`env={}`): *"'mul': operand
    shapes (4,) and (4,) broadcast to (4,), not the output shape (2,)"*.
    Both are recorded, because "the box is the sole detector" is the kind
    of claim this entry has already been wrong about twice.

    Which is why the class still matters for a RETROSPECTIVE screen: what
    refuses the document today is a property of today's tree, and the
    screen is pointed at verdicts produced by trees with fewer checks."""
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env, propagate

    def build(mul_shape):
        x = V(0, av((4,)))
        t = V(4, av(mul_shape))
        s, pr, o = V(1, av()), V(2, av((), "bool")), V(3, av((), "bool"))
        return close((
            decl(x, (4,)),
            ir.JaxprEqn(primitive="mul", invars=(x, x), outvars=(t,),
                        params=()),
            ir.JaxprEqn(primitive="reduce_sum", invars=(t,), outvars=(s,),
                        params=(("axes", (0,)), ("out_sharding", None))),
            ir.JaxprEqn(primitive="le",
                        invars=(s, ir.Literal(val=CEIL, aval=av())),
                        outvars=(pr,)),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pr,),
                        outvars=(o,)),
        ), o)

    lie = build((2,))
    verdict, findings = screen(lie)
    assert verdict == "UNTOUCHED", findings
    assert findings == [], findings

    p = propagate(lie)
    assert p is not None
    env = interval_env(lie)
    boxed = slice_obligation(lie, 0, env)
    assert isinstance(boxed, DeclinedObligation), boxed
    assert "computed a box of shape (4,)" in boxed.reason, boxed.reason
    assert "BOUND at shape" not in boxed.reason, (
        "the binding witness must NOT be what fires — the lie is consistent"
    )

    withheld = slice_obligation(lie, 0, {})
    assert isinstance(withheld, DeclinedObligation), withheld
    assert "broadcast to (4,), not the output shape (2,)" in withheld.reason

    # the unedited control passes both ways
    control = build((4,))
    assert screen(control)[0] == "UNTOUCHED"
    for e in (interval_env(control), {}):
        assert not isinstance(
            slice_obligation(control, 0, e), DeclinedObligation)


def test_a_constvar_whose_aval_contradicts_its_PAYLOAD_reads_UNTOUCHED():
    """BLIND CLASS 2. Clause 4 compares SHAPES recorded in the IR. A
    constvar's payload is a different kind of self-description — bytes,
    or a python scalar — and no clause reads it, so a constvar whose aval
    says two elements while its value decodes to four is ALL CLEAR here.

    `stelling.obligation._Slicer.slice` catches it, in its const decode
    pass, and quotes both counts."""
    import struct

    from stelling.obligation import DeclinedObligation, slice_obligation

    c = V(5, av((2,)))
    x = V(0, av((2,)))
    t = V(93, av((2,)))
    s, pr, o = V(90, av()), V(91, av((), "bool")), V(92, av((), "bool"))
    eqns = (
        decl(x, (2,)),
        ir.JaxprEqn(primitive="add", invars=(x, c), outvars=(t,), params=()),
        ir.JaxprEqn(primitive="reduce_sum", invars=(t,),
                    outvars=(s,),
                    params=(("axes", (0,)), ("out_sharding", None))),
        ir.JaxprEqn(primitive="le",
                    invars=(s, ir.Literal(val=CEIL, aval=av())),
                    outvars=(pr,)),
        ir.JaxprEqn(primitive="stelling_assert", invars=(pr,), outvars=(o,)),
    )
    payload = ir.Array(dtype="<f8", shape=(2,),
                       data=struct.pack("<2d", 1.0, 1.0))
    q = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(c,), invars=(), outvars=(o,), eqns=eqns),
        consts=(payload,))
    assert screen(q)[0] == "UNTOUCHED"

    # now make the PAYLOAD hold four while the constvar aval still says
    # two — a disagreement no clause of the screen reads. Installed past
    # `ClosedJaxpr.__post_init__`, which pairs a const against its
    # constvar's aval and refuses this; the point is that the SCREEN is
    # blind to the class, not that the door is.
    four = ir.Array(dtype="<f8", shape=(4,),
                    data=struct.pack("<4d", 1.0, 1.0, 1.0, 1.0))
    bad = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(c,), invars=(), outvars=(o,), eqns=eqns),
        consts=(payload,))
    object.__setattr__(bad, "consts", (four,))
    assert screen(bad)[0] == "UNTOUCHED", (
        "the screen reads no const payload; if it now does, this entry's "
        "blindness list is out of date"
    )
    item = slice_obligation(bad, 0, {})
    assert isinstance(item, DeclinedObligation), item
    assert "decodes to 4 element(s)" in item.reason, item.reason
    assert "aval shape (2,) holds 2" in item.reason, item.reason


def test_a_lie_on_a_jaxprs_own_OUTVARS_reads_UNTOUCHED():
    """BLIND CLASS 3. Clause 4 walks "for every equation, for every operand"
    — equation INVARS. A jaxpr's own outvar list is not an operand of any
    equation, so a `Var` there carrying a shape that disagrees with its
    binding is never compared."""
    x = V(0, av((4,)))
    tail, _out = sum_le(x, 1)
    lying_out = V(92, av((3,), "bool"))   # bound at () by stelling_assert
    q = close((decl(x, (4,)),) + tail, lying_out)
    assert screen(q)[0] == "UNTOUCHED", screen(q)[1]
    # the disagreement is real: the binding for id 92 is the scalar the
    # assert produces
    assert _bindings(q.jaxpr)[92] == ()
    assert tuple(lying_out.aval.shape) == (3,)


def test_the_entry_names_the_screens_blind_classes():
    """The prose half of F7. "The ONE screen" invites a reader to treat an
    UNTOUCHED as a clean bill of health; it is witness 1 alone, and the
    three classes above are named in the entry with what catches each."""
    text = SOUNDNESS.read_text(encoding="utf-8")
    heading = "**WHAT THIS SCREEN IS BLIND TO"
    assert heading in text, (
        "the blindness paragraph is gone; the screen then reads as total"
    )
    window = text[text.index(heading):][:2600]
    for phrase in ("consistently", "constvar", "outvars"):
        assert phrase in window, (
            f"the blindness paragraph no longer names {phrase!r}"
        )


# -- F5: the attribution table is PUBLISHED, and its arithmetic is checked --

CHANGELOG = REPO / "CHANGELOG.md"


def test_the_batch_ships_an_attribution_table_that_adds_up():
    """AUDIT 0.2.0 B6 AUDIT 3, F5 — the batch claimed *"every code change
    was reverted ALONE and the claiming tests go red"* and shipped NO
    table; `grep -ic revert` over its whole documentation diff returned 0,
    in a tree that had published one for three prior batches.

    A claim of that shape is worth exactly what a reader can re-derive, so
    the table and its census method are now in `CHANGELOG.md` and this
    checks three things about them: the method is stated, the table's own
    arithmetic holds (`raw - confounds = NET` on every row), and the
    anti-vacuity half is there — a PROSE revert must red nothing, which is
    what makes a SEMANTIC row's reds attributable to behaviour rather than
    to line numbers.

    It also holds the row this finding turns on: the slice-input reader
    reds NOTHING when reverted alone, and is recorded as unreachable AS A
    GUARD rather than claimed. `docs/norms.md` forbids proving guard
    coverage by construction; a batch that says "each change reds a test"
    with one such site in it has done exactly that."""
    text = CHANGELOG.read_text(encoding="utf-8")
    assert "CENSUS METHOD" in text, "the census method is not published"
    for phrase in ("SEMANTIC", "PROSE", "docstring-stripped AST",
                   "UNREACHABLE AS A GUARD", "git clone"):
        assert phrase in text, f"the attribution method no longer says {phrase!r}"

    anchor = "  revert (hunks, -U3)"
    assert anchor in text, "the attribution table has moved or gone"
    block = text[text.index(anchor):]
    block = block[:block.index("```")]
    rows, prose_rows = [], []
    for ln in block.split("\n"):
        parts = ln.split()
        # a row is "<label ...> raw conf NET <tests...>"; find the first
        # run of three integers
        nums = [(i, p) for i, p in enumerate(parts) if p.isdigit()]
        if len(nums) < 3:
            continue
        i = nums[0][0]
        if [p[0] for p in nums[:3]] != [i, i + 1, i + 2]:
            continue
        raw, conf, net = (int(parts[i]), int(parts[i + 1]), int(parts[i + 2]))
        assert raw - conf == net, (
            f"attribution row does not add up: {ln.strip()!r} "
            f"({raw} - {conf} != {net})"
        )
        (prose_rows if parts[0].startswith("P") else rows).append(
            (parts[0], net))
    assert len(rows) >= 11, f"only {len(rows)} semantic rows parsed: {rows}"
    assert len(prose_rows) == 3, f"prose controls: {prose_rows}"
    assert all(n == 0 for _, n in prose_rows), (
        f"a PROSE revert red something: {prose_rows} — either the "
        f"classification is wrong or the confounds are not what they say"
    )
    zero = [name for name, n in rows if n == 0]
    assert zero == ["R1c"], (
        f"exactly one semantic revert is expected to red nothing (the "
        f"slice-input reader, unreachable as a difference); got {zero}"
    )


# ---------------------------------------------------------------------------
# an attribution row names TESTS, and a test name is checkable
# ---------------------------------------------------------------------------

def _table_blocks(text: str) -> list[str]:
    return [b for b in re.findall(r"```\n(.*?)```", text, re.S)
            if "the tests that red" in b]


def _table_rows(block: str):
    """(row label, the wrapped lines of its "tests that red" column).

    The column wraps MID-IDENTIFIER with no marker, so the caller resolves
    a row under both joinings — see the test below.
    """
    label, tail, prev_first = None, [], None
    for ln in block.split("\n"):
        if "the tests that red" in ln:
            continue
        parts = ln.split()
        nums = [i for i, p in enumerate(parts) if p.isdigit()]
        start = None
        for k in range(len(nums) - 2):
            if nums[k + 1] == nums[k] + 1 and nums[k + 2] == nums[k] + 2:
                start = nums[k]
                break
        indent = len(ln) - len(ln.lstrip(" "))
        if start is not None:
            if label:
                yield label, list(tail)
            first = parts[0]
            # a row's label can sit on the line ABOVE its numbers when the
            # mutation description wraps (`OPT` does)
            label = first if re.match(r"^[A-Z(]", first) else (prev_first or first)
            tail = [" ".join(parts[start + 3:])]
        elif label is not None and indent >= 40:
            tail.append(ln.strip())
        if parts and start is None and indent < 40:
            prev_first = parts[0]
    if label:
        yield label, tail


def _live_test_names() -> set[str]:
    names: set[str] = set()
    for f in (REPO / "tests").rglob("test_*.py"):
        names |= set(re.findall(r"^def (test_\w+)", f.read_text(encoding="utf-8"), re.M))
    return names


def _quoted_fragments(text: str) -> set[str]:
    return {t.strip("_")
            for t in re.findall(r"[A-Za-z][A-Za-z0-9_]{12,}", text)
            if "_" in t}


def test_an_attribution_row_may_not_quote_a_test_that_does_not_exist():
    """AUDIT 0.2.0 B6 AUDIT 5, F4 — an attribution row's value is that a
    reader can run the test it names.

    The `tests that red` column is a claim of exactly that shape, so it is
    checkable: every identifier in it must be a fragment of a test that
    exists in this tree, or the entry must carry a RENAME ANNOTATION that
    names the row and the replacement. Both are required — an annotation
    on the wrong row leaves the right row unexplained, which is the
    finding. `R7` and `OPT` both quote
    `..._reads_the_EXTENTS_not_the_param_type` and for different reasons
    (`R7` runs against a clone at `d6b6d0b`, `OPT` on `30d4b04`); the note
    sat on `OPT` alone and named `d6b6d0b`, which is the tree `OPT`'s
    mutation REPRODUCES rather than the one it runs on.

    The column wraps mid-identifier with no marker, so a row is resolved
    under either joining — glued or spaced — and only counts as unresolved
    when BOTH fail. That is why `F3`'s `the three ..._hostile___repr__ /
    tests, one per module` is not a false positive.
    """
    text = CHANGELOG.read_text(encoding="utf-8")
    live = _live_test_names()
    assert len(live) > 1000, len(live)
    blocks = _table_blocks(text)
    assert len(blocks) >= 3, f"{len(blocks)} attribution table(s) found"

    # the annotations, taken from OUTSIDE the fenced blocks
    prose = re.sub(r"```\n.*?```", "", text, flags=re.S)
    annotations = [p for p in prose.split("\n\n") if "renam" in p.lower()]

    unexplained = []
    for block in blocks:
        for label, tail in _table_rows(block):
            best = None
            for joiner in ("", " "):
                got = _quoted_fragments(joiner.join(tail))
                bad = {g for g in got if not any(g in n for n in live)}
                if best is None or len(bad) < len(best):
                    best = bad
            if not best:
                continue
            covered = any(
                label in ann and any(n in ann for n in live)
                for ann in annotations
            )
            if not covered:
                unexplained.append((label, sorted(best)))

    assert not unexplained, (
        "an attribution row names a test that is not in this tree and "
        "carries no rename annotation naming the row and its replacement:\n"
        + "\n".join(f"    row {lab}: {frags}" for lab, frags in unexplained)
    )


def test_the_R1c_disclosure_EXHIBITS_its_pre_emption():
    """AUDIT 0.2.0 B6 AUDIT 5, F6 — `docs/norms.md` clause 1, restated so
    it is decidable, and then applied to the one site this batch records
    under it.

    The clause used to read *"it has no refusal of its own … a site with a
    refusal branch NEVER qualifies"*. Read strictly that disqualifies
    every guard in this codebase — almost all of them are bare calls to
    refusing helpers — so nothing could ever be recorded as unreachable as
    a guard, `R1c` included. It discriminated syntax where it means to
    discriminate semantics.

    Restated, it asks whether the site is the FIRST thing that would
    refuse, and it asks for the document. This is the document: a
    declaration whose `shape` param is `bytes`, installed past
    `__post_init__`. `_Slicer._declared_shape` is called ONCE, by the
    element budget, and the slice-input reader `R1c` records is never
    reached.
    """
    import traceback

    import stelling.obligation as OB
    from stelling import ir

    calls: list[str] = []
    original = OB._Slicer._declared_shape

    def _spy(self, decl, vid):
        frame = traceback.extract_stack()[-2]
        calls.append(f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}")
        return original(self, decl, vid)

    def av(shape=(), dtype="float64"):
        return ir.Aval(kind="ShapedArray", shape=shape, dtype=dtype)

    x = ir.Var(id=1, aval=av((2,)))
    s = ir.Var(id=2, aval=av())
    pred = ir.Var(id=3, aval=av((), "bool"))
    out = ir.Var(id=4, aval=av((), "bool"))
    decl = ir.JaxprEqn(
        primitive="stelling_any", invars=(), outvars=(x,),
        params=(("dtype", "float64"), ("hi", 2.0), ("lo", 1.0),
                ("shape", (2,))))
    object.__setattr__(decl, "params", tuple(sorted(
        (("dtype", "float64"), ("hi", 2.0), ("lo", 1.0),
         ("shape", b"\x02\x02")), key=lambda kv: kv[0])))
    q = ir.ClosedJaxpr(jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=(out,), eqns=(
        decl,
        ir.JaxprEqn(primitive="reduce_sum", invars=(x,), outvars=(s,),
                    params=(("axes", (0,)), ("out_sharding", None))),
        ir.JaxprEqn(primitive="le",
                    invars=(s, ir.Literal(val=9.9, aval=s.aval)),
                    outvars=(pred,)),
        ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
    )))

    OB._Slicer._declared_shape = _spy
    try:
        item = OB.slice_obligation(q, 0, {})
    finally:
        OB._Slicer._declared_shape = original

    assert isinstance(item, OB.DeclinedObligation), item
    assert "of type bytes" in item.reason, item.reason
    assert len(calls) == 1, (
        f"`_declared_shape` was called {len(calls)} time(s): {calls}. The "
        f"pre-emption clause 1 requires is that the BUDGET's call refuses "
        f"before the slice-input reader runs"
    )

    # the two call sites, read off the source so a moved line does not
    # turn the exhibit into folklore
    lines = [i + 1 for i, ln in enumerate(
        pathlib.Path(OB.__file__).read_text(encoding="utf-8").splitlines())
        if "self._declared_shape(self.producers[vid], vid)" in ln]
    assert len(lines) == 2, lines
    budget_line, slice_input_line = min(lines), max(lines)
    assert calls == [f"obligation.py:{budget_line}"], (calls, lines)
    assert slice_input_line not in {int(c.split(":")[1]) for c in calls}

    # ... and the clause and the row both say so
    norms = (REPO / "docs" / "norms.md").read_text(encoding="utf-8")
    assert "A site with a refusal branch NEVER qualifies" not in norms, (
        "clause 1 still disqualifies every bare call to a refusing helper, "
        "which is every guard in this codebase"
    )
    assert "PRE-EMPTED, and the pre-emption is" in norms, norms[:0]
    changelog = CHANGELOG.read_text(encoding="utf-8")
    assert "THE PRE-EMPTION IS NOW EXHIBITED" in changelog
    assert f"obligation.py:{budget_line} in slice" in changelog, (
        f"the R1c exhibit quotes a call site that is no longer the "
        f"budget's (now line {budget_line})"
    )
