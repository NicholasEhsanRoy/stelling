# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE ``shape``-PARAM CONTAINER RULE, MEASURED RATHER THAN ASSERTED.

`ir._SHAPE_PARAM_CONTAINERS` is the whole of the rule a declaration's
``shape`` param is judged by: `ir._validate_decl_eqn` refuses a param that
is not one of those types, and `obligation._Slicer._declared_shape`
declines it. Both ask that one object.

**WHY THIS FILE EXISTS.** Four consecutive audits of this batch found the
same defect: a principle stated in prose beside code that does something
else. The fifth instance was the door's own docstring — it described "any
sequence of extents is compared, and a param that is not a sequence of
extents at all is refused" for two commits after the code had become a
check on the param's PYTHON TYPE, under which `range`, `array.array`,
`memoryview` and every custom iterable are sequences of extents and are
all refused. The sentence was true when written and false when read, and
nothing could tell.

An accept/refuse partition over container types is not an opinion. It can
be COMPUTED, and then prose has something to be wrong against. This file
enumerates container types, measures what each face really does with one,
and requires the measured partition to be exactly the rule.

**WHAT THIS PIN COVERS**

* The population is SCANNED, not typed out: every ``type`` exported by
  `builtins`, `collections`, `collections.abc`, `array`, `queue` and
  `types` is put through a battery of generic constructor arguments, and
  everything that both constructs and iterates joins the population. So a
  container type nobody remembered — the next `memoryview` — is in the
  population by arriving in one of those namespaces, not by being noticed.
* SUBCLASSES ARE DERIVED, not listed: every surviving type that can be
  subclassed contributes a dynamically built subclass instance, so "a
  `tuple` subclass is accepted and a `str` subclass is refused" is
  measured for every type in the population rather than for the two
  someone thought of.
* An iteration that RAISES is derived the same way, for every accepted
  container type: `isinstance` is a claim about the type and not about the
  object, and that arm must decline rather than escape.
* An iteration that ANSWERS INCONSISTENTLY is derived the same way, and
  for the same reason — audit 0.2.0 B6 audit 5, F1. `isinstance` says
  nothing about what a second read returns, and an `__iter__` yielding
  `(2, 2)` once and `(1,)` afterwards is a well-formed param to the
  door's single read. What that arm must do is not decline: it is that
  the door INSTALLS the extents it validated, so the second answer is
  never asked for. That is measured in
  :func:`test_the_door_INSTALLS_what_it_VALIDATED_so_a_second_read_cannot_differ`,
  and the false VERIFIED it closes is held in
  `tests/test_aval_lie_both_faces.py`.
* BOTH FACES, and their agreement. The door (`ir`, at construction) and
  the emission (`obligation._Slicer._declared_shape`) are measured
  separately and required to partition identically. The soundness
  direction is EMISSION ⊆ DOOR: a container the emission will read is one
  the door must have compared against the outvar aval, because the
  emission mints one SMT term per element of that param and an
  uncompared param is exactly finding UNSOUND-1.

**WHAT THIS PIN DOES NOT COVER, said plainly**

* It is not a proof that the population is complete. A container type
  defined in a module the scan does not visit — numpy's `ndarray` is the
  live example, so it is added by hand below — is measured only because
  someone added it. The scan narrows "the types I remembered" to "the
  types in six standard namespaces"; it does not eliminate the class.
* It does not pin the rule against being widened. Everything derives from
  one tuple deliberately, so changing that tuple changes the rule, the
  messages and the expected partition together — which is the point of
  writing it once. What guards the VALUE is
  :func:`test_the_rule_itself_is_pinned_to_tuple_and_list`, one
  deliberate line that reds on any change.
* It does not cover the THIRD reader of a declaration's shape param.
  `propagate`'s ``stelling_any`` transfer calls ``tuple(shape)`` on
  anything iterable and is held to no container rule at all; that
  asymmetry is measured in
  :func:`test_the_TRANSFER_face_is_NOT_held_to_the_container_rule` and
  recorded as a disclosed live expectation, not fixed here.
* It says nothing about extent VALUES. Non-integer and negative extents
  are a different guard (`ir._load_extents` / `obligation._extents`) with
  its own tests.
* The EMISSION face is measured PAST the door. :func:`_decl_eqn` installs
  the candidate param with `object.__setattr__` precisely so
  `_declared_shape` can be judged on its own, which means the
  inconsistent-`__iter__` row above measures the door's install and NOT
  the emission's behaviour on a drifting object. `_declared_shape` is
  still a function and not a single read, and handed a drifting param
  directly it will still answer differently between calls; what makes
  that unreachable from a document is that every `ir.JaxprEqn` — the only
  thing `slice` is ever given — carries what the door installed. A
  hand-built equation that bypasses `__post_init__`, as this file's own
  helper does, is outside that containment and is not covered here.
  *(Until audit 0.2.0 B6 audit 6 the sentence before last was false as a
  claim about DOCUMENTS: the door installed through `k == "shape"`
  against a document-supplied KEY, and a `str` subclass answering that
  one comparison False left the raw drifting param on the equation with
  no `object.__setattr__` anywhere and reached `discharged`.
  `ir._canonical_param_keys` settles the keys before anything compares
  them; the measurement is
  `tests/test_aval_lie_both_faces.py::test_a_lying_param_KEY_can_no_
  longer_mint_a_FALSE_VERIFIED`.)*
* **The DOOR face is still judged by the container rule, and audit 6's
  generic canonicalization door does not stand in front of it.**
  `ir.JaxprEqn.__post_init__` runs `_validate_decl_eqn` BEFORE
  `_canonical_param_values`, deliberately: a `shape` param is refused by
  the rule that is about shapes, with the sentence that names it, rather
  than by the generic "this module has no exact form to store a
  `memoryview`" — which would refuse most of the population below for a
  broader and less useful reason. So the partition measured here is the
  same partition, and it is still `isinstance(param,
  ir._SHAPE_PARAM_CONTAINERS)`.

Audit 0.2.0 B6 audit 4, F1; audit 5, F1; audit 6.
"""
from __future__ import annotations

import array as _arraymod
import builtins
import collections
import collections.abc
import queue
import types

import pytest

from stelling import ir
import stelling.obligation as OB


# ---------------------------------------------------------------------------
# the population, COMPUTED
# ---------------------------------------------------------------------------

# Namespaces scanned for container types. Six standard modules rather than
# a list of names: `memoryview` and `array.array` are in the population
# because they live here, not because this batch's history mentions them.
_SCANNED = (builtins, collections, collections.abc, _arraymod, queue, types)

# A battery of generic constructor arguments. Each is tried on every type;
# a type that constructs from any of them and then ITERATES joins the
# population. The two-argument forms are what a typecode-plus-data
# container (`array.array`) needs — a shape of constructor, not a named
# type.
def _arg_batteries():
    return (
        ([2, 2],),
        ((2, 2),),
        (b"\x02\x02",),
        ("\x02\x02",),
        (range(2),),
        (2,),
        (),
        ({2: 0, 3: 0},),
        ("b", b"\x02\x02"),
        ("i", [2, 2]),
    )


def _iterates(obj) -> bool:
    try:
        list(obj)
    except Exception:  # noqa: BLE001 — not iterable is not a candidate
        return False
    return True


def _scanned_candidates() -> list[tuple[str, object]]:
    """(label, object) for every container the scan can build."""
    out: list[tuple[str, object]] = []
    seen: set[type] = set()
    for mod in _SCANNED:
        for name in sorted(dir(mod)):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name, None)
            if not isinstance(obj, type) or obj in seen:
                continue
            for args in _arg_batteries():
                try:
                    inst = obj(*args)
                except Exception:  # noqa: BLE001 — most types will not
                    continue
                if not _iterates(inst):
                    continue
                seen.add(obj)
                out.append((f"{mod.__name__}.{name}{args!r}", inst))
                break
    return out


def _subclass_candidates(base_objs) -> list[tuple[str, object]]:
    """A DERIVED subclass instance for every scanned type that allows one.

    `isinstance` is the rule, so what a subclass does is the rule's own
    edge and may not rest on anyone having listed `tuple` and `str`.
    """
    out: list[tuple[str, object]] = []
    for label, obj in base_objs:
        base = type(obj)
        try:
            sub = types.new_class("Sub_" + base.__name__, (base,))
        except Exception:  # noqa: BLE001 — final types refuse
            continue
        try:
            inst = sub(obj) if base is not tuple else sub(tuple(obj))
        except Exception:  # noqa: BLE001
            try:
                inst = sub(list(obj))
            except Exception:  # noqa: BLE001
                continue
        if not _iterates(inst):
            continue
        out.append((f"subclass of {base.__name__} (from {label})", inst))
    return out


def _refusing_iter_candidates() -> list[tuple[str, object]]:
    """For every ACCEPTED container type, a subclass whose ``__iter__``
    raises — the arm `isinstance` cannot see.

    A base that cannot be subclassed or cannot be built empty is SKIPPED
    rather than crashing the sweep: this helper reads
    `ir._SHAPE_PARAM_CONTAINERS`, so a widened rule must not turn the
    measurement into a harness error that reds for the wrong reason. What
    guards the rule's VALUE is
    :func:`test_the_rule_itself_is_pinned_to_tuple_and_list`, deliberately
    and alone.
    """
    out: list[tuple[str, object]] = []
    for base in ir._SHAPE_PARAM_CONTAINERS:
        def _raise(self):
            raise RuntimeError("will not iterate")

        try:
            sub = types.new_class(
                "Unreadable_" + base.__name__, (base,),
                exec_body=lambda ns: ns.update({"__iter__": _raise}),
            )
            inst = sub()
        except Exception:  # noqa: BLE001 — final or argument-taking bases
            continue
        out.append((f"{base.__name__} subclass whose __iter__ raises", inst))
    return out


def _lying_iter_candidates() -> list[tuple[str, object]]:
    """For every ACCEPTED container type, a subclass whose ``__iter__``
    answers ``(2, 2)`` once and ``(1,)`` afterwards — audit 0.2.0 B6
    audit 5, F1.

    The sibling of :func:`_refusing_iter_candidates`, and the arm it did
    not have. An `__iter__` that RAISES is caught by the door's `try`; an
    `__iter__` that merely answers DIFFERENTLY is a perfectly well-formed
    param on the one read the door takes, and was the object that minted a
    false VERIFIED. `(2, 2)` is `_AVAL`'s shape, so these rows agree with
    the outvar aval on the door's read and are ACCEPTED — the measurement
    is what the door then stores, not whether it refuses.

    Skipped, not crashed, for a base that cannot be subclassed or built
    empty, for the reason :func:`_refusing_iter_candidates` gives."""
    out: list[tuple[str, object]] = []
    for base in ir._SHAPE_PARAM_CONTAINERS:
        state = {"n": 0}

        def _drift(self, state=state):
            state["n"] += 1
            return iter((2, 2) if state["n"] <= 1 else (1,))

        try:
            sub = types.new_class(
                "Drifting_" + base.__name__, (base,),
                exec_body=lambda ns: ns.update({"__iter__": _drift}),
            )
            inst = sub()
        except Exception:  # noqa: BLE001 — final or argument-taking bases
            continue
        out.append(
            (f"{base.__name__} subclass whose __iter__ answers "
             f"inconsistently", inst)
        )
    return out


def _numpy_candidates() -> list[tuple[str, object]]:
    """numpy is not in a scanned namespace and its own namespace is far
    too large to sweep, so its containers are ADDED. This is the part of
    the population that rests on someone having thought of it, and it is
    separated out for exactly that reason."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001 — the core has no numpy dependency
        return []
    return [
        ("np.array([2, 2])", np.array([2, 2])),
        ("np.array(2)  [0-d]", np.array(2)),
        ("np.zeros((2, 2)).shape  [a plain tuple]", np.zeros((2, 2)).shape),
        ("tuple of np.int64", (np.int64(2), np.int64(2))),
    ]


def _population() -> list[tuple[str, object]]:
    scanned = _scanned_candidates()
    return (
        scanned
        + _subclass_candidates(scanned)
        + _refusing_iter_candidates()
        + _lying_iter_candidates()
        + _numpy_candidates()
    )


# ---------------------------------------------------------------------------
# the two faces, measured
# ---------------------------------------------------------------------------

_AVAL = ir.Aval(kind="ShapedArray", shape=(2, 2), dtype="float64")

# Message markers. Both faces compose their container refusal from
# `ir._SHAPE_PARAM_RULE`, so this marker is the sentence's own fixed part
# and the rule's name inside it cannot drift from the tuple.
_CONTAINER_MARK = "records its extents in"
_UNREADABLE_MARK = "whose iteration RAISES"

READ = "read"                      # the face got past the container gate
REFUSED_CONTAINER = "refused: container type"
REFUSED_UNREADABLE = "refused: will not iterate"


def _classify(message: str) -> str:
    if _CONTAINER_MARK in message:
        return REFUSED_CONTAINER
    if _UNREADABLE_MARK in message:
        return REFUSED_UNREADABLE
    return READ


def _door(param) -> str:
    """What `ir`'s construction-time door does with `param` as a
    declaration's ``shape``."""
    try:
        ir.JaxprEqn(
            primitive="stelling_any",
            invars=(),
            outvars=(ir.Var(id=0, aval=_AVAL),),
            params=(("dtype", "float64"), ("lo", 0.0), ("hi", 1.0),
                    ("shape", param)),
        )
        return READ
    except ir.TranscriptionError as exc:
        return _classify(str(exc))


def _decl_eqn(param) -> ir.JaxprEqn:
    """A declaration carrying `param`, installed PAST `__post_init__` so
    the emission face is measured without the door standing in front of
    it — `SOUNDNESS.md` scopes hand-built IR in, and `ir.ClosedJaxpr` is a
    public dataclass."""
    eqn = ir.JaxprEqn(
        primitive="stelling_any", invars=(),
        outvars=(ir.Var(id=0, aval=_AVAL),),
        params=(("dtype", "float64"), ("lo", 0.0), ("hi", 1.0),
                ("shape", (2, 2))),
    )
    object.__setattr__(eqn, "params", tuple(sorted(
        (("dtype", "float64"), ("lo", 0.0), ("hi", 1.0), ("shape", param)),
        key=lambda kv: kv[0])))
    return eqn


def _emission(param) -> str:
    """What `_Slicer._declared_shape` does with the same param."""
    sl = object.__new__(OB._Slicer)
    try:
        OB._Slicer._declared_shape(sl, _decl_eqn(param), 0)
        return READ
    except OB._Decline as exc:
        return _classify(exc.reason)


# ---------------------------------------------------------------------------
# the pin
# ---------------------------------------------------------------------------

def test_the_rule_itself_is_pinned_to_tuple_and_list():
    """The one deliberate line. Everything else in this file derives from
    `ir._SHAPE_PARAM_CONTAINERS`, so widening it would move the rule, the
    refusal messages and the expected partition together and red nothing.
    This is what makes that a decision rather than an edit.

    Widening it means accepting a container whose ``tuple(...)`` is not
    the extents the declaration stated: `tuple(b"34")` is `(51, 52)` and
    `tuple(memoryview(b"34"))` is the same pair — perfectly plausible
    extents nobody wrote, and the emission mints one SMT term per element
    of them.
    """
    assert ir._SHAPE_PARAM_CONTAINERS == (tuple, list), (
        "the shape-param container rule moved. Both faces and every "
        "refusal message read this tuple, so nothing else here can red on "
        "it: adding a type means a document whose extents are read out of "
        "a container jax never produced one from, and the emission mints "
        "one term per element of what it reads"
    )
    assert ir._SHAPE_PARAM_RULE == "a tuple or a list", ir._SHAPE_PARAM_RULE


def test_the_rule_is_stated_once_and_the_prose_points_at_it():
    """The mechanism, not the sentence: neither docstring restates the
    container list, and both name the object that holds it. A docstring
    that re-types the rule is the defect this file exists for."""
    door_doc = ir._validate_decl_eqn.__doc__
    emit_doc = OB._Slicer._declared_shape.__doc__
    assert "_SHAPE_PARAM_CONTAINERS" in door_doc, door_doc
    assert "_SHAPE_PARAM_CONTAINERS" in emit_doc, emit_doc

    # ... and the live refusals are composed FROM it, so the sentence a
    # user reads names the set that was actually applied
    door_msg = ""
    try:
        _door(b"\x02\x02")
    except Exception:  # noqa: BLE001 — unreachable; _door catches
        pass
    try:
        ir.JaxprEqn(
            primitive="stelling_any", invars=(),
            outvars=(ir.Var(id=0, aval=_AVAL),),
            params=(("dtype", "float64"), ("lo", 0.0), ("hi", 1.0),
                    ("shape", b"\x02\x02")),
        )
    except ir.TranscriptionError as exc:
        door_msg = str(exc)
    assert ir._SHAPE_PARAM_RULE in door_msg, door_msg
    assert "bytes" in door_msg, (
        f"the refusal no longer names the type it refused: {door_msg!r}. "
        f"`np.array([4])` was refused with 'is not a sequence of extents', "
        f"which is not why it was refused"
    )

    sl = object.__new__(OB._Slicer)
    try:
        OB._Slicer._declared_shape(sl, _decl_eqn(b"\x02\x02"), 0)
        raise AssertionError("the emission face accepted a bytes param")
    except OB._Decline as exc:
        assert ir._SHAPE_PARAM_RULE in exc.reason, exc.reason


def test_the_measured_partition_IS_the_documented_rule():
    """THE PIN. Over a computed population of container types, each face's
    real accept/refuse partition must equal ``isinstance(param,
    ir._SHAPE_PARAM_CONTAINERS)`` — and the two faces must partition
    identically.

    "Applied the rule" is the container gate alone: a `list` subclass
    whose `__iter__` raises passes `isinstance` and is then refused as
    UNREADABLE, which is a second and separate fact about the document and
    is checked as such.
    """
    population = _population()
    rows = []
    for label, param in population:
        predicted = isinstance(param, ir._SHAPE_PARAM_CONTAINERS)
        door = _door(param)
        emission = _emission(param)
        rows.append((label, predicted, door, emission))

    # 1. no face may escape raw on any of them — `_door` and `_emission`
    #    convert only the two declared exception types, so an escape has
    #    already failed the run by here; this is the sentence that says so.

    # 2. the door's container gate IS the rule
    wrong = [r for r in rows
             if (r[2] != REFUSED_CONTAINER) is not r[1]]
    assert not wrong, (
        "the DOOR's container gate no longer matches "
        f"isinstance(param, {ir._SHAPE_PARAM_CONTAINERS}) on "
        f"{len(wrong)} of {len(rows)} container(s):\n"
        + "\n".join(f"    {lab:58} rule={p} door={d}"
                    for lab, p, d, _ in wrong)
    )

    # 3. the emission's container gate IS the same rule
    wrong = [r for r in rows
             if (r[3] != REFUSED_CONTAINER) is not r[1]]
    assert not wrong, (
        "the EMISSION's container gate no longer matches the rule on "
        f"{len(wrong)} of {len(rows)} container(s):\n"
        + "\n".join(f"    {lab:58} rule={p} emission={e}"
                    for lab, p, _, e in wrong)
    )

    # 4. and the two faces agree EXACTLY. The soundness direction is
    #    emission ⊆ door: a container the emission reads must be one the
    #    door compared against the outvar aval, because the emission mints
    #    one term per element of it (UNSOUND-1). The other direction is
    #    liveness — a door that admits what the emission declines costs an
    #    UNKNOWN — and is held too, because a difference either way is a
    #    rule that exists in two copies.
    split = [r for r in rows if r[2] != r[3]]
    assert not split, (
        f"the door and the emission part company on {len(split)} of "
        f"{len(rows)} container(s), and one of the two directions is "
        f"UNSOUND-1:\n"
        + "\n".join(f"    {lab:58} door={d} emission={e}"
                    for lab, _, d, e in split)
    )

    # 5. the population must be worth measuring. A pin over an empty or
    #    degenerate population is the same defect one level up.
    accepted = [r for r in rows if r[1]]
    refused = [r for r in rows if not r[1]]
    unreadable = [r for r in rows if r[2] == REFUSED_UNREADABLE]
    assert len(refused) >= 15, (
        f"only {len(refused)} refused container(s) in the population; the "
        f"scan has stopped finding them and this pin is measuring almost "
        f"nothing"
    )
    assert len(accepted) >= 4, (
        f"only {len(accepted)} accepted container(s) — `tuple`, `list` and "
        f"a derived subclass of each are the floor"
    )
    assert len(unreadable) >= 2, (
        f"the will-not-iterate arm was measured {len(unreadable)} time(s); "
        f"`tuple` and `list` both admit a subclass whose `__iter__` raises, "
        f"so two is the floor (a rule containing a type that cannot be "
        f"subclassed contributes none, and that is skipped rather than "
        f"crashing the sweep)"
    )

    # 6. ... and it must still contain the types this batch's history is
    #    about. Not the rule — a floor on the scan, so a change that
    #    quietly stops building them is visible.
    present = {type(param).__name__ for _, param in population}
    for must in ("str", "bytes", "bytearray", "memoryview", "range", "array"):
        assert must in present, (
            f"the scan no longer builds a {must!r}; every one of these was "
            f"a real door-vs-emission inconsistency at some point in this "
            f"batch, and a population that has lost them proves less than "
            f"it did"
        )


def test_the_door_INSTALLS_what_it_VALIDATED_so_a_second_read_cannot_differ():
    """AUDIT 0.2.0 B6 AUDIT 5, F1 — the arm `isinstance` cannot see and
    the `try` cannot catch.

    The container rule is a claim about a param's TYPE. Reading it once
    and binding locally (audit 3's F1) makes the DOOR's own comparison
    honest and does nothing for the readers after it: the transfer in
    `propagate`, `_declared_shape`, `ir._encode` and
    `coverage.sub_jaxprs` each re-read the raw param, and a container
    that answers differently between reads gave them a different value
    from the one the door compared against the outvar aval.

    So the repair is not a fourth reader and not a stricter rule. It is
    that `JaxprEqn.__post_init__` installs the extents
    `ir._validate_decl_eqn` returned, and that is what this measures: over
    the whole computed population, every param the door ACCEPTS is stored
    as a plain `tuple` of plain `int`, equal to the outvar aval's extents.
    A row that stores the object it was handed is a row a second reader
    can be lied to about.
    """
    stored_rows = 0
    for label, param in _population():
        try:
            eqn = ir.JaxprEqn(
                primitive="stelling_any", invars=(),
                outvars=(ir.Var(id=0, aval=_AVAL),),
                params=(("dtype", "float64"), ("lo", 0.0), ("hi", 1.0),
                        ("shape", param)),
            )
        except ir.TranscriptionError:
            continue
        stored = eqn.params_dict()["shape"]
        assert type(stored) is tuple, (
            f"{label}: the door stored a {type(stored).__name__}, so a "
            f"later reader is reading the document's object and not the "
            f"door's answer"
        )
        assert all(type(k) is int for k in stored), (
            f"{label}: the door stored {stored!r}, whose extents are not "
            f"plain ints — `__index__` can answer twice"
        )
        assert stored == _AVAL.shape, (
            f"{label}: the door stored {stored!r} but validated the param "
            f"against the outvar aval shape {_AVAL.shape!r}"
        )
        stored_rows += 1

    assert stored_rows >= 4, (
        f"only {stored_rows} accepted row(s) reached the store check; "
        f"`tuple`, `list` and a derived subclass of each are the floor"
    )

    # ... and the drifting rows are IN that count, which is the finding.
    for label, param in _lying_iter_candidates():
        eqn = ir.JaxprEqn(
            primitive="stelling_any", invars=(),
            outvars=(ir.Var(id=0, aval=_AVAL),),
            params=(("dtype", "float64"), ("lo", 0.0), ("hi", 1.0),
                    ("shape", param)),
        )
        # the param would now answer (1,); the equation does not ask it
        assert tuple(param) == (1,), label
        assert eqn.params_dict()["shape"] == (2, 2), (
            f"{label}: the equation's shape moved to "
            f"{eqn.params_dict()['shape']!r} on a later read"
        )


def test_the_TRANSFER_face_is_NOT_held_to_the_container_rule():
    """RECORDED, NOT FIXED — the third reader, and it has no rule at all.

    `propagate`'s ``stelling_any`` transfer reads the same param with a
    bare ``tuple(shape)``, so it accepts every iterable the other two
    faces refuse. This is measured here rather than described, because
    "the propagation is stricter/looser than the emission" is exactly the
    kind of claim this batch keeps finding to be prose. Its sibling — the
    raw crash on a param that will not iterate at all — is pinned in
    `tests/test_aval_lie_both_faces.py` as a live expectation.

    Why it is not a verdict, STATED AGAINST WHAT THE CODE DOES — audit
    0.2.0 B6 audit 5, F1. The sentence that stood here was *"a document
    whose param and aval disagree is refused at the `ir` door on every
    deserialized route"*, and it was false of the document that mattered:
    a `tuple` subclass answering `(2,)` once and `(1,)` afterwards has a
    param that disagrees with its aval on every read after the first, and
    the door ACCEPTED it — the door compares one read, and one read
    agreed.

    What is true is narrower and is now a mechanism rather than a claim.
    The door compares ONE read of the param against the outvar aval and
    INSTALLS that read, so no reader after it — this transfer included —
    can be handed a different value than the one compared. A param the
    door never saw is therefore the only way to reach this transfer with
    a container the other two faces refuse, and that is what
    :func:`_decl_eqn` builds by writing past `__post_init__`: hand-built
    IR judged by interval propagation alone, which is the standing
    disclosure this transfer belongs to.
    """
    from stelling import interval as iv
    from stelling.propagate import TRANSFERS

    transfer, _tier = TRANSFERS["stelling_any"]

    # DRIVEN, not read off the source: the transfer is handed the same
    # declaration params the two faces above refused, and what it returns
    # is the measurement.
    read_by_the_transfer = []
    for label, param in _population():
        if isinstance(param, ir._SHAPE_PARAM_CONTAINERS):
            continue
        params = {"shape": param, "lo": 0.0, "hi": 1.0, "dtype": "float64"}
        try:
            (box,) = transfer(_decl_eqn(param), params, [])
        except Exception:  # noqa: BLE001 — refusing or crashing is not
            continue        # "reading it", and the crash is pinned elsewhere
        if isinstance(box, iv.IntervalArray):
            read_by_the_transfer.append((label, tuple(box.shape)))

    assert read_by_the_transfer, (
        "the transfer face no longer builds a box from any container the "
        "other two refuse. If that is because it acquired the same rule, "
        "this asymmetry has closed and the disclosure it records should go "
        "with the test"
    )
    # the shape of the asymmetry, quoted so a reader sees the size of it
    assert len(read_by_the_transfer) >= 3, read_by_the_transfer
    assert any(name.startswith("builtins.memoryview")
               or "memoryview" in name for name, _ in read_by_the_transfer), (
        f"`memoryview` — the container that drove this batch's positive "
        f"rule — is no longer among the ones the transfer reads: "
        f"{read_by_the_transfer}"
    )


@pytest.mark.parametrize(
    "param,expect",
    [
        ((2, 2), READ),
        ([2, 2], READ),
        (b"\x02\x02", REFUSED_CONTAINER),
        ("\x02\x02", REFUSED_CONTAINER),
        (memoryview(b"\x02\x02"), REFUSED_CONTAINER),
        (_arraymod.array("b", b"\x02\x02"), REFUSED_CONTAINER),
        (range(2), REFUSED_CONTAINER),
        (None, REFUSED_CONTAINER),
    ],
    ids=["tuple", "list", "bytes", "str", "memoryview", "array.array",
         "range", "None"],
)
def test_the_named_rows_read_the_way_the_record_says(param, expect):
    """The eight rows `SOUNDNESS.md` and `CHANGELOG.md` quote by name,
    held separately from the computed sweep so a reader can check the
    record against one line each."""
    assert _door(param) == expect
    assert _emission(param) == expect
