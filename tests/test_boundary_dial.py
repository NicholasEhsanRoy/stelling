# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The boundary dial: what a sub-jaxpr boundary CARRIES, and what it must not.

`stelling.propagate.propagate` and `stelling.preconditions.check` take
`boundary="opaque"` (the default) or `"transparent"`. The subject of this
module is the strict-sign certificate — `_Propagator.strict_sign`, "every
element of this value is certainly nonzero of this sign at every point of
the assumed region" — and whether it crosses the boundaries the walk
already enters.

Hand-built IR throughout: no jax, so this module runs on the zero-dep lane
too. The four real jax decorators are exercised in
`tests/test_boundary_dial_jax.py`, which reuses
`tests/test_assume_scope_identity.py`'s `_wrappers` registry rather than
building a second one.

THE THREE CLAIMS, in the order an auditor should read them:

1. **The default did not move.** `test_the_DEFAULT_battery_is_byte_for_byte_
   the_base_tree` compares a battery of queries, artifact by artifact,
   against values MEASURED ON THE TREE THIS BRANCH IS BASED ON.
2. **A carried certificate is TRUE.** The exact-`Fraction` oracle in
   `tests/test_assume_bump_boundary_div.py` — `_exact_eval`, extended here
   to descend a wrapper body — evaluates the query at points of the assumed
   region and checks every certificate the table holds, including the ones
   that got there across a boundary.
3. **A `cond` never carries one out.** The false-VERIFIED case, driven
   against a deliberately broken build.

**WHAT THIS MODULE DOES NOT REACH.**

* It says nothing about `scan`, `while`, `pjit` or `closed_call`. The walk
  does not enter those, so there is no boundary for anything to cross and
  no behaviour here to check. A future descent into one of them is a
  boundary this module has never seen.
* The exact-`Fraction` oracle has no `cond` arm (see `_exact_eval`), so the
  certificate carried INTO a cond branch is checked at the VERDICT level
  here and not against exact rationals.
* It checks the strict-sign certificate and nothing else. The other tables
  swapped at a descent — `exact`, the maybe-NaN flags, the product-taints —
  are outside the dial's reach by construction, and no assertion here would
  notice if that stopped being true. `test_the_dial_does_not_touch_the_
  OTHER_tables` is the one partial exception and it is a spot check on
  `exact`, not a census.
* The generated search at the end draws from a small grammar over one
  declared array. Its floors are tripwires for a search that collapsed,
  never claims of thoroughness.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re
from fractions import Fraction

import pytest

from stelling import ir
from stelling.coverage import DEFAULT_TRANSPARENT
from stelling.propagate import propagate

from test_assume_bump_boundary_div import (
    BOOL,
    F64,
    _assumed_points,
    _build_sumsq,
    _coeff_query,
    _exact_eval,
    any_eqn,
    any_eqn_shaped,
    close,
    eqn,
    lit,
    var,
)

I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")


def _ids(start=0):
    """A var-id allocator that is UNIQUE ACROSS SCOPES.

    jax does not number this way — ids are unique per jaxpr — and the
    propagator's scope swap exists precisely because they are not. The
    allocator here is deliberately global anyway, for one reason: the
    exact-`Fraction` oracle evaluates a wrapper body into the SAME env as
    its caller, so a collision would silently clobber a value the check
    then reads. `_exact_eval` asserts the disjointness rather than trusting
    it.

    So this module does NOT exercise cross-scope id collision. That is
    `tests/test_assume_scope_identity.py`'s subject and it is left there.
    """
    box = [start - 1]

    def nxt(aval=F64):
        box[0] += 1
        return var(box[0], aval)

    return nxt


def _shaped(n, dtype="float64"):
    return ir.Aval(kind="ShapedArray", shape=(n,), dtype=dtype)


def _boolshaped(n):
    return ir.Aval(kind="ShapedArray", shape=(n,), dtype="bool")


# ---------------------------------------------------------------------------
# the queries
# ---------------------------------------------------------------------------


def _sumsq_body(nxt, n, *, assume_inside):
    """`lambda v: sum(v*v)`, optionally stating `assume(v > 0)` first."""
    xa, ba = _shaped(n), _boolshaped(n)
    v = nxt(xa)
    body_eqns = []
    if assume_inside:
        p_, o_ = nxt(ba), nxt(ba)
        body_eqns += [
            eqn("gt", [v, lit(0.0)], p_),
            eqn("stelling_assume", [p_], o_),
        ]
    sq, s = nxt(xa), nxt(F64)
    body_eqns += [
        eqn("mul", [v, v], sq),
        eqn("reduce_sum", [sq], s, [("axes", (0,))]),
    ]
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(v,), outvars=(s,), eqns=tuple(body_eqns)
        )
    )


def wrapped_sumsq_query(prim, *, assume_inside, n=4, nest=1):
    """`assume(x>0)`; `assert_(1 / W(...W(sum(v*v))...)(x) > 0)`.

    `prim` is any member of `coverage.DEFAULT_TRANSPARENT` — the accessor
    the propagator uses (`coverage.call_body`) finds a body by TYPE and not
    by param name, so one spelling covers all four. `nest` wraps the body
    that many times; `assume_inside` moves the `assume` into the innermost
    body.

    Returns `(closed, outer_eqns)`; the second is what the exact-`Fraction`
    oracle walks.
    """
    nxt = _ids()
    xa, ba = _shaped(n), _boolshaped(n)
    x = nxt(xa)
    eqns = [any_eqn_shaped(x, 0.0, 2.0, (n,))]
    if not assume_inside:
        pa, ao = nxt(ba), nxt(ba)
        eqns += [
            eqn("gt", [x, lit(0.0)], pa),
            eqn("stelling_assume", [pa], ao),
        ]
    body = _sumsq_body(nxt, n, assume_inside=assume_inside)
    for _ in range(nest - 1):
        # a wrapper whose body is a wrapper: invar in, one call, outvar out
        iv_ = nxt(xa)
        io_ = nxt(F64)
        body = ir.ClosedJaxpr(
            jaxpr=ir.Jaxpr(
                constvars=(),
                invars=(iv_,),
                outvars=(io_,),
                eqns=(eqn(prim, [iv_], io_, [("jaxpr", body)]),),
            )
        )
    s, q, pred, out = nxt(F64), nxt(F64), nxt(BOOL), nxt(BOOL)
    eqns += [
        eqn(prim, [x], s, [("jaxpr", body)]),
        eqn("div", [lit(1.0), s], q),
        eqn("gt", [q, lit(0.0)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out]), eqns


def _assume_body(nxt):
    """`lambda v: (assume(v > 0), v)[1]` — a scalar identity that STATES a
    strict precondition on its own input and returns it."""
    v, p_, o_ = nxt(F64), nxt(BOOL), nxt(BOOL)
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(v,),
            outvars=(v,),
            eqns=(
                eqn("gt", [v, lit(0.0)], p_),
                eqn("stelling_assume", [p_], o_),
            ),
        )
    )


def certifying_body_query(construct):
    """THE COND-OUT PROHIBITION'S QUERY, and its unconditional control.

    ``x ∈ [0, 1]``; a body that states ``assume(v > 0)`` and returns ``v``;
    then ``assert_(1 / <that body's result> > 0)`` OUTSIDE the body.

    * ``construct="jit"`` — the body is an UNCONDITIONAL wrapper. It runs
      whenever the equation runs, so the precondition it states is a
      precondition of the whole query and its certificate may come out.
      Under ``boundary="transparent"`` this is VERIFIED.
    * ``construct="cond"`` — the same body is one of two BRANCHES, the
      other being the identity, under a selector the analysis only ADMITS.
      The precondition is then conditional on that branch being taken, and
      a certificate carried out would license a conclusion outside the
      branch that states it. It must stay UNKNOWN.

    The two differ in the CONSTRUCT and in nothing else that matters: the
    same body, the same declared box, the same divisor, the same assert.
    That is what makes the pair a control rather than two unrelated
    queries — the `"cond"` half staying UNKNOWN is only evidence if the
    `"jit"` half is green.

    Returns `(closed, divisor_var)`.
    """
    nxt = _ids()
    x = nxt(F64)
    eqns = [any_eqn(x, 0.0, 1.0)]
    body = _assume_body(nxt)
    d = nxt(F64)
    if construct == "jit":
        eqns.append(eqn("jit", [x], d, [("jaxpr", body)]))
    elif construct == "cond":
        by = nxt(F64)
        identity = ir.ClosedJaxpr(
            jaxpr=ir.Jaxpr(
                constvars=(), invars=(by,), outvars=(by,), eqns=()
            )
        )
        w, sel = nxt(F64), nxt(I32)
        eqns += [
            any_eqn(w, 0.0, 1.0),
            eqn(
                "convert_element_type", [w], sel,
                [("new_dtype", "int32")],
            ),
            eqn(
                "cond", [sel, x], d,
                [("branches", (identity, body))],
            ),
        ]
    else:  # pragma: no cover - a typo in a parametrisation, not a path
        raise AssertionError(construct)
    q, pred, out = nxt(F64), nxt(BOOL), nxt(BOOL)
    eqns += [
        eqn("div", [lit(1.0), d], q),
        eqn("gt", [q, lit(0.0)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out]), d


def disagreeing_premises_query():
    """**THE QUERY THAT KILLS THE PLAUSIBLE REPAIR.**

    ``x \u2208 [-1, 1]``, an unforced selector, and two branches that each
    certify their own output but on PREMISES THAT CONTRADICT EACH OTHER:

        branch 0:  assume(v > 0);  return v      \u2192 certified +1, box [0, 1]
        branch 1:  assume(v < 0);  return -v     \u2192 certified +1, box [0, 1]

    The join is ``[0, 1]`` \u2014 one-sided at zero, which is exactly the shape
    the `div` boundary gate is about \u2014 and BOTH branches agree the output
    is positive. So the repair that looks obviously sound, *"carry out
    only what every possible branch agrees on"*, fires here and certifies
    the cond's output.

    It is wrong, and the counterexample is inside the declared box:
    ``x = 1/2`` with the selector taking branch 1 gives ``-1/2``, and
    ``1/(-1/2) = -2``, which is not ``> 0``. Nothing excludes that point.
    The two ``assume``s are BRANCH-LOCAL, so neither is a precondition of
    the query; their conjunction is empty, and a reader of the stamp sees
    two `constrained assume` lines that cannot both hold.

    Returns ``(closed, cond_outvar)``.
    """
    nxt = _ids()
    x = nxt(F64)
    w, sel = nxt(F64), nxt(I32)
    b0, p0, o0 = nxt(F64), nxt(BOOL), nxt(BOOL)
    branch_positive = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(b0,), outvars=(b0,),
            eqns=(
                eqn("gt", [b0, lit(0.0)], p0),
                eqn("stelling_assume", [p0], o0),
            ),
        )
    )
    b1, p1, o1, n1 = nxt(F64), nxt(BOOL), nxt(BOOL), nxt(F64)
    branch_negated = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(b1,), outvars=(n1,),
            eqns=(
                eqn("lt", [b1, lit(0.0)], p1),
                eqn("stelling_assume", [p1], o1),
                eqn("neg", [b1], n1),
            ),
        )
    )
    cout, q, pred, out = nxt(F64), nxt(F64), nxt(BOOL), nxt(BOOL)
    eqns = [
        any_eqn(x, -1.0, 1.0),
        any_eqn(w, 0.0, 1.0),
        eqn("convert_element_type", [w], sel, [("new_dtype", "int32")]),
        eqn(
            "cond", [sel, x], cout,
            [("branches", (branch_positive, branch_negated))],
        ),
        eqn("div", [lit(1.0), cout], q),
        eqn("gt", [q, lit(0.0)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out]), cout


def _straddle_query():
    """`x ∈ [-1, 1]`, no assume, `1/x` — declines on a TRUE straddle in
    both dial positions and by a different message than the boundary one."""
    nxt = _ids()
    x = nxt(F64)
    q, pred, out = nxt(F64), nxt(BOOL), nxt(BOOL)
    return close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("div", [lit(1.0), x], q),
            eqn("gt", [q, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _no_certificate_query():
    """`x ∈ [1, 2]`, `1/x > 0` — VERIFIED with no certificate at all: the
    declared box already excludes zero, so nothing here consults the
    table. The battery needs one of these or it measures only the queries
    the dial can move."""
    nxt = _ids()
    x = nxt(F64)
    q, pred, out = nxt(F64), nxt(BOOL), nxt(BOOL)
    return close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("div", [lit(1.0), x], q),
            eqn("gt", [q, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _refuted_query():
    """`x ∈ [1, 2]`, `assert_(x < 0)` — violated at every declared point.
    In the battery so that the REFUTED face is measured too."""
    nxt = _ids()
    x = nxt(F64)
    pred, out = nxt(BOOL), nxt(BOOL)
    return close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("lt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _cond_inside_jit_query():
    """A `cond` INSIDE a `jit`: the scope path is EXTENDED at both steps,
    which is where an off-by-one in a save/restore shows up.

    The `assume` is outside everything and the DIVISION is inside a branch,
    so the certificate has to cross a wrapper boundary and then a branch
    boundary to reach it. The divisor is the BRANCH'S OWN INVAR and never
    a cond output: a cond hands its outputs no certificate at all, by
    design, so a query dividing by one would be UNKNOWN in both positions
    and would measure the prohibition rather than the nesting.
    """
    nxt = _ids()
    x = nxt(F64)
    pa, ao = nxt(BOOL), nxt(BOOL)
    eqns = [
        any_eqn(x, 0.0, 2.0),
        eqn("gt", [x, lit(0.0)], pa),
        eqn("stelling_assume", [pa], ao),
    ]
    branches = []
    for _ in range(2):
        bi, bq, bp, bo = nxt(F64), nxt(F64), nxt(BOOL), nxt(BOOL)
        branches.append(
            ir.ClosedJaxpr(
                jaxpr=ir.Jaxpr(
                    constvars=(),
                    invars=(bi,),
                    outvars=(bq,),
                    eqns=(
                        eqn("div", [lit(1.0), bi], bq),
                        eqn("gt", [bq, lit(0.0)], bp),
                        eqn("stelling_assert", [bp], bo),
                    ),
                )
            )
        )
    jv, jw, jsel, jc = nxt(F64), nxt(F64), nxt(I32), nxt(F64)
    body = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(jv,),
            outvars=(jc,),
            eqns=(
                any_eqn(jw, 0.0, 1.0),
                eqn(
                    "convert_element_type", [jw], jsel,
                    [("new_dtype", "int32")],
                ),
                eqn("cond", [jsel, jv], jc, [("branches", tuple(branches))]),
            ),
        )
    )
    s, pred, out = nxt(F64), nxt(BOOL), nxt(BOOL)
    eqns += [
        eqn("jit", [x], s, [("jaxpr", body)]),
        eqn("gt", [s, lit(-1.0)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out])


def _jit_inside_cond_query():
    """A `jit` INSIDE a `cond` branch. The assume is outside both; the
    division is inside the jit, inside the branch. Two boundaries again,
    tagged differently (`("cond", pos, i)` then `("call", pos)`)."""
    nxt = _ids()
    x = nxt(F64)
    pa, ao = nxt(BOOL), nxt(BOOL)
    w, sel = nxt(F64), nxt(I32)
    # the jit body: 1/v, asserted positive
    jv, jq, jp, jo = nxt(F64), nxt(F64), nxt(BOOL), nxt(BOOL)
    jit_body = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(jv,),
            outvars=(jq,),
            eqns=(
                eqn("div", [lit(1.0), jv], jq),
                eqn("gt", [jq, lit(0.0)], jp),
                eqn("stelling_assert", [jp], jo),
            ),
        )
    )
    branches = []
    for _ in range(2):
        bi, bo = nxt(F64), nxt(F64)
        branches.append(
            ir.ClosedJaxpr(
                jaxpr=ir.Jaxpr(
                    constvars=(), invars=(bi,), outvars=(bo,),
                    eqns=(eqn("jit", [bi], bo, [("jaxpr", jit_body)]),),
                )
            )
        )
    cout, pred, out = nxt(F64), nxt(BOOL), nxt(BOOL)
    eqns = [
        any_eqn(x, 0.0, 2.0),
        eqn("gt", [x, lit(0.0)], pa),
        eqn("stelling_assume", [pa], ao),
        any_eqn(w, 0.0, 1.0),
        eqn("convert_element_type", [w], sel, [("new_dtype", "int32")]),
        eqn("cond", [sel, x], cout, [("branches", tuple(branches))]),
        eqn("gt", [cout, lit(-1.0)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out])


# ---------------------------------------------------------------------------
# 1. THE DEFAULT DID NOT MOVE
# ---------------------------------------------------------------------------


#: The battery. Every entry is `(name, closed, semantics)`; every one is
#: hand-built IR, so nothing here reads the host's jax, its x64 flag or any
#: other fact about the environment the check runs in — the artifacts below
#: are the same on all three lanes and that is checked by running them
#: there, not by asserting it.
def battery():
    return (
        ("sumsq-no-wrapper", _coeff_query(_build_sumsq, n=4)[0], "real"),
        ("jit-assume-outside",
         wrapped_sumsq_query("jit", assume_inside=False)[0], "real"),
        ("jit-assume-inside",
         wrapped_sumsq_query("jit", assume_inside=True)[0], "real"),
        ("jit-in-jit",
         wrapped_sumsq_query("jit", assume_inside=False, nest=2)[0], "real"),
        ("remat2-assume-outside",
         wrapped_sumsq_query("remat2", assume_inside=False)[0], "real"),
        ("cond-in-jit", _cond_inside_jit_query(), "real"),
        ("jit-in-cond", _jit_inside_cond_query(), "real"),
        ("cond-certifying-branch", certifying_body_query("cond")[0], "real"),
        ("jit-certifying-body", certifying_body_query("jit")[0], "real"),
        ("straddle", _straddle_query(), "real"),
        ("no-certificate", _no_certificate_query(), "real"),
        ("refuted", _refuted_query(), "real"),
        ("ieee-jit-assume-outside",
         wrapped_sumsq_query("jit", assume_inside=False)[0], "ieee"),
        ("ieee-straddle", _straddle_query(), "ieee"),
    )


#: Every `Propagation` field this comparison IGNORES, and why each is out.
#: Everything else is compared, derived through `dataclasses.fields` so a
#: field added later joins the comparison with no edit here — the failure
#: mode of a hand-kept list is that the next field is silently outside the
#: guard.
_BASELINE_EXCLUDED = {
    # the two fields this change ADDS. They did not exist on the base tree,
    # so comparing them against it is not possible and not the question:
    # the question is whether anything ELSE moved.
    "boundary",
    "boundary_crossings",
}


#: `AssumeDisposition.eqn_id` is `id()` of an object in THIS process's
#: memory. Its own field declares `compare=False` for exactly this reason —
#: *"a field that cannot be compared across runs must not be what a
#: cross-run comparison fails on"* — but `repr` still prints it, and this
#: comparison is against a value produced in a DIFFERENT process. Measured:
#: two runs of `baseline_row` on one query in one tree differ only in this
#: number. Normalised rather than dropped, so a ledger entry appearing or
#: disappearing still moves the row.
_NOT_COMPARABLE = re.compile(r"eqn_id=\d+")


def baseline_row(closed, semantics):
    """Every artifact of one DEFAULT-position propagation, as text.

    Called with no `boundary=` keyword ON PURPOSE: that is what makes this
    function runnable against the tree this branch is based on, which is
    where the recorded values below came from. A function that passed
    `boundary="opaque"` explicitly would be untestable against a tree with
    no such keyword, and the comparison would then be against this branch's
    own idea of the default.
    """
    p = propagate(closed, semantics=semantics)
    parts = []
    for f in dataclasses.fields(p):
        if f.name in _BASELINE_EXCLUDED:
            continue
        parts.append(_NOT_COMPARABLE.sub(
            "eqn_id=<process-local>", f"{f.name}={getattr(p, f.name)!r}"
        ))
    return "\n".join(parts)


def baseline_rows():
    """`name -> (statuses, coverage summary, query hash, sha256(row))`."""
    out = {}
    for name, closed, semantics in battery():
        row = baseline_row(closed, semantics)
        p = propagate(closed, semantics=semantics)
        out[name] = (
            tuple(o.status for o in p.obligations),
            p.coverage.summary(),
            p.query_sha256,
            hashlib.sha256(row.encode()).hexdigest(),
        )
    return out


#: **THE BASELINE IS THE TREE WITHOUT THE DIAL, AND THE COMMIT IT NAMES
#: MOVED AT A MERGE.** It was produced at `8dae8cb` — the commit
#: `wip/0.3.0-p1-boundary` starts from, and the last commit before the
#: boundary dial existed. It is produced at **`46c077e`** now — and it has
#: moved TWICE, which is worth more than either move: first to `e698abb`
#: when the strict-sign census landed beside the dial, then to `46c077e`
#: when that census's own audit changed a decline message (a count of
#: side-conditioned carriers that said two where the probe table says
#: thirteen). That message is an obligation DETAIL, so it is a field this
#: battery compares. **Neither move was a defect in the dial, and both were
#: EXPLAINED before they were re-pinned** — a fixture red twice in one
#: integration and silenced twice would be worth nothing; one red twice and
#: explained twice is live. The
#: reason is the whole point of this fixture.
#:
#: The strict-sign CENSUS landed on a sibling branch and ships **default-on**:
#: it is about WHICH primitives keep the certificate, not about where it
#: travels, so it is not behind this dial. The carrying set went from ten
#: primitives to thirty-one, which moves the DEFAULT — legitimately, and for
#: a reason that has nothing to do with `boundary`. Against `8dae8cb` this
#: battery went red on the merge, correctly: *"a red here is not
#: automatically a defect in the boundary dial … it is a red that has to be
#: EXPLAINED"*, and this paragraph is the explanation.
#:
#: So the baseline is re-taken at `46c077e` — **the census WITHOUT the dial**
#: — and the question the fixture asks is unchanged and now exact: *does
#: adding the boundary dial move any default answer?* Re-pinning it to the
#: merge itself would have made it tautological; re-pinning it to the census
#: keeps it a statement about this change.
#:
#: Produced on 2026-08-28 by running THIS MODULE's own `baseline_rows()`
#: against that tree's `src/`:
#:
#:     git archive 46c077e src | tar -x -C $T
#:     env PYTHONPATH=$T/src:$PWD/tests python -c \
#:         "import test_boundary_dial as B, pprint; pprint.pprint(B.baseline_rows())"
#:
#: That command imports this file against a `stelling` that has no
#: `boundary` keyword at all, which is why `baseline_row` takes none: the
#: recorded values are the OLD tree's answers, not this branch's answers to
#: a question phrased in this branch's vocabulary.
#:
#: **WHAT THIS CAN AND CANNOT SEE.** It compares every `Propagation` field
#: except the two this change adds — obligation statuses AND details,
#: nonvacuity checks, coverage, transfers used, stamped assumptions, notes,
#: query hash, the three assume-state flags, `top_boxes`, the relational
#: assumes and the whole assume ledger — for fourteen queries covering the
#: wrapper, the cond, the nested and the ieee shapes, plus one REFUTED and
#: one query with no certificate anywhere. It does NOT see: any query shape
#: absent from that battery; anything downstream of `propagate` (the verdict
#: assembly, the solver escalation, the affine refinement); and it cannot
#: distinguish "the default did not move" from "the default moved and the
#: base tree moved with it", which is why the values are pinned to a NAMED
#: COMMIT rather than recomputed.
#:
#: A red here is not automatically a defect in the boundary dial — any
#: change that legitimately moves a decline message or a note will redden
#: it. It is a red that has to be EXPLAINED, which is the whole of what a
#: regression baseline buys.
BASELINE_WITHOUT_THE_DIAL = {'cond-certifying-branch': (('unknown',),
                            '9 eqns: 7 known (78%); 1 ⊤ across 1 primitives '
                            '(div ×1); 1 assume(s) CONSTRAINED '
                            '(stelling_assume ×1)',
                            '925bf64fda32df7d7175f3c3744983291f7924169c3099c69b287ad5e460bd6f',
                            '8c535cbfc711f064291c135d7e831b2afd11497c04e80fe1ffb164e02ca24cf3'),
 'cond-in-jit': (('unknown', 'unknown', 'unknown'),
                 '15 eqns: 11 known (73%); 1 transparent; 2 ⊤ across 1 '
                 'primitives (div ×2); 1 assume(s) CONSTRAINED '
                 '(stelling_assume ×1)',
                 '58bbbb06db544350fcd17773e7b4f7763b7b19d2ba1fbe2307d51cf196c43711',
                 '2c5333bda33248f561546e4f6a43781d43f2da5855fa46eea9c9359fbb484c32'),
 'ieee-jit-assume-outside': (('unknown',),
                             '9 eqns: 5 known (56%); 1 transparent; 2 ⊤ '
                             'across 2 primitives (div ×1, reduce_sum ×1); 1 '
                             'assume(s) CONSTRAINED (stelling_assume ×1)',
                             '90d760d2a936e53e8339aa6f630729f485c90a90592e3e2b74bbd6d27c02533f',
                             '40b6d8b0b137e3355a969729dd44a2f63c8c87a0d70e1e403cedbb750485186e'),
 'ieee-straddle': (('unknown',),
                   '4 eqns: 3 known (75%); 1 ⊤ across 1 primitives (div ×1)',
                   '45505bedf73c29b7edd3b1cabf97e8cda9baeeb4a857b5d1223b960a83724aa0',
                   '6a937f8da0379d3e834f2c598297f2674994d07c8c23a9412332c2dbe808c113'),
 'jit-assume-inside': (('unknown',),
                       '9 eqns: 6 known (67%); 1 transparent; 1 ⊤ across 1 '
                       'primitives (div ×1); 1 assume(s) CONSTRAINED '
                       '(stelling_assume ×1)',
                       '69b670aa6724004bdb9365e39d8fb574545d7c1fb703a1373b4f089e21022df8',
                       '7258fa0bbcc71ee5589fd190d5801a73a36a291486baac6045739ab79c428d23'),
 'jit-assume-outside': (('unknown',),
                        '9 eqns: 6 known (67%); 1 transparent; 1 ⊤ across 1 '
                        'primitives (div ×1); 1 assume(s) CONSTRAINED '
                        '(stelling_assume ×1)',
                        '90d760d2a936e53e8339aa6f630729f485c90a90592e3e2b74bbd6d27c02533f',
                        '91290621a0296f52c622171a13b0fa6747a7c3cc907ee371e4eacc770f1f1ca4'),
 'jit-certifying-body': (('unknown',),
                         '7 eqns: 4 known (57%); 1 transparent; 1 ⊤ across 1 '
                         'primitives (div ×1); 1 assume(s) CONSTRAINED '
                         '(stelling_assume ×1)',
                         '194ff15a11f91372146b28e877975ae08e7d00fc4f31f464c65b042a3e17eed5',
                         '0432a907a99763870c87bbb7d636af20a992c095f59e493ed51819b1c999bcef'),
 'jit-in-cond': (('unknown', 'unknown', 'unknown'),
                 '16 eqns: 11 known (69%); 2 transparent; 2 ⊤ across 1 '
                 'primitives (div ×2); 1 assume(s) CONSTRAINED '
                 '(stelling_assume ×1)',
                 '0dff25bef3a7dfad48a4a5820b3b922901bfa45e26e39710655564ac0b7a07c8',
                 '65b14aa9e821ee2e067d22169a7add6a31c0b8d21eeb58436165b1e6175300d9'),
 'jit-in-jit': (('unknown',),
                '10 eqns: 6 known (60%); 2 transparent; 1 ⊤ across 1 '
                'primitives (div ×1); 1 assume(s) CONSTRAINED '
                '(stelling_assume ×1)',
                'd916a9b89913abbf12e4c4948a3487dca91160a89076b4b99ee5d788a960ed99',
                '9d6e772758ac09f47567bff5df64d3976b7070f1d5508e61ea8e1a9960e6378f'),
 'no-certificate': (('discharged',),
                    '4 eqns: 4 known (100%)',
                    'd87a7c672ee8c8036d8cda218fc4c0bc0e7f4d24ba95bf75a94936304e831530',
                    'f0fa01f7913a9dea0203048ce9142b94e1223dd762ab51dbe63ee808ad5fef97'),
 'refuted': (('violated-over-set',),
             '3 eqns: 3 known (100%)',
             '0800c66b228b3201bff2a29c56f86152d4a62024598335c91659b5be55ec515c',
             '3a7213d3e099c3a51af423ed4d9d38b9bfe0f424eda067e588dab822f18b4726'),
 'remat2-assume-outside': (('unknown',),
                           '9 eqns: 6 known (67%); 1 transparent; 1 ⊤ across '
                           '1 primitives (div ×1); 1 assume(s) CONSTRAINED '
                           '(stelling_assume ×1)',
                           '4320f343fe28825c555148f1dd984c6cb8a0dcc9ba7bcb320f2204be8403cde6',
                           '4acc410ef789cb7a8cb386effe87ac644d655ac9c845901343f385eef1f2055e'),
 'straddle': (('unknown',),
              '4 eqns: 3 known (75%); 1 ⊤ across 1 primitives (div ×1)',
              '45505bedf73c29b7edd3b1cabf97e8cda9baeeb4a857b5d1223b960a83724aa0',
              '892b9d50d253d5b63b71b4f043645b1e8c3a7eaa5c3fee1d7523d0bbdac7d743'),
 'sumsq-no-wrapper': (('discharged',),
                      '8 eqns: 7 known (88%); 1 assume(s) CONSTRAINED '
                      '(stelling_assume ×1)',
                      '3d5af97ec0f2c66db453a4c142adbfcbaba0ac5aa69e45a5a2e6edf08cb2e7c1',
                      'b5c790c779cfb976204eae0e894b6e503518f7b8a70eb2a7e28ded01e924a41f')}


def test_the_DEFAULT_battery_is_byte_for_byte_the_base_tree():
    """The acceptance criterion that outranks every other one, measured.

    `boundary="opaque"` is the default and must be the base tree's
    behaviour with nothing added. This runs the battery THROUGH THE
    DEFAULT — no keyword — and compares every artifact against the values
    the base tree produced. See `BASELINE_WITHOUT_THE_DIAL` for what the
    comparison reaches and what it does not.
    """
    live = baseline_rows()
    assert set(live) == set(BASELINE_WITHOUT_THE_DIAL), (
        f"the battery and the baseline name different queries: "
        f"only live {sorted(set(live) - set(BASELINE_WITHOUT_THE_DIAL))}, "
        f"only baseline {sorted(set(BASELINE_WITHOUT_THE_DIAL) - set(live))}"
    )
    bad = []
    for name in sorted(live):
        if live[name] != BASELINE_WITHOUT_THE_DIAL[name]:
            bad.append(
                f"{name}:\n    base {BASELINE_WITHOUT_THE_DIAL[name]}\n"
                f"    live {live[name]}\n"
                f"    live row:\n"
                + "\n".join(
                    "      " + ln
                    for ln in baseline_row(
                        *[
                            (c, s)
                            for n, c, s in battery()
                            if n == name
                        ][0]
                    ).splitlines()
                )
            )
    assert not bad, (
        "THE DEFAULT MOVED. `boundary=\"opaque\"` must be the base tree's "
        "behaviour with nothing added to it:\n" + "\n".join(bad)
    )


def test_the_battery_is_not_a_battery_of_queries_the_dial_cannot_move():
    """THE ABSENCE HALF. A baseline over queries the dial cannot touch
    would be green whatever the carry did, so the same battery is run in
    the OTHER position and required to differ — and to differ on the
    shapes the dial is about, not merely somewhere."""
    moved = set()
    for name, closed, semantics in battery():
        a = propagate(closed, semantics=semantics, boundary="opaque")
        b = propagate(closed, semantics=semantics, boundary="transparent")
        if [o.status for o in a.obligations] != [
            o.status for o in b.obligations
        ]:
            moved.add(name)
    assert {"jit-assume-outside", "jit-assume-inside", "jit-in-jit"} <= moved, (
        f"the battery's wrapper queries do not move under the dial, so the "
        f"baseline above proves nothing about the carry; moved: "
        f"{sorted(moved)}"
    )
    assert "cond-certifying-branch" not in moved, (
        "the cond-out prohibition is not holding on the battery"
    )


# ---------------------------------------------------------------------------
# 2. the dial itself
# ---------------------------------------------------------------------------


def test_the_dial_refuses_an_unknown_value_by_name():
    """The house idiom for a dial: a named refusal quoting what was
    passed, raised at the door and not three layers down."""
    from stelling.propagate import _BOUNDARY_MODES

    q = _no_certificate_query()
    with pytest.raises(ValueError) as e:
        propagate(q, boundary="transparant")
    assert "'transparant'" in str(e.value), str(e.value)
    assert str(_BOUNDARY_MODES) in str(e.value), str(e.value)
    assert _BOUNDARY_MODES == ("opaque", "transparent"), _BOUNDARY_MODES


def test_the_position_is_recorded_on_every_propagation():
    q = _no_certificate_query()
    assert propagate(q).boundary == "opaque"
    assert propagate(q, boundary="opaque").boundary == "opaque"
    assert propagate(q, boundary="transparent").boundary == "transparent"


def test_the_POSITION_is_stamped_and_the_CROSSING_is_conditioned():
    """The asymmetry the stamp has to keep.

    The dial's POSITION is a fact about the rule the run was judged
    under and is stamped whenever it is off the position every recorded
    verdict in this project's history was taken in. A CROSSING is an ACT,
    and its disclosure appears only on the runs where the act happened —
    a line reading "certificates crossed" on a run in which none did
    would be a measurement nobody made.

    The `"opaque"` position adds NO line, and that is the byte-for-byte
    acceptance criterion rather than an oversight; `propagate`'s own
    comment at the stamping site carries the argument and the cost.
    """
    from stelling.propagate import (
        BOUNDARY_CROSSED_DISCLOSURE, BOUNDARY_TRANSPARENT_POSITION,
    )

    nothing_to_carry = _no_certificate_query()
    carries, _ = wrapped_sumsq_query("jit", assume_inside=False)

    for q in (nothing_to_carry, carries):
        opaque = propagate(q, boundary="opaque")
        assert not any(
            "boundary=" in a for a in opaque.assumptions
        ), opaque.assumptions
        assert opaque.boundary_crossings == 0

    quiet = propagate(nothing_to_carry, boundary="transparent")
    assert BOUNDARY_TRANSPARENT_POSITION in quiet.assumptions, quiet.assumptions
    assert quiet.boundary_crossings == 0
    assert not any(
        a.startswith("boundary='transparent' CARRIED") for a in quiet.assumptions
    ), (
        "a run that carried nothing disclosed a crossing: the disclosure is "
        "riding on the position instead of on the act"
    )

    loud = propagate(carries, boundary="transparent")
    assert BOUNDARY_TRANSPARENT_POSITION in loud.assumptions
    assert loud.boundary_crossings > 0
    assert (
        BOUNDARY_CROSSED_DISCLOSURE.format(n=loud.boundary_crossings)
        in loud.assumptions
    ), loud.assumptions


def test_the_stamped_lines_reach_the_verdict():
    """The propagation's assumption lines are what the stamp is built
    from, on both assembly paths. Checked through the interval assembler
    rather than asserted, because a line that stops at the `Propagation`
    is a line no reader of a verdict ever sees."""
    from stelling.propagate import BOUNDARY_TRANSPARENT_POSITION
    from stelling.verdict import make_verdict

    q, _ = wrapped_sumsq_query("jit", assume_inside=False)
    versions = dict(
        stelling_version="test", jax_version="none",
        precision_config="jax_enable_x64=True",
    )
    p = propagate(q, boundary="transparent")
    v = make_verdict(q, p, **versions)
    assert BOUNDARY_TRANSPARENT_POSITION in v.stamp.assumptions
    assert any(
        a.startswith("boundary='transparent' CARRIED")
        for a in v.stamp.assumptions
    ), v.stamp.assumptions
    assert BOUNDARY_TRANSPARENT_POSITION in v.stamp.render()

    o = make_verdict(q, propagate(q), **versions)
    assert not any("boundary=" in a for a in o.stamp.assumptions), (
        "the DEFAULT grew a stamp line; every verdict in the archive was "
        "taken without one"
    )


# ---------------------------------------------------------------------------
# 3. the wrapper carry, both directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prim", sorted(DEFAULT_TRANSPARENT))
@pytest.mark.parametrize("assume_inside", [False, True])
def test_every_transparent_wrapper_carries_the_certificate_both_ways(
    prim, assume_inside
):
    """All four members, both `assume` placements, against the ORACLE.

    The oracle is the same arithmetic with no wrapper at all
    (`tests/test_assume_bump_boundary_div.py::test_the_uncrossed_jit_query_is_green_without_the_wrapper`'s
    query): the wrapped and unwrapped programs compute the same real
    number, so under a boundary that carries the certificate they must
    reach the same verdict. Asserting `discharged` alone would pass for
    any reason at all; asserting the agreement is the claim.

    Parametrised over `coverage.DEFAULT_TRANSPARENT` itself, so a wrapper
    added to the registry cannot acquire carrying behaviour with no test
    of it. These are HAND-BUILT bodies; the four real jax decorators are
    driven in `tests/test_boundary_dial_jax.py`.
    """
    unwrapped, _ = _coeff_query(_build_sumsq, n=4)
    control = propagate(unwrapped, semantics="real", boundary="transparent")
    assert control.obligations[0].status == "discharged", (
        f"the unwrapped control is not green, so this test would be "
        f"comparing two UNKNOWNs: {control.obligations[0].detail}"
    )
    q, _ = wrapped_sumsq_query(prim, assume_inside=assume_inside)
    opaque = propagate(q, semantics="real")
    assert opaque.obligations[0].status != "discharged", (
        f"{prim}: the DEFAULT discharged a wrapped query — the default moved"
    )
    transparent = propagate(q, semantics="real", boundary="transparent")
    assert (
        transparent.obligations[0].status
        == control.obligations[0].status
        == "discharged"
    ), (
        f"{prim} (assume_inside={assume_inside}): wrapped is "
        f"{transparent.obligations[0].status!r}, unwrapped control is "
        f"{control.obligations[0].status!r} — "
        f"{transparent.obligations[0].detail}"
    )


@pytest.mark.parametrize(
    "name,query",
    [
        ("jit-in-jit", wrapped_sumsq_query("jit", assume_inside=False, nest=2)[0]),
        ("jit-in-jit-assume-inside",
         wrapped_sumsq_query("jit", assume_inside=True, nest=2)[0]),
        ("cond-in-jit", _cond_inside_jit_query()),
        ("jit-in-cond", _jit_inside_cond_query()),
    ],
)
def test_the_carry_survives_nesting(name, query):
    """The scope path is EXTENDED, never reset, so a save/restore that is
    off by one frame shows up here and nowhere shallower. Each of these is
    UNKNOWN under the default and discharged under the dial."""
    assert not propagate(query, semantics="real").all_discharged, (
        f"{name}: the DEFAULT discharged it — the default moved"
    )
    p = propagate(query, semantics="real", boundary="transparent")
    assert p.all_discharged, (
        f"{name}: {[o.status for o in p.obligations]} — "
        f"{[o.detail for o in p.obligations]}"
    )
    assert p.boundary_crossings > 0


def test_the_outer_table_is_restored_after_a_descent():
    """A carry OUT writes into the CALLER's table, and the caller's table
    is the one that was saved. A restore that handed back the callee's
    table would show as the outer scope holding an inner id."""
    from stelling.propagate import _Propagator

    q, _ = wrapped_sumsq_query("jit", assume_inside=False)
    p = _Propagator("constrain", "real", None, "transparent")
    p.run(q.jaxpr, list(q.consts), [])
    top_ids = {
        out.id for e in q.jaxpr.eqns for out in e.outvars
    } | {
        e.outvars[0].id for e in q.jaxpr.eqns
    }
    stray = set(p.strict_sign) - top_ids
    assert not stray, (
        f"the finished top-level table holds ids no top-level equation "
        f"produced: {sorted(stray)} — a sub-jaxpr's table survived its "
        f"scope"
    )


def _refutable_wrapped_query(bound):
    """`assume(x>0)` over `x ∈ [0,2]^4`; `assert_(1 / jit(sum(v*v))(x) < bound)`.

    With the certificate the divisor's box `[0, 16]` becomes usable and the
    quotient boxes to `[1/16, +inf]`, so `bound = 0.05` is DEFINITELY FALSE
    — and truly so: `Σxᵢ² ≤ 16` at every point of the declared box, hence
    `1/Σxᵢ² ≥ 0.0625 > 0.05` everywhere. Without the certificate the
    quotient is ⊤ and the obligation is undecided.
    """
    nxt = _ids()
    n = 4
    xa, ba = _shaped(n), _boolshaped(n)
    x = nxt(xa)
    pa, ao = nxt(ba), nxt(ba)
    eqns = [
        any_eqn_shaped(x, 0.0, 2.0, (n,)),
        eqn("gt", [x, lit(0.0)], pa),
        eqn("stelling_assume", [pa], ao),
    ]
    body = _sumsq_body(nxt, n, assume_inside=False)
    s, q, pred, out = nxt(F64), nxt(F64), nxt(BOOL), nxt(BOOL)
    eqns += [
        eqn("jit", [x], s, [("jaxpr", body)]),
        eqn("div", [lit(1.0), s], q),
        eqn("lt", [q, lit(bound)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out])


def test_the_dial_can_also_move_UNKNOWN_to_REFUTED():
    """**A DIRECTION THE SPEC FOR THIS CHANGE SAID DID NOT EXIST, MEASURED
    HERE BECAUSE IT DOES.**

    The brief said of the strict-sign certificate: *"the certificate's
    effect is UNKNOWN → VERIFIED and never UNKNOWN → REFUTED: dropping it
    can only widen."* The second clause is true — dropping the certificate
    replaces a bounded quotient with ⊤, which can only lose conclusions.
    The first is not. `stelling.interval.boundary_div` returns a
    HALF-INFINITE box (`[a_lo/b_hi, +inf]` for a positive dividend over
    `[0, b_hi]`), and a half-infinite box can make an upper-bound
    obligation DEFINITELY FALSE where ⊤ made it undecided.

    Driven at `8dae8cb`, with no wrapper and no dial involved —
    `assume(x > 0)` over `x ∈ [0, 2]`, `assert_(1/x < 0.4)`:

        without the assume   unknown
        with the assume      violated-over-set

    So this is a property of the CERTIFICATE and predates the boundary
    dial. It is pinned here because the dial makes the certificate reach
    queries it did not reach before, and therefore makes this direction
    reach them too: a reader who took the brief's sentence at face value
    would expect a boundary-transparent run to differ from the default
    only in the VERIFIED direction, and it does not.

    IT IS NOT A SOUNDNESS EVENT AND HERE IS WHY. A REFUTED rests on two
    things this change does not touch: that the certificate is TRUE (the
    exact-`Fraction` checks in this module and its neighbour), and that
    the narrowed region was certified inhabited (`region_inhabited`,
    `narrowing_uncertified`, `assume_dropped` and the branch-reachability
    withholding, all of which run identically in both positions and are
    asserted equal here). The refutation below is also true of the
    program: `Σxᵢ² ≤ 16` on the declared box, so `1/Σxᵢ² ≥ 0.0625` at
    every point and `< 0.05` is false at all of them.
    """
    q = _refutable_wrapped_query(0.05)
    opaque = propagate(q, semantics="real", boundary="opaque")
    transparent = propagate(q, semantics="real", boundary="transparent")
    assert opaque.obligations[0].status == "unknown", opaque.obligations[0]
    assert transparent.obligations[0].status == "violated-over-set", (
        transparent.obligations[0]
    )
    for field in ("assume_dropped", "narrowing_uncertified", "region_inhabited"):
        assert getattr(opaque, field) == getattr(transparent, field), field


def test_the_dial_does_not_touch_the_OTHER_tables():
    """A SPOT CHECK, not a census (see this module's docstring).

    `exact` is the table this change deliberately does not carry, because
    it governs whether a definite violation may become a REFUTED. If it
    ever starts crossing, the wrapper query's exactness-derived flags are
    where it would first show.
    """
    q, _ = wrapped_sumsq_query("jit", assume_inside=False)
    a = propagate(q, semantics="real", boundary="opaque")
    b = propagate(q, semantics="real", boundary="transparent")
    for field in ("assume_dropped", "narrowing_uncertified", "region_inhabited"):
        assert getattr(a, field) == getattr(b, field), (
            f"{field} moved with the boundary dial; the dial carries the "
            f"strict-sign certificate and nothing else"
        )


# ---------------------------------------------------------------------------
# 4. THE COND-OUT PROHIBITION — the false-VERIFIED case
# ---------------------------------------------------------------------------


def test_a_branch_body_assume_NEVER_certifies_anything_outside_its_branch():
    """**THE MOST IMPORTANT TEST IN THIS CHANGE.**

    `x ∈ [0, 1]`; one branch of an admitted-either-way `cond` states
    `assume(v > 0)` and returns `v`; the division is OUTSIDE the cond, on
    the cond's own output. A certificate carried out of that branch would
    let a branch-local precondition license a conclusion about a value the
    other branch also produces — and the query is then VERIFIED while the
    program at `x = 0`, a point of the DECLARED box reached through the
    other branch, divides by zero.

    Two assertions, because the verdict is the consequence and the table
    is the mechanism: the cond's output must carry no entry in the
    strict-sign table, and the obligation must not discharge. The first
    reddens even on a carry that happens not to move this verdict.

    DRIVEN, not asserted: the propagator's `cond` arm was edited to read
    the branch outvars' signs and write them onto the cond's outvars, and
    with that edit in place this query is `discharged` — a false VERIFIED
    — and both assertions below fail. The control
    (`::test_an_UNCONDITIONAL_body_may_certify_outside_itself`) is the
    same body under a `jit`, where the carry IS sound and the query IS
    green, so the UNKNOWN here is a fact about the CONSTRUCT and not
    about a query nothing could certify.
    """
    from stelling.propagate import _Propagator

    q, divisor = certifying_body_query("cond")
    p = _Propagator("constrain", "real", None, "transparent")
    p.run(q.jaxpr, list(q.consts), [])
    assert divisor.id not in p.strict_sign, (
        f"a `cond` handed its output a strict-sign certificate "
        f"({p.strict_sign[divisor.id]!r}) minted by ONE branch's assume. "
        f"The other branch states nothing, so the claim is false at every "
        f"point the analysis admits through it."
    )
    for boundary in ("opaque", "transparent"):
        r = propagate(q, semantics="real", boundary=boundary)
        assert r.obligations[0].status != "discharged", (
            f"boundary={boundary!r}: a branch-local `assume(v > 0)` "
            f"discharged `1/cond_output > 0`. The program at x = 0 — a "
            f"point of the declared box — takes the other branch and "
            f"divides by zero."
        )


def test_the_AGREE_repair_is_unsound_too_and_here_is_the_point_it_misses():
    """**THE SECOND PROHIBITION, AND IT ANSWERS AN OPEN QUESTION RATHER
    THAN RESTATING THE FIRST.**

    The test above pins the refusal against the UNCONDITIONAL out-carry —
    "write whatever one branch minted". The independent audit of `5e525ce`
    observed, correctly, that this left the *plausible* repair unpinned:
    **"carry out only what every possible branch agrees on"** passes that
    test, and passed the whole of this suite as it then stood (measured by
    the audit: 98 passed). It also observed that the obstacle this
    project's code gives for the refusal — *"provenance is not in this
    data structure"* — is the obstacle for the first repair and not for
    the second, which needs no provenance at all.

    **THE AGREE REPAIR IS UNSOUND, AND THIS IS THE COUNTEREXAMPLE.** Two
    branches can each certify their output from premises that CONTRADICT
    each other; agreement about the SIGN is not agreement about anything
    that holds when the cond runs. `disagreeing_premises_query` is the
    witness, and the point it misses is arithmetic a reader can check:

        x = 1/2, selector takes branch 1  ->  -1/2  ->  1/(-1/2) = -2
        assert(-2 > 0)                    ->  FALSE

    and `x = 1/2` is in the declared box `[-1, 1]`. Neither `assume` is a
    precondition of the QUERY — both are branch-local, and their
    conjunction is empty — so nothing excludes that point.

    DRIVEN: with the agree repair built into the `cond` arm (every
    possible branch's outvar sign collected, written onto the cond's
    outvar when they are all equal and nonzero), this query comes back
    `discharged` — "definitely true for all 1 element(s)" — carrying two
    `constrained assume` lines that cannot both hold. The shipped code
    refuses it, and refuses it for the reason written at the `cond` arm.

    The falsifying point is COMPUTED here in exact `Fraction`s rather than
    quoted, so a reader does not have to take the arithmetic on trust and
    a later change to the query cannot leave a stale sentence behind.
    """
    from stelling.propagate import _Propagator

    # the counterexample, in exact rationals, from the branch body itself
    x_star = Fraction(1, 2)
    branch1_out = -x_star                      # `neg` of the branch invar
    quotient = Fraction(1) / branch1_out
    assert branch1_out < 0 and quotient == -2
    assert not (quotient > 0), (
        "the counterexample does not falsify the obligation; this test is "
        "about a point at which `1/cond_output > 0` is FALSE"
    )
    assert -1 <= x_star <= 1, "the point must be inside the declared box"

    q, cond_out = disagreeing_premises_query()
    p = _Propagator("constrain", "real", None, "transparent")
    p.run(q.jaxpr, list(q.consts), [])
    assert cond_out.id not in p.strict_sign, (
        f"the cond's output was certified {p.strict_sign[cond_out.id]!r} "
        f"because both branches agreed on the sign. They agreed on the "
        f"sign and disagreed on the PREMISE: one holds where v > 0 and "
        f"the other where v < 0, so their agreement licenses nothing at "
        f"x = {x_star}, where the obligation is false."
    )
    for boundary in ("opaque", "transparent"):
        r = propagate(q, semantics="real", boundary=boundary)
        assert r.obligations[0].status != "discharged", (
            f"boundary={boundary!r}: `1/cond_output > 0` discharged, and "
            f"it is false at x = {x_star} through branch 1"
        )


def test_an_UNCONDITIONAL_body_may_certify_outside_itself():
    """THE CONTROL for the test above, and the anti-vacuity half.

    The same body, the same declared box, the same divisor, the same
    assert — under a `jit` instead of a `cond` branch. A `jit` body runs
    whenever its equation runs, so the precondition it states is a
    precondition of the whole query and its certificate is a fact about
    the query's own value. Under the dial this is VERIFIED.

    Without this, the UNKNOWN next door would be evidence of nothing: a
    query that is UNKNOWN because nothing could ever certify it does not
    demonstrate a prohibition.
    """
    q, _ = certifying_body_query("jit")
    assert propagate(q, semantics="real").obligations[0].status != "discharged"
    p = propagate(q, semantics="real", boundary="transparent")
    assert p.obligations[0].status == "discharged", (
        f"the unconditional control is not green, so the cond prohibition "
        f"test is comparing an UNKNOWN against an UNKNOWN: "
        f"{p.obligations[0].detail}"
    )
    assert p.boundary_crossings > 0


def test_a_certificate_carried_INTO_a_branch_still_helps():
    """The IN direction of the cond carry, which IS sound: a fact true at
    every point of the assumed region is true at every point of the subset
    on which the index selects this branch.

    Without this the prohibition above could be satisfied by carrying
    nothing at a cond at all, which is a different (and weaker) change.
    """
    q = _jit_inside_cond_query()
    assert not propagate(q, semantics="real").all_discharged
    p = propagate(q, semantics="real", boundary="transparent")
    assert p.all_discharged, [o.detail for o in p.obligations]


# ---------------------------------------------------------------------------
# 5. ieee: the dial changes nothing, in either position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("assume_inside", [False, True])
def test_the_dial_is_INERT_under_ieee(assume_inside):
    """The certificate is REAL-MODE ONLY: under a flush-to-zero semantics
    `x > 0` does not imply "certainly nonzero" (audit 0.2.0 S10), so
    nothing may write or read the table there. The gate that enforces it
    is `_Propagator._carries_signs`, whose `_carry_refusal()` conjunct
    exists for this and is the only reason a LITERAL wrapper operand
    cannot smuggle a sign across an ieee boundary.

    **THIS TEST COMPARED FIVE THINGS AND NOT THE ONE THAT MOVED.** It
    listed statuses, details, the coverage summary, the notes and the
    crossing count — and not `assumptions`, which was the only field the
    dial changed under ieee, on every ieee run. So it asserted inertness
    while the artifact of record was not inert. It now compares EVERY
    field of the `Propagation`, through `dataclasses.fields` so a field
    added later joins with no edit here, with exactly one named and
    checked exception: the assumption line that DISCLOSES the inertness.
    """
    from stelling.propagate import (
        BOUNDARY_CROSSED_DISCLOSURE, BOUNDARY_INERT_UNDER_IEEE,
        BOUNDARY_TRANSPARENT_POSITION, _Propagator,
    )

    q, _ = wrapped_sumsq_query("jit", assume_inside=assume_inside)
    runs = {}
    for boundary in ("opaque", "transparent"):
        runs[boundary] = propagate(q, semantics="ieee", boundary=boundary)
        w = _Propagator("constrain", "ieee", None, boundary)
        w.run(q.jaxpr, list(q.consts), [])
        assert not w.strict_sign, (
            f"ieee mode with boundary={boundary!r} wrote the strict-sign "
            f"table: {w.strict_sign}"
        )
    opaque, transparent = runs["opaque"], runs["transparent"]

    # EVERY field but the three the dial is allowed to move, and the two
    # bookkeeping ones are checked by name below rather than skipped.
    moved = [
        f.name
        for f in dataclasses.fields(opaque)
        if f.name not in ("assumptions", "boundary", "boundary_crossings")
        and getattr(opaque, f.name) != getattr(transparent, f.name)
    ]
    assert not moved, (
        f"under ieee the dial moved {moved}; the certificate is a claim "
        f"about \u211d and nothing may read or write it there"
    )
    assert opaque.boundary_crossings == transparent.boundary_crossings == 0
    assert transparent.boundary == "transparent"

    # ...and the assumption lines differ by EXACTLY the inertness
    # disclosure. Set difference in both directions: a line that vanished
    # under the dial would be as wrong as one that appeared.
    added = set(transparent.assumptions) - set(opaque.assumptions)
    removed = set(opaque.assumptions) - set(transparent.assumptions)
    assert not removed, removed
    assert added == {BOUNDARY_INERT_UNDER_IEEE}, added
    assert BOUNDARY_TRANSPARENT_POSITION not in transparent.assumptions, (
        "an ieee run stamped the POSITION line, whose words are \"the "
        "certificate ... was allowed to cross\" \u2014 on a walk in which "
        "`_carries_signs` was False at every descent and nothing crossed"
    )
    assert not any(
        a.startswith(BOUNDARY_CROSSED_DISCLOSURE.split("{")[0])
        for a in transparent.assumptions
    ), transparent.assumptions


def test_the_ieee_STAMP_does_not_claim_a_rule_the_walk_refused():
    """**THE F1 DEFECT, NAMED AND DRIVEN.**

    `BOUNDARY_TRANSPARENT_POSITION` says the certificate *"was allowed to
    cross the sub-jaxpr boundaries this walk enters"*. Under
    `semantics="ieee"` it was not: `_Propagator._carries_signs` is False
    at every descent, and a certificate that HAD crossed there would be
    false on a flush-to-zero target. Stamping that sentence on an ieee run
    put a rule the analysis refused into the artifact a reader trusts.

    The repair is that the stamping site consults the walk's own gate
    (`_carry_refusal`) instead of re-reading the dial, so the three
    sentences are mutually exclusive by construction. That is what this
    test checks, on both faces: what an ieee run says, and what it does
    not say.

    REDDENS ON REVERT: restore the site to `assumptions.add(
    BOUNDARY_TRANSPARENT_POSITION)` under a bare `boundary != "opaque"`
    and the second assertion fails on every ieee query.
    """
    from stelling.propagate import (
        BOUNDARY_INERT_UNDER_IEEE, BOUNDARY_TRANSPARENT_POSITION,
    )

    for build in (
        lambda: wrapped_sumsq_query("jit", assume_inside=False)[0],
        lambda: wrapped_sumsq_query("remat2", assume_inside=True)[0],
        _straddle_query,
        _no_certificate_query,
        _cond_inside_jit_query,
    ):
        q = build()
        p = propagate(q, semantics="ieee", boundary="transparent")
        assert BOUNDARY_INERT_UNDER_IEEE in p.assumptions, p.assumptions
        assert BOUNDARY_TRANSPARENT_POSITION not in p.assumptions, (
            "an ieee verdict claims the certificate was allowed to cross"
        )
    # the sentence has to SAY the two things a reader needs: that nothing
    # crossed, and that the verdict is the boundary-opaque one. Checked as
    # content rather than as an identity, because a rename of the constant
    # must not be able to satisfy this on its own.
    assert "INERT under semantics='ieee'" in BOUNDARY_INERT_UNDER_IEEE
    assert "Nothing crossed" in BOUNDARY_INERT_UNDER_IEEE
    assert "boundary='opaque' run would have used" in BOUNDARY_INERT_UNDER_IEEE


def test_the_inertness_disclosure_reaches_the_verdict():
    """A line that stops at the `Propagation` is a line no reader of a
    verdict ever sees. The ieee stamp is assembled by the same path the
    real-mode one is, so this is the ieee half of
    `::test_the_stamped_lines_reach_the_verdict`."""
    from stelling.propagate import (
        BOUNDARY_INERT_UNDER_IEEE, BOUNDARY_TRANSPARENT_POSITION,
    )
    from stelling.verdict import make_verdict

    q, _ = wrapped_sumsq_query("jit", assume_inside=False)
    versions = dict(
        stelling_version="test", jax_version="none",
        precision_config="jax_enable_x64=True",
    )
    v = make_verdict(
        q, propagate(q, semantics="ieee", boundary="transparent"), **versions
    )
    assert BOUNDARY_INERT_UNDER_IEEE in v.stamp.assumptions
    assert BOUNDARY_INERT_UNDER_IEEE in v.stamp.render()
    assert BOUNDARY_TRANSPARENT_POSITION not in v.stamp.render()


# ---------------------------------------------------------------------------
# 6. THE CERTIFICATE IS TRUE — exact rationals, across a boundary
# ---------------------------------------------------------------------------


def _certified_cells(query, eqns, points, *, boundary):
    """Run the propagator, then check every certificate it holds against
    exact `Fraction` evaluation at `points`. Returns `(signed vars,
    cell-checks performed, violations)`."""
    from stelling.propagate import _Propagator

    p = _Propagator("constrain", "real", None, boundary)
    p.run(query.jaxpr, list(query.consts), [])
    signs = dict(p.strict_sign)
    checks = 0
    bad = []
    for point in points:
        env = _exact_eval(eqns, point)
        for vid, sgn in signs.items():
            if vid not in env:  # a bool/predicate var carries no arithmetic
                continue
            for cell in env[vid]:
                checks += 1
                if not ((cell > 0) if sgn > 0 else (cell < 0)):
                    bad.append((vid, sgn, cell, point))
    return signs, checks, bad


@pytest.mark.parametrize("prim", sorted(DEFAULT_TRANSPARENT))
@pytest.mark.parametrize("assume_inside", [False, True])
@pytest.mark.parametrize("nest", [1, 2])
def test_a_BOUNDARY_CROSSED_certificate_is_TRUE_at_every_assumed_point(
    prim, assume_inside, nest
):
    """THE TEST THAT MAKES THIS CHANGE AUDITABLE: it checks the CLAIM, not
    the mechanism.

    The sibling of
    `tests/test_assume_bump_boundary_div.py::test_strict_sign_certificate_is_TRUE_at_every_assumed_point`,
    extended to the values that got their certificate ACROSS a boundary.
    Same exact-`Fraction` oracle, same sampled region, same positive
    control (`::test_the_semantic_check_catches_a_certificate_that_is_false`
    in that file, which proves the oracle can fail). No interval reasoning
    is reused: the oracle is an independent witness, not a restatement of
    the propagator.

    The certificate under attack is specifically the one on the WRAPPER's
    outvar — a var no rule in `_strict_sign_out` ever ran for, whose sign
    exists only because the boundary carried it.
    """
    query, eqns = wrapped_sumsq_query(
        prim, assume_inside=assume_inside, n=3, nest=nest
    )
    wrapper_out = [e for e in eqns if e.primitive == prim][0].outvars[0]
    signs, checks, bad = _certified_cells(
        query, eqns, _assumed_points(3, 0.0, 2.0, 3), boundary="transparent"
    )
    assert wrapper_out.id in signs, (
        f"{prim}: the wrapper's outvar carries no certificate, so this "
        f"check has nothing boundary-crossed to check: {signs}"
    )
    assert not bad, (
        f"{prim}: certificate FALSE at an assumed point — "
        + "; ".join(
            f"var {v} certified {s} but is {c} at {pt}" for v, s, c, pt in bad
        )
    )
    assert checks > 0


def test_the_boundary_certificate_check_is_not_checking_an_empty_table():
    """ABSENCE HALF for the test above: under the DEFAULT the wrapper's
    outvar has no certificate at all, so the same check there would pass
    over nothing. Stated so that a future change which quietly stops
    carrying cannot leave that test green and vacuous."""
    query, eqns = wrapped_sumsq_query("jit", assume_inside=False, n=3)
    wrapper_out = [e for e in eqns if e.primitive == "jit"][0].outvars[0]
    signs, _, _ = _certified_cells(
        query, eqns, _assumed_points(3, 0.0, 2.0, 2), boundary="opaque"
    )
    assert wrapper_out.id not in signs, signs


# ---------------------------------------------------------------------------
# 7. a generated search over the same property
# ---------------------------------------------------------------------------
#
# **WHY IT IS HERE AND NOT IN `tests/property/`.** That suite's subject is
# `stelling.preconditions.check` over harnesses drawn from
# `tests/property/_grammar.py`, and that grammar generates no sub-jaxpr
# wrapper at all — a boundary property there would need the shared grammar
# extended, which changes what every other property in that suite draws.
# Its positive-control machinery also materialises a COMMIT or applies a
# textual mutation to a scratch copy, and the mutation this property needs
# (carry the sign out of a cond) is driven directly in this branch instead.
# And `hypothesis` is installed in none of the three lanes this change is
# required green on, so a property there would be a search that does not
# run where the acceptance criteria are measured. The search below is
# therefore seeded and deterministic rather than Hypothesis-driven, and it
# runs on every lane.
#
# **WHAT IT SEARCHES.** Random arithmetic over one declared array, with
# random subtrees wrapped in a transparent wrapper, under
# `boundary="transparent"`; every certificate the finished top-level table
# holds is checked against exact `Fraction` evaluation at points of the
# assumed region.
#
# **WHAT IT DOES NOT SEARCH.** No `cond` (the oracle has no arm for one),
# no `sub` (dropped by the rules, so nothing to certify), no ieee, no
# cross-scope id collision (see `_ids`), and no shape but `(n,)` and `()`.

_GRAMMAR_OPS = ("mul", "add", "neg", "square", "reduce_sum", "lit_mul")


def _random_expr(rng, nxt, eqns, cur, n, depth, wrap_p=0.4):
    """Grow one random arithmetic step on `cur`, sometimes inside a
    wrapper. Returns the new value var and its element count (`n` for an
    array, 0 for a scalar)."""
    op = rng.choice([o for o in _GRAMMAR_OPS if n or o != "reduce_sum"])
    aval = _shaped(n) if n else F64
    body_eqns = []
    inner = nxt(aval)
    if op == "mul":
        out = nxt(aval)
        body_eqns.append(eqn("mul", [inner, inner], out))
        out_n = n
    elif op == "add":
        out = nxt(aval)
        body_eqns.append(eqn("add", [inner, inner], out))
        out_n = n
    elif op == "neg":
        out = nxt(aval)
        body_eqns.append(eqn("neg", [inner], out))
        out_n = n
    elif op == "square":
        out = nxt(aval)
        body_eqns.append(eqn("square", [inner], out))
        out_n = n
    elif op == "reduce_sum":
        out = nxt(F64)
        body_eqns.append(eqn("reduce_sum", [inner], out, [("axes", (0,))]))
        out_n = 0
    else:  # lit_mul
        out = nxt(aval)
        k = rng.choice([0.5, 2.0, -3.0, 1.25])
        body_eqns.append(eqn("mul", [lit(k), inner], out))
        out_n = n
    if depth > 0 and rng.random() < wrap_p:
        prim = rng.choice(sorted(DEFAULT_TRANSPARENT))
        body = ir.ClosedJaxpr(
            jaxpr=ir.Jaxpr(
                constvars=(), invars=(inner,), outvars=(out,),
                eqns=tuple(body_eqns),
            )
        )
        wrapped = nxt(_shaped(out_n) if out_n else F64)
        eqns.append(eqn(prim, [cur], wrapped, [("jaxpr", body)]))
        return wrapped, out_n
    # unwrapped: rewrite the body to act on `cur` directly
    for e in body_eqns:
        eqns.append(
            ir.JaxprEqn(
                primitive=e.primitive,
                invars=tuple(cur if a is inner else a for a in e.invars),
                outvars=e.outvars,
                params=e.params,
            )
        )
    return out, out_n


def _generated_query(rng, n=3, steps=4):
    """`assume(x > 0)`; a random chain; `q = 1/chain`; `assert_(q > 0)`
    AND `assert_(q < 0)`, in that order.

    **BOTH FACES, AND THROUGH A DIVISION, AND BOTH CORRECTIONS CAME FROM
    MEASUREMENT.**

    The obligation was first stated as `assert_(chain > 0)` alone, and the
    status split at this module's seed was 44 unknown / 16
    violated-over-set / **0 discharged** — an obligation-level check that
    exercised the false-REFUTED direction and never the false-VERIFIED
    one, which is the direction a verifier is judged on.

    Stating both faces of `chain` did not fix it: 60 unknown / 60
    violated-over-set / **still 0 discharged**, and the reason is the
    whole point of the certificate. A chain certified `-1` has a CLOSED
    box like `[-8, 0]`, because the exclusion of zero lives in the
    strict-sign table and not in the box — so `chain < 0` is not
    definitely true ON THE BOX and stays unknown, while `chain > 0` is
    definitely false and is violated. The asymmetry is structural, not a
    property of this grammar.

    The `div` is what reads the certificate. `1/chain` on a divisor box
    that reaches zero is ⊤ WITHOUT a certificate and a half-infinite box
    WITH one, so `q > 0` discharges exactly when the chain is certified
    positive and `q < 0` exactly when it is certified negative — which
    makes both obligations claims that the carry can move. Neither assert
    is chosen by what the propagator answered; both are always emitted,
    so nothing here is aimed at the tool's own reply.

    Returns `(closed, outer_eqns, quotient_var)`. The two obligations are
    recorded in walk order, so obligation 0 is the `> 0` face and
    obligation 1 is the `< 0` face; the search asserts that count rather
    than assuming it.
    """
    nxt = _ids()
    xa, ba = _shaped(n), _boolshaped(n)
    x = nxt(xa)
    pa, ao = nxt(ba), nxt(ba)
    eqns = [
        any_eqn_shaped(x, 0.0, 2.0, (n,)),
        eqn("gt", [x, lit(0.0)], pa),
        eqn("stelling_assume", [pa], ao),
    ]
    cur, cur_n = x, n
    for _ in range(steps):
        cur, cur_n = _random_expr(rng, nxt, eqns, cur, cur_n, depth=2)
    bo = _boolshaped(cur_n) if cur_n else BOOL
    quotient = nxt(_shaped(cur_n) if cur_n else F64)
    gt_pred, gt_out = nxt(bo), nxt(bo)
    lt_pred, lt_out = nxt(bo), nxt(bo)
    eqns += [
        eqn("div", [lit(1.0), cur], quotient),
        eqn("gt", [quotient, lit(0.0)], gt_pred),
        eqn("stelling_assert", [gt_pred], gt_out),
        eqn("lt", [quotient, lit(0.0)], lt_pred),
        eqn("stelling_assert", [lt_pred], lt_out),
    ]
    return close(eqns, [gt_out, lt_out]), eqns, quotient


def test_generated_boundary_certificates_are_TRUE_at_every_assumed_point():
    """A seeded search over the property the change rests on.

    THE FLOORS BELOW ARE TRIPWIRES FOR A SEARCH THAT COLLAPSED, not claims
    of thoroughness: a run that generated no wrapper, or certified
    nothing, would otherwise print the same green line as one that
    searched. MEASURED on this tree at seed 20260828, over 27 sampled
    points of the assumed region: 60 queries, 49 of them carrying at least
    one certificate across a boundary (8 at the most), 360 certified vars,
    22896 var-point cell checks, and 120 obligations splitting 60
    discharged / 60 violated-over-set / 0 unknown. The floors are set well
    under each of those, because a floor set at the measurement is a floor
    that reddens on the next harmless change to the grammar.

    THE PER-QUERY POSITIVE CONTROL is the negated table: if `v > 0` is
    true at a point then `v < 0` is false there, so a checker that is
    running at all must find a violation for every certificate it holds.
    A green run with the control silent would mean the exact evaluation
    never reached a certified var.

    **IT ALSO CHECKS THE OBLIGATION AND NOT ONLY THE TABLE**, which is
    the independent audit's own observation about this search: the
    certificate is the mechanism and the obligation is the claim, and it
    is the obligation-level property that catches an unsound carry
    whatever route it took to the table. Every generated query ends
    `assert_(chain > 0)`, so for each one:

    * a `discharged` obligation must be TRUE at every sampled point of
      the assumed region — the false-VERIFIED property, stated directly;
    * a `violated-over-set` obligation must be FALSE at every one — the
      false-REFUTED property, which the dial can also reach (see
      `::test_the_dial_can_also_move_UNKNOWN_to_REFUTED`);
    * an `unknown` obligation is asserted about in neither direction,
      because withholding is always allowed.

    **THE TWO FACES ARE COMPLEMENTARY AND NEITHER SUBSUMES THE OTHER**,
    which is worth stating because "check the claim, not the mechanism"
    reads like the obligation face should be strictly stronger. MEASURED,
    against a mutant that FLIPS the sign the wrapper out-carry writes
    (`self.strict_sign[out.id] = -out_signs[j]`), over these same 60
    queries and 27 points:

        certificate face   7803 violations
        obligation face       0 violations

    The certificate is wrong everywhere and no obligation moves, because
    `_t_div`'s gate reads only whether the divisor's sign is NONZERO —
    `stelling.interval.boundary_div` takes the DIRECTION from the box.
    So a direction defect is invisible at the claim level through `div`
    and visible only in the table; a defect that certified something
    genuinely unsigned would be the other way round. Both faces run.

    **AND THE SAMPLED REGION HAS TO MATCH THE QUERY'S.**
    `_assumed_points` samples the HALF-OPEN region `(0, hi]` — the region
    `assume(x > 0)` leaves — so every query this grammar builds must carry
    that assume, and every one does. A query without it would be judged
    over a region containing 0 while the oracle sampled one that does not,
    and the check would be answering a different question from the
    propagator's.
    """
    rng = random.Random(20260828)
    points = _assumed_points(3, 0.0, 2.0, 2)
    queries = crossed = certified = cells = 0
    judged = {"discharged": 0, "violated-over-set": 0, "unknown": 0}
    for _ in range(60):
        query, eqns, quotient = _generated_query(rng)
        queries += 1
        p = propagate(query, semantics="real", boundary="transparent")
        crossed += 1 if p.boundary_crossings else 0
        # THE OBLIGATION-LEVEL PROPERTY, at every sampled assumed point,
        # on both faces of the quotient. Obligation 0 is `q > 0` and
        # obligation 1 is `q < 0` (walk order, checked here).
        assert len(p.obligations) == 2, p.obligations
        faces = ((0, lambda c: c > 0), (1, lambda c: c < 0))
        for point in points:
            env = _exact_eval(eqns, point)
            for index, face in faces:
                status = p.obligations[index].status
                if status == "unknown":
                    continue  # withholding is always allowed
                want_true = status == "discharged"
                for cell in env[quotient.id]:
                    assert face(cell) is want_true, (
                        f"obligation #{index} is {status!r} but its "
                        f"predicate is {face(cell)} at assumed point "
                        f"{point} (chain = {cell})"
                    )
        for o in p.obligations:
            judged[o.status] = judged.get(o.status, 0) + 1
        signs, checks, bad = _certified_cells(
            query, eqns, points, boundary="transparent"
        )
        assert not bad, (
            "certificate FALSE at an assumed point: "
            + "; ".join(
                f"var {v} certified {s} but is {c} at {pt}"
                for v, s, c, pt in bad
            )
        )
        certified += len(signs)
        cells += checks
        if signs and checks:
            # the control: every certificate, negated, must be caught
            env = _exact_eval(eqns, points[0])
            caught = sum(
                1
                for vid, sgn in signs.items()
                if vid in env
                for cell in env[vid]
                if not ((cell > 0) if -sgn > 0 else (cell < 0))
            )
            assert caught > 0, (
                "the negated table produced no violation: the exact "
                "evaluation is not reaching any certified var, so the "
                "green above is a green from a search that could not speak"
            )
    assert queries == 60
    assert crossed >= 30, f"only {crossed} of {queries} carried anything"
    assert certified >= 60, certified
    assert cells >= 500, cells
    # the obligation-level half must have had something to judge, in the
    # direction that matters: a search in which nothing discharged would
    # check the false-VERIFIED property over an empty set. MEASURED at
    # seed 20260828 on this tree: the 60 split
    # discharged/violated-over-set/unknown as the assertion below floors.
    assert judged["discharged"] >= 10, judged
    assert judged["violated-over-set"] >= 10, judged
    assert sum(judged.values()) == 120, judged
