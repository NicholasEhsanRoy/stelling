# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The scatter fidelity gauge — measured discriminating power (the L21
instrument, ``stelling.fidelity.gauge``), plus the traced binding of the
real primitive names. Skipped without jax; the solver-backed gates
additionally need the portfolio (both wheels ship in the jax test env).

The subject is the registered transfer rows and the plan/emission/replay
bookkeeping underneath them; the battery is hand-built traced jnp
programs only (``jax.ops.segment_sum`` with duplicate-heavy static
segments, ``x.at[idx].add(v)`` with repeated indices over a NONZERO
operand, the scalar-index sugar, ``jnp.stack`` compositions, a small
normal-matrix assembly in the segment_sum style posed through the
standard pipeline, and static-index ``x.at[k].set(v)`` cases posed
relationally so the SET row's routing is load-bearing). The expected
residual is EMPTY: every mutation must be caught by a named gate, or the
gauge refuses loudly.

**THE SET ROW IS GAUGED HERE, AND IT IS THE ROW THE VERIFIED BAR EXISTS
FOR.** ``_scatter_set_plan`` had no mutation in this file while
``_scatter_add_plan`` had two — so the row still under
``verdict.VERIFIED_BARRED_PRIMITIVES`` was the one with no fidelity
gauge, and the row already cleared by a fresh adversarial auditor
(``design/scatter-rows.md``) was the one with one. That is backwards. The
audit that cleared the accumulate rows is point-in-time; a gauge is
continuous, and a row whose correctness rests on an event rather than on
a standing check has no tripwire between the event and the next change.
The three SET mutations are re-derived from the audit's description —
``off_by_one`` writes the wrong position, ``drop_write`` writes nothing,
``write_all`` writes every position — and each is pinned in the
MISSED-VIOLATION direction (a REFUTED that becomes ``discharged``), which
is the direction the bar exists to contain.
"""

from __future__ import annotations

import contextlib
from fractions import Fraction
from unittest import mock

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("z3")
pytest.importorskip("cvc5")

import math  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import stelling.obligation as OB  # noqa: E402
import stelling.propagate as P  # noqa: E402
import stelling.smt as SM  # noqa: E402
from stelling import interval as iv  # noqa: E402
from stelling import ir  # noqa: E402
from stelling.fidelity import gauge  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.obligation import (  # noqa: E402
    DeclinedObligation,
    slice_obligation,
)
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import interval_env, propagate  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


SEG = np.array([0, 0, 1, 0], dtype=np.int32)
IDX = np.array([0, 2, 0, 0], dtype=np.int32)
OWNER = np.array([0, 1, 0], dtype=np.int32)


# --- the traced binding (a row keyed on a name jax never emits would
# --- silently never fire) ----------------------------------------------------


def test_traced_names_bind_and_discharge():
    def h_seg():
        d = any_array((4,), "float64", (1.0, 1.0))
        s = jax.ops.segment_sum(d, jnp.asarray(SEG), num_segments=2)
        return assert_(s[0] >= 2.5)  # 1+1+1 accumulates to 3

    p = propagate(trace(h_seg))
    assert p.obligations[0].status == "discharged"
    assert ("scatter-add", "sound") in p.transfers_used

    def h_stack():
        a = any_array((2,), "float64", (1.0, 1.0))
        b = any_array((2,), "float64", (3.0, 3.0))
        return assert_(jnp.stack([a, b], axis=1)[0, 1] >= 2.5)

    p2 = propagate(trace(h_stack))
    assert p2.obligations[0].status == "discharged"
    assert ("stack", "exact") in p2.transfers_used

    def h_at():
        x = any_array((3,), "float64", (2.0, 2.0))
        v = any_array((4,), "float64", (5.0, 5.0))
        return assert_(x.at[jnp.asarray(IDX)].add(v)[0] >= 16.5)  # 2+15

    p3 = propagate(trace(h_at))
    assert p3.obligations[0].status == "discharged"


def test_f1_traced_totals_equal_across_dial_and_jit():
    """Third audit F1(a)/(c): the coverage denominator is a function of
    the PROGRAM — the same traced scatter-add program reports the same
    equation total under real and ieee (before the fix: 9 vs 8 — the
    recorded combiner's inner add vanished under the dial), and the
    jit-wrapped pair likewise (before: 10 vs 9)."""

    def h():
        d = any_array((4,), "float64", (1.0, 1.0))
        return assert_(
            jax.ops.segment_sum(d, jnp.asarray(SEG), num_segments=2)[0] >= 2.5
        )

    c0 = trace(h)
    assert (
        propagate(c0).coverage.total
        == propagate(c0, semantics="ieee").coverage.total
        == 9
    )

    def hj():
        d = any_array((4,), "float64", (1.0, 1.0))
        f = jax.jit(
            lambda dd: jax.ops.segment_sum(
                dd, jnp.asarray(SEG), num_segments=2
            )
        )
        return assert_(f(d)[0] >= 2.5)

    cj = trace(hj)
    assert (
        propagate(cj).coverage.total
        == propagate(cj, semantics="ieee").coverage.total
        == 10
    )


def test_f5b_default_dtype_index_sugar_propagates():
    """Third audit F5b, the traced payoff: the default-dtype at[].add
    sugar under x64 declares its index constants int64 and narrows them
    to int32 before the scatter; the statically-in-range narrowing now
    passes as an exact identity, so the accumulate row fires (before:
    the convert fell to ⊤, the index column went dynamic, and the row
    declined for a reason unrelated to its semantics)."""

    def h():
        x = any_array((3,), "float64", (2.0, 2.0))
        v = any_array((4,), "float64", (5.0, 5.0))
        idx = jnp.asarray([0, 2, 0, 0])  # default dtype: int64 under x64
        return assert_(x.at[idx].add(v)[0] >= 16.5)  # 2 + 5*3 = 17

    p = propagate(trace(h))
    assert p.obligations[0].status == "discharged"
    assert ("scatter-add", "sound") in p.transfers_used
    assert ("convert_element_type", "exact") in p.transfers_used


def test_traced_dynamic_index_declines_loudly():
    def h():
        x = any_array((3,), "float64", (0.0, 1.0))
        i = any_array((), "float64", (0.0, 2.0))
        v = any_array((), "float64", (0.0, 1.0))
        return assert_(x.at[i.astype(jnp.int32)].add(v)[0] >= -10.0)

    p = propagate(trace(h))
    assert p.obligations[0].status == "unknown"
    assert any("scatter-add" in n and "declined" in n for n in p.notes)


# --- the gauge ----------------------------------------------------------------
#
# Subject: {primitive -> transfer callable}, exercised by patching the
# live registry (tiers kept). Gates wrap their own exceptions: a mutant
# that crashes the analysis IS caught, while a baseline exception fails
# the gate and makes the gauge refuse (fidelity's invalid-stack refusal).


def _patched(subject):
    """Enter a subject: transfer entries patch the live registry; the
    optional ``__patches__`` key — a tuple of ``(module, attr, value)`` —
    patches module attributes, so PLAN/BUDGET/EMISSION-layer mutations
    are expressible too (third audit, F4c: a subject that can only swap
    transfers cannot express emission-side wrongness at all)."""
    transfers = {k: v for k, v in subject.items() if not k.startswith("__")}
    stack = contextlib.ExitStack()
    stack.enter_context(
        mock.patch.dict(
            P.TRANSFERS,
            {p: (fn, P.TRANSFERS[p][1]) for p, fn in transfers.items()},
        )
    )
    for mod, attr, val in subject.get("__patches__", ()):
        stack.enter_context(mock.patch.object(mod, attr, val))
    return stack


def _extract_ks(operand_shape, indices):
    ks = []
    for lo, hi in zip(indices.los, indices.his):
        if lo != hi or not math.isfinite(lo) or lo != math.floor(lo):
            raise iv.IntervalError("non-static index")
        k = int(lo)
        if not 0 <= k < operand_shape[0]:
            raise iv.IntervalError("out-of-range index")
        ks.append(k)
    return ks


def _scatter_variant(accumulate):
    """A scatter-add transfer with the SAME form/index discipline as the
    shipped row and a deliberately wrong accumulate loop."""

    def t(eqn, params, ins):
        if len(ins) != 3:
            return None
        operand, indices, updates = ins
        n = P._scatter_add_row_form(
            params, operand.shape, indices.shape, updates.shape,
            P._scatter_indices_dtype(eqn),
        )
        if n is None:
            return None
        ks = _extract_ks(operand.shape, indices)
        if updates.shape == operand.shape[1:]:
            updates = iv.reshape(updates, (1,) + operand.shape[1:])
        rowsz = 1
        for d in operand.shape[1:]:
            rowsz *= d
        los, his = list(operand.los), list(operand.his)
        accumulate(los, his, updates, ks, rowsz)
        outs = [iv.IntervalArray(shape=operand.shape, los=tuple(los), his=tuple(his))]
        return P._int_overflow_guard(eqn, "scatter-add", outs)

    return t


def _acc_last_wins(los, his, updates, ks, rowsz):
    # the set/add confusion: replace instead of accumulate
    for j, k in enumerate(ks):
        for t in range(rowsz):
            los[k * rowsz + t] = updates.los[j * rowsz + t]
            his[k * rowsz + t] = updates.his[j * rowsz + t]


def _acc_drop_duplicates(los, his, updates, ks, rowsz):
    seen = set()
    for j, k in enumerate(ks):
        if k in seen:
            continue
        seen.add(k)
        for t in range(rowsz):
            oi, ui = k * rowsz + t, j * rowsz + t
            los[oi] = iv._down(los[oi] + updates.los[ui])
            his[oi] = iv._up(his[oi] + updates.his[ui])


def _acc_off_by_one(los, his, updates, ks, rowsz):
    rows = len(los) // rowsz if rowsz else 0
    for j, k in enumerate(ks):
        k = (k + 1) % rows
        for t in range(rowsz):
            oi, ui = k * rowsz + t, j * rowsz + t
            los[oi] = iv._down(los[oi] + updates.los[ui])
            his[oi] = iv._up(his[oi] + updates.his[ui])


def _acc_inner_rounding(los, his, updates, ks, rowsz):
    for j, k in enumerate(ks):
        for t in range(rowsz):
            oi, ui = k * rowsz + t, j * rowsz + t
            los[oi] = iv._up(los[oi] + updates.los[ui])
            his[oi] = iv._down(his[oi] + updates.his[ui])


def _acc_updates_only(los, his, updates, ks, rowsz):
    # wrong operand-inclusion: the operand contribution is dropped
    first = set()
    for j, k in enumerate(ks):
        for t in range(rowsz):
            oi, ui = k * rowsz + t, j * rowsz + t
            if oi not in first:
                first.add(oi)
                los[oi] = updates.los[ui]
                his[oi] = updates.his[ui]
            else:
                los[oi] = iv._down(los[oi] + updates.los[ui])
                his[oi] = iv._up(his[oi] + updates.his[ui])


def _stack_axis_off_by_one(eqn, p, ins):
    axis = int(dict(p)["axis"] if isinstance(p, tuple) else p["axis"])
    rank = len(ins[0].shape)
    wrong = axis + 1 if axis + 1 <= rank else axis - 1
    return [iv.stack(list(ins), wrong)]


BASELINE = {
    "scatter-add": P.TRANSFERS["scatter-add"][0],
    "stack": P.TRANSFERS["stack"][0],
}

MUTATIONS = {
    "last-wins-instead-of-accumulate": {
        **BASELINE, "scatter-add": _scatter_variant(_acc_last_wins),
    },
    "drop-duplicate-contributions": {
        **BASELINE, "scatter-add": _scatter_variant(_acc_drop_duplicates),
    },
    "off-by-one-index-mapping": {
        **BASELINE, "scatter-add": _scatter_variant(_acc_off_by_one),
    },
    "inner-rounding-on-the-sum": {
        **BASELINE, "scatter-add": _scatter_variant(_acc_inner_rounding),
    },
    "updates-only-operand-dropped": {
        **BASELINE, "scatter-add": _scatter_variant(_acc_updates_only),
    },
    "stack-axis-off-by-one": {
        **BASELINE, "stack": _stack_axis_off_by_one,
    },
}


# --- the battery: hand-built traced programs ---------------------------------
#
# Each item: (value function usable both under the tracer and on concrete
# jnp arrays, declaration specs). The value var is located as the operand
# of the top-level comparison feeding the assert.

def v_seg(d):
    return jax.ops.segment_sum(d, jnp.asarray(SEG), num_segments=2)


def v_at(x, v):
    return x.at[jnp.asarray(IDX)].add(v)


def v_at_scalar(x, v):
    return x.at[1].add(v)


def v_stack0(a, b):
    return jnp.stack([a, b])


def v_stack1(a, b):
    return jnp.stack([a, b], axis=1)


def v_m(dx, dy):
    xx, xy, yy = dx * dx, dx * dy, dy * dy
    dd = jnp.stack(
        [jnp.stack([xx, xy], axis=1), jnp.stack([xy, yy], axis=1)], axis=1
    )
    return jax.ops.segment_sum(dd, jnp.asarray(OWNER), num_segments=2)


# (name, value_fn, [(shape, (lo, hi)), ...])
WIDE_BATTERY = [
    ("segment_sum-dup", v_seg, [((4,), (0.0, 2.0))]),
    ("at-add-dup-nonzero-operand", v_at, [((3,), (2.0, 3.0)), ((4,), (0.0, 1.0))]),
    ("at-add-scalar-idx", v_at_scalar, [((3,), (1.0, 2.0)), ((), (0.0, 1.0))]),
    ("stack-axis0", v_stack0, [((2,), (0.0, 1.0)), ((2,), (2.0, 3.0))]),
    ("stack-axis1", v_stack1, [((2,), (0.0, 1.0)), ((2,), (2.0, 3.0))]),
    ("m-assembly", v_m, [((3,), (0.4, 0.6)), ((3,), (-0.05, 0.05))]),
]

# integer-valued points: float sums are exact, so the true value sits ON
# the real endpoint and an inward-rounded bracket excludes it
POINT_BATTERY = [
    ("segment_sum-dup", v_seg, [((4,), (1.0, 1.0))]),
    ("at-add-dup-nonzero-operand", v_at, [((3,), (2.0, 2.0)), ((4,), (5.0, 5.0))]),
    ("at-add-scalar-idx", v_at_scalar, [((3,), (2.0, 2.0)), ((), (3.0, 3.0))]),
    ("stack-axis0", v_stack0, [((2,), (1.0, 1.0)), ((2,), (3.0, 3.0))]),
    ("stack-axis1", v_stack1, [((2,), (1.0, 1.0)), ((2,), (3.0, 3.0))]),
]


def _traced_value_query(value_fn, specs):
    """Trace ``assert_(value >= -1e12)`` — elementwise over the value, so
    the comparison's lhs var IS the value of interest."""

    def h():
        args = [any_array(s, "float64", b) for s, b in specs]
        return assert_(value_fn(*args) >= -1e12)

    return trace(h)


def _value_box(closed):
    """The propagated box of the comparison's lhs — the value of interest."""
    env = interval_env(closed)
    asserts = [e for e in closed.jaxpr.eqns if e.primitive == "stelling_assert"]
    pred = asserts[0].invars[0]
    producer = next(
        e for e in closed.jaxpr.eqns
        if any(o.id == pred.id for o in e.outvars)
    )
    return env[producer.invars[0].id]


def _samples(specs):
    """Deterministic sample grid: per-input linspace triplets zipped, plus
    the all-lo and all-hi corners."""
    grids = []
    for shape, (lo, hi) in specs:
        pts = [lo, (lo + hi) / 2.0, hi]
        grids.append([np.full(shape, p) for p in pts])
    combos = list(zip(*grids))
    combos.append(tuple(np.full(s, b[0]) for s, b in specs))
    combos.append(tuple(np.full(s, b[1]) for s, b in specs))
    # one asymmetric fill exercising distinct per-element routing
    combos.append(
        tuple(
            np.linspace(b[0], b[1], num=max(1, int(np.prod(s, dtype=int)))).reshape(s)
            for s, b in specs
        )
    )
    return combos


def gate_interval_soundness(subject):
    """Gate 1: dense-sampled true values are contained in the propagated
    per-element boxes, for every battery item."""
    try:
        with _patched(subject):
            for name, fn, specs in WIDE_BATTERY + POINT_BATTERY:
                closed = _traced_value_query(fn, specs)
                box = _value_box(closed)
                for combo in _samples(specs):
                    true = np.asarray(
                        fn(*(jnp.asarray(c) for c in combo))
                    ).reshape(-1)
                    if len(true) != box.size:
                        return False
                    for i, tv in enumerate(true):
                        if not (box.los[i] <= float(tv) <= box.his[i]):
                            return False
        return True
    except Exception:
        return False


def gate_point_box_exactness(subject):
    """Gate 2: point declarations propagate to boxes that contain the true
    value and are tight (a few outward ulps), per element."""
    try:
        with _patched(subject):
            for name, fn, specs in POINT_BATTERY:
                closed = _traced_value_query(fn, specs)
                box = _value_box(closed)
                args = [jnp.full(s, b[0]) for s, b in specs]
                true = np.asarray(fn(*args)).reshape(-1)
                for i, tv in enumerate(true):
                    tv = float(tv)
                    if not (box.los[i] <= tv <= box.his[i]):
                        return False
                    slack = 16 * math.ulp(max(abs(tv), 1.0))
                    if box.his[i] - box.los[i] > slack:
                        return False
        return True
    except Exception:
        return False


def _status_of(harness, subject, timeout=None):
    with _patched(subject):
        v = check(harness, vacuity_mode="inputs-only", solver_timeout_ms=timeout)
    return v


def _handir_nonzero_operand_case():
    """Hand-IR: a NONZERO-operand scatter-add posed relationally, so the
    operand term is LOAD-BEARING in the emitted sum (third audit, F4c —
    every traced segment_sum's operand is the zeros constant, making an
    operand-dropped emitted sum value-identical there; the traced
    nonzero-operand at[].add forms decline emission on the index
    normalization arithmetic, so this case must be hand-built).
    out = x.at[[0,0]].add(v); the obligation out >= x elementwise holds
    iff v0 + v1 >= 0 — solver-decided, VERIFIED."""
    def aval(shape=(), dtype="float64"):
        return ir.Aval(kind="ShapedArray", shape=shape, dtype=dtype)

    def var(i, a):
        return ir.Var(id=i, aval=a)

    import struct

    idx = ir.Literal(
        val=ir.Array(dtype="<i4", shape=(2, 1), data=struct.pack("<2i", 0, 0)),
        aval=aval((2, 1), "int32"),
    )
    x, v, out = var(0, aval((2,))), var(1, aval((2,))), var(2, aval((2,)))
    pred, ob = var(3, aval((2,), "bool")), var(4, aval((2,), "bool"))

    def any_eqn(o, lo, hi):
        return ir.JaxprEqn(
            primitive="stelling_any", invars=(), outvars=(o,),
            params=(("shape", o.aval.shape), ("dtype", o.aval.dtype),
                    ("lo", lo), ("hi", hi)),
        )

    dn = ir.NamedTupleParam(
        cls="ScatterDimensionNumbers",
        fields=(("update_window_dims", ()), ("inserted_window_dims", (0,)),
                ("scatter_dims_to_operand_dims", (0,)),
                ("operand_batching_dims", ()),
                ("scatter_indices_batching_dims", ())),
    )
    return ir.ClosedJaxpr(jaxpr=ir.Jaxpr(
        constvars=(), invars=(), outvars=(ob,),
        eqns=(
            any_eqn(x, 0.0, 1.0),
            any_eqn(v, 0.0, 1.0),
            ir.JaxprEqn(primitive="scatter-add", invars=(x, idx, v),
                        outvars=(out,),
                        params=(("dimension_numbers", dn),)),
            ir.JaxprEqn(primitive="ge", invars=(out, x), outvars=(pred,)),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,),
                        outvars=(ob,)),
        ),
    ))


_HANDIR_NONZERO = _handir_nonzero_operand_case()


def gate_emission_agreement(subject):
    """Gate 3: pipeline statuses on small crafted cases equal the
    brute-force/hand answers — point-decisive interval cases, a
    solver-decided relational case, a MULTI-SEGMENT case whose
    obligations are solver-decided PER SEGMENT (third audit F4a: with
    one effective output group, consistent group mis-routing was
    invisible), and a hand-IR NONZERO-operand relational case whose
    emitted sum's operand term is load-bearing (F4c)."""
    from stelling.solvers import SolverConfig, escalate

    def c1():  # accumulate of three 1.0s is 3: >= 2.5 holds
        d = any_array((4,), "float64", (1.0, 1.0))
        return assert_(v_seg(d)[0] >= 2.5)

    def c2():  # ... and <= 3.5 holds
        d = any_array((4,), "float64", (1.0, 1.0))
        return assert_(v_seg(d)[0] <= 3.5)

    def c3():  # relational: s[0] - d0 = d1 >= 0 — solver-decided (QF_LRA)
        d = any_array((2,), "float64", (0.0, 1.0))
        s = jax.ops.segment_sum(d, jnp.asarray(np.array([0, 0], dtype=np.int32)),
                                num_segments=1)
        return assert_(s[0] >= d[0])

    def c_multi():  # 3 segments, duplicates in segments 0 and 2; hand
        # answers per segment: s = [d0+d1, d2, d3+d4+d5]
        d = any_array((6,), "float64", (0.0, 1.0))
        s = jax.ops.segment_sum(
            d, jnp.asarray(np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)),
            num_segments=3,
        )
        assert_(s[0] >= d[0])          # d1 >= 0: holds
        assert_(s[1] <= d[2])          # equality: holds
        assert_(s[2] >= d[3] + d[4])   # d5 >= 0: holds
        return assert_(s[0] <= d[0])   # FALSE where d1 > 0: refuted

    try:
        if _status_of(c1, subject).status != "VERIFIED":
            return False
        if _status_of(c2, subject).status != "VERIFIED":
            return False
        if _status_of(c3, subject, timeout=10_000).status != "VERIFIED":
            return False
        vm = _status_of(c_multi, subject, timeout=10_000)
        if vm.status != "REFUTED":
            return False
        want = ("discharged", "discharged", "discharged", "violated-witness")
        if tuple(o.status for o in vm.obligations) != want:
            return False
        if [w.obligation_index for w in vm.witnesses] != [3]:
            return False
        # the hand-IR nonzero-operand relational case: VERIFIED, through
        # escalate directly (hand IR has no harness to re-trace)
        with _patched(subject):
            p = propagate(_HANDIR_NONZERO)
            esc = escalate(_HANDIR_NONZERO, p, SolverConfig(timeout_ms=10_000))
        if [r.outcome for r in esc.records] != ["discharged"]:
            return False
        return True
    except Exception:
        return False


def _m_statuses(harness, subject):
    with _patched(subject):
        p = propagate(trace(harness))
    return tuple(o.status for o in p.obligations)


def gate_unrolled_equivalence(subject):
    """Gate 4: the native scatter-add/stack assembly must produce THE SAME
    STATUSES as its unrolled slices+adds twin on identical declared boxes
    (intervals need not be bitwise equal; statuses must match)."""

    def obligations(native):
        dx = any_array((3,), "float64", (0.4, 0.6))
        dy = any_array((3,), "float64", (-0.05, 0.05))
        xx, xy, yy = dx * dx, dx * dy, dy * dy
        if native:
            M = v_m(dx, dy)
            m00, m01, m11 = M[0, 0, 0], M[0, 0, 1], M[0, 1, 1]
        else:
            m00 = xx[0] + xx[2]
            m01 = xy[0] + xy[2]
            m11 = yy[0] + yy[2]
        assert_(m00 + m11 >= 0.3)  # definite on baseline
        return assert_(m00 * m11 - m01 * m01 >= 0.0)  # straddles: unknown

    def h_native():
        return obligations(True)

    def h_twin():
        return obligations(False)

    def h_seg_native():
        d = any_array((4,), "float64", (1.0, 1.0))
        return assert_(v_seg(d)[0] >= 2.5)

    def h_seg_twin():
        d = any_array((4,), "float64", (1.0, 1.0))
        return assert_(d[0] + d[1] + d[3] >= 2.5)

    try:
        if _m_statuses(h_native, subject) != _m_statuses(h_twin, subject):
            return False
        if _m_statuses(h_seg_native, subject) != _m_statuses(h_seg_twin, subject):
            return False
        return True
    except Exception:
        return False


def gate_witness_replay_validity(subject):
    """Gate 5: a solver-refuted scatter-add obligation carries a witness
    that is in-box and hand-replays to a genuine violation. (The
    segment_sum index shape, deliberately: the array-index ``at[].add``
    trace carries jax's negative-index normalization — int32
    ``add``/``select_n`` — which the emission's registered integer
    posture declines; the segment_sum form is the census contact and
    emits fully.)"""

    def h():
        d = any_array((2,), "float64", (0.0, 1.0))
        s = jax.ops.segment_sum(
            d, jnp.asarray(np.array([0, 0], dtype=np.int32)), num_segments=1
        )
        return assert_(s[0] <= 1.5)

    try:
        verdict = _status_of(h, subject, timeout=10_000)
        if verdict.status != "REFUTED" or not verdict.witnesses:
            return False
        w = verdict.witnesses[0]
        vals = {name: Fraction(value) for name, value in w.values}
        for name, fr in vals.items():
            if not Fraction(0) <= fr <= Fraction(1):
                return False
        s0 = vals["x0_0"] + vals["x0_1"]  # 0 + d0 + d1: the accumulate
        return s0 > Fraction(3, 2) and bool(w.replay)
    except Exception:
        return False


def gate_budget_boundary(subject):
    """Gate 6 (third audit, F4b): the single per-obligation element budget
    must FIRE at its measured boundary, quoting both numbers. The traced
    segment_sum shape costs 2n + 3 element terms (n declared inputs, n
    update addends, operand + out + comparison): n = 254 → 511 ≤ 512
    slices; n = 255 → 513 declines. A mutation that disables the budget
    is caught by the decline not firing."""

    def h(n):
        def hh():
            d = any_array((n,), "float64", (0.0, 1.0))
            s = jax.ops.segment_sum(
                d, jnp.asarray(np.zeros(n, dtype=np.int32)), num_segments=1
            )
            return assert_(s[0] <= 1e9)

        return hh

    try:
        with _patched(subject):
            c254 = trace(h(254))
            sl = slice_obligation(c254, 0, interval_env(c254))
            if isinstance(sl, DeclinedObligation):
                return False  # under-budget side must slice
            c255 = trace(h(255))
            d = slice_obligation(c255, 0, interval_env(c255))
            if not isinstance(d, DeclinedObligation):
                return False  # over-budget side must decline...
            return (  # ...quoting both numbers and the budget
                "513" in d.reason
                and "512" in d.reason
                and "budget" in d.reason
            )
    except Exception:
        return False


def _set_row_records(build, subject):
    """Escalation outcomes for a static-index ``.set`` query, read from
    ``escalate`` rather than from ``check``. Deliberate: the scatter SET row
    is under the VERIFIED bar, so ``check`` reports UNKNOWN for a discharged
    obligation and a gate comparing verdict STATUS would be measuring the
    bar rather than the row. The record outcome is what the row decides."""
    from stelling.solvers import SolverConfig, escalate

    with _patched(subject):
        closed = trace(build)
        p = propagate(closed)
        esc = escalate(closed, p, SolverConfig(timeout_ms=10_000))
    return tuple(r.outcome for r in esc.records)


def _set_written_is_below_operand():
    """FALSE: ``s[0]`` is the written 0.5 and ``x[0]`` may be below it.
    Interval-undecidable (the difference propagates to [-0.5, 0.5]), so it
    escalates and the SET row's routing is what answers it."""
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    return assert_(s[0] - x[0] <= 0.0)


def _set_untouched_moved_by_one():
    """FALSE: element 1 is UNTOUCHED, so ``s[1] - x[1]`` is exactly 0, never
    ≥ 1. The update's box [2, 3] sits a clear distance above the operand's
    [0, 1], so any routing that hands element 1 the update instead makes the
    claim TRUE — the off-by-one and write-everywhere shapes."""
    x = any_array((3,), "float64", (0.0, 1.0))
    u = any_array((), "float64", (2.0, 3.0))
    s = x.at[0].set(u)
    return assert_(s[1] - x[1] >= 1.0)


def _set_untouched_equals_operand():
    """TRUE, and the other direction: element 1 is untouched, so the
    difference is exactly 0. A routing that writes there produces a SPURIOUS
    violation, so this case fails a mutation the two above do not."""
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    return assert_(s[1] - x[1] <= 0.0)


def gate_set_row_agreement(subject):
    """Gate 7: the static-index scatter SET row routes each output element to
    the right source. Three relational cases whose hand answers depend on
    exactly that: two FALSE claims whose refutation requires the write to
    have landed where it did, and one TRUE claim about an untouched element.

    Posed relationally on purpose — a point-decisive ``.set`` case is settled
    by the interval transfer, which these mutations do not touch, so it would
    measure nothing."""
    try:
        if _set_row_records(_set_written_is_below_operand, subject) != (
            "violated-witness",
        ):
            return False
        if _set_row_records(_set_untouched_moved_by_one, subject) != (
            "violated-witness",
        ):
            return False
        if _set_row_records(_set_untouched_equals_operand, subject) != (
            "discharged",
        ):
            return False
        return True
    except Exception:
        return False


GATES = {
    "interval-soundness": gate_interval_soundness,
    "point-box-exactness": gate_point_box_exactness,
    "emission-agreement": gate_emission_agreement,
    "unrolled-equivalence": gate_unrolled_equivalence,
    "witness-replay-validity": gate_witness_replay_validity,
    "budget-boundary": gate_budget_boundary,
    "set-row-agreement": gate_set_row_agreement,
}


# -- the plan/budget/emission-layer mutations (third audit, F4) ---------------

_REAL_PLAN = OB._scatter_add_plan


def _plan_collapse(eqns, consts, eqn):
    """Keep only the FIRST contribution per output element — the set-form
    confusion expressed at the PLAN layer (all three consumers)."""
    return [g[:1] for g in _REAL_PLAN(eqns, consts, eqn)]


def _plan_rotate(eqns, consts, eqn):
    """Rotate the contribution groups one output element — consistent
    mis-routing across emission AND replay (the auditor's survivor: with
    one effective output group it was invisible)."""
    groups = _REAL_PLAN(eqns, consts, eqn)
    if len(groups) > 1:
        return groups[1:] + groups[:1]
    return groups


def _sum_body_drop_operand(operand_term, update_terms):
    """The emitted accumulate without its operand addend — emission-side
    wrongness the replay does not share, so the infidelity machinery must
    catch it on a load-bearing operand."""
    if len(update_terms) == 1:
        return update_terms[0]
    return f"(+ {' '.join(update_terms)})"


_REAL_SET_PLAN = OB._scatter_set_plan


def _set_plan_off_by_one(eqns, consts, eqn):
    """WRITES THE WRONG POSITION: the update lands one element along. Built
    over the real plan so every decline it makes is inherited — a mutation
    that also relaxes the row's admission would be measuring two things."""
    routes = _REAL_SET_PLAN(eqns, consts, eqn)
    n = len(routes)
    written = [i for i, (op, _src) in enumerate(routes) if op == 2]
    if not written or n < 2:
        return routes
    k = (written[0] + 1) % n
    return [(2, 0) if i == k else (0, i) for i in range(n)]


def _set_plan_drop_write(eqns, consts, eqn):
    """WRITES NOTHING: every output element aliases the operand, i.e. the
    result of a DROPPED out-of-range update applied to an in-range one."""
    return [(0, i) for i in range(len(_REAL_SET_PLAN(eqns, consts, eqn)))]


def _set_plan_write_all(eqns, consts, eqn):
    """WRITES EVERY POSITION: the one scalar update is broadcast over the
    whole output — `x.at[:].set(v)` where the program wrote one element."""
    return [(2, 0) for _ in _REAL_SET_PLAN(eqns, consts, eqn)]


MUTATIONS = {
    **MUTATIONS,
    "set-plan-off-by-one-position": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_set_plan", _set_plan_off_by_one),
            (SM, "_scatter_set_plan", _set_plan_off_by_one),
        ),
    },
    "set-plan-drops-the-write": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_set_plan", _set_plan_drop_write),
            (SM, "_scatter_set_plan", _set_plan_drop_write),
        ),
    },
    "set-plan-writes-every-position": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_set_plan", _set_plan_write_all),
            (SM, "_scatter_set_plan", _set_plan_write_all),
        ),
    },
    "plan-rotate-groups": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_add_plan", _plan_rotate),
            (SM, "_scatter_add_plan", _plan_rotate),
        ),
    },
    "collapse-duplicates-in-the-plan": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_add_plan", _plan_collapse),
            (SM, "_scatter_add_plan", _plan_collapse),
        ),
    },
    "budget-never-fires": {
        **BASELINE,
        "__patches__": ((OB, "ELEMENT_BUDGET", 10**9),),
    },
    "operand-dropped-from-the-emitted-sum": {
        **BASELINE,
        "__patches__": ((SM, "_scatter_add_sum_body", _sum_body_drop_operand),),
    },
}


def test_gauge_catches_every_mutation():
    report = gauge(
        BASELINE, GATES, MUTATIONS, residual={},
        scope=("BOTH faces of the scatter-add and stack rows plus the paths "
               "downstream of them: interval soundness and point-box "
               "exactness on the transfer, emission agreement, unrolled "
               "equivalence, witness replay validity, and the element "
               "budget. ALSO the static-index scatter SET row's routing "
               "bookkeeping (_scatter_set_plan) across its three consumers "
               "— slice validation, emission and replay — through "
               "relational cases the interval transfer cannot settle. Does "
               "NOT drive the SET row's interval transfer (no SET mutation "
               "here is transfer-side), and does not drive any other "
               "primitive's rows."),
    )
    render = report.render()
    print("\n" + render)
    # expected residual: EMPTY — every mutation is caught (the gauge
    # itself refuses on an unexplained survivor; this pins it)
    assert report.residual == ()
    caught = dict(report.caught_by)
    assert caught["last-wins-instead-of-accumulate"], "the set/add confusion escaped"
    for name in MUTATIONS:
        assert caught[name], name
    # the mandated third-audit depths (F4a/b/c): the auditor's survivors
    # are now caught, collapse-in-the-plan at TWO independent gates or
    # more, and the budget/emission layers are gauged
    assert caught["plan-rotate-groups"]
    assert len(caught["collapse-duplicates-in-the-plan"]) >= 2, caught
    assert "budget-boundary" in caught["budget-never-fires"]
    assert caught["operand-dropped-from-the-emitted-sum"]
    # the SET row — the row the VERIFIED bar exists for — has a gauge now,
    # and the gate that catches its mutations is the SET-row gate rather
    # than an accumulate gate that happened to notice
    for name in ("set-plan-off-by-one-position", "set-plan-drops-the-write",
                 "set-plan-writes-every-position"):
        assert "set-row-agreement" in caught[name], (name, caught[name])
