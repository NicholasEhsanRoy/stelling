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


def _set_written_at_a_NONZERO_index():
    """FALSE, and the only fixture here whose answer depends on WHICH index
    was written rather than on whether anything was.

    ``s = x.at[2].set(u)`` with the update's box [2, 3] a clear distance above
    the operand's [0, 1]: element 1 is untouched, so ``s[1] - x[1]`` is
    exactly 0 and never ≥ 1. The claim is FALSE, and a route computed
    RELATIVE to k — one element before it, say — hands element 1 the update
    and brings it back `discharged`.

    THE THREE FIXTURES ABOVE CANNOT SEE THAT, AND THE REASON IS ARITHMETIC.
    They all write k = 0, where every route defined as an offset from k is
    either out of range or clamped back onto 0 itself. Measured, on the
    line-neutral corruption ``i == k`` -> ``i == (k - 1 if k > 0 else k)`` in
    `_scatter_set_plan`, applied to `45cf526` before this fixture existed:
    ``s = x.at[2].set(u); assert s[1] - x[1] >= 1.0`` went from
    `violated-witness` to `discharged` — a missed violation, the direction
    the bar exists for — while the whole scatter/bar screen (the seven
    `test_scatter*`, `test_verified_bar` and `test_bar_walk_parity` files)
    stayed green at 80 passed, and the FULL suite under CI's install set
    stayed green at 2004 passed, 6 skipped. Two tests caught it, both
    `pytest.importorskip("maddening")`-gated, and CI installs `".[solvers]"`
    and `".[solvers,jax]"` — never maddening. With this fixture the same
    corruption is 1 failed, 2007 passed, 6 skipped with no maddening
    installed.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    u = any_array((), "float64", (2.0, 3.0))
    s = x.at[2].set(u)
    return assert_(s[1] - x[1] >= 1.0)


def _set_plan_routes(n, k):
    """`_scatter_set_plan`'s routes for `x.at[k].set(u)` on a length-`n`
    operand, taken from a real traced query."""
    def build():
        x = any_array((n,), "float64", (0.0, 1.0))
        u = any_array((), "float64", (2.0, 3.0))
        return assert_(x.at[k].set(u) >= 0.0)

    closed = trace(build)
    eqn = next(e for e in closed.jaxpr.eqns if str(e.primitive) == "scatter")
    consts = dict(zip((v.id for v in closed.jaxpr.constvars), closed.consts))
    return OB._scatter_set_plan(closed.jaxpr.eqns, consts, eqn), eqn


# -- the ROUTE SPACE, generated by a rule rather than listed -----------------
#
# WHAT WAS WRONG WITH THE PREDECESSOR, WHICH WAS ALREADY GOOD. It exhausted k
# and SAMPLED n: `@parametrize("n", [3, 4, 5, 8])`, and `[3, 4]` at the
# surface. Sampling an axis is exactly the defect its own docstring diagnosed
# one axis over — it argued that k = {0, 2} was a sample and that corruptions
# walk between samples, then left n a sample. They do walk. Measured on this
# branch, line-neutral, `_scatter_set_plan`'s last line:
#
#     i == (k if n != 6 else 0)
#         `x=(6,); s=x.at[2].set(u); assert s[0]-x[0] >= 1.0`
#         violated-witness -> DISCHARGED (a MISSED violation)
#         full suite green: 2046 passed / 2 skipped, and 2042 / 6 under CI's
#         install set; the whole scatter+bar screen green at 118 passed
#
# and `_scatter_add_plan`'s row arithmetic was not gauged above rank 1 at all
# — this file contained no rank-2 `any_array` anywhere, and `rowsz` is 1 at
# rank 1, so every row-index expression collapses to the same value:
#
#     groups[k * rowsz + t] -> groups[k * rowsz]
#         `x=(3,2); s=x.at[1].add(u); assert s[1,0]-x[1,0] >= 4.0`
#         violated-witness -> DISCHARGED, same two green columns
#
# ADDING n = 6 AND A RANK-2 FIXTURE IS THE SAME MISTAKE ONE ROUND LATER. What
# follows sweeps a SPACE defined by a rule, asserts the space is the rule's
# (so shrinking it back to a list is itself a failure), and states what the
# space does not reach. The oracle stays jax's own execution.
#
# AND SWEEPING A SPACE BY A RULE WAS STILL NOT ENOUGH, WHICH IS THE FIFTH
# INSTANCE OF ONE PATTERN. Two more line-neutral corruptions walked straight
# past this file, both a `violated-witness` turned `discharged`, both with the
# full suite green in BOTH columns (2044 / 2 and 2040 / 6) on `e35de13`:
#
#   * `i == (k if n != 9 else 0)` in the SET row — `_set_space` stops at 8 and
#     the surface at 6, so n = 9 is gauged by nothing;
#   * `groups[(k if operand_shape[0] < 4 else 0) * rowsz + t]` in the ADD row
#     — `_add_space`'s dims stop at 3, so it contains NO axis of length >= 4,
#     and the non-degeneracy clause below was on `rowsz`, the TRAILING
#     product, which structurally cannot see a LEADING-axis-keyed corruption.
#
# RAISING THE BOUNDS IS THE MOVE THIS PROJECT HAS NOW WATCHED FAIL FOUR TIMES
# (k = {0} -> {0,2}; field probes by name -> by type; an arity family 2 -> 3
# -> 8, where the escape sat at EXACTLY the declared ceiling). So the bounds
# below are not raised — they are GUARDED: `stelling.obligation` now DECLINES
# any shape outside them, and `test_nothing_outside_the_swept_space_reaches_
# either_row` pins the two spaces EQUAL in both directions. Past the bound the
# rows do not run ungauged; they refuse, and the obligation comes back
# `unknown` instead of `discharged` or `violated-witness`.
_SET_MAX_N = 8  # the SET row is rank-1 only (rank >= 2 declines; measured)
_ADD_MAX_DIM = 3
_ADD_MAX_RANK = 3
_ADD_MAX_SIZE = 12  # keeps the sweep's trace count, and its seconds, bounded


def _set_space():
    """Every (axis length, written index) the rank-1 SET row admits, up to
    `_SET_MAX_N`. A RULE: `1 <= n <= N`, `0 <= k < n` — not a list."""
    return [(n, k) for n in range(1, _SET_MAX_N + 1) for k in range(n)]


def _add_space():
    """Every (operand shape, written index) the accumulate row admits, over
    shapes of rank 1..`_ADD_MAX_RANK` with dims in 1..`_ADD_MAX_DIM` and at
    most `_ADD_MAX_SIZE` elements."""
    import itertools

    shapes = [
        shape
        for rank in range(1, _ADD_MAX_RANK + 1)
        for shape in itertools.product(range(1, _ADD_MAX_DIM + 1), repeat=rank)
        if math.prod(shape) <= _ADD_MAX_SIZE
    ]
    return [(shape, k) for shape in shapes for k in range(shape[0])]


def test_the_set_plans_k_is_the_programs_k_over_the_whole_AXIS_SPACE():
    """THE PROPERTY OVER BOTH AXES, NOT MORE SAMPLES ON EITHER.

    For every axis length the row admits and every writable index of that
    axis, the plan routes exactly the element the program writes — checked
    against jax's own execution of the same `.set` on values that identify
    their own source, so the expectation is an oracle and not a second copy of
    the rule.

    COVERAGE, stated because a sweep that does not say what it covers is a
    sample wearing a rule's clothes:

    * axis length: EXHAUSTED for 1 <= n <= 8. Not swept above 8, and — since
      `test_nothing_outside_the_swept_space_reaches_either_row` — not
      ADMITTED above 8 either, so "not swept" no longer means "runs
      ungauged".
    * written index: EXHAUSTED, every k of every one of those axes.
    * rank: 1 ONLY, and that is the row's own boundary rather than this
      test's. Measured here: `x.at[1].set(v)` on an operand of shape (3, 2),
      (4, 3) or (3, 2, 2) DECLINES — "outside the measured static-index set
      row form" — for scalar and row-shaped updates alike. The declines are
      asserted below, so a row that silently starts admitting rank 2 fails
      here instead of running ungauged.
    * NOT covered, by construction: dynamic indices (declined, and gauged by
      `test_traced_dynamic_index_declines_loudly`), out-of-range indices
      (declined), multi-axis `.at[i, j]` (declined), and `mode` other than
      FILL_OR_DROP (declined).
    """
    space = _set_space()
    assert {n for n, _ in space} == set(range(1, _SET_MAX_N + 1)), (
        "the swept axis lengths are no longer every n up to the bound — the "
        "space has been narrowed back to a sample, which is the defect this "
        "test replaced"
    )
    assert 6 in {n for n, _ in space}, (
        "n = 6 is not in the space. It is named because a line-neutral "
        "mis-route wrong ONLY at n = 6 was green against the sampled "
        "{3, 4, 5, 8} with the whole suite passing"
    )
    assert all(ks == list(range(n)) for n, ks in (
        (n, [k for m, k in space if m == n]) for n in {n for n, _ in space}
    )), "some axis is not swept at every k"

    for n, k in space:
        routes, eqn = _set_plan_routes(n, k)
        # invars are (operand, indices, updates); the plan's positions index
        # exactly that tuple, which the row's docstring states and the shipped
        # emission relies on
        assert len(eqn.invars) == 3, eqn.invars
        expected = [(2, 0) if j == k else (0, j) for j in range(n)]

        # THE ORACLE: run the real program on values that identify their own
        # source, and confirm `expected` describes what jax actually does. If
        # this half fails, `expected` is wrong and the comparison below would
        # have been a rule compared against itself.
        xs = np.arange(1.0, n + 1.0, dtype=np.float64) * 10.0
        uv = np.float64(-7.0)
        operands = [xs, np.full(1, 1e9, dtype=np.float64), np.array([uv])]
        actual = np.asarray(jnp.asarray(xs).at[k].set(jnp.asarray(uv)))
        predicted = np.array(
            [operands[op].reshape(-1)[src] for op, src in expected]
        )
        assert np.array_equal(predicted, actual), (
            f"n={n} k={k}: the expected routes {expected} predict "
            f"{list(predicted)} where jax computes {list(actual)} — the "
            f"expectation, not the plan, is the wrong one"
        )

        assert routes == expected, (
            f"n={n} k={k}: `_scatter_set_plan` routes {routes} where the "
            f"program writes element {k} ({expected}). The plan's k is not "
            f"the program's k at this index, so every consumer of it — slice "
            f"validation, emission, replay — models a write the program did "
            f"not make"
        )

    # the rank boundary this sweep stops at, asserted rather than assumed
    for shape, ushape in (((3, 2), ()), ((3, 2), (2,)), ((4, 3), (3,)),
                          ((3, 2, 2), (2, 2))):
        def build(shape=shape, ushape=ushape):
            x = any_array(shape, "float64", (0.0, 1.0))
            u = any_array(ushape, "float64", (2.0, 3.0))
            return assert_(x.at[1].set(u).reshape(-1)[0] >= -1e9)

        closed = trace(build)
        eqn = next(e for e in closed.jaxpr.eqns if str(e.primitive) == "scatter")
        consts = dict(zip((v.id for v in closed.jaxpr.constvars), closed.consts))
        with pytest.raises(Exception) as exc:
            OB._scatter_set_plan(closed.jaxpr.eqns, consts, eqn)
        assert "outside the measured static-index set row form" in str(exc.value), (
            f"operand {shape} with updates {ushape} no longer declines out of "
            f"the SET row. The sweep above is rank-1 only BECAUSE the row is; "
            f"if that has changed, the sweep must follow it, not this "
            f"assertion be deleted"
        )


def test_the_add_plans_rows_are_the_programs_rows_over_the_SHAPE_SPACE():
    """THE ROW ARITHMETIC, GAUGED ABOVE RANK 1 AT ALL — it was not, and
    `rowsz` is 1 at rank 1, so every row-index expression in
    `_scatter_add_plan` collapsed to the same value and the whole of it was
    ungauged.

    For every shape the accumulate row admits within the bound, and every
    index of its leading axis, the plan's groups are exactly the updates
    elements jax actually accumulates onto each output element — measured by
    running the real `.add` on updates that are distinct powers of two over a
    zero operand, so each output element's value NAMES its addend set. That is
    an oracle: nothing about the expectation is derived from the plan.

    COVERAGE:

    * rank: EXHAUSTED for 1 <= rank <= 3.
    * dims: EXHAUSTED for 1..3 per axis, subject to at most 12 elements.
    * written index: EXHAUSTED, every k of the leading axis of every shape.
    * NOT covered: dims above 3, rank above 3, operands above 12 elements, and
      multiple index columns (`x.at[jnp.array([0, 2])].add(...)` is a
      different form, gauged by the segment-sum battery above). The first
      three are no longer ADMITTED either — see
      `test_nothing_outside_the_swept_space_reaches_either_row`. The fourth
      still is, and that is stated rather than closed.

    NON-DEGENERACY IS ASSERTED, not assumed, ON BOTH AXES OF THE INDEX
    EXPRESSION. `groups[k * rowsz + t]` has two ungaugeable directions and the
    first version of this clause guarded only one: `rowsz` (the product of the
    TRAILING axes — 1 at rank 1, which is what went ungauged) and `k`'s own
    axis, the LEADING one. A corruption keyed on the leading axis is invisible
    to a `rowsz` clause however wide that clause is, and one was measured:
    `groups[(k if operand_shape[0] < 4 else 0) * rowsz + t]`, green through
    the whole suite in both columns. Both must vary across the space now, and
    shrinking either back fails here.
    """
    space = _add_space()
    ranks = {len(shape) for shape, _ in space}
    rowsizes = {math.prod(shape[1:]) for shape, _ in space}
    leading = {shape[0] for shape, _ in space}
    assert ranks == set(range(1, _ADD_MAX_RANK + 1)), (
        f"the swept ranks are {sorted(ranks)}; the space is no longer the "
        f"rule's"
    )
    assert len(rowsizes - {1}) >= 2, (
        f"`rowsz` takes values {sorted(rowsizes)} across the space. At rank 1 "
        f"it is 1 and every row-index expression collapses — a space that "
        f"does not vary it cannot see the row arithmetic at all, which is "
        f"exactly the state this test was written to end"
    )
    assert len(leading - {1}) >= 2, (
        f"the LEADING axis takes values {sorted(leading)} across the space. "
        f"It is the axis `k` indexes, and the `rowsz` clause above cannot see "
        f"it: a corruption keyed on the leading axis alone was measured green "
        f"through the whole suite in both columns"
    )
    assert leading == set(range(1, _ADD_MAX_DIM + 1)), (
        f"the swept leading axes are {sorted(leading)}, not every length the "
        f"rule admits — the space has been narrowed back to a sample on the "
        f"one axis the non-degeneracy clause above was blind to"
    )

    for shape, k in space:
        def build(shape=shape, k=k):
            x = any_array(shape, "float64", (0.0, 1.0))
            u = any_array(shape[1:], "float64", (2.0, 3.0))
            return assert_(x.at[k].add(u).reshape(-1)[0] >= -1e9)

        closed = trace(build)
        eqn = next(e for e in closed.jaxpr.eqns
                   if str(e.primitive) == "scatter-add")
        consts = dict(zip((v.id for v in closed.jaxpr.constvars), closed.consts))
        groups = OB._scatter_add_plan(closed.jaxpr.eqns, consts, eqn)

        # THE ORACLE. Zero operand, updates 1, 2, 4, 8, ... so every output
        # element's value is the SUM of its addends and therefore names the
        # set of them uniquely (each updates element contributes at most once
        # under a single index column).
        rowsz = math.prod(shape[1:])
        uv = np.array([2.0 ** j for j in range(rowsz)],
                      dtype=np.float64).reshape(shape[1:])
        out = np.asarray(
            jnp.zeros(shape, jnp.float64).at[k].add(jnp.asarray(uv))
        ).reshape(-1)
        observed = [
            sorted(j for j in range(rowsz) if int(value) >> j & 1)
            for value in out
        ]
        assert sum(len(g) for g in observed) == rowsz, (
            f"shape={shape} k={k}: the oracle recovered "
            f"{sum(len(g) for g in observed)} addend(s) from jax's own output "
            f"where the program supplies {rowsz} — the sentinel values no "
            f"longer identify their source and the comparison below would be "
            f"meaningless"
        )
        assert [sorted(g) for g in groups] == observed, (
            f"shape={shape} k={k}: `_scatter_add_plan` groups {groups} where "
            f"jax accumulates {observed}. Every consumer of the plan — slice "
            f"validation, emission, replay — then models an accumulation the "
            f"program did not perform"
        )


def _set_plan_for(n, k=0):
    """`_scatter_set_plan`'s answer for `x.at[k].set(u)` on a length-`n`
    operand: the routes, or the quoted decline."""
    def build():
        x = any_array((n,), "float64", (0.0, 1.0))
        u = any_array((), "float64", (2.0, 3.0))
        return assert_(x.at[k].set(u)[0] >= -1e9)

    closed = trace(build)
    eqn = next(e for e in closed.jaxpr.eqns if str(e.primitive) == "scatter")
    consts = dict(zip((v.id for v in closed.jaxpr.constvars), closed.consts))
    try:
        return ("admitted", OB._scatter_set_plan(closed.jaxpr.eqns, consts, eqn))
    except OB._Decline as d:
        return ("declined", str(d))


def _add_plan_for(shape):
    """`_scatter_add_plan`'s answer for `x.at[0].add(u)` on `shape`: the
    groups, or the quoted decline."""
    def build():
        x = any_array(shape, "float64", (0.0, 1.0))
        u = any_array(shape[1:], "float64", (2.0, 3.0))
        return assert_(x.at[0].add(u).reshape(-1)[0] >= -1e9)

    closed = trace(build)
    eqn = next(e for e in closed.jaxpr.eqns
               if str(e.primitive) == "scatter-add")
    consts = dict(zip((v.id for v in closed.jaxpr.constvars), closed.consts))
    try:
        return ("admitted", OB._scatter_add_plan(closed.jaxpr.eqns, consts, eqn))
    except OB._Decline as d:
        return ("declined", str(d))


def test_nothing_outside_the_swept_space_reaches_either_row():
    """GUARD THE BOUND — the answer that is NOT "raise the bound".

    Both sweeps above are exhaustive over a space and blind one step past it,
    and two line-neutral corruptions were measured living in that step: SET
    wrong only at n = 9, ADD wrong only at a leading axis of 4 or more, each a
    `violated-witness` turned `discharged` with the full suite green in both
    columns. This project has watched "raise the bound" fail four times
    already — the last time the escape sat at EXACTLY the new ceiling — so
    the repair is not a larger number.

    **The ADMITTED space must not exceed the GAUGED space.** Past these bounds
    the rows DECLINE, so the routing code cannot execute on a shape nothing
    measured: the corrupted branch is unreachable rather than uncaught, and a
    row that silently starts admitting one FAILS HERE instead of running
    ungauged. This is the same discipline the rank-2 decline assertion in
    `test_the_set_plans_k_is_the_programs_k_over_the_whole_AXIS_SPACE` already
    applied to rank, extended to the two axes that had no such assertion.

    PINNED IN BOTH DIRECTIONS, which is what makes it a guard rather than a
    restatement: the sweep bound and the admission bound are asserted EQUAL
    (so widening admission without widening the sweep is red, and shrinking
    the sweep below admission is red), everything INSIDE the space must still
    be admitted (so a row that declined everything could not read as perfect),
    and everything just OUTSIDE must decline BY ITS REASON.

    Could-not-fail: the shape to avoid here is #3, asserting away its own
    trigger — a guard that declined the swept space too would make both sweeps
    vacuous. The in-space half below is what stops that, and the two sweeps
    themselves would go red if it were violated.

    WHAT IT DOES NOT BOUND, stated because an unstated scope is this repo's
    own recurring defect: the ADD row's INDEX COLUMN LENGTH. `_add_space`
    sweeps one written index; `jax.ops.segment_sum` reaches a column of 4 on
    an operand these bounds admit, and that axis is gauged by the segment-sum
    mutation battery — a battery, not an exhaustive sweep. Nothing here
    narrows admission to it.
    """
    # the two spaces are ONE space, not two numbers that happen to agree
    assert OB._SET_ROW_GAUGED_MAX_LEN == _SET_MAX_N, (
        f"the SET row admits up to {OB._SET_ROW_GAUGED_MAX_LEN} while this "
        f"file sweeps to {_SET_MAX_N}. Whichever moved, the other must: the "
        f"gap between them is exactly where the n = 9 corruption lived"
    )
    assert (OB._ADD_ROW_GAUGED_MAX_RANK, OB._ADD_ROW_GAUGED_MAX_DIM,
            OB._ADD_ROW_GAUGED_MAX_SIZE) == (
        _ADD_MAX_RANK, _ADD_MAX_DIM, _ADD_MAX_SIZE), (
        f"the ADD row admits rank/dim/size "
        f"{(OB._ADD_ROW_GAUGED_MAX_RANK, OB._ADD_ROW_GAUGED_MAX_DIM, OB._ADD_ROW_GAUGED_MAX_SIZE)} "
        f"while this file sweeps {(_ADD_MAX_RANK, _ADD_MAX_DIM, _ADD_MAX_SIZE)}"
    )

    # IN the space: still admitted, and still routed. Without this half a row
    # that declined everything would pass the decline assertions below.
    for n, k in _set_space():
        kind, payload = _set_plan_for(n, k)
        assert kind == "admitted" and payload == [
            (2, 0) if j == k else (0, j) for j in range(n)
        ], f"n={n} k={k}: the SET row no longer admits its own gauged space ({payload})"
    for shape, _k in _add_space():
        kind, payload = _add_plan_for(shape)
        assert kind == "admitted", (
            f"{shape}: the ADD row no longer admits its own gauged space "
            f"({payload})"
        )

    # JUST OUTSIDE: declines, by its reason. One step past each bound and one
    # far past it, because a guard written as `== N + 1` is not a bound.
    for n in (_SET_MAX_N + 1, _SET_MAX_N + 2, 64, 200):
        kind, payload = _set_plan_for(n, k=0)
        assert kind == "declined" and "outside the GAUGED" in payload, (
            f"n={n}: the SET row {kind} an axis length nothing gauges "
            f"({payload}). A line-neutral mis-route wrong only at n = 9 was "
            f"measured green through the whole suite in both columns; the "
            f"only thing standing between that and a missed violation is "
            f"this decline"
        )
    for shape in ((_ADD_MAX_DIM + 1,), (_ADD_MAX_DIM + 1, 2), (1, 5), (9,),
                  (3, 3, 2), (2, 2, 2, 2), (1, 1, 1, 1)):
        kind, payload = _add_plan_for(shape)
        assert kind == "declined" and "outside the GAUGED" in payload, (
            f"{shape}: the ADD row {kind} a shape nothing gauges ({payload}). "
            f"A corruption keyed on a LEADING axis of 4 or more was measured "
            f"green through the whole suite in both columns"
        )

    # ... and the shapes just outside really are outside, so the two loops
    # above are not both testing the same side of the boundary
    assert (9,) not in {shape for shape, _ in _add_space()}
    assert 9 not in {n for n, _ in _set_space()}


def test_a_shape_past_the_gauge_costs_the_ANSWER_and_never_the_SOUNDNESS():
    """THE SURFACE OF THE BOUND GUARD, and its price, measured.

    A decline is not free: an obligation the row will not model is not sliced,
    not emitted and never solver-decided, so it comes back `unknown`. That
    costs REFUTATIONS as well as discharges — at n = 9 a false claim about an
    untouched element used to come back `violated-witness` and now comes back
    `unknown`. The direction is the safe one (an ungauged route can mint a
    missed violation; an `unknown` cannot mint anything), and it is recorded
    in SOUNDNESS.md rather than left as a silent narrowing.

    Read at `escalate`, for the reason `_set_row_records` gives: the VERIFIED
    bar sits downstream and would report UNKNOWN either way.
    """
    def build_at(n):
        def build():
            x = any_array((n,), "float64", (0.0, 1.0))
            u = any_array((), "float64", (2.0, 3.0))
            s = x.at[2].set(u)
            return assert_(s[0] - x[0] >= 1.0)  # FALSE: element 0 is untouched
        return build

    assert _set_row_records(build_at(_SET_MAX_N), {}) == ("violated-witness",), (
        "the gauged bound itself stopped refuting, so the comparison below "
        "measures a broken fixture rather than the guard"
    )
    assert _set_row_records(build_at(_SET_MAX_N + 1), {}) == ("unknown",), (
        "one past the gauge the row must DECLINE, leaving the obligation "
        "undecided. A `violated-witness` here means the ungauged routing ran; "
        "a `discharged` means it ran and was wrong"
    )


def test_every_untouched_element_still_refutes_over_the_SURFACE_SPACE():
    """The same property at the SURFACE, so neither row is pinned only at its
    producer: for every k, and every element the write did NOT touch, the
    false claim `s[j] - x[j] >= 1` must still come back `violated-witness`.
    Any plan that routes element j to the update instead of to the operand
    brings that claim back `discharged` — a missed violation.

    COVERAGE, and its DELIBERATE ASYMMETRY WITH THE PRODUCER SWEEPS. Each case
    here runs a real escalation, so the surface is swept over a SMALLER space
    and says so rather than pretending otherwise: axis lengths 3..6 for the
    SET row (every k of each), and the rank-2 accumulate case that the row
    arithmetic mutation is visible in. The producer sweeps above carry the
    width — every n to 8, every shape to rank 3 — and this carries the
    end-to-end direction. A corruption that survives both is a corruption
    correct on every shape either one reaches.

    n = 6 is IN this range deliberately: the surface sampled {3, 4} and the
    mis-route wrong only at n = 6 was green through it too.

    Read at `escalate` rather than at `check` for the reason
    `_set_row_records` gives: the VERIFIED bar sits downstream and would
    return UNKNOWN either way.
    """
    ns = list(range(3, 7))
    assert 6 in ns, "n = 6 is the axis length the sampled surface walked past"
    for n in ns:
        for k in range(n):
            def build(n=n, k=k):
                x = any_array((n,), "float64", (0.0, 1.0))
                u = any_array((), "float64", (2.0, 3.0))
                s = x.at[k].set(u)
                return tuple(assert_(s[j] - x[j] >= 1.0)
                             for j in range(n) if j != k)

            outcomes = _set_row_records(build, {})
            assert outcomes == ("violated-witness",) * (n - 1), (
                f"n={n} k={k}: untouched elements gave {outcomes}. A "
                f"`discharged` here is the missed-violation direction: the "
                f"row handed an untouched element the update's box"
            )

    # the accumulate row's ROW arithmetic, at the surface, above rank 1
    for k in range(3):
        def build(k=k):
            x = any_array((3, 2), "float64", (0.0, 1.0))
            u = any_array((2,), "float64", (2.0, 3.0))
            s = x.at[k].add(u)
            # FALSE: element (k, 0) receives u[0] alone, which is at most 3
            return assert_(s[k, 0] - x[k, 0] >= 4.0)

        outcomes = _set_row_records(build, {})
        assert outcomes == ("violated-witness",), (
            f"rank-2 accumulate at k={k}: {outcomes}. A `discharged` means "
            f"the row gave element (k, 0) more than the one addend the "
            f"program adds there — the row arithmetic above rank 1, which "
            f"nothing in this file gauged until now"
        )


def gate_set_row_agreement(subject):
    """Gate 7: the static-index scatter SET row routes each output element to
    the right source. Four relational cases whose hand answers depend on
    exactly that: three FALSE claims whose refutation requires the write to
    have landed where it did, and one TRUE claim about an untouched element.

    THE FOURTH WRITES A NONZERO INDEX, and it is not a fourth of the same
    thing. A fixture set that only ever writes k = 0 gauges "did the update
    land where the plan said" but not "is the plan's k the program's k",
    because every mis-route defined relative to k collapses at k = 0. See
    `_set_written_at_a_NONZERO_index`, and the mutation
    `set-plan-writes-one-before-the-index` that only it catches.

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
        if _set_row_records(_set_written_at_a_NONZERO_index, subject) != (
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


# -- gate 8: the SET row's ADMISSION, which routing cannot reach ------------
#
# Three of the four fixtures above write index 0, and the fourth exists
# because that is NOT by itself the right shape for routing. A predecessor of
# this comment said it was — "the three fixtures above all write index 0, that
# is the right shape for ROUTING" — and that was measurably false in its own
# terms: at k = 0 every route defined as an offset from k lands back on 0 or
# out of range, so the corruption `i == k` -> `i == (k - 1 if k > 0 else k)`
# is invisible to all three while turning a `violated-witness` into a
# `discharged` at k = 2. The fourth fixture writes k = 2 for exactly that.
#
# What the index-0 fixtures genuinely do not drive is ADMISSION, which is the
# question of whether the row may model this equation at all — a different
# question from where the update lands, and the one this gate is for.
# Measured consequence: mutating `_scatter_set_plan`'s out-of-range decline
# into a CLIP-style clamp (the docstring calls that decline "the soundness
# check, not a tidiness one") is a real unsoundness — jax DROPS
# `x.at[7].set(2.0)` on a length-3 array under FILL_OR_DROP, so `s[2] - x[2]
# >= 1.0` is FALSE, yet the clamped plan brings the obligation back
# `discharged` — and the full suite stayed green (1995 passed) with only the
# scatter VERIFIED bar holding the verdict at UNKNOWN. A bar is not a gauge.
#
# So admission gets its own gate, over the declines that carry a soundness
# argument in their own text.
#
# WHICH DECLINES THOSE ARE IS COUNTED, NOT ESTIMATED. `_scatter_set_plan` has
# ELEVEN `raise _Decline` sites, and an admission gate is only as good as the
# fraction of them it drives — an unstated fraction is exactly the defect this
# file's subject is about. The census below is the accounting, in source
# order, and `test_the_admission_gate_accounts_for_every_decline_site` reads
# the site count out of the source so the two cannot drift.
#
#   #   rule                                    driven here?
#   1   arity: 3 operands, 1 output             no — malformed IR
#   2   out_shape != operand_shape              no — malformed IR
#   3   carries a combiner (update_jaxpr)       no — census fixture only
#   4   index dtype cannot cover the axis       YES  (dtype-coverage below)
#   5   outside the measured row form           no — census fixture only
#   6   mode is not FILL_OR_DROP                YES  (two fixtures, one rule)
#   7   index not statically derivable          YES  (derivability below)
#   8   indices decode to != 1 element          no — aval/value mismatch
#   9   index value is not an integer           no — defensive, after #4
#   10  index out of range for the axis         YES  (out-of-range below)
#   11  axis longer than the GAUGE              YES  (past-the-gauge below)
#
# So FIVE of the eleven rules are driven, by SIX fixtures — `clip` and
# `promise_in_bounds` are two spellings of rule #6 and expect the same quoted
# substring. A predecessor of this table said "the admission gate measures
# three rules"; that was three FIXTURES over two rules, and it named three of
# the eight undriven sites while leaving five unnamed.
#
# RULE #11 IS THE BOUND GUARD, and it is counted here rather than hidden
# behind a helper on purpose. The obvious way to add a decline without moving
# this number is to raise it from a called function — which is exactly the
# "local alias walks straight past" drift the count test's own docstring
# warns about, one level up. The number moved; the table and both assertions
# moved with it. Its own two-directional pin is
# `test_nothing_outside_the_swept_space_reaches_either_row`; the fixture below
# is what makes the ADMISSION GATE drive it, so a mutation that deletes the
# bound is caught by the gauge screen as well as by that test.
#
# The undriven six, each with its reason:
#   * #1 and #2 are malformed-IR guards. `jax.make_jaxpr` does not emit a
#     `scatter` with other arity or an output shape contradicting its operand,
#     so no traced fixture reaches them and a mutation relaxing them would be
#     inert rather than uncaught.
#   * #3 and #5 ARE driven by traced fixtures, just not here:
#     `tests/test_scatter_emission_reach.py` pins `x.at[k].apply(f)` against
#     "carries a combiner" (#3) and the multi-axis, slice-index and
#     hand-written-window forms against "outside the measured" (#5), each by
#     its quoted reason. What that census does NOT do is mutate the rule and
#     watch something fail, so those two are pinned but not gauged.
#   * #8 needs a `scatter` whose indices aval says one element and whose
#     decoded value says another; #9 needs a non-integral value in an index
#     operand rule #4 has already required to be an exactly-covering integer
#     dtype. Both are defensive, and neither has a fixture anywhere in this
#     repo. They are named so a reader knows they are unmeasured.
#
# Rule #7 is here because it was MEASURED to be a live unsoundness, not
# because it completes a table: corrupting it to guess index 0 makes
# `x.at[jnp.int32(2)].set(2.0)` model a write to element 0 while jax writes
# element 2, and the whole suite stayed green except
# `test_supported_primitives_doc` — a line-citation check on a generated page,
# which fires on the edit's line count and not on the plan starting to lie.

_ADMISSION_DECLINES = (
    ("out-of-range static index", lambda: _set_query(7, None),
     "is out of range"),
    ("mode='clip'", lambda: _set_query(0, "clip"), "is not FILL_OR_DROP"),
    ("mode='promise_in_bounds'", lambda: _set_query(0, "promise_in_bounds"),
     "is not FILL_OR_DROP"),
    ("index not statically derivable", lambda: _traced_index_set_query(),
     "not statically derivable"),
    ("index dtype cannot cover the axis", lambda: _int8_index_set_query(),
     "cannot exactly represent"),
    ("axis longer than the gauged space", lambda: _past_the_gauge_set_query(),
     "outside the GAUGED"),
)


def _set_query(index, mode):
    """`x.at[index].set(2.0)` posed relationally against an untouched
    element, so the shape is the one the routing fixtures use and only
    admission differs."""

    def build():
        x = any_array((3,), "float64", (0.0, 1.0))
        s = (x.at[index].set(2.0) if mode is None
             else x.at[index].set(2.0, mode=mode))
        return assert_(s[2] - x[2] >= 1.0)

    return build


def _traced_index_set_query():
    """RULE #7's fixture: the same `.set` with the index spelled as a jax
    scalar rather than a Python int.

    `x.at[2]` constant-folds to a literal index column; `x.at[jnp.int32(2)]`
    does not — jax emits its index-normalisation chain (`lt`, `add`,
    `select_n`, `broadcast_in_dim`) and `_exact_static_elements` returns None
    through it. Measured on jax 0.11.0. The write itself is in range and
    perfectly ordinary, which is the point: the row declines because it cannot
    SEE the index, not because the program is unusual, and a corruption that
    guesses one gets a wrong element rather than an obvious failure."""

    def build():
        x = any_array((3,), "float64", (0.0, 1.0))
        s = x.at[jnp.int32(2)].set(2.0)
        return assert_(s[2] - x[2] >= 1.0)

    return build


def _past_the_gauge_set_query():
    """RULE #11's fixture: an ordinary in-range `.set` on an operand ONE
    LONGER than the route sweep exhausts.

    Nothing about the write is unusual — index 2 of a length-9 array, static,
    in range, FILL_OR_DROP — which is the point: the row declines because the
    routing at that length is gauged by nothing, not because the program is
    odd. A line-neutral mis-route wrong only at n = 9 was measured green
    through the whole suite in both columns on `e35de13`."""

    def build():
        x = any_array((_SET_MAX_N + 1,), "float64", (0.0, 1.0))
        s = x.at[2].set(2.0)
        return assert_(s[0] - x[0] >= 1.0)

    return build


_INT8_DNUMS = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(), inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,))


def _int8_index_set_query():
    """RULE #4's fixture, hand-written because the sugar cannot express it.

    `x.at[jnp.int8(3)]` raises inside jax's own index normalisation on a
    length-200 operand (`Python integer 200 out of bounds for int8`), so the
    dtype-coverage rule is only reachable through `lax.scatter` directly. The
    index is 3 and in range; what the rule objects to is that XLA computes the
    out-of-bounds bound in the INDEX element type, so an in-range-looking
    update at a position int8 cannot represent is silently DROPPED."""

    def build():
        x = any_array((200,), "float64", (0.0, 1.0))
        s = jax.lax.scatter(
            x, jnp.asarray(np.array([3], dtype=np.int8)), jnp.asarray(2.0),
            _INT8_DNUMS, mode=jax.lax.GatherScatterMode.FILL_OR_DROP)
        return assert_(s[1] - x[1] >= 1.0)

    return build


def _set_plan_verdict(build, subject):
    """The plan's own answer for one `.set` equation: the quoted decline, or
    the routes it admitted. Read at `_scatter_set_plan` rather than at a
    verdict on purpose — the VERIFIED bar sits downstream of this row and
    would mask an admission change behind an UNKNOWN it was going to return
    anyway."""
    with _patched(subject):
        closed = trace(build)
        eqn = next(e for e in closed.jaxpr.eqns if str(e.primitive) == "scatter")
        consts = dict(zip((v.id for v in closed.jaxpr.constvars), closed.consts))
        try:
            return ("admitted", OB._scatter_set_plan(closed.jaxpr.eqns, consts, eqn))
        except OB._Decline as d:
            return ("declined", str(d))


def gate_set_row_admission(subject):
    """Gate 8: the static-index scatter SET row admits only what it models.

    Each decline below is a soundness argument in the row's own docstring, and
    each is checked BY ITS REASON, not merely by declining: an equation that
    stops short for an unrelated reason is not evidence the rule is there.
    The in-range FILL_OR_DROP case must still be admitted, or a gate that
    declined everything would read as perfect. The block comment above says
    which of the row's eleven decline sites this reaches and which it
    does not.
    """
    try:
        kind, payload = _set_plan_verdict(_set_query(0, None), subject)
        if kind != "admitted" or payload != [(2, 0), (0, 1), (0, 2)]:
            return False  # the covered form must still route, and route right
        for _label, build, expect in _ADMISSION_DECLINES:
            kind, payload = _set_plan_verdict(build(), subject)
            if kind != "declined" or expect not in payload:
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
    "set-row-admission": gate_set_row_admission,
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


def _set_plan_write_one_before_the_index(eqns, consts, eqn):
    """WRITES ONE ELEMENT BEFORE THE INDEX, AND WRITES k ITSELF WHEN k IS 0 —
    the mutation the index-0 routing fixtures cannot see.

    The plan-layer form of the line-neutral source corruption
    ``[(2, 0) if i == k else (0, i) ...]`` ->
    ``[(2, 0) if i == (k - 1 if k > 0 else k) else (0, i) ...]``. It differs
    from `_set_plan_off_by_one` in exactly the way that matters here: that one
    moves the write for EVERY k, including 0, so the existing fixtures catch
    it; this one is the identity at k = 0 and wrong everywhere else."""
    routes = _REAL_SET_PLAN(eqns, consts, eqn)
    n = len(routes)
    written = [i for i, (op, _src) in enumerate(routes) if op == 2]
    if not written:
        return routes
    k = written[0]
    k = k - 1 if k > 0 else k
    return [(2, 0) if i == k else (0, i) for i in range(n)]


def _set_plan_drop_write(eqns, consts, eqn):
    """WRITES NOTHING: every output element aliases the operand, i.e. the
    result of a DROPPED out-of-range update applied to an in-range one."""
    return [(0, i) for i in range(len(_REAL_SET_PLAN(eqns, consts, eqn)))]


def _set_plan_write_all(eqns, consts, eqn):
    """WRITES EVERY POSITION: the one scalar update is broadcast over the
    whole output — `x.at[:].set(v)` where the program wrote one element."""
    return [(2, 0) for _ in _REAL_SET_PLAN(eqns, consts, eqn)]


def _clamped_routes(eqns, consts, eqn):
    """The routes a CLIP-style admission would produce for an out-of-range
    static index: the index pulled onto the nearest in-range position."""
    operand_shape = OB._shape_of(eqn.invars[0])
    k = int(OB._exact_static_elements(eqns, consts, eqn.invars[1])[0])
    k = min(max(k, 0), operand_shape[0] - 1)
    n = OB._size(operand_shape)
    return [(2, 0) if i == k else (0, i) for i in range(n)]


def _set_plan_clamp_out_of_range(eqns, consts, eqn):
    """ADMITS AN OUT-OF-RANGE STATIC INDEX BY CLAMPING IT — the one mutation
    here that is unsound rather than merely wider, and the one the routing
    gate cannot see.

    Under FILL_OR_DROP jax DROPS the write (`jnp.array([0.,.5,1.]).at[7]
    .set(2.0)` is measured unchanged on jax 0.11.0), so the result is the
    operand and `s[2] - x[2] >= 1.0` is FALSE. Clamping models `mode='clip'`
    instead, which writes at position 2, and the obligation comes back
    `discharged`: a MISSED violation, the direction the bar exists for.
    Everything else the real plan declines is inherited."""
    try:
        return _REAL_SET_PLAN(eqns, consts, eqn)
    except OB._Decline as d:
        if "is out of range" not in str(d):
            raise
        return _clamped_routes(eqns, consts, eqn)


def _set_plan_guess_a_missing_index(eqns, consts, eqn):
    """GUESSES INDEX 0 WHEN THE INDEX IS NOT STATICALLY DERIVABLE — rule #7,
    and the second measured unsoundness in this row after the clamp.

    `x.at[jnp.int32(2)].set(2.0)` normalises through `lt`/`add`/`select_n`,
    which `_exact_static_elements` cannot fold, so the real plan declines.
    Guessing 0 instead models `out[0] = update` where jax writes `out[2]`:
    posed as `s[0] - x[0] >= 1.0` the program VIOLATES (element 0 is
    untouched, so the difference is 0) while the guessed plan substitutes 2.0
    and returns `discharged`. A MISSED violation, the direction the bar exists
    for, and neither routing gate can see it — every routing fixture uses a
    Python-int index, which folds and never reaches this branch."""
    try:
        return _REAL_SET_PLAN(eqns, consts, eqn)
    except OB._Decline as d:
        if "not statically derivable" not in str(d):
            raise
        n = OB._size(OB._shape_of(eqn.invars[0]))
        return [(2, 0) if i == 0 else (0, i) for i in range(n)]


def _set_plan_ignore_index_dtype(eqns, consts, eqn):
    """ADMITS AN INDEX DTYPE THAT CANNOT COVER THE OPERAND'S LEADING AXIS —
    rule #4, whose decline text carries its own soundness argument (XLA
    computes the out-of-bounds bound in the INDEX element type, so updates at
    positions the dtype cannot represent are silently DROPPED).

    Not unsound at the gauged index itself — 3 is representable in int8 — for
    the same reason `_set_plan_admit_any_mode` is not: the case where the rule
    matters needs a second condition the fixture does not carry. That is the
    point of gauging it here rather than trusting a downstream check, since
    there is no downstream check; the rule's only defence would be another
    rule."""
    try:
        return _REAL_SET_PLAN(eqns, consts, eqn)
    except OB._Decline as d:
        if "cannot exactly represent" not in str(d):
            raise
        return _clamped_routes(eqns, consts, eqn)


def _set_plan_admit_any_mode(eqns, consts, eqn):
    """ADMITS `mode='clip'` AND `mode='promise_in_bounds'` as if they were
    FILL_OR_DROP. Not unsound on its own — with an in-range index all three
    modes write the same element, and the range check still stops the case
    where CLIP and DROP diverge — so nothing downstream of the row can catch
    it, and that is the point: an admission rule whose only defence is a
    second admission rule has no gauge of its own."""
    try:
        return _REAL_SET_PLAN(eqns, consts, eqn)
    except OB._Decline as d:
        if "is not FILL_OR_DROP" not in str(d):
            raise
        return _clamped_routes(eqns, consts, eqn)


MUTATIONS = {
    **MUTATIONS,
    "set-plan-off-by-one-position": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_set_plan", _set_plan_off_by_one),
            (SM, "_scatter_set_plan", _set_plan_off_by_one),
        ),
    },
    "set-plan-writes-one-before-the-index": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_set_plan", _set_plan_write_one_before_the_index),
            (SM, "_scatter_set_plan", _set_plan_write_one_before_the_index),
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
    "set-plan-clamps-an-out-of-range-index": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_set_plan", _set_plan_clamp_out_of_range),
            (SM, "_scatter_set_plan", _set_plan_clamp_out_of_range),
        ),
    },
    "set-plan-admits-any-scatter-mode": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_set_plan", _set_plan_admit_any_mode),
            (SM, "_scatter_set_plan", _set_plan_admit_any_mode),
        ),
    },
    "set-plan-guesses-an-underivable-index": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_set_plan", _set_plan_guess_a_missing_index),
            (SM, "_scatter_set_plan", _set_plan_guess_a_missing_index),
        ),
    },
    "set-plan-ignores-the-index-dtype-bound": {
        **BASELINE,
        "__patches__": (
            (OB, "_scatter_set_plan", _set_plan_ignore_index_dtype),
            (SM, "_scatter_set_plan", _set_plan_ignore_index_dtype),
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
               "budget. ALSO the static-index scatter SET row "
               "(_scatter_set_plan) in BOTH of its halves, which are "
               "separately gauged because they need different fixtures: "
               "ROUTING — where the update lands — across its three "
               "consumers (slice validation, emission and replay) through "
               "FOUR relational in-range cases the interval transfer cannot "
               "settle, three writing index 0 and one writing index 2 "
               "(index 0 alone cannot see a mis-route defined relative to "
               "the written index, because every offset from 0 collapses "
               "onto 0 or out of range); and ADMISSION — whether the row "
               "may model the equation at all. `_scatter_set_plan` has ELEVEN `raise "
               "_Decline` sites and the admission gate drives FIVE of them, "
               "with six fixtures (two spellings of the mode rule), each "
               "checked by its quoted reason: the index dtype's coverage of "
               "the leading axis, the non-FILL_OR_DROP modes, an index not "
               "statically derivable, an out-of-range static index, and an "
               "axis longer than the gauged route space. The "
               "other six are NOT driven here: arity and the "
               "operand/output shape contradiction are malformed-IR guards "
               "no traced fixture reaches; the combiner and row-form "
               "declines are pinned by traced fixtures in "
               "tests/test_scatter_emission_reach.py but are not mutated "
               "anywhere, so they are pinned and not gauged; and the "
               "indices-decode-to-more-than-one and non-integral-index "
               "declines are defensive, with no fixture in this repo. Also "
               "does NOT drive the SET row's interval transfer (no SET "
               "mutation here is transfer-side), and does not drive any "
               "other primitive's rows."),
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
    # than an accumulate gate that happened to notice. Asserted as the EXACT
    # catcher set, not as membership: the admission gate opens by requiring
    # the covered in-range form to come back `admitted` AND routed to
    # [(2,0),(0,1),(0,2)], and deleting that second half is invisible to a
    # membership assertion — the routing gate still catches these three, so
    # the screen stays green while the admission gate quietly stops checking
    # that the form it admits is the form the program wrote.
    for name in ("set-plan-off-by-one-position", "set-plan-drops-the-write",
                 "set-plan-writes-every-position"):
        assert caught[name] == ("set-row-agreement", "set-row-admission"), (
            f"{name} is caught by {caught[name]}. The admission gate's own "
            f"anti-vacuity clause — the covered form must still route, and "
            f"route RIGHT — is what makes it see a mis-routed plan; if it no "
            f"longer does, that clause is gone"
        )
    # ... and the one that is invisible at k = 0, which ONLY the nonzero-index
    # routing fixture can see. Exact, for the same reason: if the admission
    # gate starts catching it the fixture has stopped being what measures it.
    assert caught["set-plan-writes-one-before-the-index"] == (
        "set-row-agreement",
    ), caught["set-plan-writes-one-before-the-index"]
    # ADMISSION is gauged separately from routing, and the split is not
    # cosmetic: the routing fixtures all write the in-range index 0, so
    # every admission rule was outside what they drive. The clamp mutation
    # is the measured unsoundness — the full suite stays green under it —
    # and it must be the ADMISSION gate that catches it, not an accident.
    for name in ("set-plan-clamps-an-out-of-range-index",
                 "set-plan-admits-any-scatter-mode",
                 "set-plan-guesses-an-underivable-index",
                 "set-plan-ignores-the-index-dtype-bound"):
        assert caught[name] == ("set-row-admission",), (
            f"{name} is caught by {caught[name]} — if the routing gates now "
            f"see it, the admission gate is no longer the thing measuring "
            f"admission and the scope sentence above is wrong"
        )


def test_the_admission_gate_accounts_for_every_decline_site():
    """THE SCOPE SENTENCE'S OWN NUMBER, READ OUT OF THE SOURCE.

    "the admission gate drives FIVE of ELEVEN decline sites" is the kind of
    claim this file exists to distrust: it was true when written and nothing
    made it stay true. A `raise _Decline` added to `_scatter_set_plan` moves
    the denominator silently, and a reader has no way to notice.

    So the denominator is counted from the function's own source, and the
    numerator from `_ADMISSION_DECLINES` (five fixtures over four rules —
    `clip` and `promise_in_bounds` are two spellings of the mode rule, which
    is why fixtures are not rules and the two numbers are stated separately).

    COUNTED FROM THE PARSE TREE, NOT FROM THE SPELLING. The first version of
    this test counted `re.findall(r"raise _Decline\\(")` over the source — a
    denominator that a local alias walks straight past (`_D = _Decline;
    raise _D(...)` adds a site the regex does not see, which is the same
    silent-drift defect one level up). What is counted here is `ast.Raise`
    nodes in the function body, which no spelling of the exception can hide
    from. That count is only the DECLINE count if every raise in this function
    is a decline, so that is checked too, by resolving each raised name in the
    module — and a name it cannot resolve to `_Decline` fails LOUDLY rather
    than being counted or skipped.
    """
    import ast
    import inspect
    import textwrap

    src = inspect.getsource(OB._scatter_set_plan)
    tree = ast.parse(textwrap.dedent(src))
    raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    for node in raises:
        exc = node.exc
        name = None
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            name = exc.func.id
        elif isinstance(exc, ast.Name):
            name = exc.id
        assert name is not None and getattr(OB, name, None) is OB._Decline, (
            f"`_scatter_set_plan` raises something this count cannot resolve "
            f"to `_Decline` (line {node.lineno} of the function, raising "
            f"{name!r}). Either it is not a decline — in which case the "
            f"denominator below is no longer the decline-site count — or it "
            f"is spelled through a binding this test cannot follow. Resolve "
            f"it deliberately; do not let the count drift"
        )
    sites = len(raises)
    assert sites == 11, (
        f"`_scatter_set_plan` now has {sites} decline sites, not 11. The "
        f"admission gate's scope sentence in test_gauge_catches_every_"
        f"mutation and the census comment above _ADMISSION_DECLINES both "
        f"quote 11 — re-derive which of the new set the gate drives, extend "
        f"it or name the gap, and update both numbers."
    )
    reasons = {expect for _label, _build, expect in _ADMISSION_DECLINES}
    assert len(_ADMISSION_DECLINES) == 6 and len(reasons) == 5, (
        f"the admission gate now runs {len(_ADMISSION_DECLINES)} fixture(s) "
        f"over {len(reasons)} distinct decline reason(s); the scope sentence "
        f"says six over five"
    )
