# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""CRLF — a corpus that ESCALATES, built for one job: show, PER OBLIGATION,
that moving the cvc5 wheel transport off `text=True` moves no verdict.

WHY IT IS NOT `scratchpad/pin/corpus_pin.py`. That is the per-obligation
instrument this repository reaches for, and it says so in its own docstring:
it "has no solver escalation (no network, no budget) so nothing here scores
`solvers.py`". This branch changes `solvers.py` and nothing else, so the
existing instrument is structurally blind to it — every row would be
byte-identical for a reason that has nothing to do with the repair. A zero
from a blind instrument is the shape this project has been wrong about three
times.

WHAT THIS ONE DOES INSTEAD. Every row is a query whose obligations interval
propagation leaves UNKNOWN, so every row really reaches `escalate` and really
spawns a solver. Each row is driven under three portfolios —

    cvc5-only   the transport under repair, on its own
    z3-only     the leg that shares no code with it, as a control on the
                harness rather than on the change
    full        the portfolio a caller actually gets, where cvc5 answers
                first and z3 corroborates

— and the whole per-obligation record is written out: outcome, detail, notes,
`answered_by`, the witness values in exact rationals, every invocation stamp
(name, version, transport, and the full emitted option set including the
`smt2_sha256` of the script that was sent), and the assembled verdict's
status and solver stamp. `diff` compares them key for key.

BOTH DIRECTIONS ARE IN THE GRID, deliberately. The defect being repaired
reaches DISCHARGE as well as refutation — a stale child's `answer unsat` was
returned as `unsat`, which is the VERIFIED path — so a corpus of refutations
alone could not have scored it. Rows come in pairs that differ only in a
declared bound: one whose obligation is unsat over the box (VERIFIED) and one
whose obligation has a genuine counterexample in it (REFUTED, replay
confirmed), plus rows that stay UNKNOWN.

ANTI-VACUITY. `run` refuses to write a file in which no solver was invoked,
no obligation was discharged, or no obligation was refuted. A corpus that
escalated nothing would otherwise produce a beautiful zero.

WHAT THIS CORPUS CANNOT SEE. It drives the cvc5 WHEEL transport, because
that is what is installed here; the binary transport is scored structurally
instead (`tests/test_solver_audit_findings.py`) and is textually unchanged.
It runs `semantics="real"` only, since `escalate` declines ieee wholly. Its
solver budget is fixed, so it says nothing about timeout behaviour. And it is
built to REACH escalation, so its ratios are not rates in any population.

Usage, from a worktree, with `JAX_PLATFORMS=cpu`:

    PYTHONPATH=<tree>/src python scratchpad/crlf/corpus_solver.py run OUT.json
    python scratchpad/crlf/corpus_solver.py diff A.json B.json

`run` needs jax + stelling + a solver and stamps the resolved
`stelling.__file__`, `jax.__version__` and both solver versions into its
output; `diff` needs none of them.
"""
from __future__ import annotations

import json
import re
import sys

TIMEOUT_MS = 30_000

# (name, only) — `None` is the full portfolio.
PORTFOLIOS = (("cvc5", ("cvc5",)), ("z3", ("z3",)), ("full", None))


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------


def _psd(b_lo, b_hi):
    """tr² <= det·(81/8) over a 2x2 symmetric matrix's entries.

    Nonlinear in three variables, so interval propagation cannot decide it
    and every row escalates. `b` in [-0.5, 0.5] makes it true over the box
    (unsat, VERIFIED); widening `b` puts a real counterexample in the box
    (sat, REFUTED with a replayable witness). 81/8 is exact in binary64, so
    the constant is not doing any rounding of its own.
    """
    def h():
        from stelling.harness import any_array, assert_, nonvacuity

        a = any_array((), "float64", (1.0, 2.0))
        c = any_array((), "float64", (1.0, 2.0))
        b = any_array((), "float64", (b_lo, b_hi))
        tr = a + c
        det = a * c - b * b
        o = assert_(tr * tr <= det * 10.125)
        ap = any_array((), "float64", (1.5, 1.5))
        bp = any_array((), "float64", (0.0, 0.0))
        cp = any_array((), "float64", (1.5, 1.5))
        n = [nonvacuity(ap >= 1.0), nonvacuity(ap <= 2.0),
             nonvacuity(bp >= -0.5), nonvacuity(bp <= 0.5),
             nonvacuity(cp >= 1.0), nonvacuity(cp <= 2.0)]
        return (o, *n)

    return h


def _expanded_square(n, c):
    """`(x-1)² >= c`, written EXPANDED so the interval leg cannot see the
    square: `x*x - 2*x + 1` over `x` in [0, 2] gives [0,4] + [-4,0] + 1 =
    [-3, 5], which decides nothing. The real answer is `c <= 0` — so `c = 0`
    is unsat over the box (DISCHARGED, the VERIFIED path) and any `c > 0` has
    a counterexample at `x = 1` (REFUTED, replay confirmed).

    The pair differs in ONE CONSTANT, which is what makes it a pair: the
    emitted script, the declared box and the number of declarations are
    otherwise identical, so a difference between the two rows is the answer
    and not the shape. `n` scales the number of `(declare-const … Real)`
    lines `smt.py` emits, and with it the length of the driver's stdout.
    """
    def h():
        from stelling.harness import any_array, assert_

        shape = () if n is None else (n,)
        x = any_array(shape, "float64", (0.0, 2.0))
        return assert_(x * x - 2.0 * x + 1.0 >= float(c))

    return h


def _product_bound(k):
    """x·y <= k over x, y in [0, 3]. The interval leg gives [0, 9], so this
    escalates for every 0 <= k < 9 and the solver refutes it."""
    def h():
        from stelling.harness import any_array, assert_

        x = any_array((), "float64", (0.0, 3.0))
        y = any_array((), "float64", (0.0, 3.0))
        return assert_(x * y <= float(k))

    return h


def _cubic(lo, hi, thr):
    """x³ - x >= thr over a box. Nonlinear, one obligation, and the threshold
    is chosen so the interval leg decides NEITHER row: on [-0.5, 0.5] the
    interval bound is -0.625 while the true minimum is -0.375, so `thr =
    -0.5` sits between them and only a solver can close it (DISCHARGED); on
    [-3, 3] the true minimum is -24 and the obligation genuinely fails
    (REFUTED)."""
    def h():
        from stelling.harness import any_array, assert_

        x = any_array((), "float64", (lo, hi))
        return assert_(x * x * x - x >= float(thr))

    return h


def _two_obligations(c):
    """TWO obligations in ONE query, so a per-obligation record has more than
    one entry to be per-obligation ABOUT — and so a change that collapsed a
    query's obligations together would show as a shape change and not only as
    a status change. Both legs escalate; both are the expanded square."""
    def h():
        from stelling.harness import any_array, assert_

        x = any_array((), "float64", (0.0, 2.0))
        y = any_array((), "float64", (0.0, 2.0))
        return (assert_(x * x - 2.0 * x + 1.0 >= float(c)),
                assert_(y * y - 4.0 * y + 4.0 >= float(c)))

    return h


def cases():
    """(name, harness_thunk, expectation) for every row.

    `expectation` is what this row exists to reach — it is asserted in `run`
    so the corpus cannot quietly stop escalating and still report a zero.
    """
    return [
        # unsat over the box: the DISCHARGE direction, which is the one the
        # repaired defect could forge and the one nothing downstream replays
        ("psd_true", _psd(-0.5, 0.5), "discharged"),
        ("cubic_true", _cubic(-0.5, 0.5, -0.5), "discharged"),
        ("square_scalar_true", _expanded_square(None, 0.0), "discharged"),
        ("square_vec4_true", _expanded_square(4, 0.0), "discharged"),
        ("square_vec16_true", _expanded_square(16, 0.0), "discharged"),
        ("square_vec64_true", _expanded_square(64, 0.0), "discharged"),
        ("two_obligations_true", _two_obligations(0.0), "discharged"),
        # sat with a counterexample in the box: REFUTED, replay confirmed
        ("psd_false", _psd(-1.4, 1.4), "violated"),
        ("cubic_false", _cubic(-3.0, 3.0, -1.0), "violated"),
        ("product_bound_4", _product_bound(4), "violated"),
        ("square_scalar_false", _expanded_square(None, 0.5), "violated"),
        ("square_vec4_false", _expanded_square(4, 0.5), "violated"),
        ("square_vec16_false", _expanded_square(16, 0.5), "violated"),
        ("two_obligations_false", _two_obligations(0.5), "violated"),
    ]


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def _stamp(s):
    return {
        "invoked": s.invoked,
        "reason": s.reason,
        "name": s.name,
        "version": s.version,
        "transport": s.transport,
        "options": [list(o) for o in (s.options or ())],
    }


def _witness(w):
    if w is None:
        return None
    return {
        "values": {k: str(v) for k, v in sorted(dict(w.values).items())},
        "detail": getattr(w, "detail", ""),
        "confirmed": getattr(w, "confirmed", None),
    }


def _record(closed, prop, only):
    from stelling.solvers import SolverConfig, escalate, make_solver_verdict

    cfg = SolverConfig(timeout_ms=TIMEOUT_MS, only=only)
    try:
        esc = escalate(closed, prop, cfg)
    except Exception as e:  # noqa: BLE001 — a refusal is an outcome
        return {"escalate": f"RAISED {type(e).__name__}: {e}"}
    out = {
        "escalation_notes": list(esc.notes),
        "semantics": esc.semantics,
        "spawns": esc.ledger.spawns,
        "n_ledger_stamps": len(esc.ledger.stamps),
        "records": [
            {
                "index": r.index,
                "outcome": r.outcome,
                "detail": r.detail,
                "notes": list(r.notes),
                "answered_by": list(r.answered_by),
                "witness": _witness(r.witness),
                "invocations": [_stamp(s) for s in r.invocations],
            }
            for r in esc.records
        ],
    }
    try:
        v = make_solver_verdict(
            closed, prop, esc,
            stelling_version="corpus",
            jax_version="corpus",
            precision_config="jax_enable_x64=True",
        )
        solver = v.stamp.solver
        out["verdict"] = {
            "status": v.status,
            "nonvacuity": v.stamp.nonvacuity,
            "solver": (
                [_stamp(s) for s in solver]
                if isinstance(solver, tuple) else str(solver)
            ),
            "obligations": [
                {"index": o.index, "status": o.status}
                for o in getattr(v, "obligations", ())
            ],
        }
    except Exception as e:  # noqa: BLE001
        out["verdict"] = f"RAISED {type(e).__name__}: {e}"
    return out


def run(path):
    import jax

    jax.config.update("jax_enable_x64", True)
    import stelling
    from stelling import _optional
    from stelling.harness import trace
    from stelling.propagate import propagate
    from stelling.solvers import _cvc5_wheel_version, _z3_version

    results = {
        "_stelling_file": stelling.__file__,
        "_jax_version": jax.__version__,
        "_cvc5_version": _cvc5_wheel_version(),
        "_z3_version": _z3_version(),
        "_cvc5_wheel": _optional.available("cvc5"),
        "rows": {},
    }
    spawns = discharged = violated = escalated = 0
    for name, h, expect in cases():
        row = {}
        closed = trace(h)
        prop = propagate(closed)
        row["_interval_statuses"] = [o.status for o in prop.obligations]
        for tag, only in PORTFOLIOS:
            rec = _record(closed, prop, only)
            row[tag] = rec
            if isinstance(rec, dict) and "records" in rec:
                spawns += rec["spawns"]
                escalated += len(rec["records"])
                for r in rec["records"]:
                    discharged += r["outcome"] == "discharged"
                    violated += r["outcome"] == "violated-witness"
        # THE ROW REACHED WHAT IT EXISTS TO REACH — across the union of the
        # three portfolios, not under one of them. A portfolio can legitimately
        # fail to produce records (`escalate` raises `SolverDisagreement` when
        # its members contradict each other), and a corpus that crashed there
        # could not be run against a mutant — which is exactly what a positive
        # control needs it to do. The anti-vacuity floor below is the part that
        # cannot be satisfied by an empty run.
        got = [
            r["outcome"]
            for tag, _ in PORTFOLIOS
            for r in row[tag].get("records", ())
        ]
        assert any(expect in g for g in got), (name, expect, got)
        results["rows"][name] = row

    # ANTI-VACUITY. A corpus that escalated nothing would report a perfect
    # zero on every comparison and score nothing at all.
    assert spawns > 0, "no solver was ever spawned"
    assert escalated > 0, "no obligation was ever escalated"
    assert discharged > 0, "no obligation was ever DISCHARGED (unsat leg dead)"
    assert violated > 0, "no obligation was ever REFUTED (sat leg dead)"

    results["_totals"] = {
        "spawns": spawns,
        "escalated_obligation_records": escalated,
        "discharged": discharged,
        "violated_witness": violated,
    }
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1, sort_keys=True, default=str)
    print(f"{len(results['rows'])} rows -> {path}")
    print(f"  stelling: {results['_stelling_file']}")
    print(f"  jax:      {results['_jax_version']}")
    print(f"  cvc5:     {results['_cvc5_version']}   z3: {results['_z3_version']}")
    print(f"  totals:   {results['_totals']}")


# ---------------------------------------------------------------------------
# the diff — PER OBLIGATION, not per query
# ---------------------------------------------------------------------------


# `assert #0: cvc5 (wheel) answered sat in 76ms` — the DURATION and nothing
# else. The answer beside it is verdict-bearing and is deliberately left in
# place; only the digits go. This is the narrowest erasure that makes the
# comparison a comparison: measured, 41 of the 41 remaining differences on
# the first run of this pair were this and nothing else, and a wall clock is
# not a property of a tree.
_MS = re.compile(r"\b\d+ms\b")


def _normalise(v, a_root, b_root):
    """Two trees are two DIRECTORIES, and a `source_info` in a detail carries
    the one it ran in. Erasing the two roots is the difference between "no
    verdict moved" and a page of path noise; erasing anything else would be
    erasing the finding — so exactly two things are erased, the tree roots
    and the millisecond durations, and both are named."""
    if not isinstance(v, str):
        return v
    for root in (a_root, b_root):
        if root:
            v = v.replace(root, "<TREE>")
    return _MS.sub("<ms>ms", v)


def _flatten(blob, prefix, out):
    if isinstance(blob, dict):
        for k, v in sorted(blob.items()):
            _flatten(v, f"{prefix}.{k}", out)
    elif isinstance(blob, list):
        for i, v in enumerate(blob):
            _flatten(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = blob


# A leaf whose key ends in one of these decides, or reports, a verdict.
_VERDICT_SUFFIXES = (".outcome", ".status", ".answered_by", ".declined")


def diff(a_path, b_path):
    with open(a_path) as fh:
        a = json.load(fh)
    with open(b_path) as fh:
        b = json.load(fh)
    print(f"A: {a['_stelling_file']}  jax {a['_jax_version']}  "
          f"cvc5 {a['_cvc5_version']}  z3 {a['_z3_version']}")
    print(f"B: {b['_stelling_file']}  jax {b['_jax_version']}  "
          f"cvc5 {b['_cvc5_version']}  z3 {b['_z3_version']}")
    print(f"A totals: {a.get('_totals')}")
    print(f"B totals: {b.get('_totals')}")
    a_root = a["_stelling_file"].split("/src/stelling/")[0]
    b_root = b["_stelling_file"].split("/src/stelling/")[0]
    fa, fb = {}, {}
    _flatten(a["rows"], "", fa)
    _flatten(b["rows"], "", fb)
    fa = {k: _normalise(v, a_root, b_root) for k, v in fa.items()}
    fb = {k: _normalise(v, a_root, b_root) for k, v in fb.items()}
    keys = sorted(set(fa) | set(fb))
    verdict_keys = [
        k for k in keys
        if any(s in k for s in _VERDICT_SUFFIXES) or ".witness." in k
    ]
    moved = [k for k in verdict_keys if fa.get(k) != fb.get(k)]
    other = [k for k in keys if k not in set(verdict_keys) and fa.get(k) != fb.get(k)]
    print(f"leaf keys compared: {len(keys)}")
    print(f"  VERDICT-BEARING keys (outcome/status/answered_by/"
          f"declined/witness): {len(verdict_keys)}")
    print(f"  verdict-bearing keys that MOVED:  {len(moved)}")
    for k in moved[:80]:
        print(f"    {k}: {fa.get(k)!r} -> {fb.get(k)!r}")
    print(f"  other keys that differ:           {len(other)}")
    for k in other[:80]:
        print(f"    {k}: {fa.get(k)!r} -> {fb.get(k)!r}")
    return 1 if (moved or other) else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "diff":
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(
            "usage: corpus_solver.py run OUT.json | diff A.json B.json"
        )
