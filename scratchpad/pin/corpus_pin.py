# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""PIN — a corpus for the shared-point pin repair, built for ONE job:
show, PER OBLIGATION, that nothing in this branch moves a verdict.

The branch's changes are pins and prose. That is exactly the class of
change where "it cannot move a verdict" is easiest to assert and easiest
to be wrong about, so the assertion is measured instead: every obligation
of every row is recorded on both trees and compared key-for-key.

WHY IT IS NOT ONE OF THE EXISTING CORPORA. `scratchpad/cert/corpus.py`
scores the non-emptiness certificate, `scratchpad/mechc/corpus.py` the
query-scoped withholding and `scratchpad/claims/corpus_b3.py` the
mechanism x region x shape grid. All three are readable and none of them
carries the two shapes this branch's findings are about: an assume inside
a `lax.cond` BRANCH (F2's counter-construction, and its cost twin whose
region is inhabited only via the UNTAKEN branch), and the `ieee`/`real`
split on the boundary point F5 scopes. Those are in the grid below.

THE AXES

    mechanism  x  obligation shape  x  trace order  x  assume mode
                                    x  semantics    x  refine leg

  mechanism — every way this tree can end a run with an uncertified
    precondition, each in an EMPTY and a NON-EMPTY variant where the
    variant is meaningful, plus the branch-scoped pair and four controls
  shape — the three legs that can mint a definite violation
    (elementwise / reduce_sum interval, affine), each row carrying one
    obligation violated over the box and one discharged over it, so the
    one-sidedness is scored on the same rows as everything else
  order — the assume traced before and after the obligations, because
    the withholding is QUERY-scoped and an order-scoped regression is
    exactly what that scoping forbids
  assume mode — `constrain` and `inert`
  semantics — `real` and `ieee`
  refine — the interval leg alone and the affine refinement over it

WHAT THIS CORPUS CANNOT SEE. It declares `float64`, `float32` and
`int32` but no other dtype; it has no solver escalation (no network, no
budget) so nothing here scores `solvers.py`; its assumed regions are
half-spaces, boxes, and one curved region (`x*x <= c`), never a
disconnected one; and it is built to REACH the shapes under repair, so
its ratios are not rates in any population.

Usage, from a worktree, with `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`:

    PYTHONPATH=<tree>/src python scratchpad/pin/corpus_pin.py run OUT.json
    python scratchpad/pin/corpus_pin.py diff A.json B.json

`run` needs jax + stelling and stamps the resolved `stelling.__file__`
and `jax.__version__` into its output; `diff` needs neither.
"""
from __future__ import annotations

import json
import sys

F = "float64"
F32 = "float32"
I32 = "int32"


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------


def _mechanisms():
    """(name, decls, assume_fns, note). `decls` is a tuple of
    (shape, dtype, lo, hi); each assume fn takes the declared values."""
    import jax.numpy as jnp

    one = (((3,), F, -1.0, 1.0),)
    pt = (((), F, 0.0, 1.0),)
    two = (((3,), F, 0.0, 10.0), ((3,), F, 5.0, 6.0))
    two_hi = (((3,), F, 7.0, 10.0), ((3,), F, 5.0, 6.0))
    narrow32 = (((2,), F32, 0.0, 1.0),)
    ints = (((2,), I32, 0.0, 4.0),)
    return [
        ("no_assume", "n/a", one, ()),
        ("certified_declared", "nonempty", pt, (lambda x: x >= 0.9,)),
        ("f8_definitely_true", "nonempty", pt, (lambda x: x + 0.0 <= 10.0,)),
        ("uncertified_narrowing", "empty", one, (lambda x: x * x <= -0.5,)),
        ("uncertified_narrowing", "nonempty", one, (lambda x: x * x <= 0.25,)),
        ("drop_reduction", "empty", one, (lambda x: jnp.all(x >= 2.0),)),
        ("drop_reduction", "nonempty", one, (lambda x: jnp.all(x >= -2.0),)),
        ("drop_relational", "empty", two_hi, (lambda a, b: a <= b,)),
        ("drop_relational", "nonempty", two, (lambda a, b: a <= b,)),
        ("drop_or", "empty", one, (lambda x: (x >= 2.0) | (x >= 3.0),)),
        ("drop_or", "nonempty", one, (lambda x: (x >= -2.0) | (x >= 3.0),)),
        ("float32_narrow", "nonempty", narrow32, (lambda x: x >= 0.5,)),
        ("int32_narrow", "nonempty", ints, (lambda x: x >= 2,)),
        # the dtype rows that actually WITHHOLD, and so actually ask the
        # certificate: the assume narrows `x * 2`, an over-approximated
        # intermediate, which is what sets `narrowing_uncertified`. The
        # `*_narrow` pair above narrows the DECLARED (exact) box instead,
        # certifies at the F7 channel and never reaches a search — they
        # are kept as the contrast, not as the measurement.
        ("uncertified_narrowing_f32", "nonempty", narrow32,
         (lambda x: x * 2.0 >= 0.5,)),
        ("uncertified_narrowing_i32", "nonempty", ints,
         (lambda x: x * 2 >= 4,)),
    ]


def _obligations(shape_name, decls):
    """(violated_fn, discharged_fn) for one obligation shape."""
    import jax.numpy as jnp

    n = len(decls)
    if n == 2:
        if shape_name == "elementwise":
            return (lambda a, b: a > 50.0), (lambda a, b: a > -50.0)
        if shape_name == "reduce_sum":
            return (
                (lambda a, b: jnp.sum(a) >= 500.0),
                (lambda a, b: jnp.sum(a) >= -500.0),
            )
        return (lambda a, b: a - a >= 0.5), (lambda a, b: a - a >= -0.5)
    dt = decls[0][1]
    if dt == I32:
        if shape_name == "elementwise":
            return (lambda x: x > 50), (lambda x: x > -50)
        if shape_name == "reduce_sum":
            return (lambda x: jnp.sum(x) >= 500), (lambda x: jnp.sum(x) >= -500)
        return (lambda x: x - x >= 1), (lambda x: x - x >= -1)
    if shape_name == "elementwise":
        return (lambda x: x > 5.0), (lambda x: x > -5.0)
    if shape_name == "reduce_sum":
        return (lambda x: jnp.sum(x) >= 100.0), (lambda x: jnp.sum(x) >= -100.0)
    return (lambda x: x - x >= 0.5), (lambda x: x - x >= -0.5)


SHAPES = ("elementwise", "reduce_sum", "affine")
ORDERS = ("before", "after")


def build_harness(decls, assumes, asserts, order):
    from stelling.harness import any_array, assert_, assume

    def h():
        vs = [any_array(s, dt, (lo, hi)) for (s, dt, lo, hi) in decls]
        out = []
        if order == "before":
            for a in assumes:
                assume(a(*vs))
            for p in asserts:
                out.append(assert_(p(*vs)))
        else:
            for p in asserts:
                out.append(assert_(p(*vs)))
            for a in assumes:
                assume(a(*vs))
        return tuple(out)

    return h


# ---------------------------------------------------------------------------
# the rows the existing corpora do not carry
# ---------------------------------------------------------------------------


def _special_harnesses():
    """Named zero-argument harnesses, outside the grid.

    `branch_assume_taken` is F2's counter-construction: the ONLY assume
    of the query sits inside a `lax.cond` branch, and the probe that pins
    the declaration to its HIGH corner walks INTO that branch, evaluates
    the assume and witnesses it. `branch_assume_untaken` is its cost
    twin: the assumed region is inhabited only by points whose walk takes
    the branch WITHOUT the assume, so the static requirement declines a
    refutation that is sound. `boundary_real_vs_ieee` is F5's dial row.
    """
    import jax
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_, assume

    def branch_assume_taken():
        x = any_array((), F, (0.0, 1.0))

        def has_assume(v):
            assume(v >= 0.25)
            return v * 2.0

        y = jax.lax.cond(x >= 0.5, has_assume, lambda v: v, x)
        return (assert_(y <= -1.0), assert_(y >= -1.0))

    def branch_assume_untaken():
        x = any_array((), F, (0.0, 1.0))

        def has_assume(v):
            assume(v >= 2.0)
            return v * 2.0

        y = jax.lax.cond(x >= 0.5, has_assume, lambda v: v, x)
        return (assert_(y <= -1.0), assert_(y >= -1.0))

    def branch_violation():
        x = any_array((), F, (-1.0, 1.0))

        def hi(v):
            return v + 10.0

        y = jax.lax.cond(x >= 0.5, hi, lambda v: v, x)
        return (assert_(y <= 5.0), assert_(y <= 500.0))

    def boundary_real_vs_ieee():
        x = any_array((2,), F, (0.1, 0.2))
        assume(x[0] + x[1] >= 0.30000000000000004)
        return (assert_(jnp.sum(x) <= -1.0), assert_(jnp.sum(x) <= 100.0))

    def nested_branch_assume():
        x = any_array((2,), F, (0.0, 1.0))

        def inner(v):
            assume(jnp.sum(v) >= 0.5)
            return v * 3.0

        y = jax.lax.cond(x[0] >= 0.5, inner, lambda v: v, x)
        return (assert_(jnp.sum(y) <= -1.0), assert_(jnp.sum(y) <= 100.0))

    return [
        ("branch_assume_taken", branch_assume_taken),
        ("branch_assume_untaken", branch_assume_untaken),
        ("branch_violation", branch_violation),
        ("boundary_real_vs_ieee", boundary_real_vs_ieee),
        ("nested_branch_assume", nested_branch_assume),
    ]


def cases():
    """(name, harness_thunk) for every row of the corpus."""
    rows = []
    for (m, r, decls, afns) in _mechanisms():
        for shape in SHAPES:
            viol, disc = _obligations(shape, decls)
            for order in ORDERS:
                rows.append((
                    f"{m}__{r}__{shape}__{order}",
                    build_harness(decls, afns, (viol, disc), order),
                ))
    rows.extend(_special_harnesses())
    return rows


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

MODES = ("constrain", "inert")
SEMANTICS = ("real", "ieee")


def _record(closed, semantics, assume_mode):
    """One propagation, both legs, as plain JSON."""
    from stelling import affine
    from stelling.propagate import propagate

    out = {}
    try:
        p = propagate(closed, semantics=semantics, assume_mode=assume_mode)
    except Exception as e:  # noqa: BLE001 — a refusal is an outcome
        return {"interval": f"RAISED {type(e).__name__}: {e}"}
    out["interval"] = {
        "obligations": [
            {"index": o.index, "status": o.status, "detail": o.detail}
            for o in p.obligations
        ],
        "nonvacuity": [
            {"index": o.index, "status": o.status} for o in p.nonvacuity_checks
        ],
        "assume_dropped": p.assume_dropped,
        "narrowing_uncertified": p.narrowing_uncertified,
        "region_inhabited": p.region_inhabited,
        "assumptions": list(p.assumptions),
        "n_notes": len(p.notes),
        "constrained": p.coverage.constrained,
    }
    try:
        r, rep = affine.refine_propagation(closed, p)
        out["affine"] = {
            "obligations": [
                {"index": o.index, "status": o.status, "detail": o.detail}
                for o in r.obligations
            ],
            "discharged": list(rep.discharged),
            "violated": list(rep.violated),
            "undecided": list(rep.undecided),
            "declined": [list(d) for d in rep.declined],
        }
    except Exception as e:  # noqa: BLE001
        out["affine"] = f"RAISED {type(e).__name__}: {e}"
    return out


def run(path):
    import jax

    jax.config.update("jax_enable_x64", True)
    import stelling
    from stelling.harness import trace
    from stelling.preconditions import check

    results = {
        "_stelling_file": stelling.__file__,
        "_jax_version": jax.__version__,
        "rows": {},
    }
    for name, h in cases():
        row = {}
        try:
            closed = trace(h)
        except Exception as e:  # noqa: BLE001
            results["rows"][name] = {"trace": f"RAISED {type(e).__name__}: {e}"}
            continue
        for sem in SEMANTICS:
            for mode in MODES:
                row[f"{sem}/{mode}"] = _record(closed, sem, mode)
        for refine in (None, "affine"):
            try:
                v = check(h, vacuity_mode="inputs-only", refine=refine)
                row[f"verdict/{refine}"] = {
                    "status": v.status,
                    "obligations": [
                        {"index": o.index, "status": o.status}
                        for o in getattr(v, "obligations", ())
                    ],
                }
            except Exception as e:  # noqa: BLE001
                row[f"verdict/{refine}"] = f"RAISED {type(e).__name__}: {e}"
        results["rows"][name] = row
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1, sort_keys=True, default=str)
    print(f"{len(results['rows'])} rows -> {path}")
    print(f"  stelling: {results['_stelling_file']}")
    print(f"  jax:      {results['_jax_version']}")


# ---------------------------------------------------------------------------
# the diff — PER OBLIGATION, not per query
# ---------------------------------------------------------------------------


def _normalise(v, a_root, b_root):
    """Two trees are two DIRECTORIES, and every `source_info` string in a
    detail or a stamped assumption carries the one it ran in. Erasing the
    root is the difference between "no verdict moved" and a page of
    path noise; erasing anything else would be erasing the finding, so
    only the two roots are erased and only from strings."""
    if not isinstance(v, str):
        return v
    for root in (a_root, b_root):
        if root:
            v = v.replace(root, "<TREE>")
    return v


def _flatten(blob, prefix, out):
    if isinstance(blob, dict):
        for k, v in sorted(blob.items()):
            _flatten(v, f"{prefix}.{k}", out)
    elif isinstance(blob, list):
        for i, v in enumerate(blob):
            _flatten(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = blob


def diff(a_path, b_path):
    with open(a_path) as fh:
        a = json.load(fh)
    with open(b_path) as fh:
        b = json.load(fh)
    print(f"A: {a['_stelling_file']}  jax {a['_jax_version']}")
    print(f"B: {b['_stelling_file']}  jax {b['_jax_version']}")
    a_root = a["_stelling_file"].split("/src/stelling/")[0]
    b_root = b["_stelling_file"].split("/src/stelling/")[0]
    fa, fb = {}, {}
    _flatten(a["rows"], "", fa)
    _flatten(b["rows"], "", fb)
    fa = {k: _normalise(v, a_root, b_root) for k, v in fa.items()}
    fb = {k: _normalise(v, a_root, b_root) for k, v in fb.items()}
    keys = sorted(set(fa) | set(fb))
    status_keys = [k for k in keys if k.endswith(".status")]
    moved = [k for k in status_keys if fa.get(k) != fb.get(k)]
    other = [
        k
        for k in keys
        if not k.endswith(".status") and fa.get(k) != fb.get(k)
    ]
    print(f"leaf keys compared: {len(keys)}")
    print(f"  per-obligation/verdict STATUS keys: {len(status_keys)}")
    print(f"  STATUS keys that MOVED:             {len(moved)}")
    for k in moved[:60]:
        print(f"    {k}: {fa.get(k)!r} -> {fb.get(k)!r}")
    print(f"  non-status keys that differ:        {len(other)}")
    for k in other[:60]:
        print(f"    {k}: {fa.get(k)!r} -> {fb.get(k)!r}")
    return 1 if (moved or other) else 0


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "diff":
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit("usage: corpus_pin.py run OUT.json | diff A.json B.json")
