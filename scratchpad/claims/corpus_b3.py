# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""B3 — a SECOND, independently generated corpus for the query-scoped
assume change, built to answer one question the first one cannot.

`scratchpad/mechc/corpus.py` is a hand-written case list, and its
"12 wrong REFUTEDs closed, 6 sound ones lost" is a count on it. The nine
harnesses of it that actually move split six-with-an-EMPTY-assumed-region
to three-with-a-NON-EMPTY one, and each contributes two obligation-runs
(the two `refine` legs), which is where 12:6 comes from: the trade ratio
IS that corpus's empty:non-empty ratio among the rows that move. Nothing
about the change fixes it at 2:1.

This corpus is therefore not another hand-picked list. It is a full
CROSS PRODUCT of the three axes that decide whether a row moves and
which way:

    mechanism M  x  region R  x  obligation shape O

  M ∈ {drop_reduction, drop_relational, drop_or, uncertified_narrowing}
        — the four ways this tree can end a run with an uncertified
          precondition
  R ∈ {empty, nonempty}
        — the two truths the oracle can find, ONE EACH, by construction:
          the empty:non-empty ratio of this corpus is 1:1 because the
          grid says so, not because a case list came out that way
  O ∈ {elementwise, reduce_sum, affine}
        — the three judging legs that can mint a definite violation

Every cell carries TWO obligations: one violated over the box and one
discharged over it, so the one-sidedness (`discharged` is never
withheld) is scored on the same rows as the trade. Every cell is then
run in both trace orders, under both `refine` legs, in both assume
modes.

Controls, outside the grid: three no-assume harnesses (which must keep
refuting on each leg), a certified declared-input narrowing, and a
definitely-true (F8-channel) assume.

WHAT THIS CORPUS CANNOT SEE. It declares only `float64`; it uses no
`lax.cond` and no `nonvacuity` (both are in the mechc corpus and not
here); its assumed regions are boxes or half-spaces of a box, never a
curved region; and it says nothing about how often any of these shapes
occurs in real harnesses. Like the first corpus it is built to REACH
the defect, so it over-samples it: neither ledger is a rate.

Usage, from a worktree, with `JAX_PLATFORMS=cpu`:

    PYTHONPATH=<tree>/src python scratchpad/claims/corpus_b3.py run OUT.json
    python scratchpad/claims/corpus_b3.py score BASE.json BRANCH.json

`run` needs jax + stelling and stamps the resolved `stelling.__file__`
and `jax.__version__` into its output; `score` needs neither.
"""
from __future__ import annotations

import itertools
import json
import sys

F = "float64"

# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------
# Each mechanism is (name, decls, assume_before_fn, note).  `decls` is a
# tuple of (shape, dtype, lo, hi).  The assume thunk takes the declared
# values in order.
#
# The two `region` variants of one mechanism differ ONLY in a constant, so
# nothing but the truth of the assumed region changes between them.


def _mechanisms():
    one = (((3,), F, -1.0, 1.0),)
    two = (((3,), F, 0.0, 10.0), ((3,), F, 5.0, 6.0))
    two_hi = (((3,), F, 7.0, 10.0), ((3,), F, 5.0, 6.0))
    import jax.numpy as jnp

    return [
        # a reduction: not in the representable census -> DROPPED whole
        ("drop_reduction", "empty", one,
         lambda x: jnp.all(x >= 2.0)),
        ("drop_reduction", "nonempty", one,
         lambda x: jnp.all(x >= -2.0)),
        # a relation between two varying quantities -> DROPPED whole
        ("drop_relational", "empty", two_hi,
         lambda a, b: a <= b),
        ("drop_relational", "nonempty", two,
         lambda a, b: a <= b),
        # a disjunction: outside the census -> DROPPED whole
        ("drop_or", "empty", one,
         lambda x: (x >= 2.0) | (x >= 3.0)),
        ("drop_or", "nonempty", one,
         lambda x: (x >= -2.0) | (x >= 3.0)),
        # a narrowing of an OVER-APPROXIMATED intermediate: applied, but
        # its satisfiability is uncertified
        ("uncertified_narrowing", "empty", one,
         lambda x: x * x <= -0.5),
        ("uncertified_narrowing", "nonempty", one,
         lambda x: x * x <= 0.25),
    ]


def _obligations(shape_name, n_decls):
    """(violated_fn, discharged_fn) for one obligation shape."""
    import jax.numpy as jnp

    if n_decls == 2:
        if shape_name == "elementwise":
            return (lambda a, b: a > 50.0), (lambda a, b: a > -50.0)
        if shape_name == "reduce_sum":
            return (lambda a, b: jnp.sum(a) >= 500.0), \
                   (lambda a, b: jnp.sum(a) >= -500.0)
        return (lambda a, b: a - a >= 0.5), (lambda a, b: a - a >= -0.5)
    if shape_name == "elementwise":
        return (lambda x: x > 5.0), (lambda x: x > -5.0)
    if shape_name == "reduce_sum":
        return (lambda x: jnp.sum(x) >= 100.0), (lambda x: jnp.sum(x) >= -100.0)
    return (lambda x: x - x >= 0.5), (lambda x: x - x >= -0.5)


SHAPES = ("elementwise", "reduce_sum", "affine")


def cases():
    """The full grid plus the controls, as
    (name, decls, assumes, asserts, note)."""
    import jax.numpy as jnp

    out = []
    for (m, r, decls, afn) in _mechanisms():
        for shape in SHAPES:
            viol, disc = _obligations(shape, len(decls))
            out.append((
                f"{m}__{r}__{shape}", decls, (afn,), (viol, disc),
                f"{m}, assumed region {r}, {shape} obligation",
            ))
    one = (((3,), F, -1.0, 1.0),)
    pt = (((), F, 0.0, 1.0),)
    v_e, d_e = _obligations("elementwise", 1)
    v_s, d_s = _obligations("reduce_sum", 1)
    v_a, d_a = _obligations("affine", 1)
    out += [
        ("control_no_assume__elementwise", one, (), (v_e, d_e),
         "CONTROL: no assume at all; the interval leg must keep refuting"),
        ("control_no_assume__reduce_sum", one, (), (v_s, d_s),
         "CONTROL: no assume at all; reduce_sum obligation"),
        ("control_no_assume__affine", one, (), (v_a, d_a),
         "CONTROL: no assume at all; the affine leg must keep refuting"),
        ("control_certified_input", pt, (lambda x: x >= 0.9,),
         (lambda x: x <= 0.5, lambda x: x <= 1.5),
         "CONTROL: certified narrowing of a DECLARED (exact) box"),
        ("control_f8_definitely_true", pt, (lambda x: x + 0.0 <= 10.0,),
         (lambda x: x + 0.0 >= 5.0, lambda x: x + 0.0 >= -5.0),
         "CONTROL: definitely-true assume, the F8 channel"),
    ]
    return out


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
# the oracle — numpy/jax over sampled points, stelling never consulted
# ---------------------------------------------------------------------------

_N_UNIFORM = 20000


def oracle(decls, assumes, asserts, seed=20260808):
    """Per obligation, over the DECLARED box under EVERY assume of the
    harness regardless of trace position: how many sampled points the
    precondition admits, and how many admitted points the obligation is
    FALSE at.

    The samples are members of the declared box by construction (uniform
    draws in it, all its corners, a product grid on the first
    declaration) — never produced by stelling and never compared with a
    stelling box.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(_N_UNIFORM):
        pts.append([rng.uniform(lo, hi, size=s).astype(np.float64)
                    for (s, _d, lo, hi) in decls])
    mids = [np.full(s, (lo + hi) / 2.0, dtype=np.float64)
            for (s, _d, lo, hi) in decls]
    for j, (s, _d, lo, hi) in enumerate(decls):
        n = int(np.prod(s)) if s else 1
        for mask in range(2 ** n):
            v = np.array([hi if (mask >> k) & 1 else lo for k in range(n)],
                         dtype=np.float64).reshape(s)
            row = list(mids)
            row[j] = v
            pts.append(row)
    s0, _d0, lo0, hi0 = decls[0]
    n0 = int(np.prod(s0)) if s0 else 1
    if n0 <= 3:
        axis = np.linspace(lo0, hi0, 21)
        grid = np.meshgrid(*([axis] * n0), indexing="ij")
        flat = np.stack([g.ravel() for g in grid], axis=1)
        for row in flat:
            r = list(mids)
            r[0] = row.reshape(s0)
            pts.append(r)
    batch = [jnp.asarray(np.stack([row[j] for row in pts]))
             for j in range(len(decls))]

    def _admit(*vs):
        ok = jnp.asarray(True)
        for a in assumes:
            ok = ok & jnp.all(a(*vs))
        return ok

    admitted = np.asarray(jax.vmap(_admit)(*batch))
    obs = []
    for p in asserts:
        truth = np.asarray(jax.vmap(lambda *vs, q=p: jnp.all(q(*vs)))(*batch))
        obs.append({"violating_admitted": int(np.sum(admitted & ~truth)),
                    "true_admitted": int(np.sum(admitted & truth))})
    return {"n_points": len(pts), "n_admitted": int(np.sum(admitted)),
            "obligations": obs}


# ---------------------------------------------------------------------------
# run / score
# ---------------------------------------------------------------------------


def run(path):
    import jax

    jax.config.update("jax_enable_x64", True)
    import stelling
    from stelling import affine as A
    from stelling import propagate as P
    from stelling._jax_compat import transcribe

    out = {"provenance": {"stelling": stelling.__file__,
                          "jax": jax.__version__,
                          "python": sys.executable},
           "cases": {}}
    for name, decls, assumes, asserts, note in cases():
        entry = {"note": note, "runs": {},
                 "oracle": oracle(decls, assumes, asserts)}
        for order, refine in itertools.product(ORDERS, (None, "affine")):
            key = f"{order}/{refine or 'none'}"
            h = build_harness(decls, assumes, asserts, order)
            cj = transcribe(jax.make_jaxpr(h)())
            rows = {}
            for mode in ("constrain", "inert"):
                p = P.propagate(cj, assume_mode=mode)
                if refine == "affine":
                    p, _r = A.refine_propagation(cj, p)
                rows[mode] = {
                    "obligations": [{"index": o.index, "status": o.status,
                                     "detail": o.detail}
                                    for o in p.obligations],
                    "notes": list(p.notes),
                    # `narrowing_uncertified` is absent from the BASE
                    # tree's Propagation, so it is read defensively: the
                    # same file has to run on both sides of the change.
                    "assume_dropped": p.assume_dropped,
                    "narrowing_uncertified": getattr(
                        p, "narrowing_uncertified", None
                    ),
                    "coverage_constrained": p.coverage.constrained,
                }
            entry["runs"][key] = rows
        out["cases"][name] = entry
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {path}: {len(out['cases'])} cases")


def score(a_path, b_path):
    A = json.load(open(a_path))
    B = json.load(open(b_path))
    print(f"A = {A['provenance']['stelling']}")
    print(f"B = {B['provenance']['stelling']}")
    total = moved = 0
    empty_closed = real_lost = neither = 0
    to_discharged = 0
    inert_diff = 0
    rows = []
    for name in sorted(A["cases"]):
        ca, cb = A["cases"][name], B["cases"][name]
        orc = ca["oracle"]
        for key in sorted(ca["runs"]):
            for mode in ("constrain", "inert"):
                ra = ca["runs"][key][mode]
                rb = cb["runs"][key][mode]
                for oa, ob in zip(ra["obligations"], rb["obligations"]):
                    total += 1
                    if oa["status"] == ob["status"]:
                        continue
                    if mode == "inert":
                        inert_diff += 1
                    moved += 1
                    if ob["status"] == "discharged":
                        to_discharged += 1
                    va = orc["obligations"][oa["index"]]["violating_admitted"]
                    if orc["n_admitted"] == 0:
                        empty_closed += 1
                        why = "region EMPTY (wrong REFUTED closed)"
                    elif va > 0:
                        real_lost += 1
                        why = f"genuine violation at {va} admitted points (LOST)"
                    else:
                        neither += 1
                        why = "neither"
                    rows.append(
                        f"{name:44s} {key + '/' + mode:22s} "
                        f"{oa['status']} -> {ob['status']}  ob#{oa['index']}  {why}"
                    )
    for r in rows:
        print(r)
    print()
    print(f"obligation-runs total:     {total}")
    print(f"obligation-runs moved:     {moved}")
    print(f"  (a) region EMPTY  (wrong REFUTED closed): {empty_closed}")
    print(f"  (b) genuine violating point (REAL LOSS):  {real_lost}")
    print(f"  (c) neither:                              {neither}")
    print(f"moves toward discharged (must be 0):        {to_discharged}")
    print(f"inert-mode diffs (must be 0):               {inert_diff}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
