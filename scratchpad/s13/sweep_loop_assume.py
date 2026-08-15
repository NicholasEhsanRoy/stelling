# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""COST + SOUNDNESS census for audit 0.2.0 S13: an `assume` inside a `scan`
or `while_loop` body, which the propagator's walk does not enter.

Same shape as fix-0.2.0-scratch/sweep_assume_scope.py, with the carrier set
moved to the constructs THAT audit is about, and one extra column: whether the
verdict says anything at all about the assume.

Ground truth is exact `Fraction` arithmetic over a grid of the declared box —
never the tool under test.

Usage:
  PYTHONPATH=<tree>/src JAX_ENABLE_X64=1 python sweep_loop_assume.py [--json=F]
"""
from __future__ import annotations

import itertools
import sys
from fractions import Fraction as F

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

from stelling.preconditions import check  # noqa: E402
from stelling.harness import any_array, assert_, assume  # noqa: E402

LO, HI = -10.0, 10.0
FLO, FHI = F(LO), F(HI)

EXPRS = {
    "sub01": (2, lambda v: v[0] - v[1], lambda v: v[0] - v[1]),
    "add01": (2, lambda v: v[0] + v[1], lambda v: v[0] + v[1]),
}
CMP = {
    "lt": (lambda a, b: a < b, lambda a, b: a < b),
    "le": (lambda a, b: a <= b, lambda a, b: a <= b),
}
ASSUME_SETS = {
    "sat1":  [(0, "lt", 1)],
    "sat2":  [(0, "lt", 1), (1, "le", 2)],
    "unsat": [(0, "lt", 1), (1, "lt", 0)],
}


def build(carrier, ndecl, tail, aset, expr_name, thresh, order):
    triples = [t for t in ASSUME_SETS[aset] if t[0] < ndecl and t[2] < ndecl]
    if not triples or EXPRS[expr_name][0] > ndecl:
        return None, None
    jax_expr = EXPRS[expr_name][1]

    def harness():
        xs = [any_array((), "float64", (LO, HI)) for _ in range(ndecl)]

        def side_body():
            for (i, op, j) in triples:
                assume(CMP[op][0](xs[i], xs[j]))

        def emit_assumes():
            if carrier == "top":
                side_body()
            elif carrier == "scan":
                def body(c, _):
                    side_body()
                    return c, 0.0
                lax.scan(body, xs[0], jnp.zeros((2,)))
            elif carrier == "while":
                def cond(state):
                    i, _ = state
                    return i < 2

                def body(state):
                    i, c = state
                    side_body()
                    return (i + 1, c)
                lax.while_loop(cond, body, (jnp.int32(0), xs[0]))
            else:  # pragma: no cover
                raise AssertionError(carrier)

        if order == "assume_first":
            emit_assumes()
        r = assert_(jax_expr(xs) <= thresh)
        if order == "assume_last":
            emit_assumes()
        for k in range(tail):
            _ = xs[0] * (1.0 + k)
        return r

    spec = dict(carrier=carrier, ndecl=ndecl, tail=tail, aset=aset,
                expr=expr_name, thresh=thresh, order=order, triples=triples)
    return harness, spec


def admitted(spec, pt):
    """A loop-body assume constrains the DECLARATIONS (they are closed over,
    not carried), and the loop runs, so the admitted set is the plain
    conjunction — the same set the top-level spelling declares."""
    for (i, op, j) in spec["triples"]:
        if not CMP[op][1](pt[i], pt[j]):
            return False
    return True


def census(spec, n):
    """(admitted points, admitted points violating the assert) on an exact
    n**ndecl Fraction grid."""
    pts = [FLO + (FHI - FLO) * F(k, n - 1) for k in range(n)]
    ex = EXPRS[spec["expr"]][2]
    thr = F(spec["thresh"])
    seen = bad = 0
    for pt in itertools.product(pts, repeat=spec["ndecl"]):
        if not admitted(spec, pt):
            continue
        seen += 1
        if not (ex(list(pt)) <= thr):
            bad += 1
    return seen, bad


def witness_problems(spec, witnesses):
    problems = []
    for w in witnesses:
        pinned = {}
        for name, val in w.values:
            if name.startswith("x") and name[1:].isdigit():
                pinned[int(name[1:])] = F(val)
        if not admitted(spec, [pinned.get(k, F(0)) for k in range(spec["ndecl"])]):
            problems.append(f"witness {dict(w.values)} violates the precondition")
    return problems


def main():
    carriers = ["scan", "while", "top"]
    ndecls = [2, 3]
    tails = [0, 1]
    gridn = 40

    rows = {}
    counts = {}
    false_refuted = []
    false_verified = []
    vacuous_verified = []
    silent = []
    total = 0
    for carrier, ndecl, tail, aset, (expr, thr), order in itertools.product(
        carriers, ndecls, tails, list(ASSUME_SETS),
        [("sub01", 0.0), ("add01", -5.0)], ["assume_last", "assume_first"],
    ):
        h, spec = build(carrier, ndecl, tail, aset, expr, thr, order)
        if h is None:
            continue
        total += 1
        try:
            v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=5000)
            status = v.status
        except Exception as e:
            status = f"RAISED {type(e).__name__}"
            v = None
        key = "|".join(str(k) for k in
                       (carrier, ndecl, tail, aset, expr, thr, order))
        counts[status] = counts.get(status, 0) + 1
        seen, bad = census(spec, gridn)
        mentions = False
        if v is not None:
            blob = "\n".join(list(v.notes) + list(v.stamp.assumptions))
            mentions = ("assume" in blob.lower())
        rows[key] = f"{status}|admitted={seen}|violating={bad}|mentions_assume={mentions}"
        if v is not None and not mentions:
            silent.append(key)
        if status == "VERIFIED":
            if seen == 0:
                vacuous_verified.append(key)
            elif bad:
                false_verified.append((key, seen, bad))
        elif status == "REFUTED":
            probs = witness_problems(spec, v.witnesses)
            if probs:
                false_refuted.append((key, probs))
            if not v.witnesses:
                false_refuted.append((key, ["REFUTED with no witness"]))

    print("=" * 74)
    print(f"generated harnesses: {total}")
    for k in sorted(counts):
        print(f"  {k:14s} {counts[k]}")
    print(f"FALSE REFUTED (witness outside the precondition): {len(false_refuted)}")
    for key, probs in false_refuted:
        print(f"   {key}  {probs}")
    print(f"FALSE VERIFIED (admitted point violates assert):  {len(false_verified)}")
    for key, seen, bad in false_verified:
        print(f"   {key}  admitted={seen} violating={bad}")
    print(f"VACUOUS VERIFIED (0 admitted points on the grid): {len(vacuous_verified)}")
    for key in vacuous_verified:
        print(f"   {key}")
    print(f"VERDICTS THAT MENTION NO ASSUME AT ALL:           {len(silent)}")
    for key in silent:
        print(f"   {key}")
    print("=" * 74)
    for flag in sys.argv:
        if flag.startswith("--json="):
            import json
            with open(flag.split("=", 1)[1], "w") as fh:
                json.dump(rows, fh, indent=0, sort_keys=True)


if __name__ == "__main__":
    main()
