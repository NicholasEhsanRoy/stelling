# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""THE WIDENED S13 COVERAGE CENSUS — audit B9's finding on the first one.

`sweep_loop_assume.py` measured 96 rows moving REFUTED -> UNKNOWN and scored
EVERY one of them a false REFUTED. That is true of THAT corpus and it is a
property of the corpus, not of the change — but NOT for the reason first
recorded here, and the correction is the useful part.

The first attribution said the cause was the comparison set (`lt`/`le` only).
THIS FILE'S OWN OUTPUT DISPROVES THAT: the row `le x add_le5` is CORRECT, a
`<=` assert under a `le` assume. The operative property is STRICTNESS, not
direction. With no axiom forwarded the solver returns the degenerate model
`(0, 0)`; a STRICT conjunct excludes it, so the witness lands outside the
precondition and the refutation was false, while a NON-STRICT one admits it,
so the refutation was correct and is now withheld. Every assume set in the
144-row corpus contains a strict `lt`, which is why it observed zero of the
category — not because `ge`/`gt` were absent.

So this corpus adds `gt` and `ge` to the comparison set and a `>=` direction
to the asserts, and it partitions the moved rows FOUR ways instead of three,
because the third category of the first sweep ("a correct REFUTED existed,
with a different witness") is not the worst one available:

  VACUOUS            the assumed region admits no grid point at all, so the
                     refutation was necessarily false
  NO_CORRECT         inhabited region, witness outside the precondition, and
                     NO admitted grid point violates the assert — nothing
                     correct was lost
  OTHER_WITNESS      inhabited region, witness outside the precondition, but
                     some admitted grid point does violate — a correct
                     REFUTED existed, with a different witness
  CORRECT_WITHHELD   the pre-fix witness itself lies in the declared box,
                     satisfies EVERY assume, and violates the assert — a
                     correct refutation with a correct witness, withheld

CORRECT_WITHHELD is the honest coverage loss. It is checked POINTWISE in
exact `Fraction` arithmetic on the solver's own witness values, so it is a
positive existence proof and does not depend on the grid; the other three
categories are grid statements and say so.

Ground truth is exact `Fraction` arithmetic — never the tool under test.

Usage (run against BOTH trees, same generator):
  PYTHONPATH=<tree>/src JAX_ENABLE_X64=1 python sweep_loop_assume_wide.py \
      --json=OUT.json
  python sweep_loop_assume_wide.py --compare PRE.json POST.json
"""
from __future__ import annotations

import itertools
import json
import sys
from fractions import Fraction as F

LO, HI = -10.0, 10.0
FLO, FHI = F(LO), F(HI)

# THE COMPARISON SET. Widened to both directions, but note the operative
# property is STRICTNESS rather than direction: `le` alone produces a
# CORRECT_WITHHELD row under a `<=` assert (see `le x add_le5` in the output).
# A strict conjunct excludes the solver's degenerate `(0, 0)` model; a
# non-strict one admits it.
CMP = {
    "lt": (lambda a, b: a < b, lambda a, b: a < b),
    "le": (lambda a, b: a <= b, lambda a, b: a <= b),
    "gt": (lambda a, b: a > b, lambda a, b: a > b),
    "ge": (lambda a, b: a >= b, lambda a, b: a >= b),
}
ASSUME_SETS = {
    "lt": [(0, "lt", 1)],
    "le": [(0, "le", 1)],
    "gt": [(0, "gt", 1)],
    "ge": [(0, "ge", 1)],
    "unsat": [(0, "lt", 1), (1, "lt", 0)],
}
# FOUR ASSERT SHAPES, BOTH DIRECTIONS. The `_ge` pair is the other half of
# the same omission: the first sweep's asserts all read `expr <= threshold`.
ASSERTS = {
    "sub_le0":   (lambda v: v[0] - v[1] <= 0.0,  lambda v: v[0] - v[1] <= F(0)),
    "sub_ge0":   (lambda v: v[0] - v[1] >= 0.0,  lambda v: v[0] - v[1] >= F(0)),
    "add_le5":   (lambda v: v[0] + v[1] <= 5.0,  lambda v: v[0] + v[1] <= F(5)),
    "add_ge_m5": (lambda v: v[0] + v[1] >= -5.0, lambda v: v[0] + v[1] >= F(-5)),
}
CARRIERS = ["scan", "while", "fori", "nested_scan", "scan_in_cond", "top"]
ORDERS = ["assume_first", "assume_last"]


def build(carrier, aset, assert_name, order):
    """The harness, and the spec the exact oracle reads. Two declarations
    throughout, so the ground-truth grid is a plain 41x41 and every category
    below is a statement about a set this file can enumerate."""
    import jax.numpy as jnp
    from jax import lax

    from stelling.harness import any_array, assert_, assume

    triples = ASSUME_SETS[aset]
    jax_assert = ASSERTS[assert_name][0]

    def harness():
        xs = [any_array((), "float64", (LO, HI)) for _ in range(2)]

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
                    return state[0] < 2

                def body(state):
                    side_body()
                    return (state[0] + 1, state[1])
                lax.while_loop(cond, body, (jnp.int32(0), xs[0]))
            elif carrier == "fori":
                def body(i, c):
                    side_body()
                    return c
                lax.fori_loop(0, 2, body, xs[0])
            elif carrier == "nested_scan":
                def outer(c, _):
                    def inner(c2, _):
                        side_body()
                        return c2, 0.0
                    lax.scan(inner, c, jnp.zeros((2,)))
                    return c, 0.0
                lax.scan(outer, xs[0], jnp.zeros((2,)))
            elif carrier == "scan_in_cond":
                def branch(a):
                    def body(c, _):
                        side_body()
                        return c, 0.0
                    lax.scan(body, a, jnp.zeros((2,)))
                    return a
                lax.cond(xs[0] > 0.0, branch, lambda a: a, xs[0])
            else:  # pragma: no cover
                raise AssertionError(carrier)

        if order == "assume_first":
            emit_assumes()
        r = assert_(jax_assert(xs))
        if order == "assume_last":
            emit_assumes()
        return r

    spec = dict(carrier=carrier, aset=aset, assert_name=assert_name,
                order=order, triples=triples)
    return harness, spec


# ---------------------------------------------------------------------------
# GROUND TRUTH — exact rationals, computed here, never by stelling
# ---------------------------------------------------------------------------


def admits(spec, pt):
    """A loop-body assume constrains the DECLARATIONS (they are closed over,
    not carried) and the loop runs, so the admitted set is the plain
    conjunction — the same set the top-level spelling declares."""
    return all(CMP[op][1](pt[i], pt[j]) for (i, op, j) in spec["triples"])


def violates(spec, pt):
    return not ASSERTS[spec["assert_name"]][1](list(pt))


def census(spec, n=41):
    pts = [FLO + (FHI - FLO) * F(k, n - 1) for k in range(n)]
    seen = bad = 0
    for pt in itertools.product(pts, repeat=2):
        if not admits(spec, pt):
            continue
        seen += 1
        if violates(spec, pt):
            bad += 1
    return seen, bad


def witness_points(spec, witnesses):
    """The solver's own witness values, as exact rationals."""
    out = []
    for w in witnesses:
        pinned = {}
        for name, val in w.values:
            if name.startswith("x") and name[1:].isdigit():
                pinned[int(name[1:])] = F(val)
        out.append([pinned.get(k, F(0)) for k in range(2)])
    return out


def in_box(pt):
    return all(FLO <= c <= FHI for c in pt)


# ---------------------------------------------------------------------------


def run(out_path):
    import jax
    jax.config.update("jax_enable_x64", True)
    from stelling.preconditions import check

    rows = {}
    counts = {}
    total = 0
    for carrier, aset, assert_name, order in itertools.product(
        CARRIERS, ASSUME_SETS, ASSERTS, ORDERS
    ):
        h, spec = build(carrier, aset, assert_name, order)
        total += 1
        key = "|".join((carrier, aset, assert_name, order))
        try:
            v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=5000)
            status = v.status
            wits = [[str(c) for c in p]
                    for p in witness_points(spec, v.witnesses)]
        except Exception as e:
            status = f"RAISED {type(e).__name__}"
            wits = []
        counts[status] = counts.get(status, 0) + 1
        seen, bad = census(spec)
        rows[key] = dict(status=status, witnesses=wits,
                         admitted=seen, violating=bad)
    print("=" * 74)
    print(f"generated harnesses: {total}")
    for k in sorted(counts):
        print(f"  {k:34s} {counts[k]}")
    print("=" * 74)
    with open(out_path, "w") as fh:
        json.dump(rows, fh, indent=0, sort_keys=True)


def classify(key, pre, post):
    """The FOUR-WAY partition of one moved row. `pre` is the pre-fix result
    (its witness is what a user would have acted on); `post` is this tree's."""
    carrier, aset, assert_name, order = key.split("|")
    # the spec, rebuilt from the key alone: `--compare` scores JSON and must
    # not need jax, a solver, or either tree on the path
    spec = dict(carrier=carrier, aset=aset, assert_name=assert_name,
                order=order, triples=ASSUME_SETS[aset])
    for w in pre["witnesses"]:
        pt = [F(c) for c in w]
        if in_box(pt) and admits(spec, pt) and violates(spec, pt):
            return "CORRECT_WITHHELD"
    if pre["admitted"] == 0:
        return "VACUOUS"
    return "OTHER_WITNESS" if pre["violating"] else "NO_CORRECT"


def compare(pre_path, post_path):
    pre = json.load(open(pre_path))
    post = json.load(open(post_path))
    assert set(pre) == set(post), "the two runs are not the same corpus"
    moved = {}
    part = {}
    by_carrier = {}
    for key in sorted(pre):
        a, b = pre[key]["status"], post[key]["status"]
        if a == b:
            continue
        moved[f"{a} -> {b}"] = moved.get(f"{a} -> {b}", 0) + 1
        if not (a == "REFUTED" and b == "UNKNOWN"):
            continue
        cat = classify(key, pre[key], post[key])
        part[cat] = part.get(cat, 0) + 1
        if cat == "CORRECT_WITHHELD":
            c = key.split("|")[0]
            by_carrier[c] = by_carrier.get(c, 0) + 1
    print("=" * 74)
    print(f"rows: {len(pre)}   moved: {sum(moved.values())}")
    for k in sorted(moved):
        print(f"  {k:24s} {moved[k]}")
    print("-" * 74)
    print("PARTITION of the REFUTED -> UNKNOWN rows "
          "(pre-fix witness scored in exact Fraction):")
    tot = sum(part.values())
    for k in ("VACUOUS", "NO_CORRECT", "OTHER_WITNESS", "CORRECT_WITHHELD"):
        n = part.get(k, 0)
        pct = f"{100.0 * n / tot:.1f}%" if tot else "-"
        print(f"  {k:18s} {n:4d}  {pct}")
    print(f"  {'TOTAL':18s} {tot:4d}")
    print("-" * 74)
    print("CORRECT_WITHHELD by carrier:")
    for k in sorted(by_carrier):
        print(f"  {k:14s} {by_carrier[k]}")
    print("=" * 74)


if __name__ == "__main__":
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        compare(sys.argv[i + 1], sys.argv[i + 2])
    else:
        out = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--json="))
        run(out)
