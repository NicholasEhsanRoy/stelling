# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""B2 — the downstream surface the end-of-run withholding widened.

Withholding a definite violation at the END of the run rather than at the
assert leaves the obligation `unknown` in the object the next two layers
read, and BOTH of them key on exactly that word: `affine.refine_propagation`
takes `[o for o in propagation.obligations if o.status == "unknown"]`, and
`solvers.escalate` takes the same list. So obligations that used to be
`violated-over-set` when those layers saw them are now offered to them.

This script dumps, per (case, order), what each layer was offered and what
it did with it, for ONE tree. Run it once per tree and diff:

    PYTHONPATH=<base>/src   python b2_surface.py BASE.json
    PYTHONPATH=<branch>/src python b2_surface.py BRANCH.json
    python b2_surface.py --diff BASE.json BRANCH.json

Corpora: `scratchpad/mechc/corpus.py` and `scratchpad/claims/corpus_b3.py`,
both of which are built to REACH the withholding, so the counts here are
counts on them and not rates.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mechc"))


def harnesses():
    """(name, nullary harness) for every case x order of both corpora."""
    import corpus as MECHC  # scratchpad/mechc/corpus.py

    import corpus_b3 as B3

    out = []
    for case in MECHC.CASES:
        for order in MECHC.orders_for(case):
            out.append((f"mechc:{case.name}/{order}",
                        MECHC.build_harness(case, order)))
    for name, decls, assumes, asserts, _note in B3.cases():
        for order in B3.ORDERS:
            out.append((f"b3:{name}/{order}",
                        B3.build_harness(decls, assumes, asserts, order)))
    return out


def collect(path):
    import jax

    jax.config.update("jax_enable_x64", True)
    import stelling
    from stelling import affine as A
    from stelling import propagate as P
    from stelling import solvers as S
    from stelling._jax_compat import transcribe

    out = {"provenance": {"stelling": stelling.__file__,
                          "jax": jax.__version__},
           "runs": {}}
    cfg = S.SolverConfig(timeout_ms=5000)
    for name, h in harnesses():
        cj = transcribe(jax.make_jaxpr(h)())
        p = P.propagate(cj)
        after_prop = [o.status for o in p.obligations]
        offered_affine = [o.index for o in p.obligations
                          if o.status == "unknown"]
        ra, rep = A.refine_propagation(cj, p)
        after_affine = [o.status for o in ra.obligations]
        esc = S.escalate(cj, p, cfg)
        n_inv = 0
        outcomes = []
        for rec in esc.records:
            n_inv += len(rec.invocations)
            outcomes.append([rec.index, rec.outcome])
        out["runs"][name] = {
            "after_propagate": after_prop,
            "offered_affine": offered_affine,
            "after_affine": after_affine,
            "affine_discharged": list(rep.discharged),
            "affine_violated": list(rep.violated),
            "offered_solver": offered_affine,
            "solver_invocations": n_inv,
            "solver_outcomes": outcomes,
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {path}: {len(out['runs'])} runs, "
          f"{sum(r['solver_invocations'] for r in out['runs'].values())} "
          f"solver invocations")


def diff(a_path, b_path):
    A = json.load(open(a_path))
    B = json.load(open(b_path))
    print(f"A = {A['provenance']['stelling']}")
    print(f"B = {B['provenance']['stelling']}")
    newly_affine = newly_solver = 0
    new_aff_discharge = new_aff_violate = 0
    new_solver_inv = 0
    new_solver_decided = 0
    verd_a = verd_b = 0
    rows = []
    for name in sorted(A["runs"]):
        ra, rb = A["runs"][name], B["runs"][name]
        add = sorted(set(rb["offered_affine"]) - set(ra["offered_affine"]))
        if add:
            newly_affine += len(add)
            newly_solver += len(add)
            rows.append(f"{name:52s} newly offered obligations {add}")
            for i in add:
                if rb["after_affine"][i] == "discharged":
                    new_aff_discharge += 1
                if rb["after_affine"][i] == "violated-over-set":
                    new_aff_violate += 1
                for idx, st in rb["solver_outcomes"]:
                    if idx == i and st not in ("unknown", "declined"):
                        new_solver_decided += 1
        new_solver_inv += rb["solver_invocations"] - ra["solver_invocations"]
        verd_a += ra["solver_invocations"]
        verd_b += rb["solver_invocations"]
    for r in rows:
        print(r)
    print()
    print(f"obligations newly offered to the AFFINE refinement: {newly_affine}")
    print(f"obligations newly offered to SOLVER escalation:     {newly_solver}")
    print(f"  of those, newly affine-DISCHARGED:                {new_aff_discharge}")
    print(f"  of those, newly affine-VIOLATED:                  {new_aff_violate}")
    print(f"  of those, decided by a solver (not unknown):      {new_solver_decided}")
    print(f"solver invocations: A {verd_a} -> B {verd_b} "
          f"(delta {new_solver_inv})")


if __name__ == "__main__":
    if sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        collect(sys.argv[1])
