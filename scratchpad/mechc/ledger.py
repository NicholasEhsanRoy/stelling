"""Emit the per-obligation ledger for the MECHC corpus, as JSON on stdout.

Run under `PYTHONPATH=<tree>/src` — the tree under measurement. The
resolved `stelling.__file__` and `jax.__version__` are stamped into the
output so no two ledgers can be compared without the provenance visible.
"""
from __future__ import annotations

import json
import sys
import traceback

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import jax  # noqa: E402

import corpus as C  # noqa: E402


def render(case, order, refine):
    from stelling import affine as A
    from stelling import propagate as P
    from stelling._jax_compat import transcribe

    h = C.build_harness(case, order)
    try:
        cj = transcribe(jax.make_jaxpr(h)())
    except Exception as e:  # noqa: BLE001
        return {"error": f"trace: {type(e).__name__}: {e}"}
    rows = {}
    for mode in ("constrain", "inert"):
        try:
            p = P.propagate(cj, assume_mode=mode)
            if refine == "affine":
                p, _rep = A.refine_propagation(cj, p)
        except Exception as e:  # noqa: BLE001
            rows[mode] = {"raised": f"{type(e).__name__}: {e}"}
            continue
        rows[mode] = {
            "obligations": [
                {"index": o.index, "status": o.status, "detail": o.detail}
                for o in p.obligations
            ],
            "nonvacuity": [
                {"index": o.index, "status": o.status, "detail": o.detail}
                for o in p.nonvacuity_checks
            ],
            "notes": list(p.notes),
            "assumptions": list(p.assumptions),
            "assume_dropped": p.assume_dropped,
            "coverage_constrained": p.coverage.constrained,
            "coverage_inert": p.coverage.inert,
            "dropped_conjuncts": p.coverage.dropped_conjuncts,
        }
    return rows


def main():
    jax.config.update("jax_enable_x64", True)
    import stelling

    out = {
        "provenance": {
            "stelling": stelling.__file__,
            "jax": jax.__version__,
            "python": sys.executable,
        },
        "cases": {},
    }
    want_oracle = "--oracle" in sys.argv
    for case in C.CASES:
        entry = {"note": case.note, "runs": {}}
        for order in C.orders_for(case):
            for refine in (None, "affine"):
                key = f"{order}/{refine or 'none'}"
                try:
                    entry["runs"][key] = render(case, order, refine)
                except Exception:  # noqa: BLE001
                    entry["runs"][key] = {"error": traceback.format_exc()}
        if want_oracle:
            entry["oracle"] = C.oracle(case)
        out["cases"][case.name] = entry
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
