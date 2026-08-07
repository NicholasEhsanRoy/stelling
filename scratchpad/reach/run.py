"""Run stelling over the corpus, join with the oracle, emit a per-obligation
and per-query ledger as JSON. Never consults stelling for ground truth."""
from __future__ import annotations

import json
import os
import sys
import traceback

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cases as CASES_MOD  # noqa: E402
import oracle as ORACLE  # noqa: E402

from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402

CASES_PATH = os.path.abspath(CASES_MOD.__file__).replace(".pyc", ".py")


class JaxCtl:
    cond = staticmethod(lambda p, t, f, *o: jax.lax.cond(p, t, f, *o))
    switch = staticmethod(lambda i, b, *o: jax.lax.switch(i, b, *o))
    scan = staticmethod(lambda f, init, xs: jax.lax.scan(f, init, xs))
    while_loop = staticmethod(lambda c, b, i: jax.lax.while_loop(c, b, i))


def _A(name, shape, dtype, bounds):
    return any_array(shape, dtype, bounds)


def _S(pred):
    return assert_(pred)


def case_line(source_info):
    """The cases.py line of the S() call site: innermost cases.py frame.
    source_info is ordered deepest-frame-first."""
    for fr in source_info:
        path = fr.split(" (")[0]
        f, _, ln = path.rpartition(":")
        if os.path.abspath(f) == CASES_PATH:
            try:
                return int(ln)
            except ValueError:
                return None
    return None


def run_stelling(fn, **kw):
    def harness():
        return fn(_A, _S, JaxCtl)
    try:
        v = check(harness, **kw)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-800:]}
    return {
        "status": v.status,
        "n_obl": len(v.obligations),
        "obl": [{"index": o.index, "status": o.status,
                 "line": case_line(o.source_info), "detail": o.detail[:160]}
                for o in v.obligations],
        "notes": list(v.notes),
    }


def main():
    out_path = sys.argv[1]
    n_random = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    only = sys.argv[3] if len(sys.argv) > 3 else None
    legs = [
        ("base", dict(vacuity_mode="all")),
        ("affine", dict(vacuity_mode="all", refine="affine")),
        ("inputs-only", dict(vacuity_mode="inputs-only")),
    ]
    results = {}
    names = sorted(CASES_MOD.CASES)
    if only:
        names = [n for n in names if only in n]
    for i, name in enumerate(names):
        fn, meta = CASES_MOD.CASES[name]
        try:
            orc = ORACLE.run_case(fn, n_random=n_random, seed=17)
        except Exception as e:  # noqa: BLE001
            results[name] = {"meta": meta, "oracle_error": f"{type(e).__name__}: {e}"}
            continue
        entry = {"meta": meta,
                 "oracle": {"n_points": orc["n_points"],
                            "n_exec": {str(k): v for k, v in orc["n_exec"].items()},
                            "n_false": {str(k): v for k, v in orc["n_false"].items()}},
                 "legs": {}}
        for leg, kw in legs:
            entry["legs"][leg] = run_stelling(fn, **kw)
        results[name] = entry
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(names)}", flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out_path}: {len(results)} cases")


if __name__ == "__main__":
    main()
