"""Order-dependence and byte-identity checks over one or two MECHC ledgers."""
from __future__ import annotations

import json
import sys

S = {"discharged": "D", "unknown": "U", "violated-over-set": "V"}


def vec(run):
    if "raised" in run:
        return "RAISED"
    return (
        "".join(S[o["status"]] for o in run["nonvacuity"])
        + "|"
        + "".join(S[o["status"]] for o in run["obligations"])
    )


def order_table(path, label):
    d = json.load(open(path))
    print(f"--- {label}: {d['provenance']['stelling']} jax {d['provenance']['jax']}")
    rows = []
    for name, c in sorted(d["cases"].items()):
        keys = sorted(c["runs"])
        for refine in ("none", "affine"):
            b = f"before/{refine}"
            a = f"after/{refine}"
            if b not in keys or a not in keys:
                continue
            vb = vec(c["runs"][b]["constrain"])
            va = vec(c["runs"][a]["constrain"])
            rows.append((name, refine, vb, va, vb == va))
    bad = [r for r in rows if not r[4]]
    for r in rows:
        mark = "  " if r[4] else "!!"
        print(f"{mark} {r[0]:38s} {r[1]:7s} before={r[2]:6s} after={r[3]:6s}")
    print(f"    order-dependent rows: {len(bad)} of {len(rows)}")
    return {(r[0], r[1]): (r[2], r[3]) for r in rows}, bad


def identity(a_path, b_path, names):
    """Full byte-identity (statuses, DETAILS, notes, assumptions, coverage)
    over the named cases and over every inert-mode run."""
    A, B = json.load(open(a_path)), json.load(open(b_path))
    diffs = []
    checked = 0
    for name in sorted(A["cases"]):
        for key in sorted(A["cases"][name]["runs"]):
            for mode in ("constrain", "inert"):
                if mode == "constrain" and name not in names:
                    continue
                ra = A["cases"][name]["runs"][key][mode]
                rb = B["cases"][name]["runs"][key][mode]
                checked += 1
                if json.dumps(ra, sort_keys=True) != json.dumps(
                    rb, sort_keys=True
                ):
                    diffs.append((name, key, mode))
    print(f"byte-identity: {checked} runs compared, {len(diffs)} differ")
    for d in diffs:
        print("   DIFF", d)
    return diffs


if __name__ == "__main__":
    base, branch = sys.argv[1], sys.argv[2]
    print("=== ORDER DEPENDENCE, baseline ===")
    order_table(base, "base")
    print()
    print("=== ORDER DEPENDENCE, branch ===")
    order_table(branch, "branch")
    print()
    # cases with NO assume at all, and cases whose every assume is CERTIFIED
    NO_ASSUME = {"no_assume_interval", "no_assume_affine"}
    CERTIFIED = {
        "certified_input_assume",
        "certified_assume_definite_violation",
        "two_obligations_certified",
        "f8_definitely_true_assume",
        "harmless_relational",
    }
    print("=== BYTE-IDENTITY: no-assume + all-certified (constrain), and "
          "EVERY case in inert mode ===")
    identity(base, branch, NO_ASSUME | CERTIFIED)
