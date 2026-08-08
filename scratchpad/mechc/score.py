"""Score two MECHC ledgers against each other, per obligation."""
from __future__ import annotations

import json
import sys

SHORT = {"discharged": "D", "unknown": "U", "violated-over-set": "V"}


def statuses(run):
    """(kind, index) -> status for one (case, order, refine, mode) run."""
    if "raised" in run:
        return {("raise", 0): "RAISED"}
    out = {}
    for o in run.get("obligations", []):
        out[("ob", o["index"])] = o["status"]
    for o in run.get("nonvacuity", []):
        out[("nv", o["index"])] = o["status"]
    return out


def load(p):
    with open(p) as f:
        return json.load(f)


def main(a_path, b_path):
    A, B = load(a_path), load(b_path)
    print(f"A = {A['provenance']}")
    print(f"B = {B['provenance']}")
    moves = []
    same = 0
    inert_diff = []
    print()
    hdr = f"{'case':38s} {'order/refine':18s} {'oracle':>18s}  base -> branch"
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(A["cases"]):
        ca, cb = A["cases"][name], B["cases"][name]
        orc = ca.get("oracle", {})
        for key in sorted(ca["runs"]):
            ra, rb = ca["runs"][key], cb["runs"][key]
            for mode in ("constrain", "inert"):
                sa, sb = statuses(ra[mode]), statuses(rb[mode])
                assert set(sa) == set(sb), (name, key, mode, set(sa) ^ set(sb))
                for k in sorted(sa):
                    kind, idx = k
                    if sa[k] == sb[k]:
                        same += 1
                        continue
                    if mode == "inert":
                        inert_diff.append((name, key, k, sa[k], sb[k]))
                    # oracle column: admitted points and violating-admitted
                    oi = orc.get("obligations", [])
                    n_adm = orc.get("n_admitted")
                    j = idx if kind == "nv" else idx + len(
                        [1 for kk in sa if kk[0] == "nv"]
                    )
                    viol = oi[j]["violating_admitted"] if j < len(oi) else "?"
                    ocol = f"adm={n_adm} viol={viol}"
                    moves.append(
                        (name, key, mode, kind, idx, sa[k], sb[k], n_adm, viol)
                    )
                    print(
                        f"{name:38s} {key + '/' + mode:18s} {ocol:>18s}  "
                        f"{SHORT.get(sa[k], sa[k])} -> {SHORT.get(sb[k], sb[k])}"
                        f"   [{kind}#{idx}]"
                    )
    print()
    print(f"obligation-runs unchanged: {same}")
    print(f"obligation-runs moved:     {len(moves)}")
    kinds = {}
    for m in moves:
        kinds[(m[5], m[6])] = kinds.get((m[5], m[6]), 0) + 1
    for k, v in sorted(kinds.items()):
        print(f"  {k[0]} -> {k[1]}: {v}")
    empty = [m for m in moves if m[7] == 0]
    real = [m for m in moves if m[7] != 0 and m[8] not in (0, "?")]
    other = [m for m in moves if m not in empty and m not in real]
    print()
    print(f"  (a) region EMPTY by oracle  (wrong REFUTED closed): {len(empty)}")
    print(f"  (b) genuine violating point (REAL LOSS):            {len(real)}")
    print(f"  (c) neither:                                        {len(other)}")
    for m in other:
        print(f"      {m}")
    print()
    print(f"inert-mode diffs (must be 0): {len(inert_diff)}")
    for d in inert_diff:
        print("   ", d)
    # detail/note byte-identity on the runs that must not move
    return moves


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
