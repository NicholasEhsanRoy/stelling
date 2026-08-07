"""Deep oracle re-check of exactly the directions that matter.

For every case stelling now calls VERIFIED (any leg), resample the
declared box far more densely and look for ANY point at which ANY
obligation is executed and false. For every case stelling now REFUTES,
confirm the oracle finds a violating point.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C
import oracle as O

d = json.load(open(sys.argv[1]))
n = int(sys.argv[2])
bad_v, bad_r, nv, nr = [], [], 0, 0
for name, e in sorted(d.items()):
    if "legs" not in e:
        continue
    st = {r.get("status") for r in e["legs"].values() if "error" not in r}
    if not ({"VERIFIED", "REFUTED"} & st):
        continue
    fn, meta = C.CASES[name]
    orc = O.run_case(fn, n_random=n, seed=98765)
    violating = {ln: c for ln, c in orc["n_false"].items() if c}
    if "VERIFIED" in st:
        nv += 1
        if violating:
            bad_v.append((name, violating, orc["n_points"]))
    if "REFUTED" in st:
        nr += 1
        if not violating:
            bad_r.append((name, orc["n_points"]))
print(f"points per case: >= {n}")
print(f"VERIFIED cases checked: {nv}   with a violating point: {len(bad_v)}")
for b in bad_v[:10]:
    print("   FALSE VERIFIED:", b)
print(f"REFUTED cases checked: {nr}   with NO violating point: {len(bad_r)}")
for b in bad_r[:10]:
    print("   FALSE REFUTED:", b)
