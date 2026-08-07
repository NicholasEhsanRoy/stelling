# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

import sys, traceback
sys.path.insert(0, "/home/nick/MSF/.wt-sz0/lab")
from oracle import sweep
import cases as C
from stelling.preconditions import check

MODES = ["inputs-only", "all"]
REFINES = [None, "affine"]

print(f"{'case':26s} {'mode':11s} {'refine':7s} {'stelling':9s} "
      f"{'oracle: pts/admitted/violating':34s} verdict")
rows = []
for c in C.CASES:
    n, adm, bad, wit = sweep(c["specs"], c["assume"], c["assertf"])
    oracle = f"{n}/{adm}/{bad}"
    for mode in MODES:
        for refine in REFINES:
            try:
                v = check(c["harness"], vacuity_mode=mode, refine=refine)
                st = v.status
            except Exception as e:
                st = f"RAISE:{type(e).__name__}"
            bad_flag = ""
            if st == "VERIFIED" and bad > 0:
                bad_flag = "  *** WRONG VERIFIED ***"
            if st == "REFUTED" and bad == 0 and adm > 0:
                bad_flag = "  *** suspicious REFUTED ***"
            print(f"{c['name']:26s} {mode:11s} {str(refine):7s} {st:9s} "
                  f"{oracle:34s}{bad_flag}")
            rows.append((c["name"], mode, refine, st, n, adm, bad, bool(bad_flag)))
    if wit:
        print(f"{'':26s} oracle witnesses: {wit}")
import json
json.dump(rows, open(sys.argv[1], "w"), indent=0)
