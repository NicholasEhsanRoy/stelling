# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Sweep through propagate() directly: both semantics x both assume_modes."""
import sys, json
sys.path.insert(0, "/home/nick/MSF/.wt-sz0/lab")
from oracle import sweep
import cases as C
from stelling.harness import trace
from stelling.propagate import propagate

rows = []
print(f"{'case':26s} {'sem':6s} {'mode':10s} {'obligation statuses':32s} oracle")
for c in C.CASES:
    n, adm, bad, wit = sweep(c["specs"], c["assume"], c["assertf"], n_random=2000)
    for sem in ("real", "ieee"):
        for am in ("constrain", "inert"):
            cj = trace(c["harness"])
            try:
                p = propagate(cj, semantics=sem, assume_mode=am)
                sts = tuple(o.status for o in p.obligations)
            except Exception as e:
                sts = (f"RAISE:{type(e).__name__}",)
            flag = ""
            if all(s == "discharged" for s in sts) and sts and bad > 0:
                flag = "  *** WRONG discharged ***"
            print(f"{c['name']:26s} {sem:6s} {am:10s} {str(sts):32s} "
                  f"{n}/{adm}/{bad}{flag}")
            rows.append((c["name"], sem, am, list(sts), n, adm, bad, bool(flag)))
json.dump(rows, open(sys.argv[1], "w"), indent=0)
