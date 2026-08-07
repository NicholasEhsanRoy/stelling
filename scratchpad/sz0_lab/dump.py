# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.insert(0, "/home/nick/MSF/.wt-sz0/lab")
import cases as C
from stelling.harness import trace
from stelling.propagate import propagate
which = sys.argv[1:] or ["A1_size0_right"]
for c in C.CASES:
    if c["name"] not in which:
        continue
    cj = trace(c["harness"])
    print("=====", c["name"])
    for e in cj.jaxpr.eqns:
        print("  ", e.primitive,
              [(getattr(v,'id',None), tuple(v.aval.shape), v.aval.dtype) for v in e.invars],
              "->",
              [(getattr(v,'id',None), tuple(v.aval.shape), v.aval.dtype) for v in e.outvars])
    p = propagate(cj)
    print("  status-ish notes:")
    for n in p.notes:
        print("    -", n)
