# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Record every _conjunct_certainly_true call on the tip branch."""
import sys, math, json
sys.path.insert(0, "/home/nick/MSF/.wt-sz0/lab")
from stelling.propagate import _Propagator
orig = _Propagator._conjunct_certainly_true
REC = []
def wrapped(self, atom):
    shape = tuple(getattr(getattr(atom, "aval", None), "shape", ()) or ())
    n = math.prod(shape) if shape else 1
    ans = orig(self, atom)
    REC.append(dict(shape=str(shape), size=n, size0=(n == 0),
                    dtype=str(getattr(atom.aval, "dtype", None)),
                    sem=self.semantics, answer=bool(ans)))
    return ans
_Propagator._conjunct_certainly_true = wrapped

import cases as C
from stelling.harness import trace
from stelling.propagate import propagate
for c in C.CASES:
    for sem in ("real", "ieee"):
        try:
            propagate(trace(c["harness"]), semantics=sem)
        except Exception as e:
            pass
print("calls:", len(REC))
print("size0 calls:", sum(r["size0"] for r in REC))
for r in REC:
    if r["size0"]:
        print("  SIZE0:", r)
json.dump(REC, open("/home/nick/MSF/.wt-sz0/lab/ctt_calls.json", "w"), indent=0)
