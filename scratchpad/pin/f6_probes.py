# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""F6: per-query probe counts for the TWO witness searches, and whether
any query pays for both."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
import jax
jax.config.update("jax_enable_x64", True)
import stelling.propagate as P
from stelling.harness import trace
import corpus_pin

worst = (0, None)
both = []
rows = 0
for name, h in corpus_pin.cases():
    try:
        closed = trace(h)
    except Exception:
        continue
    for sem in ("real", "ieee"):
        for mode in ("constrain", "inert"):
            P._instr_reset()
            try:
                P.propagate(closed, semantics=sem, assume_mode=mode)
            except Exception:
                continue
            rows += 1
            c, r = P._INSTR["cert_probes"], P._INSTR["reach_probes"]
            if c and r:
                both.append((name, sem, mode, c, r))
            if c + r > worst[0]:
                worst = (c + r, (name, sem, mode, c, r))
print(f"runs measured: {rows}")
print(f"runs paying for BOTH searches: {len(both)}")
for b in both[:10]:
    print("   ", b)
print(f"worst combined probe count: {worst[0]}  at {worst[1]}")
