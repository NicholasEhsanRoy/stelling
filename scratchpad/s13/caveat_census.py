# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""Caveat census over the 288-harness assume-scope corpus: how many
discharges carry the may-be-vacuous line, before and after."""
import itertools, os, sys, json
import jax
jax.config.update("jax_enable_x64", True)
# the 288-harness generator lives with the audit, not in this repo:
sys.path.insert(0, os.environ.get("SWEEP_DIR", "."))
import sweep_assume_scope as S
from stelling.preconditions import check
from stelling.propagate import UNCERTIFIED_PRECONDITION_PREFIX

rows = {}
counts = {"VERIFIED": 0, "caveated": 0, "clean": 0}
for carrier, ndecl, tail, aset, (expr, thr), order in itertools.product(
    ["jit", "cond", "custom_jvp", "top"], [2, 3], [0, 1, 2],
    list(S.ASSUME_SETS), [("sub01", 0.0), ("add01", -5.0)],
    ["assume_last", "assume_first"],
):
    h, spec = S.build(carrier, ndecl, tail, aset, expr, thr, order)
    if h is None:
        continue
    key = "|".join(str(k) for k in (carrier, ndecl, tail, aset, expr, thr, order))
    try:
        v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=5000)
    except Exception as e:
        rows[key] = f"RAISED {type(e).__name__}"
        continue
    caveat = any(a.startswith(UNCERTIFIED_PRECONDITION_PREFIX) for a in v.stamp.assumptions)
    rows[key] = f"{v.status}|caveat={caveat}"
    if v.status == "VERIFIED":
        counts["VERIFIED"] += 1
        counts["caveated" if caveat else "clean"] += 1
print(counts)
with open(sys.argv[1], "w") as fh:
    json.dump(rows, fh, indent=0, sort_keys=True)
