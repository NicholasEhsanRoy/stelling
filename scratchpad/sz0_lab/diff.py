# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

import json, sys, collections
b = json.load(open(sys.argv[1])); a = json.load(open(sys.argv[2]))
kb = {tuple(map(str, r[:3])): r for r in b}
ka = {tuple(map(str, r[:3])): r for r in a}
moves = collections.Counter()
print(f"{'case/mode/refine':52s} {'before':38s} -> {'after':10s} violating")
for k in kb:
    rb, ra = kb[k], ka[k]
    if rb[3] != ra[3]:
        moves[(str(rb[3]).split(':')[0], str(ra[3]))] += 1
        print(f"{'/'.join(k):52s} {str(rb[3]):38s} -> {str(ra[3]):10s} {rb[6]}")
print()
for (x, y), n in sorted(moves.items()):
    print(f"  {x} -> {y}: {n}")
print("rows total:", len(kb), " unchanged:", sum(1 for k in kb if kb[k][3] == ka[k][3]))
# soundness ledger
bad = [k for k in ka if ka[k][3] == "VERIFIED" and ka[k][6] > 0]
print("post-fix VERIFIED with a violating admitted point:", bad)
unsound_cost = [k for k in kb if kb[k][3] == "VERIFIED" and ka[k][3] != "VERIFIED" and kb[k][6] == 0]
print("VERIFIED->non-VERIFIED moves that were SOUND before:", unsound_cost)
into = [k for k in kb if kb[k][3] != "VERIFIED" and ka[k][3] == "VERIFIED"]
print("moves INTO VERIFIED:", into)
