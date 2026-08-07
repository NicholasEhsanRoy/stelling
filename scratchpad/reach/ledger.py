"""Paired baseline/after ledger. PER OBLIGATION first, per query second."""
import collections, json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyze import classify
from roles import roles

R = roles()
b_o, b_q = classify(sys.argv[1])
a_o, a_q = classify(sys.argv[2])
D = json.load(open(sys.argv[1]))

bo = {(c, l, ln): (ex, fa, st, kl) for (c, l, ln, ex, fa, st, kl) in b_o}
ao = {(c, l, ln): (ex, fa, st, kl) for (c, l, ln, ex, fa, st, kl) in a_o}
bq = {(c, l): (q, n) for (c, l, q, n) in b_q}
aq = {(c, l): (q, n) for (c, l, q, n) in a_q}

print("== PER-OBLIGATION MOVES (status before -> after) ==")
moves = collections.Counter()
buckets = collections.Counter()
for k in sorted(set(bo) | set(ao)):
    b = bo.get(k); a = ao.get(k)
    bs = b[2] if b else "<absent>"; as_ = a[2] if a else "<absent>"
    if bs == as_:
        continue
    bk = b[3] if b else "-"; ak = a[3] if a else "-"
    moves[(bs, as_, bk, ak)] += 1
    guard = D[k[0]]["meta"]["guard"]
    buckets[(bs, as_, guard, R.get(k[2], "?"))] += 1
for k in sorted(moves):
    print(f"  {moves[k]:5d}  {k[0]:18s} -> {k[1]:18s}   [{k[2]} -> {k[3]}]")
print("== MOVED OBLIGATIONS BY GUARD/ROLE ==")
for k in sorted(buckets):
    print(f"  {buckets[k]:5d}  {k[0]:18s} -> {k[1]:12s}  guard={k[2]:12s} role={k[3]}")
print("== PER-QUERY MOVES ==")
qm = collections.Counter()
for k in sorted(set(bq) | set(aq)):
    b = bq.get(k, ("<absent>", 0))[0]; a = aq.get(k, ("<absent>", 0))[0]
    if b != a:
        qm[(b, a)] += 1
for k in sorted(qm):
    print(f"  {qm[k]:5d}  {k[0]:16s} -> {k[1]}")
print("== INVARIANTS ==")
print("  obligations moving violated-over-set -> discharged:",
      sum(v for k, v in moves.items() if k[0] == "violated-over-set" and k[1] == "discharged"))
print("  obligations moving unknown -> discharged:",
      sum(v for k, v in moves.items() if k[0] == "unknown" and k[1] == "discharged"))
print("  obligations moving unknown -> violated-over-set:",
      sum(v for k, v in moves.items() if k[0] == "unknown" and k[1] == "violated-over-set"))
print("  obligations moving discharged -> anything:",
      sum(v for k, v in moves.items() if k[0] == "discharged"))
print("  queries moving INTO VERIFIED:",
      sum(v for k, v in qm.items() if "VERIFIED" in k[1]))
print("  queries moving INTO REFUTED from UNKNOWN:",
      sum(v for k, v in qm.items() if k[0] == "UNKNOWN" and "REFUTED" in k[1]))
