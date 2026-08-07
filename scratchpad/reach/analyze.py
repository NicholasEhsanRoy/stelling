"""Classify every obligation and every query against the oracle.

Primary ledger is PER OBLIGATION (the last corpus in this project scored
per QUERY and turned a measured 24:168 trade into 216:216). The per-query
roll-up is reported alongside, never instead.

The obligation universe comes from the case SOURCE, not from what the
oracle saw execute. It used to be `set(n_exec) | set(n_false)` — and
`n_false`'s keys are a subset of `n_exec`'s, so the universe WAS the
executed set and the `SWALLOWED_UNREACHED` class below could never be
reached. That is not a cosmetic dead branch: an obligation stelling drops
inside a branch no point of the box takes is exactly an obligation the
oracle never executes, so the one class that could have seen it was the
one the loop could not produce. Measured on this corpus, the class is
non-empty as soon as the universe is the source.
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "cases.py")


def source_obligations(path=CASES_PATH):
    """case name -> the lines of its `S(...)` call sites, read from the
    GENERATED source. `gen_cases.py` emits one call per line and every
    case function starts at column 0, so the split needs no parser."""
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing: the obligation universe is read from the "
            f"case source, so regenerate it first with gen_cases.py"
        )
    out, cur = {}, None
    for n, line in enumerate(open(path), 1):
        if line.startswith("def "):
            cur = line[4:].split("(")[0]
            out[cur] = []
        elif line[:1] not in (" ", "\t", "\n", "#") and not line.startswith(
            ("@", ")")
        ):
            cur = None
        elif cur is not None and "S(" in line:
            out[cur].append(n)
    return out


def classify(path, legs=("base", "affine", "inputs-only")):
    d = json.load(open(path))
    src = source_obligations()
    per_obl = []   # (case, leg, line, oracle_exec, oracle_false, st_status, klass)
    per_query = []
    for name, e in sorted(d.items()):
        if "oracle" not in e:
            per_query.append((name, "-", "ORACLE_ERROR", e.get("oracle_error")))
            continue
        n_exec = {int(k): v for k, v in e["oracle"]["n_exec"].items()}
        n_false = {int(k): v for k, v in e["oracle"]["n_false"].items()}
        all_lines = set(n_exec) | set(n_false) | set(src.get(name, ()))
        for leg in legs:
            r = e["legs"].get(leg)
            if r is None or "error" in r:
                per_query.append((name, leg, "STELLING_ERROR",
                                  (r or {}).get("error")))
                continue
            seen = {}
            for o in r["obl"]:
                if o["line"] is not None:
                    seen[o["line"]] = o
            # obligations the oracle saw executed but stelling never listed
            for ln in sorted(all_lines):
                if ln in seen:
                    continue
                ex, fa = n_exec.get(ln, 0), n_false.get(ln, 0)
                klass = "SWALLOWED_FALSE" if fa else (
                    "SWALLOWED_TRUE" if ex else "SWALLOWED_UNREACHED")
                per_obl.append((name, leg, ln, ex, fa, "<absent>", klass))
            for ln, o in sorted(seen.items()):
                ex, fa = n_exec.get(ln, 0), n_false.get(ln, 0)
                st = o["status"]
                if st == "violated-over-set":
                    if ex == 0:
                        klass = "REFUTE_ON_UNREACHABLE"      # UNSOUND
                    elif fa == 0:
                        klass = "REFUTE_CONTRADICTED"        # UNSOUND
                    else:
                        klass = "REFUTE_SOUND"
                elif st == "discharged":
                    klass = "DISCHARGE_CONTRADICTED" if fa else (
                        "DISCHARGE_VACUOUS" if ex == 0 else "DISCHARGE_SOUND")
                else:
                    klass = ("UNKNOWN_ON_UNREACHABLE" if ex == 0 else
                             ("UNKNOWN_ON_FALSE" if fa else "UNKNOWN_ON_TRUE"))
                per_obl.append((name, leg, ln, ex, fa, st, klass))
            # per-query roll-up
            any_false_exec = any(n_false.get(ln, 0) > 0 for ln in all_lines)
            swallowed = [ln for ln in all_lines if ln not in seen
                         and n_exec.get(ln, 0) > 0]
            if r["status"] == "VERIFIED":
                q = "FALSE_VERIFIED" if any_false_exec else "VERIFIED_SOUND"
            elif r["status"] == "REFUTED":
                q = "FALSE_REFUTED" if not any_false_exec else "REFUTED_SOUND"
            elif r["status"] == "DECLINED":
                q = "DECLINED"
            else:
                q = "UNKNOWN"
            per_query.append((name, leg, q, len(swallowed)))
    return per_obl, per_query


def main():
    per_obl, per_query = classify(sys.argv[1])
    print("== PER-OBLIGATION ==")
    c = collections.Counter(r[6] for r in per_obl)
    for k in sorted(c):
        print(f"  {k:28s} {c[k]}")
    print("== PER-QUERY ==")
    c2 = collections.Counter(r[2] for r in per_query)
    for k in sorted(c2):
        print(f"  {k:28s} {c2[k]}")
    print("== UNSOUND OBLIGATION ROWS (sample) ==")
    bad = [r for r in per_obl if r[6] in
           ("REFUTE_ON_UNREACHABLE", "REFUTE_CONTRADICTED",
            "DISCHARGE_CONTRADICTED", "SWALLOWED_FALSE")]
    seen = collections.Counter()
    for r in bad:
        seen[r[6]] += 1
        if seen[r[6]] <= 4:
            print("  ", r)
    print("== FALSE_VERIFIED QUERIES (sample) ==")
    fv = [r for r in per_query if r[2] == "FALSE_VERIFIED"]
    for r in fv[:10]:
        print("  ", r)
    print(f"   total FALSE_VERIFIED rows: {len(fv)}")


if __name__ == "__main__":
    main()
