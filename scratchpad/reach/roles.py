"""line -> role map for cases.py, by source shape (no stelling involved)."""
import os, re
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.py")

def roles():
    m = {}
    for i, line in enumerate(open(P), 1):
        if "S(" not in line:
            continue
        s = line.strip()
        if s.startswith("outs.append(S("):
            m[i] = "top"
        elif s.startswith("yb ="):
            m[i] = "yes"
        elif s.startswith("nb ="):
            m[i] = "no"
        elif re.match(r"b\d =", s):
            m[i] = "switch-" + s[1]
        elif s.startswith("inner_y"):
            m[i] = "inner-yes"
        elif s.startswith("inner_n"):
            m[i] = "inner-no"
        elif s.startswith("outer_n"):
            m[i] = "outer-no"
        elif s.startswith("body ="):
            m[i] = "scan-body"
        elif "return (i + 1" in s:
            m[i] = "while-body"
        elif s.startswith("r = C.cond"):
            m[i] = "cond-in-scan-body"
        elif s.startswith("yb = lambda v: C.scan"):
            m[i] = "scan-in-cond-body"
        elif "jax.jit" in s:
            m[i] = "jit-body"
        else:
            m[i] = "?" + s[:40]
    return m
