# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""Probe: can real cvc5 1.3.4 emit a splitlines() separator in a model value?

Measures ``str(solver.getValue(term))`` -- exactly what the driver stringifies
at ``_cvc5_driver.main`` -- for String sorts carrying each separator, and for
algebraic/irrational Real values.
"""
import sys

import cvc5

SEPS = {
    "\\n": "\n", "\\r": "\r", "\\x0b": "\x0b", "\\x0c": "\x0c",
    "\\x1c": "\x1c", "\\x1d": "\x1d", "\\x1e": "\x1e",
    "\\x85": "\x85", "\\u2028": " ", "\\u2029": " ",
}

# what str.splitlines() actually breaks on, measured not recalled
measured = [c for c in map(chr, range(0x110000)) if len(("a" + c + "b").splitlines()) > 1]
print("splitlines() separator set (measured):",
      " ".join(f"U+{ord(c):04X}" for c in measured))
print()

for label, ch in SEPS.items():
    tm = cvc5.TermManager()
    s = cvc5.Solver(tm)
    s.setOption("produce-models", "true")
    s.setLogic("QF_S")
    x = tm.mkConst(tm.getStringSort(), "x0")
    lit = tm.mkString("a" + ch + "b")
    s.assertFormula(tm.mkTerm(cvc5.Kind.EQUAL, x, lit))
    r = s.checkSat()
    v = s.getValue(x)
    raw = str(v)
    sanitised = raw.replace("\n", " ")
    n_writer = len(sanitised.split("\n"))
    n_reader = len(sanitised.splitlines())
    print(f"{label:9s} sat={r} str(value)={raw!r} "
          f"isReal={v.isRealValue()} "
          f"after replace(): writer_lines={n_writer} reader_lines={n_reader} "
          f"{'*** MISMATCH ***' if n_writer != n_reader else ''}")

print()
# algebraic / irrational Real: what does an opaque arith value look like?
tm = cvc5.TermManager()
s = cvc5.Solver(tm)
s.setOption("produce-models", "true")
s.setLogic("QF_NRA")
x = tm.mkConst(tm.getRealSort(), "x0")
two = tm.mkReal(2)
s.assertFormula(tm.mkTerm(cvc5.Kind.EQUAL, tm.mkTerm(cvc5.Kind.MULT, x, x), two))
s.assertFormula(tm.mkTerm(cvc5.Kind.GT, x, tm.mkReal(0)))
print("sqrt2 sat=", s.checkSat())
v = s.getValue(x)
print("  str(value)=", repr(str(v)), "isRealValue=", v.isRealValue())
sys.exit(0)
