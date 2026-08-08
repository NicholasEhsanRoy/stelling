# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""Probe 2: sweep cvc5 model-value TEXT for raw splitlines() separators.

Covers the routes the brief names -- string sorts, unicode escapes,
define-fun bodies (lambda/array values), plus datatypes, sequences,
uninterpreted sorts -- and, separately, the SYMBOL NAME channel
(``|quoted symbol|``), because the driver stringifies the term as well as
its value.
"""
import cvc5

SEPS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85  "


def scan(label, text):
    hits = [f"U+{ord(c):04X}" for c in text if c in SEPS]
    print(f"{label:34s} raw-seps={hits or 'NONE'}  text={text[:110]!r}")


def fresh(logic):
    tm = cvc5.TermManager()
    s = cvc5.Solver(tm)
    s.setOption("produce-models", "true")
    s.setLogic(logic)
    return tm, s


# --- 1. array / lambda value (the shape that motivates replace("\n"," ")) ----
tm, s = fresh("QF_ALIA")
arr = tm.mkConst(tm.mkArraySort(tm.getIntegerSort(), tm.getIntegerSort()), "a0")
for i in range(6):
    s.assertFormula(
        tm.mkTerm(cvc5.Kind.EQUAL,
                  tm.mkTerm(cvc5.Kind.SELECT, arr, tm.mkInteger(i)),
                  tm.mkInteger(i * 7))
    )
print("array sat=", s.checkSat())
scan("array value", str(s.getValue(arr)))

# --- 2. uninterpreted sort ---------------------------------------------------
tm, s = fresh("QF_UF")
u = tm.mkUninterpretedSort("U")
c = tm.mkConst(u, "u0")
print("uf sat=", s.checkSat())
scan("uninterpreted value", str(s.getValue(c)))

# --- 3. datatype -------------------------------------------------------------
tm, s = fresh("QF_UFDTLIA")
dt = tm.mkDatatypeDecl("Pair")
ctor = tm.mkDatatypeConstructorDecl("mk")
ctor.addSelector("fst", tm.getIntegerSort())
ctor.addSelector("snd", tm.getStringSort())
dt.addConstructor(ctor)
sort = s.declareDatatype("Pair", ctor)
p = tm.mkConst(sort, "p0")
sel = sort.getDatatype()[0][1].getTerm()
s.assertFormula(tm.mkTerm(cvc5.Kind.EQUAL,
                          tm.mkTerm(cvc5.Kind.APPLY_SELECTOR, sel, p),
                          tm.mkString("a\x0bb")))
print("dt sat=", s.checkSat())
scan("datatype value w/ VT string", str(s.getValue(p)))

# --- 4. sequence of strings holding separators -------------------------------
tm, s = fresh("QF_SLIA")
q = tm.mkConst(tm.getStringSort(), "s0")
s.assertFormula(tm.mkTerm(cvc5.Kind.EQUAL, q,
                          tm.mkString("".join(SEPS) + "end 1")))
print("seq sat=", s.checkSat())
scan("string of ALL separators", str(s.getValue(q)))

# --- 5. the SYMBOL NAME channel: |quoted symbol| through the real parser -----
for name, ch in (("VT", "\x0b"), ("FF", "\x0c"), ("NL", "\n"), ("NEL", "\x85")):
    tm, s = fresh("QF_LRA")
    script = f"(declare-const |a{ch}b| Real)\n(assert (= |a{ch}b| 3.0))\n(check-sat)\n"
    try:
        parser = cvc5.InputParser(s)
        parser.setStringInput(cvc5.InputLanguage.SMT_LIB_2_6, script, "probe")
        sm = parser.getSymbolManager()
        while True:
            cmd = parser.nextCommand()
            if cmd.isNull():
                break
            cmd.invoke(s, sm)
        terms = sm.getDeclaredTerms()
        for t in terms:
            scan(f"quoted symbol name [{name}]", str(t))
    except Exception as e:  # noqa: BLE001
        print(f"quoted symbol name [{name}]: parser REFUSED: "
              f"{type(e).__name__}: {str(e)[:90]!r}")
