# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""Probe 3: the model VALUE channel carries a quoted symbol, and that was
enough on its own.

`probe_cvc5_sorts.py` concluded that cvc5's printer had already closed the
VALUE channel and only the NAME channel was open. It missed the shape below
because it named its datatype constructor in ASCII (`mk`) and poisoned a
String SELECTOR value -- a string LITERAL, which is the one place cvc5 really
does escape every separator.

H1 (the conclusion under test): cvc5 escapes every `splitlines()` separator
inside a model VALUE, so the value channel was already closed.
H2 (the rival): a quoted symbol reaches the VALUE field verbatim whenever the
value's SORT or CONSTRUCTOR was declared as one -- no string literal involved.

The measurement runs the DRIVER'S OWN route (`cvc5.InputParser`, SMT_LIB_2_6
string input, `sm.getDeclaredTerms()`, `solver.getValue`) so what is read is
what `_cvc5_driver.main` would have stringified, and reads the VALUE text, not
the name text. Part 2 then hands a truncated prefix of the BASE driver's real
stdout to the BASE parent and asks whether it harvests a model.

Usage:  probe_cvc5_value_channel.py            # part 1 only
        probe_cvc5_value_channel.py --corpse   # part 1 + part 2 (needs a
                                                # PYTHONPATH pointing at a
                                                # PRE-FIX tree, e.g. 0ad22bb)
"""
from __future__ import annotations

import subprocess
import sys

import cvc5

VT = "\x0b"
SEPS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85  "


def _run_script(script: str):
    """The driver's own route, returning (answer, [(name, value)])."""
    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    solver.setOption("produce-models", "true")
    parser = cvc5.InputParser(solver)
    parser.setStringInput(cvc5.InputLanguage.SMT_LIB_2_6, script, "probe3")
    sm = parser.getSymbolManager()
    answer = ""
    while True:
        cmd = parser.nextCommand()
        if cmd.isNull():
            break
        if str(cmd).strip().startswith("(get-model"):
            continue
        result = cmd.invoke(solver, sm)
        if result:
            token = result.strip()
            if token in ("sat", "unsat", "unknown"):
                answer = token
    if answer != "sat":
        return answer, []
    return answer, [
        (str(t), str(solver.getValue(t))) for t in sm.getDeclaredTerms()
    ]


CASES = (
    (
        "uninterpreted sort, quoted name",
        f"(set-logic QF_UF)\n(declare-sort |S{VT}end 1| 0)\n"
        f"(declare-const c |S{VT}end 1|)\n(check-sat)\n",
    ),
    (
        "datatype constructor, quoted name",
        f"(set-logic QF_UFDT)\n(declare-datatypes ((D 0)) (((|c{VT}end 1|))))\n"
        f"(declare-const d D)\n(check-sat)\n",
    ),
    (
        "array of a quoted-name sort",
        f"(set-logic QF_AUFLIA)\n(declare-sort |S{VT}z| 0)\n"
        f"(declare-const a (Array Int |S{VT}z|))\n(check-sat)\n",
    ),
    (
        "CONTROL: the same sort named in ASCII",
        "(set-logic QF_UF)\n(declare-sort S 0)\n(declare-const c S)\n"
        "(check-sat)\n",
    ),
)


def part1() -> int:
    hits = 0
    for label, script in CASES:
        answer, pairs = _run_script(script)
        print(f"{label}: answer={answer}")
        for name, value in pairs:
            in_name = [f"U+{ord(c):04X}" for c in name if c in SEPS]
            in_value = [f"U+{ord(c):04X}" for c in value if c in SEPS]
            print(f"    name  = {name!r}   separators: {in_name or 'NONE'}")
            print(f"    value = {value!r}   separators: {in_value or 'NONE'}"
                  + ("   <<< THE VALUE CHANNEL CARRIES ONE" if in_value else ""))
            hits += bool(in_value)
    print()
    print(f"values carrying a raw separator: {hits} of {len(CASES)} cases "
          f"(the ASCII control is the one that carries none)")
    return hits


def part2() -> None:
    """Hand a truncated prefix of the BASE driver's real stdout to the BASE
    parent. Only meaningful on a PRE-FIX tree: on the fixed tree the writer
    escapes the separator and there is nothing to truncate around."""
    from stelling import solvers

    print(f"\nstelling under test: {solvers.__file__}")
    real = subprocess.run
    solvers._cvc5_wheel_version = lambda: "1.3.4"
    script = (
        "(set-option :produce-models true)\n(set-logic QF_UF)\n"
        f"(declare-sort |S{VT}end 1| 0)\n"
        f"(declare-const c |S{VT}end 1|)\n"
        "(check-sat)\n(get-model)\n"
    )
    child = real(
        [sys.executable, "-m", "stelling._cvc5_driver"],
        input=script, capture_output=True, text=True, timeout=120,
    )
    print(f"real child: rc={child.returncode} stdout={child.stdout!r} "
          f"({len(child.stdout)} chars)")

    def parent(stdout, rc: int):
        """`stdout` is the child's BYTES; a `str` is encoded UTF-8. The parent
        decodes for itself now, so a `str` here is not a child's output."""
        raw = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
        subprocess.run = lambda a, **kw: subprocess.CompletedProcess(
            [], rc, raw, b""
        )
        try:
            return solvers._run_cvc5_wheel(script, 60.0)
        finally:
            subprocess.run = real

    for n in range(1, len(child.stdout)):
        prefix = child.stdout[:n]
        result = parent(prefix, 0)
        if result.answer not in ("sat", "unsat", "unknown"):
            continue
        wrote_terminator = (
            prefix.endswith("\n")
            and any(ln.startswith("end ") for ln in prefix.split("\n")[:-1])
        )
        print(f"  PREFIX {n}: exit 0 -> {result.answer!r} "
              f"values={result.values} opaque={result.nonrational} "
              f"child-wrote-a-terminator-RECORD={wrote_terminator}")
        print(f"     prefix = {prefix!r}")
        print(f"     the reader's lines: {prefix.splitlines()}")


if __name__ == "__main__":
    part1()
    if "--corpse" in sys.argv:
        part2()
