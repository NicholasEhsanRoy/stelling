# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Child-process driver for the cvc5 wheel transport.

``stelling.solvers`` runs this module as ``python -m stelling._cvc5_driver``
with an SMT-LIB2 script on stdin. It feeds the script through the wheel's
own SMT-LIB2 parser (``cvc5.InputParser``, SMT_LIB_2_6 string input) —
the ``get-model`` command is realized through the model API
(``getValue``), everything else is invoked as parsed — and reports on
stdout in a line protocol::

    version <backend version>
    answer <sat|unsat|unknown>
    value <name> <exact rational p/q>        (sat only, one per declared const)
    opaque <name> <raw term text>            (sat, non-rational model value)
    end

Why a child process at all, measured on cvc5 1.3.4: the wheel's
``checkSat`` holds the GIL for the entire check, so an in-process thread
guard can never fire, and the script-level ``:tlimit`` does not reliably
preempt the coverings solver — the parent's subprocess timeout is the
wall-clock guard that actually binds. Any internal failure prints
``error <reason>`` and exits 0; the parent degrades it to UNKNOWN (the
guard rule: solver failures are quoted reasons, never crashes).

Stdlib-only at import time; cvc5 is imported inside :func:`main` via
``stelling._optional``.
"""

from __future__ import annotations

import sys

from stelling._optional import require


def main() -> int:
    out = sys.stdout
    try:
        cvc5 = require("cvc5")
        script = sys.stdin.read()
        tm = cvc5.TermManager()
        solver = cvc5.Solver(tm)
        version = solver.getVersion()
        if isinstance(version, bytes):
            version = version.decode("utf-8", "replace")
        print(f"version {version}", file=out)
        parser = cvc5.InputParser(solver)
        parser.setStringInput(
            cvc5.InputLanguage.SMT_LIB_2_6, script, "stelling-escalation"
        )
        sm = parser.getSymbolManager()
        answer = ""
        while True:
            cmd = parser.nextCommand()
            if cmd.isNull():
                break
            if str(cmd).strip().startswith("(get-model"):
                continue  # realized via the model API below
            result = cmd.invoke(solver, sm)
            if result:
                token = result.strip()
                if token in ("sat", "unsat", "unknown"):
                    answer = token
        if not answer:
            print("error script produced no check-sat answer", file=out)
            return 0
        print(f"answer {answer}", file=out)
        if answer == "sat":
            for term in sm.getDeclaredTerms():
                value = solver.getValue(term)
                if value.isRealValue():
                    print(f"value {term} {value.getRealValue()}", file=out)
                else:
                    raw = str(value).replace("\n", " ")
                    print(f"opaque {term} {raw}", file=out)
        print("end", file=out)
        return 0
    except Exception as e:  # noqa: BLE001 — the parent quotes this, never crashes
        msg = str(e).replace("\n", " ")
        print(f"error {type(e).__name__}: {msg}", file=out)
        return 0


if __name__ == "__main__":
    sys.exit(main())
