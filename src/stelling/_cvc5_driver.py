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
    end <count of value+opaque lines written>

``end <count>`` is the terminator, and the parent requires it to be the
**last line** of stdout and to state the number of model lines the parent
actually parsed. A bare token would only say "the driver reached its last
statement"; the count says "and it wrote exactly this much model", which is
what *complete* has to mean if a crashed or truncated run is to be refused
(``solvers._run_cvc5_wheel``). Driver and parent ship in the same package
and are read together — change one and you must change the other; a
mismatch degrades every run to UNKNOWN with the terminator quoted, which
is the safe direction but is still a break. **That is a claim about the
PAIR, and its two halves are not equal.** A stale DRIVER writing past this
whitelist is caught by the parent for nine of the ten separators and not
for the tenth, which is ``\\n``, the protocol's own record boundary
(measured, real children, real bytes: ``scratchpad/probe_cvc5_backstop.py``
part A, and
``tests/test_solver_audit_findings.py::test_f4wheel3_the_reader_now_refuses_nine_of_the_ten_separators``).
A stale PARENT is not caught here at all. The whitelist is what makes the
sentence true for all ten in the direction this file controls.

THE PROTOCOL'S ALPHABET, and why it is a whitelist. Every field below is
written through :func:`_token` / :func:`_tail`, which pass **printable
ASCII and nothing else**; the only ``\\n`` on this stdout is the one
``print`` puts at the end of a record. This used to be a blacklist —
``str.replace("\\n", " ")`` on the model text only — and it was too narrow
twice over:

* ``str.splitlines()``, which the parent used to read with, breaks on ten
  characters, not one (measured: U+000A U+000B U+000C U+000D U+001C U+001D
  U+001E U+0085 U+2028 U+2029). A model value carrying one of those was
  ONE line to this writer and TWO to that reader, which let the payload
  forge the terminator.
* ``\\r`` used to be worse than the rest and unfixable downstream: the
  parent captured with ``text=True``, so Python's universal-newline
  decoding turned a ``\\r`` into a real ``\\n`` **before the parent got to
  split anything** (measured), and no reader-side rule could see it.
  **THAT SENTENCE DESCRIBED A PARENT THIS PACKAGE NO LONGER HAS**
  (2026-08-09): the parent reads bytes and applies the one translation
  itself (``solvers._decode_child_stream``), so a bare ``\\r`` now reaches
  its alphabet check and is refused there — eight of the ten backstopped
  became nine. **This whitelist is still the load-bearing half and is not
  weakened by that.** The reader's share is what a STALE parent-and-driver
  pair degrades to; the boundary for a matched pair is created here, and
  ``\\n`` and ``\\r\\n`` are created here or nowhere at all, because both
  are real record boundaries that no reader can tell from two records.

A whitelist rather than a wider blacklist because the blacklist's contents
are not ours: ``str.splitlines()`` may learn a new separator, and the io
layer may learn a new translation. Printable ASCII cannot become a line
boundary under either.

:func:`_token` additionally escapes the space, because ``value`` and
``opaque`` lines are read with ``split(maxsplit=2)`` — a space inside a
NAME would shift the value into the name's field, which is the same
writer/reader disagreement one delimiter down.

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


def _esc(text: str, keep_space: bool) -> str:
    lo = 0x20 if keep_space else 0x21
    return "".join(
        c if lo <= ord(c) <= 0x7E else f"\\u{{{ord(c):x}}}" for c in text
    )


def _token(text: str) -> str:
    """One whitespace-free field: no line boundary AND no field boundary."""
    return _esc(text, keep_space=False)


def _tail(text: str) -> str:
    """A record's free-text last field: spaces are content, everything
    outside printable ASCII is escaped."""
    return _esc(text, keep_space=True)


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
        print(f"version {_token(str(version))}", file=out)
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
        written = 0
        if answer == "sat":
            for term in sm.getDeclaredTerms():
                value = solver.getValue(term)
                name = _token(str(term))
                if value.isRealValue():
                    print(f"value {name} {_tail(str(value.getRealValue()))}", file=out)
                else:
                    print(f"opaque {name} {_tail(str(value))}", file=out)
                written += 1
        print(f"end {written}", file=out)
        return 0
    except Exception as e:  # noqa: BLE001 — the parent quotes this, never crashes
        print(f"error {_tail(f'{type(e).__name__}: {e}')}", file=out)
        return 0


if __name__ == "__main__":
    sys.exit(main())
