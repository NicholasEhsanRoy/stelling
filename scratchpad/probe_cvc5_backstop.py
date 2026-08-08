# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""Probe 4: how much of the separator set does the parent's alphabet check
actually back-stop, and what would it cost to make it complete?

PART A -- COVERAGE. `solvers._run_cvc5_wheel` refuses stdout carrying a byte
outside printable ASCII and `\\n`. The comment above it called that "the
fail-closed backstop for a driver out of step with this parser". Measured
against a REAL stale child writing REAL bytes: it refuses EIGHT of the ten
`splitlines()` separators, and the two it does not refuse fail for two
different reasons that must not be run together.

* `\\n` is excluded from the check BY CONSTRUCTION -- it is the protocol's own
  record boundary, so a writer that leaves one inside a field has written two
  records and there is nothing here to detect. That half is the writer's job
  and always was.
* `\\r` is the hole. `capture_output=True, text=True` universal-newline
  decoding turns it into a real `\\n` in the parent's buffer before any
  reader-side rule runs, so no `\\r` ever reaches the check. A stale child that
  writes `opaque x1 j\\rend 2\\r` and NO terminator record of its own still gets
  `sat` with a model.

PART B -- THE COST OF CLOSING IT. The obvious repair is `text=False` plus an
explicit decode, which would put the `\\r` back in front of the check. Driven
here through the same io layer: a child whose stdout translates `\\n` to
`\\r\\n` -- the Windows default, and README.md names Windows as a platform both
solver wheels install on -- is read correctly by the SHIPPED `text=True` and is
refused outright under bytes-plus-decode. The repair trades a stale-install
hazard for a platform coupling that cannot be measured on this box.

Usage: probe_cvc5_backstop.py     (run against any tree; the numbers quoted in
                                   SOUNDNESS.md and solvers.py are from the
                                   fixed one)
"""
from __future__ import annotations

import subprocess
import sys

from stelling import solvers

REAL = subprocess.run

# a stale driver: one record per write, poisoned with `sep`, and no terminator
# record of its own anywhere.
STALE = (
    "import sys\n"
    "w = sys.stdout.buffer.write\n"
    "w(b'version 1.3.4-modified\\n')\n"
    "w(b'answer sat\\n')\n"
    "w(b'value x0 1/2\\n')\n"
    "w(b'opaque x1 j{sep}end 2{sep}')\n"
)

# a healthy driver whose stdout text layer applies a newline translation.
HEALTHY = (
    "import sys\n"
    "sys.stdout.reconfigure(newline={nl!r})\n"
    "print('version 1.3.4-modified')\n"
    "print('answer sat')\n"
    "print('value x0 1/2')\n"
    "print('end 1')\n"
)

SEPARATORS = (
    ("U+000A  \\n", "\\n"),
    ("U+000D  \\r", "\\r"),
    ("U+000B", "\\x0b"),
    ("U+000C", "\\x0c"),
    ("U+001C", "\\x1c"),
    ("U+001D", "\\x1d"),
    ("U+001E", "\\x1e"),
    ("U+0085", "\\xc2\\x85"),
    ("U+2028", "\\xe2\\x80\\xa8"),
    ("U+2029", "\\xe2\\x80\\xa9"),
)


def _parent(argv, *, as_bytes=False):
    def shim(a, **kw):
        proc = REAL(argv, capture_output=True, timeout=60)
        return subprocess.CompletedProcess(
            argv,
            proc.returncode,
            proc.stdout.decode("utf-8", "replace") if as_bytes
            else proc.stdout.decode(),
            "",
        ) if as_bytes else REAL(argv, capture_output=True, text=True, timeout=60)

    subprocess.run = shim
    try:
        return solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)
    finally:
        subprocess.run = REAL


def part_a() -> None:
    print("PART A -- what the alphabet check backstops, stale child, real bytes")
    print(f"stelling under test: {solvers.__file__}\n")
    definite = []
    for label, escape in SEPARATORS:
        argv = [sys.executable, "-c", STALE.format(sep=escape)]
        raw = REAL(argv, capture_output=True, timeout=60).stdout
        seen = REAL(argv, capture_output=True, text=True, timeout=60).stdout
        result = _parent(argv)
        verdict = result.answer
        if verdict in ("sat", "unsat", "unknown"):
            definite.append(label)
        print(f"  {label:12s} child bytes {raw[-22:]!r:28s} "
              f"parent saw {seen[-20:]!r:24s} -> {verdict!r}"
              + (f"  values={result.values}" if result.values else ""))
    print(f"\n  definite answers from a child that wrote NO terminator: "
          f"{definite or 'none'}")
    print(f"  refused by the alphabet check: {len(SEPARATORS) - len(definite)} "
          f"of {len(SEPARATORS)}")
    print("  of the two that get through: U+000A is the protocol's own record "
          "boundary and is\n  excluded from the check by construction; U+000D "
          "is the hole, and it is invisible\n  to the check rather than "
          "admitted by it.")


def part_b() -> None:
    print("\nPART B -- the cost of `text=False` + explicit decode")
    for label, newline in (("POSIX child   (newline='\\n')", "\n"),
                           ("Windows child (newline='\\r\\n')", "\r\n")):
        argv = [sys.executable, "-c", HEALTHY.format(nl=newline)]
        raw = REAL(argv, capture_output=True, timeout=60).stdout
        shipped = _parent(argv)
        candidate = _parent(argv, as_bytes=True)
        print(f"  {label}\n    child bytes            : {raw!r}")
        print(f"    (a) SHIPPED text=True  : {shipped.answer!r} "
              f"values={shipped.values}")
        print(f"    (b) bytes + decode     : {candidate.answer!r} "
              f"values={candidate.values}")


if __name__ == "__main__":
    solvers._cvc5_wheel_version = lambda: "1.3.4"
    part_a()
    part_b()
