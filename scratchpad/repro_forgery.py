# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the terminator forgery against `solvers._run_cvc5_wheel`.

R-A  real driver + real cvc5: a raw separator really does reach driver stdout.
R-B  real parent + a child truncated after a poisoned line, exit 0: THE FORGERY.
R-C  real parent + real driver + real cvc5 + a real SIGKILL: the terminator
     tell alone, defeated on bytes a real child really wrote.
"""
from __future__ import annotations

import os
import subprocess
import sys

from stelling import solvers

VT = "\x0b"
REAL_SPAWN = subprocess.run


def parent(stdout: str, rc: int):
    """Run the real parent over a child result we hand it."""
    argv = ["python", "-m", "stelling._cvc5_driver"]
    subprocess.run = lambda a, **kw: subprocess.CompletedProcess(argv, rc, stdout, "")
    try:
        return solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)
    finally:
        subprocess.run = REAL_SPAWN


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


solvers._cvc5_wheel_version = lambda: "1.3.4"

# --------------------------------------------------------------- R-A ---------
hdr("R-A  real driver + real cvc5: does a raw splitlines() separator get out?")
script = (
    "(set-option :produce-models true)\n(set-logic QF_LRA)\n"
    f"(declare-const |x0{VT}end 1| Real)\n"
    f"(assert (= |x0{VT}end 1| 3.0))\n(check-sat)\n(get-model)\n"
)
p = REAL_SPAWN(
    [sys.executable, "-m", "stelling._cvc5_driver"],
    input=script, capture_output=True, text=True, timeout=60,
)
print(f"child rc={p.returncode}  stdout={p.stdout!r}")
print(f"writer's notion of a line  (split on '\\n'): {len(p.stdout.split(chr(10))) - 1}")
print(f"reader's notion of a line  (splitlines()) : {len(p.stdout.splitlines())}")
print(f"reader's lines: {p.stdout.splitlines()}")
if len(p.stdout.split("\n")) - 1 != len(p.stdout.splitlines()):
    print(">>> DEFECT PRESENT: real cvc5 puts a raw U+000B on driver stdout "
          "and the two sides disagree about how many lines that is.")
else:
    print(">>> FIXED: the driver escaped it; both sides count the same lines.")

# --------------------------------------------------------------- R-B ---------
hdr("R-B  THE FORGERY: truncated child, exit 0, both tells blind")
# One line to the writer. `replace('\n', ' ')` passes it through untouched.
payload_line = f"opaque x1 junk{VT}end 2"
forged = "version 1.3.4\nanswer sat\nvalue x0 1/1\n" + payload_line + "\n"
print(f"child wrote {len(forged.split(chr(10))) - 1} lines; the last one it wrote is")
print(f"    {payload_line!r}")
print(f"    -- and it is ONE line to `str.split('\\n')`, the writer's splitter.")
print(f"the child never wrote a terminator and never finished its model walk.")
print(f"reader's lines: {forged.splitlines()}")
r = parent(forged, 0)
print(f"\nparent returned: answer={r.answer!r} values={r.values} "
      f"nonrational={r.nonrational} detail={r.detail!r}")
if r.answer == "sat":
    print(">>> FORGERY REPRODUCED: a definite `sat` from a truncated run.")
else:
    print(">>> refused.")

hdr("R-B controls")
for tag, out, rc in [
    ("same payload, no separator", "version 1.3.4\nanswer sat\nvalue x0 1/1\n"
     "opaque x1 junk end 2\n", 0),
    ("same payload, exit 1", forged, 1),
    ("healthy run", "version 1.3.4\nanswer sat\nvalue x0 1/1\nend 1\n", 0),
]:
    rr = parent(out, rc)
    print(f"{tag:30s} -> {rr.answer!r} values={rr.values}")

# --------------------------------------------------------------- R-C ---------
hdr("R-C  real driver + real cvc5 + a REAL SIGKILL, truncated mid-line")
# Enough declared consts to overrun the 8192-byte pipe buffer, with the
# poisoned name placed so the flushed prefix ends just after `<VT>end N`.
N = 700
lines = ["(set-option :produce-models true)", "(set-logic QF_LRA)"]
names = []
for i in range(N):
    nm = f"x{i:04d}"
    names.append(nm)
    lines.append(f"(declare-const {nm} Real)")
    lines.append(f"(assert (= {nm} {i}.0))")
lines += ["(check-sat)", "(get-model)"]
script = "\n".join(lines) + "\n"

runner = (
    "import os, signal, subprocess, sys, threading\n"
    "p = subprocess.Popen([sys.executable, '-m', 'stelling._cvc5_driver'],\n"
    "                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,\n"
    "                     stderr=subprocess.DEVNULL, text=True)\n"
    "p.stdin.write(sys.stdin.read()); p.stdin.close()\n"
    "buf = p.stdout.read(8192)\n"          # exactly one pipe-buffer flush
    "p.kill(); p.wait()\n"
    "sys.stderr.write(repr((buf, p.returncode)))\n"
)
q = REAL_SPAWN([sys.executable, "-c", runner], input=script,
               capture_output=True, text=True, timeout=180)
buf, rc = eval(q.stderr)  # noqa: S307 -- our own repr, our own process
print(f"real child killed for real: rc={rc}, {len(buf)} bytes through")
rlines = buf.splitlines()
print(f"reader sees {len(rlines)} lines; last = {rlines[-1]!r}")
print(">>> a real SIGKILL truncates real driver stdout mid-line "
      f"({len(buf)} bytes, last line {'complete' if buf.endswith(chr(10)) else 'PARTIAL'})")
print("(the poisoned-name forgery needs the tail after the separator to be "
      "exactly `end N`; see the report for why the NAME channel cannot supply "
      "that and the VALUE channel is escaped by cvc5.)")
