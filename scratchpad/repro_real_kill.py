# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""R-C: the forged terminator on bytes a REAL cvc5 child really wrote,
truncated by a REAL SIGKILL, with the parent reading the WHOLE flushed
prefix (no read limit of ours).

The child's truncation point is set by the pipe/buffer mechanics and is
deterministic for a given stream shape (measured: 53311 bytes, 15/15
trials). We place a poisoned quoted symbol so that boundary falls exactly
after its forged `end N`.

The ONLY constructed ingredient is the exit code, and it is applied
separately and labelled.
"""
from __future__ import annotations

import subprocess
import sys

from stelling import solvers

VT = "\x0b"
REAL_SPAWN = subprocess.run
HEADER = len("version 1.3.4-modified\n") + len("answer sat\n")
PLAIN = len("value x0000 1\n")

RUNNER = (
    "import subprocess, sys, time\n"
    "p = subprocess.Popen([sys.executable, '-m', 'stelling._cvc5_driver'],\n"
    "                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,\n"
    "                     stderr=subprocess.DEVNULL)\n"
    "p.stdin.write(sys.stdin.buffer.read()); p.stdin.close()\n"
    "time.sleep(0.4)\n"           # let it fill the pipe and block
    "p.kill(); p.wait()\n"
    "buf = p.stdout.read()\n"     # EVERYTHING the dead child had flushed
    "sys.stderr.buffer.write(repr((buf, p.returncode)).encode())\n"
)


def build(m: int, pad: int, total: int) -> str:
    """`m` plain consts, then the poisoned quoted symbol, then filler."""
    out = ["(set-option :produce-models true)", "(set-logic QF_LRA)"]

    def decl(nm):
        out.extend([f"(declare-const {nm} Real)", f"(assert (= {nm} 1.0))"])

    for i in range(m):
        decl(f"x{i:04d}")
    decl("|" + ("p" * pad) + VT + f"end {m}" + "|")
    for i in range(m, total):
        decl(f"x{i:04d}")
    out += ["(check-sat)", "(get-model)"]
    return "\n".join(out) + "\n"


def kill_it(script: str):
    q = REAL_SPAWN([sys.executable, "-c", RUNNER], input=script.encode(),
                   capture_output=True, timeout=300)
    buf, rc = eval(q.stderr.decode())  # noqa: S307 -- our own repr
    return buf.decode(), rc


M, TOTAL, CUT = 3800, 6000, 53311
want = VT + f"end {M}"
pad = CUT - HEADER - PLAIN * M - len("value |") - len(want)
for attempt in range(8):
    text, rc = kill_it(build(M, pad, TOTAL))
    if VT not in text:
        print(">>> FIXED: the driver escaped the separator; no raw U+000B "
              f"reaches the parent at all. tail={text[-40:]!r}")
        raise SystemExit(0)
    if text.endswith(want):
        break
    here = HEADER + PLAIN * M + len("value |") + pad + len(want)
    print(f"  calibrate {attempt}: cut={len(text)} want-cut-at={here} "
          f"tail={text[-12:]!r}")
    pad += len(text) - here

lines = text.splitlines()
print(f"\nreal cvc5, real driver, real SIGKILL: rc={rc}, "
      f"{len(text)} bytes flushed before death (the child's boundary, "
      f"not a read limit of ours)")
print(f"the child's own last written bytes: {text[-30:]!r}  "
      f"(no trailing newline: cut mid-write)")
print(f"reader sees {len(lines)} lines; LAST = {lines[-1]!r}")
print(f"the child never wrote a terminator and had {TOTAL - M} terms left "
      f"to walk.")

solvers._cvc5_wheel_version = lambda: "1.3.4"


def parent(stdout, code):
    argv = ["python", "-m", "stelling._cvc5_driver"]
    subprocess.run = lambda a, **kw: subprocess.CompletedProcess(argv, code, stdout, "")
    try:
        return solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)
    finally:
        subprocess.run = REAL_SPAWN


r = parent(text, rc)
print(f"\nreal bytes + REAL exit {rc}: answer={r.answer!r}  "
      f"detail={r.detail[:96]!r}")
r0 = parent(text, 0)
print(f"real bytes + CONSTRUCTED exit 0: answer={r0.answer!r} "
      f"values={len(r0.values)} detail={r0.detail[:60]!r}")
