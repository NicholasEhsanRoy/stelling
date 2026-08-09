# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""Property fuzzer for the cvc5 wheel transport's record protocol.

Equivalent to the spike's driver, including the two generator changes that
made it find anything: **line-boundary truncation** and an exit code drawn
**independently** of whether truncation happened.

Model of the pair, faithful on both sides:

* the WRITER is the driver: a record is ``sanitiser(text) + "\\n"``, and the
  sanitiser is whichever one the tree under test ships.
* the transport is the real ``solvers._run_cvc5_wheel``, handed the child's
  stdout after the same universal-newline decoding ``text=True`` performs.

PROPERTY (the soundness one): the transport returns a definite answer only
if the child really wrote a complete protocol AND exited 0. Ground truth is
computed from what the WRITER emitted, never from what the reader parsed.

PROPERTY (the cry-wolf one): a healthy, complete, exit-0 child is accepted.

Usage: fuzz_transport.py [examples] [seed]
"""
from __future__ import annotations

import random
import subprocess
import sys

from stelling import _cvc5_driver, solvers

REAL = subprocess.run

# Whatever the tree under test sanitises with. Base has neither name.
_TOKEN = getattr(_cvc5_driver, "_token", lambda s: s.replace("\n", " "))
_TAIL = getattr(_cvc5_driver, "_tail", lambda s: s.replace("\n", " "))

SEPS = ["\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " ", " "]
NAMES = ["x0", "x1", "end", "x0 y", "v"]
ANSWERS = ["sat", "unsat", "unknown"]


def _payload(rng: random.Random) -> str:
    """Free text for a model value, sometimes carrying a separator and a
    fragment that looks like a terminator."""
    body = rng.choice(["junk", "(root 2)", "1/1", "", "end"])
    if rng.random() < 0.75:
        n = rng.randrange(0, 5)
        return body + rng.choice(SEPS) + f"end {n}"
    return body


def generate(rng: random.Random):
    """One child run: its records, its truncation, its exit code."""
    answer = rng.choice(ANSWERS)
    records = [f"version {_TOKEN('1.3.4')}", f"answer {answer}"]
    n_model = rng.randrange(0, 5)
    for _ in range(n_model):
        name = _TOKEN(rng.choice(NAMES))
        if rng.random() < 0.5:
            records.append(f"value {name} {_TAIL(str(rng.randrange(9)) + '/1')}")
        else:
            records.append(f"opaque {name} {_TAIL(_payload(rng))}")
    if rng.random() < 0.6:                      # sometimes a second answer
        pass
    else:
        records.insert(rng.randrange(2, len(records) + 1), f"answer {rng.choice(ANSWERS)}")
    complete = rng.random() < 0.5
    if complete:
        records.append(f"end {n_model}")
    stream = "".join(r + "\n" for r in records)

    # GENERATOR CHANGE 1: truncation, at a line boundary as well as mid-line.
    truncated = False
    kind = rng.random()
    if kind < 0.25:                              # cut at a record boundary
        keep = rng.randrange(0, len(records) + 1)
        if keep < len(records):
            truncated = True
        stream = "".join(r + "\n" for r in records[:keep])
    elif kind < 0.45 and stream:                 # cut mid-write
        cut = rng.randrange(0, len(stream))
        if cut < len(stream):
            truncated = True
        stream = stream[:cut]

    # GENERATOR CHANGE 2: the exit code is INDEPENDENT of truncation.
    rc = rng.choice([0, 0, 0, 1, -9, -11])

    wrote_full = (
        not truncated
        and complete
        and [r for r in records if r.startswith("answer ")] == [f"answer {answer}"]
        and not any(r.startswith("error") for r in records)
    )
    return stream, rc, wrote_full, answer, n_model


def as_child_bytes(text: str) -> bytes:
    """The BYTES the child wrote.

    This was `decode()` — `text.replace("\\r\\n","\\n").replace("\\r","\\n")`,
    a restatement of what `capture_output=True, text=True` did to the child's
    stdout before the parent saw it. The parent decodes for itself now
    (`solvers._decode_child_stream`), so restating it here would be modelling
    the thing under test.
    """
    return text.encode("utf-8")


def run_parent(stdout: bytes, rc: int):
    subprocess.run = lambda a, **kw: subprocess.CompletedProcess([], rc, stdout, b"")
    try:
        return solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)
    finally:
        subprocess.run = REAL


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rng = random.Random(seed)
    solvers._cvc5_wheel_version = lambda: "1.3.4"

    unsound = []      # definite answer from an incomplete/crashed run
    crywolf = []      # healthy run refused
    for i in range(n):
        stream, rc, wrote_full, answer, n_model = generate(rng)
        r = run_parent(as_child_bytes(stream), rc)
        definite = r.answer in ("sat", "unsat", "unknown")
        if definite and not (wrote_full and rc == 0):
            unsound.append((i, stream, rc, wrote_full, r.answer, r.values))
        if wrote_full and rc == 0 and not definite:
            crywolf.append((i, stream, rc, r.answer, r.detail))

    print(f"examples={n} seed={seed}")
    print(f"UNSOUND (definite answer from an incomplete or crashed run): {len(unsound)}")
    for rec in unsound[:3]:
        i, stream, rc, wf, ans, vals = rec
        print(f"   first at example {i}: rc={rc} wrote_full={wf} -> {ans!r} "
              f"values={vals}\n     stdout={stream!r}")
    print(f"CRY-WOLF (healthy complete exit-0 run refused): {len(crywolf)}")
    for rec in crywolf[:3]:
        print(f"   example {rec[0]}: -> {rec[3]!r} detail={rec[4][:80]!r}\n"
              f"     stdout={rec[1]!r}")
    return 1 if (unsound or crywolf) else 0


if __name__ == "__main__":
    sys.exit(main())
