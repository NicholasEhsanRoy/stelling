# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""`tests/property/test_cvc5_protocol.py` is SKIPPED on both venvs here —
hypothesis is not installed — and this branch changes its `_FakeProc` to hand
the transport BYTES instead of a `str`. A change to a test that never runs is
a change nobody measured, so this drives that file's own grammar and its own
oracle EXHAUSTIVELY, without hypothesis, against the real transport.

WHAT IT IS LOOKING FOR, specifically. Under `str`, the fake was a reader that
did no decoding at all. Under bytes it is the real decoder, which collapses
`\\r\\n` to `\\n`. That is a NEW way for the parent's record boundaries to
disagree with `_values_actually_present`, which the property computes on the
writer's string with `split("\\n")`:

  * a payload containing a bare `\\r` -> the parent keeps it, the alphabet
    check refuses the run, and a refusal is always safe (the oracle returns
    None). No disagreement possible.
  * a record ENDING in `\\r` -> the `\\n` the writer appends makes `\\r\\n`,
    the parent sees a record boundary one character earlier than the writer
    intended, and the two value censuses could differ. THIS is the case that
    would cry wolf, and whether the grammar can even build it is a fact about
    `_PAYLOADS`, not something to reason about.

Both legs are driven: the record census over the whole payload space, and the
oracle itself over every transcript this grammar can build up to a bounded
length, at EVERY byte prefix (which is a strict superset of the mid-write and
record-boundary truncations the state machine draws) and at each exit code.

Usage:  PYTHONPATH=<tree>/src python scratchpad/crlf/probe_property_grammar.py
"""
from __future__ import annotations

import subprocess
import sys

from stelling import solvers

REAL = subprocess.run

# Verbatim from tests/property/test_cvc5_protocol.py.
SPLITLINES_ONLY = ("\v", "\f", "\r", "\x1c", "\x1d", "\x1e", "\x85", " ", " ")
DEFINITE = ("sat", "unsat", "unknown")
NAMES = ("x0", "x1", "x0_0")
RATIONALS = ("0/1", "1/1", "-1/1", "3/2", "17/4")
PAYLOADS = tuple(["q", "root", "(- 1)", "a b"]) + tuple(
    f"q{sep}{tail}"
    for sep in SPLITLINES_ONLY
    for tail in ("end 1", "end 2", "end 0", "answer unsat", "q")
)


def _values_actually_present(stdout):
    """Verbatim from the property file: the writer's rule, `split("\\n")`."""
    out = []
    for line in stdout.split("\n"):
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[0] == "value":
            out.append((parts[1], parts[2]))
    return sorted(out)


def _read(stdout: str, rc: int):
    """What `_ScriptedChild.read` does now: bytes in, the real decoder runs."""
    raw = stdout.encode("utf-8")
    proc = subprocess.CompletedProcess([], rc, raw, b"")
    real, ver = subprocess.run, solvers._cvc5_wheel_version
    subprocess.run = lambda a, **kw: proc
    solvers._cvc5_wheel_version = lambda: "1.3.4"
    try:
        return solvers._run_cvc5_wheel("(check-sat)", 5.0)
    finally:
        subprocess.run = real
        solvers._cvc5_wheel_version = ver


def _judge(res, stdout, full, rc):
    """Verbatim from the property file's `_judge`, minus the census."""
    if res.answer not in DEFINITE:
        return None  # refusing is always safe
    if stdout != full:
        return f"ACCEPTED A TRUNCATED RUN as {res.answer!r}: {stdout!r}"
    if rc != 0:
        return f"ACCEPTED A NONZERO-EXIT RUN (exit {rc}) as {res.answer!r}"
    present = _values_actually_present(stdout)
    if sorted(res.values) != present:
        return (f"HARVESTED A MODEL THAT WAS NOT WRITTEN: "
                f"read {sorted(res.values)!r} vs written {present!r}")
    return None


def leg_one_can_a_record_end_in_a_carriage_return():
    """The whole of the new exposure, as a census over the grammar."""
    records = []
    for n in NAMES:
        for q in RATIONALS:
            records.append(f"value {n} {q}")
        for t in PAYLOADS:
            records.append(f"opaque {n} {t}")
    for v in ("1.3.4", "1.2.0"):
        records.append(f"version {v}")
    for a in DEFINITE:
        records.append(f"answer {a}")
    for m in ("boom", "parse failure", "out of memory"):
        records.append(f"error {m}")
    for k in range(7):
        records.append(f"end {k}")
    for line in ("", "junk", "end 0", "end 3", "answer sat", "value x9 1/2",
                 "error boom", "version 9.9.9"):
        records.append(line)

    ends_cr = [r for r in records if r.endswith("\r")]
    holds_cr = [r for r in records if "\r" in r]
    values_with_sep = [
        r for r in records
        if r.startswith("value ") and any(s in r for s in SPLITLINES_ONLY)
    ]
    print(f"LEG 1 -- the record grammar, exhaustively: {len(records)} records")
    print(f"  records CONTAINING a bare `\\r`     : {len(holds_cr)}")
    print(f"  records ENDING in `\\r`             : {len(ends_cr)}  {ends_cr}")
    print(f"  `value` records with any separator : {len(values_with_sep)}")
    print("  a record ending in `\\r` is the ONLY way `\\r\\n` can appear in a")
    print("  stream this grammar builds, because the writer appends `\\n`; and")
    print("  a `value` record is the only kind the oracle's census counts.")
    return len(ends_cr), len(values_with_sep)


def leg_two_the_oracle_over_every_transcript_and_every_prefix(max_records=3):
    """The oracle itself, driven at every byte prefix and every exit code.

    A byte prefix is a strict superset of both truncation rules the state
    machine draws (record boundary, mid-write), so this is stronger than the
    stateful leg on the space it covers and much narrower on length.
    """
    # one record of each kind that can carry a separator, plus the terminator
    # grammar; the alphabet is deliberately the interesting slice rather than
    # the whole product, which does not finish.
    pool = (
        ["version 1.3.4", "answer sat", "answer unsat", "error boom"]
        + [f"value {n} 1/1" for n in ("x0", "x1")]
        + [f"opaque x1 {t}" for t in PAYLOADS]
        + [f"end {k}" for k in range(3)]
    )
    # THE DRIVER'S OWN GRAMMAR, and it is not optional. `end <n>` and
    # `error <why>` are its LAST record. `test_cvc5_protocol.py`'s `_emit`
    # enforces exactly this and says why: without it "the machine invents a
    # driver that writes after its own terminator and the oracle cries wolf on
    # a stream no driver emits -- measured: the first control run shrank to
    # exactly that". MEASURED HERE TOO. The first run of this probe omitted the
    # rule and reported SIX counterexamples, every one of them a transcript
    # carrying two terminators (`answer sat`, `end 0`, `end 0`) truncated after
    # the first. That is a defect in a probe's model of a driver, not in the
    # transport, and it is left written down because it is the same mistake the
    # property file already records making.
    def transcripts(depth):
        if depth == 0:
            return
        for r in pool:
            yield (r,)
            if not r.startswith(("end ", "error ")):
                for rest in transcripts(depth - 1):
                    yield (r, *rest)

    findings = []
    driven = definite = refused = 0
    for combo in transcripts(max_records):
        full = "".join(r + "\n" for r in combo)
        for cut in range(len(full) + 1):
            stdout = full[:cut]
            for rc in (0, 1, -9):
                res = _read(stdout, rc)
                driven += 1
                if res.answer in DEFINITE:
                    definite += 1
                else:
                    refused += 1
                msg = _judge(res, stdout, full, rc)
                if msg is not None:
                    findings.append((combo, cut, rc, msg))
    print(f"\nLEG 2 -- the oracle, every transcript up to {max_records} "
          f"records x every byte prefix x 3 exit codes")
    print(f"  pool: {len(pool)} record shapes")
    print(f"  driven: {driven}   definite: {definite}   refused: {refused}")
    print(f"  COUNTEREXAMPLES: {len(findings)}")
    for f in findings[:5]:
        print(f"    {f}")
    # ANTI-VACUITY: a run that accepted nothing would report 0 findings and
    # have examined nothing the property is about.
    assert definite > 0, "nothing was ever accepted; this proves nothing"
    return len(findings), definite


if __name__ == "__main__":
    print(f"stelling under test: {solvers.__file__}\n")
    ends_cr, values_with_sep = leg_one_can_a_record_end_in_a_carriage_return()
    findings, definite = leg_two_the_oracle_over_every_transcript_and_every_prefix()
    print()
    if ends_cr == 0 and values_with_sep == 0 and findings == 0:
        print(">>> the bytes fixture cannot cry wolf on this grammar: no record "
              "can end in a\n    `\\r`, no `value` record can carry a separator "
              "at all, and the oracle finds\n    nothing over the whole "
              "enumerated space.")
        sys.exit(0)
    print(">>> LOOK AGAIN.")
    sys.exit(1)
