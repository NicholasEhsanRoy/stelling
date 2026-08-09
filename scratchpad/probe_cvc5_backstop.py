# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0
"""Probe 4: how much of the separator set does the parent's alphabet check
actually back-stop, and what would it cost to make it complete?

PART A -- COVERAGE, BEFORE AND AT THIS TIP. `solvers._run_cvc5_wheel` refuses
stdout carrying a byte outside printable ASCII and `\\n`. The comment above it
once called that "the fail-closed backstop for a driver out of step with this
parser". Measured against a REAL stale child writing REAL bytes, under
`text=True` it refused EIGHT of the ten `splitlines()` separators, and the two
it did not refuse failed for two different reasons that must not be run
together.

* `\\n` is excluded from the check BY CONSTRUCTION -- it is the protocol's own
  record boundary, so a writer that leaves one inside a field has written two
  records and there is nothing here to detect. That half is the writer's job
  and always was, and it is unchanged.
* `\\r` was the hole. `capture_output=True, text=True` universal-newline
  decoding turned it into a real `\\n` in the parent's buffer before any
  reader-side rule ran, so no `\\r` ever reached the check. A stale child that
  wrote `opaque x1 j\\rend 2\\r` and NO terminator record of its own got `sat`
  with a model -- and `unsat`, the DISCHARGE direction, off the identical
  corpse (part D). THAT ROW IS THE ONE THAT MOVED: the transport decodes for
  itself now and the count reads NINE.

PART B -- ONE CANDIDATE REPAIR, AND IT IS NOT THE ONLY ONE. `text=False` plus
a RAW decode puts the `\\r` back in front of the check. Driven here through the
same io layer: a child whose stdout translates `\\n` to `\\r\\n` -- the Windows
default, and README.md names Windows as a platform both solver wheels install
on -- is read correctly by the SHIPPED `text=True` and refused outright under
raw bytes-plus-decode.

PART C -- THE ARM THAT LANDED, AND THE ONE THE ORIGINAL WRITE-UP DID NOT
MEASURE AT ALL.
`bytes.decode().replace("\\r\\n", "\\n")` is the same repair with the ONE newline
translation `text=True` was doing for us put back by hand. Part B's conclusion
was stated over the whole `text=False`-plus-decode class -- "it ALSO refuses
every healthy run whose child applies a `\\r\\n` newline translation" -- on a
measurement of the raw arm alone. That is false of this arm, and the table
below is the refutation:

  case                                   shipped   raw      crlf
  healthy POSIX   `\\n`                    sat       sat      sat
  healthy Windows `\\r\\n`                  sat       FAILED   sat
  stale `\\r`-poisoned, LF body            SAT (!)   failed   failed
  stale `\\r`-poisoned, CRLF body          SAT (!)   failed   failed
  stale `\\x0b`-poisoned                   failed    failed   failed
  separators refused, LF-body stale       8 of 10   9 of 10  9 of 10

The `crlf` arm DOMINATES the shipped reader on everything measured here:
identical on both healthy children (same answer, same values), strictly
stronger on the stale ones, and with no platform coupling. IT IS WHAT THIS
TREE NOW DOES -- `solvers._decode_child_stream`. The `shipped` column below is
therefore a record of what the transport used to be, not of what it is, and
the `crlf` column is the tip.

An arm is selected by swapping the transport's OWN decode point, so the parser
under test is the real one in every column and only the decode differs. The
`shipped` arm is the one thing here that is modelled rather than run, because
this tree no longer contains that io layer; it is checked against a real
`text=True` spawn on every child in the run
(`_universal_newlines_is_still_what_we_think`), so a drift in the model is an
assertion failure and not a quiet wrong number.

Usage: probe_cvc5_backstop.py     (run against any tree; the numbers quoted in
                                   SOUNDNESS.md and solvers.py are from this
                                   one)
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

# the same stale driver on a box whose records are CRLF-terminated: the poison
# is `\r` AND every record separator is `\r\n`. This is the case that tells the
# `crlf` arm apart from a reader that just refuses everything with a `\r` in
# it -- the separators must survive and only the poison must be caught.
STALE_CRLF = (
    "import sys\n"
    "w = sys.stdout.buffer.write\n"
    "w(b'version 1.3.4-modified\\r\\n')\n"
    "w(b'answer sat\\r\\n')\n"
    "w(b'value x0 1/2\\r\\n')\n"
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


ARMS = ("shipped", "raw", "crlf")

# The three readers, as functions of the BYTES the child wrote.
#
#   shipped -- what `capture_output=True, text=True` used to hand the parent:
#              a strict decode plus universal newlines, which maps `\r\n` AND a
#              bare `\r` to `\n`, and RAISES on undecodable bytes.
#   raw     -- bytes plus a decode and no translation whatever.
#   crlf    -- what `solvers._decode_child_stream` does at this tip: the one
#              translation `text=True` was performing, put back by hand.
#
# `shipped` is a MODEL of an io layer this tree no longer uses, so it is
# checked against the real thing on every child in this run rather than
# trusted -- see `_universal_newlines_is_still_what_we_think`.
#
# `crlf` is bound HERE, at import, and not looked up inside the arm: `_parent`
# swaps `solvers._decode_child_stream` to select an arm, so a late lookup would
# find the swapped one and recurse into itself.
_TIP_DECODE = solvers._decode_child_stream

_READERS = {
    "shipped":
        lambda raw: raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"),
    # `replace`, not strict: this is the arm as it was DEFINED and measured
    # when it was rejected, and a strict decode would silently make it a
    # different arm than the one the rejection is on record about.
    "raw": lambda raw: raw.decode("utf-8", "replace"),
    "crlf": _TIP_DECODE,
}


def _universal_newlines_is_still_what_we_think(argv) -> None:
    """Refuse to model `text=True` without checking the model on this child."""
    raw = REAL(argv, capture_output=True, timeout=60).stdout
    try:
        real_text = REAL(argv, capture_output=True, text=True, timeout=60).stdout
    except UnicodeDecodeError:
        try:
            _READERS["shipped"](raw)
        except UnicodeDecodeError:
            return  # both raise: the model is faithful on this child too
        raise AssertionError("model decoded bytes that `text=True` refused")
    modelled = _READERS["shipped"](raw)
    assert modelled == real_text, (modelled, real_text)


def _parent(argv, *, as_bytes=False, arm=None):
    """Run the real parser over the real child, through one of three readers.

    The parser takes BYTES and decodes them itself now, so an arm is chosen by
    swapping `solvers._decode_child_stream` — the transport's own decode point
    — rather than by pre-decoding behind its back. `as_bytes=True` is the old
    spelling of `arm="raw"` and is kept so the part B call sites read the same
    as they did when their numbers were taken.
    """
    if arm is None:
        arm = "raw" if as_bytes else "shipped"

    def shim(a, **kw):
        return REAL(argv, capture_output=True, timeout=60)

    real_decode = solvers._decode_child_stream
    subprocess.run = shim
    solvers._decode_child_stream = _READERS[arm]
    try:
        return solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)
    except UnicodeDecodeError:
        raise
    finally:
        subprocess.run = REAL
        solvers._decode_child_stream = real_decode


def part_a() -> None:
    print("PART A -- what the alphabet check backstops, stale child, real bytes")
    print(f"stelling under test: {solvers.__file__}\n")
    definite = {"shipped": [], "crlf": []}
    for label, escape in SEPARATORS:
        argv = [sys.executable, "-c", STALE.format(sep=escape)]
        _universal_newlines_is_still_what_we_think(argv)
        raw = REAL(argv, capture_output=True, timeout=60).stdout
        seen = REAL(argv, capture_output=True, text=True, timeout=60).stdout
        row = {}
        for arm in ("shipped", "crlf"):
            result = _parent(argv, arm=arm)
            row[arm] = result
            if result.answer in ("sat", "unsat", "unknown"):
                definite[arm].append(label)
        print(f"  {label:12s} child bytes {raw[-22:]!r:28s} "
              f"text=True saw {seen[-20:]!r:24s} "
              f"was={row['shipped'].answer!r:10s} now={row['crlf'].answer!r}"
              + (f"  values={row['crlf'].values}" if row["crlf"].values else ""))
    for arm, tag in (("shipped", "text=True, before"), ("crlf", "THIS TIP")):
        got = definite[arm]
        print(f"\n  {tag}: definite answers from a child that wrote NO "
              f"terminator: {got or 'none'}")
        print(f"  refused by the alphabet check: "
              f"{len(SEPARATORS) - len(got)} of {len(SEPARATORS)}")
    print("\n  U+000A is the protocol's own record boundary and is excluded "
          "from the check by\n  construction -- a writer that leaves one in a "
          "field wrote two records, and there\n  is nothing on this side to "
          "detect. U+000D was a HOLE: invisible to the check rather\n  than "
          "admitted by it, because universal-newline decoding had already "
          "spent it. That\n  is the row that moved.")


def part_b() -> None:
    print("\nPART B/C -- three readers on the same child, same io layer")
    print("  HEALTHY CHILDREN. Every arm must answer sat; a `failed` here is "
          "the arm crying wolf\n  on a run that was fine.")
    for label, newline in (("POSIX child   (newline='\\n')", "\n"),
                           ("Windows child (newline='\\r\\n')", "\r\n")):
        argv = [sys.executable, "-c", HEALTHY.format(nl=newline)]
        raw = REAL(argv, capture_output=True, timeout=60).stdout
        print(f"  {label}\n    child bytes            : {raw!r}")
        for tag, arm in (("(a) SHIPPED text=True ", "shipped"),
                         ("(b) bytes + decode    ", "raw"),
                         ("(c) bytes + decode +  ", "crlf")):
            r = _parent(argv, arm=arm)
            extra = ' replace("\\r\\n","\\n")' if arm == "crlf" else ""
            print(f"    {tag}{extra:24s}: {r.answer!r} values={r.values}")


def part_c() -> None:
    """The stale sweep under all three readers, on an LF body and a CRLF one.

    The count that matters is how many of the ten separators each reader
    REFUSES when the child wrote no terminator record of its own. Higher is
    better here, and it has to be read together with part B: an arm that
    refuses everything scores 10 and is useless.
    """
    for title, template in (("LF body", STALE), ("CRLF body", STALE_CRLF)):
        print(f"\n  STALE CHILD, {title}, no terminator record of its own")
        definite = {a: 0 for a in ARMS}
        for label, escape in SEPARATORS:
            argv = [sys.executable, "-c", template.format(sep=escape)]
            row = {}
            for arm in ARMS:
                r = _parent(argv, arm=arm)
                row[arm] = r.answer
                if r.answer in ("sat", "unsat", "unknown"):
                    definite[arm] += 1
            print(f"    {label:12s} shipped={row['shipped']!r:10s} "
                  f"raw={row['raw']!r:10s} crlf={row['crlf']!r:10s}")
        for arm in ARMS:
            print(f"      {arm:8s} refuses "
                  f"{len(SEPARATORS) - definite[arm]} of {len(SEPARATORS)}")


def part_d() -> None:
    """The direction that matters, and the residuals.

    A forged terminator on a truncated child is not only a spurious VIOLATION:
    the same shape returns `unsat`, which is a DISCHARGE, if that is what the
    corpse's `answer` line happened to say.
    """
    print("\nPART D -- the hole reaches the discharge direction, and the "
          "residuals of arm (c)")
    for ans in ("sat", "unsat", "unknown"):
        prog = ("import sys\nw = sys.stdout.buffer.write\n"
                "w(b'version 1.3.4-modified\\n')\n"
                f"w(b'answer {ans}\\n')\n"
                "w(b'opaque x1 j\\rend 1\\r')\n")
        argv = [sys.executable, "-c", prog]
        print(f"  stale \\r child saying `answer {ans}`: "
              + "  ".join(f"{a}={_parent(argv, arm=a).answer!r}" for a in ARMS))

    bad = [sys.executable, "-c", "import sys;sys.stdout.buffer.write("
           "b'version 1.3.4-modified\\nanswer sat\\nvalue x0 1/2\\n\\xff\\n"
           "end 2\\n')"]
    print("  child writes invalid UTF-8 (not in either arm's table):")
    for arm in ARMS:
        try:
            print(f"    {arm:8s} -> {_parent(bad, arm=arm).answer!r}")
        except Exception as e:                       # noqa: BLE001
            print(f"    {arm:8s} -> RAISED {type(e).__name__} (uncaught)")

    cr = [sys.executable, "-c", "import sys\n"
          "sys.stdout.reconfigure(newline='\\r')\n"
          "print('version 1.3.4-modified');print('answer sat')\n"
          "print('value x0 1/2');print('end 1')"]
    print("  healthy child reconfigured to BARE CR -- arm (c)'s one measured "
          "cry-wolf case;\n  no platform's `print` default produces this and "
          "`_cvc5_driver` never sets it:")
    for arm in ARMS:
        print(f"    {arm:8s} -> {_parent(cr, arm=arm).answer!r}")


if __name__ == "__main__":
    solvers._cvc5_wheel_version = lambda: "1.3.4"
    part_a()
    part_b()
    part_c()
    part_d()
