# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE cvc5 LINE PROTOCOL AS A FUZZ TARGET.

``stelling.solvers._run_cvc5_wheel`` is a *parent* reading a *child*'s stdout
over a line-oriented record protocol. The child prints ``version``/``answer``/
``value``/``opaque``/``error``/``end <n>`` records; the parent reads them back
and decides whether it may return a definite ``sat``/``unsat``/``unknown``.

**The defect class this exists for is a writer and a reader disagreeing about
where a record ends.** It has bitten here twice, and both are ancestors of
``main``:

* the parent read with ``str.splitlines()`` while the child sanitised with
  ``replace("\\n", " ")``. ``splitlines()`` breaks on **ten** characters, not
  one, so a model value carrying ``\\x0b`` was one line to the writer and two
  to the reader — and the second of those two could supply the parent's *last*
  line, i.e. **forge the terminator**, while the child had been killed
  mid-model-walk. With the child exiting 0, both of the transport's two tells
  went blind at once, which is the one thing two tells exists to prevent;
* the final record's newline cut and nothing else — ``…\\nend 4`` — read as a
  present terminator with a matching count.

Both are closed on ``main``. This file is what keeps them closed, and it is a
**fuzz target**, not a regression test for those two strings: it models the
protocol and lets the search find the next one.

────────────────────────────────────────────────────────────────────────────
THE ORACLE, AND IT IS NOT THE PARSER RESTATED
────────────────────────────────────────────────────────────────────────────

A parser property that re-implements the parser proves nothing. The invariant
here is stated in terms the parser does not compute:

    the parent may return a definite answer ONLY IF
      (1) the child exited 0, AND
      (2) nothing was truncated, AND
      (3) the model it harvested is exactly the value records present in the
          bytes it actually read.

Anything else is a definite answer resting on a transcript that was never
spoken. Note the asymmetry: refusing is always allowed. The property forbids
misplaced *confidence*, never excessive caution.

**Ordering oddities a real driver would never write are deliberately NOT
asserted against.** The first model of this protocol invented a driver that
writes records *after* its own terminator, and the search happily shrank to
exactly that — a stream no driver emits, i.e. crying wolf. ``_emit`` now
carries the driver's own grammar (``end``/``error`` is its last record), which
is a statement about ``stelling._cvc5_driver`` and not about the search.

────────────────────────────────────────────────────────────────────────────
WHY STATEFUL, AND WHAT IT COST
────────────────────────────────────────────────────────────────────────────

A flat fuzzer draws a whole stdout from a fixed template and can only vary the
slots. A ``RuleBasedStateMachine`` draws a *sequence of writes*, so the ORDER
and MULTIPLICITY of records are in the search space — ``end`` in the middle,
two ``answer``s, a record written after the child was killed — and Hypothesis
shrinks the **rule sequence**, which is a far more readable artifact than a
byte string.

Measured honestly, and against the alternative: a flat template fuzzer found
the forgery in **673** examples where this machine took **8 165** — roughly 12×
cheaper, because the driver's record order is fixed, so a template that
hard-codes it spends its whole budget on the slots. What the machine bought is
that it found **both** known defects without being told the record layout.
Both legs ship: the flat one is cheap enough for every push, the stateful one
belongs in the larger profile.

────────────────────────────────────────────────────────────────────────────
WHAT IS AND IS NOT COVERED
────────────────────────────────────────────────────────────────────────────

Covered: the parent's parse of a scripted child's stdout, including truncation
at a record boundary and at an arbitrary byte offset, non-zero exit codes,
duplicate and out-of-order records, and separator characters that ``splitlines``
breaks on but ``print`` never writes.

NOT covered: the child (``stelling._cvc5_driver``) itself — its sanitisation is
the load-bearing half of the fix and is exercised by ``tests/`` elsewhere, not
here; real cvc5; the z3 transport; the binary (non-wheel) cvc5 transport; the
subprocess machinery, which is replaced wholesale; anything above
``_run_cvc5_wheel`` in the escalation stack.

POSITIVE CONTROL: ``0ad22bb`` (the pre-fix ``splitlines()`` parser), where both
legs fail. Registered as ``cvc5-flat`` and ``cvc5-stateful``; run them with
``python tools/property_check.py --controls``.
"""

from __future__ import annotations

import subprocess as _real_subprocess

import pytest

pytest.importorskip("hypothesis", reason="needs hypothesis")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.stateful import (  # noqa: E402
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

import _profiles  # noqa: E402
import _runner  # noqa: E402

# `str.splitlines()` breaks on these; `print` never writes them. They are the
# writer/reader disagreement the protocol's alphabet whitelist exists to close.
SPLITLINES_ONLY = ("\v", "\f", "\r", "\x1c", "\x1d", "\x1e", "\x85", " ", " ")

DEFINITE = ("sat", "unsat", "unknown")


class _FakeProc:
    """What ``subprocess.run`` hands the transport: BYTES, on both streams.

    This used to hand it ``str``, which quietly made the model a reader that
    does no decoding at all — so the ``\\r`` row of ``SPLITLINES_ONLY`` above
    was being scored against a parent that did not exist in either direction:
    the shipped one translated ``\\r`` to ``\\n`` before reading, and this one
    did nothing. The transport decodes for itself now
    (``solvers._decode_child_stream``), so the child's job here is to be the
    bytes and let the real decoder run.
    """

    def __init__(self, stdout, returncode):
        self.stdout = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
        self.stderr = b""
        self.returncode = returncode


class _FakeSubprocess:
    """Stands in for the ``subprocess`` module inside ``stelling.solvers``."""

    TimeoutExpired = _real_subprocess.TimeoutExpired

    def __init__(self):
        self.stdout = ""
        self.rc = 0

    def run(self, *a, **k):
        return _FakeProc(self.stdout, self.rc)


class _ScriptedChild:
    """Patch ``solvers.subprocess`` and the version probe; restore on exit."""

    def __enter__(self):
        import stelling.solvers as S

        self.S = S
        self.fake = _FakeSubprocess()
        self._orig_sub = S.subprocess
        self._orig_ver = S._cvc5_wheel_version
        S.subprocess = self.fake
        S._cvc5_wheel_version = lambda: "1.3.4"
        return self

    def __exit__(self, *exc):
        self.S.subprocess = self._orig_sub
        self.S._cvc5_wheel_version = self._orig_ver
        return False

    def read(self, stdout, rc):
        self.fake.stdout = stdout
        self.fake.rc = rc
        return self.S._run_cvc5_wheel("(check-sat)", 5.0)


def _values_actually_present(stdout):
    """The ``value`` records in the bytes the parent read, by the WRITER's rule.

    ``split("\\n")`` is ``print``'s own boundary. Computing this with
    ``splitlines()`` would restate the defect and the property would agree with
    the bug.
    """
    out = []
    for line in stdout.split("\n"):
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[0] == "value":
            out.append((parts[1], parts[2]))
    return sorted(out)


def _judge(res, stdout, full, rc, census, *, where):
    """The invariant. Returns a failure message or ``None``."""
    census.tag("driven")
    if res.answer not in DEFINITE:
        census.tag("refused")
        return None  # refusing is always safe
    census.tag("definite")
    if stdout != full:
        return (
            f"ACCEPTED A TRUNCATED RUN as {res.answer!r} [{where}]\n"
            f"  read : {stdout!r}\n"
            f"  full : {full!r}"
        )
    if rc != 0:
        return (
            f"ACCEPTED A NONZERO-EXIT RUN (exit {rc}) as {res.answer!r} "
            f"[{where}]\n  stdout: {stdout!r}"
        )
    present = _values_actually_present(stdout)
    if sorted(res.values) != present:
        return (
            f"HARVESTED A MODEL THAT WAS NOT WRITTEN [{where}]\n"
            f"  parent harvested: {sorted(res.values)}\n"
            f"  actually present: {present}\n"
            f"  stdout: {stdout!r}"
        )
    return None


# ── leg 1: the flat template fuzzer ──────────────────────────────────────────
#
# Cheap, and measured ~12x more example-efficient than the state machine on
# this protocol, because the driver's record ORDER is fixed and a template that
# hard-codes it spends its whole budget on the slots rather than rediscovering
# a grammar. Two generator choices carry that efficiency and neither is
# incidental: the truncation point is drawn at a LINE BOUNDARY as often as at a
# byte, and the exit code is drawn INDEPENDENTLY of truncation. Without the
# second, the search never proposes "child died but exited 0", which is the
# whole defect.

_NAMES = st.sampled_from(["x0", "x1", "x0_0"])
_RATIONALS = st.sampled_from(["0/1", "1/1", "-1/1", "3/2", "17/4"])
_PAYLOADS = st.one_of(
    st.sampled_from(["q", "root", "(- 1)", "a b"]),
    st.tuples(
        st.sampled_from(SPLITLINES_ONLY),
        st.sampled_from(["end 1", "end 2", "end 0", "answer unsat", "q"]),
    ).map(lambda t: f"q{t[0]}{t[1]}"),
)


@st.composite
def _transcripts(draw):
    """A plausible driver transcript, plus how it died."""
    records = [f"version {draw(st.sampled_from(['1.3.4', '1.2.0']))}"]
    records.append(f"answer {draw(st.sampled_from(DEFINITE))}")
    n_vals = draw(st.integers(0, 3))
    n_model = 0
    for _ in range(n_vals):
        if draw(st.booleans()):
            records.append(f"value {draw(_NAMES)} {draw(_RATIONALS)}")
        else:
            records.append(f"opaque {draw(_NAMES)} {draw(_PAYLOADS)}")
        n_model += 1
    if draw(st.booleans()):
        records.append(f"end {draw(st.sampled_from([n_model, n_model, 0, 1, 9]))}")
    full = "".join(r + "\n" for r in records)
    # Truncation: at a record boundary as often as at an arbitrary byte.
    if draw(st.booleans()):
        cut = len(full)
    elif draw(st.booleans()):
        prefix = draw(st.integers(0, len(records)))
        cut = len("".join(r + "\n" for r in records[:prefix]))
        if prefix and draw(st.booleans()):
            cut -= 1  # the final record's newline, and nothing else
    else:
        cut = draw(st.integers(0, len(full)))
    # DRAWN INDEPENDENTLY OF TRUNCATION, on purpose.
    rc = draw(st.sampled_from([0, 0, 0, 1, -9, 137]))
    return full, full[:cut], rc


def test_the_parent_never_trusts_an_unspoken_transcript_flat():
    census = _runner.Census("cvc5/flat")

    @_profiles.current().settings(1500)
    @given(_transcripts())
    def search(item):
        census.draw()
        full, stdout, rc = item
        with _ScriptedChild() as child:
            res = child.read(stdout, rc)
        msg = _judge(res, stdout, full, rc, census, where="flat")
        if msg is not None:
            raise AssertionError(msg)

    search()
    # The floor is the anti-vacuity guard: a strategy that degenerated to
    # "empty stdout, exit 0" would pass this property while examining nothing
    # the property is about.
    census.require(driven=200, definite=40, refused=40)


# ── leg 2: the state machine ─────────────────────────────────────────────────


class CvcTransport(RuleBasedStateMachine):
    """Drive ``solvers._run_cvc5_wheel``'s parser through a scripted child."""

    census = _runner.Census("cvc5/stateful")

    def __init__(self):
        super().__init__()
        self.child = _ScriptedChild().__enter__()
        self.records: list[str] = []
        self.finished = False  # the driver wrote its LAST record
        self.truncate_bytes: int | None = None
        self.rc = 0

    def teardown(self):
        self.child.__exit__(None, None, None)

    # -- the child writes -----------------------------------------------------
    @precondition(lambda self: not self.finished)
    @rule(v=st.sampled_from(["1.3.4", "1.2.0"]))
    def write_version(self, v):
        self._emit(f"version {v}")

    @precondition(lambda self: not self.finished)
    @rule(a=st.sampled_from(DEFINITE))
    def write_answer(self, a):
        self._emit(f"answer {a}")

    @precondition(lambda self: not self.finished)
    @rule(n=_NAMES, q=_RATIONALS)
    def write_value(self, n, q):
        self._emit(f"value {n} {q}")

    @precondition(lambda self: not self.finished)
    @rule(n=_NAMES, t=_PAYLOADS)
    def write_opaque(self, n, t):
        self._emit(f"opaque {n} {t}")

    @precondition(lambda self: not self.finished)
    @rule(m=st.sampled_from(["boom", "parse failure", "out of memory"]))
    def write_error(self, m):
        self._emit(f"error {m}")

    @precondition(lambda self: not self.finished)
    @rule(k=st.integers(min_value=0, max_value=6))
    def write_terminator(self, k):
        self._emit(f"end {k}")

    @precondition(lambda self: not self.finished)
    @rule()
    def write_honest_terminator(self):
        n = sum(1 for r in self.records if r.startswith(("value ", "opaque ")))
        self._emit(f"end {n}")

    @precondition(lambda self: not self.finished)
    @rule(line=st.sampled_from(["", "junk", "end 0", "end 3", "answer sat",
                                "value x9 1/2", "error boom", "version 9.9.9"]))
    def write_raw(self, line):
        self._emit(line)

    # -- the child dies -------------------------------------------------------
    #
    # TWO PHYSICALLY DISTINCT KILL POINTS, because the transport's own comment
    # says the boundary is the PIPE BUFFER, not the model size: a child killed
    # after `print` returned has whole records through, while a child killed
    # inside the write has a partial record through — and "the newline cut and
    # nothing else" is then a real byte offset rather than a contrived one. A
    # first model that drew a uniform truncation FRACTION could essentially
    # never land on a record boundary and found nothing at 3000 examples.
    @precondition(lambda self: self.records)
    @rule()
    def die_at_record_boundary(self):
        self.truncate_bytes = len(self._render_full())

    @precondition(lambda self: self.records)
    @rule(data=st.data())
    def die_mid_write(self, data):
        full = self._render_full()
        self.truncate_bytes = data.draw(
            st.integers(min_value=0, max_value=len(full)), label="bytes through"
        )

    @rule(code=st.sampled_from([0, 1, -9, 137]))
    def set_exit(self, code):
        self.rc = code

    @rule()
    def survives(self):
        self.truncate_bytes = None

    # -- oracle ---------------------------------------------------------------
    def _emit(self, text):
        # THE DRIVER'S OWN GRAMMAR: `end <n>` and `error <why>` are its LAST
        # record. Without this the machine invents a driver that writes after
        # its own terminator and the oracle cries wolf on a stream no driver
        # emits — measured: the first control run shrank to exactly that.
        self.records.append(text)
        if text.startswith(("end ", "error ")):
            self.finished = True

    def _render_full(self):
        return "".join(r + "\n" for r in self.records)

    def _render(self):
        full = self._render_full()
        return full if self.truncate_bytes is None else full[: self.truncate_bytes]

    @invariant()
    def parent_never_trusts_an_unspoken_transcript(self):
        stdout = self._render()
        full = self._render_full()
        res = self.child.read(stdout, self.rc)
        msg = _judge(res, stdout, full, self.rc, type(self).census, where="stateful")
        if msg is not None:
            raise AssertionError(msg + f"\n  rules: {self.records!r}")


CvcTransport.TestCase.settings = _profiles.current().settings(
    400, stateful_step_count=10
)
TestCvcTransport = CvcTransport.TestCase


def test_the_state_machine_examined_the_protocol():
    """The state machine's anti-vacuity floor, asserted rather than assumed.

    ``RuleBasedStateMachine`` gives no place to put a floor inside the run, so
    it goes here — and this test is ordered after ``TestCvcTransport`` in the
    file, which is the order pytest collects in. If it is ever reordered ahead,
    it fails loudly (zero driven) rather than passing on an empty census, which
    is the correct direction for a tripwire.
    """
    # Measured at the `ci` profile (400 examples x 10 steps, derandomized):
    # driven=4133, refused=4107, definite=26. The definite floor is set at 10
    # rather than near 26 because it is a tripwire for the model collapsing,
    # not a target — and the ratio itself is the honest news about stateful
    # search on this protocol: 99.4% of the states it reaches are ones where
    # the parent correctly refuses, which is why the flat leg above finds the
    # same defects an order of magnitude cheaper.
    CvcTransport.census.require(driven=500, definite=10, refused=200)
