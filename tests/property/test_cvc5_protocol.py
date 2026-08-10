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

**A CONTROL THAT FIRES DOES NOT SAY WHICH OF THE THREE IT DEMONSTRATED, and for
most of this file's life exactly one of them was demonstrated by anything.**
``0ad22bb`` — the tree both ``cvc5-flat`` and ``cvc5-stateful`` point at —
violates clause (2) and only clause (2): measured over the flat leg's own 1500
``ci`` examples with all three clauses evaluated independently, 5 violations,
all of them (2). Clauses (1) and (3) are demonstrated by two further registered
controls: ``cvc5-exit-tell``, at commit ``8ef8f75``, and ``cvc5-phantom-model``,
a mutant. The measurements, and the one thing still undemonstrated (the FORGERY
route to clause (3)), are in ``_judge``'s docstring below.

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

POSITIVE CONTROLS, four of them, and which clause of the oracle each one
actually demonstrates:

* ``cvc5-flat`` / ``cvc5-stateful`` — commit ``0ad22bb``, the pre-fix
  ``splitlines()`` parser, where both legs fail. **Clause (2) only.**
* ``cvc5-exit-tell`` — commit ``8ef8f75``, before the transport had either of
  its two tells. **Clause (1).**
* ``cvc5-phantom-model`` — mutant, the harvested model deduped after the
  terminator check. **Clause (3), by duplication rather than by forgery.**

Run them with ``python tools/property_check.py --controls``.
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
    """What ``subprocess.run`` hands the transport, IN THE MODE IT WAS ASKED FOR.

    ``subprocess.run`` returns ``bytes`` on both streams unless the caller asks
    for text mode, and universal-newline-decoded ``str`` if it does. **Which of
    the two is not this fixture's choice to make.** It is the caller's, the
    caller is ``_run_cvc5_wheel``, and a positive control runs this file
    against a tree where that caller is a different one.

    Both halves have been got wrong here, in opposite directions, at the same
    cost. It first handed ``str`` unconditionally, which made the model a
    reader that does no decoding at all, so the ``\\r`` row of
    ``SPLITLINES_ONLY`` above was scored against a parent that existed in
    neither direction. It was then changed (``420cc12``) to hand ``bytes``
    unconditionally — right for today's transport, which spawns with no
    ``text=`` and decodes for itself in ``solvers._decode_child_stream``, and
    wrong for the transport at ``0ad22bb``, which spawns with ``text=True``.

    MEASURED, at ``0ad22bb``, on the first example of both legs: ``TypeError:
    sequence item 0: expected str instance, bytes found``, raised inside the
    parent's own protocol-violation message. Both cvc5 controls came back RED
    for a value ``subprocess`` would never have handed that parent, the defect
    they are registered for was never reached, and ``expect_message`` is the
    only reason this was reported as NOT DEMONSTRATED rather than as a control
    that fired.
    """

    # Universal-newline decoding is what text mode DOES (``io.TextIOWrapper``
    # with ``newline=None``): ``\r\n`` AND a bare ``\r`` become ``\n``. That is
    # not incidental here — it is the one translation
    # ``_decode_child_stream``'s docstring is about, and the ``\r`` row of
    # ``SPLITLINES_ONLY`` is only a real case against a ``text=True`` parent if
    # this fixture performs it. ``strict`` decoding, likewise, because that is
    # what text mode does and a child writing invalid UTF-8 RAISES out of the
    # transport there (measured, and tabulated in ``solvers.py``).
    #
    # THE CODEC IS NOT utf-8 BY RULE, AND THIS FIXTURE HARD-CODES IT. Real
    # ``text=True`` with ``encoding=None`` decodes with the LOCALE's preferred
    # encoding, not with utf-8. Measured on CPython 3.12.3 against a real child
    # writing ``opaque x0 q<U+0085>end 1``: with PEP 538 locale coercion and
    # PEP 540 UTF-8 mode both off (``PYTHONCOERCECLOCALE=0 PYTHONUTF8=0
    # LC_ALL=C``, preferred encoding ANSI_X3.4-1968) real ``subprocess`` raises
    # ``UnicodeDecodeError: 'ascii' codec can't decode byte 0xc2`` where this
    # fixture decodes cleanly. Bare ``LC_ALL=C`` does NOT reproduce it — 3.12
    # coerces the C locale to C.UTF-8 — so the gap needs an environment nobody
    # runs here, and it is not theoretical: three of the nine
    # ``SPLITLINES_ONLY`` characters are non-ASCII (U+0085, U+2028, U+2029), so
    # against the ``text=True`` parser at ``0ad22bb`` the real transport would
    # raise on those three rows in a non-UTF-8 locale while the property scores
    # a parse. It would raise LOUDLY, which is the safe direction, and the
    # property would be measuring the wrong parent all the same.
    def __init__(self, stdout, returncode, *, text):
        raw = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
        if text:
            self.stdout = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            self.stderr = ""
        else:
            self.stdout = raw
            self.stderr = b""
        self.returncode = returncode


class _FakeSubprocess:
    """Stands in for the ``subprocess`` module inside ``stelling.solvers``."""

    TimeoutExpired = _real_subprocess.TimeoutExpired

    # CPython opens the child's streams in text mode if ANY of these is given,
    # not just ``text=`` (``subprocess.Popen.__init__``). Reading only ``text``
    # would be right about the two trees this file is pointed at today and
    # wrong about the next one for a reason nobody would look for.
    #
    # HOW FAR THAT "AND THE NEXT ONE" REACHES, driven against the real
    # ``subprocess`` module with a real child on CPython 3.12.3: twelve keyword
    # spellings, of which ten agree with this rule, and one POSITIONAL spelling
    # that defeats it outright.
    #
    #   universal_newlines passed POSITIONALLY — real ``str``, model ``bytes``.
    #     ``run(*popenargs, **kwargs)`` forwards positionals straight to
    #     ``Popen``, where ``universal_newlines`` is the 11th parameter AFTER
    #     the command and the 12th counting ``args`` itself — ten filler
    #     positionals (``bufsize`` through ``env``) stand between them.
    #     ``run(self, *a, **k)`` below reads ``k`` alone, so a positional is
    #     invisible to it. That is precisely the forward-looking case the
    #     paragraph above claims to protect against, and it is not protected.
    #   text=True, universal_newlines=False — real raises ``SubprocessError``
    #     ("Cannot disambiguate when both text and universal_newlines are
    #     supplied but different"); the model returns text.
    #   encoding='latin-1' (with or without ``text=True``) — real decodes
    #     latin-1; ``_FakeProc`` always decodes utf-8.
    #
    # NONE OF THE THREE IS LIVE for either tree this file is pointed at — the
    # tip spawns with no io kwargs at all, ``0ad22bb`` spawns ``text=True`` by
    # keyword — so they are limits on the claim rather than on today's controls.
    # A transport that spawned any of the three would be scored against the
    # wrong parent here rather than caught, which is the same failure this
    # fixture was repaired for at ``f00375a``.
    _TEXT_MODE = ("text", "universal_newlines", "encoding", "errors")

    def __init__(self):
        self.stdout = ""
        self.rc = 0

    def run(self, *a, **k):
        # THE SPAWN KWARGS ARE THE TRANSPORT'S AND THIS READS THEM RATHER THAN
        # NAMING ITS OWN — the same repair ``_wheel_child`` in
        # ``tests/test_solver_audit_findings.py`` carries, for the same reason:
        # a fixture that names the io mode itself scores the FIXTURE's choice.
        return _FakeProc(
            self.stdout, self.rc, text=any(k.get(n) for n in self._TEXT_MODE)
        )


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
    """The invariant. Returns a failure message or ``None``.

    WHICH OF THE THREE CLAUSES THE POSITIVE CONTROL ACTUALLY DEMONSTRATES,
    written down because "the control fires" and "the property is shown to
    find what it claims to find" are different statements and this file has
    only ever checked the first.

    RE-MEASURED, BECAUSE THE FIRST VERSION OF THIS TABLE WAS A CENSUS OF THE
    WRONG TREE. Both rows are the ci profile's own derandomized 1500 examples,
    judged by a non-raising counter that evaluates all three clauses
    INDEPENDENTLY instead of short-circuiting. ``search``'s own source and this
    module's file are untouched by the instrumentation, so hypothesis's
    function digest — and with it the example sequence — is the shipped one:
    the 1500 drawn ``(full, stdout, rc)`` triples are byte-identical, in order,
    at both trees::

        tree       refused  definite  V(1)  V(2)  V(3)
        0ad22bb      1392      108       0     5     0
        the tip      1418       82       0     0     0

    ``1418 refused / 82 definite`` is **the tip's** census and was published
    here as ``0ad22bb``'s. It cannot be ``0ad22bb``'s: that parser answers
    definitely everywhere the tip does and on 26 transcripts more — the ones
    the tip's alphabet backstop refuses — so ``definite(0ad22bb) >=
    definite(tip)`` pointwise, checked example by example on this sequence. A
    second, differently seeded pair of 1500-example runs gives 88 definite at
    ``0ad22bb`` against 67 at the tip: same direction, different totals. The
    ``3 VIOLATED (2)`` published beside them reproduces at neither tree under
    this instrumentation and is WITHDRAWN. The measurement is 5, and every one
    of the 5 is clause (2) — which is the one conclusion the old table was
    offered in support of, and it survives the correction.

    (2) NOTHING WAS TRUNCATED — DEMONSTRATED, by both legs, in both of the
    shapes ``cvc5-flat``'s ``why`` names, and the split between them is **4 and
    1**, not the 3 and 3 reported when the fixture was repaired::

        4  the final record's newline cut and nothing else
             read 'version 1.2.0\\nanswer sat\\nvalue x0_0 17/4\\nend 1'
             full 'version 1.2.0\\nanswer sat\\nvalue x0_0 17/4\\nend 1\\n'
        1  a payload separator forging the terminator while the child was
           killed mid-model-walk, exit 0
             read 'version 1.3.4\\nanswer unknown\\nopaque x0_0 q\\u2028end 1\\n'
             full 'version 1.3.4\\nanswer unknown\\nopaque x0_0 q\\u2028end 1\\nend 9\\n'

    The QUALITATIVE claim that split was offered for is true and is what
    matters: both shapes are reached by the SEARCH rather than only by the
    shrink. The counter above never raises, so nothing shrank — all 1500 are
    draws — and both shapes are among them. The same 1500 draws produce zero
    failures against the tip.

    The pair printed with the old split cannot be one draw, and the property's
    own construction says so: it builds ``stdout = full[:cut]``, so the read is
    always a PREFIX of the full. The quoted read (``opaque x0_0 q\\x0cend 1``)
    is not a prefix of the quoted full (``opaque x0 q\\x1cq``) — different name,
    different separator. Two different examples were printed as one.

    (1) THE CHILD EXITED 0 — DEMONSTRATED, by the ``cvc5-exit-tell`` control at
    commit ``8ef8f75``, and by nothing at ``0ad22bb``. It cannot be
    demonstrated at ``0ad22bb``: by then the parser already refused a nonzero
    exit (``if not complete or proc.returncode != 0``, its "two tells"), so
    that tree carries the other defect and not this one.

    THIS DOCSTRING DREW TWO WRONG CONCLUSIONS FROM THAT IN A ROW. First, that
    the clause "has no place it is known to fail and therefore no
    demonstration anywhere" — which did not follow, because a registry entry
    does not have to name a commit. Then, registering it as a MUTANT on the
    ground that no revision of this tree had carried it — which is simply
    false, and one line of git says so::

        git log -S "or proc.returncode != 0" -- src/stelling/solvers.py
          b6e4783  The cvc5 wheel transport refuses a crashed child

    The guard was ADDED there, with BOTH tells at once, so its parent
    ``8ef8f75`` — a revision this registry already names, for
    ``cross-series`` — has neither. Hand-driven through this file's own
    fixture at ``8ef8f75``: ``'version 1.3.4\\nanswer sat\\nend 0\\n'`` at exit
    1 is a definite ``sat``, and ``'…\\nvalue x0 0/1\\nend 1\\n'`` at exit 137
    is a definite ``sat`` harvesting ``('x0', '0/1')``. Today's flat property
    fails there at the ci profile on an INTACT transcript at exit 1 — so
    clause (2) holds on that example and the failure is this clause's own.

    **AT THE ci PROFILE**, AND THAT IS A CLAIM ABOUT THE SEQUENCE, NOT ABOUT
    THE TREE. ``8ef8f75`` predates the splitlines fix too, so it violates
    clause (2) as well as clause (1); this function returns on the FIRST
    clause that fails and tests (2) first; so which clause a run reports is
    decided by which violating example the search reaches first. Counted
    independently over 1500 draws at ``8ef8f75`` — by a separate non-raising
    counter, so these are that counter's totals and not ``search``'s
    sequence — 458 examples violate clause (1), 284 violate clause (2), 607
    violate at least one, and this function would report (2) on 284 of them
    and (1) on 323. Measured through ``property_check.py``::

        --control cvc5-exit-tell                    FIRED       x1, x2, x4
        --control cvc5-exit-tell --profile dev      0/1 fired   3 of 3 runs
          what pytest recorded: clause (2)'s sentence, not clause (1)'s
        --control cvc5-exit-tell --profile dev --scale 2   0/1 fired

    ``ci`` is derandomized and ``dev`` is not, so this is not a budget effect:
    the clause-(1) demonstration is stable under ×4 at ``ci`` and absent at
    ``dev``. The DIRECTION IS SAFE — the tool refuses rather than passes, and
    no run of this can produce a false green — but
    ``tools/property_check.py --controls --profile dev`` is a documented
    invocation that reports this control NOT DEMONSTRATED, and that was not
    written down anywhere until now.

    WHY THE OBVIOUS FIX IS NOT TAKEN, measured rather than argued. Pinning
    ``'version 1.3.4\\nanswer sat\\nend 0\\n'`` at exit 1 as an ``@example`` on
    ``search`` makes the demonstration seed-independent and it WORKS — driven,
    the control fires at ``dev`` as well as at ``ci``. It also edits
    ``search``'s own source, and hypothesis keys its derandomized sequence on
    ``function_digest(search)``: driven, the example this property reports at
    ``0ad22bb`` moved from ``'…answer unknown\\nvalue x0_0 0/1\\nend 1'`` to
    ``'…answer sat\\nopaque x0 q\\nopaque x0 q\\nend 2'``. Every figure in this
    docstring attributed to the SHIPPED digest — the two census rows above,
    the 4-and-1 split, "byte-identical, in order, at both trees" — and the
    673/8165 example-efficiency pair in the module docstring would then be
    describing a sequence that no longer exists. That re-take is the price of
    the pin; it is not paid here, and the claim is qualified instead.

    THE COST, WITH ITS INSTRUMENT NAMED, because the two figures this entry
    has carried are of two different ones and neither said which. Three runs
    each, on a box doing other work: **0.45 / 0.38 / 0.28 s** of junit XML
    ``<testcase time=>``, and **1.59 / 1.77 / 1.23 s** of wall clock for the
    whole ``property_check.py --control`` invocation, which also materialises
    a tree and starts an interpreter. The ``0.9 s`` in the ci.yml step comment
    is the second of those.

    (3) THE MODEL IS EXACTLY THE VALUE RECORDS IN THE BYTES READ — DEMONSTRATED,
    by the ``cvc5-phantom-model`` mutant control, and the shape it needs was
    inside the alphabet all along. Two DIFFERENT ways to break this clause were
    run together here and they are separated now.

    A DUPLICATE value record is drawable today: ``_NAMES`` x ``_RATIONALS`` is
    15 combinations over up to three records, so ``value x0 0/1`` twice is an
    ordinary draw. Dedupe the harvested model AFTER the terminator check
    (``tuple(sorted(values))`` -> ``tuple(sorted(set(values)))``) and the count
    in ``end <n>`` still matches, so clauses (1) and (2) both hold and only this
    one can fail: the third branch below fires on
    ``'version 1.3.4\\nanswer sat\\nvalue x0 0/1\\nvalue x0 0/1\\nend 2\\n'``,
    reporting a harvested ``[('x0', '0/1')]`` against an actually-present
    ``[('x0', '0/1'), ('x0', '0/1')]``. Green against the unmutated tip.

    IT IS THE SLOWEST OF THE THREE cvc5 FLAT-LEG CONTROLS ON EITHER
    INSTRUMENT, and that qualitative claim is the only one the numbers were
    ever offered for. The numbers themselves have been recorded twice, of two
    different instruments, with neither labelled. Three runs each, on a box
    doing other work::

                            junit <testcase time=>   whole --control run
        cvc5-flat            1.51 / 1.30 / 1.23      2.78 / 2.29 / 2.31
        cvc5-exit-tell       0.45 / 0.38 / 0.28      1.59 / 1.77 / 1.23
        cvc5-phantom-model   2.43 / 4.25 / 2.40      6.91 / 7.09 / 8.15

    The ``2.0 s`` recorded when this control was registered is a junit figure
    and it REPRODUCES. The ``5.6 s best of three, 9.1 s worst`` that replaced
    it — and declared it irreproducible — is the wall clock of the whole
    invocation, so the correction changed instrument and then read the
    difference as a failure to reproduce. It is WITHDRAWN, and both instruments
    are written down here rather than one of them being picked. ``_PAYLOADS``
    is untouched,
    so the two example-efficiency figures in the module docstring keep
    describing the strategy that produced them.

    A FORGED value record is still out of reach, and that half of the finding
    stands. It is the shape ``0ad22bb`` actually gets wrong — hand-driven
    through this file's own fixture, no truncation, exit 0, count matching::

        version 1.3.4
        answer sat
        opaque x1 q\\x0bvalue x9 1/2      <- ONE record to the writer
        end 2

      0ad22bb: sat, harvesting x9 = 1/2, which the writer never wrote
      tip    : failed

    and no draw from ``_PAYLOADS`` can express it. Its separator payloads are
    ``q<sep><tail>`` with ``tail`` in ``end <n>`` / ``answer unsat`` / ``q``,
    and a forged ``value`` record needs a tail beginning ``value ``. A forged
    ``end`` harms only a TRUNCATED child, so it is caught as clause (2); a
    forged ``value`` is the one that harms an intact one. So what remains
    undemonstrated is narrower than "clause (3)": it is **the forgery route to
    clause (3), against the one tree that historically carried it.** Adding a
    ``value x9 1/2`` tail is what would put that route in the search, and the
    cost is re-measuring 673/8165; that trade is still open and still the
    principal's. It is no longer the price of demonstrating the clause.

    ────────────────────────────────────────────────────────────────────────

    HOW TO READ THIS DOCSTRING: **the three failure messages are deliberately
    not quoted in it.** pytest's long traceback prints a frame's ENTIRE
    function source, docstring included, from ``def`` down to the failing line,
    and ``_judge`` is on the traceback of anything that raises INSIDE it. A
    message quoted here is therefore echoed into the output
    ``tools/property_check.py`` captures, and a control whose
    ``expect_message`` is that message then scores a CRASH as a demonstration.
    Measured, with two of the three quoted here as they were: one line-neutral
    defect in ``solvers.py`` (``values=tuple(sorted(values)),`` -> ``... or
    None,``) makes ``sorted(res.values)`` raise ``TypeError`` before the oracle
    is evaluated at all, and three probe controls carrying the three shipped
    guard strings reported ``3/3 controls fired``. Read the messages off the
    ``return``s below instead. ``property_check.py`` now matches
    ``expect_message`` against the failure pytest RECORDS rather than against
    everything it echoes, which is the half of this a docstring cannot enforce,
    and ``test_suite_disclosure.py`` fails on the push that re-adds a quote.
    """
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
