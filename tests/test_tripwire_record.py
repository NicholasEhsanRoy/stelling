# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The tripwire's arithmetic, attribution and report — **without jax**.

Every test here runs in the zero-dep lane. That is not a convenience: it is
the property ``PLAN-tripwire.md`` §3 asks for, since the same discipline that
keeps ``record.py`` jax-free is what makes a finding serialisable across an
xdist process boundary. If any of these grows a jax import, the boundary has
leaked.

None of these transcribes what it should extract. The narrowing table is
generated from the dtype widths, the report assertions read the rendered text
back for facts the renderer had to compute, and the attribution tests are
driven from stack shapes MEASURED against real jax (recorded in
``record.attribute``'s docstring) rather than from what the code does.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import re
import shlex

import pytest

from stelling._tripwire import record, report


@pytest.fixture(autouse=True)
def _neither_instrument_is_left_in_a_different_state():
    """Every test in this file leaves the process's hooks as it found them.

    THIS FILE IS PURE PYTHON AND STILL DRIVES THE CANARY, which arms and
    disarms both instruments for real. One test here stubbed the tripwire's
    half and not the eager one, so ``canary.main()`` called the real
    ``disarm_eager()`` and a session running with
    ``--stelling-eager-truncation=error`` lost its detector at this file and
    ran every later file unwatched -- silently, because the escalation that
    would have said so was unreachable. The rule is the same one
    ``tests/test_tripwire_eager.py`` applies to itself, and it is asserted
    rather than trusted because "silently" is the whole problem.

    It reads the state through a helper that answers ``False`` with no jax
    installed, so it costs the zero-dependency lane nothing.
    """
    from stelling._tripwire import eager as _eager

    def state():
        try:
            return _eager.is_armed(), _tripwire_is_armed()
        except Exception:  # noqa: BLE001 - a guard may not break the suite
            return None

    def _tripwire_is_armed():
        try:
            from stelling import _tripwire

            return _tripwire.is_armed()
        except Exception:  # noqa: BLE001
            return None

    before = state()
    yield
    assert state() == before, (
        "a test in this file changed the process's arm state: it was "
        f"{before} and is now {state()}. A test that takes an instrument out "
        "must put it back -- a session armed session-wide runs every later "
        "file unwatched otherwise."
    )


def test_record_and_report_pull_in_no_jax():
    """The boundary claim, measured in a fresh interpreter.

    Not readable from this process: by the time the suite reaches here, other
    modules have imported jax and ``sys.modules`` would say yes for the wrong
    reason. A subprocess is the only place the question has an answer, and it
    carries the same positive control ``test_importing_stelling_pulls_in_no_jax``
    uses — where jax IS installed, force it in and confirm the probe can see
    it, or an environment without jax reports the same empty list having
    measured nothing.
    """
    import json
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    import stelling

    src = Path(stelling.__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(
            """
            import importlib.util, json, sys
            from stelling._tripwire import record, report
            out = {
                "jax_modules": sorted(m for m in sys.modules if m.split(".")[0] == "jax"),
                "jax_installed": importlib.util.find_spec("jax") is not None,
                "rendered": bool(report.render_denominator(record.Recorder())),
            }
            if out["jax_installed"]:
                __import__("jax")
                out["visible_when_forced"] = "jax" in sys.modules
            print(json.dumps(out))
            """
        )],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": str(src), "JAX_PLATFORMS": "cpu"},
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stdout}\n{proc.stderr}"
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["rendered"], "the probe imported the modules but never used them"
    assert got["jax_modules"] == [], (
        "importing stelling._tripwire.record or .report pulled jax in: "
        + ", ".join(got["jax_modules"])
        + ". Both are supposed to be pure Python: that is what makes them "
        "testable in the zero-dep lane AND what makes a Finding survive "
        "execnet."
    )
    if got["jax_installed"]:
        assert got["visible_when_forced"] is True, got


# --- the arithmetic ---------------------------------------------------------


@pytest.mark.parametrize("dtype", sorted(record.INT_DTYPES))
def test_the_range_is_the_dtype_width_not_a_table_of_numbers(dtype):
    """Derived from the width, so a typo in one bound cannot agree with itself."""
    signed, bits = record.INT_DTYPES[dtype]
    lo, hi = record.dtype_range(dtype)
    assert hi - lo + 1 == 2**bits
    assert (lo < 0) is signed
    assert record.in_range(lo, dtype) and record.in_range(hi, dtype)
    assert not record.in_range(lo - 1, dtype)
    assert not record.in_range(hi + 1, dtype)


@pytest.mark.parametrize("dtype", sorted(record.INT_DTYPES))
def test_narrow_is_two_s_complement_truncation_at_every_width(dtype):
    """The independent route, checked against the definition rather than
    against itself: ``narrow(v)`` must be the unique representable value
    congruent to ``v`` modulo ``2**bits``."""
    signed, bits = record.INT_DTYPES[dtype]
    lo, hi = record.dtype_range(dtype)
    modulus = 2**bits
    for value in (0, 1, -1, hi, hi + 1, lo, lo - 1, modulus, modulus + 7, -modulus - 3):
        got = record.narrow(value, dtype)
        assert lo <= got <= hi, (dtype, value, got)
        assert (got - value) % modulus == 0, (dtype, value, got)


def test_the_headline_case_by_hand():
    """``x + 256`` on int8. The one a reader checks with a pencil."""
    assert record.narrow(256, "int8") == 0
    assert record.narrow(300, "int8") == 44
    assert record.narrow(200, "int8") == -56
    assert record.narrow(-129, "int8") == 127
    assert record.narrow(256, "uint8") == 0
    assert record.narrow(-1, "uint8") == 255


def test_an_unmodelled_dtype_returns_none_rather_than_guessing():
    for dtype in ("float32", "bfloat16", "bool", "complex64", "", "int7"):
        assert record.narrow(5, dtype) is None
        assert record.in_range(5, dtype) is None
        assert record.dtype_range(dtype) is None


@pytest.mark.parametrize("dtype", sorted(record.INT_DTYPES))
def test_the_arithmetic_sentence_states_the_number_it_computes(dtype):
    """§10a.3 — and the sentence must AGREE with :func:`record.narrow`, or the
    reader's pencil and the tool disagree in print."""
    _, bits = record.INT_DTYPES[dtype]
    for value in (2**bits, 2**bits + 5, -1, 3):
        sentence = record.arithmetic_sentence(value, dtype)
        assert f"2**{bits}" in sentence
        assert str(record.narrow(value, dtype)) in sentence


# --- attribution ------------------------------------------------------------

JAXROOT = "/venv/site-packages/jax/"

# The two stacks measured on real jax and recorded in record.attribute's
# docstring, transcribed here as SHAPES: outermost first, with the frames that
# decide the answer kept and the rest elided. The point of each is the
# position of `trace_to_jaxpr_nocache` relative to the first non-jax frame.
USER_STACK = (
    ("/home/u/t.py", 10, "<module>"),
    (JAXROOT + "_src/api.py", 2137, "make_jaxpr_f"),
    (JAXROOT + "_src/interpreters/partial_eval.py", 2290, "trace_to_jaxpr_nocache"),
    ("/home/u/t.py", 54, "widen"),  # <- the writer
    (JAXROOT + "_src/numpy/array_methods.py", 1532, "op"),
    (JAXROOT + "_src/interpreters/partial_eval.py", 2470, "try_constant_folding"),
)
JAX_STACK = (
    ("/home/u/t.py", 55, "<module>"),  # <- only the CALLER of jax.random.key
    (JAXROOT + "_src/random/prng.py", 563, "random_seed"),
    (JAXROOT + "_src/interpreters/partial_eval.py", 2290, "trace_to_jaxpr_nocache"),
    (JAXROOT + "_src/random/threefry2x32.py", 73, "_threefry_seed"),  # <- jax's writer
    (JAXROOT + "_src/numpy/ufunc_api.py", 182, "__call__"),
    (JAXROOT + "_src/interpreters/partial_eval.py", 2470, "try_constant_folding"),
)


def test_a_user_written_constant_is_attributed_to_the_user_s_own_line():
    index, origin = record.attribute(USER_STACK, JAXROOT)
    assert origin == record.ORIGIN_USER
    assert USER_STACK[index] == ("/home/u/t.py", 54, "widen")


def test_a_jax_written_constant_is_not_blamed_on_the_caller():
    """The whole reason attribution is a filter. ``jax.random.key(0)`` is one
    user line calling into jax, and jax writes ``0xFFFFFFFF`` inside its own
    traced function. "Innermost non-jax frame" answers ``t.py:55`` here, which
    would tell a user they wrote a constant they have never seen."""
    index, origin = record.attribute(JAX_STACK, JAXROOT)
    assert origin == record.ORIGIN_JAX
    assert JAX_STACK[index][2] == "_threefry_seed", (
        "a suppressed fire must still NAME jax's own writer, not the fold "
        "site and not the user's call: §10a.9, a silent filter and a blind "
        "instrument look the same."
    )
    # and the naive rule really does get it wrong, which is what makes the
    # filter load-bearing rather than decorative
    naive = next(i for i in range(len(JAX_STACK) - 1, -1, -1)
                 if not JAX_STACK[i][0].startswith(JAXROOT))
    assert JAX_STACK[naive] == ("/home/u/t.py", 55, "<module>")


def test_with_no_trace_entry_at_all_the_fallback_is_lenient_and_says_so():
    """A jax that renamed its trace entries must not silently swallow
    findings. The fallback over-reports (innermost non-jax frame) rather than
    under-reporting, because an over-report is visible to a user holding the
    quoted line and the reproducer."""
    renamed = tuple(
        (f, ln, "some_new_name" if fn == "trace_to_jaxpr_nocache" else fn)
        for f, ln, fn in JAX_STACK
    )
    index, origin = record.attribute(renamed, JAXROOT)
    assert origin == record.ORIGIN_USER
    assert renamed[index] == ("/home/u/t.py", 55, "<module>")


def test_an_all_jax_stack_with_no_frames_inside_the_entry_is_unattributed():
    stack = (
        (JAXROOT + "_src/api.py", 1, "f"),
        (JAXROOT + "_src/interpreters/partial_eval.py", 2290, "trace_to_jaxpr_nocache"),
    )
    index, origin = record.attribute(stack, JAXROOT)
    assert (index, origin) == (None, record.ORIGIN_JAX)
    assert record.attribute((), JAXROOT) == (None, record.ORIGIN_UNKNOWN)


# --- the same question, asked at the EAGER site -----------------------------
#
# `attribute` answers "who wrote this constant?" from the FRAMES, using the
# trace boundary. At an EAGER narrowing there is no trace boundary, and the
# two stacks above collapse to the same shape -- a user frame with nothing but
# jax frames beneath it. What answers it there is an ENUMERATION of jax's own
# eager truncations rather than a rule over the data, and these are its pure
# Python half; the jax half -- the sweep that re-derives the enumeration, and
# the arm-time control that drives its one row -- is
# `tests/test_tripwire_eager.py`.
#
# A GENERAL PREDICATE OVER THE DATA STOOD HERE and its tests stood here with
# it: `record.carries(values, written)`, "is the narrowed integer among the
# arguments of the call that crossed into jax?". It was withdrawn because it
# was measurably wrong in both directions -- it suppressed
# `jax.tree.map(partial(jnp.full_like, fill_value=300), tree)`, a constant the
# user really wrote, in the default jit-on configuration, and it raised on
# jax's own mask whenever its container scan ran out of budget. The argument
# is in `_adapter_jax._JAX_EAGER_CONSTANTS`.


def test_a_narrowing_at_no_enumerated_jax_site_is_the_CALLER_S():
    """The default answer, and it is the one that fails closed.

    An empty map -- which is what an unarmed process has -- must attribute
    everything to the caller. A map that answered "jax's" by default would
    turn every jax release this repository has not read into a silence.
    """
    from stelling._tripwire import eager

    assert eager.jax_constant(4294967295, "uint32", "int32", ()) is None
    assert eager.jax_constant(
        300, "int", "int8", (("_src/lax/lax.py", "full"),)
    ) is None


def test_a_row_matches_on_the_SITE_and_the_VALUE_and_BOTH_DTYPES_together():
    """A row is a statement about one constant, arriving one way, at one site.

    Not a licence for the function it names -- a jax function that writes one
    constant of its own still narrows the caller's constants too -- and not a
    licence for the value anywhere else.

    THE SOURCE DTYPE IS IN THE KEY, and an audit is why. Without it a row
    about jax's own constant also suppresses a CALLER'S constant of the same
    value narrowed at the same jax function, and at the one site the map
    names those collide:
    ``jax.extend.random.threefry_prng_impl.seed(np.int64(2**32 - 1))``
    narrows twice under ``_threefry_seed`` -- the caller's seed and jax's
    mask, both ``4294967295 -> -1`` at ``int32`` -- and the three-field row
    suppressed both, then printed "written by jax ... the threefry PRNG's
    32-bit mask" at the caller's own line. The two differ in exactly one
    observable, which is the field added here: jax's mask arrives from
    ``uint32`` and a caller's seed from ``int64``. Driven against the real
    jax in ``tests/test_tripwire_eager.py``.
    """
    from stelling._tripwire import eager

    rows = {("f.py", "writer"): ((4294967295, "uint32", "int32", "the mask"),)}
    saved = dict(eager._JAX_CONSTANTS)
    eager._JAX_CONSTANTS.clear()
    eager._JAX_CONSTANTS.update(rows)
    try:
        run = (("g.py", "narrower"), ("f.py", "writer"), ("h.py", "caller"))
        assert eager.jax_constant(
            4294967295, "uint32", "int32", run
        )[:2] == ("f.py", "writer")
        # the right site, the wrong value
        assert eager.jax_constant(300, "uint32", "int32", run) is None
        # the right site and value, the wrong TARGET dtype
        assert eager.jax_constant(4294967295, "uint32", "int16", run) is None
        # the right site, value and target -- the wrong SOURCE dtype. This is
        # the caller's colliding constant, and it must NOT be suppressed.
        assert eager.jax_constant(4294967295, "int64", "int32", run) is None
        # the right value and dtypes, no site
        assert eager.jax_constant(
            4294967295, "uint32", "int32", (("g.py", "narrower"),)
        ) is None
        # the right file, a different function in it
        assert eager.jax_constant(
            4294967295, "uint32", "int32", (("f.py", "somebody_else"),)
        ) is None
    finally:
        eager._JAX_CONSTANTS.clear()
        eager._JAX_CONSTANTS.update(saved)


def test_the_row_carries_a_SENTENCE_and_the_report_prints_it():
    """A suppression nobody can read is the silence this instrument ends.

    The third field of a row is what the constant IS, and it reaches the
    report -- so a reader meets "the threefry PRNG's 32-bit mask" rather than
    a number and a file path they have to go and interpret.
    """
    from stelling._tripwire import eager

    saved = dict(eager._JAX_CONSTANTS)
    eager._JAX_CONSTANTS.clear()
    eager._JAX_CONSTANTS.update(
        {("f.py", "writer"): ((4294967295, "uint32", "int32", "the threefry mask"),)}
    )
    try:
        row = eager.jax_constant(
            4294967295, "uint32", "int32", (("f.py", "writer"),)
        )
        assert row == ("f.py", "writer", "the threefry mask")
    finally:
        eager._JAX_CONSTANTS.clear()
        eager._JAX_CONSTANTS.update(saved)


def test_the_residue_the_enumeration_leaves_is_DISCLOSED():
    """What a map cannot do is know a row nobody has written.

    The direction it fails in is the loud one -- an unenumerated constant of
    jax's own is attributed to the caller and raises -- and that is a claim
    about this instrument that has to be printed rather than left in a
    docstring.
    """
    text = " ".join(report.EAGER_UNCOVERED).upper()
    assert "ENUMERAT" in text, "the map's residue is not disclosed anywhere"
    assert "RAISES" in text, (
        "the disclosure does not say which direction the residue fails in"
    )


def test_the_user_chain_is_the_traced_region_not_the_whole_stack():
    """Everything outside the trace entry is the runner that got you here --
    forty lines of runpy/pluggy/_pytest under a real pytest session, measured.
    The frames that carry the cross-module story are the ones inside."""
    assert record.user_chain(USER_STACK, JAXROOT) == (("/home/u/t.py", 54, "widen"),)
    # ... and with no trace entry to anchor on, the lenient fallback keeps all
    renamed = tuple(
        (f, ln, "moved" if fn == "trace_to_jaxpr_nocache" else fn)
        for f, ln, fn in USER_STACK
    )
    assert record.user_chain(renamed, JAXROOT) == (
        ("/home/u/t.py", 10, "<module>"),
        ("/home/u/t.py", 54, "widen"),
    )


# --- the finding, and the boundary it has to cross --------------------------


def _finding(**kw):
    base = dict(
        file="/home/u/t.py",
        line=54,
        func="widen",
        written=256,
        from_dtype="int64",
        to_dtype="int8",
        became=0,
        origin=record.ORIGIN_USER,
        chain=(("/home/u/t.py", 10, "<module>"), ("/home/u/t.py", 54, "widen")),
    )
    base.update(kw)
    return record.Finding(**base)


def test_a_finding_survives_the_execnet_hop_as_primitives_only():
    """§6 restricts the payload to primitives and §2 says *verify* it. This
    checks the type of every leaf rather than only the round trip, because a
    round trip inside one process would pass on a payload execnet refuses."""
    payload = _finding().as_tuple()

    def leaves(obj):
        if isinstance(obj, (tuple, list)):
            for item in obj:
                yield from leaves(item)
        else:
            yield obj

    for leaf in leaves(payload):
        assert isinstance(leaf, (str, int, bool, type(None))), (leaf, type(leaf))
    assert record.Finding.from_tuple(payload) == _finding()


def test_dedup_is_by_file_line_written_dtype_and_counts_rather_than_drops():
    rec = record.Recorder()
    rec.add(_finding())
    rec.add(_finding())
    rec.add(_finding(line=99))
    assert rec.count == 2
    seen = {f.line: f.count for f in rec.sorted_findings()}
    assert seen == {54: 2, 99: 1}


def test_a_suppressed_finding_goes_to_its_own_bucket_not_the_findings():
    rec = record.Recorder()
    rec.add(_finding(origin=record.ORIGIN_JAX))
    assert rec.count == 0
    assert len(rec.sorted_suppressed()) == 1


def test_the_cap_is_disclosed_rather_than_silent():
    rec = record.Recorder(cap=3)
    for line in range(20):
        rec.add(_finding(line=line))
    assert rec.count == 3
    assert rec.dropped_over_cap == 17
    text = "\n".join(report.render_denominator(rec))
    assert "TRUNCATED" in text and "17" in text and "not a total" in text


def test_merging_two_workers_sums_counts_rather_than_duplicating_findings():
    """The xdist merge, driven without xdist: two workers that each traced the
    same wrap must report one finding seen twice."""
    left, right = record.Recorder(), record.Recorder()
    left.invocations, left.int_narrowings = 10, 4
    right.invocations, right.int_narrowings = 7, 3
    left.add(_finding())
    right.add(_finding())
    right.add(_finding(line=99))

    merged = record.Recorder()
    merged.absorb(left.as_payload())
    merged.absorb(right.as_payload())

    assert merged.invocations == 17 and merged.int_narrowings == 7
    assert merged.count == 2
    assert {f.line: f.count for f in merged.sorted_findings()} == {54: 2, 99: 1}


# --- the report is a witness ------------------------------------------------


def _Status(code="armed", detail="", rule_hash=None, known_hash=None,
            jax_version=None):
    """THE SHIPPED ``Status``, not a stand-in.

    It was a hand-written stub whose ``explanation`` returned ``detail``, and
    the render tests below therefore could not see that the primary channel
    prints ``NOT ARMED [no-module] --`` with nothing after the dash. A stub
    that reimplements the object under test measures the stub. ``Status``
    imports only :mod:`dataclasses`, so this stays runnable in a bare
    interpreter, which is the reason the stub existed.
    """
    from stelling._tripwire import Status

    return Status(
        code=code,
        detail=detail,
        rule_hash=rule_hash,
        known_hash=known_hash,
        jax_version=jax_version,
        rule_name="_convert_elt_type_folding_rule",
    )


def _rendered(rec, status=None):
    return "\n".join(report.render(status or _Status(), rec))


@pytest.fixture(scope="module")
def written_source(tmp_path_factory):
    """A real file on disk, so the quoted-line half of §10a.1 is exercised.

    Two lines with the same narrowing and different spellings: one with the
    literal on the line, one behind a name. The report has to treat them
    differently and a synthetic path would exercise neither.
    """
    path = tmp_path_factory.mktemp("tw") / "model.py"
    path.write_text(
        "def widen(x):\n"
        "    return x + 256\n"
        "\n"
        "LIMIT = 300\n"
        "\n"
        "def scale(x):\n"
        "    return x * LIMIT\n",
        encoding="utf-8",
    )
    return path


def test_every_finding_carries_the_six_things_that_make_it_a_witness(written_source):
    """§10a 1-6, read back out of the rendered text."""
    rec = record.Recorder()
    rec.invocations, rec.folded, rec.int_narrowings = 12, 9, 4
    rec.add(_finding(file=str(written_source), line=2, literal_visible=True))
    text = _rendered(rec)

    assert f"{written_source}:2" in text  # 1: the site, with file:line
    assert "2 | return x + 256" in text  # 1: the user's OWN line, quoted
    assert "innermost frame OUTSIDE JAX" in text  # 1: the rule, stated
    assert "the constant written there is 256" in text  # 2
    assert "int8 holds that as 0" in text  # 2
    assert "256 mod 2**8 = 0" in text  # 3: the arithmetic
    assert "jax.make_jaxpr(lambda a: a + 256)" in text  # 4: a reproducer
    assert "CONFIRMED" in text and "without the hook: 0" in text  # 5
    assert "OBSERVED" in text and "INFERENCE (not observed" in text  # 6


def test_a_THIRD_PARTY_constant_is_not_reported_as_something_YOU_wrote():
    """The origin filter has exactly one boundary — jax's own tree — and the
    report described it as "your own code".

    Driven with a real module in a venv's ``site-packages``: a constant written
    inside a third-party library came back under *"1 distinct out-of-range
    integer narrowing(s) in your own code"*, with *"you wrote 128"* and
    *"RULE attribution: the innermost frame of YOUR code"*. The site was named
    correctly and is checkable — the framing around it was the wrong claim, and
    it sends a reader looking for something they did not write.

    The same shape here, at report level, where every branch of it is
    reachable without installing anything.
    """
    rec = record.Recorder()
    rec.invocations, rec.folded, rec.int_narrowings = 4, 3, 2
    rec.add(
        _finding(
            file="/venv/lib/python3.12/site-packages/thirdparty/_kernels.py",
            line=8,
            func="apply_gain",
            written=128,
            became=-128,
        )
    )
    text = _rendered(rec)

    # the site is named, and that is the half that was already right
    assert "site-packages/thirdparty/_kernels.py:8 in apply_gain" in text
    # ...and nothing around it says whose code it is
    assert "in your own code" not in text
    assert "you wrote 128" not in text
    assert "YOUR code" not in text
    assert "written outside jax" in text
    assert "not the same as BY YOU" in text
    assert "innermost frame OUTSIDE JAX" in text


def test_a_finding_whose_replay_disagrees_is_withheld_not_printed():
    """§10a.5, and it is the requirement that most needs a test: the failure
    mode to design against is not missing a wrap, it is reporting one a user
    cannot reproduce.

    ``became=7`` is impossible for ``256 -> int8``. The recomputation says 0.
    The report must print the DISAGREEMENT and must not print the finding's
    claim, its arithmetic or its reproducer as if they held.
    """
    rec = record.Recorder()
    rec.int_narrowings = 1
    rec.add(_finding(became=7))
    text = _rendered(rec)

    assert "WITHHELD" in text
    assert "recomputing the narrowing independently" in text
    assert "bug in the instrument" in text
    assert "the constant written there is 256" not in text
    assert "REPRODUCE" not in text
    assert "1 are WITHHELD as disagreements" in text


def test_the_report_is_byte_identical_across_emission_orders():
    """§10a.7. Findings fire once per TRACE and trace order varies between
    runs; two runs of one suite must not produce two different reports."""
    forward, backward = record.Recorder(), record.Recorder()
    lines = [(54, 256), (12, 300), (54, 1000), (7, 128)]
    for line, written in lines:
        forward.add(_finding(line=line, written=written))
    for line, written in reversed(lines):
        backward.add(_finding(line=line, written=written))
    assert _rendered(forward) == _rendered(backward)


def test_the_denominator_is_printed_even_with_nothing_to_report():
    """§8's first absolute: "0 findings" is indistinguishable from a dead hook."""
    rec = record.Recorder()
    rec.invocations, rec.folded, rec.int_narrowings = 4812, 4000, 3011
    text = _rendered(rec)
    assert "3011 integer const-folds inspected" in text
    assert "4812 rule invocations" in text
    assert "no out-of-range integer narrowings outside jax" in text


def test_a_zero_denominator_is_called_out_rather_than_read_as_clean():
    """A patch that does nothing produces a beautiful zero."""
    text = _rendered(record.Recorder())
    assert "ZERO invocations" in text
    assert "a fact about the instrument, not about your code" in text


def test_no_run_ever_issues_a_clean_bill_of_health():
    """§8's second absolute. Both directions: findings and none."""
    empty = record.Recorder()
    empty.invocations = empty.int_narrowings = 10
    loaded = record.Recorder()
    loaded.invocations = loaded.int_narrowings = 10
    loaded.add(_finding())
    for rec in (empty, loaded):
        text = _rendered(rec)
        assert "never a clean bill of health" in text
        assert "eager execution" in text
        assert "jnp.where" in text and "jnp.clip" in text
        assert "arm order" in text


def test_a_suppressed_fire_is_named_and_counted_not_dropped():
    """§10a.9."""
    rec = record.Recorder()
    rec.invocations = rec.int_narrowings = 5
    rec.suppressed_jax = 1
    rec.add(
        _finding(
            file="/venv/site-packages/jax/_src/random/threefry2x32.py",
            line=73,
            func="_threefry_seed",
            written=4294967295,
            from_dtype="uint32",
            to_dtype="int32",
            became=-1,
            origin=record.ORIGIN_JAX,
        )
    )
    text = _rendered(rec)
    assert "jax-internal filter: ON. 1 narrowing(s) written by jax itself" in text
    assert "Named, not dropped:" in text
    assert "threefry2x32.py:73" in text
    assert "4294967295" in text and "-1" in text

    # ...and the filter says it is on even when it caught nothing, or a clean
    # run and a run with no filter print the same thing
    clean = record.Recorder()
    clean.invocations = clean.int_narrowings = 5
    clean_text = _rendered(clean)
    assert "jax-internal filter: ON. 0 narrowing(s)" in clean_text
    assert "Nothing was suppressed this run" in clean_text


def test_a_line_that_does_not_contain_the_literal_says_so(written_source):
    """The attribution confidence label. A named constant (``x * LIMIT``) is a
    true finding whose line does not show the number, and a user comparing the
    two would otherwise conclude the tool is pointing at the wrong place.

    Driven off the real file rather than the flag, so the flag and the line
    cannot agree with each other while both being wrong: the flag is
    recomputed here the way the wrapper computes it.
    """
    hidden, shown = record.Recorder(), record.Recorder()
    hidden.int_narrowings = shown.int_narrowings = 1
    for rec, line, written in ((hidden, 7, 300), (shown, 2, 256)):
        text = record.source_line(str(written_source), line)
        assert text, "the fixture file did not read back"
        rec.add(
            _finding(
                file=str(written_source),
                line=line,
                written=written,
                became=record.narrow(written, "int8"),
                literal_visible=str(written) in text,
            )
        )
    assert "7 | return x * LIMIT" in _rendered(hidden)
    assert "not textually on that line" in _rendered(hidden)
    assert "2 | return x + 256" in _rendered(shown)
    assert "not textually on that line" not in _rendered(shown)


def test_a_disabled_tripwire_says_what_still_works():
    """§4: "disabled" must never read as "you are unprotected"."""
    text = _rendered(record.Recorder(), _Status(code="no-registry", detail="moved"))
    assert "NOT ARMED [no-registry]" in text
    assert "Static checking is unaffected" in text


def test_a_non_armed_status_does_not_DISCARD_what_was_measured():
    """A non-armed status used to return before the denominator, so anything
    already collected was dropped with no count and no mention.

    That is not hypothetical under xdist: one broken worker of two makes the
    controller's status ``mixed``, and every finding the OTHER worker
    serialised back went in the bin. ``tests/test_tripwire_xdist.py`` drives
    that as a real two-worker session; this is the same property at the level
    where every branch of it is reachable.

    Both directions, because "print it anyway" is satisfiable by a renderer
    that prints a denominator of zero for a ``no-module`` session and calls it
    disclosure.
    """
    empty = _rendered(record.Recorder(), _Status(code="no-module"))
    assert "denominator:" not in empty, (
        "nothing was measured, so there is no partial to disclose and a "
        "denominator of zero is noise"
    )
    assert "PARTIAL" not in empty

    rec = record.Recorder()
    rec.invocations = 9
    rec.int_narrowings = 4
    rec.add(_finding(written=300, became=44))
    carried = _rendered(rec, _Status(code="mixed", detail="worker statuses: armed, no-entry"))
    assert "denominator: 4 integer const-folds" in carried
    assert "the constant written there is 300" in carried
    assert "PARTIAL" in carried and "not a total" in carried
    assert "never a clean bill of health" in carried

    # and the armed rendering is unchanged, byte for byte, banner and all
    armed = _rendered(rec)
    assert "PARTIAL" not in armed
    assert armed.splitlines()[3] == "" and armed.splitlines()[4].startswith("denominator:")


def test_the_primary_channel_carries_all_THREE_thirds_for_every_code():
    """§4: every message says what happened, what it MEANS, and what still
    works — and the middle third is the one the terminal used to drop.

    ``render_status`` printed ``status.detail``, which ``arm()`` leaves empty
    for all of the failure codes, so the shipped line was
    ``NOT ARMED [no-module] --``: a dangling dash where the meaning belongs.
    ``docs/overflow-tripwire.md`` says the summary states "what the code
    means", and only the ``require`` ``UsageError`` and the canary did.

    Driven over EVERY code rather than one, because the defect was that the
    codes with a non-empty detail (``below-floor``, ``unexpected:*``) read
    fine and hid the rest.
    """
    from stelling import _tripwire

    for code in _tripwire.FAILURE_CODES + ("unexpected:ValueError",):
        first = _rendered(record.Recorder(), _Status(code=code)).splitlines()[2]
        assert first.startswith(f"NOT ARMED [{code}] -- "), first
        assert not first.endswith("--"), f"[{code}] renders a dangling dash"
        meaning = first.split(" -- ", 1)[1].strip()
        assert len(meaning) > 30, f"[{code}] has no middle third: {first!r}"
        assert meaning == _tripwire.Status(code=code).meaning

    # the detail, when arming produced one, is kept rather than displaced
    text = _rendered(record.Recorder(), _Status(code="below-floor", detail="jax 0.4.7"))
    assert "older than the version" in text and "detail: jax 0.4.7" in text


def test_a_changed_rule_hash_is_visible_in_the_status_line():
    """§5: record it, never gate on it — but a canary that cannot see the
    change is a canary that reports nothing.

    THREE STATES, because `known_hash` is a lookup keyed on the running
    release and therefore has a third answer: no row at all. This test drove
    two, and the renderer's `== known_hash else "CHANGED upstream"` would
    have reported the third — a release nobody has read — as a change nobody
    measured, in the one line a canary reader believes.
    """
    same = _rendered(record.Recorder(), _Status(rule_hash="abc", known_hash="abc"))
    moved = _rendered(
        record.Recorder(),
        _Status(rule_hash="def", known_hash="abc", jax_version="0.11.0"),
    )
    unread = _rendered(
        record.Recorder(),
        _Status(rule_hash="def", known_hash=None, jax_version="0.99.0"),
    )
    assert "sha1 abc (as tested)" in same
    assert "sha1 def (CHANGED: jax 0.11.0 is recorded as abc)" in moved
    assert "sha1 def (jax 0.99.0 has NEVER BEEN READ" in unread
    # the two loud states must not read as each other
    assert "NEVER BEEN READ" not in moved and "CHANGED" not in unread


def _canary():
    """`.github/scripts/tripwire_canary.py`, imported by path.

    By path because it is a CI script and not a package module — and imported
    at all, rather than re-implemented here, for the same reason the script
    itself calls the shipped ``arm()``: a test that re-states the decision
    measures the re-statement. Its module scope imports only ``argparse``,
    ``os`` and ``sys``, so this stays runnable in the zero-dep lane, which is
    the lane this file promises.
    """
    import importlib.util
    import pathlib as _pathlib

    path = (
        _pathlib.Path(__file__).resolve().parent.parent
        / ".github" / "scripts" / "tripwire_canary.py"
    )
    spec = importlib.util.spec_from_file_location("_tripwire_canary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_canary_reads_the_version_to_hash_map_in_five_states():
    """§5 decides ARMING and not this script's exit code, so the exit code is
    decided here and the argument is in ``_hash_row``'s docstring.

    The state that must NOT be fatal is the one that looks most alarming:
    a release with no row. It is what a jax NIGHTLY is in by construction —
    the nightly workflow runs this script against one — and what the
    `control` leg enters the day jax ships a release, since that leg installs
    ``.[jax]`` and resolves to whatever is newest. A canary that pages on it
    is red every night, and an alarm that is red every night is not read.

    The state that must be fatal is the one that cannot be explained by
    upstream: a RELEASE contradicting its own row. Wheels are immutable, so
    either the row is wrong or the environment is not the jax it claims, and
    both make the rest of the page unverified.

    AND THE FIFTH, which is a correction. ``Status.hash_state`` has four
    values and this function's fallback swallowed anything else as NON-fatal,
    while ``_control_reasons`` treats an unrecognised state as fatal on a
    stated principle. Two decisions in one file answering the same question
    two ways is one of them being wrong. ``unreadable`` is now handled by
    name — it has an argument for exiting 0, namely that nothing was compared
    — and a fifth string has no argument and pages.
    """
    canary = _canary()

    matched = _Status(rule_hash="abc", known_hash="abc", jax_version="0.11.0")
    unread = _Status(rule_hash="abc", known_hash=None, jax_version="0.99.0")
    moved = _Status(rule_hash="abc", known_hash="xyz", jax_version="0.11.0")
    unreadable = _Status(rule_hash=None, known_hash="abc", jax_version="0.11.0")

    note, reason = canary._hash_row(matched)
    assert (note, reason) == ("as tested", None)

    note, reason = canary._hash_row(unread)
    assert reason is None, "a release with no row must not page the nightly"
    assert "NEVER BEEN READ" in note and "0.99.0" in note
    assert "nightly" in note, "the note must say why this is not a failure"

    note, reason = canary._hash_row(moved)
    assert reason is not None, "a release contradicting its own row is fatal"
    assert reason[0] == "hash:contradicted"
    assert "CONTRADICTS" in note and "xyz" in note and "0.11.0" in note

    note, reason = canary._hash_row(unreadable)
    assert reason is None, "an unreadable rule source claims nothing either way"

    # and the two loud states are not each other's words
    assert "CONTRADICTS" not in canary._hash_row(unread)[0]
    assert "NEVER BEEN READ" not in canary._hash_row(moved)[0]

    # THE FALLBACK PAGES. Driven through `main()` as well, in the table
    # below, because a fallback tested only here is a fallback whose caller
    # can quietly stop consulting it.
    class _FifthState(type(matched)):
        @property
        def hash_state(self):
            return "wedged"

    note, reason = canary._hash_row(_FifthState(code="armed"))
    assert reason is not None and reason[0] == "hash:unknown-state", (
        "an unrecognised hash state was swallowed as non-fatal, which is the "
        "answer the other decision in this file rejects"
    )
    assert "wedged" in reason[1], "the page must name the state it cannot read"


def test_every_failure_code_is_explained_and_documented():
    """The codes are advertised as stable and greppable, in three places: the
    tuple, the explanation table, and ``docs/overflow-tripwire.md``. A code
    added to one and not the others is a code a user greps for and a user
    reads about, in different sets.

    Also the property that makes "disabled never reads as unprotected" true of
    EVERY code rather than of the ones someone remembered.
    """
    import pathlib
    import re

    from stelling import _tripwire

    missing = [c for c in _tripwire.FAILURE_CODES if c not in _tripwire._EXPLAIN]
    assert not missing, f"failure codes with no explanation: {missing}"
    extra = [c for c in _tripwire._EXPLAIN if c not in _tripwire.FAILURE_CODES]
    assert not extra, f"explanations for codes nothing can return: {extra}"

    for code in _tripwire.FAILURE_CODES:
        text = _tripwire.Status(code=code).explanation
        assert "Static checking is unaffected" in text, (
            f"[{code}] does not say what still works, so it reads as "
            "'you are unprotected'"
        )

    page = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "overflow-tripwire.md").read_text(
        encoding="utf-8"
    )
    undocumented = [c for c in _tripwire.FAILURE_CODES if f"`{c}`" not in page]
    assert not undocumented, (
        f"docs/overflow-tripwire.md does not name {undocumented}, and it "
        "presents the list as complete"
    )
    # ...and the open-ended one, which is the only code not in the tuple
    assert "`unexpected:<ExcType>`" in page

    # AND THE OTHER DIRECTION, which is the one that made this check
    # one-sided: the three assertions above are all "tuple => elsewhere", so
    # dropping a code from BOTH the tuple and the table left this green while
    # the doc went on advertising it. The doc's own sentence is the third
    # register, so it is read back and compared, not merely searched.
    marker = "The codes are stable and greppable:"
    assert marker in page, "the doc no longer advertises the list at all"
    sentence = page.split(marker, 1)[1].split("\n\n", 1)[0]
    advertised = set(re.findall(r"`([a-z:<>A-Za-z-]+)`", sentence)) - {
        "unexpected:<ExcType>"
    }
    unreturnable = sorted(advertised - set(_tripwire.FAILURE_CODES))
    assert not unreturnable, (
        f"docs/overflow-tripwire.md advertises {unreturnable} as greppable "
        "and nothing can return them"
    )
    assert advertised == set(_tripwire.FAILURE_CODES), (
        "the doc's list and FAILURE_CODES are supposed to BE the same list"
    )

    # an unknown code still explains itself rather than rendering `None`
    fallback = _tripwire.Status(code="unexpected:ValueError").explanation
    assert "does not have a name for" in fallback
    assert "Static checking is unaffected" in fallback


# --- the canary's exit code, driven ------------------------------------------
#
# `.github/scripts/tripwire_canary.py` is the alarm. Everything below drives
# its `main()` and reads what a CI job reads: the exit status, the reason
# codes on stderr, and the rows on stdout. Nothing here re-implements the
# decision, and nothing here asserts the SHAPE of the code -- an earlier
# version of this battery walked the AST for the name of a local variable and
# went red on a rename while a deleted alarm went green.
#
# THIS FILE'S COPY RUNS WITHOUT JAX, which is the lane `tests/
# test_tripwire_arm.py` is `importorskip`'d out of and the lane that runs on
# every PR. The probe is a stub here: `main()` reaches jax only through
# `stelling._jax_compat`, and in a jax-less environment importing that module
# raises, so every armed state would collapse to `raised` and the table would
# measure one row six times. `tests/test_tripwire_arm.py` drives the same
# states through the REAL jax, the real `arm()` and a real subprocess; this
# file drives the decision. Neither is the other's substitute.


class _MovedNarrowings(record.Recorder):
    """A recorder whose ``int_narrowings`` has moved out from under the
    canary -- the shape change the ``unrenderable`` state exists for."""

    @property
    def int_narrowings(self):
        raise AttributeError("`int_narrowings` moved off Recorder")

    @int_narrowings.setter
    def int_narrowings(self, value):
        pass


class _MovedFindingField:
    """A finding whose fields have moved. Attribute access is the failure."""

    def __getattr__(self, name):
        raise AttributeError(f"finding field {name!r} moved")


def _a_finding(**kw):
    fields = dict(
        file="probe.py", line=36, func="over", written=256,
        from_dtype="int32", to_dtype="int8", became=0, origin="literal",
    )
    fields.update(kw)
    return record.Finding(**fields)


def _stub_jax(monkeypatch, probe):
    """Give `main()` a jax boundary in a lane that has no jax.

    ``make_jaxpr`` really calls the shipped ``_probe.over``, so the probe is
    the program it is everywhere else; only the tracer is a stand-in. The same
    stand-in serves the perimeter's control, whose probes take an array and
    compare it against an int -- ``float32`` is here for those.
    """
    import sys
    import types

    import stelling
    from stelling._tripwire import _probe

    class _Array:
        def __add__(self, other):
            return self

        __radd__ = __add__

    fake = types.ModuleType("stelling._jax_compat")
    fake.jax = types.SimpleNamespace(make_jaxpr=lambda fn: lambda *a, **k: fn(*a, **k))
    fake.jnp = types.SimpleNamespace(
        zeros=lambda shape, dtype: _Array(), int8="int8", int16="int16",
        float32="float32", full=lambda shape, value, dtype: _Array(),
    )
    monkeypatch.setitem(sys.modules, "stelling._jax_compat", fake)
    monkeypatch.setattr(stelling, "_jax_compat", fake, raising=False)

    # AND THE PREDICATE'S OWN MEMOS, PUT BACK ON THE WAY OUT. `prop_guard`
    # caches the modules it lazily imports and memoises promotion targets, and
    # a lookup that happens inside this window caches THE FAKE -- permanently,
    # because those are module globals and `monkeypatch` knows nothing about
    # them. Measured: one `classify()` reaching `_x64()` here left
    # `prop_guard._JAX` bound to a `SimpleNamespace` with no `.config`, so
    # every later call in the process declined with an internal AttributeError
    # and the perimeter was silently dead for the rest of the session. It did
    # not show up in file order -- `test_narrowing_perimeter.py` sorts before
    # this file -- which is exactly the shape a shuffled lane exists to catch.
    # Re-setting each to its current value records it, and teardown restores
    # it whatever this window did.
    if probe == "raises":
        def _boom(a):
            raise RuntimeError("FORCED: the probe could not execute")

        monkeypatch.setattr(_probe, "over", _boom)

    try:
        from stelling._tripwire import prop_guard
    except Exception:  # noqa: BLE001 - the zero-dep lane has no numpy
        return
    for name in ("_JNP", "_ML", "_JAX"):
        monkeypatch.setattr(prop_guard, name, getattr(prop_guard, name))
    monkeypatch.setattr(prop_guard, "_TARGET_CACHE", dict(prop_guard._TARGET_CACHE))


#: The eager detector's states this battery can put the canary in, and what
#: each one does to the two things `main()` reads: the status `arm_eager()`
#: hands back, and what the live control does.
#:
#: STUBBED AT THE INSTRUMENT AND NOT AT THE DECISION, exactly as the tripwire's
#: half is: `_eager_reasons` and `_eager_hash_row` are the shipped functions
#: and are never replaced. What is chosen is what they are given.
#:
#: THE STUB IS NEEDED AT ALL because this file's copy runs WITHOUT jax -- the
#: lane that runs on every PR -- and `arm_eager()` there would collapse every
#: row to one state, which is the shape of table that measures one thing six
#: times. `tests/test_tripwire_eager.py` drives the real detector against the
#: real jax; this drives the decision.
_EAGER_STATES = {
    "clean": ("armed", "raise-truncation", "allow", ("abc", "abc"), ()),
    "not-armed": ("no-site", "raise-truncation", "allow", ("abc", "abc"), ()),
    "did-not-fire": ("armed", "allow", "allow", ("abc", "abc"), ()),
    "cries-wolf": ("armed", "raise-truncation", "raise-truncation",
                   ("abc", "abc"), ()),
    "raised": ("armed", "raise-other", "allow", ("abc", "abc"), ()),
    "displaced": ("armed", "raise-truncation", "allow", ("abc", "abc"),
                  ("eager",)),
    "hash-changed": ("armed", "raise-truncation", "allow", ("abc", "xyz"), ()),
}


def _stub_eager(monkeypatch, choice):
    """Force the eager half of the canary into one of :data:`_EAGER_STATES`."""
    from stelling import EagerTruncationError, _tripwire
    from stelling._tripwire import Status, _probe

    code, over, under, (rule_hash, known_hash), displaced = _EAGER_STATES[choice]
    status = Status(
        code=code,
        jax_version="0.11.0",
        rule_name="_convert_element_type",
        rule_hash=rule_hash,
        known_hash=known_hash,
    )

    def _behave(kind):
        def probe(jnp):
            if kind == "raise-truncation":
                raise EagerTruncationError(
                    "stelling: 256 was TRUNCATED to 0",
                    written=256, to_dtype="int8", became=0,
                    file="probe.py", line=1, func="construct",
                )
            if kind == "raise-other":
                raise RuntimeError("FORCED: the eager control could not run")
            return None

        return probe

    monkeypatch.setattr(_tripwire, "arm_eager", lambda: status)
    monkeypatch.setattr(_tripwire, "disarm_eager", lambda: "restored")
    monkeypatch.setattr(_tripwire, "displaced", lambda: displaced)
    monkeypatch.setattr(_probe, "construct_over", _behave(over))
    monkeypatch.setattr(_probe, "construct_under", _behave(under))
    return status


#: The DUNDER PERIMETER's states this battery can put the canary in. Same
#: shape and same argument as :data:`_EAGER_STATES`, one instrument over:
#: what is chosen is the status `arm_perimeter()` hands back, what the live
#: control does in each direction, which structural facts moved, and whether
#: the promotion identity drifted. `_perimeter_reasons` -- the decision -- is
#: the shipped function and is never replaced.
#:
#: THE FACTS AND THE PROMOTION IDENTITY ARE STUBBED AT THE MEASUREMENT, and
#: that is a real limit of this table rather than a hidden one: those two rows
#: ask jax questions, and this file's copy runs in the lane that has no jax.
#: `tests/test_narrowing_perimeter.py` drives the real `_perimeter_facts` and
#: `_perimeter_promotion` against a real jax, with faults injected into each;
#: this drives the decision they feed.
_PERIMETER_STATES = {
    "clean": ("armed", "refuse", "allow", (), ()),
    "not-armed": ("no-type", "refuse", "allow", (), ()),
    "did-not-fire": ("armed", "allow", "allow", (), ()),
    "cries-wolf": ("armed", "refuse", "refuse", (), ()),
    "raised": ("armed", "raise-other", "allow", (), ()),
    "facts-moved": ("armed", "refuse", "allow", ("Py_TPFLAGS_HEAPTYPE",), ()),
    "promotion-drift": ("armed", "refuse", "allow", (),
                        ("int16/add: says int16, jax uses int32",)),
}


def _stub_perimeter(monkeypatch, canary, choice):
    """Force the perimeter half of the canary into one of :data:`_PERIMETER_STATES`."""
    import types

    from stelling import _tripwire
    from stelling._tripwire import Status, _probe
    from stelling._tripwire.perimeter import NarrowingError

    code, over, under, moved, drift = _PERIMETER_STATES[choice]
    status = Status(code=code, jax_version="0.11.0")

    def _behave(kind):
        def probe(x):
            if kind == "refuse":
                # A STAND-IN FINDING and not `prop_guard.Finding`: that
                # module imports numpy, and this file's copy runs in the lane
                # that has none. The canary reads three fields off it, which
                # is what this carries.
                raise NarrowingError(
                    types.SimpleNamespace(
                        reason="inexact", slot="le", literal=2**31 - 1,
                        narrowed_to=2147483648.0, target_dtype="float32",
                    ),
                    file="probe.py", line=1, func="compare",
                    message="stelling: 2147483647 is not exactly representable",
                )
            if kind == "raise-other":
                raise RuntimeError("FORCED: the perimeter control could not run")
            return None

        return probe

    monkeypatch.setattr(_tripwire, "arm_perimeter", lambda **kw: status)
    monkeypatch.setattr(_tripwire, "disarm_perimeter", lambda owner=None: "restored")
    monkeypatch.setattr(_probe, "compare_over", _behave(over))
    monkeypatch.setattr(_probe, "compare_under", _behave(under))
    # BOTH FACES' PROBES, because the canary drives both and a stub that
    # patched only the traced pair would leave the eager one running against
    # the stand-in tracer -- which is not a jax array and does not add.
    monkeypatch.setattr(_probe, "arith_over", _behave(over))
    monkeypatch.setattr(_probe, "arith_under", _behave(under))
    monkeypatch.setattr(
        canary, "_perimeter_facts",
        lambda face, located: ([(f"{face}: type", "PASS stub")], list(moved)),
    )
    monkeypatch.setattr(
        canary, "_perimeter_promotion",
        lambda sample=None: (f"stubbed, {len(drift)} disagree", list(drift)),
    )
    return status


class _Run:
    """One driven run of the canary, as its consumers see it."""

    def __init__(self, code, out, err, summary):
        import re

        self.code = code
        self.out = out
        self.err = err
        self.summary = summary
        self.rows = {}
        for line in out.splitlines():
            name, sep, value = line.partition(": ")
            if sep:
                self.rows.setdefault(name, value)
        self.reasons = re.findall(r"^canary \[([a-z:-]+)\]:", err, re.M)
        self.sentences = dict(
            re.findall(r"^canary \[([a-z:-]+)\]: (.*)$", err, re.M)
        )

    def __repr__(self):
        return (
            f"<exit {self.code} reasons={self.reasons} "
            f"control={self.rows.get('control state')!r}/"
            f"{self.rows.get('control report')!r} "
            f"perimeter={self.rows.get('perimeter control state')!r}>"
        )


def _drive_canary(
    capsys, tmp_path, *,
    require=False, armed=True, hash_state="as-tested",
    probe="runs", findings="one", narrowings="readable",
    summary="writable", eager="clean", perimeter="clean",
):
    """Run `main()` once with every input chosen, and collect what it emitted.

    THE INPUTS ARE FORCED, NOT THE DECISION. ``arm()`` is replaced by one
    that hands back a real :class:`Status` and a real :class:`Recorder` --
    the shipped ``Status.hash_state`` still computes the hash verdict from
    the two hashes given to it -- and the recorder's contents are chosen. No
    branch of the canary, and no verdict function, is stubbed or bypassed.

    ITS OWN MONKEYPATCH CONTEXT, undone before it returns. The table below
    is one test driving eighteen rows, and the ``monkeypatch`` FIXTURE is
    torn down once at the end of the test -- so the row that makes the probe
    raise left it raising for every row after it, and four rows that named
    the recorder were measuring a broken probe. That is the same class of
    defect as the one this whole battery exists to close, met while building
    it, and a per-row context is the fix.
    """
    with pytest.MonkeyPatch.context() as mp:
        return _drive_canary_once(
            mp, capsys, tmp_path, require=require, armed=armed,
            hash_state=hash_state, probe=probe, findings=findings,
            narrowings=narrowings, summary=summary, eager=eager,
            perimeter=perimeter,
        )


def _drive_canary_once(
    monkeypatch, capsys, tmp_path, *,
    require, armed, hash_state, probe, findings, narrowings, summary,
    eager="clean", perimeter="clean",
):
    import sys

    from stelling import _tripwire
    from stelling._tripwire import Status

    canary = _canary()
    _stub_jax(monkeypatch, probe)
    _stub_eager(monkeypatch, eager)
    _stub_perimeter(monkeypatch, canary, perimeter)

    hashes = {
        "as-tested": ("abc", "abc"),
        "changed": ("abc", "xyz"),
        "never-read": ("abc", None),
        "unreadable": (None, "abc"),
    }
    rule_hash, known_hash = hashes[hash_state]
    base = Status(
        code="armed" if armed else "no-entry",
        jax_version="0.11.0",
        rule_name="_convert_elt_type_folding_rule",
        rule_hash=rule_hash,
        known_hash=known_hash,
        registry_size=3,
    )
    assert base.hash_state == hash_state, (
        f"the shipped Status computed {base.hash_state!r} from the hashes "
        f"this row chose, not {hash_state!r}: the table is describing "
        "something other than what it drives"
    )

    rec = _MovedNarrowings() if narrowings == "moved" else record.Recorder()
    if findings == "one":
        monkeypatch.setattr(rec, "sorted_findings", lambda: [_a_finding()])
    elif findings == "field-moved":
        monkeypatch.setattr(rec, "sorted_findings", lambda: [_MovedFindingField()])
    elif findings == "unreadable":
        def _boom():
            raise AttributeError("`sorted_findings` moved off Recorder")

        monkeypatch.setattr(rec, "sorted_findings", _boom)
    else:
        assert findings == "none"

    monkeypatch.setattr(_tripwire, "arm", lambda *a, **k: (base, rec))
    monkeypatch.setattr(
        sys, "argv", ["tripwire_canary.py"] + (["--require"] if require else [])
    )

    _drive_canary_once.calls = getattr(_drive_canary_once, "calls", 0) + 1
    stem = f"step-{_drive_canary_once.calls}"
    path = (
        tmp_path / "no-such-directory" / stem
        if summary == "unwritable"
        else tmp_path / stem
    )
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))

    capsys.readouterr()
    code = canary.main()
    out = capsys.readouterr()
    written = path.read_text(encoding="utf-8") if path.exists() else ""
    return _Run(code, out.out, out.err, written)


#: (row id, kwargs, expected exit, expected reason codes, control state,
#: control report). ONE TABLE, because the canary's contract IS a table and
#: six assertions scattered over three files is how it came to have two
#: unnamed exits and a battery that measured an unrelated branch.
#:
#: Every row names its reasons EXACTLY. That is what makes the battery
#: independent of every other fatal source: an assertion that `main()`
#: returned 1 is satisfied by any reason at all -- which is how the previous
#: version of this battery came to be satisfied, in six cells, by a
#: contradicted hash row that a polluted process had manufactured, while the
#: branch it named was never reached. A set equality cannot be satisfied by
#: the wrong branch, and it cannot be masked by an extra one either: a run
#: with a second live fault fails here, loudly, instead of passing quietly.
_CANARY_TABLE = [
    ("clean", {}, 0, [], "fired", "rendered", "fired"),
    ("clean+require", {"require": True}, 0, [], "fired", "rendered", "fired"),
    # a release nobody has read is LOUD and not fatal -- the state a nightly
    # is in by construction, and the state the control leg enters the day jax
    # ships a release
    ("never-read", {"hash_state": "never-read", "require": True}, 0, [],
     "fired", "rendered", "fired"),
    ("hash-unreadable", {"hash_state": "unreadable", "require": True}, 0, [],
     "fired", "rendered", "fired"),
    ("hash-contradicted", {"hash_state": "changed"}, 1, ["hash:contradicted"],
     "fired", "rendered", "fired"),
    ("hash-contradicted+require", {"hash_state": "changed", "require": True}, 1,
     ["hash:contradicted"], "fired", "rendered", "fired"),
    # THE DEAD HOOK. `--require` must not matter: `arm()` said the hook is
    # attached and the control says nothing reached it, so the armed status
    # is unverified either way. Gating this on `--require` is the exact
    # defect this branch exists to close.
    ("dead-hook", {"findings": "none"}, 1, ["control:did-not-fire"],
     "did-not-fire", "rendered", "fired"),
    ("dead-hook+require", {"findings": "none", "require": True}, 1,
     ["control:did-not-fire"], "did-not-fire", "rendered", "fired"),
    # THE PROBE NEVER RAN. Used to exit 0, because the check was a substring
    # test for "DID NOT FIRE" against a line a raised control does not
    # contain.
    ("probe-raised", {"probe": "raises"}, 1, ["control:raised"],
     "raised", "not-run", "fired"),
    ("probe-raised+require", {"probe": "raises", "require": True}, 1,
     ["control:raised"], "raised", "not-run", "fired"),
    # THE RECORDER MOVED, three ways, and none of them may report `raised`:
    # the probe RAN in all three, and an operator sent upstream to look for a
    # broken jax when the defect is in this repository has been sent to the
    # wrong place.
    ("findings-unreadable", {"findings": "unreadable"}, 1,
     ["control:indeterminate"], "ran", "not-run", "fired"),
    ("narrowings-moved", {"narrowings": "moved"}, 1, ["control:unrenderable"],
     "fired", "unrenderable", "fired"),
    ("finding-field-moved", {"findings": "field-moved"}, 1,
     ["control:unrenderable"], "fired", "unrenderable", "fired"),
    # BOTH AT ONCE, and both must be said. This is the case the previous
    # split lost: `unrenderable` overwrote `did-not-fire`, so the page told
    # the operator the probe completing "is not in doubt" and never mentioned
    # that the hook was dead.
    ("dead-hook+narrowings-moved", {"findings": "none", "narrowings": "moved"},
     1, ["control:did-not-fire", "control:unrenderable"],
     "did-not-fire", "unrenderable", "fired"),
    # NOT ARMED is `--require`'s question and nobody else's, and the control
    # never ran, so it is not a finding.
    ("not-armed", {"armed": False}, 0, [], "not-run", "not-run", "fired"),
    ("not-armed+require", {"armed": False, "require": True}, 1, ["not-armed"],
     "not-run", "not-run", "fired"),
    ("not-armed+require+hash", {"armed": False, "require": True,
                                "hash_state": "changed"}, 1,
     ["not-armed", "hash:contradicted"], "not-run", "not-run", "fired"),
    # INFRASTRUCTURE MUST NOT PAGE -- Property 3 of the workflow. An
    # unwritable `$GITHUB_STEP_SUMMARY` used to raise `FileNotFoundError`
    # straight out of `main()`: a traceback and a 1, with no `canary:`
    # sentence, in the script that argues against exactly that.
    ("summary-unwritable", {"summary": "unwritable"}, 0, [], "fired", "rendered", "fired"),
    # --- the EAGER half. Same three questions, one instrument over: did it
    # attach, did its live control behave in BOTH directions, and is its own
    # hash row contradicted. The states are stubbed at the instrument
    # (`_EAGER_STATES`) and every decision below them is the shipped one.
    #
    # ARMING IS `--require`'s QUESTION AND THE CONTROL'S IS NOT, exactly as
    # for the tripwire: a human running this by hand against a jax that moved
    # the site wants to see it, not be paged by it. A detector that attached
    # and then failed its own control is broken in either mode.
    ("eager-not-armed", {"eager": "not-armed"}, 0, [], "fired", "rendered",
     "not-run"),
    ("eager-not-armed+require", {"eager": "not-armed", "require": True}, 1,
     ["eager:not-armed"], "fired", "rendered", "not-run"),
    # THE DEAD DETECTOR: it attached and allowed a construction it must
    # refuse. `--require` must not matter.
    ("eager-dead", {"eager": "did-not-fire"}, 1, ["eager:did-not-fire"],
     "fired", "rendered", "did-not-fire"),
    # ...and the other direction, which is why the control has two: a hook
    # replaced by "refuse everything" passes the positive probe.
    ("eager-cries-wolf", {"eager": "cries-wolf"}, 1, ["eager:cries-wolf"],
     "fired", "rendered", "cries-wolf"),
    ("eager-probe-raised", {"eager": "raised"}, 1, ["eager:raised"],
     "fired", "rendered", "raised"),
    # THE DISPLACEMENT, which is the one instrument B15's finding and this
    # hook share: something is bound over stelling's wrapper.
    ("hooks-displaced", {"eager": "displaced"}, 1, ["hooks:displaced"],
     "fired", "rendered", "fired"),
    ("eager-hash-contradicted", {"eager": "hash-changed"}, 1,
     ["eager:hash-contradicted"], "fired", "rendered", "fired"),
    # BOTH HALVES BROKEN AT ONCE, and both must be said -- the same lesson
    # the `dead-hook+narrowings-moved` row above records, across the two
    # instruments this time.
    ("both-dead+require", {"findings": "none", "eager": "did-not-fire",
                           "require": True}, 1,
     ["control:did-not-fire", "eager:did-not-fire"], "did-not-fire",
     "rendered", "did-not-fire"),
]


#: The perimeter's own rows. A SECOND TABLE and not eight more columns on the
#: first, because the first is 25 rows about two instruments and adding a
#: third column to every one of them would have meant re-typing 25 expected
#: values that no perimeter state moves. Same contract, same three questions:
#: did it attach, did its live control behave in BOTH directions, and do the
#: facts it rests on still hold.
#:
#: (name, kwargs, exit, reason codes, perimeter control state)
_PERIMETER_TABLE = [
    ("clean", {}, 0, [], "fired"),
    ("clean+require", {"require": True}, 0, [], "fired"),
    # ARMING IS `--require`'s QUESTION AND THE CONTROL'S IS NOT, exactly as
    # for the other two: a human running this by hand against a jax that moved
    # the type wants to see it, not be paged by it.
    ("not-armed", {"perimeter": "not-armed"}, 0, [], "not-run"),
    ("not-armed+require", {"perimeter": "not-armed", "require": True}, 1,
     ["perimeter:not-armed"], "not-run"),
    # THE DEAD PERIMETER: it attached and allowed a literal it must refuse.
    # `--require` must not matter -- this is the campaign's signature defect.
    ("dead", {"perimeter": "did-not-fire"}, 1, ["perimeter:did-not-fire"],
     "did-not-fire"),
    # ...and the other direction, which is why the control has two: a
    # perimeter replaced by "refuse every int" passes the positive probe.
    ("cries-wolf", {"perimeter": "cries-wolf"}, 1, ["perimeter:cries-wolf"],
     "cries-wolf"),
    ("probe-raised", {"perimeter": "raised"}, 1, ["perimeter:raised"],
     "raised"),
    # THE FACTS THE PERIMETER RESTS ON. These are the rows that redden when
    # jax takes something away -- the heap-type flag, the own slots, a warm op
    # entering Python -- and they page whether or not `--require` was passed,
    # because a perimeter resting on a fact that stopped being true is not a
    # perimeter with one feature missing.
    ("facts-moved", {"perimeter": "facts-moved"}, 1, ["perimeter:facts-moved"],
     "fired"),
    ("promotion-drift", {"perimeter": "promotion-drift"}, 1,
     ["perimeter:promotion-drift"], "fired"),
    # ALL THREE INSTRUMENTS BROKEN AT ONCE, and all three must be said.
    ("everything-dead+require", {"findings": "none", "eager": "did-not-fire",
                                 "perimeter": "did-not-fire", "require": True},
     1, ["control:did-not-fire", "eager:did-not-fire",
         "perimeter:did-not-fire"], "did-not-fire"),
]


def test_the_canary_pages_for_the_perimeters_reasons_and_no_others(
    capsys, tmp_path
):
    """The perimeter's half of the contract, driven the same way as the rest.

    The set of reason codes, exactly, for the reason the first table asserts
    the set: ``main() == 1`` is satisfied by any of a dozen branches, and a
    cell that names one while another produces it is a cell measuring nothing.
    """
    failures = []
    for name, kw, expect_code, expect_reasons, control in _PERIMETER_TABLE:
        run = _drive_canary(capsys, tmp_path, **kw)
        got = (run.code, sorted(run.reasons),
               run.rows.get("perimeter control state"))
        want = (expect_code, sorted(expect_reasons), control)
        if got != want:
            failures.append(f"  {name}: got {got}, contract {want}\n    {run.err}")
    assert not failures, (
        "the canary's perimeter reasons are not what the contract says:\n"
        + "\n".join(failures)
    )

    # EVERY DECLARED STATE IS REACHABLE, the same claim the other two
    # instruments' state sets carry: a state nothing can drive is a claim.
    canary = _canary()
    reached = {row[4] for row in _PERIMETER_TABLE}
    assert reached == set(canary.PERIMETER_CONTROL_STATES), (
        f"declared perimeter control states "
        f"{sorted(canary.PERIMETER_CONTROL_STATES)} but the table can only "
        f"reach {sorted(reached)}"
    )


def test_the_canary_pages_for_the_reasons_it_measured_and_no_others(
    capsys, tmp_path
):
    """The whole contract, driven: exit status AND which reason produced it.

    WHY THE REASON CODE AND NOT JUST THE NUMBER. Two audits have now found
    this battery vacuous, both times because ``main() == 1`` is satisfied by
    any of six branches. The second time, an unrelated test had left
    stelling's own wrapper as jax's live const-fold rule for the rest of the
    process, so every run reported a contradicted hash row; six cells that
    named the live control were all satisfied by the hash branch and the
    branch they named was never entered. Gating the control on ``--require``
    -- the original defect's shape -- kept the whole file green.

    So the assertion is the SET of reason codes, exactly. It cannot be
    satisfied by the wrong branch and it cannot be masked by an extra one.

    WHY NOT THE SENTENCES. Because a synonym is not a defect. Rewording any
    message here changes nothing a CI job reads; deleting a reason changes
    everything. The sentences are checked once, further down, for the two
    properties that are not wording: that the four fatal control findings do
    not read as each other, and that the leg guidance agrees with the
    workflow it describes.
    """
    failures = []
    seen = set()
    for name, kw, expect_code, expect_reasons, control, report, eager in (
        _CANARY_TABLE
    ):
        run = _drive_canary(capsys, tmp_path, **kw)
        seen.update(run.reasons)
        got = (run.code, sorted(run.reasons), run.rows.get("control state"),
               run.rows.get("control report"),
               run.rows.get("eager control state"))
        want = (expect_code, sorted(expect_reasons), control, report, eager)
        if got != want:
            failures.append(f"  {name}: got {got}, contract {want}\n    {run.err}")
    assert not failures, (
        "the canary's exit code, its reasons, or the state it says it was in "
        "are not what the contract says:\n" + "\n".join(failures)
    )

    # EVERY DECLARED STATE IS REACHABLE. A state named in `CONTROL_STATES`
    # that no input can produce is a claim, not a state, and the exhaustive
    # `_control_reasons` above it would be exhaustive over a set nobody
    # drives.
    canary = _canary()
    reached = {row[4] for row in _CANARY_TABLE}
    assert reached == set(canary.CONTROL_STATES), (
        f"declared control states {sorted(canary.CONTROL_STATES)} but the "
        f"table can only reach {sorted(reached)}"
    )
    reported = {row[5] for row in _CANARY_TABLE}
    assert reported == set(canary.RENDER_STATES), (
        f"declared report states {sorted(canary.RENDER_STATES)} but the "
        f"table can only reach {sorted(reported)}"
    )
    # The same claim for the eager control's states, and it is not a copy of
    # the one above: the two instruments' state sets are DIFFERENT sets --
    # this one has no `indeterminate` (there is no recorder to read; the
    # answer is whether one line raised) and gains `cries-wolf`.
    eager_reached = {row[6] for row in _CANARY_TABLE}
    assert eager_reached == set(canary.EAGER_CONTROL_STATES), (
        f"declared eager control states "
        f"{sorted(canary.EAGER_CONTROL_STATES)} but the table can only "
        f"reach {sorted(eager_reached)}"
    )


def test_the_canary_page_carries_every_reason_the_exit_code_was_built_from(
    capsys, tmp_path
):
    """The summary page is the other consumer, and it gets the same verdict.

    The workflow's header says a failure must be diagnosable from
    ``$GITHUB_STEP_SUMMARY`` without re-running anything. A table of facts
    with the verdict left on stderr is not that: the stderr of a failed step
    is in the log, and the log is the thing the summary page exists to save
    somebody from reading.
    """
    run = _drive_canary(
        capsys, tmp_path,
        findings="none", narrowings="moved", hash_state="changed", require=True,
    )
    assert run.code == 1
    for code in run.reasons:
        assert f"`{code}`" in run.summary, (
            f"the page exited 1 for {code!r} and its summary does not say so"
        )
    assert "**exit 1**" in run.summary

    clean = _drive_canary(capsys, tmp_path)
    assert clean.code == 0
    assert "**exit 0**" in clean.summary
    assert "**exit 1**" not in clean.summary


def test_the_canarys_live_control_row_reports_the_recorder_and_not_a_constant(
    capsys,
):
    """The human line is a function of what the recorder holds.

    Not a wording test -- a differential one. Two runs whose ONLY difference
    is the recorder's contents must produce two different PAGES, and each
    must carry that run's own figures. It reads the whole page rather than a
    row by name: the row that carries these figures is the human one, which
    the script's output contract says nothing re-parses, so a test that keyed
    on its label would forbid renaming a label that means nothing. A line
    hardcoded to ``0 narrowing(s)``,
    or to one canned finding, reads exactly like a live one on a healthy run
    and exactly like a live one on a dead hook; it is the shape of instrument
    this repository keeps having to withdraw.
    """
    import re
    import sys

    seen = []
    for narrowings, written, became, dtype in (
        (1, 256, 0, "int8"),
        (4, 70000, 4464, "int16"),
    ):
        canary = _canary()
        stack = pytest.MonkeyPatch.context()
        monkeypatch = stack.__enter__()
        _stub_jax(monkeypatch, "runs")
        # THE EAGER HALF IS STUBBED TOO, and it is not decoration: without
        # this, `canary.main()` calls the REAL `arm_eager()` and then the REAL
        # `disarm_eager()`, which restores jax's own attribute and clears the
        # installation record -- so a session running with
        # `--stelling-eager-truncation=error` had its detector taken out HERE
        # and every later file ran unwatched. Measured before the escalation
        # in `plugin.py` was made reachable: the session-wide armed run
        # printed `NOT ARMED [detached]` and exited 0. Same defect as the
        # `finally: disarm()` in `tests/test_tripwire_eager.py`, one file
        # over: a test that takes an instrument out must put it back, and the
        # cheapest way to put it back is never to touch it.
        _stub_eager(monkeypatch, "clean")
        # AND THE PERIMETER, for exactly the reason above, one instrument
        # over: the real `disarm_perimeter()` would release a hold this
        # process did not take and could restore slots an outer session is
        # relying on.
        _stub_perimeter(monkeypatch, canary, "clean")

        from stelling import _tripwire
        from stelling._tripwire import Status

        rec = record.Recorder()
        rec.int_narrowings = narrowings
        finding = _a_finding(written=written, became=became, to_dtype=dtype)
        monkeypatch.setattr(rec, "sorted_findings", lambda f=finding: [f])
        status = Status(code="armed", jax_version="0.11.0",
                        rule_hash="abc", known_hash="abc")
        monkeypatch.setattr(_tripwire, "arm", lambda *a, **k: (status, rec))
        monkeypatch.setattr(sys, "argv", ["tripwire_canary.py"])
        capsys.readouterr()
        assert canary.main() == 0
        page = capsys.readouterr().out
        seen.append(page)
        assert re.search(rf"\b{narrowings} narrowing\(s\)", page), (
            f"the recorder held {narrowings} narrowings and the page says "
            f"{page!r}"
        )
        assert f"{written} -> {became} ({dtype})" in page, page
        stack.__exit__(None, None, None)

    assert seen[0] != seen[1], (
        "two runs with different recorders produced the same page, so the "
        "page is not reporting the recorder"
    )


def test_an_unrecognised_control_state_pages_rather_than_passing():
    """The one reason `main()` cannot be driven into, and why it exists.

    ``control_state`` is assigned from literals inside `main()`, so a sixth
    one cannot be produced without editing the script -- which is exactly the
    event this reason is for. The decision is therefore driven directly, and
    the exit-code list at the top of the script says the state is unreachable
    rather than leaving a reader to wonder why they have never seen it.
    """
    canary = _canary()

    assert canary._control_reasons("fired", "rendered", "x") == []
    assert canary._control_reasons("not-run", "not-run", "x") == []

    wedged = canary._control_reasons("wedged", "not-run", "x")
    assert [c for c, _ in wedged] == ["control:unknown-state"]
    assert "wedged" in wedged[0][1], "the page must name the state it cannot read"

    wedged_report = canary._control_reasons("fired", "sideways", "x")
    assert [c for c, _ in wedged_report] == ["control:unknown-state"]


def test_an_UNENUMERATED_eager_truncation_of_jax_s_own_pages(monkeypatch):
    """The canary's half of the map that decides a SUPPRESSION.

    ``_adapter_jax._JAX_EAGER_CONSTANTS`` records the eager truncations jax
    performs itself; a narrowing matching no row is attributed to whoever
    called jax and RAISES there. So a jax release that adds one turns into a
    false alarm inside jax in every armed user's code, and this job's control
    leg meets a new release before any CI lane does.

    Driven directly rather than through ``main()``, for the reason
    ``control:unknown-state`` is: what the sweep finds depends on the jax
    installed, and this file's copy runs in the lane that has none. The
    SHIPPED decision function is what is called; only what it is given is
    chosen. ``tests/test_tripwire_eager.py`` drives the real sweep against
    the real jax.
    """
    canary = _canary()
    from stelling._tripwire import _adapter_jax as adapter

    # nothing to sweep is a NOTE and never a reason: no jax, or a hook that
    # did not attach, is not evidence that jax grew a constant.
    note, reason = canary._eager_sweep_row(False)
    assert reason is None and "not attached" in note

    def _sweep(result):
        monkeypatch.setattr(
            adapter, "eager_jax_constant_sweep", lambda: result, raising=False
        )

    _sweep({"code": "not-armed", "conversions": 0, "truncations": 0,
            "unmatched": (), "matched": ()})
    note, reason = canary._eager_sweep_row(True)
    assert reason is None and "not-armed" in note

    _sweep({"code": "swept", "conversions": 675, "truncations": 13,
            "x64": False,
            "unmatched": (),
            "matched": (("_src/random/threefry2x32.py", "_threefry_seed",
                         4294967295, "uint32", "int32"),)})
    note, reason = canary._eager_sweep_row(True)
    assert reason is None, note
    assert "675 conversion(s)" in note and "1 row(s) exercised" in note
    assert "JAX_ENABLE_X64=1" not in note, (
        "an x64-off sweep is the one that CAN find something, and the note "
        f"disclaimed it: {note}"
    )

    # ...AND THE SAME NUMBERS FROM AN x64-ON SWEEP ARE NOT A CLEARANCE. With
    # x64 on jax's mask widens to int64 and nothing of jax's narrows, so a
    # zero there says the sweep could not have looked. The qualification
    # travels with the number or a reader reads "0 unmatched" as "clean".
    _sweep({"code": "swept", "conversions": 729, "truncations": 0,
            "x64": True, "unmatched": (), "matched": ()})
    note, reason = canary._eager_sweep_row(True)
    assert reason is None, note
    assert "JAX_ENABLE_X64=1" in note and "could not have found" in note, (
        "a blind sweep printed its zeroes with no qualification, which reads "
        f"exactly like a sweep that looked and found nothing: {note}"
    )

    _sweep({"code": "swept", "conversions": 675, "truncations": 14,
            "unmatched": ((511, "int", "uint8",
                           (("_src/somewhere.py", "new_thing"),)),),
            "matched": ()})
    note, reason = canary._eager_sweep_row(True)
    assert reason is not None, "an unenumerated jax constant did not page"
    code, sentence = reason
    assert code == "eager:unenumerated-jax-constant"
    assert "new_thing" in sentence, (
        "the page does not name the jax function, so a reader has nothing to "
        "write a row from"
    )
    assert "UNENUMERATED" in note

    # a sweep that could not run says so and does not page: an instrument
    # that did not measure has not measured that nothing happened.
    def _boom():
        raise RuntimeError("FORCED")

    monkeypatch.setattr(
        adapter, "eager_jax_constant_sweep", _boom, raising=False
    )
    note, reason = canary._eager_sweep_row(True)
    assert reason is None and "RuntimeError" in note


def test_the_four_fatal_control_findings_do_not_read_as_each_other(
    capsys, tmp_path
):
    """The remedies differ, so the sentences must.

    `did-not-fire` sends a reader to a dead hook, `raised` to an environment
    where the probe could not run, `indeterminate` and `unrenderable` to THIS
    repository's own recorder. This is the one place the sentences are read,
    and it reads their OPENINGS rather than forbidding each other's words --
    the `raised` sentence deliberately NAMES `did not fire` in order to say
    it is not that, and a test forbidding the mention would forbid the
    distinction it exists to protect.

    READ OFF THE DRIVEN RUNS, not off the verdict function. A sentence this
    file composes by calling the decision directly is a sentence nobody has
    shown a real run prints; these are the four that four real runs printed.
    """
    import re

    sentences = {}
    for kw in (
        {"findings": "none"},
        {"probe": "raises"},
        {"findings": "unreadable"},
        {"findings": "field-moved"},
    ):
        sentences.update(_drive_canary(capsys, tmp_path, **kw).sentences)

    assert set(sentences) == {
        "control:did-not-fire", "control:raised",
        "control:indeterminate", "control:unrenderable",
    }, sorted(sentences)

    # TEN WORDS, CASE AND PUNCTUATION FOLDED AWAY. A character slice let a
    # mutation past: rewording `raised` to open "its LIVE CONTROL DID NOT
    # FIRE" -- the other finding's headline, in the sentence that goes on to
    # say it is a DIFFERENT finding -- still differed in the sixtieth
    # character. Words are what a reader takes from an opening, and the four
    # real openings need all ten to separate: `did not fire` and `did not
    # complete` agree on the first nine.
    def _opening(sentence):
        words = re.findall(r"[a-z0-9]+", sentence.lower())
        return tuple(words[:10])

    openings = {code: _opening(s) for code, s in sentences.items()}
    assert len(set(openings.values())) == 4, (
        f"two fatal findings open with the same ten words, so a reader who "
        f"stops at the first line cannot tell them apart: {openings}"
    )

    # neither of the two "this repository" findings may send anyone upstream
    for code in ("control:indeterminate", "control:unrenderable"):
        assert "not upstream" in sentences[code], code

    # the operator must not read "RAISED -- raised RuntimeError", which is
    # what the first version of this sentence produced: it opened with the
    # word and then embedded a row that already began with it
    assert "RAISED -- raised" not in sentences["control:raised"]


def test_the_canarys_documented_exit_codes_are_exactly_the_ones_it_produces():
    """The list at the top of the script, checked in BOTH directions.

    That paragraph has been wrong three times: once naming two of three
    exits, once counting ``return`` statements instead of reasons, and once
    claiming to be complete while two exits went unnamed. Prose that
    describes an exit code is the first thing an operator reads and the last
    thing anyone checks, so it is checked here against the script.

    AGAINST THE SCRIPT, AND NOT AGAINST ``_CANARY_TABLE``. This test used to
    build the produced set out of the table's expected reasons, which made it
    blind in the one direction that matters: a reason ``main()`` appends
    under a condition no row drives is a code no row names, so the set never
    grew and the list above could stay silent about it. A code added to the
    script with the table left alone was invisible here — and "the set a
    driven ``main()`` can actually produce" was then true only under an
    unstated assumption that the table is complete over the script. The
    source is parsed instead: every reason in that file is the literal pair
    ``("<code>", "<sentence>")``, which is the shape ``reasons`` is built
    from and the shape ``_control_reasons`` and ``_hash_row`` return.
    """
    import ast
    import re

    source = (
        _pathlib_for_canary() / ".github" / "scripts" / "tripwire_canary.py"
    ).read_text(encoding="utf-8")
    listed = source.split("EXIT CODES, ALL OF THEM", 1)[1].split('"""', 1)[0]
    documented = set(re.findall(r"^  1  `([a-z:-]+)`", listed, re.M))

    def _is_a_sentence(node):
        """Whether this tuple element is a string the script composed."""
        if isinstance(node, ast.Constant):
            return isinstance(node.value, str)
        if isinstance(node, ast.JoinedStr):
            return True
        if isinstance(node, ast.BinOp):
            return _is_a_sentence(node.left) or _is_a_sentence(node.right)
        return False

    produced = {
        node.elts[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Tuple)
        and len(node.elts) == 2
        and isinstance(node.elts[0], ast.Constant)
        and isinstance(node.elts[0].value, str)
        and re.fullmatch(r"[a-z][a-z-]*(:[a-z-]+)?", node.elts[0].value)
        and _is_a_sentence(node.elts[1])
    }

    assert documented == produced, (
        f"the exit-code list documents {sorted(documented)} and the script "
        f"carries reasons for {sorted(produced)}. Undocumented: "
        f"{sorted(produced - documented)}. Documented but not in the script: "
        f"{sorted(documented - produced)}."
    )

    # AND THE PARSE IS ITSELF CONTROLLED. A reader of the set above is owed
    # evidence that it is not an artifact of the shape this test happens to
    # match: every code a real driven run printed must be in it. That is the
    # table, and the two codes no input can drive, which their own tests
    # above call directly.
    driven = {code for row in _CANARY_TABLE for code in row[3]}
    driven |= {code for row in _PERIMETER_TABLE for code in row[3]}
    driven |= {"control:unknown-state", "hash:unknown-state",
               "eager:unenumerated-jax-constant"}
    assert driven <= produced, (
        f"runs of this script printed {sorted(driven - produced)}, which the "
        "parse above did not find in its source: the reasons are no longer "
        "the literal pairs this test reads, so the set it checks the "
        "documentation against is not the script's"
    )
    assert "  2  " in listed, (
        "the list no longer names argparse's exit 2, which a reader meets by "
        "mistyping `--require` and which this list twice claimed not to exist"
    )


# ==========================================================================
# THE WORKFLOW READER, AND WHY IT IS A PARSER RATHER THAN A PATTERN
# ==========================================================================
#
# WHAT WAS HERE UNTIL 2026-08-23 WAS A LINE-ANCHORED REGEX, AND THREE ROUNDS
# OF AUDIT WENT PAST IT — nine ways in the third round alone, every one of
# them ORDINARY WORKFLOW YAML that GitHub accepts and runs. All nine were
# measured on `.github/workflows/nightly-jax-canary.yml` and every one left
# this file at `95 passed`:
#
#     "JAX_ENABLE_X64": "1"              a quoted key       — the step ran ON
#     env: { JAX_ENABLE_X64: "1", … }    a flow mapping     — the step ran ON
#     env: *x64on                        an alias           — the step ran ON
#     strategy.matrix.exclude            the OFF cell excluded from the matrix
#     runs-on: before name:              a whole third leg invisible
#     if: … at JOB level                 the whole job may not run
#     defaults: {run: {shell: bash -lc}} a shell whose startup nobody read
#     set -a; . ./ci.env; set +a         sets it without naming it
#     continue-on-error: true            the step's red cannot fail the job
#
# **AND THEY FAIL IN THREE DIFFERENT DIRECTIONS, WHICH IS THE ARGUMENT.**
# The first three are ONE `env:` mapping spelled three legal ways: there the
# pattern matched nothing, the reader concluded `"unset"`, and both callers
# read `"unset"` as OFF while the step ran at ON. `matrix.exclude` is a cell
# the reader RESOLVED and still got wrong — it expanded the axis and never
# saw the entry taken back out of it. `runs-on:` before `name:` is a whole
# LEG it never saw at all. And the last four are cells it read CORRECTLY and
# was not entitled to CLAIM, because the step might not run, might run under
# a shell nobody read, or could not fail the job if it did. One pattern set
# cannot be patched into covering three failure directions; a parse and a
# whitelist can.
#
# **THE FIX IS NOT A TENTH PATTERN.** A regex cannot parse YAML, and every
# round bought one coordinate: the charset fix bought job-id characters and
# the next round found key ORDER. What is wrong is not the patterns, it is
# that a MISS was allowed to produce a VALUE. `"unset"` was a can't-tell
# wearing a cell's name — the same conflation this branch already fixed one
# layer up, moved from the value layer to the FIND layer.
#
# So there are two readers and one model:
#
# 1. **`yaml.safe_load` where PyYAML is importable.** A real parse resolves
#    quoted keys, flow mappings, aliases, key order and `exclude` together,
#    because YAML semantics replace text matching. Measured: PyYAML 6.0.3 is
#    in `/home/nick/venvs/stelling-jax` and NOT in
#    `/home/nick/venvs/stelling-nojax`, and it is not a declared dependency
#    of this project — so it cannot be required, and the zero-dep lane is
#    exactly where this guard must keep working.
# 2. **A TOTAL LINE GRAMMAR where it is not**, which is the whole of why the
#    fallback can now say "there is no setting" at all. A pattern that
#    searches can only ever report "I did not find one"; a grammar that
#    CLASSIFIES EVERY LINE of the file can report "I have read all of it and
#    it is not there". Every line the grammar cannot place makes the whole
#    reading a can't-tell — so the failure of the fallback is loud, and the
#    thing it must never do (call a miss a cell) is now structurally
#    impossible rather than patched out one spelling at a time.
#
# Where both read a file they produce the SAME mapping — the one PyYAML
# would build — and everything below `_model` is shared, so a cell is
# decided in one place.
# `test_which_reader_this_lane_uses_IS_ASSERTED_AND_NOT_ASSUMED` holds them
# to each other on the file that matters.
#
# **AND A FOURTH ROUND FOUND THE WORD "LINE".** "Classifies every line" is a
# claim about the whole file only if "line" is the file's own notion of one,
# and it was `str.split("\n")`. YAML's is wider: PyYAML breaks a line on CR,
# U+0085, U+2028 and U+2029 as well. Driven on the real workflow with ONE
# U+2028 in place of the newline between two entries of a step's `env:`
# block: the whole zero-dep suite came back IDENTICAL to clean while the
# grammar reported the cell `"unset"` — the word that means *I have read all
# of it* — for a `JAX_ENABLE_X64:` line it had never classified. `_shell_
# reason` had the same split and the same consequence, hiding
# `./setup-the-cell.sh` inside an `echo`. Two more of the same round: a key
# written TWICE merged here and replaces in PyYAML, so the two readers
# resolved one file to two different cells; and an indent-4 job key with no
# job above it raised `TypeError` out of the grammar, which is a third
# outcome beside "classify" and "refuse". See `_lines_of_this_grammar`,
# `_put` and `test_the_line_grammar_HAS_NO_THIRD_OUTCOME`.
#
# WHAT IS STILL NOT PARSED, said rather than left to be found: expressions
# (`${{ … }}`) other than a bare `matrix.<axis>` reference, shell, and what
# a third-party action does. None of the three is guessed at — each is a
# named can't-tell, and the lists that make an action or a shell word
# readable are written out below so that widening one is a decision in a
# diff rather than a silence.
#
# AND ONE THING THAT CANNOT BE A CAN'T-TELL, BECAUSE REFUSING IT WOULD
# REFUSE THE WORKFLOW: WHAT AN INVOKED PROGRAM DOES. Every step this reader
# credits runs `.venv/bin/python <something>`, and a python program is free
# to write `os.environ["JAX_ENABLE_X64"]` before it imports jax. The reader
# reads the environment the RUNNER hands the process; it does not and
# cannot follow the process.
#
# **AND THE BOUND WAS STATED TOO NARROWLY, WHICH MADE IT SOUND LIKE ONE
# STEP'S PROBLEM.** It read *"this is the bound: this step is started in
# this cell"*, and that is a claim about the step whose script this reader
# is looking at. An invoked program does not only change ITS OWN
# environment: a program that appends `JAX_ENABLE_X64=1` to the file named
# by `$GITHUB_ENV` changes the cell every LATER STEP OF THE JOB is started
# in. Driven on the real workflow, as the TWELFTH way past this reader: an
# earlier nightly step running `.venv/bin/python .github/scripts/
# prepare_cell.py`, plus the x64-OFF step's own `env:` block removed. Both
# readers said `unset`, `114 passed` in both lanes and the whole zero-dep
# suite unchanged. The `GITHUB_ENV` substring guard in `_model_step` cannot
# fire, because the name is INSIDE the program and not in the `run:` line;
# `.venv/bin/python` is on the shell word list precisely so that steps like
# that read as inert; and the step that ran at the wrong cell is a step this
# reader credited with a cell it never had.
#
# So the bound is this, and it is a bound on the JOB and not on the step:
# *every step of this job is started in the cell this file spells for it,
# unless a program one of its steps invokes wrote `$GITHUB_ENV`.* That is
# not closable from here without refusing every `.venv/bin/python` step,
# which is every step the workflow has. `--require`, and the canary's own
# live control rows, are what stand behind the rest of it — a canary that
# was handed the wrong cell by a program still measures what it measures
# and still reports the cell it ran in. The one place the boundary is
# watchable from here is the shell layer, which is why `.venv/bin/python` is
# on the word list and `.` and `source` and `env` are not: a program's own
# behaviour is somebody's code, a sourced file is a fact about the step.

#: A GitHub job id: a leading letter or `_`, then letters, digits, `-`, `_`.
#: Spelled once. This read `[a-z-]+` until 2026-08-22, and a third job
#: `nightly_x64:` — an alarm wired to the one cell the eager sweep cannot
#: redden in — was invisible to every test in this repository.
_JOB_ID = r"[A-Za-z_][A-Za-z0-9_-]*"

# JAX'S OWN GRAMMAR FOR THIS VARIABLE, because jax's is the grammar that
# decides what a run measures and this reader's opinion is worth nothing next
# to it. Its boolean-environment reader takes `y yes t true on 1` as true and
# `n no f false off 0` as false, case-insensitively, and RAISES on anything
# else. That is not read off its source -- naming the private module it lives
# in is what `tests/test_import_hygiene.py` forbids outside the adapter, and
# it caught this comment doing it. It is MEASURED, on jax 0.11.0, through the
# public `jax.config.jax_enable_x64` after import: `true`, `on`, `yes`, `t`,
# `y`, `TRUE`, `On` -> True; `false`, `off`, `no`, `n`, `f` -> False;
# unset -> False.
#
# The YAML layer agrees with it rather than fighting it, which is why one
# table serves both: `JAX_ENABLE_X64: true` and `JAX_ENABLE_X64: on` are YAML
# booleans, the runner renders them into the environment as `"true"`, and jax
# reads `"true"` as ON -- the same answer as reading the token as written.
# Same for `off`/`no` and OFF. It is also why the two readers agree on them:
# PyYAML resolves `on` to `True` and the line grammar reads the word `on`,
# and both land on `"1"`.
_X64_TRUE = ("y", "yes", "t", "true", "on", "1")
_X64_FALSE = ("n", "no", "f", "false", "off", "0")

# The step conditions this reader knows how to evaluate. A step whose `if:`
# is not one of these MIGHT NOT RUN, and a step that might not run cannot be
# credited with running in a cell -- so it is a can't-tell and not a cell.
# Driven: `if: steps.install.outcome == 'success' && false` on the x64=0
# pytest step left this file at `68 passed` while that step never runs.
# Adding a condition here is a decision that it still runs; that is the point
# of it being a written list rather than a pattern.
#
# **AND THE DECISION IS CONDITIONAL ON ITS REFERENT, WHICH IT WAS NOT.**
# `steps.install.outcome == 'success'` is a decision that the step still runs
# only while a step with `id: install` EXISTS. Driven on the real workflow:
# renaming that one step to `id: install-nightly` and touching nothing else
# left this file at `95 passed` with the cells unchanged — while on the
# runner all four canary and pytest steps skip, the `!= 'success'` branch
# goes true, and the leg reports *"SKIPPED (infrastructure)"* and comes back
# GREEN. Nothing in `tests/` or `.github/` pinned that id. Every
# `steps.<id>.` a known condition names is now resolved against the job's own
# steps, and an unresolvable one is a can't-tell.
_STEP_CONDITIONS_THIS_READER_KNOWS = frozenset({
    "steps.install.outcome == 'success'",
})

#: `steps.<id>.` inside a condition. The referent, which has to exist.
_STEP_REFERENCE = re.compile(r"\bsteps\.([A-Za-z_][A-Za-z0-9_-]*)\.")

# THE ACTIONS THIS READER HAS DECIDED DO NOT SET THIS VARIABLE. A `uses:`
# step runs somebody else's code, and a composite action writing
# `JAX_ENABLE_X64` into `$GITHUB_ENV` would change the cell of every later
# step of the job with nothing in this file to see. That was named as a known
# hole and left open; it is a can't-tell now, and these two are the written
# exception. THE CLAIM IS NARROW AND IS THE ONLY ONE BEING MADE: neither
# action sets `JAX_ENABLE_X64`. It is not a claim that either action is
# inert — an action is free to write other names into `$GITHUB_ENV`, and
# this list would still be right — which is why the entry is about this
# variable and not about the action.
_ACTIONS_THIS_READER_KNOWS = frozenset({
    "actions/checkout@v4",
    "astral-sh/setup-uv@v6",
})

# THE SHELL WORDS THIS READER HAS DECIDED CANNOT CHANGE THE ENVIRONMENT, and
# the reason this is a whitelist. The `run:` can't-tell used to be a
# SUBSTRING TEST FOR THE VARIABLE'S NAME, which asks whether the script
# mentions it rather than whether it sets it. Driven on the real workflow's
# x64 OFF pytest step:
#
#     set -a; . ./ci.env; set +a
#
# — three words, no `JAX_ENABLE_X64` anywhere, every variable in `ci.env`
# exported into the step, and this file at `95 passed`. `export`, `eval`,
# `env`, `source`, a `VAR=value` command prefix and `declare -x` are six
# more, and enumerating them is the game this reader keeps losing. So a
# script is INERT only if every command word in it is one of these; anything
# else is a can't-tell that names the word. THE LIST IS EXACTLY WHAT THIS
# WORKFLOW RUNS and nothing kept in reserve, because a word nobody uses is a
# licence nobody reviewed. Adding one is a decision that it cannot export a
# variable, in a diff, like every other list in this file.
_SHELL_WORDS_THIS_READER_KNOWS = frozenset({"uv", ".venv/bin/python", "echo"})

#: `set` is the one word with an argument this reader has an opinion about:
#: `set -a` and `set -o allexport` turn every later assignment into an
#: export, which is how the measured mutation above worked without naming
#: anything.
_SET_FLAGS_THAT_EXPORT = ("a", "allexport")

#: EVERY CHARACTER SOME READER OF A `run:` SCRIPT MAY TREAT AS A LINE BREAK.
#: `_shell_reason` split on `\n` alone until 2026-08-24 and a single U+2028
#: put `./setup-the-cell.sh` inside an `echo` — see that function for the
#: measurement and for why splitting on more than the shell itself does is
#: the conservative direction here and a refusal is not needed.
_LINE_BREAKS_A_SHELL_MIGHT_MEET = ("\r\n", "\r", "\u0085", "\u2028", "\u2029")

# THE KEYS THIS READER KNOWS, AT EACH LEVEL, AND WHY AN UNKNOWN ONE IS A
# CAN'T-TELL. FOUR of the nine ways past the old reader were a key it had no
# pattern for — a job-level `if:`, a `defaults:` block,
# `strategy.matrix.exclude` and `continue-on-error:` — and a fifth (a
# reusable-workflow `uses:` job) was named in the old docstring as a known
# hole. They are one defect: the reader read the keys it knew and was SILENT
# about the rest, and silence resolved to a cell. A key that is not here
# stops the reading instead, which also means the next GitHub feature nobody
# has heard of fails closed on arrival. (`continue-on-error:` is on the step
# list and MODELLED rather than merely refused, because a step may
# legitimately carry it — the `install` step does, and that is the whole of
# how the infrastructure carve-out works. What it may not do is carry it and
# still be counted as a step that can go red.)
#
# `on`, `permissions` and `concurrency` are listed as IRRELEVANT rather than
# modelled: nothing under them reaches a step's environment. The one route
# they could — an anchor defined there and merged into a job — is refused at
# the point of USE, because `<<` is not a job key and an alias is not a value
# either reader will resolve to a cell.
_WORKFLOW_KEYS_THIS_READER_KNOWS = frozenset({
    "name", "on", "permissions", "concurrency", "jobs", "env",
})
_JOB_KEYS_THIS_READER_KNOWS = frozenset({
    "name", "runs-on", "env", "steps", "strategy", "timeout-minutes",
    "permissions", "concurrency",
})
_STRATEGY_KEYS_THIS_READER_KNOWS = frozenset(
    {"matrix", "fail-fast", "max-parallel"}
)
_STEP_KEYS_THIS_READER_KNOWS = frozenset({
    "name", "uses", "run", "env", "if", "id", "continue-on-error", "with",
})
#: Top-level keys whose bodies cannot reach a step's environment, so the line
#: grammar skips them wholesale rather than parsing structure it has no use
#: for. See the paragraph above for why that is safe.
_WORKFLOW_KEYS_WITH_NOTHING_IN_THEM_FOR_US = frozenset({
    "name", "on", "permissions", "concurrency",
})

# `${{ matrix.<axis> }}`, the ONE expression shape this reader follows, and it
# follows it the way `tests/_lanes.py` follows `${EXTRAS}`: the chain is read
# entry by entry, and any link failing is a NAMED can't-tell rather than a
# default.
_X64_MATRIX_REF = re.compile(
    r"\$\{\{[ \t]*matrix\.([A-Za-z_][A-Za-z0-9_-]*)[ \t]*\}\}"
)

#: Is a real YAML parser importable in THIS lane? PyYAML 6.0.3 is in
#: `/home/nick/venvs/stelling-jax` and absent from `stelling-nojax`, and it is
#: not a declared dependency of this project — so this is a fact about the
#: environment and never a reason to skip: the guards run, and can fail,
#: either way.
#:
#: **AND WHERE IT IS PRESENT, IT IS PRESENT BY ACCIDENT, WHICH IS WHY THE
#: LINE GRAMMAR IS THE ONE THAT HAS TO BE RIGHT.** Measured 2026-08-23:
#: nothing in `pyproject.toml` names PyYAML and no `uv pip install` line in
#: `.github/workflows/ci.yml` names it. In the shared jax venv it arrives
#: only as a transitive requirement of test-only libraries (`flax`,
#: `maddening`, `ml_collections`).
#:
#: **WHAT THIS PARAGRAPH SAID UNTIL 2026-08-24 WAS FALSE, AND IT WAS THE
#: SENTENCE A DECISION DOCUMENT QUOTED.** It read *"the one CI job whose
#: install closure does carry it — `acceptance-reproducer` — runs eighteen
#: NAMED acceptance tests and not this file. So no CI job that runs this
#: file has a YAML parser."* The first half is a real step of that job and
#: the second half ignores the step after it. `acceptance-reproducer`
#: installs jaxfluids **with** its dependencies — deliberately, so that a
#: dependency moving jax fails the job rather than hiding — and then runs
#: ``.venv/bin/python -m pytest -q -ra``, the whole tree, which `ci.yml`
#: calls *"unnarrowed on purpose"* and which `tests/_lanes.py` already reads
#: as ``whole_suite=True``. Re-measured from installed metadata: jaxfluids
#: requires `flax`, `flax` requires `PyYAML>=5.4.1`, and `ci.yml`'s own
#: comment says pyyaml *"arrives as one of those dependencies"*. **So there
#: IS a CI job that runs this file with a YAML parser importable, and it is
#: `acceptance-reproducer` — both matrix entries.**
#:
#: Three things follow, and they are why this is worth a paragraph rather
#: than a correction:
#:
#: 1. **The parser's column of `_RESOLUTIONS` is asserted in CI**, not only
#:    wherever a developer's environment happens to carry PyYAML.
#: 2. **That job is the ONLY place the two-reader agreement can be checked
#:    at all.** `test_which_reader_this_lane_uses_IS_ASSERTED_AND_NOT_ASSUMED`
#:    compares the line grammar and the parser on the real workflow, and the
#:    comparison needs both readers present; in every other lane its second
#:    half returns without comparing anything.
#: 3. **And by `ci.yml`'s own policy that job MUST NOT be a required check**
#:    — it fetches a third-party repository at a commit, so an upstream
#:    force-push can redden it on a day this repository did not change. So
#:    the one job that can catch the two readers disagreeing is a job whose
#:    red does not block a merge. That is not an argument for making it
#:    required; it is the reason the line grammar is still the one that has
#:    to be right, and the reason its column is asserted in every lane.
_YAML_IS_IMPORTABLE = importlib.util.find_spec("yaml") is not None


def _cannot_tell(reason: str) -> str:
    """A cell nobody may read as a cell. `?` prefixed so it can never be
    mistaken for one: `"0"`, `"1"` and `"unset"` are the only readable
    values and every caller goes through :func:`_refuse_unreadable` first."""
    return "?" + reason


def _refuse_unreadable(cells, where: str) -> None:
    """FAIL on a can't-tell instead of passing over it.

    THIS IS THE WHOLE POINT OF THE SPLIT. `_x64_cells` returned the single
    string ``"unset"`` for BOTH "no setting anywhere" -- which is jax's
    default and genuinely OFF, measured -- AND "a setting I could not parse",
    and both callers then read the conflated value as OFF. Eight measured
    mutations of `.github/workflows/nightly-jax-canary.yml` ran at x64 ON
    while the guard believed otherwise and this file stayed at `68 passed`:
    `JAX_ENABLE_X64: true`, `"on"`, `${{ matrix.x64 }}` over a one-cell
    matrix, `export JAX_ENABLE_X64=1` inside the `run:` block, a
    `JAX_ENABLE_X64=1` command prefix, an earlier step writing the name into
    `$GITHUB_ENV`, `&& false` on the OFF step, and the OFF step moved into
    the other job. Two of those -- `true` and `on` -- jax itself accepts.

    NINE MORE WERE FOUND THE ROUND AFTER, and they are why the reader above
    this line is a parser: a quoted key, a flow mapping, an alias,
    `matrix.exclude`, `runs-on:` written before `name:`, a job-level `if:`,
    a `defaults:` block, `set -a; . ./ci.env; set +a`, and
    `continue-on-error: true` on the step whose red is the whole signal.
    All nine measured at `95 passed` on the real workflow: the first four
    a cell read WRONG, the fifth a whole LEG never seen, the last four a
    cell this reader was not entitled to claim at all.

    A can't-tell that defaults to the answer the caller wanted is the shape
    this campaign has already named once, in `Lane.jax`. It gets a name here
    and a red there.

    THREE MORE THE ROUND AFTER THAT, AND THEY ARE A DIFFERENT KIND OF
    MEASUREMENT, SAID SO ON PURPOSE. None of the three is a demonstrated run
    at the wrong cell — what was measured is an un-classified line reported
    as a cell, and two readers of one file landing on different cells:

    * a **U+2028** (or U+0085, or U+2029, or a CR) in place of the newline
      between two entries of a step's `env:` block. The line grammar split
      on `"\n"`, so the `JAX_ENABLE_X64:` line was swallowed into the value
      above it and never classified — and the reading came back ``"unset"``,
      the one word that means *I have read all of it*. Whether GitHub's own
      parser breaks a line there is untested here (YAML 1.1 does, 1.2 does
      not), which is why these are refused rather than resolved;
    * a **key written twice in one mapping**, which this grammar merged and
      PyYAML replaces — a reader divergence on the real workflow, with the
      zero-dep suite unmoved;
    * an **invoked program writing `$GITHUB_ENV`**, which is the one bound
      this reader states rather than closes — see the block above
      `_JOB_ID`. That one is NOT refused and cannot be, because refusing it
      refuses every step of the workflow.
    """
    unreadable = [cell[1:] for cell in cells if cell.startswith("?")]
    assert not unreadable, (
        f"{where}: this reader cannot tell which JAX_ENABLE_X64 cell "
        f"{'these steps run' if len(unreadable) > 1 else 'this step runs'} "
        f"in — {unreadable}. That is a can't-tell and not a cell, and it is "
        f"refused rather than read as OFF: seventeen of the twenty shapes "
        f"`_refuse_unreadable` lists were MEASURED running at x64 ON while "
        f"the guard that conflated the two believed otherwise, and the other "
        f"three are a cell this reader named without having read the line it "
        f"came from. Either spell the setting so "
        f"this reader can resolve it, or teach the reader the shape and say "
        f"in the same commit what you measured it to mean"
    )


def _key_name(key) -> str:
    """A mapping key as this reader names it.

    PyYAML resolves the bare words `on`, `off`, `yes` and `no` to booleans,
    so a workflow's own `on:` trigger key comes back as `True` and a
    whitelist of strings would not contain it. The line grammar reads the
    word. Normalising here is what lets ONE whitelist serve both readers.
    """
    if key is True:
        return "on"
    if key is False:
        return "off"
    return str(key)


def _x64_word(value) -> str:
    """One written value -> ``"0"`` / ``"1"`` / a can't-tell, jax's grammar.

    The value arrives already typed from PyYAML (`True`, `0`, `"1"`) and as
    written text from the line grammar (`true`, `0`, `"1"` with the quotes
    still on). Both land on the same table, which is the point: a YAML
    boolean renders into the environment as `"true"` and jax reads `"true"`
    as ON, so reading the word and reading the type give one answer.
    """
    if value is True:
        return "1"
    if value is False:
        return "0"
    if isinstance(value, int):
        token = str(value)
    elif isinstance(value, str):
        token = value
    else:
        return _cannot_tell(
            f"JAX_ENABLE_X64: {value!r} is a {type(value).__name__} and not a "
            f"value that can be put in an environment at all"
        )
    word = token.strip().strip('"').strip("'").strip().lower()
    if word in _X64_TRUE:
        return "1"
    if word in _X64_FALSE:
        return "0"
    return _cannot_tell(
        f"JAX_ENABLE_X64: {token!r} is not a value jax's own `bool_env` "
        f"reads ({'/'.join(_X64_TRUE)} or {'/'.join(_X64_FALSE)})"
    )


# --------------------------------------------------------------------------
# THE SHELL GRAMMAR: which scripts this reader will believe are inert
# --------------------------------------------------------------------------


def _shell_reason(script: str):
    """``None`` if this script cannot change the environment, else why not.

    Not "does it name JAX_ENABLE_X64" — that question was answered `no` by
    `set -a; . ./ci.env; set +a`, which exports every variable in a file the
    reader has never seen. Every command word must be one this reader has
    decided cannot export (see `_SHELL_WORDS_THIS_READER_KNOWS`), and a
    `VAR=value` command prefix, a `set` that turns on `allexport`, and an
    unparseable line are each a refusal of their own.

    **AND "LINE" WAS `str.split("\\n")`, WHICH HID A COMMAND BEHIND AN INERT
    ONE.** `shlex` does not treat CR, U+0085, U+2028 or U+2029 as whitespace,
    so `echo preparing<U+2028>./setup-the-cell.sh` tokenised as ONE command
    whose head is `echo` — whitelisted — and this function returned ``None``,
    while `_RESOLUTIONS` carries that same `./setup-the-cell.sh` as a row that
    must refuse. Measured, all four characters, at `4a13824`. Every one of
    them is now a break here, and splitting on MORE than the shell does is
    the safe direction and the reason this is a split and not a refusal: an
    extra break can only produce more command words for the whitelist to
    reject, never fewer. (Driven both ways: `#comment<U+2028>export FOO=1` was
    read as one comment line and is two lines now, the second of which
    refuses.)
    """
    joined = script
    for brk in _LINE_BREAKS_A_SHELL_MIGHT_MEET:
        joined = joined.replace(brk, "\n")
    joined = re.sub(r"\\\n[ \t]*", " ", joined)
    for raw in joined.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError as exc:  # an unbalanced quote, say
            return f"this reader cannot even tokenise the line {line!r} ({exc})"
        command: list[str] = []
        for token in [*tokens, ";"]:
            if token in (";", "&&", "||", "|", "&", "\n"):
                reason = _command_reason(command, line)
                if reason is not None:
                    return reason
                command = []
                continue
            command.append(token)
    return None


def _command_reason(command, line):
    """``None`` if one command of a script is inert, else why not."""
    words = [w for w in command if w not in ("{", "}", "(", ")")]
    while words and words[0] in (">", ">>", "<"):
        words = words[2:]
    if not words:
        return None
    head, *rest = words
    if head in (">", ">>", "<"):
        return None
    if "=" in head.split("/")[0]:
        return (
            f"the line {line!r} carries a `{head}` assignment, which sets a "
            f"variable for the command that follows it"
        )
    if head == "set":
        for flag in rest:
            stripped = flag.lstrip("-+")
            if stripped in _SET_FLAGS_THAT_EXPORT or (
                not flag.startswith(("-o", "+o")) and "a" in stripped
                and flag.startswith(("-", "+"))
            ):
                return (
                    f"the line {line!r} turns on shell `allexport`, which "
                    f"exports every variable set after it — including any a "
                    f"sourced file sets"
                )
        return None
    if head not in _SHELL_WORDS_THIS_READER_KNOWS:
        return (
            f"the line {line!r} runs `{head}`, which is not a command this "
            f"reader has decided cannot export a variable (it knows "
            f"{sorted(_SHELL_WORDS_THIS_READER_KNOWS)} and `set`)"
        )
    return None


# --------------------------------------------------------------------------
# THE MODEL: one shape, two readers
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Step:
    """One step, as either reader sees it.

    ``blockers`` are can't-tells about THIS step alone; ``job_blockers`` are
    the ones that reach every other step of the job, which is the split the
    `$GITHUB_ENV` mutant taught: a script this reader cannot read might
    write the variable for every LATER step, so it poisons the job and not
    just itself.
    """

    name: object = None
    uses: object = None
    run: object = None
    id: object = None
    condition: object = None
    env: object = None
    continue_on_error: object = None
    blockers: tuple = ()
    job_blockers: tuple = ()


@dataclasses.dataclass(frozen=True)
class _Job:
    id: str
    name: object = None
    env: object = None
    matrix: object = None
    steps: tuple = ()
    blockers: tuple = ()


@dataclasses.dataclass(frozen=True)
class _Workflow:
    parser: str
    env: object = None
    jobs: tuple = ()
    blockers: tuple = ()


class _Opaque:
    """A region the line grammar read past without modelling — see
    `_WORKFLOW_KEYS_WITH_NOTHING_IN_THEM_FOR_US`."""

    def __repr__(self):  # pragma: no cover - diagnostics only
        return "<opaque>"


class _Unloadable(str):
    """A parse that failed, carrying its own reason. `_model` refuses it,
    so a workflow PyYAML will not load is a can't-tell with a sentence
    rather than a traceback."""


def _mapping_of_strings(value, where):
    """``(mapping, blockers)`` for an ``env:`` block."""
    if value is None:
        return {}, ()
    if not isinstance(value, dict):
        return {}, (f"the {where} `env:` is {value!r}, which is not a mapping "
                    f"this reader can read entry by entry",)
    return {_key_name(k): v for k, v in value.items()}, ()


def _model(raw, parser: str) -> _Workflow:
    """The mapping a YAML parser builds -> the model both guards read.

    EVERYTHING THAT DECIDES A CELL IS DECIDED HERE, once, for both readers.
    The two parsers differ only in how they turn text into this mapping, and
    `test_which_reader_this_lane_uses_IS_ASSERTED_AND_NOT_ASSUMED` holds
    them to the same answer on the file the guards actually read.
    """
    blockers = []
    if isinstance(raw, _Unloadable):
        return _Workflow(parser=parser, blockers=(str(raw),))
    if not isinstance(raw, dict):
        return _Workflow(parser=parser,
                         blockers=("this workflow is not a mapping at all",))
    for key in raw:
        if _key_name(key) not in _WORKFLOW_KEYS_THIS_READER_KNOWS:
            blockers.append(
                f"the workflow carries a top-level `{_key_name(key)}:` key, "
                f"which this reader does not model — `defaults:` alone can "
                f"change the shell every `run:` in the file executes in"
            )
    env, env_blockers = _mapping_of_strings(raw.get("env"), "workflow's")
    blockers += list(env_blockers)

    jobs = []
    raw_jobs = raw.get("jobs")
    if raw_jobs is None:
        blockers.append("the workflow has no `jobs:` mapping")
        raw_jobs = {}
    elif not isinstance(raw_jobs, dict):
        blockers.append("the workflow's `jobs:` is not a mapping")
        raw_jobs = {}
    for job_id, body in raw_jobs.items():
        jobs.append(_model_job(_key_name(job_id), body))
    return _Workflow(parser=parser, env=env, jobs=tuple(jobs),
                     blockers=tuple(blockers))


def _model_job(job_id: str, body) -> _Job:
    blockers = []
    if not isinstance(body, dict):
        return _Job(id=job_id, blockers=(f"the `{job_id}` job is not a mapping",))
    for key in body:
        if _key_name(key) not in _JOB_KEYS_THIS_READER_KNOWS:
            blockers.append(
                f"the `{job_id}` job carries a `{_key_name(key)}:` key, which "
                f"this reader does not model — a job-level `if:` can stop the "
                f"whole job, a `uses:` makes it somebody else's workflow, and "
                f"a `defaults:` changes every script in it"
            )
    env, env_blockers = _mapping_of_strings(body.get("env"), f"`{job_id}` job's")
    blockers += list(env_blockers)

    matrix = None
    strategy = body.get("strategy")
    if strategy is not None:
        if not isinstance(strategy, dict):
            blockers.append(f"the `{job_id}` job's `strategy:` is not a mapping")
        else:
            for key in strategy:
                if _key_name(key) not in _STRATEGY_KEYS_THIS_READER_KNOWS:
                    blockers.append(
                        f"the `{job_id}` job's `strategy:` carries "
                        f"`{_key_name(key)}:`, which this reader does not model"
                    )
            matrix = strategy.get("matrix")

    steps = []
    raw_steps = body.get("steps")
    if raw_steps is None:
        raw_steps = []
    elif not isinstance(raw_steps, list):
        blockers.append(f"the `{job_id}` job's `steps:` is not a sequence")
        raw_steps = []
    for index, raw_step in enumerate(raw_steps):
        steps.append(_model_step(job_id, index, raw_step))
    return _Job(id=job_id, name=body.get("name"), env=env, matrix=matrix,
                steps=tuple(steps), blockers=tuple(blockers))


def _model_step(job_id: str, index: int, raw) -> _Step:
    where = f"step {index} of the `{job_id}` job"
    if not isinstance(raw, dict):
        return _Step(job_blockers=(f"{where} is not a mapping",))
    blockers, job_blockers = [], []
    for key in raw:
        if _key_name(key) not in _STEP_KEYS_THIS_READER_KNOWS:
            job_blockers.append(
                f"{where} carries a `{_key_name(key)}:` key, which this "
                f"reader does not model"
            )
    env, env_blockers = _mapping_of_strings(raw.get("env"), where)
    blockers += list(env_blockers)

    uses = raw.get("uses")
    if uses is not None and uses not in _ACTIONS_THIS_READER_KNOWS:
        job_blockers.append(
            f"{where} `uses: {uses}`, and an action this reader has not been "
            f"told about can write JAX_ENABLE_X64 into $GITHUB_ENV for every "
            f"later step of the job"
        )

    run = raw.get("run")
    if run is not None:
        if not isinstance(run, str):
            job_blockers.append(f"{where} has a `run:` that is not a string")
        else:
            if "GITHUB_ENV" in run:
                job_blockers.append(
                    f"{where} names $GITHUB_ENV, which sets variables for "
                    f"every LATER step of the job"
                )
            if "JAX_ENABLE_X64" in run:
                blockers.append(
                    f"{where}'s `run:` script names JAX_ENABLE_X64, and this "
                    f"reader does not evaluate shell"
                )
            reason = _shell_reason(run)
            if reason is not None:
                job_blockers.append(f"{where}: {reason}")

    continue_on_error = raw.get("continue-on-error")
    if continue_on_error is not None and continue_on_error is not False:
        blockers.append(
            f"{where} is `continue-on-error: {continue_on_error!r}`, so its "
            f"red cannot fail the job and it cannot carry a signal"
        )
    return _Step(name=raw.get("name"), uses=uses, run=run, id=raw.get("id"),
                 condition=raw.get("if"), env=env,
                 continue_on_error=continue_on_error,
                 blockers=tuple(blockers), job_blockers=tuple(job_blockers))


def _resolve_conditions(job: _Job) -> _Job:
    """A step's `if:` against the job's OWN steps, which is where its
    referent lives.

    See `_STEP_CONDITIONS_THIS_READER_KNOWS` for the measured mutation: the
    condition stayed on the list, the step it names was renamed, and the
    reader went on crediting four steps with cells they no longer run in.
    """
    ids = {step.id for step in job.steps if step.id is not None}
    steps = []
    for index, step in enumerate(job.steps):
        extra = []
        condition = step.condition
        if condition is not None:
            where = f"step {index} of the `{job.id}` job"
            written = condition if isinstance(condition, str) else repr(condition)
            if written.strip() not in _STEP_CONDITIONS_THIS_READER_KNOWS:
                extra.append(
                    f"{where}'s `if: {written}` is not a condition this "
                    f"reader can evaluate, so it cannot be credited with "
                    f"running at all"
                )
            else:
                missing = sorted(
                    set(_STEP_REFERENCE.findall(written)) - ids
                )
                if missing:
                    extra.append(
                        f"{where}'s `if: {written}` is evaluated against step "
                        f"id(s) {missing} that no step of this job carries, so "
                        f"the condition this reader was told still runs names "
                        f"nothing — on the runner it is false and the step "
                        f"SKIPS"
                    )
        steps.append(dataclasses.replace(
            step, blockers=step.blockers + tuple(extra)))
    return dataclasses.replace(job, steps=tuple(steps))


# --------------------------------------------------------------------------
# READER 1: PyYAML, where it is importable
# --------------------------------------------------------------------------


def _parse_with_yaml(text: str):
    """``yaml.safe_load``, and a MALFORMED file is a refusal and not a crash.

    A workflow this parser cannot load is a workflow nobody can read, which
    is a can't-tell like any other and belongs in `_refuse_unreadable`'s
    message rather than in a traceback.
    """
    import yaml

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return _Unloadable(f"PyYAML will not load this workflow: {exc}")



# --------------------------------------------------------------------------
# READER 2: A TOTAL LINE GRAMMAR, where it is not
# --------------------------------------------------------------------------
#
# EVERY LINE IS CLASSIFIED OR THE WHOLE READING STOPS. That is the property
# that lets this reader say "there is no setting" rather than only "I did not
# find one", and it is the difference between this and the pattern it
# replaced. The grammar is deliberately narrower than YAML: this workflow is
# two-space-indented block mappings and `- ` step sequences, and a file that
# has grown a flow mapping, an anchor, an alias, a merge key, a quoted key,
# a tab or a second document is not read PERMISSIVELY — it is not read.
#
# **AND "EVERY LINE" IS ONLY AS TOTAL AS THE WORD "LINE".** That word was
# `str.split("\n")` until 2026-08-24, and YAML's own is wider: CR, U+0085,
# U+2028 and U+2029 all break a line for PyYAML. Driven on the real workflow
# with ONE U+2028 in place of the newline between two entries of a step's
# `env:` block, the grammar reported the cell `"unset"` — *"I have read all
# of it and it is not there"* — about a `JAX_ENABLE_X64:` line it had never
# classified, and the whole zero-dep suite was identical to clean. See
# `_lines_of_this_grammar`, which is where the word is defined now.
#
# THREE MORE THINGS IT WILL NOT READ, each of them added because it was
# found rather than foreseen:
#
#   * a **U+0085 / U+2028 / U+2029** anywhere in the file — YAML 1.1 breaks
#     a line there and YAML 1.2 does not, so it is refused rather than
#     guessed about;
#   * a **key written twice in one mapping** — this grammar merged the two
#     and PyYAML keeps the last, which is a reader divergence and not the
#     one direction a fallback is allowed (see `_put`);
#   * a **sequence written at its own key's indentation** —
#     `    steps:` over `    - uses: …` — which is ordinary YAML that GitHub
#     runs. That one is refused only because this grammar measures nesting
#     in columns and that shape does not nest; it is the safe direction, and
#     it is written here rather than left to be discovered by whoever
#     re-indents the workflow.


_LINE_BLANK = re.compile(r"^[ \t]*$")
_LINE_COMMENT = re.compile(r"^[ \t]*#")
_KEY = r"[A-Za-z_][A-Za-z0-9_.-]*"
_LINE_TOP = re.compile(rf"^({_KEY}):[ \t]*(.*?)[ \t]*$")
_LINE_JOB = re.compile(rf"^  ({_JOB_ID}):[ \t]*(.*?)[ \t]*$")
_LINE_JOB_KEY = re.compile(rf"^    ({_KEY}):[ \t]*(.*?)[ \t]*$")
_LINE_SIX = re.compile(rf"^      ({_KEY}):[ \t]*(.*?)[ \t]*$")
_LINE_STEP = re.compile(rf"^      - ({_KEY}):[ \t]*(.*?)[ \t]*$")
_LINE_EIGHT = re.compile(rf"^        ({_KEY}):[ \t]*(.*?)[ \t]*$")
_LINE_TEN = re.compile(rf"^          ({_KEY}):[ \t]*(.*?)[ \t]*$")
_LINE_TEN_ITEM = re.compile(r"^          - [ \t]*(.*?)[ \t]*$")
_BLOCK_SCALAR = ("|", ">", "|-", ">-", "|+", ">+")


class _Refused(Exception):
    """The line grammar met something it will not read."""


#: THE LINE BREAKS YAML'S OWN VERSIONS DISAGREE ABOUT, and the reason this
#: grammar refuses a file that carries one instead of picking a side. YAML
#: 1.1 — which is what PyYAML implements — breaks a line on **U+0085**,
#: **U+2028** and **U+2029** as well as on LF and CR; YAML 1.2 does not, and
#: whether GitHub's own parser does is untested here. A character three
#: readers may split three ways is not a character this grammar can classify
#: lines around, so it is refused in the same breath as a tab.
_LINE_BREAKS_THE_YAML_VERSIONS_DISAGREE_ABOUT = (
    ("\u0085", "U+0085 NEXT LINE"),
    ("\u2028", "U+2028 LINE SEPARATOR"),
    ("\u2029", "U+2029 PARAGRAPH SEPARATOR"),
)


def _lines_of_this_grammar(text: str) -> list:
    """The file's LINES, by YAML's notion of one and not by `str.split`.

    **THIS WAS `text.split("\\n")` AND THE GRAMMAR'S TOTALITY CLAIM RESTED ON
    IT.** A grammar that classifies every line can say *"I have read all of
    it and the setting is not there"*; one that classifies every LF-delimited
    run can only say it about the lines it happened to cut. Driven on
    `.github/workflows/nightly-jax-canary.yml` at `4a13824`, with a single
    U+2028 in place of the newline between two entries of a step's `env:`
    block: the whole zero-dep suite was IDENTICAL to clean, this grammar
    reported the cell ``"unset"`` for a `JAX_ENABLE_X64:` line it had never
    classified — it was swallowed into the value of the entry above it — and
    PyYAML read the cell. That is exactly *"I did not find it"* wearing
    *"it is not there"*, which is the defect the grammar exists to make
    impossible. U+0085 and U+2029 behave the same way, and so does CR.

    So there are two answers here and they are different answers:

    * **CR and CRLF are NORMALISED.** A carriage return is a line break in
      YAML 1.1 and 1.2 alike, so treating it as one is reading the spec and
      not guessing. It mattered: CR was live on an in-memory string, and
      through the file path it was neutralised only by ``read_text``'s
      universal-newline translation — an accident of how the caller loaded
      the file rather than a property of this grammar. It is a property of
      this grammar now.
    * **U+0085, U+2028 and U+2029 are REFUSED**, because there the answer is
      version-dependent — see
      `_LINE_BREAKS_THE_YAML_VERSIONS_DISAGREE_ABOUT`. Splitting on them
      would make this grammar agree with PyYAML and possibly disagree with
      the runner; not splitting on them is what was measured above. A
      refusal is the third answer and the only honest one.

    THE ORDER MATTERS AND IS THE POINT OF DOING THIS FIRST. Every later check
    reads the lines this returns, so the tab scan and the document-marker
    scan see a CR-introduced line too. `re.search(r"^---", text, re.M)` did
    not: Python's `^` under `re.M` matches after a newline and not after a
    carriage return, so a second document opened by a CR was invisible to the
    check written to refuse it.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for char, name in _LINE_BREAKS_THE_YAML_VERSIONS_DISAGREE_ABOUT:
        at = text.find(char)
        if at >= 0:
            raise _Refused(
                f"line {text.count(chr(10), 0, at) + 1}: this file carries a "
                f"{name}, which YAML 1.1 breaks a line on and YAML 1.2 does "
                f"not. This grammar will not guess which of them the runner's "
                f"parser is, and a character it guessed wrong about would "
                f"hide a whole `env:` entry inside the value of the one above "
                f"it — measured, at `4a13824`, reported as the cell 'unset'"
            )
    return text.split("\n")


def _put(mapping, key, value, where: str, what: str):
    """Set ONE entry of a mapping, refusing a key that is already there.

    **DUPLICATE KEYS MERGED HERE AND REPLACE IN PyYAML, AND THAT FALSIFIED
    THE TABLE'S STATED INVARIANT.** `jobs`, `env` at all three levels and
    `strategy.matrix` were written with `setdefault`, so a second `env:` on
    one step was READ — its entries folded into the first block — where
    PyYAML keeps the last mapping and drops the first outright. Driven on
    the real workflow at `4a13824`, a second `env:` carrying only
    `JAX_PLATFORMS` on the nightly x64-OFF canary step: this grammar read
    `['1', '0']` and PyYAML read `['1', 'unset']`, the zero-dep suite stayed
    at `114 passed`, and the two-reader agreement test was the only thing
    that moved. `_RESOLUTIONS` said *"where the columns differ it is always
    the same way round — the parser RESOLVES a legal spelling and the line
    grammar REFUSES it"*; here BOTH resolved, to different cells.

    So a repeated key is refused rather than merged, which puts the
    divergence back in the one direction a fallback may have. What GitHub's
    own parser does with a duplicate key is NOT measured here — probably a
    rejection, which is a third answer again and one more reason not to pick
    one.
    """
    if key in mapping:
        raise _Refused(
            f"{where}: `{key}` is written twice in {what}. PyYAML keeps the "
            f"last one and drops the first; merging them is what this grammar "
            f"used to do and it read a different cell from the parser for it; "
            f"what the runner's parser does is untested. Three readers and up "
            f"to three answers, so this one refuses"
        )
    mapping[key] = value
    return value


def _scalar(raw: str, where: str):
    """A written scalar -> a Python value, or a refusal.

    The refusals are the point. An anchor (`&x`), an alias (`*x`), a merge
    key (`<<:`) and a flow mapping (`{a: b}`) are legal YAML this grammar
    does not resolve; the alias and the flow mapping were each a MEASURED
    way past its predecessor, at `95 passed`. A block scalar is refused here
    because only `run:` may open one, and a flow SEQUENCE reaching this
    function is a sequence somewhere a sequence does not belong — the one
    place this workflow may carry one, a `strategy.matrix` axis, reads it
    before it gets here and passes the ENTRIES down.
    """
    if raw == "":
        return None
    if raw[0] in "&*<":
        raise _Refused(
            f"{where} is `{raw}`: this reader does not resolve YAML anchors, "
            f"aliases or merge keys, and an alias to an anchored `env:` "
            f"mapping was a measured way past its predecessor"
        )
    if raw[0] in "{[":
        raise _Refused(
            f"{where} is the flow collection `{raw}`, which this reader does "
            f"not parse — `env: {{ JAX_ENABLE_X64: \"1\" }}` was a measured "
            f"way past its predecessor"
        )
    if raw in _BLOCK_SCALAR:
        raise _Refused(f"{where} opens a block scalar, which only `run:` may")
    if "#" in raw and not (raw[0] in "\"'" and raw[-1] == raw[0]):
        raise _Refused(
            f"{where} is `{raw}`, which carries a `#` this reader will not "
            f"guess is or is not a comment"
        )
    if raw[0] in "\"'":
        if len(raw) < 2 or raw[-1] != raw[0] or raw[0] in raw[1:-1]:
            raise _Refused(f"{where} is `{raw}`, which this reader cannot unquote")
        return raw[1:-1]
    # ...and an UNQUOTED scalar gets YAML 1.1's own resolution, because that
    # is what the other reader does and the two have to land on one value.
    # `on`/`off`/`yes`/`no` are booleans here exactly as PyYAML makes them,
    # which is also why `JAX_ENABLE_X64: on` reads ON in both: the runner
    # renders the boolean as `"true"` and jax reads `"true"` as ON.
    lowered = raw.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "~"):
        return None
    if re.fullmatch(r"[+-]?[0-9]+", raw):
        return int(raw)
    return raw


def _scalar_must_nest(raw: str, what: str, where: str) -> None:
    """A key that opens a mapping may not also carry a value."""
    if raw:
        raise _Refused(
            f"{where}: {what} carries the value `{raw}` on its own line "
            f"instead of opening a mapping under it — a flow mapping and an "
            f"alias are both spelled that way and this reader resolves "
            f"neither"
        )


def _parse_by_line(text: str):
    """The workflow's text -> the mapping PyYAML would build, or `_Refused`.

    Total over the lines it reads: the top-level keys, and everything inside
    `jobs:`. The bodies of `on:`, `permissions:` and `concurrency:` are
    skipped by indentation because nothing in them can reach a step's
    environment except through an alias, and an alias is refused where it is
    USED. "Line" is `_lines_of_this_grammar`'s and not `str.split`'s — see
    there for the U+2028 that produced a cell out of a line nobody read.

    AND EVERY WAY OUT OF HERE IS A RETURN OR A `_Refused`. It was not: an
    indent-4 job key with no job header above it reached `job[...] = ...`
    with `job` still `None` and raised a bare ``TypeError`` through
    `_read_workflow`, which catches `_Refused` alone — a THIRD outcome
    beside "classify" and "refuse", with no line number, no reader and no
    refusal sentence. Driven by commenting out `  control:` on the real
    workflow at `4a13824`: four tests failed on
    ``TypeError: 'NoneType' object does not support item assignment``.
    `test_the_line_grammar_HAS_NO_THIRD_OUTCOME` drives every one-line
    deletion and every one-line comment-out of the real workflow through
    here and holds that to two outcomes.
    """
    # THE LINES FIRST, BECAUSE EVERY CHECK BELOW READS THEM. See
    # `_lines_of_this_grammar` for why CR is normalised and U+0085 / U+2028 /
    # U+2029 are refused, and for the measured `"unset"` that came of
    # splitting on `\n` alone.
    lines = _lines_of_this_grammar(text)

    # ...AND BOTH FILE-LEVEL REFUSALS NAME A LINE NOW. Of the refusals this
    # grammar can raise these two were the only ones that said neither WHERE
    # nor WHAT: "this file contains a tab" over a 260-line workflow is a
    # sentence somebody has to go and grep for.
    for number, line in enumerate(lines, 1):
        if "\t" in line:
            raise _Refused(
                f"line {number}: {line!r} contains a tab; YAML forbids one in "
                f"indentation and this grammar measures indentation in "
                f"spaces, so it will not guess which kind this is"
            )
        if line.startswith("---") or line.startswith("..."):
            raise _Refused(
                f"line {number}: {line!r} is a document marker; a "
                f"multi-document workflow is outside this grammar"
            )

    workflow: dict = {}
    jobs: dict = {}
    job = env = step = matrix = axis = None
    state = "top"
    skip_below = None
    run_key_indent = None
    run_lines: list = []
    run_owner = None

    for number, line in enumerate(lines, 1):
        where = f"line {number}"
        indent = len(line) - len(line.lstrip(" "))

        if run_owner is not None:
            if _LINE_BLANK.match(line) or indent > run_key_indent:
                run_lines.append(line)
                continue
            run_owner["run"] = "\n".join(run_lines)
            run_owner, run_lines, run_key_indent = None, [], None

        if _LINE_BLANK.match(line) or _LINE_COMMENT.match(line):
            continue
        if skip_below is not None:
            if indent > skip_below:
                continue
            skip_below = None

        top = _LINE_TOP.match(line)
        if top is not None:
            key, raw = top.group(1), top.group(2)
            state = "top"
            job = env = step = matrix = axis = None
            if key in _WORKFLOW_KEYS_WITH_NOTHING_IN_THEM_FOR_US:
                _put(workflow, key, _Opaque(), where, "this workflow")
                skip_below = 0
                continue
            if key in ("jobs", "env"):
                # Same rule as the job and step levels: a key that opens a
                # mapping may not also carry a value.
                _scalar_must_nest(raw, f"the workflow's `{key}` key", where)
            if key == "jobs":
                _put(workflow, "jobs", jobs, where, "this workflow")
                state = "jobs"
                continue
            if key == "env":
                env = _put(workflow, "env", {}, where, "this workflow")
                state = "workflow-env"
                continue
            _put(workflow, key, _Opaque(), where, "this workflow")
            skip_below = 0
            continue

        if state == "workflow-env":
            entry = _LINE_TOP.match(line[2:]) if indent == 2 else None
            if entry is None or indent != 2:
                raise _Refused(f"{where}: {line!r} is not an `env:` entry")
            _put(env, entry.group(1),
                 _scalar(entry.group(2),
                         f"the workflow's `{entry.group(1)}`"),
                 where, "the workflow's `env:`")
            continue

        if state not in ("jobs", "job", "job-env", "steps", "step", "step-env",
                         "strategy", "matrix", "axis"):
            raise _Refused(f"{where}: {line!r} is outside this grammar")

        head = _LINE_JOB.match(line)
        if head is not None and indent == 2:
            if head.group(2):
                raise _Refused(
                    f"{where}: the job `{head.group(1)}` has a value on its "
                    f"own line, which this reader does not read"
                )
            job = _put(jobs, head.group(1), {}, where, "`jobs:`")
            env = step = matrix = axis = None
            state = "job"
            continue

        if indent == 4:
            key_line = _LINE_JOB_KEY.match(line)
            if key_line is None:
                raise _Refused(f"{where}: {line!r} is not a job key")
            key, raw = key_line.group(1), key_line.group(2)
            if job is None:
                # A JOB KEY BEFORE ANY JOB, which is a THIRD outcome and not
                # a refusal: `job.setdefault(...)` below raised a bare
                # `TypeError: 'NoneType' object does not support item
                # assignment` out of `_read_workflow`, which catches
                # `_Refused` alone. Driven by commenting out `  control:` on
                # the real workflow at `4a13824`: four tests failed with no
                # line number, no reader and no refusal sentence.
                # `_parse_with_yaml` guards its analogue; this is the same
                # guard, and it is the same shape as "a step key before any
                # step" eighty lines down.
                raise _Refused(
                    f"{where}: {line!r} is a job key with no job header above "
                    f"it — a two-space `<job-id>:` line under `jobs:`"
                )
            env = step = matrix = axis = None
            if key in ("steps", "env", "strategy"):
                # A NESTING KEY MAY NOT CARRY A VALUE. `env: *x64on` and
                # `env: { JAX_ENABLE_X64: "1" }` are two of the nine measured
                # ways past the pattern this grammar replaced, and both are a
                # nesting key with something on its own line.
                _scalar_must_nest(raw, f"the `{key}` key of this job", where)
            what = "this job"
            if key == "steps":
                _put(job, "steps", [], where, what)
                state = "steps"
                continue
            if key == "env":
                env = _put(job, "env", {}, where, what)
                state = "job-env"
                continue
            if key == "strategy":
                _put(job, "strategy", {}, where, what)
                state = "strategy"
                continue
            _put(job, key, _scalar(raw, f"the `{key}` key of this job"),
                 where, what)
            state = "job"
            continue

        if state == "job-env" and indent == 6:
            entry = _LINE_SIX.match(line)
            if entry is None:
                raise _Refused(f"{where}: {line!r} is not an `env:` entry")
            _put(env, entry.group(1),
                 _scalar(entry.group(2), f"this job's `{entry.group(1)}`"),
                 where, "this job's `env:`")
            continue

        if state in ("strategy", "matrix", "axis"):
            if indent == 6:
                entry = _LINE_SIX.match(line)
                if entry is None:
                    raise _Refused(f"{where}: {line!r} is not a `strategy:` key")
                if entry.group(1) == "matrix":
                    if entry.group(2):
                        raise _Refused(f"{where}: an inline `matrix:` value")
                    matrix = _put(job["strategy"], "matrix", {}, where,
                                  "this job's `strategy:`")
                    state = "matrix"
                    continue
                _put(job["strategy"], entry.group(1),
                     _scalar(entry.group(2),
                             f"this job's `strategy.{entry.group(1)}`"),
                     where, "this job's `strategy:`")
                state = "strategy"
                continue
            if indent == 8 and state in ("matrix", "axis"):
                entry = _LINE_EIGHT.match(line)
                if entry is None:
                    raise _Refused(f"{where}: {line!r} is not a matrix axis")
                axis, raw = entry.group(1), entry.group(2)
                if raw.startswith("["):
                    # A FLOW SEQUENCE, SPLIT ON COMMAS. This is the one place
                    # the grammar splits rather than parses, and it is wrong
                    # for an entry containing a quoted comma — which would
                    # give two entries where YAML gives one, and so an extra
                    # cell. The axis values this reader resolves are `"0"` and
                    # `"1"`; anything a comma could hide inside is not a value
                    # `_x64_word` reads, so the extra entry is a can't-tell
                    # and not a cell. Said here rather than left to be found.
                    if not raw.endswith("]"):
                        raise _Refused(f"{where}: an unterminated flow sequence")
                    _put(matrix, axis, [
                        _scalar(v.strip(), f"an entry of matrix axis `{axis}`")
                        for v in raw[1:-1].split(",") if v.strip()
                    ], where, "this job's `strategy.matrix:`")
                    state = "matrix"
                    continue
                if raw:
                    _put(matrix, axis, _scalar(raw, f"matrix axis `{axis}`"),
                         where, "this job's `strategy.matrix:`")
                    state = "matrix"
                    continue
                _put(matrix, axis, [], where, "this job's `strategy.matrix:`")
                state = "axis"
                continue
            if indent == 10 and state == "axis":
                item = _LINE_TEN_ITEM.match(line)
                if item is None:
                    raise _Refused(f"{where}: {line!r} is not a matrix entry")
                matrix[axis].append(
                    _scalar(item.group(1), f"an entry of matrix axis `{axis}`"))
                continue
            raise _Refused(f"{where}: {line!r} is not part of a `strategy:` block")

        if state in ("steps", "step", "step-env"):
            start = _LINE_STEP.match(line)
            if start is not None:
                step = {}
                job["steps"].append(step)
                env = None
                state = "step"
                key, raw = start.group(1), start.group(2)
                if key == "env" or raw in _BLOCK_SCALAR and key != "run":
                    raise _Refused(
                        f"{where}: this reader does not open a step on `{key}:`"
                    )
                if key == "run" and raw in _BLOCK_SCALAR:
                    run_owner, run_lines, run_key_indent = step, [], 8
                    continue
                _put(step, key, _scalar(raw, f"a step's `{key}`"),
                     where, "this step")
                continue
            if indent == 8:
                entry = _LINE_EIGHT.match(line)
                if entry is None:
                    raise _Refused(f"{where}: {line!r} is not a step key")
                key, raw = entry.group(1), entry.group(2)
                if step is None:
                    raise _Refused(f"{where}: a step key before any step")
                if key == "env":
                    _scalar_must_nest(raw, "a step's `env`", where)
                    env = _put(step, "env", {}, where, "this step")
                    state = "step-env"
                    continue
                if key == "run" and raw in _BLOCK_SCALAR:
                    # A BLOCK SCALAR IS SET WHEN IT CLOSES, so the duplicate
                    # check happens at the point the key is OPENED — by then
                    # a first `run:` has already been flushed into the step.
                    if "run" in step:
                        raise _Refused(
                            f"{where}: `run` is written twice in this step; "
                            f"see `_put` for why a repeated key is refused "
                            f"rather than merged or replaced"
                        )
                    run_owner, run_lines, run_key_indent = step, [], 8
                    state = "step"
                    continue
                _put(step, key, _scalar(raw, f"a step's `{key}`"),
                     where, "this step")
                state = "step"
                continue
            if state == "step-env" and indent == 10:
                entry = _LINE_TEN.match(line)
                if entry is None:
                    raise _Refused(f"{where}: {line!r} is not an `env:` entry")
                _put(env, entry.group(1),
                     _scalar(entry.group(2), f"a step's `{entry.group(1)}`"),
                     where, "this step's `env:`")
                continue
            raise _Refused(f"{where}: {line!r} is not part of a `steps:` block")

        raise _Refused(f"{where}: {line!r} is outside this grammar")

    if run_owner is not None:
        run_owner["run"] = "\n".join(run_lines)
    return workflow


# --------------------------------------------------------------------------
# THE ONE ENTRY POINT
# --------------------------------------------------------------------------


def _read_workflow(text: str, parser: str | None = None) -> _Workflow:
    """The workflow's text -> the model, through whichever reader is asked for.

    ``parser=None`` means *the strongest one this lane has*, which is what
    the guards use. The tests drive both by name, so the line grammar is
    exercised in the jax lane too and is never only as good as the lane that
    cannot check it.
    """
    if parser is None:
        parser = "yaml" if _YAML_IS_IMPORTABLE else "text"
    if parser == "yaml":
        return _model(_parse_with_yaml(text), "yaml")
    try:
        raw = _parse_by_line(text)
    except _Refused as exc:
        return _Workflow(parser="text", blockers=(
            f"this lane has no YAML parser and the line grammar stopped: "
            f"{exc}",))
    return _model(raw, "text")


def _canary_jobs(workflow):
    """``(job id, job)`` for every job of the nightly workflow, in order.

    Takes the workflow's TEXT or an already-read `_Workflow`. Every caller
    below FAILS on an empty list rather than passing over one — two by
    asserting it directly and the third because the cells it collects come
    from these jobs, so no jobs means no cells and no cells means red.

    THE JOBS COME OFF A PARSE NOW. This read
    ``\\n  (job):\\n    name: `` — a two-space key whose FIRST CHILD is
    ``name:`` — and YAML mapping keys are unordered, so a third job written
    with ``runs-on:`` first was invisible: driven on the real workflow, a
    `nightly_x64:` leg running the canary at `JAX_ENABLE_X64: "1"` only left
    this file at `95 passed` with `jobs seen: ['control', 'nightly']`.
    """
    if isinstance(workflow, str):
        workflow = _read_workflow(workflow)
    return [(job.id, job) for job in workflow.jobs]


def _matrix_axis(job: _Job, axis: str):
    """The values of ``strategy.matrix.<axis>``, or ``None`` if unreadable.

    ``include:`` and ``exclude:`` are NOT axes and their presence makes the
    whole expansion unreadable rather than being ignored: driven, a
    ``matrix: {x64: ["0", "1"], exclude: [{x64: "0"}]}`` over
    ``${{ matrix.x64 }}`` left this file at `95 passed` while the job runs in
    the ON cell only.
    """
    if not isinstance(job.matrix, dict):
        return None
    for key in job.matrix:
        if _key_name(key) in ("include", "exclude"):
            return None
    values = job.matrix.get(axis)
    if not isinstance(values, list) or not values:
        return None
    return values


def _x64_resolve(raw, job: _Job) -> list:
    """A written setting -> the cell(s) the steps under it actually run in.

    A `${{ matrix.<axis> }}` setting is EXPANDED, entry by entry, because a
    matrix multiplies the job and one step over a two-cell axis genuinely
    runs in both. That is the honest refactor of this workflow -- one pytest
    step, `matrix: x64: ["0", "1"]` -- and forbidding expressions outright
    rejected it while PASSING `${{ matrix.x64 }}` over `x64: ["1"]`, which
    runs in one cell. The guard was strict against the sound shape and
    permissive against the unsound one, which is the wrong way round twice.
    """
    if not isinstance(raw, str) or "${{" not in raw:
        return [_x64_word(raw)]
    ref = _X64_MATRIX_REF.fullmatch(raw.strip())
    if ref is None:
        return [_cannot_tell(
            f"JAX_ENABLE_X64: {raw!r} is an expression this reader does "
            f"not follow; only `${{{{ matrix.<axis> }}}}` is followed"
        )]
    values = _matrix_axis(job, ref.group(1))
    if not values:
        return [_cannot_tell(
            f"JAX_ENABLE_X64: {raw!r} names matrix axis {ref.group(1)!r} and "
            f"this job's `strategy.matrix` does not carry it in a spelling "
            f"this reader can expand"
        )]
    return [_x64_word(value) for value in values]


def _x64_cells(workflow, job, wanted) -> list:
    """The ``JAX_ENABLE_X64`` cell each step of ``job`` that ``wanted`` runs in.

    Every entry is ``"0"``, ``"1"``, ``"unset"`` or a ``?``-prefixed
    can't-tell, and **``"unset"`` and the can't-tell are different things** —
    that they were one string is the defect this docstring's second half is
    about. ``"unset"`` is *no setting anywhere*, which is jax's default and
    measured OFF (`jax.config.jax_enable_x64` is `False` with the variable
    unset); a can't-tell is *a setting this reader will not guess at*, and
    :func:`_refuse_unreadable` turns it into a red. Callers must call that
    before reading a cell.

    **AND `"unset"` IS NOW A THING ONLY A PARSE MAY SAY.** A pattern that
    searches can report a miss and nothing more, so the old reader's
    ``"unset"`` meant *I did not find one* — which is why `env: *x64on` and
    `"JAX_ENABLE_X64": "1"` both read as OFF. Either reader here has read
    the WHOLE file before it says the word: PyYAML by parsing it, the line
    grammar by classifying every line and refusing the reading outright on
    the first line it cannot place.

    RESOLVE THE SETTING THE WAY THE RUNNER DOES: a step's own ``env:`` wins,
    else the job's, else the workflow's top-level one. Reading only the step
    and calling a miss "unset" is how the sweep guard first went blind:
    hoisting the repeated ``env:`` up to the job -- a tidy-up this workflow
    openly invites, six steps repeat it today -- left every canary run at
    x64=1 with the guard still green.

    AND FOUR THINGS THAT ARE NOT AN ``env:`` ENTRY AT ALL SET THIS VARIABLE
    OR TAKE THE STEP AWAY, each of which this reader refuses rather than
    misses:

    * the step's ``run:`` script. Not "does it name the variable" — a
      substring test for the name answered `no` to `set -a; . ./ci.env;
      set +a`. Every command word must be one `_SHELL_WORDS_THIS_READER_
      KNOWS` names;
    * any step of the job naming ``$GITHUB_ENV``, which changes the
      environment of every LATER step, or ``uses:`` an action outside
      `_ACTIONS_THIS_READER_KNOWS`, which can do the same thing from inside
      somebody else's repository;
    * an ``if:`` this reader cannot evaluate, or one it can whose referent
      does not exist — see `_STEP_CONDITIONS_THIS_READER_KNOWS`;
    * ``continue-on-error: true``, which does not change the cell but takes
      away the step's ability to fail the job, and both callers here are
      claims about a step that CAN go red. Driven: `continue-on-error: true`
      on the real workflow's x64 OFF canary step left this file at
      `95 passed`.

    **IT IS ONE FUNCTION BECAUSE TWO CALLERS NEED THE SAME RULE**, and the
    second caller is why: the guard on the tripwire's own test step was added
    without this resolution and read no cell at all. A resolution rule this
    subtle, typed twice, is the shape
    `.github/workflows/nightly-jax-canary.yml` has already drifted on twice
    in prose. It also carried the conflation above into BOTH callers the day
    it was extracted, which is the risk a shared resolver runs and the reason
    every shape it refuses is named in `_refuse_unreadable`.
    """
    if isinstance(workflow, str):
        workflow = _read_workflow(workflow)
    if isinstance(job, str):
        jobs = dict(_canary_jobs(workflow))
        if job not in jobs and workflow.blockers:
            # A REFUSED READING HAS NO JOBS, and the reason it has none is
            # the answer. Returning the workflow's own can't-tells here is
            # what keeps `_refuse_unreadable` the one gate every caller
            # passes through, rather than some callers meeting a KeyError.
            return [_cannot_tell(reason) for reason in workflow.blockers]
        job = jobs[job]
    job = _resolve_conditions(job)

    poison = [_cannot_tell(reason) for reason in workflow.blockers]
    poison += [_cannot_tell(reason) for reason in job.blockers]
    for step in job.steps:
        poison += [_cannot_tell(reason) for reason in step.job_blockers]

    cells: list = []
    for step in job.steps:
        if not wanted(step):
            continue
        if poison or step.blockers:
            cells += poison + [_cannot_tell(r) for r in step.blockers]
            continue
        for source in (step.env, job.env, workflow.env):
            if isinstance(source, dict) and "JAX_ENABLE_X64" in source:
                cells.extend(_x64_resolve(source["JAX_ENABLE_X64"], job))
                break
        else:
            cells.append("unset")
    return cells


def test_the_canary_and_the_workflow_agree_about_the_two_legs():
    """The guidance a red run prints is a claim about a workflow. Check it.

    ONE CONSTANT, TWO MESSAGES. This paragraph used to be typed out twice,
    190 lines apart, and the two copies said different and both-wrong things:
    that the control leg runs a PINNED series (it installs ``.[jax]``, so it
    resolves to whatever is newest released) and that "the jax versions on
    the two pages tell apart" a released-jax regression from a broken
    stelling (they do not -- both legs run this repository's code).

    WHAT IS CHECKED IS THE FACT, NOT THE WORDING. The workflow is read and
    the install step of each leg is compared against what the guidance says
    that leg installs. The one lexical clause -- that neither leg may be
    described as pinned -- is CONDITIONAL on the measurement: it is in force
    only while the install steps carry no version constraint, and the day
    somebody pins a leg it stops applying on its own. It is here because this
    exact claim has been wrong in five places across three commits, and
    because a reader who believes it goes looking for a jax regression that
    the run cannot have distinguished.

    AND THAT THE LEGS STILL RUN THE ALARM AT ALL, which is a different kind
    of claim and the one this test was missing. Two mutations were green
    across this whole repository -- which is not a large claim, because this
    file is the ONLY test that reads that workflow at all: the `nightly` leg
    running the canary WITHOUT `--require`, and the `nightly` leg with the
    canary step deleted.
    Guidance about what two legs establish is worth nothing on a leg that
    measures nothing, so each job is checked for running
    `tripwire_canary.py --require`, and the file for the `needs:` the
    `control` job's comment says is deliberately absent.
    """
    import re

    canary = _canary()
    workflow = (
        _pathlib_for_canary() / ".github" / "workflows" / "nightly-jax-canary.yml"
    ).read_text(encoding="utf-8")
    read = _readable_workflow(workflow, "the guidance about the two legs")

    # ONE READER FOR JOBS, `_canary_jobs`. This line carried a SECOND copy of
    # the job-id pattern until 2026-08-22 and the copy was the same too-narrow
    # `[a-z-]+`, so a third job `nightly_x64:` was invisible here as well as
    # there and this assertion passed on a workflow with three legs. The
    # titles come off the jobs that reader already returns, which since
    # 2026-08-23 is a PARSE and no longer a pattern needing `name:` first.
    jobs = {job.id: job.name for _, job in _canary_jobs(read)}
    assert set(jobs) == {"control", "nightly"}, (
        f"the guidance names two legs and the workflow has jobs {sorted(jobs)}"
    )

    installs = re.findall(r"uv pip install[^\n]*(?:\n[^\n]*\\\n[^\n]*)*", workflow)
    control_installs = [i for i in installs if '".[jax]"' in i]
    assert control_installs, (
        "the `control` leg no longer installs `.[jax]`, which is the fact the "
        "guidance rests on"
    )
    assert "us-python.pkg.dev" in workflow, (
        "the `nightly` leg no longer installs from a nightly index"
    )

    # EACH LEG RUNS THE ALARM, AND RUNS IT WITH `--require`. Two mutations
    # survived every test in this repository: dropping `--require` from the
    # `nightly` leg, and deleting the `tripwire_canary.py` step from that leg
    # altogether. Both leave a workflow whose two jobs still install two
    # different jaxes and still describe themselves correctly, and neither
    # leaves anything that can go red -- the alarm can be removed from the
    # only workflow that runs it without a single test noticing. The guidance
    # is a claim about what the two legs MEASURE, and a leg that does not arm
    # the tripwire measures nothing to compare.
    jobs_and_blocks = _canary_jobs(read)
    assert jobs_and_blocks, (
        "no jobs found; the workflow layout this reads has moved"
    )
    for job, body in jobs_and_blocks:
        # THE STEP'S OWN `run:` SCRIPT, off the parse: this file is YAML and
        # a `#` turns the step into prose that still contains every word
        # below. Commenting the step out is the same mutation as deleting it,
        # and a parser sees that where a text scan of the block did not.
        runs = [step.run for step in body.steps
                if isinstance(step.run, str)
                and "tripwire_canary.py" in step.run]
        assert runs, (
            f"the `{job}` leg does not RUN `.github/scripts/"
            "tripwire_canary.py` -- deleted, or commented out, or moved to a "
            "line that is not a `run:`. Nothing on that leg can then fail for "
            "any of the reasons the canary exists to name"
        )
        assert all("--require" in line for line in runs), (
            f"the `{job}` leg runs the canary WITHOUT `--require`, so a "
            "tripwire that cannot arm against that leg's jax exits 0 and the "
            "leg goes green with the alarm switched off -- `auto` is right "
            f"for a user's suite and wrong for the canary: {runs}"
        )
        # AND NEITHER LEG SWITCHES OFF THE CONSTANT SWEEP. `--no-sweep` is
        # default-off and exists for this repository's own exit-code battery,
        # which drives the script a dozen times in a subprocess and would
        # otherwise pay twelve times for one measurement. On a WORKFLOW leg it
        # would silently retire the earliest signal there is that jax has
        # grown an eager truncation nobody has written a row for.
        assert all("--no-sweep" not in line for line in runs), (
            f"the `{job}` leg passes `--no-sweep`, which turns off the "
            "re-derivation of `_JAX_EAGER_CONSTANTS` on the only job that "
            f"meets a new jax release before any CI lane does: {runs}"
        )

    # AND NEITHER LEG WAITS FOR THE OTHER. The comment over the `control` job
    # says the two run concurrently and argues that sequencing them would be
    # worse, because a `needs:` makes GitHub skip the dependent leg. That
    # comment claimed the opposite ordering for one commit with no `needs:`
    # in the file to make it true, which is why the fact is measured here.
    assert not re.search(r"^\s*needs:", workflow, re.M), (
        "a job now `needs:` another and the `control` job's comment says the "
        "two legs run concurrently; one of the two has to change"
    )

    # THE MEASUREMENT: does either leg constrain the jax version?
    constrained = re.findall(r"jax[a-z]*\s*(?:[=<>~!]=|[<>])\s*[0-9]", workflow)
    assert not constrained, (
        f"a leg now constrains its jax version ({constrained}); the guidance "
        "in `tripwire_canary.py` and the job names in this workflow both say "
        "no leg does, and one of the three is now wrong"
    )

    # ...so nothing may say one is
    for where, text in (
        ("_TWO_LEGS", canary._TWO_LEGS),
        ("the job names", " ".join(jobs.values())),
    ):
        claims = [
            m for m in re.finditer(r"\bpinn?ed\b", text, re.I)
            if not re.search(r"\b(neither|not|no)\b[^.]{0,40}$", text[:m.start()], re.I)
        ]
        assert not claims, (
            f"{where} describes a leg as pinned and no leg pins its jax "
            f"version: {[m.group(0) for m in claims]}"
        )


def test_the_nightly_runs_the_tests_OF_THE_HOOK_ON_A_PRIVATE_FUNCTION():
    """The eager hook is the exposed one, and it was the one the nightly did
    not test.

    Both hooks attach to jax internals and the canary script drives both, so
    both are covered for ATTACHING. What the tripwire's own test files add is
    that a hook still FINDS, ATTRIBUTES and SUPPRESSES the right things — and
    until 2026-08-22 the nightly ran those for the const-fold hook (`arm`,
    `record`, `plugin`) and not for the eager one.

    THE ASYMMETRY IS THE ARGUMENT. The const-fold hook installs an entry in a
    private REGISTRY; the eager hook patches a module attribute on
    a private jax MODULE's function — with a signature and a set of callers,
    named by path in `design/private-jax-boundary.md` and reachable only
    through `_adapter_jax.py`. A release that reorders
    `_convert_element_type`'s parameters, or routes one construction spelling
    around it, changes what the hook sees while leaving it attached, and
    `arm_eager()`'s own self-check is the only other thing that would notice.

    IT COULD NOT BE ADDED BEFORE THE CELL WAS GREEN. The step runs at
    `JAX_ENABLE_X64: "1"`, and `tests/test_tripwire_eager.py` had EIGHT
    failures there at `844ba48` — every one of them a subject that cannot be
    present in that cell — so adding the file would have made the leg
    permanently red.

    **AND THE CELL IS THE OTHER HALF, WHICH THIS TEST DID NOT CHECK.** "115
    passed in both cells" is what licensed adding the step, and it says the
    file is GREEN in both cells — not that it WATCHES in both. It does not:
    of the eight tests converted to complements, the two that carry the
    eager detector's suppression SITING put it behind `if
    jax.config.jax_enable_x64: … return`, so the siting is asserted at x64
    OFF only. Driven, with `observe()` still counting `SUPPRESSED_JAX` but
    never writing the `SUPPRESSED` row:

        x64=0    2 failed, 113 passed
        x64=1    115 passed            <- the neutered detector is invisible

    Run at `JAX_ENABLE_X64: "1"` alone, the step added to close an
    unwatched hook was itself an instrument that could not fire — and
    nothing guarded the cell: flipping that step's setting to `"0"` left
    this file at `68 passed`, with
    `::test_each_canary_leg_runs_the_sweep_IN_THE_CELL_IT_CAN_REDDEN_IN`
    standing one function below as the precedent for exactly this check.
    So BOTH cells are required here, resolved through `_x64_cells`, the
    same rule that guard uses.

    NOT EVERY `test_tripwire_*.py` IS ON THAT STEP, and this does not ask for
    them: `test_tripwire_xdist.py` needs `pytest-xdist`, which the nightly
    does not install, and the two gate files drive the 35-route inventory
    twice over. What is checked is that the file naming the hook on a private
    FUNCTION is there, that nothing on the step names a file that is not, and
    that the file is run in both cells.
    """
    root = _pathlib_for_canary()
    workflow = _readable_workflow(
        (root / ".github" / "workflows"
         / "nightly-jax-canary.yml").read_text(encoding="utf-8"),
        "the claim that the nightly runs the hook's own tests",
    )

    # THE JOB THIS TEST NAMES, AND ONLY IT. This read the WHOLE workflow and
    # the cells below were collected from EVERY job into one flat list tied to
    # none of them, while the name and the paragraph above both say *the
    # nightly*. Driven: moving the x64 OFF pytest step bodily into the
    # `control` job left this file at `68 passed` — the union still held one
    # of each cell, and the nightly ran the tripwire's own tests at x64=1
    # alone, which is exactly the state this test exists to forbid.
    blocks = dict(_canary_jobs(workflow))
    assert "nightly" in blocks, (
        f"the workflow has no `nightly` job; its jobs are {sorted(blocks)}, "
        f"and this test is a claim about that leg by name"
    )
    nightly = blocks["nightly"]

    # A STEP'S `run:` SCRIPT AND NOTHING ELSE. This used to split the job's
    # TEXT on `- ` and then look for `-m pytest` anywhere in the chunk, so a
    # step that merely mentioned pytest in its comment preamble was one. The
    # parse carries the script itself and the comments are gone with it.
    steps = [step for step in nightly.steps
             if isinstance(step.run, str) and "-m pytest" in step.run]
    assert steps, (
        "no step of the `nightly` job runs pytest at all, so the "
        "tripwire's own tests are not run against the nightly by anything"
    )
    named = {
        name for step in steps
        for name in re.findall(r"tests/(test_[a-z0-9_]+\.py)", step.run)
    }
    assert "test_tripwire_eager.py" in named, (
        f"the nightly runs the tripwire's own tests but not "
        f"`tests/test_tripwire_eager.py`, whose hook is the one attached to a "
        f"private jax FUNCTION rather than a private registry — the thing a "
        f"nightly is most likely to break. Named: {sorted(named)}"
    )
    missing = [name for name in named if not (root / "tests" / name).is_file()]
    assert not missing, (
        f"the nightly step names test file(s) that are not in the tree: "
        f"{sorted(missing)} — that leg runs nothing for them and pytest exits "
        f"4 rather than saying so"
    )

    # ...AND IN BOTH CELLS. See the paragraph above for the measurement; the
    # short of it is that the file's suppression-siting assertions sit behind
    # an x64 branch, so a step at x64=1 alone runs the tests and watches
    # nothing that the eager hook actually does.
    cells = _x64_cells(
        workflow,
        nightly,
        lambda step: (
            isinstance(step.run, str)
            and "-m pytest" in step.run
            and "test_tripwire_eager.py" in step.run
        ),
    )
    # A CAN'T-TELL IS NOT A CELL. See `_refuse_unreadable` for the eight
    # measured shapes that ran at x64 ON while the conflated `"unset"` read
    # as OFF here.
    _refuse_unreadable(
        cells, "the `nightly` job's `tests/test_tripwire_eager.py` steps"
    )
    off = [cell for cell in cells if cell in ("0", "unset")]
    on = [cell for cell in cells if cell == "1"]
    assert off and on, (
        f"`tests/test_tripwire_eager.py` is run on the nightly at "
        f"JAX_ENABLE_X64={cells}, and it has to be run in BOTH cells. At x64 "
        f"OFF is where its suppression-siting assertions live — with the "
        f"detector's siting neutered that cell is `2 failed, 113 passed` and "
        f"x64 ON is `115 passed`, so a step in the ON cell alone runs 115 "
        f"tests and watches none of the finding, attributing or suppressing "
        f"this step was added for. At x64 ON is where the complementary "
        f"denominators bite and where this repository does its jax work"
    )


def test_each_canary_leg_runs_the_sweep_IN_THE_CELL_IT_CAN_REDDEN_IN():
    """An alarm wired to a condition that cannot occur is not an alarm.

    The canary's eager CONSTANT SWEEP is the earliest signal there is that jax
    has grown an internal eager truncation nobody has written a row for -- and
    it can only find one with ``JAX_ENABLE_X64`` OFF. With x64 on, jax's own
    threefry mask widens to ``int64``, it fits, and NOTHING of jax's narrows,
    so ``unmatched`` is empty by construction whatever the installed jax does.
    Measured on jax 0.11.0: 729 conversion(s), 0 truncation(s), 0 row(s)
    exercised at x64=1 against 675, 13, 1 at x64=0.

    For one commit both legs of that workflow set ``JAX_ENABLE_X64: "1"`` and
    nothing else, while `docs/overflow-tripwire.md` said the nightly canary
    runs the sweep. Both halves were true and the conjunction was not: the
    page could not fire on either leg. This reads the workflow rather than the
    sentence, because the sentence is what was already wrong.
    """
    import re

    workflow = _readable_workflow(
        (_pathlib_for_canary() / ".github" / "workflows"
         / "nightly-jax-canary.yml").read_text(encoding="utf-8"),
        "the claim that each leg runs the sweep where it can redden",
    )

    jobs_and_blocks = _canary_jobs(workflow)
    assert jobs_and_blocks, (
        "no jobs found; the workflow layout this test reads has moved"
    )
    for job, body in jobs_and_blocks:
        # The resolution rule lives in `_x64_cells` — see there for what it
        # refuses to guess and for the hoisted-`env:` mutant that killed the
        # first version of this guard.
        cells = _x64_cells(
            workflow,
            body,
            lambda step: (isinstance(step.run, str)
                          and "tripwire_canary.py" in step.run),
        )
        assert cells, (
            f"the `{job}` leg does not run the canary at all"
        )
        # A CAN'T-TELL IS NOT A CELL, and this assertion read one as a PASS:
        # `any(cell in ("0", "unset"))` credited a leg with the cell it can
        # redden in on the strength of a setting nobody could parse. See
        # `_refuse_unreadable` for the eight shapes that measured green here.
        _refuse_unreadable(cells, f"the `{job}` leg's `tripwire_canary.py` steps")
        assert any(cell in ("0", "unset") for cell in cells), (
            f"every `tripwire_canary.py` step on the `{job}` leg runs at "
            f"JAX_ENABLE_X64={cells}, and the eager constant sweep cannot "
            "find an unenumerated jax constant at x64=1: jax's own mask "
            "widens to int64, nothing of jax's narrows, and `unmatched` is "
            "empty by construction. That leg cannot go red for the reason "
            "`eager:unenumerated-jax-constant` exists"
        )


# --------------------------------------------------------------------------
# THE INSTRUMENT ITSELF, DRIVEN. The two guards above read one workflow, so
# every shape they refuse would otherwise be evidenced only by a mutation
# somebody once ran by hand and wrote a number down for. These drive the
# resolver directly, over synthetic workflows, and they are the regression
# tests for the eight measured shapes that ran at x64 ON while the guard
# believed otherwise.
# --------------------------------------------------------------------------

_SYNTHETIC = """\
# a workflow shaped like `.github/workflows/nightly-jax-canary.yml`
name: synthetic
{top}jobs:
  control:
    name: control — the newest released jax
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: .venv/bin/python .github/scripts/tripwire_canary.py --require

{job_head}{job_extra}{matrix}    steps:
      - uses: actions/checkout@v4
      - name: install a jax nightly
        id: {install_id}
        run: uv pip install jax
{extra}      - name: the tripwire's own tests
{condition}{step_extra}{env_block}        run: |
{script}          .venv/bin/python -m pytest -q tests/test_tripwire_eager.py
"""

#: The `nightly` job header the cases that do not care about it get. It is a
#: slot rather than a constant because YAML mapping keys are UNORDERED and
#: the predecessor reader required `name:` to come first — see the key-order
#: row below.
_NIGHTLY_HEAD = "  nightly:\n    name: nightly jax\n"

#: ...and the wanted step's `env:` block, a slot for the same reason: a flow
#: mapping, an alias and a quoted key are three legal spellings of it and all
#: three read as OFF under the pattern this replaced.
_STEP_ENV = "        env:\n{setting}          JAX_PLATFORMS: cpu\n"


def _synthetic(setting=None, matrix=None, condition=None, script="", extra="",
               top="", job_head=None, job_extra="", step_extra="",
               env_block=None, install_id="install"):
    """One `nightly` job with one wanted step, spelled however the case asks.

    THE `install` STEP IS ALWAYS THERE because the one condition this reader
    knows names it, and a condition whose referent does not exist is a step
    that SKIPS — see `_STEP_CONDITIONS_THIS_READER_KNOWS`. `install_id` is
    what the row that drives that mutation moves.
    """
    return _SYNTHETIC.format(
        setting="" if setting is None else f"          JAX_ENABLE_X64: {setting}\n",
        matrix="" if matrix is None else (
            f"    strategy:\n      matrix:\n        {matrix}\n"
        ),
        condition="" if condition is None else f"        if: {condition}\n",
        script=script,
        extra=extra,
        top=top,
        job_head=_NIGHTLY_HEAD if job_head is None else job_head,
        job_extra=job_extra,
        step_extra=step_extra,
        env_block=(
            _STEP_ENV.format(
                setting="" if setting is None
                else f"          JAX_ENABLE_X64: {setting}\n")
            if env_block is None else env_block
        ),
        install_id=install_id,
    )


def _wanted(step):
    return isinstance(step.run, str) and "-m pytest" in step.run


def _cells_of(workflow, parser=None):
    return _x64_cells(_read_workflow(workflow, parser), "nightly", _wanted)


# (label, kwargs, what the PARSER may claim, what the LINE GRAMMAR may claim).
# `None` means the reader must claim NOTHING -- a can't-tell, refused rather
# than read as a cell.
#
# **TWO COLUMNS BECAUSE THERE ARE TWO READERS, AND THE SECOND ONE IS THE
# ZERO-DEP LANE'S.** PyYAML is in `stelling-jax` and not in
# `stelling-nojax`, and it is not a declared dependency, so the guards above
# must work with a parser and without one. Where the columns differ it is
# always the same way round -- the parser RESOLVES a legal spelling and the
# line grammar REFUSES it -- and that is the shape a fallback is allowed to
# have. What it is not allowed to do is the other one: read less and claim
# the same, which is what `"unset"` did for nine spellings.
#
# **THAT INVARIANT WAS FALSIFIED ONCE, BY A KEY WRITTEN TWICE.** `jobs`,
# `env` at all three levels and `strategy.matrix` were written with
# `setdefault`, so a second `env:` on one step MERGED into the first here
# and REPLACED it in PyYAML. Driven on the real workflow at `4a13824`: this
# grammar read `['1', '0']`, PyYAML read `['1', 'unset']`, and both were
# resolutions -- neither reader refused, so "always the same way round" was
# not true of this table when it said so. It is true again because the
# grammar refuses a repeated key now (`_put`), which is a fix to the code
# and not to the sentence. The caveat is recorded rather than glossed: what
# GitHub's own parser does with a duplicate mapping key is untested here,
# so what was demonstrated is a READER DIVERGENCE and not a runner exploit.
_RESOLUTIONS = [
    # jax's own grammar, and the YAML booleans that render into it. Driven
    # against `jax.config.jax_enable_x64` on jax 0.11.0, every row.
    ('quoted "0"', dict(setting='"0"'), ["0"], ["0"]),
    ('quoted "1"', dict(setting='"1"'), ["1"], ["1"]),
    ("bare 0", dict(setting="0"), ["0"], ["0"]),
    ("bare 1", dict(setting="1"), ["1"], ["1"]),
    ("YAML true", dict(setting="true"), ["1"], ["1"]),
    ("YAML on", dict(setting="on"), ["1"], ["1"]),
    ("YAML off", dict(setting="off"), ["0"], ["0"]),
    ("YAML no", dict(setting="no"), ["0"], ["0"]),
    ('quoted "on"', dict(setting='"on"'), ["1"], ["1"]),
    ('quoted "yes"', dict(setting='"yes"'), ["1"], ["1"]),
    ("uppercase TRUE", dict(setting='"TRUE"'), ["1"], ["1"]),
    # NO SETTING ANYWHERE IS NOT A CAN'T-TELL, AND EITHER READER MAY NOW SAY
    # SO. jax's default is OFF -- driven, `jax.config.jax_enable_x64` is
    # `False` with the variable unset -- so this is a READING and the one
    # thing `"unset"` may mean. The predecessor said it by NOT FINDING one,
    # which is why nine legal spellings read as OFF; both readers here have
    # read the whole file before they say it, the parser by parsing and the
    # line grammar by classifying every line of it.
    ("no setting anywhere", dict(), ["unset"], ["unset"]),
    # ...and everything else is refused.
    ("a word jax would raise on", dict(setting='"maybe"'), None, None),
    ("the empty string", dict(setting='""'), None, None),
    # THE MATRIX CHAIN, FOLLOWED ENTRY BY ENTRY, which is what makes the
    # honest two-cell refactor legal and the one-cell pin illegal. Forbidding
    # expressions outright had it exactly the wrong way round on both.
    ("matrix over both cells",
     dict(setting="${{ matrix.x64 }}", matrix='x64: ["0", "1"]'),
     ["0", "1"], ["0", "1"]),
    ("matrix over both cells, block sequence",
     dict(setting="${{ matrix.x64 }}", matrix='x64:\n          - "0"\n          - "1"'),
     ["0", "1"], ["0", "1"]),
    ("matrix pinned to ON",
     dict(setting="${{ matrix.x64 }}", matrix='x64: ["1"]'), ["1"], ["1"]),
    ("matrix axis the job does not carry",
     dict(setting="${{ matrix.x64 }}", matrix='other: ["0", "1"]'), None, None),
    ("matrix reference with no matrix at all",
     dict(setting="${{ matrix.x64 }}"), None, None),
    ("an expression that is not a matrix reference",
     dict(setting="${{ env.CELL }}"), None, None),
    # AN AXIS ENTRY TAKEN BACK OUT. `exclude:` is not an axis and a matrix
    # that carries one is not an expansion this reader follows. Driven on the
    # real workflow: `x64: ["0", "1"]` with `exclude: [{x64: "0"}]` over
    # `${{ matrix.x64 }}` left this file at `95 passed` while the job runs in
    # the ON cell alone.
    ("a matrix axis with an exclude",
     dict(setting="${{ matrix.x64 }}",
          matrix='x64: ["0", "1"]\n        exclude:\n          - x64: "0"'),
     None, None),
    # FOUR WAYS TO SET IT THAT ARE NOT AN `env:` ENTRY. This reader does not
    # evaluate shell, does not order steps against one, and does not run
    # somebody else's action.
    ("`export` inside the run: block",
     dict(script="          export JAX_ENABLE_X64=1\n"), None, None),
    ("a command prefix",
     dict(setting='"0"', script="          JAX_ENABLE_X64=1 \\\n"), None, None),
    ("an earlier step writing $GITHUB_ENV",
     dict(setting='"0"',
          extra='      - run: echo "JAX_ENABLE_X64=1" >> $GITHUB_ENV\n'),
     None, None),
    # ...AND ONE THAT NEVER NAMES THE VARIABLE AT ALL, which is why the
    # `run:` rule is a whitelist of command words rather than a substring
    # test. Driven on the real workflow's x64 OFF pytest step: `95 passed`,
    # with the step's own `env:` still saying `"0"` and a `ci.env` this
    # reader has never seen deciding what the process actually gets. The
    # script names nothing; `set -a` exports whatever the sourced file
    # assigns.
    ("`set -a` and a sourced file",
     dict(setting='"0"',
          script="          set -a; . ./ci.env; set +a\n"), None, None),
    ("a $GITHUB_ENV write that names nothing",
     dict(setting='"0"',
          extra='      - run: cat ci.env >> $GITHUB_ENV\n'), None, None),
    ("a command word this reader has no opinion about",
     dict(setting='"0"', script="          ./setup-the-cell.sh\n"), None, None),
    ("an action this reader has not been told about",
     dict(setting='"0"', extra="      - uses: some-org/set-my-env@v1\n"),
     None, None),
    # A STEP THAT MIGHT NOT RUN CANNOT BE CREDITED WITH RUNNING IN A CELL,
    # AND NEITHER CAN ONE WHOSE RED CANNOT FAIL THE JOB.
    ("the one condition this reader knows",
     dict(setting='"0"', condition="steps.install.outcome == 'success'"),
     ["0"], ["0"]),
    ("a condition it does not",
     dict(setting='"0"', condition="steps.install.outcome == 'success' && false"),
     None, None),
    ("any other condition",
     dict(setting='"0"', condition="github.event_name == 'schedule'"),
     None, None),
    # THE CONDITION'S REFERENT, WHICH THE LIST DID NOT PIN. Driven on the
    # real workflow: `id: install` renamed to `id: install-nightly` and
    # nothing else touched left this file at `95 passed` with the cells
    # unchanged, while on the runner all four canary and pytest steps skip
    # and the leg reports "SKIPPED (infrastructure)" and goes GREEN.
    ("the condition's referent renamed away",
     dict(setting='"0"', condition="steps.install.outcome == 'success'",
          install_id="install-nightly"), None, None),
    # Driven on the real workflow's x64 OFF canary step: `95 passed`.
    ("continue-on-error on the step whose red is the signal",
     dict(setting='"0"', step_extra="        continue-on-error: true\n"),
     None, None),
    ("continue-on-error: false, which takes nothing away",
     dict(setting='"0"', step_extra="        continue-on-error: false\n"),
     ["0"], ["0"]),
    # A KEY NEITHER READER MODELS STOPS THE READING RATHER THAN BEING
    # IGNORED, which is the general form of four of the nine. A job-level
    # `if:` can stop the whole job; `defaults:` changes the shell every
    # `run:` in the file executes in (driven on the real workflow:
    # `defaults: {run: {shell: bash -lc}}` on the nightly job left this file
    # at `95 passed`, and a login shell sources profile scripts this reader
    # never sees); a job-level `uses:` makes the leg somebody else's
    # workflow entirely.
    ("a job-level if:", dict(setting='"0"',
                             job_extra="    if: github.event_name == 'schedule'\n"),
     None, None),
    ("a job-level defaults:",
     dict(setting='"0"',
          job_extra="    defaults:\n      run:\n        shell: bash -lc\n"),
     None, None),
    ("a workflow-level defaults:",
     dict(setting='"0"', top="defaults:\n  run:\n    shell: bash -lc\n"),
     None, None),
    ("a job that is somebody else's workflow",
     dict(setting='"0"', job_extra="    uses: some-org/some-repo/.github/workflows/w.yml@v1\n"),
     None, None),
    ("a step key this reader does not model",
     dict(setting='"0"', step_extra="        working-directory: /elsewhere\n"),
     None, None),
    # KEY ORDER IS NOT A FACT ABOUT A MAPPING, and the predecessor needed
    # `name:` first. Both readers key on the two-space job id alone now, so
    # this READS rather than refusing -- see
    # `test_the_reader_sees_A_JOB_IT_WAS_NOT_EXPECTING` for the
    # third leg that hid behind the old requirement.
    ("runs-on: written before name:",
     dict(setting='"0"',
          job_head="  nightly:\n    runs-on: ubuntu-latest\n    name: nightly jax\n"),
     ["0"], ["0"]),
    # THREE LEGAL SPELLINGS OF ONE `env:` MAPPING. A parser resolves all
    # three; the line grammar resolves none and REFUSES all three, which is
    # the only other answer it is allowed to give. Each was measured on the
    # real workflow at `95 passed` with the nightly canary running at x64 ON.
    ("a quoted key",
     dict(env_block='        env:\n          "JAX_ENABLE_X64": "1"\n'
                    '          JAX_PLATFORMS: cpu\n'),
     ["1"], None),
    ("a flow mapping",
     dict(env_block='        env: { JAX_ENABLE_X64: "1", JAX_PLATFORMS: cpu }\n'),
     ["1"], None),
    ("an alias to an anchored mapping",
     dict(job_extra='    env: &x64on\n      JAX_ENABLE_X64: "1"\n',
          env_block="        env: *x64on\n"),
     ["1"], None),
    # A LINE BREAK THIS GRAMMAR DOES NOT SPLIT ON AND PyYAML DOES. One
    # U+2028 in place of the newline between two `env:` entries swallowed
    # the `JAX_ENABLE_X64:` line into the value above it, and the grammar
    # reported the cell `"unset"` -- "I have read all of it and it is not
    # there" -- about a line it had never classified. Measured on the real
    # workflow at `4a13824`, and identical for U+0085 and U+2029. The
    # grammar refuses the character now; see `_lines_of_this_grammar` for
    # why refusing beats agreeing with PyYAML about it.
    ("a U+2028 where the newline goes",
     dict(env_block='        env:\n          JAX_PLATFORMS: cpu'
                    '\u2028          JAX_ENABLE_X64: "1"\n'),
     ["1"], None),
    ("a U+0085 where the newline goes",
     dict(env_block='        env:\n          JAX_PLATFORMS: cpu'
                    '\u0085          JAX_ENABLE_X64: "1"\n'),
     ["1"], None),
    ("a U+2029 where the newline goes",
     dict(env_block='        env:\n          JAX_PLATFORMS: cpu'
                    '\u2029          JAX_ENABLE_X64: "1"\n'),
     ["1"], None),
    # ...AND CR, WHICH IS THE OTHER ANSWER. A carriage return is a line
    # break in YAML 1.1 and 1.2 alike, so this grammar normalises it instead
    # of refusing it and the two readers AGREE. It read `unset` here until
    # 2026-08-24; through `read_text` it was neutralised by universal-newline
    # translation, which is an accident of how the caller loaded the file and
    # not a property of the grammar.
    ("a CR where the newline goes",
     dict(env_block='        env:\n          JAX_PLATFORMS: cpu'
                    '\r          JAX_ENABLE_X64: "1"\n'),
     ["1"], ["1"]),
    # A KEY WRITTEN TWICE. PyYAML keeps the LAST `env:` and drops the first
    # -- so the setting disappears and the reading is `unset` -- where this
    # grammar merged the two and read `1`. Both were resolutions and they
    # differed, which is the one shape the note above this table said could
    # not happen. Refused here now.
    ("a step `env:` written twice",
     dict(env_block='        env:\n          JAX_ENABLE_X64: "1"\n'
                    '        env:\n          JAX_PLATFORMS: cpu\n'),
     ["unset"], None),
    ("a job `env:` written twice",
     dict(setting='"0"',
          job_extra='    env:\n      JAX_ENABLE_X64: "1"\n'
                    '    env:\n      JAX_PLATFORMS: cpu\n'),
     ["0"], None),
    ("a matrix axis written twice",
     dict(setting="${{ matrix.x64 }}",
          matrix='x64: ["0", "1"]\n        x64: ["1"]'), ["1"], None),
]


def _reading(workflow, parser, expected, label):
    """Drive ONE reader over ONE synthetic workflow and hold it to its column.

    "Refused" is checked through the caller's own gate rather than by reading
    the sentinel, because the sentinel is not what the guards depend on.
    """
    cells = _cells_of(workflow, parser)
    if expected is None:
        with pytest.raises(AssertionError, match="cannot tell"):
            _refuse_unreadable(cells, "the synthetic job")
        assert all(cell.startswith("?") for cell in cells), (
            f"{label} ({parser}): refused, but only partly — {cells}. A step "
            f"whose cell this reader cannot resolve must contribute no cell "
            f"at all"
        )
    else:
        _refuse_unreadable(cells, f"the synthetic job ({parser})")
        assert cells == expected, (
            f"{label} ({parser}): read {cells}, not {expected}"
        )


@pytest.mark.parametrize(
    "label,kwargs,strict,plain", _RESOLUTIONS, ids=[r[0] for r in _RESOLUTIONS]
)
def test_the_cell_resolver_REFUSES_WHAT_IT_CANNOT_PARSE(
    label, kwargs, strict, plain
):
    """A can't-tell is not a cell, and it used to be spelled like one.

    `_x64_cells` returned the single string ``"unset"`` for BOTH "no setting
    anywhere" -- jax's default, genuinely OFF -- and "a setting I could not
    parse", and both callers read the conflated value as OFF. EIGHT measured
    mutations of the real workflow ran at x64 ON with this file at
    `68 passed`: `JAX_ENABLE_X64: true`; `"on"`; `${{ matrix.x64 }}` over
    `x64: ["1"]`; `export JAX_ENABLE_X64=1` in the `run:` block; a
    `JAX_ENABLE_X64=1` command prefix; an earlier step writing the name into
    `$GITHUB_ENV`; `&& false` on the OFF step; and the OFF step moved into the
    other job. `true` and `on` are not exotic -- jax's own boolean-environment
    reader takes `y yes t true on 1`, driven through `jax.config` on jax
    0.11.0 -- and the regex saw only `0`/`1`.

    NINE MORE WERE FOUND THE ROUND AFTER, all of them ordinary workflow YAML
    that GitHub runs, and they are why the reader is a parser now: a quoted
    key, a flow mapping, an alias, `matrix.exclude`, `runs-on:` before
    `name:`, a job-level `if:`, a `defaults:` block, `set -a; . ./ci.env;
    set +a`, and `continue-on-error: true` on the step whose red IS the
    signal. Each measured at `95 passed` on the real workflow: the first
    four are a cell read WRONG, the fifth a whole LEG never seen, the last
    four a cell the reader was not entitled to claim at all.

    A can't-tell that defaults to the answer the caller wanted is the shape
    this campaign has already named once, in `Lane.jax`, and this table is
    what stops it being renamed here. Each row is a workflow a reader either
    RESOLVES to named cells or REFUSES; there is no third outcome.

    **BOTH READERS ARE DRIVEN, AND THE LINE GRAMMAR IN EVERY LANE.** PyYAML
    is in `stelling-jax` and absent from `stelling-nojax`, so the fallback is
    what the zero-dep lane's guards actually use — driving it only where the
    parser exists would leave the reader the zero-dep lane runs on checked
    by nothing. The parser's column is asserted wherever PyYAML is
    importable, which is a fact about the environment and not a skip: this
    test runs, and can fail, in every lane.
    """
    workflow = _synthetic(**kwargs)
    _reading(workflow, "text", plain, label)
    if _YAML_IS_IMPORTABLE:
        _reading(workflow, "yaml", strict, label)


def test_which_reader_this_lane_uses_IS_ASSERTED_AND_NOT_ASSUMED():
    """The guards above take the strongest reader the lane has. Say which.

    A fallback nobody can see is a fallback nobody notices going wrong. This
    asserts the choice in BOTH directions -- with PyYAML importable the
    guards must be parsing, and without it they must be on the line grammar
    -- so a lane cannot quietly answer with a reader it did not mean to use,
    and neither branch is a skip.

    AND THE TWO MUST AGREE ABOUT THE FILE THE GUARDS READ. Where they differ
    on the synthetic table it is always the same way round (the parser
    resolves a legal spelling the line grammar refuses); on
    `.github/workflows/nightly-jax-canary.yml` they must resolve the same
    cells, because that is the file two guards make claims about and a
    fallback that reads it differently is a second, non-identical rule.
    """
    workflow = (
        _pathlib_for_canary() / ".github" / "workflows" / "nightly-jax-canary.yml"
    ).read_text(encoding="utf-8")
    read = _read_workflow(workflow)
    assert read.parser == ("yaml" if _YAML_IS_IMPORTABLE else "text"), (
        f"this lane read the workflow with {read.parser!r} while PyYAML "
        f"{'is' if _YAML_IS_IMPORTABLE else 'is not'} importable"
    )
    canary = (lambda step: isinstance(step.run, str)
              and "tripwire_canary.py" in step.run)
    plain = _read_workflow(workflow, "text")
    assert not plain.blockers, (
        f"the line grammar cannot read this repository's own nightly "
        f"workflow: {plain.blockers}. The zero-dep lane has no other reader, "
        f"so this is the whole of what that lane's two canary guards can "
        f"say"
    )
    by_text = {job: _x64_cells(plain, body, canary)
               for job, body in _canary_jobs(plain)}
    assert by_text and all(by_text.values()), by_text
    for job, cells in by_text.items():
        _refuse_unreadable(cells, f"the `{job}` leg, read by the line grammar")
    if not _YAML_IS_IMPORTABLE:
        return
    strict = _read_workflow(workflow, "yaml")
    by_yaml = {job: _x64_cells(strict, body, canary)
               for job, body in _canary_jobs(strict)}
    assert by_text == by_yaml, (
        f"the two readers disagree about the canary steps of this "
        f"repository's own workflow: line grammar {by_text}, parser "
        f"{by_yaml}. They are one rule in two implementations and the "
        f"guards take whichever the lane has"
    )


#: The four characters PyYAML breaks a line on and `str.split("\\n")` does
#: not, with what this grammar is required to do about each. See
#: `_lines_of_this_grammar`: CR is the spec in every YAML version, so it is
#: NORMALISED; the other three are a version disagreement, so they are
#: REFUSED.
_LINE_BREAKS_AND_WHAT_THE_GRAMMAR_OWES_THEM = (
    ("\r", "normalise"),
    ("\u0085", "refuse"),
    ("\u2028", "refuse"),
    ("\u2029", "refuse"),
)


def test_the_grammars_LINE_is_YAMLS_LINE_AND_NOT_STR_SPLITS():
    """A grammar that classifies every line is only as total as "line".

    THE MEASUREMENT THIS EXISTS FOR. `.github/workflows/nightly-jax-canary.
    yml` with ONE U+2028 in place of the newline between two entries of the
    nightly x64-OFF canary step's `env:` block, at `4a13824`: the whole
    zero-dep suite came back IDENTICAL to clean, and the reading reported the
    cell ``"unset"`` for that step — *"I have read all of it and the setting
    is not there"* — about a `JAX_ENABLE_X64:` line it had never classified,
    because `text.split("\\n")` had swallowed it into the value of the entry
    above. U+0085 and U+2029 behave the same way. That is precisely the
    "I did not find it" wearing "it is not there" that the total grammar
    exists to make impossible, which is why it is a defect and not a note.

    WHAT IS AND IS NOT ESTABLISHED. Whether GitHub's own parser breaks a
    line on U+0085 / U+2028 / U+2029 is NOT tested here — YAML 1.1 does and
    YAML 1.2 does not — so what was measured is an un-classified value
    reported as a cell and a divergence between this repository's two
    readers, and not a demonstrated runner exploit. The grammar's claim is
    about totality rather than about exploitability, so it is fixed on the
    strength of the first.

    CR IS THE OTHER ANSWER AND IS DRIVEN SEPARATELY. It is a line break in
    every YAML version, so this grammar normalises it rather than refusing
    it — and the two readers must AGREE about a file that carries one.
    Before this it was live on an in-memory string and neutralised through
    the file path only by ``read_text``'s universal-newline translation: an
    accident of loading, and this asserts it is a property of the grammar.
    """
    workflow = (
        _pathlib_for_canary() / ".github" / "workflows"
        / "nightly-jax-canary.yml"
    ).read_text(encoding="utf-8")
    lines = workflow.split("\n")
    # the two entries of the NIGHTLY x64-OFF canary step's `env:` block, in
    # the other legal order -- a mapping's keys are unordered -- so that the
    # break to be replaced is the one ABOVE the `JAX_ENABLE_X64:` line.
    # Searched from the `nightly:` job header and not from the top of the
    # file: the `control` job carries the same two lines first, and a
    # mutation spliced into the OTHER job is one the cells read below cannot
    # see at all.
    nightly = lines.index("  nightly:")
    sites = [index for index, line in enumerate(lines[nightly:], nightly)
             if line == '          JAX_ENABLE_X64: "0"']
    assert sites, (
        "the `nightly` job of the nightly workflow no longer carries a step "
        "`env:` entry spelled `JAX_ENABLE_X64: \"0\"`, which is the line "
        "this test splices a line break into. Re-point it at the step whose "
        "cell the two canary guards read, rather than deleting it"
    )
    off = sites[0]
    assert lines[off + 1] == "          JAX_PLATFORMS: cpu", lines[off + 1]
    swapped = lines[:off] + [lines[off + 1], lines[off]] + lines[off + 2:]

    def canary_cells(text):
        """The CANARY steps' cells, because the mutated `env:` is a canary
        step's. `_cells_of`'s `-m pytest` predicate selects other steps of
        the same job, and a mutation those steps cannot see is a test that
        would pass without reading anything."""
        read = _read_workflow(text, "text")
        if read.blockers:
            return [_cannot_tell(reason) for reason in read.blockers]
        return _x64_cells(read, dict(_canary_jobs(read))["nightly"],
                          _wanted_canary)

    clean = canary_cells("\n".join(swapped))
    assert clean == ["1", "0"], clean

    # POSITIVE CONTROL ON THE SPLICE SITE. Everything below reads the cells
    # of one step, and this is the proof that the line being mutilated is a
    # line those cells come from: flip its value and the reading follows.
    # Without it, splicing into the `control` job -- which carries the same
    # two lines earlier in the file -- would leave every assertion below
    # passing while watching nothing.
    flipped = swapped[:off + 1] + ['          JAX_ENABLE_X64: "1"'] + swapped[off + 2:]
    assert canary_cells("\n".join(flipped)) == ["1", "1"], (
        "the line this test mutilates is not one the cells it reads come "
        "from, so nothing below is watching anything"
    )

    for char, owed in _LINE_BREAKS_AND_WHAT_THE_GRAMMAR_OWES_THEM:
        spliced = "\n".join(
            swapped[:off] + [swapped[off] + char + swapped[off + 1]]
            + swapped[off + 2:]
        )
        cells = canary_cells(spliced)
        if owed == "refuse":
            with pytest.raises(AssertionError, match="cannot tell"):
                _refuse_unreadable(cells, "the spliced workflow")
            assert all(cell.startswith("?") for cell in cells), (
                f"{char!r} in place of a newline: this grammar produced the "
                f"cells {cells} for a file it cannot cut into lines. PyYAML "
                f"breaks a line there and YAML 1.2 does not, so the only "
                f"answer that is not a guess is a refusal"
            )
        else:
            _refuse_unreadable(cells, "the spliced workflow")
            assert cells == clean, (
                f"{char!r} in place of a newline: read {cells}, and a "
                f"carriage return is a line break in YAML 1.1 and 1.2 alike, "
                f"so this grammar owes the same reading as {clean}"
            )
        assert "unset" not in cells, (
            f"{char!r} in place of a newline: the reading says {cells}, and "
            f"`unset` means *I have read the whole file and the setting is "
            f"not there* about a line this grammar never classified. That is "
            f"the exact conflation the total grammar replaced a pattern to "
            f"make impossible"
        )

    # ...AND THE WHOLE FILE IN CRLF, which is what a Windows checkout hands
    # this reader. It was REFUSED outright before the fix -- `line 3: '\\r' is
    # outside this grammar` -- so the zero-dep lane's only reader could not
    # read a legal spelling of this repository's own workflow at all.
    crlf = _read_workflow("\r\n".join(workflow.split("\n")), "text")
    assert not crlf.blockers, crlf.blockers
    plain = _read_workflow(workflow, "text")
    assert ([(job, _x64_cells(crlf, body, _wanted_canary))
             for job, body in _canary_jobs(crlf)]
            == [(job, _x64_cells(plain, body, _wanted_canary))
                for job, body in _canary_jobs(plain)]), (
        "the same workflow with CRLF line endings does not read the same as "
        "with LF, and a carriage return is a line break in every version of "
        "YAML"
    )

    # A DOCUMENT MARKER OPENED BY A CR. The check written to refuse a
    # multi-document workflow was `re.search(r"^---", text, re.M)`, and
    # Python's `^` under `re.M` matches after a newline and NOT after a
    # carriage return -- so the one line-break character the grammar was
    # already exposed to could hide the marker from the refusal.
    hidden = _read_workflow(
        "name: a\r--- \nname: b\njobs:\n  x:\n    steps:\n      - run: echo hi\n",
        "text",
    )
    assert hidden.blockers and "document marker" in hidden.blockers[0], (
        f"a second document opened by a carriage return was read rather than "
        f"refused: {hidden.blockers}"
    )

    # AND THE SAME DEFECT IN THE SHELL READER. `shlex` does not treat any of
    # these four as whitespace, so `echo preparing<break>./setup-the-cell.sh`
    # was ONE command whose head is the whitelisted `echo` -- and
    # `_RESOLUTIONS` carries that same `./setup-the-cell.sh` as a row that
    # must refuse. Measured `None` for all four at `4a13824`.
    with_newline = _shell_reason("echo preparing\n./setup-the-cell.sh")
    assert with_newline is not None and "setup-the-cell" in with_newline
    for char, _ in _LINE_BREAKS_AND_WHAT_THE_GRAMMAR_OWES_THEM:
        reason = _shell_reason("echo preparing" + char + "./setup-the-cell.sh")
        assert reason == with_newline, (
            f"a script whose two commands are separated by {char!r} reads as "
            f"{reason!r}, where the same script with a newline is "
            f"{with_newline!r}. Splitting on more breaks than the shell does "
            f"can only put MORE command words in front of the whitelist, "
            f"which is the safe direction"
        )
    # ...and the comment case, which fails the other way round: a `#` that
    # ran to the end of a `split("\\n")` line hid whatever came after the
    # break behind it.
    hidden_export = _shell_reason("# nothing to see\u2028export JAX_ENABLE_X64=1")
    assert hidden_export is not None and "export" in hidden_export, (
        f"a `#` comment hid an `export` behind a U+2028: {hidden_export!r}"
    )


def _wanted_canary(step):
    return isinstance(step.run, str) and "tripwire_canary.py" in step.run


def test_a_KEY_WRITTEN_TWICE_is_REFUSED_and_not_merged():
    """Two readers, two answers, and neither of them a refusal.

    `jobs`, `env` at all three levels and `strategy.matrix` were written with
    `setdefault`, so a repeated key MERGED here and REPLACES in PyYAML.
    Driven on the real workflow at `4a13824` with a second `env:` on the
    nightly x64-OFF canary step carrying only `JAX_PLATFORMS`: this grammar
    read `['1', '0']` and PyYAML read `['1', 'unset']`, the whole zero-dep
    suite stayed at `114 passed`, and the two-reader agreement test was the
    only thing in the tree that moved. That falsified `_RESOLUTIONS`' stated
    invariant — *"where the columns differ it is always the same way round:
    the parser RESOLVES and the line grammar REFUSES"* — because here both
    resolved.

    The three levels of `env:` and the matrix axis are rows of that table.
    What is here is what the table has no slot for: a repeated JOB key, a
    repeated top-level key, and a step with two `run:` blocks, whose value is
    written when the block CLOSES and so needed a check of its own.

    HONESTLY BOUNDED: what GitHub's own parser does with a duplicate mapping
    key is untested here — a rejection is the likely third answer — so what
    is demonstrated is a reader divergence and not a runner exploit.
    """
    base = _synthetic(setting='"0"')
    assert not _read_workflow(base, "text").blockers

    for label, text in (
        ("a job written twice",
         base + "  nightly:\n    name: nightly again\n"
                "    runs-on: ubuntu-latest\n"),
        ("a top-level key written twice", base + "name: synthetic again\n"),
        ("a step with two `run:` blocks",
         base + "        run: |\n          echo twice\n"),
    ):
        read = _read_workflow(text, "text")
        assert read.blockers, (
            f"{label}: the line grammar read this file rather than refusing "
            f"it. A key written twice merges here and replaces in PyYAML, so "
            f"reading it is two readers with two answers"
        )
        assert "twice" in read.blockers[0], read.blockers

    # ...AND THE REAL WORKFLOW STILL READS, which is the other half: a
    # refusal that also refuses the file the guards are about buys nothing.
    workflow = (
        _pathlib_for_canary() / ".github" / "workflows"
        / "nightly-jax-canary.yml"
    ).read_text(encoding="utf-8")
    assert not _read_workflow(workflow, "text").blockers


def test_the_line_grammar_HAS_NO_THIRD_OUTCOME():
    """Classify, or refuse. Not `TypeError`.

    An indent-4 job key with no job header above it reached
    `job.setdefault(...)` with `job` still `None`, and `_read_workflow`
    catches `_Refused` alone — so the grammar raised a bare
    ``TypeError: 'NoneType' object does not support item assignment`` out of
    itself, with no line number, no reader named and no refusal sentence.
    Driven by commenting out `  control:` on the real workflow at `4a13824`:
    four tests failed that way. Red, so never a HOLE — but it is a third
    outcome beside the two this grammar claims, and `_parse_with_yaml`
    guards its own analogue by turning a `yaml.YAMLError` into a blocker.

    So the claim is driven rather than asserted: every ONE-LINE deletion and
    every ONE-LINE comment-out of this repository's own workflow goes
    through the reader, and each must come back either read or refused.
    Those two mutilations are cheap, they are what a person actually does to
    a workflow, and between them they take away every header this grammar
    keys on.
    """
    workflow = (
        _pathlib_for_canary() / ".github" / "workflows"
        / "nightly-jax-canary.yml"
    ).read_text(encoding="utf-8")
    lines = workflow.split("\n")
    assert len(lines) > 200, len(lines)

    read, refused = 0, 0
    for index in range(len(lines)):
        for mutilated in (
            lines[:index] + lines[index + 1:],
            lines[:index] + ["#" + lines[index].lstrip(" ")] + lines[index + 1:],
        ):
            text = "\n".join(mutilated)
            try:
                result = _read_workflow(text, "text")
            except Exception as exc:  # noqa: BLE001 - that is the assertion
                raise AssertionError(
                    f"the line grammar raised {type(exc).__name__}({exc}) out "
                    f"of itself on line {index + 1} of the nightly workflow "
                    f"({lines[index]!r}). Every way out of `_parse_by_line` "
                    f"has to be a reading or a `_Refused`: a caller that gets "
                    f"a traceback gets no line number, no reader and no "
                    f"refusal sentence, and `_read_workflow` catches "
                    f"`_Refused` alone"
                ) from exc
            if result.blockers:
                refused += 1
            else:
                read += 1
    # Both outcomes have to occur, or this test is asserting nothing: a
    # reader that refused everything, or read everything, would pass the
    # loop above.
    assert read and refused, (read, refused)


@pytest.mark.parametrize("parser", ["yaml", "text"])
def test_the_reader_sees_A_JOB_IT_WAS_NOT_EXPECTING(parser):
    """A job id is the cheapest thing there is to add to a workflow, and two
    readings of "job" have each missed one.

    THE CHARSET. GitHub allows a leading letter or `_` and then letters,
    digits, `-` and `_`; the pattern read `[a-z-]+`. Driven on the real
    workflow: a THIRD job `nightly_x64:` running `tripwire_canary.py` at
    `JAX_ENABLE_X64: "1"` only -- an alarm in the one cell the eager constant
    sweep cannot redden in -- left this file at `68 passed`, and
    `test_the_canary_and_the_workflow_agree_about_the_two_legs` passed too
    because it carried a SECOND copy of the same pattern.

    **AND THEN KEY ORDER, WHICH IS NOT A FACT ABOUT A MAPPING AT ALL.** The
    repaired pattern was `\n  (job):\n    name: ` — a two-space key whose
    FIRST CHILD is `name:` — and YAML mapping keys are unordered. Driven on
    the real workflow at `1f55eef`: the same third leg written with
    `runs-on:` before `name:` left this file at `95 passed`, and the reader
    reported `jobs seen: ['control', 'nightly']`. A job is a key under
    `jobs:` and that is now what both readers key on: PyYAML because it
    parses, the line grammar because it classifies the two-space key itself
    and has already read the rest of the file by the time it answers.

    Both readers are driven here, on one workflow carrying all four shapes.
    """
    workflow = _synthetic(setting='"0"') + """
  nightly_x64:
    name: nightly jax, x64 only
    runs-on: ubuntu-latest
    steps:
      - run: .venv/bin/python .github/scripts/tripwire_canary.py --require

  Build2:
    runs-on: ubuntu-latest
    name: a leading capital, a digit, and `name:` written second
    steps:
      - run: echo hello
"""
    if parser == "yaml" and not _YAML_IS_IMPORTABLE:
        # NOT A SKIP: the line grammar's column is the one the zero-dep lane
        # runs on and it is asserted in every lane. This half asserts the
        # parser's, where there is one.
        parser = "text"
    read = _read_workflow(workflow, parser)
    assert not read.blockers, read.blockers
    assert [job for job, _ in _canary_jobs(read)] == [
        "control", "nightly", "nightly_x64", "Build2"
    ], (
        "the job reader cannot see every job of this workflow, so a leg can "
        "be added that no test in this repository reads"
    )


def test_every_message_about_the_two_legs_carries_the_SAME_guidance(
    capsys, tmp_path
):
    """One constant, and the sentences that need it quote it whole.

    This paragraph was typed out twice, 190 lines apart, and the two copies
    drifted into saying different and both-wrong things -- one of them the
    withdrawn claim that the control leg runs a pinned series. Counting
    occurrences of the constant's NAME does not stop that: replacing one use
    with an inlined copy leaves the count high enough. So the runs are driven
    and each sentence is required to CONTAIN the constant, which an inlined
    reword cannot do and a rewording of the constant itself cannot break.

    THE THREE THAT NEED IT are the findings a reader answers by looking at
    the other leg: the tripwire would not arm, the hook is dead, the probe
    could not run. The `indeterminate` and `unrenderable` findings
    deliberately do NOT carry it -- they say the defect is in this repository
    and send nobody to compare anything.
    """
    canary = _canary()
    carries = {
        "not-armed": {"armed": False, "require": True},
        "control:did-not-fire": {"findings": "none"},
        "control:raised": {"probe": "raises"},
    }
    for code, kw in carries.items():
        run = _drive_canary(capsys, tmp_path, **kw)
        assert canary._TWO_LEGS in run.sentences.get(code, ""), (
            f"the {code!r} sentence gives leg guidance of its own instead of "
            "the shared constant, which is how the two copies of it came to "
            f"disagree:\n{run.sentences.get(code)}"
        )

    for code, kw in (
        ("control:indeterminate", {"findings": "unreadable"}),
        ("control:unrenderable", {"findings": "field-moved"}),
    ):
        run = _drive_canary(capsys, tmp_path, **kw)
        assert canary._TWO_LEGS not in run.sentences.get(code, ""), (
            f"{code!r} says the defect is in this repository and then sends "
            "the reader to compare the two legs anyway"
        )


def _readable_workflow(text: str, where: str):
    """The nightly workflow, READ, or a red that says why it could not be.

    THE GUARDS RUN IN A LANE WITH NO YAML PARSER and this is the sentence
    they fail with there. `stelling-nojax` has no PyYAML and PyYAML is not a
    declared dependency, so the line grammar is the whole of what those two
    guards have in the zero-dep lane — and a grammar that stops on a line it
    cannot place has to stop LOUDLY, in the file that the claim is about,
    rather than resolving to an empty job list that reads like a moved
    layout.
    """
    read = _read_workflow(text)
    assert not read.blockers, (
        f"{where}: this reader ({read.parser}) will not read "
        f"`.github/workflows/nightly-jax-canary.yml` — {list(read.blockers)}. "
        f"That is a refusal and not a reading: teach the reader the shape and "
        f"say in the same commit what you measured it to mean, or spell the "
        f"workflow so it can be read"
    )
    return read


def _pathlib_for_canary():
    import pathlib as _pathlib

    return _pathlib.Path(__file__).resolve().parents[1]
