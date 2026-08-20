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
    the program it is everywhere else; only the tracer is a stand-in.
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
        zeros=lambda shape, dtype: _Array(), int8="int8",
        full=lambda shape, value, dtype: _Array(),
    )
    monkeypatch.setitem(sys.modules, "stelling._jax_compat", fake)
    monkeypatch.setattr(stelling, "_jax_compat", fake, raising=False)

    if probe == "raises":
        def _boom(a):
            raise RuntimeError("FORCED: the probe could not execute")

        monkeypatch.setattr(_probe, "over", _boom)


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
            f"{self.rows.get('control report')!r}>"
        )


def _drive_canary(
    capsys, tmp_path, *,
    require=False, armed=True, hash_state="as-tested",
    probe="runs", findings="one", narrowings="readable",
    summary="writable", eager="clean",
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
        )


def _drive_canary_once(
    monkeypatch, capsys, tmp_path, *,
    require, armed, hash_state, probe, findings, narrowings, summary,
    eager="clean",
):
    import sys

    from stelling import _tripwire
    from stelling._tripwire import Status

    canary = _canary()
    _stub_jax(monkeypatch, probe)
    _stub_eager(monkeypatch, eager)

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

    jobs = dict(re.findall(r"\n  ([a-z-]+):\n    name: (.*)", workflow))
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
    starts = [
        (m.start(), m.group(1))
        for m in re.finditer(r"\n  ([a-z-]+):\n    name: ", workflow)
    ]
    bounds = [pos for pos, _ in starts] + [len(workflow)]
    for index, (_, job) in enumerate(starts):
        block = workflow[bounds[index]:bounds[index + 1]]
        # ON A `run:` LINE, not anywhere in the block: this file is YAML and
        # a `#` turns the step into prose that still contains every word
        # below. Commenting the step out is the same mutation as deleting it.
        runs = re.findall(r"^\s*run:[^\n]*tripwire_canary\.py[^\n]*", block, re.M)
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

    workflow = (
        _pathlib_for_canary() / ".github" / "workflows" / "nightly-jax-canary.yml"
    ).read_text(encoding="utf-8")

    starts = [
        (m.start(), m.group(1))
        for m in re.finditer(r"\n  ([a-z-]+):\n    name: ", workflow)
    ]
    assert starts, "no jobs found; the workflow layout this test reads has moved"
    bounds = [pos for pos, _ in starts] + [len(workflow)]
    for index, (_, job) in enumerate(starts):
        block = workflow[bounds[index]:bounds[index + 1]]
        # ONE STEP AT A TIME, so the setting read is the one that step runs
        # under. A job-level `env:` and another step's `env:` both live in the
        # same block, and matching across them would let an x64-off step
        # somewhere else certify an x64-on canary run.
        cells = []
        for step in re.split(r"\n      - (?=name:|uses:|run:)", block)[1:]:
            if not re.search(r"^\s*run:[^\n]*tripwire_canary\.py", step, re.M):
                continue
            setting = re.search(r'JAX_ENABLE_X64:\s*"?([01])"?', step)
            cells.append(setting.group(1) if setting else "unset")
        assert cells, (
            f"the `{job}` leg does not run the canary at all"
        )
        assert any(cell in ("0", "unset") for cell in cells), (
            f"every `tripwire_canary.py` step on the `{job}` leg runs at "
            f"JAX_ENABLE_X64={cells}, and the eager constant sweep cannot "
            "find an unenumerated jax constant at x64=1: jax's own mask "
            "widens to int64, nothing of jax's narrows, and `unmatched` is "
            "empty by construction. That leg cannot go red for the reason "
            "`eager:unenumerated-jax-constant` exists"
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


def _pathlib_for_canary():
    import pathlib as _pathlib

    return _pathlib.Path(__file__).resolve().parents[1]
