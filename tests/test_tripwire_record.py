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


def test_the_canary_reads_the_version_to_hash_map_in_three_states():
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
    """
    canary = _canary()

    matched = _Status(rule_hash="abc", known_hash="abc", jax_version="0.11.0")
    unread = _Status(rule_hash="abc", known_hash=None, jax_version="0.99.0")
    moved = _Status(rule_hash="abc", known_hash="xyz", jax_version="0.11.0")
    unreadable = _Status(rule_hash=None, known_hash="abc", jax_version="0.11.0")

    note, fatal = canary._hash_row(matched)
    assert (note, fatal) == ("as tested", False)

    note, fatal = canary._hash_row(unread)
    assert fatal is False, "a release with no row must not page the nightly"
    assert "NEVER BEEN READ" in note and "0.99.0" in note
    assert "nightly" in note, "the note must say why this is not a failure"

    note, fatal = canary._hash_row(moved)
    assert fatal is True, "a release contradicting its own row is fatal"
    assert "CONTRADICTS" in note and "xyz" in note and "0.11.0" in note

    note, fatal = canary._hash_row(unreadable)
    assert fatal is False, "an unreadable rule source claims nothing either way"

    # and the two loud states are not each other's words
    assert "CONTRADICTS" not in canary._hash_row(unread)[0]
    assert "NEVER BEEN READ" not in canary._hash_row(moved)[0]


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
