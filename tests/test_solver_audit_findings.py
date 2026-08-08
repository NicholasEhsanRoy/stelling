# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Permanent regressions for the solver-layer audit findings (F1–F7).

Each construction from the fresh-context audit of the escalation layer,
re-derived in-repo: the out-of-box "witness" (F1, unsound), sat with no
usable model (F2), duplicate/undeclared model names (F3), the crashed run
behind an unsat (F4), the constants-only refutation and the internal-error
stamp drop (F5), the invoked=False tuple element (F6), and the empty
portfolio config (F7). All run without a real solver except the one
real-portfolio F5 case, which skips when the wheels are absent.
"""

from __future__ import annotations

import math
import os
import subprocess
from fractions import Fraction

import pytest

from stelling import _optional, solvers
from stelling.propagate import propagate
from stelling.solvers import (
    NO_USABLE_MODEL,
    EmissionInfidelityError,
    SolverConfig,
    escalate,
    make_solver_verdict,
)
from stelling.verdict import Stamp, StampError, solver_absent
from test_obligation_slice import BOOL, any_eqn, close, eqn, lit, square_query, var
from test_solver_dispatch import VERSIONS, fake_solver
from test_verdict import full_stamp_kwargs, invoked_stamp


def true_over_box_query():
    """x in [-1, 1], assert(x*x <= 1.0): TRUE at every point of the box in
    ℝ — the only honest escalated verdicts are VERIFIED or UNKNOWN; the
    outward-rounded interval straddles, so it escalates."""
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, -1.0, 1.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def constants_only_query():
    """assert(0.1 + 0.2 <= nextafter(fl(0.1)+fl(0.2), -inf)): no declared
    inputs at all; false in ℝ (the exact dyadic sum exceeds the bound one
    ulp below the rounded sum), exactly decidable, but the outward-rounded
    interval straddles — so it escalates with an empty input list."""
    c1, c2 = 0.1, 0.2
    bound = math.nextafter(c1 + c2, -math.inf)
    t, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    return close(
        [
            eqn("add", [lit(c1), lit(c2)], t),
            eqn("le", [t, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def run_fake(monkeypatch, tmp_path, query, body, name):
    fake = fake_solver(tmp_path, body, name)
    monkeypatch.setenv("STELLING_CVC5", fake)
    p = propagate(query)
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(query, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    (record,) = esc.records
    return p, esc, record


MODEL = 'print("sat")\nprint("(")\nprint({0})\nprint(")")'


# --- F1 (UNSOUND): witness box membership -------------------------------------


@pytest.mark.parametrize(
    "value_text", ["(/ 3 2)", "(- (/ 3 2))"], ids=["above-hi", "below-lo"]
)
def test_f1_out_of_box_witness_raises_never_refutes(monkeypatch, tmp_path, value_text):
    # the predicate is TRUE over the whole declared box; a model outside
    # the box refutes nothing about it — the box constraints were part of
    # the emitted problem, so an escaping model is emission infidelity.
    fake = fake_solver(
        tmp_path,
        MODEL.format(f'"  (define-fun x0 () Real {value_text})"'),
        "cvc5-outofbox",
    )
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = true_over_box_query()
    p = propagate(q)
    with pytest.raises(EmissionInfidelityError) as info:
        escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    assert "escapes the declared box" in str(info.value)
    assert "bound" in str(info.value)


def test_f1_membership_is_closed_an_endpoint_witness_is_accepted(monkeypatch, tmp_path):
    # x in [0,1] ⊢ x > 0 is violated exactly at the closed endpoint 0; the
    # membership check must be closed-interval, or endpoint witnesses die
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    _, esc, record = run_fake(
        monkeypatch, tmp_path, q,
        MODEL.format('"  (define-fun x0 () Real 0.0)"'),
        "cvc5-endpoint",
    )
    assert record.outcome == "violated-witness"
    assert record.witness.values == (("x0", "0"),)


def test_f1_completed_dontcare_values_are_also_membership_checked(monkeypatch, tmp_path):
    # a partial model (one of two inputs) still refutes after completion —
    # and the completed value passes the same membership gate. y rides into
    # the slice through y*0 (so it is declared in the script) but cannot
    # affect the predicate: a genuine per-variable don't-care.
    x, y, s, zy, s2, pred, out = (
        var(0), var(1), var(2), var(3), var(4), var(5, BOOL), var(6, BOOL),
    )
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            any_eqn(y, 0.0, 1.0),
            eqn("mul", [x, x], s),
            eqn("mul", [y, lit(0.0)], zy),
            eqn("add", [s, zy], s2),
            eqn("le", [s2, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    _, esc, record = run_fake(
        monkeypatch, tmp_path, q,
        MODEL.format('"  (define-fun x0 () Real (/ 7 4))"'),
        "cvc5-partial",
    )
    assert record.outcome == "violated-witness"
    values = dict(record.witness.values)
    assert values["x0"] == "7/4"
    assert Fraction(0) <= Fraction(values["x1"]) <= Fraction(1)  # completed in-box
    assert any("don't-care" in n for n in record.notes)


# --- F2: sat with no usable model ---------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        'print("sat")',  # torn run: no model at all
        MODEL.format('"  (define-fun x0 ((y Real)) Real 1.75)"'),  # wrong arity
    ],
    ids=["no-model", "wrong-arity"],
)
def test_f2_sat_without_usable_model_degrades_quoted_never_raises(
    monkeypatch, tmp_path, body
):
    p, esc, record = run_fake(
        monkeypatch, tmp_path, square_query(), body, "cvc5-torn"
    )
    assert record.outcome == "unknown"
    assert NO_USABLE_MODEL in record.detail
    assert any(NO_USABLE_MODEL in n for n in record.notes)
    assert len(record.invocations) == 1  # the invocation happened; stamped
    v = make_solver_verdict(square_query(), p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"


# --- F3: malformed models are screened, never laundered -----------------------


def test_f3_conflicting_duplicate_definitions_make_the_model_unusable(
    monkeypatch, tmp_path
):
    body = MODEL.format(
        '"  (define-fun x0 () Real 1.0)\\n  (define-fun x0 () Real (/ 7 4))"'
    )
    p, esc, record = run_fake(monkeypatch, tmp_path, square_query(), body, "cvc5-dup")
    assert record.outcome == "unknown"  # sort order must not pick a survivor
    assert "conflicting duplicate definitions" in record.detail
    assert any("conflicting duplicate definitions" in n for n in record.notes)
    assert make_solver_verdict(square_query(), p, esc, **VERSIONS).status == "UNKNOWN"


def test_f3_identical_duplicate_definitions_pass_with_a_note(monkeypatch, tmp_path):
    body = MODEL.format(
        '"  (define-fun x0 () Real (/ 7 4))\\n  (define-fun x0 () Real (/ 7 4))"'
    )
    _, esc, record = run_fake(monkeypatch, tmp_path, square_query(), body, "cvc5-dup2")
    assert record.outcome == "violated-witness"
    assert record.witness.values == (("x0", "7/4"),)
    assert any("identical" in n and "collapsed" in n for n in record.notes)


def test_f3_undeclared_names_are_ignored_and_disclosed(monkeypatch, tmp_path):
    body = MODEL.format(
        '"  (define-fun zz () Real 4.0)\\n  (define-fun x0 () Real (/ 7 4))"'
    )
    _, esc, record = run_fake(monkeypatch, tmp_path, square_query(), body, "cvc5-zz")
    assert record.outcome == "violated-witness"  # the valid value stands …
    assert record.witness.values == (("x0", "7/4"),)
    assert any("undeclared" in n and "zz" in n for n in record.notes)  # … disclosed


# --- F4: a crashed run must never be an undisclosed verdict -------------------


def test_f4_unsat_with_segfault_banner_and_exit_134_is_unknown_quoted(
    monkeypatch, tmp_path
):
    body = (
        'import os\nprint("unsat")\nprint("Segmentation fault (core dumped)")\n'
        "sys.stdout.flush()\nos._exit(134)"
    )
    p, esc, record = run_fake(monkeypatch, tmp_path, square_query(), body, "cvc5-dying")
    assert record.outcome == "unknown"  # never a VERIFIED from a crashed run
    all_text = " ".join([record.detail, *record.notes])
    assert "134" in all_text
    assert "Segmentation fault" in all_text
    assert len(record.invocations) == 1
    assert make_solver_verdict(square_query(), p, esc, **VERSIONS).status == "UNKNOWN"


def test_f4_tolerated_get_model_error_shape_stands_with_disclosure(
    monkeypatch, tmp_path
):
    # the one tolerated non-clean shape: unsat, then the get-model error
    # (the script ends with (get-model) by design), nonzero exit — the
    # answer stands, and the noise is quoted in the notes
    body = (
        'print("unsat")\n'
        "print('(error \"cannot get model unless after a SAT or UNKNOWN "
        "response.\")')\n"
        "sys.exit(1)"
    )
    p, esc, record = run_fake(monkeypatch, tmp_path, square_query(), body, "cvc5-gm")
    assert record.outcome == "discharged"
    assert any("tolerated get-model noise" in n for n in record.notes)
    assert any("cannot get model" in n for n in record.notes)  # quoted verbatim
    v = make_solver_verdict(square_query(), p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert any("tolerated get-model noise" in n for n in v.notes)


def test_f4_unsat_with_arbitrary_trailing_stdout_is_not_tolerated(
    monkeypatch, tmp_path
):
    body = 'print("unsat")\nprint("some trailing banner")'
    p, esc, record = run_fake(monkeypatch, tmp_path, square_query(), body, "cvc5-noisy")
    assert record.outcome == "unknown"
    assert any("some trailing banner" in n for n in record.notes)


# --- F5: constants-only refutation; internal errors keep their stamps ---------


def test_f5_constants_only_refutation_is_refuted_not_a_stamperror_degrade(
    monkeypatch, tmp_path
):
    q = constants_only_query()
    p, esc, record = run_fake(
        monkeypatch, tmp_path, q, 'print("sat")\nprint("(")\nprint(")")',
        "cvc5-empty-model",
    )
    assert record.outcome == "violated-constant"
    assert record.witness is None  # no fabricated witness values
    assert len(record.invocations) == 1
    assert "StampError" not in record.detail
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "REFUTED"
    assert v.obligations[0].status == "violated-over-set"
    assert "constant refutation" in v.obligations[0].detail
    assert isinstance(v.stamp.solver, tuple) and len(v.stamp.solver) == 1
    rendered = v.render()
    assert "definitely false" in rendered  # the set-level language, accurate here
    assert "witness for assert" not in rendered  # and no witness block


@pytest.mark.skipif(
    not (_optional.available("z3") and _optional.available("cvc5")),
    reason="needs both real solver wheels",
)
def test_f5_constants_only_refutation_real_portfolio_stamps_both(monkeypatch):
    monkeypatch.delenv("STELLING_CVC5", raising=False)
    q = constants_only_query()
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(q, p, SolverConfig(timeout_ms=30_000))
    (record,) = esc.records
    assert record.outcome == "violated-constant"
    assert len(record.invocations) == 2  # both real invocations stamped
    assert {s.name for s in record.invocations} == {"cvc5", "z3"}
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "REFUTED"
    assert isinstance(v.stamp.solver, tuple) and len(v.stamp.solver) == 2


def test_f5_internal_error_degrade_keeps_invocation_stamps(monkeypatch, tmp_path):
    fake = fake_solver(
        tmp_path, MODEL.format('"  (define-fun x0 () Real (/ 7 4))"'), "cvc5-ok"
    )
    monkeypatch.setenv("STELLING_CVC5", fake)
    # the replay engine now lives behind the single validator in
    # stelling.obligation; a plain RuntimeError from it is not a
    # ReplayError, so it escapes the validator as an internal error
    from stelling import obligation

    monkeypatch.setattr(
        obligation, "evaluate_predicate",
        lambda sl, values: (_ for _ in ()).throw(RuntimeError("boom-internal")),
    )
    q = square_query()
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    (record,) = esc.records
    assert record.outcome == "unknown"
    assert record.detail.startswith("escalation attempted; internal error")
    assert "boom-internal" in record.detail
    assert len(record.invocations) == 1  # the stamp survives the error
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    # the verdict must NEVER claim "no solver invoked" over dropped stamps
    assert isinstance(v.stamp.solver, tuple) and len(v.stamp.solver) == 1


# --- F6: no absence element inside the invocation tuple -----------------------


def test_f6_invoked_false_element_in_solver_tuple_is_rejected():
    kwargs = full_stamp_kwargs()
    kwargs["solver"] = (invoked_stamp("cvc5"), solver_absent("did not run"))
    with pytest.raises(StampError, match="invoked=False"):
        Stamp(**kwargs)


def test_f6_tuple_render_never_doubles_a_reason():
    kwargs = full_stamp_kwargs()
    kwargs["solver"] = (invoked_stamp("cvc5", reason="answered unsat in 5ms"),)
    rendered = Stamp(**kwargs).render()
    assert rendered.count("answered unsat in 5ms") == 1


# --- F7: an empty portfolio is a config error, not a missing install ----------


def test_f7_empty_only_is_rejected_at_config_validation():
    with pytest.raises(ValueError, match="empty portfolio"):
        SolverConfig(timeout_ms=100, only=())


# --- F4-wheel: the crashed run, on the transport that never received F4 -------
#
# `_make_run_cvc5_binary` has refused a crashed run since F4. `_run_cvc5_wheel`
# did not, and the shape is transport-specific: the driver prints `answer` and
# THEN walks the model through native `getValue`, so a death in there leaves the
# answer on stdout with no terminator.
#
# WHAT A MOCK CAN AND CANNOT DO. A mocked `CompletedProcess` fully exhibits the
# DEFECT — `test_f4wheel_a_mock_exhibits_the_defect_it_cannot_explain` drives it
# and gets sat with 3570 values out of a returncode of -9. What a mock cannot
# show is the EXPLANATION: why a real killed child usually loses its output and
# sometimes does not. That is a fact about pipe buffering, so the tests that
# claim it spawn a real child, kill it for real, and PIN THE ENVIRONMENT the
# child runs in — because the buffering threshold is `PYTHONUNBUFFERED`'s to
# change, and an ambient environment variable must never decide whether a test
# passes.

_WHEEL_CHILD = (
    "import os, sys, signal\n"
    "mode = sys.argv[1]\n"
    "print('version 1.3.4')\n"
    "print('answer sat')\n"
    "if mode == 'big':\n"
    "    [print(f'value x{i} {i}/1') for i in range(4000)]\n"
    "if mode == 'small':\n"
    "    print('value x0 0/1'); print('value x1 1/1')\n"
    "if mode == 'clean':\n"
    "    print('value x0 0/1'); print('end 1'); sys.exit(0)\n"
    "if mode == 'clean_no_values':\n"
    "    print('end 0'); sys.exit(0)\n"
    "if mode == 'clean_opaque_named_end':\n"
    "    print('opaque x0 end'); print('end 1'); sys.exit(0)\n"
    "if mode == 'clean_value_named_end':\n"
    "    print('value end 1/1'); print('end 1'); sys.exit(0)\n"
    "if mode == 'clean_mixed':\n"
    "    print('value x0 0/1'); print('opaque x1 (root 2)')\n"
    "    print('end 2'); sys.exit(0)\n"
    "if mode == 'clean_but_exit_1':\n"
    "    print('value x0 0/1'); print('end 1'); sys.exit(1)\n"
    "if mode == 'both_tells_blind':\n"
    "    print('value x0 0/1'); sys.stdout.flush()\n"
    "    os.write(1, b'end of the resource limit\\n'); os._exit(0)\n"
    "if mode == 'end_early_then_more':\n"
    "    print('value x0 0/1'); print('end 1')\n"
    "    print('value x1 1/1'); print('value x2 2/1'); sys.exit(0)\n"
    "if mode == 'answer_twice':\n"
    "    print('value x0 0/1'); print('answer unsat'); print('end 1')\n"
    "    sys.exit(0)\n"
    "if mode == 'partial_last_line':\n"
    "    print('value x0 0/1'); print('end 1')\n"
    "    sys.stdout.write('value x1 1'); sys.exit(0)\n"
    "if mode == 'short_walk':\n"
    "    print('value x0 0/1'); print('value x1 1/1'); print('end 3')\n"
    "    sys.exit(0)\n"
    "if mode == 'end_then_die':\n"
    "    print('end 0'); sys.stdout.flush(); os.kill(os.getpid(), signal.SIGKILL)\n"
    "if mode == 'no_end_exit_zero':\n"
    "    sys.stdout.flush(); sys.exit(0)\n"
    "os.kill(os.getpid(), signal.SIGKILL)\n"
)


_REAL_SPAWN = subprocess.run  # pristine: two calls in one test must not nest


def _wheel_child(monkeypatch, tmp_path, mode, *, buffered=None, seen=None):
    """Route the wheel transport's spawn at a child we control, killed for real.

    ``buffered`` pins the child's stdout buffering instead of inheriting it:
    True forces block buffering (``PYTHONUNBUFFERED`` cleared), False forces
    none. ``None`` inherits the ambient environment, which is what production
    does — `_run_cvc5_wheel` spawns with no ``env=``.
    """
    import sys as _sys

    script = tmp_path / "fake_driver.py"
    script.write_text(_WHEEL_CHILD)
    env = None
    if buffered is not None:
        env = dict(os.environ)
        env.pop("PYTHONUNBUFFERED", None)
        if not buffered:
            env["PYTHONUNBUFFERED"] = "1"

    def route(argv, **kw):
        proc = _REAL_SPAWN(
            [_sys.executable, str(script), mode],
            input=kw.get("input", ""),
            capture_output=True,
            text=True,
            timeout=kw.get("timeout", 60),
            env=env,
        )
        if seen is not None:
            seen["stdout"] = proc.stdout
            seen["returncode"] = proc.returncode
        return proc

    monkeypatch.setattr(subprocess, "run", route)
    monkeypatch.setattr(solvers, "_cvc5_wheel_version", lambda: "1.3.4")
    return solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)


def test_f4wheel_a_child_that_answered_then_died_is_not_a_verdict(
    monkeypatch, tmp_path
):
    """THE DEFECT. A 4000-term model overruns the pipe buffer, so `answer sat`
    gets through while the terminator does not. Before the guard this returned
    sat with 3570 model values harvested from a dead process."""
    seen = {}
    r = _wheel_child(monkeypatch, tmp_path, "big", buffered=True, seen=seen)
    assert seen["returncode"] == -9
    assert "answer sat" in seen["stdout"]  # the answer really did get through
    assert r.answer == "failed"
    assert r.values == ()  # not one value taken from the corpse
    assert "not complete" in r.detail and "ABSENT" in r.detail


def test_f4wheel_the_boundary_is_the_pipe_buffer_not_the_model_size(
    monkeypatch, tmp_path
):
    """WHY IT SURVIVED, and the correction to the first reading of it.

    The same two-value model, the same real SIGKILL, twice — the only thing
    that differs is the child's stdout buffering, which is an ENVIRONMENT
    variable's to decide:

    * block-buffered (the default): nothing is flushed, the parent sees an
      empty stdout, and the PRE-EXISTING protocol check catches it. This half
      pins that pre-existing check and nothing the crashed-child guard added.
    * unbuffered (`PYTHONUNBUFFERED=1`, standard in Docker images and CI;
      equally `python -u` or any per-line flush): `answer sat` is through at
      51 bytes and only the crashed-child guard catches it.

    So the boundary is `io.DEFAULT_BUFFER_SIZE` (8192) by default and ZERO when
    the child is unbuffered — not a property of the model size. `_run_cvc5_wheel`
    spawns with no `env=`, so production inherits whichever regime it is run in;
    these two pin both rather than assume either.
    """
    buffered_seen, unbuffered_seen = {}, {}
    blocked = _wheel_child(
        monkeypatch, tmp_path, "small", buffered=True, seen=buffered_seen
    )
    leaked = _wheel_child(
        monkeypatch, tmp_path, "small", buffered=False, seen=unbuffered_seen
    )

    assert buffered_seen["stdout"] == ""  # the whole model died in the buffer
    assert blocked.answer == "failed"
    assert "protocol violation" in blocked.detail

    # unbuffered, EVERYTHING the child wrote before dying is through — all 51
    # bytes of it, `answer sat` among them, four lines short of the terminator
    assert unbuffered_seen["stdout"] == (
        "version 1.3.4\nanswer sat\nvalue x0 0/1\nvalue x1 1/1\n"
    )
    assert len(unbuffered_seen["stdout"]) == 51
    assert leaked.answer == "failed"
    assert "ABSENT" in leaked.detail  # the new guard, not the old check

    # whichever regime, no value survives the child
    assert blocked.values == () and leaked.values == ()


def test_f4wheel_a_killed_child_is_never_a_verdict_in_any_environment(
    monkeypatch, tmp_path
):
    """The invariant that must not depend on how the machine is configured:
    spawned exactly as production spawns — no `env=`, ambient environment
    inherited — a killed child yields `failed` and no values, and this test
    does not care which of the two nets caught it."""
    r = _wheel_child(monkeypatch, tmp_path, "small")
    assert r.answer == "failed"
    assert r.values == ()


def test_f4wheel_a_mock_exhibits_the_defect_it_cannot_explain(monkeypatch):
    """A mocked `CompletedProcess` DOES exhibit the defect — the earlier claim
    that it could not was wrong, and it is what motivated a real-child fixture
    that then went environment-fragile. What a mock cannot exhibit is the
    buffering EXPLANATION: it is handed a stdout, so it cannot show why a real
    child's stdout is sometimes empty and sometimes not."""
    argv = ["python", "-m", "stelling._cvc5_driver"]
    stdout = "version 1.3.4\nanswer sat\n" + "".join(
        f"value x{i} {i}/1\n" for i in range(3570)
    )
    assert len(stdout.splitlines()) == 3572
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda a, **kw: subprocess.CompletedProcess(argv, -9, stdout, ""),
    )
    monkeypatch.setattr(solvers, "_cvc5_wheel_version", lambda: "1.3.4")
    r = solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)
    assert r.answer == "failed"
    assert r.values == ()
    assert "ABSENT" in r.detail


def test_f4wheel_the_terminator_tell_fires_when_the_exit_code_cannot(
    monkeypatch, tmp_path
):
    """Exit 0, terminator absent. `returncode` is blind here; the terminator
    is not."""
    r = _wheel_child(monkeypatch, tmp_path, "no_end_exit_zero")
    assert r.answer == "failed"
    assert "ABSENT" in r.detail


def test_f4wheel_the_exit_code_tell_fires_when_the_terminator_cannot(
    monkeypatch, tmp_path
):
    """Terminator present, then death. The terminator is blind here;
    `returncode` is not. With the one above, this is why the guard reads two
    tells and not one: neither is derived from the other, and each covers the
    other's blind spot."""
    r = _wheel_child(monkeypatch, tmp_path, "end_then_die")
    assert r.answer == "failed"
    assert "terminator present" in r.detail


def test_f4wheel_a_prefix_terminator_let_both_tells_go_blind_at_once(
    monkeypatch, tmp_path
):
    """THE HOLE IN "TWO TELLS". While the terminator was a token-prefix test
    (`line.split()[0] == "end"`), a child writing `end of the resource limit`
    RAW TO FD 1 — the shape native C++ output takes, bypassing Python's
    buffer — and exiting 0 defeated both tells simultaneously, which is
    exactly what two tells is supposed to make impossible. Measured before
    this fix, at base AND at the first version of the guard: sat, 1 value.
    The terminator must be the LAST line, so the prefix no longer counts."""
    seen = {}
    r = _wheel_child(monkeypatch, tmp_path, "both_tells_blind", seen=seen)
    assert seen["returncode"] == 0  # the exit-code tell really is blind
    assert seen["stdout"].splitlines()[-1] == "end of the resource limit"
    assert r.answer == "failed"
    assert r.values == ()
    assert "ABSENT" in r.detail


def test_f4wheel_values_written_after_the_terminator_are_refused(
    monkeypatch, tmp_path
):
    """Terminator, then two more values. Measured identically at base and at
    the first version of the guard: sat with 3 values, two of them written
    AFTER the run claimed to be over."""
    r = _wheel_child(monkeypatch, tmp_path, "end_early_then_more")
    assert r.answer == "failed"
    assert r.values == ()
    assert "ABSENT" in r.detail


def test_f4wheel_a_truncated_trailing_line_is_refused(monkeypatch, tmp_path):
    """A trailing line cut off mid-write. Measured identically at base and at
    the first version of the guard: sat, and the partial line silently
    dropped — the drop is invisible precisely because the parser skips a
    `value` line it cannot split into three."""
    r = _wheel_child(monkeypatch, tmp_path, "partial_last_line")
    assert r.answer == "failed"
    assert r.values == ()
    assert "ABSENT" in r.detail


def test_f4wheel_a_second_answer_line_is_a_protocol_violation(
    monkeypatch, tmp_path
):
    """Two `answer` lines. Measured identically at base and at the first
    version of the guard: `unsat`, CARRYING a value harvested under the
    earlier `answer sat` — a model from one answer attached to another. The
    terminator cannot see this (the stream ends correctly), so it takes its
    own check."""
    r = _wheel_child(monkeypatch, tmp_path, "answer_twice")
    assert r.answer == "failed"
    assert r.values == ()
    assert "protocol violation" in r.detail


def test_f4wheel_a_short_model_walk_is_refused(monkeypatch, tmp_path):
    """`end <count>` is the driver's own tally of the model lines it wrote,
    checked against what the parser parsed, so `complete` means the walk
    finished rather than that the driver reached its last statement. This
    shape is CONSTRUCTED — it is not an observed cvc5 bug — but it is the
    only reading of "complete" the word supports."""
    r = _wheel_child(monkeypatch, tmp_path, "short_walk")
    assert r.answer == "failed"
    assert r.values == ()
    assert "ABSENT" in r.detail


def test_f4wheel_a_complete_protocol_that_exits_nonzero_is_refused(
    monkeypatch, tmp_path
):
    """DELIBERATE TIGHTENING, beyond the crashed-sat case. A clean protocol —
    answer, model, terminator — that exits 1 returned sat with its model at
    base (measured); it is `failed` now. A nonzero exit is not a transport
    this layer will discharge OR refute on, matching the binary transport's
    F4 policy. The cost is UNKNOWN, never a flipped verdict."""
    r = _wheel_child(monkeypatch, tmp_path, "clean_but_exit_1")
    assert r.answer == "failed"
    assert r.values == ()
    assert "exit 1" in r.detail and "terminator present" in r.detail


@pytest.mark.parametrize(
    "mode,answer,values,nonrational",
    [
        ("clean", "sat", (("x0", "0/1"),), False),
        ("clean_no_values", "sat", (), False),
        # the two shapes a prefix test would have to worry about, and does not:
        # their FIRST token is `opaque`/`value`, and now their last line is the
        # real terminator anyway
        ("clean_opaque_named_end", "sat", (), True),
        ("clean_value_named_end", "sat", (("end", "1/1"),), False),
        ("clean_mixed", "sat", (("x0", "0/1"),), True),
    ],
)
def test_f4wheel_every_healthy_shape_is_still_accepted(
    monkeypatch, tmp_path, mode, answer, values, nonrational
):
    """CRY-WOLF FLOOR. The strict terminator, the count and the single-answer
    check must not cost a healthy run its model — including a model line whose
    own text is `end`."""
    r = _wheel_child(monkeypatch, tmp_path, mode)
    assert r.answer == answer
    assert r.values == values
    assert r.nonrational is nonrational
    assert r.detail == ""


@pytest.mark.skipif(
    not _optional.available("cvc5"), reason="needs the cvc5 wheel"
)
def test_f4wheel_the_real_driver_and_this_parser_agree_on_the_terminator(
    tmp_path,
):
    """DRIVER/PARENT COMPATIBILITY. `end <count>` couples the two modules, so
    pin that the real driver's real stdout satisfies the real parser — a bare
    `end` from a stale driver would degrade every run to UNKNOWN, which is the
    safe direction but still a break."""
    import sys as _sys

    script = (
        "(set-option :produce-models true)\n"
        "(set-logic QF_NRA)\n"
        "(declare-const x0 Real)\n"
        "(assert (<= 1 x0))\n"
        "(assert (<= x0 2))\n"
        "(check-sat)\n"
        "(get-model)\n"
    )
    proc = subprocess.run(
        [_sys.executable, "-m", "stelling._cvc5_driver"],
        input=script,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[-1] == "end 1", proc.stdout  # one declared const, one model line
    assert "answer sat" in lines


# --- F4-wheel-2: the writer and the reader disagreed about what a line is -----
#
# The driver sanitised model text with `replace("\n", " ")`; this parent read it
# with `str.splitlines()`, which breaks on TEN characters, not one. A value
# holding one of the other nine was ONE line to the writer and TWO to the
# reader, so the payload could supply the reader's LAST line — a forged
# terminator — while the child was truncated. With the child also exiting 0
# that defeats BOTH tells at once, which is the one thing two tells exists to
# prevent.
#
# REACHABILITY, measured on cvc5 1.3.4 rather than argued: cvc5 escapes every
# separator inside a model VALUE (`"a\u{b}b"`), so the channel the sanitiser
# guarded was already closed by cvc5's own printer. It does NOT escape a quoted
# SYMBOL, which the driver interpolated unsanitised — `|a<VT>b|` comes back
# raw. stelling names its own consts `x{k}`/`x{k}_{i}` (`obligation.py`), so no
# script this tool emits can carry one. Incompleteness in a deliberate guard,
# not a live exploit — and the guard is what has to hold when the next script
# emitter is not this one.

_SEPARATORS = (
    "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " ", " ",
)


def _wheel_stdout(monkeypatch, stdout, returncode=0):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda a, **kw: subprocess.CompletedProcess([], returncode, stdout, ""),
    )
    monkeypatch.setattr(solvers, "_cvc5_wheel_version", lambda: "1.3.4")
    return solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)


def test_f4wheel2_splitlines_breaks_on_ten_characters_not_one():
    """The fact the whole finding rests on, measured off Python rather than
    recalled: `str.splitlines()`'s separator set. If this ever grows, the
    driver's WHITELIST still holds (printable ASCII cannot join this set) —
    which is the reason it is a whitelist."""
    found = {c for c in map(chr, range(0x110000)) if len(("a" + c + "b").splitlines()) > 1}
    assert found == {"\n", "\r", *_SEPARATORS}
    # and every one of them is outside the protocol's alphabet
    assert all(not (" " <= c <= "~") for c in found)


@pytest.mark.parametrize("sep", _SEPARATORS)
def test_f4wheel2_a_payload_cannot_forge_the_terminator(monkeypatch, sep):
    """THE DEFECT. One line to the writer, two to the reader: the tail forges
    the terminator and its count, the child never wrote one, and it exits 0 so
    the other tell is blind too. Measured at base: `sat` with 1 value."""
    forged = f"version 1.3.4\nanswer sat\nvalue x0 1/1\nopaque x1 junk{sep}end 2\n"
    assert len(forged.splitlines()) == 5      # what the reader used to see
    assert len(forged.split("\n")) - 1 == 4   # what the writer actually wrote
    r = _wheel_stdout(monkeypatch, forged)
    assert r.answer == "failed"
    assert r.values == ()


def test_f4wheel2_the_same_payload_without_a_separator_is_the_control(monkeypatch):
    """POSITIVE CONTROL for the parametrisation above: identical bytes but for
    the separator. It was refused before this fix and is refused now, so the
    separator — not the shape — is what the test above is about."""
    r = _wheel_stdout(
        monkeypatch,
        "version 1.3.4\nanswer sat\nvalue x0 1/1\nopaque x1 junk end 2\n",
    )
    assert r.answer == "failed"
    assert "not complete" in r.detail and "ABSENT" in r.detail


@pytest.mark.parametrize("sep", _SEPARATORS)
def test_f4wheel2_the_reader_narrowing_refuses_on_its_own(monkeypatch, sep):
    """THE NARROWING, measured as its own mechanism rather than through the
    alphabet check that fires first. `split("\\n")` is the writer's boundary,
    so the poisoned record stays ONE record and the forged `end 2` is no longer
    the last line — the terminator check refuses it. `splitlines()` on the same
    bytes accepts it. Both readings are computed here, so the difference is a
    measurement and not a claim about which branch ran."""
    forged = f"version 1.3.4\nanswer sat\nvalue x0 1/1\nopaque x1 junk{sep}end 2\n"
    narrow = [ln for ln in forged.split("\n")][:-1]
    assert narrow[-1] != "end 2"          # refused: the terminator is absent
    assert forged.splitlines()[-1] == "end 2"  # accepted: the forgery lands


def test_f4wheel2_a_stale_driver_is_a_protocol_violation(monkeypatch):
    """FAIL CLOSED on a driver out of step with this parser. The two ship
    together, but a stale install is exactly what the driver's docstring says
    degrades to UNKNOWN; a byte outside the protocol's alphabet is now refused
    with that said, rather than interpreted."""
    r = _wheel_stdout(
        monkeypatch, "version 1.3.4\nanswer sat\nopaque x0 a\x0bb\nend 1\n"
    )
    assert r.answer == "failed"
    assert "alphabet" in r.detail


def test_f4wheel2_carriage_return_is_the_writers_alone_to_stop(monkeypatch):
    """WHY THE WRITER IS THE LOAD-BEARING HALF, and the measurement that
    settles widen-the-writer against narrow-the-reader.

    `capture_output=True, text=True` means Python's universal-newline decoding
    turns a `\\r` into a real `\\n` BEFORE this parent sees the string. By the
    time any rule here runs there is nothing left to detect: the bytes below
    are what a stale driver's `\\r` payload looks like after decoding, and this
    parent accepts them — correctly, because they are indistinguishable from a
    child that really wrote two records. No reader-side rule can close this;
    only `_cvc5_driver._tail` escaping the `\\r` can."""
    decoded = "version 1.3.4\nanswer sat\nvalue x0 1/1\nopaque x1 junk\nend 2\n"
    r = _wheel_stdout(monkeypatch, decoded)
    assert r.answer == "sat"  # the parent is blind here, and cannot not be
    # ...and the driver is not: nothing it writes can decode into that.
    from stelling import _cvc5_driver

    assert _cvc5_driver._tail("junk\rend 2") == "junk\\u{d}end 2"


@pytest.mark.parametrize(
    "sep", ("\n", "\r", *_SEPARATORS),
)
def test_f4wheel2_the_driver_escapes_every_separator_in_every_field(sep):
    """THE FIX, on the writer. Name, value and error text all go through the
    same whitelist, and a field can no longer become two records."""
    from stelling import _cvc5_driver

    for fn in (_cvc5_driver._token, _cvc5_driver._tail):
        out = fn(f"a{sep}b")
        assert sep not in out
        assert len(out.splitlines()) == 1
        assert len(out.split("\n")) == 1


def test_f4wheel2_the_driver_escapes_a_space_in_a_NAME_but_not_in_a_value():
    """The same disagreement one delimiter down: `value`/`opaque` lines are read
    with `split(maxsplit=2)`, so a space inside a NAME shifts the value into the
    name's field. Names are tokens; the trailing field is free text and keeps
    its spaces (`opaque x0 (root 2)` must survive)."""
    from stelling import _cvc5_driver

    assert _cvc5_driver._token("a b") == "a\\u{20}b"
    assert _cvc5_driver._tail("(root 2)") == "(root 2)"


def test_f4wheel2_printable_ascii_is_passed_through_untouched():
    """CRY-WOLF FLOOR for the whitelist: every character the protocol actually
    uses survives it, so no healthy field is rewritten."""
    from stelling import _cvc5_driver

    printable = "".join(map(chr, range(0x21, 0x7F)))
    assert _cvc5_driver._token(printable) == printable
    assert _cvc5_driver._tail(" " + printable) == " " + printable


@pytest.mark.skipif(not _optional.available("cvc5"), reason="needs the cvc5 wheel")
def test_f4wheel2_real_cvc5_emits_a_raw_separator_in_a_quoted_symbol():
    """REACHABILITY, measured on the real backend through the real driver.

    cvc5 escapes separators inside a model VALUE, so the channel the old
    sanitiser guarded was already closed by cvc5's printer. It does not escape
    a quoted SYMBOL — which the driver used to interpolate unsanitised. This
    pins both halves: the raw separator really does reach the driver, and the
    driver really does neutralise it. stelling names its own consts `x{k}`, so
    no script this tool emits can carry one; the guard is for the next emitter,
    not this one."""
    import sys as _sys

    vt = "\x0b"
    script = (
        "(set-option :produce-models true)\n(set-logic QF_LRA)\n"
        f"(declare-const |x0{vt}end 1| Real)\n"
        f"(assert (= |x0{vt}end 1| 3.0))\n(check-sat)\n(get-model)\n"
    )
    proc = subprocess.run(
        [_sys.executable, "-m", "stelling._cvc5_driver"],
        input=script, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert vt not in proc.stdout                       # escaped by the driver
    assert "\\u{b}" in proc.stdout                     # ...and disclosed, not dropped
    assert len(proc.stdout.splitlines()) == len(proc.stdout.split("\n")) - 1
    assert proc.stdout.splitlines()[-1] == "end 1"     # the terminator is the driver's


def test_f4wheel2_a_final_record_without_its_newline_is_not_a_record(monkeypatch):
    """FOUND BY THE FUZZER ON THE FIX, not reasoned into it. A record is
    `text + "\\n"` — that is what `print` writes — so a final record whose
    newline never got out is one the child did not finish writing. This parser
    accepted it: `end 4` with a matching count and exit 0 was a definite
    answer, under `splitlines()` and under `split("\\n")` alike. 86 of 86
    residual findings over 200k examples had this shape and no other."""
    body = "version 1.3.4\nanswer sat\nvalue x0 1/1\nend 1"
    assert _wheel_stdout(monkeypatch, body).answer == "failed"
    assert "not newline-terminated" in _wheel_stdout(monkeypatch, body).detail
    # NEGATIVE CONTROL: the same bytes, properly terminated, still answer.
    r = _wheel_stdout(monkeypatch, body + "\n")
    assert r.answer == "sat" and r.values == (("x0", "1/1"),) and r.detail == ""


def test_f4wheel2_property_fuzz_no_definite_answer_from_an_incomplete_run():
    """The spike's property, seeded and bounded so it runs in the suite: the
    transport returns a definite answer only if the child wrote a complete
    protocol AND exited 0 — with ground truth taken from what the WRITER
    emitted, never from what the reader parsed.

    Both generator changes that made the spike find anything are here:
    truncation at a line boundary as well as mid-write, and an exit code drawn
    INDEPENDENTLY of whether truncation happened. Measured on this fix: 0
    counterexamples over 200k examples across 10 seeds; measured at 0ad22bb
    with the same generator: 1428.
    """
    import random

    from stelling import _cvc5_driver

    seps = ["\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " "]
    rng = random.Random(20260807)
    unsound, crywolf, seen_sat = [], [], 0
    for _ in range(4000):
        answer = rng.choice(["sat", "unsat", "unknown"])
        recs = ["version 1.3.4", f"answer {answer}"]
        n = rng.randrange(0, 4)
        for _i in range(n):
            name = _cvc5_driver._token(rng.choice(["x0", "x1", "end", "x0 y"]))
            tail = rng.choice(["junk", "(root 2)", "end", ""])
            if rng.random() < 0.75:
                tail += rng.choice(seps) + f"end {rng.randrange(4)}"
            kind = "value" if rng.random() < 0.5 else "opaque"
            body = "1/1" if kind == "value" else tail
            recs.append(f"{kind} {name} {_cvc5_driver._tail(body)}")
        complete = rng.random() < 0.5
        if complete:
            recs.append(f"end {n}")
        stream = "".join(r + "\n" for r in recs)
        truncated = False
        roll = rng.random()
        if roll < 0.25:
            keep = rng.randrange(0, len(recs) + 1)
            truncated = keep < len(recs)
            stream = "".join(r + "\n" for r in recs[:keep])
        elif roll < 0.45 and stream:
            stream = stream[: rng.randrange(0, len(stream))]
            truncated = True
        rc = rng.choice([0, 0, 0, 1, -9])
        wrote_full = not truncated and complete
        # what `capture_output=True, text=True` hands the parent
        decoded = stream.replace("\r\n", "\n").replace("\r", "\n")
        proc = subprocess.CompletedProcess([], rc, decoded, "")
        real = subprocess.run
        subprocess.run = lambda a, _p=proc, **kw: _p
        old_version = solvers._cvc5_wheel_version
        solvers._cvc5_wheel_version = lambda: "1.3.4"
        try:
            r = solvers._run_cvc5_wheel("(check-sat)\n", 60.0)
        finally:
            subprocess.run = real
            solvers._cvc5_wheel_version = old_version
        definite = r.answer in ("sat", "unsat", "unknown")
        if definite and not (wrote_full and rc == 0):
            unsound.append((decoded, rc, r.answer))
        if wrote_full and rc == 0 and not definite:
            crywolf.append((decoded, r.answer, r.detail))
        if definite:
            seen_sat += 1
    assert unsound == [], unsound[:3]
    # ANTI-VACUITY: the property is not passing because nothing was accepted.
    assert seen_sat > 200, seen_sat
    assert crywolf == [], crywolf[:3]
