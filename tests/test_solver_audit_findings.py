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

import ast
import math
import os
import subprocess
import sys
from fractions import Fraction

import pytest

from stelling import _optional, reproduce, solvers
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
        # FORWARD THE TRANSPORT'S OWN SPAWN KWARGS. This used to name
        # `text=True` and `capture_output=True` itself, which made every test
        # routed through here a measurement of the FIXTURE's io choice rather
        # than the transport's — and pinned the fixture to a reading of the
        # child's stdout the transport had stopped doing. Only the argv and
        # the environment are the fixture's business; `seen["stdout"]` is
        # therefore the child's raw BYTES, which is what it was always
        # standing in for.
        proc = _REAL_SPAWN([_sys.executable, str(script), mode], env=env, **kw)
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
    assert b"answer sat" in seen["stdout"]  # the answer really did get through
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

    assert buffered_seen["stdout"] == b""  # the whole model died in the buffer
    assert blocked.answer == "failed"
    assert "protocol violation" in blocked.detail

    # unbuffered, EVERYTHING the child wrote before dying is through — all 51
    # bytes of it, `answer sat` among them, four lines short of the terminator
    assert unbuffered_seen["stdout"] == (
        b"version 1.3.4\nanswer sat\nvalue x0 0/1\nvalue x1 1/1\n"
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
        lambda a, **kw: subprocess.CompletedProcess(
            argv, -9, stdout.encode("utf-8"), b""
        ),
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
    assert seen["stdout"].splitlines()[-1] == b"end of the resource limit"
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


def test_f4wheel_the_terminator_must_be_the_LAST_line_not_merely_present(
    monkeypatch,
):
    """POSITION, separated from COUNT — the test above does not separate them.

    `…_values_written_after_the_terminator_are_refused` writes `end` early and
    then two MORE values, so the count inside the terminator no longer matches
    what this parser tallies. A rule that asked only *"is a matching `end <n>`
    ANYWHERE in the stream?"* refuses that stream for the wrong reason, and is
    therefore not distinguished by it. This one writes the terminator with the
    RIGHT final count and then one further record, so the only thing left to
    refuse it on is WHERE the terminator sits.

    MEASURED, real child, real bytes, in this worktree: the rule
    `complete = terminated and any(l == f"end {len(values) + opaques}"
    for l in lines)` — position dropped, count kept — passes the ENTIRE suite
    at this tip (2494 passed, 7 skipped, jax 0.11.0) and returns `sat` with
    `(('x0', '1/2'),)` on the stream below, where the shipped rule returns
    `failed`. That is a value harvested from a record the child wrote AFTER
    announcing the run was over.

    UNCHANGED FROM `9564728` — not a defect this branch introduced, and not a
    claim it made. It is closed here because this branch is the one that
    narrowed this reader, and a surviving mutant on the rule next door is
    worth a line more than it is worth a note.
    """
    prog = (
        "import sys\n"
        "sys.stdout.buffer.write("
        "b'version 1.3.4\\nanswer sat\\nend 1\\nvalue x0 1/2\\n')\n"
    )

    def route(argv, **kw):
        return _REAL_SPAWN([sys.executable, "-c", prog], **kw)

    monkeypatch.setattr(subprocess, "run", route)
    monkeypatch.setattr(solvers, "_cvc5_wheel_version", lambda: "1.3.4")
    r = solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)
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
    """Hand the parent the BYTES a child wrote.

    This used to hand it a `str` — what `text=True` had already decoded — so
    every case below was a measurement of a MODEL of the io layer rather than
    of the io layer itself. The parent decodes for itself now, so the fixture's
    job is to be the child's stdout and nothing more. A `str` here is encoded
    UTF-8, which is what `print` writes; pass `bytes` where the point is a byte
    the protocol's alphabet does not contain.
    """
    raw = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda a, **kw: subprocess.CompletedProcess([], returncode, raw, b""),
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


def test_f4wheel2_a_record_boundary_is_the_writers_alone_to_stop(monkeypatch):
    """WHY THE WRITER IS STILL THE LOAD-BEARING HALF — narrowed to what is
    actually true of it.

    THIS TEST NAMED `\\r` AND THAT IS NO LONGER THE CASE. It was called
    "carriage return is the writer's alone to stop", and its body handed the
    parent the string universal-newline decoding produced, asserting that shape
    is indistinguishable from a child that really wrote two records. The parent
    decodes for itself now and KEEPS a bare `\\r`, so that shape is refused here
    (`test_f4wheel3_the_reader_now_refuses_nine_of_the_ten_separators`).

    What survives is the claim about a REAL record boundary. A `\\n` inside a
    field is two records by definition, and so is a `\\r\\n` on the platform
    `README.md` names for both solver wheels. No rule on the reader's side
    could tell either from two records, which is why the whitelist on
    `_cvc5_driver._token`/`._tail` is load-bearing and not decorative."""
    r = _wheel_stdout(
        monkeypatch,
        "version 1.3.4\nanswer sat\nvalue x0 1/1\nopaque x1 junk\nend 2\n",
    )
    assert r.answer == "sat"  # the parent is blind here, and cannot not be
    # ...and the driver is not: nothing it writes can decode into that.
    from stelling import _cvc5_driver

    assert _cvc5_driver._tail("junk\nend 2") == "junk\\u{a}end 2"
    assert _cvc5_driver._tail("junk\rend 2") == "junk\\u{d}end 2"


def test_f4wheel2_universal_newline_decoding_is_measured_not_modelled():
    """THE MEASUREMENT THAT CONDEMNED `text=True`, kept because it is the
    reason the parent stopped using it. Raw BYTES go in; `text=True` maps BOTH
    `\\r\\n` and a bare `\\r` to `\\n`, so by the time any rule in the transport
    ran there was no `\\r` left to test for.

    The transport's own decoder is measured on the SAME bytes beside it. The
    difference is one character, and that one character is the whole of the
    difference between backstopping eight separators and nine.

    THE FRACTION ITSELF MOVED HERE AND IS PINNED ELSEWHERE. What stood below
    was `…_the_alphabet_backstop_refuses_eight_of_the_ten`, which fed the
    parent a MODEL of this decoding (`_as_the_parent_receives_it`) rather than
    bytes. Its successor spawns a real child and reads nine:
    `test_f4wheel3_the_reader_now_refuses_nine_of_the_ten_separators`."""
    argv = [sys.executable, "-c",
            r"import sys; sys.stdout.buffer.write(b'a\rb\r\nc\nd')"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert proc.stdout == "a\nb\nc\nd"  # two \r gone, both became \n
    raw = subprocess.run(argv, capture_output=True, timeout=60).stdout
    assert raw == b"a\rb\r\nc\nd"
    assert solvers._decode_child_stream(raw) == "a\rb\nc\nd"  # the bare \r kept


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
        # THE BYTES THE CHILD WROTE, not a model of what a decoder does to
        # them. This line used to be `stream.replace("\r\n", "\n").replace(
        # "\r", "\n")` — the parent's `text=True` decoding, restated here. The
        # parent decodes for itself now, so restating it would be modelling
        # the thing under test; and the restatement was a no-op in any case,
        # since every record above goes through `_cvc5_driver._tail`, which
        # leaves no `\r` in the stream to translate.
        proc = subprocess.CompletedProcess([], rc, stream.encode("utf-8"), b"")
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
            unsound.append((stream, rc, r.answer))
        if wrote_full and rc == 0 and not definite:
            crywolf.append((stream, r.answer, r.detail))
        if definite:
            seen_sat += 1
    assert unsound == [], unsound[:3]
    # ANTI-VACUITY: the property is not passing because nothing was accepted.
    assert seen_sat > 200, seen_sat
    assert crywolf == [], crywolf[:3]


# --- F4-wheel-3: the `\r` hole, closed on the READER, and what it costs -------
#
# `text=True` performs UNIVERSAL-NEWLINE decoding, which turns a `\r` into a
# real `\n` before any rule in `_run_cvc5_wheel` can run. So a stale driver —
# one out of step with this parser, i.e. a partial upgrade — could write
# `opaque x1 j\rend 2\r`, no terminator record of its own anywhere, and get a
# DEFINITE answer with a model. `sat` was the spurious-violation direction;
# `unsat` off the same corpse is the DISCHARGE direction, and that one has no
# downstream backstop at all.
#
# Three readers were measured over the same real children before this landed:
#
#   case                             (a) text=True  (b) decode  (c) decode+CRLF
#   healthy POSIX   `\n`             sat            sat         sat
#   healthy Windows `\r\n`           sat            FAILED      sat
#   stale `\r`, LF body              SAT (!)        failed      failed
#   stale `\r`, CRLF body            SAT (!)        failed      failed
#   stale `\x0b`                     failed         failed      failed
#   separators refused (LF stale)    8 of 10        9 of 10     9 of 10
#   child writes invalid UTF-8       RAISES         failed      failed
#
# (c) is what shipped: identical to (a) on both healthy children, strictly
# stronger on both stale ones, and with no platform coupling — `README.md`
# names Windows for both solver wheels, which is what rules (b) out. It puts
# back BY HAND the one translation `text=True` was doing and nothing else.
#
# EVERY TEST IN THIS BLOCK MEASURES THE PARENT'S OWN IO CHOICE, so none of them
# may name it: `_wheel_real_child` forwards whatever `_run_cvc5_wheel` asked
# `subprocess.run` for. `_wheel_child` above pins `text=True` in the shim
# itself, which would make these tests a measurement of the fixture.


def _stale_child(answer: str, payload: bytes) -> str:
    """A stale driver: raw bytes, one write per record, NO terminator record.

    Whatever `end <n>` the parent ends up seeing is inside the payload, so
    accepting one is accepting a terminator the child never wrote.

    THE ANSWER RECORD IS HOISTED INTO A LOCAL, AND THAT IS NOT A STYLE
    CHOICE. It used to be spelled ``f"w({f'answer {answer}\\n'.encode()!r})\\n"``
    — a nested f-string whose ``\\n`` sits inside the OUTER f-string's
    expression part. That is PEP 701 syntax and it is Python 3.12+ only, while
    ``pyproject.toml`` declares ``requires-python = ">=3.10"`` and the sdist
    ships ``/tests``. Measured on this box at 53f9f84: every tracked ``.py``
    parses under 3.10 and 3.11 EXCEPT this one, and ``python -m pytest`` on a
    3.11 interpreter came back ``6 skipped, 1 error`` — a collection error, so
    the suite examined nothing. ``src/`` was and is clean on both.

    The obvious cheap guard does not work and was driven rather than assumed:
    ``ast.parse(src, feature_version=(3, 10))`` on a 3.12 host PARSES this
    construct, so a same-interpreter floor check would have been vacuous
    against exactly this defect. Catching it needs a real floor interpreter,
    and NO JOB IN ``.github/workflows/`` RUNS ONE.

    THE SENTENCE THAT USED TO SAY SO NAMED THE WRONG LANE, and the correction
    is the smaller half. It read "every ``uv venv`` but the
    ``acceptance-any-pytree`` lane's takes the runner's default". Re-counted
    across both workflow files: SEVEN ``uv venv`` invocations, SIX of them
    bare. The one that is not bare is ``uv venv --python 3.12`` in the
    ``acceptance-reproducer`` job; ``acceptance-any-pytree``'s is bare like the
    rest. Cited by JOB NAME rather than by line, since these move.

    The load-bearing half is untouched by that, and is why the paragraph
    exists: 3.12 is not a floor interpreter, so pinning it changes nothing
    about this defect. Six lanes take whatever the runner image ships and the
    seventh takes a version above the floor, so which interpreter the release
    gate's suite runs is a property of the runner image and not of this
    repository, and 3.10 is exercised by no job at all.
    """
    answer_record = f"answer {answer}\n".encode()
    return (
        "import sys\n"
        "w = sys.stdout.buffer.write\n"
        "w(b'version 1.3.4\\n')\n"
        f"w({answer_record!r})\n"
        "w(b'value x0 1/2\\n')\n"
        f"w({payload!r})\n"
    )


_HEALTHY_CHILD = (
    "import sys\n"
    "sys.stdout.reconfigure(newline={nl!r})\n"
    "print('version 1.3.4')\n"
    "print('answer sat')\n"
    "print('value x0 1/2')\n"
    "print('end 1')\n"
)


def _wheel_real_child(monkeypatch, prog):
    """Spawn a real child, forwarding the transport's OWN spawn kwargs."""

    def route(argv, **kw):
        return _REAL_SPAWN([sys.executable, "-c", prog], **kw)

    monkeypatch.setattr(subprocess, "run", route)
    monkeypatch.setattr(solvers, "_cvc5_wheel_version", lambda: "1.3.4")
    return solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)


def test_f4wheel3_the_transport_asks_for_bytes_and_never_for_text(monkeypatch):
    """THE MECHANISM, pinned where it is chosen rather than only where it is
    felt. `text=`/`encoding=`/`universal_newlines=` all switch Python's
    universal-newline decoding on, and any of the three reopens the hole below
    without changing a line of the parser. The child's stdin gets bytes for the
    same reason: there is no text mode left to encode it."""
    seen = {}

    def route(argv, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(
            argv, 0, b"version 1.3.4\nanswer unsat\nend 0\n", b""
        )

    monkeypatch.setattr(subprocess, "run", route)
    monkeypatch.setattr(solvers, "_cvc5_wheel_version", lambda: "1.3.4")
    r = solvers._run_cvc5_wheel("(check-sat)\n", 60.0)
    assert r.answer == "unsat"
    assert isinstance(seen["input"], bytes)
    assert not {"text", "encoding", "errors", "universal_newlines"} & set(seen)


def test_f4wheel3_the_decode_puts_back_one_translation_and_only_one():
    """The whole of the reader-side change, as a function. `\\r\\n` is a record
    boundary on the platform `README.md` names for both solver wheels, so it
    has to survive; a BARE `\\r` is not a boundary anywhere `_cvc5_driver` runs,
    so it must reach the alphabet check intact instead of being silently
    promoted to one."""
    assert solvers._decode_child_stream(b"a\r\nb\n") == "a\nb\n"
    assert solvers._decode_child_stream(b"a\rb\n") == "a\rb\n"
    assert solvers._decode_child_stream(b"a\nb\n") == "a\nb\n"
    # and undecodable bytes become a character the alphabet check refuses,
    # rather than an exception out of the transport
    assert solvers._decode_child_stream(b"\xff") == "�"


@pytest.mark.parametrize("answer", ("sat", "unsat", "unknown"))
def test_f4wheel3_a_stale_carriage_return_child_is_refused_in_both_directions(
    monkeypatch, answer
):
    """THE DEFECT, and the direction that matters.

    Measured at `9564728`, real child, real bytes: `sat` with a model, and
    `unsat` off the identical corpse — a false DISCHARGE, which is the VERIFIED
    path and the one with no downstream backstop. `_require_valid_refutation`
    catches a bad `sat` by replaying the witness; nothing replays an `unsat`.
    """
    r = _wheel_real_child(monkeypatch, _stale_child(answer, b"opaque x1 j\rend 2\r"))
    assert r.answer == "failed"
    assert r.values == ()
    assert "alphabet" in r.detail


def test_f4wheel3_the_reader_now_refuses_nine_of_the_ten_separators(monkeypatch):
    """HOW FAR THE READER-SIDE BACKSTOP REACHES, pinned as a fraction. It was
    EIGHT: `\\r` was invisible to the alphabet check rather than admitted by it,
    because universal-newline decoding had already spent it.

    `\\n` is the one that remains, and it is excluded BY CONSTRUCTION — it is
    the protocol's own record boundary, so a writer that leaves one inside a
    field has written two records and there is nothing on this side to detect.
    That half is the writer's and always was.

    If this ever reads 8 again the reader lost a capability; if it reads 10 the
    reader started refusing its own record separator."""
    refused, definite = set(), set()
    for sep in ("\n", "\r", *_SEPARATORS):
        raw = sep.encode("utf-8")
        r = _wheel_real_child(
            monkeypatch, _stale_child("sat", b"opaque x1 j" + raw + b"end 2" + raw)
        )
        if r.answer == "failed" and "alphabet" in r.detail:
            refused.add(sep)
        if r.answer in ("sat", "unsat", "unknown"):
            definite.add(sep)
    assert refused == {"\r", *_SEPARATORS}
    assert definite == {"\n"}
    assert len(refused) == 9 and len(refused) + len(definite) == 10


def test_f4wheel3_a_crlf_inside_a_field_is_a_record_boundary_and_stays_one(
    monkeypatch,
):
    """THE RESIDUAL, STATED RATHER THAN LEFT TO BE DISCOVERED. The count above
    is over single characters. The two-character sequence `\\r\\n` is a genuine
    record boundary under this reader, exactly as it is under `text=True`, so a
    stale driver that puts one inside a FIELD is indistinguishable from a
    Windows child that ended a record there — and is accepted, here as before.
    This is the `\\n` row of the table wearing a second spelling, not a new
    hole: it is byte-for-byte what the shipped reader did, so nothing regressed
    and nothing improved. The writer's whitelist is what closes it."""
    r = _wheel_real_child(
        monkeypatch, _stale_child("sat", b"opaque x1 j\r\nend 2\r\n")
    )
    assert r.answer == "sat"  # unchanged from `text=True`; the writer stops it
    from stelling import _cvc5_driver

    assert _cvc5_driver._tail("j\r\nend 2") == "j\\u{d}\\u{a}end 2"


@pytest.mark.parametrize(
    "nl, label", (("\n", "POSIX"), ("\r\n", "Windows"))
)
def test_f4wheel3_a_healthy_child_answers_identically_on_either_line_ending(
    monkeypatch, nl, label
):
    """THE CRY-WOLF FLOOR, and the measurement that chose (c) over (b).

    A raw `bytes.decode()` with no translation refuses the `\\r\\n` child
    outright — every record ends in a byte outside the protocol's alphabet.
    `README.md` names Windows as a platform both solver wheels install on, so
    that arm buys the ninth separator by breaking a healthy run. Same answer
    AND same values on both, or the repair is not the repair that was chosen.
    """
    r = _wheel_real_child(monkeypatch, _HEALTHY_CHILD.format(nl=nl))
    assert r.answer == "sat", label
    assert r.values == (("x0", "1/2"),), label
    assert r.detail == ""


def test_f4wheel3_invalid_utf8_fails_closed_instead_of_escaping(monkeypatch):
    """FAIL CLOSED WHERE THE TRANSPORT CANNOT DECIDE. A child writing a byte
    that is not UTF-8 used to raise `UnicodeDecodeError` straight out of this
    function — measured at `9564728`, uncaught. `escalate`'s catch-all degraded
    it to `unknown`, so the verdict direction was safe by accident and the
    transport's own contract (`answer` in a fixed set) was not being kept.
    It is a protocol violation, which is a thing this layer already knows how
    to say."""
    prog = (
        "import sys\n"
        "sys.stdout.buffer.write(b'version 1.3.4\\nanswer sat\\n"
        "value x0 1/2\\n\\xff\\nend 2\\n')\n"
    )
    r = _wheel_real_child(monkeypatch, prog)
    assert r.answer == "failed"
    assert r.values == ()
    assert "alphabet" in r.detail


def test_f4wheel3_the_one_measured_cry_wolf_case_is_a_child_stelling_never_ships(
    monkeypatch,
):
    """THE COST, NAMED. Arm (c) refuses a HEALTHY child whose records end in a
    bare `\\r`. No platform's `print` default produces that — `newline=None`
    writes `os.linesep`, which is `\\n` or `\\r\\n` and never `\\r` — and the
    driver never reconfigures the stream, which is asserted here rather than
    remembered, since an edit that added one would turn every run on every
    platform into UNKNOWN."""
    import inspect

    from stelling import _cvc5_driver

    r = _wheel_real_child(monkeypatch, _HEALTHY_CHILD.format(nl="\r"))
    assert r.answer == "failed"
    assert "alphabet" in r.detail
    assert "reconfigure" not in inspect.getsource(_cvc5_driver)


# --- F4-wheel-2, the sweep: the same class anywhere else a writer meets a
# --- reader. Three places round-trip text across a process or a splitter; none
# --- of them is live, and these pin why rather than leaving it read once.


@pytest.mark.parametrize("sep", ("\n", "\r", *_SEPARATORS))
def test_f4wheel2_sweep_the_binary_transport_tokenizes_separators_alike(sep):
    """The binary cvc5 leg splits with `splitlines()` and re-joins with "\\n"
    before parsing. That normalisation is a no-op for the reader it feeds:
    `_tokenize_sexpr` splits on `str.split()`, whose whitespace set already
    contains every one of these, so no separator can move a model value from
    one define-fun into another."""
    base = solvers._tokenize_sexpr("(define-fun x0 () Real 1)")
    assert solvers._tokenize_sexpr(
        f"(define-fun{sep}x0{sep}(){sep}Real{sep}1)"
    ) == base
    assert solvers._model_values_from_text(
        f"(define-fun{sep}x0{sep}(){sep}Real{sep}1)"
    ) == ((("x0", "1"),), False)


def test_f4wheel2_the_two_line_end_sets_are_not_nested():
    """THE WITHDRAWN CLAIM, PINNED SO IT CANNOT BE RE-ASSERTED.

    What stood here asserted that Python's statement-separator set is a strict
    SUBSET of `str.splitlines()`', which would mean `reproduce.py`'s
    `splitlines()`-based no-`import stelling` scan sees MORE line-starts than
    the tokenizer and can only cry wolf. It also could not have detected its
    own falsity: it parametrised over `splitlines()` separators ONLY, and its
    second assertion (`not is_python_line_end or is_splitlines_boundary`) was
    implied by its first (`is_splitlines_boundary`) on every one of them.

    Measured over the whole code-point range instead of assumed, both sides
    read off this interpreter. Neither set contains the other."""
    splits = {
        c for c in map(chr, range(0x110000))
        if len(("a" + c + "b").splitlines()) > 1
    }
    compiles = set()
    for cp in range(0x110000):
        ch = chr(cp)
        try:
            compile("x=1" + ch + "y=2", "<probe>", "exec")
        except (SyntaxError, ValueError):
            continue
        compiles.add(ch)

    # WHAT THESE FOUR ARE AND ARE NOT. The comment here read "each of the four
    # is independently falsifiable; none implies another". THE SECOND HALF WAS
    # FALSE, and is measured so rather than argued: enumerate all 4096
    # (splits, compiles) pairs over a six-symbol universe modelling this
    # alphabet, and ask for each assertion whether the other three entail it —
    #
    #   A2 & A3 & A4  =>  A1   IMPLIED, 0 counterexamples of 4096
    #   A1 & A3 & A4  =>  A2   IMPLIED, 0
    #   A1 & A2 & A4  =>  A3   independent, 3 counterexamples
    #   A1 & A2 & A3  =>  A4   IMPLIED, 0
    #
    # and the tightest form of the one that matters: A1 & A4 => A2 on its own,
    # 0 counterexamples, since A2 is A1 minus A1&A4 by set algebra. THREE of
    # the four are entailed by the rest; only A3 — the cry-wolf direction, the
    # one carrying the eight separators — is not. They stay as four because
    # four named facts read better than one derived one, but the redundancy is
    # written down instead of denied: a mutation that moved only what A2 says
    # could not redden this test THROUGH A2.
    assert compiles == {"\n", "\r", "#", ";"}         # A1
    assert compiles - splits == {"#", ";"}            # A2, a MISS is possible
    assert splits - compiles == set(_SEPARATORS)      # A3, the cry-wolf leg
    assert splits & compiles == {"\n", "\r"}          # A4


@pytest.mark.parametrize(
    "text, the_line_scan_sees_it",
    (
        # `;` carries a real statement past a line-start scan
        ("x = 1; import stelling\ny = 2\n", False),
        ("x = 1; from stelling.harness import assert_\n", False),
        ("x = 1; import stelling.solvers as s\n", False),
        ("if True: import stelling\n", False),
        # and the shape the line scan was already catching, still caught
        ("import stelling\n", True),
        ("from stelling import verdict\n", True),
    ),
)
def test_f4wheel2_the_emitter_refuses_an_import_a_line_scan_cannot_see(
    text, the_line_scan_sees_it
):
    """The counter-construction, and the emitter's answer to it.

    `reproduce.py`'s refusal is called *"a structural refusal at the point of
    emission, not a comment asking the next author to be careful"* — so its
    charter is a future edit of `_TEMPLATE`, not today's caller text (which
    `one_line` already funnels). A refusal a semicolon walks past does not
    discharge that charter, so the emitter now also walks the parse tree."""
    line_hits = [
        line for line in text.splitlines()
        if line.strip().startswith(("import stelling", "from stelling"))
    ]
    assert bool(line_hits) is the_line_scan_sees_it
    # ...and in every case there really is an import of the tool in the tree
    found = {
        alias.name for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module or "" for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.ImportFrom)
    }
    assert any(n == "stelling" or n.startswith("stelling.") for n in found)
    with pytest.raises(reproduce.ReproducerError, match="reaches back into"):
        reproduce._refuse_tool_import(text)


@pytest.mark.parametrize(
    "text",
    (
        "import json\nx = 1\n",
        # the tree walk is EXACT: an import inside a string literal is text,
        # not a statement, and the walk does not fire on it
        "S = 'x = 1; import stelling'\n",
        "from json import loads\nx = 1\n",
        "import json as stelling_j\n",
    ),
)
def test_f4wheel2_the_parse_tree_leg_does_not_cry_wolf(text):
    """Non-vacuity in the other direction: the added refusal has to accept
    text that carries no import of the tool, or it would refuse every
    reproducer and the test above would pass for the wrong reason."""
    reproduce._refuse_tool_import(text)


def test_f4wheel2_the_line_scan_is_kept_and_still_cries_wolf():
    """The line scan is not replaced by the tree walk: it fires on an
    `import stelling` inside a docstring, which the parse tree correctly does
    not report as an import at all. A false alarm at the point of emission is
    the cheap direction, so both legs stay and neither implies the other."""
    text = 'D = """\nimport stelling\n"""\n'
    assert not [
        node for node in ast.walk(ast.parse(text))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    with pytest.raises(reproduce.ReproducerError, match="reaches back into"):
        reproduce._refuse_tool_import(text)


def test_f4wheel2_unparseable_text_is_left_to_the_callers_own_compile():
    """The tree walk runs only when the text parses. When it does not, this
    function stays silent so the caller's `compile` reports it with the
    "does not parse" message the emitter has always given."""
    reproduce._refuse_tool_import("x = (\n")


def test_f4wheel2_sweep_the_z3_transport_has_no_record_protocol():
    """z3 runs in-process over the API: names and values arrive as objects, so
    there is no writer, no reader and no boundary to disagree about. Pinned
    structurally so a future move to a text transport has to face this."""
    import inspect

    src = inspect.getsource(solvers._run_z3)
    assert "splitlines" not in src and "subprocess" not in src
    assert "decl.name()" in src  # values come off the model API, not off text
