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
# answer on stdout with no `end`.
#
# These drive a REAL child through a REAL pipe and kill it for real, because the
# whole defect lives in what block buffering does or does not flush — a mocked
# CompletedProcess cannot exhibit it. The `small` case is the reason it survived
# review: it is already safe, and it is the only size a hand-written fixture
# reaches for.

_WHEEL_CHILD = (
    "import os, sys, signal\n"
    "mode = sys.argv[1]\n"
    "print('version 1.3.4')\n"
    "print('answer sat')\n"
    "if mode == 'big':\n"
    "    [print(f'value x{i} {i}/1') for i in range(4000)]\n"
    "if mode == 'clean':\n"
    "    print('value x0 0/1'); print('end'); sys.exit(0)\n"
    "if mode == 'end_then_die':\n"
    "    print('end'); sys.stdout.flush(); os.kill(os.getpid(), signal.SIGKILL)\n"
    "if mode == 'no_end_exit_zero':\n"
    "    sys.stdout.flush(); sys.exit(0)\n"
    "os.kill(os.getpid(), signal.SIGKILL)\n"
)


def _wheel_child(monkeypatch, tmp_path, mode):
    """Route the wheel transport's spawn at a child we control, killed for real."""
    import subprocess
    import sys as _sys

    script = tmp_path / "fake_driver.py"
    script.write_text(_WHEEL_CHILD)
    real_run = subprocess.run

    def route(argv, **kw):
        return real_run(
            [_sys.executable, str(script), mode],
            input=kw.get("input", ""),
            capture_output=True,
            text=True,
            timeout=kw.get("timeout", 60),
        )

    monkeypatch.setattr(subprocess, "run", route)
    monkeypatch.setattr(solvers, "_cvc5_wheel_version", lambda: "1.3.4")
    return solvers._run_cvc5_wheel("(check-sat)\n(get-model)\n", 60.0)


def test_f4wheel_a_child_that_answered_then_died_is_not_a_verdict(
    monkeypatch, tmp_path
):
    """THE DEFECT. A 4000-term model flushes the pipe, so `answer sat` gets
    through while `end` does not. Before the guard this returned sat with 3570
    model values harvested from a dead process — a witness that cannot
    reproduce, behind a definitive REFUTED."""
    r = _wheel_child(monkeypatch, tmp_path, "big")
    assert r.answer == "failed"
    assert r.values == ()  # not one value taken from the corpse
    assert "not complete" in r.detail and "ABSENT" in r.detail


def test_f4wheel_a_small_model_was_always_caught_and_that_is_why_it_survived(
    monkeypatch, tmp_path
):
    """The same death, below the buffer: stdout never flushes, so the parent
    sees nothing and the pre-existing protocol check catches it. Pinned because
    it is the measurement that explains the miss, not because it was broken."""
    r = _wheel_child(monkeypatch, tmp_path, "small")
    assert r.answer == "failed"
    assert "protocol violation" in r.detail


def test_f4wheel_the_terminator_tell_fires_when_the_exit_code_cannot(
    monkeypatch, tmp_path
):
    """Exit 0, terminator absent. `returncode` is blind here; `end` is not."""
    r = _wheel_child(monkeypatch, tmp_path, "no_end_exit_zero")
    assert r.answer == "failed"
    assert "ABSENT" in r.detail


def test_f4wheel_the_exit_code_tell_fires_when_the_terminator_cannot(
    monkeypatch, tmp_path
):
    """Terminator present, then death. `end` is blind here; `returncode` is not.
    With the one above, this is why the guard reads two tells and not one:
    neither is derived from the other, and each covers the other's blind spot."""
    r = _wheel_child(monkeypatch, tmp_path, "end_then_die")
    assert r.answer == "failed"
    assert "terminator present" in r.detail


def test_f4wheel_a_healthy_run_is_untouched(monkeypatch, tmp_path):
    """Cry-wolf floor: the guard must not cost a clean sat its model."""
    r = _wheel_child(monkeypatch, tmp_path, "clean")
    assert r.answer == "sat"
    assert r.values == (("x0", "0/1"),)
    assert r.detail == ""
