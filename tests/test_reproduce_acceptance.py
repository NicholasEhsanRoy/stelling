# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The reproducer's acceptance: emitted files, run as files, outside.

Every test here writes a reproducer to a temporary directory and RUNS IT
IN A SUBPROCESS whose working directory is not this repository and whose
import path cannot reach :mod:`stelling` at all (a blocker module on
``PYTHONPATH`` raises on any attempt). That is the only way to measure
the property the whole feature rests on — *this file does not import
stelling* — because a check performed in-process, with stelling already
imported, measures nothing.

The two targets are external code:

* **JAX-Fluids 0.2.1**, ``WENO5Base.smoothness`` — the same function the
  `square` row's acceptance uses (``test_square_acceptance_jaxfluids``),
  called verbatim from the installed package.
* **MADDENING**, ``HeatNode.update`` — the flagship, at the refuting
  configuration recorded in ``docs/verdict-ledger.md``: ``α = 1.0``,
  ``n_cells = 4``, ``dt = 0.1``, ``T ∈ [0, 100]^4`` in float32, node
  output ``101.0`` against a declared bound of ``100.0``.

and two constructed cases exercise the other two execution results:
:data:`~stelling.reproduce.DIVERGED` (a violation smaller than binary64
can represent) and :data:`~stelling.reproduce.UNREACHABLE` (a witness
excluded by a structural caller precondition).

The subjects live in ``tests/reproduce_subjects.py``, not here, and that
placement is itself a measured result: the emitted file imports the
module its TARGET lives in, this module imports stelling, and the first
version of this acceptance therefore had every reproducer stop at "the
target could not be imported (stelling is blocked)". The program module
and the harness module are separate in a real tree for the same reason.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

jax = pytest.importorskip("jax")


from stelling.preconditions import check  # noqa: E402
from stelling.reproduce import (  # noqa: E402
    CONFIRMED,
    DIVERGED,
    EXECUTION_MODES,
    NOT_EXECUTED_EXIT,
    RESULT_EXIT,
    SCHEMA,
    SIDECAR_KEYS,
    UNREACHABLE,
    Subject,
    write_reproducer,
)

TIMEOUT_MS = 60_000
BOX = (-1.0, 1.0)


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ── the subjects, imported from the module the reproducer will import ────────
#
# They live in tests/reproduce_subjects.py, not here, because the emitted
# file imports the module its TARGET lives in — and this module imports
# stelling. Measured, not anticipated: the first version of this file put
# the subjects here and every reproducer stopped at
# "the target ... could not be imported (ImportError: stelling is blocked)".
# A user's tree already has this shape; the acceptance now has it too.

from reproduce_subjects import (  # noqa: E402
    _Holder,
    absorbed_increment,
    underflowing_square,
    heat_node_max_principle,
    weight_pair_sum,
    weights_are_normalized,
    weno5_central_vs_neighbours,
)

NORMALIZED = (
    "both weights are produced by the same normalization w_i = a_i / sum(a), "
    "so they sum to exactly 1 by construction; the per-weight [0, 1] envelope "
    "is an algebraic consequence of that and is strictly wider than any "
    "caller's range. Structural, not a measured span"
)

NO_PRECONDITION = (
    "every five-point stencil of cell values in the declared range is one a "
    "caller can present to this stencil operator; no caller precondition "
    "narrows it"
)


WENO5 = Subject(
    name="weno5-central-is-smoothest",
    fn=weno5_central_vs_neighbours,
    relation="<=",
    declarations=tuple(((), "float64", BOX) for _ in range(5)),
    no_precondition_reason=NO_PRECONDITION,
)

HEAT = Subject(
    name="heatnode-maximum-principle",
    fn=heat_node_max_principle,
    relation="<=",
    declarations=(((4,), "float32", (0.0, 100.0)),),
    no_precondition_reason=(
        "the node's temperature state is declared over its own operating "
        "range [0, 100] and every point of it is a state a driven "
        "trajectory occupies; no caller precondition narrows it"
    ),
)

NO_CALLER_NARROWING = (
    "the envelope is a range of ordinary small nonnegative reals; nothing "
    "about a caller excludes any of it"
)

UNDERFLOW = Subject(
    name="underflowing-square",
    fn=underflowing_square,
    relation="<=",
    declarations=(((), "float32", (0.0, 2.0 ** -100)),),
    no_precondition_reason=NO_CALLER_NARROWING,
)

ABSORBED = Subject(
    name="absorbed-increment",
    fn=absorbed_increment,
    relation="<=",
    declarations=(((), "float64", (0.0, 2.0 ** -70)),),
    no_precondition_reason=NO_CALLER_NARROWING,
)

WEIGHTS = Subject(
    name="normalized-weight-pair",
    fn=weight_pair_sum,
    relation="<=",
    declarations=(((), "float64", (0.0, 1.0)), ((), "float64", (0.0, 1.0))),
    precondition=weights_are_normalized,
    precondition_reason=NORMALIZED,
)


# ── running an emitted file the way a user would ─────────────────────────────

_BLOCKER = textwrap.dedent(
    '''
    """Make `import stelling` impossible, so the acceptance measures it."""
    import sys


    class _Blocked:
        def find_module(self, name, path=None):
            return None

        def find_spec(self, name, path=None, target=None):
            if name == "stelling" or name.startswith("stelling."):
                raise ImportError(
                    "stelling is blocked: a reproducer that imports it is "
                    "the tool checking itself with the tool"
                )
            return None


    sys.meta_path.insert(0, _Blocked())
    '''
)


def _elsewhere(tmp_path):
    """A working directory that is not the repo, plus a path on which
    stelling cannot be imported and this test module can."""
    where = tmp_path / "elsewhere"
    where.mkdir(exist_ok=True)
    (where / "sitecustomize.py").write_text(_BLOCKER)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(where), os.path.dirname(os.path.abspath(__file__))]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONSTARTUP", None)
    return where, env


def _run(emission, tmp_path):
    where, env = _elsewhere(tmp_path)
    proc = subprocess.run(
        [sys.executable, emission.path],
        cwd=str(where), env=env, capture_output=True, text=True, timeout=600,
    )
    sidecar = None
    if os.path.exists(emission.sidecar_path):
        with open(emission.sidecar_path) as fh:
            sidecar = json.load(fh)
    return proc, sidecar


def _emit(subject, tmp_path, **kw):
    v = check(
        subject.harness, vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT_MS
    )
    assert v.status == "REFUTED", v.render()
    return v, write_reproducer(v, subject, str(tmp_path), **kw)


def _blocker_really_blocks(tmp_path):
    where, env = _elsewhere(tmp_path)
    probe = subprocess.run(
        [sys.executable, "-c", "import stelling"],
        cwd=str(where), env=env, capture_output=True, text=True, timeout=120,
    )
    return probe.returncode != 0 and "blocked" in probe.stderr


# ── the control on the control ───────────────────────────────────────────────


def test_the_acceptance_environment_really_cannot_import_stelling(tmp_path):
    """Every acceptance below is worthless if this is not true, and the
    venv has stelling installed, so it must be measured rather than
    assumed."""
    assert _blocker_really_blocks(tmp_path)


# ── acceptance 1: the WENO5 refutation, against installed jaxfluids ──────────


@pytest.mark.parametrize("dep", ["jaxfluids", "z3", "cvc5"])
def test_weno5_dependencies_present(dep):
    pytest.importorskip(dep)


def test_weno5_reproducer_runs_outside_the_repo_and_CONFIRMS(tmp_path):
    pytest.importorskip("jaxfluids")
    pytest.importorskip("z3")
    pytest.importorskip("cvc5")
    v, em = _emit(WENO5, tmp_path)
    assert em.runnable, em.unconstructible
    proc, sidecar = _run(em, tmp_path)
    assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
    assert f"== {CONFIRMED}" in proc.stdout, proc.stdout
    assert sidecar["execution"]["result"] == CONFIRMED
    assert sidecar["query"] == v.stamp.query_content_hash
    assert sidecar["fragment"] == "QF_NRA"
    # the published surface, on a real run: exactly the pinned key set
    assert set(sidecar) == set(SIDECAR_KEYS)
    assert set(sidecar["execution"]) == {
        "result", "detail", "reachable", "lhs", "rhs", "modes"
    }
    assert sidecar["schema"] == SCHEMA
    assert set(sidecar["execution"]["modes"]) == set(EXECUTION_MODES)
    # the executed sides: beta_1 strictly above max(beta_0, beta_2)
    (lhs,), (rhs,) = sidecar["execution"]["lhs"], sidecar["execution"]["rhs"]
    assert lhs > rhs, sidecar["execution"]
    # and it is a genuine kink, not a degenerate all-zero stencil
    assert lhs > 0.0 and rhs >= 0.0


def test_the_weno5_reproducer_calls_jaxfluids_and_not_a_copy_of_it(tmp_path):
    """The file must reach the installed library. A reproducer that
    re-implemented the target would confirm its own arithmetic."""
    pytest.importorskip("jaxfluids")
    pytest.importorskip("z3")
    _, em = _emit(WENO5, tmp_path)
    assert "weno5_central_vs_neighbours" in em.source
    assert "jaxfluids" not in em.source  # the import happens in OUR module
    assert "square" not in em.source and "smoothness" not in em.source


# ── acceptance 2: the flagship HeatNode witness ──────────────────────────────


def test_heatnode_reproducer_runs_outside_the_repo_and_CONFIRMS(tmp_path):
    """101.0 against 100.0, executed, from a file, with no stelling."""
    pytest.importorskip("maddening")
    pytest.importorskip("z3")
    pytest.importorskip("cvc5")
    v, em = _emit(HEAT, tmp_path)
    assert em.runnable, em.unconstructible
    proc, sidecar = _run(em, tmp_path)
    assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
    assert f"== {CONFIRMED}" in proc.stdout, proc.stdout
    assert sidecar["execution"]["result"] == CONFIRMED
    # the ledger's own numbers: node output max 101.0, bound 100.0
    assert max(sidecar["execution"]["lhs"]) == 101.0, sidecar["execution"]
    assert set(sidecar["execution"]["rhs"]) == {100.0}
    assert "101.0" in proc.stdout and "100.0" in proc.stdout
    # THE WITNESS IS NOT REPRESENTABLE IN THE DECLARED DTYPE, and the file
    # says so. 847249408/13421773 is an exact rational the solver produced;
    # float32 rounds it to 63.125, and a reader comparing the printed
    # rational to the printed array has to be told which one was executed.
    # The ledger's own record of this witness quotes both.
    assert "847249408/13421773" in proc.stdout
    assert "NOT representable in the" in proc.stdout
    assert "x0[0] = 847249408/13421773" in proc.stdout.split(
        "NOT representable"
    )[1]


def test_the_heatnode_verdict_is_the_ledgers_verdict(tmp_path):
    """The reproducer must not change the verdict — it reproduces the one
    recorded in docs/verdict-ledger.md, elements 1 and 2 violating."""
    pytest.importorskip("maddening")
    pytest.importorskip("z3")
    v, _ = _emit(HEAT, tmp_path)
    (w,) = v.witnesses
    assert w.violating_elements == (1, 2), w.violating_elements
    assert v.status == "REFUTED"


# ── acceptance 3: DIVERGED, which must never render as a failure ─────────────


def test_diverged_is_reported_when_the_dtype_cannot_hold_the_violation(tmp_path):
    pytest.importorskip("z3")
    v, em = _emit(UNDERFLOW, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
    assert sidecar["execution"]["result"] == DIVERGED, proc.stdout
    assert f"== {DIVERGED}" in proc.stdout
    # THE LOAD-BEARING RENDERING PROPERTY: it is not a failure, anywhere.
    assert "NOT A FAILED CHECK" in proc.stdout
    assert CONFIRMED not in proc.stdout
    assert proc.returncode == 0
    # both sides printed, and equal in the program's own dtype
    assert sidecar["execution"]["lhs"] == [0.0]
    assert sidecar["execution"]["rhs"] == [0.0]
    # and DIVERGED requires EVERY mode that ran to hold
    assert sidecar["execution"]["modes"] == {"eager": True, "jit": True}


def test_the_diverged_case_is_genuinely_a_real_float_gap(tmp_path):
    """Established four ways, because a DIVERGED that is really a mistake
    is the worst possible use of this token.

    1. the witness is a STRICTLY POSITIVE rational inside the declared box;
    2. in exact rational arithmetic ``x*x > 0``, so the assertion is
       genuinely false there — computed here with
       :class:`fractions.Fraction`, not by asking stelling again;
    3. in the program's declared dtype, float32, the product underflows to
       exactly ``0.0``, so the assertion holds;
    4. and the gap is the DTYPE's, not the arithmetic's: the identical
       product in float64 is strictly positive, so it is not that the
       computation is wrong anywhere — the declared type cannot hold the
       answer.
    """
    from fractions import Fraction

    import numpy as np

    pytest.importorskip("z3")
    v, _ = _emit(UNDERFLOW, tmp_path)
    (w,) = v.witnesses
    ((name, text),) = w.values
    x = Fraction(text)
    assert name == "x0"
    assert Fraction(0) < x <= Fraction(2) ** -100            # (1)
    assert x * x > 0                                         # (2)
    x32 = np.float32(float(x))
    assert x32 > np.float32(0.0)
    assert float(x32 * x32) == 0.0                           # (3)
    assert float(np.float64(x32) * np.float64(x32)) > 0.0    # (4)


def test_positive_control_the_same_function_CONFIRMS_on_a_wider_box(tmp_path):
    """The mutation that separates 'DIVERGED is a finding' from 'DIVERGED
    is what this emitter always says'. Identical function, identical
    relation, identical dtype; only the envelope moves out of the
    underflow region, and the result flips to CONFIRMED."""
    pytest.importorskip("z3")
    wide = Subject(
        name="underflowing-square-wide",
        fn=underflowing_square,
        relation="<=",
        declarations=(((), "float32", (0.0, 1.0)),),
        no_precondition_reason=NO_CALLER_NARROWING,
    )
    _, em = _emit(wide, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert sidecar["execution"]["result"] == CONFIRMED, proc.stdout
    assert sidecar["execution"]["lhs"][0] > 0.0


def test_the_two_execution_modes_are_both_run_and_disagreement_is_reported(
    tmp_path,
):
    """The finding that made ``modes`` a schema field.

    ``(1 + x) - 1 <= 0`` over a binary64 envelope inside the absorbed
    region holds EAGERLY — the increment is absorbed — and is FALSE under
    ``jax.jit``, because XLA's algebraic simplifier rewrites the
    expression to ``x``. Running one mode and calling it "the program"
    would report the other one's answer as this one's. CONFIRMED wins
    when the modes disagree: the violation IS executable, in the program
    as compiled.
    """
    pytest.importorskip("z3")
    _, em = _emit(ABSORBED, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert sidecar["execution"]["modes"] == {"eager": True, "jit": False}, (
        proc.stdout
    )
    assert sidecar["execution"]["result"] == CONFIRMED
    assert "THE TWO MODES DISAGREE" in proc.stdout
    assert "eager: the assertion HOLDS" in proc.stdout
    assert "jit  : the assertion is FALSE" in proc.stdout
    assert "jit" in sidecar["execution"]["detail"]


# ── acceptance 4: UNREACHABLE, and its argument ──────────────────────────────


def test_unreachable_is_reported_and_carries_the_argument_that_licenses_it(
    tmp_path,
):
    pytest.importorskip("z3")
    _, em = _emit(WEIGHTS, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
    assert sidecar["execution"]["result"] == UNREACHABLE
    assert sidecar["execution"]["reachable"] is False
    # the two episodes' lesson, printed with the result, every time
    assert NORMALIZED in proc.stdout
    assert NORMALIZED in sidecar["execution"]["detail"]
    assert "measured from one trajectory" in proc.stdout
    # UNREACHABLE dominates: the program is NOT executed at that point
    assert CONFIRMED not in proc.stdout
    assert "executing YOUR function" not in proc.stdout
    assert sidecar["execution"]["lhs"] is None


def test_negative_control_the_same_witness_CONFIRMS_without_the_precondition(
    tmp_path,
):
    """Mutation on the one field that decides it. Same function, same
    envelope, same witness — drop the precondition and the identical run
    reports CONFIRMED, so UNREACHABLE is the precondition talking and not
    a property of this subject."""
    pytest.importorskip("z3")
    without = Subject(
        name="normalized-weight-pair-undeclared",
        fn=weight_pair_sum,
        relation="<=",
        declarations=WEIGHTS.declarations,
        no_precondition_reason=(
            "stated deliberately for the control: no caller precondition is "
            "declared on this pair"
        ),
    )
    _, em = _emit(without, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert sidecar["execution"]["result"] == CONFIRMED, proc.stdout
    assert UNREACHABLE in proc.stdout  # named as untested …
    assert "was not tested" in proc.stdout  # … and said to be untested


# ── the uncallable target, stated rather than crashed ────────────────────────


def test_a_bound_method_target_is_named_as_the_fixture_problem(tmp_path):
    """The measured shape: a method bound to an instance built at run
    time. The file cannot rebuild the instance, and says exactly that."""
    pytest.importorskip("z3")
    subject = Subject(
        name="bound-method-target",
        fn=_Holder().sides,
        relation="<=",
        declarations=(
            ((), "float64", (1.0, 3.0)),
            ((), "float64", (1.0, 3.0)),
        ),
        no_precondition_reason="control case; no precondition is declared",
    )
    _, em = _emit(subject, tmp_path)
    assert not em.runnable
    assert "BOUND to a _Holder instance" in em.unconstructible
    proc, sidecar = _run(em, tmp_path)
    assert proc.returncode == NOT_EXECUTED_EXIT, proc.stdout + proc.stderr
    assert "NO EXECUTION RESULT" in proc.stdout
    assert "WHAT COULD NOT BE CONSTRUCTED" in proc.stdout
    assert "_Holder" in proc.stdout
    assert sidecar["execution"]["result"] is None
    # and everything else in the sidecar is still there — the provenance
    # survives even when the execution leg does not
    assert set(sidecar) == set(SIDECAR_KEYS)
    assert sidecar["schema"] == SCHEMA
