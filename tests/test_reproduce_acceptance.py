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
import pathlib
import subprocess
import sys
import textwrap
from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402


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
    """x64 ON for every test EXCEPT the ones that ask otherwise.

    It used to be unconditional, which meant nothing in this file ever ran
    at any other setting — and a mutation making the emitted file ignore
    `SIDECAR["x64"]` and force True survived all 61 tests. x64 is the one
    global the DIVERGED/CONFIRMED distinction lives on.
    """
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.fixture
def x64_off():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
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
    donating_step,
    mixed_dtype_underflow,
    eager_only_absorbs,
    only_under_jit,
    overflowing_bound,
    raises_in_every_mode,
    scatter_write,
    widening_square,
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

def _blocker(package: str) -> str:
    """A sitecustomize that makes ``import <package>`` impossible.

    Parameterised, because two different questions need it: "does the
    emitted file import stelling" (it must not) and "does the target
    really reach the installed library" (it must). The second is the only
    way to measure the second question — asserting on the emitted TEXT
    measures the wrong direction, and a hand-written copy of the target
    passes that.
    """
    return textwrap.dedent(
        f'''
        """Make `import {package}` impossible, so the acceptance measures it."""
        import sys


        class _Blocked:
            def find_spec(self, name, path=None, target=None):
                if name == "{package}" or name.startswith("{package}."):
                    raise ImportError(
                        "{package} is blocked on this path by the acceptance"
                    )
                return None


        sys.meta_path.insert(0, _Blocked())
        '''
    )


def _elsewhere(tmp_path, blocked="stelling"):
    """A working directory that is not the repo, plus a path on which
    ``blocked`` cannot be imported and this test module can."""
    where = tmp_path / "elsewhere"
    where.mkdir(exist_ok=True)
    (where / "sitecustomize.py").write_text(_blocker(blocked))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(where), os.path.dirname(os.path.abspath(__file__))]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONSTARTUP", None)
    return where, env


def _run(emission, tmp_path, blocked="stelling", **env_extra):
    where, env = _elsewhere(tmp_path, blocked)
    env.update(env_extra)
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


def _blocker_really_blocks(tmp_path, package="stelling"):
    where, env = _elsewhere(tmp_path, package)
    probe = subprocess.run(
        [sys.executable, "-c", f"import {package}"],
        cwd=str(where), env=env, capture_output=True, text=True, timeout=120,
    )
    return probe.returncode != 0 and "blocked" in probe.stderr


# ── the control on the control ───────────────────────────────────────────────


def test_the_acceptance_environment_really_cannot_import_stelling(tmp_path):
    """Every acceptance below is worthless if this is not true, and the
    venv has stelling installed, so it must be measured rather than
    assumed."""
    assert _blocker_really_blocks(tmp_path)
    assert _blocker_really_blocks(tmp_path, "jaxfluids")


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
        "result", "detail", "reachable", "lhs", "rhs", "modes", "sides_from"
    }
    assert sidecar["schema"] == SCHEMA
    assert set(sidecar["execution"]["modes"]) == set(EXECUTION_MODES)
    # the executed sides: beta_1 strictly above max(beta_0, beta_2)
    (lhs,), (rhs,) = sidecar["execution"]["lhs"], sidecar["execution"]["rhs"]
    assert lhs > rhs, sidecar["execution"]
    # and it is a genuine kink, not a degenerate all-zero stencil
    assert lhs > 0.0 and rhs >= 0.0


def test_the_weno5_reproducer_calls_jaxfluids_and_not_a_copy_of_it(tmp_path):
    """The file must REACH the installed library, and this measures that by
    taking the library away.

    Its predecessor asserted that the emitted TEXT did not contain
    "jaxfluids"/"square"/"smoothness" — the opposite direction, and
    structurally guaranteed, since no user code is ever pasted into the
    file. A hand-written copy of the Jiang-Shu indicators that never
    imports jaxfluids passed it. So: run the same reproducer a second time
    with jaxfluids blocked on the path. If the target genuinely reaches the
    installed package the run must produce NO EXECUTION RESULT; if it were
    a local reimplementation, blocking jaxfluids would change nothing and
    the CONFIRMED would still appear.
    """
    pytest.importorskip("jaxfluids")
    pytest.importorskip("z3")
    pytest.importorskip("cvc5")
    _, em = _emit(WENO5, tmp_path)

    ok, _ = _run(em, tmp_path)
    assert f"== {CONFIRMED}" in ok.stdout, ok.stdout

    # the executed numbers must be the library's own, at the same point
    from jaxfluids.stencils.reconstruction.shock_capturing.weno5_base import (
        WENO5Base,
    )

    _, ok_side = _run(em, tmp_path)
    point = [
        jnp.asarray(float(Fraction(t)), jnp.float64)
        for t in ok_side["witness"].values()
    ]
    b0, b1, b2 = WENO5Base.smoothness(None, *point)
    assert ok_side["execution"]["lhs"] == [float(b1)], ok_side["execution"]
    assert ok_side["execution"]["rhs"] == [float(max(b0, b2))]

    blocked, side = _run(em, tmp_path, blocked="jaxfluids")
    assert blocked.returncode == NOT_EXECUTED_EXIT, blocked.stdout
    assert "NO EXECUTION RESULT" in blocked.stdout
    assert "jaxfluids is blocked" in blocked.stdout, (
        "with jaxfluids blocked the run must fail FOR THAT REASON; if it "
        "does not, the target is not reaching the installed library"
    )
    assert side["execution"]["result"] is None


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


MIXED = Subject(
    name="mixed-dtype-underflow",
    fn=mixed_dtype_underflow,
    relation="<=",
    declarations=(
        ((), "float32", (0.0, 2.0 ** -100)),
        ((), "float64", (0.0, 1.0)),
    ),
    no_precondition_reason=NO_CALLER_NARROWING,
)


def test_the_diverged_detail_names_every_declared_dtype(tmp_path):
    """It hard-wired ``envelope[0]["dtype"]``, so a contract declaring a
    float64 input first and a float32 one second reported the divergence
    as float64's when it is float32's."""
    pytest.importorskip("z3")
    _, em = _emit(MIXED, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert sidecar["execution"]["result"] == DIVERGED, proc.stdout
    detail = sidecar["execution"]["detail"]
    assert "float32" in detail and "float64" in detail, detail


def test_invented_witness_values_are_marked_as_invented_in_the_sidecar(
    tmp_path,
):
    """``unused64`` is declared and never reached by the obligation, so no
    solver and no replay ever assigned it a value — this file invented it
    from the declared box. It has to be in ``witness`` (it is the point
    that was executed) AND distinguishable from a solved value, which it
    previously was not."""
    pytest.importorskip("z3")
    v, em = _emit(MIXED, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert sidecar["witness_filled"] == ["x1"], sidecar["witness"]
    assert set(sidecar["witness"]) == {"x0", "x1"}
    # the solver really did not name it — the claim the marking rests on
    assert "x1" not in dict(v.witnesses[0].values)
    assert "INVENTED it from the declared box" in proc.stdout


def test_the_file_checks_at_run_time_whether_the_tool_got_loaded(tmp_path):
    """The exact version of the independence check.

    The emitter scans the target's module source for an ``import
    stelling``, which cannot see a helper that imports it two modules
    down. The file itself can: by the time the target is imported, the
    tool is either in ``sys.modules`` or it is not. Run without the
    blocker, against a target whose module DOES reach stelling, and the
    disclosure must appear."""
    pytest.importorskip("z3")
    from test_reproduce import _leaky_target

    subject = Subject(
        name="leaky-target",
        fn=_leaky_target,
        relation="<=",
        declarations=(
            ((), "float64", (1.0, 3.0)),
            ((), "float64", (1.0, 3.0)),
        ),
        no_precondition_reason="control case; no precondition is declared",
    )
    _, em = _emit(subject, tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.dirname(
             os.path.abspath(__file__))), "src")]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, em.path], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
    assert "loaded stelling into this" in proc.stdout, proc.stdout


def test_the_generated_file_rounds_the_witness_correctly_once(tmp_path):
    """``dtype(float(f))`` rounds TWICE for anything narrower than
    binary64, and double rounding lands one ulp off for a rational just
    past a float32 midpoint that binary64 rounds back onto it. One ulp at
    a bound is the whole difference between the assertion holding and not,
    so the file would report the wrong result for the right reason.

    The generated ``_nearest`` is imported from the emitted file itself,
    so this measures the code that actually runs."""
    import importlib.util

    import numpy as np
    from fractions import Fraction

    pytest.importorskip("z3")
    _, em = _emit(UNDERFLOW, tmp_path)
    spec = importlib.util.spec_from_file_location("emitted_repro", em.path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # main() is guarded; nothing executes

    f32 = np.dtype("float32")
    lo = np.float32(1.0)
    hi = np.nextafter(lo, np.float32(2.0))
    mid = (Fraction(float(lo)) + Fraction(float(hi))) / 2
    v = mid + Fraction(1, 2 ** 80)          # strictly above the midpoint
    assert float(v) == float(mid)           # binary64 rounds it back
    assert np.float32(float(v)) == lo       # …and the second rounding is wrong
    assert mod._nearest(np, f32, v) == hi   # the corrected one is not
    # and _BUILD must use it — the array the file actually hands the target
    decl = {"dtype": "float32", "shape": []}
    arr, rounded = mod._build(np, decl, [str(v)])
    assert np.asarray(arr).item() == hi, np.asarray(arr).item()
    assert rounded == [0]                    # and it says the value moved
    # it agrees with the plain conversion wherever that is already right
    for probe in ("1/3", "63.125", "0", "7/8"):
        f = Fraction(probe)
        assert mod._nearest(np, f32, f) == np.float32(float(f)), probe
        built, _ = mod._build(np, decl, [probe])
        assert np.asarray(built).item() == np.float32(float(f)), probe


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


# ── the standard JAX write, which used to yield no result at all ────────────


SCATTER = Subject(
    name="jax-functional-write",
    fn=scatter_write,
    relation="<=",
    declarations=(((4,), "float32", (0.0, 100.0)),),
    no_precondition_reason=(
        "every temperature field in the declared range is one a caller can "
        "present; no caller precondition narrows it"
    ),
)


def test_the_standard_jax_functional_write_produces_a_result(tmp_path):
    """``x.at[i].set(v)`` is the functional-update idiom every JAX program
    uses, and it does not exist on ``numpy.ndarray``. Building numpy inputs
    made eager raise ``AttributeError``, which the file turned into "no
    execution result ... nothing is claimed here in either direction" — for
    a violation it could have confirmed, and which ``jax.jit`` executed
    happily on the same numpy input. A wrongly-silent file is a defect in
    the same family as a wrong one."""
    pytest.importorskip("z3")
    v, em = _emit(SCATTER, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
    assert sidecar["execution"]["result"] == CONFIRMED, proc.stdout
    # and it is the program's real answer: 2*51 = 102 against the bound
    assert max(sidecar["execution"]["lhs"]) > 100.0, sidecar["execution"]
    assert sidecar["execution"]["modes"] == {"eager": False, "jit": False}


def test_the_inputs_the_file_builds_are_jax_arrays(tmp_path):
    """The mechanism behind the test above, pinned separately so a
    refactor that reverts to numpy fails for the reason it is wrong."""
    pytest.importorskip("z3")
    _, em = _emit(SCATTER, tmp_path)
    assert "jnp.asarray(arr)" in em.source
    assert 'importlib.import_module("jax.numpy")' in em.source


def test_a_mode_that_raises_does_not_cancel_the_mode_that_runs(tmp_path):
    """The other half of the same defect, and the half that survived the
    input fix: the file stopped as soon as EAGER raised, so it never
    reached jit. With numpy inputs gone, eager no longer raises on the
    functional-write target — so this target raises eagerly on purpose and
    runs under jit, which is exactly the shape the original failure had."""
    pytest.importorskip("z3")
    subject = Subject(
        name="jit-only-target",
        fn=only_under_jit,
        relation="<=",
        declarations=(
            ((), "float64", (1.0, 3.0)),
            ((), "float64", (1.0, 3.0)),
        ),
        no_precondition_reason="both factors are free inputs of the caller",
    )
    _, em = _emit(subject, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
    assert sidecar["execution"]["result"] == CONFIRMED, proc.stdout
    assert sidecar["execution"]["modes"] == {"eager": None, "jit": False}
    assert sidecar["execution"]["sides_from"] == "jit"
    assert sidecar["execution"]["lhs"][0] > sidecar["execution"]["rhs"][0]


def test_no_execution_result_needs_BOTH_modes_to_have_failed(tmp_path):
    """Silence is still available, and still honest, when neither mode can
    run — and only then.

    Its predecessor used an UNCONSTRUCTIBLE target, so no mode was ever
    attempted and it measured the `_NO_MODES` shape (covered elsewhere)
    rather than the property in its own name. This target is perfectly
    constructible and raises in both modes."""
    pytest.importorskip("z3")
    subject = Subject(
        name="raises-in-every-mode",
        fn=raises_in_every_mode,
        relation="<=",
        declarations=(
            ((), "float64", (1.0, 3.0)),
            ((), "float64", (1.0, 3.0)),
        ),
        no_precondition_reason="control case; no precondition is declared",
    )
    _, em = _emit(subject, tmp_path)
    assert em.runnable, "the target must be CONSTRUCTIBLE for this to measure"
    # the control first: with the switch off the very same file runs
    ok, ok_side = _run(em, tmp_path)
    assert ok_side["execution"]["result"] == CONFIRMED, ok.stdout
    proc, sidecar = _run(em, tmp_path, STELLING_REPRO_RAISE="1")
    assert proc.returncode == NOT_EXECUTED_EXIT, proc.stdout
    assert sidecar["execution"]["result"] is None
    assert sidecar["execution"]["modes"] == {"eager": None, "jit": None}
    assert "in both execution modes" in sidecar["execution"]["detail"]
    assert "ValueError" in sidecar["execution"]["detail"]


# ── the modes must not share buffers, and DIVERGED needs both ───────────────


DONATING = Subject(
    name="donating-step",
    fn=donating_step,
    relation="<=",
    declarations=(((), "float64", (0.0, 2.0 ** -70)),),
    no_precondition_reason=NO_CALLER_NARROWING,
)


def test_a_donated_buffer_does_not_make_the_other_mode_unrunnable(tmp_path):
    """THE FILE DESTROYED ITS OWN INPUT AND THEN PUBLISHED "Nothing here is
    wrong".

    `jax.jit(..., donate_argnums=0)` — the standard step-function idiom —
    deletes its argument. Both modes were handed the SAME buffers, so the
    eager call destroyed the input, the jit call raised "Array has been
    deleted", and the "a mode that raises no longer cancels the mode that
    runs" rule then treated eager as the whole answer: `== DIVERGED …
    Nothing here is wrong, the verdict is unchanged`.

    Ground truth at the same witness with a fresh buffer, measured
    directly in :func:`test_the_donation_ground_truth`: jit gives
    8.47e-22 against 0.0, which is FALSE. The correct result is CONFIRMED.
    """
    pytest.importorskip("z3")
    _, em = _emit(DONATING, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
    assert sidecar["execution"]["result"] == CONFIRMED, proc.stdout
    assert sidecar["execution"]["modes"] == {"eager": True, "jit": False}
    assert sidecar["execution"]["sides_from"] == "jit"
    assert sidecar["execution"]["lhs"][0] > sidecar["execution"]["rhs"][0]
    assert "Array has been deleted" not in proc.stdout


def test_the_donation_ground_truth(tmp_path):
    """The independent half of the test above: what the program does at
    that witness, measured here rather than taken from the file."""
    import numpy as np

    pytest.importorskip("z3")
    v, _ = _emit(DONATING, tmp_path)
    x = float(Fraction(dict(v.witnesses[0].values)["x0"]))
    eager_l, eager_r = donating_step(jnp.asarray(x, jnp.float64))
    assert float(np.asarray(eager_l)) <= eager_r          # absorbed
    jit_l, jit_r = jax.jit(donating_step)(jnp.asarray(x, jnp.float64))
    assert float(np.asarray(jit_l)) > jit_r               # and not, compiled


def test_diverged_requires_every_mode_to_have_run_and_held(tmp_path):
    """DIVERGED says the property holds "in the program's own dtype". A
    mode that raised was not measured, and an unmeasured mode is not an
    agreeing one — so when nothing was false and a mode could not run,
    this file has no answer and says so, rather than claiming the weaker
    fact under the stronger token.

    ``eager_only_absorbs`` holds eagerly (the increment is absorbed) and
    cannot run under jit (an ordinary numpy round-trip raises on a
    tracer). The control is the same file with the switch off, where both
    modes run and DIVERGED is available.
    """
    pytest.importorskip("z3")
    subject = Subject(
        name="eager-only-absorbs",
        fn=eager_only_absorbs,
        relation="<=",
        declarations=(((), "float64", (0.0, 2.0 ** -70)),),
        no_precondition_reason=NO_CALLER_NARROWING,
    )
    _, em = _emit(subject, tmp_path)

    proc, side = _run(em, tmp_path, STELLING_REPRO_NUMPY="1")
    assert proc.returncode == NOT_EXECUTED_EXIT, proc.stdout
    assert side["execution"]["result"] is None
    assert side["execution"]["modes"] == {"eager": True, "jit": None}
    assert "DIVERGED needs every mode" in side["execution"]["detail"]

    # the control: both modes run, and the same holding measurement IS
    # reported — so the refusal above is the missing mode talking
    ctrl, ctrl_side = _run(em, tmp_path)
    assert ctrl_side["execution"]["modes"] == {"eager": True, "jit": False}, (
        ctrl.stdout
    )


# ── a witness the declared dtype cannot hold at all ─────────────────────────


def test_a_witness_above_the_dtype_range_still_produces_a_result(tmp_path):
    """`_nearest` computed `Fraction(float(inf))`, which raises
    OverflowError, so a witness above float32's finite range turned into
    NO EXECUTION RESULT for a violation the program does exhibit —
    overflow to inf IS the correctly rounded conversion."""
    pytest.importorskip("z3")
    # the upper bound is ABOVE float32's finite maximum (3.4e38) on
    # purpose: any witness past it converts to inf, which is what used to
    # raise inside `_nearest`
    subject = Subject(
        name="overflowing-bound",
        fn=overflowing_bound,
        relation="<=",
        declarations=(
            ((), "float32", (0.0, 1e40)),
            ((), "float64", (3.5e38, 3.5e38)),
        ),
        no_precondition_reason=NO_CALLER_NARROWING,
    )
    v, em = _emit(subject, tmp_path)
    assert float(Fraction(dict(v.witnesses[0].values)["x0"])) > 3.4e38
    proc, sidecar = _run(em, tmp_path)
    assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
    assert sidecar["execution"]["result"] == CONFIRMED, proc.stdout
    # the executed left side really is infinite, and the sidecar spells it
    # in a way every JSON parser reads
    assert sidecar["execution"]["lhs"] == ["inf"], sidecar["execution"]

    def strict(c):
        raise AssertionError(f"invalid JSON token {c!r}")

    raw = pathlib.Path(em.sidecar_path).read_text()
    assert "Infinity" not in raw
    json.loads(raw, parse_constant=strict)


# ── the sidecar must be JSON everyone can read ──────────────────────────────


def test_the_sidecar_is_strict_json_even_with_non_finite_numbers(tmp_path):
    """Python emits the bare tokens `Infinity`/`NaN` and reads them back;
    jq, JSON.parse, Go and serde reject them. A published surface only its
    author's language can parse is not published."""
    pytest.importorskip("z3")
    subject = Subject(
        name="half-infinite-envelope",
        fn=overflowing_bound,
        relation="<=",
        declarations=(
            ((), "float32", (0.0, float("inf"))),
            ((), "float64", (3.5e38, 3.5e38)),
        ),
        no_precondition_reason=NO_CALLER_NARROWING,
    )
    _, em = _emit(subject, tmp_path)
    proc, _ = _run(em, tmp_path)
    raw = pathlib.Path(em.sidecar_path).read_text()
    assert "Infinity" not in raw and "NaN" not in raw, raw
    # a STRICT reader: parse_constant fires on exactly the invalid tokens
    def strict(c):
        raise AssertionError(f"invalid JSON token {c!r}")

    doc = json.loads(raw, parse_constant=strict)
    assert doc["envelope"][0]["hi"] == "inf"
    assert doc["execution"]["lhs"] == ["inf"] or isinstance(
        doc["execution"]["lhs"][0], (float, str)
    )


# ── the published sides must be a real counterexample ───────────────────────


def test_the_published_sides_come_from_a_mode_where_the_assertion_is_FALSE(
    tmp_path,
):
    """A counterexample that satisfies its own relation is not one.

    On the mode-disagreement subject the sides were captured from EAGER
    only, so a CONFIRMED published ``lhs=[0.0], rhs=[0.0]`` beside
    ``relation="<="`` — a consumer checking the published numbers against
    the published relation found the counterexample HOLDING, while the
    violating numbers existed only on stdout. The sides now come from a
    mode where the assertion is false whenever one exists, and
    ``sides_from`` says which."""
    pytest.importorskip("z3")
    _, em = _emit(ABSORBED, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    e = sidecar["execution"]
    assert e["result"] == CONFIRMED
    assert e["modes"] == {"eager": True, "jit": False}
    assert e["sides_from"] == "jit"
    # the published pair really violates the published relation
    assert e["lhs"][0] > e["rhs"][0], e
    assert not (e["lhs"][0] <= e["rhs"][0])


def test_every_confirmed_publishes_sides_that_violate_the_relation(tmp_path):
    """The general form, over every CONFIRMED case in this file — the
    property a consumer is entitled to assume and could not."""
    pytest.importorskip("z3")
    pytest.importorskip("jaxfluids")
    import operator

    ops = {"<=": operator.le, ">=": operator.ge, "<": operator.lt,
           ">": operator.gt}
    for subject in (WENO5, SCATTER, ABSORBED):
        _, em = _emit(subject, tmp_path)
        proc, side = _run(em, tmp_path)
        e = side["execution"]
        assert e["result"] == CONFIRMED, (subject.name, proc.stdout)
        assert e["sides_from"] in EXECUTION_MODES
        rel = ops[side["relation"]]
        lhs, rhs = e["lhs"], e["rhs"]
        pairs = zip(lhs, rhs * len(lhs) if len(rhs) == 1 else rhs)
        assert any(not rel(a, b) for a, b in pairs), (subject.name, e)


# ── x64 is the setting the whole distinction lives on ───────────────────────


def test_the_file_restores_the_precision_its_query_was_traced_under(
    tmp_path, x64_off
):
    """The single global the DIVERGED/CONFIRMED distinction depends on, and
    nothing in this file ran at any other setting — so mutating the emitted
    file to ignore `SIDECAR["x64"]` and force True survived every test.

    `widening_square` asks for float64 explicitly. Traced and run under x64
    OFF the request is truncated to float32, the product underflows to
    exactly 0.0 and the assertion HOLDS: DIVERGED. Forced to x64 on, the
    same source computes in float64, stays positive, and the assertion is
    FALSE: CONFIRMED. Two different answers from one file, so the file
    must restore the setting its query was traced under."""
    pytest.importorskip("z3")
    assert jax.config.jax_enable_x64 is False
    subject = Subject(
        name="widening-square",
        fn=widening_square,
        relation="<=",
        declarations=(((), "float32", (0.0, 2.0 ** -100)),),
        no_precondition_reason=NO_CALLER_NARROWING,
    )
    _, em = _emit(subject, tmp_path)
    proc, sidecar = _run(em, tmp_path)
    assert sidecar["x64"] is False
    assert sidecar["execution"]["result"] == DIVERGED, proc.stdout

    # and with the setting forced the other way, the SAME file reports the
    # other result — which is what makes restoring it load-bearing
    forced = pathlib.Path(str(tmp_path / "forced.py"))
    forced.write_text(
        pathlib.Path(em.path).read_text().replace(
            'jax.config.update("jax_enable_x64", SIDECAR["x64"])',
            'jax.config.update("jax_enable_x64", True)',
        )
    )

    class E:
        path = str(forced)
        sidecar_path = str(forced)[:-3] + ".json"

    proc2, side2 = _run(E(), tmp_path)
    assert side2["execution"]["result"] == CONFIRMED, proc2.stdout


# ── the tool-load disclosure, exactly ───────────────────────────────────────


def test_a_lazily_imported_stelling_is_disclosed_too(tmp_path):
    """The check moved AFTER the calls. Asked before them it missed an
    `import stelling` inside the function — the same one an emission-time
    scan of the module's top level also misses. The source scan that used
    to run instead was wrong in the other direction as well: it accused a
    module whose DOCSTRING merely quoted such a line."""
    pytest.importorskip("z3")
    from reproduce_subjects_bound import (
        lazily_reaches_stelling,
        only_mentions_it_in_prose,
    )

    env_base = os.path.dirname(os.path.abspath(__file__))
    for fn, expect in ((lazily_reaches_stelling, True),
                       (only_mentions_it_in_prose, False)):
        subject = Subject(
            name=f"leak-{fn.__name__}",
            fn=fn,
            relation="<=",
            declarations=(
                ((), "float64", (1.0, 3.0)),
                ((), "float64", (1.0, 3.0)),
            ),
            no_precondition_reason="control case; no precondition",
        )
        _, em = _emit(subject, tmp_path)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [env_base, os.path.join(os.path.dirname(env_base), "src")]
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, em.path], cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == RESULT_EXIT, proc.stdout + proc.stderr
        got = "loaded stelling into this" in proc.stdout
        assert got is expect, (fn.__name__, proc.stdout)
        if expect:
            assert "RUNNING the target loaded" in proc.stdout


# ── a precondition that ran and held is not an unknown one ──────────────────


def test_a_measured_reachability_is_not_published_as_unknown(tmp_path):
    """`null` means "no precondition was declared". A run whose
    precondition RAN AND HELD, and whose target then raised, published
    `null` too — a measured value indistinguishable from an unknowable
    one."""
    pytest.importorskip("z3")
    from reproduce_subjects import always_producible

    subject = Subject(
        name="held-then-raised",
        fn=raises_in_every_mode,
        relation="<=",
        declarations=(((), "float64", (1.0, 3.0)), ((), "float64", (1.0, 3.0))),
        precondition=always_producible,
        precondition_reason=(
            "structural: both factors are free inputs of the caller, so "
            "every point of the declared envelope is producible"
        ),
    )
    _, em = _emit(subject, tmp_path)
    proc, sidecar = _run(em, tmp_path, STELLING_REPRO_RAISE="1")
    assert sidecar["execution"]["result"] is None, sidecar["execution"]
    assert sidecar["execution"]["reachable"] is True, sidecar["execution"]
    assert "caller precondition holds at the witness: True" in proc.stdout
    # and the two null-shaped answers stay distinguishable
    _, no_pre = _emit(
        Subject(
            name="held-then-raised-no-precondition",
            fn=raises_in_every_mode,
            relation="<=",
            declarations=subject.declarations,
            no_precondition_reason="none is declared, deliberately",
        ),
        tmp_path,
    )
    _, side2 = _run(no_pre, tmp_path, STELLING_REPRO_RAISE="1")
    assert side2["execution"]["reachable"] is None, side2["execution"]


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
    assert "WHAT IS MISSING" in proc.stdout
    assert "_Holder" in proc.stdout
    assert sidecar["execution"]["result"] is None
    # and everything else in the sidecar is still there — the provenance
    # survives even when the execution leg does not
    assert set(sidecar) == set(SIDECAR_KEYS)
    assert sidecar["schema"] == SCHEMA
