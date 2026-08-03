# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The `square` row's gauge — written and committed BEFORE the row.

`jnp.square` binds jax's own ``square_p`` (a first-class primitive on jax
0.11.0, not sugar for ``integer_pow``), so it has an interval transfer and
no emission: an obligation whose slice traverses one cannot reach the
solver, and a property that is genuinely false over the declared box comes
back as an honest-but-unactionable decline. This file is the instrument
that measures that gap and, once the row exists, measures the row.

It is written first on purpose. A gauge written afterward tests what was
already built; run against the pre-row tree this one must report a DECLINE
on every stage it drives, and :func:`fidelity.gauge` must REFUSE to bless
the battery because the baseline cannot pass its own gates. That refusal
is the baseline guard, and it is reproducible: ``python
tests/test_square_row_gauge_jax.py`` prints the reading on whatever tree
``PYTHONPATH`` selects.

SCOPE — what these gates REACH (CONTRIBUTING.md, "an instrument must
declare its SCOPE"):

* **BOTH faces of the `square` row under `real` semantics.** The transfer
  face: the propagated box, checked for containment of the values jax
  actually computes on this target, sampled EAGERLY and under ``jax.jit``
  (the oracle is the target, never numpy). The emission face:
  ``slice_unknown_obligations`` reaching an ``ObligationSlice`` rather than
  a ``DeclinedObligation``, its fragment classification, the SMT-LIB2 text
  ``stelling.smt.emit`` produces, the portfolio dispatch, the exact-rational
  replay that promotes a model to a ``Witness``, and the end-to-end
  ``preconditions.check`` verdict — the last of these driven BOTH eagerly
  and with the `square` fused inside a ``jit`` call, where the equation is
  one transparent descent below the top level rather than at it.
* **The independent leg**: the witness executed back through the real
  traced jnp program, eagerly and under ``jit`` — the only check that
  shares no code with either the emission or the replay.
* **The integer-dtype emission guard** (SMT Reals are unbounded; jax
  integer arithmetic wraps).
* **Both semantics modes, asymmetrically**: `real` gets the whole pipeline;
  `ieee` is driven as a DECLINE probe only, and that is all there is to
  drive — ``_ieee_square`` declines by design and ``escalate`` refuses
  every ieee propagation whatever primitive it contains. No ieee `square`
  transfer is gauged here because none exists.
* **The registry census, read from the definitions** — the sets are
  imported and printed by :func:`measure`, never recalled.

SCOPE — what these gates DO NOT reach, and are therefore no evidence about:

* **The affine refinement domain.** ``stelling.affine.AFFINE_SUPPORTED``
  admits ``integer_pow`` at y=2 and does not admit `square`; this gauge
  drives ``refine="affine"`` only far enough to record that the refinement
  DECLINES (sound: it decides nothing), and gauges no affine transfer.
* **Complex operands.** ``tests/test_square_complex.py`` owns that, on the
  transfer face; nothing here re-measures it.
* **Array-shaped `square` past one small battery item.** The element-budget
  accounting belongs to ``ELEMENT_BUDGET``'s own instrument.
* **Any other primitive's row.** One negative registry control is asserted
  (``sqrt`` must STILL be outside the emission set) and nothing more.
* **The `ieee` transfer face of `square`**, which does not exist.

Positive control: a property genuinely FALSE inside the declared box must
come back REFUTED with a witness that replays and that violates the
predicate when executed through jax. Negative control: the same program
over the same box with the bound moved one unit — genuinely TRUE, and
still interval-undecidable — must NOT produce a witness; and the emission
set must not have widened past `square`. Without both, "everything caught"
and "the instrument is blind" print the same page (CONTRIBUTING.md, "a
measurement whose result is an ABSENCE needs a positive control").
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from fractions import Fraction
from unittest import mock

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("z3")
pytest.importorskip("cvc5")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import stelling.obligation as OB  # noqa: E402
import stelling.propagate as P  # noqa: E402
import stelling.smt as SM  # noqa: E402
from stelling import affine as AF  # noqa: E402
from stelling import interval as iv  # noqa: E402
from stelling.fidelity import gauge  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import interval_env, propagate  # noqa: E402
from stelling.solvers import SolverConfig, escalate  # noqa: E402

TIMEOUT_MS = 30_000

SCOPE = (
    "BOTH faces of the `square` row under real semantics — the interval "
    "transfer (containment against jax on this target, eager AND jit) and "
    "the emission face (slice, fragment, SMT text, dispatch, exact-rational "
    "replay, end-to-end verdict eager AND under jit) — plus the witness "
    "executed back through jax, the integer-dtype emission guard, and ieee "
    "as a DECLINE probe. Does NOT drive: the affine domain, complex "
    "operands, the element budget, any other primitive's row, or an ieee "
    "square transfer (there is none)."
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# --- the two controls, as programs -------------------------------------------
#
# ONE box, ONE program, two bounds one apart. ``x**2 - x`` attains exactly
# 6 at BOTH ends of ``[-2, 3]``, so ``<= 5`` is genuinely FALSE inside the
# box and ``<= 6`` is genuinely TRUE inside it — and INTERVAL PROPAGATION
# CANNOT DECIDE EITHER: square([-2, 3]) = [0, 9] and x = [-2, 3] give
# [-3, 11], which straddles both bounds. That matters more than the pair
# being tidy. A negative control the cheap layers discharge never runs the
# stage under test, and would report "did not fire" about an instrument
# that never reached the solver at all (CONTRIBUTING.md: "a probe whose
# assertion was trivially true").

BOX = (-2.0, 3.0)
FALSE_BOUND = 5.0  # violated at x = 3 and at x = -2, where x^2 - x = 6
TRUE_BOUND = 6.0   # the exact maximum over the box: holds, everywhere


def _value(x):
    """The program under test, usable under the tracer and on concrete
    arrays alike — one function, so the executed reproduction cannot be of
    a different program from the traced one."""
    return jnp.square(x) - x


def _harness(bound, *, box=BOX, jit=False):
    fn = jax.jit(_value) if jit else _value

    def h():
        x = any_array((), "float64", box)
        return (assert_(fn(x) <= bound),)

    return h


def _int_harness():
    """An integer-dtype square: the emission must decline it (jax integer
    arithmetic wraps; SMT-LIB2 Reals do not)."""

    def h():
        x = any_array((), "int32", (2_000_000_000.0, 2_100_000_000.0))
        return (assert_(jnp.square(x) - x >= 0),)

    return h


# --- subjects and patching ---------------------------------------------------


def _patched(subject):
    """Enter a subject: ``transfers`` entries patch the live transfer
    registry and ``__patches__`` entries (``(module, attr, value)``) patch
    module attributes, so EMISSION- and REPLAY-layer wrongness is
    expressible and not only transfer-layer wrongness."""
    stack = contextlib.ExitStack()
    transfers = subject.get("transfers", {})
    if transfers:
        stack.enter_context(
            mock.patch.dict(
                P.TRANSFERS,
                {p: (fn, P.TRANSFERS[p][1]) for p, fn in transfers.items()},
            )
        )
    for mod, attr, val in subject.get("__patches__", ()):
        stack.enter_context(mock.patch.object(mod, attr, val))
    return stack


@contextlib.contextmanager
def _maybe_linear_fragment(subject):
    """The fragment mutation, expressed as a METHOD patch rather than a
    module attribute (``_fragment`` is a ``_Slicer`` method, so
    ``__patches__`` cannot reach it)."""
    if not subject.get("__linear_fragment__"):
        yield
        return
    with mock.patch.object(
        OB._Slicer, "_fragment", lambda self, inputs, eqns: OB.QF_LRA
    ):
        yield


BASELINE: dict = {}


def _emit_identity(term):
    return term


def _emit_double(term):
    return f"(+ {term} {term})"


def _emit_cube(term):
    return f"(* {term} {term} {term})"


def _replay_identity(v):
    return v


def _replay_negated(v):
    return -(v * v)


def _transfer_no_zero_floor(eqn, params, ins):
    """A `square` transfer that squares the endpoints and forgets that a
    STRADDLING box attains 0 — ``[-2, 3] -> [4, 9]``, which EXCLUDES values
    jax computes. Transfer-face wrongness the emission face does not share,
    so it is the mutation that makes a face asymmetry visible."""
    a = ins[0]
    los, his = [], []
    for lo, hi in zip(a.los, a.his):
        vals = (lo * lo, hi * hi)
        los.append(iv._down(min(vals)))
        his.append(iv._up(max(vals)))
    return [iv.IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))]


def _mutations():
    """The battery, built lazily so the row's mutation seams are looked up
    against whichever tree is loaded. Pre-row the emission/replay seams do
    not exist yet and their mutations are simply absent — the baseline
    fails its own gates there anyway, and the reading is the refusal."""
    muts = {
        "row-absent-from-the-emission-set": {
            "__patches__": ((OB, "_SUPPORTED", OB._SUPPORTED - {"square"}),),
        },
        "fragment-claims-linear": {"__linear_fragment__": True},
        "int-overflow-guard-off": {
            "__patches__": (
                (OB, "_INT_OVERFLOW_EMITTED", OB._INT_OVERFLOW_EMITTED - {"square"}),
            ),
        },
        "transfer-forgets-the-zero-floor": {
            "transfers": {"square": _transfer_no_zero_floor},
        },
    }
    if hasattr(SM, "_square_body"):
        muts["emit-as-the-identity"] = {
            "__patches__": ((SM, "_square_body", _emit_identity),),
        }
        muts["emit-as-a-doubling"] = {
            "__patches__": ((SM, "_square_body", _emit_double),),
        }
        muts["emit-as-a-cube"] = {
            "__patches__": ((SM, "_square_body", _emit_cube),),
        }
    if hasattr(OB, "_square_value"):
        muts["replay-as-the-identity"] = {
            "__patches__": ((OB, "_square_value", _replay_identity),),
        }
        muts["replay-negated"] = {
            "__patches__": ((OB, "_square_value", _replay_negated),),
        }
    return muts


# --- the gates ---------------------------------------------------------------
#
# Every gate returns True = the subject PASSED it, False = the gate CAUGHT
# the subject. Exceptions are converted to a catch DELIBERATELY and only
# here: ``EmissionInfidelityError`` and ``SolverDisagreement`` are the
# machinery WORKING — a mutation that trips one has been caught, not
# crashed — and fidelity.gauge would otherwise propagate them as broken
# gates. That conversion is the reason a gate must never be the only thing
# asserting a behaviour: the row's own tests assert the loud errors loudly.


def _run(harness, subject, *, timeout=TIMEOUT_MS, refine=None):
    with _patched(subject), _maybe_linear_fragment(subject):
        return check(
            harness,
            vacuity_mode="inputs-only",
            solver_timeout_ms=timeout,
            refine=refine,
        )


def _violating_witness(v):
    """The single witness's rational value, or None when there is none."""
    if v.status != "REFUTED" or len(v.witnesses) != 1:
        return None
    (w,) = v.witnesses
    if len(w.values) != 1:
        return None
    ((_, text),) = w.values
    return Fraction(text)


def gate_refutes_the_false_property(subject):
    """POSITIVE CONTROL. ``x**2 - x <= 5`` over ``[-2, 3]`` is violated at
    both ends and interval propagation cannot see it, so the row must
    produce REFUTED with a replay-confirmed witness. Pre-row this is a
    decline, which is the whole finding."""
    try:
        return _violating_witness(_run(_harness(FALSE_BOUND), subject)) is not None
    except Exception:  # EmissionInfidelityError included: a CATCH, not a crash
        return False


def gate_refutes_under_jit(subject):
    """The same property with the `square` fused inside a ``jit`` call.
    Under ``jit`` the equation is not at top level — it sits one transparent
    descent below — so a row reached only by the eager shape would pass the
    gate above and be bypassed here."""
    try:
        return (
            _violating_witness(_run(_harness(FALSE_BOUND, jit=True), subject))
            is not None
        )
    except Exception:
        return False


def gate_does_not_refute_the_true_property(subject):
    """NEGATIVE CONTROL. Same box, same program, bound one higher — and
    ``x**2 - x`` attains exactly 6 at both ends, so ``<= 6`` HOLDS. No
    witness may exist. Interval propagation cannot discharge it either, so
    passing this gate means the solver ran and answered unsat, not that a
    cheap layer settled it before the row was reached."""
    try:
        v = _run(_harness(TRUE_BOUND), subject)
        return v.status != "REFUTED" and not v.witnesses
    except Exception:
        return False


def gate_witness_executes_through_jax(subject):
    """The independent leg: the witness's rational value, executed through
    the SAME jnp program EAGERLY and under ``jit``, must actually violate
    the predicate. Shares no code with the emission or the replay."""
    try:
        x = _violating_witness(_run(_harness(FALSE_BOUND), subject))
        if x is None or not (BOX[0] <= x <= BOX[1]):
            return False
        xs = jnp.asarray(float(x), jnp.float64)
        eager = float(np.asarray(_value(xs)))
        jitted = float(np.asarray(jax.jit(_value)(xs)))
        return eager > FALSE_BOUND and jitted > FALSE_BOUND and eager == jitted
    except Exception:
        return False


def _square_box(box):
    """The propagated box of the `square` EQUATION's output — the transfer
    face's own quantity, located by primitive name so the gate cannot end
    up measuring the subtraction downstream of it."""
    closed = trace(_harness(FALSE_BOUND, box=box))
    env = interval_env(closed)
    sq = [e for e in closed.jaxpr.eqns if str(e.primitive) == "square"]
    if not sq:
        raise AssertionError(
            "the fixture produced no 'square' equation — the gauge would be "
            "measuring a different program"
        )
    return env[sq[0].outvars[0].id]


def gate_interval_containment_eager_and_jit(subject):
    """TRANSFER face. Every value jax computes on this target over the
    declared box lies inside the propagated box — sampled eagerly and under
    ``jit``, because a rule keyed on a name a fused program never binds
    passes eagerly and is bypassed under ``jit``. The oracle is the TARGET,
    not numpy."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            for box in (BOX, (-2.0, 2.0), (0.0, 1.5), (-3.0, -1.0)):
                b = _square_box(box)
                lo, hi = box
                pts = [lo, hi, 0.0 if lo <= 0.0 <= hi else lo,
                       (lo + hi) / 2.0, lo + (hi - lo) / 3.0]
                for p in pts:
                    xs = jnp.asarray(p, jnp.float64)
                    eager = float(np.asarray(jnp.square(xs)))
                    jitted = float(np.asarray(jax.jit(jnp.square)(xs)))
                    if eager != jitted:
                        return False
                    for val in (eager, jitted):
                        if not (b.los[0] <= val <= b.his[0]):
                            return False
        return True
    except Exception:
        return False


def _slice_of(harness):
    closed = trace(harness)
    p = propagate(closed)
    items = OB.slice_unknown_obligations(closed, p, interval_env(closed))
    if len(items) != 1 or isinstance(items[0], OB.DeclinedObligation):
        return None
    return items[0]


def gate_fragment_is_nonlinear(subject):
    """`square` is a PRODUCT of a declared input with itself, so a slice
    that traverses one is QF_NRA. Emitting it under a linear logic claims a
    fragment the problem is not in."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            sl = _slice_of(_harness(FALSE_BOUND))
            return sl is not None and sl.fragment == OB.QF_NRA
    except Exception:
        return False


def gate_integer_dtype_declines(subject):
    """The emission is STRICTER than the transfer on purpose: an integer
    square must decline rather than be relaxed onto unbounded Reals."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            closed = trace(_int_harness())
            p = propagate(closed)
            if not [o for o in p.obligations if o.status == "unknown"]:
                return True  # the transfer settled it; nothing to emit
            items = OB.slice_unknown_obligations(closed, p, interval_env(closed))
            return all(isinstance(i, OB.DeclinedObligation) for i in items)
    except Exception:
        return False


def gate_replay_agrees_with_jax(subject):
    """EMISSION == REPLAY, measured rather than declared: at a grid of
    rational points inside the declared box, the exact-rational replay's
    verdict on the predicate must equal the one jax's own execution gives.
    This is the gate that separates a wrong replay from a wrong solver."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            sl = _slice_of(_harness(FALSE_BOUND))
            if sl is None:
                return False
            (name,) = [i.name for i in sl.inputs]
            for num, den in ((-2, 1), (-3, 2), (0, 1), (1, 1), (5, 2), (3, 1)):
                q = Fraction(num, den)
                replayed = OB.evaluate_predicate(sl, {name: q})
                executed = bool(
                    np.asarray(
                        jax.jit(lambda z: _value(z) <= FALSE_BOUND)(
                            jnp.asarray(float(q), jnp.float64)
                        )
                    )
                )
                if replayed != executed:
                    return False
        return True
    except Exception:
        return False


GATES = {
    "refutes-the-false-property": gate_refutes_the_false_property,
    "refutes-under-jit": gate_refutes_under_jit,
    "does-not-refute-the-true-property": gate_does_not_refute_the_true_property,
    "witness-executes-through-jax": gate_witness_executes_through_jax,
    "interval-containment-eager-and-jit": gate_interval_containment_eager_and_jit,
    "fragment-is-nonlinear": gate_fragment_is_nonlinear,
    "integer-dtype-declines": gate_integer_dtype_declines,
    "replay-agrees-with-jax": gate_replay_agrees_with_jax,
}


# --- the stage table ---------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """One run of the instrument against whatever tree is loaded."""

    tree: str
    registries: tuple[tuple[str, str], ...]
    stages: tuple[tuple[str, str, str], ...]  # (stage, outcome, detail)
    battery: str  # the fidelity report, or the refusal that replaced it

    def render(self) -> str:
        lines = [
            "== square-row gauge — reading",
            f"tree: {self.tree}",
            f"SCOPE — what these gates reach: {SCOPE}",
            "-- registries, printed from the definitions --",
        ]
        for name, val in self.registries:
            lines.append(f"  {name}: {val}")
        lines.append("-- stages in scope --")
        width = max((len(s) for s, _, _ in self.stages), default=0)
        for stage, outcome, detail in self.stages:
            lines.append(f"  {stage:<{width}s}  {outcome:<8s}  {detail}")
        lines.append("-- battery --")
        lines.append(self.battery)
        lines.append(
            f"== every line above is scoped to: {SCOPE} — a surface these "
            f"gates do not drive is NOT gauged by this reading =="
        )
        return "\n".join(lines)


def _outcome(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — a stage table records, never raises
        return ("ERROR", f"{type(e).__name__}: {e}")


def _stage_slice(jit=False):
    def run():
        closed = trace(_harness(FALSE_BOUND, jit=jit))
        p = propagate(closed)
        item = OB.slice_unknown_obligations(closed, p, interval_env(closed))[0]
        if isinstance(item, OB.DeclinedObligation):
            return ("DECLINE", item.reason)
        return ("REACH", f"{type(item).__name__} fragment={item.fragment}")

    return run


def _stage_emit():
    sl = _slice_of(_harness(FALSE_BOUND))
    if sl is None:
        return ("DECLINE", "no slice to emit")
    body = [
        ln for ln in SM.emit(sl, "z3", TIMEOUT_MS).text.splitlines()
        if "define-fun" in ln
    ]
    return ("REACH", f"{len(body)} define-fun line(s); last: {body[-1] if body else '-'}")


def _stage_verdict(bound, label, jit=False):
    def run():
        v = check(
            _harness(bound, jit=jit),
            vacuity_mode="inputs-only",
            solver_timeout_ms=TIMEOUT_MS,
        )
        if v.status == "REFUTED":
            (w,) = v.witnesses
            return ("REFUTED", f"{label}: witness {dict(w.values)}")
        return (
            "DECLINE" if v.status == "UNKNOWN" else v.status,
            f"{label}: {'; '.join(v.notes) or 'no notes'}",
        )

    return run


def _stage_ieee():
    closed = trace(_harness(FALSE_BOUND))
    p = propagate(closed, semantics="ieee")
    e = escalate(closed, p, SolverConfig(timeout_ms=TIMEOUT_MS))
    st = p.obligations[0].status
    return (
        "DECLINE" if st == "unknown" else st.upper(),
        f"transfer -> {st}; escalation -> {'; '.join(e.notes)[:120]}",
    )


def _stage_affine():
    v = check(
        _harness(FALSE_BOUND),
        vacuity_mode="inputs-only",
        solver_timeout_ms=None,
        refine="affine",
    )
    return (v.status, f"refine='affine' (OUT OF SCOPE, recorded): {v.notes}")


def _stage_int():
    closed = trace(_int_harness())
    p = propagate(closed)
    if not [o for o in p.obligations if o.status == "unknown"]:
        return ("SETTLED", f"transfer decided it: {p.obligations[0].status}")
    item = OB.slice_unknown_obligations(closed, p, interval_env(closed))[0]
    if isinstance(item, OB.DeclinedObligation):
        return ("DECLINE", item.reason)
    return ("REACH", f"emitted as {item.fragment} — the guard did not fire")


def measure() -> Reading:
    """Run the instrument and return the reading. Never raises: a stage
    that blows up is recorded as ``ERROR`` with the exception quoted, so a
    pre-row tree produces a readable page instead of a traceback."""
    import stelling

    registries = (
        ("obligation._SUPPORTED has 'square'", repr("square" in OB._SUPPORTED)),
        (
            "obligation._REPLAY_SUPPORTED has 'square'",
            repr("square" in OB._REPLAY_SUPPORTED),
        ),
        (
            "obligation._INT_OVERFLOW_EMITTED has 'square'",
            repr("square" in OB._INT_OVERFLOW_EMITTED),
        ),
        (
            "obligation._INT_SAFE_EMITTED has 'square'",
            repr("square" in OB._INT_SAFE_EMITTED),
        ),
        ("propagate.TRANSFERS['square'] tier", repr(P.TRANSFERS.get("square", (None, "-"))[1])),
        (
            "propagate.IEEE_TRANSFERS['square'] tier",
            repr(P.IEEE_TRANSFERS.get("square", (None, "-"))[1]),
        ),
        ("propagate._INT_COMPUTING has 'square'", repr("square" in P._INT_COMPUTING)),
        ("affine.AFFINE_SUPPORTED has 'square'", repr("square" in AF.AFFINE_SUPPORTED)),
        (
            "NEGATIVE registry control — 'sqrt' in _SUPPORTED",
            repr("sqrt" in OB._SUPPORTED),
        ),
        ("emission set size", repr(len(OB._SUPPORTED))),
        ("battery size", repr(len(_mutations()))),
    )
    stages = tuple(
        (name,) + _outcome(fn)
        for name, fn in (
            ("bind-eager", lambda: (
                "REACH",
                repr([str(e.primitive)
                      for e in trace(_harness(FALSE_BOUND)).jaxpr.eqns]),
            )),
            ("bind-jit", lambda: (
                "REACH",
                repr([str(e.primitive)
                      for e in trace(_harness(FALSE_BOUND, jit=True)).jaxpr.eqns]),
            )),
            ("transfer-real", lambda: (
                "REACH",
                f"square box over {BOX} = "
                f"[{_square_box(BOX).los[0]}, {_square_box(BOX).his[0]}]",
            )),
            ("slice-eager", _stage_slice()),
            ("slice-under-jit", _stage_slice(jit=True)),
            ("emit", _stage_emit),
            (
                "verdict-positive-control",
                _stage_verdict(FALSE_BOUND, "x^2 - x <= 5 over [-2,3]: FALSE inside"),
            ),
            (
                "verdict-positive-control-jit",
                _stage_verdict(FALSE_BOUND, "same, square fused in jit", jit=True),
            ),
            (
                "verdict-negative-control",
                _stage_verdict(TRUE_BOUND, "x^2 - x <= 6 over [-2,3]: TRUE inside"),
            ),
            ("integer-dtype", _stage_int),
            ("ieee-semantics", _stage_ieee),
            ("affine-refinement", _stage_affine),
        )
    )
    try:
        battery = gauge(
            BASELINE, GATES, _mutations(), residual={}, scope=SCOPE
        ).render()
    except Exception as e:  # noqa: BLE001 — the refusal IS the pre-row reading
        battery = f"fidelity.gauge REFUSED: {type(e).__name__}: {e}"
    return Reading(
        tree=stelling.__file__,
        registries=registries,
        stages=stages,
        battery=battery,
    )


# --- the assertions ----------------------------------------------------------
#
# These fail on the pre-row tree, deliberately: that failure is the
# baseline guard. :func:`measure` above is the part that must stay runnable
# on either tree.


def test_the_battery_is_not_empty_and_the_baseline_reaches_the_solver():
    """Anti-vacuity, first. A battery of zero mutations and a pipeline in
    which every point declines print the same perfect score, so: the
    battery must be non-empty, and the positive control must record an
    ACTUAL solver invocation — not a decline that happens to be green."""
    muts = _mutations()
    assert len(muts) >= 9, sorted(muts)
    v = check(
        _harness(FALSE_BOUND), vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT_MS
    )
    solver = v.stamp.solver
    assert isinstance(solver, tuple) and solver, (
        f"the positive control recorded no solver invocation ({solver}) — the "
        f"instrument did not reach the stage it claims to gauge"
    )
    assert all(s.invoked for s in solver)


def test_both_controls_are_interval_undecidable_so_the_row_is_what_decides():
    """The controls must both survive interval propagation, or the gates
    above measure the cheap layer instead of the row."""
    for bound in (FALSE_BOUND, TRUE_BOUND):
        p = propagate(trace(_harness(bound)))
        assert p.obligations[0].status == "unknown", (bound, p.obligations)


def test_positive_control_refutes_with_a_witness_that_replays():
    v = check(
        _harness(FALSE_BOUND), vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT_MS
    )
    assert v.status == "REFUTED", v.render()
    (w,) = v.witnesses
    assert "exact-rational replay" in w.replay
    ((_, text),) = w.values
    x = Fraction(text)
    assert BOX[0] <= x <= BOX[1]
    assert x * x - x > FALSE_BOUND


def test_positive_control_refutes_under_jit_too():
    """Containment under `jit` as well as eagerly: under `jit` the `square`
    is one transparent descent below the top level, and a row reached only
    by the eager shape is bypassed there."""
    closed = trace(_harness(FALSE_BOUND, jit=True))
    assert "square" not in {str(e.primitive) for e in closed.jaxpr.eqns}, (
        "the jit fixture put `square` at top level — it is not testing the "
        "descent it claims to"
    )
    v = check(
        _harness(FALSE_BOUND, jit=True),
        vacuity_mode="inputs-only",
        solver_timeout_ms=TIMEOUT_MS,
    )
    assert v.status == "REFUTED", v.render()
    ((_, text),) = v.witnesses[0].values
    x = Fraction(text)
    assert x * x - x > FALSE_BOUND


def test_negative_control_does_not_fire():
    """Same box, same program, bound one higher: no witness, ever."""
    v = check(
        _harness(TRUE_BOUND), vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT_MS
    )
    assert v.status == "VERIFIED", v.render()
    assert v.witnesses == ()


def test_negative_registry_control_the_emission_set_did_not_widen():
    """A row that closed `square` by widening the whitelist would be a
    different and much worse change. `sqrt` is the sentinel: nonlinear,
    adjacent, and deliberately outside."""
    assert "sqrt" not in OB._SUPPORTED
    assert "exp" not in OB._SUPPORTED
    assert "square" in OB._SUPPORTED


def test_every_gate_passes_the_baseline_and_catches_its_battery():
    report = gauge(BASELINE, GATES, _mutations(), residual={}, scope=SCOPE)
    print("\n" + report.render())
    assert report.residual == ()
    caught = dict(report.caught_by)
    for name in _mutations():
        assert caught[name], f"{name} survived every gate — a measured hole"
    assert "refutes-the-false-property" in caught["row-absent-from-the-emission-set"]
    assert "refutes-under-jit" in caught["row-absent-from-the-emission-set"]
    assert "fragment-is-nonlinear" in caught["fragment-claims-linear"]
    assert "integer-dtype-declines" in caught["int-overflow-guard-off"]
    assert (
        "interval-containment-eager-and-jit"
        in caught["transfer-forgets-the-zero-floor"]
    )
    assert caught["emit-as-the-identity"] and caught["emit-as-a-doubling"]
    assert caught["replay-as-the-identity"] and caught["replay-negated"]


def test_the_two_faces_are_gauged_and_their_asymmetry_is_visible():
    """A survivor count cannot see a one-face regression ("caught" is
    disjunctive). The transfer-face mutation must be caught by the transfer
    gate and ADMITTED by the emission gates — that disagreement is the
    signal, and its absence would mean this gauge has one face."""
    report = gauge(BASELINE, GATES, _mutations(), residual={}, scope=SCOPE)
    asym = {name: (c, a) for name, c, a in report.asymmetries}
    assert "transfer-forgets-the-zero-floor" in asym, report.render()
    caught, admitted = asym["transfer-forgets-the-zero-floor"]
    assert "interval-containment-eager-and-jit" in caught
    assert "refutes-the-false-property" in admitted


def test_ieee_mode_is_exercised_as_a_decline_and_says_so():
    """Both semantics modes are driven. `ieee` has nothing else to drive:
    `_ieee_square` declines by design (the schedule is fixed but the rounded
    self-product needs a transfer nobody has written), and `escalate`
    refuses every ieee propagation whatever it contains."""
    closed = trace(_harness(FALSE_BOUND))
    p = propagate(closed, semantics="ieee")
    assert p.semantics == "ieee"
    assert p.obligations[0].status == "unknown"
    # the measured decline text, quoted rather than paraphrased
    assert any("'square' has no sound rule" in n for n in p.notes), p.notes
    assert p.coverage.unknown == 1
    e = escalate(closed, p, SolverConfig(timeout_ms=TIMEOUT_MS))
    assert e.records and all(r.outcome == "unknown" for r in e.records)
    assert not any(r.invocations for r in e.records)


def test_out_of_scope_affine_is_recorded_as_a_decline_not_gauged():
    """OUT OF SCOPE, stated rather than left implicit: `square` is not in
    AFFINE_SUPPORTED, so the refinement declines. Sound (it decides
    nothing) and named here so nobody reads this gauge as covering it."""
    assert "square" not in AF.AFFINE_SUPPORTED
    v = check(
        _harness(FALSE_BOUND),
        vacuity_mode="inputs-only",
        solver_timeout_ms=None,
        refine="affine",
    )
    assert v.status == "UNKNOWN"
    assert any("square" in n for n in v.notes), v.notes


def test_in_process_stubs_are_destroyed_and_the_registries_restored():
    """No stubbed verdict is recorded as a verdict. Every mutation above is
    a context-managed in-process patch; this asserts the live registries
    survive the whole battery unchanged."""

    def snapshot():
        return (
            frozenset(OB._SUPPORTED),
            frozenset(OB._REPLAY_SUPPORTED),
            frozenset(OB._INT_OVERFLOW_EMITTED),
            P.TRANSFERS["square"],
            getattr(SM, "_square_body", None),
            getattr(OB, "_square_value", None),
            OB._Slicer._fragment,
        )

    before = snapshot()
    gauge(BASELINE, GATES, _mutations(), residual={}, scope=SCOPE)
    assert snapshot() == before


def test_the_reading_renders_and_names_its_scope():
    reading = measure()
    text = reading.render()
    print("\n" + text)
    assert "SCOPE" in text and "Does NOT drive" in text
    assert reading.tree.endswith("stelling/__init__.py")
    assert dict((s, o) for s, o, _ in reading.stages)["slice-eager"] == "REACH"
    assert dict((s, o) for s, o, _ in reading.stages)["slice-under-jit"] == "REACH"


if __name__ == "__main__":  # the baseline-guard entry point
    jax.config.update("jax_enable_x64", True)
    print(measure().render())
