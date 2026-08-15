# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The `pow` row's fidelity gauge — measured discriminating power over BOTH
of its emission branches.

`pow` is the one emission row whose output is not a body but ASSERTED LINES.
An integer exponent expands to a product; a non-integer one declares a fresh
`aux` and asserts `aux^q = x^p`, because SMT-LIB2's real theory has no power
operator to name. That second branch states three separable claims — a fresh
constant per element, the even-`q` root guard, and which side of the equation
carries which exponent — and audit 0.2.0 found three soundness defects in the
row (S1, S2, S3) without any of them being expressible as a mutation, because
the row had no named seam at all. This file is the instrument that would have
expressed them.

**IT IS DELIBERATELY BAR-INDEPENDENT.** Every discharge gate reads the
per-obligation OUTCOME off `stelling.solvers.escalate` rather than the
`check()` STATUS, for the reason `tests/test_scatter_gauge_jax.py`'s
`_set_row_records` gives: if `pow` were ever added to
`verdict.VERIFIED_BARRED_PRIMITIVES`, `check` would report UNKNOWN for a
correctly discharged obligation and every such gate would silently start
measuring the bar instead of the row. Measured: with `pow` in that set the
whole battery below reads identically, and nine PRE-EXISTING tests elsewhere
in the suite go red (eleven counting the two detectors in
`tests/test_bar_membership_policy.py`, which owns that measurement). A bar is not a gauge, and this gauge does not depend on one.

SCOPE — what these gates REACH (CONTRIBUTING.md, "an instrument must declare
its SCOPE"):

* **BOTH exponent branches under `real` semantics.** The INTEGER branch
  (product expansion, positive and negative exponents) and the RATIONAL
  branch (`aux^q = x^p`, `q` even and odd), each driven through the slice,
  the fragment stamp, the SMT text, the portfolio dispatch, the
  exact-rational replay, and the end-to-end `preconditions.check` verdict.
* **BOTH faces.** The emission face as above; the transfer face
  (`interval.pow_`'s corner rule) by containment against the values jax
  computes on this target, sampled eagerly AND under `jax.jit` — the oracle
  is the TARGET, never numpy and never stelling.
* **The independent leg**: every witness executed back through the real
  traced jnp program, eagerly and under `jit`.
* **Two admission guards**: the non-dyadic exponent refusal (audit S1 — a
  nearby rational is a DIFFERENT function) and the negative-base refusal
  under a fractional exponent (jax returns NaN there and the Real encoding
  does not model it).
* **An ARRAY-shaped rational `pow`**, one item, for the per-element freshness
  of the auxiliary constant.

SCOPE — what these gates DO NOT reach, and are therefore no evidence about:

* **`integer_pow`'s row.** It is a different primitive with its own
  equation, and this battery's seams are `pow`'s own
  (`smt._pow_integer_body`, `smt._pow_rational_lines`,
  `smt._pow_aux_name`) precisely so that a mutation here does not move it.
  The renderer they share, `smt._repeated_product`, is NOT in this battery
  for the same reason — mutating it moves both rows, so a catch would not
  be attributable to either.
* **`ieee` semantics.** `escalate` refuses every ieee propagation whatever
  it contains, and the ieee `pow` transfer rides a declared libm accuracy
  budget whose gauge is `tests/test_libm_budget.py`. Driven here only as a
  decline probe.
* **The two emission CAPS** (`INTEGER_POW_EXPANSION_CAP`,
  `RATIONAL_POW_DEGREE_CAP`). `tests/test_pow_audit_findings.py` pins both
  boundaries and the 448-pair admitted set; re-measuring them here would be
  a second instrument on one claim, and mutating the degree cap upward
  cannot be done safely — a non-dyadic exponent admitted past it renders a
  product with 3.6e16 factors.
* **`pow`'s INTEGER-DTYPE guard**, and it is not gauged because it is
  UNREACHABLE rather than because nobody wrote a gate. `pow` is in
  `obligation._INT_OVERFLOW_EMITTED`, but no jax program can put an integer
  dtype on it: measured on jax 0.11.0, `lax.pow` raises *"pow does not
  accept dtype int32 at position 0"* and `jnp.power(int32, 2.0)` inserts a
  `convert_element_type` to float64 before the `pow`, while
  `jnp.power(int32, 2)` binds `integer_pow` instead. So the entry is
  defence-in-depth, the mutation that removes it changes nothing any
  jax-traced gate can see, and the guard is recorded UNCOVERED rather than
  gauged (docs/norms.md, "Guard coverage is proven by mutation"). The
  unreachability itself is pinned —
  `test_the_integer_dtype_guard_is_UNREACHABLE_through_jax` — so that a jax
  release which admits it fails loudly instead of leaving this paragraph
  stale.
* **The affine refinement domain**, which does not admit `pow`.
* **Any other primitive's row**, and any array shape past the one item named
  above.

Positive control: a property genuinely FALSE inside the declared box must come
back REFUTED with a witness that replays exactly and that violates the
predicate when executed through jax. Negative control: the same programs with
the bound moved to the exact extremum — genuinely TRUE, and still
interval-undecidable — must come back `discharged` from `escalate` with no
witness. Without both, "everything caught" and "the instrument is blind" print
the same page.
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
from stelling.fidelity import gauge  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import interval_env, propagate  # noqa: E402
from stelling.solvers import SolverConfig, escalate  # noqa: E402

TIMEOUT_MS = 30_000

SCOPE = (
    "BOTH exponent branches of the `pow` row under real semantics — the "
    "integer product expansion (positive and negative exponents) and the "
    "rational aux encoding (q even and odd) — through the slice, the fragment "
    "stamp, the SMT text, the portfolio dispatch, the exact-rational replay "
    "and the end-to-end verdict, eager AND under jit; the interval transfer's "
    "corner rule by containment against jax on this target; the witness "
    "executed back through jax; the non-dyadic-exponent and negative-base "
    "admission guards; and one array-shaped rational pow for "
    "per-element aux freshness. Does NOT drive: `integer_pow`'s row, the "
    "shared `_repeated_product` renderer, ieee semantics past a decline "
    "probe, either emission CAP, the INTEGER-DTYPE guard (unreachable through "
    "jax — see the module docstring), the affine domain, or any other array "
    "shape."
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# --- the controls, as programs ----------------------------------------------
#
# EVERY ONE OF THESE IS INTERVAL-UNDECIDABLE, and that is asserted below
# rather than assumed. A control the cheap layer settles never runs the stage
# under test and would report "did not fire" about an instrument that never
# reached the row.
#
# The integer-branch pair uses a box that STRADDLES ZERO, which is what makes
# it interval-undecidable for free: `interval.pow_` has a sound corner rule
# only for strictly positive bases, so the transfer declines to ⊤ there. The
# rational-branch and negative-exponent fixtures need a strictly positive box
# (a fractional exponent on a negative base is NaN in jax and the emission
# refuses it), so their undecidability comes from a CORRELATION the interval
# domain cannot see instead.

INT_BOX = (-2.0, 3.0)
INT_TRUE_BOUND = 24.0   # x^3 - x attains exactly 24 at x = 3: TRUE inside
INT_FALSE_BOUND = 23.0  # ... so 23 is violated there: FALSE inside


def _int_value(x):
    return jnp.power(x, 3.0) - x


def _int_harness(bound, *, jit=False):
    fn = jax.jit(_int_value) if jit else _int_value

    def h():
        x = any_array((), "float64", INT_BOX)
        return (assert_(fn(x) <= bound),)

    return h


def _neg_exponent_harness(bound):
    """`x**-2 * x * x` is exactly 1 in ℝ, so `<= 1.0` is TRUE and `<= 0.999`
    is FALSE — and the interval domain sees [0.25, 1] x [1, 4] = [0.25, 4],
    which decides neither. The reciprocal is the whole content: drop it and
    the emitted expression is `x^2 * x * x = x^4`, which reaches 16."""

    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return (assert_(jnp.power(x, -2.0) * x * x <= bound),)

    return h


def _rat_upper_harness(bound):
    """`sqrt(x) <= 2` over [1, 4] is TRUE at the bound and FALSE below it.
    Interval-undecidable because `pow_`'s corner rule gives [1, 2] and the
    obligation asks about the endpoint itself... which it would DISCHARGE. So
    the correlation is what makes it escalate: `sqrt(x) * 1 - (x - x)`."""

    def h():
        x = any_array((), "float64", (1.0, 4.0))
        r = jnp.power(x, 0.5)
        return (assert_(r + (x - x) <= bound),)

    return h


def _rat_lower_harness(bound):
    """The LOWER-bound direction, and it is not a duplicate: it is the only
    shape in which dropping the even-`q` guard changes an answer. Without
    `aux >= 0` the negation of `sqrt(x) >= 1` is satisfied by the NEGATIVE
    root, which the guard exists to exclude and which jax never computes."""

    def h():
        x = any_array((), "float64", (1.0, 4.0))
        r = jnp.power(x, 0.5)
        return (assert_(r + (x - x) >= bound),)

    return h


def _rat_false_harness():
    """FALSE, and the shape a WRONG EXPONENT hides in: `sqrt(x) <= 3` over
    [1, 16] is violated at x = 16. An emitted `aux^3 = x` instead of
    `aux^2 = x` makes the largest reachable value 16^(1/3) < 3, so the
    violation disappears — a missed violation, which is the direction with
    nothing downstream to catch it."""

    def h():
        x = any_array((), "float64", (1.0, 16.0))
        return (assert_(jnp.power(x, 0.5) <= 3.0),)

    return h


def _rat_vector_harness():
    """The ARRAY item: two elements of one rational `pow`, and a claim that
    is only false because they are INDEPENDENT. `sqrt(x0) - sqrt(x1) <= 0.9`
    is violated at x = [4, 1] (2 - 1 = 1). One auxiliary constant shared
    between the elements makes the difference identically 0 and the violation
    vanishes."""

    def h():
        x = any_array((2,), "float64", (1.0, 4.0))
        r = jnp.power(x, 0.5)
        return (assert_(r[0] - r[1] <= 0.9),)

    return h


def _non_dyadic_harness():
    """AUDIT S1's headline. `0.1` is not `1/10`: the binary64 literal denotes
    `3602879701896397/36028797018963968` exactly, whose auxiliary encoding is
    far over the degree cap, so this must DECLINE. The predecessor rationalised
    to `1/10` and emitted a script about a different function, which verified
    a claim false at the declared upper bound."""

    def h():
        x = any_array((), "float64", (1.0, 1e300))
        return (assert_(jnp.power(x, 0.1) <= 1e30),)

    return h


def _negative_base_harness():
    """A fractional exponent over a box reaching below zero: jax returns NaN,
    and the Real encoding (which has a root for odd `q` and none for even)
    does not model that. Must decline."""

    def h():
        x = any_array((), "float64", (-1.0, 4.0))
        return (assert_(jnp.power(x, 0.5) <= 2.0),)

    return h


# --- subjects and patching --------------------------------------------------


def _patched(subject):
    """Enter a subject: ``transfers`` entries patch the live transfer registry
    and ``__patches__`` entries (``(module, attr, value)``) patch module
    attributes, so EMISSION- and REPLAY-layer wrongness is expressible and not
    only transfer-layer wrongness."""
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
    """The fragment mutation, as a METHOD patch (``_fragment`` is a
    ``_Slicer`` method, so ``__patches__`` cannot reach it)."""
    if not subject.get("__linear_fragment__"):
        yield
        return
    with mock.patch.object(
        OB._Slicer, "_fragment", lambda self, inputs, eqns: OB.QF_LRA
    ):
        yield


BASELINE: dict = {}


# -- emission mutations, all against `pow`'s OWN seams ------------------------


def _emit_int_off_by_one(term, exp_val):
    return SM._repeated_product(term, abs(exp_val) + 1)


def _emit_int_exponent_ignored(term, exp_val):
    return SM._repeated_product(term, 2)


def _emit_int_no_reciprocal(term, exp_val):
    return SM._repeated_product(term, abs(exp_val))


def _emit_rat_sides_swapped(aux_name, base, p, q):
    lines = [f"(declare-const {aux_name} Real)"]
    if q % 2 == 0:
        lines.append(f"(assert (>= {aux_name} 0.0))")
    lines.append(
        f"(assert (= {SM._repeated_product(aux_name, p)} "
        f"{SM._repeated_product(base, q)}))"
    )
    return lines


def _emit_rat_guard_dropped(aux_name, base, p, q):
    return [
        f"(declare-const {aux_name} Real)",
        f"(assert (= {SM._repeated_product(aux_name, q)} "
        f"{SM._repeated_product(base, p)}))",
    ]


def _emit_rat_never_asserted(aux_name, base, p, q):
    """The constraint EMITTED but never asserted: the line is written as a
    comment, so the script declares `aux` and says nothing about it."""
    lines = [f"(declare-const {aux_name} Real)"]
    if q % 2 == 0:
        lines.append(f"(assert (>= {aux_name} 0.0))")
    lines.append(
        f"; (assert (= {SM._repeated_product(aux_name, q)} "
        f"{SM._repeated_product(base, p)}))"
    )
    return lines


def _emit_rat_denominator_off_by_one(aux_name, base, p, q):
    lines = [f"(declare-const {aux_name} Real)"]
    if q % 2 == 0:
        lines.append(f"(assert (>= {aux_name} 0.0))")
    lines.append(
        f"(assert (= {SM._repeated_product(aux_name, q + 1)} "
        f"{SM._repeated_product(base, p)}))"
    )
    return lines


def _emit_rat_aux_is_the_base(aux_name, base, p, q):
    return [
        f"(declare-const {aux_name} Real)",
        f"(assert (= {aux_name} {base}))",
    ]


def _emit_shared_aux(out_id, element, n_out):
    return f"aux_{out_id}"


def _emit_rat_one_declaration_for_two_elements(aux_name, base, p, q):
    """The WELL-FORMED form of aux-sharing, and the reason it is a separate
    battery item from the name collision above.

    Sharing the NAME alone emits two ``declare-const`` of one symbol, which
    is illegal SMT-LIB2: both backends refuse the script and the obligation
    comes back `unknown`. That is a catch, but it is a catch by
    MALFORMEDNESS, and a reader could take it for evidence that the gauge
    would notice a well-formed sharing. It would not, unless something
    expresses one — so this does.

    Paired with :func:`_emit_shared_aux` it declares the symbol once and
    then constrains it TWICE, which is well-formed and says
    ``x0_0 == x0_1``: the difference of the two roots collapses to 0, and a
    claim that is FALSE at ``[4, 1]`` is silently DISCHARGED. That is the
    missed-violation direction, with nothing downstream to catch it.

    Keyed on the base term's element suffix rather than on state, so the
    scalar fixtures (whose base is ``x0``) are untouched and the catch is
    attributable to the vector gate."""
    lines = []
    if not base.endswith("_1"):
        lines.append(f"(declare-const {aux_name} Real)")
        if q % 2 == 0:
            lines.append(f"(assert (>= {aux_name} 0.0))")
    lines.append(
        f"(assert (= {SM._repeated_product(aux_name, q)} "
        f"{SM._repeated_product(base, p)}))"
    )
    return lines


# -- replay mutations ---------------------------------------------------------


def _replay_exponent_inverted(v, p, q):
    return OB._exact_rational_power(v, q, p)


def _replay_identity(v, p, q):
    return v


# -- transfer mutation --------------------------------------------------------


def _transfer_pow_is_the_base(eqn, params, ins):
    """A `pow` transfer that returns the BASE's box unchanged — right at
    exponent 1 and wrong everywhere else, so it EXCLUDES values jax computes.
    Transfer-face wrongness the emission face does not share, which is the
    mutation that makes a face asymmetry visible."""
    import stelling.interval as iv

    a = ins[0]
    return [iv.IntervalArray(shape=a.shape, los=a.los, his=a.his)]


def _always_emittable(exp_float):
    return None


def _rationalise_the_exponent(exp_float):
    """AUDIT S1's own defect, as a mutation: the exponent replaced by a nearby
    low-denominator rational. `float(Fraction(1, 10))` rounds back to `0.1`, so
    no binary64 distance can see the substitution."""
    return Fraction(exp_float).limit_denominator(128)


def _mutations():
    """The battery, built lazily so the row's seams are looked up against
    whichever tree is loaded."""
    muts = {
        "row-absent-from-the-emission-set": {
            "__patches__": ((OB, "_SUPPORTED", OB._SUPPORTED - {"pow"}),),
        },
        "fragment-claims-linear": {"__linear_fragment__": True},
        "rational-admission-always-yes": {
            "__patches__": ((OB, "rational_pow_problem", _always_emittable),),
        },
        "exponent-rationalised-to-a-nearby-fraction": {
            "__patches__": ((OB, "pow_exponent_rational", _rationalise_the_exponent),),
        },
        "transfer-is-the-base": {"transfers": {"pow": _transfer_pow_is_the_base}},
        "emit-integer-off-by-one": {
            "__patches__": ((SM, "_pow_integer_body", _emit_int_off_by_one),),
        },
        "emit-integer-exponent-ignored": {
            "__patches__": ((SM, "_pow_integer_body", _emit_int_exponent_ignored),),
        },
        "emit-integer-loses-the-reciprocal": {
            "__patches__": ((SM, "_pow_integer_body", _emit_int_no_reciprocal),),
        },
        "emit-rational-sides-swapped": {
            "__patches__": ((SM, "_pow_rational_lines", _emit_rat_sides_swapped),),
        },
        "emit-rational-root-guard-dropped": {
            "__patches__": ((SM, "_pow_rational_lines", _emit_rat_guard_dropped),),
        },
        "emit-rational-constraint-never-asserted": {
            "__patches__": ((SM, "_pow_rational_lines", _emit_rat_never_asserted),),
        },
        "emit-rational-denominator-off-by-one": {
            "__patches__": (
                (SM, "_pow_rational_lines", _emit_rat_denominator_off_by_one),
            ),
        },
        "emit-rational-aux-is-the-base": {
            "__patches__": ((SM, "_pow_rational_lines", _emit_rat_aux_is_the_base),),
        },
        "emit-rational-aux-shared-across-elements": {
            "__patches__": ((SM, "_pow_aux_name", _emit_shared_aux),),
        },
        "emit-rational-one-aux-for-two-elements": {
            "__patches__": (
                (SM, "_pow_aux_name", _emit_shared_aux),
                (SM, "_pow_rational_lines",
                 _emit_rat_one_declaration_for_two_elements),
            ),
        },
        "replay-exponent-inverted": {
            "__patches__": ((OB, "_exact_rational_power", _replay_exponent_inverted),),
        },
        "replay-as-the-identity": {
            "__patches__": ((OB, "_exact_rational_power", _replay_identity),),
        },
    }
    return muts


# --- the gates ---------------------------------------------------------------
#
# Every gate returns True = the subject PASSED it, False = the gate CAUGHT the
# subject. Exceptions are converted to a catch DELIBERATELY and only here:
# `EmissionInfidelityError` and `SolverDisagreement` are the machinery WORKING
# — a mutation that trips one has been caught, not crashed — and
# `fidelity.gauge` would otherwise propagate them as broken gates. That
# conversion is why a gate must never be the only thing asserting a behaviour:
# the row's own tests assert the loud errors loudly.


def _run(harness, subject, *, timeout=TIMEOUT_MS):
    with _patched(subject), _maybe_linear_fragment(subject):
        return check(harness, vacuity_mode="inputs-only",
                     solver_timeout_ms=timeout)


def _outcomes(harness, subject, *, timeout=TIMEOUT_MS):
    """The per-obligation ESCALATION outcomes, not the verdict status. See
    the module docstring: this is what keeps the discharge gates independent
    of `verdict.VERIFIED_BARRED_PRIMITIVES`."""
    with _patched(subject), _maybe_linear_fragment(subject):
        closed = trace(harness)
        p = propagate(closed)
        esc = escalate(closed, p, SolverConfig(timeout_ms=timeout))
        return tuple(r.outcome for r in esc.records)


def _sole_witness(v):
    """The single witness's rational value, or None when there is none."""
    if v.status != "REFUTED" or len(v.witnesses) != 1:
        return None
    (w,) = v.witnesses
    if len(w.values) != 1:
        return None
    ((_, text),) = w.values
    return Fraction(text)


def gate_refutes_the_false_integer_property(subject):
    """POSITIVE CONTROL, integer branch. `x^3 - x <= 23` over [-2, 3] is
    violated at x = 3 (24 > 23) and the interval transfer declines the
    straddling box, so the product expansion is what answers it."""
    try:
        return _sole_witness(_run(_int_harness(INT_FALSE_BOUND), subject)) is not None
    except Exception:  # EmissionInfidelityError included: a CATCH, not a crash
        return False


def gate_refutes_under_jit(subject):
    """The same property with the `pow` fused inside a ``jit`` call, where the
    equation sits one transparent descent below the top level. A row reached
    only by the eager shape passes the gate above and is bypassed here."""
    try:
        return (
            _sole_witness(_run(_int_harness(INT_FALSE_BOUND, jit=True), subject))
            is not None
        )
    except Exception:
        return False


def gate_discharges_the_true_integer_property(subject):
    """NEGATIVE CONTROL, integer branch. Bound moved to the exact maximum, so
    the property HOLDS everywhere in the box and nothing may refute it."""
    try:
        return _outcomes(_int_harness(INT_TRUE_BOUND), subject) == ("discharged",)
    except Exception:
        return False


def gate_discharges_the_negative_exponent_identity(subject):
    """`x**-2 * x * x <= 1.0` is exactly 1 in ℝ. The RECIPROCAL is the whole
    content: without it the emitted value is `x^4`, which reaches 16."""
    try:
        return _outcomes(_neg_exponent_harness(1.0), subject) == ("discharged",)
    except Exception:
        return False


def gate_discharges_the_true_rational_upper_bound(subject):
    """NEGATIVE CONTROL, rational branch, UPPER direction: `sqrt(x) <= 2` over
    [1, 4] holds, at the endpoint."""
    try:
        return _outcomes(_rat_upper_harness(2.0), subject) == ("discharged",)
    except Exception:
        return False


def gate_discharges_the_true_rational_lower_bound(subject):
    """NEGATIVE CONTROL, rational branch, LOWER direction — the only shape in
    which the even-`q` root guard changes an answer."""
    try:
        return _outcomes(_rat_lower_harness(1.0), subject) == ("discharged",)
    except Exception:
        return False


def gate_refutes_the_false_rational_property(subject):
    """POSITIVE CONTROL, rational branch: `sqrt(x) <= 3` over [1, 16] is
    violated above 9, and a wrong DENOMINATOR makes the violation vanish."""
    try:
        return _sole_witness(_run(_rat_false_harness(), subject)) is not None
    except Exception:
        return False


def gate_refutes_the_false_vector_property(subject):
    """The ARRAY item: `sqrt(x0) - sqrt(x1) <= 0.9` is violated at [4, 1], and
    only because the two auxiliaries are INDEPENDENT."""
    try:
        v = _run(_rat_vector_harness(), subject)
        return v.status == "REFUTED" and bool(v.witnesses)
    except Exception:
        return False


def gate_witness_executes_through_jax(subject):
    """The independent leg: the witness's rational value, executed through the
    SAME jnp program EAGERLY and under ``jit``, must actually violate the
    predicate. Shares no code with the emission or the replay."""
    try:
        x = _sole_witness(_run(_int_harness(INT_FALSE_BOUND), subject))
        if x is None or not (INT_BOX[0] <= x <= INT_BOX[1]):
            return False
        xs = jnp.asarray(float(x), jnp.float64)
        eager = float(np.asarray(_int_value(xs)))
        jitted = float(np.asarray(jax.jit(_int_value)(xs)))
        return (eager > INT_FALSE_BOUND and jitted > INT_FALSE_BOUND
                and eager == jitted)
    except Exception:
        return False


def _pow_box(box, exponent):
    """The propagated box of the `pow` EQUATION's output — located by
    primitive name so the gate cannot end up measuring something downstream."""

    def h():
        x = any_array((), "float64", box)
        return (assert_(jnp.power(x, exponent) <= 1e308),)

    closed = trace(h)
    env = interval_env(closed)
    eqns = [e for e in closed.jaxpr.eqns if str(e.primitive) == "pow"]
    if not eqns:
        raise AssertionError(
            "the fixture produced no 'pow' equation — the gauge would be "
            "measuring a different program"
        )
    return env[eqns[0].outvars[0].id]


def gate_interval_containment_eager_and_jit(subject):
    """TRANSFER face. Every value jax computes on this target over the
    declared box lies inside the propagated box — sampled eagerly and under
    ``jit``, because a rule keyed on a name a fused program never binds passes
    eagerly and is bypassed under ``jit``. The oracle is the TARGET, not numpy
    and not stelling.

    STRICTLY POSITIVE BOXES ONLY, because that is the whole domain of
    `interval.pow_`'s corner rule: a base reaching 0 or below has no sound
    rule there and the transfer declines to ⊤, which contains everything and
    would make this gate vacuous."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            for box, exponent in (
                ((1.0, 4.0), 0.5),
                ((1.0, 16.0), 0.25),
                ((0.5, 2.0), 3.0),
                ((1.0, 2.0), -2.0),
                ((2.0, 8.0), 1.5),
            ):
                b = _pow_box(box, exponent)
                lo, hi = box
                pts = [lo, hi, (lo + hi) / 2.0, lo + (hi - lo) / 3.0]
                for pt in pts:
                    xs = jnp.asarray(pt, jnp.float64)
                    fn = (lambda z, e=exponent: jnp.power(z, e))
                    eager = float(np.asarray(fn(xs)))
                    jitted = float(np.asarray(jax.jit(fn)(xs)))
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
    """A `pow` slice is QF_NRA on BOTH branches, and for two different
    reasons: the integer expansion is a product of a declared input with
    itself, and the rational encoding's `aux^q` is a product of a fresh symbol
    with itself WHATEVER the base is (audit 0.2.0 M9). Emitting either under a
    linear logic claims a fragment the problem is not in — and both backends
    then refuse the script."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            for harness in (_int_harness(INT_FALSE_BOUND), _rat_false_harness()):
                sl = _slice_of(harness)
                if sl is None or sl.fragment != OB.QF_NRA:
                    return False
        return True
    except Exception:
        return False


def _declines_emission(harness):
    closed = trace(harness)
    p = propagate(closed)
    if not [o for o in p.obligations if o.status == "unknown"]:
        return None  # the transfer settled it; nothing was ever emitted
    items = OB.slice_unknown_obligations(closed, p, interval_env(closed))
    return all(isinstance(i, OB.DeclinedObligation) for i in items)


def gate_non_dyadic_exponent_declines(subject):
    """AUDIT S1. `0.1` denotes a dyadic rational far over the degree cap, so
    the obligation must decline — never be emitted about a nearby fraction."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            if _declines_emission(_non_dyadic_harness()) is not True:
                return False
        return _run(_non_dyadic_harness(), subject).status != "VERIFIED"
    except Exception:
        return False


def gate_negative_base_declines(subject):
    """A fractional exponent over a box reaching below zero is NaN in jax and
    has no faithful Real encoding, so the slice must decline."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            return _declines_emission(_negative_base_harness()) is not False
    except Exception:
        return False


def gate_replay_agrees_with_jax(subject):
    """EMISSION == REPLAY, measured rather than declared: at a grid of
    rational points inside the declared box, the exact-rational replay's
    verdict on the predicate must equal the one jax's own execution gives.
    Driven on the RATIONAL branch, whose replay is the one audit S3 found
    doing float arithmetic behind an exactness claim."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            harness = _rat_false_harness()
            sl = _slice_of(harness)
            if sl is None:
                return False
            (name,) = [i.name for i in sl.inputs]
            for num in (1, 4, 9, 16):
                q = Fraction(num)
                replayed = OB.evaluate_predicate(sl, {name: q})
                executed = bool(
                    np.asarray(
                        jax.jit(lambda z: jnp.power(z, 0.5) <= 3.0)(
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
    "refutes-the-false-integer-property": gate_refutes_the_false_integer_property,
    "refutes-under-jit": gate_refutes_under_jit,
    "discharges-the-true-integer-property": gate_discharges_the_true_integer_property,
    "discharges-the-negative-exponent-identity":
        gate_discharges_the_negative_exponent_identity,
    "discharges-the-true-rational-upper-bound":
        gate_discharges_the_true_rational_upper_bound,
    "discharges-the-true-rational-lower-bound":
        gate_discharges_the_true_rational_lower_bound,
    "refutes-the-false-rational-property": gate_refutes_the_false_rational_property,
    "refutes-the-false-vector-property": gate_refutes_the_false_vector_property,
    "witness-executes-through-jax": gate_witness_executes_through_jax,
    "interval-containment-eager-and-jit": gate_interval_containment_eager_and_jit,
    "fragment-is-nonlinear": gate_fragment_is_nonlinear,
    "non-dyadic-exponent-declines": gate_non_dyadic_exponent_declines,
    "negative-base-declines": gate_negative_base_declines,
    "replay-agrees-with-jax": gate_replay_agrees_with_jax,
}

RESIDUAL: dict[str, str] = {}


# --- the stage table ---------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """One run of the instrument against whatever tree is loaded."""

    tree: str
    registries: tuple[tuple[str, str], ...]
    stages: tuple[tuple[str, str, str], ...]
    battery: str

    def render(self) -> str:
        lines = [
            "== pow-row gauge — reading",
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


def _stage_emit(harness, label):
    def run():
        sl = _slice_of(harness)
        if sl is None:
            return ("DECLINE", f"{label}: no slice to emit")
        text = SM.emit(sl, "z3", TIMEOUT_MS).text.splitlines()
        body = [ln for ln in text
                if "define-fun" in ln or "declare-const aux" in ln
                or ln.startswith("(assert (=") or ln.startswith("(assert (>=")]
        return ("REACH", f"{label}: {len(body)} row line(s); "
                         f"last: {body[-1] if body else '-'}")

    return run


def _stage_verdict(harness, label):
    def run():
        v = check(harness, vacuity_mode="inputs-only",
                  solver_timeout_ms=TIMEOUT_MS)
        if v.status == "REFUTED":
            (w,) = v.witnesses
            return ("REFUTED", f"{label}: witness {dict(w.values)}")
        return (
            "DECLINE" if v.status == "UNKNOWN" else v.status,
            f"{label}: {'; '.join(v.notes)[:200] or 'no notes'}",
        )

    return run


def _stage_escalation(harness, label):
    def run():
        return ("REACH", f"{label}: outcomes {_outcomes(harness, {})}")

    return run


def _stage_ieee():
    closed = trace(_int_harness(INT_FALSE_BOUND))
    p = propagate(closed, semantics="ieee")
    e = escalate(closed, p, SolverConfig(timeout_ms=TIMEOUT_MS))
    st = p.obligations[0].status
    return (
        "DECLINE" if st == "unknown" else st.upper(),
        f"transfer -> {st}; escalation -> {'; '.join(e.notes)[:120]}",
    )


def measure() -> Reading:
    """Run the instrument and return the reading. Never raises: a stage that
    blows up is recorded as ``ERROR`` with the exception quoted."""
    import stelling

    registries = (
        ("obligation._SUPPORTED has 'pow'", repr("pow" in OB._SUPPORTED)),
        ("obligation._REPLAY_SUPPORTED has 'pow'",
         repr("pow" in OB._REPLAY_SUPPORTED)),
        ("obligation._INT_OVERFLOW_EMITTED has 'pow'",
         repr("pow" in OB._INT_OVERFLOW_EMITTED)),
        ("obligation.INTEGER_POW_EXPANSION_CAP",
         repr(OB.INTEGER_POW_EXPANSION_CAP)),
        ("obligation.RATIONAL_POW_DEGREE_CAP",
         repr(OB.RATIONAL_POW_DEGREE_CAP)),
        ("propagate.TRANSFERS['pow'] tier",
         repr(P.TRANSFERS.get("pow", (None, "-"))[1])),
        ("propagate.IEEE_TRANSFERS['pow'] tier",
         repr(P.IEEE_TRANSFERS.get("pow", (None, "-"))[1])),
        ("smt._FOLDABLE has 'pow'", repr("pow" in SM._FOLDABLE)),
        ("affine.AFFINE_SUPPORTED has 'pow'", repr("pow" in AF.AFFINE_SUPPORTED)),
        ("NEGATIVE registry control — 'exp' in _SUPPORTED",
         repr("exp" in OB._SUPPORTED)),
        ("emission set size", repr(len(OB._SUPPORTED))),
        ("battery size", repr(len(_mutations()))),
    )
    stages = tuple(
        (name,) + _outcome(fn)
        for name, fn in (
            ("bind-eager", lambda: (
                "REACH",
                repr([str(e.primitive)
                      for e in trace(_int_harness(INT_FALSE_BOUND)).jaxpr.eqns]),
            )),
            ("bind-jit", lambda: (
                "REACH",
                repr([str(e.primitive) for e in
                      trace(_int_harness(INT_FALSE_BOUND, jit=True)).jaxpr.eqns]),
            )),
            ("transfer-real-positive-box", lambda: (
                "REACH",
                f"pow([1,4], 0.5) = [{_pow_box((1.0, 4.0), 0.5).los[0]}, "
                f"{_pow_box((1.0, 4.0), 0.5).his[0]}]",
            )),
            ("emit-integer", _stage_emit(_int_harness(INT_FALSE_BOUND), "x^3 - x")),
            ("emit-rational", _stage_emit(_rat_false_harness(), "sqrt(x)")),
            ("emit-rational-vector", _stage_emit(_rat_vector_harness(), "sqrt(x[:])")),
            ("verdict-positive-integer",
             _stage_verdict(_int_harness(INT_FALSE_BOUND), "x^3 - x <= 23 FALSE")),
            ("verdict-positive-rational",
             _stage_verdict(_rat_false_harness(), "sqrt(x) <= 3 over [1,16] FALSE")),
            ("verdict-positive-vector",
             _stage_verdict(_rat_vector_harness(), "sqrt(x0)-sqrt(x1) <= 0.9 FALSE")),
            ("escalation-negative-integer",
             _stage_escalation(_int_harness(INT_TRUE_BOUND), "x^3 - x <= 24 TRUE")),
            ("escalation-negative-rational-upper",
             _stage_escalation(_rat_upper_harness(2.0), "sqrt(x) <= 2 TRUE")),
            ("escalation-negative-rational-lower",
             _stage_escalation(_rat_lower_harness(1.0), "sqrt(x) >= 1 TRUE")),
            ("escalation-negative-exponent",
             _stage_escalation(_neg_exponent_harness(1.0), "x^-2 * x * x <= 1 TRUE")),
            ("admission-non-dyadic", lambda: (
                "DECLINE" if _declines_emission(_non_dyadic_harness()) else "REACH",
                "x ** 0.1 over [1, 1e300]",
            )),
            ("admission-negative-base", lambda: (
                "DECLINE" if _declines_emission(_negative_base_harness()) else "REACH",
                "x ** 0.5 over [-1, 4]",
            )),
            ("ieee-semantics", _stage_ieee),
        )
    )
    try:
        battery = gauge(
            BASELINE, GATES, _mutations(), residual=RESIDUAL, scope=SCOPE
        ).render()
    except Exception as e:  # noqa: BLE001
        battery = f"fidelity.gauge REFUSED: {type(e).__name__}: {e}"
    return Reading(
        tree=stelling.__file__,
        registries=registries,
        stages=stages,
        battery=battery,
    )


# --- the assertions ----------------------------------------------------------


def test_the_battery_is_not_empty_and_the_baseline_reaches_the_solver():
    """Anti-vacuity, first. A battery of zero mutations and a pipeline in
    which every point declines print the same perfect score, so: the battery
    must be non-empty, and the positive control must record an ACTUAL solver
    invocation — not a decline that happens to be green."""
    muts = _mutations()
    assert len(muts) >= 17, sorted(muts)
    v = check(_int_harness(INT_FALSE_BOUND), vacuity_mode="inputs-only",
              solver_timeout_ms=TIMEOUT_MS)
    solver = v.stamp.solver
    assert isinstance(solver, tuple) and solver, (
        f"the positive control recorded no solver invocation ({solver}) — the "
        f"instrument did not reach the stage it claims to gauge"
    )
    assert all(s.invoked for s in solver)


def test_every_control_is_interval_undecidable_so_the_row_is_what_decides():
    """The controls must all survive interval propagation, or the gates
    measure the cheap layer instead of the row."""
    for label, harness in (
        ("int-false", _int_harness(INT_FALSE_BOUND)),
        ("int-true", _int_harness(INT_TRUE_BOUND)),
        ("int-jit", _int_harness(INT_FALSE_BOUND, jit=True)),
        ("neg-exponent", _neg_exponent_harness(1.0)),
        ("rat-upper", _rat_upper_harness(2.0)),
        ("rat-lower", _rat_lower_harness(1.0)),
        ("rat-false", _rat_false_harness()),
        ("rat-vector", _rat_vector_harness()),
    ):
        p = propagate(trace(harness))
        assert all(o.status == "unknown" for o in p.obligations), (
            f"{label}: {[o.status for o in p.obligations]} — the transfer "
            f"settled it, so the gate driving it never reaches the row"
        )


def test_every_control_really_binds_pow_and_not_integer_pow():
    """ANTI-VACUITY. `x ** 2` binds `integer_pow` and `x ** 2.0` binds `pow`;
    a fixture that drifted to the former would gauge a different row while
    looking green."""
    for label, harness, at_top in (
        ("int-false", _int_harness(INT_FALSE_BOUND), True),
        ("int-jit", _int_harness(INT_FALSE_BOUND, jit=True), False),
        ("neg-exponent", _neg_exponent_harness(1.0), True),
        ("rat-false", _rat_false_harness(), True),
        ("rat-vector", _rat_vector_harness(), True),
    ):
        closed = trace(harness)
        prims = {str(e.primitive) for e in closed.jaxpr.eqns}
        assert ("pow" in prims) is at_top, (
            f"{label}: top-level primitives {sorted(prims)} — the jit fixture "
            f"must NOT have `pow` at top level (it is testing the descent) "
            f"and every other one must"
        )
        assert "integer_pow" not in prims, (
            f"{label}: bound `integer_pow`, not `pow` — this gauge would be "
            f"measuring the wrong row"
        )


def test_positive_control_refutes_with_a_witness_that_replays():
    v = check(_int_harness(INT_FALSE_BOUND), vacuity_mode="inputs-only",
              solver_timeout_ms=TIMEOUT_MS)
    assert v.status == "REFUTED", v.render()
    (w,) = v.witnesses
    assert "exact-rational replay" in w.replay
    ((_, text),) = w.values
    x = Fraction(text)
    assert INT_BOX[0] <= x <= INT_BOX[1]
    assert x ** 3 - x > INT_FALSE_BOUND


def test_positive_control_refutes_under_jit_too():
    closed = trace(_int_harness(INT_FALSE_BOUND, jit=True))
    assert "pow" not in {str(e.primitive) for e in closed.jaxpr.eqns}, (
        "the jit fixture put `pow` at top level — it is not testing the "
        "descent it claims to"
    )
    v = check(_int_harness(INT_FALSE_BOUND, jit=True),
              vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT_MS)
    assert v.status == "REFUTED", v.render()
    ((_, text),) = v.witnesses[0].values
    x = Fraction(text)
    assert x ** 3 - x > INT_FALSE_BOUND


def test_negative_controls_discharge_and_produce_no_witness():
    """Read at `escalate`, so this measures the ROW and not the bar."""
    for label, harness in (
        ("int-true", _int_harness(INT_TRUE_BOUND)),
        ("neg-exponent", _neg_exponent_harness(1.0)),
        ("rat-upper", _rat_upper_harness(2.0)),
        ("rat-lower", _rat_lower_harness(1.0)),
    ):
        assert _outcomes(harness, {}) == ("discharged",), (
            f"{label}: {_outcomes(harness, {})}"
        )


def test_the_integer_dtype_guard_is_UNREACHABLE_through_jax():
    """`pow` is in ``obligation._INT_OVERFLOW_EMITTED`` and NOTHING this
    gauge can trace reaches that guard — so it is recorded UNCOVERED, and the
    unreachability is pinned rather than asserted in prose.

    Three measurements on jax 0.11.0, all three of which must hold for the
    disclosure to stay true:

    * ``lax.pow`` REFUSES an integer operand outright, so there is no direct
      route at all;
    * ``jnp.power(int32, 2.0)`` converts to float64 FIRST, so the `pow`
      equation's operand dtype is never the integer one;
    * ``jnp.power(int32, 2)`` binds ``integer_pow``, a different row with its
      own guard.

    If a jax release ever admits an integer `pow`, this test fails and the
    module docstring's UNCOVERED paragraph has to be replaced by a gate — which
    is the point of pinning it. The guard itself is NOT removed on the strength
    of this: an unreachable guard costs nothing and the emission set is not the
    only door into `smt.emit`."""
    import jax.lax as lax

    with pytest.raises(TypeError, match="pow does not accept dtype"):
        jax.make_jaxpr(lambda b: lax.pow(b, 2.0))(jnp.int32(3))

    converted = jax.make_jaxpr(lambda b: jnp.power(b, 2.0))(jnp.int32(3))
    rows = [(str(e.primitive), tuple(str(v.aval.dtype) for v in e.invars))
            for e in converted.eqns]
    assert rows[0][0] == "convert_element_type", rows
    assert [r for r in rows if r[0] == "pow"] == [("pow", ("float64",) * 2)], rows

    integral = jax.make_jaxpr(lambda b: jnp.power(b, 2))(jnp.int32(3))
    assert [str(e.primitive) for e in integral.eqns] == ["integer_pow"]


def test_the_row_has_the_named_seams_a_gauge_needs():
    """The three seams this battery mutates, asserted to exist and to be the
    ones the emission actually calls. A seam nothing routes through is a
    mutation that changes no bytes and a battery that measures nothing."""
    for name in ("_pow_integer_body", "_pow_rational_lines", "_pow_aux_name"):
        assert hasattr(SM, name), name
    import inspect

    src = inspect.getsource(SM.emit)
    for name in ("_pow_integer_body", "_pow_rational_lines", "_pow_aux_name"):
        assert name in src, f"{name} is not called by smt.emit"


def test_the_pow_seams_do_not_move_the_integer_pow_row():
    """The property `_square_body` was extracted to preserve, asserted for
    `pow`: a mutation of this row's seam must leave `integer_pow`'s emitted
    text byte-identical, or a catch here is not attributable to `pow`."""
    def h():
        x = any_array((), "float64", (-2.0, 3.0))
        return (assert_(x ** 3 - x <= INT_FALSE_BOUND),)

    closed = trace(h)
    assert "integer_pow" in {str(e.primitive) for e in closed.jaxpr.eqns}, (
        "this fixture no longer binds `integer_pow`; it cannot show the "
        "independence it claims"
    )
    sl = _slice_of(h)
    assert sl is not None
    before = SM.emit(sl, "z3", TIMEOUT_MS).text
    for mutation in ("emit-integer-off-by-one", "emit-rational-sides-swapped",
                     "emit-rational-aux-shared-across-elements"):
        with _patched(_mutations()[mutation]):
            assert SM.emit(sl, "z3", TIMEOUT_MS).text == before, (
                f"{mutation} moved the `integer_pow` row's emitted text — the "
                f"two rows share a seam and this battery is measuring both"
            )


def test_every_gate_passes_the_baseline_and_catches_its_battery():
    report = gauge(BASELINE, GATES, _mutations(), residual=RESIDUAL, scope=SCOPE)
    print("\n" + report.render())
    caught = dict(report.caught_by)
    for name in _mutations():
        assert caught[name], f"{name} survived every gate — a measured hole"
    # the attributions the battery was built to measure, named rather than
    # left to the aggregate
    assert "refutes-the-false-integer-property" in caught[
        "row-absent-from-the-emission-set"]
    assert "fragment-is-nonlinear" in caught["fragment-claims-linear"]
    assert "non-dyadic-exponent-declines" in caught[
        "exponent-rationalised-to-a-nearby-fraction"]
    assert "interval-containment-eager-and-jit" in caught["transfer-is-the-base"]
    assert "discharges-the-true-rational-lower-bound" in caught[
        "emit-rational-root-guard-dropped"]
    assert "refutes-the-false-rational-property" in caught[
        "emit-rational-denominator-off-by-one"]
    assert "refutes-the-false-vector-property" in caught[
        "emit-rational-aux-shared-across-elements"]
    # the well-formed sharing, whose catch is a MISSED VIOLATION rather than a
    # refused script — asserted to be caught by the vector gate ALONE, because
    # that exclusivity is what says the scalar gates are blind to it
    assert caught["emit-rational-one-aux-for-two-elements"] == (
        "refutes-the-false-vector-property",)
    assert "discharges-the-negative-exponent-identity" in caught[
        "emit-integer-loses-the-reciprocal"]


def test_the_two_faces_are_gauged_and_their_asymmetry_is_visible():
    """A survivor count cannot see a one-face regression ("caught" is
    disjunctive). The transfer-face mutation must be caught by the transfer
    gate and ADMITTED by at least one emission gate — that disagreement is the
    signal, and its absence would mean this gauge has one face."""
    report = gauge(BASELINE, GATES, _mutations(), residual=RESIDUAL, scope=SCOPE)
    asym = {name: (c, a) for name, c, a in report.asymmetries}
    assert "transfer-is-the-base" in asym, report.render()
    caught, admitted = asym["transfer-is-the-base"]
    assert "interval-containment-eager-and-jit" in caught
    assert admitted, (
        "every gate caught the transfer mutation, so the two faces are not "
        "visibly independent here"
    )


def test_ieee_mode_is_exercised_as_a_decline_and_says_so():
    """Both semantics modes are driven. `ieee` has nothing else to drive here:
    `escalate` refuses every ieee propagation whatever it contains, so no ieee
    `pow` EMISSION exists to gauge. The ieee TRANSFER rides a declared libm
    accuracy budget and is `tests/test_libm_budget.py`'s subject, not this
    file's."""
    closed = trace(_int_harness(INT_FALSE_BOUND))
    p = propagate(closed, semantics="ieee")
    assert p.semantics == "ieee"
    e = escalate(closed, p, SolverConfig(timeout_ms=TIMEOUT_MS))
    assert not any(r.invocations for r in e.records)


def test_out_of_scope_affine_is_recorded_as_a_decline_not_gauged():
    """OUT OF SCOPE, stated rather than left implicit: `pow` is not in
    AFFINE_SUPPORTED, so the refinement declines. Sound (it decides nothing)
    and named here so nobody reads this gauge as covering it."""
    assert "pow" not in AF.AFFINE_SUPPORTED


def test_in_process_stubs_are_destroyed_and_the_registries_restored():
    """No stubbed verdict is recorded as a verdict. Every mutation above is a
    context-managed in-process patch; this asserts the live registries survive
    the whole battery unchanged."""

    def snapshot():
        return (
            frozenset(OB._SUPPORTED),
            frozenset(OB._INT_OVERFLOW_EMITTED),
            P.TRANSFERS["pow"],
            SM._pow_integer_body,
            SM._pow_rational_lines,
            SM._pow_aux_name,
            OB._exact_rational_power,
            OB.pow_exponent_rational,
            OB.rational_pow_problem,
            OB._Slicer._fragment,
        )

    before = snapshot()
    gauge(BASELINE, GATES, _mutations(), residual=RESIDUAL, scope=SCOPE)
    assert snapshot() == before


def test_the_reading_renders_and_names_its_scope():
    reading = measure()
    text = reading.render()
    print("\n" + text)
    assert "SCOPE" in text and "Does NOT drive" in text
    assert reading.tree.endswith("stelling/__init__.py")
    outcomes = dict((s, o) for s, o, _ in reading.stages)
    assert outcomes["emit-integer"] == "REACH"
    assert outcomes["emit-rational"] == "REACH"
    assert outcomes["admission-non-dyadic"] == "DECLINE"


if __name__ == "__main__":  # the runnable-reading entry point
    jax.config.update("jax_enable_x64", True)
    print(measure().render())
