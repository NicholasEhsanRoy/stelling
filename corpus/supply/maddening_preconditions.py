# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Two precondition obligations on the MADDENING wavelet elliptic solver.

Target: ``/home/nick/MSF/msf/MADDENING/src/maddening/nodes/adaptive/``
(``wavelet_elliptic.py`` + ``wavelets/matrixfree.py`` — read in place;
this module IMPORTS nothing from maddening, all four harnesses are hand
transcriptions). No commit pin is recorded: the MADDENING repo's git
history is out of scope for this transcription, so the lines below are
quoted from the working tree as read.

THE TRANSCRIBED SOURCE, verbatim
--------------------------------
The operator (``wavelets/matrixfree.py``, ``make_varcoeff_apply``,
lines 112-136 — the ``-∇·(a∇·) + mass·I`` form)::

    112: def make_varcoeff_apply(a_grid: jax.Array, side: int, dim: int, h: float,
    113:                         mass: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    114:     '''Matrix-free ``(-∇·(a∇·) + mass·I)`` matching :func:`operator.physical_varcoeff`.
    ...
    123:     a = jnp.asarray(a_grid).reshape((side,) * dim)
    124:     inv_h2 = 1.0 / h ** 2
    125:
    126:     def apply(u_flat: jax.Array) -> jax.Array:
    127:         u = u_flat.reshape((side,) * dim)
    128:         out = mass * u
    129:         for d in range(dim):
    130:             a_plus = 0.5 * (a + jnp.roll(a, -1, axis=d)) * inv_h2   # face to +1
    131:             a_minus = 0.5 * (a + jnp.roll(a, 1, axis=d)) * inv_h2   # face to -1
    132:             out = out + a_plus * (u - jnp.roll(u, -1, axis=d))
    133:             out = out + a_minus * (u - jnp.roll(u, 1, axis=d))
    134:         return out.reshape(-1)
    135:
    136:     return apply

The θ → a path (``wavelet_elliptic.py``, ``_coeff_field``, lines
231-236)::

    231:     def _coeff_field(self, theta) -> jax.Array:
    232:         if self._coeff_fn is not None:
    233:             return self._coeff_fn(theta)        # θ → a
    234:         if self._a_fixed is not None:
    235:             return self._a_fixed                # fixed a (θ does not drive A)
    236:         return theta                            # a = θ (the coefficient itself)

The default path is ``a = θ`` directly (line 236: ``coeff_fn`` defaults
to ``None`` in the signature, line 121, and ``a`` defaults to ``None``,
line 120), so the declared θ range IS the range of the cell coefficient
field ``a``. The matrix-free apply then face-averages ``a`` (lines
130-131 above) before it multiplies any difference, so the coefficients
the operator actually applies are ``a_plus`` / ``a_minus``.

SIGN/RANGE CONSTRAINT between θ and the operator coefficient: **none**.
``_coeff_field`` (lines 231-236) applies no clamp, softplus, abs, or
projection on the default path — ``return theta`` verbatim — and
``make_varcoeff_apply`` (lines 123-131) applies only reshape, an affine
face average ``0.5 * (a + jnp.roll(a, ±1, axis=d))``, and the positive
constant ``inv_h2``. Nothing constrains the sign or range.

The mass scalar (``wavelet_elliptic.py`` signature, line 125, and its
storage/use)::

    125:         mass: float = 1.0,
    161:         self.mass = float(mass)

and it enters the operator as the ``mass·I`` term at
``matrixfree.py:128``: ``out = mass * u`` (``make_varcoeff_apply``
itself also defaults ``mass: float = 1.0``, line 113; the node passes
``mass=self.mass`` at wavelet_elliptic.py:185 and :251).

WHAT THIS POSES
---------------
Obligation 1 — coefficient positivity (the operator's validity needs
``a > 0`` everywhere). The code's own path from θ to the face
coefficients is transcribed un-simplified: ``a = θ`` (default path,
wavelet_elliptic.py:236), then matrixfree.py:123-124 and 130-131. Two
asserts per posing, one per face array — the code path produces TWO
face-coefficient arrays per axis, which is more than the
``field_positive`` template's single transformed value expresses, so
these two harnesses are written directly (the template's declare →
code-transform → assert pattern, stated by hand).

  Posing A: θ declared over [1e-6, 1e2];  a_plus > 0 and a_minus > 0.
  Posing B: θ declared over [-2.0, 1e2];  the identical construction.

Shape choice, stated: dim = 1, side = 4, h = 1/side = 0.25 (the node
sets ``side = n_coarse * (2 ** n_levels)``, wavelet_elliptic.py:147,
and ``self._h = 1.0 / side``, line 164; side = 4 is the node's own
n_coarse = 2, n_levels = 1 configuration). The face average needs an
array (``jnp.roll`` couples neighbours); side = 4 is small and keeps
the ±1 neighbours distinct. ``jnp.roll`` traces to slice + concatenate
inside a ``_roll_static`` jit sub-jaxpr — whatever the propagation does
with that is the result, reported verbatim.

Obligation 2 — the mass scalar (``mass ≠ 0``; the ``mass·I`` term is
what removes the periodic operator's nullspace). The
``scalar_nonzero`` template fits exactly and is used as-is.

  Posing C: mass declared at the point [1.0, 1.0] (the default — line
            125 quoted above);  mass ≠ 0.
  Posing D: mass declared over [0.0, 1.0];  mass ≠ 0.

Runs: A and C interval-only; B and D interval-only AND escalated
(z3 + cvc5 portfolio, timeout 20 000 ms per invocation).

PRECISION, recorded
-------------------
Everything runs under ``jax_enable_x64=True`` with float64
declarations. The verbatim python-float constants (1e-6, 1e2, -2.0,
0.5, 0.25, 0.0, 1.0) must meet the declared float64 arrays without a
narrowing ``convert_element_type[f64->f32]``. ``inv_h2 = 1.0 / h ** 2``
is python-float arithmetic at trace time in the transcribed code (``h``
is a python float in the node) and is transcribed the same way here.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import stelling  # noqa: E402
from stelling._jax_compat import jax_version, trace  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import scalar_nonzero  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.solvers import (  # noqa: E402
    SolverConfig,
    escalate,
    make_solver_verdict,
)
from stelling.verdict import make_verdict  # noqa: E402

DIM = 1        # smallest dim the face-average path is exercised at
SIDE = 4       # = n_coarse * 2**n_levels with n_coarse=2, n_levels=1
H = 1.0 / SIDE  # wavelet_elliptic.py:164  self._h = 1.0 / side

PRECISION = (
    "jax_enable_x64=True — all declarations are float64; the verbatim "
    "python-float constants (1e-6, 1e2, -2.0, 0.5, 0.25, 0.0, 1.0) must meet "
    "them without a narrowing convert_element_type[f64->f32]. dim=1, side=4, "
    "h=0.25 stated in the module docstring; inv_h2 is trace-time python-float "
    "arithmetic exactly as in the transcribed code"
)

TIMEOUT_MS = 20_000

print(
    f"target: MADDENING wavelet_elliptic (read in place, no commit pin — "
    f"git history out of scope) | jax {jax_version()} | "
    f"stelling {stelling.__version__}"
)
print(f"shape choice: dim={DIM}, side={SIDE}, h={H} (inv_h2={1.0 / H ** 2})")
print()


# --- harnesses ---------------------------------------------------------------


def _coeff_path(theta):
    """θ → face coefficients, the code's own construction, un-simplified.

    a = θ                       wavelet_elliptic.py:236 (default _coeff_field)
    reshape                     matrixfree.py:123
    inv_h2 = 1.0 / h ** 2       matrixfree.py:124
    face averages, axis d = 0   matrixfree.py:130-131 (dim = 1)
    """
    a = theta                                                # wavelet_elliptic.py:236
    a = jnp.asarray(a).reshape((SIDE,) * DIM)                # matrixfree.py:123
    inv_h2 = 1.0 / H ** 2                                    # matrixfree.py:124
    a_plus = 0.5 * (a + jnp.roll(a, -1, axis=0)) * inv_h2    # matrixfree.py:130
    a_minus = 0.5 * (a + jnp.roll(a, 1, axis=0)) * inv_h2    # matrixfree.py:131
    return a_plus, a_minus


def h_a():
    """Posing A — θ over [1e-6, 1e2]; code-path face coefficients > 0."""
    theta = any_array((SIDE,), "float64", (1e-6, 1e2))
    a_plus, a_minus = _coeff_path(theta)
    return assert_(a_plus > 0.0), assert_(a_minus > 0.0)


def h_b():
    """Posing B — identical construction, θ over [-2.0, 1e2]."""
    theta = any_array((SIDE,), "float64", (-2.0, 1e2))
    a_plus, a_minus = _coeff_path(theta)
    return assert_(a_plus > 0.0), assert_(a_minus > 0.0)


def h_c():
    """Posing C — mass at its default point [1.0, 1.0]; mass ≠ 0."""
    _mass, obligation = scalar_nonzero("float64", (1.0, 1.0))
    return obligation


def h_d():
    """Posing D — mass over [0.0, 1.0]; mass ≠ 0."""
    _mass, obligation = scalar_nonzero("float64", (0.0, 1.0))
    return obligation


# --- reporting ---------------------------------------------------------------


def dump_propagation(label, p):
    print(f"-- {label}: interval propagation")
    for o in p.obligations:
        print(f"  assert #{o.index}: {o.status} — {o.detail}")
        for s in o.source_info:
            print(f"    at {s}")
    print(f"  nonvacuity checks: {len(p.nonvacuity_checks)}")
    print(f"  semantics: {p.semantics}")
    print("  propagation notes, verbatim:")
    if not p.notes:
        print("    (none)")
    for n in p.notes:
        print(f"    note: {n}")
    print(f"  coverage line: {p.coverage.summary()}")
    print(
        "  transfers used: "
        + (", ".join(f"{n} [{t}]" for n, t in p.transfers_used) or "(none)")
    )
    print()


def dump_escalation(label, e):
    print(f"-- {label}: escalation")
    print(f"  escalation.semantics: {e.semantics}")
    print(f"  ledger: spawns={e.ledger.spawns} stamps={len(e.ledger.stamps)}")
    print("  escalation-level notes, verbatim:")
    if not e.notes:
        print("    (none)")
    for n in e.notes:
        print(f"    note: {n}")
    if not e.records:
        print("  records: (none — nothing was left unknown to escalate)")
    for r in e.records:
        print(f"  record for assert #{r.index}: outcome={r.outcome}")
        print(f"    detail: {r.detail}")
        if not r.invocations:
            print("    invocations: (none — no solver was invoked)")
        for i, s in enumerate(r.invocations):
            print(
                f"    invocation [{i}]: invoked={s.invoked} name={s.name} "
                f"version={s.version} transport={s.transport}"
            )
            print(f"      options={dict(s.options or ())}")
            print(f"      reason: {s.reason}")
        if r.witness is None:
            print("    witness: (none)")
        else:
            print(f"    witness for assert #{r.witness.obligation_index}:")
            for name, value in r.witness.values:
                print(f"      {name} = {value}")
            print(f"      produced by: {r.witness.produced_by}")
            print(f"      replay: {r.witness.replay}")
        if not r.notes:
            print("    record notes: (none)")
        for n in r.notes:
            print(f"    record note: {n}")
    print()


def run(label, harness, *, escalated):
    print("=" * 78)
    print(f"==== {label}")
    print("=" * 78)
    cj = trace(harness)
    print(f"query content hash: {cj.content_hash()}")
    print()

    p = propagate(cj)
    dump_propagation(label, p)

    print(f"-- {label}: VERDICT (interval-only)")
    v = make_verdict(
        cj,
        p,
        stelling_version=stelling.__version__,
        jax_version=jax_version(),
        precision_config=PRECISION,
    )
    print(v.render())
    print()

    if not escalated:
        return

    e = escalate(cj, p, SolverConfig(timeout_ms=TIMEOUT_MS))
    dump_escalation(label, e)

    print(f"-- {label}: VERDICT (with solver escalation)")
    sv = make_solver_verdict(
        cj,
        p,
        e,
        stelling_version=stelling.__version__,
        jax_version=jax_version(),
        precision_config=PRECISION,
    )
    print(sv.render())
    print()


run(
    "A — coefficient positivity, θ ∈ [1e-6, 1e2], face coeffs > 0 "
    "(interval-only)",
    h_a,
    escalated=False,
)
run(
    "B — coefficient positivity, θ ∈ [-2.0, 1e2], face coeffs > 0 "
    "(interval + escalated)",
    h_b,
    escalated=True,
)
run(
    "C — mass ≠ 0, mass ∈ [1.0, 1.0] (the signature default; interval-only)",
    h_c,
    escalated=False,
)
run(
    "D — mass ≠ 0, mass ∈ [0.0, 1.0] (interval + escalated)",
    h_d,
    escalated=True,
)
