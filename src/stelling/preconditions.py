# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Assumed-precondition obligation templates — input-side, pre-solve.

The class (`design/precondition-class.md`): properties a numerical
method's validity rests on, living in the input/coefficient data before
the expensive computation, that the code assumes — often literally
asserts to a library — and never verifies, and that fail silently (a
wrong answer under a healthy-looking convergence flag). Pointwise or
scalar properties of declared inputs: no solve, no mask, no relational
machinery — the interval core decides these today.

**The class boundary is exactly the solve.** These templates state
preconditions on the *inputs* to a solve (coefficient positivity over an
envelope, a config scalar being nonzero). Properties of the *solve's
behaviour* — conditioning, residual-implies-error — are out of class; an
obligation that needs the solve's behaviour, or that crosses an
active-set mask, has left this module's territory.

Two templates are implemented. Four more class members are named in
`design/precondition-class.md` for the template design and deliberately
not implemented until an instance demands them: operator symmetry,
non-zero/positive diagonal, special-function domain, and (input-side
only) operating-envelope bounds.

Each template declares its inputs via :func:`stelling.harness.any_array`
— so the envelope lands in the traced query and the content hash covers
it — applies the *code's own* transform, and states the obligation. The
transform argument exists so the harness traces the target's real
construction (faithfulness); passing a hand-simplified transform is an
L11 hand step and should be disclosed and priced wherever the verdict
travels.

This module imports jax-free (the harness import happens inside the
templates, at call time): importing it costs nothing in a bare
environment, but *calling* a template requires the ``[jax]`` extra —
tracing happens through the harness primitives.

:func:`check` is the front door: it runs a harness end-to-end — trace,
propagate, assemble the stamped verdict, optionally escalate undecided
obligations to an SMT solver — without the caller touching the
propagation internals. The solver is invoked **only** when a timeout is
passed explicitly (the never-on-defaults discipline, applied to the
convenience layer too) and only if a solver is installed; otherwise the
verdict is interval-only and the stamp records the solver's absence.
"""

from __future__ import annotations

__all__ = ["check", "field_positive", "scalar_nonzero"]


def field_positive(shape, dtype, envelope, transform=None, *, bound=0.0):
    """Pointwise positivity/coercivity of a (transformed) input field.

    Declares ``field`` over ``envelope``, applies ``transform`` — **your
    code's own construction**, not a hand-simplified stand-in — and
    states ``value > bound`` at every point of every value the transform
    produces. The canonical instance is variable-coefficient
    ellipticity: every CG or Cholesky on ``-∇·(a∇·)`` assumes the
    coefficient ``a`` is positive, and the assumption usually lives in
    the inputs unchecked.

    ``transform`` may return a single array **or a tuple/list of
    arrays** — real constructions often produce several quantities that
    must each satisfy the precondition (a conservative discretisation
    yields per-axis *face* coefficient arrays from one cell field; each
    face array must be positive). One obligation is stated per produced
    value, and the return mirrors the transform's shape:

    * single value in → ``(field, obligation)``;
    * tuple/list in → ``(field, tuple_of_obligations)``.

    Return the obligation(s) from your harness so none can be dropped as
    dead code; the field is returned so further obligations can be
    stated over the same declaration.
    """
    from stelling.harness import any_array, assert_

    field = any_array(shape, dtype, envelope)
    value = transform(field) if transform is not None else field
    if isinstance(value, (tuple, list)):
        return field, tuple(assert_(v > bound) for v in value)
    return field, assert_(value > bound)


def scalar_nonzero(dtype, envelope):
    """A config scalar the method's well-posedness needs to be nonzero.

    Declares the scalar over ``envelope`` (its admissible configuration
    range — pose the *range*, not only the default: the question is
    whether the configuration space admits the singular value, not
    whether the default happens to avoid it) and states ``scalar ≠ 0``.
    The canonical instance is a mass/shift term that removes a
    differential operator's nullspace under periodic boundary
    conditions.

    Returns ``(scalar, obligation)``.
    """
    from stelling.harness import any_array, assert_

    scalar = any_array((), dtype, envelope)
    return scalar, assert_(scalar != 0.0)


def check(harness, *, vacuity_mode, solver_timeout_ms=None):
    """Run a precondition harness end-to-end and return the stamped
    :class:`stelling.verdict.Verdict` — with the vacuity check built in:
    **this entry point cannot return an unchecked VERIFIED.**

    ``harness`` is a zero-argument function that declares inputs (via
    the templates above or :func:`stelling.harness.any_array` directly)
    and returns its obligations. The verdict's status is ``VERIFIED``,
    ``REFUTED`` (with a replay-confirmed concrete witness when a solver
    found one), or ``UNKNOWN`` — never guessed; anything the analysis
    could not decide says so, with the reason quoted in the notes.

    ``vacuity_mode`` is **required** (no default — the two registered
    procedures answer different questions, and a silently-picked mode
    would let a caller run the wrong one without saying so; see
    :mod:`stelling.vacuity`). ``"inputs-only"`` is the standard choice
    for precondition checks: non-point declarations widen, transcribed
    constants and thresholds hold still. ``"all"`` widens every
    declaration. On a VERIFIED verdict, the identical query is re-run
    with the declared bounds widened per the mode, at the same pipeline
    depth (escalated iff the original call escalated): an obligation
    that **still discharges with its bounds gone** never depended on the
    declared envelope — a range theorem, or a mis-posed envelope — and
    the verdict says so **in itself** (a note per such obligation and a
    stamped vacuity line), instead of relying on someone re-running the
    control by discipline. The status stays VERIFIED — the claim is
    true — but a CI consumer can and should treat an
    envelope-not-load-bearing VERIFIED as a flag.

    ``solver_timeout_ms``: pass an explicit per-obligation time limit to
    escalate interval-undecided obligations to the SMT portfolio (needs
    the ``[solvers]`` extra, or one of them). **There is no default** —
    a solver run is a stamped, reproducible event and its budget is part
    of what the stamp records. Omit it and the verdict is interval-only.

    Version, precision, and solver stamps are filled from the live
    environment; the precision entry records the *actual*
    ``jax_enable_x64`` state at trace time, not an assumption.
    """
    import dataclasses

    import stelling as _stelling
    from stelling._jax_compat import jax_version, trace, x64_enabled
    from stelling.propagate import propagate
    from stelling.vacuity import widen
    from stelling.verdict import make_verdict

    cj = trace(harness)
    p = propagate(cj)
    versions = dict(
        stelling_version=_stelling.__version__,
        jax_version=jax_version(),
        precision_config=f"jax_enable_x64={x64_enabled()}",
    )

    def _finish(closed, prop):
        if solver_timeout_ms is None:
            return make_verdict(closed, prop, **versions)
        from stelling.solvers import SolverConfig, escalate, make_solver_verdict

        esc = escalate(closed, prop, SolverConfig(timeout_ms=int(solver_timeout_ms)))
        return make_solver_verdict(closed, prop, esc, **versions)

    v = _finish(cj, p)
    if v.status != "VERIFIED":
        return v  # widening cannot rescue an UNKNOWN/REFUTED; nothing to check

    wcj = widen(cj, mode=vacuity_mode)
    wide = _finish(wcj, propagate(wcj))
    still = [
        o.index for o in wide.obligations if o.status == "discharged"
    ]
    if still:
        vac_line = (
            f"vacuity checked (mode={vacuity_mode}): obligation(s) "
            f"{tuple(still)} discharge with the declared bounds widened to "
            f"(-inf, inf) — the verdict does not depend on the declared "
            f"envelope for them (a range theorem, or a mis-posed envelope)"
        )
        notes = v.notes + tuple(
            f"obligation #{i}: discharges with all declared bounds widened "
            f"(vacuity mode={vacuity_mode}) — envelope not load-bearing"
            for i in still
        )
    else:
        vac_line = (
            f"vacuity checked (mode={vacuity_mode}): no obligation "
            f"discharges with the declared bounds widened — the declared "
            f"envelope is load-bearing for this VERIFIED"
        )
        notes = v.notes
    stamp = dataclasses.replace(
        v.stamp, assumptions=v.stamp.assumptions + (vac_line,)
    )
    return dataclasses.replace(v, stamp=stamp, notes=notes)
