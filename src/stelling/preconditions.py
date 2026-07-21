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
    verdict, _ = _pipeline(
        harness, vacuity_mode=vacuity_mode, solver_timeout_ms=solver_timeout_ms
    )
    return verdict


def _pipeline(harness, *, vacuity_mode, solver_timeout_ms):
    """The one pipeline behind :func:`check` — trace, propagate, optional
    solver escalation, stamped verdict assembly, and the VERIFIED widen
    re-check at the same pipeline depth.

    Returns ``(verdict, closed)``: exactly the verdict :func:`check`
    returns (behavior-identical extraction — check() is this helper with
    the traced query dropped), plus the traced :class:`stelling.ir`
    query. The extraction exists for :mod:`stelling.contracts`, whose
    requires face reuses this ONE pipeline rather than growing a second,
    and which quotes interval straddles from the query it gets back.
    Private: the public entry points are :func:`check` and
    :func:`stelling.contracts.check_contract`.
    """
    import dataclasses

    import stelling as _stelling
    from stelling._jax_compat import jax_version, trace, x64_enabled
    from stelling.propagate import propagate
    from stelling.vacuity import _MODES, widen
    from stelling.verdict import make_verdict

    # Eager argument validation, BEFORE tracing (audit F7/F10). The widen
    # used to be the only vacuity_mode validator and it runs only on a
    # VERIFIED — so a typo'd mode could ride green through UNKNOWN paths
    # for a project's whole life and first explode on the day a VERIFIED
    # occurred. Same wording as widen's own refusal; _MODES is the single
    # source of truth. The timeout is exactly int-or-None: no str parsing,
    # no float truncation — a solver budget is a stamped, reproducible
    # quantity, and a silently-coerced one would stamp a number the caller
    # never wrote.
    if vacuity_mode not in _MODES:
        raise ValueError(
            f"widen mode must be one of {_MODES}, got {vacuity_mode!r}"
        )
    if solver_timeout_ms is not None and (
        isinstance(solver_timeout_ms, bool)
        or not isinstance(solver_timeout_ms, int)
    ):
        raise TypeError(
            f"solver_timeout_ms must be an int number of milliseconds or "
            f"None, got {type(solver_timeout_ms).__name__} "
            f"({solver_timeout_ms!r})"
        )

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
        # widening cannot rescue an UNKNOWN/REFUTED; nothing to check
        return v, cj

    wcj = widen(cj, mode=vacuity_mode)
    if wcj == cj or wcj.content_hash() == cj.content_hash():
        # The widened query is IDENTICAL to the original (under
        # "inputs-only", point declarations hold still, so an all-point
        # envelope widens to itself). Re-running it would prove nothing
        # about the envelope — the old path here stamped "discharges with
        # the declared bounds widened", which was measured false on
        # exactly this case (audit F2: mode='all' on the same query says
        # load-bearing). The instrument is inert: no re-run, no
        # load-bearing claim in either direction, and the stamp says so.
        anys = [
            e for e in cj.jaxpr.eqns if e.primitive == "stelling_any"
        ]
        if not anys:
            why = (
                "the query declares no bounded inputs, so widening "
                "changes nothing"
            )
        elif vacuity_mode == "inputs-only" and all(
            e.params_dict()["lo"] == e.params_dict()["hi"] for e in anys
        ):
            why = (
                "every declared input is a point interval, so this mode "
                "widens nothing on this query; mode='all' would also "
                "widen transcribed constants"
            )
        else:
            why = "the widened query is identical to the original"
        vac_line = (
            f"vacuity instrument inert (mode={vacuity_mode}): {why} — "
            f"the widen re-check would re-run the identical query and "
            f"measure nothing, so it did not run"
        )
        stamp = dataclasses.replace(
            v.stamp, assumptions=v.stamp.assumptions + (vac_line,)
        )
        return dataclasses.replace(v, stamp=stamp), cj

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
    # The widen re-check runs at the same pipeline depth, so on the
    # escalated path it makes real solver invocations the vacuity line
    # then relies on — and a relied-on invocation must be stamped (audit
    # F3: 10 measured transport spawns vs 2 stamped). Every re-check
    # invocation is appended to the final stamp's ledger view, tagged so
    # a reader can tell the re-check's asks from the original run's; the
    # interval-only path has an interval-only re-check (nothing to
    # append) and its absence line stays accurate.
    wide_solver = wide.stamp.solver
    recheck_inv = tuple(
        dataclasses.replace(
            s, reason=f"vacuity widen re-check: {s.reason}"
        )
        for s in (wide_solver if isinstance(wide_solver, tuple) else ())
    )
    if recheck_inv:
        orig = v.stamp.solver
        solver_field = (
            orig if isinstance(orig, tuple) else ()
        ) + recheck_inv
    else:
        solver_field = v.stamp.solver
    stamp = dataclasses.replace(
        v.stamp,
        solver=solver_field,
        assumptions=v.stamp.assumptions + (vac_line,),
    )
    return dataclasses.replace(v, stamp=stamp, notes=notes), cj
