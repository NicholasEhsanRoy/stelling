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


def check(harness, *, vacuity_mode, semantics="real", solver_timeout_ms=None,
          refine=None, solver=None, strict=False, libm_budget=None,
          falsify=None):
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

    ``refine``: ``None`` (default — the refinement never runs, and every
    existing path is byte-identical) or ``"affine"`` to attempt the
    affine-form (zonotope) refinement on each interval-undecided
    obligation AFTER interval judging and BEFORE any solver escalation
    (:mod:`stelling.affine`): the escalation then sees only what the
    refinement left undecided. Never on by default — the solver-opt-in
    precedent; an unknown value raises eagerly at entry. The stamp
    records the refinement ran, each affine-decided obligation's
    mechanism note names affine, and the VERIFIED widen re-check runs at
    the same depth (refined iff the original refined).

    ``solver``: restrict the SMT portfolio to a single solver by name
    (``"z3"`` or ``"cvc5"``). ``None`` (default) uses the full installed
    portfolio. Only meaningful when ``solver_timeout_ms`` is also passed.

    ``libm_budget``: **required to use ``exp`` or ``pow`` under**
    ``semantics="ieee"``. ``None`` (the default) makes those transfers
    DECLINE, with the measured evidence and the exact line to write in
    the decline. Pass a shipped profile name — ``"xla-cpu-2026-08"`` — or
    a :class:`stelling.propagate.LibmBudget` of your own, to declare how
    far the function your backend executes may be from the true value.
    Under ieee semantics a verdict is a claim about the float the program
    computes, and stelling's bracket is built around the ``math`` module
    of the host running the analysis; the two differ by up to 5.5 float32
    ulps on the measured backend, so the assumption is made an explicit,
    named, dated declaration instead of a silent default (audit 0.2.0 S9,
    S11). It is stamped on the verdict as **declared, not verified**, and
    a budget smaller than your backend's real error mints a VERIFIED
    stelling cannot catch. Passing it under ``semantics="real"`` raises —
    it has no meaning there.

    ``falsify``: ``None`` (default — the probe never runs,
    :mod:`stelling.falsify` is never imported, and every existing path is
    byte-identical) or ``"sample"`` to try, after a VERIFIED, to FALSIFY
    it by executing the real program at concrete points inside the
    declared set. **UNRELEASED AND UNAUDITED**; it is here to be
    measured, not to be relied on, and it exists because this library is
    asymmetric about its two answers: a REFUTED's witness is replayed
    through the real program, and a VERIFIED — a universal claim with no
    witness — has had nothing downstream at all.

    Two properties of it are not negotiable and are enforced rather than
    described. First, **it can only refute**: finding nothing adds no
    confidence, and the note it appends says so in its own sentence, so
    no reader can take a probed VERIFIED for a better VERIFIED. Second,
    when it does find a violation it **raises**
    :class:`stelling.falsify.VerifiedFalsified` rather than returning any
    status, because a violated discharge is a defect in *stelling*, not a
    finding about the caller's program, and none of VERIFIED/REFUTED/
    UNKNOWN can say that. The module's docstring carries the argument,
    the two dispositions that were rejected, the list of what the probe
    is allowed to import and why that set is the independent one, and —
    measured, not assumed — the class of defect it cannot reach.

    Version, precision, and solver stamps are filled from the live
    environment; the precision entry records the *actual*
    ``jax_enable_x64`` state at trace time, not an assumption.
    """
    from stelling import ir
    from stelling.verdict import declined as _declined

    if semantics not in ("real", "ieee"):
        raise ValueError(f"semantics must be 'real' or 'ieee', got {semantics!r}")
    if semantics == "ieee" and solver_timeout_ms is not None:
        raise ValueError(
            "solver_timeout_ms and semantics='ieee' are contradictory: "
            "the SMT backends emit over the reals (QF_LRA/QF_NRA) and "
            "cannot model format-specific rounding or overflow. Remove "
            "solver_timeout_ms for ieee mode, or use semantics='real' "
            "for solver escalation."
        )

    try:
        verdict, _ = _pipeline(
            harness,
            vacuity_mode=vacuity_mode,
            semantics=semantics,
            solver_timeout_ms=solver_timeout_ms,
            refine=refine,
            solver=solver,
            libm_budget=libm_budget,
            falsify=falsify,
        )
    except ir.TranscriptionError as e:
        # stelling could not READ the query. That is a capability gap, not a
        # judgment and not a harness defect, so it returns a status rather
        # than raising: a batch caller — a CI soak, a graph-wide pass — must
        # be able to record "declined" for one node and carry on to the next.
        # Raising makes the first unsupported node kill the run.
        #
        # Deliberately NOT caught here: harness defects (an empty declared
        # set, an unsatisfiable assume) and jax's own tracing failures. The
        # first are the caller's bug and must stay loud; the second happen
        # upstream of stelling, and swallowing them would mislabel "jax
        # cannot trace this program" as "stelling does not cover it".
        if strict:
            raise
        return _declined(str(e))
    return verdict


def _pipeline(harness, *, vacuity_mode, semantics="real", solver_timeout_ms,
              refine=None, solver=None, libm_budget=None, falsify=None):
    """The one pipeline behind :func:`check` — trace, propagate, optional
    affine refinement (``refine="affine"``, never on by default), optional
    solver escalation, stamped verdict assembly, and the VERIFIED widen
    re-check at the same pipeline depth (refined iff the original
    refined, escalated iff the original escalated).

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
    from stelling.vacuity import (
        _MODES, NestedDeclaration, declaration_bounds, unwidened, widen,
    )
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
    # the refinement dial is validated eagerly like the vacuity mode: an
    # unknown value must raise at entry, never ride green until the day
    # it would have mattered
    if refine not in (None, "affine"):
        raise ValueError(
            f"refine must be None or 'affine', got {refine!r}"
        )
    # the falsification dial, validated eagerly for the reason every other
    # dial here is: it only DOES anything on a VERIFIED, so a typo'd value
    # would ride green through every UNKNOWN path until the day a VERIFIED
    # happened.  The accepted values live in `stelling.falsify.FALSIFY_MODES`
    # and are spelled out here rather than imported, because importing the
    # probe module imports jax, and this validation must happen in every
    # environment -- including the ones where `falsify` is left off.
    if falsify not in (None, "sample"):
        raise ValueError(
            f"falsify must be None or 'sample', got {falsify!r}"
        )
    if solver is not None and solver not in ("z3", "cvc5"):
        raise ValueError(
            f"solver must be None, 'z3', or 'cvc5', got {solver!r}"
        )
    # the libm budget is validated eagerly like every other dial — a
    # typo'd profile name must raise where it was written, not arrive as a
    # decline three layers down that reads like a stelling limitation
    from stelling.propagate import (
        LIBM_BUDGET_REAL_MODE_REFUSAL, resolve_libm_budget,
    )

    libm_budget = resolve_libm_budget(libm_budget)
    if libm_budget is not None and semantics != "ieee":
        raise ValueError(LIBM_BUDGET_REAL_MODE_REFUSAL)

    from stelling._tripwire import (
        _pop_gate, _push_gate, fires_count as _fires_count,
    )
    from stelling._tripwire import _adapter_jax as _adapter

    armed = _fires_count() is not None
    if armed:
        recorder_before = _adapter._installed.get("recorder")
        _push_gate()
        try:
            # Fresh closure defeats jax.make_jaxpr's identity cache, ensuring
            # the fold rule fires fresh and the gate can observe narrowings.
            cj = trace(lambda: harness())
        finally:
            narrowings = _pop_gate()
        # If the recorder changed (disarm/rearm during trace) or the tripwire
        # was disarmed, the wrapper may have stopped firing mid-trace while
        # narrowings had already occurred. Refuse to certify.
        recorder_after = _adapter._installed.get("recorder")
        if recorder_after is not recorder_before or _fires_count() is None:
            narrowings = max(narrowings, 1)
    else:
        cj = trace(harness)
        narrowings = 0

    if narrowings > 0:
        from stelling.verdict import Stamp, Verdict, solver_absent

        versions = dict(
            stelling_version=_stelling.__version__,
            jax_version=jax_version(),
            precision_config=f"jax_enable_x64={x64_enabled()}",
        )
        reason = (
            f"trace unfaithful: {narrowings} integer narrowing(s) detected "
            f"during tracing — the jaxpr does not represent the program as "
            f"written. Enable the overflow tripwire (pytest -p stelling."
            f"overflow) to see which constants were narrowed."
        )
        stamp = Stamp(
            stelling_version=versions["stelling_version"],
            jax_version=versions["jax_version"],
            precision_config=versions["precision_config"],
            device_class="none: no concrete execution in this verdict",
            arithmetic_mode="not reached: trace gate refused",
            semantics="not reached: trace gate refused",
            query_content_hash=cj.content_hash(),
            nonvacuity="not reached: trace gate refused",
            solver=solver_absent(
                "not reached: the trace gate refused before propagation"
            ),
            transfer_tiers=(),
            transfer_provenance=(),
            assumptions=(),
            coverage="not reached: trace gate refused",
            top_despite_coverage=None,
        )
        v = Verdict(
            status="UNKNOWN",
            obligations=(),
            stamp=stamp,
            notes=(reason,),
            witnesses=(),
        )
        return v, cj

    p = propagate(cj, semantics=semantics, libm_budget=libm_budget)
    versions = dict(
        stelling_version=_stelling.__version__,
        jax_version=jax_version(),
        precision_config=f"jax_enable_x64={x64_enabled()}",
    )

    def _finish(closed, prop):
        # the refinement slots here, inside the ONE finishing path both
        # the original run and the widen re-check go through — that is
        # what makes the re-check run at the same depth (refined iff the
        # original refined, escalated iff the original escalated).
        # Returns the verdict AND the refinement record: the vacuity line
        # derives its reduced-power disclosure from the two records
        # (which layers DECIDED originally vs in the re-check), never
        # from flags.
        refinement = None
        if refine == "affine":
            from stelling.affine import refine_propagation

            prop, refinement = refine_propagation(closed, prop)
        if solver_timeout_ms is None:
            return (
                make_verdict(closed, prop, refinement=refinement, **versions),
                refinement,
            )
        from stelling.solvers import SolverConfig, escalate, make_solver_verdict

        only = (solver,) if solver is not None else None
        esc = escalate(closed, prop, SolverConfig(timeout_ms=int(solver_timeout_ms), only=only))
        return (
            make_solver_verdict(
                closed, prop, esc, refinement=refinement, **versions
            ),
            refinement,
        )

    v, orig_ref = _finish(cj, p)
    if v.status != "VERIFIED":
        # widening cannot rescue an UNKNOWN/REFUTED; nothing to check
        return v, cj

    # THE FALSIFICATION PROBE, and it is DEFAULT-OFF: with `falsify=None`
    # -- every documented call today -- neither the import nor the branch
    # below happens, and the verdict returned is byte-identical to the one
    # this line was inserted above.  `tests/test_falsify_default_path.py`
    # measures that rather than asserting it.
    #
    # It sits HERE, on the VERIFIED path and beside the widen re-check,
    # because this is where a VERIFIED first exists on the one pipeline
    # both front doors take, and because it answers the question the widen
    # re-check does not: widen asks whether the envelope was load-bearing,
    # this asks whether the discharge is TRUE of the program.
    #
    # It may RAISE, and that is the design (`stelling.falsify`'s docstring
    # carries the argument and the two dispositions rejected).  A firing
    # means the analysis discharged something the program violates at a
    # point the analysis admitted: a defect in stelling, not a finding
    # about the caller's program, and there is no verdict status that says
    # so.  Nothing is caught here -- letting it past this frame is the
    # whole point.
    #
    # WHAT THE NOTE MAY SAY IS CONSTRAINED. `stamp_line` is a sentence
    # about work done and carries its own disclaimer, because a VERIFIED
    # that grew a line a reader could take as "probed, therefore better"
    # would be a verdict that reads above its evidence -- and a probe that
    # can only refute produces no evidence at all when it finds nothing.
    # THE STATUSES COME FROM THE VERDICT, NOT FROM THE PROPAGATION, and
    # the difference is the probe's whole reach on the solver path.
    # `propagate` leaves an obligation "unknown" and escalation upgrades it
    # to "discharged" in `make_solver_verdict`; measured, `x**2 <= 150` over
    # [0, 10] is propagation-unknown and solver-discharged. Passing the
    # propagation's view made the probe decline with "no obligation was
    # discharged" on exactly the VERIFIEDs a solver decided -- which is
    # where `VERIFIED_BARRED_PRIMITIVES` lives and where an emission defect
    # that MISSED a violation would be. The claim under attack is the one
    # the verdict makes, so it is the one the probe is handed.
    if falsify is not None:
        from stelling.falsify import probe as _probe

        _report = _probe(
            harness,
            statuses=[o.status for o in v.obligations],
            semantics=semantics,
        )
        v = dataclasses.replace(v, notes=v.notes + (_report.stamp_line(),))

    # THE ONE RULE (audit finding 4 + the depth defect, unified): the
    # load-bearing note may be emitted ONLY IF widen actually moved EVERY
    # declared bound. Both defects were the same fact seen twice -- the
    # verdict claimed the envelope was widened when part of it was not:
    # a nested declaration the rewrite never reaches, and a POINT declaration
    # that `inputs-only` holds still by design. Neither is a bug in widen;
    # the bug was claiming otherwise. So this checks the claim directly
    # rather than enumerating the shapes that can falsify it.
    before = declaration_bounds(cj)
    try:
        wcj = widen(cj, mode=vacuity_mode)
    except NestedDeclaration as e:
        wcj, unwidened_why = cj, str(e)
    else:
        unwidened_why = unwidened(before, declaration_bounds(wcj))
    if (unwidened_why is not None or wcj == cj
            or wcj.content_hash() == cj.content_hash()):
        why = unwidened_why or "the widened query is identical to the original"
        # deliberately does NOT contain the phrase the note uses: a reader
        # (or a test) scanning for that phrase must not match a line whose
        # whole point is that no such claim was made.
        tail = (
            "so the envelope's role in this verdict is left uncharacterised"
        )
        vac_line = (
            f"vacuity instrument inert (mode={vacuity_mode}): {why} — {tail}"
        )
        stamp = dataclasses.replace(
            v.stamp, assumptions=v.stamp.assumptions + (vac_line,)
        )
        return dataclasses.replace(v, stamp=stamp), cj

    wide, wide_ref = _finish(
        wcj, propagate(wcj, semantics=semantics, libm_budget=libm_budget)
    )
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
        # State the MEASUREMENT, never the load-bearing inference: the
        # old dash-clause ("the declared envelope is load-bearing for
        # this VERIFIED") asserted an inference the re-run does not
        # establish, and it is measurably false for range theorems the
        # widened mechanisms cannot re-derive (an interval-⊤ square, an
        # affine-decided cancellation whose widened boxes decline by
        # construction). What the re-run measured is exactly this:
        vac_line = (
            f"vacuity checked (mode={vacuity_mode}): no obligation "
            f"discharges with the declared bounds widened — under the "
            f"mechanism(s) that ran, this VERIFIED was not re-derivable "
            f"without the declared envelope"
        )
        # Reduced-power disclosure, derived from the two measured
        # refinement records (decided counts) and the re-check's actual
        # solver invocations — not from flags: when the affine domain
        # decided obligations originally but decided nothing in the
        # widened re-check, the re-check ran WEAKER than the original,
        # and envelope-independence of those obligations was not
        # measured at all.
        if (
            orig_ref is not None
            and orig_ref.decided
            and not (wide_ref is not None and wide_ref.decided)
        ):
            wide_solver_ran = isinstance(wide.stamp.solver, tuple)
            ran = (
                "interval and solver escalation"
                if wide_solver_ran
                else "interval-only"
            )
            hint = (
                ""
                if wide_solver_ran
                else "; an explicit solver_timeout_ms measures it"
            )
            vac_line += (
                f". The re-check ran weaker than the original ({ran}: the "
                f"affine refinement declines unbounded boxes by "
                f"construction), so envelope-independence of the "
                f"affine-decided obligation(s) was not measured{hint}."
            )
        notes = v.notes
    # WHAT THE MEASUREMENT IS NOT, WHEN THE PRECONDITION IS UNSETTLED (audit
    # 0.2.0 S7's second half). "No obligation discharges with the bounds
    # widened" is a real dependence on the envelope, and a reader takes it
    # for SUBSTANTIVE — this VERIFIED says something. On a run whose assumed
    # region was never shown non-empty it can be the exact opposite, and the
    # audit measured that case: `dt ∈ [5,10]`, `dt_max ∈ [0,1]`,
    # `assume(dt < dt_max)`. Widening makes `dt < dt_max` SATISFIABLE again,
    # so the negated obligation becomes sat and the re-check "fails to
    # re-derive" — the instrument measured a dependence on the envelope, just
    # not the one the sentence is read as claiming.
    #
    # A DETECTED empty case no longer reaches here: `solvers._dispatch_
    # obligation` raises `UnsatisfiableAssumptionError` when one obligation's
    # script states the whole contradiction, so there is no VERIFIED to
    # stamp. An UNDETECTED one still does, and the qualification is what it
    # gets: a contradiction spread across obligation cones (audit B3) leaves
    # every script with a satisfiable relaxation, so nothing proves the
    # region empty, and it arrives here through the same uncertified line as
    # the undecided case. What the qualification therefore covers: an
    # undecided region check, a cone-split empty region, and an interval
    # narrowing/drop whose satisfiability was never certified. Keyed on the
    # shared prefix rather than on a list of mechanisms, so a mechanism added
    # later qualifies the sentence without anyone remembering to come here.
    from stelling.propagate import UNCERTIFIED_PRECONDITION_PREFIX

    if any(
        a.startswith(UNCERTIFIED_PRECONDITION_PREFIX)
        for a in v.stamp.assumptions
    ):
        vac_line += (
            ". WHAT THIS MEASUREMENT DOES NOT SAY: this run's assumed region "
            "was never shown non-empty (see the 'precondition satisfiability "
            "uncertified' line), so every obligation of it may be vacuously "
            "true of an empty precondition. Widening a declared bound can "
            "make an unsatisfiable precondition SATISFIABLE again, so neither "
            "outcome of this re-check is evidence that the VERIFIED is "
            "substantive"
        )
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
