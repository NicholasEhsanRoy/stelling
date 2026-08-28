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
          falsify=None, boundary="opaque"):
    """Run a precondition harness end-to-end and return the stamped
    :class:`stelling.verdict.Verdict` — with the vacuity check built in:
    **this entry point cannot return an unchecked VERIFIED.**

    ``harness`` is a zero-argument function that declares inputs (via
    the templates above or :func:`stelling.harness.any_array` directly)
    and returns its obligations. The verdict's status is ``VERIFIED``,
    ``REFUTED`` (with a replay-confirmed concrete witness when a solver
    found one), or ``UNKNOWN`` — never guessed; anything the analysis
    could not decide says so, with the reason quoted in the notes.

    **With the overflow tripwire armed** (``pytest -p stelling.overflow``)
    this call also runs the trace gate, which refuses to judge a jaxpr that
    does not represent the program as written — and refuses, separately and
    in different words, when it could not watch the whole trace. Making the
    watch complete against jax's caches costs one ``jax.clear_caches()``
    per call, which is process-global and drops the caller's own compiled
    functions;
    ``docs/overflow-tripwire.md`` prices it. Nothing of this happens when
    the tripwire is not armed, which is the default.

    The gate watches a FINITE set of routes by which a constant can be
    narrowed, and its silence is worth exactly that set: routes such as
    ``jnp.full(shape, N, dt)`` destroy the constant before the watched site
    and are VERIFIED with zero fires. The eviction closes the warm-cache
    route against jax's own caches in a single-threaded process, and no
    further — a memo jax does not own, or a competing thread, is outside
    it. Both lists are enumerated, with their measurements, in
    ``stelling._tripwire.report.UNCOVERED`` and in
    ``tests/test_tripwire_gate_coverage.py::GATE_COVERAGE``.

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
    declared set. **SHIPPED, AND ``experimental``** — the level in
    ``DOCUMENTATION_ARCHITECTURE.md`` §8.5's table, where it means *"may
    change without notice"* with guarantee *"none"*, so this keyword,
    :func:`stelling.falsify.probe`'s signature and every name in that
    module's ``__all__`` may change or be withdrawn in any release with
    **no deprecation cycle and no notice**. It is not that table's
    ``provisional``, which is the neighbouring level and DOES promise one
    minor's notice; this paragraph and that module's heading both said
    the wrong one of the two, next to their own disclosure that
    ``probe()``'s first parameter changed name and type inside one
    release cycle — which is a change with no notice, i.e. the definition
    of the word they were not using. The level is read from
    :data:`stelling.falsify.STABILITY`, which is the one place it is
    written down.

    **AUDITED IS A DIFFERENT QUESTION FROM THE LEVEL, AND IT IS
    ANSWERED.** The fire condition's exact reading is checked against an
    independent ``Fraction`` oracle sharing no code with that module, on
    every run, by ``tests/test_probe_oracle.py`` (297 gate readings, 270
    obligation readings, all 270 point-comparisons agreeing, and the
    counts asserted there rather than quoted here — as is what that file
    does NOT reach, which is 22 of the 32 single-edit defects it was
    measured against). That is evidence about what the probe DOES; the
    level above is a statement about what its SURFACE may do next, and
    neither buys the other. This sentence used to cite *"363 gate
    readings and 363 agreements"*, a figure that appeared nowhere in this
    repository except the two sentences citing it; a number a reader
    cannot re-derive is the same defect as a check that does not exist,
    which is a rule this project applies to everything else it ships.
    :mod:`stelling.falsify`'s docstring opens with the SIX DISCLOSURES,
    which are a different axis from the level and are not implied by it:
    a level is a promise about future changes to the surface, and those
    six are what the instrument does not do today. **Two of its six items
    are a decline standing in for a guard**: an ``ieee`` firing cannot
    tell a caller's own ``libm_budget`` declaration from an unsound
    analysis, so the firing
    message names the declaration instead of guessing; and the rational
    replay does not descend a loop or branch body, which this release
    GUARDS — the descent refuses a body that does not run once per
    equation, by a name, an iteration count and a signature — and does not
    close. **A THIRD ITEM IS NOT ONE OF THEM AND THIS SENTENCE USED TO
    SAY IT WAS**: the analysis's constrained region is never read — the
    one this file used to call *"the one hole"* — and there is no decline
    standing in for anything there, because the probe does not read the
    analysis's region correctly OR incorrectly; it does not read it at
    all. It is the MIRROR of that class rather than an instance of it, it
    is disclosed and open, and it is the one item of the six that cannot
    be closed inside that module's import rule. Counting it among the
    guarded ones was an overclaim in the reassuring direction. The same
    docstring carries the measured reach on ordinary ``jnp`` code. It
    exists because this library is asymmetric about its two answers: a
    REFUTED's witness is replayed through the real program, and a
    VERIFIED — a universal claim with no witness — has had nothing
    downstream at all.

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

    **SO THIS FUNCTION CAN RAISE, AND ONE OF THE TWO IS OUTSIDE
    ``Exception``.** With ``falsify="sample"`` a caller must expect
    :class:`stelling.falsify.VerifiedFalsified` (an ``AssertionError``:
    the probe broke a VERIFIED, which is stelling's defect) and
    :class:`stelling.falsify.ProbeInvariantViolated` (a
    ``BaseException``: a fact the probe's own readings rest on did not
    hold, so it has nothing to say about the verdict either way). The
    second is deliberately outside ``Exception`` AND outside
    ``AssertionError`` so that neither the ordinary batch idiom nor the
    catch-a-soundness-event idiom can silently swallow an instrument's
    alarm — which means **``except Exception:`` will not contain it**, and
    a batch runner that must survive one has to name it. With
    ``falsify=None`` neither can be raised at all. A probe that CANNOT
    run does not raise: an unsampleable declaration, a dtype the sampler
    cannot construct, or a 64-bit declaration under ``jax_enable_x64=0``
    all produce a verdict carrying a named decline in its notes. See
    ``docs/preconditions.md``.

    The same keyword is accepted by
    :func:`stelling.contracts.check_contract` and
    :func:`stelling.inductive.check_inductive_step`, which mint VERIFIEDs
    through this same pipeline. It reached only THIS door when it landed,
    which made the probe's reach an accident of which function carried the
    keyword rather than a decision; the decision is that every door that
    can mint a VERIFIED can ask for the downstream check.

    ``boundary``: ``"opaque"`` (the default) or ``"transparent"``, passed
    straight through to :func:`stelling.propagate.propagate` — and to the
    VERIFIED widen re-check, at the same position, for the same reason the
    re-check runs at the same pipeline depth: a control run under a
    different rule than the run it controls measures the rule and not the
    envelope. Under the default this door is byte-identical to the one
    that shipped without the keyword. Under ``"transparent"`` the
    strict-sign certificate crosses the sub-jaxpr boundaries the walk
    already enters (IN and OUT for the unconditional wrappers, IN only for
    a ``cond`` branch); that function's docstring carries the argument,
    the exclusions and the ieee behaviour.

    **IT MOVES VERDICTS IN BOTH DIRECTIONS, AND THIS SENTENCE USED TO NAME
    ONLY ONE.** It read *"which can turn an UNKNOWN into a VERIFIED on a
    query whose division sits inside a ``jit``"* — true, and the reason
    the dial exists, but half of what a caller switching it on is signing
    up for. It can equally turn an UNKNOWN into a **REFUTED**, because
    :func:`stelling.interval.boundary_div` returns a HALF-INFINITE box and
    a half-infinite box can make an upper-bound obligation definitely
    false where ⊤ left it undecided. That second direction is a property
    of the certificate rather than of the dial — driven at `8dae8cb`, with
    no wrapper and no dial: ``x ∈ [0, 2]``, ``assert_(1/x < 0.4)`` is
    ``unknown`` without the ``assume`` and ``violated-over-set`` with it —
    but the dial makes the certificate reach queries it did not reach
    before, and therefore makes that direction reach them too. Pinned at
    ``tests/test_boundary_dial.py::test_the_dial_can_also_move_UNKNOWN_to_REFUTED``.
    Neither direction can be a WRONG answer for the reason the certificate
    is sound; both are answers the default does not give.

    The verdict discloses the position in its stamped assumption lines
    when it is off the default, and ``docs/reading-a-verdict.md`` says how
    to read a stamp that carries no such line. Under ``semantics="ieee"``
    the dial is INERT — the certificate is a claim about ℝ — and a run
    that asked for it there stamps a line saying so rather than the
    position line, which would otherwise tell the reader that a rule
    unsound under ieee had been allowed.

    **IT IS NOT ACCEPTED BY THE TWO SIBLING DOORS**,
    :func:`stelling.contracts.check_contract` and
    :func:`stelling.inductive.check_inductive_step`, and that is the
    ``semantics`` precedent rather than an omission: neither of those
    takes ``semantics`` either, because both are ANALYSIS-MODE dials that
    change what the walk is allowed to conclude, while ``falsify`` — which
    all three do take — is a downstream check on a VERIFIED those doors
    can mint. A caller who wants a boundary-transparent contract or
    inductive step runs the trace through :func:`check` or through
    :func:`stelling.propagate.propagate` directly; plumbing it is a
    decision about those doors' surfaces and belongs with whatever also
    plumbs ``semantics``.

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
            boundary=boundary,
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
              refine=None, solver=None, libm_budget=None, falsify=None,
              boundary="opaque"):
    """The one pipeline behind :func:`check` — trace, propagate, optional
    affine refinement (``refine="affine"``, never on by default), optional
    solver escalation, stamped verdict assembly, and the VERIFIED widen
    re-check at the same pipeline depth (refined iff the original
    refined, escalated iff the original escalated).

    THE TRACE GATE, when the overflow tripwire is armed, has THREE states
    and not two: observed-and-clean proceeds, observed-and-narrowed refuses
    with ``trace unfaithful``, and NOT-FULLY-OBSERVED refuses in its own
    words. The third exists because the gate's silence is evidence of
    nothing when part of the trace was replayed from a cache, the instrument
    stopped watching, or the instrument was DISPLACED by something bound
    over it, and reporting any of those as "0 narrowings detected" — or, as
    B14 left it, as "1 narrowing detected" — describes a measurement nobody
    made. Observation is made complete — with respect to
    JAX's caches, in a single-threaded process, which is the whole of what
    is claimed — by EVICTING those caches rather than by detecting
    incompleteness; the comment at the gate carries the measurement that
    decides between those, and ``_tripwire.report.UNCOVERED`` carries what
    the two qualifiers leave outside.

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
    from stelling._jax_compat import jax_version, trace_with_jaxpr, x64_enabled
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
        LIBM_BUDGET_REAL_MODE_REFUSAL, _check_boundary, resolve_libm_budget,
    )

    # the boundary dial, validated by the SAME function `propagate` uses
    # rather than by a second spelling of its value set here.
    #
    # **THE REASON GIVEN FOR VALIDATING IT HERE RATHER THAN IN `check` WAS
    # WRONG, AND THE SITING IS RIGHT ANYWAY.** It read: *"validated in
    # this shared helper and not in `check` because `check_contract`
    # reaches the pipeline without passing through `check`, and a dial
    # checked at only one of two doors is a dial that is unchecked at the
    # other."* The first clause is true and the inference does not follow:
    # `check_contract` has no `boundary` parameter at all — pinned by
    # `tests/test_boundary_dial_jax.py::test_the_two_SIBLING_doors_deliberately_do_not_take_the_dial`
    # — so it cannot supply a bad value and there is no second door to be
    # unchecked. The check placed there today is a check of `check`'s own
    # argument reached one frame down.
    #
    # It stays here because THIS is the function every door's boundary
    # argument must pass through, whichever door grows one next: a door
    # that gains the keyword and forwards it acquires the refusal by
    # forwarding, and cannot ship an unvalidated dial by forgetting a line
    # at its own entry. Eager either way — this runs before the trace, so
    # a typo raises where it was written and not as a decline three layers
    # down.
    _check_boundary(boundary)
    libm_budget = resolve_libm_budget(libm_budget)
    if libm_budget is not None and semantics != "ieee":
        raise ValueError(LIBM_BUDGET_REAL_MODE_REFUSAL)

    from stelling._tripwire import (
        _pop_gate, _push_gate, displaced as _displaced,
        evict_trace_caches as _evict_trace_caches, fires_count as _fires_count,
    )
    from stelling._tripwire import _adapter_jax as _adapter

    # THE GATE HAS THREE STATES, NOT TWO (B15). It used to have two — clean
    # and narrowed — and read its own silence as the first of them. It is not:
    # a `@jax.jit` helper whose trace cache some earlier trace already warmed
    # is REPLAYED, not traced, so the fold rule never runs over its body and
    # the gate's zero establishes "I observed no narrowing", never "no
    # narrowing occurred". Measured on jax 0.11.0: one harness with a jitted
    # helper, checked four times, gives UNKNOWN then three VERIFIEDs; two
    # DIFFERENT harnesses sharing one jitted helper give UNKNOWN then a WRONG
    # VERIFIED about a program whose constant was destroyed.
    #
    # The fix is EVICTION, not detection, and that is a measurement rather
    # than a preference: `jax.jit(f, inline=True)` replays a warm body and
    # leaves NO nested jaxpr behind to detect the replay by, and jax publishes
    # no per-jit trace counter on a public surface. Emptying the cache first
    # makes the observation complete WITH RESPECT TO JAX'S CACHES, IN A
    # SINGLE-THREADED PROCESS. It empties jax's caches and no others, so a
    # constant narrowed into a memo jax does not own (`jaxpr_as_fun` over a
    # saved jaxpr, a user `lru_cache`, `jax.closure_convert`) is still
    # unobserved; and jax's cache is process-global while this counter is
    # per-thread, so a competing thread can re-warm a body inside the
    # eviction-to-trace window — measured 0/400 wrong VERIFIED
    # single-threaded and 247/400 with four competing threads, against
    # 399/400 before the eviction existed. Both are disclosed with their
    # numbers in `report.UNCOVERED`. The third state is for when even the
    # eviction could not be done — see `unobserved` below.
    armed = _fires_count() is not None
    unobserved = None
    if armed:
        recorder_before = _adapter._installed.get("recorder")
        eviction = _evict_trace_caches()
        _push_gate()
        try:
            # Fresh closure defeats jax.make_jaxpr's identity cache; the
            # eviction above defeats every trace cache BELOW it, which the
            # fresh closure never reached.
            cj, jaxpr = trace_with_jaxpr(lambda: harness())
        finally:
            narrowings = _pop_gate()
        # THREE ways the watch can be PARTIAL, and none of them is a
        # narrowing. The recorder changing identity, or the tripwire being
        # disarmed, means the wrapper stopped counting partway through; a
        # DISPLACEMENT means our wrapper is no longer the live entry, so it
        # was never called at all; a failed eviction means it never got to
        # see the cached regions. All three are "we did not look", and B14's
        # `narrowings = max(narrowings, 1)` said "1 integer narrowing(s)
        # detected" about the first of them — sending a reader to hunt a
        # narrowed constant that was never observed.
        #
        # THE DISPLACEMENT CHECK IS B15's AUDIT FINDING, and it is the one
        # that was silent rather than merely mislabelled. `live_check()`
        # existed, cost nothing and was consulted NOWHERE on this path:
        # rebinding the const-fold registry entry over our wrapper after
        # arming leaves the recorder's identity unchanged and
        # `fires_count()` unchanged — so both tests below pass — while our
        # wrapper is never called again, so the gate's counter stays at zero
        # and `check()` returns **VERIFIED on a route the inventory calls
        # `watched`**. Measured on `main` before this batch, byte-identical.
        # It is asked AFTER the trace so that a rebind performed DURING one
        # is caught as well as one that pre-dates it; a patch installed and
        # removed inside the window is not detectable by anything here and
        # stays disclosed in `report.UNCOVERED`.
        recorder_after = _adapter._installed.get("recorder")
        displaced = _displaced()
        if recorder_after is not recorder_before or _fires_count() is None:
            unobserved = (
                "the overflow tripwire stopped watching partway through this "
                "trace (it was disarmed, or re-armed onto a different "
                "recorder, while the harness was being traced)"
            )
        elif displaced:
            # NAMED RATHER THAN DESCRIBED, because more than one hook can be
            # armed and the consequence differs. A displaced `const-fold`
            # hook means the counter below is the count of a wrapper that was
            # not running; a displaced `eager` hook means an out-of-range
            # constant destroyed at construction inside this harness would
            # have gone unreported instead of raising. Either way an
            # instrument that was watching this process was not, and this
            # verdict would rest on an observation nobody completed.
            unobserved = (
                f"one of stelling's hooks was DISPLACED — something else is "
                f"bound over stelling's wrapper for: {', '.join(displaced)} — "
                f"so an instrument that was watching this process was not "
                f"running while this harness was traced"
            )
        elif eviction != "evicted":
            unobserved = (
                f"jax's trace caches could not be emptied before this trace "
                f"({eviction}), so any jit helper whose cache was already "
                f"warm was replayed from that cache instead of being traced "
                f"under the instrument"
            )
    else:
        cj, jaxpr = trace_with_jaxpr(harness)
        narrowings = 0

    if narrowings > 0 or unobserved is not None:
        from stelling.verdict import Stamp, Verdict, solver_absent

        versions = dict(
            stelling_version=_stelling.__version__,
            jax_version=jax_version(),
            precision_config=f"jax_enable_x64={x64_enabled()}",
        )
        if narrowings > 0:
            reason = (
                f"trace unfaithful: {narrowings} integer narrowing(s) "
                f"detected during tracing — the jaxpr does not represent the "
                f"program as written. Enable the overflow tripwire (pytest -p "
                f"stelling.overflow) to see which constants were narrowed."
            )
            if unobserved is not None:
                # The count is a floor, and saying so is the whole point of
                # keeping the third state separate: a reader who fixes the
                # narrowing this names must not read the next clean run as
                # proof there was only one.
                reason += (
                    f" That count is a LOWER BOUND and not a total: part of "
                    f"this trace was not observed at all — {unobserved}."
                )
        else:
            # THE THIRD STATE GETS ITS OWN SENTENCE. "No narrowing was seen"
            # and "no narrowing occurred" are different claims and only the
            # first one was ever established here; a reader sent to look for a
            # narrowed constant, when the real answer is that nobody looked,
            # has been sent to the wrong place.
            reason = (
                f"trace NOT FULLY OBSERVED: the overflow tripwire could not "
                f"watch all of this trace, so this run has no evidence either "
                f"way about the part it did not watch. THIS IS NOT A REPORT "
                f"THAT A CONSTANT WAS NARROWED — none was seen, and none was "
                f"seen not to be. Cause: {unobserved}."
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

    p = propagate(
        cj, semantics=semantics, libm_budget=libm_budget, boundary=boundary
    )
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
    #
    # AND THE PROGRAM UNDER ATTACK IS THE ONE THE VERDICT IS ABOUT, WHICH
    # USED TO BE LEFT TO A JAX MEMO. The probe was handed `harness` and
    # traced it AGAIN for itself; whether that second `jax.make_jaxpr`
    # re-ran the body was decided by jax's trace cache, and the armed
    # branch above defeats that cache deliberately (a fresh closure, after
    # `jax.clear_caches()`). Measured: harness-body invocations per
    # `check()` were 1 unarmed and 2 under `pytest -p stelling.overflow`
    # with `falsify="sample"` -- so an impure harness gave the probe a
    # genuinely different program, and NOTHING compared the two. Driven
    # through the public API, both directions: a CORRECT VERIFIED made to
    # raise "stelling is UNSOUND at this query", and a VERIFIED carrying a
    # "NO VIOLATION WAS FOUND" work report about a program the verdict is
    # not about -- the same call, the same stelling, VERIFIED under
    # `pytest` and "UNSOUND" under `pytest -p stelling.overflow`, which
    # `check()`'s own docstring recommends. `jaxpr` here is jax's own
    # object from the ONE trace this pipeline took, so the probe has no
    # second program to disagree with. The transcription `cj` is NOT
    # passed: the probe may not read `stelling.ir`, and that rule is
    # untouched by handing it the jax object it was tracing for itself.
    #
    # THE STAMPED ASSUMPTIONS GO WITH THE STATUSES, and for the same
    # reason. A firing says "stelling is UNSOUND at this query", but an
    # `ieee` VERIFIED resting on a caller-declared `libm_budget` is not
    # stelling's own claim -- the stamp says so in the words "DECLARED,
    # NOT VERIFIED" -- and under-declaring it made the probe raise a
    # soundness alarm against stelling for the caller's declaration.
    # Passed as plain strings, exactly as `statuses` is, so the probe
    # still imports nothing that produced them.
    if falsify is not None:
        from stelling.falsify import probe as _probe

        _report = _probe(
            jaxpr,
            statuses=[o.status for o in v.obligations],
            semantics=semantics,
            assumptions=v.stamp.assumptions,
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
        # THE SAME DIAL AS THE RUN IT CONTROLS. The widen re-check exists
        # to ask whether the declared envelope was load-bearing, and it
        # can only ask that if everything else about the two runs is
        # equal — the same reason it runs refined iff the original
        # refined and escalated iff the original escalated.
        wcj,
        propagate(
            wcj,
            semantics=semantics,
            libm_budget=libm_budget,
            boundary=boundary,
        ),
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
