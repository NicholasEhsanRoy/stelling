# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Inductive step verification for loop invariants.

Given a loop body and declared invariant bounds, verify that one step of the
body preserves the invariant: if the state starts within bounds, it stays
within bounds after one iteration.  When verified **unconditionally**,
induction gives the invariant for all iterations — see the assume caveat
below for what makes a VERIFIED conditional instead.

The single entry point is :func:`check_inductive_step`.  It constructs a
harness that declares inputs at the invariant bounds, runs one step of the
body, and asserts each output is within the same bounds.  The harness is
then fed to :func:`stelling.preconditions.check` for the interval/solver
verdict.

**What this proves:** the invariant is PRESERVED by one step (the inductive
step of a proof by induction).

**What this does NOT prove:**

* That the initial state satisfies the invariant.  The user must verify
  separately that their starting state is within bounds.
* Convergence, stability, or attractiveness.  Only boundedness preservation.
* **The inductive step at all, if the body states an ``assume``** (audit
  0.2.0 M5).  An ``assume`` inside the body is a precondition on the WHOLE
  query, so a VERIFIED then reads "every state IN THE ASSUMED SUB-REGION
  stays within bounds after one step" — and the induction does not close,
  because the successor state need not re-enter that sub-region.  Measured:
  body ``x -> 1.5x`` on ``[-1, 1]`` under ``assume(x <= 0.5)`` and
  ``assume(x >= -0.5)`` is VERIFIED, and iterating from the admitted
  ``x = 0.4`` leaves ``[-1, 1]`` at step 3.  The verdict says so — the note
  reads ``inductive step CONDITIONAL — NOT the inductive step`` — and the
  fix is to put the restriction in ``state_bounds``, where the successor is
  checked against the same set the predecessor was drawn from.
* Anything at all, if the body's assumes CONTRADICT each other.  That is a
  harness defect and it raises
  :class:`stelling.propagate.UnsatisfiableAssumptionError` rather than
  returning a verdict (audit 0.2.0 S7): an empty assumed region makes every
  obligation vacuously true, and before this refusal existed the body
  ``x, y -> (x + y) * 10`` on ``[-1, 1]²`` under ``assume(x < y)`` and
  ``assume(y < x)`` returned VERIFIED with "the invariant is preserved by
  one step" — from ``x = y = 0.5`` one step gives ``10.0``.

This module imports jax-free (the harness import happens inside the function,
at call time): importing it costs nothing in a bare environment, but *calling*
:func:`check_inductive_step` requires the ``[jax]`` extra.
"""

from __future__ import annotations

__all__ = ["check_inductive_step"]


def check_inductive_step(
    body,
    state_bounds,
    constants=None,
    *,
    vacuity_mode="inputs-only",
    solver_timeout_ms=None,
    solver=None,
    refine=None,
    strict=False,
):
    """Verify the inductive step: one iteration of ``body`` preserves bounds.

    Parameters
    ----------
    body : callable
        A function ``(state: dict[str, array], constants: dict[str, any])
        -> dict[str, array]`` representing one loop iteration.  Must be
        JAX-traceable.  Receives one array per state variable and a dict
        of constants, returns a dict of arrays (the new state) with the
        same keys as ``state_bounds``.
    state_bounds : dict[str, tuple]
        Declared bounds for each state variable.  Keys are variable names;
        values are either:

        * ``((lo, hi), dtype)`` — scalar state (shape ``()``), backward
          compatible with v0.1.
        * ``((lo, hi), dtype, shape)`` — array state with the given shape.
          The same bounds apply elementwise to every element of the array.

        ``(lo, hi)`` is the invariant interval, ``dtype`` is a
        JAX-compatible dtype string or object, and ``shape`` is a tuple of
        non-negative integers (e.g. ``(3,)`` for a 3-vector).
    constants : dict[str, any] or None
        Extra parameters held constant across iterations, passed directly
        to ``body``.  Not declared as symbolic inputs; their concrete values
        are traced into the jaxpr.
    vacuity_mode : str
        Passed to :func:`stelling.preconditions.check`.  Default
        ``"inputs-only"`` (the standard choice for precondition checks).
    solver_timeout_ms : int or None
        Optional per-obligation solver time limit.  See
        :func:`stelling.preconditions.check`.
    solver : str or None
        Restrict SMT portfolio to a single solver (``"z3"`` or ``"cvc5"``).
    refine : str or None
        ``None`` or ``"affine"``.  See :func:`stelling.preconditions.check`.
    strict : bool
        If True, raise on transcription errors instead of declining.

    Returns
    -------
    stelling.verdict.Verdict
        The verdict.  ``VERIFIED`` means the invariant is preserved by one
        step — UNLESS the body states an ``assume``, in which case the claim
        is restricted to the assumed sub-region and is NOT the inductive
        step; the appended note says which of the two it is, and the module
        docstring's "What this does NOT prove" list carries the reason.
        ``REFUTED`` means the body can escape the bounds from some
        starting point within them.  ``UNKNOWN`` means the analysis could not
        decide — pass ``solver_timeout_ms`` to escalate undecided obligations
        to the SMT portfolio, which often resolves straddles the interval
        domain cannot.

        Outward-rounded interval arithmetic introduces 1-ULP imprecision per
        operation, so an invariant whose boundary is EXACTLY met (e.g., a body
        that preserves the bound to the last digit) may straddle.  The solver
        handles these; alternatively, widen the declared bounds by a small
        margin.

    Raises
    ------
    ValueError
        If ``state_bounds`` is empty, or if ``body`` returns keys that do not
        match ``state_bounds``.
    TypeError
        If ``state_bounds`` values have the wrong structure.
    stelling.propagate.UnsatisfiableAssumptionError
        If the body's assumes admit no state at all (a subclass of
        ``ValueError``).  A harness defect, raised rather than returned for
        the same reason :func:`stelling.preconditions.check` lets it through:
        an empty assumed region makes every obligation vacuously true, so
        nothing was verified and a status would say otherwise.
    """
    # -- Validate inputs eagerly, before tracing --------------------------------
    if not state_bounds:
        raise ValueError(
            "state_bounds must be non-empty: at least one state variable "
            "is required for inductive step verification"
        )
    if constants is None:
        constants = {}

    # Validate the shape of each state_bounds entry and normalize to
    # (bounds, dtype, shape) triples.
    _normalized_bounds = {}
    for name, spec in state_bounds.items():
        if not isinstance(name, str):
            raise TypeError(
                f"state_bounds keys must be strings, got {type(name).__name__} "
                f"for key {name!r}"
            )
        try:
            if len(spec) == 3:
                bounds, dtype, shape = spec
            elif len(spec) == 2:
                bounds, dtype = spec
                shape = ()
            else:
                raise ValueError("wrong length")
            lo, hi = bounds
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"state_bounds[{name!r}] must be ((lo, hi), dtype) or "
                f"((lo, hi), dtype, shape), got {spec!r}"
            ) from e
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            raise TypeError(
                f"state_bounds[{name!r}] bounds must be numeric, "
                f"got lo={lo!r}, hi={hi!r}"
            )
        if not isinstance(shape, tuple) or not all(
            isinstance(d, int) and d >= 0 for d in shape
        ):
            raise TypeError(
                f"state_bounds[{name!r}] shape must be a tuple of "
                f"non-negative integers, got {shape!r}"
            )
        _normalized_bounds[name] = ((lo, hi), dtype, shape)

    # -- Build the harness -------------------------------------------------------
    def harness():
        from stelling.harness import any_array, assert_

        # Declare each state variable with the invariant bounds
        state = {}
        for name, ((lo, hi), dtype, shape) in _normalized_bounds.items():
            state[name] = any_array(shape, dtype, (lo, hi))

        # Run one step of the body
        new_state = body(state, constants)

        # Validate that the body returned the expected keys
        if not isinstance(new_state, dict):
            raise ValueError(
                f"body must return a dict, got {type(new_state).__name__}. "
                f"The body function should return a dict with the same keys "
                f"as state_bounds: {set(state_bounds.keys())}"
            )
        missing = set(state_bounds.keys()) - set(new_state.keys())
        if missing:
            raise ValueError(
                f"body return dict is missing state keys: {sorted(missing)}. "
                f"Expected keys: {sorted(state_bounds.keys())}"
            )
        extra = set(new_state.keys()) - set(state_bounds.keys())
        if extra:
            raise ValueError(
                f"body return dict has unexpected keys: {sorted(extra)}. "
                f"Expected keys: {sorted(state_bounds.keys())}"
            )

        # Assert: each output must stay within the invariant bounds
        # (assert_ is elementwise on arrays, so >= and <= broadcast correctly)
        obligations = []
        for name, ((lo, hi), _dtype, _shape) in _normalized_bounds.items():
            out = new_state[name]
            obligations.append(assert_(out >= lo))
            obligations.append(assert_(out <= hi))

        return tuple(obligations)

    # -- Run the pipeline --------------------------------------------------------
    from stelling.preconditions import check

    verdict = check(
        harness,
        vacuity_mode=vacuity_mode,
        solver_timeout_ms=solver_timeout_ms,
        solver=solver,
        refine=refine,
        strict=strict,
    )

    # -- Annotate the verdict with inductive-step context ------------------------
    import dataclasses

    # WHOSE OBLIGATIONS ARE WHOSE (audit 0.2.0 M4). The harness appends
    # exactly two obligations per state variable, in `_normalized_bounds`
    # order, AFTER `body` has been traced — so a body that declares its own
    # `assert_` puts those first and shifts every index by however many it
    # declared. The predecessor read `lo_idx = 2*i` off the raw index and
    # named the wrong variable and the wrong bound whenever it did; a softer
    # misalignment produced an EMPTY `escaped` list and fell back to the
    # generic sentence, which is how it stayed invisible.
    #
    # The offset is derivable rather than assumed: the count is fixed
    # (`2 * len(_normalized_bounds)`) and the position is fixed (last), both
    # by construction of the harness twenty lines up, and jaxpr equations are
    # in trace order. A body assert inside a `jit` is bound where the `jit`
    # is called, which is still before the loop below runs. If the arithmetic
    # ever fails to hold — an obligation dropped from the report, say — the
    # clamp below leaves the offset at 0 and the generic sentence is used,
    # which is the same fallback the empty-`escaped` path already takes.
    own = 2 * len(_normalized_bounds)
    offset = max(len(verdict.obligations) - own, 0)

    # DID AN ASSUME IN THE BODY MAKE THIS CLAIM CONDITIONAL (audit 0.2.0 M5).
    # `check_inductive_step` traces `body` inside the harness, so an `assume`
    # written in the body is a precondition on the WHOLE query — and a
    # VERIFIED then means "for every state in the ASSUMED SUB-REGION, one
    # step stays in bounds", which is not the inductive step: the successor
    # state need not re-enter that sub-region, so the induction does not
    # close and the invariant does NOT follow for all iterations.
    #
    # Read off the stamped conditionality line rather than off a count,
    # because the two mechanisms that grant a precondition are different and
    # both must be caught: an interval NARROWING (`stelling.propagate`) and a
    # relational assume FORWARDED to the solver as a positive axiom
    # (`stelling.solvers`). Only the first stamped anything before this
    # build, which is precisely why M5's measured example — an assume that
    # narrows — was disclosed while the forwarded half was silent. A
    # disposition that excludes nothing (a conjunct definitely true over the
    # boxes) writes no such line and correctly triggers no caveat; a DROPPED
    # assume writes none either, and correctly so — a drop makes the judged
    # set a SUPERSET, so the VERIFIED proves more than the inductive step
    # needs, not less.
    from stelling.propagate import CONDITIONAL_ON_PRECONDITION

    conditional = bool(verdict.stamp) and any(
        CONDITIONAL_ON_PRECONDITION in a for a in verdict.stamp.assumptions
    )

    if verdict.status == "VERIFIED":
        note = (
            "inductive step: all state variables stay within declared bounds "
            "after one iteration — the invariant is preserved by one step. "
            "ASSUMPTION: this does NOT verify that the initial state is within "
            "bounds; the user must check that separately."
        )
        if conditional:
            note = (
                "inductive step CONDITIONAL — NOT the inductive step: an "
                "assume in the body is a precondition on the whole query, so "
                "what was verified is that every state IN THE ASSUMED "
                "SUB-REGION stays within declared bounds after one iteration. "
                "That does not close the induction: the successor state need "
                "not re-enter the assumed sub-region, so the invariant does "
                "NOT follow for all iterations. Measured instance: body "
                "x -> 1.5x on invariant [-1, 1] under assumes |x| <= 0.5 is "
                "VERIFIED, and iterating from the admitted x = 0.4 leaves "
                "[-1, 1] at step 3. To verify the inductive step, state the "
                "restriction in the DECLARED BOUNDS (state_bounds) instead of "
                "in an assume — then the successor is checked against the "
                "same set the predecessor was drawn from. ASSUMPTION: this "
                "does NOT verify that the initial state is within bounds; the "
                "user must check that separately."
            )
    elif verdict.status == "REFUTED":
        # Identify which state variables escaped
        escaped = []
        for i, name in enumerate(_normalized_bounds.keys()):
            lo_idx = offset + 2 * i
            hi_idx = offset + 2 * i + 1
            for ob in verdict.obligations:
                if ob.index == lo_idx and ob.status in (
                    "violated-over-set", "violated-witness"
                ):
                    escaped.append(f"{name} (below lower bound)")
                elif ob.index == hi_idx and ob.status in (
                    "violated-over-set", "violated-witness"
                ):
                    escaped.append(f"{name} (above upper bound)")
        violated_own = [
            ob.index for ob in verdict.obligations
            if ob.index >= offset
            and ob.status in ("violated-over-set", "violated-witness")
        ]
        if escaped:
            detail = ", ".join(escaped)
            note = (
                f"inductive step REFUTED: the body can escape the declared "
                f"bounds from a starting point within them. Escaped: {detail}"
            )
        elif offset and not violated_own:
            # the REFUTED is entirely the BODY's own assert(s), not a bound
            # escape — the generic sentence would have blamed the invariant
            # for an obligation the invariant is not about (audit 0.2.0 M4's
            # softer half, which fell through to it silently)
            note = (
                f"inductive step: the invariant bounds themselves were not "
                f"refuted — every violated obligation of this run is an "
                f"assert_ declared INSIDE the body (obligation index < "
                f"{offset}), not one of the {own} bound checks this function "
                f"appends. Whether the invariant is preserved is therefore "
                f"NOT settled by this REFUTED: read the per-obligation "
                f"statuses"
            )
        else:
            note = (
                "inductive step REFUTED: the body can escape the declared "
                "bounds from a starting point within them"
            )
    elif verdict.status == "UNKNOWN":
        note = (
            "inductive step UNDECIDED: the analysis could not determine "
            "whether the invariant is preserved by one step"
        )
    else:
        # DECLINED
        note = (
            "inductive step DECLINED: the harness could not be transcribed"
        )

    return dataclasses.replace(verdict, notes=verdict.notes + (note,))
