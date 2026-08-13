# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Inductive step verification for loop invariants.

Given a loop body and declared invariant bounds, verify that one step of the
body preserves the invariant: if the state starts within bounds, it stays
within bounds after one iteration.  When verified, induction gives the
invariant for all iterations.

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
        step.  ``REFUTED`` means the body can escape the bounds from some
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

    if verdict.status == "VERIFIED":
        note = (
            "inductive step: all state variables stay within declared bounds "
            "after one iteration — the invariant is preserved by one step. "
            "ASSUMPTION: this does NOT verify that the initial state is within "
            "bounds; the user must check that separately."
        )
    elif verdict.status == "REFUTED":
        # Identify which state variables escaped
        escaped = []
        for i, name in enumerate(_normalized_bounds.keys()):
            lo_idx = 2 * i
            hi_idx = 2 * i + 1
            for ob in verdict.obligations:
                if ob.index == lo_idx and ob.status in (
                    "violated-over-set", "violated-witness"
                ):
                    escaped.append(f"{name} (below lower bound)")
                elif ob.index == hi_idx and ob.status in (
                    "violated-over-set", "violated-witness"
                ):
                    escaped.append(f"{name} (above upper bound)")
        if escaped:
            detail = ", ".join(escaped)
            note = (
                f"inductive step REFUTED: the body can escape the declared "
                f"bounds from a starting point within them. Escaped: {detail}"
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
