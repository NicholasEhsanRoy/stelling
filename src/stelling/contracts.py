# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Assume-guarantee contracts for a linear-solve leg — two faces, one
mechanized.

:mod:`stelling.preconditions` stops exactly at the solve: it checks what
goes *in*, and its docs name conditioning as "a different, planned
layer". This module is that layer's first slice: a **contract** with
exactly two faces.

* **requires** — MECHANIZED. A conditioning precondition posed as
  ordinary stelling obligations and pushed through the ONE existing
  pipeline (trace -> interval propagation -> obligation extraction ->
  optional SMT escalation), via the same machinery as
  :func:`stelling.preconditions.check` (a shared private helper; no
  second pipeline exists). The requires face yields a real
  :class:`stelling.verdict.Verdict` with the full standard stamp;
  ``vacuity_mode`` is required exactly as in ``check()``, and a VERIFIED
  gets the widen re-check at the same pipeline depth. Solvers run only
  under an explicit ``solver_timeout_ms`` — never on defaults.

* **ensures** — DECLARED, never checked. A statement that is a *theorem
  given the requires*, carried on the contract verdict with the status
  token :data:`ENSURES_DECLARED` (``"DECLARED"`` — a distinct token that
  can never be confused with VERIFIED/REFUTED/UNKNOWN), explicitly
  conditional on the requires face (identified by its content), and
  stamped with its derivation. **No constructible path through this
  layer's types can produce or carry an ensures status other than
  DECLARED**: :class:`EnsuresFace` refuses any other status at
  construction (including via ``dataclasses.replace``), the containers
  :class:`Contract` and :class:`ContractVerdict` accept only the exact
  sealed :class:`EnsuresFace` type — subclasses and duck-typed
  stand-ins are refused deliberately, because a subclass can override
  anything — and :func:`check_contract` carries the caller's face
  through unchanged. Interpreter-level bypasses
  (``object.__setattr__`` / ``object.__new__``) are outside the threat
  model, exactly as for every frozen dataclass in this codebase. A
  contract with no exact derivation available declares no ensures and
  says so (:func:`coefficient_contrast`).

**Emptiness asymmetry.** A requires-VERIFIED is a universally
quantified statement over the declared envelope: *every* point of the
envelope satisfies the requires. It does NOT certify that the envelope
is nonempty, and nothing here phrases it existentially. The ensures
face reads "for every point of the envelope satisfying the requires
..." — vacuously true on an empty envelope, and its text says so. A
caller who needs nonemptiness tied to real data should look at
:mod:`stelling.exactness` / the nonvacuity machinery, not at this
layer; the standard stamp's nonvacuity field already records
``UNCHECKED`` here, and the standard may-be-vacuous note rides on any
VERIFIED.

**Closed half-spaces only.** Both templates pose only non-strict
inequalities (stelling-wide discipline). The properties they encode
("positive definite", "strictly positive") are OPEN conditions with no
closed-inequality characterization, so each template poses the
topological **closure** of its intended set and documents the boundary
exactly:

* :func:`conditioning_2x2` poses PSD (all principal minors >= 0) plus
  the ratio conjunct; the posed set is
  ``{M SPD with cond_2(M) <= kappa} ∪ {M = 0}`` — the closure, whose
  single non-SPD member is the zero matrix (proof in the template's
  docstring). The ensures excludes that point by name.
* :func:`coefficient_contrast` poses ``field >= 0`` plus the
  division-free contrast conjunct; the posed set is
  ``{field strictly positive with max/min <= C} ∪ {the zero field}`` —
  again the closure, boundary again the single zero point.

This module imports jax-free (harness/pipeline imports happen inside
functions, exactly like :mod:`stelling.preconditions`); calling a
template's checker needs the ``[jax]`` extra, and escalation needs a
solver installed plus an explicit timeout.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from typing import Callable

# binary64's largest finite magnitude, for the one authoring refusal that
# names the storable range — the same number any_array's refusal quotes
_F_MAX = sys.float_info.max

__all__ = [
    "Contract",
    "ContractVerdict",
    "ENSURES_DECLARED",
    "EnsuresFace",
    "check_contract",
    "coefficient_contrast",
    "conditioning_2x2",
    "conditioning_2x2_field",
]

# The ONE status an ensures face can ever have. Deliberately distinct
# from the verdict statuses VERIFIED/REFUTED/UNKNOWN so no reader or
# tool can mistake a declared theorem for a mechanized result.
ENSURES_DECLARED = "DECLARED"


@dataclass(frozen=True)
class EnsuresFace:
    """The declared (never checked) guarantee face of a contract.

    ``status`` is structurally pinned to :data:`ENSURES_DECLARED`:
    construction with any other value — including via
    ``dataclasses.replace`` — raises. There is no upgrade path; a
    checked guarantee would be a different layer's object, not a
    mutated one of these. This is a **sealed** type: the containers
    (:class:`Contract`, :class:`ContractVerdict`) accept exactly this
    class and refuse subclasses, so a subclass that drops this
    refusal cannot flow anywhere. Text fields are single physical
    lines — an embedded newline could forge verdict-looking lines in
    the render and the stamp, so it is refused here (the same
    screening posture the solver layer applies to model echoes).
    """

    statement: str  # the theorem, conditional wording included
    derivation: str  # how it follows from the requires (the stamped line)
    conditional_on: str  # the requires face, identified by its content
    status: str = ENSURES_DECLARED

    def __post_init__(self) -> None:
        if self.status != ENSURES_DECLARED:
            raise ValueError(
                f"an ensures face is {ENSURES_DECLARED} and nothing else — "
                f"got status {self.status!r}. The ensures is never checked "
                f"by interval propagation or a solver, so no code path may "
                f"mint a VERIFIED/REFUTED/UNKNOWN (or any other) ensures."
            )
        for field_name in ("statement", "derivation", "conditional_on"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"EnsuresFace.{field_name} must be populated (a "
                    f"whitespace-only or non-string value is not a "
                    f"statement) — an unstated ensures face is not a face"
                )
            if "\n" in value or "\r" in value:
                raise ValueError(
                    f"EnsuresFace.{field_name} must be a single physical "
                    f"line — render and stamp integrity: an embedded "
                    f"newline could forge verdict-looking lines (e.g. a "
                    f"fake '== VERIFIED' or solver line) at column 0 of "
                    f"the render"
                )


def _require_sealed_ensures(owner: str, ensures) -> None:
    """The exact-type funnel both containers run on their ensures field.

    ``type(...) is EnsuresFace`` — deliberately not ``isinstance``: a
    subclass can override ``__post_init__`` and carry any status, and a
    duck-typed stand-in can carry anything at all, so the DECLARED-only
    invariant wants a sealed type at every container door. The status
    re-check is belt-and-braces (one invariant, two mechanisms): it is
    unreachable through a constructible EnsuresFace, and checked anyway.
    """
    if ensures is None:
        return
    if type(ensures) is not EnsuresFace:
        raise ValueError(
            f"{owner}.ensures must be exactly stelling.contracts."
            f"EnsuresFace (a sealed type) or None; got "
            f"{type(ensures).__name__}. Subclasses and duck-typed "
            f"stand-ins are refused deliberately — a subclass can "
            f"override anything, and the DECLARED-only invariant wants a "
            f"sealed type."
        )
    if ensures.status != ENSURES_DECLARED:
        raise ValueError(
            f"{owner}.ensures carries status {ensures.status!r}; an "
            f"ensures face is {ENSURES_DECLARED} and nothing else"
        )


@dataclass(frozen=True)
class Contract:
    """A two-faced contract: a requires harness plus at most one
    declared ensures. Exactly one of ``ensures`` /
    ``no_ensures_reason`` is populated — a contract either declares its
    guarantee (with derivation) or says out loud why it declares none.
    ``ensures`` is accepted only as the exact sealed
    :class:`EnsuresFace` type (or None).
    """

    name: str
    requires_description: str  # the requires content, in words
    harness: Callable[[], tuple]  # zero-arg; poses the requires obligations
    ensures: EnsuresFace | None
    no_ensures_reason: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.requires_description:
            raise ValueError(
                "Contract.name and Contract.requires_description must be "
                "populated"
            )
        if not callable(self.harness):
            raise ValueError("Contract.harness must be a zero-arg callable")
        if (self.ensures is None) == (not self.no_ensures_reason):
            raise ValueError(
                "exactly one of Contract.ensures / Contract.no_ensures_reason "
                "must be populated: declare the guarantee with its "
                "derivation, or say why none is declared"
            )
        _require_sealed_ensures("Contract", self.ensures)


@dataclass(frozen=True)
class ContractVerdict:
    """Both faces of a checked contract.

    ``requires`` is a real :class:`stelling.verdict.Verdict` (standard
    stamp, standard obligations, standard witnesses); ``ensures`` is the
    contract's declared face, carried through UNCHANGED by
    :func:`check_contract`, and accepted only as the exact sealed
    :class:`EnsuresFace` type (or None, with the reason populated) —
    this container is directly constructible, so it runs the same
    funnel :class:`Contract` runs. There is deliberately no combined
    top-level ``status``: the mechanized claim is ``requires_status``
    and the declared face is ``ensures_status`` — collapsing them into
    one token is exactly the confusion the DECLARED token exists to
    prevent.
    """

    contract_name: str
    requires_description: str
    requires: object  # stelling.verdict.Verdict (kept untyped: jax-free import)
    ensures: EnsuresFace | None
    no_ensures_reason: str = ""

    def __post_init__(self) -> None:
        if (self.ensures is None) == (not self.no_ensures_reason):
            raise ValueError(
                "exactly one of ContractVerdict.ensures / "
                "ContractVerdict.no_ensures_reason must be populated: a "
                "checked contract either carries its declared guarantee or "
                "the reason none is declared"
            )
        _require_sealed_ensures("ContractVerdict", self.ensures)

    @property
    def requires_status(self) -> str:
        """VERIFIED / REFUTED / UNKNOWN — the mechanized face."""
        return self.requires.status

    @property
    def ensures_status(self) -> str:
        """:data:`ENSURES_DECLARED` when an ensures is declared, else the
        literal ``"none declared"`` — never a verdict status."""
        return (
            self.ensures.status if self.ensures is not None else "none declared"
        )

    def render(self) -> str:
        lines = [
            f"== contract {self.contract_name!r} — assume-guarantee, two faces",
            "requires face (MECHANIZED — the standard stelling verdict below, "
            "full stamp included):",
            f"  posed as: {self.requires_description}",
            self.requires.render(),
        ]
        if self.ensures is not None:
            e = self.ensures
            lines += [
                f"ensures face ({e.status}):",
                f"  status: {e.status} — this face is declared, NOT checked: "
                f"no interval propagation and no solver examined it. It is a "
                f"theorem conditional on the requires face and holds only "
                f"where the requires holds.",
                f"  statement: {e.statement}",
                f"  conditional on the requires face: {e.conditional_on}",
                f"  derivation: {e.derivation}",
                "  reading: for every point of the declared envelope "
                "satisfying the requires, the statement holds; at points "
                "violating the requires nothing is claimed, and on an "
                "envelope no point of which satisfies the requires the "
                "statement is vacuously true. Neither face certifies that "
                "the envelope, or its requires-satisfying subset, is "
                "nonempty — nonemptiness is stelling.exactness's claim, not "
                "this layer's.",
            ]
        else:
            lines.append(f"ensures: {self.no_ensures_reason}")
        return "\n".join(lines)


# -- templates ----------------------------------------------------------------


def _closed_range(template: str, name: str, rng):
    """Authoring-time validation of a declared closed range — the
    loudness :func:`stelling.harness.any_array` applies at trace time,
    moved to authoring, where these templates' own refusal messages
    already claim refusals happen. An authored contract should never be
    a delayed explosion: a reversed or NaN range declares an empty
    envelope (NaN fails ``lo <= hi``), and a non-finite endpoint is
    refused outright — these templates pose bounded closed envelopes.
    """
    # VALIDATE on the exact classified value, RETURN the caller's own values.
    # Returning floats was the fourth route into a soundness hole found three
    # times already: `any_array`'s storability guard judges the operand it is
    # handed, so a pre-converted bound arrives rounded and the guard never
    # sees the value it exists to judge. Measured: `_closed_range("t", "n",
    # (0, 2**53 + 1))` returned `9007199254740992.0`, which both mis-declared
    # the envelope AND printed the wrong number into the contract's own
    # requires-description. Validating THROUGH float() had the same class of
    # hole one notch smaller — `float('0.25')` validated a str the harness
    # then re-rounded, and `float(10**400)` escaped as a bare OverflowError —
    # so validation now runs on the same exact classification any_array
    # decides with (stelling._bound_spelling, jax-free and numpy-lazy, so
    # authoring stays eager and bare-environment). Spellings outside the
    # classifier's closed family are refused HERE, at authoring, with the
    # family named; the decision for everything accepted is unchanged and is
    # still made by any_array at trace time, on the caller's own values.
    from stelling._bound_spelling import (
        ACCEPTED_SPELLINGS,
        binary64_image,
        declared_bound_value,
    )

    lo_x = declared_bound_value(rng[0])
    hi_x = declared_bound_value(rng[1])
    for v, x in ((rng[0], lo_x), (rng[1], hi_x)):
        if x is None:
            raise ValueError(
                f"{template}: {name} endpoint {v!r} (type "
                f"{type(v).__qualname__}) is not an accepted bound "
                f"spelling: a bound must be {ACCEPTED_SPELLINGS}. It is "
                f"refused rather than converted, because a conversion whose "
                f"exactness this layer cannot judge is how a declared bound "
                f"gets silently rounded — refusing at authoring time"
            )
    lo, hi = binary64_image(lo_x), binary64_image(hi_x)
    if not lo <= hi:
        raise ValueError(
            f"{template}: {name} = ({rng[0]!r}, {rng[1]!r}) declares an "
            f"empty envelope (lo > hi, or a NaN endpoint); refusing at "
            f"authoring time"
        )
    if not (math.isfinite(lo) and math.isfinite(hi)):
        if isinstance(lo_x, float) or isinstance(hi_x, float):
            # a genuinely infinite declared endpoint (the classifier
            # returns a float only for ±inf/nan, and nan was refused above)
            raise ValueError(
                f"{template}: {name} = ({rng[0]!r}, {rng[1]!r}) has a "
                f"non-finite endpoint; this template poses bounded closed "
                f"envelopes — refusing at authoring time"
            )
        # both endpoints are FINITE declared values, so at least one lies
        # outside binary64's finite range: "non-finite endpoint" would be a
        # false diagnosis (the phase1 lens caught exactly that message on
        # the longdouble spelling), and float() on the int spelling used to
        # escape as a bare OverflowError before any message at all
        raise ValueError(
            f"{template}: {name} = ({rng[0]!r}, {rng[1]!r}) has an endpoint "
            f"outside binary64's finite range "
            f"[{(-_F_MAX)!r}, {_F_MAX!r}]; the IR records bounds as "
            f"binary64, so the endpoint cannot be recorded at all — "
            f"refusing at authoring time"
        )
    return rng[0], rng[1]


def conditioning_2x2(dtype, a_range, c_range, b_range, kappa) -> Contract:
    """Contract for the solve leg on a symmetric 2x2 ``M = [[a, b], [b, c]]``
    with entries over declared closed ranges.

    **requires (mechanized):** the conjunction, posed as four closed
    obligations through jax ops (products included, so the standard
    fragment router classifies the nonlinear ones QF_NRA):

    1. ``a >= 0``
    2. ``c >= 0``
    3. ``a*c - b*b >= 0``
    4. ``(a + c)^2 <= (a*c - b*b) * (kappa + 1/kappa + 2)``

    **The closed-form posing choice and its boundary.** "M is SPD with
    ``cond_2(M) <= kappa``" is the intended requires, but SPD is an open
    condition — it has NO characterization by non-strict polynomial
    inequalities, and stelling poses closed half-spaces only. Obligations
    1-3 are the exact closed characterization of positive
    SEMI-definiteness for a symmetric 2x2 (all principal minors
    nonnegative: both diagonal entries and the determinant). Obligation 4
    is the probe's exact reduction (corpus/supply/la_contract_probe.py
    Part 2): for SPD M, ``tr^2/det = r + 1/r + 2`` with ``r = cond_2(M)``,
    and ``r + 1/r + 2`` is strictly increasing for ``r >= 1``, so
    ``cond_2(M) <= kappa  <=>  tr^2 <= det * (kappa + 1/kappa + 2)``.

    The posed conjunction equals ``{M SPD with cond_2(M) <= kappa} ∪
    {M = 0}`` — the topological closure of the intended set, and the
    boundary is exactly the single point ``M = 0``: if ``det > 0`` then
    (with 1-3) ``a, c > 0``, ``tr > 0``, both eigenvalues are positive
    and the reduction gives ``cond_2 <= kappa``; if ``det = 0`` then 4
    forces ``tr = 0``, 1-2 force ``a = c = 0``, and ``det = -b^2 = 0``
    forces ``b = 0`` — the zero matrix, the closure's one non-SPD member.
    The ensures excludes it by name (``M^-1`` does not exist there).

    ``kappa`` must be finite and ``>= 1``: ``cond_2`` is always ``>= 1``,
    and because ``f(kappa) = f(1/kappa)`` for ``f(r) = r + 1/r + 2``, a
    ``kappa < 1`` would silently pose ``cond_2 <= 1/kappa`` instead of an
    infeasible bound — refused at authoring time rather than mis-posed.
    The coefficient ``kappa + 1/kappa + 2`` is computed in binary64 and
    enters the traced query as a transcribed constant (for ``kappa = 8``
    it is exactly ``10.125``); the stamped semantics is the standard ℝ
    claim over the declared envelope with this constant.

    **ensures (DECLARED, never checked):** norm-sensitivity of the solve
    output, kappa-derived from the same conditioning data (probe Part 4):
    for every envelope point where the requires holds and ``M != 0``,
    and every right-hand side ``r``:
    ``||M^-1 r||_2 <= (kappa / ||M||_2) * ||r||_2``.
    """
    if isinstance(kappa, bool) or not isinstance(kappa, (int, float)):
        raise ValueError(f"kappa must be a real number, got {kappa!r}")
    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa < 1.0:
        raise ValueError(
            f"kappa must be finite and >= 1 (cond_2 is always >= 1, and the "
            f"reduction tr^2 <= det*(kappa + 1/kappa + 2) encodes "
            f"cond_2 <= kappa only for kappa >= 1 — f(kappa) = f(1/kappa), "
            f"so kappa < 1 would silently pose cond_2 <= 1/kappa); got "
            f"{kappa!r}"
        )
    coeff = kappa + 1.0 / kappa + 2.0  # binary64; transcribed into the query
    a_lo, a_hi = _closed_range("conditioning_2x2", "a_range", a_range)
    c_lo, c_hi = _closed_range("conditioning_2x2", "c_range", c_range)
    b_lo, b_hi = _closed_range("conditioning_2x2", "b_range", b_range)

    requires_description = (
        f"M = [[a, b], [b, c]] with a in [{a_lo!r}, {a_hi!r}], "
        f"c in [{c_lo!r}, {c_hi!r}], b in [{b_lo!r}, {b_hi!r}] ({dtype}): "
        f"a >= 0 AND c >= 0 AND a*c - b*b >= 0 (closed principal-minor / "
        f"PSD form) AND (a + c)^2 <= (a*c - b*b) * {coeff!r} (the exact "
        f"reduction of cond_2(M) <= {kappa!r}; coefficient = kappa + "
        f"1/kappa + 2), at every point of the declared envelope — the "
        f"closed-form posing of 'M is symmetric positive-definite with "
        f"cond_2(M) <= {kappa!r}', exact up to the single closure point "
        f"M = 0"
    )

    def harness():
        from stelling.harness import any_array, assert_

        a = any_array((), dtype, (a_lo, a_hi))
        c = any_array((), dtype, (c_lo, c_hi))
        b = any_array((), dtype, (b_lo, b_hi))
        tr = a + c
        det = a * c - b * b
        return (
            assert_(a >= 0.0),
            assert_(c >= 0.0),
            assert_(det >= 0.0),
            assert_(tr * tr <= det * coeff),
        )

    ensures = EnsuresFace(
        statement=(
            f"for every point (a, b, c) of the declared envelope at which "
            f"the requires holds and M is not the zero matrix — "
            f"equivalently, at every envelope point where M is symmetric "
            f"positive-definite with cond_2(M) <= {kappa!r} — and for "
            f"every right-hand side r: ||M^-1 r||_2 <= "
            f"({kappa!r} / ||M||_2) * ||r||_2 (norm-sensitivity of the "
            f"solve output). Universally quantified over the declared "
            f"envelope: nothing is claimed at points violating the "
            f"requires, the statement is vacuously true if no envelope "
            f"point satisfies the requires, and it does not assert that "
            f"any such point exists."
        ),
        derivation=(
            f"||M^-1 r||_2 <= ||M^-1||_2 * ||r||_2 (operator-norm "
            f"inequality), and ||M^-1||_2 = cond_2(M) / ||M||_2 <= "
            f"{kappa!r} / ||M||_2 wherever the requires holds with M != 0 "
            f"(cond_2 = ||M||_2 * ||M^-1||_2 by definition; the requires "
            f"face bounds it by kappa — probe: "
            f"corpus/supply/la_contract_probe.py Part 4). The single "
            f"non-SPD point the closed requires admits, M = 0, is excluded "
            f"by name in the statement: M^-1 does not exist there."
        ),
        conditional_on=requires_description,
    )
    return Contract(
        name="conditioning_2x2",
        requires_description=requires_description,
        harness=harness,
        ensures=ensures,
    )


def conditioning_2x2_field(shape, dtype, theta_range, kappa, transform) -> Contract:
    """Contract for the solve leg on a caller-built FAMILY of symmetric 2x2
    matrices — :func:`conditioning_2x2` lifted over a transform, mirroring
    :func:`coefficient_contrast`'s transform convention.

    ``theta`` is declared over ``shape``/``theta_range`` (one closed range,
    every element), and ``transform(theta)`` — **your code's own
    construction**, e.g. a per-cell normal-matrix assembly over real mesh
    connectivity — returns either

    * an array ``[..., 2, 2]`` of matrices (any leading batch shape, e.g.
      per-cell ``[N, 2, 2]``), or
    * a triple ``(a, b, c)`` of same-shaped arrays/scalars, read as the
      family ``[[a, b], [b, c]]`` — validated at trace time: every
      non-scalar element must have one identical shape (a mixed pair
      like ``(2, 1)`` with ``(3,)`` would silently outer-broadcast to a
      family the caller never posed, and is refused naming both shapes);
      scalars broadcast against the non-scalar shape.

    ``transform=None`` is the identity: ``theta`` itself is the family and
    must be declared with trailing shape ``(2, 2)`` (refused at authoring
    otherwise). A tuple/list return of any length other than 3, a
    non-array/non-triple return, an array whose trailing shape is not
    ``(2, 2)``, and a family of ZERO matrices are all refused loudly at
    trace time — never a silent zero-obligation UNKNOWN.

    **requires (mechanized), per matrix of the family** — the same four
    closed conjuncts as :func:`conditioning_2x2`, posed elementwise over
    the batch (a VERIFIED means every matrix at every envelope point):

    1. ``a >= 0``
    2. ``c >= 0``
    3. ``a*c - b*b >= 0``
    4. ``(a + c)^2 <= (a*c - b*b) * (kappa + 1/kappa + 2)``

    with the identical closed-form/closure argument, per matrix: the posed
    set per family member is ``{M SPD with cond_2(M) <= kappa} ∪ {M = 0}``
    — the closure, whose single non-SPD member is the zero matrix (proof
    in :func:`conditioning_2x2`'s docstring; the ensures excludes that
    point by name). ``kappa`` carries the same validation (finite,
    ``>= 1``), and the coefficient ``kappa + 1/kappa + 2`` is computed in
    binary64 and transcribed into the query.

    **PLUS, on the full-array path only: symmetry is POSED, never
    silently assumed.** The four conjuncts read ``b`` off ONE off-diagonal
    (``b = M[..., 0, 1]``), and reading one off-diagonal of a
    possibly-asymmetric matrix would smuggle an unchecked precondition —
    the conjuncts would then certify the conditioning of a symmetrized
    stand-in, not of the matrix the transform actually built. So a
    ``[..., 2, 2]`` return poses two additional closed obligations,

    5. ``M[..., 0, 1] <= M[..., 1, 0]``
    6. ``M[..., 1, 0] <= M[..., 0, 1]``,

    making the requires conjunction exactly "every matrix of the family is
    symmetric AND lies in the closed conditioning set". If the caller
    passes an ``(a, b, c)`` triple instead, passing one off-diagonal IS
    their symmetry assertion — the caller vouches, in the API's own shape,
    that both off-diagonals are the same value, and no symmetry obligation
    is posed (exactly as :func:`conditioning_2x2` itself).

    Expect the symmetry pair to be interval-UNDECIDED over non-point
    envelopes even when the transform is symmetric by construction: the
    two off-diagonals are distinct element terms with equal propagated
    boxes, and equal non-point boxes straddle a ``<=`` in both
    directions. Measured, this holds even at ALL-POINT envelopes when
    the off-diagonal is a computed product — outward rounding widens
    ``0.0 * theta`` to ``[-5e-324, 5e-324]``, off the point. That
    UNKNOWN is honest, and solver escalation discharges it trivially
    when the entries share structure (e.g. ``mul(d0, d1)`` vs
    ``mul(d1, d0)`` from a transcribed outer product). On an asymmetric
    family the same pair REFUTES with a concrete witness — which is the
    point.

    Family assembly must stay on supported primitives: build stacked
    entries with ``jnp.concatenate`` over ``reshape``\\ d pieces —
    measured on jax 0.11.0, ``jnp.stack`` traces to a ``stack`` primitive
    with no transfer row, which would send the whole family to ⊤.

    Vacuity, the emission budget, and never-on solver defaults are
    inherited unchanged through :func:`check_contract`.

    **Scale, measured.** Solver escalation is gated by the ONE
    per-obligation emission budget
    (:data:`stelling.obligation.ELEMENT_BUDGET` = 512 element terms — a
    committed, measured-solver-cost decision; it does not move for this
    template). With a per-cell assembly costing ~11 element terms per
    matrix (the worked shape: a per-cell normal-matrix assembly of a
    solver's least-squares stencil), the conditioning conjunct
    escalates up to N = 46 matrices and declines loudly at N = 47 —
    "517 element terms ... over the per-obligation emission budget of
    512", both numbers quoted in the decline. Interval-decidable
    obligations keep working at any N. So on a real mesh, pose
    deliberate sub-families — a region, a sampled subset, a known-worst
    cell class — rather than the whole mesh; a whole-mesh family past a
    few dozen cells gets no solver help, honestly declined, never
    silently approximated.

    **ensures (DECLARED, never checked), per matrix:** the same
    kappa-derived norm-sensitivity face as :func:`conditioning_2x2`,
    worded over the family — for every point of the envelope and every
    matrix of the family at which the requires holds and that matrix is
    nonzero, and every right-hand side ``r``:
    ``||M^-1 r||_2 <= (kappa / ||M||_2) * ||r||_2``.
    """
    if isinstance(kappa, bool) or not isinstance(kappa, (int, float)):
        raise ValueError(f"kappa must be a real number, got {kappa!r}")
    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa < 1.0:
        raise ValueError(
            f"kappa must be finite and >= 1 (cond_2 is always >= 1, and the "
            f"reduction tr^2 <= det*(kappa + 1/kappa + 2) encodes "
            f"cond_2 <= kappa only for kappa >= 1 — f(kappa) = f(1/kappa), "
            f"so kappa < 1 would silently pose cond_2 <= 1/kappa); got "
            f"{kappa!r}"
        )
    coeff = kappa + 1.0 / kappa + 2.0  # binary64; transcribed into the query
    if transform is not None and not callable(transform):
        raise ValueError(
            f"conditioning_2x2_field: transform must be a callable or None "
            f"(None = identity: theta itself is the family), got "
            f"{transform!r}"
        )
    dims = []
    for d in shape:
        # per-dim validation at authoring, the same funnel
        # coefficient_contrast runs (ir.py posture): exact Python ints
        # only, no negative extents. Zero extents are NOT refused here —
        # there is no extremum fold in this template, and the family's
        # own emptiness is checked at trace time where the transform's
        # batch shape is known.
        if isinstance(d, bool) or not isinstance(d, int):
            raise ValueError(
                f"conditioning_2x2_field: shape {tuple(shape)!r} has a "
                f"non-int extent {d!r} ({type(d).__name__}); extents must "
                f"be Python ints (no coercion); refusing at authoring time"
            )
        if d < 0:
            raise ValueError(
                f"conditioning_2x2_field: shape {tuple(shape)!r} has a "
                f"negative extent {d!r}: no jax program constructs such an "
                f"array; refusing at authoring time"
            )
        dims.append(d)
    dims = tuple(dims)
    if transform is None and (len(dims) < 2 or dims[-2:] != (2, 2)):
        raise ValueError(
            f"conditioning_2x2_field: transform=None declares theta itself "
            f"as the matrix family, so shape must end in (2, 2); got "
            f"{dims}; refusing at authoring time"
        )
    lo, hi = _closed_range("conditioning_2x2_field", "theta_range", theta_range)

    requires_description = (
        f"family M = transform(theta) for theta over shape {dims} ({dtype}) "
        f"with every element in [{lo!r}, {hi!r}]"
        f"{' (identity transform: theta is the family)' if transform is None else ''}: "
        f"per matrix of the family, a >= 0 AND c >= 0 AND a*c - b*b >= 0 "
        f"(closed principal-minor / PSD form) AND (a + c)^2 <= "
        f"(a*c - b*b) * {coeff!r} (the exact reduction of cond_2(M) <= "
        f"{kappa!r}; coefficient = kappa + 1/kappa + 2), at every point of "
        f"the declared envelope — where (a, b, c) is the caller's triple if "
        f"the transform returns one (passing one off-diagonal IS the "
        f"caller's symmetry assertion), and a = M[..,0,0], b = M[..,0,1], "
        f"c = M[..,1,1] with symmetry POSED as two additional closed "
        f"obligations M[..,0,1] <= M[..,1,0] AND M[..,1,0] <= M[..,0,1] if "
        f"it returns full [..., 2, 2] matrices — the closed-form posing of "
        f"'every matrix of the family is symmetric positive-definite with "
        f"cond_2(M) <= {kappa!r}', exact up to the single per-matrix "
        f"closure point M = 0"
    )

    def harness():
        from stelling.harness import any_array, assert_

        theta = any_array(dims, dtype, (lo, hi))
        value = transform(theta) if transform is not None else theta
        if isinstance(value, (tuple, list)):
            if len(value) != 3:
                raise ValueError(
                    f"conditioning_2x2_field: the transform returned a "
                    f"{type(value).__name__} of length {len(value)}; the "
                    f"convention is an array [..., 2, 2] of matrices or a "
                    f"triple (a, b, c) of same-shaped arrays/scalars"
                )
            a, b, c = value
            # same-shaped arrays or scalars only (audit round 2, F6): a
            # mixed pair of non-scalar shapes ((2, 1) with (3,)) would
            # silently outer-broadcast to a family the caller never
            # posed. Scalars broadcast against the one non-scalar shape;
            # that is the documented convention.
            non_scalar = {
                name_: tuple(getattr(v, "shape", ()))
                for name_, v in (("a", a), ("b", b), ("c", c))
                if tuple(getattr(v, "shape", ())) != ()
            }
            if len(set(non_scalar.values())) > 1:
                raise ValueError(
                    f"conditioning_2x2_field: the (a, b, c) triple mixes "
                    f"non-scalar shapes {non_scalar}; same-shaped arrays "
                    f"or scalars only — mixed non-scalar shapes would "
                    f"silently broadcast to a family the caller never posed"
                )
            b10 = None  # the caller's off-diagonal IS their symmetry assertion
        else:
            m_shape = getattr(value, "shape", None)
            if m_shape is None:
                raise ValueError(
                    f"conditioning_2x2_field: the transform returned "
                    f"{type(value).__name__}, which is neither an array "
                    f"[..., 2, 2] nor an (a, b, c) triple"
                )
            m_shape = tuple(m_shape)
            if len(m_shape) < 2 or m_shape[-2:] != (2, 2):
                raise ValueError(
                    f"conditioning_2x2_field: the transform returned an "
                    f"array of shape {m_shape}; a matrix family must have "
                    f"trailing shape (2, 2)"
                )
            batch = 1
            for dim in m_shape[:-2]:
                batch *= dim
            if batch == 0:
                raise ValueError(
                    f"conditioning_2x2_field: the transform returned a "
                    f"family of zero matrices (shape {m_shape}) — nothing "
                    f"to pose"
                )
            a = value[..., 0, 0]
            b = value[..., 0, 1]
            b10 = value[..., 1, 0]
            c = value[..., 1, 1]
        tr = a + c
        det = a * c - b * b
        obligations = [
            assert_(a >= 0.0),
            assert_(c >= 0.0),
            assert_(det >= 0.0),
            assert_(tr * tr <= det * coeff),
        ]
        if b10 is not None:
            # symmetry POSED (array path): reading b off one off-diagonal
            # above is licensed only by these two closed obligations, never
            # silently assumed. The asserts are CALLED after the four
            # conjuncts' asserts so the obligation indices are stable
            # across both return paths (0-3 the conjuncts; 4-5 the
            # symmetry pair) — an obligation's index is its assert's
            # position in the traced query, which is call order.
            obligations.append(assert_(b <= b10))
            obligations.append(assert_(b10 <= b))
        return tuple(obligations)

    ensures = EnsuresFace(
        statement=(
            f"for every point of the declared envelope and every matrix of "
            f"the family at which the requires holds and that matrix is not "
            f"the zero matrix — equivalently, at every envelope point and "
            f"family position where M is symmetric positive-definite with "
            f"cond_2(M) <= {kappa!r} — and for every right-hand side r: "
            f"||M^-1 r||_2 <= ({kappa!r} / ||M||_2) * ||r||_2 "
            f"(norm-sensitivity of the solve output, per matrix). "
            f"Universally quantified over the declared envelope and the "
            f"family: nothing is claimed at points or matrices violating "
            f"the requires, the statement is vacuously true if no envelope "
            f"point satisfies the requires, and it does not assert that "
            f"any such point exists."
        ),
        derivation=(
            f"per matrix of the family: ||M^-1 r||_2 <= ||M^-1||_2 * "
            f"||r||_2 (operator-norm inequality), and ||M^-1||_2 = "
            f"cond_2(M) / ||M||_2 <= {kappa!r} / ||M||_2 wherever the "
            f"requires holds with M != 0 (cond_2 = ||M||_2 * ||M^-1||_2 by "
            f"definition; the requires face bounds it by kappa per matrix "
            f"— probe: corpus/supply/la_contract_probe.py Part 4). The "
            f"single per-matrix closure point M = 0 is excluded by name in "
            f"the statement: M^-1 does not exist there."
        ),
        conditional_on=requires_description,
    )
    return Contract(
        name="conditioning_2x2_field",
        requires_description=requires_description,
        harness=harness,
        ensures=ensures,
    )


def coefficient_contrast(
    shape, dtype, chi_range, contrast_bound, transform=None
) -> Contract:
    """Contract on a coefficient field built by the caller's ``transform``
    from a declared parameter envelope (the transform convention mirrors
    :func:`stelling.preconditions.field_positive`: your code's own
    construction, ``None`` for identity, a tuple/list return posing the
    obligation pair per produced value). The motivating construction is
    a positive coefficient field built by your construction from a
    declared parameter envelope, e.g. ``mu = 1 + chi``.

    **requires (mechanized), per produced value:**

    1. ``field >= 0`` at every point (closed positivity), and
    2. ``max(field) <= C * min(field)`` — the division-free
       multiplicative contrast form. No division is emitted, by
       construction (a division-based ``max/min <= C`` would also hit
       the escalation div-guard whenever the minimum may be zero).

    **The closed posing and its boundary** (same shape as
    :func:`conditioning_2x2`): "strictly positive with contrast
    ``max/min <= C``" is open; the posed closed conjunction equals
    ``{field strictly positive with max <= C*min} ∪ {the identically-zero
    field}`` — its closure. If ``min = 0``, conjunct 2 forces
    ``max <= 0`` and conjunct 1 then forces every element to 0; the zero
    field is the single admitted non-positive point.

    **The extremum encoding.** stelling's propagation/emission supports
    neither ``reduce_min`` nor ``reduce_max`` (checked), and this build
    deliberately does not add core transfer/emission rows (a censused-row
    addition is its own audited build). Instead ``max``/``min`` are
    posed as a pairwise fold over already-supported primitives: the
    field is reshaped flat (``reshape``, structural), split into
    single-element ``slice``\\ s (structural — the emitted terms ALIAS
    the field's element terms, sharing preserved), and folded through
    binary ``max``/``min`` (audited exact transfers; emitted as the
    exact ``ite`` selection). In ℝ the fold IS the extremum — exact in
    both directions, not an approximation — and the interval transfer
    chain computes exactly ``[min-of-los, min-of-his]`` /
    ``[max-of-los, max-of-his]``, the exact image over the element
    boxes. Everything is judged and emitted by the ONE existing budget:
    with the canonical ``mu = 1 + chi`` transform the contrast
    obligation costs ``4n`` element terms, so the per-obligation gate
    (:data:`stelling.obligation.ELEMENT_BUDGET` = 512) admits ``n <=
    128`` and anything above declines loudly with both numbers quoted —
    never silently approximated.

    ``contrast_bound`` must be finite and ``>= 1`` (a positive field's
    contrast is always ``>= 1``; a smaller bound is satisfiable only by
    the zero-field closure point and is refused as an authoring error).
    Zero-size fields are refused: an extremum of no elements does not
    exist (measured jax refuses size-0 reductions).

    **ensures: none declared.** ``contrast <= C`` does not by itself
    bound the condition number of a discretized operator — that step
    needs mesh/discretization constants this contract does not have, and
    manufacturing one would state a theorem the requires does not imply.
    The contract verdict says "ensures: none declared".
    """
    if isinstance(contrast_bound, bool) or not isinstance(
        contrast_bound, (int, float)
    ):
        raise ValueError(
            f"contrast_bound must be a real number, got {contrast_bound!r}"
        )
    bound = float(contrast_bound)
    if not math.isfinite(bound) or bound < 1.0:
        raise ValueError(
            f"contrast_bound must be finite and >= 1 (a positive field's "
            f"max/min is always >= 1; a smaller bound admits only the "
            f"zero-field closure point); got {contrast_bound!r}"
        )
    dims = []
    for d in shape:
        # per-dim validation at authoring, funnel principle (the same
        # posture ir.py applies to Aval shapes): exact Python ints only —
        # no str coercion ("4" is malformed input, not a size), bool is
        # an int subclass and not an extent, and a negative extent names
        # an array no jax program constructs.
        if isinstance(d, bool) or not isinstance(d, int):
            raise ValueError(
                f"coefficient_contrast: shape {tuple(shape)!r} has a "
                f"non-int extent {d!r} ({type(d).__name__}); extents must "
                f"be Python ints (no coercion); refusing at authoring time"
            )
        if d < 0:
            raise ValueError(
                f"coefficient_contrast: shape {tuple(shape)!r} has a "
                f"negative extent {d!r}: no jax program constructs such an "
                f"array; refusing at authoring time"
            )
        dims.append(d)
    dims = tuple(dims)
    n_declared = 1
    for d in dims:
        n_declared *= d
    if n_declared == 0:
        raise ValueError(
            f"coefficient_contrast over shape {dims} declares zero "
            f"elements: max/min of an empty field does not exist (measured "
            f"jax refuses size-0 reductions); refusing at authoring time"
        )
    lo, hi = _closed_range("coefficient_contrast", "chi_range", chi_range)

    requires_description = (
        f"field = transform(chi) for chi over shape {dims} ({dtype}) with "
        f"every element in [{lo!r}, {hi!r}]"
        f"{' (identity transform)' if transform is None else ''}: "
        f"field >= 0 at every point (closed positivity) AND max(field) <= "
        f"{bound!r} * min(field) (division-free multiplicative contrast "
        f"form), per produced value, at every point of the declared "
        f"envelope — the closed-form posing of 'the field is strictly "
        f"positive with contrast max/min <= {bound!r}', exact up to the "
        f"single closure point (the identically-zero field)"
    )

    def harness():
        from stelling._jax_compat import fold_extrema
        from stelling.harness import any_array, assert_

        chi = any_array(dims, dtype, (lo, hi))
        value = transform(chi) if transform is not None else chi
        values = (
            tuple(value) if isinstance(value, (tuple, list)) else (value,)
        )
        if not values:
            # loud at trace time, never a silent zero-obligation UNKNOWN
            raise ValueError(
                "coefficient_contrast: the transform returned no values — "
                "nothing to pose; the transform convention mirrors "
                "field_positive: return the produced field, or a "
                "tuple/list of produced values"
            )
        obligations = []
        for v in values:
            obligations.append(assert_(v >= 0.0))
            mx, mn = fold_extrema(v)
            obligations.append(assert_(mx <= bound * mn))
        return tuple(obligations)

    return Contract(
        name="coefficient_contrast",
        requires_description=requires_description,
        harness=harness,
        ensures=None,
        no_ensures_reason=(
            f"none declared: contrast(field) <= {bound!r} does not by "
            f"itself bound the condition number of a discretized operator "
            f"— that step needs mesh/discretization constants this "
            f"contract does not have, and manufacturing one would state a "
            f"theorem the requires does not imply. This contract is "
            f"requires-only."
        ),
    )


# -- the checker --------------------------------------------------------------


def check_contract(contract, *, vacuity_mode, solver_timeout_ms=None, refine=None):
    """Check a contract's requires face through the standard pipeline and
    return the :class:`ContractVerdict` carrying both faces.

    The requires face runs exactly :func:`stelling.preconditions.check`'s
    machinery (shared helper — one pipeline): interval propagation over
    the traced obligations, the affine refinement **only** under an
    explicit ``refine="affine"`` (never-on defaults, exactly as in
    ``check()``; an unknown value raises eagerly), solver escalation
    **only** under an explicit ``solver_timeout_ms`` (never-on
    defaults), the full standard stamp, and — ``vacuity_mode`` being
    required here exactly as there — the widen re-check on every
    VERIFIED at the same pipeline depth.

    Interval-undecided obligations arrive with the straddle already
    QUOTED in their detail — the propagated closed intervals of both
    comparison sides, produced by the ONE propagation layer itself
    (docs/proposed-decline-messages.md #1), so an UNKNOWN says which
    numbers failed to separate on every front door. This layer used to
    mirror that quote as its own bolted-on note from a RE-propagated
    inert-mode env; the mirror is gone — one pipeline, one quote, judged
    boxes.

    One contract-layer disclosure rides on the returned requires verdict,
    via the standard append-only mechanics: a declared ensures face is
    recorded in the stamp's assumption lines (derivation quoted, DECLARED
    named); an absent one is recorded as a note quoting the contract's
    reason.

    The ensures face itself is carried through UNCHANGED — this function
    cannot mint, upgrade, or drop one.
    """
    import dataclasses

    from stelling.preconditions import _pipeline

    verdict, _ = _pipeline(
        contract.harness,
        vacuity_mode=vacuity_mode,
        solver_timeout_ms=solver_timeout_ms,
        refine=refine,
    )
    if contract.ensures is not None:
        ensures_line = (
            f"contract {contract.name!r} ensures face: "
            f"{contract.ensures.status} — declared, never checked by any "
            f"mechanized step; conditional on the requires face and holding "
            f"only where it holds (vacuously true if no envelope point "
            f"satisfies the requires); derivation: "
            f"{contract.ensures.derivation}"
        )
        stamp = dataclasses.replace(
            verdict.stamp,
            assumptions=verdict.stamp.assumptions + (ensures_line,),
        )
        verdict = dataclasses.replace(verdict, stamp=stamp)
    else:
        none_line = (
            f"contract {contract.name!r} ensures face: "
            f"{contract.no_ensures_reason}"
        )
        verdict = dataclasses.replace(
            verdict, notes=verdict.notes + (none_line,)
        )
    return ContractVerdict(
        contract_name=contract.name,
        requires_description=contract.requires_description,
        requires=verdict,
        ensures=contract.ensures,
        no_ensures_reason=contract.no_ensures_reason,
    )
