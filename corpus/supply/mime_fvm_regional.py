# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Two regional obligations on the MIME fvm over-relaxed orthogonal coefficient.

Target: ``/home/nick/MSF/msf/MIME/src/mime/nodes/environment/fvm/``
(``operators.py`` + ``mesh.py`` — read in place; this module IMPORTS
nothing from mime, both harnesses are hand transcriptions). The sibling
harness ``mime_fvm.py`` records the pin as ``mime-engine @ 7ce1efb4311b``;
that hash is inherited from that file and is NOT re-verified here (the
repo's git history is out of scope for this transcription). The installed
distribution version is printed at import time instead.

THE TRANSCRIBED SOURCE, verbatim
--------------------------------
``operators.py`` lines 248-253 (the non-orthogonal branch of
``laplacian_orthogonal``)::

    248:    if non_orthogonal:
    249:        # Over-relaxed orthogonal coefficient E_f = |Sf|² / (Sf·d).
    250:        Sf_dot_d = jnp.sum(mesh.Sf * mesh.d, axis=1)        # [N_faces]
    251:        ortho_coeff = mesh.area ** 2 / Sf_dot_d
    252:    else:
    253:        ortho_coeff = mesh.area / mesh.d_mag                # |Sf|/|d|

What ``area`` IS, so that "``area`` as the code obtains it" is not a
guess — three independent statements in the same package::

    mesh.py:48-50   Sf : ... Outward face area vector (magnitude = face
                    area, direction = outward normal).
    mesh.py:53-54   area : ... Face area magnitude.
    unstructured.py:13   * ``Sf = area · n`` with ``n`` the unit face
                    normal oriented **owner → neighbour**

i.e. ``area == |Sf|``. R2 therefore obtains ``area`` as
``sqrt(Sf_x² + Sf_y² + Sf_z²)`` and then squares it, exactly as
operators.py:251 writes ``mesh.area ** 2``. No ``|Sf|² = Sf·Sf``
identity is applied in R2 — that substitution is hand algebra, and R2
is posed as the un-simplified form.

WHAT THIS POSES
---------------
R1 — the polar form, with two DISCLOSED DERIVATIONS applied by hand
before transcription:

  (D1) polar substitution: ``Sf·d = |Sf|·|d|·cos(Sf,d)``. Exact in ℝ;
       it is the definition of the angle between two nonzero vectors.
       Introduces the alignment cosine ``c`` as a first-class declared
       quantity, which the raw form has no name for.
  (D2) the ``a²/a`` cancellation: with ``a = |Sf|``,
       ``a² / (a·dm·c) = a / (dm·c)``. Exact in ℝ **for a > 0 only**;
       the declared box ``a ∈ [0.5, 2.0]`` excludes a = 0, so the
       cancellation is licensed on the declared region. It is NOT
       licensed in general (a degenerate zero-area face makes both
       sides 0/0), and it is NOT an identity in binary64 — under the
       stamped ``semantics="real"`` that gap is out of scope, and the
       stamp says so.

  Declared: a = |Sf| ∈ [0.5, 2.0]; dm = |d| ∈ [0.5, 2.0];
  c = cos(Sf,d) ∈ [0.11, 1.0].
  Obligation:  a / (dm · c) <= 8.0

R2 — the code's raw form, NO derivation, NO ``assume``, and no lower
bound on the alignment. Six independent component declarations; the dot
product written out as ``jnp.sum(mesh.Sf * mesh.d, axis=1)`` expands it;
``area`` obtained as the magnitude of the declared ``Sf`` and squared as
operators.py:251 squares it.

  Obligation:  area² / (Sf_x·d_x + Sf_y·d_y + Sf_z·d_z) <= 8.0

R2 COMPONENT BOXES, and why
---------------------------
Every component of ``Sf`` and of ``d`` is declared in
``[-1.1547005383792515, +1.1547005383792515]``.

That endpoint is the largest float64 ``x`` with ``3·x² <= 4`` exactly
(checked below with ``fractions.Fraction``, and asserted at import) —
i.e. the box is the origin-centred cube inscribed in the ball of radius
2.0. Rationale: R1's magnitude ceiling is ``|Sf| <= 2.0`` and
``|d| <= 2.0``, and this is the largest axis-aligned component box that
NEVER exceeds it: the corner magnitude is
``sqrt(3.9999999999999996) < 2.0``, so the ceiling is tight to within
5e-17 and is never violated by any admitted point.

The disclosed mismatch, which no component box can remove: an
axis-aligned box that is closed under the sign symmetry the raw form has
(nothing in ``area²/(Sf·d)`` distinguishes an octant) necessarily
contains the origin, so it cannot express R1's magnitude FLOOR
(``|Sf| >= 0.5``, ``|d| >= 0.5``). The declared R2 region therefore
admits magnitudes in [0, 2.0], and it also omits some vectors with
magnitude in [0.5, 2.0] that lie outside the inscribed cube (e.g.
``(2, 0, 0)``). The R2 region is thus neither a subset nor a superset of
R1's magnitude region — it is the honest box hull of the same ceiling,
and the missing floor is a declared fact about R2, not a defect to be
patched with an ``assume`` (the spec forbids one, and this harness adds
none).

The rejected alternative, recorded: ``[-2.0, +2.0]`` per component (the
circumscribed cube, the choice the sibling ``mime_fvm.py`` h_f2 makes)
admits ``|Sf|`` up to ``2·sqrt(3) ≈ 3.464``, which is not consistent
with the stated ceiling of 2.0.

R2b — a labelled SIDE MEASUREMENT, not a substitute for R2
----------------------------------------------------------
R2's numerator goes through ``sqrt`` then ``integer_pow``, neither of
which has a registered transfer. R2b re-poses the SAME obligation over
the SAME boxes with one hand identity applied — ``|Sf|² = Sf·Sf``, i.e.
``area² = Sf_x² + Sf_y² + Sf_z²`` (exact in ℝ, and the substitution the
sibling h_f2 makes) — for the sole purpose of attributing what R2's ⊤
cost. R2's verdict is the R2 result; R2b is reported separately and
decides nothing about R2. Nothing else differs: same boxes, same bound,
no assume.

PRECISION, recorded
-------------------
Everything runs under ``jax_enable_x64=True``. Both harnesses declare
float64 scalars whose verbatim python-float constants (0.11, 0.5, 2.0,
8.0, 1.1547005383792515) must meet them without a narrowing
``convert_element_type[f64->f32]`` — the x64-off failure mode recorded in
``maddening_cfl.py``'s docstring. NOTE the disclosed dtype gap: the real
mesh arrays are float32 (``mesh.py:111-115`` pins ``Sf``/``area``/``d``
as float32), so these declarations are a dtype WIDER than the arrays the
transcribed lines actually carry. Under the stamped
``semantics="real"`` the obligation is judged in exact real arithmetic
and the declaration dtype does not change what is judged; under
``semantics="ieee"`` it would, and no run here is an ieee run.
"""

import math
from fractions import Fraction
from importlib.metadata import version as _pkg_version

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import stelling  # noqa: E402
from stelling._jax_compat import jax_version, trace  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.solvers import (  # noqa: E402
    SolverConfig,
    escalate,
    make_solver_verdict,
)
from stelling.verdict import make_verdict  # noqa: E402

# --- the R2 component-box endpoint, and its exact justification --------------

COMP = 1.1547005383792515  # verbatim literal; no runtime sqrt in the declaration
_c = Fraction(COMP)
assert 3 * _c * _c <= 4, "the declared cube is not inscribed in the ball |v| <= 2"
assert 3 * Fraction(math.nextafter(COMP, math.inf)) ** 2 > 4, (
    "a larger float64 endpoint would still be inscribed — use it instead"
)
MAX_CORNER_SQ = float(3 * _c * _c)  # exact rational, printed as a float

PRECISION = (
    "jax_enable_x64=True — R1/R2 declare float64 scalars that must meet their "
    "verbatim python-float constants (0.11, 0.5, 2.0, 8.0, 1.1547005383792515) "
    "without a narrowing convert_element_type[f64->f32] (the x64-off failure "
    "mode recorded in maddening_cfl.py); disclosed gap: the real mesh arrays "
    "these lines carry are float32 (mesh.py:111-115), so the declarations are "
    "wider than the transcribed code's dtypes — immaterial under the stamped "
    "semantics=real (exact real arithmetic), material under ieee, and no run "
    "here is an ieee run"
)

TIMEOUT_MS = 20_000

try:
    MIME_VERSION = _pkg_version("mime-engine")
except Exception:  # pragma: no cover - version report only
    MIME_VERSION = "unversioned"

print(
    f"pinned: mime-engine {MIME_VERSION} (commit hash NOT verified here) | "
    f"jax {jax_version()} | stelling {stelling.__version__}"
)
print(
    f"R2 component box: [-{COMP}, +{COMP}] per component; largest float64 with "
    f"3x^2 <= 4 exactly (3x^2 = {MAX_CORNER_SQ!r}, corner |v| = "
    f"{math.sqrt(MAX_CORNER_SQ)!r} <= 2.0)"
)
print()


# --- harnesses ---------------------------------------------------------------


def h_r1():
    """R1 — polar form of operators.py:250-251, derivations D1+D2 applied.

    E_f = |Sf|^2 / (Sf.d)  ==(D1)==  a^2 / (a * dm * c)  ==(D2, a>0)==
    a / (dm * c).
    """
    a = any_array((), "float64", (0.5, 2.0))    # |Sf|  (mesh.py:53-54 "area")
    dm = any_array((), "float64", (0.5, 2.0))   # |d|   (mesh.py:115 "d_mag")
    c = any_array((), "float64", (0.11, 1.0))   # cos(Sf, d)
    return assert_(a / (dm * c) <= 8.0)


def h_r2():
    """R2 — the raw form of operators.py:250-251, no derivation, no assume.

    Sf_dot_d = jnp.sum(mesh.Sf * mesh.d, axis=1)  written out componentwise;
    area obtained as |Sf| (mesh.py:48-54, unstructured.py:13) and squared
    exactly as ``mesh.area ** 2`` squares it.
    """
    sfx = any_array((), "float64", (-COMP, COMP))
    sfy = any_array((), "float64", (-COMP, COMP))
    sfz = any_array((), "float64", (-COMP, COMP))
    dx_ = any_array((), "float64", (-COMP, COMP))
    dy_ = any_array((), "float64", (-COMP, COMP))
    dz_ = any_array((), "float64", (-COMP, COMP))
    Sf_dot_d = sfx * dx_ + sfy * dy_ + sfz * dz_          # operators.py:250
    area = jnp.sqrt(sfx * sfx + sfy * sfy + sfz * sfz)    # area == |Sf|
    return assert_(area**2 / Sf_dot_d <= 8.0)             # operators.py:251


def h_r2b():
    """R2b — SIDE MEASUREMENT ONLY. R2 with one hand identity: |Sf|^2 = Sf.Sf.

    Same boxes, same bound, no assume. Exists only to attribute R2's ⊤;
    it is not R2's verdict.
    """
    sfx = any_array((), "float64", (-COMP, COMP))
    sfy = any_array((), "float64", (-COMP, COMP))
    sfz = any_array((), "float64", (-COMP, COMP))
    dx_ = any_array((), "float64", (-COMP, COMP))
    dy_ = any_array((), "float64", (-COMP, COMP))
    dz_ = any_array((), "float64", (-COMP, COMP))
    Sf_dot_d = sfx * dx_ + sfy * dy_ + sfz * dz_
    area_sq = sfx * sfx + sfy * sfy + sfz * sfz  # |Sf|^2, identity applied
    return assert_(area_sq / Sf_dot_d <= 8.0)


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


def run(label, harness):
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
    return cj, p, e, v, sv


run("R1 — polar form, a/(dm*c) <= 8.0 over [0.5,2]x[0.5,2]x[0.11,1]", h_r1)
run("R2 — raw form, area^2/(Sf.d) <= 8.0 over the inscribed component cube", h_r2)
run(
    "R2b — SIDE MEASUREMENT ONLY (|Sf|^2 = Sf.Sf applied by hand); "
    "not R2's verdict",
    h_r2b,
)
