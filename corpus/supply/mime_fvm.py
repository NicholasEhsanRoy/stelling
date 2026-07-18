# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Three harnesses against the REAL MIME fvm non-orthogonal Laplacian.

Target: mime-engine @ 7ce1efb4311b, ``src/mime/nodes/environment/fvm/``
(operators.py + mesh.py — the only two sim files this harness reads or
imports from).

F1 poses the over-relaxed orthogonal coefficient of the non-orthogonal
path as a hand transcription — operators.py:251-253, ``E_f = |Sf|^2 /
(Sf.d)`` in polar form with the a^2/a cancellation done by hand — over
declared face-geometry boxes, then re-runs it under an INPUTS-ONLY
⊤-widening (only non-point ``stelling_any`` declarations widen; the
nonvacuity point declarations stay). F2 poses the same quantity in raw
vector form, with the cos-angle bound stated as an ``assume`` on the dot
product instead of a declaration box. F3 imports the real operator
(``laplacian_orthogonal(..., non_orthogonal=True)``) and the real mesh
constructor (``make_cartesian_mesh_2d``) and runs the operator on the
smallest mesh with a non-empty interior face graph.

Mesh choice, recorded: ``make_cartesian_mesh_2d(1, 1, ...)`` constructs
(1 cell) but has ``N_faces=0`` — an empty interior face graph whose trace
contains none of the operator's gather→compute→scatter structure, so
there is no face flux for the harness to say anything about. The
smallest mesh the constructor can produce that exercises the operator is
2x1: N_cells=2, N_faces=1 (interior), 6 boundary faces across 4 patches.
The mesh object and the phi prototype are built OUTSIDE the trace (the
prototype-outside-trace pattern from maddening_cfl.py); mesh arrays enter
the query as consts.

Precision, recorded: everything runs under ``jax_enable_x64=True``. The
F1/F2 declarations are float64 scalars whose verbatim python-float
constants (0.71, 8.0, the box endpoints) must meet them without a
narrowing ``convert_element_type[f64->f32]`` — the x64-off failure mode
recorded in maddening_cfl.py's docstring. F3's mesh and field arrays are
float32, pinned by ``make_cartesian_mesh_2d``'s explicit dtype default,
so the operator's real dtypes trace unchanged under x64; only
jax-internal index plumbing (lu pivots, iota) widens to int64, and none
of it carries field data.
"""

import math
from collections import Counter
from importlib.metadata import version as _pkg_version

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_2d  # noqa: E402
from mime.nodes.environment.fvm.operators import laplacian_orthogonal  # noqa: E402

import stelling  # noqa: E402
from stelling import ir  # noqa: E402
from stelling.coverage import DEFAULT_TRANSPARENT, sub_jaxprs  # noqa: E402
from stelling._jax_compat import jax_version, trace  # noqa: E402
from stelling.harness import (  # noqa: E402
    any_array,
    any_pytree,
    assert_,
    assume,
    nonvacuity,
)
from stelling.propagate import TRANSFERS, propagate  # noqa: E402
from stelling.verdict import make_verdict  # noqa: E402

INF = math.inf

PRECISION = (
    "jax_enable_x64=True — F1/F2 declare float64 scalars that must meet "
    "their verbatim python-float constants (0.71, 8.0) without a narrowing "
    "convert_element_type[f64->f32] (the x64-off failure mode recorded in "
    "maddening_cfl.py); F3's mesh/field arrays are float32 by "
    "make_cartesian_mesh_2d's explicit dtype default and trace unchanged — "
    "only jax-internal lu/iota index plumbing widens to int64, none of it "
    "carrying field data"
)

try:
    MIME_VERSION = _pkg_version("mime-engine")
except Exception:  # pragma: no cover - version report only
    MIME_VERSION = "unversioned"

print(
    f"pinned: mime-engine {MIME_VERSION} @ 7ce1efb4311b | "
    f"jax {jax_version()} | stelling {stelling.__version__}"
)


# --- INPUTS-ONLY ⊤-widening --------------------------------------------------
#
# The tautology_test.py widen() pattern with one added condition: only
# stelling_any equations whose DECLARED bounds are not a point (lo != hi)
# are rewritten to (-inf, +inf). Point declarations (the nonvacuity
# operating points) and literals stay untouched.


def widen_inputs_only(closed: ir.ClosedJaxpr) -> ir.ClosedJaxpr:
    j = closed.jaxpr
    for eqn in j.eqns:  # loud guard: no nested declarations may escape
        for sub in sub_jaxprs(eqn):
            for e in sub.eqns:
                assert e.primitive != "stelling_any", (
                    "nested stelling_any would escape widening — restructure"
                )
    eqns = []
    for eqn in j.eqns:
        if eqn.primitive == "stelling_any":
            params = dict(eqn.params)
            if params["lo"] != params["hi"]:  # non-point declarations only
                params["lo"], params["hi"] = -INF, INF
                eqn = ir.JaxprEqn(
                    primitive=eqn.primitive,
                    invars=eqn.invars,
                    outvars=eqn.outvars,
                    params=tuple(params.items()),
                    effects=eqn.effects,
                    source_info=eqn.source_info,
                )
        eqns.append(eqn)
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=j.constvars,
            invars=j.invars,
            outvars=j.outvars,
            eqns=tuple(eqns),
            effects=j.effects,
            debug_info=j.debug_info,
        ),
        consts=closed.consts,
    )


def pose(name, harness):
    """Trace and propagate; if blocked, report the exact error and stage."""
    print(f"==== {name}")
    try:
        cj = trace(harness)
    except Exception as e:
        print(f"  BLOCKED at trace/transcribe: {type(e).__name__}: {e}")
        print()
        return None, None
    try:
        p = propagate(cj)
    except Exception as e:
        print(f"  BLOCKED at propagation: {type(e).__name__}: {e}")
        print()
        return cj, None
    print(f"  POSED. obligations: {[o.status for o in p.obligations]}")
    if p.nonvacuity_checks:
        print(f"  nonvacuity: {[c.status for c in p.nonvacuity_checks]}")
    print(f"  coverage: {p.coverage.summary()}")
    print()
    return cj, p


# --- HARNESS F1: the scalar arithmetic check ---------------------------------
#
# A transcription of one expression; the sim package is not imported in
# the harness body.


def h_f1():
    # transcribes operators.py:251-253, E_f = |Sf|^2/(Sf.d), polar form with the exact a^2/a cancellation
    a = any_array((), "float64", (0.5, 2.0))    # face area |Sf|
    dm = any_array((), "float64", (0.5, 2.0))   # |d|
    c = any_array((), "float64", (0.71, 1.0))   # cos(Sf,d)
    ob = assert_(a / (dm * c) <= 8.0)
    # nonvacuity: a stated operating point strictly inside each box
    a0 = any_array((), "float64", (1.0, 1.0))
    dm0 = any_array((), "float64", (1.0, 1.0))
    c0 = any_array((), "float64", (0.9, 0.9))
    return (
        ob,
        nonvacuity(a0 > 0.5),
        nonvacuity(a0 < 2.0),
        nonvacuity(dm0 > 0.5),
        nonvacuity(dm0 < 2.0),
        nonvacuity(c0 > 0.71),
        nonvacuity(c0 < 1.0),
    )


# --- HARNESS F2: the raw-vector form -----------------------------------------


def h_f2():
    sfx = any_array((), "float64", (-2.0, 2.0))
    sfy = any_array((), "float64", (-2.0, 2.0))
    sfz = any_array((), "float64", (-2.0, 2.0))
    dx_ = any_array((), "float64", (-2.0, 2.0))
    dy_ = any_array((), "float64", (-2.0, 2.0))
    dz_ = any_array((), "float64", (-2.0, 2.0))
    dot = sfx * dx_ + sfy * dy_ + sfz * dz_
    area2 = sfx * sfx + sfy * sfy + sfz * sfz
    sf_norm = jnp.sqrt(area2)
    d_norm = jnp.sqrt(dx_ * dx_ + dy_ * dy_ + dz_ * dz_)
    assume(dot >= 0.71 * sf_norm * d_norm)
    return assert_(area2 / dot <= 8.0)


# --- HARNESS F3: the real imported operator ----------------------------------
#
# laplacian_orthogonal(phi, mesh, *, mu_face=1.0, boundary_specs=None,
#                      non_orthogonal=False, grad_boundary_values=None)
# is the fvm operator with the non_orthogonal=True flux path
# (operators.py:179-318). The mesh comes from make_cartesian_mesh_2d, a
# constructor defined inside the fvm package (mesh.py:222). Both the mesh
# and the phi prototype are constructed OUTSIDE the trace.

MESH = make_cartesian_mesh_2d(2, 1, 1.0, 1.0)  # smallest with N_faces > 0
N_BOUNDARY = sum(int(p.owner.size) for p in MESH.patches)
print(
    f"real mesh: make_cartesian_mesh_2d(2, 1, 1.0, 1.0) -> N_cells={MESH.N_cells} "
    f"N_faces={MESH.N_faces} (interior), {N_BOUNDARY} boundary faces across "
    f"{len(MESH.patches)} patches {[p.name for p in MESH.patches]}; "
    f"field dtype {np.asarray(MESH.V).dtype}"
)
print(
    "  (1x1 constructs — 1 cell — but has N_faces=0: an empty interior face "
    "graph traces none of the operator's face-flux path, so 2x1 is the "
    "smallest mesh that exercises it)"
)

PHI_PROTO = np.zeros((MESH.N_cells,), dtype=np.asarray(MESH.V).dtype)


def h_f3():
    phi = any_pytree(PHI_PROTO, (0.0, 1.0))  # per-cell field box, real dtype
    out = laplacian_orthogonal(phi, MESH, mu_face=1.0, non_orthogonal=True)
    ob = assert_(out <= 1e6)  # every element of the output
    # nonvacuity: the point field value 0.5 lies inside the declared box
    y0 = any_array((), str(PHI_PROTO.dtype), (0.5, 0.5))
    return ob, nonvacuity(y0 > 0.0), nonvacuity(y0 < 1.0)


# --- run ---------------------------------------------------------------------

cj_f1, p_f1 = pose(
    "HARNESS F1 — E_f transcription over face-geometry boxes", h_f1
)
if cj_f1 is not None:
    wp = propagate(widen_inputs_only(cj_f1))
    stats = [o.status for o in wp.obligations]
    survived = [i for i, s in enumerate(stats) if s == "discharged"]
    print(
        f"F1 under INPUTS-ONLY ⊤-widening (every non-point stelling_any -> "
        f"(-inf, +inf); the point declarations stay): obligations {stats}, "
        f"nonvacuity {[c.status for c in wp.nonvacuity_checks]}"
    )
    if survived:
        print(
            f"  -> TAUTOLOGICAL: obligation(s) {survived} still discharge "
            f"without the declared bounds"
        )
    else:
        print(
            "  -> the obligation does NOT survive ⊤ — the declared bounds "
            "are load-bearing"
        )
    print()

cj_f2, p_f2 = pose("HARNESS F2 — raw-vector form, cos bound as an assume", h_f2)
if p_f2 is not None:
    print(f"F2 obligation status: {[o.status for o in p_f2.obligations]}")
    print("F2 propagation notes, verbatim:")
    if not p_f2.notes:
        print("  (none)")
    for n in p_f2.notes:
        print(f"  note: {n}")
    print(f"F2 coverage line: {p_f2.coverage.summary()}")
    print()

cj_f3, p_f3 = pose(
    "HARNESS F3 — real laplacian_orthogonal(non_orthogonal=True) on the "
    "real 2x1 mesh",
    h_f3,
)

# --- verdict renders ---------------------------------------------------------

for label, cj, p in (
    ("F1", cj_f1, p_f1),
    ("F2", cj_f2, p_f2),
    ("F3", cj_f3, p_f3),
):
    print(f"== verdict {label}")
    if cj is None or p is None:
        print("  (blocked before propagation — no verdict to render)")
        print()
        continue
    v = make_verdict(
        cj,
        p,
        stelling_version=stelling.__version__,
        jax_version=jax_version(),
        precision_config=PRECISION,
    )
    print(v.render())
    print()


# --- F3 census table: every ⊤/declined/unreached primitive -------------------


def f3_census(cj: ir.ClosedJaxpr, p) -> None:
    if cj is None or p is None:
        print("  (harness F3 did not propagate — no census)")
        return
    # unreached equations, per primitive, with the ⊤ owner that hid them —
    # a static walk replicating the propagator's traversal (transparent
    # wrappers descend; known/handled primitives descend; unknown
    # primitives bury their sub-jaxprs). Totals are cross-checked against
    # the live counter below, loudly.
    handled = set(TRANSFERS) | {"cond", "stelling_assume"}
    unreached: Counter = Counter()
    owners: dict[str, Counter] = {}

    def walk(jx: ir.Jaxpr, reached: bool, owner: str | None) -> None:
        for e in jx.eqns:
            if not reached:
                unreached[e.primitive] += 1
                owners.setdefault(e.primitive, Counter())[owner] += 1
                for s in sub_jaxprs(e):
                    walk(s, False, owner)
            elif e.primitive in DEFAULT_TRANSPARENT:
                for s in sub_jaxprs(e):
                    walk(s, True, None)
            elif e.primitive in handled:
                for s in sub_jaxprs(e):
                    walk(s, True, None)
            else:
                for s in sub_jaxprs(e):
                    walk(s, False, e.primitive)

    walk(cj.jaxpr, True, None)
    assert sum(unreached.values()) == p.coverage.unreached, (
        "census walk disagrees with the live coverage counter — fix the walk"
    )
    print(
        "status      primitive             count  reason"
    )
    for name, count in p.coverage.unknown_primitives:
        decline_notes = [n for n in p.notes if repr(name) in n]
        if name in TRANSFERS:
            status = "declined"
            reason = " | ".join(decline_notes) or (
                "registered transfer declined this form"
            )
        else:
            status = "⊤"
            reason = "no transfer registered for this primitive; outputs bound to ⊤"
        print(f"{status:<11} {name:<21} ×{count:<5} {reason}")
    for name, count in sorted(unreached.items(), key=lambda kv: (-kv[1], kv[0])):
        whose = ", ".join(f"{o!r}" for o in owners[name] if o)
        print(
            f"{'unreached':<11} {name:<21} ×{count:<5} never analyzed — inside "
            f"a sub-jaxpr of ⊤ {whose}"
        )
    print(f"totals: {p.coverage.summary()}")
    print(
        "transfers used: "
        + ", ".join(f"{n} [{t}]" for n, t in p.transfers_used)
    )


print("== F3 census: every ⊤/declined/unreached primitive")
f3_census(cj_f3, p_f3)
