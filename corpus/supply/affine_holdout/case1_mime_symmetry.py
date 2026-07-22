# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

# Holdout case 1 runner — the commuted-product symmetry pair in the real
# MIME LSQ socket (KA-B of
# /home/nick/MSF/msf/MIME/verification/stelling/mime_lsq_conditioning.py).
#
# MEASUREMENT ONLY. This script replicates the socket's KA-B posing
# verbatim (assembler + transform copied from the socket file, provenance
# lines cited) so the symmetry pair can be measured in isolation:
#   run A: interval-only  -> the pair's straddle notes, verbatim
#   run B: escalated      -> the solver invocations spent ONLY on the pair
# The socket itself asserts (line 197) that scatter-add has NO transfer;
# under stelling >= commit 3f78fdd that assertion fails, so the socket
# cannot run un-modified — this runner carries the KA-B measurement
# forward unchanged (the KA transforms never used segment_sum; they use
# the unrolled assembler, which traces to supported primitives only).
#
# Run:
#   <venv-jax>/bin/python case1_mime_symmetry.py
# Requires: stelling, mime-engine, jax importable (venv-jax has all).

import inspect
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_2d  # noqa: E402
from mime.nodes.environment.fvm.operators import grad_least_squares  # noqa: E402

import stelling  # noqa: E402
from stelling._jax_compat import jax_version  # noqa: E402
from stelling.contracts import check_contract, conditioning_2x2_field  # noqa: E402
from stelling.harness import any_array  # noqa: E402

print(
    f"pinned: jax {jax_version()} | stelling {stelling.__version__} | "
    f"socket source: /home/nick/MSF/msf/MIME/verification/stelling/"
    f"mime_lsq_conditioning.py (KA-B, replicated verbatim)"
)

KAPPA = 8.0
TIMEOUT_MS = 20000
VMODE = "inputs-only"

# reg from the REAL function's signature default (socket line 108)
REG = inspect.signature(grad_least_squares).parameters["reg"].default
assert REG == 1e-30, f"operators.py reg default moved: {REG!r}"

# the real mesh (socket lines 119-139)
MESH = make_cartesian_mesh_2d(2, 1, 1.0, 1.0, dtype=jnp.float64)
D_INT = np.asarray(MESH.d)
PATCH_NAMES = tuple(p.name for p in MESH.patches)
PATCH_OWNER = {p.name: np.asarray(p.owner) for p in MESH.patches}
PATCH_D = {p.name: np.asarray(p.d) for p in MESH.patches}


def mesh_static(mesh):  # socket lines 237-246
    return {
        "n_cells": mesh.N_cells,
        "n_faces": mesh.N_faces,
        "owner": np.asarray(mesh.owner),
        "neigh": np.asarray(mesh.neighbour),
        "patch_names": tuple(p.name for p in mesh.patches),
        "patch_owner": {p.name: np.asarray(p.owner) for p in mesh.patches},
    }


_STATIC = mesh_static(MESH)


def _cat_scalars(*scalars):  # socket lines 174-178
    return jnp.concatenate([jnp.reshape(s, (1,)) for s in scalars])


def assemble_M_family(ms, d_int, patch_d, fed):  # socket lines 249-290
    rows = []
    for P in range(ms["n_cells"]):
        groups = []
        g = [d_int[f] for f in range(ms["n_faces"]) if ms["owner"][f] == P]
        if g:
            groups.append(g)
        g = [d_int[f] for f in range(ms["n_faces"]) if ms["neigh"][f] == P]
        if g:
            groups.append(g)
        for name in ms["patch_names"]:
            if name not in fed:
                continue
            po = ms["patch_owner"][name]
            g = [patch_d[name][k] for k in range(po.size) if po[k] == P]
            if g:
                groups.append(g)
        assert groups, f"cell {P} has an empty stencil on this mesh"
        m00 = m01 = m10 = m11 = None
        for g in groups:
            s00 = s01 = s10 = s11 = None
            for (dx, dy) in g:
                t00, t01, t10, t11 = dx * dx, dx * dy, dy * dx, dy * dy
                s00 = t00 if s00 is None else s00 + t00
                s01 = t01 if s01 is None else s01 + t01
                s10 = t10 if s10 is None else s10 + t10
                s11 = t11 if s11 is None else s11 + t11
            m00 = s00 if m00 is None else m00 + s00
            m01 = s01 if m01 is None else m01 + s01
            m10 = s10 if m10 is None else m10 + s10
            m11 = s11 if m11 is None else m11 + s11
        m00 = m00 + REG
        m11 = m11 + REG
        rows.append(_cat_scalars(m00, m01, m10, m11))
    return jnp.reshape(jnp.concatenate(rows), (ms["n_cells"], 2, 2))


def _declared_point(v):  # socket lines 516-517
    return any_array((), "float64", (float(v), float(v)))


def transform_fed(theta):  # socket lines 555-564 (KA-B)
    d_int = [(theta, _declared_point(D_INT[0, 1]))]
    patch_d = {
        name: [
            (_declared_point(PATCH_D[name][k, 0]), _declared_point(PATCH_D[name][k, 1]))
            for k in range(PATCH_OWNER[name].size)
        ]
        for name in PATCH_NAMES
    }
    return assemble_M_family(_STATIC, d_int, patch_d, fed=PATCH_NAMES)


def contract():
    return conditioning_2x2_field(
        (), "float64", (float(D_INT[0, 0]), float(D_INT[0, 0])), KAPPA,
        transform_fed,
    )


# --- run A: interval-only (no solver) ---------------------------------------
print("\n==== run A: KA-B posing, interval path only (no solver_timeout_ms)")
t0 = time.perf_counter()
ka_b_iv = check_contract(contract(), vacuity_mode=VMODE)
t_iv = time.perf_counter() - t0
print(f"  wall time: {t_iv:.3f} s")
print(f"  requires_status: {ka_b_iv.requires_status}")
print(f"  obligation statuses: {[o.status for o in ka_b_iv.requires.obligations]}")
for i, o in enumerate(ka_b_iv.requires.obligations):
    print(f"  obligation #{i}: {o.status} — {o.detail}")
print("  requires notes, verbatim:")
for n in ka_b_iv.requires.notes:
    print(f"    note: {n}")

# --- run B: escalated --------------------------------------------------------
print(f"\n==== run B: same posing, solver_timeout_ms={TIMEOUT_MS}")
t0 = time.perf_counter()
ka_b = check_contract(contract(), vacuity_mode=VMODE, solver_timeout_ms=TIMEOUT_MS)
t_esc = time.perf_counter() - t0
print(f"  wall time: {t_esc:.3f} s")
print(f"  requires_status: {ka_b.requires_status}")
for i, o in enumerate(ka_b.requires.obligations):
    print(f"  obligation #{i}: {o.status} — {o.detail}")
stamps = ka_b.requires.stamp.solver
stamps = stamps if isinstance(stamps, tuple) else (stamps,)
print(f"  solver invocations: {len(stamps)}")
for i, s in enumerate(stamps):
    print(f"    [{i}] invoked={s.invoked} {s.name} {s.version} — {s.reason}")
print("  requires notes, verbatim (per-solver timings live here):")
for n in ka_b.requires.notes:
    print(f"    note: {n}")

# which asserts the invocations were spent on, tallied from the reasons
import re  # noqa: E402

spent = {}
for s in stamps:
    m = re.search(r"assert #(\d+)", s.reason)
    if m:
        spent.setdefault(int(m.group(1)), 0)
        spent[int(m.group(1))] += 1
print(f"  invocations per assert index: {spent}")
print(
    "  (asserts #4/#5 are the posed symmetry pair M[..,0,1] <= M[..,1,0] "
    "and converse — contracts.py:690/691; #0-#3 are the conditioning "
    "conjuncts)"
)
