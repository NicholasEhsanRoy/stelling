# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Unguided precondition sweep of lineax's solver-shaped modules.

Mechanical census-and-pose over four precondition classes in
``lineax/_solver/*.py``, ``_solve.py``, ``_operator.py``, ``_norm.py``,
``_misc.py`` (lineax 0.1.1 as installed in the probe venv):

1. division / reciprocal — denominator nonzero;
2. sqrt / log / fractional pow — argument domain;
3. config scalar with a default — nonzero / admissible range;
4. matrix/field data whose property a solver assumes (tags, diagonals).

Every pose transcribes the code's own construction of the site quantity
(calling the target's own helpers — ``tree_dot``, ``two_norm``,
``resolve_rcond``, ``LSMR._givens`` — wherever the construction is a
callable unit, inlining the exact source lines where the quantity is an
unexposed intermediate). Loop-carried scalars are declared as bounded
inputs at loop entry (disclosed per pose). Envelopes: a generic
sign-unknown box (-10, 10) (int-ish configs: (-100, 100)) plus any
documented/enforced range, both posed. No outcome is preferred; a ⊤ or
a decline is a result to quote.

Run: ``venv-jax/bin/python lineax_preconditions.py`` (prints every
verdict render verbatim; interval-only first, then a 20 s SMT
escalation for whatever is interval-UNKNOWN).

Do not commit.
"""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jax.lax as lax  # noqa: E402

from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check, scalar_nonzero  # noqa: E402

# The target's own constructions, used as transforms verbatim.
from lineax._misc import resolve_rcond  # noqa: E402
from lineax._norm import sum_squares, tree_dot, two_norm  # noqa: E402
from lineax._solver.lsmr import LSMR  # noqa: E402

GEN = (-10.0, 10.0)  # generic sign-unknown box, disclosed
GEN_INT = (-100, 100)  # generic box for integer-valued configs
N = 4  # static vector length for array poses (SMT emission budget)

_lsmr = LSMR(rtol=1e-6, atol=1e-6)  # instance only to reach _givens verbatim


# ---------------------------------------------------------------------------
# bicgstab.py / cg.py / gmres.py — tolerance-scale denominators
# ---------------------------------------------------------------------------


def p01_scale_denom_documented():
    """bicgstab.py:96/122-123, cg.py:147/156-157, gmres.py:119/137-138.

    `b_scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω`
    then `norm1 = self.norm((r**ω / b_scale**ω).ω)` — denominator must be
    nonzero. atol, rtol over the enforced range [0, 10] ("Tolerances
    must be non-negative"); vector over the generic box.
    """
    atol = any_array((), "float64", (0.0, 10.0))
    rtol = any_array((), "float64", (0.0, 10.0))
    b = any_array((N,), "float64", GEN)
    b_scale = atol + rtol * jnp.abs(b)  # ω-form reduces to this for one leaf
    return (assert_(b_scale != 0.0),)


def p02_scale_denom_generic():
    """Same construction, all inputs over the generic box (-10, 10)."""
    atol = any_array((), "float64", GEN)
    rtol = any_array((), "float64", GEN)
    b = any_array((N,), "float64", GEN)
    b_scale = atol + rtol * jnp.abs(b)
    return (assert_(b_scale != 0.0),)


# ---------------------------------------------------------------------------
# bicgstab.py — loop-body denominators
# ---------------------------------------------------------------------------


def p03_bicgstab_rho():
    """bicgstab.py:139 `beta = (rho_new / rho) * (alpha / omega)` — rho.

    rho is loop-carried; its construction (line 138) is
    `rho_new = tree_dot(r0, r)`. r0, r declared at loop entry (generic).
    """
    r0 = any_array((N,), "float64", GEN)
    r = any_array((N,), "float64", GEN)
    rho = tree_dot(r0, r)
    return (assert_(rho != 0.0),)


def p04_bicgstab_omega():
    """bicgstab.py:139 — omega denominator; its construction (line 153)
    is `omega_new = tree_dot(s, t) / tree_dot(t, t)`. s, t declared."""
    s = any_array((N,), "float64", GEN)
    t = any_array((N,), "float64", GEN)
    omega = tree_dot(s, t) / tree_dot(t, t)
    return (assert_(omega != 0.0),)


def p05_bicgstab_alpha_denom():
    """bicgstab.py:147 `alpha_new = rho_new / tree_dot(r0, v_new)`.

    With default options: r0 = vector (y0 = 0), v_new = A @ p_new
    (identity preconditioner), MatrixLinearOperator.mv per
    _operator.py:271. b, A, p declared (generic).
    """
    b = any_array((N,), "float64", GEN)
    a_mat = any_array((N, N), "float64", GEN)
    p = any_array((N,), "float64", GEN)
    v_new = jnp.matmul(a_mat, p, precision=lax.Precision.HIGHEST)
    denom = tree_dot(b, v_new)
    return (assert_(denom != 0.0),)


def p06_bicgstab_tdott():
    """bicgstab.py:153 `omega_new = tree_dot(s, t) / tree_dot(t, t)` —
    denominator tree_dot(t, t)."""
    t = any_array((N,), "float64", GEN)
    return (assert_(tree_dot(t, t) != 0.0),)


# ---------------------------------------------------------------------------
# cg.py — denominators, config, tags
# ---------------------------------------------------------------------------


def p07_cg_inner_prod():
    """cg.py:171-173 `mat_p = operator.mv(p); inner_prod =
    tree_dot(mat_p, p); alpha = gamma / inner_prod`. A, p declared."""
    a_mat = any_array((N, N), "float64", GEN)
    p = any_array((N,), "float64", GEN)
    mat_p = jnp.matmul(a_mat, p, precision=lax.Precision.HIGHEST)
    inner_prod = tree_dot(mat_p, p)
    return (assert_(inner_prod != 0.0),)


def p08_cg_gamma_prev():
    """cg.py:202-205 `z = preconditioner.mv(r); gamma = tree_dot(z, r);
    beta = gamma / gamma_prev`. Default preconditioner is the identity
    (misc.py:37), so z = r; r declared at loop entry."""
    r = any_array((N,), "float64", GEN)
    z = r  # IdentityLinearOperator.mv fast path, _operator.py:776
    gamma = tree_dot(z, r)
    return (assert_(gamma != 0.0),)


def p09_cg_stabilise_every_int():
    """cg.py:72/198 `stabilise_every: int | None = 10`;
    `(eqxi.unvmap_max(step) % self.stabilise_every) == 0` — modulus
    denominator must be nonzero. Faithful int64 dtype, generic box."""
    n = any_array((), "int64", GEN_INT)
    return (assert_(n != 0),)


def p10_cg_stabilise_every_float():
    """Same site, float64 relaxation of the int config (disclosed),
    generic box."""
    _, ob = scalar_nonzero("float64", (-100.0, 100.0))
    return (ob,)


def p11_cg_stabilise_every_documented():
    """Same site, documented neighbourhood of the default 10:
    [1, 100] (docstring: 'Every `stabilise_every` steps')."""
    _, ob = scalar_nonzero("float64", (1.0, 100.0))
    return (ob,)


def p12_cg_operator_definite():
    """cg.py:95-99 — `CG()` requires a PSD/NSD *tag*; docstring: 'The
    operator should be positive or negative definite.' The tag is
    honoured, never checked (_operator.py:1878). Quadratic-form pose:
    x·(A@x) > 0 (definite) and >= 0 (the tag's literal claim)."""
    a_mat = any_array((N, N), "float64", GEN)
    x = any_array((N,), "float64", GEN)
    ax = jnp.matmul(a_mat, x, precision=lax.Precision.HIGHEST)
    form = tree_dot(x, ax)
    return (assert_(form > 0.0), assert_(form >= 0.0))


def p13_cg_preconditioner_definite():
    """cg.py:120-121 `if not is_positive_semidefinite(preconditioner):
    raise ValueError("The preconditioner must be positive definite.")`
    — tag honoured, never checked. Quadratic form on M."""
    m_mat = any_array((N, N), "float64", GEN)
    x = any_array((N,), "float64", GEN)
    mx = jnp.matmul(m_mat, x, precision=lax.Precision.HIGHEST)
    form = tree_dot(x, mx)
    return (assert_(form > 0.0),)


# ---------------------------------------------------------------------------
# gmres.py — _normalise denominator, config
# ---------------------------------------------------------------------------


def p14_gmres_safe_norm():
    """gmres.py:404-412 `norm = two_norm(x); eps = jnp.finfo(norm.dtype).eps;
    breakdown = norm < eps; safe_norm = jnp.where(breakdown, jnp.inf, norm);
    x_normalised = (x**ω / safe_norm).ω` — denominator safe_norm."""
    x = any_array((N,), "float64", GEN)
    norm = two_norm(x)
    eps = jnp.finfo(norm.dtype).eps
    breakdown = norm < eps
    safe_norm = jnp.where(breakdown, jnp.inf, norm)
    return (assert_(safe_norm != 0.0),)


def p15_gmres_restart_int():
    """gmres.py:62/128 `restart: int = 20`; `restart = min(self.restart,
    size)` — Krylov subspace size; admissible range needs restart >= 1.
    Faithful int64, generic box."""
    n = any_array((), "int64", GEN_INT)
    return (assert_(n > 0),)


def p16_gmres_restart_float():
    """Same site, float64 relaxation, generic box + documented [1, 100]
    (default 20)."""
    n = any_array((), "float64", (-100.0, 100.0))
    m = any_array((), "float64", (1.0, 100.0))
    return (assert_(n > 0.0), assert_(m > 0.0))


def p17_gmres_stagnation_iters():
    """gmres.py:63/150-151 `stagnation_iters: int = 20`;
    `stagnation_counter < self.stagnation_iters` — zero halts every
    solve at once; admissible range needs >= 1. Float relaxation,
    generic + documented [1, 100]."""
    n = any_array((), "float64", (-100.0, 100.0))
    m = any_array((), "float64", (1.0, 100.0))
    return (assert_(n > 0.0), assert_(m > 0.0))


# ---------------------------------------------------------------------------
# lsmr.py — guarded normalisations, loop denominators, sqrt domains,
#           Givens rotations, config
# ---------------------------------------------------------------------------


def p18_lsmr_const10():
    """lsmr.py:126 `min_dim > (jnp.iinfo(int_dtype).max / 10)` —
    denominator is the literal 10."""
    return (assert_(jnp.asarray(10.0) != 0.0),)


def p19_lsmr_select_norm_denom():
    """lsmr.py:142/153/219/223 `u = (ω(u) / lax.select(beta == 0.0, 1.0,
    beta).astype(dtype)).ω` — guarded denominator, beta = self.norm(u)
    (default two_norm). One pose for the four identical constructions."""
    u = any_array((N,), "float64", GEN)
    beta = two_norm(u)
    denom = lax.select(beta == 0.0, jnp.asarray(1.0), beta)
    return (assert_(denom != 0.0),)


def p20_lsmr_loop_scalar_denoms():
    """lsmr.py:263 `/(rhoold * rhobarold)`, 268 `/(st["rho"] *
    st["rhobar"])`, 271 `/st["rho"]`, 296 `/st["rhodold"]`, 312-314
    `/jnp.minimum(st["minrbar"], rhotemp)` — loop-carried scalar
    denominators declared at loop entry (generic box; init values are
    1.0 / finfo.max per lines 174-191)."""
    rho = any_array((), "float64", GEN)
    rhobar = any_array((), "float64", GEN)
    rhoold = any_array((), "float64", GEN)
    rhobarold = any_array((), "float64", GEN)
    rhodold = any_array((), "float64", GEN)
    minrbar = any_array((), "float64", GEN)
    rhotemp = any_array((), "float64", GEN)
    return (
        assert_(rhoold * rhobarold != 0.0),  # line 263
        assert_(rho * rhobar != 0.0),  # line 268
        assert_(rho != 0.0),  # line 271
        assert_(rhodold != 0.0),  # line 296
        assert_(jnp.minimum(minrbar, rhotemp) != 0.0),  # line 312-314
    )


def p21_lsmr_givens_r_faithful():
    """lsmr.py:293-295 `... / rhotildeold` where `_, _, rhotildeold =
    self._givens(st["rhodold"], thetabar)` — the code's own _givens
    (lax.cond nest, lsmr.py:361-409) transcribed by calling it."""
    a = any_array((), "float64", GEN)
    b = any_array((), "float64", GEN)
    _, _, r = _lsmr._givens(a, b)
    return (assert_(r != 0.0),)


def p22_lsmr_normr_sqrt_documented():
    """lsmr.py:298-300 `st["normr"] = jnp.sqrt(st["delta"] + (st["betad"]
    - taud) ** 2 + st["betadd"] ** 2)` — sqrt domain. delta over its
    initialized-from-zero accumulation range [0, 10] (line 185
    `delta=0.0`, line 297 `delta + betacheck**2`); others generic."""
    delta = any_array((), "float64", (0.0, 10.0))
    betad = any_array((), "float64", GEN)
    taud = any_array((), "float64", GEN)
    betadd = any_array((), "float64", GEN)
    arg = delta + (betad - taud) ** 2 + betadd**2
    return (assert_(arg >= 0.0),)


def p23_lsmr_normr_sqrt_generic():
    """Same site, delta over the generic sign-unknown box."""
    delta = any_array((), "float64", GEN)
    betad = any_array((), "float64", GEN)
    taud = any_array((), "float64", GEN)
    betadd = any_array((), "float64", GEN)
    arg = delta + (betad - taud) ** 2 + betadd**2
    return (assert_(arg >= 0.0),)


def p24_lsmr_normA2_sqrt():
    """lsmr.py:303-304/339 `st["normA2"] = st["normA2"] + st["beta"]**2;
    normA = jnp.sqrt(st["normA2"])` — sqrt domain. Two poses: normA2
    over documented accumulation range [0, 10] (init `normA2=alpha**2`,
    line 188) and over the generic box."""
    norma2_doc = any_array((), "float64", (0.0, 10.0))
    norma2_gen = any_array((), "float64", GEN)
    beta = any_array((), "float64", GEN)
    return (
        assert_(norma2_doc + beta**2 >= 0.0),
        assert_(norma2_gen + beta**2 >= 0.0),
    )


def p25_lsmr_givens_tau_sqrt_domain():
    """lsmr.py:390-391 `tau = a / lax.select(b == 0.0, 1.0, b);
    s = jnp.sign(b) / jnp.sqrt(1.0 + tau**2)` — sqrt domain
    1 + tau**2 >= 0, with tau's own construction."""
    a = any_array((), "float64", GEN)
    b = any_array((), "float64", GEN)
    tau = a / lax.select(b == 0.0, jnp.asarray(1.0), b)
    return (assert_(1.0 + tau**2 >= 0.0),)


def p26_lsmr_givens_sqrt_denom():
    """lsmr.py:391/398 — denominator jnp.sqrt(1.0 + tau**2) != 0."""
    a = any_array((), "float64", GEN)
    b = any_array((), "float64", GEN)
    tau = a / lax.select(b == 0.0, jnp.asarray(1.0), b)
    return (assert_(jnp.sqrt(1.0 + tau**2) != 0.0),)


def p27_lsmr_select_guard():
    """lsmr.py:390/397 `tau = a / lax.select(b == 0.0, 1.0, b)` — the
    guarded denominator lax.select(b == 0.0, 1.0, b) != 0."""
    b = any_array((), "float64", GEN)
    denom = lax.select(b == 0.0, jnp.asarray(1.0), b)
    return (assert_(denom != 0.0),)


def p28_lsmr_givens_r_denom_chain():
    """lsmr.py:391-393 `s = jnp.sign(b) / jnp.sqrt(1.0 + tau**2);
    r = b / lax.select(s == 0.0, 1.0, s)` — denominator with s's full
    construction (mirrors 398-400 for c)."""
    a = any_array((), "float64", GEN)
    b = any_array((), "float64", GEN)
    tau = a / lax.select(b == 0.0, jnp.asarray(1.0), b)
    s = jnp.sign(b) / jnp.sqrt(1.0 + tau**2)
    denom = lax.select(s == 0.0, jnp.asarray(1.0), s)
    return (assert_(denom != 0.0),)


def p29_lsmr_conlim():
    """lsmr.py:74/325 `conlim: float = 1e8`; `st["condA"] > self.conlim`
    — conlim = 0 halts immediately (condA >= 1): admissible range needs
    conlim > 0. Documented range [0, 1e12] (docstring + non-negativity
    check, lsmr.py:81-82)."""
    c = any_array((), "float64", (0.0, 1.0e12))
    return (assert_(c > 0.0),)


def p30_lsmr_conlim_generic():
    """Same site, generic box."""
    c = any_array((), "float64", GEN)
    return (assert_(c > 0.0),)


def p31_lsmr_damp():
    """lsmr.py:104 `damp = 0.0` — hard-coded damping factor ('damp is
    not supported at this time'); admissibility damp >= 0 posed on the
    constant."""
    return (assert_(jnp.asarray(0.0) >= 0.0),)


# ---------------------------------------------------------------------------
# svd.py / diagonal.py — masked reciprocals, diagonal data, tags
# ---------------------------------------------------------------------------


def _svd_safe_s(s):
    """svd.py:66-74 construction with rcond=None default, n = m = N."""
    rcond = resolve_rcond(None, N, N, s.dtype)
    rcond = jnp.array(rcond, dtype=s.dtype)
    rcond = rcond * s[0]  # s.size > 0
    mask = s > rcond
    safe_s = jnp.where(mask, s, 1)
    return safe_s


def p32_svd_safe_s_generic():
    """svd.py:73-74 `safe_s = jnp.where(mask, s, 1); s_inv = jnp.where(
    mask, jnp.array(1.0) / safe_s, 0)` — denominator safe_s != 0.
    Singular values declared over the generic sign-unknown box."""
    s = any_array((N,), "float64", GEN)
    return (assert_(_svd_safe_s(s) != 0.0),)


def p33_svd_safe_s_documented():
    """Same site, s over [0, 10] — the factorization's documented
    property (singular values are nonnegative)."""
    s = any_array((N,), "float64", (0.0, 10.0))
    return (assert_(_svd_safe_s(s) != 0.0),)


def p34_diagonal_wellposed_diag():
    """diagonal.py:81 `solution = vector / diag` with well_posed=True —
    no guard: every diagonal entry assumed nonzero (class 4: diagonal
    assumed nonzero by a diagonal solve)."""
    diag = any_array((N,), "float64", GEN)
    return (assert_(diag != 0.0),)


def p35_diagonal_guarded():
    """diagonal.py:76-81 well_posed=False path: `rcond = resolve_rcond(
    self.rcond, size, size, diag.dtype); abs_diag = jnp.abs(diag);
    diag = jnp.where(abs_diag > rcond * jnp.max(abs_diag), diag,
    jnp.inf)` then `solution = vector / diag` — guarded denominator."""
    diag = any_array((N,), "float64", GEN)
    rcond = resolve_rcond(None, N, N, diag.dtype)
    abs_diag = jnp.abs(diag)
    diag_guarded = jnp.where(abs_diag > rcond * jnp.max(abs_diag), diag, jnp.inf)
    return (assert_(diag_guarded != 0.0),)


def p36_diagonal_unit_tag():
    """diagonal.py:61-62 `if has_unit_diagonal(operator): return None,
    ...` — unit-diagonal tag honoured (solve returns vector unchanged),
    never checked. The assumed data property: diag == 1 everywhere.
    Direct harness (equality obligation; templates don't fit)."""
    diag = any_array((N,), "float64", GEN)
    return (assert_(diag == 1.0),)


# ---------------------------------------------------------------------------
# triangular.py — diagonal data the back-substitution divides by
# ---------------------------------------------------------------------------


def p37_triangular_diag_nonzero():
    """triangular.py:81-83 `jsp.linalg.solve_triangular(matrix, vector,
    ..., unit_diagonal=unit_diagonal)` — back-substitution divides by
    the stored diagonal entries (unit_diagonal=False path). Input-side
    precondition: diag(matrix) != 0 elementwise. The code has no
    extraction construction (LAPACK consumes the matrix); the transform
    is plain diagonal selection of the declared matrix (disclosed)."""
    mat = any_array((N, N), "float64", GEN)
    d = jnp.diagonal(mat)
    return (assert_(d != 0.0),)


def p38_triangular_unit_tag():
    """triangular.py:63/82 — unit_diagonal tag honoured (LAPACK ignores
    the stored diagonal), never checked: assumed property diag == 1."""
    mat = any_array((N, N), "float64", GEN)
    return (assert_(jnp.diagonal(mat) == 1.0),)


# ---------------------------------------------------------------------------
# cholesky.py — definiteness tags honoured, never checked
# ---------------------------------------------------------------------------


def p39_cholesky_psd():
    """cholesky.py:45-50/60 — PSD tag admits `cho_factor(matrix)`;
    docstring: 'The operator must be square, nonsingular, and either
    positive or negative definite.' Quadratic form x·(A@x): > 0
    (definite, what the factorization needs) and >= 0 (the tag's
    literal semidefinite claim)."""
    a_mat = any_array((N, N), "float64", GEN)
    x = any_array((N,), "float64", GEN)
    ax = jnp.matmul(a_mat, x, precision=lax.Precision.HIGHEST)
    form = tree_dot(x, ax)
    return (assert_(form > 0.0), assert_(form >= 0.0))


def p40_cholesky_nsd():
    """cholesky.py:58-59 `if is_nsd: matrix = -matrix` then cho_factor —
    the NSD branch's assumed property: x·((-A)@x) > 0, with the code's
    own negation as the transform."""
    a_mat = any_array((N, N), "float64", GEN)
    x = any_array((N,), "float64", GEN)
    negax = jnp.matmul(-a_mat, x, precision=lax.Precision.HIGHEST)
    form = tree_dot(x, negax)
    return (assert_(form > 0.0),)


# ---------------------------------------------------------------------------
# _norm.py — sqrt domain, jvp denominator, rms denominator
# ---------------------------------------------------------------------------


def p41_two_norm_sqrt_domain():
    """_norm.py:82 `return jnp.sqrt(sum_squares(x))` — sqrt domain:
    sum_squares(x) = tree_dot(x, x).real >= 0, the code's own
    construction (_norm.py:56)."""
    x = any_array((N,), "float64", GEN)
    return (assert_(sum_squares(x) >= 0.0),)


def p42_two_norm_jvp_denom():
    """_norm.py:91-98 `pred = (out == 0) | jnp.isinf(out); denominator =
    jnp.where(pred, 1, out); div = (x**ω / denominator).ω` — guarded
    denominator, out = two_norm(x)."""
    x = any_array((N,), "float64", GEN)
    out = two_norm(x)
    pred = (out == 0) | jnp.isinf(out)
    denominator = jnp.where(pred, 1, out)
    return (assert_(denominator != 0.0),)


def p43_rms_norm_denom():
    """_norm.py:120 `return two_norm(x) / math.sqrt(size)` — denominator
    is the Python constant math.sqrt(size), size >= 1 on this path
    (size == 0 returns early at line 113-118); posed at size = 4."""
    return (assert_(jnp.asarray(math.sqrt(N)) != 0.0),)


# ---------------------------------------------------------------------------
# _misc.py — resolve_rcond (rcond config for Diagonal/SVD/CG)
# ---------------------------------------------------------------------------


def p44_resolve_rcond_default():
    """_misc.py:36 `return 2 * jnp.finfo(dtype).eps * max(n, m)` — the
    rcond=None default (Diagonal.rcond/SVD.rcond default None;
    cg.py:131). Cutoff positivity posed on the constant (n = m = 4)."""
    val = resolve_rcond(None, N, N, jnp.float64)
    return (assert_(jnp.asarray(val) > 0.0),)


def p45_resolve_rcond_user_generic():
    """_misc.py:38 `return jnp.where(rcond < 0, jnp.finfo(dtype).eps,
    rcond)` — user-supplied rcond path; the cutoff the guards need
    positive. Generic box."""
    rcond = any_array((), "float64", GEN)
    out = resolve_rcond(rcond, N, N, jnp.float64)
    return (assert_(out > 0.0),)


def p46_resolve_rcond_user_documented():
    """Same site, rcond over [0, 1] (docstring: 'the cutoff for handling
    zero entries... Defaults to machine precision times N')."""
    rcond = any_array((), "float64", (0.0, 1.0))
    out = resolve_rcond(rcond, N, N, jnp.float64)
    return (assert_(out > 0.0),)


# ---------------------------------------------------------------------------
# _operator.py — DivLinearOperator scalar
# ---------------------------------------------------------------------------


def p47_div_operator_scalar():
    """_operator.py:224-228 (`__truediv__`) and 1112/1115/1118/1969/2043/
    2384 — every DivLinearOperator path divides by `self.scalar`,
    unchecked: scalar != 0 over the generic box (no documented range)."""
    _, ob = scalar_nonzero("float64", GEN)
    return (ob,)


# ---------------------------------------------------------------------------
# shared config-scalar poses (tolerances, max_steps)
# ---------------------------------------------------------------------------


def p48_tolerances_nonneg_generic():
    """bicgstab.py:57-61 (=cg.py:75-79, gmres.py:65-69, lsmr.py:76-82)
    'Tolerances must be non-negative.' — enforced only for Python
    int/float configs (`isinstance(self.rtol, (int, float))`);
    array-typed tolerances bypass the check. Admissible-range
    precondition atol >= 0, rtol >= 0 over the generic box."""
    atol = any_array((), "float64", GEN)
    rtol = any_array((), "float64", GEN)
    return (assert_(atol >= 0.0), assert_(rtol >= 0.0))


def p49_tolerances_nonneg_documented():
    """Same sites over the enforced range [0, 10]."""
    atol = any_array((), "float64", (0.0, 10.0))
    rtol = any_array((), "float64", (0.0, 10.0))
    return (assert_(atol >= 0.0), assert_(rtol >= 0.0))


def p50_max_steps_positive():
    """bicgstab.py:55, cg.py:73, gmres.py:61, lsmr.py:73 `max_steps: int
    | None = None` — when an int is given, the iteration budget needs
    max_steps >= 1 (0 halts at once as failure). Float relaxation
    (disclosed), generic box + documented [1, 1e6]."""
    n = any_array((), "float64", (-1.0e6, 1.0e6))
    m = any_array((), "float64", (1.0, 1.0e6))
    return (assert_(n > 0.0), assert_(m > 0.0))


def p51_triangular_diag_nonzero_direct():
    """triangular.py:81-83 again — the same input-side property with the
    diagonal entries declared directly as the bounded input (declaration
    form, after the jnp.diagonal extraction form (p37) traced into
    iota/gather and the tool declined; both forms reported)."""
    d = any_array((N,), "float64", GEN)
    return (assert_(d != 0.0),)


def p52_triangular_unit_tag_direct():
    """triangular.py:63/82 again — unit-diagonal tag property, diagonal
    entries declared directly (declaration form of p38)."""
    d = any_array((N,), "float64", GEN)
    return (assert_(d == 1.0),)


POSES = [
    p01_scale_denom_documented,
    p02_scale_denom_generic,
    p03_bicgstab_rho,
    p04_bicgstab_omega,
    p05_bicgstab_alpha_denom,
    p06_bicgstab_tdott,
    p07_cg_inner_prod,
    p08_cg_gamma_prev,
    p09_cg_stabilise_every_int,
    p10_cg_stabilise_every_float,
    p11_cg_stabilise_every_documented,
    p12_cg_operator_definite,
    p13_cg_preconditioner_definite,
    p14_gmres_safe_norm,
    p15_gmres_restart_int,
    p16_gmres_restart_float,
    p17_gmres_stagnation_iters,
    p18_lsmr_const10,
    p19_lsmr_select_norm_denom,
    p20_lsmr_loop_scalar_denoms,
    p21_lsmr_givens_r_faithful,
    p22_lsmr_normr_sqrt_documented,
    p23_lsmr_normr_sqrt_generic,
    p24_lsmr_normA2_sqrt,
    p25_lsmr_givens_tau_sqrt_domain,
    p26_lsmr_givens_sqrt_denom,
    p27_lsmr_select_guard,
    p28_lsmr_givens_r_denom_chain,
    p29_lsmr_conlim,
    p30_lsmr_conlim_generic,
    p31_lsmr_damp,
    p32_svd_safe_s_generic,
    p33_svd_safe_s_documented,
    p34_diagonal_wellposed_diag,
    p35_diagonal_guarded,
    p36_diagonal_unit_tag,
    p37_triangular_diag_nonzero,
    p38_triangular_unit_tag,
    p39_cholesky_psd,
    p40_cholesky_nsd,
    p41_two_norm_sqrt_domain,
    p42_two_norm_jvp_denom,
    p43_rms_norm_denom,
    p44_resolve_rcond_default,
    p45_resolve_rcond_user_generic,
    p46_resolve_rcond_user_documented,
    p47_div_operator_scalar,
    p48_tolerances_nonneg_generic,
    p49_tolerances_nonneg_documented,
    p50_max_steps_positive,
    p51_triangular_diag_nonzero_direct,
    p52_triangular_unit_tag_direct,
]


def main():
    for fn in POSES:
        name = fn.__name__
        print(f"\n{'=' * 72}\n### {name}\n{fn.__doc__}\n{'-' * 72}")
        try:
            v = check(fn)
        except Exception as e:  # a tool refusal is a result to quote
            print(f"[interval] EXCEPTION: {type(e).__name__}: {e}")
            continue
        print("[interval]")
        print(v.render())
        if v.status == "UNKNOWN":
            try:
                v2 = check(fn, solver_timeout_ms=20_000)
            except Exception as e:
                print(f"[escalated 20000ms] EXCEPTION: {type(e).__name__}: {e}")
                continue
            print("[escalated 20000ms]")
            print(v2.render())


if __name__ == "__main__":
    main()
