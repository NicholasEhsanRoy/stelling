# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Dense-linear-algebra census over optimistix/lineax core solve paths.

Measures how central dense linear algebra is in the two solver libraries:
every harness targets a core solve path (a linear solve, a least-squares
fit, a root-find, a minimisation), is traced whole with ``jax.make_jaxpr``,
transcribed via ``stelling._jax_compat.transcribe``, and counted at every
depth by ``stelling.census``. The report is printed to stdout; nothing is
written.

The LA primitive set (fixed up front): ``lu``, ``custom_linear_solve``,
``triangular_solve``, ``cholesky``, ``qr``, ``eig``, ``eigh``, ``svd``,
plus any ``*_solve`` variant encountered (this catches lineax's
library-defined ``linear_solve`` primitive and jax's
``tridiagonal_solve``), plus — recorded judgment — the jax 0.11 QR-family
lowering names ``geqrf`` / ``householder_product`` / ``ormqr``, which are
what ``jnp.linalg.qr(mode="raw")`` and lineax's Q-application actually
trace to; excluding them would erase the QR factorization from the count.
``dot_general`` is a SECONDARY row: contraction, not factorization —
counted separately, never mixed into LA totals. ``lu_pivots_to_permutation``
(pivot bookkeeping on the LU solve path) is likewise reported adjacent,
outside the LA totals.

Surface vs deep: lineax's ``linear_solve`` is an equinox-defined opaque
primitive — ``solver.init`` (the factorization) runs in traced Python and
is visible, but ``solver.compute`` (the back-substitution) runs inside the
primitive's impl and carries NO sub-jaxpr, so surface traces cannot see
it. Each ``*-deep/*`` harness therefore traces the concrete solver's
``init`` + ``compute`` methods directly, making the hidden solve path
visible. Every harness is tagged ``surface`` or ``deep`` in the report.

Transcription deviation (recorded per harness): the optimistix paths that
solve against a ``FunctionLinearOperator`` carry, in the ``linear_solve``
equation's ``static`` param, a closure-converted ``ClosedJaxpr`` of the
linearised operator whose ``.consts`` slots still hold leaked
``DynamicJaxprTracer`` objects (equinox passes the live values separately
as the primitive's invars; the static copy's consts are unreachable
metadata). ``stelling._jax_compat.transcribe`` correctly refuses to
materialise a tracer, so this file subclasses the transcriber to record
each such const as a sentinel — stelling core is untouched, and the
number of sanitised consts is reported for every harness that needed it.

Run inside an environment with stelling, jax, optimistix, and lineax:

    python corpus/la_census.py
"""

from __future__ import annotations

import datetime
import importlib.metadata
import traceback
from collections import Counter

import jax
import jax.numpy as jnp

import stelling
from stelling import _jax_compat, census, ir
from stelling.coverage import DEFAULT_TRANSPARENT, sub_jaxprs

# --- the LA primitive set ----------------------------------------------------

LA_NAMED = frozenset(
    {"lu", "custom_linear_solve", "triangular_solve", "cholesky", "qr", "eig", "eigh", "svd"}
)
# jax 0.11 lowering names for the dense QR family: jnp.linalg.qr(mode="raw")
# binds `geqrf`, and applying/materialising Q binds `ormqr` /
# `householder_product`. Counted as LA (judgment recorded in the report).
LA_QR_FAMILY = frozenset({"geqrf", "householder_product", "ormqr"})
# LA-adjacent bookkeeping, reported outside the LA totals.
LA_ADJACENT = frozenset({"lu_pivots_to_permutation"})


def la_class(name: str) -> str | None:
    """'la' | 'dot_general' | 'adjacent' | None for every primitive name."""
    if name == "dot_general":
        return "dot_general"
    if name in LA_ADJACENT:
        return "adjacent"
    if name in LA_NAMED or name in LA_QR_FAMILY or name.endswith("_solve"):
        return "la"
    return None


# --- concrete problem data (built outside the traces: setup is not census) --

_DENSE = jnp.array(
    [
        [3.0, 1.0, 0.5, 0.2],
        [0.7, 2.5, 0.4, 0.1],
        [0.2, 0.6, 2.8, 0.3],
        [0.1, 0.4, 0.9, 3.2],
    ]
)
_DENSE_RHS = jnp.array([1.0, 2.0, 3.0, 4.0])

_RECT = jnp.array(
    [
        [1.0, 0.4, 0.1],
        [0.3, 1.2, 0.2],
        [0.2, 0.5, 1.4],
        [0.8, 0.1, 0.6],
        [0.1, 0.9, 0.3],
        [0.5, 0.2, 1.1],
    ]
)
_RECT_RHS = jnp.array([1.0, 0.5, 2.0, 1.5, 0.2, 0.8])

_B = jnp.array(
    [
        [1.0, 0.2, 0.1, 0.0],
        [0.3, 1.1, 0.2, 0.1],
        [0.1, 0.4, 0.9, 0.3],
        [0.2, 0.1, 0.5, 1.2],
    ]
)
_SPD = _B @ _B.T + jnp.eye(4)  # concrete SPD 4x4, built before any trace
_SPD_RHS = jnp.array([0.5, 1.0, 1.5, 2.0])


# --- lineax surface harnesses: lx.linear_solve, internals behind the ---------
# --- opaque `linear_solve` primitive (only solver.init is visible) -----------


def _lineax_surface(operator_fn, matrix, rhs, solver=None):
    import lineax as lx

    def solve(matrix, rhs):
        operator = operator_fn(matrix)
        if solver is None:
            return lx.linear_solve(operator, rhs).value
        return lx.linear_solve(operator, rhs, solver).value

    picked = (
        type(solver).__name__
        if solver is not None
        else type(
            lx.AutoLinearSolver(well_posed=True).select_solver(operator_fn(matrix))
        ).__name__
    )
    return picked, jax.make_jaxpr(solve)(matrix, rhs)


def harness_lineax_dense_auto():
    import lineax as lx

    picked, cj = _lineax_surface(lx.MatrixLinearOperator, _DENSE, _DENSE_RHS)
    return (
        f"lx.linear_solve, dense square 4x4, default AutoLinearSolver(well_posed=True) -> {picked}",
        "surface",
        cj,
    )


def harness_lineax_spd_auto():
    import lineax as lx

    def op(matrix):
        return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)

    picked, cj = _lineax_surface(op, _SPD, _SPD_RHS)
    return (
        f"lx.linear_solve, SPD-tagged 4x4, default AutoLinearSolver(well_posed=True) -> {picked}",
        "surface",
        cj,
    )


def harness_lineax_lstsq_auto():
    import lineax as lx

    solver = lx.AutoLinearSolver(well_posed=False)
    picked = type(solver.select_solver(lx.MatrixLinearOperator(_RECT))).__name__

    def solve(matrix, rhs):
        return lx.linear_solve(lx.MatrixLinearOperator(matrix), rhs, solver).value

    return (
        f"lx.linear_solve, rectangular 6x3 least squares, AutoLinearSolver(well_posed=False) -> {picked}",
        "surface",
        jax.make_jaxpr(solve)(_RECT, _RECT_RHS),
    )


def harness_lineax_dense_lu():
    import lineax as lx

    _, cj = _lineax_surface(lx.MatrixLinearOperator, _DENSE, _DENSE_RHS, lx.LU())
    return ("lx.linear_solve, dense square 4x4, explicit lx.LU()", "surface", cj)


def harness_lineax_lstsq_qr():
    import lineax as lx

    _, cj = _lineax_surface(lx.MatrixLinearOperator, _RECT, _RECT_RHS, lx.QR())
    return ("lx.linear_solve, rectangular 6x3 least squares, explicit lx.QR()", "surface", cj)


def harness_lineax_spd_cholesky():
    import lineax as lx

    def op(matrix):
        return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)

    _, cj = _lineax_surface(op, _SPD, _SPD_RHS, lx.Cholesky())
    return ("lx.linear_solve, SPD-tagged 4x4, explicit lx.Cholesky()", "surface", cj)


# --- lineax deep harnesses: the concrete solver's init + compute traced ------
# --- directly, bypassing the opaque `linear_solve` primitive -----------------


def _lineax_deep(solver_cls, operator_fn, matrix, rhs):
    def solve(matrix, rhs):
        operator = operator_fn(matrix)
        solver = solver_cls()
        state = solver.init(operator, {})
        solution, result, stats = solver.compute(state, rhs, {})
        return solution

    name = solver_cls.__name__
    return (
        f"{name}.init + {name}.compute traced directly (the path hidden inside `linear_solve`)",
        "deep",
        jax.make_jaxpr(solve)(matrix, rhs),
    )


def harness_lineax_deep_lu():
    import lineax as lx

    d, level, cj = _lineax_deep(lx.LU, lx.MatrixLinearOperator, _DENSE, _DENSE_RHS)
    return (f"dense square 4x4: {d}", level, cj)


def harness_lineax_deep_qr():
    import lineax as lx

    d, level, cj = _lineax_deep(lx.QR, lx.MatrixLinearOperator, _RECT, _RECT_RHS)
    return (f"rectangular 6x3 least squares: {d}", level, cj)


def harness_lineax_deep_cholesky():
    import lineax as lx

    def op(matrix):
        return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)

    d, level, cj = _lineax_deep(lx.Cholesky, op, _SPD, _SPD_RHS)
    return (f"SPD-tagged 4x4: {d}", level, cj)


def harness_lineax_deep_svd():
    import lineax as lx

    d, level, cj = _lineax_deep(lx.SVD, lx.MatrixLinearOperator, _RECT, _RECT_RHS)
    return (f"rectangular 6x3 least squares: {d}", level, cj)


# --- optimistix surface harnesses: the three core solve paths ----------------

_T_GRID = jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
_DECAY_DATA = jnp.array([2.0, 1.4, 0.95, 0.68, 0.47, 0.33])  # roughly 2*exp(-t)


def _decay_residual(y, args):
    scale, rate = y
    return scale * jnp.exp(-rate * _T_GRID) - _DECAY_DATA


def harness_optx_gauss_newton():
    import optimistix as optx

    def solve(y0):
        sol = optx.least_squares(
            _decay_residual,
            optx.GaussNewton(rtol=1e-8, atol=1e-8),
            y0,
            max_steps=32,
            throw=False,
        )
        return sol.value

    return (
        "optx.least_squares, 2-parameter exponential-decay fit (6 residuals), Gauss-Newton",
        "surface",
        jax.make_jaxpr(solve)(jnp.array([1.0, 0.5])),
    )


def harness_optx_levenberg_marquardt():
    import optimistix as optx

    def solve(y0):
        sol = optx.least_squares(
            _decay_residual,
            optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8),
            y0,
            max_steps=32,
            throw=False,
        )
        return sol.value

    return (
        "optx.least_squares, 2-parameter exponential-decay fit (6 residuals), Levenberg-Marquardt",
        "surface",
        jax.make_jaxpr(solve)(jnp.array([1.0, 0.5])),
    )


def harness_optx_newton_root():
    import optimistix as optx

    def system(y, args):
        return jnp.stack(
            [
                y[0] + 0.5 * y[1] ** 2 - 1.2,
                y[1] * y[2] + 0.3 * y[0] - 0.8,
                y[2] ** 2 + y[0] - 1.5,
            ]
        )

    def solve(y0):
        sol = optx.root_find(
            system,
            optx.Newton(rtol=1e-8, atol=1e-8),
            y0,
            max_steps=32,
            throw=False,
        )
        return sol.value

    return (
        "optx.root_find, 3-equation nonlinear system, Newton",
        "surface",
        jax.make_jaxpr(solve)(jnp.array([1.0, 1.0, 1.0])),
    )


def harness_optx_bfgs_minimise():
    import optimistix as optx

    def rosenbrock(y, args):
        return jnp.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2)

    def solve(y0):
        sol = optx.minimise(
            rosenbrock,
            optx.BFGS(rtol=1e-6, atol=1e-6),
            y0,
            max_steps=32,
            throw=False,
        )
        return sol.value

    return (
        "optx.minimise, 4-D Rosenbrock, BFGS",
        "surface",
        jax.make_jaxpr(solve)(jnp.zeros(4)),
    )


HARNESSES = {
    "lineax/dense-auto": harness_lineax_dense_auto,
    "lineax/spd-auto": harness_lineax_spd_auto,
    "lineax/lstsq-auto": harness_lineax_lstsq_auto,
    "lineax/dense-lu": harness_lineax_dense_lu,
    "lineax/lstsq-qr": harness_lineax_lstsq_qr,
    "lineax/spd-chol": harness_lineax_spd_cholesky,
    "lineax-deep/lu": harness_lineax_deep_lu,
    "lineax-deep/qr": harness_lineax_deep_qr,
    "lineax-deep/chol": harness_lineax_deep_cholesky,
    "lineax-deep/svd": harness_lineax_deep_svd,
    "optimistix/gn-lstsq": harness_optx_gauss_newton,
    "optimistix/lm-lstsq": harness_optx_levenberg_marquardt,
    "optimistix/newton-root": harness_optx_newton_root,
    "optimistix/bfgs-min": harness_optx_bfgs_minimise,
}


# --- depth-context walk: census taxonomy + human-readable chain --------------


def eqn_sites(closed: ir.ClosedJaxpr, transparent: frozenset[str] = DEFAULT_TRANSPARENT):
    """Yield (context, chain, eqn) for every equation at every depth.

    ``context`` follows stelling.census exactly: ``top`` at depth 0,
    ``transparent`` while only transparent wrappers have been crossed,
    ``nested`` beneath any non-transparent structured primitive. ``chain``
    is an interrogate-style human-readable path of owning equations.
    """
    stack: list[tuple[ir.Jaxpr, str, str]] = [(closed.jaxpr, "top", "top")]
    while stack:
        jaxpr, context, chain = stack.pop()
        for eqn in jaxpr.eqns:
            yield context, chain, eqn
            child = (
                "transparent"
                if context != "nested" and eqn.primitive in transparent
                else "nested"
            )
            label = eqn.primitive
            if eqn.primitive == "jit":
                label = f"jit[{eqn.params_dict().get('name', '?')}]"
            for sub in sub_jaxprs(eqn):
                stack.append((sub, child, f"{chain} > {label}"))


_JAX_PLUMBING = (
    "source_info_util.py",
    "partial_eval.py",
    "/core.py",
    "traceback_util.py",
    "/pjit.py",
    "/api.py",
    "dispatch.py",
)


def _shorten(frame: str) -> str:
    path, _, rest = frame.partition(" ")
    return f"{'/'.join(path.split('/')[-2:])} {rest}"


def _library_frame(eqn: ir.JaxprEqn) -> str:
    """Attribute the eqn to the function that bound it.

    Frames are innermost-first. Preference: innermost lineax/optimistix
    frame, then equinox; inside jax's own jit wrappers the traceback is
    truncated to jax-internal frames, so fall back to the innermost
    non-plumbing jax frame, marked ``jax:`` (the enclosing chain still
    names the jit's entry function).
    """
    for libs in (("/lineax/", "/optimistix/"), ("/equinox/",)):
        for frame in eqn.source_info:
            if any(lib in frame for lib in libs):
                return _shorten(frame)
    for frame in eqn.source_info:
        if "/jax/" in frame and not any(p in frame for p in _JAX_PLUMBING):
            return f"jax: {_shorten(frame)}"
    return "(no attributable frame)"


class _LeakedTracerTolerantTranscriber(_jax_compat._Transcriber):
    """Transcriber that tolerates leaked-tracer consts in static metadata.

    Only ever reached for consts of the closure-converted operator jaxpr
    stored in ``linear_solve``'s ``static`` param (see module docstring);
    every such const is recorded as an ``ir.SentinelParam`` and counted,
    never silently dropped.
    """

    def __init__(self) -> None:
        super().__init__()
        self.leaked_tracer_consts = 0

    def value(self, v):
        if type(v).__name__.endswith("Tracer"):
            self.leaked_tracer_consts += 1
            return ir.SentinelParam(cls=f"leaked:{type(v).__name__}")
        return super().value(v)


def transcribe_tolerant(closed_jaxpr) -> tuple[ir.ClosedJaxpr, int]:
    """``_jax_compat.transcribe`` plus the leaked-tracer-const workaround."""
    transcriber = _LeakedTracerTolerantTranscriber()
    out = transcriber.closed_jaxpr(closed_jaxpr)
    return out, transcriber.leaked_tracer_consts


def _version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    print("# Dense-LA census — optimistix / lineax core solve paths")
    print()
    print(f"Run {datetime.date.today().isoformat()}, jax {jax.__version__} "
          f"(jaxlib {_version('jaxlib')}), stelling {stelling.__version__}, "
          f"optimistix {_version('optimistix')}, lineax {_version('lineax')}, "
          f"equinox {_version('equinox')}.")
    print()

    accumulator = census.CensusAccumulator()
    per_harness: dict[str, dict] = {}

    for label, harness in HARNESSES.items():
        try:
            description, level, closed_jaxpr = harness()
            leaked = 0
            try:
                transcribed = _jax_compat.transcribe(closed_jaxpr)
            except Exception as exc:
                if "Tracer" not in type(exc).__name__:
                    raise
                transcribed, leaked = transcribe_tolerant(closed_jaxpr)
            solo = census.CensusAccumulator()
            solo.add(label, transcribed)
            accumulator.add(label, transcribed)
            sites = [
                (context, chain, eqn)
                for context, chain, eqn in eqn_sites(transcribed)
                if la_class(eqn.primitive) is not None
            ]
            per_harness[label] = {
                "description": description,
                "level": level,
                "total": solo.freeze().total,
                "sites": sites,
                "leaked": leaked,
            }
            note = f"  [{leaked} leaked-tracer consts sanitised]" if leaked else ""
            print(f"[ok]   {label:<22} {level:<7} {solo.freeze().total:>5} eqns — {description}{note}")
        except Exception as exc:  # a failing harness is a finding, not an abort
            reason = f"{type(exc).__name__}: {exc}"
            per_harness[label] = {"description": f"FAILED — {reason.splitlines()[0][:160]}",
                                  "level": "-", "total": 0, "sites": []}
            print(f"[fail] {label:<22} {reason.splitlines()[0][:120]}")
            traceback.print_exc()

    result = accumulator.freeze()
    n = len(result.targets)

    print()
    print(f"## Census table — {result.total} equations, {len(result.primitives)} "
          f"distinct primitives, {n} solve paths")
    print()
    print(result.markdown_table())
    print()

    # -- per-harness totals and LA density -----------------------------------
    print("## Per-harness totals and LA density (dot_general separate, never in LA)")
    print()
    print("| harness | level | total eqns | LA eqns | LA density | dot_general | adjacent |")
    print("|---|---|---|---|---|---|---|")
    for label, entry in per_harness.items():
        kinds = Counter(la_class(eqn.primitive) for _, _, eqn in entry["sites"])
        la, dg, adj = kinds.get("la", 0), kinds.get("dot_general", 0), kinds.get("adjacent", 0)
        total = entry["total"]
        density = f"{la / total:.1%}" if total else "-"
        print(f"| {label} | {entry['level']} | {total} | {la} | {density} | {dg} | {adj} |")
    print()

    # -- breadth of each LA primitive across solve paths ---------------------
    print("## LA-primitive breadth (solve paths containing it, of "
          f"{n} traced)")
    print()
    for p in result.primitives:
        if la_class(p.name) == "la":
            print(f"- `{p.name}`: count {p.count}, breadth {p.breadth}/{n}")
    for p in result.primitives:
        if la_class(p.name) == "dot_general":
            print(f"- `{p.name}` (secondary — contraction, not factorization): "
                  f"count {p.count}, breadth {p.breadth}/{n}")
    for p in result.primitives:
        if la_class(p.name) == "adjacent":
            print(f"- `{p.name}` (adjacent bookkeeping, outside LA totals): "
                  f"count {p.count}, breadth {p.breadth}/{n}")
    print()

    # -- per-occurrence depth context ----------------------------------------
    print("## LA occurrences with depth context")
    print()
    print("One row per (harness, primitive, context, chain, binding library frame);")
    print("x-counts collapse identical rows. dot_general/adjacent rows included,")
    print("marked, for completeness.")
    print()
    print("| harness | level | primitive | class | context | chain | bound at |")
    print("|---|---|---|---|---|---|---|")
    for label, entry in per_harness.items():
        rows = Counter(
            (eqn.primitive, la_class(eqn.primitive), context, chain, _library_frame(eqn))
            for context, chain, eqn in entry["sites"]
        )
        for (prim, kind, context, chain, frame), count in rows.items():
            mult = f" x{count}" if count > 1 else ""
            print(f"| {label} | {entry['level']} | `{prim}`{mult} | {kind} "
                  f"| {context} | `{chain}` | {frame} |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
