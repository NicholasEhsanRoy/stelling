# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Trace the corpus, transcribe, census, and write design/primitive-census.md.

Each harness targets its library's core computational path — the inner
solver/stepper/sampler loop, not constructors — and traces the top-level
API with ``jax.make_jaxpr``, which captures the entire nested program
(scan/while bodies, jit wrappers) for the census walk. Harness method,
corpus, jax version, and date are recorded in the artifact: an undated
census is a rumour.

Run inside an environment with stelling, jax, and the corpus libraries:

    python corpus/run_census.py
"""

from __future__ import annotations

import datetime
import importlib.metadata
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp

import stelling
from stelling import _jax_compat, census

ARTIFACT = Path(__file__).resolve().parents[1] / "design" / "primitive-census.md"


# --- harnesses: one core computational path per library --------------------


def harness_diffrax():
    import diffrax

    def vector_field(t, y, args):  # Lotka-Volterra
        prey, pred = y[0], y[1]
        return jnp.stack([prey * (1.0 - pred), pred * (prey - 1.0)])

    term = diffrax.ODETerm(vector_field)

    def solve(y0):
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
            saveat=diffrax.SaveAt(t1=True),
        )
        return sol.ys

    return (
        "adaptive Tsit5 solve of Lotka-Volterra via diffeqsolve (PID controller)",
        jax.make_jaxpr(solve)(jnp.array([1.0, 2.0])),
    )


def harness_optimistix():
    import optimistix as optx

    def rosenbrock(y, args):
        return jnp.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2)

    def minimise(y0):
        sol = optx.minimise(
            rosenbrock,
            optx.BFGS(rtol=1e-6, atol=1e-6),
            y0,
            max_steps=64,
            throw=False,
        )
        return sol.value

    return (
        "BFGS minimisation of 4-D Rosenbrock via optx.minimise",
        jax.make_jaxpr(minimise)(jnp.zeros(4)),
    )


def harness_lineax():
    import lineax as lx

    def solve(matrix, rhs):
        return lx.linear_solve(lx.MatrixLinearOperator(matrix), rhs, solver=lx.LU()).value

    return (
        "dense LU linear solve via lx.linear_solve",
        jax.make_jaxpr(solve)(jnp.eye(6) + 0.1, jnp.ones(6)),
    )


def harness_numpyro():
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer.util import log_density

    covariates = jnp.ones((10, 3))
    observed = jnp.ones(10)

    def model(x, y_obs):
        weights = numpyro.sample("w", dist.Normal(0.0, 1.0).expand([3]).to_event(1))
        bias = numpyro.sample("b", dist.Normal(0.0, 1.0))
        numpyro.sample("y", dist.Normal(x @ weights + bias, 1.0), obs=y_obs)

    def grad_log_density(weights, bias):
        value, _ = log_density(model, (covariates, observed), {}, {"w": weights, "b": bias})
        return value

    return (
        "gradient of Bayesian-regression log density (the HMC core computation)",
        jax.make_jaxpr(jax.grad(grad_log_density, argnums=(0, 1)))(jnp.zeros(3), 0.0),
    )


def harness_blackjax():
    import blackjax

    def log_density(position):
        return -0.5 * jnp.sum(position**2)

    hmc = blackjax.hmc(
        log_density,
        step_size=0.1,
        inverse_mass_matrix=jnp.ones(4),
        num_integration_steps=8,
    )

    def step(position, rng_key):
        state = hmc.init(position)
        new_state, _ = hmc.step(rng_key, state)
        return new_state.position

    return (
        "one HMC step (init + leapfrog + accept) on a Gaussian target",
        jax.make_jaxpr(step)(jnp.zeros(4), jax.random.PRNGKey(0)),
    )


def harness_jax_md():
    import jax_md

    displacement, shift = jax_md.space.periodic(10.0)
    energy_fn = jax_md.energy.soft_sphere_pair(displacement)

    def potential(positions):
        return energy_fn(positions)

    return (
        "soft-sphere pair energy over periodic space (pairwise kernel)",
        jax.make_jaxpr(potential)(jnp.zeros((16, 2))),
    )


def harness_jax_md_neighbor():
    # the neighbor-list path — the indexing-heavy core of real MD runs,
    # and the value model's canonical suspect. Traced separately so a
    # failure here doesn't lose the pair-energy census.
    import jax_md

    displacement, shift = jax_md.space.periodic(10.0)
    neighbor_fn, energy_fn = jax_md.energy.soft_sphere_neighbor_list(displacement, 10.0)
    xs = jnp.linspace(0.5, 9.5, 4)
    positions = jnp.stack(jnp.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    neighbors = neighbor_fn.allocate(positions)

    def energy_and_update(new_positions):
        return energy_fn(new_positions, neighbor=neighbors), neighbors.update(new_positions).idx

    return (
        "soft-sphere energy via neighbor list + neighbor update (indexing path)",
        jax.make_jaxpr(energy_and_update)(positions),
    )


def harness_jax_cfd():
    import jax_cfd.base as cfd

    grid = cfd.grids.Grid((16, 16), domain=((0.0, 1.0), (0.0, 1.0)))

    def advect(scalar_data, u_data, v_data):
        scalar = cfd.grids.GridVariable(
            cfd.grids.GridArray(scalar_data, (0.5, 0.5), grid),
            cfd.boundaries.periodic_boundary_conditions(2),
        )
        velocity = tuple(
            cfd.grids.GridVariable(
                cfd.grids.GridArray(data, offset, grid),
                cfd.boundaries.periodic_boundary_conditions(2),
            )
            for data, offset in ((u_data, (1.0, 0.5)), (v_data, (0.5, 1.0)))
        )
        return cfd.advection.advect_van_leer(scalar, velocity, dt=0.01).data

    shape = (16, 16)
    return (
        "van Leer advection of a scalar on a 16x16 staggered grid",
        jax.make_jaxpr(advect)(jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape)),
    )


HARNESSES = {
    "diffrax": [harness_diffrax],
    "optimistix": [harness_optimistix],
    "lineax": [harness_lineax],
    "numpyro": [harness_numpyro],
    "blackjax": [harness_blackjax],
    "jax-md": [harness_jax_md, harness_jax_md_neighbor],
    "jax-cfd": [harness_jax_cfd],
}


def _version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    rows = []  # (target, version, description, eqn count | failure)
    saturation = []  # (target, top-10 before, top-10 after)
    accumulator = census.CensusAccumulator()

    for target, harnesses in HARNESSES.items():
        version = _version(target)
        target_added = False
        for harness in harnesses:
            try:
                description, closed_jaxpr = harness()
                transcribed = _jax_compat.transcribe(closed_jaxpr)
                solo = census.CensusAccumulator()
                solo.add(target, transcribed)
                before = accumulator.freeze().top(10)
                accumulator.add(target, transcribed)
                after = accumulator.freeze().top(10)
                if not target_added:
                    saturation.append((target, before, after))
                    target_added = True
                rows.append((target, version, description, f"{solo.freeze().total}"))
                print(f"[ok]   {target:<11} {solo.freeze().total} eqns — {description[:60]}")
            except Exception as exc:  # a failing harness is a finding, not an abort
                reason = f"{type(exc).__name__}: {exc}"
                rows.append((target, version, "—", f"FAILED — {reason.splitlines()[0][:160]}"))
                print(f"[fail] {target:<11} {reason.splitlines()[0][:120]}")
                traceback.print_exc()

    result = accumulator.freeze()
    reorderings = [
        f"after `{t}`: {'reordered' if b and b != a[: len(b)] else 'stable' if b else 'first entry'}"
        for t, b, a in saturation
    ]
    saturated = bool(saturation) and all(
        b == a[: len(b)] for _, b, a in saturation[1:] if b
    )

    wedge_rows = [p for p in result.primitives if p.wedge]
    lines = [
        "# Primitive census — scientific JAX corpus",
        "",
        f"**Status:** evidence artifact, run {datetime.date.today().isoformat()}.",
        f"jax {jax.__version__}, stelling {stelling.__version__}, harness method:",
        "hand-written minimal harnesses per library, each targeting the core",
        "computational path (solver loop / sampler step / stencil kernel), traced",
        "whole with `jax.make_jaxpr` and counted at every depth by",
        "`stelling.census` (registry-independent). Re-verify trigger: any jax",
        "series bump, or any corpus addition. **An undated census is a rumour;",
        "this one will go stale and says when it was taken.**",
        "",
        "> **This is an inventory, not a value signal.** It makes no claim about",
        "> whether stelling finds bugs, and it is not pre-registered because it",
        "> claims nothing (`design/value-model.md`). The moment a number below is",
        "> cited as evidence that stelling is useful, the census has become the",
        "> thing it isn't. Its two jobs: set the transfer-registry priority",
        "> order, and date-stamp what the ecosystem's code is made of.",
        "",
        "## Corpus",
        "",
        "| target | version | harness (core path) | eqns |",
        "|---|---|---|---|",
    ]
    for target, version, description, count in rows:
        lines.append(f"| {target} | {version} | {description} | {count} |")
    lines += [
        "",
        "Corpus scope: mature maintained libraries only. The research-code arm",
        "(`design/value-model.md`'s range criterion) is **not represented** —",
        "ordinary research scripts are not pip-installable, and tracing them",
        "needs the interception harness method. Recorded as a gap, not solved.",
        "",
        "## Saturation",
        "",
        f"Criterion: add targets until the top-10 ranking stops reordering. "
        f"Status: **{'saturated at this corpus size' if saturated else 'NOT saturated — the ranking was still reordering as targets were added; the registry priority order below is provisional and the corpus must grow'}**.",
        "",
    ]
    lines += [f"- {r}" for r in reorderings]
    lines += [
        "",
        "## Wedge-relevant primitives (gather / scatter\\* / dynamic\\_slice / dynamic\\_update\\_slice)",
        "",
        "> **Read these rows with the harnesses in mind.** Wedge counts are a",
        "> property of the paths the harnesses exercised, not of the libraries:",
        "> paths not traced contribute nothing, and indexing-heavy paths (e.g.",
        "> jax-md neighbor lists) appear only insofar as a harness reaches them.",
        "> A low count here is a fact about this corpus and these harnesses. It",
        "> cannot support \"the bug class is rare\" — the same way no number in",
        "> this file may support \"stelling is useful.\"",
        "",
    ]
    if wedge_rows:
        n = len(result.targets)
        for p in wedge_rows:
            lines.append(
                f"- `{p.name}`: {p.count} eqns across {p.breadth}/{n} targets"
            )
    else:
        lines.append("- none observed in this corpus")
    lines += [
        "",
        f"## Full table — {result.total} equations, {len(result.primitives)} distinct primitives, {len(result.targets)} targets",
        "",
        result.markdown_table(),
        "",
    ]
    ARTIFACT.write_text("\n".join(lines))
    print(f"\nwrote {ARTIFACT}")
    print(f"total: {result.total} eqns, {len(result.primitives)} primitives, saturated: {saturated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
