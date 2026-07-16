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


def harness_maddening():
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")  # disconnected-node warnings
        from maddening.core.graph_manager import GraphManager
        from maddening.nodes.ball import BallNode
        from maddening.nodes.heat import HeatNode
        from maddening.nodes.table import TableNode

        gm = GraphManager()
        gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
        gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0, elasticity=0.7))
        gm.add_node(HeatNode(name="rod", timestep=0.01, n_cells=32, thermal_diffusivity=0.05))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()
        external = gm._default_external_inputs()
        state = gm._state
    return (
        "one coupled multi-physics graph step (table→ball contact + 32-cell "
        "heat stencil) via the compiled scheduler",
        jax.make_jaxpr(lambda s: gm._compiled_step(s, external))(state),
    )


# Held-out arms are censused separately and NEVER vote on the priority
# order: a registry partly ordered by its author's own code is a registry
# built for one user. See the held-out section of the artifact for framing.
HELD_OUT = {"maddening": [harness_maddening]}

# Cost tiers for the schedule section. Membership is a judgement, recorded
# as data so it is auditable; primitives outside every tier are reported
# loudly rather than absorbed.
SCHEDULE_TIERS = [
    ("0 — transparent (already done)", frozenset({"jit", "custom_jvp_call", "custom_vjp_call", "remat2"})),
    ("1a — free: structural/identity", frozenset({
        "copy", "stop_gradient", "broadcast_in_dim", "reshape", "squeeze",
        "slice", "concatenate", "iota", "pad", "stack", "split", "transpose", "rev"})),
    ("1b — free: elementwise/compare/join one-liners", frozenset({
        "add", "add_any", "sub", "mul", "div", "neg", "abs", "min", "max",
        "select_n", "eq", "ne", "lt", "le", "gt", "and", "or", "not", "sign",
        "integer_pow", "sqrt", "rem", "convert_element_type"})),
    ("1c — free: fixed-shape reduction folds", frozenset({
        "reduce_sum", "reduce_max", "reduce_min", "reduce_and", "reduce_or", "cumsum"})),
    ("2 — medium: bilinear/permutation", frozenset({"dot_general", "sort", "pow"})),
    ("3 — wedge targets (the Stage-1 work itself)", frozenset({
        "gather", "scatter", "scatter-add", "dynamic_slice", "dynamic_update_slice"})),
    ("4 — control flow (few eqns gate hundreds nested)", frozenset({"cond", "while", "scan"})),
    ("5 — float-boundary (see the nextafter note)", frozenset({
        "nextafter", "is_finite", "bitcast_convert_type", "exp", "log", "erf_inv"})),
    ("6 — PRNG / bit-level (bounded-adversary story)", frozenset({
        "random_wrap", "random_unwrap", "random_bits", "random_seed",
        "random_split", "random_fold_in", "shift_right_logical"})),
    ("7 — library-defined (open primitive set — design/open-primitive-set.md)", frozenset({
        "select_if_vmap", "unvmap_any", "unvmap_max", "nonbatchable",
        "nondifferentiable_backward", "maybe_set", "linear_solve"})),
    ("8 — escape hatches (⊤ forever)", frozenset({"pure_callback"})),
    ("9 — dense linear algebra (contract-level treatment later)", frozenset({"lu"})),
]


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

    held_rows = []  # (arm, version, description, Census | None)
    for arm, harnesses in HELD_OUT.items():
        arm_version = _version(arm)
        for harness in harnesses:
            try:
                description, closed_jaxpr = harness()
                arm_acc = census.CensusAccumulator()
                arm_acc.add(arm, _jax_compat.transcribe(closed_jaxpr))
                held_rows.append((arm, arm_version, description, arm_acc.freeze()))
                print(f"[held-out ok]   {arm:<11} {arm_acc.freeze().total} eqns")
            except Exception as exc:
                reason = f"FAILED — {type(exc).__name__}: {str(exc).splitlines()[0][:140]}"
                held_rows.append((arm, arm_version, reason, None))
                print(f"[held-out fail] {arm:<11} {reason[:120]}")

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
        "is **not represented** — and growth is re-aimed accordingly: since the",
        "schedule below is ordered by cost rather than count, saturation no",
        "longer drives corpus growth. The reason to grow is that **the research",
        "arm is the unprimed arm**: every current target is a mature library",
        "written by people who know exactly where the gather clamps, and every",
        "harness was written by someone who knew what he hoped to find.",
        "Interception on research code is the only sample with neither",
        "property — a value-model instrument, not a ranking refinement.",
        "",
        "## Saturation",
        "",
        f"Criterion: add targets until the top-10 ranking stops reordering. "
        f"Status: **{'saturated at this corpus size' if saturated else 'NOT saturated — the ranking was still reordering as targets were added; the per-primitive ordering is provisional'}** "
        f"(which constrains the *inventory*, not the schedule — see below).",
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
        ">",
        "> **The symmetry cuts both ways.** The high counts are harness-shaped",
        "> too, and in a worse way: `gather` ×7 exists because we went looking",
        "> for it, knowing the neighbor list was the value model's canonical",
        "> suspect. A census can launder a prior about where bugs live into",
        "> apparent evidence; these rows are where that would happen. Read them",
        "> as *where we looked*, never as *where bugs are*. Interrogation of",
        "> these rows: `design/census-interrogation.md`.",
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
    # -- schedule: cost, not count, orders the work -------------------------
    prim_count = {p.name: p.count for p in result.primitives}
    tier_rows, classified = [], set()
    for title, names in SCHEDULE_TIERS:
        eqns = sum(prim_count.get(x, 0) for x in names)
        present = sorted(x for x in names if x in prim_count)
        tier_rows.append((title, eqns, present))
        classified |= names
    unclassified = sorted(set(prim_count) - classified)
    free_eqns = sum(e for t, e, _ in tier_rows if t.startswith(("0", "1")))
    lines += [
        "## Schedule — the inventory is not the build order",
        "",
        f"Ranked by **cost**, not count. Tiers 0–1 — transparency (already",
        f"done), structural/identity ops, elementwise one-liners, and",
        f"fixed-shape folds — cover **{free_eqns}/{result.total} equations",
        f"({free_eqns / max(result.total, 1):.0%})** with a registry of",
        "near-one-liners. They do not depend on saturation and will not",
        "reorder out of the top: build them first. The hard decisions live in",
        "tiers 3–4; the census informs *which* of them matter, not *when*.",
        "",
        "| tier | eqns | % | primitives present in this corpus |",
        "|---|---|---|---|",
    ]
    for title, eqns, present in tier_rows:
        pct = f"{eqns / max(result.total, 1):.0%}"
        lines.append(f"| {title} | {eqns} | {pct} | {', '.join(f'`{p}`' for p in present) or '—'} |")
    if unclassified:
        lines.append(f"| **unclassified — assign a tier** | — | — | {', '.join(f'`{p}`' for p in unclassified)} |")
    lines += [
        "",
        "## Notes",
        "",
        "**`nextafter` ×100 — the float boundary is on page one.** In ℝ,",
        "`nextafter(x, y)` *is* `x`: the primitive has no real-arithmetic",
        "content, and the first census of the ecosystem's best ODE library put",
        "it at rank six by count (diffrax's step-size controller works in",
        "ulps). Its transfer function must be float-aware even under the",
        "ℝ-with-margin semantics — bound it as x ± ulp(x), monotone, easy —",
        "and robust invariants with slack ≫ ulp absorb it, so this vindicates",
        "design commitment 2 rather than threatening it. But the founding doc",
        "framed ℝ-vs-IEEE as a Stage-2 decision, and the swamp showed up in",
        "the first inventory. Noticed deliberately, here.",
        "",
    ]

    # -- held-out arms: censused, compared, never voting --------------------
    if held_rows:
        lines += [
            "## Held-out arm — censused, compared, never voting",
            "",
            "The priority order above is set by the public corpus **only**. The",
            "arms below are the maintainer's own code: included in the ranking",
            "they would train the registry on its author's test set, so they",
            "are held out and asked a different question — *is their profile",
            "covered by the order the public corpus produced?* Covered →",
            "evidence the registry generalises past its sources. Not covered →",
            "evidence about the mature-vs-research gap, from the only sample of",
            "it available. Do not fold these into the corpus later as \"one",
            "more target.\"",
            "",
            "| arm | version | harness | eqns | distinct | covered by public set | novel primitives |",
            "|---|---|---|---|---|---|---|",
        ]
        for arm, version, desc, held_result in held_rows:
            if held_result is None:
                lines.append(f"| {arm} | {version} | {desc} | — | — | — | — |")
                continue
            novel = [(p.name, p.count) for p in held_result.primitives if p.name not in prim_count]
            novel_eqns = sum(c for _, c in novel)
            covered = held_result.total - novel_eqns
            novel_text = ", ".join(f"`{n}` ×{c}" for n, c in novel) or "none"
            lines.append(
                f"| {arm} | {version} | {desc} | {held_result.total} "
                f"| {len(held_result.primitives)} "
                f"| {covered}/{held_result.total} ({covered / max(held_result.total, 1):.0%}) "
                f"| {novel_text} |"
            )
        lines += [
            "",
            "MADDENING pins `jax<0.6`. **The census demonstrated only that the",
            "*trace* succeeds on newer jax** — `make_jaxpr` never invokes XLA,",
            "and the cap was placed against slow *compilation* on other",
            "versions, which the census is silent about. The general lesson:",
            "the cap encodes a real observation (a performance measurement)",
            "through a mechanism that cannot express it (a correctness-shaped",
            "version bound), exporting a constraint consumers never opted into",
            "— demonstrated live when installing it downgraded a shared",
            "environment and broke every other library in it. The fix is the",
            "pattern stelling ships: no cap, a documented tested range, and a",
            "*measurement* where the concern is a measurement.",
            "",
        ]

    lines += [
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
