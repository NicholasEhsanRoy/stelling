# The precision probe — decompose the 10⁴ gap, registered before computing

**Status:** REGISTRATION, 2026-07-18. Work order 14's §1, unchanged. No new
probes: this re-derives one number three ways on an object already in the
repo (`corpus/supply/layer_pid.py`, the PID ranking bound on the #386
system). stelling forbidden; pen, numpy, and Z3; four-hour analogue;
fidelity demotions recorded.

## The object

The PID proof *succeeded* — ranking function exists, pillars Z3-proved —
and returned **N ≈ 1.3×10⁹ accepted steps against a filed max_steps of
10⁵**. The error constant was `C = L⁶` with `L` the dominant Jacobian
scale taken from the supply-proved box. The question: which part of the
10⁴ looseness is *domain imprecision* (interval dependency problem —
affine/zonotope forms fix it, in plan) and which is *set
over-approximation* (box ≫ reachable tube — tubes fix it, not in plan)?

## The three numbers — definitions fixed

The formula is held fixed across all three: `N = span / (ρ·(tol/C)^(1/6))`
with `C = L⁶`, `tol = 10⁻⁸`, `ρ = 0.2`, `span = 100` — exactly
`layer_pid.py`'s. **Only the set over which `L` is sup'd moves.** That is
the isolating control: the error-model proxy is a constant of the
experiment, so whatever gap the three numbers do *not* explain is
attributable to the model term and is reported as such, separately.

| number | how |
|---|---|
| **N_box = 1.3×10⁹** | as filed (`layer_pid.py`) |
| **N_affine** | the error-constant computation redone with affine/zonotope treatment over the *same* box. Method: if the sup of `L(x1,c) = e^{a3}·x1·e^b/(c+e^b)² + e^{a4}` over the box is attained at a box point (to be Z3-checked over directed-rounding brackets, as in `hit386_termination.py`), then the interval bound is exact and **no affine form over the same box can be tighter** — N_affine is then the corner value, a *sound floor* for the whole in-plan machinery class |
| **N_tube** | numerically integrate from a sample of initial conditions in the region; take the ranges `x1` (and `c`) **actually occupy**; recompute `C` on those. Deliberately unsound — a bound on the *achievable*, not a proof |

Sampling protocol, fixed: primary trajectory = the incident's own
`y0 = (4.1155, 6.8318)`; secondary = a grid of ICs across the proved box
(`x1 ∈ [6.8, 415]`, `c ∈ [0.019, e^{a1})` — the invariant is one-sided in
`c`; the sampling cap `c < e^{a1}` is `x0 > 0`, recorded as a choice).
Integration: fixed-step RK4 at a step well under the stiffness limit,
plus scipy BDF/LSODA as cross-check; per-trajectory sup of `L` along the
trajectory, and the union across ICs, both reported. As an annotation
(not a band input): the *empirical accepted-step count* of an adaptive
stiff integrator at the issue's tolerance, from `y0`, over the span.

## Bands — fixed (from the work order, against N_filed = 10⁵)

| finding | reading |
|---|---|
| **N_tube > 10⁵** | **Dead for this hit.** No machinery helps — not zonotopes, not tubes. The looseness is in the method, not the domain. Loudest; most changes the model |
| **N_affine < 10⁵** | **In-plan machinery suffices.** Reclassify affine/zonotope from nice-to-have to **prerequisite** |
| **N_tube < 10⁵ < N_affine** | **Tubes needed, not in plan** |

Regardless of outcome, into the categories artifact: **precision is not a
nice-to-have, it is viability** — an envelope's value is a function of its
tightness, and tightness had never been measured.

## Prior and contamination, recorded

Reconstruction required re-reading `layer_pid.py` and
`hit386_termination.py` before this registration existed — unavoidable,
and it means the analysis below was in hand *before* the bands were
committed (the bands themselves are the work order's, fixed upstream of
the analyst). Recorded rather than hidden:

> **Prior: the dead band.** Grounds from reconstruction: (1) `L` is
> separately monotone in `x1` and `c` with single occurrences, so the
> interval sup over the box is *exact* — the dependency problem is absent
> and N_affine ≈ N_box; (2) the system's equilibrium sits at `c* ≈ 0.03,
> x1* ≈ 414.5` — *inside the boundary layer that produced L ≈ 1.2×10⁵* —
> so every trajectory, including the incident's own, converges to the
> worst region of the box, and N_tube ≈ N_box too. If that holds, the
> entire 10⁴ gap lives in the third term the work order's dichotomy does
> not name: the **error-model contract** `C = L⁶` itself.
>
> What would kill the prior: the numerics showing trajectories do not
> approach the `c*` layer from sampled ICs, or the occupied-range sup of
> `L` coming in orders below 1.2×10⁵.

One fidelity note from reconstruction, recorded before computing: the
filed `N_box = 1.3×10⁹` evaluated `L` at the boundary-layer point
(`c* ≈ 0.03`), not at the box corner (`c = 0.019`); the sound
interval-over-the-box value is expected slightly *worse* (~1.7×10⁹). The
filed number was already tube-grade in its `L`. Reported in the reading
either way.
