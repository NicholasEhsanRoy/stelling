# The LA census and the contract-feasibility probe — registered before either runs

**Status:** REGISTRATION, 2026-07-18. No build in this pass: two sizing
probes and one free check, resolving the affine/LA build order that is
currently a guess resting on one real-job data point (F3's tier-9 wall)
and one census equation (lineax's undercounted `lu`).

## The hypothesis under test — stated so the probes can break it

> Affine is the keystone (independent demand: FVM's stencil correlation,
> LBM's Mach ratio; and the substrate LA contracts need). LA is a
> contract layer **on** affine. The solver is the escalation valve that
> fires when a conditioning obligation turns out nonlinear. Build order:
> affine → LA-as-contract → solver-when-a-real-obligation-needs-it.

Break conditions: §1 finds LA rare-and-Nick-specific (LA deferred or
dropped; next pass is affine-or-publish); §2/Q1 finds the conditioning
precondition factors cleanly for intervals (the stack is not real).

## §1 — The LA census (kill or confirm the selection artifact)

The tier-9 "defer" was filed on **one** `lu` sighting the census itself
flagged as an undercount (lineax's solve hid inside an equinox opaque),
from an ODE/MCMC-heavy corpus — exactly where dense LA is hidden or
absent. **Affine may have been prioritised over LA as a selection
artifact.**

Measure: `lu`, `custom_linear_solve`, `triangular_solve`, `inv`-routes,
`cholesky`, `qr`, `eig`/`eigh`, `svd` — occurrence, breadth, and **depth
context** (core computational path vs incidental) — over **optimistix
and lineax**, the censused-but-never-LA-examined solver libraries,
extending the existing harnesses only as far as the solve path.

Reading, fixed before the count:

| finding | reading |
|---|---|
| LA central in optimistix/lineax (core path, at breadth) | the census under-weighted LA; the tier-9 defer was a selection artifact; **LA rises** — a general-tool frontier, not a Nick-specific one |
| LA genuinely rare even in solver libraries | F3 was specific to the author's solvers; LA is a usefulness-on-his-solvers investment, and that distinction drives whether it is built at all |
| mixed / hidden behind opaques | the census cannot see it (the lineax undercount again) — a recorded blind spot; §2 carries the weight |

The answer is stated in the registered words: *was affine prioritised on
a real signal or a sampling artifact.*

## §2 — The contract-feasibility probe (hand-work F3's real obligation; no transfer built)

F3's ⊤ was the LSQ gradient's solve: per cell, normal equations with
`A = DᵀD` (D = neighbor position deltas — **pure mesh geometry**; only
the RHS carries φ). The contract shape is assume-guarantee across the
boundary (the project's registered CROWN-shape): `requires:` A
well-conditioned over the geometric region; `ensures:` a bound on the
solve's output. LAPACK is cited, never reproved.

**Q1 — is the precondition dependency-shaped?** Prior, registered: YES,
in a nameable way — `det(A) = Σdx²·Σdy² − (Σdxdy)²` is **Cauchy–Schwarz
slack**, a correlation of correlated sums; intervals lose it twice over.

**Q2 — is it nonlinear past affine?** Prior, registered: the
conditioning bound for the 2×2 symmetric case reduces via
`tr²/det = κ + 1/κ + 2` at the ratio bound to a **small polynomial
(QF_NRA) validity question** — degree ≤ 2 in the entries, 3 variables —
which affine forms cannot close in general and an SMT solver checks
directly. If the hand-work confirms this, **the solver has its first
real customer in thirty-five passes**, and the check is run by the
supply-probe instrument (pen + Z3, the registered hand-proof method —
not a solver integration).

**Q3 — does the `ensures` reach F3's gradient property?** Prior,
registered, and it is a **middle case Nick's table did not list**: a
residual-only `ensures` does *not* bound the gradient (the gradient is
the solution, not the residual) — but the needed `ensures` is a
**norm-sensitivity bound** `‖x‖ ≤ κ·‖b‖/‖A‖`, derivable from the *same*
conditioning data as the precondition: one clean layer, stronger than
residual-only, far short of backward-error analysis. Backward error
enters only when the ℝ-semantics dial moves (the float-solve's ΔA is the
stamped gap, as everywhere).

Reading table (Nick's, fixed) plus the registered middle row:

| Q1 | Q2 | Q3 | reading |
|---|---|---|---|
| dependency-shaped | closes with affine | residual suffices | best case: LA is one clean contract layer on affine |
| dependency-shaped | needs solver | residual suffices | LA needs affine **and** the solver — the solver finally has a customer |
| any | any | gradient needs backward-error | LA is a rabbit hole for gradient properties; scope LA to residual-reachable obligations |
| dependency-shaped | needs solver | **sensitivity-ensures suffices (the registered prior)** | the stack is real, the solver has a customer, and the ensures is κ-derived — LA is one layer, priced honestly as requires(QF_NRA) + ensures(sensitivity) |

## §4 — The LBM Mach check (free; affine's independent customer)

Ring discipline: the LBM solver source only, no other MIME code, blind
to validation results. Question: is the Mach constraint `u/c_s < Ma_max`
affine-shaped and LA-free? Prior: yes and more so — `u` is a ratio of
correlated moment sums (`Σcᵢfᵢ / Σfᵢ`), the `0.8·s_max` shape with a
normalization on top, and no linear solve anywhere in it. If confirmed,
affine has a second real-solver customer independent of the entire LA
thread.

## §3 — What the probes jointly decide

§1: general frontier vs Nick-specific. §2: whether the stack is real,
what LA costs, whether the solver has a customer. Together: the build
order becomes a decision. If the hypothesis breaks, affine stands alone
on its stencil/Mach demand and the next pass is affine-or-publish.
Either outcome is a decision the probes earn; **neither is a build.**

---

# Reading, part 2 (2026-07-18 — §1's census and the §3 joint decision)

## §1 — the answer, in the registered words

**Affine was prioritised on a real signal; LA was de-prioritised on a
sampling artifact.** The census (`corpus/la_census.py`: 14 solve paths,
1225 equations, optimistix 0.1.0 / lineax 0.1.1, jax pinned):

- **LA is always-core when present, zero incidental**: 28 LA equations,
  every one on the core computational path of its solve, with quoted
  functions (`LU.init` `lu.py:53`, `QR.compute` `qr.py:89–92`,
  `Cholesky.compute` `cholesky.py:74`, `SVD.init` `svd.py:51`, …).
  Breadth: `linear_solve` 9/14, `lu` and `geqrf` 4/14 each,
  `cholesky`/`triangular_solve` 3/14, `svd` 2/14.
- **And thin by equation count**: 2.3% of all equations; 11–21% density
  inside lineax's surface traces, 0.6–0.8% inside optimistix's real
  solver loops — a couple of factorization equations at the heart of
  hundreds of housekeeping ones, every Newton/GN/LM iteration passing
  through them.
- **And structurally hidden — the original undercount, mechanically
  mapped**: lineax's `linear_solve` is its own equinox primitive with
  **no sub-jaxpr**; `solver.init` (the factorization) traces *before*
  the bind, `solver.compute` (the back-substitution) is invisible to any
  trace walk. Surface traces see `lu`/`geqrf`/`cholesky` and never the
  `triangular_solve`/`ormqr` behind the boundary — the deep harnesses
  had to trace the concrete solvers' `init`/`compute` directly. (lineax
  does not use jax's `custom_linear_solve` at all — MIME's F3 and the
  ecosystem route through *different* mechanisms to the same tier.)
- Two census facts for the record: BFGS's zero-LA is real
  (`use_inverse=True` takes the inverse-Hessian matvec branch; the
  configured Cholesky is unreachable), and the QR family lowers to
  `geqrf`/`ormqr` on jax 0.11 (the composite `qr` never appears —
  counting rule recorded). One transcription finding, recorded not
  built: optimistix's function-operator paths leak
  `DynamicJaxprTracer`s inside equinox static metadata; stock
  transcription correctly refuses, and the census used a local
  subclassed transcriber (sentinel-recorded, stelling core untouched).

**Reading, per the registered table:** the first row's condition is met
(core path, at breadth) *and* the third row's hiding is confirmed and
mapped. The tier-9 "defer on one equation" was a **selection artifact**
— but the corrected weight has a precise shape: **few, always-central,
boundary-hidden** — which is not a transfer-registry shape at all. It is
**exactly the contract shape**: one assume-guarantee boundary at the
solve, not thirty transfer rows. The ecosystem's own opaque
`linear_solve` boundary is the natural contract attachment point. LA
rises — as a contract layer, and as a general-tool frontier (lineax and
optimistix are ecosystem libraries, not the author's).

## §3 — the joint decision, and the one correction the probes earned

The keystone hypothesis is **confirmed with a sharpening**:

- **Affine is the keystone for linear-correlation obligations** — two
  independent real-solver customers (FVM's stencil, LBM's Mach ratio),
  both linear-in-the-correlated-quantities. Confirmed.
- **The correction: LA-as-contract sits on the *solver*, not on
  affine.** §2 showed the `requires` (conditioning) is *quadratic past
  plain affine* and closes as QF_NRA — the solver's demonstrated first
  customer — while the `ensures` is κ-derived arithmetic. The original
  "LA on affine" is amended: **affine and the solver are sibling
  escalations for different obligation shapes** (linear correlations vs
  polynomial feasibility), and the LA contract needs the solver leg.
- **Build order, as decided by the probes:** affine (independent demand,
  two customers, linear shape) and solver-integration (the LA
  `requires`, plus any future QF_NRA obligation) are both licensed *in
  demand terms*; LA-as-contract follows the solver leg; each build
  remains gated (fresh-context builder, audit gate, no counts) as
  registered. The next pass is a **choice among affine / solver /
  publish** — a decision, not a drift, which is what the probes were
  for.

---

# Reading, part 1 (2026-07-18 — §2 and §4; §1's census pending)

## §2 — the contract probe landed the registered fourth row, with demonstrations

`corpus/supply/la_contract_probe.py` (the supply-probe instrument):

- **Q1 — dependency-shaped, demonstrated.** The same conditioning
  obligation Z3 proves (below) straddles for intervals over the same
  region: `tr²/det ∈ [1.00, 21.33]` against the threshold 10.125 — `a`
  and `c` are shared between `tr` and `det`, and `det` itself is
  Cauchy–Schwarz slack in the underlying `d`-vectors. Quadratic past
  plain affine.
- **Q2 — the solver's first demonstrated customer.** The conditioning
  bound reduces exactly (`tr²/det = κ + 1/κ + 2` at the ratio bound) to
  a 3-variable degree-2 **QF_NRA validity**, and **Z3 decides it both
  ways on the real obligation**: PROVED (unsat) over the well-shaped
  region (`b ∈ ±0.5`); a concrete violating witness (`a=3/2, b=5/4,
  c=3/2`) over the sliver-reaching region. Thirty-five passes of zero
  search-shaped UNKNOWNs end here: an obligation intervals straddle and
  a solver closes, from a real job. (The designed-and-idle solver
  architecture now has its test case; integration remains a later, gated
  pass.)
- **Q3 — the registered middle row confirmed.** Residual-only `ensures`
  does not bound the gradient (the gradient *is* the solution). The
  needed `ensures` is norm-sensitivity `‖x‖ ≤ κ·‖b‖/‖A‖` — derived from
  the *same* conditioning data as the `requires`. One clean layer;
  backward error enters only if the ℝ-semantics dial moves.
- **The grounding fact:** F3's own per-cell normal matrix is rank-1 plus
  the `reg=1e-30` diagonal — **cond ≈ 2.5×10²⁹ on the probe's own
  mesh**. The `requires` is not hypothetical; boundary-starved and
  sliver stencils are the real failure geometry. (Also a small honest
  lesson en route: the naive `(tr−disc)/2` eigenvalue cancels to 0.0 in
  float here — the stable `det/λ_max` form was needed to *compute the
  conditioning number* of a matrix about conditioning.)

**§2 reading: the stack is real** — the precondition is
dependency-shaped and its closure is solver-shaped; the ensures is
κ-derived; **LA is one contract layer, priced as requires(QF_NRA) +
ensures(sensitivity), sitting on relational/affine substrate with the
solver as the closure instrument.**

## §4 — the LBM Mach check: affine's independent customer, confirmed

Ring-clean (LBM solver files only, no validation docs): the constraint
is the node's own `ValidatedRegime("Ma", 0.0, 0.1)` — *"Lattice Mach
number (max over all nodes, incl. fin tips) must be < 0.1"*
(`fluid_node.py:182–185`; the input clamp `u_max_lattice = 0.0577 =
0.1/√3`, `:959, :968`). The bounded quantity: `rho = jnp.sum(f, axis=-1)`
(`pallas_lbm.py:124`), `momentum = jnp.matmul(f, e)` (`:129`),
`u = momentum/rho` — **a ratio of correlated moment sums** (`Σfᵢeᵢ/Σfᵢ`,
whose true range is the convex hull of the lattice velocities — exactly
the correlation intervals discard), and **LA-free**: no factorization,
no solve, anywhere in the path. Affine has a second real-solver
customer independent of the entire LA thread. (Adjacent scar noted in
passing, not this check's subject: the momentum's near-cancellation is
TF32-fragile — `pallas_lbm.py:125–128` — a float-precision scar in the
Mach path's neighborhood, of the ℝ-gap's kind.)
