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
