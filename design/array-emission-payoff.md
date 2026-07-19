# Scoring array-aware emission against the private criterion — before the build

**Status:** REGISTRATION, 2026-07-19, written and committed **before the
analysis is read**. A paper analysis of an unbuilt capability's payoff;
**no build, no run, no new measurement.** The private criterion
(`design/private-track-criterion.md`) already killed one build that
looked justified (assume-emission, on clause (iii)); it rules on this one
before the largest remaining emission budget is spent.

## Why this is worth doing on paper

The headline is unclaimed and the map was corrected by a layer: the
code's own form does not fire because **SMT emission is scalar-only** — a
documented v1 scope at seven sites, and my own solver-build spec
decision. The obstacle is therefore **array-aware emission**, a real
build with a design and audit surface.

**The suspicion, stated so it can be tested:** F1 and F2 both failed
clause (iv) — regression-on-change, not per-run — because the scar's
failure lives *outside the region both forms assume*. **The code-form
check may inherit exactly that ceiling**, making array-aware emission a
large build that upgrades **fidelity** (the code's own form) without
upgrading **relevance** (still not a per-run check). That is the
fidelity-not-relevance trap this whole read-first detour exists to catch.

## Admissible evidence (fixed)

**Only already-measured results**: the F1/F2 scores
(`design/private-track-criterion.md`), the R1/R2 reading and the Part A
corrections (`design/regional-obligation.md`), the R3 probe measurements
recorded in Part A, the three-rows outcome (`193a9e0`), and the MIME
source facts at the pinned ref. **No new runs.** If the analysis turns
out to need a measurement that does not exist, that is a stop — say so
rather than estimating.

## The three candidate obligations — the analysis must say which one the build delivers

Array-aware emission is the *same build* whichever obligation you emit.
**Which obligation the code-form check would pose is the whole question**,
and it is answerable from the record:

- **F2-shape** — the code's own form with the alignment floor as an
  `assume`. Assumes the scar away by construction.
- **R2-shape** — the code's own form over independent component boxes.
  Measured (Part A, Correction 1): that region genuinely contains poles
  and sign reversals — an anti-aligned pair is inside it and
  `assert(Sf·d > 0)` is *violated-over-set*. It is **not the mesh's
  region**.
- **R3-shape** — the code's own expression over a region parameterised so
  the declared coordinates *are* the geometric ones (Part A's probe:
  `Sf`, `d` constructed from `(a, dm, c)`; the code's own
  `jnp.sum(Sf*d)` and `area**2/Sf_dot_d` traced verbatim). Measured to
  reach 100% propagation coverage and decline **only** at emission.

## How each clause is scored (fixed before reading)

- **(i)** — the *formula* must be traced from the code, not hand-derived.
  A hand-built **input parameterisation** is a separate question from a
  hand-derived **formula**: the first goes stale only if the code starts
  reading a quantity the parameterisation misrepresents; the second goes
  stale on any formula edit. Score both, and name the residual risk of
  whichever is present.
- **(ii)** — CI time, estimated from the measured scalar solve plus the
  emission's term growth. **State the growth model** (element-wise
  conjunction / quantified / bounded unrolling) rather than assuming one.
- **(iii)** — fires on the scar. **Determine first**, because if it
  fails, (iv) is moot: does the delivered obligation *assume* the floor
  (silent on `cos ≈ 0.11`) or *pose over* the region containing it?
- **(iv)** — recurring value. The decisive comparison is against **the
  one-liner that would fire** — and the measured fact that no such
  one-liner exists in the FVM package must not be used to inflate the
  score: the comparison is against the one-liner a maintainer *would*
  write, not against the empty set. Per the criterion, (iv) admits
  regression-on-change value but that value is **temporal and cannot be
  cleared by inspection**; the measured rate is **zero floor-code changes
  since the fix introduced it**.

## Bands (as commissioned; fixed before the analysis is read)

| finding | reading | action |
|---|---|---|
| code-form check poses over the **live-mesh region** and clears (i)∧(iii)∧(iv) | array-aware emission earns the **first genuine CI-shaped usefulness result** — build justified | surface, recommend, scope the surfaces |
| clears (i)∧(iii) but **(iv) is regression-on-change** | a **fidelity upgrade, not a relevance one** — same ceiling F1/F2 hit. **Not justified on the headline alone** | surface; honest recommendation is consolidate/publish |
| **(iii) fails** for the code form | the headline would be **earned-but-empty** — a verdict that assumes the scar away | surface as a finding |

**No band is amended after reading, and the criterion is not relaxed to
justify a build.** If the code-form check inherits the ceiling, that is
the finding and the output is "consolidate/publish," not "build anyway."

## Non-goals

No build. No run. No CI wiring. No affine (two sightings, not three). No
user contact. MIME held out from counting; any headline is private-track,
never an E2a count.
