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

---

# Reading (2026-07-19 — registration `f923078` preceded it; no new runs)

## Which obligation the build would deliver: **R3-shape**, and it needs no assume

From the record, decided rather than assumed:

- **F2-shape is not the candidate.** It takes the floor as an `assume`,
  so it is silent on `cos ≈ 0.11` by construction — that is why F2 failed
  clause (iii) in the first place. Array-aware emission would make it
  *fire*, producing a verdict that assumes the scar away. Nobody would
  build it for that.
- **R2-shape is not the candidate either.** Measured (Part A,
  Correction 1): its component-box region genuinely contains poles and
  sign reversals — an anti-aligned pair lies inside it and
  `assert(Sf·d > 0)` is *violated-over-set*. A refutation there is a true
  statement about a region that **includes meshes the solver never
  produces**, and the witness would likely land near `cos ≈ 0`, not at
  the alignment the scar actually reached. It is not the mesh's region.
- **R3-shape is the candidate.** The code's own expression —
  `jnp.sum(Sf*d)` then `area**2 / Sf_dot_d`, traced verbatim — over a
  region whose declared coordinates *are* the geometric ones. Measured in
  Part A: **100% propagation coverage, declining only at emission.**

**A correction to my own earlier line rides here.** When R2 walled, I
wrote that the blocker "points at affine/relational domains or a
multiplicative reformulation," and the band's gloss said it would
"resurrect assume-emission." **Neither is required.** The geometric
parameterisation supplies what a relational `assume` would have supplied,
so **R3-shape needs array emission only** — assume-emission stays dead on
its own clause-(iii) failure and is not resurrected by this analysis.

## The clause scores

**(i) — the formula is the code's; the parameterisation is not.**
**Passes, with a named residual risk.** The formula is traced verbatim
from `operators.py:250–251`, so a formula edit is tracked automatically —
which is exactly what F1 fails. What is hand-built is the *input
parameterisation*, and it goes stale under a different and narrower
condition: only if the code begins reading a geometric quantity the
parameterisation misrepresents (it currently reads `mesh.area` and
`Sf_dot_d` and nothing else). **That staleness would not be
self-detecting**, and it is the honest residual — smaller than F1's, not
zero.

**(ii) — CI time: comfortable, and the growth model is the finding.**
The obligation is **per-face, and faces are independent**, so it is a
single symbolic obligation *regardless of mesh size* — no per-face
blow-up. The array shapes involved are a fixed `(3,)` dot product, so
the emission growth model is **bounded unrolling of small static
shapes**: three scalar terms, not quantification and not per-element
conjunction over a mesh. Measured scalar solve is ~70 ms; three terms
does not change that materially.

**This materially shrinks the build.** What is needed is not
general array-aware emission but **static-shape unrolling for small
fixed extents** — a much smaller surface than "array emission" implies.
Recorded because it changes the *cost* side of the recommendation even
though it does not change the *relevance* side.

**(iii) — fires on the scar: passes.** R3-shape poses over a region whose
`cos` lower bound is a declared input, so setting it to the scar's
`0.11` puts the failure inside the region rather than outside it. The
solver already produces exactly this split on the algebraically
equivalent form (VERIFIED safe / REFUTED-with-witness at `cos = 1/8`),
and Part A measured the code-shaped uncancelled form escalating the same
way at `cos ≈ 0.117`. So (iii) is not in doubt.

**(iv) — regression-on-change. Not cleared.** Scored against the
one-liner a maintainer *would* write, per the registration:

| axis | one-liner `assert mesh.min_cos >= 0.71` | R3-per-run check | marginal value |
|---|---|---|---|
| a mesh degrades | fires | fires | **none** |
| the coefficient formula changes | **silently keeps passing** — it encodes a constant derived from a formula it never reads | re-derives from the actual code and fires | **real, but only on code change** |

So the check's entire marginal value over the one-liner accrues **on code
change** — which is regression-on-change, the same clause-(iv) status
F1 and F2 hold, and the criterion is explicit that this value is
**temporal and cannot be cleared by inspection**. The measured rate
stands at **zero floor-code changes since the fix introduced them**.

**One strengthening, recorded because it is real and still does not clear
the clause:** R3-per-run **eliminates the `0.71` constant entirely**. It
does not need a floor at all — it takes the mesh's actual alignment and
the code's actual formula and asks whether the coefficient is bounded.
That removes a stale-constant hazard the one-liner structurally has (the
one-liner is *wrong, silently*, if the formula drifts). This makes the
code-form check **strictly better than F1/F2 on (iv)'s mechanism** while
leaving (iv) unmet on (iv)'s own terms, because the mechanism only pays
out when the code moves.

## Band: the middle row

> **Clears (i) ∧ (iii); (iv) is regression-on-change. Array-aware
> emission is a fidelity upgrade, not a relevance one — a cleaner (i) on
> a check whose relevance ceiling is already reached. Not justified on
> the headline alone.**

The sharpest way to put it: **the CI-shaped check that exists today is
R1**, at zero build cost. It already fires on the scar with a replayed
witness. Array emission buys clause **(i)** on top of that — the code's
own form instead of a hand-derived one — and **nothing on (iii) or
(iv)**. That is worth something; it is not worth the largest remaining
emission budget on the strength of the headline.

**Recommendation: consolidate/publish, not build.** And the honest note
that goes with it — the build is *cheaper* than the work order assumed
(static-shape unrolling, not general array emission), so if it is ever
built it should be built for a reason other than this headline.

## What would move this band, stated so it can be checked rather than assumed

(iv)'s temporal condition is a fact about the *workflow*, not the code,
and I cannot measure it:

- **if the FVM coefficient code is under active development** (rather
  than zero changes in the measured window), the machine-re-derives value
  starts paying out and (iv) moves toward clearing;
- **if meshes are generated per-run** rather than being fixed assets, the
  mesh axis stops being covered by a one-liner a human maintains and the
  per-run mechanism matters more;
- **if the same check would cover several coefficients** (the
  orthogonal/over-relaxed pair, the LSQ gradient, the flux limiters)
  rather than this one, the fixed build cost amortises across obligations
  and the calculus changes.

Each is Nick's to answer; none is inferable from the repository. **They
are the questions that would flip this to the top band**, and they should
be answered before the build is reconsidered — not after it is built.
