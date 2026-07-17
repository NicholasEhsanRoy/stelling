# The value model, v0.2

**Status:** normative, 2026-07-18. Supersedes `value-model.md` (v0.1) *as
the value model*; v0.1 is deliberately not edited — it is a standing
registration whose wedge falsifier still binds any wedge run, and its
census re-registration window remains the maintainer's open decision.
Written, as ordered, the same pass `design/precision-probe.md` landed —
which it landed **dead**.

**The job, unchanged from v0.1's framing: one claim that could be wrong,
and what would show it.** Two pages; links, not summaries.

## The claim

> For a **maintainer of solver-shaped JAX code** — code with
> data-dependent inner iteration; "library" was the tracker's selection
> effect (17 of 20 hits are solver infrastructure, and the one custom-code
> sample has the same core: `design/tracker-probe.md`,
> `design/maddening-archaeology.md`) — machine-checked **qualitative state
> predicates** on solver state (box invariants that discharge guard and
> singularity obligations; liveness of the defences themselves) are cheap
> enough to mechanize and bear on real recorded incidents.

That is the whole claim. This model prices one layer and abandons the
other, by name.

## What it prices, and what it abandons

**Priced: the state-predicate layer.** Both supply-probe invariants were
box-shaped, two hand-rounds each, plausibly reachable by forward interval
propagation (`design/supply-probe.md`). **13 of 20** recorded hits are
addressable by predicate shape (`design/layer-probe.md` §3) — with the
wrong-layer caveat welded on: *addressable* means the predicate can be
stated and checked at the state layer, not that it discharges the
incident; measured incident discharge is **0/3**. The qualitative export
survives the precision result: `c ≥ 0.019` discharges the log-singularity
obligation regardless of tightness.

**Abandoned, not silently: the quantitative / incident-discharge layer.**
Three measured reasons. (1) 0/3 incidents discharged; every layer
descended minted more contracts (|f| ≤ M → NN-Jacobian bound →
dt-conditions → error models), and the NN-Jacobian contract has **no
maintained JAX-native supplier** (jax_verify archived 2023-08; auto_LiRPA
is PyTorch — process-boundary import at best; `design/layer-probe.md` §5).
(2) The one quantitative bound this project produced was measured **10⁴×
vacuous** against its filed property and ~4.5×10⁶ against reality, with
the in-plan precision machinery measured at **1.0×** (affine — the box sup
is Z3-proved corner-attained) and the out-of-plan machinery at **1.3×**
(tubes — the reachable tube *fills* the proved box): the gap is the
error-model contract itself (`design/precision-probe.md`). (3) What would
have to be true to reprice this layer: a stiffness-aware local-error
contract (solution-derivative / B-convergence tier) *with a supplier*, and
a standing tightness discipline — no bound ships without its ratio to the
property it serves. Neither exists.

**The multiplier, not a target: drift.** ~3.6/yr silent semantic changes
(a floor; `design/semantics-drift-probe.md`) decay every hand-checked
predicate; the value form is **differential only** (re-check per jax
series bump), welded to the circularity limit — stelling's transfers *are*
its jax model, so only cross-version diffs are trustworthy signal.

## The experiment and the falsifier

Two experiments, neither needing user contact.

**E1 — first result, already registered:** the guard experiment over the
17 clamp sites (`design/guard-experiment.md`); its bands and falsifier
stand as written there.

**E2 — the mechanization test — split (2026-07-18), because "mechanize"
was hiding two products.** The priced layer's "cheap" is evidenced only by
hand proofs; but *checking* a stated predicate and *deriving* one are
different claims with different markets, and a falsifier that can't tell
them apart isn't one.

- **E2a — check mode.** The harness states the box; stelling verifies its
  inductiveness. Tests whether **checking** is mechanizable — the user
  writes the invariant; the product is a methodology. **E2a carries the
  bands below** (they were written for it), beginning with re-deriving
  hit386's own box with no hand assistance beyond harness setup
  (operational definitions: `design/e2a-registration.md`).
- **E2b — derive mode.** Forward propagation with widening; stelling
  infers the box. Tests **push-button** — the tool. E2b is E2a plus a
  fixpoint (the widening loop's inner step *is* an inductiveness check),
  so it cannot be built first. **Its bands are not registered here and
  will not be guessed**: they get registered when E2b is reachable, not
  before.

E2a's bands, fixed now:

| mechanized predicates (of 13) | reading |
|---|---|
| **0** — including failure to re-derive the already-hand-proved box | **Falsified.** The cheap layer is not cheap when mechanized; the model dies, reported in the same sentence as the number |
| 1–3 | Weak. Publish, re-aim; not grounds to build further |
| **≥4, across ≥2 libraries** | Supported. The ≥2-libraries clause is load-bearing, as in v0.1 |

Anti-rationalizations, v0.1's in spirit: a predicate the tool *almost*
derives counts 0; "the harness was wrong" is a work item, never a band
adjustment; the corpus (the 13, as listed in `design/layer-probe.md`) does
not change after a run.

## The corrections this model inherits — all of them, none quietly

- The 20-hit corpus is **post-detection by construction**
  (`design/tracker-probe.md`, scope note): this model can never claim
  detection value.
- Its cost signals are **calendar proxies**; the demand band's cost
  evidence is weaker than filed.
- The attribution reframe came back **mixed** — median in-thread ratio
  0.45, misdiagnoses 5/20 (`design/attribution-probe.md`). **The
  proposer's prior lost**, and no diagnostic-dominance claim appears in
  this model.
- **The instrument confound has now bitten three times** (wedge/style,
  drift/"silent means not filed", attribution/named-in-body): trackers
  measure what gets written down afterwards. jmd#339 — the corpus's worst
  cost signal — scores 0/2 in-thread because the user narrowed it before
  filing.

## Not in this model

- The census, rule provenance, and archaeology are inventories and
  history — context here, never value evidence (the census artifact says
  so about itself).
- Elimination mode is written, attacked, and **unadopted**
  (`design/jax-verification-categories.md`); if it ever enters, it enters
  by its own registration. The five layer-error misdiagnoses are its
  shape, not its test.
- Kills stay killed; the "valuable for new projects" hatch stays closed
  (byproduct policy, same artifact).

## Revision list — banked, not run

1. WO14 §2: the jmd#339 deep-read and the frame's unsampled cells.
2. WO14 §3: the CROWN export path (process-boundary contract import).
3. The stiff local-error contract: does a checkable B-convergence-tier
   contract exist for Kvaerno-class solvers, and who supplies it?
   (minted by `design/precision-probe.md`).
4. Elimination mode, if ever — its own corpus and falsifier first.
5. ~~E2's operational definition of "no hand assistance" — fixed before
   E2 runs~~ — resolved for E2a by `design/e2a-registration.md`
   (2026-07-18); still open for E2b, whose registration happens when E2b
   is reachable.
