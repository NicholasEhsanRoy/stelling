# The supply probe — can the invariants be proved, registered before reading source

**Status:** REGISTRATION, 2026-07-18. Every supported band so far measured
**demand**; nothing has measured whether an invariant **catches** anything.
This probe takes three tracker hits, writes their properties formally, and
attempts the inductive proof **by hand** — library source, pen and paper,
and Z3 driven directly on a hand-transcribed model of the step function.
**stelling is forbidden**: it has no transfer functions past tiers 0–1,
and using it would confound problem difficulty with tool maturity. The
problem is what decides the product.

## Selection — mechanical, computed here, before any source is read

Strata per the verified technique split (categories artifact): invariant
(9 hits), termination (5 hits). Rule, fixed: **rank by recorded comment
count within stratum (probe-1 hit table), tiebreak by days-to-close
(longer = costlier); take the top 2 from the invariant stratum and the
top 1 from termination.** Cost-ranked selection deliberately tests the
expensive end of demand: a tool verdict from this sample is strong; a
methodology verdict is a statement about the priced end, and is reported
as such.

**Computed selection (before proving):**

| hit | stratum | cost signal |
|---|---|---|
| **numpyro#1360** — NUTS seed-dependent divergences on a regression model | invariant | c20 |
| **diffrax#368** — integration failures inside NN-ODE training | invariant | c16 |
| **diffrax#386** — step explosion at specific parameter values | termination | c12 |

If a selected hit proves unreadable, the next by the same rule is taken
and the substitution is reported. The selection is not re-run.

## What a round is — fixed

> **A round = any change to the invariant *or the region* required to make
> the inductiveness check pass.** Narrowing the region to rescue an
> invariant is the user writing more spec, exactly as adding a conjunct
> is.

Not rounds: transcription typos, re-reading code, correcting my own
algebra. **Model simplifications are not rounds either — they are
fidelity losses, recorded per hit, and they demote the outcome to
"proved for the simplified model."**

## Budget and outcomes — fixed

Budget: the agent analog of four hours per hit — a bounded work budget
(capped strengthening rounds and bounded solver attempts); where it stops,
the stop and the remaining obstacle are reported.

| outcome | meaning |
|---|---|
| proved in N rounds | report N and the final invariant |
| not provable by induction on the step function | a **technique gap** — ranking function, supermartingale, contract on a learned component: named explicitly |
| gave up at budget | where, and what the obstacle was |
| property was wrong | the tracker probe's one-line property did not survive contact — recorded loudly as a finding about the tracker probe |

Per proved hit, the **inferability classification**: box/interval
constraints on state components (forward propagation with widening
plausibly finds it — push-button live) vs nonlinear relations between
components (inference will not find it — the user writes it;
methodology).

## Bands — fixed

| finding across the three | reading |
|---|---|
| 0–1 rounds typical, box-shaped invariants | **Tool.** Push-button survives; the value model aims at L1+L2 as ranked |
| 2–4 rounds | **Borderline.** The Dafny loop; the model prices the strengthening honestly and stops saying push-button |
| 5+ rounds, a page of algebra, or nonlinear invariants | **Methodology for experts.** Different product, different market |
| ≥1 technique gap | reported separately and prominently; bears on the ranking regardless of round counts |

**The distribution matters more than the mean** — one 0 and two 6s is two
products, not an average of 4, and will be reported as two products.

## The limit — stated now

**The prover is not a user.** An LLM with unbounded patience is probably
better at the algebra and worse at knowing which conjunct is physically
natural. The round count measures **the problem, not the user** — the
right thing for deciding the product, and the artifact must not read as a
measurement of user effort.

---

# Reading (2026-07-18 — proofs in `corpus/supply/`, Z3-checked)

## Per-hit outcomes

| hit | outcome | rounds | invariant shape | technique gaps |
|---|---|---|---|---|
| **diffrax#386** (termination) | state-space half **proved for the continuous flow**: `x1 ∈ [6.8, 415] ∧ exp(a1)−x0 ≥ 0.019` — the log-singularity boundary *repels*, all four edge-flux obligations Z3-PROVED over directed-rounding brackets of the exp constants. Termination half: **not provable by induction on the step map** | 2 | **box** | controller/error-model contract + ranking argument (the L2 "different machinery" claim, now measured at its own selected hit) |
| **diffrax#368** (invariant) | **proved for the continuous flow over the entire parameter family** (k ∈ [−3,3]⁴): box `x0 ∈ [−8.1, 8.1], x1 ∈ [−3.1, 3.1]`, every edge reducing to a linear sign fact after the monotone-exp rewrite, Z3-PROVED — **conditional on the contract `|f(x,p)| ≤ M`** for the pretrained NN. And the proved invariant **does not discharge the incident**: the failure was NaN inside Kvaerno5's Newton iteration, and Newton trial iterates are not flow states | 2 (+1 forced contract) | **box** | (1) learned-component contract — CROWN-tier, the roadmap's "learned nodes" item; (2) obligations over **solver-internal iterates** — no roadmap item covers them |
| **numpyro#1360** (invariant) | **property was wrong** — false over its own registered region, necessarily: the incident (2–3 of 10 seeds diverge) is its counterexample. Computed at a region corner: curvature `n/σ² ≈ 1.1e3`, adapted-to-bulk ε exceeds the leapfrog stability limit **~37×** → geometric energy-error growth → divergence. The supply object is the *envelope* (which inits hold?), which is nonlinear in state **and coupled to the adaptation trajectory** | n/a | not box | trajectory/expectation-tier machinery — the founding doc's supermartingale bullet, its farthest item |

**Fidelity demotions, as registered:** both proofs are for the
**continuous flow**; discrete-step preservation (Kvaerno5's implicit step
can transiently exit the box) is unproven and would cost further rounds or
a dt-condition. The #386 box also has a trivially-inward `x0 ≥ 0` face by
sign inspection (omitted from the script, noted here).

## The loud finding about the tracker probe

Two of three one-line properties did not survive contact, in two
different ways: #1360's is **false over its own region** (the incident is
the counterexample — the tracker's property test wrote the *desired*
invariant over the *incident's* region), and #368's **conflates layers**
(flow state vs solver-internal state; "solve completes finite" is a
property of Newton iterates the flow invariant cannot reach). The property
test produced constructive properties; it did not check them against the
incident they came from. Any future registration using one-line properties
inherits this correction.

## Band assessment — the distribution, not the mean

Where proving happened: 2 rounds each, box-shaped, edge checks reducing to
monotone-rewrite sign facts — nominally the **borderline** band's cheap
end, and the boxes are exactly the shape forward propagation with widening
plausibly infers. But the registration's technique-gap row dominates:
**four distinct gaps across three hits** — controller/ranking,
learned-component contract, solver-internal iterates,
trajectory/expectation tier — and none of the four is in the Stage-2
plan.

**The headline: the provable layer and the incident layer are different
layers.** Induction on the step map reaches state-space boxes cheaply
(tool-shaped, inference-plausible, contract-gated). The incidents live in
solver internals, controller dynamics, and adaptation coupling —
technique-gapped, methodology-shaped where reachable at all. One sample of
three, cost-ranked (the expensive end of demand, as the selection rule
biases and the registration says); but on this sample, **"push-button
verdicts about the incidents that motivated the demand" is not what
induction on the step function sells.** What it sells cheaply is the
state-space envelope — which is also precisely the artifact shape (bounds,
envelopes) the categories ranking already favored.

Prover-is-not-a-user limit restated: these round counts measure the
problem. A scientist would not have found the c-boundary repulsion
argument in four hours or might have found it in one; neither fact is in
this data.
