# The value model

**Status:** design note, v0.1, 2026-07-16. Normative for the Stage-1 experiment
and its reporting. **Written before the experiment ran** — that is the point of
it; see "Why this is pre-registered" below. Re-verify trigger: the experiment
completing.

`DOCUMENTATION_ARCHITECTURE.md` is a cost model: it says what legitimacy costs
and nothing about what a verdict is worth. A decision instrument with one half
cannot produce a decision, and the half we had is the half that argues for
stopping. This is the other half, and its only job is to **state a claim that
could be wrong, and say what would show it.**

## The claim

> JAX scientific code contains silent out-of-bounds indexing at a rate that
> matters, the existing tools structurally cannot find it, and stelling's wedge
> can.

Three conjuncts. All three must hold. The third is nearly free (§below); the
second is an argument; **the first is an empirical question nobody has answered,
and it is the whole thing.**

## The bug class

Under `jit`, an out-of-bounds gather does not raise — it **clamps**, silently
returning the nearest valid element. An out-of-bounds scatter **drops**, silently
discarding the update. No exception, no NaN, no warning: the program returns a
plausible array of the right shape and dtype, and the physics is wrong.

What makes it survive review:

- **It has no symptom.** Every failure mode a scientist is trained to look for —
  crash, NaN, inf, shape error — is absent by construction. The output is finite,
  well-shaped, and wrong.
- **It is boundary-conditional.** Clamping bites only when an index actually goes
  out of range: a particular geometry, timestep, configuration. The test suite
  runs the configurations someone thought of.
- **It reads correctly.** `x[i + 1]` is not a suspicious line; the bug is in the
  reachable range of `i`, which is somewhere else.
- **Eager and compiled disagree.** Outside `jit` JAX raises on some of these;
  inside `jit` it clamps — so the interactive loop is clean and the production run
  is wrong, the one place a developer's instincts are inverted.

Neighbour lists, halo and stencil kernels, and hand-rolled indexing over a mesh
are where this lives.

## The counterfactual

What people do today: runtime checks (`checkify`), unit tests, property-based
fuzzing, and looking at the output to see whether it seems physical.

The observation that unifies them:

> **They are all point methods. The wedge is a region method.**

A test checks one input. A fuzzer checks many inputs, chosen by a process that
does not know the assumes. `checkify` checks the inputs you actually ran. Each can
only speak about configurations that were sampled; none can say the bug is
*absent*. The wedge answers "is there **any** index, anywhere in the assumed
region, that goes out of bounds" — over a region no enumeration covers.

## The evidence question

**Does the bug class exist in real, maintained, public JAX code at a rate that
matters?**

Unknown — and unknown to anyone, which is itself informative, since the tooling
that would answer it is what we propose to build. The priors run both ways and
neither is worth much: the failure is invisible, so its absence from issue
trackers means nothing (you do not file a bug you never noticed); but this code is
written by careful people who compare against analytical results, and wrong
physics tends to surface eventually as wrong physics even when the mechanism is
never named. Both stories fit everything currently known. **That is why this gets
an experiment rather than an argument.**

## The experiment

**N = 5** substantial public JAX codebases. Trace the examples and core kernels,
run the Stage-1 wedge, and classify every in-bounds obligation it cannot discharge.

Candidates, named now, before any of them has been looked at:

| # | Codebase | Why it is in the set |
|---|---|---|
| 1 | **jax-md** | Neighbour lists — the canonical suspect. Buffer overflow on neighbour-list capacity is a *known* pattern here, which makes it the best test of whether the wedge finds what is already known to exist |
| 2 | **jax-cfd** | Stencil and halo exchange — indexing over a mesh with boundary handling |
| 3 | **diffrax** | Adaptive steppers, dense interpolation, buffer indexing over step history |
| 4 | **numpyro** | Different shape: indexing driven by plate/sample structure rather than geometry |
| 5 | **Optimistix** or **lineax** | Solver internals; a fifth that is not a PDE code, to avoid five draws from one distribution |

The corpus is being built anyway for the primitive census (`founding.md`, Stage 1)
and as the differential-testing bed. **The experiment is nearly free** — the census
plus a classification step. We are not proposing work to test the claim; we are
proposing to *read* work already scheduled.

Every obligation the wedge cannot discharge lands in exactly one bucket:

| Bucket | Meaning |
|---|---|
| **Real** | A reachable index, in-region, that goes out of bounds. Confirmed by a checked witness (§2.2 of the architecture) — a concrete input that clamps under `jit` |
| **Spurious** | The wedge cannot prove in-bounds, but no witness exists. Our imprecision, a work item, not a finding |
| **Guarded** | Out-of-bounds is prevented by something outside the traced region — a caller's check, a config invariant. A finding about the *harness*, not the code |
| **Unknown** | ⊤ in the chain, or timeout. Reported as coverage, claimed as nothing |
| **Intentional** | Out-of-range *by design inside the traced region*: sentinel padding or clamp-then-discard, where the author relies on the clamp being harmless. Identified by dataflow — shared-predicate nullification or sentinel provenance (`design/census-interrogation.md` §4). A checked witness at such a site is neither Real nor Spurious |

Only **Real** counts. The classification is done before the count is looked at.

*(The **Intentional** bucket was added 2026-07-17 — before any wedge run —
from the census interrogation's guard anatomy: the census corpus showed
zero pure-mask sites, but the pattern exists (jax-md's neighbor padding
generates out-of-range indices deliberately), and a witness there would
otherwise misclassify as Real. Adding it later, after a run, would be the
renegotiation this document forbids.)*

## The falsifier

Pre-registered. This is the number that kills the claim:

> **Zero Real out-of-bounds across the five codebases falsifies the wedge's value
> claim.** The project stops or re-aims.

And the ambiguous band, fixed now so it cannot be renegotiated later:

| Real findings across N=5 | Reading | Action |
|---|---|---|
| **0** | **Falsified.** The bug class does not occur at a rate that matters in maintained public JAX. The counterfactual methods are evidently sufficient | Stop, or re-aim at a different bug class — the inductive-invariant story (Stage 2) is a *different* value claim and would need its own model and its own falsifier, written before its own experiment |
| **1–2** | **Weak.** Consistent with the class being real and rare. Rare is not worthless — but it is not the wedge's claim, and it does not justify the wedge as the sequencing choice | Re-aim. Publish the finding; do not build the wedge out on this basis |
| **≥ 3, in ≥ 2 codebases** | **Supported.** The class is real, it is distributed rather than one project's mistake, and no existing tool found it first | Proceed as `founding.md` scopes it |

The "≥ 2 codebases" clause is load-bearing: three findings in one project is a
fact about that project, not about JAX.

**Two anti-rationalisations, also pre-registered**, because these are the moves we
will want to make on the day:

- **Spurious findings do not count, however many there are.** A wedge that reports
  a hundred unprovable obligations and zero witnesses has demonstrated our
  imprecision, not their bugs. Only a checked witness counts. If the temptation on
  the day is to argue that the spurious pile "shows the risk," that is the
  temptation this line exists to refuse.
- **"But the bugs are in the code people write, not the libraries"** is not a
  permitted escape from a zero. It may even be true — library code is the most
  reviewed code in the ecosystem. But it was true *before* the experiment, and it
  was not stated as a scope limit *before* the experiment, so it cannot be
  discovered afterwards as a reason the result does not count. If the claim is
  about user code rather than library code, that is a **different experiment with
  a different corpus**, and it must be pre-registered as such — before it runs.

## Why this is pre-registered

`DOCUMENTATION_ARCHITECTURE.md` §10.8 requires a **positive control**: a harness
claimed to license the removal of work ships with a mutation that *must* come back
red, because a harness that cannot fail proves nothing and vacuity is invisible
from the inside.

The whole architecture exists because a green verdict that says more than it
earned is the defect. **A value model without a pre-registered falsifier is that
same defect one level up** — a claim that can absorb any evidence, held by people
who cannot see from the inside that it has stopped being a claim. Zero findings
would get explained. Spurious ones would get counted. The explanations would be
*good*; that is what makes it dangerous.

This document applies its epistemics to verdicts and had never applied them to
itself. This is §10.8's discipline turned inward — the argument was already
written, it just pointed the wrong way.

The number is 0, the corpus is those five, and both were fixed on 2026-07-16,
before the wedge ran on anything.

## What the census says about this experiment (recorded 2026-07-17, before any wedge run)

The primitive census (`design/primitive-census.md`) and its interrogation
(`design/census-interrogation.md`) were read against this experiment's
assumptions. **The falsifier above stands as written.** This section exists
so that what was known *before* the experiment cannot later be presented as
discovered *after* it — in either direction.

Facts, from traced IR of the census corpus:

- **jax-cfd's canonical stencil path contains zero wedge primitives.** Its
  periodic halos are built by `jnp.pad` — fourteen `concatenate` equations
  and static `slice`s, all shape-checked at trace time. An out-of-bounds
  shift in this style is a trace-time error, not a silent runtime clamp:
  **the style structurally cannot have the bug the wedge detects.**
- **`rem` is coordinate wraparound, not index wraparound.** All seven
  occurrences are jax-md's periodic-space float mod, consumed by
  `add`/`lt`/`ne`/`select_n`; none feeds anything indexable.
- **The wedge's target surface in this corpus is 19 of 1823 equations
  (1%), in 2 of 7 targets** — diffrax's step-history buffers and jax-md's
  neighbor path — and **16 of the 17 non-trivial sites carry explicit
  guards (`select_n`, usually with `lt`, sometimes `min`) in the index
  cone.** Library authors mask their indices by hand, everywhere we
  looked.
- **Guarding converts the failure mode out of the wedge's class.** Where a
  guard clamps an index into range, in-bounds holds by construction; the
  residual bug (a *wrong guard bound*, silently reading the wrong element)
  is in-bounds and therefore outside the property the wedge proves. Whether
  each observed guard is a clamp or a mask was not verified — but a green
  wedge verdict on defended code largely confirms the guard exists, not
  that the physics indexes correctly.
- Symmetry: the gather counts exist because the harness aimed at the value
  model's canonical suspect. Where we looked, not where bugs are.

**The constraint:** this experiment's pre-registered corpus is drawn from
exactly the population the census shows to be style-protected and
hand-guarded. A zero would kill the claim as registered — legitimately,
since the claim is about maintained public JAX — but part of that zero
would be attributable to *style*, a mechanism this document did not
anticipate when the corpus was fixed.

**The decision this forces, and its deadline:** either proceed on the
registered corpus accepting that reading, or re-register the experiment
toward the arm the census cannot see — research code, via the interception
harness — **before the wedge runs on anything**. The anti-rationalisation
clause above forbids discovering "the bugs are in user code" after a zero;
this section, dated before any run, is the one legitimate window in which
that re-registration can happen. The choice is the maintainer's; the census
only says the window is now.

A separately-registered second experiment over the guard sites now exists:
`design/guard-experiment.md`. It is an **input** to this decision, not a
resolution of it, and it runs nothing until the decision is made.
